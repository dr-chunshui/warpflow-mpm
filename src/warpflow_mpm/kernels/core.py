"""Core Warp compute kernels for explicit UL-MPM (USL/MUSL).

All kernels are float64 (wp.vec3d / wp.mat33d) to match the C++ Eigen code.
Numeric literals inside kernels must be wrapped with wp.float64(...).

Grid addressing: each rank owns a dense padded node block of shape
(npad0, npad1, npad2) = (n0+2g, n1+2g, n2+2g) flattened row-major:
    local_index = i*npad1*npad2 + j*npad2 + k
with global node index (gi, gj, gk) = g0 + (i, j, k), where g0 is the padded
block's global start (vec3i).  Node positions live in the x0n array, and the
B-spline boundary types in the ntype array.
"""

import warp as wp

F64 = wp.float64
V3 = wp.vec3d
M3 = wp.mat33d

from warpflow_mpm.basis import BASIS_BERNSTEIN, basis_deriv, basis_eval

ZERO3 = wp.vec3d(F64(0.0), F64(0.0), F64(0.0))

# constitutive model kinds (must match solvers.py)
MODEL_LINEAR = 0
MODEL_NEO_HOOKEAN = 1
MODEL_EOS = 2

EOS_LINEAR = 0
EOS_FLUID = 1
EOS_SHOCK = 2

STRENGTH_LINEAR = 0
STRENGTH_FLUID = 1
STRENGTH_PLASTIC = 2
STRENGTH_JOHNSON_COOK = 3


@wp.func
def eye3(s: F64) -> M3:
    return M3(s, F64(0.0), F64(0.0),
              F64(0.0), s, F64(0.0),
              F64(0.0), F64(0.0), s)


@wp.func
def inv3(m: M3) -> M3:
    """3x3 matrix inverse via cofactors (f64)."""
    c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2]
    c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    c10 = m[2][1] * m[0][2] - m[2][2] * m[0][1]
    c11 = m[2][2] * m[0][0] - m[2][0] * m[0][2]
    c12 = m[2][0] * m[0][1] - m[2][1] * m[0][0]
    c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1]
    c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2]
    c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    det = m[0][0] * c00 + m[1][0] * c10 + m[2][0] * c20
    inv_det = F64(1.0) / det
    return M3(c00 * inv_det, c10 * inv_det, c20 * inv_det,
              c01 * inv_det, c11 * inv_det, c21 * inv_det,
              c02 * inv_det, c12 * inv_det, c22 * inv_det)


@wp.func
def polar_rotation(m: M3) -> M3:
    """Rotation factor of the polar decomposition via Newton iteration
    R <- 0.5 (R + R^-T).  The polar factor is unique for det(m) > 0, so this
    matches the SVD-based PolDec in Karamelo's mpm_math.h."""
    R = m
    for _k in range(12):
        RmT = inv3(wp.transpose(R))
        R = F64(0.5) * (R + RmT)
    return R


@wp.func
def deviator(m: M3) -> M3:
    tr = m[0][0] + m[1][1] + m[2][2]
    return m - eye3(tr / F64(3.0))


@wp.func
def frobenius(m: M3) -> F64:
    s = F64(0.0)
    for i in range(3):
        for j in range(3):
            s = s + m[i][j] * m[i][j]
    return wp.sqrt(s)


@wp.func
def smul(s: F64, v: V3) -> V3:
    return V3(s * v[0], s * v[1], s * v[2])


@wp.func
def local_index(gi: wp.int32, gj: wp.int32, gk: wp.int32,
                g0: wp.vec3i, npad: wp.vec3i) -> wp.int32:
    i = gi - g0[0]
    j = gj - g0[1]
    k = gk - g0[2]
    return i * npad[1] * npad[2] + j * npad[2] + k


@wp.func
def stencil_base(kind: wp.int32, fi: F64) -> wp.int32:
    """First stencil node index along one axis.

    linear:    floor(fi)  (2 nodes)
    bernstein: 2*int(fi), even-aligned (3 nodes on the h/2 lattice)
    splines:   floor(fi) - 1  (4 nodes)
    """
    if kind == BASIS_BERNSTEIN:
        i0 = 2 * wp.int32(fi)
        if i0 >= 1:
            if i0 % 2 != 0:
                i0 = i0 - 1
        return i0
    if kind == 0:
        return wp.int32(wp.floor(fi))
    return wp.int32(wp.floor(fi)) - 1


# ---------------------------------------------------------------------------
# Shape-function weights for one particle/node pair.
# Returns (wf, wfd) with wfd already scaled by inv_h (dN/dx).

@wp.func
def weights(kind: wp.int32, dim: wp.int32,
            xp: V3, xn: V3, nt: wp.vec3i, inv_h: F64):
    r = (xp - xn) * inv_h
    s0 = basis_eval(kind, r[0], nt[0])
    s1 = F64(1.0)
    s2 = F64(1.0)
    if dim >= 2:
        s1 = basis_eval(kind, r[1], nt[1])
    if dim == 3:
        s2 = basis_eval(kind, r[2], nt[2])
    wf = s0 * s1 * s2

    d0 = basis_deriv(kind, r[0], nt[0], inv_h)
    d1 = F64(0.0)
    d2 = F64(0.0)
    if dim >= 2:
        d1 = basis_deriv(kind, r[1], nt[1], inv_h)
    if dim == 3:
        d2 = basis_deriv(kind, r[2], nt[2], inv_h)
    wx = d0 * s1 * s2
    wy = F64(0.0)
    wz = F64(0.0)
    if dim >= 2:
        wy = s0 * d1 * s2
    if dim == 3:
        wz = s0 * s1 * d2
    return wf, V3(wx, wy, wz)


# ---------------------------------------------------------------------------
# P2G: mass + momentum scatter (atomic)

@wp.kernel
def p2g_mass_momentum(
    x: wp.array(dtype=V3),
    pv: wp.array(dtype=V3),
    pm: wp.array(dtype=F64),
    gmass: wp.array(dtype=F64),
    gmv: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
    rigid_flag: wp.array(dtype=wp.int32), solid_rigid: wp.int32,
):
    p = wp.tid()
    xp = x[p]
    mp = pm[p]
    vp = pv[p]
    fi = (xp - boxlo) * inv_h
    i0 = stencil_base(kind, fi[0])

    for di in range(4):
        if di < stencil:
            gi = i0 + di
            if gi >= 0 and gi < ng[0]:
                fx = fi[0]
                j0 = stencil_base(kind, fi[1])
                for dj in range(4):
                    if dj < stencil:
                        gj = j0 + dj
                        if gj >= 0 and gj < ng[1]:
                            k0 = stencil_base(kind, fi[2])
                            for dk in range(4):
                                if dk < stencil:
                                    gk = k0 + dk
                                    if gk >= 0 and gk < ng[2]:
                                        li = local_index(gi, gj, gk, g0, npad)
                                        xn = x0n[li]
                                        nt = ntype[li]
                                        wf, wfd = weights(kind, dim, xp, xn, nt, inv_h)
                                        if wf != F64(0.0):
                                            if solid_rigid == 1:
                                                rigid_flag[li] = 1
                                            if solid_rigid == 0 and rigid_flag[li] == 1:
                                                continue
                                            wp.atomic_add(gmass, li, wf * mp)
                                            wp.atomic_add(gmv, li, smul(wf * mp, vp))


# ---------------------------------------------------------------------------
# P2G: momentum-only scatter from a position snapshot (MUSL re-scatter, port
# of Solid::compute_velocity_nodes with start-of-step weights)

@wp.kernel
def p2g_momentum(
    x: wp.array(dtype=V3),
    pv: wp.array(dtype=V3),
    pm: wp.array(dtype=F64),
    gmv: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
    rigid_flag: wp.array(dtype=wp.int32), solid_rigid: wp.int32,
):
    p = wp.tid()
    xp = x[p]
    mp = pm[p]
    vp = pv[p]
    fi = (xp - boxlo) * inv_h
    i0 = stencil_base(kind, fi[0])

    for di in range(4):
        if di < stencil:
            gi = i0 + di
            if gi >= 0 and gi < ng[0]:
                j0 = stencil_base(kind, fi[1])
                for dj in range(4):
                    if dj < stencil:
                        gj = j0 + dj
                        if gj >= 0 and gj < ng[1]:
                            k0 = stencil_base(kind, fi[2])
                            for dk in range(4):
                                if dk < stencil:
                                    gk = k0 + dk
                                    if gk >= 0 and gk < ng[2]:
                                        li = local_index(gi, gj, gk, g0, npad)
                                        xn = x0n[li]
                                        nt = ntype[li]
                                        wf, wfd = weights(kind, dim, xp, xn, nt, inv_h)
                                        if wf != F64(0.0):
                                            if solid_rigid == 0 and rigid_flag[li] == 1:
                                                continue
                                            wp.atomic_add(gmv, li, smul(wf * mp, vp))


# ---------------------------------------------------------------------------
# P2G: internal force (stress divergence) + external body force per particle

@wp.kernel
def p2g_forces(
    x: wp.array(dtype=V3),
    stress: wp.array(dtype=M3),     # UL: Cauchy sigma; TL: vol0 * PK1
    vol: wp.array(dtype=F64),
    mbp: wp.array(dtype=V3),
    gf: wp.array(dtype=V3),
    gmb: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32, is_tl: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
    rigid_flag: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    xp = x[p]
    sig = stress[p]
    volp = vol[p]
    mbpp = mbp[p]
    fi = (xp - boxlo) * inv_h
    i0 = stencil_base(kind, fi[0])

    for di in range(4):
        if di < stencil:
            gi = i0 + di
            if gi >= 0 and gi < ng[0]:
                j0 = stencil_base(kind, fi[1])
                for dj in range(4):
                    if dj < stencil:
                        gj = j0 + dj
                        if gj >= 0 and gj < ng[1]:
                            k0 = stencil_base(kind, fi[2])
                            for dk in range(4):
                                if dk < stencil:
                                    gk = k0 + dk
                                    if gk >= 0 and gk < ng[2]:
                                        li = local_index(gi, gj, gk, g0, npad)
                                        xn = x0n[li]
                                        nt = ntype[li]
                                        wf, wfd = weights(kind, dim, xp, xn, nt, inv_h)
                                        if wf != F64(0.0):
                                            if is_tl == 1:
                                                # f -= vol0PK1 . grad N  (vol0 folded in)
                                                wp.atomic_add(gf, li, wp.mul(sig, wfd) * F64(-1.0))
                                            else:
                                                wp.atomic_add(gf, li, smul(-volp, wp.mul(sig, wfd)))
                                            if rigid_flag[li] == 0:
                                                wp.atomic_add(gmb, li, smul(wf, mbpp))


# ---------------------------------------------------------------------------
# Grid update: PIC velocity from accumulated momentum, then USL velocity

@wp.kernel
def grid_velocity_divide(
    gmv: wp.array(dtype=V3),
    gmass: wp.array(dtype=F64),
    gv: wp.array(dtype=V3),
):
    i = wp.tid()
    m = gmass[i]
    if m > F64(0.0):
        gv[i] = gmv[i] * (F64(1.0) / m)
    else:
        gv[i] = V3(F64(0.0), F64(0.0), F64(0.0))


@wp.kernel
def grid_update_velocities(
    gv: wp.array(dtype=V3),
    gf: wp.array(dtype=V3),
    gmb: wp.array(dtype=V3),
    gmass: wp.array(dtype=F64),
    gv_update: wp.array(dtype=V3),
    dt: F64,
    rigid_flag: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    m = gmass[i]
    if m > F64(0.0):
        inv_m = F64(1.0) / m
        f_tot = gf[i] + gmb[i]
        gv_update[i] = gv[i] + smul(dt * inv_m, f_tot)
    else:
        gv_update[i] = gv[i]
    if rigid_flag[i] == 1:
        # rigid nodes move at the rigid-body velocity (no acceleration)
        gv_update[i] = gv[i]


# ---------------------------------------------------------------------------
# Dirichlet velocity BC on a list of nodes

@wp.kernel
def apply_node_velocity_bc(
    node_list: wp.array(dtype=wp.int32),
    gv: wp.array(dtype=V3),
    gv_update: wp.array(dtype=V3),
    val: V3,
    setx: wp.int32, sety: wp.int32, setz: wp.int32,
):
    t = wp.tid()
    li = node_list[t]
    vu = gv_update[li]
    v = gv[li]
    if setx == 1:
        vu = V3(val[0], vu[1], vu[2])
        v = V3(val[0], v[1], v[2])
    if sety == 1:
        vu = V3(vu[0], val[1], vu[2])
        v = V3(v[0], val[1], v[2])
    if setz == 1:
        vu = V3(vu[0], vu[1], val[2])
        v = V3(v[0], v[1], val[2])
    gv_update[li] = vu
    gv[li] = v


# ---------------------------------------------------------------------------
# Body force: mb[node] += g * mass[node]   (after ghost exchange, consistent)

@wp.kernel
def apply_body_force(
    gmass: wp.array(dtype=F64),
    gmb: wp.array(dtype=V3),
    g: V3,
):
    i = wp.tid()
    wp.atomic_add(gmb, i, smul(gmass[i], g))


# ---------------------------------------------------------------------------
# G2P: gather grid velocities -> particle PIC velocity, acceleration, position

@wp.kernel
def g2p(
    xw: wp.array(dtype=V3),
    x: wp.array(dtype=V3),
    gv: wp.array(dtype=V3),
    gv_update: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
    dt: F64,
    v_pic: wp.array(dtype=V3),
    acc: wp.array(dtype=V3),
    solid_rigid: wp.int32,
):
    p = wp.tid()
    # weights from the start-of-step positions (C++ MUSL semantics);
    # the position update writes to the live x array
    xp = xw[p]
    fi = (xp - boxlo) * inv_h
    i0 = stencil_base(kind, fi[0])

    v_pic_p = V3(F64(0.0), F64(0.0), F64(0.0))
    acc_p = V3(F64(0.0), F64(0.0), F64(0.0))

    for di in range(4):
        if di < stencil:
            gi = i0 + di
            if gi >= 0 and gi < ng[0]:
                j0 = stencil_base(kind, fi[1])
                for dj in range(4):
                    if dj < stencil:
                        gj = j0 + dj
                        if gj >= 0 and gj < ng[1]:
                            k0 = stencil_base(kind, fi[2])
                            for dk in range(4):
                                if dk < stencil:
                                    gk = k0 + dk
                                    if gk >= 0 and gk < ng[2]:
                                        li = local_index(gi, gj, gk, g0, npad)
                                        xn = x0n[li]
                                        nt = ntype[li]
                                        wf, wfd = weights(kind, dim, xp, xn, nt, inv_h)
                                        if wf != F64(0.0):
                                            dv = gv_update[li] - gv[li]
                                            v_pic_p = v_pic_p + smul(wf, gv_update[li])
                                            acc_p = acc_p + smul(wf, dv)

    inv_dt = F64(1.0) / dt
    v_pic[p] = v_pic_p
    if solid_rigid == 1:
        acc[p] = V3(F64(0.0), F64(0.0), F64(0.0))
    else:
        acc[p] = acc_p * inv_dt
    # accumulate positions (C++: x += dt * v_update); weights came from xw
    x[p] = x[p] + smul(dt, v_pic_p)


# ---------------------------------------------------------------------------
# FLIP/PIC blend:  v = (1-alpha) v_pic + alpha (v + dt*a)

@wp.kernel
def advance_particles(
    v: wp.array(dtype=V3),
    a: wp.array(dtype=V3),
    v_pic: wp.array(dtype=V3),
    alpha: F64,
    dt: F64,
):
    p = wp.tid()
    vp = v[p]
    ap = a[p]
    flip_v = vp + smul(dt, ap)
    pic_v = v_pic[p]
    one_m = F64(1.0) - alpha
    v[p] = smul(one_m, pic_v) + smul(alpha, flip_v)


# ---------------------------------------------------------------------------
# Velocity gradient L from grid velocities (USL: gv_update, MUSL: gv)

@wp.kernel
def compute_L(
    x: wp.array(dtype=V3),
    vn: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
    L: wp.array(dtype=M3),
):
    p = wp.tid()
    xp = x[p]
    fi = (xp - boxlo) * inv_h
    i0 = stencil_base(kind, fi[0])

    l00 = F64(0.0)
    l01 = F64(0.0)
    l02 = F64(0.0)
    l10 = F64(0.0)
    l11 = F64(0.0)
    l12 = F64(0.0)
    l20 = F64(0.0)
    l21 = F64(0.0)
    l22 = F64(0.0)

    for di in range(4):
        if di < stencil:
            gi = i0 + di
            if gi >= 0 and gi < ng[0]:
                j0 = stencil_base(kind, fi[1])
                for dj in range(4):
                    if dj < stencil:
                        gj = j0 + dj
                        if gj >= 0 and gj < ng[1]:
                            k0 = stencil_base(kind, fi[2])
                            for dk in range(4):
                                if dk < stencil:
                                    gk = k0 + dk
                                    if gk >= 0 and gk < ng[2]:
                                        li = local_index(gi, gj, gk, g0, npad)
                                        xn = x0n[li]
                                        nt = ntype[li]
                                        wf, wfd = weights(kind, dim, xp, xn, nt, inv_h)
                                        if wf != F64(0.0):
                                            v_n = vn[li]
                                            l00 = l00 + v_n[0] * wfd[0]
                                            l01 = l01 + v_n[0] * wfd[1]
                                            l10 = l10 + v_n[1] * wfd[0]
                                            l11 = l11 + v_n[1] * wfd[1]
                                            if dim == 3:
                                                l02 = l02 + v_n[0] * wfd[2]
                                                l12 = l12 + v_n[1] * wfd[2]
                                                l20 = l20 + v_n[2] * wfd[0]
                                                l21 = l21 + v_n[2] * wfd[1]
                                                l22 = l22 + v_n[2] * wfd[2]

    L[p] = M3(l00, l01, l02, l10, l11, l12, l20, l21, l22)


# ---------------------------------------------------------------------------
# Deformation gradient / volume / density / strain-rate update (UL)

@wp.kernel
def update_F_vol_D(
    L: wp.array(dtype=M3),      # UL: velocity gradient; TL: Fdot
    F: wp.array(dtype=M3),
    J: wp.array(dtype=F64),
    vol0: wp.array(dtype=F64),
    vol: wp.array(dtype=F64),
    rho0: wp.array(dtype=F64),
    rho: wp.array(dtype=F64),
    D: wp.array(dtype=M3),
    R: wp.array(dtype=M3),
    Finv: wp.array(dtype=M3),
    dt: F64,
    is_tl: wp.int32,
):
    p = wp.tid()
    Fp = F[p]
    eye = eye3(F64(1.0))

    if is_tl == 1:
        # TL (port of update_deformation_gradient TL branch):
        # F += dt Fdot; polar decomposition; L = Fdot Finv; D = R^T sym(L) R
        Fdot = L[p]
        Fnew = Fp + wp.mul(Fdot, dt)
        F[p] = Fnew
        Fin = inv3(Fnew)
        Finv[p] = Fin
        Jp = wp.determinant(Fnew)
        J[p] = Jp
        vol[p] = Jp * vol0[p]
        rho[p] = rho0[p] * (F64(1.0) / Jp)
        Rp = polar_rotation(Fnew)
        R[p] = Rp
        Lt = wp.mul(Fdot, Fin)
        sym = F64(0.5) * (Lt + wp.transpose(Lt))
        D[p] = wp.mul(wp.mul(wp.transpose(Rp), sym), Rp)
    else:
        # UL: F = (I + dt L) F; D = sym(L)
        Lp = L[p]
        inc = wp.mul(Lp, dt)
        Fnew = wp.mul(eye + inc, Fp)
        F[p] = Fnew
        Fin = inv3(Fnew)
        Finv[p] = Fin
        Jp = wp.determinant(Fnew)
        J[p] = Jp
        vol[p] = Jp * vol0[p]
        rho[p] = rho0[p] * (F64(1.0) / Jp)
        R[p] = eye3(F64(1.0))
        D[p] = F64(0.5) * (Lp + wp.transpose(Lp))


# ---------------------------------------------------------------------------
# Linear-elastic stress update:  sigma += 2G (dt D) + lam tr(dt D) I

@wp.kernel
def update_stress(
    sigma: wp.array(dtype=M3),
    D: wp.array(dtype=M3),
    strain_el: wp.array(dtype=M3),
    F: wp.array(dtype=M3),
    R: wp.array(dtype=M3),
    Finv: wp.array(dtype=M3),
    vol0: wp.array(dtype=F64),
    vol0PK1: wp.array(dtype=M3),
    ep: wp.array(dtype=F64),        # effective plastic strain
    epdot: wp.array(dtype=F64),     # effective plastic strain rate
    damage: wp.array(dtype=F64),    # damage variable [0,1]
    damage_init: wp.array(dtype=F64),  # accumulated damage init
    model: wp.int32, eos_kind: wp.int32, strength_kind: wp.int32,
    is_tl: wp.int32,
    G: F64, lam: F64, K: F64, cellsize: F64, signal_velocity: F64,
    # EOS params
    eos_rho0: F64, eos_gamma: F64, eos_c0: F64, eos_S: F64,
    eos_Q1: F64, eos_Q2: F64,
    # strength params
    st_yield: F64,        # plastic yield stress
    st_A: F64, st_B: F64, st_n: F64, st_epsdot0: F64, st_C: F64, st_m: F64,
    # damage params (Johnson-Cook): d1 d2 d3
    dmg_d1: F64, dmg_d2: F64, dmg_d3: F64, has_damage: wp.int32,
    dt: F64,
):
    p = wp.tid()

    if model == MODEL_LINEAR:
        # linear elastic (UL and TL): sigma += 2G (dt D) + lam tr(dt D) I
        eps_inc = wp.mul(D[p], dt)
        strain_el[p] = strain_el[p] + eps_inc
        tr = eps_inc[0][0] + eps_inc[1][1] + eps_inc[2][2]
        sig = sigma[p] + wp.mul(eps_inc, F64(2.0) * G) + eye3(lam * tr)
        sigma[p] = sig
        if is_tl == 1:
            # vol0PK1 = vol0 J R sigma R^T Finv^T   (solid.cpp:1276-1278)
            R_s_Rt = wp.mul(wp.mul(R[p], sig), wp.transpose(R[p]))
            tmp = wp.mul(R_s_Rt, wp.transpose(Finv[p]))
            vol0PK1[p] = wp.mul(tmp, vol0[p] * wp.determinant(F[p]))

    elif model == MODEL_NEO_HOOKEAN:
        # port of solid.cpp update_stress NH branch
        Fp = F[p]
        Fin = inv3(Fp)  # Finv array is filled by update_F_vol_D; compute inline
        FinT = wp.transpose(Fin)
        # PK1 = G (F - F^-T) + lam ln(J) F^-T   (solid.cpp:1285)
        PK1 = wp.mul(Fp - FinT, G) + wp.mul(FinT, lam * wp.log(wp.determinant(Fp)))
        if is_tl == 1:
            vol0PK1[p] = wp.mul(PK1, vol0[p])
        sigma[p] = wp.mul(wp.mul(Fp, wp.transpose(PK1)),
                          F64(1.0) / wp.determinant(Fp))
        FT = wp.transpose(Fp)
        strain_el[p] = F64(0.5) * (wp.mul(FT, Fp) - eye3(F64(1.0)))

    else:
        # EOS + strength composite (port of solid.cpp update_stress else branch)
        Dp = D[p]
        Jp = wp.determinant(F[p])
        rho_p = F64(1.0) / Jp  # rho / rho0 = 1/J

        # --- pressure ---
        d_pre = damage[p]
        if eos_kind == EOS_LINEAR:
            pH = K * (F64(1.0) - Jp)
        elif eos_kind == EOS_FLUID:
            mu = rho_p  # rho/rho0 = 1/J
            pH = K * (wp.pow(mu, eos_gamma) - F64(1.0))
        else:
            mu = rho_p - F64(1.0)
            denom = F64(1.0) - (eos_S - F64(1.0)) * mu
            pH = eos_c0 * eos_c0 * mu * (F64(1.0) + mu) / (denom * denom)
            pH = pH  # rho0 factor folded: rho0 * c0^2 ... needs rho0!
            pH = pH * eos_rho0
            if eos_Q1 != F64(0.0) or eos_Q2 != F64(0.0):
                tr_eps = Dp[0][0] + Dp[1][1] + Dp[2][2]
                if tr_eps < F64(0.0):
                    q = (eos_rho0 * rho_p * cellsize
                         * (eos_Q1 * cellsize * tr_eps * tr_eps
                            - eos_Q2 * eos_c0 * wp.sqrt(Jp) * tr_eps))
                    pH = pH + q

        # --- deviatoric stress + plasticity ---
        inc_p = F64(0.0)
        Gd = G
        ys = st_yield
        if strength_kind == STRENGTH_LINEAR:
            sig_dev = deviator(sigma[p]) + wp.mul(deviator(Dp), F64(2.0) * Gd * dt)
        elif strength_kind == STRENGTH_FLUID:
            sig_dev = wp.mul(deviator(Dp), F64(2.0) * Gd)
        elif strength_kind == STRENGTH_PLASTIC:
            s_trial = deviator(sigma[p]) + wp.mul(deviator(Dp), F64(2.0) * Gd * dt)
            J2 = wp.sqrt(F64(1.5)) * frobenius(s_trial)
            sig_dev = s_trial
            if J2 >= ys:
                inc_p = (J2 - ys) / (F64(3.0) * Gd)
                sig_dev = wp.mul(s_trial, ys / J2)
        else:
            # Johnson-Cook
            epsdot_ratio = epdot[p] / st_epsdot0
            if epsdot_ratio < F64(1.0):
                epsdot_ratio = F64(1.0)
            if ep[p] < F64(1.0e-10):
                ys_eff = st_A
            else:
                ys_eff = st_A + st_B * wp.pow(ep[p], st_n)
            if st_C != F64(0.0):
                ys_eff = ys_eff * wp.pow(F64(1.0) + epsdot_ratio, st_C)
            s_trial = sigma[p] + wp.mul(Dp, F64(2.0) * Gd * dt)
            s_trial = deviator(s_trial)
            J2 = wp.sqrt(F64(1.5)) * frobenius(s_trial)
            sig_dev = s_trial
            if J2 >= ys_eff:
                inc_p = (J2 - ys_eff) / (F64(3.0) * Gd)
                sig_dev = wp.mul(s_trial, ys_eff / J2)

        # --- bookkeeping (port of solid.cpp:1319-1329) ---
        ep[p] = ep[p] + inc_p
        tav = F64(1000.0) * cellsize / signal_velocity
        rate = epdot[p] - epdot[p] * dt / tav + inc_p / tav
        if rate < F64(0.0):
            rate = F64(0.0)
        epdot[p] = rate

        # --- final sigma ---
        if d_pre > F64(0.0) and pH < F64(0.0):
            pH = pH * (F64(1.0) - d_pre)
        sig = eye3(-pH) + sig_dev
        sigma[p] = sig

        # strain_el (port of solid.cpp:1357-1361)
        tr_D = Dp[0][0] + Dp[1][1] + Dp[2][2]
        prev_tr = (strain_el[p][0][0] + strain_el[p][1][1] + strain_el[p][2][2])
        strain_el[p] = eye3((dt * tr_D + prev_tr) / F64(3.0)) + wp.mul(sig_dev, F64(1.0) / G)

        if is_tl == 1:
            R_s_Rt = wp.mul(wp.mul(R[p], sig), wp.transpose(R[p]))
            tmp = wp.mul(R_s_Rt, wp.transpose(Finv[p]))
            vol0PK1[p] = wp.mul(tmp, vol0[p] * wp.determinant(F[p]))


# ---------------------------------------------------------------------------
# CFL timestep:  dt_CFL = h / (sqrt((K + 4G/3)/rho) + |v|_inf)

@wp.kernel
def compute_dt_cfl(
    v: wp.array(dtype=V3),
    rho: wp.array(dtype=F64),
    K: F64,
    G: F64,
    inv_h: F64,
    dt_cfl: wp.array(dtype=F64),
):
    p = wp.tid()
    vp = v[p]
    c = wp.sqrt((K + F64(4.0) / F64(3.0) * G) * (F64(1.0) / rho[p]))
    vmax = wp.max(wp.abs(vp[0]), wp.max(wp.abs(vp[1]), wp.abs(vp[2])))
    # dt_CFL = h / (c_p + |v|inf)   (port of solid.cpp:1429)
    wp.atomic_min(dt_cfl, 0, F64(1.0) / ((c + vmax) * inv_h))


# ---------------------------------------------------------------------------
# Migration: compute owning rank of each particle

@wp.kernel
def compute_owner(
    x: wp.array(dtype=V3),
    boxlo: V3,
    inv_len: V3,
    pg: wp.vec3i,
    dim: wp.int32,
    owner: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    xp = x[p]
    fx = (xp[0] - boxlo[0]) * inv_len[0]
    ix = wp.int32(fx)
    if ix < 0:
        ix = 0
    pgx = pg[0]
    if ix >= pgx:
        ix = pgx - 1
    iy = 0
    if dim >= 2:
        fy = (xp[1] - boxlo[1]) * inv_len[1]
        iy = wp.int32(fy)
        if iy < 0:
            iy = 0
        if iy >= pg[1]:
            iy = pg[1] - 1
    iz = 0
    if dim == 3:
        fz = (xp[2] - boxlo[2]) * inv_len[2]
        iz = wp.int32(fz)
        if iz < 0:
            iz = 0
        if iz >= pg[2]:
            iz = pg[2] - 1
    owner[p] = ix + pgx * (iy + pg[1] * iz)


# ---------------------------------------------------------------------------
# Energy diagnostics (host-triggered, small reductions)

@wp.kernel
def kinetic_energy(
    v: wp.array(dtype=V3),
    mass: wp.array(dtype=F64),
    out: wp.array(dtype=F64),
):
    p = wp.tid()
    vp = v[p]
    ke = F64(0.5) * mass[p] * (vp[0] * vp[0] + vp[1] * vp[1] + vp[2] * vp[2])
    wp.atomic_add(out, 0, ke)


@wp.kernel
def strain_energy(
    sigma: wp.array(dtype=M3),
    strain_el: wp.array(dtype=M3),
    vol: wp.array(dtype=F64),
    out: wp.array(dtype=F64),
):
    p = wp.tid()
    s = sigma[p]
    e = strain_el[p]
    # double contraction sigma : eps
    se = (s[0][0] * e[0][0] + s[0][1] * e[0][1] + s[0][2] * e[0][2]
          + s[1][0] * e[1][0] + s[1][1] * e[1][1] + s[1][2] * e[1][2]
          + s[2][0] * e[2][0] + s[2][1] * e[2][1] + s[2][2] * e[2][2])
    wp.atomic_add(out, 0, F64(0.5) * vol[p] * se)


# ---------------------------------------------------------------------------
# APIC transfer (ports of compute_velocity_nodes_APIC /
# compute_rate_deformation_gradient_UL_APIC; C matrix = particle L array)

@wp.kernel
def p2g_mass_momentum_apic(
    xw: wp.array(dtype=V3),
    pos: wp.array(dtype=V3),
    pv: wp.array(dtype=V3),
    pm: wp.array(dtype=F64),
    C: wp.array(dtype=M3),
    gmass: wp.array(dtype=F64),
    gmv: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
):
    p = wp.tid()
    xp = xw[p]
    pp = pos[p]
    mp = pm[p]
    vp = pv[p]
    Cp = C[p]
    fi = (xp - boxlo) * inv_h
    i0 = stencil_base(kind, fi[0])

    for di in range(4):
        if di < stencil:
            gi = i0 + di
            if gi >= 0 and gi < ng[0]:
                j0 = stencil_base(kind, fi[1])
                for dj in range(4):
                    if dj < stencil:
                        gj = j0 + dj
                        if gj >= 0 and gj < ng[1]:
                            k0 = stencil_base(kind, fi[2])
                            for dk in range(4):
                                if dk < stencil:
                                    gk = k0 + dk
                                    if gk >= 0 and gk < ng[2]:
                                        li = local_index(gi, gj, gk, g0, npad)
                                        xn = x0n[li]
                                        nt = ntype[li]
                                        wf, wfd = weights(kind, dim, xp, xn, nt, inv_h)
                                        if wf != F64(0.0):
                                            dx = xn - pp
                                            vaff = vp + wp.mul(Cp, dx)
                                            wp.atomic_add(gmass, li, wf * mp)
                                            wp.atomic_add(gmv, li, smul(wf * mp, vaff))


@wp.kernel
def compute_L_apic(
    xw: wp.array(dtype=V3),
    pos: wp.array(dtype=V3),
    vn: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
    di_factor: F64,
    L: wp.array(dtype=M3),
):
    p = wp.tid()
    xp = xw[p]
    pp = pos[p]
    fi = (xp - boxlo) * inv_h
    i0 = stencil_base(kind, fi[0])

    l00 = F64(0.0)
    l01 = F64(0.0)
    l02 = F64(0.0)
    l10 = F64(0.0)
    l11 = F64(0.0)
    l12 = F64(0.0)
    l20 = F64(0.0)
    l21 = F64(0.0)
    l22 = F64(0.0)

    for di in range(4):
        if di < stencil:
            gi = i0 + di
            if gi >= 0 and gi < ng[0]:
                j0 = stencil_base(kind, fi[1])
                for dj in range(4):
                    if dj < stencil:
                        gj = j0 + dj
                        if gj >= 0 and gj < ng[1]:
                            k0 = stencil_base(kind, fi[2])
                            for dk in range(4):
                                if dk < stencil:
                                    gk = k0 + dk
                                    if gk >= 0 and gk < ng[2]:
                                        li = local_index(gi, gj, gk, g0, npad)
                                        xn = x0n[li]
                                        nt = ntype[li]
                                        wf, wfd = weights(kind, dim, xp, xn, nt, inv_h)
                                        if wf != F64(0.0):
                                            dx = xn - pp
                                            v_n = vn[li]
                                            l00 = l00 + v_n[0] * dx[0] * wf
                                            l01 = l01 + v_n[0] * dx[1] * wf
                                            l02 = l02 + v_n[0] * dx[2] * wf
                                            l10 = l10 + v_n[1] * dx[0] * wf
                                            l11 = l11 + v_n[1] * dx[1] * wf
                                            l12 = l12 + v_n[1] * dx[2] * wf
                                            l20 = l20 + v_n[2] * dx[0] * wf
                                            l21 = l21 + v_n[2] * dx[1] * wf
                                            l22 = l22 + v_n[2] * dx[2] * wf

    L[p] = wp.mul(M3(l00, l01, l02, l10, l11, l12, l20, l21, l22), di_factor)


# ---------------------------------------------------------------------------
# CPDI (updated Lagrangian, 2D) — ports of ULCPDI::compute_grid_weight_...
# style 0 = CPDI-R4 (particle domain edges rp, weights = 0.25 sum of corner
# shape functions), style 1 = CPDI-Q4 (corner particles xpc, quadrature
# weights with a/b coefficients).
#
# One fused kernel scatters mass, momentum, internal force and external body
# force per particle (weights recomputed once).  2D only (as C++ master).

@wp.kernel
def cpdi_p2g(
    x: wp.array(dtype=V3),
    pv: wp.array(dtype=V3),
    pm: wp.array(dtype=F64),
    sigma: wp.array(dtype=M3),
    vol: wp.array(dtype=F64),
    mbp: wp.array(dtype=V3),
    e1: wp.array(dtype=V3),     # R4: particle domain edge vectors
    e2: wp.array(dtype=V3),
    xpc: wp.array(dtype=V3),    # Q4: 4 corner positions (4*ip + ic)
    gmass: wp.array(dtype=F64),
    gmv: wp.array(dtype=V3),
    gf: wp.array(dtype=V3),
    gmb: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32, style: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
):
    p = wp.tid()
    xp = x[p]

    # particle domain corners (2D, nc = 4)
    c0 = V3(F64(0.0), F64(0.0), F64(0.0))
    c1 = V3(F64(0.0), F64(0.0), F64(0.0))
    c2 = V3(F64(0.0), F64(0.0), F64(0.0))
    c3 = V3(F64(0.0), F64(0.0), F64(0.0))
    e1v = V3(F64(0.0), F64(0.0), F64(0.0))
    e2v = V3(F64(0.0), F64(0.0), F64(0.0))
    if style == 0:
        e1v = e1[p]
        e2v = e2[p]
        c0 = xp - e1v - e2v
        c1 = xp + e1v - e2v
        c2 = xp + e1v + e2v
        c3 = xp - e1v + e2v
    else:
        c0 = xpc[4 * p + 0]
        c1 = xpc[4 * p + 1]
        c2 = xpc[4 * p + 2]
        c3 = xpc[4 * p + 3]

    volp = vol[p]
    inv_Vp = F64(1.0) / volp

    # Q4 quadrature coefficients (port of ulcpdi.cpp:305-313)
    aa = F64(0.0)
    bb = F64(0.0)
    alpha_over_Vp = F64(0.0)
    sixVp = F64(0.0)
    if style == 1:
        aa = ((c3[0] - c0[0]) * (c1[1] - c2[1])
              - (c1[0] - c2[0]) * (c3[1] - c0[1]))
        bb = ((c2[0] - c3[0]) * (c0[1] - c1[1])
              - (c0[0] - c1[0]) * (c2[1] - c3[1]))
        alpha_over_Vp = F64(0.0417) * inv_Vp
        sixVp = F64(6.0) * volp

    # window: union of the corner stencils (corners within one cell of xp
    # for moderate deformation; 2 extra nodes of margin per side)
    fi = (xp - boxlo) * inv_h
    i0 = wp.int32(wp.floor(fi[0])) - 2
    j0 = wp.int32(wp.floor(fi[1])) - 2

    for di in range(6):
        gi = i0 + di
        if gi >= 0 and gi < ng[0]:
            for dj in range(6):
                gj = j0 + dj
                if gj >= 0 and gj < ng[1]:
                    k0 = wp.int32(wp.floor(fi[2]))
                    gk = k0
                    if gk >= 0 and gk < ng[2]:
                        li = local_index(gi, gj, gk, g0, npad)
                        xn = x0n[li]
                        nt = ntype[li]

                        # corner shape functions at this node
                        r0 = (c0 - xn) * inv_h
                        r1 = (c1 - xn) * inv_h
                        r2 = (c2 - xn) * inv_h
                        r3 = (c3 - xn) * inv_h
                        w0 = basis_eval(kind, r0[0], nt[0]) * basis_eval(kind, r0[1], nt[1])
                        w1 = basis_eval(kind, r1[0], nt[0]) * basis_eval(kind, r1[1], nt[1])
                        w2 = basis_eval(kind, r2[0], nt[0]) * basis_eval(kind, r2[1], nt[1])
                        w3 = basis_eval(kind, r3[0], nt[0]) * basis_eval(kind, r3[1], nt[1])

                        wf = F64(0.0)
                        wx = F64(0.0)
                        wy = F64(0.0)
                        if style == 0:
                            wf = F64(0.25) * (w0 + w1 + w2 + w3)
                            # gradient (port of ulcpdi.cpp:346-357)
                            wx = (w0 - w2) * (e1v[1] - e2v[1]) + (w1 - w3) * (e1v[1] + e2v[1])
                            wy = (w0 - w2) * (e2v[0] - e1v[0]) - (w1 - w3) * (e1v[0] + e2v[0])
                            wx = wx * inv_Vp
                            wy = wy * inv_Vp
                        else:
                            wf = alpha_over_Vp * ((sixVp - aa - bb) * w0
                                                  + (sixVp - aa + bb) * w1
                                                  + (sixVp + aa + bb) * w2
                                                  + (sixVp + aa - bb) * w3)
                            wx = w0 * (c1[1] - c3[1]) + w1 * (c2[1] - c0[1]) \
                                + w2 * (c3[1] - c1[1]) + w3 * (c0[1] - c2[1])
                            wy = w0 * (c3[0] - c1[0]) + w1 * (c0[0] - c2[0]) \
                                + w2 * (c1[0] - c3[0]) + w3 * (c2[0] - c0[0])
                            wx = wx * F64(0.5) * inv_Vp
                            wy = wy * F64(0.5) * inv_Vp

                        if wf > F64(1.0e-12):
                            wfd = V3(wx, wy, F64(0.0))
                            mp = pm[p]
                            sig = sigma[p]
                            vp = pv[p]
                            mbpp = mbp[p]
                            wp.atomic_add(gmass, li, wf * mp)
                            wp.atomic_add(gmv, li, smul(wf * mp, vp))
                            wp.atomic_add(gf, li, smul(-volp, wp.mul(sig, wfd)))
                            wp.atomic_add(gmb, li, smul(wf, mbpp))


@wp.kernel
def cpdi_q4_g2p(
    x: wp.array(dtype=V3),
    xpc: wp.array(dtype=V3),
    gv: wp.array(dtype=V3),
    gv_update: wp.array(dtype=V3),
    x0n: wp.array(dtype=V3),
    ntype: wp.array(dtype=wp.vec3i),
    kind: wp.int32, stencil: wp.int32, dim: wp.int32,
    boxlo: V3, inv_h: F64,
    g0: wp.vec3i, ng: wp.vec3i, npad: wp.vec3i,
    dt: F64,
    vol: wp.array(dtype=F64),
    v_pic: wp.array(dtype=V3),
    xpc_new: wp.array(dtype=V3),
):
    p = wp.tid()
    xp = x[p]

    c0 = xpc[4 * p + 0]
    c1 = xpc[4 * p + 1]
    c2 = xpc[4 * p + 2]
    c3 = xpc[4 * p + 3]
    volp = vol[p]
    inv_Vp = F64(1.0) / volp
    aa = ((c3[0] - c0[0]) * (c1[1] - c2[1])
          - (c1[0] - c2[0]) * (c3[1] - c0[1]))
    bb = ((c2[0] - c3[0]) * (c0[1] - c1[1])
          - (c0[0] - c1[0]) * (c2[1] - c3[1]))
    alpha_over_Vp = F64(0.0417) * inv_Vp
    sixVp = F64(6.0) * volp

    fi = (xp - boxlo) * inv_h
    i0 = wp.int32(wp.floor(fi[0])) - 2
    j0 = wp.int32(wp.floor(fi[1])) - 2

    v_pic_p = V3(F64(0.0), F64(0.0), F64(0.0))
    vc0 = V3(F64(0.0), F64(0.0), F64(0.0))
    vc1 = V3(F64(0.0), F64(0.0), F64(0.0))
    vc2 = V3(F64(0.0), F64(0.0), F64(0.0))
    vc3 = V3(F64(0.0), F64(0.0), F64(0.0))

    for di in range(6):
        gi = i0 + di
        if gi >= 0 and gi < ng[0]:
            for dj in range(6):
                gj = j0 + dj
                if gj >= 0 and gj < ng[1]:
                    gk = wp.int32(wp.floor(fi[2]))
                    if gk >= 0 and gk < ng[2]:
                        li = local_index(gi, gj, gk, g0, npad)
                        xn = x0n[li]
                        nt = ntype[li]
                        r0 = (c0 - xn) * inv_h
                        r1 = (c1 - xn) * inv_h
                        r2 = (c2 - xn) * inv_h
                        r3 = (c3 - xn) * inv_h
                        w0 = basis_eval(kind, r0[0], nt[0]) * basis_eval(kind, r0[1], nt[1])
                        w1 = basis_eval(kind, r1[0], nt[0]) * basis_eval(kind, r1[1], nt[1])
                        w2 = basis_eval(kind, r2[0], nt[0]) * basis_eval(kind, r2[1], nt[1])
                        w3 = basis_eval(kind, r3[0], nt[0]) * basis_eval(kind, r3[1], nt[1])

                        wf = alpha_over_Vp * ((sixVp - aa - bb) * w0
                                              + (sixVp - aa + bb) * w1
                                              + (sixVp + aa + bb) * w2
                                              + (sixVp + aa - bb) * w3)
                        if wf > F64(1.0e-12):
                            vu = gv_update[li]
                            v_pic_p = v_pic_p + smul(wf, vu)
                            vc0 = vc0 + smul(w0, vu)
                            vc1 = vc1 + smul(w1, vu)
                            vc2 = vc2 + smul(w2, vu)
                            vc3 = vc3 + smul(w3, vu)

    v_pic[p] = v_pic_p
    xpc_new[4 * p + 0] = c0 + smul(dt, vc0)
    xpc_new[4 * p + 1] = c1 + smul(dt, vc1)
    xpc_new[4 * p + 2] = c2 + smul(dt, vc2)
    xpc_new[4 * p + 3] = c3 + smul(dt, vc3)


@wp.kernel
def update_rp_domain(
    F: wp.array(dtype=M3),
    rp0_e1: wp.array(dtype=V3),
    rp0_e2: wp.array(dtype=V3),
    rp_e1: wp.array(dtype=V3),
    rp_e2: wp.array(dtype=V3),
):
    # port of Solid::update_particle_domain (CPDI-R4): rp = F rp0
    p = wp.tid()
    Fp = F[p]
    rp_e1[p] = wp.mul(Fp, rp0_e1[p])
    rp_e2[p] = wp.mul(Fp, rp0_e2[p])


@wp.kernel
def polygon_volume(
    xpc: wp.array(dtype=V3),
    vol0: wp.array(dtype=F64),
    vol: wp.array(dtype=F64),
    J: wp.array(dtype=F64),
):
    # 2D shoelace formula over the 4 corners (port of the vol_cpdi branch)
    p = wp.tid()
    a = xpc[4 * p + 0]
    b = xpc[4 * p + 1]
    c = xpc[4 * p + 2]
    d = xpc[4 * p + 3]
    area = F64(0.5) * (a[0] * b[1] - b[0] * a[1]
                       + b[0] * c[1] - c[0] * b[1]
                       + c[0] * d[1] - d[0] * c[1]
                       + d[0] * a[1] - a[0] * d[1])
    vol[p] = wp.abs(area)
    J[p] = wp.abs(area) * (F64(1.0) / vol0[p])


@wp.kernel
def apply_body_force_per_node(
    gmass: wp.array(dtype=F64),
    gmb: wp.array(dtype=V3),
    g: wp.array(dtype=V3),     # per-node body acceleration
):
    i = wp.tid()
    wp.atomic_add(gmb, i, smul(gmass[i], g[i]))


@wp.kernel
def contact_hertz(
    x1: wp.array(dtype=V3), vol1: wp.array(dtype=F64),
    mbp1: wp.array(dtype=V3),
    x2: wp.array(dtype=V3), vol2: wp.array(dtype=F64),
    mbp2: wp.array(dtype=V3),
    Estar: F64, dt: F64, dim: wp.int32, max_cellsize: F64,
):
    """Hertz contact (port of fix_contact_hertz.cpp)."""
    i, j = wp.tid()
    dx = x2[j] - x1[i]
    if wp.abs(dx[0]) >= max_cellsize or wp.abs(dx[1]) >= max_cellsize:
        return
    if dim == 3 and wp.abs(dx[2]) >= max_cellsize:
        return

    if dim == 2:
        Rp1 = F64(0.5) * wp.sqrt(vol1[i])
        Rp2 = F64(0.5) * wp.sqrt(vol2[j])
    else:
        Rp1 = F64(0.5) * wp.cbrt(vol1[i])
        Rp2 = F64(0.5) * wp.cbrt(vol2[j])
    Rp = Rp1 + Rp2

    if wp.abs(dx[0]) >= Rp or wp.abs(dx[1]) >= Rp:
        return
    if dim == 3 and wp.abs(dx[2]) >= Rp:
        return

    r = wp.length(dx)
    if r >= Rp:
        return
    p = Rp - r  # penetration
    fmag = F64(0.25) * F64(3.14159265358979) * Estar * \
        wp.sqrt(Rp1 * Rp2 / Rp * p * p * p)
    f = wp.mul(dx, fmag * (F64(1.0) / r))
    wp.atomic_add(mbp1, i, f * F64(-1.0))
    wp.atomic_add(mbp2, j, f)


@wp.kernel
def compute_average_velocity(
    v: wp.array(dtype=V3),
    vol: wp.array(dtype=F64),
    out: wp.array(dtype=F64),
):
    p = wp.tid()
    vp = v[p]
    wp.atomic_add(out, 0, wp.length(vp) * vol[p])


@wp.kernel
def compute_max_plastic_strain(
    ep: wp.array(dtype=F64),
    out: wp.array(dtype=F64),
):
    p = wp.tid()
    wp.atomic_max(out, 0, ep[p])
