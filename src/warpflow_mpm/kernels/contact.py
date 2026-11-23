"""Contact: penalty contact between two solids (port of
fix_contact_min_penetration.cpp).

Applied at the start of the step into the particles' mbp (body-force
accumulator), before the P2G scatter.  O(np1 x np2) pair kernel with the
C++ 3-level screening; contact pairs spanning MPI ranks behave exactly like
C++ master (local particles only).
"""

import warp as wp

F64 = wp.float64
V3 = wp.vec3d


@wp.kernel
def contact_min_penetration(
    x1: wp.array(dtype=V3), v1: wp.array(dtype=V3),
    vol1: wp.array(dtype=F64), m1: wp.array(dtype=F64),
    mbp1: wp.array(dtype=V3),
    x2: wp.array(dtype=V3), v2: wp.array(dtype=V3),
    vol2: wp.array(dtype=F64), m2: wp.array(dtype=F64),
    mbp2: wp.array(dtype=V3),
    mu: F64, dt: F64, dim: wp.int32, max_cellsize: F64,
):
    i, j = wp.tid()
    dx = x2[j] - x1[i]

    # extremely gross screening (per-axis, port of the C++ cascade)
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

    # gross screening
    if wp.abs(dx[0]) >= Rp or wp.abs(dx[1]) >= Rp:
        return
    if dim == 3 and wp.abs(dx[2]) >= Rp:
        return

    r = wp.length(dx)
    if r >= Rp:
        return

    inv_r = F64(1.0) / r
    fmag = (m1[i] * m2[j] / ((m1[i] + m2[j]) * dt * dt) * (F64(1.0) - Rp * inv_r))
    f = wp.mul(dx, fmag)

    if mu != F64(0.0):
        dv = v2[j] - v1[i]
        vt = dv - wp.mul(dx, wp.dot(dv, dx) * inv_r * inv_r)
        vtn = wp.length(vt)
        if vtn != F64(0.0):
            vt = wp.mul(vt, F64(1.0) / vtn)
            ffric = mu * fmag * r
            f = f - wp.mul(vt, ffric)

    wp.atomic_add(mbp1, i, f)
    wp.atomic_add(mbp2, j, f * F64(-1.0))
