"""Ghost-node pack/unpack kernels for MPI exchange.

Each exchanged region ("rect") is a dense sub-block of the padded local grid,
launched as a 3D grid with dims (rdi, rdj, rdk) and an offset (r0i, r0j, r0k)
in padded-local coordinates.  wp.tid() is unpacked (warp 1.19 requires tuple
unpacking for multi-dimensional launches).

Layouts (f64 per node):
  - 10-float: [mass, mvx, mvy, mvz, fx, fy, fz, mbx, mby, mbz]
  - 6-float:  [vx, vy, vz, vux, vuy, vuz]
  - 3-float:  [vx, vy, vz] (single vec3 field)
"""

import warp as wp

F64 = wp.float64
V3 = wp.vec3d


@wp.kernel
def pack10(
    gmass: wp.array(dtype=F64),
    gmv: wp.array(dtype=V3),
    gf: wp.array(dtype=V3),
    gmb: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 10 * (ti * rdj * rdk + tj * rdk + tk)
    buf[b + 0] = gmass[li]
    mv = gmv[li]
    buf[b + 1] = mv[0]
    buf[b + 2] = mv[1]
    buf[b + 3] = mv[2]
    f = gf[li]
    buf[b + 4] = f[0]
    buf[b + 5] = f[1]
    buf[b + 6] = f[2]
    mb = gmb[li]
    buf[b + 7] = mb[0]
    buf[b + 8] = mb[1]
    buf[b + 9] = mb[2]


@wp.kernel
def unpack10_add(
    gmass: wp.array(dtype=F64),
    gmv: wp.array(dtype=V3),
    gf: wp.array(dtype=V3),
    gmb: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 10 * (ti * rdj * rdk + tj * rdk + tk)
    gmass[li] = gmass[li] + buf[b + 0]
    mv = gmv[li]
    gmv[li] = V3(mv[0] + buf[b + 1], mv[1] + buf[b + 2], mv[2] + buf[b + 3])
    f = gf[li]
    gf[li] = V3(f[0] + buf[b + 4], f[1] + buf[b + 5], f[2] + buf[b + 6])
    mb = gmb[li]
    gmb[li] = V3(mb[0] + buf[b + 7], mb[1] + buf[b + 8], mb[2] + buf[b + 9])


@wp.kernel
def unpack10_write(
    gmass: wp.array(dtype=F64),
    gmv: wp.array(dtype=V3),
    gf: wp.array(dtype=V3),
    gmb: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 10 * (ti * rdj * rdk + tj * rdk + tk)
    gmass[li] = buf[b + 0]
    gmv[li] = V3(buf[b + 1], buf[b + 2], buf[b + 3])
    gf[li] = V3(buf[b + 4], buf[b + 5], buf[b + 6])
    gmb[li] = V3(buf[b + 7], buf[b + 8], buf[b + 9])


@wp.kernel
def pack_vu(
    gv: wp.array(dtype=V3),
    gv_update: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 6 * (ti * rdj * rdk + tj * rdk + tk)
    v = gv[li]
    buf[b + 0] = v[0]
    buf[b + 1] = v[1]
    buf[b + 2] = v[2]
    vu = gv_update[li]
    buf[b + 3] = vu[0]
    buf[b + 4] = vu[1]
    buf[b + 5] = vu[2]


@wp.kernel
def unpack_vu_write(
    gv: wp.array(dtype=V3),
    gv_update: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 6 * (ti * rdj * rdk + tj * rdk + tk)
    gv[li] = V3(buf[b + 0], buf[b + 1], buf[b + 2])
    gv_update[li] = V3(buf[b + 3], buf[b + 4], buf[b + 5])


@wp.kernel
def pack_v3(
    field: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 3 * (ti * rdj * rdk + tj * rdk + tk)
    v = field[li]
    buf[b + 0] = v[0]
    buf[b + 1] = v[1]
    buf[b + 2] = v[2]


@wp.kernel
def unpack_v3_add(
    field: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 3 * (ti * rdj * rdk + tj * rdk + tk)
    v = field[li]
    field[li] = V3(v[0] + buf[b + 0], v[1] + buf[b + 1], v[2] + buf[b + 2])


@wp.kernel
def unpack_v3_write(
    field: wp.array(dtype=V3),
    r0i: wp.int32, r0j: wp.int32, r0k: wp.int32,
    rdi: wp.int32, rdj: wp.int32, rdk: wp.int32,
    s0: wp.int32, s1: wp.int32,
    buf: wp.array(dtype=F64),
):
    ti, tj, tk = wp.tid()
    li = (r0i + ti) * s0 + (r0j + tj) * s1 + (r0k + tk)
    b = 3 * (ti * rdj * rdk + tj * rdk + tk)
    field[li] = V3(buf[b + 0], buf[b + 1], buf[b + 2])
