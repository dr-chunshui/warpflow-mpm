"""Per-rank background grid: node ownership, ghost regions, MPI exchange.

Port of Karamelo's Grid::init node-ownership rules (grid.cpp:68-266) with a
dense padded local block instead of jagged ghost-node lists:

  - global node grid: ng[d] nodes at boxlo + k*h  (tag = ng2*ng1*i + ng2*j + k)
  - axis ownership for rank m (along pg ranks, even subdomain splits):
        lo = ceil(m*L/h) [+1 if exact and m>0]
        hi = ceil((m+1)*L/h) [+1 if exact and m+1<pg]
        last rank: extend while h*(lo+nx-0.5) < min(subhi, boxhi)
  - local padded block: [lo-g, hi+g) clamped to [0, ng) per axis
  - ghost region: padded block minus owned block; exchanged rect-by-rect with
    all (up to 3^dim-1) neighbors; rect = my-ghost ∩ neighbor-owned (and the
    mirrored rect for the receive side).
"""

import math

import numpy as np
import warp as wp

from warpflow_mpm.kernels import ghost as gk

F64 = wp.float64
V3 = wp.vec3d


def global_node_count(L: float, h: float) -> int:
    """Number of global nodes on one axis (port of Grid::init)."""
    n = int(L / h) + 1
    while n * h <= L + 0.5 * h:
        n += 1
    return n


def axis_ownership(m: int, pg: int, L: float, h: float) -> tuple:
    """(lo, hi) global node-index range owned by rank m along one axis.

    hi is exclusive; port of the noffsetlo/noffsethi logic + last-rank
    extension loop in Grid::init.
    """
    offset_lo = m * L / pg
    lo_raw = math.ceil(offset_lo / h)
    lo = lo_raw + 1 if (m > 0 and abs(lo_raw * h - offset_lo) < 1e-12) else lo_raw

    offset_hi = min((m + 1) * L / pg, L)
    hi_raw = math.ceil(offset_hi / h)
    hi = hi_raw + 1 if (m < pg - 1 and abs(hi_raw * h - offset_hi) < 1e-12) else hi_raw

    if m == pg - 1:
        nx = hi - lo
        while h * (lo + nx - 0.5) < offset_hi:
            nx += 1
        hi = lo + nx
    return lo, hi


def _ceil_div(a: int, b: int) -> int:
    return -(a // -b)


class Grid:
    def __init__(self, universe, boxlo, boxhi, cellsize, basis_kind, device):
        self.universe = universe
        self.device = device
        self.dim = universe.dimension
        self.cellsize = cellsize
        self.basis_kind = basis_kind
        self.boxlo = np.asarray(boxlo, dtype=np.float64)  # grid's own origin
        from warpflow_mpm.basis import BASIS_NODE_SPACING
        self.h_nodes = cellsize * BASIS_NODE_SPACING[basis_kind]
        self.ghost_width = 1 if basis_kind == 0 else 2  # linear vs splines
        g = self.ghost_width

        # global node counts
        self.ng = [global_node_count(boxhi[d] - boxlo[d], cellsize)
                   for d in range(3)]
        if self.dim < 2:
            self.ng[1] = 1
        if self.dim < 3:
            self.ng[2] = 1

        # per-axis ownership tables for every rank coordinate
        self._own_lo = []
        self._own_hi = []
        for d in range(3):
            pg = universe.procgrid[d]
            L = boxhi[d] - boxlo[d]
            los, his = [], []
            for m in range(pg):
                lo, hi = axis_ownership(m, pg, L, self.h_nodes)
                los.append(lo)
                his.append(hi)
            self._own_lo.append(los)
            self._own_hi.append(his)

        pg = universe.procgrid
        self.own_lo = [self._own_lo[d][universe.myloc[d]] for d in range(3)]
        self.own_hi = [self._own_hi[d][universe.myloc[d]] for d in range(3)]

        # padded block
        self.pad_lo = [max(0, self.own_lo[d] - g) for d in range(3)]
        self.pad_hi = [min(self.ng[d], self.own_hi[d] + g) for d in range(3)]
        self.npad = [self.pad_hi[d] - self.pad_lo[d] for d in range(3)]
        if self.npad[0] <= 0 or self.npad[1] <= 0 or self.npad[2] <= 0:
            raise RuntimeError(
                f"rank {universe.me}: empty local grid block "
                f"own={self.own_lo}..{self.own_hi}")
        self.nnodes_local = (self.own_hi[0] - self.own_lo[0]) * \
                            (self.own_hi[1] - self.own_lo[1]) * \
                            (self.own_hi[2] - self.own_lo[2])
        self.nnodes_padded = self.npad[0] * self.npad[1] * self.npad[2]

        # strides
        self.s0 = self.npad[1] * self.npad[2]
        self.s1 = self.npad[2]

        # node positions + ntype arrays
        gi = np.arange(self.pad_lo[0], self.pad_hi[0])
        gj = np.arange(self.pad_lo[1], self.pad_hi[1])
        gk = np.arange(self.pad_lo[2], self.pad_hi[2])
        GX, GY, GZ = np.meshgrid(gi, gj, gk, indexing="ij")
        gx = GX.ravel()
        gy = GY.ravel()
        gz = GZ.ravel()
        pos = np.stack([boxlo[0] + self.h_nodes * gx,
                        boxlo[1] + self.h_nodes * gy,
                        boxlo[2] + self.h_nodes * gz], axis=1)
        self.x0n = wp.array(pos, dtype=wp.vec3d, device=device)

        from warpflow_mpm.basis import ntype_axis, BASIS_BERNSTEIN
        if basis_kind == 0:
            ntx = np.zeros(gx.shape[0], dtype=np.int32)
            nty = np.zeros(gx.shape[0], dtype=np.int32)
            ntz = np.zeros(gx.shape[0], dtype=np.int32)
        elif basis_kind == BASIS_BERNSTEIN:
            # C++ Grid::init: ntype = global_index % 2 (parity)
            ntx = gx % 2
            nty = gy % 2
            ntz = gz % 2
        else:
            ntx = ntype_axis(self.pad_lo[0], self.npad[0], self.ng[0])[gx - self.pad_lo[0]]
            nty = ntype_axis(self.pad_lo[1], self.npad[1], self.ng[1])[gy - self.pad_lo[1]]
            ntz = ntype_axis(self.pad_lo[2], self.npad[2], self.ng[2])[gz - self.pad_lo[2]]
        ntype = np.stack([ntx, nty, ntz], axis=1)
        self.ntype = wp.array(ntype, dtype=wp.vec3i, device=device)

        # state arrays
        self.mass = wp.zeros(self.nnodes_padded, dtype=F64, device=device)
        self.mv = wp.zeros(self.nnodes_padded, dtype=V3, device=device)
        self.v = wp.zeros(self.nnodes_padded, dtype=V3, device=device)
        self.v_update = wp.zeros(self.nnodes_padded, dtype=V3, device=device)
        self.f = wp.zeros(self.nnodes_padded, dtype=V3, device=device)
        self.mb = wp.zeros(self.nnodes_padded, dtype=V3, device=device)
        self.rigid_flag = wp.zeros(self.nnodes_padded, dtype=wp.int32,
                                   device=device)

        # ghost exchange rect table: per neighbor rank, (send_rect, recv_rect)
        # send_rect = my-ghost ∩ neighbor-owned; recv_rect = my-owned ∩ neighbor-ghost
        self.rects = self._build_rects(boxlo, boxhi, cellsize)

        self.g0 = wp.vec3i(self.pad_lo[0], self.pad_lo[1], self.pad_lo[2])
        self.ngv = wp.vec3i(self.ng[0], self.ng[1], self.ng[2])
        self.npadv = wp.vec3i(self.npad[0], self.npad[1], self.npad[2])

    # ------------------------------------------------------------------
    def owned_range(self, rank: int) -> tuple:
        """(lo, hi) global node ranges of a rank along each axis."""
        loc = [0, 0, 0]
        pgx, pgy, pgz = self.universe.procgrid
        loc[0] = rank % pgx
        loc[1] = (rank // pgx) % pgy
        loc[2] = rank // (pgx * pgy)
        return ([self._own_lo[d][loc[d]] for d in range(3)],
                [self._own_hi[d][loc[d]] for d in range(3)])

    def _axis_rect(self, d: int, di: int, q_lo: int, q_hi: int, send: bool):
        """Global [lo,hi) span on axis d of one exchange rect.

        send=True:  my ghost region ∩ q-owned (contributions flowing out)
        send=False: my owned region ∩ q-ghost (write-back flowing in)
        di: neighbor offset sign along this axis (-1, 0, +1).
        """
        g = self.ghost_width
        own_lo, own_hi = self.own_lo[d], self.own_hi[d]
        pad_lo, pad_hi = self.pad_lo[d], self.pad_hi[d]
        if send:
            if di > 0:
                lo, hi = own_hi, min(pad_hi, q_hi)
            elif di < 0:
                lo, hi = max(pad_lo, q_lo), own_lo
            else:
                lo, hi = own_lo, own_hi
        else:
            if di > 0:
                lo, hi = max(own_lo, q_lo - g), own_hi
            elif di < 0:
                lo, hi = own_lo, min(own_hi, q_hi + g)
            else:
                lo, hi = own_lo, own_hi
        return lo, hi

    def _rect_intersect(self, offset, q_lo, q_hi, send: bool):
        rect_lo = [0, 0, 0]
        rect_hi = [0, 0, 0]
        vol = 1
        for d in range(3):
            lo, hi = self._axis_rect(d, offset[d], q_lo[d], q_hi[d], send)
            rect_lo[d], rect_hi[d] = lo, hi
            vol *= max(0, hi - lo)
        return rect_lo, rect_hi, vol

    def _build_rects(self, boxlo, boxhi, cellsize):
        """For each nonzero-offset neighbor with rank >= 0, compute the pair of
        exchange rects (global index space)."""
        rects = {}
        for off in self.universe.neighbor_offsets:
            q = self.universe.neighbor_rank(off)
            if q < 0:
                continue
            q_lo, q_hi = self.owned_range(q)
            send_lo, send_hi, sv = self._rect_intersect(off, q_lo, q_hi, send=True)
            recv_lo, recv_hi, rv = self._rect_intersect(off, q_lo, q_hi, send=False)
            if sv > 0 or rv > 0:
                rects[q] = (send_lo, send_hi, sv, recv_lo, recv_hi, rv)
        return rects

    def _rect_local(self, rect_lo):
        return (rect_lo[0] - self.pad_lo[0],
                rect_lo[1] - self.pad_lo[1],
                rect_lo[2] - self.pad_lo[2])

    def _launch_pack(self, kernel, arrays, rect_lo, rect_hi, ncomp, device):
        rl = self._rect_local(rect_lo)
        rd = (rect_hi[0] - rect_lo[0], rect_hi[1] - rect_lo[1], rect_hi[2] - rect_lo[2])
        buf = wp.empty(rd[0] * rd[1] * rd[2] * ncomp, dtype=F64, device=device)
        wp.launch(kernel, dim=(rd[0], rd[1], rd[2]),
                  inputs=[*arrays, rl[0], rl[1], rl[2],
                          rd[0], rd[1], rd[2], self.s0, self.s1, buf],
                  device=device)
        return buf

    def _launch_unpack(self, kernel, arrays, rect_lo, rect_hi, buf, ncomp):
        rl = self._rect_local(rect_lo)
        rd = (rect_hi[0] - rect_lo[0], rect_hi[1] - rect_lo[1], rect_hi[2] - rect_lo[2])
        wp.launch(kernel, dim=(rd[0], rd[1], rd[2]),
                  inputs=[*arrays, rl[0], rl[1], rl[2],
                          rd[0], rd[1], rd[2], self.s0, self.s1, buf],
                  device=self.device)

    # ------------------------------------------------------------------
    def exchange_fields_10(self):
        """Accumulate ghost contributions into owned nodes, then write owned
        values back to ghosts (mass, mv, f, mb)."""
        comm = self.universe.comm
        for q in sorted(self.rects):
            send_lo, send_hi, sv, recv_lo, recv_hi, rv = self.rects[q]
            send_buf = None
            if sv > 0:
                send_buf = self._launch_pack(
                    gk.pack10, [self.mass, self.mv, self.f, self.mb],
                    send_lo, send_hi, 10, self.device)
            recv_buf = None
            if rv > 0:
                recv_buf = wp.empty(rv * 10, dtype=F64, device=self.device)
            sb = send_buf.numpy() if sv > 0 else np.empty(0)
            rb = np.empty(rv * 10) if rv > 0 else np.empty(0)
            comm.Sendrecv(sb, dest=q, sendtag=0, recvbuf=rb, source=q, recvtag=0)
            if rv > 0:
                recv_buf = wp.array(rb, dtype=F64, device=self.device)
                self._launch_unpack(gk.unpack10_add,
                                    [self.mass, self.mv, self.f, self.mb],
                                    recv_lo, recv_hi, recv_buf, 10)
        # write-back phase
        for q in sorted(self.rects):
            send_lo, send_hi, sv, recv_lo, recv_hi, rv = self.rects[q]
            send_buf = None
            if rv > 0:  # I own the recv rect on the other side; send my owned data
                send_buf = self._launch_pack(
                    gk.pack10, [self.mass, self.mv, self.f, self.mb],
                    recv_lo, recv_hi, 10, self.device)
            recv_buf = None
            if sv > 0:
                recv_buf = wp.empty(sv * 10, dtype=F64, device=self.device)
            sb = send_buf.numpy() if rv > 0 else np.empty(0)
            rb = np.empty(sv * 10) if sv > 0 else np.empty(0)
            comm.Sendrecv(sb, dest=q, sendtag=1, recvbuf=rb, source=q, recvtag=1)
            if sv > 0:
                recv_buf = wp.array(rb, dtype=F64, device=self.device)
                self._launch_unpack(gk.unpack10_write,
                                    [self.mass, self.mv, self.f, self.mb],
                                    send_lo, send_hi, recv_buf, 10)

    def exchange_vu(self):
        """Exchange (v, v_update) so ghost nodes hold owner values."""
        comm = self.universe.comm
        for q in sorted(self.rects):
            send_lo, send_hi, sv, recv_lo, recv_hi, rv = self.rects[q]
            send_buf = None
            if rv > 0:
                send_buf = self._launch_pack(
                    gk.pack_vu, [self.v, self.v_update],
                    recv_lo, recv_hi, 6, self.device)
            recv_buf = None
            if sv > 0:
                recv_buf = wp.empty(sv * 6, dtype=F64, device=self.device)
            sb = send_buf.numpy() if rv > 0 else np.empty(0)
            rb = np.empty(sv * 6) if sv > 0 else np.empty(0)
            comm.Sendrecv(sb, dest=q, sendtag=2, recvbuf=rb, source=q, recvtag=2)
            if sv > 0:
                recv_buf = wp.array(rb, dtype=F64, device=self.device)
                self._launch_unpack(gk.unpack_vu_write,
                                    [self.v, self.v_update],
                                    send_lo, send_hi, recv_buf, 6)

    def exchange_v3(self, field):
        """Accumulate + write-back a single vec3 field (MUSL re-scatter)."""
        comm = self.universe.comm
        for q in sorted(self.rects):
            send_lo, send_hi, sv, recv_lo, recv_hi, rv = self.rects[q]
            send_buf = None
            if sv > 0:
                send_buf = self._launch_pack(gk.pack_v3, [field],
                                             send_lo, send_hi, 3, self.device)
            recv_buf = None
            if rv > 0:
                recv_buf = wp.empty(rv * 3, dtype=F64, device=self.device)
            sb = send_buf.numpy() if sv > 0 else np.empty(0)
            rb = np.empty(rv * 3) if rv > 0 else np.empty(0)
            comm.Sendrecv(sb, dest=q, sendtag=3, recvbuf=rb, source=q, recvtag=3)
            if rv > 0:
                recv_buf = wp.array(rb, dtype=F64, device=self.device)
                self._launch_unpack(gk.unpack_v3_add, [field],
                                    recv_lo, recv_hi, recv_buf, 3)
        for q in sorted(self.rects):
            send_lo, send_hi, sv, recv_lo, recv_hi, rv = self.rects[q]
            send_buf = None
            if rv > 0:
                send_buf = self._launch_pack(gk.pack_v3, [field],
                                             recv_lo, recv_hi, 3, self.device)
            recv_buf = None
            if sv > 0:
                recv_buf = wp.empty(sv * 3, dtype=F64, device=self.device)
            sb = send_buf.numpy() if rv > 0 else np.empty(0)
            rb = np.empty(sv * 3) if sv > 0 else np.empty(0)
            comm.Sendrecv(sb, dest=q, sendtag=4, recvbuf=rb, source=q, recvtag=4)
            if sv > 0:
                recv_buf = wp.array(rb, dtype=F64, device=self.device)
                self._launch_unpack(gk.unpack_v3_write, [field],
                                    send_lo, send_hi, recv_buf, 3)

    # ------------------------------------------------------------------
    def zero_state(self):
        self.mass.zero_()
        self.mv.zero_()
        self.f.zero_()
        self.mb.zero_()
        self.rigid_flag.zero_()

    def zero_v(self):
        self.v.zero_()

    def node_positions_np(self):
        return self.x0n.numpy()

    def select_nodes(self, inside_fn) -> wp.array:
        """Local indices of padded nodes passing inside_fn(pos(n,3)->(n,) bool)."""
        pos = self.x0n.numpy()
        mask = inside_fn(pos)
        idx = np.nonzero(mask)[0].astype(np.int32)
        return wp.array(idx, dtype=wp.int32, device=self.device)
