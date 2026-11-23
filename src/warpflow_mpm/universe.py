"""MPI universe: process grid, subdomain decomposition, communication schedule.

Port of Karamelo's universe.cpp (set_proc_grid, tile2d/tile3d, sendnrecv)
and Domain::set_local_box (even splits along each axis).
"""

import math

import numpy as np
from mpi4py import MPI


def _tile2d(p: int) -> tuple:
    """Split p procs into a 2D tile close to square (port of tile2d)."""
    n = int(math.floor(math.sqrt(p)))
    while n > 1 and (p // n) * n != p:
        n -= 1
    return (n, p // n)


def _tile3d(p: int) -> tuple:
    """Split p procs into a 3D tile close to cube (port of tile3d)."""
    n = int(round(p ** (1.0 / 3.0)))
    while n > 1 and (p // n) * n != p:
        n -= 1
    a, bc = _tile2d(p // n)
    return (n, a, bc)


class Universe:
    """MPI communicator state and domain decomposition (port of Universe +
    Domain::set_local_box)."""

    def __init__(self, dimension: int, boxlo, boxhi):
        self.comm = MPI.COMM_WORLD
        self.me = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

        self.dimension = dimension
        self.boxlo = np.asarray(boxlo, dtype=np.float64)
        self.boxhi = np.asarray(boxhi, dtype=np.float64)

        # procgrid[d] = number of procs along axis d; myloc[d] = my coordinate
        self.procgrid = [1, 1, 1]
        self.myloc = [0, 0, 0]
        n = self.nprocs

        if n > 1 and dimension == 1:
            self.procgrid[0] = n
            self.myloc[0] = self.me
        elif n > 1 and dimension == 2:
            lengths = self.boxhi - self.boxlo
            # width = shorter axis index, length = longer axis index
            width, length = (1, 0) if lengths[0] > lengths[1] else (0, 1)
            a, b = _tile2d(n)          # a along width, b along length (C++ convention)
            self.procgrid[width] = a
            self.procgrid[length] = b
            self.myloc[0] = self.me % self.procgrid[0]
            self.myloc[1] = self.me // self.procgrid[0]
        elif n > 1 and dimension == 3:
            lengths = self.boxhi - self.boxlo
            order = np.argsort(lengths)  # ascending: shortest axis first
            tile = _tile3d(n)            # tile[0] shortest axis ... tile[2] longest
            for k in range(3):
                self.procgrid[order[k]] = tile[k]
            pgx, pgy = self.procgrid[0], self.procgrid[1]
            self.myloc[2] = self.me // (pgx * pgy)
            self.myloc[1] = self.me // pgx - self.myloc[2] * pgy
            self.myloc[0] = self.me - (self.myloc[1] + self.myloc[2] * pgy) * pgx

        self._set_local_box()
        self._set_procneigh()
        self._set_sendnrecv()

        # 3x3x3 neighborhood offsets (rank deltas), used for ghost-rect exchange.
        # Includes diagonals: with a padded ghost region, corner ghosts belong to
        # diagonal ranks (C++ sends ghost nodes to any rank whose extended
        # subdomain contains them; the rect decomposition below is equivalent).
        self.neighbor_offsets = self._build_neighbor_offsets()

    # ------------------------------------------------------------------
    def _set_local_box(self):
        """Even split of the box along each decomposed axis."""
        sublo = np.zeros(3)
        subhi = np.zeros(3)
        for d in range(self.dimension):
            length = self.boxhi[d] - self.boxlo[d]
            h = length / self.procgrid[d]
            sublo[d] = self.boxlo[d] + self.myloc[d] * h
            subhi[d] = sublo[d] + h
        self.sublo, self.subhi = sublo, subhi

    def _set_procneigh(self):
        """procneigh[d][s]: rank of neighbor along axis d side s, -1 if none."""
        self.procneigh = [[-1, -1], [-1, -1], [-1, -1]]
        pg = self.procgrid
        for d in range(self.dimension):
            if self.sublo[d] > self.boxlo[d] + 1.0e-12:
                stride = 1 if d == 0 else (pg[0] if d == 1 else pg[0] * pg[1])
                self.procneigh[d][0] = self.me - stride
            if self.subhi[d] < self.boxhi[d] - 1.0e-12:
                stride = 1 if d == 0 else (pg[0] if d == 1 else pg[0] * pg[1])
                self.procneigh[d][1] = self.me + stride

    def _set_sendnrecv(self):
        """Deadlock-free point-to-point schedule over face neighbors.

        Port of Universe::set_proc_grid's even/odd coloring (universe.cpp:188+).
        Entries are [send(1)/recv(0), jproc, step]; used with blocking Send/Recv.
        For Sendrecv-style exchanges the neighbor iteration order alone suffices.
        """
        self.sendnrecv = []
        pg = self.procgrid
        if self.dimension < 2:
            # 1D: alternate direction by parity
            for step in (0, 1):
                if self.me % 2 == step:
                    j = self.procneigh[0][1]
                    if j != -1:
                        self.sendnrecv.append([0, j, step])
                        self.sendnrecv.append([1, j, step])
                else:
                    j = self.procneigh[0][0]
                    if j != -1:
                        self.sendnrecv.append([1, j, step])
                        self.sendnrecv.append([0, j, step])
            return

        steps = []  # list of (axis, side) processed in 2 passes per axis
        for d in range(self.dimension):
            steps.append((d, 1))
            steps.append((d, 0))
        stride = [1, pg[0], pg[0] * pg[1]]

        for step, (d, side) in enumerate(steps):
            jproc = self.procneigh[d][side]
            if jproc == -1:
                continue
            parity = (self.me // stride[d]) % 2 if d > 0 else self.me % 2
            # C++: first pass side=1 handled by even ranks receiving first,
            # side=0 by odd ranks; keep it simple and symmetric with Sendrecv.
            first = (parity == 0) if side == 1 else (parity == 1)
            if first:
                self.sendnrecv.append([0, jproc, step])  # recv first
                self.sendnrecv.append([1, jproc, step])
            else:
                self.sendnrecv.append([1, jproc, step])  # send first
                self.sendnrecv.append([0, jproc, step])

    # ------------------------------------------------------------------
    def _build_neighbor_offsets(self):
        """Canonical list of 3^dim neighbor (di,dj,dk) offsets (excluding self)."""
        dims = range(self.dimension)
        offsets = []
        for di in (-1, 0, 1):
            if self.dimension < 3 and di != 0 and False:
                continue
            for dj in ((-1, 0, 1) if self.dimension >= 2 else (0,)):
                for dk in ((-1, 0, 1) if self.dimension >= 3 else (0,)):
                    if (di, dj, dk) == (0, 0, 0):
                        continue
                    offsets.append((di, dj, dk))
        return offsets

    def neighbor_rank(self, offset) -> int:
        """Rank of the neighbor at (di,dj,dk) offset, or -1 if at domain edge."""
        pgx, pgy, pgz = self.procgrid
        ix, iy, iz = self.myloc
        jx, jy, jz = ix + offset[0], iy + offset[1], iz + offset[2]
        if not (0 <= jx < pgx):
            return -1
        if self.dimension >= 2 and not (0 <= jy < pgy):
            return -1
        if self.dimension >= 3 and not (0 <= jz < pgz):
            return -1
        return jx + pgx * (jy + pgy * jz)

    def owner_rank(self, x: float, y: float, z: float) -> int:
        """O(1) owner-rank lookup (port of Domain::which_CPU_owns_me)."""
        pgx, pgy, pgz = self.procgrid
        ix = int((x - self.boxlo[0]) / (self.boxhi[0] - self.boxlo[0]) * pgx)
        ix = min(max(ix, 0), pgx - 1)
        if self.dimension >= 2:
            iy = int((y - self.boxlo[1]) / (self.boxhi[1] - self.boxlo[1]) * pgy)
            iy = min(max(iy, 0), pgy - 1)
        else:
            iy = 0
        if self.dimension >= 3:
            iz = int((z - self.boxlo[2]) / (self.boxhi[2] - self.boxlo[2]) * pgz)
            iz = min(max(iz, 0), pgz - 1)
        else:
            iz = 0
        return ix + pgx * (iy + pgy * iz)

    def inside_subdomain(self, x: float, y: float, z: float) -> bool:
        return (self.sublo[0] <= x <= self.subhi[0]
                and (self.dimension < 2 or self.sublo[1] <= y <= self.subhi[1])
                and (self.dimension < 3 or self.sublo[2] <= z <= self.subhi[2]))

    # ------------------------------------------------------------------
    def allreduce_min(self, value: float) -> float:
        return self.comm.allreduce(value, op=MPI.MIN)

    def allreduce_sum(self, value: float) -> float:
        return self.comm.allreduce(value, op=MPI.SUM)
