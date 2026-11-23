"""Grid ownership / ghost-rect consistency tests (pure host, no GPU)."""

import numpy as np
import pytest

from warpflow_mpm.grid import axis_ownership, global_node_count

BOXLO, BOXHI = -1.0, 1.0


@pytest.mark.parametrize("L,h", [(2.0, 0.1), (2.0, 0.13), (1.7, 0.05), (1.0, 0.5)])
def test_global_node_count_covers_domain(L, h):
    n = global_node_count(L, h)
    assert n * h > L + 0.5 * h - 1e-9
    assert (n - 1) * h <= L + 0.5 * h


@pytest.mark.parametrize("pg,L,h", [(1, 2.0, 0.1), (2, 2.0, 0.1), (3, 2.0, 0.1),
                                    (4, 2.0, 0.1), (5, 1.7, 0.07), (8, 2.0, 0.13)])
def test_ownership_tiles_axis(pg, L, h):
    """Owned ranges tile the global node grid exactly: no gaps, no overlaps."""
    ng = global_node_count(L, h)
    ranges = [axis_ownership(m, pg, L, h) for m in range(pg)]
    # contiguity
    for m in range(pg - 1):
        assert ranges[m][1] == ranges[m + 1][0], (m, ranges)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == ng
    for lo, hi in ranges:
        assert hi > lo  # every rank owns at least one node


def test_axis_ownership_boundary_node_rule():
    """A node falling exactly on a sub-boundary belongs to the left rank.

    L=2, h=0.5, pg=2: nodes at x=0,0.5,1.0 -> rank 0; x=1.5,2.0 -> rank 1
    (C++ noffsetlo/noffsethi rule: the shared node at x=1.0 goes left).
    """
    lo0, hi0 = axis_ownership(0, 2, 2.0, 0.5)
    lo1, hi1 = axis_ownership(1, 2, 2.0, 0.5)
    assert (lo0, hi0) == (0, 3)
    assert (lo1, hi1) == (3, 5)


def test_rect_pair_consistency():
    """For every neighbor pair, each rank's send rect equals the partner's
    recv rect (same global box) — the property Sendrecv depends on."""
    import warp as wp  # noqa: F401  (ensures warp init before universe import)

    from warpflow_mpm.universe import Universe

    for nprocs, dim in [(2, 2), (4, 2), (8, 2), (2, 3), (8, 3)]:
        # emulate all ranks in-process by constructing Universe repeatedly?
        # Universe reads COMM_WORLD, so instead re-implement the neighbor
        # enumeration used by Grid with pure functions.
        from warpflow_mpm.grid import Grid
        h = 0.1
        boxlo = np.zeros(3)
        boxhi = np.array([1.0, 1.0, 1.0])
        ng = [Grid.global_node_count(boxhi[d] - boxlo[d], h) if hasattr(Grid, "global_node_count")
              else global_node_count(boxhi[d] - boxlo[d], h) for d in range(3)]

        def owned(rank, pg):
            loc = [rank % pg[0], (rank // pg[0]) % pg[1], rank // (pg[0] * pg[1])]
            lo, hi = [], []
            for d in range(3):
                l, hh = axis_ownership(loc[d], pg[d], boxhi[d] - boxlo[d], h)
                lo.append(l)
                hi.append(hh)
            return lo, hi

        pg = [1, 1, 1]
        # same tiling rule as Universe
        import math
        if nprocs == 2:
            pg = [2, 1, 1] if dim == 1 else ([1, 2, 1] if dim == 2 else [1, 1, 2])
        elif nprocs == 4:
            pg = [2, 2, 1]
        elif nprocs == 8:
            pg = [2, 2, 2] if dim == 3 else [4, 2, 1]

        rects = {}
        g = 1
        for rank in range(nprocs):
            my_lo, my_hi = owned(rank, pg)
            pad_lo = [max(0, my_lo[d] - g) for d in range(3)]
            pad_hi = [min(ng[d], my_hi[d] + g) for d in range(3)]
            for off in [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                        for dk in (-1, 0, 1)]:
                if dim == 2 and off[2] != 0:
                    continue
                if dim == 1 and (off[1] != 0 or off[2] != 0):
                    continue
                if off == (0, 0, 0):
                    continue
                # neighbor rank
                loc = [rank % pg[0], (rank // pg[0]) % pg[1], rank // (pg[0] * pg[1])]
                nloc = [loc[d] + off[d] for d in range(3)]
                if any(nloc[d] < 0 or nloc[d] >= pg[d] for d in range(3)):
                    continue
                q = nloc[0] + pg[0] * (nloc[1] + pg[1] * nloc[2])
                q_lo, q_hi = owned(q, pg)
                # send rect: my ghost ∩ q owned
                s_lo, s_hi, sv = [], [], 1
                r_lo, r_hi, rv = [], [], 1
                for d in range(3):
                    di = off[d]
                    if di > 0:
                        lo, hi = my_hi[d], min(pad_hi[d], q_hi[d])
                    elif di < 0:
                        lo, hi = max(pad_lo[d], q_lo[d]), my_lo[d]
                    else:
                        lo, hi = my_lo[d], my_hi[d]
                    s_lo.append(lo)
                    s_hi.append(hi)
                    sv *= max(0, hi - lo)
                    if di > 0:
                        lo, hi = max(my_lo[d], q_lo[d] - g), my_hi[d]
                    elif di < 0:
                        lo, hi = my_lo[d], min(my_hi[d], q_hi[d] + g)
                    else:
                        lo, hi = my_lo[d], my_hi[d]
                    r_lo.append(lo)
                    r_hi.append(hi)
                    rv *= max(0, hi - lo)
                if sv or rv:
                    rects[(rank, q)] = ((tuple(s_lo), tuple(s_hi), sv),
                                        (tuple(r_lo), tuple(r_hi), rv))
        # consistency: rank a's send rect to b == rank b's recv rect from a
        for (a, b), (send, recv) in rects.items():
            assert (b, a) in rects, f"missing reciprocal for ({a},{b})"
            back_send, back_recv = rects[(b, a)]
            assert send == back_recv, (a, b, send, back_recv)
            assert recv == back_send, (a, b, recv, back_send)
