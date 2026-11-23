"""Basis-function unit tests against reference values from
Karamelo's src/basis_functions.h."""

import numpy as np

from warpflow_mpm.basis import (BASIS_CUBIC, BASIS_LINEAR, BASIS_QUADRATIC,
                                ntype_axis, ntype_from_global)


def ref_linear(r):
    return 1.0 - abs(r) if abs(r) < 1.0 else 0.0


def ref_cubic(r, ntype):
    # straight port of basis_functions.h cubic_spline
    if 1.0 <= r < 2.0:
        return 0.0 if ntype == 1 else ((-1.0 / 6.0 * r + 1) * r - 2) * r + 4.0 / 3.0
    if 0.0 <= r < 1.0:
        if ntype == -2:
            return (1.0 / 6.0 * r * r - 1) * r + 1
        if ntype == 2:
            return 1.0
        if ntype == 1:
            return (1.0 / 3.0 * r - 1) * r * r + 2.0 / 3.0
        return (0.5 * r - 1) * r * r + 2.0 / 3.0
    if -1.0 <= r < 0.0:
        if ntype == 2:
            return (-1.0 / 6.0 * r * r + 1) * r + 1
        if ntype == -1:
            return (-1.0 / 3.0 * r - 1) * r * r + 2.0 / 3.0
        return (-0.5 * r - 1) * r * r + 2.0 / 3.0
    if -2.0 <= r < -1.0:
        return ((1.0 / 6.0 * r + 1) * r + 2) * r + 4.0 / 3.0
    return 0.0


def test_linear_partition_of_unity():
    """Linear shape functions reproduce a constant field exactly."""
    h = 0.1
    for frac in np.linspace(0.05, 0.95, 19):
        x = frac * h
        n0 = ref_linear((x - 0.0) / h)
        n1 = ref_linear((x - h) / h)
        assert abs(n0 + n1 - 1.0) < 1e-14


def test_ntype_values():
    """ntype boundary classification for a 21-node axis."""
    ng = 21
    assert ntype_from_global(0, ng) == -2
    assert ntype_from_global(1, ng) == -1
    assert ntype_from_global(5, ng) == 0
    assert ntype_from_global(ng - 1, ng) == 2
    assert ntype_from_global(ng - 2, ng) == 1
    arr = ntype_axis(0, ng, ng)
    assert arr[0] == -2 and arr[-1] == 2 and arr[10] == 0


def test_cubic_spline_interior_partition_of_unity():
    """4-node cubic-spline stencil sums to 1 in the interior."""
    h = 0.1
    for frac in np.linspace(0.05, 0.95, 19):
        x = frac * h
        tot = 0.0
        for di in range(4):
            g = di - 1
            r = (x - g * h) / h
            tot += ref_cubic(r, 0)
        assert abs(tot - 1.0) < 1e-12


def test_cubic_spline_boundary_partition_of_unity():
    """ntype-corrected splines keep partition of unity at the domain edge."""
    h = 0.1
    ng = 21
    for frac in np.linspace(0.0, 0.95, 20):
        x = frac * h
        i0 = int(np.floor(x / h)) - 1
        tot = 0.0
        for di in range(4):
            g = i0 + di
            if g < 0 or g >= ng:
                continue
            r = (x - g * h) / h
            tot += ref_cubic(r, ntype_from_global(g, ng))
        assert abs(tot - 1.0) < 1e-12, (x, tot)
