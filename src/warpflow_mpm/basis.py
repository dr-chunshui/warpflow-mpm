"""Shape (basis) functions and their gradients — Warp device functions.

Faithful port of Karamelo's src/basis_functions.h (linear, quadratic-spline,
cubic-spline, bernstein-quadratic), including the ntype boundary-node
corrections that keep B-splines partition-of-unity at domain edges.

`r` is (x_particle - x_node)/cellsize, signed. `ntype` per axis is computed
once at grid setup from the node's global index:
    linear:            0
    cubic/quadratic:   min(2, g) - min(ng-1-g, 2)
    bernstein:         g % 2 (parity)
"""

import numpy as np
import warp as wp

F64 = wp.float64
I32 = wp.int32

BASIS_LINEAR = 0
BASIS_QUADRATIC = 1
BASIS_CUBIC = 2
BASIS_BERNSTEIN = 3

# stencil nodes per axis: linear -> 2, bernstein -> 3, quadratic/cubic -> 4
BASIS_STENCIL = {BASIS_LINEAR: 2, BASIS_QUADRATIC: 4, BASIS_CUBIC: 4,
                 BASIS_BERNSTEIN: 3}
# node lattice spacing factor vs cellsize: bernstein grids use h/2
BASIS_NODE_SPACING = {BASIS_LINEAR: 1.0, BASIS_QUADRATIC: 1.0,
                      BASIS_CUBIC: 1.0, BASIS_BERNSTEIN: 0.5}


@wp.func
def n_linear(r: F64, ntype: I32) -> F64:
    ra = wp.abs(r)
    if ra >= F64(1.0):
        return F64(0.0)
    return F64(1.0) - ra


@wp.func
def dn_linear(r: F64, ntype: I32, inv_h: F64) -> F64:
    if r >= F64(1.0) or r <= F64(-1.0) or r == F64(0.0):
        return F64(0.0)
    if r > F64(0.0):
        return -inv_h
    return inv_h


@wp.func
def n_cubic(r: F64, ntype: I32) -> F64:
    if r >= F64(1.0) and r < F64(2.0):
        if ntype == 1:
            return F64(0.0)
        return ((F64(-1.0) / F64(6.0) * r + F64(1.0)) * r - F64(2.0)) * r + F64(4.0) / F64(3.0)
    if r >= F64(0.0) and r < F64(1.0):
        if ntype == -2:
            return (F64(1.0) / F64(6.0) * r * r - F64(1.0)) * r + F64(1.0)
        if ntype == 2:
            return F64(1.0)
        if ntype == 1:
            return (F64(1.0) / F64(3.0) * r - F64(1.0)) * r * r + F64(2.0) / F64(3.0)
        return (F64(0.5) * r - F64(1.0)) * r * r + F64(2.0) / F64(3.0)
    if r >= F64(-1.0) and r < F64(0.0):
        if ntype == 2:
            return (F64(-1.0) / F64(6.0) * r * r + F64(1.0)) * r + F64(1.0)
        if ntype == -1:
            return (F64(-1.0) / F64(3.0) * r - F64(1.0)) * r * r + F64(2.0) / F64(3.0)
        return (F64(-0.5) * r - F64(1.0)) * r * r + F64(2.0) / F64(3.0)
    if r >= F64(-2.0) and r < F64(-1.0):
        return ((F64(1.0) / F64(6.0) * r + F64(1.0)) * r + F64(2.0)) * r + F64(4.0) / F64(3.0)
    return F64(0.0)


@wp.func
def dn_cubic(r: F64, ntype: I32, inv_h: F64) -> F64:
    if r >= F64(1.0) and r < F64(2.0):
        if ntype == 1:
            return -inv_h
        return inv_h * ((F64(-0.5) * r + F64(2.0)) * r - F64(2.0))
    if r >= F64(0.0) and r < F64(1.0):
        if ntype == -2:
            return inv_h * (F64(0.5) * r * r - F64(1.0))
        if ntype == 2:
            return inv_h
        if ntype == 1:
            return inv_h * r * (r - F64(2.0))
        return inv_h * (F64(3.0) / F64(2.0) * r - F64(2.0)) * r
    if r >= F64(-1.0) and r < F64(0.0):
        if ntype == 2:
            return inv_h * (F64(-0.5) * r * r + F64(1.0))
        if ntype == -1:
            return inv_h * (-r - F64(2.0)) * r
        return inv_h * (F64(-3.0) / F64(2.0) * r - F64(2.0)) * r
    if r >= F64(-2.0) and r < F64(-1.0):
        return inv_h * ((F64(0.5) * r + F64(2.0)) * r + F64(2.0))
    return F64(0.0)


@wp.func
def n_quadratic(r: F64, ntype: I32) -> F64:
    if ntype == 0:
        if r >= F64(0.5) and r < F64(1.5):
            return (F64(0.5) * r - F64(1.5)) * r + F64(1.125)
        if r >= F64(-0.5) and r < F64(0.5):
            return -r * r + F64(0.75)
        if r >= F64(-1.5) and r < F64(0.5):
            return (F64(0.5) * r + F64(1.5)) * r + F64(1.125)
        return F64(0.0)
    if ntype == -2:
        if r >= F64(0.0) and r < F64(0.5):
            return F64(1.0) - r
        if r >= F64(0.5) and r < F64(1.5):
            return (F64(0.5) * r - F64(1.5)) * r + F64(1.125)
        return F64(0.0)
    if ntype == -1:
        if r >= F64(-1.0) and r < F64(-0.5):
            return F64(1.0) + r
        if r >= F64(-0.5) and r < F64(0.5):
            return -r * r + F64(0.75)
        if r >= F64(0.5) and r < F64(1.5):
            return (F64(0.5) * r - F64(1.5)) * r + F64(1.125)
        return F64(0.0)
    if ntype == 1:
        if r >= F64(-1.5) and r < F64(-0.5):
            return (F64(0.5) * r + F64(1.5)) * r + F64(1.125)
        if r >= F64(-0.5) and r < F64(0.5):
            return -r * r + F64(0.75)
        if r >= F64(0.5) and r < F64(1.0):
            return F64(1.0) - r
        return F64(0.0)
    # ntype == 2
    if r >= F64(-1.5) and r < F64(-0.5):
        return (F64(0.5) * r + F64(1.5)) * r + F64(1.125)
    if r >= F64(-0.5) and r <= F64(0.0):
        return F64(1.0) + r
    return F64(0.0)


@wp.func
def dn_quadratic(r: F64, ntype: I32, inv_h: F64) -> F64:
    if ntype == 0:
        if r >= F64(0.5) and r < F64(1.5):
            return inv_h * (r - F64(1.5))
        if r >= F64(-0.5) and r < F64(0.5):
            return -F64(2.0) * inv_h * r
        if r >= F64(-1.5) and r < F64(0.5):
            return inv_h * (r + F64(1.5))
        return F64(0.0)
    if ntype == -2:
        if r >= F64(0.0) and r < F64(0.5):
            return -inv_h
        if r >= F64(0.5) and r < F64(1.5):
            return inv_h * (r - F64(1.5))
        return F64(0.0)
    if ntype == -1:
        if r >= F64(-1.0) and r < F64(-0.5):
            return inv_h
        if r >= F64(-0.5) and r < F64(0.5):
            return -F64(2.0) * inv_h * r
        if r >= F64(0.5) and r < F64(1.5):
            return inv_h * (r - F64(1.5))
        return F64(0.0)
    if ntype == 1:
        if r >= F64(-1.5) and r < F64(-0.5):
            return inv_h * (r + F64(1.5))
        if r >= F64(-0.5) and r < F64(0.5):
            return -F64(2.0) * inv_h * r
        if r >= F64(0.5) and r < F64(1.0):
            return -inv_h
        return F64(0.0)
    # ntype == 2
    if r >= F64(-1.5) and r < F64(-0.5):
        return inv_h * (r + F64(1.5))
    if r >= F64(-0.5) and r <= F64(0.0):
        return inv_h
    return F64(0.0)


@wp.func
def n_bernstein(r_: F64, ntype: I32) -> F64:
    r = wp.abs(r_)
    if r >= F64(1.0):
        return F64(0.0)
    if ntype == 1:
        # interior node (odd global index)
        if r >= F64(0.5):
            return F64(0.0)
        return F64(0.5) - F64(2.0) * r * r
    # edge node (even global index)
    return (F64(1.0) - r) * (F64(1.0) - r)


@wp.func
def dn_bernstein(r_signed: F64, ntype: I32, inv_h: F64) -> F64:
    r = wp.abs(r_signed)
    if r >= F64(1.0):
        return F64(0.0)
    if ntype == 1:
        if r > F64(0.5):
            return F64(0.0)
        return -F64(4.0) * r_signed * inv_h
    if r_signed > F64(0.0):
        return -F64(2.0) * (F64(1.0) - r_signed) * inv_h
    return F64(2.0) * (F64(1.0) + r_signed) * inv_h


@wp.func
def basis_eval(kind: I32, r: F64, ntype: I32) -> F64:
    if kind == BASIS_LINEAR:
        return n_linear(r, ntype)
    if kind == BASIS_QUADRATIC:
        return n_quadratic(r, ntype)
    if kind == BASIS_BERNSTEIN:
        return n_bernstein(r, ntype)
    return n_cubic(r, ntype)


@wp.func
def basis_deriv(kind: I32, r: F64, ntype: I32, inv_h: F64) -> F64:
    if kind == BASIS_LINEAR:
        return dn_linear(r, ntype, inv_h)
    if kind == BASIS_QUADRATIC:
        return dn_quadratic(r, ntype, inv_h)
    if kind == BASIS_BERNSTEIN:
        return dn_bernstein(r, ntype, inv_h)
    return dn_cubic(r, ntype, inv_h)


# ---------------------------------------------------------------------------
# Host-side helpers

BASIS_NAME_TO_KIND = {
    "linear": BASIS_LINEAR,
    "quadratic-spline": BASIS_QUADRATIC,
    "cubic-spline": BASIS_CUBIC,
    "bernstein": BASIS_BERNSTEIN,
}

SUPPORTED_BASIS = tuple(BASIS_NAME_TO_KIND)


def ntype_from_global(g: int, ng: int) -> int:
    """ntype of a node with global index g on an axis of ng nodes (splines)."""
    return min(2, g) - min(ng - 1 - g, 2)


def ntype_axis(g0: int, count: int, ng: int) -> np.ndarray:
    """ntype values for global indices g0..g0+count-1 on one axis."""
    g = np.arange(g0, g0 + count)
    return np.minimum(2, g) - np.minimum(ng - 1 - g, 2)
