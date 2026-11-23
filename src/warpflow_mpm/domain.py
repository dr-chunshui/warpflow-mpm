"""Global domain: box, geometric regions, particle population.

Regions mirror Karamelo's region command subset: box/rectangle (full box
volumes), cylinder (axis-aligned with z, or in-plane circle in 2D), sphere.
Region `inside` tests are vectorized with numpy; the same test is available
as a Warp device function in kernels/geom.py for BC/IC kernels.
"""

import numpy as np


def region_inside_np(region, pts: np.ndarray, dimension: int) -> np.ndarray:
    """Vectorized inside-test, pts shape (n, dim+1?) -> (n,) bool.

    pts columns: x, y, (z). Missing axes filled with 0 by caller.
    """
    n = pts.shape[0]
    x = pts[:, 0]
    y = pts[:, 1] if pts.shape[1] > 1 else np.zeros(n)
    z = pts[:, 2] if pts.shape[1] > 2 else np.zeros(n)
    c = np.asarray(region.center + (0.0,) * (3 - len(region.center)))

    t = region.type
    axis = getattr(region, "axis", "z")
    if t in ("box", "rectangle"):
        half = np.asarray(region.size + (0.0,) * (3 - len(region.size)))
        inside = np.ones(n, dtype=bool)
        for d in range(dimension):
            inside &= (pts[:, d] >= c[d] - half[d]) & (pts[:, d] <= c[d] + half[d])
        return inside
    if t == "cylinder":
        # axis-aligned cylinder (axis: x|y|z; z in 2D); 2D: circle in-plane
        # exterior=True selects r >= radius (outside the cylinder)
        outside = getattr(region, "exterior", False)
        if dimension == 2 or axis == "z":
            r = np.hypot(x - c[0], y - c[1])
            if outside:
                inside = r >= region.radius
            else:
                inside = (r >= region.inner_radius) & (r <= region.radius)
            if dimension == 3:
                inside &= np.abs(z - c[2]) <= 0.5 * region.height
        elif axis == "x":
            r = np.hypot(y - c[1], z - c[2])
            band = np.abs(x - c[0]) <= 0.5 * region.height
            inside = (r >= region.radius) if outside else \
                (r >= region.inner_radius) & (r <= region.radius)
            inside = inside & band
        else:  # axis == "y"
            r = np.hypot(x - c[0], z - c[2])
            band = np.abs(y - c[1]) <= 0.5 * region.height
            inside = (r >= region.radius) if outside else \
                (r >= region.inner_radius) & (r <= region.radius)
            inside = inside & band
        return inside
    if t == "sphere":
        if dimension == 2:
            return np.hypot(x - c[0], y - c[1]) <= region.radius
        return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) <= region.radius
    raise ValueError(f"unknown region type '{t}'")


def populate_region(region, cellsize: float, boxlo, boxhi, dimension: int,
                    ppc: int):
    """Generate ALL particles inside `region` on a ppc-per-cell lattice.

    Returns (positions, volumes) for the whole region; the caller filters to
    particles whose owner rank is the local rank (keeps ownership consistent
    with migration: [lo, hi) semantics, no double counting on split planes).
    Volume per particle = (cellsize/ppc)^dim.
    """
    h = cellsize / ppc
    n = [int(np.floor((boxhi[d] - boxlo[d]) / h - 1e-12)) + 1
         for d in range(dimension)]

    axes = [boxlo[d] + h * (np.arange(n[d]) + 0.5) for d in range(dimension)]

    if dimension == 2:
        X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    else:
        X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    mask = region_inside_np(region, pts, dimension)
    pts = pts[mask]
    vol0 = h ** dimension
    return pts, np.full(pts.shape[0], vol0)


def region_limits(region, dimension: int):
    """Axis-aligned bounding box of a region (port of Region::limits).

    Returns ((xlo, ylo, zlo), (xhi, yhi, zhi)); z span 0 for 2D.
    """
    c = np.asarray(region.center + (0.0,) * (3 - len(region.center)))
    t = region.type
    if t in ("box", "rectangle"):
        half = np.asarray(region.size + (0.0,) * (3 - len(region.size)))
        lo, hi = c - half, c + half
    elif t == "cylinder":
        R = region.radius
        lo = np.array([c[0] - R, c[1] - R, c[2] - R])
        hi = np.array([c[0] + R, c[1] + R, c[2] + R])
        if dimension == 3:
            half_h = 0.5 * region.height
            if getattr(region, "axis", "z") == "x":
                lo[0] = c[0] - half_h
                hi[0] = c[0] + half_h
            elif getattr(region, "axis", "z") == "y":
                lo[1] = c[1] - half_h
                hi[1] = c[1] + half_h
            else:
                lo[2] = c[2] - half_h
                hi[2] = c[2] + half_h
        else:
            lo[2] = hi[2] = 0.0
    elif t == "sphere":
        R = region.radius
        lo, hi = c - R, c + R
        if dimension == 2:
            lo[2] = hi[2] = 0.0
    else:
        raise ValueError(f"unknown region type '{t}'")
    if dimension == 2:
        lo[2] = hi[2] = 0.0
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def region_axis(region) -> str:
    return getattr(region, "axis", "z")
