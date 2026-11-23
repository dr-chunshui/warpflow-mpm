"""Boundary/initial conditions: node BC lists, body force, initial velocity.

Phase 2 adds force_nodes (total nodal force split over a node group, port of
fix_force_nodes.cpp: mb[in] += F/n over the group's nodes with mass > 0).
"""

import numpy as np
import warp as wp

from warpflow_mpm.domain import region_inside_np
from warpflow_mpm.kernels import core
from warpflow_mpm.kernels.core import M3

V3 = wp.vec3d
M3T = wp.mat33d


class _NodeBC:
    def __init__(self, bc_id, region, val):
        self.id = bc_id
        self.region = region
        self.val = val            # (vx, vy, vz) with None = free
        self._lists = {}          # grid id -> wp.array of node indices

    def node_list(self, grid, dimension):
        key = id(grid)
        if key not in self._lists:
            region = self.region

            def inside(pos, region=region):
                return region_inside_np(region, pos, dimension)

            self._lists[key] = grid.select_nodes(inside)
        return self._lists[key]


class _ForceNodesBC:
    """Port of fix_force_nodes.cpp: total force F split equally over the
    n group nodes with mass > 0: mb[in] += F/n."""

    def __init__(self, bc_id, region, total_force):
        self.id = bc_id
        self.region = region
        self.force = total_force  # (fx, fy, fz)
        self._lists = {}

    def node_list(self, grid, dimension):
        key = id(grid)
        if key not in self._lists:
            region = self.region

            def inside(pos, region=region):
                return region_inside_np(region, pos, dimension)

            self._lists[key] = grid.select_nodes(inside)
        return self._lists[key]


class BoundaryConditions:
    """Compiled BC/IC objects applied by the solver."""

    def __init__(self):
        self.velocity_node_bcs = []   # _NodeBC
        self.force_node_bcs = []      # _ForceNodesBC
        self.body_force = None        # (gx, gy, gz) or None
        self.initial_velocity = []    # (region, vec3)
        self.contact_pairs = []       # (id, s1n, s2n, mu, ctype)
        self.velocity_particles = []
        self.initial_stress = []      # (region, s11..s23)
        self.initial_velocity_nodes = []  # _NodeBC  # (id, solid_name, region, vec3, wp.array idx per solid)

    _regions_by_id = {}

    @classmethod
    def build(cls, cfg, universe, grids, solids, regions_by_id):
        bc = cls()
        bc._regions_by_id = dict(regions_by_id)
        for entry in cfg.boundary_conditions:
            if getattr(entry, "type", "") == "velocity_nodes":
                bc.velocity_node_bcs.append(_NodeBC(
                    entry.id, regions_by_id[entry.region],
                    (entry.vx, entry.vy, entry.vz)))
            elif getattr(entry, "type", "") == "force_nodes":
                f = tuple(entry.force) + (0.0,) * (3 - len(entry.force))
                bc.force_node_bcs.append(_ForceNodesBC(
                    entry.id, regions_by_id[entry.region], f))
            elif getattr(entry, "type", "") == "contact":
                bc.contact_pairs.append((entry.id, entry.solid1, entry.solid2,
                                         entry.mu, "min_penetration"))
            elif getattr(entry, "type", "") == "contact_hertz":
                bc.contact_pairs.append((entry.id, entry.solid1, entry.solid2,
                                         0.0, "hertz"))
            elif getattr(entry, "type", "") == "initial_stress":
                bc.initial_stress.append((regions_by_id[entry.region],
                    (entry.s11, entry.s22, entry.s33,
                     entry.s12, entry.s13, entry.s23)))
            elif getattr(entry, "type", "") == "initial_velocity_nodes":
                bc.initial_velocity_nodes.append(_NodeBC(
                    entry.id, regions_by_id[entry.region],
                    (entry.vx, entry.vy, entry.vz)))
            elif getattr(entry, "type", "") == "velocity_particles":
                v = tuple(float(x) for x in entry.v)
                v = v + (0.0,) * (3 - len(v))
                bc.velocity_particles.append(
                    (entry.id, entry.solid, entry.region, v, None))
            elif getattr(entry, "type", "") == "body_force":
                bc.body_force = entry.g
        for entry in cfg.initial_conditions:
            if entry.type in ("initial_velocity", "velocity"):
                bc.initial_velocity.append(
                    (regions_by_id[entry.select], np.array(entry.v)))
        return bc

    def apply_initial_conditions(self, solids, universe):
        for region, vvec in self.initial_velocity:
            for solid in solids:
                if solid.np_local == 0:
                    continue
                pts = solid.x.numpy()
                mask = region_inside_np(region, pts, universe.dimension)
                if mask.any():
                    vnp = solid.v.numpy()
                    vnp[mask] = vvec
                    solid.v = wp.array(vnp, dtype=V3, device=solid.device)

    def apply_node_bcs(self, grids, dimension=3):
        """Dirichlet velocity BCs — call after grid_update_velocities and
        again after the MUSL re-scatter."""
        for bc in self.velocity_node_bcs:
            for grid in grids:
                nl = bc.node_list(grid, dimension)
                if nl.size == 0:
                    continue
                val = bc.val
                v = wp.vec3d(float(val[0] or 0.0), float(val[1] or 0.0),
                             float(val[2] or 0.0))
                sx = 1 if val[0] is not None else 0
                sy = 1 if val[1] is not None else 0
                sz = 1 if val[2] is not None else 0
                wp.launch(core.apply_node_velocity_bc, dim=nl.size,
                          inputs=[nl, grid.v, grid.v_update, v, sx, sy, sz],
                          device=grid.device)

    def apply_force_nodes(self, grids, dimension=3):
        """Apply after P2G, before ghost exchange: mb += F/n over the
        group's nodes with mass > 0 (C++ fix_force_nodes semantics)."""
        for bc in self.force_node_bcs:
            for grid in grids:
                nl = bc.node_list(grid, dimension)
                if nl.size == 0:
                    continue
                mask = grid.mass.numpy()[nl.numpy()] > 0.0
                idx = nl.numpy()[mask]
                if idx.size == 0:
                    continue
                f = np.asarray(bc.force, dtype=np.float64) / idx.size
                mb = grid.mb.numpy()
                mb[idx] += f
                grid.mb = wp.array(mb, dtype=V3, device=grid.device)

    def build_velocity_particle_indices(self, solids, universe):
        """Resolve particle index lists once (group semantics: fixed at
        setup, port of the Karamelo group mask)."""
        for k, (bc_id, sname, region_name, v, _) in enumerate(self.velocity_particles):
            region = self._regions_by_id[region_name]
            for s in solids:
                if s.id != sname or s.np_local == 0:
                    continue
                pts = s.x.numpy()
                mask = region_inside_np(region, pts, universe.dimension)
                idx = np.nonzero(mask)[0].astype(np.int32)
                self.velocity_particles[k] = (bc_id, sname, region_name, v,
                                              wp.array(idx, dtype=wp.int32,
                                                       device=s.device))

    def apply_initial_stress(self, solids, universe):
        """Port of fix_initial_stress: set particle sigma from expressions
        at step 1 (applied once by the solver)."""
        from warpflow_mpm.domain import region_inside_np
        for region, svals in self.initial_stress:
            for s in solids:
                if s.np_local == 0:
                    continue
                pts = s.x0.numpy()
                mask = region_inside_np(region, pts, universe.dimension)
                if not mask.any():
                    continue
                sig = s.sigma.numpy()
                # 2D: only s11, s22, s12
                for pidx in np.nonzero(mask)[0]:
                    sig[pidx, 0, 0] = svals[0]
                    sig[pidx, 1, 1] = svals[1]
                    sig[pidx, 2, 2] = svals[2]
                    sig[pidx, 0, 1] = sig[pidx, 1, 0] = svals[3]
                    sig[pidx, 0, 2] = sig[pidx, 2, 0] = svals[4]
                    sig[pidx, 1, 2] = sig[pidx, 2, 1] = svals[5]
                s.sigma = wp.array(sig, dtype=M3T, device=s.device)

    def apply_initial_velocity_nodes(self, grids, dimension=3):
        """Port of fix_initial_velocity_nodes: set node velocities once."""
        for bc in self.initial_velocity_nodes:
            for grid in grids:
                nl = bc.node_list(grid, dimension)
                if nl.size == 0:
                    continue
                val = bc.val
                v = wp.vec3d(float(val[0] or 0.0), float(val[1] or 0.0),
                             float(val[2] or 0.0))
                sx = 1 if val[0] is not None else 0
                sy = 1 if val[1] is not None else 0
                sz = 1 if val[2] is not None else 0
                wp.launch(core.apply_node_velocity_bc, dim=nl.size,
                          inputs=[nl, grid.v, grid.v_update, v, sx, sy, sz],
                          device=grid.device)

    def apply_velocity_particles(self, solids):
        """Port of fix_velocity_particles: prescribe particle velocities
        (applied at INITIAL_INTEGRATE and POST_ADVANCE_PARTICLES)."""
        for bc_id, sname, _region, v, idx in self.velocity_particles:
            if idx is None or idx.size == 0:
                continue
            for s in solids:
                if s.id != sname or s.np_local == 0:
                    continue
                vnp = s.v.numpy()
                vv = wp.vec3d(*[float(x) for x in v])
                vv_arr = np.tile(np.array([v[0], v[1], v[2]]), (idx.size, 1))
                vnp[idx.numpy()] = vv_arr
                s.v = wp.array(vnp, dtype=V3, device=s.device)

    def apply_body_force(self, grids, sim=None):
        if self.body_force is None:
            return
        if any(isinstance(x, str) for x in self.body_force):
            self._apply_body_force_expr(grids, sim)
            return
        g = wp.vec3d(*[float(x) for x in self.body_force])
        for grid in grids:
            wp.launch(core.apply_body_force, dim=grid.nnodes_padded,
                      inputs=[grid.mass, grid.mb, g], device=grid.device)

    def _apply_body_force_expr(self, grids, sim):
        """Space/time-dependent body force (port of fix_body_force's
        per-node expression evaluation with x0/y0/z0/time)."""
        for grid in grids:
            pos = grid.x0n.numpy()
            arrays = {
                'x0': pos[:, 0], 'y0': pos[:, 1], 'z0': pos[:, 2],
                'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2],
                'time': sim.time,
            }
            gnp = np.zeros((pos.shape[0], 3))
            for d in range(3):
                comp = self.body_force[d]
                if isinstance(comp, str):
                    gnp[:, d] = sim.eval_expr(comp, arrays)
                else:
                    gnp[:, d] = float(comp)
            garr = wp.array(gnp, dtype=V3, device=grid.device)
            wp.launch(core.apply_body_force_per_node,
                      dim=grid.nnodes_padded,
                      inputs=[grid.mass, grid.mb, garr], device=grid.device)
