"""Explicit MPM solvers: UL-MPM and TL-MPM (USL / MUSL schemes).

Faithful port of Karamelo's musl.cpp / usl.cpp run loops, ulmpm.cpp and
tlmpm.cpp methods, plus particle migration (ULMPM::exchange_particles).

Design:
  - UL: all solids share one background grid (C++ domain->grid).
  - TL: each solid has its own grid built on the solid's material bounding
    box (C++ Solid::grid for is_TL), weights evaluated at material positions
    x0 and therefore constant in time.
  - All gathers/scatters use start-of-step weight positions (s.xw): a
    snapshot of x for UL (C++ start-of-step weight lists) or x0 for TL.
"""

import numpy as np
import warp as wp

from warpflow_mpm import materials as mat_mod
from warpflow_mpm.basis import BASIS_STENCIL
from warpflow_mpm.kernels import core
from warpflow_mpm.kernels.contact import contact_min_penetration
from warpflow_mpm.kernels.core import (EOS_FLUID, EOS_LINEAR, EOS_SHOCK,
                                       MODEL_EOS, MODEL_LINEAR,
                                       MODEL_NEO_HOOKEAN, STRENGTH_FLUID,
                                       STRENGTH_JOHNSON_COOK, STRENGTH_LINEAR,
                                       STRENGTH_PLASTIC)
from warpflow_mpm.particles import MIG_NCOMP, _mig_pack, _mig_unpack

V3 = wp.vec3d
M3 = wp.mat33d
F64 = wp.float64
I3 = wp.vec3i


def _model_kind(mat) -> int:
    if getattr(mat, "rigid", False):
        return MODEL_LINEAR  # rigid solids skip constitutive updates via mat->rigid checks
    if isinstance(mat, mat_mod.LinearElastic):
        return MODEL_LINEAR
    if isinstance(mat, mat_mod.NeoHookean):
        return MODEL_NEO_HOOKEAN
    if isinstance(mat, mat_mod.EOSStrength):
        return MODEL_EOS
    raise TypeError(f"unsupported material {type(mat)}")


def _eos_kind(mat) -> int:
    eos = getattr(mat, "eos", None)
    if isinstance(eos, mat_mod.EOSLinear):
        return EOS_LINEAR
    if isinstance(eos, mat_mod.EOSFluid):
        return EOS_FLUID
    if isinstance(eos, mat_mod.EOSShock):
        return EOS_SHOCK
    return EOS_LINEAR


def _strength_kind(mat) -> int:
    st = getattr(mat, "strength", None)
    if isinstance(st, mat_mod.StrengthLinear):
        return STRENGTH_LINEAR
    if isinstance(st, mat_mod.StrengthFluid):
        return STRENGTH_FLUID
    if isinstance(st, mat_mod.StrengthPlastic):
        return STRENGTH_PLASTIC
    if isinstance(st, mat_mod.StrengthJohnsonCook):
        return STRENGTH_JOHNSON_COOK
    return STRENGTH_LINEAR


class MPMSolver:
    def __init__(self, universe, grids, solids, bcs, basis_kind, method_cfg):
        """grids: one shared grid (UL) or one per solid (TL)."""
        self.uni = universe
        self.grids = grids
        self.solids = solids
        self.bcs = bcs
        self.kind = basis_kind
        self.stencil = BASIS_STENCIL[basis_kind]
        self.dim = universe.dimension
        self.device = grids[0].device
        self.scheme = method_cfg.scheme
        self.alpha = float(method_cfg.pic_flip)
        self.is_tl = method_cfg.type == "tlmpm"
        self.is_cpdi = method_cfg.type == "ulcpdi"
        self.cpdi_style = 0 if method_cfg.style == "r4" else 1
        self.use_apic = method_cfg.transfer == "apic"
        self.method_type = method_cfg.type

        # (solid, grid) working pairs: UL pairs every solid with the shared
        # grid; TL pairs each solid with its own.
        if self.is_tl:
            self.pairs = list(zip(solids, grids))
        else:
            self.pairs = [(s, grids[0]) for s in solids]
        self.grid = grids[0]

        self.boxlo_v = wp.vec3d(*[float(v) for v in universe.boxlo])
        self.dt_cfl = wp.array([1.0e22], dtype=F64, device=self.device)
        self.bcs.build_velocity_particle_indices(solids, universe)
        self.bcs.apply_initial_stress(solids, universe)
        self.bcs.apply_initial_velocity_nodes(self.grids, universe.dimension)

    # ------------------------------------------------------------------
    def _common_args(self, grid):
        # NOTE: boxlo is the GRID's own origin (TL grids live on the solid's
        # material bbox, which differs from the domain box)
        boxlo_v = wp.vec3d(*[float(v) for v in grid.boxlo[:3]])
        h = grid.cellsize
        return (grid.x0n, grid.ntype, self.kind, self.stencil, self.dim,
                boxlo_v, 1.0 / h, grid.g0, grid.ngv, grid.npadv)

    def _weight_positions(self):
        """Snapshot start-of-step weight positions (UL) / x0 (TL)."""
        for s in self.solids:
            if s.np_local > 0:
                s.xw = s.x0 if self.is_tl else wp.clone(s.x)

    def _p2g(self):
        """Scatter mass + momentum, then forces (per (solid, grid) pair).

        Rigid solids scatter first and stamp the node rigid flags; non-rigid
        solids skip rigid nodes (port of the C++ grid->rigid machinery).
        CPDI solids use the fused corner-quadrature scatter instead."""
        is_tl = 1 if self.is_tl else 0

        if self.is_cpdi:
            for g in self.grids:
                g.zero_state()
            for s, g in self.pairs:
                if s.np_local == 0:
                    continue
                common = self._common_args(g)
                wp.launch(core.cpdi_p2g, dim=s.np_local,
                          inputs=[s.x, s.v, s.mass, s.sigma, s.vol, s.mbp,
                                  s.rp_e1, s.rp_e2, s.xpc,
                                  g.mass, g.mv, g.f, g.mb,
                                  *common[:5], self.cpdi_style, *common[5:]],
                          device=self.device)
            return

        # stamp rigid flags from rigid solids first
        any_rigid = any(getattr(s.mat, "rigid", False) for s in self.solids)
        if any_rigid:
            for s, g in self.pairs:
                if s.np_local == 0 or not getattr(s.mat, "rigid", False):
                    continue
                common = self._common_args(g)
                wp.launch(core.p2g_mass_momentum, dim=s.np_local,
                          inputs=[s.xw, s.v, s.mass, g.mass, g.mv, *common,
                                  g.rigid_flag, 1],
                          device=self.device)

        for s, g in self.pairs:
            if s.np_local == 0:
                continue
            common = self._common_args(g)
            sr = 1 if getattr(s.mat, "rigid", False) else 0
            if self.use_apic:
                pos = s.x0 if self.is_tl else s.x
                wp.launch(core.p2g_mass_momentum_apic, dim=s.np_local,
                          inputs=[s.xw, pos, s.v, s.mass, s.L, g.mass, g.mv,
                                  *common],
                          device=self.device)
            else:
                wp.launch(core.p2g_mass_momentum, dim=s.np_local,
                          inputs=[s.xw, s.v, s.mass, g.mass, g.mv, *common,
                                  g.rigid_flag, sr],
                          device=self.device)
        for s, g in self.pairs:
            if s.np_local == 0:
                continue
            common = self._common_args(g)
            stress = s.vol0PK1 if self.is_tl else s.sigma
            wp.launch(core.p2g_forces, dim=s.np_local,
                      inputs=[s.xw, stress, s.vol, s.mbp, g.f, g.mb,
                              *common[:5], is_tl, *common[5:], g.rigid_flag],
                      device=self.device)

    def _grid_update(self, dt):
        for _, g in self.pairs:
            wp.launch(core.grid_velocity_divide, dim=g.nnodes_padded,
                      inputs=[g.mv, g.mass, g.v], device=self.device)
            wp.launch(core.grid_update_velocities, dim=g.nnodes_padded,
                      inputs=[g.v, g.f, g.mb, g.mass, g.v_update, dt,
                              g.rigid_flag],
                      device=self.device)

    def _g2p(self, dt):
        if self.is_cpdi and self.cpdi_style == 1:
            for s, g in self.pairs:
                if s.np_local == 0:
                    continue
                common = self._common_args(g)
                wp.launch(core.cpdi_q4_g2p, dim=s.np_local,
                          inputs=[s.x, s.xpc, g.v, g.v_update, *common, dt,
                                  s.vol, s.v_pic, s.xpc_new],
                          device=self.device)
                s.xpc, s.xpc_new = s.xpc_new, s.xpc
            return
        for s, g in self.pairs:
            if s.np_local == 0:
                continue
            common = self._common_args(g)
            sr = 1 if getattr(s.mat, "rigid", False) else 0
            wp.launch(core.g2p, dim=s.np_local,
                      inputs=[s.xw, s.x, g.v, g.v_update, *common, dt,
                              s.v_pic, s.a, sr],
                      device=self.device)

    def _advance(self, dt):
        for s in self.solids:
            if s.np_local == 0:
                continue
            wp.launch(core.advance_particles, dim=s.np_local,
                      inputs=[s.v, s.a, s.v_pic, self.alpha, dt],
                      device=self.device)

    def _rescatter(self):
        """MUSL: re-scatter updated particle velocities -> grid v with
        start-of-step weights and start-of-step mass (C++ musl semantics).
        All solids accumulate into their grid's mv before the exchange and
        division (UL: shared grid; TL: one grid per solid)."""
        for g in self.grids:
            g.mv.zero_()
        for s, g in self.pairs:
            if s.np_local == 0:
                continue
            common = self._common_args(g)
            sr = 1 if getattr(s.mat, "rigid", False) else 0
            wp.launch(core.p2g_momentum, dim=s.np_local,
                      inputs=[s.xw, s.v, s.mass, g.mv, *common,
                              g.rigid_flag, sr],
                      device=self.device)
        for g in self.grids:
            g.exchange_v3(g.mv)
            wp.launch(core.grid_velocity_divide, dim=g.nnodes_padded,
                      inputs=[g.mv, g.mass, g.v], device=self.device)

    def _update_stress(self, dt):
        for s, g in self.pairs:
            if s.np_local == 0:
                continue
            if getattr(s.mat, "rigid", False):
                # rigid particles: F stays identity, no constitutive update
                continue
            common = self._common_args(g)
            vn = g.v_update if self.scheme == "usl" else g.v
            if self.use_apic:
                pos = s.x0 if self.is_tl else s.x
                wp.launch(core.compute_L_apic, dim=s.np_local,
                          inputs=[s.xw, pos, vn, *common,
                                  wp.float64(s.di_factor), s.L],
                          device=self.device)
            else:
                wp.launch(core.compute_L, dim=s.np_local,
                          inputs=[s.xw, vn, *common, s.L], device=self.device)
            wp.launch(core.update_F_vol_D, dim=s.np_local,
                      inputs=[s.L, s.F, s.J, s.vol0, s.vol, s.rho0, s.rho,
                              s.D, s.R, s.Finv, dt,
                              1 if self.is_tl else 0],
                      device=self.device)
            self._launch_constitutive(s, g, dt)
            if self.is_cpdi:
                if self.cpdi_style == 0:
                    wp.launch(core.update_rp_domain, dim=s.np_local,
                              inputs=[s.F, s.rp0_e1, s.rp0_e2, s.rp_e1, s.rp_e2],
                              device=self.device)
                else:
                    wp.launch(core.polygon_volume, dim=s.np_local,
                              inputs=[s.xpc, s.vol0, s.vol, s.J],
                              device=self.device)

    def _launch_constitutive(self, s, grid, dt):
        """Dispatch the constitutive update for one solid."""
        mat = s.mat
        model = _model_kind(mat)
        eos_k = _eos_kind(mat) if model == MODEL_EOS else EOS_LINEAR
        st_k = _strength_kind(mat) if model == MODEL_EOS else STRENGTH_LINEAR

        eos_rho0 = float(getattr(getattr(mat, "eos", None), "rho0",
                                 getattr(mat, "rho0", 1.0)))
        eos_gamma = float(getattr(getattr(mat, "eos", None), "gamma", 0.0))
        eos_c0 = float(getattr(getattr(mat, "eos", None), "c0", 0.0))
        eos_S = float(getattr(getattr(mat, "eos", None), "S", 0.0))
        eos_Q1 = float(getattr(getattr(mat, "eos", None), "Q1", 0.0))
        eos_Q2 = float(getattr(getattr(mat, "eos", None), "Q2", 0.0))
        st_yield = float(getattr(getattr(mat, "strength", None),
                                 "yield_stress", 0.0))
        st_A = float(getattr(getattr(mat, "strength", None), "A", 0.0))
        st_B = float(getattr(getattr(mat, "strength", None), "B", 0.0))
        st_n = float(getattr(getattr(mat, "strength", None), "n", 0.0))
        st_epsdot0 = float(getattr(getattr(mat, "strength", None),
                                   "epsdot0", 1.0))
        st_C = float(getattr(getattr(mat, "strength", None), "C", 0.0))
        st_m = float(getattr(getattr(mat, "strength", None), "m", 0.0))

        has_dmg = 1 if getattr(mat, 'damage', None) is not None else 0
        dmg = getattr(mat, 'damage', None)
        wp.launch(core.update_stress, dim=s.np_local,
                  inputs=[s.sigma, s.D, s.strain_el, s.F, s.R, s.Finv,
                          s.vol0, s.vol0PK1, s.ep, s.epdot,
                          s.damage, s.damage_init,
                          model, eos_k, st_k, 1 if self.is_tl else 0,
                          wp.float64(mat.G), wp.float64(getattr(mat, 'lam', mat.K - 2.0 * mat.G / 3.0)),
                          wp.float64(mat.K), wp.float64(grid.cellsize),
                          wp.float64(getattr(mat, "signal_velocity",
                                             mat.p_wave_speed)),
                          wp.float64(eos_rho0), wp.float64(eos_gamma),
                          wp.float64(eos_c0), wp.float64(eos_S),
                          wp.float64(eos_Q1), wp.float64(eos_Q2),
                          wp.float64(st_yield), wp.float64(st_A),
                          wp.float64(st_B), wp.float64(st_n),
                          wp.float64(st_epsdot0), wp.float64(st_C),
                          wp.float64(st_m),
                          wp.float64(getattr(dmg, 'd1', 0.0) if dmg else 0.0),
                          wp.float64(getattr(dmg, 'd2', 0.0) if dmg else 0.0),
                          wp.float64(getattr(dmg, 'd3', 0.0) if dmg else 0.0),
                          has_dmg, dt],
                  device=self.device)

    # ------------------------------------------------------------------
    def compute_dt_cfl(self) -> float:
        self.dt_cfl.fill_(1.0e22)
        for s, g in self.pairs:
            if s.np_local == 0:
                continue
            mat = s.mat
            wp.launch(core.compute_dt_cfl, dim=s.np_local,
                      inputs=[s.v, s.rho, wp.float64(mat.K), wp.float64(mat.G),
                              wp.float64(1.0 / g.cellsize),
                              self.dt_cfl],
                      device=self.device)
        return float(self.dt_cfl.numpy()[0])

    def adapt_dt(self, dt: float, dt_factor: float) -> float:
        if dt is not None:
            return dt
        local = self.compute_dt_cfl()
        return self.uni.allreduce_min(local) * dt_factor

    # ------------------------------------------------------------------
    def exchange_particles(self):
        """Migrate particles that left their subdomain (UL only; TL particles
        never leave their material grid)."""
        if self.is_tl:
            return
        me = self.uni.me
        nprocs = self.uni.nprocs
        comm = self.uni.comm

        inv_len = wp.vec3d(
            self.uni.procgrid[0] / max(self.uni.boxhi[0] - self.uni.boxlo[0], 1e-300),
            self.uni.procgrid[1] / max(self.uni.boxhi[1] - self.uni.boxlo[1], 1e-300),
            self.uni.procgrid[2] / max(self.uni.boxhi[2] - self.uni.boxlo[2], 1e-300),
        )
        pg = I3(*self.uni.procgrid)

        ncol = MIG_NCOMP + 1
        per_solid = []
        any_out = False
        for isolid, s in enumerate(self.solids):
            if s.np_local == 0:
                per_solid.append((isolid, None, None))
                continue
            wp.launch(core.compute_owner, dim=s.np_local,
                      inputs=[s.x, self.boxlo_v, inv_len, pg, self.dim, s.owner],
                      device=self.device)
            owners = s.owner.numpy()
            if (owners != me).any():
                any_out = True
                per_solid.append((isolid, _mig_pack(s), owners))
            else:
                per_solid.append((isolid, None, owners))

        empty = np.empty((0, ncol))
        send_list = [empty for _ in range(nprocs)]
        if any_out:
            for isolid, buf, owners in per_solid:
                if buf is None:
                    continue
                ids = np.full((buf.shape[0], 1), float(isolid))
                buf_ids = np.concatenate([ids, buf], axis=1)
                for q in range(nprocs):
                    if q == me:
                        continue
                    sel = buf_ids[owners == q]
                    if sel.shape[0]:
                        send_list[q] = sel

        recv_list = comm.alltoall([np.ascontiguousarray(b) for b in send_list])

        if not any_out and all(b.shape[0] == 0 for b in recv_list):
            return

        recv_by_solid = {}
        for q, buf in enumerate(recv_list):
            if buf.shape[0] == 0:
                continue
            pts = buf[:, 1:4]
            for d in range(self.dim):
                bad = ((pts[:, d] < self.uni.sublo[d] - 1e-9)
                       | (pts[:, d] > self.uni.subhi[d] + 1e-9))
                if bad.any():
                    raise RuntimeError(
                        f"particle received from CPU {q} does not belong "
                        f"on CPU {me}")
            sids = np.round(buf[:, 0]).astype(int)
            for isolid in np.unique(sids):
                recv_by_solid.setdefault(isolid, []).append(buf[sids == isolid, 1:])

        for isolid, buf, owners in per_solid:
            s = self.solids[isolid]
            rows = recv_by_solid.get(isolid)
            if buf is None and not rows:
                continue
            if buf is None:
                buf = _mig_pack(s)
            if owners is not None:
                kept = buf[owners == me]
            else:
                kept = buf[:0]
            if rows:
                all_buf = np.vstack([kept] + rows)
            else:
                all_buf = kept
            s.rebuild(_mig_unpack(np.ascontiguousarray(all_buf)))

    # ------------------------------------------------------------------
    def step(self, dt: float):
        """One MUSL or USL timestep (port of musl.cpp / usl.cpp)."""
        scheme = self.scheme

        for s in self.solids:
            s.zero_mbp()
        self.bcs.apply_velocity_particles(self.solids)

        # contact (port of fix_contact_min_penetration INITIAL_INTEGRATE)
        solids_by_id = {s.id: s for s in self.solids}
        for _cid, s1n, s2n, mu, ctype in self.bcs.contact_pairs:
            s1 = solids_by_id[s1n]
            s2 = solids_by_id[s2n]
            if s1.np_local == 0 or s2.np_local == 0:
                continue
            max_cs = max(self.pairs[self.solids.index(s1)][1].cellsize,
                         self.pairs[self.solids.index(s2)][1].cellsize)
            if ctype == "hertz":
                m1, m2 = s1.mat, s2.mat
                Estar = 1.0 / ((1 - m1.nu * m1.nu) / m1.E +
                               (1 - m2.nu * m2.nu) / m2.E)
                wp.launch(core.contact_hertz,
                          dim=(s1.np_local, s2.np_local),
                          inputs=[s1.x, s1.vol, s1.mbp,
                                  s2.x, s2.vol, s2.mbp,
                                  wp.float64(Estar), dt, self.dim,
                                  wp.float64(max_cs)],
                          device=self.device)
            else:
                wp.launch(contact_min_penetration,
                          dim=(s1.np_local, s2.np_local),
                          inputs=[s1.x, s1.v, s1.vol, s1.mass, s1.mbp,
                                  s2.x, s2.v, s2.vol, s2.mass, s2.mbp,
                                  wp.float64(mu), dt, self.dim,
                                  wp.float64(max_cs)],
                          device=self.device)

        # start-of-step weight positions
        self._weight_positions()

        # P2G + ghost exchange (mass, momentum, forces, body force)
        for _, g in self.pairs:
            g.zero_state()
        self._p2g()
        self.bcs.apply_force_nodes(self.grids, self.dim)
        self.bcs.apply_body_force(self.grids, sim=self)
        for g in self.grids:
            g.exchange_fields_10()

        # grid update + node BCs
        self._grid_update(dt)
        self.bcs.apply_node_bcs(self.grids)

        # G2P + FLIP/PIC blend
        self._g2p(dt)
        self._advance(dt)
        self.bcs.apply_velocity_particles(self.solids)

        if scheme == "musl":
            self._rescatter()
            self.bcs.apply_node_bcs(self.grids)

        # velocity gradient, F/J/vol, constitutive update
        self._update_stress(dt)

        # particle migration + timestep
        self.exchange_particles()
