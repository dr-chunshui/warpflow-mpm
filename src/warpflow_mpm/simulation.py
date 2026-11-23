"""Simulation assembly and run loop."""

import os
import time as pytime

import numpy as np
import warp as wp

from warpflow_mpm import config as cfg_mod
from warpflow_mpm.basis import BASIS_NAME_TO_KIND
from warpflow_mpm.bcs import BoundaryConditions
from warpflow_mpm.dumps import EnergyLog, ParticleDump
from warpflow_mpm.grid import Grid
from warpflow_mpm.kernels import core
from warpflow_mpm.materials import LinearElastic
from warpflow_mpm.particles import Solid

from warpflow_mpm.universe import Universe


def _pick_device(rank: int = 0):
    """cuda:<rank % n_gpu> for multi-GPU nodes, cpu fallback."""
    cuda_devs = [d for d in wp.get_devices() if "cuda" in str(d)]
    if cuda_devs:
        return cuda_devs[rank % len(cuda_devs)]
    return "cpu"


class Simulation:
    def __init__(self, cfg: cfg_mod.Config, log_path="warpflow.log.csv"):
        self.cfg = cfg

        boxlo3 = tuple(cfg.boxlo) + (0.0,) * (3 - cfg.dimension)
        boxhi3 = tuple(cfg.boxhi) + (0.0,) * (3 - cfg.dimension)
        self.universe = Universe(cfg.dimension, boxlo3, boxhi3)
        self.device = _pick_device(self.universe.me)
        if self.universe.me == 0:
            rank_info = (f"warpflow-mpm: {self.universe.nprocs} rank(s), "
                         f"procgrid={self.universe.procgrid[:cfg.dimension]}, "
                         f"device={self.device}")
            print(rank_info)

        self.basis_kind = BASIS_NAME_TO_KIND[cfg.basis]
        self.is_tl = cfg.method.type == "tlmpm"
        regions_by_id = {r.id: r for r in cfg.regions}

        # materials
        self.materials = {}
        from warpflow_mpm import materials as mat_mod
        for m in cfg.materials:
            if m.model == "linear-elastic":
                self.materials[m.id] = mat_mod.LinearElastic.from_E_nu(
                    m.rho, m.E, m.nu)
            elif m.model == "neo-hookean":
                self.materials[m.id] = mat_mod.NeoHookean.from_E_nu(
                    m.rho, m.E, m.nu)
            elif m.model == "rigid":
                self.materials[m.id] = mat_mod.Rigid(rho0=m.rho)
            elif m.model == "eos-strength":
                eos_kind = m.eos.get("type")
                if eos_kind == "linear":
                    eos = mat_mod.EOSLinear(rho0=m.eos["rho0"], K=m.eos["K"])
                elif eos_kind == "fluid":
                    eos = mat_mod.EOSFluid(rho0=m.eos["rho0"], K=m.eos["K"],
                                           gamma=m.eos["gamma"])
                elif eos_kind == "shock":
                    eos = mat_mod.EOSShock(
                        rho0=m.eos["rho0"], K=m.eos["K"], c0=m.eos["c0"],
                        S=m.eos["S"], gamma=m.eos["gamma"], Tr=m.eos["Tr"],
                        cv=m.eos["cv"], Q1=m.eos.get("Q1", 0.0),
                        Q2=m.eos.get("Q2", 0.0))
                else:
                    raise cfg_mod.ConfigError(f"unknown eos type {eos_kind}")
                st_kind = m.strength.get("type")
                if st_kind == "linear":
                    strength = mat_mod.StrengthLinear(G=m.strength["G"])
                elif st_kind == "fluid":
                    strength = mat_mod.StrengthFluid(G=m.strength["G"])
                elif st_kind == "plastic":
                    strength = mat_mod.StrengthPlastic(
                        G=m.strength["G"], yield_stress=m.strength["yield_stress"])
                elif st_kind == "johnson-cook":
                    strength = mat_mod.StrengthJohnsonCook(
                        G=m.strength["G"], A=m.strength["A"], B=m.strength["B"],
                        n=m.strength["n"], epsdot0=m.strength["epsdot0"],
                        C=m.strength["C"], m=m.strength["m"],
                        Tr=m.strength["Tr"], Tm=m.strength["Tm"])
                else:
                    raise cfg_mod.ConfigError(f"unknown strength type {st_kind}")
                self.materials[m.id] = mat_mod.EOSStrength(
                    rho0=eos.rho0, eos=eos, strength=strength)
            else:
                raise cfg_mod.ConfigError(f"unknown material model {m.model}")

        # solids: populate on every rank, keep particles inside local subdomain
        self.solids = []
        solid_regions = []
        for sd in cfg.solids:
            mat = self.materials[sd.material]
            region = regions_by_id[sd.region]
            pts, vols = self._populate(region, sd.ppc)
            rho0 = np.full(pts.shape[0], mat.rho0)
            solid = Solid(sd.id, mat, self.device)
            solid.populate(pts, vols, rho0)
            self.solids.append(solid)
            solid_regions.append(region)
            if self.universe.me == 0:
                print(f"solid '{sd.id}': {pts.shape[0]} particles on this rank")

        if self.is_tl:
            # TLMPM: one grid per solid, built on the solid's material bbox.
            # C++ adjusts solidlo/solidhi to the generated particle extent
            # during populate (solid.cpp:2467+) — replicate: bbox of ALL
            # lattice points inside the region (deterministic per rank).
            from warpflow_mpm.domain import region_limits
            self.grids = []
            for region in solid_regions:
                lo, hi = region_limits(region, cfg.dimension)
                lo3 = np.asarray(lo, dtype=np.float64)
                hi3 = np.asarray(hi, dtype=np.float64)
                self.grids.append(Grid(self.universe, lo3, hi3,
                                       cfg.cellsize, self.basis_kind,
                                       self.device))
        else:
            self.grids = [Grid(self.universe, np.asarray(boxlo3),
                               np.asarray(boxhi3), cfg.cellsize,
                               self.basis_kind, self.device)]
        self.grid = self.grids[0]  # primary grid (UL: the shared grid)

        # BC/IC
        self.bcs = BoundaryConditions.build(cfg, self.universe, self.grids,
                                            self.solids, regions_by_id)
        self.bcs.apply_initial_conditions(self.solids, self.universe)

        # solver
        from warpflow_mpm.solvers import MPMSolver
        self.solver = MPMSolver(self.universe, self.grids, self.solids,
                                self.bcs, self.basis_kind, cfg.method)
        self.bcs._sim = self

        # APIC inertia tensor factor (port of compute_inertia_tensor):
        # linear TL: 16/4 (1 ppc) or 16/3 (2 ppc); cubic: 3; quadratic: 4
        if self.solver.use_apic:
            from warpflow_mpm.basis import BASIS_LINEAR, BASIS_CUBIC, BASIS_QUADRATIC
            for sd, solid in zip(cfg.solids, self.solids):
                if self.basis_kind == BASIS_LINEAR:
                    if sd.ppc == 1:
                        c = 16.0 / 4.0
                    elif sd.ppc == 2:
                        c = 16.0 / 3.0
                    else:
                        raise cfg_mod.ConfigError(
                            "APIC + linear basis supports ppc 1 or 2")
                elif self.basis_kind == BASIS_CUBIC:
                    c = 3.0
                elif self.basis_kind == BASIS_QUADRATIC:
                    c = 4.0
                else:
                    raise cfg_mod.ConfigError(
                        "APIC unsupported for this basis")
                solid.di_factor = c / self.grid.cellsize ** 2
        else:
            for solid in self.solids:
                solid.di_factor = 0.0

        # CPDI particle domain initialization (2D): edges = (h/2, 0), (0, h/2)
        if self.solver.is_cpdi:
            h = cfg.cellsize
            for solid in self.solids:
                n = solid.np_local
                e1 = np.tile(np.array([0.5 * h, 0.0, 0.0]), (n, 1))
                e2 = np.tile(np.array([0.0, 0.5 * h, 0.0]), (n, 1))
                solid.rp0_e1 = wp.array(e1, dtype=wp.vec3d, device=self.device)
                solid.rp0_e2 = wp.array(e2, dtype=wp.vec3d, device=self.device)
                solid.rp_e1 = wp.clone(solid.rp0_e1)
                solid.rp_e2 = wp.clone(solid.rp0_e2)
                x0 = solid.x0.numpy()
                q = np.zeros((4 * n, 3))
                q[0::4] = x0 - e1 - e2
                q[1::4] = x0 + e1 - e2
                q[2::4] = x0 + e1 + e2
                q[3::4] = x0 - e1 + e2
                solid.xpc = wp.array(q, dtype=wp.vec3d, device=self.device)
                solid.xpc_new = wp.zeros(4 * n, dtype=wp.vec3d,
                                         device=self.device)

        # expression variables (ordered evaluation, numpy-backed math)
        self.variables = dict(cfg.variables)
        self._expr_cache = {}

        # output
        self.dump = None
        if cfg.output.particle_dump:
            self.dump = ParticleDump(cfg.output.particle_dump,
                                     cfg.output.fields, self.universe)
        self.elog = EnergyLog(self.universe, log_path)
        # C++ parity (update.cpp): dt starts at 1e-16 when adaptive; the
        # CFL-based dt is computed after the first step via adjust_dt.
        self.dt = cfg.run.dt if cfg.run.dt is not None else 1.0e-16
        self.time = 0.0

    def _populate(self, region, ppc):
        u = self.universe
        return self._populate_impl(region, ppc, u)

    def _populate_impl(self, region, ppc, u):
        from warpflow_mpm.domain import populate_region
        pts, vols = populate_region(
            region, self.cfg.cellsize, u.boxlo, u.boxhi, u.dimension, ppc)
        # keep only particles owned by this rank (consistent with migration)
        if u.nprocs > 1:
            owners = np.array([u.owner_rank(*p[:3]) for p in pts])
            pts = pts[owners == u.me]
            vols = vols[owners == u.me]
        return pts, vols

    def eval_expr(self, expr: str, arrays: dict) -> np.ndarray:
        """Evaluate an expression over per-node/particle arrays.

        Supports numpy math (sin, cos, atan2, sqrt, pi), the config's
        `variables:` (evaluated in order, may reference `time`), and the
        keys of `arrays` (e.g. x, y, x0, y0, time).
        """
        import numpy as _np
        ns = {
            'sin': _np.sin, 'cos': _np.cos, 'tan': _np.tan,
            'sqrt': _np.sqrt, 'atan2': _np.arctan2, 'abs': _np.abs,
            'exp': _np.exp, 'log': _np.log, 'PI': _np.pi, 'pi': _np.pi,
            'min': _np.minimum, 'max': _np.maximum,
        }
        ns.update(arrays)
        # evaluate config variables in order (cache per time step)
        for name, vexpr in self.cfg.variables.items():
            if name in ns:
                continue
            ns[name] = eval(self._compile(vexpr), {'__builtins__': {}}, ns)
        return eval(self._compile(expr), {'__builtins__': {}}, ns)

    def _compile(self, expr: str):
        import re
        key = re.sub(r'time', 'TIMEVAR', expr)  # expressions re-evaluated
        if key not in self._expr_cache:
            e = expr.replace('^', '**')
            self._expr_cache[key] = compile(e, '<expr>', 'eval')
        return self._expr_cache[key]

    def _check_solution(self):
        if not self.cfg.diagnostics:
            return
        u_np = {}
        rows = []
        for s in self.solids:
            if s.np_local == 0:
                continue
            x0 = s.x0.numpy()
            u_np['x'] = x0[:, 0]
            u_np['y'] = x0[:, 1]
            u_np['z'] = x0[:, 2] if x0.shape[1] > 2 else np.zeros(len(x0))
            u_np['x0'] = x0[:, 0]
            u_np['y0'] = x0[:, 1]
            u_np['z0'] = x0[:, 2] if x0.shape[1] > 2 else np.zeros(len(x0))
            u_np['time'] = self.time
            xcur = s.x.numpy()
            vol0 = s.vol0.numpy()
            err = 0.0
            uth2 = 0.0
            for d in range(self.cfg.dimension):
                u_expr = '0'
                for dd in self.cfg.diagnostics:
                    if dd.type == 'check_solution':
                        u_expr = dd.u[d] if d < len(dd.u) else '0'
                u = self.eval_expr(u_expr, u_np)
                du = xcur[:, d] - x0[:, d]
                err += float(np.sum(vol0 * (u - du) ** 2))
                uth2 += float(np.sum(vol0 * u ** 2))
            rel = (err / uth2) ** 0.5 if uth2 > 0 else 0.0
            rows.append(rel)
        rel_l2 = max(rows) if rows else 0.0
        self._last_error = rel_l2
        if self.universe.me == 0:
            print(f"    check_solution: relative L2 error = {rel_l2:.6e}")

    # ------------------------------------------------------------------
    def _energies(self):
        ke = wp.zeros(1, dtype=wp.float64, device=self.device)
        es = wp.zeros(1, dtype=wp.float64, device=self.device)
        for s in self.solids:
            if s.np_local == 0:
                continue
            wp.launch(core.kinetic_energy, dim=s.np_local,
                      inputs=[s.v, s.mass, ke], device=self.device)
            wp.launch(core.strain_energy, dim=s.np_local,
                      inputs=[s.sigma, s.strain_el, s.vol, es],
                      device=self.device)
        ke_v = self.universe.allreduce_sum(float(ke.numpy()[0]))
        es_v = self.universe.allreduce_sum(float(es.numpy()[0]))
        return ke_v, es_v

    def _output(self, step):
        if self.dump is not None:
            self.dump.write(step, self.solids)
        ke, es = self._energies()
        if self.cfg.diagnostics:
            self._check_solution()
        self.elog.record(step, self.time, self.dt, ke, es)
        if self.universe.me == 0:
            print(f"step {step}  t = {self.time:.6g}  dt = {self.dt:.6g}  "
                  f"Ek = {ke:.6g}  Es = {es:.6g}")

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        nprocs = self.universe.nprocs
        step = 0
        t_start = pytime.time()

        self._output(0)

        end_step = cfg.run.end_step
        end_time = cfg.run.end_time
        interval = max(1, cfg.output.interval)

        while True:
            if end_step > 0 and step >= end_step:
                break
            if end_step == 0 and self.time >= end_time:
                break

            # C++ replicates: first step runs with the initial dt (1e-16 when
            # adaptive), then adjust_dt sets the CFL-based dt.
            self.solver.step(self.dt)
            step += 1
            self.time += self.dt

            self.dt = self.solver.adapt_dt(cfg.run.dt, cfg.run.dt_factor)

            if step % interval == 0:
                self._output(step)

        self._output(step)
        wall = pytime.time() - t_start
        if self.universe.me == 0:
            print(f"done: {step} steps in {wall:.2f}s "
                  f"({step / max(wall, 1e-9):.1f} steps/s)")
        return step
