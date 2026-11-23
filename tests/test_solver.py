"""Solver-level tests: momentum conservation and stability (single rank)."""

import numpy as np
import pytest
import yaml

from warpflow_mpm.config import load_config
from warpflow_mpm.simulation import Simulation

BASE_CFG = {
    "dimension": 2,
    "box": [[-1.0, -1.0], [1.0, 1.0]],
    "cellsize": 0.2,
    "basis": "linear",
    "method": {"type": "ulmpm", "transfer": "flip", "pic_flip": 0.99,
               "scheme": "musl"},
    "materials": [
        {"id": "mat1", "model": "linear-elastic", "rho": 1000.0,
         "E": 1.0e5, "nu": 0.3},
    ],
    "regions": [
        {"id": "rBlock", "type": "box", "center": [-0.4, 0.0],
         "size": [0.3, 0.3]},
    ],
    "solids": [{"id": "sBlock", "material": "mat1", "region": "rBlock",
                "ppc": 2}],
    "initial_conditions": [
        {"id": "v0", "type": "initial_velocity", "select": "rBlock",
         "v": [0.05, 0.0]},
    ],
    "output": {"interval": 1000},
    "run": {"end_step": 1},
}


@pytest.fixture(scope="module")
def tmp_cfg_factory(tmp_path_factory):
    def make(**overrides):
        cfg = {**BASE_CFG, **overrides}
        path = tmp_path_factory.mktemp("cfg") / "cfg.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        return load_config(path)
    return make


def test_momentum_conservation(tmp_cfg_factory):
    """A free-floating block must conserve total linear momentum (no BCs)."""
    cfg = tmp_cfg_factory(run={"end_step": 60})
    sim = Simulation(cfg)
    sim.dt = 1.0e-4
    solver = sim.solver

    def total_px():
        return sum(float((s.v.numpy()[:, 0] * s.mass.numpy()).sum())
                   for s in sim.solids)

    p0 = total_px()
    for _ in range(60):
        solver.step(sim.dt)
        sim.time += sim.dt
        p = total_px()
        assert p == pytest.approx(p0, rel=1e-9), (p0, p)


def test_free_block_energy_stable(tmp_cfg_factory):
    """No external work: kinetic + strain energy must stay bounded."""
    cfg = tmp_cfg_factory(run={"end_step": 60})
    sim = Simulation(cfg)
    sim.dt = 1.0e-4
    solver = sim.solver

    e0 = None
    for _ in range(60):
        solver.step(sim.dt)
        sim.time += sim.dt
        ke, es = sim._energies()
        e = ke + es
        if e0 is None:
            e0 = e
        assert e < 2.0 * e0 + 1e-9, (e0, e)


def test_fixed_nodes_hold(tmp_cfg_factory):
    """velocity_nodes BC pins the selected grid nodes exactly."""
    cfg = tmp_cfg_factory(
        regions=[{"id": "rBlock", "type": "box", "center": [0.0, -0.4],
                  "size": [0.3, 0.3]},
                 {"id": "rBottom", "type": "box", "center": [0.0, -1.0],
                  "size": [1.5, 0.15]}],
        solids=[{"id": "sBlock", "material": "mat1", "region": "rBlock",
                 "ppc": 2}],
        boundary_conditions=[
            {"id": "fix", "type": "velocity_nodes", "region": "rBottom",
             "vx": 0, "vy": 0},
            {"id": "g", "type": "body_force", "g": [0, -9.8]},
        ],
        initial_conditions=[],
        run={"end_step": 40},
    )
    sim = Simulation(cfg)
    sim.dt = 1.0e-4
    solver = sim.solver
    g = solver.grid
    bcs = solver.bcs

    vbc_node_ids = []
    for bc in bcs.velocity_node_bcs:
        nl = bc.node_list(g, cfg.dimension)
        vbc_node_ids.append(nl.numpy())
    assert sum(len(v) for v in vbc_node_ids) > 0

    for _ in range(40):
        solver.step(sim.dt)
        sim.time += sim.dt

    for node_list in vbc_node_ids:
        vu = g.v_update.numpy()[node_list]
        assert np.abs(vu).max() == pytest.approx(0.0, abs=1e-14)
