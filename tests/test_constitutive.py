"""Constitutive model unit tests: analytic checks against the C++ formulas.

Each test drives a single-particle update_stress / update_F_vol_D launch and
compares against hand-computed values of the ported constitutive models.
"""

import math

import numpy as np
import pytest
import warp as wp

from warpflow_mpm.kernels import core
from warpflow_mpm.kernels.core import (EOS_FLUID, EOS_LINEAR, EOS_SHOCK,
                                       MODEL_EOS, MODEL_LINEAR,
                                       MODEL_NEO_HOOKEAN, STRENGTH_FLUID,
                                       STRENGTH_JOHNSON_COOK, STRENGTH_LINEAR,
                                       STRENGTH_PLASTIC)

DEV = 'cuda:0' if wp.get_cuda_devices() else 'cpu'


def _launch_stress(sigma0, D, F0=None, model=MODEL_LINEAR, eos_k=EOS_LINEAR,
                   st_k=STRENGTH_LINEAR, G=1.0e5, lam=1.0e5, K=1.0e5,
                   st_yield=0.0, st_A=0.0, st_B=0.0, st_n=0.0,
                   st_epsdot0=1.0, st_C=0.0, st_m=0.0,
                   eos_rho0=1000.0, eos_gamma=0.0, eos_c0=0.0, eos_S=0.0,
                   eos_Q1=0.0, eos_Q2=0.0, dt=1e-6, cellsize=0.1,
                   signal_velocity=1.0):
    sigma = wp.array(sigma0.reshape(1, 3, 3), dtype=wp.mat33d, device=DEV)
    Darr = wp.array(D.reshape(1, 3, 3), dtype=wp.mat33d, device=DEV)
    strain_el = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    F = wp.array((F0 if F0 is not None else np.eye(3)).reshape(1, 3, 3),
                 dtype=wp.mat33d, device=DEV)
    R = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    Finv = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    vol0 = wp.array([1.0], dtype=wp.float64, device=DEV)
    vol0PK1 = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    ep = wp.zeros(1, dtype=wp.float64, device=DEV)
    epdot = wp.zeros(1, dtype=wp.float64, device=DEV)
    damage = wp.zeros(1, dtype=wp.float64, device=DEV)
    damage_init = wp.zeros(1, dtype=wp.float64, device=DEV)
    wp.launch(core.update_stress, dim=1,
              inputs=[sigma, Darr, strain_el, F, R, Finv, vol0, vol0PK1,
                      ep, epdot, damage, damage_init,
                      model, eos_k, st_k, 0,
                      wp.float64(G), wp.float64(lam), wp.float64(K),
                      wp.float64(cellsize), wp.float64(signal_velocity),
                      wp.float64(eos_rho0), wp.float64(eos_gamma),
                      wp.float64(eos_c0), wp.float64(eos_S),
                      wp.float64(eos_Q1), wp.float64(eos_Q2),
                      wp.float64(st_yield), wp.float64(st_A),
                      wp.float64(st_B), wp.float64(st_n),
                      wp.float64(st_epsdot0), wp.float64(st_C),
                      wp.float64(st_m),
                      wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), 0,
                      wp.float64(dt)],
              device=DEV)
    out = {
        "sigma": sigma.numpy()[0],
        "strain_el": strain_el.numpy()[0],
        "R": R.numpy()[0],
        "Finv": Finv.numpy()[0],
        "vol0PK1": vol0PK1.numpy()[0],
        "ep": ep.numpy()[0],
        "epdot": epdot.numpy()[0],
    }
    return out


def test_linear_elastic_uniaxial_rate():
    """sigma += dt (2G D + lam tr(D) I) for a pure shear-strain rate."""
    G, lam, dt = 3.0e5, 1.0e5, 2.0e-6
    D = np.zeros((3, 3))
    D[0, 1] = D[1, 0] = 1.0e3   # shear rate
    out = _launch_stress(np.zeros((3, 3)), D, G=G, lam=lam, dt=dt)
    expected = dt * (2 * G * D + lam * np.trace(dt * D) * np.eye(3))
    assert np.allclose(out["sigma"], expected, rtol=1e-13)
    assert np.allclose(out["strain_el"], dt * D, rtol=1e-13)


def test_neo_hookean_volumetric():
    """F = lambda I: PK1_11 = G(l - 1/l) + 3 ln l; sigma = PK1/l^2."""
    G, lam = 3.0e5, 1.5e5
    stretch = 1.1
    F = np.eye(3) * stretch
    J = stretch ** 3
    PK1_11 = G * (stretch - 1.0 / stretch) + lam * math.log(J) / stretch
    sigma_11 = PK1_11 / stretch ** 2
    out = _launch_stress(np.zeros((3, 3)), np.zeros((3, 3)), F0=F,
                         model=MODEL_NEO_HOOKEAN, G=G, lam=lam)
    assert out["sigma"][0, 0] == pytest.approx(sigma_11, rel=1e-12)
    assert out["sigma"][1, 1] == pytest.approx(sigma_11, rel=1e-12)
    assert out["sigma"][0, 1] == pytest.approx(0.0, abs=1e-12)


def test_neo_hookean_shear_block():
    """F = I + gamma e_xy: sigma_12 = G*gamma/J with J = 1."""
    G, lam = 3.0e5, 1.5e5
    gamma = 0.05
    F = np.eye(3)
    F[0, 1] = gamma
    J = 1.0
    # PK1_12 = G*(F12) + lam*ln(1)*FinvT_12 = G*gamma
    # sigma = F PK1^T / J -> sigma_12 = F_1k PK1_k2 = PK1_12 (F=I row 1)
    out = _launch_stress(np.zeros((3, 3)), np.zeros((3, 3)), F0=F,
                         model=MODEL_NEO_HOOKEAN, G=G, lam=lam)
    assert out["sigma"][0, 1] == pytest.approx(G * gamma, rel=1e-12)


def test_j2_plastic_radial_return():
    """Trial stress beyond yield must be scaled back onto the yield surface."""
    G = 3.0e5
    yield_stress = 1.0e4
    dt = 1.0e-6
    D = np.zeros((3, 3))
    D[0, 0] = -5.0e5
    D[1, 1] = 5.0e5      # pure deviatoric rate; |trial| >> yield
    sigma0 = np.zeros((3, 3))
    s_trial = sigma0 + dt * 2 * G * (D - np.trace(D) / 3 * np.eye(3))
    J2 = math.sqrt(1.5) * np.linalg.norm(s_trial)
    out = _launch_stress(sigma0, D, model=MODEL_EOS, eos_k=EOS_LINEAR,
                         st_k=STRENGTH_PLASTIC, G=G, st_yield=yield_stress,
                         dt=dt)
    s_final = out["sigma"] - np.trace(out["sigma"]) / 3 * np.eye(3)
    assert math.sqrt(1.5) * np.linalg.norm(s_final) == pytest.approx(
        yield_stress, rel=1e-12)
    assert out["ep"] == pytest.approx((J2 - yield_stress) / (3 * G), rel=1e-12)


def test_fluid_strength_and_eos():
    """Fluid: S = 2G dev(D); fluid EOS: p = K (mu^gamma - 1)."""
    G, K, gamma = 1.0e3, 2.0e6, 7.0
    J = 0.95
    F = np.eye(3) * J ** (1.0 / 3.0)
    dt = 1e-6
    D = np.zeros((3, 3))
    D[0, 1] = D[1, 0] = 100.0
    out = _launch_stress(np.zeros((3, 3)), D, F0=F, model=MODEL_EOS,
                         eos_k=EOS_FLUID, st_k=STRENGTH_FLUID,
                         G=G, K=K, eos_rho0=1000.0, eos_gamma=gamma, dt=dt)
    mu = 1.0 / J
    p_expected = K * (mu ** gamma - 1.0)
    assert out["sigma"][0, 0] == pytest.approx(-p_expected, rel=1e-12)
    # StrengthFluid returns the deviatoric stress directly (no dt factor):
    # S = 2G dev(D)
    assert out["sigma"][0, 1] == pytest.approx(2 * G * D[0, 1], rel=1e-12)


def test_shock_eos_pressure():
    """Mie-Gruneisen compression pressure (no viscosity at D.trace()=0)."""
    rho0, c0, S = 1000.0, 1500.0, 1.5
    J = 0.8
    F = np.eye(3) * J ** (1.0 / 3.0)
    mu = 1.0 / J - 1.0
    p_expected = rho0 * c0 ** 2 * mu * (1 + mu) / (1 - (S - 1) * mu) ** 2
    out = _launch_stress(np.zeros((3, 3)), np.zeros((3, 3)), F0=F,
                         model=MODEL_EOS, eos_k=EOS_SHOCK, st_k=STRENGTH_LINEAR,
                         G=1e3, K=1e9, eos_rho0=rho0, eos_c0=c0, eos_S=S,
                         dt=1e-6)
    assert out["sigma"][0, 0] == pytest.approx(-p_expected, rel=1e-12)


def test_polar_decomposition_rotation():
    """update_F_vol_D TL: R must be orthogonal and det(R)=+1."""
    # random deformation with det>0
    rng = np.random.default_rng(42)
    A = rng.standard_normal((3, 3)) * 0.3 + np.eye(3)
    if np.linalg.det(A) < 0:
        A[:, 0] *= -1.0
    F = wp.array(A.reshape(1, 3, 3), dtype=wp.mat33d, device=DEV)
    L = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    J = wp.ones(1, dtype=wp.float64, device=DEV)
    vol0 = wp.ones(1, dtype=wp.float64, device=DEV)
    vol = wp.zeros(1, dtype=wp.float64, device=DEV)
    rho0 = wp.ones(1, dtype=wp.float64, device=DEV)
    rho = wp.zeros(1, dtype=wp.float64, device=DEV)
    D = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    R = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    Finv = wp.zeros(1, dtype=wp.mat33d, device=DEV)
    wp.launch(core.update_F_vol_D, dim=1,
              inputs=[L, F, J, vol0, vol, rho0, rho, D, R, Finv,
                      wp.float64(1e-6), 1],
              device=DEV)
    Rn = R.numpy()[0]
    assert np.allclose(Rn @ Rn.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(Rn) == pytest.approx(1.0, abs=1e-10)
    # R should equal the polar factor of F (unique for det>0)
    U, S, Vt = np.linalg.svd(A)
    R_ref = U @ Vt
    assert np.allclose(np.abs(Rn), np.abs(R_ref), atol=1e-8)
