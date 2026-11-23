"""Materials: linear-elastic, neo-hookean, and EOS+strength composites.

Port of Karamelo's material.cpp assembly:
  - linear / neo-hookean: K, G, lambda from (E, nu)
  - eos-strength: rho0/K from the EOS, G from the strength model
Constitutive updates live in kernels/core.py as Warp kernels; these classes
carry the parameters.
"""

import math
from dataclasses import dataclass


@dataclass
class LinearElastic:
    rho0: float
    E: float
    nu: float
    G: float
    K: float
    lam: float

    @classmethod
    def from_E_nu(cls, rho: float, E: float, nu: float) -> "LinearElastic":
        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return cls(rho0=rho, E=E, nu=nu, G=G, K=K, lam=lam)

    # C++ material.cpp:762 (linear material): c = sqrt(K / rho0)
    @property
    def signal_velocity(self) -> float:
        return math.sqrt(self.K / self.rho0)

    @property
    def p_wave_speed(self) -> float:
        return math.sqrt((self.K + 4.0 / 3.0 * self.G) / self.rho0)


@dataclass
class NeoHookean:
    """Compressible Neo-Hookean (port of material.cpp NEO_HOOKEAN branch)."""

    rho0: float
    E: float
    nu: float
    G: float
    K: float
    lam: float

    @classmethod
    def from_E_nu(cls, rho: float, E: float, nu: float) -> "NeoHookean":
        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return cls(rho0=rho, E=E, nu=nu, G=G, K=K, lam=lam)

    @property
    def signal_velocity(self) -> float:
        return math.sqrt(self.K / self.rho0)

    @property
    def p_wave_speed(self) -> float:
        return math.sqrt((self.K + 4.0 / 3.0 * self.G) / self.rho0)


# ---------------------------------------------------------------------------
# EOS models (port of eos_*.cpp compute_pressure; damage-free phase)

@dataclass
class EOSLinear:
    rho0: float
    K: float

    def compute_pressure(self, J: float, rho: float, D_trace: float,
                         cellsize: float) -> float:
        return self.K * (1.0 - J)


@dataclass
class EOSFluid:
    rho0: float
    K: float
    gamma: float

    def compute_pressure(self, J: float, rho: float, D_trace: float,
                         cellsize: float) -> float:
        mu = rho / self.rho0
        return self.K * (mu ** self.gamma - 1.0)


@dataclass
class EOSShock:
    """Mie-Gruneisen with artificial viscosity (port of eos_shock.cpp)."""

    rho0: float
    K: float
    c0: float
    S: float
    gamma: float
    Tr: float
    cv: float
    Q1: float = 0.0
    Q2: float = 0.0

    @property
    def artificial_viscosity(self) -> bool:
        return not (self.Q1 == 0.0 and self.Q2 == 0.0)

    def compute_pressure(self, J: float, rho: float, D_trace: float,
                         cellsize: float) -> float:
        mu = rho / self.rho0 - 1.0
        pH = (self.rho0 * self.c0 ** 2 * mu * (1.0 + mu)
              / (1.0 - (self.S - 1.0) * mu) ** 2)
        p = pH  # Gamma*(e-e0) term is zero for the isothermal examples
        if self.artificial_viscosity and D_trace < 0.0:
            q = (rho * cellsize
                 * (self.Q1 * cellsize * D_trace * D_trace
                    - self.Q2 * self.c0 * math.sqrt(J) * D_trace))
            p += q
        return p


# ---------------------------------------------------------------------------
# Strength models (port of strength_*.cpp update_deviatoric_stress)

@dataclass
class StrengthLinear:
    G: float

    def update_deviatoric(self, sigma, D, dt):
        """sigma, D are 3x3 numpy arrays; returns new deviatoric stress."""
        dev = D - np_trace_third(D)
        return sigma + dt * 2.0 * self.G * dev


@dataclass
class StrengthFluid:
    G: float

    def update_deviatoric(self, sigma, D, dt):
        return 2.0 * self.G * deviator_np(D)


@dataclass
class StrengthPlastic:
    """J2 radial return (port of strength_plastic.cpp)."""

    G: float
    yield_stress: float

    def update_deviatoric(self, sigma, D, dt):
        dev_rate = 2.0 * self.G * deviator_np(D)
        s_trial = deviator_np(sigma) + dt * dev_rate
        J2 = math.sqrt(1.5) * float(np.linalg.norm(s_trial))
        if J2 < self.yield_stress:
            return s_trial, 0.0
        inc = (J2 - self.yield_stress) / (3.0 * self.G)
        return s_trial * (self.yield_stress / J2), inc


@dataclass
class StrengthSwift:
    """Swift flow stress + radial return (port of strength_swift.cpp).

    sigma_f = [A + B (eps_p - C)^n] (1 - damage)
    """

    G: float
    A: float
    B: float
    C: float
    n: float

    def update_deviatoric(self, sigma, D, dt, eff_plastic_strain, epsdot):
        if eff_plastic_strain < self.C:
            ys = self.A
        else:
            ys = self.A + self.B * (eff_plastic_strain - self.C) ** self.n
        s_trial = deviator_np(sigma) + dt * 2.0 * self.G * deviator_np(D)
        J2 = math.sqrt(1.5) * float(np.linalg.norm(s_trial))
        if J2 < ys:
            return s_trial, 0.0
        inc = (J2 - ys) / (3.0 * self.G)
        return s_trial * (ys / J2), inc


@dataclass
class StrengthJohnsonCook:
    """Johnson-Cook flow stress + radial return (port of strength_jc.cpp).

    Isothermal for phase 2 (examples run without thermo-mechanical coupling,
    so T = Tr and the thermal softening factor is 1).
    """

    G: float
    A: float
    B: float
    n: float
    epsdot0: float
    C: float
    m: float
    Tr: float
    Tm: float

    def yield_stress_at(self, eff_plastic_strain: float, epsdot: float) -> float:
        ratio = max(epsdot / self.epsdot0, 1.0)
        if eff_plastic_strain < 1.0e-10:
            ys = self.A
        else:
            ys = self.A + self.B * eff_plastic_strain ** self.n
        if self.C != 0:
            ys *= (1.0 + ratio) ** self.C
        # T = Tr for isothermal runs -> thermal factor 1
        return ys

    def update_deviatoric(self, sigma, D, dt, eff_plastic_strain, epsdot):
        ys = self.yield_stress_at(eff_plastic_strain, epsdot)
        s_trial = deviator_np(sigma + dt * 2.0 * self.G * D)
        J2 = math.sqrt(1.5) * float(np.linalg.norm(s_trial))
        if J2 < ys:
            return s_trial, 0.0
        inc = (J2 - ys) / (3.0 * self.G)
        return s_trial * (ys / J2), inc


def deviator_np(M):
    return M - np_trace_third(M)


def np_trace_third(M):
    return (M[0, 0] + M[1, 1] + M[2, 2]) / 3.0 * _eye3()


def _eye3():
    import numpy as np
    return np.eye(3)


@dataclass
class Rigid:
    """Rigid material (port of material.cpp 'rigid'): infinite stiffness;
    nodes touched by rigid particles become Dirichlet-like for other solids."""

    rho0: float = 1000.0
    E: float = 0.0
    nu: float = 0.0
    G: float = 0.0
    K: float = 0.0
    lam: float = 0.0
    rigid: bool = True

    @property
    def signal_velocity(self) -> float:
        return 1.0e30

    @property
    def p_wave_speed(self) -> float:
        return 1.0e30


# ---------------------------------------------------------------------------
# Composite EOS+strength material

@dataclass
class EOSStrength:
    """Port of material.cpp eos-strength: rho0/K from EOS, G from strength."""

    rho0: float
    eos: object   # EOSLinear | EOSFluid | EOSShock
    strength: object  # StrengthLinear | StrengthFluid | ...
    damage: object = None  # DamageJohnsonCook or None

    @property
    def K(self) -> float:
        return self.eos.K

    @property
    def G(self) -> float:
        return self.strength.G

    @property
    def p_wave_speed(self) -> float:
        return math.sqrt((self.K + 4.0 / 3.0 * self.G) / self.rho0)


@dataclass
class DamageJohnsonCook:
    """Johnson-Cook damage model (port of damage_jc.cpp).

    Failure strain eps_f = (d1 + d2 exp(d3 eta)) (1 + d4 ln(epsdot/epsdot0))
    (1 - d5 T*)  where eta = triaxiality = -p / seq.
    damage_init += deps_p / eps_f;  damage = min(10 (damage_init - 1), 1).
    """

    d1: float
    d2: float
    d3: float
    d4: float = 0.0
    d5: float = 0.0
    epsdot0: float = 1.0

    def compute_damage(self, damage_init: float, p: float, seq: float,
                       eff_plastic_strain_rate: float,
                       plastic_strain_increment: float) -> float:
        import math
        if plastic_strain_increment <= 0.0 or seq <= 0.0:
            return damage_init
        eta = -p / seq  # triaxiality
        eps_f = self.d1 + self.d2 * math.exp(self.d3 * eta)
        if self.d4 != 0.0:
            ratio = max(eff_plastic_strain_rate / self.epsdot0, 1.0)
            eps_f *= (1.0 + self.d4 * math.log(ratio))
        if eps_f <= 0.0:
            return damage_init
        return damage_init + plastic_strain_increment / eps_f
