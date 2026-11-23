"""Particle (Solid) state: SoA Warp arrays, population, migration I/O."""

import numpy as np
import warp as wp

F64 = wp.float64
V3 = wp.vec3d
M3 = wp.mat33d

# migration buffer layout (f64 per particle): x(3) x0(3) v(3) F(9) sigma(6)
# strain_el(6) J vol vol0 ep epdot -> 35 doubles
MIG_NCOMP = 35


def _mat33_to6(m: np.ndarray) -> np.ndarray:
    """(n,3,3) -> (n,6) [s11,s22,s33,s12,s13,s23]."""
    return np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
                     m[:, 0, 1], m[:, 0, 2], m[:, 1, 2]], axis=1)


def _6_to_mat33(c: np.ndarray) -> np.ndarray:
    """(n,6) [s11,s22,s33,s12,s13,s23] -> symmetric (n,3,3)."""
    n = c.shape[0]
    m = np.empty((n, 3, 3))
    m[:, 0, 0] = c[:, 0]
    m[:, 1, 1] = c[:, 1]
    m[:, 2, 2] = c[:, 2]
    m[:, 0, 1] = m[:, 1, 0] = c[:, 3]
    m[:, 0, 2] = m[:, 2, 0] = c[:, 4]
    m[:, 1, 2] = m[:, 2, 1] = c[:, 5]
    return m


def _mig_pack(p: "Solid") -> np.ndarray:
    n = p.np_local
    buf = np.empty((n, MIG_NCOMP))
    buf[:, 0:3] = p.x.numpy()[:n]
    buf[:, 3:6] = p.x0.numpy()[:n]
    buf[:, 6:9] = p.v.numpy()[:n]
    buf[:, 9:18] = p.F.numpy()[:n].reshape(n, 9)
    buf[:, 18:24] = _mat33_to6(p.sigma.numpy()[:n])
    buf[:, 24:30] = _mat33_to6(p.strain_el.numpy()[:n])
    buf[:, 30] = p.J.numpy()[:n]
    buf[:, 31] = p.vol.numpy()[:n]
    buf[:, 32] = p.vol0.numpy()[:n]
    buf[:, 33] = p.ep.numpy()[:n]
    buf[:, 34] = p.epdot.numpy()[:n]
    return buf


def _mig_unpack(buf: np.ndarray):
    """Return dict of numpy arrays for (re)building particle arrays."""
    n = buf.shape[0]
    return {
        "x": buf[:, 0:3],
        "x0": buf[:, 3:6],
        "v": buf[:, 6:9],
        "F": buf[:, 9:18].reshape(n, 3, 3),
        "sigma": _6_to_mat33(buf[:, 18:24]),
        "strain_el": _6_to_mat33(buf[:, 24:30]),
        "J": buf[:, 30],
        "vol": buf[:, 31],
        "vol0": buf[:, 32],
        "ep": buf[:, 33],
        "epdot": buf[:, 34],
    }


class Solid:
    """One body of particles sharing a material."""

    def __init__(self, sid: str, material, device):
        self.id = sid
        self.mat = material
        self.device = device
        self.np_local = 0
        self.empty = True

    def populate(self, pts: np.ndarray, vols: np.ndarray, rho0: np.ndarray):
        """Create particle arrays from host positions (single-rank setup)."""
        n = pts.shape[0]
        self.np_local = n
        dev = self.device
        zeros3 = np.zeros((n, 3))
        zeros33 = np.zeros((n, 3, 3))
        self.x = wp.array(pts, dtype=V3, device=dev)
        self.x0 = wp.array(pts.copy(), dtype=V3, device=dev)
        self.v = wp.zeros(n, dtype=V3, device=dev)
        self.v_pic = wp.zeros(n, dtype=V3, device=dev)
        self.a = wp.zeros(n, dtype=V3, device=dev)
        self.mbp = wp.zeros(n, dtype=V3, device=dev)
        self.F = wp.array(np.tile(np.eye(3), (n, 1, 1)), dtype=M3, device=dev)
        self.L = wp.zeros(n, dtype=M3, device=dev)
        self.D = wp.zeros(n, dtype=M3, device=dev)
        self.sigma = wp.zeros(n, dtype=M3, device=dev)
        self.strain_el = wp.zeros(n, dtype=M3, device=dev)
        self.R = wp.zeros(n, dtype=M3, device=dev)
        self.Finv = wp.zeros(n, dtype=M3, device=dev)
        self.vol0PK1 = wp.zeros(n, dtype=M3, device=dev)
        self.ep = wp.zeros(n, dtype=F64, device=dev)
        self.epdot = wp.zeros(n, dtype=F64, device=dev)
        self.damage = wp.zeros(n, dtype=F64, device=dev)
        self.damage_init = wp.zeros(n, dtype=F64, device=dev)
        # CPDI particle domain (2D): R4 edges and Q4 corners
        self.rp0_e1 = wp.zeros(n, dtype=V3, device=dev)
        self.rp0_e2 = wp.zeros(n, dtype=V3, device=dev)
        self.rp_e1 = wp.zeros(n, dtype=V3, device=dev)
        self.rp_e2 = wp.zeros(n, dtype=V3, device=dev)
        self.xpc = wp.zeros(4 * n, dtype=V3, device=dev)
        self.xpc_new = wp.zeros(4 * n, dtype=V3, device=dev)
        self.J = wp.ones(n, dtype=F64, device=dev)
        self.vol = wp.array(vols, dtype=F64, device=dev)
        self.vol0 = wp.array(vols, dtype=F64, device=dev)
        self.rho = wp.array(rho0, dtype=F64, device=dev)
        self.rho0 = wp.array(rho0, dtype=F64, device=dev)
        self.mass = wp.array(vols * rho0, dtype=F64, device=dev)
        self.owner = wp.zeros(n, dtype=wp.int32, device=dev)
        self.empty = False

    def rebuild(self, arrays: dict):
        """Replace particle arrays after migration (numpy dict from _mig_unpack)."""
        n = arrays["x"].shape[0]
        self.np_local = n
        dev = self.device
        self.x = wp.array(arrays["x"], dtype=V3, device=dev)
        self.x0 = wp.array(arrays["x0"], dtype=V3, device=dev)
        self.v = wp.array(arrays["v"], dtype=V3, device=dev)
        self.v_pic = wp.zeros(n, dtype=V3, device=dev)
        self.a = wp.zeros(n, dtype=V3, device=dev)
        self.mbp = wp.zeros(n, dtype=V3, device=dev)
        self.F = wp.array(arrays["F"], dtype=M3, device=dev)
        self.L = wp.zeros(n, dtype=M3, device=dev)
        self.D = wp.zeros(n, dtype=M3, device=dev)
        self.sigma = wp.array(arrays["sigma"], dtype=M3, device=dev)
        self.strain_el = wp.array(arrays["strain_el"], dtype=M3, device=dev)
        self.R = wp.zeros(n, dtype=M3, device=dev)
        self.Finv = wp.zeros(n, dtype=M3, device=dev)
        self.vol0PK1 = wp.zeros(n, dtype=M3, device=dev)
        self.ep = wp.array(arrays["ep"], dtype=F64, device=dev)
        self.epdot = wp.array(arrays["epdot"], dtype=F64, device=dev)
        self.damage = wp.zeros(n, dtype=F64, device=dev)
        self.damage_init = wp.zeros(n, dtype=F64, device=dev)
        self.rp0_e1 = wp.zeros(n, dtype=V3, device=dev)
        self.rp0_e2 = wp.zeros(n, dtype=V3, device=dev)
        self.rp_e1 = wp.zeros(n, dtype=V3, device=dev)
        self.rp_e2 = wp.zeros(n, dtype=V3, device=dev)
        self.xpc = wp.zeros(4 * n, dtype=V3, device=dev)
        self.xpc_new = wp.zeros(4 * n, dtype=V3, device=dev)
        self.J = wp.array(arrays["J"], dtype=F64, device=dev)
        self.vol = wp.array(arrays["vol"], dtype=F64, device=dev)
        self.vol0 = wp.array(arrays["vol0"], dtype=F64, device=dev)
        rho0 = np.full(n, self.mat.rho0)
        self.rho0 = wp.array(rho0, dtype=F64, device=dev)
        self.rho = wp.array(rho0 / np.maximum(arrays["J"], 1e-300),
                            dtype=F64, device=dev)
        self.mass = wp.array(arrays["vol0"] * rho0, dtype=F64, device=dev)
        self.owner = wp.zeros(n, dtype=wp.int32, device=dev)
        self.xw = self.x
        self.empty = n == 0

    def zero_mbp(self):
        self.mbp.zero_()

    def pack_for_migration(self) -> np.ndarray:
        return _mig_pack(self)

    @staticmethod
    def unpack_from_migration(buf: np.ndarray) -> dict:
        return _mig_unpack(buf)
