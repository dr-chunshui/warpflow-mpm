"""LAMMPS-style particle dumps and scalar energy logging (Karamelo-compatible)."""

import numpy as np

FIELD_NAMES = ("x", "y", "z", "vx", "vy", "vz",
               "s11", "s22", "s33", "s12", "s13", "s23",
               "e11", "e22", "e33", "e12", "e13", "e23",
               "volume", "mass", "J")


class ParticleDump:
    def __init__(self, filename_pattern, fields, universe):
        self.pattern = filename_pattern
        self.fields = tuple(fields)
        self.universe = universe
        unknown = [f for f in self.fields if f not in FIELD_NAMES]
        if unknown:
            raise ValueError(f"unknown dump fields {unknown}; supported: {FIELD_NAMES}")

    def _filename(self, step: int) -> str:
        name = self.pattern
        if "*" in name:
            pre, post = name.split("*", 1)
            if self.universe.nprocs > 1:
                pre += f"proc-{self.universe.me}."
            name = pre + str(step) + post
        return name

    def write(self, step: int, solids):
        n_total = sum(s.np_local for s in solids)
        lines = [
            "ITEM: TIMESTEP",
            "0",
            "ITEM: NUMBER OF ATOMS",
            str(n_total),
            "ITEM: BOX BOUNDS sm sm sm",
        ]
        bl, bh = self.universe.boxlo, self.universe.boxhi
        for d in range(3):
            lines.append(f"{bl[d]:.17g} {bh[d]:.17g}")
        lines.append("ITEM: ATOMS id type tag " + " ".join(self.fields))

        cols = {f: [] for f in self.fields}
        types = []
        for isolid, s in enumerate(solids):
            if s.np_local == 0:
                continue
            n = s.np_local
            x = s.x.numpy()[:n]
            v = s.v.numpy()[:n]
            sig = s.sigma.numpy()[:n].reshape(n, 3, 3)
            eps = s.strain_el.numpy()[:n].reshape(n, 3, 3)
            vol = s.vol.numpy()[:n]
            mass = s.mass.numpy()[:n]
            J = s.J.numpy()[:n]
            ptag = np.arange(n, dtype=np.int64)  # per-rank index as tag (phase 1)
            data = {
                "x": x[:, 0], "y": x[:, 1], "z": x[:, 2],
                "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2],
                "s11": sig[:, 0, 0], "s22": sig[:, 1, 1], "s33": sig[:, 2, 2],
                "s12": sig[:, 0, 1], "s13": sig[:, 0, 2], "s23": sig[:, 1, 2],
                "e11": eps[:, 0, 0], "e22": eps[:, 1, 1], "e33": eps[:, 2, 2],
                "e12": eps[:, 0, 1], "e13": eps[:, 0, 2], "e23": eps[:, 1, 2],
                "volume": vol, "mass": mass, "J": J,
            }
            for f in self.fields:
                cols[f].append(data[f])
            types.append(np.full(n, isolid + 1, dtype=np.int64))

        for f in self.fields:
            if cols[f]:
                cols[f] = np.concatenate(cols[f])
            else:
                cols[f] = np.zeros(0)
        if types:
            types = np.concatenate(types)
        else:
            types = np.zeros(0, dtype=np.int64)

        with open(self._filename(step), "w") as fh:
            fh.write("\n".join(lines) + "\n")
            for i in range(types.shape[0]):
                fh.write(f"{i + 1} {types[i]} {i + 1} ")
                fh.write(" ".join(f"{cols[f][i]:.17g}" for f in self.fields))
                fh.write("\n")


class EnergyLog:
    """CSV log of diagnostics: step, time, dt, kinetic energy, strain energy."""

    def __init__(self, universe, path="log.csv"):
        self.universe = universe
        self.path = path
        self.rows = []
        if universe.me == 0:
            with open(path, "w") as fh:
                fh.write("step,time,dt,kinetic_energy,strain_energy\n")

    def record(self, step, time, dt, ke, es):
        self.rows.append((step, time, dt, ke, es))
        if self.universe.me == 0:
            with open(self.path, "a") as fh:
                fh.write(f"{step},{time:.17g},{dt:.17g},{ke:.17g},{es:.17g}\n")
