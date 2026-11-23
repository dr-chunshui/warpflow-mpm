"""YAML configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

BASIS_TYPES = ("linear", "quadratic-spline", "cubic-spline", "bernstein")
SCHEME_TYPES = ("usl", "musl")  # usf not implemented yet
TRANSFER_TYPES = ("flip", "pic", "apic")
METHOD_TYPES = ("ulmpm", "tlmpm", "ulcpdi")
CPDI_STYLES = ("r4", "q4")


class ConfigError(ValueError):
    pass


@dataclass
class MethodConfig:
    type: str = "ulmpm"          # ulmpm | tlmpm | ulcpdi
    transfer: str = "flip"       # flip | pic | apic
    pic_flip: float = 0.99       # alpha: 0 = pure PIC, 1 = pure FLIP
    scheme: str = "musl"         # usl | musl
    style: str = "r4"            # CPDI style: r4 | q4


@dataclass
class MaterialConfig:
    id: str
    model: str                   # linear-elastic | neo-hookean | eos-strength | rigid
    rho: float = 0.0
    E: float = 0.0
    nu: float = 0.0
    eos: dict | None = None      # {type, rho0, K, [gamma], [c0, S, cv, Tr, Q1, Q2]}
    strength: dict | None = None  # {type, G, [yield_stress], [A, B, n, epsdot0, C, m, Tr, Tm]}


@dataclass
class RegionConfig:
    id: str
    type: str                    # box | cylinder | sphere | rectangle
    axis: str = "z"              # cylinder axis (x | y | z)
    inner_radius: float = 0.0    # cylinder annulus hole (0 = solid)
    exterior: bool = False       # select r >= radius instead of r <= radius
    center: tuple = ()
    size: tuple = ()             # box half-sizes
    radius: float = 0.0
    height: float = 0.0          # cylinder height (2D: ignored)


@dataclass
class SolidConfig:
    id: str
    material: str
    region: str                  # region id to populate from
    ppc: int = 2                 # particles per cell edge


@dataclass
class VelocityNodesBC:
    id: str
    region: str                  # region id selecting grid nodes
    type: str = "velocity_nodes"
    vx: float | None = None      # None = leave free
    vy: float | None = None
    vz: float | None = None


@dataclass
class BodyForceBC:
    id: str
    type: str = "body_force"
    g: tuple = (0.0, 0.0, 0.0)   # body acceleration (gravity)


@dataclass
class ForceNodesBC:
    id: str
    region: str                  # region id selecting grid nodes
    force: tuple                 # total force, split over selected nodes
    type: str = "force_nodes"


@dataclass
class ContactBC:
    id: str
    solid1: str
    solid2: str
    mu: float = 0.0
    type: str = "contact"


@dataclass
class InitialStressBC:
    id: str
    region: str
    s11: float = 0.0
    s22: float = 0.0
    s33: float = 0.0
    s12: float = 0.0
    s13: float = 0.0
    s23: float = 0.0
    type: str = "initial_stress"


@dataclass
class InitialVelocityNodesBC:
    id: str
    region: str
    vx: float | None = None
    vy: float | None = None
    vz: float | None = None
    type: str = "initial_velocity_nodes"


@dataclass
class VelocityParticlesBC:
    id: str
    solid: str                   # solid id
    region: str                  # region selecting particles
    v: tuple = (0.0, 0.0, 0.0)
    type: str = "velocity_particles"


@dataclass
class CheckSolution:
    id: str
    u: tuple                     # manufactured-solution displacement expressions
    type: str = "check_solution"


@dataclass
class InitialVelocity:
    id: str
    select: str                  # region id selecting particles
    type: str = "initial_velocity"
    v: tuple = (0.0, 0.0, 0.0)


@dataclass
class OutputConfig:
    interval: int = 100
    particle_dump: str | None = None    # filename pattern with '*'
    fields: tuple = ("x", "y", "vx", "vy")
    log: tuple = ("kinetic_energy",)


@dataclass
class RunConfig:
    end_time: float = 0.0
    end_step: int = 0
    dt: float | None = None      # None = CFL-adaptive
    dt_factor: float = 0.9


@dataclass
class Config:
    dimension: int
    boxlo: tuple
    boxhi: tuple
    cellsize: float
    basis: str
    precision: str
    method: MethodConfig
    materials: list
    regions: list
    solids: list
    boundary_conditions: list
    initial_conditions: list
    output: OutputConfig
    run: RunConfig
    diagnostics: list = field(default_factory=list)
    variables: dict = field(default_factory=dict)

    @property
    def dt_given(self) -> bool:
        return self.run.dt is not None


def _num_or_str(v):
    """Numbers pass through; strings are kept as expressions."""
    if isinstance(v, str):
        return v
    return float(v)


def _need(cond: bool, msg: str):
    if not cond:
        raise ConfigError(msg)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    _need(isinstance(raw, dict), f"{path}: empty or invalid YAML")

    dim = int(raw.get("dimension", 3))
    _need(dim in (2, 3), f"dimension must be 2 or 3, got {dim}")

    box = raw.get("box")
    _need(box and len(box) == 2, "box must be [[x0,y0,(z0)],[x1,y1,(z1)]]")
    boxlo = tuple(float(v) for v in box[0])
    boxhi = tuple(float(v) for v in box[1])
    _need(len(boxlo) == dim and len(boxhi) == dim,
          f"box coordinates must have {dim} components")

    cellsize = float(raw["cellsize"])
    _need(cellsize > 0, "cellsize must be > 0")

    basis = raw.get("basis", "linear")
    _need(basis in BASIS_TYPES, f"basis must be one of {BASIS_TYPES}")

    precision = raw.get("precision", "float64")
    _need(precision in ("float32", "float64"), "precision must be float32/float64")

    m = raw.get("method", {})
    method = MethodConfig(
        type=m.get("type", "ulmpm"),
        transfer=m.get("transfer", "flip"),
        pic_flip=float(m.get("pic_flip", 0.99)),
        scheme=m.get("scheme", "musl"),
    )
    _need(method.type in METHOD_TYPES, f"method type must be one of {METHOD_TYPES}")
    _need(method.transfer in TRANSFER_TYPES, f"transfer must be one of {TRANSFER_TYPES}")
    _need(method.scheme in SCHEME_TYPES, f"scheme must be one of {SCHEME_TYPES}")

    materials = []
    for md in raw.get("materials", []):
        model = md.get("model")
        _need(model in ("linear-elastic", "neo-hookean", "eos-strength", "rigid"),
              f"material '{md.get('id')}': model must be linear-elastic, "
              "neo-hookean, eos-strength or rigid")
        if model == "eos-strength":
            # coerce numerics (YAML parses 1.4e6 as a string)
            eos = {k: (float(v) if k != "type" else v)
                   for k, v in (md.get("eos") or {}).items()}
            strength = {k: (float(v) if k != "type" else v)
                        for k, v in (md.get("strength") or {}).items()}
            materials.append(MaterialConfig(
                id=md["id"], model=model, eos=eos, strength=strength,
                rho=float(eos.get("rho0", 0.0)),
            ))
        elif model == "rigid":
            materials.append(MaterialConfig(
                id=md["id"], model=model,
                rho=float(md.get("rho", 1000.0)),
            ))
        else:
            materials.append(MaterialConfig(
                id=md["id"], model=model,
                rho=float(md["rho"]), E=float(md["E"]), nu=float(md["nu"]),
            ))

    regions = []
    for rd in raw.get("regions", []):
        regions.append(RegionConfig(
            id=rd["id"], type=rd["type"],
            axis=rd.get("axis", "z"),
            inner_radius=float(rd.get("inner_radius", 0.0)),
            exterior=bool(rd.get("exterior", False)),
            center=tuple(float(v) for v in rd.get("center", ())),
            size=tuple(float(v) for v in rd.get("size", ())),
            radius=float(rd.get("radius", 0.0)),
            height=float(rd.get("height", 0.0)),
        ))

    solids = []
    for sd in raw.get("solids", []):
        solids.append(SolidConfig(
            id=sd["id"], material=sd["material"], region=sd["region"],
            ppc=int(sd.get("ppc", 2)),
        ))

    bcs = []
    for bd in raw.get("boundary_conditions", []):
        t = bd.get("type")
        if t == "velocity_nodes":
            bcs.append(VelocityNodesBC(
                id=bd.get("id", t), region=bd["region"],
                vx=bd.get("vx"), vy=bd.get("vy"),
                vz=bd.get("vz") if dim == 3 else None,
            ))
        elif t == "body_force":
            g = tuple(_num_or_str(v) for v in bd.get("g", (0.0, 0.0, 0.0)))
            g = g + (0.0,) * (3 - len(g))
            bcs.append(BodyForceBC(id=bd.get("id", t), g=g))
        elif t == "force_nodes":
            f = tuple(float(v) for v in bd.get("force", (0.0, 0.0, 0.0)))
            bcs.append(ForceNodesBC(id=bd.get("id", t), region=bd["region"],
                                    force=f))
        elif t == "contact":
            bcs.append(ContactBC(id=bd.get("id", t), solid1=bd["solid1"],
                                 solid2=bd["solid2"],
                                 mu=float(bd.get("mu", 0.0))))
        elif t == "contact_hertz":
            bcs.append(ContactBC(id=bd.get("id", t), solid1=bd["solid1"],
                                 solid2=bd["solid2"], mu=0.0,
                                 type="contact_hertz"))
        elif t == "initial_stress":
            bcs.append(InitialStressBC(
                id=bd.get("id", t), region=bd["region"],
                s11=float(bd.get("s11", 0)), s22=float(bd.get("s22", 0)),
                s33=float(bd.get("s33", 0)), s12=float(bd.get("s12", 0)),
                s13=float(bd.get("s13", 0)), s23=float(bd.get("s23", 0))))
        elif t == "initial_velocity_nodes":
            bcs.append(InitialVelocityNodesBC(
                id=bd.get("id", t), region=bd["region"],
                vx=bd.get("vx"), vy=bd.get("vy"),
                vz=bd.get("vz") if dim == 3 else None))
        elif t == "velocity_particles":
            v = tuple(float(x) for x in bd.get("v", (0.0, 0.0, 0.0)))
            bcs.append(VelocityParticlesBC(id=bd.get("id", t),
                                           solid=bd["solid"], region=bd["region"],
                                           v=v))
        else:
            raise ConfigError(f"unknown boundary condition type '{t}'")

    ics = []
    for id_ in raw.get("initial_conditions", []):
        t = id_.get("type")
        if t == "initial_velocity" or t == "velocity":
            v = tuple(float(x) for x in id_.get("v", (0.0, 0.0, 0.0)))
            v = v + (0.0,) * (3 - len(v))
            ics.append(InitialVelocity(id=id_.get("id", t), select=id_["select"], v=v))
        else:
            raise ConfigError(f"unknown initial condition type '{t}'")

    od = raw.get("output", {})
    output = OutputConfig(
        interval=int(od.get("interval", 100)),
        particle_dump=od.get("particle_dump"),
        fields=tuple(od.get("fields", ("x", "y", "vx", "vy"))),
        log=tuple(od.get("log", ("kinetic_energy",))),
    )

    rd = raw.get("run", {})
    run = RunConfig(
        end_time=float(rd.get("end_time", 0.0)),
        end_step=int(rd.get("end_step", 0)),
        dt=float(rd["dt"]) if rd.get("dt") is not None else None,
        dt_factor=float(rd.get("dt_factor", 0.9)),
    )

    diagnostics = []
    for dd in raw.get("diagnostics", []):
        if dd.get("type") == "check_solution":
            u = tuple(str(x) for x in dd.get("u", ("0", "0", "0")))
            diagnostics.append(CheckSolution(id=dd.get("id", "error"), u=u))

    cfg = Config(
        dimension=dim, boxlo=boxlo, boxhi=boxhi, cellsize=cellsize,
        basis=basis, precision=precision, method=method,
        materials=materials, regions=regions, solids=solids,
        boundary_conditions=bcs, initial_conditions=ics,
        diagnostics=diagnostics, variables=raw.get("variables", {}) or {},
        output=output, run=run,
    )
    _validate_refs(cfg)
    return cfg


def _validate_refs(cfg: Config):
    mat_ids = {m.id for m in cfg.materials}
    reg_ids = {r.id for r in cfg.regions}
    for s in cfg.solids:
        _need(s.material in mat_ids, f"solid '{s.id}': unknown material '{s.material}'")
        _need(s.region in reg_ids, f"solid '{s.id}': unknown region '{s.region}'")
    for bc in cfg.boundary_conditions:
        if isinstance(bc, (VelocityNodesBC, ForceNodesBC, InitialStressBC,
                           InitialVelocityNodesBC)):
            _need(bc.region in reg_ids, f"BC '{bc.id}': unknown region '{bc.region}'")
    for ic in cfg.initial_conditions:
        _need(ic.select in reg_ids, f"IC '{ic.id}': unknown region '{ic.select}'")
    solid_ids = {sd.id for sd in cfg.solids}
    for bc in cfg.boundary_conditions:
        if isinstance(bc, VelocityParticlesBC):
            _need(bc.solid in solid_ids,
                  f"BC '{bc.id}': unknown solid '{bc.solid}'")
            _need(bc.region in reg_ids, f"BC '{bc.id}': unknown region '{bc.region}'")
