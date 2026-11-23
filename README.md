# warpflow-mpm

GPU-accelerated explicit Material Point Method (MPM) solver: **NVIDIA Warp**
kernels for intra-node compute, **MPI** domain decomposition for multi-node
parallelism. A Python port of the core of [Karamelo](https://github.com/karameloms/karamelo)
(C++/Eigen/MPI), restructured for GPUs.

```
mpirun -np 4 warpflow run configs/bouncing_balls.yaml
```

## Features

- **Methods**: UL-MPM, TL-MPM, **UL-CPDI** (updated Lagrangian CPDI-R4/Q4),
  explicit **USL / MUSL** schemes, **FLIP/PIC/APIC** transfers
  (APIC: affine C-matrix + inertia tensor Di, port of
  compute_velocity_nodes_APIC), 2D and 3D. TL grids are built per solid on
  the material bounding box.
- **Basis functions**: linear, quadratic-spline, cubic-spline, **Bernstein-
  quadratic** (h/2 node lattice with parity ntype) — including the `ntype`
  boundary-node corrections that keep splines partition-of-unity at domain
  edges.
- **Constitutive models** (unified Warp kernel, float64):
  linear-elastic, **neo-hookean** (`PK1 = G(F - F^-T) + lam ln(J) F^-T`),
  and **eos-strength** composites with
  EOS ∈ {linear, **fluid** (K(mu^gamma-1)), **shock** (Mie-Grueneisen +
  artificial viscosity)} x strength ∈ {linear, **fluid**, **plastic** (J2
  radial return), **johnson-cook**, **swift**} + **Johnson-Cook damage**
  (triaxiality-based failure initiation and stress degradation).
  Polar decomposition via Newton iteration (unique factor for det(F)>0).
- **Boundary / initial conditions**: Dirichlet velocity on grid node regions,
  **initial velocity nodes**, **nodal force (`force_nodes`)**, nodal body
  force (gravity, constant or **per-node expression** with x0/y0/z0/time),
  **initial stress** (sigma tensor from expressions), initial + prescribed
  particle velocities (**`velocity_particles`**), **rigid materials** (nodes
  touched by rigid solids become velocity boundaries for other solids), and
  **contact between solid pairs**: penalty (`min_penetration`) and **Hertz**
  (`contact_hertz`).
- **Diagnostics**: kinetic/strain energy, **check_solution** (manufactured-
  solution L2 error), **average_velocity**, **max_plastic_strain**.
- **Parallelism**: MPI ranks own axis-aligned subdomains (aspect-aware
  `procgrid` tiling, port of `Universe::set_proc_grid`); ghost-node exchange
  and particle migration each step; CFL `dt` via `MPI_MIN`; multi-GPU aware
  (`cuda:<rank % n_gpu>`).
- **Regions**: box / cylinder (axis x|y|z, **annulus** via inner_radius,
  **exterior** mode) / sphere.
- **GPU kernels**: all field math in float64 `wp.vec3d`/`wp.mat33d` kernels
  (matching C++ Eigen double precision), atomics for P2G scatter, on-the-fly
  shape functions (no CSR adjacency), dense padded node blocks with
  rect-based ghost exchange.
- **I/O**: LAMMPS-style per-rank particle dumps (`dump_p.*.LAMMPS`, same
  header format as Karamelo) + CSV energy log (kinetic/strain energy).

## Install

```bash
conda activate general_env           # Python >= 3.10, MPI, CUDA driver
pip install -e /path/to/warpflow-mpm # pulls warp-lang, mpi4py, numpy, pyyaml
```

Note: warpflow sets `wp.config.llvm_cuda = True` (override with
`WARPFLOW_LLVM_CUDA=0`) because the default NVRTC+PCH path hangs on some
setups (observed on WSL2 with an RTX 5060 / sm_120). The first launch
JIT-compiles the kernel module (~2 min with LLVM); it is cached afterwards.

## Usage

```bash
# single process (GPU)
warpflow run configs/bouncing_balls.yaml

# MPI multi-process (all ranks share the config; domain is split automatically)
mpirun -np 4 warpflow run configs/bouncing_balls.yaml

# tests
python -m pytest tests/ -q
```

Dumps are written per rank as `dump_p.proc-N.<step>.LAMMPS`; the energy log
goes to `warpflow.log.csv` (override with `--log`).

## Config format (YAML)

```yaml
dimension: 2
box: [[-1.0, -1.0], [1.0, 1.0]]     # lower / upper corner per axis
cellsize: 0.1
basis: linear                        # linear | quadratic-spline | cubic-spline
precision: float64                   # float32 not implemented yet
method:
  type: ulmpm                        # ulmpm | tlmpm
  transfer: flip                     # flip | pic (alpha = pic_flip)
  pic_flip: 0.99                     # 0 = pure PIC, 1 = pure FLIP
  scheme: musl                       # musl (recommended) | usl
materials:
  - {id: mat1, model: linear-elastic, rho: 1050, E: 1.0e6, nu: 0.3}
  # model: neo-hookean  (same fields)
  # model: eos-strength + eos: {type: fluid|shock|linear, ...}
  #                    + strength: {type: fluid|plastic|johnson-cook|swift|linear, ...}
  #                    + damage: {type: johnson-cook, d1..d3}
regions:
  - {id: rBall1, type: cylinder, center: [-0.5, -0.5], radius: 0.4}
  # types: box (center+size=half-sizes) | cylinder (axis: x|y|z) | sphere
solids:
  - {id: sBall1, material: mat1, region: rBall1, ppc: 2}
boundary_conditions:
  - {id: fixed, type: velocity_nodes, region: rBottom, vx: 0, vy: 0}
  - {id: grav,  type: body_force, g: [0, -9.8]}
  - {id: tip,   type: force_nodes, region: rTip, force: [0, -1300, 0]}
  - {id: ct,    type: contact, solid1: sA, solid2: sB, mu: 0.0}
initial_conditions:
  - {id: v0, type: initial_velocity, select: rBall1, v: [0.1, 0.1]}
output:
  interval: 20
  particle_dump: dump_p.*.LAMMPS
  fields: [x, y, vx, vy, s11, s22, s12, volume, mass]  # see dumps.py FIELD_NAMES
run:
  end_time: 3.5                      # or end_step: N
  dt_factor: 1.0                     # dt = dt_factor * h / (c_p + |v|inf)
  dt: null                           # set a constant dt to skip CFL adaptation
```

## Architecture

```
src/warpflow_mpm/
├── config.py        YAML -> dataclasses (validated)
├── universe.py      MPI ranks, procgrid tiling, subdomain split,
│                    neighbor offsets, O(1) owner-rank lookup
├── domain.py        regions (box/cylinder/sphere), lattice population
├── grid.py          per-rank padded node block, node-ownership rules
│                    (port of Grid::init), ghost-rect table, MPI exchange
├── particles.py     per-solid SoA Warp arrays + migration pack/unpack
├── basis.py         shape functions as @wp.func (f64, ntype-corrected)
├── kernels/
│   ├── core.py      p2g (mass/momentum/forces), g2p, grid update, BCs,
│   │                velocity gradient L, F/J/vol/D update, linear-elastic
│   │                stress, CFL dt, energies, owner computation
│   └── ghost.py     pack/unpack kernels for ghost-rect MPI exchange
├── materials.py     linear-elastic (registry for future models)
├── bcs.py           node BC lists, body force, initial velocities
├── dumps.py         LAMMPS-style particle dumps + energy log
├── solvers.py       ULMPM step (USL/MUSL) + particle migration (alltoall)
├── simulation.py    assembly + run loop + output cadence
└── cli.py           warpflow run config.yaml
```

### Parallel design notes

- **Node ownership** follows Karamelo's rules (`Grid::init`): a node falling
  exactly on a subdomain boundary belongs to the left/lower rank; the last
  rank's block extends so the global node grid is tiled exactly (tested).
- **Ghost exchange** uses dense rectangles instead of C++'s per-node lists:
  each rank exchanges `(my ghost ∩ neighbor owned)` rects with all 3^dim−1
  neighbors (including diagonals), in two phases — accumulate ghost
  contributions into owners, then overwrite ghosts with owner values —
  matching `Grid::reduce_ghost_nodes` semantics with contiguous, kernel-friendly
  transfers. Pair consistency is unit-tested.
- **Particle migration** keeps C++'s semantics (owner lookup by position,
  verify on arrival, append); it runs as a single `comm.alltoall` per step in
  which every rank participates (empty ranks included) to keep collectives
  ordered, and is skipped entirely when no particle crossed a subdomain.
- **MUSL re-scatter** uses start-of-step weights, exactly like C++: particle
  positions are snapshotted (`x_prev`) before the G2P move, and the gathers,
  re-scatter and velocity-gradient kernels all evaluate shape functions at
  `x_prev` — reproducing Karamelo's start-of-step weight semantics with a
  single GPU array copy per step.

## Validation

Against the C++ Karamelo build on the same machine (particle positions,
nearest-neighbour matching):

| Case (C++ example) | Agreement |
|---|---|
| `bouncing_balls.yaml` (Bouncing_balls/ULMPM, 1259 steps) | max ~7e-7 at every sampled step, 1-rank and MPI-4 **bit-for-bit identical** |
| `two_disks.yaml` (two-disks.mpm, FLIP=1, dt=0.01) | 1e-16 before contact, ~7e-7 through contact and bounce |
| `bouncing_balls2.yaml` (bouncing_balls2.mpm, dt=0.001, 3500 steps) | 1e-15 before contact, ~7e-7 after |
| `bouncing_balls_tlmpm.yaml` (TLMPM + penalty contact) | 1e-15 free flight, ~1e-7 through the contact |
| `dam_break.yaml` (EOS fluid + fluid strength, cubic-spline) | ~7e-7 through the whole collapse (C++ itself crashes at t=0.437) |
| `fsi.yaml` (EOS fluid + elastic obstacle, FLIP=1, 5000 steps) | 5e-6 -> 6e-4 over the run (chaotic free-surface amplification) |
| `cantilever.yaml` (TLMPM + neo-hookean, large deflection) | 2e-3 at step 200, ~4e-2 at step 2000 (C++ goes unstable at t=1.47 itself) |
| `taylor_bar.yaml` (shock EOS + Johnson-Cook plasticity) | 6e-3 -> 1.2e-2 through the plastic impact phase (~0.1% of bar length) |
| `bouncing_balls_apic.yaml` (TLMPM + APIC + contact) | 1e-15 free flight, energy conserved through contact |
| `compressed_ball.yaml` (rigid plate + velocity_particles) | plate position exact, ball compression stable (stress scale ~1.5-2x cpp) |
| `vertical_bar_bernstein.yaml` (TLMPM + NH + Bernstein basis) | runs stably; deviation ~0.45 cells at step 20 (see note) |
| `vertical_bar_cpdi_r4.yaml` (ULCPDI-R4 + NH) | runs stably (C++ reference itself crashes with J<=0) |
| `vertical_bar.yaml` (ULMPM gravity column) | fixed nodes hold to machine zero, ringing amplitude matches theory |

Additional checks (26 unit tests): total momentum conserved to 1e-9 over 60
steps; fixed-node BC holds to machine zero; 3D sphere matches 1 vs 2 ranks;
constitutive kernels match analytic values (linear elasticity, neo-hookean
volumetric/shear, J2 radial return, fluid/shock EOS pressure, polar
decomposition orthogonality); splines reproduce partition of unity at domain
edges.

## Known limitations

- `usl` scheme is provided but, exactly like the C++ code on the same input,
  is unstable for the bouncing-balls contact case (C++ aborts with `J<=0`);
  **use the default `musl`**.
- `float32` precision is accepted by the config but not implemented — kernels
  are float64 throughout.
- Thermal coupling is not ported: Johnson-Cook strength runs isothermal
  (T = Tr, thermal softening factor = 1), as in the C++ examples.
- No restart / checkpoint files yet.
- MLS transfer is not ported (FLIP / PIC / APIC are).
- Bernstein-quadratic basis is implemented (h/2 node lattice + ntype
  corrections) but agreement vs C++ shows a ~0.45-cell deviation at early
  steps (see Validation); other bases match to float64 round-off.
- Per-step particle migration does a host round-trip when particles cross
  subdomains; when they do not, the step is pure GPU.
- Multi-rank runs reproduce single-rank results bit-for-bit on the tested
  cases (ghost rect exchange + migration verified).

## Units and conventions

Same as Karamelo: SI by convention (whatever units you put in), box-aligned
grid with nodes at `boxlo + k*h`, global node tag `nz*ny*i + nz*j + k`,
LAMMPS-style dumps.
