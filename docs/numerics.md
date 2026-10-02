# Numerical assumptions and verified cases

The released PDE datasets map an initial field to one final-time field.
The tensor loader converts inputs and targets to contiguous `float32` tensors.
Models use channels last; spatial axes are `[N]` in 1D and `[height, width]`
(`y`, then `x`) in 2D. The [code guide](code-guide.md) describes the losses
and [annotated advection example](../configs/examples/advection_quickstart.yaml).

## Equations, domains, and time steps

The generators in [`data/generators.py`](../src/otno/data/generators.py) use
unit domains. Setting `symmetry.length` changes the transformation, not the
generated domain. For another physical length, extend the generator and pass
that length consistently to its field construction, solver, and symmetry.
Galilean `symmetry.final_time` must match `dataset.final_time` because the
output shift is boost times elapsed time. The configuration validator checks
individual values; it does not certify these physical cross-field agreements.

| Dataset | Equation and numerical method | Boundary and reference settings |
| --- | --- | --- |
| `advection1d` | $u_t+c u_x=0$; Fourier shift by $cT$, with no time integrator | Periodic `[0,1)`; quickstart `N=32`, `T=0.25`, `c=0.7` |
| `burgers1d` | $u_t+u u_x=\nu u_{xx}$; pseudo-spectral derivatives and explicit RK4 | Periodic `[0,1)`; reusable config `N=128`, `T=0.5`, `nu=0.01`, requested `dt=0.001` |
| `navier_stokes_vorticity2d` and `_boosted` | $\omega_t+v\cdot\nabla\omega=\nu\Delta\omega$; pseudo-spectral derivatives and explicit RK4 | Periodic unit square; release `N=64`, `T=0.5`, `nu=0.001`, requested `dt=0.001`, no forcing |
| `heat1d_dirichlet` | $u_t=\kappa u_{xx}$; project onto sine modes and multiply by $e^{-\kappa(\pi k)^2 T}$ | Zero Dirichlet values at `x=0,1`; `N=64`, `T=0.1`, `kappa=0.01`; no time integrator |

`solve_advection_1d` and `solve_burgers_1d` are in
[`solvers1d.py`](../src/otno/data/solvers1d.py);
`solve_navier_stokes_vorticity_2d` is in
[`solvers2d.py`](../src/otno/data/solvers2d.py);
`solve_heat_1d_dirichlet` is in the generators module. Advection is exact for
the represented Fourier modes subject to floating-point and Nyquist effects.
Heat uses a finite sine projection: `modes` controls the initial field and
`solve_modes` controls the retained solution modes; keep them resolved by the
grid (below `N-1`).

RK4 uses `steps=max(1,ceil(T/dt))` and actual step `h=T/steps`, reaching exactly
the requested final time. It has no adaptive stability control. The nonlinear
term is filtered with the tensor-product 2/3 rule when `dealias: true`; the
viscous term remains explicit. Changing resolution, amplitude, viscosity,
boost radius, or final time can change stability and integration error. A
positive `dt` and finite output alone do not establish accuracy. For changed
settings, compare held-out fields at `dt`, `dt/2`, and `dt/4`, and check
transformation closure against the solver. Use a new dataset path when changing
generation parameters; existing cache metadata is validated.

The Navier–Stokes implementation uses $\Delta\psi=\omega$ and
$v=(\psi_y,-\psi_x)+b$, as defined in `navier_stokes_vorticity_rhs_2d`.
It sets the zero Fourier mode of the streamfunction to zero; generated
vorticity fields have zero mean. This sign convention matters when comparing
external velocity/vorticity data. The RHS accepts a supplied forcing tensor,
but released generators use unforced dynamics. A fixed forcing generally
needs its own transformed action to preserve the declared symmetries.

The local convergence tests in
[`test_solver_validation.py`](../tests/test_solver_validation.py) cover smooth,
small-amplitude Burgers `N=64,T=0.02` and Navier–Stokes `N=16,T=0.01`, comparing
`dt=0.002` against `0.001` with relative discrepancy below `2e-4`. They do not
prove convergence for every accepted configuration. The solver
validation command records convergence and closure diagnostics; the closure
table command evaluates fields from the problem configuration:

```bash
python scripts/validate_solvers.py
python scripts/make_solver_closure_tables.py --config configs/problems/2d_navier_stokes_galilean.yaml
```

## Boundaries and transformations

Periodic grids sample `j/N` and omit the repeated endpoint. Fourier shifts
implement `f(x-shift)`; positive shifts move features toward increasing
coordinates. Two-dimensional vectors are `[x,y]`. The real FFT representation
on an even grid has a special Nyquist bin: arbitrary sub-grid shifts need not
round-trip exactly when substantial energy occupies that bin. The verified
round-trip tests use smooth fields with low Fourier content. Keep initial modes
below Nyquist and check finer-grid behavior for rough fields.

Translations act on both initial and final periodic fields. Burgers Galilean
actions add a scalar boost to the initial velocity and apply
`u(x-boost*T)+boost` to the output. Boosted Navier–Stokes actions add a velocity
increment to its two ambient-velocity input channels and shift the final
vorticity by that increment times `T`, leaving initial vorticity unchanged.

For vorticity D4 reflections, use `d4_vorticity2d` (alias of
`D4Pseudoscalar2D`), which flips the sign under reflection. `d4_scalar2d`
permutes scalar fields without changing sign. The vorticity action applies its
sign to every channel: the verified D4 setup therefore uses one vorticity
channel. Applying it directly to `[omega,b_x,b_y]` would fail to rotate the
ambient velocity as a vector. A combined boost/D4 experiment requires a new
paired action and tests.

The heat grid includes both endpoints using `linspace(0,1,N)`. Its translation
uses linear interpolation with zero padding, not a periodic shift. With
`use_mask: true`, comparison includes only output positions whose source
coordinate `x-shift` lies within `[mask_margin,1-mask_margin]`; the binary mask
broadcasts over channels. **Translations do not preserve a fixed-wall heat
problem exactly.** Removing padded entries limits the comparison to common
support, but boundary influence can still change interior solutions. This is
a boundary diagnostic, not an exact symmetry claim. See
[`1d_heat_dirichlet_common_mask_evaluation.yaml`](../configs/ablations/1d_heat_dirichlet_common_mask_evaluation.yaml)
for the common evaluation protocol.

Paired actions and masks are implemented in
[`symmetry/transforms.py`](../src/otno/symmetry/transforms.py), with shape,
axis, reflection, padding, and mask checks in
[`test_transforms.py`](../tests/test_transforms.py). Solver identities for
advection translation, Burgers boosts, and Navier–Stokes translations/boosts
are checked in [`test_symmetry_identities.py`](../tests/test_symmetry_identities.py);
the D4 vorticity closure check is in the solver validation tests.

## Channel meanings and grid restrictions

| Case | Input → output channels | Restrictions |
| --- | --- | --- |
| Scalar PDEs | `[initial field]` → `[final field]` | Channels-last tensors; generated 2D PDE grids are square |
| Boosted Navier–Stokes | `[omega0,b_x,b_y]` → `[omega(T)]` | `b_x,b_y` are spatially constant; model has `in_channels=3,out_channels=1`; action channel indices must match |
| rMD17 forces | `[x,y,z,Z/charge_scale]` → `[F_x,F_y,F_z]` per atom | Fixed atom count/order; ethanol reference has 9 atoms, positions centered, `charge_scale=10`; force vectors rotate, charge feature does not |

The released `molecule_mlp` predicts forces directly from positions and species
features. It does not derive forces from an energy or enforce energy
conservation. Molecular rigid motion rotates positions/forces and translates
positions only. The bundled ethanol split fixes samples through `old_indices`;
its raw-source and split verification are described in [Reproduction](reproducibility.md).

FNO1d/FNO2d accept varying grid sizes, but this does not establish resolution
transfer. Retained modes are clipped to available bins; in FNO2d keep
`2*modes1 <= height` to avoid overlapping positive/negative row bands, and
`modes2 <= width//2+1`. The quickstart uses six 1D modes on 32 points.
`add_grid: true` appends absolute coordinates and can break translation
equivariance; its 1D grid also follows periodic `j/N`, even when modeling heat
targets sampled on an endpoint-inclusive grid.

DeepONet2d requires the configured fixed sensor grid (`grid_height`,
`grid_width`). CNO2d requires both spatial dimensions divisible by
`2**n_layers` and uses circular padding/resampling. D4 GFNO experiments use
square grids, `modes1 == modes2`, and `add_grid: false`; its centered frequency
selection excludes unpaired even-grid Nyquist bins. Canonical translation
frames rely on a usable first Fourier mode and can be ambiguous near a zero
first-mode amplitude; observable Galilean canonicalization reads the constant
boost channels. See [`models/`](../src/otno/models/),
[`test_fno_shapes.py`](../tests/test_fno_shapes.py),
[`test_cno.py`](../tests/test_cno.py), and
[`test_canonical.py`](../tests/test_canonical.py).

## Verified combinations

Configuration construction checks and numerical identities are distinct from
training benchmarks. The following combinations have concrete evidence; new
cross-products of backbone, symmetry, boundary, and data need their own checks.

| Combination | Evidence and scope |
| --- | --- |
| Periodic advection + FNO1d + translation + `aug_orbit` | Runnable [quickstart](../configs/examples/advection_quickstart.yaml); exact represented-mode solver identity and CPU training/resume tests |
| Periodic Burgers + FNO1d / canonical FNO1d + translation/Galilean mixture | [Problem config](../configs/problems/1d_burgers_translation_galilean.yaml), [four-method seed sweep](../configs/ablations/1d_final_four_method_seeds.yaml), short-time solver identity |
| Boosted periodic Navier–Stokes N64 + FNO2d + Galilean actions | [Problem config](../configs/problems/2d_navier_stokes_galilean.yaml) and [release suites](../configs/release.yaml): augmentation/orbit controls, tangent objectives, semi-supervised training, FNO adaptation, observable canonicalization; solver boost identity |
| Boosted periodic Navier–Stokes N64 + DeepONet2d or CNO2d + Galilean finite-orbit training | [DeepONet sweep](../configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml), [CNO sweep](../configs/ablations/2d_galilean_n64_2pct_cno2d_5seed.yaml); parameter-gradient and reporting tests |
| Unboosted periodic Navier–Stokes N64 + FNO2d or D4 GFNO + vorticity D4 | [D4 completion sweep](../configs/ablations/2d_d4_five_seed_completion.yaml), D4 closure and GFNO equivariance tests; translation/D4 mixture is constructible in the [problem config](../configs/problems/2d_navier_stokes_translation_d4.yaml) |
| Dirichlet heat + FNO1d + masked nonperiodic translation | [Common-mask evaluation](../configs/ablations/1d_heat_dirichlet_common_mask_evaluation.yaml); interpolation/mask and data-generation tests; approximate boundary diagnostic |
| rMD17 ethanol + molecule MLP + molecular actions | [Problem config](../configs/problems/rmd17_ethanol_force_500.yaml), [control sweep](../configs/ablations/rmd17_ethanol_500_controls.yaml), force/species transformation and split tests; fixed-molecule finite objectives |

Tangent implementations are available for periodic 1D/2D translations and
Burgers/Navier–Stokes boosts. D4, nonperiodic interpolation, and molecular
actions lack tangent implementations and are rejected before training; see
[`test_config_validation.py`](../tests/test_config_validation.py).
Partial adaptation parameter selection is verified for FNO projectors and
final blocks. GPU portability, arbitrary forcing, non-square solver domains,
variable atom counts, other molecules, and combined vector/D4 channels are
not established by these release experiments. Finite-orbit consistency adds
an empirical objective; hard equivariance is supplied only by specific
backbones/wrappers and their stated assumptions.
