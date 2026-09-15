# Bubble-plume solver (2D Euler-Euler, electrolytic gas generation)

A finite-volume two-fluid (Euler-Euler) solver for the bubble plume rising
from a gas-evolving electrode in a narrow vertical channel, written for the
Computational Multiphase Flow course project (TU Delft).

Liquid phase: 6 M KOH electrolyte. Dispersed phase: hydrogen bubbles produced
at the electrode wall (x = 0) by Faraday's law, at a prescribed current
density. Gravity acts downward along y; the domain is a 2D vertical slice of
the channel gap.

## Requirements

Python 3.10+ with:

```
pip install -r requirements.txt
```

(numpy and scipy only -- the solver has no other dependencies.)

## Running a case

```
python run.py
```

Everything the solver needs is defined in two places:

- `config/case.py` -- the physical and numerical parameters (dataclasses),
  edited directly for a permanent change.
- the `USER KNOBS` block at the top of `run.py` -- the same parameters,
  convenient to change per run without touching `config/case.py`.

A run writes its results to `output/` (mesh, restart checkpoint, time
series, and `solver_log.csv`, one row every `PRINT_EVERY` iterations). The
folder starts empty in this repository; running `run.py` recreates it. If
`output/fields_latest.npz` already exists, `run.py` resumes from it
automatically -- delete `output/` (or set `RESTART_FROM = "none"` in
`run.py`) to start fresh.

## Physical model

- **Momentum**: segregated two-fluid velocities, coupled to a shared pressure
  field through a Partial Elimination Algorithm (PEA) for the interphase
  drag, inside a PISO/PIMPLE time-stepping loop.
- **Drag**: Tomiyama "contaminated" correlation (Schiller-Naumann at low
  Reynolds number, Eotvos-number correction for bubble deformation).
- **Turbulence**: off (laminar) or Prandtl mixing length with van Driest
  wall damping, selected by `turb.model`.
- **Bubble-induced turbulence**: Sato's model, added to the mixing-length
  eddy viscosity when `turb.bit_sato = True`.
- **Gas dispersion**: two optional contributions to the alpha diffusivity --
  turbulent (`nu_t / sigma_d`, zero when laminar) and hydrodynamic/bubble
  -induced (`C_hydro_disp * alpha_g * d_b * |U_g - U_l|`, nonzero even when
  laminar).
- **Solid (collisional) pressure**: a Johnson-Jackson-type force that
  resists packing of the gas phase at high volume fraction, applied
  everywhere in the domain (`solid_pressure` parameters).

Only the combinations above were exercised in the study this case belongs
to -- there is no unused alternative drag law or turbulence closure left in
the code.

## Convergence and residuals

This solver is explicit (PISO with adaptive sub-stepping), not an iterative
steady-state solver, so there is no single residual that drops to a
tolerance and stops the run. Convergence is tracked instead through
`output/solver_log.csv`:

- `Co_max`, `Fo_max` -- the Courant and Fourier numbers actually used that
  step, which should sit close to (never above) `cfl` and `diffusive_number`
  from `config/case.py`. A limiter's identity (`dt_limiter` column) tells
  you which term (convection, liquid/gas diffusion, or alpha dispersion) is
  restricting the time step.
- `dt_retries`, `dt_rejections_total` -- how often a step had to be redone
  at a smaller dt because it produced a non-finite field or an unphysical
  velocity spike. These should be near zero in a healthy run.
- `cont_rms`, `cont_max` -- the mixture (volume) continuity residual after
  the pressure projection. It should settle to a small, statistically
  steady band once the initial transient has passed, rather than keep
  growing; a slowly rising `cont_max` over a long run is worth a second
  look before trusting late-time statistics.
- `alpha_mean`, `alpha_sum` -- the domain-averaged and total gas holdup.
  Because gas is generated continuously and only leaves through the top
  outflow, these keep rising for a while even in a well-behaved run; look
  for them to approach a roughly constant growth rate (statistically
  stationary plume) rather than a plateau, since this is an unsteady,
  meandering flow and not a steady-state problem.

## Post-processing

`postproc/postprocess_all_plume.py` turns the snapshots in `output/` into
the figures and tables used in the report (see `postproc/README.txt`). It
needs a few extra packages (`postproc/requirements.txt`).

## Repository layout

```
config/     case parameters (config/case.py)
src/        solver modules (see src/README.txt)
postproc/   single-case post-processing script
run.py      entry point: python run.py
output/     results of a run (empty here, regenerated)
```
