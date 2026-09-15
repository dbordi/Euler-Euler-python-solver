Solver source. One module per concern, imported by src.solver.Solver.

  mesh.py              cell-centred finite-volume grid, with optional
                        geometric refinement near the electrode wall
  fields.py             the ghost-padded state arrays (u, v, alpha, p, nut)
  operators.py           gradient, divergence, diffusion and convection
                        (upwind / van Leer TVD) on the grid
  boundary.py            ghost-cell boundary conditions for every field
  source.py               electrolytic gas generation term (Faraday's law)
  alpha_transport.py     gas volume-fraction transport equation
  interfacial.py         Tomiyama drag law and the Johnson-Jackson solid
                        (collisional) pressure model
  properties_modes.py    turbulent eddy viscosity: laminar or Prandtl
                        mixing length, plus optional Sato bubble-induced
                        turbulence
  momentum.py             segregated two-fluid momentum predictor with the
                        Partial Elimination Algorithm (PEA) for drag
  pressure.py             variable-coefficient pressure Poisson equation
                        (Rhie-Chow face fluxes) and the phase velocity
                        correction
  solver.py                PIMPLE/PISO time-stepping loop, adaptive time
                        step, and per-step diagnostics
  restart.py              checkpoint save/load for resuming a run
  diagnostics.py          post-hoc plume/boundary-layer measures used by
                        the post-processing scripts

There is only one momentum closure (segregated PEA), one drag law (Tomiyama,
contaminated system), and two turbulence modes (laminar, mixing length) --
these are the only combinations that were used in the study this case
belongs to.
