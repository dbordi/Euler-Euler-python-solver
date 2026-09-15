"""
Gas volume-fraction transport: advection + optional dispersion + source.
D_disp = D_turb (nut/sigma_d, on/off via num.dispersion) + D_hydro
(C_hydro_disp*alpha_g*d_b*|U_g-U_l|, on/off via num.dispersion_hydro). See
config/case.py for the two switches and the report for the physics.
"""

import numpy as np
from . import operators as op
from . import boundary


def dispersion_diffusivity(f, case):
    """Total scalar dispersion diffusivity D_disp on the interior grid (ny, nx).

    Returns a (possibly zero) interior array.  Exposed so the time-step control
    can fold the largest D_disp into the diffusive stability limit.
    """
    c = case
    D = np.zeros(op.interior(f.alpha).shape)
    if c.num.dispersion:
        D = D + op.interior(f.nut) / c.turb.sigma_d
    if c.num.dispersion_hydro:
        a = op.interior(f.alpha)
        urel = np.sqrt((op.interior(f.u2) - op.interior(f.u1))**2
                       + (op.interior(f.v2) - op.interior(f.v1))**2)
        D = D + c.turb.C_hydro_disp * a * c.fluids.d_b * urel
    return D


def advance_alpha(f, case, grid, dt, mdot):
    a_old = op.interior(f.alpha)

    conv = op.advect(f.alpha, f.u2, f.v2, grid, case.num.convection)   # -div(alpha U_g)

    D = dispersion_diffusivity(f, case)
    if np.any(D):
        D_p = _pad_zero_grad(D)
        disp = op.diffuse(f.alpha, D_p, grid)                # div(D_disp grad alpha)
    else:
        disp = 0.0

    src = mdot / case.fluids.rho_g                            # volumetric gas source

    a_new = a_old + dt * (conv + disp + src)
    a_new = np.clip(a_new, 0.0, case.num.alpha_max)

    f.alpha[1:-1, 1:-1] = a_new
    boundary.apply_bc(f)
    return f.alpha


def _pad_zero_grad(F):
    """Pad an interior array with one zero-gradient ghost layer -> (ny+2, nx+2)."""
    P = np.empty((F.shape[0] + 2, F.shape[1] + 2))
    P[1:-1, 1:-1] = F
    P[:, 0] = P[:, 1]; P[:, -1] = P[:, -2]
    P[0, :] = P[1, :]; P[-1, :] = P[-2, :]
    # corners (set after edges are filled from interior; redo edges from interior)
    P[1:-1, 0] = F[:, 0]; P[1:-1, -1] = F[:, -1]
    P[0, 1:-1] = F[0, :]; P[-1, 1:-1] = F[-1, :]
    return P
