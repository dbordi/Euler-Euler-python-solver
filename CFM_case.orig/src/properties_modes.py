"""
Turbulence closure (case.turb.model): "laminar" sets nu_t=0; "mixing_length"
uses Prandtl mixing length (l_m = kappa*y_wall) with van Driest damping.
Sato bubble-induced turbulence (BIT) is added on top via case.turb.bit_sato.
"""

import numpy as np
from . import operators as op


def _strain_magnitude(f, grid):
    dudx = op.grad_x(f.u1, grid)
    dudy = op.grad_y(f.u1, grid)
    dvdx = op.grad_x(f.v1, grid)
    dvdy = op.grad_y(f.v1, grid)
    # |S| = sqrt(2 S_ij S_ij)
    return np.sqrt(2.0 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2) + 1e-30


def _wall_shear_v(f, grid):
    """Wall-parallel (v) velocity gradient at the nearest x-wall, per row -> (ny,)."""
    v1 = f.v1[1:-1, 1:-1]
    g_left = np.abs(v1[:, 0]) / (0.5 * grid.dx[0] + 1e-30)
    g_right = np.abs(v1[:, -1]) / (0.5 * grid.dx[-1] + 1e-30)
    return np.maximum(g_left, g_right)


def _sato_bit(f, case):
    if not case.turb.bit_sato:
        return 0.0
    a = op.interior(f.alpha)
    urel = np.sqrt((op.interior(f.u2) - op.interior(f.u1))**2
                   + (op.interior(f.v2) - op.interior(f.v1))**2)
    return case.turb.C_mu_bit * a * case.fluids.d_b * urel


def update_nut(f, case, grid):
    c = case
    nu_l = c.fluids.mu_l / c.fluids.rho_l

    # --- laminar: no eddy viscosity at all ---
    if c.turb.model == "laminar":
        f.nut[:, :] = 0.0
        return f.nut

    kappa, A_plus = c.turb.kappa, c.turb.A_plus
    yw = grid.wall_dist                                    # (ny, nx)
    Smag = _strain_magnitude(f, grid)

    # van Driest damping from a near-wall friction velocity estimate
    tau_w = c.fluids.mu_l * np.abs(_wall_shear_v(f, grid))  # (ny,)
    u_tau = np.sqrt(tau_w / c.fluids.rho_l) + 1e-12         # (ny,)
    yplus = yw * (u_tau[:, None]) / nu_l
    damping = 1.0 - np.exp(-yplus / A_plus)

    l_m = kappa * yw * damping
    nut_shear = l_m**2 * Smag

    nut = nut_shear + _sato_bit(f, c)
    nut = np.minimum(nut, c.turb.nut_max_factor * nu_l)    # stability cap

    f.nut[1:-1, 1:-1] = nut
    f.nut[:, 0] = f.nut[:, 1]; f.nut[:, -1] = f.nut[:, -2]
    f.nut[0, :] = f.nut[1, :]; f.nut[-1, :] = f.nut[-2, :]
    return f.nut


def plume_reynolds(f, case, grid):
    """Diagnostic only: bulk plume Reynolds number Re_p = v_max*b/nu_l at
    mid-height (v_max peak liquid velocity, b gas-weighted plume half-width).
    """
    nu_l = case.fluids.mu_l / case.fluids.rho_l
    v1 = f.v1[1:-1, 1:-1]
    a = op.interior(f.alpha)
    jmid = v1.shape[0] // 2
    vmax = np.abs(v1[jmid]).max()
    xc = grid.xc
    xcg = (a[jmid] * xc).sum() / (a[jmid].sum() + 1e-30)
    b = np.sqrt((a[jmid] * (xc - xcg)**2).sum() / (a[jmid].sum() + 1e-30))
    return float(vmax * max(b, grid.dx.min()) / nu_l)
