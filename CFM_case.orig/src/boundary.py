"""
Boundary conditions (ghost-cell filling). x=0 is the electrode wall, y=0
the bottom.

If CASE.inlet.U_in <= 0 the bottom is a closed wall; otherwise it becomes a
velocity inlet (liquid u_l=0, v_l=U_in; gas u_g=0, v_g=U_in, alpha_g=
alpha_g_in). Pressure keeps zero normal gradient at the inlet; the top
pressure is anchored in the pressure solver.
"""

import numpy as np
from config.case import CASE


def bottom_inflow_velocity() -> float:
    """Positive upward inlet velocity, or 0.0 when the bottom is a wall."""
    return max(0.0, float(CASE.inlet.U_in))


def bottom_inflow_active():
    """True when the bottom should be treated as an inlet rather than a wall."""
    return bottom_inflow_velocity() > 0.0


def apply_bc(f):
    _bc_velocity(f)
    _bc_alpha(f)
    _bc_pressure(f)


def _bc_velocity(f):
    inlet_on = bottom_inflow_active()
    Uin = bottom_inflow_velocity()

    # ---- liquid ----
    # left/right vertical walls: no-slip for both components
    for U in (f.u1, f.v1):
        U[:, 0] = -U[:, 1]
        U[:, -1] = -U[:, -2]

    if inlet_on:
        # bottom velocity inlet. Ghost values impose the face value by averaging:
        # U_face = 0.5*(U_ghost + U_inside).
        f.u1[0, :] = -f.u1[1, :]              # u_l,face = 0
        f.v1[0, :] = 2.0 * Uin - f.v1[1, :]  # v_l,face = Uin
    else:
        # closed bottom wall: no-slip liquid
        f.u1[0, :] = -f.u1[1, :]
        f.v1[0, :] = -f.v1[1, :]

    # top outflow: zero gradient
    f.u1[-1, :] = f.u1[-2, :]
    f.v1[-1, :] = f.v1[-2, :]

    # ---- gas ----
    # left/right walls: free-slip gas, no normal penetration
    f.u2[:, 0] = -f.u2[:, 1]
    f.u2[:, -1] = -f.u2[:, -2]
    f.v2[:, 0] = f.v2[:, 1]
    f.v2[:, -1] = f.v2[:, -2]

    if inlet_on:
        # bottom inlet. Since alpha_g_in is usually zero, this does not inject
        # gas volume, but keeps the dispersed-phase velocity well-conditioned.
        f.u2[0, :] = -f.u2[1, :]              # u_g,face = 0
        f.v2[0, :] = 2.0 * Uin - f.v2[1, :]  # v_g,face = Uin
    else:
        # closed bottom wall: free-slip gas, no normal penetration
        f.v2[0, :] = -f.v2[1, :]
        f.u2[0, :] = f.u2[1, :]

    # top outflow: zero gradient
    f.u2[-1, :] = f.u2[-2, :]
    f.v2[-1, :] = f.v2[-2, :]


def _bc_alpha(f):
    a = f.alpha
    inlet_on = bottom_inflow_active()

    # vertical walls: zero gradient
    a[:, 0] = a[:, 1]
    a[:, -1] = a[:, -2]

    if inlet_on:
        # robust fixed inflow alpha for upwind/TVD advection; normally 0.0
        a[0, :] = CASE.inlet.alpha_g_in
    else:
        # closed bottom wall: zero gradient
        a[0, :] = a[1, :]

    # top outflow: zero gradient
    a[-1, :] = a[-2, :]

    # keep ghosts physical
    np.clip(a, 0.0, CASE.num.alpha_max, out=a)


def _bc_pressure(f):
    p = f.p
    # side walls and bottom inlet/wall: zero normal pressure gradient
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    p[0, :] = p[1, :]
    # top outflow: Dirichlet is imposed in the pressure matrix; mirror here
    p[-1, :] = p[-2, :]
