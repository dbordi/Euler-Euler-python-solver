"""
Interfacial momentum exchange: Tomiyama 'contaminated' drag model.
drag_coeff() uses an algebraically expanded form of K_d that stays finite
as |U_g-U_l| -> 0 (the naive C_D*|Urel| form is 0*inf there); see the
report for the derivation and the Schiller-Naumann / Eotvos terms.
"""

import numpy as np
from . import operators as op


def eotvos(case):
    """Eo = (rho_l - rho_g) * g * d_b^2 / sigma."""
    fl = case.fluids
    return (fl.rho_l - fl.rho_g) * case.g * fl.d_b**2 / fl.sigma


def drag_coeff(f, case):
    """Volumetric drag coefficient K_d, interior grid [kg/(m^3 s)]. Called
    every PIMPLE outer iteration with the current velocity field.

    drag_alpha_floor keeps alpha_g from hitting exactly 0 (which would zero
    K_d and leave the gas velocity unconstrained); momentum.py uses the
    same floor for the gas drag number.
    """
    fl = case.fluids
    a = op.interior(f.alpha)
    a_eff = np.maximum(a, case.num.drag_alpha_floor)

    du = op.interior(f.u2) - op.interior(f.u1)
    dv = op.interior(f.v2) - op.interior(f.v1)
    Urel = np.sqrt(du**2 + dv**2)

    Re = fl.rho_l * Urel * fl.d_b / fl.mu_l
    Eo = eotvos(case)

    K_SN = (18.0 * fl.mu_l * a_eff / fl.d_b**2) * (1.0 + 0.15 * Re**0.687)
    K_Eo = 2.0 * fl.rho_l * (Eo / (Eo + 4.0)) * a_eff * Urel / fl.d_b
    return np.maximum(K_SN, K_Eo)


def terminal_slip(case):
    """Terminal rise velocity from the Tomiyama drag balance, solved by
    fixed-point iteration from the Stokes estimate."""
    fl = case.fluids
    Eo = eotvos(case)
    U_t = (fl.rho_l - fl.rho_g) * case.g * fl.d_b**2 / (18.0 * fl.mu_l)  # Stokes guess

    for _ in range(60):
        Re = fl.rho_l * U_t * fl.d_b / fl.mu_l
        cd_sn = (24.0 / max(Re, 1e-12)) * (1.0 + 0.15 * Re**0.687)
        cd_eo = (8.0 / 3.0) * Eo / (Eo + 4.0)
        CD = max(cd_sn, cd_eo)
        U_new = np.sqrt(4.0 / 3.0 * (fl.rho_l - fl.rho_g) * case.g * fl.d_b
                        / (fl.rho_l * CD))
        if abs(U_new - U_t) < 1e-12:
            break
        U_t = 0.5 * U_t + 0.5 * U_new   # damped update
    return U_t


def drag_info(case):
    """Print drag model diagnostics at terminal conditions."""
    fl = case.fluids
    Eo = eotvos(case)
    U_t_stokes = (fl.rho_l - fl.rho_g) * case.g * fl.d_b**2 / (18.0 * fl.mu_l)
    U_t = terminal_slip(case)
    Re_t = fl.rho_l * U_t * fl.d_b / fl.mu_l
    Re_safe = max(Re_t, 1e-12)
    cd_sn = (24.0 / Re_safe) * (1.0 + 0.15 * Re_safe**0.687)
    cd_eo = (8.0 / 3.0) * Eo / (Eo + 4.0)
    CD = max(cd_sn, cd_eo)
    tau_p = fl.rho_g * fl.d_b**2 / (18.0 * fl.mu_l)
    K_stokes = 18.0 * fl.mu_l / fl.d_b**2   # per unit alpha_g
    print(f"  Drag model   : Tomiyama contaminated (Eq. 2.7)")
    print(f"  Eo           = {Eo:.6f}  (C_D_Eo = {cd_eo:.5f}, negligible vs C_D_SN = {cd_sn:.3f})")
    print(f"  U_terminal   = {U_t*1e3:.3f} mm/s  (Stokes = {U_t_stokes*1e3:.3f} mm/s)")
    print(f"  Re_terminal  = {Re_t:.4f}")
    print(f"  C_D(Re_t)    = {CD:.3f}")
    print(f"  tau_p        = {tau_p:.2e} s")
    print(f"  K_d(Stokes)  = {K_stokes:.2f} kg/(m^3 s)  per unit alpha_g")

# -----------------------------------------------------------------------------
# Johnson--Jackson-type collisional / solid pressure (applied everywhere)
# -----------------------------------------------------------------------------

def collisional_pressure(alpha, case):
    """p_coll(alpha) [Pa] for the Johnson--Jackson-type law."""
    spm = case.solid_pressure
    amin = spm.alpha_min
    amax = spm.alpha_jj_max
    eps = max(spm.alpha_clip_eps, 1e-12)
    a = np.asarray(alpha)
    ae = np.clip(a, amin, amax - eps)
    da = np.maximum(ae - amin, 0.0)
    denom = np.maximum(amax - ae, eps)
    C = spm.gamma0 * case.fluids.rho_l * case.g * case.fluids.d_b
    p = C * da**spm.gamma1 / denom**spm.gamma2
    p = np.where(a > amin, p, 0.0)
    return p


def collisional_G(alpha, case):
    """G(alpha)=dp_coll/dalpha [Pa] for M_coll = -G grad(alpha)."""
    spm = case.solid_pressure
    amin = spm.alpha_min
    amax = spm.alpha_jj_max
    eps = max(spm.alpha_clip_eps, 1e-12)
    a = np.asarray(alpha)
    ae = np.clip(a, amin, amax - eps)
    da = np.maximum(ae - amin, 0.0)
    denom = np.maximum(amax - ae, eps)
    C = spm.gamma0 * case.fluids.rho_l * case.g * case.fluids.d_b
    g1, g2 = spm.gamma1, spm.gamma2

    term1 = 0.0 if g1 == 0 else g1 * da**max(g1 - 1.0, 0.0) / denom**g2
    term2 = g2 * da**g1 / denom**(g2 + 1.0)
    G = C * (term1 + term2)
    G = np.where(a > amin, G, 0.0)
    return G


def collisional_force(f, case, grid):
    """Volumetric solid-pressure force on gas, (Mx, My) [N/m^3]."""
    a = op.interior(f.alpha)
    G = collisional_G(a, case)
    grad_ax = op.grad_x(f.alpha, grid)
    grad_ay = op.grad_y(f.alpha, grid)

    Mx = -G * grad_ax
    My = -G * grad_ay
    return Mx, My


def collisional_acceleration(f, case, grid):
    """Gas acceleration due to solid pressure, capped [m/s^2]."""
    Mx, My = collisional_force(f, case, grid)
    a = op.interior(f.alpha)
    ag = np.maximum(a, case.num.drag_alpha_floor)
    ax = Mx / (ag * case.fluids.rho_g)
    ay = My / (ag * case.fluids.rho_g)

    cap = case.solid_pressure.accel_cap
    if cap > 0.0:
        mag = np.sqrt(ax*ax + ay*ay)
        scale = np.minimum(1.0, cap / (mag + 1e-30))
        ax *= scale
        ay *= scale
        Mx = ax * ag * case.fluids.rho_g
        My = ay * ag * case.fluids.rho_g

    return ax, ay, Mx, My


def solid_pressure_info(case):
    spm = case.solid_pressure
    print("  Solid pressure: Johnson-Jackson type, applied everywhere")
    print(f"    gamma0={spm.gamma0:g}, gamma1={spm.gamma1:g}, gamma2={spm.gamma2:g}")
    print(f"    alpha_min={spm.alpha_min:g}, alpha_jj_max={spm.alpha_jj_max:g}, accel_cap={spm.accel_cap:g} m/s^2")
