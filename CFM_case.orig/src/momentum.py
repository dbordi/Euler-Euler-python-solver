"""
Phase momentum closure: explicit convection/diffusion predictors for both
phases, drag coupled implicitly via a local 2x2 Partial Elimination
Algorithm (PEA). Drag coefficient from interfacial.drag_coeff(). See the
report for the PEA derivation and the Tomiyama drag law.
"""

import numpy as np
from . import operators as op
from . import interfacial


def _pad(F):
    P = np.empty((F.shape[0] + 2, F.shape[1] + 2))
    P[1:-1, 1:-1] = F
    P[:, 0] = P[:, 1]; P[:, -1] = P[:, -2]
    P[0, :] = P[1, :]; P[-1, :] = P[-2, :]
    P[1:-1, 0] = F[:, 0]; P[1:-1, -1] = F[:, -1]
    P[0, 1:-1] = F[0, :]; P[-1, 1:-1] = F[-1, :]
    return P


def segregated_pea(f, case, grid, dt):
    """
    Build the explicit per-phase predictors, then eliminate the implicit
    2x2 drag coupling into Hh1/Hh2 and mobilities D1, D2 such that
    U_k = Hh_k - D_k grad p (see the report for the elimination algebra).
    K_d is re-evaluated from the current velocities every call (Picard
    linearisation when n_outer > 1).

    Returns dict(Hh1u, Hh1v, Hh2u, Hh2v, D1, D2, K, ...), interior arrays.
    """
    c = case
    nu_l = c.fluids.mu_l / c.fluids.rho_l
    nu_g = c.fluids.mu_g / c.fluids.rho_g
    # rho_g is tiny, so mu_g/rho_g is a kinematic artefact rather than a real
    # dispersed-phase stress -- cap it at gas_visc_cap_factor*nu_l.
    fac = c.num.gas_visc_cap_factor
    if fac and fac > 0.0:
        nu_g = min(nu_g, fac * nu_l)
    nut = op.interior(f.nut)
    nu_eff_l = _pad(nu_l + nut)
    nu_eff_g = _pad(nu_g + nut)
    sch = c.num.convection

    a = op.interior(f.alpha)
    al = np.clip(1.0 - a, 1e-6, 1.0)

    # same floor as interfacial.drag_coeff(), so the gas stays gently
    # slaved to the liquid in empty cells rather than going unconstrained
    alpha_drag_floor = c.num.drag_alpha_floor
    ag = np.clip(a, alpha_drag_floor, None)
    rho_l, rho_g = c.fluids.rho_l, c.fluids.rho_g

    # explicit per-mass predictors Hhat_k = U_k^old + dt(adv + diff) (+ buoyancy)
    Hh1u = op.interior(f.u1) + dt * (op.advect(f.u1, f.u1, f.v1, grid, sch)
                                     + op.diffuse(f.u1, nu_eff_l, grid))
    Hh1v = op.interior(f.v1) + dt * (op.advect(f.v1, f.u1, f.v1, grid, sch)
                                     + op.diffuse(f.v1, nu_eff_l, grid))
    g2v = (rho_g - rho_l) / rho_g * c.gy            # gas buoyancy accel (p_rgh)

    ax_coll, ay_coll, Mx_coll, My_coll = interfacial.collisional_acceleration(f, c, grid)
    if c.solid_pressure.reaction_on_liquid:
        Hh1u += dt * (-Mx_coll / (al * rho_l))
        Hh1v += dt * (-My_coll / (al * rho_l))

    Hh2u = op.interior(f.u2) + dt * (op.advect(f.u2, f.u2, f.v2, grid, sch)
                                     + op.diffuse(f.u2, nu_eff_g, grid) + ax_coll)
    Hh2v = op.interior(f.v2) + dt * (op.advect(f.v2, f.u2, f.v2, grid, sch)
                                     + op.diffuse(f.v2, nu_eff_g, grid) + g2v + ay_coll)

    K = interfacial.drag_coeff(f, c)
    k1 = dt * K / (al * rho_l)
    k2 = dt * K / (ag * rho_g)
    det = 1.0 + k1 + k2

    def elim(H1, H2):
        Hh1 = ((1.0 + k2) * H1 + k1 * H2) / det
        Hh2 = (k2 * H1 + (1.0 + k1) * H2) / det
        return Hh1, Hh2

    e1u, e2u = elim(Hh1u, Hh2u)
    e1v, e2v = elim(Hh1v, Hh2v)

    D1 = dt * ((1.0 + k2) / rho_l + k1 / rho_g) / det
    D2 = dt * (k2 / rho_l + (1.0 + k1) / rho_g) / det

    return dict(Hh1u=e1u, Hh1v=e1v, Hh2u=e2u, Hh2v=e2v,
                D1=D1, D2=D2, K=K, drag_model="Tomiyama_contaminated",
                Mx_coll=Mx_coll, My_coll=My_coll)
