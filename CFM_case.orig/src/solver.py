"""
PIMPLE solver for the 2D two-fluid bubble plume.

One step: pick dt from the CFL/diffusion limits, evaluate the gas source and
advance alpha explicitly, update nu_t, then run n_outer PIMPLE iterations
(each re-assembling the PEA momentum predictor and Tomiyama drag from the
current velocity, then n_corr PISO pressure correctors), finishing with
under-relaxation and a BC update.

With n_outer=1 (the default) this is just PISO. Larger n_outer re-linearises
convection and drag each outer pass, which helps nonlinear coupling but
doesn't relax the explicit dt limits -- raising cfl/dt_max still needs its
own sensitivity check.
"""

import numpy as np

from . import boundary, momentum, alpha_transport, interfacial
from . import operators as op
from . import properties_modes as properties
from .pressure import PressureSolver
from .source import GasSourceField


class Solver:
    def __init__(self, case, grid, fields):
        self.case  = case
        self.grid  = grid
        self.f     = fields
        self.psolver = PressureSolver(case, grid)
        self.gas     = GasSourceField(case, grid)
        self.t  = 0.0
        self.it = 0
        self.dt = case.num.dt_init
        self._dt_headroom = float(case.num.dt_max)   # adaptive ceiling, grows/shrinks
        self._last_S_vol = np.zeros(grid.shape)
        self._last_dt_limits = {}
        self._last_step_retries = 0
        self.rejected_steps_total = 0
        boundary.apply_bc(self.f)

    def _snapshot(self):
        f = self.f
        return (f.u1.copy(), f.v1.copy(), f.u2.copy(), f.v2.copy(),
                f.alpha.copy(), f.p.copy(), f.nut.copy())

    def _restore(self, snap):
        f = self.f
        (f.u1[:], f.v1[:], f.u2[:], f.v2[:], f.alpha[:], f.p[:], f.nut[:]) = snap

    def _step_acceptable(self, d):
        """Cheap a-posteriori check that a step did not misbehave."""
        if not d["finite"]:
            return False
        umax = self.case.num.dt_reject_umax
        if umax and umax > 0.0:
            if max(d["u_liq_max"], d["u_gas_max"],
                   d["v_liq_max"], d["v_gas_max"]) > umax:
                return False
        return True

    @staticmethod
    def _padded_zero_gradient(A):
        """Ghost-pad an interior scalar using zero-gradient values."""
        P = np.empty((A.shape[0] + 2, A.shape[1] + 2), dtype=float)
        P[1:-1, 1:-1] = A
        P[:, 0] = P[:, 1]; P[:, -1] = P[:, -2]
        P[0, :] = P[1, :]; P[-1, :] = P[-2, :]
        return P

    @staticmethod
    def _max_with_location(A):
        """Return max(A), (j,i); use (-1,-1) for an empty/zero operator."""
        if A.size == 0:
            return 0.0, (-1, -1)
        k = int(np.argmax(A))
        j, i = np.unravel_index(k, A.shape)
        value = float(A[j, i])
        if value <= 0.0:
            return 0.0, (-1, -1)
        return value, (int(j), int(i))

    def _local_convective_rate(self, Up, Vp):
        """Cell-local advective rate based on the actual face velocities.

        For a uniform one-directional flow this reduces to |U|/dx, while on a
        graded mesh each cell uses its own dx/dy. Taking the larger magnitude on
        the two faces is conservative for sign changes without double-counting
        a constant through-flow.
        """
        g = self.grid
        uf = op._lin_face_x(Up, g)
        vf = op._lin_face_y(Vp, g)
        rate = (np.maximum(np.abs(uf[:, :-1]), np.abs(uf[:, 1:]))
                / g.dx[None, :]
                + np.maximum(np.abs(vf[:-1, :]), np.abs(vf[1:, :]))
                / g.dy[:, None])
        return self._max_with_location(rate)

    def _local_diffusive_rate(self, Gamma, *, west, east, south, north):
        """Maximum diagonal rate of div(Gamma grad(.)) on the FV mesh.

        Boundary labels describe how a ghost value depends on its adjacent
        interior value:
          zero_gradient -> Fg=Fp, boundary diagonal multiplier 0
          fixed_ghost   -> Fg=constant, multiplier 1
          dirichlet     -> Fg=2*Fface-Fp, multiplier 2

        The explicit positivity bound is dt <= 1/max(diagonal_rate). The user
        safety factor ``diffusive_number`` is applied to that bound.
        """
        g = self.grid
        G = np.asarray(Gamma, dtype=float)
        if G.ndim == 0:
            G = np.full(g.shape, float(G))
        if G.shape != g.shape:
            raise ValueError(f"Diffusivity shape {G.shape} does not match grid {g.shape}")
        if not np.any(G > 0.0):
            return 0.0, (-1, -1)

        Gp = self._padded_zero_gradient(G)
        Gxf = op._lin_face_x(Gp, g)
        Gyf = op._lin_face_y(Gp, g)

        cw = Gxf[:, :-1] / (g.dx[None, :] * g.dxc[:-1][None, :])
        ce = Gxf[:, 1:]  / (g.dx[None, :] * g.dxc[1:][None, :])
        cs = Gyf[:-1, :] / (g.dy[:, None] * g.dyc[:-1][:, None])
        cn = Gyf[1:, :]  / (g.dy[:, None] * g.dyc[1:][:, None])
        rate = cw + ce + cs + cn

        multiplier = {"zero_gradient": 0.0,
                      "fixed_ghost": 1.0,
                      "dirichlet": 2.0}
        try:
            mw, me = multiplier[west], multiplier[east]
            ms, mn = multiplier[south], multiplier[north]
        except KeyError as exc:
            raise ValueError(f"Unknown diffusion boundary type: {exc.args[0]}") from exc

        # Replace the generic interior-neighbour multiplier (1) at boundaries.
        rate[:, 0]  += (mw - 1.0) * cw[:, 0]
        rate[:, -1] += (me - 1.0) * ce[:, -1]
        rate[0, :]  += (ms - 1.0) * cs[0, :]
        rate[-1, :] += (mn - 1.0) * cn[-1, :]
        return self._max_with_location(rate)

    @staticmethod
    def _dt_from_rate(number, rate):
        return float(number / rate) if rate > 0.0 else float("inf")

    def _local_fv_dt_limits(self):
        """Cell-local CFL and explicit-diffusion limits for all equations."""
        g, c, f = self.grid, self.case, self.f
        cfl = float(c.num.cfl)
        diff_no = float(c.num.diffusive_number)

        conv_l, loc_conv_l = self._local_convective_rate(f.u1, f.v1)
        conv_g, loc_conv_g = self._local_convective_rate(f.u2, f.v2)
        dt_conv_l = self._dt_from_rate(cfl, conv_l)
        dt_conv_g = self._dt_from_rate(cfl, conv_g)

        nu_l = c.fluids.mu_l / c.fluids.rho_l
        nu_g = c.fluids.mu_g / c.fluids.rho_g
        fac = c.num.gas_visc_cap_factor
        if fac and fac > 0.0:
            nu_g = min(nu_g, fac * nu_l)
        nut = op.interior(f.nut)

        # Liquid: no slip at x walls and at the bottom; zero gradient at top.
        diff_l, loc_diff_l = self._local_diffusive_rate(
            nu_l + nut, west="dirichlet", east="dirichlet",
            south="dirichlet", north="zero_gradient")

        # Gas-u: no normal penetration at x walls. At a closed bottom it is
        # free-slip (zero gradient); at an inlet u_face=0 (Dirichlet).
        gas_u_south = "dirichlet" if c.inlet.U_in > 0.0 else "zero_gradient"
        diff_gu, loc_diff_gu = self._local_diffusive_rate(
            nu_g + nut, west="dirichlet", east="dirichlet",
            south=gas_u_south, north="zero_gradient")
        # Gas-v: tangentially free-slip at x walls, prescribed/no-penetration at
        # the bottom, zero gradient at the outlet.
        diff_gv, loc_diff_gv = self._local_diffusive_rate(
            nu_g + nut, west="zero_gradient", east="zero_gradient",
            south="dirichlet", north="zero_gradient")
        if diff_gu >= diff_gv:
            diff_g, loc_diff_g = diff_gu, loc_diff_gu
        else:
            diff_g, loc_diff_g = diff_gv, loc_diff_gv

        D_disp = alpha_transport.dispersion_diffusivity(f, c)
        alpha_south = "fixed_ghost" if c.inlet.U_in > 0.0 else "zero_gradient"
        diff_a, loc_diff_a = self._local_diffusive_rate(
            D_disp, west="zero_gradient", east="zero_gradient",
            south=alpha_south, north="zero_gradient")

        dt_diff_l = self._dt_from_rate(diff_no, diff_l)
        dt_diff_g = self._dt_from_rate(diff_no, diff_g)
        dt_diff_a = self._dt_from_rate(diff_no, diff_a)
        candidates = {
            "cfl_liquid": dt_conv_l,
            "cfl_gas": dt_conv_g,
            "diff_liquid": dt_diff_l,
            "diff_gas": dt_diff_g,
            "diff_alpha": dt_diff_a,
            "dt_max": float(c.num.dt_max),
        }
        locations = {
            "cfl_liquid": loc_conv_l, "cfl_gas": loc_conv_g,
            "diff_liquid": loc_diff_l, "diff_gas": loc_diff_g,
            "diff_alpha": loc_diff_a, "dt_max": (-1, -1),
        }
        limiter = min(candidates, key=candidates.get)
        j, i = locations[limiter]
        x = float(g.xc[i]) if i >= 0 else float("nan")
        y = float(g.yc[j]) if j >= 0 else float("nan")
        return dict(
            mode="local_fv", stability_limiter=limiter,
            dt_stability=float(candidates[limiter]),
            dt_cfl=float(min(dt_conv_l, dt_conv_g)),
            dt_cfl_liquid=float(dt_conv_l), dt_cfl_gas=float(dt_conv_g),
            dt_diff_liquid=float(dt_diff_l), dt_diff_gas=float(dt_diff_g),
            dt_diff_alpha=float(dt_diff_a),
            conv_rate_max=float(max(conv_l, conv_g)),
            diff_rate_max=float(max(diff_l, diff_g, diff_a)),
            limiter_j=int(j), limiter_i=int(i),
            limiter_x_m=x, limiter_y_m=y)

    def _adaptive_dt(self):
        limits = self._local_fv_dt_limits()
        self._last_dt_limits = limits
        return limits["dt_stability"]

    def _choose_dt(self, dt_cap=None, use_headroom=True):
        stability = self._adaptive_dt()
        headroom = self._dt_headroom if use_headroom else float("inf")
        external = float(dt_cap) if dt_cap is not None else float("inf")
        if external <= 0.0:
            raise ValueError("dt_cap must be positive")
        selected = min(stability, headroom, external)
        reason = self._last_dt_limits["stability_limiter"]
        if headroom < stability and headroom <= external:
            reason = "retry_headroom"
        if external < stability and external < headroom:
            reason = "external_cap"
        self._last_dt_limits.update(
            dt_headroom=float(headroom), dt_external_cap=float(external),
            dt_selected=float(selected), actual_limiter=reason)
        return float(selected)

    def step(self, dt_cap=None):
        c = self.case
        # nu_t is an algebraic closure of the current fields. Update it before
        # selecting dt so the limiter never uses a one-step-old diffusivity.
        properties.update_nut(self.f, c, self.grid)
        self._last_step_retries = 0
        if not c.num.dt_adaptive_reject:
            self.dt = self._choose_dt(dt_cap=dt_cap, use_headroom=False)
            d = self._advance(self.dt)
            d["dt_retries"] = 0
            d["dt_rejections_total"] = int(self.rejected_steps_total)
            return d

        snap = self._snapshot()
        t0, it0 = self.t, self.it
        shrink = c.num.dt_reject_shrink
        grow = c.num.dt_grow
        max_retries = int(c.num.dt_max_retries)
        dt_min = float(c.num.dt_min)

        for attempt in range(max_retries + 1):
            # ceiling = physical stability limit, additionally capped by the
            # adaptive headroom (shrunk by previous rejections).
            self.dt = self._choose_dt(dt_cap=dt_cap, use_headroom=True)
            d = self._advance(self.dt)
            if self._step_acceptable(d):
                # clean step: let the headroom grow back toward dt_max
                self._dt_headroom = min(self._dt_headroom * grow, c.num.dt_max)
                self._last_step_retries = attempt
                d["dt_retries"] = int(attempt)
                d["dt_rejections_total"] = int(self.rejected_steps_total)
                return d
            # rejected: roll back and retry at a smaller dt
            self.rejected_steps_total += 1
            self._restore(snap)
            self.t, self.it = t0, it0
            self._dt_headroom = max(self.dt * shrink, dt_min)

        # Could not find an acceptable step; advance once more so the caller's
        # finite-check sees the failure and stops gracefully.
        self.dt = self._choose_dt(dt_cap=dt_cap, use_headroom=True)
        d = self._advance(self.dt)
        self._last_step_retries = max_retries + 1
        d["dt_retries"] = int(max_retries + 1)
        d["dt_rejections_total"] = int(self.rejected_steps_total)
        return d

    def _advance(self, dt):
        """Advance the solution by one step of size dt."""
        c, g, f = self.case, self.grid, self.f

        # nu_t was already updated before dt selection
        mdot = self.gas.rate(self.t)
        alpha_transport.advance_alpha(f, c, g, dt, mdot)

        S_vol = mdot * (1.0 / c.fluids.rho_g - 1.0 / c.fluids.rho_l)
        self._last_S_vol = S_vol.copy()

        self._pimple_segregated(dt, S_vol)

        boundary.apply_bc(f)
        self.t  += dt
        self.it += 1
        return self._diagnostics()

    def _pimple_segregated(self, dt, S_vol):
        """One or more PIMPLE outer iterations of the segregated PEA system.

        Each outer pass re-assembles H_hat from the current velocity (so the
        convection nonlinearity and the Tomiyama drag both see the latest
        field), runs n_corr PISO pressure correctors, then under-relaxes:
        U^{outer+1} = omega*U^{PISO} + (1-omega)*U^{outer,old}. omega=1
        (the default) skips relaxation entirely -- plain PISO.
        """
        c, g, f = self.case, self.grid, self.f
        alpha   = f.alpha[1:-1, 1:-1]
        relax_U = c.num.relax_U

        for outer in range(c.num.n_outer):

            # Store velocity at the start of this outer iteration for under-relaxation.
            # Only needed when relax_U < 1.0 AND more than one outer iteration.
            if relax_U < 1.0 and c.num.n_outer > 1:
                u1_outer = f.u1[1:-1, 1:-1].copy()
                v1_outer = f.v1[1:-1, 1:-1].copy()
                u2_outer = f.u2[1:-1, 1:-1].copy()
                v2_outer = f.v2[1:-1, 1:-1].copy()

            # PEA predictor: re-assembled from current U -> handles non-linearity
            # Also re-evaluates Tomiyama K_d(U_rel) from latest phase velocities.
            pea = momentum.segregated_pea(f, c, g, dt)

            # PISO inner correctors
            for _ in range(c.num.n_corr):
                u1, v1, u2, v2 = self.psolver.solve_segregated(
                    f, pea, alpha, S_vol, dt)
                f.u1[1:-1, 1:-1] = u1
                f.v1[1:-1, 1:-1] = v1
                f.u2[1:-1, 1:-1] = u2
                f.v2[1:-1, 1:-1] = v2
                boundary.apply_bc(f)

            # Under-relaxation at the end of the outer iteration.
            # omega * U_PISO + (1-omega) * U_outer_old
            if relax_U < 1.0 and c.num.n_outer > 1:
                f.u1[1:-1, 1:-1] = relax_U * f.u1[1:-1, 1:-1] + (1-relax_U) * u1_outer
                f.v1[1:-1, 1:-1] = relax_U * f.v1[1:-1, 1:-1] + (1-relax_U) * v1_outer
                f.u2[1:-1, 1:-1] = relax_U * f.u2[1:-1, 1:-1] + (1-relax_U) * u2_outer
                f.v2[1:-1, 1:-1] = relax_U * f.v2[1:-1, 1:-1] + (1-relax_U) * v2_outer
                boundary.apply_bc(f)

    def _diagnostics(self):
        f = self.f
        g = self.grid
        I = (slice(1, -1), slice(1, -1))

        a = f.alpha[I]
        u1 = f.u1[I]; v1 = f.v1[I]
        u2 = f.u2[I]; v2 = f.v2[I]
        p = f.p[I]
        nut = f.nut[I]

        ul = np.sqrt(u1*u1 + v1*v1)
        ug = np.sqrt(u2*u2 + v2*v2)
        slip = np.sqrt((u2-u1)**2 + (v2-v1)**2)

        # Mixture-volume continuity residual after projection:
        # div[(1-alpha)U_l + alpha U_g] - S_vol.
        qx = (1.0 - f.alpha) * f.u1 + f.alpha * f.u2
        qy = (1.0 - f.alpha) * f.v1 + f.alpha * f.v2
        cont = op.divergence(qx, qy, g) - self._last_S_vol
        cont_rms = float(np.sqrt(np.mean(cont**2)))
        cont_max = float(np.max(np.abs(cont)))

        p_coll = interfacial.collisional_pressure(a, self.case)
        axc, ayc, Mxc, Myc = interfacial.collisional_acceleration(f, self.case, g)
        Mmag = np.sqrt(Mxc*Mxc + Myc*Myc)
        amag = np.sqrt(axc*axc + ayc*ayc)
        p_coll_max = float(np.max(p_coll))
        M_coll_max = float(np.max(Mmag))
        a_coll_max = float(np.max(amag))

        finite = bool(
            np.all(np.isfinite(f.u1)) and np.all(np.isfinite(f.v1)) and
            np.all(np.isfinite(f.u2)) and np.all(np.isfinite(f.v2)) and
            np.all(np.isfinite(f.alpha)) and np.all(np.isfinite(f.p)) and
            np.all(np.isfinite(f.nut))
        )

        lim = self._last_dt_limits
        conv_rate = float(lim.get("conv_rate_max", 0.0))
        diff_rate = float(lim.get("diff_rate_max", 0.0))

        return dict(
            t=float(self.t), it=int(self.it), dt=float(self.dt),
            dt_limiter_mode=str(lim.get("mode", "unknown")),
            dt_limiter=str(lim.get("actual_limiter", lim.get("stability_limiter", "unknown"))),
            dt_stability=float(lim.get("dt_stability", float("nan"))),
            dt_cfl=float(lim.get("dt_cfl", float("nan"))),
            dt_cfl_liquid=float(lim.get("dt_cfl_liquid", float("nan"))),
            dt_cfl_gas=float(lim.get("dt_cfl_gas", float("nan"))),
            dt_diff_liquid=float(lim.get("dt_diff_liquid", float("nan"))),
            dt_diff_gas=float(lim.get("dt_diff_gas", float("nan"))),
            dt_diff_alpha=float(lim.get("dt_diff_alpha", float("nan"))),
            Co_max=float(self.dt * conv_rate),
            Fo_max=float(self.dt * diff_rate),
            dt_limiter_i=int(lim.get("limiter_i", -1)),
            dt_limiter_j=int(lim.get("limiter_j", -1)),
            dt_limiter_x_m=float(lim.get("limiter_x_m", float("nan"))),
            dt_limiter_y_m=float(lim.get("limiter_y_m", float("nan"))),
            dt_retries=int(self._last_step_retries),
            dt_rejections_total=int(self.rejected_steps_total),
            alpha_min=float(a.min()),
            alpha_max=float(a.max()),
            alpha_mean=float(a.mean()),
            alpha_sum=float(a.sum()),
            u_liq_max=float(ul.max()),
            u_gas_max=float(ug.max()),
            v_liq_max=float(np.abs(v1).max()),
            v_gas_max=float(np.abs(v2).max()),
            u1_rms=float(np.sqrt(np.mean(u1*u1))),
            v1_rms=float(np.sqrt(np.mean(v1*v1))),
            u2_rms=float(np.sqrt(np.mean(u2*u2))),
            v2_rms=float(np.sqrt(np.mean(v2*v2))),
            slip_max=float(slip.max()),
            slip_mean=float(slip.mean()),
            p_min=float(p.min()),
            p_max=float(p.max()),
            p_rms=float(np.sqrt(np.mean(p*p))),
            nut_max=float(nut.max()),
            p_coll_max=p_coll_max,
            M_coll_max=M_coll_max,
            a_coll_max=a_coll_max,
            cont_rms=cont_rms,
            cont_max=cont_max,
            finite=finite)
