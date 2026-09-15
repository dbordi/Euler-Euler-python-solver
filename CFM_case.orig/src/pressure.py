"""
Pressure-velocity coupling for the segregated two-fluid (PEA) momentum closure.

Both phase velocities are eliminated against the drag coupling in
momentum.segregated_pea() into per-phase mobilities D1, D2. The mixture
(volume-weighted) predicted velocity and mobility

    Q  = (1-alpha) Hh1 + alpha Hh2
    Dp = (1-alpha) D1  + alpha D2

feed a variable-coefficient Poisson equation for p (Rhie-Chow face fluxes
avoid checkerboarding on the collocated grid):

    div( Dp grad p ) = div(Q) - S_vol

Both phase velocities are then corrected with the same pressure field:
U_k = Hh_k - D_k grad p. p is the DYNAMIC pressure p_rgh (uniform hydrostatic
part removed); the top outflow uses a Dirichlet anchor p = p_ref, side walls
and the bottom inlet/wall use homogeneous Neumann.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from . import operators as op


class PressureSolver:
    def __init__(self, case, grid):
        self.case = case
        self.grid = grid
        self.ny, self.nx = grid.ny, grid.nx
        self.N = self.nx * self.ny
        self.dx, self.dy, self.dxc, self.dyc = grid.dx, grid.dy, grid.dxc, grid.dyc

    def _matrix_var(self, De, Dw, Dn, Ds, cN_top):
        """Assemble the compact variable-coefficient Poisson matrix."""
        nx = self.nx
        g = self.grid
        dx, dy, dxc, dyc = self.dx, self.dy, self.dxc, self.dyc
        Ae = De * (dy[:, None] / dxc[None, 1:]) / g.Vol
        Aw = Dw * (dy[:, None] / dxc[None, :-1]) / g.Vol
        An = Dn * (dx[None, :] / dyc[1:, None]) / g.Vol
        As = Ds * (dx[None, :] / dyc[:-1, None]) / g.Vol
        Aw[:, 0] = 0.0; Ae[:, -1] = 0.0; As[0, :] = 0.0
        An[-1, :] = 0.0
        aP = -(Aw + Ae + As + An)
        aP[-1, :] -= cN_top
        N = self.N
        main = aP.ravel()
        east = Ae.ravel(); west = Aw.ravel(); north = An.ravel(); south = As.ravel()
        A = sp.diags([main, east[:-1], west[1:], north[:-nx], south[nx:]],
                     [0, 1, -1, nx, -nx], format='csr')
        return A

    def solve_segregated(self, f, pea, alpha, S_vol, dt):
        """
        PEA projection: build the mixture predicted flux Q and mobility Dp from
        the eliminated phase predictors, solve the variable-coefficient Poisson
        (with Rhie-Chow), and correct BOTH phase velocities.
        Returns u1, v1, u2, v2 (interior).
        """
        g = self.grid
        nx, ny = self.nx, self.ny
        dx, dy, dxc, dyc = self.dx, self.dy, self.dxc, self.dyc
        p_ref = self.case.num.p_ref_value

        a = alpha
        al = 1.0 - a
        D1, D2 = pea["D1"], pea["D2"]
        Dp = al * D1 + a * D2                          # mixture mobility

        # mixture predicted (eliminated) velocity
        Qu = al * pea["Hh1u"] + a * pea["Hh2u"]
        Qv = al * pea["Hh1v"] + a * pea["Hh2v"]

        Dp_p = _pad(Dp)
        Dxf = op._lin_face_x(Dp_p, g)
        Dyf = op._lin_face_y(Dp_p, g)

        # Rhie-Chow mixture face flux
        Qup = _pad(Qu); Qvp = _pad(Qv)
        Quf = op._lin_face_x(Qup, g); Qvf = op._lin_face_y(Qvp, g)
        pp = _pad(op.interior(f.p))
        pp[:, 0] = pp[:, 1]; pp[:, -1] = pp[:, -2]; pp[0, :] = pp[1, :]
        pp[-1, 1:-1] = 2.0 * p_ref - op.interior(f.p)[-1, :]
        gpx = _pad(op.grad_x(pp, g)); gpy = _pad(op.grad_y(pp, g))
        gc_x = (pp[1:-1, 1:] - pp[1:-1, :-1]) / dxc[None, :]
        gc_y = (pp[1:, 1:-1] - pp[:-1, 1:-1]) / dyc[:, None]
        gi_x = op._lin_face_x(gpx, g); gi_y = op._lin_face_y(gpy, g)
        Ff_x = Quf - Dxf * (gc_x - gi_x)
        Ff_y = Qvf - Dyf * (gc_y - gi_y)
        Ff_x[:, 0] = 0.0; Ff_x[:, -1] = 0.0
        # Bottom face: closed wall if U_in<=0, otherwise prescribed mixture inflow.
        Ff_y[0, :] = self.case.inlet.U_in if self.case.inlet.U_in > 0.0 else 0.0

        divPhi = ((Ff_x[:, 1:] - Ff_x[:, :-1]) / dx[None, :]
                  + (Ff_y[1:, :] - Ff_y[:-1, :]) / dy[:, None])
        rhs = divPhi - S_vol

        De = Dxf[:, 1:]; Dw = Dxf[:, :-1]; Dn = Dyf[1:, :]; Ds = Dyf[:-1, :]
        cN_top = Dn[-1, :] * (dx / (0.5 * dy[-1])) / g.Vol[-1, :]
        b = rhs.copy()
        b[-1, :] -= cN_top * p_ref
        A = self._matrix_var(De, Dw, Dn, Ds, cN_top)
        bvec = b.ravel()
        # D1/D2 and therefore the Poisson matrix are unchanged across pressure
        # correctors within one PEA predictor, so reuse the LU factorisation
        # instead of refactorising every corrector when n_corr > 1.
        if self.case.num.reuse_pressure_lu:
            if "_poisson_lu" not in pea:
                pea["_poisson_lu"] = spla.factorized(A.tocsc())
            p = pea["_poisson_lu"](bvec).reshape(ny, nx)
        else:
            p = spla.spsolve(A, bvec).reshape(ny, nx)

        relax = self.case.num.relax_p
        p_old = op.interior(f.p)
        p_new = p_old + relax * (p - p_old)
        f.p[1:-1, 1:-1] = p_new

        # cell pressure gradient (with BC ghosts) for phase correction
        ppn = _pad(p_new)
        ppn[:, 0] = ppn[:, 1]; ppn[:, -1] = ppn[:, -2]; ppn[0, :] = ppn[1, :]
        ppn[-1, 1:-1] = 2.0 * p_ref - p_new[-1, :]
        dpdx = op.grad_x(ppn, g); dpdy = op.grad_y(ppn, g)

        u1 = pea["Hh1u"] - D1 * dpdx
        v1 = pea["Hh1v"] - D1 * dpdy
        u2 = pea["Hh2u"] - D2 * dpdx
        v2 = pea["Hh2v"] - D2 * dpdy
        return u1, v1, u2, v2


def _pad(F):
    P = np.empty((F.shape[0] + 2, F.shape[1] + 2))
    P[1:-1, 1:-1] = F
    P[:, 0] = P[:, 1]; P[:, -1] = P[:, -2]
    P[0, :] = P[1, :]; P[-1, :] = P[-2, :]
    P[1:-1, 0] = F[:, 0]; P[1:-1, -1] = F[:, -1]
    P[0, 1:-1] = F[0, :]; P[-1, 1:-1] = F[-1, :]
    return P
