"""
Electrolytic hydrogen source term, applied in the n_electrode_cells columns
next to the electrode wall (x=0). See the report for the derivation:
m_dot = rho_g * j * V_m * f_i / (2 * F * V_i).
"""

import numpy as np


class GasSourceField:
    def __init__(self, case, grid):
        self.case = case
        self.grid = grid
        self._build_mask()
        self._build_base_rate()

    def _build_mask(self):
        g, s = self.grid, self.case.source
        ny, nx = g.ny, g.nx
        H = self.case.geom.H
        self.mask = np.zeros((ny, nx), dtype=bool)
        y0, y1 = s.electrode_y_start * H, s.electrode_y_end * H
        active_rows = (g.yc >= y0) & (g.yc <= y1)
        ncol = max(1, s.n_electrode_cells)
        for i in range(ncol):
            self.mask[active_rows, i] = True

    def _build_base_rate(self):
        c, g, s = self.case, self.grid, self.case.source
        rho_g = c.fluids.rho_g
        V_m = c.fluids.V_m
        areal = rho_g * s.j * V_m / (2.0 * s.F)   # [kg/(m^2 s)]
        # volumetric source per cell = areal * (f_i/V_i), f_i/V_i = 1/dx here
        self.mdot = np.zeros((g.ny, g.nx))
        ncol = max(1, s.n_electrode_cells)
        for i in range(ncol):
            self.mdot[self.mask[:, i], i] = areal / g.dx[i]
        # store for diagnostics
        self.areal_mass_flux = areal
        self.superficial_gas_velocity = areal / rho_g   # m/s at the wall

    def rate(self, t):
        """Mass source [kg/(m^3 s)] at time t, with linear ramp-up."""
        s = self.case.source
        ramp = 1.0 if s.ramp_time <= 0 else min(1.0, t / s.ramp_time)
        return ramp * self.mdot
