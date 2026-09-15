"""
State-field container (ghost-padded).

Holds the unknowns for both phases plus shared pressure and gas fraction.
All arrays are padded with one ghost layer: shape (ny+2, nx+2).
"""

import numpy as np


class Fields:
    def __init__(self, grid):
        ny, nx = grid.ny, grid.nx
        s = (ny + 2, nx + 2)

        # gas volume fraction (phase 2); liquid is 1 - alpha
        self.alpha = np.zeros(s)

        # liquid velocity (phase 1)
        self.u1 = np.zeros(s)
        self.v1 = np.zeros(s)
        # gas velocity (phase 2)
        self.u2 = np.zeros(s)
        self.v2 = np.zeros(s)

        # shared pressure (gauge)
        self.p = np.zeros(s)

        # turbulent (eddy) kinematic viscosity, shared
        self.nut = np.zeros(s)

        self.grid = grid

    def alpha_liq(self):
        return 1.0 - self.alpha

    def as_dict_interior(self):
        I = (slice(1, -1), slice(1, -1))
        return dict(alpha=self.alpha[I].copy(),
                    u1=self.u1[I].copy(), v1=self.v1[I].copy(),
                    u2=self.u2[I].copy(), v2=self.v2[I].copy(),
                    p=self.p[I].copy(), nut=self.nut[I].copy())
