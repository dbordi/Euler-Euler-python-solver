"""
Cartesian 2D mesh with near-wall grading in x.

Collocated finite-volume grid (all unknowns at cell centres), with a
Rhie-Chow face-flux projection for pressure-velocity coupling -- this avoids
checkerboarding without a fully staggered layout.

xc, yc: cell-centre coordinates. xf, yf: face coordinates. dx, dy: cell
sizes. Vol: cell volume (unit depth). wall_dist: distance to the nearest
x-wall, used by the mixing-length turbulence closure.
"""

import numpy as np


def _faces_prescribed_first(length, n, first_cells):
    """n+1 faces where the first len(first_cells) cells have exactly the
    given widths; the rest grow geometrically from the last one to fill
    the domain.
    """
    widths = [float(w) for w in first_cells if float(w) > 0.0]
    k = len(widths)
    if k == 0:
        return np.linspace(0.0, length, n + 1)
    if k >= n:                                # prescribed more than we have cells
        widths = widths[:n]
        faces = np.concatenate([[0.0], np.cumsum(widths)])
        return faces / faces[-1] * length

    used = sum(widths)
    rem_len = length - used
    rem_n = n - k
    if rem_len <= 0.0:                         # prescribed widths overrun L -> rescale
        faces = np.concatenate([[0.0], np.cumsum(widths)])
        extra = np.full(rem_n, faces[-1] / max(k, 1))
        faces = np.concatenate([faces, faces[-1] + np.cumsum(extra)])
        return faces / faces[-1] * length

    # geometric continuation: h_last*g, h_last*g^2, ... (rem_n cells) sum to rem_len
    h_last = widths[-1]
    target = rem_len / h_last

    def series(g):
        if abs(g - 1.0) < 1e-15:
            return float(rem_n)
        log_term = rem_n * np.log(g)
        if log_term > 700.0:
            return np.inf
        return g * (np.exp(log_term) - 1.0) / (g - 1.0)

    lo, hi = 1.0, 2.0
    if series(1.0) >= target:
        g = 1.0                               # continuation must shrink; clamp to uniform-ish
    else:
        while series(hi) < target and hi < 1.0e6:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if series(mid) > target:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-14:
                break
        g = 0.5 * (lo + hi)

    cont = h_last * g ** np.arange(1, rem_n + 1)
    all_widths = np.concatenate([widths, cont])
    faces = np.concatenate([[0.0], np.cumsum(all_widths)])
    faces[-1] = length
    return faces


def _symmetric_core_from_last_width(core_length, core_n, h_last):
    """core_n symmetric widths filling core_length, continuing from h_last
    on both sides. Falls back to a uniform core if h_last is too large to
    keep growing and still fit.
    """
    if core_n <= 0:
        return np.array([], dtype=float)
    if core_length <= 0.0:
        return np.full(core_n, 1.0 / core_n, dtype=float)

    target = core_length / max(h_last, 1e-30)
    if target <= core_n:
        return np.full(core_n, core_length / core_n, dtype=float)

    n_side = core_n // 2
    has_centre_cell = (core_n % 2 == 1)

    def series(g):
        if abs(g - 1.0) < 1e-15:
            return float(core_n)
        powers_side = g ** np.arange(1, n_side + 1)
        total = 2.0 * powers_side.sum()
        if has_centre_cell:
            total += g ** (n_side + 1)
        return total

    lo, hi = 1.0, 2.0
    while series(hi) < target and hi < 1.0e6:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if series(mid) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14:
            break
    g = 0.5 * (lo + hi)

    side = h_last * g ** np.arange(1, n_side + 1)
    if has_centre_cell:
        core = np.concatenate([side, [h_last * g ** (n_side + 1)], side[::-1]])
    else:
        core = np.concatenate([side, side[::-1]])
    core *= core_length / core.sum()           # snap roundoff, keep the wall cells exact
    return core


def _faces_prescribed_both(length, n, first_cells):
    """n+1 faces with the same prescribed first cells mirrored at both
    x-walls, e.g. [20e-6, 30e-6, 50e-6] gives identical 20/30/50 um cells
    at x=0 and x=L, growing symmetrically towards a coarser core.
    """
    widths = np.array([float(w) for w in first_cells if float(w) > 0.0], dtype=float)
    k = len(widths)
    if k == 0:
        return np.linspace(0.0, length, n + 1)

    if 2 * k >= n:
        # too few cells for k on each wall plus a core -- trim and rescale
        k_eff = max((n - 1) // 2, 0)
        left = widths[:k_eff]
        core_n = n - 2 * k_eff
        core = np.ones(core_n, dtype=float)
        all_widths = np.concatenate([left, core, left[::-1]])
        all_widths = all_widths / all_widths.sum() * length
        faces = np.concatenate([[0.0], np.cumsum(all_widths)])
        faces[-1] = length
        return faces

    used = 2.0 * widths.sum()
    core_length = length - used
    core_n = n - 2 * k
    if core_length <= 0.0:
        # prescribed cells overrun the domain -- keep symmetry, drop exact sizes
        all_widths = np.concatenate([widths, np.ones(core_n), widths[::-1]])
        all_widths = all_widths / all_widths.sum() * length
        faces = np.concatenate([[0.0], np.cumsum(all_widths)])
        faces[-1] = length
        return faces

    core = _symmetric_core_from_last_width(core_length, core_n, widths[-1])
    all_widths = np.concatenate([widths, core, widths[::-1]])
    faces = np.concatenate([[0.0], np.cumsum(all_widths)])
    faces[-1] = length
    return faces


def _x_faces(case):
    """x faces: uniform if no near-wall cells are given, otherwise graded
    from case.mesh.x_first_cells (mirrored at both walls in "both" mode).
    """
    g, m = case.geom, case.mesh
    if not m.x_first_cells:
        return np.linspace(0.0, g.L, m.nx + 1)
    if str(m.x_grading_mode).strip().lower() == "both":
        return _faces_prescribed_both(g.L, m.nx, m.x_first_cells)
    return _faces_prescribed_first(g.L, m.nx, m.x_first_cells)


def _y_faces(case):
    g, m = case.geom, case.mesh
    return np.linspace(0.0, g.H, m.ny + 1)


class Grid:
    def __init__(self, case):
        g, m = case.geom, case.mesh
        self.nx, self.ny = m.nx, m.ny

        self.xf = _x_faces(case)
        self.yf = _y_faces(case)

        self.dx = np.diff(self.xf)
        self.dy = np.diff(self.yf)
        self.xc = 0.5 * (self.xf[:-1] + self.xf[1:])
        self.yc = 0.5 * (self.yf[:-1] + self.yf[1:])

        # 2D helper arrays, indexed [j, i] = [y, x]
        self.DX = np.tile(self.dx, (self.ny, 1))
        self.DY = np.tile(self.dy[:, None], (1, self.nx))
        self.Vol = self.DX * self.DY

        # distance to the nearest vertical wall, for the mixing-length closure
        d_left = self.xc
        d_right = g.L - self.xc
        self.wall_dist = np.tile(np.minimum(d_left, d_right), (self.ny, 1))

        # padded (ghost-cell) metrics -- one ghost layer, mirroring the adjacent cell
        self.dxp = np.concatenate([[self.dx[0]], self.dx, [self.dx[-1]]])
        self.dyp = np.concatenate([[self.dy[0]], self.dy, [self.dy[-1]]])
        self.xcp = np.concatenate([[self.xc[0] - self.dx[0]], self.xc,
                                   [self.xc[-1] + self.dx[-1]]])
        self.ycp = np.concatenate([[self.yc[0] - self.dy[0]], self.yc,
                                   [self.yc[-1] + self.dy[-1]]])
        self.dxc = np.diff(self.xcp)
        self.dyc = np.diff(self.ycp)

    @property
    def shape(self):
        return (self.ny, self.nx)

    def save(self, path):
        np.savez(path, xf=self.xf, yf=self.yf, xc=self.xc, yc=self.yc,
                 dx=self.dx, dy=self.dy, nx=self.nx, ny=self.ny)

    @classmethod
    def load(cls, path, case):
        # grids are cheap to rebuild; the npz is kept for inspection only
        return cls(case)


def build(case):
    return Grid(case)
