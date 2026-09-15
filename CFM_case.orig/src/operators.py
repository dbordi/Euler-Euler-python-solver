"""
Finite-volume operators on a ghost-padded collocated grid.

Padded fields have shape (ny+2, nx+2); the physical interior is [1:-1, 1:-1].
Ghost cells (index 0 and -1 on each axis) are filled by boundary.apply_bc
before any operator here is called. Operators return interior-sized (ny, nx)
arrays unless noted.
"""

import numpy as np


def interior(Fp):
    """View of the physical interior of a padded field."""
    return Fp[1:-1, 1:-1]


def grad_x(Fp, grid):
    """d/dx at cell centres (central, non-uniform), interior (ny, nx)."""
    # face values on the two x-faces of every interior cell
    dxc = grid.dxc                                  # (nx+1,)
    # gradient across west and east faces, then average to centre
    dFdx_face = (Fp[1:-1, 1:] - Fp[1:-1, :-1]) / dxc[None, :]   # (ny, nx+1)
    return 0.5 * (dFdx_face[:, :-1] + dFdx_face[:, 1:])


def grad_y(Fp, grid):
    dyc = grid.dyc
    dFdy_face = (Fp[1:, 1:-1] - Fp[:-1, 1:-1]) / dyc[:, None]   # (ny+1, nx)
    return 0.5 * (dFdy_face[:-1, :] + dFdy_face[1:, :])


def divergence(Up, Vp, grid):
    """
    div(U) for a cell-centred vector field (Up, Vp), conservative via faces.
    Face-normal velocities are linearly interpolated to faces.
    Returns interior (ny, nx).
    """
    dx, dy = grid.dx, grid.dy
    # x-faces: interpolate u to interior x-faces (ny, nx+1)
    uf = _lin_face_x(Up, grid)
    vf = _lin_face_y(Vp, grid)
    dudx = (uf[:, 1:] - uf[:, :-1]) / dx[None, :]
    dvdy = (vf[1:, :] - vf[:-1, :]) / dy[:, None]
    return dudx + dvdy


def _lin_face_x(Fp, grid):
    """Linear interpolation of a padded field to interior x-faces -> (ny, nx+1)."""
    dxp = grid.dxp
    Fr = Fp[1:-1, 1:]          # right cells of each face (nx+1 faces)
    Fl = Fp[1:-1, :-1]
    wl = dxp[1:]               # weight of left cell ~ size of right cell
    wr = dxp[:-1]
    return (Fl * wl[None, :] + Fr * wr[None, :]) / (wl + wr)[None, :]


def _lin_face_y(Fp, grid):
    dyp = grid.dyp
    Ft = Fp[1:, 1:-1]
    Fb = Fp[:-1, 1:-1]
    wb = dyp[1:]
    wt = dyp[:-1]
    return (Fb * wb[:, None] + Ft * wt[:, None]) / (wb + wt)[:, None]


def advect_upwind(Fp, Up, Vp, grid):
    """
    -div(F * U) using first-order upwind face values (bounded, stable).
    F, U, V are padded; returns interior (ny, nx).
    """
    dx, dy = grid.dx, grid.dy
    uf = _lin_face_x(Up, grid)            # face-normal x velocity (ny, nx+1)
    vf = _lin_face_y(Vp, grid)            # face-normal y velocity (ny+1, nx)

    # upwind face value of F on x-faces
    Fl = Fp[1:-1, :-1]
    Fr = Fp[1:-1, 1:]
    Fx = np.where(uf >= 0.0, Fl, Fr)      # (ny, nx+1)
    flux_x = uf * Fx

    Fb = Fp[:-1, 1:-1]
    Ft = Fp[1:, 1:-1]
    Fy = np.where(vf >= 0.0, Fb, Ft)
    flux_y = vf * Fy

    div = ((flux_x[:, 1:] - flux_x[:, :-1]) / dx[None, :]
           + (flux_y[1:, :] - flux_y[:-1, :]) / dy[:, None])
    return -div


def diffuse(Fp, Gam_p, grid):
    """
    div(Gamma grad F) with cell-centred diffusivity Gamma (padded).
    Gamma is harmonically/linearly interpolated to faces. Returns interior.
    """
    dx, dy, dxc, dyc = grid.dx, grid.dy, grid.dxc, grid.dyc
    # face diffusivities (linear interp is fine and robust here)
    Gxf = _lin_face_x(Gam_p, grid)        # (ny, nx+1)
    Gyf = _lin_face_y(Gam_p, grid)        # (ny+1, nx)

    dFx = (Fp[1:-1, 1:] - Fp[1:-1, :-1]) / dxc[None, :]   # grad at x-faces
    dFy = (Fp[1:, 1:-1] - Fp[:-1, 1:-1]) / dyc[:, None]

    flux_x = Gxf * dFx
    flux_y = Gyf * dFy
    return ((flux_x[:, 1:] - flux_x[:, :-1]) / dx[None, :]
            + (flux_y[1:, :] - flux_y[:-1, :]) / dy[:, None])


def _vanleer(r):
    """van Leer flux limiter."""
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def advect_limited(Fp, Up, Vp, grid):
    """
    -div(F * U) with a van Leer TVD scheme (low numerical diffusion, bounded).
    Falls back to upwind on the first/last interior face where the second
    upwind point is unavailable. F, U, V padded; returns interior (ny, nx).
    """
    dx, dy = grid.dx, grid.dy
    uf = _lin_face_x(Up, grid)            # (ny, nx+1)
    vf = _lin_face_y(Vp, grid)            # (ny+1, nx)

    Fx = _tvd_faces_x(Fp, uf)
    Fy = _tvd_faces_y(Fp, vf)
    flux_x = uf * Fx
    flux_y = vf * Fy
    div = ((flux_x[:, 1:] - flux_x[:, :-1]) / dx[None, :]
           + (flux_y[1:, :] - flux_y[:-1, :]) / dy[:, None])
    return -div


def _tvd_faces_x(Fp, uf):
    eps = 1e-30
    L = Fp[1:-1, :]                          # (ny, nx+2)
    nx = L.shape[1] - 2
    Fxe = np.where(uf >= 0.0, Fp[1:-1, :-1], Fp[1:-1, 1:])   # upwind baseline
    if nx < 3:
        return Fxe
    # interior faces f = 1 .. nx-1  -> columns 1..nx-1 of uf
    ufi = uf[:, 1:nx]
    # positive flow: donor=col f, upwind=col f-1, downwind=col f+1
    Fd_p = L[:, 1:nx]; Fu_p = L[:, 0:nx-1]; Fdd_p = L[:, 2:nx+1]
    rp = (Fd_p - Fu_p) / np.where(np.abs(Fdd_p - Fd_p) < eps, eps, Fdd_p - Fd_p)
    val_p = Fd_p + 0.5 * _vanleer(rp) * (Fdd_p - Fd_p)
    # negative flow: donor=col f+1, upwind=col f+2, downwind=col f
    Fd_m = L[:, 2:nx+1]; Fu_m = L[:, 3:nx+2]; Fdd_m = L[:, 1:nx]
    rm = (Fd_m - Fu_m) / np.where(np.abs(Fdd_m - Fd_m) < eps, eps, Fdd_m - Fd_m)
    val_m = Fd_m + 0.5 * _vanleer(rm) * (Fdd_m - Fd_m)
    Fxe[:, 1:nx] = np.where(ufi >= 0.0, val_p, val_m)
    return Fxe


def _tvd_faces_y(Fp, vf):
    eps = 1e-30
    L = Fp[:, 1:-1]                          # (ny+2, nx)
    ny = L.shape[0] - 2
    Fye = np.where(vf >= 0.0, Fp[:-1, 1:-1], Fp[1:, 1:-1])
    if ny < 3:
        return Fye
    vfi = vf[1:ny, :]
    Fd_p = L[1:ny, :]; Fu_p = L[0:ny-1, :]; Fdd_p = L[2:ny+1, :]
    rp = (Fd_p - Fu_p) / np.where(np.abs(Fdd_p - Fd_p) < eps, eps, Fdd_p - Fd_p)
    val_p = Fd_p + 0.5 * _vanleer(rp) * (Fdd_p - Fd_p)
    Fd_m = L[2:ny+1, :]; Fu_m = L[3:ny+2, :]; Fdd_m = L[1:ny, :]
    rm = (Fd_m - Fu_m) / np.where(np.abs(Fdd_m - Fd_m) < eps, eps, Fdd_m - Fd_m)
    val_m = Fd_m + 0.5 * _vanleer(rm) * (Fdd_m - Fd_m)
    Fye[1:ny, :] = np.where(vfi >= 0.0, val_p, val_m)
    return Fye


def advect(Fp, Up, Vp, grid, scheme="upwind"):
    """Dispatch convection scheme."""
    if scheme == "limited":
        return advect_limited(Fp, Up, Vp, grid)
    return advect_upwind(Fp, Up, Vp, grid)
