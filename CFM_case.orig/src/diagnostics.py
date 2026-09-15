"""
Post-hoc plume diagnostics used by the post-processing scripts:
plume_centroid, plume_width, and bl_thickness (boundary-layer/wall-jet
measures of v_l(x,y) on each horizontal line, x from the electrode wall).
See the report for how each thickness is defined and normalised.
"""

import numpy as np


def plume_centroid(alpha, xc):
    """x_cg(y): first moment of alpha in x at each height. Shape (ny,)."""
    num = (alpha * xc[None, :]).sum(axis=1)
    den = alpha.sum(axis=1) + 1e-30
    return num / den


def plume_width(alpha, xc):
    """RMS lateral spread of the gas at each height. Shape (ny,)."""
    xcg = plume_centroid(alpha, xc)
    num = (alpha * (xc[None, :] - xcg[:, None])**2).sum(axis=1)
    den = alpha.sum(axis=1) + 1e-30
    return np.sqrt(num / den)


def bl_thickness(v, xc, frac_edge=0.05):
    """
    Boundary-layer/wall-jet measures of v(x) for one horizontal line.
    Returns (delta_int, flux_int, half, disp, mom, vmax, x_at_vmax); see the
    report for definitions. Only v>0 contributes, so return flow doesn't
    register as negative thickness.
    """
    x = np.asarray(xc)
    vp = np.maximum(np.asarray(v), 0.0)
    vmax = float(vp.max())
    if vmax <= 1e-14:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    im = int(np.argmax(vp))
    xm = float(x[im])

    # Edge: first point after the peak where the positive velocity is small.
    ie = len(x) - 1
    for k in range(im, len(x)):
        if vp[k] < frac_edge * vmax:
            ie = k
            break

    xs = x[:ie + 1]
    vs = vp[:ie + 1]
    vn = np.clip(vs / vmax, 0.0, None)

    flux_int = _trapz(vs, xs)              # m^2/s
    delta_int = flux_int / vmax            # m, equivalent top-hat thickness

    # Half-width on the outer side of the peak.
    half = float(x[ie] - xm)
    for k in range(im, ie + 1):
        if vp[k] <= 0.5 * vmax:
            half = float(x[k] - xm)
            break

    disp = _trapz(1.0 - vn, xs)
    mom = _trapz(vn * (1.0 - vn), xs)
    return delta_int, flux_int, half, disp, mom, vmax, xm


def bl_profile_vs_height(v_field, xc):
    """Apply bl_thickness at every height. Returns dict of arrays (ny,)."""
    ny = v_field.shape[0]
    delta = np.zeros(ny)
    flux = np.zeros(ny)
    half = np.zeros(ny)
    disp = np.zeros(ny)
    mom = np.zeros(ny)
    vmax = np.zeros(ny)
    xm = np.zeros(ny)
    for j in range(ny):
        (delta[j], flux[j], half[j], disp[j], mom[j],
         vmax[j], xm[j]) = bl_thickness(v_field[j], xc)
    return dict(delta=delta, flux=flux, half=half, disp=disp,
                mom=mom, vmax=vmax, xm=xm)


def _trapz(y, x):
    """Trapezoidal integration compatible with numpy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
