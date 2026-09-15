"""
Unified post-processing for the Euler-Euler wall/free plume solver.

Put this file in each case folder and run from the case root:

    python postprocess_all_plume.py --current-a 25.6 --inflow-cms 2.5 --gap-cm 5

or, if placed in postproc/, run from the project/case root in the same way.

The script reads output/fields_<time>.npz and writes all figures and CSV files to:

    output/postproc_all/

Main outputs:
    alpha_snapshot_final.png / alpha_time_mean.png / alpha_evolution.gif
    streamlines_final.png / streamlines_time_mean.png   (3-panel: alpha, v_l, |U_l| with liquid streamlines)
    velocity_quiver_final.png / velocity_quiver_time_mean.png   (alpha map with liquid velocity arrows)
    profiles_velocity_final.png / profiles_velocity_mean.png
    profiles_alpha_final.png / profiles_alpha_mean.png
    profiles_slip_final.png / profiles_slip_mean.png
    profiles_nut_final.png / profiles_nut_mean.png
    bl_no_fit_final.png / bl_no_fit_mean.png
    bl_piecewise_final.png / bl_piecewise_mean.png
    bl_profiles_<case_tag>.csv
    bl_piecewise_fits_<case_tag>.csv
    reverse_flow_<case_tag>.csv
    reverse_flow_all_heights_<case_tag>.csv
    case_metrics_<case_tag>.csv
    velocity_profiles_<case_tag>.csv   (long/tidy format for sweep-level analysis)

Notes:
    * Plot titles are intentionally omitted; panel labels (a), (b), ... sit a
      fixed distance below each panel regardless of its aspect ratio.
    * Legends are boxed with a white background, or placed outside the axes
      below the panel row when several panels share one legend.
    * Profile x-limit default is 12 mm, as in the previous profile script.
    * Alpha/streamline/quiver 2D maps cap the rendered height:width ratio
      (physical aspect is preserved up to that cap) so narrow-gap cases do not
      blow up into an unreadable, page-long sliver; an induced horizontal
      stretch beyond the cap is disclosed with a small in-panel note.
    * BL scaling is fitted in log-log space with an automatic 1-, 2- or
      3-segment piecewise power law. Segments beyond the first (near-source)
      one must span a substantially wider height range, so the far field is
      not chopped into short, insignificant pieces; a segment with too few
      points or too much scatter is flagged low-confidence in the plot/CSV.
    * BL plots use x in [1e-2 m, H - bl_end_margin_m] (default margin 5 cm)
      to drop the outlet-affected region near the domain top.
    * A no-fit BL plot is always written as requested.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

_PUB_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#000000", "#6B6B6B",
]
_PUB_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.3,
    "axes.labelsize": 8.8,
    "axes.linewidth": 0.85,
    "axes.unicode_minus": False,
    "axes.prop_cycle": cycler(color=_PUB_COLORS),
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    # Boxed, white-background legends everywhere by default; figures that need
    # a shared legend instead place it outside the axes, below the panel row.
    "legend.fontsize": 7.3,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "0.55",
    "legend.framealpha": 0.92,
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.35,
    "legend.handlelength": 1.7,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "lines.linewidth": 1.55,
    "lines.markersize": 4.6,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

#_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.abspath(os.getcwd())

OUT_DIR = os.path.join(_PROJECT_ROOT, "output")
SAVE_DIR = os.path.join(OUT_DIR, "postproc_all")
_FIELD_KEYS = ("alpha", "u1", "v1", "u2", "v2", "nut")


def _long_path(path: str) -> str:
    """Return an extended-length ("\\\\?\\") form of an absolute Windows path.

    Case tags carry gap/inflow/current metadata and can push filenames past
    the classic 260-character MAX_PATH limit once nested a few folders deep
    under a long project root (as under a synced OneDrive tree); the
    "\\\\?\\" prefix asks the Win32 API to bypass that limit without needing
    any system-wide long-path policy enabled. A no-op on other platforms.
    """
    if os.name != "nt":
        return path
    ap = os.path.abspath(path)
    if ap.startswith("\\\\?\\"):
        return ap
    if ap.startswith("\\\\"):
        return "\\\\?\\UNC\\" + ap.lstrip("\\")
    return "\\\\?\\" + ap


def _open_w(path: str, newline: Optional[str] = "", encoding: Optional[str] = None):
    return open(_long_path(path), "w", newline=newline, encoding=encoding)


def _makedirs(path: str) -> None:
    os.makedirs(_long_path(path), exist_ok=True)

@dataclass
class Snapshot:
    time: float
    path: str


@dataclass
class CaseMeta:
    case_name: str
    current_a: float
    current_density_a_m2: float
    inflow_cms: float
    gap_cm: float
    gap_m: float
    variant: str
    reference_class: str
    tag: str


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def numeric_field_files(out_dir: str = OUT_DIR) -> List[Snapshot]:
    files: List[Snapshot] = []
    for path in glob.glob(os.path.join(out_dir, "fields_*.npz")):
        stem = os.path.basename(path)[len("fields_"):-len(".npz")]
        try:
            t = float(stem)
        except ValueError:
            continue
        files.append(Snapshot(time=t, path=path))
    files.sort(key=lambda s: s.time)
    if not files:
        latest = os.path.join(out_dir, "fields_latest.npz")
        if os.path.isfile(latest):
            with np.load(latest) as data:
                t_latest = float(data["t"]) if "t" in data.files else 0.0
            files.append(Snapshot(time=t_latest, path=latest))
        else:
            raise FileNotFoundError(
                f"No numeric fields_<time>.npz or fields_latest.npz found in {out_dir}. Run the solver first."
            )
    return files


def load_stacks(files: List[Snapshot], skip_fraction: float = 0.0) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not (0.0 <= skip_fraction < 1.0):
        raise ValueError("skip_fraction must satisfy 0 <= skip_fraction < 1")

    start = min(int(np.floor(skip_fraction * len(files))), len(files) - 1)
    files = files[start:]

    times: List[float] = []
    stacks: Dict[str, List[np.ndarray]] = {k: [] for k in _FIELD_KEYS}
    xc = yc = None

    for snap in files:
        d = np.load(snap.path)
        missing = {"alpha", "xc", "yc"}.difference(d.files)
        if missing:
            raise KeyError(f"{snap.path} missing required arrays: {sorted(missing)}")
        times.append(snap.time)
        base = np.asarray(d["alpha"], dtype=float)
        for k in _FIELD_KEYS:
            arr = np.asarray(d[k], dtype=float) if k in d.files else np.zeros_like(base)
            stacks[k].append(arr)
        xc = np.asarray(d["xc"], dtype=float)
        yc = np.asarray(d["yc"], dtype=float)

    return np.asarray(times, dtype=float), {k: np.stack(v, axis=0) for k, v in stacks.items()}, xc, yc


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _save_figure_all(fig, path: str, dpi: int = 600, transparent_copy: bool = True) -> None:
    root, _ = os.path.splitext(path)
    # pad_inches is deliberately a bit more generous than matplotlib's own
    # default: a rotated axis label's ascender/descender can otherwise be
    # shaved off by one or two pixels at the tight-bbox edge.
    fig.savefig(_long_path(path), dpi=max(int(dpi), 600), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(_long_path(root + ".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(_long_path(root + ".svg"), bbox_inches="tight", pad_inches=0.06)
    if transparent_copy:
        fig.savefig(_long_path(root + "_transparent.svg"), transparent=True, bbox_inches="tight", pad_inches=0.06)


def _prettify_ax(ax, grid: bool = True) -> None:
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if grid:
        ax.grid(True, which="major", color="0.88", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def _series_style(index: int) -> Dict[str, Any]:
    return {"color": _PUB_COLORS[index % len(_PUB_COLORS)], "marker": _PUB_MARKERS[index % len(_PUB_MARKERS)]}


def _add_panel_labels(axes, offset_pt: float = 15.0, fontsize: float = 8.6) -> None:
    """Place (a), (b), ... just below each panel.

    The offset is specified in points (fixed physical distance), not axes
    fraction, so the label sits at a consistent, small gap under the axis
    regardless of the panel's aspect ratio (critical for the tall, narrow
    channel maps where an axes-fraction offset would land far off the page).
    """
    visible = [ax for ax in np.atleast_1d(axes).ravel() if ax.get_visible()]
    for i, ax in enumerate(visible):
        ax.annotate(
            f"({chr(ord('a') + i)})", xy=(0.5, 0.0), xycoords="axes fraction",
            xytext=(0.0, -float(offset_pt)), textcoords="offset points",
            ha="center", va="top", fontsize=fontsize, clip_on=False, annotation_clip=False,
        )


def _legend_unique(handles, labels):
    seen = set(); out_h, out_l = [], []
    for h, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label); out_h.append(h); out_l.append(label)
    return out_h, out_l


def _collect_legend_entries(axes):
    """Collect all series keys so sparse panels cannot drop colour legends."""
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        if not ax.get_visible():
            continue
        h_ax, l_ax = ax.get_legend_handles_labels()
        handles.extend(h_ax)
        labels.extend(l_ax)
    return _legend_unique(handles, labels)


def _show_all_tick_numbers(axes) -> None:
    for ax in np.atleast_1d(axes).ravel():
        if ax.get_visible():
            ax.tick_params(axis="x", which="both", labelbottom=True)
            ax.tick_params(axis="y", which="both", labelleft=True)


def _fig_legend_below(fig, axes, handles, labels, ncol: Optional[int] = None, y: float = 0.01) -> None:
    """Shared legend placed outside the panel row, below every panel.

    Uses a boxed, white-background frame (global rcParams) so it stays legible
    over any panel-label text or axis ticks it may sit near.
    """
    handles, labels = _legend_unique(handles, labels)
    if not handles:
        return
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
               bbox_transform=fig.transFigure, ncol=ncol or min(4, len(labels)),
               frameon=True, facecolor="white", edgecolor="0.55", framealpha=0.96)


def _confidence_flag(row: Dict[str, Any], min_points_ok: int = 8,
                     min_r2_log: float = 0.85, max_rmse_log: float = 0.12) -> bool:
    """True when a segment is long enough and follows a clear log-log trend.

    RMSE is used instead of total SSE so a long, useful segment is not
    penalised merely for containing more grid points.
    """
    npts = int(row.get("n_points", 0))
    r2 = float(row.get("r2_log", np.nan))
    rmse = float(row.get("rmse_log", np.inf))
    return (
        npts >= min_points_ok
        and np.isfinite(r2) and r2 >= min_r2_log
        and np.isfinite(rmse) and rmse <= max_rmse_log
    )


def _fmt_number_for_tag(x: float, ndigits: int = 4) -> str:
    if not np.isfinite(x):
        return "NA"
    s = f"{float(x):.{ndigits}g}"
    return s.replace("-", "m").replace(".", "p").replace("+", "")


def _sanitize_tag(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(text).strip())
    return text.strip("_") or "case"


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _first_float_by_keys(flat: Dict[str, Any], key_patterns: Sequence[str]) -> Optional[float]:
    pats = [re.compile(p, re.I) for p in key_patterns]
    for key, val in flat.items():
        if any(p.search(key) for p in pats):
            try:
                return float(val)
            except Exception:
                continue
    return None


def _first_text_by_keys(flat: Dict[str, Any], key_patterns: Sequence[str]) -> Optional[str]:
    pats = [re.compile(p, re.I) for p in key_patterns]
    for key, val in flat.items():
        if any(p.search(key) for p in pats) and val is not None:
            text = str(val).strip()
            if text:
                return text
    return None


def _parse_value_from_text(text: str, patterns: Sequence[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return float(m.group(1).replace("p", "."))
            except Exception:
                pass
    return None


def infer_case_metadata(args: argparse.Namespace, xc: np.ndarray) -> CaseMeta:
    flat: Dict[str, Any] = {}
    # Generator metadata is authoritative for variant/current density.  Runtime
    # case_summary supplies the values actually used by the solver.
    for metadata_path in (
        os.path.join(_PROJECT_ROOT, "case_metadata.json"),
        os.path.join(OUT_DIR, "case_summary.json"),
    ):
        if not os.path.isfile(metadata_path):
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                flat.update(_flatten_dict(json.load(fh)))
        except Exception:
            pass

    case_name = (
        args.case_name
        or _first_text_by_keys(flat, [r"(^|\.)case_name$"])
        or os.path.basename(os.path.abspath(_PROJECT_ROOT))
    )

    folder_text = " ".join([
        os.path.basename(os.path.abspath(_PROJECT_ROOT)),
        os.path.basename(os.path.dirname(os.path.abspath(_PROJECT_ROOT))),
    ])

    current_a = args.current_a
    if current_a is None:
        current_a = _first_float_by_keys(flat, [r"current.*a$", r"current_a", r"I_?A$", r"electrode.*current"])
    if current_a is None:
        current_a = _parse_value_from_text(folder_text, [r"(?:I|current)[_\- ]*([0-9]+(?:[p\.]\d+)?)\s*A"])
    if current_a is None:
        current_a = float("nan")

    current_density_a_m2 = args.current_density_a_m2
    if current_density_a_m2 is None:
        current_density_a_m2 = _first_float_by_keys(flat, [
            r"current_density.*a.*m2$", r"source_j_A_m2$", r"source\.j$",
        ])
    if current_density_a_m2 is None:
        current_density_a_m2 = _parse_value_from_text(
            folder_text, [r"(?:J|current_density)[_\- ]*([0-9]+(?:[p\.]\d+)?)"]
        )
    if current_density_a_m2 is None:
        current_density_a_m2 = float("nan")

    inflow_cms = args.inflow_cms
    if inflow_cms is None:
        inflow_cms = _first_float_by_keys(flat, [r"inflow.*cm", r"inlet.*cm", r"U.*in.*cm", r"bottom.*flow.*cm"])
    if inflow_cms is None:
        inflow_ms = _first_float_by_keys(flat, [r"inflow.*m.?s", r"inlet.*m.?s", r"bottom.*flow.*m"])
        if inflow_ms is not None:
            inflow_cms = 100.0 * inflow_ms
    if inflow_cms is None:
        inflow_cms = _parse_value_from_text(folder_text, [r"(?:U|inflow|inlet)[_\- ]*([0-9]+(?:[p\.]\d+)?)\s*(?:cms|cm_s|cmps)"])
    if inflow_cms is None:
        inflow_cms = float("nan")

    gap_cm = args.gap_cm
    if gap_cm is None:
        gap_cm = _first_float_by_keys(flat, [r"gap.*cm", r"width.*cm", r"channel.*cm"])
    if gap_cm is None:
        gap_m = _first_float_by_keys(flat, [r"gap.*m$", r"geom\.L$", r"\bL$", r"width.*m", r"channel.*m"])
        if gap_m is not None:
            gap_cm = 100.0 * gap_m
    if gap_cm is None:
        gap_cm = _parse_value_from_text(folder_text, [r"(?:G|gap|width)[_\- ]*([0-9]+(?:[p\.]\d+)?)\s*cm"])
    if gap_cm is None:
        # Approximate physical domain width from cell centres.
        dx = float(np.nanmedian(np.diff(xc))) if len(xc) > 1 else 0.0
        gap_cm = 100.0 * (float(np.nanmax(xc) - np.nanmin(xc)) + dx)

    gap_m = 0.01 * float(gap_cm) if np.isfinite(gap_cm) else float("nan")
    variant = (
        args.variant
        or _first_text_by_keys(flat, [r"(^|\.)variant$"])
        or "unknown"
    )
    reference_class = (
        args.reference_class
        or _first_text_by_keys(flat, [r"(^|\.)reference_class$"])
        or ("far_wall_reference" if np.isfinite(gap_cm) and float(gap_cm) >= 15.0 else "confined_channel")
    )
    if np.isfinite(current_a):
        forcing_tag = f"I{_fmt_number_for_tag(current_a)}A"
    elif np.isfinite(current_density_a_m2):
        forcing_tag = f"J{_fmt_number_for_tag(current_density_a_m2)}Am2"
    else:
        forcing_tag = "forcingNA"
    tag = _sanitize_tag(
        f"{case_name}_{forcing_tag}_U{_fmt_number_for_tag(inflow_cms)}cms_G{_fmt_number_for_tag(gap_cm)}cm"
    )
    return CaseMeta(
        case_name=case_name,
        current_a=float(current_a),
        current_density_a_m2=float(current_density_a_m2),
        inflow_cms=float(inflow_cms),
        gap_cm=float(gap_cm),
        gap_m=float(gap_m),
        variant=str(variant),
        reference_class=str(reference_class),
        tag=tag,
    )


def read_nu_l(default: float = 1.639e-6) -> float:
    summary_path = os.path.join(OUT_DIR, "case_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                info = json.load(fh)
            flat = _flatten_dict(info)
            nu = _first_float_by_keys(flat, [r"nu_l", r"kinematic.*visc"])
            if nu and np.isfinite(nu):
                return float(nu)
            mu = _first_float_by_keys(flat, [r"mu_l", r"viscosity.*liquid", r"liquid.*mu"])
            rho = _first_float_by_keys(flat, [r"rho_l", r"density.*liquid", r"liquid.*rho"])
            if mu and rho:
                return float(mu) / float(rho)
        except Exception:
            pass
    return default


# -----------------------------------------------------------------------------
# Alpha plots / GIF
# -----------------------------------------------------------------------------

def _crop_to_xmax(field: np.ndarray, xc: np.ndarray, xmax_m: Optional[float]):
    if xmax_m is None or xmax_m <= 0:
        return field, xc
    mask = xc <= float(xmax_m)
    if not np.any(mask):
        mask[np.argmin(np.abs(xc - float(xmax_m)))] = True
    return field[..., mask], xc[mask]


_FIELD_ASPECT_CAP = 5.0     # max height:width box ratio allowed for a single field panel
_FIELD_FIGSIZE_1 = (3.15, 4.5)   # single-panel field map (alpha snapshot, alpha GIF)
_FIELD_FIGSIZE_3 = (7.15, 4.6)   # 1x3 panel row (streamlines / quiver figure)


def _configure_field_axes(ax, xc_plot_m: np.ndarray, yc_m: np.ndarray,
                          aspect_cap: float = _FIELD_ASPECT_CAP, max_x_ticks: int = 4) -> float:
    """Bound a channel-map axes to a legible, fixed-size box.

    Real channel gaps range from 0.5 to 15 cm against a ~0.5 m tall domain, so
    a literal 1:1 physical aspect blows the figure up to an unusable, page-long
    sliver for the narrow gaps. Instead we cap the rendered height:width ratio
    and, when the cap is active, disclose the induced horizontal stretch in a
    small in-panel note so the distortion is never silently misleading.
    The x-axis is drawn in millimetres with a handful of major ticks; the
    narrow physical range (few mm to a few cm) otherwise produces a wall of
    overlapping decimal tick labels.
    """
    xmin, xmax = float(np.min(xc_plot_m)) * 1e3, float(np.max(xc_plot_m)) * 1e3
    ymin, ymax = float(np.min(yc_m)), float(np.max(yc_m))
    xr_mm = max(xmax - xmin, 1e-9)
    yr_m = max(ymax - ymin, 1e-9)
    data_ratio = yr_m / (xr_mm * 1e-3)  # both sides converted to metres before dividing
    box_ratio = min(data_ratio, float(aspect_cap))
    ax.set_box_aspect(box_ratio)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x_ticks, prune=None))
    return float(data_ratio / box_ratio) if box_ratio > 0 else 1.0


def _stretch_note(ax, stretch: float) -> None:
    if stretch > 1.05:
        ax.text(0.03, 0.985, rf"$x$ stretched $\times{stretch:.0f}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.3, color="0.25", zorder=12,
                bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.88, "pad": 1.3, "linewidth": 0.6})


def _positive_log_limits(alpha: np.ndarray, vmax: Optional[float] = None, floor: float = 1e-8) -> Tuple[float, float]:
    finite = np.asarray(alpha[np.isfinite(alpha)], dtype=float)
    positive = finite[finite > 0.0]
    vmin = max(float(np.nanpercentile(positive, 1.0)), floor) if positive.size else floor
    if vmax is None:
        vmax = max(float(np.nanpercentile(finite, 99.5)) if finite.size else vmin * 10.0, vmin * 10.0)
    vmax = max(float(vmax), vmin * 1.01)
    return vmin, vmax


def save_alpha_colormap(alpha: np.ndarray, xc: np.ndarray, yc: np.ndarray, path: str,
                        vmax: Optional[float] = None, xmax_m: Optional[float] = 0.10,
                        log_scale: bool = True, alpha_floor: float = 1e-8) -> None:
    alpha_plot, xc_plot = _crop_to_xmax(alpha, xc, xmax_m)
    X, Y = np.meshgrid(xc_plot * 1e3, yc)
    fig, ax = plt.subplots(figsize=_FIELD_FIGSIZE_1)
    if log_scale:
        vmin, vmax_eff = _positive_log_limits(alpha_plot, vmax=vmax, floor=alpha_floor)
        im = ax.pcolormesh(X, Y, np.clip(alpha_plot, vmin, None), shading="auto", cmap="viridis",
                           rasterized=True, norm=LogNorm(vmin=vmin, vmax=vmax_eff))
        cbar_label = r"Gas fraction, $\alpha_g$ (-)"
    else:
        vmax_eff = max(float(vmax) if vmax is not None else float(np.nanmax(alpha_plot)), 1e-12)
        im = ax.pcolormesh(X, Y, alpha_plot, shading="auto", cmap="viridis", rasterized=True,
                           vmin=0.0, vmax=vmax_eff)
        cbar_label = r"Gas fraction, $\alpha_g$ (-)"

    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"Height, $z$ (m)")
    stretch = _configure_field_axes(ax, xc_plot, yc)
    cbar = fig.colorbar(im, ax=ax, pad=0.05, fraction=0.09)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=7.0)
    _prettify_ax(ax, grid=False)
    _stretch_note(ax, stretch)
    _add_panel_labels([ax], offset_pt=24.0)
    fig.subplots_adjust(left=0.20, right=0.86, bottom=0.23, top=0.98)
    _save_figure_all(fig, path, dpi=600)
    plt.close(fig)


def save_alpha_gif(times: np.ndarray, alpha_stack: np.ndarray, xc: np.ndarray, yc: np.ndarray, path: str,
                   fps: int = 10, max_frames: int = 120, xmax_m: Optional[float] = 0.10,
                   log_scale: bool = True, alpha_floor: float = 1e-8) -> None:
    nt = len(times)
    if nt == 0:
        return
    frame_idx = np.unique(np.linspace(0, nt - 1, max_frames).astype(int)) if nt > max_frames else np.arange(nt)
    A_full = alpha_stack[frame_idx]
    A, xc_plot = _crop_to_xmax(A_full, xc, xmax_m)
    T = times[frame_idx]
    vmax = max(float(np.nanpercentile(A, 99.5)), 1e-12)
    X, Y = np.meshgrid(xc_plot * 1e3, yc)

    fig, ax = plt.subplots(figsize=_FIELD_FIGSIZE_1)
    if log_scale:
        vmin, vmax_eff = _positive_log_limits(A, vmax=vmax, floor=alpha_floor)
        A_plot = np.clip(A, vmin, None)
        im = ax.pcolormesh(X, Y, A_plot[0], shading="auto", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax_eff))
        cbar_label = r"Gas fraction, $\alpha_g$ (-)"
    else:
        A_plot = A
        im = ax.pcolormesh(X, Y, A_plot[0], shading="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        cbar_label = r"Gas fraction, $\alpha_g$ (-)"

    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"Height, $z$ (m)")
    stretch = _configure_field_axes(ax, xc_plot, yc)
    time_label = ax.text(0.04, 0.03, f"t = {T[0]:.3f} s", transform=ax.transAxes, va="bottom", fontsize=7.5,
                         bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.88, "pad": 1.6, "linewidth": 0.6})
    cbar = fig.colorbar(im, ax=ax, pad=0.05, fraction=0.09)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=7.0)
    _prettify_ax(ax, grid=False)
    _stretch_note(ax, stretch)
    fig.subplots_adjust(left=0.20, right=0.86, bottom=0.13, top=0.98)

    def update(k: int):
        im.set_array(A_plot[k].ravel())
        time_label.set_text(f"t = {T[k]:.3f} s")
        return im, time_label

    anim = FuncAnimation(fig, update, frames=len(frame_idx), blit=False)
    anim.save(_long_path(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2D flow visualization: streamlines and velocity-arrow maps
# -----------------------------------------------------------------------------

def _uniform_regrid(xc: np.ndarray, yc: np.ndarray, field: np.ndarray,
                    n_uniform: int = 70) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a (ny, nx) field from a graded x-mesh onto a uniform x grid.

    matplotlib's streamplot assumes an evenly spaced grid; the solver's mesh
    is refined near the walls, so streamlines drawn directly on xc would be
    geometrically distorted. yc is uniform already (checked, not assumed) and
    is passed through unchanged.
    """
    x_uniform = np.linspace(float(np.min(xc)), float(np.max(xc)), int(n_uniform))
    dy = np.diff(np.asarray(yc, dtype=float))
    y_is_uniform = dy.size == 0 or np.allclose(dy, dy[0], rtol=1e-6, atol=1e-12)
    x_native = np.asarray(xc, dtype=float)
    y_native = np.asarray(yc, dtype=float)
    values = np.asarray(field, dtype=float)
    out = np.vstack([np.interp(x_uniform, x_native, row) for row in values])
    y_uniform = y_native
    if not y_is_uniform:
        # Interpolate each x-column if a future mesh grades y as well.
        y_uniform = np.linspace(float(np.min(yc)), float(np.max(yc)), len(yc))
        out = np.column_stack([
            np.interp(y_uniform, y_native, out[:, i]) for i in range(out.shape[1])
        ])
    return x_uniform, y_uniform, out


def save_streamlines_plot(fields: Dict[str, np.ndarray], xc: np.ndarray, yc: np.ndarray, path: str,
                          xmax_m: Optional[float] = 0.10, n_uniform: int = 70,
                          alpha_floor: float = 1e-8) -> None:
    """Three-panel liquid-flow streamline map: gas fraction, vertical liquid
    velocity (diverging) and liquid speed, each with liquid streamlines
    overlaid so the recirculation pattern is visible against each background.
    """
    alpha_c, xc_plot = _crop_to_xmax(fields["alpha"], xc, xmax_m)
    u1_c, _ = _crop_to_xmax(fields["u1"], xc, xmax_m)
    v1_c, _ = _crop_to_xmax(fields["v1"], xc, xmax_m)

    x_u, y_u, u1_u = _uniform_regrid(xc_plot, yc, u1_c, n_uniform)
    _, _, v1_u = _uniform_regrid(xc_plot, yc, v1_c, n_uniform)
    _, _, alpha_u = _uniform_regrid(xc_plot, yc, alpha_c, n_uniform)
    speed_u = np.sqrt(u1_u ** 2 + v1_u ** 2)
    x_mm = x_u * 1e3
    Xg, Yg = np.meshgrid(x_mm, y_u)

    fig, axes = plt.subplots(1, 3, figsize=_FIELD_FIGSIZE_3)

    # Panel (a): gas fraction background.
    vmin, vmax_eff = _positive_log_limits(alpha_u, floor=alpha_floor)
    im0 = axes[0].pcolormesh(Xg, Yg, np.clip(alpha_u, vmin, None), shading="auto", cmap="viridis",
                             rasterized=True, norm=LogNorm(vmin=vmin, vmax=vmax_eff))
    cb0 = fig.colorbar(im0, ax=axes[0], pad=0.06, fraction=0.09)
    cb0.set_label(r"Gas fraction, $\alpha_g$ (-)")
    cb0.ax.tick_params(labelsize=6.6)

    # Panel (b): vertical liquid velocity, diverging about zero.
    vmax_v = max(float(np.nanpercentile(np.abs(v1_u), 99.0)), 1e-9)
    im1 = axes[1].pcolormesh(Xg, Yg, v1_u, shading="auto", cmap="RdBu_r", rasterized=True,
                             norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_v, vmax=vmax_v))
    cb1 = fig.colorbar(im1, ax=axes[1], pad=0.06, fraction=0.09)
    cb1.set_label(r"Liquid vertical velocity, $v_\ell$ (m s$^{-1}$)")
    cb1.ax.tick_params(labelsize=6.6)

    # Panel (c): liquid speed magnitude.
    vmax_s = max(float(np.nanpercentile(speed_u, 99.5)), 1e-9)
    im2 = axes[2].pcolormesh(Xg, Yg, speed_u, shading="auto", cmap="viridis", rasterized=True,
                             vmin=0.0, vmax=vmax_s)
    cb2 = fig.colorbar(im2, ax=axes[2], pad=0.06, fraction=0.09)
    cb2.set_label(r"Liquid speed, $|\mathbf{U}_\ell|$ (m s$^{-1}$)")
    cb2.ax.tick_params(labelsize=6.6)

    stretches = []
    for ax in axes:
        # streamplot integrates in the axis's own coordinate units, so the
        # horizontal velocity is converted to mm/s to match the mm x-axis.
        ax.streamplot(x_mm, y_u, u1_u * 1e3, v1_u, color="white", density=0.9,
                     linewidth=0.55, arrowsize=0.6, broken_streamlines=False)
        stretches.append(_configure_field_axes(ax, xc_plot, y_u))
        _prettify_ax(ax, grid=False)
        _stretch_note(ax, stretches[-1])
    axes[0].set_ylabel(r"Height, $z$ (m)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    _show_all_tick_numbers(axes)
    fig.supxlabel(r"Distance from electrode, $x$ (mm)", y=0.085)
    _add_panel_labels(axes)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.98, wspace=0.22)
    _save_figure_all(fig, path, dpi=450)
    plt.close(fig)


def save_velocity_quiver_plot(fields: Dict[str, np.ndarray], xc: np.ndarray, yc: np.ndarray, path: str,
                              xmax_m: Optional[float] = 0.10, n_arrows_x: int = 10, n_arrows_y: int = 26,
                              alpha_floor: float = 1e-8) -> None:
    """Single-panel gas-fraction map with liquid-velocity arrows.

    Uses the native (possibly graded) x grid directly -- quiver has no
    uniform-grid requirement, so this is a robust companion to the
    interpolated streamline plot above and a good fallback if the resampling
    in save_streamlines_plot ever looks suspect for a very coarse mesh.
    """
    alpha_c, xc_plot = _crop_to_xmax(fields["alpha"], xc, xmax_m)
    u1_c, _ = _crop_to_xmax(fields["u1"], xc, xmax_m)
    v1_c, _ = _crop_to_xmax(fields["v1"], xc, xmax_m)

    ny, nx = alpha_c.shape
    ix = np.unique(np.linspace(0, nx - 1, min(n_arrows_x, nx)).astype(int))
    iy = np.unique(np.linspace(0, ny - 1, min(n_arrows_y, ny)).astype(int))

    fig, ax = plt.subplots(figsize=_FIELD_FIGSIZE_1)
    Xg, Yg = np.meshgrid(xc_plot * 1e3, yc)
    vmin, vmax_eff = _positive_log_limits(alpha_c, floor=alpha_floor)
    im = ax.pcolormesh(Xg, Yg, np.clip(alpha_c, vmin, None), shading="auto", cmap="viridis",
                       rasterized=True, norm=LogNorm(vmin=vmin, vmax=vmax_eff))
    Xq, Yq = np.meshgrid(xc_plot[ix] * 1e3, yc[iy])
    Uq = u1_c[np.ix_(iy, ix)]
    Vq = v1_c[np.ix_(iy, ix)]
    # Arrows are drawn at unit length (direction only) rather than true
    # magnitude: the vertical liquid velocity varies by two to three orders
    # of magnitude between the near-source region and the bulk, so a
    # magnitude-scaled quiver would make almost every arrow above the
    # entrance region invisible. Magnitude is shown instead by the
    # complementary streamline figure (colour-coded, continuously resolved).
    speed = np.sqrt(Uq ** 2 + Vq ** 2)
    speed_floor = max(float(np.nanpercentile(speed, 5.0)), 1e-12)
    Un = Uq / np.maximum(speed, speed_floor)
    Vn = Vq / np.maximum(speed, speed_floor)
    ax.quiver(Xq, Yq, Un, Vn, color="0.08", scale=1.35 * len(iy), scale_units="height",
             width=0.006, headwidth=3.3, headlength=3.8, pivot="mid",
             edgecolors="white", linewidths=0.35)
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"Height, $z$ (m)")
    stretch = _configure_field_axes(ax, xc_plot, yc)
    cbar = fig.colorbar(im, ax=ax, pad=0.05, fraction=0.09)
    cbar.set_label(r"Gas fraction, $\alpha_g$ (-)")
    cbar.ax.tick_params(labelsize=7.0)
    _prettify_ax(ax, grid=False)
    _stretch_note(ax, stretch)
    _add_panel_labels([ax], offset_pt=24.0)
    fig.subplots_adjust(left=0.20, right=0.86, bottom=0.23, top=0.98)
    _save_figure_all(fig, path, dpi=600)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Boundary-layer metrics and piecewise scaling
# -----------------------------------------------------------------------------

def bl_profile(v_field: np.ndarray, xc: np.ndarray, edge_fraction: float = 0.05) -> Dict[str, np.ndarray]:
    ny, nx = v_field.shape
    delta = np.zeros(ny)
    raw_flux = np.zeros(ny)
    half_width = np.zeros(ny)
    vmax_arr = np.zeros(ny)
    x_vmax = np.zeros(ny)
    plume_edge = np.zeros(ny)

    x = np.asarray(xc, dtype=float)
    for j in range(ny):
        vp = np.maximum(v_field[j, :], 0.0)
        vmax = float(np.max(vp))
        vmax_arr[j] = vmax
        if vmax <= 1e-14:
            continue
        imax = int(np.argmax(vp))
        x_vmax[j] = x[imax]

        iedge = nx - 1
        for i in range(imax, nx):
            if vp[i] <= edge_fraction * vmax:
                iedge = i
                break
        plume_edge[j] = x[iedge]

        xs = x[: iedge + 1]
        vs = vp[: iedge + 1]
        flux = trapz(vs, xs)
        raw_flux[j] = flux
        delta[j] = flux / vmax

        half = x[iedge] - x[imax]
        for i in range(imax, iedge + 1):
            if vp[i] <= 0.5 * vmax:
                half = x[i] - x[imax]
                break
        half_width[j] = max(float(half), 0.0)

    return {
        "delta": delta,
        "raw_flux": raw_flux,
        "half_width": half_width,
        "vmax": vmax_arr,
        "x_vmax": x_vmax,
        "plume_edge": plume_edge,
    }


def mean_profile_from_stack(v1_stack: np.ndarray, xc: np.ndarray) -> Dict[str, np.ndarray]:
    profiles = [bl_profile(v1_stack[k], xc) for k in range(v1_stack.shape[0])]
    keys = profiles[0].keys()
    return {key: np.mean([p[key] for p in profiles], axis=0) for key in keys}


def _valid_bl_mask(yc: np.ndarray, delta: np.ndarray, y_min_frac: float, y_max_frac: float) -> np.ndarray:
    y = np.asarray(yc, dtype=float)
    d = np.asarray(delta, dtype=float)
    y_max = float(np.nanmax(y)) if y.size else 0.0
    return (
        (y > max(1e-12, y_min_frac * y_max))
        & (y < y_max_frac * y_max)
        & np.isfinite(y)
        & np.isfinite(d)
        & (d > 1e-12)
    )


def _continuous_log_piecewise_fit(xlog: np.ndarray, ylog: np.ndarray,
                                  cuts: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    """Fit a continuous segmented line in log space for fixed cut indices.

    A hinge basis keeps adjacent power-law regimes continuous at each
    transition.  This avoids the artificial jumps produced by fitting every
    segment independently while retaining a distinct exponent per regime.
    """
    cut_x = np.asarray(
        [0.5 * (xlog[c - 1] + xlog[c]) for c in cuts], dtype=float
    )
    columns = [np.ones_like(xlog), xlog]
    columns.extend(np.maximum(0.0, xlog - value) for value in cut_x)
    design = np.column_stack(columns)
    try:
        beta, _, rank, _ = np.linalg.lstsq(design, ylog, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < design.shape[1] or not np.all(np.isfinite(beta)):
        return None

    prediction = design @ beta
    residual = ylog - prediction
    sse_total = float(np.sum(residual * residual))
    dof = max(len(xlog) - design.shape[1], 1)
    sigma2 = sse_total / dof
    try:
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
    except np.linalg.LinAlgError:
        covariance = np.full((design.shape[1], design.shape[1]), np.nan)
    return {
        "beta": beta,
        "cut_x": cut_x,
        "prediction": prediction,
        "residual": residual,
        "covariance": covariance,
        "sse_log_total": sse_total,
    }


# Minimum height span (in natural-log units) required of a fitted segment.
# The near-source segment is allowed to be short-lived (the first few cm often
# show a distinct entry/development trend), but every segment further out must
# cover a substantially wider stretch of height so that a "regime" is backed
# by an extended, physically meaningful range rather than a handful of
# neighbouring points -- this avoids chopping the far field into short,
# statistically insignificant slivers.
_MIN_LOG_SPAN_FIRST = 0.30   # ~exp(0.30) = 1.35x in height
_MIN_LOG_SPAN_REST = 0.60    # ~exp(0.60) = 1.82x in height
_MAX_CUT_CANDIDATES = 120    # bounded search; fit itself still uses every valid point


def piecewise_powerlaw_fit(yc: np.ndarray, delta: np.ndarray, segments: int = 2,
                           min_points: int = 5, y_min_frac: float = 0.08,
                           y_max_frac: float = 0.92,
                           min_log_span_first: float = _MIN_LOG_SPAN_FIRST,
                           min_log_span_rest: float = _MIN_LOG_SPAN_REST) -> Optional[Dict[str, Any]]:
    mask = _valid_bl_mask(yc, delta, y_min_frac, y_max_frac)
    y = np.asarray(yc[mask], dtype=float)
    d = np.asarray(delta[mask], dtype=float)
    order = np.argsort(y)
    y = y[order]
    d = d[order]
    npts = len(y)
    if npts < max(segments * min_points, 2 * segments + 1):
        return None

    xlog = np.log(y)
    ylog = np.log(d)
    min_span = [min_log_span_first] + [min_log_span_rest] * (segments - 1)
    best: Optional[Dict[str, Any]] = None

    n_cut_candidates = min(_MAX_CUT_CANDIDATES, max(npts - 2 * min_points + 1, 1))
    target_x = np.linspace(xlog[min_points], xlog[npts - min_points], n_cut_candidates)
    candidate_cuts = np.unique(np.searchsorted(xlog, target_x, side="left"))
    candidate_cuts = candidate_cuts[
        (candidate_cuts >= min_points) & (candidate_cuts <= npts - min_points)
    ]

    if segments == 1:
        cuts_list = [()]
    elif segments == 2:
        cuts_list = [(int(i),) for i in candidate_cuts]
    elif segments == 3:
        cuts_list = []
        for i in candidate_cuts:
            for j in candidate_cuts:
                if int(j) - int(i) >= min_points:
                    cuts_list.append((int(i), int(j)))
    else:
        raise ValueError("segments must be 1, 2 or 3")

    for cuts in cuts_list:
        bounds = (0,) + tuple(cuts) + (npts,)
        ok = True
        for s in range(segments):
            a, b = bounds[s], bounds[s + 1]
            if b - a < min_points:
                ok = False
                break
            if (xlog[b - 1] - xlog[a]) < min_span[s]:
                ok = False
                break
        if not ok:
            continue

        fit = _continuous_log_piecewise_fit(xlog, ylog, tuple(cuts))
        if fit is None:
            continue
        beta = np.asarray(fit["beta"], dtype=float)
        covariance = np.asarray(fit["covariance"], dtype=float)
        prediction = np.asarray(fit["prediction"], dtype=float)
        residual = np.asarray(fit["residual"], dtype=float)
        cut_x = np.asarray(fit["cut_x"], dtype=float)
        sse_total = float(fit["sse_log_total"])
        rows = []
        for s in range(segments):
            a, b = bounds[s], bounds[s + 1]
            slope_weights = np.zeros_like(beta)
            slope_weights[1] = 1.0
            if s:
                slope_weights[2:2 + s] = 1.0
            exponent = float(slope_weights @ beta)
            exponent_var = float(slope_weights @ covariance @ slope_weights)
            exponent_se = math.sqrt(max(exponent_var, 0.0)) if np.isfinite(exponent_var) else float("nan")
            logC = float(beta[0] - np.dot(beta[2:2 + s], cut_x[:s]))
            seg_residual = residual[a:b]
            seg_sse = float(np.sum(seg_residual * seg_residual))
            seg_rmse = float(math.sqrt(seg_sse / max(b - a, 1)))
            seg_y = ylog[a:b]
            ss_tot = float(np.sum((seg_y - np.mean(seg_y)) ** 2))
            seg_r2 = float(1.0 - seg_sse / ss_tot) if ss_tot > 1e-300 else float("nan")
            rows.append({
                "segment_id": s + 1,
                "y_start_m": float(y[a]),
                "y_end_m": float(y[b - 1]),
                "C": float(math.exp(logC)),
                "exponent_n": float(exponent),
                "exponent_se": float(exponent_se),
                "exponent_ci95_low": float(exponent - 1.96 * exponent_se),
                "exponent_ci95_high": float(exponent + 1.96 * exponent_se),
                "sse_log": seg_sse,
                "rmse_log": seg_rmse,
                "r2_log": seg_r2,
                "n_points": int(b - a),
                "y_span_log": float(xlog[b - 1] - xlog[a]),
                "y_span_ratio": float(y[b - 1] / y[a]),
            })
        for row in rows:
            row["low_confidence"] = not _confidence_flag(row)
        # Continuous model: one intercept, one slope per segment and one
        # selected transition location per join.
        k_params = 1 + segments + max(0, segments - 1)
        bic = npts * math.log(max(sse_total / max(npts, 1), 1e-300)) + k_params * math.log(max(npts, 2))
        candidate = {
            "fit_model": "continuous_log_hinge",
            "segments": segments,
            "rows": rows,
            "sse_log_total": float(sse_total),
            "bic": float(bic),
            "transition_y_m": [float(math.exp(value)) for value in cut_x],
            "valid_y_m": y,
            "valid_delta_m": d,
            "predicted_delta_m": np.exp(prediction),
        }
        if best is None or candidate["bic"] < best["bic"]:
            best = candidate
    return best


def choose_piecewise_fit(yc: np.ndarray, delta: np.ndarray, mode: str = "auto",
                         min_points: int = 5, y_min_frac: float = 0.08,
                         y_max_frac: float = 0.92,
                         min_log_span_first: float = _MIN_LOG_SPAN_FIRST,
                         min_log_span_rest: float = _MIN_LOG_SPAN_REST,
                         min_bic_improvement: float = 6.0) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if mode == "auto":
        # Segment counts are tried in order of scientific preference (more
        # regimes resolved first); a plain 1-segment power law is kept as a
        # fallback so a case that cannot support the minimum regime span
        # required for 2-3 segments still gets an overall fit instead of none.
        for seg in (3, 2, 1):
            fit = piecewise_powerlaw_fit(yc, delta, seg, min_points, y_min_frac, y_max_frac,
                                         min_log_span_first, min_log_span_rest)
            if fit is not None:
                candidates.append(fit)
    else:
        seg = int(mode)
        fit = piecewise_powerlaw_fit(yc, delta, seg, min_points, y_min_frac, y_max_frac,
                                     min_log_span_first, min_log_span_rest)
        if fit is not None:
            candidates.append(fit)
    if not candidates:
        return None
    if mode != "auto":
        candidates[0]["bic_improvement_min"] = float(min_bic_improvement)
        return candidates[0]

    # Prefer the simplest adequate model.  Add another regime only when its
    # BIC improves by a meaningful amount, preventing small numerical gains
    # from fragmenting the profile into extra pieces.
    by_segments = {int(item["segments"]): item for item in candidates}
    selected = by_segments[min(by_segments)]
    for seg in sorted(k for k in by_segments if k > int(selected["segments"])):
        candidate = by_segments[seg]
        if float(candidate["bic"]) <= float(selected["bic"]) - float(min_bic_improvement):
            selected = candidate
    selected["bic_improvement_min"] = float(min_bic_improvement)
    return selected


def _bl_xlim(yc: np.ndarray, end_margin_m: float = 0.05, x_min_m: float = 1.0e-2) -> Tuple[float, float]:
    """Boundary-layer x-limits: from x_min_m up to the domain top minus the
    last end_margin_m (default 5 cm), which is usually affected by the outlet
    / free-surface boundary and not representative of the bulk BL scaling."""
    H = float(np.nanmax(yc)) if len(yc) else float("nan")
    if not np.isfinite(H):
        return x_min_m, x_min_m * 35.0
    x_max = max(H - float(end_margin_m), x_min_m * 1.5)
    return float(x_min_m), float(x_max)


def _bl_ylim(delta: np.ndarray, mask: np.ndarray, pad_decades: float = 0.15) -> Optional[Tuple[float, float]]:
    vals = np.asarray(delta[mask], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 2:
        return None
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        return None
    pad = (math.log10(hi) - math.log10(lo)) * pad_decades + pad_decades
    return 10.0 ** (math.log10(lo) - pad), 10.0 ** (math.log10(hi) + pad)


def save_bl_no_fit_plot(yc: np.ndarray, profile: Dict[str, np.ndarray], path: str,
                        y_min_frac: float = 0.08, y_max_frac: float = 0.92,
                        end_margin_m: float = 0.05, x_min_m: float = 1.0e-2) -> None:
    delta = np.asarray(profile["delta"], dtype=float)
    half = np.asarray(profile["half_width"], dtype=float)
    good = _valid_bl_mask(yc, delta, 0.0, 1.0)
    fit_window = _valid_bl_mask(yc, delta, y_min_frac, y_max_frac)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.85))
    ax1.plot(delta, yc, color=_PUB_COLORS[0], lw=1.55, label=r"$\delta_{\mathrm{BL}}$")
    ax1.plot(half, yc, color=_PUB_COLORS[1], lw=1.55, label="Half-width")
    ax1.set_xlabel(r"Width (m)")
    ax1.set_ylabel(r"Height, $z$ (m)")
    _prettify_ax(ax1)
    ax1.legend(loc="best")
    ax2.loglog(yc[good], delta[good], color=_PUB_COLORS[0], marker="o", ms=3.2, lw=1.35,
               markevery=max(1, int(np.count_nonzero(good)) // 12), label=r"$\delta_{\mathrm{BL}}$")
    if np.any(fit_window):
        ax2.loglog(yc[fit_window], delta[fit_window], marker="o", ms=3.1, lw=0.0,
                   markerfacecolor="none", markeredgecolor=_PUB_COLORS[1], label="Fit range")
    ax2.set_xlabel(r"Height, $z$ (m)")
    ax2.set_ylabel(r"Integral width, $\delta_{\mathrm{BL}}$ (m)")
    ax2.set_xlim(*_bl_xlim(yc, end_margin_m, x_min_m))
    ylim = _bl_ylim(delta, good)
    if ylim is not None:
        ax2.set_ylim(*ylim)
    _prettify_ax(ax2)
    ax2.legend(loc="best")
    _add_panel_labels([ax1, ax2], offset_pt=24.0)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.28, top=0.98, wspace=0.34)
    _save_figure_all(fig, path, dpi=600)
    plt.close(fig)

def save_bl_piecewise_plot(yc: np.ndarray, profile: Dict[str, np.ndarray], fit: Optional[Dict[str, Any]], path: str,
                           end_margin_m: float = 0.05, x_min_m: float = 1.0e-2) -> None:
    delta = np.asarray(profile["delta"], dtype=float)
    good = _valid_bl_mask(yc, delta, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(3.55, 2.9))
    ax.loglog(yc[good], delta[good], color=_PUB_COLORS[0], marker="o", ms=3.0, lw=1.3,
              markevery=max(1, int(np.count_nonzero(good)) // 12), label=r"$\delta_{\mathrm{BL}}$", zorder=3)
    n_labels = 1
    if fit is not None:
        for idx, row in enumerate(fit["rows"]):
            yy = np.linspace(float(row["y_start_m"]), float(row["y_end_m"]), 100)
            dd = float(row["C"]) * yy ** float(row["exponent_n"])
            sty = _series_style(idx + 1)
            low_conf = bool(row.get("low_confidence", False))
            ls = ":" if low_conf else "--"
            label = rf"Seg. {row['segment_id']}: $n={row['exponent_n']:.2f}$" + (" (low conf.)" if low_conf else "")
            ax.loglog(yy, dd, ls, color=sty["color"], lw=1.5 if not low_conf else 1.15,
                      alpha=1.0 if not low_conf else 0.75, label=label, zorder=4)
            n_labels += 1
        for ytr in fit.get("transition_y_m", []):
            ax.axvline(float(ytr), color="0.45", ls=":", lw=0.85, zorder=1)
    ax.set_xlabel(r"Height, $z$ (m)")
    ax.set_ylabel(r"Integral width, $\delta_{\mathrm{BL}}$ (m)")
    ax.set_xlim(*_bl_xlim(yc, end_margin_m, x_min_m))
    ylim = _bl_ylim(delta, good)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _prettify_ax(ax)
    # A boxed in-panel legend with up to 4 entries (raw curve + up to 3
    # segments) tends to cover half the data at this figure size, so the
    # legend is placed outside, below the axes (and below the panel label),
    # in two compact columns.
    handles, labels = ax.get_legend_handles_labels()
    _add_panel_labels([ax], offset_pt=24.0)
    _fig_legend_below(fig, [ax], handles, labels,
                      ncol=2 if n_labels > 2 else n_labels, y=0.01)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.34 if n_labels > 2 else 0.30, top=0.98)
    _save_figure_all(fig, path, dpi=600)
    plt.close(fig)

def height_rows(yc: np.ndarray, height_fracs: Sequence[float]) -> List[Tuple[float, int, float]]:
    H = float(np.nanmax(yc)) if len(yc) else 0.0
    rows = []
    used = set()
    for hf in height_fracs:
        target = float(np.clip(hf, 0.0, 1.0)) * H
        j = int(np.argmin(np.abs(yc - target)))
        if j in used:
            continue
        used.add(j)
        rows.append((float(yc[j]), j, float(yc[j] / H) if H > 0 else float("nan")))
    return rows


def find_reverse_flow_boundary(xc: np.ndarray, v_profile: np.ndarray, gap_m: float,
                               eps_abs: float = 1e-10, eps_rel: float = 1e-3) -> Dict[str, float]:
    x = np.asarray(xc, dtype=float)
    v = np.asarray(v_profile, dtype=float)
    finite = np.isfinite(x) & np.isfinite(v)
    x = x[finite]
    v = v[finite]
    if x.size < 2:
        return {"has_upward_plume": 0.0, "has_reverse_flow": 0.0, "x_zero_reverse_m": np.nan,
                "plume_to_gap_fraction": np.nan, "vmax_m_s": np.nan, "vmin_after_peak_m_s": np.nan, "x_vmax_m": np.nan}

    vmax = float(np.nanmax(v))
    vmin = float(np.nanmin(v))
    tol = max(float(eps_abs), abs(vmax) * float(eps_rel))
    if not np.isfinite(vmax) or vmax <= tol:
        return {"has_upward_plume": 0.0, "has_reverse_flow": 0.0, "x_zero_reverse_m": np.nan,
                "plume_to_gap_fraction": 0.0, "vmax_m_s": vmax, "vmin_after_peak_m_s": vmin, "x_vmax_m": np.nan}

    imax = int(np.nanargmax(v))
    x_vmax = float(x[imax])
    x0 = np.nan
    has_reverse = 0.0
    vmin_after = float(np.nanmin(v[imax:])) if imax < len(v) else np.nan

    for i in range(imax + 1, len(v)):
        if v[i - 1] > tol and v[i] <= 0.0:
            # Linear interpolation of v=0 between i-1 and i.
            denom = v[i] - v[i - 1]
            if abs(denom) > 1e-30:
                x0 = float(x[i - 1] - v[i - 1] * (x[i] - x[i - 1]) / denom)
            else:
                x0 = float(x[i])
            if np.nanmin(v[i:]) < -tol:
                has_reverse = 1.0
            break

    if not np.isfinite(x0):
        # No negative return flow after the upward plume peak inside the resolved gap.
        x0 = float(gap_m) if np.isfinite(gap_m) and gap_m > 0 else float(np.nanmax(x))
        has_reverse = 0.0

    denom_gap = float(gap_m) if np.isfinite(gap_m) and gap_m > 0 else float(np.nanmax(x) - np.nanmin(x))
    frac = float(x0 / denom_gap) if denom_gap > 0 and np.isfinite(x0) else float("nan")
    return {
        "has_upward_plume": 1.0,
        "has_reverse_flow": has_reverse,
        "x_zero_reverse_m": float(x0),
        "plume_to_gap_fraction": frac,
        "vmax_m_s": vmax,
        "vmin_after_peak_m_s": vmin_after,
        "x_vmax_m": x_vmax,
    }


def _apply_xlimit(ax, xlim_mm: Optional[float]) -> None:
    if xlim_mm is not None and xlim_mm > 0:
        ax.set_xlim(0.0, float(xlim_mm))


def plot_velocity_profiles(rows: List[Tuple[float, int, float]], xc: np.ndarray, fields: Dict[str, np.ndarray],
                           reverse_by_j: Dict[int, Dict[str, float]], path: str,
                           xlim_mm: Optional[float] = 12.0) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), sharex=True)
    xmm = xc * 1e3
    for idx, (yv, j, hf) in enumerate(rows):
        sty = _series_style(idx)
        label = rf"$z/H={hf:.2f}$"
        axes[0].plot(xmm, fields["v1"][j], color=sty["color"], lw=1.45, label=label)
        axes[1].plot(xmm, fields["v2"][j], color=sty["color"], lw=1.45, label=label)
        axes[2].plot(xmm, fields["u1"][j], color=sty["color"], lw=1.45, label=label)
        rev = reverse_by_j.get(j, {})
        if rev and np.isfinite(rev.get("x_zero_reverse_m", np.nan)):
            axes[0].plot([rev["x_zero_reverse_m"] * 1e3], [0.0], marker=sty["marker"],
                         color=sty["color"], ms=3.8, ls="none")
    ylabels = [
        r"$v_\ell$ (m s$^{-1}$)",
        r"$v_g$ (m s$^{-1}$)",
        r"$u_\ell$ (m s$^{-1}$)",
    ]
    for ax, yl in zip(axes, ylabels):
        ax.axhline(0.0, color="0.35", lw=0.75)
        ax.set_ylabel(yl)
        _apply_xlimit(ax, xlim_mm)
        _prettify_ax(ax)
    _show_all_tick_numbers(axes)
    fig.supxlabel(r"Distance from electrode, $x$ (mm)", y=0.105)
    handles, labels = _collect_legend_entries(axes)
    _add_panel_labels(axes)
    _fig_legend_below(fig, axes, handles, labels, ncol=min(4, len(labels)), y=0.01)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.97, wspace=0.38)
    _save_figure_all(fig, path, dpi=600)
    plt.close(fig)

def plot_single_profile_quantity(rows: List[Tuple[float, int, float]], xc: np.ndarray, data_by_name: Dict[str, np.ndarray],
                                 ylabel: str, path: str, xlim_mm: Optional[float] = 12.0,
                                 scale: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    xmm = xc * 1e3
    for idx, (yv, j, hf) in enumerate(rows):
        sty = _series_style(idx)
        ax.plot(xmm, data_by_name[j] * scale, color=sty["color"], lw=1.5, label=rf"$z/H={hf:.2f}$")
    ax.set_xlabel(r"Distance from electrode, $x$ (mm)")
    ax.set_ylabel(ylabel)
    _apply_xlimit(ax, xlim_mm)
    _prettify_ax(ax)
    ax.legend(loc="best")
    _add_panel_labels([ax], offset_pt=24.0)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.98)
    _save_figure_all(fig, path, dpi=600)
    plt.close(fig)

def write_profiles_csv(rows: List[Tuple[float, int, float]], xc: np.ndarray, fields: Dict[str, np.ndarray], path: str) -> None:
    headers = ["x_m"]
    cols: List[np.ndarray] = [np.asarray(xc, dtype=float)]
    for (yv, j, _) in rows:
        tag = f"y{yv:.3f}"
        du = fields["u2"][j] - fields["u1"][j]
        dv = fields["v2"][j] - fields["v1"][j]
        slip = np.sqrt(du * du + dv * dv)
        for name, data in (
            ("alpha", fields["alpha"][j]),
            ("u_l", fields["u1"][j]),
            ("v_l", fields["v1"][j]),
            ("u_g", fields["u2"][j]),
            ("v_g", fields["v2"][j]),
            ("slip", slip),
            ("nu_t", fields["nut"][j]),
        ):
            headers.append(f"{name}_{tag}")
            cols.append(np.asarray(data, dtype=float))
    with _open_w(path) as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for i in range(len(xc)):
            writer.writerow([c[i] for c in cols])


def write_velocity_profiles_long_csv(rows: List[Tuple[float, int, float]],
                                     xc: np.ndarray,
                                     yc: np.ndarray,
                                     fields_by_state: Dict[str, Dict[str, np.ndarray]],
                                     meta: CaseMeta,
                                     path: str) -> None:
    """Write selected velocity/alpha profiles in a tidy format for sweep analysis.

    This is the bridge between per-case post-processing and the sweep-level
    analysis script. It keeps every x point in the mesh. Plot x-limits are never
    applied to the CSV.
    """
    meta_dict = _meta_columns(meta)
    meta_cols = list(meta_dict.keys())
    H = float(np.nanmax(yc)) if len(yc) else float("nan")
    gap_m = float(meta.gap_m) if np.isfinite(meta.gap_m) and meta.gap_m > 0 else float("nan")

    headers = meta_cols + [
        "state", "height_m", "height_over_H", "row_index",
        "x_m", "x_mm", "x_over_gap", "gap_fraction",
        "alpha", "u_liquid_mps", "v_liquid_mps", "u_gas_mps", "v_gas_mps",
        "slip_mps", "nut_m2s",
    ]

    with _open_w(path) as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for state, fields in fields_by_state.items():
            for (yv, j, y_norm) in rows:
                du = fields["u2"][j] - fields["u1"][j]
                dv = fields["v2"][j] - fields["v1"][j]
                slip = np.sqrt(du * du + dv * dv)
                hnorm = float(y_norm) if np.isfinite(float(y_norm)) else (float(yv) / H if H > 0 else float("nan"))
                for i, x in enumerate(xc):
                    x = float(x)
                    x_over_gap = x / gap_m if np.isfinite(gap_m) and gap_m > 0 else float("nan")
                    writer.writerow(list(meta_dict.values()) + [
                        state, float(yv), hnorm, int(j),
                        x, x * 1e3, x_over_gap, x_over_gap,
                        float(fields["alpha"][j, i]),
                        float(fields["u1"][j, i]),
                        float(fields["v1"][j, i]),
                        float(fields["u2"][j, i]),
                        float(fields["v2"][j, i]),
                        float(slip[i]),
                        float(fields["nut"][j, i]),
                    ])


# -----------------------------------------------------------------------------
# CSV outputs
# -----------------------------------------------------------------------------

def _meta_columns(meta: CaseMeta) -> Dict[str, Any]:
    return {
        "case_name": meta.case_name,
        "case_tag": meta.tag,
        "current_a": meta.current_a,
        "current_density_a_m2": meta.current_density_a_m2,
        "inflow_cms": meta.inflow_cms,
        "gap_cm": meta.gap_cm,
        "gap_m": meta.gap_m,
        "variant": meta.variant,
        "reference_class": meta.reference_class,
    }


def write_bl_profiles_csv(yc: np.ndarray, profiles: Dict[str, Dict[str, np.ndarray]], meta: CaseMeta, path: str) -> None:
    fields = ["delta", "raw_flux", "half_width", "vmax", "x_vmax", "plume_edge"]
    meta_cols = list(_meta_columns(meta).keys())
    with _open_w(path) as fh:
        writer = csv.writer(fh)
        header = meta_cols + ["state", "y_m", "y_over_H"] + fields
        writer.writerow(header)
        H = float(np.nanmax(yc)) if len(yc) else float("nan")
        for state, prof in profiles.items():
            for j, y in enumerate(yc):
                meta_vals = list(_meta_columns(meta).values())
                row = meta_vals + [state, float(y), float(y / H) if H > 0 else float("nan")]
                row += [float(prof[f][j]) for f in fields]
                writer.writerow(row)


def write_piecewise_csv(fits: Dict[str, Optional[Dict[str, Any]]], meta: CaseMeta, H: float, path: str) -> None:
    meta_cols = list(_meta_columns(meta).keys())
    fit_cols = [
        "state", "fit_model", "selected_segments", "segment_id", "C", "exponent_n",
        "exponent_se", "exponent_ci95_low", "exponent_ci95_high", "y_start_m", "y_end_m",
        "y_start_over_H", "y_end_over_H", "n_points", "y_span_log", "y_span_ratio",
        "sse_log", "rmse_log", "r2_log", "low_confidence", "sse_log_total", "bic",
        "bic_improvement_min",
        "transition_before_m", "transition_after_m", "transition_before_over_H", "transition_after_over_H",
    ]
    with _open_w(path) as fh:
        writer = csv.writer(fh)
        writer.writerow(meta_cols + fit_cols)
        for state, fit in fits.items():
            if fit is None:
                writer.writerow(list(_meta_columns(meta).values()) + [state] + [""] * (len(fit_cols) - 1))
                continue
            transitions = list(fit.get("transition_y_m", []))
            for row in fit["rows"]:
                sid = int(row["segment_id"])
                tb = transitions[sid - 2] if sid >= 2 and sid - 2 < len(transitions) else float("nan")
                ta = transitions[sid - 1] if sid - 1 < len(transitions) else float("nan")
                writer.writerow(list(_meta_columns(meta).values()) + [
                    state, str(fit.get("fit_model", "")), int(fit["segments"]), sid,
                    float(row["C"]), float(row["exponent_n"]),
                    float(row.get("exponent_se", float("nan"))),
                    float(row.get("exponent_ci95_low", float("nan"))),
                    float(row.get("exponent_ci95_high", float("nan"))),
                    float(row["y_start_m"]), float(row["y_end_m"]),
                    float(row["y_start_m"]) / H if H > 0 else float("nan"),
                    float(row["y_end_m"]) / H if H > 0 else float("nan"),
                    int(row["n_points"]), float(row.get("y_span_log", float("nan"))),
                    float(row.get("y_span_ratio", float("nan"))),
                    float(row["sse_log"]), float(row.get("rmse_log", float("nan"))),
                    float(row.get("r2_log", float("nan"))),
                    bool(row.get("low_confidence", False)),
                    float(fit["sse_log_total"]), float(fit["bic"]),
                    float(fit.get("bic_improvement_min", float("nan"))),
                    tb, ta, tb / H if H > 0 and np.isfinite(tb) else float("nan"),
                    ta / H if H > 0 and np.isfinite(ta) else float("nan"),
                ])


def write_reverse_flow_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with _open_w(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_reverse_rows(state: str, field: Dict[str, np.ndarray], xc: np.ndarray, yc: np.ndarray,
                      selected_rows: Optional[List[Tuple[float, int, float]]], meta: CaseMeta,
                      all_heights: bool = False) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, float]]]:
    H = float(np.nanmax(yc)) if len(yc) else float("nan")
    meta_dict = _meta_columns(meta)
    if all_heights:
        iter_rows = [(float(y), int(j), float(y / H) if H > 0 else float("nan")) for j, y in enumerate(yc)]
    else:
        iter_rows = selected_rows or []
    out: List[Dict[str, Any]] = []
    by_j: Dict[int, Dict[str, float]] = {}
    for yv, j, y_norm in iter_rows:
        rev = find_reverse_flow_boundary(xc, field["v1"][j], meta.gap_m)
        by_j[j] = rev
        row = {
            **meta_dict,
            "state": state,
            "y_m": yv,
            "y_over_H": y_norm,
            **rev,
        }
        out.append(row)
    return out, by_j


def write_case_metrics_csv(meta: CaseMeta, times: np.ndarray, yc: np.ndarray, xc: np.ndarray,
                           fits: Dict[str, Optional[Dict[str, Any]]], reverse_rows: List[Dict[str, Any]], path: str) -> None:
    H = float(np.nanmax(yc)) if len(yc) else float("nan")
    dx = float(np.nanmedian(np.diff(xc))) if len(xc) > 1 else float("nan")
    selected_mean_reverse = [r for r in reverse_rows if r.get("state") == "time_mean_field"]
    fracs = [float(r["plume_to_gap_fraction"]) for r in selected_mean_reverse if np.isfinite(float(r["plume_to_gap_fraction"]))]
    mean_frac = float(np.nanmean(fracs)) if fracs else float("nan")
    max_frac = float(np.nanmax(fracs)) if fracs else float("nan")
    has_rev_count = int(sum(1 for r in selected_mean_reverse if float(r.get("has_reverse_flow", 0.0)) > 0.5))

    fit_mean = fits.get("time_mean_field")
    row = {
        **_meta_columns(meta),
        "project_root": _PROJECT_ROOT,
        "snapshots_used": int(len(times)),
        "time_start_s": float(times[0]) if len(times) else float("nan"),
        "time_end_s": float(times[-1]) if len(times) else float("nan"),
        "nx": int(len(xc)),
        "ny": int(len(yc)),
        "dx_m_median": dx,
        "height_H_m": H,
        "selected_piecewise_segments_mean": int(fit_mean["segments"]) if fit_mean else "",
        "piecewise_transition_1_mean_m": fit_mean["transition_y_m"][0] if fit_mean and len(fit_mean.get("transition_y_m", [])) > 0 else "",
        "piecewise_transition_2_mean_m": fit_mean["transition_y_m"][1] if fit_mean and len(fit_mean.get("transition_y_m", [])) > 1 else "",
        "mean_selected_plume_to_gap_fraction": mean_frac,
        "max_selected_plume_to_gap_fraction": max_frac,
        "n_selected_heights_with_reverse_flow": has_rev_count,
    }
    with _open_w(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified plume post-processing with BL scaling and reverse-flow metrics.")
    parser.add_argument("--case-name", type=str, default=None, help="Case label written to CSV files. Default: folder name.")
    parser.add_argument("--current-a", type=float, default=None, help="Input current [A], written to all CSV files.")
    parser.add_argument("--current-density-a-m2", type=float, default=None,
                        help="Electrode current density [A/m^2]. Normally read from case_metadata.json.")
    parser.add_argument("--inflow-cms", type=float, default=None, help="Bottom inflow velocity [cm/s], written to all CSV files.")
    parser.add_argument("--gap-cm", type=float, default=None, help="Channel gap/width [cm], written to all CSV files and used for plume/gap fraction.")
    parser.add_argument("--variant", type=str, default=None,
                        help="Closure variant label. Normally read from case_metadata.json.")
    parser.add_argument("--reference-class", type=str, default=None,
                        help="confined_channel or far_wall_reference; normally automatic.")
    parser.add_argument("--heights", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9],
                        help="Heights as fractions of H for profile and selected reverse-flow CSV. Default: 0.25 0.5 0.75 0.9")
    parser.add_argument("--skip-fraction", type=float, default=0.0,
                        help="Fraction of initial snapshots to skip for time means. Default: 0.0")
    parser.add_argument("--gif-fps", type=int, default=10, help="GIF frames per second. Default: 10")
    parser.add_argument("--max-gif-frames", type=int, default=120, help="Maximum frames in GIF. Default: 120")
    parser.add_argument("--alpha-xmax-m", type=float, default=0.10,
                        help="Horizontal crop for alpha plots/GIF [m]. Default: 0.10")
    parser.add_argument("--alpha-linear", action="store_true", help="Use linear alpha color scale instead of logarithmic scale.")
    parser.add_argument("--alpha-floor", type=float, default=1e-8, help="Minimum positive alpha for LogNorm. Default: 1e-8")
    parser.add_argument("--xlim-mm", type=float, default=12.0,
                        help="x-axis limit for profile plots [mm]. Default: 12 mm. Use <=0 for automatic.")
    parser.add_argument("--nu-l", type=float, default=None, help="Molecular kinematic viscosity for nu_t reference line.")
    parser.add_argument("--piecewise-segments", type=str, default="auto", choices=["auto", "2", "3"],
                        help="Piecewise BL fit segments. Default auto chooses 2 or 3 by BIC in log-space.")
    parser.add_argument("--piecewise-min-points", type=int, default=8,
                        help="Minimum points per piecewise segment. Default: 8")
    parser.add_argument("--piecewise-min-log-span-first", type=float, default=_MIN_LOG_SPAN_FIRST,
                        help="Minimum ln(y_end/y_start) span of the first (near-source) BL segment. "
                             f"Default: {_MIN_LOG_SPAN_FIRST:g}")
    parser.add_argument("--piecewise-min-log-span-rest", type=float, default=_MIN_LOG_SPAN_REST,
                        help="Minimum ln(y_end/y_start) span required of every later BL segment, kept "
                             "wider than the first so far-field regimes are not chopped into short, "
                             f"insignificant pieces. Default: {_MIN_LOG_SPAN_REST:g}")
    parser.add_argument("--piecewise-bic-improvement", type=float, default=6.0,
                        help="Minimum BIC decrease required before adding another regime. Default: 6")
    parser.add_argument("--fit-y-min-m", type=float, default=1.0e-2,
                        help="Lower physical height included in the BL fit [m]. Default: 1e-2")
    parser.add_argument("--fit-y-min-frac", type=float, default=None,
                        help="Optional lower y/H fit bound; overrides --fit-y-min-m when supplied.")
    parser.add_argument("--fit-y-max-frac", type=float, default=1.0,
                        help="Optional upper y/H cap, applied in addition to --bl-end-margin-m. Default: 1")
    parser.add_argument("--bl-x-min-m", type=float, default=1.0e-2,
                        help="Lower height displayed in BL scaling plots [m]. Default: 1e-2")
    parser.add_argument("--bl-end-margin-m", type=float, default=0.05,
                        help="Height trimmed off the top of the domain in BL scaling plots [m] "
                             "(outlet/free-surface region is not representative). Default: 0.05")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF creation.")
    parser.add_argument("--no-streamlines", action="store_true", help="Skip the 2D streamline figure.")
    parser.add_argument("--no-quiver", action="store_true", help="Skip the 2D velocity-arrow figure.")
    parser.add_argument("--stream-n-uniform", type=int, default=70,
                        help="Uniform-grid resolution (x) used to resample velocity for streamplot. Default: 70")
    args = parser.parse_args()

    _makedirs(SAVE_DIR)
    files = numeric_field_files(OUT_DIR)
    times, stacks, xc, yc = load_stacks(files, skip_fraction=args.skip_fraction)
    meta = infer_case_metadata(args, xc)
    H = float(np.nanmax(yc)) if len(yc) else float("nan")
    if not np.isfinite(H) or H <= 0.0:
        parser.error("the mesh does not contain a positive domain height")
    if args.piecewise_min_points < 3:
        parser.error("--piecewise-min-points must be at least 3")
    if args.bl_x_min_m <= 0.0 or args.fit_y_min_m <= 0.0:
        parser.error("--bl-x-min-m and --fit-y-min-m must be positive")
    if args.bl_end_margin_m < 0.0 or args.bl_end_margin_m >= H:
        parser.error("--bl-end-margin-m must lie between 0 and the domain height")

    fit_y_min_frac = (
        float(args.fit_y_min_frac)
        if args.fit_y_min_frac is not None
        else float(args.fit_y_min_m) / H
    )
    fit_y_max_frac = min(
        float(args.fit_y_max_frac),
        max((H - float(args.bl_end_margin_m)) / H, 0.0),
    )
    if not 0.0 <= fit_y_min_frac < fit_y_max_frac <= 1.0:
        parser.error("effective BL fit bounds are invalid; check the fit limits and end margin")

    with _open_w(os.path.join(SAVE_DIR, f"case_conditions_{meta.tag}.json"), encoding="utf-8") as fh:
        json.dump(_meta_columns(meta), fh, indent=2)

    final = {k: v[-1] for k, v in stacks.items()}
    mean = {k: np.mean(v, axis=0) for k, v in stacks.items()}
    rows = height_rows(yc, args.heights)
    nu_l = args.nu_l if args.nu_l is not None else read_nu_l()
    xlim_mm = args.xlim_mm if args.xlim_mm and args.xlim_mm > 0 else None

    # 2D alpha plots and GIF.
    vmax_alpha = max(float(np.nanpercentile(stacks["alpha"], 99.5)), 1e-12)
    save_alpha_colormap(final["alpha"], xc, yc, os.path.join(SAVE_DIR, "alpha_snapshot_final.png"),
                        vmax=vmax_alpha, xmax_m=args.alpha_xmax_m,
                        log_scale=not args.alpha_linear, alpha_floor=args.alpha_floor)
    save_alpha_colormap(mean["alpha"], xc, yc, os.path.join(SAVE_DIR, "alpha_time_mean.png"),
                        vmax=vmax_alpha, xmax_m=args.alpha_xmax_m,
                        log_scale=not args.alpha_linear, alpha_floor=args.alpha_floor)
    if not args.no_gif:
        save_alpha_gif(times, stacks["alpha"], xc, yc, os.path.join(SAVE_DIR, "alpha_evolution.gif"),
                       fps=args.gif_fps, max_frames=args.max_gif_frames,
                       xmax_m=args.alpha_xmax_m, log_scale=not args.alpha_linear,
                       alpha_floor=args.alpha_floor)

    # 2D streamline / velocity-arrow maps (final snapshot and time-mean field).
    if not args.no_streamlines:
        save_streamlines_plot(final, xc, yc, os.path.join(SAVE_DIR, "streamlines_final.png"),
                              xmax_m=args.alpha_xmax_m, n_uniform=args.stream_n_uniform,
                              alpha_floor=args.alpha_floor)
        save_streamlines_plot(mean, xc, yc, os.path.join(SAVE_DIR, "streamlines_time_mean.png"),
                              xmax_m=args.alpha_xmax_m, n_uniform=args.stream_n_uniform,
                              alpha_floor=args.alpha_floor)
    if not args.no_quiver:
        save_velocity_quiver_plot(final, xc, yc, os.path.join(SAVE_DIR, "velocity_quiver_final.png"),
                                  xmax_m=args.alpha_xmax_m, alpha_floor=args.alpha_floor)
        save_velocity_quiver_plot(mean, xc, yc, os.path.join(SAVE_DIR, "velocity_quiver_time_mean.png"),
                                  xmax_m=args.alpha_xmax_m, alpha_floor=args.alpha_floor)

    # Reverse flow rows and profile plots.
    rev_final_rows, rev_final_by_j = make_reverse_rows("final", final, xc, yc, rows, meta, all_heights=False)
    rev_mean_rows, rev_mean_by_j = make_reverse_rows("time_mean_field", mean, xc, yc, rows, meta, all_heights=False)
    reverse_selected = rev_final_rows + rev_mean_rows
    write_reverse_flow_csv(reverse_selected, os.path.join(SAVE_DIR, f"reverse_flow_{meta.tag}.csv"))

    rev_all_final, _ = make_reverse_rows("final", final, xc, yc, rows, meta, all_heights=True)
    rev_all_mean, _ = make_reverse_rows("time_mean_field", mean, xc, yc, rows, meta, all_heights=True)
    write_reverse_flow_csv(rev_all_final + rev_all_mean, os.path.join(SAVE_DIR, f"reverse_flow_all_heights_{meta.tag}.csv"))

    plot_velocity_profiles(rows, xc, final, rev_final_by_j, os.path.join(SAVE_DIR, "profiles_velocity_final.png"), xlim_mm=xlim_mm)
    plot_velocity_profiles(rows, xc, mean, rev_mean_by_j, os.path.join(SAVE_DIR, "profiles_velocity_mean.png"), xlim_mm=xlim_mm)

    # Other profile plots.
    alpha_final_by_j = {j: final["alpha"][j] for _, j, _ in rows}
    alpha_mean_by_j = {j: mean["alpha"][j] for _, j, _ in rows}
    plot_single_profile_quantity(rows, xc, alpha_final_by_j, r"Gas fraction, $\alpha_g$ (-)",
                                 os.path.join(SAVE_DIR, "profiles_alpha_final.png"), xlim_mm=xlim_mm)
    plot_single_profile_quantity(rows, xc, alpha_mean_by_j, r"Gas fraction, $\alpha_g$ (-)",
                                 os.path.join(SAVE_DIR, "profiles_alpha_mean.png"), xlim_mm=xlim_mm)

    slip_final_by_j = {}
    slip_mean_by_j = {}
    for _, j, _ in rows:
        slip_final_by_j[j] = np.sqrt((final["u2"][j] - final["u1"][j])**2 + (final["v2"][j] - final["v1"][j])**2)
        slip_mean_by_j[j] = np.sqrt((mean["u2"][j] - mean["u1"][j])**2 + (mean["v2"][j] - mean["v1"][j])**2)
    plot_single_profile_quantity(rows, xc, slip_final_by_j, r"Slip velocity, $|\mathbf{U}_g-\mathbf{U}_\ell|$ (mm s$^{-1}$)",
                                 os.path.join(SAVE_DIR, "profiles_slip_final.png"), xlim_mm=xlim_mm, scale=1e3)
    plot_single_profile_quantity(rows, xc, slip_mean_by_j, r"Slip velocity, $|\mathbf{U}_g-\mathbf{U}_\ell|$ (mm s$^{-1}$)",
                                 os.path.join(SAVE_DIR, "profiles_slip_mean.png"), xlim_mm=xlim_mm, scale=1e3)

    nut_final_by_j = {j: final["nut"][j] for _, j, _ in rows}
    nut_mean_by_j = {j: mean["nut"][j] for _, j, _ in rows}
    # Manual nu_t plots, to include molecular reference line.
    for data, name in [(nut_final_by_j, "profiles_nut_final.png"), (nut_mean_by_j, "profiles_nut_mean.png")]:
        fig, ax = plt.subplots(figsize=(3.55, 2.75))
        xmm = xc * 1e3
        for idx, (yv, j, hf) in enumerate(rows):
            sty = _series_style(idx)
            ax.plot(xmm, data[j], color=sty["color"], lw=1.5, label=rf"$z/H={hf:.2f}$")
        ax.axhline(float(nu_l), color="0.30", ls="--", lw=0.95, label=r"Molecular $\nu_\ell$")
        ax.set_xlabel(r"Distance from electrode, $x$ (mm)")
        ax.set_ylabel(r"Turbulent kinematic viscosity, $\nu_t$ (m$^2$ s$^{-1}$)")
        _apply_xlimit(ax, xlim_mm)
        _prettify_ax(ax)
        ax.legend(loc="best")
        _add_panel_labels([ax], offset_pt=24.0)
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.25, top=0.98)
        _save_figure_all(fig, os.path.join(SAVE_DIR, name), dpi=600)
        plt.close(fig)

    write_profiles_csv(rows, xc, final, os.path.join(SAVE_DIR, f"profiles_final_{meta.tag}.csv"))
    write_profiles_csv(rows, xc, mean, os.path.join(SAVE_DIR, f"profiles_mean_{meta.tag}.csv"))
    write_velocity_profiles_long_csv(
        rows, xc, yc, {"final": final, "time_mean_field": mean}, meta,
        os.path.join(SAVE_DIR, f"velocity_profiles_{meta.tag}.csv"),
    )

    # BL profiles, no-fit plots and piecewise fits.
    bl_last = bl_profile(final["v1"], xc)
    bl_mean_field = bl_profile(mean["v1"], xc)
    bl_mean_profiles = mean_profile_from_stack(stacks["v1"], xc)
    profiles_for_csv = {
        "final": bl_last,
        "time_mean_field": bl_mean_field,
        "mean_of_instantaneous": bl_mean_profiles,
    }
    write_bl_profiles_csv(yc, profiles_for_csv, meta, os.path.join(SAVE_DIR, f"bl_profiles_{meta.tag}.csv"))

    save_bl_no_fit_plot(yc, bl_last, os.path.join(SAVE_DIR, "bl_no_fit_final.png"),
                       fit_y_min_frac, fit_y_max_frac, args.bl_end_margin_m, args.bl_x_min_m)
    save_bl_no_fit_plot(yc, bl_mean_field, os.path.join(SAVE_DIR, "bl_no_fit_mean.png"),
                       fit_y_min_frac, fit_y_max_frac, args.bl_end_margin_m, args.bl_x_min_m)

    fits = {
        "final": choose_piecewise_fit(yc, bl_last["delta"], args.piecewise_segments,
                                      args.piecewise_min_points, fit_y_min_frac, fit_y_max_frac,
                                      args.piecewise_min_log_span_first, args.piecewise_min_log_span_rest,
                                      args.piecewise_bic_improvement),
        "time_mean_field": choose_piecewise_fit(yc, bl_mean_field["delta"], args.piecewise_segments,
                                                args.piecewise_min_points, fit_y_min_frac, fit_y_max_frac,
                                                args.piecewise_min_log_span_first, args.piecewise_min_log_span_rest,
                                                args.piecewise_bic_improvement),
        "mean_of_instantaneous": choose_piecewise_fit(yc, bl_mean_profiles["delta"], args.piecewise_segments,
                                                      args.piecewise_min_points, fit_y_min_frac, fit_y_max_frac,
                                                      args.piecewise_min_log_span_first, args.piecewise_min_log_span_rest,
                                                      args.piecewise_bic_improvement),
    }
    save_bl_piecewise_plot(yc, bl_last, fits["final"], os.path.join(SAVE_DIR, "bl_piecewise_final.png"),
                           args.bl_end_margin_m, args.bl_x_min_m)
    save_bl_piecewise_plot(yc, bl_mean_field, fits["time_mean_field"], os.path.join(SAVE_DIR, "bl_piecewise_mean.png"),
                           args.bl_end_margin_m, args.bl_x_min_m)
    write_piecewise_csv(fits, meta, H, os.path.join(SAVE_DIR, f"bl_piecewise_fits_{meta.tag}.csv"))

    write_case_metrics_csv(meta, times, yc, xc, fits, reverse_selected, os.path.join(SAVE_DIR, f"case_metrics_{meta.tag}.csv"))

    with _open_w(os.path.join(SAVE_DIR, "postprocess_summary.txt"), encoding="utf-8") as fh:
        fh.write("Unified plume post-processing summary\n")
        fh.write("=====================================\n")
        fh.write(f"project_root: {_PROJECT_ROOT}\n")
        fh.write(f"output_directory: {OUT_DIR}\n")
        fh.write(f"save_directory: {SAVE_DIR}\n")
        fh.write(f"case_tag: {meta.tag}\n")
        fh.write(f"current_a: {meta.current_a}\n")
        fh.write(f"current_density_a_m2: {meta.current_density_a_m2}\n")
        fh.write(f"inflow_cms: {meta.inflow_cms}\n")
        fh.write(f"gap_cm: {meta.gap_cm}\n")
        fh.write(f"variant: {meta.variant}\n")
        fh.write(f"reference_class: {meta.reference_class}\n")
        fh.write(f"snapshots_used: {len(times)}\n")
        fh.write(f"time_range_s: {times[0]:.8g} to {times[-1]:.8g}\n")
        fh.write(f"skip_fraction: {args.skip_fraction}\n")
        fh.write(f"mesh: nx={len(xc)}, ny={len(yc)}\n")
        fh.write(f"profile_xlim_mm: {xlim_mm}\n")
        fh.write(f"alpha_xmax_m: {args.alpha_xmax_m}\n")
        fh.write(f"piecewise_segments: {args.piecewise_segments}\n")
        fh.write(f"piecewise_min_points: {args.piecewise_min_points}\n")
        fh.write(f"piecewise_min_log_span_first: {args.piecewise_min_log_span_first}\n")
        fh.write(f"piecewise_min_log_span_rest: {args.piecewise_min_log_span_rest}\n")
        fh.write(f"piecewise_bic_improvement: {args.piecewise_bic_improvement}\n")
        fh.write(f"fit_y_min_m_effective: {fit_y_min_frac * H}\n")
        fh.write(f"fit_y_max_m_effective: {fit_y_max_frac * H}\n")
        fh.write(f"bl_x_min_m: {args.bl_x_min_m}\n")
        fh.write(f"bl_end_margin_m: {args.bl_end_margin_m}\n")
        for state, fit in fits.items():
            fh.write(f"\nBL piecewise fit state={state}\n")
            if fit is None:
                fh.write("  no valid fit\n")
            else:
                fh.write(f"  selected_segments: {fit['segments']}\n")
                fh.write(f"  transitions_y_m: {fit.get('transition_y_m', [])}\n")
                for row in fit["rows"]:
                    conf = "low confidence" if row.get("low_confidence") else "ok"
                    fh.write(
                        f"  segment {row['segment_id']}: C={row['C']:.8e}, n={row['exponent_n']:.6f}, "
                        f"y=[{row['y_start_m']:.6g}, {row['y_end_m']:.6g}] m, "
                        f"r2_log={row.get('r2_log', float('nan')):.4f}, n_points={row['n_points']}, "
                        f"confidence={conf}\n"
                    )

    print("Saved unified post-processing outputs in:")
    print(f"  {SAVE_DIR}")
    print(f"Case tag: {meta.tag}")
    print("Key CSV files:")
    print(f"  - bl_piecewise_fits_{meta.tag}.csv")
    print(f"  - reverse_flow_{meta.tag}.csv")
    print(f"  - reverse_flow_all_heights_{meta.tag}.csv")
    print(f"  - case_metrics_{meta.tag}.csv")
    print(f"  - velocity_profiles_{meta.tag}.csv")


if __name__ == "__main__":
    main()
