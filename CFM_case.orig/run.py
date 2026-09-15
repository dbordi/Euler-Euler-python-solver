"""
run.py -- Euler-Euler bubble-plume solver driver.

Run from the project root:
    python run.py
"""

import os
import time
import json
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

from config.case import CASE
from src import mesh as meshmod
from src.fields import Fields
from src.solver import Solver
from src import diagnostics as diag
from src import interfacial
from src import restart as restart_io
import src.properties_modes as properties_modes

# ---- case parameters for this run -----------------------------------------
RESOLUTION   = (20, 1000)     # (nx, ny)
H_DOMAIN     = 0.5            # m, channel height
T_END        = 20.0           # s
SOURCE_J     = 2000           # A/m^2

CFL_TARGET       = 0.20       # local face-based Courant target
DIFFUSIVE_NUMBER = 0.80       # fraction of the local explicit FV bound
DT_MAX           = 1.0e-3     # s

# "auto" resumes output/fields_latest.npz if present, else starts fresh.
RESTART_FROM = "auto"
CLEAN_OUTPUT_ON_FRESH_START = True

TURB_MODEL        = "mixing_length"   # "laminar" | "mixing_length"
BIT_SATO_ON       = True
ALPHA_DISPERSION  = True       # turbulent dispersion D_t=nu_t/sigma_d (zero if laminar)
ALPHA_DISPERSION_HYDRO = True  # bubble-induced dispersion (nonzero even if laminar)
CONVECTION        = "limited"  # "upwind" | "limited" (use "limited" to see dispersion)

# Near-electrode mesh grading. X_FIRST_CELLS are exact widths at the wall(s);
# the rest of the gap grows geometrically to fill the domain.
X_GRADING_MODE = "both"    # "electrode" (near x=0 only) | "both" (mirrored at x=L)
X_FIRST_CELLS  = (70e-6, 70e-6, 100e-6)

SOLID_PRESSURE_ACCEL_CAP = 1500.0    # m/s^2; <=0 disables the cap

N_OUTER      = 1              # 1 = PISO; >1 = PIMPLE outer Picard iterations
N_CORR       = 1              # pressure correctors per outer iteration
RELAX_U      = 1
RELAX_P      = 1
PRINT_EVERY  = 10             # console + CSV logging interval, in iterations
REUSE_PRESSURE_LU = True

# Bottom inlet: 0 or negative -> closed wall.
BOTTOM_INFLOW_U = 0.00        # m/s, upward liquid inlet velocity
BOTTOM_ALPHA_G  = 0.0         # gas fraction at the bottom inlet, usually 0
# -----------------------------------------------------------------------------


def apply_overrides():
    CASE.geom.H = H_DOMAIN
    CASE.mesh.nx, CASE.mesh.ny = RESOLUTION
    CASE.num.t_end = T_END
    CASE.source.j = SOURCE_J
    CASE.num.cfl = float(CFL_TARGET)
    CASE.num.diffusive_number = float(DIFFUSIVE_NUMBER)
    CASE.num.dt_max = float(DT_MAX)

    CASE.turb.model = TURB_MODEL
    CASE.turb.bit_sato = bool(BIT_SATO_ON)
    CASE.num.dispersion = bool(ALPHA_DISPERSION)
    CASE.num.dispersion_hydro = bool(ALPHA_DISPERSION_HYDRO)
    CASE.num.convection = CONVECTION

    CASE.mesh.x_grading_mode = str(X_GRADING_MODE).strip().lower()
    CASE.mesh.x_first_cells = tuple(X_FIRST_CELLS)

    CASE.num.n_outer = N_OUTER
    CASE.num.n_corr = N_CORR
    CASE.num.relax_U = RELAX_U
    CASE.num.relax_p = RELAX_P
    CASE.num.print_every = PRINT_EVERY
    CASE.num.reuse_pressure_lu = bool(REUSE_PRESSURE_LU)

    CASE.inlet.U_in = max(0.0, float(BOTTOM_INFLOW_U))
    CASE.inlet.alpha_g_in = float(np.clip(BOTTOM_ALPHA_G, 0.0, CASE.num.alpha_max))

    CASE.solid_pressure.accel_cap = float(SOLID_PRESSURE_ACCEL_CAP)


def apply_initial_inflow(f):
    """Initialize the domain with a through-flow if the bottom inlet is active."""
    if CASE.inlet.U_in > 0.0:
        f.v1[1:-1, 1:-1] = CASE.inlet.U_in
        f.v2[1:-1, 1:-1] = CASE.inlet.U_in


def requested_log_columns():
    return [
        "it", "t", "dt",
        "dt_limiter_mode", "dt_limiter", "dt_stability",
        "dt_cfl", "dt_cfl_liquid", "dt_cfl_gas",
        "dt_diff_liquid", "dt_diff_gas", "dt_diff_alpha",
        "Co_max", "Fo_max",
        "dt_limiter_i", "dt_limiter_j", "dt_limiter_x_m", "dt_limiter_y_m",
        "dt_retries", "dt_rejections_total",
        "alpha_min", "alpha_max", "alpha_mean", "alpha_sum",
        "u_liq_max", "u_gas_max", "v_liq_max", "v_gas_max",
        "u1_rms", "v1_rms", "u2_rms", "v2_rms",
        "slip_max", "slip_mean",
        "p_min", "p_max", "p_rms", "nut_max",
        "p_coll_max", "M_coll_max", "a_coll_max",
        "cont_rms", "cont_max",
    ]


def prepare_log(path, append_existing):
    """Create a new log, or extend an old one's schema without losing prior rows."""
    requested = requested_log_columns()
    path = Path(path)
    if not append_existing or not path.is_file() or path.stat().st_size == 0:
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerow(requested)
        return requested

    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        old_columns = next(reader, [])
    columns = list(dict.fromkeys(old_columns + requested))
    if columns != old_columns:
        tmp = path.with_name(path.name + ".tmp")
        with path.open("r", newline="") as src, tmp.open("w", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=columns)
            writer.writeheader()
            for row in reader:
                writer.writerow(row)
        os.replace(tmp, path)
    return columns


def append_log(path, columns, d):
    row = [d.get(col, "") for col in columns]
    with open(path, "a", newline="") as fh:
        csv.writer(fh).writerow(row)


def clean_output(outdir):
    """Remove only solver-generated files on an explicitly requested fresh run."""
    suffixes = (".npz", ".png", ".gif", ".csv", ".txt", ".json", ".log")
    for path in Path(outdir).iterdir():
        if path.is_file() and path.name.endswith(suffixes):
            path.unlink()


def empty_timeseries():
    return dict(t=[], xcg=[], uprobe=[], amax=[], amean=[])


def load_existing_timeseries(path, restart_t):
    ts = empty_timeseries()
    path = Path(path)
    if not path.is_file():
        return ts
    try:
        with np.load(path, allow_pickle=False) as old:
            t = np.asarray(old["t"], dtype=float)
            keep = t <= restart_t + 1e-12
            ts["t"] = t[keep].tolist()
            for key in ("xcg", "uprobe", "amax", "amean"):
                ts[key] = np.asarray(old[key])[keep].tolist()
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Warning: could not reuse {path.name}: {exc}")
        return empty_timeseries()
    return ts


def save_timeseries(path, ts, grid):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp,
             t=np.asarray(ts["t"]), xcg=np.asarray(ts["xcg"]),
             uprobe=np.asarray(ts["uprobe"]), amax=np.asarray(ts["amax"]),
             amean=np.asarray(ts["amean"]),
             probe_heights=np.asarray(CASE.post.probe_heights),
             H=CASE.geom.H, L=CASE.geom.L, yc=grid.yc, xc=grid.xc)
    os.replace(tmp, path)


def load_existing_alpha_snaps(path, restart_t, grid):
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with np.load(path, allow_pickle=False) as old:
            times = np.asarray(old["t"], dtype=float)
            alpha = np.asarray(old["alpha"])
            if alpha.shape[1:] != grid.shape:
                raise ValueError("snapshot mesh does not match current grid")
            keep = np.flatnonzero(times <= restart_t + 1e-12)
            return [(float(times[k]), alpha[k].copy()) for k in keep]
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Warning: could not reuse {path.name}: {exc}")
        return []


def save_alpha_snaps(path, snaps, grid):
    if not snaps:
        return
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp,
             t=np.asarray([s[0] for s in snaps]),
             alpha=np.stack([s[1] for s in snaps], axis=0),
             xc=grid.xc, yc=grid.yc)
    os.replace(tmp, path)


def print_diagnostics(d):
    print(
        f"it={d['it']:7d} "
        f"t={d['t']:8.4f}s "
        f"dt={d['dt']:.2e} "
        f"lim={d['dt_limiter']} "
        f"Co={d['Co_max']:.3f} Fo={d['Fo_max']:.3f} "
        f"alpha=[{d['alpha_min']:.2e},{d['alpha_max']:.3f}] "
        f"amean={d['alpha_mean']:.2e} "
        f"|Ul|max={d['u_liq_max']*1e3:8.2f} mm/s "
        f"|Ug|max={d['u_gas_max']*1e3:8.2f} mm/s "
        f"slip={d['slip_max']*1e3:8.2f} mm/s "
        f"p=[{d['p_min']:.2e},{d['p_max']:.2e}] "
        f"Mcoll={d['M_coll_max']:.2e} "
        f"retry={d['dt_retries']} "
        f"cont_rms={d['cont_rms']:.2e}"
    )


def main():
    apply_overrides()

    outdir = ROOT / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    restart_path = restart_io.resolve_restart_path(RESTART_FROM, outdir, ROOT)
    restarting = restart_path is not None
    if not restarting and CLEAN_OUTPUT_ON_FRESH_START:
        clean_output(outdir)
    print(f"Writing output to: {outdir}")
    print(f"Run mode: {'RESTART' if restarting else 'FRESH START'}")

    grid = meshmod.build(CASE)
    f = Fields(grid)
    apply_initial_inflow(f)
    solver = Solver(CASE, grid, f)
    log_path = outdir / "solver_log.csv"

    restart_info = None
    if restarting:
        restart_info = restart_io.load_checkpoint(
            restart_path, f, solver, grid, log_path=log_path)
        print(f"Restarted from {restart_path}")
        print(f"  t={solver.t:.9g} s, it={solver.it}, previous dt={solver.dt:.3e} s")
    grid.save(outdir / "mesh.npz")

    case_info = {
        "resolution": list((CASE.mesh.nx, CASE.mesh.ny)),
        "H_domain_m": CASE.geom.H,
        "L_gap_m": CASE.geom.L,
        "t_end_s": CASE.num.t_end,
        "source_j_A_m2": CASE.source.j,
        "turb_model": CASE.turb.model,
        "bit_sato": CASE.turb.bit_sato,
        "alpha_dispersion": CASE.num.dispersion,
        "convection": CASE.num.convection,
        "cfl_target": CASE.num.cfl,
        "diffusive_number": CASE.num.diffusive_number,
        "dt_max_s": CASE.num.dt_max,
        "drag_model": CASE.num.drag_model,
        "drag_alpha_floor": CASE.num.drag_alpha_floor,
        "bubble_diameter_m": CASE.fluids.d_b,
        "surface_tension_N_m": CASE.fluids.sigma,
        "Eotvos": interfacial.eotvos(CASE),
        "terminal_slip_m_s": interfacial.terminal_slip(CASE),
        "n_outer": CASE.num.n_outer,
        "n_corr": CASE.num.n_corr,
        "relax_U": CASE.num.relax_U,
        "relax_p": CASE.num.relax_p,
        "reuse_pressure_lu": CASE.num.reuse_pressure_lu,
        "bottom_inflow_U_m_s": CASE.inlet.U_in,
        "bottom_alpha_g": CASE.inlet.alpha_g_in,
        "bottom_mode": "inlet" if CASE.inlet.U_in > 0.0 else "wall",
        "solid_pressure": CASE.solid_pressure.__dict__,
        "restart": restart_info,
    }
    with open(outdir / "case_summary.json", "w") as fh:
        json.dump(case_info, fh, indent=2)

    print("=" * 80)
    print("Euler-Euler plume solver")
    print(f"  mesh {grid.nx}x{grid.ny}={grid.nx*grid.ny}, H={CASE.geom.H} m, "
          f"j={CASE.source.j} A/m^2")
    print(f"  bottom: {'inlet' if CASE.inlet.U_in > 0 else 'wall'}  "
          f"U_in={CASE.inlet.U_in:.3e} m/s  alpha_g_in={CASE.inlet.alpha_g_in:.3e}")
    print(f"  turb={CASE.turb.model}  SatoBIT={CASE.turb.bit_sato}  "
          f"alpha_dispersion={CASE.num.dispersion}  conv={CASE.num.convection}")
    print(f"  CFL={CASE.num.cfl:g}  Fo={CASE.num.diffusive_number:g}  "
          f"dt_max={CASE.num.dt_max:.3e} s")
    print(f"  PIMPLE outer={CASE.num.n_outer} corr={CASE.num.n_corr} "
          f"relax_U={CASE.num.relax_U} relax_p={CASE.num.relax_p}")
    print(f"  drag={CASE.num.drag_model}  Eo={interfacial.eotvos(CASE):.3e} "
          f"Ut={interfacial.terminal_slip(CASE)*1e3:.3f} mm/s")
    interfacial.solid_pressure_info(CASE)
    print("=" * 80)

    log_columns = prepare_log(log_path, append_existing=restarting)

    h_rows = [int(np.clip(hf, 0, 1) * (grid.ny - 1)) for hf in CASE.post.probe_heights]
    if restarting:
        ts = load_existing_timeseries(outdir / "timeseries.npz", solver.t)
        snaps = load_existing_alpha_snaps(outdir / "alpha_snaps.npz", solver.t, grid)
        next_probe = restart_io.next_event_time(solver.t, CASE.post.probe_dt)
        next_snap = restart_io.next_event_time(solver.t, CASE.post.snap_dt)
    else:
        ts = empty_timeseries()
        snaps = []
        next_snap = next_probe = 0.0

    start_t, start_it = solver.t, solver.it
    last_logged_it = -1
    t0 = time.time()

    while solver.t < CASE.num.t_end - 1e-14:
        remaining = CASE.num.t_end - solver.t
        d = solver.step(dt_cap=remaining)
        if not d["finite"]:
            print("!! non-finite fields, stopping.")
            break

        if solver.t + 1e-14 >= next_probe:
            a = f.alpha[1:-1, 1:-1]
            xcg = diag.plume_centroid(a, grid.xc)
            ts["t"].append(solver.t)
            ts["xcg"].append([xcg[r] for r in h_rows])
            ts["uprobe"].append(float(f.u1[1 + h_rows[len(h_rows)//2], 1]))
            ts["amax"].append(d["alpha_max"])
            ts["amean"].append(d["alpha_mean"])
            next_probe = restart_io.next_event_time(solver.t, CASE.post.probe_dt)

        if solver.t + 1e-14 >= next_snap:
            snaps.append((solver.t, f.alpha[1:-1, 1:-1].copy()))
            field_path = outdir / f"fields_{solver.t:.3f}.npz"
            restart_io.save_checkpoint(field_path, f, solver, grid)
            restart_io.save_checkpoint(outdir / "fields_latest.npz", f, solver, grid)
            save_timeseries(outdir / "timeseries.npz", ts, grid)
            next_snap = restart_io.next_event_time(solver.t, CASE.post.snap_dt)

        if solver.it % CASE.num.print_every == 0:
            print_diagnostics(d)
            append_log(log_path, log_columns, d)
            last_logged_it = solver.it

    # Always leave an exact final/latest restart, also when t_end is not a dump time.
    restart_io.save_checkpoint(outdir / "fields_latest.npz", f, solver, grid)
    if solver.it > start_it and last_logged_it != solver.it:
        print_diagnostics(d)
        append_log(log_path, log_columns, d)

    print(f"done in {time.time()-t0:.1f}s, {solver.it-start_it} new steps "
          f"({solver.it} total), advanced {solver.t-start_t:.6g} s. "
          f"Re_plume(final)~{properties_modes.plume_reynolds(f, CASE, grid):.0f}")

    save_timeseries(outdir / "timeseries.npz", ts, grid)
    save_alpha_snaps(outdir / "alpha_snaps.npz", snaps, grid)
    print(f"{outdir} written. Log: {log_path}")
    print("Post-process with:  python postproc/postprocess_all_plume.py")


if __name__ == "__main__":
    main()
