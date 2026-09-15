"""Checkpoint/restart helpers for the plume solver.

The checkpoint holds every prognostic field plus the physical time and the
adaptive-dt controller state. Older fields_latest.npz files that predate the
retry controller still load fine: the missing bits are recovered from the
CSV log, or from the case defaults if there's no log either.
"""

from pathlib import Path
import csv
import os
import numpy as np

from . import boundary


FIELD_NAMES = ("alpha", "u1", "v1", "u2", "v2", "p", "nut")


def resolve_restart_path(spec, outdir, project_root):
    """Resolve ``none``/``auto``/explicit checkpoint selection.

    ``auto`` resumes from ``output/fields_latest.npz`` when it exists and starts
    fresh otherwise. An explicit missing path is an error rather than a silent
    fresh start.
    """
    value = "" if spec is None else str(spec).strip()
    if value.lower() in {"", "none", "off", "false", "0", "fresh"}:
        return None
    if value.lower() == "auto":
        candidate = Path(outdir) / "fields_latest.npz"
        return candidate.resolve() if candidate.is_file() else None

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Restart checkpoint not found: {candidate}")
    return candidate


def _last_logged_state(log_path):
    path = Path(log_path)
    if not path.is_file():
        return 0, None
    last_it = 0
    last_dt = None
    try:
        with path.open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    it = int(float(row.get("it", 0)))
                except (TypeError, ValueError):
                    continue
                if it >= last_it:
                    last_it = it
                    try:
                        candidate_dt = float(row.get("dt", ""))
                        last_dt = candidate_dt if candidate_dt > 0.0 else last_dt
                    except (TypeError, ValueError):
                        pass
    except (OSError, csv.Error):
        return 0, None
    return last_it, last_dt


def load_checkpoint(path, fields, solver, grid, log_path=None):
    """Load and validate a checkpoint into existing fields/solver objects."""
    path = Path(path)
    logged_it, logged_dt = _last_logged_state(log_path) if log_path else (0, None)
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in FIELD_NAMES if name not in data]
        if "t" not in data:
            missing.append("t")
        if missing:
            raise ValueError(f"Checkpoint {path} is missing: {', '.join(missing)}")

        if "xc" in data and not np.allclose(data["xc"], grid.xc, rtol=1e-10, atol=1e-12):
            raise ValueError("Restart x mesh does not match the current case")
        if "yc" in data and not np.allclose(data["yc"], grid.yc, rtol=1e-10, atol=1e-12):
            raise ValueError("Restart y mesh does not match the current case")

        for name in FIELD_NAMES:
            values = np.asarray(data[name])
            if values.shape != grid.shape:
                raise ValueError(
                    f"Restart field {name!r} has shape {values.shape}; expected {grid.shape}")
            getattr(fields, name)[1:-1, 1:-1] = values

        solver.t = float(np.asarray(data["t"]).reshape(()))
        if "it" in data:
            solver.it = int(np.asarray(data["it"]).reshape(()))
        else:
            solver.it = logged_it
        if "dt" in data:
            solver.dt = float(np.asarray(data["dt"]).reshape(()))
        elif logged_dt is not None:
            solver.dt = logged_dt
        if "dt_headroom" in data:
            solver._dt_headroom = float(np.asarray(data["dt_headroom"]).reshape(()))
        else:
            # Old checkpoints did not store the retry controller. Restart from
            # the normal stability ceiling; the local limiter and rejection
            # controller still protect the first resumed step.
            solver._dt_headroom = float(solver.case.num.dt_max)
        if "rejected_steps_total" in data:
            solver.rejected_steps_total = int(
                np.asarray(data["rejected_steps_total"]).reshape(()))

    boundary.apply_bc(fields)
    return {
        "path": str(path), "t": float(solver.t), "it": int(solver.it),
        "dt": float(solver.dt), "dt_headroom": float(solver._dt_headroom),
    }


def checkpoint_payload(fields, solver, grid):
    """Return the serialisable state shared by snapshots and restart files."""
    return dict(
        t=float(solver.t), it=int(solver.it), dt=float(solver.dt),
        dt_headroom=float(solver._dt_headroom),
        rejected_steps_total=int(solver.rejected_steps_total),
        xc=grid.xc, yc=grid.yc, **fields.as_dict_interior())


def save_checkpoint(path, fields, solver, grid):
    """Atomically write a restartable NPZ checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **checkpoint_payload(fields, solver, grid))
    os.replace(tmp, path)


def next_event_time(t, interval):
    """First positive schedule multiple strictly later than ``t``."""
    if interval <= 0.0:
        return float("inf")
    scale = max(1.0, abs(t), abs(interval))
    tol = 32.0 * np.finfo(float).eps * scale
    return (np.floor((t + tol) / interval) + 1.0) * interval
