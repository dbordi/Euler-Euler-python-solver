Post-processing for a single case.

Run it from the case root, after run.py has produced output/fields_*.npz:

    python postproc/postprocess_all_plume.py --current-a 25.6 --inflow-cms 0 --gap-cm 0.5

The three arguments are only used to label plots and file names; the script
reads whatever snapshots are in output/ regardless of what you pass. Run with
--help to see all options (boundary-layer fit window, figure formats, etc.).

Figures and CSV tables are written to output/postproc_all/.

This script needs a few extra packages beyond the solver itself: pandas,
matplotlib and pillow (see requirements.txt in this folder).

Note: this is the single-case script. It is not the same as the sweep
post-processing tools used to batch-process many cases at once -- those are
kept outside this repository.
