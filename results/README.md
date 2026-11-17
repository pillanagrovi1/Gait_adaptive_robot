# results/

Everything here except this file is generated and git-ignored:

- `raw/<experiment>/<run_id>/seed<k>.csv`, `meta.json` — written by `scripts/run_sweep.py`
- `m4/<experiment>.csv` — written by `scripts/measure_m4.py`
- `figures/<experiment>_*.png`, `_summary.csv` — written by `scripts/make_plots.py`

When a result goes into `docs/results.md`, record the git commit, the config
file and the exact command, so it can be regenerated. Commit a figure only
when you report it (add it with `git add -f`).
