# Raw per-seed logs of reported experiments

Every number in `docs/results.md` can be recomputed from these archives without rerunning anything:

```bash
tar -xzf results/raw_archive/pendulum_main_tuned.tar.gz -C results/raw
python3 scripts/make_plots.py pendulum_main_tuned        # figures + summary CSV
python3 scripts/results_table.py pendulum_main_tuned     # the table in docs/results.md
```

Each run folder holds `seed<k>.csv` (one row per training episode), `seed<k>_eval.csv` (greedy evaluations at
step checkpoints) and `seed<k>_meta.json` (config, memory, underflow, git commit).
