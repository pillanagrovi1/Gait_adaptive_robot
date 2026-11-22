"""What the result tables are fed by (second review, findings 12-15, 20)."""
import json
import math
import sys
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tinyrl import analysis, harness, plots, sweep

ROOT = Path(__file__).resolve().parents[2]
TASK, RUN = {"name": "pendulum"}, {"steps": 600, "eval_every_steps": 200, "eval_episodes": 1}
PUSH = {"kind": "baseline_const", "action": 0}


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("metric,better", [("auc", 2 / 3), ("x_to_threshold", 1 / 3), ("recovery", 1 / 3)])
def test_paired_diff_counts_only_complete_pairs(monkeypatch, metric, better):            # 12
    runs = {"a": pd.Series([3.0, 3.0, 1.0, math.nan]), "b": pd.Series([2.0, 2.0, 2.0, 2.0])}
    monkeypatch.setattr(analysis, "per_seed", lambda exp, run, m: runs[run])
    r = analysis.paired_diff("any", "a", "b", metric)
    assert (r["n"], r["n_missing"]) == (3, 1)
    assert r["a_better_frac"] == pytest.approx(better)


def test_unfinished_seed_is_not_analysed(tmp_path, monkeypatch):                         # 15
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    exp = {"name": "e", "task": TASK, "run": RUN, "seeds": [0, 1, 2], "learners": [{"label": "push", "base": PUSH}]}
    (tmp_path / "e.json").write_text(json.dumps(exp))
    rows = {}
    for s in exp["seeds"]:
        rows[s] = harness.run(TASK, PUSH, s, RUN, out_dir=tmp_path / "e" / "push")["rows"]
    (tmp_path / "e/push/seed1_meta.json").unlink()
    (tmp_path / "e/push/seed1_eval.csv").unlink()
    v = analysis.per_seed(tmp_path / "e.json", "push", "online")
    assert list(v.index) == [0, 2]
    assert v[2] == pytest.approx(rows[2][-1]["train_return"])
    assert analysis.summarize(tmp_path / "e.json").iloc[0]["n_seeds"] == 2


def test_resume_refuses_another_config(tmp_path, monkeypatch):                           # 14
    rs = script("run_sweep")
    spec = sweep.expand(sweep.load("smoke"))[0]
    d = tmp_path / "smoke" / spec["run_id"]
    d.mkdir(parents=True)
    old = {"task": spec["task"], "run": spec["run"], "learner": {**spec["learner"], "alpha": 0.987}}
    (d / "seed0_meta.json").write_text(json.dumps(old))
    monkeypatch.setattr(rs.harness, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", "smoke", "--dry-run"])
    with pytest.raises(SystemExit) as e:
        rs.main()
    assert "another config" in str(e.value.code)


@pytest.mark.parametrize("row,fails", [
    ({"host_match": 1}, False),
    ({"host_match": 0}, True),
    ({"error": "does not fit in SRAM (link failed)"}, False),
    ({"error": "build failed: x"}, True),
])
def test_m4_failures_stop_the_pipeline(row, fails):                                      # 13
    assert bool(script("measure_m4").broken([{"run_id": "r", **row}])) == fails


def test_no_empty_panels(tmp_path, monkeypatch):                                         # 20
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    harness.run(TASK, {"kind": "tabular", "bins": 5}, 0, RUN, out_dir=tmp_path / "e" / "ran")
    exp = {"name": "e", "task": TASK, "run": RUN, "seeds": [0], "learners": [
        {"label": "ran", "base": {"kind": "tabular", "bins": 5}},
        {"label": "never", "base": {"kind": "tabular", "bins": 7}}]}
    seen = []
    monkeypatch.setattr(plots.plt, "close", lambda fig: seen.extend(a.get_title(loc="left") for a in fig.axes))
    plots.learning_curves(exp, sweep.expand(exp), tmp_path / "c.png")
    assert seen == ["ran"]
