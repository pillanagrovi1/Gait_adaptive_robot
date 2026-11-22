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


def test_online_skips_the_episode_cut_by_the_budget(tmp_path, monkeypatch):                # audit 7
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    task, learner = {"name": "cartpole"}, {"kind": "baseline_const", "action": 0}
    run = {"steps": 1003, "eval_every_steps": 1003, "eval_episodes": 1}
    exp = {"name": "e", "task": task, "run": run, "seeds": [0], "learners": [{"label": "c", "base": learner}]}
    (tmp_path / "e.json").write_text(json.dumps(exp))
    rows = harness.run(task, learner, 0, run, out_dir=tmp_path / "e" / "c")["rows"]
    assert rows[-1]["steps"] < max(r["steps"] for r in rows[:-1])          # the budget cut the last one short
    tail = rows[:-1][-max(1, (len(rows) - 1) // 5):]
    assert analysis.per_seed(tmp_path / "e.json", "c", "online").loc[0] == pytest.approx(
        sum(r["train_return"] for r in tail) / len(tail))


def test_memory_estimate_uses_measured_overhead(tmp_path, monkeypatch):                     # audit 3
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    task = {"name": "pendulum"}
    run = {"steps": 400, "eval_every_steps": 400, "eval_episodes": 1}
    exp = {"name": "e", "task": task, "run": run, "seeds": [0], "learners": [
        {"label": "t", "base": {"kind": "tabular"}, "sweep": {"bins": [5, 7]}}]}
    (tmp_path / "e.json").write_text(json.dumps(exp))
    metas = {b: harness.run(task, {"kind": "tabular", "bins": b}, 0, run, out_dir=tmp_path / "e" / f"t_bins{b}")["meta"]
             for b in (5, 7)}
    m4 = {"t_bins5": {"cfg_id": metas[5]["cfg_id"], "peak_bytes": "1008", "fits": "1", "agent_bytes": "368",
                      "stack_learner": "320", "host_match": "1"},
          "t_bins7": {"cfg_id": metas[7]["cfg_id"], "peak_bytes": "", "fits": "0", "error": "does not fit"}}
    monkeypatch.setattr(analysis, "load_m4", lambda name: m4)
    d = analysis.summarize(tmp_path / "e.json").set_index("run_id")
    assert d.loc["t_bins7", "peak_source"] == "host-estimate"
    assert d.loc["t_bins7", "peak_bytes"] == metas[7]["arena_bytes"] + 368 + 320
    exp["learners"].append({"label": "rnd", "base": {"kind": "baseline_random"}})
    (tmp_path / "e.json").write_text(json.dumps(exp))
    harness.run(task, {"kind": "baseline_random"}, 0, run, out_dir=tmp_path / "e" / "rnd")
    assert analysis.summarize(tmp_path / "e.json").set_index("run_id").loc["rnd", "peak_bytes"] == 0
