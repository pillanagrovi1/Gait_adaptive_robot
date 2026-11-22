"""The configs say what was pre-registered on 22 Nov 2026 (docs/decisions.md, D49-D53)."""
import json

import pytest

from tinyrl import analysis, harness, sweep

TUNED = {"table": ["alpha"], "linear": ["alpha"], "net": ["alpha", "target_period"],
         "net-stream": ["alpha", "obgd_kappa", "lam"]}


def test_cartpole_tunes_at_the_full_budget():
    exp = sweep.load("cartpole_main")
    assert {**exp["run"], **exp["tune"].get("run", {})}["steps"] == exp["run"]["steps"]      # D49


@pytest.mark.parametrize("name", ["replay_ratio", "adaptation_pendulum", "adaptation_pendulum_scratch"])
def test_hyperparameters_come_from_pendulum_tuning(name):
    src = {x["label"]: x["base"] for x in sweep.load("pendulum_main_tuned")["learners"]}
    got = {x["label"]: x["base"] for x in sweep.load(name)["learners"]}
    pairs = [(x, x) for x in TUNED] + ([("net-stream-cbp", "net-stream")] if "net-stream-cbp" in got else [])
    for label, origin in pairs:                                                              # D51
        for key in TUNED[origin]:
            assert got[label][key] == src[origin][key], (label, key)


@pytest.mark.parametrize("name,seeds", [("replay_ratio", 10), ("cartpole_main", 10), ("stream_ablation", 10),
                                        ("go1_gait_v3", 10), ("adaptation_pendulum", 20),
                                        ("adaptation_pendulum_scratch", 20)])
def test_seed_counts(name, seeds):
    assert sweep.load(name)["seeds"] == list(range(seeds))


def test_threshold_reports_how_many_seeds_reached_it(tmp_path, monkeypatch):                 # D53
    task, learner = {"name": "pendulum"}, {"kind": "baseline_const", "action": 1}
    run = {"steps": 600, "eval_every_steps": 200, "eval_episodes": 1}
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    for threshold, reached in ((-1e9, 2), (0, 0)):
        exp = {"name": "e", "task": task, "run": run, "seeds": [0, 1], "metrics": {"threshold": threshold},
               "learners": [{"label": "zero", "base": learner}]}
        (tmp_path / "e.json").write_text(json.dumps(exp))
        for seed in exp["seeds"]:
            harness.run(task, learner, seed, run, out_dir=tmp_path / "e" / "zero")
        assert analysis.summarize(tmp_path / "e.json").iloc[0]["x_to_threshold_reached"] == reached
