"""Tools added for the proposed *_v2 experiments (D70): conditional tuning, paired sweeps, own eval states."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tinyrl import harness, sweep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tune():
    spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts" / "tune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_alpha_is_tuned_only_where_obgd_is_off(tune):
    space = {"obgd_kappa": {"choice": [0, 2.0]}, "lam": {"choice": [0, 0.8]},
             "alpha": {"log_uniform": [0.05, 1.0], "when": {"obgd_kappa": 0}, "otherwise": 1.0}}
    rng = np.random.default_rng(0)
    draws = [tune.sample(space, rng) for _ in range(200)]
    on = [d["alpha"] for d in draws if d["obgd_kappa"] > 0]
    off = [d["alpha"] for d in draws if d["obgd_kappa"] == 0]
    assert on and off and all(a == 1.0 for a in on)
    assert all(0.05 <= a <= 1.0 for a in off) and len(set(off)) == len(off)


def test_old_spaces_draw_the_same_values(tune):
    """No "when" key: the generator is used exactly as before, so old tuning reproduces."""
    space = {"alpha": {"log_uniform": [0.05, 1.0]}, "lam": {"choice": [0.6, 0.8, 0.9]}}
    a = tune.sample(space, np.random.default_rng(12345))
    rng = np.random.default_rng(12345)
    assert a == {"alpha": float(np.exp(rng.uniform(np.log(0.05), np.log(1.0)))),
                 "lam": [0.6, 0.8, 0.9][int(rng.integers(3))]}


def test_sweep_zip_pairs_values():
    exp = {"name": "x", "task": {"name": "pendulum"}, "seeds": [0],
           "learners": [{"label": "linear", "base": {"kind": "linear"}, "sweep_zip": True,
                         "sweep": {"features": [64, 256], "tiles": [2, 5]}}]}
    runs = sweep.expand(exp)
    assert [(r["learner"]["features"], r["learner"]["tiles"]) for r in runs] == [(64, 2), (256, 5)]
    exp["learners"][0]["sweep"]["tiles"] = [2]
    with pytest.raises(ValueError):
        sweep.expand(exp)


def test_eval_base_seed_changes_the_evaluation_states():
    task, learner = {"name": "pendulum"}, {"kind": "tabular", "bins": 5}
    run = {"steps": 400, "decay_steps": 200, "eval_every_steps": 200, "eval_episodes": 2}
    a = harness.run(task, learner, 0, run)["evals"]
    b = harness.run(task, learner, 0, {**run, "eval_base_seed": 10_000})["evals"]
    c = harness.run(task, learner, 0, {**run, "eval_base_seed": 50_000})["evals"]
    assert a == b
    assert [e["eval_return"] for e in a] != [e["eval_return"] for e in c]


@pytest.mark.parametrize("name", ["pendulum_main_v2", "cartpole_main_v2"])
def test_v2_design_is_as_registered(name):
    """D70: every learner that can use traces tunes lam and kappa from the same grid, alpha only where ObGD is
    off, the Watkins cut on; same K; tuning at the full budget on its own evaluation states."""
    exp = sweep.load(name)
    t = exp["tune"]
    assert t["run"]["steps"] == exp["run"]["steps"]
    assert t["run"]["eval_base_seed"] >= exp["run"].get("eval_base_seed", 10_000) + exp["run"]["eval_episodes"]
    assert not set(t["seeds"]) & set(exp["seeds"]) and exp["seeds"] == list(range(10))
    learners = {L["label"]: L for L in exp["learners"] if not L["base"]["kind"].startswith("baseline")}
    assert set(t["space"]) == set(learners)
    for label, L in learners.items():
        sp = t["space"][label]
        family = label.split("-angle")[0].split("-trig")[0]
        if family in ("table", "linear", "net-stream"):
            assert sp["lam"]["choice"] == [0, 0.6, 0.8, 0.9] and sp["obgd_kappa"]["choice"] == [0, 1.5, 2.0, 3.0]
            assert sp["alpha"]["when"] == {"obgd_kappa": 0} and sp["alpha"]["otherwise"] == 1.0
            assert L["base"]["watkins"] == 1
        elif family == "net-sgd":
            assert set(sp) == {"alpha"} and not {"lam", "obgd_kappa", "replay_cap", "target_period"} & set(L["base"])
        if family == "linear":
            assert L["sweep_zip"] and len(L["sweep"]["tiles"]) == len(L["sweep"]["features"])


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matched_pick_uses_only_fitting_runs_and_selection_auc():
    """D70 comparison 2: per family, the highest selection AUC among the runs that fit; tables never count."""
    pick = _script("compare").pick
    sram = {"net_hidden16": 3000, "net-stream_hidden64": 9000, "linear_features256-tiles1": 2000,
            "linear_features4096-tiles3": 20000, "table_bins12": 1000}
    auc = {"net_hidden16": 100, "net-stream_hidden64": 400, "linear_features256-tiles1": 50,
           "linear_features4096-tiles3": 500, "table_bins12": 999}
    assert pick(6 * 1024, sram, auc) == {"network": "net_hidden16", "linear": "linear_features256-tiles1"}
    assert pick(16 * 1024, sram, auc) == {"network": "net-stream_hidden64", "linear": "linear_features256-tiles1"}
    assert pick(64 * 1024, sram, auc)["linear"] == "linear_features4096-tiles3"
    assert pick(1024, sram, auc) == {}


def test_hash_load_matches_d70_table():
    load = _script("results_table").hash_load
    assert round(load({"tilings": 8, "tiles": 2, "features": 64}, 2), 2) == 1.12
    assert round(load({"tilings": 8, "tiles": 7, "features": 4096}, 3), 2) == 1.00
    assert round(load({"tilings": 16, "tiles": 2, "features": 1024}, 4), 2) == 1.27


def test_selection_config_uses_tuning_seeds_and_states_only(tune):
    exp = sweep.load("cartpole_main_v2")
    t = exp.pop("tune")
    sel = tune.select_config({**exp, "name": "cartpole_main_v2_tuned"}, t)
    assert sel["name"] == "cartpole_main_v2_tuned_select" and sel["seeds"] == t["seeds"]
    assert not set(sel["seeds"]) & set(exp["seeds"]) and sel["run"]["eval_base_seed"] == 50_000
    assert {s["run_id"] for s in sweep.expand(sel)} == {s["run_id"] for s in sweep.expand(exp)}
