"""Tools for the proposed Go1 v4 (D71): fixed settings, the headroom grid, a held-out final evaluation."""
import importlib.util
from pathlib import Path

import pytest

from tinyrl import analysis, harness
from tinyrl.envs import make_env

ROOT = Path(__file__).resolve().parents[2]
GO1 = {"name": "go1", "obs_mode": "imu4pc", "phase_lock": True, "roughness": 0.1,
       "action_mode": "incremental", "stumble": True}


@pytest.fixture
def headroom():
    spec = importlib.util.spec_from_file_location("go1_headroom", ROOT / "scripts" / "go1_headroom.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_is_the_180_reachable_settings_and_contains_b3(headroom):
    g = headroom.grid()
    assert len(g) == 6 * 6 * 5 == len({tuple(s.values()) for s in g})
    assert headroom.DEFAULT in g and headroom.DEFAULT == {"clearance": 1.0, "posture_gain": 0.0, "height": 0.0}


def test_init_params_hold_a_setting_every_episode():
    env = make_env({**GO1, "init_params": {"clearance": 1.4, "height": -0.05}})
    for seed in (1, 2):
        env.reset(seed=seed)
        assert env.params["clearance"] == 1.4 and env.params["height"] == -0.05
        env.step(0)
        assert env.params["clearance"] == 1.4
    with pytest.raises(ValueError):
        make_env({**GO1, "init_params": {"clearance": 2.0}})


def test_default_env_is_unchanged():
    a, b = make_env(GO1), make_env({**GO1, "init_params": {}})
    a.reset(seed=3), b.reset(seed=3)
    assert a.params == b.params == {"clearance": 1.0, "posture_gain": 0.0, "height": 0.0}


def test_heldout_evaluation_is_recorded_and_read(tmp_path, monkeypatch):
    task, learner = {"name": "pendulum"}, {"kind": "baseline_const", "action": 1}
    run = {"steps": 400, "eval_every_steps": 200, "eval_episodes": 1,
           "heldout_episodes": 3, "heldout_base_seed": 20000}
    res = harness.run(task, learner, 0, run, out_dir=tmp_path / "e" / "zero")
    assert isinstance(res["meta"]["eval_heldout"], float)
    plain = harness.run(task, learner, 0, {k: v for k, v in run.items() if not k.startswith("heldout")})
    assert "eval_heldout" not in plain["meta"]
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    import json
    exp = {"name": "e", "task": task, "run": run, "seeds": [0], "learners": [{"label": "zero", "base": learner}]}
    (tmp_path / "e.json").write_text(json.dumps(exp))
    assert analysis.per_seed(tmp_path / "e.json", "zero", "heldout").iloc[0] == res["meta"]["eval_heldout"]
