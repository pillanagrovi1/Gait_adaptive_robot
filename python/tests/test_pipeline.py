"""Fast checks of the Python side: binding, config expansion, harness."""
from pathlib import Path

import numpy as np
import pytest

from tinyrl import cbind, sweep

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not cbind.LIB_PATH.exists(), reason="run `make` first")


def test_cfg_struct_matches_c():
    # sizeof(tr_cfg_t) = 3 + 8 + 8 + 2 + 1 + 3 + 3 + 2 int32 fields
    assert cbind.C.sizeof(cbind.Cfg) == 30 * 4


@pytest.mark.parametrize("kind,extra", [
    ("tabular", {"bins": 11}),
    ("linear", {"features": 512}),
    ("mlp", {"hidden": 16, "replay_cap": 100, "batch": 4, "target_period": 50}),
])
def test_agent_roundtrip(kind, extra):
    lo, hi = np.array([-1.0, -2.0]), np.array([1.0, 2.0])
    ag = cbind.Agent({"kind": kind, **extra}, lo, hi, 3, seed=0)
    assert ag.mem_bytes > 0 and ag.params > 0
    s = np.array([0.3, -0.5])
    for _ in range(200):
        a = ag.act(s, 0.5)
        assert 0 <= a < 3
        ag.observe(s, a, 1.0, s, True)
    q = ag.qvalues(s)
    assert q.shape == (3,) and np.all(np.isfinite(q))


def test_memory_formulas():
    lo, hi = np.zeros(2), np.ones(2)
    t = cbind.Agent({"kind": "tabular", "bins": 10}, lo, hi, 3, 0)
    assert t.params == 10 * 10 * 3
    lin = cbind.Agent({"kind": "linear", "features": 256}, lo, hi, 3, 0)
    assert lin.params == 256 * 3
    n = cbind.Agent({"kind": "mlp", "hidden": 8}, lo, hi, 3, 0)
    assert n.params == 8 * 2 + 8 + 3 * 8 + 3


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        cbind.Agent({"kind": "tabular", "bins": 1000}, np.zeros(4), np.ones(4), 2, 0)


def test_every_config_expands():
    for f in sorted((ROOT / "configs").glob("*.json")):
        runs = sweep.expand(sweep.load(f))
        assert runs, f
        for r in runs:
            assert r["run_id"] and r["task"]["name"] in ("pendulum", "cartpole", "go1")


def test_harness_short_run(tmp_path):
    pytest.importorskip("mujoco")
    from tinyrl import harness
    out = harness.run({"name": "pendulum"}, {"kind": "tabular", "bins": 7, "gamma": 0.95}, seed=0,
                      run_cfg={"episodes": 4, "eval_every": 2, "eval_episodes": 1}, out_dir=tmp_path)
    assert (tmp_path / "seed0.csv").exists() and (tmp_path / "meta.json").exists()
    assert len(out["rows"]) == 4
    assert out["rows"][1]["eval_return"] != ""
