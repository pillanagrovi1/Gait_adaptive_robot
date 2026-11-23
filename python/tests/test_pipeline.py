"""Fast checks of the Python side: binding, config expansion, harness."""
from pathlib import Path

import numpy as np
import pytest

from tinyrl import cbind, sweep

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not cbind.LIB_PATH.exists(), reason="run `make` first")


def test_cfg_struct_matches_c():
    # sizeof(tr_cfg_t) = 3 + 12 + 12 + 2 + 1 + 3 + 3 + 2 + 5 + 4 + 1 (watkins) int32 fields
    assert cbind.C.sizeof(cbind.Cfg) == 48 * 4


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


@pytest.mark.parametrize("kind,extra", [
    ("tabular", {"bins": 11}),
    ("linear", {"features": 512}),
    ("mlp", {"hidden": 16, "norm": 1, "round_mode": 1}),
])
def test_stream_agents_learn_one_step(kind, extra):
    """Stream Q(lambda) path (ObGD + traces) fits Q(s,a) = +-0.5*s0 on terminal transitions."""
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    ag = cbind.Agent({"kind": kind, "alpha": 1.0 if kind != "mlp" else 0.5, "obgd_kappa": 2.0,
                      "lam": 0.8, **extra}, lo, hi, 2, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20000):
        s = rng.uniform(-1, 1, 2)
        a = int(rng.integers(2))
        ag.episode_start()
        ag.observe(s, a, 0.5 * s[0] * (1 if a else -1), s, True)
    err = np.mean([abs(ag.qvalues(np.array([x, 0.2]))[1] - 0.5 * x) for x in np.linspace(-0.9, 0.9, 7)])
    assert err < 0.1, err
    lost, total = ag.underflow()
    assert total > 0 and 0 <= lost <= total


def test_unknown_learner_key_rejected():
    with pytest.raises(ValueError):
        cbind.Agent({"kind": "tabular", "lambda": 0.8}, np.zeros(2), np.ones(2), 2, 0)


def test_harness_step_budget(tmp_path):
    pytest.importorskip("mujoco")
    from tinyrl import harness
    out = harness.run({"name": "pendulum", "obs_mode": "trig"},
                      {"kind": "mlp", "hidden": 8, "alpha": 1.0, "obgd_kappa": 2.0, "lam": 0.8, "gamma": 0.95},
                      seed=0, run_cfg={"steps": 1000, "eval_every_steps": 250, "eval_episodes": 1},
                      out_dir=tmp_path)
    assert [e["step"] for e in out["evals"]] == [250, 500, 750, 1000]
    assert out["meta"]["env_steps"] == 1000 and out["meta"]["budget"] == "steps"
    assert (tmp_path / "seed0_eval.csv").exists() and (tmp_path / "seed0_meta.json").exists()
    assert out["meta"]["update_terms"] > 0


@pytest.mark.parametrize("kw,n_act,obs_dim", [
    ({}, 6, 4),
    ({"obs_mode": "imu4pc", "phase_lock": True, "action_mode": "factored"}, 18, 7),
    ({"obs_mode": "imu4pc", "phase_lock": True, "action_mode": "incremental", "stumble": True}, 7, 10),
    ({"obs_mode": "imu4p", "phase_lock": True, "tegotae_sigma": 1.0, "posture_gain": 0.5}, 6, 5),
])
def test_go1_modes(kw, n_act, obs_dim):
    pytest.importorskip("mujoco")
    from tinyrl.envs.go1 import Go1Gait
    env = Go1Gait(roughness=0.05, **kw)
    assert env.n_actions == n_act and env.obs_dim == obs_dim
    o = env.reset(seed=0)
    for a in range(3):
        o, r, term, trunc = env.step(a % n_act)
        assert o.shape == (obs_dim,) and np.all(np.isfinite(o))
    assert np.all(o >= env.obs_lo - 1.0) and np.all(o <= env.obs_hi + 1.0)


def test_pil_roundtrip_qemu():
    """Learner on the emulated M4 (processor-in-the-loop) == learner on the host, bit for bit."""
    import shutil
    import socket
    import subprocess
    import time
    from pathlib import Path
    if not shutil.which("qemu-system-arm") or not shutil.which("arm-none-eabi-gcc"):
        pytest.skip("QEMU / ARM toolchain not installed")
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["make", "-s", "-C", str(root / "m4" / "pil"), "qemu"], check=True, capture_output=True)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    q = subprocess.Popen(["qemu-system-arm", "-M", "netduinoplus2", "-nographic", "-monitor", "none",
                          "-serial", f"tcp:127.0.0.1:{port},server=on,wait=off,nodelay=on",
                          "-kernel", str(root / "build" / "pil" / "qemu" / "pil.elf")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.3)
        from tinyrl.pil import PilAgent
        L = {"kind": "mlp", "hidden": 8, "alpha": 1.0, "obgd_kappa": 2.0, "lam": 0.8, "norm": 1, "round_mode": 1}
        lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        chip = PilAgent({**L, "pil": f"tcp:127.0.0.1:{port}"}, lo, hi, 3, seed=5)
        host = cbind.Agent(L, lo, hi, 3, seed=5)
        rng = np.random.default_rng(1)
        for k in range(300):
            s1, s2 = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
            if k % 50 == 0:
                chip.episode_start()
                host.episode_start()
            a1, a2 = chip.act(s1, 0.2), host.act(s1, 0.2)
            assert a1 == a2
            assert chip.observe(s1, a1, 0.1, s2, k % 50 == 49) == host.observe(s1, a2, 0.1, s2, k % 50 == 49)
        assert chip.greedy(np.array([0.3, -0.2])) == host.greedy(np.array([0.3, -0.2]))
        chip.close()
    finally:
        q.kill()
