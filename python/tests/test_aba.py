"""D57: A -> B -> A shift, and the paired differences built on it."""
import json

import pytest

from tinyrl import analysis, harness

TASK = {"name": "pendulum", "obs_mode": "trig"}
RUN = {"steps": 2400, "eval_every_steps": 200, "eval_episodes": 2}
B = {"shift_at_step": 800, "shift_level": 0.5}
ABA = {**RUN, **B, "shift_back_at_step": 1600}
PUSH = {"kind": "baseline_const", "action": 0}      # constant torque, so the rod's mass matters


def test_harness_switches_back():
    a = harness.run(TASK, PUSH, 0, RUN)
    ab = harness.run(TASK, PUSH, 0, {**RUN, **B})
    aba = harness.run(TASK, PUSH, 0, ABA)
    assert aba["evals"][:8] == ab["evals"][:8]       # up to and including step 1600: A, then B
    assert aba["evals"][8:] == a["evals"][8:]        # afterwards: A again
    assert aba["meta"]["eval_after_shift_back"] == a["evals"][7]["eval_return"]
    assert aba["rows"][8:] == a["rows"][8:]          # the training environment switched back too


@pytest.mark.parametrize("bad", [
    {"steps": 400, "shift_back_at_step": 200},                          # nothing to switch back from
    {"steps": 400, "shift_at_step": 200, "shift_back_at_step": 100},    # before the shift
    {"steps": 400, "shift_at_step": 200, "shift_back_at_step": 400},    # no second A window
    {"episodes": 2, "shift_at": 1, "shift_back_at_step": 100},          # step budgets only
])
def test_harness_rejects_bad_switch_back(bad):
    with pytest.raises(ValueError):
        harness.run(TASK, PUSH, 0, bad)


def test_windows_and_paired_differences(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    exps = {}
    for name, run in (("aba", ABA), ("scratch", {**RUN, "steps": 800, "shift_at_step": 0, "shift_level": 0.5})):
        exp = {"name": name, "task": TASK, "run": run, "seeds": [0, 1], "learners": [{"label": "push", "base": PUSH}]}
        exps[name] = tmp_path / f"{name}.json"
        exps[name].write_text(json.dumps(exp))
        for seed in exp["seeds"]:
            out = harness.run(TASK, PUSH, seed, run, out_dir=tmp_path / name / "push")
            if name == "aba" and seed == 0:
                first = out
    ev = [e["eval_return"] for e in first["evals"]]
    get = lambda m: analysis.per_seed(exps["aba"], "push", m).loc[0]  # noqa: E731
    assert get("final") == pytest.approx(sum(ev[1:4]) / 3)
    assert get("post_auc") == pytest.approx(sum(ev[4:8]) / 4)         # B window only
    assert get("a2_auc") == pytest.approx(sum(ev[8:12]) / 4)
    assert get("back") == pytest.approx(first["meta"]["eval_after_shift_back"])
    assert get("online") == pytest.approx(first["rows"][3]["train_return"])  # pre-shift episodes only

    forgot = analysis.paired_diff(exps["aba"], "push", "push", "back", metric_b="final")
    assert forgot["n"] == 2 and forgot["mean_diff"] == pytest.approx(0)      # a fixed policy forgets nothing
    ft = analysis.paired_diff(exps["aba"], "push", "push", "post_auc", exp_b=exps["scratch"], metric_b="auc")
    assert ft["n"] == 2 and ft["mean_diff"] == pytest.approx(0)              # and transfers nothing
