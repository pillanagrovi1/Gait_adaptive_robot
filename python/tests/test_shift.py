"""D58: the evaluation at the shift step still belongs to the old task."""
from tinyrl import harness


def test_eval_at_shift_step_is_pre_shift():
    task = {"name": "pendulum", "obs_mode": "trig"}
    run = {"steps": 600, "eval_every_steps": 200, "eval_episodes": 2}
    learner = {"kind": "baseline_random"}
    plain = harness.run(task, learner, 0, run)["evals"]
    shifted = harness.run(task, learner, 0, {**run, "shift_at_step": 400, "shift_level": 0.5})["evals"]
    assert shifted[:2] == plain[:2]      # steps 200 and 400
    assert shifted[2] != plain[2]        # step 600 is on the new task
