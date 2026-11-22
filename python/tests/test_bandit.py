"""D54: a bandit baseline is its Q-learning twin with gamma = 0 and nothing else changed."""
import pytest

from tinyrl import sweep


@pytest.mark.parametrize("exp,bandit,twin", [
    ("stream_ablation", "table-bandit", "table-sgd"),
    ("stream_ablation", "linear-bandit", "linear-sgd"),
    ("go1_gait_v3", "linear-bandit", "linear"),
])
def test_bandit_twin(exp, bandit, twin):
    L = {x["label"]: x for x in sweep.load(exp)["learners"]}
    assert L[bandit] == {**L[twin], "label": bandit, "base": {**L[twin]["base"], "gamma": 0.0}}
