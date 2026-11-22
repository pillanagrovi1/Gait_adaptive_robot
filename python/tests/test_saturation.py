"""D55: the saturation counter reaches the run metadata."""
import pytest

from tinyrl import harness


@pytest.mark.parametrize("alpha,clips", [(0.3, False), (30000.0, True)])
def test_saturation_in_meta(alpha, clips):
    m = harness.run({"name": "pendulum"}, {"kind": "tabular", "bins": 5, "alpha": alpha}, 0,
                    {"steps": 400, "eval_every_steps": 200, "eval_episodes": 1})["meta"]
    assert (m["saturated"] > 0) == clips
    assert m["saturation_frac"] == m["saturated"] / m["update_terms"]
