"""scripts/compare.py: the pre-registered comparisons, as tables."""
import importlib.util
from pathlib import Path

import pytest

from tinyrl import sweep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cmp():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mean,half,word", [(5, 1, "higher"), (-5, 1, "lower"), (5, 6, "inconclusive"),
                                            (5, 5, "inconclusive"), (0, 0, "inconclusive"),
                                            (1, float("nan"), "inconclusive")])
def test_verdict(cmp, mean, half, word):
    assert word in cmp.verdict(mean, half)


def test_every_comparison_names_real_runs(cmp):
    for name, rows in cmp.COMPARISONS.items():
        for c in rows:
            for exp, run in ((name, c.a), (c.exp_b or name, c.b)):
                src = exp if (ROOT / "configs" / f"{exp}.json").exists() else exp.removesuffix("_tuned")
                assert run in {r["run_id"] for r in sweep.expand(sweep.load(src))}, (exp, run)


def test_primary_set_is_the_registered_one(cmp):
    n = {k: sum(c.primary for c in v) for k, v in cmp.COMPARISONS.items()}
    assert n == {"pendulum_main_tuned": 0, "replay_ratio": 1, "cartpole_main_tuned": 6,
                 "adaptation_pendulum": 12, "stream_ablation": 10, "go1_gait_v3": 10}
