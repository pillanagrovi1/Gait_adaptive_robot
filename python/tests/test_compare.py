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
                 "adaptation_pendulum": 12, "stream_ablation": 10, "go1_gait_v3": 10,
                 "cartpole_main_v2_tuned": 6, "pendulum_main_v2_tuned": 4,
                 "go1_gait_v4": 5, "pendulum_linear_wide_tuned": 0,
                 "adaptation_pendulum_v2": 14}


@pytest.mark.parametrize("ratio,band", [(1.0, "about equal"), (0.67, "about equal"), (1.5, "about equal"),
                                        (3.0, "3-10x"), (10.0, "3-10x"),
                                        (0.66, None), (2.9, None), (10.1, None)])
def test_memory_bands(cmp, ratio, band):
    got = cmp.memory_band(ratio)
    assert (got is None) if band is None else got.startswith(band)


def fake_sweep(cmp, monkeypatch):
    """Two encodings. At equal SRAM linear is bad on trig and good on angle; at 5x it is better."""
    import pandas as pd
    val = {"S_angle": -200, "S_trig": -210, "L_angle_1": -220, "L_trig_1": -700, "L_angle_5": -100, "L_trig_5": -90}
    rows = [{"run_id": k, "label": "net-stream" if k[0] == "S" else "linear", "sweep_obs_mode": k.split("_")[1],
             "sweep_features": float("nan") if k[0] == "S" else float(k[-1]), "final_mean": v,
             "peak_bytes": 5000 if k.endswith("_5") else 1000} for k, v in val.items()]
    monkeypatch.setattr(cmp.analysis, "summarize", lambda exp: pd.DataFrame(rows))
    monkeypatch.setattr(cmp.analysis, "paired_diff", lambda exp, a, b, m: {"mean_diff": val[a] - val[b], "ci95": 50.0})


def test_by_sram_pairs_and_counts(cmp, monkeypatch, capsys):
    fake_sweep(cmp, monkeypatch)
    cmp.by_sram("e")
    out = capsys.readouterr().out
    same, across = out.split("linear's better encoding")
    assert "| S_angle (1000) | L_angle_1 (1000) |" in same and "| S_trig (1000) | L_trig_1 (1000) |" in same
    assert "| S_angle (1000) | L_trig_1 (1000) |" not in same and "| S_trig (1000) | L_angle_5 (5000) |" not in same
    assert "| about equal (within 1.5x) | final | 2 | 1 | 1 | 0 |" in same       # +20 and +490
    assert "| 3-10x less for the streaming net | final | 2 | 0 | 0 | 2 |" in same
    assert "| S_trig (1000) | L_angle_1 (1000) |" in across and "L_trig_1" not in across


def test_by_sram_without_rows_refuses(cmp, monkeypatch, capsys):
    import pandas as pd
    for empty in (pd.DataFrame(), pd.DataFrame([{"run_id": "t", "label": "table", "peak_bytes": 1000}])):
        monkeypatch.setattr(cmp.analysis, "summarize", lambda exp: empty)
        with pytest.raises(SystemExit) as e:
            cmp.by_sram("e")
        assert "tar -xzf results/raw_archive/e.tar.gz -C results/raw" in str(e.value.code)
        assert capsys.readouterr().out == ""
