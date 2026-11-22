"""D52: intervals on paired differences are Student-t at every n."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tinyrl import analysis


@pytest.mark.parametrize("n", [10, 20, 30])
def test_paired_diff_uses_t(monkeypatch, n):
    d = np.random.default_rng(1).normal(size=n)
    runs = {"a": pd.Series(d), "b": pd.Series(np.zeros(n))}
    monkeypatch.setattr(analysis, "per_seed", lambda exp, run, metric: runs[run])
    r = analysis.paired_diff("any", "a", "b")
    assert r["ci_method"] == "t"
    assert r["ci95"] == pytest.approx(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
