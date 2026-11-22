"""D56: flash = text + data of the benchmark firmware, in the summary."""
import json

import pytest

from tinyrl import analysis, harness


@pytest.mark.parametrize("measured,expected", [({"flash_text": "9000", "ram_data": "12"}, 9012), ({}, None)])
def test_flash_in_summary(tmp_path, monkeypatch, measured, expected):
    task = {"name": "pendulum"}
    learner = {"kind": "tabular", "bins": 5}
    run = {"steps": 600, "eval_every_steps": 200, "eval_episodes": 1}
    exp = {"name": "e", "task": task, "run": run, "seeds": [0], "learners": [{"label": "table", "base": learner}]}
    (tmp_path / "e.json").write_text(json.dumps(exp))
    meta = harness.run(task, learner, 0, run, out_dir=tmp_path / "e" / "table")["meta"]
    m4 = {"cfg_id": meta["cfg_id"], "peak_bytes": "700", "fits": "1", **measured}
    monkeypatch.setattr(analysis, "RAW", tmp_path)
    monkeypatch.setattr(analysis, "load_m4", lambda name: {"table": m4})
    row = analysis.summarize(tmp_path / "e.json").iloc[0]
    if expected is None:
        assert row["flash_bytes"] != row["flash_bytes"]      # NaN
    else:
        assert row["flash_bytes"] == expected
