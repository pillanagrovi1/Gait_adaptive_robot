"""D59: a run is finished only once its seed<k>_meta.json exists."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_resume_reruns_interrupted_seed(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("run_sweep", ROOT / "scripts" / "run_sweep.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    first = rs.sweep.expand(rs.sweep.load("smoke"))[0]
    run_id = first["run_id"]
    d = tmp_path / "smoke" / run_id
    d.mkdir(parents=True)
    (d / "seed0.csv").write_text("interrupted: no meta file\n")
    (d / "seed1.csv").write_text("finished\n")
    (d / "seed1_meta.json").write_text(json.dumps({k: first[k] for k in ("task", "learner", "run")}))
    monkeypatch.setattr(rs.harness, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", "smoke", "--dry-run"])
    rs.main()
    out = capsys.readouterr().out
    assert f"{run_id} seed 0" in out
    assert f"{run_id} seed 1" not in out
