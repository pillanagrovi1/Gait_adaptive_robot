"""scripts/results_table.py prints a table only if its provenance holds."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def build(root, commits=("abc1234", "abc1234"), in_summary=("table_bins5", "table_bins7"), n_seeds=2,
          config_seeds=(0, 1), raw_runs=("table_bins5", "table_bins7")):
    exp = {"name": "e", "task": {"name": "pendulum"}, "seeds": list(config_seeds),
           "learners": [{"label": "table", "base": {"kind": "tabular"}, "sweep": {"bins": [5, 7]}}]}
    for d in ("configs", "results/figures"):
        (root / d).mkdir(parents=True)
    (root / "configs/e.json").write_text(json.dumps(exp))
    for run in raw_runs:
        d = root / "results/raw/e" / run
        d.mkdir(parents=True)
        ran = {"task": exp["task"], "run": {}, "learner": {"kind": "tabular", "bins": int(run[len("table_bins"):])}}
        for seed, c in enumerate(commits):
            (d / f"seed{seed}_meta.json").write_text(json.dumps({**ran, **({"git_commit": c} if c else {})}))
    pd.DataFrame([{"run_id": r, "kind": "tabular", "n_seeds": n_seeds, "ci_method": "t", "final_mean": -300.0,
                   "final_ci95": 20.0, "auc_mean": -500.0, "auc_ci95": 10.0, "online_mean": -600.0,
                   "online_ci95": 5.0, "peak_bytes": 1008, "peak_source": "m4-qemu"} for r in in_summary]
                 ).to_csv(root / "results/figures/e_summary.csv", index=False)


def table(root, monkeypatch, capsys, *args):
    spec = importlib.util.spec_from_file_location("results_table", ROOT / "scripts" / "results_table.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    monkeypatch.setattr(rt, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["results_table.py", "e", *args])
    code = 0
    try:
        rt.main()
    except SystemExit as e:
        code = e.code
    return (code, *capsys.readouterr())


def test_good_case_prints(tmp_path, monkeypatch, capsys):
    build(tmp_path)
    code, out, _ = table(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "commit `abc1234`" in out and "run_experiment.sh e`" in out and "| table_bins7 |" in out


@pytest.mark.parametrize("case,message", [
    ({"commits": ("abc1234", "def5678")}, "more than one commit"),
    ({"commits": ("abc1234+dirty", "abc1234+dirty")}, "+dirty"),
    ({"n_seeds": 3}, "2 seed metas, summary says 3"),
    ({"config_seeds": (0, 1, 2)}, "config says 3"),
    ({"in_summary": ("table_bins5",)}, "table_bins7: in results/raw/ but not in the summary"),
    ({"raw_runs": ("table_bins5",)}, "table_bins7: 0 seed metas"),
])
def test_refuses(tmp_path, monkeypatch, capsys, case, message):
    build(tmp_path, **case)
    code, out, err = table(tmp_path, monkeypatch, capsys, "--command", "given")
    assert (code, out) == (1, "")
    assert message in err


def test_no_raw_files_prints_the_extract_command(tmp_path, monkeypatch, capsys):
    build(tmp_path, raw_runs=())
    code, out, err = table(tmp_path, monkeypatch, capsys)
    assert (code, out) == (1, "")
    assert "mkdir -p results/raw" in err and "tar -xzf results/raw_archive/e.tar.gz -C results/raw" in err


def test_zero_commits_is_not_an_error(tmp_path, monkeypatch, capsys):
    build(tmp_path, commits=(None, None))
    code, out, _ = table(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "commit not recorded (runs predate commit tracking)" in out and "HEAD now" not in out


def test_partial_experiment_needs_the_real_command(tmp_path, monkeypatch, capsys):
    build(tmp_path, in_summary=("table_bins5",), raw_runs=("table_bins5",))
    code, out, err = table(tmp_path, monkeypatch, capsys)
    assert (code, out) == (1, "")
    assert "--command" in err
    real = "python3 scripts/run_sweep.py e --only table"
    code, out, _ = table(tmp_path, monkeypatch, capsys, "--command", real)
    assert code == 0 and f"command `{real}`" in out and "run_experiment.sh" not in out


def test_baselines_have_no_memory_size(tmp_path, monkeypatch, capsys):
    build(tmp_path)
    p = tmp_path / "results/figures/e_summary.csv"
    d = pd.read_csv(p)
    d.loc[0, ["kind", "peak_source"]] = ["baseline_random", "host-estimate"]
    d.loc[0, "peak_bytes"] = 220
    d.to_csv(p, index=False)
    code, out, _ = table(tmp_path, monkeypatch, capsys)
    assert code == 0 and "220" not in out and "| 1008 |" in out


def test_flash_and_saturation_columns_only_if_measured(tmp_path, monkeypatch, capsys):
    build(tmp_path)
    _, out, _ = table(tmp_path, monkeypatch, capsys)
    assert "flash" not in out and "saturation" not in out      # summaries from before D55/D56
    p = tmp_path / "results/figures/e_summary.csv"
    d = pd.read_csv(p)
    d["flash_bytes"] = [15540.0, float("nan")]
    d["saturation_frac"] = [0.0000042, float("nan")]
    d.to_csv(p, index=False)
    code, out, _ = table(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert "flash, bench firmware with all three learners (B) | update saturation |" in out
    assert "| 15540 | 0.00042% |" in out and "| — | — |\n" in out


def edit_summary(root, **cols):
    p = root / "results/figures/e_summary.csv"
    d = pd.read_csv(p)
    for k, v in cols.items():
        d[k] = v
    d.to_csv(p, index=False)


def test_refuses_runs_of_another_config(tmp_path, monkeypatch, capsys):     # 14
    build(tmp_path)
    p = tmp_path / "configs/e.json"
    exp = json.loads(p.read_text())
    exp["learners"][0]["base"]["alpha"] = 0.9
    p.write_text(json.dumps(exp))
    code, out, err = table(tmp_path, monkeypatch, capsys)
    assert (code, out) == (1, "") and "differ from configs/e.json" in err


def test_refuses_host_m4_mismatch(tmp_path, monkeypatch, capsys):           # 13
    build(tmp_path)
    edit_summary(tmp_path, host_match=[1, 0])
    code, out, err = table(tmp_path, monkeypatch, capsys)
    assert (code, out) == (1, "") and "host != M4" in err


def test_marks_what_does_not_fit(tmp_path, monkeypatch, capsys):            # 16
    build(tmp_path)
    edit_summary(tmp_path, fits_f446=[1, 0], peak_bytes=[1008, 166188], peak_source=["m4-qemu", "host-estimate"])
    _, out, _ = table(tmp_path, monkeypatch, capsys)
    assert "| 166188 (est.), does not fit |" in out


def test_adaptation_columns(tmp_path, monkeypatch, capsys):                 # 17
    build(tmp_path)
    edit_summary(tmp_path, post_auc_mean=-400.0, post_auc_ci95=30.0, back_mean=-250.0, back_ci95=25.0,
                 recovered_mean=0.75)
    _, out, _ = table(tmp_path, monkeypatch, capsys)
    assert "post-shift AUC (B window)" in out and "-400 ± 30" in out and "-250 ± 25" in out and "75%" in out


def test_small_returns_keep_a_decimal(tmp_path, monkeypatch, capsys):       # 19
    build(tmp_path)
    edit_summary(tmp_path, final_mean=18.46, final_ci95=0.31)
    _, out, _ = table(tmp_path, monkeypatch, capsys)
    assert "| 18.5 ± 0.3 |" in out
