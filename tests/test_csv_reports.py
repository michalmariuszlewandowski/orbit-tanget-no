from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claim_summary_reproduces_archived_scientific_reductions(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = _script("reproduce_paper_artifacts")
    directory = ROOT / "runs" / "paper_tables"
    actual = module._claim_summary(
        headline=pd.read_csv(directory / "2d_galilean_n64_2pct_headline_lambda_0p1.aggregate.csv"),
        label_efficiency=pd.read_csv(
            directory / "2d_galilean_n64_label_efficiency_lambda_0p1.aggregate.csv"
        ),
        severity=pd.read_csv(
            directory / "2d_galilean_n64_2pct_ood_severity_lambda_0p1.aggregate.csv"
        ),
    )
    expected = pd.read_csv(directory / "2d_galilean_n64_claim_summary.csv")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


def test_metric_tables_write_csv_only(tmp_path, monkeypatch):
    module = _script("make_paper_tables")
    for seed, relative_l2 in ((23, 0.1), (31, 0.3)):
        run = tmp_path / "runs" / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "test_metrics.json").write_text(
            json.dumps({"method": "aug", "seed": seed, "relative_l2": relative_l2}),
            encoding="utf-8",
        )
    prefix = tmp_path / "tables" / "main"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_paper_tables.py",
            "--runs",
            str(tmp_path / "runs"),
            "--out-prefix",
            str(prefix),
            "--group-cols",
            "method",
        ],
    )
    module.main()
    assert {path.name for path in prefix.parent.iterdir()} == {
        "main.runs.csv",
        "main.aggregate.csv",
    }
    runs = pd.read_csv(prefix.with_suffix(".runs.csv"))
    assert set(runs["seed"]) == {23, 31}
    assert pd.read_csv(prefix.with_suffix(".aggregate.csv"))["relative_l2_mean"].iloc[
        0
    ] == pytest.approx(0.2)


def _cached_compute_tables(module, directory):
    prefix = directory / "2d_galilean_n64_training_compute"
    columns = [
        "optimizer_steps",
        "model_forward_passes_est",
        "backward_passes_est",
        "augmented_labeled_samples_est",
        "orbit_consistency_samples_est",
    ]
    runs = pd.DataFrame(
        [
            {
                "run_dir": f"missing/seed_{seed}",
                "method": "aug",
                "method_label": "FNO + aug.",
                "seed": seed,
                "data_fraction": 0.02,
                "steps_per_epoch": 6,
                **dict.fromkeys(columns, 600),
            }
            for seed in (23, 31)
        ]
    )
    aggregate = (
        runs.groupby(["method_label", "data_fraction", "steps_per_epoch"])[columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = ["_".join(part for part in col if part) for col in aggregate.columns]
    directory.mkdir(parents=True)
    runs.to_csv(prefix.with_suffix(".runs.csv"), index=False)
    aggregate.to_csv(prefix.with_suffix(".aggregate.csv"), index=False)
    return prefix


def test_cached_diagnostics_reproduce_figure_without_training_logs(tmp_path, monkeypatch):
    module = _script("make_paper_diagnostics")
    table_dir = tmp_path / "tables"
    prefix = _cached_compute_tables(module, table_dir)
    preserved = {path: path.read_bytes() for path in table_dir.iterdir()}
    observations = pd.DataFrame(
        [
            {
                "run_dir": f"missing/seed_{seed}",
                "method": "aug",
                "seed": seed,
                "data_fraction": 0.02,
                "lambda_orbit": 0.1,
                "config.symmetry.max_boost": 0.5,
                "max_boost": 0.5,
                "severity": "small",
                "relative_l2": value,
                "orbit_ood_relative_l2": value,
                "equivariance_defect_relative": value,
            }
            for seed, value in ((23, 0.1), (31, 0.3), (47, 0.5))
        ]
    )
    input_csv = tmp_path / "input.csv"
    observations.to_csv(input_csv, index=False)

    def no_compute(*args, **kwargs):
        raise AssertionError("cached diagnostics attempted to recompute from training logs")

    monkeypatch.setattr(module, "write_compute_table", no_compute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_paper_diagnostics.py",
            "--cached-compute",
            "--root",
            str(tmp_path),
            "--table-dir",
            str(table_dir),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--headline-csv",
            str(input_csv),
            "--label-csv",
            str(input_csv),
            "--lambda-csv",
            str(input_csv),
            "--severity-csv",
            str(input_csv),
            "--oracle-runs-csv",
            str(tmp_path / "absent-oracle.csv"),
        ],
    )
    module.main()
    assert not (tmp_path / "missing").exists()
    assert all(path.read_bytes() == contents for path, contents in preserved.items())
    assert prefix.with_suffix(".aggregate.csv").exists()
    assert {path.suffix for path in (tmp_path / "figures").iterdir()} == {".pdf", ".png"}
    assert not list(tmp_path.rglob("*.tex"))


@pytest.mark.parametrize("problem", ["missing", "empty", "schema", "counts", "groups"])
def test_cached_diagnostics_reject_incomplete_or_invalid_compute_csvs(tmp_path, problem):
    module = _script("make_paper_diagnostics")
    prefix = _cached_compute_tables(module, tmp_path / "tables")
    path = prefix.with_suffix(".aggregate.csv")
    if problem == "missing":
        path.unlink()
    elif problem == "empty":
        path.write_text("", encoding="utf-8")
    else:
        frame = pd.read_csv(path)
        if problem == "schema":
            frame = frame.drop(columns=["optimizer_steps_mean"])
        elif problem == "counts":
            frame.loc[0, "optimizer_steps_mean"] = float("inf")
        else:
            frame.loc[0, "method_label"] = "other method"
        frame.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="Missing input CSV|Invalid cached compute"):
        module.load_cached_compute_table(prefix)
