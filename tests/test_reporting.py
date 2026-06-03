import json

import pandas as pd

from otno.reporting import aggregate_runs, collect_run_rows


def test_collect_run_rows_adds_label_fraction_from_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"method": "aug_orbit", "relative_l2": 0.1}),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "training:",
                "  data_fraction: 0.25",
                "  lambda_orbit: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    rows = collect_run_rows(tmp_path)

    assert rows[0]["data_fraction"] == 0.25
    assert rows[0]["lambda_orbit"] == 0.5
    assert rows[0]["config.training.data_fraction"] == 0.25


def test_aggregate_runs_groups_by_fraction_and_flattens_columns():
    df = pd.DataFrame(
        [
            {
                "method": "aug",
                "model_name": "fno1d",
                "dataset_kind": "burgers1d",
                "data_fraction": 0.05,
                "relative_l2": 0.2,
            },
            {
                "method": "aug",
                "model_name": "fno1d",
                "dataset_kind": "burgers1d",
                "data_fraction": 0.25,
                "relative_l2": 0.1,
            },
        ]
    )

    agg = aggregate_runs(df)

    assert "relative_l2_mean" in agg.columns
    assert len(agg.columns) == len(set(agg.columns))
    assert set(agg["data_fraction"]) == {0.05, 0.25}
