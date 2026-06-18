#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    ("relative_l2", "ID L2"),
    ("orbit_ood_relative_l2", "OOD L2"),
    ("equivariance_defect_relative", "Eq. defect"),
]

T_CRIT_95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636305,
    6: 2.4469118511449692,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
}


def _fmt(value: float) -> str:
    if abs(value) < 0.01 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _fmt_pm(mean: float, std: float) -> str:
    return rf"\({_fmt(mean)}\pm{_fmt(std)}\)"


def _tex_text(value: object) -> str:
    return str(value).replace("%", r"\%")


def _paired_rows(
    path: Path,
    *,
    setting: str,
    reference: str,
    candidate: str,
) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    return _paired_rows_from_frames(
        df[df["method"] == reference],
        df[df["method"] == candidate],
        setting=setting,
        reference_label=reference,
        candidate_label=candidate,
    )


def _apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for col, value in filters.items():
        if col not in out.columns:
            raise SystemExit(f"Missing filter column {col!r}")
        if isinstance(value, float):
            values = pd.to_numeric(out[col], errors="coerce")
            out = out[(values - value).abs() < 1e-9]
        else:
            out = out[out[col] == value]
    return out


def _paired_rows_from_tables(
    reference_path: Path,
    candidate_path: Path,
    *,
    setting: str,
    reference_label: str,
    candidate_label: str,
    reference_filters: dict[str, object],
    candidate_filters: dict[str, object],
) -> list[dict[str, object]]:
    reference_df = _apply_filters(pd.read_csv(reference_path), reference_filters)
    candidate_df = _apply_filters(pd.read_csv(candidate_path), candidate_filters)
    return _paired_rows_from_frames(
        reference_df,
        candidate_df,
        setting=setting,
        reference_label=reference_label,
        candidate_label=candidate_label,
    )


def _paired_rows_from_frames(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    setting: str,
    reference_label: str,
    candidate_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, label in METRICS:
        reference = reference_df[["seed", metric]].dropna().rename(columns={metric: "reference"})
        candidate = candidate_df[["seed", metric]].dropna().rename(columns={metric: "candidate"})
        pivot = reference.merge(candidate, on="seed", how="inner").sort_values("seed")
        if pivot.empty:
            raise SystemExit(f"No paired seeds found for {setting} / {metric}")
        diff = pivot["candidate"] - pivot["reference"]
        n = int(diff.shape[0])
        mean = float(diff.mean())
        std = float(diff.std(ddof=1)) if n > 1 else 0.0
        sem = std / (n**0.5) if n > 0 else 0.0
        tcrit = T_CRIT_95.get(n - 1, 1.96)
        half_width = tcrit * sem
        ref_mean = float(pivot["reference"].mean())
        cand_mean = float(pivot["candidate"].mean())
        reduction = 100.0 * (ref_mean - cand_mean) / ref_mean
        rows.append(
            {
                "setting": setting,
                "metric": label,
                "reference_method": reference_label,
                "candidate_method": candidate_label,
                "seed_count": n,
                "reference_mean": ref_mean,
                "candidate_mean": cand_mean,
                "paired_delta_mean": mean,
                "paired_delta_ci95_low": mean - half_width,
                "paired_delta_ci95_high": mean + half_width,
                "relative_reduction_pct": reduction,
            }
        )
    return rows


def write_paired_table(out_prefix: Path, table_dir: Path) -> pd.DataFrame:
    rows = []
    rows.extend(
        _paired_rows(
            table_dir / "2d_galilean_n64_2pct_headline_lambda_0p1.runs.csv",
            setting="N64 Galilean 2% labels",
            reference="aug",
            candidate="aug_orbit",
        )
    )
    rows.extend(
        _paired_rows_from_tables(
            table_dir / "2d_galilean_n64_label_efficiency_lambda_0p1.runs.csv",
            table_dir / "2d_galilean_n64_2pct_lambda_extended.runs.csv",
            setting="N64 Galilean 2% labels, lambda=0.30",
            reference_label="aug",
            candidate_label="aug_orbit_lambda_0.30",
            reference_filters={"method": "aug", "data_fraction": 0.02},
            candidate_filters={"method": "aug_orbit", "lambda_orbit": 0.30},
        )
    )
    rows.extend(
        _paired_rows(
            table_dir / "2d_galilean_n64_10pct_lambda_0p1.runs.csv",
            setting="N64 Galilean 10% labels",
            reference="aug",
            candidate="aug_orbit",
        )
    )
    rows.extend(
        _paired_rows(
            table_dir / "1d_final_four_method.runs.csv",
            setting="1D Burgers",
            reference="aug",
            candidate="aug_orbit",
        )
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_prefix.with_suffix(".paired.csv"), index=False)
    with out_prefix.with_suffix(".paired.tex").open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Paired seed differences for central comparisons. "
            "Delta is candidate minus reference, so negative values indicate improvement.}\n"
            "\\label{tab:paired-reviewer-comparisons}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{llrrrr}\n"
            "\\toprule\n"
            "Setting & Metric & Seeds & Reference & Candidate & Paired delta [95\\% CI] \\\\\n"
            "\\midrule\n"
        )
        for row in rows:
            setting = _tex_text(row["setting"])
            f.write(
                f"{setting} & {row['metric']} & {row['seed_count']} & "
                f"\\({_fmt(row['reference_mean'])}\\) & \\({_fmt(row['candidate_mean'])}\\) & "
                f"\\({_fmt(row['paired_delta_mean'])}\\,[{_fmt(row['paired_delta_ci95_low'])},"
                f"{_fmt(row['paired_delta_ci95_high'])}]\\) \\\\\n"
            )
        f.write(
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "}\n"
            "\\end{table*}\n"
        )
    return df


def write_absolute_label_table(out_prefix: Path, table_dir: Path) -> pd.DataFrame:
    parts = [
        pd.read_csv(table_dir / "2d_galilean_n64_label_efficiency_lambda_0p1.aggregate.csv"),
        pd.read_csv(table_dir / "2d_galilean_n64_10pct_lambda_0p1.aggregate.csv"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["method"].isin(["aug", "aug_orbit"])].copy()
    df = df.sort_values(["data_fraction", "method"])
    columns = [
        "data_fraction",
        "method",
        "relative_l2_mean",
        "relative_l2_std",
        "orbit_ood_relative_l2_mean",
        "orbit_ood_relative_l2_std",
        "equivariance_defect_relative_mean",
        "equivariance_defect_relative_std",
        "latency_ms_per_sample_mean",
        "latency_ms_per_sample_std",
        "relative_l2_count",
    ]
    df[columns].to_csv(out_prefix.with_suffix(".absolute_label_efficiency.csv"), index=False)
    with out_prefix.with_suffix(".absolute_label_efficiency.tex").open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Absolute N64 Galilean label-efficiency metrics. "
            "This table complements percentage reductions with the underlying values.}\n"
            "\\label{tab:n64-label-efficiency-absolute}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{llrrrrr}\n"
            "\\toprule\n"
            "Labels & Method & ID L2 & OOD L2 & Eq. defect & Latency ms/sample & Seeds \\\\\n"
            "\\midrule\n"
        )
        for _, row in df.iterrows():
            method = "FNO + aug." if row["method"] == "aug" else "FNO + aug. + orbit"
            f.write(
                f"{100 * float(row['data_fraction']):.0f}\\% & {method} & "
                f"{_fmt_pm(row['relative_l2_mean'], row['relative_l2_std'])} & "
                f"{_fmt_pm(row['orbit_ood_relative_l2_mean'], row['orbit_ood_relative_l2_std'])} & "
                f"{_fmt_pm(row['equivariance_defect_relative_mean'], row['equivariance_defect_relative_std'])} & "
                f"{_fmt_pm(row['latency_ms_per_sample_mean'], row['latency_ms_per_sample_std'])} & "
                f"{int(row['relative_l2_count'])} \\\\\n"
            )
        f.write(
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "}\n"
            "\\end{table*}\n"
        )
    return df[columns]


def write_wrong_symmetry_table(out_prefix: Path, table_dir: Path) -> pd.DataFrame:
    headline = pd.read_csv(table_dir / "2d_galilean_n64_2pct_headline_lambda_0p1.runs.csv")
    wrong = pd.read_csv(table_dir / "2d_galilean_n64_2pct_wrong_symmetry.runs.csv")
    no_output_path = table_dir / "2d_galilean_n64_2pct_no_output_control.runs.csv"
    no_output = pd.read_csv(no_output_path) if no_output_path.exists() else pd.DataFrame()
    seed_sources = [set(headline["seed"]), set(wrong["seed"])]
    if not no_output.empty:
        seed_sources.append(set(no_output["seed"]))
    seed_set = sorted(set.intersection(*seed_sources))
    rows = []
    aug = headline[(headline["method"] == "aug") & (headline["seed"].isin(seed_set))].copy()
    aug["control_label"] = "FNO + aug."
    physical = headline[(headline["method"] == "aug_orbit") & (headline["seed"].isin(seed_set))].copy()
    physical["control_label"] = "FNO + aug. + orbit"
    shuffled = wrong[wrong["seed"].isin(seed_set)].copy()
    shuffled["control_label"] = "FNO + aug. + shuffled orbit"
    controls = [aug, physical, shuffled]
    if not no_output.empty:
        no_output = no_output[no_output["seed"].isin(seed_set)].copy()
        no_output["control_label"] = "FNO + aug. + no output transform"
        controls.append(no_output)
    combined = pd.concat(controls, ignore_index=True)
    for label, group in combined.groupby("control_label", sort=False):
        row = {"control_label": label, "seed_count": int(group["seed"].nunique())}
        for metric, _ in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_prefix.with_suffix(".wrong_symmetry.csv"), index=False)
    with out_prefix.with_suffix(".wrong_symmetry.tex").open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Wrong-symmetry control at 2\\% labels over matched seeds. "
            "The no-output row removes the physical output action, while the shuffled-orbit row "
            "pairs each transformed prediction with the wrong orbit target in the minibatch.}\n"
            "\\label{tab:n64-wrong-symmetry-control}\n"
            "\\begin{tabular}{lrrrr}\n"
            "\\toprule\n"
            "Method & ID L2 & OOD L2 & Eq. defect & Seeds \\\\\n"
            "\\midrule\n"
        )
        for _, row in df.iterrows():
            f.write(
                f"{row['control_label']} & "
                f"{_fmt_pm(row['relative_l2_mean'], row['relative_l2_std'])} & "
                f"{_fmt_pm(row['orbit_ood_relative_l2_mean'], row['orbit_ood_relative_l2_std'])} & "
                f"{_fmt_pm(row['equivariance_defect_relative_mean'], row['equivariance_defect_relative_std'])} & "
                f"{int(row['seed_count'])} \\\\\n"
            )
        f.write(
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewer-facing paper supplement tables.")
    parser.add_argument("--out-prefix", default="runs/paper_tables/2d_galilean_n64_reviewer")
    parser.add_argument("--table-dir", default="runs/paper_tables")
    args = parser.parse_args()
    out_prefix = Path(args.out_prefix)
    table_dir = Path(args.table_dir)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paired = write_paired_table(out_prefix, table_dir)
    absolute = write_absolute_label_table(out_prefix, table_dir)
    wrong = write_wrong_symmetry_table(out_prefix, table_dir)
    print(f"wrote {out_prefix.with_suffix('.paired.csv')}")
    print(f"wrote {out_prefix.with_suffix('.paired.tex')}")
    print(f"wrote {out_prefix.with_suffix('.absolute_label_efficiency.csv')}")
    print(f"wrote {out_prefix.with_suffix('.absolute_label_efficiency.tex')}")
    print(f"wrote {out_prefix.with_suffix('.wrong_symmetry.csv')}")
    print(f"wrote {out_prefix.with_suffix('.wrong_symmetry.tex')}")
    print(paired.to_string(index=False))
    print(absolute.to_string(index=False))
    print(wrong.to_string(index=False))


if __name__ == "__main__":
    main()
