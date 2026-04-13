from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SEEDS = [42, 43, 44]
LOW_RESOURCE_RATIOS = ["10pct", "30pct", "50pct", "100pct"]


def _group_mean_std(df: pd.DataFrame, by: list[str], value_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(by)[value_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [" ".join(c).strip() for c in agg.columns.to_flat_index()]
    return agg


def summarize_main(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    quality_rows = []
    for seed in SEEDS:
        seed_root = root / "main" / f"seed_{seed}"
        for main_path in sorted(seed_root.glob("*/tables/main_results.csv")):
            quality_path = main_path.parent / "quality_results.csv"
            if main_path.exists():
                df = pd.read_csv(main_path)
                df["Seed"] = seed
                rows.append(df)
            if quality_path.exists():
                q = pd.read_csv(quality_path)
                q["Seed"] = seed
                quality_rows.append(q)
    main_df = pd.concat(rows, ignore_index=True)
    main_summary = _group_mean_std(main_df, ["Method"], ["SST-2 Acc", "SST-2 F1"])
    if quality_rows:
        qual_df = pd.concat(quality_rows, ignore_index=True)
        qual_summary = _group_mean_std(qual_df, ["Method"], ["Label Success", "Semantic Sim", "Edit Similarity"])
    else:
        qual_summary = pd.DataFrame()
    return main_summary, qual_summary


def summarize_low_resource(root: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for ratio in LOW_RESOURCE_RATIOS:
            ratio_root = root / "low_resource" / f"seed_{seed}" / ratio
            for path in sorted(ratio_root.glob("*/tables/main_results.csv")):
                df = pd.read_csv(path)
                df["Seed"] = seed
                df["Ratio"] = ratio
                rows.append(df)
    low_df = pd.concat(rows, ignore_index=True)
    return _group_mean_std(low_df, ["Method", "Ratio"], ["SST-2 Acc"])


def summarize_ablation(project_root: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = project_root / "outputs" / "tables" / "repeats" / "ablation" / f"seed_{seed}_results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["Seed"] = seed
        rows.append(df)
    abl_df = pd.concat(rows, ignore_index=True)
    return _group_mean_std(abl_df, ["Variant"], ["Acc", "F1", "Label Success"])


def summarize_ood(root: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        seed_root = root / "main" / f"seed_{seed}"
        for path in sorted(seed_root.glob("*/tables/ood_results.csv")):
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["Seed"] = seed
            rows.append(df)
    ood_df = pd.concat(rows, ignore_index=True)
    return _group_mean_std(ood_df, ["Method", "OOD Dataset"], ["ID Acc", "OOD Acc", "Robustness Gap"])


def _to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--repeats-root", default="outputs/repeats")
    parser.add_argument("--output-dir", default="outputs/repeats_summary")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    repeats_root = project_root / args.repeats_root
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    main_summary, quality_summary = summarize_main(repeats_root)
    low_summary = summarize_low_resource(repeats_root)
    ablation_summary = summarize_ablation(project_root)
    ood_summary = summarize_ood(repeats_root)

    main_summary.to_csv(output_dir / "main_repeats_summary.csv", index=False)
    if not quality_summary.empty:
        quality_summary.to_csv(output_dir / "quality_repeats_summary.csv", index=False)
    low_summary.to_csv(output_dir / "low_resource_repeats_summary.csv", index=False)
    ablation_summary.to_csv(output_dir / "ablation_repeats_summary.csv", index=False)
    ood_summary.to_csv(output_dir / "ood_repeats_summary.csv", index=False)

    md_parts = [
        "# Repeated Experiment Summary / 三次重复实验汇总",
        "",
        "## Main Results / 主实验结果",
        "",
        _to_markdown_table(main_summary),
        "",
    ]
    if not quality_summary.empty:
        md_parts.extend(
            [
                "## Quality Evaluation / 质量评估",
                "",
                _to_markdown_table(quality_summary),
                "",
            ]
        )
    md_parts.extend(
        [
            "## Low-Resource Results / 低资源结果",
            "",
            _to_markdown_table(low_summary),
            "",
            "## Ablation Results / 消融结果",
            "",
            _to_markdown_table(ablation_summary),
            "",
            "## OOD Results / OOD 结果",
            "",
            _to_markdown_table(ood_summary),
            "",
        ]
    )
    (output_dir / "repeated_experiment_summary.md").write_text("\n".join(md_parts), encoding="utf-8")


if __name__ == "__main__":
    main()
