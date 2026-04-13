from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ORDER = ["No Augmentation", "Standard Augmentation", "Single-LLM Counterfactual", "Single-LLM + Filtering", "AgentCF (Ours)"]
RATIOS = ["10pct", "30pct", "50pct", "100pct"]
LABELS = {"10pct": "10%", "30pct": "30%", "50pct": "50%", "100pct": "100%"}


def _collect(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for ratio in RATIOS:
        for config_name in ["baseline", "agentcf"]:
            csv_path = root / ratio / config_name / "tables" / "main_results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df["ratio"] = ratio
            rows.extend(df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []
    for method in ORDER:
        row = {"Method": method}
        for ratio in RATIOS:
            matched = df[(df["Method"] == method) & (df["ratio"] == ratio)]
            row[f"{LABELS[ratio]} Data Acc"] = round(float(matched.iloc[0]["SST-2 Acc"]) * 100, 2) if not matched.empty else None
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs/low_resource_matrix")
    parser.add_argument("--output-dir", default="outputs/low_resource_matrix")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _collect(input_root)
    if raw_df.empty:
        raise FileNotFoundError(f"No low-resource result CSVs found under {input_root}")

    raw_csv = output_dir / "low_resource_results_all.csv"
    raw_df.to_csv(raw_csv, index=False)

    summary_df = _build_summary(raw_df)
    summary_csv = output_dir / "low_resource_summary.csv"
    summary_md = output_dir / "low_resource_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text("# Low-Resource Summary\n\n" + _to_markdown(summary_df) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
