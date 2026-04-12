from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_main_table(rows: list[dict], out_path: str = "outputs/tables/main_results.csv") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def save_quality_table(rows: list[dict], out_path: str = "outputs/tables/quality_results.csv") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

