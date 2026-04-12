from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.models.train import run_experiment
from src.utils.config import load_config


def _variant_overrides() -> dict[str, dict]:
    return {
        "full_agentcf": {},
        "w_o_planner": {"ablation": {"w_o_planner": True}},
        "w_o_verifier_feedback": {"ablation": {"w_o_verifier_feedback": True}, "augmentation": {"max_retry_rounds": 0}},
        "w_o_selector": {"ablation": {"w_o_selector": True}, "augmentation": {"keep_top_k": 3}},
        "w_o_dynamic_threshold": {"ablation": {"w_o_dynamic_threshold": True}, "thresholds": {"filtering_mode": "fixed"}},
        "single_agent_version": {"ablation": {"single_agent_version": True}, "augmentation": {"method": "single_agent"}},
    }


def _merge(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


async def run_ablation(config_path: str) -> None:
    base = load_config(config_path)
    rows: list[dict] = []
    for name, override in _variant_overrides().items():
        cfg = _merge(base, override)
        cfg["methods"] = ["AgentCF (Ours)"]
        cfg_path = Path(f"outputs/logs/ablation_{name}.yaml")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        await run_experiment(str(cfg_path))
        main_path = Path("outputs/tables/main_results.csv")
        quality_path = Path("outputs/tables/quality_results.csv")
        if not main_path.exists():
            continue
        main_df = pd.read_csv(main_path)
        q_df = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
        acc = float(main_df.iloc[0]["SST-2 Acc"]) if len(main_df) else 0.0
        f1 = float(main_df.iloc[0]["SST-2 F1"]) if len(main_df) else 0.0
        label_success = float(q_df.iloc[-1]["Label Success"]) if len(q_df) else 0.0
        rows.append(
            {
                "Variant": name,
                "Acc": round(acc, 4),
                "F1": round(f1, 4),
                "Label Success": round(label_success, 4),
                "Human Validity": "",
            }
        )
    if rows:
        out = Path("outputs/tables/ablation_results.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation.yaml")
    args = parser.parse_args()
    asyncio.run(run_ablation(args.config))


if __name__ == "__main__":
    main()
