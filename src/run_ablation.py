from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.models.train import run_experiment
from src.utils.config import load_config
from src.utils.runtime import start_local_vllm, stop_local_vllm


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


async def run_ablation(
    config_path: str,
    *,
    train_samples: int | None = None,
    run_name: str = "ablation",
) -> None:
    base = load_config(config_path)
    if train_samples is not None:
        base["train_samples"] = int(train_samples)
    rows: list[dict] = []
    for name, override in _variant_overrides().items():
        cfg = _merge(base, override)
        cfg["methods"] = ["AgentCF (Ours)"]
        cfg["output_root"] = f"outputs/{run_name}/{name}"
        cfg_path = Path(f"outputs/logs/{run_name}_{name}.yaml")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        start_local_vllm(cfg)
        try:
            await run_experiment(str(cfg_path))
        finally:
            stop_local_vllm(cfg)
        variant_root = Path(cfg["output_root"])
        main_path = variant_root / "tables" / "main_results.csv"
        quality_path = variant_root / "tables" / "quality_results.csv"
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
        out = Path(f"outputs/tables/{run_name}_results.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--run-name", default="ablation")
    args = parser.parse_args()
    asyncio.run(run_ablation(args.config, train_samples=args.train_samples, run_name=args.run_name))


if __name__ == "__main__":
    main()
