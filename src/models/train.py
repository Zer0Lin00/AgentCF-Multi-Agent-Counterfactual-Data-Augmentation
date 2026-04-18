from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.augmentation.agentcf_pipeline import build_agentcf_aug
from src.augmentation.single_agent import build_single_agent_cf
from src.augmentation.single_cf import build_single_cf
from src.augmentation.standard_aug import build_standard_aug
from src.data.load_data import load_dataset_splits, maybe_subsample, save_splits
from src.data.preprocess import preprocess_df
from src.models.classifier import HFClassifier
from src.models.evaluate import save_main_table, save_quality_table
from src.utils.config import load_config
from src.utils.seed import set_seed


def _merge_train(original: pd.DataFrame, aug: pd.DataFrame, ratio: float) -> pd.DataFrame:
    if aug.empty:
        out = original.copy()
        out["source"] = "original"
        return out
    n = min(int(len(original) * ratio), len(aug))
    picked = aug.sample(n=n, random_state=42) if n < len(aug) else aug
    base = original.copy()
    base["source"] = "original"
    return pd.concat([base, picked[["id", "text", "label", "source"]]], ignore_index=True)


async def run_experiment(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))

    splits = load_dataset_splits(cfg["dataset"])
    splits = {k: preprocess_df(v) for k, v in splits.items()}
    low_resource_ratio = float(cfg.get("low_resource_ratio", 1.0))
    if 0 < low_resource_ratio < 1.0:
        sampled, _ = train_test_split(
            splits["train"],
            train_size=low_resource_ratio,
            stratify=splits["train"]["label"],
            random_state=int(cfg["seed"]),
        )
        splits["train"] = sampled.reset_index(drop=True)
    splits["train"] = maybe_subsample(splits["train"], int(cfg.get("train_samples", 0)), int(cfg["seed"]))
    splits["validation"] = maybe_subsample(
        splits["validation"], int(cfg.get("eval_samples", 0)), int(cfg["seed"])
    )
    save_splits(splits, "data/processed")

    train_df = splits["train"]
    val_df = splits["validation"]
    ratio = float(cfg["augmentation"].get("ratio", 1.0))

    methods = cfg.get(
        "methods",
        [
            "No Augmentation",
            "Standard Augmentation",
            "Single-LLM Counterfactual",
            "Single-LLM + Filtering",
            "AgentCF (Ours)",
        ],
    )

    quality_rows: list[dict] = []
    result_rows: list[dict] = []
    for method in methods:
        clf = HFClassifier(model_name=cfg["model_name"], max_length=int(cfg["max_length"]))
        aug_method = str(cfg.get("augmentation", {}).get("method", "agentcf")).lower()
        if method == "No Augmentation":
            aug_df = pd.DataFrame(columns=["id", "text", "label", "source"])
            ver_df = pd.DataFrame()
        elif method == "Standard Augmentation":
            aug_df = build_standard_aug(train_df, ratio=ratio)
            ver_df = pd.DataFrame()
        elif method in {"Single-LLM Counterfactual", "Single-LLM + Filtering"}:
            aug_df, ver_df = await build_single_cf(train_df, cfg)
            if method == "Single-LLM + Filtering" and not ver_df.empty:
                ver_df = ver_df[
                    (ver_df["label_score"] >= cfg["thresholds"]["label_score"])
                    & (ver_df["semantic_score"] >= cfg["thresholds"]["semantic_score"])
                    & (ver_df["final_score"] >= cfg["thresholds"]["final_score"])
                ]
                aug_df = aug_df[aug_df["id"].isin(ver_df["id"])]
        else:
            if aug_method == "single_agent":
                aug_df, ver_df = await build_single_agent_cf(train_df, cfg)
            else:
                aug_df, _stats = await build_agentcf_aug(train_df, cfg)
                ver_path = "outputs/checkpoints/verifications.jsonl"
                ver_df = pd.read_json(ver_path, lines=True) if Path(ver_path).exists() else pd.DataFrame()

        merged_train = _merge_train(train_df, aug_df, ratio=ratio)
        out_dir = f"outputs/checkpoints/{method.lower().replace(' ', '_').replace('+', 'plus').replace('-', '_')}"
        metrics = clf.train_and_eval(merged_train, val_df, cfg, out_dir=out_dir)
        result_rows.append(
            {
                "Method": method,
                "SST-2 Acc": round(metrics["acc"], 4),
                "SST-2 F1": round(metrics["f1"], 4),
                "TrainSize": len(merged_train),
            }
        )

        if not ver_df.empty:
            quality_rows.append(
                {
                    "Method": method,
                    "Label Success": round(float(ver_df["label_score"].mean()), 4),
                    "Semantic Sim": round(float(ver_df["semantic_score"].mean()), 4),
                    "Edit Similarity": round(float(ver_df["minimality_score"].mean()), 4),
                    "Final Score": round(float(ver_df["final_score"].mean()), 4),
                }
            )

    save_main_table(result_rows)
    if quality_rows:
        save_quality_table(quality_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    asyncio.run(run_experiment(args.config))


if __name__ == "__main__":
    main()
