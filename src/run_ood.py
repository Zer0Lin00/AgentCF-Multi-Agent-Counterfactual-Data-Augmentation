from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import time

import pandas as pd

from src.augmentation.agentcf_pipeline import build_agentcf_aug
from src.augmentation.single_cf import build_single_cf
from src.augmentation.standard_aug import build_standard_aug
from src.data.load_data import load_dataset_splits, maybe_subsample, save_splits
from src.data.preprocess import preprocess_df
from src.models.classifier import HFClassifier
from src.models.evaluate import save_quality_table
from src.utils.config import load_config
from src.utils.runtime import LLM_METHODS
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


def _release_local_vllm(cfg: dict, method: str, already_released: bool) -> bool:
    if already_released or method not in LLM_METHODS:
        return already_released
    runtime_cfg = cfg.get("runtime", {})
    if not runtime_cfg.get("release_vllm_after_generation", False):
        return already_released
    base_url = os.getenv(cfg.get("llm", {}).get("base_url_env", "OPENAI_BASE_URL"), "")
    if "localhost" not in base_url and "127.0.0.1" not in base_url:
        return already_released
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(5)
    return True


async def run_ood(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    output_root = Path(cfg.get("output_root", "outputs"))

    id_splits = load_dataset_splits(cfg["dataset"])
    id_splits = {k: preprocess_df(v) for k, v in id_splits.items()}
    id_splits["train"] = maybe_subsample(id_splits["train"], int(cfg.get("train_samples", 0)), int(cfg["seed"]))
    id_splits["validation"] = maybe_subsample(id_splits["validation"], int(cfg.get("eval_samples", 0)), int(cfg["seed"]))
    save_splits(id_splits, str(output_root / "data" / "processed"))

    ood_splits = load_dataset_splits(cfg["ood_dataset"])
    ood_splits = {k: preprocess_df(v) for k, v in ood_splits.items()}
    ood_eval = ood_splits["test"].reset_index(drop=True)
    ood_eval_samples = int(cfg.get("ood_eval_samples", 0))
    if ood_eval_samples > 0:
        ood_eval = maybe_subsample(ood_eval, ood_eval_samples, int(cfg["seed"]))
    save_splits({"ood_test": ood_eval}, str(output_root / "data" / "processed"))

    train_df = id_splits["train"]
    val_df = id_splits["validation"]
    id_test_df = id_splits.get("test", val_df)
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
    ood_rows: list[dict] = []
    vllm_released = False

    for method in methods:
        clf = HFClassifier(model_name=cfg["model_name"], max_length=int(cfg["max_length"]))
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
            selected_path = output_root / "selected_counterfactuals" / "selected.jsonl"
            ver_path = output_root / "checkpoints" / "verifications.jsonl"
            if selected_path.exists() and ver_path.exists():
                print("[AgentCF] Found existing checkpoint, skipping LLM generation.", flush=True)
                aug_df = pd.read_json(selected_path, lines=True)
                ver_df = pd.read_json(ver_path, lines=True)
            else:
                aug_df, _ = await build_agentcf_aug(train_df, cfg)
                ver_df = pd.read_json(ver_path, lines=True)

        vllm_released = _release_local_vllm(cfg, method, vllm_released)
        merged_train = _merge_train(train_df, aug_df, ratio=ratio)
        run_dir = output_root / "checkpoints" / method.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        val_metrics = clf.train_and_eval(merged_train, val_df, cfg, out_dir=str(run_dir))
        id_metrics = clf.evaluate_df(id_test_df, cfg, out_dir=str(run_dir / "id_test_eval"))
        ood_metrics = clf.evaluate_df(ood_eval, cfg, out_dir=str(run_dir / "ood_eval"))

        ood_rows.append(
            {
                "Method": method,
                "ID Acc": round(id_metrics["acc"], 4),
                "ID F1": round(id_metrics["f1"], 4),
                "Validation Acc": round(val_metrics["acc"], 4),
                "Validation F1": round(val_metrics["f1"], 4),
                "OOD Dataset": str(cfg["ood_dataset"]),
                "OOD Acc": round(ood_metrics["acc"], 4),
                "OOD F1": round(ood_metrics["f1"], 4),
                "Robustness Gap": round(id_metrics["acc"] - ood_metrics["acc"], 4),
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

    out = output_root / "tables" / "ood_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ood_rows).to_csv(out, index=False)
    if quality_rows:
        save_quality_table(quality_rows, out_path=str(output_root / "tables" / "quality_results.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    asyncio.run(run_ood(args.config))


if __name__ == "__main__":
    main()
