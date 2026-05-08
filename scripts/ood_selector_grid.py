from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.data.load_data import load_dataset_splits, maybe_subsample, save_splits
from src.data.preprocess import preprocess_df
from src.models.classifier import HFClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed


GRID = [
    {"name": "current_proxy", "ratio": 0.45, "weight": 0.55, "final": 0.66, "semantic": 0.76, "domain": 0.62, "style": 0.00},
    {"name": "current_weight_100", "ratio": 0.45, "weight": 1.00, "final": 0.66, "semantic": 0.76, "domain": 0.62, "style": 0.00},
    {"name": "current_weight_075", "ratio": 0.45, "weight": 0.75, "final": 0.66, "semantic": 0.76, "domain": 0.62, "style": 0.00},
    {"name": "current_weight_035", "ratio": 0.45, "weight": 0.35, "final": 0.66, "semantic": 0.76, "domain": 0.62, "style": 0.00},
    {"name": "current_weight_025", "ratio": 0.45, "weight": 0.25, "final": 0.66, "semantic": 0.76, "domain": 0.62, "style": 0.00},
    {"name": "current_ratio_025", "ratio": 0.25, "weight": 0.35, "final": 0.66, "semantic": 0.76, "domain": 0.62, "style": 0.00},
    {"name": "conservative_low_weight", "ratio": 0.15, "weight": 0.35, "final": 0.70, "semantic": 0.78, "domain": 0.65, "style": 0.55},
    {"name": "high_conf_low_weight", "ratio": 0.20, "weight": 0.35, "final": 0.75, "semantic": 0.80, "domain": 0.70, "style": 0.60},
    {"name": "tiny_high_quality", "ratio": 0.08, "weight": 0.25, "final": 0.72, "semantic": 0.80, "domain": 0.70, "style": 0.65},
    {"name": "semantic_strict", "ratio": 0.20, "weight": 0.30, "final": 0.68, "semantic": 0.84, "domain": 0.62, "style": 0.65},
    {"name": "more_low_weight", "ratio": 0.35, "weight": 0.25, "final": 0.70, "semantic": 0.78, "domain": 0.65, "style": 0.60},
    {"name": "domain_strict_tiny", "ratio": 0.10, "weight": 0.30, "final": 0.70, "semantic": 0.78, "domain": 0.75, "style": 0.60},
]


def _method_dir(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def _merge(original: pd.DataFrame, aug: pd.DataFrame, ratio: float, weight: float) -> pd.DataFrame:
    base = original.copy()
    base["source"] = "original"
    base["sample_weight"] = 1.0
    if aug.empty or ratio <= 0:
        return base
    n = min(int(len(original) * ratio), len(aug))
    picked = aug.sample(n=n, random_state=42) if n < len(aug) else aug
    picked = picked[["id", "text", "label", "source"]].copy()
    picked["sample_weight"] = weight
    return pd.concat([base, picked], ignore_index=True)


def _select_aug(ver_path: Path, train_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    ver = pd.read_json(ver_path, lines=True)
    id_to_label = dict(zip(train_df["id"], train_df["label"]))
    for col, default in [
        ("general_sentiment_score", ver["label_score"]),
        ("domain_invariant_score", ver["label_score"]),
        ("style_robustness_score", 1.0),
    ]:
        if col not in ver.columns:
            ver[col] = default
    keep = ver[
        (ver["final_score"] >= spec["final"])
        & (ver["semantic_score"] >= spec["semantic"])
        & (ver["domain_invariant_score"] >= spec["domain"])
        & (ver["style_robustness_score"] >= spec["style"])
    ].copy()
    if keep.empty:
        return pd.DataFrame(columns=["id", "text", "label", "source", "candidate_id", "final_score"])
    keep["selector_proxy_score"] = (
        0.40 * keep["final_score"]
        + 0.25 * keep["domain_invariant_score"]
        + 0.20 * keep["semantic_score"]
        + 0.15 * keep["style_robustness_score"]
    )
    picked = keep.sort_values("selector_proxy_score", ascending=False).groupby("id", as_index=False).head(1)
    rows = []
    for _, row in picked.iterrows():
        original_label = int(id_to_label.get(row["id"], 0))
        rows.append(
            {
                "id": row["id"],
                "text": row["candidate_text"],
                "label": 1 - original_label,
                "source": "agentcf",
                "candidate_id": row["candidate_id"],
                "final_score": float(row["final_score"]),
            }
        )
    return pd.DataFrame(rows)


def _run_one(seed: int, cfg: dict, generated_root: Path, out_root: Path, grid: list[dict]) -> list[dict]:
    set_seed(seed)
    cfg = deepcopy(cfg)
    cfg["seed"] = seed
    cfg["output_root"] = str(out_root / f"seed_{seed}")
    cfg["ood_eval_samples"] = int(cfg.get("ood_eval_samples", 5000))

    id_splits = load_dataset_splits(cfg["dataset"])
    id_splits = {k: preprocess_df(v) for k, v in id_splits.items()}
    id_splits["train"] = maybe_subsample(id_splits["train"], int(cfg.get("train_samples", 0)), seed)
    id_splits["validation"] = maybe_subsample(id_splits["validation"], int(cfg.get("eval_samples", 0)), seed)
    save_splits(id_splits, Path(cfg["output_root"]) / "data" / "processed")

    ood_splits = load_dataset_splits(cfg["ood_dataset"])
    ood_splits = {k: preprocess_df(v) for k, v in ood_splits.items()}
    ood_eval = maybe_subsample(ood_splits["test"].reset_index(drop=True), int(cfg["ood_eval_samples"]), seed)

    train_df = id_splits["train"]
    val_df = id_splits["validation"]
    id_test = id_splits.get("test", val_df)
    rows: list[dict] = []

    baseline_train = _merge(train_df, pd.DataFrame(), 0.0, 1.0)
    clf = HFClassifier(model_name=cfg["model_name"], max_length=int(cfg["max_length"]))
    run_dir = Path(cfg["output_root"]) / "checkpoints" / "no_aug_proxy"
    val = clf.train_and_eval(baseline_train, val_df, cfg, out_dir=str(run_dir))
    id_metrics = clf.evaluate_df(id_test, cfg, out_dir=str(run_dir / "id_eval"))
    ood = clf.evaluate_df(ood_eval, cfg, out_dir=str(run_dir / "ood_eval"))
    rows.append({"Seed": seed, "Config": "no_aug_proxy", "Selected": 0, "TrainSize": len(baseline_train), "ID Acc": id_metrics["acc"], "Validation Acc": val["acc"], "OOD Acc": ood["acc"], "OOD F1": ood["f1"]})

    ver_path = generated_root / f"seed_{seed}" / "checkpoints" / "verifications.jsonl"
    for spec in grid:
        aug = _select_aug(ver_path, train_df, spec)
        merged = _merge(train_df, aug, spec["ratio"], spec["weight"])
        clf = HFClassifier(model_name=cfg["model_name"], max_length=int(cfg["max_length"]))
        run_dir = Path(cfg["output_root"]) / "checkpoints" / _method_dir(spec["name"])
        val = clf.train_and_eval(merged, val_df, cfg, out_dir=str(run_dir))
        id_metrics = clf.evaluate_df(id_test, cfg, out_dir=str(run_dir / "id_eval"))
        ood = clf.evaluate_df(ood_eval, cfg, out_dir=str(run_dir / "ood_eval"))
        rows.append({"Seed": seed, "Config": spec["name"], "Selected": len(aug), "TrainSize": len(merged), "ID Acc": id_metrics["acc"], "Validation Acc": val["acc"], "OOD Acc": ood["acc"], "OOD F1": ood["f1"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ood_agentcf.yaml")
    parser.add_argument("--generated-root", default="outputs/ood_agentcf_v1")
    parser.add_argument("--output-root", default="outputs/ood_selector_grid_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--ood-eval-samples", type=int, default=5000)
    parser.add_argument("--grid-names", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["ood_eval_samples"] = args.ood_eval_samples
    cfg["runtime"]["release_vllm_after_generation"] = False
    grid = [spec for spec in GRID if not args.grid_names or spec["name"] in set(args.grid_names)]
    if not grid:
        raise SystemExit("No grid specs selected")
    out_root = Path(args.output_root)
    all_rows = []
    for seed in args.seeds:
        all_rows.extend(_run_one(seed, cfg, Path(args.generated_root), out_root, grid))
    raw = pd.DataFrame(all_rows)
    summary = raw.groupby("Config")[["OOD Acc", "OOD F1", "ID Acc", "Validation Acc", "Selected", "TrainSize"]].agg(["mean", "std"]).reset_index()
    summary.columns = [" ".join(c).strip() for c in summary.columns.to_flat_index()]
    out_root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_root / "ood_selector_grid_raw.csv", index=False)
    summary.sort_values("OOD Acc mean", ascending=False).to_csv(out_root / "ood_selector_grid_summary.csv", index=False)
    print(summary.sort_values("OOD Acc mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
