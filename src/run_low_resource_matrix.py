from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from math import floor
from pathlib import Path

import yaml

from src.data.load_data import load_dataset_splits
from src.data.preprocess import preprocess_df
from src.models.train import run_experiment
from src.utils.config import load_config
from src.utils.runtime import start_local_vllm, stop_local_vllm

RATIOS: list[tuple[str, float]] = [("10pct", 0.10), ("30pct", 0.30), ("50pct", 0.50), ("100pct", 1.00)]


def _slugify_config(config_path: str) -> str:
    return Path(config_path).stem


async def _run_one(cfg: dict, tmp_name: str) -> None:
    tmp = Path("outputs/logs") / tmp_name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    start_local_vllm(cfg)
    try:
        await run_experiment(str(tmp))
    finally:
        stop_local_vllm(cfg)


async def run_matrix(configs: list[str], output_root: str) -> None:
    for config_path in configs:
        base = load_config(config_path)
        base_samples = int(base.get("train_samples", 0))
        if base_samples <= 0:
            dataset_name = str(base.get("dataset", "sst2"))
            full_train = preprocess_df(load_dataset_splits(dataset_name)["train"])
            base_samples = len(full_train)

        config_name = _slugify_config(config_path)
        for ratio_name, fraction in RATIOS:
            cfg = deepcopy(base)
            cfg["train_samples"] = max(1, floor(base_samples * fraction))
            cfg["output_root"] = f"{output_root}/{ratio_name}/{config_name}"
            await _run_one(cfg, f"tmp_low_resource_{config_name}_{ratio_name}.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/baseline.yaml", "configs/agentcf.yaml"],
    )
    parser.add_argument("--output-root", default="outputs/low_resource_matrix")
    args = parser.parse_args()
    asyncio.run(run_matrix(args.configs, args.output_root))


if __name__ == "__main__":
    main()
