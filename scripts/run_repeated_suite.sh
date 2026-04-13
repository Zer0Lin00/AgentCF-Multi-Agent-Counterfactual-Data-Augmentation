#!/usr/bin/env bash
set -e
python3 -m src.run_repeated_suite --tasks main low_resource ablation ood --seeds 42 43 44
