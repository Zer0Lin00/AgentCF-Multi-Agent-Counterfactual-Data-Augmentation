#!/usr/bin/env bash
set -e
python3 -m src.run_low_resource_matrix --configs configs/baseline.yaml configs/agentcf.yaml
