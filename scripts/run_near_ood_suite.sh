#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export PLANNER_MODEL="${PLANNER_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
export VERIFIER_MODEL="${VERIFIER_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"

VLLM_LOG="${VLLM_LOG:-/root/autodl-tmp/vllm_near_ood.log}"
VLLM_MODEL="${VLLM_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"

ensure_vllm() {
  if curl -sS "${OPENAI_BASE_URL%/v1}/health" >/dev/null 2>&1; then
    return
  fi
  echo "[Near-OOD] Starting local vLLM..."
  nohup /root/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --served-model-name Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    > "$VLLM_LOG" 2>&1 &

  for _ in $(seq 1 120); do
    if curl -sS "${OPENAI_BASE_URL%/v1}/health" >/dev/null 2>&1; then
      echo "[Near-OOD] vLLM is ready."
      return
    fi
    sleep 2
  done
  echo "[Near-OOD] vLLM failed to become ready in time." >&2
  exit 1
}

CONFIGS=(
  "configs/ood_sst2_rotten_agentcf.yaml"
  "configs/ood_sst2_yelp_agentcf.yaml"
  "configs/ood_sst2_amazon_agentcf.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  ensure_vllm
  echo "[Near-OOD] Running $cfg"
  python -m src.run_ood --config "$cfg"
done

echo "[Near-OOD] Done. Results:"
for out in \
  "outputs/ood_sst2_rotten_agentcf_v1/tables/ood_results.csv" \
  "outputs/ood_sst2_yelp_agentcf_v1/tables/ood_results.csv" \
  "outputs/ood_sst2_amazon_agentcf_v1/tables/ood_results.csv"; do
  if [[ -f "$out" ]]; then
    echo "  - $out"
  fi
done