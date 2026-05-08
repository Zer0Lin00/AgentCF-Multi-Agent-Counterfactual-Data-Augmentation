# AgentCF 实验项目

本项目实现了 `AgentCF: Multi-Agent Counterfactual Data Augmentation` 的可运行版本，覆盖：
- SST-2 主任务
- 4 个基线与 AgentCF 主流程
- 低资源与消融脚本
- 自动质量评估与结果表导出

当前这份代码已经不只是 SST-2 的单次复现，而是同时支持：
- SST-2 ID 训练 + OOD 评估
- 近域 OOD：Rotten Tomatoes / Yelp / Amazon
- NLI：MNLI / HANS
- AgentCF / No-Filter / Single-LLM / Single-LLM + Filtering / Standard / No Augmentation 等对照方法

## 近期改进（摘要）

我们针对工程化与可复现性做了多项改进：
- 增加 checkpoint / 断点续跑支持：生成阶段（selected.jsonl / verifications.jsonl）可被重用，避免重复调用 LLM。
- 重构 `verifier` 的加权 scoring 与阈值逻辑，提高质量评估一致性；新增可直接从 `verifications.jsonl` 生成 `quality_results.csv` 的工具。
- 支持 NLI（MNLI/HANS）判定逻辑与相应 prompt，便于扩展到 NLI 任务。
- 新增 `no-filter` 变体与多种 configs（见 `configs/`），便于做消融与对照实验。
- 改进并发与显存管理：在生成后自动释放 vLLM（`runtime.release_vllm_after_generation`），并提供 `checkpoint_every_n_samples` 与 `concurrency` 配置项。

这些改进主要体现在 `configs/`、`src/agents/`、`src/augmentation/` 与 `scripts/` 中。

## 1. 环境依赖

Python 3.12，CUDA 12.8（Blackwell 架构 GPU 需要）。

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. LLM 部署（vLLM）

本项目需要一个 OpenAI 兼容的 LLM API。推荐使用 vLLM 本地部署：

```bash
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载模型（国内镜像）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /path/to/Qwen2.5-7B-Instruct

# 启动服务
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen2.5-7B-Instruct \
  --port 8000 --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.95
```

建议在 `screen` 或 `tmux` 中后台运行。

## 3. 配置 .env

在项目根目录创建 `.env`：

```env
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8000/v1
PLANNER_MODEL=/path/to/Qwen2.5-7B-Instruct
GENERATOR_MODEL=/path/to/Qwen2.5-7B-Instruct
VERIFIER_MODEL=/path/to/Qwen2.5-7B-Instruct
```

当 API 不可用时，系统会自动回退到规则版 agent，保证流程可跑。

## 4. 数据下载方式

首次运行自动通过 HuggingFace `datasets` 下载，国内需设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

支持数据集：
- `glue/sst2`
- `mnli`
- `hans`
- `rotten_tomatoes`
- `yelp_polarity`
- `amazon_polarity`
- `imdb`

## 5. 运行 Baseline

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m src.models.train --config configs/baseline.yaml
```

包含以下方法：No Augmentation、Standard Augmentation、Single-LLM Counterfactual、Single-LLM + Filtering。

**注意**：运行 baseline 时 vLLM 必须在运行。训练 DistilBERT 时若显存不足，可先关闭 vLLM：

```bash
pkill -f vllm
```

## 6. 运行 AgentCF

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m src.models.train --config configs/agentcf.yaml
```

AgentCF 分两阶段：
1. **LLM 生成阶段**：调用 vLLM 生成反事实样本，结果存入 `outputs/checkpoints/`
2. **训练阶段**：关闭 vLLM 释放显存，用生成数据训练 DistilBERT

若中途中断，重新运行会自动跳过 LLM 生成阶段（断点续传）。

## 7. 运行 OOD / Near-OOD

OOD 实验直接通过 `src/run_ood.py` 驱动。它会在同一个配置里按顺序跑以下方法：
- `No Augmentation`
- `Standard Augmentation`
- `Single-LLM Counterfactual`
- `Single-LLM + Filtering`
- `AgentCF (Ours)`

如果你只想跑某一个 OOD 场景，可以直接指定 config：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1

# SST-2 -> Rotten / Yelp / Amazon
python -m src.run_ood --config configs/ood_sst2_rotten_agentcf.yaml
python -m src.run_ood --config configs/ood_sst2_yelp_agentcf.yaml
python -m src.run_ood --config configs/ood_sst2_amazon_agentcf.yaml

# NLI -> MNLI / HANS
python -m src.run_ood --config configs/mnli_hans_agentcf.yaml
```

如果你想一次性跑完整近域 OOD 套件，直接用脚本：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/run_near_ood_suite.sh
```

恢复 / 调试建议：
- 如果本地 vLLM 已启动，`src.run_ood.py` 会直接复用 `OPENAI_BASE_URL`。
- 若你在离线环境里运行，建议先设置 `DISABLE_SAFETENSORS_CONVERSION=1`，避免 transformers 触发在线转换请求。
- 生成阶段的中间结果保存在 `outputs/<run_name>/selected_counterfactuals/selected.jsonl` 和 `outputs/<run_name>/checkpoints/verifications.jsonl`，重新运行会优先复用这些 checkpoint。
- 结果表会写到 `outputs/<run_name>/tables/ood_results.csv` 和 `outputs/<run_name>/tables/quality_results.csv`。

## 8. 运行 low-resource 实验

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/run_low_resource.sh
```

如果要跑与你当前主实验对应的完整 low-resource 矩阵（`10%/30%/50%/100%`，并覆盖 `baseline + AgentCF`），使用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/run_low_resource_matrix.sh

# 汇总结果
python -m src.summarize_low_resource \
  --input-root outputs/low_resource_matrix \
  --output-dir outputs/low_resource_matrix
```

说明：
- low-resource 比例基于配置文件里的 `train_samples` 自动推导。当前默认配置是 `1000`，因此会实际运行 `100 / 300 / 500 / 1000` 个训练样本。
- 结果会分别写到 `outputs/low_resource_matrix/<ratio>/<config>/`。
- 汇总表会生成到 `outputs/low_resource_matrix/low_resource_summary.{csv,md}`。

## 9. 运行 ablation

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/run_ablation.sh
```

## 10. 输出文件位置

| 文件 | 说明 |
|------|------|
| `data/processed/` | 预处理后的数据集 |
| `outputs/generated_candidates/candidates.jsonl` | LLM 生成的候选样本 |
| `outputs/checkpoints/verifications.jsonl` | Verifier 打分结果 |
| `outputs/selected_counterfactuals/selected.jsonl` | Selector 筛选后的样本 |
| `outputs/tables/main_results.csv` | 主实验结果表（`SST-2 Acc/F1` 为 test 集指标，额外包含 validation 指标） |
| `outputs/ood_sst2_*/tables/ood_results.csv` | OOD 结果表（不同 OOD 数据集：Rotten / Yelp / Amazon） |
| `outputs/ood_sst2_*/tables/quality_results.csv` | OOD 质量表（由 `verifications.jsonl` 汇总生成） |
| `outputs/tables/quality_results.csv` | 自动质量评估表 |
| `outputs/logs/llm_calls.jsonl` | LLM 调用日志 |

## 11. 如何复现实验表格

```bash
# Step 1: 跑 baseline
HF_ENDPOINT=https://hf-mirror.com python -m src.models.train --config configs/baseline.yaml

# Step 2: 跑 AgentCF
HF_ENDPOINT=https://hf-mirror.com python -m src.models.train --config configs/agentcf.yaml

# Step 3: 跑 OOD
HF_ENDPOINT=https://hf-mirror.com OPENAI_BASE_URL=http://localhost:8000/v1 \
  python -m src.run_ood --config configs/ood_sst2_yelp_agentcf.yaml

# 结果读取
cat outputs/tables/main_results.csv
cat outputs/tables/quality_results.csv
cat outputs/ood_sst2_yelp_agentcf_v1/tables/ood_results.csv
cat outputs/ood_sst2_yelp_agentcf_v1/tables/quality_results.csv
```

说明：
- `main_results.csv` 中的 `SST-2 Acc` / `SST-2 F1` 是最终 `test` 集结果。
- `Validation Acc` / `Validation F1` 仅用于记录训练时的开发集表现，不应替代最终测试结果。
- `ood_results.csv` 中的 `ID Acc/F1` 是 ID 集结果，`OOD Acc/F1` 是对应 OOD 数据集结果，`Robustness Gap = ID Acc - OOD Acc`。

## 12. 注意事项

- **显存管理**：vLLM（7B fp16）占用约 14GB，DistilBERT 训练额外需要约 1GB。若显存不足，LLM 生成和模型训练需分开进行（先生成后关 vLLM 再训练）
- **阈值调整**：`configs/default.yaml` 中的 `thresholds` 根据实际分数分布调整，当前设置适配 Qwen2.5-7B-Instruct
- **并发控制**：`agentcf_pipeline.py` 默认 50 路并发，`single_cf.py` 默认 20 路并发，可根据网络和 GPU 情况调整
- **断点续跑**：`src/run_ood.py` 会优先检测 `selected_counterfactuals/selected.jsonl` 和 `checkpoints/verifications.jsonl`，已有 checkpoint 时会跳过重复生成
