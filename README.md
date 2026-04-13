# AgentCF 实验项目

本项目实现了 `AgentCF: Multi-Agent Counterfactual Data Augmentation` 的可运行版本，覆盖：
- SST-2 主任务
- 4 个基线与 AgentCF 主流程
- 低资源与消融脚本
- 自动质量评估与结果表导出

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

## 7. 运行 low-resource 实验

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

## 8. 运行 ablation

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/run_ablation.sh
```

## 9. 输出文件位置

| 文件 | 说明 |
|------|------|
| `data/processed/` | 预处理后的数据集 |
| `outputs/generated_candidates/candidates.jsonl` | LLM 生成的候选样本 |
| `outputs/checkpoints/verifications.jsonl` | Verifier 打分结果 |
| `outputs/selected_counterfactuals/selected.jsonl` | Selector 筛选后的样本 |
| `outputs/tables/main_results.csv` | 主实验结果表（`SST-2 Acc/F1` 为 test 集指标，额外包含 validation 指标） |
| `outputs/tables/quality_results.csv` | 自动质量评估表 |
| `outputs/logs/llm_calls.jsonl` | LLM 调用日志 |

## 10. 如何复现实验表格

```bash
# Step 1: 跑 baseline
HF_ENDPOINT=https://hf-mirror.com python -m src.models.train --config configs/baseline.yaml

# Step 2: 跑 AgentCF
HF_ENDPOINT=https://hf-mirror.com python -m src.models.train --config configs/agentcf.yaml

# 结果读取
cat outputs/tables/main_results.csv
cat outputs/tables/quality_results.csv
```

说明：
- `main_results.csv` 中的 `SST-2 Acc` / `SST-2 F1` 是最终 `test` 集结果。
- `Validation Acc` / `Validation F1` 仅用于记录训练时的开发集表现，不应替代最终测试结果。

## 11. 注意事项

- **显存管理**：vLLM（7B fp16）占用约 14GB，DistilBERT 训练额外需要约 1GB。若显存不足，LLM 生成和模型训练需分开进行（先生成后关 vLLM 再训练）
- **阈值调整**：`configs/default.yaml` 中的 `thresholds` 根据实际分数分布调整，当前设置适配 Qwen2.5-7B-Instruct
- **并发控制**：`agentcf_pipeline.py` 默认 50 路并发，`single_cf.py` 默认 20 路并发，可根据网络和 GPU 情况调整
