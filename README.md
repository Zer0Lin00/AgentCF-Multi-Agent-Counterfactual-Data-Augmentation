# AgentCF 实验项目

本项目实现了 `AgentCF: Multi-Agent Counterfactual Data Augmentation` 的最小可运行版本，覆盖：
- SST-2 主任务
- 4 个基线与 AgentCF 主流程
- 低资源与消融脚本
- 自动质量评估与结果表导出

## 1. 环境依赖

```bash
pip install -r requirements.txt
```

## 2. 数据下载方式

首次运行会自动通过 HuggingFace `datasets` 下载：
- `glue/sst2`
- `imdb`

## 3. 运行 Baseline

```bash
bash scripts/run_baseline.sh
```

或：

```bash
python -m src.models.train --config configs/baseline.yaml
```

## 4. 运行 AgentCF

```bash
bash scripts/run_agentcf.sh
```

或：

```bash
python -m src.models.train --config configs/agentcf.yaml
```

## 5. 运行 low-resource 实验

```bash
bash scripts/run_low_resource.sh
```

## 6. 运行 ablation

```bash
bash scripts/run_ablation.sh
```

## 7. 输出文件位置

- 数据处理结果：`data/processed/`
- 候选生成：`outputs/generated_candidates/candidates.jsonl`
- 验证打分：`outputs/checkpoints/verifications.jsonl`
- 最终筛选：`outputs/selected_counterfactuals/selected.jsonl`
- 主结果表：`outputs/tables/main_results.csv`
- 自动质量表：`outputs/tables/quality_results.csv`

## 8. 如何复现实验表格

1. 跑主实验：`python -m src.models.train --config configs/default.yaml`
2. 结果直接读取：
   - `outputs/tables/main_results.csv`
   - `outputs/tables/quality_results.csv`
3. 可将 CSV 复制进论文模板表格（17.1/17.3）。

## 9. LLM 配置

通过 `.env` 控制：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `PLANNER_MODEL`
- `GENERATOR_MODEL`
- `VERIFIER_MODEL`

当 API 不可用时，系统会回退到规则版 agent，保证流程可跑。
