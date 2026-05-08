# 实验进度记录

## 已完成的改进

### 1. Consistency Score 修复
- 文件：`src/agents/verifier.py`
- 改动：`_consistency_score` 从字符串 `in` 匹配改为 embedding cosine similarity
- `elements_to_preserve` 为空时 fallback 到原文与候选的语义相似度（而非硬编码 0.8）

### 2. 引入样本难度指标
- 文件：`src/agents/verifier.py`、`src/metrics/quality_score.py`
- 改动：新增 `_difficulty_score` 方法，分类器置信度越接近 0.5 分数越高
- `verify()` 返回值新增 `difficulty_score` 字段
- `final_quality_score` 新增 `difficulty_score` 参数

### 3. Generator Prompt 改进
- 文件：`prompts/generator_prompt.txt`
- 改动：明确要求只修改 `spurious_features` 和 `elements_to_change`，禁止修改 `causal_features`

### 4. 样本间并发处理
- 文件：`src/augmentation/agentcf_pipeline.py`
- 改动：用 `asyncio.Semaphore` + `asyncio.as_completed` 实现样本级并发（默认 10 路）

### 5. 配置更新
- 文件：`configs/default.yaml`
- 改动：加入 `difficulty_score: 0.10` 权重，加入 `concurrency: 10`

### 6. OOD 评测统计
- 文件：`src/run_ood.py`
- 改动：quality table 中加入 `difficulty_score` 统计列

### 7. 缺失模块补全
- 新建：`src/data/__init__.py`、`src/data/load_data.py`、`src/data/preprocess.py`
- SST-2 使用本地 glue 缓存（`/root/.cache/huggingface/datasets/glue/sst2`）
- IMDb 网络不通时返回空 DataFrame，不影响主实验

### 8. LLM 客户端修复
- 文件：`src/utils/llm.py`
- 改动：`AsyncOpenAI` 改为懒加载，`is_closed()` 时自动重建，解决并发场景下客户端被关闭的问题

---

## 当前实验状态

- vLLM 已安装（v0.19.1），使用 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`
- 实验正在运行中（`python -m src.models.train --config configs/agentcf.yaml`）
- 断点续传已支持，中断后重跑会跳过已完成的 LLM 生成

### 运行命令
```bash
cd /root/autodl-tmp/AgentCF-Multi-Agent-Counterfactual-Data-Augmentation

# 终端1：启动 vLLM
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --port 8000

# 终端2：运行实验
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export PLANNER_MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export GENERATOR_MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
python -m src.models.train --config configs/agentcf.yaml
```

---

## 待完成的工作

### 高优先级

1. **验证实验结果**
   - 确认 `outputs/tables/main_results.csv` 中 AgentCF 的 Acc 是否优于基线
   - 确认 `difficulty_score` 字段出现在 `outputs/checkpoints/verifications.jsonl` 中
   - 对比改进前后的 `final_score` 分布变化

2. **下载 IMDb 数据集并跑 OOD 评测**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python -c "from datasets import load_dataset; load_dataset('imdb'); print('done')"
   python -m src.run_ood --config configs/agentcf.yaml
   ```
   - 查看 `outputs/tables/ood_results.csv` 中的 Robustness Gap

3. **消融实验**
   ```bash
   bash scripts/run_ablation.sh
   ```
   - 验证 difficulty_score 和新 consistency_score 对结果的贡献

### 中优先级

4. **Standard Augmentation 基线增强**
   - 当前 `src/augmentation/standard_aug.py` 只有 12 词的同义词词典，过于简单
   - 可考虑用 `nlpaug` 或 `back-translation` 替代

5. **低资源实验稳定性**
   - 10% 数据时 std=0.075 过高，建议增加重复次数从 3 次到 5 次
   - 修改 `scripts/run_low_resource.sh` 中的 `--repeats` 参数

6. **Planner prompt 优化**
   - 当前 `prompts/planner_prompt.txt` 对 `spurious_features` 的定义不够清晰
   - 可加入 few-shot 示例帮助模型更准确地区分因果/伪相关特征

### 低优先级

7. **完整重复实验**（论文级别）
   ```bash
   bash scripts/run_repeated_suite.sh
   ```
   - 3次重复取均值，结果才有统计意义

8. **train_samples 扩大**
   - 当前 `default.yaml` 中 `train_samples: 1000`，正式实验应用完整 SST-2（67k）
   - 修改 `configs/default.yaml`：`train_samples: 0`（0 表示使用全量）
