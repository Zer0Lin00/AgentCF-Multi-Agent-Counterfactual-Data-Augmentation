# Repeated Experiment Analysis (Bilingual)

## 1. Overall Takeaways / 总体结论

- AgentCF achieves the best mean SST-2 accuracy in the main repeated experiment: **86.16 ± 0.46**.
- AgentCF outperforms No Augmentation by **+2.56** accuracy points and Standard Augmentation by **+1.26** points.
- In low-resource settings, AgentCF is strongest at **10%**, **50%**, and **100%** data, but it is **not** the best method at **30%** data.
- On OOD evaluation from SST-2 to IMDb, AgentCF keeps the best in-domain accuracy, but it does **not** achieve the best OOD accuracy and shows the **largest robustness gap**.
- In ablation, removing the planner causes the clearest drop, and **full AgentCF remains the best reference variant** under the same reporting convention as the main experiment.

- AgentCF 在主实验三次重复中的平均准确率最高，为 **86.16 ± 0.46**。
- 相比 No Augmentation，AgentCF 提升 **2.56** 个百分点；相比 Standard Augmentation，提升 **1.26** 个百分点。
- 在低资源实验中，AgentCF 在 **10%**、**50%**、**100%** 数据量下表现最好，但在 **30%** 数据量下**不是**最优方法。
- 在 SST-2 训练、IMDb 测试的 OOD 评估中，AgentCF 的域内精度最高，但 **OOD 精度不是最高**，且 **robustness gap 最大**。
- 在消融实验中，去掉 planner 的性能下降最明显；按与主实验一致的统计口径，**full AgentCF 仍然是最佳参考变体**。

## 2. Main Results / 主实验

| Method | Acc mean | Acc std | F1 mean | F1 std |
| --- | ---: | ---: | ---: | ---: |
| AgentCF (Ours) | 86.16 | 0.46 | 86.14 | 0.46 |
| Standard Augmentation | 84.90 | 0.35 | 84.89 | 0.34 |
| Single-LLM + Filtering | 84.29 | 0.20 | 84.27 | 0.18 |
| Single-LLM Counterfactual | 84.02 | 0.98 | 84.02 | 0.97 |
| No Augmentation | 83.60 | 0.87 | 83.60 | 0.87 |

Key reading:

- AgentCF is the best method in the main benchmark by a clear margin.
- The gain over Standard Augmentation is meaningful and stable across three runs.
- Single-LLM based baselines improve over No Augmentation only modestly.

解读：

- AgentCF 在主基准上是最优方法，而且领先幅度比较明确。
- 相比 Standard Augmentation，增益是稳定存在的，不是单次随机波动。
- 两个 Single-LLM 基线相对 No Augmentation 只有有限提升。

## 3. Quality Evaluation / 质量评估

| Method | Label Success | Semantic Sim | Edit Similarity |
| --- | ---: | ---: | ---: |
| Single-LLM + Filtering | 98.47 | 82.16 | 72.17 |
| AgentCF (Ours) | 43.65 | 61.47 | 57.50 |
| Single-LLM Counterfactual | 40.21 | 60.72 | 56.45 |

Key reading:

- AgentCF is better than Single-LLM Counterfactual on all three quality metrics, but only slightly.
- Single-LLM + Filtering dominates the automatic quality metrics by a very large margin.
- Therefore, the accuracy gain of AgentCF does **not** come from looking better under these three automatic quality indicators.

解读：

- AgentCF 相比 Single-LLM Counterfactual 的质量指标是全面更好的，但优势不算大。
- Single-LLM + Filtering 在三个自动质量指标上明显占优。
- 这说明 AgentCF 的分类性能提升，并不是简单来自这三项自动指标上的“更高质量”。

## 4. Low-Resource Results / 低资源实验

| Ratio | Best Method | Best Acc | AgentCF Acc |
| --- | --- | ---: | ---: |
| 10% | AgentCF (Ours) | 60.82 | 60.82 |
| 30% | Standard Augmentation | 81.99 | 80.47 |
| 50% | AgentCF (Ours) | 83.26 | 83.26 |
| 100% | AgentCF (Ours) | 85.63 | 85.63 |

Key reading:

- AgentCF is especially useful at **10%** data, where it exceeds No Augmentation by **+5.32** points.
- At **50%** and **100%**, AgentCF remains the top method.
- At **30%**, Standard Augmentation is better than AgentCF by about **1.53** points, so low-resource superiority is not universal.

解读：

- 在 **10%** 数据量下，AgentCF 的价值最明显，相比 No Augmentation 提升 **5.32** 个百分点。
- 在 **50%** 和 **100%** 数据量下，AgentCF 仍然是最优方法。
- 但在 **30%** 数据量下，Standard Augmentation 比 AgentCF 高约 **1.53** 个点，因此不能把低资源优势表述为“全区间成立”。

## 5. OOD Results / OOD 结果

| Method | ID Acc | OOD Acc | Robustness Gap |
| --- | ---: | ---: | ---: |
| AgentCF (Ours) | 86.16 | 77.93 | 8.24 |
| Single-LLM + Filtering | 84.60 | 78.46 | 6.14 |
| Standard Augmentation | 84.90 | 78.13 | 6.77 |
| No Augmentation | 83.60 | 77.91 | 5.69 |
| Single-LLM Counterfactual | 84.63 | 76.98 | 7.65 |

Key reading:

- AgentCF has the highest in-domain accuracy.
- However, AgentCF does not achieve the best OOD accuracy; Single-LLM + Filtering is slightly higher on IMDb.
- AgentCF also has the largest robustness gap, so the current method improves ID performance more than cross-domain robustness.

解读：

- AgentCF 的域内精度最高。
- 但在 IMDb 上，AgentCF 不是 OOD 最优；Single-LLM + Filtering 的 OOD 精度略高。
- 同时 AgentCF 的 robustness gap 最大，说明当前版本更像是在提升域内性能，而不是显著提升跨域鲁棒性。

## 6. Ablation Results / 消融实验

| Variant | Acc mean | Label Success |
| --- | ---: | ---: |
| full_agentcf | 86.16 | 43.73 |
| single_agent_version | 85.67 | 44.02 |
| w_o_dynamic_threshold | 85.36 | 43.93 |
| w_o_verifier_feedback | 85.21 | 47.64 |
| w_o_selector | 85.17 | 44.18 |
| w_o_planner | 84.56 | 62.88 |

Key reading:

- Removing the planner causes the largest drop relative to full AgentCF: about **-1.61** points.
- The selector and verifier feedback matter, but their removal hurts less.
- The single-agent version is below full AgentCF by about **-0.50** points.
- `w/o planner` has a much higher label success rate but lower classification accuracy, which suggests that higher automatic label flipping alone does not guarantee better downstream utility.

解读：

- 去掉 planner 带来的性能下降最大，相比 full AgentCF 约 **-1.61** 个点。
- selector 和 verifier feedback 也有作用，但影响没有 planner 明显。
- `single_agent_version` 相比 full AgentCF 低约 **0.50** 个点。
- `w/o planner` 的 label success rate 很高，但最终分类性能更差，说明“更容易翻标签”并不等价于“更有用的增强样本”。

## 7. Suggested Paper Wording / 论文表述建议

Recommended safe claims:

- AgentCF consistently improves the main SST-2 benchmark over all comparison methods.
- AgentCF is particularly effective in very low-resource settings such as 10% training data.
- The current AgentCF implementation improves in-domain accuracy, but its OOD robustness remains limited.
- Ablation shows the planner is important, and full AgentCF remains stronger than the tested simplified single-agent variant under the aligned reporting convention.

建议在论文中使用的稳妥表述：

- AgentCF 在主实验 SST-2 上稳定优于所有对比方法。
- AgentCF 在极低资源场景，尤其是 10% 数据量下，表现出明显优势。
- 当前实现主要提升的是域内性能，OOD 鲁棒性提升仍然有限。
- 消融实验表明 planner 是重要模块；在与主实验对齐的统计口径下，full AgentCF 也优于当前测试的 single-agent 简化版本。
