# Heat Predictor TODO

本文只记录尚未完成的工作。当前实现见 [实现说明](CODEX_CEPH.md)，分支规则见
[Main/Dev 流程](BRANCH_WORKFLOW.md)。

## 热预测比例偏低

最新五负载测试的宏平均预测热比例为 `43.18%`，实际热比例为 `47.41%`；
`Precision=93.76%`、`Recall=85.04%`，合计 `FN=102,050`、`FP=40,451`。
GraphChi 和 AI 推理分别少预测 `6.30`、`5.91` 个百分点。问题主要是热样本漏判，
不是预测结果整体过热。

保持以下基线不变：

- 标签仍为未来10秒访问数是否达到 deadline 时的 `K_window`。
- ARF 判热阈值固定为 `0.50`，冷热训练权重均为 `1.0`。
- 不以强制匹配预测/实际热比例为目标，也不直接降低阈值或提高热类权重。

### P1：强证据热样本补判

仅对 ARF 判冷的样本检查预测时已有的2秒访问证据：

```text
required_short_count =
    max(3, ceil(K_context * short_window / future_window))
  = max(3, ceil(K_context / 5))

strong_hot_evidence =
    short_2s_access_count >= required_short_count
```

该条件表示过去2秒的访问速率投影到未来10秒后足以达到当前 `K_context`，且至少已有
3次访问，避免一次随机访问直接触发热预测。最终候选判决为：

```text
predicted_hot =
    arf_hot_probability >= 0.50
    or strong_hot_evidence
```

先使用现有 Trace 按时间顺序做因果回放。补判只能读取预测时已经存在的
`short_2s_access_count` 和 `K_context`，不得读取未来访问数、`K_window` 或实际标签。
报告每种负载纠正的 FN、引入的 FP、净新增正确数、Accuracy、Balanced Accuracy、
Precision、Recall 和预测/实际热比例。

进入在线实现前必须同时满足：

- 五种负载的 `rescued_FN - introduced_FP` 均不小于0，合计值大于0。
- 合并 Accuracy 上升，单项 Accuracy 下降不超过 `0.10` 个百分点。
- 宏平均 Recall 上升至少 `1.00` 个百分点，Precision 下降不超过 `0.50` 个百分点。
- 预测热比例与实际热比例的宏平均绝对差缩小。

在线实现只修改最终预测判决，不改变训练标签、样本权重或 ARF 概率。Trace 需要保留
原始概率和是否发生强证据补判，避免把规则输出误解释为 ARF 概率超过 `0.50`。
完成算法探针、Trace/replay、五负载各一次测试后，再决定是否发布。

### P2：因果概率校准备选

只有 P1 未通过时才评估概率校准。使用已完成标签维护有界的 raw-probability
可靠性直方图，以历史样本估计校准概率，最终仍按 `0.50` 判热。校准器不得读取未到期
标签，必须 reset 清空，并需要单独报告 Brier score、ECE 和各概率区间的可靠性。

该方案状态更多，实际效果接近动态决策边界，因此不与 P1 同时实施，也不作为当前
首选。
