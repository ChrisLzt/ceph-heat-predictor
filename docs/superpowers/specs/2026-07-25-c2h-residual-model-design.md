# C2H 监督残差模型设计

## 目标

验证 V2 是否应由“直接 ARF 分类”改为“持续性基础规则 + C2H 监督残差模型”，并
回答残差学习器是否仍需要 ARF。

基础规则固定为：

```text
base_hot =
    past_window_access_count >= future_access_threshold_at_prediction
```

残差模型只处理 `base_hot == false` 的样本，预测未来10秒标签是否实际为热。基础
规则判热时不允许残差模型翻转为冷，本轮不实现 H2C 修正。

## 候选方案

| Profile | 主预测 | C2H残差学习器 |
|---|---|---|
| `B0_persistence` | 持续性规则 | 无 |
| `B1_direct_arf25` | 25树ARF | 无 |
| `R1_c2h_ht` | 持续性规则 | 单棵Hoeffding Tree |
| `R2_c2h_arf10` | 持续性规则 | 10树ARF |
| `R3_c2h_arf25` | 持续性规则 | 25树ARF |

所有学习器使用当前三维 feature、固定 seed、相同事件顺序、相同训练延迟和快照发布
节奏。ARF warning、后台树和 drift replacement 均关闭。残差学习器训练所有
`base_hot == false` 的已完成样本：

```text
target = actual_hot
```

不能只训练原模型的 FN，否则没有“保持冷”的负样本，无法学习何时不应翻转。

## 推理规则

```text
if base_hot:
    final_probability = 1.0
    final_label = hot
else:
    residual_probability = residual_model.predict(features)
    final_probability = residual_probability
    final_label = residual_probability >= correction_threshold
```

未训练模型返回零投票时不翻转基础结果，并记录 cold-start fallback。

`correction_threshold` 候选为 `0.50`、`0.60`、`0.70`。每个负载按时间前向划分：

- `[0s, 120s)`：冷启动，只参与在线训练。
- `[120s, 360s)`：训练区。
- `[360s, 480s)`：validation，只用于选择阈值。
- `[480s, 600s]`：test，只评估一次。

阈值先按五负载 validation 合并 Accuracy 最大选择；并列时依次选择 Balanced
Accuracy 更高、阈值更高的方案，避免无差别扩大热预测。

## 数据与隔离

使用现有五负载 V2 Trace 和 `recent-K` override。Trace 已验证事件顺序、标签和
`past_window_access_count`；所有 profile 必须读取完全相同的记录。残差能力首先只
在 `dev` 的 replay 中实现，不修改生产 `HeatPredictor`、Trace schema、OSD/MGR
接口或 PerfCounter。

## 指标与决策

报告五负载及宏/微汇总：

- Accuracy、Balanced Accuracy、Precision、Recall；
- TP、FP、TN、FN、预测/实际热比例；
- 基础规则判冷样本中的正确翻转、新增 FP 和净修正量；
- 冷启动 fallback、训练样本数、快照数及 replay wall time。

优先级为 Accuracy，其次 Balanced Accuracy。进入在线实现必须同时满足：

1. 相对 `B0_persistence` 的五负载宏平均 Accuracy 为正增益；
2. 至少三种负载 Accuracy 不下降；
3. 全局净修正量 `corrected_FN - introduced_FP > 0`。

ARF只有在残差 ARF 的 Accuracy 明显高于单树，并且收益足以解释额外回放耗时与
在线成本时才保留。若所有残差方案不优于 B0，则不修改生产预测路径。

## 错误处理

- 非法概率、类别数错误或非有限值使该 replay 失败，不静默回退。
- 非 `recent-K` override、Trace/override 数量不一致或序号错位直接拒绝。
- profile、树数量和修正阈值必须显式记录到输出，保证报告可复现。

## 实验结论

五负载独立 test 段结果表明：

- 直接25树ARF相对持续性规则的宏平均 Accuracy 提升 `0.366 pp`；
- 单树、10树ARF、25树ARF残差相对持续性规则分别变化
  `-0.256 pp`、`-0.141 pp`、`-0.103 pp`；
- 最佳残差方案纠正758个FN，同时引入1106个FP，净修正为 `-348`；
- 25树残差ARF相对单树提升 `0.154 pp`，但回放耗时为 `4.38x`。

因此残差ARF虽比单树略强，但“持续性规则 + 只修C2H”的整体架构不成立。本轮不修改
生产预测路径，也不执行在线五负载验证。直接ARF仍有独立价值，因为它能同时学习C2H
和H2C，而残差方案只能增加热预测。
