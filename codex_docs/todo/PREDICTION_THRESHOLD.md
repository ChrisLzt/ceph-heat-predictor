# 预测阈值 TODO

预测阈值把森林输出的 `pred_hot_proba` 转成预测冷热类别。训练权重固定为 `1.0`，本文件
不修改 feature、模型和实际热标签算法。

## 候选

| Profile | 定义 | 编译开关 |
| --- | --- | --- |
| P0 fixed | `pred_hot_proba >= 0.50` 判热 | `HP_ENABLE_PREDICTION_CALIBRATION=0` |
| P1 dynamic | 监督概率直方图在 `0.40~0.60` 中最大化窗口 accuracy | `HP_ENABLE_PREDICTION_CALIBRATION=1` |

P1 使用最近 `10000` 条已评估 I/O，最少 `1000` 条后启用，每 `500` 条更新一次候选，
概率直方图为 `1001` bins，有效阈值使用 `0.10` EMA。相同 accuracy 候选优先选择最
接近当前阈值者。

历史同版本结果中，P0 四负载宏平均 accuracy 为 `87.1063%`，P1 为 `86.9703%`；差异
只有 `0.1360` 个百分点，不能证明任一方案稳定胜出。原始报告位于：

`/home/chris/ceph-test/new_workload/hp_runs/reports/20260712_threshold_weight_regression/`

## 待执行

- [x] 保留 P0/P1 编译 profile，并用探针验证固定阈值不保留校准样本、动态阈值不越过
  `0.40~0.60`。
- [x] 在 H0 和 H1 下分别比较 P0/P1，判断动态预测阈值是否与热阈值更新速度存在交互。
- [x] 记录当前/候选预测阈值、当前/候选窗口 accuracy 和最终负载指标。
- [x] 若 P1 的候选窗口 accuracy 更高但正式负载 accuracy 更低，在报告中标记为校准
  滞后或窗口内过拟合，不自动复测。

2026-07-13 的 2x2 测试中，P1 的 accuracy 主效应为 `-0.0599` 个百分点，预测延迟增加
约 `21.2 us`；候选窗口 accuracy 的小幅提升没有转化为正式负载收益。完整结果见：

`/home/chris/ceph-test/new_workload/hp_runs/reports/20260713_014808_threshold_2x2/REPORT.md`
