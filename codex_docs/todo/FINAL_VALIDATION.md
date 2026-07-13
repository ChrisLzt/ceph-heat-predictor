# 阈值 2×2 测试 TODO

下一轮只测试热阈值 EMA 和预测阈值两个维度。

## Profile 矩阵

| Profile | 热阈值 | 预测阈值 |
| --- | --- | --- |
| H0P0 | 固定 EMA `0.10` | 固定 `0.50` |
| H0P1 | 固定 EMA `0.10` | 监督直方图动态阈值 |
| H1P0 | `0.50 * confidence` | 固定 `0.50` |
| H1P1 | `0.50 * confidence` | 监督直方图动态阈值 |

每个 profile 依次运行 MapReduce、GraphChi、AI training、AI inference，各一次，共
`4 × 4 = 16` 项正式负载。H0P0 是最简单的固定基线；H1P1 是当前完整动态方案。

## 执行与报告

- [x] 四个 profile 先通过 `hp_algorithm_probe` 和编译参数检查。
- [x] 每轮使用相同代码、数据集、负载顺序和 `fwdrate=1000`，开始前 reset。
- [x] 保存每轮最终 `hp_status.json` 和 Vdbench 输出，不保存 before status。
- [x] 表格记录 accuracy、precision、recall、预测/实际热比例、候选/有效热阈值、
  confidence、当前/候选预测阈值、训练 drop、rate 和 MB/s。
- [x] 先比较 H0P0/H0P1 和 H1P0/H1P1，判断预测阈值主效应；再比较 H0P0/H1P0 和
  H0P1/H1P1，判断热阈值 EMA 主效应。
- [x] 同时报告热阈值与预测阈值的交互，不用不同 actual-hot 比例下的 raw accuracy
  单独推断模型判别能力。
- [x] 每项只运行一次。巨大误差或异常写入中文报告，由用户决定是否复测。

## 结果

测试于 2026-07-13 完成，16 项负载均一次通过。H0P0 的宏平均 accuracy 最高，为
`87.0595%`；预测阈值动态化的主效应为 `-0.0599` 个百分点，热阈值 EMA 动态化的主
效应为 `-0.0223` 个百分点，交互效应为 `-0.0101` 个百分点。

完整报告：

`/home/chris/ceph-test/new_workload/hp_runs/reports/20260713_014808_threshold_2x2/REPORT.md`
