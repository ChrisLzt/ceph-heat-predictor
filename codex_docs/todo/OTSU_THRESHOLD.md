# 热阈值与 Otsu TODO

热阈值根据未来窗口热度生成实际冷热标签。本文件只比较同一个 Otsu 候选的有效阈值
跟踪速度，其他算法和类别权重不进入本轮矩阵。

## 共同算法

- Otsu 使用时间归一化 log-heat 直方图，热度范围 `1~3000`，bin 宽 `0.05`。
- 至少 `32` 个 object 且存在两个非空 bin、总方差大于 0 时才生成候选。
- `separation` 是最优类间方差与总方差之比。
- `sharpness` 按达到最优类间方差 `99%` 的候选之间会改变分类的 object 比例计算；
  歧义比例达到 `0.20` 时归零，宽空谷不会降低 sharpness。
- `confidence = separation^0.80 * sharpness^0.20`，不包含 sample confidence。

## 候选

| Profile | 有效热阈值更新 |
| --- | --- |
| H0 fixed EMA | `effective += 0.10 * (candidate - effective)` |
| H1 dynamic EMA | `effective += 0.50 * confidence * (candidate - effective)` |

两种方案使用相同 histogram、候选阈值和 object 生命周期。H0 用于判断固定平滑是否比
confidence 调速更稳定；H1 用于提高热点变化时的跟踪速度。

## 待执行

- [x] 探针验证宽空谷的歧义样本数为 0、单调分布存在正歧义样本数，H0/H1 均不会在
  单次更新中越过候选。
- [x] 在 P0 和 P1 下分别比较 H0/H1，记录候选/有效热阈值、confidence、状态和实际热
  比例。
- [x] 比较负载结束时的 candidate-to-effective gap：H1 明显缩小 gap，但没有提升整体
  accuracy。
- [x] 检查 H1 的结束状态和正式指标；未发现吞吐异常，不自动复测。

本轮只保存结束状态，不能从时间序列判断阈值反向移动响应或短期震荡。完整结果见：

`/home/chris/ceph-test/new_workload/hp_runs/reports/20260713_014808_threshold_2x2/REPORT.md`

旧 O1/O2 报告位于：

`/home/chris/ceph-test/new_workload/hp_runs/reports/20260712_otsu_profiles/`
