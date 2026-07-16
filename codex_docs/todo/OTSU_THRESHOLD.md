# 热阈值与 Otsu TODO

热阈值根据 object 总热度生成实际冷热标签。本文件只比较同一个 Otsu 候选的有效阈值
跟踪速度，其他算法和类别权重不进入本轮矩阵。

## 共同算法

- 当前 D2 Otsu 为每个 object 保留最新总热度 score：
  `score = ln(clamp(heat, 10, 29809.58)) - decay_factor * timestamp`。
  score 进入800个固定 bin，bin 宽 `0.01`；生命周期由 object 替换和100万 object
  阈值窗口容量决定，不使用60秒时间环。
- 至少 `32` 个 object 投票且存在两个非空 bin、总方差大于 0 时才生成候选。
- `separation` 是最优类间方差与总方差之比。
- `sharpness` 按达到最优类间方差 `99%` 的候选之间会改变分类的样本比例计算；
  歧义比例达到 `0.20` 时归零，宽空谷不会降低 sharpness。
- `confidence = separation^0.80 * sharpness^0.20`，不包含 sample confidence。

## 候选

| Profile | 有效热阈值更新 |
| --- | --- |
| H0 fixed EMA | 每1秒参考区间 `effective += 0.10 * (candidate - effective)` |
| H1 dynamic EMA | 每1秒参考区间 `effective += 0.50 * confidence * (candidate - effective)` |

两种方案使用相同 score histogram、候选阈值和 object 生命周期。H0 用于判断固定平滑是否比
confidence 调速更稳定；H1 用于提高热点变化时的跟踪速度。实际 gain 按距离上次有效
更新的单调时间复合，避免 OSD IOPS 改变阈值的墙钟响应速度。

## 数据源对照（已完成）

- D0：每个 object 保留最新完成的未来10秒新增热度；
- D1：每个已完成 EQ I/O 按未来新增热度投票；
- D2：每个 object 保留最新总热度 score。

同一组五负载单轮对照中 D2 最优，当前默认使用 D2，并进一步把 feature 和10秒标签
统一为总热度。对齐后的单轮五负载平均 accuracy/balanced accuracy 为
`81.11%/80.18%`，没有 EQ、训练 drop 或预测错误。保持 H0 固定 EMA `0.10` 和 P0
固定预测阈值 `0.50`，避免同时改变多个维度；是否重新比较 H1/P1 由后续明确实验决定。

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
