# 热阈值与 Otsu TODO

热阈值根据未来窗口热度生成实际冷热标签。本文件只比较同一个 Otsu 候选的有效阈值
跟踪速度，其他算法和类别权重不进入本轮矩阵。

## 共同算法

- 当前 Otsu 使用最近60秒内每个 object 最新完成窗口的 `future_window_added_heat`，经
  `log1p` 后进入10000个固定 bin，bin 宽 `0.01`；60个一秒时间槽负责淘汰。
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

两种方案使用相同 histogram、候选阈值和60秒投票生命周期。H0 用于判断固定平滑是否比
confidence 调速更稳定；H1 用于提高热点变化时的跟踪速度。实际 gain 按距离上次有效
更新的单调时间复合，避免 OSD IOPS 改变阈值的墙钟响应速度。

## 数据源对照（待执行）

- 当前 baseline：每个 object 保留最新完成的未来10秒新增热度；标签量相同，但 Otsu
  不按同一 object 的 I/O 次数重复加权。
- 原始对照：每个已完成 EQ I/O 投一票，用于测试访问频率加权是否比 object 等权更好。
- 控制组：继续使用 object 总热度，但把 Otsu 热度下限从 `1` 提高到 `10`。该下限对应
  初始热度 `100` 在10秒后保留 `1/10`；它只用于衡量长期冷 object 堆积的影响，不解决
  总热度阈值与新增热度标签之间的口径差异。

数据源确定前保持 H0 固定 EMA `0.10` 和 P0 固定预测阈值 `0.50`，避免同时改变多个
维度。数据源确定后再决定是否需要重新比较 H1/P1。

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
