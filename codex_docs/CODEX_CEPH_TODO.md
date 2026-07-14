# Heat Predictor TODO

当前 baseline 为未来10秒新增热度标签、H0 固定热阈值 EMA `0.10` 和 P0 固定预测阈值
`0.50`。未来只研究新增热度阈值来源以及 H1/P1 动态阈值；feature、森林参数、训练权重、
并发和一般性能优化不再列入 TODO，热、冷训练样本权重固定为 `1.0`。

当前热度、访问间隔、5秒/10秒访问窗口、Otsu 时间环和 EQ 标签均使用单调时间。H0 的
`0.10` 是每1秒参考区间的 EMA gain；空闲窗口清理和 iterator reservation 已进入
baseline，不再作为 TODO。`awaiting_prediction` 正常应为0，暂不增加超时或独立上限。

## 测试约束

- 参数以 `src/heatpredictor/hp_config.h` 为准。
- 五个正式负载为 MapReduce、GraphChi、HPC、AI training、AI inference；均使用
  `4 MiB` 纯读，速率以各负载当前配置为准。
- 每个对照实验只改变目标维度；feature、25 棵树、`max_features=NUM_FEATURES`、
  EQ 10 秒和数据集必须保持一致。
- 每个 profile 的每个负载只运行一次。巨大误差、计数异常或与历史冲突只写入中文
  报告，是否复测由用户决定。
- 每轮开始前 reset 并确认 `active+clean`；结束后等待各 OSD 训练队列清零，只保存一次
  MGR `hp_status.json`。
- `hp_train_drop_count` 必须为 0。报告保留 accuracy、balanced accuracy、precision、
  recall、预测/实际热比例，并记录真实起止时间。
- 报告放在 `/home/chris/ceph-test/new_workload/hp_runs/reports/` 的带日期目录中。

## 当前 baseline 与候选

| 维度 | 固定方案 | 动态方案 |
| --- | --- | --- |
| 热阈值 EMA | H0：固定 `0.10` | H1：`0.50 * confidence` |
| 预测阈值 | P0：固定 `0.50` | P1：监督概率直方图动态阈值 |

当前代码固定为 H0P0；H1/P1 只在明确发起后续实验时重新启用。

## 新增热度阈值

当前标签固定使用
`max(0, decayed_total_heat_at_deadline - decayed_entry_heat_at_deadline)`，目标是预测当前
I/O 之后10秒内是否产生足够的新热访问。当前 Otsu 在最近60秒内按 object 保留最新一条
已完成窗口的 `future_window_added_heat`：10000个固定 `log1p` bin、宽度 `0.01`，并由
60个一秒时间槽负责淘汰。后续比较以下投票口径：

- 当前 baseline：每个 object 只用最新完成窗口的新增热度投一票；后续同一 object 的
  新结果以 O(1) 替换旧票，避免 I/O 频率对热门 object 二次加权；
- 原始对照：每个已完成 EQ I/O 向新增热度直方图投一票，用于判断按 object 去重是否
  丢失访问强度信息；
- 保留“object 总热度 Otsu、热度下限 `10`”作为对照。该方案将超过一个10秒标签窗口
  未访问的热度压入冷端，可用于量化长期冷 object 对阈值的影响；但它与“未来10秒新增
  热度”标签口径不同，不作为默认最终方案。

待执行顺序：

- [x] 实现并用算法探针验证每 object 最新 `future_window_added_heat` 投票、替换、乱序
  拒绝和60秒淘汰。
- [ ] 比较当前每 object 最新一票和原始每 I/O 一票两种新增热度口径。
- [ ] 运行“object 总热度 Otsu、下限 `10`”对照，并检查 accuracy、precision、recall
  以及预测/实际热比例。
- [ ] 根据同一组五负载单轮结果确定热阈值数据源，再决定是否重新启用 H1/P1。

报告需保留 accuracy、balanced accuracy、precision、recall、预测/实际热比例、未来
访问次数、新增热度、训练样本数、Otsu 保留 object 数以及 EQ/训练 drop。Otsu 数据源对照
完成前不能把标签比例异常直接归因于模型。

详细定义：

- [OTSU_THRESHOLD.md](todo/OTSU_THRESHOLD.md)：H0/H1 热阈值跟踪。
- [PREDICTION_THRESHOLD.md](todo/PREDICTION_THRESHOLD.md)：P0/P1 预测阈值。
- [FINAL_VALIDATION.md](todo/FINAL_VALIDATION.md)：H0/H1 × P0/P1 的 2×2 测试矩阵。

执行规范见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，但实验器本身不再作为
功能 TODO。H1/P1 的历史 2×2 已完成，重新启用动态方案或切换 Otsu 数据源前必须先通过 L1/L2，
再按单轮 L3 规则生成中文报告。
