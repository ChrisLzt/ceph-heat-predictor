# Heat Predictor TODO

当前研究实际热标签阈值、模型预测阈值和时间型评价窗口。feature、森林参数、训练权重、
并发和一般性能优化不再列入 TODO；热、冷训练样本权重固定为 `1.0`。

## 测试约束

- 参数以 `src/heatpredictor/hp_config.h` 为准。
- 四个正式负载为 MapReduce、GraphChi、AI training、AI inference；均使用 `1 MiB`
  纯读和 `fwdrate=1000`，不运行 HPC。
- 除两个阈值策略外，feature、25 棵树、`max_features=NUM_FEATURES`、EQ 10000 和数据集
  必须保持一致。
- 每个 profile 的每个负载只运行一次。巨大误差、计数异常或与历史冲突只写入中文
  报告，是否复测由用户决定。
- 每轮开始前 reset 并确认 `active+clean`；结束后等待各 OSD 训练队列清零，只保存一次
  MGR `hp_status.json`。
- `hp_train_drop_count` 必须为 0。报告保留 accuracy、precision、recall、预测/实际热
  比例，并记录真实起止时间。
- 报告放在 `/home/chris/ceph-test/new_workload/hp_runs/reports/` 的带日期目录中。

## 当前候选

| 维度 | 固定方案 | 动态方案 |
| --- | --- | --- |
| 热阈值 EMA | H0：固定 `0.10` | H1：`0.50 * confidence` |
| 预测阈值 | P0：固定 `0.50` | P1：监督概率直方图动态阈值 |

详细定义：

- [OTSU_THRESHOLD.md](todo/OTSU_THRESHOLD.md)：H0/H1 热阈值跟踪。
- [PREDICTION_THRESHOLD.md](todo/PREDICTION_THRESHOLD.md)：P0/P1 预测阈值。
- [FINAL_VALIDATION.md](todo/FINAL_VALIDATION.md)：H0/H1 × P0/P1 的 2×2 测试矩阵。
- [TIME_BASED_EQ.md](todo/TIME_BASED_EQ.md)：将固定 I/O 数评价窗口改为固定时间窗口。

执行规范见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，但实验器本身不再作为
功能 TODO。执行顺序为：L1 profile 检查 -> L2 部署检查 -> 2×2 L3 单轮矩阵 -> 中文报告。

时间型 EQ 是独立的后续改动。阈值矩阵完成前不实施，实施时以
`eq-io-window-baseline` 分支作为按 I/O 窗口的对照版本。
