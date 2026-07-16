# Heat Predictor TODO

当前 baseline 为10秒结束时 object 总热度标签、H0 固定热阈值 EMA `0.10` 和 P0 固定预测阈值
`0.50`。未来主要研究 D0/D1 新增热度对照以及 H1/P1 动态阈值；另保留一项模型管线
正确性诊断，对比动态 `StandardScaler + ARF` 和直接 `ARF`。其他 feature、森林参数、
训练权重、并发和一般性能优化不再列入 TODO，热、冷训练样本权重固定为 `1.0`。

当前热度、访问间隔、5秒/10秒访问窗口、Otsu score 原点和 EQ 标签均使用单调时间。H0 的
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
- 每轮开始前 reset 并确认 `active+clean`；结束后等待各 OSD 训练队列清零，保存最终
  MGR `hp_status.json`。需要阶段诊断的实验额外每10秒保存一次 MGR 快照。
- `hp_train_drop_count` 必须为 0。报告保留 accuracy、balanced accuracy、precision、
  recall、预测/实际热比例，并记录真实起止时间。
- 报告放在 `/home/chris/ceph-test/new_workload/hp_runs/reports/` 的带日期目录中。

## 当前 baseline 与候选

| 维度 | 固定方案 | 动态方案 |
| --- | --- | --- |
| 热阈值 EMA | H0：固定 `0.10` | H1：`0.50 * confidence` |
| 预测阈值 | P0：固定 `0.50` | P1：监督概率直方图动态阈值 |

当前代码固定为 H0P0；H1/P1 只在明确发起后续实验时重新启用。

## 模型管线诊断

- [x] 对比 S0：当前 `StandardScaler + ARF` 与 S1：直接 `ARF`。
- 动机：在线 scaler 的均值和方差持续变化，但已经写入决策树的历史分裂阈值不会同步
  变换，可能在阶段切换后造成特征坐标漂移。当前特征已经使用 `log2p1` 压缩，树模型
  本身也不要求统一量纲，因此直接 ARF 是合理对照。
- 该实验只改变 scaler，保持特征、标签、Otsu、H0P0、25 棵树、随机种子、数据集和
  负载配置不变。S0/S1 的五个正式负载各运行一次。
- 除常规指标外，每10秒采集一次 MGR 累计混淆矩阵，相邻快照作差后按 Vdbench 的真实
  `Starting RD=` 时间归段。归段时回退10秒标签窗口；报告分别汇总首阶段前30秒冷启动、
  各阶段前30秒热点切换和其余稳定区间，防止最终累计值掩盖 scaler 对概念迁移的影响。
- 验收优先级：balanced accuracy 和 accuracy 均不得下降；若差异小于 `0.2` 个百分点，
  再以预测延迟和实现复杂度决定是否移除 scaler。
- 2026-07-16 单轮五负载结果：S1 平均 accuracy/balanced accuracy 分别下降
  `0.29/0.49` 个百分点，预测延迟仅降低 `0.10%`；冷启动 recall 下降 `12.35` 个百分点，
  稳定阶段基本相当。保留 S0，详见
  [StandardScaler 消融测试报告](../../ceph-test/new_workload/hp_runs/reports/20260716_063127_scaler_s0_s1/REPORT.md)。

## Trace 数据集

- [ ] 增加可显式开关的已完成评估 I/O trace，用于离线复现标签、特征消融和错误样本分析；
  正式 baseline 默认关闭，避免改变延迟结果。
- 每条记录至少包含匿名 object key、预测时单调时间、标签完成时间、预测时 feature、预测
  热概率、预测阈值、预测类别、最终标签、标签热度、标签热阈值和所属负载阶段。
- 明确记录 fallback、EQ drop、预测错误等未进入正常混淆矩阵的原因，正常样本与异常样本
  不得混算。
- OSD 前台只写有界内存队列，由后台线程批量落盘；队列满时计数并丢 trace，不得阻塞 I/O、
  预测或训练。reset 时轮换 trace session，防止相邻实验混合。
- 优先使用定长二进制或结构化批量格式，并提供离线转换脚本；报告记录 schema 版本、代码
  commit、配置哈希、OSD id 和 trace drop 数。

## Otsu 数据源

当前 D2 baseline 使用每 object 最新总热度：feature、Otsu 和 deadline 标签统一为总热度。
Otsu 使用800个固定 score bin、宽度 `0.01`，热度范围 `10` 至约 `29810`；每 object
只保留一票。D0/D1 仅保留为编译期对照，不进入默认部署：

- D0：每个 object 只保留最新的未来新增热度投票；
- D1：每个已完成 EQ I/O 按未来新增热度投票。

已完成同一组五负载单轮数据源对照：D2 的平均 accuracy/balanced accuracy 为
`79.13%/78.69%`，优于 D0 的 `77.22%/73.91%` 和 D1 的 `76.70%/74.66%`。
D2 总热度 feature/Otsu/label 对齐后的单轮五负载验证也已完成，平均
accuracy/balanced accuracy 为 `81.11%/80.18%`，且没有 EQ、训练 drop 或预测错误。
该结果改变了标签语义，不能与上一轮作为严格单变量对照；D0/D1 仅在明确要求时重跑。

完整报告：
[20260715 Otsu 数据源对照](../../ceph-test/new_workload/hp_runs/reports/20260715_074512_otsu_data_source_matrix/REPORT.md)。

D2 对齐验证报告：
[20260715 Score 总热度 Otsu](../../ceph-test/new_workload/hp_runs/reports/20260715_155808_score_total_heat_otsu/REPORT.md)。

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
