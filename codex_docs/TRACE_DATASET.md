# Heat Predictor Trace 设计

Trace 仅存在于 `dev`，用于离线复现已完成评估的 I/O、feature 消融和错误样本分析；
默认关闭，不进入 `main`。

## 行为与接口

- `ceph osd hp trace start [phase] [directory]`：在所有 up OSD 上开启新 session；重复
  start 会先完整落盘旧队列，再以新阶段名轮换文件。
- `ceph osd hp trace stop`：停止接收新记录，排空队列并关闭文件。
- `ceph daemon osd.N object_hp status` 和 `ceph osd hp status` 输出 trace 开关、队列、
  已写、丢弃和写错误计数。
- `object_hp reset`、enable 和 disable 在 trace 已开启时轮换 session，避免相邻实验混合。
- 默认目录为 `/var/log/ceph`；文件名包含 OSD id、墙钟时间和 session id。

## 数据路径

预测前台和 EQ 到期线程只调用 `try_submit()`。它通过原子占位写入容量为 `65536`
的有界 MPSC ring；只有 ring 确实已满时才增加 drop 计数，不争用 writer 锁且不等待
磁盘。独立 writer 线程每批最多写 `4096` 条定长记录，stop/rotate 时排空并 flush。

每个文件只对应一个 phase。文件头记录 schema、record/feature 大小、OSD id、session、
墙钟/单调起点、配置哈希和 Ceph Git 版本。记录保存匿名 object key、预测/标签时刻、预测时
feature、概率与固定阈值、预测类别、最终标签、标签总热度与热阈值、未来访问次数，以及正常、
EQ 容量丢弃、预测错误和 cold-start fallback 标记。未完成评估的异常记录使用 `label=-1`，
不会与正常混淆矩阵样本混算。

## 验证

- `hp_trace_probe` 验证文件头、正常/异常记录、后台排空和 session 轮换；
  `hp_algorithm_probe` 验证固定算法基线和 EQ 公开生命周期。
- Python 转换脚本严格校验 magic、schema、header size 和 record size，再输出 CSV。
- 编译 `ceph-osd`、`ceph-mgr` 和全量 Ninja target；Trace 关闭时重新运行现有算法探针。

## 离线诊断

正式采集完成后执行：

```bash
python3 test_sh/analyze_hp_trace.py \
  --run-root /path/to/run \
  --output-dir /path/to/run/offline_analysis
```

分析器逐个 OSD 流式读取 CSV，校验 metadata schema、CSV 记录数、时间顺序、概率范围，
以及 `predicted_label`/`actual_label` 与对应阈值的一致性。预测时的单调时钟通过文件头锚点
转换为墙钟，再关联 `phase_intervals.tsv`。不同 OSD 的指标在文件校验完成后才合并；object
身份固定为 `(workload, osd_id, object_key_hash)`。

输出内容：

- `summary.json`：输入清单、全局/负载混淆矩阵及校准、object 摘要。
- `workload_summary.tsv`：Accuracy、Balanced Accuracy、Precision、Recall、ECE、Brier。
- `phase_error.tsv`：按热点阶段和 `cold_start`/`hotspot_transition`/`steady` 分解错误。
- `calibration.tsv`：10 桶概率校准数据。
- `margin_analysis.tsv`：预测边界与标签边界的联合分桶，可区分边界错误和高置信错误。
- `feature_stats.tsv`：真实冷热类及 TP/FP/TN/FN 的 feature 在线矩。
- `feature_correlation.tsv`：feature 与标签、预测概率及 feature 两两 Pearson 相关系数。
- `object_stats.tsv`：访问次数、错误率及头部 object 集中度。
- `REPORT.md`：中文结果索引和阶段摘要。

## 精确回放门槛

回放工具只接受 **Trace schema v2** 中已完成评估的记录。这里的 `v2` 是文件格式
版本，不是 Heat Predictor V2 算法版本。每个 OSD 独立按
`prediction_time_ns` 预测、按 `label_completion_time_ns` 训练；同一时刻先预测后训练。
训练模型达到 500 个样本或模拟时间经过 1 秒后发布只读预测快照。逐记录 TSV 保持原始
Trace 文件顺序，供 phase reporter 与 CSV、metadata 和 `phase_intervals.tsv` 流式关联。

```bash
g++ -std=c++17 -O2 -pthread \
  -Ibuild/src/include -Ibuild/include -Isrc -Itest_sh \
  test_sh/hp_trace_replay.cc -Lbuild/lib -Wl,-rpath,build/lib \
  -lceph-common -o /tmp/hp_trace_replay

/tmp/hp_trace_replay WORKLOAD/trace/osd.0.bin \
  --output REPLAY_DIR/WORKLOAD/osd.0.replay.tsv

python3 test_sh/analyze_hp_replay.py \
  --run-root RUN_ROOT \
  --replay-dir REPLAY_DIR \
  --output-dir REPORT_DIR
```

正式 gate 固定为：记录和标签关联 100%、类别一致率至少 99%、概率 MAE 不超过 0.01、
概率绝对误差 P95 不超过 0.05、replay 与线上 Accuracy 差不超过 0.2 个百分点。先且仅先
回放 MapReduce `osd.0`；任一指标失败就停止 Trace schema v2 实验，不运行其他 OSD，也不做
feature 消融。

Trace schema v2 没有保存预测快照 generation/version，因此只能解释已记录策略，不能保证模型
快照发布时序完全复现。若 baseline parity 无法通过，应先给 Trace 增加快照代次，而不是放宽
一致性标准。
