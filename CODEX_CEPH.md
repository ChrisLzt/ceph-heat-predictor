# CODEX Context: Ceph Object Heat Predictor

本文档记录 Ceph 侧 object-layer heat predictor 的当前实现状态。参数以代码为准，主要在 `src/heatpredictor/hp_config.h`、`src/heatpredictor/hp_evaluation_queue.h` 和 `src/heatpredictor/heat_predictor.h`。

## 目标和边界

当前原型在 OSD 内对 RADOS object 做在线冷热识别，只输出统计和预测结果，不执行数据迁移或分层放置。

- Ceph 适配层：`src/osd/ObjectHeatPredictor.*`
- 算法入口：`src/heatpredictor/heat_predictor.h`
- 算法组件：`hp_config.h`、`hp_types.h`、`hp_quantile_window.h`、`hp_evaluation_queue.h`
- Hook 入口：`src/osd/PrimaryLogPG.cc`
- MGR 汇总：`src/mgr/DaemonServer.cc`、`src/mgr/MgrCommands.h`

不要把 Ceph 类型、op 解析、perf counter 细节放进 `HeatPredictor` 核心类。

## OSD Hook

有效 hook：

- 函数：`PrimaryLogPG::do_osd_ops(OpContext *ctx, vector<OSDOp>& ops)`
- 位置：`ZERO -> TRUNCATE` 规范化之后，主 `switch (op.op)` 之前
- 调用：`hp_notify_osd_object_op(cct, soid, op)`
- 初始化：`OSD::final_init()` 调用 `init_osd_object_hp_status(cct)`
- perf section：`object_hp_status`

当前纳入模型的普通 object op：

- read：`READ`、`SYNC_READ`、`SPARSE_READ`
- write：`WRITE`、`WRITEFULL`、`WRITESAME`

暂不纳入：`ZERO`、`TRUNCATE`、`APPEND`、`CMPEXT`、`CHECKSUM`、`MAPEXT`、omap、class、watch、cache/tier 管理类 op。

## Key 和特征

当前粒度是 object 级，不再按 offset bucket 切分。hook 输入只保留：

- `pool = soid.pool`
- `ceph_object_hash = soid.get_hash()`
- `object_name_hash = std::hash<object_t>{}(soid.oid)`
- `size = op.extent.length`，`WRITESAME` 使用 `op.writesame.length`

key 构造：

```cpp
key = make_object_key(pool, ceph_object_hash, object_name_hash);
```

`TraceItem` 只保存预测和训练实际使用的字段：`index`、`operation`、`size`、`key`、`current_heat`、`hot_threshold`、`access_count`、`last_access_distance`、`object_age`、`pred_hot_proba`、`pred`。

模型特征数量以 `NUM_FEATURES` 为准，当前为 7 个：

1. `operation`
2. `log2(size + 1)`
3. `log2(access_count + 1)`
4. `access_count > 1 ? current_heat / max(hot_threshold, 1) : 0`
5. `log2(last_access_distance + 1)`
6. `log2(object_age + 1)`
7. `object_age > 0 ? access_count / (object_age + 1) : 0`

`pool` 和两个 object hash 只参与 key，不作为模型特征。

## 当前参数

当前参数位于 `hp_config.h`：

```cpp
HP_HOT_QUANTILE = 0.85
HP_HOT_CLASS_WEIGHT = 4.0
HP_HOT_PREDICT_THRESHOLD = 0.50
HP_EVALUATION_WINDOW = 10000
HP_LABEL_THRESHOLD_WINDOW_CAPACITY = 1000000
HP_HEAT_INCREMENT = 100.0
HP_LRU_CAPACITY = 100000
HP_HEAT_RETAIN_RATIO = 1.0 / 10.0
HP_REPORT_STATS_WINDOW_CAPACITY = 400000
```

热度衰减系数由 `hp_heat_decay_alpha(evaluation_window)` 计算，使热度在一个评估窗口后保留 `HP_HEAT_RETAIN_RATIO`。

## EvaluationQueue 和标签

冷热标签由 `EvaluationQueue` 延迟生成：

- 每条 I/O 独立进入 `pending_queue`，不按 object 合并。
- 第 `t` 条 I/O 在 `t + HP_EVALUATION_WINDOW` 时出队并生成标签。
- 同一 object 在 `heat_map` 中共享 `heat/access_count/pending_count/first_access/last_access`。
- 到期标签使用未来窗口内新增热度：`future_heat = decayed_total_heat - decayed_entry_heat`。
- `future_heat > hot_threshold` 标记为实际热。
- 热样本训练权重固定为 `HP_HOT_CLASS_WEIGHT`，冷样本权重为 1。

WT/阈值窗口维护 object 当前热度分布：

- `record_object_heat(key, heat, timestamp)` 每次访问更新该 object 当前热度。
- `hot_list` 是按 `log(heat) - alpha * timestamp` 存储的可求分位数树。
- `hot_list_order` 是 `std::list`，`hot_list_by_key` 保存 list iterator；同一 object 更新时会删除旧位置再插入队尾，不保留 stale entry。
- 超过 `HP_LABEL_THRESHOLD_WINDOW_CAPACITY` 时淘汰最久未更新 object。
- `hot_threshold` 是当前 object 热度分布的 `HP_HOT_QUANTILE` 分位数，按当前 timestamp 还原为真实热度。

LRU 只管理无 pending 的 object 状态：

- `pending_count > 0` 的 object 不在 LRU，保证 I/O 到期时状态仍存在。
- `pending_count == 0` 后 object 进入 `lru_list`。
- `lru_list.size() > HP_LRU_CAPACITY` 时删除队首 object 的 `heat_map` 状态。

稳定运行时：

```text
hp_io_count = hp_labeled_io_total + hp_pending_io_count
hp_labeled_io_total = TP + FP + TN + FN
```

## 模型和训练

模型由 `PipelineClassifier(StandardScaler, ARFClassifier)` 组成。ARF 参数以 `HeatPredictor::make_model()` 中构造参数为准。

后台训练流程：

- 前台 `predict()` 使用只读 `prediction_snapshot`，并把访问送入 `EvaluationQueue`。
- 到期时同步更新 I/O 级 TP/FP/TN/FN，再把样本送入后台训练队列。
- 后台线程只训练 `train_model`，不直接修改前台正在使用的快照。
- `BATCH_SIZE = 100` 只控制后台训练线程唤醒频率；后台仍有 50ms 定时唤醒，不会无限等待凑满 100 条。
- 每累计训练 `MODEL_UPDATE_REPORT_INTERVAL = 2000` 个样本后，从 `train_model` clone 一个新的 `prediction_snapshot` 并发布。
- `hp_snapshot_publish_count` 表示已发布的预测快照次数。
- 超过 `MAX_TRAIN_QUEUE_LENGTH` 时丢弃最老训练样本并增加 drop 计数。

预测逻辑只使用模型输出概率：

```cpp
pred_hot_proba = predict_proba_one(features)[1]
pred = pred_hot_proba >= HP_HOT_PREDICT_THRESHOLD
```

旧 heat rule 分支已移除。

并发约束：

- `train_model_mutex` 只保护后台训练模型和 clone 过程。
- `prediction_snapshot_mutex` 只保护快照指针发布和前台复制 `shared_ptr`。
- 前台拿到局部 `shared_ptr` 后对只读快照执行 `predict_proba_one()`。
- reset、predict、后台训练使用固定锁顺序，避免旧 batch 写入新模型。

## Reset 接口

单 OSD 清空：

```bash
ceph daemon osd.<id> object_hp reset
```

全局清空：

```bash
ceph osd hp reset
```

reset 需要清空：

- `train_model` 和 `prediction_snapshot`，包括 scaler 和 ARF 树状态
- `EvaluationQueue` 的 pending 队列、共享热度表、LRU、阈值窗口
- 后台训练队列、`model_update_train_count`、`pending_notify`、`snapshot_publish_count`
- 混淆矩阵、预测冷热计数、op 计数
- `object_hp_status` perf counter 的 U64 字段

reset 后应通过下面任一命令看到计数归零：

```bash
ceph daemon osd.0 perf dump object_hp_status
ceph osd hp status -f json-pretty
```

## MGR 汇总

命令：

```bash
ceph osd hp status
ceph osd hp status -f json-pretty
ceph osd hp reset
```

`osd hp status` 默认只输出所有 OSD 的 summary，不展开每个 OSD。数据来自 MGR 已收到的 OSD perf counter。

输出分组：

- `summary.osds`：`up_osds`、`reporting_osds`、`missing_osds`
- `summary.samples`：I/O 总数、已评估数、pending 数
- `summary.heat_state`：共享热度状态、LRU 数量、`hp_hot_threshold_avg`
- `summary.confusion_matrix`：TP/FP/TN/FN
- `summary.actual_behavior`：实际热/冷样本在未来窗口内的 object 平均访问次数、到期平均热度和分布
- `summary.prediction`：预测比例和评估指标
- `summary.training`：训练队列、丢弃样本和预测快照发布次数
- `summary.latency`：所有上报 OSD 的预测耗时总和、次数和全局平均值
- `summary.read_ops`：read 类 op 计数
- `summary.write_ops`：write 类 op 计数

汇总规则由 `DaemonServer.cc` 的 `object_hp_counter_fields` 表驱动：

- `sum`：计数字段直接求和。
- `io_weighted`：如 `hp_pred_hot_percent`，按 `hp_io_count` 加权平均。
- `hot_weighted` / `cold_weighted`：按实际热/冷样本数加权平均。
- `osd_average`：如 `hp_hot_threshold_avg`，按上报 OSD 数简单平均，仅作参考。
- `none`：不直接聚合，由全局 TP/FP/TN/FN 重新计算。

MGR 汇总会把 OSD perf 中 `x10000` 的阈值和平均访问次数还原成浮点值；比例类指标输出为 `0~100` 的百分比数值，保留 5 位有效数字。`hp_predict_latency` 汇总各 OSD 的总纳秒数和次数，再计算全局平均，不直接平均各 OSD 的平均值。

## object_hp_status 字段

OSD perf 命令：

```bash
ceph daemon osd.0 perf dump object_hp_status
```

字段顺序需要在 enum、PerfCountersBuilder、更新逻辑、输出逻辑中保持一致。

| 字段 | 含义 |
| --- | --- |
| `hp_io_count` | 进入冷热识别的有效 object op 数 |
| `hp_labeled_io_total` | 已完成未来窗口评估的 I/O 数 |
| `hp_pending_io_count` | 当前待评估 I/O 数 |
| `hp_heat_state_count` | 共享热度表中的 object 数 |
| `hp_lru_count` | 无 pending、位于 LRU 的 object 数 |
| `hp_true_positive_count` | 实际热且预测热 |
| `hp_false_positive_count` | 实际冷但预测热 |
| `hp_true_negative_count` | 实际冷且预测冷 |
| `hp_false_negative_count` | 实际热但预测冷 |
| `hp_actual_hot_object_avg_future_access_count` | 实际热样本的未来窗口内 object 平均访问次数，除以 `10000` |
| `hp_actual_cold_object_avg_future_access_count` | 实际冷样本的未来窗口内 object 平均访问次数，除以 `10000` |
| `hp_actual_hot_object_avg_heat` | 实际热样本到期时的 object 平均热度，除以 `10000` |
| `hp_actual_cold_object_avg_heat` | 实际冷样本到期时的 object 平均热度，除以 `10000` |
| `hp_pred_hot_percent` | 在线预测路径中预测热比例，除以 `10000` |
| `hp_eval_pred_hot_percent` | 评估样本预测热比例 |
| `hp_eval_actual_hot_percent` | 评估样本实际热比例 |
| `hp_hot_accuracy` | `(TP + TN) / labeled_total * 10000` |
| `hp_hot_precision` | `TP / (TP + FP) * 10000` |
| `hp_hot_recall` | `TP / (TP + FN) * 10000` |
| `hp_hot_threshold` | 当前 OSD 本地热度阈值 |
| `hp_train_queue_length` | 后台训练队列长度 |
| `hp_train_drop_count` | 后台训练队列满后丢弃的样本数 |
| `hp_snapshot_publish_count` | 每累计训练 `MODEL_UPDATE_REPORT_INTERVAL` 个样本发布一次预测快照 |
| `hp_op_read_count` | `CEPH_OSD_OP_READ` 数量 |
| `hp_op_sync_read_count` | `CEPH_OSD_OP_SYNC_READ` 数量 |
| `hp_op_sparse_read_count` | `CEPH_OSD_OP_SPARSE_READ` 数量 |
| `hp_op_write_count` | `CEPH_OSD_OP_WRITE` 数量 |
| `hp_op_writefull_count` | `CEPH_OSD_OP_WRITEFULL` 数量 |
| `hp_op_writesame_count` | `CEPH_OSD_OP_WRITESAME` 数量 |
| `hp_predict_latency` | 预测路径采样耗时，当前每 1000 条有效 I/O 记录一次 |

MGR 汇总公式：

```text
hp_io_count             = hp_labeled_io_total + hp_pending_io_count
hp_labeled_io_total     = TP + FP + TN + FN
eval_pred_hot_percent   = (TP + FP) / hp_labeled_io_total * 100
eval_actual_hot_percent = (TP + FN) / hp_labeled_io_total * 100
hot_accuracy            = (TP + TN) / hp_labeled_io_total * 100
hot_precision           = TP / (TP + FP) * 100
hot_recall              = TP / (TP + FN) * 100
```

分母为 0 时指标输出 0。评估时重点看 accuracy、precision、recall、actual hot percent 和 confusion matrix。

## 构建、安装和重启

代码改动后可以直接执行全量构建、安装、ldconfig 和重启：

```bash
cd /home/chris/ceph-heat-predictor/build
sudo env CCACHE_TEMPDIR=/tmp ninja -j64
sudo ninja install
sudo ldconfig
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo systemctl restart ceph-mgr@s52
sudo ceph -s
```

常见缺失开发包：

```bash
sudo apt install -y libfmt-dev libsqlite3-dev liblttng-ust-dev xfslibs-dev
```

注意事项：

- 当前输出 section 是 `object_hp_status`，不是旧的 `hp_status`。
- `PerfCountersBuilder` 的 object_hp 字段需要能被 MGR 收集，否则 `osd hp status` 会出现 `missing_osds`。
- reset 后如果 workload 立即进入，统计会马上增长；判断 reset 是否成功应在 workload 停止或下一轮开始前完成。
