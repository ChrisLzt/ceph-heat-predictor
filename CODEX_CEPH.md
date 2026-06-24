# CODEX Context: Ceph Object Heat Predictor

本文档记录 Ceph 侧 object-layer heat predictor 的当前实现状态。测试流程见 `CODEX_TEST.md`，部署命令见 `CEPH_OPERATIONS_MANUAL.md`。

## 目标和边界

当前原型在 OSD 内对 RADOS object bucket 做在线冷热识别，只输出统计和预测结果，不执行数据迁移或分层放置。

- Ceph 适配层：`src/osd/ObjectHeatPredictor.*`
- 通用算法：`src/heatpredictor/`
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

## Key、Bucket 和特征

hook 输入：

- `pool = soid.pool`
- `ceph_object_hash = soid.get_hash()`
- `object_name_hash = std::hash<object_t>{}(soid.oid)`
- `object_offset = op.extent.offset`
- `size = op.extent.length`
- `object_bucket = object_offset >> HP_BUCKET_SHIFT`

key 构造：

```cpp
key = make_object_key(
    pool, ceph_object_hash, object_name_hash, object_bucket);
```

当前 bucket 粒度：

```cpp
HP_BUCKET_SHIFT = 16; // 64KB
```

运行参数以 `src/heatpredictor/heat_predictor.h` 中的常量和构造参数为准，文档只记录当前值。

模型特征数量以 `NUM_FEATURES` 为准，当前为：

1. `operation`
2. `log2(size + 1)`
3. `log2(object_bucket + 1)`
4. `log2(access_count + 1)`
5. `access_count > 1 ? current_heat / max(hot_threshold, 1) : 0`

`pool`和两个 object hash 只参与 key，不作为模型特征。首次访问 bucket
时热度比值置 0，避免初始 heating 误导模型。

## 模型和训练

模型由 `PipelineClassifier(StandardScaler, ARFClassifier)` 组成。ARF 参数以 `HeatPredictor::make_model()` 中构造参数为准。

冷热标签由 `EvaluationQueue` 延迟生成：

- 每条 I/O 独立进入待评估队列，不按 bucket 合并
- `HP_EVALUATION_WINDOW = 20000`，单位是当前 OSD 的有效 object op 数
- `HP_THRESHOLD_HISTORY_CAPACITY = 400000`，与评估窗口解耦，严格保留
  最近 400000 个到期热度样本
- 同一 bucket 共享 `heat/access_count/pending_count`
- `HP_HEAT_INCREMENT = 100`，首次及后续每次访问贡献相同热度
- `HP_HOT_QUANTILE = 0.80`
- `HP_HOT_CLASS_WEIGHT = 1.2`
- 第 `t` 条 I/O 在 `t + 20000` 时按共享热度生成标签
- `pending_count == 0` 的 bucket 进入容量为 20000 的 LRU
- 训练权重为 `min(3, 1 + log2(1 + future_access_count))`

当前训练按 I/O 进行，不按 bucket 去重。同一高频 bucket 会同时产生更多
训练样本，并使单样本获得更高访问权重，因此存在频次和权重的双重放大。
I/O 级评估不受训练权重影响。

状态结构：

- `pending_queue`：每条 I/O 一个 `TraceItem`，保存预测时特征和结果
- `heat_map`：每 bucket 一份共享热度、访问次数和 pending 数
- `lru_list`：只包含 `pending_count == 0` 的 bucket，队首按容量驱逐
- `pending_count > 0` 的 bucket 不在 LRU，保证到期时状态仍存在

稳定运行时：

```text
hp_io_count = hp_labeled_io_total + hp_pending_io_count
hp_labeled_io_total = TP + FP + TN + FN
```

后台训练流程：

- 前台 `predict()` 使用 active model，并把访问送入 `EvaluationQueue`
- 到期时同步更新 I/O 级 TP/FP/TN/FN，再把样本送入后台训练队列
- 后台线程训练 shadow model
- 每训练 `SWAP_INTERVAL` 个样本后交换 active/shadow model
- 超过 `MAX_TRAIN_QUEUE_LENGTH` 时丢弃最老训练样本并增加 drop 计数

并发约束：

- `predict()` 全程持有 `swap_mutex` 调用 `predict_one()`，避免 swap 后旧 active 变成 shadow 并被后台训练线程并发写入。
- reset、predict、后台训练使用固定锁顺序，避免旧 batch 写入新模型。
- 未来若要缩短预测锁范围，应改成只读快照发布，不能只拷贝 `shared_ptr` 后无锁读取。

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

- active/shadow model，包括 scaler 和 ARF 树状态
- `EvaluationQueue` 的 pending 队列、共享热度表、LRU 和阈值历史
- 后台训练队列、`shadow_train_count`、`pending_notify`、`swap_count`
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

- `summary.osds`：`up_osds`、`reporting_osds`
- `summary.samples`：I/O 总数、已评估数、pending 数
- `summary.heat_state`：共享热度状态和 LRU 数量
- `summary.confusion_matrix`：TP/FP/TN/FN
- `summary.prediction`：预测比例和评估指标
- `summary.training`：训练队列、丢弃样本和模型切换
- `summary.latency`：所有上报 OSD 的预测耗时总和、次数和全局平均值
- `summary.read_ops`：read 类 op 计数
- `summary.write_ops`：write 类 op 计数
- `missing_osds`：up 但尚未上报 `object_hp_status` 的 OSD

汇总规则：

- 计数字段直接求和。
- `hp_pred_hot_percent` 按 `hp_io_count` 加权平均。
- `hp_eval_pred_hot_percent`、`hp_eval_actual_hot_percent`、`hp_hot_accuracy`、`hp_hot_precision`、`hp_hot_recall` 由全局 TP/FP/TN/FN 重新计算。
- `hp_hot_threshold` 不做跨 OSD 融合，不同 OSD 的本地阈值没有直接平均意义。
- `hp_predict_latency` 分别汇总各 OSD 的总纳秒数和次数，再计算全局平均；
  不直接平均各 OSD 的平均值。

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
| `hp_heat_state_count` | 共享热度表中的 bucket 数 |
| `hp_lru_count` | 无 pending、位于 LRU 的 bucket 数 |
| `hp_true_positive_count` | 实际热且预测热 |
| `hp_false_positive_count` | 实际冷但预测热 |
| `hp_true_negative_count` | 实际冷且预测冷 |
| `hp_false_negative_count` | 实际热但预测冷 |
| `hp_pred_hot_percent` | 在线预测路径中预测热比例，除以 `10000` |
| `hp_eval_pred_hot_percent` | 评估样本预测热比例 |
| `hp_eval_actual_hot_percent` | 评估样本实际热比例 |
| `hp_hot_accuracy` | `(TP + TN) / labeled_total * 10000` |
| `hp_hot_precision` | `TP / (TP + FP) * 10000` |
| `hp_hot_recall` | `TP / (TP + FN) * 10000` |
| `hp_hot_threshold` | 当前 OSD 本地热度阈值 |
| `hp_train_queue_length` | 后台训练队列长度 |
| `hp_train_drop_count` | 后台训练队列满后丢弃的样本数 |
| `hp_swap_count` | active/shadow 模型切换次数 |
| `hp_op_read_count` | `CEPH_OSD_OP_READ` 数量 |
| `hp_op_sync_read_count` | `CEPH_OSD_OP_SYNC_READ` 数量 |
| `hp_op_sparse_read_count` | `CEPH_OSD_OP_SPARSE_READ` 数量 |
| `hp_op_write_count` | `CEPH_OSD_OP_WRITE` 数量 |
| `hp_op_writefull_count` | `CEPH_OSD_OP_WRITEFULL` 数量 |
| `hp_op_writesame_count` | `CEPH_OSD_OP_WRITESAME` 数量 |
| `hp_predict_latency` | 预测路径采样耗时，当前每 1000 条有效 I/O 记录一次 |

MGR 延迟输出：

```text
summary.latency.hp_predict_latency.avgcount
summary.latency.hp_predict_latency.sum_ns
summary.latency.hp_predict_latency.avgtime_ns
```

公式：

```text
hp_io_count             = hp_labeled_io_total + hp_pending_io_count
hp_labeled_io_total     = TP + FP + TN + FN
eval_pred_hot_percent   = (TP + FP) / hp_labeled_io_total * 10000
eval_actual_hot_percent = (TP + FN) / hp_labeled_io_total * 10000
hot_accuracy            = (TP + TN) / hp_labeled_io_total * 10000
hot_precision           = TP / (TP + FP) * 10000
hot_recall              = TP / (TP + FN) * 10000
```

分母为 0 时指标输出 0。评估时重点看 precision、recall、actual hot percent 和 confusion matrix；冷样本占多数时不要只看 accuracy。

## 构建和安装

常用构建：

```bash
cd /home/chris/ceph-heat-predictor/build
env CCACHE_TEMPDIR=/tmp ninja ceph-osd ceph-mgr -j"$(nproc)"
```

常见缺失开发包：

```bash
sudo apt install -y libfmt-dev libsqlite3-dev liblttng-ust-dev xfslibs-dev
```

安装当前 OSD/MGR：

```bash
sudo install -o root -g root -m 0755 bin/ceph-osd /usr/bin/ceph-osd
sudo install -o root -g root -m 0755 bin/ceph-mgr /usr/bin/ceph-mgr
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo systemctl restart ceph-mgr@s52.service
```

## 注意事项

- 当前输出 section 是 `object_hp_status`，不是旧的 `hp_status`。
- `PerfCountersBuilder` 的 object_hp 字段需要能被 MGR 收集，否则 `osd hp status` 会出现 `missing_osds`。
- reset 后如果 workload 立即进入，统计会马上增长；判断 reset 是否成功应在 workload 停止或下一轮开始前完成。
- 当前待评估队列固定保留最近 20000 条有效 I/O；共享热度表额外保留最多 20000 个无 pending bucket。
