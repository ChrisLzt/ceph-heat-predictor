# CODEX Context: Ceph Object Heat Predictor

本文档记录当前 Ceph 冷热识别原型的实现状态和关键参数。测试流程统一见 `CODEX_TEST.md`。

## 当前目标

当前项目是在 Ceph OSD 内加入在线冷热识别逻辑。现阶段只输出冷热判断结果，不实现“热数据写入 SSD、冷数据写入 HDD”的实际放置或迁移逻辑。

识别结果应绑定到 RADOS object 或 object 内 bucket，为后续分层存储保留空间。

## Hook 位置

当前有效 hook：

- hook 文件：`src/osd/PrimaryLogPG.cc`
- 适配层文件：`src/osd/ObjectHeatPredictor.h`、`src/osd/ObjectHeatPredictor.cc`
- 函数：`PrimaryLogPG::do_osd_ops(OpContext *ctx, vector<OSDOp>& ops)`
- 调用点：`do_osd_ops()` 的 `for` 循环内，`ZERO -> TRUNCATE` 规范化之后，主 `switch (op.op)` 之前
- 状态输出：`object_hp_status`
- 初始化位置：`src/osd/OSD.cc::final_init()` 调用 `init_osd_object_hp_status(cct)`

当前结构：

```text
src/osd/PrimaryLogPG.cc
  -> hp_notify_osd_object_op(cct, soid, op)

src/osd/ObjectHeatPredictor.cc
  -> 过滤 OSD op
  -> 提取 offset/length
  -> 维护 object_hp_status perf counter
  -> 调用 HeatPredictor::predict()

src/heatpredictor/
  -> 通用冷热识别算法
```

`PrimaryLogPG.cc` 只保留 hook 调用，不再直接包含冷热识别算法、op 解析、perf counter 注册等实现细节。

记录的 op：

- read-like：`READ`、`SYNC_READ`、`SPARSE_READ`
- write-like：`WRITE`、`WRITEFULL`、`WRITESAME`

暂时跳过：`ZERO`、`TRUNCATE`、`APPEND`、`CMPEXT`、`CHECKSUM`、`MAPEXT`、omap、class、watch、cache/tier 管理类 op。

设计原则：

- 当前目标是统计 vdbench、fio、ior 等测试工具产生的普通对象读写。
- 不把校验、extent 查询、append、zero、元数据、class、watch、cache/tier 管理类 op 混入模型。

## Object Key 和 Bucket

当前 object hook 输入字段：

- `pool`：来自 `soid.pool`
- `ceph_object_hash`：来自 `soid.get_hash()`
- `object_name_hash`：来自 `std::hash<object_t>{}(soid.oid)`
- `object_offset`：来自 OSD op extent offset
- `size`：来自 OSD op extent length
- `object_bucket`：`object_offset >> HP_BUCKET_SHIFT`

key 使用：

```cpp
object_hash = mix64(ceph_object_hash) ^ mix64(object_name_hash);
key = hash(pool, object_hash, object_bucket);
```

注意：

- `object_hash` 作为 key 的一部分，用于区分不同 RADOS object。
- `object_hash` 不作为模型特征，因为 hash 数值本身没有冷热语义。
- `soid.get_hash()` 是 Ceph 用于对象分布/PG 映射的 hash，不是对象唯一 ID；不能只用 `pool + soid.get_hash() + bucket` 作为最终 key。

## HeatPredictor 实现

核心文件：

- `src/heatpredictor/heat_predictor.h`
- `src/heatpredictor/include/ARFClassifier.h`
- `src/heatpredictor/include/HoeffdingTree*.h/.tpp`
- `src/heatpredictor/include/StandardScaler.h`
- `src/heatpredictor/include/utils.h`

当前特征数量以 `src/heatpredictor/heat_predictor.h` 中的 `NUM_FEATURES` 为准。

当前 bucket 粒度以 `HP_BUCKET_SHIFT` 为准。

当前热标签分位数以 `HP_HOT_QUANTILE` 为准。

当前 object 层特征：

1. `operation`
2. `log2(size + 1)`
3. `log2(object_bucket + 1)`
4. `log2(access_count + 1)`
5. `access_count > 1 ? current_heat / max(hot_threshold, 1) : 0`

特征设计要点：

- `offset` 已由 `bucket` 表达，不再作为模型特征。
- `pool` 和 `object_hash` 只用于 key，不再作为模型特征。
- 首次访问某个 bucket 时 `current_heat` 使用 `heating` 初始值，所以首次访问时热度比值置 0。
- `log2()` 用于压缩 size、bucket、access_count 的长尾分布；`StandardScaler` 再统一均值和方差。

## StandardScaler

当前 `StandardScaler` 是在线无监督归一化器，忽略传入的 `y`，只用 `x` 更新统计量。

当前实现：

- `counts` 使用 `uint64_t`
- 使用 Welford 写法维护 `means` 和 `m2s`
- `transform_one()` 使用总体方差 `m2 / count`

调用链：

```text
HeatPredictor::train_worker()
  -> shadow_model->learn_one(to_feat(sample.item), sample.label, weight)
    -> PipelineClassifier::learn_one(x, y, w)
      -> transformer->learn_one(x, y)
      -> classifier->learn_one(transformer->transform_one(x), y, w)
```

## 标签生成队列

冷热标签由 `EvaluationQueue` 根据延迟后的访问热度生成。

当前队列参数以 `EvaluationQueue` 构造函数为准，重点关注：

- `max_size`
- `hot_threshold`
- `alpha`
- `heating`
- `waiting`

热度更新：

```cpp
new_heat = exp(delta_ts * alpha) * old_heat + heating
```

出队逻辑：

1. `key_order` 按 bucket 首次进入队列的顺序保存 key。
2. `item_map` 保存 bucket 的当前热度、首次访问时间、最近访问时间、访问次数和原始样本。
3. 每次 enqueue 后检查队首 bucket。
4. 如果队首 bucket 已等待至少 `waiting`，则出队并生成标签。
5. 如果队列超过 `max_size`，也会触发队首出队。

热标签判断：

```cpp
is_hot = val.heat > get_hot_threshold();
```

`HP_HOT_QUANTILE` 控制热标签分位数，数值越高，标为 hot 的样本越少。

## 模型参数

当前模型由 `PipelineClassifier` 组合：

```cpp
PipelineClassifier(
    new StandardScaler<NUM_FEATURES>(),
    new ARFClassifier<...>(...)
)
```

当前 ARF 参数以 `make_model()` 中的 `ARFClassifier` 构造参数为准，重点关注：

- `n_models`
- `max_features`
- `seed`
- `grace_period`
- `lambda_value`
- `delta`
- `tau`
- `max_share_to_split`
- `min_branch_fraction`

热/冷训练权重以 `train_worker()` 中的 `weight` 计算逻辑为准。

`ARFClassifier::learn_one()` 中外部权重 `w` 会乘到 Poisson 采样得到的 `k` 上：

```cpp
double sample_weight = w * k;
model->learn_one(x, y, sample_weight);
```

## 后台训练

预测线程使用 active model，训练线程使用 shadow model。

关键参数以 `HeatPredictor` 中的常量为准：

- `SWAP_INTERVAL`
- `BATCH_SIZE`
- `MAX_TRAIN_QUEUE_LENGTH`

流程：

1. 前台 `predict()` 预测，并把访问送入 `EvaluationQueue`。
2. bucket 出队后生成训练样本。
3. 训练样本进入后台训练队列。
4. 后台线程批量训练 shadow model。
5. 每训练 `SWAP_INTERVAL` 个样本后交换 active/shadow model。

`BATCH_SIZE` 是唤醒后台训练线程的通知阈值，不代表每批样本一定马上训练完成。如果样本产生速度高于后台训练速度，训练队列仍可能积压。当前队列超过 `MAX_TRAIN_QUEUE_LENGTH` 后，会从队头丢弃最老训练样本，保留新样本。

## 测试入口

测试流程、后台运行方式、日志目录和大规模数据构造注意事项统一维护在 `CODEX_TEST.md`。

## object_hp_status 字段

`ceph daemon osd.0 perf dump object_hp_status` 输出 JSON，核心字段：

| 字段 | 含义 | 解释方式 |
| --- | --- | --- |
| `hp_count` | 已进入冷热识别的 object op 数量 | hook 每次有效读写 op 递增 |
| `hp_train_total` | 已参与准确率统计的训练样本数 | 来自延迟标注后的样本 |
| `hp_hot_percent` | 预测为热的比例 | 数值除以 `10000` |
| `hp_actual_hot_percent` | 延迟标注后实际热样本比例 | 数值除以 `10000` |
| `hp_accuracy` | 总体准确率 | 数值除以 `10000` |
| `hp_hot_precision` | 热类 precision | 数值除以 `10000` |
| `hp_hot_recall` | 热类 recall | 数值除以 `10000` |
| `hp_hot_threshold` | 当前热度阈值 | perf 输出中按内部格式放大 |
| `hp_train_queue_length` | 后台训练队列长度 | 长时间升高说明训练线程跟不上 |
| `hp_swap_count` | active/shadow 模型切换次数 | 每训练 `SWAP_INTERVAL` 个样本后增加 |
| `hp_dequeue_waiting_count` | 因等待时间到期出队的样本数 | 用于判断 `waiting` 是否主导标注延迟 |
| `hp_dequeue_max_size_count` | 因队列达到上限出队的样本数 | 用于判断 `max_size` 是否过小 |
| `hp_eval_hot_percent` | 参与评估样本中预测为热的比例 | 更适合判断模型实际热预测占比 |
| `hp_predict_latency` | 预测路径耗时统计 | `avgtime` 或 `sum / avgcount` 为平均耗时 |

普通 OSD op 计数字段：

| 字段 | 含义 |
| --- | --- |
| `hp_op_read_count` | `CEPH_OSD_OP_READ` 数量 |
| `hp_op_sync_read_count` | `CEPH_OSD_OP_SYNC_READ` 数量 |
| `hp_op_sparse_read_count` | `CEPH_OSD_OP_SPARSE_READ` 数量 |
| `hp_op_write_count` | `CEPH_OSD_OP_WRITE` 数量 |
| `hp_op_writefull_count` | `CEPH_OSD_OP_WRITEFULL` 数量 |
| `hp_op_writesame_count` | `CEPH_OSD_OP_WRITESAME` 数量 |

perf counter 刷新频率由 `src/osd/ObjectHeatPredictor.cc` 中的 `object_hp_logger_update_interval` 控制。当前逻辑按固定样本间隔刷新，不再使用启动阶段每次请求都刷新的 warmup 逻辑。

评估时不要只看 `accuracy`。冷样本占多数时，全预测冷也会得到较高 accuracy。重点看：

- `hp_hot_precision`
- `hp_hot_recall`
- `hp_hot_percent` 是否长期接近 0
- `hp_actual_hot_percent`
- `accuracy - (1 - actual_hot_percent)` 是否明显大于 0

## 后续优化方向

1. 对比 5 特征版本与之前 8 特征版本的 precision/recall。
2. 如果 `hp_hot_percent` 仍长期偏低，可考虑模型预测叠加规则兜底，例如 `current_heat >= threshold * 0.9` 时直接判热。
3. 如果训练队列仍积压，可把后台线程从“一次 swap 整个队列”改成“每轮最多取固定数量样本”。
4. 如果热阈值收敛慢，可继续调整 `HP_HOT_QUANTILE`、`waiting`、`alpha`、`SWAP_INTERVAL`。
5. 如果要严格复现实验，需要检查 `ARFClassifier` 内随机引擎是否完全使用固定 seed。

## 注意事项

- 当前输出 section 是 `object_hp_status`，不是旧的 `hp_status`。
- 当前 object 层结果可以绑定到 RADOS object 或 object 内 bucket；如果未来要上升到 CephFS 文件级冷热，还需要额外的文件到 object 映射逻辑。
