# CODEX Context: Ceph Object Heat Predictor

本文档记录 Ceph object heat predictor 的当前实现；参数以 `src/heatpredictor/hp_config.h` 为准。

## 目标和边界

当前原型在 OSD 内对 RADOS object 做在线冷热识别，只输出统计和预测结果，不执行数据迁移或分层放置。

- Ceph 适配层：`src/osd/ObjectHeatPredictor.*`
- 算法入口：`src/heatpredictor/heat_predictor.h`
- 算法组件：`hp_config.h`、`hp_types.h`、`hp_features.h`、`hp_evaluation_queue.h`、`hp_otsu_histogram.h`、`hp_prediction_threshold.h`
- Hook 入口：`src/osd/PrimaryLogPG.cc`
- MGR 汇总：`src/mgr/DaemonServer.cc`、`src/mgr/MgrCommands.h`

不要把 Ceph 类型、op 解析、perf counter 细节放进 `HeatPredictor` 核心类。

## OSD Hook

`PrimaryLogPG` 在各 op 完成 Ceph 原生范围规范化和参数校验后调用
`hp_notify_osd_object_op(cct, soid, op_type)`。初始化由 `OSD::final_init()`
调用 `init_osd_object_hp_status(cct)`；perf section 为 `object_hp_status`。

当前纳入模型的普通 object op：

- read：`READ`、`SYNC_READ`、`SPARSE_READ`
- write：`WRITE`、`WRITEFULL`、`WRITESAME`

暂不纳入：`ZERO`、`TRUNCATE`、`APPEND`、`CMPEXT`、`CHECKSUM`、`MAPEXT`、omap、class、watch、cache/tier 管理类 op。

## Key 和特征

当前粒度是 object 级，不再按 offset bucket 切分。预测器从 `soid` 提取：

- `pool = soid.pool`
- `ceph_object_hash = soid.get_hash()`
- `object_name_hash = std::hash<object_t>{}(soid.oid)`

key 构造：

```cpp
key = make_object_key(pool, ceph_object_hash, object_name_hash);
```

`hp_notify_osd_object_op()` 的输入只保留 `CephContext`、`soid` 和已经确认有效的原始 op 类型，不接收 `offset`、`length` 或 `object_size`。有效范围完全复用 `PrimaryLogPG` 的原生处理结果：

- `READ/SYNC_READ` 在 `do_read()` 完成 truncate、零长度整对象读展开和越界裁剪后通知。
- `SPARSE_READ` 在 `do_sparse_read()` 完成有效范围裁剪后通知。
- `WRITE/WRITEFULL` 在对应分支完成参数校验和 truncate 调整后，仅对非零数据范围通知。
- `WRITESAME` 由其生成的内部 `WRITE` 路径通知一次，并保留 `WRITESAME` 类型计数，避免双重统计。

`TraceItem` 保存预测时快照及评估所需字段：`index`、`key`、`current_heat`、`hot_threshold`、`access_count`、`last_access_distance`、`past_window_access_count`、`recent_window_access_count`、`pred_hot_proba`、`pred`。

模型特征数量以 `NUM_FEATURES` 为准，当前为 6 个：

1. `heat_threshold_margin = log2p1(current_heat) - log2p1(max(hot_threshold, 1))`
2. `last_access_distance_log2p1 = log2p1(last_access_distance)`
3. `current_heat_log2p1 = log2p1(current_heat)`
4. `past_window_access_count_log2p1 = log2p1(pending_count)`
5. `heat_concentration = log2p1(current_heat / (HP_HEAT_INCREMENT * (pending_count + 1)))`
6. `access_acceleration = log2(((recent_count + 1) / 2500) / ((pending_count + 1) / 10000))`

`log2p1(x)` 表示 `std::log2(1 + x)`。`pending_count` 和 `recent_count` 均在当前 I/O 入队前读取，分别近似同一 object 在过去 `10000` 和 `2500` 条 I/O 中的访问次数。短窗口用有界 key 队列和 `HeatState::short_count` 增量维护，不扫描 EQ。

`operation`、`size`、`offset`、`pool` 和两个 object hash 不作为模型特征；`operation` 只在 OSD hook 层用于 op 计数，`size/offset` 不进入预测器。

每条 I/O 的 feature 是入队和预测时的瞬时快照；出队训练时复用该快照生成训练输入，只用未来窗口结果生成 label，避免把预测之后的状态泄露进 feature。

## 当前参数

- ARF：`N_MODELS=25`、`MAX_FEATURES=NUM_FEATURES`、`SEED=591422`、`GRACE_PERIOD=100`、`LAMBDA=4`、`DELTA=0.001`、`TAU=0.05`、`MAX_SHARE_TO_SPLIT=0.99`、`MIN_BRANCH_FRACTION=0.01`。
- 预测：初始阈值 `0.50`，范围 `0.40~0.60`，EMA `0.10`；监督校准窗口
  `10000`、最少样本 `1000`、每 `500` 个评估样本更新、概率直方图 `1001` bins；
  冷热训练样本权重均为 `1.0`。
- 容量：`EVALUATION_WINDOW=10000`、`ACCESS_ACCELERATION_WINDOW=2500`、`LRU_CAPACITY=100000`、`LABEL_THRESHOLD_WINDOW_CAPACITY=1000000`、`REPORT_STATS_WINDOW_CAPACITY=400000`。
- 热度：`HEAT_INCREMENT=100.0`、一个 EQ 窗口后保留 `1/10`。
- 动态热阈值：初始热度 `100`；Otsu 最少 `32` objects、每 `100` 次更新；热度
  clamp `1~3000`，log bin 宽 `0.05`。总置信度使用 separation/sharpness 的加权
  几何平均，权重为 `0.80/0.20`；sharpness 按近似最优阈值之间会改变分类的 object
  比例计算，该比例达到 `0.20` 时归零。O2 单次 score 更新增益不超过 `0.50`。
- 阈值状态：`0 initializing`、`1 tracking`、`2 holding`。

热度衰减系数由 `hp_heat_decay_alpha(evaluation_window)` 计算，使热度在一个评估窗口后保留 `HP_HEAT_RETAIN_RATIO`。

## EvaluationQueue 和标签

冷热标签由 `EvaluationQueue` 延迟生成：

- 每条 I/O 独立进入 `pending_queue`，不按 object 合并。
- 第 `t` 条 I/O 在 `t + HP_EVALUATION_WINDOW` 时出队并生成标签。
- 同一 object 在 `heat_map` 中共享 `heat/access_count/pending_count/last_access`。
- 到期标签使用未来窗口内新增热度：`future_heat = decayed_total_heat - decayed_entry_heat`。
- `future_heat > hot_threshold` 标记为实际热。
- 冷热样本训练权重均为 `1.0`，不再用类别权重改变 accuracy 的错误代价。
- 已评估 I/O 的 `(pred_hot_proba, label)` 进入有界监督校准窗口。校准器用冷热两组
  固定概率直方图寻找窗口 accuracy 最大的阈值；同分时选择最接近当前阈值的候选，
  再经过 EMA 和 `0.40~0.60` 边界更新有效阈值。
- 概率直方图插入和淘汰为 O(1)，每 500 个样本扫描固定 1001 bins；标签仍延迟一个
  EQ 窗口产生，校准结果只影响后续 I/O，不回写历史预测。
- future-access 报告窗口用 FIFO + Fenwick 树维护精确整数分位数；future-heat 浮点窗口仍使用 PBDS。

WT/阈值窗口维护 object 当前热度分布：

- `threshold_order_stats` 按 `log(heat) - alpha * timestamp` 保存每个 object 的
  最新热度；`threshold_order` 管理更新顺序，超过容量时淘汰最旧项。PBDS 当前不参与
  阈值决策，保留精确热度排名能力；当前模型不使用 percentile 特征。
- `otsu_histogram` 对同一批 object 的热度 score 做直方图统计；score 低于当前 `HP_OTSU_HEAT_MIN` 对应下限时物理合并到下限 bin，高于 `HP_OTSU_HEAT_MAX` 时按上限逻辑 clamp。
- Otsu 一次扫描得到候选 score、类间/总方差比，以及达到最优类间方差 `99%` 的
  候选之间会改变分类的 object 数。该数量除以总样本数形成 sharpness 的歧义比例；
  没有样本的宽空谷不会降低 sharpness。separation 和 sharpness 形成总置信度，再用
  `effective_score += 0.50 * confidence * (candidate_score - effective_score)`
  连续更新。运行期不使用固定 quantile fallback，也不硬拒绝低分离度候选。
- 样本不足保持初始阈值并标记 `initializing`；有效候选且增益非零为 `tracking`；
  平坦分布、无效候选或零置信度为 `holding`。直方图每 100 次更新，不扫描整个 TW。

LRU 只管理无 pending 的 object 状态：

- `pending_count > 0` 的 object 不在 LRU，保证 I/O 到期时状态仍存在。
- `pending_count == 0` 后 object 进入 `lru_list`。
- `lru_list.size() > HP_LRU_CAPACITY` 时删除队首 object 的 `heat_map` 状态。

## 模型和训练

模型为 `PipelineClassifier(StandardScaler, ARFClassifier)`；ARF 参数见 `hp_config.h`。

算法约束：子叶继承分裂统计；Naive Bayes 在 log 空间计算并回退类别先验；树内存限制区分节点类型；输入和参数均校验；ADWIN 使用 64 位窗口计数。回归测试见 `test_sh/hp_algorithm_probe.cc`。

后台训练流程：

- 前台 `predict()` 使用只读 `prediction_snapshot`，并把访问送入 `EvaluationQueue`。
- 到期时同步更新 I/O 级 TP/FP/TN/FN，再把样本送入后台训练队列。
- 后台线程只训练 `train_model`，不直接修改前台正在使用的快照。
- `BATCH_SIZE = 100` 同时限制一次 dequeue 数和通知间隔；批次间释放 reset 共享锁，另有 50ms 定时唤醒。
- 每累计训练 `MODEL_UPDATE_REPORT_INTERVAL = 500` 个样本后，从 `train_model` clone 一个新的 `prediction_snapshot` 并发布。
- `hp_snapshot_publish_count` 表示已发布的预测快照次数。
- 超过 `MAX_TRAIN_QUEUE_LENGTH` 时丢弃最老训练样本并增加 drop 计数。

预测快照只复制 scaler、active trees 和投票权重，不构造 warning/drift 训练状态。发布后只读。预测复用线程本地缓冲区；固定长度状态使用内嵌数组。

预测只比较 `predict_proba_one_into()` 的热概率与动态预测阈值。

并发约束：

- `train_model_mutex` 只保护后台训练模型和 clone 过程。
- `prediction_snapshot` 使用原子 `shared_ptr` 发布和读取。
- `eq_mutex` 内按 index 准备 feature、预留地址稳定的 pending slot，并评估已到期 slot；25 棵树的只读快照预测在锁外执行。
- 预测完成后短暂持有 `eq_mutex` 填充 slot；仅当满队列的最老 slot 就绪时通知等待线程，不允许结果越过旧 index。
- `reset_mutex` 覆盖预留、预测和提交，阻止 reset/disable 后旧 slot 写入新 EQ；后台训练遵守固定锁顺序。
- 8线程探针对比旧方案提速 9%~19%；顺序测试和 ASan/UBSan/TSan 通过。

## Reset 接口

单 OSD 使用 `ceph daemon osd.<id> object_hp reset`，全局使用 `ceph osd hp reset`。

`ceph daemon osd.<id> object_hp status` 是只读实时状态接口，直接读取当前 OSD 的
pending、训练队列、drop 和快照计数。异步训练线程不会持续刷新 perf counter，因此
测试结束后判断训练队列是否清空必须使用该接口，不能只依赖 MGR 中最后上报的
`hp_train_queue_length`。

reset 需要清空：

- `train_model` 和 `prediction_snapshot`，包括 scaler 和 ARF 树状态
- `EvaluationQueue` 的 pending 队列、共享热度表、LRU、阈值窗口
- 有效热阈值恢复 `HP_HEAT_INCREMENT`，Otsu 候选和置信度清零，状态恢复 `initializing`
- 后台训练队列、`model_update_train_count`、`pending_notify`、`snapshot_publish_count`
- 混淆矩阵、预测冷热计数、op 计数
- `object_hp_status` perf counter 的 U64 字段

reset 后用 OSD perf 或 `ceph osd hp status -f json-pretty` 确认计数归零。

## MGR 汇总

命令为 `ceph osd hp status`、`ceph osd hp status -f json-pretty` 和 `ceph osd hp reset`。

`osd hp status` 默认只输出所有 OSD 的 summary，不展开每个 OSD。数据来自 MGR 已收到的 OSD perf counter。

输出分组：

- `summary.osds`：`up_osds`、`reporting_osds`、`missing_osds`
- `summary.samples/confusion_matrix`：I/O 总数、已评估数、pending 数和 TP/FP/TN/FN
- `summary.heat_state`：共享热度状态、LRU、Otsu 直方图 bin/object 数、有效/候选
  热阈值、分离度、总/sharpness 置信度和各阈值状态的 OSD 数
- `summary.actual_behavior`：实际热/冷样本的平均未来访问次数、平均到期热度、热冷
  比值，以及 `p99/p95/p50` 分布。
- `summary.prediction`：accuracy/precision/recall、预测/实际热比例、当前及候选预测
  阈值、校准窗口样本数、当前及候选窗口 accuracy，以及实际热/冷样本平均预测热概率。
- `summary.training`：训练队列、丢弃样本和预测快照发布次数。
- `summary.latency`：所有上报 OSD 的预测耗时总和、次数和全局平均值
- `summary.read_ops/write_ops`：read/write 类 op 计数

汇总规则由 `DaemonServer.cc` 的 `object_hp_counter_fields` 表驱动：

- `sum`：计数字段直接求和。
- `hot_weighted` / `cold_weighted`：按实际热/冷样本数加权平均。
- `calibration_weighted`：按各 OSD 的预测阈值校准窗口样本数加权平均。
- `otsu_weighted`：按各 OSD 的 Otsu 直方图 object 数加权候选阈值和置信度。
- `osd_average`：如 `hp_hot_threshold_avg`，按上报 OSD 数简单平均，仅作参考。
- `none`：不直接聚合，由全局 TP/FP/TN/FN 重新计算。

MGR 汇总会把 OSD perf 中 `x10000` 的阈值和平均访问次数还原成浮点值；比例类指标输出为 `0~100` 的百分比数值，保留 5 位有效数字。`hp_predict_latency` 汇总各 OSD 的总纳秒数和次数，再计算全局平均，不直接平均各 OSD 的平均值。

## object_hp_status 字段

OSD perf 命令为 `ceph daemon osd.0 perf dump object_hp_status`。

字段顺序在 enum、声明、更新和 reset 中保持一致，按以下组输出：

- 样本/状态：I/O、labeled、pending、heat state、LRU、Otsu bin/object 数。
- 混淆矩阵与直观指标：TP/FP/TN/FN、accuracy、precision、recall、预测/实际热比例。
- actual behavior：热/冷样本的平均未来访问、平均热度、平均预测热概率和
  future-access/future-heat 的 `p99/p95/p50`。
- 预测阈值校准：当前/候选概率阈值、样本数和两个窗口 accuracy。
- 热度阈值：`hp_hot_threshold`、`hp_otsu_candidate_threshold`、
  `hp_otsu_separation`、`hp_otsu_confidence`、`hp_otsu_sharpness_confidence`、
  `hp_hot_threshold_method`。
- 训练、op 和延迟：队列/丢弃/快照计数、read/write 分类计数和每 1000 条有效
  I/O 采样一次的 `hp_predict_latency`。

OSD 的比例、概率、置信度和浮点平均值使用 `x10000` 整数；MGR 负责还原。
候选阈值在无有效 Otsu 候选时为 0；状态编号为 0 initializing、1 tracking、2 holding。

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
