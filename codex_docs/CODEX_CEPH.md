# CODEX Context: Ceph Object Heat Predictor

本文档记录 Ceph object heat predictor 的当前实现；参数以 `src/heatpredictor/hp_config.h` 为准。

当前 H1 实现命名为 `v1.2`：每条 I/O 在预测时保存热度阈值，10秒后使用该阈值生成
总热度标签。与 `v1.1` 相比，模型、feature、Otsu 和预测阈值不变，只消除标签对未来
阈值更新的依赖。

2026-07-24 五负载单轮对照中，`v1.2` 的加权 Accuracy/Balanced Accuracy 为
`87.76%/88.27%`，相对 `v1.1` 为 `-0.14/-0.03 pp`；Precision `+0.23 pp`、
Recall `-0.61 pp`。该改动按因果标签语义保留，不宣称提高准确率。完整结果见
`/home/chris/ceph-test/new_workload/hp_runs/reports/20260724_004418_v1_2_prediction_threshold_snapshot_all5/REPORT.md`。

## 目标和边界

当前原型在 OSD 内对 RADOS object 做在线冷热识别，只输出统计和预测结果，不执行数据迁移或分层放置。

- Ceph 适配层：`src/osd/ObjectHeatPredictor.*`
- 算法入口：`src/heatpredictor/heat_predictor.h`
- 算法组件：`hp_config.h`、`hp_types.h`、`hp_features.h`、`hp_evaluation_queue.h`、
  `hp_expiry_heap.h`、`hp_score_otsu_histogram.h`
- 统计契约：`src/heatpredictor/hp_telemetry.h`
- Hook 入口：`src/osd/PrimaryLogPG.cc`
- MGR 汇总：`src/mgr/ObjectHeatPredictorStatus.*`；命令适配位于
  `src/mgr/DaemonServer.cc`、`src/mgr/MgrCommands.h`

Ceph 类型、op 解析和 perf counter 细节位于适配层；`HeatPredictor` 核心类只包含
可复用算法。修改这些边界和统计字段时应遵守仓库根目录 `AGENTS.md` 的工程约束。

## OSD Hook

`PrimaryLogPG` 在各 op 完成 Ceph 原生范围规范化和参数校验后调用
`hp_notify_osd_object_op(cct, soid, op_type)`。初始化由 `OSD::final_init()`
调用 `init_osd_object_hp_status(cct, whoami)`；perf section 为 `object_hp_status`。

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

`PredictionSample` 保存预测时快照及评估所需字段：`io_sequence`、
`object_key_hash`、`heat_after_current_access`、
`heat_label_threshold_at_prediction`、`tracked_access_count`、
`time_since_previous_access_ns`、`predicted_hot_probability` 和
`predicted_label`。

模型特征数量以 `NUM_FEATURES` 为准，默认实现为3个：

1. `heat_threshold_margin = log2p1(heat_after_current_access) - log2p1(max(heat_label_threshold_at_prediction, 1))`
2. `previous_access_interval_encoded = tracked_access_count <= 1 ? 0 : 1 + log2p1(time_since_previous_access_ns / 1s)`
3. `current_heat_log2p1 = log2p1(heat_after_current_access)`

其他候选 feature 已从生产实现移除；模型固定使用上述三维输入。

`log2p1(x)` 表示 `std::log2(1 + x)`。访问间隔编码中，`0` 明确表示没有历史访问，
已有历史的有效间隔从 `1` 开始，避免首次访问和同一时刻的重复访问混淆。

`operation`、`size`、`offset`、`pool` 和两个 object hash 不作为模型特征；`operation` 只在 OSD hook 层用于 op 计数，`size/offset` 不进入预测器。

每条 I/O 的 feature 是入队和预测时的瞬时快照；出队训练时复用该快照生成训练输入，只用未来窗口结果生成 label，避免把预测之后的状态泄露进 feature。

## 当前参数

- ARF：`N_MODELS=25`、`MAX_FEATURES=NUM_FEATURES`、`SEED=591422`、`GRACE_PERIOD=100`、`LAMBDA=4`、`DELTA=0.001`、`TAU=0.05`、`MAX_SHARE_TO_SPLIT=0.99`、`MIN_BRANCH_FRACTION=0.01`。
- 预测：固定阈值 `0.50`，不维护动态校准状态。训练直接使用单位样本权重，冷热标签
  均为 `1.0`，不存在类别专用权重。
- 容量：`HP_FUTURE_LABEL_WINDOW_NS=10s`、
  `HP_PENDING_EVALUATION_CAPACITY=1000000`、`HP_HEAT_DECAY_HORIZON_NS=10s`；另有 `HP_LRU_CAPACITY=1000000`、
  `HP_HEAT_LABEL_THRESHOLD_OBJECT_CAPACITY=1000000`、
  `HP_REPORT_SAMPLE_WINDOW_CAPACITY=400000`。
- 热度：`HEAT_INCREMENT=100.0`、无访问10秒后保留 `1/5`。
- 热阈值：初始值 `100`；Otsu 按 object 维护最新总热度的时间归一化 score，使用2000个
  固定 bin、宽度 `0.01`。下限是一次访问热度经过一个衰减周期后的值，当前为
  `100 * 0.20 = 20`；只有当前总热度严格大于20的 object 才保留一票，等于或低于20
  时立即移除。上限按相同2000个对数 bin 推导，当前约为 `9.70e9`。最少32个 object
  投票，每100次
  有效更新或最迟1秒重算一次。有效阈值以每1秒参考区间固定 EMA `0.10` 跟踪候选，
  实际 gain 只按经过时间复合。
- 阈值状态：`0 initializing`、`1 tracking`、`2 holding`。

EQ 和 LRU 的100万是按 OSD 设置的硬上限，不会在启动时完整预分配；EQ list、
`heat_map` 和 LRU 节点按实际使用增长。LRU 只限制无 pending 的空闲
object，不是 `heat_map` 总量的严格上限。时间域热度使用直接指数计算，不再按 LRU
容量分配衰减查表。EQ 真正接近100万样本或 LRU 接近100万 object 时，节点、哈希表和
状态表合计可能达到数百 MiB/OSD，正式大负载测试必须同时记录 OSD RSS。

热度衰减系数由 `hp_heat_decay_log_factor_per_ns(HP_HEAT_DECAY_HORIZON_NS)` 计算，使
object 在10秒无访问后保留 `HP_HEAT_RETAINED_AFTER_DECAY_HORIZON=1/5`。EQ、热度、
Otsu score 原点和访问距离全部使用 `steady_clock`。I/O 序号只保留用于样本顺序和统计。
Otsu 维护 object 当前
总热度的时间归一化 score，空闲时会随热度衰减平移直方图下界。

## EvaluationQueue 和标签

冷热标签由 `EvaluationQueue` 延迟生成：

- 每条 I/O 独立进入 `pending_evaluations`，不按 object 合并。
- 每条 I/O 使用 `std::chrono::steady_clock` 记录入队时间，经过 10 秒后出队并生成
  标签。专用到期线程使用 `condition_variable::wait_until()` 睡眠到队首的准确
  deadline，因此空闲负载也能到期，不再依赖训练线程轮询。
- 预测在 `eq_mutex` 外执行。稳定 list 节点分别记录 `prediction_complete` 和
  `label_complete`：到期游标只按时间生成标签，不等待预测，也不会阻止后续 deadline。
  两侧均完成时才更新统计并进入训练；标签先完成的节点计入
  `hp_awaiting_prediction_count`，但不再占用 EQ 准入容量。`EvaluationQueue` 以只能移动的
  opaque ticket 封装稳定 list iterator，两种完成顺序都可 O(1) 定位并删除节点，调用方
  无法访问队列内部容器。
- 前台 I/O 在同一个 `eq_mutex` 临界区内先清理到期样本，再更新当前 I/O 的热状态和
  feature，防止 deadline 之后的访问污染旧样本标签。
- pending 上限为 `1000000`。达到上限时跳过新 I/O 的评价样本，但仍完成预测并更新
  object 热状态；`hp_eval_drop_count` 记录该情况，不允许为了腾位置提前评价旧样本。
- 同一 object 在 `heat_map` 中共享 `heat/tracked_access_count/pending_evaluation_count/last_access_time_ns`。
- 到期标签只使用该 object 在固定 deadline 的衰减后总热度。即使后台线程晚于 deadline
  执行，热度也只投影到该 deadline，线程调度延迟不会改变标签；不维护窗口新增热度。
- `total_heat_at_deadline > heat_label_threshold_at_prediction` 标记为实际热。阈值与
  feature 0 在预测时取自同一快照；10秒窗口内的 Otsu 更新不能反向修改该样本的标签。
- 冷热样本训练权重均为 `1.0`，不再用类别权重改变 accuracy 的错误代价。
- 当前直接使用固定预测阈值 `0.50`，不保留监督校准样本、候选阈值或动态范围。
- future-access 报告窗口不限制合法访问次数；整数和浮点报告窗口都用 FIFO + PBDS
  维护有界、精确分位数。

阈值窗口维护每个 object 的最新总热度投票及其更新顺序：

- `threshold_entries_by_key` 是 object key 到 Otsu bin 和顺序节点的哈希索引；
  `threshold_order` 是按最近一次热度投票更新时间排列的 `std::list<uint64_t>`。object
  更新时用 `splice` 把原节点移到队尾，超过容量时从队首淘汰。因此它是更新顺序，不是按
  热度排序，也不是 LRU 热度状态表。
- `threshold_expiry_heap` 是 object key 到精确下限到期时间的索引最小堆；每个 object
  最多一个节点。访问刷新和任意删除为 `O(log n)`，到期时只查看堆顶，不保留失效事件。
- percentile feature 及其 PBDS 顺序统计树已经删除；Otsu 删除投票只依赖
  `threshold_entries_by_key` 保存的 bin 和 `threshold_order` 节点迭代器。
- `otsu_histogram` 使用固定 `uint64_t[2000]` 聚合
  `score = ln(clamp(total_heat, 20, 9.70e9)) - decay_factor * timestamp_ns`，bin 宽
  为 `0.01`。clamp 只作用于已经满足 `total_heat > 20` 的投票；低于等于20的 object
  不进入第0个 bin。每个 object 只保存一个投票；再次访问时直方图和顺序链表更新为
  O(1)，精确到期堆刷新为 O(log n)。
- score 下界随单调时间右移。只有下界跨过完整 bin 时才物理维护：低于新下界的 bin
  合并到第0个 bin；超过热度上限的值逻辑 clamp 到最后一个 bin。固定容量使 Otsu 扫描
  与 object 数量无关，最多扫描2000个 bin。
- Otsu 一次扫描选择类间方差最大的候选 score，再用
  `effective_score += gain(dt) * (candidate_score - effective_score)` 更新，其中
  `gain(1s)=0.10`。每次更新把当前热度阈值和候选投影到同一时刻的 score 空间，EMA 后
  再转回热度；已发布热度阈值在两次更新间保持不变。运行期不使用固定 quantile fallback，
  也不维护额外的分离度或 sharpness 置信度状态。
- object 访问会替换其总热度投票；阈值重算和 EMA 使用当前单调时间，晚到处理不能令
  EMA 时间回退。预测样本保存准入时的阈值，后续发布的阈值只影响新样本。
- object 投票不足保持初始阈值并标记 `initializing`；有效候选为 `tracking`；平坦分布或无效
  候选为 `holding`。直方图每100次有效 object 投票更新或最多1秒重算，计算只扫描固定2000个
  bin，不扫描 WT 或 object 表。

LRU 只管理不再受评价队列或访问时间窗口保护的 object 状态：

- `pending_evaluation_count` 大于0的 object 不在 LRU，保证标签到期前状态存在。
- pending 计数为0后 object 进入 `lru_list`；再次访问时从 LRU 移回活跃状态。
- `lru_list.size() > HP_LRU_CAPACITY` 时删除队首 object 的 `heat_map` 状态。
- `hp_protected_heat_state_count = hp_heat_state_count - hp_lru_count`；另记录 reset 以来
  每个 OSD 的 heat state 峰值和 LRU 淘汰数。MGR 对三项均求和，其中峰值是各 OSD
  独立峰值之和，不表示同一时刻的全局峰值。

## 模型和训练

模型为 `PipelineClassifier(StandardScaler, ARFClassifier)`；ARF 参数见 `hp_config.h`。

算法约束：子叶继承分裂统计；Naive Bayes 在 log 空间计算并回退类别先验；树内存限制区分节点类型；输入和参数均校验；ADWIN 使用 64 位窗口计数。

后台训练流程：

- 前台 `predict()` 使用只读 `prediction_snapshot`，并把访问送入 `EvaluationQueue`。
- 预测保持同步执行。未训练森林的合法全零投票按冷预测并保留 EQ 样本，用未来标签
  启动训练。模型异常、类别数错误、NaN/Inf 或越界概率按冷 fallback 返回，同时取消
  对应 EQ 样本、增加 `hp_predict_error_count` 和 `hp_eval_drop_count`，不进入指标、预测
  校准或训练；异常不传播到 Ceph I/O。OSD hook 另保留最外层异常隔离。
- 到期时同步更新 I/O 级 TP/FP/TN/FN，再把样本送入后台训练队列。
- 后台线程只训练 `train_model`，不直接修改前台正在使用的快照。
- `BATCH_SIZE = 100` 限制一次 dequeue 数；批次间释放 reset 共享锁，
  没有训练样本时由训练队列条件变量阻塞等待。
- 每累计训练 `MODEL_UPDATE_REPORT_INTERVAL = 500` 个样本，或已有新训练且距上次发布
  达到1秒时，从 `train_model` clone 新的 `prediction_snapshot` 并发布。
- `hp_snapshot_publish_count` 表示已发布的预测快照次数。
- ARF 训练模型累计 warning、drift、后台树晋升/丢弃/训练更新，并记录当前活动后台树数；
  预测快照不共享或修改这些训练遥测。
- 超过 `MAX_TRAIN_QUEUE_LENGTH` 时丢弃最老训练样本并增加 drop 计数。

预测快照只复制 scaler、active trees 和投票权重，不构造 warning/drift 训练状态。发布后只读。预测复用线程本地缓冲区；固定长度状态使用内嵌数组。

预测只比较 `predict_proba_one_into()` 的热概率与固定阈值 `0.50`。

并发约束：

- `train_model_mutex` 只保护后台训练模型和 clone 过程。
- `prediction_snapshot` 使用原子 `shared_ptr` 发布和读取。
- `eq_mutex` 内通过 `EvaluationQueue::begin_prediction()` 按单调时间准备 feature、
  预留 opaque ticket 对应的 pending slot，并摘取已到期 slot；25 棵树的只读快照预测
  在锁外执行。
- 预测完成后短暂持有 `eq_mutex` 填充 slot；标签到期和预测返回通过两个完成标志会合，
  标签游标不会被未完成预测阻塞。
- 专用到期线程检查 EQ deadline 时遵守 `reset_mutex(shared) -> eq_mutex` 锁序；等待
  deadline 时不持有这两把锁。若编译期重新启用访问速率 feature，同一线程也处理5秒/10秒
  访问事件。维护队列从空变为非空、reset、enable/disable
  和析构都会通过独立条件变量及 wake sequence 唤醒它，避免丢失通知。
- `reset_mutex` 覆盖预留、预测和提交，阻止 reset/disable 后旧 slot 写入新 EQ；后台训练遵守固定锁顺序。
- 性能和并发正确性以当前 `hp_performance_probe`、ASan/UBSan 和 TSan 结果为准。

## Reset 接口

单 OSD 使用 `ceph daemon osd.<id> object_hp reset`，全局使用 `ceph osd hp reset`。

`ceph daemon osd.<id> object_hp status` 是只读实时状态接口，直接读取当前 OSD 的
pending、训练队列、drop 和快照计数。后台到期线程每累计完成1000个 EQ deadline
刷新一次 perf counter；不足1000个的尾部状态和异步训练队列仍可能尚未上报。因此
测试结束后判断训练队列是否清空必须使用该接口，不能只依赖 MGR 中最后上报的
`hp_train_queue_length`。

reset 需要清空：

- `train_model` 和 `prediction_snapshot`，包括 scaler 和 ARF 树状态
- `EvaluationQueue` 的 pending 队列、共享热度表、LRU、阈值窗口
- 有效热阈值恢复 `HP_HEAT_INCREMENT`，Otsu 候选清零，状态恢复 `initializing`
- 后台训练队列、`model_update_train_count`、`pending_notify`、`snapshot_publish_count`
- ARF warning/drift 和后台树生命周期遥测
- 混淆矩阵、预测冷热计数、op 计数
- `object_hp_status` perf counter 的 U64 字段

reset 后用 OSD perf 或 `ceph osd hp status -f json-pretty` 确认计数归零。

## MGR 汇总

命令为 `ceph osd hp status`、`ceph osd hp status -f json-pretty` 和 `ceph osd hp reset`。

`osd hp status` 默认只输出所有 OSD 的 summary，不展开每个 OSD。数据来自 MGR 已收到的 OSD perf counter。

输出分组：

- `summary.osds`：`up_osds`、`reporting_osds`、`missing_osds`
- `summary.samples/confusion_matrix`：I/O 总数、已评估数、pending 数、评价丢弃数
  和 TP/FP/TN/FN
- `summary.heat_state`：共享热度状态、LRU、Otsu 非空 bin/保留 object 数、有效/候选
  热阈值和各阈值状态的 OSD 数
- `summary.actual_behavior`：实际热/冷样本的平均未来访问次数、热冷比值，以及各 OSD
  `p99/p95/p50` 按对应冷热样本数加权后的平均值。后者使用
  `*_osd_pXX_weighted_avg` 命名，不表示合并全体样本后的全局分位数。
- `summary.prediction`：accuracy/balanced accuracy/precision/recall、预测/实际热比例、
  预测错误数，以及实际热/冷样本平均预测热概率。固定阈值 `0.50` 不重复输出。
- `summary.training`：训练队列、丢弃样本和预测快照发布次数。
- `summary.model_adaptation`：ARF warning、drift、后台树晋升/丢弃/训练更新累计值，
  以及当前活动后台树数；前五项从 reset 起累计，最后一项是当前 gauge。
- `summary.latency`：所有上报 OSD 的逐次预测耗时总和、次数和全局平均值
- `summary.read_ops/write_ops`：read/write 类 op 计数

OSD/MGR 共享的字段名、原始单位和聚合规则由
`src/heatpredictor/hp_telemetry.h` 定义；纯聚合逻辑位于
`src/mgr/ObjectHeatPredictorStatus.cc`，`DaemonServer` 只负责读取 daemon state 和输出
JSON：

- `sum`：计数字段直接求和。
- `hot_weighted` / `cold_weighted`：按实际热/冷样本数加权平均。
- `otsu_weighted`：按各 OSD 的 Otsu 直方图保留 object 数加权候选阈值。
- `osd_average`：如 `hp_hot_threshold_avg`，按上报 OSD 数简单平均，仅作参考。
- `none`：不直接聚合，由全局 TP/FP/TN/FN 重新计算。

MGR 汇总会把 OSD perf 中 `x10000` 的阈值和平均访问次数还原成浮点值；比例类指标输出为 `0~100` 的百分比数值，保留 5 位有效数字。`hp_predict_latency` 汇总各 OSD 的总纳秒数和次数，再计算全局平均，不直接平均各 OSD 的平均值。

## object_hp_status 字段

OSD perf 命令为 `ceph daemon osd.0 perf dump object_hp_status`。

当前字段按以下组输出：

- 样本/状态：I/O、labeled、pending、评价丢弃、heat state、LRU、Otsu 非空 bin/保留
  object 数。
- 混淆矩阵与直观指标：TP/FP/TN/FN、accuracy、balanced accuracy、precision、recall、
  预测/实际热比例。MGR 从所有 OSD 汇总后的 TP/FP/TN/FN 计算全局 balanced accuracy，
  不平均各 OSD 的局部值。
- actual behavior：热/冷标签样本的平均未来访问次数、平均预测热概率，以及访问次数
  分布的 `p99/p95/p50`。相关字段使用
  `hp_hot_labeled_sample_*` / `hp_cold_labeled_sample_*` 前缀；热冷比值为
  `hp_future_access_count_hot_cold_ratio`。
- 预测：输出 `hp_predict_error_count`；固定预测阈值 `0.50` 不作为运行状态输出。
- 热度阈值：`hp_hot_threshold`、`hp_otsu_candidate_threshold`、
  `hp_hot_threshold_method`。
- 训练、op 和延迟：队列/丢弃/快照计数、read/write 分类计数和每 1000 条有效
  I/O 采样一次的 `hp_predict_latency`。

OSD 的比例、概率和浮点平均值使用 `x10000` 整数；MGR 负责还原。
候选阈值在无有效 Otsu 候选时为 0；状态编号为 0 initializing、1 tracking、2 holding。

MGR 汇总公式：

```text
hp_io_count = hp_labeled_io_total + hp_pending_io_count
            + hp_awaiting_prediction_count + hp_eval_drop_count
hp_labeled_io_total     = TP + FP + TN + FN
eval_pred_hot_percent   = (TP + FP) / hp_labeled_io_total * 100
eval_actual_hot_percent = (TP + FN) / hp_labeled_io_total * 100
hot_accuracy            = (TP + TN) / hp_labeled_io_total * 100
hot_precision           = TP / (TP + FP) * 100
hot_recall              = TP / (TP + FN) * 100
```

分母为 0 时指标输出 0。评估时重点看 accuracy、precision、recall、actual hot percent 和 confusion matrix。
