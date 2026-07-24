# Ceph Object Heat Predictor

本文记录生产分支 `main` 的 Heat Predictor 实现。参数以
`src/heatpredictor/hp_config.h` 为准。

当前实现对每条 I/O 在预测时保存 feature 和 Otsu 热度阈值，10秒后把同一 object 在
deadline 的衰减总热度与该阈值比较生成标签。它只输出预测和统计，不执行数据迁移
或分层放置。

## 代码边界

- OSD 适配与 PerfCounters：`src/osd/ObjectHeatPredictor.*`
- OSD hook：`src/osd/PrimaryLogPG.cc`
- 算法入口：`src/heatpredictor/heat_predictor.h`
- EQ、feature、Otsu、到期堆和类型：`src/heatpredictor/hp_*.h`
- ARF、Hoeffding Tree、ADWIN 和 scaler：`src/heatpredictor/include/`
- OSD/MGR 统计契约：`src/heatpredictor/hp_telemetry.h`
- MGR 聚合：`src/mgr/ObjectHeatPredictorStatus.*`
- MGR 命令与输出：`src/mgr/MgrCommands.h`、`src/mgr/DaemonServer.cc`

Ceph 类型、op 解析和 PerfCounters 留在 OSD 适配层；`HeatPredictor` 不依赖 Ceph
object 类型。Trace、探针和 replay 只存在于 `dev`。

## Hook 与 object key

`PrimaryLogPG` 在各 op 完成原生范围规范化和参数校验后调用：

```cpp
hp_notify_osd_object_op(cct, soid, op_type);
```

纳入普通 object op：

- read：`READ`、`SYNC_READ`、`SPARSE_READ`
- write：`WRITE`、`WRITEFULL`、`WRITESAME`

`ZERO`、`TRUNCATE`、`APPEND`、omap、class、watch、cache/tier、恢复和其他管理类
op 不进入模型。`WRITESAME` 只由其内部 `WRITE` 路径通知一次。

粒度固定为 RADOS object，不再按 offset bucket 切分。key 组合：

```cpp
make_object_key(
    soid.pool,
    soid.get_hash(),
    std::hash<object_t>{}(soid.oid));
```

offset、length、operation、pool 和 hash 都不是模型 feature。offset/length 只由
Ceph 原生路径判断有效范围；operation 只用于 read/write 分类计数。

## Feature

`PredictionSample` 保存预测时快照、预测结果和10秒后评价所需字段。模型固定3维：

```text
heat_threshold_margin =
    log2(1 + heat_after_current_access)
  - log2(1 + max(heat_threshold_at_prediction, 1))

previous_access_interval_encoded =
    tracked_access_count <= 1
      ? 0
      : 1 + log2(1 + previous_access_interval_seconds)

current_heat_log2p1 = log2(1 + heat_after_current_access)
```

间隔编码中的 `0` 表示此前没有访问；已有历史从 `1` 开始。feature 在预测时生成，
到期训练复用同一快照，不能读取预测后的 object 状态。

## 固定参数

| 组件 | 当前设置 |
|---|---|
| 标签窗口 | `10s` |
| 未完成 EQ / LRU / Otsu object 上限 | 每 OSD 各 `1,000,000` |
| 单轮到期维护上限 | EQ 和 Otsu 各 `1,000` |
| 报告样本窗口 | `400,000` |
| 每次访问增量 | `100` |
| 无访问10秒后的热度保留率 | `1/5` |
| Otsu 有效总热度 | 严格大于 `20` |
| Otsu 直方图 | `2000` 个 `ln(heat)` bin，宽度 `0.01` |
| Otsu 总热度上限 | 约 `9.70e9`，超出 clamp 到末 bin |
| Otsu 最少 object | `32` |
| Otsu 重算 | 每100次有效投票更新或最长1秒 |
| Otsu 阈值 EMA | 1秒参考增益 `0.10` |
| 预测阈值 / 样本权重 | 固定 `0.50` / 冷热均为 `1.0` |
| ARF | 25棵树，3个候选 feature，seed `591422` |
| 训练批次 / 队列上限 | `100` / `200,000` |
| 快照发布 | 每500个新训练样本，或有新训练且最长1秒 |

三个100万容量都是硬上限，不在启动时完整预分配。EQ、哈希表、list 和 object 状态
按需增长；接近上限时可能消耗数百 MiB/OSD，正式大负载应同时记录 OSD RSS。

## I/O、EQ 与标签

1. 前台在 `eq_mutex` 内最多清理一批已到期样本，再更新当前 object 热度并创建
   `PredictionSample`；剩余到期项由专用线程分批追赶。
2. 每条 I/O 独立占一个稳定 EQ 节点；同一 object 的热度、访问计数、pending 数和
   上次访问时间在 `heat_map` 共享。
3. 只读模型快照的预测在 `eq_mutex` 外同步执行，随后用 opaque ticket 以 `O(1)`
   定位并提交结果。
4. 专用到期线程用 `condition_variable::wait_until()` 等待队首 deadline；空闲时
   也会到期，不依赖新 I/O 或训练线程轮询。
5. 标签线程晚于 deadline 执行时，仍把 object 热度投影到准确 deadline。
6. `total_heat_at_deadline > heat_threshold_at_prediction` 为热，否则为冷。
7. 预测和标签都完成后更新 TP/FP/TN/FN，并把样本送入后台训练队列。

预测时阈值与 feature 0 来自同一快照；窗口内后续 Otsu 更新不能反向改变标签。
全局只维护当前 Otsu 热阈值；每个 EQ item 只保存预测时的阈值数值，不维护全局
阈值历史，也不在到期时按 deadline 重新查询阈值。
EQ 中未完成节点总数（等待 deadline 的 pending 加等待预测返回的
`awaiting_prediction`）达到上限时跳过新 EQ 样本，但仍预测并更新 object 热状态，
同时增加 `hp_eval_drop_count`。不能提前评价旧样本来腾空间。

未训练森林的合法零投票按冷预测并保留样本，用未来标签启动训练。模型异常、
NaN/Inf、类别数或概率非法时按冷返回，同时取消该 EQ 样本，增加预测错误和评价
丢弃计数，不进入训练或混淆矩阵，也不影响 Ceph I/O。

## 热度、Otsu 与 LRU

热度按单调时间指数衰减，使无访问10秒后恰好保留 `1/5`。I/O 序号只用于样本顺序
和统计，不决定衰减或到期。

Otsu 对当前总热度严格大于20的 object 每个保留一票：

- `threshold_entries_by_key`：object 到 bin 和顺序节点的索引。
- `threshold_order`：按最后一次投票更新时间排列的 `std::list`，只负责100万
  object 的容量淘汰，不按热度排序。
- `threshold_expiry_heap`：每个 object 最多一个节点的索引最小堆，用于在热度衰减
  到20时精确移除投票。
- `otsu_histogram`：固定 `uint64_t[2000]`；每个 object 再次访问时替换旧投票。

直方图 score 为：

```text
ln(clamp(total_heat, 20, 9.70e9)) - decay_factor * timestamp_ns
```

时间右移导致 score 下界跨过完整 bin 时，低于新下界的 bin 物理合并到第0个 bin；
上限逻辑 clamp 到末 bin。一次 Otsu 扫描最多检查2000个 bin，与 object 数无关。
候选阈值用固定 EMA 平滑；运行期没有 quantile fallback、动态 EMA 或置信度控制。
到期投票按每轮上限分批删除；尚有已到期积压时不重算阈值，最后一批排空后立即重算，
避免用部分过期投票更新阈值。

阈值状态：

- `0 initializing`：投票不足或尚无有效阈值
- `1 tracking`：有效候选正在更新
- `2 holding`：分布暂时无法产生候选，保留上一阈值

`initializing` 只表示 Otsu 没有动态候选，不表示标签阈值不可用。构造和 reset 后
有效阈值固定为100；若工作集长期少于32个有效投票 object，状态会一直保持
`initializing`，但每条 I/O 仍保存阈值100，正常预测、生成标签、训练并进入混淆
矩阵。此时不会出现“永久不预测”，代价是阈值不能根据该稀疏工作集自适应。

LRU 只管理 `pending_evaluation_count == 0` 的 object 热度状态。再次访问时从 LRU
移回 protected；LRU 超过100万才删除队首 `heat_map` 状态。protected 状态不会因
LRU 容量被删，因此 `heat_map` 总量不严格等于 LRU 上限。

## 模型、训练与并发

模型为 `PipelineClassifier(StandardScaler, ARFClassifier)`：

- 前台只读原子发布的 `prediction_snapshot`。
- 后台线程分批训练独占的 `train_model`；队列为空时条件变量阻塞。
- 关闭时最多完成已经取出的当前批次；尚在训练队列中的后续样本直接丢弃，
  不为无持久化模型延长 OSD 退出时间。
- 训练或到期线程出现未预期异常时增加 `hp_background_error_count` 并禁用模块，
  保留 OSD 服务；OSD 会立即刷新本地 PerfCounters，MGR 在下一次 daemon report
  后看到新状态。重新执行 `enable` 会通过完整 reset 恢复。
- 发布时只 clone scaler、活动树和投票权重，不复制 warning/drift 训练状态。
- `prediction_snapshot` 发布后只读，前台预测复用线程本地缓冲区。
- ARF 遥测记录 warning、drift、后台树晋升/丢弃/训练更新和活动后台树数。

锁序和职责：

- `reset_mutex` 隔离 reset/enable/disable 与完整预测生命周期。
- OSD adapter 先取得 reset 共享锁再检查 enabled，确保 reset 前进入的 hook 要么
  完成后被 reset 清除，要么作为 reset 后的新 I/O 处理。
- `eq_mutex` 只保护 EQ、heat state、Otsu 和 LRU，不包住25棵树的预测。
- `train_model_mutex` 只保护后台模型训练和 clone。
- 到期线程遵守 `reset_mutex(shared) -> eq_mutex`；等待 deadline 时不持锁。
- 每轮最多处理1000个 EQ deadline 和1000个 Otsu expiry，随后释放 `eq_mutex`；
  `status()` 只复制状态，不执行到期、Otsu 重算或 EMA 更新。
- 标签和预测可按任意顺序完成；稳定节点和 opaque ticket 避免线性查找。

## 控制接口

```bash
# 单 OSD Admin Socket
sudo ceph daemon osd.0 object_hp status
sudo ceph daemon osd.0 object_hp reset
sudo ceph daemon osd.0 object_hp enable
sudo ceph daemon osd.0 object_hp disable
sudo ceph daemon osd.0 perf dump object_hp_status

# 集群级 MGR
sudo ceph osd hp status -f json-pretty
sudo ceph osd hp reset
sudo ceph osd hp enable
sudo ceph osd hp disable
```

enable/disable 都会完整 reset；reset 保持当前启用状态。reset 清空 EQ、heat/Otsu/LRU、
训练模型、预测快照、队列、统计、op 计数和预测延迟，并把热阈值恢复为100。
reset 返回的 `discarded_pending_io` 包含等待 deadline 和等待预测返回的全部未完成
EQ 节点。

`object_hp status` 只读实时 pending、训练和模型状态，并在返回前刷新
PerfCounters，不推进 Otsu 或到期维护；MGR 汇总可能短暂滞后。完整流程见
[MGR 操作说明](MGR_HP_OPERATIONS.md)。

## 统计与聚合

计数字段求和；热/冷行为字段按对应样本数加权；Otsu 候选按投票 object 数加权；
`hp_hot_threshold_avg` 只按上报 OSD 简单平均。MGR 从全局 TP/FP/TN/FN 重新计算指标，
不平均各 OSD 的局部指标。OSD 的浮点数用 `x10000` 整数传输，MGR 还原并保留5位
有效数字。预测延迟逐次累计，MGR 用总纳秒数除以总次数。

```text
labeled = TP + FP + TN + FN
io = labeled + pending + awaiting_prediction + eval_drop

accuracy          = (TP + TN) / labeled
balanced_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
precision         = TP / (TP + FP)
recall            = TP / (TP + FN)
pred_hot_percent  = (TP + FP) / labeled
actual_hot_percent = (TP + FN) / labeled
```

分母为0时输出0。PerfCounters 的大部分状态每1000次 I/O 或1000个到期样本刷新；
`hp_predict_latency` 对每次受支持 op 的预测累计。
