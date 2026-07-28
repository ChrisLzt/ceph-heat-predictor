# Ceph Object Heat Predictor

本文记录当前 Heat Predictor 实现。常量以
`src/heatpredictor/hp_config.h` 为准；部署流程见
[Ceph 操作手册](CEPH_OPERATIONS_MANUAL.md)。

模块按 RADOS object 预测：对每条 I/O 判断同一 object 在未来
`(t, t + 10s)` 内的访问次数，是否达到该未来窗口结束时的动态阈值 `K_window`。
预测时的 `K_context` 只描述当前历史窗口，作为 feature 使用。模块只输出预测与
统计，不执行迁移或分层放置。

## 代码边界

- OSD hook 与适配：`src/osd/PrimaryLogPG.cc`、
  `src/osd/ObjectHeatPredictor.*`
- 算法入口：`src/heatpredictor/heat_predictor.h`
- EQ：`src/heatpredictor/hp_evaluation_queue.h`
- 动态 `K`：`src/heatpredictor/hp_future_access_threshold.h`
- feature、类型与统计契约：`src/heatpredictor/hp_*.h`
- ARF、Hoeffding Tree、detector 与 scaler：`src/heatpredictor/include/`
- MGR 聚合与输出：`src/mgr/ObjectHeatPredictorStatus.*`、
  `src/mgr/DaemonServer.cc`

Ceph op 解析留在 OSD 层；算法层只接收匿名 object key。Trace、探针、replay 和
离线分析仅由 `dev` 保留。

## Hook 与 object key

`PrimaryLogPG` 在 Ceph 完成 op 参数校验和范围规范化后调用：

```cpp
hp_notify_osd_object_op(cct, soid, op_type);
```

支持 `READ`、`SYNC_READ`、`SPARSE_READ`、`WRITE`、`WRITEFULL` 和
`WRITESAME`。`WRITESAME` 只记录一次。管理、恢复、omap、class、watch、
cache/tier 等路径不进入模型。

粒度固定为 RADOS object，不按 offset 切分：

```cpp
make_object_key(
    soid.pool,
    soid.get_hash(),
    std::hash<object_t>{}(soid.oid));
```

offset、length、operation、pool 和 hash 不是模型 feature；operation 只用于
read/write 计数。

## 标签与 Feature

每条 I/O 创建一个独立 EQ item。当前 I/O 和恰好位于
deadline 的访问都不计入未来窗口：

```text
future_access_count =
    tracked_access_count_at_deadline
  - tracked_access_count_after_current_access

actual_hot = future_access_count >= K_window_at_deadline
```

`K_context` 在预测时从严格过去10秒 object 计数直方图读取；`K_window` 在样本
到期后，从实际未来窗口末端的同类直方图读取。deadline 按1ms划分微批；同一微批
使用最后一个 deadline 的一次原始 Otsu 结果，阈值时间误差小于1ms，避免逐 I/O
扫描直方图。

模型固定五维 feature：

```text
past_access_count_margin =
    log2(1 + past_10s_access_count)
  - log2(1 + K_context)

previous_access_interval_encoded =
    first_access ? 0 : 1 + log2(1 + previous_interval_seconds)

current_heat_log2p1 = log2(1 + current_heat)

projected_count_margin =
    log2(1 + short_2s_access_count / 2 * 10)
  - log2(1 + K_context)

short_access_count_log2p1 =
    log2(1 + short_2s_access_count)
```

`past_10s_access_count` 是当前 item 到来前，严格
`(prediction_time - 10s, prediction_time)` 内同一 object 的访问数。
`short_2s_access_count` 是 `(prediction_time - 2s, prediction_time)` 内同一 object
的历史访问数。两个计数都不包含当前 I/O。feature 在预测时生成；后台训练复用该
快照，不读取未来状态。

## 动态访问阈值 K

阈值模块对 object 等权，而训练和混淆矩阵对 I/O 等权：

- 独立访问事件队列维护严格滚动10秒窗口，不依赖 EQ 是否接纳样本、预测是否成功。
- 每个当前计数大于0的 object 在直方图中恰好投一票；计数变化只移动该票。
- 计数归零立即删除票，窗口内未访问 object 不参与 Otsu。
- 当前 item 先读取入队前的 `past_10s_access_count` 和 `K_context`，再记入窗口。
- EQ 到期前先移除左边界事件，再强制计算一次 `K_window` 并生成标签。

正观察映射到固定直方图：

```text
score = log2(1 + current_past_10s_access_count)
score_min = 1.0
bin_width = 0.01
bin_count = 2000
```

最大可表示约 `2^21 - 1 = 2,097,151` 次/10秒，超出值进入最后一个 bin。
Otsu 扫描最多 2000 个 bin，与 object 数无关。

阈值状态：

- `sparse`：正 object 少于32个、少于两个非空 bin 或无有效分割，发布 `K=1`。
- `tracking`：Otsu 分割有效，直接发布原始动态 `K`。

常规路径每100个 object 票变化或最长1秒重算；EQ 标签微批会在直方图发生变化时
强制重算。批量过期只移动 object 票，批末统一维护，避免中途重复扫描。阈值不使用
EMA、holding 或固定 quantile，Otsu score 向上取整转换为整数 `K`。

## EQ、热度与 LRU

1. 前台取得 `eq_mutex` 后先清空所有已到期批次，避免当前 I/O 泄漏进旧标签。
2. 清理到期的10秒/2秒访问事件，读取 `K_context` 和 feature，再记录当前访问并
   尝试创建稳定 EQ 节点。
3. 在 `eq_mutex` 外使用只读模型快照同步预测，再用 opaque ticket `O(1)` 提交。
4. 专用线程按 deadline 唤醒，每批最多处理1000个 item，并按1ms微批计算
   `K_window`；无新 I/O 时也会完成标签。
5. 微批先更新严格窗口直方图，再用同一 `K_window` 标注；标签与预测都完成后才
   进入混淆矩阵与训练队列。

EQ pending 与 awaiting-prediction 合计达到100万时，新样本不再入 EQ，并增加
`hp_eval_drop_count`；不能提前评价旧样本腾空间。前台必须在当前 I/O 记账前追平
全部已到期 item，后台维护保持有界批次。

严格访问窗口独立保存最近10秒的每条访问事件，因此其空间复杂度是
`O(最近10秒I/O数)`，不受 EQ 容量限制。这里不设置硬上限；丢弃窗口事件会使
`past_10s_access_count`、`K_context` 和 `K_window` 失真。

热度只作为第三个 feature 和 object 状态保留。每次访问增加100，无访问10秒后保留
`1/5`。它不再决定标签或 Otsu 阈值。

`heat_map` 保存共享热度、累计访问数、10秒/2秒访问数、pending 数和上次访问时间。
三种保护计数均为0的 object 才进入 LRU；访问事件由同一个 expiry 线程按时间清理，
因此无新 I/O 时也会释放状态。LRU 超过100万才删除最久未访问状态。protected
object 不受 LRU 上限淘汰，因此 `heat_map` 总量可能高于100万。

## 模型、训练与并发

模型为 `PipelineClassifier(StandardScaler, ARFClassifier)`：

- 25棵树、5个候选 feature、seed `591422`。
- 预测阈值固定 `0.50`，冷热训练权重均为 `1.0`。
- warning 与 drift detector 固定不触发；现有树继续在线学习，但不创建后台树或替换
  当前树。
- 前台只读原子发布的 `prediction_snapshot`。
- 后台线程独占训练模型，每批100个样本，队列上限200,000。
- 每500个训练样本或有新训练且最长1秒发布一次预测快照。
- 关闭时最多完成已取出的当前训练批次。
- 未训练森林的合法零投票按冷预测，但仍保留 EQ item 以启动训练。
- 非法概率或模型异常按冷返回并取消 EQ 样本，不影响已记录的10秒访问事件，也不
  影响 Ceph I/O。
- 后台异常会禁用模块、清空训练队列并刷新状态；enable 通过完整 reset 恢复。

锁顺序为 `reset_mutex(shared) -> evaluation_transition_mutex -> eq_mutex` 或
`evaluation_stats_mutex`；`eq_mutex` 和 `evaluation_stats_mutex` 互不嵌套。
`evaluation_transition_mutex` 只覆盖一次样本状态迁移：I/O 计数和 EQ 状态更新完成后，
立即提交该批样本的混淆矩阵与报告统计。状态查询持有同一把迁移锁后依次复制 EQ 和
统计状态，因此单 OSD 快照始终满足：

```text
hp_io_count
  = hp_labeled_io_total
  + hp_pending_io_count
  + hp_awaiting_prediction_count
  + hp_eval_drop_count
```

模型预测、Trace 转换和训练入队均在迁移锁外执行。训练模型只由训练线程修改，reset
由 `reset_mutex(unique)` 串行化。状态查询只复制状态，不推进 EQ 或阈值。

OSD 将同一个 `HeatPredictorStatus` 发布到 PerfCounters 时串行化写者，并在普通状态
字段前后写入相同的非零发布代次。PerfCounters 按字段顺序采集；MGR 只聚合首尾代次
一致的 OSD 报告，从而拒绝采集期间的新旧字段混合。发布代次是内部传输字段，不进入
用户汇总输出。预测延迟使用独立的 PerfCounters 累加器，不进入该字段组。

## 控制接口

```bash
# 单 OSD
sudo ceph daemon osd.0 object_hp status
sudo ceph daemon osd.0 object_hp reset
sudo ceph daemon osd.0 object_hp enable
sudo ceph daemon osd.0 object_hp disable
sudo ceph daemon osd.0 perf dump object_hp_status

# 集群 MGR
sudo ceph osd hp status -f json-pretty
sudo ceph osd hp reset
sudo ceph osd hp enable
sudo ceph osd hp disable
```

enable/disable 都执行完整 reset；reset 保持当前启用状态。reset 清空 EQ、访问
窗口、动态 `K`、heat/LRU、模型、训练队列和统计，并恢复 `sparse/K=1`。

## 统计与聚合

OSD 暴露当前实际生效的 `K`、阈值状态、正 object 数、归零次数、上限 clamp 数及
sparse 样本数。MGR 输出上报 OSD 的 `K` 最小值、最大值、平均值及 sparse/tracking
OSD 数。

ARF adaptation 字段为兼容现有状态契约而保留；当前实现不启用 warning、后台树和
drift replacement，相关计数正常情况下均为0。

计数字段求和，行为均值按对应样本数加权。MGR 从全局 TP/FP/TN/FN 重新计算：

```text
labeled = TP + FP + TN + FN
accuracy          = (TP + TN) / labeled
balanced_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
precision         = TP / (TP + FP)
recall            = TP / (TP + FN)
pred_hot_percent  = (TP + FP) / labeled
actual_hot_percent = (TP + FN) / labeled
```

分母为0时输出0。预测延迟逐次累计。大部分 PerfCounters 每1000次 I/O 或到期样本
刷新；仅阈值定时维护引起的状态变化会主动刷新一次。

冷热标签的未来访问数分位数采用容量40万的滑动固定对数直方图近似维护：
`log2(1+x)`、bin width `0.01`、2101个 bin。每个保留样本只保存一个16位 bin
下标，更新为 `O(1)`；状态查询最多扫描2101个 bin。
