# Ceph Object Heat Predictor

本文记录当前 Heat Predictor 实现。常量以
`src/heatpredictor/hp_config.h` 为准；部署流程见
[Ceph 操作手册](CEPH_OPERATIONS_MANUAL.md)。

模块按 RADOS object 预测：对每条 I/O 判断同一 object 在未来
`(t, t + 10s)` 内的访问次数是否达到预测时保存的动态阈值 `K`。模块只输出预测与
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

每条 I/O 创建一个独立 EQ item，并保存预测时的 `K`。当前 I/O 和恰好位于
deadline 的访问都不计入未来窗口：

```text
future_access_count =
    tracked_access_count_at_deadline
  - tracked_access_count_after_current_access

actual_hot = future_access_count >= K_at_prediction
```

预测与标签始终回答同一个问题；后续 `K` 的变化不会改写已入队样本。

模型固定五维 feature：

```text
past_access_count_margin =
    log2(1 + past_10s_access_count)
  - log2(1 + K_at_prediction)

previous_access_interval_encoded =
    first_access ? 0 : 1 + log2(1 + previous_interval_seconds)

current_heat_log2p1 = log2(1 + current_heat)

projected_count_margin =
    log2(1 + short_2s_access_count / 2 * 10)
  - log2(1 + K_at_prediction)

short_access_count_log2p1 =
    log2(1 + short_2s_access_count)
```

`past_10s_access_count` 是当前 item 入队前，同一 object 尚未到期的 EQ item 数。
`short_2s_access_count` 是 `(prediction_time - 2s, prediction_time)` 内同一 object
的历史访问数。两个计数都不包含当前 I/O。feature 在预测时生成；后台训练复用该
快照，不读取未来状态。

## 动态访问阈值 K

阈值模块对 object 等权，而训练和混淆矩阵对 I/O 等权：

- 每个 object 只保留当前正 EQ 计数的一票。
- EQ 成功入队后发布增加后的计数；到期或取消后发布减少后的计数。
- 新的正计数替换旧票；计数归零时删除旧票。
- 当前 item 先读取入队前的计数和 `K`，不会把自己计入 feature。
- 未来标签完成只生成标签和训练样本，不再更新阈值。
- object 票数只受 `1,000,000` 硬上限约束，没有 TTL。

正观察映射到固定直方图：

```text
score = log2(1 + current_past_10s_eq_count)
score_min = 1.0
bin_width = 0.01
bin_count = 2000
```

最大可表示约 `2^21 - 1 = 2,097,151` 次/10秒，超出值进入最后一个 bin。
Otsu 扫描最多 2000 个 bin，与 object 数无关。

阈值状态：

- `sparse`：正 object 少于32个、少于两个非空 bin 或无有效分割，发布 `K=1`。
- `tracking`：Otsu 候选有效，发布动态 `K`。
- `holding`：有效候选暂时消失，最多保留上一个 `K` 10秒，之后回到 `K=1`。

每100个 object 票变化或最长1秒重算。第一个有效候选直接发布；后续对 score 使用
按经过时间归一化的 EMA，1秒参考增益为 `0.10`，再向上取整转换为整数 `K`。

## EQ、热度与 LRU

1. 前台取得 `eq_mutex` 后先清空所有已到期批次，避免当前 I/O 泄漏进旧标签。
2. 清理到期的2秒短窗口事件，更新 object 共享访问计数和热度，读取 `K` 与
   feature，并创建稳定 EQ 节点。
3. 在 `eq_mutex` 外使用只读模型快照同步预测，再用 opaque ticket `O(1)` 提交。
4. 专用线程按 deadline 唤醒，每批最多处理1000个 item；无新 I/O 时也会完成标签。
5. EQ 入队、到期和取消同步更新 object 阈值票；标签与预测都完成后才进入混淆
   矩阵与训练队列。

EQ pending 与 awaiting-prediction 合计达到100万时，新样本不再入 EQ，并增加
`hp_eval_drop_count`；不能提前评价旧样本腾空间。前台必须在当前 I/O 记账前追平
全部已到期 item，后台维护保持有界批次。

热度只作为第三个 feature 和 object 状态保留。每次访问增加100，无访问10秒后保留
`1/5`。它不再决定标签或 Otsu 阈值。

`heat_map` 保存共享热度、累计访问数、2秒访问数、pending 数和上次访问时间。
pending 与2秒访问数均为0的 object 才进入 LRU；短窗口事件由同一个 expiry 线程
按时间清理，因此无新 I/O 时也会释放状态。LRU 超过100万才删除最久未访问状态。
protected object 不受 LRU 上限淘汰，因此 `heat_map` 总量可能高于100万。

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
- 非法概率或模型异常按冷返回、取消 EQ 样本并同步减少 object 阈值票，不影响
  Ceph I/O。
- 后台异常会禁用模块、清空训练队列并刷新状态；enable 通过完整 reset 恢复。

锁顺序为 `reset_mutex(shared) -> eq_mutex`。树预测不持有 `eq_mutex`；训练模型和
clone 只受 `train_model_mutex` 保护。状态查询只复制状态，不推进 EQ 或阈值。

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

enable/disable 都执行完整 reset；reset 保持当前启用状态。reset 清空 EQ、动态 `K`、
heat/LRU、模型、训练队列和统计，并恢复 `sparse/K=1`。

## 统计与聚合

OSD 暴露当前/候选 `K`、阈值状态、正 object 数、零观察数、上限 clamp 数及
sparse/holding 样本数。MGR 输出上报 OSD 的 `K` 最小值、最大值、平均值及各状态
OSD 数；候选 `K` 按正 object 数加权。

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
