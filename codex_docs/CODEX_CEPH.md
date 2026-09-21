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
- OSD 生命周期与命令入口：`OSDService::object_hp`、`src/osd/ObjectHeatPredictor.*`
- MGR HP 命令：`src/mgr/ObjectHeatPredictorCommands.*`；`DaemonServer` 只路由命令，
  提供连接检查与 Objecter 访问。
- MGR 聚合与输出：`src/mgr/ObjectHeatPredictorStatus.*`、
  `src/mgr/ObjectHeatPredictorStatusFormatter.*`

Ceph op 解析、`hobject_t` 映射、PerfCounters 和命令注册留在 OSD 适配层。
算法目录不包含 Ceph 运行时头文件，断言使用 Release 中同样生效的 `hp_assert`；
可以只用标准 C++17/线程库编译。算法入口保留 pool/hash/name-hash 三个整数和原
`make_object_key` 映射，因此此次重构不改变已有对象键。Trace、探针、replay 和
离线分析仅由 `dev` 保留。

每个 `OSDService` 持有一个 `ObjectHeatPredictor` 适配实例；实现通过 PIMPL 隐藏，
不再使用进程级全局预测器或全局回调状态。构造后默认关闭，只建立统计/队列外壳，
不创建森林或预测快照；enable 才创建模型，首次预测才启动后台线程。
disable 清空统计并释放模型；已启动的线程继续等待，不等同于卸载模块。

正常 OSD 退出先注销管理命令并停止请求线程，在 `OSDService::shutdown()` 完成
定时器及 Objecter 回调收尾后调用模块 `shutdown()`：等待到期/训练
线程及其回调结束，最后注销并销毁 perf logger。析构也执行该清理，
支持初始化失败后的回收。shutdown 是终止操作，宿主不得与观察/命令并发调用；
Ceph 原有直接 `_exit()` 的 fast-shutdown 路径仍由进程退出回收资源。

## Hook 与 object key

`PrimaryLogPG` 在 Ceph 完成 op 参数校验和范围规范化后调用：

```cpp
osd->object_hp.observe(soid, op.op, effective_length);
```

支持 `READ`、`SYNC_READ`、`SPARSE_READ`、`WRITE`、`WRITEFULL` 和
`WRITESAME`。有效长度为 0 的事件由适配层忽略。四个 PG hook 仍位于原来的
读范围规范化、稀疏读范围规范化、WRITE 校验和 WRITEFULL 校验之后；不移到分发
入口或后端完成回调，因此这里是通过当时检查的请求观察，不代表最终 I/O 成功。

WRITESAME 由 Ceph 原流程转换为 WRITE，在该 WRITE hook 只记录一次；不再通过
`do_osd_ops` 的额外参数保留原操作编号。这会把该路径的操作统计从
`hp_op_writesame_count` 移到 `hp_op_write_count`，不改变样本总数、对象键或算法
feature。旧 `hp_op_writesame_count` 字段保留用于状态契约兼容。管理、恢复、omap、
class、watch、cache/tier 等专用路径不增加新的 hook。

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

模型采用 C4 七维 feature，前五维保持原定义：

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

新增第六、七维分别使用 τ=30s、60s 的指数慢历史。在当前访问记账前：

```text
A_tau(t) = sum(exp(-(t - prior_access_time) / tau))
T = t - first_observed_access_time
exposure = T > 0 ? 1 - exp(-T / tau) : 1
corrected_count = A_tau(t) * (10 / tau) / max(exposure, 1/4)
slow_feature_tau = log2(1 + corrected_count) - log2(1 + K_context)
```

慢历史不包含当前访问；同一时间戳下已记账的较早访问包含在内。
观测年龄为本 OSD 当前保留状态的首次访问起算，不是文件年龄。
每个对象只增加两个累加器和首次访问时间；随原有 LRU 淘汰或 reset 一起清空。
预测时保存两个 corrected_count，延迟训练使用该时刻的值。
旧热度继续保留；不启用文件上下文或热纠偏。

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
`10%`（连续指数衰减，20秒后剩1%）。它不再决定标签或 Otsu 阈值。

`heat_map` 保存共享热度、累计访问数、10秒/2秒访问数、pending 数、访问时间和 C4 慢历史状态。
三种保护计数均为0的 object 才进入 LRU；访问事件由同一个 expiry 线程按时间清理，
因此无新 I/O 时也会释放状态。LRU 超过100万才删除最久未访问状态。protected
object 不受 LRU 上限淘汰，因此 `heat_map` 总量可能高于100万。

## 模型、训练与并发

模型为 `PipelineClassifier(StandardScaler, ARFClassifier)`：

- 25棵树、7个候选 feature、seed `591422`。
- 预测阈值固定 `0.50`，冷热训练权重均为 `1.0`。
- warning 与 drift detector 固定不触发；现有树继续在线学习，但不创建后台树或替换
  当前树。
- 叶节点固定输出本叶冷热累计训练权重的比例；不再按历史正确权重切换到朴素贝叶斯。
  为保持本次快照频率实验的控制条件，叶内部原有统计更新与复制结构暂时保留。
- 前台只读原子发布的 `prediction_snapshot`，模型与成熟训练样本数作为一个整体发布。
- 当前快照训练样本数小于3000时，用 `past_10s_access_count >= K_context` 输出0/1概率；
  达到3000后使用模型概率。标签和训练仍正常进行；后台尚未发布的训练不结束保护。
  reset 同时清空训练计数、快照计数和预热输出计数。
- 后台线程独占训练模型，每批100个样本，队列上限200,000。
- 每2000个训练样本或有新训练且最长2秒发布一次预测快照。
- 关闭时最多完成已取出的当前训练批次。
- 预热结束后模型合法零投票仍按冷预测；预热期间直接使用历史规则并保留 EQ item。
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

模型预测和训练入队均在迁移锁外执行。训练模型只由训练线程修改，reset
由 `reset_mutex(unique)` 串行化。状态查询只复制状态，不推进 EQ 或阈值。

OSD 将同一个 `HeatPredictorStatus` 发布到 PerfCounters 时串行化写者，并在普通状态
字段前后写入相同的非零发布代次。PerfCounters 按字段顺序采集；MGR 只聚合首尾代次
一致的 OSD 报告，从而拒绝采集期间的新旧字段混合。发布代次是内部传输字段，不进入
用户汇总输出。预测延迟使用独立的 PerfCounters 累加器，不进入该字段组。

### 分裂候选与叶子停用

叶子类别占比超过 `max_share_to_split` 时，本轮不尝试分裂，继续累计样本。
常量特征或未通过 `min_branch_fraction` 的候选不参加 Hoeffding 比较；没有有效候选
时继续学习，不能把默认 `feature=-1` 占位值解释为停用请求。
只有显式启用 `merit_preprune` 且选中预剪枝候选时，才在分裂决策路径停用叶子。
该候选明确标记为预剪枝，信息增益为0。当前模型默认不启用它。
最大深度和内存限制引起的停用仍保留。

在 `dev` 运行 `bash test_sh/test_hp_model_regressions.sh`，覆盖暂时无候选后恢复冷热
学习、预测快照隔离、深度/内存限制、后台训练、并发统计和 Trace 回放契约。
设置 `HP_SANITIZERS=address,undefined` 可执行相同用例的 sanitizer 检查。
算法探针显式开启其训练/预测 fixture；生产默认关闭状态不变。

## 控制接口

```bash
# 单 OSD
sudo ceph daemon osd.0 object_hp status
sudo ceph daemon osd.0 object_hp reset
sudo ceph daemon osd.0 object_hp enable
sudo ceph daemon osd.0 object_hp disable
sudo ceph daemon osd.0 perf dump object_hp_status

# 集群 MGR
sudo ceph osd hp status                  # 简单摘要
sudo ceph osd hp status --detail -f json-pretty  # 完整统计
sudo ceph osd hp reset
sudo ceph osd hp enable
sudo ceph osd hp disable
```

enable/disable 都执行完整 reset；reset 保持当前启用状态。reset 清空 EQ、访问
窗口、动态 `K`、heat/LRU、模型、训练队列和统计，并恢复 `sparse/K=1`。

## 统计与聚合

MGR status 默认输出简单摘要；`--detail` 保留完整统计字段。两种模式的内容、
无样本显示和脚本兼容规则见 [MGR 操作说明](MGR_HP_OPERATIONS.md)。单 OSD
`object_hp status` 和 PerfCounters 接口保持原有格式。

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

单OSD `object_hp status` 额外报告 `hp_trained_sample_count`、
`hp_snapshot_trained_sample_count`、`hp_warmup_prediction_count`，以及当前预热门槛、
固定多数类开关、快照样本门槛和最长间隔。这些是现场诊断字段，不改变MGR简要输出。
训练完成计数可在后台继续增长；状态中的各训练字段不构成停止训练的事务快照。

单 OSD 状态还报告 `hp_feature_policy=C4`、`hp_feature_count=7`、两个慢历史时间常数
及归一化倍率上限，用于核对实际加载版本。离线候选收益不代表部署后的线上验收结果。
