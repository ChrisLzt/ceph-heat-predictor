# Cold-to-Hot Crossing Feature Design

## 目标

在不增加模型维数、不修改标签、Otsu、预测阈值和 ARF 参数的前提下，让第四个 feature
专门表达“本次访问前仍冷，但近期访问趋势可能使 object 在10秒标签截止时转热”。

## 现有问题

1. 第二次访问产生第一个有效间隔时，fast/slow rate 仍从0做 EWMA，短间隔访问被严重
   低估；模型直到更多访问后才看到明显升温信号。
2. `expected_future_heat_margin` 同时包含当前热度和未来新增热度，与已有当前热度 feature
   高度重复，不能明确区分“已经热”和“正在转热”。
3. 下降趋势会直接下调预测访问率，但突然停止访问无法从预测时刻观测；这会给模型加入
   不可靠的热转冷推断。

## 设计

### 访问率初始化

- 第一次访问没有间隔，fast/slow rate 均为0。
- 第二次访问直接令 fast/slow rate 等于观测率 `1/dt`。
- 第三次及以后继续使用现有时间感知 EWMA：

  ```text
  alpha = 1 - exp(-dt/tau)
  rate  = (1-alpha)*old_rate + alpha/dt
  ```

### 非对称升温速率

只奖励可观测的上升趋势，不用负趋势主动推断未来停止访问：

```text
activation_rate = fast_rate + beta * max(0, fast_rate-slow_rate)
```

保留 `tau_fast=2s`、`tau_slow=10s`、`beta=0.5`，隔离本次语义修复的影响。

### 冷转热 crossing margin

当前存储的是本次访问增加热度后的值。由于每次访问固定增加 `heat_increment`：

```text
pre_access_heat = max(0, heat_after_current_access - heat_increment)
```

若 `pre_access_heat` 已高于当前阈值，第四维输出0。否则按10秒指数衰减计算：

```text
retained              = exp(-lambda*T)
effective_time        = (1-retained)/lambda
required_heat         = max(0, threshold-current_heat*retained)
expected_added_heat   = heat_increment*activation_rate*effective_time
cold_to_hot_margin    = log2p1(expected_added_heat)-log2p1(required_heat)
```

正值表示预计新增热度足以填补缺口，负值表示不足。第一至第三维继续提供当前热状态和访问
间隔，第四维只补充升温证据。

## 数据兼容性

特征数量仍为4，但第四维语义发生变化。Trace 配置哈希必须加入 feature schema version，
避免旧 Trace 被误当作兼容输入。

## 验证

1. 合成测试覆盖首次有效间隔、上升/下降趋势和 crossing margin 正负方向。
2. Probe、ASan/UBSan、性能探针、Trace 回放和 Python 分析测试全部通过。
3. 部署后每个正式负载只运行一次，保存10秒 MGR 快照和最终状态。
4. 开启 completed-evaluation Trace，单独计算访问前冷且 deadline 热的 recall/precision，
   同时报告整体 Accuracy、Balanced Accuracy、Precision、Recall 和延迟。
