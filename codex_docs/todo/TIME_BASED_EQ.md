# 时间型评价窗口

## 目标

将当前“经过 10000 个 OSD 本地 I/O 后评价”改为“经过固定时间后评价”。默认窗口
暂定 10 秒，使标签含义不再随 IOPS、OSD 数量和负载速率变化。

按 I/O 窗口的完整基线保存在 `eq-io-window-baseline` 分支。

## 推荐语义

对时刻 `t` 到达的每条 I/O，在 `(t, t + 10s]` 内累计同一 object 的未来衰减热度，
到期后使用动态热阈值生成标签。不是判断 `t + 10s` 瞬间是否恰好发生访问。

不能只替换 EQ 的出队条件。以下语义必须同时从 I/O 序号改为单调时间：

- 热度衰减和 `last_access_distance`；
- EQ 中的未来热度及过去窗口访问数；
- access acceleration 短窗口，默认由 2500 I/O 改为 2.5 秒；
- Otsu score 使用的 timestamp 和动态边界。

I/O 总序号继续保留，只用于唯一标识、计数和稳定排序。时间来源使用
`std::chrono::steady_clock`，不得使用可能回拨的系统时间。

## 队列与并发

- 待评价项按到期时间有序进入 `deque`，无需使用堆。
- 每 OSD 的待评价数量约为 `OSD IOPS × 10s`，必须配置硬上限。
- 达到上限时丢弃新评价样本但继续更新 object 热状态，不能提前出队，否则会改变标签
  时间范围；增加评价准入丢弃计数。
- 到期维护只在锁内摘取已到期项，标签统计和训练入队尽量移出 `eq_mutex` 临界区。
- 空闲期间可以延迟到下一次 I/O 清理；若要求状态准时归零，再增加轻量定时唤醒，不能
  让模型训练阻塞到期维护。

## 初始参数

```text
HP_EVALUATION_WINDOW_SECONDS = 10
HP_ACCESS_ACCELERATION_WINDOW_SECONDS = 2.5
HP_HEAT_RETAIN_RATIO = 0.1（经过 10 秒后保留 10%）
HP_HEAT_DECAY_ALPHA = ln(0.1) / 10s
HP_EVALUATION_MAX_PENDING = 根据单 OSD 峰值 IOPS 确定
```

队列上限不能直接沿用 10000。它只够覆盖每 OSD 约 1000 IOPS 的 10 秒窗口；正式取值
前必须采集各负载单 OSD 峰值 IOPS。

## 输出

保留 pending 数量，并增加：

- 评价样本准入丢弃数；
- 已到期样本数；
- 到期处理延迟的平均值和 p99；
- 实际评价窗口时长的平均值和 p99。

## 验证

1. 使用可注入的单调时钟做 L1 测试，禁止依赖真实 `sleep(10)`。
2. 覆盖 10 秒边界、同一 object 多次访问、空闲到期、队列上限和 reset。
3. L2 验证 enable/disable/reset 后没有遗留定时项或训练样本。
4. L3 使用相同访问模式和不同 IOPS 各运行一次；时间版的标签比例应比 I/O版稳定。
5. 报告准确率、precision、recall、预测/实际热比例、队列峰值、丢弃数和延迟。

## 非目标

首轮不同时修改feature集合、森林参数、热数据权重、Otsu策略或预测阈值策略，避免无法
归因测试差异。
