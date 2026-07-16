# MGR 冷热识别操作说明

本文只说明通过 Ceph MGR 操作和监控冷热识别模块的方法。命令作用于当前所有
`up` 状态的 OSD，不包含单 OSD Admin Socket 接口。

## 1. 输出格式

`-f` 是 `--format` 的缩写。推荐的两种格式为：

```bash
# 紧凑 JSON，适合 jq、Python 和监控脚本解析
sudo ceph osd hp status -f json

# 带缩进和换行的 JSON，适合人工查看或保存报告
sudo ceph osd hp status -f json-pretty
```

`-f json-pretty` 只改变命令输出格式，不改变冷热识别状态或测试结果。

## 2. 查看集群汇总状态

```bash
sudo ceph osd hp status -f json-pretty
```

该命令读取 MGR 已收到的 OSD perf counter，并输出所有 `up` OSD 的汇总结果。

主要分组如下：

- `summary.osds`：OSD 数量、上报情况以及模块启用状态；
- `summary.samples`：进入识别流程、已完成评价、等待评价和丢弃的 I/O；
- `summary.heat_state`：跟踪对象、LRU、Otsu 阈值及置信度；
- `summary.confusion_matrix`：TP、FP、TN 和 FN；
- `summary.actual_behavior`：热冷样本未来窗口中的实际访问行为；
- `summary.prediction`：accuracy、precision、recall 和预测阈值；
- `summary.training`：训练队列、训练丢弃和模型快照；
- `summary.latency`：预测次数和预测延迟；
- `summary.read_ops`、`summary.write_ops`：各类读写操作数量。

正常情况下，样本数量满足：

```text
hp_io_count
  = hp_labeled_io_total
  + hp_pending_io_count
  + hp_awaiting_prediction_count
  + hp_eval_drop_count
```

只查看常用字段：

```bash
sudo ceph osd hp status -f json |
  jq '.summary | {
    osds,
    samples,
    heat_state,
    confusion_matrix,
    prediction,
    training
  }'
```

周期观察：

```bash
watch -n 10 'sudo ceph osd hp status -f json-pretty'
```

## 3. 启用冷热识别

```bash
sudo ceph osd hp enable -f json-pretty
```

该命令会：

1. 启用所有当前 `up` OSD 的冷热识别模块；
2. 同时清空之前的模型、热度状态和统计数据。

适合在造数据完成、正式测试开始前执行。执行后检查：

```bash
sudo ceph osd hp status -f json |
  jq '.summary.osds'
```

期望满足：

```text
enabled_osds == up_osds
disabled_osds == 0
reporting_osds == up_osds
missing_osds == []
```

## 4. 关闭冷热识别

```bash
sudo ceph osd hp disable -f json-pretty
```

该命令会关闭所有当前 `up` OSD 的冷热识别，并清空此前的模型、热度状态和统计
数据。适合在 prepare 造数据前执行，避免造数据 I/O 污染正式测试结果。

确认关闭：

```bash
sudo ceph osd hp status -f json |
  jq '.summary.osds'
```

期望满足：

```text
disabled_osds == up_osds
enabled_osds == 0
```

## 5. 重置冷热识别状态

```bash
sudo ceph osd hp reset -f json-pretty
```

`reset` 会清空：

- I/O、标签、TP、FP、TN 和 FN 计数；
- pending、awaiting 和训练队列；
- 热度表、LRU、Otsu 状态和阈值窗口；
- 训练模型、预测快照和预测延迟；
- 读写操作计数。

`reset` 保持模块当前的启用或关闭状态不变。

MGR 会把 reset 请求发送给所有当前 `up` OSD。命令返回并不保证 MGR 已经收到
每个 OSD 新上报的零值，因此需要继续查询：

```bash
sudo ceph osd hp status -f json |
  jq '.summary | {
    osds,
    samples,
    heat_state,
    confusion_matrix,
    training
  }'
```

开始正式测试前，至少确认：

```text
reporting_osds == up_osds
missing_osds == []

hp_io_count == 0
hp_labeled_io_total == 0
hp_pending_io_count == 0
hp_awaiting_prediction_count == 0
hp_eval_drop_count == 0

hp_train_queue_length == 0
hp_train_drop_count == 0
hp_snapshot_publish_count == 0
```

## 6. 推荐操作流程

造数据前关闭模块：

```bash
sudo ceph osd hp disable -f json-pretty
```

造数据完成后启用模块并清空状态：

```bash
sudo ceph osd hp enable -f json-pretty
```

等待 MGR 汇总显示所有 OSD 已上报且计数归零，然后启动正式负载。测试过程中建议
每 10～30 秒保存一次状态：

```bash
sudo ceph osd hp status -f json-pretty > hp_status_$(date +%Y%m%d_%H%M%S).json
```

负载停止后，等待 `pending`、`awaiting_prediction` 和训练队列清空，再保存最终
汇总：

```bash
sudo ceph osd hp status -f json-pretty > hp_status_final.json
```

需要开始下一轮独立实验时，再执行一次 `reset`，并等待 MGR 观察到归零状态。

## 7. 注意事项

- MGR 状态来自 OSD 周期上报，可能短暂滞后于 OSD 内部实时状态；
- `enable` 和 `disable` 都会重置冷热识别状态；
- `reset` 不改变模块启用状态；
- `missing_osds` 非空或 `reporting_osds != up_osds` 时不应开始正式测试；
- 比较不同实验前，每轮都应独立 reset，避免上一轮模型和统计污染结果；
- `status` 是只读命令，`reset`、`enable` 和 `disable` 会改变集群冷热识别状态。
