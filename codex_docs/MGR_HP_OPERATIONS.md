# MGR 冷热识别操作说明

本文只说明通过 MGR 操作和监控所有 `up` OSD 上的 Heat Predictor。单 OSD 实时接口
见 [实现说明](CODEX_CEPH.md)，集群部署见
[操作手册](CEPH_OPERATIONS_MANUAL.md)。

## 状态

```bash
# 默认：五行关键摘要
sudo ceph osd hp status

# 简单模式的结构化输出
sudo ceph osd hp status -f json

# 详细模式：完整调试信息，默认 json-pretty
sudo ceph osd hp status --detail

# 实验采集、reset 检查及报告脚本使用完整 JSON
sudo ceph osd hp status --detail -f json
```

`--detail` 选择内容范围，`-f` 只控制格式。简单模式默认显示：

- 已启用、已上报、up OSD 数；
- 已评估 I/O 数；
- Accuracy（首位）、Precision、Recall；
- 预测热比例与实际热比例；
- 累计平均预测器调用耗时，单位微秒，不是端到端 I/O 延迟。

无已评估样本或指标分母为0时，对应指标显示 `N/A`；无计时样本时耗时显示
`N/A`。简单 JSON 对应值为 `null`，延迟路径为
`summary.latency.hp_predict_latency.avgtime_us`。

无 up OSD、上报缺失、已上报 OSD 中有未启用模块，或预测错误、后台错误、评估丢弃、
训练丢弃计数非零时，简单文本追加 `ALERT`，简单 JSON 增加 `summary.alerts`。
错误及丢弃计数是自 reset 以来的累计值；队列非零本身不视为异常。
上报缺失时，指标仅覆盖本次接受的 OSD 报告。

详细模式保留原有 JSON 字段和层级，包括纳秒延迟字段和无样本时的零值约定。
旧脚本必须改用 `--detail`；仅指定 `-f json` 仍然是简单内容。
MGR 汇总 OSD 周期上报的 PerfCounters，详细模式的分组为：

- `osds`：up、reporting、enabled、disabled 和 missing OSD；
- `samples`：I/O、已完成标签、pending、awaiting 和 drop；
- `heat_state`：heat/LRU、Otsu 投票、阈值和状态；
- `confusion_matrix`：TP、FP、TN 和 FN；
- `actual_behavior`：实际热/冷样本的未来访问行为；
- `prediction`：accuracy、balanced accuracy、precision、recall 和预测/实际热比例；
- `training`、`model_adaptation`、`latency`、`read_ops`、`write_ops`。

`hp_background_error_count > 0` 表示训练或到期线程发生异常，Heat Predictor 会自动
转为 disabled 以避免影响 OSD。OSD 会立即刷新本地 PerfCounters，MGR 在下一次
daemon report 后看到新状态；排查后执行 `ceph osd hp enable` 完整 reset 并恢复。

`dev` 构建还会输出 `trace`，不属于 `main` 的稳定统计契约。完整字段和聚合公式见
[实现说明](CODEX_CEPH.md)。

`model_adaptation` 的主动漂移检测当前关闭，六个计数正常均为0，仅在详细模式保留。
`actual_behavior` 的 `*_osd_p99/p95/p50_weighted_avg` 是各 OSD 访问次数分位数的
加权平均，不是集群分位数，也不是预测延迟分位数。`hp_heat_state_peak_count` 是各
OSD 历史状态数量峰值之和，不表示集群同时占用峰值。

正常情况下：

```text
hp_io_count
  = hp_labeled_io_total
  + hp_pending_io_count
  + hp_awaiting_prediction_count
  + hp_eval_drop_count
```

Heat Predictor 在单 OSD 内以同一个状态迁移边界发布上述计数，因此一次 OSD 上报内部
满足该等式。MGR 聚合的是各 OSD 最近一次 daemon report；不同 OSD 的上报时刻可以
不同，但每份已接收的 OSD 状态自身必须一致。OSD PerfCounters 使用首尾发布代次
检测采集期间的新旧字段混合；代次缺失或不一致的报告不会参与汇总，该 OSD 在本次
查询中计入 `missing_osds`，等待下一次完整 daemon report。
升级该协议时必须同时部署并重启 OSD 和 MGR，旧 OSD 因缺少代次字段会显示为 missing。

每个 OSD 的 EQ 硬上限约束
`hp_pending_io_count + hp_awaiting_prediction_count`，等待预测返回的样本也占容量。

常用查询：

```bash
sudo ceph osd hp status --detail -f json |
  jq '.summary | {
    osds,
    samples,
    heat_state,
    confusion_matrix,
    prediction,
    training
  }'
```

## 控制

| 命令 | 作用 |
|---|---|
| `sudo ceph osd hp enable` | 启用所有 up OSD，并完整 reset |
| `sudo ceph osd hp disable` | 禁用所有 up OSD，并完整 reset |
| `sudo ceph osd hp reset` | 保持启用状态，清空模型、队列、热度和统计；丢弃数包含 pending 和 awaiting |

造数据前使用 `disable`，正式测试前使用 `enable`。命令返回只表示请求已发送，MGR
可能尚未收到 OSD 的新状态。

## 开始测试前

先检查：

```bash
sudo ceph osd hp status --detail -f json |
  jq '.summary | {
    osds,
    samples,
    confusion_matrix,
    training
  }'
```

至少满足：

```text
reporting_osds == up_osds
missing_osds == []
enabled_osds == up_osds

hp_io_count == 0
hp_labeled_io_total == 0
hp_pending_io_count == 0
hp_awaiting_prediction_count == 0
hp_eval_drop_count == 0
hp_train_queue_length == 0
```

若 MGR 尚未归零，优先用实时接口确认 OSD：

```bash
sudo ceph daemon osd.0 object_hp status
sudo ceph daemon osd.1 object_hp status
```

## 标准流程

```bash
# 造数据
sudo ceph osd hp disable -f json-pretty
./prepare_data.sh

# 开始独立测试
sudo ceph osd hp enable -f json-pretty
# 等待上述归零条件满足
./run_test.sh

# 测试期间每 10～30 秒采集
sudo ceph osd hp status --detail -f json-pretty \
  > hp_status_$(date +%Y%m%d_%H%M%S).json

# 测试结束；等待 pending、awaiting 和训练队列排空
sudo ceph osd hp status --detail -f json-pretty > hp_status_final.json
```

不同实验之间执行 `reset` 并重新确认归零。`status` 是只读命令；`reset`、`enable`
和 `disable` 会改变集群状态。`missing_osds` 非空或
`reporting_osds != up_osds` 时不要开始正式测试。
