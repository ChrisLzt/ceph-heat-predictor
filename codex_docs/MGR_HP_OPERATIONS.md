# MGR 冷热识别操作说明

本文只说明通过 MGR 操作和监控所有 `up` OSD 上的 Heat Predictor。单 OSD 实时接口
见 [实现说明](CODEX_CEPH.md)，集群部署见
[操作手册](CEPH_OPERATIONS_MANUAL.md)。

## 状态

```bash
# 适合脚本解析
sudo ceph osd hp status -f json

# 适合人工查看
sudo ceph osd hp status -f json-pretty
```

`-f` 只控制输出格式。MGR 汇总 OSD 周期上报的 PerfCounters，主要分组为：

- `osds`：up、reporting、enabled、disabled 和 missing OSD；
- `samples`：I/O、已完成标签、pending、awaiting 和 drop；
- `heat_state`：heat/LRU、Otsu 投票、阈值和状态；
- `confusion_matrix`：TP、FP、TN 和 FN；
- `actual_behavior`：实际热/冷样本的未来访问行为；
- `prediction`：accuracy、balanced accuracy、precision、recall 和预测/实际热比例；
- `training`、`model_adaptation`、`latency`、`read_ops`、`write_ops`。

`dev` 构建还会输出 `trace`，不属于 `main` 的稳定统计契约。完整字段和聚合公式见
[实现说明](CODEX_CEPH.md)。

正常情况下：

```text
hp_io_count
  = hp_labeled_io_total
  + hp_pending_io_count
  + hp_awaiting_prediction_count
  + hp_eval_drop_count
```

常用查询：

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

## 控制

| 命令 | 作用 |
|---|---|
| `sudo ceph osd hp enable` | 启用所有 up OSD，并完整 reset |
| `sudo ceph osd hp disable` | 禁用所有 up OSD，并完整 reset |
| `sudo ceph osd hp reset` | 保持启用状态，清空模型、队列、热度和统计 |

造数据前使用 `disable`，正式测试前使用 `enable`。命令返回只表示请求已发送，MGR
可能尚未收到 OSD 的新状态。

## 开始测试前

先检查：

```bash
sudo ceph osd hp status -f json |
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
sudo ceph osd hp status -f json-pretty \
  > hp_status_$(date +%Y%m%d_%H%M%S).json

# 测试结束；等待 pending、awaiting 和训练队列排空
sudo ceph osd hp status -f json-pretty > hp_status_final.json
```

不同实验之间执行 `reset` 并重新确认归零。`status` 是只读命令；`reset`、`enable`
和 `disable` 会改变集群状态。`missing_osds` 非空或
`reporting_osds != up_osds` 时不要开始正式测试。
