# Ceph KernelDevice 冷热识别测试流程

本文说明当前冷热识别实验的基本流程、日志位置和输出字段含义。冷热识别逻辑位于 Ceph 用户态 BlockDevice 的 KernelDevice 后端，读写 IO 进入 `KernelDevice::_notify()` 后调用 `HeatPredictor::predict()`，并通过 `hp_status` perf counters 输出预测与训练状态。

## 1. 测试目标

测试关注两个方面：

1. IO 路径是否正常触发冷热识别。
2. `hp_status` 指标是否能反映预测数量、冷热比例、训练样本量、准确率和模型切换次数。

测试使用 `~/test/测试脚本` 下的 vdbench 配置产生负载。

## 2. 运行前检查

确认已使用包含冷热识别改动的 Ceph 二进制启动 OSD，并确认目标 OSD 能看到 `hp_status` perf counter：

```bash
ceph daemon osd.<id> perf schema hp_status
```

如果使用 admin socket 路径，也可以写成：

```bash
ceph daemon /path/to/ceph-osd.<id>.asok perf schema hp_status
```

确认 vdbench 配置中的 anchor 路径已经挂载并可访问。例如：

```text
anchor=/mnt/ikcdir/1G
```

## 3. 启动 vdbench 负载

进入测试配置目录，按实验需要运行对应 vdbench 配置：

```bash
cd ~/test/测试脚本
vdbench -f <config-file> -o <output-dir> > <log-file> 2>&1
```

建议每轮测试使用一个独立的 `RUN_ID`，格式为：

```text
YYYYMMDD_HHMMSS
```

每个 workload 独立记录开始时间、结束时间、vdbench 输出目录和日志文件。若需要批量运行，只需要保证每组 workload 的日志能按名称和时间区分。

## 4. 采集冷热识别指标

vdbench 日志只记录 workload 执行情况；冷热识别结果需要从 Ceph perf counter 采集。建议测试期间单独开一个采样进程：

```bash
OSD_ID=<id>
OUT=/home/hust/vdbench_logs/hp_status_osd${OSD_ID}_$(date +%Y%m%d_%H%M%S).log

while true; do
  date '+===== %F %T ====='
  ceph daemon osd.${OSD_ID} perf dump hp_status
  sleep 5
done >"${OUT}" 2>&1
```

如果一轮测试包含多个 OSD，需要对每个 OSD 分别采样。

## 5. vdbench 输出格式

总控日志建议记录如下信息：

```text
Run ID: 20260602_103806
Config: ~/test/测试脚本
Output: <vdbench-output-root>/20260602_103806
Logs:   <vdbench-log-root>/20260602_103806
Start:  2026-06-02 10:38:06

===== START 1G_4M_128_seq_write 2026-06-02 10:38:06 =====
Config: ~/test/测试脚本/1G文件4M块128线程顺序写.txt
Output: <vdbench-output-root>/20260602_103806/1G_4M_128_seq_write
Log:    <vdbench-log-root>/20260602_103806/1G_4M_128_seq_write.log
===== DONE  1G_4M_128_seq_write 2026-06-02 10:48:15 =====
```

其中：

- `Output`：vdbench 原始输出目录。
- `Logs`：每个 workload 的 stdout/stderr 日志目录。
- `START` / `DONE`：用于和 `hp_status` 采样日志按时间对齐。
- `<name>.log`：单个 workload 的 vdbench 运行日志。

## 6. hp_status 输出格式

`ceph daemon osd.<id> perf dump hp_status` 输出为 JSON。核心结构如下：

```json
{
  "hp_status": {
    "hp_count": 123456,
    "hp_train_total": 10000,
    "hp_hot_percent": 2345,
    "hp_actual_hot_percent": 2100,
    "hp_accuracy": 8765,
    "hp_hot_precision": 8123,
    "hp_hot_recall": 7654,
    "hp_hot_threshold": 2050000,
    "hp_train_queue_length": 0,
    "hp_swap_count": 10,
    "hp_predict_latency": {
      "avgcount": 123456,
      "sum": 1.234567
    }
  }
}
```

字段含义：

| 字段 | 含义 | 解释方式 |
| --- | --- | --- |
| `hp_count` | 已进入冷热识别的 IO 数量 | `_notify()` 每次调用递增 |
| `hp_train_total` | 已参与准确率统计的训练样本数 | 来自延迟标注后的样本 |
| `hp_hot_percent` | 预测为热的比例 | 数值除以 `10000` 得到 0 到 1 的比例 |
| `hp_actual_hot_percent` | 延迟标注后实际热样本比例 | 数值除以 `10000` |
| `hp_accuracy` | 预测准确率 | 数值除以 `10000` |
| `hp_hot_precision` | 热类 precision | 数值除以 `10000` |
| `hp_hot_recall` | 热类 recall | 数值除以 `10000` |
| `hp_hot_threshold` | 当前热度阈值 | 数值除以 `10000` 得到内部阈值 |
| `hp_train_queue_length` | 后台训练队列长度 | 长时间升高说明训练线程跟不上 |
| `hp_swap_count` | active/shadow 模型切换次数 | 每训练 `SWAP_INTERVAL` 个样本后增加 |
| `hp_predict_latency` | 预测路径耗时统计 | `sum / avgcount` 为平均耗时 |

百分比字段示例：

```text
hp_accuracy = 8765
accuracy = 8765 / 10000 = 0.8765 = 87.65%
```

## 7. 结果判读

一轮 workload 内应重点观察：

1. `hp_count` 是否随 IO 持续增长。若不增长，说明 IO 没有进入 `_notify()` 打点路径。
2. `hp_train_total` 是否延迟增长。由于 `EvaluationQueue` 需要等待样本成熟，训练样本不会和 `hp_count` 同步增加。
3. `hp_train_queue_length` 是否长期堆积。长期堆积说明后台训练吞吐不足。
4. `hp_swap_count` 是否增长。若训练样本足够但一直为 0，需要检查后台训练线程或 `SWAP_INTERVAL`。
5. `hp_hot_percent` 与 `hp_actual_hot_percent` 是否明显偏离。偏离较大说明预测分布和延迟标注分布不一致。
6. `hp_predict_latency` 是否异常升高。该指标反映 `_notify()` 中预测和入队的额外开销。

## 8. 推荐记录格式

实验报告建议记录以下信息：

```text
Run ID:
Ceph commit/build:
OSD IDs:
vdbench config dir: ~/test/测试脚本
vdbench output dir:
vdbench log dir:
hp_status sample log:

workload:
  name:
  start:
  end:
  hp_count_before:
  hp_count_after:
  hp_train_total_after:
  hp_hot_percent_after:
  hp_actual_hot_percent_after:
  hp_accuracy_after:
  hp_hot_precision_after:
  hp_hot_recall_after:
  hp_hot_threshold_after:
  hp_train_queue_length_after:
  hp_swap_count_after:
  avg_predict_latency:
```

其中 `before` / `after` 建议从 `hp_status` 采样日志中按 workload 的 `START` / `DONE` 时间截取。
