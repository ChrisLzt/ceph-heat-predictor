# CODEX Context: ICFS/IDFS Heat Predictor Port

本文档记录 Ceph object-layer heat predictor 向 `/home/hust/icfs` 的迁移约定。Ceph 源实现见 `CODEX_CEPH.md`，测试流程见 `CODEX_TEST.md`。

## 项目判断

`/home/hust/icfs` 是基于 Ceph 17.2.6 深度改名后的 IDFS/ICFS 工程。当前迁移目标是传统 OSS 路径，不是 crimson。

主要命名变化：

| Ceph | ICFS/IDFS |
| --- | --- |
| `src/osd` | `src/oss` |
| `OSD` | `OSS` |
| `PrimaryLogPG` | `IdfsPrimaryLogPG` |
| `OSDOp` | `OSSOp` |
| `ceph_osd_op` | `idfs_oss_op` |
| `CEPH_OSD_OP_*` | `IDFS_OSS_OP_*` |
| `CephContext` | `IdfsContext` |
| `PerfCounters` | `IdfsPerfCounters` |
| `ceph-osd` | `idfs-oss` |

## Hook 路径

传统 OSS 路径核心文件：

- `src/oss/PrimaryLogPG.cc`
- `src/oss/MergePG.cc`
- `src/oss/OSS.cc`
- `src/oss/ObjectHeatPredictor.*`

已覆盖或需要保持覆盖的 hook：

- `IdfsPrimaryLogPG::do_oss_ops()`
- `IdfsMergePG::do_oss_ops()`
- `IdfsMergePG::do_oss_ops_journal_mode()`

推荐位置与 Ceph 相同：op 规范化之后、主 `switch (op.op)` 之前。

原因：

- 已有 object 上下文和 `soid`。
- op 语义已规范化。
- 能覆盖一条请求中的每个子操作。
- 不依赖读写执行结果。

## Object Key 和字段

ICFS object 标识仍应组合多个字段，不能只用分布 hash。

建议 key 组成：

- `soid.pool`
- `soid.get_hash()`
- object name hash
- `object_offset >> HP_BUCKET_SHIFT`

常用字段：

- `soid.pool`
- `soid.oid`
- `soid.oid.name`
- `soid.get_hash()`
- `op.extent.offset`
- `op.extent.length`
- `op.writesame.offset`
- `op.writesame.length`

如果 op 是否使用 extent 不确定，优先使用 ICFS 自带判断函数，例如 `idfs_oss_op_uses_extent(op.op)`。

## 记录的操作

第一阶段只记录用户数据读写。

read-like：

- `IDFS_OSS_OP_READ`
- `IDFS_OSS_OP_SYNC_READ`
- `IDFS_OSS_OP_SPARSE_READ`
- `IDFS_OSS_OP_SCATTER_READ`

write-like：

- `IDFS_OSS_OP_WRITE`
- `IDFS_OSS_OP_WRITEFULL`
- `IDFS_OSS_OP_WRITESAME`
- `IDFS_OSS_OP_SCATTER_WRITE`

暂不纳入：omap、class、watch/notify、scrub/recovery/backfill、cache/tier 管理类 op、纯元数据 op。

## 适配层边界

ICFS 侧适配层应集中在：

```text
src/oss/ObjectHeatPredictor.h
src/oss/ObjectHeatPredictor.cc
```

适配层职责：

- 过滤普通数据读写 OSS op。
- 提取 offset、length、operation。
- 把 `idfs_hobject_t + idfs_oss_op` 转为 `HeatPredictor::predict()` 输入。
- 注册并更新 `object_hp_status` perf counter。
- 统计 ICFS 特有 scatter read/write op。

`src/heatpredictor/` 保持通用，不引入 ICFS 类型。

## HeatPredictor 同步要求

ICFS 的 `src/heatpredictor/` 应与 Ceph 当前实现保持同步，重点包括：

- 4KB bucket：`HP_BUCKET_SHIFT = 12`
- 队列上限：`EvaluationQueue max_size = 5000`
- hot quantile：`HP_HOT_QUANTILE = 0.80`
- `GaussianSplitter` 默认 `n_split = 5`
- `lhs_dist/rhs_dist` 初始化为 `0`
- active/shadow swap 的锁注释和锁顺序
- `hot_accuracy`、precision、recall 等指标由 TP/FP/TN/FN 计算

需要同步的目录：

```text
src/heatpredictor/
src/heatpredictor/include/
```

## Perf Counter

section 名保持：

```text
object_hp_status
```

核心字段应与 Ceph 同名，便于测试脚本复用：

- `hp_count`
- `hp_labeled_total`
- TP/FP/TN/FN 四个计数
- `hp_pred_hot_percent`
- `hp_eval_pred_hot_percent`
- `hp_eval_actual_hot_percent`
- `hp_hot_accuracy`
- `hp_hot_precision`
- `hp_hot_recall`
- `hp_train_queue_length`
- `hp_swap_count`
- `hp_dequeue_waiting_count`
- `hp_dequeue_max_size_count`
- read/write op 计数

ICFS 额外字段：

- `hp_op_scatter_read_count`
- `hp_op_scatter_write_count`

注意：`object_hp_status` 是单个 OSS daemon 的本地状态。多 OSS 汇总需要 mgr/vas 或外部脚本轮询所有 OSS 后聚合。

## 构建和验证

构建目标：

```bash
cd /home/hust/icfs
ninja -C build idfs-oss
```

不要直接套用 Ceph 的安装命令：

```bash
sudo install build/bin/ceph-osd /usr/bin/ceph-osd
```

ICFS 的服务名、二进制路径、admin socket 路径都以当前部署环境为准。

验证顺序：

1. `idfs-oss` 能正常启动。
2. admin socket 能 dump perf schema。
3. `object_hp_status` section 存在。
4. vdbench 测试时 `hp_count` 增长。
5. `hp_labeled_total`、TP/FP/TN/FN、precision、recall 有变化。
6. reset 后计数归零。

如果 `hp_count` 不增长，优先检查当前池是否走未覆盖的新 PG 路径，或 workload 是否只产生了未纳入统计的管理类 op。

## 风险点

- 新 PG 类型或旁路路径可能漏 IO。
- `soid.get_hash()` 不是唯一 object id，不能单独当 key。
- 内部管理 op 混入训练会污染标签。
- 多 OSS 环境不能直接把本地 perf counter 当全局指标。
- ICFS 命名与 Ceph 不完全一致，迁移时不要机械替换。
