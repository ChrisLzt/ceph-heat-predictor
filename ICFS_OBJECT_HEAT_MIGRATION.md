# ICFS Object Heat Predictor Migration Notes

本文档记录对 `/home/hust/icfs` 的初步调研结果，目标是后续把当前 Ceph OSD object 层冷热识别模块迁移到 ICFS/IDFS 中。

当前只做调研和迁移定位，不修改 `/home/hust/icfs`。

## 代码来源和整体判断

`/home/hust/icfs` 是一个基于 Ceph 17.2.6 深度改名和魔改后的工程。

已确认信息：

- 顶层工程名为 `idfs`。
- 顶层版本号为 `17.2.6`。
- 远端仓库为内部 SSH 仓库。
- 当前工作区干净。

主要命名变化：

| Ceph | ICFS/IDFS |
| --- | --- |
| `src/osd` | `src/oss` |
| `OSD` | `OSS` |
| `PrimaryLogPG` | `IdfsPrimaryLogPG` |
| `MOSDOp` | `IdfsMOSSOp` |
| `MOSDOpReply` | `IdfsMOSSOpReply` |
| `OSDOp` | `OSSOp` |
| `ceph_osd_op` | `idfs_oss_op` |
| `CEPH_OSD_OP_*` | `IDFS_OSS_OP_*` |
| `CephContext` | `IdfsContext` |
| `PerfCounters` | `IdfsPerfCounters` |
| `PerfCountersBuilder` | `IdfsPerfCountersBuilder` |
| `ceph-osd` target | `idfs-oss` target |

## 传统 OSS 路径

当前应优先迁移到传统 OSS 路径，而不是 crimson 路径。

传统路径核心文件：

- `src/oss/PrimaryLogPG.cc`
- `src/oss/PrimaryLogPG.h`
- `src/oss/OSS.cc`
- `src/oss/OSS.h`
- `src/oss/oss_perf_counters.cc`
- `src/oss/oss_perf_counters.h`

构建目标：

```bash
ninja -C build idfs-oss
```

`src/crimson/oss` 也存在，但除非明确部署 crimson OSS，否则不要优先改这条路径。

## IdfsPrimaryLogPG

`IdfsPrimaryLogPG` 是 ICFS 中常规 replicated PG 的主要实现，对应 Ceph 的 `PrimaryLogPG`。

关键函数：

```cpp
int IdfsPrimaryLogPG::do_oss_ops(OpContext *ctx, std::vector<OSSOp>& ops)
```

这个函数对应当前 Ceph 热识别版本里的：

```cpp
int PrimaryLogPG::do_osd_ops(OpContext *ctx, vector<OSDOp>& ops)
```

它的职责是遍历一条客户端请求中的 `OSSOp` 子操作，按 `op.op` 分派到 read/write/omap/class/tier/watch 等实际处理逻辑。

推荐 hook 点：

1. 在 `for (auto p = ops.begin(); ...)` 循环内部。
2. 取得 `OSSOp& oss_op` 和 `idfs_oss_op& op` 之后。
3. 在 `ZERO -> TRUNCATE` 等 op 规范化之后。
4. 在真正执行 op 的主 `switch (op.op)` 之前。

原因：

- 这个位置已经有 object 上下文和 `soid`。
- `op.op` 已经过必要的规范化，避免把同一个真实操作按多个语义统计。
- 还没有执行读写逻辑，不会受到执行结果和后续副作用影响。
- 能覆盖同一个 RADOS object 请求里的每个子操作。

## IdfsMergePG

`IdfsMergePG` 是 ICFS 中新增或深度魔改的 PG 类型，继承自 `IdfsPrimaryLogPG`：

```cpp
class IdfsMergePG: public IdfsPrimaryLogPG
```

它和普通 `IdfsPrimaryLogPG` 的主要区别是引入了 append-write / merge / journal-mode 相关路径。代码中可以看到它持有或使用：

- `IdfsMergeWAL`
- `IdfsMergeMemTable`
- `IdfsMergeCache`
- `IdfsMergeIndex`
- `MergeGC`
- L2P / P2L 相关结构
- journal mode transaction path

`IdfsMergePG` 自己重写了：

```cpp
int IdfsMergePG::do_oss_ops(OpContext *ctx, std::vector<OSSOp>& ops)
```

并且还有：

```cpp
int IdfsMergePG::do_oss_ops_journal_mode(OpContext *ctx, std::vector<OSSOp>& ops)
```

因此如果 ICFS 当前测试池或未来生产池走 MergePG/append-write 路径，只 hook `IdfsPrimaryLogPG::do_oss_ops()` 会漏掉 MergePG 的 IO。

迁移建议：

- 把热识别通知逻辑封装为公共 helper。
- 在 `IdfsPrimaryLogPG::do_oss_ops()` 中调用。
- 在 `IdfsMergePG::do_oss_ops()` 中也调用。
- 如果 journal mode 确认会绕过普通 `do_oss_ops()`，再检查并补充 `do_oss_ops_journal_mode()`。

## 对象和操作字段

ICFS 中 object 标识仍然有类似 Ceph 的 `idfs_hobject_t`：

```cpp
const idfs_hobject_t& soid = oi.soid;
```

可用字段：

- `soid.pool`
- `soid.oid`
- `soid.oid.name`
- `soid.get_hash()`
- `soid.get_key()`
- `soid.get_namespace()`

`soid.get_hash()` 是对象分布相关 hash，不是全局唯一 object id。后续 key 仍建议组合：

- pool
- `soid.get_hash()`
- object name hash
- bucket

`OSSOp` 定义在 `src/oss/oss_types.h`：

```cpp
struct OSSOp {
  idfs_oss_op op;
  sobject_t soid;
  idfs::buffer::list indata, outdata;
  errorcode32_t rval = 0;
  ...
};
```

`idfs_oss_op` 定义在 `src/include/uds.h`，读写常见字段和 Ceph 基本一致：

```cpp
op.extent.offset
op.extent.length
op.writesame.offset
op.writesame.length
```

可用 `idfs_oss_op_uses_extent(op.op)` 判断该 op 是否使用 extent。

## 建议记录的操作

第一阶段只记录和数据热度最直接相关的读写操作。

read-like：

- `IDFS_OSS_OP_READ`
- `IDFS_OSS_OP_SYNC_READ`
- `IDFS_OSS_OP_SPARSE_READ`
- 可选：`IDFS_OSS_OP_SCATTER_READ`

write-like：

- `IDFS_OSS_OP_WRITE`
- `IDFS_OSS_OP_WRITEFULL`
- `IDFS_OSS_OP_WRITESAME`
- 可选：`IDFS_OSS_OP_SCATTER_WRITE`

暂时不建议纳入第一阶段：

- omap 操作
- class 调用
- watch/notify
- cache/tier 管理操作
- scrub/recovery/backfill 内部操作
- pure metadata op

原因是这些 op 不一定代表用户数据访问热度，混入后容易污染标签。

## Perf Counter 状态输出

当前 Ceph 版本输出 section 为：

```text
object_hp_status
```

迁移到 ICFS 时可以保留这个 section 名，便于测试脚本复用。

但类型需要改成 ICFS 的 perf counter 类型：

```cpp
IdfsPerfCounters *object_hp_logger = nullptr;
IdfsPerfCountersBuilder b(cct, "object_hp_status", first, last);
```

初始化位置建议放在：

```cpp
void OSS::final_init()
```

该函数里已经注册了 OSS admin socket 命令，适合作为 object heat predictor 状态初始化入口。

ICFS 已有 OSS perf counter 构建函数：

```cpp
IdfsPerfCounters *build_oss_logger(IdfsContext *cct)
```

位于：

- `src/oss/oss_perf_counters.cc`
- `src/oss/oss_perf_counters.h`

可选做法：

1. 单独在 `PrimaryLogPG.cc` 中创建 `object_hp_status`，和当前 Ceph 版本一致。
2. 或者把 object heat perf counter 的 enum 和 builder 放入 `oss_perf_counters.*`，风格更贴近 ICFS。

短期迁移推荐第 1 种，改动范围更小。

## HeatPredictor 模块迁移

当前 Ceph 版本中热识别模块已经独立在：

```text
src/heatpredictor/
```

迁移到 ICFS 时建议保持同样路径：

```text
/home/hust/icfs/src/heatpredictor/
```

需要拷贝：

- `heat_predictor.h`
- `include/ARFClassifier.h`
- `include/Classifier.h`
- `include/GaussianSplitter.h`
- `include/HoeffdingTree*.h/.tpp`
- `include/HoeffdingTreeClassifier*.h/.tpp`
- `include/HotList.h`
- `include/Metrics.h`
- `include/PipelineClassifier.h`
- `include/StandardScaler.h`
- `include/Transformer.h`
- `include/TreeBase*.h/.tpp`
- `include/utils.h`

如果 `#include "heatpredictor/heat_predictor.h"` 无法直接找到，再给 `oss` 目标补 include path。

## 适配层建议

不要把大量 ICFS 类型判断塞进 `HeatPredictor` 核心类。建议在 `PrimaryLogPG.cc` / `MergePG.cc` 附近保留一层轻量 wrapper：

```cpp
static inline bool hp_track_oss_op(uint16_t op);
static inline int hp_oss_op_type(uint16_t op);
static inline void hp_oss_op_extent(const idfs_oss_op& op, uint64_t *offset, uint64_t *length);
static inline void hp_notify_oss_object_op(IdfsContext *cct, const idfs_hobject_t& soid, const idfs_oss_op& op);
```

这样 `HeatPredictor` 继续只关心通用字段：

- index
- operation
- size
- pool
- object hash
- object name hash
- offset
- access count seed

## 迁移步骤建议

1. 拷贝 `src/heatpredictor/` 到 `/home/hust/icfs/src/heatpredictor/`。
2. 在 `src/oss/PrimaryLogPG.cc` 中 include `heatpredictor/heat_predictor.h`。
3. 添加 ICFS 版 wrapper，将 `idfs_hobject_t + idfs_oss_op` 转成 `HeatPredictor::predict()` 输入。
4. 在 `IdfsPrimaryLogPG::do_oss_ops()` 的规范化之后、主 switch 之前调用 wrapper。
5. 在 `src/oss/PrimaryLogPG.h` 声明初始化函数，例如 `init_oss_object_hp_status(IdfsContext *cct)`。
6. 在 `OSS::final_init()` 中调用初始化函数。
7. 编译 `idfs-oss`。
8. 用 admin socket 或 perf dump 检查 `object_hp_status` 是否出现。
9. 如果 MergePG 池有 IO 但指标不增长，再把同一 wrapper 插到 `IdfsMergePG::do_oss_ops()`。
10. 如果 journal mode 绕过普通路径，再补充 `do_oss_ops_journal_mode()`。

## 编译和验证

编译目标：

```bash
cd /home/hust/icfs
ninja -C build idfs-oss
```

需要根据 ICFS 的实际部署方式确认二进制和插件安装路径。不要直接沿用 Ceph 的：

```bash
sudo install build/bin/ceph-osd /usr/bin/ceph-osd
```

ICFS 对应目标是 `idfs-oss`，服务名、二进制路径、admin socket 路径都需要以当前环境为准。

验证方向：

- `idfs-oss` 能正常启动。
- admin socket 能 dump perf schema。
- `object_hp_status` section 存在。
- 正式 vdbench 测试时 `hp_count` 增长。
- `hp_train_total`、`hp_actual_hot_percent`、`hp_hot_percent`、precision/recall 有变化。
- 如果 `hp_count` 不增长，优先检查当前池是否走 `IdfsMergePG`。

## 风险点

1. **只 hook PrimaryLogPG 可能漏 IO**

   ICFS 的 `IdfsMergePG` 重写了 `do_oss_ops()`。如果测试池是 MergePG/append-write 池，需要同步 hook。

2. **类型名不能照搬 Ceph**

   `CephContext`、`PerfCounters`、`OSDOp`、`ceph_osd_op` 在 ICFS 中都有改名。

3. **不要误 hook crimson 路径**

   当前传统路径是 `src/oss`。`src/crimson/oss` 是另一套实现，除非确认部署 crimson，否则不要先改。

4. **object hash 不能单独当唯一 key**

   `soid.get_hash()` 是分布 hash，可能冲突。应继续组合 object name hash 和 bucket。

5. **内部管理 op 不应混入第一阶段训练**

   recovery、scrub、tier/cache 管理 op 可能不代表用户真实冷热访问，第一阶段应跳过。

