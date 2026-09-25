# 历史merge生产模块移植参考

> 本文保留不含Trace的历史merge生产版本移植清单，来源截至 `140169f115c`。下文“当前merge”“此分支”“本轮验证”均指该历史语境，不描述当前dev。
> 自2026-09-25起开发测试统一在dev，且保留Trace和离线工具；本次整合及验证入口见[CACHE_DEV_INTEGRATION.md](CACHE_DEV_INTEGRATION.md)。移植当前dev时须额外核对Trace依赖和OSD初始化签名，不能直接照抄本文的无Trace文件清单。

本文面向将 `merge` 的当前模块移植到另一套 Ceph v17.2.7 修改版的开发者。
缓存包含 `5df98634d6f` 的预取压力恢复及 `9970f4d4598` 的模块化重构；
HP 算法来源为 `afd18c8e01`，
并已加入 2026-09-21 的实例生命周期与浅耦合重构；历史来源和本轮验证见
[CACHE_C4_INTEGRATION.md](CACHE_C4_INTEGRATION.md)。

移植应复制独立模块、按目标控制流合并 Ceph 接入点，不能整体覆盖目标的核心文件。
算法详细定义以 [CODEX_CEPH.md](CODEX_CEPH.md) 为准，缓存命令与统计以
[ONODE_CACHE_OPERATIONS.md](ONODE_CACHE_OPERATIONS.md) 为准。

## 1. 范围和默认行为

| 模块 | 当前实现 | 未覆盖配置时的默认状态 |
|---|---|---|
| Onode 缓存 | 原生 LRU 与 S3FIFO 在线切换 | LRU，S3FIFO 未开启 |
| Buffer 缓存 | 保持其启动策略，Onode 切换不影响它 | 2Q |
| HP | object 粒度的七维 C4 在线学习 | disabled |

`bluestore_cache_type` 默认 `2q`；`2q/lru` 对应 Onode LRU，`s3fifo` 对应 Onode
S3FIFO，并将 Buffer 启动策略映射为2Q。已有配置可以覆盖默认值。
“关闭智能缓存”表示切回 LRU，不是清空或禁用基础缓存。

两个模块独立控制。HP 输出预测和统计，不自动切换 Onode 策略，也不执行迁移或分层。
此分支不包含 dev-only HP Trace、离线回放和候选算法；不要从 dev 整目录覆盖过来。

## 2. 文件接入清单

### 可复制的独立模块

在目标没有同名自定义实现时，复制以下当前 merge 文件：

```text
src/heatpredictor/
src/osd/ObjectHeatPredictor.cc
src/osd/ObjectHeatPredictor.h
src/mgr/ObjectHeatPredictorCommands.cc
src/mgr/ObjectHeatPredictorCommands.h
src/mgr/ObjectHeatPredictorStatus.cc
src/mgr/ObjectHeatPredictorStatus.h
src/mgr/ObjectHeatPredictorStatusFormatter.cc
src/mgr/ObjectHeatPredictorStatusFormatter.h
src/os/bluestore/OnodeCache.h
src/os/bluestore/OnodeCache.cc
src/os/bluestore/OnodeCacheShard.cc
src/os/bluestore/OnodePrefetch.cc
src/os/bluestore/OnodeCacheBudget.h
```

`src/heatpredictor/` 是 header-only 算法模块，必须连同 `include/` 整体迁移。
EQ、特征、动态阈值、统计和模型都由其中头文件实现。

缓存实现已在压力恢复版本基础上拆为独立编译单元。它仍需适配 BlueStore 的原生类型、
锁和引用管理，不是与 Ceph 无关的通用插件；详见
[ONODE_CACHE_MODULE.md](ONODE_CACHE_MODULE.md)。

### 必须按语义合并的 Ceph 文件

| 文件 | 需要接入的内容 |
|---|---|
| `src/common/options/global.yaml.in` | S3FIFO 配置项及枚举 |
| `src/os/ObjectStore.h` | 缓存策略查询/切换接口；HP 数据观察回调的注册、注销及保留对象过滤 |
| `src/os/ObjectStoreAccess.h` | 独立观察桥，注销等待在途通知；依赖 hp_access_type.h 的 Read/Write 类型 |
| `src/os/bluestore/BlueStore.h/.cc` | 原生 Onode/OnodeSpace hook、控制器持有、生命周期、查询触发、预算与磁盘格式适配；队列、策略状态和 worker 在独立模块；BlueStore.cc 另有 read/readv/_write 的 HP 通知 |
| `src/os/CMakeLists.txt` | 注册 `OnodeCache.cc`、`OnodeCacheShard.cc`、`OnodePrefetch.cc` |
| `src/osd/CMakeLists.txt` | `ObjectHeatPredictor.cc` |
| `src/osd/OSD.h/.cc` | OSDService 持有 HP 实例；初始化后绑定 ObjectStore 回调；关闭先注销并排空回调再销毁 HP；保留命令路由及 Onode 命令 |
| `src/osd/PrimaryLogPG.cc` | 删除旧四处 HP 通知和 HP include，避免与存储层重复计数 |
| `src/mgr/CMakeLists.txt` | HP Commands、Status 和 StatusFormatter 三个实现文件 |
| `src/mgr/DaemonServer.cc` | 转交 HP 专用命令模块，提供连接检查及 Objecter 访问 |
| `src/mgr/MgrCommands.h` | HP 命令及 `status --detail` |
| `src/mgr/PyModuleRegistry.h` | 提供 Objecter 访问路径 |
| `src/test/objectstore/CMakeLists.txt` | 缓存测试及 unittest_storage_object_access 存储观察测试 |

缓存测试文件为 `src/test/objectstore/test_bluestore_onode_cache.cc`。
不要按旧文档行号贴代码；先确认目标版本的类型、锁、生命周期和构建方式。

## 3. 缓存迁移要点

### 状态与队列

`SwitchableOnodeCacheShard` 继承 `LruOnodeCacheShard`，保存当前 `Policy` 和 S3FIFO
的 small/main/ghost 队列。Onode 的 `s3fifo_freq` 饱和到3，`s3fifo_queue` 标记队列归属。
LRU、small、main 复用 intrusive hook，同一 Onode 不可同时链接到多个队列。

保留 `_add`、`_rm`、`_touch`、`_trim_to`、pin/unpin、跨 shard 迁移和 age bin 的
调用约定。持锁操作必须作用于当前 shard；目标改变这些约定时，应逐项适配，而非只替换类名。
Ghost 保存对象标识，当前查找/删除为线性扫描；移植不顺带更换数据结构。

### 在线切换与启动配置

```bash
ceph daemon osd.0 onode_cache status
ceph daemon osd.0 onode_cache policy s3fifo
ceph daemon osd.0 onode_cache policy lru
```

首次部署新接口需要加载新 OSD 二进制；之后切换不需要重启或重新挂载。
`ObjectStore` 默认接口返回 `EOPNOTSUPP`；BlueStore 对无效策略返回 `EINVAL`，
分片尚未建立时返回 `EAGAIN`。目标后端没有该实现时不能伪造成功。

切换保持 shard 地址、驻留对象、引用、容量、年龄统计和累计命中计数：

- LRU → S3FIFO：原 LRU 条目按原顺序进入 small，频次归零。
- S3FIFO → LRU：main 在前、small 在后，保留各队列内部顺序。
- 两个方向均清空 ghost；重复设置同一策略不清空历史，也不增加策略代次。
- 每个 OSD 逐 shard 加锁切换，成功返回时全部生效；不是跨 OSD 原子操作。

在线命令只改变运行时状态，不写配置数据库；重启后仍由 `bluestore_cache_type` 决定。
以下调参项在 shard 创建时读取，在线切换命令不负责修改它们：

| 配置 | 默认值 |
|---|---:|
| `bluestore_cache_s3fifo_small_ratio` | 0.10 |
| `bluestore_cache_s3fifo_ghost_ratio` | 0.90 |
| `bluestore_cache_s3fifo_promotion_threshold` | 2 |

### 统计

保留 `effective_policy`、`cache_instance`、`policy_generation`、切换及采样时间、
`buffer_cache_policy`、逐 shard 队列长度和 `onode_hits/onode_misses`。
命中率使用同一实例内的计数增量计算；Onode 查询命中率不是数据块命中率或端到端 I/O 成功率。
保留 `buffer_bytes`、`buffer_hit_bytes`、`buffer_miss_bytes` 的 `PRIO_USEFUL` 优先级。

## 4. HP 迁移要点

### Hook 与标签

BlueStore 在 read/readv 数据读取前及 _write 入口（普通/journal分支之前）通知实际存储对象。
OSDService 持有 HP，OSD 注册观察回调，关闭时先注销并等待在途通知再销毁 HP。
旧 PG hook 必须删除；readv 多区间一次通知，数据缓存命中也通知，内部重试不重复。
当前不区分来源，只覆盖上述入口；详细过滤条件、未覆盖路径和兼容字段见 CODEX_CEPH.md。
不要在完成回调中再次通知。不要把 dev 的 Trace 或候选算法随 hook 一起移入 merge。

粒度为 RADOS object。每次 I/O 预测同一 object 在未来 `(t,t+10s)` 的访问数是否达到
到期阈值 `K_window`。当前访问与恰在 deadline 的访问不计入未来窗口。
`K_context` 来自预测时历史，只用于特征及预热规则，不替代真实标签阈值。

Otsu 对活跃 object 的过去10秒计数等权投票，使用固定2000-bin直方图。
到期按1ms微批处理；严格窗口、阈值和标签生成必须一起迁移，不能用简化累计计数替代。

### C4 特征与学习策略

七维特征为原有过去10秒计数裕量、访问间隔、旧热度、2秒投影裕量、2秒次数，
加上 τ=30秒、60秒的两个慢历史特征：

```text
A_tau = 当前访问之前的访问按 exp(-访问年龄/tau) 加权求和
T = 从当前保留状态首次观测访问起的时间（秒）
exposure = T > 0 ? 1 - exp(-T/tau) : 1
corrected_count = A_tau * (10/tau) / max(exposure, 1/4)
slow_feature = log2(1 + corrected_count) - log2(1 + K_context)
```

当前访问在生成慢历史特征后才计入；同时间戳下已经记账的较早访问会计入。
保存预测时的特征输入用于延迟训练，不能在标签成熟时重新读取对象状态。
旧热度每次访问增加100，10秒无访问后降为10%，继续作为第三维特征。
慢历史状态随原有 LRU 淘汰/reset 清空；EQ容量100万，闲置对象LRU容量100万。

模型为 `StandardScaler<7> + 25-tree ARF`，每次分裂考虑全部7维，seed=591422，
Poisson λ=4，冷热样本权重均为1，预测阈值0.50。叶固定输出多数类统计概率，ADWIN关闭。
保留当前分裂候选过滤和显式预剪枝语义。

预热保护看的是已发布快照的成熟训练计数：少于3000时输出
`past_10s_count >= K_context`，训练仍正常进行。模型和该计数作为一个整体原子发布。
每2000个训练样本，或有新训练且距上次发布达到2秒时发布新快照。

### 生命周期、控制和统计

HP 默认 disabled；enable/disable 均完整 reset，reset 保持启用状态。
每个 OSDService 拥有独立适配实例，算法实现通过 PIMPL 隐藏；不要恢复为进程全局变量。
默认关闭时不创建森林或快照，enable 创建模型，首次预测启动工作线程。disable 释放
模型，已启动线程等待；重启 OSD 不恢复模型或启用状态。
正常退出先注销命令、停止请求与服务后台调用，再执行模块 shutdown、等待回调/线程
退出并注销 perf logger；不能持有预测器控制锁执行 join。fast-shutdown 的 `_exit()`
仍直接结束进程。核心使用独立且 Release 生效的 `hp_assert.h`，无需 Ceph 运行库。
保留 EQ 到期线程、后台训练、只读模型快照及原锁顺序。异常不能传播为 Ceph I/O 失败；
后台异常会禁用模块并刷新状态。

```bash
ceph daemon osd.0 object_hp status
ceph daemon osd.0 object_hp enable
ceph daemon osd.0 object_hp disable
ceph daemon osd.0 object_hp reset
ceph osd hp status
ceph osd hp status --detail -f json
```

OSD/MGR 共用 `hp_telemetry.h`；MGR 仅接受首尾发布代次一致的报告。
迁移协议时保证双方定义兼容，并逐个确认 reporting 状态，不把“命令已下发”当作操作完成。
默认状态是五行简要摘要，脚本需要完整统计时必须显式使用 `--detail`。
同时迁移新 formatter 和 CMake 引用，不能只更新 `DaemonServer.cc`。

## 5. 接入和验证顺序

1. 固定目标版本、保存已有改动，先确认目标原本能构建。
2. 接入缓存配置、ObjectStore/BlueStore 和 OSD 命令，编译并运行缓存测试。
3. 整体移植当前 merge 的 HP 算法，接入 OSD hook、控制和 PerfCounters。
4. 接入 MGR 聚合、formatter、命令和 CMake，构建 OSD/MGR。
5. 在获准的测试环境加载新二进制，分别核对默认行为、缓存切换和 HP enable/reset。
6. 再进行联合负载和性能评估，单元测试不替代线上验收。

本仓库回归入口：

```bash
git diff --check
ninja -C build ceph-osd ceph-mgr \
  unittest_bluestore_onode_cache unittest_bluestore_types
build/bin/unittest_bluestore_onode_cache --gtest_repeat=10
build/bin/unittest_bluestore_types \
  --gtest_filter=-bluestore_blob_t.csum_bench:sb_info_space_efficient_map_t.size
```

目标系统还需验证：

- 切换不丢驻留对象、引用、计数；Onode 策略变化不改变 Buffer 策略。
- HP disabled 时不累计 I/O，enable/reset 后旧窗口、模型和计数全部清空。
- 等待标签及训练排空后，错误/丢弃计数和 OSD/MGR 汇总满足预期。
- 单 OSD 始终满足 `io = labeled + pending + awaiting + eval_drop`。
- 新接口无命令冲突，目标系统原有测试仍通过。

当前整合回归结果见整合说明。其他 Ceph 修改版的可编译性、运行正确性、命中率和
Accuracy 收益均须在目标环境确认；本机历史结果不能直接作为目标环境验收结论。

对象身份适配传递pool、placement hash、完整name、namespace、snapshot、locator key；
完整相等比较位于HP身份表，不使用hash值作为唯一身份。身份ID在进程内不复用，随热状态回收映射。
