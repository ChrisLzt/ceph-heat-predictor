# Onode 缓存模块边界

本次重构以压力恢复实现 `5df98634d6fa7dff7bdfcd6f806d11427f15dead` 为基线，
目标是降低后续合并和维护对 `BlueStore.cc/.h` 的侵入，不改变缓存算法、负载或考核口径。
这是 BlueStore 专用的内部模块，不是可以脱离 Ceph 使用的通用缓存插件。

## 文件职责

| 文件 | 职责 |
|---|---|
| `src/os/bluestore/OnodeCache.h/.cc` | 单实例控制器：策略锁、在线切换、状态输出、分片创建与配额；封装预取生命周期 |
| `src/os/bluestore/OnodeCacheShard.cc` | LRU/S3FIFO 分片实现、队列迁移、淘汰和 pin/unpin；LRU 代码为原实现搬移 |
| `src/os/bluestore/OnodePrefetch.cc` | 后台扫描、解码、预算准入、有界回收、重试与诊断计数 |
| `src/os/bluestore/OnodeCacheBudget.h` | 不依赖 BlueStore 的分片配额计算；本次不改算法 |
| `src/os/bluestore/BlueStore.cc/.h` | 原生对象和分片生命周期、引用管理、存储格式，以及下表中的接入点 |

`BlueStore.h` 对控制器仅保留前向声明和一个 `unique_ptr`，不再包含预取工作队列、
线程实现、策略状态或其锁的定义。控制器每个 BlueStore 实例分配一次；没有每次 I/O 的
额外动态分配，没有新引入通用虚接口，也没有把私有字段改为 public。

## 原系统接入点

| 接入点 | 保留原因与约束 |
|---|---|
| 构造、`_mount`、`_umount` | 创建控制器；仍在原有位置启动、停止并 join 预取线程，先停止 worker 再关闭 DB |
| `Collection::get_onode` | 持 collection 锁时通知扫描；不把后台预取计入 demand hit/miss |
| `MempoolThread::_resize_shards` | 发布原生计算出的 metadata 预算，调用配额分配；不改变自动调节器或 Buffer 预算 |
| `set_cache_shards` | 委托创建分片；新增 Onode 分片继承当前在线策略，Buffer 保持启动策略 |
| `set/get_onode_cache_policy` | 保持 ObjectStore/OSD 的现有接口，委托模块完成切换和输出 |
| `OnodeCache::read_onode_record` | 一个原生格式适配函数，复用 `get_object_key` 与 `PREFIX_OBJ`；不复制磁盘键编码 |
| Onode、OnodeSpace、Collection 的少量字段与 hook | 保留 intrusive 队列状态、pin/unpin、查询 touch、预取首次使用/移除计数和扫描代次 |

最后一项不能简单删除：缓存淘汰依赖真实引用和分片锁，统计依赖真实查询与对象移除。
把这些操作放在外围代理里，会失去原生生命周期保证。本次没有再修改这些 hook 的行为。

## 保持不变的契约

- 策略仍为 LRU/S3FIFO，默认配置、S3FIFO 参数、自动调节及 OSD 内存目标均不改。
- 在线切换仍先验证全部分片；切回 LRU 先取消预取，切到 S3FIFO 在切换完成后激活。
- 锁顺序保持策略锁、分片锁的原有关系；worker 不带着队列锁执行 DB/collection 操作。
- `onode_cache status` 的字段名、代次、实例标识、时间戳和预取诊断字段保持兼容。
- `onode_hits/onode_misses` 仍只记真实 demand 查询；预取读、加载、首次使用分开计数。
- 预算保护、128 步有界回收、回退等待及原候选重试保持上一版行为。
- Heat Predictor、客户端、工作负载、数据、配置项定义和磁盘格式均不改。
- Buffer 的实现仍在原文件；模块仅保留原有的创建和启动策略映射，不改数据缓存算法。

## 移植方式

复制上述四个新增模块文件和原有 `OnodeCacheBudget.h`，在 `src/os/CMakeLists.txt`
中注册三个 `.cc`。按语义合并原生类型与接入点，不能用本仓库的整个 `BlueStore.cc/.h`
覆盖同学服务器的修改版。

本轮 diff 中大段删除是迁移到独立编译单元，不是删除 LRU 或预取能力。
后续修改回收、限速、状态字段时，主要修改模块内部；只有 Ceph 原生锁、引用或存储
格式契约发生变化时，才需要适配接入点。

部署新二进制仍需要滚动重启 OSD；运行时的 LRU/S3FIFO 切换仍不需要重启。
本轮编译使用独立 CloudLab builder，不替换运行中的 OSD，也不改变现有集群配置。

## 验证

现有缓存、配额、并发切换、真实读写、删除/重挂载、压力恢复测试全部保留。
新增测试覆盖控制器的启动策略映射、实例隔离和未启动 worker 的诊断字段。
CloudLab 独立 builder 中，`ceph-osd`、`unittest_bluestore_onode_cache`、
`unittest_bluestore_types` 编译通过；真实 CTest 入口、快速测试 50 轮、压力测试 5 轮
和类型测试通过。类型测试沿用上轮排除两个规模/性能用例的设置，不是全量 Ceph 测试。
详见 [模块化验证记录](../qa/experimental/onode-cache-module-20260921/README.md)。
这次结构重构不代表五种负载或 128 节点命中率达标。
