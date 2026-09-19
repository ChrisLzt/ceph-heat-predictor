# Onode 查询命中率与在线缓存切换

本接口针对 BlueStore Onode 缓存，在原生 LRU 与 S3FIFO 之间在线切换。
考核展示名称为 **IO 请求命中率（Onode 查询口径）**，不是数据块命中率、
客户端请求完成率，也不与 NFS/MDS/数据缓存命中率相乘。

在同学现有服务器上组织测试与控制端接入，先阅读
[现有集群测试与对接手册](EXISTING_CLUSTER_CACHE_TEST.md)。本文接口不依赖特定
主机、三节点拓扑、容器名称、挂载方式或固定缓存容量；下面 `osd.0` 仅为语法示例，
执行时必须替换为已确认的实际 OSD 清单，不能据此假定只需控制一个 OSD。

## 命令

使用已配置 Ceph 访问凭据、具有对应 OSD 命令权限的管理客户端执行：

```bash
# 读取实际状态，不修改缓存
ceph tell osd.0 onode_cache status -f json

# 关闭智能缓存：回到原生 Onode LRU
ceph tell osd.0 onode_cache policy lru -f json

# 开启智能缓存：切换到 Onode S3FIFO
ceph tell osd.0 onode_cache policy s3fifo -f json

# 对目标集群所有 OSD 操作；正式实验必须逐个核对返回结果
ceph tell 'osd.*' onode_cache policy lru -f json
ceph tell 'osd.*' onode_cache policy s3fifo -f json
ceph tell 'osd.*' onode_cache status -f json
```

在有 admin socket 访问权限的 OSD 本机/容器内，也可使用：

```bash
ceph daemon osd.0 onode_cache status
ceph daemon osd.0 onode_cache policy s3fifo
ceph daemon osd.0 onode_cache policy lru
```

`policy` 仅接受 `lru`、`s3fifo`；不支持的存储后端返回 `EOPNOTSUPP`。
无效策略或不能用于 S3FIFO 的启动参数返回 `EINVAL`，不改变缓存。
重复设置当前策略为幂等操作，不清空 S3FIFO 频次或 ghost 历史。

## 生效与持久性

- 所有运行实验的 OSD 需要先升级到包含此接口的二进制。首次升级仍需重启；
  升级后的策略切换不需要重启 OSD、卸载 CephFS 或重新挂载客户端。
- 同一缓存分片地址保持不变。队列迁移在分片锁内完成，保留缓存对象、引用、
  pinned 状态、容量、年龄统计和累计命中/未命中计数。
- LRU 转 S3FIFO 时，已有 LRU 条目按原有顺序进入 small 队列，频次归零。
  S3FIFO 转 LRU 时，main 队列在前、small 队列在后，各自队列内部顺序保留。
  两个方向都清空 ghost 历史。不会伪造跨算法访问历史。
- 切换成本与已链接 Onode 数量成正比，会短暂占用分片锁；不能保证零延迟扰动。
  同一 OSD 逐分片切换，命令成功返回时全部分片已生效；不是跨 OSD 原子操作。
- BlueStore buffer cache、RocksDB、CephFS 客户端缓存保持不变。
  “关闭智能缓存”意味着使用 LRU 基线，不代表关闭所有缓存。
- 此接口仅改变运行时策略，不写入持久配置。OSD 重启后按已有
  `bluestore_cache_type` 选择启动策略：`2q/lru` 对应 Onode LRU，
  `s3fifo` 对应 Onode S3FIFO。不要用 `config set bluestore_cache_type`
  替代本接口实施在线切换。
- S3FIFO 三个调参项仍在分片创建时读取，本接口不负责在线调整这些参数。

## 状态契约

成功返回的单 OSD JSON 根对象包含下列字段（不是 `.onode_cache` 子对象）：

| 字段 | 含义 |
|---|---|
| `effective_policy` | 当前所有分片实际策略；意外不一致时为 `mixed` |
| `cache_instance` | 本次 BlueStore 实例 UUID，重启后变化，用于隔离计数周期 |
| `policy_generation` | 本实例成功改变策略的次数，幂等请求不增加 |
| `runtime_only` | 固定为 `true`，不持久化到配置数据库 |
| `buffer_cache_policy` | 启动时选择的实际 buffer 策略，切换 Onode 不改变它 |
| `last_switch_started_ns/completed_ns` | 最近一次有效切换的服务器 Unix 纳秒时间；未切换时为 0 |
| `sample_time_ns` | 此次计数读取附近的服务器 Unix 纳秒时间 |
| `onode_hits/onode_misses` | 现有 BlueStore Onode 查询累计计数，不是百分比 |
| `shards` | 各分片策略、resident/unlinked 条目及 LRU/small/main/ghost 队列长度 |

`unlinked_onodes` 是不在淘汰队列中的驻留条目数，不是所有在途引用的精确计数。
并发 I/O 仍可增加命中/未命中计数；状态是观测快照，不冻结业务请求。

## 控制端对接与 10 分钟流程

1. 预先生成并持久化实验数据；升级 OSD 并验证每个目标 OSD 支持新命令。
2. 记录固定的实验 OSD 清单、工作负载、缓存容量和版本。对每个目标 OSD 设置
   `lru`，读取状态确认所有分片生效，再开始计时和正式负载。
   控制端使用单调时钟计时，各节点同步墙钟以对齐服务器时间戳。
3. 0 至 180 秒维持 LRU；负载预测和冷热识别按各自接口关闭。
   每隔 1 至 5 秒记录实际读写 IOPS、热点变化和 Onode 原始计数。
4. 到 180 秒，对同一 OSD 清单下发 `s3fifo`，同时通过各自接口启用另两项功能。
   保存每个 OSD 的响应、切换起止时间和后续状态，确认全部成功后才显示全部已开启。
5. 继续同一负载到 600 秒，不重启、不清缓存、不重新生成数据。
   保留请求切换时间与实际完成时间之间的过渡区间，不能把迟到或失败的 OSD 显示为已开启。
6. 分开输出 LRU 阶段、切换区间、S3FIFO 阶段的结果及原始时间序列。
   若 OSD 集合变化、实例重启、命令失败或状态缺失，明确标记异常，不能按完整实验通过。

本接口不启停负载预测、冷热识别，也不将其输出接入 S3FIFO 淘汰逻辑。
三项功能的定时协调、曲线绘制及准确率计算仍由控制端负责。

## 命中率计算

同一 OSD、同一 `cache_instance` 的相邻采样点相减：

```text
delta_hits_i   = hits_i(t1)   - hits_i(t0)
delta_misses_i = misses_i(t1) - misses_i(t0)

cluster_onode_hit_rate =
  sum(delta_hits_i) / sum(delta_hits_i + delta_misses_i) * 100%
```

按查询次数加权汇总，不平均各 OSD 百分比。分母为 0 时显示“无样本”，
不能填 100%。实例变化、计数回退或目标 OSD 缺失时，这个区间标记无效。
跨越 `policy_generation` 变化的窗口标记为切换窗口，不混入纯 LRU/S3FIFO 阶段。

阶段汇总同样对该阶段有效区间的增量求和。指标要求是 **大于 95%**，
按未四舍五入的实测数值判断；代码切换成功不等于指标达标。
每次实验必须保存样本量、采样覆盖范围、缓存容量、工作集规模与异常记录。
这些计数覆盖 OSD 的全部 Onode 查询，不能按客户端或 pool 自动过滤。
验收应隔离无关负载，并记录恢复、回填、scrub 等后台活动，避免把它们归因于实验。

## 验证

```bash
cmake --build build --target unittest_bluestore_onode_cache -j2
build/bin/unittest_bluestore_onode_cache
build/bin/unittest_bluestore_onode_cache --gtest_repeat=50 --gtest_break_on_failure
```

测试直接使用 BlueStore 缓存实现，覆盖暖缓存双向迁移、pinned/不存在对象、
S3FIFO main/ghost 队列、重复设置、跨分片 unpin、并发查询、状态与新增分片继承。
单元测试不能替代实际 SSD 集群上的完整负载与 10 分钟可视化验收。
