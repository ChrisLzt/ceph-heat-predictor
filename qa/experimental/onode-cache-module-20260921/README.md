# Onode 缓存模块化验证

## 范围

- 基线：压力恢复提交 `5df98634d6fa7dff7bdfcd6f806d11427f15dead`。
- 分支：`refactor/onode-cache-module-20260921`。
- 本次只调整内部模块边界，不更改淘汰、预取、冷热识别或统计规则。
- [模块与移植说明](../../../codex_docs/ONODE_CACHE_MODULE.md)。

`BlueStore.cc` 净减少 1,010 行、`BlueStore.h` 净减少 48 行。删除部分主要迁移到
`OnodeCache.h/.cc`、`OnodeCacheShard.cc` 和 `OnodePrefetch.cc`，不是删除功能。
OnodeSpace 的原生生命周期和计数函数保持一致；LRU/S3FIFO 主体仅搬移并清理行尾空格。
worker 仅调整所属类型、策略锁访问路径和原生 DB 读取适配，不改控制流程。

## 远程正确性验证

日期：2026-09-21。环境为 CloudLab Linux 独立 builder，镜像
`ceph-onode-builder:20260918-v2`；不是同学服务器或 128 节点的验收结果。
没有替换运行中的 OSD，也没有重启集群或更改其参数。没有在 Mac 上启动 Docker。

| 检查 | 结果 |
|---|---|
| `ceph-osd`、`unittest_bluestore_onode_cache`、`unittest_bluestore_types` | 构建成功 |
| 实际 CTest 入口 `unittest_bluestore_onode_cache` | 通过，测试进程 14.51 秒 |
| 缓存、预算、策略测试重复 50 次 | 每轮 28 项，共 1,400 项通过 |
| 真实存储压力恢复测试重复 5 次 | 每轮 3 项，共 15 项通过 |
| BlueStore 类型测试 | 27 项通过 |
| 本轮 8 个代码/构建文件 SHA-256 | 与远程构建输入一致 |
| 27 个固定 Heat Predictor 文件 SHA-256 | 保持不变 |
| `git diff --check` | 通过 |

保留全部旧测试；新增启动配置映射、实例状态隔离和未启动 worker 的诊断接口测试。
真实存储测试包含读写、属性、删除、关闭重开、预算压力恢复和候选重试。
类型测试沿用基线排除 `sb_info_space_efficient_map_t.size` 与
`bluestore_blob_t.csum_bench`；没有运行完整 Ceph 测试集或 sanitizer。
编译仍有基线已存在的 btree/原子操作编译警告，不能称为零警告构建。

## 证据与边界

- [构建输入及二进制身份](evidence/verification.json)。
- [CTest 原始输出](evidence/ctest.txt)。
- [日志摘要及 SHA-256](evidence/test-summary.json)。
- [搬移与未改行为的源码核对](evidence/structure-verification.json)。

构建复用了已有的远程增量 checkout，并不是整个仓库的全新干净构建。
身份校验覆盖此次 8 个修改文件及 27 个固定 HP 文件，不扩大为全树同一 SHA 的声明。
Git 中保留紧凑证据，完整原始日志保存在本次工作区
`outputs/onode-cache-module-20260921/` 及远程相同任务名的构建目录。

这证明拆分后已构建并通过上述回归门禁，不证明性能完全无变化。
没有重新执行五种完整负载、连续 case 联合测试或 128 节点验收；不能据此宣称所有窗口
命中率达到 96%。部署新二进制仍需滚动重启；后续在线切换接口保持原样。
