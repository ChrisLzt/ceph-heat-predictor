# CODEX 文档索引

本目录维护dev中缓存、冷热识别、Trace及离线工具的有效说明。
自2026-09-25起，日常开发和测试统一使用dev；merge仅作为历史分支保留。`AGENTS.md` 只规定代理行为和
检查级别；本文件负责文档导航，两者不重复维护内容清单。

- [DEV_ALGORITHM_STATUS.md](DEV_ALGORITHM_STATUS.md)：当前dev算法、已采用变更及未采用实验候选总览。
- [RIVER_SETTINGS.md](RIVER_SETTINGS.md)：dev采用的八项River参数与行为及验证范围。
- [CODEX_CEPH.md](CODEX_CEPH.md)：Ceph object-layer Heat Predictor 的稳定实现说明。
- [CEPH_OPERATIONS_MANUAL.md](CEPH_OPERATIONS_MANUAL.md)：单节点构建、部署和运维命令。
- [MGR_HP_OPERATIONS.md](MGR_HP_OPERATIONS.md)：集群级状态、开关和 reset 操作。
- [BRANCH_WORKFLOW.md](BRANCH_WORKFLOW.md)：`dev`统一开发测试、`merge`历史保留及`main`发布职责。

- [ONODE_CACHE_OPERATIONS.md](ONODE_CACHE_OPERATIONS.md)：LRU/S3FIFO在线切换、预取控制和命中率口径。
- [ONODE_CACHE_MODULE.md](ONODE_CACHE_MODULE.md)：缓存模块与BlueStore接入边界。
- [EXISTING_CLUSTER_CACHE_TEST.md](EXISTING_CLUSTER_CACHE_TEST.md)：现有服务器缓存测试与控制端接口。
- [CACHE_HEAT_PREDICTOR_PORTING_GUIDE.md](CACHE_HEAT_PREDICTOR_PORTING_GUIDE.md)：历史merge生产模块移植参考，不是当前dev含Trace的完整文件清单。
- [CACHE_DEV_INTEGRATION.md](CACHE_DEV_INTEGRATION.md)：2026-09-25缓存迁入dev的范围、保留项和验证入口。
- [CACHE_C4_INTEGRATION.md](CACHE_C4_INTEGRATION.md)：此前merge整合历史，不能当作本轮验证结果。

以下活动文档只在 `dev` 分支维护，不随生产代码发布到 `main`：

- [CODEX_CEPH_TODO.md](CODEX_CEPH_TODO.md)：尚未实施的 V2 工作。
- [TRACE_DATASET.md](TRACE_DATASET.md)：Trace schema、采集和离线回放约定。

历史设计与实施计划位于 `docs/superpowers/`，不作为当前 TODO，也不随活动文档路径
迁移而重写。
