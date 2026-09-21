# CODEX 文档索引

本目录维护 `merge` 分支的缓存与冷热识别实现、操作和迁移说明。
`AGENTS.md` 规定代理行为和检查级别；本文件只负责文档导航。

- [CODEX_CEPH.md](CODEX_CEPH.md)：当前 C4 object-layer Heat Predictor 的算法、线程和状态契约。
- [MGR_HP_OPERATIONS.md](MGR_HP_OPERATIONS.md)：冷热识别开关、reset、简要及详细状态输出。
- [ONODE_CACHE_OPERATIONS.md](ONODE_CACHE_OPERATIONS.md)：Onode LRU/S3FIFO 在线切换、状态和命中率口径。
- [ONODE_CACHE_MODULE.md](ONODE_CACHE_MODULE.md)：缓存模块边界、BlueStore 必要接入点及保持行为不变的模块化重构。
- [CEPH_OPERATIONS_MANUAL.md](CEPH_OPERATIONS_MANUAL.md)：单节点构建、部署和运维。
- [CACHE_C4_INTEGRATION.md](CACHE_C4_INTEGRATION.md)：缓存与 C4 的代码来源、整合范围及验证边界。
- [CACHE_HEAT_PREDICTOR_PORTING_GUIDE.md](CACHE_HEAT_PREDICTOR_PORTING_GUIDE.md)：将当前两个模块移植到其他 Ceph v17.2.7 修改版的接入清单。
- [EXISTING_CLUSTER_CACHE_TEST.md](EXISTING_CLUSTER_CACHE_TEST.md)：现有服务器的测试与控制端对接入口；按实际 OSD、客户端、缓存配置执行，不套用历史实验环境。
- [S3FIFO 实现修复与验证](../qa/experimental/onode-s3fifo-hit-accounting-20260919/README.md)：lookup 频次与 ghost 缩容修复、原服务器应用方式、三项成对实测及未达标边界。
- [Onode 有界预取与联合验证](../qa/experimental/onode-prefetch-20260920/README.md)：默认关闭的实验预取、控制端接入、五类固定负载结果及短窗口未达标边界；不是 128 节点验收结果。
- [Onode 预取压力恢复](../qa/experimental/onode-prefetch-pressure-20260921/README.md)：连续负载下的有界回收、候选重试、诊断字段和现有服务器复测步骤；正确性验证不代表命中率验收通过。
- [BRANCH_WORKFLOW.md](BRANCH_WORKFLOW.md)：`main`/`dev` 的长期职责和发布流程。

`merge` 保留两个生产模块及缓存回归测试；不包含 `dev` 的 HP Trace 采集、实验回放或候选算法。
默认关闭 HP；未覆盖启动配置时，Onode 使用原生 LRU、Buffer 使用2Q，S3FIFO 未启用。
