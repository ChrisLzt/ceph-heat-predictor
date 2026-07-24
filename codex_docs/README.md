# CODEX 文档索引

本目录维护当前有效的实现说明和 Ceph 操作手册。`AGENTS.md` 只规定代理行为和
检查级别；本文件负责文档导航，两者不重复维护内容清单。

- [CODEX_CEPH.md](CODEX_CEPH.md)：Ceph object-layer Heat Predictor 的稳定实现说明。
- [CEPH_OPERATIONS_MANUAL.md](CEPH_OPERATIONS_MANUAL.md)：单节点构建、部署和运维命令。
- [MGR_HP_OPERATIONS.md](MGR_HP_OPERATIONS.md)：集群级状态、开关和 reset 操作。
- [BRANCH_WORKFLOW.md](BRANCH_WORKFLOW.md)：`main`/`dev` 职责、开发、发布和重新对齐流程。

以下活动文档只在 `dev` 分支维护，不随生产代码发布到 `main`：

- [CODEX_CEPH_TODO.md](CODEX_CEPH_TODO.md)：尚未实施的 V2 工作。
- [TRACE_DATASET.md](TRACE_DATASET.md)：Trace schema、采集和离线回放约定。

历史设计与实施计划位于 `docs/superpowers/`，不作为当前 TODO，也不随活动文档路径
迁移而重写。
