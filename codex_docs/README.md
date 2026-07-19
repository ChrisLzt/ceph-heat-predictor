# CODEX 文档索引

本目录维护 `main` 当前有效的实现说明和 Ceph 操作手册。`AGENTS.md` 只规定
代理行为和检查级别；本文件负责文档导航，两者不重复维护内容清单。

- [CODEX_CEPH.md](CODEX_CEPH.md)：Ceph object-layer Heat Predictor 的稳定实现说明。
- [CODEX_ICFS.md](CODEX_ICFS.md)：向 ICFS/IDFS 迁移的接口和风险说明。
- [CEPH_OPERATIONS_MANUAL.md](CEPH_OPERATIONS_MANUAL.md)：单节点构建、部署和运维命令。

实验计划、测试规范和 Trace 设计只在 `dev` 分支维护。

历史设计与实施计划位于 `docs/superpowers/`，不作为当前 TODO，也不随活动文档路径
迁移而重写。
