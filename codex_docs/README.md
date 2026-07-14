# CODEX 文档索引

本目录维护当前有效的代理上下文、后续计划和 Ceph 操作手册。`AGENTS.md` 只规定
代理行为和检查级别；本文件负责文档导航，两者不重复维护内容清单。

- [CODEX_CEPH.md](CODEX_CEPH.md)：Ceph object-layer Heat Predictor 的稳定实现说明。
- [CODEX_ICFS.md](CODEX_ICFS.md)：向 ICFS/IDFS 迁移的接口和风险说明。
- [CODEX_AUTO.md](CODEX_AUTO.md)：自动修改和负载测试约束。
- [CEPH_OPERATIONS_MANUAL.md](CEPH_OPERATIONS_MANUAL.md)：单节点构建、部署和运维命令。
- [CODEX_CEPH_TODO.md](CODEX_CEPH_TODO.md)：Ceph 后续工作的总索引、全局约束和执行依赖。
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)：实验工具、数据一致性和性能采集
  规范，不作为功能 TODO。
- [PREDICTION_THRESHOLD.md](todo/PREDICTION_THRESHOLD.md)：固定/动态预测阈值。
- [OTSU_THRESHOLD.md](todo/OTSU_THRESHOLD.md)：固定/动态热阈值 EMA。
- [FINAL_VALIDATION.md](todo/FINAL_VALIDATION.md)：两个阈值维度的 2×2 测试矩阵。

历史设计与实施计划位于 `docs/superpowers/`，不作为当前 TODO，也不随活动文档路径
迁移而重写。
