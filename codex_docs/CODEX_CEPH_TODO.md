# Heat Predictor TODO

本文件只记录尚未完成的工作。当前实现见 [实现说明](CODEX_CEPH.md)，Trace 约定见
[Trace 设计](TRACE_DATASET.md)，历史实验报告保留在
`/home/chris/ceph-test/new_workload/hp_runs/reports/`。

## 当前基线

生产语义已冻结：10秒标签窗口、object 总热度 Otsu、预测时保存热阈值、
固定 Otsu EMA `0.10`、固定预测阈值 `0.50`、冷热单位样本权重、25棵 ARF 和3个
feature。新增热度标签、动态预测阈值、动态 EMA、类别权重、快速树及其他未通过
验证的候选不再属于活动任务；对应的生产实现残留已删除。

当前基线没有待实施算法项。正确性缺陷应先在 `dev` 增加确定性复现，修复并完成
对应检查后再选择生产改动进入 `main`。

## V2：未来访问次数冷热定义

目标是预测同一 object 在未来10秒内的访问次数是否达到预测时保存的动态阈值 `K`。
每条 I/O 保留独立 EQ item，Otsu 对 object 等权；稀疏模式使用 `K=1`。完整语义与
实施步骤以以下文档为准：

- [V2 设计](../docs/superpowers/specs/2026-07-23-v2-future-access-hotness-design.md)
- [V2 实施计划](../docs/superpowers/plans/2026-07-23-v2-future-access-hotness.md)

执行顺序：

1. 固定 `dev` 基线并运行现有探针。
2. 实现 threshold 模块、EQ 标签和3维 feature。
3. 同步 OSD/MGR 遥测、生命周期和 `dev` Trace。
4. 完成算法、并发、构建和单节点正确性验证。
5. 正确性确认后运行五种正式 Vdbench 负载，首轮不设置准确率硬门槛。

V2 只在 `dev` 开发，生产路径不保留多版本宏分支；发布规则见
[分支流程](BRANCH_WORKFLOW.md)。
