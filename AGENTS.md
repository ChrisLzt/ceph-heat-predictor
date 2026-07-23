# 代理说明

本仓库 `main` 分支包含 Ceph OSD object 级热度预测器的确定实现。完整测试、消融
方案和 Trace 功能保留在 `dev` 分支。

修改前先阅读 `codex_docs/README.md`，然后只阅读其中标明且与当前任务相关的有效文档。
分支职责和发布流程见 `codex_docs/BRANCH_WORKFLOW.md`。

不要在 `main` 中加入算法测试源、Trace、实验 profile、编译期方案开关或已淘汰实现。
新方案应先在 `dev` 中开发和验证，确认后只把生产代码发布到 `main`，不得整体合并
`dev`。

日常开发默认在 `dev` 进行。生产改动和 dev-only 改动必须拆成可独立审查的提交；
发布时只选择生产提交进入 `main`，再以新的 `main` 为基线重新对齐 `dev` 的
Trace、测试和开发文档层。发布标签只能指向 `main` 的生产提交。

除非用户明确要求检查历史代码，否则不要沿用旧 KernelDevice/BlockDevice 热度预测器的
假设。

当前实现仅位于 object 层。Ceph 特有的 object op 适配应保留在
`src/osd/ObjectHeatPredictor.*`；可复用算法应保留在 `src/heatpredictor/`。

`main` 的改动至少执行 `git diff --check` 和受影响 Ceph 目标编译。完整算法、并发和
负载验证在 `dev` 完成；发布到 `main` 后只验证生产构建及必要的集成状态。
