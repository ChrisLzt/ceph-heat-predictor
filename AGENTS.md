# 代理说明

修改前先阅读 `codex_docs/README.md`，然后只阅读其中标明且与当前任务相关的有效文档。
缓存与冷热识别的日常开发、调试和测试统一在 `dev`（`/home/chris/ceph-heat-predictor`）进行；
`merge` 仅保留历史整合记录，`main` 只保留确定的生产实现。分支职责和发布流程统一见
`codex_docs/BRANCH_WORKFLOW.md`，不要在本文件重复维护。

除非用户明确要求检查历史代码，否则不要沿用旧 KernelDevice/BlockDevice 热度预测器的
假设。

当前实现仅位于 object 层。HP生命周期及控制适配位于 `src/osd/ObjectHeatPredictor.*`，
真实存储对象读写观测接入位于 BlueStore；可复用算法保留在 `src/heatpredictor/`。
Onode缓存策略及预取保留在 `src/os/bluestore/OnodeCache*` 和 `OnodePrefetch.cc`。

`main` 的改动至少执行 `git diff --check` 和受影响 Ceph 目标编译。完整算法、并发和
负载验证在 `dev` 完成；发布到 `main` 后只验证生产构建及必要的集成状态。

未经用户在当前任务中明确要求，不执行 commit、amend、squash、merge、rebase、tag
或 push。
