# Main/Dev 分支流程

本文档只规定 `main` 和 `dev` 的长期职责。代理行为约束以仓库根目录
`AGENTS.md` 为准，算法和接口说明由 `CODEX_CEPH.md` 维护。

## `main`

`main` 是可构建、可部署和可打标签的生产分支，只保留确定的 Heat Predictor、
Ceph OSD/MGR 适配、生产配置及稳定文档。它不保留 Trace、测试探针、实验 profile、
多套算法分支、TODO、实验计划或原始报告。

## `dev`

`dev` 是日常开发和验证分支，以当前 `main` 为生产基线，可额外保留未发布生产改动、
Trace/replay、离线分析、测试、TODO 和实验文档。

## 开发与发布

1. 日常开发、调试和负载验证默认在 `dev` 进行。
2. 生产改动与 dev-only 改动必须分开；Trace、测试/分析和开发文档统一压缩为
   `main` 之上的一个 dev-only 汇总提交，不再按类别拆分。
3. 验证通过后只选择生产代码和稳定文档发布到 `main`，不得整体合并 `dev`。
4. `main` 通过生产构建和必要集成检查后才能打标签，标签只指向 `main`。
5. 发布后以新的 `main` 为生产基线重新生成 dev-only 汇总提交，使 `dev` 保持为
   `main` 加一个提交。
6. 直接在 `main` 修复的问题也必须同步回 `dev`。
7. 未经用户在当前任务中明确要求，不执行 commit、rebase、merge、tag 或 push。
