# 实验执行规范

本文件只记录已经可用的实验工具和归档约束，不维护功能 TODO。运行次数、profile 和
选择指标以 `CODEX_CEPH_TODO.md` 为准。

## 工具

- 执行器：`/home/chris/ceph-test/new_workload/run_hp_matrix.sh`
- 汇总器：`/home/chris/ceph-test/new_workload/tools/summarize_hp_matrix.py`
- 构建和部署：`CEPH_OPERATIONS_MANUAL.md`

执行器支持 profile 参数校验、`--dry-run`、显式 repetition、统一 reset、
`active+clean` 检查、OSD 实时训练队列等待和结果归档。默认 `--repetitions 1`；大于 1
的值和 `--repetition-start` 只能在用户明确要求复测后使用。

## 每轮文件

- 测试结束后的单个 `hp_status.json`
- `metadata.json`、`run.log` 和负载原始输出
- 汇总后的 `results.tsv`

不保存 `hp_status_before.json` 或逐 OSD status 文件。逐 OSD status 只用于等待训练队列
清空；最终统计统一来自一次 MGR `ceph osd hp status -f json-pretty`。

`metadata.json` 记录两个仓库的 commit、关键配置 SHA-256、内核、Ceph 版本、开始/
结束时间和集群状态。`results.tsv` 至少包含 profile/workload、I/O/labeled/pending/drop、
TP/FP/TN/FN、accuracy/precision/recall、预测/实际热比例、热阈值与预测阈值、置信度、
快照数、预测延迟、rate 和 MB/s。

## 有效性检查

- `io - pending == labeled`
- `hp_train_drop_count == 0`
- 测试前 PG `active+clean`
- 队列清空后再保存最终状态
- 汇总器从 TP/FP/TN/FN 重算指标，并与 MGR 输出核对

2026-07-12 Otsu 矩阵已经验证 dry-run、reset、归档和汇总流程；28 次历史有效负载的
accuracy 重算与 MGR 最大误差为 `0.000492` 个百分点。
