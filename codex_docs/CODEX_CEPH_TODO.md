# Heat Predictor TODO

当前活动 TODO 只保留 baseline 验证和 P1 全 I/O 预测错误诊断。已完成记录在
`CODEX_CEPH.md`、`TRACE_DATASET.md` 和 ceph-test 报告中，不再把已完成清单塞进 TODO。
Trace 是 `dev` 的离线设计和工具参考，不是新的活动 TODO。

## A. 冻结项/不再继续

- 稳定算法基准固定为：object 总热度 Otsu 和标签、10秒标签窗口、时间归一化固定 EMA
  `0.10`、固定预测阈值 `0.50`、冷热单位样本权重、25棵树。对应的新增热度、动态预测
  阈值、动态 EMA 和类别权重实现已从活动源码删除。
- 已有结论不再重跑：H1/P1 动态阈值、D0/D1、新/总热度数据源、去掉 StandardScaler、
  减慢热度衰减、同 object 过去窗口/旧标签专用模型、类别权重、森林参数和一般性能调优。
- 未来 oracle O1/O2 绝不能成为在线 feature；它们只用于离线后验依赖诊断。
- 不复制已完成实验的详细公式、长结果或历史矩阵；只保留当前决策所需的结论。

## B. Baseline 验证

- 默认模型使用3个已选 feature；未通过 gate 的候选 feature 及其额外 object 状态已经
  移除，`dev` 继续保留 Trace/replay。
- [ ] 至少验证 replay/parity 和 L1；发布前必须通过 L2，若行为或指标可能变化再进行一次
  L3；不得把未通过 gate 的候选发布到 `main`。
- [x] 2026-07-18 已增加 ARF warning/drift/后台树生命周期遥测，并在三种 no-migration
  Trace 上比较 R0、grace=50 和更敏感 ADWIN。后两者平均 C2H Recall 只提升
  `0.49/0.13 pp`，未达到 `+10 pp` gate，因此默认 grace/warning/drift 参数保持 R0，
  不运行组合参数。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260718_075405_arf_adaptation/REPORT.md`。
- [x] 2026-07-18 已验证20棵长期树+5棵轮换快速树。5000/10000训练样本生命周期分别
  触发890/443次重建，但平均C2H Recall相对baseline下降`1.19/0.50 pp`，BAcc均下降
  `0.10 pp`；默认快速树数量保持0，不再继续用更快ARF遗忘解决C2H。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260718_084512_arf_fast_cohort/REPORT.md`。
- feature 尚未最终冻结，因此暂不删除 margin、access-rate 和 scaler 的实验
  profile；最终选择后再精简这些编译期入口。

## C. P1：全 I/O 预测错误诊断

- [x] 以每条 I/O 在预测时刻的结果和10秒后实际标签构造标准
  `TN/FP/FN/TP`；这些结果不再命名为 C2C/C2H/H2C/H2H，也不解释为 object 状态迁移。
- [x] 复用已有 MapReduce、GraphChi、HPC no-migration Trace，稳态主区间固定为
  `[120s,600s)`，不重跑 Vdbench。
- [x] 离线分析输出全程/稳态矩阵、0.1预测概率分箱、四类结果的 feature 精确分位数、
  object macro Accuracy 与 Top 1/5/10% 错误集中度、30秒时序。
- [ ] 根据稳态结果确认主要瓶颈属于概率边界、feature 不可分、少数 object 集中错误，
  还是局部时间段退化；一次只选择一个证据最强的方向进入受控实验。
- C2H/H2C 样本量较小，相关 oracle、OSD context、快速树和状态迁移统计暂时冻结；除非
  全 I/O 证据重新指向该问题，否则不继续优化其 Recall。

## D. 共同实验约束

- 参数以 `src/heatpredictor/hp_config.h` 为准。
- 一轮一次；异常不自动复测。
- 报告路径为 `/home/chris/ceph-test/new_workload/hp_runs/reports/` 的带日期目录。
- 无未来泄漏；`hp_train_drop_count=0`。
- P1 首轮只做离线分析；只有通过 gate 后才修改在线源码。异常结果不自动升级为复测或
  新的活动方向。
