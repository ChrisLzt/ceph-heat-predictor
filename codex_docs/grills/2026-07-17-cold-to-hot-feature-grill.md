# 冷转热 Feature Grill

## Goal

优化当前未来热度投影 feature，使模型更早识别“预测时仍冷、10秒标签窗口结束时变热”的
object，同时不把无法从当前信息判断的突然热转冷作为该 feature 的主要目标。

## Context Read

- 当前输入为当前热度 margin、前次访问间隔、当前热度和未来热度投影 margin。
- 当前访问率使用2秒/10秒时间感知 EWMA，初值均为0。
- 第二次访问的首次间隔样本仍与0做 EWMA；当 `dt -> 0` 时，首次估计趋近
  `1/tau`，不能反映真实的高瞬时访问率。这会延迟冷转热识别。
- 当前投影把未来10秒访问率视为恒定，既混合当前热度和新增访问贡献，也会对突然停止访问
  的 object 保留过强预测。
- 上一轮四特征在线结果整体 Accuracy/BAcc 提高 `0.2969/0.1263 pp`，但热点迁移阶段
  Accuracy 下降 `0.4632 pp`，尚未证明冷转热识别改善。

## Non-Goals

- 不试图在没有任何新观测时预测外部负载导致的突然热转冷。
- 不改变 Otsu、标签语义、预测阈值、训练权重或 ARF 参数。
- 不增加 object 访问时间序列或后台定时器。

## Decisions

### Decision 1: Accuracy 与冷转热 Recall 的优先级

**Question:** 是否允许以增加普通冷样本 FP 为代价提高冷转热 Recall？

**Recommended answer:** 不把两者无条件交换。冷转热 Recall 必须提高，同时五负载平均
Accuracy/BAcc 不下降，任何单负载 Accuracy 下降不超过 `0.2 pp`，当前冷样本上的
activation precision 不能明显下降。

**User decision:** 首轮暂不设置 Accuracy/BAcc、FP 或 activation precision 的硬约束，
先观察冷转热信号能提升到什么程度。

**Reasoning:** 项目此前把 Accuracy 作为硬指标；只提高热预测倾向也能增加 Recall，但不代表
feature 真正识别了转热趋势。

**Dependencies:** Trace 分析需按预测时当前冷热与 deadline 标签构造转换子集。

**Failure mode if wrong:** 模型通过扩大预测热比例获得更高 Recall，同时增加大量 FP。

### Decision 2: Feature 语义

**Question:** 继续输出“预计总热度 margin”，还是改为只表示冷 object 填补热度缺口的证据？

**Recommended answer:** 改为非对称的 `cold_to_hot_crossing_margin`。对预测时仍冷的 object，
比较预计未来新增热度与跨越热阈值所需热度；对预测时已热的 object 输出中性值0。

**User decision:** 采用推荐方案，用 `cold_to_hot_crossing_margin` 替换原
`expected_future_heat_margin`，维持四维输入，不增加第五维。

**Reasoning:** 当前热度已有独立 feature。新 feature 应隔离未来访问贡献，显式表达冷转热，
避免重复编码总热度并干扰热转冷样本。

**Dependencies:** 首次有效间隔必须正确初始化访问率。

**Failure mode if wrong:** 新维度继续与当前热度高度相关，森林只能学到“已经热”，不能学到
“正在变热”。

## Domain Terms

- **访问前冷 object**：本次 I/O 增加热度之前，衰减到当前时刻的 object 热度不高于当前
  热阈值。当前每次 I/O 固定增加 `HP_HEAT_INCREMENT`，因此可由
  `max(0, heat_after_current_access - HP_HEAT_INCREMENT)` 得到访问前热度，无需新增状态。
- **冷转热样本**：访问前为冷，但本次预测对应的10秒 deadline 标签为热。不能用
  `heat_after_current_access <= threshold_at_prediction` 定义，因为当前每次访问增加100，
  而实测阈值常在20至30附近，访问后几乎都会暂时越过阈值。
- **activation precision**：预测时仍冷且被预测为热的样本中，deadline 实际为热的比例。
- **cold-to-hot recall**：全部冷转热样本中被预测为热的比例。
- **crossing margin**：预计未来新增热度相对跨越当前热阈值所需热度的对数比值。

## ADR Candidates

无。该 feature 仍处于 `dev` 实验阶段，容易撤回，不需要 ADR。

## Open Questions

无阻塞问题。首轮不设置整体指标硬门槛，结果按合成语义、转换子集和完整负载三个层次报告。

## Ready For Brainstorming

- [x] Goal is clear
- [x] Non-goals are clear
- [x] Key decisions have user answers
- [x] Blocking facts have been checked from source
- [x] Domain terms are named consistently
- [x] Risks have owners or mitigations
- [x] No blocking open questions remain
