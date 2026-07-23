# Heat Predictor TODO

当前执行顺序固定为：先完善并冻结现有“10秒后 object 总热度 + Otsu 阈值”版本，
再创建以未来访问收益定义冷热的新版本。两个版本不得在同一轮实验中交叉修改标签、
feature 和模型。已完成记录在 `CODEX_CEPH.md`、`TRACE_DATASET.md` 和 ceph-test 报告中，
不再把已完成清单塞进 TODO。Trace 是 `dev` 的离线设计和工具参考，不是新的活动 TODO。

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
- [x] 2026-07-23 已通过 replay/parity、L1、L2，并以 MapReduce 单轮完成必要的 L3
  验证；不得把未通过 gate 的候选发布到 `main`。
- [x] 2026-07-18 已增加 ARF warning/drift/后台树生命周期遥测，并在三种 no-migration
  Trace 上比较 R0、grace=50 和更敏感 ADWIN。后两者平均 C2H Recall 只提升
  `0.49/0.13 pp`，未达到 `+10 pp` gate，因此默认 grace/warning/drift 参数保持 R0，
  不运行组合参数。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260718_075405_arf_adaptation/REPORT.md`。
- [x] 2026-07-18 已验证20棵长期树+5棵轮换快速树。5000/10000训练样本生命周期分别
  触发890/443次重建，但平均C2H Recall相对baseline下降`1.19/0.50 pp`，BAcc均下降
  `0.10 pp`；默认快速树数量保持0，不再继续用更快ARF遗忘解决C2H。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260718_084512_arf_fast_cohort/REPORT.md`。
- 当前 H1 版本固定使用3个已选 feature；实验 profile 仅留在 `dev` 供 replay/Trace
  对照，不进入 `main` 的确定方案。

## C. P1：全 I/O 预测错误诊断

- [x] 以每条 I/O 在预测时刻的结果和10秒后实际标签构造标准
  `TN/FP/FN/TP`；这些结果不再命名为 C2C/C2H/H2C/H2H，也不解释为 object 状态迁移。
- [x] 复用已有 MapReduce、GraphChi、HPC no-migration Trace，稳态主区间固定为
  `[120s,600s)`，不重跑 Vdbench。
- [x] 离线分析输出全程/稳态矩阵、0.1预测概率分箱、四类结果的 feature 精确分位数、
  object macro Accuracy 与 Top 1/5/10% 错误集中度、30秒时序。
- [x] 2026-07-23 结合五负载在线混淆矩阵、30秒阶段统计和已有因果 Trace 完成归因：
  FN 占全部错误 `68.85%`；热点切换样本占阶段样本 `17.15%`、贡献阶段错误
  `24.31%`；无 drop、预测错误或容量淘汰。主要瓶颈为未来10秒访问不可见，其次为阶段
  切换适应滞后和动态 Otsu 边界附近标签歧义。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260723_030335_d1_heat_gt20_all5/REPORT.md`。
- C2H/H2C 样本量较小，相关 oracle、OSD context、快速树和状态迁移统计暂时冻结；除非
  全 I/O 证据重新指向该问题，否则不继续优化其 Recall。

## D. 投影残差预测

同源标签检测表明，无未来访问的10秒热度投影已达到 `81.68%` Accuracy；在线 ARF
达到 `86.41%`，但对真正由未来访问改变标签的热样本 Recall 只有 `53.21%`。后续模型
只处理投影无法确定的未来访问贡献，不再重复学习确定性热度衰减。

- [x] **残差分类：**若 `projected_heat = current_heat * 0.20` 高于预测时热阈值则直接
  判热；其余投影冷样本使用固定 `0.50` 阈值的分类模型预测未来10秒访问能否使其跨过
  热阈值。训练和验证只使用投影冷样本，最终指标在完整测试集计算。
- [x] **未来热度回归：**使用预测时 feature 估计未来新增热度
  `label_heat - projected_heat`，计算
  `predicted_final_heat = projected_heat + predicted_future_heat`，再与预测时热阈值比较。
  回归输出必须非负，不得使用截止时刻阈值或未来访问计数作为输入。
- 两种方案均使用最新五负载 Trace 的时间前向切分：`[120s,350s)` 训练、
  `[360s,470s)` 验证、`[480s,600s)` 测试，两个10秒隔离区不参与训练。
- 候选配置只根据验证集完整流水线 Accuracy 选择；测试集仅用于最终一次评估。对照项
  必须包含在线 ARF、无未来访问投影和投影规则与原 ARF 的直接组合。
- 重点报告完整测试集 Accuracy/BAcc，以及投影冷子集的 Accuracy、未来访问决定型热
  样本 Recall 和模型相对无未来访问投影的增益。离线收益稳定后才允许修改在线源码。
- 2026-07-20 五负载测试结论：在线 ARF、残差分类、未来热度回归的五负载平均 Accuracy
  分别为 `87.39%/87.16%/86.35%`；样本加权的未来访问决定型热样本 Recall 分别为
  `60.15%/50.69%/37.03%`。两种残差方案均未通过 Accuracy gate，不进入在线源码。
  报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260720_231339_projection_residual_models/REPORT.md`。

## E. 长周期 Object 历史诊断

短窗口和 fast/slow EWMA 已无稳定收益，但未来访问决定型热样本中有大量 object 在过去
10秒没有访问。先复用最新五负载 Trace 检查更长时间尺度的信息，不修改在线源码：

- [x] B1：在当前3维基础上增加30/60/120秒 object 访问计数和 object/OSD 60秒访问占比。
- [x] B2：增加最近4次访问间隔的均值、变异系数和最新间隔/均值比例。
- [x] B3：组合 B1 与 B2；O1 使用完整600秒 object 频率和 rank percentile，仅作为明确
  存在未来泄漏的身份 oracle，不允许进入在线实现。
- 时间前向切分保持 `[120s,350s)` 训练、`[360s,470s)` 验证、`[480s,600s)` 测试；
  同时间戳事件不进入过去窗口，测试集不参与模型或阈值选择。
- Gate：五负载平均 Accuracy 至少 `+0.5 pp`，任一负载不低于 `-0.2 pp`，且 Balanced
  Accuracy 和未来访问决定型热样本 Recall 均不下降。未通过则停止该方向。
- 2026-07-21 结论：B3 相对相同离线模型 B0 的平均 Accuracy/BAcc/未来访问决定型热
  Recall 分别变化 `+0.65/+1.11/+0.91 pp`，最差单负载 `-0.06 pp`，说明长期历史存在
  小幅增量信号；但相对现有在线 ARF，AI 训练 Accuracy 下降 `2.00 pp`，加权未来访问
  决定型热 Recall 下降 `5.18 pp`，未通过部署 gate。完整运行 object 频率 oracle 的平均
  Accuracy 也只比在线 ARF 高 `1.06 pp`，且 AI 训练下降 `3.03 pp`。因此不向在线 OSD
  增加精确窗口计数状态，本方向结束。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260721_003648_long_horizon_object_history/REPORT.md`。

## F. 共同实验约束

- 参数以 `src/heatpredictor/hp_config.h` 为准。
- 一轮一次；异常不自动复测。
- 报告路径为 `/home/chris/ceph-test/new_workload/hp_runs/reports/` 的带日期目录。
- 无未来泄漏；`hp_train_drop_count=0`。
- P1 首轮只做离线分析；只有通过 gate 后才修改在线源码。异常结果不自动升级为复测或
  新的活动方向。

## G. Otsu投票人口诊断

- [x] 2026-07-22 复用最新五负载D2 Trace，按OSD重建每object最新总热度投票，并对比
  全部object、高于热度下限object和最近10秒访问object三种人口。共验证`1,456,058`
  条Trace和`1,440`个稳态快照；D0阈值重建P95相对误差仅`0.01%`。
- 全部object中最低热度bin占比中位数为`87.97%`；剔除该堆积后只保留`12.05%`投票，
  Otsu候选阈值由`50.27`提高到`123.63`（`2.39x`），分离度由`0.780`降至`0.606`。
  原模型不重训时，D1实际热比例为`61.05%`，Accuracy为`84.30%`，较D0下降
  `2.11 pp`。
  证据支持当前Otsu主要区分长期未访问与近期活跃object，而不是活跃object内部稳定的
  冷热双峰。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260722_190353_otsu_population_diagnostic/REPORT.md`。
- [x] 2026-07-22 已按D1因果阈值逐OSD重算预测时feature 0和10秒标签，并让同配置的
  StandardScaler与25棵ARF从空模型重新训练。五负载稳态Accuracy为`88.25%`，相对
  D1零适应提高`3.95 pp`，相对D0提高`1.84 pp`；AI推理、AI训练、GraphChi和HPC均
  高于D0，MapReduce低于D0 `1.19 pp`。这说明D1标签体系可以被现有模型学习，但尚不
  能证明其冷热语义更合理，也不能直接替换线上D0。报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260722_193706_d1_retrained/REPORT.md`。
- [x] 2026-07-22 已完成D1在线实现和五负载单轮验证。每个高于热度下限的object只维护
  一个到期索引节点，到达下限时仅删除Otsu投票，不删除`heat_map`/LRU状态。全程样本
  加权Accuracy/BAcc/Precision/Recall为`87.64%/88.09%/93.07%/85.92%`，实际热比例
  `60.37%`；最终Otsu投票仅占heat state的`0.90%~2.06%`，证明最低bin堆积已在线移除。
  五项均无评估或训练丢弃，报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260722_202409_d1_online/REPORT.md`。
- [x] D1到期索引已由`std::multimap`替换为每object单节点的索引最小堆；object刷新复用
  Otsu顺序链表节点，并移除未参与当前三维模型的独立长期访问窗口状态。新增受保护状态、
  heat state峰值和LRU淘汰统计，用于后续容量诊断。MapReduce单轮未发现端到端行为或
  吞吐回归，但索引堆EQ微基准常数高于旧实现，不能作为已证明的提速项；报告：
  `/home/chris/ceph-test/new_workload/hp_runs/reports/20260723_002523_d1_indexed_heap_mapreduce/REPORT.md`。
- [x] 当前 H1 目标固定为“10秒截止时相对 Otsu 阈值的热状态”，D1 只在当前衰减总热度
  严格大于20的活跃人口内维护阈值。基于未来访问收益的新冷热定义独立放入 H2，不再把
  D1/D2反事实阈值混入当前版本。

## H. 版本推进顺序

### H1. 完善并冻结当前定义

当前版本继续使用：每条 I/O 在预测时刻取 feature，10秒后将 object 衰减总热度与该
截止时刻生效的 Otsu 热阈值比较得到标签。完成以下事项前，不修改标签语义：

- [x] 完成 P1 全 I/O 错误诊断，区分未来访问不可见、Otsu 边界翻转、feature 不可分、
  冷启动和运行异常的贡献；后续一次只验证证据最强的一个方向。
- [x] 核验 Otsu 每100次 object 热度观测或最长约1秒重算、时间归一化 EMA `0.10`、
  `initializing/tracking/holding` 状态和完全空闲时延迟更新的行为是否符合当前定义。
- [x] 在现有标签下完成必要的 feature、容量、预测延迟和队列/drop 检查；只保留通过
  Accuracy gate 的改动，不为新标签提前增加线上状态。
- [x] 固定参数、统计字段、reset 行为和测试基线，在文档中明确当前版本预测的是
  “10秒截止时相对 Otsu 阈值的热状态”，不是 object 的长期冷热身份。
- [x] 2026-07-23 完成 L1/L2、完整构建安装和五负载单轮 L3。五负载共
  `1,453,610` 个已评估 I/O，Accuracy/BAcc 为 `87.83%/88.28%`，无评价/训练丢弃、
  预测错误或 LRU 淘汰；相对旧 D1 Accuracy `+0.19 pp`、预测延迟 `-16.65%`。当前
  H1 版本冻结；是否开始 H2 由用户确认。

### H2. 基于未来访问收益的新冷热定义（暂缓实施）

新版本不再使用模拟热度或 Otsu 生成真实标签。对时刻 `t` 的一次 object I/O，排除当前
I/O，只统计 `(t, t+10s]` 内同一 object 的后续访问：

```text
future_access_count >= K  -> hot
future_access_count < K   -> cold
```

- `K` 必须由 object 提升/迁移成本和后续访问收益决定；在当前等大小4MiB整 object 读
  负载上，先离线比较 `K=1/2/3`，不得仅按最高 Accuracy 选择。
- 生产环境升级为 `future_benefit > promotion_cost`，按读写类型、字节数、延迟收益和
  高速层容量计算；容量有限时按未来收益排序，只将可容纳的最高收益 object 定义为热。
- 当前模拟热度保留为近期访问强度 feature；fast/slow 访问率、上次访问间隔等只能使用
  预测时刻之前的信息。Otsu 降为当前热度分布观测或冷启动 fallback，不参与真实标签。
- 新版本必须从冻结的 H1 基线创建独立实验实现，复用同一 Trace 做因果离线比较；禁止
  使用未来访问次数、截止时刻热度或截止时刻阈值作为输入 feature。
- 首轮必须报告 Accuracy、Balanced Accuracy、Precision、Recall、Specificity、实际热
  比例和各 `K` 的迁移收益解释，并与 H1 分开命名、分开保存报告。
- 只有标签定义、`K` 的成本依据和离线 gate 均确认后，才允许修改在线 OSD 代码。
