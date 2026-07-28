# C2H Residual Model Implementation Plan

> **For agentic workers:** Execute task-by-task with test-first verification.
> Repository policy overrides generic skill guidance: leave all changes
> uncommitted unless the user explicitly requests a Git operation.

**Goal:** 在现有 V2 Trace 上严格比较持续性规则、直接ARF、单树残差和ARF残差，
判断残差模型是否有效以及是否需要ARF。

**Architecture:** 在 dev-only replay 内增加预测策略与模型工厂，保持Trace、标签和
事件顺序不变。分析脚本对逐记录输出做时间前向切分，在validation选择修正阈值，
只在test报告最终指标。

**Tech Stack:** C++17、现有Classifier/Hoeffding Tree/ARF、Python 3标准库、现有
Trace/replay。

## Global Constraints

- 不修改生产 `HeatPredictor`，除非离线结果通过设计门槛。
- 不修改 Trace schema、OSD/MGR命令或统计契约。
- 代码由 main agent 修改；子 agent 仅可监控长时间测试。
- 不执行 commit 或 push。

---

### Task 1: Replay预测策略

**Files:**
- Modify: `test_sh/hp_trace_replay.h`
- Modify: `test_sh/test_hp_trace_replay.cc`

**Interfaces:**
- Produce: `HpReplayPredictionMode`
- Produce: `parse_hp_replay_prediction_mode(const std::string&)`
- Produce: `hp_replay_base_hot(const HpTraceRecord&)`
- Extend: `HpReplayOptions` with prediction mode and residual tree count

- [x] 写失败测试：验证最近窗口访问次数与动态K的基础规则、base-hot不调用残差模型、
  base-cold使用残差概率、零投票不翻转。
- [x] 运行 `test_hp_trace_replay`，确认新测试因接口不存在而失败。
- [x] 实现最小预测策略和参数校验。
- [x] 再次运行测试并确认通过。

### Task 2: 残差模型工厂和训练门控

**Files:**
- Modify: `test_sh/hp_trace_replay.h`
- Modify: `test_sh/test_hp_trace_replay.cc`

**Interfaces:**
- Produce: `make_hp_replay_residual_model(HpReplayModelKind, size_t)`
- Produce: `hp_replay_should_train(const HpTraceRecord&, HpReplayPredictionMode)`
- Extend: `HpReplayResult` with eligible/trained residual counts

- [x] 写失败测试：base-cold正负样本都训练，base-hot样本不训练；单树、10树、
  25树工厂均可预测和clone。
- [x] 运行测试并确认按预期失败。
- [x] 使用现有 `HoeffdingTreeClassifier` 和禁用drift的ARF实现工厂与训练门控。
- [x] 运行测试并确认通过且现有direct-ARF replay行为不变。

### Task 3: CLI与逐记录输出

**Files:**
- Modify: `test_sh/hp_trace_replay.cc`
- Modify: `test_sh/hp_trace_replay.h`
- Modify: `test_sh/test_hp_trace_replay.cc`

**Interfaces:**
- Add CLI: `--prediction-mode direct-arf|persistence|c2h-ht|c2h-arf`
- Add CLI: `--residual-trees N`
- Add TSV: `base_label`, `residual_hot_probability`, `residual_applied`

- [x] 写失败测试：TSV包含新增稳定列且保持Trace原始顺序。
- [x] 运行测试并确认失败。
- [x] 实现CLI校验、汇总字段和TSV输出。
- [x] 运行测试并确认通过。

### Task 4: 五负载分析器

**Files:**
- Create: `/home/chris/ceph-test/new_workload/tools/analyze_hp_c2h_residual_replay.py`
- Create: `/home/chris/ceph-test/new_workload/tests/test_analyze_hp_c2h_residual_replay.py`

**Interfaces:**
- Consume: 五组profile replay TSV及负载metadata
- Produce: `validation_thresholds.tsv`
- Produce: `test_metrics.tsv`
- Produce: `corrections.tsv`
- Produce: `REPORT.md`

- [x] 写失败单元测试：时间切分边界、二分类指标、validation阈值选择和净修正量。
- [x] 运行Python测试并确认失败。
- [x] 实现分析器，只使用validation选择阈值，test不参与选择。
- [x] 运行Python测试并确认通过。

### Task 5: 构建和离线矩阵

**Files:**
- Output:
  `/home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_v2_c2h_residual_replay/`

- [x] 构建 `hp_trace_replay` 和 `test_hp_trace_replay`。
- [x] 运行C++ replay单元测试和相关Python测试。
- [x] 对五负载、两个OSD运行B0/B1/R1/R2/R3，全部使用recent-K override。
- [x] 运行分析器并生成中文报告。
- [x] 核对记录数、标签、阈值、训练数、fallback和每个profile的确定性。

### Task 6: 条件化在线验证

**Files:**
- Modify only if Task 5 passes: `src/heatpredictor/`
- Test: `test_sh/hp_algorithm_probe.cc`
- Output:
  `/home/chris/ceph-test/new_workload/hp_runs/reports/<timestamp>_v2_c2h_residual_online/`

- [x] 若所有残差方案未通过门槛，记录“不修改生产代码”并结束。
- [ ] 若有方案通过，先为基础规则门控、残差训练筛选和reset写失败探针。
- [ ] 实现离线胜出方案并通过算法、并发、状态和构建检查。
- [ ] 安装、`ldconfig`、重启OSD/MGR并完成五负载各一次。
- [ ] 与当前V2在线基线比较指标和预测延迟，形成最终结论。
