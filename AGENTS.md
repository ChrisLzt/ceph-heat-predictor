# 代理说明

本仓库包含 Ceph OSD object 级热度预测器原型，以及将其移植到 ICFS/IDFS 的相关记录。

修改前先阅读 `codex_docs/README.md`，然后只阅读其中标明且与当前任务相关的有效文档。

测试负载和报告在本仓库之外维护，路径为：
`/home/chris/ceph-test/new_workload/`。

除非用户明确要求检查历史代码，否则不要沿用旧 KernelDevice/BlockDevice 热度预测器的
假设。

当前实现仅位于 object 层。Ceph 特有的 object op 适配应保留在
`src/osd/ObjectHeatPredictor.*`；可复用算法应保留在 `src/heatpredictor/`。

## 统计字段变更契约

- 修改 `object_hp_status` perf 字段时，enum、`PerfCountersBuilder` 声明、实时更新、
  reset 和输出分组的顺序必须保持一致。
- 新增、删除或重命名需要由 `ceph osd hp status` 汇总的字段时，必须同步修改 OSD
  perf、MGR `object_hp_counter_fields`、派生公式和 JSON 输出。仅供 OSD 使用的字段必须
  明确标注为不参与 MGR 汇总。

## 主代理与子代理职责

使用 ChatGPT 5.6 Sol（`gpt-5.6-sol`）作为主代理。主代理负责仓库分析、实验设计、
实现方案、改动边界、任务拆分、调试决策、高风险 Ceph 操作、结果解释、代码审核以及
最终集成决策。

### 代码改动执行流程

所有代码和文档改动必须由 Sol 主代理直接完成，包括方案设计、文件编辑、调试、测试、
结果核验和必要修正。不得把代码、脚本、配置或文档的编写和修改委派给任何子代理。

子代理只允许用于监控长时间运行的构建或测试。监控任务统一使用 ChatGPT 5.6 Luna
（`gpt-5.6-luna`，reasoning effort `low`），负责轮询进程和 Ceph 状态、按固定间隔读取
日志、检查明确的完成条件，以及向主代理报告原始状态。不得使用 Terra 或其他子代理
执行代码修改、文档修改、问题诊断、修复决策、测试结果分析或实验结论判断。

每个 Luna 任务都必须说明：

- 需要观察的准确命令、进程、目录或日志；
- 任务是否只读，以及允许创建哪些文件（如有）；
- 轮询间隔和明确的成功、失败及超时条件；
- 必须返回给主代理的字段或日志片段。

Luna 子代理必须保持只读，不得修改任何文件、执行 `apply_patch`、安装软件、重启服务、
reset Heat Predictor、停止进程、选择修复方案、执行 Git 操作或得出最终实验结论。
发现异常状态时，应报告异常，并将诊断和恢复留给 Sol 主代理。

主代理仍负责根据原始日志和当前系统状态核对子代理报告。仅凭子代理报告不足以证明
构建或测试通过。如果监控任务把中间警告误报为最终失败，主代理必须检查原始输出并
修正或恢复工作流程。

## 检查级别

选择能够覆盖当前改动的最低检查级别。不得仅因为更高级别的命令可用，就默认执行更高
级别的检查。

- **L0，文档：**执行 `git diff --check`，并验证有效 Markdown 链接。
- **L1，本地算法：**L0 加 `hp_algorithm_probe` 和受影响目标的编译。适用于不需要
  运行 Ceph 进程即可验证行为的算法、参数和纯头文件改动。
- **L2，Ceph 集成：**L1 加 `codex_docs/CEPH_OPERATIONS_MANUAL.md` 中的完整构建和
  安装流程；只重启受影响的服务，等待 `active+clean`，然后 reset 并检查 Heat
  Predictor 状态。
- **L3，负载验证：**L2 加负载模型验证、每个必要正式负载运行一次、队列/drop/计数
  检查以及中文报告，并遵守
  [`/home/chris/ceph-test/new_workload/EXPERIMENT_PROTOCOL.md`](/home/chris/ceph-test/new_workload/EXPERIMENT_PROTOCOL.md)
  的执行器、归档 schema、数据一致性和结果校验约束。

在 L3 中，每个必要负载默认且只运行一次。如果结果存在巨大误差、异常或与历史结论
冲突，应在中文报告中记录并停止。是否复测以及复测次数只能由用户决定，禁止自动增加
重复次数。ASan/UBSan 仅在容器、内存或核心算法发生变化时增加；TSan 仅在并发逻辑
变化时增加；性能探针仅在性能路径变化时增加。
