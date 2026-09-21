# Ceph object HP 浅耦合重构

范围：仅 dev 的 Ceph 接入，不修改缓存、ICFS、线上部署或 Git 历史。

已确认方案：
1. 算法目录仅依赖标准 C++；保留 C4、对象键、计数/标签窗口及快照策略。
2. OSDService 持有一个 PIMPL 适配模块，替代进程全局状态；关闭状态不创建森林，正常退出在调用方停止后 join 工作线程、停止 Trace、移除 perf logger。
3. PrimaryLogPG 的四个原位置各保留一次观察调用；有效长度过滤归适配层。删除 WRITESAME 原操作覆盖参数，转换后计入 WRITE，采样数不变。
4. OSD 命令注册/处理归适配模块，MGR HP 命令读取与分发移入专用模块。保留命令、输出、异步 sent 语义和 Trace。

实施和验证：
- 先运行现有模型/并发/Trace 与状态输出基线。
- 新增独立核心及生命周期测试，先确认其在旧实现失败。
- 实现核心边界、OSD 实例和命令封装，再提取 MGR 分发。
- 验证独立编译（无 Ceph 生成头文件/库）、关闭状态不分配模型、复位/启停、对象键隔离、线程与 Trace 退出；运行既有回归、适配层命令和 MGR 输出验证；编译 ceph-osd/ceph-mgr；git diff --check。
- 不把离线回归表述为线上五负载验收或性能提升；未部署，线上效果待确认。

生命周期约束：宿主在 shutdown 前注销命令并停止请求线程，shutdown 不与观察/命令并发；后台回调必须在 logger 和模块状态销毁前退出。正常 disable/reset 保留已有命令语义，disable 并非卸载模块。

## 完成与验证记录（2026-09-21）

验证对象是 dev 基线 `afd18c8e01` 上的本次未提交工作区，不是已部署版本或正式五负载验收。

- OSDService 实例拥有适配模块；四个 PG hook 保留原采样位置与有效长度；WRITESAME 转换后计入 WRITE，不再传递原操作覆盖参数。
- OSD 命令注册/分发及 Trace 逻辑收拢到适配模块；MGR 读取、聚合和异步控制逻辑移入 `ObjectHeatPredictorCommands.*`。
- 核心可独立编译，默认关闭不分配森林。回调在构造时绑定实例、运行中不替换；终止时等待回调及线程退出。模块清理位于 `OSDService::shutdown()` 的末尾，在请求线程、命令入口、服务定时器及 Objecter 收尾之后。
- 新独立编译测试在旧代码首先失败于缺少 Ceph 生成头 `acconfig.h`；替换依赖后通过。随后清理了离线探针中不再需要的 Ceph 断言桩。

| 验证 | 结果 |
| --- | --- |
| 修改前模型/并发/Trace 回归、状态输出基线 | 通过 |
| `bash test_sh/test_hp_model_regressions.sh` | 通过；不引用 Ceph build 目录或库 |
| `HP_SANITIZERS=address,undefined bash test_sh/test_hp_model_regressions.sh` | 通过；含生命周期、回调退出屏障、模型策略、并发统计、Trace/回放 |
| `HP_SANITIZERS=address,undefined bash test_sh/test_hp_osd_module.sh` | 通过；真实 AdminSocket 注册、事件过滤、两实例隔离、并发控制、Trace 轮转和 logger 回收 |
| `bash test_sh/test_hp_status_output.sh` | 8 项输出测试和状态契约探针通过 |
| `cmake --build build --target ceph-osd ceph-mgr -j24` | 最终代码构建通过；现有 root 所属产物需 sudo，仅构建未安装 |
| `hp_performance_probe.cc` 独立编译 | 通过；未据此产生新的性能结论 |
| feature/config/EQ/threshold/quantile 头文件与 HEAD 对比 | 除断言提供者/名称与空白外一致 |
| `git diff --check` | 通过 |

MGR 真实集群下发、完整 OSD 启停和五负载效果仍待部署后确认；本轮没有运行这些线上验证。
本轮未执行 commit、merge、push 或部署；`ceph-merge` 未改动。
