# Cold-to-Hot Crossing Feature Implementation Plan

> 在现有 dirty `dev` 工作树中执行，因为本实验依赖尚未提交的 C4 与 Trace 代码。除非用户
> 明确要求，不创建 commit，不 push。

## Task 1：先锁定失败语义

**修改：** `test_sh/hp_algorithm_probe.cc`

1. 增加首次有效间隔直接初始化观测访问率的测试。
2. 增加下降趋势不扣减 activation rate 的测试。
3. 增加访问前已热输出0、冷对象 crossing margin 正负方向的测试。
4. 更新队列集成测试和最终 feature 顺序断言。
5. 编译并确认测试因生产接口尚不存在而失败。

## Task 2：最小实现

**修改：** `hp_config.h`、`hp_types.h`、`hp_features.h`、
`hp_evaluation_queue.h`、`heat_predictor.h`

1. 增加观测访问率 helper；第二次访问直接初始化 fast/slow rate。
2. 将趋势外推改为只奖励正趋势。
3. 用 `cold_to_hot_crossing_margin` 替换旧第四维及相关命名。
4. 使用 `EvaluationQueue` 实际的 `heat_increment`，保证测试构造参数与线上一致。
5. 增加 feature schema version，并纳入 Trace 配置哈希。

## Task 3：本地验证与部署

1. 运行算法探针的 RED/GREEN 流程。
2. 运行 ASan/UBSan、性能探针、Trace C++ 回放及全部 Python 测试。
3. 执行完整 Ninja 构建、安装、`ldconfig` 和 Ceph 服务重启。
4. 确认 OSD 全部 up/in、PG active+clean、Heat Predictor reset 正常。

## Task 4：五负载在线测试

1. 使用 D2 baseline 参数，五个 Vdbench 负载各运行一次。
2. 每10秒保存 MGR 状态；每个负载开启独立 Trace session，结束后排空并停止。
3. 生成中文报告，对比三特征 C4 和旧未来总热度第四维。
4. 报告整体、阶段、冷转热子集、预测延迟及所有 queue/drop/error 状态。

## Task 5：文档收尾

更新 `CODEX_CEPH.md` 与 `CODEX_CEPH_TODO.md`，只记录实际实现和测试结论，不把单轮
波动写成确定性结论。
