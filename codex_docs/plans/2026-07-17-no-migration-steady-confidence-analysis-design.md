# 无热点迁移稳态置信度分析设计

## 目标

使用固定热点负载隔离热点迁移影响，分析稳态预测错误的置信度分布，并确认各置信度区间
主要由 FP 还是 FN 构成。

## 测试范围

只运行 MapReduce、GraphChi 和 HPC 三种 `no_migration_workload`，每种负载运行一次并采集
完整评估 Trace。AI training 和 AI inference 不进入本轮测试。每个负载开始后的前120秒
视为冷启动并排除，剩余480秒视为稳态；阶段中的顺序/随机读取变化仍保留，因为热点 object
集合没有迁移。

## 统计口径

Trace 已包含 `predicted_hot_probability`、预测阈值、预测类别、真实标签和预测时间，无需
修改 OSD schema。对稳态 FP 使用 `confidence=p_hot`，对稳态 FN 使用
`confidence=1-p_hot`。置信度按 `[0.5,0.6)`、`[0.6,0.7)`、`[0.7,0.8)`、
`[0.8,0.9)` 和 `[0.9,1.0]` 分桶。

输出按负载和三负载汇总的 FP/FN 数量、各桶占对应错误类型的比例、各桶 FP/FN 构成，
并同时报告稳态 Accuracy、Balanced Accuracy、Precision 和 Recall。生成
`steady_error_confidence.tsv` 和中文 `REPORT.md`。

## 实现边界

扩展现有负载 runner，使其可以从 `no_migration_workload` 目录选择负载；默认
`new_workload` 行为保持不变。置信度统计只进入离线 Trace 分析工具，不增加在线 perf
计数，不改变模型、标签、热阈值、预测阈值或 Trace schema。结果必须记录 Ceph commit、
配置哈希、Trace 完整性以及 EQ/训练/Trace drop 状态。
