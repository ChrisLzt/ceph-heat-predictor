# dev当前实现与实验候选状态（2026-09-26）

本文说明 `/home/chris/ceph-heat-predictor` 的dev算法。本表首次核对时基线HEAD为`691603cbd16ddce22dbd5beca7c56248a6577a94`，近期变更当时尚未提交、推送或部署；后续提交包含下述实现。发布状态与线上测试结果以各批次发布记录为准，不能把离线结果当作线上验收。

## 当前默认算法

|项目|dev工作区实现|
|---|---|
|模型|25棵ARF，固定seed591422|
|输入|C4共7维，保留旧热度；慢历史30/60秒、观察时长修正和倍率上限4|
|特征具体内容|10秒计数相对阈值裕量、上次访问间隔、旧热度、2秒计数外推裕量、2秒计数、30/60秒慢历史裕量|
|输入标准化|不使用StandardScaler；原特征log2编码和C4修正仍在|
|随机特征|每叶固定随机选3维，即round(sqrt(7))|
|训练抽样|Poisson λ=6|
|叶预测|MC类别累计概率；没有按时间衰减类别累计|
|森林投票|各树累计普通Accuracy；零分回退权重1|
|切点|每个所选特征10个历史min/max间等距内部点；整个叶子原候选全无效时补冷热均值中点|
|中点限制|两类有观测、均值有限、收益有限且>0、两侧估算权重占比>1%；原候选存在时完全不启用中点|
|分裂|每50新增加权样本尝试；子叶等待起点是继承权重；delta=.01、tau=.05|
|收益门槛|原数值候选没有显式>0门槛；中点另要求>0。单有效候选直接分裂，多候选按Hoeffding差值或tau|
|纯度保护|纯叶或多数类权重占比>99%继续等待，不因纯度永久停用|
|空候选/深度限制|均已删除；depth仅作结构记录|
|内存保护|每棵树估算100MiB预算，每100万累计训练权重检查；可停用叶子，没有恢复机制，不是RSS硬上限|
|漂移|默认NeverDrift，ADWIN/背景树未启用|
|预测目标|未来10秒访问次数与记录阈值比较，Otsu相关机制保留|
|历史/热度|历史窗口10秒，短窗口2秒；热度10秒保留10%|
|预热/快照|已发布训练样本不足3000用过去热规则；每2000样本或2秒发布|
|模型策略版本|6；参与Trace配置hash，特征维数/Trace记录格式未改|

代码依据：`src/heatpredictor/hp_config.h`、`hp_features.h`、`heat_predictor.h`、`include/{ARFClassifier,GaussianSplitter,HoeffdingTree,HoeffdingTreeClassifier,TreeBase}.*`。

## 接入、缓存、Trace

HP在BlueStore真实存储对象读写路径观测，由OSD管理生命周期，关闭路径优化已在dev。缓存来自历史merge 140169f的OnodeCache模块，LRU/S3FIFO、预取及预算/回收功能已迁入dev；与HP独立控制。代码默认HP关闭、Onode LRU、预取关闭，服务器实际配置可以覆盖默认，不能据此推断在线开关。

dev原有Trace、分析与离线回放工具保留。缓存迁入范围与验证见[CACHE_DEV_INTEGRATION.md](CACHE_DEV_INTEGRATION.md)。本轮不改ICFS；不能声称ICFS算法已自动同步本次参数和中点。

## 已测试但未采用为默认算法的主要方案

以下报告各有自己的旧版本/数据/参数，不把其中“未修改生产”的历史描述当作今天状态；未采用也不等于都应继续合入。

|方案|目前状态与已有证据|实验依据|
|---|---|---|
|经验计数替代高斯收益评估；2048样本缓冲、分位数切点|未合入；收益随负载变化，额外内存/计算代价|[经验切点](</home/chris/ceph-tool/results/hp-empirical-splits-20260924/REPORT.md>)、[关闭标准化+缓冲](</home/chris/ceph-tool/results/hp-raw-buffer-splits-20260924/REPORT.md>)|
|缓冲中全部相邻间隙搜索|未合入；更充分搜索没有稳定优于少量分位数|[全间隙复测](</home/chris/ceph-tool/results/hp-raw-all-gaps-20260925/REPORT.md>)|
|16/32/64桶直方图切点|未合入；有负载收益但WRF退步，桶数增加不保证改善|[直方图](</home/chris/ceph-tool/results/hp-histogram-splits-20260924/REPORT.md>)|
|历史min/max收缩、原候选无效时条件收缩|未合入；条件收缩旧Trace未产生可用备用候选，与刚加入的均值中点不是同一方案|[条件收缩](</home/chris/ceph-tool/results/hp-conditional-range-20260922/REPORT.md>)|
|叶子统计直接重置、后台候选叶/子树替换|未合入；仅部分AI推理小幅改善，未形成普遍收益|[叶恢复](</home/chris/ceph-tool/results/hp-leaf-recovery-20260925/REPORT.md>)|
|已知阶段清叶、整森林重建、旧延迟标签丢弃|未合入；WRF下降，GraphChi提升可由预热规则解释|[阶段重置](</home/chris/ceph-tool/results/hp-phase-reset-20260925/REPORT.md>)|
|阶段过渡时只回退规则、保留原模型|仅做过已知阶段/预热掩码的输出层对照；未有可直接部署的在线触发方案|[阶段重置G对照](</home/chris/ceph-tool/results/hp-phase-reset-20260925/REPORT.md>)|
|更多访问间隔、多尺度计数、观察时长等特征|未合入；ExtraTrees筛选有价值，但11/14维在线算法回放未显示整体改善|[筛选](</home/chris/ceph-tool/results/hp-feature-screen-20260924/REPORT.md>)、[回放](</home/chris/ceph-tool/results/hp-feature-online-replay-20260924/REPORT.md>)|
|2秒辅助预测模型、访问间隔变化第8特征|未合入；辅助方案Recall下降，节奏特征有负载取舍|[短期辅助与节奏](</home/chris/ceph-tool/results/hp-short-aux-cadence-20260924/REPORT.md>)|
|同文件其他对象访问上下文|未合入；该历史实现没有整体收益|[候选消融](</home/chris/ceph-tool/results/current-candidate-ablation-20260918/REPORT.md>)|
|热纠偏|未作为默认预测策略；C4基础上历史净少错13条，收益不足以优先增加策略|[候选消融](</home/chris/ceph-tool/results/current-candidate-ablation-20260918/REPORT.md>)|
|去掉旧热度输入|未采用；历史C4下五负载Acc均下降，所以当前仍保留|[旧热度消融](</home/chris/ceph-tool/results/old-heat-ablation-20260918/REPORT.md>)|
|ADWIN与背景树|已有核心/回放支持，默认关闭；不能写成“完全没有代码”|[漂移+短周期](</home/chris/ceph-tool/results/drift-short-window-20260916/REPORT.md>)|
|NBA叶预测、River全默认组合|当前仍MC；官方River组合已测试，但不能将组合结果当作C++单独启用NBA的消融|[官方River](</home/chris/ceph-tool/results/hp-river-full-20260925/REPORT.md>)|
|2/5秒未来窗口、1/0.5秒短周期等历史配置|不是当前默认；曾测试/曾部署部分配置，后来按用户要求回到10秒/2秒|[窗口对照](</home/chris/ceph-tool/results/future-window-2s5s-20260915/REPORT.md>)、[短周期](</home/chris/ceph-tool/results/drift-short-window-20260916/REPORT.md>)|
|选择性标准化|当前采用全部移除StandardScaler，不保留“仅间隔不标准化”作为默认|[选择性标准化](</home/chris/ceph-tool/results/hp-selective-scaler-all5-20260925/REPORT.md>)|

EFDT曾在讨论中提出；本次未定位到能确认其完整测试完成的报告，标为**待确认**，不宣称已验证有效或无效。新增正收益硬门槛、叶重新激活、分裂前实时内存保护、训练类别计数衰减也没有作为本轮实现加入；不能将讨论当完成验证。

## 本轮中点接入验证

实验B指标与树增长证据：[中点五负载报告](</home/chris/ceph-tool/results/hp-midpoint-growth-20260926/REPORT.md>)。它是历史Trace离线结果，不是线上验收。

生产接入验证记录：[本轮报告](</home/chris/ceph-tool/results/hp-midpoint-dev-20260926/REPORT.md>)。包括完整HP/Trace回归、专项sanitizer、Ceph目标编译及当前dev对实验B的10份Trace逐字节对照；最终完成状态以该报告为准。

已知独立验证缺口：`test_sh/hp_osd_module_probe.cc`仍向`observe`传旧CEPH_OSD_OP枚举，而接口要求`HpAccessType`。这是既有测试适配问题，本轮没有修复或宣称该脚本通过；不等于Ceph目标编译失败。
