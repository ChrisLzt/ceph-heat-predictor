# 缓存命中率分析与 CloudLab 验证

日期：2026-09-19。本文区分同学提供的历史数据与本次 CloudLab 新测量。

## 结论与考核口径

1. **本次 CloudLab 五项完整复测均达到 Onode 查询命中率严格 >96%**，最低为 AI 训练的 99.702009%；总体查询数加权为 **99.876629%**，五项等权平均为 99.863457%。
2. 主口径保持负责人已接受的 **Onode 查询命中率**，不需要混入 MDS 指标。它是元数据查询口径，不是应用数据读请求命中率。
3. 同一固定缓存预算、完整 658.17 GiB 数据集、3+7 分钟配置完成了一轮串行测试。原始材料的总体加权本已为 98.811629%，但 Baleen/WRF 单项未过 95%；本次五项单项均超过阈值。
4. **不可把差值全部归因于 S3FIFO**：LRU 本轮也均超过 99%；硬件、缓存初态/容量及客户端路径与原材料不同，尤其原材料是内核 CephFS、本次是 ceph-fuse。WRF 本轮 S3FIFO 甚至略低于本轮 LRU。
5. 这是有明确环境边界的单轮工程测试，不是 CCF 标准认证或同条件算法优劣证明。负载分布实际偏差、AI 训练一次约 12 秒采样缺口、单副本及混合服务版本均在下文披露。最终环境或客户端路径变化时应重新测量。

### 本次 CloudLab 新结果

运行编号：`fixed8g-ses-postprepare-001`。以下为名义 600 秒内、全部 OSD 已确认 S3FIFO 后的稳定区间。所有 case 都有 COMPLETE 标记，退出码为 0，测试前后文件 inode/size 清单一致。

| 负载 | LRU 查询命中率 | S3FIFO 查询命中率 | S3FIFO 命中数 | S3FIFO 未命中数 | 有效 S3FIFO 秒 |
|---|---:|---:|---:|---:|---:|
| Baleen | 99.899384% | **99.916884%** | 352,228 | 293 | 417.751 |
| GraphChi | 99.791021% | **99.913115%** | 3,265,850 | 2,840 | 417.651 |
| WRF | 99.863200% | **99.835487%** | 2,776,973 | 4,576 | 417.341 |
| AI 训练 | 99.206428% | **99.702009%** | 1,506,282 | 4,502 | 406.595 |
| AI 推理 | 99.561498% | **99.949791%** | 3,344,327 | 1,680 | 416.061 |

总体：`11,245,660 / (11,245,660 + 13,891) × 100% = 99.876629%`。这不是把五个百分比直接相加或取平均得到的请求比例。

### 独立诊断结果

| 负载 | extent-map 查询命中率 | 数据缓存字节命中率 | MDS 遍历命中率 | 冷热模块准确率 |
|---|---:|---:|---:|---:|
| Baleen | 99.166304% | 37.148916% | 100%（1,020/1,020） | 93.443626% |
| GraphChi | 99.260495% | 0.994467% | 100%（888/888） | 83.519796% |
| WRF | 98.600989% | 13.021184% | 100%（736/736） | 82.506356% |
| AI 训练 | 98.466855% | 3.133960% | N/A（0/0） | 94.661637% |
| AI 推理 | 99.234433% | 2.073053% | N/A（0/0） | 93.273708% |

冷热结果使用系统内部延迟标签及 TP/FP/TN/FN 口径，不是另建外部 ground truth 的独立验收。五项结束时各 OSD 标签总数与混淆矩阵一致，训练/待预测/待评价队列排空，预测错误、后台错误、评价丢样及训练丢样均为 0。完整 precision/recall/specificity 另见 `hot-cold-results.csv`；例如 Baleen 的 specificity 只有 26.68%，不能只引用其总体准确率来说明两类识别都同样好。本轮没有进行负载预测准确率或控制端完整可视化验收。

### 时序与采样质量

五项全部确认切换分别发生在 180.260、180.246、180.264、180.241、181.174 秒。不是把命令发送时间假定为所有 OSD 已完成切换。

GraphChi 完整 RD 日程为 609.988 秒，WRF 为 611.995 秒；按完整日程统计的 S3FIFO 命中率分别为 99.914207%、99.838399%，仍超过 96%。其他三项日程为 600 秒。

AI 训练有一个 11.989 秒采样间隔超过预设 10 秒上限，主结果剔除了该区间，有效覆盖为名义后 420 秒的 96.808%。缺口两端计数差仍为 44,097 次命中、142 次未命中（99.679016%），但未补入主结果。其余四项覆盖为 99.062%-99.465%；损失主要来自起止及切换边界，不是按命中率高低挑选窗口。所有原始采样均保留。

补充短窗口诊断：五项有效 S3FIFO 采样区间的最低命中率为 98.276776%，30 秒查询数加权分箱最低为 99.486551%，均出现在 AI 训练。该诊断仅针对已有有效样本，采样缺口仍是未观测区间，不构成“连续每个瞬间均通过”的保证，也不改变阶段累计主口径。逐项最小值见 `realtime-diagnostics.csv`。

Vdbench 五项均为只读配置，但 OSD 全局统计中 AI 训练阶段出现写操作计数，其来源未单独归因，不能把该计数称为应用写负载。图中的 IOPS 是 OSD 层操作数，不是 Vdbench 应用请求数。

### 原始材料复算

以下全部来自提供的材料包，**不是本次 CloudLab 新结果**。时间窗口沿用材料中的有效 S3FIFO 窗口，计数已逐窗口与原始采样核对。

| 负载 | Onode 查询命中率 | extent-map 查询命中率 | 数据缓存字节命中率 | Onode >95% | Onode >96% |
|---|---:|---:|---:|---|---|
| Baleen | 94.717284% | 90.312290% | 85.084726% | 否 | 否 |
| GraphChi | 95.816634% | 94.199844% | 11.019574% | 是 | 否 |
| WRF | 91.911814% | 90.331528% | 47.107617% | 否 | 否 |
| AI 训练 | 99.004400% | 92.394801% | 7.404538% | 是 | 是 |
| AI 推理 | 99.825167% | 97.268623% | 5.183254% | 是 | 是 |

总体 S3FIFO 命中数为 4,798,352，未命中数为 57,708：

`4,798,352 / (4,798,352 + 57,708) × 100% = 98.811629%`

AI 推理查询量较大，对总体加权结果影响也较大。如果选择总体作为验收口径，应事先确定负载集合、各项运行时间、访问分布、并发及速率，并完整保留逐项结果。不能测完以后只保留有利负载或有利窗口。

## 输入与完整性

- Ceph 指定提交：[fbfd7114508d14b7e582263bd8a4fbc3883fa39f](https://github.com/ChrisLzt/ceph-heat-predictor/commit/fbfd7114508d14b7e582263bd8a4fbc3883fa39f)。
- 负载指定提交：[5406ff849346908e7029fa6a883ff0ed97b9540a](https://github.com/ChrisLzt/ceph-test/commit/5406ff849346908e7029fa6a883ff0ed97b9540a)。
- 实际参数以附件中的冻结配置 `01-current-joint-test/configs/*/rendered` 为准。提供的最终源码状态文件及 working-tree diff 为空；包内某些说明提到的早期本地改动，不能据此认定最终状态仍未提交。
- `FILES.csv` 列出的 5,405 个文件、334,766,506 字节全部通过大小及 SHA-256 校验。
- 五项数据总计 **706,704,703,488 字节，658.170044 GiB，43,859 个数据文件**。
- Baleen 188.279297 GiB、GraphChi 128 GiB、WRF 112 GiB、AI 训练 114.890747 GiB、AI 推理 115 GiB。不缩减为每项 32 GiB。
- 原材料没有提供 MDS 原始计数，也没有提供逐对象 Onode 缺失原因轨迹，不能回算 MDS 命中率或断言所有 miss 都属于容量 miss。

## 原结果的解释边界

### 负载来源与改编

同学补充说明：大数据负载来自论文 trace 转换，图计算来自论文算法设计，高性能计算基于 WRF，AI 两项改编自 CCF AI 测试。冻结模型进一步限定了复现范围：

| 类别 | 可核查来源/模型 | 本次实际运行，不应混淆 |
|---|---|---|
| 大数据 | Baleen Region4 trace 的冻结统计 | Vdbench 600 秒原生频率快照；原始时序已移除，GET/PUT 均映射为 read，不是原 trace 逐请求重放 |
| 图计算 | GraphChi PSW 结构、web-Google 图统计 | 合成等字节片段及阶段性 Zipf 访问；不运行原生 GraphChi，不称为实测图计算 trace |
| 高性能计算 | WRF 输入、history、restart 访问结构 | 由 Vdbench 实现的研究负载，写转读；不运行 WRF 数值模拟本体 |
| AI 训练/推理 | 用户提供的 CCF AI/SES 1.2.0 模板 | 冻结布局与访问模型的研究改编；使用 SES 生命周期，但不是原封不动的 CCF 标准测试或认证 |

论文出处可对应为 [Baleen: ML Admission & Prefetching for Flash Caches，FAST 2024](https://www.usenix.org/conference/fast24/presentation/wong) 和 [GraphChi: Large-Scale Graph Computation on Just a PC，OSDI 2012](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/kyrola)。Baleen 冻结 manifest 指向 CMU Baleen24 Region4 trace，GraphChi manifest 指向该 OSDI 论文及 SNAP web-Google 数据集。论文来源可靠并不自动证明转换后的 Vdbench 负载保持了原应用全部语义；当前测试也不复现 Baleen 论文的 ML admission/prefetching 系统或其 Disk-head Time 主指标。

冻结配置中的 `xfersize` 小数出现在传输大小分布的百分比权重，而传输字节数仍然按 4 KiB 对齐。兼容修复针对概率采样精度，不是允许任意小数个字节。Baleen 的当前版本是静态频率快照，不能把它说成保留了原 trace 的热点迁移时序。

本次补充附件中的《AI场景测试指引_V1.1.0_20250616》封面标注“CCF 信息存储技术专委会”，《SES_User_Manual_V1.1.0-0616》第 18-19 页定义了 AI 训练/推理模型。训练带宽原模型读写比例为 70:30，推理原模型为 80:20；冻结模型的 `source_write_to_read=true` 与此对应。原手册还规定了按可用空间、存储内存及主机内存计算的预置数据量下限。本次严格保留同学冻结的 114.89/115 GiB 布局，而不是按新硬件重新生成标准认证规模，因此不声称满足原标准测试的全部条件。

**对命中率的影响不是中性的**：写转读改变了 Onode 查询路径和数据缓存行为；Zipf 热点集中度、热点迁移、文件拆分及缓存容量都会改变工作集和重复查询比例。可以统一使用同学提供的负载测三个项目指标，但应冻结同一版本并重新测量，不能沿用旧负载下的命中率。当前五项配置均为只读，不能凭这套负载演示“读写两类 IOPS 都随业务变化”；写 IOPS 变化需要另行设计并经负责人认可，不能悄悄混入本次复测。

原环境为单机、两个 OSD，共用一块 NVMe 的两个分区；本次 CloudLab 是三个物理节点、三个 SSD OSD，因此不是硬件相同的复现。

原材料的 `workload_output/.../config.html` 明确记录 `/mnt/cephfs type ceph`，即内核 CephFS 客户端；本次为 `fuse.ceph-fuse`。客户端路径并不相同。以本次 Baleen 稳定 S3FIFO 窗口为例，OSD 读操作数为 349,697、读返回字节为 44,668,641,280，平均每个后端读约 127,735 字节，而 Vdbench 整体平均应用 I/O 约 3.77 MB。这支持“大请求被拆成较小后端查询”的解释。同一 Onode 可因此在一次应用大 I/O 内被多次查询；跨环境的命中率变化不能全部归因于 S3FIFO 或扩大缓存。最终若采用内核 CephFS 客户端，应以该路径重新验证，不直接沿用 FUSE 路径的数值。

原环境每 OSD 的 `osd_memory_target=4 GiB` 且开启自动调节，这不是 4 GiB 独占 Onode 缓存。Onode、extent-map、RocksDB 等内存需求会共同影响缓存行为。WRF 采样中每 OSD 约有 1.76 万至 2.03 万个 resident Onode，但没有足够证据区分首次访问、容量淘汰和策略淘汰各占多少。

前 180 秒与后 420 秒处于不同时间、不同热点阶段，并继承之前的数据准备和测试缓存历史。因此不能把前后差值直接解释成纯 LRU/S3FIFO 因果比较。严格算法比较还需要相同内存、相同初态和同一完整轨迹的独立 LRU/S3FIFO 重放。

当前集成版 C4 冷热识别与 S3FIFO 同时运行，但不能据此称为“冷热预测直接驱动 S3FIFO 淘汰”。本次保留这种真实实现边界。前 3 分钟是 LRU、HP 关闭，不是关闭所有 Ceph 缓存。

GraphChi/WRF 多个 RD 之间存在调度空隙。报告同时统计名义 600 秒窗口和完整 RD 时间范围，不静默删除最后阶段。

### Vdbench 配比报表缺陷与真实分布

原材料与 CloudLab 的 Baleen 均出现 `Observed Workload skew` 警告。但 stock 二进制的 `SkewReport.reportFileEndOfRunSkew()` 用 `long` 累加各 FWD 的小数 IOPS，每次累加都会截断小数；随后又用未截断的单项 IOPS 除以该总值。因此 `skew.html` 的实际占比总和分别约为 **125.96% / 144.01%**，不能直接作为真实分布偏差。

本次没有在运行中修改报表或负载二进制，而是使用 `audit_vdbench.py` 从每个 FWD 的 histogram **整数操作次数**独立复算，并核对所有 FWD 总数与整体 histogram 完全一致：

| Baleen 数据来源 | 应用操作数，Vdbench histogram 窗口 | 最大单 FWD 配比偏差 | 总变差距离 | 零操作 FWD |
|---|---:|---:|---:|---:|
| 原材料 | 108,674 | 4.517439 个百分点 | 4.740169% | 3/255 |
| 本次 CloudLab | 18,110 | 3.795526 个百分点 | 7.790656% | 8/255 |

这些是 Vdbench 非 warmup histogram 窗口的应用操作次数，不等于 Onode 查询次数。大 I/O 会拆成多个后端请求，Onode 查询也不与应用 I/O 一一对应。计数复算不改变缓存命中率，只纠正配比报表的错误分母。

**验收边界**：完整执行冻结配置且测得命中率超过阈值，不自动等于访问分布符合某项标准。当前还未约定 FWD 分布偏差容许范围。若正式验收要求例如每组偏差小于 1 个百分点，需要在另一轮实验中调整并冻结发压速率/并发或调度实现后重新验证；不能删除偏差记录或将当前结果包装为标准认证通过。

## CloudLab 实验配置

| 项目 | 本次配置 |
|---|---|
| 节点 | hp117、hp118、hp081，Ubuntu 24.04，约 64 GiB RAM/节点 |
| 服务版本 | 三个 OSD 和 MGR 编译自指定 fbfd711 提交；MON/MDS/现有 CephFS 客户端保留原 lab 版本，单独披露，不声称所有进程均已升级 |
| 服务及构建位置 | 全部在 CloudLab；未启动本机 Docker |
| 容量 | OSD0 约 279 GiB 原 SSD 块设备；OSD1/2 各 320 GiB 已实际分配的 SSD 文件及 direct-I/O loop 块设备，总计约 919 GiB |
| 文件后端说明 | OSD1/2 不是独占裸 SSD；保留 ext4 文件后端开销，不作为原始块设备性能标定 |
| 数据持久化 | `/mnt/ceph-lab/cephfs/cache-study-20260919/`，实际 CephFS 挂载，不落根分区 |
| 副本 | 实验池单副本，保留 `POOL_NO_REDUNDANCY` 警告；不是生产可靠性配置 |
| 缓存预算 | 每 OSD 固定 BlueStore 8 GiB、metadata 70%、KV 20%、自动调节关闭；两种策略使用相同预算 |
| OSD 内存目标 | 每 OSD 16 GiB；不能当成 Onode 实际占用，另采集 mempool 与 resident Onode |
| 策略 | 原 buffer cache 保持 2q，仅 Onode 在线切换 LRU → S3FIFO |
| 测量 | 所有数据生成并核验完成后，五项依次运行；每项原始 read-only、direct-I/O、max-rate、600 秒配置 |
| 初态 | 数据准备后状态，保留缓存，不声称冷启动；准备可跨数据集并行，正式测量不并行 |
| 保留内容 | 原来的四个 1 GiB 测试文件、旧源码和旧构建均保留 |

### 运行时复现限制

首份附件缺少实际使用的 `vdbench-fractional-xfer-v1` 及 SES 1.2.0 运行目录。用户随后补充 CCF AI 测试附件，其中 SES **1.2.0 / build 1.2.0.25061615** 的 40 个锁定源码文件全部与负载仓库要求一致，source fingerprint 为 `de6aa8e4a71a226a0dc2651018b99709c20289053c28130e9fdc5f279538a595`。SES 依赖已在 CloudLab 独立 venv 中配置，不运行附件的全局安装或存储清理脚本。

修改版 Vdbench 运行目录仍未提供。普通 Vdbench 的 `FwgEntry.getXferSize()` 使用整数随机百分位和整数累计，会截断小数比例，因此不能直接用于原始小数传输分布。

CloudLab 准备了独立标识的 `vdbench-cloudlab-fractional-v2`：以固定提交的 Vdbench 二进制为底，仅替换该采样方法，保留原字段、方法签名、journal-recovery 分支及其他方法，采用 double 概率累计。其真实 `getXferSize()` 已通过 900 万次抽样分布检验及单值检查；其他方法字节码也通过比较。它不是同学缺失 jar 的逐字节副本，随机序列也不能保证相同。

本次 AI 两项已使用补齐的真实 SES 生命周期和研究报告入口，执行原冻结 Vdbench 配置。原始 `ses_adapter.py` 白名单不接受 `current`，但材料内成功结果又标记为 `current`，存在需要对方核对实际加载代码的复现差异。本次在独立 `ses_adapter_cloudlab.py` 中仅允许已验证的 `2026-09-17 / phase_zipf099` 别名，保持配置内容不变并记录补丁。不声称 SES 标准认证。数据生成使用 stock Vdbench 的整数大小参数，不受上述小数采样缺陷影响。

## 统计规则

主指标：`Σ有效窗口、OSD Δonode_hits / Σ有效窗口、OSD (Δonode_hits + Δonode_misses)`。

范围为三个 OSD 的全局 Onode 查询，包括各池及正常 Ceph 后台活动，不是逐应用请求去重后的命中率。测试期间只运行一项目标负载，但不能据此假定所有后台查询已经被剔除。

- 按查询量加权，不对节点百分比取平均。
- 计数器重置、OSD 实例变化、缺失节点、采样长缺口必须标记，不拼接成正常区间。
- 在线切换期间单独列为 transition，不混进纯 LRU 或纯 S3FIFO。
- 无查询样本显示 N/A，不显示 100%。严格 >95 与 ≥95 不混用。
- MDS：单独展示 `Δtraverse_hit / Δtraverse`，命名为路径遍历命中比例；这不是数据读取命中率。
- extent-map：单独展示 `Δonode_shard_hits / Δ(onode_shard_hits + onode_shard_misses)`。
- 数据缓存：单独展示命中字节数占比，不冒充请求数占比。
- OSD object-context 查询命中比例作为另一项诊断指标，不与 Onode 合并。
- 冷热识别独立记录 TP/FP/TN/FN、accuracy、precision、recall、丢样与错误；不与缓存命中率混算。

## 本次验证状态

已完成：原始材料校验和独立复算；指定提交远程编译；90 次 Onode 测试及 27 项 BlueStore 类型测试；三个 OSD/MGR 升级；SSD 容量扩展；固定缓存配置验证；小数采样兼容修复验证；三节点原始采样预检；统计、分布审计和结果导出脚本共 20 项测试；SES 源码锁定核验及隔离研究入口测试；五项 658.17 GiB 持久化数据集完整核验。

已完成整套五项测量及结果复算。测试后 3 个 OSD up/in、137 PG active+clean、MDS active，唯一健康警告是预先披露的 `POOL_NO_REDUNDANCY`。三个 OSD 均恢复 LRU，冷热模块关闭；完整数据集保留，未重格式化或删除。此前仅预检及等待中止的运行目录不作为测量结果。

负载分布由整数 histogram 操作数复算。各 case 的最大单 FWD 偏差为：Baleen 3.795526、GraphChi 2.086019、WRF 11.271878、AI 训练 0.242349、AI 推理 0.036590 个百分点。WRF 偏差较大，正式验收需事先确定是否接受 `fwdrate=max` 的实际分布，或另行固定速率/并发后重跑；本报告不宣称这项条件已经通过。

## 可复核文件

- `materials-audit.json`：原始附件校验与各项历史指标复算。
- `analyze_materials.py`：独立复算程序。
- `PROTOCOL.md`：新测量前固定的规则与限制。
- `configure_study.py`：共同缓存预算设置及运行时核查。
- `prepare_workloads.py`、`prepare_remaining.py`：完整数据生成与文件清单核验。
- `study_agent.py`、`run_cloudlab_study.py`：远程采样、3+7 分钟在线切换和逐项执行。
- `analyze_cloudlab.py`、`test_analyze_cloudlab.py`：窗口判定、各指标独立统计与单元测试。
- `cloudlab-runs/preflight-001/`：三节点预检证据，不是正式测量结果。
- `cloudlab-runs/fixed8g-ses-postprepare-001/onode-results.csv`：主指标，逐项计数与有效时间。
- 同目录 `cache-diagnostics.csv`、`hot-cold-results.csv`：各缓存层及冷热结果，分别输出。
- 同目录 `sampling-quality.csv`、`workload-distribution.csv`、`realtime-diagnostics.csv`：时间覆盖、实际负载分布及有效短窗口诊断。
- `SKEW_REPORT_BUG.md`、`SkewReport.javap.txt`：Vdbench 错误分母的二进制证据。
- `RUNBOOK.md`：环境位置及重复测试入口；重复运行使用新 run id，不并发启动两个控制器。
- 同一结果目录的 `cloudlab-timeseries.png`：30 秒查询数加权命中率及 OSD 读/写操作时序，采样缺口显式标记。
- `cache-study-results-20260919.tar.gz`：可交付结果包，内含逐文件 SHA-256 清单，不包含大数据集、完整源码/镜像或认证凭据。
