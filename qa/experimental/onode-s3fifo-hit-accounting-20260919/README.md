# S3FIFO 实现修复与对照验证

## 状态

代码回归及前三项真实负载的新旧版本对照已完成。**没有达到全部 case 严格大于
95% 的目标，不能标为验收通过或达标版。** Baleen 两版均超过 96%，GraphChi 和
WRF 两版均低于 95%。WRF 修复版还有一次采样缺口。

实际实现提交为
[`da10bdb168750e4428b0b98790c88b5d7a7f34f6`](https://github.com/ChrisLzt/ceph-heat-predictor/commit/da10bdb168750e4428b0b98790c88b5d7a7f34f6)，
只包含三个 C++/测试文件，可以单独 cherry-pick，不要求引入本报告或实验脚本。

## 实际代码改动

基于集成源码 `fbfd7114508d14b7e582263bd8a4fbc3883fa39f`，修改
`src/os/bluestore/BlueStore.cc`、`BlueStore.h` 及对应 C++ 测试。

1. 已进入淘汰队列的 Onode 在每次成功 lookup 时更新 S3FIFO 频次，仍饱和于 3。
   旧实现将频次更新放在最后一次 unpin 上，重叠查询会合并为一次复用，普通内部
   引用释放也可能被当作访问。LRU 的 recency 更新、未入队对象的准入时机不变。
2. ghost 队列缩容时保留较新的淘汰记录，丢弃最旧记录。
   原 `set_capacity` 会丢弃队尾的新记录，改为 `rset_capacity`。

没有改变 Onode hits/misses 的计数位置、分母、考核阈值、存储格式或 HP 算法。
这是缓存策略正确性修复，不意味着所有负载的命中率必然提高。

## 回归证据

- 旧实现：原有 9 项测试通过，新增 3 项测试全部失败。
- 修复后：12 项 Onode 测试通过，完整测试重复 50 轮通过（600 项次）。
- 远端 `ceph-osd` 目标编译通过；27 项 BlueStore 类型测试通过。
  按既有测试方式排除 `sb_info_space_efficient_map_t.size` 和
  `bluestore_blob_t.csum_bench`，不是宣称整个 Ceph 测试集通过。
- 本地统计分析器的 25 项离线测试通过，包含新增的 5 项子集/全量完成标记测试。
  本机未运行 Docker、Ceph 或负载。

新增测试直接调用 BlueStore 实现，不是用另写的缓存模型代替生产代码：

- `OverlappingHitsPromoteReusedOnode`：旧版丢失并发复用，热点对象被淘汰；修复后保留。
- `ReferenceReleaseIsNotAnotherHit`：内部引用释放不能增加频次，且频次上限仍为 3。
- `GhostShrinkKeepsMostRecentEvictions`：缩容后最近淘汰对象仍可通过 ghost 准入 main。

## 二进制身份

两个二进制的版本字符串均包含原基线 SHA，因此必须用内容哈希区分。

| 二进制 | SHA-256 |
|---|---|
| baseline | `fab74239f8e461a1ef86156545550f0410aad6a1d58285b82bd49e50f8403b5d` |
| fixed | `dc39db8fbf8879cccce2e48d85e8f2757d1745d7dedba1776bea219c245a15b4` |

## 负载对照设计

- 内核 CephFS 客户端，非 ceph-fuse。
- 每 OSD `osd_memory_target=4294967296`、`bluestore_cache_autotune=true`、
  `bluestore_cache_size=0`、meta/KV 比例各 0.45、cache_min=128 MiB。
- 不以此前固定 8 GiB 缓存、FUSE 客户端下的 >99% 结果作为本修复的性能证据。
- 数据不重建、不缩到 32 GiB；复用已准备的完整持久化数据和原有工作负载配置。
- 每轮按原来的 Baleen、GraphChi、WRF 顺序测试，每 case 600 秒；180 秒切换
  LRU→S3FIFO 并启用 HP。保留 GraphChi，以保持 WRF 的前置工作负载历史。
- 两个版本各自开始前重启 OSD，并重新挂载内核客户端，控制初始缓存状态。
  case 内不重启、不清缓存。第二个 case 保留第一个 case 的服务器缓存历史。
- 测试前必须所有 PG `active+clean`。对照期间暂停自动均衡，结束后恢复原状态。
- 按预定的相邻计数增量计算，切换窗口、采样缺口和无效窗口单列，不择优选时段。

## 外推限制

这里只是在 CloudLab 上验证代码，不是同学原服务器实测：此处三个 OSD，原材料
中两个 OSD；硬件、OSD 总内存预算、内核版本、初始缓存和工作负载运行时均有差异。
现有数据此前通过 FUSE 生成，本次只切换访问客户端，没有重建数据的物理布局。
因此生成阶段的请求拆分等影响不能仅靠切换挂载方式消除。两版使用相同的这份数据。
每轮清空的是 OSD 进程内缓存并重新挂载客户端，没有清空宿主机全部内核页缓存。
这里覆盖前三项 workload 的成对诊断，不是原来的五项串行全量验收。
用于小数传输尺寸的 Vdbench 是已记录的独立兼容实现，不是同学未提供的原始 jar。

即使这里超过 95% 或 96%，也不能据此保证原服务器达标；若旧版同样超过阈值，
更不能将达标归因于本修复。需要保持同学原服务器配置开展独立验证。

## 结果

统一采用稳定 S3FIFO 窗口的 `sum(delta_hits) / sum(delta_hits + delta_misses)`。
跨 OSD 按查询数加权，不合并不同 workload 来掩盖单项不达标，不混入 MDS 或 buffer
命中率。下表为实际采样值，不是置信区间或性能提升承诺。

| 负载 | 旧版 S3FIFO | 修复版 S3FIFO | 差值（百分点） | 修复版 >95% / >96% |
|---|---:|---:|---:|---|
| Baleen | 96.009602% | 96.086499% | +0.076898 | 是 / 是 |
| GraphChi | 91.869869% | 91.817393% | -0.052476 | 否 / 否 |
| WRF | 89.329695% | 89.963091% | +0.633396 | 否 / 否 |

| 负载 | 旧版 hits / misses | 修复版 hits / misses | 旧版 / 修复版稳定 S3FIFO 覆盖秒 |
|---|---:|---:|---:|
| Baleen | 251982 / 10473 | 250338 / 10196 | 417.258 / 413.740 |
| GraphChi | 150741 / 13340 | 149296 / 13305 | 417.440 / 416.360 |
| WRF | 161400 / 19279 | 148440 / 16561 | 417.488 / 385.559 |

LRU 阶段也存在轮间波动：Baleen 93.404852% / 93.007169%，GraphChi
83.142679% / 83.102212%，WRF 91.999732% / 91.570444%（旧版 / 修复版）。
因此单次配对的微小变化不能归因为代码改善。WRF 缺口进一步限制了比较。

### 质量与覆盖范围

- 六次负载进程均正常退出（返回码 0），两个预定三项序列均完成。
- 两版有效缓存配置快照一致，三台 OSD 的执行文件哈希逐一核对。
- 两版三项数据的前后清单身份一致：Baleen 29819 文件、202163355648 字节，
  GraphChi 2048 文件、137438953472 字节，WRF 1792 文件、120259084288 字节。
  清单校验基于文件名/stat/分配状态，没有为此完整读取并哈希文件内容。
- 固定在 180 秒请求切换；稳定阶段只使用实际确认后且未跨策略变化的相邻计数。
  按事先规则排除名义 600 秒窗口以外的数据，不延长窗口凑指标。
- 修复版 WRF 的序号 187→188 存在约 31.3 秒的采样间隔，约位于第 552→583 秒。
  超过既有 10 秒上限，未计入上表；没有填补、插值或将缺失窗口按 100% 计算。
  这是实时遥测覆盖不足，不能将该项视为无缺口的 10 分钟验收。
- 其余五次未发现窗口内采样缺口或 OSD 实例变化。所有阶段之外的边界窗口另列。
- Vdbench 各阶段整数直方图操作数与其 summary 对账一致。实际 FWD 比例并非
  配置比例的精确复现：GraphChi 各阶段总变差约 17.88% 至 21.03%，WRF 约
  4.17% 至 8.71%，Baleen 约 1.41% 至 1.94%；Baleen 两版各有两个 FWD 未产生操作。
  完整偏差记录见 `s3fifo-pair-distribution-summary.json`。负载配置相同不等于请求
  序列完全相同，不能将这次结果外推为原服务器的算法增益。
- 直方图为 Vdbench 的统计阶段（不含其 warmup），与 Onode 查询窗口分母不同，
  不用直方图操作数替代 Onode 查询数。
- `comparison.json`/`comparison.csv` 保存查询数、覆盖秒、驻留条目和抽样内存。
  抽样 mempool 不是进程 RSS 或整机总内存，不能据此宣称低于某个系统内存上限。

本次足以证明两处实现行为修复及测试覆盖，不足以证明所有负载达到 95%。还需在
同学原服务器上保留原内核客户端、实际缓存预算、完整负载和相同初始条件复测，
对未命中路径进一步诊断。当前聚合计数不能区分首次访问与容量/替换未命中。

`kernel4g-baseline-001` 在 Baleen 开始后主动中止，以补入 WRF 前面的 GraphChi。
该轮只作为中止记录保留，不参与新旧版本比较。正式配对使用后缀 `002`。

## 在原服务器应用代码

这部分不依赖 CloudLab 的主机名、OSD 编号、容器名或挂载目录。先在同学实际
Ceph 源码的独立验证分支应用实现提交（以 `fbfd711` 集成版为已验证基线）：

```bash
git fetch origin fix/onode-s3fifo-hit-accounting
git cherry-pick -x da10bdb168750e4428b0b98790c88b5d7a7f34f6
cmake --build build --target ceph-osd unittest_bluestore_onode_cache unittest_bluestore_types -j8
build/bin/unittest_bluestore_onode_cache
build/bin/unittest_bluestore_onode_cache --gtest_repeat=50 --gtest_break_on_failure
build/bin/unittest_bluestore_types --gtest_filter=-sb_info_space_efficient_map_t.size:bluestore_blob_t.csum_bench
```

`build` 和并行度应替换为实际构建目录与可用资源；依赖、工具链和配置沿用其
现有 Ceph 构建，不能直接照搬本实验二进制到不同系统。若现有源码已经改变，
先审查冲突及合并内容，不覆盖其他人的修改。

按原服务器既有部署流程备份原二进制并升级实际 OSD 清单，记录每个运行文件
SHA-256；首次升级需要重启 OSD。确认集群恢复正常后再测试，不初始化磁盘、
不重建池和持久化负载。单副本环境没有滚动升级的冗余保护，必须安排维护窗口。
升级后 LRU/S3FIFO 仍通过原 `onode_cache policy` 接口在线切换，控制端及命中率
公式不需要变更；HP 开关与准确率仍独立处理。

正式复测必须保存实际生效配置、客户端/内核版本、精确 workload/runtime 身份、
切换应答、完整原始计数及逐 case 结果。不要把本报告的三项选择当作五项全量完成。

## 证据与环境收尾

源代码、紧凑结果和回归 XML 随分支提交。大体积原始计数与 HTML 另行打包，
交付清单 `DELIVERY-QA.json` 记录每个文件及包的 SHA-256；只有紧凑 JSON 不能
独立复算所有实验结论。历史 CloudLab 控制器仍是实验专用，不是原服务器部署器。

最终核对三个 OSD 均 up/in、137 个 PG 均 `active+clean`，balancer 已恢复开启，
`noout` 已解除。集群仍有原有单副本 `POOL_NO_REDUNDANCY` 警告，不能称为
`HEALTH_OK` 或冗余验证。状态保存在 `final-state.json`。
两个序列结束时均将三个 OSD 恢复为 LRU、关闭 HP，保留 fixed 执行文件和全部数据。
本报告不包含 AI 训练/推理本轮复测，也不包含控制端三项功能联合可视化验收。
