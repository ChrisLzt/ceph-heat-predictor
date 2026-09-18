# Onode 在线缓存切换与 C4 整合

本次基于 `merge` 的 `a09205984bf`，整合两个已明确的代码来源：

- 缓存：`feat/onode-cache-online-switch` 的
  `3de6e677ae6d01e104af8188176bae8b706c4665`。
- 冷热识别：`dev` 的 `afd18c8e012e47b2b1aab7e0c817e3c725b8c5bf`。

## 模块边界

缓存保留指定提交的生产代码及九项回归测试。新增 Onode LRU/S3FIFO 在线切换，
保持分片地址、驻留对象、引用、年龄统计、累计命中计数，切换不改变 Buffer cache。
命令和统计口径见 [ONODE_CACHE_OPERATIONS.md](ONODE_CACHE_OPERATIONS.md)。

HP 同步七维 C4：原五维（包括旧热度）加 τ30/60秒指数慢历史，观测年龄补偿上限4。
保留10秒历史/标签、2秒短期、旧热度10秒降至10%、3000已发布训练样本预热、
固定多数类叶、2000样本/2秒快照、ADWIN关闭及分裂候选修复。
MGR 同步五行简要摘要和 `--detail` 完整生产统计。
算法定义和控制接口见 [CODEX_CEPH.md](CODEX_CEPH.md) 与
[MGR_HP_OPERATIONS.md](MGR_HP_OPERATIONS.md)。

这是同一 OSD/MGR 源码中的模块共存，不增加预测驱动缓存策略的自动反馈。
缓存选择仍由 `onode_cache policy` 控制，HP 仍只输出预测与统计。
HP 默认关闭，调用 enable 后按原约定完整 reset；缓存切换不调用 HP reset。

沿用 merge 的生产边界，不加入 dev-only Trace、实验回放或候选算法。
源 dev 的算法回归使用整合头文件独立编译执行。

## 构建与验证入口

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_TESTS=ON \
  -DWITH_RADOSGW=OFF -DWITH_MGR_DASHBOARD_FRONTEND=OFF \
  -DWITH_SYSTEM_BOOST=ON -DWITH_MANPAGE=OFF
ninja -C build -j32 ceph-osd ceph-mgr \
  unittest_bluestore_onode_cache unittest_bluestore_types
build/bin/unittest_bluestore_onode_cache --gtest_repeat=10
build/bin/unittest_bluestore_types \
  --gtest_filter=-bluestore_blob_t.csum_bench:sb_info_space_efficient_map_t.size
```

本机使用 GCC11、系统 Boost1.74，单独构建目录不覆盖线上 C4 产物。
具体执行结果以本轮实验目录中的日志和报告为准：
`/home/chris/ceph-tool/results/cache-c4-integration-20260918/`。

## 文档与验证边界

文档入口和迁移清单已同步当前整合实现，分别见
[README.md](README.md) 和 [迁移说明](CACHE_HEAT_PREDICTOR_PORTING_GUIDE.md)。
本次代码整合与文档整理不安装、不重启线上服务、不切换线上缓存。
独立构建和模块回归不等于完整线上联合负载验收；命中率、Accuracy、延迟和资源收益待实测。

## 本轮结果（2026-09-18）

- ceph-osd、ceph-mgr、两个缓存测试目标均成功编译链接。
- 九项 Onode 切换测试连续十轮全部通过（90次），BlueStore 类型测试27项通过。
- C4算法、EQ/并发、分裂候选、预热/快照/reset回归通过。
- 同一24,000样本流下，整合版与源dev的七维特征和模型概率逐项完全一致。
- 生产状态输出8项及聚合契约测试通过。
- 缓存源文件与指定提交逐字节一致。

源码整合时仅针对开发版Trace上下文调整HP状态结构和MGR输出；未修改缓存提交行为，
未重新设计C4公式或模型策略。验证报告保留在上述独立实验目录中。
