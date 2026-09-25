# 缓存迁入dev（2026-09-25）

统一开发测试分支为dev，工作目录`/home/chris/ceph-heat-predictor`。
本轮将merge `140169f115cb4c0bdcd533f1f1db03891fedbf83` 的缓存相关内容迁入dev `691603cbd1` 工作区。
本次按缓存文件与接口整合，不创建合并提交、不推送、不部署，不删除merge历史。

## 整合范围

- OnodeCache控制器、LRU/S3FIFO分片、预取和预算模块及BlueStore接入点。
- 缓存配置项、ObjectStore接口、OSD admin-socket状态/策略命令和CMake注册。
- Onode缓存回归测试、缓存操作手册和相关qa/experimental材料。
- OSD初始化保留dev的whoami参数，供Trace记录OSD身份使用。

HP核心、参数、观测hook、Trace采集/命令/状态、分析和回放脚本保持dev已提交版本。
用户明确放弃的未提交中点兜底候选、测试和审查材料先归档，再从工作区移除；不随本次整合发布。
未迁入与缓存无关的容器脚本、GitHub配置或HP算法差异。

## 默认状态与验证入口

缓存默认值沿用merge：预取默认关闭；未覆盖启动配置时Onode使用LRU。HP默认关闭。
缓存与HP保持独立控制。本次没有修改服务器运行配置。

现有build目录含root持有的产物，本轮配置、编译和缓存CTest沿用sudo；新建用户可写构建目录时可去掉sudo。

```bash
sudo -n cmake -S . -B build -DWITH_TESTS=ON
sudo -n ninja -C build -j12 ceph-osd ceph-mgr unittest_bluestore_onode_cache unittest_bluestore_types
sudo -n ctest --test-dir build -R '^unittest_bluestore_onode_cache$' --output-on-failure
build/bin/unittest_bluestore_types --gtest_filter=-bluestore_blob_t.csum_bench:sb_info_space_efficient_map_t.size
bash test_sh/test_hp_model_regressions.sh
bash test_sh/test_hp_osd_module.sh
bash test_sh/test_hp_storage_gate.sh
bash test_sh/test_hp_status_output.sh
python3 -m unittest test_sh.test_analyze_hp_ablation test_sh.test_analyze_hp_replay test_sh.test_hp_threshold_drift_matrix_analysis test_sh.test_hp_trace_analysis test_sh.test_project_hp_trace_v1_csv
```

本轮执行日志、缓存来源清单和废弃内容备份保存在：
`/home/chris/ceph-tool/results/cache-into-dev-20260925/`。
构建、32项缓存测试、27项BlueStore类型测试、HP模型/Trace/存储开关/状态输出及30项Python分析测试通过。旧OSD生命周期探针因仍使用操作码接口而编译失败，已确认是dev原有不兼容；本轮未修改该测试。完整记录见该目录REPORT.md，不宣称所有测试通过。
历史qa材料和缓存手册里的旧测试结论不代表本次工作区验证，不代表五负载重跑或线上验收。
