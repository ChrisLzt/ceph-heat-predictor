# Ceph Heat Predictor 计划更新

本文档只记录已经确定但尚未合并进 `CODEX_CEPH.md` 的更新计划。完成并验证后，应把最终状态合并到 `CODEX_CEPH.md`，再从本文档删除。

状态：本节已实现并同步到 `CODEX_CEPH.md`，暂保留作为本轮实现审阅记录；确认后可删除。

## 预测快照 snapshot / clone

目标：降低前台 `predict()` 延迟，避免前台预测和后台训练长期竞争同一个 `model_mutex`。

### 设计

- 后台持有 `train_model`，只由训练线程调用 `learn_one()`。
- 前台持有只读 `prediction_snapshot`，只用于 `predict_proba_one()`。
- 后台每训练 `MODEL_UPDATE_REPORT_INTERVAL` 个样本后，从 `train_model` 生成一个新的预测快照并原子式发布。
- 前台预测只短暂复制 `shared_ptr`，不等待后台训练。
- 快照允许最多一个发布周期的模型滞后；当前 `MODEL_UPDATE_REPORT_INTERVAL = 2000`，相对 `HP_EVALUATION_WINDOW = 10000` 可以接受。

### clone 范围

预测快照只复制前台预测需要的状态：

- `StandardScaler`：复制 `counts/means/m2s`。
- `PipelineClassifier`：复制 transformer 和 classifier。
- `ARFClassifier`：复制 active trees 和投票用 metrics；不复制 background tree、warning/drift detector、rng。
- `HoeffdingTreeClassifier`：深拷贝 `_root` 和预测所需树状态。
- `BranchOrLeaf` / `NumericBinaryBranch` / `LeafNaiveBayesAdaptive` / `RandomLeafNaiveBayesAdaptive`：递归深拷贝节点、stats、splitters 和叶子预测权重。
- `GaussianSplitter` 当前成员为值类型，可默认复制。

### 并发约束

- `train_model_mutex` 只保护后台训练模型。
- `prediction_snapshot_mutex` 只保护快照指针发布和前台复制 `shared_ptr`。
- 前台拿到局部 `shared_ptr` 后在锁外预测。
- 快照对象发布后不得再训练或修改。
- reset 需要同时重建 `train_model` 和 `prediction_snapshot`，并清空训练队列、EQ 和统计。

### 验收

- 独立探针验证：clone 后快照预测与原模型当时预测一致；继续训练原模型不会改变旧快照输出。
- `ceph-osd`、`ceph-mgr` 编译通过。
- install、ldconfig、重启 OSD/MGR 后 `ceph -s` 正常。
- 对比 `hp_predict_latency.avgtime_ns`，预期训练高峰时下降。
