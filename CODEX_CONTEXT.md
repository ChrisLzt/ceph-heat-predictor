# CODEX Context: Ceph Heat Predictor

本文档记录当前冷热识别原型的实现状态、关键参数、测试脚本，以及下一阶段把识别位置从 BlockDevice 层迁移到 CephFS/RADOS 语义层的计划。目标是后续继续开发时能快速恢复上下文。

## 当前目标

当前项目是在 Ceph OSD 内加入在线冷热识别逻辑。现阶段只需要输出冷热判断结果，不需要实现“热数据写入 SSD、冷数据写入 HDD”的实际迁移或放置逻辑。

不过，下一阶段的 hook 位置需要为后续分层存储预留空间：识别结果应能关联到 CephFS/RADOS 数据对象，而不是只能关联到底层块设备偏移。

## 当前 Hook 位置

当前 hook 位于 BlockDevice/KernelDevice 层：

- 文件：`src/blk/kernel/KernelDevice.cc`
- 函数：`KernelDevice::_notify(uint64_t off, uint64_t len, int type)`
- 调用：`KernelDevice::hp.predict(index, type, len, off, 1)`

当前输入含义：

- `off`：底层块设备偏移
- `len`：IO 长度
- `type`：读写类型
- `index`：全局递增访问序号

这个位置适合做块设备层原型验证，但不适合作为 CephFS 冷热分层的最终 hook。原因是 BlockDevice 层已经丢失了 CephFS 文件、RADOS object、pool、namespace 等语义信息，后续无法可靠地把冷热结果映射回某个文件或对象。

## 当前冷热识别实现

核心文件：

- `src/blk/kernel/heat_predictor.h`
- `src/blk/kernel/include/ARFClassifier.h`
- `src/blk/kernel/include/HoeffdingTree*.h/.tpp`
- `src/blk/kernel/include/StandardScaler.h`
- `src/blk/kernel/include/utils.h`

当前特征数量：

```cpp
#define NUM_FEATURES 4
```

当前 bucket 粒度：

```cpp
static constexpr uint64_t HP_BUCKET_SHIFT = 20; // 1 MiB
static constexpr uint64_t HP_PAGE_SHIFT = 12;   // 4 KiB
```

每个访问样本转换成 4 个特征：

1. `bucket = address >> HP_BUCKET_SHIFT`
2. `operation`
3. `log2(size + 1)`
4. `page_offset = (address >> HP_PAGE_SHIFT) & HP_PAGE_OFFSET_MASK`

注意：这里的 `address` 当前是块设备偏移。迁移到 RADOS object 层后，应改成 object 内逻辑偏移或 object bucket，而不是 OSD 后端物理偏移。

## 驱逐队列和标签生成

冷热标签不是外部直接给定，而是由 `EvaluationQueue` 根据访问热度生成。

当前队列参数：

```cpp
EvaluationQueue(
    int max_size = 30000,
    double hot_threshold = 200,
    bool training = true,
    double alpha = -0.00008,
    int heating = 200,
    uint64_t waiting = 80000
)
```

含义：

- `max_size`：队列最多保存的活跃 bucket 数。
- `waiting`：一个 bucket 从首次进入队列开始，至少等待多少个访问时间戳后才允许出队并生成训练标签。
- `alpha`：热度随时间衰减的指数系数。
- `heating`：每次访问带来的热度增量。
- `hot_threshold`：当前热标签阈值。

热度更新形式：

```cpp
new_heat = exp(delta_ts * alpha) * old_heat + heating
```

出队逻辑：

1. `key_order` 保存 bucket 首次进入队列的顺序。
2. `item_map` 保存 bucket 的当前热度、首次访问时间、最近访问时间和样本信息。
3. 每次 enqueue 后检查 `key_order.front()`。
4. 如果队首 bucket 已等待至少 `waiting`，则出队并生成标签。
5. 如果队列超过 `max_size`，也会触发出队。

热标签判断：

```cpp
is_hot = val.heat > get_hot_threshold();
```

当前热阈值来自历史出队热度分布的分位数：

```cpp
static constexpr double HP_HOT_QUANTILE = 0.85;
```

即大致把历史出队热度排名靠前的 15% 视为热数据。

## 当前模型

模型由 `PipelineClassifier` 组合：

```cpp
PipelineClassifier(
    new StandardScaler<NUM_FEATURES>(),
    new ARFClassifier<...>(...)
)
```

当前 ARF 参数：

```cpp
ARFClassifier<NUM_FEATURES, 2,
    DetectorFactory<ADWIN<5>, 10>,
    DetectorFactory<ADWIN<5>, 1>>(
        6,       // n_models
        4,       // max_features
        591422,  // seed
        100,     // grace_period
        4,       // lambda_value
        0.001,   // delta
        0.05,    // tau
        0.99,    // max_share_to_split
        0.01     // min_branch_fraction
)
```

主要含义：

- `n_models = 6`：森林里 6 棵在线 Hoeffding tree。
- `max_features = 4`：每个随机叶节点候选特征数，目前等于全部特征数。
- `grace_period = 100`：叶节点累计一定权重后才尝试分裂。
- `lambda_value = 4`：在线 bagging 的 Poisson 采样强度。
- `delta = 0.001`：Hoeffding bound 参数，影响分裂激进程度。
- `tau = 0.05`：tie-breaking 阈值。
- `ADWIN<5>`：漂移检测窗口压缩参数。
- warning detector 使用 `DetectorFactory<ADWIN<5>, 10>`。
- drift detector 使用 `DetectorFactory<ADWIN<5>, 1>`。

训练权重：

```cpp
double weight = sample.label ? 1.5 : 1.0;
shadow_model->learn_one(to_feat(sample.item), sample.label, weight);
```

`ARFClassifier::learn_one()` 中已经把外部权重 `w` 乘到 Poisson 采样得到的 `k` 上：

```cpp
double sample_weight = w * k;
model->learn_one(x, y, sample_weight);
```

## 后台训练和模型交换

预测线程使用 active model，训练线程使用 shadow model。

关键参数：

```cpp
static constexpr int SWAP_INTERVAL = 2000;
static constexpr int BATCH_SIZE = 500;
```

流程：

1. 前台 `predict()` 进行预测，并把访问送入 `EvaluationQueue`。
2. bucket 出队后生成训练样本。
3. 训练样本进入后台训练队列。
4. 后台线程批量训练 shadow model。
5. 每训练 `SWAP_INTERVAL` 个样本后交换 active/shadow model。

当前 `hp_status` 输出指标包括：

- `hp_count`
- `hp_train_total`
- `hp_hot_percent`
- `hp_actual_hot_percent`
- `hp_accuracy`
- `hp_hot_precision`
- `hp_hot_recall`
- `hp_hot_threshold`
- `hp_train_queue_length`
- `hp_swap_count`
- `hp_predict_latency`

评估时需要注意：`accuracy` 容易被冷数据占比掩盖。更应该重点看：

- `hp_hot_precision`
- `hp_hot_recall`
- `hp_hot_percent` 是否长期接近 0
- `hp_actual_hot_percent`
- `accuracy - (1 - actual_hot_percent)` 是否明显大于 0

## 当前测试脚本

测试目录：

- `test_sh/config_skew_data/`：构造数据。
- `test_sh/config_skew_run/`：正式运行测试。
- `test_sh/config_skew/`：早期未拆分版本。

主要脚本：

- `test_sh/prepare_skew_data.sh`
- `test_sh/run_skew_tests_reset_hp.sh`
- `test_sh/run_skew_full_background.sh`
- `test_sh/mytest.sh`
- `test_sh/restart-osd.sh`

当前约定：

- 构造数据阶段可以使用 Vdbench `-c`。
- 正式测试阶段不要使用 `-c`，否则可能触发 cleanup，导致 `MISSING_PARENT` 或数据被清理。
- 正式测试前通过重启 OSD 清空当前内存中的 `hp_status` 和 predictor 状态。
- `run_skew_tests_reset_hp.sh` 会在每个测试前重启 `osd.0` 并等待 OSD up/in、PG clean。

合并后的数据构造配置：

- `1G文件数据构造.txt`
- `4K文件数据构造.txt`

这些用于替代多份重复的 1G/4K 数据构造脚本。

## 为什么 BlockDevice 层不适合 CephFS 分层

BlockDevice 层看到的是 OSD 本地存储后端的物理偏移。对于 CephFS 来说，用户访问路径大致是：

```text
CephFS client
  -> RADOS object op
  -> OSD PrimaryLogPG
  -> ObjectStore transaction
  -> BlockDevice
```

到了 BlockDevice 层以后，信息已经变成后端块设备地址。此时无法直接知道：

- 这个 IO 属于哪个 CephFS 文件。
- 这个 IO 属于哪个 RADOS object。
- 这个 object 属于哪个 pool。
- 这个 object 未来是否应该迁移到 SSD pool 或 HDD pool。

所以 BlockDevice 层只能证明“底层物理区域是否热”，不能为 CephFS 语义层分层提供足够信息。

## 下一阶段推荐 Hook

下一阶段建议把冷热识别 hook 放在 OSD 的 RADOS object op 层，优先考虑：

- `src/osd/PrimaryLogPG.cc`
- `PrimaryLogPG::do_op()`
- `PrimaryLogPG::execute_ctx()`
- `PrimaryLogPG::do_osd_ops()`
- `PrimaryLogPG::do_read()`
- 写操作相关分支，如 `CEPH_OSD_OP_WRITE`、`CEPH_OSD_OP_WRITEFULL`

推荐优先从 `PrimaryLogPG::do_osd_ops(OpContext *ctx, vector<OSDOp>& ops)` 附近入手，因为这里能看到：

- 当前 object context。
- op 类型。
- RADOS object 级别读写操作。
- object 内 offset/len。

在这个层面，冷热识别事件应改成 object 语义：

```cpp
struct HeatObjectEvent {
    uint64_t index;
    int op_type;
    uint64_t pool_id;
    hobject_t object;
    uint64_t object_offset;
    uint64_t length;
};
```

未来输出结果建议保留：

```cpp
struct HeatObjectResult {
    uint64_t pool_id;
    hobject_t object;
    uint64_t object_bucket;
    uint64_t object_offset;
    uint64_t length;
    int op_type;
    int pred_hot;
    double heat;
    double threshold;
};
```

这已经能满足“给出冷热判断结果”的需求，同时为未来 SSD/HDD 分层保留必要上下文。

## 下一阶段设计原则

1. 不在下一阶段实现 SSD/HDD 放置或迁移。
2. 不再用底层物理块地址作为主要 key。
3. 识别结果必须绑定到 RADOS object 或 object 内 bucket。
4. 保留当前 BlockDevice predictor，作为底层实验对照。
5. 新增 object-level predictor，不直接破坏当前 BlockDevice 原型。
6. `hp_status` 可以扩展为 object-level 版本，例如 `object_hp_status`。
7. 后续如果要真正分层，应该由 CephFS/MDS/mgr 或外部管理逻辑消费 object-level 冷热结果。

## 推荐实现路径

第一步：新增 object-level 事件结构和 predictor 包装。

- 可以复用当前 `HeatPredictor` 的在线学习框架。
- 特征从 block address 改成 object 语义。
- key 从 `address >> HP_BUCKET_SHIFT` 改成 `{pool_id, object_id, object_offset_bucket}`。

第二步：在 `PrimaryLogPG::do_osd_ops()` 中提取读写 op。

- 对 `CEPH_OSD_OP_READ` 记录读事件。
- 对 `CEPH_OSD_OP_WRITE`、`CEPH_OSD_OP_WRITEFULL`、`CEPH_OSD_OP_WRITESAME` 记录写事件。
- 对 metadata/class/omap 操作暂时跳过，避免污染数据热度。

第三步：添加 object-level perf counter。

建议输出：

- object 访问总数
- object 训练样本数
- 预测热比例
- 实际热标签比例
- precision
- recall
- threshold
- pending queue length
- model swap count
- predict latency

第四步：用现有 Vdbench CephFS 测试验证。

当前 Vdbench 通过 CephFS 写文件，但在 OSD op 层看到的是 RADOS object。测试仍然可以复用现有 `config_skew_run`，但分析对象应从物理 bucket 改成 RADOS object bucket。

## 当前注意事项

- 现在的 hook 在 BlockDevice 层，不能直接支持 CephFS 文件级或对象级分层。
- 当前模型的 `accuracy` 不应单独作为主要评价指标。
- 当前 `hp_hot_percent` 偏低时，通常意味着模型过于保守，应该结合 precision/recall 和 actual hot percent 判断。
- 当前 `StandardScaler` 是在线更新，训练曲线会受特征分布变化影响。
- 当前 `ARFClassifier` 的 `seed` 字段存在，但随机引擎初始化仍使用 `std::random_device()`，如果要严格复现实验，需要进一步固定随机种子。
- `test_sh/out/` 和 `test_sh/logs/` 是运行产物，不应提交到 GitHub。

