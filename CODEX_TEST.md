# CODEX Test Context

本文档维护冷热识别测试流程。Ceph 实现细节见 `CODEX_CEPH.md`，ICFS 迁移细节见 `CODEX_ICFS.md`，单节点部署命令见 `Ceph操作手册.md`。

## Ceph 单节点测试

当前主测试目录在仓库外：

```text
/home/chris/测试脚本/单节点测试/
```

关键脚本：

- `造数据.sh`：只构造数据。
- `后台运行标准IO测试.sh run`：前台串行运行测试。
- `后台运行标准IO测试.sh background`：后台运行测试。
- `后台运行标准IO测试.sh stop`：停止后台测试和 Vdbench slave。

当前有效配置只保留冷热 IO：

```text
冷热IO/造数据/1G文件冷热数据构造.txt
冷热IO/运行测试/1G文件4K块32线程冷热随机读写.txt
```

数据集：

- 仅保留 1G 文件数据集。
- hot：30GiB。
- cold：120GiB。
- 总量：150GiB。
- 运行测试：4KB 随机读写，`threads=32`，`rdpct=50`，hot/cold 访问 skew 为 `80/20`。

容量原则：容量不要按 8:2 生成；更合理的是 hot/cold 容量约 2:8，访问 skew 约 8:2，形成明显单位容量热度差异。

运行流程：

```bash
/home/chris/测试脚本/单节点测试/造数据.sh
/home/chris/测试脚本/单节点测试/后台运行标准IO测试.sh run
```

后台运行：

```bash
/home/chris/测试脚本/单节点测试/后台运行标准IO测试.sh background
```

停止：

```bash
/home/chris/测试脚本/单节点测试/后台运行标准IO测试.sh stop
```

脚本行为：

- 每个测试前执行 `ceph osd hp reset`。
- 轮询 `ceph osd hp status -f json`，确认计数归零后开始测试。
- 测试期间定期采集展开后的 `ceph osd hp status -f json-pretty`。

输出目录：

```text
/home/chris/测试脚本/单节点测试/logs/<RUN_ID>/
/home/chris/测试脚本/单节点测试/out/<RUN_ID>/
/home/chris/测试脚本/单节点测试/hp_status/<RUN_ID>/
```

## Ceph 仓库内旧测试

仓库内仍保留早期 skew 测试脚本：

```text
test_sh/config_skew_data/
test_sh/config_skew_run/
test_sh/prepare_skew_data.sh
test_sh/run_skew_tests_reset_hp.sh
test_sh/run_skew_full_background.sh
```

这些脚本可用于回归，但当前单节点主流程以 `/home/chris/测试脚本/单节点测试/` 为准。`test_sh/config_skew/` 是早期未拆分版本，尽量不要作为主流程继续扩展。

## 构建和安装

Ceph OSD/MGR：

```bash
cd /home/chris/ceph-heat-predictor/build
env CCACHE_TEMPDIR=/tmp ninja ceph-osd ceph-mgr -j"$(nproc)"
sudo install -o root -g root -m 0755 bin/ceph-osd /usr/bin/ceph-osd
sudo install -o root -g root -m 0755 bin/ceph-mgr /usr/bin/ceph-mgr
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo systemctl restart ceph-mgr@s52.service
```

常见构建依赖：

```bash
sudo apt install -y libfmt-dev libsqlite3-dev liblttng-ust-dev xfslibs-dev
```

MGR Python 依赖：

```bash
sudo apt install -y python3-prettytable python3-pecan
```

## object_hp_status 检查

单 OSD：

```bash
ceph daemon osd.0 perf dump object_hp_status
```

全局汇总：

```bash
ceph osd hp status
ceph osd hp status -f json-pretty
```

清空：

```bash
ceph osd hp reset
```

评估时重点看：

- `hp_true_positive_count`
- `hp_false_positive_count`
- `hp_true_negative_count`
- `hp_false_negative_count`
- `hp_hot_precision`
- `hp_hot_recall`
- `hp_eval_actual_hot_percent`
- `hp_dequeue_max_size_count`

不要只看 `hp_hot_accuracy`。冷样本占多数时，全预测冷也可能得到较高 accuracy。

## 队列覆盖检查

当前 Ceph 默认：

```text
HP_BUCKET_SHIFT = 12      # 4KB
EvaluationQueue max_size = 5000
```

150GiB 数据集约有：

```text
150 GiB / 4 KiB = 39,321,600 buckets
```

单 OSD 约 75GiB，即约 19,660,800 buckets。5000 队列无法覆盖整个数据集，这是预期结果。

运行时判断：

- `hp_dequeue_max_size_count` 增长：队列因容量上限出队，活跃 key 超过队列。
- 只有 `hp_dequeue_waiting_count` 增长：可能活跃工作集较小，队列未被打满。
- `hp_labeled_total` 增长：延迟标注正常发生。

## ICFS 多节点测试

多节点 ICFS/vdbench 测试目录：

```text
/home/hust/测试脚本/
```

常见结构：

- `config_data/`：构造数据。
- `config_run/`：正式测试。
- `run_prepare_data_background.sh`：后台构造数据。
- `run_vdbench_tests.sh`：运行正式测试。
- `logs/`、`out/`：运行产物。

当前已知客户端：

- `host1001`
- `host1003`
- `host1005`
- `host1007`

当前已知存储节点：

- `idfs01`
- `idfs02`
- `idfs03`

ICFS 的 `object_hp_status` 是每个 OSS daemon 的本地状态。跨 OSS 汇总需要外部脚本或 mgr/vas 聚合。

## 提交注意事项

- `logs/`、`out/`、`hp_status/` 是运行产物，不应提交。
- 大规模造数据前先确认 CephFS 或 ICFS 文件系统可用容量。
- Vdbench 多客户端测试需要保证客户端时间同步。
- 如果遇到 `ENOSPC`，先定位历史测试目录和实际挂载点，不要直接删除数据。
