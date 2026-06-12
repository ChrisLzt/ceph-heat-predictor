# CODEX Test Context

本文档集中维护冷热识别相关测试流程。Ceph 实现细节见 `CODEX_CEPH.md`，ICFS 迁移和实现细节见 `CODEX_ICFS.md`。

## Ceph 单节点测试

测试目录位于本仓库：

- `test_sh/config_skew_data/`：构造数据。
- `test_sh/config_skew_run/`：正式运行测试。
- `test_sh/config_skew/`：早期未拆分版本，尽量不要继续作为主流程使用。

主要脚本：

- `test_sh/prepare_skew_data.sh`：只构造数据，默认 `HP_SAMPLE=0`。
- `test_sh/run_skew_tests_reset_hp.sh`：正式测试，每个测试前重启 OSD 清空 predictor 状态。
- `test_sh/run_skew_full_background.sh`：后台一键运行数据构造和正式测试。
- `test_sh/mytest.sh`：单轮 vdbench 执行和 `object_hp_status` 采样封装。
- `test_sh/restart-osd.sh`：简单重启 `osd.0`。

运行前检查：

```bash
ceph -s
ceph pg stat
ceph daemon osd.0 perf schema object_hp_status
ceph daemon osd.0 perf dump object_hp_status
```

编译并安装 OSD：

```bash
cd ~/ceph-heat-predictor
ninja -C build ceph-osd
sudo install -m 0755 build/bin/ceph-osd /usr/bin/ceph-osd
sudo systemctl restart ceph-osd@0
```

只构造数据：

```bash
cd ~/ceph-heat-predictor
./test_sh/prepare_skew_data.sh
```

运行全部正式测试：

```bash
cd ~/ceph-heat-predictor
./test_sh/run_skew_tests_reset_hp.sh
```

后台一键运行：

```bash
cd ~/ceph-heat-predictor
nohup ./test_sh/run_skew_full_background.sh > test_sh/logs/background/$(date +%Y%m%d_%H%M%S)_skew_full.nohup.log 2>&1 &
```

日志位置：

```text
test_sh/logs/<RUN_ID>/<workload>.vdbench.log
test_sh/logs/<RUN_ID>/<workload>.hp_status.log
test_sh/out/<RUN_ID>/<workload>/
```

## ICFS 多节点测试

多节点 ICFS/vdbench 测试脚本位于：

```text
/home/hust/测试脚本/
```

目录结构：

- `config_data/`：构造数据。
- `config_run/`：正式运行测试。
- `run_prepare_data_background.sh`：后台一键构造数据。
- `run_vdbench_tests.sh`：运行正式测试。
- `logs/`、`out/`：运行产物。

当前客户端：

- `host1001`
- `host1003`
- `host1005`
- `host1007`

当前存储节点：

- `idfs01`
- `idfs02`
- `idfs03`

只构造数据：

```bash
cd /home/hust/测试脚本
./run_prepare_data_background.sh
```

正式测试：

```bash
cd /home/hust/测试脚本
./run_vdbench_tests.sh
```

## 测试注意事项

- 大规模构造数据时避免过高文件数制造元数据压力。
- 多客户端 vdbench 需要保持客户端时间同步，否则可能触发 heartbeat 问题。
- FSD 共享 anchor 的多客户端 format/cleanup 容易竞争；构造流程默认不使用全局 `-c` 清理参数。
- 如果图计算或大对象构造报 `ENOSPC`，先检查 `/mnt/ikcdir` 可用空间和历史测试目录，不要直接删除数据，除非用户明确确认。
- Ceph 仓库内的 `test_sh/out/` 和 `test_sh/logs/` 是运行产物，不应提交到 GitHub。
- ICFS 测试目录中的 `logs/` 和 `out/` 也是运行产物，不应作为源码变更提交。

## object_hp_status 检查

Ceph 单节点示例：

```bash
ceph daemon osd.0 perf dump object_hp_status
```

ICFS 多 OSS 环境中，`object_hp_status` 当前是每个 OSS daemon 的本地状态。跨 OSS 汇总需要由 mgr/vas 或外部脚本轮询所有 OSS 后再聚合。

评估时不要只看 `accuracy`。冷样本占多数时，全预测冷也会得到较高 accuracy。重点看：

- `hp_hot_precision`
- `hp_hot_recall`
- `hp_hot_percent` 是否长期接近 0
- `hp_eval_hot_percent`
- `hp_actual_hot_percent`
- `accuracy - (1 - actual_hot_percent)` 是否明显大于 0
