# ceph-osd 编译与更新流程

本文档用于修改冷热识别代码后，只编译并更新 `ceph-osd`。不要为了更新冷热识别去跑完整 `ninja install`，否则会触发 dashboard/npm、测试组件和大量无关安装。

## 1. 编译 ceph-osd

进入项目构建目录：

```bash
cd /home/hust/ceph-heat-predictor/build
```

只编译 OSD：

```bash
ninja ceph-osd -j$(nproc)
```

确认新二进制存在：

```bash
ls -lh /home/hust/ceph-heat-predictor/build/bin/ceph-osd
```

## 2. 推荐安装方式

优先使用脚本同时安装 `ceph-osd` 和相关插件：

```bash
cd /home/hust/ceph-heat-predictor
./test_sh/update/install_ceph_osd_with_plugins.sh
```

脚本会：

- 备份当前 `/usr/bin/ceph-osd` 和将要覆盖的插件。
- 安装 `build/bin/ceph-osd` 到 `/usr/bin/ceph-osd`。
- 同步安装 `build/lib` 中的 erasure-code、compressor、crypto 插件到 `/usr/lib/ceph/`。
- 重启 `ceph-osd@0`。
- 输出 `systemctl status`、`ceph -s` 和 `object_hp_status` 检查结果。

常用环境变量：

```bash
OSD_ID=0 ./test_sh/update/install_ceph_osd_with_plugins.sh
BUILD_FIRST=1 ./test_sh/update/install_ceph_osd_with_plugins.sh
START_OSD=0 ./test_sh/update/install_ceph_osd_with_plugins.sh
```

不要只替换 `/usr/bin/ceph-osd`。Ceph OSD 启动时会校验插件版本，如果系统里的 `/usr/lib/ceph/erasure-code/`、`/usr/lib/ceph/compressor/` 或 `/usr/lib/ceph/crypto/` 仍是旧插件，可能出现：

```text
expected plugin ... version <new> but it claims to be <old> instead
```

## 3. 手工备份当前系统 ceph-osd

确认系统正在使用的路径：

```bash
which ceph-osd
```

备份旧版本：

```bash
sudo cp /usr/bin/ceph-osd /usr/bin/ceph-osd.bak.$(date +%Y%m%d_%H%M%S)
```

## 4. 手工替换 ceph-osd

安装新编译的 OSD 二进制：

```bash
sudo install -m 0755 /home/hust/ceph-heat-predictor/build/bin/ceph-osd /usr/bin/ceph-osd
```

如果选择手工替换，还必须同步安装插件：

```bash
sudo install -d -m 0755 /usr/lib/ceph/erasure-code /usr/lib/ceph/compressor /usr/lib/ceph/crypto
sudo cp -a /home/hust/ceph-heat-predictor/build/lib/libec_*.so* /usr/lib/ceph/erasure-code/
sudo cp -a /home/hust/ceph-heat-predictor/build/lib/libceph_snappy.so* /usr/lib/ceph/compressor/
sudo cp -a /home/hust/ceph-heat-predictor/build/lib/libceph_zlib.so* /usr/lib/ceph/compressor/
sudo cp -a /home/hust/ceph-heat-predictor/build/lib/libceph_zstd.so* /usr/lib/ceph/compressor/
sudo cp -a /home/hust/ceph-heat-predictor/build/lib/libceph_lz4.so* /usr/lib/ceph/compressor/
sudo cp -a /home/hust/ceph-heat-predictor/build/lib/libceph_crypto_*.so* /usr/lib/ceph/crypto/
```

## 5. 重启 OSD

```bash
sudo systemctl reset-failed ceph-osd@0
sudo systemctl restart ceph-osd@0
```

如果系统之前出现过 `start-limit-hit`，`reset-failed` 是必要的。

## 6. 检查 Ceph 状态

```bash
ceph -s
```

确认 `osd.0` 是 `up`：

```bash
ceph osd tree
```

检查冷热识别 perf counter：

```bash
sudo ceph daemon osd.0 perf dump object_hp_status
```

如果能看到 `hp_count`、`hp_hot_percent`、`hp_actual_hot_percent`、`hp_hot_precision`、`hp_hot_recall` 等字段，说明新的 OSD 已经启用。

## 7. 失败回滚

如果 OSD 启动失败，先看日志：

```bash
sudo journalctl -u ceph-osd@0 -n 120 --no-pager
```

恢复备份版本。把下面的时间戳替换为实际备份文件名：

```bash
sudo cp /usr/bin/ceph-osd.bak.YYYYMMDD_HHMMSS /usr/bin/ceph-osd
sudo chmod +x /usr/bin/ceph-osd
sudo systemctl reset-failed ceph-osd@0
sudo systemctl restart ceph-osd@0
ceph -s
```

## 8. 不推荐操作

不要直接执行：

```bash
ninja install
```

原因：

- 会安装整套 Ceph 组件，不只是 `ceph-osd`。
- 可能触发 dashboard 前端 npm 构建。
- 可能覆盖 `/usr/bin`、`/usr/lib`、`/etc` 下的多个系统组件。
- 对当前冷热识别实验没有必要。

如果只改了 `src/heatpredictor/heat_predictor.h`、`src/heatpredictor/include/`、`src/osd/PrimaryLogPG.cc` 或相关头文件，手动替换 `build/bin/ceph-osd` 即可。
