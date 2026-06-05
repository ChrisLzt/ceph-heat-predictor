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

## 2. 备份当前系统 ceph-osd

确认系统正在使用的路径：

```bash
which ceph-osd
```

备份旧版本：

```bash
sudo cp /usr/bin/ceph-osd /usr/bin/ceph-osd.bak.$(date +%Y%m%d_%H%M%S)
```

## 3. 替换 ceph-osd

安装新编译的 OSD 二进制：

```bash
sudo install -m 0755 /home/hust/ceph-heat-predictor/build/bin/ceph-osd /usr/bin/ceph-osd
```

## 4. 重启 OSD

```bash
sudo systemctl reset-failed ceph-osd@0
sudo systemctl restart ceph-osd@0
```

如果系统之前出现过 `start-limit-hit`，`reset-failed` 是必要的。

## 5. 检查 Ceph 状态

```bash
ceph -s
```

确认 `osd.0` 是 `up`：

```bash
ceph osd tree
```

检查冷热识别 perf counter：

```bash
sudo ceph daemon osd.0 perf dump hp_status
```

如果能看到 `hp_count`、`hp_hot_percent`、`hp_actual_hot_percent`、`hp_hot_precision`、`hp_hot_recall` 等字段，说明新的 OSD 已经启用。

## 6. 失败回滚

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

## 7. 不推荐操作

不要直接执行：

```bash
ninja install
```

原因：

- 会安装整套 Ceph 组件，不只是 `ceph-osd`。
- 可能触发 dashboard 前端 npm 构建。
- 可能覆盖 `/usr/bin`、`/usr/lib`、`/etc` 下的多个系统组件。
- 对当前冷热识别实验没有必要。

如果只改了 `src/blk/kernel/heat_predictor.h`、`KernelDevice.cc`、`BlockDevice.cc` 或相关头文件，手动替换 `build/bin/ceph-osd` 即可。
