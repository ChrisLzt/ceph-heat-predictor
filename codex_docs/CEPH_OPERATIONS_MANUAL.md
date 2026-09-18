# Ceph 单节点操作手册

当前环境：

- hostname `s52`，MON IP `192.168.1.52`
- 数据盘 WD Green SN350 `/dev/nvme1n1`，序列号 `223447804020`
- 两个 450 GiB 分区对应 `osd.0/1`
- 单副本 `size=1/min_size=1`

两个 OSD 位于同一块 SSD，仅用于测试。不要操作系统盘 `/dev/nvme0n1`。

## 1. 环境变量

```bash
export HOST=s52
export MON_IP=192.168.1.52
export PUBLIC_NETWORK=192.168.1.0/24
export FSID=$(uuidgen)
export CEPH_REPO=/home/chris/ceph-heat-predictor
export OSD_DISK=/dev/nvme1n1
export OSD_DISK_SERIAL=223447804020
```

## 2. 构建、安装和重启

首次配置：

```bash
cd "$CEPH_REPO"
git submodule update --force --init --recursive --progress
./install-deps.sh
./do_cmake.sh -DWITH_RADOSGW=OFF -DWITH_TESTS=OFF \
  -DWITH_MGR_DASHBOARD_FRONTEND=OFF
mkdir -p src/pybind/mgr/dashboard/frontend/dist
```

每次代码更新后统一全量构建和安装，避免 OSD/MGR/插件版本不一致：

```bash
cd "$CEPH_REPO/build"
sudo env CCACHE_TEMPDIR=/tmp ninja -j64
sudo env DESTDIR=/ ninja install
sudo ldconfig
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo systemctl restart ceph-mgr@${HOST}
sudo ceph -s
```

修改公共库、MON 或 MDS 后再执行：

```bash
sudo systemctl restart ceph-mon@${HOST} ceph-mds@${HOST}
sudo ceph -s
```

### s52 的库搜索路径（2026-09-14 实测修复）

本机构建的库安装到 `/usr/lib` 和 `/usr/lib/ceph`。
`cmake/modules/Distutils.cmake` 仅在定义 DESTDIR 时为 Debian 添加
`--install-layout=deb`；因此使用 `sudo env DESTDIR=/ ninja install` 将 Python 绑定
安装到系统 `dist-packages`。未定义 DESTDIR 时会装到
`/usr/lib/python3.10/site-packages`，MGR 子解释器仍可能优先使用旧系统绑定。
系统另有多架构目录和 `/usr/local` 中的旧 Ceph
库/绑定。只更新二进制并运行 ldconfig，仍可能使 MGR 加载旧 librbd，报 fmt v7
未定义符号，触发 `MGR_MODULE_DEPENDENCY`。

本机已在 `/etc/systemd/system/ceph-mgr@s52.service.d/zz-current-build.conf` 配置：

```ini
[Service]
Environment=LD_LIBRARY_PATH=/usr/lib:/usr/lib/ceph
Environment=PYTHONPATH=/usr/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages
```

该配置文件名保证晚于原 `pythonpath.conf` 加载；保留原文件。
本次同时完成 Debian 布局安装，系统 rbd 绑定与 site-packages 中当前产物 SHA-256
相同；PYTHONPATH 本身不能替代这一安装步骤。此 drop-in 在本机安装布局下优先选择当前构建的库和绑定。
安装前缀或 Python 版本改变后必须重新核对路径，不能直接套用。变更 drop-in 后执行
`systemctl daemon-reload` 和 MGR 重启；检查 `ceph health detail`，并检查 MGR
`/proc/<pid>/maps` 中实际加载路径。不要用关闭模块或 mute 告警代替依赖修复。

全量版本更新还需重启 MON/MDS，等待全部服务恢复后用 `ceph versions` 确认五个
守护进程版本一致、145 个 PG 为 active+clean。本机当前允许保留
`POOL_NO_REDUNDANCY`。实验入口另比较构建与已安装二进制的 GNU build ID，并记录
脏工作树和 SHA-256；版本里的 Git SHA 不包含未提交修改。

## 3. 初始化 MON/MGR

```bash
sudo install -d -o ceph -g ceph -m 0755 \
  /etc/ceph /run/ceph /var/run/ceph /var/log/ceph \
  /var/lib/ceph /var/lib/ceph/osd /var/lib/ceph/bootstrap-osd
sudo tee /etc/ceph/ceph.conf >/dev/null <<EOF
[global]
fsid = ${FSID}
mon_initial_members = ${HOST}
mon_host = ${MON_IP}
public_network = ${PUBLIC_NETWORK}
auth_cluster_required = none
auth_service_required = none
auth_client_required = none
osd pool default size = 1
osd pool default min_size = 1
osd_crush_chooseleaf_type = 0
mon_allow_pool_size_one = true
mon_allow_pool_delete = true
EOF
sudo ceph-authtool --create-keyring /tmp/ceph.mon.keyring \
  --gen-key -n mon. --cap mon 'allow *'
sudo ceph-authtool --create-keyring /etc/ceph/ceph.client.admin.keyring \
  --gen-key -n client.admin --cap mon 'allow *' --cap osd 'allow *' \
  --cap mds 'allow *' --cap mgr 'allow *'
sudo ceph-authtool --create-keyring /var/lib/ceph/bootstrap-osd/ceph.keyring \
  --gen-key -n client.bootstrap-osd --cap mon 'profile bootstrap-osd' \
  --cap mgr 'allow r'
sudo ceph-authtool /tmp/ceph.mon.keyring \
  --import-keyring /etc/ceph/ceph.client.admin.keyring
sudo ceph-authtool /tmp/ceph.mon.keyring \
  --import-keyring /var/lib/ceph/bootstrap-osd/ceph.keyring
sudo chown ceph:ceph /tmp/ceph.mon.keyring \
  /var/lib/ceph/bootstrap-osd/ceph.keyring
monmaptool --create --add "$HOST" "$MON_IP" --fsid "$FSID" /tmp/monmap
sudo install -d -o ceph -g ceph -m 0755 /var/lib/ceph/mon/ceph-${HOST}
sudo -u ceph ceph-mon --mkfs -i "$HOST" \
  --monmap /tmp/monmap --keyring /tmp/ceph.mon.keyring
sudo systemctl start ceph-mon@${HOST}
sudo install -d -o ceph -g ceph -m 0755 /var/lib/ceph/mgr/ceph-${HOST}
sudo ceph auth get-or-create mgr.${HOST} mon 'allow profile mgr' \
  osd 'allow *' mds 'allow *' \
  | sudo tee /var/lib/ceph/mgr/ceph-${HOST}/keyring >/dev/null
sudo chown -R ceph:ceph /var/lib/ceph/mgr/ceph-${HOST}
sudo systemctl start ceph-mgr@${HOST}
sudo ceph -s
```

## 4. 创建两个 OSD

以下命令会清空数据盘。先核对型号和序列号：

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL
test "$(cat /sys/block/$(basename "$OSD_DISK")/device/serial)" \
  = "$OSD_DISK_SERIAL"
sudo ceph-volume lvm zap "$OSD_DISK" --destroy
sudo wipefs -a "$OSD_DISK"
sudo sgdisk -o \
  -n 1:1MiB:+450GiB -t 1:8e00 -c 1:ceph-osd-0 \
  -n 2:0:+450GiB -t 2:8e00 -c 2:ceph-osd-1 \
  "$OSD_DISK"
sudo partprobe "$OSD_DISK"
sudo udevadm settle
sudo ceph-volume lvm create --data "${OSD_DISK}p1"
sudo ceph-volume lvm create --data "${OSD_DISK}p2"
sudo ceph osd tree
```

## 5. CephFS

```bash
sudo install -d -o ceph -g ceph -m 0755 /var/lib/ceph/mds/ceph-${HOST}
sudo ceph auth get-or-create mds.${HOST} mon 'profile mds' \
  mgr 'profile mds' mds 'allow *' osd 'allow *' \
  | sudo tee /var/lib/ceph/mds/ceph-${HOST}/keyring >/dev/null
sudo chown -R ceph:ceph /var/lib/ceph/mds/ceph-${HOST}
sudo systemctl start ceph-mds@${HOST}
sudo ceph osd pool create cephfs_meta 16 16
sudo ceph osd pool create cephfs_data 128 128
for pool in cephfs_meta cephfs_data; do
  sudo ceph osd pool set "$pool" size 1 --yes-i-really-mean-it
  sudo ceph osd pool set "$pool" min_size 1
  sudo ceph osd pool set "$pool" pg_autoscale_mode off
done
sudo ceph fs new myfs cephfs_meta cephfs_data
sudo ceph osd pool set .mgr pg_autoscale_mode off
sudo ceph osd pool set .mgr pg_num 1
watch -n 2 'ceph osd pool get .mgr pg_num; ceph osd pool get .mgr pgp_num; ceph -s'
```

已有 CephFS 时只把 `cephfs_data/meta` 的 `pg_num/pgp_num` 调整为 `128/16` 并关闭
autoscale。PG merge 完成且所有 PG 恢复 `active+clean` 后再测试。

挂载：

```bash
sudo grep "key =" /etc/ceph/ceph.client.admin.keyring | awk '{print $3}' \
  | sudo tee /etc/ceph/admin.secret >/dev/null
sudo chmod 600 /etc/ceph/admin.secret
sudo install -d -m 0755 /mnt/cephfs
sudo mount -t ceph ${MON_IP}:6789:/ /mnt/cephfs \
  -o rw,name=admin,secretfile=/etc/ceph/admin.secret
sudo chown lzt:lzt /mnt/cephfs
sudo install -d -o "$USER" -g "$USER" -m 0755 /mnt/cephfs/vdbench
```

## 6. 五负载准备与观测

当前五用例位于 `/home/chris/ceph-test/SINGLE_workload/`，使用 SINGLE v2；旧 v1
入口已停用。离线检查：

```bash
cd /home/chris/ceph-test
./SINGLE_workload/validate_all.sh
```

获得本轮造数授权后，关闭预测器并使用独立准备入口；新目录和容量检查通过才写入：

```bash
sudo ceph osd hp disable -f json-pretty
python3 -m workload_common.single_v2 preflight --case all
python3 -m workload_common.single_v2 prepare --case all --execute \
  --results /home/chris/ceph-tool/results/single-v2-prepare-唯一批次名
python3 -m workload_common.single_v2 verify --case all
```

该入口不会自动删除旧数据。容量、模型、READY 验证边界以
`/home/chris/ceph-test/SINGLE_workload/README.md` 为准；准备成功只证明文件布局、
数量、大小和已分配空间符合设计，不是测量结果。

实验矩阵位于 `/home/chris/ceph-tool/heat_predictor/run_hp_matrix.sh`，默认只预览；
后续获准测量时才加 `--execute`。AI 测量需要锁定的 SES 源码和隔离 Python 环境。
当前入口协议见 `/home/chris/ceph-tool/heat_predictor/EXPERIMENT_PROTOCOL.md`。

每轮开关、reset、状态采集和归零判据见
[MGR 操作说明](MGR_HP_OPERATIONS.md)。OSD `object_hp status` 是实时状态，
Perf/MGR 汇总可能短暂滞后。采集完整 JSON 必须使用 `hp status --detail -f json`。
