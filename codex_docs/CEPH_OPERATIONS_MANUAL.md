# Ceph 操作手册

单节点测试环境：

- hostname：`s52`
- MON IP：`192.168.1.52`
- 数据盘：WD Green SN350 `/dev/nvme1n1`，序列号 `223447804020`
- OSD：两个 450GiB 分区，对应 `osd.0/1`
- 副本：`size=1`、`min_size=1`

两个 OSD 位于同一块物理 SSD，仅用于测试。不要操作系统盘 `/dev/nvme0n1`。

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

## 2. 构建和更新

首次配置：

```bash
cd "$CEPH_REPO"
git submodule update --force --init --recursive --progress
./install-deps.sh
export EXTRA_CMAKE_ARGS="-DWITH_RADOSGW=OFF -DWITH_TESTS=OFF -DWITH_MGR_DASHBOARD_FRONTEND=OFF"
./do_cmake.sh $EXTRA_CMAKE_ARGS
mkdir -p "$CEPH_REPO/src/pybind/mgr/dashboard/frontend/dist"
```

每次代码更新后，Codex 可以直接执行全量构建、安装、ldconfig 和重启：

```bash
cd "$CEPH_REPO/build"
sudo env CCACHE_TEMPDIR=/tmp ninja -j64
sudo ninja install
sudo ldconfig
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo systemctl restart ceph-mgr@${HOST}
sudo ceph -s
```

若修改了公共库、MON/MDS或插件：

```bash
sudo systemctl restart ceph-mon@${HOST} ceph-mds@${HOST}
sudo ceph -s
```

## 3. 初始化 MON/MGR

创建目录并写入配置：

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
```

创建 keyring：

```bash
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
sudo chown ceph:ceph /tmp/ceph.mon.keyring /var/lib/ceph/bootstrap-osd/ceph.keyring
```

初始化 MON/MGR：

```bash
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

确认数据盘身份：

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL
test "$(cat /sys/block/$(basename "$OSD_DISK")/device/serial)" = "$OSD_DISK_SERIAL"
```

以下命令会清空数据盘，只在允许丢弃数据时执行：

```bash
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
sudo ceph osd pool set cephfs_meta size 1 --yes-i-really-mean-it
sudo ceph osd pool set cephfs_meta min_size 1
sudo ceph osd pool set cephfs_meta pg_autoscale_mode off
sudo ceph osd pool set cephfs_data size 1 --yes-i-really-mean-it
sudo ceph osd pool set cephfs_data min_size 1
sudo ceph osd pool set cephfs_data pg_autoscale_mode off
sudo ceph fs new myfs cephfs_meta cephfs_data
```

已有 CephFS 时只调整 PG 和关闭 autoscale：

```bash
sudo ceph osd pool set cephfs_data pg_num 128
sudo ceph osd pool set cephfs_data pgp_num 128
sudo ceph osd pool set cephfs_meta pg_num 16
sudo ceph osd pool set cephfs_meta pgp_num 16
sudo ceph osd pool set cephfs_data pg_autoscale_mode off
sudo ceph osd pool set cephfs_meta pg_autoscale_mode off
sudo ceph -s
sudo ceph osd pool ls detail
```

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

## 6. 测试和观测

```bash
/home/chris/测试脚本/单节点测试/造数据.sh
/home/chris/测试脚本/单节点测试/后台运行标准IO测试.sh run
```

```bash
sudo ceph daemon osd.0 perf dump object_hp_status
sudo ceph daemon osd.1 perf dump object_hp_status
sudo ceph daemon osd.0 object_hp status
sudo ceph daemon osd.1 object_hp status
sudo ceph osd hp status -f json-pretty
sudo ceph osd hp reset
```

OSD `object_hp status` 直接读取实时训练队列；测试结束后不要只用可能滞后的
perf/MGR 队列字段判断训练是否完成。

汇总包括 `samples`、`heat_state`、`confusion_matrix`、`actual_behavior`、
`prediction`、`training`、`latency`、`read_ops` 和 `write_ops`。
`actual_behavior` 当前按 object 级统计；重点看 precision、recall 及 TP/FP/TN/FN。
