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
sudo ninja install
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

## 6. 五负载测试与观测

正式测试位于 `/home/chris/ceph-test/new_workload/`。修改负载后先执行：

```bash
cd /home/chris/ceph-test
./new_workload/validate_all.sh
```

以 MapReduce 为例，造数据时关闭识别，测试前重新启用；enable/disable 都会 reset：

```bash
sudo ceph osd hp disable -f json-pretty
./new_workload/bigdata_mapreduce_vdbench_v1/prepare_data.sh
sudo ceph osd hp enable -f json-pretty
./new_workload/bigdata_mapreduce_vdbench_v1/run_test.sh
```

其他负载同样使用各目录的 `prepare_data.sh` 和 `run_test.sh`。容量和阶段定义以
`/home/chris/ceph-test/new_workload/README.md`、`WORKLOAD_SUMMARY.md` 为准。

每轮测试前后的开关、reset、状态采集和归零判据统一见
[MGR 操作说明](MGR_HP_OPERATIONS.md)。OSD `object_hp status` 是实时状态，
Perf/MGR 汇总可能短暂滞后。
