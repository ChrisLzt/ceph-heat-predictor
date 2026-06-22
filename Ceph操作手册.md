# Ceph 操作手册

本文档用于单节点测试集群：

- hostname：`s52`
- MON IP：`192.168.1.52`
- OSD：两个 100G loop 文件
- 副本：单副本，`size=1`、`min_size=1`
- 目的：测试 `ceph-heat-predictor`，不是生产部署

不要对系统盘 `/dev/nvme0n1` 执行 `parted`、`wipefs`、`mkfs` 或 `ceph-volume zap --destroy`。

## 0. 变量

```bash
export HOST=s52
export MON_IP=192.168.1.52
export PUBLIC_NETWORK=192.168.1.0/24
export FSID=$(uuidgen)
export CEPH_REPO=/home/chris/ceph-heat-predictor
export LOOP_DIR=/var/lib/ceph-loop
export LOOP_SIZE=100G
```

挂载点约定：

- `/mnt/cephfs`：当前 Vdbench/CephFS 测试使用。
- `/mnt/ceph_test`：仅 RBD 测试使用，不要和 CephFS 混用。

## 1. 构建

首次准备：

```bash
cd "$CEPH_REPO"
git submodule update --force --init --recursive --progress
./install-deps.sh
sudo apt install -y libfmt-dev libsqlite3-dev liblttng-ust-dev xfslibs-dev
sudo apt install -y python3-prettytable python3-pecan
```

配置构建：

```bash
cd "$CEPH_REPO"
export EXTRA_CMAKE_ARGS="-DWITH_RADOSGW=OFF -DWITH_TESTS=OFF -DWITH_MGR_DASHBOARD_FRONTEND=OFF"
./do_cmake.sh $EXTRA_CMAKE_ARGS
```

编译常用目标：

```bash
cd "$CEPH_REPO/build"
env CCACHE_TEMPDIR=/tmp ninja ceph-osd ceph-mgr -j"$(nproc)"
```

如果执行 `sudo ninja install` 且 dashboard frontend 被关闭，可能需要补空目录：

```bash
mkdir -p "$CEPH_REPO/src/pybind/mgr/dashboard/frontend/dist"
```

只更新 OSD/MGR：

```bash
cd "$CEPH_REPO/build"
sudo install -o root -g root -m 0755 bin/ceph-osd /usr/bin/ceph-osd
sudo install -o root -g root -m 0755 bin/ceph-mgr /usr/bin/ceph-mgr
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo systemctl restart ceph-mgr@${HOST}
sudo ceph -s
```

## 2. 重装前清理

查看残留：

```bash
pgrep -af 'ceph|ceph-|rados|rbd|radosgw|ceph-osd|ceph-mon|ceph-mgr|ceph-mds|ceph-volume|cephadm'
systemctl list-units --type=service --all 'ceph*' 'rbd*'
systemctl list-unit-files 'ceph*' 'rbd*'
```

停止自启：

```bash
sudo systemctl stop ceph.target ceph-mon.target ceph-mgr.target ceph-osd.target ceph-radosgw.target rbdmap.service || true
sudo systemctl disable ceph.target rbdmap.service || true
sudo systemctl disable /etc/systemd/system/multi-user.target.wants/ceph-volume@*.service || true
sudo systemctl daemon-reload
sudo systemctl reset-failed
```

备份旧配置和数据：

```bash
sudo mv /etc/ceph /etc/ceph.bak.$(date +%F-%H%M%S) 2>/dev/null || true
sudo mv /var/lib/ceph /var/lib/ceph.bak.$(date +%F-%H%M%S) 2>/dev/null || true
sudo mv /var/log/ceph /var/log/ceph.bak.$(date +%F-%H%M%S) 2>/dev/null || true
```

## 3. 初始化 MON/MGR

创建目录：

```bash
sudo install -d -o ceph -g ceph -m 0755 /etc/ceph /run/ceph /var/run/ceph
sudo install -d -o ceph -g ceph -m 0755 /var/log/ceph /var/lib/ceph
sudo install -d -o ceph -g ceph -m 0755 /var/lib/ceph/osd /var/lib/ceph/bootstrap-osd
```

写入 `/etc/ceph/ceph.conf`：

```bash
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
mon_data_avail_warn = 10
mon_data_avail_crit = 2
EOF
sudo chown ceph:ceph /etc/ceph/ceph.conf
sudo chmod 644 /etc/ceph/ceph.conf
```

创建 keyring：

```bash
sudo ceph-authtool --create-keyring /tmp/ceph.mon.keyring \
  --gen-key -n mon. --cap mon 'allow *'

sudo ceph-authtool --create-keyring /etc/ceph/ceph.client.admin.keyring \
  --gen-key -n client.admin \
  --cap mon 'allow *' --cap osd 'allow *' --cap mds 'allow *' --cap mgr 'allow *'

sudo ceph-authtool --create-keyring /var/lib/ceph/bootstrap-osd/ceph.keyring \
  --gen-key -n client.bootstrap-osd \
  --cap mon 'profile bootstrap-osd' --cap mgr 'allow r'

sudo ceph-authtool /tmp/ceph.mon.keyring --import-keyring /etc/ceph/ceph.client.admin.keyring
sudo ceph-authtool /tmp/ceph.mon.keyring --import-keyring /var/lib/ceph/bootstrap-osd/ceph.keyring
sudo chown ceph:ceph /tmp/ceph.mon.keyring /var/lib/ceph/bootstrap-osd/ceph.keyring
sudo chown root:ceph /etc/ceph/ceph.client.admin.keyring
sudo chmod 640 /etc/ceph/ceph.client.admin.keyring
```

初始化 MON：

```bash
monmaptool --create --add "$HOST" "$MON_IP" --fsid "$FSID" /tmp/monmap
sudo install -d -o ceph -g ceph -m 0755 /var/lib/ceph/mon/ceph-${HOST}
sudo -u ceph ceph-mon --mkfs -i "$HOST" --monmap /tmp/monmap --keyring /tmp/ceph.mon.keyring
sudo systemctl start ceph-mon@${HOST}
sudo ceph mon enable-msgr2
sudo ceph config set mon auth_allow_insecure_global_id_reclaim false
sudo ceph -s
```

初始化 MGR：

```bash
sudo install -d -o ceph -g ceph -m 0755 /var/lib/ceph/mgr/ceph-${HOST}
sudo ceph auth get-or-create mgr.${HOST} \
  mon 'allow profile mgr' osd 'allow *' mds 'allow *' \
  | sudo tee /var/lib/ceph/mgr/ceph-${HOST}/keyring >/dev/null
sudo chown ceph:ceph /var/lib/ceph/mgr/ceph-${HOST}/keyring
sudo chmod 600 /var/lib/ceph/mgr/ceph-${HOST}/keyring
sudo systemctl start ceph-mgr@${HOST}
sudo ceph -s
```

## 4. 两个 loop OSD

创建 loop 文件和 LVM：

```bash
sudo install -d -o root -g root -m 0755 "$LOOP_DIR"
sudo fallocate -l "$LOOP_SIZE" "$LOOP_DIR/osd0.img"
sudo fallocate -l "$LOOP_SIZE" "$LOOP_DIR/osd1.img"

export LOOP0=$(sudo losetup --find --show "$LOOP_DIR/osd0.img")
export LOOP1=$(sudo losetup --find --show "$LOOP_DIR/osd1.img")

sudo pvcreate "$LOOP0"
sudo pvcreate "$LOOP1"
sudo vgcreate ceph-loop0 "$LOOP0"
sudo vgcreate ceph-loop1 "$LOOP1"
sudo lvcreate -n osd0 -l 100%FREE ceph-loop0
sudo lvcreate -n osd1 -l 100%FREE ceph-loop1
```

创建 OSD：

```bash
sudo ceph-volume lvm create --data /dev/ceph-loop0/osd0
sudo ceph-volume lvm create --data /dev/ceph-loop1/osd1
sudo ceph osd tree
sudo ceph -s
```

断电或重启后，loop 映射会消失。如果 `.img` 文件还在，不要 purge，也不要重新 `truncate/fallocate`，只需恢复映射：

```bash
sudo losetup --find --show "$LOOP_DIR/osd0.img"
sudo losetup --find --show "$LOOP_DIR/osd1.img"
sudo vgchange -ay ceph-loop0 ceph-loop1
sudo ceph-volume lvm activate --all
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo ceph -s
```

## 5. CephFS 和测试目录

启动 MDS：

```bash
sudo install -d -o ceph -g ceph -m 0755 /var/lib/ceph/mds/ceph-${HOST}
sudo ceph auth get-or-create mds.${HOST} \
  mon 'profile mds' mgr 'profile mds' mds 'allow *' osd 'allow *' \
  | sudo tee /var/lib/ceph/mds/ceph-${HOST}/keyring >/dev/null
sudo chown ceph:ceph /var/lib/ceph/mds/ceph-${HOST}/keyring
sudo chmod 600 /var/lib/ceph/mds/ceph-${HOST}/keyring
sudo systemctl start ceph-mds@${HOST}
```

创建 CephFS：

```bash
sudo ceph osd pool create cephfs_meta 16 16
sudo ceph osd pool create cephfs_data 64 64
sudo ceph osd pool set cephfs_meta size 1 --yes-i-really-mean-it
sudo ceph osd pool set cephfs_meta min_size 1
sudo ceph osd pool set cephfs_data size 1 --yes-i-really-mean-it
sudo ceph osd pool set cephfs_data min_size 1
sudo ceph fs new myfs cephfs_meta cephfs_data
sudo ceph fs status
```

挂载：

```bash
sudo grep "key =" /etc/ceph/ceph.client.admin.keyring \
  | awk '{print $3}' \
  | sudo tee /etc/ceph/admin.secret >/dev/null
sudo chmod 600 /etc/ceph/admin.secret

sudo install -d -m 0755 /mnt/cephfs
sudo mount -t ceph ${MON_IP}:6789:/ /mnt/cephfs \
  -o rw,name=admin,secretfile=/etc/ceph/admin.secret
sudo install -d -o "$USER" -g "$USER" -m 0755 /mnt/cephfs/vdbench
findmnt /mnt/cephfs
df -h /mnt/cephfs
```

当前单节点冷热测试：

```bash
/home/chris/测试脚本/单节点测试/造数据.sh
/home/chris/测试脚本/单节点测试/后台运行标准IO测试.sh run
```

后台和停止：

```bash
/home/chris/测试脚本/单节点测试/后台运行标准IO测试.sh background
/home/chris/测试脚本/单节点测试/后台运行标准IO测试.sh stop
```

当前数据集只保留 1G 文件冷热数据：hot 30GiB，cold 120GiB，总计 150GiB；运行测试为 4KB 随机读写，32 线程，`rdpct=50`，hot/cold 访问 skew 为 `80/20`。

## 6. Heat Predictor 观测

单 OSD：

```bash
sudo ceph daemon osd.0 perf dump object_hp_status
sudo ceph daemon osd.1 perf dump object_hp_status
```

全局汇总和清空：

```bash
sudo ceph osd hp status
sudo ceph osd hp status -f json-pretty
sudo ceph osd hp reset
```

`ceph osd hp status` 默认只输出汇总分组：

- `samples`
- `confusion_matrix`
- `prediction`
- `training`
- `label_queue`
- `read_ops`
- `write_ops`

重点看 precision、recall、TP/FP/TN/FN，不要只看 `hp_hot_accuracy`。

## 7. 常见故障

`ceph-mon` 因 `/run/ceph` 缺失 ABRT：

```bash
sudo install -d -o ceph -g ceph -m 0775 /run/ceph
echo 'd /run/ceph 0775 ceph ceph -' | sudo tee /etc/tmpfiles.d/ceph.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/ceph.conf
sudo systemctl restart ceph-mon@${HOST}
```

`ceph` 命令缺 Python 包：

```bash
sudo apt install -y python3-prettytable
```

`restful` 模块缺 `pecan`：

```bash
sudo apt install -y python3-pecan
sudo systemctl restart ceph-mgr@${HOST}
```

构建缺开发库：

```bash
sudo apt install -y libfmt-dev libsqlite3-dev liblttng-ust-dev xfslibs-dev
```

如果 `health detail` 只显示 `POOL_NO_REDUNDANCY`，且当前就是单节点单副本测试，可以忽略。

## 8. 危险清理

删除 CephFS 和 pool：

```bash
sudo umount /mnt/cephfs
sudo ceph fs fail myfs
sudo ceph fs rm myfs --yes-i-really-mean-it
sudo ceph osd pool delete cephfs_data cephfs_data --yes-i-really-really-mean-it
sudo ceph osd pool delete cephfs_meta cephfs_meta --yes-i-really-really-mean-it
```

删除旧 OSD 记录：

```bash
sudo ceph osd out <id>
sudo systemctl stop ceph-osd@<id>
sudo ceph osd crush remove osd.<id>
sudo ceph auth del osd.<id>
sudo ceph osd rm <id>
```

删除 loop OSD 文件会丢失数据：

```bash
losetup -a | grep "$LOOP_DIR"
sudo losetup -d /dev/loopX
sudo vgremove -ff -y ceph-loop0 ceph-loop1
sudo rm -f "$LOOP_DIR/osd0.img" "$LOOP_DIR/osd1.img"
```
