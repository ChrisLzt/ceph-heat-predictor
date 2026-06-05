#!/bin/bash
set -eu

sudo rm -f /var/log/ceph/ceph-osd.0.log
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo systemctl restart ceph-osd@0
