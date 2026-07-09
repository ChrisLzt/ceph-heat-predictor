#!/bin/bash

set -eux

export HOST=s52
export CEPH_REPO=/home/chris/ceph-heat-predictor
sudo systemctl reset-failed ceph-osd@0 ceph-osd@1
sudo systemctl reset-failed ceph-mgr@${HOST}
sudo systemctl restart ceph-osd@0 ceph-osd@1
sudo systemctl restart ceph-mgr@${HOST}
sudo ceph -s