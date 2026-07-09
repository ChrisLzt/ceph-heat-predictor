#!/bin/bash

set -eux

export HOST=s52
export CEPH_REPO=/home/chris/ceph-heat-predictor
cd "$CEPH_REPO/build"
sudo ninja -j64
sudo ninja install
sudo ldconfig