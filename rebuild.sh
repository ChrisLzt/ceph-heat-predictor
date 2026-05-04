#!/bin/bash
set -eu

cd build
ninja -j32
sudo ninja install
sudo ldconfig
