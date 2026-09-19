#!/bin/bash
set -euo pipefail
unset CEPH_CONF
git config --global --add safe.directory /lab/src
test "$(git -C /lab/src rev-parse HEAD)" = fbfd7114508d14b7e582263bd8a4fbc3883fa39f
cmake --build /lab/build --parallel 12 --target ceph-osd unittest_bluestore_onode_cache unittest_bluestore_types
/lab/build/bin/unittest_bluestore_onode_cache --gtest_repeat=10 --gtest_break_on_failure --gtest_output=xml:/lab/results/onode-tests.xml
/lab/build/bin/unittest_bluestore_types --gtest_filter=-sb_info_space_efficient_map_t.size:bluestore_blob_t.csum_bench --gtest_output=xml:/lab/results/bluestore-types.xml
cmake --build /lab/build-mgr --parallel 12 --target ceph-mgr
/lab/build/bin/ceph-osd --version
/lab/build-mgr/bin/ceph-mgr --version
