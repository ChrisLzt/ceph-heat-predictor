#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/hust/ceph-heat-predictor}"
BUILD_DIR="${BUILD_DIR:-${REPO_DIR}/build}"

sudo install -m 0755 "${BUILD_DIR}/bin/ceph-osd" /usr/bin/ceph-osd

sudo install -d -m 0755 /usr/lib/ceph/erasure-code
sudo install -m 0755 "${BUILD_DIR}"/lib/libec_*.so /usr/lib/ceph/erasure-code/

sudo install -d -m 0755 /usr/lib/ceph/compressor
sudo install -m 0755 "${BUILD_DIR}"/lib/libceph_snappy.so.2.0.0 /usr/lib/ceph/compressor/
sudo install -m 0755 "${BUILD_DIR}"/lib/libceph_zlib.so.2.0.0 /usr/lib/ceph/compressor/
sudo install -m 0755 "${BUILD_DIR}"/lib/libceph_zstd.so.2.0.0 /usr/lib/ceph/compressor/
sudo install -m 0755 "${BUILD_DIR}"/lib/libceph_lz4.so.2.0.0 /usr/lib/ceph/compressor/
sudo ln -sfn libceph_snappy.so.2.0.0 /usr/lib/ceph/compressor/libceph_snappy.so.2
sudo ln -sfn libceph_zlib.so.2.0.0 /usr/lib/ceph/compressor/libceph_zlib.so.2
sudo ln -sfn libceph_zstd.so.2.0.0 /usr/lib/ceph/compressor/libceph_zstd.so.2
sudo ln -sfn libceph_lz4.so.2.0.0 /usr/lib/ceph/compressor/libceph_lz4.so.2
sudo ln -sfn libceph_snappy.so.2 /usr/lib/ceph/compressor/libceph_snappy.so
sudo ln -sfn libceph_zlib.so.2 /usr/lib/ceph/compressor/libceph_zlib.so
sudo ln -sfn libceph_zstd.so.2 /usr/lib/ceph/compressor/libceph_zstd.so
sudo ln -sfn libceph_lz4.so.2 /usr/lib/ceph/compressor/libceph_lz4.so

sudo install -d -m 0755 /usr/lib/ceph/crypto
sudo install -m 0755 "${BUILD_DIR}"/lib/libceph_crypto_openssl.so /usr/lib/ceph/crypto/
sudo install -m 0755 "${BUILD_DIR}"/lib/libceph_crypto_isal.so.1.0.0 /usr/lib/ceph/crypto/
sudo ln -sfn libceph_crypto_isal.so.1.0.0 /usr/lib/ceph/crypto/libceph_crypto_isal.so.1
sudo ln -sfn libceph_crypto_isal.so.1 /usr/lib/ceph/crypto/libceph_crypto_isal.so

echo "Installed ceph-osd and Ceph plugins from ${BUILD_DIR}"
