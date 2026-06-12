// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "ObjectHeatPredictor.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>

#include "common/ceph_context.h"
#include "common/ceph_time.h"
#include "common/hobject.h"
#include "common/perf_counters.h"
#include "include/rados.h"
#include "heatpredictor/heat_predictor.h"

namespace {

HeatPredictor osd_object_heat_predictor;
PerfCounters *osd_object_hp_logger = nullptr;
std::mutex osd_object_hp_logger_mtx;
static constexpr uint64_t object_hp_logger_update_interval = 1000;

enum {
  object_hp_first = 592000,
  object_hp_count,
  object_hp_train_total,
  object_hp_hot_percent,
  object_hp_eval_hot_percent,
  object_hp_actual_hot_percent,
  object_hp_accuracy,
  object_hp_hot_precision,
  object_hp_hot_recall,
  object_hp_hot_threshold,
  object_hp_train_queue_length,
  object_hp_swap_count,
  object_hp_dequeue_waiting_count,
  object_hp_dequeue_max_size_count,
  object_hp_op_read_count,
  object_hp_op_sync_read_count,
  object_hp_op_sparse_read_count,
  object_hp_op_write_count,
  object_hp_op_writefull_count,
  object_hp_op_writesame_count,
  object_hp_predict_latency,
  object_hp_last
};

struct ObjectHpOpCounters {
  std::atomic<uint64_t> read{0};
  std::atomic<uint64_t> sync_read{0};
  std::atomic<uint64_t> sparse_read{0};
  std::atomic<uint64_t> write{0};
  std::atomic<uint64_t> writefull{0};
  std::atomic<uint64_t> writesame{0};
};

ObjectHpOpCounters osd_object_hp_op_counters;

static inline uint64_t hp_mul10000(double x)
{
  if (x <= 0) {
    return 0;
  }
  return static_cast<uint64_t>(x * 10000);
}

static void hp_ensure_object_logger(CephContext *cct)
{
  if (osd_object_hp_logger != nullptr || cct == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(osd_object_hp_logger_mtx);
  if (osd_object_hp_logger != nullptr) {
    return;
  }

  PerfCountersBuilder b(cct, "object_hp_status", object_hp_first, object_hp_last);
  b.add_u64(object_hp_count, "hp_count", "count");
  b.add_u64(object_hp_train_total, "hp_train_total", "train total");
  b.add_u64(object_hp_hot_percent, "hp_hot_percent", "hot percent (x10000)");
  b.add_u64(object_hp_eval_hot_percent, "hp_eval_hot_percent", "evaluated prediction hot percent (x10000)");
  b.add_u64(object_hp_actual_hot_percent, "hp_actual_hot_percent", "actual hot percent (x10000)");
  b.add_u64(object_hp_accuracy, "hp_accuracy", "accuracy (x10000)");
  b.add_u64(object_hp_hot_precision, "hp_hot_precision", "hot precision (x10000)");
  b.add_u64(object_hp_hot_recall, "hp_hot_recall", "hot recall (x10000)");
  b.add_u64(object_hp_hot_threshold, "hp_hot_threshold", "threshold (x10000)");
  b.add_u64(object_hp_train_queue_length, "hp_train_queue_length", "train queue length");
  b.add_u64(object_hp_swap_count, "hp_swap_count", "model swap count");
  b.add_u64(object_hp_dequeue_waiting_count, "hp_dequeue_waiting_count", "dequeue count caused by waiting timeout");
  b.add_u64(object_hp_dequeue_max_size_count, "hp_dequeue_max_size_count", "dequeue count caused by queue max size");
  b.add_u64(object_hp_op_read_count, "hp_op_read_count", "read op count");
  b.add_u64(object_hp_op_sync_read_count, "hp_op_sync_read_count", "sync read op count");
  b.add_u64(object_hp_op_sparse_read_count, "hp_op_sparse_read_count", "sparse read op count");
  b.add_u64(object_hp_op_write_count, "hp_op_write_count", "write op count");
  b.add_u64(object_hp_op_writefull_count, "hp_op_writefull_count", "writefull op count");
  b.add_u64(object_hp_op_writesame_count, "hp_op_writesame_count", "writesame op count");
  b.add_time_avg(object_hp_predict_latency, "hp_predict_latency", "predict latency");
  osd_object_hp_logger = b.create_perf_counters();
  cct->get_perfcounters_collection()->add(osd_object_hp_logger);
}

static void hp_update_object_logger(ceph::timespan predict_latency)
{
  PerfCounters *logger = osd_object_hp_logger;
  if (logger == nullptr) {
    return;
  }

  uint64_t cnt = osd_object_heat_predictor.hp_index.load();
  uint64_t train_total = osd_object_heat_predictor.get_total_weight();
  uint64_t hot_cnt = osd_object_heat_predictor.hot_cnt.load();
  uint64_t cold_cnt = osd_object_heat_predictor.cold_cnt.load();
  double hot_percent = (hot_cnt + cold_cnt > 0)
    ? static_cast<double>(hot_cnt) / (hot_cnt + cold_cnt) : 0;
  uint64_t actual_hot = osd_object_heat_predictor.get_actual_hot();
  uint64_t actual_cold = osd_object_heat_predictor.get_actual_cold();
  double actual_hot_percent = (actual_hot + actual_cold > 0)
    ? static_cast<double>(actual_hot) / (actual_hot + actual_cold) : 0;

  logger->tinc(object_hp_predict_latency, predict_latency);
  logger->set(object_hp_count, cnt);
  logger->set(object_hp_train_total, train_total);
  logger->set(object_hp_hot_percent, hp_mul10000(hot_percent));
  logger->set(object_hp_eval_hot_percent, hp_mul10000(osd_object_heat_predictor.get_hot_prediction_percent()));
  logger->set(object_hp_actual_hot_percent, hp_mul10000(actual_hot_percent));
  logger->set(object_hp_accuracy, hp_mul10000(osd_object_heat_predictor.get_accuracy()));
  logger->set(object_hp_hot_precision, hp_mul10000(osd_object_heat_predictor.get_hot_precision()));
  logger->set(object_hp_hot_recall, hp_mul10000(osd_object_heat_predictor.get_hot_recall()));
  logger->set(object_hp_hot_threshold, hp_mul10000(osd_object_heat_predictor.get_hot_threshold()));
  logger->set(object_hp_train_queue_length, osd_object_heat_predictor.get_train_queue_length());
  logger->set(object_hp_swap_count, osd_object_heat_predictor.get_swap_count());
  logger->set(object_hp_dequeue_waiting_count, osd_object_heat_predictor.get_dequeue_waiting_count());
  logger->set(object_hp_dequeue_max_size_count, osd_object_heat_predictor.get_dequeue_max_size_count());
  logger->set(object_hp_op_read_count, osd_object_hp_op_counters.read.load(std::memory_order_relaxed));
  logger->set(object_hp_op_sync_read_count, osd_object_hp_op_counters.sync_read.load(std::memory_order_relaxed));
  logger->set(object_hp_op_sparse_read_count, osd_object_hp_op_counters.sparse_read.load(std::memory_order_relaxed));
  logger->set(object_hp_op_write_count, osd_object_hp_op_counters.write.load(std::memory_order_relaxed));
  logger->set(object_hp_op_writefull_count, osd_object_hp_op_counters.writefull.load(std::memory_order_relaxed));
  logger->set(object_hp_op_writesame_count, osd_object_hp_op_counters.writesame.load(std::memory_order_relaxed));
}

static inline bool hp_should_update_object_logger(uint64_t index)
{
  return (index % object_hp_logger_update_interval) == 0;
}

static inline bool hp_track_osd_op(uint16_t op)
{
  switch (op) {
  case CEPH_OSD_OP_READ:
  case CEPH_OSD_OP_SYNC_READ:
  case CEPH_OSD_OP_SPARSE_READ:
  case CEPH_OSD_OP_WRITE:
  case CEPH_OSD_OP_WRITEFULL:
  case CEPH_OSD_OP_WRITESAME:
    return true;
  default:
    return false;
  }
}

static inline int hp_osd_op_type(uint16_t op)
{
  switch (op) {
  case CEPH_OSD_OP_READ:
  case CEPH_OSD_OP_SYNC_READ:
  case CEPH_OSD_OP_SPARSE_READ:
    return 1;
  case CEPH_OSD_OP_WRITE:
  case CEPH_OSD_OP_WRITEFULL:
  case CEPH_OSD_OP_WRITESAME:
    return 0;
  default:
    return 0;
  }
}

static inline void hp_count_osd_op(uint16_t op)
{
  switch (op) {
  case CEPH_OSD_OP_READ:
    osd_object_hp_op_counters.read.fetch_add(1, std::memory_order_relaxed);
    break;
  case CEPH_OSD_OP_SYNC_READ:
    osd_object_hp_op_counters.sync_read.fetch_add(1, std::memory_order_relaxed);
    break;
  case CEPH_OSD_OP_SPARSE_READ:
    osd_object_hp_op_counters.sparse_read.fetch_add(1, std::memory_order_relaxed);
    break;
  case CEPH_OSD_OP_WRITE:
    osd_object_hp_op_counters.write.fetch_add(1, std::memory_order_relaxed);
    break;
  case CEPH_OSD_OP_WRITEFULL:
    osd_object_hp_op_counters.writefull.fetch_add(1, std::memory_order_relaxed);
    break;
  case CEPH_OSD_OP_WRITESAME:
    osd_object_hp_op_counters.writesame.fetch_add(1, std::memory_order_relaxed);
    break;
  default:
    break;
  }
}

static inline void hp_osd_op_extent(
  const ceph_osd_op& op,
  uint64_t *offset,
  uint64_t *length)
{
  if (op.op == CEPH_OSD_OP_WRITESAME) {
    *offset = op.writesame.offset;
    *length = op.writesame.length;
  } else if (op.op == CEPH_OSD_OP_WRITEFULL) {
    *offset = 0;
    *length = op.extent.length;
  } else {
    *offset = op.extent.offset;
    *length = op.extent.length;
  }
}

} // namespace

void init_osd_object_hp_status(CephContext *cct)
{
  hp_ensure_object_logger(cct);
}

void hp_notify_osd_object_op(CephContext *cct,
                             const hobject_t& soid,
                             const ceph_osd_op& op)
{
  if (!hp_track_osd_op(op.op)) {
    return;
  }

  uint64_t offset = 0;
  uint64_t length = 0;
  hp_osd_op_extent(op, &offset, &length);
  if (length == 0) {
    return;
  }

  hp_count_osd_op(op.op);
  hp_ensure_object_logger(cct);
  auto start_time = ceph::mono_clock::now();
  uint64_t index = osd_object_heat_predictor.hp_index.fetch_add(1) + 1;
  osd_object_heat_predictor.predict(
    index,
    hp_osd_op_type(op.op),
    length,
    soid.pool,
    soid.get_hash(),
    std::hash<object_t>{}(soid.oid),
    offset,
    1);
  auto end_time = ceph::mono_clock::now();
  if (hp_should_update_object_logger(index)) {
    hp_update_object_logger(end_time - start_time);
  }
}
