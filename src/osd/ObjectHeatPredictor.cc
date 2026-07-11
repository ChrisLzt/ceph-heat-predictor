// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "ObjectHeatPredictor.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>

#include "common/ceph_context.h"
#include "common/ceph_time.h"
#include "common/Formatter.h"
#include "common/hobject.h"
#include "common/perf_counters.h"
#include "include/rados.h"
#include "heatpredictor/heat_predictor.h"

namespace {

HeatPredictor osd_object_heat_predictor;
PerfCounters *osd_object_hp_logger = nullptr;
std::mutex osd_object_hp_logger_mtx;
std::shared_mutex osd_object_hp_reset_mtx;
static constexpr uint64_t object_hp_logger_update_interval = 1000;

enum {
  object_hp_first = 591422,
  object_hp_enabled,
  object_hp_io_count,
  object_hp_labeled_io_total,
  object_hp_pending_io_count,
  object_hp_heat_state_count,
  object_hp_lru_count,
  object_hp_otsu_histogram_bin_count,
  object_hp_otsu_histogram_object_count,
  object_hp_true_positive_count,
  object_hp_false_positive_count,
  object_hp_true_negative_count,
  object_hp_false_negative_count,
  object_hp_actual_hot_object_avg_future_access_count,
  object_hp_actual_cold_object_avg_future_access_count,
  object_hp_actual_hot_object_avg_heat,
  object_hp_actual_cold_object_avg_heat,
  object_hp_actual_hot_future_access_p99,
  object_hp_actual_hot_future_access_p95,
  object_hp_actual_hot_future_access_p50,
  object_hp_actual_cold_future_access_p99,
  object_hp_actual_cold_future_access_p95,
  object_hp_actual_cold_future_access_p50,
  object_hp_actual_hot_future_heat_p99,
  object_hp_actual_hot_future_heat_p95,
  object_hp_actual_hot_future_heat_p50,
  object_hp_actual_cold_future_heat_p99,
  object_hp_actual_cold_future_heat_p95,
  object_hp_actual_cold_future_heat_p50,
  object_hp_hot_accuracy,
  object_hp_hot_precision,
  object_hp_hot_recall,
  object_hp_eval_pred_hot_percent,
  object_hp_eval_actual_hot_percent,
  object_hp_actual_hot_avg_pred_hot_percent,
  object_hp_actual_cold_avg_pred_hot_percent,
  object_hp_hot_predict_threshold,
  object_hp_hot_predict_threshold_target,
  object_hp_predict_calibration_sample_count,
  object_hp_predict_calibration_current_accuracy,
  object_hp_predict_calibration_target_accuracy,
  object_hp_hot_threshold,
  object_hp_otsu_candidate_threshold,
  object_hp_otsu_separation,
  object_hp_otsu_confidence,
  object_hp_otsu_sample_confidence,
  object_hp_otsu_sharpness_confidence,
  object_hp_hot_threshold_method,
  object_hp_train_queue_length,
  object_hp_train_drop_count,
  object_hp_snapshot_publish_count,
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

static void hp_reset_osd_op_counters()
{
  osd_object_hp_op_counters.read.store(0, std::memory_order_relaxed);
  osd_object_hp_op_counters.sync_read.store(0, std::memory_order_relaxed);
  osd_object_hp_op_counters.sparse_read.store(0, std::memory_order_relaxed);
  osd_object_hp_op_counters.write.store(0, std::memory_order_relaxed);
  osd_object_hp_op_counters.writefull.store(0, std::memory_order_relaxed);
  osd_object_hp_op_counters.writesame.store(0, std::memory_order_relaxed);
}

static inline uint64_t hp_mul10000(double x)
{
  if (x <= 0) {
    return 0;
  }
  return static_cast<uint64_t>(x * 10000);
}

static inline uint64_t hp_ratio10000(uint64_t numerator, uint64_t denominator)
{
  if (denominator == 0 || numerator == 0) {
    return 0;
  }
  return static_cast<uint64_t>(
    (static_cast<long double>(numerator) * 10000) / denominator);
}

static inline uint64_t hp_avg10000(double sum, uint64_t count)
{
  if (count == 0 || sum <= 0) {
    return 0;
  }
  return static_cast<uint64_t>(
    (static_cast<long double>(sum) * 10000) / count);
}

static void hp_set_distribution_summary(
  PerfCounters *logger,
  int p99_id,
  int p95_id,
  int p50_id,
  const HpDistributionSummary& summary)
{
  logger->set(p99_id, hp_mul10000(summary.p99));
  logger->set(p95_id, hp_mul10000(summary.p95));
  logger->set(p50_id, hp_mul10000(summary.p50));
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
  b.set_prio_default(PerfCountersBuilder::PRIO_USEFUL);
  b.add_u64(object_hp_enabled, "hp_enabled", "heat predictor enabled");
  b.add_u64(object_hp_io_count, "hp_io_count", "predicted I/O total");
  b.add_u64(object_hp_labeled_io_total, "hp_labeled_io_total", "evaluated I/O total");
  b.add_u64(object_hp_pending_io_count, "hp_pending_io_count", "pending evaluation I/O count");
  b.add_u64(object_hp_heat_state_count, "hp_heat_state_count", "tracked object heat state count");
  b.add_u64(object_hp_lru_count, "hp_lru_count", "object heat states in LRU");
  b.add_u64(object_hp_otsu_histogram_bin_count,
            "hp_otsu_histogram_bin_count",
            "occupied Otsu histogram bin count");
  b.add_u64(object_hp_otsu_histogram_object_count,
            "hp_otsu_histogram_object_count",
            "objects represented by the Otsu histogram");
  b.add_u64(object_hp_true_positive_count, "hp_true_positive_count", "true positive count");
  b.add_u64(object_hp_false_positive_count, "hp_false_positive_count", "false positive count");
  b.add_u64(object_hp_true_negative_count, "hp_true_negative_count", "true negative count");
  b.add_u64(object_hp_false_negative_count, "hp_false_negative_count", "false negative count");
  b.add_u64(object_hp_actual_hot_object_avg_future_access_count,
            "hp_actual_hot_object_avg_future_access_count",
            "average future access count of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_cold_object_avg_future_access_count,
            "hp_actual_cold_object_avg_future_access_count",
            "average future access count of actual cold objects (x10000)");
  b.add_u64(object_hp_actual_hot_object_avg_heat,
            "hp_actual_hot_object_avg_heat",
            "average future heat of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_cold_object_avg_heat,
            "hp_actual_cold_object_avg_heat",
            "average future heat of actual cold objects (x10000)");
  b.add_u64(object_hp_actual_hot_future_access_p99,
            "hp_actual_hot_future_access_p99",
            "p99 future access count of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_hot_future_access_p95,
            "hp_actual_hot_future_access_p95",
            "p95 future access count of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_hot_future_access_p50,
            "hp_actual_hot_future_access_p50",
            "p50 future access count of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_cold_future_access_p99,
            "hp_actual_cold_future_access_p99",
            "p99 future access count of actual cold objects (x10000)");
  b.add_u64(object_hp_actual_cold_future_access_p95,
            "hp_actual_cold_future_access_p95",
            "p95 future access count of actual cold objects (x10000)");
  b.add_u64(object_hp_actual_cold_future_access_p50,
            "hp_actual_cold_future_access_p50",
            "p50 future access count of actual cold objects (x10000)");
  b.add_u64(object_hp_actual_hot_future_heat_p99,
            "hp_actual_hot_future_heat_p99",
            "p99 future heat of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_hot_future_heat_p95,
            "hp_actual_hot_future_heat_p95",
            "p95 future heat of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_hot_future_heat_p50,
            "hp_actual_hot_future_heat_p50",
            "p50 future heat of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_cold_future_heat_p99,
            "hp_actual_cold_future_heat_p99",
            "p99 future heat of actual cold objects (x10000)");
  b.add_u64(object_hp_actual_cold_future_heat_p95,
            "hp_actual_cold_future_heat_p95",
            "p95 future heat of actual cold objects (x10000)");
  b.add_u64(object_hp_actual_cold_future_heat_p50,
            "hp_actual_cold_future_heat_p50",
            "p50 future heat of actual cold objects (x10000)");
  b.add_u64(object_hp_hot_accuracy,
            "hp_hot_accuracy",
            "hot prediction accuracy (x10000)");
  b.add_u64(object_hp_hot_precision,
            "hp_hot_precision",
            "hot prediction precision (x10000)");
  b.add_u64(object_hp_hot_recall,
            "hp_hot_recall",
            "hot prediction recall (x10000)");
  b.add_u64(object_hp_eval_pred_hot_percent,
            "hp_eval_pred_hot_percent",
            "evaluated predicted-hot percentage (x10000)");
  b.add_u64(object_hp_eval_actual_hot_percent,
            "hp_eval_actual_hot_percent",
            "evaluated actual-hot percentage (x10000)");
  b.add_u64(object_hp_actual_hot_avg_pred_hot_percent,
            "hp_actual_hot_avg_pred_hot_percent",
            "average predicted hot probability of actual hot objects (x10000)");
  b.add_u64(object_hp_actual_cold_avg_pred_hot_percent,
            "hp_actual_cold_avg_pred_hot_percent",
            "average predicted hot probability of actual cold objects (x10000)");
  b.add_u64(object_hp_hot_predict_threshold,
            "hp_hot_predict_threshold",
            "hot prediction probability threshold (x10000)");
  b.add_u64(object_hp_hot_predict_threshold_target,
            "hp_hot_predict_threshold_target",
            "accuracy-optimal calibration target threshold (x10000)");
  b.add_u64(object_hp_predict_calibration_sample_count,
            "hp_predict_calibration_sample_count",
            "evaluated samples in prediction threshold calibration window");
  b.add_u64(object_hp_predict_calibration_current_accuracy,
            "hp_predict_calibration_current_accuracy",
            "calibration-window accuracy at current threshold (x10000)");
  b.add_u64(object_hp_predict_calibration_target_accuracy,
            "hp_predict_calibration_target_accuracy",
            "calibration-window accuracy at target threshold (x10000)");
  b.add_u64(object_hp_hot_threshold,
            "hp_hot_threshold",
            "effective heat threshold (x10000)");
  b.add_u64(object_hp_otsu_candidate_threshold,
            "hp_otsu_candidate_threshold",
            "current Otsu candidate heat threshold (x10000)");
  b.add_u64(object_hp_otsu_separation,
            "hp_otsu_separation",
            "Otsu between-class variance ratio (x10000)");
  b.add_u64(object_hp_otsu_confidence,
            "hp_otsu_confidence",
            "weighted Otsu threshold confidence (x10000)");
  b.add_u64(object_hp_otsu_sample_confidence,
            "hp_otsu_sample_confidence",
            "Otsu histogram sample-count confidence (x10000)");
  b.add_u64(object_hp_otsu_sharpness_confidence,
            "hp_otsu_sharpness_confidence",
            "Otsu near-optimal-plateau sharpness confidence (x10000)");
  b.add_u64(object_hp_hot_threshold_method,
            "hp_hot_threshold_method",
            "threshold state: 0 initializing, 1 tracking, 2 holding");
  b.add_u64(object_hp_train_queue_length, "hp_train_queue_length", "train queue length");
  b.add_u64(object_hp_train_drop_count, "hp_train_drop_count", "dropped training sample count");
  b.add_u64(object_hp_snapshot_publish_count, "hp_snapshot_publish_count", "prediction snapshot publish count");
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

  auto stats = osd_object_heat_predictor.get_evaluation_stats();
  uint64_t io_count = stats.io_count;
  uint64_t labeled_io_total = stats.labeled_io_total;
  uint64_t true_positive = stats.true_positive;
  uint64_t false_positive = stats.false_positive;
  uint64_t true_negative = stats.true_negative;
  uint64_t false_negative = stats.false_negative;
  uint64_t actual_hot_count = true_positive + false_negative;
  uint64_t actual_cold_count = true_negative + false_positive;
  uint64_t actual_hot_object_avg_future_access_count =
    hp_ratio10000(stats.actual_hot_object_access_count_sum, actual_hot_count);
  uint64_t actual_cold_object_avg_future_access_count =
    hp_ratio10000(stats.actual_cold_object_access_count_sum, actual_cold_count);
  uint64_t actual_hot_object_avg_heat =
    hp_avg10000(stats.actual_hot_object_heat_sum, actual_hot_count);
  uint64_t actual_cold_object_avg_heat =
    hp_avg10000(stats.actual_cold_object_heat_sum, actual_cold_count);
  uint64_t hot_accuracy =
    hp_ratio10000(true_positive + true_negative, labeled_io_total);
  uint64_t hot_precision =
    hp_ratio10000(true_positive, true_positive + false_positive);
  uint64_t hot_recall =
    hp_ratio10000(true_positive, true_positive + false_negative);
  uint64_t eval_pred_hot_percent =
    hp_ratio10000(true_positive + false_positive, labeled_io_total);
  uint64_t eval_actual_hot_percent =
    hp_ratio10000(actual_hot_count, labeled_io_total);
  uint64_t actual_hot_avg_pred_hot_percent =
    hp_avg10000(stats.actual_hot_pred_hot_proba_sum, actual_hot_count);
  uint64_t actual_cold_avg_pred_hot_percent =
    hp_avg10000(stats.actual_cold_pred_hot_proba_sum, actual_cold_count);

  logger->set(object_hp_enabled, stats.enabled ? 1 : 0);
  logger->set(object_hp_io_count, io_count);
  logger->set(object_hp_labeled_io_total, labeled_io_total);
  logger->set(object_hp_pending_io_count, stats.pending_io_count);
  logger->set(object_hp_heat_state_count, stats.heat_state_count);
  logger->set(object_hp_lru_count, stats.lru_count);
  logger->set(object_hp_otsu_histogram_bin_count,
              stats.otsu_histogram_bin_count);
  logger->set(object_hp_otsu_histogram_object_count,
              stats.otsu_histogram_object_count);
  logger->set(object_hp_true_positive_count, true_positive);
  logger->set(object_hp_false_positive_count, false_positive);
  logger->set(object_hp_true_negative_count, true_negative);
  logger->set(object_hp_false_negative_count, false_negative);
  logger->set(object_hp_actual_hot_object_avg_future_access_count,
              actual_hot_object_avg_future_access_count);
  logger->set(object_hp_actual_cold_object_avg_future_access_count,
              actual_cold_object_avg_future_access_count);
  logger->set(object_hp_actual_hot_object_avg_heat,
              actual_hot_object_avg_heat);
  logger->set(object_hp_actual_cold_object_avg_heat,
              actual_cold_object_avg_heat);
  hp_set_distribution_summary(
    logger,
    object_hp_actual_hot_future_access_p99,
    object_hp_actual_hot_future_access_p95,
    object_hp_actual_hot_future_access_p50,
    stats.actual_hot_future_access);
  hp_set_distribution_summary(
    logger,
    object_hp_actual_cold_future_access_p99,
    object_hp_actual_cold_future_access_p95,
    object_hp_actual_cold_future_access_p50,
    stats.actual_cold_future_access);
  hp_set_distribution_summary(
    logger,
    object_hp_actual_hot_future_heat_p99,
    object_hp_actual_hot_future_heat_p95,
    object_hp_actual_hot_future_heat_p50,
    stats.actual_hot_future_heat);
  hp_set_distribution_summary(
    logger,
    object_hp_actual_cold_future_heat_p99,
    object_hp_actual_cold_future_heat_p95,
    object_hp_actual_cold_future_heat_p50,
    stats.actual_cold_future_heat);
  logger->set(object_hp_hot_accuracy, hot_accuracy);
  logger->set(object_hp_hot_precision, hot_precision);
  logger->set(object_hp_hot_recall, hot_recall);
  logger->set(object_hp_eval_pred_hot_percent, eval_pred_hot_percent);
  logger->set(object_hp_eval_actual_hot_percent, eval_actual_hot_percent);
  logger->set(object_hp_actual_hot_avg_pred_hot_percent,
              actual_hot_avg_pred_hot_percent);
  logger->set(object_hp_actual_cold_avg_pred_hot_percent,
              actual_cold_avg_pred_hot_percent);
  logger->set(object_hp_hot_predict_threshold,
              hp_mul10000(stats.hot_predict_threshold));
  logger->set(object_hp_hot_predict_threshold_target,
              hp_mul10000(stats.hot_predict_threshold_target));
  logger->set(object_hp_predict_calibration_sample_count,
              stats.predict_calibration_sample_count);
  logger->set(object_hp_predict_calibration_current_accuracy,
              hp_mul10000(stats.predict_calibration_current_accuracy));
  logger->set(object_hp_predict_calibration_target_accuracy,
              hp_mul10000(stats.predict_calibration_target_accuracy));
  logger->set(object_hp_hot_threshold, hp_mul10000(stats.hot_threshold));
  logger->set(object_hp_otsu_candidate_threshold,
              hp_mul10000(stats.otsu_candidate_threshold));
  logger->set(object_hp_otsu_separation,
              hp_mul10000(stats.otsu_separation));
  logger->set(object_hp_otsu_confidence,
              hp_mul10000(stats.otsu_confidence));
  logger->set(object_hp_otsu_sample_confidence,
              hp_mul10000(stats.otsu_sample_confidence));
  logger->set(object_hp_otsu_sharpness_confidence,
              hp_mul10000(stats.otsu_sharpness_confidence));
  logger->set(object_hp_hot_threshold_method, stats.hot_threshold_method);
  logger->set(object_hp_train_queue_length, osd_object_heat_predictor.get_train_queue_length());
  logger->set(object_hp_train_drop_count, osd_object_heat_predictor.get_train_drop_count());
  logger->set(object_hp_snapshot_publish_count,
              osd_object_heat_predictor.get_snapshot_publish_count());
  logger->set(object_hp_op_read_count, osd_object_hp_op_counters.read.load(std::memory_order_relaxed));
  logger->set(object_hp_op_sync_read_count, osd_object_hp_op_counters.sync_read.load(std::memory_order_relaxed));
  logger->set(object_hp_op_sparse_read_count, osd_object_hp_op_counters.sparse_read.load(std::memory_order_relaxed));
  logger->set(object_hp_op_write_count, osd_object_hp_op_counters.write.load(std::memory_order_relaxed));
  logger->set(object_hp_op_writefull_count, osd_object_hp_op_counters.writefull.load(std::memory_order_relaxed));
  logger->set(object_hp_op_writesame_count, osd_object_hp_op_counters.writesame.load(std::memory_order_relaxed));
  logger->tinc(object_hp_predict_latency, predict_latency);
}

static void hp_zero_object_logger()
{
  PerfCounters *logger = osd_object_hp_logger;
  if (logger == nullptr) {
    return;
  }

  logger->reset();
  logger->set(object_hp_enabled,
              osd_object_heat_predictor.is_enabled() ? 1 : 0);
  logger->set(object_hp_io_count, 0);
  logger->set(object_hp_labeled_io_total, 0);
  logger->set(object_hp_pending_io_count, 0);
  logger->set(object_hp_heat_state_count, 0);
  logger->set(object_hp_lru_count, 0);
  logger->set(object_hp_otsu_histogram_bin_count, 0);
  logger->set(object_hp_otsu_histogram_object_count, 0);
  logger->set(object_hp_true_positive_count, 0);
  logger->set(object_hp_false_positive_count, 0);
  logger->set(object_hp_true_negative_count, 0);
  logger->set(object_hp_false_negative_count, 0);
  logger->set(object_hp_actual_hot_object_avg_future_access_count, 0);
  logger->set(object_hp_actual_cold_object_avg_future_access_count, 0);
  logger->set(object_hp_actual_hot_object_avg_heat, 0);
  logger->set(object_hp_actual_cold_object_avg_heat, 0);
  hp_set_distribution_summary(
    logger,
    object_hp_actual_hot_future_access_p99,
    object_hp_actual_hot_future_access_p95,
    object_hp_actual_hot_future_access_p50,
    {});
  hp_set_distribution_summary(
    logger,
    object_hp_actual_cold_future_access_p99,
    object_hp_actual_cold_future_access_p95,
    object_hp_actual_cold_future_access_p50,
    {});
  hp_set_distribution_summary(
    logger,
    object_hp_actual_hot_future_heat_p99,
    object_hp_actual_hot_future_heat_p95,
    object_hp_actual_hot_future_heat_p50,
    {});
  hp_set_distribution_summary(
    logger,
    object_hp_actual_cold_future_heat_p99,
    object_hp_actual_cold_future_heat_p95,
    object_hp_actual_cold_future_heat_p50,
    {});
  logger->set(object_hp_hot_accuracy, 0);
  logger->set(object_hp_hot_precision, 0);
  logger->set(object_hp_hot_recall, 0);
  logger->set(object_hp_eval_pred_hot_percent, 0);
  logger->set(object_hp_eval_actual_hot_percent, 0);
  logger->set(object_hp_actual_hot_avg_pred_hot_percent, 0);
  logger->set(object_hp_actual_cold_avg_pred_hot_percent, 0);
  logger->set(object_hp_hot_predict_threshold,
              hp_mul10000(HP_HOT_PREDICT_THRESHOLD));
  logger->set(object_hp_hot_predict_threshold_target,
              hp_mul10000(HP_HOT_PREDICT_THRESHOLD));
  logger->set(object_hp_predict_calibration_sample_count, 0);
  logger->set(object_hp_predict_calibration_current_accuracy, 0);
  logger->set(object_hp_predict_calibration_target_accuracy, 0);
  logger->set(object_hp_hot_threshold, hp_mul10000(HP_HEAT_INCREMENT));
  logger->set(object_hp_otsu_candidate_threshold, 0);
  logger->set(object_hp_otsu_separation, 0);
  logger->set(object_hp_otsu_confidence, 0);
  logger->set(object_hp_otsu_sample_confidence, 0);
  logger->set(object_hp_otsu_sharpness_confidence, 0);
  logger->set(object_hp_hot_threshold_method,
              HP_THRESHOLD_METHOD_INITIALIZING);
  logger->set(object_hp_train_queue_length, 0);
  logger->set(object_hp_train_drop_count, 0);
  logger->set(object_hp_snapshot_publish_count, 0);
  logger->set(object_hp_op_read_count, 0);
  logger->set(object_hp_op_sync_read_count, 0);
  logger->set(object_hp_op_sparse_read_count, 0);
  logger->set(object_hp_op_write_count, 0);
  logger->set(object_hp_op_writefull_count, 0);
  logger->set(object_hp_op_writesame_count, 0);
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

} // namespace

void init_osd_object_hp_status(CephContext *cct)
{
  hp_ensure_object_logger(cct);
  hp_zero_object_logger();
}

void hp_dump_osd_object_heat_predictor_status(CephContext *cct,
                                              ceph::Formatter *f)
{
  std::shared_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);

  hp_ensure_object_logger(cct);
  f->open_object_section("object_hp_status");
  f->dump_bool("enabled", osd_object_heat_predictor.is_enabled());
  f->dump_unsigned("hp_io_count", osd_object_heat_predictor.hp_index.load());
  f->dump_unsigned("hp_labeled_io_total",
                   osd_object_heat_predictor.get_total_weight());
  f->dump_unsigned("hp_pending_io_count",
                   osd_object_heat_predictor.get_pending_io_count());
  f->dump_unsigned("hp_train_queue_length",
                   osd_object_heat_predictor.get_train_queue_length());
  f->dump_unsigned("hp_train_drop_count",
                   osd_object_heat_predictor.get_train_drop_count());
  f->dump_unsigned("hp_snapshot_publish_count",
                   osd_object_heat_predictor.get_snapshot_publish_count());
  f->close_section();
}

void hp_reset_osd_object_heat_predictor(CephContext *cct, ceph::Formatter *f)
{
  std::unique_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);

  hp_ensure_object_logger(cct);
  uint64_t discarded_pending_io = osd_object_heat_predictor.reset();
  hp_reset_osd_op_counters();
  hp_zero_object_logger();

  if (f != nullptr) {
    f->open_object_section("object_hp_reset");
    f->dump_bool("ok", true);
    f->dump_bool("enabled", osd_object_heat_predictor.is_enabled());
    f->dump_unsigned("discarded_pending_io", discarded_pending_io);
    f->dump_unsigned("hp_io_count", osd_object_heat_predictor.hp_index.load());
    f->dump_unsigned("hp_labeled_io_total", osd_object_heat_predictor.get_total_weight());
    f->dump_unsigned("hp_pending_io_count", osd_object_heat_predictor.get_pending_io_count());
    f->dump_unsigned("hp_heat_state_count", osd_object_heat_predictor.get_heat_state_count());
    f->dump_unsigned("hp_lru_count", osd_object_heat_predictor.get_lru_count());
    f->dump_unsigned("hp_otsu_histogram_bin_count",
                     osd_object_heat_predictor.get_otsu_histogram_bin_count());
    f->dump_unsigned("hp_otsu_histogram_object_count",
                     osd_object_heat_predictor.get_otsu_histogram_object_count());
    f->dump_unsigned("hp_train_queue_length", osd_object_heat_predictor.get_train_queue_length());
    f->dump_unsigned("hp_train_drop_count", osd_object_heat_predictor.get_train_drop_count());
    f->dump_unsigned("hp_snapshot_publish_count",
                     osd_object_heat_predictor.get_snapshot_publish_count());
    f->close_section();
  }
}

void hp_set_osd_object_heat_predictor_enabled(CephContext *cct,
                                              ceph::Formatter *f,
                                              bool enabled)
{
  std::unique_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);

  hp_ensure_object_logger(cct);
  uint64_t discarded_pending_io =
    osd_object_heat_predictor.set_enabled(enabled);
  hp_reset_osd_op_counters();
  hp_zero_object_logger();

  if (f != nullptr) {
    f->open_object_section(enabled ? "object_hp_enable" : "object_hp_disable");
    f->dump_bool("ok", true);
    f->dump_bool("enabled", osd_object_heat_predictor.is_enabled());
    f->dump_unsigned("discarded_pending_io", discarded_pending_io);
    f->dump_unsigned("hp_io_count", osd_object_heat_predictor.hp_index.load());
    f->dump_unsigned("hp_labeled_io_total",
                     osd_object_heat_predictor.get_total_weight());
    f->dump_unsigned("hp_pending_io_count",
                     osd_object_heat_predictor.get_pending_io_count());
    f->dump_unsigned("hp_heat_state_count",
                     osd_object_heat_predictor.get_heat_state_count());
    f->dump_unsigned("hp_lru_count",
                     osd_object_heat_predictor.get_lru_count());
    f->dump_unsigned("hp_otsu_histogram_bin_count",
                     osd_object_heat_predictor.get_otsu_histogram_bin_count());
    f->dump_unsigned("hp_otsu_histogram_object_count",
                     osd_object_heat_predictor.get_otsu_histogram_object_count());
    f->dump_unsigned("hp_train_queue_length",
                     osd_object_heat_predictor.get_train_queue_length());
    f->dump_unsigned("hp_train_drop_count",
                     osd_object_heat_predictor.get_train_drop_count());
    f->dump_unsigned("hp_snapshot_publish_count",
                     osd_object_heat_predictor.get_snapshot_publish_count());
    f->close_section();
  }
}

void hp_notify_osd_object_op(CephContext *cct,
                             const hobject_t& soid,
                             uint16_t op)
{
  if (!osd_object_heat_predictor.is_enabled()) {
    return;
  }

  if (!hp_track_osd_op(op)) {
    return;
  }

  std::shared_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);
  hp_count_osd_op(op);
  hp_ensure_object_logger(cct);
  auto start_time = ceph::mono_clock::now();
  uint64_t index = 0;
  osd_object_heat_predictor.predict(
    soid.pool,
    soid.get_hash(),
    std::hash<object_t>{}(soid.oid),
    &index);
  auto end_time = ceph::mono_clock::now();
  if (hp_should_update_object_logger(index)) {
    hp_update_object_logger(end_time - start_time);
  }
}
