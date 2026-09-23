// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "ObjectHeatPredictor.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>

#include "common/ceph_context.h"
#include "common/admin_socket.h"
#include "common/ceph_time.h"
#include "common/Formatter.h"
#include "common/hobject.h"
#include "common/perf_counters.h"
#include "common/version.h"
#include "include/rados.h"
#include "heatpredictor/heat_predictor.h"
#include "heatpredictor/hp_telemetry.h"

namespace hp_field = ceph::hp_telemetry::field;

struct ObjectHeatPredictor::Impl {
  CephContext* context = nullptr;
  std::shared_ptr<std::atomic<bool>> observation_gate;

  HeatPredictor osd_object_heat_predictor{
    [this](uint64_t count) { hp_record_object_expiry_progress(count); },
    [this] { hp_record_object_background_error(); }};

  ~Impl() {
    if (observation_gate) observation_gate->store(false, std::memory_order_release);
    // Do not hold reset/logger locks while joining callback threads.
    osd_object_heat_predictor.shutdown();
    if (osd_object_hp_logger) {
      context->get_perfcounters_collection()->remove(osd_object_hp_logger);
      delete osd_object_hp_logger;
    }
  }
  PerfCounters *osd_object_hp_logger = nullptr;
  std::mutex osd_object_hp_logger_mtx;
  std::shared_mutex osd_object_hp_reset_mtx;
  static constexpr uint64_t object_hp_logger_update_interval = 1000;
  uint64_t object_hp_expiry_since_logger_update = 0;
  uint64_t osd_object_hp_publish_generation = 0;
  int osd_object_hp_osd_id = -1;
  std::string osd_object_hp_trace_phase;
  std::string osd_object_hp_trace_directory = "/var/log/ceph";

  enum {
    object_hp_first = 591422,
    // PerfCounters are dumped in enum order; these markers bracket one
    // publication of all ordinary status fields.
    object_hp_status_publish_generation_begin,
    object_hp_enabled,
    object_hp_io_count,
    object_hp_labeled_io_total,
    object_hp_pending_io_count,
    object_hp_awaiting_prediction_count,
    object_hp_eval_drop_count,
    object_hp_heat_state_count,
    object_hp_lru_count,
    object_hp_protected_heat_state_count,
    object_hp_heat_state_peak_count,
    object_hp_lru_eviction_count,
    object_hp_otsu_histogram_bin_count,
    object_hp_future_access_threshold,
    object_hp_threshold_state,
    object_hp_otsu_positive_object_count,
    object_hp_otsu_zero_observation_count,
    object_hp_otsu_upper_clamped_object_count,
    object_hp_sparse_threshold_sample_count,
    object_hp_true_positive_count,
    object_hp_false_positive_count,
    object_hp_true_negative_count,
    object_hp_false_negative_count,
    object_hp_hot_labeled_sample_avg_future_access_count,
    object_hp_cold_labeled_sample_avg_future_access_count,
    object_hp_hot_labeled_sample_future_access_count_p99,
    object_hp_hot_labeled_sample_future_access_count_p95,
    object_hp_hot_labeled_sample_future_access_count_p50,
    object_hp_cold_labeled_sample_future_access_count_p99,
    object_hp_cold_labeled_sample_future_access_count_p95,
    object_hp_cold_labeled_sample_future_access_count_p50,
    object_hp_hot_accuracy,
    object_hp_hot_balanced_accuracy,
    object_hp_hot_precision,
    object_hp_hot_recall,
    object_hp_eval_pred_hot_percent,
    object_hp_eval_actual_hot_percent,
    object_hp_actual_hot_avg_pred_hot_percent,
    object_hp_actual_cold_avg_pred_hot_percent,
    object_hp_predict_error_count,
    object_hp_background_error_count,
    object_hp_train_queue_length,
    object_hp_train_drop_count,
    object_hp_snapshot_publish_count,
    object_hp_arf_warning_count,
    object_hp_arf_drift_count,
    object_hp_arf_background_promotion_count,
    object_hp_arf_background_discard_count,
    object_hp_arf_background_training_update_count,
    object_hp_arf_active_background_count,
    object_hp_trace_enabled,
    object_hp_trace_queue_length,
    object_hp_trace_written_count,
    object_hp_trace_drop_count,
    object_hp_trace_write_error_count,
    object_hp_op_read_count,
    object_hp_op_write_count,
    object_hp_status_publish_generation_end,
    object_hp_predict_latency,
    object_hp_last
  };

  struct ObjectHpOpCounters {
    std::atomic<uint64_t> read{0};
    std::atomic<uint64_t> write{0};
  };

  ObjectHpOpCounters osd_object_hp_op_counters;

  void hp_reset_osd_op_counters()
  {
    osd_object_hp_op_counters.read.store(0, std::memory_order_relaxed);
    osd_object_hp_op_counters.write.store(0, std::memory_order_relaxed);
  }

  uint64_t hp_mul10000(double x)
  {
    if (x <= 0) {
      return 0;
    }
    return static_cast<uint64_t>(x * 10000);
  }

  uint64_t hp_ratio10000(uint64_t numerator, uint64_t denominator)
  {
    if (denominator == 0 || numerator == 0) {
      return 0;
    }
    return static_cast<uint64_t>(
      (static_cast<long double>(numerator) * 10000) / denominator);
  }

  uint64_t hp_avg10000(double sum, uint64_t count)
  {
    if (count == 0 || sum <= 0) {
      return 0;
    }
    return static_cast<uint64_t>(
      (static_cast<long double>(sum) * 10000) / count);
  }

  void hp_set_distribution_summary(
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

  // Called once by init, before control commands can enable the predictor.
  void hp_create_object_logger(CephContext *cct)
  {
    ceph_assert(cct && osd_object_hp_logger == nullptr);

    PerfCountersBuilder b(cct, "object_hp_status", object_hp_first, object_hp_last);
    b.set_prio_default(PerfCountersBuilder::PRIO_USEFUL);
    b.add_u64(object_hp_status_publish_generation_begin,
              hp_field::status_publish_generation_begin,
              "object heat predictor status publication generation begin");
    b.add_u64(object_hp_enabled, hp_field::enabled, "heat predictor enabled");
    b.add_u64(object_hp_io_count, hp_field::io_count, "predicted I/O total");
    b.add_u64(object_hp_labeled_io_total, hp_field::labeled_io_total, "evaluated I/O total");
    b.add_u64(object_hp_pending_io_count, hp_field::pending_io_count, "pending evaluation I/O count");
    b.add_u64(object_hp_awaiting_prediction_count,
              hp_field::awaiting_prediction_count,
              "label-complete samples awaiting prediction completion");
    b.add_u64(object_hp_eval_drop_count,
              hp_field::eval_drop_count,
              "I/O evaluations dropped by capacity or invalid prediction");
    b.add_u64(object_hp_heat_state_count, hp_field::heat_state_count, "tracked object heat state count");
    b.add_u64(object_hp_lru_count, hp_field::lru_count, "object heat states in LRU");
    b.add_u64(object_hp_protected_heat_state_count,
              hp_field::protected_heat_state_count,
              "object heat states protected from LRU eviction");
    b.add_u64(object_hp_heat_state_peak_count,
              hp_field::heat_state_peak_count,
              "peak tracked object heat state count since reset");
    b.add_u64(object_hp_lru_eviction_count,
              hp_field::lru_eviction_count,
              "object heat states evicted by the LRU since reset");
    b.add_u64(object_hp_otsu_histogram_bin_count,
              hp_field::otsu_histogram_bin_count,
              "occupied Otsu histogram bin count");
    b.add_u64(object_hp_future_access_threshold,
              hp_field::future_access_threshold,
              "current future-window access threshold K");
    b.add_u64(object_hp_threshold_state,
              hp_field::threshold_state,
              "threshold state: 0 sparse, 1 tracking");
    b.add_u64(object_hp_otsu_positive_object_count,
              hp_field::otsu_positive_object_count,
              "objects with a retained positive future-access vote");
    b.add_u64(object_hp_otsu_zero_observation_count,
              hp_field::otsu_zero_observation_count,
              "object recent-access counts that reached zero since reset");
    b.add_u64(object_hp_otsu_upper_clamped_object_count,
              hp_field::otsu_upper_clamped_object_count,
              "object votes clamped into the final Otsu bin");
    b.add_u64(object_hp_sparse_threshold_sample_count,
              hp_field::sparse_threshold_sample_count,
              "samples enqueued with sparse fallback K=1");
    b.add_u64(object_hp_true_positive_count, hp_field::true_positive_count, "true positive count");
    b.add_u64(object_hp_false_positive_count, hp_field::false_positive_count, "false positive count");
    b.add_u64(object_hp_true_negative_count, hp_field::true_negative_count, "true negative count");
    b.add_u64(object_hp_false_negative_count, hp_field::false_negative_count, "false negative count");
    b.add_u64(object_hp_hot_labeled_sample_avg_future_access_count,
              hp_field::hot_labeled_sample_avg_future_access_count,
              "average future-window access count of hot-labeled samples (x10000)");
    b.add_u64(object_hp_cold_labeled_sample_avg_future_access_count,
              hp_field::cold_labeled_sample_avg_future_access_count,
              "average future-window access count of cold-labeled samples (x10000)");
    b.add_u64(object_hp_hot_labeled_sample_future_access_count_p99,
              hp_field::hot_labeled_sample_future_access_count_p99,
              "p99 future-window access count of hot-labeled samples (x10000)");
    b.add_u64(object_hp_hot_labeled_sample_future_access_count_p95,
              hp_field::hot_labeled_sample_future_access_count_p95,
              "p95 future-window access count of hot-labeled samples (x10000)");
    b.add_u64(object_hp_hot_labeled_sample_future_access_count_p50,
              hp_field::hot_labeled_sample_future_access_count_p50,
              "p50 future-window access count of hot-labeled samples (x10000)");
    b.add_u64(object_hp_cold_labeled_sample_future_access_count_p99,
              hp_field::cold_labeled_sample_future_access_count_p99,
              "p99 future-window access count of cold-labeled samples (x10000)");
    b.add_u64(object_hp_cold_labeled_sample_future_access_count_p95,
              hp_field::cold_labeled_sample_future_access_count_p95,
              "p95 future-window access count of cold-labeled samples (x10000)");
    b.add_u64(object_hp_cold_labeled_sample_future_access_count_p50,
              hp_field::cold_labeled_sample_future_access_count_p50,
              "p50 future-window access count of cold-labeled samples (x10000)");
    b.add_u64(object_hp_hot_accuracy,
              hp_field::hot_accuracy,
              "hot prediction accuracy (x10000)");
    b.add_u64(object_hp_hot_balanced_accuracy,
              hp_field::hot_balanced_accuracy,
              "balanced hot/cold prediction accuracy (x10000)");
    b.add_u64(object_hp_hot_precision,
              hp_field::hot_precision,
              "hot prediction precision (x10000)");
    b.add_u64(object_hp_hot_recall,
              hp_field::hot_recall,
              "hot prediction recall (x10000)");
    b.add_u64(object_hp_eval_pred_hot_percent,
              hp_field::eval_pred_hot_percent,
              "evaluated predicted-hot percentage (x10000)");
    b.add_u64(object_hp_eval_actual_hot_percent,
              hp_field::eval_actual_hot_percent,
              "evaluated actual-hot percentage (x10000)");
    b.add_u64(object_hp_actual_hot_avg_pred_hot_percent,
              hp_field::actual_hot_avg_pred_hot_percent,
              "average predicted hot probability of actual hot objects (x10000)");
    b.add_u64(object_hp_actual_cold_avg_pred_hot_percent,
              hp_field::actual_cold_avg_pred_hot_percent,
              "average predicted hot probability of actual cold objects (x10000)");
    b.add_u64(object_hp_predict_error_count,
              hp_field::predict_error_count,
              "prediction exceptions or invalid probability outputs");
    b.add_u64(object_hp_background_error_count,
              hp_field::background_error_count,
              "background training or expiry worker exceptions");
    b.add_u64(object_hp_train_queue_length, hp_field::train_queue_length, "train queue length");
    b.add_u64(object_hp_train_drop_count, hp_field::train_drop_count, "dropped training sample count");
    b.add_u64(object_hp_snapshot_publish_count, hp_field::snapshot_publish_count, "prediction snapshot publish count");
    b.add_u64(object_hp_arf_warning_count,
              hp_field::arf_warning_count,
              "ARF warning detections");
    b.add_u64(object_hp_arf_drift_count,
              hp_field::arf_drift_count,
              "ARF drift detections and current-tree replacements");
    b.add_u64(object_hp_arf_background_promotion_count,
              hp_field::arf_background_promotion_count,
              "ARF background trees promoted after drift");
    b.add_u64(object_hp_arf_background_discard_count,
              hp_field::arf_background_discard_count,
              "ARF background trees discarded by later warnings");
    b.add_u64(object_hp_arf_background_training_update_count,
              hp_field::arf_background_training_update_count,
              "ARF training updates applied to background trees");
    b.add_u64(object_hp_arf_active_background_count,
              hp_field::arf_active_background_count,
              "currently active ARF background trees");
    b.add_u64(object_hp_trace_enabled,
              hp_field::trace_enabled,
              "completed-evaluation trace enabled");
    b.add_u64(object_hp_trace_queue_length,
              hp_field::trace_queue_length,
              "trace records awaiting disk write");
    b.add_u64(object_hp_trace_written_count,
              hp_field::trace_written_count,
              "trace records written");
    b.add_u64(object_hp_trace_drop_count,
              hp_field::trace_drop_count,
              "trace records dropped by queue pressure");
    b.add_u64(object_hp_trace_write_error_count,
              hp_field::trace_write_error_count,
              "trace records or sessions lost to write errors");
    b.add_u64(object_hp_op_read_count, hp_field::op_read_count, "read op count");
    b.add_u64(object_hp_op_write_count, hp_field::op_write_count, "write op count");
    b.add_u64(object_hp_status_publish_generation_end,
              hp_field::status_publish_generation_end,
              "object heat predictor status publication generation end");
    b.add_time_avg(object_hp_predict_latency, hp_field::predict_latency, "predict latency");
    osd_object_hp_logger = b.create_perf_counters();
    cct->get_perfcounters_collection()->add(osd_object_hp_logger);
  }

  void hp_set_trace_logger(PerfCounters *logger,
                                  const HpTraceStatus& status)
  {
    logger->set(object_hp_trace_enabled, status.enabled ? 1 : 0);
    logger->set(object_hp_trace_queue_length, status.queue_length);
    logger->set(object_hp_trace_written_count, status.written_count);
    logger->set(object_hp_trace_drop_count, status.drop_count);
    logger->set(object_hp_trace_write_error_count,
                status.write_error_count);
  }

  void hp_update_object_logger_from_status(
    PerfCounters *logger,
    const HeatPredictorStatus& predictor_status,
    ceph::timespan predict_latency,
    bool record_predict_latency)
  {
    const auto& stats = predictor_status.evaluation;
    uint64_t io_count = stats.io_count;
    uint64_t labeled_io_total = stats.labeled_io_total;
    uint64_t true_positive = stats.true_positive;
    uint64_t false_positive = stats.false_positive;
    uint64_t true_negative = stats.true_negative;
    uint64_t false_negative = stats.false_negative;
    uint64_t actual_hot_count = true_positive + false_negative;
    uint64_t actual_cold_count = true_negative + false_positive;
    uint64_t hot_labeled_sample_avg_future_access_count =
      hp_ratio10000(stats.hot_labeled_sample_future_access_count_sum, actual_hot_count);
    uint64_t cold_labeled_sample_avg_future_access_count =
      hp_ratio10000(stats.cold_labeled_sample_future_access_count_sum, actual_cold_count);
    uint64_t hot_accuracy =
      hp_ratio10000(true_positive + true_negative, labeled_io_total);
    uint64_t hot_balanced_accuracy = hp_mul10000(
      hp_binary_balanced_accuracy(
        true_positive, false_positive, true_negative, false_negative));
    uint64_t hot_precision =
      hp_ratio10000(true_positive, true_positive + false_positive);
    uint64_t hot_recall =
      hp_ratio10000(true_positive, true_positive + false_negative);
    uint64_t eval_pred_hot_percent =
      hp_ratio10000(true_positive + false_positive, labeled_io_total);
    uint64_t eval_actual_hot_percent =
      hp_ratio10000(actual_hot_count, labeled_io_total);
    uint64_t actual_hot_avg_pred_hot_percent =
      hp_avg10000(stats.hot_labeled_sample_predicted_hot_probability_sum, actual_hot_count);
    uint64_t actual_cold_avg_pred_hot_percent =
      hp_avg10000(stats.cold_labeled_sample_predicted_hot_probability_sum, actual_cold_count);

    std::unique_lock<std::mutex> publish_lock(osd_object_hp_logger_mtx);
    if (++osd_object_hp_publish_generation == 0) {
      ++osd_object_hp_publish_generation;
    }
    const uint64_t publish_generation =
      osd_object_hp_publish_generation;
    // The MGR rejects a dump unless it observes the same generation at both
    // ends, so no partially published field group enters cluster aggregation.
    logger->set(
      object_hp_status_publish_generation_begin, publish_generation);
    logger->set(object_hp_enabled, stats.enabled ? 1 : 0);
    logger->set(object_hp_io_count, io_count);
    logger->set(object_hp_labeled_io_total, labeled_io_total);
    logger->set(object_hp_pending_io_count, stats.pending_io_count);
    logger->set(object_hp_awaiting_prediction_count,
                stats.awaiting_prediction_count);
    logger->set(object_hp_eval_drop_count, stats.eval_drop_count);
    logger->set(object_hp_heat_state_count, stats.heat_state_count);
    logger->set(object_hp_lru_count, stats.lru_count);
    logger->set(object_hp_protected_heat_state_count,
                stats.protected_heat_state_count);
    logger->set(object_hp_heat_state_peak_count,
                stats.heat_state_peak_count);
    logger->set(object_hp_lru_eviction_count, stats.lru_eviction_count);
    logger->set(object_hp_otsu_histogram_bin_count,
                stats.otsu_histogram_bin_count);
    logger->set(object_hp_future_access_threshold,
                stats.future_access_threshold);
    logger->set(object_hp_threshold_state, stats.threshold_state);
    logger->set(object_hp_otsu_positive_object_count,
                stats.otsu_positive_object_count);
    logger->set(object_hp_otsu_zero_observation_count,
                stats.otsu_zero_observation_count);
    logger->set(object_hp_otsu_upper_clamped_object_count,
                stats.otsu_upper_clamped_object_count);
    logger->set(object_hp_sparse_threshold_sample_count,
                stats.sparse_threshold_sample_count);
    logger->set(object_hp_true_positive_count, true_positive);
    logger->set(object_hp_false_positive_count, false_positive);
    logger->set(object_hp_true_negative_count, true_negative);
    logger->set(object_hp_false_negative_count, false_negative);
    logger->set(object_hp_hot_labeled_sample_avg_future_access_count,
                hot_labeled_sample_avg_future_access_count);
    logger->set(object_hp_cold_labeled_sample_avg_future_access_count,
                cold_labeled_sample_avg_future_access_count);
    hp_set_distribution_summary(
      logger,
      object_hp_hot_labeled_sample_future_access_count_p99,
      object_hp_hot_labeled_sample_future_access_count_p95,
      object_hp_hot_labeled_sample_future_access_count_p50,
      stats.hot_labeled_sample_future_access_count);
    hp_set_distribution_summary(
      logger,
      object_hp_cold_labeled_sample_future_access_count_p99,
      object_hp_cold_labeled_sample_future_access_count_p95,
      object_hp_cold_labeled_sample_future_access_count_p50,
      stats.cold_labeled_sample_future_access_count);
    logger->set(object_hp_hot_accuracy, hot_accuracy);
    logger->set(object_hp_hot_balanced_accuracy, hot_balanced_accuracy);
    logger->set(object_hp_hot_precision, hot_precision);
    logger->set(object_hp_hot_recall, hot_recall);
    logger->set(object_hp_eval_pred_hot_percent, eval_pred_hot_percent);
    logger->set(object_hp_eval_actual_hot_percent, eval_actual_hot_percent);
    logger->set(object_hp_actual_hot_avg_pred_hot_percent,
                actual_hot_avg_pred_hot_percent);
    logger->set(object_hp_actual_cold_avg_pred_hot_percent,
                actual_cold_avg_pred_hot_percent);
    logger->set(object_hp_predict_error_count,
                predictor_status.predict_error_count);
    logger->set(object_hp_background_error_count,
                predictor_status.background_error_count);
    logger->set(object_hp_train_queue_length,
                predictor_status.train_queue_length);
    logger->set(object_hp_train_drop_count,
                predictor_status.train_drop_count);
    logger->set(object_hp_snapshot_publish_count,
                predictor_status.snapshot_publish_count);
    const ArfAdaptationStats adaptation_stats =
      predictor_status.arf_adaptation;
    logger->set(object_hp_arf_warning_count,
                adaptation_stats.warning_count);
    logger->set(object_hp_arf_drift_count,
                adaptation_stats.drift_count);
    logger->set(object_hp_arf_background_promotion_count,
                adaptation_stats.background_promotion_count);
    logger->set(object_hp_arf_background_discard_count,
                adaptation_stats.background_discard_count);
    logger->set(object_hp_arf_background_training_update_count,
                adaptation_stats.background_training_update_count);
    logger->set(object_hp_arf_active_background_count,
                adaptation_stats.active_background_count);
    hp_set_trace_logger(logger, predictor_status.trace);
    logger->set(object_hp_op_read_count, osd_object_hp_op_counters.read.load(std::memory_order_relaxed));
    logger->set(object_hp_op_write_count, osd_object_hp_op_counters.write.load(std::memory_order_relaxed));
    logger->set(
      object_hp_status_publish_generation_end, publish_generation);
    publish_lock.unlock();
    if (record_predict_latency) {
      logger->tinc(object_hp_predict_latency, predict_latency);
    }
  }

  void hp_update_object_logger(ceph::timespan predict_latency,
                                      bool record_predict_latency = true)
  {
    PerfCounters *logger = osd_object_hp_logger;
    if (logger == nullptr) {
      return;
    }
    hp_update_object_logger_from_status(
      logger,
      osd_object_heat_predictor.status(),
      predict_latency,
      record_predict_latency);
  }

  void hp_record_object_predict_latency(ceph::timespan predict_latency)
  {
    PerfCounters *logger = osd_object_hp_logger;
    if (logger != nullptr) {
      logger->tinc(object_hp_predict_latency, predict_latency);
    }
  }

  void hp_record_object_expiry_progress(uint64_t expired_count)
  {
    std::shared_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);
    if (expired_count == 0) {
      hp_update_object_logger(ceph::timespan::zero(), false);
      return;
    }
    object_hp_expiry_since_logger_update += expired_count;
    if (object_hp_expiry_since_logger_update <
        object_hp_logger_update_interval) {
      return;
    }

    object_hp_expiry_since_logger_update %= object_hp_logger_update_interval;
    hp_update_object_logger(ceph::timespan::zero(), false);
  }

  void hp_record_object_background_error()
  {
    std::shared_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);
    // Serialize gate publication with commands; a late error callback must
    // observe a subsequently re-enabled core rather than close its gate.
    if (observation_gate)
      observation_gate->store(osd_object_heat_predictor.is_enabled(),
                              std::memory_order_release);
    hp_update_object_logger(ceph::timespan::zero(), false);
  }

  void hp_zero_object_logger()
  {
    std::lock_guard<std::mutex> publish_lock(osd_object_hp_logger_mtx);
    PerfCounters *logger = osd_object_hp_logger;
    if (logger == nullptr) {
      return;
    }

    if (++osd_object_hp_publish_generation == 0) {
      ++osd_object_hp_publish_generation;
    }
    const uint64_t publish_generation =
      osd_object_hp_publish_generation;
    logger->reset();
    logger->set(
      object_hp_status_publish_generation_begin, publish_generation);
    object_hp_expiry_since_logger_update = 0;
    logger->set(object_hp_enabled,
                osd_object_heat_predictor.is_enabled() ? 1 : 0);
    logger->set(object_hp_io_count, 0);
    logger->set(object_hp_labeled_io_total, 0);
    logger->set(object_hp_pending_io_count, 0);
    logger->set(object_hp_awaiting_prediction_count, 0);
    logger->set(object_hp_eval_drop_count, 0);
    logger->set(object_hp_heat_state_count, 0);
    logger->set(object_hp_lru_count, 0);
    logger->set(object_hp_protected_heat_state_count, 0);
    logger->set(object_hp_heat_state_peak_count, 0);
    logger->set(object_hp_lru_eviction_count, 0);
    logger->set(object_hp_otsu_histogram_bin_count, 0);
    logger->set(object_hp_future_access_threshold, 1);
    logger->set(object_hp_threshold_state,
                static_cast<uint64_t>(HpThresholdState::sparse));
    logger->set(object_hp_otsu_positive_object_count, 0);
    logger->set(object_hp_otsu_zero_observation_count, 0);
    logger->set(object_hp_otsu_upper_clamped_object_count, 0);
    logger->set(object_hp_sparse_threshold_sample_count, 0);
    logger->set(object_hp_true_positive_count, 0);
    logger->set(object_hp_false_positive_count, 0);
    logger->set(object_hp_true_negative_count, 0);
    logger->set(object_hp_false_negative_count, 0);
    logger->set(object_hp_hot_labeled_sample_avg_future_access_count, 0);
    logger->set(object_hp_cold_labeled_sample_avg_future_access_count, 0);
    hp_set_distribution_summary(
      logger,
      object_hp_hot_labeled_sample_future_access_count_p99,
      object_hp_hot_labeled_sample_future_access_count_p95,
      object_hp_hot_labeled_sample_future_access_count_p50,
      {});
    hp_set_distribution_summary(
      logger,
      object_hp_cold_labeled_sample_future_access_count_p99,
      object_hp_cold_labeled_sample_future_access_count_p95,
      object_hp_cold_labeled_sample_future_access_count_p50,
      {});
    logger->set(object_hp_hot_accuracy, 0);
    logger->set(object_hp_hot_balanced_accuracy, 0);
    logger->set(object_hp_hot_precision, 0);
    logger->set(object_hp_hot_recall, 0);
    logger->set(object_hp_eval_pred_hot_percent, 0);
    logger->set(object_hp_eval_actual_hot_percent, 0);
    logger->set(object_hp_actual_hot_avg_pred_hot_percent, 0);
    logger->set(object_hp_actual_cold_avg_pred_hot_percent, 0);
    logger->set(object_hp_predict_error_count, 0);
    logger->set(object_hp_background_error_count, 0);
    logger->set(object_hp_train_queue_length, 0);
    logger->set(object_hp_train_drop_count, 0);
    logger->set(object_hp_snapshot_publish_count, 0);
    logger->set(object_hp_arf_warning_count, 0);
    logger->set(object_hp_arf_drift_count, 0);
    logger->set(object_hp_arf_background_promotion_count, 0);
    logger->set(object_hp_arf_background_discard_count, 0);
    logger->set(object_hp_arf_background_training_update_count, 0);
    logger->set(object_hp_arf_active_background_count, 0);
    hp_set_trace_logger(
      logger, osd_object_heat_predictor.get_trace_status());
    logger->set(object_hp_op_read_count, 0);
    logger->set(object_hp_op_write_count, 0);
    logger->set(
      object_hp_status_publish_generation_end, publish_generation);
  }

  bool hp_should_update_object_logger(uint64_t index)
  {
    return (index % object_hp_logger_update_interval) == 0;
  }

  bool hp_track_osd_op(HpAccessType op) {
    return op == HpAccessType::Read || op == HpAccessType::Write;
  }
  void hp_count_osd_op(HpAccessType op) {
    auto& count = op == HpAccessType::Read
      ? osd_object_hp_op_counters.read : osd_object_hp_op_counters.write;
    count.fetch_add(1, std::memory_order_relaxed);
  }

  void init_osd_object_hp_status(CephContext *cct, int osd_id)
  {
    std::unique_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);
    context = cct;
    osd_object_hp_osd_id = osd_id;
    hp_create_object_logger(cct);
    hp_zero_object_logger();
  }

  void hp_dump_future_access_threshold(
    ceph::Formatter *f,
    const HeatPredictorStats& stats)
  {
    f->open_object_section("future_access_threshold");
    f->dump_unsigned("hp_future_access_threshold",
                     stats.future_access_threshold);
    f->dump_unsigned("hp_threshold_state", stats.threshold_state);
    f->dump_unsigned("hp_otsu_histogram_bin_count",
                     stats.otsu_histogram_bin_count);
    f->dump_unsigned("hp_otsu_positive_object_count",
                     stats.otsu_positive_object_count);
    f->dump_unsigned("hp_otsu_zero_observation_count",
                     stats.otsu_zero_observation_count);
    f->dump_unsigned("hp_otsu_upper_clamped_object_count",
                     stats.otsu_upper_clamped_object_count);
    f->dump_unsigned("hp_sparse_threshold_sample_count",
                     stats.sparse_threshold_sample_count);
    f->close_section();
  }

  void hp_dump_osd_object_heat_predictor_status(ceph::Formatter *f)
  {
    std::shared_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);

    const auto predictor_status = osd_object_heat_predictor.status();
    if (osd_object_hp_logger != nullptr) {
      hp_update_object_logger_from_status(
        osd_object_hp_logger,
        predictor_status,
        ceph::timespan::zero(),
        false);
    }
    const auto& stats = predictor_status.evaluation;
    f->open_object_section("object_hp_status");
    f->dump_bool("enabled", stats.enabled);
    f->dump_unsigned("hp_trained_sample_count", predictor_status.trained_sample_count);
    f->dump_unsigned("hp_snapshot_trained_sample_count", predictor_status.snapshot_trained_sample_count);
    f->dump_unsigned("hp_warmup_prediction_count", predictor_status.warmup_prediction_count);
    f->dump_unsigned("hp_warmup_trained_samples", HP_WARMUP_TRAINED_SAMPLES);
    f->dump_bool("hp_leaf_majority_only", HP_LEAF_MAJORITY_ONLY);
    f->dump_string("hp_feature_policy", "C4");
    f->dump_string("hp_observation_scope", "storage_object");
    f->dump_string("hp_observation_backend", "bluestore");
    f->dump_unsigned("hp_feature_count", NUM_FEATURES);
    f->dump_float("hp_slow_history_tau_30_seconds", HP_SLOW_HISTORY_TAU_SECONDS[0]);
    f->dump_float("hp_slow_history_tau_60_seconds", HP_SLOW_HISTORY_TAU_SECONDS[1]);
    f->dump_float("hp_slow_history_max_multiplier", HP_SLOW_HISTORY_MAX_MULTIPLIER);
    f->dump_unsigned("hp_snapshot_sample_interval", HP_SNAPSHOT_PUBLISH_SAMPLE_INTERVAL);
    f->dump_unsigned("hp_snapshot_max_interval_ns", HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS);
    f->dump_unsigned("hp_io_count", stats.io_count);
    f->dump_unsigned("hp_labeled_io_total", stats.labeled_io_total);
    f->dump_unsigned("hp_pending_io_count", stats.pending_io_count);
    f->dump_unsigned(
      "hp_awaiting_prediction_count",
      stats.awaiting_prediction_count);
    f->dump_unsigned("hp_eval_drop_count", stats.eval_drop_count);
    hp_dump_future_access_threshold(f, stats);
    f->dump_unsigned("hp_train_queue_length",
                     predictor_status.train_queue_length);
    f->dump_unsigned("hp_train_drop_count",
                     predictor_status.train_drop_count);
    f->dump_unsigned("hp_snapshot_publish_count",
                     predictor_status.snapshot_publish_count);
    const ArfAdaptationStats adaptation_stats =
      predictor_status.arf_adaptation;
    f->open_object_section("model_adaptation");
    f->dump_unsigned("hp_arf_warning_count",
                     adaptation_stats.warning_count);
    f->dump_unsigned("hp_arf_drift_count",
                     adaptation_stats.drift_count);
    f->dump_unsigned("hp_arf_background_promotion_count",
                     adaptation_stats.background_promotion_count);
    f->dump_unsigned("hp_arf_background_discard_count",
                     adaptation_stats.background_discard_count);
    f->dump_unsigned("hp_arf_background_training_update_count",
                     adaptation_stats.background_training_update_count);
    f->dump_unsigned("hp_arf_active_background_count",
                     adaptation_stats.active_background_count);
    f->close_section();
    f->dump_unsigned("hp_predict_error_count",
                     predictor_status.predict_error_count);
    f->dump_unsigned("hp_background_error_count",
                     predictor_status.background_error_count);
    const auto& trace_status = predictor_status.trace;
    f->open_object_section("trace");
    f->dump_bool("enabled", trace_status.enabled);
    f->dump_unsigned("session_id", trace_status.session_id);
    f->dump_unsigned("queue_length", trace_status.queue_length);
    f->dump_unsigned("written_count", trace_status.written_count);
    f->dump_unsigned("drop_count", trace_status.drop_count);
    f->dump_unsigned("write_error_count", trace_status.write_error_count);
    f->dump_string("path", trace_status.path);
    f->dump_string("phase", trace_status.phase);
    f->close_section();
    f->close_section();
  }

  void hp_rotate_trace_if_enabled()
  {
    const HpTraceStatus status =
      osd_object_heat_predictor.get_trace_status();
    if (!status.enabled || osd_object_hp_osd_id < 0) {
      return;
    }
    osd_object_heat_predictor.start_trace(
      osd_object_hp_trace_directory,
      osd_object_hp_osd_id,
      osd_object_hp_trace_phase,
      git_version_to_str());
  }

  void hp_reset_osd_object_heat_predictor(ceph::Formatter *f)
  {
    std::unique_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);

    uint64_t discarded_pending_io = osd_object_heat_predictor.reset();
    hp_rotate_trace_if_enabled();
    hp_reset_osd_op_counters();
    hp_zero_object_logger();

    if (f != nullptr) {
      const auto predictor_status = osd_object_heat_predictor.status();
      const auto& stats = predictor_status.evaluation;
      f->open_object_section("object_hp_reset");
      f->dump_bool("ok", true);
      f->dump_bool("enabled", stats.enabled);
      f->dump_unsigned("discarded_pending_io", discarded_pending_io);
      f->dump_unsigned("hp_io_count", stats.io_count);
      f->dump_unsigned("hp_labeled_io_total", stats.labeled_io_total);
      f->dump_unsigned("hp_pending_io_count", stats.pending_io_count);
      f->dump_unsigned(
        "hp_awaiting_prediction_count",
        stats.awaiting_prediction_count);
      f->dump_unsigned("hp_eval_drop_count", stats.eval_drop_count);
      f->dump_unsigned("hp_heat_state_count", stats.heat_state_count);
      f->dump_unsigned("hp_lru_count", stats.lru_count);
      f->dump_unsigned(
        "hp_protected_heat_state_count",
        stats.protected_heat_state_count);
      f->dump_unsigned(
        "hp_heat_state_peak_count",
        stats.heat_state_peak_count);
      f->dump_unsigned(
        "hp_lru_eviction_count",
        stats.lru_eviction_count);
      hp_dump_future_access_threshold(f, stats);
      f->dump_unsigned("hp_train_queue_length",
                       predictor_status.train_queue_length);
      f->dump_unsigned("hp_train_drop_count",
                       predictor_status.train_drop_count);
      f->dump_unsigned("hp_snapshot_publish_count",
                       predictor_status.snapshot_publish_count);
      f->dump_unsigned("hp_predict_error_count",
                       predictor_status.predict_error_count);
      f->dump_unsigned("hp_background_error_count",
                       predictor_status.background_error_count);
      f->close_section();
    }
  }

  void hp_set_osd_object_heat_predictor_enabled(ceph::Formatter *f,
                                                bool enabled)
  {
    std::unique_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);

    // Close before reset/model allocation, including a failed enable.
    if (observation_gate)
      observation_gate->store(false, std::memory_order_release);

    uint64_t discarded_pending_io =
      osd_object_heat_predictor.set_enabled(enabled);
    // Publish only after the core transition; inner enabled checks still guard
    // observations that passed the gate before disable acquired reset_lock.
    if (observation_gate)
      observation_gate->store(enabled, std::memory_order_release);
    hp_rotate_trace_if_enabled();
    hp_reset_osd_op_counters();
    hp_zero_object_logger();

    if (f != nullptr) {
      const auto predictor_status = osd_object_heat_predictor.status();
      const auto& stats = predictor_status.evaluation;
      f->open_object_section(enabled ? "object_hp_enable" : "object_hp_disable");
      f->dump_bool("ok", true);
      f->dump_bool("enabled", stats.enabled);
      f->dump_unsigned("discarded_pending_io", discarded_pending_io);
      f->dump_unsigned("hp_io_count", stats.io_count);
      f->dump_unsigned("hp_labeled_io_total", stats.labeled_io_total);
      f->dump_unsigned("hp_pending_io_count", stats.pending_io_count);
      f->dump_unsigned(
        "hp_awaiting_prediction_count",
        stats.awaiting_prediction_count);
      f->dump_unsigned("hp_eval_drop_count", stats.eval_drop_count);
      f->dump_unsigned("hp_heat_state_count",
                       stats.heat_state_count);
      f->dump_unsigned("hp_lru_count",
                       stats.lru_count);
      f->dump_unsigned(
        "hp_protected_heat_state_count",
        stats.protected_heat_state_count);
      f->dump_unsigned(
        "hp_heat_state_peak_count",
        stats.heat_state_peak_count);
      f->dump_unsigned(
        "hp_lru_eviction_count",
        stats.lru_eviction_count);
      hp_dump_future_access_threshold(f, stats);
      f->dump_unsigned("hp_train_queue_length",
                       predictor_status.train_queue_length);
      f->dump_unsigned("hp_train_drop_count",
                       predictor_status.train_drop_count);
      f->dump_unsigned("hp_snapshot_publish_count",
                       predictor_status.snapshot_publish_count);
      f->dump_unsigned("hp_predict_error_count",
                       predictor_status.predict_error_count);
      f->dump_unsigned("hp_background_error_count",
                       predictor_status.background_error_count);
      f->close_section();
    }
  }

  void hp_start_osd_object_heat_predictor_trace(
      ceph::Formatter *f,
      const std::string& phase,
      const std::string& directory)
  {
    std::unique_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);

    osd_object_hp_trace_phase = phase;
    osd_object_hp_trace_directory =
      directory.empty() ? "/var/log/ceph" : directory;
    const bool ok = osd_object_hp_osd_id >= 0 &&
      osd_object_heat_predictor.start_trace(
        osd_object_hp_trace_directory,
        osd_object_hp_osd_id,
        osd_object_hp_trace_phase,
        git_version_to_str());
    hp_update_object_logger(ceph::timespan::zero(), false);

    if (f != nullptr) {
      const HpTraceStatus status =
        osd_object_heat_predictor.get_trace_status();
      f->open_object_section("object_hp_trace_start");
      f->dump_bool("ok", ok);
      f->dump_bool("enabled", status.enabled);
      f->dump_unsigned("session_id", status.session_id);
      f->dump_string("path", status.path);
      f->dump_string("phase", status.phase);
      f->dump_unsigned("write_error_count", status.write_error_count);
      f->close_section();
    }
  }

  void hp_stop_osd_object_heat_predictor_trace(ceph::Formatter *f)
  {
    std::unique_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);
    osd_object_heat_predictor.stop_trace();
    hp_update_object_logger(ceph::timespan::zero(), false);

    if (f != nullptr) {
      const HpTraceStatus status =
        osd_object_heat_predictor.get_trace_status();
      f->open_object_section("object_hp_trace_stop");
      f->dump_bool("ok", true);
      f->dump_bool("enabled", status.enabled);
      f->dump_unsigned("session_id", status.session_id);
      f->dump_unsigned("written_count", status.written_count);
      f->dump_unsigned("drop_count", status.drop_count);
      f->dump_unsigned("write_error_count", status.write_error_count);
      f->dump_string("path", status.path);
      f->close_section();
    }
  }

  void hp_notify_osd_object_op(const hobject_t& soid,
                               HpAccessType op)
  {
    // Disabled observations need only an atomic load, not the shared lock.
    // Keep the check under the lock too: disable/reset may race this fast path.
    if (!osd_object_heat_predictor.is_enabled() || !hp_track_osd_op(op)) {
      return;
    }

    std::shared_lock<std::shared_mutex> reset_lock(osd_object_hp_reset_mtx);
    if (!osd_object_heat_predictor.is_enabled()) {
      return;
    }

    hp_count_osd_op(op);
    auto start_time = ceph::mono_clock::now();
    uint64_t index = 0;
    try {
      osd_object_heat_predictor.predict(
        HpObjectIdentityView{soid.pool, soid.get_hash(), soid.oid.name,
                             soid.nspace, static_cast<uint64_t>(soid.snap),
                             soid.get_key()},
        &index);
    } catch (...) {
      osd_object_heat_predictor.record_predict_error();
      index = 0;
    }
    auto end_time = ceph::mono_clock::now();
    hp_record_object_predict_latency(end_time - start_time);
    if (hp_should_update_object_logger(index)) {
      hp_update_object_logger(ceph::timespan::zero(), false);
    }
  }

};

ObjectHeatPredictor::ObjectHeatPredictor() : impl(std::make_unique<Impl>()) {}
ObjectHeatPredictor::~ObjectHeatPredictor() = default;

void ObjectHeatPredictor::init(CephContext* cct, int osd_id,
                               std::shared_ptr<std::atomic<bool>> gate) {
  ceph_assert(cct && impl && impl->context == nullptr);
  impl->observation_gate = std::move(gate);
  if (impl->observation_gate)
    impl->observation_gate->store(false, std::memory_order_release);
  impl->init_osd_object_hp_status(cct, osd_id);
}

void ObjectHeatPredictor::observe(const hobject_t& object, HpAccessType op,
                                  uint64_t effective_length) {
  if (effective_length != 0 && impl) {
    impl->hp_notify_osd_object_op(object, op);
  }
}

void ObjectHeatPredictor::shutdown() { impl.reset(); }

bool ObjectHeatPredictor::handle_command(std::string_view prefix,
                                         const cmdmap_t& cmdmap,
                                         ceph::Formatter* f) {
  if (!impl || !impl->context) return false;
  if (prefix == "object_hp reset") {
    impl->hp_reset_osd_object_heat_predictor(f);
  } else if (prefix == "object_hp status") {
    impl->hp_dump_osd_object_heat_predictor_status(f);
  } else if (prefix == "object_hp enable") {
    impl->hp_set_osd_object_heat_predictor_enabled(f, true);
  } else if (prefix == "object_hp disable") {
    impl->hp_set_osd_object_heat_predictor_enabled(f, false);
  } else if (prefix == "object_hp trace start") {
    std::string phase;
    std::string directory;
    ceph::common::cmd_getval(cmdmap, "phase", phase);
    ceph::common::cmd_getval(cmdmap, "directory", directory);
    impl->hp_start_osd_object_heat_predictor_trace(
      f, phase, directory);
  } else if (prefix == "object_hp trace stop") {
    impl->hp_stop_osd_object_heat_predictor_trace(f);
  } else {
    return false;
  }
  return true;
}

void ObjectHeatPredictor::register_commands(AdminSocket* admin_socket,
                                             AdminSocketHook* asok_hook) {
  ceph_assert(impl && impl->context);
  int r;
  r = admin_socket->register_command("object_hp reset", asok_hook,
				     "reset object heat predictor state");
  ceph_assert(r == 0);
  r = admin_socket->register_command("object_hp status", asok_hook,
				     "show live object heat predictor state");
  ceph_assert(r == 0);
  r = admin_socket->register_command("object_hp enable", asok_hook,
				     "enable and reset object heat predictor");
  ceph_assert(r == 0);
  r = admin_socket->register_command("object_hp disable", asok_hook,
				     "disable and reset object heat predictor");
  ceph_assert(r == 0);
  r = admin_socket->register_command(
    "object_hp trace start "
    "name=phase,type=CephString,req=false "
    "name=directory,type=CephString,req=false",
    asok_hook,
    "start or rotate completed-evaluation trace");
  ceph_assert(r == 0);
  r = admin_socket->register_command(
    "object_hp trace stop",
    asok_hook,
    "drain and stop completed-evaluation trace");
  ceph_assert(r == 0);
}
