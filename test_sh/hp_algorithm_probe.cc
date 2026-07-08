#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

#include "heatpredictor/heat_predictor.h"

namespace {

TraceItem make_item(uint64_t index, uint64_t key)
{
  return TraceItem{
    index,  // index
    0,      // operation
    4096,   // size
    key,    // key
    0.0,    // current_heat
    0.0,    // hot_threshold
    0,      // access_count
    0,      // last_access_distance
    0,      // object_age
    0,      // pred_hot_proba
    0       // pred
  };
}

void require(bool condition, const char *message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void require_close(double lhs, double rhs, const char *message)
{
  if (std::abs(lhs - rhs) > 0.000001) {
    std::cerr << "FAIL: " << message
              << " lhs=" << lhs << " rhs=" << rhs << std::endl;
    std::exit(1);
  }
}

double expected_next_predict_threshold(double current_threshold, double ratio)
{
  double safe_ratio = std::max(ratio, std::numeric_limits<double>::min());
  double feedback = std::clamp(
      safe_ratio,
      HP_PRED_ACTUAL_HOT_RATIO_MIN,
      HP_PRED_ACTUAL_HOT_RATIO_MAX);
  double target = current_threshold * feedback;
  return std::clamp(
      (1.0 - HP_HOT_PREDICT_THRESHOLD_EMA_ALPHA) * current_threshold +
      HP_HOT_PREDICT_THRESHOLD_EMA_ALPHA * target,
      HP_HOT_PREDICT_THRESHOLD_MIN,
      HP_HOT_PREDICT_THRESHOLD_MAX);
}

void require_proba_close(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    const char *message)
{
  require(lhs.size() == rhs.size(), "probability vectors should match size");
  for (size_t i = 0; i < lhs.size(); ++i) {
    require_close(lhs[i], rhs[i], message);
  }
}

template <typename T, typename = void>
struct has_predicted_hot : std::false_type {};

template <typename T>
struct has_predicted_hot<T, std::void_t<decltype(std::declval<T>().predicted_hot)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_predicted_cold : std::false_type {};

template <typename T>
struct has_predicted_cold<T, std::void_t<decltype(std::declval<T>().predicted_cold)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_actual_hot_counter : std::false_type {};

template <typename T>
struct has_actual_hot_counter<T, std::void_t<decltype(std::declval<T>().actual_hot)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_actual_cold_counter : std::false_type {};

template <typename T>
struct has_actual_cold_counter<T, std::void_t<decltype(std::declval<T>().actual_cold)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_hot_predict_threshold_stat : std::false_type {};

template <typename T>
struct has_hot_predict_threshold_stat<
    T, std::void_t<decltype(std::declval<T>().hot_predict_threshold)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_otsu_histogram_bin_count_stat : std::false_type {};

template <typename T>
struct has_otsu_histogram_bin_count_stat<
    T, std::void_t<decltype(std::declval<T>().otsu_histogram_bin_count)>>
    : std::true_type {};

void test_stats_drop_online_prediction_ratio_source()
{
  require(!has_predicted_hot<HeatPredictorStats>::value,
          "HeatPredictorStats should not expose predicted_hot");
  require(!has_predicted_cold<HeatPredictorStats>::value,
          "HeatPredictorStats should not expose predicted_cold");
}

void test_predictor_drops_unused_actual_label_counters()
{
  require(!has_actual_hot_counter<HeatPredictor>::value,
          "HeatPredictor should not keep unused actual_hot counter");
  require(!has_actual_cold_counter<HeatPredictor>::value,
          "HeatPredictor should not keep unused actual_cold counter");
}

void test_stats_export_hot_predict_threshold()
{
  require(has_hot_predict_threshold_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should expose hot_predict_threshold");
}

void test_stats_export_otsu_histogram_bin_count()
{
  require(has_otsu_histogram_bin_count_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should expose Otsu histogram bin count");
}

void test_otsu_update_cost_knobs()
{
  require(HP_OTSU_EAGER_OBJECTS == 0,
          "Otsu eager updates should be disabled for fixed interval updates");
  require(HP_OTSU_UPDATE_INTERVAL == 100,
          "Otsu update interval should refresh every 100 IO observations");
  require_close(HP_OTSU_HEAT_MIN, 1.0,
                "Otsu heat clamp lower bound should be 1.0");
  require_close(HP_OTSU_HEAT_MAX, 3000.0,
                "Otsu heat clamp upper bound should be 3000.0");
  require_close(HP_OTSU_LOG_HEAT_BIN_WIDTH, 0.05,
                "Otsu log-heat histogram bin width should be 0.05");
}

void test_learning_lag_knobs()
{
  require(HP_EVALUATION_WINDOW == 2000,
          "evaluation window should be 2000 for hotspot migration sensitivity");
  require(HeatPredictor::MODEL_UPDATE_REPORT_INTERVAL == 500,
          "snapshot publish interval should be 500 trained samples");
}

void test_prediction_clone_is_independent()
{
  std::unique_ptr<Classifier> model(HeatPredictor::make_model());
  const std::vector<double> hot = {1.0, 12.0, 4.0, 3.0, 1.0, 5.0, 0.8};
  const std::vector<double> cold = {0.0, 4.0, 1.0, 0.0, 6.0, 20.0, 0.1};

  for (int i = 0; i < 80; ++i) {
    model->learn_one(hot, 1, 1.0);
    model->learn_one(cold, 0, 1.0);
  }

  std::vector<double> original_before = model->predict_proba_one(hot);
  std::unique_ptr<Classifier> snapshot = model->clone_for_prediction();
  std::vector<double> snapshot_before = snapshot->predict_proba_one(hot);

  require_proba_close(
      original_before,
      snapshot_before,
      "snapshot should match source model at clone time");

  for (int i = 0; i < 80; ++i) {
    model->learn_one(hot, 0, 5.0);
    model->learn_one(cold, 1, 5.0);
  }

  std::vector<double> snapshot_after = snapshot->predict_proba_one(hot);
  require_proba_close(
      snapshot_before,
      snapshot_after,
      "snapshot should be independent from later source training");
}

void test_predictor_enable_disable_resets_and_gates_io()
{
  HeatPredictor predictor;
  uint64_t index = 0;

  require(predictor.is_enabled(),
          "heat predictor should be enabled by default");
  predictor.predict(0, 4096, 1, 1, 1, &index);
  require(index == 1, "enabled predictor should process the first IO");
  require(predictor.hp_index.load() == 1,
          "enabled predictor should count processed IO");

  uint64_t discarded = predictor.set_enabled(false);
  require(discarded == 1,
          "disable should reset and report discarded pending IO");
  require(!predictor.is_enabled(),
          "disable should turn off the predictor");
  require(predictor.hp_index.load() == 0,
          "disable should reset processed IO count");

  index = 99;
  predictor.predict(0, 4096, 1, 1, 1, &index);
  require(index == 0,
          "disabled predictor should report no processed IO index");
  require(predictor.hp_index.load() == 0,
          "disabled predictor should not count IO");
  require(predictor.get_pending_io_count() == 0,
          "disabled predictor should not enqueue IO");

  predictor.predict(0, 4096, 1, 1, 1, nullptr);
  uint64_t enabled_discarded = predictor.set_enabled(true);
  require(enabled_discarded == 0,
          "enable should reset even when disabled left no pending IO");
  require(predictor.is_enabled(),
          "enable should turn on the predictor");
  require(predictor.hp_index.load() == 0,
          "enable should reset processed IO count");

  predictor.predict(0, 4096, 1, 1, 1, &index);
  require(index == 1,
          "predictor should process IO again after enable");
}

void test_future_heat_label_ignores_decayed_history_only()
{
  EvaluationQueue eq(
    2,      // evaluation_window
    8,      // lru_capacity
    5.0,    // hot_threshold
    100.0); // heat_increment

  TraceItem first = make_item(1, 1);
  eq.prepare_features(first);
  require(!eq.enqueue(first).has_value(), "first item should stay pending");

  TraceItem second = make_item(2, 2);
  eq.prepare_features(second);
  require(!eq.enqueue(second).has_value(), "second item should stay pending");

  TraceItem third = make_item(3, 3);
  eq.prepare_features(third);
  auto evaluated = eq.enqueue(third);
  require(evaluated.has_value(), "third item should evict first item");
  require(evaluated->future_access_count == 0,
          "evicted item should have no future accesses");
  require(evaluated->future_heat < 0.000001,
          "future heat should be zero when there are no future accesses");
  require(evaluated->label == 0,
          "future heat label should be cold without future accesses");
}

void test_balanced_accuracy_penalizes_missing_hot_class()
{
  Accuracy<2> metric;
  for (int i = 0; i < 90; ++i) {
    metric.update(0, 0);
  }
  for (int i = 0; i < 10; ++i) {
    metric.update(1, 0);
  }

  require(std::abs(metric.get_accuracy() - 0.90) < 0.000001,
          "overall accuracy should be 90%");
  require(std::abs(metric.get_balanced_accuracy() - 0.50) < 0.000001,
          "balanced accuracy should average cold and hot recall");
}

void test_quantile_window_keeps_recent_values()
{
  HpQuantileWindow window(5);
  for (int i = 1; i <= 6; ++i) {
    window.insert(static_cast<double>(i));
  }

  HpDistributionSummary summary = window.summary();
  require(summary.count == 5, "quantile window should keep its capacity");
  require(std::abs(summary.max - 6.0) < 0.000001,
          "max should be the largest retained value");
  require(std::abs(summary.p50 - 4.0) < 0.000001,
          "p50 should be computed from retained values");
  require(std::abs(summary.p90 - 6.0) < 0.000001,
          "p90 should be computed from retained values");
  require(std::abs(summary.p95 - 6.0) < 0.000001,
          "p95 should be computed from retained values");
  require(std::abs(summary.p99 - 6.0) < 0.000001,
          "p99 should be computed from retained values");
}

void test_object_key_uses_object_identity()
{
  uint64_t key = HeatPredictor::make_object_key(
      1,      // pool
      11,     // ceph_object_hash
      22);    // object_name_hash
  uint64_t other_object_key = HeatPredictor::make_object_key(
      1,      // pool
      11,     // ceph_object_hash
      23);    // object_name_hash

  require(key != other_object_key,
          "object-level key should still distinguish objects");
}

void test_prepare_features_tracks_recency_and_age()
{
  EvaluationQueue eq(
    16,     // evaluation_window
    8,      // lru_capacity
    5.0,    // hot_threshold
    100.0); // heat_increment

  TraceItem first = make_item(10, 42);
  eq.prepare_features(first);
  require(!eq.enqueue(first).has_value(),
          "first recency probe item should stay pending");
  require(first.access_count == 1,
          "first object access should initialize access_count");
  require(first.last_access_distance == 0,
          "first object access should have zero last_access_distance");
  require(first.object_age == 0,
          "first object access should have zero object_age");

  TraceItem second = make_item(15, 42);
  eq.prepare_features(second);
  require(!eq.enqueue(second).has_value(),
          "second recency probe item should stay pending");
  require(second.access_count == 2,
          "second object access should increment access_count");
  require(second.last_access_distance == 5,
          "second object access should record distance from previous access");
  require(second.object_age == 5,
          "second object access should record age from first access");

  TraceItem third = make_item(25, 42);
  eq.prepare_features(third);
  require(!eq.enqueue(third).has_value(),
          "third recency probe item should stay pending");
  require(third.access_count == 3,
          "third object access should increment access_count");
  require(third.last_access_distance == 10,
          "third object access should update distance from previous access");
  require(third.object_age == 15,
          "third object access should preserve first access timestamp");
}

void test_default_capacity_parameters()
{
  EvaluationQueue eq;
  require(eq.evaluation_window == HP_EVALUATION_WINDOW,
          "default evaluation window should use HP_EVALUATION_WINDOW");
  require(eq.lru_capacity == HP_LRU_CAPACITY,
          "default LRU capacity should use HP_LRU_CAPACITY");
  require(eq.hot_list_cap == HP_LABEL_THRESHOLD_WINDOW_CAPACITY,
          "default threshold window should use label threshold capacity");
}

void test_training_batch_size_is_low_latency()
{
  require(HeatPredictor::BATCH_SIZE == 100,
          "training notification batch size should favor low latency");
}

void test_threshold_window_tracks_object_current_heat()
{
  EvaluationQueue eq(
    4,      // evaluation_window
    8,      // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  eq.record_object_heat(42, 10.0, 1);
  eq.update_hot_threshold(1);
  require(eq.hot_list.size() == 1,
          "first object should create one object heat entry");
  require(std::abs(eq.hot_threshold - 10.0) < 0.000001,
          "single-object threshold should use current object heat");

  eq.record_object_heat(42, 30.0, 2);
  eq.update_hot_threshold(2);
  require(eq.hot_list.size() == 1,
          "same object should replace object heat instead of appending");
  require(std::abs(eq.hot_threshold - 30.0) < 0.000001,
          "same object replacement should update threshold heat");

  eq.record_object_heat(7, 20.0, 2);
  eq.update_hot_threshold(2);
  require(eq.hot_list.size() == 2,
          "different object should create a second object heat entry");

  eq.record_object_heat(42, 5.0, 2);
  eq.update_hot_threshold(2);
  require(eq.hot_list.size() == 2,
          "replacing one object should not change object heat entry count");
  require(std::abs(eq.hot_threshold - 20.0) < 0.000001,
          "replacement should remove the old object heat from threshold");
}

void test_threshold_window_order_has_one_entry_per_object()
{
  EvaluationQueue eq(
    4,      // evaluation_window
    8,      // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  eq.record_object_heat(1, 10.0, 1);
  eq.record_object_heat(2, 20.0, 2);
  eq.record_object_heat(3, 30.0, 3);
  eq.record_object_heat(2, 40.0, 4);

  require(eq.hot_list.size() == 3,
          "threshold tree should keep one entry per object");
  require(eq.hot_list_order.size() == 3,
          "threshold recency list should not retain stale object entries");
}

void test_object_heat_threshold_decays_idle_objects()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    8,      // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  eq.record_object_heat(1, 100.0, 1);
  eq.record_object_heat(2, 100.0, 10001);
  eq.update_hot_threshold(10001);

  require(std::abs(eq.hot_threshold - 100.0) < 0.000001,
          "new object should outrank an idle object after one decay window");
}

void test_otsu_threshold_separates_bimodal_object_heat()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  for (uint64_t i = 0; i < 100; ++i) {
    eq.record_object_heat(i, 10.0, 1);
  }
  for (uint64_t i = 100; i < 120; ++i) {
    eq.record_object_heat(i, 1000.0, 1);
  }
  eq.update_hot_threshold(1);

  require(eq.hot_threshold > 10.0,
          "Otsu threshold should rise above the cold object heat");
  require(eq.hot_threshold < 500.0,
          "Otsu threshold should not collapse to the hot cluster value");
  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_OTSU,
          "Otsu threshold method should be reported when Otsu is active");
  require_close(eq.hot_quantile_threshold, eq.hot_threshold,
                "reported quantile threshold should match the active Otsu threshold");
  require(eq.otsu_separation >= HP_OTSU_MIN_SEPARATION,
          "Otsu separation should be reported when Otsu is active");
}

void test_otsu_histogram_tracks_threshold_window_entries()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment
  eq.hot_list_cap = 2;

  eq.record_object_heat(1, 10.0, 1);
  require(eq.otsu_histogram.size() == 1,
          "histogram should track the first threshold object");
  require(eq.otsu_histogram.bin_count() == 1,
          "histogram should report one occupied bin for one object");

  eq.record_object_heat(1, 20.0, 2);
  require(eq.otsu_histogram.size() == 1,
          "histogram should replace existing object heat");
  require(eq.otsu_histogram.bin_count() == 1,
          "histogram should keep one occupied bin after object replacement");

  eq.record_object_heat(2, 30.0, 2);
  require(eq.otsu_histogram.size() == 2,
          "histogram should track distinct threshold objects");
  require(eq.otsu_histogram.bin_count() == 2,
          "histogram should report occupied map bins separately from sample count");

  eq.record_object_heat(3, 40.0, 3);
  require(eq.hot_list.size() == 2,
          "threshold tree should evict to hot_list_cap");
  require(eq.otsu_histogram.size() == eq.hot_list.size(),
          "histogram should evict with the threshold tree");
  require(eq.otsu_histogram.bin_count() == eq.hot_list.size(),
          "histogram bin count should evict with the threshold tree");
}

void test_otsu_threshold_window_clamps_low_scores_without_dropping_entries()
{
  EvaluationQueue eq(
    2000,   // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  eq.record_object_heat(1, HP_OTSU_HEAT_MIN, 1);
  require(eq.hot_list_by_key.count(1) == 1,
          "fresh min-heat object should enter the threshold window");
  require(eq.otsu_histogram.size() == 1,
          "histogram should track the fresh threshold object");

  eq.record_object_heat(2, HP_OTSU_HEAT_MIN, 10000);
  require(eq.hot_list_by_key.count(1) == 1,
          "threshold window should keep old objects until TW size eviction");
  require(eq.hot_list_by_key.count(2) == 1,
          "current min-heat object should remain in the threshold window");
  require(eq.otsu_histogram.size() == eq.hot_list_by_key.size(),
          "histogram sample count should stay synchronized with threshold entries");
  require(eq.otsu_histogram.bin_count() == 1,
          "old low-score bins should be merged into the current lower-bound bin");

  eq.record_object_heat(1, HP_OTSU_HEAT_MIN, 10001);
  require(eq.hot_list_by_key.size() == 2,
          "updating an old low-score object should replace instead of append");
  require(eq.otsu_histogram.size() == eq.hot_list_by_key.size(),
          "histogram replacement should keep one sample per threshold object");
}

void test_threshold_ema_smooths_large_otsu_shift()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  for (uint64_t i = 0; i < 100; ++i) {
    eq.record_object_heat(i, 10.0, 1);
  }
  for (uint64_t i = 100; i < 120; ++i) {
    eq.record_object_heat(i, 1000.0, 1);
  }
  eq.update_hot_threshold(1);

  double first_threshold = eq.hot_threshold;
  require(first_threshold > 10.0 && first_threshold < 500.0,
          "initial Otsu threshold should split the first bimodal heat set");

  for (uint64_t i = 100; i < 120; ++i) {
    eq.record_object_heat(i, 1000000.0, 2);
  }
  eq.update_hot_threshold(2);

  require(eq.hot_threshold > first_threshold,
          "EMA threshold should still move toward the new hot cluster");
  require(eq.hot_threshold < 500.0,
          "EMA threshold should smooth a large Otsu threshold jump");
}

void test_otsu_threshold_updates_every_100_observations()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  for (uint64_t i = 0; i < 99; ++i) {
    eq.record_object_heat(i, 10.0, i + 1);
  }
  require_close(eq.hot_threshold, 100.0,
                "Otsu threshold should not update before 100 observations");

  eq.record_object_heat(99, 10.0, 100);
  require(eq.hot_threshold > 9.9 && eq.hot_threshold < 10.1,
          "Otsu threshold should update on the 100th observation");
}

void test_threshold_reports_quantile_fallback()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  eq.record_object_heat(1, 10.0, 1);
  eq.record_object_heat(2, 20.0, 1);
  eq.update_hot_threshold(1);

  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_QUANTILE,
          "small threshold windows should report quantile fallback");
  require_close(eq.hot_threshold, eq.hot_quantile_threshold,
                "fallback threshold should equal quantile threshold");
  require_close(eq.otsu_separation, 0.0,
                "Otsu separation should be zero when Otsu is not active");
}

void test_simple_training_weight_uses_hot_class_weight()
{
  require(std::abs(HP_HOT_CLASS_WEIGHT - 3.0) < 0.000001,
          "hot class weight should be fixed at 3.0");

  EvaluationQueue hot_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem first_hot = make_item(1, 42);
  first_hot.pred = 1;
  hot_eq.prepare_features(first_hot);
  require(!hot_eq.enqueue(first_hot).has_value(),
          "first hot weight probe item should stay pending");

  TraceItem second_hot = make_item(2, 42);
  second_hot.pred = 1;
  hot_eq.prepare_features(second_hot);
  hot_eq.hot_threshold = 50.0;
  auto hot = hot_eq.enqueue(second_hot);
  require(hot.has_value(), "second access should evaluate first item");
  require(hot->label == 1, "first item should be labeled hot");
  require(std::abs(hot->training_weight - HP_HOT_CLASS_WEIGHT) < 0.000001,
          "hot sample should use fixed hot class weight");

  EvaluationQueue cold_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem first_cold = make_item(1, 1);
  cold_eq.prepare_features(first_cold);
  require(!cold_eq.enqueue(first_cold).has_value(),
          "first cold weight probe item should stay pending");

  TraceItem second_cold = make_item(2, 2);
  cold_eq.prepare_features(second_cold);
  auto cold = cold_eq.enqueue(second_cold);
  require(cold.has_value(), "second object should evaluate first item");
  require(cold->label == 0, "first item should be labeled cold");
  require(std::abs(cold->training_weight - 1.0) < 0.000001,
          "cold sample should keep unit weight");
}

void test_pred_actual_hot_ratio_tracks_prediction_bias()
{
  EvaluationQueue false_positive_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem cold_pred_hot = make_item(1, 1);
  cold_pred_hot.pred = 1;
  false_positive_eq.prepare_features(cold_pred_hot);
  require(!false_positive_eq.enqueue(cold_pred_hot).has_value(),
          "first false-positive probe item should stay pending");

  TraceItem next_cold = make_item(2, 2);
  next_cold.pred = 1;
  false_positive_eq.prepare_features(next_cold);
  auto false_positive = false_positive_eq.enqueue(next_cold);
  require(false_positive.has_value(),
          "second false-positive probe item should evaluate first item");
  require(false_positive->label == 0,
          "first false-positive probe item should be actually cold");
  require(false_positive_eq.pred_actual_hot_ratio() > 1.0,
          "false positives should make predicted/actual hot ratio exceed 1");

  EvaluationQueue false_negative_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem hot_pred_cold = make_item(1, 42);
  hot_pred_cold.pred = 0;
  false_negative_eq.prepare_features(hot_pred_cold);
  require(!false_negative_eq.enqueue(hot_pred_cold).has_value(),
          "first false-negative probe item should stay pending");

  TraceItem next_hot = make_item(2, 42);
  next_hot.pred = 0;
  false_negative_eq.prepare_features(next_hot);
  false_negative_eq.hot_threshold = 50.0;
  auto false_negative = false_negative_eq.enqueue(next_hot);
  require(false_negative.has_value(),
          "second false-negative probe item should evaluate first item");
  require(false_negative->label == 1,
          "first false-negative probe item should be actually hot");
  require(false_negative_eq.pred_actual_hot_ratio() < 1.0,
          "false negatives should make predicted/actual hot ratio fall below 1");
}

void test_dynamic_hot_predict_threshold_uses_pred_actual_ratio()
{
  require_close(HP_PRED_ACTUAL_HOT_RATIO_MIN, 0.80,
                "dynamic predict threshold ratio lower clamp should be 0.80");
  require_close(HP_PRED_ACTUAL_HOT_RATIO_MAX, 1.25,
                "dynamic predict threshold ratio upper clamp should be 1.25");
  require_close(HP_HOT_PREDICT_THRESHOLD, 0.50,
                "hot predict threshold should start at 0.50");
  require_close(HP_HOT_PREDICT_THRESHOLD_MIN, 0.45,
                "hot predict threshold lower bound should be 0.45");
  require_close(HP_HOT_PREDICT_THRESHOLD_MAX, 0.55,
                "hot predict threshold upper bound should be 0.55");
  require_close(HP_HOT_PREDICT_THRESHOLD_EMA_ALPHA, 0.10,
                "hot predict threshold EMA alpha should be 0.10");

  EvaluationQueue formula_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment
  require_close(
      formula_eq.next_hot_predict_threshold_for_ratio(1.1),
      expected_next_predict_threshold(HP_HOT_PREDICT_THRESHOLD, 1.1),
      "dynamic predict threshold should update from current threshold and ratio");
  require_close(
      formula_eq.next_hot_predict_threshold_for_ratio(2.0),
      expected_next_predict_threshold(HP_HOT_PREDICT_THRESHOLD, 2.0),
      "dynamic predict threshold should clamp high feedback ratios");
  require_close(
      formula_eq.next_hot_predict_threshold_for_ratio(0.5),
      expected_next_predict_threshold(HP_HOT_PREDICT_THRESHOLD, 0.5),
      "dynamic predict threshold should clamp low feedback ratios");

  EvaluationQueue cold_biased_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem first_hot = make_item(1, 42);
  first_hot.pred = 0;
  cold_biased_eq.prepare_features(first_hot);
  require(!cold_biased_eq.enqueue(first_hot).has_value(),
          "first dynamic weight item should stay pending");

  TraceItem second_hot = make_item(2, 42);
  second_hot.pred = 0;
  cold_biased_eq.prepare_features(second_hot);
  cold_biased_eq.hot_threshold = 50.0;
  auto hot = cold_biased_eq.enqueue(second_hot);
  require(hot.has_value(), "second dynamic weight item should evaluate first");
  require(hot->label == 1, "dynamic weight item should be hot");
  require_close(hot->training_weight, HP_HOT_CLASS_WEIGHT,
                "hot sample weight should stay fixed under cold-biased history");
  require(cold_biased_eq.dynamic_hot_predict_threshold <
          HP_HOT_PREDICT_THRESHOLD,
          "cold-biased prediction should lower hot predict threshold");

  EvaluationQueue hot_biased_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem first_cold = make_item(1, 1);
  first_cold.pred = 1;
  hot_biased_eq.prepare_features(first_cold);
  require(!hot_biased_eq.enqueue(first_cold).has_value(),
          "first hot-biased item should stay pending");

  TraceItem second_cold = make_item(2, 2);
  second_cold.pred = 1;
  hot_biased_eq.prepare_features(second_cold);
  auto cold = hot_biased_eq.enqueue(second_cold);
  require(cold.has_value(), "second hot-biased item should evaluate first");
  require(cold->label == 0, "hot-biased probe should first evaluate cold");

  TraceItem first_hot_after_bias = make_item(3, 42);
  first_hot_after_bias.pred = 1;
  hot_biased_eq.prepare_features(first_hot_after_bias);
  (void)hot_biased_eq.enqueue(first_hot_after_bias);

  TraceItem second_hot_after_bias = make_item(4, 42);
  second_hot_after_bias.pred = 1;
  hot_biased_eq.prepare_features(second_hot_after_bias);
  hot_biased_eq.hot_threshold = 50.0;
  auto hot_after_bias = hot_biased_eq.enqueue(second_hot_after_bias);
  require(hot_after_bias.has_value(),
          "hot sample after hot-biased history should be evaluated");
  require(hot_after_bias->label == 1,
          "hot sample after hot-biased history should be hot");
  require_close(hot_after_bias->training_weight, HP_HOT_CLASS_WEIGHT,
                "hot sample weight should stay fixed under hot-biased history");
  require(hot_biased_eq.dynamic_hot_predict_threshold >
          HP_HOT_PREDICT_THRESHOLD,
          "hot-biased prediction should raise hot predict threshold");
}

} // namespace

int main()
{
  test_stats_drop_online_prediction_ratio_source();
  test_predictor_drops_unused_actual_label_counters();
  test_stats_export_hot_predict_threshold();
  test_stats_export_otsu_histogram_bin_count();
  test_otsu_update_cost_knobs();
  test_learning_lag_knobs();
  test_prediction_clone_is_independent();
  test_predictor_enable_disable_resets_and_gates_io();
  test_future_heat_label_ignores_decayed_history_only();
  test_balanced_accuracy_penalizes_missing_hot_class();
  test_quantile_window_keeps_recent_values();
  test_object_key_uses_object_identity();
  test_prepare_features_tracks_recency_and_age();
  test_default_capacity_parameters();
  test_training_batch_size_is_low_latency();
  test_threshold_window_tracks_object_current_heat();
  test_threshold_window_order_has_one_entry_per_object();
  test_object_heat_threshold_decays_idle_objects();
  test_otsu_threshold_separates_bimodal_object_heat();
  test_otsu_histogram_tracks_threshold_window_entries();
  test_otsu_threshold_window_clamps_low_scores_without_dropping_entries();
  test_threshold_ema_smooths_large_otsu_shift();
  test_otsu_threshold_updates_every_100_observations();
  test_threshold_reports_quantile_fallback();
  test_simple_training_weight_uses_hot_class_weight();
  test_pred_actual_hot_ratio_tracks_prediction_bias();
  test_dynamic_hot_predict_threshold_uses_pred_actual_ratio();
  std::cout << "PASS: hp algorithm probe" << std::endl;
  return 0;
}
