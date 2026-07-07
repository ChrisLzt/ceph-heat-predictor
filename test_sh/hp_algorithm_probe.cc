#include <cmath>
#include <iostream>
#include <memory>

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
  require(eq.hot_list.size() == 1,
          "first object should create one object heat entry");
  require(std::abs(eq.hot_threshold - 10.0) < 0.000001,
          "single-object threshold should use current object heat");

  eq.record_object_heat(42, 30.0, 2);
  require(eq.hot_list.size() == 1,
          "same object should replace object heat instead of appending");
  require(std::abs(eq.hot_threshold - 30.0) < 0.000001,
          "same object replacement should update threshold heat");

  eq.record_object_heat(7, 20.0, 2);
  require(eq.hot_list.size() == 2,
          "different object should create a second object heat entry");

  eq.record_object_heat(42, 5.0, 2);
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

  require(std::abs(eq.hot_threshold - 100.0) < 0.000001,
          "new object should outrank an idle object after one decay window");
}

void test_simple_training_weight_uses_hot_class_weight()
{
  EvaluationQueue hot_eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem first_hot = make_item(1, 42);
  hot_eq.prepare_features(first_hot);
  require(!hot_eq.enqueue(first_hot).has_value(),
          "first hot weight probe item should stay pending");

  TraceItem second_hot = make_item(2, 42);
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

} // namespace

int main()
{
  test_prediction_clone_is_independent();
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
  test_simple_training_weight_uses_hot_class_weight();
  std::cout << "PASS: hp algorithm probe" << std::endl;
  return 0;
}
