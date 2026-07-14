#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "heatpredictor/heat_predictor.h"
#include "heatpredictor/hp_prediction_threshold.h"
#include "osd/ObjectHeatPredictor.h"

namespace ceph {

void __ceph_assert_fail(const assert_data& ctx)
{
  std::cerr << "ceph_assert failed: " << ctx.assertion
            << " at " << ctx.file << ":" << ctx.line << std::endl;
  std::abort();
}

void __ceph_assert_fail(
    const char *assertion,
    const char *file,
    int line,
    const char *function)
{
  (void)function;
  std::cerr << "ceph_assert failed: " << assertion
            << " at " << file << ":" << line << std::endl;
  std::abort();
}

} // namespace ceph

namespace {

std::atomic<uint64_t> expiry_progress_callback_count{0};

void record_expiry_progress(uint64_t expired_count)
{
  expiry_progress_callback_count.fetch_add(
      expired_count, std::memory_order_relaxed);
}

using HpNotifySignature = void (*)(CephContext *, const hobject_t&, uint16_t);
using HpStatusSignature = void (*)(CephContext *, ceph::Formatter *);

static_assert(
    std::is_same_v<decltype(&hp_notify_osd_object_op), HpNotifySignature>,
    "OSD hook must pass only an already validated object op type");
static_assert(
    std::is_same_v<decltype(&hp_dump_osd_object_heat_predictor_status),
                   HpStatusSignature>,
    "OSD status command must expose live heat predictor state");

PredictionSample make_item(uint64_t io_sequence, uint64_t object_key_hash)
{
  return PredictionSample{
    io_sequence,      // io_sequence
    object_key_hash,  // object_key_hash
    0.0,              // heat_after_current_access
    0.0,              // heat_label_threshold_at_prediction
    0,      // tracked_access_count
    0,      // time_since_previous_access_ns
    0,      // long_window_access_count
    0,      // short_window_access_count
    0.0,    // heat_percentile
    0,      // predicted_hot_probability
    0       // predicted_label
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

template <typename Fn>
void require_invalid_argument(Fn&& fn, const char *message)
{
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  } catch (...) {
    require(false, message);
  }
  require(false, message);
}

std::vector<double> model_features(std::initializer_list<double> values)
{
  std::vector<double> result(NUM_FEATURES, 0.0);
  std::copy_n(values.begin(), std::min(values.size(), result.size()),
              result.begin());
  return result;
}

void train_seed_probe_model(Classifier& model);

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

void test_log2p1_transform()
{
  require_close(hp_log2p1(0.0), 0.0,
                "log2p1 should map zero to zero");
  require_close(hp_log2p1(1.0), 1.0,
                "log2p1 should compute log2(1 + x)");
  require_close(hp_log2p1(3.0), 2.0,
                "log2p1 should preserve exact powers of two after +1");
}

void require_probability_distribution(
    const std::vector<double>& proba,
    const char *message)
{
  require(proba.size() == 2, message);
  require(std::isfinite(proba[0]) && std::isfinite(proba[1]), message);
  require(proba[0] >= 0.0 && proba[1] >= 0.0, message);
  require_close(proba[0] + proba[1], 1.0, message);
}

void test_naive_bayes_never_prefers_unseen_class()
{
  using Splitter = GaussianSplitter<NUM_FEATURES, 2>;
  Splitter splitter;
  splitter.update(1.0, 0, 1.0);

  std::array<Splitter*, NUM_FEATURES> splitters{};
  splitters[0] = &splitter;
  std::unordered_map<int, double> observed{{0, 1.0}};
  std::vector<double> features(NUM_FEATURES, 1.0);
  std::vector<double> proba(2, 0.0);

  do_naive_bayes_prediction<NUM_FEATURES, 2>(
      proba, features, observed, splitters);

  require_probability_distribution(
      proba, "single-class Naive Bayes output should be finite");
  require(proba[0] > proba[1],
          "Naive Bayes must not prefer an unseen class");
}

void test_naive_bayes_handles_underflow()
{
  using Splitter = GaussianSplitter<NUM_FEATURES, 2>;
  std::array<std::unique_ptr<Splitter>, NUM_FEATURES> owned_splitters;
  std::array<Splitter*, NUM_FEATURES> splitters{};
  for (int i = 0; i < NUM_FEATURES; ++i) {
    auto splitter = std::make_unique<Splitter>();
    splitter->update(0.0, 0, 1.0);
    splitter->update(0.1, 0, 1.0);
    splitter->update(1.0, 1, 1.0);
    splitter->update(1.1, 1, 1.0);
    splitters[i] = splitter.get();
    owned_splitters[i] = std::move(splitter);
  }

  std::unordered_map<int, double> observed{{0, 2.0}, {1, 2.0}};
  std::vector<double> features(NUM_FEATURES, 1.0e100);
  std::vector<double> proba(2, 0.0);

  do_naive_bayes_prediction<NUM_FEATURES, 2>(
      proba, features, observed, splitters);
  require_probability_distribution(
      proba, "underflowed Naive Bayes output should remain normalized");
}

void test_tree_split_preserves_child_stats()
{
  BaseTreeClassifier<1, 2> tree(
      1, 2, 0.1, 1.0, 0.99, 0.01, 7);
  const std::vector<double> cold{0.0};
  const std::vector<double> hot{10.0};

  for (int i = 0; i < 10 && (!tree._root || tree._root->is_leaf); ++i) {
    tree.learn_one(cold, 0);
    tree.learn_one(hot, 1);
  }

  require(tree._root != nullptr && !tree._root->is_leaf,
          "separable stream should split the root leaf");
  auto *branch = static_cast<NumericBinaryBranch<1, 2>*>(tree._root);
  const double branch_weight = sum(branch->stats);
  const double child_weight =
      branch->children[0]->total_weight() +
      branch->children[1]->total_weight();
  require_close(child_weight, branch_weight,
                "split children should preserve the parent class weight");
}

void test_tree_memory_limit_distinguishes_active_leaves()
{
  using Tree = BaseTreeClassifier<1, 2>;
  using Leaf = RandomLeafNaiveBayesAdaptive<1, 2>;
  using Branch = NumericBinaryBranch<1, 2>;

  require(estimate_tree_memory_bytes<1, 2>(static_cast<Tree*>(nullptr)) == 0,
          "null tree memory estimate should be zero");

  Tree empty_tree(1, 2, 0.1, 1.0, 0.99, 0.01, 7);
  const size_t empty_tree_size = estimate_tree_memory_bytes<1, 2>(&empty_tree);
  require(empty_tree_size > 0,
          "empty tree estimate should include the classifier object");
  empty_tree._max_byte_size = 1.0;
  empty_tree._estimate_model_size();
  require(empty_tree._root == nullptr,
          "empty tree memory enforcement should not create a root");

  const size_t active_leaf_size = estimate_leaf_memory_bytes<1, 2>();
  const size_t inactive_leaf_size =
      estimate_inactive_leaf_memory_bytes<1, 2>();
  require(active_leaf_size > inactive_leaf_size,
          "active leaf estimate should include splitter state");

  Tree tree(1, 2, 0.1, 1.0, 0.99, 0.01, 7);
  auto *left = new Leaf(1, 1, 11);
  auto *right = new Leaf(1, 1, 13);
  left->stats[0] = 1.0;
  right->stats[1] = 1.0;
  tree._root = new Branch(0, 0.5, left, right);
  tree._max_byte_size = static_cast<double>(
      empty_tree_size + estimate_branch_memory_bytes<1, 2>() +
      active_leaf_size + inactive_leaf_size);

  tree._estimate_model_size();
  int active_count = 0;
  for (auto *leaf : tree._root->iter_leaves()) {
    active_count += leaf->is_active ? 1 : 0;
  }
  require(active_count == 1,
          "memory limit should deactivate only enough leaves to fit");
}

void test_model_parameter_and_feature_bounds()
{
  require_invalid_argument([] {
    ARFClassifier<NUM_FEATURES, 2> invalid_models(
        0, NUM_FEATURES, 7, 100, 4, 0.001, 0.05, 0.99, 0.01);
  }, "ARF should reject a non-positive tree count");

  require_invalid_argument([] {
    ARFClassifier<NUM_FEATURES, 2> invalid_lambda(
        1, NUM_FEATURES, 7, 100, 0, 0.001, 0.05, 0.99, 0.01);
  }, "ARF should reject a non-positive Poisson lambda");

  BaseTreeClassifier<1, 2> clamped_features(
      0, 2, 0.1, 1.0, 0.99, 0.01, 7);
  const std::vector<double> cold{0.0};
  const std::vector<double> hot{10.0};
  for (int i = 0;
       i < 10 && (!clamped_features._root || clamped_features._root->is_leaf);
       ++i) {
    clamped_features.learn_one(cold, 0);
    clamped_features.learn_one(hot, 1);
  }
  require(clamped_features._root != nullptr &&
          !clamped_features._root->is_leaf,
          "max_features should clamp to at least one feature");

  StandardScaler<2> scaler;
  require_invalid_argument([&] {
    scaler.transform_one(std::vector<double>{1.0});
  }, "scaler should reject a feature vector with the wrong size");

  BaseTreeClassifier<2, 2> fixed_width_tree(
      2, 2, 0.1, 1.0, 0.99, 0.01, 7);
  require_invalid_argument([&] {
    fixed_width_tree.learn_one(std::vector<double>{1.0}, 0);
  }, "tree should reject a feature vector with the wrong size");
}

void test_adwin_bucket_count_and_numeric_state()
{
  AdaptiveWindowing<5> window;
  require(window._calculate_bucket_size(40) == (1ULL << 40),
          "ADWIN bucket size must not overflow 32-bit counters");
  uint64_t detection_signature = 0;
  for (int i = 0; i < 10000; ++i) {
    if (window.update((i / 250) % 2 == 0 ? 0.0 : 1.0)) {
      detection_signature = detection_signature * 1315423911ULL +
          static_cast<uint64_t>(i + 1);
    }
    int physical_bucket_count = 0;
    for (const auto *bucket : window.bucket_deque) {
      physical_bucket_count += bucket->current_idx;
    }
    require(window.n_buckets == physical_bucket_count,
            "ADWIN logical and physical bucket counts should match");
    require(window.width > 0 && std::isfinite(window.total) &&
            std::isfinite(window.variance) && window.variance >= -0.000001,
            "ADWIN numeric state should remain finite");
  }
  require(detection_signature == 8581702740251664352ULL,
          "ADWIN detection decisions should remain stable");
  require(window.width == 256 && window.n_buckets == 25,
          "ADWIN final window structure should remain stable");
}

void test_reusable_probability_output_matches_return_api()
{
  std::unique_ptr<Classifier> model(HeatPredictor::make_model());
  train_seed_probe_model(*model);
  const std::vector<double> probe =
      model_features({3.0, 1.5, 7.0, 5.0, 1.2});
  const std::vector<double> expected = model->predict_proba_one(probe);
  std::vector<double> actual{99.0, 98.0, 97.0};

  model->predict_proba_one_into(probe, actual);
  require_proba_close(
      actual, expected,
      "reusable probability output must preserve prediction results");
}

void test_final_feature_vector()
{
  PredictionSample item = make_item(100, 42);
  item.heat_after_current_access = 1023.0;
  item.heat_label_threshold_at_prediction = 255.0;
  item.tracked_access_count = 7;
  item.time_since_previous_access_ns = 3ULL * 1000 * 1000 * 1000;
  item.long_window_access_count = 3;

  const std::vector<double>& feat = HeatPredictor::to_feat(item);
  require(feat.size() == NUM_FEATURES,
          "final feature vector should match NUM_FEATURES");

  const double time_since_previous_access = hp_log2p1(3.0);
  const double heat_after_current_access = hp_log2p1(1023.0);
  const double long_window_access_count = hp_log2p1(3.0);
  const double threshold_margin =
      hp_log2p1(1023.0) - hp_log2p1(255.0);
  const double heat_concentration = hp_log2p1(
      1023.0 / (HP_HEAT_INCREMENT * 4.0));
  const std::vector<double> expected = {
      threshold_margin, time_since_previous_access, heat_after_current_access,
      long_window_access_count, heat_concentration};

  require(feat.size() >= expected.size(),
          "final feature vector should retain all five base features");
  for (size_t i = 0; i < expected.size(); ++i) {
    require_close(feat[i], expected[i],
                  "final feature vector should preserve feature order");
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

template <typename T, typename = void>
struct has_otsu_confidence_stats : std::false_type {};

template <typename T>
struct has_otsu_confidence_stats<T, std::void_t<
    decltype(std::declval<T>().otsu_histogram_object_count),
    decltype(std::declval<T>().otsu_candidate_threshold),
    decltype(std::declval<T>().otsu_confidence),
    decltype(std::declval<T>().otsu_sharpness_confidence)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_otsu_histogram_sample_count_stat : std::false_type {};

template <typename T>
struct has_otsu_histogram_sample_count_stat<
    T, std::void_t<decltype(std::declval<T>().otsu_histogram_sample_count)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_otsu_sample_confidence_stat : std::false_type {};

template <typename T>
struct has_otsu_sample_confidence_stat<
    T, std::void_t<decltype(std::declval<T>().otsu_sample_confidence)>>
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
  HeatPredictor predictor;
  const HeatPredictorStats stats = predictor.get_evaluation_stats();
  require_close(stats.hot_predict_threshold, HP_HOT_PREDICT_THRESHOLD,
                "predictor should export the initial effective threshold");
  require_close(stats.hot_predict_threshold_target, HP_HOT_PREDICT_THRESHOLD,
                "predictor should export the initial target threshold");
  require(stats.predict_calibration_sample_count == 0,
          "new predictor should have an empty calibration window");
  require_close(stats.predict_calibration_current_accuracy, 0.0,
                "new predictor should report zero current-window accuracy");
  require_close(stats.predict_calibration_target_accuracy, 0.0,
                "new predictor should report zero target-window accuracy");
}

void test_stats_export_otsu_histogram_bin_count()
{
  require(has_otsu_histogram_bin_count_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should expose Otsu histogram bin count");
  require(has_otsu_confidence_stats<HeatPredictorStats>::value,
          "HeatPredictorStats should expose Otsu confidence state");
  require(!has_otsu_histogram_sample_count_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should not call object votes I/O samples");
  require(!has_otsu_sample_confidence_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should not expose removed sample confidence");
}

void test_otsu_update_cost_knobs()
{
  require(HP_OTSU_EAGER_OBJECTS == 0,
          "Otsu eager updates should be disabled for fixed interval updates");
  require(HP_OTSU_UPDATE_INTERVAL == 100,
          "Otsu update interval should refresh every 100 object-vote updates");
  require_close(HP_OTSU_NEAR_OPTIMAL_RATIO, 0.99,
                "Otsu near-optimal candidates should use 99% of best variance");
  require_close(HP_OTSU_SHARPNESS_FULL_AMBIGUOUS_RATIO, 0.20,
                "Otsu sharpness should reach zero when 20% of objects are ambiguous");
  require_close(HP_OTSU_FIXED_EMA_ALPHA, 0.10,
                "fixed-EMA profile should retain its 0.10 control gain");
  require_close(HP_OTSU_CONFIDENCE_MAX_UPDATE_ALPHA, 0.50,
                "confidence profile threshold gain should be capped at 0.50");
  require(HP_OTSU_HISTOGRAM_BIN_COUNT == 10000,
          "future-added-heat Otsu should use exactly 10000 fixed bins");
  require_close(HP_OTSU_LOG1P_HEAT_BIN_WIDTH, 0.01,
                "future-added-heat Otsu log1p bin width should be 0.01");
  require(HP_OTSU_HISTORY_SLOT_COUNT == 60,
          "future-added-heat Otsu should retain 60 one-second slots");
  require(HP_OTSU_HISTORY_SLOT_NS == 1000ULL * 1000 * 1000,
          "each future-added-heat Otsu history slot should span one second");
  require(HP_OTSU_HISTORY_WINDOW_NS == 60ULL * 1000 * 1000 * 1000,
          "future-added-heat Otsu history should span 60 seconds");

  // These constants are retained only for the pending total-heat control.
  require_close(HP_OTSU_HEAT_MIN, 1.0,
                "total-heat control lower bound should remain 1.0");
  require_close(HP_OTSU_HEAT_MAX, 3000.0,
                "total-heat control upper bound should remain 3000.0");
  require_close(HP_OTSU_LOG_HEAT_BIN_WIDTH, 0.05,
                "total-heat control log-heat bin width should remain 0.05");
}

void test_learning_lag_knobs()
{
  require(HP_HEAT_DECAY_HORIZON_NS == HP_FUTURE_LABEL_WINDOW_NS,
          "heat decay horizon should match the 10-second label window");
  require(HP_FUTURE_LABEL_WINDOW_NS == 10ULL * 1000 * 1000 * 1000,
          "evaluation queue should use a fixed 10-second duration");
  require(HP_PENDING_EVALUATION_CAPACITY == 1000000,
          "evaluation queue should bound pending samples per OSD");
  require(HP_LRU_CAPACITY == 1000000,
          "LRU should retain up to one million idle objects per OSD");
  require(HeatPredictor::MODEL_UPDATE_REPORT_INTERVAL == 500,
          "snapshot publish interval should be 500 trained samples");
  require(HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS == 1000ULL * 1000 * 1000,
          "snapshot publication should have a one-second freshness bound");
}

void test_snapshot_publish_uses_sample_or_time_trigger()
{
  HeatPredictor predictor;
  predictor.model_update_train_count = 0;
  predictor.last_snapshot_publish_time_ns = 100;

  require(!predictor.record_model_update_batch(101),
          "one fresh sample should not immediately publish a snapshot");
  require(predictor.record_model_update_batch(
              100 + HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS),
          "a trained model should publish after the freshness interval");
  require(predictor.model_update_train_count == 0,
          "time-triggered snapshot publication should reset the sample count");
}

void test_training_batch_is_bounded_and_fifo()
{
  std::queue<TrainingSample> source;
  for (uint64_t index = 1; index <= 150; ++index) {
    source.push(TrainingSample{make_item(index, index), 0, 1.0});
  }

  std::queue<TrainingSample> batch =
      HeatPredictor::take_training_batch(source);
  require(batch.size() == static_cast<size_t>(HeatPredictor::BATCH_SIZE),
          "one training batch must not exceed BATCH_SIZE");
  require(source.size() == 50,
          "bounded dequeue must leave excess samples in the shared queue");
  require(batch.front().item.io_sequence == 1,
          "bounded dequeue must preserve the oldest training sample");
  uint64_t last_batch_index = 0;
  while (!batch.empty()) {
    last_batch_index = batch.front().item.io_sequence;
    batch.pop();
  }
  require(last_batch_index == 100 && source.front().item.io_sequence == 101,
          "bounded dequeue must preserve FIFO order across batches");
}

void test_training_tail_sample_wakes_worker()
{
  HeatPredictor predictor;
  predictor.ensure_started();
  predictor.enqueue_training_sample(
      TrainingSample{make_item(1, 1), 0, 1.0});

  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(2);
  while (predictor.get_train_queue_length() != 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(predictor.get_train_queue_length() == 0,
          "a tail batch smaller than BATCH_SIZE must wake the worker");
}

void train_seed_probe_model(Classifier& model)
{
  const std::vector<double> hot_a =
      model_features({6.0, 4.0, 1.0, 5.0, 0.9, 3.0, 2.0});
  const std::vector<double> hot_b =
      model_features({5.0, 3.5, 2.0, 7.0, 0.7, 2.5, 1.5});
  const std::vector<double> cold_a =
      model_features({1.0, 0.0, 10.0, 80.0, 0.02, 0.2, 0.1});
  const std::vector<double> cold_b =
      model_features({2.0, 0.2, 6.0, 50.0, 0.05, 0.4, 0.2});

  for (int i = 0; i < 180; ++i) {
    model.learn_one(hot_a, 1, 1.0);
    model.learn_one(cold_a, 0, 1.0);
    if (i % 3 == 0) {
      model.learn_one(hot_b, 1, 2.0);
    }
    if (i % 5 == 0) {
      model.learn_one(cold_b, 0, 1.0);
    }
  }
}

void test_model_parameters_are_configured()
{
  constexpr int expected_features = 5
#if HP_ENABLE_ACCESS_RATE_CHANGE
      + 1
#endif
#if HP_ENABLE_HEAT_PERCENTILE
      + 1
#endif
      ;
  require(NUM_FEATURES == expected_features,
          "model feature count should match the enabled feature profile");
  require(HP_ARF_N_MODELS == 25,
          "ARF should use 25 trees for better ensemble stability");
  require(HP_ARF_MAX_FEATURES == NUM_FEATURES,
          "final ARF should consider every feature at a split");
  require(HP_ARF_SEED == 591422,
          "ARF seed should remain explicit and reproducible");
  require(HP_ARF_GRACE_PERIOD == 100,
          "ARF grace period should remain at the current split cadence");
  require(HP_ARF_LAMBDA == 4,
          "ARF online bagging lambda should remain unchanged");
}

void test_model_seed_is_reproducible()
{
  std::unique_ptr<Classifier> first(HeatPredictor::make_model());
  std::unique_ptr<Classifier> second(HeatPredictor::make_model());
  train_seed_probe_model(*first);
  train_seed_probe_model(*second);

  const std::vector<double> probe_hot =
      model_features({7.0, 4.5, 1.0, 6.0, 1.0, 3.5, 2.5});
  const std::vector<double> probe_cold =
      model_features({1.0, 0.0, 12.0, 90.0, 0.01, 0.1, 0.05});
  require_proba_close(
      first->predict_proba_one(probe_hot),
      second->predict_proba_one(probe_hot),
      "same seed and training sequence should produce same hot probability");
  require_proba_close(
      first->predict_proba_one(probe_cold),
      second->predict_proba_one(probe_cold),
      "same seed and training sequence should produce same cold probability");
}

void test_prediction_clone_is_independent()
{
  std::unique_ptr<Classifier> model(HeatPredictor::make_model());
  const std::vector<double> hot =
      model_features({4.0, 3.0, 1.0, 5.0, 0.8, 2.0, 1.0});
  const std::vector<double> cold =
      model_features({1.0, 0.0, 6.0, 20.0, 0.1, 0.2, 0.1});

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

class PredictionFailureProbeClassifier : public Classifier {
public:
  enum class Mode {
    throws,
    nan_probability,
    out_of_range_probability,
    wrong_class_count
  };

  explicit PredictionFailureProbeClassifier(Mode mode) : mode(mode) {}

  void learn_one(const std::vector<double>&, int, double) override {}

  std::vector<double> predict_proba_one(
      const std::vector<double>&) override {
    switch (mode) {
    case Mode::throws:
      throw std::runtime_error("prediction probe failure");
    case Mode::nan_probability:
      return {0.0, std::numeric_limits<double>::quiet_NaN()};
    case Mode::out_of_range_probability:
      return {-0.1, 1.1};
    case Mode::wrong_class_count:
      return {1.0};
    }
    throw std::runtime_error("unreachable prediction probe mode");
  }

  std::unique_ptr<Classifier> clone_for_prediction() const override {
    return std::make_unique<PredictionFailureProbeClassifier>(mode);
  }

private:
  Mode mode;
};

void test_prediction_exception_fails_open_and_cancels_evaluation()
{
  HeatPredictor predictor;
  predictor.publish_prediction_snapshot(
      std::make_shared<PredictionFailureProbeClassifier>(
          PredictionFailureProbeClassifier::Mode::throws));
  uint64_t io_sequence = 0;
  int result = 1;
  bool threw = false;

  try {
    result = predictor.predict(1, 11, 111, &io_sequence);
  } catch (...) {
    threw = true;
  }

  require(!threw, "prediction exceptions must not escape into the IO path");
  require(result == 0, "prediction exceptions should fail open as cold");
  require(io_sequence == 1,
          "failed prediction should retain the admitted IO sequence");
  require(predictor.get_predict_error_count() == 1,
          "prediction exceptions should increment the error counter");
  {
    std::lock_guard<std::mutex> lock(predictor.eq_mutex);
    require(predictor.eq->pending_evaluations.empty(),
            "failed prediction should cancel its evaluation sample");
  }
  require(predictor.get_train_queue_length() == 0,
          "failed prediction should not enqueue training work");
  require(predictor.get_total_weight() == 0,
          "failed prediction should not enter evaluation metrics");
  require(predictor.get_eval_drop_count() == 1,
          "failed prediction should be counted as a dropped evaluation");
}

void test_invalid_prediction_probabilities_fail_open()
{
  const std::vector<PredictionFailureProbeClassifier::Mode> modes = {
      PredictionFailureProbeClassifier::Mode::nan_probability,
      PredictionFailureProbeClassifier::Mode::out_of_range_probability,
      PredictionFailureProbeClassifier::Mode::wrong_class_count};

  for (size_t i = 0; i < modes.size(); ++i) {
    HeatPredictor predictor;
    predictor.publish_prediction_snapshot(
        std::make_shared<PredictionFailureProbeClassifier>(modes[i]));
    uint64_t io_sequence = 0;

    int result = predictor.predict(
        1, static_cast<uint64_t>(20 + i),
        static_cast<uint64_t>(200 + i), &io_sequence);

    require(result == 0, "invalid probabilities should fail open as cold");
    require(io_sequence == 1,
            "invalid probability should retain the admitted IO sequence");
    require(predictor.get_predict_error_count() == 1,
            "invalid probability should increment the error counter");
    std::lock_guard<std::mutex> lock(predictor.eq_mutex);
    require(predictor.eq->pending_evaluations.empty(),
            "invalid probability should cancel its evaluation sample");
    require(predictor.eq->evaluation_drop_count() == 1,
            "invalid probability should count one dropped evaluation");
  }
}

void test_predictor_enable_disable_resets_and_gates_io()
{
  HeatPredictor predictor;
  uint64_t index = 0;

  require(predictor.is_enabled(),
          "heat predictor should be enabled by default");
  predictor.predict(1, 1, 1, &index);
  require(index == 1, "enabled predictor should process the first IO");
  require(predictor.processed_io_count.load() == 1,
          "enabled predictor should count processed IO");

  uint64_t discarded = predictor.set_enabled(false);
  require(discarded == 1,
          "disable should reset and report discarded pending IO");
  require(!predictor.is_enabled(),
          "disable should turn off the predictor");
  require(predictor.processed_io_count.load() == 0,
          "disable should reset processed IO count");
  HeatPredictorStats disabled_stats = predictor.get_evaluation_stats();
  require_close(disabled_stats.heat_label_threshold, HP_HEAT_INCREMENT,
                "disable should restore the initial heat threshold");
  require_close(disabled_stats.otsu_candidate_threshold, 0.0,
                "disable should clear the Otsu candidate");
  require_close(disabled_stats.otsu_confidence, 0.0,
                "disable should clear Otsu confidence");
  require(disabled_stats.hot_threshold_method ==
              HP_THRESHOLD_METHOD_INITIALIZING,
          "disable should restore the initializing threshold state");

  index = 99;
  predictor.predict(1, 1, 1, &index);
  require(index == 0,
          "disabled predictor should report no processed IO index");
  require(predictor.processed_io_count.load() == 0,
          "disabled predictor should not count IO");
  require(predictor.get_pending_io_count() == 0,
          "disabled predictor should not enqueue IO");

  predictor.predict(1, 1, 1, nullptr);
  uint64_t enabled_discarded = predictor.set_enabled(true);
  require(enabled_discarded == 0,
          "enable should reset even when disabled left no pending IO");
  require(predictor.is_enabled(),
          "enable should turn on the predictor");
  require(predictor.processed_io_count.load() == 0,
          "enable should reset processed IO count");
  HeatPredictorStats enabled_stats = predictor.get_evaluation_stats();
  require_close(enabled_stats.heat_label_threshold, HP_HEAT_INCREMENT,
                "enable should restore the initial heat threshold");
  require_close(enabled_stats.otsu_candidate_threshold, 0.0,
                "enable should clear the Otsu candidate");
  require_close(enabled_stats.otsu_confidence, 0.0,
                "enable should clear Otsu confidence");
  require(enabled_stats.hot_threshold_method ==
              HP_THRESHOLD_METHOD_INITIALIZING,
          "enable should restore the initializing threshold state");

  predictor.predict(1, 1, 1, &index);
  require(index == 1,
          "predictor should process IO again after enable");
}

void test_concurrent_predict_preserves_index_and_evaluation_counts()
{
  HeatPredictor predictor;
  predictor.eq = std::make_unique<EvaluationQueue>(
    HP_HEAT_DECAY_HORIZON_NS,
    HP_LRU_CAPACITY,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    HP_SHORT_ACCESS_WINDOW_NS,
    1ULL << 62,
    HP_PENDING_EVALUATION_CAPACITY);
  constexpr int thread_count = 8;
  constexpr int operations_per_thread = 1500;
  constexpr uint64_t total_operations =
      thread_count * operations_per_thread;
  std::atomic<bool> start{false};
  std::array<std::vector<uint64_t>, thread_count> indices;
  std::vector<std::thread> workers;
  workers.reserve(thread_count);

  for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
    indices[thread_id].reserve(operations_per_thread);
    workers.emplace_back([&, thread_id] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < operations_per_thread; ++i) {
        uint64_t index = 0;
        predictor.predict(
            1,
            static_cast<uint64_t>(thread_id + 1),
            static_cast<uint64_t>(i + 1),
            &index);
        indices[thread_id].push_back(index);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }

  std::vector<uint64_t> all_indices;
  all_indices.reserve(total_operations);
  for (const auto& thread_indices : indices) {
    all_indices.insert(
        all_indices.end(), thread_indices.begin(), thread_indices.end());
  }
  std::sort(all_indices.begin(), all_indices.end());
  require(all_indices.size() == total_operations,
          "concurrent prediction should return one index per IO");
  for (uint64_t i = 0; i < total_operations; ++i) {
    require(all_indices[i] == i + 1,
            "concurrent prediction indices must be unique and contiguous");
  }

  HeatPredictorStats stats = predictor.get_evaluation_stats();
  require(stats.io_count == total_operations,
          "concurrent prediction should count every IO");
  require(stats.pending_io_count == total_operations,
          "IO count alone must not expire a time-window sample");
  require(stats.labeled_io_total == 0,
          "a future time deadline should keep every fast probe IO pending");
  require(stats.eval_drop_count == 0,
          "concurrent probe should remain below the pending admission limit");
  require(stats.io_count ==
          stats.pending_io_count + stats.labeled_io_total +
              stats.eval_drop_count,
          "concurrent prediction counts should preserve IO conservation");
  require(predictor.get_train_drop_count() == 0,
          "concurrent prediction should not drop training samples");

  predictor.reset();
}

void test_dedicated_expiry_worker_expires_idle_time_window()
{
  HeatPredictor predictor;
  expiry_progress_callback_count.store(0, std::memory_order_relaxed);
  predictor.set_expiry_progress_callback(record_expiry_progress);
  predictor.eq = std::make_unique<EvaluationQueue>(
    HP_HEAT_DECAY_HORIZON_NS,
    HP_LRU_CAPACITY,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    HP_SHORT_ACCESS_WINDOW_NS,
    1000 * 1000, // 1 ms
    HP_PENDING_EVALUATION_CAPACITY);

  predictor.predict(1, 1, 1, nullptr);
  require(predictor.expiry_thread.joinable(),
          "time-window expiry should use a dedicated background worker");
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(2);
  while (predictor.get_pending_io_count() != 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  HeatPredictorStats stats = predictor.get_evaluation_stats();
  require(stats.pending_io_count == 0 && stats.labeled_io_total == 1,
          "dedicated expiry worker should expire an idle due sample");
  require(expiry_progress_callback_count.load(std::memory_order_relaxed) == 1,
          "expiry worker should report each completed deadline to its adapter");
  predictor.set_expiry_progress_callback(nullptr);
  predictor.reset();
}

void test_concurrent_enable_disable_does_not_reuse_inflight_slots()
{
  HeatPredictor predictor;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  for (int thread_id = 0; thread_id < 4; ++thread_id) {
    workers.emplace_back([&, thread_id] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < 500; ++i) {
        predictor.predict(
            1,
            static_cast<uint64_t>(thread_id + 1),
            static_cast<uint64_t>(i + 1),
            nullptr);
      }
    });
  }

  std::thread toggler([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < 5; ++i) {
      predictor.set_enabled(false);
      predictor.set_enabled(true);
    }
  });

  start.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }
  toggler.join();

  predictor.set_enabled(true);
  uint64_t index = 0;
  predictor.predict(1, 1, 1, &index);
  require(index == 1,
          "final enable should discard every in-flight generation");
  HeatPredictorStats stats = predictor.get_evaluation_stats();
  require(stats.io_count == 1 && stats.pending_io_count == 1 &&
          stats.labeled_io_total == 0,
          "post-enable EQ should contain only the new generation");
  predictor.reset();
}

void test_future_added_heat_label_excludes_entry_heat()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000,  // heat_decay_horizon_ns
    8,      // lru_capacity
    50.0,   // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    duration,
    8);     // pending_evaluation_capacity

  PredictionSample first = make_item(1, 1);
  eq.prepare_features(first, 10);
  require(eq.enqueue(first, 10), "first item should stay pending");

  auto evaluated = eq.expire_due_evaluations(10 + duration);
  require(evaluated.size() == 1,
          "the first item should reach its time deadline");
  require(evaluated.front().future_window_access_count == 0,
          "item should have no future accesses");
  require_close(evaluated.front().future_window_added_heat, 0.0,
                "future added heat must exclude entry heat and history");
  require(evaluated.front().label == 0,
          "an item without future heat should be labeled cold");
}

void test_future_added_heat_label_includes_later_access()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000,  // heat_decay_horizon_ns
    8,      // lru_capacity
    50.0,   // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    duration,
    8);     // pending_evaluation_capacity

  PredictionSample first = make_item(1, 1);
  eq.prepare_features(first, 10);
  require(eq.enqueue(first, 10), "first item should stay pending");

  PredictionSample later = make_item(2, 1);
  eq.prepare_features(later, 510);

  auto evaluated = eq.expire_due_evaluations(10 + duration);
  require(evaluated.size() == 1,
          "the first item should reach its time deadline");
  require(evaluated.front().future_window_access_count == 1,
          "the later access should be counted in the future window");
  require_close(
      evaluated.front().future_window_added_heat,
      eq.decay_heat(100.0, 510, 10 + duration),
      "future added heat should contain only the later access contribution");
  require(evaluated.front().label == 1,
          "sufficient future added heat should be labeled hot");
}

void test_delayed_expiry_uses_exact_deadline_heat()
{
  constexpr uint64_t duration = 1000;
  constexpr uint64_t enqueue_time = 10;
  constexpr uint64_t later_access_time = 510;
  constexpr double threshold = 85.0;

  const auto make_queue = [&] {
    return std::make_unique<EvaluationQueue>(
      10000, 8, threshold, 100.0, HP_SHORT_ACCESS_WINDOW_NS,
      duration, 8);
  };
  auto on_time = make_queue();
  auto delayed = make_queue();

  for (EvaluationQueue *eq : {on_time.get(), delayed.get()}) {
    PredictionSample first = make_item(1, 1);
    eq->prepare_features(first, enqueue_time);
    require(eq->enqueue(first, enqueue_time),
            "deadline stability probe should admit the first item");

    PredictionSample later = make_item(2, 1);
    eq->prepare_features(later, later_access_time);
  }

  auto exact = on_time->expire_due_evaluations(enqueue_time + duration);
  auto late = delayed->expire_due_evaluations(enqueue_time + duration + 1000);
  require(exact.size() == 1 && late.size() == 1,
          "both deadline stability probes should complete one item");
  require_close(
      late.front().future_window_added_heat,
      exact.front().future_window_added_heat,
      "scheduler delay must not change heat measured at the exact deadline");
  require(late.front().label == exact.front().label,
          "scheduler delay must not change the future-window label");
}

void test_evaluation_queue_expires_by_time()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000,  // heat_decay_horizon_ns
    8,      // lru_capacity
    110.0,  // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    duration,
    8);     // pending_evaluation_capacity

  PredictionSample first = make_item(1, 1);
  eq.prepare_features(first, 100);
  require(eq.enqueue(first, 100),
          "first time-window item should be admitted");

  require(eq.expire_due_evaluations(1099).empty(),
          "elapsed time must not expire an item before its deadline");
  auto expired = eq.expire_due_evaluations(1100);
  require(expired.size() == 1 && expired.front().item.io_sequence == 1,
          "item should expire exactly at its time deadline");
  require(expired.front().label == 0,
          "item without a future access should remain cold");
}

void test_evaluation_queue_reports_head_expiry_schedule()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 8);

  auto empty = eq.expiry_schedule(100);
  require(empty.state == EvaluationQueue::ExpiryScheduleState::empty,
          "empty EQ should wait for a new head");

  PredictionSample item = make_item(1, 1);
  eq.prepare_features(item, 100);
  auto reservation = eq.reserve_prediction(item, 100);
  require(reservation.accepted,
          "expiry schedule item should be admitted");

  auto waiting = eq.expiry_schedule(1099);
  require(waiting.state ==
              EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
              waiting.deadline_ns == 1100,
          "head schedule should expose its exact monotonic deadline");

  auto due = eq.expiry_schedule(1100);
  require(due.state == EvaluationQueue::ExpiryScheduleState::due,
          "deadline should become due without waiting for prediction");
  require(eq.expire_due_evaluations(1100).empty(),
          "label-only completion should not emit a training sample");
  require(eq.expiry_schedule(1100).state ==
              EvaluationQueue::ExpiryScheduleState::due,
          "the access-window event should remain due after label completion");
  eq.expire_due_access_windows(1100);
  auto after_label = eq.expiry_schedule(1100);
  require(after_label.state ==
              EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
          after_label.deadline_ns == 100 + HP_SHORT_ACCESS_WINDOW_NS,
          "labeled sample should leave only its later short-window cleanup");
  auto completed = eq.complete_prediction(reservation.position, 0.25, 0);
  require(completed.size() == 1,
          "late prediction should rendezvous with the completed label");
}

void test_evaluation_queue_expires_before_current_access_updates_heat()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 8);

  PredictionSample first = make_item(1, 1);
  eq.prepare_features(first, 100);
  require(eq.enqueue(first, 100), "first ordering item should be admitted");

  PredictionSample current = make_item(2, 1);
  auto expired = eq.expire_before_prepare(current, 1100);
  require(expired.size() == 1,
          "deadline guard should expire the old item");
  require(expired.front().future_window_access_count == 0,
          "current access must not enter the old item's future window");
  require(current.tracked_access_count == 2,
          "current access should update heat state after expiration");
}

void test_evaluation_queue_expires_prediction_complete_batch()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 8);

  for (uint64_t index = 1; index <= 3; ++index) {
    PredictionSample item = make_item(index, index);
    eq.prepare_features(item, 50);
    require(eq.enqueue(item, 50),
            "batch probe item should be admitted");
  }

  auto expired = eq.expire_due_evaluations(1050);
  require(expired.size() == 3,
          "all prediction_complete items sharing a deadline should expire in one batch");
  require(eq.pending_size() == 0,
          "batch expiration should drain every due item");
}

void test_evaluation_queue_labels_past_unfinished_prediction()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 8);

  PredictionSample first = make_item(1, 1);
  eq.prepare_features(first, 10);
  auto first_reservation = eq.reserve_prediction(first, 10);
  require(first_reservation.accepted,
          "first staged item should be admitted");

  PredictionSample second = make_item(2, 2);
  eq.prepare_features(second, 10);
  auto second_reservation = eq.reserve_prediction(second, 10);
  require(second_reservation.accepted,
          "second staged item should be admitted");
  eq.complete_prediction(second_reservation.position, 0.8, 1);

  auto on_time = eq.expire_due_evaluations(1010);
  require(on_time.size() == 1 &&
              on_time.front().item.io_sequence == second.io_sequence,
          "a later prediction-complete sample should be labeled on time");
  require(eq.pending_size() == 0,
          "deadline completion should release both samples from EQ capacity");
  auto late_first =
      eq.complete_prediction(first_reservation.position, 0.2, 0);
  require(late_first.size() == 1 &&
              late_first.front().item.io_sequence == first.io_sequence,
          "late oldest prediction should emit its stored label exactly once");
}

void test_labeled_sample_waiting_for_prediction_releases_eq_capacity()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 1);

  PredictionSample first = make_item(1, 1);
  eq.prepare_features(first, 10);
  auto first_reservation = eq.reserve_prediction(first, 10);
  require(first_reservation.accepted,
          "first staged item should fill the one-entry EQ");

  require(eq.expire_due_evaluations(1010).empty(),
          "label completion alone should not emit a training sample");
  require(eq.pending_size() == 0,
          "a completed deadline should stop consuming EQ capacity");
  require(eq.awaiting_prediction_size() == 1,
          "label-complete sample should remain visible as awaiting prediction");

  PredictionSample second = make_item(2, 2);
  eq.prepare_features(second, 1010);
  auto second_reservation = eq.reserve_prediction(second, 1010);
  require(second_reservation.accepted,
          "new sample should be admitted while the old prediction is unfinished");

  auto late_first =
      eq.complete_prediction(first_reservation.position, 0.2, 0);
  require(late_first.size() == 1,
          "late prediction should complete the detached rendezvous sample");
  require(eq.awaiting_prediction_size() == 0,
          "prediction rendezvous should remove the completed join state");
  eq.complete_prediction(second_reservation.position, 0.8, 1);
}

void test_unfinished_prediction_does_not_hide_next_deadline()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 2);

  PredictionSample first = make_item(1, 1);
  eq.prepare_features(first, 10);
  auto first_reservation = eq.reserve_prediction(first, 10);

  PredictionSample second = make_item(2, 2);
  eq.prepare_features(second, 20);
  auto second_reservation = eq.reserve_prediction(second, 20);
  eq.complete_prediction(second_reservation.position, 0.8, 1);

  require(eq.expire_due_evaluations(1010).empty(),
          "first label should wait for its unfinished prediction");
  eq.expire_due_access_windows(1010);
  auto next = eq.expiry_schedule(1010);
  require(next.state == EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
              next.deadline_ns == 1020,
          "label-only completion must reveal the next deadline");

  eq.complete_prediction(first_reservation.position, 0.2, 0);
}

void test_evaluation_queue_bounds_pending_without_leaking_heat_state()
{
  constexpr uint64_t duration = 1000;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 2);

  for (uint64_t index = 1; index <= 2; ++index) {
    PredictionSample item = make_item(index, index);
    eq.prepare_features(item, 0);
    require(eq.enqueue(item, 0),
            "items within the pending limit should be admitted");
  }

  PredictionSample dropped = make_item(3, 3);
  eq.prepare_features(dropped, 0);
  require(!eq.enqueue(dropped, 0),
          "an item beyond the pending limit should skip evaluation admission");
  require(eq.evaluation_drop_count() == 1,
          "evaluation admission drop should be counted");
  require(eq.pending_size() == 2,
          "admission drop must not evict an item before its deadline");
  require(eq.heat_state_size() == 3,
          "non-admitted object should remain in bounded heat-state tracking");

  PredictionSample revisit = make_item(4, 3);
  eq.prepare_features(revisit, 0);
  require(revisit.tracked_access_count == 2,
          "non-admitted object should retain a valid reusable heat state");
}

void require_trace_item_equal(
    const PredictionSample& lhs,
    const PredictionSample& rhs,
    const char *message)
{
  require(lhs.io_sequence == rhs.io_sequence && lhs.object_key_hash == rhs.object_key_hash &&
          lhs.tracked_access_count == rhs.tracked_access_count &&
          lhs.time_since_previous_access_ns == rhs.time_since_previous_access_ns &&
          lhs.long_window_access_count == rhs.long_window_access_count &&
          lhs.short_window_access_count == rhs.short_window_access_count &&
          lhs.predicted_label == rhs.predicted_label,
          message);
  require_close(lhs.heat_after_current_access, rhs.heat_after_current_access, message);
  require_close(lhs.heat_label_threshold_at_prediction, rhs.heat_label_threshold_at_prediction, message);
  require_close(lhs.predicted_hot_probability, rhs.predicted_hot_probability, message);
}

void require_evaluated_batches_equal(
    const std::vector<EvaluatedSample>& lhs,
    const std::vector<EvaluatedSample>& rhs,
    const char *message)
{
  require(lhs.size() == rhs.size(), message);
  for (size_t i = 0; i < lhs.size(); ++i) {
    require_trace_item_equal(lhs[i].item, rhs[i].item, message);
    require(lhs[i].label == rhs[i].label &&
            lhs[i].future_window_access_count == rhs[i].future_window_access_count,
            message);
    require_close(lhs[i].training_weight, rhs[i].training_weight, message);
    require_close(lhs[i].future_window_added_heat, rhs[i].future_window_added_heat, message);
  }
}

void test_pending_prediction_slots_preserve_synchronous_sequence()
{
  constexpr uint64_t duration = 3;
  EvaluationQueue synchronous(
    3, 16, 50.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 64);
  EvaluationQueue staged(
    3, 16, 50.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, duration, 64);

  for (uint64_t index = 1; index <= 20; ++index) {
    auto sync_evaluated = synchronous.expire_due_evaluations(index);
    auto staged_evaluated = staged.expire_due_evaluations(index);
    require_evaluated_batches_equal(
        sync_evaluated, staged_evaluated,
        "staged expiration must match synchronous enqueue");

    PredictionSample sync_item = make_item(index, index % 5);
    PredictionSample staged_item = make_item(index, index % 5);
    const int pred = static_cast<int>((index / 3) % 2);
    const double proba = pred ? 0.75 : 0.25;

    synchronous.prepare_features(sync_item, index);
    staged.prepare_features(staged_item, index);
    require_trace_item_equal(
        sync_item, staged_item,
        "staged feature preparation must match synchronous EQ");

    sync_item.predicted_label = pred;
    sync_item.predicted_hot_probability = proba;
    require(synchronous.enqueue(sync_item, index),
            "synchronous item should be admitted");

    auto reservation = staged.reserve_prediction(staged_item, index);
    require(reservation.accepted,
            "staged item should be admitted");
    staged.complete_prediction(reservation.position, proba, pred);
    require_close(
        synchronous.hot_predict_threshold(),
        staged.hot_predict_threshold(),
        "staged threshold feedback must preserve synchronous order");
  }

  auto sync_tail = synchronous.expire_due_evaluations(100);
  auto staged_tail = staged.expire_due_evaluations(100);
  require_evaluated_batches_equal(
      sync_tail, staged_tail,
      "staged tail expiration must match synchronous enqueue");
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

void test_binary_balanced_accuracy_from_counts()
{
  require_close(
      hp_binary_balanced_accuracy(40, 10, 30, 20),
      0.5 * (40.0 / 60.0 + 30.0 / 40.0),
      "binary balanced accuracy should average hot and cold recall");
  require_close(
      hp_binary_balanced_accuracy(0, 0, 90, 0),
      0.50,
      "a missing hot class should contribute zero hot recall");
  require_close(
      hp_binary_balanced_accuracy(10, 90, 10, 90),
      0.10,
      "MGR aggregation should compute from summed confusion counts");
  require_close(
      hp_binary_balanced_accuracy(0, 0, 0, 0),
      0.0,
      "an empty confusion matrix should report zero balanced accuracy");
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

void test_integer_quantile_window_matches_pbds_window()
{
  HpQuantileWindow reference(7);
  HpIntegerQuantileWindow integer_window(7);
  const uint64_t values[] = {
      0, 10, 3, 3, 7, 1, 9, 2, 8, 8, 4, 6, 5, 0, 10
  };

  for (uint64_t value : values) {
    reference.insert(static_cast<double>(value));
    integer_window.insert(value);
    HpDistributionSummary expected = reference.summary();
    HpDistributionSummary actual = integer_window.summary();
    require(actual.count == expected.count,
            "integer quantile count must match PBDS");
    require_close(actual.max, expected.max,
                  "integer quantile max must match PBDS");
    require_close(actual.p50, expected.p50,
                  "integer quantile p50 must match PBDS");
    require_close(actual.p90, expected.p90,
                  "integer quantile p90 must match PBDS");
    require_close(actual.p95, expected.p95,
                  "integer quantile p95 must match PBDS");
    require_close(actual.p99, expected.p99,
                  "integer quantile p99 must match PBDS");
  }

  integer_window.clear();
  require(integer_window.summary().count == 0,
          "cleared integer quantile window must be empty");
}

void test_integer_quantile_window_accepts_large_access_counts()
{
  HpIntegerQuantileWindow window(4);
  bool accepted = true;
  try {
    window.insert(10001);
    window.insert(250000);
  } catch (const std::invalid_argument&) {
    accepted = false;
  }

  require(accepted,
          "future-access statistics must not reject counts above 10000");
  HpDistributionSummary summary = window.summary();
  require(summary.count == 2,
          "large future-access counts should remain in the report window");
  require_close(summary.max, 250000.0,
                "large future-access maximum should be reported exactly");
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

void test_prepare_features_tracks_recency()
{
  EvaluationQueue eq(
    16,     // heat_decay_horizon_ns
    8,      // lru_capacity
    5.0,    // heat_label_threshold
    100.0); // heat_increment

  PredictionSample first = make_item(10, 42);
  eq.prepare_features(first, 10);
  require(eq.enqueue(first, 0),
          "first recency probe item should stay pending");
  require(first.tracked_access_count == 1,
          "first object access should initialize tracked_access_count");
  require(first.time_since_previous_access_ns == 0,
          "first object access should have zero elapsed access time");

  PredictionSample second = make_item(15, 42);
  eq.prepare_features(second, 15);
  require(eq.enqueue(second, 0),
          "second recency probe item should stay pending");
  require(second.tracked_access_count == 2,
          "second object access should increment tracked_access_count");
  require(second.time_since_previous_access_ns == 5,
          "second object access should record elapsed monotonic time");

  PredictionSample third = make_item(25, 42);
  eq.prepare_features(third, 25);
  require(eq.enqueue(third, 0),
          "third recency probe item should stay pending");
  require(third.tracked_access_count == 3,
          "third object access should increment tracked_access_count");
  require(third.time_since_previous_access_ns == 10,
          "third object access should update elapsed monotonic time");
}

void test_time_domain_heat_decay_and_recency()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  EvaluationQueue eq(
    HP_HEAT_DECAY_HORIZON_NS,
    8,
    5.0,
    100.0);

  PredictionSample first = make_item(1, 42);
  eq.prepare_features(first, second_ns);
  require(first.time_since_previous_access_ns == 0,
          "first object access should have zero elapsed access time");

  PredictionSample second = make_item(999999, 42);
  eq.prepare_features(second, 11 * second_ns);
  require(second.time_since_previous_access_ns == 10 * second_ns,
          "recency must use monotonic elapsed time, not intervening OSD I/O count");
  require_close(second.heat_after_current_access, 110.0,
                "heat should retain one tenth after the 10-second horizon");
}

#if HP_ENABLE_ACCESS_RATE_CHANGE
void test_short_access_window_uses_time_and_counts_prior_io()
{
  constexpr uint64_t short_window_ns = 5ULL * 1000 * 1000 * 1000;
  EvaluationQueue eq(
    4,
    16,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    short_window_ns,
    20ULL * 1000 * 1000 * 1000,
    16);

  PredictionSample first = make_item(1, 7);
  eq.prepare_features(first, 0);
  require(first.short_window_access_count == 0,
          "first access should see no prior short-history access");

  PredictionSample before_deadline = make_item(2, 8);
  eq.prepare_features(before_deadline, short_window_ns - 1);
  require(eq.heat_map.at(7).short_window_access_count == 1,
          "short-history access should remain before its time deadline");

  PredictionSample at_deadline = make_item(3, 7);
  eq.prepare_features(at_deadline, short_window_ns);
  require(at_deadline.short_window_access_count == 0,
          "short-history access should expire exactly at its time deadline");
  require(eq.heat_map.at(8).short_window_access_count == 1,
          "a newer object should remain in the short-history window");
}

void test_long_access_window_uses_time_and_counts_prior_io()
{
  constexpr uint64_t long_window_ns = 20ULL * 1000 * 1000 * 1000;
  EvaluationQueue eq(
    4,
    16,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    HP_SHORT_ACCESS_WINDOW_NS,
    long_window_ns,
    16);

  PredictionSample first = make_item(1, 7);
  eq.prepare_features(first, 0);
  require(first.long_window_access_count == 0,
          "first access should see no prior long-window access");

  PredictionSample before_deadline = make_item(2, 7);
  eq.prepare_features(before_deadline, long_window_ns - 1);
  require(before_deadline.long_window_access_count == 1,
          "long-window access should remain before its time deadline");

  PredictionSample at_deadline = make_item(3, 7);
  eq.prepare_features(at_deadline, long_window_ns);
  require(at_deadline.long_window_access_count == 1,
          "the oldest access should expire exactly at the long-window deadline");
}

void test_long_access_window_counts_eq_admission_drops()
{
  constexpr uint64_t long_window_ns = 20ULL * 1000 * 1000 * 1000;
  EvaluationQueue eq(
    4,
    16,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    HP_SHORT_ACCESS_WINDOW_NS,
    long_window_ns,
    1);

  PredictionSample first = make_item(1, 9);
  eq.prepare_features(first, 0);
  require(eq.enqueue(first, 0), "first EQ sample should be admitted");

  PredictionSample dropped = make_item(2, 9);
  eq.prepare_features(dropped, 1);
  require(!eq.enqueue(dropped, 1), "second EQ sample should be dropped");

  PredictionSample third = make_item(3, 9);
  eq.prepare_features(third, 2);
  require(third.long_window_access_count == 2,
          "strict long window must include an access whose EQ sample was dropped");
}

void test_access_windows_schedule_idle_cleanup()
{
  constexpr uint64_t short_window_ns = 5ULL * 1000 * 1000 * 1000;
  constexpr uint64_t long_window_ns = 20ULL * 1000 * 1000 * 1000;
  constexpr uint64_t start_ns = 100;
  EvaluationQueue eq(
    HP_HEAT_DECAY_HORIZON_NS,
    16,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    short_window_ns,
    long_window_ns,
    16);

  PredictionSample item = make_item(1, 77);
  eq.prepare_features(item, start_ns);

  auto short_schedule = eq.expiry_schedule(start_ns);
  require(short_schedule.state ==
              EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
          short_schedule.deadline_ns == start_ns + short_window_ns,
          "idle maintenance should first schedule the short access deadline");

  eq.expire_due_access_windows(start_ns + short_window_ns);
  require(eq.heat_map.at(77).short_window_access_count == 0 &&
          eq.heat_map.at(77).long_window_access_count == 1,
          "short cleanup should not remove the long-window event early");

  auto long_schedule = eq.expiry_schedule(start_ns + short_window_ns);
  require(long_schedule.state ==
              EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
          long_schedule.deadline_ns == start_ns + long_window_ns,
          "idle maintenance should next schedule the long access deadline");

  eq.expire_due_access_windows(start_ns + long_window_ns);
  require(eq.heat_map.at(77).long_window_access_count == 0 &&
          eq.lru_size() == 1,
          "idle cleanup should release the object to the LRU");
}

void test_access_rate_change_uses_log2p1_rates()
{
  require_close(hp_access_rate_change_log2p1(0, 0), 0.0,
                "two idle windows should have zero access-rate change");
  require_close(hp_access_rate_change_log2p1(5, 10), 0.0,
                "equal per-second access rates should have zero change");
  require(hp_access_rate_change_log2p1(8, 10) > 0.0,
          "a recent access burst should have positive rate change");
  require(hp_access_rate_change_log2p1(2, 10) < 0.0,
          "a recent slowdown should have negative rate change");
}
#endif

#if HP_ENABLE_HEAT_PERCENTILE
void test_heat_percentile_counts_ties_and_replacements()
{
  EvaluationQueue eq(8, 16);
  eq.record_object_heat(1, 100.0, 10);
  eq.record_object_heat(2, 100.0, 10);
  eq.record_object_heat(3, 400.0, 10);

  require_close(eq.object_heat_percentile(1), 2.0 / 3.0,
                "equal heat scores should share an upper-rank percentile");
  require_close(eq.object_heat_percentile(2), 2.0 / 3.0,
                "all ties should receive the same percentile");
  require_close(eq.object_heat_percentile(3), 1.0,
                "maximum heat should have percentile one");

  eq.record_object_heat(1, 800.0, 10);
  require_close(eq.object_heat_percentile(1), 1.0,
                "replacing object heat should update its percentile");
  require_close(eq.object_heat_percentile(99), 0.0,
                "unknown object should have percentile zero");
}
#endif

#if HP_ENABLE_ACCESS_RATE_CHANGE || HP_ENABLE_HEAT_PERCENTILE
void test_optional_features_follow_base_feature_order()
{
  PredictionSample item = make_item(100, 42);
  item.heat_after_current_access = 1023.0;
  item.heat_label_threshold_at_prediction = 255.0;
  item.time_since_previous_access_ns = 3ULL * 1000 * 1000 * 1000;
  item.long_window_access_count = 10;
#if HP_ENABLE_ACCESS_RATE_CHANGE
  item.short_window_access_count = 5;
#endif
#if HP_ENABLE_HEAT_PERCENTILE
  item.heat_percentile = 0.75;
#endif

  const auto& feat = HeatPredictor::to_feat(item);
  size_t next = 5;
#if HP_ENABLE_ACCESS_RATE_CHANGE
  require_close(feat[next++], 0.0,
                "access-rate change should follow the five base features");
#endif
#if HP_ENABLE_HEAT_PERCENTILE
  require_close(feat[next++], 0.75,
                "heat percentile should follow rate change when both exist");
#endif
  require(next == feat.size(),
          "optional feature order should consume the complete feature vector");
}
#endif

void test_default_capacity_parameters()
{
  EvaluationQueue eq;
  require(eq.heat_decay_horizon_ns == HP_HEAT_DECAY_HORIZON_NS,
          "default heat decay should use the configured time horizon");
  require(eq.lru_capacity == HP_LRU_CAPACITY,
          "default LRU capacity should use HP_LRU_CAPACITY");
  require(eq.heat_label_threshold_object_capacity == HP_HEAT_LABEL_THRESHOLD_OBJECT_CAPACITY,
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
    4,      // heat_decay_horizon_ns
    8,      // lru_capacity
    100.0,  // heat_label_threshold
    100.0); // heat_increment

  eq.record_object_heat(42, 10.0, 1);
  require(eq.threshold_order_stats.size() == 1,
          "first object should create one object heat entry");

  eq.record_object_heat(42, 30.0, 2);
  require(eq.threshold_order_stats.size() == 1,
          "same object should replace object heat instead of appending");

  eq.record_object_heat(7, 20.0, 2);
  require(eq.threshold_order_stats.size() == 2,
          "different object should create a second object heat entry");

  eq.record_object_heat(42, 5.0, 2);
  require(eq.threshold_order_stats.size() == 2,
          "replacing one object should not change object heat entry count");
}

void test_threshold_window_order_has_one_entry_per_object()
{
  EvaluationQueue eq(
    4,      // heat_decay_horizon_ns
    8,      // lru_capacity
    100.0,  // heat_label_threshold
    100.0); // heat_increment

  eq.record_object_heat(1, 10.0, 1);
  eq.record_object_heat(2, 20.0, 2);
  eq.record_object_heat(3, 30.0, 3);
  eq.record_object_heat(2, 40.0, 4);

  require(eq.threshold_order_stats.size() == 3,
          "threshold tree should keep one entry per object");
  require(eq.threshold_order.size() == 3,
          "threshold recency list should not retain stale object entries");
}

void test_future_added_heat_otsu_uses_fixed_timed_histogram()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  HpOtsuHistogram histogram;

  require(histogram.bin_capacity() == 10000,
          "future-added-heat Otsu should use exactly 10000 fixed bins");
  require(histogram.observe(1, 0.0, 0, 0),
          "current zero-added-heat object should enter the histogram");
  require(histogram.observe(2, std::expm1(0.02), 0, 0),
          "a distinct object log1p score should enter a distinct bin");
  require(histogram.size() == 2 && histogram.bin_count() == 2,
          "fixed histogram should count retained samples and occupied bins");

  histogram.advance_to(2 * second_ns);
  require(histogram.size() == 2,
          "skipping an empty second must not discard unexpired samples");
  histogram.advance_to(60 * second_ns);
  require(histogram.empty(),
          "samples should expire at the 60-second history boundary");
}

void test_future_added_heat_otsu_handles_late_samples()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  HpOtsuHistogram histogram;

  require(histogram.observe(1, 10.0, 0, 59 * second_ns),
          "an object vote 59 seconds old should remain in the history window");
  require(!histogram.observe(2, 20.0, 0, 60 * second_ns),
          "an object vote already 60 seconds old should be rejected");
  require(histogram.size() == 0,
          "advancing to 60 seconds should expire the prior boundary sample");
}

void test_future_added_heat_otsu_keeps_latest_vote_per_object()
{
  HpOtsuHistogram histogram;

  require(histogram.observe(1, 10.0, 0, 0),
          "first object vote should enter Otsu history");
  require(histogram.observe(2, 1000.0, 0, 0),
          "second object vote should enter Otsu history");
  require(histogram.size() == 2 && histogram.bin_count() == 2,
          "two objects in distinct bins should create two retained votes");

  require(histogram.observe(1, 1000.0, 0, 0),
          "newer completed vote should replace the same object's old vote");
  require(histogram.size() == 2,
          "repeated I/O for one object must not add another Otsu vote");
  require(histogram.bin_count() == 1,
          "replacing an object vote must remove its old occupied bin");
}

void test_future_added_heat_otsu_expires_latest_object_vote()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  HpOtsuHistogram histogram;

  require(histogram.observe(1, 10.0, 0, 0),
          "initial object vote should enter the first time slot");
  require(histogram.observe(1, 1000.0, second_ns, second_ns),
          "new object vote should move ownership to the latest time slot");

  histogram.advance_to(60 * second_ns);
  require(histogram.size() == 1,
          "expiring the old slot must not remove the object's newer vote");
  histogram.advance_to(61 * second_ns);
  require(histogram.empty(),
          "latest object vote should expire at its own history boundary");
}

void test_future_added_heat_otsu_rejects_older_object_update()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  HpOtsuHistogram histogram;

  require(histogram.observe(1, 1000.0, 10 * second_ns, 10 * second_ns),
          "latest object vote should enter Otsu history");
  require(histogram.observe(2, 1000.0, 10 * second_ns, 10 * second_ns),
          "comparison object should share the same high-heat bin");
  require(!histogram.observe(1, 10.0, 9 * second_ns, 10 * second_ns),
          "an older delayed result must not replace the latest object vote");
  require(histogram.size() == 2 && histogram.bin_count() == 1,
          "rejected older result must leave the latest distribution unchanged");
}

void test_future_added_heat_label_precedes_otsu_observation()
{
  EvaluationQueue eq(
    HP_HEAT_DECAY_HORIZON_NS,
    8,
    1000.0,  // heat_label_threshold
    700.0,   // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    0,       // future_label_window_ns
    8);

  for (uint64_t i = 0; i < 99; ++i) {
    eq.record_future_added_heat(
        1000 + i, i < 50 ? 0.0 : 100.0, 0, 0);
  }
  require_close(eq.heat_label_threshold, 1000.0,
                "99 completed samples should not update the threshold");

  PredictionSample first = make_item(1, 42);
  eq.prepare_features(first, 0);
  require(eq.enqueue(first, 0),
          "label-order probe sample should enter the evaluation queue");
  PredictionSample later = make_item(2, 42);
  eq.prepare_features(later, 0);

  auto evaluated = eq.expire_due_evaluations(0);
  require(evaluated.size() == 1,
          "zero-length test window should complete the queued sample");
  require_close(evaluated.front().future_window_added_heat, 700.0,
                "later same-time access should add one heat increment");
  require(evaluated.front().label == 0,
          "the 100th sample must be labeled by the preceding threshold");
  require(eq.heat_label_threshold < 700.0,
          "the 100th sample should update the threshold only after labeling");
}

void test_otsu_histogram_reports_distribution_shape()
{
  HpOtsuHistogram histogram;
  for (size_t i = 0; i < 500; ++i) {
    require(histogram.observe(2 * i, 10.0, 0, 0),
            "low added-heat object should enter Otsu history");
    require(histogram.observe(2 * i + 1, 1000.0, 0, 0),
            "high added-heat object should enter Otsu history");
  }

  auto result = histogram.otsu_result();
  require(result.has_value(), "clear bimodal samples should produce Otsu result");
  require(result->object_count == 1000,
          "Otsu result should report retained object-vote count");
  require(result->separation > 0.99,
          "clear bimodal samples should have high separation confidence");
  require(result->ambiguous_object_count == 0,
          "an empty gap between two peaks should not make objects ambiguous");
}

void test_otsu_histogram_rejects_insufficient_and_constant_samples()
{
  HpOtsuHistogram insufficient;
  for (size_t i = 0; i + 1 < HP_OTSU_MIN_OBJECTS; ++i) {
    insufficient.observe(i, i % 2 == 0 ? 0.0 : 10.0, 0, 0);
  }
  require(!insufficient.otsu_result().has_value(),
          "Otsu should wait for the minimum completed-sample count");

  HpOtsuHistogram constant;
  for (size_t i = 0; i < 1000; ++i) {
    constant.observe(i, 10.0, 0, 0);
  }
  require(!constant.otsu_result().has_value(),
          "constant heat should not produce an arbitrary Otsu threshold");
}

void test_otsu_histogram_keeps_weak_candidate()
{
  HpOtsuHistogram histogram;
  histogram.observe(0, 0.0, 0, 0);
  for (size_t i = 0; i < 998; ++i) {
    histogram.observe(i + 1, 10.0, 0, 0);
  }
  histogram.observe(999, 100.0, 0, 0);

  auto result = histogram.otsu_result();
  require(result.has_value(),
          "weak nonconstant distributions should return a confidence-scored candidate");
  require(result->separation < 0.60,
          "weak candidate should demonstrate removal of the old separation gate");
}

void test_otsu_histogram_handles_monotonic_distribution()
{
  HpOtsuHistogram histogram;
  for (size_t i = 0; i < 1000; ++i) {
    histogram.observe(i, static_cast<double>(i), 0, 0);
  }

  auto result = histogram.otsu_result();
  require(result.has_value(),
          "monotonic nonconstant heat should still produce a candidate");
  require(std::isfinite(result->threshold_score),
          "monotonic candidate score should remain finite");
  require(result->ambiguous_object_count > 0,
          "multiple near-optimal partitions should report ambiguous objects");
  require(result->ambiguous_object_count <= result->object_count,
          "ambiguous objects must stay within the histogram population");
}

void test_otsu_histogram_clamps_upper_scores()
{
  HpOtsuHistogram histogram;
  histogram.observe(1, std::numeric_limits<double>::max(), 0, 0);
  histogram.observe(2, std::numeric_limits<double>::infinity(), 0, 0);
  require(histogram.size() == 2 && histogram.bin_count() == 1,
          "finite and infinite overflow heat should share the final fixed bin");
}

void test_object_heat_window_is_independent_from_otsu_history()
{
  EvaluationQueue eq(
    10000,  // heat_decay_horizon_ns
    200,    // lru_capacity
    100.0,  // heat_label_threshold
    100.0); // heat_increment
  eq.heat_label_threshold_object_capacity = 2;

  eq.record_object_heat(1, 10.0, 1);
  eq.record_object_heat(2, 30.0, 2);
  eq.record_object_heat(3, 40.0, 3);
  require(eq.threshold_order_stats.size() == 2,
          "threshold tree should evict to its configured capacity");
  require(eq.otsu_histogram.empty(),
          "object current heat must not feed future-added-heat Otsu");
}

void test_otsu_confidence_formula()
{
  HpOtsuResult sharp{0.0, 1.0, 0, 1000};
  require_close(EvaluationQueue::otsu_sharpness_confidence_for(sharp), 1.0,
                "equivalent thresholds that reclassify no samples should have full confidence");
  HpOtsuResult flat{0.0, 1.0, 200, 1000};
  require_close(EvaluationQueue::otsu_sharpness_confidence_for(flat), 0.0,
                "20 percent ambiguous samples should have zero sharpness");

  require_close(EvaluationQueue::otsu_total_confidence_for(1.0, 1.0),
                1.0, "full confidence components should produce full confidence");
  require_close(EvaluationQueue::otsu_total_confidence_for(1.0, 0.0),
                0.0, "zero sharpness should stop threshold movement");
}

void test_time_normalized_ema_gain()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  require_close(EvaluationQueue::ema_gain_for_elapsed(0.10, 0), 0.0,
                "zero elapsed time should produce zero EMA gain");
  require_close(EvaluationQueue::ema_gain_for_elapsed(0.10, second_ns), 0.10,
                "one reference interval should preserve the configured gain");
  require_close(EvaluationQueue::ema_gain_for_elapsed(0.10, 2 * second_ns), 0.19,
                "two reference intervals should compose the EMA gain");
}

void test_future_added_heat_threshold_holds_during_idle_time()
{
  EvaluationQueue eq;
  constexpr uint64_t start_ns = 123;
  require_close(
      eq.heat_label_threshold_at(start_ns + HP_HEAT_DECAY_HORIZON_NS),
      HP_HEAT_INCREMENT,
      "future-added-heat threshold should remain fixed before observations");
  require_close(
      eq.heat_label_threshold_at(start_ns + HP_HEAT_DECAY_HORIZON_NS),
      HP_HEAT_INCREMENT,
      "completed future-added-heat threshold should not decay while idle");
}

void test_otsu_recomputes_after_wall_clock_interval()
{
  EvaluationQueue eq;
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  for (uint64_t i = 0; i < HP_OTSU_MIN_OBJECTS; ++i) {
    const double heat = i < HP_OTSU_MIN_OBJECTS / 2 ? 10.0 : 1000.0;
    const uint64_t timestamp = i * second_ns / (HP_OTSU_MIN_OBJECTS - 1);
    eq.record_future_added_heat(i, heat, timestamp, timestamp);
  }
  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_TRACKING,
          "new observations should recompute Otsu after one wall-clock second");
}

void test_otsu_ema_time_never_moves_backward_for_late_sample()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  constexpr uint64_t current_time = 61 * second_ns;
  EvaluationQueue eq;

  for (uint64_t i = 0; i < HP_OTSU_MIN_OBJECTS; ++i) {
    const double heat = i < HP_OTSU_MIN_OBJECTS / 2 ? 10.0 : 1000.0;
    require(eq.otsu_histogram.observe(
                i, heat, current_time, current_time),
            "EMA monotonicity probe should populate the Otsu histogram");
  }
  eq.update_hot_threshold(current_time);
  require(eq.last_otsu_ema_update_time_ns == current_time,
          "initial Otsu update should establish the current EMA time");

  eq.threshold_observation_count = HP_OTSU_UPDATE_INTERVAL - 1;
  eq.record_future_added_heat(
      1000, 500.0, current_time - 2 * second_ns, current_time);
  require(eq.last_otsu_ema_update_time_ns == current_time,
          "a late observation must not move the Otsu EMA clock backward");
}

void test_otsu_threshold_state_machine()
{
  EvaluationQueue initializing;
  for (uint64_t i = 0; i + 1 < HP_OTSU_MIN_OBJECTS; ++i) {
    initializing.record_future_added_heat(
        i, i % 2 == 0 ? 10.0 : 1000.0, 1, 1);
  }
  initializing.update_hot_threshold(1);
#if HP_OTSU_PROFILE != HP_OTSU_PROFILE_LEGACY
  require(initializing.hot_threshold_method == HP_THRESHOLD_METHOD_INITIALIZING,
          "insufficient samples should keep threshold initialization state");
  require_close(initializing.heat_label_threshold, HP_HEAT_INCREMENT,
                "initializing should keep the configured initial threshold");
#else
  require(initializing.hot_threshold_method == HP_THRESHOLD_METHOD_HOLDING,
          "legacy policy should publish its quantile while Otsu initializes");
  require_close(initializing.heat_label_threshold, 1000.0,
                "legacy initialization should use the p85 object heat");
#endif
  require_close(initializing.otsu_candidate_threshold, 0.0,
                "initializing should not expose a stale candidate");

  EvaluationQueue holding;
  for (uint64_t i = 0; i < 1000; ++i) {
    holding.record_future_added_heat(i, 10.0, 1, 1);
  }
  holding.update_hot_threshold(1);
  require(holding.hot_threshold_method == HP_THRESHOLD_METHOD_HOLDING,
          "constant samples should hold the effective threshold");
#if HP_OTSU_PROFILE != HP_OTSU_PROFILE_LEGACY
  require_close(holding.heat_label_threshold, HP_HEAT_INCREMENT,
                "holding should not replace the effective threshold");
#else
  require_close(holding.heat_label_threshold, 10.0,
                "legacy policy should use its p85 fallback while Otsu is unavailable");
#endif

  EvaluationQueue tracking(
      HP_HEAT_DECAY_HORIZON_NS, 2000, 20.0, 100.0);
  for (uint64_t i = 0; i < 500; ++i) {
    tracking.otsu_histogram.observe(2 * i, 10.0, 1, 1);
    tracking.otsu_histogram.observe(2 * i + 1, 1000.0, 1, 1);
  }
  const uint64_t update_time = 1 + HP_OTSU_EMA_REFERENCE_INTERVAL_NS;
  const double threshold_before_update = tracking.heat_label_threshold;
  tracking.update_hot_threshold(update_time);
  const double initial_score =
      HpOtsuHistogram::score_for_heat(threshold_before_update);
  const double candidate_score =
      HpOtsuHistogram::score_for_heat(tracking.otsu_candidate_threshold);
  const double expected_score = initial_score +
      HP_OTSU_CONFIDENCE_MAX_UPDATE_ALPHA * tracking.otsu_confidence *
      (candidate_score - initial_score);
  require(tracking.hot_threshold_method == HP_THRESHOLD_METHOD_TRACKING,
          "clear bimodal samples should track the Otsu candidate");
  require_close(
      tracking.otsu_candidate_threshold_at(
          update_time + HP_HEAT_DECAY_HORIZON_NS),
      tracking.otsu_candidate_threshold,
      "completed-sample Otsu candidate should remain fixed while idle");
#if HP_OTSU_PROFILE == HP_OTSU_PROFILE_CONFIDENCE
  require(std::abs(tracking.heat_label_threshold - tracking.otsu_candidate_threshold) <=
              std::abs(threshold_before_update -
                       tracking.otsu_candidate_threshold),
          "confidence gain should not move the effective threshold away from its candidate");
  require_close(HpOtsuHistogram::score_for_heat(
                    tracking.heat_label_threshold),
                expected_score,
                "effective threshold should apply confidence-scaled score gain");
#elif HP_OTSU_PROFILE == HP_OTSU_PROFILE_FIXED_EMA
  const double fixed_ema_expected_score = initial_score +
      HP_OTSU_FIXED_EMA_ALPHA * (candidate_score - initial_score);
  require_close(HpOtsuHistogram::score_for_heat(
                    tracking.heat_label_threshold),
                fixed_ema_expected_score,
                "fixed-EMA profile should use the maximum score-space gain");
#else
  const double legacy_expected_score = initial_score +
      HP_LEGACY_OTSU_EMA_ALPHA * (candidate_score - initial_score);
  require_close(HpOtsuHistogram::score_for_heat(
                    tracking.heat_label_threshold),
                legacy_expected_score,
                "legacy policy should use a fixed score-space EMA gain");
#endif
}

void test_otsu_threshold_updates_every_100_observations()
{
  EvaluationQueue eq(
    10000,  // heat_decay_horizon_ns
    200,    // lru_capacity
    100.0,  // heat_label_threshold
    100.0); // heat_increment

  for (uint64_t i = 0; i < 99; ++i) {
    const double heat = 1000.0 * static_cast<double>(i) / 99.0;
    eq.record_future_added_heat(i, heat, 1, 1);
  }
  require_close(eq.heat_label_threshold, 100.0,
                "Otsu threshold should not update before 100 observations");

  eq.record_future_added_heat(99, 1000.0, 1, 1);
  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_TRACKING,
          "Otsu threshold should update on the 100th observation");
}

void test_legacy_otsu_rejects_weak_separation()
{
#if HP_OTSU_PROFILE == HP_OTSU_PROFILE_LEGACY
  EvaluationQueue eq(10000, 2000, 100.0, 100.0);
  eq.record_future_added_heat(0, 0.0, 1, 1);
  for (uint64_t i = 1; i < 999; ++i) {
    eq.record_future_added_heat(i, 10.0, 1, 1);
  }
  eq.record_future_added_heat(999, 100.0, 1, 1);

  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_HOLDING,
          "legacy Otsu should reject candidates below 0.60 separation");
  require_close(eq.heat_label_threshold, 100.0,
                "legacy weak-candidate fallback should hold without object heat");
#endif
}

void test_training_samples_use_unit_weight()
{
  require_close(HP_HOT_CLASS_WEIGHT, 1.0,
                "hot class weight should be fixed at 1.0");

  EvaluationQueue hot_eq(
    100,    // heat_decay_horizon_ns
    8,      // lru_capacity
    50.0,   // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    1,      // future_label_window_ns
    8);     // pending_evaluation_capacity

  PredictionSample first_hot = make_item(1, 42);
  first_hot.predicted_label = 1;
  hot_eq.prepare_features(first_hot, 0);
  require(hot_eq.enqueue(first_hot, 0),
          "first hot weight probe item should stay pending");

  PredictionSample second_hot = make_item(2, 42);
  second_hot.predicted_label = 1;
  hot_eq.prepare_features(second_hot, 0);
  hot_eq.heat_label_threshold = 50.0;
  require(hot_eq.enqueue(second_hot, 0),
          "second hot weight probe item should stay pending");
  auto hot = hot_eq.expire_due_evaluations(1);
  require(hot.size() == 2, "both hot probe items should reach their deadline");
  require(hot.front().label == 1, "first item should be labeled hot");
  require_close(hot.front().training_weight, 1.0,
                "hot sample should use unit weight");

  EvaluationQueue cold_eq(
    1,      // heat_decay_horizon_ns
    8,      // lru_capacity
    150.0,  // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    1,      // future_label_window_ns
    8);     // pending_evaluation_capacity

  PredictionSample first_cold = make_item(1, 1);
  cold_eq.prepare_features(first_cold, 0);
  require(cold_eq.enqueue(first_cold, 0),
          "first cold weight probe item should stay pending");

  PredictionSample second_cold = make_item(2, 2);
  cold_eq.prepare_features(second_cold, 0);
  require(cold_eq.enqueue(second_cold, 0),
          "second cold weight probe item should stay pending");
  auto cold = cold_eq.expire_due_evaluations(1);
  require(cold.size() == 2, "both cold probe items should reach their deadline");
  require(cold.front().label == 0, "first item should be labeled cold");
  require_close(cold.front().training_weight, 1.0,
                "cold sample should use unit weight");
}

void test_evaluation_queue_feeds_supervised_calibrator()
{
  EvaluationQueue eq(
    1,      // heat_decay_horizon_ns
    8,      // lru_capacity
    150.0,  // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    1,      // future_label_window_ns
    8);     // pending_evaluation_capacity

  PredictionSample first = make_item(1, 1);
  first.predicted_label = 1;
  first.predicted_hot_probability = 0.90;
  eq.prepare_features(first, 0);
  require(eq.enqueue(first, 0),
          "first calibration probe item should stay pending");

  PredictionSample second = make_item(2, 2);
  second.predicted_label = 0;
  second.predicted_hot_probability = 0.10;
  eq.prepare_features(second, 0);
  require(eq.enqueue(second, 0),
          "second calibration probe item should stay pending");
  auto evaluated = eq.expire_due_evaluations(1);
  require(evaluated.size() == 2,
          "both calibration probe items should reach their deadline");
  require(evaluated.front().label == 0,
          "first calibration probe item should be actually cold");
#if HP_ENABLE_PREDICTION_CALIBRATION
  require(eq.prediction_calibration_size() == 2,
          "expired I/Os should enter the supervised calibration window");
#else
  require(eq.prediction_calibration_size() == 0,
          "fixed prediction threshold must not retain calibration samples");
#endif
  require_close(eq.hot_predict_threshold(), HP_HOT_PREDICT_THRESHOLD,
                "insufficient calibration samples should keep the initial threshold");
}

void test_supervised_threshold_maximizes_window_accuracy()
{
  HpPredictionThresholdCalibrator calibrator(
    8,    // capacity
    4,    // update_interval
    4,    // min_samples
    0.50, // initial_threshold
    0.40, // min_threshold
    0.60, // max_threshold
    1.0); // ema_alpha

  calibrator.observe(0.49, 0);
  calibrator.observe(0.51, 0);
  calibrator.observe(0.55, 1);
  calibrator.observe(0.56, 1);

  require_close(calibrator.target_threshold(), 0.511,
                "calibrator should choose the closest accuracy-optimal bin");
  require_close(calibrator.threshold(), 0.511,
                "alpha one should publish the target threshold directly");
  require_close(calibrator.current_accuracy(), 1.0,
                "selected threshold should classify the window perfectly");
  require_close(calibrator.target_accuracy(), 1.0,
                "target accuracy should report the selected candidate");
}

void test_experiment_profile_parameters()
{
  require(HP_ENABLE_PREDICTION_CALIBRATION == 0,
          "the baseline should use a fixed 0.50 prediction threshold");
  require(HP_OTSU_PROFILE == HP_OTSU_PROFILE_FIXED_EMA,
          "the baseline should use a fixed 0.10 Otsu EMA gain");
#if HP_PREDICTION_RANGE_PROFILE == HP_PREDICTION_RANGE_CW
  require_close(HP_HOT_PREDICT_THRESHOLD_MIN, 0.20,
                "CW prediction threshold lower bound should be 0.20");
  require_close(HP_HOT_PREDICT_THRESHOLD_MAX, 0.80,
                "CW prediction threshold upper bound should be 0.80");
#else
  require_close(HP_HOT_PREDICT_THRESHOLD_MIN, 0.40,
                "C0 prediction threshold lower bound should be 0.40");
  require_close(HP_HOT_PREDICT_THRESHOLD_MAX, 0.60,
                "C0 prediction threshold upper bound should be 0.60");
#endif
  require(HP_OTSU_PROFILE >= HP_OTSU_PROFILE_LEGACY &&
          HP_OTSU_PROFILE <= HP_OTSU_PROFILE_CONFIDENCE,
          "Otsu experiment profile should be valid");
}

void test_supervised_threshold_is_stable_and_bounded()
{
  HpPredictionThresholdCalibrator stable(
    8, 4, 4, 0.50, 0.40, 0.60, 1.0);
  for (int i = 0; i < 4; ++i) {
    stable.observe(0.90, 1);
  }
  require_close(stable.target_threshold(), 0.50,
                "equal-accuracy candidates should keep the current threshold");

  HpPredictionThresholdCalibrator bounded(
    8, 4, 4, 0.50, 0.40, 0.60, 1.0);
  bounded.observe(0.59, 0);
  bounded.observe(0.599, 0);
  bounded.observe(0.90, 1);
  bounded.observe(0.95, 1);
  require_close(bounded.target_threshold(), 0.60,
                "candidate threshold should respect the configured upper bound");
}

void test_supervised_threshold_applies_ema_and_fifo_eviction()
{
  HpPredictionThresholdCalibrator calibrator(
    4, 4, 4, 0.50, 0.40, 0.60, 0.10);
  calibrator.observe(0.49, 0);
  calibrator.observe(0.51, 0);
  calibrator.observe(0.55, 1);
  calibrator.observe(0.56, 1);
  require_close(calibrator.threshold(), 0.5011,
                "calibrator should smooth the first target with EMA");

  calibrator.observe(0.41, 0);
  calibrator.observe(0.42, 0);
  calibrator.observe(0.45, 1);
  calibrator.observe(0.46, 1);
  require(calibrator.size() == 4,
          "calibration FIFO should stay within its configured capacity");
  require(calibrator.target_threshold() < 0.50,
          "evicted observations must not affect the next target");
  require(calibrator.threshold() < 0.5011,
          "EMA threshold should move toward the new lower target");
}

} // namespace

int main()
{
  test_log2p1_transform();
  test_naive_bayes_never_prefers_unseen_class();
  test_naive_bayes_handles_underflow();
  test_tree_split_preserves_child_stats();
  test_tree_memory_limit_distinguishes_active_leaves();
  test_model_parameter_and_feature_bounds();
  test_adwin_bucket_count_and_numeric_state();
  test_final_feature_vector();
  test_stats_drop_online_prediction_ratio_source();
  test_predictor_drops_unused_actual_label_counters();
  test_stats_export_hot_predict_threshold();
  test_stats_export_otsu_histogram_bin_count();
  test_otsu_update_cost_knobs();
  test_learning_lag_knobs();
  test_snapshot_publish_uses_sample_or_time_trigger();
  test_training_batch_is_bounded_and_fifo();
  test_training_tail_sample_wakes_worker();
  test_model_parameters_are_configured();
  test_model_seed_is_reproducible();
  test_reusable_probability_output_matches_return_api();
  test_prediction_clone_is_independent();
  test_prediction_exception_fails_open_and_cancels_evaluation();
  test_invalid_prediction_probabilities_fail_open();
  test_predictor_enable_disable_resets_and_gates_io();
  test_concurrent_predict_preserves_index_and_evaluation_counts();
  test_dedicated_expiry_worker_expires_idle_time_window();
  test_concurrent_enable_disable_does_not_reuse_inflight_slots();
  test_future_added_heat_label_excludes_entry_heat();
  test_future_added_heat_label_includes_later_access();
  test_otsu_ema_time_never_moves_backward_for_late_sample();
  test_delayed_expiry_uses_exact_deadline_heat();
  test_evaluation_queue_expires_by_time();
  test_evaluation_queue_reports_head_expiry_schedule();
  test_evaluation_queue_expires_before_current_access_updates_heat();
  test_evaluation_queue_expires_prediction_complete_batch();
  test_evaluation_queue_labels_past_unfinished_prediction();
  test_labeled_sample_waiting_for_prediction_releases_eq_capacity();
  test_unfinished_prediction_does_not_hide_next_deadline();
  test_evaluation_queue_bounds_pending_without_leaking_heat_state();
  test_pending_prediction_slots_preserve_synchronous_sequence();
  test_balanced_accuracy_penalizes_missing_hot_class();
  test_binary_balanced_accuracy_from_counts();
  test_quantile_window_keeps_recent_values();
  test_integer_quantile_window_matches_pbds_window();
  test_integer_quantile_window_accepts_large_access_counts();
  test_object_key_uses_object_identity();
  test_prepare_features_tracks_recency();
  test_time_domain_heat_decay_and_recency();
#if HP_ENABLE_ACCESS_RATE_CHANGE
  test_short_access_window_uses_time_and_counts_prior_io();
  test_long_access_window_uses_time_and_counts_prior_io();
  test_long_access_window_counts_eq_admission_drops();
  test_access_windows_schedule_idle_cleanup();
  test_access_rate_change_uses_log2p1_rates();
#endif
#if HP_ENABLE_HEAT_PERCENTILE
  test_heat_percentile_counts_ties_and_replacements();
#endif
#if HP_ENABLE_ACCESS_RATE_CHANGE || HP_ENABLE_HEAT_PERCENTILE
  test_optional_features_follow_base_feature_order();
#endif
  test_default_capacity_parameters();
  test_training_batch_size_is_low_latency();
  test_threshold_window_tracks_object_current_heat();
  test_threshold_window_order_has_one_entry_per_object();
  test_future_added_heat_otsu_uses_fixed_timed_histogram();
  test_future_added_heat_otsu_handles_late_samples();
  test_future_added_heat_otsu_keeps_latest_vote_per_object();
  test_future_added_heat_otsu_expires_latest_object_vote();
  test_future_added_heat_otsu_rejects_older_object_update();
  test_future_added_heat_label_precedes_otsu_observation();
  test_otsu_histogram_reports_distribution_shape();
  test_otsu_histogram_rejects_insufficient_and_constant_samples();
  test_otsu_histogram_keeps_weak_candidate();
  test_otsu_histogram_handles_monotonic_distribution();
  test_otsu_histogram_clamps_upper_scores();
  test_object_heat_window_is_independent_from_otsu_history();
  test_otsu_confidence_formula();
  test_time_normalized_ema_gain();
  test_future_added_heat_threshold_holds_during_idle_time();
  test_otsu_recomputes_after_wall_clock_interval();
  test_otsu_threshold_state_machine();
  test_otsu_threshold_updates_every_100_observations();
  test_legacy_otsu_rejects_weak_separation();
  test_training_samples_use_unit_weight();
  test_evaluation_queue_feeds_supervised_calibrator();
  test_experiment_profile_parameters();
  test_supervised_threshold_maximizes_window_accuracy();
  test_supervised_threshold_is_stable_and_bounded();
  test_supervised_threshold_applies_ema_and_fifo_eviction();
  std::cout << "PASS: hp algorithm probe" << std::endl;
  return 0;
}
