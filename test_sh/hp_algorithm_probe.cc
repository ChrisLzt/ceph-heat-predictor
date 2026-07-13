#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
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

using HpNotifySignature = void (*)(CephContext *, const hobject_t&, uint16_t);
using HpStatusSignature = void (*)(CephContext *, ceph::Formatter *);

static_assert(
    std::is_same_v<decltype(&hp_notify_osd_object_op), HpNotifySignature>,
    "OSD hook must pass only an already validated object op type");
static_assert(
    std::is_same_v<decltype(&hp_dump_osd_object_heat_predictor_status),
                   HpStatusSignature>,
    "OSD status command must expose live heat predictor state");

TraceItem make_item(uint64_t index, uint64_t key)
{
  return TraceItem{
    index,  // index
    key,    // key
    0.0,    // current_heat
    0.0,    // hot_threshold
    0,      // access_count
    0,      // last_access_distance
    0,      // past_window_access_count
    0,      // recent_window_access_count
    0.0,    // heat_percentile
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
  TraceItem item = make_item(100, 42);
  item.current_heat = 1023.0;
  item.hot_threshold = 255.0;
  item.access_count = 7;
  item.last_access_distance = 3;
  item.past_window_access_count = 3;

  const std::vector<double>& feat = HeatPredictor::to_feat(item);
  require(feat.size() == NUM_FEATURES,
          "final feature vector should match NUM_FEATURES");

  const double last_access_distance = hp_log2p1(3.0);
  const double current_heat = hp_log2p1(1023.0);
  const double past_window_access_count = hp_log2p1(3.0);
  const double threshold_margin =
      hp_log2p1(1023.0) - hp_log2p1(255.0);
  const double heat_concentration = hp_log2p1(
      1023.0 / (HP_HEAT_INCREMENT * 4.0));
  const std::vector<double> expected = {
      threshold_margin, last_access_distance, current_heat,
      past_window_access_count, heat_concentration};

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
  require(!has_otsu_sample_confidence_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should not expose removed sample confidence");
}

void test_otsu_update_cost_knobs()
{
  require(HP_OTSU_EAGER_OBJECTS == 0,
          "Otsu eager updates should be disabled for fixed interval updates");
  require(HP_OTSU_UPDATE_INTERVAL == 100,
          "Otsu update interval should refresh every 100 IO observations");
  require_close(HP_OTSU_NEAR_OPTIMAL_RATIO, 0.99,
                "Otsu near-optimal candidates should use 99% of best variance");
  require_close(HP_OTSU_SHARPNESS_FULL_AMBIGUOUS_RATIO, 0.20,
                "Otsu sharpness should reach zero when 20% of samples are ambiguous");
  require_close(HP_OTSU_FIXED_EMA_ALPHA, 0.10,
                "fixed-EMA profile should retain its 0.10 control gain");
  require_close(HP_OTSU_CONFIDENCE_MAX_UPDATE_ALPHA, 0.50,
                "confidence profile threshold gain should be capped at 0.50");
  require_close(HP_OTSU_HEAT_MIN, 1.0,
                "Otsu heat clamp lower bound should be 1.0");
  require_close(HP_OTSU_HEAT_MAX, 3000.0,
                "Otsu heat clamp upper bound should be 3000.0");
  require_close(HP_OTSU_LOG_HEAT_BIN_WIDTH, 0.05,
                "Otsu log-heat histogram bin width should be 0.05");
}

void test_learning_lag_knobs()
{
  require(HP_EVALUATION_WINDOW == 10000,
          "evaluation window should be 10000 for current workload coverage");
  require(HeatPredictor::MODEL_UPDATE_REPORT_INTERVAL == 500,
          "snapshot publish interval should be 500 trained samples");
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
  require(batch.front().item.index == 1,
          "bounded dequeue must preserve the oldest training sample");
  uint64_t last_batch_index = 0;
  while (!batch.empty()) {
    last_batch_index = batch.front().item.index;
    batch.pop();
  }
  require(last_batch_index == 100 && source.front().item.index == 101,
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
#if HP_ENABLE_ACCESS_ACCELERATION
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

void test_predictor_enable_disable_resets_and_gates_io()
{
  HeatPredictor predictor;
  uint64_t index = 0;

  require(predictor.is_enabled(),
          "heat predictor should be enabled by default");
  predictor.predict(1, 1, 1, &index);
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
  HeatPredictorStats disabled_stats = predictor.get_evaluation_stats();
  require_close(disabled_stats.hot_threshold, HP_HEAT_INCREMENT,
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
  require(predictor.hp_index.load() == 0,
          "disabled predictor should not count IO");
  require(predictor.get_pending_io_count() == 0,
          "disabled predictor should not enqueue IO");

  predictor.predict(1, 1, 1, nullptr);
  uint64_t enabled_discarded = predictor.set_enabled(true);
  require(enabled_discarded == 0,
          "enable should reset even when disabled left no pending IO");
  require(predictor.is_enabled(),
          "enable should turn on the predictor");
  require(predictor.hp_index.load() == 0,
          "enable should reset processed IO count");
  HeatPredictorStats enabled_stats = predictor.get_evaluation_stats();
  require_close(enabled_stats.hot_threshold, HP_HEAT_INCREMENT,
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
  require(stats.pending_io_count == HP_EVALUATION_WINDOW,
          "concurrent prediction should retain one evaluation window");
  require(stats.labeled_io_total ==
          total_operations - HP_EVALUATION_WINDOW,
          "concurrent prediction should evaluate every expired IO once");
  require(stats.io_count ==
          stats.pending_io_count + stats.labeled_io_total,
          "concurrent prediction counts should preserve IO conservation");
  require(predictor.get_train_drop_count() == 0,
          "concurrent prediction should not drop training samples");

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

void require_trace_item_equal(
    const TraceItem& lhs,
    const TraceItem& rhs,
    const char *message)
{
  require(lhs.index == rhs.index && lhs.key == rhs.key &&
          lhs.access_count == rhs.access_count &&
          lhs.last_access_distance == rhs.last_access_distance &&
          lhs.past_window_access_count == rhs.past_window_access_count &&
          lhs.pred == rhs.pred,
          message);
  require_close(lhs.current_heat, rhs.current_heat, message);
  require_close(lhs.hot_threshold, rhs.hot_threshold, message);
  require_close(lhs.pred_hot_proba, rhs.pred_hot_proba, message);
}

void require_evaluated_item_equal(
    const std::optional<EvaluatedItem>& lhs,
    const std::optional<EvaluatedItem>& rhs,
    const char *message)
{
  require(lhs.has_value() == rhs.has_value(), message);
  if (!lhs.has_value()) {
    return;
  }
  require_trace_item_equal(lhs->item, rhs->item, message);
  require(lhs->label == rhs->label &&
          lhs->future_access_count == rhs->future_access_count,
          message);
  require_close(lhs->training_weight, rhs->training_weight, message);
  require_close(lhs->future_heat, rhs->future_heat, message);
}

void test_pending_prediction_slots_preserve_synchronous_sequence()
{
  EvaluationQueue synchronous(3, 16, 50.0, 100.0);
  EvaluationQueue staged(3, 16, 50.0, 100.0);

  for (uint64_t index = 1; index <= 20; ++index) {
    TraceItem sync_item = make_item(index, index % 5);
    TraceItem staged_item = make_item(index, index % 5);
    const int pred = static_cast<int>((index / 3) % 2);
    const double proba = pred ? 0.75 : 0.25;

    synchronous.prepare_features(sync_item);
    staged.prepare_features(staged_item);
    require_trace_item_equal(
        sync_item, staged_item,
        "staged feature preparation must match synchronous EQ");

    sync_item.pred = pred;
    sync_item.pred_hot_proba = proba;
    auto sync_evaluated = synchronous.enqueue(sync_item);

    require(staged.can_reserve_prediction(),
            "completed oldest slot should permit the next reservation");
    auto reservation = staged.reserve_prediction(staged_item);
    staged.complete_prediction(reservation.slot, proba, pred);

    require_evaluated_item_equal(
        sync_evaluated, reservation.evaluated,
        "staged evaluation must match synchronous enqueue");
    require_close(
        synchronous.hot_predict_threshold(),
        staged.hot_predict_threshold(),
        "staged threshold feedback must preserve synchronous order");
  }
}

void test_pending_prediction_slots_block_oldest_out_of_order_completion()
{
  EvaluationQueue eq(2, 16, 50.0, 100.0);

  TraceItem first = make_item(1, 1);
  eq.prepare_features(first);
  auto first_reservation = eq.reserve_prediction(first);

  TraceItem second = make_item(2, 2);
  eq.prepare_features(second);
  auto second_reservation = eq.reserve_prediction(second);

  require(!eq.can_reserve_prediction(),
          "full EQ should block while its oldest prediction is incomplete");
  bool notify_for_second =
      eq.complete_prediction(second_reservation.slot, 0.8, 1);
  require(!notify_for_second,
          "completing a non-oldest slot must not notify EQ waiters");
  require(!eq.can_reserve_prediction(),
          "a newer completed prediction must not bypass the oldest slot");

  bool notify_for_first =
      eq.complete_prediction(first_reservation.slot, 0.2, 0);
  require(notify_for_first,
          "completing the full EQ's oldest slot must notify waiters");
  require(eq.can_reserve_prediction(),
          "completing the oldest prediction should release the next index");

  TraceItem third = make_item(3, 3);
  eq.prepare_features(third);
  auto third_reservation = eq.reserve_prediction(third);
  require(third_reservation.evaluated.has_value(),
          "third reservation should evaluate the oldest slot");
  require(third_reservation.evaluated->item.index == 1 &&
          third_reservation.evaluated->item.pred == 0,
          "ordered evaluation must consume the oldest prediction result");
  bool notify_for_third =
      eq.complete_prediction(third_reservation.slot, 0.7, 1);
  require(!notify_for_third,
          "completing a non-oldest replacement slot must not notify waiters");
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

void test_integer_quantile_window_matches_pbds_window()
{
  HpQuantileWindow reference(7);
  HpIntegerQuantileWindow integer_window(7, 10);
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

  require_invalid_argument([&] {
    integer_window.insert(11);
  }, "integer quantile window must reject values outside its domain");
  integer_window.clear();
  require(integer_window.summary().count == 0,
          "cleared integer quantile window must be empty");
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

  TraceItem second = make_item(15, 42);
  eq.prepare_features(second);
  require(!eq.enqueue(second).has_value(),
          "second recency probe item should stay pending");
  require(second.access_count == 2,
          "second object access should increment access_count");
  require(second.last_access_distance == 5,
          "second object access should record distance from previous access");

  TraceItem third = make_item(25, 42);
  eq.prepare_features(third);
  require(!eq.enqueue(third).has_value(),
          "third recency probe item should stay pending");
  require(third.access_count == 3,
          "third object access should increment access_count");
  require(third.last_access_distance == 10,
          "third object access should update distance from previous access");
}

#if HP_ENABLE_ACCESS_ACCELERATION
void test_short_access_window_is_bounded_and_counts_prior_io()
{
  EvaluationQueue eq(4, 16, HP_HEAT_INCREMENT, HP_HEAT_INCREMENT, 2);

  TraceItem first = make_item(1, 7);
  eq.prepare_features(first);
  require(first.recent_window_access_count == 0,
          "first access should see no prior short-window access");
  eq.enqueue(first);

  TraceItem second = make_item(2, 7);
  eq.prepare_features(second);
  require(second.recent_window_access_count == 1,
          "second access should see the first short-window access");
  eq.enqueue(second);

  TraceItem other = make_item(3, 8);
  eq.prepare_features(other);
  require(other.recent_window_access_count == 0,
          "unseen object should have zero short-window accesses");
  eq.enqueue(other);

  TraceItem fourth = make_item(4, 7);
  eq.prepare_features(fourth);
  require(fourth.recent_window_access_count == 1,
          "short-window eviction should decrement the expired object count");
}

void test_access_acceleration_uses_smoothed_window_rates()
{
  require_close(hp_access_acceleration(4, 19), 0.0,
                "equal smoothed access rates should have zero acceleration");
  require(hp_access_acceleration(8, 19) > 0.0,
          "a recent access burst should have positive acceleration");
  require(hp_access_acceleration(1, 19) < 0.0,
          "a recent slowdown should have negative acceleration");
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

#if HP_ENABLE_ACCESS_ACCELERATION || HP_ENABLE_HEAT_PERCENTILE
void test_optional_features_follow_base_feature_order()
{
  TraceItem item = make_item(100, 42);
  item.current_heat = 1023.0;
  item.hot_threshold = 255.0;
  item.last_access_distance = 3;
  item.past_window_access_count = 19;
#if HP_ENABLE_ACCESS_ACCELERATION
  item.recent_window_access_count = 4;
#endif
#if HP_ENABLE_HEAT_PERCENTILE
  item.heat_percentile = 0.75;
#endif

  const auto& feat = HeatPredictor::to_feat(item);
  size_t next = 5;
#if HP_ENABLE_ACCESS_ACCELERATION
  require_close(feat[next++], 0.0,
                "access acceleration should follow the five base features");
#endif
#if HP_ENABLE_HEAT_PERCENTILE
  require_close(feat[next++], 0.75,
                "heat percentile should follow acceleration when both exist");
#endif
  require(next == feat.size(),
          "optional feature order should consume the complete feature vector");
}
#endif

void test_default_capacity_parameters()
{
  EvaluationQueue eq;
  require(eq.evaluation_window == HP_EVALUATION_WINDOW,
          "default evaluation window should use HP_EVALUATION_WINDOW");
  require(eq.lru_capacity == HP_LRU_CAPACITY,
          "default LRU capacity should use HP_LRU_CAPACITY");
  require(eq.threshold_window_capacity == HP_LABEL_THRESHOLD_WINDOW_CAPACITY,
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
    4,      // evaluation_window
    8,      // lru_capacity
    100.0,  // hot_threshold
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

void test_otsu_histogram_reports_distribution_shape()
{
  HpOtsuHistogram histogram;
  for (size_t i = 0; i < 500; ++i) {
    histogram.insert(std::log(10.0));
    histogram.insert(std::log(1000.0));
  }

  auto result = histogram.otsu_result();
  require(result.has_value(), "clear bimodal samples should produce Otsu result");
  require(result->sample_count == 1000,
          "Otsu result should report histogram object count");
  require(result->separation > 0.99,
          "clear bimodal samples should have high separation confidence");
  require(result->ambiguous_sample_count == 0,
          "an empty gap between two peaks should not make samples ambiguous");
}

void test_otsu_histogram_rejects_insufficient_and_constant_samples()
{
  HpOtsuHistogram insufficient;
  for (size_t i = 0; i + 1 < HP_OTSU_MIN_OBJECTS; ++i) {
    insufficient.insert(i % 2 == 0 ? 0.0 : 1.0);
  }
  require(!insufficient.otsu_result().has_value(),
          "Otsu should wait for the minimum object count");

  HpOtsuHistogram constant;
  for (size_t i = 0; i < 1000; ++i) {
    constant.insert(1.0);
  }
  require(!constant.otsu_result().has_value(),
          "constant heat should not produce an arbitrary Otsu threshold");
}

void test_otsu_histogram_keeps_weak_candidate()
{
  HpOtsuHistogram histogram;
  histogram.insert(-1.0);
  for (size_t i = 0; i < 998; ++i) {
    histogram.insert(0.0);
  }
  histogram.insert(1.0);

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
    histogram.insert(static_cast<double>(i) / 100.0);
  }

  auto result = histogram.otsu_result();
  require(result.has_value(),
          "monotonic nonconstant heat should still produce a candidate");
  require(std::isfinite(result->threshold_score),
          "monotonic candidate score should remain finite");
  require(result->ambiguous_sample_count > 0,
          "multiple near-optimal partitions should report ambiguous samples");
  require(result->ambiguous_sample_count <= result->sample_count,
          "ambiguous samples must stay within the histogram population");
}

void test_otsu_histogram_is_stable_under_large_score_translation()
{
  HpOtsuHistogram base;
  HpOtsuHistogram translated;
  constexpr double offset = 1.0e9;
  for (size_t i = 0; i < 500; ++i) {
    const double low = static_cast<double>(i % 10) * 0.05;
    const double high = 2.0 + static_cast<double>(i % 10) * 0.05;
    base.insert(low);
    base.insert(high);
    translated.insert(offset + low);
    translated.insert(offset + high);
  }

  auto base_result = base.otsu_result();
  auto translated_result = translated.otsu_result();
  require(base_result.has_value() && translated_result.has_value(),
          "large common score offset should not invalidate Otsu variance");
  require(std::abs(base_result->separation - translated_result->separation) <
              0.001,
          "Otsu separation should be translation invariant");
  require(std::abs(
              (translated_result->threshold_score - offset) -
              base_result->threshold_score) < 0.1,
          "Otsu candidate should shift by the common score offset");
}

void test_otsu_histogram_tracks_threshold_window_entries()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment
  eq.threshold_window_capacity = 2;

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
  require(eq.threshold_order_stats.size() == 2,
          "threshold tree should evict to its configured capacity");
  require(eq.otsu_histogram.size() == eq.threshold_order_stats.size(),
          "histogram should evict with the threshold tree");
  require(eq.otsu_histogram.bin_count() == eq.threshold_order_stats.size(),
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
  require(eq.threshold_entries_by_key.count(1) == 1,
          "fresh min-heat object should enter the threshold window");
  require(eq.otsu_histogram.size() == 1,
          "histogram should track the fresh threshold object");

  eq.record_object_heat(2, HP_OTSU_HEAT_MIN, 10000);
  require(eq.threshold_entries_by_key.count(1) == 1,
          "threshold window should keep old objects until TW size eviction");
  require(eq.threshold_entries_by_key.count(2) == 1,
          "current min-heat object should remain in the threshold window");
  require(eq.otsu_histogram.size() == eq.threshold_entries_by_key.size(),
          "histogram sample count should stay synchronized with threshold entries");
  require(eq.otsu_histogram.bin_count() == 1,
          "old low-score bins should be merged into the current lower-bound bin");

  eq.record_object_heat(1, HP_OTSU_HEAT_MIN, 10001);
  require(eq.threshold_entries_by_key.size() == 2,
          "updating an old low-score object should replace instead of append");
  require(eq.otsu_histogram.size() == eq.threshold_entries_by_key.size(),
          "histogram replacement should keep one sample per threshold object");
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

void test_otsu_threshold_state_machine()
{
  EvaluationQueue initializing;
  for (uint64_t i = 0; i + 1 < HP_OTSU_MIN_OBJECTS; ++i) {
    initializing.record_object_heat(i, i % 2 == 0 ? 10.0 : 1000.0, 1);
  }
  initializing.update_hot_threshold(1);
#if HP_OTSU_PROFILE != HP_OTSU_PROFILE_LEGACY
  require(initializing.hot_threshold_method == HP_THRESHOLD_METHOD_INITIALIZING,
          "insufficient samples should keep threshold initialization state");
  require_close(initializing.hot_threshold, HP_HEAT_INCREMENT,
                "initializing should keep the configured initial threshold");
#else
  require(initializing.hot_threshold_method == HP_THRESHOLD_METHOD_HOLDING,
          "legacy policy should publish its quantile while Otsu initializes");
  require_close(initializing.hot_threshold, 1000.0,
                "legacy initialization should use the p85 object heat");
#endif
  require_close(initializing.otsu_candidate_threshold, 0.0,
                "initializing should not expose a stale candidate");

  EvaluationQueue holding;
  for (uint64_t i = 0; i < 1000; ++i) {
    holding.record_object_heat(i, 10.0, 1);
  }
  holding.update_hot_threshold(1);
  require(holding.hot_threshold_method == HP_THRESHOLD_METHOD_HOLDING,
          "constant samples should hold the effective threshold");
#if HP_OTSU_PROFILE != HP_OTSU_PROFILE_LEGACY
  require_close(holding.hot_threshold, HP_HEAT_INCREMENT,
                "holding should not replace the effective threshold");
#else
  require_close(holding.hot_threshold, 10.0,
                "legacy policy should use its p85 fallback while Otsu is unavailable");
#endif

  EvaluationQueue tracking(10000, 2000, 20.0, 100.0);
  for (uint64_t i = 0; i < 400; ++i) {
    const double score = 1.5 + static_cast<double>(i % 101) / 100.0;
    tracking.record_object_heat(i, std::exp(score), 1);
  }
  for (uint64_t i = 400; i < 800; ++i) {
    const double score = 5.5 + static_cast<double>(i % 101) / 100.0;
    tracking.record_object_heat(i, std::exp(score), 1);
  }
  for (uint64_t i = 800; i < 1000; ++i) {
    const double score = 8.0 * static_cast<double>(i - 800) / 199.0;
    tracking.record_object_heat(i, std::exp(score), 1);
  }
  const double threshold_before_update = tracking.hot_threshold;
  tracking.update_hot_threshold(1);
  const double initial_score = tracking.to_heat_score(
      threshold_before_update, 1);
  const double candidate_score = tracking.to_heat_score(
      tracking.otsu_candidate_threshold, 1);
  const double expected_score = initial_score +
      HP_OTSU_CONFIDENCE_MAX_UPDATE_ALPHA * tracking.otsu_confidence *
      (candidate_score - initial_score);
  require(tracking.hot_threshold_method == HP_THRESHOLD_METHOD_TRACKING,
          "clear bimodal samples should track the Otsu candidate");
#if HP_OTSU_PROFILE == HP_OTSU_PROFILE_CONFIDENCE
  require(std::abs(tracking.hot_threshold - tracking.otsu_candidate_threshold) <=
              std::abs(threshold_before_update -
                       tracking.otsu_candidate_threshold),
          "confidence gain should not move the effective threshold away from its candidate");
  require_close(tracking.to_heat_score(tracking.hot_threshold, 1),
                expected_score,
                "effective threshold should apply confidence-scaled score gain");
#elif HP_OTSU_PROFILE == HP_OTSU_PROFILE_FIXED_EMA
  const double fixed_ema_expected_score = initial_score +
      HP_OTSU_FIXED_EMA_ALPHA * (candidate_score - initial_score);
  require_close(tracking.to_heat_score(tracking.hot_threshold, 1),
                fixed_ema_expected_score,
                "fixed-EMA profile should use the maximum score-space gain");
#else
  const double legacy_expected_score = initial_score +
      HP_LEGACY_OTSU_EMA_ALPHA * (candidate_score - initial_score);
  require_close(tracking.to_heat_score(tracking.hot_threshold, 1),
                legacy_expected_score,
                "legacy policy should use a fixed score-space EMA gain");
#endif
}

void test_otsu_threshold_updates_every_100_observations()
{
  EvaluationQueue eq(
    10000,  // evaluation_window
    200,    // lru_capacity
    100.0,  // hot_threshold
    100.0); // heat_increment

  for (uint64_t i = 0; i < 99; ++i) {
    const double score = 8.0 * static_cast<double>(i) / 99.0;
    eq.record_object_heat(i, std::exp(score), 1);
  }
  require_close(eq.hot_threshold, 100.0,
                "Otsu threshold should not update before 100 observations");

  eq.record_object_heat(99, std::exp(8.0), 1);
  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_TRACKING,
          "Otsu threshold should update on the 100th observation");
}

void test_legacy_otsu_rejects_weak_separation()
{
#if HP_OTSU_PROFILE == HP_OTSU_PROFILE_LEGACY
  EvaluationQueue eq(10000, 2000, 100.0, 100.0);
  eq.record_object_heat(0, std::exp(1.0), 1);
  for (uint64_t i = 1; i < 999; ++i) {
    eq.record_object_heat(i, std::exp(2.0), 1);
  }
  eq.record_object_heat(999, std::exp(3.0), 1);

  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_HOLDING,
          "legacy Otsu should reject candidates below 0.60 separation");
  require_close(eq.hot_threshold, std::exp(2.0),
                "legacy weak-candidate fallback should publish p85 heat");
#endif
}

void test_training_samples_use_unit_weight()
{
  require_close(HP_HOT_CLASS_WEIGHT, 1.0,
                "hot class weight should be fixed at 1.0");

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
  require_close(hot->training_weight, 1.0,
                "hot sample should use unit weight");

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
  require_close(cold->training_weight, 1.0,
                "cold sample should use unit weight");
}

void test_evaluation_queue_feeds_supervised_calibrator()
{
  EvaluationQueue eq(
    1,      // evaluation_window
    8,      // lru_capacity
    50.0,   // hot_threshold
    100.0); // heat_increment

  TraceItem first = make_item(1, 1);
  first.pred = 1;
  first.pred_hot_proba = 0.90;
  eq.prepare_features(first);
  require(!eq.enqueue(first).has_value(),
          "first calibration probe item should stay pending");

  TraceItem second = make_item(2, 2);
  second.pred = 0;
  second.pred_hot_proba = 0.10;
  eq.prepare_features(second);
  auto evaluated = eq.enqueue(second);
  require(evaluated.has_value(),
          "second calibration probe item should evaluate the first item");
  require(evaluated->label == 0,
          "first calibration probe item should be actually cold");
#if HP_ENABLE_PREDICTION_CALIBRATION
  require(eq.prediction_calibration_size() == 1,
          "expired I/O should enter the supervised calibration window");
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
  test_training_batch_is_bounded_and_fifo();
  test_training_tail_sample_wakes_worker();
  test_model_parameters_are_configured();
  test_model_seed_is_reproducible();
  test_reusable_probability_output_matches_return_api();
  test_prediction_clone_is_independent();
  test_predictor_enable_disable_resets_and_gates_io();
  test_concurrent_predict_preserves_index_and_evaluation_counts();
  test_concurrent_enable_disable_does_not_reuse_inflight_slots();
  test_future_heat_label_ignores_decayed_history_only();
  test_pending_prediction_slots_preserve_synchronous_sequence();
  test_pending_prediction_slots_block_oldest_out_of_order_completion();
  test_balanced_accuracy_penalizes_missing_hot_class();
  test_quantile_window_keeps_recent_values();
  test_integer_quantile_window_matches_pbds_window();
  test_object_key_uses_object_identity();
  test_prepare_features_tracks_recency();
#if HP_ENABLE_ACCESS_ACCELERATION
  test_short_access_window_is_bounded_and_counts_prior_io();
  test_access_acceleration_uses_smoothed_window_rates();
#endif
#if HP_ENABLE_HEAT_PERCENTILE
  test_heat_percentile_counts_ties_and_replacements();
#endif
#if HP_ENABLE_ACCESS_ACCELERATION || HP_ENABLE_HEAT_PERCENTILE
  test_optional_features_follow_base_feature_order();
#endif
  test_default_capacity_parameters();
  test_training_batch_size_is_low_latency();
  test_threshold_window_tracks_object_current_heat();
  test_threshold_window_order_has_one_entry_per_object();
  test_otsu_histogram_reports_distribution_shape();
  test_otsu_histogram_rejects_insufficient_and_constant_samples();
  test_otsu_histogram_keeps_weak_candidate();
  test_otsu_histogram_handles_monotonic_distribution();
  test_otsu_histogram_is_stable_under_large_score_translation();
  test_otsu_histogram_tracks_threshold_window_entries();
  test_otsu_threshold_window_clamps_low_scores_without_dropping_entries();
  test_otsu_confidence_formula();
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
