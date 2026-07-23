#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "heatpredictor/heat_predictor.h"
#include "heatpredictor/hp_expiry_heap.h"
#include "heatpredictor/hp_quantile_window.h"
#include "heatpredictor/hp_trace.h"
#include "osd/ObjectHeatPredictor.h"

#ifdef HP_OTSU_DATA_SOURCE
#error "Otsu data-source experiments must not remain in the fixed baseline"
#endif

#ifdef HP_OTSU_PROFILE
#error "dynamic Otsu EMA profiles must not remain in the fixed baseline"
#endif

#ifdef HP_ENABLE_PREDICTION_CALIBRATION
#error "dynamic prediction-threshold calibration must not remain in the fixed baseline"
#endif

#ifdef HP_PREDICTION_RANGE_PROFILE
#error "prediction-threshold range profiles must not remain in the fixed baseline"
#endif

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

class AlwaysDetectProbe {
public:
  bool drift_detected{false};

  void update(double)
  {
    drift_detected = true;
  }
};

class NeverDetectProbe {
public:
  bool drift_detected{false};

  void update(double)
  {
    drift_detected = false;
  }
};

struct AlwaysDetectProbeFactory {
  using DetectorType = AlwaysDetectProbe;

  static DetectorType create()
  {
    return {};
  }
};

struct NeverDetectProbeFactory {
  using DetectorType = NeverDetectProbe;

  static DetectorType create()
  {
    return {};
  }
};

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
    0,      // predicted_hot_probability
    0       // predicted_label
  };
}

void require(bool condition, const char *message);
void require_close(double lhs, double rhs, const char *message);

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

void record_active_otsu_heat(
    EvaluationQueue& eq,
    uint64_t object_key,
    double heat,
    uint64_t timestamp)
{
  eq.record_object_heat(object_key, heat, timestamp);
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

  require_invalid_argument([] {
    ARFClassifier<NUM_FEATURES, 2> invalid_fast_count(
        2, NUM_FEATURES, 7, 100, 4, 0.001, 0.05, 0.99, 0.01,
        nullptr, 3, 300);
  }, "ARF should reject a fast cohort larger than the ensemble");

  require_invalid_argument([] {
    ARFClassifier<NUM_FEATURES, 2> invalid_fast_lifetime(
        2, NUM_FEATURES, 7, 100, 4, 0.001, 0.05, 0.99, 0.01,
        nullptr, 2, 1);
  }, "ARF should reject a fast-tree lifetime shorter than its cohort");

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

void test_arf_fast_cohort_rotates_only_when_enabled()
{
  const auto features = model_features({0.0, 0.0, 0.0});

  auto baseline_telemetry = std::make_shared<ArfAdaptationTelemetry>();
  ARFClassifier<NUM_FEATURES, 2,
      NeverDetectProbeFactory,
      NeverDetectProbeFactory> baseline(
          4, NUM_FEATURES, 7, 100, 4, 0.001, 0.05, 0.99, 0.01,
          baseline_telemetry, 0, 0);
  for (int i = 0; i < 8; ++i) {
    baseline.learn_one(features, i % 2);
  }
  require(baseline_telemetry->snapshot().fast_model_reset_count == 0,
          "disabled fast cohort must not rotate trees");

  auto fast_telemetry = std::make_shared<ArfAdaptationTelemetry>();
  ARFClassifier<NUM_FEATURES, 2,
      NeverDetectProbeFactory,
      NeverDetectProbeFactory> fast(
          4, NUM_FEATURES, 7, 100, 4, 0.001, 0.05, 0.99, 0.01,
          fast_telemetry, 2, 4);
  for (int i = 0; i < 8; ++i) {
    fast.learn_one(features, i % 2);
  }
  require(fast_telemetry->snapshot().fast_model_reset_count == 4,
          "two fast trees with lifetime four should rotate every two samples");
}

void test_arf_adaptation_telemetry_tracks_background_replacement()
{
  auto telemetry = std::make_shared<ArfAdaptationTelemetry>();
  ARFClassifier<NUM_FEATURES, 2,
      AlwaysDetectProbeFactory,
      NeverDetectProbeFactory> model(
          1, NUM_FEATURES, 7, 100, 4, 0.001, 0.05, 0.99, 0.01,
          telemetry);
  const auto features = model_features({0.0, 0.0, 0.0});

  for (int i = 0; i < 20; ++i) {
    model.learn_one(features, i % 2);
  }

  const ArfAdaptationStats stats = telemetry->snapshot();
  require(stats.warning_count >= 2,
          "scripted warning detector should create multiple backgrounds");
  require(stats.drift_count == 0 &&
          stats.background_promotion_count == 0,
          "warning-only stream must not report drift or promotion");
  require(stats.active_background_count == 1,
          "warning-only stream should retain one active background");
  require(stats.background_discard_count == stats.warning_count - 1,
          "repeated warnings should count discarded background trees");
  require(stats.background_training_update_count > 0,
          "retained backgrounds should receive later training updates");
}

void test_arf_adaptation_telemetry_tracks_drift_promotion()
{
  auto telemetry = std::make_shared<ArfAdaptationTelemetry>();
  ARFClassifier<NUM_FEATURES, 2,
      AlwaysDetectProbeFactory,
      AlwaysDetectProbeFactory> model(
          1, NUM_FEATURES, 7, 100, 4, 0.001, 0.05, 0.99, 0.01,
          telemetry);
  const auto features = model_features({0.0, 0.0, 0.0});

  for (int i = 0; i < 10; ++i) {
    model.learn_one(features, i % 2);
  }

  const ArfAdaptationStats stats = telemetry->snapshot();
  require(stats.warning_count > 0 &&
          stats.warning_count == stats.drift_count,
          "paired scripted detectors should warn and drift together");
  require(stats.background_promotion_count == stats.drift_count,
          "every scripted drift should promote its background tree");
  require(stats.background_discard_count == 0,
          "immediate promotion must not discard a background tree");
  require(stats.active_background_count == 0,
          "promoted backgrounds should leave no active background tree");

  telemetry->reset();
  const ArfAdaptationStats reset = telemetry->snapshot();
  require(reset.warning_count == 0 && reset.drift_count == 0 &&
          reset.background_promotion_count == 0 &&
          reset.background_discard_count == 0 &&
          reset.background_training_update_count == 0 &&
          reset.active_background_count == 0 &&
          reset.fast_model_reset_count == 0 &&
          reset.fast_model_background_discard_count == 0,
          "telemetry reset should clear cumulative and gauge fields");
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

  const double previous_access_interval_encoded = 1.0 + hp_log2p1(3.0);
  const double heat_after_current_access = hp_log2p1(1023.0);
  const double threshold_margin =
      hp_log2p1(1023.0) - hp_log2p1(255.0);
  const double projected_heat = 1023.0 *
      std::exp(hp_heat_decay_log_factor_per_ns(
                   HP_HEAT_DECAY_HORIZON_NS) *
               static_cast<double>(HP_FUTURE_LABEL_WINDOW_NS));
  const double projected_heat_margin =
      hp_log2p1(projected_heat) - hp_log2p1(255.0);
#if HP_HEAT_MARGIN_PROFILE == 1
  std::vector<double> expected = {
      projected_heat_margin, previous_access_interval_encoded,
      heat_after_current_access};
#elif HP_HEAT_MARGIN_PROFILE == 2
  std::vector<double> expected = {
      threshold_margin, projected_heat_margin,
      previous_access_interval_encoded, heat_after_current_access};
#else
  std::vector<double> expected = {
      threshold_margin, previous_access_interval_encoded,
      heat_after_current_access};
#endif
#if HP_ENABLE_ACCESS_RATE_CHANGE
  expected.push_back(hp_access_rate_change_log2p1(
      item.short_window_access_count, item.long_window_access_count));
#endif
  require(feat.size() == expected.size(),
          "final feature vector should contain only the selected features");
  for (size_t i = 0; i < expected.size(); ++i) {
    require_close(feat[i], expected[i],
                  "final feature vector should preserve feature order");
  }
}

void test_previous_access_interval_encoding()
{
  PredictionSample item = make_item(100, 42);
#if HP_HEAT_MARGIN_PROFILE == HP_HEAT_MARGIN_BOTH
  constexpr size_t feature_index = 2;
#else
  constexpr size_t feature_index = 1;
#endif

  item.tracked_access_count = 1;
  item.time_since_previous_access_ns = 0;
  require_close(
      HeatPredictor::to_feat(item)[feature_index], 0.0,
      "first observation should encode missing previous access as zero");

  item.tracked_access_count = 2;
  require_close(
      HeatPredictor::to_feat(item)[feature_index], 1.0,
      "a real zero interval should remain distinct from missing history");

  item.time_since_previous_access_ns = 3ULL * 1000 * 1000 * 1000;
  require_close(
      HeatPredictor::to_feat(item)[feature_index], 1.0 + hp_log2p1(3.0),
      "a real previous access interval should use the offset log2p1 encoding");
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
struct has_cold_to_hot_crossing_margin : std::false_type {};

template <typename T>
struct has_cold_to_hot_crossing_margin<T, std::void_t<
    decltype(std::declval<T>().cold_to_hot_crossing_margin)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_heat_percentile_member : std::false_type {};

template <typename T>
struct has_heat_percentile_member<T, std::void_t<
    decltype(std::declval<T>().heat_percentile)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_threshold_order_stats_member : std::false_type {};

template <typename T>
struct has_threshold_order_stats_member<T, std::void_t<
    decltype(std::declval<T>().threshold_order_stats)>>
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
    decltype(std::declval<T>().otsu_histogram_vote_count),
    decltype(std::declval<T>().otsu_candidate_threshold),
    decltype(std::declval<T>().otsu_confidence),
    decltype(std::declval<T>().otsu_sharpness_confidence)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_otsu_histogram_object_count_stat : std::false_type {};

template <typename T>
struct has_otsu_histogram_object_count_stat<
    T, std::void_t<decltype(std::declval<T>().otsu_histogram_object_count)>>
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

void test_fixed_baseline_drops_obsolete_state()
{
  require(!has_cold_to_hot_crossing_margin<PredictionSample>::value,
          "PredictionSample should not retain the rejected crossing feature");
  require(!has_hot_predict_threshold_stat<HeatPredictorStats>::value,
          "fixed prediction threshold should not be exported as runtime state");
  require(!has_otsu_confidence_stats<HeatPredictorStats>::value,
          "unused Otsu confidence diagnostics should not remain in stats");
  require(!has_heat_percentile_member<PredictionSample>::value,
          "PredictionSample should not retain the rejected percentile feature");
  require(!has_threshold_order_stats_member<EvaluationQueue>::value,
          "EvaluationQueue should not retain the percentile PBDS index");
}

void test_stats_export_otsu_histogram_bin_count()
{
  require(has_otsu_histogram_bin_count_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should expose Otsu histogram bin count");
  require(!has_otsu_histogram_object_count_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should use a source-neutral Otsu vote count");
  require(!has_otsu_sample_confidence_stat<HeatPredictorStats>::value,
          "HeatPredictorStats should not expose removed sample confidence");
}

void test_otsu_update_cost_knobs()
{
  require(HP_OTSU_EAGER_OBJECTS == 0,
          "Otsu eager updates should be disabled for fixed interval updates");
  require(HP_OTSU_UPDATE_INTERVAL == 100,
          "Otsu update interval should refresh every 100 object-vote updates");
  require_close(HP_OTSU_EMA_ALPHA, 0.10,
                "Otsu should use the fixed 0.10 EMA gain");
  require(HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT == 800,
          "score total-heat Otsu should use exactly 800 fixed bins");
  require_close(HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH, 0.01,
                "score total-heat Otsu log bin width should be 0.01");
  require_close(
      HP_OTSU_TOTAL_HEAT_MIN,
      HP_HEAT_INCREMENT * HP_HEAT_RETAINED_AFTER_DECAY_HORIZON,
      "Otsu lower bound should equal one access after a decay horizon");
  require_close(
      HP_OTSU_TOTAL_HEAT_MAX,
      HP_OTSU_TOTAL_HEAT_MIN * std::exp(
          HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT *
          HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH),
      "score total-heat upper bound should span 800 logarithmic bins");
}

void test_score_total_heat_histogram_moves_with_time()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  const double decay_log_factor = hp_heat_decay_log_factor_per_ns(
      HP_HEAT_DECAY_HORIZON_NS);
  HpScoreOtsuHistogram histogram;

  require(histogram.bin_capacity() == 800,
          "score total-heat histogram should have a fixed 800-bin capacity");
  const double cold_score = HpScoreOtsuHistogram::score_for_heat_at(
      10.0, 0, decay_log_factor);
  const double hot_score = HpScoreOtsuHistogram::score_for_heat_at(
      1000.0, 0, decay_log_factor);
  require_close(
      HpScoreOtsuHistogram::heat_for_score_at(
          hot_score, 0, decay_log_factor),
      1000.0,
      "score conversion should round-trip total heat at the same time");

  histogram.advance_lower_bound(cold_score);
  const auto cold_bin = histogram.insert(cold_score);
  const auto hot_bin = histogram.insert(hot_score);
  require(histogram.size() == 2 && histogram.bin_count() == 2,
          "two total-heat populations should occupy two score bins");

  const double later_min_score = HpScoreOtsuHistogram::score_for_heat_at(
      10.0, HP_HEAT_DECAY_HORIZON_NS, decay_log_factor);
  histogram.advance_lower_bound(later_min_score);
  require(histogram.size() == 2,
          "moving the lower score bound should merge rather than discard votes");
  histogram.erase(cold_bin);
  histogram.erase(hot_bin);
  require(histogram.empty(),
          "stored absolute bins should remain erasable after lower-bound movement");

  const double decayed_heat = HpScoreOtsuHistogram::heat_for_score_at(
      hot_score, HP_HEAT_DECAY_HORIZON_NS, decay_log_factor);
  require_close(
      decayed_heat,
      1000.0 * HP_HEAT_RETAINED_AFTER_DECAY_HORIZON,
                "normalized score should preserve configured time decay");
  (void)second_ns;
}

void test_score_total_heat_label_uses_deadline_threshold()
{
  EvaluationQueue eq(
    HP_HEAT_DECAY_HORIZON_NS,
    8,
    50.0,   // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    0,      // future_label_window_ns
    8);

  PredictionSample item = make_item(1, 42);
  eq.prepare_features(item, 0);
  require(eq.enqueue(item, 0),
          "deadline-total-heat probe should enter the evaluation queue");
  auto evaluated = eq.expire_due_evaluations(0);
  require(evaluated.size() == 1,
          "zero-length deadline-total-heat probe should complete immediately");
  require(evaluated.front().label == 1,
          "total heat 100 should be hot against total-heat threshold 50");
  require_close(evaluated.front().label_heat, 100.0,
                "the label should use object total heat at the deadline");
}

void test_score_total_heat_label_ignores_later_threshold_updates()
{
  constexpr uint64_t deadline = 100;
  constexpr uint64_t later_time = 200;
  EvaluationQueue eq(
    1000,   // heat_decay_horizon_ns
    100,
    50.0,   // heat_label_threshold
    100.0,  // heat_increment
    HP_SHORT_ACCESS_WINDOW_NS,
    deadline,
    100);

  PredictionSample item = make_item(1, 42);
  eq.prepare_features(item, 0);
  require(eq.enqueue(item, 0),
          "threshold-history probe should enter the evaluation queue");
  const double deadline_threshold = eq.heat_label_threshold_for_label(deadline);

  for (uint64_t i = 0; i < HP_OTSU_MIN_VOTES; ++i) {
    eq.record_object_heat(
        1000 + i,
        i < HP_OTSU_MIN_VOTES / 2 ? 30.0 : 1000.0,
        later_time);
  }
  eq.update_hot_threshold(later_time);
  require_close(eq.heat_label_threshold_for_label(deadline),
                deadline_threshold,
                "an Otsu update after the deadline must not rewrite its label threshold");
  auto evaluated = eq.expire_due_evaluations(later_time);
  require(evaluated.size() == 1 && evaluated.front().label == 1,
          "late label completion should use the threshold version active at its deadline");
  require(eq.threshold_heat_history.size() == 1,
          "threshold history should retain only its newest version without pending labels");
}

void test_score_total_heat_ema_uses_current_heat_threshold()
{
  constexpr uint64_t horizon_ns = 1000;
  constexpr uint64_t update_time_ns = 2 * horizon_ns;
  EvaluationQueue eq(
    horizon_ns,
    100,
    100.0,  // heat_label_threshold
    100.0);
  PredictionSample item = make_item(1, 42);
  eq.prepare_features(item, 0);

  for (uint64_t i = 0; i < HP_OTSU_MIN_VOTES; ++i) {
    eq.record_object_heat(
        1000 + i,
        i < HP_OTSU_MIN_VOTES / 2 ? 30.0 : 1000.0,
        update_time_ns);
  }
  auto candidate = eq.score_otsu_histogram.otsu_result();
  require(candidate.has_value(),
          "long-idle recovery probe should produce an Otsu candidate");
  const double current_threshold_score = HpScoreOtsuHistogram::score_for_heat_at(
      100.0,
      update_time_ns,
      eq.heat_decay_log_factor_per_ns);
  const double expected_score = current_threshold_score +
      HP_OTSU_EMA_ALPHA *
          (candidate->threshold_score - current_threshold_score);
  const double expected_heat = HpScoreOtsuHistogram::heat_for_score_at(
      expected_score,
      update_time_ns,
      eq.heat_decay_log_factor_per_ns);
  eq.update_hot_threshold(update_time_ns);
  require(eq.otsu_candidate_threshold > HP_OTSU_TOTAL_HEAT_MIN,
          "reported total-heat Otsu candidate should be available after tracking");
  require_close(eq.heat_label_threshold,
                expected_heat,
                "EMA should compare the current heat threshold and candidate at one timestamp");
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
    source.push(TrainingSample{make_item(index, index), 0});
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
      TrainingSample{make_item(1, 1), 0});

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
  constexpr int expected_features = 3
#if HP_HEAT_MARGIN_PROFILE == 2
      + 1
#endif
#if HP_ENABLE_ACCESS_RATE_CHANGE
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
  require(HP_ARF_GRACE_PERIOD == HP_ARF_GRACE_PERIOD_VALUE,
          "ARF grace period should match the selected replay profile");
  require(HP_ARF_LAMBDA == 4,
          "ARF online bagging lambda should remain unchanged");
  require(HP_ARF_WARNING_DELTA_PERMILLE >
          HP_ARF_DRIFT_DELTA_PERMILLE,
          "ARF warning detector should remain more sensitive than drift");
  require(HP_ARF_WARNING_DELTA_PERMILLE ==
              HP_ARF_WARNING_DELTA_PERMILLE_VALUE &&
          HP_ARF_DRIFT_DELTA_PERMILLE ==
              HP_ARF_DRIFT_DELTA_PERMILLE_VALUE,
          "ARF detector deltas should match the selected replay profile");
  require(HP_ARF_FAST_MODEL_COUNT == HP_ARF_FAST_MODEL_COUNT_VALUE &&
          HP_ARF_FAST_MODEL_LIFETIME_SAMPLES ==
              HP_ARF_FAST_MODEL_LIFETIME_SAMPLES_VALUE,
          "ARF fast cohort should match the selected replay profile");
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

void test_total_heat_label_includes_entry_heat()
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
  require_close(
      evaluated.front().label_heat,
      eq.decay_heat(100.0, 10, 10 + duration),
      "total-heat label should include decayed entry heat");
  require(evaluated.front().label == 1,
          "deadline total heat above the total-heat threshold should be hot");
}

void test_total_heat_label_includes_later_access()
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
  const double expected_total_heat =
      eq.decay_heat(100.0, 10, 510) + 100.0;
  require_close(
      evaluated.front().label_heat,
      eq.decay_heat(expected_total_heat, 510, 10 + duration),
      "total-heat label should include entry heat and the later access");
  require(evaluated.front().label == 1,
          "sufficient total heat should be labeled hot");
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
      late.front().label_heat,
      exact.front().label_heat,
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
  auto after_label = eq.expiry_schedule(1100);
  require(after_label.state ==
              EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
          after_label.deadline_ns == 10100,
          "Otsu maintenance should retain the object until heat reaches 20");
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
    require_close(lhs[i].label_heat, rhs[i].label_heat, message);
    require_close(lhs[i].label_heat_threshold,
                  rhs[i].label_heat_threshold, message);
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
  require_close(
      second.heat_after_current_access,
      100.0 * HP_HEAT_RETAINED_AFTER_DECAY_HORIZON + HP_HEAT_INCREMENT,
      "heat should retain the configured fraction after the decay horizon");
}

void test_long_window_trace_field_uses_pending_evaluation_count()
{
  EvaluationQueue eq(
    HP_HEAT_DECAY_HORIZON_NS,
    16,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    HP_SHORT_ACCESS_WINDOW_NS,
    HP_FUTURE_LABEL_WINDOW_NS,
    1);

  PredictionSample first = make_item(1, 9);
  eq.prepare_features(first, 0);
  require(first.long_window_access_count == 0,
          "first access should see no earlier pending evaluation");
  require(eq.enqueue(first, 0), "first evaluation should be admitted");

  PredictionSample dropped = make_item(2, 9);
  eq.prepare_features(dropped, 1);
  require(dropped.long_window_access_count == 1,
          "compatibility count should include the admitted pending sample");
  require(!eq.enqueue(dropped, 1), "second evaluation should be dropped");

  PredictionSample after_drop = make_item(3, 9);
  eq.prepare_features(after_drop, 2);
  require(after_drop.long_window_access_count == 1,
          "a dropped evaluation must not create long-window runtime state");
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

void test_short_access_window_schedules_idle_cleanup()
{
  constexpr uint64_t short_window_ns = 5ULL * 1000 * 1000 * 1000;
  constexpr uint64_t start_ns = 100;
  EvaluationQueue eq(
    HP_HEAT_DECAY_HORIZON_NS,
    16,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    short_window_ns,
    HP_FUTURE_LABEL_WINDOW_NS,
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
          eq.lru_size() == 1,
          "short-window cleanup should release the object to the LRU");
  require(eq.expiry_schedule(start_ns + short_window_ns).state ==
              EvaluationQueue::ExpiryScheduleState::empty,
          "no independent long-window event should remain scheduled");
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

#if HP_ENABLE_ACCESS_RATE_CHANGE
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

  const auto& feat = HeatPredictor::to_feat(item);
  size_t next = HP_BASE_FEATURE_COUNT;
#if HP_ENABLE_ACCESS_RATE_CHANGE
  require_close(feat[next++], 0.0,
                "access-rate change should follow the configured base features");
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

  eq.record_object_heat(42, 30.0, 1);
  require(eq.threshold_entries_by_key.size() == 1,
          "first object should create one object heat entry");

  eq.record_object_heat(42, 40.0, 2);
  require(eq.threshold_entries_by_key.size() == 1,
          "same object should replace object heat instead of appending");

  eq.record_object_heat(7, 50.0, 2);
  require(eq.threshold_entries_by_key.size() == 2,
          "different object should create a second object heat entry");

  eq.record_object_heat(42, 60.0, 2);
  require(eq.threshold_entries_by_key.size() == 2,
          "replacing one object should not change object heat entry count");
}

void test_threshold_window_order_has_one_entry_per_object()
{
  EvaluationQueue eq(
    4,      // heat_decay_horizon_ns
    8,      // lru_capacity
    100.0,  // heat_label_threshold
    100.0); // heat_increment

  eq.record_object_heat(1, 1000.0, 1);
  eq.record_object_heat(2, 2000.0, 2);
  eq.record_object_heat(3, 3000.0, 3);
  eq.record_object_heat(2, 4000.0, 4);

  require(eq.threshold_entries_by_key.size() == 3,
          "threshold index should keep one entry per object");
  require(eq.threshold_order.size() == 3,
          "threshold recency list should not retain stale object entries");
}

void test_heat_state_lifecycle_telemetry()
{
  EvaluationQueue eq(
    HP_HEAT_DECAY_HORIZON_NS,
    1,
    HP_HEAT_INCREMENT,
    HP_HEAT_INCREMENT,
    HP_SHORT_ACCESS_WINDOW_NS,
    HP_FUTURE_LABEL_WINDOW_NS,
    0);

  PredictionSample first = make_item(1, 101);
  eq.prepare_features(first, 0);
  require(eq.protected_heat_state_size() == 1,
          "a newly accessed object should be protected before admission");
  require(!eq.enqueue(first, 0),
          "zero evaluation capacity should reject the first sample");
#if HP_ENABLE_ACCESS_RATE_CHANGE
  eq.expire_due_access_windows(HP_SHORT_ACCESS_WINDOW_NS);
#endif
  require(eq.protected_heat_state_size() == 0 && eq.lru_size() == 1,
          "a rejected object should become an idle LRU state");

  PredictionSample second = make_item(2, 202);
  constexpr uint64_t second_time_ns =
      HP_ENABLE_ACCESS_RATE_CHANGE ? HP_SHORT_ACCESS_WINDOW_NS + 1 : 1;
  eq.prepare_features(second, second_time_ns);
  require(!eq.enqueue(second, second_time_ns),
          "zero evaluation capacity should reject the second sample");
#if HP_ENABLE_ACCESS_RATE_CHANGE
  eq.expire_due_access_windows(
      second_time_ns + HP_SHORT_ACCESS_WINDOW_NS);
#endif
  require(eq.heat_state_size() == 1 && eq.lru_size() == 1,
          "LRU capacity should retain only one idle heat state");
  require(eq.heat_state_peak_size() == 2,
          "heat-state peak should include the pre-eviction high-water mark");
  require(eq.lru_eviction_count() == 1,
          "LRU eviction telemetry should count removed heat states");
}

void test_expiry_heap_orders_refreshes_and_erases_keys()
{
  HpExpiryHeap heap;
  require(heap.empty(), "a new expiry heap should be empty");

  heap.upsert(3, 30);
  heap.upsert(1, 10);
  heap.upsert(2, 20);
  require(heap.size() == 3,
          "expiry heap should keep one node per inserted key");
  require(!heap.due_key(9).has_value(),
          "expiry heap should not expose a future deadline");
  require(heap.due_key(10) == std::optional<uint64_t>(1),
          "expiry heap should expose the earliest due key");

  heap.upsert(1, 40);
  require(heap.size() == 3,
          "refreshing a key should update its node in place");
  require(heap.due_key(20) == std::optional<uint64_t>(2),
          "refreshing the root should restore heap order");
  require(heap.erase(2), "expiry heap should erase an indexed key");
  require(!heap.erase(2),
          "erasing a missing expiry key should report false");
  require(heap.due_key(30) == std::optional<uint64_t>(3),
          "arbitrary erase should preserve the remaining heap order");

  require(heap.erase(3),
          "expiry heap should remove the remaining earlier deadline");
  heap.upsert(4, 40);
  require(heap.due_key(40) == std::optional<uint64_t>(1),
          "equal deadlines should use the object key as a stable tie-breaker");
  heap.clear();
  require(heap.empty() && heap.size() == 0,
          "clearing the expiry heap should remove nodes and indexes");
}

void test_object_heat_window_bounds_otsu_votes()
{
  EvaluationQueue eq(
    10000,  // heat_decay_horizon_ns
    200,    // lru_capacity
    100.0,  // heat_label_threshold
    100.0); // heat_increment
  eq.heat_label_threshold_object_capacity = 2;

  eq.record_object_heat(1, 30.0, 1);
  eq.record_object_heat(2, 30.0, 2);
  eq.record_object_heat(3, 40.0, 3);
  require(eq.threshold_entries_by_key.size() == 2,
          "threshold index should evict to its configured capacity");
  require(eq.score_otsu_histogram.size() == 2,
          "Otsu should keep one total-heat vote per retained object");
  require(eq.threshold_expiry_heap.size() == 2,
          "D1 should keep one expiry node per retained Otsu object");
}

void test_otsu_population_excludes_floor_heat_objects()
{
  constexpr uint64_t decay_horizon_ns = 1000;
  EvaluationQueue eq(
    decay_horizon_ns,
    8,
    100.0,
    100.0);

  eq.record_object_heat(1, HP_OTSU_TOTAL_HEAT_MIN, 0);
  require(eq.score_otsu_histogram.empty(),
          "D1 should not insert an object already at the heat floor");

  eq.record_object_heat(1, 100.0, 0);
  require(eq.score_otsu_histogram.size() == 1,
          "D1 should retain an object while its heat is above the floor");
  eq.advance_otsu_history(decay_horizon_ns - 1);
  require(eq.score_otsu_histogram.size() == 1,
          "D1 should retain the vote until its exact floor crossing");
  eq.advance_otsu_history(decay_horizon_ns);
  require(eq.score_otsu_histogram.empty(),
          "D1 should remove the vote at its floor crossing");
  require(eq.threshold_entries_by_key.empty() && eq.threshold_order.empty(),
          "D1 expiry should remove both threshold indexes");
  require(eq.threshold_expiry_heap.empty(),
          "D1 expiry should remove the object's expiry index node");
}

void test_otsu_population_refresh_ignores_stale_expiry()
{
  constexpr uint64_t decay_horizon_ns = 1000;
  EvaluationQueue eq(
    decay_horizon_ns,
    8,
    100.0,
    100.0);

  eq.record_object_heat(1, 100.0, 0);
  eq.record_object_heat(1, 100.0, decay_horizon_ns / 2);
  require(eq.threshold_expiry_heap.size() == 1,
          "refreshing a D1 vote should replace rather than append expiry state");
  eq.advance_otsu_history(decay_horizon_ns);
  require(eq.score_otsu_histogram.size() == 1,
          "a refreshed D1 vote should survive its stale expiry event");
  eq.advance_otsu_history(decay_horizon_ns + decay_horizon_ns / 2);
  require(eq.score_otsu_histogram.empty(),
          "a refreshed D1 vote should expire at its new floor crossing");
}

void test_otsu_population_schedules_idle_floor_crossing()
{
  constexpr uint64_t decay_horizon_ns = 1000;
  EvaluationQueue eq(
    decay_horizon_ns,
    8,
    100.0,
    100.0);

  eq.record_object_heat(1, 100.0, 0);
  const auto waiting = eq.expiry_schedule(0);
  require(waiting.state ==
              EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
          waiting.deadline_ns == decay_horizon_ns,
          "idle maintenance should wake at the Otsu vote floor crossing");
  require(eq.expiry_schedule(decay_horizon_ns).state ==
              EvaluationQueue::ExpiryScheduleState::due,
          "the Otsu floor crossing should become due without another I/O");
}

void test_otsu_population_uses_unclamped_heat_for_expiry()
{
  constexpr uint64_t decay_horizon_ns = 1000000;
  EvaluationQueue eq(
    decay_horizon_ns,
    8,
    100.0,
    100.0);
  const double heat = HP_OTSU_TOTAL_HEAT_MAX * 2.0;
  const double decay_log_factor = hp_heat_decay_log_factor_per_ns(
    decay_horizon_ns);
  const uint64_t clamped_expiry = static_cast<uint64_t>(std::ceil(
    std::log(HP_OTSU_TOTAL_HEAT_MIN / HP_OTSU_TOTAL_HEAT_MAX) /
    decay_log_factor));
  const uint64_t actual_expiry = static_cast<uint64_t>(std::ceil(
    std::log(HP_OTSU_TOTAL_HEAT_MIN / heat) / decay_log_factor));
  require(actual_expiry > clamped_expiry,
          "the probe heat should outlive the histogram ceiling");

  eq.record_object_heat(1, heat, 0);
  eq.advance_otsu_history(clamped_expiry);
  require(eq.score_otsu_histogram.size() == 1,
          "histogram clamping must not shorten an object's Otsu lifetime");
  eq.advance_otsu_history(actual_expiry);
  require(eq.score_otsu_histogram.empty(),
          "an above-ceiling object should leave Otsu at its actual floor crossing");
}

void test_otsu_population_expiry_rounds_toward_the_cold_side()
{
  constexpr uint64_t start_ns = 100;
  constexpr double heat = 150.0;
  EvaluationQueue eq(
    10000, 8, 5.0, 100.0, HP_SHORT_ACCESS_WINDOW_NS, 1000, 8);

  eq.record_object_heat(1, heat, start_ns);
  const double decay_ns = std::log(HP_OTSU_TOTAL_HEAT_MIN / heat) /
      hp_heat_decay_log_factor_per_ns(10000);
  const uint64_t expected_deadline =
      start_ns + static_cast<uint64_t>(std::ceil(decay_ns));
  const auto schedule = eq.expiry_schedule(start_ns);
  require(schedule.state ==
              EvaluationQueue::ExpiryScheduleState::waiting_deadline &&
          schedule.deadline_ns == expected_deadline,
          "Otsu expiry must round up so heat is no longer above 20");
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

void test_otsu_recomputes_after_wall_clock_interval()
{
  EvaluationQueue eq;
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  for (uint64_t i = 0; i < HP_OTSU_MIN_VOTES; ++i) {
    const double heat = i < HP_OTSU_MIN_VOTES / 2 ? 30.0 : 1000.0;
    const uint64_t timestamp = i * second_ns / (HP_OTSU_MIN_VOTES - 1);
    record_active_otsu_heat(eq, i, heat, timestamp);
  }
  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_TRACKING,
          "new observations should recompute Otsu after one wall-clock second");
}

void test_otsu_ema_time_never_moves_backward_for_late_sample()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  constexpr uint64_t current_time = 61 * second_ns;
  EvaluationQueue eq;

  for (uint64_t i = 0; i < HP_OTSU_MIN_VOTES; ++i) {
    const double heat = i < HP_OTSU_MIN_VOTES / 2 ? 30.0 : 1000.0;
    record_active_otsu_heat(eq, i, heat, current_time);
  }
  eq.update_hot_threshold(current_time);
  require(eq.last_otsu_ema_update_time_ns == current_time,
          "initial Otsu update should establish the current EMA time");

  eq.update_hot_threshold(current_time - 2 * second_ns);
  require(eq.last_otsu_ema_update_time_ns == current_time,
          "a late observation must not move the Otsu EMA clock backward");
}

void test_otsu_threshold_updates_every_100_observations()
{
  EvaluationQueue eq(
    10000,  // heat_decay_horizon_ns
    200,    // lru_capacity
    100.0,  // heat_label_threshold
    100.0); // heat_increment

  for (uint64_t i = 0; i < 99; ++i) {
    const double heat = 30.0 + 970.0 * static_cast<double>(i) / 99.0;
    record_active_otsu_heat(eq, i, heat, 1);
  }
  require_close(eq.heat_label_threshold, 100.0,
                "Otsu threshold should not update before 100 observations");

  record_active_otsu_heat(eq, 99, 1000.0, 1);
  require(eq.hot_threshold_method == HP_THRESHOLD_METHOD_TRACKING,
          "Otsu threshold should update on the 100th observation");
}

void test_trace_writer_drains_and_rotates_sessions()
{
  HpTraceWriter writer(8, 2);
  require(writer.start(
              "/tmp", 7, "phase-a", "deadbeef", 0x1234, NUM_FEATURES),
          "trace writer should start in an existing directory");
  const std::string first_path = writer.status().path;

  HpTraceRecord first{};
  first.io_sequence = 11;
  first.object_key_hash = 101;
  first.outcome = static_cast<uint8_t>(HpTraceOutcome::evaluated);
  first.actual_label = 1;
  first.predicted_label = 1;
  first.features[0] = 1.25;
  require(writer.try_submit(first),
          "enabled trace writer should accept a record");

  require(writer.start(
              "/tmp", 7, "phase-b", "deadbeef", 0x1234, NUM_FEATURES),
          "starting a new session should drain and rotate the old session");
  const std::string second_path = writer.status().path;
  require(first_path != second_path,
          "rotated trace sessions should use distinct files");

  HpTraceRecord second{};
  second.io_sequence = 12;
  second.object_key_hash = 102;
  second.outcome = static_cast<uint8_t>(HpTraceOutcome::prediction_error);
  second.actual_label = -1;
  second.predicted_label = 0;
  require(writer.try_submit(second),
          "rotated trace session should accept records");
  writer.stop();

  auto read_session = [](const std::string& path,
                         const char *phase,
                         uint64_t expected_sequence,
                         HpTraceOutcome expected_outcome) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "trace file should be readable after stop");
    HpTraceFileHeader header{};
    HpTraceRecord record{};
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    input.read(reinterpret_cast<char *>(&record), sizeof(record));
    require(input.good(), "trace file should contain one complete record");
    require(std::memcmp(header.magic, HP_TRACE_MAGIC, sizeof(header.magic)) == 0,
            "trace file magic should identify the schema");
    require(header.schema_version == HP_TRACE_SCHEMA_VERSION,
            "trace file should report the current schema");
    require(header.header_size == sizeof(HpTraceFileHeader),
            "trace header should report its packed size");
    require(header.record_size == sizeof(HpTraceRecord),
            "trace header should report the packed record size");
    require(header.osd_id == 7,
            "trace header should preserve the OSD id");
    require(std::string(header.phase) == phase,
            "trace header should preserve the workload phase");
    require(record.io_sequence == expected_sequence,
            "trace record should preserve the I/O sequence");
    require(record.outcome == static_cast<uint8_t>(expected_outcome),
            "trace record should preserve the completion outcome");
    char extra = 0;
    input.read(&extra, 1);
    require(input.eof(), "trace session should not contain extra records");
  };

  read_session(first_path, "phase-a", 11, HpTraceOutcome::evaluated);
  read_session(second_path, "phase-b", 12, HpTraceOutcome::prediction_error);
  std::remove(first_path.c_str());
  std::remove(second_path.c_str());

  const HpTraceStatus status = writer.status();
  require(!status.enabled, "stopped trace writer should report disabled");
  require(status.written_count == 2,
          "trace status should accumulate records across sessions");
  require(status.drop_count == 0,
          "uncontended trace writes should not be dropped");
  require(!writer.try_submit(first),
          "disabled trace writer should ignore records");
}

void test_evaluated_sample_maps_to_trace_schema()
{
  PredictionSample item = make_item(23, 456);
  item.heat_after_current_access = 321.0;
  item.heat_label_threshold_at_prediction = 120.0;
  item.tracked_access_count = 8;
  item.time_since_previous_access_ns = 900;
  item.long_window_access_count = 7;
  item.short_window_access_count = 3;
  item.predicted_hot_probability = 0.75;
  item.predicted_label = 1;
  EvaluatedSample evaluated{
      item,
      1,
      5,
      100,
      200,
      210,
      300.0,
      250.0,
      true,
  };

  const HpTraceRecord record =
      HeatPredictor::trace_record_for_evaluated(evaluated);
  require(record.io_sequence == 23 && record.object_key_hash == 456,
          "trace record should preserve anonymous sample identity");
  require(record.prediction_time_ns == 100 &&
              record.label_deadline_ns == 200 &&
              record.label_completion_time_ns == 210,
          "trace record should preserve prediction and label timing");
  require_close(record.predicted_hot_probability, 0.75,
                "trace record should preserve predicted probability");
  require_close(record.hot_predict_threshold, 0.50,
                "trace record should preserve prediction threshold");
  require_close(record.label_heat, 300.0,
                "trace record should preserve label heat");
  require_close(record.label_heat_threshold, 250.0,
                "trace record should preserve label heat threshold");
  require(record.actual_label == 1 && record.predicted_label == 1,
          "trace record should preserve both classes");
  require((record.flags & HP_TRACE_FLAG_COLD_START_FALLBACK) != 0,
          "trace record should preserve cold-start fallback state");
}

void test_trace_writer_handles_concurrent_producers_without_losing_accepts()
{
  constexpr int producer_count = 4;
  constexpr int records_per_producer = 2000;
  constexpr uint64_t attempted = producer_count * records_per_producer;
  HpTraceWriter writer(attempted + 1, 128);
  require(writer.start(
              "/tmp", 8, "concurrent", "deadbeef", 0x5678, NUM_FEATURES),
          "concurrent trace writer should start");
  const std::string path = writer.status().path;
  std::atomic<uint64_t> accepted{0};
  std::vector<std::thread> producers;
  producers.reserve(producer_count);
  for (int producer = 0; producer < producer_count; ++producer) {
    producers.emplace_back([&, producer] {
      for (int index = 0; index < records_per_producer; ++index) {
        HpTraceRecord record{};
        record.io_sequence =
            static_cast<uint64_t>(producer * records_per_producer + index + 1);
        record.outcome = static_cast<uint8_t>(HpTraceOutcome::evaluated);
        if (writer.try_submit(record)) {
          accepted.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  writer.stop();

  const HpTraceStatus status = writer.status();
  require(status.queue_length == 0,
          "trace stop should drain the concurrent producer queue");
  require(status.drop_count == 0,
          "a trace queue with enough capacity must not drop on lock contention");
  require(accepted.load(std::memory_order_relaxed) == attempted,
          "a sufficiently large trace queue should accept every record");
  require(status.written_count == attempted,
          "every concurrently submitted trace record should reach disk");
  require(status.write_error_count == 0,
          "concurrent trace writes should not report disk errors");
  std::remove(path.c_str());
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
  test_arf_adaptation_telemetry_tracks_background_replacement();
  test_arf_adaptation_telemetry_tracks_drift_promotion();
  test_arf_fast_cohort_rotates_only_when_enabled();
  test_adwin_bucket_count_and_numeric_state();
  test_final_feature_vector();
  test_previous_access_interval_encoding();
  test_stats_drop_online_prediction_ratio_source();
  test_predictor_drops_unused_actual_label_counters();
  test_fixed_baseline_drops_obsolete_state();
  test_stats_export_otsu_histogram_bin_count();
  test_otsu_update_cost_knobs();
  test_score_total_heat_histogram_moves_with_time();
  test_score_total_heat_label_uses_deadline_threshold();
  test_score_total_heat_label_ignores_later_threshold_updates();
  test_score_total_heat_ema_uses_current_heat_threshold();
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
  test_total_heat_label_includes_entry_heat();
  test_total_heat_label_includes_later_access();
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
  test_long_window_trace_field_uses_pending_evaluation_count();
#if HP_ENABLE_ACCESS_RATE_CHANGE
  test_short_access_window_uses_time_and_counts_prior_io();
  test_short_access_window_schedules_idle_cleanup();
  test_access_rate_change_uses_log2p1_rates();
#endif
#if HP_ENABLE_ACCESS_RATE_CHANGE
  test_optional_features_follow_base_feature_order();
#endif
  test_default_capacity_parameters();
  test_training_batch_size_is_low_latency();
  test_threshold_window_tracks_object_current_heat();
  test_threshold_window_order_has_one_entry_per_object();
  test_heat_state_lifecycle_telemetry();
  test_expiry_heap_orders_refreshes_and_erases_keys();
  test_object_heat_window_bounds_otsu_votes();
  test_otsu_population_excludes_floor_heat_objects();
  test_otsu_population_refresh_ignores_stale_expiry();
  test_otsu_population_schedules_idle_floor_crossing();
  test_otsu_population_uses_unclamped_heat_for_expiry();
  test_otsu_population_expiry_rounds_toward_the_cold_side();
  test_time_normalized_ema_gain();
  test_otsu_recomputes_after_wall_clock_interval();
  test_otsu_threshold_updates_every_100_observations();
  test_trace_writer_drains_and_rotates_sessions();
  test_evaluated_sample_maps_to_trace_schema();
  test_trace_writer_handles_concurrent_producers_without_losing_accepts();
  std::cout << "PASS: hp algorithm probe" << std::endl;
  return 0;
}
