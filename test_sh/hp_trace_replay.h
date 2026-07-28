#ifndef CEPH_TEST_SH_HP_TRACE_REPLAY_H
#define CEPH_TEST_SH_HP_TRACE_REPLAY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "heatpredictor/heat_predictor.h"
#include "heatpredictor/include/HoeffdingTreeClassifier.h"

static constexpr uint64_t HP_REPLAY_SNAPSHOT_SAMPLE_INTERVAL = 500;

enum class HpReplayAdaptationProfile : uint8_t {
  baseline = 0,
  conservative = 1,
  disabled = 2,
};

enum class HpReplayPredictionMode : uint8_t {
  direct_arf = 0,
  persistence = 1,
  c2h_hoeffding_tree = 2,
  c2h_arf = 3,
};

inline HpReplayPredictionMode parse_hp_replay_prediction_mode(
    const std::string& value) {
  if (value == "direct-arf") {
    return HpReplayPredictionMode::direct_arf;
  }
  if (value == "persistence") {
    return HpReplayPredictionMode::persistence;
  }
  if (value == "c2h-ht") {
    return HpReplayPredictionMode::c2h_hoeffding_tree;
  }
  if (value == "c2h-arf") {
    return HpReplayPredictionMode::c2h_arf;
  }
  throw std::invalid_argument("unknown replay prediction mode: " + value);
}

inline const char* hp_replay_prediction_mode_name(
    HpReplayPredictionMode mode) {
  switch (mode) {
    case HpReplayPredictionMode::direct_arf:
      return "direct-arf";
    case HpReplayPredictionMode::persistence:
      return "persistence";
    case HpReplayPredictionMode::c2h_hoeffding_tree:
      return "c2h-ht";
    case HpReplayPredictionMode::c2h_arf:
      return "c2h-arf";
  }
  throw std::invalid_argument("invalid replay prediction mode");
}

inline HpReplayAdaptationProfile parse_hp_replay_adaptation_profile(
    const std::string& value) {
  if (value == "baseline") {
    return HpReplayAdaptationProfile::baseline;
  }
  if (value == "conservative") {
    return HpReplayAdaptationProfile::conservative;
  }
  if (value == "disabled") {
    return HpReplayAdaptationProfile::disabled;
  }
  throw std::invalid_argument("unknown replay adaptation profile: " + value);
}

inline const char* hp_replay_adaptation_profile_name(
    HpReplayAdaptationProfile profile) {
  switch (profile) {
    case HpReplayAdaptationProfile::baseline:
      return "baseline";
    case HpReplayAdaptationProfile::conservative:
      return "conservative";
    case HpReplayAdaptationProfile::disabled:
      return "disabled";
  }
  throw std::invalid_argument("invalid replay adaptation profile");
}

template <typename Detector, uint64_t Numerator, uint64_t Denominator>
struct HpReplayScaledDetectorFactory {
  static_assert(Denominator > 0, "detector delta denominator must be positive");
  using DetectorType = Detector;

  static Detector create() {
    return Detector(
        static_cast<double>(Numerator) /
        static_cast<double>(Denominator));
  }
};

template <typename WarningDetectorFactory, typename DriftDetectorFactory>
inline std::unique_ptr<Classifier> make_hp_replay_model_with_detectors(
    std::shared_ptr<ArfAdaptationTelemetry> adaptation_telemetry) {
  auto *classifier = new ARFClassifier<
      NUM_FEATURES,
      2,
      WarningDetectorFactory,
      DriftDetectorFactory>(
          HP_ARF_N_MODELS,
          HP_ARF_MAX_FEATURES,
          HP_ARF_SEED,
          HP_ARF_GRACE_PERIOD,
          HP_ARF_LAMBDA,
          HP_ARF_DELTA,
          HP_ARF_TAU,
          HP_ARF_MAX_SHARE_TO_SPLIT,
          HP_ARF_MIN_BRANCH_FRACTION,
          std::move(adaptation_telemetry));
  return std::make_unique<PipelineClassifier>(
      new StandardScaler<NUM_FEATURES>(), classifier);
}

inline std::unique_ptr<Classifier> make_hp_replay_model(
    std::shared_ptr<ArfAdaptationTelemetry> adaptation_telemetry,
    HpReplayAdaptationProfile profile =
        HpReplayAdaptationProfile::baseline) {
  using BaselineWarningFactory =
      DetectorFactory<ADWIN<5>, HP_ARF_WARNING_DELTA_PERMILLE>;
  using BaselineDriftFactory =
      DetectorFactory<ADWIN<5>, HP_ARF_DRIFT_DELTA_PERMILLE>;
  using ConservativeWarningFactory =
      HpReplayScaledDetectorFactory<ADWIN<5>, 1, 1000>;
  using ConservativeDriftFactory =
      HpReplayScaledDetectorFactory<ADWIN<5>, 1, 10000>;
  using DisabledFactory = DetectorFactory<NeverDriftDetector>;

  switch (profile) {
    case HpReplayAdaptationProfile::baseline:
      return make_hp_replay_model_with_detectors<
          BaselineWarningFactory, BaselineDriftFactory>(
              std::move(adaptation_telemetry));
    case HpReplayAdaptationProfile::conservative:
      return make_hp_replay_model_with_detectors<
          ConservativeWarningFactory, ConservativeDriftFactory>(
              std::move(adaptation_telemetry));
    case HpReplayAdaptationProfile::disabled:
      return make_hp_replay_model_with_detectors<
          DisabledFactory, DisabledFactory>(
              std::move(adaptation_telemetry));
  }
  throw std::invalid_argument("invalid replay adaptation profile");
}

inline std::unique_ptr<Classifier> make_hp_replay_residual_model(
    HpReplayPredictionMode mode,
    size_t tree_count,
    std::shared_ptr<ArfAdaptationTelemetry> adaptation_telemetry) {
  if (mode == HpReplayPredictionMode::c2h_hoeffding_tree) {
    if (tree_count != 1) {
      throw std::invalid_argument(
          "C2H Hoeffding tree replay requires one tree");
    }
    return std::make_unique<PipelineClassifier>(
        new StandardScaler<NUM_FEATURES>(),
        new HoeffdingTreeClassifier<NUM_FEATURES, 2>(
            HP_ARF_GRACE_PERIOD,
            HP_ARF_DELTA,
            HP_ARF_TAU,
            HP_ARF_MAX_SHARE_TO_SPLIT,
            HP_ARF_MIN_BRANCH_FRACTION));
  }
  if (mode == HpReplayPredictionMode::c2h_arf) {
    if (tree_count == 0 ||
        tree_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "C2H residual ARF tree count is out of range");
    }
    using DisabledFactory = DetectorFactory<NeverDriftDetector>;
    auto* classifier = new ARFClassifier<
        NUM_FEATURES, 2, DisabledFactory, DisabledFactory>(
            static_cast<int>(tree_count),
            HP_ARF_MAX_FEATURES,
            HP_ARF_SEED,
            HP_ARF_GRACE_PERIOD,
            HP_ARF_LAMBDA,
            HP_ARF_DELTA,
            HP_ARF_TAU,
            HP_ARF_MAX_SHARE_TO_SPLIT,
            HP_ARF_MIN_BRANCH_FRACTION,
            std::move(adaptation_telemetry));
    return std::make_unique<PipelineClassifier>(
        new StandardScaler<NUM_FEATURES>(), classifier);
  }
  throw std::invalid_argument(
      "residual model factory requires a C2H replay mode");
}

struct HpReplayTrace {
  HpTraceFileHeader header{};
  std::vector<HpTraceRecord> records;
};

struct HpReplayDecision {
  double final_hot_probability = 0.0;
  int8_t final_label = 0;
  bool residual_applied = false;
  bool cold_start_fallback = false;
};

inline bool hp_replay_base_hot(const HpTraceRecord& record) {
  return record.past_window_access_count >=
      record.future_access_threshold_at_prediction;
}

inline bool hp_replay_uses_residual_model(HpReplayPredictionMode mode) {
  return mode == HpReplayPredictionMode::c2h_hoeffding_tree ||
      mode == HpReplayPredictionMode::c2h_arf;
}

inline bool hp_replay_should_train(
    const HpTraceRecord& record,
    HpReplayPredictionMode mode) {
  if (mode == HpReplayPredictionMode::direct_arf) {
    return true;
  }
  if (mode == HpReplayPredictionMode::persistence) {
    return false;
  }
  return !hp_replay_base_hot(record);
}

inline HpReplayDecision hp_replay_decision(
    const HpTraceRecord& record,
    HpReplayPredictionMode mode,
    double model_hot_probability,
    bool model_cold_start_fallback,
    double hot_threshold) {
  if (!std::isfinite(model_hot_probability) ||
      model_hot_probability < 0.0 || model_hot_probability > 1.0 ||
      !std::isfinite(hot_threshold) ||
      hot_threshold < 0.0 || hot_threshold > 1.0) {
    throw std::invalid_argument("invalid replay decision probability");
  }

  if (mode == HpReplayPredictionMode::persistence) {
    const bool base_hot = hp_replay_base_hot(record);
    return HpReplayDecision{
        base_hot ? 1.0 : 0.0,
        static_cast<int8_t>(base_hot),
        false,
        false,
    };
  }

  if (hp_replay_uses_residual_model(mode)) {
    if (hp_replay_base_hot(record)) {
      return HpReplayDecision{1.0, 1, false, false};
    }
    return HpReplayDecision{
        model_hot_probability,
        static_cast<int8_t>(
            !model_cold_start_fallback &&
            model_hot_probability >= hot_threshold),
        true,
        model_cold_start_fallback,
    };
  }

  return HpReplayDecision{
      model_hot_probability,
      static_cast<int8_t>(
          !model_cold_start_fallback &&
          model_hot_probability >= hot_threshold),
      false,
      model_cold_start_fallback,
  };
}

struct HpReplayRecordOverride {
  uint64_t io_sequence = 0;
  double feature_0 = 0.0;
  int8_t actual_label = 0;
  uint64_t future_access_threshold_at_prediction = 1;
};

inline std::vector<HpReplayRecordOverride> read_hp_replay_overrides(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error(
        "cannot open replay override file: " + path.string());
  }
  std::string header;
  std::getline(input, header);
  if (!header.empty() && header.back() == '\r') {
    header.pop_back();
  }
  if (header !=
      "io_sequence\tfeature_0\tactual_label\tfuture_access_threshold_at_prediction") {
    throw std::runtime_error("invalid replay override header");
  }
  std::vector<HpReplayRecordOverride> overrides;
  uint64_t io_sequence = 0;
  double feature_0 = 0.0;
  int actual_label = 0;
  uint64_t future_access_threshold_at_prediction = 0;
  while (input >> io_sequence >> feature_0 >> actual_label >>
         future_access_threshold_at_prediction) {
    if (!std::isfinite(feature_0) ||
        future_access_threshold_at_prediction == 0 ||
        (actual_label != 0 && actual_label != 1)) {
      throw std::runtime_error("invalid replay override value");
    }
    overrides.push_back(HpReplayRecordOverride{
        io_sequence,
        feature_0,
        static_cast<int8_t>(actual_label),
        future_access_threshold_at_prediction,
    });
  }
  if (!input.eof()) {
    throw std::runtime_error("malformed replay override row");
  }
  return overrides;
}

inline void apply_hp_replay_overrides(
    HpReplayTrace& trace,
    const std::vector<HpReplayRecordOverride>& overrides) {
  if (trace.records.size() != overrides.size()) {
    throw std::runtime_error(
        "replay override count does not match Trace record count");
  }
  for (size_t index = 0; index < trace.records.size(); ++index) {
    auto& record = trace.records[index];
    const auto& replacement = overrides[index];
    if (record.io_sequence != replacement.io_sequence) {
      throw std::runtime_error(
          "replay override io_sequence does not match Trace order");
    }
    if (!std::isfinite(replacement.feature_0) ||
        replacement.future_access_threshold_at_prediction == 0 ||
        (replacement.actual_label != 0 && replacement.actual_label != 1)) {
      throw std::runtime_error("invalid replay override value");
    }
    record.features[0] = replacement.feature_0;
    record.actual_label = replacement.actual_label;
    record.future_access_threshold_at_prediction =
        replacement.future_access_threshold_at_prediction;
  }
}

enum class HpReplayEventType : uint8_t {
  prediction = 0,
  training = 1,
};

struct HpReplayEvent {
  uint64_t timestamp_ns = 0;
  uint64_t io_sequence = 0;
  size_t record_index = 0;
  HpReplayEventType type = HpReplayEventType::prediction;
};

class HpReplaySnapshotSchedule {
 public:
  HpReplaySnapshotSchedule(
      uint64_t start_time_ns,
      uint64_t sample_interval,
      uint64_t max_interval_ns)
      : last_publish_time_ns_(start_time_ns),
        sample_interval_(sample_interval),
        max_interval_ns_(max_interval_ns) {
    if (sample_interval_ == 0 || max_interval_ns_ == 0) {
      throw std::invalid_argument(
          "snapshot publish intervals must be positive");
    }
  }

  bool record_training(uint64_t now_ns) {
    ++trained_since_publish_;
    const bool sample_count_due =
        trained_since_publish_ >= sample_interval_;
    const bool time_due = now_ns >= last_publish_time_ns_ &&
        now_ns - last_publish_time_ns_ >= max_interval_ns_;
    if (!sample_count_due && !time_due) {
      return false;
    }
    trained_since_publish_ = 0;
    last_publish_time_ns_ = now_ns;
    ++publish_count_;
    return true;
  }

  uint64_t publish_count() const {
    return publish_count_;
  }

 private:
  uint64_t last_publish_time_ns_ = 0;
  uint64_t trained_since_publish_ = 0;
  uint64_t sample_interval_ = 0;
  uint64_t max_interval_ns_ = 0;
  uint64_t publish_count_ = 0;
};

struct HpReplayOptions {
  uint64_t snapshot_sample_interval =
      HP_REPLAY_SNAPSHOT_SAMPLE_INTERVAL;
  uint64_t snapshot_max_interval_ns =
      HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS;
  bool require_matching_config = true;
  HpReplayAdaptationProfile adaptation_profile =
      HpReplayAdaptationProfile::baseline;
  HpReplayPredictionMode prediction_mode =
      HpReplayPredictionMode::direct_arf;
  size_t residual_tree_count = 10;
  double correction_threshold = 0.50;
  std::array<bool, NUM_FEATURES> disabled_features{};

  void disable_feature(size_t index) {
    if (index >= disabled_features.size()) {
      throw std::out_of_range("replay feature index is out of range");
    }
    disabled_features[index] = true;
  }
};

struct HpReplayRecordResult {
  double replayed_hot_probability = 0.0;
  double residual_hot_probability = 0.0;
  int8_t replayed_label = 0;
  int8_t base_label = 0;
  bool residual_applied = false;
  bool cold_start_fallback = false;
};

struct HpReplayResult {
  std::vector<HpReplayRecordResult> records;
  uint64_t trained_sample_count = 0;
  uint64_t residual_eligible_sample_count = 0;
  uint64_t residual_trained_sample_count = 0;
  uint64_t snapshot_publish_count = 0;
  ArfAdaptationStats adaptation_stats;
};

struct HpReplayParityMetrics {
  uint64_t record_count = 0;
  uint64_t class_agreement_count = 0;
  uint64_t cold_start_fallback_count = 0;
  double class_agreement = 0.0;
  double probability_mae = 0.0;
  double probability_rmse = 0.0;
  double probability_abs_error_p95 = 0.0;
  double online_accuracy = 0.0;
  double replay_accuracy = 0.0;
  double accuracy_delta = 0.0;
  double online_hot_ratio = 0.0;
  double replay_hot_ratio = 0.0;
  double actual_hot_ratio = 0.0;
};

inline void validate_hp_replay_header(const HpTraceFileHeader& header) {
  if (std::memcmp(header.magic, HP_TRACE_MAGIC, sizeof(header.magic)) != 0) {
    throw std::runtime_error("invalid Heat Predictor Trace magic");
  }
  if (header.schema_version != HP_TRACE_SCHEMA_VERSION) {
    throw std::runtime_error("unsupported Heat Predictor Trace schema");
  }
  if (header.header_size != sizeof(HpTraceFileHeader)) {
    throw std::runtime_error("Heat Predictor Trace header size mismatch");
  }
  if (header.record_size != sizeof(HpTraceRecord)) {
    throw std::runtime_error("Heat Predictor Trace record size mismatch");
  }
  if (header.feature_count != NUM_FEATURES) {
    throw std::runtime_error("Heat Predictor Trace feature count mismatch");
  }
}

inline void validate_hp_replay_record(const HpTraceRecord& record) {
  if (record.outcome != static_cast<uint8_t>(HpTraceOutcome::evaluated)) {
    throw std::runtime_error("replay requires evaluated Trace records");
  }
  if (record.io_sequence == 0 ||
      record.prediction_time_ns > record.label_deadline_ns ||
      record.label_deadline_ns > record.label_completion_time_ns) {
    throw std::runtime_error("invalid Trace record sequence or timestamps");
  }
  if (record.predicted_label < 0 || record.predicted_label > 1 ||
      record.actual_label < 0 || record.actual_label > 1) {
    throw std::runtime_error("Trace labels must be binary");
  }
  if (!std::isfinite(record.predicted_hot_probability) ||
      record.predicted_hot_probability < 0.0 ||
      record.predicted_hot_probability > 1.0 ||
      !std::isfinite(record.hot_predict_threshold) ||
      record.hot_predict_threshold < 0.0 ||
      record.hot_predict_threshold > 1.0) {
    throw std::runtime_error("invalid Trace probability or threshold");
  }
  for (size_t index = 0; index < NUM_FEATURES; ++index) {
    if (!std::isfinite(record.features[index])) {
      throw std::runtime_error("Trace feature must be finite");
    }
  }
}

inline HpReplayTrace read_hp_trace(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open Trace file: " + path.string());
  }

  HpReplayTrace trace;
  input.read(reinterpret_cast<char*>(&trace.header), sizeof(trace.header));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(trace.header))) {
    throw std::runtime_error("Trace file is shorter than its header");
  }
  validate_hp_replay_header(trace.header);

  std::unordered_set<uint64_t> io_sequences;
  while (true) {
    HpTraceRecord record{};
    input.read(reinterpret_cast<char*>(&record), sizeof(record));
    const auto bytes = input.gcount();
    if (bytes == 0 && input.eof()) {
      break;
    }
    if (bytes != static_cast<std::streamsize>(sizeof(record))) {
      throw std::runtime_error("Trace file ends with a partial record");
    }
    validate_hp_replay_record(record);
    if (!io_sequences.insert(record.io_sequence).second) {
      throw std::runtime_error("Trace contains duplicate io_sequence");
    }
    trace.records.push_back(record);
  }
  if (trace.records.empty()) {
    throw std::runtime_error("Trace contains no evaluated records");
  }
  return trace;
}

inline std::vector<HpReplayEvent> make_replay_events(
    const HpReplayTrace& trace) {
  std::vector<HpReplayEvent> events;
  events.reserve(trace.records.size() * 2);
  for (size_t index = 0; index < trace.records.size(); ++index) {
    const auto& record = trace.records[index];
    events.push_back(HpReplayEvent{
        record.prediction_time_ns,
        record.io_sequence,
        index,
        HpReplayEventType::prediction,
    });
    events.push_back(HpReplayEvent{
        record.label_completion_time_ns,
        record.io_sequence,
        index,
        HpReplayEventType::training,
    });
  }
  std::sort(events.begin(), events.end(),
      [](const HpReplayEvent& left, const HpReplayEvent& right) {
        if (left.timestamp_ns != right.timestamp_ns) {
          return left.timestamp_ns < right.timestamp_ns;
        }
        if (left.type != right.type) {
          return left.type == HpReplayEventType::prediction;
        }
        if (left.io_sequence != right.io_sequence) {
          return left.io_sequence < right.io_sequence;
        }
        return left.record_index < right.record_index;
      });
  return events;
}

inline std::pair<double, bool> replay_hot_probability(
    Classifier& model,
    const std::vector<double>& features) {
  std::vector<double> probabilities;
  model.predict_proba_one_into(features, probabilities);
  if (probabilities.size() != 2) {
    throw std::runtime_error("replay model returned the wrong class count");
  }

  double total = 0.0;
  for (const double probability : probabilities) {
    if (!std::isfinite(probability) || probability < 0.0 ||
        probability > 1.0) {
      throw std::runtime_error("replay model returned invalid probability");
    }
    total += probability;
  }
  if (!std::isfinite(total)) {
    throw std::runtime_error("replay model probability sum is invalid");
  }
  if (total <= std::numeric_limits<double>::epsilon()) {
    if (total != 0.0) {
      throw std::runtime_error("replay model returned a malformed vote sum");
    }
    return {0.0, true};
  }
  return {probabilities[1] / total, false};
}

inline std::vector<double> hp_replay_features(
    const HpTraceRecord& record,
    const HpReplayOptions& options) {
  std::vector<double> features(
      record.features, record.features + NUM_FEATURES);
  for (size_t index = 0; index < features.size(); ++index) {
    if (options.disabled_features[index]) {
      features[index] = 0.0;
    }
  }
  return features;
}

inline HpReplayResult replay_hp_trace(
    const HpReplayTrace& trace,
    const HpReplayOptions& options = {}) {
  validate_hp_replay_header(trace.header);
  if (options.require_matching_config &&
      trace.header.config_hash != hp_trace_config_hash()) {
    throw std::runtime_error(
        "Trace configuration does not match the replay build");
  }
  if (trace.records.empty()) {
    throw std::runtime_error("cannot replay an empty Trace");
  }
  for (const auto& record : trace.records) {
    validate_hp_replay_record(record);
  }
  if (!std::isfinite(options.correction_threshold) ||
      options.correction_threshold < 0.0 ||
      options.correction_threshold > 1.0) {
    throw std::invalid_argument(
        "replay correction threshold must be in [0, 1]");
  }

  auto adaptation_telemetry =
      std::make_shared<ArfAdaptationTelemetry>();
  std::unique_ptr<Classifier> train_model;
  if (options.prediction_mode == HpReplayPredictionMode::direct_arf) {
    train_model = make_hp_replay_model(
        adaptation_telemetry, options.adaptation_profile);
  } else if (hp_replay_uses_residual_model(options.prediction_mode)) {
    const size_t tree_count =
        options.prediction_mode ==
                HpReplayPredictionMode::c2h_hoeffding_tree
            ? 1 : options.residual_tree_count;
    train_model = make_hp_replay_residual_model(
        options.prediction_mode, tree_count, adaptation_telemetry);
  }
  std::unique_ptr<Classifier> prediction_snapshot =
      train_model != nullptr
          ? train_model->clone_for_prediction()
          : nullptr;
  HpReplaySnapshotSchedule snapshot_schedule(
      trace.header.start_monotonic_time_ns,
      options.snapshot_sample_interval,
      options.snapshot_max_interval_ns);

  HpReplayResult result;
  result.records.resize(trace.records.size());
  const auto events = make_replay_events(trace);
  for (const auto& event : events) {
    const auto& record = trace.records[event.record_index];
    const std::vector<double> features =
        hp_replay_features(record, options);
    if (event.type == HpReplayEventType::prediction) {
      double model_hot_probability = 0.0;
      bool model_cold_start_fallback = false;
      const bool needs_model_prediction =
          options.prediction_mode == HpReplayPredictionMode::direct_arf ||
          (hp_replay_uses_residual_model(options.prediction_mode) &&
           !hp_replay_base_hot(record));
      if (needs_model_prediction) {
        if (prediction_snapshot == nullptr) {
          throw std::runtime_error(
              "replay prediction mode has no prediction model");
        }
        const auto prediction =
            replay_hot_probability(*prediction_snapshot, features);
        model_hot_probability = prediction.first;
        model_cold_start_fallback = prediction.second;
      }
      const double decision_threshold =
          hp_replay_uses_residual_model(options.prediction_mode)
              ? options.correction_threshold
              : record.hot_predict_threshold;
      const auto decision = hp_replay_decision(
          record,
          options.prediction_mode,
          model_hot_probability,
          model_cold_start_fallback,
          decision_threshold);
      auto& replayed = result.records[event.record_index];
      replayed.replayed_hot_probability =
          decision.final_hot_probability;
      replayed.residual_hot_probability =
          decision.residual_applied ? model_hot_probability : 0.0;
      replayed.replayed_label = decision.final_label;
      replayed.base_label =
          static_cast<int8_t>(hp_replay_base_hot(record));
      replayed.residual_applied = decision.residual_applied;
      replayed.cold_start_fallback =
          decision.cold_start_fallback;
      continue;
    }

    if (!hp_replay_should_train(record, options.prediction_mode)) {
      continue;
    }
    if (train_model == nullptr) {
      throw std::runtime_error(
          "replay training mode has no training model");
    }
    if (hp_replay_uses_residual_model(options.prediction_mode)) {
      ++result.residual_eligible_sample_count;
    }
    train_model->learn_one(
        features, static_cast<int>(record.actual_label));
    ++result.trained_sample_count;
    if (hp_replay_uses_residual_model(options.prediction_mode)) {
      ++result.residual_trained_sample_count;
    }
    if (snapshot_schedule.record_training(event.timestamp_ns)) {
      prediction_snapshot = train_model->clone_for_prediction();
    }
  }
  result.snapshot_publish_count = snapshot_schedule.publish_count();
  result.adaptation_stats = adaptation_telemetry->snapshot();
  return result;
}

inline HpReplayParityMetrics calculate_hp_replay_parity(
    const HpReplayTrace& trace,
    const HpReplayResult& result) {
  if (trace.records.empty() ||
      trace.records.size() != result.records.size()) {
    throw std::runtime_error(
        "Trace and replay result record counts do not match");
  }

  HpReplayParityMetrics metrics;
  metrics.record_count = trace.records.size();
  uint64_t online_correct = 0;
  uint64_t replay_correct = 0;
  uint64_t online_hot = 0;
  uint64_t replay_hot = 0;
  uint64_t actual_hot = 0;
  double absolute_error_sum = 0.0;
  double squared_error_sum = 0.0;
  std::vector<double> absolute_errors;
  absolute_errors.reserve(trace.records.size());

  for (size_t index = 0; index < trace.records.size(); ++index) {
    const auto& online = trace.records[index];
    const auto& replayed = result.records[index];
    if (!std::isfinite(replayed.replayed_hot_probability) ||
        replayed.replayed_hot_probability < 0.0 ||
        replayed.replayed_hot_probability > 1.0 ||
        replayed.replayed_label < 0 || replayed.replayed_label > 1) {
      throw std::runtime_error("replay result contains an invalid prediction");
    }

    metrics.class_agreement_count +=
        online.predicted_label == replayed.replayed_label;
    metrics.cold_start_fallback_count += replayed.cold_start_fallback;
    online_correct += online.predicted_label == online.actual_label;
    replay_correct += replayed.replayed_label == online.actual_label;
    online_hot += online.predicted_label == 1;
    replay_hot += replayed.replayed_label == 1;
    actual_hot += online.actual_label == 1;

    const double absolute_error = std::abs(
        replayed.replayed_hot_probability -
        online.predicted_hot_probability);
    absolute_error_sum += absolute_error;
    squared_error_sum += absolute_error * absolute_error;
    absolute_errors.push_back(absolute_error);
  }

  const double count = static_cast<double>(metrics.record_count);
  metrics.class_agreement =
      static_cast<double>(metrics.class_agreement_count) / count;
  metrics.probability_mae = absolute_error_sum / count;
  metrics.probability_rmse = std::sqrt(squared_error_sum / count);
  std::sort(absolute_errors.begin(), absolute_errors.end());
  const size_t p95_index = static_cast<size_t>(
      std::ceil(0.95 * count)) - 1;
  metrics.probability_abs_error_p95 = absolute_errors[p95_index];
  metrics.online_accuracy = static_cast<double>(online_correct) / count;
  metrics.replay_accuracy = static_cast<double>(replay_correct) / count;
  metrics.accuracy_delta =
      metrics.replay_accuracy - metrics.online_accuracy;
  metrics.online_hot_ratio = static_cast<double>(online_hot) / count;
  metrics.replay_hot_ratio = static_cast<double>(replay_hot) / count;
  metrics.actual_hot_ratio = static_cast<double>(actual_hot) / count;
  return metrics;
}

inline void write_hp_replay_tsv(
    std::ostream& output,
    const HpReplayTrace& trace,
    const HpReplayResult& result) {
  if (trace.records.size() != result.records.size()) {
    throw std::runtime_error(
        "Trace and replay result record counts do not match");
  }
  output << "io_sequence\tobject_key_hash\tprediction_time_ns"
         << "\tlabel_completion_time_ns\tonline_hot_probability"
         << "\treplay_hot_probability\tprobability_abs_error"
         << "\thot_predict_threshold\tonline_label\tbase_label"
         << "\tresidual_hot_probability\tresidual_applied\treplay_label"
         << "\tactual_label\tcold_start_fallback\n";
  output << std::setprecision(17);
  for (size_t index = 0; index < trace.records.size(); ++index) {
    const auto& online = trace.records[index];
    const auto& replayed = result.records[index];
    output << online.io_sequence << '\t'
           << online.object_key_hash << '\t'
           << online.prediction_time_ns << '\t'
           << online.label_completion_time_ns << '\t'
           << online.predicted_hot_probability << '\t'
           << replayed.replayed_hot_probability << '\t'
           << std::abs(replayed.replayed_hot_probability -
                       online.predicted_hot_probability) << '\t'
           << online.hot_predict_threshold << '\t'
           << static_cast<int>(online.predicted_label) << '\t'
           << static_cast<int>(replayed.base_label) << '\t'
           << replayed.residual_hot_probability << '\t'
           << static_cast<int>(replayed.residual_applied) << '\t'
           << static_cast<int>(replayed.replayed_label) << '\t'
           << static_cast<int>(online.actual_label) << '\t'
           << static_cast<int>(replayed.cold_start_fallback) << '\n';
  }
  if (!output) {
    throw std::runtime_error("failed to write replay TSV");
  }
}

#endif
