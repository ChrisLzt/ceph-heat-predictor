#ifndef CEPH_TEST_SH_HP_FEATURE_EXPERIMENT_H
#define CEPH_TEST_SH_HP_FEATURE_EXPERIMENT_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hp_trace_replay.h"

static constexpr size_t HP_EXPERIMENT_FEATURE_PAST_ACCESS_MARGIN = 0;
static constexpr size_t HP_EXPERIMENT_FEATURE_PREVIOUS_INTERVAL = 1;
static constexpr size_t HP_EXPERIMENT_FEATURE_CURRENT_HEAT = 2;
static constexpr size_t HP_EXPERIMENT_FEATURE_ACCESS_RATE_TREND = 3;
static constexpr size_t HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN = 4;
static constexpr size_t HP_EXPERIMENT_FEATURE_EXPIRING_ACCESS_FRACTION = 5;
static constexpr size_t HP_EXPERIMENT_FEATURE_INTERVAL_CONTRACTION = 6;
static constexpr size_t
    HP_EXPERIMENT_FEATURE_ROBUST_PROJECTED_COUNT_MARGIN = 7;
static constexpr size_t HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT = 8;
static constexpr size_t HP_EXPERIMENT_FEATURE_SECOND_PREVIOUS_INTERVAL = 9;
static constexpr size_t HP_EXPERIMENT_FEATURE_EXACT_INTERVAL_CONTRACTION = 10;
static constexpr size_t HP_EXPERIMENT_FEATURE_COUNT = 11;

static constexpr uint64_t HP_EXPERIMENT_SHORT_WINDOW_NS =
    2ULL * 1000 * 1000 * 1000;
static constexpr uint64_t HP_EXPERIMENT_EXPIRY_LOOKAHEAD_NS =
    2ULL * 1000 * 1000 * 1000;

using HpExperimentFeatureValues =
    std::array<double, HP_EXPERIMENT_FEATURE_COUNT>;

struct HpExperimentProfile {
  const char* name = "";
  std::array<size_t, HP_EXPERIMENT_FEATURE_COUNT> feature_indices{};
  size_t feature_count = 0;
};

inline HpExperimentProfile hp_experiment_profile(
    const char* name,
    std::initializer_list<size_t> indices) {
  if (indices.size() > HP_EXPERIMENT_FEATURE_COUNT) {
    throw std::invalid_argument("too many experiment features");
  }
  HpExperimentProfile profile;
  profile.name = name;
  profile.feature_count = indices.size();
  std::copy(indices.begin(), indices.end(), profile.feature_indices.begin());
  return profile;
}

inline HpExperimentProfile parse_hp_experiment_profile(
    const std::string& value) {
  constexpr size_t count = HP_EXPERIMENT_FEATURE_PAST_ACCESS_MARGIN;
  constexpr size_t interval = HP_EXPERIMENT_FEATURE_PREVIOUS_INTERVAL;
  constexpr size_t heat = HP_EXPERIMENT_FEATURE_CURRENT_HEAT;
  constexpr size_t rate = HP_EXPERIMENT_FEATURE_ACCESS_RATE_TREND;
  constexpr size_t projected =
      HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN;
  constexpr size_t expiry =
      HP_EXPERIMENT_FEATURE_EXPIRING_ACCESS_FRACTION;
  constexpr size_t contraction =
      HP_EXPERIMENT_FEATURE_INTERVAL_CONTRACTION;
  constexpr size_t robust_projected =
      HP_EXPERIMENT_FEATURE_ROBUST_PROJECTED_COUNT_MARGIN;
  constexpr size_t short_count =
      HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT;
  constexpr size_t second_interval =
      HP_EXPERIMENT_FEATURE_SECOND_PREVIOUS_INTERVAL;
  constexpr size_t exact_contraction =
      HP_EXPERIMENT_FEATURE_EXACT_INTERVAL_CONTRACTION;

  if (value == "base") {
    return hp_experiment_profile("base", {count, interval, heat});
  }
  if (value == "base+rate-trend") {
    return hp_experiment_profile(
        "base+rate-trend", {count, interval, heat, rate});
  }
  if (value == "base+projected-margin") {
    return hp_experiment_profile(
        "base+projected-margin", {count, interval, heat, projected});
  }
  if (value == "base+robust-projected-margin") {
    return hp_experiment_profile(
        "base+robust-projected-margin",
        {count, interval, heat, robust_projected});
  }
  if (value == "base+short-count") {
    return hp_experiment_profile(
        "base+short-count", {count, interval, heat, short_count});
  }
  if (value == "base+rate-trend+short-count") {
    return hp_experiment_profile(
        "base+rate-trend+short-count",
        {count, interval, heat, rate, short_count});
  }
  if (value == "base+projected-margin+short-count") {
    return hp_experiment_profile(
        "base+projected-margin+short-count",
        {count, interval, heat, projected, short_count});
  }
  if (value == "base+robust-projected-margin+short-count") {
    return hp_experiment_profile(
        "base+robust-projected-margin+short-count",
        {count, interval, heat, robust_projected, short_count});
  }
  if (value == "base+second-interval") {
    return hp_experiment_profile(
        "base+second-interval",
        {count, interval, heat, second_interval});
  }
  if (value == "base+exact-interval-contraction") {
    return hp_experiment_profile(
        "base+exact-interval-contraction",
        {count, interval, heat, exact_contraction});
  }
  if (value ==
      "base+projected-margin+short-count+second-interval") {
    return hp_experiment_profile(
        "base+projected-margin+short-count+second-interval",
        {count, interval, heat, projected, short_count,
         second_interval});
  }
  if (value ==
      "base+projected-margin+short-count"
      "+exact-interval-contraction") {
    return hp_experiment_profile(
        "base+projected-margin+short-count"
        "+exact-interval-contraction",
        {count, interval, heat, projected, short_count,
         exact_contraction});
  }
  if (value == "base+expiring-fraction") {
    return hp_experiment_profile(
        "base+expiring-fraction", {count, interval, heat, expiry});
  }
  if (value == "base+interval-contraction") {
    return hp_experiment_profile(
        "base+interval-contraction",
        {count, interval, heat, contraction});
  }
  if (value == "base+projected-margin+rate-trend") {
    return hp_experiment_profile(
        "base+projected-margin+rate-trend",
        {count, interval, heat, projected, rate});
  }
  if (value == "base+projected-margin+expiring-fraction") {
    return hp_experiment_profile(
        "base+projected-margin+expiring-fraction",
        {count, interval, heat, projected, expiry});
  }
  if (value == "base+projected-margin+interval-contraction") {
    return hp_experiment_profile(
        "base+projected-margin+interval-contraction",
        {count, interval, heat, projected, contraction});
  }
  if (value ==
      "base+projected-margin+expiring-fraction"
      "+interval-contraction") {
    return hp_experiment_profile(
        "base+projected-margin+expiring-fraction"
        "+interval-contraction",
        {count, interval, heat, projected, expiry, contraction});
  }
  if (value == "all") {
    return hp_experiment_profile(
        "all",
        {count, interval, heat, rate, projected, expiry, contraction,
         robust_projected, short_count, second_interval,
         exact_contraction});
  }
  if (value == "all-rate-projected") {
    return hp_experiment_profile(
        "all-rate-projected",
        {count, interval, heat, rate, projected});
  }
  if (value == "all-rate-expiry") {
    return hp_experiment_profile(
        "all-rate-expiry",
        {count, interval, heat, rate, expiry});
  }
  if (value == "all-rate-contraction") {
    return hp_experiment_profile(
        "all-rate-contraction",
        {count, interval, heat, rate, contraction});
  }
  throw std::invalid_argument("unknown feature experiment profile: " + value);
}

inline std::vector<size_t> hp_experiment_prediction_order(
    const HpReplayTrace& trace) {
  std::vector<size_t> order(trace.records.size());
  for (size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  std::sort(order.begin(), order.end(),
      [&trace](size_t left, size_t right) {
        const auto& lhs = trace.records[left];
        const auto& rhs = trace.records[right];
        if (lhs.prediction_time_ns != rhs.prediction_time_ns) {
          return lhs.prediction_time_ns < rhs.prediction_time_ns;
        }
        if (lhs.io_sequence != rhs.io_sequence) {
          return lhs.io_sequence < rhs.io_sequence;
        }
        return left < right;
      });
  return order;
}

inline std::vector<HpExperimentFeatureValues>
reconstruct_hp_experiment_features(const HpReplayTrace& trace) {
  std::vector<HpExperimentFeatureValues> result(trace.records.size());
  std::unordered_map<uint64_t, std::deque<uint64_t>> histories;
  histories.reserve(std::min<size_t>(trace.records.size(), 65536));

  for (const size_t record_index : hp_experiment_prediction_order(trace)) {
    const auto& record = trace.records[record_index];
    auto& history = histories[record.object_key_hash];
    const uint64_t now_ns = record.prediction_time_ns;
    const uint64_t long_window_start =
        now_ns > HP_FUTURE_LABEL_WINDOW_NS
            ? now_ns - HP_FUTURE_LABEL_WINDOW_NS
            : 0;
    while (!history.empty() &&
           (now_ns >= HP_FUTURE_LABEL_WINDOW_NS
                ? history.front() <= long_window_start
                : false)) {
      history.pop_front();
    }
    if (history.size() != record.past_window_access_count) {
      throw std::runtime_error(
          "Trace history does not match past_window_access_count for io " +
          std::to_string(record.io_sequence));
    }

    const auto short_begin =
        now_ns >= HP_EXPERIMENT_SHORT_WINDOW_NS
            ? std::upper_bound(
                history.begin(),
                history.end(),
                now_ns - HP_EXPERIMENT_SHORT_WINDOW_NS)
            : history.begin();
    const size_t short_count =
        static_cast<size_t>(history.end() - short_begin);
    const size_t long_count = history.size();

    const double short_seconds =
        static_cast<double>(HP_EXPERIMENT_SHORT_WINDOW_NS) / 1e9;
    const double long_seconds =
        static_cast<double>(HP_FUTURE_LABEL_WINDOW_NS) / 1e9;
    const double short_rate =
        static_cast<double>(short_count) / short_seconds;
    const double long_rate =
        static_cast<double>(long_count) / long_seconds;
    const double threshold = static_cast<double>(
        std::max<uint64_t>(
            record.future_access_threshold_at_prediction, 1));

    size_t expiring_count = 0;
    if (now_ns >=
        HP_FUTURE_LABEL_WINDOW_NS - HP_EXPERIMENT_EXPIRY_LOOKAHEAD_NS) {
      const uint64_t expiry_cutoff =
          now_ns -
          (HP_FUTURE_LABEL_WINDOW_NS -
           HP_EXPERIMENT_EXPIRY_LOOKAHEAD_NS);
      expiring_count = static_cast<size_t>(
          std::upper_bound(
              history.begin(), history.end(), expiry_cutoff) -
          history.begin());
    }

    double interval_contraction = 0.0;
    double second_interval_encoded = 0.0;
    double exact_interval_contraction = 0.0;
    if (history.size() >= 2 && now_ns >= history.back()) {
      const double recent_interval_seconds =
          static_cast<double>(now_ns - history.back()) / 1e9;
      const double mean_interval_seconds =
          static_cast<double>(history.back() - history.front()) /
          static_cast<double>(history.size() - 1) / 1e9;
      interval_contraction =
          std::log2(1.0 + mean_interval_seconds) -
          std::log2(1.0 + recent_interval_seconds);

      const double second_interval_seconds =
          static_cast<double>(
              history.back() - history[history.size() - 2]) /
          1e9;
      second_interval_encoded =
          1.0 + std::log2(1.0 + second_interval_seconds);
      exact_interval_contraction =
          std::log2(1.0 + second_interval_seconds) -
          std::log2(1.0 + recent_interval_seconds);
    }

    HpExperimentFeatureValues values{};
    for (size_t index = 0; index < NUM_FEATURES; ++index) {
      values[index] = record.features[index];
    }
    values[HP_EXPERIMENT_FEATURE_ACCESS_RATE_TREND] =
        std::log2(1.0 + short_rate) - std::log2(1.0 + long_rate);
    values[HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN] =
        std::log2(1.0 + short_rate * long_seconds) -
        std::log2(1.0 + threshold);
    const double confidence_scale =
        std::max(1.0, threshold / 2.0);
    const double short_rate_confidence =
        static_cast<double>(short_count) /
        (static_cast<double>(short_count) + confidence_scale);
    const double robust_projected_rate =
        long_rate +
        short_rate_confidence * std::max(0.0, short_rate - long_rate);
    values[HP_EXPERIMENT_FEATURE_ROBUST_PROJECTED_COUNT_MARGIN] =
        std::log2(1.0 + robust_projected_rate * long_seconds) -
        std::log2(1.0 + threshold);
    values[HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT] =
        std::log2(1.0 + static_cast<double>(short_count));
    values[HP_EXPERIMENT_FEATURE_EXPIRING_ACCESS_FRACTION] =
        long_count == 0
            ? 0.0
            : static_cast<double>(expiring_count) /
                static_cast<double>(long_count);
    values[HP_EXPERIMENT_FEATURE_INTERVAL_CONTRACTION] =
        interval_contraction;
    values[HP_EXPERIMENT_FEATURE_SECOND_PREVIOUS_INTERVAL] =
        second_interval_encoded;
    values[HP_EXPERIMENT_FEATURE_EXACT_INTERVAL_CONTRACTION] =
        exact_interval_contraction;
    result[record_index] = values;
    history.push_back(now_ns);
  }
  return result;
}

inline std::vector<double> hp_experiment_select_features(
    const HpExperimentFeatureValues& values,
    const HpExperimentProfile& profile) {
  if (profile.feature_count == 0 ||
      profile.feature_count > HP_EXPERIMENT_FEATURE_COUNT) {
    throw std::invalid_argument("invalid experiment feature count");
  }
  std::vector<double> selected;
  selected.reserve(profile.feature_count);
  for (size_t index = 0; index < profile.feature_count; ++index) {
    const size_t feature_index = profile.feature_indices[index];
    if (feature_index >= values.size()) {
      throw std::out_of_range("experiment feature index is out of range");
    }
    selected.push_back(values[feature_index]);
  }
  return selected;
}

template <size_t FeatureCount>
inline std::unique_ptr<Classifier> make_hp_feature_experiment_model(
    std::shared_ptr<ArfAdaptationTelemetry> adaptation_telemetry) {
  using DisabledFactory = DetectorFactory<NeverDriftDetector>;
  auto* classifier = new ARFClassifier<
      FeatureCount, 2, DisabledFactory, DisabledFactory>(
          HP_ARF_N_MODELS,
          static_cast<int>(FeatureCount),
          HP_ARF_SEED,
          HP_ARF_GRACE_PERIOD,
          HP_ARF_LAMBDA,
          HP_ARF_DELTA,
          HP_ARF_TAU,
          HP_ARF_MAX_SHARE_TO_SPLIT,
          HP_ARF_MIN_BRANCH_FRACTION,
          std::move(adaptation_telemetry));
  return std::make_unique<PipelineClassifier>(
      new StandardScaler<FeatureCount>(), classifier);
}

template <size_t FeatureCount>
inline HpReplayResult replay_hp_feature_experiment_with_dimension(
    const HpReplayTrace& trace,
    const HpExperimentProfile& profile,
    const std::vector<HpExperimentFeatureValues>& all_features) {
  if (profile.feature_count != FeatureCount ||
      all_features.size() != trace.records.size()) {
    throw std::invalid_argument(
        "feature experiment profile dimension mismatch");
  }

  auto adaptation_telemetry =
      std::make_shared<ArfAdaptationTelemetry>();
  auto train_model =
      make_hp_feature_experiment_model<FeatureCount>(
          adaptation_telemetry);
  auto prediction_snapshot = train_model->clone_for_prediction();
  HpReplaySnapshotSchedule snapshot_schedule(
      trace.header.start_monotonic_time_ns,
      HP_REPLAY_SNAPSHOT_SAMPLE_INTERVAL,
      HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS);

  HpReplayResult result;
  result.records.resize(trace.records.size());
  for (const auto& event : make_replay_events(trace)) {
    const auto& record = trace.records[event.record_index];
    const auto features = hp_experiment_select_features(
        all_features[event.record_index], profile);
    if (event.type == HpReplayEventType::prediction) {
      const auto [probability, fallback] =
          replay_hot_probability(*prediction_snapshot, features);
      auto& replayed = result.records[event.record_index];
      replayed.replayed_hot_probability = probability;
      replayed.replayed_label = static_cast<int8_t>(
          !fallback && probability >= record.hot_predict_threshold);
      replayed.base_label =
          static_cast<int8_t>(hp_replay_base_hot(record));
      replayed.cold_start_fallback = fallback;
      continue;
    }

    train_model->learn_one(
        features, static_cast<int>(record.actual_label));
    ++result.trained_sample_count;
    if (snapshot_schedule.record_training(event.timestamp_ns)) {
      prediction_snapshot = train_model->clone_for_prediction();
    }
  }
  result.snapshot_publish_count = snapshot_schedule.publish_count();
  result.adaptation_stats = adaptation_telemetry->snapshot();
  return result;
}

inline HpReplayResult replay_hp_feature_experiment(
    const HpReplayTrace& trace,
    const HpExperimentProfile& profile) {
  validate_hp_replay_header(trace.header);
  if (trace.header.config_hash != hp_trace_config_hash()) {
    throw std::runtime_error(
        "Trace configuration does not match the experiment build");
  }
  if (trace.records.empty()) {
    throw std::runtime_error(
        "cannot run a feature experiment on an empty Trace");
  }
  for (const auto& record : trace.records) {
    validate_hp_replay_record(record);
  }

  const auto all_features =
      reconstruct_hp_experiment_features(trace);
  switch (profile.feature_count) {
    case 3:
      return replay_hp_feature_experiment_with_dimension<3>(
          trace, profile, all_features);
    case 4:
      return replay_hp_feature_experiment_with_dimension<4>(
          trace, profile, all_features);
    case 5:
      return replay_hp_feature_experiment_with_dimension<5>(
          trace, profile, all_features);
    case 6:
      return replay_hp_feature_experiment_with_dimension<6>(
          trace, profile, all_features);
    case 7:
      return replay_hp_feature_experiment_with_dimension<7>(
          trace, profile, all_features);
    case 8:
      return replay_hp_feature_experiment_with_dimension<8>(
          trace, profile, all_features);
    case 9:
      return replay_hp_feature_experiment_with_dimension<9>(
          trace, profile, all_features);
    case 10:
      return replay_hp_feature_experiment_with_dimension<10>(
          trace, profile, all_features);
    case 11:
      return replay_hp_feature_experiment_with_dimension<11>(
          trace, profile, all_features);
    default:
      throw std::invalid_argument(
          "unsupported feature experiment dimension");
  }
}

#endif
