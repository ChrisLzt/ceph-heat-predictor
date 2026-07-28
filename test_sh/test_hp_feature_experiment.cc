#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hp_feature_experiment.h"

namespace {

constexpr uint64_t SECOND_NS = 1000000000ULL;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(
    double actual,
    double expected,
    const std::string& message) {
  if (std::abs(actual - expected) > 1e-12) {
    throw std::runtime_error(
        message + ": actual=" + std::to_string(actual) +
        " expected=" + std::to_string(expected));
  }
}

HpTraceRecord make_record(
    uint64_t sequence,
    uint64_t object_key,
    uint64_t prediction_time_ns,
    uint64_t past_count,
    uint64_t threshold = 3) {
  HpTraceRecord record{};
  record.io_sequence = sequence;
  record.object_key_hash = object_key;
  record.prediction_time_ns = prediction_time_ns;
  record.label_deadline_ns =
      prediction_time_ns + HP_FUTURE_LABEL_WINDOW_NS;
  record.label_completion_time_ns = record.label_deadline_ns;
  record.future_access_threshold_at_prediction = threshold;
  record.past_window_access_count = past_count;
  record.outcome = static_cast<uint8_t>(HpTraceOutcome::evaluated);
  record.actual_label = 0;
  record.predicted_label = 0;
  record.hot_predict_threshold = HP_HOT_PREDICT_THRESHOLD;
  record.features[0] = static_cast<double>(sequence);
  record.features[1] = static_cast<double>(sequence + 10);
  record.features[2] = static_cast<double>(sequence + 20);
  return record;
}

HpReplayTrace make_history_trace() {
  HpReplayTrace trace;
  std::memcpy(
      trace.header.magic, HP_TRACE_MAGIC, sizeof(trace.header.magic));
  trace.header.schema_version = HP_TRACE_SCHEMA_VERSION;
  trace.header.header_size = sizeof(HpTraceFileHeader);
  trace.header.record_size = sizeof(HpTraceRecord);
  trace.header.feature_count = NUM_FEATURES;
  trace.header.start_monotonic_time_ns = 0;
  trace.header.config_hash = hp_trace_config_hash();
  trace.records = {
      make_record(5, 42, 10 * SECOND_NS, 3),
      make_record(1, 42, 0, 0),
      make_record(4, 42, 9 * SECOND_NS, 3),
      make_record(2, 42, 1 * SECOND_NS, 1),
      make_record(3, 42, 8500 * SECOND_NS / 1000, 2),
  };
  return trace;
}

void test_reconstructs_strict_history_features_without_future_data() {
  const auto trace = make_history_trace();
  const auto reconstructed =
      reconstruct_hp_experiment_features(trace);
  require(reconstructed.size() == trace.records.size(),
          "feature reconstruction lost records");

  const auto& values = reconstructed[0];
  require_close(
      values[HP_EXPERIMENT_FEATURE_PAST_ACCESS_MARGIN],
      trace.records[0].features[0],
      "base count margin must come from the recorded prediction");
  require_close(
      values[HP_EXPERIMENT_FEATURE_PREVIOUS_INTERVAL],
      trace.records[0].features[1],
      "base previous interval must come from the recorded prediction");
  require_close(
      values[HP_EXPERIMENT_FEATURE_CURRENT_HEAT],
      trace.records[0].features[2],
      "base heat must come from the recorded prediction");

  const double short_rate = 2.0 / 2.0;
  const double long_rate = 3.0 / 10.0;
  require_close(
      values[HP_EXPERIMENT_FEATURE_ACCESS_RATE_TREND],
      std::log2(1.0 + short_rate) - std::log2(1.0 + long_rate),
      "short/long access-rate trend is wrong");
  require_close(
      values[HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN],
      std::log2(1.0 + short_rate * 10.0) - std::log2(1.0 + 3.0),
      "projected future-count margin is wrong");
  const double confidence = 2.0 / (2.0 + 1.5);
  const double robust_rate =
      long_rate + confidence * (short_rate - long_rate);
  require_close(
      values[HP_EXPERIMENT_FEATURE_ROBUST_PROJECTED_COUNT_MARGIN],
      std::log2(1.0 + robust_rate * 10.0) -
          std::log2(1.0 + 3.0),
      "K/2-regularized projected future-count margin is wrong");
  require_close(
      values[HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT],
      std::log2(1.0 + 2.0),
      "short access count feature is wrong");
  require_close(
      values[HP_EXPERIMENT_FEATURE_EXPIRING_ACCESS_FRACTION],
      1.0 / 3.0,
      "imminent expiry fraction is wrong");
  require_close(
      values[HP_EXPERIMENT_FEATURE_INTERVAL_CONTRACTION],
      std::log2(1.0 + 4.0) - std::log2(1.0 + 1.0),
      "interval contraction is wrong");
  require_close(
      values[HP_EXPERIMENT_FEATURE_SECOND_PREVIOUS_INTERVAL],
      1.0 + std::log2(1.0 + 0.5),
      "second previous interval is wrong");
  require_close(
      values[HP_EXPERIMENT_FEATURE_EXACT_INTERVAL_CONTRACTION],
      std::log2(1.0 + 0.5) - std::log2(1.0 + 1.0),
      "exact interval contraction is wrong");
}

void test_early_history_does_not_drop_timestamp_zero() {
  const auto reconstructed =
      reconstruct_hp_experiment_features(make_history_trace());
  const auto& values = reconstructed[3];
  require_close(
      values[HP_EXPERIMENT_FEATURE_ACCESS_RATE_TREND],
      std::log2(1.0 + 0.5) - std::log2(1.0 + 0.1),
      "early short window dropped a valid timestamp-zero access");
  require_close(
      values[HP_EXPERIMENT_FEATURE_EXPIRING_ACCESS_FRACTION],
      0.0,
      "early history incorrectly marked an access as expiring");
  require_close(
      values[HP_EXPERIMENT_FEATURE_SECOND_PREVIOUS_INTERVAL],
      0.0,
      "second previous interval must be zero without two prior accesses");
  require_close(
      values[HP_EXPERIMENT_FEATURE_EXACT_INTERVAL_CONTRACTION],
      0.0,
      "exact interval contraction must be zero without two prior accesses");
}

void test_robust_projection_clamps_confidence_scale_for_small_k() {
  auto trace = make_history_trace();
  trace.records[0].future_access_threshold_at_prediction = 1;
  const auto reconstructed =
      reconstruct_hp_experiment_features(trace);
  const double short_rate = 2.0 / 2.0;
  const double long_rate = 3.0 / 10.0;
  const double confidence = 2.0 / (2.0 + 1.0);
  const double robust_rate =
      long_rate + confidence * (short_rate - long_rate);
  require_close(
      reconstructed[0]
          [HP_EXPERIMENT_FEATURE_ROBUST_PROJECTED_COUNT_MARGIN],
      std::log2(1.0 + robust_rate * 10.0) - std::log2(1.0 + 1.0),
      "robust projection did not clamp K/2 confidence scale to one");
}

void test_reconstruction_rejects_missing_history() {
  auto trace = make_history_trace();
  trace.records[0].past_window_access_count = 4;
  bool rejected = false;
  try {
    (void)reconstruct_hp_experiment_features(trace);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected,
          "reconstruction accepted a Trace with missing object history");
}

void test_profiles_select_expected_features() {
  const auto base = parse_hp_experiment_profile("base");
  require(base.feature_count == 3,
          "base profile must retain the production dimension");
  require(base.feature_indices[0] ==
              HP_EXPERIMENT_FEATURE_PAST_ACCESS_MARGIN &&
              base.feature_indices[1] ==
                  HP_EXPERIMENT_FEATURE_PREVIOUS_INTERVAL &&
              base.feature_indices[2] ==
                  HP_EXPERIMENT_FEATURE_CURRENT_HEAT,
          "base profile changed production feature ordering");

  const auto all = parse_hp_experiment_profile("all");
  require(all.feature_count == HP_EXPERIMENT_FEATURE_COUNT,
          "all profile must enable every candidate feature");

  const auto expiry =
      parse_hp_experiment_profile("base+expiring-fraction");
  require(expiry.feature_count == 4 &&
              expiry.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_EXPIRING_ACCESS_FRACTION,
          "expiry profile selected the wrong candidate");

  const auto robust_projected =
      parse_hp_experiment_profile("base+robust-projected-margin");
  require(robust_projected.feature_count == 4 &&
              robust_projected.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_ROBUST_PROJECTED_COUNT_MARGIN,
          "robust projected profile selected the wrong candidate");

  const auto short_count =
      parse_hp_experiment_profile("base+short-count");
  require(short_count.feature_count == 4 &&
              short_count.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT,
          "short-count profile selected the wrong candidate");

  const auto rate_short =
      parse_hp_experiment_profile("base+rate-trend+short-count");
  require(rate_short.feature_count == 5 &&
              rate_short.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_ACCESS_RATE_TREND &&
              rate_short.feature_indices[4] ==
                  HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT,
          "rate-trend short-count profile changed feature ordering");

  const auto projected_short =
      parse_hp_experiment_profile(
          "base+projected-margin+short-count");
  require(projected_short.feature_count == 5 &&
              projected_short.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN &&
              projected_short.feature_indices[4] ==
                  HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT,
          "projected short-count profile changed feature ordering");

  const auto robust_projected_short =
      parse_hp_experiment_profile(
          "base+robust-projected-margin+short-count");
  require(robust_projected_short.feature_count == 5 &&
              robust_projected_short.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_ROBUST_PROJECTED_COUNT_MARGIN &&
              robust_projected_short.feature_indices[4] ==
                  HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT,
          "robust projected short-count profile changed feature ordering");

  const auto second_interval =
      parse_hp_experiment_profile("base+second-interval");
  require(second_interval.feature_count == 4 &&
              second_interval.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_SECOND_PREVIOUS_INTERVAL,
          "second-interval profile selected the wrong candidate");

  const auto exact_contraction =
      parse_hp_experiment_profile("base+exact-interval-contraction");
  require(exact_contraction.feature_count == 4 &&
              exact_contraction.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_EXACT_INTERVAL_CONTRACTION,
          "exact-contraction profile selected the wrong candidate");

  const auto projected_short_second =
      parse_hp_experiment_profile(
          "base+projected-margin+short-count+second-interval");
  require(projected_short_second.feature_count == 6 &&
              projected_short_second.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN &&
              projected_short_second.feature_indices[4] ==
                  HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT &&
              projected_short_second.feature_indices[5] ==
                  HP_EXPERIMENT_FEATURE_SECOND_PREVIOUS_INTERVAL,
          "projected short-count second-interval profile changed ordering");

  const auto projected_short_exact =
      parse_hp_experiment_profile(
          "base+projected-margin+short-count"
          "+exact-interval-contraction");
  require(projected_short_exact.feature_count == 6 &&
              projected_short_exact.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN &&
              projected_short_exact.feature_indices[4] ==
                  HP_EXPERIMENT_FEATURE_SHORT_ACCESS_COUNT &&
              projected_short_exact.feature_indices[5] ==
                  HP_EXPERIMENT_FEATURE_EXACT_INTERVAL_CONTRACTION,
          "projected short-count exact-contraction profile changed ordering");

  const auto projected_expiry_contraction =
      parse_hp_experiment_profile(
          "base+projected-margin+expiring-fraction"
          "+interval-contraction");
  require(projected_expiry_contraction.feature_count == 6 &&
              projected_expiry_contraction.feature_indices[3] ==
                  HP_EXPERIMENT_FEATURE_PROJECTED_COUNT_MARGIN &&
              projected_expiry_contraction.feature_indices[4] ==
                  HP_EXPERIMENT_FEATURE_EXPIRING_ACCESS_FRACTION &&
              projected_expiry_contraction.feature_indices[5] ==
                  HP_EXPERIMENT_FEATURE_INTERVAL_CONTRACTION,
          "combined profile changed candidate ordering");

  bool rejected = false;
  try {
    (void)parse_hp_experiment_profile("unknown");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unknown experiment profile was accepted");
}

void test_profile_replay_preserves_delayed_training() {
  auto trace = make_history_trace();
  const auto result = replay_hp_feature_experiment(
      trace, parse_hp_experiment_profile("base+rate-trend"));
  require(result.records.size() == trace.records.size(),
          "experiment replay lost prediction records");
  require(result.trained_sample_count == trace.records.size(),
          "experiment replay did not train every completed sample");
  require(result.residual_trained_sample_count == 0,
          "direct feature replay unexpectedly trained a residual model");
  for (const auto& record : result.records) {
    require(std::isfinite(record.replayed_hot_probability),
            "experiment replay produced a non-finite probability");
  }
}

}  // namespace

int main() {
  try {
    test_reconstructs_strict_history_features_without_future_data();
    test_early_history_does_not_drop_timestamp_zero();
    test_robust_projection_clamps_confidence_scale_for_small_k();
    test_reconstruction_rejects_missing_history();
    test_profiles_select_expected_features();
    test_profile_replay_preserves_delayed_training();
    std::cout << "PASS: hp feature experiment\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
