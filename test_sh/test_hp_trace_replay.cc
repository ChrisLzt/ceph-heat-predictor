#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hp_recent_threshold_replay.h"
#include "hp_trace_replay.h"

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
    const char *)
{
  std::cerr << "ceph_assert failed: " << assertion
            << " at " << file << ":" << line << std::endl;
  std::abort();
}

} // namespace ceph

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

HpTraceFileHeader make_header() {
  HpTraceFileHeader header{};
  std::memcpy(header.magic, HP_TRACE_MAGIC, sizeof(header.magic));
  header.schema_version = HP_TRACE_SCHEMA_VERSION;
  header.header_size = sizeof(HpTraceFileHeader);
  header.record_size = sizeof(HpTraceRecord);
  header.feature_count = NUM_FEATURES;
  header.osd_id = 0;
  header.session_id = 1;
  header.start_wall_time_ns = 1000;
  header.start_monotonic_time_ns = 100;
  header.config_hash = hp_trace_config_hash();
  std::strncpy(header.phase, "fixture", sizeof(header.phase) - 1);
  return header;
}

HpTraceRecord make_record(
    uint64_t sequence,
    uint64_t prediction_time_ns,
    uint64_t completion_time_ns,
    int actual_label = 1) {
  HpTraceRecord record{};
  record.io_sequence = sequence;
  record.object_key_hash = sequence * 10;
  record.prediction_time_ns = prediction_time_ns;
  record.label_deadline_ns = completion_time_ns;
  record.label_completion_time_ns = completion_time_ns;
  for (size_t index = 0; index < NUM_FEATURES; ++index) {
    record.features[index] = static_cast<double>(sequence + index);
  }
  record.predicted_hot_probability = 0.5;
  record.hot_predict_threshold = 0.5;
  record.future_access_threshold_at_prediction = 2;
  record.future_window_access_count = actual_label == 1 ? 3 : 0;
  record.future_window_access_threshold = 2;
  record.outcome = static_cast<uint8_t>(HpTraceOutcome::evaluated);
  record.predicted_label = 1;
  record.actual_label = static_cast<int8_t>(actual_label);
  return record;
}

std::filesystem::path fixture_path(const std::string& name) {
  return std::filesystem::temp_directory_path() / name;
}

void write_trace(
    const std::filesystem::path& path,
    const HpTraceFileHeader& header,
    const std::vector<HpTraceRecord>& records) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(
      reinterpret_cast<const char*>(records.data()),
      static_cast<std::streamsize>(records.size() * sizeof(HpTraceRecord)));
  require(output.good(), "failed to write replay fixture");
}

void test_reader_and_event_order() {
  const auto path = fixture_path("hp_trace_replay_reader.bin");
  write_trace(path, make_header(), {
      make_record(2, 200, 300),
      make_record(1, 100, 200),
  });

  const HpReplayTrace trace = read_hp_trace(path);
  require(trace.records.size() == 2, "reader lost trace records");
  const auto events = make_replay_events(trace);
  require(events.size() == 4, "each evaluated record needs two events");
  require(events[0].type == HpReplayEventType::prediction &&
          events[0].io_sequence == 1,
          "earliest prediction must be first");
  require(events[1].type == HpReplayEventType::prediction &&
          events[1].io_sequence == 2,
          "prediction must precede training on a timestamp tie");
  require(events[2].type == HpReplayEventType::training &&
          events[2].io_sequence == 1,
          "tied training event must follow prediction");
  require(events[3].type == HpReplayEventType::training &&
          events[3].io_sequence == 2,
          "latest training event must be last");
  std::filesystem::remove(path);
}

void test_reader_rejects_bad_magic() {
  const auto path = fixture_path("hp_trace_replay_bad_magic.bin");
  auto header = make_header();
  header.magic[0] = 'X';
  write_trace(path, header, {make_record(1, 100, 200)});
  bool rejected = false;
  try {
    (void)read_hp_trace(path);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "reader accepted an invalid trace magic");
  std::filesystem::remove(path);
}

void test_snapshot_schedule_uses_count_or_time() {
  HpReplaySnapshotSchedule count_schedule(100, 2, 1000);
  require(!count_schedule.record_training(150),
          "one sample must not reach a two-sample publish interval");
  require(count_schedule.record_training(160),
          "the second sample must publish a snapshot");
  require(count_schedule.publish_count() == 1,
          "sample-triggered publish count is wrong");

  HpReplaySnapshotSchedule time_schedule(100, 100, 50);
  require(time_schedule.record_training(150),
          "elapsed freshness interval must publish a snapshot");
  require(time_schedule.publish_count() == 1,
          "time-triggered publish count is wrong");
}

void test_feature_mask_applies_to_prediction_and_training_input() {
  static_assert(NUM_FEATURES >= 2,
                "feature-mask probe needs at least two model features");
  const HpTraceRecord record = make_record(7, 100, 200, 1);
  HpReplayOptions options;
  options.disable_feature(0);
  options.disable_feature(NUM_FEATURES - 1);
  const auto features = hp_replay_features(record, options);
  require(features.size() == NUM_FEATURES,
          "feature mask must preserve the compiled feature dimension");
  for (size_t index = 0; index < NUM_FEATURES; ++index) {
    const double expected = index == 0 || index == NUM_FEATURES - 1
        ? 0.0 : record.features[index];
    require(features[index] == expected,
            "feature mask changed the wrong replay feature");
  }
  bool rejected = false;
  try {
    options.disable_feature(NUM_FEATURES);
  } catch (const std::out_of_range&) {
    rejected = true;
  }
  require(rejected, "out-of-range feature mask must be rejected");
}

void test_counterfactual_overrides_are_strict_and_scoped() {
  const auto override_path = fixture_path("hp_replay_overrides_crlf.tsv");
  {
    std::ofstream output(override_path, std::ios::binary | std::ios::trunc);
    output << "io_sequence\tfeature_0\tactual_label\tfuture_access_threshold_at_prediction\r\n"
           << "1\t7.5\t0\t120\r\n"
           << "2\t-1.5\t1\t140\r\n";
  }
  const auto parsed_overrides = read_hp_replay_overrides(override_path);
  require(parsed_overrides.size() == 2,
          "override reader must accept a CRLF TSV");
  std::filesystem::remove(override_path);

  HpReplayTrace trace;
  trace.header = make_header();
  trace.records = {
      make_record(1, 100, 200, 1),
      make_record(2, 110, 210, 0),
  };
  const double original_feature_1 = trace.records[0].features[1];
  apply_hp_replay_overrides(trace, parsed_overrides);
  require(trace.records[0].features[0] == 7.5 &&
          trace.records[0].features[1] == original_feature_1,
          "override must only replace feature zero");
  require(trace.records[0].actual_label == 0 &&
          trace.records[0].future_access_threshold_at_prediction == 120 &&
          trace.records[1].actual_label == 1 &&
          trace.records[1].future_access_threshold_at_prediction == 140,
          "override did not replace V2 labels and thresholds");

  bool rejected = false;
  try {
    apply_hp_replay_overrides(trace, {
        HpReplayRecordOverride{2, 0.0, 0, 100},
        HpReplayRecordOverride{1, 0.0, 0, 100},
    });
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "override sequence mismatch must be rejected");
}

void test_recent_threshold_overrides_use_strict_trailing_window() {
  HpReplayTrace trace;
  trace.header = make_header();
  trace.records = {
      make_record(1, 100, 200, 0),
      make_record(2, 110, 210, 0),
      make_record(3, 120, 220, 0),
      make_record(4, 100 + HP_FUTURE_LABEL_WINDOW_NS, 300, 0),
  };
  for (auto& record : trace.records) {
    record.object_key_hash = 42;
    record.future_window_access_count = 0;
    record.future_window_access_threshold = 1;
  }
  trace.records[0].past_window_access_count = 0;
  trace.records[1].past_window_access_count = 1;
  trace.records[2].past_window_access_count = 2;
  trace.records[3].past_window_access_count = 2;

  HpRecentThresholdReplayOptions options;
  options.minimum_positive_objects = 1;
  options.update_interval = 0;
  const auto generated =
      make_hp_recent_threshold_overrides(trace, options);
  require(generated.overrides.size() == trace.records.size(),
          "recent threshold override count is wrong");
  require(generated.past_count_mismatch_count == 0,
          "strict trailing reconstruction disagrees with Trace history");
  require(generated.overrides[0].future_access_threshold_at_prediction == 1,
          "cold-start recent threshold must begin at one");
  require(generated.overrides[3].feature_0 ==
              std::log2(1.0 + 2.0) - std::log2(1.0 + 1.0),
          "recent threshold override must rebuild feature zero");
}

void test_recent_threshold_overrides_preserve_original_record_order() {
  HpReplayTrace trace;
  trace.header = make_header();
  trace.records = {
      make_record(2, 200, 300, 1),
      make_record(1, 100, 200, 0),
  };
  trace.records[0].object_key_hash = 20;
  trace.records[1].object_key_hash = 10;
  trace.records[0].past_window_access_count = 0;
  trace.records[1].past_window_access_count = 0;

  HpRecentThresholdReplayOptions options;
  options.minimum_positive_objects = 1;
  const auto generated =
      make_hp_recent_threshold_overrides(trace, options);
  require(generated.overrides[0].io_sequence == 2 &&
          generated.overrides[1].io_sequence == 1,
          "recent threshold output must preserve Trace storage order");
}

void test_replay_adaptation_profiles_are_explicit() {
  require(parse_hp_replay_adaptation_profile("baseline") ==
              HpReplayAdaptationProfile::baseline,
          "baseline adaptation profile is missing");
  require(parse_hp_replay_adaptation_profile("conservative") ==
              HpReplayAdaptationProfile::conservative,
          "conservative adaptation profile is missing");
  require(parse_hp_replay_adaptation_profile("disabled") ==
              HpReplayAdaptationProfile::disabled,
          "disabled adaptation profile is missing");

  NeverDriftDetector detector;
  for (int index = 0; index < 1000; ++index) {
    detector.update(1.0);
  }
  require(!detector.drift_detected,
          "disabled adaptation detector must never report drift");

  bool rejected = false;
  try {
    (void)parse_hp_replay_adaptation_profile("unknown");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unknown adaptation profile must be rejected");
}

void test_replay_prediction_modes_and_c2h_gate() {
  require(parse_hp_replay_prediction_mode("direct-arf") ==
              HpReplayPredictionMode::direct_arf,
          "direct ARF replay mode is missing");
  require(parse_hp_replay_prediction_mode("persistence") ==
              HpReplayPredictionMode::persistence,
          "persistence replay mode is missing");
  require(parse_hp_replay_prediction_mode("c2h-ht") ==
              HpReplayPredictionMode::c2h_hoeffding_tree,
          "C2H Hoeffding tree replay mode is missing");
  require(parse_hp_replay_prediction_mode("c2h-arf") ==
              HpReplayPredictionMode::c2h_arf,
          "C2H ARF replay mode is missing");

  auto record = make_record(1, 100, 200, 0);
  record.future_access_threshold_at_prediction = 3;
  record.past_window_access_count = 2;
  require(!hp_replay_base_hot(record),
          "past count below K must be base-cold");
  record.past_window_access_count = 3;
  require(hp_replay_base_hot(record),
          "past count equal to K must be base-hot");

  const auto base_hot = hp_replay_decision(
      record, HpReplayPredictionMode::c2h_arf,
      0.01, false, 0.50);
  require(base_hot.final_label == 1 &&
              base_hot.final_hot_probability == 1.0 &&
              !base_hot.residual_applied &&
              !base_hot.cold_start_fallback,
          "C2H residual must not override a base-hot decision");

  record.past_window_access_count = 2;
  const auto corrected = hp_replay_decision(
      record, HpReplayPredictionMode::c2h_arf,
      0.75, false, 0.60);
  require(corrected.final_label == 1 &&
              corrected.final_hot_probability == 0.75 &&
              corrected.residual_applied &&
              !corrected.cold_start_fallback,
          "C2H residual must correct a confident base-cold prediction");

  const auto retained_cold = hp_replay_decision(
      record, HpReplayPredictionMode::c2h_hoeffding_tree,
      0.0, true, 0.50);
  require(retained_cold.final_label == 0 &&
              retained_cold.final_hot_probability == 0.0 &&
              retained_cold.residual_applied &&
              retained_cold.cold_start_fallback,
          "untrained residual model must preserve base-cold");
}

void test_residual_model_factory_and_training_gate() {
  auto record = make_record(1, 100, 200, 1);
  record.future_access_threshold_at_prediction = 3;
  record.past_window_access_count = 2;
  require(hp_replay_should_train(
              record, HpReplayPredictionMode::direct_arf),
          "direct ARF must train every evaluated sample");
  require(!hp_replay_should_train(
              record, HpReplayPredictionMode::persistence),
          "persistence mode must not train a model");
  require(hp_replay_should_train(
              record, HpReplayPredictionMode::c2h_hoeffding_tree) &&
              hp_replay_should_train(
                  record, HpReplayPredictionMode::c2h_arf),
          "C2H residual models must train base-cold samples");
  record.past_window_access_count = 3;
  require(!hp_replay_should_train(
              record, HpReplayPredictionMode::c2h_hoeffding_tree) &&
              !hp_replay_should_train(
                  record, HpReplayPredictionMode::c2h_arf),
          "C2H residual models must skip base-hot samples");

  auto exercise_model = [](HpReplayPredictionMode mode, size_t tree_count) {
    auto telemetry = std::make_shared<ArfAdaptationTelemetry>();
    auto model = make_hp_replay_residual_model(
        mode, tree_count, telemetry);
    require(model != nullptr, "residual model factory returned null");
    const std::vector<double> cold_features(NUM_FEATURES, 0.0);
    const std::vector<double> hot_features(NUM_FEATURES, 1.0);
    model->learn_one(cold_features, 0);
    model->learn_one(hot_features, 1);
    auto snapshot = model->clone_for_prediction();
    const auto [probability, fallback] =
        replay_hot_probability(*snapshot, hot_features);
    require(std::isfinite(probability) && !fallback,
            "trained residual snapshot returned no votes");
  };

  exercise_model(HpReplayPredictionMode::c2h_hoeffding_tree, 1);
  exercise_model(HpReplayPredictionMode::c2h_arf, 10);
  exercise_model(HpReplayPredictionMode::c2h_arf, 25);

  bool rejected = false;
  try {
    auto telemetry = std::make_shared<ArfAdaptationTelemetry>();
    (void)make_hp_replay_residual_model(
        HpReplayPredictionMode::c2h_arf, 0, telemetry);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "zero-tree residual ARF must be rejected");
}

void test_replay_modes_apply_base_gate_and_training_scope() {
  HpReplayTrace trace;
  trace.header = make_header();
  trace.records = {
      make_record(1, 100, 200, 0),
      make_record(2, 110, 210, 1),
  };
  trace.records[0].future_access_threshold_at_prediction = 2;
  trace.records[0].past_window_access_count = 2;
  trace.records[1].future_access_threshold_at_prediction = 2;
  trace.records[1].past_window_access_count = 1;

  HpReplayOptions persistence_options;
  persistence_options.prediction_mode =
      HpReplayPredictionMode::persistence;
  const auto persistence =
      replay_hp_trace(trace, persistence_options);
  require(persistence.trained_sample_count == 0 &&
              persistence.residual_eligible_sample_count == 0 &&
              persistence.residual_trained_sample_count == 0,
          "persistence replay must not train a model");
  require(persistence.records[0].base_label == 1 &&
              persistence.records[0].replayed_label == 1 &&
              persistence.records[1].base_label == 0 &&
              persistence.records[1].replayed_label == 0,
          "persistence replay must emit the base decision");

  HpReplayOptions residual_options;
  residual_options.prediction_mode =
      HpReplayPredictionMode::c2h_hoeffding_tree;
  residual_options.residual_tree_count = 1;
  const auto residual = replay_hp_trace(trace, residual_options);
  require(residual.trained_sample_count == 1 &&
              residual.residual_eligible_sample_count == 1 &&
              residual.residual_trained_sample_count == 1,
          "C2H replay must train only the base-cold sample");
  require(residual.records[0].base_label == 1 &&
              !residual.records[0].residual_applied &&
              residual.records[0].replayed_label == 1,
          "base-hot replay record must bypass the residual model");
  require(residual.records[1].base_label == 0 &&
              residual.records[1].residual_applied &&
              residual.records[1].cold_start_fallback &&
              residual.records[1].replayed_label == 0,
          "untrained residual model must retain a base-cold prediction");

  HpReplayOptions direct_options;
  direct_options.prediction_mode =
      HpReplayPredictionMode::direct_arf;
  direct_options.adaptation_profile =
      HpReplayAdaptationProfile::disabled;
  const auto direct = replay_hp_trace(trace, direct_options);
  require(direct.trained_sample_count == 2 &&
              direct.residual_eligible_sample_count == 0 &&
              direct.residual_trained_sample_count == 0,
          "direct ARF replay must train all samples");
}

void test_replay_publishes_trained_snapshot_deterministically() {
  HpReplayTrace trace;
  trace.header = make_header();
  trace.records = {
      make_record(1, 100, 200, 1),
      make_record(2, 110, 210, 0),
      make_record(3, 300, 400, 1),
  };
  HpReplayOptions options;
  options.snapshot_sample_interval = 2;
  options.snapshot_max_interval_ns = 1000;

  const HpReplayResult first = replay_hp_trace(trace, options);
  const HpReplayResult second = replay_hp_trace(trace, options);
  require(first.records.size() == 3,
          "replay result must preserve original record count");
  require(first.records[0].cold_start_fallback &&
          first.records[1].cold_start_fallback,
          "predictions before training must use the empty snapshot");
  require(!first.records[2].cold_start_fallback,
          "prediction after snapshot publish must see trained votes");
  require(first.snapshot_publish_count == 1,
          "two trained samples should publish one snapshot");
  require(first.adaptation_stats.warning_count == 0 &&
          first.adaptation_stats.drift_count == 0 &&
          first.adaptation_stats.background_promotion_count == 0 &&
          first.adaptation_stats.background_discard_count == 0 &&
          first.adaptation_stats.background_training_update_count == 0 &&
          first.adaptation_stats.active_background_count == 0,
          "short stable replay should export an empty adaptation snapshot");
  for (size_t index = 0; index < first.records.size(); ++index) {
    require(first.records[index].replayed_hot_probability ==
            second.records[index].replayed_hot_probability,
            "fixed seed replay must be deterministic and isolated");
  }
}

void test_parity_metrics_and_original_order_tsv() {
  HpReplayTrace trace;
  trace.header = make_header();
  trace.records = {
      make_record(4, 400, 500, 1),
      make_record(1, 100, 200, 0),
      make_record(3, 300, 400, 1),
      make_record(2, 200, 300, 0),
  };
  const std::array<double, 4> original_probability = {0.9, 0.1, 0.4, 0.2};
  const std::array<int8_t, 4> original_label = {1, 0, 0, 0};
  const std::array<double, 4> replay_probability = {0.8, 0.4, 0.7, 0.2};
  const std::array<int8_t, 4> replay_label = {1, 1, 1, 0};

  HpReplayResult result;
  result.records.resize(trace.records.size());
  for (size_t index = 0; index < trace.records.size(); ++index) {
    trace.records[index].predicted_hot_probability =
        original_probability[index];
    trace.records[index].predicted_label = original_label[index];
    result.records[index].replayed_hot_probability =
        replay_probability[index];
    result.records[index].replayed_label = replay_label[index];
  }

  const HpReplayParityMetrics metrics =
      calculate_hp_replay_parity(trace, result);
  require(std::abs(metrics.class_agreement - 0.5) < 1e-12,
          "class agreement calculation is wrong");
  require(std::abs(metrics.probability_mae - 0.175) < 1e-12,
          "probability MAE calculation is wrong");
  require(std::abs(metrics.probability_rmse - std::sqrt(0.0475)) < 1e-12,
          "probability RMSE calculation is wrong");
  require(std::abs(metrics.probability_abs_error_p95 - 0.3) < 1e-12,
          "probability P95 calculation is wrong");
  require(std::abs(metrics.online_accuracy - 0.75) < 1e-12 &&
          std::abs(metrics.replay_accuracy - 0.75) < 1e-12 &&
          std::abs(metrics.accuracy_delta) < 1e-12,
          "accuracy parity calculation is wrong");

  std::ostringstream output;
  write_hp_replay_tsv(output, trace, result);
  const std::string tsv = output.str();
  require(tsv.find(
              "io_sequence\tobject_key_hash\tprediction_time_ns"
              "\tlabel_completion_time_ns\tonline_hot_probability"
              "\treplay_hot_probability\tprobability_abs_error"
              "\thot_predict_threshold\tonline_label\tbase_label"
              "\tresidual_hot_probability\tresidual_applied"
              "\treplay_label\tactual_label\tcold_start_fallback") == 0,
          "TSV header is missing residual replay fields");
  const auto first_row = tsv.find('\n') + 1;
  require(tsv.compare(first_row, 2, "4\t") == 0,
          "TSV output must preserve original Trace record order");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--write-fixture") {
      write_trace(argv[2], make_header(), {
          make_record(1, 100, 200, 1),
          make_record(2, 110, 210, 0),
          make_record(3, 300, 400, 1),
      });
      return 0;
    }
    if (argc != 1) {
      throw std::runtime_error(
          "usage: test_hp_trace_replay [--write-fixture PATH]");
    }
    test_reader_and_event_order();
    test_reader_rejects_bad_magic();
    test_snapshot_schedule_uses_count_or_time();
    test_feature_mask_applies_to_prediction_and_training_input();
    test_counterfactual_overrides_are_strict_and_scoped();
    test_recent_threshold_overrides_use_strict_trailing_window();
    test_recent_threshold_overrides_preserve_original_record_order();
    test_replay_adaptation_profiles_are_explicit();
    test_replay_prediction_modes_and_c2h_gate();
    test_residual_model_factory_and_training_gate();
    test_replay_modes_apply_base_gate_and_training_scope();
    test_replay_publishes_trained_snapshot_deterministically();
    test_parity_metrics_and_original_order_tsv();
    std::cout << "PASS: hp trace replay\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
