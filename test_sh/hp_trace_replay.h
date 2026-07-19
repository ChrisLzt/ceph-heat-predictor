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

struct HpReplayTrace {
  HpTraceFileHeader header{};
  std::vector<HpTraceRecord> records;
};

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
      HeatPredictor::MODEL_UPDATE_REPORT_INTERVAL;
  uint64_t snapshot_max_interval_ns =
      HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS;
  bool require_matching_config = true;
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
  int8_t replayed_label = 0;
  bool cold_start_fallback = false;
};

struct HpReplayResult {
  std::vector<HpReplayRecordResult> records;
  uint64_t trained_sample_count = 0;
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
      trace.header.config_hash != HeatPredictor::trace_config_hash()) {
    throw std::runtime_error(
        "Trace configuration does not match the replay build");
  }
  if (trace.records.empty()) {
    throw std::runtime_error("cannot replay an empty Trace");
  }
  for (const auto& record : trace.records) {
    validate_hp_replay_record(record);
  }

  auto adaptation_telemetry =
      std::make_shared<ArfAdaptationTelemetry>();
  std::unique_ptr<Classifier> train_model(
      HeatPredictor::make_model(adaptation_telemetry));
  std::unique_ptr<Classifier> prediction_snapshot =
      train_model->clone_for_prediction();
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
      const auto [hot_probability, cold_start_fallback] =
          replay_hot_probability(*prediction_snapshot, features);
      auto& replayed = result.records[event.record_index];
      replayed.replayed_hot_probability = hot_probability;
      replayed.replayed_label = static_cast<int8_t>(
          hot_probability >= record.hot_predict_threshold);
      replayed.cold_start_fallback = cold_start_fallback;
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
         << "\thot_predict_threshold\tonline_label\treplay_label"
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
           << static_cast<int>(replayed.replayed_label) << '\t'
           << static_cast<int>(online.actual_label) << '\t'
           << static_cast<int>(replayed.cold_start_fallback) << '\n';
  }
  if (!output) {
    throw std::runtime_error("failed to write replay TSV");
  }
}

#endif
