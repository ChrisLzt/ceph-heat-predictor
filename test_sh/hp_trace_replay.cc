#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "hp_trace_replay.h"

namespace {

struct CliOptions {
  std::filesystem::path trace_path;
  std::filesystem::path output_path;
  std::filesystem::path override_path;
  HpReplayOptions replay_options;
};

CliOptions parse_options(int argc, char** argv) {
  if (argc < 4) {
    throw std::runtime_error(
        "usage: hp_trace_replay TRACE.bin --output replay.tsv "
        "[--record-overrides overrides.tsv] "
        "[--drop-feature INDEX ...]");
  }
  CliOptions options;
  options.trace_path = argv[1];
  bool has_output = false;
  for (int index = 2; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::runtime_error("replay option is missing its value");
    }
    const std::string option = argv[index];
    const std::string value = argv[index + 1];
    if (option == "--output") {
      if (has_output) {
        throw std::runtime_error("--output may be specified only once");
      }
      options.output_path = value;
      has_output = true;
      continue;
    }
    if (option == "--drop-feature") {
      size_t parsed = 0;
      size_t feature_index = 0;
      try {
        feature_index = std::stoul(value, &parsed);
      } catch (const std::exception&) {
        throw std::runtime_error("invalid feature index: " + value);
      }
      if (parsed != value.size() || feature_index >= NUM_FEATURES) {
        throw std::runtime_error("invalid feature index: " + value);
      }
      if (options.replay_options.disabled_features[feature_index]) {
        throw std::runtime_error(
            "feature index is duplicated: " + value);
      }
      options.replay_options.disable_feature(feature_index);
      continue;
    }
    if (option == "--record-overrides") {
      if (!options.override_path.empty()) {
        throw std::runtime_error(
            "--record-overrides may be specified only once");
      }
      options.override_path = value;
      continue;
    }
    throw std::runtime_error("unknown replay option: " + option);
  }
  if (options.trace_path.empty() || !has_output ||
      options.output_path.empty()) {
    throw std::runtime_error("Trace and output paths must not be empty");
  }
  return options;
}

std::string disabled_features_string(const HpReplayOptions& options) {
  std::string output;
  for (size_t index = 0; index < options.disabled_features.size(); ++index) {
    if (!options.disabled_features[index]) {
      continue;
    }
    if (!output.empty()) {
      output += ',';
    }
    output += std::to_string(index);
  }
  return output.empty() ? "none" : output;
}

void write_summary(
    const HpReplayTrace& trace,
    const HpReplayResult& result,
    const HpReplayParityMetrics& metrics,
    const CliOptions& options) {
  std::cout << std::setprecision(17)
            << "osd_id=" << trace.header.osd_id << '\n'
            << "session_id=" << trace.header.session_id << '\n'
            << "config_hash=" << trace.header.config_hash << '\n'
            << "arf_grace_period=" << HP_ARF_GRACE_PERIOD << '\n'
            << "arf_warning_delta="
            << static_cast<double>(HP_ARF_WARNING_DELTA_PERMILLE) / 1000.0
            << '\n'
            << "arf_drift_delta="
            << static_cast<double>(HP_ARF_DRIFT_DELTA_PERMILLE) / 1000.0
            << '\n'
            << "disabled_features="
            << disabled_features_string(options.replay_options) << '\n'
            << "record_overrides="
            << (options.override_path.empty()
                    ? "none" : options.override_path.string()) << '\n'
            << "output=" << options.output_path.string() << '\n'
            << "records=" << metrics.record_count << '\n'
            << "trained_samples=" << result.trained_sample_count << '\n'
            << "snapshot_publishes=" << result.snapshot_publish_count << '\n'
            << "arf_warnings="
            << result.adaptation_stats.warning_count << '\n'
            << "arf_drifts="
            << result.adaptation_stats.drift_count << '\n'
            << "arf_background_promotions="
            << result.adaptation_stats.background_promotion_count << '\n'
            << "arf_background_discards="
            << result.adaptation_stats.background_discard_count << '\n'
            << "arf_background_training_updates="
            << result.adaptation_stats.background_training_update_count
            << '\n'
            << "arf_active_backgrounds="
            << result.adaptation_stats.active_background_count << '\n'
            << "cold_start_fallbacks="
            << metrics.cold_start_fallback_count << '\n'
            << "class_agreement=" << metrics.class_agreement << '\n'
            << "probability_mae=" << metrics.probability_mae << '\n'
            << "probability_rmse=" << metrics.probability_rmse << '\n'
            << "probability_abs_error_p95="
            << metrics.probability_abs_error_p95 << '\n'
            << "online_accuracy=" << metrics.online_accuracy << '\n'
            << "replay_accuracy=" << metrics.replay_accuracy << '\n'
            << "accuracy_delta=" << metrics.accuracy_delta << '\n'
            << "online_hot_ratio=" << metrics.online_hot_ratio << '\n'
            << "replay_hot_ratio=" << metrics.replay_hot_ratio << '\n'
            << "actual_hot_ratio=" << metrics.actual_hot_ratio << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions options = parse_options(argc, argv);
    HpReplayTrace trace = read_hp_trace(options.trace_path);
    if (!options.override_path.empty()) {
      apply_hp_replay_overrides(
          trace, read_hp_replay_overrides(options.override_path));
    }
    const HpReplayResult result =
        replay_hp_trace(trace, options.replay_options);
    const HpReplayParityMetrics metrics =
        calculate_hp_replay_parity(trace, result);

    const auto parent = options.output_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    std::ofstream output(options.output_path, std::ios::trunc);
    if (!output) {
      throw std::runtime_error(
          "cannot create replay output: " + options.output_path.string());
    }
    write_hp_replay_tsv(output, trace, result);
    output.close();
    if (!output) {
      throw std::runtime_error(
          "cannot finish replay output: " + options.output_path.string());
    }
    write_summary(trace, result, metrics, options);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hp_trace_replay: " << error.what() << '\n';
    return 1;
  }
}
