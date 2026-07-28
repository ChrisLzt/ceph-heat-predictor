#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
        "[--adaptation-profile baseline|conservative|disabled] "
        "[--prediction-mode direct-arf|persistence|c2h-ht|c2h-arf] "
        "[--residual-trees N] [--correction-threshold P] "
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
    if (option == "--adaptation-profile") {
      options.replay_options.adaptation_profile =
          parse_hp_replay_adaptation_profile(value);
      continue;
    }
    if (option == "--prediction-mode") {
      options.replay_options.prediction_mode =
          parse_hp_replay_prediction_mode(value);
      continue;
    }
    if (option == "--residual-trees") {
      size_t parsed = 0;
      unsigned long long tree_count = 0;
      try {
        tree_count = std::stoull(value, &parsed);
      } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid residual tree count: " + value);
      }
      if (parsed != value.size() || tree_count == 0 ||
          tree_count > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(
            "invalid residual tree count: " + value);
      }
      options.replay_options.residual_tree_count =
          static_cast<size_t>(tree_count);
      continue;
    }
    if (option == "--correction-threshold") {
      size_t parsed = 0;
      double threshold = 0.0;
      try {
        threshold = std::stod(value, &parsed);
      } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid correction threshold: " + value);
      }
      if (parsed != value.size() || !std::isfinite(threshold) ||
          threshold < 0.0 || threshold > 1.0) {
        throw std::runtime_error(
            "invalid correction threshold: " + value);
      }
      options.replay_options.correction_threshold = threshold;
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
            << (options.replay_options.adaptation_profile ==
                        HpReplayAdaptationProfile::baseline
                    ? static_cast<double>(
                          HP_ARF_WARNING_DELTA_PERMILLE) / 1000.0
                    : options.replay_options.adaptation_profile ==
                              HpReplayAdaptationProfile::conservative
                          ? 0.001
                          : 0.0)
            << '\n'
            << "arf_drift_delta="
            << (options.replay_options.adaptation_profile ==
                        HpReplayAdaptationProfile::baseline
                    ? static_cast<double>(
                          HP_ARF_DRIFT_DELTA_PERMILLE) / 1000.0
                    : options.replay_options.adaptation_profile ==
                              HpReplayAdaptationProfile::conservative
                          ? 0.0001
                          : 0.0)
            << '\n'
            << "adaptation_profile="
            << hp_replay_adaptation_profile_name(
                   options.replay_options.adaptation_profile)
            << '\n'
            << "prediction_mode="
            << hp_replay_prediction_mode_name(
                   options.replay_options.prediction_mode)
            << '\n'
            << "residual_tree_count="
            << options.replay_options.residual_tree_count << '\n'
            << "correction_threshold="
            << options.replay_options.correction_threshold << '\n'
            << "disabled_features="
            << disabled_features_string(options.replay_options) << '\n'
            << "record_overrides="
            << (options.override_path.empty()
                    ? "none" : options.override_path.string()) << '\n'
            << "output=" << options.output_path.string() << '\n'
            << "records=" << metrics.record_count << '\n'
            << "trained_samples=" << result.trained_sample_count << '\n'
            << "residual_eligible_samples="
            << result.residual_eligible_sample_count << '\n'
            << "residual_trained_samples="
            << result.residual_trained_sample_count << '\n'
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
