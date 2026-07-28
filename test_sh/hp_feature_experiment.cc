#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "hp_feature_experiment.h"

namespace {

struct CliOptions {
  std::filesystem::path trace_path;
  std::filesystem::path output_path;
  HpExperimentProfile profile;
  bool has_profile = false;
};

CliOptions parse_options(int argc, char** argv) {
  if (argc != 6) {
    throw std::runtime_error(
        "usage: hp_feature_experiment TRACE.bin "
        "--profile PROFILE --output replay.tsv");
  }
  CliOptions options;
  options.trace_path = argv[1];
  bool has_output = false;
  for (int index = 2; index < argc; index += 2) {
    const std::string name = argv[index];
    const std::string value = argv[index + 1];
    if (name == "--profile") {
      if (options.has_profile) {
        throw std::runtime_error(
            "--profile may be specified only once");
      }
      options.profile = parse_hp_experiment_profile(value);
      options.has_profile = true;
      continue;
    }
    if (name == "--output") {
      if (has_output) {
        throw std::runtime_error(
            "--output may be specified only once");
      }
      options.output_path = value;
      has_output = true;
      continue;
    }
    throw std::runtime_error("unknown feature experiment option: " + name);
  }
  if (!options.has_profile || !has_output ||
      options.trace_path.empty() || options.output_path.empty()) {
    throw std::runtime_error(
        "Trace, profile and output paths must not be empty");
  }
  return options;
}

void write_summary(
    const HpReplayTrace& trace,
    const HpReplayResult& result,
    const HpReplayParityMetrics& metrics,
    const CliOptions& options) {
  std::cout << std::setprecision(17)
            << "profile=" << options.profile.name << '\n'
            << "feature_count=" << options.profile.feature_count << '\n'
            << "osd_id=" << trace.header.osd_id << '\n'
            << "records=" << metrics.record_count << '\n'
            << "trained_samples=" << result.trained_sample_count << '\n'
            << "snapshot_publishes="
            << result.snapshot_publish_count << '\n'
            << "cold_start_fallbacks="
            << metrics.cold_start_fallback_count << '\n'
            << "online_accuracy=" << metrics.online_accuracy << '\n'
            << "experiment_accuracy=" << metrics.replay_accuracy << '\n'
            << "experiment_hot_ratio=" << metrics.replay_hot_ratio << '\n'
            << "actual_hot_ratio=" << metrics.actual_hot_ratio << '\n'
            << "output=" << options.output_path.string() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions options = parse_options(argc, argv);
    const HpReplayTrace trace = read_hp_trace(options.trace_path);
    const HpReplayResult result =
        replay_hp_feature_experiment(trace, options.profile);
    const HpReplayParityMetrics metrics =
        calculate_hp_replay_parity(trace, result);

    const auto parent = options.output_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    std::ofstream output(options.output_path, std::ios::trunc);
    if (!output) {
      throw std::runtime_error(
          "cannot create feature experiment output: " +
          options.output_path.string());
    }
    write_hp_replay_tsv(output, trace, result);
    output.close();
    if (!output) {
      throw std::runtime_error(
          "cannot finish feature experiment output: " +
          options.output_path.string());
    }
    write_summary(trace, result, metrics, options);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hp_feature_experiment: " << error.what() << '\n';
    return 1;
  }
}
