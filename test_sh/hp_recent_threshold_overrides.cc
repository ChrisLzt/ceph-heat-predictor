#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "hp_recent_threshold_replay.h"

int main(int argc, char** argv) {
  try {
    if (argc != 4 || std::string(argv[2]) != "--output") {
      throw std::runtime_error(
          "usage: hp_recent_threshold_overrides TRACE.bin "
          "--output overrides.tsv");
    }
    const std::filesystem::path trace_path = argv[1];
    const std::filesystem::path output_path = argv[3];
    HpReplayTrace trace = read_hp_trace(trace_path);
    const HpRecentThresholdReplayResult result =
        make_hp_recent_threshold_overrides(trace);
    if (result.past_count_mismatch_count != 0) {
      throw std::runtime_error(
          "reconstructed recent counts do not match Trace history");
    }

    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    std::ofstream output(output_path, std::ios::trunc);
    if (!output) {
      throw std::runtime_error(
          "cannot create recent threshold override file");
    }
    output << std::setprecision(17);
    write_hp_recent_threshold_overrides(output, result);
    output.close();
    if (!output) {
      throw std::runtime_error(
          "cannot finish recent threshold override file");
    }

    const double record_count =
        static_cast<double>(result.overrides.size());
    std::cout << std::setprecision(17)
              << "trace=" << trace_path.string() << '\n'
              << "output=" << output_path.string() << '\n'
              << "records=" << result.overrides.size() << '\n'
              << "past_count_mismatches="
              << result.past_count_mismatch_count << '\n'
              << "threshold_changes="
              << result.threshold_change_count << '\n'
              << "label_changes=" << result.label_change_count << '\n'
              << "label_agreement="
              << (record_count == 0.0
                      ? 0.0
                      : 1.0 -
                          static_cast<double>(result.label_change_count) /
                              record_count)
              << '\n'
              << "baseline_hot_ratio="
              << (record_count == 0.0
                      ? 0.0
                      : static_cast<double>(result.baseline_hot_count) /
                          record_count)
              << '\n'
              << "recent_hot_ratio="
              << (record_count == 0.0
                      ? 0.0
                      : static_cast<double>(result.recent_hot_count) /
                          record_count)
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hp_recent_threshold_overrides: "
              << error.what() << '\n';
    return 1;
  }
}
