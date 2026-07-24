#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "heatpredictor/hp_trace.h"
#include "heatpredictor/hp_trace_record.h"

namespace {

void require(bool condition, const char *message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

void require_close(double actual, double expected, const char *message)
{
  require(std::fabs(actual - expected) < 1e-9, message);
}

void test_evaluated_sample_mapping()
{
  PredictionSample item{
      23,
      456,
      321.0,
      120.0,
      4,
      500000000ULL,
      0.75,
      1,
  };
  EvaluatedSample evaluated{
      item,
      1,
      3,
      100,
      200,
      210,
      300.0,
      250.0,
      true,
  };

  const HpTraceRecord record = hp_trace_record_for_evaluated(evaluated);
  require(record.io_sequence == 23 && record.object_key_hash == 456,
          "trace identity must match the prediction sample");
  require(record.prediction_time_ns == 100 &&
              record.label_deadline_ns == 200 &&
              record.label_completion_time_ns == 210,
          "trace timing must match the evaluated sample");
  require_close(record.predicted_hot_probability, 0.75,
                "trace probability must be preserved");
  require_close(record.hot_predict_threshold, HP_HOT_PREDICT_THRESHOLD,
                "trace prediction threshold must match production");
  require_close(record.label_heat, 300.0,
                "trace label heat must be preserved");
  require_close(record.label_heat_threshold, 250.0,
                "trace label threshold must be preserved");
  require(record.actual_label == 1 && record.predicted_label == 1,
          "trace labels must be preserved");
  require(record.long_window_access_count == 0 &&
              record.short_window_access_count == 0,
          "retired access windows must remain zero in schema v2");
  require((record.flags & HP_TRACE_FLAG_COLD_START_FALLBACK) != 0,
          "trace flags must preserve cold-start fallback");
}

void test_writer_rotation()
{
  HpTraceWriter writer(8, 2);
  require(writer.start(
              "/tmp", 7, "phase-a", "deadbeef",
              hp_trace_config_hash(), NUM_FEATURES),
          "trace writer must start");
  const std::string first_path = writer.status().path;

  HpTraceRecord first{};
  first.io_sequence = 11;
  first.outcome = static_cast<uint8_t>(HpTraceOutcome::evaluated);
  require(writer.try_submit(first), "trace writer must accept a record");

  require(writer.start(
              "/tmp", 7, "phase-b", "deadbeef",
              hp_trace_config_hash(), NUM_FEATURES),
          "trace writer must rotate");
  const std::string second_path = writer.status().path;
  require(first_path != second_path, "rotated sessions need unique paths");

  HpTraceRecord second{};
  second.io_sequence = 12;
  second.outcome = static_cast<uint8_t>(HpTraceOutcome::prediction_error);
  require(writer.try_submit(second), "rotated writer must accept a record");
  writer.stop();

  auto read_sequence = [](const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    HpTraceFileHeader header{};
    HpTraceRecord record{};
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    input.read(reinterpret_cast<char *>(&record), sizeof(record));
    require(input.good(), "trace session must contain one record");
    require(header.schema_version == HP_TRACE_SCHEMA_VERSION,
            "trace schema version must match");
    return record.io_sequence;
  };
  require(read_sequence(first_path) == 11,
          "first trace session must be drained before rotation");
  require(read_sequence(second_path) == 12,
          "second trace session must be drained on stop");
  std::remove(first_path.c_str());
  std::remove(second_path.c_str());
}

} // namespace

int main()
{
  test_evaluated_sample_mapping();
  test_writer_rotation();
  std::puts("PASS: hp trace probe");
  return 0;
}
