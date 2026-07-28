#ifndef CEPH_TEST_SH_HP_RECENT_THRESHOLD_REPLAY_H
#define CEPH_TEST_SH_HP_RECENT_THRESHOLD_REPLAY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hp_trace_replay.h"

struct HpRecentThresholdReplayOptions {
  uint64_t window_ns = HP_FUTURE_LABEL_WINDOW_NS;
  size_t minimum_positive_objects =
      HP_FUTURE_ACCESS_OTSU_MIN_POSITIVE_OBJECTS;
  size_t update_interval = HP_FUTURE_ACCESS_OTSU_UPDATE_INTERVAL;
  uint64_t recompute_max_interval_ns =
      HP_FUTURE_ACCESS_OTSU_RECOMPUTE_MAX_INTERVAL_NS;
};

struct HpRecentThresholdReplayResult {
  std::vector<HpReplayRecordOverride> overrides;
  uint64_t past_count_mismatch_count = 0;
  uint64_t threshold_change_count = 0;
  uint64_t label_change_count = 0;
  uint64_t baseline_hot_count = 0;
  uint64_t recent_hot_count = 0;
};

namespace hp_recent_threshold_detail {

struct AccessEvent {
  uint64_t timestamp_ns;
  uint64_t object_key_hash;
};

inline uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
  return lhs > std::numeric_limits<uint64_t>::max() - rhs
      ? std::numeric_limits<uint64_t>::max()
      : lhs + rhs;
}

}  // namespace hp_recent_threshold_detail

inline HpRecentThresholdReplayResult make_hp_recent_threshold_overrides(
    const HpReplayTrace& trace,
    const HpRecentThresholdReplayOptions& options = {}) {
  if (options.window_ns == 0 ||
      options.recompute_max_interval_ns == 0) {
    throw std::invalid_argument(
        "recent threshold replay intervals must be positive");
  }

  std::vector<size_t> prediction_order(trace.records.size());
  for (size_t index = 0; index < prediction_order.size(); ++index) {
    prediction_order[index] = index;
  }
  std::sort(
      prediction_order.begin(), prediction_order.end(),
      [&trace](size_t lhs, size_t rhs) {
        const auto& left = trace.records[lhs];
        const auto& right = trace.records[rhs];
        if (left.prediction_time_ns != right.prediction_time_ns) {
          return left.prediction_time_ns < right.prediction_time_ns;
        }
        return left.io_sequence < right.io_sequence;
      });

  HpFutureAccessThreshold threshold(
      options.minimum_positive_objects,
      options.update_interval,
      options.recompute_max_interval_ns);
  std::deque<hp_recent_threshold_detail::AccessEvent> access_order;
  std::unordered_map<uint64_t, uint64_t> recent_counts;
  recent_counts.reserve(65536);

  HpRecentThresholdReplayResult result;
  result.overrides.resize(trace.records.size());
  uint64_t previous_threshold = threshold.current_threshold();
  bool has_previous_threshold = false;

  for (const size_t record_index : prediction_order) {
    const auto& record = trace.records[record_index];
    const uint64_t now_ns = record.prediction_time_ns;
    while (!access_order.empty() &&
           hp_recent_threshold_detail::saturating_add(
               access_order.front().timestamp_ns, options.window_ns) <=
               now_ns) {
      const uint64_t object_key_hash =
          access_order.front().object_key_hash;
      access_order.pop_front();
      auto count = recent_counts.find(object_key_hash);
      if (count == recent_counts.end() || count->second == 0) {
        throw std::runtime_error(
            "recent threshold replay count underflow");
      }
      const uint64_t old_count = count->second;
      --count->second;
      threshold.update_object_count(
          old_count, count->second, now_ns);
      if (count->second == 0) {
        recent_counts.erase(count);
      }
    }
    threshold.maintain(now_ns);

    const auto current_count = recent_counts.find(
        record.object_key_hash);
    const uint64_t reconstructed_past_count =
        current_count == recent_counts.end() ? 0 : current_count->second;
    if (reconstructed_past_count != record.past_window_access_count) {
      ++result.past_count_mismatch_count;
    }

    const uint64_t current_threshold = threshold.current_threshold();
    if (has_previous_threshold &&
        current_threshold != previous_threshold) {
      ++result.threshold_change_count;
    }
    previous_threshold = current_threshold;
    has_previous_threshold = true;

    const int8_t actual_label = static_cast<int8_t>(
        record.future_window_access_count >=
            record.future_window_access_threshold);
    result.overrides[record_index] = HpReplayRecordOverride{
        record.io_sequence,
        std::log2(
            1.0 + static_cast<double>(record.past_window_access_count)) -
            std::log2(1.0 + static_cast<double>(current_threshold)),
        actual_label,
        current_threshold,
    };
    result.baseline_hot_count += record.actual_label == 1 ? 1 : 0;
    result.recent_hot_count += actual_label == 1 ? 1 : 0;
    result.label_change_count +=
        actual_label != record.actual_label ? 1 : 0;

    access_order.push_back(
        hp_recent_threshold_detail::AccessEvent{
            now_ns, record.object_key_hash});
    uint64_t& object_count =
        recent_counts[record.object_key_hash];
    const uint64_t old_count = object_count;
    ++object_count;
    threshold.update_object_count(
        old_count, object_count, now_ns);
  }
  return result;
}

inline void write_hp_recent_threshold_overrides(
    std::ostream& output,
    const HpRecentThresholdReplayResult& result) {
  output
      << "io_sequence\tfeature_0\tactual_label\t"
         "future_access_threshold_at_prediction\n";
  for (const auto& record : result.overrides) {
    output << record.io_sequence << '\t'
           << record.feature_0 << '\t'
           << static_cast<int>(record.actual_label) << '\t'
           << record.future_access_threshold_at_prediction << '\n';
  }
}

#endif
