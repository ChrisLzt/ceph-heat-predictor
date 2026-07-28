#ifndef CEPH_HEATPREDICTOR_HP_TRACE_RECORD_H
#define CEPH_HEATPREDICTOR_HP_TRACE_RECORD_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hp_config.h"
#include "hp_features.h"
#include "hp_trace.h"
#include "hp_types.h"

static constexpr uint64_t HP_TRACE_FEATURE_SCHEMA_VERSION = 7;

template <typename T>
inline void hp_trace_hash_value(uint64_t& hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
}

inline uint64_t hp_trace_config_hash()
{
    uint64_t hash = 1469598103934665603ULL;
    const std::array<uint64_t, 12> integer_values = {
        HP_TRACE_FEATURE_SCHEMA_VERSION,
        NUM_FEATURES,
        HP_ARF_N_MODELS,
        HP_ARF_MAX_FEATURES,
        HP_ARF_SEED,
        HP_FUTURE_LABEL_WINDOW_NS,
        HP_SHORT_ACCESS_WINDOW_NS,
        HP_PENDING_EVALUATION_CAPACITY,
        HP_LRU_CAPACITY,
        HP_FUTURE_ACCESS_OTSU_UPDATE_INTERVAL,
        HP_FUTURE_ACCESS_OTSU_RECOMPUTE_MAX_INTERVAL_NS,
        HP_FUTURE_ACCESS_OTSU_BIN_COUNT,
    };
    const std::array<double, 5> floating_values = {
        HP_HOT_PREDICT_THRESHOLD,
        HP_HEAT_INCREMENT,
        HP_HEAT_RETAINED_AFTER_DECAY_HORIZON,
        HP_FUTURE_ACCESS_OTSU_SCORE_MIN,
        HP_FUTURE_ACCESS_OTSU_BIN_WIDTH,
    };
    for (const auto value : integer_values) {
        hp_trace_hash_value(hash, value);
    }
    for (const auto value : floating_values) {
        hp_trace_hash_value(hash, value);
    }
    return hash;
}

inline void hp_fill_trace_features(
        HpTraceRecord& record,
        const PredictionSample& item)
{
    static_assert(NUM_FEATURES <= HP_TRACE_MAX_FEATURES);
    const auto& features = hp_to_features(item);
    std::copy_n(features.begin(), features.size(), record.features);
}

inline HpTraceRecord hp_trace_record_for_evaluated(
        const EvaluatedSample& evaluated)
{
    HpTraceRecord record{};
    const PredictionSample& item = evaluated.item;
    record.io_sequence = item.io_sequence;
    record.object_key_hash = item.object_key_hash;
    record.prediction_time_ns = evaluated.prediction_time_ns;
    record.label_deadline_ns = evaluated.label_deadline_ns;
    record.label_completion_time_ns = evaluated.label_completion_time_ns;
    hp_fill_trace_features(record, item);
    record.heat_after_current_access = item.heat_after_current_access;
    record.predicted_hot_probability = item.predicted_hot_probability;
    record.hot_predict_threshold = HP_HOT_PREDICT_THRESHOLD;
    record.future_access_threshold_at_prediction =
        item.future_access_threshold_at_prediction;
    record.past_window_access_count = item.past_window_access_count;
    record.tracked_access_count_after_current_access =
        item.tracked_access_count_after_current_access;
    record.time_since_previous_access_ns =
        item.time_since_previous_access_ns;
    record.future_window_access_count =
        evaluated.future_window_access_count;
    record.future_window_access_threshold =
        evaluated.future_window_access_threshold;
    record.outcome = static_cast<uint8_t>(HpTraceOutcome::evaluated);
    record.flags = evaluated.cold_start_fallback
        ? HP_TRACE_FLAG_COLD_START_FALLBACK
        : HP_TRACE_FLAG_NONE;
    record.predicted_label = static_cast<int8_t>(item.predicted_label);
    record.actual_label = static_cast<int8_t>(evaluated.label);
    return record;
}

inline HpTraceRecord hp_trace_record_for_incomplete(
        const PredictionSample& item,
        uint64_t prediction_time_ns,
        HpTraceOutcome outcome,
        uint8_t flags)
{
    HpTraceRecord record{};
    record.io_sequence = item.io_sequence;
    record.object_key_hash = item.object_key_hash;
    record.prediction_time_ns = prediction_time_ns;
    hp_fill_trace_features(record, item);
    record.heat_after_current_access = item.heat_after_current_access;
    record.predicted_hot_probability = item.predicted_hot_probability;
    record.hot_predict_threshold = HP_HOT_PREDICT_THRESHOLD;
    record.future_access_threshold_at_prediction =
        item.future_access_threshold_at_prediction;
    record.past_window_access_count = item.past_window_access_count;
    record.tracked_access_count_after_current_access =
        item.tracked_access_count_after_current_access;
    record.time_since_previous_access_ns =
        item.time_since_previous_access_ns;
    record.outcome = static_cast<uint8_t>(outcome);
    record.flags = flags;
    record.predicted_label = static_cast<int8_t>(item.predicted_label);
    record.actual_label = -1;
    return record;
}

#endif
