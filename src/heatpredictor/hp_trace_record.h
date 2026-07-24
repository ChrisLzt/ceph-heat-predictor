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

static constexpr uint64_t HP_TRACE_FEATURE_SCHEMA_VERSION = 4;

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
    const std::array<uint64_t, 9> integer_values = {
        HP_TRACE_FEATURE_SCHEMA_VERSION,
        NUM_FEATURES,
        HP_ARF_N_MODELS,
        HP_ARF_MAX_FEATURES,
        HP_ARF_SEED,
        HP_FUTURE_LABEL_WINDOW_NS,
        HP_PENDING_EVALUATION_CAPACITY,
        HP_LRU_CAPACITY,
        HP_HEAT_LABEL_THRESHOLD_OBJECT_CAPACITY,
    };
    const std::array<double, 7> floating_values = {
        HP_HOT_PREDICT_THRESHOLD,
        HP_HEAT_INCREMENT,
        HP_HEAT_RETAINED_AFTER_DECAY_HORIZON,
        HP_OTSU_EMA_ALPHA,
        HP_OTSU_TOTAL_HEAT_MIN,
        HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH,
        HP_OTSU_TOTAL_HEAT_MAX,
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
    record.heat_label_threshold_at_prediction =
        item.heat_label_threshold_at_prediction;
    record.predicted_hot_probability = item.predicted_hot_probability;
    record.hot_predict_threshold = HP_HOT_PREDICT_THRESHOLD;
    record.label_heat = evaluated.label_heat;
    record.label_heat_threshold = evaluated.label_heat_threshold;
    record.tracked_access_count = item.tracked_access_count;
    record.time_since_previous_access_ns =
        item.time_since_previous_access_ns;
    record.future_window_access_count =
        evaluated.future_window_access_count;
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
    record.heat_label_threshold_at_prediction =
        item.heat_label_threshold_at_prediction;
    record.predicted_hot_probability = item.predicted_hot_probability;
    record.hot_predict_threshold = HP_HOT_PREDICT_THRESHOLD;
    record.tracked_access_count = item.tracked_access_count;
    record.time_since_previous_access_ns =
        item.time_since_previous_access_ns;
    record.outcome = static_cast<uint8_t>(outcome);
    record.flags = flags;
    record.predicted_label = static_cast<int8_t>(item.predicted_label);
    record.actual_label = -1;
    return record;
}

#endif
