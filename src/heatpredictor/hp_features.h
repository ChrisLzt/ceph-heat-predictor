#ifndef CEPH_HEATPREDICTOR_HP_FEATURES_H
#define CEPH_HEATPREDICTOR_HP_FEATURES_H

#include <algorithm>
#include <cmath>
#include <vector>

#include "hp_config.h"
#include "hp_types.h"

inline double hp_log2p1(double value) {
    return std::log2(1.0 + value);
}

inline double hp_nanoseconds_to_seconds(uint64_t nanoseconds) {
    constexpr double nanoseconds_per_second = 1000000000.0;
    return static_cast<double>(nanoseconds) / nanoseconds_per_second;
}

inline double hp_previous_access_interval_encoded(
        uint64_t tracked_access_count,
        uint64_t time_since_previous_access_ns) {
    if (tracked_access_count <= 1) {
        return 0.0;
    }
    return 1.0 + hp_log2p1(
        hp_nanoseconds_to_seconds(time_since_previous_access_ns));
}

inline double hp_projected_count_margin(
        uint64_t short_window_access_count,
        uint64_t future_access_threshold) {
    const double short_window_seconds =
        hp_nanoseconds_to_seconds(HP_SHORT_ACCESS_WINDOW_NS);
    const double future_window_seconds =
        hp_nanoseconds_to_seconds(HP_FUTURE_LABEL_WINDOW_NS);
    ceph_assert(short_window_seconds > 0.0);
    const double projected_future_access_count =
        static_cast<double>(short_window_access_count) /
        short_window_seconds * future_window_seconds;
    return hp_log2p1(projected_future_access_count) -
        hp_log2p1(static_cast<double>(
            std::max<uint64_t>(future_access_threshold, 1)));
}

inline const std::vector<double>& hp_to_features(const PredictionSample& item) {
    thread_local std::vector<double> features(NUM_FEATURES);
    const double threshold = static_cast<double>(std::max<uint64_t>(
        item.future_access_threshold_at_prediction, 1));
    const double heat_after_current_access = hp_log2p1(item.heat_after_current_access);

    size_t next = 0;
    features[next++] =
        hp_log2p1(static_cast<double>(item.past_window_access_count)) -
        hp_log2p1(threshold);
    features[next++] = hp_previous_access_interval_encoded(
        item.tracked_access_count_after_current_access,
        item.time_since_previous_access_ns);
    features[next++] = heat_after_current_access;
    features[next++] = hp_projected_count_margin(
        item.short_window_access_count,
        item.future_access_threshold_at_prediction);
    features[next++] =
        hp_log2p1(static_cast<double>(item.short_window_access_count));
    ceph_assert(next == features.size());
    return features;
}

#endif
