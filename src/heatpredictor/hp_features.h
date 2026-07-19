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

inline const std::vector<double>& hp_to_features(const PredictionSample& item) {
    thread_local std::vector<double> features(NUM_FEATURES);
    const double threshold = std::max(item.heat_label_threshold_at_prediction, 1.0);
    const double threshold_log2p1 = hp_log2p1(threshold);
    const double heat_after_current_access = hp_log2p1(item.heat_after_current_access);

    size_t next = 0;
    features[next++] = heat_after_current_access - threshold_log2p1;
    features[next++] = hp_previous_access_interval_encoded(
        item.tracked_access_count,
        item.time_since_previous_access_ns);
    features[next++] = heat_after_current_access;
    ceph_assert(next == features.size());
    return features;
}

#endif
