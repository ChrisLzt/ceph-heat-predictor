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

inline double hp_access_rate_change_log2p1(
        uint64_t short_window_access_count,
        uint64_t long_window_access_count) {
    constexpr double nanoseconds_per_second = 1000000000.0;
    const double short_window_seconds =
        static_cast<double>(HP_SHORT_ACCESS_WINDOW_NS) /
        nanoseconds_per_second;
    const double future_label_window_seconds =
        static_cast<double>(HP_FUTURE_LABEL_WINDOW_NS) /
        nanoseconds_per_second;
    const double short_rate =
        static_cast<double>(short_window_access_count) /
        short_window_seconds;
    const double long_rate =
        static_cast<double>(long_window_access_count) /
        future_label_window_seconds;
    return hp_log2p1(short_rate) - hp_log2p1(long_rate);
}

inline const std::vector<double>& hp_to_features(const PredictionSample& item) {
    thread_local std::vector<double> features(NUM_FEATURES);
    const double threshold = std::max(item.heat_label_threshold_at_prediction, 1.0);
    const double heat_after_current_access = hp_log2p1(item.heat_after_current_access);
    const double long_window_access_count = hp_log2p1(
        static_cast<double>(
            item.long_window_access_count));

    size_t next = 0;
    features[next++] = heat_after_current_access - hp_log2p1(threshold);
    features[next++] = hp_log2p1(hp_nanoseconds_to_seconds(
        item.time_since_previous_access_ns));
    features[next++] = heat_after_current_access;
    features[next++] = long_window_access_count;
    features[next++] = hp_log2p1(
        item.heat_after_current_access /
        (HP_HEAT_INCREMENT *
         static_cast<double>(
             item.long_window_access_count + 1)));
#if HP_ENABLE_ACCESS_RATE_CHANGE
    features[next++] = hp_access_rate_change_log2p1(
        item.short_window_access_count,
        item.long_window_access_count);
#endif
#if HP_ENABLE_HEAT_PERCENTILE
    features[next++] = item.heat_percentile;
#endif
    ceph_assert(next == features.size());
    return features;
}

#endif
