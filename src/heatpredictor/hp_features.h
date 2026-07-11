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

inline double hp_access_acceleration(
        uint64_t recent_window_access_count,
        uint64_t past_window_access_count) {
    const double recent_rate =
        static_cast<double>(recent_window_access_count + 1) /
        static_cast<double>(HP_ACCESS_ACCELERATION_WINDOW);
    const double past_rate =
        static_cast<double>(past_window_access_count + 1) /
        static_cast<double>(HP_EVALUATION_WINDOW);
    return std::log2(recent_rate / past_rate);
}

inline const std::vector<double>& hp_to_features(const TraceItem& item) {
    thread_local std::vector<double> features(NUM_FEATURES);
    const double threshold = std::max(item.hot_threshold, 1.0);
    const double current_heat = hp_log2p1(item.current_heat);
    const double past_window_access_count = hp_log2p1(
        static_cast<double>(item.past_window_access_count));

    size_t next = 0;
    features[next++] = current_heat - hp_log2p1(threshold);
    features[next++] = hp_log2p1(
        static_cast<double>(item.last_access_distance));
    features[next++] = current_heat;
    features[next++] = past_window_access_count;
    features[next++] = hp_log2p1(
        item.current_heat /
        (HP_HEAT_INCREMENT *
         static_cast<double>(item.past_window_access_count + 1)));
#if HP_ENABLE_ACCESS_ACCELERATION
    features[next++] = hp_access_acceleration(
        item.recent_window_access_count,
        item.past_window_access_count);
#endif
#if HP_ENABLE_HEAT_PERCENTILE
    features[next++] = item.heat_percentile;
#endif
    ceph_assert(next == features.size());
    return features;
}

#endif
