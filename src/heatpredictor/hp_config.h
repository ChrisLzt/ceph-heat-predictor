#ifndef CEPH_HEATPREDICTOR_HP_CONFIG_H
#define CEPH_HEATPREDICTOR_HP_CONFIG_H

#include <cmath>
#include <cstddef>

#include "common/debug.h"

#define NUM_FEATURES 7

static constexpr double HP_HOT_QUANTILE = 0.85;
static constexpr double HP_HOT_CLASS_WEIGHT = 4.0;
static constexpr double HP_HOT_PREDICT_THRESHOLD = 0.50;
static constexpr size_t HP_EVALUATION_WINDOW = 10000;
static constexpr size_t HP_LABEL_THRESHOLD_WINDOW_CAPACITY = 1000000;
static constexpr double HP_HEAT_INCREMENT = 100.0;
static constexpr size_t HP_LRU_CAPACITY = 100000;
static constexpr double HP_HEAT_RETAIN_RATIO = 1.0 / 10.0;
static constexpr size_t HP_REPORT_STATS_WINDOW_CAPACITY = 400000;

inline double hp_heat_decay_alpha(size_t evaluation_window) {
    ceph_assert(evaluation_window > 0);
    return std::log(HP_HEAT_RETAIN_RATIO) /
        static_cast<double>(evaluation_window);
}

#endif
