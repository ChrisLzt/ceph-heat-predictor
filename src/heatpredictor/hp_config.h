#ifndef CEPH_HEATPREDICTOR_HP_CONFIG_H
#define CEPH_HEATPREDICTOR_HP_CONFIG_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/debug.h"

#define NUM_FEATURES 7

// Prediction and training policy.
static constexpr double HP_HOT_PREDICT_THRESHOLD = 0.50;
static constexpr double HP_HOT_PREDICT_THRESHOLD_MIN = 0.45;
static constexpr double HP_HOT_PREDICT_THRESHOLD_MAX = 0.55;
static constexpr double HP_HOT_PREDICT_THRESHOLD_EMA_ALPHA = 0.10;
static constexpr double HP_HOT_CLASS_WEIGHT = 3.0;
static constexpr double HP_PRED_ACTUAL_HOT_RATIO_SMOOTHING = 1.0;
static constexpr double HP_PRED_ACTUAL_HOT_RATIO_MIN = 0.80;
static constexpr double HP_PRED_ACTUAL_HOT_RATIO_MAX = 1.25;

// Evaluation and retained object state.
static constexpr size_t HP_EVALUATION_WINDOW = 2000;
static constexpr size_t HP_LRU_CAPACITY = 100000;
static constexpr size_t HP_LABEL_THRESHOLD_WINDOW_CAPACITY = 1000000;

// Heat model.
static constexpr double HP_HEAT_INCREMENT = 100.0;
static constexpr double HP_HEAT_RETAIN_RATIO = 1.0 / 10.0;

// Reporting windows.
static constexpr size_t HP_REPORT_STATS_WINDOW_CAPACITY = 400000;

// Dynamic hot threshold.
static constexpr double HP_HOT_QUANTILE = 0.85;
static constexpr size_t HP_OTSU_MIN_OBJECTS = 32;
static constexpr double HP_OTSU_MIN_SEPARATION = 0.60;
static constexpr double HP_OTSU_EMA_ALPHA = 0.10;
static constexpr size_t HP_OTSU_EAGER_OBJECTS = 1000;
static constexpr size_t HP_OTSU_UPDATE_INTERVAL = 1000;
static constexpr uint64_t HP_THRESHOLD_METHOD_NONE = 0;
static constexpr uint64_t HP_THRESHOLD_METHOD_QUANTILE = 1;
static constexpr uint64_t HP_THRESHOLD_METHOD_OTSU = 2;

inline double hp_heat_decay_alpha(size_t evaluation_window) {
    ceph_assert(evaluation_window > 0);
    return std::log(HP_HEAT_RETAIN_RATIO) /
        static_cast<double>(evaluation_window);
}

#endif
