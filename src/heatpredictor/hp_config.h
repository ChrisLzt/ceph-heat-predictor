#ifndef CEPH_HEATPREDICTOR_HP_CONFIG_H
#define CEPH_HEATPREDICTOR_HP_CONFIG_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/debug.h"

static constexpr size_t NUM_FEATURES = 6;

// Adaptive Random Forest model.
static constexpr int HP_ARF_N_MODELS = 25;
static constexpr int HP_ARF_MAX_FEATURES = NUM_FEATURES;
static constexpr int HP_ARF_SEED = 591422;
static constexpr int HP_ARF_GRACE_PERIOD = 100;
static constexpr int HP_ARF_LAMBDA = 4;
static constexpr double HP_ARF_DELTA = 0.001;
static constexpr double HP_ARF_TAU = 0.05;
static constexpr double HP_ARF_MAX_SHARE_TO_SPLIT = 0.99;
static constexpr double HP_ARF_MIN_BRANCH_FRACTION = 0.01;

// Prediction and training policy.
static constexpr double HP_HOT_PREDICT_THRESHOLD = 0.50;
static constexpr uint64_t HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;

// Evaluation and retained object state.
static constexpr uint64_t HP_FUTURE_LABEL_WINDOW_NS =
    10ULL * 1000 * 1000 * 1000;
static constexpr size_t HP_PENDING_EVALUATION_CAPACITY = 1000000;
static constexpr uint64_t HP_SHORT_ACCESS_WINDOW_NS =
    5ULL * 1000 * 1000 * 1000;
static constexpr size_t HP_LRU_CAPACITY = 1000000;
static constexpr size_t HP_HEAT_LABEL_THRESHOLD_OBJECT_CAPACITY = 1000000;

// Heat model.
static constexpr double HP_HEAT_INCREMENT = 100.0;
static constexpr double HP_HEAT_RETAINED_AFTER_DECAY_HORIZON = 1.0 / 10.0;
static constexpr uint64_t HP_HEAT_DECAY_HORIZON_NS =
    HP_FUTURE_LABEL_WINDOW_NS;

// Reporting windows.
static constexpr size_t HP_REPORT_SAMPLE_WINDOW_CAPACITY = 400000;

// Dynamic heat-threshold policies shared by total-heat and added-heat profiles.
static constexpr size_t HP_OTSU_MIN_VOTES = 32;
static constexpr double HP_OTSU_NEAR_OPTIMAL_RATIO = 0.99;
static constexpr double HP_OTSU_SHARPNESS_FULL_AMBIGUOUS_RATIO = 0.20;
static constexpr double HP_OTSU_SEPARATION_CONFIDENCE_WEIGHT = 0.80;
static constexpr double HP_OTSU_SHARPNESS_CONFIDENCE_WEIGHT = 0.20;
static constexpr double HP_OTSU_FIXED_EMA_ALPHA = 0.10;
static constexpr uint64_t HP_OTSU_EMA_REFERENCE_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr uint64_t HP_OTSU_RECOMPUTE_MAX_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr size_t HP_OTSU_EAGER_OBJECTS = 0;
static constexpr size_t HP_OTSU_UPDATE_INTERVAL = 100;

// Store one score-normalized current-total-heat vote per object.
static constexpr double HP_OTSU_TOTAL_HEAT_MIN = 10.0;
static constexpr size_t HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT = 800;
static constexpr double HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH = 0.01;
static constexpr double HP_OTSU_TOTAL_HEAT_MAX = 29809.579870417285;
static constexpr uint64_t HP_THRESHOLD_METHOD_INITIALIZING = 0;
static constexpr uint64_t HP_THRESHOLD_METHOD_TRACKING = 1;
static constexpr uint64_t HP_THRESHOLD_METHOD_HOLDING = 2;

static_assert(HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT >= 2,
              "score Otsu histogram needs at least two bins");
static_assert(HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH > 0.0,
              "score Otsu histogram bin width must be positive");

inline double hp_heat_decay_log_factor_per_ns(uint64_t horizon_ns) {
    ceph_assert(horizon_ns > 0);
    return std::log(HP_HEAT_RETAINED_AFTER_DECAY_HORIZON) /
        static_cast<double>(horizon_ns);
}

#endif
