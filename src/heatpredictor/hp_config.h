#ifndef CEPH_HEATPREDICTOR_HP_CONFIG_H
#define CEPH_HEATPREDICTOR_HP_CONFIG_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/debug.h"

static constexpr size_t NUM_FEATURES = 3;

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
static constexpr int HP_ARF_WARNING_DELTA_PERMILLE = 10;
static constexpr int HP_ARF_DRIFT_DELTA_PERMILLE = 1;

static_assert(HP_ARF_GRACE_PERIOD > 0,
              "ARF grace period must be positive");
static_assert(HP_ARF_WARNING_DELTA_PERMILLE > 0 &&
              HP_ARF_WARNING_DELTA_PERMILLE < 1000,
              "ARF warning delta must be in (0, 1)");
static_assert(HP_ARF_DRIFT_DELTA_PERMILLE > 0 &&
              HP_ARF_DRIFT_DELTA_PERMILLE < 1000,
              "ARF drift delta must be in (0, 1)");

// Prediction and training policy.
static constexpr double HP_HOT_PREDICT_THRESHOLD = 0.50;
static constexpr uint64_t HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;

// Evaluation and retained object state.
static constexpr uint64_t HP_FUTURE_LABEL_WINDOW_NS =
    10ULL * 1000 * 1000 * 1000;
static constexpr size_t HP_PENDING_EVALUATION_CAPACITY = 1000000;
static constexpr size_t HP_LRU_CAPACITY = 1000000;
static constexpr size_t HP_EXPIRY_MAINTENANCE_BATCH_SIZE = 1000;

// Future-access threshold policy.
static constexpr size_t HP_FUTURE_ACCESS_THRESHOLD_OBJECT_CAPACITY = 1000000;
static constexpr size_t HP_FUTURE_ACCESS_OTSU_MIN_POSITIVE_OBJECTS = 32;
static constexpr size_t HP_FUTURE_ACCESS_OTSU_UPDATE_INTERVAL = 100;
static constexpr uint64_t HP_FUTURE_ACCESS_OTSU_RECOMPUTE_MAX_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr uint64_t HP_FUTURE_ACCESS_THRESHOLD_HOLD_NS =
    10ULL * 1000 * 1000 * 1000;
static constexpr double HP_FUTURE_ACCESS_THRESHOLD_EMA_ALPHA = 0.10;
static constexpr double HP_FUTURE_ACCESS_OTSU_SCORE_MIN = 1.0;
static constexpr double HP_FUTURE_ACCESS_OTSU_BIN_WIDTH = 0.01;
static constexpr size_t HP_FUTURE_ACCESS_OTSU_BIN_COUNT = 2000;

// Heat model.
static constexpr double HP_HEAT_INCREMENT = 100.0;
static constexpr double HP_HEAT_RETAINED_AFTER_DECAY_HORIZON = 1.0 / 5.0;
static constexpr uint64_t HP_HEAT_DECAY_HORIZON_NS =
    HP_FUTURE_LABEL_WINDOW_NS;

// Reporting windows.
static constexpr size_t HP_REPORT_SAMPLE_WINDOW_CAPACITY = 400000;

static_assert(HP_FUTURE_ACCESS_OTSU_BIN_COUNT >= 2,
              "future-access Otsu histogram needs at least two bins");
static_assert(HP_FUTURE_ACCESS_OTSU_BIN_WIDTH > 0.0,
              "future-access Otsu histogram bin width must be positive");

inline double hp_heat_decay_log_factor_per_ns(uint64_t horizon_ns) {
    ceph_assert(horizon_ns > 0);
    return std::log(HP_HEAT_RETAINED_AFTER_DECAY_HORIZON) /
        static_cast<double>(horizon_ns);
}

#endif
