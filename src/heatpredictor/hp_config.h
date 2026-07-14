#ifndef CEPH_HEATPREDICTOR_HP_CONFIG_H
#define CEPH_HEATPREDICTOR_HP_CONFIG_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/debug.h"

#ifndef HP_ENABLE_ACCESS_RATE_CHANGE
#define HP_ENABLE_ACCESS_RATE_CHANGE 1
#endif

#ifndef HP_ENABLE_HEAT_PERCENTILE
#define HP_ENABLE_HEAT_PERCENTILE 0
#endif

#ifndef HP_ENABLE_PREDICTION_CALIBRATION
#define HP_ENABLE_PREDICTION_CALIBRATION 0
#endif

#define HP_PREDICTION_RANGE_C0 0
#define HP_PREDICTION_RANGE_CW 1
#ifndef HP_PREDICTION_RANGE_PROFILE
#define HP_PREDICTION_RANGE_PROFILE HP_PREDICTION_RANGE_C0
#endif

#define HP_OTSU_PROFILE_LEGACY 0
#define HP_OTSU_PROFILE_FIXED_EMA 1
#define HP_OTSU_PROFILE_CONFIDENCE 2
#ifndef HP_OTSU_PROFILE
#define HP_OTSU_PROFILE HP_OTSU_PROFILE_FIXED_EMA
#endif

#define HP_BASE_FEATURE_COUNT 5
#define NUM_FEATURES \
    (HP_BASE_FEATURE_COUNT + HP_ENABLE_ACCESS_RATE_CHANGE + \
     HP_ENABLE_HEAT_PERCENTILE)

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
#if HP_PREDICTION_RANGE_PROFILE == HP_PREDICTION_RANGE_C0
static constexpr double HP_HOT_PREDICT_THRESHOLD_MIN = 0.40;
static constexpr double HP_HOT_PREDICT_THRESHOLD_MAX = 0.60;
#elif HP_PREDICTION_RANGE_PROFILE == HP_PREDICTION_RANGE_CW
static constexpr double HP_HOT_PREDICT_THRESHOLD_MIN = 0.20;
static constexpr double HP_HOT_PREDICT_THRESHOLD_MAX = 0.80;
#else
#error "invalid HP_PREDICTION_RANGE_PROFILE"
#endif
static constexpr double HP_HOT_PREDICT_THRESHOLD_EMA_ALPHA = 0.10;
static constexpr size_t HP_PREDICT_CALIBRATION_WINDOW = 10000;
static constexpr size_t HP_PREDICT_CALIBRATION_UPDATE_INTERVAL = 500;
static constexpr size_t HP_PREDICT_CALIBRATION_MIN_SAMPLES = 1000;
static constexpr size_t HP_PREDICT_PROBABILITY_BIN_COUNT = 1001;
static constexpr double HP_HOT_CLASS_WEIGHT = 1.0;
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

// Dynamic hot threshold over each object's latest completed future-added heat.
static constexpr size_t HP_OTSU_MIN_OBJECTS = 32;
static constexpr double HP_OTSU_NEAR_OPTIMAL_RATIO = 0.99;
static constexpr double HP_OTSU_SHARPNESS_FULL_AMBIGUOUS_RATIO = 0.20;
static constexpr double HP_OTSU_SEPARATION_CONFIDENCE_WEIGHT = 0.80;
static constexpr double HP_OTSU_SHARPNESS_CONFIDENCE_WEIGHT = 0.20;
static constexpr double HP_OTSU_FIXED_EMA_ALPHA = 0.10;
static constexpr uint64_t HP_OTSU_EMA_REFERENCE_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr uint64_t HP_OTSU_RECOMPUTE_MAX_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr double HP_OTSU_CONFIDENCE_MAX_UPDATE_ALPHA = 0.50;
static constexpr double HP_LEGACY_HOT_QUANTILE = 0.85;
static constexpr double HP_LEGACY_OTSU_EMA_ALPHA = 0.10;
static constexpr double HP_LEGACY_OTSU_MIN_SEPARATION = 0.60;
static constexpr size_t HP_OTSU_EAGER_OBJECTS = 0;
static constexpr size_t HP_OTSU_UPDATE_INTERVAL = 100;
static constexpr size_t HP_OTSU_HISTOGRAM_BIN_COUNT = 10000;
static constexpr double HP_OTSU_LOG1P_HEAT_BIN_WIDTH = 0.01;
static constexpr uint64_t HP_OTSU_HISTORY_SLOT_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr size_t HP_OTSU_HISTORY_SLOT_COUNT = 60;
static constexpr uint64_t HP_OTSU_HISTORY_WINDOW_NS =
    HP_OTSU_HISTORY_SLOT_NS * HP_OTSU_HISTORY_SLOT_COUNT;

// Retained for the object-total-heat control experiment.
static constexpr double HP_OTSU_HEAT_MIN = 1.0;
static constexpr double HP_OTSU_HEAT_MAX = 3000.0;
static constexpr double HP_OTSU_LOG_HEAT_BIN_WIDTH = 0.05;
static constexpr uint64_t HP_THRESHOLD_METHOD_INITIALIZING = 0;
static constexpr uint64_t HP_THRESHOLD_METHOD_TRACKING = 1;
static constexpr uint64_t HP_THRESHOLD_METHOD_HOLDING = 2;

static_assert(HP_OTSU_PROFILE >= HP_OTSU_PROFILE_LEGACY &&
              HP_OTSU_PROFILE <= HP_OTSU_PROFILE_CONFIDENCE,
              "invalid HP_OTSU_PROFILE");
static_assert(HP_OTSU_HISTOGRAM_BIN_COUNT >= 2,
              "Otsu histogram needs at least two bins");
static_assert(HP_OTSU_LOG1P_HEAT_BIN_WIDTH > 0.0,
              "Otsu histogram bin width must be positive");
static_assert(HP_OTSU_HISTORY_SLOT_COUNT > 0 &&
              HP_OTSU_HISTORY_SLOT_NS > 0,
              "Otsu history needs positive time slots");

inline double hp_heat_decay_log_factor_per_ns(uint64_t horizon_ns) {
    ceph_assert(horizon_ns > 0);
    return std::log(HP_HEAT_RETAINED_AFTER_DECAY_HORIZON) /
        static_cast<double>(horizon_ns);
}

#endif
