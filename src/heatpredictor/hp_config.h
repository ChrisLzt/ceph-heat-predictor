#ifndef CEPH_HEATPREDICTOR_HP_CONFIG_H
#define CEPH_HEATPREDICTOR_HP_CONFIG_H

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "common/debug.h"

#ifndef HP_ENABLE_ACCESS_RATE_CHANGE
#define HP_ENABLE_ACCESS_RATE_CHANGE 0
#endif

#ifndef HP_ENABLE_STANDARD_SCALER
#define HP_ENABLE_STANDARD_SCALER 1
#endif

#define HP_HEAT_MARGIN_CURRENT 0
#define HP_HEAT_MARGIN_PROJECTED 1
#define HP_HEAT_MARGIN_BOTH 2
#ifndef HP_HEAT_MARGIN_PROFILE
#define HP_HEAT_MARGIN_PROFILE HP_HEAT_MARGIN_CURRENT
#endif

#define HP_BASE_FEATURE_COUNT \
    (3 + (HP_HEAT_MARGIN_PROFILE == HP_HEAT_MARGIN_BOTH))
#define NUM_FEATURES \
    (HP_BASE_FEATURE_COUNT + HP_ENABLE_ACCESS_RATE_CHANGE)

// Adaptive Random Forest model.
#ifndef HP_ARF_GRACE_PERIOD_VALUE
#define HP_ARF_GRACE_PERIOD_VALUE 100
#endif
#ifndef HP_ARF_WARNING_DELTA_PERMILLE_VALUE
#define HP_ARF_WARNING_DELTA_PERMILLE_VALUE 10
#endif
#ifndef HP_ARF_DRIFT_DELTA_PERMILLE_VALUE
#define HP_ARF_DRIFT_DELTA_PERMILLE_VALUE 1
#endif
#ifndef HP_ARF_FAST_MODEL_COUNT_VALUE
#define HP_ARF_FAST_MODEL_COUNT_VALUE 0
#endif
#ifndef HP_ARF_FAST_MODEL_LIFETIME_SAMPLES_VALUE
#define HP_ARF_FAST_MODEL_LIFETIME_SAMPLES_VALUE 0
#endif

static constexpr int HP_ARF_N_MODELS = 25;
static constexpr int HP_ARF_MAX_FEATURES = NUM_FEATURES;
static constexpr int HP_ARF_SEED = 591422;
static constexpr int HP_ARF_GRACE_PERIOD = HP_ARF_GRACE_PERIOD_VALUE;
static constexpr int HP_ARF_LAMBDA = 4;
static constexpr double HP_ARF_DELTA = 0.001;
static constexpr double HP_ARF_TAU = 0.05;
static constexpr double HP_ARF_MAX_SHARE_TO_SPLIT = 0.99;
static constexpr double HP_ARF_MIN_BRANCH_FRACTION = 0.01;
static constexpr int HP_ARF_WARNING_DELTA_PERMILLE =
    HP_ARF_WARNING_DELTA_PERMILLE_VALUE;
static constexpr int HP_ARF_DRIFT_DELTA_PERMILLE =
    HP_ARF_DRIFT_DELTA_PERMILLE_VALUE;
static constexpr int HP_ARF_FAST_MODEL_COUNT =
    HP_ARF_FAST_MODEL_COUNT_VALUE;
static constexpr uint64_t HP_ARF_FAST_MODEL_LIFETIME_SAMPLES =
    HP_ARF_FAST_MODEL_LIFETIME_SAMPLES_VALUE;

static_assert(HP_ARF_GRACE_PERIOD > 0,
              "ARF grace period must be positive");
static_assert(HP_ARF_WARNING_DELTA_PERMILLE > 0 &&
              HP_ARF_WARNING_DELTA_PERMILLE < 1000,
              "ARF warning delta must be in (0, 1)");
static_assert(HP_ARF_DRIFT_DELTA_PERMILLE > 0 &&
              HP_ARF_DRIFT_DELTA_PERMILLE < 1000,
              "ARF drift delta must be in (0, 1)");
static_assert(HP_ARF_FAST_MODEL_COUNT >= 0 &&
              HP_ARF_FAST_MODEL_COUNT <= HP_ARF_N_MODELS,
              "ARF fast model count must fit in the ensemble");
static_assert(
    (HP_ARF_FAST_MODEL_COUNT == 0 &&
     HP_ARF_FAST_MODEL_LIFETIME_SAMPLES == 0) ||
    (HP_ARF_FAST_MODEL_COUNT > 0 &&
     HP_ARF_FAST_MODEL_LIFETIME_SAMPLES >=
         static_cast<uint64_t>(HP_ARF_FAST_MODEL_COUNT)),
    "ARF fast model lifetime must cover its cohort");

// Prediction and training policy.
static constexpr double HP_HOT_PREDICT_THRESHOLD = 0.50;
static constexpr uint64_t HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;

static constexpr uint64_t HP_FEATURE_SCHEMA_VERSION = 4;

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
static constexpr double HP_HEAT_RETAINED_AFTER_DECAY_HORIZON = 1.0 / 5.0;
static constexpr uint64_t HP_HEAT_DECAY_HORIZON_NS =
    HP_FUTURE_LABEL_WINDOW_NS;

// Reporting windows.
static constexpr size_t HP_REPORT_SAMPLE_WINDOW_CAPACITY = 400000;

// Object-total-heat Otsu threshold policy.
static constexpr size_t HP_OTSU_MIN_VOTES = 32;
static constexpr double HP_OTSU_EMA_ALPHA = 0.10;
static constexpr uint64_t HP_OTSU_EMA_REFERENCE_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr uint64_t HP_OTSU_RECOMPUTE_MAX_INTERVAL_NS =
    1ULL * 1000 * 1000 * 1000;
static constexpr size_t HP_OTSU_EAGER_OBJECTS = 0;
static constexpr size_t HP_OTSU_UPDATE_INTERVAL = 100;

// Store one score-normalized current-total-heat vote per object.
static constexpr double HP_OTSU_TOTAL_HEAT_MIN =
    HP_HEAT_INCREMENT * HP_HEAT_RETAINED_AFTER_DECAY_HORIZON;
static constexpr size_t HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT = 800;
static constexpr double HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH = 0.01;
static constexpr double HP_SCORE_OTSU_HEAT_RANGE_RATIO =
    2980.9579870417283;  // exp(800 * 0.01)
static constexpr double HP_OTSU_TOTAL_HEAT_MAX =
    HP_OTSU_TOTAL_HEAT_MIN * HP_SCORE_OTSU_HEAT_RANGE_RATIO;
static constexpr uint64_t HP_THRESHOLD_METHOD_INITIALIZING = 0;
static constexpr uint64_t HP_THRESHOLD_METHOD_TRACKING = 1;
static constexpr uint64_t HP_THRESHOLD_METHOD_HOLDING = 2;

static_assert(HP_ENABLE_STANDARD_SCALER == 0 ||
              HP_ENABLE_STANDARD_SCALER == 1,
              "invalid HP_ENABLE_STANDARD_SCALER");
static_assert(HP_HEAT_MARGIN_PROFILE >= HP_HEAT_MARGIN_CURRENT &&
              HP_HEAT_MARGIN_PROFILE <= HP_HEAT_MARGIN_BOTH,
              "invalid HP_HEAT_MARGIN_PROFILE");
static_assert(HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT >= 2,
              "score Otsu histogram needs at least two bins");
static_assert(HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH > 0.0,
              "score Otsu histogram bin width must be positive");
static_assert(HP_OTSU_TOTAL_HEAT_MIN > 0.0 &&
              HP_OTSU_TOTAL_HEAT_MAX > HP_OTSU_TOTAL_HEAT_MIN,
              "score Otsu heat bounds must be positive and ordered");

inline double hp_heat_decay_log_factor_per_ns(uint64_t horizon_ns) {
    ceph_assert(horizon_ns > 0);
    return std::log(HP_HEAT_RETAINED_AFTER_DECAY_HORIZON) /
        static_cast<double>(horizon_ns);
}

#endif
