#ifndef CEPH_HEATPREDICTOR_HP_TYPES_H
#define CEPH_HEATPREDICTOR_HP_TYPES_H

#include <cstdint>
#include <list>

struct TraceItem {
    uint64_t index;
    uint64_t operation;
    uint64_t size;
    uint64_t key;
    double current_heat;
    double hot_threshold;
    uint64_t access_count;
    uint64_t last_access_distance;
    uint64_t object_age;
    double pred_hot_proba;
    int pred;
};

struct HeatState {
    double heat;
    uint64_t last_access;
    uint64_t first_access;
    uint64_t access_count;
    uint64_t pending_count;
    std::list<uint64_t>::iterator lru_position;
};

struct EvaluatedItem {
    TraceItem item;
    int label;
    double training_weight;
    uint64_t future_access_count;
    double future_heat;
};

struct HpDistributionSummary {
    uint64_t count = 0;
    double max = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

struct TrainingSample {
    TraceItem item;
    int label;
    double weight;
};

struct HeatPredictorStats {
    uint64_t io_count;
    uint64_t labeled_io_total;
    uint64_t pending_io_count;
    uint64_t heat_state_count;
    uint64_t lru_count;
    uint64_t true_positive;
    uint64_t false_positive;
    uint64_t true_negative;
    uint64_t false_negative;
    uint64_t actual_hot_object_access_count_sum;
    uint64_t actual_cold_object_access_count_sum;
    // Exported as hp_actual_hot_object_avg_heat and
    // hp_actual_cold_object_avg_heat after averaging.
    double actual_hot_object_heat_sum;
    double actual_cold_object_heat_sum;
    double actual_hot_pred_hot_proba_sum;
    double actual_cold_pred_hot_proba_sum;
    HpDistributionSummary actual_hot_future_access;
    HpDistributionSummary actual_cold_future_access;
    HpDistributionSummary actual_hot_future_heat;
    HpDistributionSummary actual_cold_future_heat;
    double hot_threshold;
    double hot_quantile_threshold;
    uint64_t hot_threshold_method;
    double otsu_separation;
    double hot_predict_threshold;
    double pred_actual_hot_ratio;
    double dynamic_hot_class_weight;
};

#endif
