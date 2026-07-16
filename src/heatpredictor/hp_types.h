#ifndef CEPH_HEATPREDICTOR_HP_TYPES_H
#define CEPH_HEATPREDICTOR_HP_TYPES_H

#include <cstdint>
#include <list>

struct PredictionSample {
    uint64_t io_sequence;
    uint64_t object_key_hash;
    double heat_after_current_access;
    double heat_label_threshold_at_prediction;
    uint64_t tracked_access_count;
    uint64_t time_since_previous_access_ns;
    uint64_t long_window_access_count;
    uint64_t short_window_access_count;
    double predicted_hot_probability;
    int predicted_label;
};

struct ObjectHeatState {
    double heat;
    uint64_t last_access_time_ns;
    uint64_t tracked_access_count;
    uint64_t pending_evaluation_count;
    uint64_t short_window_access_count;
    uint64_t long_window_access_count;
    std::list<uint64_t>::iterator lru_position;
};

struct EvaluatedSample {
    PredictionSample item;
    int label;
    uint64_t future_window_access_count;
    double future_window_added_heat;
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
    PredictionSample item;
    int label;
};

struct HeatPredictorStats {
    bool enabled;
    uint64_t io_count;
    uint64_t labeled_io_total;
    uint64_t pending_io_count;
    uint64_t awaiting_prediction_count;
    uint64_t eval_drop_count;
    uint64_t heat_state_count;
    uint64_t lru_count;
    uint64_t otsu_histogram_bin_count;
    uint64_t otsu_histogram_vote_count;
    uint64_t true_positive;
    uint64_t false_positive;
    uint64_t true_negative;
    uint64_t false_negative;
    uint64_t hot_labeled_sample_future_access_count_sum;
    uint64_t cold_labeled_sample_future_access_count_sum;
    // Exported as the hot/cold labeled-sample average future added heat.
    double hot_labeled_sample_future_added_heat_sum;
    double cold_labeled_sample_future_added_heat_sum;
    double hot_labeled_sample_predicted_hot_probability_sum;
    double cold_labeled_sample_predicted_hot_probability_sum;
    HpDistributionSummary hot_labeled_sample_future_access_count;
    HpDistributionSummary cold_labeled_sample_future_access_count;
    HpDistributionSummary hot_labeled_sample_future_added_heat;
    HpDistributionSummary cold_labeled_sample_future_added_heat;
    double heat_label_threshold;
    double otsu_candidate_threshold;
    double otsu_separation;
    double otsu_confidence;
    double otsu_sharpness_confidence;
    uint64_t hot_threshold_method;
    double hot_predict_threshold;
};

#endif
