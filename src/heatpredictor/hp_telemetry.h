#ifndef CEPH_HEATPREDICTOR_HP_TELEMETRY_H
#define CEPH_HEATPREDICTOR_HP_TELEMETRY_H

namespace ceph::hp_telemetry {

enum class Aggregate {
    none,
    sum,
    osd_average,
    hot_weighted,
    cold_weighted,
    otsu_weighted,
};

enum class Unit {
    count,
    scaled_x10000,
    nanoseconds,
};

struct CounterField {
    const char* name;
    Aggregate aggregate;
    const char* aggregate_name = nullptr;
    Unit unit = Unit::count;
};

struct AverageField {
    const char* name;
    Unit unit;
};

namespace field {
inline constexpr char enabled[] = "hp_enabled";
inline constexpr char io_count[] = "hp_io_count";
inline constexpr char labeled_io_total[] = "hp_labeled_io_total";
inline constexpr char pending_io_count[] = "hp_pending_io_count";
inline constexpr char awaiting_prediction_count[] =
    "hp_awaiting_prediction_count";
inline constexpr char eval_drop_count[] = "hp_eval_drop_count";
inline constexpr char heat_state_count[] = "hp_heat_state_count";
inline constexpr char lru_count[] = "hp_lru_count";
inline constexpr char protected_heat_state_count[] =
    "hp_protected_heat_state_count";
inline constexpr char heat_state_peak_count[] = "hp_heat_state_peak_count";
inline constexpr char lru_eviction_count[] = "hp_lru_eviction_count";
inline constexpr char otsu_histogram_bin_count[] =
    "hp_otsu_histogram_bin_count";
inline constexpr char otsu_histogram_vote_count[] =
    "hp_otsu_histogram_vote_count";
inline constexpr char true_positive_count[] = "hp_true_positive_count";
inline constexpr char false_positive_count[] = "hp_false_positive_count";
inline constexpr char true_negative_count[] = "hp_true_negative_count";
inline constexpr char false_negative_count[] = "hp_false_negative_count";
inline constexpr char hot_labeled_sample_avg_future_access_count[] =
    "hp_hot_labeled_sample_avg_future_access_count";
inline constexpr char cold_labeled_sample_avg_future_access_count[] =
    "hp_cold_labeled_sample_avg_future_access_count";
inline constexpr char hot_labeled_sample_future_access_count_p99[] =
    "hp_hot_labeled_sample_future_access_count_p99";
inline constexpr char hot_labeled_sample_future_access_count_p95[] =
    "hp_hot_labeled_sample_future_access_count_p95";
inline constexpr char hot_labeled_sample_future_access_count_p50[] =
    "hp_hot_labeled_sample_future_access_count_p50";
inline constexpr char cold_labeled_sample_future_access_count_p99[] =
    "hp_cold_labeled_sample_future_access_count_p99";
inline constexpr char cold_labeled_sample_future_access_count_p95[] =
    "hp_cold_labeled_sample_future_access_count_p95";
inline constexpr char cold_labeled_sample_future_access_count_p50[] =
    "hp_cold_labeled_sample_future_access_count_p50";
inline constexpr char hot_accuracy[] = "hp_hot_accuracy";
inline constexpr char hot_balanced_accuracy[] = "hp_hot_balanced_accuracy";
inline constexpr char hot_precision[] = "hp_hot_precision";
inline constexpr char hot_recall[] = "hp_hot_recall";
inline constexpr char eval_pred_hot_percent[] = "hp_eval_pred_hot_percent";
inline constexpr char eval_actual_hot_percent[] = "hp_eval_actual_hot_percent";
inline constexpr char actual_hot_avg_pred_hot_percent[] =
    "hp_actual_hot_avg_pred_hot_percent";
inline constexpr char actual_cold_avg_pred_hot_percent[] =
    "hp_actual_cold_avg_pred_hot_percent";
inline constexpr char predict_error_count[] = "hp_predict_error_count";
inline constexpr char hot_threshold[] = "hp_hot_threshold";
inline constexpr char otsu_candidate_threshold[] =
    "hp_otsu_candidate_threshold";
inline constexpr char hot_threshold_method[] = "hp_hot_threshold_method";
inline constexpr char train_queue_length[] = "hp_train_queue_length";
inline constexpr char train_drop_count[] = "hp_train_drop_count";
inline constexpr char snapshot_publish_count[] = "hp_snapshot_publish_count";
inline constexpr char arf_warning_count[] = "hp_arf_warning_count";
inline constexpr char arf_drift_count[] = "hp_arf_drift_count";
inline constexpr char arf_background_promotion_count[] =
    "hp_arf_background_promotion_count";
inline constexpr char arf_background_discard_count[] =
    "hp_arf_background_discard_count";
inline constexpr char arf_background_training_update_count[] =
    "hp_arf_background_training_update_count";
inline constexpr char arf_active_background_count[] =
    "hp_arf_active_background_count";
inline constexpr char op_read_count[] = "hp_op_read_count";
inline constexpr char op_sync_read_count[] = "hp_op_sync_read_count";
inline constexpr char op_sparse_read_count[] = "hp_op_sparse_read_count";
inline constexpr char op_write_count[] = "hp_op_write_count";
inline constexpr char op_writefull_count[] = "hp_op_writefull_count";
inline constexpr char op_writesame_count[] = "hp_op_writesame_count";
inline constexpr char predict_latency[] = "hp_predict_latency";
} // namespace field

inline constexpr CounterField counter_fields[] = {
    {field::enabled, Aggregate::sum},
    {field::io_count, Aggregate::sum},
    {field::labeled_io_total, Aggregate::sum},
    {field::pending_io_count, Aggregate::sum},
    {field::awaiting_prediction_count, Aggregate::sum},
    {field::eval_drop_count, Aggregate::sum},
    {field::heat_state_count, Aggregate::sum},
    {field::lru_count, Aggregate::sum},
    {field::protected_heat_state_count, Aggregate::sum},
    {field::heat_state_peak_count, Aggregate::sum},
    {field::lru_eviction_count, Aggregate::sum},
    {field::otsu_histogram_bin_count, Aggregate::sum},
    {field::otsu_histogram_vote_count, Aggregate::sum},
    {field::true_positive_count, Aggregate::sum},
    {field::false_positive_count, Aggregate::sum},
    {field::true_negative_count, Aggregate::sum},
    {field::false_negative_count, Aggregate::sum},
    {field::hot_labeled_sample_avg_future_access_count,
     Aggregate::hot_weighted, nullptr, Unit::scaled_x10000},
    {field::cold_labeled_sample_avg_future_access_count,
     Aggregate::cold_weighted, nullptr, Unit::scaled_x10000},
    {field::hot_labeled_sample_future_access_count_p99,
     Aggregate::hot_weighted,
     "hp_hot_labeled_sample_future_access_count_osd_p99_weighted_avg",
     Unit::scaled_x10000},
    {field::hot_labeled_sample_future_access_count_p95,
     Aggregate::hot_weighted,
     "hp_hot_labeled_sample_future_access_count_osd_p95_weighted_avg",
     Unit::scaled_x10000},
    {field::hot_labeled_sample_future_access_count_p50,
     Aggregate::hot_weighted,
     "hp_hot_labeled_sample_future_access_count_osd_p50_weighted_avg",
     Unit::scaled_x10000},
    {field::cold_labeled_sample_future_access_count_p99,
     Aggregate::cold_weighted,
     "hp_cold_labeled_sample_future_access_count_osd_p99_weighted_avg",
     Unit::scaled_x10000},
    {field::cold_labeled_sample_future_access_count_p95,
     Aggregate::cold_weighted,
     "hp_cold_labeled_sample_future_access_count_osd_p95_weighted_avg",
     Unit::scaled_x10000},
    {field::cold_labeled_sample_future_access_count_p50,
     Aggregate::cold_weighted,
     "hp_cold_labeled_sample_future_access_count_osd_p50_weighted_avg",
     Unit::scaled_x10000},
    {field::hot_accuracy, Aggregate::none, nullptr, Unit::scaled_x10000},
    {field::hot_balanced_accuracy, Aggregate::none, nullptr,
     Unit::scaled_x10000},
    {field::hot_precision, Aggregate::none, nullptr, Unit::scaled_x10000},
    {field::hot_recall, Aggregate::none, nullptr, Unit::scaled_x10000},
    {field::eval_pred_hot_percent, Aggregate::none, nullptr,
     Unit::scaled_x10000},
    {field::eval_actual_hot_percent, Aggregate::none, nullptr,
     Unit::scaled_x10000},
    {field::actual_hot_avg_pred_hot_percent, Aggregate::hot_weighted,
     nullptr, Unit::scaled_x10000},
    {field::actual_cold_avg_pred_hot_percent, Aggregate::cold_weighted,
     nullptr, Unit::scaled_x10000},
    {field::predict_error_count, Aggregate::sum},
    {field::hot_threshold, Aggregate::osd_average,
     "hp_hot_threshold_avg", Unit::scaled_x10000},
    {field::otsu_candidate_threshold, Aggregate::otsu_weighted,
     "hp_otsu_candidate_threshold_avg", Unit::scaled_x10000},
    {field::hot_threshold_method, Aggregate::none},
    {field::train_queue_length, Aggregate::sum},
    {field::train_drop_count, Aggregate::sum},
    {field::snapshot_publish_count, Aggregate::sum},
    {field::arf_warning_count, Aggregate::sum},
    {field::arf_drift_count, Aggregate::sum},
    {field::arf_background_promotion_count, Aggregate::sum},
    {field::arf_background_discard_count, Aggregate::sum},
    {field::arf_background_training_update_count, Aggregate::sum},
    {field::arf_active_background_count, Aggregate::sum},
    {field::op_read_count, Aggregate::sum},
    {field::op_sync_read_count, Aggregate::sum},
    {field::op_sparse_read_count, Aggregate::sum},
    {field::op_write_count, Aggregate::sum},
    {field::op_writefull_count, Aggregate::sum},
    {field::op_writesame_count, Aggregate::sum},
};

inline constexpr AverageField average_fields[] = {
    {field::predict_latency, Unit::nanoseconds},
};

inline constexpr const char* aggregate_name(const CounterField& field)
{
    return field.aggregate_name != nullptr ? field.aggregate_name : field.name;
}

} // namespace ceph::hp_telemetry

#endif
