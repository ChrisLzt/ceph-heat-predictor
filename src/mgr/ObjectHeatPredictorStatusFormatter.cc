#include "ObjectHeatPredictorStatusFormatter.h"

#include <array>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>

#include "common/Formatter.h"
#include "heatpredictor/include/Metrics.h"

namespace ceph::mgr {
namespace {

double hp_ratio(uint64_t numerator, uint64_t denominator)
{
  if (denominator == 0 || numerator == 0) {
    return 0.0;
  }
  return static_cast<double>(
    static_cast<long double>(numerator) / denominator);
}

double hp_percent(uint64_t numerator, uint64_t denominator)
{
  return hp_ratio(numerator, denominator) * 100.0;
}

double hp_from_x10000(long double value)
{
  return static_cast<double>(value / 10000.0L);
}

double hp_percent_from_x10000(long double value)
{
  return static_cast<double>(value / 100.0L);
}

void hp_dump_float(Formatter *f, std::string_view name, double value)
{
  f->dump_format_unquoted(name, "%.5g", value);
}

void dump_detail(ObjectHpClusterStatus& cluster_status, Formatter* f)
{
  auto& summary = cluster_status.sum;
  auto& weighted_sum = cluster_status.weighted_sum;
  auto& weighted_count = cluster_status.weighted_count;
  const auto& missing_osds = cluster_status.missing_osds;
  const uint64_t threshold_state_sparse_osds =
    cluster_status.threshold_state_sparse_osds;
  const uint64_t threshold_state_tracking_osds =
    cluster_status.threshold_state_tracking_osds;
  const uint64_t enabled_osds = cluster_status.enabled_osds;
  const uint64_t disabled_osds = cluster_status.disabled_osds;
  const uint64_t predict_latency_sum_ns =
    cluster_status.predict_latency_sum_ns;
  const uint64_t predict_latency_count =
    cluster_status.predict_latency_count;

  f->open_object_section("osd_hp_status");
  f->open_object_section("summary");

  f->open_object_section("osds");
  f->dump_unsigned("up_osds", cluster_status.up_osds);
  f->dump_unsigned("reporting_osds", cluster_status.reporting_osds);
  f->dump_unsigned("enabled_osds", enabled_osds);
  f->dump_unsigned("disabled_osds", disabled_osds);
  f->open_array_section("missing_osds");
  for (auto osd : missing_osds) {
    f->dump_int("osd", osd);
  }
  f->close_section();
  f->close_section();

  f->open_object_section("samples");
  f->dump_unsigned("hp_io_count", summary["hp_io_count"]);
  f->dump_unsigned("hp_labeled_io_total", summary["hp_labeled_io_total"]);
  f->dump_unsigned("hp_pending_io_count", summary["hp_pending_io_count"]);
  f->dump_unsigned("hp_awaiting_prediction_count",
                   summary["hp_awaiting_prediction_count"]);
  f->dump_unsigned("hp_eval_drop_count", summary["hp_eval_drop_count"]);
  f->close_section();

  f->open_object_section("heat_state");
  f->dump_unsigned("hp_heat_state_count", summary["hp_heat_state_count"]);
  f->dump_unsigned("hp_lru_count", summary["hp_lru_count"]);
  f->dump_unsigned("hp_protected_heat_state_count",
                   summary["hp_protected_heat_state_count"]);
  f->dump_unsigned("hp_heat_state_peak_count",
                   summary["hp_heat_state_peak_count"]);
  f->dump_unsigned("hp_lru_eviction_count",
                   summary["hp_lru_eviction_count"]);
  f->dump_unsigned("hp_otsu_histogram_bin_count",
                   summary["hp_otsu_histogram_bin_count"]);
  f->dump_unsigned("hp_otsu_positive_object_count",
                   summary["hp_otsu_positive_object_count"]);
  f->dump_unsigned("hp_otsu_zero_observation_count",
                   summary["hp_otsu_zero_observation_count"]);
  f->dump_unsigned("hp_otsu_upper_clamped_object_count",
                   summary["hp_otsu_upper_clamped_object_count"]);
  f->dump_unsigned("hp_sparse_threshold_sample_count",
                   summary["hp_sparse_threshold_sample_count"]);
  f->open_object_section("future_access_threshold");
  f->dump_unsigned("min", cluster_status.future_access_threshold_min);
  f->dump_unsigned("max", cluster_status.future_access_threshold_max);
  {
    uint64_t weight =
      weighted_count["hp_future_access_threshold_avg"];
    hp_dump_float(
      f, "avg",
      weight > 0
        ? weighted_sum["hp_future_access_threshold_avg"] / weight
        : 0.0);
  }
  f->open_object_section("state_osds");
  f->dump_unsigned("sparse", threshold_state_sparse_osds);
  f->dump_unsigned("tracking", threshold_state_tracking_osds);
  f->close_section();
  f->close_section();
  f->close_section();

  f->open_object_section("confusion_matrix");
  f->dump_unsigned("hp_true_positive_count", summary["hp_true_positive_count"]);
  f->dump_unsigned("hp_false_positive_count", summary["hp_false_positive_count"]);
  f->dump_unsigned("hp_true_negative_count", summary["hp_true_negative_count"]);
  f->dump_unsigned("hp_false_negative_count", summary["hp_false_negative_count"]);
  f->close_section();

  f->open_object_section("actual_behavior");
  auto weighted_x10000_value = [&](const std::string& field) {
    uint64_t weight = weighted_count[field];
    return weight > 0
      ? hp_from_x10000(weighted_sum[field] / weight)
      : 0.0;
  };
  auto dump_weighted_x10000 = [&](const std::string& field) {
    hp_dump_float(f, field, weighted_x10000_value(field));
  };
  {
    uint64_t weight =
      weighted_count["hp_hot_labeled_sample_avg_future_access_count"];
    hp_dump_float(
      f,
      "hp_hot_labeled_sample_avg_future_access_count",
      weight > 0 ? hp_from_x10000(
        weighted_sum["hp_hot_labeled_sample_avg_future_access_count"] /
        weight) : 0.0);
  }
  {
    uint64_t weight =
      weighted_count["hp_cold_labeled_sample_avg_future_access_count"];
    hp_dump_float(
      f,
      "hp_cold_labeled_sample_avg_future_access_count",
      weight > 0 ? hp_from_x10000(
        weighted_sum["hp_cold_labeled_sample_avg_future_access_count"] /
        weight) : 0.0);
  }
  {
    double cold_avg =
      weighted_x10000_value("hp_cold_labeled_sample_avg_future_access_count");
    hp_dump_float(
      f,
      "hp_future_access_count_hot_cold_ratio",
      cold_avg > 0
        ? weighted_x10000_value(
            "hp_hot_labeled_sample_avg_future_access_count") / cold_avg
        : 0.0);
  }
  dump_weighted_x10000(
    "hp_hot_labeled_sample_future_access_count_osd_p99_weighted_avg");
  dump_weighted_x10000(
    "hp_hot_labeled_sample_future_access_count_osd_p95_weighted_avg");
  dump_weighted_x10000(
    "hp_hot_labeled_sample_future_access_count_osd_p50_weighted_avg");
  dump_weighted_x10000(
    "hp_cold_labeled_sample_future_access_count_osd_p99_weighted_avg");
  dump_weighted_x10000(
    "hp_cold_labeled_sample_future_access_count_osd_p95_weighted_avg");
  dump_weighted_x10000(
    "hp_cold_labeled_sample_future_access_count_osd_p50_weighted_avg");
  f->close_section();

  f->open_object_section("prediction");
  const uint64_t tp = summary["hp_true_positive_count"];
  const uint64_t fp = summary["hp_false_positive_count"];
  const uint64_t tn = summary["hp_true_negative_count"];
  const uint64_t fn = summary["hp_false_negative_count"];
  const uint64_t labeled_total = tp + fp + tn + fn;
  hp_dump_float(f, "hp_hot_accuracy", hp_percent(tp + tn, labeled_total));
  hp_dump_float(
    f, "hp_hot_balanced_accuracy",
    100.0 * hp_binary_balanced_accuracy(tp, fp, tn, fn));
  hp_dump_float(f, "hp_hot_precision", hp_percent(tp, tp + fp));
  hp_dump_float(f, "hp_hot_recall", hp_percent(tp, tp + fn));
  hp_dump_float(f, "hp_eval_pred_hot_percent", hp_percent(tp + fp, labeled_total));
  hp_dump_float(f, "hp_eval_actual_hot_percent", hp_percent(tp + fn, labeled_total));
  f->dump_unsigned("hp_predict_error_count",
                   summary["hp_predict_error_count"]);
  f->dump_unsigned("hp_background_error_count",
                   summary["hp_background_error_count"]);
  {
    uint64_t weight =
      weighted_count["hp_actual_hot_avg_pred_hot_percent"];
    hp_dump_float(
      f,
      "hp_actual_hot_avg_pred_hot_percent",
      weight > 0 ? hp_percent_from_x10000(
        weighted_sum["hp_actual_hot_avg_pred_hot_percent"] / weight) : 0.0);
  }
  {
    uint64_t weight =
      weighted_count["hp_actual_cold_avg_pred_hot_percent"];
    hp_dump_float(
      f,
      "hp_actual_cold_avg_pred_hot_percent",
      weight > 0 ? hp_percent_from_x10000(
        weighted_sum["hp_actual_cold_avg_pred_hot_percent"] / weight) : 0.0);
  }
  f->close_section();

  f->open_object_section("training");
  f->dump_unsigned("hp_train_queue_length", summary["hp_train_queue_length"]);
  f->dump_unsigned("hp_train_drop_count", summary["hp_train_drop_count"]);
  f->dump_unsigned("hp_snapshot_publish_count", summary["hp_snapshot_publish_count"]);
  f->close_section();

  f->open_object_section("model_adaptation");
  f->dump_unsigned("hp_arf_warning_count",
                   summary["hp_arf_warning_count"]);
  f->dump_unsigned("hp_arf_drift_count",
                   summary["hp_arf_drift_count"]);
  f->dump_unsigned("hp_arf_background_promotion_count",
                   summary["hp_arf_background_promotion_count"]);
  f->dump_unsigned("hp_arf_background_discard_count",
                   summary["hp_arf_background_discard_count"]);
  f->dump_unsigned("hp_arf_background_training_update_count",
                   summary["hp_arf_background_training_update_count"]);
  f->dump_unsigned("hp_arf_active_background_count",
                   summary["hp_arf_active_background_count"]);
  f->close_section();

  f->open_object_section("trace");
  f->dump_unsigned("enabled_osds", summary["hp_trace_enabled"]);
  f->dump_unsigned("hp_trace_queue_length",
                   summary["hp_trace_queue_length"]);
  f->dump_unsigned("hp_trace_written_count",
                   summary["hp_trace_written_count"]);
  f->dump_unsigned("hp_trace_drop_count",
                   summary["hp_trace_drop_count"]);
  f->dump_unsigned("hp_trace_write_error_count",
                   summary["hp_trace_write_error_count"]);
  f->close_section();

  f->open_object_section("latency");
  f->open_object_section("hp_predict_latency");
  f->dump_unsigned("avgcount", predict_latency_count);
  f->dump_unsigned("sum_ns", predict_latency_sum_ns);
  f->dump_unsigned(
    "avgtime_ns",
    predict_latency_count > 0
      ? predict_latency_sum_ns / predict_latency_count
      : 0);
  f->close_section();
  f->close_section();

  f->open_object_section("read_ops");
  f->dump_unsigned("hp_op_read_count", summary["hp_op_read_count"]);
  f->dump_unsigned("hp_op_sync_read_count", summary["hp_op_sync_read_count"]);
  f->dump_unsigned("hp_op_sparse_read_count", summary["hp_op_sparse_read_count"]);
  f->close_section();

  f->open_object_section("write_ops");
  f->dump_unsigned("hp_op_write_count", summary["hp_op_write_count"]);
  f->dump_unsigned("hp_op_writefull_count", summary["hp_op_writefull_count"]);
  f->dump_unsigned("hp_op_writesame_count", summary["hp_op_writesame_count"]);
  f->close_section();

  f->close_section();
  f->close_section();

}

struct BriefMetric {
  const char* name;
  const char* label;
  std::optional<double> value;
};

std::optional<double> percent_if_available(uint64_t count, uint64_t total)
{
  return total ? std::optional<double>(hp_percent(count, total)) : std::nullopt;
}

void dump_brief(ObjectHpClusterStatus& status, Formatter* f, std::ostream& out)
{
  auto& sum = status.sum;
  const auto tp = sum["hp_true_positive_count"];
  const auto fp = sum["hp_false_positive_count"];
  const auto tn = sum["hp_true_negative_count"];
  const auto fn = sum["hp_false_negative_count"];
  const auto labeled = sum["hp_labeled_io_total"];
  const std::array<BriefMetric, 5> metrics{{
    {"hp_hot_accuracy", "Accuracy", percent_if_available(tp + tn, labeled)},
    {"hp_hot_precision", "Precision", percent_if_available(tp, tp + fp)},
    {"hp_hot_recall", "Recall", percent_if_available(tp, tp + fn)},
    {"hp_eval_pred_hot_percent", "predicted", percent_if_available(tp + fp, labeled)},
    {"hp_eval_actual_hot_percent", "actual", percent_if_available(tp + fn, labeled)}
  }};
  const std::optional<double> latency_us = status.predict_latency_count
    ? std::optional<double>(static_cast<double>(status.predict_latency_sum_ns) /
                            status.predict_latency_count / 1000.0)
    : std::nullopt;
  const std::array<const char*, 4> error_fields{{
    "hp_predict_error_count", "hp_background_error_count",
    "hp_eval_drop_count", "hp_train_drop_count"
  }};
  bool has_alerts = status.up_osds == 0 || !status.missing_osds.empty() ||
                    status.disabled_osds > 0;
  for (const auto* field : error_fields) {
    has_alerts = has_alerts || sum[field] > 0;
  }

  if (f) {
    f->open_object_section("osd_hp_status");
    f->open_object_section("summary");
    f->open_object_section("osds");
    f->dump_unsigned("enabled_osds", status.enabled_osds);
    f->dump_unsigned("reporting_osds", status.reporting_osds);
    f->dump_unsigned("up_osds", status.up_osds);
    f->close_section();
    f->open_object_section("samples");
    f->dump_unsigned("hp_labeled_io_total", labeled);
    f->close_section();
    f->open_object_section("prediction");
    for (const auto& metric : metrics) {
      if (metric.value) {
        hp_dump_float(f, metric.name, *metric.value);
      } else {
        f->dump_format_unquoted(metric.name, "null");
      }
    }
    f->close_section();
    f->open_object_section("latency");
    f->open_object_section("hp_predict_latency");
    if (latency_us) {
      hp_dump_float(f, "avgtime_us", *latency_us);
    } else {
      f->dump_format_unquoted("avgtime_us", "null");
    }
    f->close_section();
    f->close_section();
    if (has_alerts) {
      f->open_object_section("alerts");
      if (status.up_osds == 0) {
        f->dump_bool("no_up_osds", true);
      }
      if (!status.missing_osds.empty()) {
        f->open_array_section("missing_osds");
        for (auto osd : status.missing_osds) {
          f->dump_int("osd", osd);
        }
        f->close_section();
      }
      if (status.disabled_osds) {
        f->dump_unsigned("disabled_osds", status.disabled_osds);
      }
      for (const auto* field : error_fields) {
        if (sum[field]) {
          f->dump_unsigned(field, sum[field]);
        }
      }
      f->close_section();
    }
    f->close_section();
    f->close_section();
    f->flush(out);
    return;
  }

  std::ostringstream text;
  text << std::fixed << std::setprecision(2);
  text << "OSDs: enabled " << status.enabled_osds
       << " / reporting " << status.reporting_osds << " / up " << status.up_osds
       << "\nEvaluated I/O: " << labeled << '\n';
  auto print_metric = [&](const BriefMetric& metric) {
    text << metric.label << ": ";
    if (metric.value) {
      text << *metric.value << '%';
    } else {
      text << "N/A";
    }
  };
  if (labeled) {
    for (size_t i = 0; i < 3; ++i) {
      if (i) {
        text << "  ";
      }
      print_metric(metrics[i]);
    }
    text << "\nHot ratio: ";
    print_metric(metrics[3]);
    text << " / ";
    print_metric(metrics[4]);
  } else {
    text << "Accuracy / Precision / Recall: N/A (no evaluated samples)\n"
         << "Hot ratio: N/A (no evaluated samples)";
  }
  text << "\nMean prediction latency: ";
  if (latency_us) {
    text << *latency_us << " us\n";
  } else {
    text << "N/A (no timing samples)\n";
  }
  if (status.up_osds == 0) {
    text << "ALERT: no up OSDs\n";
  }
  if (!status.missing_osds.empty()) {
    text << "ALERT: missing OSDs:";
    for (auto osd : status.missing_osds) {
      text << ' ' << osd;
    }
    text << " (statistics cover reporting OSDs only)\n";
  }
  if (status.disabled_osds) {
    text << "ALERT: disabled OSDs: " << status.disabled_osds << '\n';
  }
  for (const auto* field : error_fields) {
    if (sum[field]) {
      text << "ALERT: " << field << ": " << sum[field] << '\n';
    }
  }
  out << text.str();
}

} // namespace

void format_object_hp_status(ObjectHpClusterStatus status,
                             Formatter* formatter, std::ostream& out, bool detail)
{
  if (!detail) {
    dump_brief(status, formatter, out);
    return;
  }
  std::unique_ptr<Formatter> fallback;
  if (!formatter) {
    fallback.reset(Formatter::create("json-pretty"));
    formatter = fallback.get();
  }
  dump_detail(status, formatter);
  formatter->flush(out);
}

} // namespace ceph::mgr
