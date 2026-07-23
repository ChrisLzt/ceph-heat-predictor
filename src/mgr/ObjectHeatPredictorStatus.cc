#include "ObjectHeatPredictorStatus.h"

#include "heatpredictor/hp_telemetry.h"

namespace ceph::mgr {
namespace {

uint64_t counter_value(
    const std::map<std::string, uint64_t>& counters,
    const char* name)
{
  auto value = counters.find(name);
  return value != counters.end() ? value->second : 0;
}

} // namespace

ObjectHpClusterStatus aggregate_object_hp_status(
    const std::vector<ObjectHpOsdStatus>& osds)
{
  using hp_telemetry::Aggregate;
  using hp_telemetry::aggregate_name;
  using hp_telemetry::counter_fields;
  namespace field = hp_telemetry::field;

  ObjectHpClusterStatus result;
  result.up_osds = osds.size();

  for (const auto& osd : osds) {
    if (!osd.reporting) {
      result.missing_osds.push_back(osd.osd_id);
      continue;
    }
    ++result.reporting_osds;
    result.predict_latency_sum_ns += osd.predict_latency_sum_ns;
    result.predict_latency_count += osd.predict_latency_count;

    const uint64_t actual_hot_count =
      counter_value(osd.counters, field::true_positive_count) +
      counter_value(osd.counters, field::false_negative_count);
    const uint64_t actual_cold_count =
      counter_value(osd.counters, field::true_negative_count) +
      counter_value(osd.counters, field::false_positive_count);
    const uint64_t otsu_vote_count =
      counter_value(osd.counters, field::otsu_histogram_vote_count);

    if (counter_value(osd.counters, field::enabled) > 0) {
      ++result.enabled_osds;
    } else {
      ++result.disabled_osds;
    }

    switch (counter_value(osd.counters, field::hot_threshold_method)) {
    case 1:
      ++result.threshold_method_tracking_osds;
      break;
    case 2:
      ++result.threshold_method_holding_osds;
      break;
    default:
      ++result.threshold_method_initializing_osds;
      break;
    }

    for (const auto& descriptor : counter_fields) {
      auto value = osd.counters.find(descriptor.name);
      if (value == osd.counters.end()) {
        continue;
      }
      const char* output_name = aggregate_name(descriptor);
      switch (descriptor.aggregate) {
      case Aggregate::sum:
        result.sum[descriptor.name] += value->second;
        break;
      case Aggregate::osd_average:
        result.weighted_sum[output_name] += value->second;
        ++result.weighted_count[output_name];
        break;
      case Aggregate::hot_weighted:
        if (actual_hot_count > 0) {
          result.weighted_sum[output_name] +=
            static_cast<long double>(value->second) * actual_hot_count;
          result.weighted_count[output_name] += actual_hot_count;
        }
        break;
      case Aggregate::cold_weighted:
        if (actual_cold_count > 0) {
          result.weighted_sum[output_name] +=
            static_cast<long double>(value->second) * actual_cold_count;
          result.weighted_count[output_name] += actual_cold_count;
        }
        break;
      case Aggregate::otsu_weighted:
        if (otsu_vote_count > 0) {
          result.weighted_sum[output_name] +=
            static_cast<long double>(value->second) * otsu_vote_count;
          result.weighted_count[output_name] += otsu_vote_count;
        }
        break;
      case Aggregate::none:
        break;
      }
    }
  }

  return result;
}

} // namespace ceph::mgr
