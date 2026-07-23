#ifndef CEPH_MGR_OBJECT_HEAT_PREDICTOR_STATUS_H
#define CEPH_MGR_OBJECT_HEAT_PREDICTOR_STATUS_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ceph::mgr {

struct ObjectHpOsdStatus {
  int32_t osd_id = -1;
  bool reporting = false;
  std::map<std::string, uint64_t> counters;
  uint64_t predict_latency_sum_ns = 0;
  uint64_t predict_latency_count = 0;
};

struct ObjectHpClusterStatus {
  uint64_t up_osds = 0;
  uint64_t reporting_osds = 0;
  uint64_t enabled_osds = 0;
  uint64_t disabled_osds = 0;
  std::vector<int32_t> missing_osds;
  std::map<std::string, uint64_t> sum;
  std::map<std::string, long double> weighted_sum;
  std::map<std::string, uint64_t> weighted_count;
  uint64_t threshold_method_initializing_osds = 0;
  uint64_t threshold_method_tracking_osds = 0;
  uint64_t threshold_method_holding_osds = 0;
  uint64_t predict_latency_sum_ns = 0;
  uint64_t predict_latency_count = 0;
};

ObjectHpClusterStatus aggregate_object_hp_status(
    const std::vector<ObjectHpOsdStatus>& osds);

} // namespace ceph::mgr

#endif
