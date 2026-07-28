#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "heatpredictor/hp_telemetry.h"
#include "mgr/ObjectHeatPredictorStatus.h"

namespace {

namespace field = ceph::hp_telemetry::field;
using ceph::mgr::ObjectHpOsdStatus;

void require(bool condition, const char *message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

ObjectHpOsdStatus reporting_osd(
    int32_t id,
    uint64_t threshold,
    uint64_t state,
    uint64_t positive_objects)
{
  ObjectHpOsdStatus osd;
  osd.osd_id = id;
  osd.reporting = true;
  osd.counters[field::status_publish_generation_begin] = 1;
  osd.counters[field::enabled] = 1;
  osd.counters[field::future_access_threshold] = threshold;
  osd.counters[field::threshold_state] = state;
  osd.counters[field::otsu_positive_object_count] = positive_objects;
  osd.counters[field::status_publish_generation_end] = 1;
  return osd;
}

} // namespace

int main()
{
  auto sparse = reporting_osd(0, 1, 0, 10);
  auto tracking = reporting_osd(1, 5, 1, 30);
  ObjectHpOsdStatus missing;
  missing.osd_id = 2;

  const auto status = ceph::mgr::aggregate_object_hp_status(
      {sparse, tracking, missing});
  require(status.up_osds == 3 && status.reporting_osds == 2,
          "MGR must distinguish up and reporting OSDs");
  require(status.missing_osds.size() == 1 &&
              status.missing_osds[0] == 2,
          "MGR must retain missing OSD ids");
  require(status.future_access_threshold_min == 1 &&
              status.future_access_threshold_max == 5,
          "MGR must report threshold extrema");
  require(status.threshold_state_sparse_osds == 1 &&
              status.threshold_state_tracking_osds == 1,
          "MGR must count sparse and tracking threshold states");

  const std::string threshold_avg = "hp_future_access_threshold_avg";
  require(status.weighted_count.at(threshold_avg) == 2,
          "threshold average must include every reporting OSD");
  require(std::abs(
              static_cast<double>(
                  status.weighted_sum.at(threshold_avg) /
                  status.weighted_count.at(threshold_avg)) -
              3.0) < 0.000001,
          "threshold average must be OSD-weighted");

  auto torn = reporting_osd(3, 9, 1, 50);
  torn.counters[field::status_publish_generation_end] = 2;
  const auto torn_status = ceph::mgr::aggregate_object_hp_status(
      {tracking, torn});
  require(torn_status.reporting_osds == 1,
          "MGR must reject an OSD status published across generations");
  require(torn_status.missing_osds.size() == 1 &&
              torn_status.missing_osds[0] == 3,
          "MGR must report a torn OSD status as missing");

  auto incomplete = reporting_osd(4, 3, 1, 20);
  incomplete.counters.erase(field::status_publish_generation_end);
  const auto incomplete_status =
      ceph::mgr::aggregate_object_hp_status({incomplete});
  require(incomplete_status.reporting_osds == 0 &&
              incomplete_status.missing_osds.size() == 1,
          "MGR must reject an OSD status without publication markers");

  std::cout << "PASS: heat predictor status contract probe" << std::endl;
  return 0;
}
