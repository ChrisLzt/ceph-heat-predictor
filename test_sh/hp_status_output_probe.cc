#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/Formatter.h"
#include "heatpredictor/hp_telemetry.h"
#include "mgr/ObjectHeatPredictorStatus.h"
#include "mgr/ObjectHeatPredictorStatusFormatter.h"

int main(int argc, char** argv)
{
  if (argc != 4) {
    return 2;
  }
  const std::string scenario = argv[1];
  if (scenario == "command") {
#define COMMAND(sig, help, module, perm) \
    if (std::string(sig).find("osd hp status") == 0) std::cout << sig;
#include "mgr/MgrCommands.h"
#undef COMMAND
    return 0;
  }
  namespace field = ceph::hp_telemetry::field;
  std::vector<ceph::mgr::ObjectHpOsdStatus> osds;
  for (int id = 0; id < 2 && scenario != "no_osds"; ++id) {
    ceph::mgr::ObjectHpOsdStatus osd;
    osd.osd_id = id;
    osd.reporting = true;
    auto& c = osd.counters;
    c[field::status_publish_generation_begin] = 1;
    c[field::status_publish_generation_end] = 1;
    c[field::enabled] = !(scenario == "disabled" && id == 1);
    c[field::future_access_threshold] = scenario == "reset" ? 1 : 2 + id * 2;
    c[field::threshold_state] = scenario == "reset" ? 0 : id;
    c[field::otsu_positive_object_count] = scenario == "reset" ? 0 : 20 + id * 20;
    if (scenario != "empty" && scenario != "reset") {
      c[field::true_positive_count] = id == 0 ? 10 : 50;
      c[field::false_positive_count] = id == 0 ? 1 : 3;
      c[field::true_negative_count] = 15;
      c[field::false_negative_count] = id == 0 ? 4 : 2;
      c[field::labeled_io_total] = id == 0 ? 30 : 70;
      c[field::pending_io_count] = scenario == "complete" ? 0 : 2;
      c[field::io_count] = c[field::labeled_io_total] + c[field::pending_io_count];
      c[field::train_queue_length] = scenario == "complete" ? 0 : 1;
      c[field::snapshot_publish_count] = 7;
      c[field::op_write_count] = 5 + id;
      c[field::op_read_count] = c[field::io_count] - c[field::op_write_count];
      c[field::heat_state_count] = 20;
      c[field::hot_labeled_sample_avg_future_access_count] = 120000;
      c[field::cold_labeled_sample_avg_future_access_count] = 10000;
      osd.predict_latency_count = id == 0 ? 1 : 3;
      osd.predict_latency_sum_ns = id == 0 ? 100000 : 256000;
    }
    if (scenario == "no_timing") {
      osd.predict_latency_count = osd.predict_latency_sum_ns = 0;
    }
    if (scenario == "cold_only") {
      c[field::true_positive_count] = c[field::false_positive_count] = 0;
      c[field::false_negative_count] = 0;
      c[field::true_negative_count] = 50;
      c[field::labeled_io_total] = 50;
      c[field::io_count] = 52;
    }
    if (scenario == "missing" && id == 1) {
      c[field::status_publish_generation_end] = 2;
    }
    if (scenario == "errors") {
      c[field::predict_error_count] = 1;
      c[field::background_error_count] = 2;
      c[field::eval_drop_count] = 3;
      c[field::train_drop_count] = 4;
      c[field::io_count] += 3;
    }
    osds.push_back(std::move(osd));
  }
  auto status = ceph::mgr::aggregate_object_hp_status(osds);
  std::unique_ptr<ceph::Formatter> formatter(ceph::Formatter::create(argv[2]));
  ceph::mgr::format_object_hp_status(
    std::move(status), formatter.get(), std::cout,
    std::string(argv[3]) == "detail");
}
