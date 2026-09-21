#include "ObjectHeatPredictorCommands.h"

#include <cerrno>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <vector>
#include "ObjectHeatPredictorStatus.h"
#include "ObjectHeatPredictorStatusFormatter.h"
#include "ClusterState.h"
#include "DaemonState.h"
#include "common/Formatter.h"
#include "common/debug.h"
#include "heatpredictor/hp_telemetry.h"

#define dout_context g_ceph_context
#define dout_subsys ceph_subsys_mgr
#undef dout_prefix
#define dout_prefix *_dout << "mgr.object_hp " << __func__ << " "

namespace {
  using ceph::hp_telemetry::counter_fields;

  bool get_object_hp_counter(
    const DaemonStatePtr& state,
    const std::string& field,
    uint64_t *value)
  {
    const std::string path = "object_hp_status." + field;
    auto i = state->perf_counters.instances.find(path);
    if (i == state->perf_counters.instances.end()) {
      return false;
    }
    auto t = state->perf_counters.types.find(path);
    if (t == state->perf_counters.types.end() ||
        (t->second.type & PERFCOUNTER_LONGRUNAVG) ||
        i->second.get_data().empty()) {
      return false;
    }
    *value = i->second.get_latest_data().v;
    return true;
  }

  bool get_object_hp_average(
    const DaemonStatePtr& state,
    const std::string& field,
    uint64_t *sum,
    uint64_t *count)
  {
    const std::string path = "object_hp_status." + field;
    auto i = state->perf_counters.instances.find(path);
    if (i == state->perf_counters.instances.end()) {
      return false;
    }
    auto t = state->perf_counters.types.find(path);
    if (t == state->perf_counters.types.end() ||
        !(t->second.type & PERFCOUNTER_LONGRUNAVG) ||
        i->second.get_data_avg().empty()) {
      return false;
    }
    const auto& latest = i->second.get_latest_data_avg();
    *sum = latest.s;
    *count = latest.c;
    return true;
  }
}
namespace ceph::mgr {
bool is_object_hp_command(const std::string& prefix) {
  return prefix == "osd hp status" || prefix == "osd hp reset" ||
         prefix == "osd hp enable" || prefix == "osd hp disable" ||
         prefix == "osd hp trace start" || prefix == "osd hp trace stop";
}

int handle_object_hp_command(
    const std::string& prefix, const cmdmap_t& cmdmap,
    Formatter* f, bufferlist& out, std::ostream& ss,
    ClusterState& cluster_state, DaemonStateIndex& daemon_state,
    const std::function<bool(int32_t)>& is_connected,
    const std::function<Objecter&()>& get_objecter) {
  int r = 0;
  if (prefix == "osd hp status") {
    std::set<int32_t> up_osds;
    cluster_state.with_osdmap([&](const OSDMap& osdmap) {
      osdmap.get_up_osds(up_osds);
    });

    std::vector<ceph::mgr::ObjectHpOsdStatus> osd_statuses;
    osd_statuses.reserve(up_osds.size());
    for (auto osd : up_osds) {
      ceph::mgr::ObjectHpOsdStatus osd_status;
      osd_status.osd_id = osd;
      DaemonStatePtr state = daemon_state.get(DaemonKey{"osd", std::to_string(osd)});
      if (!state) {
        osd_statuses.push_back(std::move(osd_status));
        continue;
      }

      {
        std::lock_guard l(state->lock);
        uint64_t hp_io_count = 0;
        uint64_t labeled_io_total = 0;
        bool has_hp_io_count =
          get_object_hp_counter(
            state, ceph::hp_telemetry::field::io_count, &hp_io_count);
        bool has_labeled_io_total =
          get_object_hp_counter(
            state, ceph::hp_telemetry::field::labeled_io_total,
            &labeled_io_total);
        if (!has_hp_io_count && !has_labeled_io_total) {
          osd_statuses.push_back(std::move(osd_status));
          continue;
        }
        osd_status.reporting = true;
        for (const auto& field : counter_fields) {
          uint64_t value = 0;
          if (get_object_hp_counter(state, field.name, &value)) {
            osd_status.counters[field.name] = value;
          }
        }
        get_object_hp_average(
          state, ceph::hp_telemetry::field::predict_latency,
          &osd_status.predict_latency_sum_ns,
          &osd_status.predict_latency_count);
      }
      osd_statuses.push_back(std::move(osd_status));
    }

    auto cluster_status =
      ceph::mgr::aggregate_object_hp_status(osd_statuses);
    bool detail = false;
    ceph::common::cmd_getval(cmdmap, "detail", detail);
    std::ostringstream output;
    ceph::mgr::format_object_hp_status(
      std::move(cluster_status), f, output, detail);
    out.append(output.str());
    return 0;
  } else if (prefix == "osd hp reset" ||
             prefix == "osd hp enable" ||
             prefix == "osd hp disable" ||
             prefix == "osd hp trace start" ||
             prefix == "osd hp trace stop") {
    std::unique_ptr<Formatter> default_formatter;
    if (!f) {
      default_formatter.reset(Formatter::create("json-pretty"));
      f = default_formatter.get();
    }
    const std::string action = prefix == "osd hp enable" ? "enable" :
      (prefix == "osd hp disable" ? "disable" :
       (prefix == "osd hp trace start" ? "trace start" :
        (prefix == "osd hp trace stop" ? "trace stop" : "reset")));
    const std::string osd_prefix = "object_hp " + action;
    std::set<int32_t> up_osds;
    cluster_state.with_osdmap([&](const OSDMap& osdmap) {
      osdmap.get_up_osds(up_osds);
    });

    std::vector<int32_t> sent_osds;
    std::vector<int32_t> missing_osds;
    std::string phase;
    std::string directory;
    if (prefix == "osd hp trace start") {
      ceph::common::cmd_getval(cmdmap, "phase", phase);
      ceph::common::cmd_getval(cmdmap, "directory", directory);
    }
    std::ostringstream hp_cmd_stream;
    JSONFormatter hp_cmd_formatter;
    hp_cmd_formatter.open_object_section("command");
    hp_cmd_formatter.dump_string("prefix", osd_prefix);
    if (!phase.empty()) {
      hp_cmd_formatter.dump_string("phase", phase);
    }
    if (!directory.empty()) {
      hp_cmd_formatter.dump_string("directory", directory);
    }
    hp_cmd_formatter.close_section();
    hp_cmd_formatter.flush(hp_cmd_stream);
    const std::string hp_cmd = hp_cmd_stream.str();
    for (auto osd : up_osds) {
      if (!is_connected(osd)) {
        missing_osds.push_back(osd);
        continue;
      }
      sent_osds.push_back(osd);
      ceph_tid_t tid;
      bufferlist inbl;
      get_objecter().osd_command(
        osd,
        {hp_cmd},
        inbl,
        &tid,
        [osd, osd_prefix](boost::system::error_code ec,
                          std::string outs,
                          bufferlist outbl) {
          if (ec) {
            dout(1) << osd_prefix << " failed on osd." << osd
                    << ": " << ec.message() << " " << outs << dendl;
          } else {
            dout(10) << osd_prefix << " finished on osd." << osd
                     << ": " << outs << dendl;
          }
        });
    }

    f->open_object_section("osd_hp_control");
    f->dump_string("action", action);
    f->dump_unsigned("requested", up_osds.size());
    f->dump_unsigned("sent", sent_osds.size());
    f->dump_unsigned("not_connected", missing_osds.size());
    f->open_array_section("sent_osds");
    for (auto osd : sent_osds) {
      f->dump_int("osd", osd);
    }
    f->close_section();
    f->open_array_section("not_connected_osds");
    for (auto osd : missing_osds) {
      f->dump_int("osd", osd);
    }
    f->close_section();
    f->close_section();
    f->flush(out);

    if (!missing_osds.empty()) {
      ss << "sent " << osd_prefix << " to " << sent_osds.size()
         << " osd(s); " << missing_osds.size() << " osd(s) not connected";
      r = sent_osds.empty() ? -EAGAIN : 0;
    } else {
      ss << "sent " << osd_prefix << " to " << sent_osds.size() << " osd(s)";
    }
    return r;
  }
  return -EINVAL;
}
}
