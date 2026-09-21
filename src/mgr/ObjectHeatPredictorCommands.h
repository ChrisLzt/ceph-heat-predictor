#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include "common/cmdparse.h"
#include "include/buffer_fwd.h"

class ClusterState;
class DaemonStateIndex;
class Objecter;
namespace ceph { class Formatter; }

namespace ceph::mgr {
bool is_object_hp_command(const std::string& prefix);
// Control replies report asynchronous dispatch, not successful OSD execution.
// The host holds its usual command/connection lock while invoking this helper.
int handle_object_hp_command(
    const std::string& prefix, const cmdmap_t& cmdmap,
    Formatter* formatter, bufferlist& output, std::ostream& message,
    ClusterState& cluster_state, DaemonStateIndex& daemon_state,
    const std::function<bool(int32_t)>& is_connected,
    const std::function<Objecter&()>& get_objecter);
}
