#include <cstdlib>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include "common/admin_socket.h"
#include "common/ceph_context.h"
#include "common/Formatter.h"
#include "common/hobject.h"
#include "common/perf_counters_collection.h"
#include "include/rados.h"
#include "osd/ObjectHeatPredictor.h"

// Linker wrapping observes real adapter lock acquisitions without adding
// instrumentation or test hooks to production code. Count this thread only.
static thread_local unsigned long read_lock_calls = 0;
extern "C" int __real_pthread_rwlock_rdlock(pthread_rwlock_t* lock);
extern "C" int __wrap_pthread_rwlock_rdlock(pthread_rwlock_t* lock) {
  ++read_lock_calls;
  return __real_pthread_rwlock_rdlock(lock);
}

static void require(bool ok, const char* message) {
  if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
static void command(ObjectHeatPredictor& hp, const std::string& prefix,
                    const std::string& label = "", const cmdmap_t& args = {}) {
  ceph::JSONFormatter formatter;
  require(hp.handle_command(prefix, args, &formatter), "unhandled HP command");
  if (!label.empty()) {
    std::ostringstream out;
    formatter.flush(out);
    std::cout << label << '\t' << out.str() << '\n';
  }
}
static void perf(CephContext& cct, const std::string& label) {
  ceph::JSONFormatter formatter;
  cct.get_perfcounters_collection()->dump_formatted(&formatter, false);
  std::ostringstream out;
  formatter.flush(out);
  std::cout << label << '\t' << out.str() << '\n';
}

class ModuleHook : public AdminSocketHook {
  ObjectHeatPredictor& hp;
public:
  explicit ModuleHook(ObjectHeatPredictor& hp) : hp(hp) {}
  int call(std::string_view prefix, const cmdmap_t& args, const bufferlist&,
           ceph::Formatter* formatter, std::ostream&, bufferlist&) override {
    return hp.handle_command(prefix, args, formatter) ? 0 : -EINVAL;
  }
};

int main(int argc, char** argv) {
  require(argc == 2, "Trace directory required");
  CephContext first(CEPH_ENTITY_TYPE_OSD), second(CEPH_ENTITY_TYPE_OSD);
  ObjectHeatPredictor a, b;
  hobject_t early(object_t("early-object"), "", CEPH_NOSNAP, 43, 1, "");
  a.observe(early, CEPH_OSD_OP_READ, 1);
  ceph::JSONFormatter early_formatter;
  require(!a.handle_command("object_hp enable", {}, &early_formatter),
          "uninitialized module must reject enable");
  perf(first, "before_init");
  a.init(&first, 0);
  perf(first, "after_init");
  b.init(&second, 1);
  hobject_t object(object_t("test-object"), "", CEPH_NOSNAP, 42, 1, "");
  const auto initially_disabled_locks = read_lock_calls;
  for (int i = 0; i < 1000; ++i) a.observe(object, CEPH_OSD_OP_READ, 1);
  require(read_lock_calls == initially_disabled_locks,
          "disabled observation must not acquire a read lock");
  command(a, "object_hp status", "initial");
  require(!a.handle_command("not an HP command", {}, nullptr),
          "module must not claim other commands");

  ModuleHook hook(a);
  a.register_commands(first.get_admin_socket(), &hook);
  // Exercise the registered command signature and the real admin-socket parser.
  bufferlist out;
  std::ostringstream error;
  require(first.get_admin_socket()->execute_command(
            {"{\"prefix\":\"object_hp enable\"}"}, {}, error, &out) == 0,
          "registered enable command failed");
  first.get_admin_socket()->unregister_commands(&hook);

  a.observe(object, CEPH_OSD_OP_STAT, 1);
  a.observe(object, CEPH_OSD_OP_READ, 0);
  command(a, "object_hp status", "filtered");
  const auto enabled_locks = read_lock_calls;
  for (auto op : {CEPH_OSD_OP_READ, CEPH_OSD_OP_SYNC_READ,
                 CEPH_OSD_OP_SPARSE_READ, CEPH_OSD_OP_WRITE,
                 CEPH_OSD_OP_WRITEFULL, CEPH_OSD_OP_WRITESAME}) {
    a.observe(object, op, 1);
  }
  require(read_lock_calls > enabled_locks,
          "enabled observations must retain synchronization");
  command(a, "object_hp status", "observed");
  perf(first, "observed_perf");
  command(b, "object_hp status", "other_instance");
  command(a, "object_hp reset", "reset");

  cmdmap_t args;
  args["phase"] = std::string("refactor-probe");
  args["directory"] = std::string(argv[1]);
  command(a, "object_hp trace start", "trace_start", args);
  a.observe(object, CEPH_OSD_OP_READ, 1);
  command(a, "object_hp reset", "trace_reset");
  command(a, "object_hp status", "trace_rotated");
  command(a, "object_hp trace stop", "trace_stop");

  // Real adapter locks: reset/disable must exclude observations, while status
  // and background publications remain safe. No live Ceph daemon is involved.
  std::vector<std::thread> observers;
  for (int worker = 0; worker < 4; ++worker) {
    observers.emplace_back([&] {
      for (int i = 0; i < 2000; ++i) a.observe(object, CEPH_OSD_OP_READ, 1);
    });
  }
  for (int i = 0; i < 10; ++i) {
    command(a, "object_hp reset");
    command(a, "object_hp disable");
    command(a, "object_hp enable");
    command(a, "object_hp status");
  }
  for (auto& observer : observers) observer.join();
  command(a, "object_hp disable", "disabled");
  const auto disabled_again_locks = read_lock_calls;
  for (int i = 0; i < 1000; ++i) a.observe(object, CEPH_OSD_OP_READ, 1);
  require(read_lock_calls == disabled_again_locks,
          "observations after disable must not acquire a read lock");
  command(a, "object_hp status", "disabled_observed");
  command(a, "object_hp enable");
  command(a, "object_hp trace start", "trace_at_shutdown", args);
  a.observe(object, CEPH_OSD_OP_READ, 1);
  a.shutdown();
  a.shutdown();
  perf(first, "after_shutdown");
  {
    ObjectHeatPredictor replacement;
    replacement.init(&first, 0);
    command(replacement, "object_hp status", "replacement");
  }
  perf(first, "after_destructor");
  std::cerr << "OSD module probe: PASS\n";
}
