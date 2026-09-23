// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
#pragma once

#include <cstdint>
#include <atomic>
#include "heatpredictor/hp_access_type.h"
#include <memory>
#include <string>
#include <string_view>
#include "include/common_fwd.h"
#include "common/cmdparse.h"

namespace ceph { class Formatter; }
class AdminSocket;
class AdminSocketHook;
struct hobject_t;

// One module per OSDService. The host owns command/request routing; this module
// owns all predictor and perf-counter state. No storage I/O is performed.
class ObjectHeatPredictor {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  ObjectHeatPredictor();
  ~ObjectHeatPredictor();
  ObjectHeatPredictor(const ObjectHeatPredictor&) = delete;
  ObjectHeatPredictor& operator=(const ObjectHeatPredictor&) = delete;

  // Call once with a non-null context before registering/dispatching commands.
  // Creates perf counters even while disabled. Init must not race commands;
  // observations before init are ignored because the predictor is disabled.
  // Optional storage gate is bound at init, before commands or observations.
  void init(CephContext* cct,
            std::shared_ptr<std::atomic<bool>> observation_gate = {});
  void register_commands(AdminSocket* socket, AdminSocketHook* hook);
  bool handle_command(std::string_view prefix, const cmdmap_t& cmdmap,
                      ceph::Formatter* formatter);
  // Called by the host storage observer with the actual stored object identity.
  void observe(const hobject_t& object, HpAccessType op, uint64_t effective_length);
  // Terminal, idempotent. Host must first drain callers and unregister commands.
  void shutdown();
};
