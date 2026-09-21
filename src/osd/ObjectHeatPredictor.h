// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
#pragma once

#include <cstdint>
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

  void init(CephContext* cct);
  void register_commands(AdminSocket* socket, AdminSocketHook* hook);
  bool handle_command(std::string_view prefix, const cmdmap_t& cmdmap,
                      ceph::Formatter* formatter);
  // Call only at the existing validated/normalized PG observation positions.
  void observe(const hobject_t& object, uint16_t op, uint64_t effective_length);
  // Terminal, idempotent. Host must first drain callers and unregister commands.
  void shutdown();
};
