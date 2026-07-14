// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
#pragma once

#include <cstdint>

#include "include/common_fwd.h"

namespace ceph {
class Formatter;
}

struct hobject_t;

void init_osd_object_hp_status(CephContext *cct);
void hp_dump_osd_object_heat_predictor_status(CephContext *cct,
                                              ceph::Formatter *f);
void hp_reset_osd_object_heat_predictor(CephContext *cct, ceph::Formatter *f);
void hp_set_osd_object_heat_predictor_enabled(CephContext *cct,
                                              ceph::Formatter *f,
                                              bool enabled);
void hp_notify_osd_object_op(CephContext *cct,
                             const hobject_t& soid,
                             uint16_t op);
