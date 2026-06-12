// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
#pragma once

#include "include/common_fwd.h"

struct ceph_osd_op;
struct hobject_t;

void init_osd_object_hp_status(CephContext *cct);
void hp_notify_osd_object_op(CephContext *cct,
                             const hobject_t& soid,
                             const ceph_osd_op& op);
