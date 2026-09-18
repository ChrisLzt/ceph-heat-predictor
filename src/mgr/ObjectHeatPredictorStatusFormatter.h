#ifndef CEPH_MGR_OBJECT_HEAT_PREDICTOR_STATUS_FORMATTER_H
#define CEPH_MGR_OBJECT_HEAT_PREDICTOR_STATUS_FORMATTER_H

#include <iosfwd>

#include "ObjectHeatPredictorStatus.h"

namespace ceph {
class Formatter;
}

namespace ceph::mgr {

// A null formatter selects brief text, or the legacy pretty JSON in detail mode.
void format_object_hp_status(ObjectHpClusterStatus status,
                             ceph::Formatter* formatter,
                             std::ostream& out, bool detail);

} // namespace ceph::mgr

#endif
