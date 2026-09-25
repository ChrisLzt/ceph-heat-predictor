// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bluestore_cache {

struct OnodeShardDemand {
  uint64_t resident;
  bool s3fifo;
};

inline std::vector<uint64_t> onode_quotas(
    uint64_t budget, const std::vector<OnodeShardDemand>& shards,
    bool rebalance)
{
  if (shards.empty()) {
    return {};
  }
  const uint64_t fair = budget / shards.size();
  std::vector<uint64_t> quotas(shards.size(), fair);
  if (!rebalance || !fair ||
      !std::all_of(shards.begin(), shards.end(),
                   [](const auto& shard) { return shard.s3fifo; })) {
    return quotas;
  }

  // Donate only unused fair-share capacity. Keep a small admission reserve
  // so an idle shard can grow again when the workload moves to it.
  const uint64_t reserve = std::max<uint64_t>(1, fair / 16);
  uint64_t available = budget;
  std::vector<std::size_t> recipients;
  for (std::size_t i = 0; i < shards.size(); ++i) {
    if (shards[i].resident >= fair - reserve) {
      recipients.push_back(i);
    } else {
      quotas[i] = shards[i].resident + reserve;
    }
    available -= quotas[i];
  }
  if (!recipients.empty()) {
    const uint64_t extra = available / recipients.size();
    uint64_t remainder = available % recipients.size();
    for (auto i : recipients) {
      quotas[i] += extra + (remainder ? 1 : 0);
      if (remainder) {
        --remainder;
      }
    }
  }
  return quotas;
}

} // namespace bluestore_cache
