// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "OnodeCache.h"
#include "OnodeCacheBudget.h"
#include "common/Clock.h"

#define dout_context cct
#define dout_subsys ceph_subsys_bluestore
#undef dout_prefix
#define dout_prefix *_dout << "bluestore(" << store->path << ") "

using ceph::Formatter;

BlueStore::OnodeCache::OnodeCache(BlueStore* store)
  : store(store), cct(store->cct), prefetch(store)
{
  cache_instance.generate_random();
}

void BlueStore::OnodeCache::set_shards(unsigned num)
{
  std::lock_guard policy_guard(policy_lock);
  dout(10) << __func__ << " " << num << dendl;
  size_t oold = store->onode_cache_shards.size();
  size_t bold = store->buffer_cache_shards.size();
  ceph_assert(num >= oold && num >= bold);
  std::string onode_policy = cct->_conf->bluestore_cache_type;
  if (oold) {
    std::lock_guard l(store->onode_cache_shards.front()->lock);
    onode_policy = store->onode_cache_shards.front()->_get_policy() ==
      OnodeCacheShard::Policy::S3FIFO ? "s3fifo" : "lru";
  }
  if (!bold) {
    buffer_cache_policy = cct->_conf->bluestore_cache_type;
    if (buffer_cache_policy == "s3fifo") {
      buffer_cache_policy = "2q";
    }
  }
  store->onode_cache_shards.resize(num);
  store->buffer_cache_shards.resize(num);
  for (unsigned i = oold; i < num; ++i) {
    store->onode_cache_shards[i] =
        OnodeCacheShard::create(cct, onode_policy,
                                 store->logger);
  }
  for (unsigned i = bold; i < num; ++i) {
    store->buffer_cache_shards[i] =
        BufferCacheShard::create(cct, buffer_cache_policy,
                                 store->logger);
  }
}

int BlueStore::OnodeCache::set_policy(const std::string& policy, Formatter* f)
{
  if (policy != "lru" && policy != "s3fifo") {
    return -EINVAL;
  }
  std::lock_guard policy_guard(policy_lock);
  if (store->onode_cache_shards.empty()) {
    return -EAGAIN;
  }
  const auto next = policy == "s3fifo" ? OnodeCacheShard::Policy::S3FIFO :
    OnodeCacheShard::Policy::LRU;
  // Validate every shard before changing any of them. Tunables are immutable
  // for a shard's lifetime, including while its LRU policy is active.
  for (auto* shard : store->onode_cache_shards) {
    std::lock_guard l(shard->lock);
    if (!shard->_supports_policy(next)) {
      return -EINVAL;
    }
  }
  bool changed = false;
  const auto started = ceph_clock_now().to_nsec();
  if (next == OnodeCacheShard::Policy::LRU) {
    prefetch.set_active(false);
  }
  for (auto* shard : store->onode_cache_shards) {
    std::lock_guard l(shard->lock);
    if (shard->_get_policy() != next) {
      shard->_set_policy(next);
      changed = true;
    }
  }
  if (changed) {
    ++policy_generation;
    switch_started_ns = started;
    switch_completed_ns = ceph_clock_now().to_nsec();
    dout(1) << __func__ << " effective_policy=" << policy
            << " generation=" << policy_generation << dendl;
  }
  if (next == OnodeCacheShard::Policy::S3FIFO) {
    prefetch.set_active(true);
  }
  dump_policy(f);
  return 0;
}

int BlueStore::OnodeCache::get_policy(Formatter* f)
{
  std::lock_guard policy_guard(policy_lock);
  if (store->onode_cache_shards.empty()) {
    return -EAGAIN;
  }
  dump_policy(f);
  return 0;
}

void BlueStore::OnodeCache::dump_policy(Formatter* f)
{
  f->open_object_section("onode_cache");
  f->dump_stream("cache_instance") << cache_instance;
  f->dump_unsigned("policy_generation", policy_generation);
  f->dump_bool("runtime_only", true);
  f->dump_bool("s3fifo_shard_rebalance_enabled",
               cct->_conf.get_val<bool>("bluestore_cache_s3fifo_rebalance_shards"));
  f->dump_string("buffer_cache_policy", buffer_cache_policy);
  f->dump_unsigned("last_switch_started_ns", switch_started_ns);
  f->dump_unsigned("last_switch_completed_ns", switch_completed_ns);
  std::string effective_policy;
  uint64_t prefetch_loaded = 0, prefetch_used = 0, prefetch_unused = 0;
  f->open_array_section("shards");
  for (size_t i = 0; i < store->onode_cache_shards.size(); ++i) {
    auto* shard = store->onode_cache_shards[i];
    std::lock_guard l(shard->lock);
    const std::string current = shard->_get_policy() ==
      OnodeCacheShard::Policy::S3FIFO ? "s3fifo" : "lru";
    if (effective_policy.empty()) {
      effective_policy = current;
    } else if (effective_policy != current) {
      effective_policy = "mixed";
    }
    f->open_object_section("shard");
    f->dump_unsigned("id", i);
    shard->_dump_policy(f);
    f->dump_unsigned("prefetch_loaded", shard->prefetch_loaded);
    f->dump_unsigned("prefetch_used", shard->prefetch_used);
    f->dump_unsigned("prefetch_unused", shard->prefetch_unused);
    prefetch_loaded += shard->prefetch_loaded;
    prefetch_used += shard->prefetch_used;
    prefetch_unused += shard->prefetch_unused;
    f->close_section();
  }
  f->close_section();
  f->dump_string("effective_policy", effective_policy);
  f->dump_unsigned("sample_time_ns", ceph_clock_now().to_nsec());
  // Process-lifetime counters, not a percentage: consumers use window deltas.
  f->dump_unsigned("onode_hits", store->logger->get(l_bluestore_onode_hits));
  f->dump_unsigned("onode_misses", store->logger->get(l_bluestore_onode_misses));
  f->open_object_section("prefetch");
  prefetch.dump(f);
  f->dump_unsigned("loaded", prefetch_loaded);
  f->dump_unsigned("used", prefetch_used);
  f->dump_unsigned("unused_removed", prefetch_unused);
  f->close_section();
  f->close_section();
}

void BlueStore::OnodeCache::set_shard_quotas(uint64_t max_shard_onodes)
{
  const auto onode_shards = store->onode_cache_shards.size();
  if (store->cct->_conf.get_val<bool>("bluestore_cache_s3fifo_rebalance_shards")) {
    // Serialize snapshots and target updates with online policy changes.
    // Reuse the same total entry budget; never borrow from the buffer cache.
    std::lock_guard policy_guard(policy_lock);
    std::vector<bluestore_cache::OnodeShardDemand> demand;
    demand.reserve(onode_shards);
    for (auto* shard : store->onode_cache_shards) {
      std::lock_guard l(shard->lock);
      demand.push_back({shard->_get_num(),
                       shard->_get_policy() == OnodeCacheShard::Policy::S3FIFO});
    }
    auto quotas = bluestore_cache::onode_quotas(
        max_shard_onodes * onode_shards, demand, true);
    for (size_t i = 0; i < onode_shards; ++i) {
      store->onode_cache_shards[i]->set_max(quotas[i]);
    }
  } else {
    for (auto* shard : store->onode_cache_shards) {
      shard->set_max(max_shard_onodes);
    }
  }
}
