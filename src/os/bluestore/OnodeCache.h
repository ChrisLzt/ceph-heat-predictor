// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
#pragma once

#include "BlueStore.h"

// BlueStore-specific adapter. Keep policy/worker state out of BlueStore.h;
// native Onode ownership, collection locking and disk format remain unchanged.
struct BlueStore::OnodeCache {
  explicit OnodeCache(BlueStore* store);

  void init() { prefetch.init(); }
  void shutdown() { prefetch.shutdown(); }
  void schedule(Collection* collection) { prefetch.schedule(collection); }
  void set_meta_budget(uint64_t bytes) { prefetch.meta_limit.store(bytes); }
  void set_shard_quotas(uint64_t max_shard_onodes);
  void set_shards(unsigned num);
  int set_policy(const std::string& policy, ceph::Formatter* f);
  int get_policy(ceph::Formatter* f);

private:
  friend struct OnodePrefetchTestPeer;
  BlueStore* const store;
  CephContext* const cct;
  // Serializes administrative policy changes; never taken on the I/O path.
  ceph::mutex policy_lock =
    ceph::make_mutex("BlueStore::onode_cache_policy_lock");
  uint64_t policy_generation = 0;
  uuid_d cache_instance;
  uint64_t switch_started_ns = 0;
  uint64_t switch_completed_ns = 0;
  std::string buffer_cache_policy;
  void dump_policy(ceph::Formatter* f);

  // Implemented beside the native key encoder in BlueStore.cc, so the worker
  // does not duplicate or expose BlueStore's on-disk key format.
  int read_onode_record(const ghobject_t& oid, std::string* key,
                        ceph::bufferlist* value) const;

  struct PrefetchThread : public Thread {
    struct Work {
      CollectionRef collection;
      ghobject_t next;
      uint64_t generation;
      uint32_t bits;
    };
    BlueStore* store;
    ceph::mutex lock = ceph::make_mutex("BlueStore::OnodePrefetchThread");
    ceph::condition_variable cond;
    std::deque<Work> queue;
    bool stop = false;
    bool configured = false;
    bool reclaim_enabled = false;
    uint64_t rate = 1024;
    uint64_t max_queued = 256;
    uint64_t max_record = 1048576;
    uint64_t next_generation = 0;
    std::atomic<uint64_t> generation{0};
    std::atomic<uint64_t> meta_limit{0};
    std::atomic<uint64_t> scanned{0}, reads{0}, encoded_bytes{0};
    std::atomic<uint64_t> errors{0}, oversized{0}, queue_full{0};
    std::atomic<uint64_t> pressure_pauses{0};
    std::atomic<uint64_t> shard_pressure_pauses{0}, resident_skips{0};
    std::atomic<uint64_t> lock_retries{0}, candidate_retries{0};
    std::atomic<uint64_t> reclaim_passes{0}, reclaim_examined{0};
    std::atomic<uint64_t> reclaim_evicted{0}, reclaim_no_progress{0};
    std::atomic<uint64_t> reclaim_lock_skips{0};
    size_t reclaim_shard = 0;
    std::chrono::steady_clock::time_point next_reclaim{};

    explicit PrefetchThread(BlueStore* s) : store(s) {}
    void init();
    void shutdown();
    void set_active(bool active); // called under OnodeCache::policy_lock
    void schedule(Collection* collection);
    bool memory_available() const;
    void reclaim_space(uint64_t epoch, OnodeCacheShard* target = nullptr);
    void dump(ceph::Formatter* f);
    void* entry() override;
  } prefetch;
};
