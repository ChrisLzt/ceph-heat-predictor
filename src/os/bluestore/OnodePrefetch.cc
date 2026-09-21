// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "OnodeCache.h"

using ceph::bufferlist;
using ceph::Formatter;

void BlueStore::OnodeCache::PrefetchThread::init()
{
  std::lock_guard policy_guard(store->onode_cache->policy_lock);
  configured = store->cct->_conf.get_val<bool>("bluestore_onode_prefetch");
  reclaim_enabled = store->cct->_conf.get_val<bool>("bluestore_onode_prefetch_reclaim");
  if (!configured) {
    return;
  }
  rate = store->cct->_conf.get_val<uint64_t>("bluestore_onode_prefetch_rate");
  max_queued = store->cct->_conf.get_val<uint64_t>("bluestore_onode_prefetch_max_queued");
  max_record = store->cct->_conf.get_val<Option::size_t>("bluestore_onode_prefetch_max_record");
  stop = false;
  bool active = !store->onode_cache_shards.empty();
  for (auto* shard : store->onode_cache_shards) {
    std::lock_guard l(shard->lock);
    active = active && shard->_get_policy() == OnodeCacheShard::Policy::S3FIFO;
  }
  set_active(active);
  create("bstore_prefetch");
}

void BlueStore::OnodeCache::PrefetchThread::set_active(bool active)
{
  std::lock_guard l(lock);
  if (!active || !configured) {
    generation.store(0);
    queue.clear();
  } else if (!generation.load()) {
    generation.store(++next_generation);
  }
  cond.notify_all();
}

void BlueStore::OnodeCache::PrefetchThread::shutdown()
{
  if (!configured) {
    return;
  }
  {
    std::lock_guard policy_guard(store->onode_cache->policy_lock);
    set_active(false);
    std::lock_guard l(lock);
    stop = true;
    cond.notify_all();
  }
  join();
}

void BlueStore::OnodeCache::PrefetchThread::schedule(Collection* collection)
{
  const auto epoch = generation.load();
  if (!epoch || collection->prefetch_generation.load() == epoch ||
      !collection->cid.is_pg() || !collection->exists) {
    return;
  }
  // Called with the collection lock held. Never wait for the work queue.
  std::unique_lock l(lock, std::try_to_lock);
  if (!l.owns_lock() || generation.load() != epoch ||
      collection->prefetch_generation.load() == epoch) {
    return;
  }
  if (queue.size() >= max_queued) {
    ++queue_full;
    return;
  }
  queue.push_back({CollectionRef(collection), ghobject_t(), epoch,
                   collection->cnode.bits});
  collection->prefetch_generation.store(epoch);
  cond.notify_one();
}

bool BlueStore::OnodeCache::PrefetchThread::memory_available() const
{
  const auto limit = meta_limit.load();
  return limit && store->mempool_thread.meta_cache->_get_used_bytes() <
    limit - limit / 5;
}

void BlueStore::OnodeCache::PrefetchThread::reclaim_space(
  uint64_t epoch, OnodeCacheShard* target)
{
  const auto now = std::chrono::steady_clock::now();
  if (!reclaim_enabled || generation.load() != epoch || !meta_limit.load() ||
      now < next_reclaim || store->onode_cache_shards.empty()) {
    return;
  }
  next_reclaim = now + std::chrono::milliseconds(100);
  ++reclaim_passes;
  const auto used_before = store->mempool_thread.meta_cache->_get_used_bytes();
  const auto limit_before = meta_limit.load();
  const bool need_bytes = used_before >= limit_before - limit_before / 5;
  uint64_t evicted = 0;
  // At most 128 queue operations per pass, independent of the scan rate.
  // Leave a small hysteresis margin if this bounded pass can reach it.
  for (unsigned i = 0; i < 32 && generation.load() == epoch; ++i) {
    const auto limit = meta_limit.load();
    if (!limit) {
      break;
    }
    const bool byte_pressure = store->mempool_thread.meta_cache->_get_used_bytes() >=
      limit - limit / 4;
    if (!byte_pressure && !target) {
      break;
    }
    auto* shard = byte_pressure ? store->onode_cache_shards[
      reclaim_shard++ % store->onode_cache_shards.size()] : target;
    std::unique_lock l(shard->lock, std::try_to_lock);
    if (!l.owns_lock()) {
      ++reclaim_lock_skips;
      continue;
    }
    if (generation.load() != epoch) {
      break;
    }
    const auto quota = shard->max.load();
    if (!byte_pressure &&
        (!quota || shard->_get_num() < quota - quota / 10 - quota / 20)) {
      break;
    }
    auto result = shard->_reclaim_for_prefetch(4);
    reclaim_examined.fetch_add(result.examined);
    evicted += result.evicted;
    // Destruction can release extents/blobs as well as the Onode itself.
    // Do not estimate freed bytes as evicted_count * sizeof(Onode).
  }
  reclaim_evicted.fetch_add(evicted);
  if (!evicted || (need_bytes &&
      store->mempool_thread.meta_cache->_get_used_bytes() >= used_before)) {
    ++reclaim_no_progress;
    // Shared/pinned metadata or concurrent demand can defeat byte recovery.
    // Back off instead of continuously churning demand entries for no relief.
    next_reclaim = now + std::chrono::seconds(1);
  }
}

void BlueStore::OnodeCache::PrefetchThread::dump(Formatter* f)
{
  std::lock_guard l(lock);
  f->dump_bool("configured", configured);
  f->dump_bool("reclaim_enabled", reclaim_enabled);
  f->dump_bool("active", generation.load() != 0);
  f->dump_unsigned("generation", generation.load());
  f->dump_unsigned("queued_pgs", queue.size());
  f->dump_unsigned("rate", rate);
  f->dump_unsigned("meta_budget_bytes", meta_limit.load());
  f->dump_unsigned("meta_used_bytes", store->mempool_thread.meta_cache->_get_used_bytes());
  f->dump_unsigned("scanned", scanned.load());
  f->dump_unsigned("db_reads", reads.load());
  f->dump_unsigned("encoded_bytes", encoded_bytes.load());
  f->dump_unsigned("errors", errors.load());
  f->dump_unsigned("oversized", oversized.load());
  f->dump_unsigned("queue_full", queue_full.load());
  f->dump_unsigned("pressure_pauses", pressure_pauses.load());
  f->dump_unsigned("shard_pressure_pauses", shard_pressure_pauses.load());
  f->dump_unsigned("resident_skips", resident_skips.load());
  f->dump_unsigned("lock_retries", lock_retries.load());
  f->dump_unsigned("candidate_retries", candidate_retries.load());
  f->dump_unsigned("reclaim_passes", reclaim_passes.load());
  f->dump_unsigned("reclaim_examined", reclaim_examined.load());
  f->dump_unsigned("reclaim_evicted", reclaim_evicted.load());
  f->dump_unsigned("reclaim_no_progress", reclaim_no_progress.load());
  f->dump_unsigned("reclaim_lock_skips", reclaim_lock_skips.load());
}

void* BlueStore::OnodeCache::PrefetchThread::entry()
{
  constexpr int batch = 32;
  while (true) {
    Work work;
    {
      std::unique_lock l(lock);
      cond.wait(l, [&] { return stop || !queue.empty(); });
      if (stop) {
        return nullptr;
      }
      work = std::move(queue.front());
      queue.pop_front();
    }
    const auto started = std::chrono::steady_clock::now();
    auto c = work.collection;
    bool again = false;
    std::vector<ghobject_t> objects;
    ghobject_t next;
    if (generation.load() == work.generation) {
      // A split can invalidate the cursor. Do not pass it to collection_list.
      std::shared_lock l(c->lock, std::try_to_lock);
      if (!l.owns_lock()) {
        ++lock_retries;
        again = true;
      } else if (c->exists && c->cnode.bits == work.bits) {
        // Validate the queued collection before reclaiming for it. A deleted
        // or split PG must not keep driving cache eviction under pressure.
        if (!memory_available()) {
          ++pressure_pauses;
          reclaim_space(work.generation);
        }
        if (memory_available()) {
          // Raw-key order avoids the sorted iterator's unbounded collision chunk.
          int r = store->_collection_list(c.get(), work.next, ghobject_t::get_max(),
                                         batch, true, &objects, &next);
          if (r < 0) {
            ++errors;
          } else {
            work.next = next;
            again = !next.is_max();
          }
        } else {
          again = true;
        }
      } else {
        auto expected = work.generation;
        c->prefetch_generation.compare_exchange_strong(expected, 0);
      }
    }
    for (const auto& oid : objects) {
      ++scanned;
      // collection_list uses an inclusive cursor. A temporary rejection must
      // not advance past this object (or the rest of this bounded batch).
      auto retry = [&] {
        work.next = oid;
        again = true;
        ++candidate_retries;
      };
      std::shared_lock l(c->lock, std::try_to_lock);
      if (generation.load() != work.generation) {
        break;
      }
      if (!l.owns_lock()) {
        ++lock_retries;
        retry();
        break;
      }
      if (!c->exists || c->cnode.bits != work.bits) {
        auto expected = work.generation;
        c->prefetch_generation.compare_exchange_strong(expected, 0);
        again = false;
        break;
      }
      if (!c->contains(oid)) {
        continue;
      }
      using Admission = OnodeSpace::PrefetchAdmission;
      const auto admission = c->onode_space.prefetch_admission(oid);
      if (admission == Admission::resident) {
        ++resident_skips;
        continue;
      }
      if (admission == Admission::inactive) {
        retry();
        break;
      }
      if (!memory_available() || admission == Admission::full) {
        if (!memory_available()) {
          ++pressure_pauses;
        }
        if (admission == Admission::full) {
          ++shard_pressure_pauses;
        }
        reclaim_space(work.generation, c->get_onode_cache());
        retry();
        break;
      }
      std::string key;
      bufferlist value;
      ++reads;
      int r = store->onode_cache->read_onode_record(oid, &key, &value);
      if (r == -ENOENT) {
        continue;
      }
      if (r < 0 || !value.length()) {
        ++errors;
        continue;
      }
      encoded_bytes.fetch_add(value.length());
      if (value.length() > max_record) {
        ++oversized;
        continue;
      }
      // Decode the same metadata as demand I/O, under the same collection lock.
      // No extent faults, object data reads, or demand lookup accounting here.
      OnodeRef o(Onode::create_decode(c, oid, key, value));
      if (!memory_available()) {
        ++pressure_pauses;
        // Account the decoded candidate while reclaiming. Otherwise releasing
        // it before every retry could hide the headroom needed to admit it.
        reclaim_space(work.generation, c->get_onode_cache());
      }
      if (!memory_available() ||
          !c->onode_space.add_prefetched(o, &generation, work.generation)) {
        if (c->onode_space.prefetch_admission(oid) == Admission::resident) {
          ++resident_skips;
        } else {
          // Release the decoded candidate before the next pressure check.
          retry();
          break;
        }
      }
    }
    {
      std::unique_lock l(lock);
      if (again && generation.load() == work.generation) {
        if (queue.size() < max_queued) {
          queue.push_back(std::move(work));
        } else {
          ++queue_full;
          c->prefetch_generation.store(0);
        }
      }
      // Charge a full batch even for skipped candidates; no catch-up bursts.
      const auto delay = std::chrono::microseconds((batch * 1000000ULL + rate - 1) / rate);
      cond.wait_until(l, started + delay, [&] {
        return stop || generation.load() != work.generation;
      });
      if (stop) {
        return nullptr;
      }
    }
  }
}
