// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "BlueStore.h"

#include <cmath>

#define dout_context cct
#define dout_subsys ceph_subsys_bluestore

using std::string;

// LruOnodeCacheShard
struct LruOnodeCacheShard : public BlueStore::OnodeCacheShard {
  typedef boost::intrusive::list<
    BlueStore::Onode,
    boost::intrusive::member_hook<
      BlueStore::Onode,
      boost::intrusive::list_member_hook<>,
      &BlueStore::Onode::lru_item> > list_t;

  list_t lru;

  explicit LruOnodeCacheShard(CephContext *cct) : BlueStore::OnodeCacheShard(cct) {}

  void _add(BlueStore::Onode* o, int level) override
  {
    o->set_cached();
    if (o->pin_nref == 1) {
      (level > 0) ? lru.push_front(*o) : lru.push_back(*o);
      o->cache_age_bin = age_bins.front();
      *(o->cache_age_bin) += 1;
    }
    ++num; // we count both pinned and unpinned entries
    dout(20) << __func__ << " " << this << " " << o->oid << " added, num="
             << num << dendl;
  }
  void _rm(BlueStore::Onode* o) override
  {
    o->clear_cached();
    if (o->lru_item.is_linked()) {
      *(o->cache_age_bin) -= 1;
      lru.erase(lru.iterator_to(*o));
    }
    ceph_assert(num);
    --num;
    dout(20) << __func__ << " " << this << " " << " " << o->oid << " removed, num=" << num << dendl;
  }

  void _maybe_unpin(BlueStore::Onode* o) override
  {
    if (o->is_cached() && o->pin_nref == 1) {
      if(!o->lru_item.is_linked()) {
        if (o->exists) {
	  lru.push_front(*o);
	  o->cache_age_bin = age_bins.front();
	  *(o->cache_age_bin) += 1;
	  dout(20) << __func__ << " " << this << " " << o->oid << " unpinned"
                   << dendl;
        } else {
	  ceph_assert(num);
	  --num;
	  o->clear_cached();
	  dout(20) << __func__ << " " << this << " " << o->oid << " removed"
                   << dendl;
          // remove will also decrement nref
          o->c->onode_space._remove(o->oid);
        }
      } else if (o->exists) {
        // move onode within LRU
        lru.erase(lru.iterator_to(*o));
        lru.push_front(*o);
        if (o->cache_age_bin != age_bins.front()) {
          *(o->cache_age_bin) -= 1;
          o->cache_age_bin = age_bins.front();
          *(o->cache_age_bin) += 1;
        }
        dout(20) << __func__ << " " << this << " " << o->oid << " touched"
                 << dendl;
      }
    }
  }

  void _trim_to(uint64_t new_size) override
  {
    if (new_size >= lru.size()) {
      return; // don't even try
    }
    uint64_t n = num - new_size; // note: we might get empty LRU
                                 // before n == 0 due to pinned
                                 // entries. And hence being unable
                                 // to reach new_size target.
    while (n-- > 0 && lru.size() > 0) {
      BlueStore::Onode *o = &lru.back();
      lru.pop_back();

      dout(20) << __func__ << "  rm " << o->oid << " "
               << o->nref << " " << o->cached << dendl;

      *(o->cache_age_bin) -= 1;
      if (o->pin_nref > 1) {
        dout(20) << __func__ << " " << this << " " << " " << " " << o->oid << dendl;
      } else {
	ceph_assert(num);
        --num;
        o->clear_cached();
        o->c->onode_space._remove(o->oid);
      }
    }
  }
  void _move_pinned(OnodeCacheShard *to, BlueStore::Onode *o) override
  {
    if (to == this) {
      return;
    }
    _rm(o);
    ceph_assert(o->nref > 1);
    to->_add(o, 0);
  }
  void add_stats(uint64_t *onodes, uint64_t *pinned_onodes) override
  {
    std::lock_guard l(lock);
    *onodes += num;
    *pinned_onodes += num - lru.size();
  }
};

// SwitchableOnodeCacheShard
//
// Implements the S3FIFO cache eviction algorithm (SOSP'23 paper:
// "FIFO Queues are All You Need for Cache Eviction" by Juncheng Yang et al.)
// adapted for BlueStore's Onode cache.
//
// Architecture:
//   - small_q  (Small FIFO): ~10% of cache entries, admits new onodes
//   - main_q   (Main FIFO):  ~90% of cache entries, uses 2-bit Clock algorithm
//   - ghost_q  (Ghost FIFO): stores oids of recently evicted onodes (metadata only)
//
// Algorithm flow:
//   INSERT:  ghost hit → main head;  miss → small head
//   ACCESS:  freq++ (no position change — FIFO property)
//   EVICT:   main over capacity or small empty → evict_main (2-bit Clock)
//            otherwise → evict_small
//   PROMOTE: small eviction with freq >= threshold → move to main
//   DEMOTE:  small eviction with freq < threshold → evict + insert into ghost
//
// The shard address, counts and age bins stay stable across policy changes.
struct SwitchableOnodeCacheShard : public LruOnodeCacheShard {
  typedef boost::intrusive::list<
    BlueStore::Onode,
    boost::intrusive::member_hook<
      BlueStore::Onode,
      boost::intrusive::list_member_hook<>,
      &BlueStore::Onode::lru_item> > list_t;

  // S3FIFO queues (all reuse Onode::lru_item hook — onode is in exactly one
  // of small_q or main_q at any time, or in neither when pinned)
  list_t small_q;
  list_t main_q;
  Policy policy;

  // Ghost FIFO: circular buffer storing only ghobject_t keys
  boost::circular_buffer<ghobject_t> ghost_q;

  // Configurable parameters (read from ceph config)
  const double small_ratio;
  const double ghost_capacity_ratio;
  const uint8_t promotion_threshold;

  SwitchableOnodeCacheShard(CephContext *cct, Policy initial_policy)
    : LruOnodeCacheShard(cct),
      policy(initial_policy),
      ghost_q(1),  // initial minimal capacity, resized in _trim_to
      small_ratio(cct->_conf.get_val<double>(
          "bluestore_cache_s3fifo_small_ratio")),
      ghost_capacity_ratio(cct->_conf.get_val<double>(
          "bluestore_cache_s3fifo_ghost_ratio")),
      promotion_threshold(static_cast<uint8_t>(
          cct->_conf.get_val<uint64_t>(
              "bluestore_cache_s3fifo_promotion_threshold")))
  {
    ceph_assert(_supports_policy(initial_policy));
  }

  bool _supports_policy(Policy next) const override {
    return next == Policy::LRU ||
      (small_ratio > 0.0 && small_ratio < 1.0 &&
       std::isfinite(ghost_capacity_ratio) && ghost_capacity_ratio > 0.0 &&
       promotion_threshold >= 1);
  }

  void _set_policy(Policy next) override {
    if (policy == next) {
      return;
    }
    ghost_q.clear();
    if (next == Policy::S3FIFO) {
      ceph_assert(small_q.empty() && main_q.empty());
      small_q.splice(small_q.end(), lru);
      for (auto& o : small_q) {
        o.s3fifo_queue = BlueStore::Onode::Q_SMALL;
        o.s3fifo_freq = 0;
      }
    } else {
      ceph_assert(lru.empty());
      // S3FIFO has no global recency order. Keep each queue's FIFO order,
      // placing its main (reused) entries ahead of probationary entries.
      lru.splice(lru.end(), main_q);
      lru.splice(lru.end(), small_q);
      for (auto& o : lru) {
        o.s3fifo_queue = BlueStore::Onode::Q_NONE;
        o.s3fifo_freq = 0;
      }
    }
    policy = next;
  }

  Policy _get_policy() const override {
    return policy;
  }

  void _dump_policy(ceph::Formatter* f) const override {
    f->dump_string("effective_policy", policy == Policy::S3FIFO ? "s3fifo" : "lru");
    f->dump_unsigned("resident_onodes", num.load());
    f->dump_unsigned("target_onodes", max.load());
    f->dump_unsigned("unlinked_onodes", num - lru.size() - small_q.size() - main_q.size());
    f->dump_unsigned("lru_entries", lru.size());
    f->dump_unsigned("small_entries", small_q.size());
    f->dump_unsigned("main_entries", main_q.size());
    f->dump_unsigned("ghost_entries", ghost_q.size());
  }

  // =========================================================================
  // Ghost queue helpers
  // =========================================================================

  /// Insert an oid into the ghost FIFO.  If full, the oldest entry is
  /// automatically overwritten (circular_buffer push_back on full buffer).
  void _ghost_insert(const ghobject_t& oid) {
    ghost_q.push_back(oid);
  }

  /// Look up and remove an oid from the ghost FIFO.
  /// Returns true if found (and removed), false otherwise.
  bool _ghost_find_and_remove(const ghobject_t& oid) {
    for (auto it = ghost_q.begin(); it != ghost_q.end(); ++it) {
      if (*it == oid) {
        ghost_q.erase(it);
        return true;
      }
    }
    return false;
  }

  // =========================================================================
  // S3FIFO eviction helpers
  // =========================================================================

  /// Evict one entry from the small FIFO.
  /// If the candidate's frequency >= promotion_threshold, promote it to main;
  /// otherwise truly evict it and record its oid in the ghost queue.
  /// Caller must hold lock.
  void _evict_small() {
    // Take the oldest entry from small (tail = least recently inserted)
    BlueStore::Onode *candidate = &small_q.back();
    small_q.pop_back();
    // pop_back resets lru_item, so candidate is no longer linked

    if (candidate->s3fifo_freq >= promotion_threshold) {
      // PROMOTION: move to main FIFO
      candidate->s3fifo_freq = 0;        // reset freq for the main clock
      candidate->s3fifo_queue = BlueStore::Onode::Q_MAIN;
      main_q.push_front(*candidate);
    } else {
      // TRUE EVICTION
      candidate->s3fifo_queue = BlueStore::Onode::Q_NONE;
      *(candidate->cache_age_bin) -= 1;

      _ghost_insert(candidate->oid);

      if (candidate->pin_nref > 1) {
        dout(20) << __func__ << " " << this << " skipping pinned " << candidate->oid
                << dendl;
        return;
      }

      ceph_assert(num);
      --num;
      candidate->clear_cached();
      candidate->c->onode_space._remove(candidate->oid);
    }
  }

  /// Evict one entry from the main FIFO (2-bit Clock).
  /// If the candidate's frequency >= 1, decrement it and reinsert at head
  /// (giving it another chance).  If frequency == 0, truly evict.
  /// Caller must hold lock.
  void _evict_main() {
    BlueStore::Onode *candidate = &main_q.back();
    main_q.pop_back();
    // pop_back resets lru_item, so candidate is no longer linked

    if (candidate->s3fifo_freq >= 1) {
      // 2-bit Clock REINSERTION: decrement counter, move to head
      // Original S3FIFO: freq = MIN(freq, 3) - 1
      uint8_t old_freq = candidate->s3fifo_freq;
      candidate->s3fifo_freq = (old_freq < 3) ? (old_freq - 1) : 2;
      main_q.push_front(*candidate);
      // stays in main, queue unchanged
    } else {
      // TRUE EVICTION: freq == 0
      candidate->s3fifo_queue = BlueStore::Onode::Q_NONE;
      *(candidate->cache_age_bin) -= 1;

      if (candidate->pin_nref > 1) {
        dout(20) << __func__ << " " << this << " skipping pinned " << candidate->oid
                << dendl;
        return;
      }

      ceph_assert(num);
      --num;
      candidate->clear_cached();
      candidate->c->onode_space._remove(candidate->oid);
    }
  }


  // =========================================================================
  // OnodeCacheShard interface implementation
  // =========================================================================

  void _add(BlueStore::Onode* o, int level) override {
    if (policy == Policy::LRU) {
      o->s3fifo_queue = BlueStore::Onode::Q_NONE;
      o->s3fifo_freq = 0;
      return LruOnodeCacheShard::_add(o, level);
    }
    o->set_cached();

    if (o->pin_nref == 1) {
      // Check ghost queue: a ghost hit means this onode was recently evicted
      // and should skip the small queue, going directly to main.
      if (_ghost_find_and_remove(o->oid)) {
        o->s3fifo_queue = BlueStore::Onode::Q_MAIN;
        o->s3fifo_freq = 0;
        main_q.push_front(*o);
      } else {
        o->s3fifo_queue = BlueStore::Onode::Q_SMALL;
        o->s3fifo_freq = 0;
        (level > 0) ? small_q.push_front(*o) : small_q.push_back(*o);
      }
      o->cache_age_bin = age_bins.front();
      *(o->cache_age_bin) += 1;
    } else {
      // Pinned: counted in num but not placed in any list
      o->s3fifo_queue = BlueStore::Onode::Q_NONE;
      o->s3fifo_freq = 0;
    }

    ++num; // count both pinned and unpinned entries
    dout(20) << __func__ << " " << this << " " << o->oid << " added, num="
             << num << dendl;
  }

  void _rm(BlueStore::Onode* o) override {
    if (policy == Policy::LRU) {
      return LruOnodeCacheShard::_rm(o);
    }
    o->clear_cached();
    if (o->lru_item.is_linked()) {
      *(o->cache_age_bin) -= 1;
      switch (o->s3fifo_queue) {
      case BlueStore::Onode::Q_SMALL:
        small_q.erase(small_q.iterator_to(*o));
        break;
      case BlueStore::Onode::Q_MAIN:
        main_q.erase(main_q.iterator_to(*o));
        break;
      default:
        ceph_abort_msg("S3FIFO: _rm called on onode with unexpected queue");
      }
      o->s3fifo_queue = BlueStore::Onode::Q_NONE;
    }
    ceph_assert(num);
    --num;
    dout(20) << __func__ << " " << this << " " << o->oid << " removed, num="
             << num << dendl;
  }

  void _touch(BlueStore::Onode* o) override {
    // Count reuse while the entry is in an eviction queue, not when the
    // last reference is released. Overlapping lookups must count separately.
    // Unlinked entries still start their admission history when unpinned.
    if (policy == Policy::S3FIFO && o->lru_item.is_linked() &&
        o->s3fifo_freq < 3) {
      ++o->s3fifo_freq;
    }
  }

  void _maybe_unpin(BlueStore::Onode* o) override {
    if (policy == Policy::LRU) {
      return LruOnodeCacheShard::_maybe_unpin(o);
    }
    if (o->is_cached() && o->pin_nref == 1) {
      if (!o->lru_item.is_linked()) {
        // Previously pinned, not in any list — now rejoin
        if (o->exists) {
          // Check ghost: if ghost hit, go directly to main
          if (_ghost_find_and_remove(o->oid)) {
            o->s3fifo_queue = BlueStore::Onode::Q_MAIN;
            o->s3fifo_freq = 0;
            main_q.push_front(*o);
          } else {
            o->s3fifo_queue = BlueStore::Onode::Q_SMALL;
            o->s3fifo_freq = 0;
            small_q.push_front(*o);
          }
          o->cache_age_bin = age_bins.front();
          *(o->cache_age_bin) += 1;
          dout(20) << __func__ << " " << this << " " << o->oid
                   << " unpinned" << dendl;
        } else {
          // Onode no longer exists — remove from cache
          ceph_assert(num);
          --num;
          o->clear_cached();
          dout(20) << __func__ << " " << this << " " << o->oid
                   << " removed" << dendl;
          o->c->onode_space._remove(o->oid);
        }
      } else {
        // Lookup already recorded reuse. Reference release only updates age;
        // it must not count as another hit or change FIFO position.
        if (o->cache_age_bin != age_bins.front()) {
          *(o->cache_age_bin) -= 1;
          o->cache_age_bin = age_bins.front();
          *(o->cache_age_bin) += 1;
        }
        dout(20) << __func__ << " " << this << " " << o->oid
                 << " freq " << static_cast<int>(o->s3fifo_freq)
                 << dendl;
      }
    }
  }

  void _trim_to(uint64_t new_size) override {
    if (policy == Policy::LRU) {
      return LruOnodeCacheShard::_trim_to(new_size);
    }
    // Dynamically resize ghost queue based on current target capacity
    uint64_t ghost_cap = static_cast<uint64_t>(new_size * ghost_capacity_ratio);
    if (ghost_cap < 1) ghost_cap = 1;
    // push_back appends the newest eviction. When autotuning shrinks the
    // budget, discard the oldest ghosts, as normal FIFO insertion does.
    ghost_q.rset_capacity(ghost_cap);

    uint64_t total = small_q.size() + main_q.size();
    if (new_size >= total) {
      return; // nothing to evict — pinned entries keep num > lru size, same as LRU
    }

    // Compute target sizes for small and main FIFOs.
    // Guard against underflow when new_size is very small (e.g., 0 for flush).
    uint64_t target_small = static_cast<uint64_t>(new_size * small_ratio);
    if (target_small < 1 && new_size > 0) target_small = 1;
    uint64_t target_main = (new_size > target_small) ? (new_size - target_small) : 0;

    // Safety bound: unlike LRU where every iteration reduces list size,
    // S3FIFO promotions (small→main) and 2-bit Clock reinsertions keep
    // total constant.  Each entry in main may need up to 4 operations
    // (3 reinsertions via 2-bit Clock + 1 true eviction).  Plus, entries
    // pinned via lazy removal consume iterations without reducing total.
    // We use a generous bound to allow the 2-bit Clock to converge while
    // still preventing infinite loops.
    uint64_t safety = total * 4 + num - new_size;

    while (total > new_size && total > 0 && safety > 0) {
      // Original S3FIFO eviction policy:
      // if main is over capacity OR small is empty → evict from main
      // otherwise → evict from small
      if (main_q.size() > 0 &&
          (main_q.size() > target_main || small_q.empty())) {
        _evict_main();
      } else if (small_q.size() > 0) {
        _evict_small();
      } else {
        break; // no more evictable entries (both queues empty)
      }
      safety--;
      total = small_q.size() + main_q.size();
    }
  }

  PrefetchReclaimResult _reclaim_for_prefetch(uint64_t max_steps) override {
    PrefetchReclaimResult result;
    if (policy != Policy::S3FIFO || cct->_conf->objectstore_blackhole) {
      return result;
    }
    const auto before = num.load();
    const auto capacity = max.load();
    const auto target_main = capacity - static_cast<uint64_t>(capacity * small_ratio);
    // Reuse normal eviction and pin handling, but bound clock rotations and
    // stop after one eviction so the worker can recheck actual metadata bytes.
    while (result.examined < max_steps && (!small_q.empty() || !main_q.empty())) {
      ++result.examined;
      if (!small_q.empty() && small_q.back().prefetched) {
        _evict_small();
      } else if (!main_q.empty() &&
                 (main_q.size() > target_main || small_q.empty())) {
        _evict_main();
      } else {
        _evict_small();
      }
      if (num.load() < before) {
        result.evicted = before - num.load();
        break;
      }
    }
    return result;
  }

  void _move_pinned(OnodeCacheShard *to, BlueStore::Onode *o) override {
    if (to == this) {
      return;
    }
    _rm(o);
    ceph_assert(o->nref > 1);
    to->_add(o, 0);
  }

  void add_stats(uint64_t *onodes, uint64_t *pinned_onodes) override {
    std::lock_guard l(lock);
    *onodes += num;
    *pinned_onodes += num - (lru.size() + small_q.size() + main_q.size());
  }
};

// OnodeCacheShard
BlueStore::OnodeCacheShard *BlueStore::OnodeCacheShard::create(
    CephContext* cct,
    string type,
    PerfCounters *logger)
{
  auto *c = new SwitchableOnodeCacheShard(
    cct, type == "s3fifo" ? Policy::S3FIFO : Policy::LRU);
  c->logger = logger;
  return c;
}

void BlueStore::OnodeCacheShard::maybe_unpin(Onode* o)
{
  auto* shard = this;
  std::unique_lock l(shard->lock);
  // split_cache may have moved the onode while we waited. Dispatch on the
  // shard whose lock we hold, not the original shard or its former policy.
  while (shard != o->c->get_onode_cache()) {
    l.unlock();
    shard = o->c->get_onode_cache();
    l = std::unique_lock(shard->lock);
  }
  shard->_maybe_unpin(o);
}
