// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "include/types.h"
#include "common/Formatter.h"
#include "common/ceph_context.h"
#include "common/config.h"
#include "common/perf_counters.h"
#include "global/global_context.h"
#include "json_spirit/json_spirit.h"
#include "os/bluestore/BlueStore.h"
#include "os/bluestore/OnodeCacheBudget.h"
#include "gtest/gtest.h"
#include "store_test_fixture.h"

#include <memory>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>

// A separate worker lets tests control its budget without racing the mounted
// store's autotuner or adding a production budget-override command.
struct OnodePrefetchTestPeer {
  BlueStore::OnodePrefetchThread worker;
  explicit OnodePrefetchTestPeer(BlueStore* store) : worker(store) {}
  ~OnodePrefetchTestPeer() { worker.shutdown(); }

  void start(uint64_t budget) {
    worker.init();
    worker.meta_limit.store(budget);
  }
  void schedule(ObjectStore::CollectionHandle& ch) {
    auto* c = static_cast<BlueStore::Collection*>(ch.get());
    std::shared_lock l(c->lock);
    worker.schedule(c);
  }
  uint64_t used_bytes() const {
    return worker.store->mempool_thread.meta_cache->_get_used_bytes();
  }
  auto pause_resizing() {
    return std::unique_lock(worker.store->mempool_thread.lock);
  }
  void set_shard_quotas(uint64_t quota) {
    for (auto* shard : worker.store->onode_cache_shards) {
      shard->set_max(quota);
    }
  }
  uint64_t loaded() const {
    uint64_t count = 0;
    for (auto* shard : worker.store->onode_cache_shards) {
      std::lock_guard l(shard->lock);
      count += shard->prefetch_loaded;
    }
    return count;
  }
};

namespace {
using Policy = BlueStore::OnodeCacheShard::Policy;

TEST(OnodeCacheBudget, DisabledOrMixedPoliciesKeepOriginalQuotas)
{
  using bluestore_cache::onode_quotas;
  EXPECT_EQ((std::vector<uint64_t>{100, 100}),
            onode_quotas(200, {{0, true}, {100, true}}, false));
  EXPECT_EQ((std::vector<uint64_t>{100, 100}),
            onode_quotas(200, {{0, false}, {100, true}}, true));
}

TEST(OnodeCacheBudget, BorrowIdleQuotasWithoutIncreasingBudget)
{
  EXPECT_EQ((std::vector<uint64_t>{6, 6, 194, 194}),
            bluestore_cache::onode_quotas(
                400, {{0, true}, {0, true}, {100, true}, {100, true}}, true));
  EXPECT_EQ((std::vector<uint64_t>{100, 100, 100, 100}),
            bluestore_cache::onode_quotas(
                400, {{100, true}, {100, true}, {100, true}, {100, true}}, true));
}

TEST(OnodeCacheBudget, UnderusedShardsRetainResidentsAndAdmissionReserve)
{
  auto quotas = bluestore_cache::onode_quotas(
      400, {{50, true}, {0, true}, {100, true}, {100, true}}, true);
  EXPECT_EQ((std::vector<uint64_t>{56, 6, 169, 169}), quotas);
  EXPECT_EQ(400u, std::accumulate(quotas.begin(), quotas.end(), uint64_t{0}));
  EXPECT_EQ((std::vector<uint64_t>{6, 6}),
            bluestore_cache::onode_quotas(200, {{0, true}, {0, true}}, true));
}

TEST(OnodeCacheBudget, ZeroSmallAndLargeBudgetsDoNotOverflow)
{
  using bluestore_cache::onode_quotas;
  EXPECT_TRUE(onode_quotas(100, {}, true).empty());
  EXPECT_EQ((std::vector<uint64_t>{0, 0}),
            onode_quotas(1, {{10, true}, {0, true}}, true));
  const auto largest = std::numeric_limits<uint64_t>::max();
  auto quotas = onode_quotas(largest, {{0, true}, {largest, true}}, true);
  EXPECT_EQ(largest, quotas[0] + quotas[1]);
}

TEST(OnodeCacheBudget, RecomputingForMigratedPressurePreservesTotal)
{
  auto a = bluestore_cache::onode_quotas(
      400, {{100, true}, {0, true}, {100, true}, {0, true}}, true);
  auto b = bluestore_cache::onode_quotas(
      400, {{0, true}, {100, true}, {0, true}, {100, true}}, true);
  EXPECT_EQ(a[0], b[1]);
  EXPECT_EQ(a[2], b[3]);
  EXPECT_EQ(400u, std::accumulate(b.begin(), b.end(), uint64_t{0}));
}

json_spirit::mObject decode_status(ceph::JSONFormatter& f)
{
  std::ostringstream out;
  f.flush(out);
  json_spirit::mValue value;
  EXPECT_TRUE(json_spirit::read(out.str(), value)) << out.str();
  return value.get_obj();
}

class OnodePrefetchStore : public StoreTestFixture {
protected:
  OnodePrefetchStore() : StoreTestFixture("bluestore") {}
  virtual bool automatic_prefetch() const { return true; }
  void SetUp() override {
    g_conf()._clear_safe_to_start_threads();
    SetVal(g_conf(), "bluestore_onode_prefetch", automatic_prefetch() ? "true" : "false");
    SetVal(g_conf(), "bluestore_onode_prefetch_reclaim", "true");
    SetVal(g_conf(), "bluestore_block_size", "1073741824");
    SetVal(g_conf(), "bluestore_cache_autotune", "false");
    SetVal(g_conf(), "bluestore_cache_size", "134217728");
    SetVal(g_conf(), "bluestore_cache_type", "2q");
    g_conf().apply_changes(nullptr);
    StoreTestFixture::SetUp();
  }
  json_spirit::mObject status(const char* policy = nullptr) {
    ceph::JSONFormatter f;
    auto* s = static_cast<BlueStore*>(store.get());
    EXPECT_EQ(0, policy ? s->set_onode_cache_policy(policy, &f) :
                          s->get_onode_cache_policy(&f));
    return decode_status(f);
  }
  static ghobject_t oid(int n) {
    return ghobject_t(hobject_t(object_t("prefetch-" + std::to_string(n)),
                                "", CEPH_NOSNAP, 0, 1, ""));
  }
};

TEST_F(OnodePrefetchStore, RealMetadataPrefetchPreservesReadsWritesDeletesAndGating)
{
  coll_t cid(spg_t(pg_t(0, 1), shard_id_t::NO_SHARD));
  ch = store->create_new_collection(cid);
  bufferlist value;
  value.append("original");
  ObjectStore::Transaction t;
  t.create_collection(cid, 0);
  for (int i = 0; i < 128; ++i) {
    t.write(cid, oid(i), 0, value.length(), value);
  }
  ASSERT_EQ(0, store->queue_transaction(ch, std::move(t)));
  ch->flush();
  CloseAndReopen();
  ch = store->open_collection(cid);
  ASSERT_TRUE(ch);
  const auto allocation_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  bool allocated = false;
  do {
    allocated = true;
    const auto current = status();
    for (const auto& shard : current.at("shards").get_array()) {
      allocated &= shard.get_obj().at("target_onodes").get_int64() > 256;
    }
    if (allocated) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } while (std::chrono::steady_clock::now() < allocation_deadline);
  ASSERT_TRUE(allocated);
  auto initial = status();
  ASSERT_FALSE(initial.at("prefetch").get_obj().at("active").get_bool());
  bufferlist actual;
  ASSERT_EQ(8, store->read(ch, oid(0), 0, 8, actual));
  EXPECT_EQ("original", actual.to_str());
  EXPECT_EQ(0, status().at("prefetch").get_obj().at("loaded").get_int64());
  status("s3fifo");
  actual.clear();
  ASSERT_EQ(8, store->read(ch, oid(0), 0, 8, actual));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  json_spirit::mObject ready;
  do {
    ready = status();
    if (ready.at("prefetch").get_obj().at("loaded").get_int64() >= 127) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } while (std::chrono::steady_clock::now() < deadline);
  EXPECT_EQ(127, ready.at("prefetch").get_obj().at("loaded").get_int64());
  EXPECT_EQ(initial.at("onode_misses").get_int64() + 1,
            ready.at("onode_misses").get_int64());
  const auto misses = ready.at("onode_misses").get_int64();
  for (int i = 1; i < 128; ++i) {
    actual.clear();
    ASSERT_EQ(8, store->read(ch, oid(i), 0, 8, actual));
    ASSERT_EQ("original", actual.to_str());
  }
  EXPECT_EQ(misses, status().at("onode_misses").get_int64());
  EXPECT_EQ(127, status().at("prefetch").get_obj().at("used").get_int64());
  bufferlist replacement;
  replacement.append("modified");
  ObjectStore::Transaction update;
  update.write(cid, oid(1), 0, 8, replacement);
  update.remove(cid, oid(2));
  ASSERT_EQ(0, store->queue_transaction(ch, std::move(update)));
  ch->flush();
  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(status("lru").at("prefetch").get_obj().at("active").get_bool());
    status("s3fifo");
    actual.clear();
    ASSERT_EQ(8, store->read(ch, oid(1), 0, 8, actual));
    EXPECT_EQ("modified", actual.to_str());
    actual.clear();
    EXPECT_EQ(-ENOENT, store->read(ch, oid(2), 0, 8, actual));
  }
  // Remount while a scan may be queued, then check persisted data.
  ch.reset();
  CloseAndReopen();
  ch = store->open_collection(cid);
  actual.clear();
  ASSERT_EQ(8, store->read(ch, oid(1), 0, 8, actual));
  EXPECT_EQ("modified", actual.to_str());
  actual.clear();
  EXPECT_EQ(-ENOENT, store->read(ch, oid(2), 0, 8, actual));
  ch.reset();
}

class OnodePrefetchPressureStore : public OnodePrefetchStore {
protected:
  bool automatic_prefetch() const override { return false; }

  void prepare(bool reclaim) {
    coll_t cid(spg_t(pg_t(0, 1), shard_id_t::NO_SHARD));
    ch = store->create_new_collection(cid);
    bufferlist value, attr;
    value.append("data");
    attr.append(std::string(16384, 'x'));
    ObjectStore::Transaction t;
    t.create_collection(cid, 0);
    for (int i = 0; i < 256; ++i) {
      t.write(cid, oid(i), 0, value.length(), value);
      t.setattr(cid, oid(i), "payload", attr);
    }
    ASSERT_EQ(0, store->queue_transaction(ch, std::move(t)));
    ch->flush();
    ch.reset();
    CloseAndReopen();
    ch = store->open_collection(cid);
    ASSERT_TRUE(ch);
    ASSERT_TRUE(wait_for([&] {
      const auto current = status();
      for (const auto& shard : current.at("shards").get_array()) {
        if (shard.get_obj().at("target_onodes").get_int64() <= 512) {
          return false;
        }
      }
      return true;
    }));
    // Keep a previous case's metadata resident; do not restart at activation.
    for (int i = 0; i < 128; ++i) {
      bufferlist actual;
      ASSERT_EQ(4, store->read(ch, oid(i), 0, 4, actual));
    }
    status("s3fifo");
    g_conf()._clear_safe_to_start_threads();
    SetVal(g_conf(), "bluestore_onode_prefetch", "true");
    SetVal(g_conf(), "bluestore_onode_prefetch_reclaim", reclaim ? "true" : "false");
    g_conf().apply_changes(nullptr);
  }

  template<typename Predicate>
  bool wait_for(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    do {
      if (predicate()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }

  void check_data() {
    for (int i = 0; i < 256; ++i) {
      bufferlist actual;
      ASSERT_EQ(4, store->read(ch, oid(i), 0, 4, actual));
      ASSERT_EQ("data", actual.to_str());
      bufferptr attr;
      ASSERT_EQ(0, store->getattr(ch, oid(i), "payload", attr));
      ASSERT_EQ(16384u, attr.length());
      ASSERT_EQ(std::string(16384, 'x'), std::string(attr.c_str(), attr.length()));
    }
  }
};

TEST_F(OnodePrefetchPressureStore, ReclaimsResidentMetadataAndLoadsWithoutRestart)
{
  prepare(true);
  ASSERT_FALSE(HasFatalFailure());
  const auto before = status();
  {
    OnodePrefetchTestPeer peer(static_cast<BlueStore*>(store.get()));
    auto pause = peer.pause_resizing();
    peer.set_shard_quotas(1024);
    const auto budget = peer.used_bytes();
    ASSERT_GT(budget, 1048576u);
    peer.start(budget);
    peer.schedule(ch);
    ASSERT_TRUE(wait_for([&] { return peer.loaded() > 0; }));
    EXPECT_GT(peer.worker.pressure_pauses.load(), 0u);
    EXPECT_GT(peer.worker.reclaim_evicted.load(), 0u);
    EXPECT_LE(peer.worker.reclaim_examined.load(), peer.worker.reclaim_passes.load() * 128);
    EXPECT_EQ(before.at("onode_hits").get_int64(), status().at("onode_hits").get_int64());
    EXPECT_EQ(before.at("onode_misses").get_int64(), status().at("onode_misses").get_int64());
    EXPECT_EQ(budget, peer.worker.meta_limit.load());
    EXPECT_EQ(1u, peer.worker.generation.load());
  }
  check_data();
  coll_t cid(spg_t(pg_t(0, 1), shard_id_t::NO_SHARD));
  bufferlist replacement;
  replacement.append("next");
  ObjectStore::Transaction t;
  t.write(cid, oid(1), 0, 4, replacement);
  t.remove(cid, oid(2));
  ASSERT_EQ(0, store->queue_transaction(ch, std::move(t)));
  ch->flush();
  ch.reset();
  CloseAndReopen();
  ch = store->open_collection(cid);
  bufferlist actual;
  ASSERT_EQ(4, store->read(ch, oid(1), 0, 4, actual));
  EXPECT_EQ("next", actual.to_str());
  actual.clear();
  EXPECT_EQ(-ENOENT, store->read(ch, oid(2), 0, 4, actual));
}

TEST_F(OnodePrefetchPressureStore, DisabledReclaimWaitsAndResumesAtSameGeneration)
{
  prepare(false);
  ASSERT_FALSE(HasFatalFailure());
  {
    OnodePrefetchTestPeer peer(static_cast<BlueStore*>(store.get()));
    auto pause = peer.pause_resizing();
    peer.set_shard_quotas(1024);
    peer.start(0);
    peer.schedule(ch);
    ASSERT_TRUE(wait_for([&] { return peer.worker.pressure_pauses.load() >= 3; }));
    EXPECT_EQ(0u, peer.loaded());
    EXPECT_EQ(0u, peer.worker.reclaim_evicted.load());
    peer.worker.meta_limit.store(peer.used_bytes());
    const auto pauses = peer.worker.pressure_pauses.load();
    ASSERT_TRUE(wait_for([&] { return peer.worker.pressure_pauses.load() > pauses + 3; }));
    EXPECT_EQ(0u, peer.loaded());
    EXPECT_EQ(0u, peer.worker.reclaim_passes.load());
    peer.worker.meta_limit.store(peer.used_bytes() * 4);
    ASSERT_TRUE(wait_for([&] { return peer.loaded() > 0; }));
    EXPECT_EQ(1u, peer.worker.generation.load());
  }
  check_data();
}

TEST_F(OnodePrefetchPressureStore, ShardPressureRetriesTheRejectedCandidate)
{
  prepare(true);
  ASSERT_FALSE(HasFatalFailure());
  {
    OnodePrefetchTestPeer peer(static_cast<BlueStore*>(store.get()));
    auto pause = peer.pause_resizing();
    auto* c = static_cast<BlueStore::Collection*>(ch.get());
    auto* shard = c->get_onode_cache();
    uint64_t missing = 0;
    for (int i = 0; i < 256; ++i) {
      missing += c->onode_space.prefetch_admission(oid(i)) !=
        BlueStore::OnodeSpace::PrefetchAdmission::resident;
    }
    shard->set_max(0);
    peer.start(peer.used_bytes() * 4);
    peer.schedule(ch);
    const bool blocked = wait_for([&] {
      return peer.worker.shard_pressure_pauses.load() >= 3;
    });
    const auto blocked_loads = peer.loaded();
    peer.set_shard_quotas(1024);
    ASSERT_TRUE(blocked);
    EXPECT_EQ(0u, blocked_loads);
    EXPECT_GE(peer.worker.candidate_retries.load(), 3u);
    ASSERT_TRUE(wait_for([&] { return peer.loaded() == missing; }));
    EXPECT_EQ(1u, peer.worker.generation.load());
    EXPECT_EQ(0u, peer.worker.reclaim_evicted.load());
  }
  check_data();
}

class OnodeCacheSwitch : public ::testing::Test {
protected:
  BlueStore store{g_ceph_context, "", 4096};
  std::unique_ptr<PerfCounters> logger;
  std::unique_ptr<BlueStore::OnodeCacheShard> cache;
  std::unique_ptr<BlueStore::BufferCacheShard> buffers;
  BlueStore::CollectionRef coll;

  void SetUp() override {
    PerfCountersBuilder b(g_ceph_context, "onode_switch_test",
                          l_bluestore_onode_hits - 1,
                          l_bluestore_onode_misses + 1);
    b.add_u64_counter(l_bluestore_onode_hits, "onode_hits");
    b.add_u64_counter(l_bluestore_onode_misses, "onode_misses");
    logger.reset(b.create_perf_counters());
    cache.reset(BlueStore::OnodeCacheShard::create(
      g_ceph_context, "lru", logger.get()));
    buffers.reset(BlueStore::BufferCacheShard::create(
      g_ceph_context, "2q", logger.get()));
    cache->set_max(100000);
    coll = ceph::make_ref<BlueStore::Collection>(
      &store, cache.get(), buffers.get(), coll_t());
  }

  void TearDown() override {
    coll->onode_space.clear();
    coll.reset();
  }

  static ghobject_t oid(int n) {
    return ghobject_t(hobject_t(object_t("onode-" + std::to_string(n)),
                               "", CEPH_NOSNAP, 0, 1, ""));
  }

  BlueStore::OnodeRef add(int n, bool exists = true) {
    BlueStore::OnodeRef o(new BlueStore::Onode(coll.get(), oid(n), ""));
    o->exists = exists;
    return coll->onode_space.add_onode(o->oid, o);
  }

  void switch_to(Policy policy) {
    std::lock_guard l(cache->lock);
    cache->_set_policy(policy);
    EXPECT_EQ(policy, cache->_get_policy());
  }

  json_spirit::mObject snapshot() {
    std::lock_guard l(cache->lock);
    ceph::JSONFormatter f;
    f.open_object_section("cache");
    cache->_dump_policy(&f);
    f.close_section();
    return decode_status(f);
  }
};

TEST_F(OnodeCacheSwitch, PrefetchDoesNotManufactureDemandHits)
{
  switch_to(Policy::S3FIFO);
  BlueStore::OnodeRef o(new BlueStore::Onode(coll.get(), oid(1), ""));
  o->exists = true;
  ASSERT_TRUE(coll->onode_space.can_prefetch(oid(1)));
  ASSERT_TRUE(coll->onode_space.add_prefetched(o));
  EXPECT_FALSE(coll->onode_space.can_prefetch(oid(1)));
  EXPECT_FALSE(coll->onode_space.add_prefetched(o));
  EXPECT_EQ(1u, cache->prefetch_loaded);
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_misses));
  ASSERT_EQ(o, coll->onode_space.lookup(oid(1)));
  ASSERT_EQ(o, coll->onode_space.lookup(oid(1)));
  EXPECT_EQ(1u, cache->prefetch_used);
  EXPECT_EQ(2u, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_misses));
  EXPECT_FALSE(o->prefetched);
}

TEST_F(OnodeCacheSwitch, PrefetchRejectsLruAbsentObjectsAndCanceledWork)
{
  BlueStore::OnodeRef o(new BlueStore::Onode(coll.get(), oid(1), ""));
  o->exists = true;
  EXPECT_FALSE(coll->onode_space.can_prefetch(oid(1)));
  EXPECT_FALSE(coll->onode_space.add_prefetched(o));
  switch_to(Policy::S3FIFO);
  o->exists = false;
  EXPECT_FALSE(coll->onode_space.add_prefetched(o));
  o->exists = true;
  std::atomic<uint64_t> epoch{0};
  EXPECT_FALSE(coll->onode_space.add_prefetched(o, &epoch, 1));
  epoch = 2;
  EXPECT_FALSE(coll->onode_space.add_prefetched(o, &epoch, 1));
  EXPECT_TRUE(coll->onode_space.add_prefetched(o, &epoch, 2));
}

TEST_F(OnodeCacheSwitch, AdmissionDoesNotEvictWithoutWorkerReclamation)
{
  switch_to(Policy::S3FIFO);
  cache->set_max(10);
  for (int i = 0; i < 9; ++i) {
    add(i);
  }
  BlueStore::OnodeRef o(new BlueStore::Onode(coll.get(), oid(99), ""));
  o->exists = true;
  EXPECT_FALSE(coll->onode_space.can_prefetch(oid(99)));
  EXPECT_FALSE(coll->onode_space.add_prefetched(o));
  EXPECT_EQ(9u, cache->_get_num());
  for (int i = 0; i < 9; ++i) {
    EXPECT_TRUE(coll->onode_space.lookup(oid(i)));
  }
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_misses));
  cache->set_max(0);
  EXPECT_FALSE(coll->onode_space.add_prefetched(o));
}

TEST_F(OnodeCacheSwitch, PressureReclaimMakesRoomWithoutChangingQuotaOrDemandCounters)
{
  switch_to(Policy::S3FIFO);
  cache->set_max(10);
  for (int i = 0; i < 9; ++i) {
    add(i);
  }
  ASSERT_FALSE(coll->onode_space.can_prefetch(oid(99)));
  {
    std::lock_guard l(cache->lock);
    auto result = cache->_reclaim_for_prefetch(4);
    EXPECT_LE(result.examined, 4u);
    EXPECT_EQ(1u, result.evicted);
    EXPECT_EQ(8u, cache->_get_num());
  }
  EXPECT_EQ(10u, cache->max.load());
  EXPECT_TRUE(coll->onode_space.can_prefetch(oid(99)));
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_misses));
}

TEST_F(OnodeCacheSwitch, PressureReclaimProtectsPinnedMetadataAndBoundsClockWork)
{
  switch_to(Policy::S3FIFO);
  auto pinned = add(1);
  pinned->onode.size = 4096;
  for (int i = 2; i < 18; ++i) {
    add(i);
    coll->onode_space.lookup(oid(i));
    coll->onode_space.lookup(oid(i));
  }
  const auto hits = logger->get(l_bluestore_onode_hits);
  {
    std::lock_guard l(cache->lock);
    EXPECT_EQ(0u, cache->_reclaim_for_prefetch(0).examined);
    auto first = cache->_reclaim_for_prefetch(1);
    EXPECT_LE(first.examined, 1u);
    EXPECT_EQ(0u, first.evicted);
    for (int i = 0; i < 100; ++i) {
      auto result = cache->_reclaim_for_prefetch(4);
      EXPECT_LE(result.examined, 4u);
      EXPECT_LE(result.evicted, 1u);
    }
    EXPECT_EQ(1u, cache->_get_num());
  }
  EXPECT_EQ(hits, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(4096u, coll->onode_space.lookup(oid(1))->onode.size);
}

TEST_F(OnodeCacheSwitch, PressureReclaimIsInactiveUnderLruAndAccountsUnusedOnce)
{
  add(1);
  {
    std::lock_guard l(cache->lock);
    EXPECT_EQ(0u, cache->_reclaim_for_prefetch(128).examined);
    EXPECT_EQ(1u, cache->_get_num());
  }
  coll->onode_space.clear();
  switch_to(Policy::S3FIFO);
  {
    BlueStore::OnodeRef o(new BlueStore::Onode(coll.get(), oid(2), ""));
    o->exists = true;
    ASSERT_TRUE(coll->onode_space.add_prefetched(o));
  }
  {
    std::lock_guard l(cache->lock);
    EXPECT_EQ(1u, cache->_reclaim_for_prefetch(4).evicted);
    EXPECT_EQ(1u, cache->prefetch_unused);
    EXPECT_EQ(0u, cache->_reclaim_for_prefetch(4).evicted);
    EXPECT_EQ(1u, cache->prefetch_unused);
  }
}

TEST_F(OnodeCacheSwitch, UnusedPrefetchIsCountedOnceOnEvictionOrClear)
{
  switch_to(Policy::S3FIFO);
  for (int i = 0; i < 16; ++i) {
    BlueStore::OnodeRef o(new BlueStore::Onode(coll.get(), oid(i), ""));
    o->exists = true;
    ASSERT_TRUE(coll->onode_space.add_prefetched(o));
  }
  {
    std::lock_guard l(cache->lock);
    cache->_trim_to(8);
  }
  EXPECT_GT(cache->prefetch_unused, 0u);
  coll->onode_space.clear();
  EXPECT_EQ(16u, cache->prefetch_unused);
  EXPECT_EQ(0u, cache->prefetch_used);
  coll->onode_space.clear();
  EXPECT_EQ(16u, cache->prefetch_unused);
}

TEST_F(OnodeCacheSwitch, DuplicatePrefetchCannotReplaceDirtyDemandOnode)
{
  switch_to(Policy::S3FIFO);
  auto resident = add(1);
  resident->onode.size = 1024;
  BlueStore::OnodeRef disk(new BlueStore::Onode(coll.get(), oid(1), ""));
  disk->exists = true;
  disk->onode.size = 64;
  EXPECT_FALSE(coll->onode_space.add_prefetched(disk));
  EXPECT_EQ(1024u, coll->onode_space.lookup(oid(1))->onode.size);
  EXPECT_EQ(0u, cache->prefetch_loaded);
}

TEST_F(OnodeCacheSwitch, ConcurrentPrefetchAndPolicyChangesPreserveDemandAccounting)
{
  std::thread worker([&] {
    for (int i = 0; i < 1000; ++i) {
      BlueStore::OnodeRef o(new BlueStore::Onode(coll.get(), oid(i), ""));
      o->exists = true;
      coll->onode_space.add_prefetched(o);
    }
  });
  for (int i = 0; i < 1000; ++i) {
    switch_to(i % 2 ? Policy::LRU : Policy::S3FIFO);
    coll->onode_space.lookup(oid(i));
  }
  worker.join();
  EXPECT_EQ(1000u, logger->get(l_bluestore_onode_hits) +
                     logger->get(l_bluestore_onode_misses));
  coll->onode_space.clear();
  EXPECT_EQ(cache->prefetch_loaded, cache->prefetch_used + cache->prefetch_unused);
}

TEST_F(OnodeCacheSwitch, WarmRoundTripPreservesEntriesBinsAndCounters)
{
  for (int i = 0; i < 32; ++i) {
    add(i);
  }
  ASSERT_FALSE(coll->onode_space.lookup(oid(100)));
  cache->set_bin_count(3);
  cache->shift_bins();
  const auto bins = cache->sum_bins(0, 3);
  for (auto policy : {Policy::S3FIFO, Policy::LRU, Policy::S3FIFO}) {
    const auto hits = logger->get(l_bluestore_onode_hits);
    const auto misses = logger->get(l_bluestore_onode_misses);
    auto* const address = cache.get();
    switch_to(policy);
    EXPECT_EQ(address, coll->get_onode_cache());
    EXPECT_EQ(32u, cache->_get_num());
    EXPECT_EQ(bins, cache->sum_bins(0, 3));
    EXPECT_EQ(hits, logger->get(l_bluestore_onode_hits));
    EXPECT_EQ(misses, logger->get(l_bluestore_onode_misses));
    for (int i = 0; i < 32; ++i) {
      ASSERT_TRUE(coll->onode_space.lookup(oid(i)));
    }
  }
  EXPECT_EQ(96u, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(1u, logger->get(l_bluestore_onode_misses));
}

TEST_F(OnodeCacheSwitch, BorrowedQuotaRetainsRealOnodes)
{
  switch_to(Policy::S3FIFO);
  cache->set_max(4);
  for (int i = 0; i < 6; ++i) {
    add(i);
  }
  auto quotas = bluestore_cache::onode_quotas(
      16, {{cache->_get_num(), true}, {0, true}, {0, true}, {0, true}}, true);
  ASSERT_EQ(13u, quotas[0]);
  EXPECT_EQ(16u, std::accumulate(quotas.begin(), quotas.end(), uint64_t{0}));
  cache->set_max(quotas[0]);
  for (int i = 0; i < 6; ++i) {
    if (!coll->onode_space.lookup(oid(i))) {
      add(i);
    }
  }
  const auto hits = logger->get(l_bluestore_onode_hits);
  const auto misses = logger->get(l_bluestore_onode_misses);
  for (int i = 0; i < 60; ++i) {
    ASSERT_TRUE(coll->onode_space.lookup(oid(i % 6)));
  }
  EXPECT_EQ(hits + 60, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(misses, logger->get(l_bluestore_onode_misses));
  EXPECT_EQ(6u, cache->_get_num());
  EXPECT_EQ(13u, snapshot().at("target_onodes").get_uint64());
}

TEST_F(OnodeCacheSwitch, PinnedAndNonexistentOnodesSurviveSwitch)
{
  auto live = add(1);
  auto absent = add(2, false);
  EXPECT_EQ(2, snapshot().at("unlinked_onodes").get_int());
  switch_to(Policy::S3FIFO);
  absent.reset();
  live.reset();
  EXPECT_EQ(1u, cache->_get_num());
  EXPECT_EQ(1, snapshot().at("small_entries").get_int());

  auto pinned_linked = coll->onode_space.lookup(oid(1));
  switch_to(Policy::LRU);
  cache->flush();
  EXPECT_EQ(1u, cache->_get_num());
  EXPECT_FALSE(pinned_linked->lru_item.is_linked());
  switch_to(Policy::S3FIFO);
  pinned_linked.reset();
  EXPECT_EQ(1u, cache->sum_bins(0, 1));
  cache->flush();
  EXPECT_TRUE(cache->empty());
  EXPECT_EQ(0u, cache->sum_bins(0, 1));
}

TEST_F(OnodeCacheSwitch, MainGhostAndIdempotentSwitch)
{
  switch_to(Policy::S3FIFO);
  for (int i = 0; i < 20; ++i) {
    add(i);
  }
  ASSERT_TRUE(coll->onode_space.lookup(oid(0)));
  ASSERT_TRUE(coll->onode_space.lookup(oid(0)));
  {
    std::lock_guard l(cache->lock);
    cache->_trim_to(19);
  }
  auto before = snapshot();
  EXPECT_EQ(1, before.at("main_entries").get_int());
  EXPECT_EQ(1, before.at("ghost_entries").get_int());
  switch_to(Policy::S3FIFO);
  EXPECT_EQ(before, snapshot());
  switch_to(Policy::LRU);
  EXPECT_EQ(19, snapshot().at("lru_entries").get_int());
  EXPECT_EQ(0, snapshot().at("ghost_entries").get_int());
  EXPECT_EQ(19u, cache->sum_bins(0, 1));
  cache->flush();
  EXPECT_TRUE(cache->empty());
  EXPECT_EQ(0u, cache->sum_bins(0, 1));
}

TEST_F(OnodeCacheSwitch, OverlappingHitsPromoteReusedOnode)
{
  switch_to(Policy::S3FIFO);
  for (int i = 0; i < 20; ++i) {
    add(i);
  }
  // Keep both lookup references alive: two hits, only one final unpin.
  auto first = coll->onode_space.lookup(oid(0));
  auto second = coll->onode_space.lookup(oid(0));
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto* reused = first.get();
  EXPECT_EQ(2, reused->s3fifo_freq);
  second.reset();
  first.reset();
  EXPECT_EQ(2, reused->s3fifo_freq);
  {
    std::lock_guard l(cache->lock);
    cache->_trim_to(19);
  }
  EXPECT_EQ(1, snapshot().at("main_entries").get_int());
  EXPECT_EQ(19u, cache->_get_num());
  EXPECT_EQ(19u, cache->sum_bins(0, 1));
  EXPECT_TRUE(coll->onode_space.lookup(oid(0)));
  EXPECT_EQ(3u, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_misses));
}

TEST_F(OnodeCacheSwitch, ReferenceReleaseIsNotAnotherHit)
{
  switch_to(Policy::S3FIFO);
  auto inserted = add(0);
  auto* raw = inserted.get();
  inserted.reset();
  EXPECT_EQ(0, raw->s3fifo_freq);
  {
    auto hit = coll->onode_space.lookup(oid(0));
    auto copy = hit;
    EXPECT_EQ(1, raw->s3fifo_freq);
  }
  EXPECT_EQ(1, raw->s3fifo_freq);
  {
    BlueStore::OnodeRef internal_reference(raw);
  }
  EXPECT_EQ(1, raw->s3fifo_freq);
  EXPECT_EQ(1u, logger->get(l_bluestore_onode_hits));
  for (int i = 0; i < 10; ++i) {
    auto hit = coll->onode_space.lookup(oid(0));
    EXPECT_LE(raw->s3fifo_freq, 3);
  }
  EXPECT_EQ(3, raw->s3fifo_freq);
}

TEST_F(OnodeCacheSwitch, GhostShrinkKeepsMostRecentEvictions)
{
  switch_to(Policy::S3FIFO);
  for (int i = 0; i < 10; ++i) {
    add(i);
  }
  {
    std::lock_guard l(cache->lock);
    cache->_trim_to(5);
  }
  EXPECT_EQ(4, snapshot().at("ghost_entries").get_int());
  // Remove residents so a subsequent shrink does not add new ghosts.
  coll->onode_space.clear();
  {
    std::lock_guard l(cache->lock);
    cache->_trim_to(2);
  }
  EXPECT_EQ(1, snapshot().at("ghost_entries").get_int());
  add(4);
  EXPECT_EQ(1, snapshot().at("main_entries").get_int());
  add(1);
  EXPECT_EQ(1, snapshot().at("small_entries").get_int());
}

TEST_F(OnodeCacheSwitch, StaleUnpinDispatchesToDestinationShard)
{
  auto other_cache = std::unique_ptr<BlueStore::OnodeCacheShard>(
    BlueStore::OnodeCacheShard::create(g_ceph_context, "s3fifo", logger.get()));
  auto other_buffers = std::unique_ptr<BlueStore::BufferCacheShard>(
    BlueStore::BufferCacheShard::create(g_ceph_context, "2q", logger.get()));
  other_cache->set_max(100000);
  auto dest = ceph::make_ref<BlueStore::Collection>(
    &store, other_cache.get(), other_buffers.get(),
    coll_t(spg_t(pg_t(0, 1), shard_id_t::NO_SHARD)));
  dest->cnode.bits = 0;
  auto hold = add(1);
  auto* raw = hold.get();
  coll->split_cache(dest.get());
  hold.reset();
  cache->maybe_unpin(raw);
  EXPECT_TRUE(cache->empty());
  EXPECT_EQ(1u, other_cache->_get_num());
  EXPECT_EQ(0u, cache->sum_bins(0, 1));
  EXPECT_EQ(1u, other_cache->sum_bins(0, 1));
  EXPECT_EQ(BlueStore::Onode::Q_SMALL, raw->s3fifo_queue);
  dest->onode_space.clear();
}

TEST_F(OnodeCacheSwitch, ConcurrentLookupsAndRepeatedSwitches)
{
  for (int i = 0; i < 32; ++i) {
    add(i);
  }
  std::atomic<bool> start{false};
  auto reader = [&] {
    while (!start.load()) {
      std::this_thread::yield();
    }
    for (int i = 0; i < 10000; ++i) {
      EXPECT_TRUE(coll->onode_space.lookup(oid(i % 32)));
    }
  };
  std::thread a(reader), b(reader);
  start = true;
  for (int i = 0; i < 1000; ++i) {
    switch_to(i % 2 ? Policy::LRU : Policy::S3FIFO);
  }
  a.join();
  b.join();
  EXPECT_EQ(32u, cache->_get_num());
  EXPECT_EQ(32u, cache->sum_bins(0, 1));
  EXPECT_EQ(20000u, logger->get(l_bluestore_onode_hits));
  EXPECT_EQ(0u, logger->get(l_bluestore_onode_misses));
}

TEST(OnodeCachePolicy, StoreFanoutStatusAndInvalidPolicy)
{
  BlueStore store(g_ceph_context, "", 4096);
  store.set_cache_shards(4);
  auto status = [&](const std::string& policy) {
    ceph::JSONFormatter f;
    EXPECT_EQ(0, policy.empty() ? store.get_onode_cache_policy(&f) :
              store.set_onode_cache_policy(policy, &f));
    return decode_status(f);
  };
  auto initial = status("lru");
  const auto generation = initial.at("policy_generation").get_uint64();
  auto enabled = status("s3fifo");
  EXPECT_EQ("s3fifo", enabled.at("effective_policy").get_str());
  EXPECT_EQ(generation + 1, enabled.at("policy_generation").get_uint64());
  EXPECT_EQ(initial.at("cache_instance"), enabled.at("cache_instance"));
  EXPECT_EQ(initial.at("buffer_cache_policy"), enabled.at("buffer_cache_policy"));
  EXPECT_EQ(initial.at("onode_hits"), enabled.at("onode_hits"));
  EXPECT_EQ(initial.at("onode_misses"), enabled.at("onode_misses"));
  EXPECT_EQ(4u, enabled.at("shards").get_array().size());
  for (const auto& shard : enabled.at("shards").get_array()) {
    EXPECT_EQ("s3fifo", shard.get_obj().at("effective_policy").get_str());
  }
  EXPECT_EQ(generation + 1, status("s3fifo").at("policy_generation").get_uint64());
  ceph::JSONFormatter invalid;
  EXPECT_EQ(-EINVAL, store.set_onode_cache_policy("2q", &invalid));
  EXPECT_EQ("s3fifo", status("").at("effective_policy").get_str());
  store.set_cache_shards(6);
  auto expanded = status("");
  EXPECT_EQ("s3fifo", expanded.at("effective_policy").get_str());
  EXPECT_EQ(6u, expanded.at("shards").get_array().size());
  EXPECT_EQ(generation + 2, status("lru").at("policy_generation").get_uint64());
}

TEST_F(OnodeCacheSwitch, ConcurrentEvictionInsertionAndSwitch)
{
  cache->set_max(16);
  std::atomic<bool> start{false};
  auto reader = [&] {
    while (!start.load()) {
      std::this_thread::yield();
    }
    for (int i = 0; i < 2000; ++i) {
      auto o = coll->onode_space.lookup(oid(i % 32));
      if (!o) {
        o = add(i % 32);
      }
      EXPECT_TRUE(o->exists);
    }
  };
  std::thread a(reader), b(reader);
  start = true;
  for (int i = 0; i < 500; ++i) {
    std::lock_guard l(cache->lock);
    cache->_set_policy(i % 2 ? Policy::LRU : Policy::S3FIFO);
    cache->_trim_to(8);
  }
  a.join();
  b.join();
  EXPECT_EQ(4000u, logger->get(l_bluestore_onode_hits) +
                     logger->get(l_bluestore_onode_misses));
  EXPECT_EQ(cache->_get_num(), cache->sum_bins(0, 1));
  cache->flush();
  EXPECT_TRUE(cache->empty());
  EXPECT_EQ(0u, cache->sum_bins(0, 1));
}

TEST(OnodeCachePolicy, FactoryKeepsStartupCompatibility)
{
  for (const auto* name : {"lru", "2q", "s3fifo"}) {
    std::unique_ptr<BlueStore::OnodeCacheShard> shard(
      BlueStore::OnodeCacheShard::create(g_ceph_context, name, nullptr));
    const auto expected = std::string(name) == "s3fifo" ? Policy::S3FIFO : Policy::LRU;
    EXPECT_EQ(expected, shard->_get_policy());
  }
}

TEST(OnodeCachePolicy, InvalidS3fifoTunablesLeaveLruUnchanged)
{
  CephContext cct(CEPH_ENTITY_TYPE_OSD);
  ASSERT_EQ(0, cct._conf.set_val("bluestore_cache_type", "lru"));
  ASSERT_EQ(0, cct._conf.set_val("bluestore_cache_s3fifo_small_ratio", "0"));
  cct._conf.apply_changes(nullptr);
  BlueStore store(&cct, "", 4096);
  store.set_cache_shards(4);
  ceph::JSONFormatter invalid;
  EXPECT_EQ(-EINVAL, store.set_onode_cache_policy("s3fifo", &invalid));
  ceph::JSONFormatter f;
  ASSERT_EQ(0, store.get_onode_cache_policy(&f));
  auto result = decode_status(f);
  EXPECT_EQ("lru", result.at("effective_policy").get_str());
  EXPECT_EQ(0u, result.at("policy_generation").get_uint64());
}
} // namespace
