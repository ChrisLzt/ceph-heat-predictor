// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-

#include "include/types.h"
#include "common/Formatter.h"
#include "common/ceph_context.h"
#include "common/perf_counters.h"
#include "global/global_context.h"
#include "json_spirit/json_spirit.h"
#include "os/bluestore/BlueStore.h"
#include "gtest/gtest.h"

#include <memory>
#include <sstream>
#include <thread>

namespace {
using Policy = BlueStore::OnodeCacheShard::Policy;

json_spirit::mObject decode_status(ceph::JSONFormatter& f)
{
  std::ostringstream out;
  f.flush(out);
  json_spirit::mValue value;
  EXPECT_TRUE(json_spirit::read(out.str(), value)) << out.str();
  return value.get_obj();
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
