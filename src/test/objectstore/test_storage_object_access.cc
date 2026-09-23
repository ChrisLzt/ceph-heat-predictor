#include "os/ObjectStore.h"
#include "common/config.h"
#include "store_test_fixture.h"
#include <mutex>
#include <vector>

class StorageObjectAccess : public StoreTestFixture {
public:
  StorageObjectAccess() : StoreTestFixture("bluestore") {}
  void SetUp() override {
    g_conf()._clear_safe_to_start_threads();
    SetVal(g_conf(), "bluestore_block_size", "2147483648");
    SetVal(g_conf(), "bluestore_block_create", "true");
    SetVal(g_conf(), "bluestore_fsck_on_mount", "false");
    StoreTestFixture::SetUp();
  }
  void TearDown() override {
    if (store) store->clear_data_access_observer();
    StoreTestFixture::TearDown();
  }
};

TEST_F(StorageObjectAccess, ActualStorageIdentityAndReadCounting) {
  coll_t cid(spg_t(pg_t(0, 1), shard_id_t::NO_SHARD));
  ch = store->create_new_collection(cid);
  ghobject_t physical(hobject_t(object_t("agg.data.P"), "", CEPH_NOSNAP, 0, 1, ""));
  ghobject_t missing(hobject_t(object_t("missing"), "", CEPH_NOSNAP, 0, 1, ""));
  std::mutex mutex;
  std::vector<std::pair<hobject_t, HpAccessType>> events;
  std::vector<uint64_t> lengths;
  store->set_data_access_observer([&](const hobject_t& oid, HpAccessType kind, uint64_t length) {
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_GT(length, 0u);
    events.emplace_back(oid, kind);
    lengths.push_back(length);
  });
  struct ClearObserver {
    ObjectStore* store;
    ~ClearObserver() { store->clear_data_access_observer(); }
  } clear_before_event_storage_dies{store.get()};
  bufferlist data; data.append_zero(8192);
  ObjectStore::Transaction write;
  write.create_collection(cid, 0);
  write.write(cid, physical, 0, data.length(), data);
  ASSERT_EQ(0, store->queue_transaction(ch, std::move(write)));
  ch->flush();
  { std::lock_guard<std::mutex> lock(mutex);
    ASSERT_EQ(1u, events.size()); EXPECT_EQ(physical.hobj, events[0].first);
    EXPECT_EQ(HpAccessType::Write, events[0].second); }
  bufferlist result;
  ASSERT_EQ(8192, store->read(ch, physical, 0, 0, result));
  ASSERT_EQ(4096, store->read(ch, physical, 0, 4096, result)); // cached read still observed
  interval_set<uint64_t> ranges; ranges.insert(0, 1024); ranges.insert(4096, 1024);
  ASSERT_EQ(2048, store->readv(ch, physical, ranges, result));
  ASSERT_EQ(4096, store->read(ch, physical, 4096, 8192, result)); // clip at EOF
  ASSERT_EQ(0, store->read(ch, physical, 9000, 1, result)); // beyond EOF
  interval_set<uint64_t> empty;
  ASSERT_EQ(0, store->readv(ch, physical, empty, result));
  ASSERT_EQ(-ENOENT, store->read(ch, missing, 0, 10, result));
  { std::lock_guard<std::mutex> lock(mutex);
    ASSERT_EQ(5u, events.size());
    EXPECT_EQ((std::vector<uint64_t>{8192,8192,4096,2048,4096}), lengths);
    for (size_t i=1; i<events.size(); ++i) {
      EXPECT_EQ(physical.hobj, events[i].first); EXPECT_EQ(HpAccessType::Read, events[i].second);
    } }
  // Non-data metadata mutations must not look like data writes.
  ObjectStore::Transaction attr;
  bufferlist value; value.append("x"); attr.setattr(cid, physical, "meta", value);
  ASSERT_EQ(0, store->queue_transaction(ch, std::move(attr))); ch->flush();
  { std::lock_guard<std::mutex> lock(mutex); ASSERT_EQ(5u, events.size()); }
  ObjectStore::Transaction bookkeeping;
  ghobject_t pgmeta = ghobject_t::make_pgmeta(1, 0, shard_id_t::NO_SHARD);
  bookkeeping.write(cid, pgmeta, 0, value.length(), value);
  ASSERT_EQ(0, store->queue_transaction(ch, std::move(bookkeeping))); ch->flush();
  { std::lock_guard<std::mutex> lock(mutex); ASSERT_EQ(5u, events.size()); }
  store->clear_data_access_observer();
  ASSERT_EQ(4096, store->read(ch, physical, 0, 4096, result));
  { std::lock_guard<std::mutex> lock(mutex); ASSERT_EQ(5u, events.size()); }
}
