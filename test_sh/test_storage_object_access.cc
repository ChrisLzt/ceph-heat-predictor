#include "os/ObjectStoreAccess.h"
#include <cassert>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
  ObjectStoreAccess<int> access;
  std::vector<int> seen;
  access.notify(17, HpAccessType::Read, 1); // no observer
  access.set([&](const int& object, HpAccessType kind, uint64_t length) {
    assert(length == 4096);
    seen.push_back(object);
    assert(kind == HpAccessType::Read || kind == HpAccessType::Write);
  });
  access.notify(99, HpAccessType::Read, 0);
  access.notify(17, HpAccessType::Read, 4096);
  access.notify(18, HpAccessType::Write, 4096);
  assert((seen == std::vector<int>{17,18})); // identity is the actual supplied storage object
  access.clear();
  access.notify(19, HpAccessType::Read, 4096);
  assert(seen.size() == 2);
  access.set([](const int&, HpAccessType, uint64_t) { throw std::runtime_error("observer failure"); });
  access.notify(20, HpAccessType::Write, 1); // cannot break storage I/O
  access.clear();

  std::promise<void> entered, release, clearing;
  auto unblock = release.get_future().share();
  std::atomic<int> calls{0};
  access.set([&](const int&, HpAccessType, uint64_t) {
    ++calls; entered.set_value(); unblock.wait();
  });
  std::thread io([&] { access.notify(21, HpAccessType::Read, 1); });
  entered.get_future().wait();
  auto drained = std::async(std::launch::async, [&] { clearing.set_value(); access.clear(); });
  clearing.get_future().wait();
  assert(drained.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
  release.set_value();
  io.join(); drained.get();
  access.notify(22, HpAccessType::Read, 1);
  assert(calls == 1);
  access.clear(); // idempotent
}
