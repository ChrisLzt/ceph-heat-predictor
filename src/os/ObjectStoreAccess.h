#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include "heatpredictor/hp_access_type.h"

// Host-owned callback, independent of the predictor and storage backend.
// set/clear must not be called from a callback. clear drains in-flight calls
// before the host destroys their target. Callback failures never fail data I/O.
template<class Object>
class ObjectStoreAccess {
public:
  using Callback = std::function<void(const Object&, HpAccessType, uint64_t)>;
  void set(Callback next) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    callback = std::move(next);
    connected.store(bool(callback), std::memory_order_release);
  }
  void clear() {
    connected.store(false, std::memory_order_release);
    std::unique_lock<std::shared_mutex> lock(mutex);
    callback = nullptr;
  }
  void notify(const Object& object, HpAccessType kind, uint64_t length) noexcept {
    if (!length || !connected.load(std::memory_order_acquire)) return;
    try {
      std::shared_lock<std::shared_mutex> lock(mutex);
      if (callback) callback(object, kind, length);
    } catch (...) {
      // This optional observer must not affect the storage operation.
    }
  }
private:
  std::atomic<bool> connected{false};
  std::shared_mutex mutex;
  Callback callback;
};
