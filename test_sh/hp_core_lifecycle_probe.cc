// This probe deliberately builds without Ceph headers, generated config or libs.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#define private public
#include "heatpredictor/heat_predictor.h"
#undef private

static void require(bool ok, const char* message) {
  if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}

static void test_shutdown_waits_for_callback() {
  std::mutex lock;
  std::condition_variable cv;
  bool entered = false, release = false, stopping = false, finished = false;
  HeatPredictor hp([&](uint64_t expired) {
    if (!expired) return;
    std::unique_lock guard(lock);
    entered = true;
    cv.notify_all();
    cv.wait(guard, [&] { return release; });
  });
  hp.set_enabled(true);
  hp.eq = std::make_unique<EvaluationQueue>(
      1000000000ULL, 100, 100.0, 10000000ULL, 100, 1000000ULL);
  hp.predict(1, 1, 1, nullptr);
  {
    std::unique_lock guard(lock);
    require(cv.wait_for(guard, std::chrono::seconds(2), [&] { return entered; }),
            "expiry must invoke the instance callback");
  }
  std::thread shutdown([&] {
    { std::lock_guard guard(lock); stopping = true; cv.notify_all(); }
    hp.shutdown();
    { std::lock_guard guard(lock); finished = true; cv.notify_all(); }
  });
  {
    std::unique_lock guard(lock);
    cv.wait(guard, [&] { return stopping; });
    require(!cv.wait_for(guard, std::chrono::milliseconds(20), [&] { return finished; }),
            "shutdown must not return while callback still uses owner state");
    release = true;
    cv.notify_all();
  }
  shutdown.join();
  require(finished, "shutdown completes after callback returns");
}

int main() {
  test_shutdown_waits_for_callback();
  for (int repetition = 0; repetition < 3; ++repetition) {
    HeatPredictor hp;
    require(!hp.train_model && !hp.get_prediction_snapshot(),
            "disabled construction must not create a forest");
    require(!hp.is_enabled() && hp.status().evaluation.io_count == 0,
            "disabled status must be readable");
    hp.reset();
    require(!hp.train_model, "disabled reset must stay lazy");
    hp.set_enabled(true);
    require(hp.train_model && hp.get_prediction_snapshot(), "enable creates model");
    uint64_t sequence = 0;
    hp.predict(1, 22, 33, &sequence);
    require(sequence == 1, "first observation recorded");
    require(hp.train_thread.joinable() && hp.expiry_thread.joinable(),
            "observation starts workers");
    require(hp.reset() == 1, "reset reports discarded pending sample");
    require(hp.is_enabled() && hp.status().evaluation.io_count == 0,
            "reset preserves enable and clears counters");
    hp.predict(1, 22, 33, &sequence);
    require(hp.set_enabled(false) == 1, "disable discards pending sample");
    require(!hp.train_model && !hp.get_prediction_snapshot(), "disable releases forest");
    hp.predict(1, 22, 33, &sequence);
    require(sequence == 0 && hp.status().evaluation.io_count == 0,
            "disabled observations ignored");
    hp.set_enabled(true);
    hp.predict(1, 22, 33, &sequence);
    require(sequence == 1, "re-enable starts fresh");
    hp.shutdown();
    hp.shutdown();
    require(!hp.train_thread.joinable() && !hp.expiry_thread.joinable(),
            "shutdown joins both workers and is idempotent");
  }
  require(HeatPredictor::make_object_key(1, 2, 3) !=
          HeatPredictor::make_object_key(2, 2, 3), "pools stay isolated");
  std::cout << "core lifecycle: PASS\n";
}
