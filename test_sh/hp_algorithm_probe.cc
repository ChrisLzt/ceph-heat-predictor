#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/debug.h"

#define private public
#include "heatpredictor/heat_predictor.h"
#undef private

#include "heatpredictor/hp_evaluation_queue.h"
#include "heatpredictor/hp_features.h"
#include "heatpredictor/hp_future_access_threshold.h"
#include "heatpredictor/include/ARFClassifier.h"
#include "heatpredictor/include/drift/ADWIN.h"

namespace ceph {

void __ceph_assert_fail(const assert_data& ctx)
{
  std::cerr << "ceph_assert failed: " << ctx.assertion
            << " at " << ctx.file << ":" << ctx.line << std::endl;
  std::abort();
}

void __ceph_assert_fail(
    const char *assertion,
    const char *file,
    int line,
    const char *)
{
  std::cerr << "ceph_assert failed: " << assertion
            << " at " << file << ":" << line << std::endl;
  std::abort();
}

} // namespace ceph

namespace {

void require(bool condition, const char *message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void require_close(double lhs, double rhs, const char *message)
{
  if (std::abs(lhs - rhs) > 0.000001) {
    std::cerr << "FAIL: " << message
              << " lhs=" << lhs << " rhs=" << rhs << std::endl;
    std::exit(1);
  }
}

PredictionSample make_sample(uint64_t sequence, uint64_t object_key)
{
  return PredictionSample{
      sequence,
      object_key,
      0.0,
      0,
      0,
      0,
      0,
      0,
      0.0,
      0};
}

std::atomic<uint64_t> background_error_notification_count{0};

void record_background_error_notification()
{
  background_error_notification_count.fetch_add(
      1, std::memory_order_relaxed);
}

class BlockingTrainingClassifier : public Classifier {
 private:
  std::mutex mutex;
  std::condition_variable condition;
  bool first_sample_released = false;
  size_t learned_sample_count = 0;

 public:
  void learn_one(
      const std::vector<double>&, int, double = 1.0) override
  {
    std::unique_lock<std::mutex> lock(mutex);
    ++learned_sample_count;
    condition.notify_all();
    if (learned_sample_count == 1) {
      condition.wait(lock, [this] { return first_sample_released; });
    }
  }

  std::vector<double> predict_proba_one(
      const std::vector<double>&) override
  {
    return {1.0, 0.0};
  }

  std::unique_ptr<Classifier> clone_for_prediction() const override
  {
    return std::make_unique<BlockingTrainingClassifier>();
  }

  void wait_until_first_sample_starts()
  {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return learned_sample_count == 1; });
  }

  void release_first_sample()
  {
    std::lock_guard<std::mutex> lock(mutex);
    first_sample_released = true;
    condition.notify_all();
  }

  size_t learned_count()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return learned_sample_count;
  }
};

class ThrowingTrainingClassifier : public Classifier {
 private:
  std::mutex mutex;
  std::condition_variable condition;
  bool learn_called = false;

 public:
  void learn_one(
      const std::vector<double>&, int, double = 1.0) override
  {
    {
      std::lock_guard<std::mutex> lock(mutex);
      learn_called = true;
    }
    condition.notify_all();
    throw std::runtime_error("injected training failure");
  }

  std::vector<double> predict_proba_one(
      const std::vector<double>&) override
  {
    return {1.0, 0.0};
  }

  std::unique_ptr<Classifier> clone_for_prediction() const override
  {
    return std::make_unique<ThrowingTrainingClassifier>();
  }

  void wait_until_learn_is_called()
  {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return learn_called; });
  }
};

void test_training_shutdown_finishes_only_current_batch()
{
  HeatPredictor predictor;
  auto classifier = std::make_shared<BlockingTrainingClassifier>();
  predictor.train_model = classifier;
  predictor.last_snapshot_publish_time_ns = HeatPredictor::monotonic_now_ns();

  const size_t queued_count =
      2 * static_cast<size_t>(HeatPredictor::BATCH_SIZE);
  for (size_t index = 0; index < queued_count; ++index) {
    predictor.train_queue.push(TrainingSample{
        make_sample(index + 1, index + 1),
        static_cast<int>(index % 2)});
  }

  predictor.train_running.store(true);
  predictor.train_thread =
      std::thread(&HeatPredictor::train_worker, &predictor);
  classifier->wait_until_first_sample_starts();

  predictor.train_running.store(false);
  predictor.train_queue_cv.notify_all();
  classifier->release_first_sample();
  predictor.train_thread.join();

  require(classifier->learned_count() == HeatPredictor::BATCH_SIZE,
          "shutdown must finish only the batch already in progress");
  require(predictor.train_queue.size() == HeatPredictor::BATCH_SIZE,
          "shutdown must leave later queued samples unprocessed");
}

void test_training_exception_disables_predictor_without_terminating()
{
  HeatPredictor predictor;
  background_error_notification_count.store(0);
  predictor.set_background_error_callback(
      record_background_error_notification);
  auto classifier = std::make_shared<ThrowingTrainingClassifier>();
  predictor.train_model = classifier;
  predictor.train_queue.push(
      TrainingSample{make_sample(1, 1), 0});

  predictor.train_running.store(true);
  predictor.train_thread =
      std::thread(&HeatPredictor::train_worker, &predictor);
  classifier->wait_until_learn_is_called();

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while ((predictor.background_error_count.load() == 0 ||
          background_error_notification_count.load() == 0) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }

  require(predictor.background_error_count.load() == 1,
          "training exception must increment the background error count");
  require(!predictor.is_enabled(),
          "background exception must disable the predictor");
  require(background_error_notification_count.load() == 1,
          "background exception must notify the OSD status adapter");
  predictor.set_enabled(true);
  require(predictor.is_enabled(),
          "enable must reset and recover a disabled predictor");
  require(predictor.background_error_count.load() == 0,
          "successful recovery reset must clear the background error count");

  predictor.train_running.store(false);
  predictor.train_queue_cv.notify_all();
  predictor.train_thread.join();
}

void test_fixed_baseline_configuration()
{
  require(NUM_FEATURES == 5, "baseline must expose exactly five features");
  require(HP_ARF_N_MODELS == 25, "baseline must use 25 ARF trees");
  require(HP_ARF_MAX_FEATURES == NUM_FEATURES,
          "each split must consider all baseline features");
  require_close(HP_HOT_PREDICT_THRESHOLD, 0.50,
                "prediction threshold must remain fixed");
  require(HP_FUTURE_ACCESS_OTSU_BIN_COUNT == 2000,
          "future-access histogram capacity must match V2");
  require_close(HP_FUTURE_ACCESS_OTSU_BIN_WIDTH, 0.01,
                "future-access histogram width must match V2");
  require(HP_EXPIRY_MAINTENANCE_BATCH_SIZE == 1000,
          "expiry maintenance must use a bounded batch");
}

void test_feature_encoding()
{
  PredictionSample sample = make_sample(1, 17);
  sample.heat_after_current_access = 300.0;
  sample.future_access_threshold_at_prediction = 4;
  sample.past_window_access_count = 1;
  sample.short_window_access_count = 2;
  sample.tracked_access_count_after_current_access = 1;

  const auto& cold_start = hp_to_features(sample);
  require(cold_start.size() == NUM_FEATURES,
          "feature vector must match NUM_FEATURES");
  require_close(
      cold_start[0],
      hp_log2p1(1.0) - hp_log2p1(4.0),
      "feature 0 must encode past-access margin relative to K");
  require_close(cold_start[1], 0.0,
                "first access must use the reserved interval encoding");
  require_close(cold_start[2], hp_log2p1(300.0),
                "feature 2 must encode current total heat");
  require_close(
      cold_start[3],
      hp_log2p1(10.0) - hp_log2p1(4.0),
      "feature 3 must project the two-second access rate over ten seconds");
  require_close(cold_start[4], hp_log2p1(2.0),
                "feature 4 must encode the short-window access count");

  sample.tracked_access_count_after_current_access = 2;
  sample.time_since_previous_access_ns = 3ULL * 1000 * 1000 * 1000;
  const auto& repeated = hp_to_features(sample);
  require_close(repeated[1], 1.0 + hp_log2p1(3.0),
                "repeat access must encode the previous interval");
}

void test_short_access_window_excludes_current_io()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  EvaluationQueue queue(
      1000 * second_ns,
      16,
      100.0,
      10 * second_ns,
      32,
      2 * second_ns);

  auto first = queue.begin_prediction(make_sample(1, 7), 0);
  require(first.sample.short_window_access_count == 0,
          "first access must see an empty short window");
  queue.complete_prediction(std::move(*first.ticket), 0.0, 0);

  auto second = queue.begin_prediction(make_sample(2, 7), second_ns);
  require(second.sample.short_window_access_count == 1,
          "current access must not count itself in the short window");
  queue.complete_prediction(std::move(*second.ticket), 0.0, 0);

  auto third = queue.begin_prediction(
      make_sample(3, 7), second_ns + second_ns / 2);
  require(third.sample.short_window_access_count == 2,
          "short window must retain both earlier accesses");
  queue.complete_prediction(std::move(*third.ticket), 0.0, 0);

  auto fourth = queue.begin_prediction(make_sample(4, 7), 3 * second_ns);
  require(fourth.sample.short_window_access_count == 1,
          "events on or before the left boundary must expire");
  queue.complete_prediction(std::move(*fourth.ticket), 0.0, 0);

  EvaluationQueue idle_queue(
      1000 * second_ns,
      16,
      100.0,
      10 * second_ns,
      32,
      2 * second_ns);
  auto idle = idle_queue.begin_prediction(make_sample(1, 9), 0);
  idle_queue.cancel_prediction(std::move(*idle.ticket), 0);
  require(idle_queue.status(0).lru_count == 0,
          "a recent access must protect its object from idle eviction");
  auto maintenance = idle_queue.maintain_expiry(2 * second_ns);
  require(maintenance.processed,
          "expiry maintenance must wake for a short-window deadline");
  require(idle_queue.status(2 * second_ns).lru_count == 0,
          "the strict ten-second access window must still protect the object");
  idle_queue.maintain_expiry(10 * second_ns);
  require(idle_queue.status(10 * second_ns).lru_count == 1,
          "long-window expiry must release an otherwise idle object");
}

void test_future_access_threshold_lifecycle()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  HpFutureAccessThreshold threshold(
      4,  // minimum positive objects
      2,  // object changes before recompute
      second_ns);

  auto initial = threshold.status();
  require(initial.current_threshold == 1,
          "sparse future-access threshold must start at one");
  require(initial.state == HpThresholdState::sparse,
          "future-access threshold must start sparse");

  threshold.update_object_count(0, 2, second_ns);
  threshold.update_object_count(0, 8, second_ns);
  threshold.update_object_count(0, 2, second_ns);
  threshold.update_object_count(0, 8, second_ns);
  auto tracking = threshold.status();
  require(tracking.positive_object_count == 4,
          "threshold population must contain one vote per positive object");
  require(tracking.state == HpThresholdState::tracking,
          "two positive count groups must publish an Otsu threshold");
  require(tracking.current_threshold > 2 &&
              tracking.current_threshold <= 8,
          "published threshold must separate the count groups");

  threshold.update_object_count(2, 0, 2 * second_ns);
  require(threshold.status().positive_object_count == 3,
          "zero count must remove the object's positive vote");
  require(threshold.status().zero_observation_count == 1,
          "positive-to-zero transitions must be counted");
  require(threshold.status().state == HpThresholdState::sparse,
          "invalid population must immediately return to sparse");
  require(threshold.current_threshold() == 1,
          "sparse state must publish K=1");

  threshold.clear();
  auto cleared = threshold.status();
  require(cleared.current_threshold == 1 &&
              cleared.positive_object_count == 0 &&
              cleared.zero_observation_count == 0,
          "reset must clear votes, counters, and publication state");
}

void test_threshold_only_maintenance_reports_status_change()
{
  HpFutureAccessThreshold threshold(32, 100, 1000);
  for (uint64_t key = 1; key <= 32; ++key) {
    threshold.update_object_count(0, key <= 16 ? 1 : 8, 0);
  }
  require(threshold.status().state ==
              HpThresholdState::tracking,
          "becoming ready must publish the raw Otsu threshold immediately");
  const uint64_t first_threshold = threshold.current_threshold();

  threshold.update_object_count(1, 0, 1);
  require(threshold.status().state == HpThresholdState::sparse,
          "losing readiness must immediately publish sparse K=1");
  require(first_threshold > 1 && threshold.current_threshold() == 1,
          "raw Otsu publication must not retain or smooth an invalid K");
}

void test_foreground_drains_due_backlog_before_current_access()
{
  constexpr uint64_t window_ns = 10;
  constexpr size_t due_count = HP_EXPIRY_MAINTENANCE_BATCH_SIZE + 1;
  EvaluationQueue queue(
      1000, due_count + 1, 100.0, window_ns, due_count + 1);

  for (size_t index = 0; index < due_count; ++index) {
    auto pending = queue.begin_prediction(
        make_sample(index + 1, 77), 0);
    require(pending.ticket.has_value(),
            "backlog fixture must reserve every due item");
    queue.complete_prediction(std::move(*pending.ticket), 0.0, 0);
  }

  auto current = queue.begin_prediction(
      make_sample(due_count + 1, 77), window_ns);
  require(current.evaluated.size() == due_count,
          "foreground must drain every overdue batch before current access");
  require(current.evaluated.back().future_window_access_count == 0,
          "current deadline access must not leak into the last overdue item");
    queue.cancel_prediction(std::move(*current.ticket), window_ns);
}

void test_deadline_microbatches_preserve_threshold_timeline()
{
  constexpr uint64_t millisecond_ns = 1000ULL * 1000;
  constexpr uint64_t window_ns = 10 * millisecond_ns;
  EvaluationQueue queue(
      1000 * window_ns,
      256,
      100.0,
      window_ns,
      256,
      2 * millisecond_ns);

  auto first_target = queue.begin_prediction(make_sample(1, 1), 0);
  require(first_target.ticket.has_value(),
          "first microbatch target must reserve an evaluation slot");
  queue.complete_prediction(std::move(*first_target.ticket), 0.0, 0);

  uint64_t sequence = 2;
  for (uint64_t key = 100; key < 116; ++key) {
    auto access = queue.begin_prediction(
        make_sample(sequence++, key), millisecond_ns);
    queue.complete_prediction(std::move(*access.ticket), 0.0, 0);
  }
  for (uint64_t key = 116; key < 132; ++key) {
    for (uint64_t repeat = 0; repeat < 8; ++repeat) {
      auto access = queue.begin_prediction(
          make_sample(sequence++, key), millisecond_ns);
      queue.complete_prediction(std::move(*access.ticket), 0.0, 0);
    }
  }

  const uint64_t second_target_sequence = sequence++;
  auto second_target = queue.begin_prediction(
      make_sample(second_target_sequence, 2), 2 * millisecond_ns);
  require(second_target.ticket.has_value(),
          "second microbatch target must reserve an evaluation slot");
  queue.complete_prediction(std::move(*second_target.ticket), 0.0, 0);

  auto maintenance = queue.maintain_expiry(12 * millisecond_ns, 256);
  const auto first = std::find_if(
      maintenance.evaluated.begin(),
      maintenance.evaluated.end(),
      [](const EvaluatedSample& sample) {
        return sample.item.io_sequence == 1;
      });
  const auto second = std::find_if(
      maintenance.evaluated.begin(),
      maintenance.evaluated.end(),
      [second_target_sequence](const EvaluatedSample& sample) {
        return sample.item.io_sequence == second_target_sequence;
      });
  require(first != maintenance.evaluated.end() &&
              second != maintenance.evaluated.end(),
          "both deadline microbatch targets must be evaluated");
  require(first->future_window_access_threshold > 1,
          "the 10ms deadline must retain the earlier two-group threshold");
  require(second->future_window_access_threshold == 1,
          "the 12ms deadline must observe the later sparse threshold");
}

void test_distribution_window_uses_compact_histogram_entries()
{
  HpIntegerQuantileWindow window(4);
  using RetainedEntry = typename decltype(window.order)::value_type;
  require(sizeof(RetainedEntry) <= sizeof(uint16_t),
          "reporting window must retain only one compact histogram bin per sample");

  window.insert(1);
  window.insert(2);
  window.insert(4);
  window.insert(8);
  window.insert(16);
  const auto summary = window.summary();
  require(summary.count == 4,
          "reporting histogram must preserve sliding-window capacity");
  require(std::abs(summary.p50 - 4.0) <= 1.0 &&
              std::abs(summary.p99 - 16.0) <= 1.0,
          "log-histogram quantiles must remain close to retained integers");
}

void test_future_access_threshold_capacity_and_clamp()
{
  HpFutureAccessThreshold threshold(
      32,
      100,
      1000);
  const uint64_t above_range =
      HpFutureAccessThreshold::maximum_representable_count() + 1;

  threshold.update_object_count(0, above_range, 0);
  auto status = threshold.status();
  require(status.positive_object_count == 1,
          "positive count must add one object vote");
  require(status.upper_clamped_object_count == 1,
          "counts above the histogram must be clamped exactly");

  threshold.update_object_count(above_range, 1, 1);
  status = threshold.status();
  require(status.positive_object_count == 1,
          "moving a vote must not change object population size");
  require(status.upper_clamped_object_count == 0,
          "moving below the upper bound must clear clamp telemetry");

  threshold.update_object_count(1, 0, 2);
  status = threshold.status();
  require(status.positive_object_count == 0 &&
              status.zero_observation_count == 1,
          "moving to zero must remove the vote and count the transition");
}

void test_evaluation_queue_lifecycle()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(
      1000,
      2,
      100.0,
      window_ns,
      8,
      window_ns);

  auto first = queue.begin_prediction(make_sample(1, 7), 0);
  require(first.ticket.has_value(),
          "first prediction must reserve an evaluation slot");
  require(first.sample.tracked_access_count_after_current_access == 1,
          "first access must initialize object state");
  require(first.sample.past_window_access_count == 0,
          "first access must see an empty past window");
  require(first.sample.future_access_threshold_at_prediction == 1,
          "sparse mode must store K=1");
  require(queue.complete_prediction(
              std::move(*first.ticket), 0.8, 1).empty(),
          "label must not complete before the future window");

  auto second = queue.begin_prediction(make_sample(2, 7), 5);
  require(second.ticket.has_value(),
          "repeat access must reserve another evaluation slot");
  require(second.sample.tracked_access_count_after_current_access == 2,
          "repeat access must advance the object access count");
  require(second.sample.past_window_access_count == 1,
          "repeat access must see the first pending item");
  require(queue.complete_prediction(
              std::move(*second.ticket), 0.7, 1).empty(),
          "second label must wait for its own deadline");

  auto first_expiry = queue.maintain_expiry(window_ns);
  require(first_expiry.processed,
          "deadline maintenance must process due evaluations");
  require(first_expiry.evaluated.size() == 1,
          "only the first sample must expire at the first deadline");
  require(first_expiry.evaluated[0].future_window_access_count == 1,
          "future access count must include the repeat access");
  require(first_expiry.evaluated[0].label == 1,
          "the repeated object must be labeled hot");

  auto second_expiry = queue.maintain_expiry(5 + window_ns);
  require(second_expiry.evaluated.size() == 1,
          "second sample must expire at its own deadline");
  const auto status = queue.status(5 + window_ns);
  require(status.pending_io_count == 0,
          "all completed labels must leave the pending queue");
  require(status.awaiting_prediction_count == 0,
          "normal prediction completion must not leave late results");
  require(status.lru_count == 1,
          "an unprotected object must enter the idle LRU");
}

void test_eq_recent_counts_drive_threshold_before_labels_complete()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(
      1000,
      64,
      100.0,
      window_ns,
      256);

  for (uint64_t key = 1; key <= 32; ++key) {
    auto first = queue.begin_prediction(make_sample(key, key), 0);
    require(first.ticket.has_value(),
            "recent-count fixture must admit the first access");
    queue.complete_prediction(std::move(*first.ticket), 0.0, 0);
  }
  require(queue.future_access_threshold.status().positive_object_count == 32,
          "EQ admission must publish one recent-count vote per object");
  require(queue.future_access_threshold.status().state ==
              HpThresholdState::sparse,
          "uniform recent counts must not fabricate an Otsu split");

  uint64_t sequence = 33;
  for (uint64_t repeat = 0; repeat < 7; ++repeat) {
    for (uint64_t key = 17; key <= 32; ++key) {
      auto access = queue.begin_prediction(
          make_sample(sequence++, key), repeat + 1);
      require(access.ticket.has_value(),
              "recent-count fixture must admit repeated accesses");
      queue.complete_prediction(std::move(*access.ticket), 0.0, 0);
    }
  }
  const auto threshold = queue.future_access_threshold.status();
  require(threshold.positive_object_count == 32,
          "repeated accesses must replace rather than duplicate object votes");
  require(threshold.state == HpThresholdState::tracking,
          "recent EQ counts must publish K before any future label completes");
  require(threshold.current_threshold > 1,
          "recent EQ count groups must produce a nontrivial K");
}

void test_eq_cancellation_removes_recent_count_vote()
{
  EvaluationQueue queue(1000, 8, 100.0, 10, 8);
  auto pending = queue.begin_prediction(make_sample(1, 77), 0);
  require(pending.ticket.has_value(),
          "cancellation fixture must admit the access");
  require(queue.future_access_threshold.status().positive_object_count == 1,
          "admitted access must immediately create a recent-count vote");

  queue.cancel_prediction(std::move(*pending.ticket), 1);
  const auto status = queue.future_access_threshold.status();
  require(status.positive_object_count == 1,
          "prediction cancellation must not erase the access-window vote");

  queue.maintain_expiry(10);
  require(queue.future_access_threshold.status().positive_object_count == 0,
          "the access-window vote must expire at its own ten-second deadline");
}

void test_evaluation_capacity_drop()
{
  EvaluationQueue queue(1000, 2, 100.0, 10, 1);
  auto first = queue.begin_prediction(make_sample(1, 1), 0);
  require(first.ticket.has_value(),
          "first item must consume the only evaluation slot");

  auto dropped = queue.begin_prediction(make_sample(2, 2), 1);
  require(!dropped.ticket.has_value(),
          "capacity overflow must decline the second reservation");
  require(queue.status(1).evaluation_drop_count == 1,
          "capacity overflow must increment the drop counter");
  require(queue.future_access_threshold.status().positive_object_count == 2,
          "capacity overflow must not erase the rejected I/O from K");
  queue.cancel_prediction(std::move(*first.ticket), 1);
  require(queue.status(1).pending_io_count == 0,
          "cancelling a prediction must release its slot");
}

void test_awaiting_prediction_consumes_evaluation_capacity()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(1000, 2, 100.0, window_ns, 1);

  auto awaiting = queue.begin_prediction(make_sample(1, 1), 0);
  require(awaiting.ticket.has_value(),
          "first item must reserve the only evaluation slot");
  auto expiry = queue.maintain_expiry(window_ns);
  require(expiry.expired_evaluation_count == 1,
          "first item must finish its label at the deadline");
  require(queue.status(window_ns).awaiting_prediction_count == 1,
          "label-complete item must remain awaiting its prediction");
  require(queue.status(window_ns).otsu_zero_observation_count == 1,
          "label completion must update threshold observations before prediction");

  auto dropped = queue.begin_prediction(
      make_sample(2, 2), window_ns + 1);
  require(!dropped.ticket.has_value(),
          "awaiting prediction must continue consuming total EQ capacity");

  auto completed = queue.complete_prediction(
      std::move(*awaiting.ticket), 0.0, 0);
  require(completed.size() == 1,
          "late prediction must finalize its already-labeled item");

  auto accepted = queue.begin_prediction(
      make_sample(3, 3), window_ns + 2);
  require(accepted.ticket.has_value(),
          "finalizing the awaiting item must release total EQ capacity");
  queue.cancel_prediction(
      std::move(*accepted.ticket), window_ns + 2);
}

void test_deadline_window_threshold_controls_label()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(1000, 256, 100.0, window_ns, 256);

  auto target = queue.begin_prediction(make_sample(1, 5), 0);
  require(target.sample.future_access_threshold_at_prediction == 1,
          "sparse prediction context must initially use K=1");
  queue.complete_prediction(std::move(*target.ticket), 0.0, 0);

  uint64_t sequence = 2;
  for (uint64_t key = 100; key < 116; ++key) {
    auto access = queue.begin_prediction(
        make_sample(sequence++, key), 1);
    queue.complete_prediction(std::move(*access.ticket), 0.0, 0);
  }
  for (uint64_t key = 116; key < 132; ++key) {
    for (uint64_t repeat = 0; repeat < 8; ++repeat) {
      auto access = queue.begin_prediction(
          make_sample(sequence++, key), 1);
      queue.complete_prediction(std::move(*access.ticket), 0.0, 0);
    }
  }
  auto repeat = queue.begin_prediction(
      make_sample(sequence++, 5), 5);
  queue.complete_prediction(std::move(*repeat.ticket), 0.0, 0);

  auto expiry = queue.maintain_expiry(window_ns);
  require(expiry.evaluated.size() == 1,
          "target item must complete at its own deadline");
  require(expiry.evaluated[0].future_window_access_count == 1,
          "target item must observe its one repeat access");
  require(expiry.evaluated[0].label == 0,
          "deadline window K must override the stale prediction-time K=1");
}

void test_reset_counts_all_incomplete_evaluations()
{
  constexpr uint64_t window_ns = HP_FUTURE_LABEL_WINDOW_NS;
  HeatPredictor predictor;

  auto awaiting = predictor.eq->begin_prediction(
      make_sample(1, 1), 0);
  require(awaiting.ticket.has_value(),
          "first reset fixture item must reserve an EQ slot");
  predictor.eq->maintain_expiry(window_ns);
  require(predictor.eq->status(window_ns).awaiting_prediction_count == 1,
          "first reset fixture item must await prediction");

  auto pending = predictor.eq->begin_prediction(
      make_sample(2, 2), window_ns + 1);
  require(pending.ticket.has_value(),
          "second reset fixture item must remain deadline-pending");
  require(predictor.reset() == 2,
          "reset must count pending and awaiting evaluations");
}

void test_status_snapshot_preserves_sample_accounting()
{
  constexpr uint64_t microsecond_ns = 1000;
  constexpr size_t producer_count = 4;
  constexpr size_t predictions_per_producer = 5000;
  HeatPredictor predictor;
  predictor.eq = std::make_unique<EvaluationQueue>(
      1000 * 1000 * microsecond_ns,
      producer_count * predictions_per_producer,
      100.0,
      100 * microsecond_ns,
      producer_count * predictions_per_producer,
      50 * microsecond_ns);

  std::atomic<size_t> active_producers{producer_count};
  std::atomic<uint64_t> inconsistent_snapshots{0};
  std::thread observer([&] {
    while (active_producers.load(std::memory_order_acquire) != 0) {
      const auto status = predictor.status().evaluation;
      const uint64_t accounted =
          status.labeled_io_total +
          status.pending_io_count +
          status.awaiting_prediction_count +
          status.eval_drop_count;
      if (status.io_count != accounted) {
        inconsistent_snapshots.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });

  std::vector<std::thread> producers;
  producers.reserve(producer_count);
  for (size_t producer = 0; producer < producer_count; ++producer) {
    producers.emplace_back([&, producer] {
      for (size_t index = 0; index < predictions_per_producer; ++index) {
        const uint64_t object = producer * 128 + index % 128;
        predictor.predict(1, object, object, nullptr);
      }
      active_producers.fetch_sub(1, std::memory_order_release);
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  observer.join();

  require(inconsistent_snapshots.load(std::memory_order_relaxed) == 0,
          "status must atomically account for labeled, queued, and dropped I/O");
}

void test_expiry_maintenance_is_batched()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(1000, 8, 100.0, window_ns, 8);

  for (uint64_t sequence = 1; sequence <= 3; ++sequence) {
    auto pending = queue.begin_prediction(
        make_sample(sequence, sequence), 0);
    require(pending.ticket.has_value(),
            "test item must reserve an evaluation slot");
    require(queue.complete_prediction(
                std::move(*pending.ticket), 0.0, 0).empty(),
            "test labels must remain pending until their deadline");
  }

  auto first_batch = queue.maintain_expiry(1000, 2);
  require(first_batch.expired_evaluation_count == 2,
          "first maintenance pass must honor the requested batch limit");
  require(first_batch.evaluated.size() == 2,
          "first maintenance pass must return only one bounded batch");
  require(first_batch.next_schedule.state ==
              EvaluationQueue::ExpiryScheduleState::due,
          "remaining due work must stay immediately schedulable");
  auto second_batch = queue.maintain_expiry(1000, 2);
  require(second_batch.expired_evaluation_count == 1,
          "second maintenance pass must drain the remaining item");
  require(queue.status(1000).pending_io_count == 0,
          "all due items must finish after repeated bounded passes");
}

void test_deadline_access_is_excluded_from_future_window()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(1000, 8, 100.0, window_ns, 8);
  auto first = queue.begin_prediction(make_sample(1, 9), 0);
  require(first.ticket.has_value(),
          "deadline fixture must reserve its first item");
  queue.complete_prediction(std::move(*first.ticket), 0.0, 0);

  auto at_deadline = queue.begin_prediction(make_sample(2, 9), window_ns);
  require(at_deadline.evaluated.size() == 1,
          "foreground access must expire the old item before accounting itself");
  require(at_deadline.evaluated[0].future_window_access_count == 0,
          "access exactly at the deadline must be outside the old window");
  require(at_deadline.evaluated[0].label == 0,
          "zero future accesses must be cold under sparse K=1");
  queue.cancel_prediction(
      std::move(*at_deadline.ticket), window_ns);
}

void test_arf_probability_distribution()
{
  using Model = ARFClassifier<
      NUM_FEATURES,
      2,
      DetectorFactory<ADWIN<5>, HP_ARF_WARNING_DELTA_PERMILLE>,
      DetectorFactory<ADWIN<5>, HP_ARF_DRIFT_DELTA_PERMILLE>>;
  Model model(
      3,
      NUM_FEATURES,
      HP_ARF_SEED,
      HP_ARF_GRACE_PERIOD,
      HP_ARF_LAMBDA,
      HP_ARF_DELTA,
      HP_ARF_TAU,
      HP_ARF_MAX_SHARE_TO_SPLIT,
      HP_ARF_MIN_BRANCH_FRACTION);

  const std::vector<double> cold = {-2.0, 4.0, 2.0, -1.0, 0.0};
  const std::vector<double> hot = {2.0, 1.0, 8.0, 2.0, 2.0};
  for (int index = 0; index < 1000; ++index) {
    model.learn_one(cold, 0, 1.0);
    model.learn_one(hot, 1, 1.0);
  }

  const auto probability = model.predict_proba_one(hot);
  require(probability.size() == 2,
          "binary classifier must return two probabilities");
  require(std::isfinite(probability[0]) && std::isfinite(probability[1]),
          "classifier probabilities must be finite");
  require_close(probability[0] + probability[1], 1.0,
                "classifier probabilities must sum to one");
  require(probability[1] > probability[0],
          "trained hot sample must favor the hot class");
}

void test_production_model_disables_arf_adaptation()
{
  auto telemetry = std::make_shared<ArfAdaptationTelemetry>();
  std::unique_ptr<Classifier> model(
      HeatPredictor::make_model(telemetry));
  const std::vector<double> features(NUM_FEATURES, 1.0);

  for (int index = 0; index < 4000; ++index) {
    model->learn_one(features, 0, 1.0);
  }
  for (int index = 0; index < 4000; ++index) {
    model->learn_one(features, 1, 1.0);
  }

  const auto stats = telemetry->snapshot();
  require(stats.warning_count == 0,
          "production ARF must not create warning background trees");
  require(stats.drift_count == 0,
          "production ARF must not replace trees after drift");
  require(stats.background_promotion_count == 0 &&
              stats.background_discard_count == 0 &&
              stats.background_training_update_count == 0 &&
              stats.active_background_count == 0,
          "disabled adaptation must leave all background telemetry at zero");

  const auto probability = model->predict_proba_one(features);
  require(probability.size() == 2 &&
              std::isfinite(probability[0]) &&
              std::isfinite(probability[1]),
          "disabling drift must preserve online training and prediction");
}

} // namespace

int main()
{
  test_training_shutdown_finishes_only_current_batch();
  test_training_exception_disables_predictor_without_terminating();
  test_fixed_baseline_configuration();
  test_feature_encoding();
  test_short_access_window_excludes_current_io();
  test_future_access_threshold_lifecycle();
  test_threshold_only_maintenance_reports_status_change();
  test_future_access_threshold_capacity_and_clamp();
  test_evaluation_queue_lifecycle();
  test_production_model_disables_arf_adaptation();
  test_eq_recent_counts_drive_threshold_before_labels_complete();
  test_eq_cancellation_removes_recent_count_vote();
  test_evaluation_capacity_drop();
  test_awaiting_prediction_consumes_evaluation_capacity();
  test_deadline_window_threshold_controls_label();
  test_reset_counts_all_incomplete_evaluations();
  test_status_snapshot_preserves_sample_accounting();
  test_expiry_maintenance_is_batched();
  test_deadline_access_is_excluded_from_future_window();
  test_foreground_drains_due_backlog_before_current_access();
  test_distribution_window_uses_compact_histogram_entries();
  test_deadline_microbatches_preserve_threshold_timeline();
  test_arf_probability_distribution();
  std::cout << "PASS: heat predictor algorithm probe" << std::endl;
  return 0;
}
