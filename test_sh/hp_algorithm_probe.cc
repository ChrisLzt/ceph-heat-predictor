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
#include "heatpredictor/hp_expiry_heap.h"
#include "heatpredictor/hp_features.h"
#include "heatpredictor/hp_score_otsu_histogram.h"
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
      0.0,
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
  require(NUM_FEATURES == 3, "baseline must expose exactly three features");
  require(HP_ARF_N_MODELS == 25, "baseline must use 25 ARF trees");
  require(HP_ARF_MAX_FEATURES == NUM_FEATURES,
          "each split must consider all baseline features");
  require_close(HP_HOT_PREDICT_THRESHOLD, 0.50,
                "prediction threshold must remain fixed");
  require_close(HP_OTSU_EMA_ALPHA, 0.10,
                "Otsu EMA gain must remain fixed");
  require(HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT == 2000,
          "score histogram capacity must match the selected baseline");
  require_close(HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH, 0.01,
                "score histogram width must match the selected baseline");
  require(HP_EXPIRY_MAINTENANCE_BATCH_SIZE == 1000,
          "expiry maintenance must use a bounded batch");
}

void test_feature_encoding()
{
  PredictionSample sample = make_sample(1, 17);
  sample.heat_after_current_access = 300.0;
  sample.heat_label_threshold_at_prediction = 100.0;
  sample.tracked_access_count = 1;

  const auto& cold_start = hp_to_features(sample);
  require(cold_start.size() == NUM_FEATURES,
          "feature vector must match NUM_FEATURES");
  require_close(
      cold_start[0],
      hp_log2p1(300.0) - hp_log2p1(100.0),
      "feature 0 must encode heat margin");
  require_close(cold_start[1], 0.0,
                "first access must use the reserved interval encoding");
  require_close(cold_start[2], hp_log2p1(300.0),
                "feature 2 must encode current total heat");

  sample.tracked_access_count = 2;
  sample.time_since_previous_access_ns = 3ULL * 1000 * 1000 * 1000;
  const auto& repeated = hp_to_features(sample);
  require_close(repeated[1], 1.0 + hp_log2p1(3.0),
                "repeat access must encode the previous interval");
}

void test_expiry_heap()
{
  HpExpiryHeap heap;
  heap.upsert(1, 30);
  heap.upsert(2, 10);
  heap.upsert(3, 20);
  require(heap.earliest_deadline_ns() == 10,
          "heap must expose the earliest deadline");
  require(!heap.due_key(9).has_value(),
          "heap must not expire a future entry");
  require(heap.due_key(10) == 2,
          "heap must return the due key");

  heap.upsert(1, 5);
  require(heap.due_key(5) == 1,
          "upsert must restore heap order");
  require(heap.erase(1), "erase must remove an existing key");
  require(!heap.erase(1), "erase must report a missing key");
  require(heap.earliest_deadline_ns() == 10,
          "erase must restore the next deadline");
}

void test_score_otsu_histogram()
{
  HpScoreOtsuHistogram histogram;
  const double decay = hp_heat_decay_log_factor_per_ns(
      HP_HEAT_DECAY_HORIZON_NS);
  const double minimum_score = HpScoreOtsuHistogram::score_for_heat_at(
      HP_OTSU_TOTAL_HEAT_MIN, 0, decay);
  histogram.advance_lower_bound(minimum_score);

  std::vector<HpScoreOtsuHistogram::AbsoluteBin> bins;
  bins.reserve(HP_OTSU_MIN_VOTES);
  for (size_t index = 0; index < HP_OTSU_MIN_VOTES; ++index) {
    const double heat = index < HP_OTSU_MIN_VOTES / 2 ? 40.0 : 1000.0;
    bins.push_back(histogram.insert(
        HpScoreOtsuHistogram::score_for_heat_at(heat, 0, decay)));
  }

  require(histogram.size() == HP_OTSU_MIN_VOTES,
          "histogram must count one vote per inserted object");
  require(histogram.bin_count() == 2,
          "two heat populations must occupy two bins");
  const auto result = histogram.otsu_result();
  require(result.has_value(),
          "two populated heat groups must produce an Otsu threshold");
  const double threshold = HpScoreOtsuHistogram::heat_for_score_at(
      result->threshold_score, 0, decay);
  require(threshold > 40.0 && threshold < 1000.0,
          "Otsu threshold must separate the two heat groups");

  for (auto bin : bins) {
    histogram.erase(bin);
  }
  require(histogram.empty(), "erasing all votes must empty the histogram");
}

void test_evaluation_queue_lifecycle()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(
      1000,
      2,
      100.0,
      100.0,
      window_ns,
      8);

  auto first = queue.begin_prediction(make_sample(1, 7), 0);
  require(first.ticket.has_value(),
          "first prediction must reserve an evaluation slot");
  require(first.sample.tracked_access_count == 1,
          "first access must initialize object state");
  require(queue.complete_prediction(
              std::move(*first.ticket), 0.8, 1).empty(),
          "label must not complete before the future window");

  auto second = queue.begin_prediction(make_sample(2, 7), 5);
  require(second.ticket.has_value(),
          "repeat access must reserve another evaluation slot");
  require(second.sample.tracked_access_count == 2,
          "repeat access must advance the object access count");
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

void test_evaluation_capacity_drop()
{
  EvaluationQueue queue(1000, 2, 100.0, 100.0, 10, 1);
  auto first = queue.begin_prediction(make_sample(1, 1), 0);
  require(first.ticket.has_value(),
          "first item must consume the only evaluation slot");

  auto dropped = queue.begin_prediction(make_sample(2, 2), 1);
  require(!dropped.ticket.has_value(),
          "capacity overflow must decline the second reservation");
  require(queue.status(1).evaluation_drop_count == 1,
          "capacity overflow must increment the drop counter");
  queue.cancel_prediction(std::move(*first.ticket));
  require(queue.status(1).pending_io_count == 0,
          "cancelling a prediction must release its slot");
}

void test_awaiting_prediction_consumes_evaluation_capacity()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(1000, 2, 100.0, 100.0, window_ns, 1);

  auto awaiting = queue.begin_prediction(make_sample(1, 1), 0);
  require(awaiting.ticket.has_value(),
          "first item must reserve the only evaluation slot");
  auto expiry = queue.maintain_expiry(window_ns);
  require(expiry.expired_evaluation_count == 1,
          "first item must finish its label at the deadline");
  require(queue.status(window_ns).awaiting_prediction_count == 1,
          "label-complete item must remain awaiting its prediction");

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
  queue.cancel_prediction(std::move(*accepted.ticket));
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

void test_expiry_maintenance_is_batched()
{
  constexpr uint64_t window_ns = 10;
  EvaluationQueue queue(1000, 8, 100.0, 100.0, window_ns, 8);

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
  require(queue.threshold_entries_by_key.size() == 1,
          "Otsu expiry cleanup must honor the same batch limit");

  auto second_batch = queue.maintain_expiry(1000, 2);
  require(second_batch.expired_evaluation_count == 1,
          "second maintenance pass must drain the remaining item");
  require(queue.status(1000).pending_io_count == 0,
          "all due items must finish after repeated bounded passes");
  require(queue.threshold_entries_by_key.empty(),
          "repeated maintenance must drain remaining expired Otsu votes");
}

void test_otsu_recomputes_after_expiry_backlog_is_drained()
{
  EvaluationQueue queue;
  for (uint64_t sequence = 1; sequence <= HP_OTSU_MIN_VOTES; ++sequence) {
    auto item = queue.begin_prediction(
        make_sample(sequence, sequence), 0);
    require(item.ticket.has_value(),
            "Otsu fixture item must reserve an evaluation slot");
    queue.cancel_prediction(std::move(*item.ticket));
  }

  require(queue.threshold_entries_by_key.size() == HP_OTSU_MIN_VOTES,
          "Otsu fixture must contain the minimum object vote count");
  const uint64_t expiry_time_ns = HP_HEAT_DECAY_HORIZON_NS + 1000;
  queue.advance_otsu_history(expiry_time_ns, 10);
  require(queue.threshold_entries_by_key.size() ==
              HP_OTSU_MIN_VOTES - 10,
          "first Otsu cleanup pass must leave a due backlog");
  require(queue.last_otsu_recompute_time_ns == 0,
          "Otsu must not recompute against a partially cleaned backlog");

  while (!queue.threshold_entries_by_key.empty()) {
    queue.advance_otsu_history(expiry_time_ns, 10);
  }
  require(queue.last_otsu_recompute_time_ns == expiry_time_ns,
          "Otsu must recompute immediately after the backlog is drained");
}

void test_status_does_not_advance_otsu_state()
{
  constexpr uint64_t second_ns = 1000ULL * 1000 * 1000;
  EvaluationQueue queue;
  auto pending = queue.begin_prediction(make_sample(1, 1), 0);
  require(pending.ticket.has_value(),
          "test item must reserve an evaluation slot");
  require(queue.threshold_entries_by_key.size() == 1,
          "first object must contribute one Otsu vote");

  const auto status = queue.status(60 * second_ns);
  require(status.otsu_histogram_vote_count == 1,
          "status must report the stored Otsu vote without expiring it");
  require(queue.threshold_entries_by_key.size() == 1,
          "status must not mutate Otsu retention state");
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

  const std::vector<double> cold = {-2.0, 4.0, 2.0};
  const std::vector<double> hot = {2.0, 1.0, 8.0};
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

} // namespace

int main()
{
  test_training_shutdown_finishes_only_current_batch();
  test_training_exception_disables_predictor_without_terminating();
  test_fixed_baseline_configuration();
  test_feature_encoding();
  test_expiry_heap();
  test_score_otsu_histogram();
  test_evaluation_queue_lifecycle();
  test_evaluation_capacity_drop();
  test_awaiting_prediction_consumes_evaluation_capacity();
  test_reset_counts_all_incomplete_evaluations();
  test_expiry_maintenance_is_batched();
  test_otsu_recomputes_after_expiry_backlog_is_drained();
  test_status_does_not_advance_otsu_state();
  test_arf_probability_distribution();
  std::cout << "PASS: heat predictor algorithm probe" << std::endl;
  return 0;
}
