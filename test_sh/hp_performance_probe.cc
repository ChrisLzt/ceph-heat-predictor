#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "heatpredictor/heat_predictor.h"
#include "heatpredictor/hp_quantile_window.h"

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
    const char *function)
{
  (void)function;
  std::cerr << "ceph_assert failed: " << assertion
            << " at " << file << ":" << line << std::endl;
  std::abort();
}

} // namespace ceph

namespace {

using Clock = std::chrono::steady_clock;

std::vector<double> feature(double a, double b, double c, double d, double e)
{
  const std::array<double, 5> values = {a, b, c, d, e};
  std::vector<double> result(NUM_FEATURES, 0.0);
  std::copy_n(
      values.begin(), std::min(values.size(), result.size()), result.begin());
  return result;
}

PredictionSample trace_item(uint64_t io_sequence, uint64_t object_key_hash)
{
  return PredictionSample{
      io_sequence, object_key_hash, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0
  };
}

void train_model(Classifier& model, int sample_count = 3000)
{
  const std::vector<std::vector<double>> samples = {
      feature(0.2, 8.0, 1.0, 0.0, 0.1),
      feature(0.5, 5.0, 2.0, 1.0, 0.2),
      feature(2.0, 2.0, 5.0, 3.0, 0.8),
      feature(4.0, 1.0, 8.0, 6.0, 1.5),
      feature(6.0, 0.5, 10.0, 9.0, 2.0),
      feature(-0.2, 12.0, 0.5, 0.0, 0.05),
  };
  const int labels[] = {0, 0, 1, 1, 1, 0};

  for (int i = 0; i < sample_count; ++i) {
    const size_t idx = static_cast<size_t>(i) % samples.size();
    model.learn_one(samples[idx], labels[idx], labels[idx] ? 3.0 : 1.0);
  }
}

struct ConcurrentEqResult {
  double ns_per_op;
  double checksum;
};

template <typename Window>
double benchmark_quantile_window(
    Window& window,
    int operation_count,
    double *checksum)
{
  const auto start = Clock::now();
  for (int i = 0; i < operation_count; ++i) {
    const uint64_t value =
        (static_cast<uint64_t>(i) * 2654435761ULL) %
        (HP_HEAT_DECAY_HORIZON_NS + 1);
    window.insert(value);
  }
  HpDistributionSummary summary = window.summary();
  const auto end = Clock::now();
  *checksum += summary.count + summary.max + summary.p50 +
      summary.p90 + summary.p95 + summary.p99;
  return std::chrono::duration<double, std::nano>(end - start).count() /
      operation_count;
}

ConcurrentEqResult benchmark_serialized_prediction(
    Classifier& snapshot,
    int thread_count,
    int operations_per_thread)
{
  EvaluationQueue eq;
  std::mutex eq_mutex;
  uint64_t next_index = 0;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  std::vector<double> checksums(static_cast<size_t>(thread_count), 0.0);
  workers.reserve(static_cast<size_t>(thread_count));

  const auto begin = Clock::now();
  for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
    workers.emplace_back([&, thread_id] {
      std::vector<double> proba;
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < operations_per_thread; ++i) {
        std::lock_guard<std::mutex> lock(eq_mutex);
        const uint64_t index = ++next_index;
        const uint64_t now_ns = index * 1000000ULL;
        PredictionSample item = trace_item(index, index % 12000);
        auto evaluated = eq.expire_before_prepare(item, now_ns);
        snapshot.predict_proba_one_into(HeatPredictor::to_feat(item), proba);
        item.predicted_hot_probability = proba.size() > 1 ? proba[1] : 0.0;
        item.predicted_label =
            item.predicted_hot_probability >= HP_HOT_PREDICT_THRESHOLD;
        eq.enqueue(item, now_ns);
        checksums[static_cast<size_t>(thread_id)] += item.predicted_hot_probability;
        for (const auto& sample : evaluated) {
          checksums[static_cast<size_t>(thread_id)] += sample.label;
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }
  const auto end = Clock::now();

  double checksum = 0.0;
  for (double value : checksums) {
    checksum += value;
  }
  const double operation_count =
      static_cast<double>(thread_count) * operations_per_thread;
  return {
      std::chrono::duration<double, std::nano>(end - begin).count() /
          operation_count,
      checksum
  };
}

ConcurrentEqResult benchmark_two_phase_prediction(
    Classifier& snapshot,
    int thread_count,
    int operations_per_thread)
{
  EvaluationQueue eq;
  std::mutex eq_mutex;
  uint64_t next_index = 0;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  std::vector<double> checksums(static_cast<size_t>(thread_count), 0.0);
  workers.reserve(static_cast<size_t>(thread_count));

  const auto begin = Clock::now();
  for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
    workers.emplace_back([&, thread_id] {
      std::vector<double> proba;
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < operations_per_thread; ++i) {
        std::optional<EvaluationQueue::PendingIterator> position;
        std::vector<EvaluatedSample> evaluated;
        PredictionSample item;
        double predict_threshold = HP_HOT_PREDICT_THRESHOLD;
        {
          std::lock_guard<std::mutex> lock(eq_mutex);
          const uint64_t index = ++next_index;
          const uint64_t now_ns = index * 1000000ULL;
          item = trace_item(index, index % 12000);
          evaluated = eq.expire_before_prepare(item, now_ns);
          auto reservation = eq.reserve_prediction(item, now_ns);
          if (reservation.accepted) {
            position = reservation.position;
          }
        }

        snapshot.predict_proba_one_into(HeatPredictor::to_feat(item), proba);
        item.predicted_hot_probability = proba.size() > 1 ? proba[1] : 0.0;
        item.predicted_label = item.predicted_hot_probability >= predict_threshold;
        if (position.has_value()) {
          std::lock_guard<std::mutex> lock(eq_mutex);
          eq.complete_prediction(
              *position, item.predicted_hot_probability, item.predicted_label);
        }

        checksums[static_cast<size_t>(thread_id)] += item.predicted_hot_probability;
        for (const auto& sample : evaluated) {
          checksums[static_cast<size_t>(thread_id)] += sample.label;
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }
  const auto end = Clock::now();

  double checksum = 0.0;
  for (double value : checksums) {
    checksum += value;
  }
  const double operation_count =
      static_cast<double>(thread_count) * operations_per_thread;
  return {
      std::chrono::duration<double, std::nano>(end - begin).count() /
          operation_count,
      checksum
  };
}

} // namespace

int main()
{
  AdaptiveWindowing<5> adwin;
  uint64_t adwin_detection_signature = 0;
  for (int i = 0; i < 10000; ++i) {
    if (adwin.update((i / 250) % 2 == 0 ? 0.0 : 1.0)) {
      adwin_detection_signature =
          adwin_detection_signature * 1315423911ULL +
          static_cast<uint64_t>(i + 1);
    }
  }

  constexpr int training_rounds = 3;
  constexpr int training_samples = 1000;
  std::vector<double> training_elapsed_ns;
  training_elapsed_ns.reserve(training_rounds);
  double training_checksum = 0.0;
  const auto training_probe = feature(3.0, 1.5, 7.0, 5.0, 1.2);
  for (int round = 0; round < training_rounds; ++round) {
    std::unique_ptr<Classifier> training_model(HeatPredictor::make_model());
    const auto start = Clock::now();
    train_model(*training_model, training_samples);
    const auto end = Clock::now();
    training_elapsed_ns.push_back(
        std::chrono::duration<double, std::nano>(end - start).count() /
        training_samples);
    training_checksum += training_model->predict_proba_one(training_probe)[1];
  }
  std::sort(training_elapsed_ns.begin(), training_elapsed_ns.end());

  std::unique_ptr<Classifier> train(HeatPredictor::make_model());
  train_model(*train);
  std::unique_ptr<Classifier> snapshot = train->clone_for_prediction();

  const std::vector<std::vector<double>> probes = {
      feature(0.1, 10.0, 1.0, 0.0, 0.1),
      feature(1.0, 4.0, 3.0, 2.0, 0.4),
      feature(3.0, 1.5, 7.0, 5.0, 1.2),
      feature(5.0, 0.5, 9.0, 8.0, 1.8),
  };

  constexpr int clone_rounds = 7;
  std::vector<double> clone_elapsed_ns;
  clone_elapsed_ns.reserve(clone_rounds);
  double clone_checksum = 0.0;
  for (int round = 0; round < clone_rounds; ++round) {
    const auto start = Clock::now();
    std::unique_ptr<Classifier> clone = train->clone_for_prediction();
    const auto end = Clock::now();
    clone_elapsed_ns.push_back(
        std::chrono::duration<double, std::nano>(end - start).count());
    clone_checksum += clone->predict_proba_one(probes[round % probes.size()])[1];
  }
  std::sort(clone_elapsed_ns.begin(), clone_elapsed_ns.end());

  double warmup_checksum = 0.0;
  for (int i = 0; i < 5000; ++i) {
    warmup_checksum += snapshot->predict_proba_one(
        probes[static_cast<size_t>(i) % probes.size()])[1];
  }

  constexpr int rounds = 5;
  constexpr int predictions_per_round = 50000;
  std::vector<double> elapsed_ns;
  elapsed_ns.reserve(rounds);
  double checksum = 0.0;

  for (int round = 0; round < rounds; ++round) {
    const auto start = Clock::now();
    for (int i = 0; i < predictions_per_round; ++i) {
      checksum += snapshot->predict_proba_one(
          probes[static_cast<size_t>(i) % probes.size()])[1];
    }
    const auto end = Clock::now();
    elapsed_ns.push_back(
        std::chrono::duration<double, std::nano>(end - start).count() /
        predictions_per_round);
  }

  std::sort(elapsed_ns.begin(), elapsed_ns.end());

  constexpr int eq_rounds = 3;
  constexpr int eq_ops_per_round = 50000;
  std::vector<double> eq_elapsed_ns;
  eq_elapsed_ns.reserve(eq_rounds);
  double eq_checksum = 0.0;
  for (int round = 0; round < eq_rounds; ++round) {
    EvaluationQueue eq;
    const auto start = Clock::now();
    for (int i = 0; i < eq_ops_per_round; ++i) {
      const uint64_t index = static_cast<uint64_t>(i + 1);
      const uint64_t now_ns = index * 1000000ULL;
      PredictionSample item = trace_item(index, index % 12000);
      item.predicted_label = static_cast<int>((index / 7) % 2);
      auto evaluated = eq.expire_before_prepare(item, now_ns);
      eq.enqueue(item, now_ns);
      eq_checksum += eq.heat_label_threshold * 0.000001;
      for (const auto& sample : evaluated) {
        eq_checksum += sample.label +
            sample.future_window_access_count * 0.00001;
      }
    }
    const auto end = Clock::now();
    eq_elapsed_ns.push_back(
        std::chrono::duration<double, std::nano>(end - start).count() /
        eq_ops_per_round);
  }
  std::sort(eq_elapsed_ns.begin(), eq_elapsed_ns.end());

  constexpr int concurrent_rounds = 5;
  constexpr int concurrent_threads = 8;
  constexpr int concurrent_ops_per_thread = 10000;
  std::array<double, concurrent_rounds> serialized_ns{};
  std::array<double, concurrent_rounds> two_phase_ns{};
  double concurrent_checksum = 0.0;
  for (int round = 0; round < concurrent_rounds; ++round) {
    ConcurrentEqResult serialized = benchmark_serialized_prediction(
        *snapshot, concurrent_threads, concurrent_ops_per_thread);
    ConcurrentEqResult two_phase = benchmark_two_phase_prediction(
        *snapshot, concurrent_threads, concurrent_ops_per_thread);
    serialized_ns[static_cast<size_t>(round)] = serialized.ns_per_op;
    two_phase_ns[static_cast<size_t>(round)] = two_phase.ns_per_op;
    concurrent_checksum += serialized.checksum + two_phase.checksum;
  }
  std::sort(serialized_ns.begin(), serialized_ns.end());
  std::sort(two_phase_ns.begin(), two_phase_ns.end());

  constexpr int quantile_rounds = 3;
  constexpr int quantile_operations = 450000;
  std::array<double, quantile_rounds> pbds_quantile_ns{};
  std::array<double, quantile_rounds> integer_quantile_ns{};
  double pbds_quantile_checksum = 0.0;
  double integer_quantile_checksum = 0.0;
  for (int round = 0; round < quantile_rounds; ++round) {
    HpQuantileWindow pbds_window(HP_REPORT_SAMPLE_WINDOW_CAPACITY);
    pbds_quantile_ns[static_cast<size_t>(round)] =
        benchmark_quantile_window(
            pbds_window, quantile_operations, &pbds_quantile_checksum);

    HpIntegerQuantileWindow integer_window(
        HP_REPORT_SAMPLE_WINDOW_CAPACITY);
    integer_quantile_ns[static_cast<size_t>(round)] =
        benchmark_quantile_window(
            integer_window, quantile_operations, &integer_quantile_checksum);
  }
  std::sort(pbds_quantile_ns.begin(), pbds_quantile_ns.end());
  std::sort(integer_quantile_ns.begin(), integer_quantile_ns.end());

  std::cout << std::fixed << std::setprecision(6)
            << "adwin_detection_signature=" << adwin_detection_signature << '\n'
            << "adwin_width=" << adwin.width << '\n'
            << "adwin_bucket_count=" << adwin.n_buckets << '\n'
            << "training_checksum=" << training_checksum << '\n'
            << "training_ns_per_op_median="
            << training_elapsed_ns[training_rounds / 2] << '\n'
            << "clone_checksum=" << clone_checksum << '\n'
            << "clone_ns_median=" << clone_elapsed_ns[clone_rounds / 2] << '\n'
            << "warmup_checksum=" << warmup_checksum << '\n'
            << "prediction_checksum=" << checksum << '\n'
            << "predict_ns_per_op_median=" << elapsed_ns[rounds / 2] << '\n'
            << "eq_checksum=" << eq_checksum << '\n'
            << "eq_ns_per_op_median=" << eq_elapsed_ns[eq_rounds / 2] << '\n'
            << "concurrent_checksum=" << concurrent_checksum << '\n'
            << "serialized_predict_ns_per_op_median="
            << serialized_ns[concurrent_rounds / 2] << '\n'
            << "two_phase_predict_ns_per_op_median="
            << two_phase_ns[concurrent_rounds / 2] << '\n'
            << "pbds_quantile_checksum=" << pbds_quantile_checksum << '\n'
            << "integer_quantile_checksum="
            << integer_quantile_checksum << '\n'
            << "pbds_quantile_ns_per_op_median="
            << pbds_quantile_ns[quantile_rounds / 2] << '\n'
            << "integer_quantile_ns_per_op_median="
            << integer_quantile_ns[quantile_rounds / 2] << '\n';
  return 0;
}
