#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "heatpredictor/hp_evaluation_queue.h"
#include "heatpredictor/hp_features.h"
#include "heatpredictor/hp_score_otsu_histogram.h"

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

using Clock = std::chrono::steady_clock;

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

template <typename Fn>
double benchmark_ns_per_operation(size_t count, Fn&& operation)
{
  const auto start = Clock::now();
  for (size_t index = 0; index < count; ++index) {
    operation(index);
  }
  const auto stop = Clock::now();
  return std::chrono::duration<double, std::nano>(stop - start).count() /
      static_cast<double>(count);
}

} // namespace

int main()
{
  constexpr size_t feature_operations = 1000000;
  PredictionSample feature_sample = make_sample(1, 1);
  feature_sample.heat_after_current_access = 300.0;
  feature_sample.heat_label_threshold_at_prediction = 100.0;
  feature_sample.tracked_access_count = 2;
  feature_sample.time_since_previous_access_ns = 1000000000ULL;
  volatile double feature_checksum = 0.0;
  const double feature_ns = benchmark_ns_per_operation(
      feature_operations,
      [&](size_t) {
        feature_checksum += hp_to_features(feature_sample)[0];
      });

  constexpr size_t histogram_operations = 200000;
  HpScoreOtsuHistogram histogram;
  const double decay = hp_heat_decay_log_factor_per_ns(
      HP_HEAT_DECAY_HORIZON_NS);
  histogram.advance_lower_bound(
      HpScoreOtsuHistogram::score_for_heat_at(
          HP_OTSU_TOTAL_HEAT_MIN, 0, decay));
  const double histogram_ns = benchmark_ns_per_operation(
      histogram_operations,
      [&](size_t index) {
        const double heat = 20.0 + static_cast<double>(index % 5000);
        const auto bin = histogram.insert(
            HpScoreOtsuHistogram::score_for_heat_at(heat, 0, decay));
        histogram.erase(bin);
      });

  constexpr size_t queue_operations = 100000;
  EvaluationQueue queue(1000000, 10000, 100.0, 100.0, 1000, 10000);
  const double queue_ns = benchmark_ns_per_operation(
      queue_operations,
      [&](size_t index) {
        const uint64_t now = static_cast<uint64_t>(index) * 1001;
        auto result = queue.begin_prediction(
            make_sample(index + 1, index % 4096), now);
        if (result.ticket.has_value()) {
          queue.complete_prediction(
              std::move(*result.ticket), 0.5, index % 2);
        }
        queue.maintain_expiry(now);
      });

  std::cout << std::fixed << std::setprecision(2)
            << "feature_ns_per_op=" << feature_ns << '\n'
            << "histogram_update_ns_per_op=" << histogram_ns << '\n'
            << "evaluation_queue_ns_per_op=" << queue_ns << '\n'
            << "feature_checksum=" << feature_checksum << '\n'
            << "PASS: heat predictor performance probe" << std::endl;
  return 0;
}
