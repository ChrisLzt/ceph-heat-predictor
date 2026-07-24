#ifndef CEPH_HEATPREDICTOR_HEAT_PREDICTOR_H
#define CEPH_HEATPREDICTOR_HEAT_PREDICTOR_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "include/ARFClassifier.h"
#include "include/Classifier.h"
#include "include/Metrics.h"
#include "include/PipelineClassifier.h"
#include "include/StandardScaler.h"

#include "hp_config.h"
#include "hp_evaluation_queue.h"
#include "hp_features.h"
#include "hp_integer_quantile_window.h"
#include "hp_trace_record.h"
#include "hp_types.h"

struct HeatPredictorStatus {
    HeatPredictorStats evaluation;
    size_t train_queue_length;
    uint64_t train_drop_count;
    uint64_t snapshot_publish_count;
    ArfAdaptationStats arf_adaptation;
    uint64_t predict_error_count;
    uint64_t background_error_count;
    HpTraceStatus trace;
};

class HeatPredictor {
public:
    using ExpiryProgressCallback = void (*)(uint64_t);
    using BackgroundErrorCallback = void (*)();

private:
    static uint64_t mix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    static uint64_t make_object_key(
            int64_t pool,
            uint64_t ceph_object_hash,
            uint64_t object_name_hash) {
        uint64_t key = mix64(static_cast<uint64_t>(pool));
        key ^= mix64(
            mix64(ceph_object_hash) ^ mix64(object_name_hash));
        return key;
    }

    static Classifier* make_model(
            std::shared_ptr<ArfAdaptationTelemetry> adaptation_telemetry =
                nullptr) {
        auto* classifier = new ARFClassifier<NUM_FEATURES, 2,
                DetectorFactory<ADWIN<5>,
                    HP_ARF_WARNING_DELTA_PERMILLE>,
                DetectorFactory<ADWIN<5>,
                    HP_ARF_DRIFT_DELTA_PERMILLE>>(
                    HP_ARF_N_MODELS,
                    HP_ARF_MAX_FEATURES,
                    HP_ARF_SEED,
                    HP_ARF_GRACE_PERIOD,
                    HP_ARF_LAMBDA,
                    HP_ARF_DELTA,
                    HP_ARF_TAU,
                    HP_ARF_MAX_SHARE_TO_SPLIT,
                    HP_ARF_MIN_BRANCH_FRACTION,
                    std::move(adaptation_telemetry));
        return new PipelineClassifier(
            new StandardScaler<NUM_FEATURES>(), classifier);
    }

    // reset_mutex gates full-state reset against foreground prediction and
    // background training without serializing normal predict/train concurrency.
    mutable std::shared_mutex reset_mutex;

    // ── 在线模型 ────────────────────────────────────────────────
    // train_model 只由后台训练线程更新；prediction_snapshot 是前台预测
    // 使用的只读快照。训练线程定期 clone train_model 并发布新快照，
    // 避免前台预测长期等待 learn_one()。
    std::shared_ptr<ArfAdaptationTelemetry> adaptation_telemetry =
        std::make_shared<ArfAdaptationTelemetry>();
    std::shared_ptr<Classifier> train_model;
    mutable std::mutex train_model_mutex;
    std::shared_ptr<Classifier> prediction_snapshot;

    std::unique_ptr<EvaluationQueue> eq;
    std::mutex eq_mutex;
    Accuracy<2> accu;

    static constexpr int MODEL_UPDATE_REPORT_INTERVAL = 500;
    int model_update_train_count{0};
    uint64_t last_snapshot_publish_time_ns{0};
    std::atomic<uint64_t> snapshot_publish_count{0};

    // ── 后台训练线程 ────────────────────────────────────────────
    static constexpr int BATCH_SIZE = 100;
    static constexpr size_t MAX_TRAIN_QUEUE_LENGTH = 200000;
    std::queue<TrainingSample> train_queue;
    std::mutex train_queue_mutex;
    std::condition_variable train_queue_cv;
    std::thread train_thread;
    std::once_flag start_flag;
    std::atomic<bool> train_running{false};
    std::atomic<bool> enabled{true};
    std::atomic<uint64_t> train_drop_count{0};
    std::atomic<uint64_t> predict_error_count{0};
    std::atomic<uint64_t> background_error_count{0};

    // EQ deadlines have their own scheduler. The wait mutex is intentionally
    // separate from reset_mutex/eq_mutex so reset is never blocked by sleep.
    std::mutex expiry_wait_mutex;
    std::condition_variable expiry_cv;
    std::thread expiry_thread;
    std::atomic<bool> expiry_running{false};
    std::atomic<uint64_t> expiry_wake_sequence{0};
    std::atomic<ExpiryProgressCallback> expiry_progress_callback{nullptr};
    std::atomic<BackgroundErrorCallback> background_error_callback{nullptr};
    HpTraceWriter trace_writer;

    static const std::vector<double>& to_feat(const PredictionSample& item) {
        return hp_to_features(item);
    }

    static uint64_t monotonic_now_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    static std::optional<double> validated_hot_probability(
            const std::vector<double>& probabilities) {
        if (probabilities.size() != 2) {
            return std::nullopt;
        }

        double total = 0.0;
        for (double probability : probabilities) {
            if (!std::isfinite(probability) ||
                probability < 0.0 || probability > 1.0) {
                return std::nullopt;
            }
            total += probability;
        }
        if (!std::isfinite(total) ||
            total <= std::numeric_limits<double>::epsilon()) {
            // An untrained ARF has no class votes yet. Treat that expected
            // cold-start state as a cold prediction so its future label can
            // bootstrap training; malformed non-zero output still fails.
            return total == 0.0 ? std::optional<double>(0.0) : std::nullopt;
        }
        return probabilities[1] / total;
    }

    static std::shared_ptr<Classifier> to_shared_model(
            std::unique_ptr<Classifier> model) {
        return std::shared_ptr<Classifier>(std::move(model));
    }

    bool record_model_update_batch(uint64_t now_ns) {
        ++model_update_train_count;
        const bool sample_count_due =
            model_update_train_count >= MODEL_UPDATE_REPORT_INTERVAL;
        const bool time_due = last_snapshot_publish_time_ns != 0 &&
            now_ns >= last_snapshot_publish_time_ns &&
            now_ns - last_snapshot_publish_time_ns >=
                HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS;
        if (!sample_count_due && !time_due) {
            return false;
        }
        model_update_train_count = 0;
        last_snapshot_publish_time_ns = now_ns;
        snapshot_publish_count++;
        return true;
    }

    std::shared_ptr<Classifier> clone_train_model_for_prediction() const {
        return to_shared_model(train_model->clone_for_prediction());
    }

    void publish_prediction_snapshot(std::shared_ptr<Classifier> snapshot) {
        std::atomic_store_explicit(
            &prediction_snapshot, std::move(snapshot),
            std::memory_order_release);
    }

    std::shared_ptr<Classifier> get_prediction_snapshot() const {
        return std::atomic_load_explicit(
            &prediction_snapshot, std::memory_order_acquire);
    }

    static std::queue<TrainingSample> take_training_batch(
            std::queue<TrainingSample>& source) {
        std::queue<TrainingSample> batch;
        size_t count = std::min(
            source.size(), static_cast<size_t>(BATCH_SIZE));
        while (count-- > 0) {
            batch.push(std::move(source.front()));
            source.pop();
        }
        return batch;
    }

    void enqueue_training_sample(TrainingSample sample) {
        if (!is_enabled()) {
            return;
        }
        bool should_notify = false;
        {
            std::lock_guard<std::mutex> lock(train_queue_mutex);
            if (!is_enabled()) {
                return;
            }
            should_notify = train_queue.empty();
            train_queue.push(std::move(sample));
            if (train_queue.size() > MAX_TRAIN_QUEUE_LENGTH) {
                train_queue.pop();
                train_drop_count++;
            }
        }
        if (should_notify) {
            train_queue_cv.notify_one();
        }
    }

    void record_evaluated_locked(const EvaluatedSample& evaluated) {
        if (evaluated.label) {
            hot_labeled_sample_future_access_count_sum +=
                evaluated.future_window_access_count;
            hot_labeled_sample_predicted_hot_probability_sum +=
                evaluated.item.predicted_hot_probability;
            hot_labeled_sample_future_access_count_window.insert(
                evaluated.future_window_access_count);
        } else {
            cold_labeled_sample_future_access_count_sum +=
                evaluated.future_window_access_count;
            cold_labeled_sample_predicted_hot_probability_sum +=
                evaluated.item.predicted_hot_probability;
            cold_labeled_sample_future_access_count_window.insert(
                evaluated.future_window_access_count);
        }
        accu.update(evaluated.label, evaluated.item.predicted_label);
    }

    std::vector<TrainingSample> training_samples_from_evaluated_locked(
            std::vector<EvaluatedSample> evaluated) {
        std::vector<TrainingSample> samples;
        samples.reserve(evaluated.size());
        for (auto& item : evaluated) {
            record_evaluated_locked(item);
            if (trace_writer.is_enabled()) {
                trace_writer.try_submit(
                    hp_trace_record_for_evaluated(item));
            }
            samples.push_back(TrainingSample{
                std::move(item.item), item.label});
        }
        return samples;
    }

    void enqueue_training_samples(std::vector<TrainingSample> samples) {
        for (auto& sample : samples) {
            enqueue_training_sample(std::move(sample));
        }
    }

    void notify_expiry_worker() {
        expiry_wake_sequence.fetch_add(1, std::memory_order_release);
        expiry_cv.notify_one();
    }

    void handle_background_error() noexcept {
        background_error_count.fetch_add(1, std::memory_order_relaxed);
        enabled.store(false, std::memory_order_release);
        try {
            std::lock_guard<std::mutex> lock(train_queue_mutex);
            std::queue<TrainingSample> empty;
            std::swap(train_queue, empty);
        } catch (...) {
        }
        train_queue_cv.notify_all();
        notify_expiry_worker();
        auto callback = background_error_callback.load(
            std::memory_order_acquire);
        if (callback != nullptr) {
            try {
                callback();
            } catch (...) {
            }
        }
    }

    void expiry_worker() noexcept {
        while (expiry_running.load(std::memory_order_acquire)) {
            try {
                const uint64_t observed_sequence =
                    expiry_wake_sequence.load(std::memory_order_acquire);
                std::optional<uint64_t> deadline_ns;
                bool processed_due_maintenance = false;
                uint64_t expired_evaluation_count = 0;

                {
                    std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
                    if (is_enabled() &&
                        expiry_running.load(std::memory_order_acquire)) {
                        std::vector<TrainingSample> samples;
                        {
                            std::lock_guard<std::mutex> eq_lock(eq_mutex);
                            const uint64_t now_ns = monotonic_now_ns();
                            auto maintenance = eq->maintain_expiry(now_ns);
                            samples = training_samples_from_evaluated_locked(
                                std::move(maintenance.evaluated));
                            expired_evaluation_count =
                                maintenance.expired_evaluation_count;
                            processed_due_maintenance =
                                maintenance.processed;
                            if (!maintenance.processed &&
                                maintenance.next_schedule.state ==
                                       EvaluationQueue::ExpiryScheduleState::
                                           waiting_deadline) {
                                deadline_ns =
                                    maintenance.next_schedule.deadline_ns;
                            }
                        }
                        enqueue_training_samples(std::move(samples));
                    }
                }

                if (expired_evaluation_count > 0) {
                    auto callback = expiry_progress_callback.load(
                        std::memory_order_acquire);
                    if (callback != nullptr) {
                        callback(expired_evaluation_count);
                    }
                }

                if (processed_due_maintenance) {
                    continue;
                }

                std::unique_lock<std::mutex> wait_lock(expiry_wait_mutex);
                const auto scheduling_changed = [this, observed_sequence] {
                    return !expiry_running.load(std::memory_order_acquire) ||
                        expiry_wake_sequence.load(std::memory_order_acquire) !=
                            observed_sequence;
                };
                if (deadline_ns.has_value()) {
                    const auto deadline =
                        std::chrono::steady_clock::time_point(
                            std::chrono::nanoseconds(*deadline_ns));
                    expiry_cv.wait_until(
                        wait_lock, deadline, scheduling_changed);
                } else {
                    expiry_cv.wait(wait_lock, scheduling_changed);
                }
            } catch (...) {
                handle_background_error();
            }
        }
    }

    void train_worker() noexcept {
        while (true) {
            try {
                std::unique_lock<std::mutex> lock(train_queue_mutex);
                train_queue_cv.wait(lock, [this] {
                    return !train_running ||
                        (is_enabled() && !train_queue.empty());
                });

                if (!train_running) break;

                lock.unlock();
                std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
                lock.lock();
                if (!train_running) break;
                if (!is_enabled() || train_queue.empty()) continue;

                std::queue<TrainingSample> batch =
                    take_training_batch(train_queue);
                lock.unlock();

                while (!batch.empty()) {
                    TrainingSample sample = std::move(batch.front());
                    batch.pop();
                    std::shared_ptr<Classifier> next_snapshot;

                    {
                        std::lock_guard<std::mutex> lock(train_model_mutex);
                        train_model->learn_one(
                            to_feat(sample.item), sample.label);
                        if (record_model_update_batch(monotonic_now_ns())) {
                            next_snapshot =
                                clone_train_model_for_prediction();
                        }
                    }

                    if (next_snapshot) {
                        publish_prediction_snapshot(
                            std::move(next_snapshot));
                    }
                }
            } catch (...) {
                handle_background_error();
            }
        }
    }

    std::atomic<uint64_t> processed_io_count{0};
    uint64_t hot_labeled_sample_future_access_count_sum{0};
    uint64_t cold_labeled_sample_future_access_count_sum{0};
    double hot_labeled_sample_predicted_hot_probability_sum{0};
    double cold_labeled_sample_predicted_hot_probability_sum{0};
    HpIntegerQuantileWindow hot_labeled_sample_future_access_count_window;
    HpIntegerQuantileWindow cold_labeled_sample_future_access_count_window;

public:
    HeatPredictor() {
        train_model.reset(make_model(adaptation_telemetry));
        prediction_snapshot = clone_train_model_for_prediction();
        eq = std::make_unique<EvaluationQueue>();
        last_snapshot_publish_time_ns = monotonic_now_ns();
    }

    ~HeatPredictor() {
        if (expiry_running.exchange(false)) {
            notify_expiry_worker();
        }
        if (expiry_thread.joinable()) {
            expiry_thread.join();
        }
        if (train_running.exchange(false)) {
            train_queue_cv.notify_all();
            if (train_thread.joinable()) {
                train_thread.join();
            }
        }
    }

private:
    // 懒启动：首次 predict() 调用时才创建后台训练线程，避免静态初始化阶段 spawn thread
    void ensure_started() {
        std::call_once(start_flag, [this] {
            train_running = true;
            expiry_running = true;
            try {
                train_thread = std::thread(
                    &HeatPredictor::train_worker, this);
                expiry_thread = std::thread(
                    &HeatPredictor::expiry_worker, this);
            } catch (...) {
                expiry_running = false;
                notify_expiry_worker();
                train_running = false;
                train_queue_cv.notify_all();
                if (expiry_thread.joinable()) {
                    expiry_thread.join();
                }
                if (train_thread.joinable()) {
                    train_thread.join();
                }
                throw;
            }
        });
    }

public:
    HeatPredictor(const HeatPredictor&) = delete;
    HeatPredictor& operator=(const HeatPredictor&) = delete;

    uint64_t reset() {
        std::unique_lock<std::shared_mutex> reset_lock(reset_mutex);
        uint64_t discarded_pending = 0;

        {
            std::lock_guard<std::mutex> lock(train_queue_mutex);
            std::queue<TrainingSample> empty;
            std::swap(train_queue, empty);
        }
        std::shared_ptr<Classifier> next_snapshot;
        {
            std::lock_guard<std::mutex> lock(train_model_mutex);
            adaptation_telemetry->reset();
            train_model.reset(make_model(adaptation_telemetry));
            next_snapshot = clone_train_model_for_prediction();
        }
        publish_prediction_snapshot(std::move(next_snapshot));

        {
            std::lock_guard<std::mutex> lock(eq_mutex);
            const auto queue_status = eq->status(monotonic_now_ns());
            discarded_pending =
                queue_status.pending_io_count +
                queue_status.awaiting_prediction_count;
            eq = std::make_unique<EvaluationQueue>();
        }

        accu.clear();
        model_update_train_count = 0;
        last_snapshot_publish_time_ns = monotonic_now_ns();
        snapshot_publish_count.store(0);
        processed_io_count.store(0);
        hot_labeled_sample_future_access_count_sum = 0;
        cold_labeled_sample_future_access_count_sum = 0;
        hot_labeled_sample_predicted_hot_probability_sum = 0;
        cold_labeled_sample_predicted_hot_probability_sum = 0;
        hot_labeled_sample_future_access_count_window.clear();
        cold_labeled_sample_future_access_count_window.clear();
        train_drop_count.store(0);
        predict_error_count.store(0);
        background_error_count.store(0);
        notify_expiry_worker();
        return discarded_pending;
    }

    bool is_enabled() const {
        return enabled.load(std::memory_order_acquire);
    }

    void set_expiry_progress_callback(ExpiryProgressCallback callback) {
        expiry_progress_callback.store(callback, std::memory_order_release);
    }

    void set_background_error_callback(BackgroundErrorCallback callback) {
        background_error_callback.store(callback, std::memory_order_release);
    }

    uint64_t set_enabled(bool next_enabled) {
        enabled.store(false, std::memory_order_release);
        uint64_t discarded_pending = reset();
        enabled.store(next_enabled, std::memory_order_release);
        train_queue_cv.notify_all();
        notify_expiry_worker();
        return discarded_pending;
    }

    int predict(int64_t pool, uint64_t ceph_object_hash,
            uint64_t object_name_hash, uint64_t *io_sequence_out) {
        if (!is_enabled()) {
            if (io_sequence_out != nullptr) {
                *io_sequence_out = 0;
            }
            return 0;
        }

        try {
            ensure_started();
        } catch (...) {
            predict_error_count.fetch_add(1, std::memory_order_relaxed);
            if (io_sequence_out != nullptr) {
                *io_sequence_out = 0;
            }
            return 0;
        }

        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        if (!is_enabled()) {
            if (io_sequence_out != nullptr) {
                *io_sequence_out = 0;
            }
            return 0;
        }
        uint64_t object_key_hash = make_object_key(
            pool, ceph_object_hash, object_name_hash);

        std::vector<TrainingSample> expired_samples;
        uint64_t io_sequence = 0;
        PredictionSample item = {
            0,                // io_sequence
            object_key_hash,  // object_key_hash
            0.0,              // heat_after_current_access
            0.0,              // heat_label_threshold_at_prediction
            0,    // tracked_access_count
            0,    // time_since_previous_access_ns
            0.0,  // predicted_hot_probability
            0     // predicted_label
        };

        int res;
        std::optional<EvaluationQueue::PredictionTicket> pending_evaluation;
        std::shared_ptr<Classifier> snapshot;
        bool maintenance_schedule_changed = false;
        uint64_t prediction_time_ns = 0;
        {
            std::lock_guard<std::mutex> eq_lock(eq_mutex);
            const uint64_t now_ns = monotonic_now_ns();
            prediction_time_ns = now_ns;
            // fetch_add() returns the previous sequence; the current I/O is +1.
            const uint64_t previous_io_sequence =
                processed_io_count.fetch_add(1);
            io_sequence = previous_io_sequence + 1;
            item.io_sequence = io_sequence;
            if (io_sequence_out != nullptr) {
                *io_sequence_out = io_sequence;
            }

            auto begin = eq->begin_prediction(std::move(item), now_ns);
            item = begin.sample;
            expired_samples = training_samples_from_evaluated_locked(
                std::move(begin.evaluated));
            snapshot = get_prediction_snapshot();
            if (begin.ticket.has_value()) {
                pending_evaluation.emplace(std::move(*begin.ticket));
            }
            maintenance_schedule_changed =
                begin.expiry_schedule_changed;
        }
        if (maintenance_schedule_changed) {
            notify_expiry_worker();
        }
        enqueue_training_samples(std::move(expired_samples));

        bool prediction_failed = false;
        bool cold_start_fallback = false;
        if (snapshot) {
            try {
                thread_local std::vector<double> proba;
                snapshot->predict_proba_one_into(to_feat(item), proba);
                cold_start_fallback = proba.size() == 2 &&
                    proba[0] == 0.0 && proba[1] == 0.0;
                auto hot_probability = validated_hot_probability(proba);
                if (hot_probability.has_value()) {
                    item.predicted_hot_probability = *hot_probability;
                    res = item.predicted_hot_probability >=
                        HP_HOT_PREDICT_THRESHOLD ? 1 : 0;
                } else {
                    prediction_failed = true;
                }
            } catch (...) {
                prediction_failed = true;
            }
        } else {
            prediction_failed = true;
        }
        const bool evaluation_reserved = pending_evaluation.has_value();
        if (prediction_failed) {
            predict_error_count.fetch_add(1, std::memory_order_relaxed);
            res = 0;
            if (pending_evaluation.has_value()) {
                std::lock_guard<std::mutex> eq_lock(eq_mutex);
                eq->cancel_prediction(std::move(*pending_evaluation));
            }
            if (trace_writer.is_enabled()) {
                const uint8_t flags = evaluation_reserved
                    ? HP_TRACE_FLAG_NONE
                    : HP_TRACE_FLAG_EVALUATION_CAPACITY_DROP;
                trace_writer.try_submit(hp_trace_record_for_incomplete(
                    item,
                    prediction_time_ns,
                    HpTraceOutcome::prediction_error,
                    flags));
            }
        } else {
            item.predicted_label = res;
        }
        if (!prediction_failed && pending_evaluation.has_value()) {
            std::vector<TrainingSample> completed_samples;
            {
                std::lock_guard<std::mutex> eq_lock(eq_mutex);
                completed_samples = training_samples_from_evaluated_locked(
                    eq->complete_prediction(
                        std::move(*pending_evaluation),
                        item.predicted_hot_probability,
                        item.predicted_label,
                        cold_start_fallback));
            }
            enqueue_training_samples(std::move(completed_samples));
        } else if (!prediction_failed && trace_writer.is_enabled()) {
            uint8_t flags = HP_TRACE_FLAG_EVALUATION_CAPACITY_DROP;
            if (cold_start_fallback) {
                flags |= HP_TRACE_FLAG_COLD_START_FALLBACK;
            }
            trace_writer.try_submit(hp_trace_record_for_incomplete(
                item,
                prediction_time_ns,
                HpTraceOutcome::evaluation_capacity_drop,
                flags));
        }

        return res;
    }

    HeatPredictorStatus status() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        HeatPredictorStats evaluation = [&] {
            std::lock_guard<std::mutex> eq_lock(eq_mutex);
            const auto queue_status = eq->status(monotonic_now_ns());
            return HeatPredictorStats{
                is_enabled(),
                processed_io_count.load(),
                accu.get_total_weight(),
                queue_status.pending_io_count,
                queue_status.awaiting_prediction_count,
                queue_status.evaluation_drop_count,
                queue_status.heat_state_count,
                queue_status.lru_count,
                queue_status.protected_heat_state_count,
                queue_status.heat_state_peak_count,
                queue_status.lru_eviction_count,
                queue_status.otsu_histogram_bin_count,
                queue_status.otsu_histogram_vote_count,
                accu.true_positives(),
                accu.false_positives(),
                accu.true_negatives(),
                accu.false_negatives(),
                hot_labeled_sample_future_access_count_sum,
                cold_labeled_sample_future_access_count_sum,
                hot_labeled_sample_predicted_hot_probability_sum,
                cold_labeled_sample_predicted_hot_probability_sum,
                hot_labeled_sample_future_access_count_window.summary(),
                cold_labeled_sample_future_access_count_window.summary(),
                queue_status.heat_label_threshold,
                queue_status.otsu_candidate_threshold,
                queue_status.hot_threshold_method
            };
        }();
        size_t train_queue_length = 0;
        {
            std::lock_guard<std::mutex> lock(train_queue_mutex);
            train_queue_length = train_queue.size();
        }
        return HeatPredictorStatus{
            std::move(evaluation),
            train_queue_length,
            train_drop_count.load(),
            snapshot_publish_count.load(),
            adaptation_telemetry->snapshot(),
            predict_error_count.load(std::memory_order_relaxed),
            background_error_count.load(std::memory_order_relaxed),
            trace_writer.status()};
    }

    void record_predict_error() {
        predict_error_count.fetch_add(1, std::memory_order_relaxed);
    }

    bool start_trace(
            const std::string& directory,
            int32_t osd_id,
            const std::string& phase,
            const std::string& git_commit) {
        return trace_writer.start(
            directory,
            osd_id,
            phase,
            git_commit,
            hp_trace_config_hash(),
            NUM_FEATURES);
    }

    void stop_trace() {
        trace_writer.stop();
    }

    HpTraceStatus get_trace_status() const {
        return trace_writer.status();
    }

};

#endif
