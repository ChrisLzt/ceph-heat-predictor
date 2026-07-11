#ifndef CEPH_BLK_HEAT_PREDICTOR_H
#define CEPH_BLK_HEAT_PREDICTOR_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
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
#include "hp_quantile_window.h"
#include "hp_types.h"

class HeatPredictor {
public:
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

    static Classifier* make_model() {
        return new PipelineClassifier(
            new StandardScaler<NUM_FEATURES>(),
            new ARFClassifier<NUM_FEATURES, 2,
                DetectorFactory<ADWIN<5>, 10>,
                DetectorFactory<ADWIN<5>, 1>>(
                    HP_ARF_N_MODELS,
                    HP_ARF_MAX_FEATURES,
                    HP_ARF_SEED,
                    HP_ARF_GRACE_PERIOD,
                    HP_ARF_LAMBDA,
                    HP_ARF_DELTA,
                    HP_ARF_TAU,
                    HP_ARF_MAX_SHARE_TO_SPLIT,
                    HP_ARF_MIN_BRANCH_FRACTION)
        );
    }

    // reset_mutex gates full-state reset against foreground prediction and
    // background training without serializing normal predict/train concurrency.
    mutable std::shared_mutex reset_mutex;

    // ── 在线模型 ────────────────────────────────────────────────
    // train_model 只由后台训练线程更新；prediction_snapshot 是前台预测
    // 使用的只读快照。训练线程定期 clone train_model 并发布新快照，
    // 避免前台预测长期等待 learn_one()。
    std::shared_ptr<Classifier> train_model;
    mutable std::mutex train_model_mutex;
    std::shared_ptr<Classifier> prediction_snapshot;

    std::unique_ptr<EvaluationQueue> eq;
    std::mutex eq_mutex;
    std::condition_variable eq_ready_cv;
    Accuracy<2> accu;

    static constexpr int MODEL_UPDATE_REPORT_INTERVAL = 500;
    int model_update_train_count{0};
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

    static const std::vector<double>& to_feat(const TraceItem& item) {
        return hp_to_features(item);
    }

    static std::shared_ptr<Classifier> to_shared_model(
            std::unique_ptr<Classifier> model) {
        return std::shared_ptr<Classifier>(std::move(model));
    }

    bool record_model_update_batch() {
        if (++model_update_train_count < MODEL_UPDATE_REPORT_INTERVAL) {
            return false;
        }
        model_update_train_count = 0;
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
        bool should_notify = false;
        {
            std::lock_guard<std::mutex> lock(train_queue_mutex);
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

    void train_worker() {
        while (true) {
            std::unique_lock<std::mutex> lock(train_queue_mutex);
            train_queue_cv.wait_for(lock, std::chrono::milliseconds(50), [this] {
                return !train_queue.empty() || !train_running;
            });

            if (!train_running && train_queue.empty()) break;

            lock.unlock();
            std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
            lock.lock();
            if (!train_running && train_queue.empty()) break;
            if (train_queue.empty()) continue;

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
                        to_feat(sample.item), sample.label, sample.weight);
                    if (record_model_update_batch()) {
                        next_snapshot = clone_train_model_for_prediction();
                    }
                }

                if (next_snapshot) {
                    publish_prediction_snapshot(std::move(next_snapshot));
                }
            }
        }
    }

    std::atomic<uint64_t> hp_index{0};
    uint64_t actual_hot_object_access_count_sum{0};
    uint64_t actual_cold_object_access_count_sum{0};
    double actual_hot_object_heat_sum{0};
    double actual_cold_object_heat_sum{0};
    double actual_hot_pred_hot_proba_sum{0};
    double actual_cold_pred_hot_proba_sum{0};
    HpIntegerQuantileWindow actual_hot_future_access_window;
    HpIntegerQuantileWindow actual_cold_future_access_window;
    HpQuantileWindow actual_hot_future_heat_window;
    HpQuantileWindow actual_cold_future_heat_window;

    HeatPredictor() {
        train_model.reset(make_model());
        prediction_snapshot = clone_train_model_for_prediction();
        eq = std::make_unique<EvaluationQueue>();
    }

    ~HeatPredictor() {
        if (train_running.exchange(false)) {
            train_queue_cv.notify_all();
            if (train_thread.joinable()) {
                train_thread.join();
            }
        }
    }

    // 懒启动：首次 predict() 调用时才创建后台训练线程，避免静态初始化阶段 spawn thread
    void ensure_started() {
        std::call_once(start_flag, [this] {
            train_running = true;
            train_thread = std::thread(&HeatPredictor::train_worker, this);
        });
    }

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
            train_model.reset(make_model());
            next_snapshot = clone_train_model_for_prediction();
        }
        publish_prediction_snapshot(std::move(next_snapshot));

        {
            std::lock_guard<std::mutex> lock(eq_mutex);
            discarded_pending = eq->pending_size();
            eq = std::make_unique<EvaluationQueue>();
        }

        accu.clear();
        model_update_train_count = 0;
        snapshot_publish_count.store(0);
        hp_index.store(0);
        actual_hot_object_access_count_sum = 0;
        actual_cold_object_access_count_sum = 0;
        actual_hot_object_heat_sum = 0;
        actual_cold_object_heat_sum = 0;
        actual_hot_pred_hot_proba_sum = 0;
        actual_cold_pred_hot_proba_sum = 0;
        actual_hot_future_access_window.clear();
        actual_cold_future_access_window.clear();
        actual_hot_future_heat_window.clear();
        actual_cold_future_heat_window.clear();
        train_drop_count.store(0);
        return discarded_pending;
    }

    bool is_enabled() const {
        return enabled.load(std::memory_order_acquire);
    }

    uint64_t set_enabled(bool next_enabled) {
        enabled.store(false, std::memory_order_release);
        uint64_t discarded_pending = reset();
        enabled.store(next_enabled, std::memory_order_release);
        return discarded_pending;
    }

    int predict(int64_t pool, uint64_t ceph_object_hash,
            uint64_t object_name_hash, uint64_t *index_out) {
        if (!is_enabled()) {
            if (index_out != nullptr) {
                *index_out = 0;
            }
            return 0;
        }

        ensure_started();

        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        if (!is_enabled()) {
            if (index_out != nullptr) {
                *index_out = 0;
            }
            return 0;
        }
        uint64_t key = make_object_key(
            pool, ceph_object_hash, object_name_hash);

        uint64_t index = 0;
        TraceItem item = {
            0,    // index
            key,  // key
            0.0,  // current_heat
            0.0,  // hot_threshold
            0,    // access_count
            0,    // last_access_distance
            0,    // past_window_access_count
            0,    // recent_window_access_count
            0.0,  // heat_percentile
            0.0,  // pred_hot_proba
            0     // pred
        };

        int res;
        std::optional<EvaluatedItem> evaluated;
        EvaluationQueue::PendingSlot *pending_slot = nullptr;
        std::shared_ptr<Classifier> snapshot;
        double hot_predict_threshold = HP_HOT_PREDICT_THRESHOLD;
        {
            std::unique_lock<std::mutex> eq_lock(eq_mutex);
            eq_ready_cv.wait(eq_lock, [this] {
                return eq->can_reserve_prediction();
            });
            // fetch_add() returns the old value; +1 makes the first index 1.
            index = hp_index.fetch_add(1) + 1;
            item.index = index;
            if (index_out != nullptr) {
                *index_out = index;
            }

            eq->prepare_features(item);
            snapshot = get_prediction_snapshot();
            hot_predict_threshold = eq->hot_predict_threshold();
            auto reservation = eq->reserve_prediction(item);
            pending_slot = reservation.slot;
            evaluated = std::move(reservation.evaluated);
            if (evaluated.has_value()) {
                if (evaluated->label) {
                    actual_hot_object_access_count_sum +=
                        evaluated->future_access_count;
                    actual_hot_object_heat_sum += evaluated->future_heat;
                    actual_hot_pred_hot_proba_sum +=
                        evaluated->item.pred_hot_proba;
                    actual_hot_future_access_window.insert(
                        evaluated->future_access_count);
                    actual_hot_future_heat_window.insert(
                        evaluated->future_heat);
                } else {
                    actual_cold_object_access_count_sum +=
                        evaluated->future_access_count;
                    actual_cold_object_heat_sum += evaluated->future_heat;
                    actual_cold_pred_hot_proba_sum +=
                        evaluated->item.pred_hot_proba;
                    actual_cold_future_access_window.insert(
                        evaluated->future_access_count);
                    actual_cold_future_heat_window.insert(
                        evaluated->future_heat);
                }
                accu.update(evaluated->label, evaluated->item.pred);
            }
        }

        try {
            if (snapshot) {
                thread_local std::vector<double> proba;
                snapshot->predict_proba_one_into(to_feat(item), proba);
                item.pred_hot_proba = proba.size() > 1 ? proba[1] : 0.0;
                res = item.pred_hot_proba >= hot_predict_threshold ? 1 : 0;
            } else {
                item.pred_hot_proba = 0.0;
                res = 0;
            }
        } catch (...) {
            bool should_notify = false;
            {
                std::lock_guard<std::mutex> eq_lock(eq_mutex);
                should_notify =
                    eq->complete_prediction(pending_slot, 0.0, 0);
            }
            if (should_notify) {
                eq_ready_cv.notify_all();
            }
            throw;
        }
        item.pred = res;
        bool should_notify = false;
        {
            std::lock_guard<std::mutex> eq_lock(eq_mutex);
            should_notify = eq->complete_prediction(
                pending_slot, item.pred_hot_proba, item.pred);
        }
        if (should_notify) {
            eq_ready_cv.notify_all();
        }

        if (evaluated.has_value()) {
            TrainingSample sample{
                evaluated->item,
                evaluated->label,
                evaluated->training_weight
            };
            enqueue_training_sample(std::move(sample));
        }
        return res;
    }

    HeatPredictorStats get_evaluation_stats() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> eq_lock(eq_mutex);
        return HeatPredictorStats{
            is_enabled(),
            hp_index.load(),
            accu.get_total_weight(),
            eq->pending_size(),
            eq->heat_state_size(),
            eq->lru_size(),
            eq->otsu_histogram_bin_count(),
            eq->otsu_histogram_object_count(),
            accu.true_positives(),
            accu.false_positives(),
            accu.true_negatives(),
            accu.false_negatives(),
            actual_hot_object_access_count_sum,
            actual_cold_object_access_count_sum,
            actual_hot_object_heat_sum,
            actual_cold_object_heat_sum,
            actual_hot_pred_hot_proba_sum,
            actual_cold_pred_hot_proba_sum,
            actual_hot_future_access_window.summary(),
            actual_cold_future_access_window.summary(),
            actual_hot_future_heat_window.summary(),
            actual_cold_future_heat_window.summary(),
            eq->hot_threshold,
            eq->otsu_candidate_threshold,
            eq->otsu_separation,
            eq->otsu_confidence,
            eq->otsu_sample_confidence,
            eq->otsu_sharpness_confidence,
            eq->hot_threshold_method,
            eq->hot_predict_threshold(),
            eq->hot_predict_threshold_target(),
            eq->prediction_calibration_size(),
            eq->prediction_calibration_current_accuracy(),
            eq->prediction_calibration_target_accuracy()
        };
    }

    uint64_t get_total_weight() { return accu.get_total_weight(); }
    size_t get_train_queue_length() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(train_queue_mutex);
        return train_queue.size();
    }
    uint64_t get_snapshot_publish_count() {
        return snapshot_publish_count.load();
    }
    uint64_t get_pending_io_count() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(eq_mutex);
        return eq->pending_size();
    }
    uint64_t get_heat_state_count() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(eq_mutex);
        return eq->heat_state_size();
    }
    uint64_t get_lru_count() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(eq_mutex);
        return eq->lru_size();
    }
    uint64_t get_otsu_histogram_bin_count() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(eq_mutex);
        return eq->otsu_histogram_bin_count();
    }
    uint64_t get_otsu_histogram_object_count() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(eq_mutex);
        return eq->otsu_histogram_object_count();
    }
    uint64_t get_train_drop_count() { return train_drop_count.load(); }
};

#endif
