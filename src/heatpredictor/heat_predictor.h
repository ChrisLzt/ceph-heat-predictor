#ifndef CEPH_BLK_HEAT_PREDICTOR_H
#define CEPH_BLK_HEAT_PREDICTOR_H

#include <algorithm>
#include <cmath>
#include <iterator>
#include <list>
#include <optional>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

#include "include/ARFClassifier.h"
#include "include/PipelineClassifier.h"
#include "include/StandardScaler.h"

#include <atomic>
#include "common/debug.h"

#define NUM_FEATURES 5
static constexpr uint64_t HP_BUCKET_SHIFT = 16;
static constexpr double HP_HOT_QUANTILE = 0.80;
static constexpr double HP_HOT_CLASS_WEIGHT = 1.2;
static constexpr size_t HP_EVALUATION_WINDOW = 20000;
static constexpr size_t HP_THRESHOLD_HISTORY_CAPACITY = 400000;
static constexpr double HP_HEAT_INCREMENT = 100.0;
static constexpr double HP_MAX_TRAINING_WEIGHT = 3.0;
static constexpr size_t HP_LRU_CAPACITY = 20000;
static constexpr double HP_HEAT_RETAIN_RATIO = 1.0 / 10.0;

inline double hp_heat_decay_alpha(size_t evaluation_window) {
    ceph_assert(evaluation_window > 0);
    return std::log(HP_HEAT_RETAIN_RATIO) /
        static_cast<double>(evaluation_window);
}

struct TraceItem {
    uint64_t index;
    uint64_t operation;
    uint64_t size;
    uint64_t key;
    int64_t pool;
    uint64_t object_offset;
    uint64_t object_bucket;
    double current_heat;
    double hot_threshold;
    uint64_t access_count;
    int pred;
};

struct HeatState {
    double heat;
    uint64_t last_access;
    uint64_t access_count;
    uint64_t pending_count;
    std::list<uint64_t>::iterator lru_position;
};

struct EvaluatedItem {
    TraceItem item;
    int label;
    double training_weight;
};

class EvaluationQueue {
public:
    std::atomic<double> hot_threshold;
    double alpha;
    double heat_increment;
    size_t evaluation_window;
    size_t lru_capacity;

    std::queue<TraceItem> pending_queue;
    std::unordered_map<uint64_t, HeatState> heat_map;
    std::list<uint64_t> lru_list;
    std::vector<double> exp_table;

    size_t hot_list_cap;
    typedef __gnu_pbds::tree<
        std::pair<double, uint64_t>,
        __gnu_pbds::null_type,
        std::less<std::pair<double, uint64_t>>,
        __gnu_pbds::rb_tree_tag,
        __gnu_pbds::tree_order_statistics_node_update
    > pbds_set;
    pbds_set hot_list;
    std::queue<std::pair<double, uint64_t>> hot_list_order;
    uint64_t pbds_counter = 0;

    EvaluationQueue(
            size_t evaluation_window = HP_EVALUATION_WINDOW,
            size_t lru_capacity = HP_LRU_CAPACITY,
            double hot_threshold = HP_HEAT_INCREMENT,
            double heat_increment = HP_HEAT_INCREMENT) :
            hot_threshold(hot_threshold),
            alpha(hp_heat_decay_alpha(evaluation_window)),
            heat_increment(heat_increment),
            evaluation_window(evaluation_window),
            lru_capacity(lru_capacity),
            hot_list_cap(HP_THRESHOLD_HISTORY_CAPACITY) {
        exp_table.resize(evaluation_window + lru_capacity + 1);
        for (size_t i = 0; i < exp_table.size(); ++i) {
            exp_table[i] = std::exp(static_cast<double>(i) * alpha);
        }
    }

    double decay_heat(
            double last_heat, uint64_t last_ts, uint64_t cur_ts) const {
        if (cur_ts <= last_ts) {
            return last_heat;
        }
        uint64_t delta = cur_ts - last_ts;
        double factor = delta < exp_table.size()
            ? exp_table[delta]
            : std::exp(static_cast<double>(delta) * alpha);
        return factor * last_heat;
    }

    void record_expired_heat(double heat) {
        auto entry = std::make_pair(heat, ++pbds_counter);
        hot_list.insert(entry);
        hot_list_order.push(entry);
        if (hot_list.size() > hot_list_cap) {
            hot_list.erase(hot_list_order.front());
            hot_list_order.pop();
        }

        size_t idx = static_cast<size_t>(HP_HOT_QUANTILE * hot_list.size());
        if (idx >= hot_list.size()) {
            idx = hot_list.size() - 1;
        }
        hot_threshold.store(hot_list.find_by_order(idx)->first);
    }

    void prepare_features(TraceItem& item) {
        auto it = heat_map.find(item.key);
        if (it == heat_map.end()) {
            auto [inserted, ok] = heat_map.emplace(
                item.key,
                HeatState{
                    heat_increment,
                    item.index,
                    1,
                    0,
                    lru_list.end()
                });
            ceph_assert(ok);
            it = inserted;
        } else {
            HeatState& state = it->second;
            if (state.pending_count == 0) {
                ceph_assert(state.lru_position != lru_list.end());
                lru_list.erase(state.lru_position);
                state.lru_position = lru_list.end();
            }
            state.heat =
                decay_heat(state.heat, state.last_access, item.index) +
                heat_increment;
            state.last_access = item.index;
            state.access_count++;
        }

        const HeatState& state = it->second;
        item.current_heat = state.heat;
        item.hot_threshold = hot_threshold.load();
        item.access_count = state.access_count;
    }

    std::optional<EvaluatedItem> enqueue(TraceItem item) {
        auto state_it = heat_map.find(item.key);
        ceph_assert(state_it != heat_map.end());
        state_it->second.pending_count++;
        pending_queue.push(item);

        if (pending_queue.size() <= evaluation_window) {
            return std::nullopt;
        }

        TraceItem expired = pending_queue.front();
        pending_queue.pop();

        auto expired_state_it = heat_map.find(expired.key);
        ceph_assert(expired_state_it != heat_map.end());
        HeatState& expired_state = expired_state_it->second;
        ceph_assert(expired_state.pending_count > 0);

        double expired_heat = decay_heat(
            expired_state.heat, expired_state.last_access, item.index);
        int label = expired_heat > hot_threshold.load() ? 1 : 0;
        uint64_t future_access_count =
            expired_state.access_count - expired.access_count;
        double training_weight = std::min(
            HP_MAX_TRAINING_WEIGHT,
            1.0 + std::log2(1.0 + static_cast<double>(future_access_count)));

        record_expired_heat(expired_heat);

        expired_state.pending_count--;
        if (expired_state.pending_count == 0) {
            lru_list.push_back(expired.key);
            expired_state.lru_position = std::prev(lru_list.end());
        }

        if (lru_list.size() > lru_capacity) {
            uint64_t victim = lru_list.front();
            lru_list.pop_front();
            auto victim_it = heat_map.find(victim);
            ceph_assert(victim_it != heat_map.end());
            ceph_assert(victim_it->second.pending_count == 0);
            heat_map.erase(victim_it);
        }

        return EvaluatedItem{expired, label, training_weight};
    }

    size_t pending_size() const { return pending_queue.size(); }
    size_t heat_state_size() const { return heat_map.size(); }
    size_t lru_size() const { return lru_list.size(); }
};

struct TrainingSample {
    TraceItem item;
    int label;
    double weight;
};

struct HeatPredictorStats {
    uint64_t io_count;
    uint64_t labeled_io_total;
    uint64_t pending_io_count;
    uint64_t heat_state_count;
    uint64_t lru_count;
    uint64_t true_positive;
    uint64_t false_positive;
    uint64_t true_negative;
    uint64_t false_negative;
    uint64_t predicted_hot;
    uint64_t predicted_cold;
    double hot_threshold;
};

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
            uint64_t object_name_hash,
            uint64_t object_bucket) {
        uint64_t key = mix64(static_cast<uint64_t>(pool));
        key ^= mix64(
            mix64(ceph_object_hash) ^ mix64(object_name_hash));
        key ^= mix64(object_bucket);
        return key;
    }

    static Classifier* make_model() {
        return new PipelineClassifier(
            new StandardScaler<NUM_FEATURES>(),
            new ARFClassifier<NUM_FEATURES, 2,
                DetectorFactory<ADWIN<5>, 10>,
                DetectorFactory<ADWIN<5>, 1>>(9, NUM_FEATURES, 591422, 100, 4, 0.001, 0.05, 0.99, 0.01)
        );
    }

    // reset_mutex gates full-state reset against foreground prediction and
    // background training without serializing normal predict/train concurrency.
    mutable std::shared_mutex reset_mutex;

    // ── 双缓冲模型 ──────────────────────────────────────────────
    // active_model : 前台预测模型。predict() 全程持有 swap_mutex 调用
    //                predict_one()，避免 swap 后旧 active 变成 shadow 并被
    //                后台训练线程并发写入。
    // shadow_model : 后台训练模型。每训练 SWAP_INTERVAL 个样本后，在
    //                swap_mutex 下与 active_model 直接 swap。
    // 如果未来要缩短锁范围，需要改成 shadow 连续训练、active 使用只读
    // 快照发布，不能只拷贝 shared_ptr 后无锁读取。
    std::shared_ptr<Classifier> active_model;
    std::shared_ptr<Classifier> shadow_model;
    mutable std::mutex swap_mutex;

    std::unique_ptr<EvaluationQueue> eq;
    std::mutex eq_mutex;
    Accuracy<2> accu;

    static constexpr int SWAP_INTERVAL = 2000;
    int shadow_train_count{0};
    std::atomic<uint64_t> swap_count{0};

    // ── 后台训练线程 ────────────────────────────────────────────
    static constexpr int BATCH_SIZE = 400;
    static constexpr size_t MAX_TRAIN_QUEUE_LENGTH = 200000;
    std::queue<TrainingSample> train_queue;
    std::mutex train_queue_mutex;
    std::condition_variable train_queue_cv;
    std::thread train_thread;
    std::once_flag start_flag;
    std::atomic<bool> train_running{false};
    std::atomic<int> pending_notify{0};
    std::atomic<uint64_t> train_drop_count{0};

    static const std::vector<double>& to_feat(const TraceItem& item) {
        thread_local std::vector<double> feat(NUM_FEATURES);
        double threshold = item.hot_threshold > 1.0 ? item.hot_threshold : 1.0;

        feat[0] = static_cast<double>(item.operation);
        feat[1] = std::log2(static_cast<double>(item.size) + 1.0);
        feat[2] = std::log2(static_cast<double>(item.object_bucket) + 1.0);
        feat[3] = std::log2(static_cast<double>(item.access_count) + 1.0);
        feat[4] = item.access_count > 1 ? item.current_heat / threshold : 0.0;
        return feat;
    }

    void try_swap() {
        if (++shadow_train_count < SWAP_INTERVAL) return;
        shadow_train_count = 0;
        swap_count++;
        {
            std::lock_guard<std::mutex> lock(swap_mutex);
            std::swap(active_model, shadow_model);
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

            std::queue<TrainingSample> batch;
            std::swap(batch, train_queue);
            lock.unlock();

            while (!batch.empty()) {
                TrainingSample sample = batch.front();
                batch.pop();

                if (sample.label == 1) {
                    sample.weight *= HP_HOT_CLASS_WEIGHT;
                }

                shadow_model->learn_one(
                    to_feat(sample.item), sample.label, sample.weight);
                try_swap();
            }
        }
    }

    std::atomic<uint64_t> hp_index{0};
    std::atomic<uint64_t> hot_cnt{0}, cold_cnt{0};
    std::atomic<uint64_t> actual_hot{0}, actual_cold{0};

    HeatPredictor() {
        active_model.reset(make_model());
        shadow_model.reset(make_model());
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
        pending_notify.store(0);

        {
            std::lock_guard<std::mutex> lock(swap_mutex);
            active_model.reset(make_model());
            shadow_model.reset(make_model());
        }

        {
            std::lock_guard<std::mutex> lock(eq_mutex);
            discarded_pending = eq->pending_size();
            eq = std::make_unique<EvaluationQueue>();
        }

        accu.clear();
        shadow_train_count = 0;
        swap_count.store(0);
        hp_index.store(0);
        hot_cnt.store(0);
        cold_cnt.store(0);
        actual_hot.store(0);
        actual_cold.store(0);
        train_drop_count.store(0);
        return discarded_pending;
    }

    int predict(int operation, uint64_t size,
            int64_t pool, uint64_t ceph_object_hash, uint64_t object_name_hash,
            uint64_t object_offset,
            uint64_t *index_out) {
        ensure_started();

        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        uint64_t object_bucket = object_offset >> HP_BUCKET_SHIFT;
        uint64_t key = make_object_key(
            pool, ceph_object_hash, object_name_hash, object_bucket);

        uint64_t index = 0;
        TraceItem item = {
            0,                                // index
            static_cast<uint64_t>(operation), // operation
            size,                             // size
            key,                              // key
            pool,                             // pool
            object_offset,                    // object_offset
            object_bucket,                    // object_bucket
            0.0,                              // current_heat
            0.0,                              // hot_threshold
            0,                                // access_count
            0                                 // pred
        };

        int res;
        std::optional<EvaluatedItem> evaluated;
        {
            std::lock_guard<std::mutex> eq_lock(eq_mutex);
            // fetch_add() returns the old value; +1 makes the first index 1.
            index = hp_index.fetch_add(1) + 1;
            item.index = index;
            if (index_out != nullptr) {
                *index_out = index;
            }

            eq->prepare_features(item);
            {
                std::lock_guard<std::mutex> lock(swap_mutex);
                res = active_model->predict_one(to_feat(item));
            }
            res ? hot_cnt++ : cold_cnt++;
            item.pred = res;

            evaluated = eq->enqueue(item);
            if (evaluated.has_value()) {
                evaluated->label ? actual_hot++ : actual_cold++;
                accu.update(evaluated->label, evaluated->item.pred);
            }
        }

        if (evaluated.has_value()) {
            TrainingSample sample{
                evaluated->item,
                evaluated->label,
                evaluated->training_weight
            };
            {
                std::lock_guard<std::mutex> lock(train_queue_mutex);
                train_queue.push(sample);
                while (train_queue.size() > MAX_TRAIN_QUEUE_LENGTH) {
                    train_queue.pop();
                    train_drop_count++;
                }
            }
            if (++pending_notify >= BATCH_SIZE) {
                pending_notify = 0;
                train_queue_cv.notify_one();
            }
        }
        return res;
    }

    HeatPredictorStats get_evaluation_stats() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> eq_lock(eq_mutex);
        return HeatPredictorStats{
            hp_index.load(),
            accu.get_total_weight(),
            eq->pending_size(),
            eq->heat_state_size(),
            eq->lru_size(),
            accu.true_positives(),
            accu.false_positives(),
            accu.true_negatives(),
            accu.false_negatives(),
            hot_cnt.load(),
            cold_cnt.load(),
            eq->hot_threshold.load()
        };
    }

    uint64_t get_total_weight() { return accu.get_total_weight(); }
    double get_accuracy() { return accu.get_accuracy(); }
    double get_hot_precision() { return accu.get_hot_precision(); }
    double get_hot_recall() { return accu.get_hot_recall(); }
    double get_hot_prediction_percent() { return accu.get_hot_prediction_percent(); }
    uint64_t get_true_positives() { return accu.true_positives(); }
    uint64_t get_false_positives() { return accu.false_positives(); }
    uint64_t get_true_negatives() { return accu.true_negatives(); }
    uint64_t get_false_negatives() { return accu.false_negatives(); }
    double get_hot_threshold() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(eq_mutex);
        return eq->hot_threshold.load();
    }
    size_t get_train_queue_length() {
        std::shared_lock<std::shared_mutex> reset_lock(reset_mutex);
        std::lock_guard<std::mutex> lock(train_queue_mutex);
        return train_queue.size();
    }
    uint64_t get_swap_count() { return swap_count.load(); }
    uint64_t get_actual_hot() { return actual_hot.load(); }
    uint64_t get_actual_cold() { return actual_cold.load(); }
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
    uint64_t get_train_drop_count() { return train_drop_count.load(); }
};

#endif
