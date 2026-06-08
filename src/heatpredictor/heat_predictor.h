#ifndef CEPH_BLK_HEAT_PREDICTOR_H
#define CEPH_BLK_HEAT_PREDICTOR_H

#include <cmath>
#include <optional>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
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
static constexpr uint64_t HP_BUCKET_SHIFT = 20;
static constexpr double HP_HOT_QUANTILE = 0.85;

struct TraceItem {
    uint64_t index;
    uint64_t operation;
    uint64_t size;
    uint64_t key;
    int64_t pool;
    uint64_t object_hash;
    uint64_t object_offset;
    uint64_t object_bucket;
    double current_heat;
    double hot_threshold;
    uint64_t access_count;
    int pred;
};

struct QueueValue {
    double heat;
    uint64_t init_access;
    uint64_t last_access;
    uint64_t access_count;
    TraceItem item;
};

class EvaluationQueue {
public:
    int max_size;
    std::atomic<double> hot_threshold;
    bool training;
    double alpha;
    int heating;
    uint64_t waiting;

    uint64_t ts;

    std::queue<uint64_t> key_order;
    std::unordered_map<uint64_t, QueueValue> item_map;

    size_t hot_list_cap;
    typedef __gnu_pbds::tree<
        std::pair<double, int>,
        __gnu_pbds::null_type,
        std::less<std::pair<double, int>>,
        __gnu_pbds::rb_tree_tag,
        __gnu_pbds::tree_order_statistics_node_update
    > pbds_set;
    pbds_set hot_list;
    pbds_set hot_list_backup;
    int pbds_counter = 0;

    std::vector<double> exp_table;

    EvaluationQueue(int max_size = 30000, double hot_threshold = 200, bool training = true,
            double alpha = -0.00008, int heating = 200, uint64_t waiting = 20000) :
            max_size(max_size), hot_threshold(hot_threshold),
            training(training), alpha(alpha), heating(heating),
            waiting(waiting) {
        if (training) { hot_list_cap = 20 * max_size; }
        exp_table.resize(waiting * 2);
        for (size_t i = 0; i < exp_table.size(); ++i) {
            exp_table[i] = std::exp(i * alpha);
        }
    }

    double heat_calculation(double last_heat, bool access, int64_t last_ts, int64_t cur_ts) {
        return exp_table[std::min(static_cast<size_t>(cur_ts - last_ts), exp_table.size() - 1)] * last_heat + (access ? heating : 0);
    }

    double get_hot_threshold() {
        if (hot_list.empty()) return hot_threshold.load();

        size_t idx = static_cast<size_t>(HP_HOT_QUANTILE * hot_list.size());
        double threshold = hot_list.find_by_order(idx)->first;
        hot_threshold.store(threshold);

        return threshold;
    }

    std::optional<std::pair<QueueValue, bool>> dequeue() {
        if (key_order.empty()) return std::nullopt;

        uint64_t key = key_order.front();
        key_order.pop();

        auto it = item_map.find(key);
        if (it == item_map.end()) return std::nullopt;

        QueueValue val = it->second;
        item_map.erase(it);

        double new_heat = heat_calculation(val.heat, false, val.last_access, this->ts);
        val.heat = new_heat;
        val.last_access = this->ts;
        bool is_hot = val.heat > get_hot_threshold();

        if (training) {
            hot_list.insert({val.heat, ++pbds_counter});
            if (hot_list.size() > hot_list_cap) {
                hot_list_backup.insert({val.heat, pbds_counter});
                if (hot_list_backup.size() > hot_list_cap) {
                    hot_list.swap(hot_list_backup);
                    hot_list_backup.clear();
                }
            }
        }

        return std::make_pair(val, is_hot);
    }

    void fill_object_features(TraceItem& item) {
        this->ts = item.index;
        item.hot_threshold = hot_threshold.load();

        auto existing_it = item_map.find(item.key);
        if (existing_it != item_map.end()) {
            QueueValue& existing_val = existing_it->second;
            item.current_heat = heat_calculation(
                existing_val.heat, true, existing_val.last_access, this->ts);
            item.access_count = existing_val.access_count + 1;
        } else {
            item.current_heat = static_cast<double>(this->heating);
            item.access_count = 1;
        }
    }

    std::optional<std::pair<QueueValue, bool>> enqueue(TraceItem item) {
        this->ts = item.index;
        std::optional<std::pair<QueueValue, bool>> return_val = std::nullopt;

        auto existing_it = item_map.find(item.key);
        if (existing_it != item_map.end()) {
            QueueValue& existing_val = existing_it->second;
            double new_heat = heat_calculation(existing_val.heat, true, existing_val.last_access, this->ts);
            existing_val.heat = new_heat;
            existing_val.last_access = this->ts;
            existing_val.access_count++;
        } else {
            QueueValue new_val = {static_cast<double>(this->heating), this->ts, this->ts, 1, item};
            item_map[item.key] = new_val;
            key_order.push(item.key);
        }

        while (!key_order.empty()) {
            uint64_t first_key = key_order.front();
            auto first_it = item_map.find(first_key);
            if (first_it == item_map.end()) {
                key_order.pop();
                continue;
            }

            if (this->ts - first_it->second.init_access >= this->waiting) {
                return_val = dequeue();
            }
            break;
        }

        if (!return_val.has_value() && item_map.size() > static_cast<size_t>(max_size)) {
            return_val = dequeue();
        }

        return return_val;
    }
};

struct TrainingSample {
    TraceItem item;
    int label;
    int update_accu;
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
            int64_t pool, uint64_t object_hash, uint64_t object_bucket) {
        uint64_t key = mix64(static_cast<uint64_t>(pool));
        key ^= mix64(object_hash);
        key ^= mix64(object_bucket);
        return key;
    }

    static uint64_t make_object_hash(
            uint64_t ceph_object_hash, uint64_t object_name_hash) {
        return mix64(ceph_object_hash) ^ mix64(object_name_hash);
    }

    static Classifier* make_model() {
        return new PipelineClassifier(
            new StandardScaler<NUM_FEATURES>(),
            new ARFClassifier<NUM_FEATURES, 2,
                DetectorFactory<ADWIN<5>, 10>,
                DetectorFactory<ADWIN<5>, 1>>(5, 5, 591422, 100, 4, 0.001, 0.05, 0.99, 0.01)
        );
    }

    // ── 双缓冲模型 ──────────────────────────────────────────────
    // active_model : 前台预测，predict() 在 swap_mutex 下拷贝 shared_ptr 后无锁读取
    // shadow_model : 后台训练线程独占，每 SWAP_INTERVAL 次训练后与 active_model 直接 swap
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
    static constexpr int BATCH_SIZE = 500;
    static constexpr size_t MAX_TRAIN_QUEUE_LENGTH = 200000;
    std::queue<TrainingSample> train_queue;
    std::mutex train_queue_mutex;
    std::condition_variable train_queue_cv;
    std::thread train_thread;
    std::once_flag start_flag;
    std::atomic<bool> train_running{false};
    std::atomic<int> pending_notify{0};

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

            std::queue<TrainingSample> batch;
            std::swap(batch, train_queue);
            lock.unlock();

            while (!batch.empty()) {
                TrainingSample sample = batch.front();
                batch.pop();

                sample.label ? actual_hot++ : actual_cold++;
                double weight = sample.label ? 3.0 : 1.0;
                shadow_model->learn_one(to_feat(sample.item), sample.label, weight);
                if (sample.update_accu)
                    accu.update(sample.label, sample.item.pred);
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

    int predict(uint64_t index, int operation, uint64_t size,
            int64_t pool, uint64_t ceph_object_hash, uint64_t object_name_hash,
            uint64_t object_offset,
            int update_accu) {
        ensure_started();

        uint64_t object_hash = make_object_hash(ceph_object_hash, object_name_hash);
        uint64_t object_bucket = object_offset >> HP_BUCKET_SHIFT;
        uint64_t key = make_object_key(pool, object_hash, object_bucket);

        TraceItem item = {
            index,
            static_cast<uint64_t>(operation),
            size,
            key,
            pool,
            object_hash,
            object_offset,
            object_bucket,
            0.0,
            0.0,
            0,
            0
        };

        int res;
        {
            std::shared_ptr<Classifier> m;
            {
                {
                    std::lock_guard<std::mutex> eq_lock(eq_mutex);
                    eq->fill_object_features(item);
                }
                std::lock_guard<std::mutex> lock(swap_mutex);
                m = active_model;
                res = m->predict_one(to_feat(item));
            }
        }
        res ? hot_cnt++ : cold_cnt++;
        item.pred = res;

        std::optional<std::pair<QueueValue, bool>> r_item_opt;
        {
            std::lock_guard<std::mutex> lock(eq_mutex);
            r_item_opt = eq->enqueue(item);
        }

        if (r_item_opt.has_value()) {
            TrainingSample sample;
            sample.item = r_item_opt->first.item;
            sample.label = r_item_opt->second ? 1 : 0;
            sample.update_accu = update_accu;
            {
                std::lock_guard<std::mutex> lock(train_queue_mutex);
                train_queue.push(sample);
                while (train_queue.size() > MAX_TRAIN_QUEUE_LENGTH) {
                    train_queue.pop();
                }
            }
            if (++pending_notify >= BATCH_SIZE) {
                pending_notify = 0;
                train_queue_cv.notify_one();
            }
        }
        return res;
    }

    uint64_t get_total_weight() { return accu.get_total_weight(); }
    double get_accuracy() { return accu.get_accuracy(); }
    double get_hot_precision() { return accu.get_hot_precision(); }
    double get_hot_recall() { return accu.get_hot_recall(); }
    double get_hot_threshold() { return eq->hot_threshold.load(); }
    size_t get_train_queue_length() {
        std::lock_guard<std::mutex> lock(train_queue_mutex);
        return train_queue.size();
    }
    uint64_t get_swap_count() { return swap_count.load(); }
    uint64_t get_actual_hot() { return actual_hot.load(); }
    uint64_t get_actual_cold() { return actual_cold.load(); }
};

#endif
