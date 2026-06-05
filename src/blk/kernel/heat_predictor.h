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

#define NUM_FEATURES 4
static constexpr uint64_t HP_BUCKET_SHIFT = 20;
static constexpr uint64_t HP_PAGE_SHIFT = 12;
static constexpr uint64_t HP_PAGE_OFFSET_MASK =
    (1ULL << (HP_BUCKET_SHIFT - HP_PAGE_SHIFT)) - 1;
static constexpr double HP_HOT_QUANTILE = 0.85;

struct TraceItem {
    uint64_t index;
    uint64_t operation;
    uint64_t size;
    uint64_t address;
    int pred;
};

struct QueueValue {
    double heat;
    uint64_t init_access;
    uint64_t last_access;
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
            double alpha = -0.00008, int heating = 200, uint64_t waiting = 80000) :
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

    std::optional<std::pair<QueueValue, bool>> enqueue(TraceItem item) {
        uint64_t item_key = item.address >> HP_BUCKET_SHIFT;
        this->ts = item.index;
        std::optional<std::pair<QueueValue, bool>> return_val = std::nullopt;

        auto existing_it = item_map.find(item_key);
        if (existing_it != item_map.end()) {
            QueueValue& existing_val = existing_it->second;
            double new_heat = heat_calculation(existing_val.heat, true, existing_val.last_access, this->ts);
            existing_val.heat = new_heat;
            existing_val.last_access = this->ts;
        } else {
            QueueValue new_val = {static_cast<double>(this->heating), this->ts, this->ts, item};
            item_map[item_key] = new_val;
            key_order.push(item_key);
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
    static Classifier* make_model() {
        return new PipelineClassifier(
            new StandardScaler<NUM_FEATURES>(),
            new ARFClassifier<NUM_FEATURES, 2,
                DetectorFactory<ADWIN<5>, 10>,
                DetectorFactory<ADWIN<5>, 1>>(6, 4, 591422, 100, 4, 0.001, 0.05, 0.99, 0.01)
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
    std::queue<TrainingSample> train_queue;
    std::mutex train_queue_mutex;
    std::condition_variable train_queue_cv;
    std::thread train_thread;
    std::once_flag start_flag;
    std::atomic<bool> train_running{false};
    std::atomic<int> pending_notify{0};

    static const std::vector<double>& to_feat(const TraceItem& item) {
        thread_local std::vector<double> feat(NUM_FEATURES);
        uint64_t bucket = item.address >> HP_BUCKET_SHIFT;
        uint64_t page_offset = (item.address >> HP_PAGE_SHIFT) & HP_PAGE_OFFSET_MASK;

        feat[0] = static_cast<double>(bucket);
        feat[1] = static_cast<double>(item.operation);
        feat[2] = std::log2(static_cast<double>(item.size) + 1.0);
        feat[3] = static_cast<double>(page_offset);
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
                double weight = sample.label ? 1.5 : 1.0;
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

    int predict(uint64_t index, int operation, uint64_t size, uint64_t address, int update_accu) {
        ensure_started();

        TraceItem item = {index, static_cast<uint64_t>(operation), size, address, 0};

        int res;
        {
            std::shared_ptr<Classifier> m;
            {
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
