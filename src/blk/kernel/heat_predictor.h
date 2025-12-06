#ifndef CEPH_BLK_HEAT_PREDICTOR_H
#define CEPH_BLK_HEAT_PREDICTOR_H

#include <cmath>
#include <optional>
#include <list>
#include <unordered_map>
#include <set>
#include <mutex>

#include "include/ARFClassifier.h"

#define NUM_FEATURES 4

struct TraceItem {
    uint64_t n_instr;
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
    int max_size;
    double hot_thred;
    bool training;
    double alpha;
    int heating;
    uint64_t waiting;

    uint64_t ts;

    std::list<uint64_t> key_order; 
    std::unordered_map<uint64_t, QueueValue> item_map;

    std::multiset<double> hot_list;
    std::multiset<double> backup_hot_list;
    size_t hot_list_cap;
public:
    EvaluationQueue(int max_size, double hot_thred, bool training=false, 
            double alpha=-0.03, int heating=200, uint64_t waiting=10000) :
            max_size(max_size), hot_thred(hot_thred), training(training), 
            alpha(alpha), heating(heating), waiting(waiting) {
        if (training) { hot_list_cap = 50 * max_size; }
    }

    double heat_calculation(double last_heat, bool access, int64_t last_ts, int64_t cur_ts) {
        return std::exp((cur_ts - last_ts) * alpha) * last_heat + (access ? heating : 0);
    }

    double get_p80hot() {
        if (hot_list.empty()) return hot_thred;

        size_t idx = static_cast<size_t>(0.8 * hot_list.size());
        auto it = hot_list.begin();
        std::advance(it, idx);
        if (it != hot_list.end()) {
            return *it;
        }
        return hot_thred;
    }

    std::optional<std::pair<QueueValue, bool>> dequeue() {
        if (key_order.empty()) return std::nullopt;

        // pop the earliest
        int64_t key = key_order.front();
        key_order.pop_front();

        auto it = item_map.find(key);
        if (it == item_map.end()) return std::nullopt; // Should not happen

        QueueValue val = it->second;
        item_map.erase(it);

        // update heat
        double new_heat = heat_calculation(val.heat, false, val.last_access, this->ts);
        val.heat = new_heat;
        val.last_access = this->ts;
        bool is_hot = val.heat > get_p80hot();

        if (training) {
            hot_list.insert(val.heat);
            if (hot_list.size() > hot_list_cap) {
                backup_hot_list.insert(val.heat);
                if (backup_hot_list.size() > hot_list_cap) {
                    hot_list = backup_hot_list; // copy assignment
                    backup_hot_list.clear();
                }
            }
        }

        return std::make_pair(val, is_hot);
    }

    std::optional<std::pair<QueueValue, bool>> enqueue(TraceItem item) {
        uint64_t item_key = item.address;
        std::optional<std::pair<QueueValue, bool>> return_val = std::nullopt;
        
        this->ts = item.n_instr;

        if (item_map.find(item_key) != item_map.end()) {
            QueueValue& existing_val = item_map[item_key];
            
            double new_heat = heat_calculation(existing_val.heat, true, existing_val.last_access, this->ts);

            existing_val.heat = new_heat;
            existing_val.last_access = this->ts;

            if (!key_order.empty()) {
                int64_t first_key = key_order.front();
                QueueValue& first_val = item_map[first_key];
                
                if (this->ts - first_val.init_access > this->waiting) {
                    return_val = dequeue();
                }
            }
            return return_val;
        } 
        else if (item_map.size() == (size_t)max_size) {
            return_val = dequeue();
        }

        QueueValue new_val = {static_cast<double>(this->heating), this->ts, this->ts, item};

        item_map[item_key] = new_val;
        key_order.push_back(item_key);

        return return_val;
    }
};

class HeatPredictor {
    Classifier* model;
    EvaluationQueue* eq;
    Accuracy<2> accu;
    std::mutex model_mutex;

    static std::vector<double> item_to_vector(const TraceItem& item) {
        return std::vector<double> {static_cast<double>(item.n_instr), 
                                static_cast<double>(item.operation), 
                                static_cast<double>(item.size), 
                                static_cast<double>(item.address) };
    }
public:
    uint64_t n_instr;
    HeatPredictor() {
        model = new ARFClassifier<NUM_FEATURES, 2, DetectorFactory<ADWIN<5>, 10>, DetectorFactory<ADWIN<5>, 1> >
            (10, 5, 42); // (n_models, max_features, seed);
        eq = new EvaluationQueue(30, 0.134, true);
    }
    ~HeatPredictor() {
        delete model;
        delete eq;
    }
    std::pair<int, double> predict(uint64_t n_instr, int operation, uint64_t size, uint64_t address) {
        std::lock_guard<std::mutex> lock(model_mutex);
        TraceItem item = {n_instr, static_cast<uint64_t>(operation), size, address, 0};

        int res = model->predict_one(item_to_vector(item));
        item.pred = res;
        auto r_item_opt = eq->enqueue(item);

        if (r_item_opt.has_value()) {
            TraceItem r_item = r_item_opt->first.item;
            bool label = r_item_opt->second;
            accu.update(label?1:0, r_item.pred);
            model->learn_one(item_to_vector(r_item), label?1:0);
        }

        return std::make_pair(res, accu.get());
    }
};

#endif