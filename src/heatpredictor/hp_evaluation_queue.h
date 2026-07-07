#ifndef CEPH_HEATPREDICTOR_HP_EVALUATION_QUEUE_H
#define CEPH_HEATPREDICTOR_HP_EVALUATION_QUEUE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <limits>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

#include "common/debug.h"
#include "hp_config.h"
#include "hp_types.h"

class EvaluationQueue {
public:
    double hot_threshold;
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
    struct HotListEntry {
        std::pair<double, uint64_t> score;
        std::list<uint64_t>::iterator order_position;
    };
    std::list<uint64_t> hot_list_order;
    std::unordered_map<uint64_t, HotListEntry> hot_list_by_key;
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
            hot_list_cap(HP_LABEL_THRESHOLD_WINDOW_CAPACITY) {
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

    double to_heat_score(double heat, uint64_t timestamp) const {
        double positive_heat = std::max(
            heat, std::numeric_limits<double>::min());
        return std::log(positive_heat) -
            alpha * static_cast<double>(timestamp);
    }

    double from_heat_score(double score, uint64_t timestamp) const {
        double log_heat = score + alpha * static_cast<double>(timestamp);
        double min_log = std::log(std::numeric_limits<double>::min());
        double max_log = std::log(std::numeric_limits<double>::max());
        if (log_heat <= min_log) {
            return 0.0;
        }
        if (log_heat >= max_log) {
            return std::numeric_limits<double>::max();
        }
        return std::exp(log_heat);
    }

    void update_hot_threshold(uint64_t timestamp) {
        if (hot_list.empty()) {
            return;
        }
        size_t idx = static_cast<size_t>(HP_HOT_QUANTILE * hot_list.size());
        if (idx >= hot_list.size()) {
            idx = hot_list.size() - 1;
        }
        hot_threshold =
            from_heat_score(hot_list.find_by_order(idx)->first, timestamp);
    }

    void record_object_heat(uint64_t key, double heat, uint64_t timestamp) {
        if (hot_list_cap == 0) {
            return;
        }

        auto old = hot_list_by_key.find(key);
        if (old != hot_list_by_key.end()) {
            hot_list.erase(old->second.score);
            hot_list_order.erase(old->second.order_position);
        }

        auto entry = std::make_pair(
            to_heat_score(heat, timestamp), ++pbds_counter);
        hot_list.insert(entry);
        hot_list_order.push_back(key);
        hot_list_by_key[key] = HotListEntry{
            entry,
            std::prev(hot_list_order.end())
        };

        while (hot_list_by_key.size() > hot_list_cap) {
            ceph_assert(!hot_list_order.empty());
            uint64_t victim = hot_list_order.front();
            hot_list_order.pop_front();

            auto victim_it = hot_list_by_key.find(victim);
            ceph_assert(victim_it != hot_list_by_key.end());
            hot_list.erase(victim_it->second.score);
            hot_list_by_key.erase(victim_it);
        }

        update_hot_threshold(timestamp);
    }

    void prepare_features(TraceItem& item) {
        auto it = heat_map.find(item.key);
        if (it == heat_map.end()) {
            auto [inserted, ok] = heat_map.emplace(
                item.key,
                HeatState{
                    heat_increment,
                    item.index,
                    item.index,
                    1,
                    0,
                    lru_list.end()
                });
            ceph_assert(ok);
            it = inserted;
            item.last_access_distance = 0;
            item.object_age = 0;
        } else {
            HeatState& state = it->second;
            item.last_access_distance = item.index - state.last_access;
            item.object_age = item.index - state.first_access;
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
        item.access_count = state.access_count;
        record_object_heat(item.key, state.heat, item.index);
        item.hot_threshold = hot_threshold;
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

        double expired_total_heat = decay_heat(
            expired_state.heat, expired_state.last_access, item.index);
        double decayed_entry_heat =
            decay_heat(expired.current_heat, expired.index, item.index);
        double future_heat =
            std::max(0.0, expired_total_heat - decayed_entry_heat);
        int label = future_heat > hot_threshold ? 1 : 0;
        uint64_t future_access_count =
            expired_state.access_count - expired.access_count;
        double training_weight = label == 1 ? HP_HOT_CLASS_WEIGHT : 1.0;

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

        return EvaluatedItem{
            expired,
            label,
            training_weight,
            future_access_count,
            future_heat
        };
    }

    size_t pending_size() const { return pending_queue.size(); }
    size_t heat_state_size() const { return heat_map.size(); }
    size_t lru_size() const { return lru_list.size(); }
};

#endif
