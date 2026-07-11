#ifndef CEPH_HEATPREDICTOR_HP_EVALUATION_QUEUE_H
#define CEPH_HEATPREDICTOR_HP_EVALUATION_QUEUE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <list>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

#include "common/debug.h"
#include "hp_config.h"
#include "hp_otsu_histogram.h"
#include "hp_prediction_threshold.h"
#include "hp_types.h"

class EvaluationQueue {
public:
    struct PendingSlot {
        TraceItem item;
        bool ready;

        PendingSlot(TraceItem item, bool ready) :
                item(std::move(item)), ready(ready) {}
        PendingSlot(const PendingSlot&) = delete;
        PendingSlot& operator=(const PendingSlot&) = delete;
    };

    struct PredictionReservation {
        PendingSlot *slot;
        std::optional<EvaluatedItem> evaluated;
    };

    double hot_threshold;
    double otsu_candidate_threshold;
    uint64_t hot_threshold_method;
    double otsu_separation;
    double otsu_confidence;
    double otsu_sample_confidence;
    double otsu_sharpness_confidence;
    double alpha;
    double heat_increment;
    size_t evaluation_window;
    size_t short_window_capacity;
    size_t lru_capacity;
    HpPredictionThresholdCalibrator prediction_calibrator;

    std::deque<PendingSlot> pending_queue;
    std::deque<uint64_t> short_window_keys;
    std::unordered_map<uint64_t, HeatState> heat_map;
    std::list<uint64_t> lru_list;
    std::vector<double> exp_table;

    size_t threshold_window_capacity;
    typedef __gnu_pbds::tree<
        std::pair<double, uint64_t>,
        __gnu_pbds::null_type,
        std::less<std::pair<double, uint64_t>>,
        __gnu_pbds::rb_tree_tag,
        __gnu_pbds::tree_order_statistics_node_update
    > pbds_set;
    pbds_set threshold_order_stats;
    struct ThresholdWindowEntry {
        std::pair<double, uint64_t> score;
        std::list<uint64_t>::iterator order_position;
    };
    std::list<uint64_t> threshold_order;
    std::unordered_map<uint64_t, ThresholdWindowEntry> threshold_entries_by_key;
    HpOtsuHistogram otsu_histogram;
    uint64_t pbds_counter = 0;
    uint64_t threshold_observation_count = 0;

    EvaluationQueue(
            size_t evaluation_window = HP_EVALUATION_WINDOW,
            size_t lru_capacity = HP_LRU_CAPACITY,
            double hot_threshold = HP_HEAT_INCREMENT,
            double heat_increment = HP_HEAT_INCREMENT,
            size_t short_window_capacity =
                HP_ACCESS_ACCELERATION_WINDOW) :
            hot_threshold(hot_threshold),
            otsu_candidate_threshold(0.0),
            hot_threshold_method(HP_THRESHOLD_METHOD_INITIALIZING),
            otsu_separation(0.0),
            otsu_confidence(0.0),
            otsu_sample_confidence(0.0),
            otsu_sharpness_confidence(0.0),
            alpha(hp_heat_decay_alpha(evaluation_window)),
            heat_increment(heat_increment),
            evaluation_window(evaluation_window),
            short_window_capacity(std::min(
                short_window_capacity, evaluation_window)),
            lru_capacity(lru_capacity),
            threshold_window_capacity(HP_LABEL_THRESHOLD_WINDOW_CAPACITY) {
#if HP_ENABLE_ACCESS_ACCELERATION
        ceph_assert(short_window_capacity > 0);
#endif
        exp_table.resize(evaluation_window + lru_capacity + 1);
        for (size_t i = 0; i < exp_table.size(); ++i) {
            exp_table[i] = std::exp(static_cast<double>(i) * alpha);
        }
    }

    double hot_predict_threshold() const {
        return prediction_calibrator.threshold();
    }

    double hot_predict_threshold_target() const {
        return prediction_calibrator.target_threshold();
    }

    size_t prediction_calibration_size() const {
        return prediction_calibrator.size();
    }

    double prediction_calibration_current_accuracy() const {
        return prediction_calibrator.current_accuracy();
    }

    double prediction_calibration_target_accuracy() const {
        return prediction_calibrator.target_accuracy();
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

    static double otsu_sample_confidence_for(size_t sample_count) {
        if (sample_count <= HP_OTSU_MIN_OBJECTS) {
            return 0.0;
        }
        if (sample_count >= HP_OTSU_FULL_CONFIDENCE_OBJECTS) {
            return 1.0;
        }
        return static_cast<double>(sample_count - HP_OTSU_MIN_OBJECTS) /
            static_cast<double>(
                HP_OTSU_FULL_CONFIDENCE_OBJECTS - HP_OTSU_MIN_OBJECTS);
    }

    static double otsu_sharpness_confidence_for(
            const HpOtsuResult& result) {
        if (result.occupied_score_range <= 0.0) {
            return 0.0;
        }
        const double plateau_ratio = result.near_optimal_score_range /
            result.occupied_score_range;
        return std::clamp(
            1.0 - plateau_ratio / HP_OTSU_SHARPNESS_FULL_WIDTH_RATIO,
            0.0,
            1.0);
    }

    static double otsu_total_confidence_for(
            double separation_confidence,
            double sample_confidence,
            double sharpness_confidence) {
        separation_confidence = std::clamp(
            separation_confidence, 0.0, 1.0);
        sample_confidence = std::clamp(sample_confidence, 0.0, 1.0);
        sharpness_confidence = std::clamp(
            sharpness_confidence, 0.0, 1.0);
        if (separation_confidence == 0.0 ||
            sample_confidence == 0.0 ||
            sharpness_confidence == 0.0) {
            return 0.0;
        }
        return std::pow(
                separation_confidence,
                HP_OTSU_SEPARATION_CONFIDENCE_WEIGHT) *
            std::pow(
                sample_confidence,
                HP_OTSU_SAMPLE_CONFIDENCE_WEIGHT) *
            std::pow(
                sharpness_confidence,
                HP_OTSU_SHARPNESS_CONFIDENCE_WEIGHT);
    }

    void update_hot_threshold(uint64_t timestamp) {
        maintain_otsu_lower_bound(timestamp);

        const size_t object_count = otsu_histogram.size();
        otsu_sample_confidence = otsu_sample_confidence_for(object_count);
        if (object_count < HP_OTSU_MIN_OBJECTS) {
            clear_otsu_candidate_state();
            hot_threshold_method = HP_THRESHOLD_METHOD_INITIALIZING;
            return;
        }

        auto result = otsu_histogram.otsu_result();
        if (!result.has_value()) {
            clear_otsu_candidate_state();
            otsu_sample_confidence = otsu_sample_confidence_for(object_count);
            hot_threshold_method = HP_THRESHOLD_METHOD_HOLDING;
            return;
        }

        otsu_candidate_threshold = std::clamp(
            from_heat_score(result->threshold_score, timestamp),
            HP_OTSU_HEAT_MIN,
            HP_OTSU_HEAT_MAX);
        otsu_separation = result->separation;
        otsu_sharpness_confidence =
            otsu_sharpness_confidence_for(*result);
        otsu_confidence = otsu_total_confidence_for(
            otsu_separation,
            otsu_sample_confidence,
            otsu_sharpness_confidence);
        const double gain = HP_OTSU_MAX_UPDATE_ALPHA * otsu_confidence;
        if (gain <= std::numeric_limits<double>::epsilon()) {
            hot_threshold_method = HP_THRESHOLD_METHOD_HOLDING;
            return;
        }

        const double effective_score = to_heat_score(hot_threshold, timestamp);
        const double next_effective_score = effective_score +
            gain * (result->threshold_score - effective_score);
        hot_threshold = std::clamp(
            from_heat_score(next_effective_score, timestamp),
            HP_OTSU_HEAT_MIN,
            HP_OTSU_HEAT_MAX);
        hot_threshold_method = HP_THRESHOLD_METHOD_TRACKING;
    }

    void record_object_heat(uint64_t key, double heat, uint64_t timestamp) {
        if (threshold_window_capacity == 0) {
            return;
        }

        maintain_otsu_lower_bound(timestamp);

        auto old = threshold_entries_by_key.find(key);
        if (old != threshold_entries_by_key.end()) {
            erase_threshold_entry(old, timestamp);
        }

        auto entry = std::make_pair(
            threshold_score_for_heat(heat, timestamp),
            ++pbds_counter);

        threshold_order_stats.insert(entry);
        otsu_histogram.insert(histogram_score_for_threshold_score(
            entry.first, timestamp));
        threshold_order.push_back(key);
        threshold_entries_by_key[key] = ThresholdWindowEntry{
            entry,
            std::prev(threshold_order.end())
        };

        while (threshold_entries_by_key.size() > threshold_window_capacity) {
            ceph_assert(!threshold_order.empty());
            uint64_t victim = threshold_order.front();
            auto victim_it = threshold_entries_by_key.find(victim);
            ceph_assert(victim_it != threshold_entries_by_key.end());
            erase_threshold_entry(victim_it, timestamp);
        }

        ++threshold_observation_count;
        if (threshold_entries_by_key.size() <= HP_OTSU_EAGER_OBJECTS ||
            threshold_observation_count % HP_OTSU_UPDATE_INTERVAL == 0) {
            update_hot_threshold(timestamp);
        }
    }

    double object_heat_percentile(uint64_t key) const {
        auto entry = threshold_entries_by_key.find(key);
        if (entry == threshold_entries_by_key.end() ||
            threshold_order_stats.empty()) {
            return 0.0;
        }
        const auto upper_bound = std::make_pair(
            entry->second.score.first,
            std::numeric_limits<uint64_t>::max());
        const size_t rank = threshold_order_stats.order_of_key(upper_bound);
        return static_cast<double>(rank) /
            static_cast<double>(threshold_order_stats.size());
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
                    0,
                    lru_list.end()
                });
            ceph_assert(ok);
            it = inserted;
            item.last_access_distance = 0;
        } else {
            HeatState& state = it->second;
            item.last_access_distance = item.index - state.last_access;
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
        item.past_window_access_count = state.pending_count;
        item.recent_window_access_count = state.short_count;
        item.heat_percentile = 0.0;
        record_object_heat(item.key, state.heat, item.index);
#if HP_ENABLE_HEAT_PERCENTILE
        item.heat_percentile = object_heat_percentile(item.key);
#endif
        item.hot_threshold = hot_threshold;

#if HP_ENABLE_ACCESS_ACCELERATION
        it->second.short_count++;
        short_window_keys.push_back(item.key);
        if (short_window_keys.size() > short_window_capacity) {
            const uint64_t expired_key = short_window_keys.front();
            short_window_keys.pop_front();
            auto expired = heat_map.find(expired_key);
            ceph_assert(expired != heat_map.end());
            ceph_assert(expired->second.short_count > 0);
            expired->second.short_count--;
        }
#endif
    }

    bool can_reserve_prediction() const {
        return pending_queue.size() < evaluation_window ||
            pending_queue.front().ready;
    }

    PredictionReservation reserve_prediction(TraceItem item) {
        PendingSlot *slot = nullptr;
        auto evaluated = enqueue_impl(std::move(item), false, &slot);
        return PredictionReservation{slot, std::move(evaluated)};
    }

    bool complete_prediction(
            PendingSlot *slot,
            double pred_hot_proba,
            int pred) {
        ceph_assert(slot != nullptr);
        ceph_assert(!slot->ready);
        bool should_notify =
            pending_queue.size() >= evaluation_window &&
            slot == &pending_queue.front();
        slot->item.pred_hot_proba = pred_hot_proba;
        slot->item.pred = pred;
        slot->ready = true;
        return should_notify;
    }

    std::optional<EvaluatedItem> enqueue(TraceItem item) {
        PendingSlot *slot = nullptr;
        return enqueue_impl(std::move(item), true, &slot);
    }

    size_t pending_size() const { return pending_queue.size(); }
    size_t heat_state_size() const { return heat_map.size(); }
    size_t lru_size() const { return lru_list.size(); }
    size_t otsu_histogram_bin_count() const {
        return otsu_histogram.bin_count();
    }
    size_t otsu_histogram_object_count() const {
        return otsu_histogram.size();
    }

private:
    std::optional<EvaluatedItem> enqueue_impl(
            TraceItem item,
            bool prediction_ready,
            PendingSlot **current_slot) {
        ceph_assert(current_slot != nullptr);
        ceph_assert(can_reserve_prediction());
        auto state_it = heat_map.find(item.key);
        ceph_assert(state_it != heat_map.end());
        state_it->second.pending_count++;
        pending_queue.emplace_back(std::move(item), prediction_ready);

        if (pending_queue.size() <= evaluation_window) {
            *current_slot = &pending_queue.back();
            return std::nullopt;
        }

        ceph_assert(pending_queue.front().ready);
        TraceItem expired = pending_queue.front().item;
        pending_queue.pop_front();
        *current_slot = &pending_queue.back();

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
        prediction_calibrator.observe(expired.pred_hot_proba, label);
        double training_weight =
            label == 1 ? HP_HOT_CLASS_WEIGHT : 1.0;

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
#if HP_ENABLE_ACCESS_ACCELERATION
            ceph_assert(victim_it->second.short_count == 0);
#endif
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
    double otsu_score_min(uint64_t timestamp) const {
        return to_heat_score(HP_OTSU_HEAT_MIN, timestamp);
    }

    double threshold_score_for_heat(
            double heat,
            uint64_t timestamp) const {
        return to_heat_score(heat, timestamp);
    }

    double histogram_score_for_threshold_score(
            double score,
            uint64_t timestamp) const {
        return std::max(score, otsu_score_min(timestamp));
    }

    void erase_threshold_entry(
            std::unordered_map<uint64_t, ThresholdWindowEntry>::iterator it,
            uint64_t timestamp) {
        threshold_order_stats.erase(it->second.score);
        otsu_histogram.erase(histogram_score_for_threshold_score(
            it->second.score.first, timestamp));
        threshold_order.erase(it->second.order_position);
        threshold_entries_by_key.erase(it);
    }

    void maintain_otsu_lower_bound(uint64_t timestamp) {
        otsu_histogram.clamp_lower_bound(otsu_score_min(timestamp));
    }

    void clear_otsu_candidate_state() {
        otsu_candidate_threshold = 0.0;
        otsu_separation = 0.0;
        otsu_confidence = 0.0;
        otsu_sharpness_confidence = 0.0;
    }
};

#endif
