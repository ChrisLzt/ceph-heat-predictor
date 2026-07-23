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

#include "common/debug.h"
#include "hp_config.h"
#include "hp_expiry_heap.h"
#include "hp_features.h"
#include "hp_score_otsu_histogram.h"
#include "hp_types.h"

class EvaluationQueue {
public:
    enum class ExpiryScheduleState {
        empty,
        waiting_deadline,
        due
    };

    struct ExpirySchedule {
        ExpiryScheduleState state;
        uint64_t deadline_ns;
    };

    struct PendingEvaluation {
        PredictionSample item;
        bool prediction_complete;
        bool label_complete;
        uint64_t enqueue_time_ns;
        int actual_label;
        uint64_t future_window_access_count;
        uint64_t label_deadline_ns;
        uint64_t label_completion_time_ns;
        double label_heat;
        double label_heat_threshold;
        bool cold_start_fallback;

        PendingEvaluation(PredictionSample item, bool prediction_complete, uint64_t enqueue_time_ns = 0) :
                item(std::move(item)),
                prediction_complete(prediction_complete),
                label_complete(false),
                enqueue_time_ns(enqueue_time_ns),
                actual_label(0),
                future_window_access_count(0),
                label_deadline_ns(0),
                label_completion_time_ns(0),
                label_heat(0.0),
                label_heat_threshold(HP_HEAT_INCREMENT),
                cold_start_fallback(false) {}
        PendingEvaluation(const PendingEvaluation&) = delete;
        PendingEvaluation& operator=(const PendingEvaluation&) = delete;
    };

    using PendingIterator = std::list<PendingEvaluation>::iterator;

    struct PredictionReservation {
        PendingIterator position;
        bool accepted;
    };

    struct AccessWindowEntry {
        uint64_t object_key_hash;
        uint64_t access_time_ns;
    };

    struct ThresholdHeatVersion {
        uint64_t effective_time_ns;
        double heat;
    };

    double heat_label_threshold;
    double otsu_candidate_threshold;
    double otsu_candidate_threshold_score = 0.0;
    bool otsu_candidate_available = false;
    uint64_t hot_threshold_method;
    bool otsu_ema_time_initialized = false;
    uint64_t last_otsu_ema_update_time_ns = 0;
    bool otsu_recompute_time_initialized = false;
    uint64_t last_otsu_recompute_time_ns = 0;
    double heat_decay_log_factor_per_ns;
    double heat_increment;
    uint64_t heat_decay_horizon_ns;
    uint64_t future_label_window_ns;
    size_t pending_evaluation_capacity;
    uint64_t evaluation_drop_count_value = 0;
    size_t heat_state_peak_count_value = 0;
    uint64_t lru_eviction_count_value = 0;
    uint64_t short_access_window_ns;
    size_t lru_capacity;

    std::list<PendingEvaluation> pending_evaluations;
    std::list<PendingEvaluation>::iterator next_deadline;
    size_t pending_deadline_count = 0;
    std::deque<AccessWindowEntry> short_access_window_entries;
    std::deque<ThresholdHeatVersion> threshold_heat_history;
    std::unordered_map<uint64_t, ObjectHeatState> heat_map;
    std::list<uint64_t> lru_list;

    size_t heat_label_threshold_object_capacity;
    struct ThresholdWindowEntry {
        HpScoreOtsuHistogram::AbsoluteBin otsu_bin;
        std::list<uint64_t>::iterator order_position;
    };
    std::list<uint64_t> threshold_order;
    std::unordered_map<uint64_t, ThresholdWindowEntry> threshold_entries_by_key;
    HpExpiryHeap threshold_expiry_heap;
    HpScoreOtsuHistogram score_otsu_histogram;
    uint64_t threshold_observation_count = 0;

    EvaluationQueue(
            uint64_t heat_decay_horizon_ns = HP_HEAT_DECAY_HORIZON_NS,
            size_t lru_capacity = HP_LRU_CAPACITY,
            double heat_label_threshold = HP_HEAT_INCREMENT,
            double heat_increment = HP_HEAT_INCREMENT,
            uint64_t short_access_window_ns =
                HP_SHORT_ACCESS_WINDOW_NS,
            uint64_t future_label_window_ns = HP_FUTURE_LABEL_WINDOW_NS,
            size_t pending_evaluation_capacity = HP_PENDING_EVALUATION_CAPACITY) :
            heat_label_threshold(heat_label_threshold),
            otsu_candidate_threshold(0.0),
            hot_threshold_method(HP_THRESHOLD_METHOD_INITIALIZING),
            heat_decay_log_factor_per_ns(
                hp_heat_decay_log_factor_per_ns(heat_decay_horizon_ns)),
            heat_increment(heat_increment),
            heat_decay_horizon_ns(heat_decay_horizon_ns),
            future_label_window_ns(future_label_window_ns),
            pending_evaluation_capacity(pending_evaluation_capacity),
            short_access_window_ns(short_access_window_ns),
            lru_capacity(lru_capacity),
            next_deadline(pending_evaluations.end()),
            heat_label_threshold_object_capacity(
                HP_HEAT_LABEL_THRESHOLD_OBJECT_CAPACITY) {
#if HP_ENABLE_ACCESS_RATE_CHANGE
        ceph_assert(short_access_window_ns > 0);
#endif
    }

    double decay_heat(
            double last_heat, uint64_t last_ts, uint64_t cur_ts) const {
        if (cur_ts <= last_ts) {
            return last_heat;
        }
        const uint64_t delta = cur_ts - last_ts;
        const double factor = std::exp(
            static_cast<double>(delta) * heat_decay_log_factor_per_ns);
        return factor * last_heat;
    }

    double to_heat_score(double heat, uint64_t timestamp) const {
        double positive_heat = std::max(
            heat, std::numeric_limits<double>::min());
        return std::log(positive_heat) -
            heat_decay_log_factor_per_ns * static_cast<double>(timestamp);
    }

    double from_heat_score(double score, uint64_t timestamp) const {
        double log_heat = score +
            heat_decay_log_factor_per_ns * static_cast<double>(timestamp);
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

    double heat_label_threshold_at(uint64_t timestamp) const {
        (void)timestamp;
        return heat_label_threshold;
    }

    double otsu_candidate_threshold_at(uint64_t timestamp) const {
        (void)timestamp;
        if (!otsu_candidate_available) {
            return 0.0;
        }
        return otsu_candidate_threshold;
    }

    double heat_label_threshold_for_label(uint64_t deadline_ns) const {
        if (threshold_heat_history.empty()) {
            return heat_label_threshold_at(deadline_ns);
        }
        auto upper = std::upper_bound(
            threshold_heat_history.begin(),
            threshold_heat_history.end(),
            deadline_ns,
            [](uint64_t timestamp, const ThresholdHeatVersion& version) {
                return timestamp < version.effective_time_ns;
            });
        if (upper == threshold_heat_history.begin()) {
            return upper->heat;
        }
        --upper;
        return upper->heat;
    }

    static double ema_gain_for_elapsed(
            double reference_gain,
            uint64_t elapsed_ns) {
        reference_gain = std::clamp(reference_gain, 0.0, 1.0);
        if (reference_gain == 0.0 || elapsed_ns == 0) {
            return 0.0;
        }
        if (reference_gain == 1.0) {
            return 1.0;
        }
        const double intervals = static_cast<double>(elapsed_ns) /
            static_cast<double>(HP_OTSU_EMA_REFERENCE_INTERVAL_NS);
        return 1.0 - std::pow(1.0 - reference_gain, intervals);
    }

    void update_hot_threshold(uint64_t timestamp) {
        initialize_heat_threshold(timestamp);
        maintain_score_otsu_lower_bound(timestamp);
        const size_t vote_count = score_otsu_histogram.size();
        if (vote_count < HP_OTSU_MIN_VOTES) {
            clear_otsu_candidate_state();
            hot_threshold_method = HP_THRESHOLD_METHOD_INITIALIZING;
            heat_label_threshold = heat_label_threshold_at(timestamp);
            return;
        }

        auto result = score_otsu_histogram.otsu_result();
        if (!result.has_value()) {
            clear_otsu_candidate_state();
            hot_threshold_method = HP_THRESHOLD_METHOD_HOLDING;
            return;
        }

        otsu_candidate_threshold_score = result->threshold_score;
        otsu_candidate_available = true;
        otsu_candidate_threshold =
            HpScoreOtsuHistogram::heat_for_score_at(
                otsu_candidate_threshold_score,
                timestamp,
                heat_decay_log_factor_per_ns);
        const uint64_t effective_timestamp = otsu_ema_time_initialized
            ? std::max(timestamp, last_otsu_ema_update_time_ns)
            : timestamp;
        const uint64_t elapsed_ns = otsu_ema_time_initialized
            ? effective_timestamp - last_otsu_ema_update_time_ns
            : HP_OTSU_EMA_REFERENCE_INTERVAL_NS;
        const double gain = ema_gain_for_elapsed(
            HP_OTSU_EMA_ALPHA, elapsed_ns);
        otsu_ema_time_initialized = true;
        last_otsu_ema_update_time_ns = effective_timestamp;
        if (gain > std::numeric_limits<double>::epsilon()) {
            const double current_effective_score =
                total_heat_score_for_otsu(
                    heat_label_threshold, effective_timestamp);
            const double next_effective_score = current_effective_score +
                gain * (result->threshold_score - current_effective_score);
            publish_heat_label_threshold(
                HpScoreOtsuHistogram::heat_for_score_at(
                    next_effective_score,
                    effective_timestamp,
                    heat_decay_log_factor_per_ns),
                effective_timestamp);
        }
        hot_threshold_method = HP_THRESHOLD_METHOD_TRACKING;
    }

    void record_object_heat(uint64_t key, double heat, uint64_t timestamp) {
        if (heat_label_threshold_object_capacity == 0) {
            return;
        }

        initialize_heat_threshold(timestamp);
        maintain_score_otsu_lower_bound(timestamp);
        auto old = threshold_entries_by_key.find(key);
        const double bounded_heat = std::clamp(
            heat, HP_OTSU_TOTAL_HEAT_MIN, HP_OTSU_TOTAL_HEAT_MAX);
        if (bounded_heat <= HP_OTSU_TOTAL_HEAT_MIN) {
            if (old != threshold_entries_by_key.end()) {
                erase_threshold_entry(old);
            }
            return;
        }

        const auto otsu_bin = score_otsu_histogram.insert(
            total_heat_score_for_otsu(bounded_heat, timestamp));
        if (old == threshold_entries_by_key.end()) {
            threshold_order.push_back(key);
            threshold_entries_by_key.emplace(
                key,
                ThresholdWindowEntry{
                    otsu_bin,
                    std::prev(threshold_order.end())
                });
        } else {
            score_otsu_histogram.erase(old->second.otsu_bin);
            threshold_order.splice(
                threshold_order.end(),
                threshold_order,
                old->second.order_position);
            old->second.otsu_bin = otsu_bin;
        }
        threshold_expiry_heap.upsert(
            key, otsu_vote_expiry_time(heat, timestamp));

        while (threshold_entries_by_key.size() > heat_label_threshold_object_capacity) {
            ceph_assert(!threshold_order.empty());
            uint64_t victim = threshold_order.front();
            auto victim_it = threshold_entries_by_key.find(victim);
            ceph_assert(victim_it != threshold_entries_by_key.end());
            erase_threshold_entry(victim_it);
        }

        ++threshold_observation_count;
        if (!otsu_recompute_time_initialized) {
            otsu_recompute_time_initialized = true;
            last_otsu_recompute_time_ns = timestamp;
        }
        const bool time_due = timestamp >= last_otsu_recompute_time_ns &&
            timestamp - last_otsu_recompute_time_ns >=
                HP_OTSU_RECOMPUTE_MAX_INTERVAL_NS;
        if (score_otsu_histogram.size() <= HP_OTSU_EAGER_OBJECTS ||
            threshold_observation_count % HP_OTSU_UPDATE_INTERVAL == 0 ||
            time_due) {
            update_hot_threshold(timestamp);
            last_otsu_recompute_time_ns = timestamp;
        }
    }

    void advance_otsu_history(uint64_t now_ns) {
        initialize_heat_threshold(now_ns);
        maintain_score_otsu_lower_bound(now_ns);
        const bool time_due = otsu_recompute_time_initialized &&
            now_ns >= last_otsu_recompute_time_ns &&
            now_ns - last_otsu_recompute_time_ns >=
                HP_OTSU_RECOMPUTE_MAX_INTERVAL_NS;
        if (time_due) {
            update_hot_threshold(now_ns);
            last_otsu_recompute_time_ns = now_ns;
        }
    }

    void prepare_features(PredictionSample& item, uint64_t now_ns) {
        expire_due_access_windows(now_ns);
        advance_otsu_history(now_ns);
        auto it = heat_map.find(item.object_key_hash);
        if (it == heat_map.end()) {
            auto [inserted, ok] = heat_map.emplace(
                item.object_key_hash,
                ObjectHeatState{
                    heat_increment,
                    now_ns,
                    1,
                    0,
                    0,
                    lru_list.end()
                });
            ceph_assert(ok);
            it = inserted;
            heat_state_peak_count_value = std::max(
                heat_state_peak_count_value, heat_map.size());
            item.time_since_previous_access_ns = 0;
        } else {
            ObjectHeatState& state = it->second;
            item.time_since_previous_access_ns = now_ns >= state.last_access_time_ns
                ? now_ns - state.last_access_time_ns
                : 0;
            if (state.lru_position != lru_list.end()) {
                ceph_assert(state.pending_evaluation_count == 0);
                ceph_assert(state.short_window_access_count == 0);
                lru_list.erase(state.lru_position);
                state.lru_position = lru_list.end();
            }
            state.heat =
                decay_heat(state.heat, state.last_access_time_ns, now_ns) +
                heat_increment;
            state.last_access_time_ns = now_ns;
            state.tracked_access_count++;
        }

        const ObjectHeatState& state = it->second;
        item.heat_after_current_access = state.heat;
        item.tracked_access_count = state.tracked_access_count;
        item.long_window_access_count = state.pending_evaluation_count;
        item.short_window_access_count =
            state.short_window_access_count;
        record_object_heat(item.object_key_hash, state.heat, now_ns);
        item.heat_label_threshold_at_prediction =
            heat_label_threshold_at(now_ns);

#if HP_ENABLE_ACCESS_RATE_CHANGE
        it->second.short_window_access_count++;
        short_access_window_entries.push_back(
            AccessWindowEntry{item.object_key_hash, now_ns});
#endif
    }

    std::vector<EvaluatedSample> expire_due_evaluations(uint64_t now_ns) {
        std::vector<EvaluatedSample> evaluated;
        while (next_deadline != pending_evaluations.end()) {
            const PendingEvaluation& slot = *next_deadline;
            if (now_ns < slot.enqueue_time_ns ||
                now_ns - slot.enqueue_time_ns < future_label_window_ns) {
                break;
            }
            const uint64_t deadline_ns = slot.enqueue_time_ns >
                    std::numeric_limits<uint64_t>::max() -
                        future_label_window_ns
                ? std::numeric_limits<uint64_t>::max()
                : slot.enqueue_time_ns + future_label_window_ns;
            advance_otsu_history(deadline_ns);
            auto due = next_deadline++;
            auto completed = evaluate_deadline(due, now_ns);
            if (completed.has_value()) {
                evaluated.push_back(std::move(*completed));
            }
        }
        return evaluated;
    }

    void expire_due_access_windows(uint64_t now_ns) {
#if HP_ENABLE_ACCESS_RATE_CHANGE
        expire_short_access_window(now_ns);
#else
        (void)now_ns;
#endif
    }

    ExpirySchedule expiry_schedule(uint64_t now_ns) const {
        std::optional<uint64_t> earliest_deadline;
        const auto include_deadline = [&earliest_deadline](
                uint64_t start_ns, uint64_t duration_ns) {
            const uint64_t deadline_ns = start_ns >
                    std::numeric_limits<uint64_t>::max() - duration_ns
                ? std::numeric_limits<uint64_t>::max()
                : start_ns + duration_ns;
            if (!earliest_deadline.has_value() ||
                deadline_ns < *earliest_deadline) {
                earliest_deadline = deadline_ns;
            }
        };

        if (next_deadline != pending_evaluations.end()) {
            include_deadline(
                next_deadline->enqueue_time_ns, future_label_window_ns);
        }
#if HP_ENABLE_ACCESS_RATE_CHANGE
        if (!short_access_window_entries.empty()) {
            include_deadline(
                short_access_window_entries.front().access_time_ns,
                short_access_window_ns);
        }
#endif
        if (auto otsu_deadline =
                threshold_expiry_heap.earliest_deadline_ns()) {
            if (!earliest_deadline.has_value() ||
                *otsu_deadline < *earliest_deadline) {
                earliest_deadline = *otsu_deadline;
            }
        }
        if (!earliest_deadline.has_value()) {
            return ExpirySchedule{ExpiryScheduleState::empty, 0};
        }
        if (now_ns < *earliest_deadline) {
            return ExpirySchedule{
                ExpiryScheduleState::waiting_deadline, *earliest_deadline};
        }
        return ExpirySchedule{ExpiryScheduleState::due, *earliest_deadline};
    }

    std::vector<EvaluatedSample> expire_before_prepare(
            PredictionSample& item,
            uint64_t now_ns) {
        auto evaluated = expire_due_evaluations(now_ns);
        prepare_features(item, now_ns);
        return evaluated;
    }

    PredictionReservation reserve_prediction(
            PredictionSample item,
            uint64_t now_ns) {
        PendingIterator position = enqueue_time_impl(
            std::move(item), false, now_ns);
        const bool accepted = position != pending_evaluations.end();
        return PredictionReservation{position, accepted};
    }

    bool enqueue(PredictionSample item, uint64_t now_ns) {
        return enqueue_time_impl(std::move(item), true, now_ns) !=
            pending_evaluations.end();
    }

    uint64_t evaluation_drop_count() const {
        return evaluation_drop_count_value;
    }

    std::vector<EvaluatedSample> complete_prediction(
            PendingIterator position,
            double predicted_hot_probability,
            int predicted_label,
            bool cold_start_fallback = false) {
        ceph_assert(position != pending_evaluations.end());
        ceph_assert(!position->prediction_complete);
        position->item.predicted_hot_probability = predicted_hot_probability;
        position->item.predicted_label = predicted_label;
        position->cold_start_fallback = cold_start_fallback;
        position->prediction_complete = true;
        if (!position->label_complete) {
            return {};
        }
        std::vector<EvaluatedSample> completed;
        completed.push_back(finalize_evaluation(position));
        return completed;
    }

    void cancel_prediction(PendingIterator position) {
        ceph_assert(position != pending_evaluations.end());

        if (!position->label_complete) {
            ceph_assert(pending_deadline_count > 0);
            auto state = heat_map.find(position->item.object_key_hash);
            ceph_assert(state != heat_map.end());
            ceph_assert(state->second.pending_evaluation_count > 0);
            state->second.pending_evaluation_count--;
            pending_deadline_count--;
            if (position == next_deadline) {
                next_deadline = std::next(position);
            }
            make_idle_if_unprotected(position->item.object_key_hash,
                                     state->second);
        }
        ++evaluation_drop_count_value;
        pending_evaluations.erase(position);
        prune_threshold_heat_history();
    }

    size_t pending_size() const { return pending_deadline_count; }
    size_t awaiting_prediction_size() const {
        ceph_assert(pending_evaluations.size() >= pending_deadline_count);
        return pending_evaluations.size() - pending_deadline_count;
    }
    size_t heat_state_size() const { return heat_map.size(); }
    size_t lru_size() const { return lru_list.size(); }
    size_t protected_heat_state_size() const {
        ceph_assert(heat_map.size() >= lru_list.size());
        return heat_map.size() - lru_list.size();
    }
    size_t heat_state_peak_size() const {
        return heat_state_peak_count_value;
    }
    uint64_t lru_eviction_count() const {
        return lru_eviction_count_value;
    }
    size_t otsu_histogram_bin_count() const {
        return score_otsu_histogram.bin_count();
    }
    size_t otsu_histogram_vote_count() const {
        return score_otsu_histogram.size();
    }

private:
    void expire_short_access_window(uint64_t now_ns) {
        while (!short_access_window_entries.empty()) {
            const AccessWindowEntry& entry = short_access_window_entries.front();
            if (now_ns < entry.access_time_ns ||
                now_ns - entry.access_time_ns < short_access_window_ns) {
                break;
            }

            const uint64_t expired_key = entry.object_key_hash;
            short_access_window_entries.pop_front();
            auto expired = heat_map.find(expired_key);
            ceph_assert(expired != heat_map.end());
            ceph_assert(expired->second.short_window_access_count > 0);
            expired->second.short_window_access_count--;
            make_idle_if_unprotected(expired_key, expired->second);
        }
    }

    void enforce_lru_capacity() {
        while (lru_list.size() > lru_capacity) {
            uint64_t victim = lru_list.front();
            lru_list.pop_front();
            auto victim_it = heat_map.find(victim);
            ceph_assert(victim_it != heat_map.end());
            ceph_assert(victim_it->second.pending_evaluation_count == 0);
            ceph_assert(
                victim_it->second.short_window_access_count == 0);
            heat_map.erase(victim_it);
            ++lru_eviction_count_value;
        }
    }

    void make_idle_if_unprotected(uint64_t key, ObjectHeatState& state) {
        if (state.pending_evaluation_count != 0 ||
            state.short_window_access_count != 0) {
            return;
        }
        if (state.lru_position == lru_list.end()) {
            lru_list.push_back(key);
            state.lru_position = std::prev(lru_list.end());
        }
        enforce_lru_capacity();
    }

    std::optional<EvaluatedSample> evaluate_deadline(
            std::list<PendingEvaluation>::iterator position,
            uint64_t now_ns) {
        PendingEvaluation& expired = *position;
        ceph_assert(!expired.label_complete);
        ceph_assert(pending_deadline_count > 0);

        auto expired_state_it = heat_map.find(expired.item.object_key_hash);
        ceph_assert(expired_state_it != heat_map.end());
        ObjectHeatState& expired_state = expired_state_it->second;
        ceph_assert(expired_state.pending_evaluation_count > 0);

        const uint64_t deadline_ns = expired.enqueue_time_ns >
                std::numeric_limits<uint64_t>::max() - future_label_window_ns
            ? std::numeric_limits<uint64_t>::max()
            : expired.enqueue_time_ns + future_label_window_ns;
        double expired_total_heat = decay_heat(
            expired_state.heat, expired_state.last_access_time_ns, deadline_ns);
        expired.label_heat = expired_total_heat;
        expired.label_heat_threshold =
            heat_label_threshold_for_label(deadline_ns);
        expired.actual_label = expired.label_heat >
            expired.label_heat_threshold ? 1 : 0;
        expired.label_deadline_ns = deadline_ns;
        expired.label_completion_time_ns = now_ns;
        expired.future_window_access_count =
            expired_state.tracked_access_count -
            expired.item.tracked_access_count;
        expired.label_complete = true;

        expired_state.pending_evaluation_count--;
        pending_deadline_count--;
        make_idle_if_unprotected(
            expired.item.object_key_hash, expired_state);
        prune_threshold_heat_history();

        if (!expired.prediction_complete) {
            return std::nullopt;
        }
        return finalize_evaluation(position);
    }

    EvaluatedSample finalize_evaluation(
            std::list<PendingEvaluation>::iterator position) {
        PendingEvaluation& completed = *position;
        ceph_assert(completed.prediction_complete);
        ceph_assert(completed.label_complete);

        EvaluatedSample evaluated{
            std::move(completed.item),
            completed.actual_label,
            completed.future_window_access_count,
            completed.enqueue_time_ns,
            completed.label_deadline_ns,
            completed.label_completion_time_ns,
            completed.label_heat,
            completed.label_heat_threshold,
            completed.cold_start_fallback
        };
        pending_evaluations.erase(position);
        return evaluated;
    }

    PendingIterator enqueue_time_impl(
            PredictionSample item,
            bool prediction_complete,
            uint64_t now_ns) {
        auto state_it = heat_map.find(item.object_key_hash);
        ceph_assert(state_it != heat_map.end());
        if (pending_deadline_count >= pending_evaluation_capacity) {
            ++evaluation_drop_count_value;
            make_idle_if_unprotected(item.object_key_hash, state_it->second);
            return pending_evaluations.end();
        }

        state_it->second.pending_evaluation_count++;
        const bool needs_deadline_head =
            next_deadline == pending_evaluations.end();
        pending_evaluations.emplace_back(
            std::move(item), prediction_complete, now_ns);
        auto inserted = std::prev(pending_evaluations.end());
        if (needs_deadline_head) {
            next_deadline = inserted;
        }
        pending_deadline_count++;
        return inserted;
    }

    double total_heat_score_for_otsu(
            double heat,
            uint64_t timestamp) const {
        return HpScoreOtsuHistogram::score_for_heat_at(
            heat, timestamp, heat_decay_log_factor_per_ns);
    }

    void initialize_heat_threshold(uint64_t timestamp) {
        if (!threshold_heat_history.empty()) {
            return;
        }
        heat_label_threshold = std::clamp(
            heat_label_threshold,
            HP_OTSU_TOTAL_HEAT_MIN,
            HP_OTSU_TOTAL_HEAT_MAX);
        publish_heat_label_threshold(heat_label_threshold, timestamp);
    }

    void publish_heat_label_threshold(
            double heat,
            uint64_t effective_time_ns) {
        heat_label_threshold = std::clamp(
            heat, HP_OTSU_TOTAL_HEAT_MIN, HP_OTSU_TOTAL_HEAT_MAX);
        if (!threshold_heat_history.empty() &&
            threshold_heat_history.back().effective_time_ns ==
                effective_time_ns) {
            threshold_heat_history.back().heat = heat_label_threshold;
            return;
        }
        ceph_assert(threshold_heat_history.empty() ||
                    threshold_heat_history.back().effective_time_ns <
                        effective_time_ns);
        threshold_heat_history.push_back(
            ThresholdHeatVersion{effective_time_ns, heat_label_threshold});
    }

    void prune_threshold_heat_history() {
        if (threshold_heat_history.size() <= 1) {
            return;
        }
        if (next_deadline == pending_evaluations.end()) {
            const ThresholdHeatVersion latest = threshold_heat_history.back();
            threshold_heat_history.clear();
            threshold_heat_history.push_back(latest);
            return;
        }
        const uint64_t earliest_required_deadline =
            next_deadline->enqueue_time_ns >
                    std::numeric_limits<uint64_t>::max() -
                        future_label_window_ns
                ? std::numeric_limits<uint64_t>::max()
                : next_deadline->enqueue_time_ns + future_label_window_ns;
        while (threshold_heat_history.size() > 1 &&
               threshold_heat_history[1].effective_time_ns <=
                   earliest_required_deadline) {
            threshold_heat_history.pop_front();
        }
    }

    void maintain_score_otsu_lower_bound(uint64_t timestamp) {
        score_otsu_histogram.advance_lower_bound(
            total_heat_score_for_otsu(HP_OTSU_TOTAL_HEAT_MIN, timestamp));
        while (auto expired_key = threshold_expiry_heap.due_key(timestamp)) {
            auto entry = threshold_entries_by_key.find(*expired_key);
            ceph_assert(entry != threshold_entries_by_key.end());
            erase_threshold_entry(entry);
        }
    }

    uint64_t otsu_vote_expiry_time(
            double heat,
            uint64_t timestamp) const {
        if (heat <= HP_OTSU_TOTAL_HEAT_MIN) {
            return timestamp;
        }
        if (!std::isfinite(heat)) {
            return std::numeric_limits<uint64_t>::max();
        }
        const double decay_ns = std::log(
            HP_OTSU_TOTAL_HEAT_MIN / heat) /
            heat_decay_log_factor_per_ns;
        ceph_assert(std::isfinite(decay_ns) && decay_ns > 0.0);
        const double remaining = static_cast<double>(
            std::numeric_limits<uint64_t>::max() - timestamp);
        if (decay_ns >= remaining) {
            return std::numeric_limits<uint64_t>::max();
        }
        return timestamp + static_cast<uint64_t>(std::ceil(decay_ns));
    }

    void erase_threshold_entry(
            std::unordered_map<uint64_t, ThresholdWindowEntry>::iterator it) {
        score_otsu_histogram.erase(it->second.otsu_bin);
        threshold_order.erase(it->second.order_position);
        ceph_assert(threshold_expiry_heap.erase(it->first));
        threshold_entries_by_key.erase(it);
    }

    void clear_otsu_candidate_state() {
        otsu_candidate_threshold = 0.0;
        otsu_candidate_threshold_score = 0.0;
        otsu_candidate_available = false;
        otsu_ema_time_initialized = false;
        last_otsu_ema_update_time_ns = 0;
    }
};

#endif
