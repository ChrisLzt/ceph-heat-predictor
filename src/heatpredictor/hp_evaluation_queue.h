#ifndef CEPH_HEATPREDICTOR_HP_EVALUATION_QUEUE_H
#define CEPH_HEATPREDICTOR_HP_EVALUATION_QUEUE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <list>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/debug.h"
#include "hp_config.h"
#include "hp_features.h"
#include "hp_future_access_threshold.h"
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

private:
    struct PendingEvaluation {
        PredictionSample item;
        bool prediction_complete;
        bool label_complete;
        uint64_t enqueue_time_ns;
        int actual_label;
        uint64_t future_window_access_count;
        uint64_t label_deadline_ns;
        uint64_t label_completion_time_ns;
        bool cold_start_fallback;

        PendingEvaluation(
                PredictionSample item,
                bool prediction_complete,
                uint64_t enqueue_time_ns = 0) :
                item(std::move(item)),
                prediction_complete(prediction_complete),
                label_complete(false),
                enqueue_time_ns(enqueue_time_ns),
                actual_label(0),
                future_window_access_count(0),
                label_deadline_ns(0),
                label_completion_time_ns(0),
                cold_start_fallback(false) {}
        PendingEvaluation(const PendingEvaluation&) = delete;
        PendingEvaluation& operator=(const PendingEvaluation&) = delete;
    };

    struct ShortAccessEvent {
        uint64_t object_key_hash;
        uint64_t timestamp_ns;
    };

    using PendingIterator = std::list<PendingEvaluation>::iterator;

public:
    class PredictionTicket {
        friend class EvaluationQueue;

        PendingIterator position;
        bool valid;

        explicit PredictionTicket(PendingIterator position) :
                position(position),
                valid(true) {}

    public:
        PredictionTicket(const PredictionTicket&) = delete;
        PredictionTicket& operator=(const PredictionTicket&) = delete;
        PredictionTicket(PredictionTicket&& other) noexcept :
                position(other.position),
                valid(other.valid) {
            other.valid = false;
        }
        PredictionTicket& operator=(PredictionTicket&&) = delete;
    };

    struct BeginPredictionResult {
        PredictionSample sample;
        std::vector<EvaluatedSample> evaluated;
        std::optional<PredictionTicket> ticket;
        bool expiry_schedule_changed;
    };

    struct ExpiryMaintenanceResult {
        std::vector<EvaluatedSample> evaluated;
        uint64_t expired_evaluation_count;
        bool threshold_status_changed;
        bool processed;
        ExpirySchedule next_schedule;
    };

private:
    double heat_decay_log_factor_per_ns;
    double heat_increment;
    uint64_t future_label_window_ns;
    uint64_t short_access_window_ns;
    size_t pending_evaluation_capacity;
    size_t lru_capacity;
    uint64_t evaluation_drop_count_value = 0;
    size_t heat_state_peak_count_value = 0;
    uint64_t lru_eviction_count_value = 0;
    uint64_t threshold_holding_sample_count_value = 0;
    uint64_t sparse_threshold_sample_count_value = 0;

    std::list<PendingEvaluation> pending_evaluations;
    PendingIterator next_deadline;
    size_t pending_deadline_count = 0;
    std::unordered_map<uint64_t, ObjectHeatState> heat_map;
    std::list<uint64_t> lru_list;
    std::deque<ShortAccessEvent> short_access_events;
    HpFutureAccessThreshold future_access_threshold;

public:
    EvaluationQueue(
            uint64_t heat_decay_horizon_ns = HP_HEAT_DECAY_HORIZON_NS,
            size_t lru_capacity = HP_LRU_CAPACITY,
            double heat_increment = HP_HEAT_INCREMENT,
            uint64_t future_label_window_ns = HP_FUTURE_LABEL_WINDOW_NS,
            size_t pending_evaluation_capacity =
                HP_PENDING_EVALUATION_CAPACITY,
            uint64_t short_access_window_ns =
                HP_SHORT_ACCESS_WINDOW_NS) :
            heat_decay_log_factor_per_ns(
                hp_heat_decay_log_factor_per_ns(heat_decay_horizon_ns)),
            heat_increment(heat_increment),
            future_label_window_ns(future_label_window_ns),
            short_access_window_ns(short_access_window_ns),
            pending_evaluation_capacity(pending_evaluation_capacity),
            lru_capacity(lru_capacity),
            next_deadline(pending_evaluations.end()) {
        ceph_assert(short_access_window_ns > 0);
        heat_map.reserve(std::min<size_t>(
            lru_capacity, static_cast<size_t>(65536)));
    }

private:
    static uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
        return lhs > std::numeric_limits<uint64_t>::max() - rhs
            ? std::numeric_limits<uint64_t>::max()
            : lhs + rhs;
    }

    double decay_heat(
            double last_heat,
            uint64_t last_timestamp_ns,
            uint64_t current_timestamp_ns) const {
        if (current_timestamp_ns <= last_timestamp_ns) {
            return last_heat;
        }
        const uint64_t elapsed_ns =
            current_timestamp_ns - last_timestamp_ns;
        return std::exp(
            static_cast<double>(elapsed_ns) *
                heat_decay_log_factor_per_ns) *
            last_heat;
    }

    bool next_evaluation_is_due(uint64_t now_ns) const {
        return next_deadline != pending_evaluations.end() &&
            now_ns >= saturating_add(
                next_deadline->enqueue_time_ns,
                future_label_window_ns);
    }

    void expire_short_accesses(uint64_t now_ns) {
        while (!short_access_events.empty() &&
               now_ns >= saturating_add(
                   short_access_events.front().timestamp_ns,
                   short_access_window_ns)) {
            const uint64_t key =
                short_access_events.front().object_key_hash;
            short_access_events.pop_front();
            auto state_position = heat_map.find(key);
            ceph_assert(state_position != heat_map.end());
            ObjectHeatState& state = state_position->second;
            ceph_assert(state.short_window_access_count > 0);
            --state.short_window_access_count;
            make_idle_if_unprotected(key, state);
        }
    }

    void prepare_features(PredictionSample& item, uint64_t now_ns) {
        future_access_threshold.maintain(now_ns);
        expire_short_accesses(now_ns);
        auto state_position = heat_map.find(item.object_key_hash);
        if (state_position == heat_map.end()) {
            auto [inserted, ok] = heat_map.emplace(
                item.object_key_hash,
                ObjectHeatState{
                    heat_increment,
                    now_ns,
                    1,
                    0,
                    0,
                    lru_list.end()});
            ceph_assert(ok);
            state_position = inserted;
            heat_state_peak_count_value = std::max(
                heat_state_peak_count_value, heat_map.size());
            item.time_since_previous_access_ns = 0;
        } else {
            ObjectHeatState& state = state_position->second;
            item.time_since_previous_access_ns =
                now_ns >= state.last_access_time_ns
                    ? now_ns - state.last_access_time_ns
                    : 0;
            if (state.lru_position != lru_list.end()) {
                ceph_assert(state.pending_evaluation_count == 0);
                ceph_assert(state.short_window_access_count == 0);
                lru_list.erase(state.lru_position);
                state.lru_position = lru_list.end();
            }
            state.heat =
                decay_heat(
                    state.heat,
                    state.last_access_time_ns,
                    now_ns) +
                heat_increment;
            state.last_access_time_ns = now_ns;
            ++state.tracked_access_count;
        }

        const ObjectHeatState& state = state_position->second;
        item.heat_after_current_access = state.heat;
        item.future_access_threshold_at_prediction =
            future_access_threshold.current_threshold();
        item.past_window_access_count = state.pending_evaluation_count;
        item.short_window_access_count =
            state.short_window_access_count;
        item.tracked_access_count_after_current_access =
            state.tracked_access_count;
        ++state_position->second.short_window_access_count;
        short_access_events.push_back(
            ShortAccessEvent{item.object_key_hash, now_ns});
    }

public:
    BeginPredictionResult begin_prediction(
            PredictionSample item,
            uint64_t now_ns) {
        const auto schedule_before = expiry_schedule(now_ns);
        std::vector<EvaluatedSample> evaluated;

        // Foreground accounting must not pass an overdue item: otherwise the
        // current access could leak into a right-open historical window.
        while (next_evaluation_is_due(now_ns)) {
            auto batch = expire_due_evaluations(
                now_ns, HP_EXPIRY_MAINTENANCE_BATCH_SIZE);
            ceph_assert(!batch.empty() || !next_evaluation_is_due(now_ns));
            evaluated.insert(
                evaluated.end(),
                std::make_move_iterator(batch.begin()),
                std::make_move_iterator(batch.end()));
        }

        prepare_features(item, now_ns);
        PredictionSample prepared_sample = item;
        PendingIterator position =
            enqueue_time_impl(std::move(item), false, now_ns);
        std::optional<PredictionTicket> ticket =
            position != pending_evaluations.end()
                ? std::optional<PredictionTicket>(
                    PredictionTicket(position))
                : std::nullopt;
        const auto schedule_after = expiry_schedule(now_ns);
        const bool schedule_changed =
            schedule_after.state != ExpiryScheduleState::empty &&
            (schedule_before.state == ExpiryScheduleState::empty ||
             schedule_after.deadline_ns < schedule_before.deadline_ns);
        return BeginPredictionResult{
            std::move(prepared_sample),
            std::move(evaluated),
            std::move(ticket),
            schedule_changed};
    }

private:
    std::vector<EvaluatedSample> expire_due_evaluations(
            uint64_t now_ns,
            size_t max_evaluations) {
        std::vector<EvaluatedSample> evaluated;
        std::vector<HpFutureAccessObservation> observations;
        evaluated.reserve(std::min(max_evaluations, pending_deadline_count));
        observations.reserve(std::min(max_evaluations, pending_deadline_count));

        size_t processed_count = 0;
        while (processed_count < max_evaluations &&
               next_evaluation_is_due(now_ns)) {
            auto due = next_deadline++;
            ++processed_count;
            auto completed = evaluate_deadline(
                due, now_ns, observations);
            if (completed.has_value()) {
                evaluated.push_back(std::move(*completed));
            }
        }
        future_access_threshold.apply_observations(observations, now_ns);
        return evaluated;
    }

    ExpirySchedule expiry_schedule(uint64_t now_ns) const {
        std::optional<uint64_t> earliest_deadline;
        if (next_deadline != pending_evaluations.end()) {
            earliest_deadline = saturating_add(
                next_deadline->enqueue_time_ns,
                future_label_window_ns);
        }
        if (auto threshold_deadline =
                future_access_threshold.maintenance_deadline_ns();
            threshold_deadline.has_value() &&
            (!earliest_deadline.has_value() ||
             *threshold_deadline < *earliest_deadline)) {
            earliest_deadline = *threshold_deadline;
        }
        if (!short_access_events.empty()) {
            const uint64_t short_access_deadline = saturating_add(
                short_access_events.front().timestamp_ns,
                short_access_window_ns);
            if (!earliest_deadline.has_value() ||
                short_access_deadline < *earliest_deadline) {
                earliest_deadline = short_access_deadline;
            }
        }

        if (!earliest_deadline.has_value()) {
            return ExpirySchedule{ExpiryScheduleState::empty, 0};
        }
        if (now_ns < *earliest_deadline) {
            return ExpirySchedule{
                ExpiryScheduleState::waiting_deadline,
                *earliest_deadline};
        }
        return ExpirySchedule{
            ExpiryScheduleState::due,
            *earliest_deadline};
    }

public:
    std::vector<EvaluatedSample> complete_prediction(
            PredictionTicket&& ticket,
            double predicted_hot_probability,
            int predicted_label,
            bool cold_start_fallback = false) {
        ceph_assert(ticket.valid);
        PendingIterator position = ticket.position;
        ticket.valid = false;
        ceph_assert(position != pending_evaluations.end());
        ceph_assert(!position->prediction_complete);
        position->item.predicted_hot_probability =
            predicted_hot_probability;
        position->item.predicted_label = predicted_label;
        position->cold_start_fallback = cold_start_fallback;
        position->prediction_complete = true;
        if (!position->label_complete) {
            return {};
        }
        return {finalize_evaluation(position)};
    }

    void cancel_prediction(
            PredictionTicket&& ticket,
            uint64_t now_ns) {
        ceph_assert(ticket.valid);
        PendingIterator position = ticket.position;
        ticket.valid = false;
        ceph_assert(position != pending_evaluations.end());

        if (!position->label_complete) {
            ceph_assert(pending_deadline_count > 0);
            auto state = heat_map.find(position->item.object_key_hash);
            ceph_assert(state != heat_map.end());
            ceph_assert(state->second.pending_evaluation_count > 0);
            --state->second.pending_evaluation_count;
            --pending_deadline_count;
            if (position == next_deadline) {
                next_deadline = std::next(position);
            }
            future_access_threshold.apply_observation(
                HpFutureAccessObservation{
                    position->item.object_key_hash,
                    now_ns,
                    state->second.pending_evaluation_count},
                now_ns);
            make_idle_if_unprotected(
                position->item.object_key_hash,
                state->second);
        }
        ++evaluation_drop_count_value;
        pending_evaluations.erase(position);
    }

    ExpiryMaintenanceResult maintain_expiry(
            uint64_t now_ns,
            size_t max_evaluations =
                HP_EXPIRY_MAINTENANCE_BATCH_SIZE) {
        const auto schedule = expiry_schedule(now_ns);
        if (schedule.state != ExpiryScheduleState::due) {
            return ExpiryMaintenanceResult{
                {}, 0, false, false, schedule};
        }

        const auto threshold_before = future_access_threshold.status();
        const size_t pending_before = pending_deadline_count;
        expire_short_accesses(now_ns);
        auto evaluated = expire_due_evaluations(
            now_ns, max_evaluations);
        future_access_threshold.maintain(now_ns);
        const auto threshold_after = future_access_threshold.status();
        const bool threshold_status_changed =
            threshold_before.current_threshold !=
                threshold_after.current_threshold ||
            threshold_before.candidate_threshold !=
                threshold_after.candidate_threshold ||
            threshold_before.state != threshold_after.state ||
            threshold_before.positive_object_count !=
                threshold_after.positive_object_count ||
            threshold_before.zero_observation_count !=
                threshold_after.zero_observation_count ||
            threshold_before.upper_clamped_object_count !=
                threshold_after.upper_clamped_object_count ||
            threshold_before.occupied_bin_count !=
                threshold_after.occupied_bin_count;
        return ExpiryMaintenanceResult{
            std::move(evaluated),
            pending_before - pending_deadline_count,
            threshold_status_changed,
            true,
            expiry_schedule(now_ns)};
    }

    EvaluationQueueStatus status(uint64_t now_ns) const {
        (void)now_ns;
        ceph_assert(pending_evaluations.size() >= pending_deadline_count);
        ceph_assert(heat_map.size() >= lru_list.size());
        const auto threshold = future_access_threshold.status();
        return EvaluationQueueStatus{
            pending_deadline_count,
            pending_evaluations.size() - pending_deadline_count,
            evaluation_drop_count_value,
            heat_map.size(),
            lru_list.size(),
            heat_map.size() - lru_list.size(),
            heat_state_peak_count_value,
            lru_eviction_count_value,
            threshold.occupied_bin_count,
            threshold.current_threshold,
            threshold.candidate_threshold,
            static_cast<uint64_t>(threshold.state),
            threshold.positive_object_count,
            threshold.zero_observation_count,
            threshold.upper_clamped_object_count,
            threshold_holding_sample_count_value,
            sparse_threshold_sample_count_value};
    }

private:
    void enforce_lru_capacity() {
        while (lru_list.size() > lru_capacity) {
            const uint64_t victim = lru_list.front();
            lru_list.pop_front();
            auto victim_position = heat_map.find(victim);
            ceph_assert(victim_position != heat_map.end());
            ceph_assert(
                victim_position->second.pending_evaluation_count == 0);
            ceph_assert(
                victim_position->second.short_window_access_count == 0);
            heat_map.erase(victim_position);
            ++lru_eviction_count_value;
        }
    }

    void make_idle_if_unprotected(
            uint64_t key,
            ObjectHeatState& state) {
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
            PendingIterator position,
            uint64_t now_ns,
            std::vector<HpFutureAccessObservation>& observations) {
        PendingEvaluation& expired = *position;
        ceph_assert(!expired.label_complete);
        ceph_assert(pending_deadline_count > 0);

        auto state_position =
            heat_map.find(expired.item.object_key_hash);
        ceph_assert(state_position != heat_map.end());
        ObjectHeatState& state = state_position->second;
        ceph_assert(state.pending_evaluation_count > 0);
        ceph_assert(
            state.tracked_access_count >=
            expired.item.tracked_access_count_after_current_access);

        const uint64_t deadline_ns = saturating_add(
            expired.enqueue_time_ns, future_label_window_ns);
        expired.future_window_access_count =
            state.tracked_access_count -
            expired.item.tracked_access_count_after_current_access;
        expired.actual_label =
            expired.future_window_access_count >=
                expired.item.future_access_threshold_at_prediction
            ? 1
            : 0;
        expired.label_deadline_ns = deadline_ns;
        expired.label_completion_time_ns = now_ns;
        expired.label_complete = true;

        --state.pending_evaluation_count;
        --pending_deadline_count;
        observations.push_back(HpFutureAccessObservation{
            expired.item.object_key_hash,
            deadline_ns,
            state.pending_evaluation_count});
        make_idle_if_unprotected(
            expired.item.object_key_hash, state);

        if (!expired.prediction_complete) {
            return std::nullopt;
        }
        return finalize_evaluation(position);
    }

    EvaluatedSample finalize_evaluation(PendingIterator position) {
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
            completed.cold_start_fallback};
        pending_evaluations.erase(position);
        return evaluated;
    }

    PendingIterator enqueue_time_impl(
            PredictionSample item,
            bool prediction_complete,
            uint64_t now_ns) {
        auto state_position = heat_map.find(item.object_key_hash);
        ceph_assert(state_position != heat_map.end());
        if (pending_evaluations.size() >= pending_evaluation_capacity) {
            ++evaluation_drop_count_value;
            make_idle_if_unprotected(
                item.object_key_hash, state_position->second);
            return pending_evaluations.end();
        }

        const HpThresholdState threshold_state =
            future_access_threshold.status().state;
        if (threshold_state == HpThresholdState::sparse) {
            ++sparse_threshold_sample_count_value;
        } else if (threshold_state == HpThresholdState::holding) {
            ++threshold_holding_sample_count_value;
        }

        ++state_position->second.pending_evaluation_count;
        const bool needs_deadline_head =
            next_deadline == pending_evaluations.end();
        pending_evaluations.emplace_back(
            std::move(item), prediction_complete, now_ns);
        auto inserted = std::prev(pending_evaluations.end());
        if (needs_deadline_head) {
            next_deadline = inserted;
        }
        ++pending_deadline_count;
        future_access_threshold.apply_observation(
            HpFutureAccessObservation{
                inserted->item.object_key_hash,
                now_ns,
                state_position->second.pending_evaluation_count},
            now_ns);
        return inserted;
    }
};

#endif
