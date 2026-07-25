#ifndef CEPH_HEATPREDICTOR_HP_FUTURE_ACCESS_THRESHOLD_H
#define CEPH_HEATPREDICTOR_HP_FUTURE_ACCESS_THRESHOLD_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/debug.h"
#include "hp_config.h"

enum class HpThresholdState : uint64_t {
    sparse = 0,
    tracking = 1,
    holding = 2
};

struct HpFutureAccessObservation {
    uint64_t object_key_hash;
    uint64_t label_deadline_ns;
    uint64_t future_access_count;
};

struct HpFutureAccessThresholdStatus {
    uint64_t current_threshold;
    uint64_t candidate_threshold;
    HpThresholdState state;
    uint64_t positive_object_count;
    uint64_t zero_observation_count;
    uint64_t upper_clamped_object_count;
    uint64_t occupied_bin_count;
};

class HpFutureAccessThreshold {
private:
    static constexpr double score_min = HP_FUTURE_ACCESS_OTSU_SCORE_MIN;
    static constexpr double bin_width = HP_FUTURE_ACCESS_OTSU_BIN_WIDTH;
    static constexpr size_t bin_count = HP_FUTURE_ACCESS_OTSU_BIN_COUNT;

    struct ObjectVote {
        uint64_t label_deadline_ns;
        size_t bin;
        bool upper_clamped;
        std::list<uint64_t>::iterator order_position;
    };

    size_t object_capacity;
    size_t minimum_positive_objects;
    size_t update_interval;
    uint64_t recompute_max_interval_ns;
    uint64_t hold_interval_ns;

    std::array<uint64_t, bin_count> histogram{};
    uint64_t occupied_bin_count_value = 0;
    uint64_t upper_clamped_object_count_value = 0;
    uint64_t zero_observation_count_value = 0;
    size_t pending_object_changes = 0;
    std::list<uint64_t> object_order;
    std::unordered_map<uint64_t, ObjectVote> votes_by_object;

    HpThresholdState state_value = HpThresholdState::sparse;
    uint64_t current_threshold_value = 1;
    uint64_t candidate_threshold_value = 0;
    double published_score = score_min;
    bool published_score_initialized = false;
    bool recompute_time_initialized = false;
    uint64_t last_recompute_time_ns = 0;
    bool valid_candidate_time_initialized = false;
    uint64_t last_valid_candidate_time_ns = 0;

public:
    explicit HpFutureAccessThreshold(
            size_t object_capacity =
                HP_FUTURE_ACCESS_THRESHOLD_OBJECT_CAPACITY,
            size_t minimum_positive_objects =
                HP_FUTURE_ACCESS_OTSU_MIN_POSITIVE_OBJECTS,
            size_t update_interval =
                HP_FUTURE_ACCESS_OTSU_UPDATE_INTERVAL,
            uint64_t recompute_max_interval_ns =
                HP_FUTURE_ACCESS_OTSU_RECOMPUTE_MAX_INTERVAL_NS,
            uint64_t hold_interval_ns =
                HP_FUTURE_ACCESS_THRESHOLD_HOLD_NS) :
            object_capacity(object_capacity),
            minimum_positive_objects(minimum_positive_objects),
            update_interval(update_interval),
            recompute_max_interval_ns(recompute_max_interval_ns),
            hold_interval_ns(hold_interval_ns) {
        votes_by_object.reserve(std::min<size_t>(
            object_capacity, static_cast<size_t>(65536)));
    }

    static uint64_t maximum_representable_count() {
        const double score_max =
            score_min + bin_width * static_cast<double>(bin_count);
        return static_cast<uint64_t>(std::floor(std::exp2(score_max) - 1.0));
    }

    uint64_t current_threshold() const {
        return current_threshold_value;
    }

    HpFutureAccessThresholdStatus status() const {
        return HpFutureAccessThresholdStatus{
            current_threshold_value,
            candidate_threshold_value,
            state_value,
            votes_by_object.size(),
            zero_observation_count_value,
            upper_clamped_object_count_value,
            occupied_bin_count_value};
    }

    std::optional<uint64_t> maintenance_deadline_ns() const {
        std::optional<uint64_t> deadline;
        if (recompute_time_initialized && !votes_by_object.empty()) {
            deadline = saturating_add(
                last_recompute_time_ns, recompute_max_interval_ns);
        }
        if (state_value == HpThresholdState::holding &&
            valid_candidate_time_initialized) {
            const uint64_t hold_deadline = saturating_add(
                last_valid_candidate_time_ns, hold_interval_ns);
            if (!deadline.has_value() || hold_deadline < *deadline) {
                deadline = hold_deadline;
            }
        }
        return deadline;
    }

    void apply_observations(
            const std::vector<HpFutureAccessObservation>& observations,
            uint64_t now_ns) {
        if (observations.empty()) {
            maintain(now_ns);
            return;
        }

        std::unordered_map<uint64_t, HpFutureAccessObservation> latest;
        latest.reserve(observations.size());
        for (const auto& observation : observations) {
            auto [position, inserted] = latest.emplace(
                observation.object_key_hash, observation);
            if (!inserted &&
                observation.label_deadline_ns >=
                    position->second.label_deadline_ns) {
                position->second = observation;
            }
        }

        std::vector<HpFutureAccessObservation> ordered;
        ordered.reserve(latest.size());
        for (const auto& [key, observation] : latest) {
            (void)key;
            ordered.push_back(observation);
        }
        std::sort(
            ordered.begin(), ordered.end(),
            [](const auto& lhs, const auto& rhs) {
                if (lhs.label_deadline_ns != rhs.label_deadline_ns) {
                    return lhs.label_deadline_ns < rhs.label_deadline_ns;
                }
                return lhs.object_key_hash < rhs.object_key_hash;
            });

        const bool was_ready = ready_for_otsu();
        for (const auto& observation : ordered) {
            apply_observation_value(observation);
        }
        finish_observation_batch(was_ready, now_ns);
    }

    void apply_observation(
            const HpFutureAccessObservation& observation,
            uint64_t now_ns) {
        const bool was_ready = ready_for_otsu();
        apply_observation_value(observation);
        finish_observation_batch(was_ready, now_ns);
    }

private:
    void finish_observation_batch(bool was_ready, uint64_t now_ns) {
        const bool became_ready = !was_ready && ready_for_otsu();
        if (!recompute_time_initialized) {
            recompute_time_initialized = true;
            last_recompute_time_ns = now_ns;
        }
        const bool change_due =
            update_interval == 0 || pending_object_changes >= update_interval;
        const bool time_due =
            now_ns >= last_recompute_time_ns &&
            now_ns - last_recompute_time_ns >= recompute_max_interval_ns;
        if (became_ready || change_due || time_due) {
            recompute(now_ns);
        }
    }

public:
    void maintain(uint64_t now_ns) {
        const bool time_due = recompute_time_initialized &&
            now_ns >= last_recompute_time_ns &&
            now_ns - last_recompute_time_ns >= recompute_max_interval_ns;
        const bool hold_due =
            state_value == HpThresholdState::holding &&
            valid_candidate_time_initialized &&
            now_ns >= last_valid_candidate_time_ns &&
            now_ns - last_valid_candidate_time_ns >= hold_interval_ns;
        if (time_due || hold_due) {
            recompute(now_ns);
        }
    }

    void clear() {
        histogram.fill(0);
        occupied_bin_count_value = 0;
        upper_clamped_object_count_value = 0;
        zero_observation_count_value = 0;
        pending_object_changes = 0;
        object_order.clear();
        votes_by_object.clear();
        state_value = HpThresholdState::sparse;
        current_threshold_value = 1;
        candidate_threshold_value = 0;
        published_score = score_min;
        published_score_initialized = false;
        recompute_time_initialized = false;
        last_recompute_time_ns = 0;
        valid_candidate_time_initialized = false;
        last_valid_candidate_time_ns = 0;
    }

private:
    static uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
        return lhs > std::numeric_limits<uint64_t>::max() - rhs
            ? std::numeric_limits<uint64_t>::max()
            : lhs + rhs;
    }

    static double score_for_count(uint64_t count) {
        return std::log2(1.0 + static_cast<double>(count));
    }

    static uint64_t threshold_for_score(double score) {
        const double count = std::exp2(score) - 1.0;
        if (!std::isfinite(count) ||
            count >= static_cast<double>(
                std::numeric_limits<uint64_t>::max())) {
            return std::numeric_limits<uint64_t>::max();
        }
        return std::max<uint64_t>(
            1, static_cast<uint64_t>(std::ceil(count)));
    }

    static size_t bin_for_score(double score) {
        if (score <= score_min) {
            return 0;
        }
        const double relative = std::floor((score - score_min) / bin_width);
        if (relative >= static_cast<double>(bin_count)) {
            return bin_count - 1;
        }
        return static_cast<size_t>(relative);
    }

    static double bin_center(size_t bin) {
        return score_min +
            (static_cast<double>(bin) + 0.5) * bin_width;
    }

    bool ready_for_otsu() const {
        return votes_by_object.size() >= minimum_positive_objects &&
            occupied_bin_count_value >= 2;
    }

    void increment_bin(size_t bin) {
        ceph_assert(bin < histogram.size());
        if (histogram[bin] == 0) {
            ++occupied_bin_count_value;
        }
        ceph_assert(histogram[bin] < std::numeric_limits<uint64_t>::max());
        ++histogram[bin];
    }

    void decrement_bin(size_t bin) {
        ceph_assert(bin < histogram.size());
        ceph_assert(histogram[bin] > 0);
        --histogram[bin];
        if (histogram[bin] == 0) {
            ceph_assert(occupied_bin_count_value > 0);
            --occupied_bin_count_value;
        }
    }

    void erase_vote(
            std::unordered_map<uint64_t, ObjectVote>::iterator position) {
        decrement_bin(position->second.bin);
        if (position->second.upper_clamped) {
            ceph_assert(upper_clamped_object_count_value > 0);
            --upper_clamped_object_count_value;
        }
        object_order.erase(position->second.order_position);
        votes_by_object.erase(position);
    }

    void apply_observation_value(
            const HpFutureAccessObservation& observation) {
        auto old = votes_by_object.find(observation.object_key_hash);
        if (old != votes_by_object.end() &&
            observation.label_deadline_ns < old->second.label_deadline_ns) {
            return;
        }

        if (observation.future_access_count == 0) {
            ++zero_observation_count_value;
            if (old != votes_by_object.end()) {
                erase_vote(old);
                ++pending_object_changes;
            }
            return;
        }

        const double score = score_for_count(observation.future_access_count);
        const double score_max =
            score_min + bin_width * static_cast<double>(bin_count);
        const bool upper_clamped = score > score_max;
        const size_t bin = bin_for_score(std::min(score, score_max));

        if (old == votes_by_object.end()) {
            object_order.push_back(observation.object_key_hash);
            auto [inserted, ok] = votes_by_object.emplace(
                observation.object_key_hash,
                ObjectVote{
                    observation.label_deadline_ns,
                    bin,
                    upper_clamped,
                    std::prev(object_order.end())});
            ceph_assert(ok);
            (void)inserted;
            increment_bin(bin);
            if (upper_clamped) {
                ++upper_clamped_object_count_value;
            }
        } else {
            decrement_bin(old->second.bin);
            if (old->second.upper_clamped) {
                ceph_assert(upper_clamped_object_count_value > 0);
                --upper_clamped_object_count_value;
            }
            object_order.splice(
                object_order.end(),
                object_order,
                old->second.order_position);
            old->second.label_deadline_ns = observation.label_deadline_ns;
            old->second.bin = bin;
            old->second.upper_clamped = upper_clamped;
            increment_bin(bin);
            if (upper_clamped) {
                ++upper_clamped_object_count_value;
            }
        }
        ++pending_object_changes;

        while (votes_by_object.size() > object_capacity) {
            ceph_assert(!object_order.empty());
            auto victim = votes_by_object.find(object_order.front());
            ceph_assert(victim != votes_by_object.end());
            erase_vote(victim);
            ++pending_object_changes;
        }
    }

    std::optional<double> otsu_threshold_score() const {
        if (!ready_for_otsu()) {
            return std::nullopt;
        }

        uint64_t total_count = 0;
        double total_sum = 0.0;
        for (size_t bin = 0; bin < histogram.size(); ++bin) {
            total_count += histogram[bin];
            total_sum +=
                bin_center(bin) * static_cast<double>(histogram[bin]);
        }
        ceph_assert(total_count == votes_by_object.size());

        uint64_t lhs_count = 0;
        double lhs_sum = 0.0;
        double best_variance = -1.0;
        size_t last_lhs_bin = 0;
        size_t next_rhs_bin = 0;
        for (size_t bin = 0; bin + 1 < histogram.size(); ++bin) {
            if (histogram[bin] == 0) {
                continue;
            }
            lhs_count += histogram[bin];
            lhs_sum +=
                bin_center(bin) * static_cast<double>(histogram[bin]);
            const uint64_t rhs_count = total_count - lhs_count;
            if (lhs_count == 0 || rhs_count == 0) {
                continue;
            }
            size_t rhs_bin = bin + 1;
            while (rhs_bin < histogram.size() &&
                   histogram[rhs_bin] == 0) {
                ++rhs_bin;
            }
            if (rhs_bin == histogram.size()) {
                break;
            }
            const double lhs_mean =
                lhs_sum / static_cast<double>(lhs_count);
            const double rhs_mean =
                (total_sum - lhs_sum) / static_cast<double>(rhs_count);
            const double difference = lhs_mean - rhs_mean;
            const double variance =
                static_cast<double>(lhs_count) *
                static_cast<double>(rhs_count) *
                difference * difference;
            if (variance > best_variance) {
                best_variance = variance;
                last_lhs_bin = bin;
                next_rhs_bin = rhs_bin;
            }
        }
        if (best_variance <= 0.0) {
            return std::nullopt;
        }
        return (bin_center(last_lhs_bin) + bin_center(next_rhs_bin)) / 2.0;
    }

    static double ema_gain_for_elapsed(uint64_t elapsed_ns) {
        if (elapsed_ns == 0) {
            return 0.0;
        }
        const double intervals = static_cast<double>(elapsed_ns) /
            static_cast<double>(
                HP_FUTURE_ACCESS_OTSU_RECOMPUTE_MAX_INTERVAL_NS);
        return 1.0 - std::pow(
            1.0 - HP_FUTURE_ACCESS_THRESHOLD_EMA_ALPHA, intervals);
    }

    void recompute(uint64_t now_ns) {
        const auto candidate_score = otsu_threshold_score();
        pending_object_changes = 0;
        recompute_time_initialized = true;
        const uint64_t elapsed_ns =
            now_ns >= last_recompute_time_ns
                ? now_ns - last_recompute_time_ns
                : 0;
        last_recompute_time_ns = now_ns;

        if (!candidate_score.has_value()) {
            candidate_threshold_value = 0;
            if (valid_candidate_time_initialized &&
                now_ns >= last_valid_candidate_time_ns &&
                now_ns - last_valid_candidate_time_ns < hold_interval_ns) {
                state_value = HpThresholdState::holding;
                return;
            }
            state_value = HpThresholdState::sparse;
            current_threshold_value = 1;
            published_score = score_min;
            published_score_initialized = false;
            valid_candidate_time_initialized = false;
            last_valid_candidate_time_ns = 0;
            return;
        }

        candidate_threshold_value =
            threshold_for_score(*candidate_score);
        if (!published_score_initialized) {
            published_score = *candidate_score;
            published_score_initialized = true;
        } else {
            const double gain = ema_gain_for_elapsed(elapsed_ns);
            published_score +=
                gain * (*candidate_score - published_score);
        }
        current_threshold_value = threshold_for_score(published_score);
        state_value = HpThresholdState::tracking;
        valid_candidate_time_initialized = true;
        last_valid_candidate_time_ns = now_ns;
    }
};

#endif
