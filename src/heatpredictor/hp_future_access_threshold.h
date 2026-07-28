#ifndef CEPH_HEATPREDICTOR_HP_FUTURE_ACCESS_THRESHOLD_H
#define CEPH_HEATPREDICTOR_HP_FUTURE_ACCESS_THRESHOLD_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "common/debug.h"
#include "hp_config.h"

enum class HpThresholdState : uint64_t {
    sparse = 0,
    tracking = 1
};

struct HpFutureAccessThresholdStatus {
    uint64_t current_threshold;
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

    size_t minimum_positive_objects;
    size_t update_interval;
    uint64_t recompute_max_interval_ns;

    std::array<uint64_t, bin_count> histogram{};
    uint64_t positive_object_count_value = 0;
    uint64_t occupied_bin_count_value = 0;
    uint64_t upper_clamped_object_count_value = 0;
    uint64_t zero_observation_count_value = 0;
    size_t pending_object_changes = 0;
    bool histogram_dirty = false;

    HpThresholdState state_value = HpThresholdState::sparse;
    uint64_t current_threshold_value = 1;
    bool recompute_time_initialized = false;
    uint64_t last_recompute_time_ns = 0;

public:
    explicit HpFutureAccessThreshold(
            size_t minimum_positive_objects =
                HP_FUTURE_ACCESS_OTSU_MIN_POSITIVE_OBJECTS,
            size_t update_interval =
                HP_FUTURE_ACCESS_OTSU_UPDATE_INTERVAL,
            uint64_t recompute_max_interval_ns =
                HP_FUTURE_ACCESS_OTSU_RECOMPUTE_MAX_INTERVAL_NS) :
            minimum_positive_objects(minimum_positive_objects),
            update_interval(update_interval),
            recompute_max_interval_ns(recompute_max_interval_ns) {}

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
            state_value,
            positive_object_count_value,
            zero_observation_count_value,
            upper_clamped_object_count_value,
            occupied_bin_count_value};
    }

    std::optional<uint64_t> maintenance_deadline_ns() const {
        if (!histogram_dirty ||
            !recompute_time_initialized ||
            positive_object_count_value == 0) {
            return std::nullopt;
        }
        return saturating_add(
            last_recompute_time_ns, recompute_max_interval_ns);
    }

    void update_object_count(
            uint64_t old_count,
            uint64_t new_count,
            uint64_t now_ns) {
        update_object_count_deferred(old_count, new_count);
        maintain(now_ns);
    }

    void update_object_count_deferred(
            uint64_t old_count,
            uint64_t new_count) {
        if (old_count == new_count) {
            return;
        }

        if (old_count > 0) {
            const BinValue old_bin = bin_for_count(old_count);
            decrement_bin(old_bin.bin);
            ceph_assert(positive_object_count_value > 0);
            --positive_object_count_value;
            if (old_bin.upper_clamped) {
                ceph_assert(upper_clamped_object_count_value > 0);
                --upper_clamped_object_count_value;
            }
        }
        if (new_count > 0) {
            const BinValue new_bin = bin_for_count(new_count);
            increment_bin(new_bin.bin);
            ++positive_object_count_value;
            if (new_bin.upper_clamped) {
                ++upper_clamped_object_count_value;
            }
        } else {
            ++zero_observation_count_value;
        }

        ++pending_object_changes;
        histogram_dirty = true;
    }

    void maintain(uint64_t now_ns) {
        if (!histogram_dirty) {
            return;
        }
        if (!recompute_time_initialized) {
            recompute_time_initialized = true;
            last_recompute_time_ns = now_ns;
        }
        const bool change_due =
            update_interval == 0 || pending_object_changes >= update_interval;
        const bool readiness_changed =
            ready_for_otsu() !=
                (state_value == HpThresholdState::tracking);
        const bool time_due =
            now_ns >= last_recompute_time_ns &&
            now_ns - last_recompute_time_ns >= recompute_max_interval_ns;
        if (readiness_changed || change_due || time_due) {
            recompute(now_ns);
        }
    }

    void recompute_now(uint64_t now_ns) {
        if (histogram_dirty) {
            recompute(now_ns);
        }
    }

    void clear() {
        histogram.fill(0);
        positive_object_count_value = 0;
        occupied_bin_count_value = 0;
        upper_clamped_object_count_value = 0;
        zero_observation_count_value = 0;
        pending_object_changes = 0;
        histogram_dirty = false;
        state_value = HpThresholdState::sparse;
        current_threshold_value = 1;
        recompute_time_initialized = false;
        last_recompute_time_ns = 0;
    }

private:
    struct BinValue {
        size_t bin;
        bool upper_clamped;
    };

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

    static BinValue bin_for_count(uint64_t count) {
        const double score = score_for_count(count);
        const double score_max =
            score_min + bin_width * static_cast<double>(bin_count);
        return BinValue{
            bin_for_score(std::min(score, score_max)),
            score > score_max};
    }

    static double bin_center(size_t bin) {
        return score_min +
            (static_cast<double>(bin) + 0.5) * bin_width;
    }

    bool ready_for_otsu() const {
        return positive_object_count_value >= minimum_positive_objects &&
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
        ceph_assert(total_count == positive_object_count_value);

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
            if (rhs_count == 0) {
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

    void recompute(uint64_t now_ns) {
        const auto candidate_score = otsu_threshold_score();
        pending_object_changes = 0;
        histogram_dirty = false;
        recompute_time_initialized = true;
        last_recompute_time_ns = now_ns;

        if (!candidate_score.has_value()) {
            state_value = HpThresholdState::sparse;
            current_threshold_value = 1;
            return;
        }

        current_threshold_value =
            threshold_for_score(*candidate_score);
        state_value = HpThresholdState::tracking;
    }
};

#endif
