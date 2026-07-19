#ifndef CEPH_HEATPREDICTOR_HP_SCORE_OTSU_HISTOGRAM_H
#define CEPH_HEATPREDICTOR_HP_SCORE_OTSU_HISTOGRAM_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "common/debug.h"
#include "hp_config.h"

struct HpOtsuResult {
    double threshold_score;
};

class HpScoreOtsuHistogram {
public:
    using AbsoluteBin = int64_t;

    static constexpr size_t bin_capacity() {
        return HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT;
    }

    static double score_for_heat_at(
            double heat,
            uint64_t timestamp_ns,
            double decay_log_factor_per_ns) {
        const double bounded_heat = std::clamp(
            heat, HP_OTSU_TOTAL_HEAT_MIN, HP_OTSU_TOTAL_HEAT_MAX);
        return std::log(bounded_heat) - decay_log_factor_per_ns *
            static_cast<double>(timestamp_ns);
    }

    static double heat_for_score_at(
            double score,
            uint64_t timestamp_ns,
            double decay_log_factor_per_ns) {
        const double log_heat = score + decay_log_factor_per_ns *
            static_cast<double>(timestamp_ns);
        if (log_heat <= std::log(HP_OTSU_TOTAL_HEAT_MIN)) {
            return HP_OTSU_TOTAL_HEAT_MIN;
        }
        if (log_heat >= std::log(HP_OTSU_TOTAL_HEAT_MAX)) {
            return HP_OTSU_TOTAL_HEAT_MAX;
        }
        return std::exp(log_heat);
    }

    void advance_lower_bound(double min_score) {
        const AbsoluteBin next_origin = absolute_bin_for(min_score);
        if (!origin_initialized) {
            origin_initialized = true;
            origin_bin = next_origin;
            return;
        }
        if (next_origin <= origin_bin) {
            return;
        }

        const uint64_t distance = static_cast<uint64_t>(
            next_origin - origin_bin);
        if (distance >= bins.size()) {
            bins.fill(0);
            if (total_count != 0) {
                bins[0] = total_count;
                occupied_bin_count = 1;
            } else {
                occupied_bin_count = 0;
            }
            origin_bin = next_origin;
            return;
        }

        BinArray shifted{};
        uint64_t lower_count = 0;
        for (size_t index = 0; index <= distance; ++index) {
            lower_count += bins[index];
        }
        shifted[0] = lower_count;
        for (size_t index = 1; index + distance < bins.size(); ++index) {
            shifted[index] = bins[index + distance];
        }
        bins = shifted;
        origin_bin = next_origin;
        refresh_occupied_bin_count();
    }

    AbsoluteBin insert(double score) {
        ceph_assert(origin_initialized);
        const AbsoluteBin stored_bin = std::clamp(
            absolute_bin_for(score),
            origin_bin,
            maximum_absolute_bin());
        const size_t relative_bin = relative_bin_for(stored_bin);
        if (bins[relative_bin] == 0) {
            ++occupied_bin_count;
        }
        ceph_assert(bins[relative_bin] < std::numeric_limits<uint64_t>::max());
        ++bins[relative_bin];
        ++total_count;
        return stored_bin;
    }

    void erase(AbsoluteBin stored_bin) {
        ceph_assert(origin_initialized);
        const size_t relative_bin = relative_bin_for(
            std::max(stored_bin, origin_bin));
        ceph_assert(bins[relative_bin] > 0);
        ceph_assert(total_count > 0);
        --bins[relative_bin];
        --total_count;
        if (bins[relative_bin] == 0) {
            ceph_assert(occupied_bin_count > 0);
            --occupied_bin_count;
        }
    }

    void clear() {
        bins.fill(0);
        total_count = 0;
        occupied_bin_count = 0;
        origin_initialized = false;
        origin_bin = 0;
    }

    size_t size() const {
        return total_count;
    }

    size_t bin_count() const {
        return occupied_bin_count;
    }

    bool empty() const {
        return total_count == 0;
    }

    std::optional<HpOtsuResult> otsu_result() const {
        if (total_count < HP_OTSU_MIN_VOTES || occupied_bin_count < 2) {
            return std::nullopt;
        }

        const size_t first_bin = next_occupied_bin(0);
        ceph_assert(first_bin < bins.size());
        const double origin_score = bin_center(first_bin);
        double shifted_total_sum = 0.0;
        for (size_t bin = first_bin; bin < bins.size(); ++bin) {
            if (bins[bin] == 0) {
                continue;
            }
            const double shifted_center = bin_center(bin) - origin_score;
            shifted_total_sum += shifted_center *
                static_cast<double>(bins[bin]);
        }

        double best_between_variance = -1.0;
        double best_threshold_score = 0.0;
        for_each_partition(
            shifted_total_sum,
            [&](size_t lhs_bin,
                size_t rhs_bin,
                uint64_t,
                double between_variance) {
                if (between_variance > best_between_variance) {
                    best_between_variance = between_variance;
                    best_threshold_score =
                        (bin_center(lhs_bin) + bin_center(rhs_bin)) / 2.0;
                }
            });
        if (best_between_variance < 0.0) {
            return std::nullopt;
        }

        return HpOtsuResult{best_threshold_score};
    }

private:
    using BinArray = std::array<uint64_t, HP_SCORE_OTSU_HISTOGRAM_BIN_COUNT>;

    static AbsoluteBin absolute_bin_for(double score) {
        const double scaled = std::floor(
            score / HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH);
        if (scaled <= static_cast<double>(
                std::numeric_limits<AbsoluteBin>::min())) {
            return std::numeric_limits<AbsoluteBin>::min();
        }
        if (scaled >= static_cast<double>(
                std::numeric_limits<AbsoluteBin>::max())) {
            return std::numeric_limits<AbsoluteBin>::max();
        }
        return static_cast<AbsoluteBin>(scaled);
    }

    AbsoluteBin maximum_absolute_bin() const {
        ceph_assert(origin_initialized);
        return origin_bin + static_cast<AbsoluteBin>(bins.size() - 1);
    }

    size_t relative_bin_for(AbsoluteBin absolute_bin) const {
        ceph_assert(origin_initialized);
        ceph_assert(absolute_bin >= origin_bin);
        const uint64_t relative = static_cast<uint64_t>(
            absolute_bin - origin_bin);
        return static_cast<size_t>(std::min<uint64_t>(
            relative, bins.size() - 1));
    }

    double bin_center(size_t relative_bin) const {
        return (static_cast<double>(origin_bin) +
                static_cast<double>(relative_bin) + 0.5) *
            HP_SCORE_OTSU_LOG_HEAT_BIN_WIDTH;
    }

    size_t next_occupied_bin(size_t first) const {
        while (first < bins.size() && bins[first] == 0) {
            ++first;
        }
        return first;
    }

    template <typename Fn>
    void for_each_partition(double shifted_total_sum, Fn&& fn) const {
        uint64_t lhs_count_u = 0;
        double lhs_sum = 0.0;
        const double count = static_cast<double>(total_count);
        const size_t first_bin = next_occupied_bin(0);
        const double score_origin = bin_center(first_bin);
        size_t lhs_bin = first_bin;
        while (lhs_bin < bins.size()) {
            const size_t rhs_bin = next_occupied_bin(lhs_bin + 1);
            if (rhs_bin == bins.size()) {
                break;
            }

            lhs_count_u += bins[lhs_bin];
            lhs_sum += (bin_center(lhs_bin) - score_origin) *
                static_cast<double>(bins[lhs_bin]);
            const double lhs_count = static_cast<double>(lhs_count_u);
            const double rhs_count = count - lhs_count;
            ceph_assert(lhs_count > 0.0 && rhs_count > 0.0);
            const double lhs_mean = lhs_sum / lhs_count;
            const double rhs_mean =
                (shifted_total_sum - lhs_sum) / rhs_count;
            const double mean_diff = lhs_mean - rhs_mean;
            const double between_variance =
                (lhs_count * rhs_count * mean_diff * mean_diff) /
                (count * count);
            fn(lhs_bin, rhs_bin, lhs_count_u, between_variance);
            lhs_bin = rhs_bin;
        }
    }

    void refresh_occupied_bin_count() {
        occupied_bin_count = static_cast<size_t>(std::count_if(
            bins.begin(), bins.end(), [](uint64_t count) {
                return count != 0;
            }));
    }

    BinArray bins{};
    uint64_t total_count = 0;
    size_t occupied_bin_count = 0;
    bool origin_initialized = false;
    AbsoluteBin origin_bin = 0;
};

#endif
