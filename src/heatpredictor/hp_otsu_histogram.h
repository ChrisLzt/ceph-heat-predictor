#ifndef CEPH_HEATPREDICTOR_HP_OTSU_HISTOGRAM_H
#define CEPH_HEATPREDICTOR_HP_OTSU_HISTOGRAM_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>

#include "common/debug.h"
#include "hp_config.h"

class HpOtsuHistogram {
public:
    static double score_for_heat(double heat) {
        ceph_assert(HP_OTSU_HEAT_MIN > 0.0);
        ceph_assert(HP_OTSU_HEAT_MAX >= HP_OTSU_HEAT_MIN);
        return std::log(std::clamp(
            heat, HP_OTSU_HEAT_MIN, HP_OTSU_HEAT_MAX));
    }

    static double heat_for_score(double score) {
        return std::clamp(
            std::exp(score), HP_OTSU_HEAT_MIN, HP_OTSU_HEAT_MAX);
    }

    void insert(double score) {
        const int64_t bin = bin_for(score);
        add_count_to_bin(bin, 1);
    }

    void erase(double score) {
        const int64_t bin = bin_for(score);
        auto it = bins.find(bin);
        ceph_assert(it != bins.end());
        ceph_assert(it->second.count > 0);

        BinStats& stats = it->second;
        stats.count--;
        total_count--;
        total_sum -= bin_center(bin);
        total_square_sum -= bin_center(bin) * bin_center(bin);

        if (stats.count == 0) {
            bins.erase(it);
        }
    }

    void clamp_lower_bound(double min_score) {
        const int64_t min_bin = bin_for(min_score);
        if (lower_bound_initialized && min_bin <= lower_bound_bin) {
            return;
        }
        lower_bound_initialized = true;
        lower_bound_bin = min_bin;

        while (!bins.empty()) {
            auto it = bins.begin();
            if (it->first >= min_bin) {
                break;
            }

            uint64_t count = it->second.count;
            total_count -= count;
            total_sum -= bin_center(it->first) * static_cast<double>(count);
            total_square_sum -=
                bin_center(it->first) * bin_center(it->first) *
                static_cast<double>(count);
            bins.erase(it);
            add_count_to_bin(min_bin, count);
        }
    }

    size_t size() const {
        return total_count;
    }

    size_t bin_count() const {
        return bins.size();
    }

    bool empty() const {
        return total_count == 0;
    }

    std::optional<double> otsu_threshold_score(double *separation) const {
        if (total_count < HP_OTSU_MIN_OBJECTS || bins.size() < 2) {
            return std::nullopt;
        }

        const double count = static_cast<double>(total_count);
        const double total_mean = total_sum / count;
        const double total_variance =
            total_square_sum / count - total_mean * total_mean;
        if (total_variance <= 0.0) {
            return std::nullopt;
        }

        uint64_t lhs_count_u = 0;
        double lhs_sum = 0.0;
        double best_between_variance = -1.0;
        double best_threshold_score = 0.0;

        for (auto it = bins.begin(); it != bins.end(); ++it) {
            auto next = std::next(it);
            if (next == bins.end()) {
                break;
            }

            lhs_count_u += it->second.count;
            lhs_sum +=
                bin_center(it->first) *
                static_cast<double>(it->second.count);

            const double lhs_count = static_cast<double>(lhs_count_u);
            const double rhs_count = count - lhs_count;
            if (lhs_count <= 0.0 || rhs_count <= 0.0) {
                continue;
            }

            const double lhs_mean = lhs_sum / lhs_count;
            const double rhs_mean = (total_sum - lhs_sum) / rhs_count;
            const double mean_diff = lhs_mean - rhs_mean;
            const double between_variance =
                (lhs_count * rhs_count * mean_diff * mean_diff) /
                (count * count);
            if (between_variance > best_between_variance) {
                best_between_variance = between_variance;
                best_threshold_score =
                    (bin_center(it->first) + bin_center(next->first)) / 2.0;
            }
        }

        if (best_between_variance < 0.0) {
            return std::nullopt;
        }

        *separation = best_between_variance / total_variance;
        if (*separation < HP_OTSU_MIN_SEPARATION) {
            return std::nullopt;
        }
        return best_threshold_score;
    }

private:
    struct BinStats {
        uint64_t count = 0;
    };

    int64_t bin_for(double score) const {
        ceph_assert(HP_OTSU_LOG_HEAT_BIN_WIDTH > 0.0);
        const double scaled =
            std::floor(score / HP_OTSU_LOG_HEAT_BIN_WIDTH);
        if (scaled <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
            return std::numeric_limits<int64_t>::min();
        }
        if (scaled >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
            return std::numeric_limits<int64_t>::max();
        }
        return static_cast<int64_t>(scaled);
    }

    double bin_center(int64_t bin) const {
        return (static_cast<double>(bin) + 0.5) *
            HP_OTSU_LOG_HEAT_BIN_WIDTH;
    }

    void add_count_to_bin(int64_t bin, uint64_t count) {
        if (count == 0) {
            return;
        }
        BinStats& stats = bins[bin];
        stats.count += count;
        const double center = bin_center(bin);
        total_count += count;
        total_sum += center * static_cast<double>(count);
        total_square_sum += center * center * static_cast<double>(count);
    }

    std::map<int64_t, BinStats> bins;
    uint64_t total_count = 0;
    double total_sum = 0.0;
    double total_square_sum = 0.0;
    bool lower_bound_initialized = false;
    int64_t lower_bound_bin = std::numeric_limits<int64_t>::min();
};

#endif
