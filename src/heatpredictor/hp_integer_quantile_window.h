#ifndef CEPH_HEATPREDICTOR_HP_INTEGER_QUANTILE_WINDOW_H
#define CEPH_HEATPREDICTOR_HP_INTEGER_QUANTILE_WINDOW_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>

#include "hp_config.h"
#include "hp_types.h"

class HpIntegerQuantileWindow {
private:
    static constexpr double bin_width =
        HP_REPORT_LOG_HISTOGRAM_BIN_WIDTH;
    static constexpr size_t bin_count =
        HP_REPORT_LOG_HISTOGRAM_BIN_COUNT;
    using BinIndex = uint16_t;

public:
    explicit HpIntegerQuantileWindow(
            size_t capacity = HP_REPORT_SAMPLE_WINDOW_CAPACITY) :
            capacity(capacity) {}

    void insert(uint64_t value) {
        if (capacity == 0) {
            return;
        }

        const BinIndex bin = bin_for_value(value);
        ceph_assert(bins[bin] < std::numeric_limits<uint64_t>::max());
        ++bins[bin];
        order.push_back(bin);
        if (order.size() > capacity) {
            ceph_assert(bins[order.front()] > 0);
            --bins[order.front()];
            order.pop_front();
        }
    }

    void clear() {
        bins.fill(0);
        order.clear();
    }

    HpDistributionSummary summary() const {
        if (order.empty()) {
            return {};
        }

        return HpDistributionSummary{
            static_cast<uint64_t>(order.size()),
            quantile(1.0),
            quantile(0.50),
            quantile(0.90),
            quantile(0.95),
            quantile(0.99)
        };
    }

private:
    size_t capacity;
    std::array<uint64_t, bin_count> bins{};
    std::deque<BinIndex> order;

    static BinIndex bin_for_value(uint64_t value) {
        const double score =
            std::log2(1.0 + static_cast<double>(value));
        const double relative = std::floor(score / bin_width);
        if (!std::isfinite(relative) ||
            relative >= static_cast<double>(bin_count)) {
            return static_cast<BinIndex>(bin_count - 1);
        }
        return static_cast<BinIndex>(
            std::max(0.0, relative));
    }

    static double value_for_bin(size_t bin) {
        if (bin == 0) {
            return 0.0;
        }
        const double center =
            (static_cast<double>(bin) + 0.5) * bin_width;
        return std::round(std::exp2(center) - 1.0);
    }

    double quantile(double q) const {
        const uint64_t rank = std::max<uint64_t>(
            1,
            static_cast<uint64_t>(
                std::ceil(q * static_cast<double>(order.size()))));
        uint64_t cumulative = 0;
        for (size_t bin = 0; bin < bins.size(); ++bin) {
            cumulative += bins[bin];
            if (cumulative >= rank) {
                return value_for_bin(bin);
            }
        }
        return value_for_bin(bins.size() - 1);
    }
};

#endif
