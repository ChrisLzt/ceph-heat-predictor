#ifndef CEPH_HEATPREDICTOR_HP_INTEGER_QUANTILE_WINDOW_H
#define CEPH_HEATPREDICTOR_HP_INTEGER_QUANTILE_WINDOW_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hp_config.h"
#include "hp_types.h"

class HpIntegerQuantileWindow {
public:
    explicit HpIntegerQuantileWindow(
            size_t capacity = HP_REPORT_STATS_WINDOW_CAPACITY,
            uint64_t max_value = HP_EVALUATION_WINDOW) :
            capacity(capacity), max_value(max_value) {
        if (max_value > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument(
                "integer quantile domain exceeds uint32_t");
        }
        fenwick.resize(static_cast<size_t>(max_value) + 2, 0);
    }

    void insert(uint64_t value) {
        if (value > max_value) {
            throw std::invalid_argument(
                "integer quantile value exceeds configured domain");
        }
        if (capacity == 0) {
            return;
        }

        add(static_cast<uint32_t>(value));
        order.push_back(static_cast<uint32_t>(value));
        if (order.size() > capacity) {
            remove(order.front());
            order.pop_front();
        }
    }

    void clear() {
        std::fill(fenwick.begin(), fenwick.end(), 0);
        order.clear();
    }

    HpDistributionSummary summary() const {
        if (order.empty()) {
            return {};
        }

        return HpDistributionSummary{
            static_cast<uint64_t>(order.size()),
            select_rank(static_cast<uint64_t>(order.size())),
            quantile(0.50),
            quantile(0.90),
            quantile(0.95),
            quantile(0.99)
        };
    }

private:
    size_t capacity;
    uint64_t max_value;
    std::vector<uint64_t> fenwick;
    std::deque<uint32_t> order;

    void add(uint32_t value) {
        for (size_t i = static_cast<size_t>(value) + 1;
             i < fenwick.size(); i += i & (~i + 1)) {
            fenwick[i]++;
        }
    }

    void remove(uint32_t value) {
        for (size_t i = static_cast<size_t>(value) + 1;
             i < fenwick.size(); i += i & (~i + 1)) {
            if (fenwick[i] == 0) {
                throw std::logic_error("integer quantile count underflow");
            }
            fenwick[i]--;
        }
    }

    double quantile(double q) const {
        uint64_t rank = static_cast<uint64_t>(
            std::ceil(q * static_cast<double>(order.size())));
        return select_rank(std::max<uint64_t>(rank, 1));
    }

    double select_rank(uint64_t rank) const {
        size_t index = 0;
        uint64_t prefix = 0;
        size_t step = 1;
        while ((step << 1) < fenwick.size()) {
            step <<= 1;
        }

        for (; step > 0; step >>= 1) {
            size_t next = index + step;
            if (next < fenwick.size() &&
                prefix + fenwick[next] < rank) {
                index = next;
                prefix += fenwick[next];
            }
        }
        return static_cast<double>(index);
    }
};

#endif
