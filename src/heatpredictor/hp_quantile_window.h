#ifndef CEPH_HEATPREDICTOR_HP_QUANTILE_WINDOW_H
#define CEPH_HEATPREDICTOR_HP_QUANTILE_WINDOW_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

#include "hp_config.h"
#include "hp_types.h"

class HpQuantileWindow {
public:
    typedef __gnu_pbds::tree<
        std::pair<double, uint64_t>,
        __gnu_pbds::null_type,
        std::less<std::pair<double, uint64_t>>,
        __gnu_pbds::rb_tree_tag,
        __gnu_pbds::tree_order_statistics_node_update
    > pbds_set;

    explicit HpQuantileWindow(
            size_t capacity = HP_REPORT_SAMPLE_WINDOW_CAPACITY) :
            capacity(capacity) {}

    void insert(double value) {
        if (capacity == 0) {
            return;
        }

        auto entry = std::make_pair(value, ++counter);
        values.insert(entry);
        order.push_back(entry);
        if (values.size() > capacity) {
            values.erase(order.front());
            order.pop_front();
        }
    }

    void clear() {
        values.clear();
        order.clear();
        counter = 0;
    }

    HpDistributionSummary summary() const {
        if (values.empty()) {
            return {};
        }

        return HpDistributionSummary{
            static_cast<uint64_t>(values.size()),
            values.find_by_order(values.size() - 1)->first,
            quantile(0.50),
            quantile(0.90),
            quantile(0.95),
            quantile(0.99)
        };
    }

private:
    size_t capacity;
    pbds_set values;
    std::deque<std::pair<double, uint64_t>> order;
    uint64_t counter = 0;

    double quantile(double q) const {
        if (values.empty()) {
            return 0.0;
        }
        size_t idx = static_cast<size_t>(
            std::ceil(q * static_cast<double>(values.size())));
        idx = idx == 0 ? 0 : idx - 1;
        if (idx >= values.size()) {
            idx = values.size() - 1;
        }
        return values.find_by_order(idx)->first;
    }
};

#endif
