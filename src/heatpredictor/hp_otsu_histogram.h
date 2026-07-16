#ifndef CEPH_HEATPREDICTOR_HP_OTSU_HISTOGRAM_H
#define CEPH_HEATPREDICTOR_HP_OTSU_HISTOGRAM_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>

#include "common/debug.h"
#include "hp_config.h"

struct HpOtsuResult {
    double threshold_score;
    double separation;
    uint64_t ambiguous_vote_count;
    uint64_t vote_count;
};

class HpOtsuHistogram {
public:
    using AggregateBins =
        std::array<uint64_t, HP_OTSU_HISTOGRAM_BIN_COUNT>;
    using SlotCount = uint32_t;

    HpOtsuHistogram() :
            aggregate_bins(std::make_unique<AggregateBins>()),
            history_bins(std::make_unique<SlotCount[]>(
                HP_OTSU_HISTORY_SLOT_COUNT * HP_OTSU_HISTOGRAM_BIN_COUNT)) {
        slot_epochs.fill(invalid_epoch);
    }

    HpOtsuHistogram(const HpOtsuHistogram&) = delete;
    HpOtsuHistogram& operator=(const HpOtsuHistogram&) = delete;

    static constexpr size_t bin_capacity() {
        return HP_OTSU_HISTOGRAM_BIN_COUNT;
    }

    static double score_for_heat(double heat) {
        if (!std::isfinite(heat)) {
            return heat > 0.0
                ? max_representable_score()
                : 0.0;
        }
        return std::log1p(std::max(0.0, heat));
    }

    static double heat_for_score(double score) {
        if (score <= 0.0) {
            return 0.0;
        }
        if (!std::isfinite(score) || score >= max_representable_score()) {
            return std::expm1(max_representable_score());
        }
        return std::expm1(score);
    }

    bool observe(
            uint64_t object_key,
            double added_heat,
            uint64_t sample_time_ns,
            uint64_t now_ns) {
        if (sample_time_ns > now_ns) {
            return false;
        }

        advance_to(now_ns);
        const uint64_t sample_epoch = sample_time_ns / HP_OTSU_HISTORY_SLOT_NS;
        const uint64_t now_epoch = now_ns / HP_OTSU_HISTORY_SLOT_NS;
        if (now_epoch - sample_epoch >= HP_OTSU_HISTORY_SLOT_COUNT) {
            return false;
        }

        prepare_slot(sample_epoch);
#if HP_OTSU_DATA_SOURCE != HP_OTSU_DATA_SOURCE_IO_ADDED
        auto previous = object_entries.find(object_key);
        if (previous != object_entries.end()) {
            if (sample_time_ns < previous->second.sample_time_ns) {
                return false;
            }
            erase_object_entry(previous);
        }
#endif

        const size_t bin = bin_for_heat(added_heat);
        const size_t offset = slot_offset(sample_epoch) + bin;
        ceph_assert(history_bins[offset] <
                    std::numeric_limits<SlotCount>::max());
        if ((*aggregate_bins)[bin] == 0) {
            ++occupied_bin_count;
        }
        ++history_bins[offset];
        ++(*aggregate_bins)[bin];
        ++total_count;

#if HP_OTSU_DATA_SOURCE != HP_OTSU_DATA_SOURCE_IO_ADDED
        const size_t slot = static_cast<size_t>(
            sample_epoch % HP_OTSU_HISTORY_SLOT_COUNT);
        slot_objects[slot].push_back(object_key);
        auto object_position = std::prev(slot_objects[slot].end());
        auto [inserted, ok] = object_entries.emplace(
            object_key,
            ObjectEntry{bin, sample_epoch, sample_time_ns, object_position});
        ceph_assert(ok);
        (void)inserted;
#endif
        return true;
    }

    bool advance_to(uint64_t now_ns) {
        const uint64_t now_epoch = now_ns / HP_OTSU_HISTORY_SLOT_NS;
        if (!latest_epoch_initialized) {
            latest_epoch_initialized = true;
            latest_epoch = now_epoch;
            return false;
        }
        if (now_epoch <= latest_epoch) {
            return false;
        }

        const uint64_t elapsed = now_epoch - latest_epoch;
        if (elapsed >= HP_OTSU_HISTORY_SLOT_COUNT) {
            const bool changed = total_count > 0;
            clear_counts();
            latest_epoch = now_epoch;
            latest_epoch_initialized = true;
            return changed;
        }

        bool changed = false;
        for (uint64_t step = 1; step <= elapsed; ++step) {
            changed = prepare_slot(latest_epoch + step) || changed;
        }
        latest_epoch = now_epoch;
        return changed;
    }

    void clear() {
        clear_counts();
        latest_epoch_initialized = false;
        latest_epoch = 0;
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
        ceph_assert(first_bin < HP_OTSU_HISTOGRAM_BIN_COUNT);
        const double origin = bin_center(first_bin);
        const double count = static_cast<double>(total_count);
        double shifted_total_sum = 0.0;
        double shifted_total_square_sum = 0.0;
        for (size_t bin = 0; bin < HP_OTSU_HISTOGRAM_BIN_COUNT; ++bin) {
            const uint64_t bin_count = (*aggregate_bins)[bin];
            if (bin_count == 0) {
                continue;
            }
            const double shifted_center = bin_center(bin) - origin;
            shifted_total_sum += shifted_center * static_cast<double>(bin_count);
            shifted_total_square_sum += shifted_center * shifted_center *
                static_cast<double>(bin_count);
        }

        const double total_mean = shifted_total_sum / count;
        const double total_variance =
            shifted_total_square_sum / count - total_mean * total_mean;
        if (total_variance <= 0.0) {
            return std::nullopt;
        }

        uint64_t best_lhs_count = 0;
        double best_between_variance = -1.0;
        double best_threshold_score = 0.0;
        for_each_partition(
            shifted_total_sum,
            [&](size_t lhs_bin,
                size_t rhs_bin,
                uint64_t lhs_count,
                double between_variance) {
                if (between_variance > best_between_variance) {
                    best_between_variance = between_variance;
                    best_threshold_score =
                        (bin_center(lhs_bin) + bin_center(rhs_bin)) / 2.0;
                    best_lhs_count = lhs_count;
                }
            });

        if (best_between_variance < 0.0) {
            return std::nullopt;
        }

        const double near_optimal_minimum =
            HP_OTSU_NEAR_OPTIMAL_RATIO * best_between_variance;
        uint64_t near_optimal_min_lhs_count = best_lhs_count;
        uint64_t near_optimal_max_lhs_count = best_lhs_count;
        for_each_partition(
            shifted_total_sum,
            [&](size_t,
                size_t,
                uint64_t lhs_count,
                double between_variance) {
                if (between_variance >= near_optimal_minimum) {
                    near_optimal_min_lhs_count = std::min(
                        near_optimal_min_lhs_count, lhs_count);
                    near_optimal_max_lhs_count = std::max(
                        near_optimal_max_lhs_count, lhs_count);
                }
            });

        return HpOtsuResult{
            best_threshold_score,
            std::clamp(best_between_variance / total_variance, 0.0, 1.0),
            near_optimal_max_lhs_count - near_optimal_min_lhs_count,
            total_count
        };
    }

private:
    using SlotObjects = std::list<uint64_t>;

    struct ObjectEntry {
        size_t bin;
        uint64_t epoch;
        uint64_t sample_time_ns;
        SlotObjects::iterator slot_position;
    };

    using ObjectEntries = std::unordered_map<uint64_t, ObjectEntry>;

    static constexpr uint64_t invalid_epoch =
        std::numeric_limits<uint64_t>::max();

    static constexpr double max_representable_score() {
        return static_cast<double>(HP_OTSU_HISTOGRAM_BIN_COUNT) *
            HP_OTSU_LOG1P_HEAT_BIN_WIDTH;
    }

    static size_t bin_for_heat(double heat) {
        const double score = score_for_heat(heat);
        if (score >= max_representable_score()) {
            return HP_OTSU_HISTOGRAM_BIN_COUNT - 1;
        }
        const size_t bin = static_cast<size_t>(
            std::floor(score / HP_OTSU_LOG1P_HEAT_BIN_WIDTH));
        return std::min(bin, HP_OTSU_HISTOGRAM_BIN_COUNT - 1);
    }

    static double bin_center(size_t bin) {
        return (static_cast<double>(bin) + 0.5) *
            HP_OTSU_LOG1P_HEAT_BIN_WIDTH;
    }

    static size_t slot_offset(uint64_t epoch) {
        return static_cast<size_t>(epoch % HP_OTSU_HISTORY_SLOT_COUNT) *
            HP_OTSU_HISTOGRAM_BIN_COUNT;
    }

    size_t next_occupied_bin(size_t first) const {
        while (first < HP_OTSU_HISTOGRAM_BIN_COUNT &&
               (*aggregate_bins)[first] == 0) {
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
        const double origin = bin_center(first_bin);
        size_t lhs_bin = first_bin;
        while (lhs_bin < HP_OTSU_HISTOGRAM_BIN_COUNT) {
            const size_t rhs_bin = next_occupied_bin(lhs_bin + 1);
            if (rhs_bin == HP_OTSU_HISTOGRAM_BIN_COUNT) {
                break;
            }

            const uint64_t bin_count = (*aggregate_bins)[lhs_bin];
            lhs_count_u += bin_count;
            lhs_sum += (bin_center(lhs_bin) - origin) *
                static_cast<double>(bin_count);
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

    bool prepare_slot(uint64_t epoch) {
        const size_t slot = static_cast<size_t>(
            epoch % HP_OTSU_HISTORY_SLOT_COUNT);
        if (slot_epochs[slot] == epoch) {
            return false;
        }

        const size_t offset = slot * HP_OTSU_HISTOGRAM_BIN_COUNT;
        bool changed = false;
        if (slot_epochs[slot] != invalid_epoch) {
#if HP_OTSU_DATA_SOURCE != HP_OTSU_DATA_SOURCE_IO_ADDED
            for (uint64_t object_key : slot_objects[slot]) {
                auto object = object_entries.find(object_key);
                ceph_assert(object != object_entries.end());
                ceph_assert(object->second.epoch == slot_epochs[slot]);
                object_entries.erase(object);
            }
            slot_objects[slot].clear();
#endif
            for (size_t bin = 0; bin < HP_OTSU_HISTOGRAM_BIN_COUNT; ++bin) {
                const uint64_t expired = history_bins[offset + bin];
                if (expired == 0) {
                    continue;
                }
                ceph_assert((*aggregate_bins)[bin] >= expired);
                ceph_assert(total_count >= expired);
                (*aggregate_bins)[bin] -= expired;
                total_count -= expired;
                changed = true;
                if ((*aggregate_bins)[bin] == 0) {
                    ceph_assert(occupied_bin_count > 0);
                    --occupied_bin_count;
                }
                history_bins[offset + bin] = 0;
            }
        }
        slot_epochs[slot] = epoch;
        return changed;
    }

    void erase_object_entry(ObjectEntries::iterator object) {
        const ObjectEntry entry = object->second;
        const size_t slot = static_cast<size_t>(
            entry.epoch % HP_OTSU_HISTORY_SLOT_COUNT);
        const size_t offset = slot_offset(entry.epoch) + entry.bin;

        ceph_assert(slot_epochs[slot] == entry.epoch);
        ceph_assert(history_bins[offset] > 0);
        ceph_assert((*aggregate_bins)[entry.bin] > 0);
        ceph_assert(total_count > 0);
        --history_bins[offset];
        --(*aggregate_bins)[entry.bin];
        --total_count;
        if ((*aggregate_bins)[entry.bin] == 0) {
            ceph_assert(occupied_bin_count > 0);
            --occupied_bin_count;
        }
        slot_objects[slot].erase(entry.slot_position);
        object_entries.erase(object);
    }

    void clear_counts() {
        aggregate_bins->fill(0);
        std::fill_n(
            history_bins.get(),
            HP_OTSU_HISTORY_SLOT_COUNT * HP_OTSU_HISTOGRAM_BIN_COUNT,
            SlotCount{0});
        slot_epochs.fill(invalid_epoch);
        for (auto& objects : slot_objects) {
            objects.clear();
        }
        object_entries.clear();
        total_count = 0;
        occupied_bin_count = 0;
    }

    std::unique_ptr<AggregateBins> aggregate_bins;
    std::unique_ptr<SlotCount[]> history_bins;
    std::array<uint64_t, HP_OTSU_HISTORY_SLOT_COUNT> slot_epochs;
    std::array<SlotObjects, HP_OTSU_HISTORY_SLOT_COUNT> slot_objects;
    ObjectEntries object_entries;
    uint64_t total_count = 0;
    size_t occupied_bin_count = 0;
    bool latest_epoch_initialized = false;
    uint64_t latest_epoch = 0;
};

#endif
