#ifndef CEPH_HEATPREDICTOR_HP_PREDICTION_THRESHOLD_H
#define CEPH_HEATPREDICTOR_HP_PREDICTION_THRESHOLD_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>

#include "hp_config.h"

class HpPredictionThresholdCalibrator {
public:
    explicit HpPredictionThresholdCalibrator(
            size_t capacity = HP_PREDICT_CALIBRATION_WINDOW,
            size_t update_interval = HP_PREDICT_CALIBRATION_UPDATE_INTERVAL,
            size_t min_samples = HP_PREDICT_CALIBRATION_MIN_SAMPLES,
            double initial_threshold = HP_HOT_PREDICT_THRESHOLD,
            double min_threshold = HP_HOT_PREDICT_THRESHOLD_MIN,
            double max_threshold = HP_HOT_PREDICT_THRESHOLD_MAX,
            double ema_alpha = HP_HOT_PREDICT_THRESHOLD_EMA_ALPHA) :
            capacity(capacity),
            update_interval(update_interval),
            min_samples(min_samples),
            min_threshold(min_threshold),
            max_threshold(max_threshold),
            ema_alpha(ema_alpha),
            current_threshold(initial_threshold),
            candidate_threshold(initial_threshold) {
        if (capacity == 0 || update_interval == 0 || min_samples == 0 ||
            min_samples > capacity ||
            !(min_threshold >= 0.0 && min_threshold <= max_threshold &&
              max_threshold <= 1.0) ||
            !(initial_threshold >= min_threshold &&
              initial_threshold <= max_threshold) ||
            !(ema_alpha > 0.0 && ema_alpha <= 1.0)) {
            throw std::invalid_argument(
                "invalid prediction threshold calibrator parameters");
        }
    }

    void observe(double probability, int label) {
        if (!std::isfinite(probability) || probability < 0.0 ||
            probability > 1.0 || (label != 0 && label != 1)) {
            throw std::invalid_argument("invalid calibration observation");
        }

        const uint16_t bin = probability_bin(probability);
        observations.push_back(Observation{bin, label == 1});
        counts_for(label)[bin]++;

        if (observations.size() > capacity) {
            const Observation expired = observations.front();
            observations.pop_front();
            auto& counts = expired.hot ? hot_counts : cold_counts;
            if (counts[expired.bin] == 0) {
                throw std::logic_error("invalid calibration histogram state");
            }
            counts[expired.bin]--;
        }

        observation_count++;
        if (observations.size() >= min_samples &&
            observation_count % update_interval == 0) {
            update_threshold();
        }
    }

    size_t size() const {
        return observations.size();
    }

    double threshold() const {
        return current_threshold;
    }

    double target_threshold() const {
        return candidate_threshold;
    }

    double current_accuracy() const {
        return last_current_accuracy;
    }

    double target_accuracy() const {
        return last_candidate_accuracy;
    }

private:
    static_assert(HP_PREDICT_PROBABILITY_BIN_COUNT >= 2,
                  "prediction probability histogram needs at least two bins");
    static_assert(
        HP_PREDICT_PROBABILITY_BIN_COUNT <=
            static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1,
        "prediction probability histogram bin index must fit uint16_t");

    struct Observation {
        uint16_t bin;
        bool hot;
    };

    static constexpr size_t bin_count = HP_PREDICT_PROBABILITY_BIN_COUNT;
    static constexpr double bin_scale = static_cast<double>(bin_count - 1);

    uint16_t probability_bin(double probability) const {
        const double scaled = std::floor(probability * bin_scale + 1e-12);
        return static_cast<uint16_t>(std::clamp(
            scaled, 0.0, bin_scale));
    }

    size_t threshold_bin(double threshold) const {
        const double scaled = std::ceil(threshold * bin_scale - 1e-12);
        return static_cast<size_t>(std::clamp(
            scaled, 0.0, bin_scale));
    }

    double threshold_for_bin(size_t bin) const {
        return static_cast<double>(bin) / bin_scale;
    }

    std::array<uint64_t, bin_count>& counts_for(int label) {
        return label == 1 ? hot_counts : cold_counts;
    }

    uint64_t correct_count(size_t threshold) const {
        uint64_t correct = 0;
        for (size_t bin = 0; bin < bin_count; ++bin) {
            correct += bin < threshold ? cold_counts[bin] : hot_counts[bin];
        }
        return correct;
    }

    void update_threshold() {
        const size_t min_bin = threshold_bin(min_threshold);
        const size_t max_bin = threshold_bin(max_threshold);
        const size_t current_bin = threshold_bin(current_threshold);

        uint64_t cold_below = 0;
        uint64_t hot_at_or_above = 0;
        for (uint64_t count : hot_counts) {
            hot_at_or_above += count;
        }

        uint64_t best_correct = 0;
        size_t best_bin = current_bin;
        size_t best_distance = std::numeric_limits<size_t>::max();

        for (size_t bin = 0; bin < bin_count; ++bin) {
            if (bin >= min_bin && bin <= max_bin) {
                const uint64_t correct = cold_below + hot_at_or_above;
                const size_t distance = bin > current_bin
                    ? bin - current_bin
                    : current_bin - bin;
                if (correct > best_correct ||
                    (correct == best_correct && distance < best_distance)) {
                    best_correct = correct;
                    best_bin = bin;
                    best_distance = distance;
                }
            }
            cold_below += cold_counts[bin];
            hot_at_or_above -= hot_counts[bin];
        }

        candidate_threshold = threshold_for_bin(best_bin);
        last_candidate_accuracy = static_cast<double>(best_correct) /
            static_cast<double>(observations.size());
        current_threshold = std::clamp(
            (1.0 - ema_alpha) * current_threshold +
            ema_alpha * candidate_threshold,
            min_threshold,
            max_threshold);
        last_current_accuracy = static_cast<double>(
            correct_count(threshold_bin(current_threshold))) /
            static_cast<double>(observations.size());
    }

    size_t capacity;
    size_t update_interval;
    size_t min_samples;
    double min_threshold;
    double max_threshold;
    double ema_alpha;
    double current_threshold;
    double candidate_threshold;
    double last_current_accuracy = 0.0;
    double last_candidate_accuracy = 0.0;
    uint64_t observation_count = 0;
    std::deque<Observation> observations;
    std::array<uint64_t, bin_count> hot_counts{};
    std::array<uint64_t, bin_count> cold_counts{};
};

#endif
