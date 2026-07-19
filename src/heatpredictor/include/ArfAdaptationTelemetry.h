#ifndef ARF_ADAPTATION_TELEMETRY_H
#define ARF_ADAPTATION_TELEMETRY_H

#include <atomic>
#include <cstdint>

struct ArfAdaptationStats {
    uint64_t warning_count{0};
    uint64_t drift_count{0};
    uint64_t background_promotion_count{0};
    uint64_t background_discard_count{0};
    uint64_t background_training_update_count{0};
    uint64_t active_background_count{0};
    uint64_t fast_model_reset_count{0};
    uint64_t fast_model_background_discard_count{0};
};

class ArfAdaptationTelemetry {
public:
    void record_warning(bool discarded_existing_background) {
        warning_count.fetch_add(1, std::memory_order_relaxed);
        if (discarded_existing_background) {
            background_discard_count.fetch_add(
                1, std::memory_order_relaxed);
        } else {
            active_background_count.fetch_add(
                1, std::memory_order_relaxed);
        }
    }

    void record_background_training_update() {
        background_training_update_count.fetch_add(
            1, std::memory_order_relaxed);
    }

    void record_drift(bool promoted_background) {
        drift_count.fetch_add(1, std::memory_order_relaxed);
        if (promoted_background) {
            background_promotion_count.fetch_add(
                1, std::memory_order_relaxed);
            active_background_count.fetch_sub(
                1, std::memory_order_relaxed);
        }
    }

    void record_fast_model_reset(bool discarded_background) {
        fast_model_reset_count.fetch_add(1, std::memory_order_relaxed);
        if (discarded_background) {
            fast_model_background_discard_count.fetch_add(
                1, std::memory_order_relaxed);
            active_background_count.fetch_sub(
                1, std::memory_order_relaxed);
        }
    }

    ArfAdaptationStats snapshot() const {
        return ArfAdaptationStats{
            warning_count.load(std::memory_order_relaxed),
            drift_count.load(std::memory_order_relaxed),
            background_promotion_count.load(std::memory_order_relaxed),
            background_discard_count.load(std::memory_order_relaxed),
            background_training_update_count.load(std::memory_order_relaxed),
            active_background_count.load(std::memory_order_relaxed),
            fast_model_reset_count.load(std::memory_order_relaxed),
            fast_model_background_discard_count.load(
                std::memory_order_relaxed),
        };
    }

    void reset() {
        warning_count.store(0, std::memory_order_relaxed);
        drift_count.store(0, std::memory_order_relaxed);
        background_promotion_count.store(0, std::memory_order_relaxed);
        background_discard_count.store(0, std::memory_order_relaxed);
        background_training_update_count.store(0, std::memory_order_relaxed);
        active_background_count.store(0, std::memory_order_relaxed);
        fast_model_reset_count.store(0, std::memory_order_relaxed);
        fast_model_background_discard_count.store(
            0, std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> warning_count{0};
    std::atomic<uint64_t> drift_count{0};
    std::atomic<uint64_t> background_promotion_count{0};
    std::atomic<uint64_t> background_discard_count{0};
    std::atomic<uint64_t> background_training_update_count{0};
    std::atomic<uint64_t> active_background_count{0};
    std::atomic<uint64_t> fast_model_reset_count{0};
    std::atomic<uint64_t> fast_model_background_discard_count{0};
};

#endif
