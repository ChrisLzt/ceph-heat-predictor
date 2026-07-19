#ifndef CEPH_HEATPREDICTOR_HP_TRACE_H
#define CEPH_HEATPREDICTOR_HP_TRACE_H

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static constexpr uint32_t HP_TRACE_SCHEMA_VERSION = 2;
static constexpr size_t HP_TRACE_MAX_FEATURES = 8;
static constexpr char HP_TRACE_MAGIC[8] = {
    'H', 'P', 'T', 'R', 'A', 'C', 'E', '1'};

enum class HpTraceOutcome : uint8_t {
    evaluated = 0,
    evaluation_capacity_drop = 1,
    prediction_error = 2,
};

enum HpTraceFlags : uint8_t {
    HP_TRACE_FLAG_NONE = 0,
    HP_TRACE_FLAG_COLD_START_FALLBACK = 1U << 0,
    HP_TRACE_FLAG_EVALUATION_CAPACITY_DROP = 1U << 1,
};

#pragma pack(push, 1)
struct HpTraceFileHeader {
    char magic[8];
    uint32_t schema_version;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t feature_count;
    int32_t osd_id;
    uint32_t reserved;
    uint64_t session_id;
    uint64_t start_wall_time_ns;
    uint64_t start_monotonic_time_ns;
    uint64_t config_hash;
    char git_commit[64];
    char phase[64];
};

struct HpTraceRecord {
    uint64_t io_sequence;
    uint64_t object_key_hash;
    uint64_t prediction_time_ns;
    uint64_t label_deadline_ns;
    uint64_t label_completion_time_ns;
    double features[HP_TRACE_MAX_FEATURES];
    double heat_after_current_access;
    double heat_label_threshold_at_prediction;
    double predicted_hot_probability;
    double hot_predict_threshold;
    double label_heat;
    double label_heat_threshold;
    uint64_t tracked_access_count;
    uint64_t time_since_previous_access_ns;
    uint64_t long_window_access_count;
    uint64_t short_window_access_count;
    uint64_t future_window_access_count;
    uint8_t outcome;
    uint8_t flags;
    int8_t predicted_label;
    int8_t actual_label;
    uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(HpTraceFileHeader) == 192,
              "trace file header layout is part of the disk schema");
static_assert(sizeof(HpTraceRecord) == 200,
              "trace record layout is part of the disk schema");

struct HpTraceStatus {
    bool enabled = false;
    uint64_t session_id = 0;
    size_t queue_length = 0;
    uint64_t written_count = 0;
    uint64_t drop_count = 0;
    uint64_t write_error_count = 0;
    std::string path;
    std::string phase;
};

class HpTraceWriter {
public:
    static constexpr size_t DEFAULT_QUEUE_CAPACITY = 65536;
    static constexpr size_t DEFAULT_BATCH_SIZE = 4096;

private:
    struct QueueSlot {
        std::atomic<size_t> sequence{0};
        HpTraceRecord record{};
    };

    const size_t queue_capacity;
    const size_t batch_size;
    std::unique_ptr<QueueSlot[]> queue;
    std::atomic<size_t> enqueue_position{0};
    size_t dequeue_position = 0;
    std::atomic<size_t> queue_length{0};
    std::atomic<uint64_t> submit_generation{0};
    std::atomic<uint64_t> active_submitters{0};
    mutable std::mutex lifecycle_mutex;
    std::mutex wait_mutex;
    std::condition_variable cv;
    std::ofstream output;
    std::thread writer_thread;
    std::atomic<bool> stop_requested{false};
    uint64_t next_session_id = 0;
    uint64_t current_session_id = 0;
    std::string current_path;
    std::string current_phase;
    std::atomic<uint64_t> written_count{0};
    std::atomic<uint64_t> drop_count{0};
    std::atomic<uint64_t> write_error_count{0};

    static uint64_t wall_time_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }

    static uint64_t monotonic_time_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    template <size_t N>
    static void copy_text(char (&destination)[N], const std::string& source) {
        static_assert(N > 0, "trace text field must include a terminator");
        const size_t length = std::min(source.size(), N - 1);
        std::memcpy(destination, source.data(), length);
        destination[length] = '\0';
    }

    static std::string make_path(
            const std::string& directory,
            int32_t osd_id,
            uint64_t timestamp_ns,
            uint64_t session_id) {
        std::string path = directory;
        if (!path.empty() && path.back() != '/') {
            path.push_back('/');
        }
        path += "ceph-object-hp-trace-osd." + std::to_string(osd_id) +
            "-" + std::to_string(timestamp_ns) +
            "-s" + std::to_string(session_id) + ".bin";
        return path;
    }

    void reset_queue() {
        enqueue_position.store(0, std::memory_order_relaxed);
        dequeue_position = 0;
        queue_length.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < queue_capacity; ++i) {
            queue[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool try_dequeue(HpTraceRecord& record) {
        const size_t position = dequeue_position;
        QueueSlot& slot = queue[position % queue_capacity];
        const size_t sequence =
            slot.sequence.load(std::memory_order_acquire);
        const auto difference = static_cast<std::intptr_t>(sequence) -
            static_cast<std::intptr_t>(position + 1);
        if (difference != 0) {
            return false;
        }

        record = slot.record;
        slot.sequence.store(
            position + queue_capacity, std::memory_order_release);
        dequeue_position = position + 1;
        queue_length.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    void writer_loop() {
        std::vector<HpTraceRecord> pending;
        pending.reserve(batch_size);
        size_t records_since_flush = 0;

        while (true) {
            pending.clear();
            HpTraceRecord record{};
            while (pending.size() < batch_size && try_dequeue(record)) {
                pending.push_back(record);
            }
            if (pending.empty()) {
                if (stop_requested.load(std::memory_order_acquire) &&
                    queue_length.load(std::memory_order_acquire) == 0) {
                    break;
                }
                std::unique_lock<std::mutex> lock(wait_mutex);
                cv.wait_for(lock, std::chrono::milliseconds(10), [this] {
                    return stop_requested.load(std::memory_order_acquire) ||
                        queue_length.load(std::memory_order_acquire) != 0;
                });
                continue;
            }

            size_t offset = 0;
            while (offset < pending.size()) {
                const size_t count = std::min(
                    batch_size, pending.size() - offset);
                output.write(
                    reinterpret_cast<const char *>(pending.data() + offset),
                    static_cast<std::streamsize>(
                        count * sizeof(HpTraceRecord)));
                if (output.good()) {
                    written_count.fetch_add(
                        count, std::memory_order_relaxed);
                } else {
                    write_error_count.fetch_add(
                        pending.size() - offset,
                        std::memory_order_relaxed);
                    break;
                }
                offset += count;
            }
            records_since_flush += pending.size();
            if (records_since_flush >= batch_size) {
                output.flush();
                if (!output.good()) {
                    write_error_count.fetch_add(
                        1, std::memory_order_relaxed);
                }
                records_since_flush = 0;
            }
        }
    }

    void stop_locked() {
        uint64_t generation =
            submit_generation.load(std::memory_order_acquire);
        if ((generation & 1U) != 0) {
            submit_generation.fetch_add(1, std::memory_order_acq_rel);
        }
        while (active_submitters.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
        stop_requested.store(true, std::memory_order_release);
        cv.notify_all();
        if (writer_thread.joinable()) {
            writer_thread.join();
        }
        if (output.is_open()) {
            output.flush();
            if (!output.good()) {
                write_error_count.fetch_add(1, std::memory_order_relaxed);
            }
            output.close();
        }
        stop_requested.store(false, std::memory_order_relaxed);
    }

public:
    explicit HpTraceWriter(
            size_t queue_capacity = DEFAULT_QUEUE_CAPACITY,
            size_t batch_size = DEFAULT_BATCH_SIZE) :
        queue_capacity(std::max<size_t>(2, queue_capacity)),
        batch_size(std::max<size_t>(1, batch_size)),
        queue(std::make_unique<QueueSlot[]>(this->queue_capacity)) {
        reset_queue();
    }

    ~HpTraceWriter() {
        stop();
    }

    HpTraceWriter(const HpTraceWriter&) = delete;
    HpTraceWriter& operator=(const HpTraceWriter&) = delete;

    bool start(
            const std::string& directory,
            int32_t osd_id,
            const std::string& phase,
            const std::string& git_commit,
            uint64_t config_hash,
            uint32_t feature_count) {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
        stop_locked();
        if (directory.empty() || feature_count > HP_TRACE_MAX_FEATURES) {
            return false;
        }

        const uint64_t timestamp_ns = wall_time_ns();
        const uint64_t session_id = ++next_session_id;
        const std::string path = make_path(
            directory, osd_id, timestamp_ns, session_id);
        std::ofstream next_output(
            path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!next_output.is_open()) {
            write_error_count.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        HpTraceFileHeader header{};
        std::memcpy(header.magic, HP_TRACE_MAGIC, sizeof(header.magic));
        header.schema_version = HP_TRACE_SCHEMA_VERSION;
        header.header_size = sizeof(HpTraceFileHeader);
        header.record_size = sizeof(HpTraceRecord);
        header.feature_count = feature_count;
        header.osd_id = osd_id;
        header.session_id = session_id;
        header.start_wall_time_ns = timestamp_ns;
        header.start_monotonic_time_ns = monotonic_time_ns();
        header.config_hash = config_hash;
        copy_text(header.git_commit, git_commit);
        copy_text(header.phase, phase);
        next_output.write(
            reinterpret_cast<const char *>(&header), sizeof(header));
        next_output.flush();
        if (!next_output.good()) {
            write_error_count.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        reset_queue();
        output = std::move(next_output);
        current_session_id = session_id;
        current_path = path;
        current_phase = phase;
        try {
            writer_thread = std::thread(&HpTraceWriter::writer_loop, this);
        } catch (...) {
            output.close();
            write_error_count.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        submit_generation.fetch_add(1, std::memory_order_release);
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
        stop_locked();
    }

    bool try_submit(const HpTraceRecord& record) {
        const uint64_t generation =
            submit_generation.load(std::memory_order_acquire);
        if ((generation & 1U) == 0) {
            return false;
        }
        active_submitters.fetch_add(1, std::memory_order_acq_rel);
        if (submit_generation.load(std::memory_order_acquire) != generation) {
            active_submitters.fetch_sub(1, std::memory_order_release);
            return false;
        }

        size_t position = enqueue_position.load(std::memory_order_relaxed);
        QueueSlot *slot = nullptr;
        while (true) {
            slot = &queue[position % queue_capacity];
            const size_t sequence =
                slot->sequence.load(std::memory_order_acquire);
            const auto difference = static_cast<std::intptr_t>(sequence) -
                static_cast<std::intptr_t>(position);
            if (difference == 0) {
                if (enqueue_position.compare_exchange_weak(
                        position, position + 1,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    break;
                }
            } else if (difference < 0) {
                active_submitters.fetch_sub(1, std::memory_order_release);
                drop_count.fetch_add(1, std::memory_order_relaxed);
                return false;
            } else {
                position = enqueue_position.load(std::memory_order_relaxed);
            }
        }

        slot->record = record;
        queue_length.fetch_add(1, std::memory_order_relaxed);
        slot->sequence.store(position + 1, std::memory_order_release);
        active_submitters.fetch_sub(1, std::memory_order_release);
        cv.notify_one();
        return true;
    }

    bool is_enabled() const {
        return (submit_generation.load(std::memory_order_acquire) & 1U) != 0;
    }

    HpTraceStatus status() const {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        HpTraceStatus result;
        result.enabled = is_enabled();
        result.written_count = written_count.load(std::memory_order_relaxed);
        result.drop_count = drop_count.load(std::memory_order_relaxed);
        result.write_error_count =
            write_error_count.load(std::memory_order_relaxed);
        result.queue_length =
            queue_length.load(std::memory_order_relaxed);
        result.session_id = current_session_id;
        result.path = current_path;
        result.phase = current_phase;
        return result;
    }
};

#endif
