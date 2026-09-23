#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

// Host-neutral hobject identity. Intentionally excludes shard and generation.
// Views are borrowed only for the duration of the foreground lookup.
struct HpObjectIdentityView {
    int64_t pool;
    uint32_t placement_hash;
    std::string_view name;
    std::string_view nspace;
    uint64_t snapshot;
    std::string_view locator;
};

struct HpObjectIdentityHash {
    size_t operator()(HpObjectIdentityView object) const noexcept {
        size_t hash = std::hash<int64_t>{}(object.pool);
        auto mix = [&](size_t value) {
            hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        };
        mix(object.placement_hash);
        mix(std::hash<std::string_view>{}(object.name));
        mix(std::hash<std::string_view>{}(object.nspace));
        mix(std::hash<uint64_t>{}(object.snapshot));
        mix(std::hash<std::string_view>{}(object.locator));
        return hash;
    }
};

inline uint64_t hp_allocate_object_id() {
    // Also survives predictor reset and registry eviction. IDs are local to
    // this process, not persistent object identifiers or cross-OSD keys.
    static std::atomic<uint64_t> next{1};
    uint64_t value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("HP object ID exhausted");
        if (next.compare_exchange_weak(value, value + 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            return value;
    }
}

// Protected by the owning EvaluationQueue's existing mutex. Hashes only
// select candidates; complete fields decide identity, including collisions.
template<class Hasher = HpObjectIdentityHash>
class HpObjectIdentityRegistry {
    struct Entry {
        uint64_t id;
        size_t hash;
        int64_t pool;
        uint32_t placement_hash;
        std::string name, nspace;
        uint64_t snapshot;
        std::string locator;
        Entry(uint64_t id, size_t hash, HpObjectIdentityView v)
          : id(id), hash(hash), pool(v.pool), placement_hash(v.placement_hash),
            name(v.name), nspace(v.nspace), snapshot(v.snapshot), locator(v.locator) {}
        bool equals(HpObjectIdentityView v) const noexcept {
            return pool == v.pool && placement_hash == v.placement_hash &&
                snapshot == v.snapshot && name == v.name &&
                nspace == v.nspace && locator == v.locator;
        }
    };
    // References/pointers to unordered_map elements survive rehash.
    std::unordered_multimap<size_t, Entry> entries;
    std::unordered_map<uint64_t, const Entry*> by_id;
    Hasher hasher;
public:
    HpObjectIdentityRegistry() = default;
    HpObjectIdentityRegistry(const HpObjectIdentityRegistry&) = delete;
    HpObjectIdentityRegistry& operator=(const HpObjectIdentityRegistry&) = delete;
    uint64_t resolve(HpObjectIdentityView v) {
        const size_t hash = hasher(v);
        const auto range = entries.equal_range(hash);
        for (auto p = range.first; p != range.second; ++p)
            if (p->second.equals(v)) return p->second.id;
        const uint64_t id = hp_allocate_object_id();
        auto p = entries.emplace(hash, Entry(id, hash, v));
        try {
            by_id.emplace(id, &p->second);
        } catch (...) {
            entries.erase(p);
            throw;
        }
        return id;
    }
    void erase(uint64_t id) {
        auto p = by_id.find(id);
        if (p == by_id.end()) return;
        const Entry* entry = p->second;
        const auto range = entries.equal_range(entry->hash);
        for (auto i = range.first; i != range.second; ++i) {
            if (&i->second == entry) {
                entries.erase(i);
                by_id.erase(p);
                return;
            }
        }
        throw std::logic_error("HP object identity registry inconsistent");
    }
    size_t size() const noexcept { return by_id.size(); }
};
