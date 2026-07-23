#ifndef CEPH_HEATPREDICTOR_HP_EXPIRY_HEAP_H
#define CEPH_HEATPREDICTOR_HP_EXPIRY_HEAP_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

class HpExpiryHeap {
public:
    void upsert(uint64_t key, uint64_t deadline_ns) {
        auto existing = positions.find(key);
        if (existing == positions.end()) {
            const size_t index = nodes.size();
            nodes.push_back(Node{deadline_ns, key});
            positions.emplace(key, index);
            sift_up(index);
            return;
        }

        const size_t index = existing->second;
        nodes[index].deadline_ns = deadline_ns;
        restore_heap_at(index);
    }

    bool erase(uint64_t key) {
        auto position = positions.find(key);
        if (position == positions.end()) {
            return false;
        }

        const size_t index = position->second;
        const size_t last = nodes.size() - 1;
        positions.erase(position);
        if (index != last) {
            nodes[index] = nodes[last];
            positions[nodes[index].key] = index;
        }
        nodes.pop_back();
        if (index < nodes.size()) {
            restore_heap_at(index);
        }
        return true;
    }

    std::optional<uint64_t> due_key(uint64_t now_ns) const {
        if (nodes.empty() || nodes.front().deadline_ns > now_ns) {
            return std::nullopt;
        }
        return nodes.front().key;
    }

    std::optional<uint64_t> earliest_deadline_ns() const {
        if (nodes.empty()) {
            return std::nullopt;
        }
        return nodes.front().deadline_ns;
    }

    size_t size() const {
        return nodes.size();
    }

    bool empty() const {
        return nodes.empty();
    }

    void clear() {
        nodes.clear();
        positions.clear();
    }

private:
    struct Node {
        uint64_t deadline_ns;
        uint64_t key;
    };

    static bool less(const Node& lhs, const Node& rhs) {
        return lhs.deadline_ns < rhs.deadline_ns ||
            (lhs.deadline_ns == rhs.deadline_ns && lhs.key < rhs.key);
    }

    void swap_nodes(size_t lhs, size_t rhs) {
        std::swap(nodes[lhs], nodes[rhs]);
        positions[nodes[lhs].key] = lhs;
        positions[nodes[rhs].key] = rhs;
    }

    void sift_up(size_t index) {
        while (index > 0) {
            const size_t parent = (index - 1) / 2;
            if (!less(nodes[index], nodes[parent])) {
                return;
            }
            swap_nodes(index, parent);
            index = parent;
        }
    }

    void sift_down(size_t index) {
        while (true) {
            const size_t left = index * 2 + 1;
            if (left >= nodes.size()) {
                return;
            }
            const size_t right = left + 1;
            size_t smallest = left;
            if (right < nodes.size() && less(nodes[right], nodes[left])) {
                smallest = right;
            }
            if (!less(nodes[smallest], nodes[index])) {
                return;
            }
            swap_nodes(index, smallest);
            index = smallest;
        }
    }

    void restore_heap_at(size_t index) {
        if (index > 0 && less(nodes[index], nodes[(index - 1) / 2])) {
            sift_up(index);
        } else {
            sift_down(index);
        }
    }

    std::vector<Node> nodes;
    std::unordered_map<uint64_t, size_t> positions;
};

#endif
