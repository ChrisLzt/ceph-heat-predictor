# ifndef HOEFFDING_TREE_TPP
# define HOEFFDING_TREE_TPP

# include "HoeffdingTree.h"

# include <vector>
# include <algorithm>

# include "TreeBase.h"

template <int num_features, int num_labels>
HoeffdingTree<num_features, num_labels>::~HoeffdingTree() {
    delete _root;
}

template <int num_features, int num_labels>
void HoeffdingTree<num_features, num_labels>::estimate_leaves() {
    _active_leaf_size_estimate = estimate_leaf_memory_bytes<num_features, num_labels>();
    _inactive_leaf_size_estimate =
        estimate_inactive_leaf_memory_bytes<num_features, num_labels>();
}

template <int num_features, int num_labels>
void HoeffdingTree<num_features, num_labels>::_enforce_size_limit(
        size_t estimated_model_size) {
    if (static_cast<double>(estimated_model_size) <= _max_byte_size) {
        return;
    }
    if (_root == nullptr) {
        _growth_allowed = false;
        return;
    }
    if (stop_mem_management) {
        _growth_allowed = false;
        return;
    }

    std::vector<LeafNaiveBayesAdaptive<num_features, num_labels>*> leaves = _root->iter_leaves();
    _n_active_leaves = 0;
    _n_inactive_leaves = 0;
    for (const auto *leaf : leaves) {
        if (leaf->is_active) {
            ++_n_active_leaves;
        } else {
            ++_n_inactive_leaves;
        }
    }
    std::sort(leaves.begin(), leaves.end(), [](LeafNaiveBayesAdaptive<num_features, num_labels>* a, LeafNaiveBayesAdaptive<num_features, num_labels>* b) {
        return a->calculate_promise() < b->calculate_promise();});

    const size_t leaf_saving =
        _active_leaf_size_estimate > _inactive_leaf_size_estimate
        ? static_cast<size_t>(
            _active_leaf_size_estimate - _inactive_leaf_size_estimate)
        : 0;
    for (auto *leaf : leaves) {
        if (static_cast<double>(estimated_model_size) <= _max_byte_size) {
            break;
        }
        if (leaf->is_active) {
            leaf->deactivate();
            --_n_active_leaves;
            ++_n_inactive_leaves;
            estimated_model_size = estimated_model_size > leaf_saving
                ? estimated_model_size - leaf_saving
                : 0;
        }
    }
    _growth_allowed =
        static_cast<double>(estimated_model_size) <= _max_byte_size;
}

template <int num_features, int num_labels>
void HoeffdingTree<num_features, num_labels>::_estimate_model_size() {
    const size_t model_size =
        estimate_tree_memory_bytes<num_features, num_labels>(this);
    _enforce_size_limit(model_size);
}

# endif
