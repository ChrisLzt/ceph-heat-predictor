# ifndef TREE_BASE_TPP
# define TREE_BASE_TPP

# include "TreeBase.h"

# include <vector>
# include <queue>
# include <algorithm>

# include "HoeffdingTree.h"
# include "HoeffdingTreeClassifier.h"
# include "utils.h"
# include "GaussianSplitter.h"

template <int num_features, int num_labels>
BranchOrLeaf<num_features, num_labels>*
NumericBinaryBranch<num_features, num_labels>::clone_for_prediction() const {
    BranchOrLeaf<num_features, num_labels>* left =
        children[0] != nullptr ? children[0]->clone_for_prediction() : nullptr;
    BranchOrLeaf<num_features, num_labels>* right =
        children[1] != nullptr ? children[1]->clone_for_prediction() : nullptr;
    return new NumericBinaryBranch<num_features, num_labels>(
        feature, threshold, left, right, this->stats);
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::copy_leaf_state_to(
        LeafNaiveBayesAdaptive<num_features, num_labels> *copy) const {
    copy->stats = this->stats;
    copy->_mc_correct_weight = _mc_correct_weight;
    copy->_nb_correct_weight = _nb_correct_weight;
    copy->last_split_attempt_at = last_split_attempt_at;
    copy->depth = depth;
    copy->is_active = is_active;

    for (auto splitter : copy->splitters) {
        delete splitter;
    }
    copy->splitters.fill(nullptr);
    for (int i = 0; i < num_features; ++i) {
        if (splitters[i] != nullptr) {
            copy->splitters[i] =
                new GaussianSplitter<num_features, num_labels>(*splitters[i]);
        }
    }
}

template <int num_features, int num_labels>
BranchOrLeaf<num_features, num_labels>*
LeafNaiveBayesAdaptive<num_features, num_labels>::clone_for_prediction() const {
    auto *copy = new LeafNaiveBayesAdaptive<num_features, num_labels>(depth);
    copy_leaf_state_to(copy);
    return copy;
}

template <int num_features, int num_labels>
BranchOrLeaf<num_features, num_labels>*
RandomLeafNaiveBayesAdaptive<num_features, num_labels>::clone_for_prediction()
        const {
    auto *copy = new RandomLeafNaiveBayesAdaptive<num_features, num_labels>(
        this->depth, max_features);
    this->copy_leaf_state_to(copy);
    copy->feature_indices = feature_indices;
    copy->feature_count = feature_count;
    return copy;
}

template <int num_features, int num_labels>
std::vector<BranchFactory<num_features, num_labels> > LeafNaiveBayesAdaptive<num_features, num_labels>::best_split_suggestions(
    HoeffdingTree<num_features, num_labels>* tree,
    double max_share_to_split, double min_branch_fraction) {
    std::vector<BranchFactory<num_features, num_labels> > best_suggestions;
    double maj_class = max_value(this->stats);
    if (maj_class > 0.0 && maj_class / total_weight() > max_share_to_split) {
        best_suggestions.push_back(BranchFactory<num_features, num_labels>());
    } else {
        // super
        if (tree->merit_preprune) {
            BranchFactory<num_features, num_labels> null_split;
            best_suggestions.push_back(null_split);
        }
        for (int i=0;i<num_features;i++) {
            if (splitters[i] != nullptr) {
                best_suggestions.push_back(splitters[i]->best_evaluated_split_suggestion(this->stats, i, min_branch_fraction));
            }
        }
    }
    return best_suggestions;
}

template <int num_features, int num_labels>
double LeafNaiveBayesAdaptive<num_features, num_labels>::calculate_promise() {
    double total_seen = sum(this->stats);
    if (total_seen > 0.0) {
        return total_seen - max_value(this->stats);
    }
    return 0.0;
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::deactivate() {
    is_active = false;
    for (int i=0;i<num_features;i++) {
        if (splitters[i] != nullptr) {
            delete splitters[i];
            splitters[i] = nullptr;
        }
    }
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::update_splitters(const std::vector<double>& x, int y, double w) {
    if (x.size() != num_features || y < 0 || y >= num_labels) {
        throw std::invalid_argument("invalid leaf training sample");
    }
    for (size_t i=0;i<x.size();i++) {
        if (splitters[i] == nullptr) {
            splitters[i] = new GaussianSplitter<num_features, num_labels>();
        }
        splitters[i]->update(x[i], y, w);
    }
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::prediction(std::vector<double>& proba, const std::vector<double>& x) {
    if (is_active && _nb_correct_weight >= _mc_correct_weight) {
       do_naive_bayes_prediction<num_features, num_labels>(proba, x, this->stats, splitters);
    } else {
        normalize_values_in_dict(proba, this->stats);
    }
}

template <int num_features, int num_labels>
void LeafNaiveBayesAdaptive<num_features, num_labels>::learn_one(const std::vector<double>& x, int y, double w) {
    if(is_active) {
        thread_local std::vector<double> mc_pred;
        mc_pred.assign(num_labels, 0.0);
        normalize_values_in_dict(mc_pred, this->stats);
        if (this->stats.size() == 0 || max_index(mc_pred) == y) {
            _mc_correct_weight += w;
        }
        thread_local std::vector<double> nb_pred;
        do_naive_bayes_prediction<num_features, num_labels>(nb_pred, x, this->stats, splitters);
        if (max_index(nb_pred) == y) {
            _nb_correct_weight += w;
        }
    }
    // learn
    this->stats[y] += w;
    if(is_active) {
        update_splitters(x, y, w);
    }
}

template <int num_features, int num_labels>
size_t estimate_tree_memory_bytes(HoeffdingTree<num_features, num_labels>* tree) {
    if (tree == nullptr) {
        return 0;
    }

    size_t total = sizeof(HoeffdingTreeClassifier<num_features, num_labels>);
    total += num_labels * sizeof(int);
    if (tree->_root == nullptr) {
        return total;
    }

    std::queue<BranchOrLeaf<num_features, num_labels>*> q;
    q.push(tree->_root);
    while (!q.empty()) {
        BranchOrLeaf<num_features, num_labels>* b = q.front();
        q.pop();
        if (b == nullptr) {
            continue;
        }
        if (b->is_leaf) {
            auto *leaf =
                static_cast<LeafNaiveBayesAdaptive<num_features, num_labels>*>(b);
            total += leaf->is_active
                ? estimate_leaf_memory_bytes<num_features, num_labels>()
                : estimate_inactive_leaf_memory_bytes<num_features, num_labels>();
        } else {
            total += estimate_branch_memory_bytes<num_features, num_labels>();
            NumericBinaryBranch<num_features, num_labels>* branch = static_cast<NumericBinaryBranch<num_features, num_labels>*>(b);
            q.push(branch->children[0]);
            q.push(branch->children[1]);
        }
    }
    return total;
}

template <int num_labels>
constexpr size_t estimate_class_stats_memory_bytes() {
    return num_labels * (
        sizeof(std::pair<const int, double>) + 2 * sizeof(void*));
}

template <int num_features, int num_labels>
constexpr size_t estimate_branch_memory_bytes() {
    return sizeof(NumericBinaryBranch<num_features, num_labels>) +
        estimate_class_stats_memory_bytes<num_labels>();
}

template <int num_features, int num_labels>
constexpr size_t estimate_inactive_leaf_memory_bytes() {
    return sizeof(RandomLeafNaiveBayesAdaptive<num_features, num_labels>) +
        estimate_class_stats_memory_bytes<num_labels>();
}

template <int num_features, int num_labels>
constexpr size_t estimate_leaf_memory_bytes() {
    constexpr size_t splitter_size =
        sizeof(GaussianSplitter<num_features, num_labels>);
    return estimate_inactive_leaf_memory_bytes<num_features, num_labels>() +
        num_features * splitter_size;
}

template <int num_features, int num_labels>
void RandomLeafNaiveBayesAdaptive<num_features, num_labels>::update_splitters(const std::vector<double>& x, int y, double w) {
    if (x.size() != num_features || y < 0 || y >= num_labels) {
        throw std::invalid_argument("invalid random leaf training sample");
    }
    if (feature_count == 0) {
        _sample_features();
    }
    for (size_t i=0;i<feature_count;i++) {
        int fi = feature_indices[i];
        if (this->splitters[fi] == nullptr) {
            this->splitters[fi] = new GaussianSplitter<num_features, num_labels>();
        }
        this->splitters[fi]->update(x[fi], y, w);
    }
}

# endif
