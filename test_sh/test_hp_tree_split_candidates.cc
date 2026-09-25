#include <cmath>
#include <iostream>
#include <stdexcept>
#include "ARFClassifier.h"

using Tree = HoeffdingTreeClassifier<5, 2>;
using Leaf = LeafNaiveBayesAdaptive<5, 2>;
const std::vector<double> cold = {-1, 0, 0, 0, 0};
const std::vector<double> hot = {1, 0, 0, 0, 0};
const std::vector<double> constant(5, 0);

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void require_active(Tree& tree) {
    require(tree._root && tree._root->is_leaf &&
            static_cast<Leaf*>(tree._root)->is_active,
            "temporary lack of a split must preserve an active leaf");
}
void learn_separable(Tree& tree) {
    for (int i = 0; i < 10000; ++i)
        tree.learn_one(i % 2 ? hot : cold, i % 2);
    require(!tree._root->is_leaf, "later separable data must grow a branch");
    require(tree.predict_proba_one(cold)[1] < 0.2 &&
            tree.predict_proba_one(hot)[1] > 0.8,
            "later data must recover both prediction classes");
    auto snapshot = tree.clone_for_prediction();
    const auto before = snapshot->predict_proba_one(hot);
    require(before == tree.predict_proba_one(hot), "snapshot must preserve predictions");
    for (int i = 0; i < 2000; ++i) tree.learn_one(hot, 0);
    require(snapshot->predict_proba_one(hot) == before,
            "training must not mutate the published snapshot");
}
void purity_waits() {
    {
        Tree tree(100, .001, .05, .99, .01);
        for (int i = 0; i < 199; ++i) tree.learn_one(cold, 0);
        tree.learn_one(hot, 1);
        require_active(tree);
        learn_separable(tree);
    }
}
void constant_features_wait() {
    Tree tree(100, .001, .05, .99, .01);
    for (int i = 0; i < 1500; ++i) tree.learn_one(constant, i % 2);
    require_active(tree);
    learn_separable(tree);
}
void rejected_small_branch_waits() {
    Tree tree(100, .001, .05, 1.0, .01);
    for (int batch = 0; batch < 8; ++batch) {
        for (int i = 0; i < 199; ++i) tree.learn_one(cold, 0);
        tree.learn_one(hot, 1);
    }
    require_active(tree);
    learn_separable(tree);
}
void valid_split_beats_empty_features() {
    Tree tree(100, .001, .05, .99, .01);
    learn_separable(tree);
}
struct EmptyObserversTree : Tree {
    void attempt_without_observers() {
        _root = new Leaf(0);
        _root->stats = {{0, 50}, {1, 50}};
        _n_active_leaves = 1;
        _attempt_to_split(static_cast<Leaf*>(_root), nullptr, 0);
    }
};
void empty_observers_must_keep_learning() {
    EmptyObserversTree tree;
    tree.attempt_without_observers();
    require_active(tree);
    learn_separable(tree);
}
void old_depth_boundary_keeps_learning() {
    Tree tree(50, .01, .05, .99, .01);
    tree.learn_one(cold, 0);
    auto* leaf = static_cast<Leaf*>(tree._root);
    leaf->depth = 980; // Former limit: test the boundary without a huge tree.
    tree.learn_one(hot, 1);
    require_active(tree);
    learn_separable(tree);
    auto* branch = static_cast<NumericBinaryBranch<5,2>*>(tree._root);
    require(static_cast<Leaf*>(branch->children[0])->depth == 981,
            "children must grow beyond the former depth cap");
}
void memory_limit_still_applies() {
    Tree tree(100, .001, .05, .99, .01);
    for (int i = 0; i < 1500; ++i) tree.learn_one(constant, i % 2);
    require_active(tree);
    tree._max_byte_size = 1;
    tree._estimate_model_size();
    require(!static_cast<Leaf*>(tree._root)->is_active,
            "memory pressure must still deactivate a leaf");
}
int main() {
    int failed = 0;
    const std::pair<const char*, void(*)()> tests[] = {
        {"purity_waits", purity_waits},
        {"constant_features_wait", constant_features_wait},
        {"rejected_small_branch_waits", rejected_small_branch_waits},
        {"valid_split_beats_empty_features", valid_split_beats_empty_features},
        {"empty_observers_must_keep_learning", empty_observers_must_keep_learning},
        {"old_depth_boundary_keeps_learning", old_depth_boundary_keeps_learning},
        {"memory_limit_still_applies", memory_limit_still_applies},
    };
    for (const auto& test : tests) {
        try { test.second(); std::cout << "PASS: " << test.first << '\n'; }
        catch (const std::exception& e) {
            ++failed; std::cerr << "FAIL: " << test.first << ": " << e.what() << '\n';
        }
    }
    return failed ? 1 : 0;
}
