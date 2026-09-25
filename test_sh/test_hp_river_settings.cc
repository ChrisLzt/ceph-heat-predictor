#include <cmath>
#include <iostream>
#include <stdexcept>
#include "ARFClassifier.h"
#include "../src/heatpredictor/hp_config.h"

void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Forest : ARFClassifier<7, 2> {
  void verify() {
    check(max_features == 3, "round(sqrt(7)) must select 3 features");
    check(lambda_value == 6 && grace_period == 50 && delta == .01,
          "ARF defaults must match River");
    _init_ensemble();
    for (auto* model : models) {
      auto* tree = dynamic_cast<BaseTreeClassifier<7, 2>*>(model);
      check(tree != nullptr, "forest must contain randomized base trees");
    }
  }
};
struct RandomLeaf : RandomLeafNaiveBayesAdaptive<7,2> {
  RandomLeaf() : RandomLeafNaiveBayesAdaptive<7,2>(0, HP_ARF_MAX_FEATURES, 7) {}
  void verify() {
    learn_one(std::vector<double>(7,1),0);
    auto indices=feature_indices;
    check(feature_count==3,"each leaf must observe three distinct features");
    check(indices[0]!=indices[1] && indices[0]!=indices[2] && indices[1]!=indices[2],
          "feature sampling must be without replacement");
    learn_one(std::vector<double>(7,2),1);
    check(indices==feature_indices,"feature subset must remain fixed for the leaf");
  }
};
void settings() {
  RandomLeaf leaf; leaf.verify();
  check(HP_ARF_MAX_FEATURES == 3 && HP_ARF_LAMBDA == 6 &&
        HP_ARF_GRACE_PERIOD == 50 && HP_ARF_DELTA == .01,
        "production parameters must match selected River defaults");
  Forest f; f.verify();
}
void cuts() {
  GaussianSplitter<7, 2> s;
  s.update(0, 0, 50); s.update(11, 1, 50);
  auto split = s.best_evaluated_split_suggestion({{0, 50}, {1, 50}}, 0, .01);
  check(std::abs(split.threshold - 1) < 1e-12,
        "ten interior cuts on [0,11] start at 1");
}
void children_wait() {
  HoeffdingTreeClassifier<7,2> tree(50, .01);
  for (int i=0; i<50; ++i) { std::vector<double> x(7,0); x[0]=i%2 ? 11 : 0; tree.learn_one(x,i%2); }
  check(!tree._root->is_leaf,"fixture must split");
  auto* branch=static_cast<NumericBinaryBranch<7,2>*>(tree._root);
  auto* leaf=static_cast<LeafNaiveBayesAdaptive<7,2>*>(branch->children[0]);
  double inherited=leaf->total_weight();
  check(inherited > 0 && leaf->last_split_attempt_at == inherited,
        "child grace starts at inherited weight");
  for(int i=0;i<49;++i) tree.learn_one(std::vector<double>(7,0),0);
  check(leaf->last_split_attempt_at == inherited,"child must wait 50 new weighted samples");
  tree.learn_one(std::vector<double>(7,0),0);
  check(leaf->last_split_attempt_at == inherited+50,"child attempts at grace boundary");
  auto copy=tree.clone_for_prediction();
  auto* t=dynamic_cast<HoeffdingTreeClassifier<7,2>*>(copy.get());
  auto* b=static_cast<NumericBinaryBranch<7,2>*>(t->_root);
  check(static_cast<LeafNaiveBayesAdaptive<7,2>*>(b->children[0])->last_split_attempt_at == inherited+50,
        "snapshot must retain child attempt baseline");
}
void invalid_candidates_wait() {
  HoeffdingTreeClassifier<7,2> tree(50,.01);
  for(int i=0;i<4000;++i) tree.learn_one(std::vector<double>(7,0),i%2);
  check(tree._root->is_leaf && static_cast<LeafNaiveBayesAdaptive<7,2>*>(tree._root)->is_active,
        "invalid numeric candidates must not deactivate a leaf");
}
struct VotingForest : ARFClassifier<7,2> {
  VotingForest() : ARFClassifier<7,2>(2) {
    _init_ensemble();
    for(int i=0;i<2;++i) models[i]->learn_one(std::vector<double>(7,0),i);
    for(int i=0;i<10;++i) {
      int y=i==9;
      _metrics[0].update(y,0); // Accuracy .9, balanced accuracy .5
      _metrics[1].update(y,i>=6); // Accuracy .7, balanced accuracy 5/6
    }
    _prediction_weights_valid=false;
  }
  void clear_metrics() {
    for(auto& metric:_metrics) metric.clear();
    _prediction_weights_valid=false;
  }
};
void accuracy_vote() {
  VotingForest f;
  const std::vector<double> x(7,0);
  auto uncached=f.clone_for_prediction();
  check(std::abs(uncached->predict_proba_one(x)[1]-7.0/16)<1e-12,
        "uncached snapshot must use Accuracy, not balanced accuracy");
  check(std::abs(f.predict_proba_one(x)[1]-7.0/16)<1e-12,
        "live voting must use Accuracy");
  auto cached=f.clone_for_prediction();
  check(cached->predict_proba_one(x)==uncached->predict_proba_one(x),
        "cached and uncached snapshots must agree");
  f.clear_metrics();
  check(f.predict_proba_one(x)[1]==.5,"zero metrics retain unit fallback weights");
  check(std::abs(cached->predict_proba_one(x)[1]-7.0/16)<1e-12,
        "later metric updates must not affect snapshot");
}
int main() {
  int failed=0;
  for(auto test : {std::pair<const char*,void(*)()>{"settings",settings},{"cuts",cuts},
                  {"children_wait",children_wait},{"invalid_candidates_wait",invalid_candidates_wait},{"accuracy_vote",accuracy_vote}}) {
    try { test.second(); std::cout<<"PASS "<<test.first<<"\n"; }
    catch(const std::exception& e) { ++failed; std::cerr<<"FAIL "<<test.first<<": "<<e.what()<<"\n"; }
  }
  return failed ? 1 : 0;
}
