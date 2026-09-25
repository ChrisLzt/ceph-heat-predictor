#include "hp_trace_replay.h"
#include <cassert>
#include <iostream>
struct Tree:HoeffdingTreeClassifier<7,2>{using HoeffdingTreeClassifier<7,2>::_attempt_to_split;};
int main(){
 LeafNaiveBayesAdaptive<7,2> leaf(0);std::vector<double> x(7,0);
 leaf.learn_one(x,0,100000);x[0]=1;leaf.learn_one(x,1,100000);x[0]=1000;leaf.learn_one(x,0,1);
 auto c=leaf.best_split_suggestions(.99,.01);
 assert(c.size()==1 && c[0].feature==0 && c[0].merit>0 && c[0].threshold>0 && c[0].threshold<1);
 LeafNaiveBayesAdaptive<7,2> constant(0);
 constant.learn_one(std::vector<double>(7,0),0,100);constant.learn_one(std::vector<double>(7,0),1,100);
 assert(constant.best_split_suggestions(.99,.01).empty());
 LeafNaiveBayesAdaptive<7,2> separable(0);
 separable.learn_one(std::vector<double>(7,0),0,100);separable.learn_one(std::vector<double>(7,1),1,100);
 GaussianSplitter<7,2> original;original.update(0,0,100);original.update(1,1,100);
 auto expected=original.best_evaluated_split_suggestion({{0,100},{1,100}},0,.01);
 auto kept=separable.best_split_suggestions(.99,.01);assert(kept.size()==7);
 for(const auto& candidate:kept)assert(candidate.threshold==expected.threshold&&candidate.merit==expected.merit);
 Tree t;t._root=leaf.clone_for_prediction();auto snapshot=t.clone_for_prediction();auto before=snapshot->predict_proba_one(std::vector<double>(7,0));
 t._attempt_to_split(static_cast<LeafNaiveBayesAdaptive<7,2>*>(t._root),nullptr,0);
 assert(!t._root->is_leaf);
 double weight=0;for(auto*l:t._root->iter_leaves()){weight+=l->total_weight();assert(l->last_split_attempt_at==l->total_weight());}
 assert(std::abs(weight-200001)<1e-6);assert(snapshot->predict_proba_one(std::vector<double>(7,0))==before);
 t._max_byte_size=1;t._estimate_model_size();for(auto*l:t._root->iter_leaves())assert(!l->is_active);
 std::cout<<"PASS constant rejected; original candidates retained; single midpoint splits; inherited wait, snapshot, memory guard\n";
 std::cout<<"PASS outlier range rescued by class-mean midpoint\n";
}
