#include "heatpredictor/hp_object_identity.h"
#include "heatpredictor/heat_predictor.h"
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>
#include <set>
#include <string>
struct CollisionHash { size_t operator()(HpObjectIdentityView) const { return 7; } };
int main() {
  HpObjectIdentityView a{26, 123, "object", "ns", 99, "locator"};
  HpObjectIdentityRegistry<CollisionHash> ids;
  auto id = ids.resolve(a);
  assert(ids.resolve(a)==id);
  std::set<uint64_t> all{id};
  for (int field=0;field<6;++field) {
    auto b=a;
    if(field==0)++b.pool;
    if(field==1)++b.placement_hash;
    if(field==2)b.name="other";
    if(field==3)b.nspace="other";
    if(field==4)++b.snapshot;
    if(field==5)b.locator="other";
    assert(all.insert(ids.resolve(b)).second);
  }
  { std::string transient="transient"; auto b=a;b.name=transient;ids.resolve(b); }
  auto b=a;b.name="transient";auto transient_id=ids.resolve(b);
  assert(ids.resolve(b)==transient_id);
  ids.erase(id);assert(ids.resolve(a)!=id);
  // Maintenance must keep identities protected by pending labels or windows.
  EvaluationQueue q(100,0,1.0,100,100,20);
  PredictionSample s{};s.io_sequence=1;
  auto first=q.begin_prediction(s,1,&a);
  auto first_id=first.sample.object_key_hash;
  auto other=a;other.nspace="different";s.io_sequence=2;
  auto second=q.begin_prediction(s,2,&other);
  assert(second.sample.object_key_hash!=first_id);
  assert(second.sample.past_window_access_count==0);
  q.complete_prediction(std::move(*second.ticket),0,0);
  s.io_sequence=3;auto repeat=q.begin_prediction(s,3,&a);
  assert(repeat.sample.object_key_hash==first_id);
  assert(repeat.sample.past_window_access_count==1);
  q.complete_prediction(std::move(*repeat.ticket),1,1);
  auto expired=q.maintain_expiry(1000,100);
  assert(q.identity_count()==0);
  s.io_sequence=4;auto fresh=q.begin_prediction(s,1001,&a);
  assert(fresh.sample.object_key_hash!=first_id);
  assert(fresh.sample.past_window_access_count==0);
  // The old prediction finishes only after the same name has a new ID.
  auto delayed=q.complete_prediction(std::move(*first.ticket),1,1);
  assert(delayed.size()==1 && delayed[0].future_window_access_count==1);
  assert(delayed[0].item.object_key_hash==first_id);
  q.complete_prediction(std::move(*fresh.ticket),0,0);
  q.maintain_expiry(2000,100);
  assert(q.identity_count()==0);
  // Capacity drops and cancellation must release identities once windows expire.
  EvaluationQueue limited(100,0,1.0,100,1,20);
  auto kept=limited.begin_prediction(s,1,&a);
  auto dropped=limited.begin_prediction(s,2,&other);
  assert(!dropped.ticket && limited.identity_count()==2);
  limited.cancel_prediction(std::move(*kept.ticket),3);
  limited.maintain_expiry(1000,100);
  assert(limited.identity_count()==0);
  // Full production API uses the same IDs under concurrent foreground calls.
  HeatPredictor hp;
  hp.set_enabled(true);
  std::vector<std::thread> workers;
  for (int i=0;i<4;++i) workers.emplace_back([&,i] {
    auto v=a;std::string ns="namespace"+std::to_string(i);v.nspace=ns;
    for(int j=0;j<100;++j) hp.predict(v,nullptr);
  });
  for(auto& worker:workers) worker.join();
  auto st=hp.status();
  assert(st.evaluation.io_count==400 && st.evaluation.heat_state_count==4);
  assert(st.predict_error_count==0 && st.background_error_count==0);
  hp.reset();
  hp.predict(a,nullptr);
  assert(hp.status().evaluation.heat_state_count==1);
  hp.set_enabled(false);
  hp.predict(a,nullptr);
  assert(hp.status().evaluation.io_count==0);
  hp.shutdown();
  // Rehashing the registry must not invalidate its reverse index.
  HpObjectIdentityRegistry<> many;
  std::vector<uint64_t> registered;
  for(int i=0;i<10000;++i) { auto v=a;v.snapshot=i;registered.push_back(many.resolve(v)); }
  for(auto x:registered) many.erase(x);
  assert(many.size()==0);
  bool rejected=false;
  try { q.begin_prediction(s,2001); } catch(const std::invalid_argument&) {rejected=true;}
  assert(rejected);
  std::cout<<"PASS full identity collision, ownership, label isolation, eviction and non-reuse\n";
}
