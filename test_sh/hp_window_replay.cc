// Offline horizon experiment: recorded historical features stay unchanged.
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include "hp_trace_replay.h"

void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void relabel(HpReplayTrace& trace, uint64_t horizon,
    const std::function<void(HpTraceRecord&, const PredictionSample&)>& prepare = {}) {
  require(horizon>0,"zero horizon");
  auto eq=std::make_unique<EvaluationQueue>(HP_HEAT_DECAY_HORIZON_NS,HP_LRU_CAPACITY,HP_HEAT_INCREMENT,horizon);
  std::vector<size_t> order(trace.records.size());std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](size_t a,size_t b){
    auto& x=trace.records[a];auto& y=trace.records[b];
    return std::tie(x.prediction_time_ns,x.io_sequence)<std::tie(y.prediction_time_ns,y.io_sequence);
  });
  std::unordered_map<uint64_t,size_t> by_sequence;
  for(size_t i=0;i<trace.records.size();++i) require(by_sequence.emplace(trace.records[i].io_sequence,i).second,"duplicate sequence");
  std::vector<bool> completed(trace.records.size());uint64_t count=0,changed=0,future_mismatch=0;
  auto accept=[&](const std::vector<EvaluatedSample>& batch){
    for(auto& e:batch){
      auto index=by_sequence.at(e.item.io_sequence);auto& r=trace.records[index];
      require(!completed[index],"duplicate completed record");completed[index]=true;++count;
      changed+=r.actual_label!=e.label;future_mismatch+=r.future_window_access_count!=e.future_window_access_count;
      r.label_deadline_ns=e.label_deadline_ns;r.label_completion_time_ns=e.label_completion_time_ns;
      r.future_window_access_count=e.future_window_access_count;r.future_window_access_threshold=e.future_window_access_threshold;
      r.actual_label=e.label; // Keep original prediction/features purely as provenance; replay retrains from scratch.
    }
  };
  std::deque<uint64_t> deadlines;
  auto drain=[&](uint64_t until){
    while(!deadlines.empty() && deadlines.front()<=until){
      uint64_t due=deadlines.front();
      while(!deadlines.empty() && deadlines.front()==due) deadlines.pop_front();
      accept(eq->maintain_expiry(due,trace.records.size()).evaluated);
    }
  };
  uint64_t last=0;
  for(auto index:order){
    auto& r=trace.records[index];last=r.prediction_time_ns;drain(last);
    PredictionSample sample{};sample.io_sequence=r.io_sequence;sample.object_key_hash=r.object_key_hash;
    auto begin=eq->begin_prediction(sample,last);accept(begin.evaluated);
    if (prepare) prepare(r, begin.sample);
    require(last<=UINT64_MAX-horizon,"deadline overflow");deadlines.push_back(last+horizon);
    require(begin.ticket.has_value(),"EQ capacity loss");accept(eq->complete_prediction(std::move(*begin.ticket),0.0,0));
  }
  require(last<=UINT64_MAX-horizon,"deadline overflow");drain(last+horizon);
  require(count==trace.records.size(),"incomplete label reconstruction");
  std::cout<<"relabel records="<<count<<" label_changes_vs_source="<<changed<<" future_count_changes_vs_source="<<future_mismatch<<std::endl;
}
void self_test() {
  for (uint64_t seconds : {2ULL,5ULL,10ULL}) {
    uint64_t h=seconds*1000000000ULL;
    HpReplayTrace t;
    for (auto when : {100ULL,100ULL+h-1,100ULL+h}) {
      HpTraceRecord r{};r.io_sequence=t.records.size()+1;r.object_key_hash=1;
      r.prediction_time_ns=when;r.features[0]=42.0;r.hot_predict_threshold=.5;
      t.records.push_back(r);
    }
    relabel(t,h);
    require(t.records[0].label_deadline_ns==100+h,"deadline must match requested horizon");
    require(t.records[0].future_window_access_count==1,"right-open future excludes exact deadline");
    require(t.records[0].future_window_access_threshold==1,"sparse threshold must remain one");
    require(t.records[0].actual_label==1,"one future access is hot under sparse threshold");
    require(t.records.back().future_window_access_count==0 && t.records.back().actual_label==0,"tail must drain cold");
    require(t.records[0].features[0]==42.0,"historical features must stay unchanged");
  }
  std::cout << "PASS: 2/5/10s boundaries, sparse labels, tail drain, feature preservation\n";
}
#ifndef HP_WINDOW_REPLAY_NO_MAIN
int main(int argc,char** argv) {
 try {
  if(argc==2 && std::string(argv[1])=="--self-test") {self_test();return 0;}
  require(argc==4,"usage: hp_window_replay TRACE.bin SECONDS NEW_OUTPUT.tsv");
  const uint64_t sec=std::stoull(argv[2]);require(sec==2||sec==5||sec==10,"only 2,5,10 seconds supported");
  require(!std::filesystem::exists(argv[3]),"refuse existing output");
  auto trace=read_hp_trace(argv[1]);relabel(trace,sec*1000000000ULL);
  HpReplayOptions options;options.adaptation_profile=HpReplayAdaptationProfile::disabled;
  auto result=replay_hp_trace(trace,options);require(result.trained_sample_count==trace.records.size(),"training sample loss");
  std::ofstream out(argv[3]);write_hp_replay_tsv(out,trace,result);out.close();require(out.good(),"output write failed");
  std::cout<<"completed horizon="<<sec<<" records="<<trace.records.size()<<" trained="<<result.trained_sample_count<<" snapshots="<<result.snapshot_publish_count<<"\n";
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}

#endif
