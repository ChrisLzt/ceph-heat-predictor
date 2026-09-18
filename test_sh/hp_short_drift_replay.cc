// Experimental driver reusing the frozen horizon relabel implementation.
#define HP_WINDOW_REPLAY_NO_MAIN
#include "hp_window_replay.cc"

uint64_t rebuild_short_features(HpReplayTrace& trace,uint64_t short_ns) {
  require(short_ns>0,"short window must be positive");
  std::vector<size_t> order(trace.records.size());std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](size_t a,size_t b){
    auto& x=trace.records[a];auto& y=trace.records[b];
    return std::tie(x.prediction_time_ns,x.io_sequence)<std::tie(y.prediction_time_ns,y.io_sequence);
  });
  std::unordered_map<uint64_t,std::deque<uint64_t>> events;
  uint64_t mismatch=0;
  for(auto index:order){
    auto& r=trace.records[index];auto now=r.prediction_time_ns;auto& q=events[r.object_key_hash];
    while(!q.empty() && now>=short_ns && q.front()<=now-short_ns)q.pop_front();
    uint64_t count=std::lower_bound(q.begin(),q.end(),now)-q.begin();
    double log_count=std::log2(1.0+count);
    if(short_ns==2000000000ULL && std::abs(r.features[4]-log_count)>1e-9)++mismatch;
    r.features[4]=log_count;
    r.features[3]=std::log2(1.0+count*(2000000000.0/short_ns))-
      std::log2(1.0+std::max<uint64_t>(1,r.future_access_threshold_at_prediction));
    q.push_back(now);
  }
  std::cout<<"short_ns="<<short_ns<<" source_2s_count_mismatches="<<mismatch<<std::endl;
  return mismatch;
}
void test_short_windows() {
  for (uint64_t width : {500000000ULL,1000000000ULL,2000000000ULL}) {
    HpReplayTrace t;
    for(uint64_t when : std::array<uint64_t,4>{1, width, width+1, width+1}) {
      HpTraceRecord r{};r.io_sequence=t.records.size()+1;r.object_key_hash=7;
      r.prediction_time_ns=when;r.future_access_threshold_at_prediction=2;
      r.features[0]=11;r.features[1]=12;r.features[2]=13;
      r.features[4]=t.records.empty()?0.0:std::log2(2.0);t.records.push_back(r);
    }
    rebuild_short_features(t,width);
    require(std::abs(t.records[2].features[4]-std::log2(2.0))<1e-12,"short window left boundary/current access excluded");
    require(t.records[2].features[4]==t.records[3].features[4],"same timestamp is outside strict past");
    double expected=std::log2(1.0+2000000000.0/width)-std::log2(3.0);
    require(std::abs(t.records[2].features[3]-expected)<1e-12,"projection must target future 2 seconds");
    require(t.records[2].features[0]==11 && t.records[2].features[1]==12 && t.records[2].features[2]==13,"preserve historical and heat features");
  }
  self_test();
  ADWIN<5> detector(.001);bool detected=false;
  for(int i=0;i<10000;++i){detector.update(i<5000?0:1);detected|=detector.drift_detected;}
  require(detected,"ADWIN must detect sustained synthetic error shift");
  std::cout<<"PASS: short 0.5/1/2s boundaries, projection, retained features and ADWIN shift\n";
}
#ifndef HP_SHORT_DRIFT_REPLAY_NO_MAIN
int main(int argc,char** argv) {
 try {
  if(argc==2 && std::string(argv[1])=="--self-test"){test_short_windows();return 0;}
  require(argc==5,"usage: hp_short_drift_replay TRACE.bin SHORT_MS disabled|baseline NEW_OUTPUT.tsv");
  auto ms=std::stoull(argv[2]);require(ms==500||ms==1000||ms==2000,"unsupported short window");
  auto profile=parse_hp_replay_adaptation_profile(argv[3]);require(profile!=HpReplayAdaptationProfile::conservative,"unsupported profile");
  require(!std::filesystem::exists(argv[4]),"refuse existing output");
  auto trace=read_hp_trace(argv[1]);relabel(trace,2000000000ULL);
  auto mismatch=rebuild_short_features(trace,ms*1000000ULL);
  require(ms!=2000 || mismatch==0,"source 2s short count reconstruction mismatch");
  HpReplayOptions options;options.adaptation_profile=profile;
  auto result=replay_hp_trace(trace,options);require(result.trained_sample_count==trace.records.size(),"training sample loss");
  std::ofstream out(argv[4]);write_hp_replay_tsv(out,trace,result);out.close();require(out.good(),"output write failed");
  auto a=result.adaptation_stats;
  if(profile==HpReplayAdaptationProfile::disabled) require(a.drift_count==0 && a.warning_count==0,"disabled model emitted drift");
  std::cout<<"completed records="<<trace.records.size()<<" trained="<<result.trained_sample_count<<" snapshots="<<result.snapshot_publish_count
    <<" warning="<<a.warning_count<<" drift="<<a.drift_count<<" promotions="<<a.background_promotion_count<<" discards="<<a.background_discard_count
    <<" background_updates="<<a.background_training_update_count<<" active_background="<<a.active_background_count<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}

#endif
