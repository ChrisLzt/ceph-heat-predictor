#define HP_SHORT_DRIFT_REPLAY_NO_MAIN
#include "hp_short_drift_replay.cc"

void prepare_history_heat(HpReplayTrace& trace, bool verify_source = false) {
 std::unordered_map<uint64_t,std::pair<double,uint64_t>> heat;
 relabel(trace,2000000000ULL,[&](HpTraceRecord& r,const PredictionSample& sample){
   auto [it,inserted]=heat.emplace(r.object_key_hash,std::make_pair(0.0,r.prediction_time_ns));
   require(r.prediction_time_ns>=it->second.second,"heat events out of order");
   auto elapsed=r.prediction_time_ns-it->second.second;
   double value=it->second.first*std::exp2(-static_cast<double>(elapsed)/1000000000.0)+HP_HEAT_INCREMENT;
   it->second={value,r.prediction_time_ns};
   if(verify_source) require(std::abs(std::log2(1.0+sample.past_window_access_count)-r.features[4])<1e-9,"history2s must match source short2s counts");
   r.heat_after_current_access=value;
   r.future_access_threshold_at_prediction=sample.future_access_threshold_at_prediction;
   r.past_window_access_count=sample.past_window_access_count;
   r.tracked_access_count_after_current_access=sample.tracked_access_count_after_current_access;
   r.time_since_previous_access_ns=sample.time_since_previous_access_ns;
   r.features[0]=std::log2(1.0+sample.past_window_access_count)-std::log2(1.0+std::max<uint64_t>(1,sample.future_access_threshold_at_prediction));
   r.features[1]=hp_previous_access_interval_encoded(sample.tracked_access_count_after_current_access,sample.time_since_previous_access_ns);
   r.features[2]=std::log2(1.0+value);
 });
 require(heat.size()<=HP_LRU_CAPACITY,"experiment requires no object state eviction");
 std::cout<<"history_ns=2000000000 heat_half_life_ns=1000000000 objects="<<heat.size()<<std::endl;
}
void test_history_heat() {
 HpReplayTrace trace;
 for(uint64_t t:std::array<uint64_t,3>{1,1000000001ULL,2000000001ULL}){
  HpTraceRecord r{};r.io_sequence=trace.records.size()+1;r.object_key_hash=7;r.prediction_time_ns=t;
  r.hot_predict_threshold=.5;trace.records.push_back(r);
 }
 prepare_history_heat(trace);
 require(trace.records[1].heat_after_current_access==150.0,"1s half-life: second access heat must be150");
 require(trace.records[2].heat_after_current_access==175.0,"two half-life steps must accumulate to175");
 require(trace.records[2].past_window_access_count==1,"strict past2s must expire left boundary");
 require(trace.records[2].future_access_threshold_at_prediction==1,"history2s sparse context threshold");
 require(std::abs(trace.records[2].features[0])<1e-12,"history2s count margin");
 require(std::abs(trace.records[2].features[2]-std::log2(176.0))<1e-12,"heat feature matches new heat");
 require(trace.records.back().actual_label==0,"future2s tail still drained");
 test_short_windows();
 ADWIN<5> original(.001),conservative(.0001);int first_original=-1,first_conservative=-1;
 for(int i=0;i<10000;++i){
  int error=i<5000?0:1;original.update(error);conservative.update(error);
  if(original.drift_detected && first_original<0)first_original=i;
  if(conservative.drift_detected && first_conservative<0)first_conservative=i;
 }
 require(first_original>=0 && first_conservative>=first_original,"conservative ADWIN should detect sustained fixture no earlier");
 std::cout<<"ADWIN first shift original="<<first_original<<" conservative="<<first_conservative<<std::endl;
 std::cout<<"PASS: historical2s boundaries and heat half-life1s\n";
}
#ifndef HP_HISTORY_HEAT_REPLAY_NO_MAIN
int main(int argc,char** argv){
 try{
  if(argc==2 && std::string(argv[1])=="--self-test"){test_history_heat();return 0;}
  require(argc==4,"usage: hp_history_heat_replay TRACE.bin disabled|baseline|conservative NEW_OUTPUT.tsv");
  require(!std::filesystem::exists(argv[3]),"refuse existing output");
  auto profile=parse_hp_replay_adaptation_profile(argv[2]);auto trace=read_hp_trace(argv[1]);
  prepare_history_heat(trace,true);rebuild_short_features(trace,1000000000ULL);
  HpReplayOptions options;options.adaptation_profile=profile;
  auto result=replay_hp_trace(trace,options);require(result.trained_sample_count==trace.records.size(),"training sample loss");
  std::ofstream out(argv[3]);write_hp_replay_tsv(out,trace,result);out.close();require(out.good(),"output write failed");
  auto a=result.adaptation_stats;
  if(profile==HpReplayAdaptationProfile::disabled)require(a.drift_count==0 && a.warning_count==0,"disabled detector emitted events");
  std::cout<<"completed records="<<trace.records.size()<<" trained="<<result.trained_sample_count<<" snapshots="<<result.snapshot_publish_count
    <<" warning="<<a.warning_count<<" drift="<<a.drift_count<<" promotions="<<a.background_promotion_count<<" discards="<<a.background_discard_count
    <<" background_updates="<<a.background_training_update_count<<" active_background="<<a.active_background_count<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}

#endif
