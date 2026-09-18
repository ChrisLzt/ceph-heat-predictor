#define HP_HISTORY_HEAT_REPLAY_NO_MAIN
#include "hp_history_heat_replay.cc"

void mask_short_features(HpReplayTrace& trace) {
  for(auto& r:trace.records) { r.features[3]=0.0; r.features[4]=0.0; }
}
void test_short_ablation() {
  HpReplayTrace trace;
  HpTraceRecord r{};
  r.io_sequence=7;r.actual_label=1;r.prediction_time_ns=123;
  for(int i=0;i<5;++i)r.features[i]=1.25*(i+1);
  trace.records.push_back(r);
  mask_short_features(trace);
  const auto& got=trace.records[0];
  require(got.features[3]==0 && got.features[4]==0,"short features must be constant zero");
  for(int i=0;i<3;++i)require(got.features[i]==r.features[i],"non-short feature changed");
  require(got.io_sequence==7 && got.actual_label==1 && got.prediction_time_ns==123,"label or sample identity changed");
  test_history_heat();
  std::cout<<"PASS: short feature masking preserves other inputs and labels\n";
}
int main(int argc,char** argv) {
 try {
  if(argc==2 && std::string(argv[1])=="--self-test"){test_short_ablation();return 0;}
  require(argc==4,"usage: hp_short_ablation TRACE.bin full|no_short NEW_OUTPUT.tsv");
  std::string arm=argv[2];require(arm=="full" || arm=="no_short","unknown arm");
  require(!std::filesystem::exists(argv[3]),"refuse existing output");
  auto trace=read_hp_trace(argv[1]);
  prepare_history_heat(trace,true);rebuild_short_features(trace,1000000000ULL);
  if(arm=="no_short")mask_short_features(trace);
  std::cout<<"source_config_hash="<<trace.header.config_hash<<" replay_config_hash="<<hp_trace_config_hash()<<std::endl;
  HpReplayOptions options;
  // Input is an archived 10s trace; labels and features above were explicitly rebuilt.
  options.require_matching_config=false;
  options.adaptation_profile=HpReplayAdaptationProfile::disabled;
  auto result=replay_hp_trace(trace,options);
  require(result.trained_sample_count==trace.records.size(),"training sample loss");
  std::ofstream out(argv[3]);write_hp_replay_tsv(out,trace,result);out.close();require(out.good(),"output write failed");
  auto a=result.adaptation_stats;require(a.drift_count==0 && a.warning_count==0,"disabled detector emitted events");
  std::cout<<"completed records="<<trace.records.size()<<" trained="<<result.trained_sample_count<<" snapshots="<<result.snapshot_publish_count
    <<" warning="<<a.warning_count<<" drift="<<a.drift_count<<" promotions="<<a.background_promotion_count<<" discards="<<a.background_discard_count
    <<" background_updates="<<a.background_training_update_count<<" active_background="<<a.active_background_count<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
