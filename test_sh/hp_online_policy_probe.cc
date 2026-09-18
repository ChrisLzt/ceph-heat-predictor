#define main existing_algorithm_probe_main
#include "hp_algorithm_probe.cc"
#undef main

struct MajorityPolicyLeaf : LeafNaiveBayesAdaptive<5, 2> {
  MajorityPolicyLeaf() : LeafNaiveBayesAdaptive<5, 2>(0) {}
  void favor_bayes() { _nb_correct_weight = 100; _mc_correct_weight = 0; }
};

void test_fixed_majority_policy() {
  MajorityPolicyLeaf leaf;
  for (int i = 0; i < 10; ++i) {
    const int label = i >= 8;
    leaf.learn_one(std::vector<double>(5, label * 10.0 + i * .01), label);
  }
  leaf.favor_bayes();
  std::vector<double> proba(2);
  leaf.prediction(proba, std::vector<double>(5, 10.085));
  require_close(proba[1], .2, "production leaf must use local weighted majority even when Bayes won historically");
}

void test_untrained_history_guard() {
  HeatPredictor predictor;
  predictor.set_enabled(true);
  uint64_t sequence = 0;
  require(predictor.predict(26, 111, 222, &sequence) == 0,
          "first object access is past-cold");
  require(predictor.predict(26, 111, 222, &sequence) == 1,
          "untrained repeated object must use past-hot rule");
}

void test_warmup_tracks_published_model_and_reset() {
  HeatPredictor predictor;
  predictor.set_enabled(true);
  predictor.train_model = std::make_shared<BlockingTrainingClassifier>();
  predictor.trained_sample_count.store(2999);
  predictor.publish_prediction_snapshot(predictor.clone_train_model_for_prediction());
  uint64_t sequence = 0;
  require(predictor.predict(26, 333, 444, &sequence) == 0, "new object is cold under guard");
  predictor.trained_sample_count.store(3000);
  require(predictor.predict(26, 333, 444, &sequence) == 1,
          "unpublished training must not end guard on snapshot 2999");
  const auto old_snapshot = predictor.get_prediction_snapshot();
  predictor.publish_prediction_snapshot(predictor.clone_train_model_for_prediction());
  require(old_snapshot->trained_samples == 2999,
          "held snapshot count must stay immutable across publication");
  require(predictor.predict(26, 333, 444, &sequence) == 0,
          "snapshot 3000 must use model even when history says hot");
  require(predictor.status().warmup_prediction_count == 2,
          "guard counter must count only protected outputs");
  predictor.reset();
  const auto status = predictor.status();
  require(status.trained_sample_count == 0 &&
              status.snapshot_trained_sample_count == 0 &&
              status.warmup_prediction_count == 0,
          "reset must clear training and guard state atomically");
  require(predictor.predict(26, 333, 444, &sequence) == 0 &&
              predictor.predict(26, 333, 444, &sequence) == 1,
          "reset must restore history guard");
}

void test_snapshot_sample_or_time_boundary() {
  HeatPredictor predictor;
  predictor.last_snapshot_publish_time_ns = 1;
  for (uint64_t i = 1; i < HP_SNAPSHOT_PUBLISH_SAMPLE_INTERVAL; ++i)
    require(!predictor.record_model_update_batch(1), "must wait until sample boundary");
  require(predictor.record_model_update_batch(1), "sample boundary must publish");
  require(!predictor.record_model_update_batch(HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS),
          "time boundary excludes one ns before interval");
  require(predictor.record_model_update_batch(1 + HP_SNAPSHOT_PUBLISH_MAX_INTERVAL_NS),
          "elapsed time must publish without enough new samples");
}

int main() {
  test_fixed_majority_policy();
  test_untrained_history_guard();
  test_warmup_tracks_published_model_and_reset();
  test_snapshot_sample_or_time_boundary();
  std::cout << "PASS online prediction policy: leaf, warmup, publication, reset" << std::endl;
}
