# ifndef ARFCLASSIFIER_H
# define ARFCLASSIFIER_H

# include <iostream>
# include <vector>
# include <cmath>
# include <random>
# include <memory>
# include <stdexcept>

# include "drift/DetectorConcept.h"
# include "Classifier.h"
# include "HoeffdingTreeClassifier.tpp"
# include "Metrics.h"
# include "utils.h"
# include "drift/ADWIN.h"

template <int num_features, int num_labels>
class BaseTreeClassifier : public HoeffdingTreeClassifier<num_features, num_labels> {
protected:
    int max_features;
    int seed;
    std::default_random_engine leaf_seed_rng;
    static constexpr int normalize_max_features(int value) {
        return value < 1 ? 1 : (value > num_features ? num_features : value);
    }
public:
    BaseTreeClassifier(int max_features=2, int grace_period = 200,
        double delta = 1e-7, double tau = 0.05,
        double max_share_to_split = 0.99,
        double min_branch_fraction = 0.01,
        int seed = 0) :
        HoeffdingTreeClassifier<num_features, num_labels>(grace_period, delta, tau, max_share_to_split, min_branch_fraction),
        max_features(normalize_max_features(max_features)),
        seed(seed),
        leaf_seed_rng(seed) {}
    std::unique_ptr<Classifier> clone_for_prediction() const override {
        auto copy = std::unique_ptr<BaseTreeClassifier>(
            new BaseTreeClassifier(
                max_features, this->grace_period, this->delta, this->tau,
                this->max_share_to_split, this->min_branch_fraction, seed));
        copy->leaf_seed_rng = leaf_seed_rng;
        this->copy_prediction_state_to(copy.get());
        return copy;
    }
    BranchOrLeaf<num_features, num_labels>* _new_leaf(
        LeafNaiveBayesAdaptive<num_features, num_labels>* parent=nullptr) override {
        int depth;
        if (parent == nullptr) {
            depth = 0;
        } else {
            depth = parent->depth + 1;
        }
        return new RandomLeafNaiveBayesAdaptive<num_features, num_labels>(
            depth, max_features, static_cast<int>(leaf_seed_rng()));
    }
};

// max features is always sqrt
// warning_detection & drift_detection always on
template <int num_features, int num_labels,
    typename WarningDetectorFactory=DetectorFactory<ADWIN<5>, 10>,
    typename DriftDetectorFactory=DetectorFactory<ADWIN<5>, 1> >
class ARFClassifier : public Classifier {
    static_assert(IsDetectorFactory_v<WarningDetectorFactory>, "Invalid Warning Factory");
    static_assert(IsDetectorFactory_v<DriftDetectorFactory>, "Invalid Drift Factory");
protected:
    struct PredictionOnlyTag {};
    std::vector<Classifier*> models;
    std::vector<BaseTreeClassifier<num_features, num_labels>*> _background;
    std::vector<typename DriftDetectorFactory::DetectorType> _drift_detectors;
    std::vector<typename WarningDetectorFactory::DetectorType> _warning_detectors;
    std::vector<Accuracy<num_labels> > _metrics;
    std::vector<double> _prediction_weights;
    bool _prediction_weights_valid = false;
    std::vector<int> _drift_tracker;
    std::vector<int> _warning_tracker;
    std::default_random_engine _rng;
    int n_models;
    int max_features;
    int seed;
    int grace_period;
    int lambda_value;
    double delta;
    double tau;
    double max_share_to_split;
    double min_branch_fraction;
    void validate_parameters() const {
        if (n_models <= 0) {
            throw std::invalid_argument("ARF n_models must be positive");
        }
        if (grace_period <= 0 || lambda_value <= 0) {
            throw std::invalid_argument(
                "ARF grace_period and lambda_value must be positive");
        }
        if (!(delta > 0.0 && delta < 1.0) || tau < 0.0 ||
            !(max_share_to_split > 0.0 && max_share_to_split <= 1.0) ||
            !(min_branch_fraction >= 0.0 && min_branch_fraction < 0.5)) {
            throw std::invalid_argument("invalid ARF tree probability parameter");
        }
    }
    void initialize_training_state() {
        _metrics = std::vector<Accuracy<num_labels> >(n_models);
        _background = std::vector<BaseTreeClassifier<num_features, num_labels>*>(n_models, nullptr);
        _drift_detectors = std::vector<typename DriftDetectorFactory::DetectorType>(n_models);
        _warning_detectors = std::vector<typename WarningDetectorFactory::DetectorType>(n_models);
        for (int i=0;i<n_models;i++) {
            _drift_detectors[i] = DriftDetectorFactory::create();
            _warning_detectors[i] = WarningDetectorFactory::create();
        }
        _drift_tracker = std::vector<int>(n_models, 0);
        _warning_tracker = std::vector<int>(n_models, 0);
    }
    static constexpr int normalize_max_features(int value) {
        return value < 1 ? 1 : (value > num_features ? num_features : value);
    }
    int model_seed(int idx) const {
        return seed + 1000003 * (idx + 1);
    }
    void refresh_prediction_weights() {
        if (_prediction_weights_valid) {
            return;
        }
        _prediction_weights.resize(n_models);
        for (int i = 0; i < n_models; ++i) {
            const double metric = _metrics[i].get_balanced_accuracy();
            _prediction_weights[i] = metric > 0.0 ? metric : 1.0;
        }
        _prediction_weights_valid = true;
    }
    void _init_ensemble() {
        for (Classifier* model : models) {
            if (model != nullptr)
                delete model;
        }
        models.clear();
        models.reserve(n_models);
        for (int i = 0; i < n_models; i++) {
            models.push_back(new BaseTreeClassifier<num_features, num_labels>
                (max_features, grace_period, delta, tau, max_share_to_split,
                 min_branch_fraction, model_seed(i)));
        }
    }
    inline int _drift_detector_input(int y_true, int y_pred) {
        return (y_true == y_pred) ? 0 : 1;
    }
    ARFClassifier(PredictionOnlyTag,
        int n_models, int max_features, int seed, int grace_period,
        int lambda_value, double delta, double tau,
        double max_share_to_split, double min_branch_fraction)
        : _rng(seed), n_models(n_models),
        max_features(normalize_max_features(max_features)), seed(seed),
        grace_period(grace_period), lambda_value(lambda_value), delta(delta),
        tau(tau), max_share_to_split(max_share_to_split),
        min_branch_fraction(min_branch_fraction) {
        validate_parameters();
        _prediction_weights.assign(n_models, 1.0);
        _prediction_weights_valid = true;
    }
public:
    ARFClassifier(int n_models=10, int max_features=(int)(std::sqrt(num_features)),
        int seed=1037, int grace_period=100, int lambda_value=4,
        double delta=0.001, double tau = 0.05,
        double max_share_to_split = 0.99, double min_branch_fraction = 0.01)
        : _rng(seed), n_models(n_models),
        max_features(normalize_max_features(max_features)), seed(seed),
        grace_period(grace_period), lambda_value(lambda_value), delta(delta), tau(tau),
        max_share_to_split(max_share_to_split), min_branch_fraction(min_branch_fraction) {
        validate_parameters();
        _prediction_weights.assign(n_models, 1.0);
        initialize_training_state();
    }
    ~ARFClassifier() {
        for (Classifier* model : models) {
            if (model != nullptr)
                delete model;
        }
        for (BaseTreeClassifier<num_features, num_labels>* model : _background) {
            if (model != nullptr)
                delete model;
        }
    }
    std::unique_ptr<Classifier> clone_for_prediction() const override {
        auto copy = std::unique_ptr<ARFClassifier>(
            new ARFClassifier(
                PredictionOnlyTag{},
                n_models, max_features, seed, grace_period, lambda_value,
                delta, tau, max_share_to_split, min_branch_fraction));
        if (_prediction_weights_valid) {
            copy->_prediction_weights = _prediction_weights;
        } else {
            copy->_prediction_weights.resize(n_models);
            for (int i = 0; i < n_models; ++i) {
                const double metric = _metrics[i].get_balanced_accuracy();
                copy->_prediction_weights[i] = metric > 0.0 ? metric : 1.0;
            }
        }
        copy->_prediction_weights_valid = true;
        copy->models.clear();
        copy->models.reserve(n_models);
        if (models.empty()) {
            copy->_init_ensemble();
        } else {
            for (Classifier* model : models) {
                copy->models.push_back(
                    model != nullptr
                        ? model->clone_for_prediction().release()
                        : nullptr);
            }
        }
        return copy;
    }
    void learn_one(const std::vector<double>& x, int y, double w=1.0) {
        if (x.size() != num_features || y < 0 || y >= num_labels ||
            !(w > 0.0) || !std::isfinite(w)) {
            throw std::invalid_argument("invalid ARF training sample");
        }
        if (models.size() == 0) {
            _init_ensemble();
        }
        _prediction_weights_valid = false;
        for (int i=0;i<n_models;i++) {
            Classifier* model = models[i];
            int y_pred = model->predict_one(x);
            _metrics[i].update(y, y_pred);
            int k = poisson(lambda_value, &_rng);
            if (k > 0) {
                double sample_weight = w * k;

                if (_background[i] != nullptr) {
                    _background[i]->learn_one(x, y, sample_weight);
                }
                model->learn_one(x, y, sample_weight);

                int drift_input = _drift_detector_input(y, y_pred);
                _warning_detectors[i].update(drift_input);
                if (_warning_detectors[i].drift_detected) {
                    if (_background[i]) {
                        delete _background[i];
                    }
                    _background[i] = new BaseTreeClassifier<num_features, num_labels>
                        (max_features, grace_period, delta, tau,
                         max_share_to_split, min_branch_fraction,
                         model_seed(i) + _warning_tracker[i] + 1);
                    _warning_detectors[i] = WarningDetectorFactory::create();
                    _warning_tracker[i]++;
                }

                _drift_detectors[i].update(drift_input);
                if (_drift_detectors[i].drift_detected) {
                    if (_background[i] != nullptr) {
                        if (models[i]) {
                            delete models[i];
                        }
                        models[i] = _background[i];
                        _background[i] = nullptr;
                        _warning_detectors[i] = WarningDetectorFactory::create();
                        _drift_detectors[i] = DriftDetectorFactory::create();
                        _metrics[i].clear();
                    } else {
                        if (models[i]) {
                            delete models[i];
                        }
                        models[i] = new BaseTreeClassifier<num_features, num_labels>
                            (max_features, grace_period, delta, tau,
                             max_share_to_split, min_branch_fraction,
                             model_seed(i) + _drift_tracker[i] + 1);
                        _drift_detectors[i] = DriftDetectorFactory::create();
                        _metrics[i].clear();
                    }
                    _drift_tracker[i]++;
                }
            }
        }
    }
    std::vector<double> predict_proba_one(const std::vector<double>& x) override {
        std::vector<double> proba;
        predict_proba_one_into(x, proba);
        return proba;
    }
    void predict_proba_one_into(
            const std::vector<double>& x,
            std::vector<double>& proba) override {
        if (x.size() != num_features) {
            throw std::invalid_argument("invalid ARF feature count");
        }
        proba.assign(num_labels, 0.0);
        if (models.size() == 0) {
            _init_ensemble();
        } else {
            refresh_prediction_weights();
            thread_local std::vector<double> tree_proba;
            for (int i=0;i<n_models;i++) {
                Classifier* model = models[i];
                model->predict_proba_one_into(x, tree_proba);
                for (int j=0;j<num_labels;j++) {
                    proba[j] += tree_proba[j] * _prediction_weights[i];
                }
            }
            double total = std::accumulate(proba.begin(), proba.end(), 0.0);
            for (int i=0;i<num_labels;i++) {
                if (total > 0.0) {
                    proba[i] /= total;
                } else {
                    proba[i] = 0.0;
                }
            }
        }
    }
};

# endif
