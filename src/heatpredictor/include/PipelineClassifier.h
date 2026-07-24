# ifndef PIPELINE_CLASSIFIER_H
# define PIPELINE_CLASSIFIER_H

# include "Classifier.h"
# include "Transformer.h"
# include <memory>

class PipelineClassifier : public Classifier {
private:
    Transformer* transformer;
    Classifier* classifier;
public:
    PipelineClassifier(const PipelineClassifier& other) = delete;
    PipelineClassifier& operator=(const PipelineClassifier& other) = delete;
    PipelineClassifier(Transformer* transformer, Classifier* classifier) : 
        transformer(transformer), classifier(classifier) {}
    ~PipelineClassifier() {
        delete transformer;
        delete classifier;
    }
    void learn_one(const std::vector<double>& x, int y, double w=1.0) override {
        transformer->learn_one(x, y);
        thread_local std::vector<double> transformed;
        transformer->transform_one_into(x, transformed);
        classifier->learn_one(transformed, y, w);
    }
    std::vector<double> predict_proba_one(const std::vector<double>& x) override {
        std::vector<double> proba;
        predict_proba_one_into(x, proba);
        return proba;
    }
    void predict_proba_one_into(
            const std::vector<double>& x,
            std::vector<double>& proba) override {
        thread_local std::vector<double> transformed;
        transformer->transform_one_into(x, transformed);
        classifier->predict_proba_one_into(transformed, proba);
    }
    std::unique_ptr<Classifier> clone_for_prediction() const override {
        std::unique_ptr<Transformer> transformer_copy = transformer->clone();
        std::unique_ptr<Classifier> classifier_copy =
            classifier->clone_for_prediction();
        auto copy = std::unique_ptr<PipelineClassifier>(
            new PipelineClassifier(
                transformer_copy.get(), classifier_copy.get()));
        transformer_copy.release();
        classifier_copy.release();
        return copy;
    }
};

# endif
