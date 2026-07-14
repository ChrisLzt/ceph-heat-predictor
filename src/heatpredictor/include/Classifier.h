# ifndef CLASSIFIER_H
# define CLASSIFIER_H

# include <vector>
# include <algorithm>
# include <memory>

class Classifier {
public:
    virtual void learn_one(const std::vector<double>& x, int y, double w=1.0) = 0;
    virtual std::vector<double> predict_proba_one(const std::vector<double>& x) = 0;
    virtual void predict_proba_one_into(
            const std::vector<double>& x,
            std::vector<double>& proba) {
        proba = predict_proba_one(x);
    }
    virtual std::unique_ptr<Classifier> clone_for_prediction() const = 0;
    virtual int predict_one(const std::vector<double>& x) {
        thread_local std::vector<double> proba;
        predict_proba_one_into(x, proba);
        return std::distance(proba.begin(), std::max_element(proba.begin(), proba.end()));
    }
    virtual ~Classifier() = default;
};

# endif
