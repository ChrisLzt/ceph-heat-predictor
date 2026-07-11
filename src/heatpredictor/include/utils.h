# ifndef UTILS_H
# define UTILS_H

# include <cmath>
# include <array>
# include <limits>
# include <vector>
# include <unordered_map>
# include <random>
# include <algorithm>
# include <stdexcept>
# include "GaussianSplitter.h"

inline double sum(const std::unordered_map<int, double>& x) {
    double res = 0.0;
    for (const auto& kv : x) {
        res += kv.second;
    }
    return res;
}

inline double max_value(const std::unordered_map<int, double>& x) {
    if (x.empty()) return 0.0;
    return std::max_element(x.begin(), x.end(), [](const auto& a, const auto& b) { return a.second < b.second;})->second;
}

inline int max_index(const std::vector<double>& x) {
    return std::distance(x.begin(), std::max_element(x.begin(), x.end()));
}

inline int poisson(int lambda, std::default_random_engine* gen) {
    thread_local int cached_lambda = -1;
    thread_local std::poisson_distribution<int> dis(1);

    if (cached_lambda != lambda) {
        dis = std::poisson_distribution<int>(lambda);
        cached_lambda = lambda;
    }
    return dis(*gen);
}

template <int num_features, int num_labels>
void do_naive_bayes_prediction(std::vector<double>& votes, const std::vector<double>& x,
    const std::unordered_map<int, double>& observed_class_distribution,
    const std::array<GaussianSplitter<num_features, num_labels>*, num_features>& splitters){
    if (x.size() != num_features || splitters.size() != num_features) {
        throw std::invalid_argument("invalid Naive Bayes feature count");
    }
    votes.assign(num_labels, 0.0);
    double total_weight = sum(observed_class_distribution);
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        return;
    }

    auto use_class_priors = [&]() {
        votes.assign(num_labels, 0.0);
        for (const auto& [label, weight] : observed_class_distribution) {
            if (label >= 0 && label < num_labels && weight > 0.0 &&
                std::isfinite(weight)) {
                votes[label] = weight / total_weight;
            }
        }
    };

    const double min_density = std::numeric_limits<double>::min();
    std::vector<double> log_votes(
        num_labels, -std::numeric_limits<double>::infinity());
    for (const auto& kv : observed_class_distribution) {
        if (kv.first < 0 || kv.first >= num_labels ||
            !(kv.second > 0.0) || !std::isfinite(kv.second)) {
            continue;
        }

        double log_vote = std::log(kv.second / total_weight);
        for (size_t i=0;i<splitters.size();i++) {
            if (splitters[i] == nullptr) {
                continue;
            }
            double tmp = splitters[i]->cond_proba(x[i], kv.first);
            if (!(tmp > 0.0) || !std::isfinite(tmp)) {
                tmp = min_density;
            }
            log_vote += std::log(tmp);
        }
        log_votes[kv.first] = log_vote;
    }

    double max_ll = *std::max_element(log_votes.begin(), log_votes.end());
    if (!std::isfinite(max_ll)) {
        use_class_priors();
        return;
    }

    double lse = 0.0;
    for (double d : log_votes) {
        if (std::isfinite(d)) {
            lse += std::exp(d - max_ll);
        }
    }
    if (!(lse > 0.0) || !std::isfinite(lse)) {
        use_class_priors();
        return;
    }
    const double log_normalizer = max_ll + std::log(lse);

    for (size_t i=0;i<votes.size();i++) {
        votes[i] = std::isfinite(log_votes[i])
            ? std::exp(log_votes[i] - log_normalizer)
            : 0.0;
    }
}

inline void normalize_values_in_dict(std::vector<double>& res_dict, const std::unordered_map<int, double>& dictionary, 
    double factor=0.0, bool raise_error=false){
    if (factor == 0.0) {
        factor = sum(dictionary);
    }
    if (factor == 0.0) {
        if (raise_error)
            throw std::runtime_error("Can not normalize, normalization factor is 0");
        for (auto& kv : dictionary) {
            res_dict[kv.first] = kv.second;
        }
    } else {
        for (auto& kv : dictionary) {
            res_dict[kv.first] = kv.second / factor;
        }
    }
}

# endif
