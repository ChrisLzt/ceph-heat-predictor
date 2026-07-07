# ifndef STANDARD_SCALER_H
# define STANDARD_SCALER_H

# include "Transformer.h"

# include <cmath>
# include <vector>
# include <array>
# include <cstdint>
# include <memory>

// always with_std
template <int num_features>
class StandardScaler : public Transformer {
private:
    std::array<uint64_t, num_features> counts{};
    std::array<double, num_features> means{};
    std::array<double, num_features> m2s{};
public:
    void learn_one(const std::vector<double>& x, int y) override {
        (void) y; // unused
        for (size_t i = 0; i < x.size(); i++) {
            counts[i] += 1;
            double delta = x[i] - means[i];
            means[i] += delta / static_cast<double>(counts[i]);
            double delta2 = x[i] - means[i];
            m2s[i] += delta * delta2;
        }
    }
    std::vector<double> transform_one(const std::vector<double>& x) override {
        std::vector<double> res(x.size(), 0.0);
        for (size_t i = 0; i < x.size(); i++) {
            double var = counts[i] > 0 ? m2s[i] / static_cast<double>(counts[i]) : 0.0;
            res[i] = var > 0.0 ? (x[i] - means[i]) / std::sqrt(var) : 0.0;
        }
        return res;
    }
    std::unique_ptr<Transformer> clone() const override {
        return std::unique_ptr<Transformer>(new StandardScaler(*this));
    }
};

# endif
