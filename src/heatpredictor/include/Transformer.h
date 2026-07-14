# ifndef TRANSFORMER_H
# define TRANSFORMER_H

# include <vector>
# include <memory>

class Transformer {
public:
    virtual void learn_one(const std::vector<double>& x, int y) = 0;
    virtual std::vector<double> transform_one(const std::vector<double>& x) = 0;
    virtual void transform_one_into(
            const std::vector<double>& x,
            std::vector<double>& transformed) {
        transformed = transform_one(x);
    }
    virtual std::unique_ptr<Transformer> clone() const = 0;
    virtual ~Transformer() = default;
};

# endif
