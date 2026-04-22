# ifndef METRICS_H
# define METRICS_H

template <int num_labels>
class ConfusionMatrix {
private:
    uint64_t data[num_labels][num_labels] = {0};
    // uint64_t sum_row[num_labels] = {0};
    // uint64_t sum_col[num_labels] = {0};
    // int n_samples = 0;
    std::mutex mtx;
public:
    uint64_t total_weight = 0;
    void update(int y_true, int y_pred, uint64_t w = 1) {
        std::lock_guard<std::mutex> lock(mtx);
        // n_samples++;
        data[y_true][y_pred] += w;
        total_weight += w;
        // sum_row[y_true] += w;
        // sum_col[y_pred] += w;
    }
    uint64_t total_true_positives() {
        std::lock_guard<std::mutex> lock(mtx);
        uint64_t total = 0;
        for (int i = 0; i < num_labels; i++) {
            total += data[i][i];
        }
        return total;
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < num_labels; ++i) {
            for (int j = 0; j < num_labels; ++j) {
                data[i][j] = 0;
            }
        }
        total_weight = 0;
    }
};

template <int num_labels>
class Accuracy {
private:
    ConfusionMatrix<num_labels> cm;
public:
    void update(int y_true, int y_pred, uint64_t w=1) {
        cm.update(y_true, y_pred, w);
    }
    uint64_t get_total_weight() {
        return cm.total_weight;
    }
    double get_accuracy() {
        if (cm.total_weight > 0) {
            return (double) cm.total_true_positives() / cm.total_weight;
        } else {
            return 0;
        }
    }
    void clear() {
        cm.clear();
    }
};

# endif