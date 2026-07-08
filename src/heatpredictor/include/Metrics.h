# ifndef METRICS_H
# define METRICS_H

#include <cstdint>
#include <mutex>

template <int num_labels>
class ConfusionMatrix {
private:
    uint64_t data[num_labels][num_labels] = {0};
    // uint64_t sum_row[num_labels] = {0};
    // uint64_t sum_col[num_labels] = {0};
    // int n_samples = 0;
    mutable std::mutex mtx;
    uint64_t total_weight = 0;
public:
    ConfusionMatrix() = default;
    ConfusionMatrix(const ConfusionMatrix& other) {
        std::lock_guard<std::mutex> lock(other.mtx);
        for (int i = 0; i < num_labels; ++i) {
            for (int j = 0; j < num_labels; ++j) {
                data[i][j] = other.data[i][j];
            }
        }
        total_weight = other.total_weight;
    }
    ConfusionMatrix& operator=(const ConfusionMatrix& other) {
        if (this == &other) {
            return *this;
        }
        std::scoped_lock lock(mtx, other.mtx);
        for (int i = 0; i < num_labels; ++i) {
            for (int j = 0; j < num_labels; ++j) {
                data[i][j] = other.data[i][j];
            }
        }
        total_weight = other.total_weight;
        return *this;
    }
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
    uint64_t get(int row, int col) {
        std::lock_guard<std::mutex> lock(mtx);
        return data[row][col];
    }
    uint64_t get_total_weight() {
        std::lock_guard<std::mutex> lock(mtx);
        return total_weight;
    }
    double get_accuracy() {
        std::lock_guard<std::mutex> lock(mtx);
        if (total_weight == 0) return 0;

        uint64_t total = 0;
        for (int i = 0; i < num_labels; i++) {
            total += data[i][i];
        }
        return static_cast<double>(total) / total_weight;
    }
    double get_balanced_accuracy() {
        std::lock_guard<std::mutex> lock(mtx);
        if (total_weight == 0) return 0;

        double recall_sum = 0.0;
        for (int label = 0; label < num_labels; ++label) {
            uint64_t actual = 0;
            for (int pred = 0; pred < num_labels; ++pred) {
                actual += data[label][pred];
            }
            recall_sum += actual > 0
                ? static_cast<double>(data[label][label]) / actual
                : 0.0;
        }
        return recall_sum / num_labels;
    }
    double get_label_precision(int label) {
        std::lock_guard<std::mutex> lock(mtx);
        uint64_t predicted = 0;
        for (int i = 0; i < num_labels; i++) {
            predicted += data[i][label];
        }
        return predicted > 0 ? static_cast<double>(data[label][label]) / predicted : 0;
    }
    double get_label_recall(int label) {
        std::lock_guard<std::mutex> lock(mtx);
        uint64_t actual = 0;
        for (int i = 0; i < num_labels; i++) {
            actual += data[label][i];
        }
        return actual > 0 ? static_cast<double>(data[label][label]) / actual : 0;
    }
    double get_label_prediction_percent(int label) {
        std::lock_guard<std::mutex> lock(mtx);
        if (total_weight == 0) return 0;

        uint64_t predicted = 0;
        for (int i = 0; i < num_labels; i++) {
            predicted += data[i][label];
        }
        return static_cast<double>(predicted) / total_weight;
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
        return cm.get_total_weight();
    }
    double get_accuracy() {
        return cm.get_accuracy();
    }
    double get_balanced_accuracy() {
        return cm.get_balanced_accuracy();
    }
    // label=1=hot, label=0=cold
    // TP: 预测热，实际热   TN: 预测冷，实际冷
    // FP: 预测热，实际冷   FN: 预测冷，实际热
    uint64_t true_positives()  { return cm.get(1, 1); }
    uint64_t true_negatives()  { return cm.get(0, 0); }
    uint64_t false_positives() { return cm.get(0, 1); }
    uint64_t false_negatives() { return cm.get(1, 0); }
    double get_hot_precision() {
        return cm.get_label_precision(1);
    }
    double get_hot_recall() {
        return cm.get_label_recall(1);
    }
    double get_hot_prediction_percent() {
        return cm.get_label_prediction_percent(1);
    }
    void clear() {
        cm.clear();
    }
};

# endif
