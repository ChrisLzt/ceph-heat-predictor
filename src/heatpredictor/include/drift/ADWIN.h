# ifndef ADWIN_H
# define ADWIN_H

# include <cmath>
# include <array>
# include <cstdint>
# include <deque>
# include <limits>
# include <stdexcept>

template <int max_size>
class Bucket {
public:
    std::array<double, max_size + 1> total_array{};
    std::array<double, max_size + 1> variance_array{};
    int current_idx=0;
    void insert_data(double value, double variance) {
        total_array[current_idx] = value;
        variance_array[current_idx] = variance;
        current_idx++;
    }
    void compress(int n_elements) {
        for (int i=n_elements;i<=max_size;i++) {
            total_array[i-n_elements] = total_array[i];
            variance_array[i-n_elements] = variance_array[i];
        }
        for (int i=max_size-n_elements+1;i<=max_size;i++) {
            total_array[i] = 0.0;
            variance_array[i] = 0.0;
        }
        current_idx -= n_elements;
    }
};

template <int max_buckets=5>
class AdaptiveWindowing {
public:
//private:
    std::deque<Bucket<max_buckets>*> bucket_deque =
        std::deque<Bucket<max_buckets>*>(1, new Bucket<max_buckets>());
    double delta;
    double total = 0.0;
    double variance = 0.0;
    int clock;
    int min_window_length;
    int grace_period;
    int n_buckets = 0;
    int max_n_buckets = 0;
    uint64_t width = 0;
    uint64_t tick = 0;
    uint64_t n_detections = 0;
    uint64_t _calculate_bucket_size(size_t row) const {
        if (row >= std::numeric_limits<uint64_t>::digits) {
            throw std::overflow_error("ADWIN bucket row exceeds uint64_t");
        }
        return uint64_t{1} << row;
    }
    void _compress_buckets() {
        Bucket<max_buckets>* bucket = bucket_deque[0];
        Bucket<max_buckets>* next_bucket = nullptr;
        size_t idx = 0;
        while (bucket) {
            int k = bucket->current_idx;
            if (k == max_buckets+1) {
                if (idx + 1 < bucket_deque.size()) {
                    next_bucket = bucket_deque[idx+1];
                } else {
                    next_bucket = new Bucket<max_buckets>();
                    bucket_deque.push_back(next_bucket);
                }
                uint64_t n1 = _calculate_bucket_size(idx);
                uint64_t n2 = _calculate_bucket_size(idx);
                double n1_d = static_cast<double>(n1);
                double n2_d = static_cast<double>(n2);
                double mu1 = bucket->total_array[0] / n1_d;
                double mu2 = bucket->total_array[1] / n2_d;

                double total12 = bucket->total_array[0] + bucket->total_array[1];
                double temp = n1_d * n2_d * (mu1 - mu2) * (mu1 - mu2) /
                    (n1_d + n2_d);
                double v12 = bucket->variance_array[0] + bucket->variance_array[1] + temp;
                next_bucket->insert_data(total12, v12);
                n_buckets--;
                bucket->compress(2);

                if (next_bucket->current_idx <= max_buckets) {
                    break;
                }
            } else {
                break;
            }
            if (idx + 1 < bucket_deque.size()) {
                bucket = bucket_deque[idx+1];
            } else {
                bucket = nullptr;
            }
            idx++;
        }
    }
    void _insert_element(double value, double variance) {
        Bucket<max_buckets>* bucket = bucket_deque[0];
        bucket->insert_data(value, variance);
        n_buckets++;

        if (n_buckets > max_n_buckets) {
            max_n_buckets = n_buckets;
        }

        width++;
        double increment_variance = 0.0;
        if (width > 1) {
            increment_variance = (width - 1) * (value - total / (width - 1))
                * (value - total / (width - 1)) / width;
        }
        this->variance += increment_variance;
        total += value;

        _compress_buckets();
    }
    uint64_t _delete_element() {
        Bucket<max_buckets>* bucket = bucket_deque[bucket_deque.size()-1];
        uint64_t n = _calculate_bucket_size(bucket_deque.size()-1);
        double n_d = static_cast<double>(n);
        double u = bucket->total_array[0];
        double mu = u / n_d;
        double v = bucket->variance_array[0];

        width -= n;
        total -= u;
        double width_d = static_cast<double>(width);
        double mu_window = total / width_d;
        double increment_variance = v +
            n_d * width_d * (mu - mu_window) * (mu - mu_window) /
            (n_d + width_d);
        variance -= increment_variance;

        bucket->compress(1);
        n_buckets--;

        if (bucket->current_idx == 0) {
            delete bucket_deque.back();
            bucket_deque.pop_back();
        }

        return n;
    }
    bool _evaluate_cut(double n0, double n1, double delta_mean, double delta) {
        double delta_prime = std::log(2 * std::log(width) / delta);
        double m_recip = (1.0 / (n0 - min_window_length + 1)) + (1.0 / (n1 - min_window_length + 1));
        double epsilon = std::sqrt(2 * m_recip * variance / width * delta_prime) + 2.0 / 3.0 * delta_prime * m_recip;
        return (std::abs(delta_mean) > epsilon);
    }
    bool _detect_change() {
        bool change_detected = false;
        bool exit_flag = false;
        tick++;

        if ((tick % static_cast<uint64_t>(clock) == 0) &&
            (width > static_cast<uint64_t>(grace_period))) {
            bool reduce_width = true;
            while (reduce_width) {
                reduce_width = false;
                exit_flag = false;
                uint64_t n0 = 0;
                uint64_t n1 = width;
                double u0 = 0.0;
                double u1 = total;

                for (int idx=bucket_deque.size()-1;idx>=0;idx--) {
                    if (exit_flag) {
                        break;
                    }
                    Bucket<max_buckets>* bucket = bucket_deque[idx];

                    for (int k=0;k<bucket->current_idx;k++) {
                        n0 += _calculate_bucket_size(idx);
                        n1 -= _calculate_bucket_size(idx);
                        u0 += bucket->total_array[k];
                        u1 -= bucket->total_array[k];

                        if ((idx == 0) && (k == bucket->current_idx - 1)) {
                            exit_flag = true;
                            break;
                        }

                        double delta_mean = (u0 / n0) - (u1 / n1);
                        if ((n1 >= static_cast<uint64_t>(min_window_length)) &&
                            (n0 >= static_cast<uint64_t>(min_window_length))
                            && (_evaluate_cut(n0, n1, delta_mean, delta))) {
                            reduce_width = true;
                            change_detected = true;
                            if (width > 0) {
                                n0 -= _delete_element();
                                exit_flag = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (change_detected) {
            n_detections++;
        }

        return change_detected;
    }
public:
    AdaptiveWindowing(double delta=0.002, int clock=32, int min_window_length=5, int grace_period=10)
        : delta(delta), clock(clock), min_window_length(min_window_length),
        grace_period(grace_period) {}
    // move constructor
    AdaptiveWindowing(AdaptiveWindowing&& other) : AdaptiveWindowing() { *this = std::move(other); }
    ~AdaptiveWindowing() {
        for (Bucket<max_buckets>* bucket : bucket_deque) {
            if (bucket == nullptr) continue;
            delete bucket;
        }
    }
    // move assignment
    AdaptiveWindowing& operator=(AdaptiveWindowing&& other) {
        if (this == &other) return *this;
        this->delta = other.delta;
        this->clock = other.clock;
        this->min_window_length = other.min_window_length;
        this->grace_period = other.grace_period;
        for (auto bucket : this->bucket_deque) {
            if (bucket == nullptr) continue;
            delete bucket;
        }
        this->bucket_deque = std::move(other.bucket_deque);
        other.bucket_deque.clear();
        this->width = other.width;
        this->total = other.total;
        this->variance = other.variance;
        this->n_detections = other.n_detections;
        this->tick = other.tick;
        this->n_buckets = other.n_buckets;
        this->max_n_buckets = other.max_n_buckets;
        return *this;
    }
    bool update(double value) {
        _insert_element(value, 0.0);

        return _detect_change();
    }
};

template <int max_buckets=5>
class ADWIN {
public:
    bool drift_detected = false;
private:
    AdaptiveWindowing<max_buckets> _helper;
    double delta;
    int clock;
    int min_window_length;
    int grace_period;
    void _reset() {
        drift_detected = false;
        _helper = AdaptiveWindowing<max_buckets>(delta, clock, min_window_length, grace_period);
    }
public:
    ADWIN(double delta=0.002, int clock=32, int min_window_length=5, int grace_period=10)
        : delta(delta), clock(clock), min_window_length(min_window_length),
        grace_period(grace_period) {
        _reset();
    }
    // move constructor
    ADWIN(ADWIN&& other) : ADWIN() { *this = std::move(other); }
    // move assignment
    ADWIN& operator=(ADWIN&& other) {
        if (this == &other) return *this;
        this->delta = other.delta;
        this->clock = other.clock;
        this->min_window_length = other.min_window_length;
        this->grace_period = other.grace_period;
        this->_helper = std::move(other._helper);
        this->drift_detected = other.drift_detected;
        return *this;
    }
    void update(double x) {
        if (drift_detected) {
            _reset();
        }
        drift_detected = _helper.update(x);
    }
};

# endif
