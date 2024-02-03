#pragma once
#include <vector>

namespace Statistics {

    template <typename T>
    void check_arguments(std::vector<T> const& vector) {
        if (vector.empty()) {
            throw std::invalid_argument("Statistics: vector size cannot be zero");
        }
    }

    template <typename T>
    void check_arguments(T* ptr, unsigned int size) {
        if (ptr == nullptr) {
            throw std::invalid_argument("Statistics: input pointer cannot be null");
        }
        if (size == 0) {
            throw std::invalid_argument("Statistics: size cannot be zero");
        }
    }

    template <typename T>
    std::vector<T> generate_vector(T* ptr, unsigned int size) {
        check_arguments(ptr, size);
        return std::vector<T>(ptr, ptr + size);
    }

    template <typename T>
    T median(std::vector<T> vector) {
        check_arguments(vector);
        std::sort(vector.begin(), vector.end());
        int vector_size = vector.size();
        if (vector_size == 1) {
            return vector.at(0);
        }
        else if (vector_size == 2) {
            return (vector.at(0) + vector.at(1)) / 2.;
        }
        else if (vector_size % 2 == 0) {
            //if size is even, return arithmetic mean of the midlle vector members
            return ((vector.at(vector_size / 2) - 1) + vector.at(vector_size / 2)) / 2.;
        }
        else {
            //if size is odd, return midlle member (beware of floor at size/2 divide operation)
            return vector.at((vector_size / 2) + 1);
        }
    }

    template <typename T>
    T median(T* array, unsigned size) {
        return median(generate_vector(array, size));
    }

    template <typename T>
    T mean(std::vector<T> const& vector) {
        check_arguments(vector);
        return std::accumulate(vector.begin(), vector.end(), T(0)) / vector.size();
    }

    template <typename T>
    T mean(T* array, unsigned size) {
        return mean(generate_vector(array, size));
    }

    template <typename T>
    T geometric_mean(std::vector<T> const& vector) {
        check_arguments(vector);
        T product = std::accumulate(vector.begin(), vector.end(), T(1), std::multiplies<T>());
        return std::pow(product, 1.0 / vector.size());
    }

    template <typename T>
    T geometric_mean(T* array, unsigned size) {
        return geometric_mean(generate_vector(array, size));
    }

    template <typename T>
    std::vector<T> squared_residuals(std::vector<T> vector, T mean) {
        check_arguments(vector);
        std::for_each(vector.begin(), vector.end(), [mean](T& value) { value = std::pow(value - mean, 2); });
        return vector;
    }

    template <typename T>
    std::vector<T> squared_residuals(T* array, T mean, unsigned size) {
        return squared_residuals(generate_vector(array, size), mean);
    }

    template <typename T>
    T sum_squared_residuals(std::vector<T> vector, T mean) {
        check_arguments(vector);
        std::vector<T> sr = squared_residuals(vector, mean);
        return std::accumulate(sr.begin(), sr.end(), T(0));
    }

    template <typename T>
    T sum_squared_residuals(T* array, T mean, unsigned size) {
        return sum_squared_residuals(generate_vector(array, size), mean);
    }

    template <typename T>
    T variance(std::vector<T> const& vector) {
        check_arguments(vector);
        return sum_squared_residuals(vector, mean(vector)) / vector.size();
    }

    template <typename T>
    T variance(T* array, unsigned int size) {
        return variance(generate_vector(array, size));
    }

    template <typename T>
    T standard_deviation(std::vector<T> const& vector) {
        return sqrt(variance(vector));
    }

    template <typename T>
    T standard_deviation(T* array, unsigned int size) {
        return standard_deviation(generate_vector(array, size));
    }

    template <typename T>
    T cod(std::vector<T> target, std::vector<T> residuals) {
        check_arguments(target);
        check_arguments(residuals);

        // square it
        residuals = squared_residuals(residuals, T(0));

        // create sums
        T residual_sum_squares = std::accumulate(residuals.begin(), residuals.end(), T(0));
        T total_sum_squares = sum_squared_residuals(target, mean(target));
        return 1 - (residual_sum_squares / total_sum_squares);
    }

    template <typename T>
    T cod(T* target, T* residuals, unsigned int size) {
        return cod(generate_vector(target, size), generate_vector(residuals, size));
    }

    template <typename T>
    T coda(std::vector<T> target, std::vector<T> residuals, unsigned int predictor_count) {
        check_arguments(target);
        check_arguments(residuals);
        // square it
        residuals = squared_residuals(residuals, T(0));
        // create sums
        T residual_sum_squares = std::accumulate(residuals.begin(), residuals.end(), T(0));
        T total_sum_squares = sum_squared_residuals(target, mean(target));
        return 1 - ((residual_sum_squares / (target.size() - predictor_count)) / (total_sum_squares / (target.size() - 1)));
    }

    template <typename T>
    T coda(T* target, T* residuals, unsigned int size, unsigned int predictor_count) {
        return coda(generate_vector(target, size), generate_vector(residuals, size), predictor_count);
    }
}