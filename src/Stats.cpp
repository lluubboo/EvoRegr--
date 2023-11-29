#include <algorithm>
#include <vector>
#include <numeric>
#include <tgmath.h>
#include "Stats.hpp"

/**
 * @brief Checks if a vector is empty.
 *
 * This function checks if the input vector is empty and throws an exception if it is.
 *
 * @tparam T The type of the elements in the vector.
 * @param vector The vector to check.
 * @throws std::invalid_argument If the vector is empty.
 */
template <typename T>
void DescriptiveStatistics::check_arguments(std::vector<T> const& vector) {
    if (vector.empty()) {
        throw std::invalid_argument("check_arguments: vector size cannot be zero");
    }
}

template void DescriptiveStatistics::check_arguments<int>(std::vector<int> const& vector);
template void DescriptiveStatistics::check_arguments<float>(std::vector<float> const& vector);
template void DescriptiveStatistics::check_arguments<double>(std::vector<double> const& vector);

/**
 * @brief Checks if a pointer is null or the size is zero.
 *
 * This function checks if the input pointer is null or the size is zero and throws an exception in these cases.
 *
 * @tparam T The type of the elements in the array.
 * @param ptr A pointer to the first element of the array.
 * @param size The number of elements in the array.
 * @throws std::invalid_argument If ptr is a null pointer or size is zero.
 */
template <typename T>
void DescriptiveStatistics::check_arguments(T* ptr, unsigned int size) {
    if (ptr == nullptr) {
        throw std::invalid_argument("check_arguments: input pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("check_arguments: size cannot be zero");
    }
}

template void DescriptiveStatistics::check_arguments<int>(int* ptr, unsigned int size);
template void DescriptiveStatistics::check_arguments<float>(float* ptr, unsigned int size);
template void DescriptiveStatistics::check_arguments<double>(double* ptr, unsigned int size);

/**
 * @brief Generates a vector from a pointer and a size.
 *
 * This function creates a std::vector from a given pointer and size.
 * It checks if the pointer is null or the size is zero and throws an exception in these cases.
 *
 * @tparam T The type of the elements in the vector.
 * @param ptr A pointer to the first element of the array to convert to a vector.
 * @param size The number of elements in the array.
 * @return A std::vector containing the elements from the array.
 * @throws std::invalid_argument If ptr is a null pointer or size is zero.
 */
template <typename T>
std::vector<T> DescriptiveStatistics::generate_vector(T* ptr, unsigned int size) {
    check_arguments(ptr, size);
    return std::vector<T>(ptr, ptr + size);
}

template std::vector<double> DescriptiveStatistics::generate_vector(double* ptr, unsigned int size);
template std::vector<float> DescriptiveStatistics::generate_vector(float* ptr, unsigned int size);
template std::vector<int> DescriptiveStatistics::generate_vector(int* ptr, unsigned int size);


/**
 * Calculates the median of an array.
 * 
 * @tparam T The type of the elements in the array.
 * @param array Pointer to the array.
 * @param size The size of the array.
 * @return The median value.
 * @throws std::invalid_argument if the input vector is empty.
 */
template <typename T>
T DescriptiveStatistics::median(T* array, unsigned size) {
    return DescriptiveStatistics::median(generate_vector(array, size));
}

//explicit instantiation 
template double DescriptiveStatistics::median(double*, unsigned int);
template int DescriptiveStatistics::median(int*, unsigned int);
template float DescriptiveStatistics::median(float*, unsigned int);

/**
 * Calculates the median of a given vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return The median value of the vector.
 * @throws std::invalid_argument if the input vector is empty.
 */
template <typename T>
T DescriptiveStatistics::median(std::vector<T> vector) {
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
        return (vector.at(vector_size / 2) + vector.at((vector_size / 2) + 1)) / 2.;
    }
    else {
        //if size is odd, return midlle member (beware of floor at size/2 divide operation)
        return vector.at((vector_size / 2) + 1);
    }
}

//explicit instantiation 
template double DescriptiveStatistics::median(std::vector<double>);
template int DescriptiveStatistics::median(std::vector<int>);
template float DescriptiveStatistics::median(std::vector<float>);

/**
 * Calculates the mean of a vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return The mean value of the vector.
 * @throws std::invalid_argument if the input vector is empty.
 */
template <typename T>
T DescriptiveStatistics::mean(std::vector<T> const& vector) {
    check_arguments(vector);
    return std::reduce(vector.begin(), vector.end()) / vector.size();
}

//explicit instantiation 
template double DescriptiveStatistics::mean(std::vector<double> const&);
template int DescriptiveStatistics::mean(std::vector<int> const&);
template float DescriptiveStatistics::mean(std::vector<float> const&);


/**
 * Calculates the geometric mean of a vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return The geometric mean of the vector.
 * @throws std::invalid_argument if the input vector is empty.
 */
template <typename T>
T DescriptiveStatistics::geometric_mean(std::vector<T> const& vector) {
    check_arguments(vector);
    T product = std::accumulate(vector.begin(), vector.end(), 1.0, std::multiplies<T>());
    return std::pow(product, 1.0 / vector.size());
}

//explicit instantiation 
template double DescriptiveStatistics::geometric_mean(std::vector<double> const&);
template int DescriptiveStatistics::geometric_mean(std::vector<int> const&);
template float DescriptiveStatistics::geometric_mean(std::vector<float> const&);

/**
 * Calculates the standard deviation of a vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return The standard deviation of the vector.
 * @throws std::invalid_argument if the input vector is empty.
 */
template <typename T>
T DescriptiveStatistics::standard_deviation(std::vector<T> const& vector) {
    check_arguments(vector);
    std::vector<T> squared_residuals = DescriptiveStatistics::squared_residuals(vector);
    return sqrt(std::reduce(squared_residuals.begin(), squared_residuals.end()) / (squared_residuals.size() - 1));
}

//explicit instantiation 
template double DescriptiveStatistics::standard_deviation(std::vector<double> const&);
template int DescriptiveStatistics::standard_deviation(std::vector<int> const&);
template float DescriptiveStatistics::standard_deviation(std::vector<float> const&);

/**
 * Calculates the standard deviation of an array.
 * 
 * @tparam T The type of elements in the array.
 * @param array Pointer to the array.
 * @param size The size of the array.
 * @return The standard deviation of the array.
 * @throws std::invalid_argument if the input vector is empty.
 */
template <typename T>
T DescriptiveStatistics::standard_deviation(T* array, unsigned int size) {
    std::vector<T> squared_residuals = DescriptiveStatistics::squared_residuals(generate_vector(array, size));
    return sqrt(std::reduce(squared_residuals.begin(), squared_residuals.end()) / (squared_residuals.size() - 1));
}

//explicit instantiation 
template double DescriptiveStatistics::standard_deviation(double*, unsigned int);
template int DescriptiveStatistics::standard_deviation(int*, unsigned int);
template float DescriptiveStatistics::standard_deviation(float*, unsigned int);

/**
 * Calculates the squared residuals of a vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return A vector containing the squared residuals.
 * @throws std::invalid_argument if the input vector is empty.
 */
template <typename T>
std::vector<T> DescriptiveStatistics::squared_residuals(std::vector<T> vector) {
    check_arguments(vector);
    T mean = DescriptiveStatistics::mean(vector);
    std::for_each(vector.begin(), vector.end(), [&mean](T& x) {x -= mean; x *= x;});
    return vector;
}

//explicit instantiation 
template std::vector<double> DescriptiveStatistics::squared_residuals(std::vector<double>);
template std::vector<int> DescriptiveStatistics::squared_residuals(std::vector<int>);
template std::vector<float> DescriptiveStatistics::squared_residuals(std::vector<float>);
