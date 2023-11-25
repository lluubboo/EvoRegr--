#include <algorithm>
#include <vector>
#include <numeric>
#include <tgmath.h>
#include "Stats.hpp"


/**
 * Calculates the median of an array.
 * 
 * @tparam T The type of the elements in the array.
 * @param array Pointer to the array.
 * @param size The size of the array.
 * @return The median value.
 */
template <typename T>
T DescriptiveStatistics::median(T* array, int size) {
    std::vector<T> vec(array, array + size);
    return DescriptiveStatistics::median(vec);
}

//explicit instantiation 
template double DescriptiveStatistics::median(double*, int);
template int DescriptiveStatistics::median(int*, int);
template float DescriptiveStatistics::median(float*, int);

/**
 * Calculates the median of a given vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return The median value of the vector.
 */
template <typename T>
T DescriptiveStatistics::median(std::vector<T> vector) {
    std::sort(vector.begin(), vector.end());
    int vector_size = vector.size();
    if (vector.empty()) {
        return std::numeric_limits<T>::max();
    }
    else if (vector_size == 1) {
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
 */
template <typename T>
T DescriptiveStatistics::mean(std::vector<T> const& vector) {
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
 */
template <typename T>
T DescriptiveStatistics::geometric_mean(std::vector<T> const& vector) {
    T product = std::accumulate(vector.begin(), vector.end(), 1.0, std::multiplies<T>());
    return std::pow(product, 1.0 / vector.size());
}

//explicit instantiation 
template double DescriptiveStatistics::geometric_mean(std::vector<double> const&);
template int DescriptiveStatistics::geometric_mean(std::vector<int> const&);
template float DescriptiveStatistics::geometric_mean(std::vector<float> const&);

/**
 * Calculates the standard deviation of a given vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return The standard deviation of the vector.
 */
template <typename T>
T DescriptiveStatistics::standard_deviation(std::vector<T> const& vector) {
    std::vector<T> squared_residuals = DescriptiveStatistics::squared_residuals(vector);
    return sqrt(std::reduce(squared_residuals.begin(), squared_residuals.end()) / (squared_residuals.size() - 1));
}

//explicit instantiation 
template double DescriptiveStatistics::standard_deviation(std::vector<double> const&);
template int DescriptiveStatistics::standard_deviation(std::vector<int> const&);
template float DescriptiveStatistics::standard_deviation(std::vector<float> const&);

/**
 * Calculates the squared residuals of a vector.
 * 
 * @tparam T The type of elements in the vector.
 * @param vector The input vector.
 * @return A vector containing the squared residuals.
 */
template <typename T>
std::vector<T> DescriptiveStatistics::squared_residuals(std::vector<T> vector) {
    T mean = DescriptiveStatistics::mean(vector);
    std::for_each(vector.begin(), vector.end(), [&mean](T& x) {x -= mean; x *= x;});
    return vector;
}

//explicit instantiation 
template std::vector<double> DescriptiveStatistics::squared_residuals(std::vector<double>);
template std::vector<int> DescriptiveStatistics::squared_residuals(std::vector<int>);
template std::vector<float> DescriptiveStatistics::squared_residuals(std::vector<float>);
