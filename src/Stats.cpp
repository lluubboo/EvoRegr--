#include <algorithm>
#include <vector>
#include <numeric>
#include <tgmath.h>
#include "Stats.hpp"


/**
 * The function calculates the median of an array of numbers.
 *
 * @param array An array of double values.
 * @param size The parameter "size" represents the number of elements in the array.
 *
 * @return the median value of the given array.
 */
double DescriptiveStatistics::median(double* array, int size) {

    std::vector<double> vec(array, array + size);
    std::sort(vec.begin(), vec.end());

    if (vec.size() % 2 == 0) {
        //if size is even, return arithmetic mean of the midlle vector members
        return (vec.at(size / 2) + vec.at((size / 2) + 1)) / 2.;
    }
    else {
        //if size is odd, return midlle member (beware of floor at size/2 divide operation)
        return vec.at((size / 2) + 1);
    }
}


/**
 * The function calculates the median of a given vector of doubles.
 *
 * @param vector The "vector" parameter is a vector of double values. It represents the dataset for
 * which we want to calculate the median.
 *
 * @return the median value of the given vector of doubles.
 */
double DescriptiveStatistics::median(std::vector<double> vector) {
    std::sort(vector.begin(), vector.end());
    int vector_size = vector.size();
    if (vector_size % 2 == 0) {
        //if size is even, return arithmetic mean of the midlle vector members
        return (vector.at(vector_size / 2) + vector.at((vector_size / 2) + 1)) / 2.;
    }
    else {
        //if size is odd, return midlle member (beware of floor at size/2 divide operation)
        return vector.at((vector_size / 2) + 1);
    }
}

/**
 * The function calculates the mean of a vector of doubles.
 *
 * @param vector The parameter "vector" is a vector of type double, which is a container that stores a
 * sequence of double values.
 *
 * @return The mean (average) of the given vector of doubles.
 */
double DescriptiveStatistics::mean(std::vector<double> const& vector) {
    return std::reduce(vector.begin(), vector.end()) / vector.size();
}

/**
 * The function calculates the standard deviation of a vector of numbers.
 * 
 * @param vector The "vector" parameter is a vector of double values for which we want to calculate the
 * standard deviation.
 * 
 * @return the standard deviation of the input vector.
 */
double DescriptiveStatistics::standard_deviation(std::vector<double> const& vector) {
    std::vector<double> squared_residuals = DescriptiveStatistics::squared_residuals(vector);
    return sqrt(std::reduce(squared_residuals.begin(), squared_residuals.end()) / (squared_residuals.size() - 1));
}

/**
 * The function calculates the squared residuals of a vector by subtracting the mean from each element
 * and squaring the result.
 * 
 * @param vector The "vector" parameter is a std::vector<double> object that represents a collection of
 * double values.
 * 
 * @return a vector of squared residuals.
 */
std::vector<double> DescriptiveStatistics::squared_residuals(std::vector<double> vector) {
    double mean = DescriptiveStatistics::mean(vector);
    std::for_each(vector.begin(), vector.end(), [&mean](double& x) {x -= mean; x *= x;});
    return vector;
}
