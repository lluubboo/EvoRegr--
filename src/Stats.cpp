#include <algorithm>
#include <vector>
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