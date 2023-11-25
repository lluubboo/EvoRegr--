#include <random>
#include "RandomNumberGenerator.hpp"

/**
 * Generates a random integer within the specified range.
 *
 * @param min The minimum value of the range (inclusive).
 * @param max The maximum value of the range (inclusive).
 * @param random_engine The random engine to use for generating the random number.
 * @return A random integer within the specified range.
 */
template<typename Engine>
int RandomNumbers::rand_interval_int(int const min, int const max, Engine& random_engine) {
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(random_engine);
}

template int RandomNumbers::rand_interval_int(int const min, int const max, std::mt19937& random_engine);
template int RandomNumbers::rand_interval_int(int const min, int const max, XoshiroCpp::Xoshiro256Plus& random_engine);

/**
 * Generates a random decimal number within the specified range.
 *
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @param random_engine The random engine to use for generating the number.
 * @return A random decimal number within the specified range.
 */
template<typename Engine, typename Number>
Number RandomNumbers::rand_interval_decimal_number(Number const min, Number const max, Engine& random_engine) {
    std::uniform_real_distribution<Number> distribution(min, max);
    return distribution(random_engine);
}

template double RandomNumbers::rand_interval_decimal_number(double const min, double const max, std::mt19937& random_engine);
template double RandomNumbers::rand_interval_decimal_number(double const min, double const max, XoshiroCpp::Xoshiro256Plus& random_engine);
template float RandomNumbers::rand_interval_decimal_number(float const min, float const max, std::mt19937& random_engine);
template float RandomNumbers::rand_interval_decimal_number(float const min, float const max, XoshiroCpp::Xoshiro256Plus& random_engine);
