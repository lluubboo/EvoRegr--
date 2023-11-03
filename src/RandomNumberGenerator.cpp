#include <random>
#include "RandomNumberGenerator.hpp"

int RandomNumbers::rand_interval_int(int const min, int const max, XoshiroCpp::Xoshiro256Plus& random_engine) {
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(random_engine);
}

float RandomNumbers::rand_interval_float(float const min, float const max, XoshiroCpp::Xoshiro256Plus& random_engine) {
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(random_engine);
}