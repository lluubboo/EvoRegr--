#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include "XoshiroCpp.hpp"

namespace Random {

    template <class T>
    std::vector<T> randomChoices(std::vector<T> const& source, int sample_size, XoshiroCpp::Xoshiro256Plus& random_engine) {
        std::vector<T> sample;
        std::sample(source.begin(), source.end(), std::back_inserter(sample), sample_size, random_engine);
        return sample;
    };
}