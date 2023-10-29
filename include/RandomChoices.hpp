#pragma once
#include <vector>
#include <random>
#include "XoshiroCpp.hpp"

namespace Random {

    template <class T>
    std::vector<T> randomChoices(std::vector<T> const& source, int sample_size, XoshiroCpp::Xoshiro256Plus& random_engine) {
        std::uniform_int_distribution<int> distribution(0, source.size()-1);
        std::vector<T> sample(sample_size);
        std::generate(sample.begin(), sample.end(), [&]() {return source.at(distribution(random_engine));});
        return sample;
    };
}