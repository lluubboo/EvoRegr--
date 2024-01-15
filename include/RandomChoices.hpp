#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include "RandomNumberGenerator.hpp"
#include "XoshiroCpp.hpp"
#include "EvoIndividual.hpp"

namespace Random {

    template <class T>
    std::vector<T> randomChoices(std::vector<T> const& source, int sample_size, XoshiroCpp::Xoshiro256Plus& random_engine) {
        std::vector<T> sample;
        std::sample(source.begin(), source.end(), std::back_inserter(sample), sample_size, random_engine);
        return sample;
    };

    template <class T>
    std::vector<T> randomChoices(T* source, int array_size, int sample_size, XoshiroCpp::Xoshiro256Plus& random_engine) {
        std::vector<T> sample;
        std::sample(source, source + array_size, std::back_inserter(sample), sample_size, random_engine);
        return sample;
    }

    template <class T>
    T randomChoice(std::vector<T> const& source, XoshiroCpp::Xoshiro256Plus& random_engine) {
        return source[RandomNumbers::rand_interval_int(0, source.size() - 1, random_engine)];
    };
}