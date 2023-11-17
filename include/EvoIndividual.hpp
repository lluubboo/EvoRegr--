#pragma once
#include <iostream>
#include <tuple>
#include <vector>
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"

class EvoIndividual {
public:
    EvoIndividual();
    std::vector<std::string> merge_chromosome_to_string_vector() const;
    std::vector<std::string> robust_chromosome_to_string_vector() const;
    std::vector<std::string> transform_predictor_chromosome_to_string_vector() const;
    std::vector<std::string> transform_target_chromosome_to_string_vector() const;
    std::string to_string_code() const;
    std::string to_math_formula() const;
    void evaluate(double);

    std::vector<MergeAllele> merger_chromosome;
    std::vector<TransformXAllele> x_transformer_chromosome;
    std::vector<TransformYAllele> y_transformer_chromosome;
    std::vector<RobustAllele> robuster_chromosome;

    double fitness;
    bool is_healthy;
};
