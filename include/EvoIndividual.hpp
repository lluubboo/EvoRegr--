#pragma once
#include <iostream>
#include <tuple>
#include <vector>
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"

class EvoIndividual {
public:
    std::string to_string() const;
    std::string to_string_code() const;

    std::vector<MergeAllele> merger_chromosome;
    std::vector<TransformXAllele> x_transformer_chromosome;
    std::vector<TransformYAllele> y_transformer_chromosome;
    std::vector<RobustAllele> robuster_chromosome;
    
    double fitness;
};
