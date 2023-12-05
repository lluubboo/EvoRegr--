#pragma once
#include <shared_mutex>
#include <vector>
#include <iostream>
#include <limits>
#include "EvoGene.hpp"

class EvoIndividual {
public:
    
    EvoIndividual() : fitness(std::numeric_limits<double>::max()), is_healthy(false) {};

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

class EvoPopulation {

    std::vector<EvoIndividual> _population;
    std::shared_mutex _mutex;

public:
    EvoPopulation(unsigned int size) : _population(size) {};
    void batch_population_move(std::vector<EvoIndividual>& subpopulation, std::vector<EvoIndividual>::iterator begin);
    void element_pushback(EvoIndividual& individual);
};
