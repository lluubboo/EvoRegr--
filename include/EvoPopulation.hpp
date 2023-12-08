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
    mutable std::shared_mutex _mutex;
    static std::array<unsigned int, 2> calculate_migration_interval(unsigned int island_id, unsigned int island_count, unsigned int generation_size_limit);

public:
    
    EvoPopulation(unsigned int size) : _population(size) {};
    EvoPopulation(std::vector<EvoIndividual> population) : _population(population) {};

    void move_to_population(size_t index, EvoIndividual& individual) noexcept;
    void batch_population_move(EvoPopulation& subpopulation, size_t index) noexcept;
    void batch_swap_individuals(size_t island_id, size_t island_count, size_t ratio, XoshiroCpp::Xoshiro256Plus& random_engine) noexcept;
    void element_pushback(EvoIndividual& individual) noexcept;
    void clear();
    void reserve(size_t size);

    bool swap_individuals(size_t index1, size_t index2) noexcept;

    EvoIndividual get_individual(size_t index) noexcept;
    EvoIndividual get_random_individual(XoshiroCpp::Xoshiro256Plus& random_engine) noexcept;
    std::vector<EvoIndividual>::iterator begin();
    std::vector<EvoIndividual>::iterator end();
    std::vector<EvoIndividual>::const_iterator begin() const;
    std::vector<EvoIndividual>::const_iterator end() const;
    size_t size() const noexcept;

};
