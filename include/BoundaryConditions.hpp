#pragma once
#include <cstddef>
#include <vector>
#include <array>

/**
 * @brief Struct representing the boundary conditions for the evolutionary algorithm.
 */
struct EvoBoundaryConditions {
    size_t island_generation_size;         
    size_t generation_count;              
    size_t interaction_cols;
    size_t mutation_ratio;
    size_t basis_function_complexity;
    float regularization_parameter;
    size_t island_count;
    size_t migration_ratio;             
    size_t migration_interval;
    size_t test_ratio;              
    size_t global_generation_size;
    size_t migrants_count;                 
    size_t elites_count;
    std::vector<std::array<size_t, 2>> island_borders;
    size_t test_set_size;
    size_t training_set_size;


    EvoBoundaryConditions() :
        island_generation_size(100),
        generation_count(100),
        interaction_cols(0),
        mutation_ratio(15),
        basis_function_complexity(4),
        regularization_parameter(1.0),
        island_count(12),
        migration_ratio(5),
        migration_interval(5),
        test_ratio(20),
        global_generation_size(island_generation_size* island_count),
        migrants_count(static_cast<size_t>((global_generation_size* migration_ratio) / 100)),
        elites_count(static_cast<size_t>((island_generation_size * 5) / 100)),
        island_borders(0),
        test_set_size(0),
        training_set_size(0)
    {

        // Initialize island_borders
        for (size_t i = 0; i < island_count; ++i) {
            size_t start = i * island_generation_size;
            size_t end = start + island_generation_size - 1;
            island_borders.push_back({start, end});
        }
    }

    EvoBoundaryConditions(
        size_t island_generation_size,
        size_t generation_count,
        size_t interaction_cols,
        size_t mutation_ratio,
        size_t basis_function_complexity,
        float regularization_parameter,
        size_t island_count,
        size_t migration_ratio,
        size_t migration_interval,
        size_t test_ratio
        ) :
        island_generation_size(island_generation_size),
        generation_count(generation_count),
        interaction_cols(interaction_cols),
        mutation_ratio(mutation_ratio),
        basis_function_complexity(basis_function_complexity),
        regularization_parameter(regularization_parameter),
        island_count(island_count),
        migration_ratio(migration_ratio),
        migration_interval(migration_interval),
        test_ratio(test_ratio),
        global_generation_size(island_generation_size* island_count),
        migrants_count(static_cast<size_t>((global_generation_size* migration_ratio) / 100)),
        elites_count(static_cast<size_t>((island_generation_size * 5) / 100)),
        island_borders(0),
        test_set_size(0),
        training_set_size(0)
    {

        // Initialize island_borders

        for (size_t i = 0; i < island_count; ++i) {
            size_t start = i * island_generation_size;
            size_t end = start + island_generation_size - 1;
            island_borders.push_back({start, end});
        }
    }
};