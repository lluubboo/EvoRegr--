#pragma once
#include <cstddef>

/**
 * @brief Struct representing the boundary conditions for the evolutionary algorithm.
 */
struct EvoBoundaryConditions {
    size_t island_generation_size;          ///< The island generation size.
    size_t generation_count;                ///< The maximum number of generations.
    size_t interaction_cols;                ///< The number of columns used for interaction.
    size_t mutation_ratio;                  ///< The mutation ratio as a percentage.
    size_t island_count;                    ///< The number of islands in the algorithm.
    size_t migration_ratio;                 ///< The migration ratio as a percentage.
    size_t migration_interval;              ///< The interval at which migration occurs.
    size_t global_generation_size;          ///< The maximum size of the global generation.
    size_t migrants_count;                  ///< The number of migrants in each migration.

    EvoBoundaryConditions() :
        island_generation_size(100),
        generation_count(100),
        interaction_cols(0),
        mutation_ratio(15),
        island_count(12),
        migration_ratio(5),
        migration_interval(5),
        global_generation_size(island_generation_size* island_count),
        migrants_count(static_cast<size_t>((global_generation_size* migration_ratio) / 100))
    {}

    EvoBoundaryConditions(
        size_t island_generation_size,
        size_t generation_count,
        size_t interaction_cols,
        size_t mutation_ratio,
        size_t island_count,
        size_t migration_ratio,
        size_t migration_interval
    ) :
        island_generation_size(island_generation_size),
        generation_count(generation_count),
        interaction_cols(interaction_cols),
        mutation_ratio(mutation_ratio),
        island_count(island_count),
        migration_ratio(migration_ratio),
        migration_interval(migration_interval),
        global_generation_size(island_generation_size* island_count),
        migrants_count(static_cast<size_t>((global_generation_size* migration_ratio) / 100))
    {}
};