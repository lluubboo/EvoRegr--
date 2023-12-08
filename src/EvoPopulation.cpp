#include <algorithm>
#include <mutex>
#include "EvoPopulation.hpp"
#include "RandomChoices.hpp"

/**
 * The function "evaluate" assigns a fitness value to an EvoIndividual object and determines if it is
 * healthy based on the value.
 *
 * @param value The parameter "value" is a double that represents the fitness value of the individual.
 */
void EvoIndividual::evaluate(double value) {
    fitness = value;
    is_healthy = (value == std::numeric_limits<double>::max()) ? false : true;
}

/**
 * The function "merge_chromosome_to_string_vector" converts each allele in the "merger_chromosome"
 * vector to a string and stores them in a new vector.
 *
 * @return a vector of strings.
 */
std::vector<std::string> EvoIndividual::merge_chromosome_to_string_vector() const {
    std::vector<std::string> string_vector;
    for (auto const& allele : merger_chromosome) {
        string_vector.push_back(allele.to_string());
    }
    return string_vector;
}

/**
 * The function converts the alleles of a robust chromosome into a vector of strings.
 *
 * @return The function `robust_chromosome_to_string_vector` returns a `std::vector<std::string>`.
 */
std::vector<std::string> EvoIndividual::robust_chromosome_to_string_vector() const {
    std::vector<std::string> string_vector;
    for (auto const& allele : robuster_chromosome) {
        string_vector.push_back(allele.to_string());
    }
    return string_vector;
};

/**
 * The function transforms a chromosome of alleles into a vector of strings.
 *
 * @return a vector of strings.
 */
std::vector<std::string> EvoIndividual::transform_predictor_chromosome_to_string_vector() const {
    std::vector<std::string> string_vector;
    for (auto const& allele : x_transformer_chromosome) {
        string_vector.push_back(allele.to_string());
    }
    return string_vector;
};

/**
 * The function transforms the target chromosome of an EvoIndividual object into a vector of strings.
 *
 * @return a vector of strings.
 */
std::vector<std::string> EvoIndividual::transform_target_chromosome_to_string_vector() const {
    std::vector<std::string> string_vector;
    for (auto const& allele : y_transformer_chromosome) {
        string_vector.push_back(allele.to_string());
    }
    return string_vector;
};

/**
 * The function `to_math_formula` returns a string representation of a mathematical formula based on
 * the values of certain chromosomes.
 *
 * @return a string representation of a mathematical formula.
 */
std::string EvoIndividual::to_math_formula() const {
    std::string formula;
    //append predictor transformation
    for (unsigned int i = 0; i < merger_chromosome.size() && i < x_transformer_chromosome.size(); i++) {
        formula.append(merger_chromosome.at(i).to_string());
        formula.append(x_transformer_chromosome.at(i).to_string());
        formula.append((i == merger_chromosome.size() - 1) ? "" : " + ");
    }
    //append target transformation
    formula.append(" = (Y)" + y_transformer_chromosome.at(0).to_string() + "\n");
    return formula;
}

/**
 * The function `to_string_code()` returns a string representation of the genome of an EvoIndividual
 * object for cache purposes. This code is key to the cache of fitness values.
 *
 * @return a string representation of the code of an EvoIndividual object.
 */
std::string EvoIndividual::to_string_code() const {
    std::string string_genome;
    // merger chromosome
    for (auto const& allele : merger_chromosome) {
        string_genome += allele.to_string_code();
    }
    // transform X chromosome
    for (auto const& allele : x_transformer_chromosome) {
        string_genome += allele.to_string_code();
    }
    // transform Y chromosome
    string_genome += y_transformer_chromosome.at(0).to_string_code();
    // robuster chromosome
    string_genome += robuster_chromosome.at(0).to_string_code();
    return string_genome;
}

/**
 * Returns the size of the population.
 *
 * @return The number of individuals in the population.
 *
 * This function returns the number of individuals in the population, which is represented by the `_population` member variable.
 */
size_t EvoPopulation::size() const noexcept {
    return _population.size();
}

/**
 * Moves elements from a subpopulation to the main population.
 *
 * @param subpopulation A vector of EvoIndividuals to be moved.
 * @param begin An iterator pointing to the position in the main population where the elements should be moved.
 *
 * This function locks the population for thread safety, moves the elements from the subpopulation to the main population,
 * and then clears the subpopulation.
 */
void EvoPopulation::batch_population_move(EvoPopulation& subpopulation, size_t index) noexcept {
    std::unique_lock lock(_mutex);
    {
        auto begin = _population.begin() + index;
        size_t remaining_space = std::distance(begin, _population.end());

        if (subpopulation.size() <= remaining_space) {
            std::move(subpopulation.begin(), subpopulation.end(), begin);
            subpopulation.clear();
        }
        else {
            std::cerr << "Error: subpopulation size exceeds remaining space in population." << std::endl;
        }
    }
}

/**
 * Adds an individual to the end of the population.
 *
 * @param individual The EvoIndividual to be added.
 *
 * This function moves the individual to the end of the population. Does not lock the population for thread safety.
 */
void EvoPopulation::element_pushback(EvoIndividual& individual) noexcept {
    _population.push_back(std::move(individual));
}

/**
 * Swaps two individuals within the population.
 *
 * @param index1 The index of the first individual to be swapped.
 * @param index2 The index of the second individual to be swapped.
 * @return true if the swap was successful, false if either index was out of range.
 *
 * This function locks the population for thread safety. If both indices are valid, it swaps the individuals at these indices in the population and returns true. If either index is out of range, it returns false without modifying the population.
 */
bool EvoPopulation::swap_individuals(size_t index1, size_t index2) noexcept {
    std::unique_lock lock(_mutex);
    {
        if (index1 < _population.size() && index2 < _population.size() && index1 != index2) {
            std::swap(_population[index1], _population[index2]);
            return true;
        }
        else {
            std::cerr << "Error: cannot swap individuals. " << std::endl;
            return false;
        }
    }
}

/**
 * Swaps a number of individuals within the population.
 *
 * @param island_id The ID of the island for which to perform the swaps.
 * @param island_count The total number of islands.
 * @param ratio The ratio of the population size that determines the number of swaps.
 * @param random_engine A reference to a random number generator.
 * @return true if the swaps were successful, false otherwise.
 *
 * This function calculates a migration interval based on the island ID, the total number of islands, and the population size. It then performs a number of swaps within this interval. The number of swaps is determined by the ratio parameter. Each swap involves two randomly chosen individuals within the migration interval.
 */
void EvoPopulation::batch_swap_individuals(size_t island_id, size_t island_count, size_t ratio, XoshiroCpp::Xoshiro256Plus& random_engine) noexcept {
    std::array<unsigned int, 2> migration_interval = EvoPopulation::calculate_migration_interval(
        island_id,
        island_count,
        _population.size()
    );
    size_t migrants_count = ratio * (_population.size() / island_count);
    for (size_t i = 0; i < migrants_count; i++) {
        swap_individuals(RandomNumbers::rand_interval_int(migration_interval[0], migration_interval[1], random_engine),
            RandomNumbers::rand_interval_int(migration_interval[0], migration_interval[1], random_engine));
    }
}

/**
 * Calculates the migration interval for a given island.
 *
 * @param island_id The ID of the island for which to calculate the migration interval.
 * @param island_count The total number of islands.
 * @param generation_size_limit The maximum size of a generation.
 * @return An array of two unsigned integers representing the start and end of the migration interval.
 *
 * This function calculates the range of generations that a given island can migrate to. The start of the range is the maximum of 0 and the product of the island ID minus 1 and the generation size limit. The end of the range is the minimum of the product of the island ID plus 2 and the generation size limit, and the product of the island count and the generation size limit.
 */
std::array<unsigned int, 2> EvoPopulation::calculate_migration_interval(unsigned int island_id, unsigned int island_count, unsigned int generation_size_limit) {
    return { std::max(0u, (island_id - 1) * generation_size_limit),
    std::min((island_id + 2) * generation_size_limit, island_count * generation_size_limit) };
}

/**
 * @brief Gets a random individual from the population.
 *
 * This function selects a random individual from the population using a random number generator.
 * The random number generator is passed as a parameter and is used to generate a random index into the population vector.
 *
 * @param random_engine A reference to a `XoshiroCpp::Xoshiro256Plus` random number generator.
 *
 * @return EvoIndividual The randomly selected individual from the population.
 *
 * @note This function is marked `noexcept`, meaning it does not throw exceptions.
 */
EvoIndividual EvoPopulation::get_random_individual(XoshiroCpp::Xoshiro256Plus& random_engine) noexcept {
    return get_individual(RandomNumbers::rand_interval_int(0, _population.size() - 1, random_engine));
}

/**
 * @brief Get an individual from the population at a specific index.
 *
 * This method returns the individual at the specified index in the _population vector.
 * If the index is out of range, an error message is printed to the standard error stream and a default-constructed EvoIndividual is returned.
 *
 * @param index The position in the population from where the individual should be retrieved.
 * @return The individual at the specified index in the population.
 */
EvoIndividual EvoPopulation::get_individual(size_t index) noexcept {
    std::shared_lock lock(_mutex);
    {
        if (index >= _population.size()) {
            std::cerr << "Error: index out of range." << std::endl;
            return EvoIndividual();
        }
        return _population[index];
    }
}

/**
 * @brief Move an individual to a specific position in the population.
 *
 * This method moves the given individual to the specified index in the _population vector.
 * If the index is out of range, an error message is printed to the standard error stream.
 *
 * @param index The position in the population where the individual should be moved.
 * @param individual The individual to be moved into the population.
 */
void EvoPopulation::move_to_population(size_t index, EvoIndividual& individual) noexcept {
    std::unique_lock lock(_mutex);
    {
        if (index < _population.size()) {
            _population[index] = std::move(individual);
        }
        else {
            std::cerr << "Error: index out of range, individual cannot be setted." << std::endl;
        }
    }
}

/**
 * @brief Get an iterator to the beginning of the population.
 *
 * This method returns an iterator pointing to the first element in the _population vector.
 * If the vector is empty, the returned iterator will be equal to end().
 *
 * @return An iterator to the first element of the _population vector.
 */
std::vector<EvoIndividual>::iterator EvoPopulation::begin() {
    return _population.begin();
}

/**
 * @brief Get an iterator to the end of the population.
 *
 * This method returns an iterator pointing to the past-the-end element in the _population vector.
 * The past-the-end element is the theoretical element that would follow the last element in the vector.
 * It does not point to any element, and thus shall not be dereferenced.
 *
 * @return An iterator to the element following the last element of the _population vector.
 */
std::vector<EvoIndividual>::iterator EvoPopulation::end() {
    return _population.end();
}

/**
 * @brief Get a const iterator to the beginning of the population.
 *
 * This method returns a const_iterator pointing to the first element in the _population vector.
 * If the vector is empty, the returned iterator will be equal to end().
 *
 * @return A const_iterator to the first element of the _population vector.
 */
std::vector<EvoIndividual>::const_iterator EvoPopulation::begin() const {
    return _population.begin();
}

/**
 * @brief Get an iterator to the end of the population.
 *
 * This method returns a const_iterator pointing to the past-the-end element in the _population vector.
 * The past-the-end element is the theoretical element that would follow the last element in the vector.
 * It does not point to any element, and thus shall not be dereferenced.
 *
 * @return A const_iterator to the element following the last element of the _population vector.
 */
std::vector<EvoIndividual>::const_iterator EvoPopulation::end() const {
    return _population.end();
}

/**
 * @brief Clears the population.
 *
 * This method removes all elements from the _population vector.
 * After this call, _population.size() returns zero.
 * Method is not thread safe.
 */
void EvoPopulation::clear() {
    _population.clear();
}

/**
 * @brief Reserves storage for the population.
 *
 * This method increases the capacity of the _population vector to a value
 * that's greater or equal to size. If size is greater than the current
 * vector capacity, new storage is allocated, otherwise the method does nothing.
 *
 * @param size Number of elements for which to reserve storage.
 */
void EvoPopulation::reserve(size_t size) {
    _population.reserve(size);
}