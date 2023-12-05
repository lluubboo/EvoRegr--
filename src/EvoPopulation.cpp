#include "EvoPopulation.hpp"
#include <algorithm>
#include <mutex>

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
 * Moves elements from a subpopulation to the main population.
 *
 * @param subpopulation A vector of EvoIndividuals to be moved.
 * @param begin An iterator pointing to the position in the main population where the elements should be moved.
 *
 * This function locks the population for thread safety, moves the elements from the subpopulation to the main population,
 * and then clears the subpopulation.
 */
void EvoPopulation::batch_population_move(std::vector<EvoIndividual>& subpopulation, std::vector<EvoIndividual>::iterator begin) {
    std::unique_lock lock(_mutex);
    std::move(subpopulation.begin(), subpopulation.end(), begin);
    subpopulation.clear();
}

/**
 * Adds an individual to the end of the population.
 *
 * @param individual The EvoIndividual to be added.
 *
 * This function locks the population for thread safety and moves the individual to the end of the population.
 */
void EvoPopulation::element_pushback(EvoIndividual& individual) {
    std::unique_lock lock(_mutex);
    _population.push_back(std::move(individual));
}