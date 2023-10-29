#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include <Eigen/Dense>
#include "XoshiroCpp.hpp"

#include "EvoIndividual.hpp"

std::string EvoIndividual::to_string() const {
    std::string string_genome;
    // merger chromosome
    string_genome = "*******************************************PREDICTOR MERGE CHROMOSOME******************************************\n\n";
    for (auto const& allele : merger_chromosome) {
        string_genome += allele.to_string();
    }
    // transform X chromosome
    string_genome += "*****************************************PREDICTOR TRANSFORM CHROMOSOME****************************************\n\n";
    for (auto const& allele : x_transformer_chromosome) {
        string_genome += allele.to_string();
    }
    // transform Y chromosome
    string_genome += "******************************************TARGET TRANSFORM CHROMOSOME******************************************\n\n";
    string_genome += y_transformer_chromosome.at(0).to_string();
    // robuster chromosome
    string_genome += "**********************************************ROBUSTER CHROMOSOME**********************************************\n\n";
    string_genome += robuster_chromosome.at(0).to_string();
    return string_genome;
}

std::string EvoIndividual::to_string_code() const {
    
    std::string string_genome;

    // merger chromosome
    for (auto const& allele : merger_chromosome) {
        string_genome += allele.to_string();
    }

    // transform X chromosome
    for (auto const& allele : x_transformer_chromosome) {
        string_genome += allele.to_string();
    }

    // transform Y chromosome
    string_genome += y_transformer_chromosome.at(0).to_string();

    // robuster chromosome
    string_genome += robuster_chromosome.at(0).to_string();

    return string_genome;
}