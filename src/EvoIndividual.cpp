#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include <Eigen/Dense>
#include "XoshiroCpp.hpp"
#include "EvoIndividual.hpp"

EvoIndividual::EvoIndividual() : fitness(std::numeric_limits<double>::max()), is_healthy(false) {}

void EvoIndividual::evaluate(double value) {
    fitness = value;
    is_healthy = (value == std::numeric_limits<double>::max()) ? false : true;
}

std::string EvoIndividual::to_string() const {

    std::string string_genome;

    for (int i = 0; i < merger_chromosome.size(); i++) {
        string_genome += merger_chromosome[i].to_string() + x_transformer_chromosome[i].to_string();
        string_genome += (i == merger_chromosome.size() - 1) ? "" : " + ";
    }

    string_genome += "\n\n";

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