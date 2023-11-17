#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <set>
#include <chrono>
#include <numeric>
#include <matplot/matplot.h>
#include "EvoAPI.hpp"
#include "IOTools.hpp"
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"
#include "Stats.hpp"
#include "Plotter.hpp"
#include "omp.h"

EvoAPI::EvoAPI(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols) {
    this->generation_size_limit = generation_size_limit;
    this->generation_count_limit = generation_count_limit;
    this->interaction_cols = interaction_cols;

    // desynchronize C++ and C I/O streams
    //std::ios_base::sync_with_stdio(false);
}

/**
 * The function `load_file` prompts the user to enter a filename, opens the file, and processes its
 * contents if it is successfully opened.
 */
void EvoAPI::load_file() {

    // loop while file is not opened
    while (true) {

        std::ifstream file;

        std::cout << "Enter filename (or 'x' to exit): " << "\n";

        std::getline(std::cin, filename);

        file.open(filename);

        if (file.is_open()) {
            try {
                create_regression_input(parse_csv(filename));
                break;
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing file " << filename << ": " << e.what() << "\n";
            }
            catch (...) {
                std::cerr << "An unknown error occurred while processing file " << filename << "\n";
            }
        }
        else if (filename == "x") {

            // User wants to exit, so exit
            std::cout << "Exiting..." << "\n";
            exit(0);
        }
        else if (filename == "") {
            try {
                create_regression_input(parse_csv("C://Users//lubomir.balaz//Desktop//Projekty 2023//EvoRegr++//data//TestDataSpan.csv"));
                break;
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing file " << filename << ": " << e.what() << "\n";
            }
            catch (...) {
                std::cerr << "An unknown error occurred while processing file " << filename << "\n";
            }
        }
        else {
            // File failed to open, print an error message and try again
            std::cerr << "Failed to open file " << filename << ", please try again.\n";
        }
        std::cout << "\n" << "File loaded... " << "\n";
    }
}

/**
 * The function `create_regression_input` takes in a tuple containing the number of rows, number of
 * columns, and a vector of data, and creates a predictor matrix and a target vector for regression
 * analysis.
 *
 * @param input The input parameter is a tuple containing three elements:
 */
void EvoAPI::create_regression_input(std::tuple<int, int, std::vector<double>> input) {

    std::vector<double> data = std::get<2>(input);

    //input matrix rows (with header)
    int m_input{ std::get<0>(input) };
    //input matrix columns (with target column)
    int n_input{ std::get<1>(input) };

    // predictor matrix column count (n_input - 1 (because of target column) + 1 + interaction columns)
    int n_output = n_input + interaction_cols;

    // + 1 because of header row
    int m_output = m_input + 1;

    // mark y column indexed from 0 (is last every time)
    int target_col_index = n_input - 1;

    // initialize predictors matrix to matrix of ones because of x0 and interaction columns
    x = Eigen::MatrixXd::Ones(m_output, n_output);
    y.resize(m_output, 1);

    for (int row = 0; row < m_output; ++row) {
        for (int col = 0; col < n_input; ++col) {
            // last column is always Y or in other words regressant, dependant variable
            if (col == target_col_index) {
                y(row, 0) = data[col + n_input * row];
            }
            // fil predictors (first is x0 column of 1, last are interaction filled default to 1 too - but they are able to mutate)
            if (col < target_col_index) {
                x(row, col + 1) = data[col + n_input * row];
            }
        }
    }
}

/**
 * The `predict` function performs a genetic algorithm to generate and evaluate a population of
 * individuals for a specified number of generations.
 */
void EvoAPI::predict() {

    std::cout << "Prediction started..." << "\n\n";

    // random engines for parallel loops
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines = create_random_engines(12346, omp_get_max_threads());

    // EvoIndividual containers
    std::vector<EvoIndividual> generation(0);
    std::vector<EvoIndividual> past_generation(0);

#pragma omp declare reduction(merge_individuals : std::vector<EvoIndividual> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = omp_orig)

    for (int gen_index = 0; gen_index < generation_count_limit; gen_index++) {

#pragma omp parallel for reduction (merge_individuals : generation) schedule(dynamic)
        for (int entity_index = 0; entity_index < generation_size_limit; entity_index++) {

            generation.reserve(generation_size_limit);

            EvoIndividual newborn;

            if (gen_index == 0) {
                //generate random individual
                newborn = Factory::getRandomEvoIndividual(y.rows(), x.cols(), random_engines[omp_get_thread_num()]);
            }
            else {
                //crossover & mutation [vector sex]
                newborn = Reproduction::reproduction(
                    Selection::tournament_selection(past_generation, random_engines[omp_get_thread_num()]),
                    x.cols(),
                    x.rows(),
                    random_engines[omp_get_thread_num()]
                );
            }

            newborn.evaluate(
                FitnessEvaluator::get_fitness(
                    Transform::data_transformation_robust(
                        x,
                        y,
                        newborn
                    )
                )
            );

            generation.push_back(std::move(newborn));
        }

        //new generation become old generation
        past_generation = std::move(generation);

        generation_postprocessing(past_generation, gen_index);
    }
}

void EvoAPI::show_result() {
    titan_postprocessing();
    show_regression_summary();
    show_titan_history();
    show_regression_coefficients();
    show_genotype();
    show_formula();
}

/**
 * The function "generation_postprocessing" takes a vector of EvoIndividual objects and an integer
 * representing the generation index, and performs post-processing tasks on the generation data.
 *
 * @param generation A vector of EvoIndividual objects representing a generation of individuals.
 * @param generation_index The generation index is an integer value that represents the index or number
 * of the current generation being processed. It is used to keep track of the progress of the
 * evolutionary algorithm.
 */
void EvoAPI::generation_postprocessing(std::vector<EvoIndividual> const& generation, int generation_index) {
    // ordered set of fitnesses
    std::set<double> generation_fitness;

    for (auto& individual : generation) {
        titan_evaluation(individual, generation_index);
        generation_fitness.insert(individual.fitness);
    }

    // create fitness metrics
    process_generation_fitness(generation_fitness);
}

/**
 * The function sets a new titan and adds its fitness and generation index to the titan history.
 *
 * @param titan The "titan" parameter is an object of type EvoIndividual, which represents a specific
 * individual in the evolutionary algorithm. It contains information about the individual's fitness and
 * other characteristics.
 * @param generation_index The generation index represents the index or number of the current
 * generation in the evolutionary algorithm. It is used to keep track of the progress and history of
 * the algorithm.
 */
void EvoAPI::setTitan(EvoIndividual titan, int generation_index) {
    this->titan = titan;
    this->titan_history.push_back(titan.fitness);
    this->titan_history.push_back(generation_index);
}

/**
 * The function `titan_evaluation` compares the fitness of a participant with the fitness of the
 * current titan and updates the titan if the participant has a lower fitness.
 *
 * @param participant The participant parameter is an object of the EvoIndividual class, which
 * represents an individual in the evolutionary algorithm. It contains information about the
 * individual's fitness and other attributes.
 * @param generation_index The generation index is an integer value that represents the current
 * generation of the evolutionary algorithm. It is used to keep track of the progress of the algorithm
 * and can be used for various purposes, such as logging or analysis.
 */
void EvoAPI::titan_evaluation(EvoIndividual participant, int generation_index) {
    if (participant.fitness < titan.fitness) setTitan(participant, generation_index);
}

/**
 * The function creates a vector of random engines using the Xoshiro256Plus algorithm, with each engine
 * having a unique seed based on the master random engine.
 *
 * @param seed The seed is a 64-bit unsigned integer used to initialize the master random engine. It
 * determines the starting point of the random number sequence.
 * @param count The parameter "count" represents the number of random engines to create.
 *
 * @return a vector of XoshiroCpp::Xoshiro256Plus random engines.
 */
std::vector<XoshiroCpp::Xoshiro256Plus> EvoAPI::create_random_engines(std::uint64_t seed, int count) {

    XoshiroCpp::Xoshiro256Plus master_random_engine(seed);
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines;

    //for each possible thread create its random engine with unique seed
    for (int i = 0;i < count;i++) {
        master_random_engine.longJump();
        random_engines.emplace_back(master_random_engine.serialize());
    }

    return random_engines;
}

/**
 * The function `titan_postprocessing()` performs data transformation and regression analysis on the
 * Titan dataset. Saves more detailed result.
 */
void EvoAPI::titan_postprocessing() {
    // data without outliers
    titan_robust_dataset = Transform::data_transformation_robust(x, y, titan);
    // data witho outliers
    titan_nonrobust_dataset = Transform::data_transformation_nonrobust(x, y, titan);
    // regression result
    titan_result = solve_system_by_ldlt_detailed(titan_robust_dataset.predictor, titan_robust_dataset.target);
}

/**
 * The function generates a summary matrix for a regression result. The matrix includes the original target values,
 * the predicted values, the difference between the original and predicted values, and the percentage difference.
 *
 * @param result The result is a constant reference to a RegressionDetailedResult object, which contains the regression results.
 * @param original_x The original_x is an Eigen::MatrixXd object representing the original predictor data.
 * @param original_y The original_y is an Eigen::VectorXd object representing the original target data.
 *
 * @return an Eigen::MatrixXd object representing the summary matrix. Each row corresponds to a data point, and the columns are:
 *         - Column 0: Original target values
 *         - Column 1: Predicted target values
 *         - Column 2: Difference between original and predicted values
 *         - Column 3: Percentage difference between original and predicted values
 */
Eigen::MatrixXd EvoAPI::get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd original_x, Eigen::VectorXd original_y) {
    Eigen::MatrixXd regression_result_matrix(original_x.rows(), 4);

    // get target for coefficients not disturbed by outliers
    Eigen::VectorXd prediction = original_x * result.theta;

    // transform target back to original values for presentation p
    titan.y_transformer_chromosome.at(0).transformBack(original_y);
    titan.y_transformer_chromosome.at(0).transformBack(prediction);

    //assembly result matrix
    regression_result_matrix.col(0) = original_y;
    regression_result_matrix.col(1) = prediction;
    regression_result_matrix.col(2) = regression_result_matrix.col(0) - regression_result_matrix.col(1);
    regression_result_matrix.col(3) = 100. - ((regression_result_matrix.col(1).array() / regression_result_matrix.col(0).array()) * 100);

    return regression_result_matrix;
}

/**
 * The function takes a set of fitness values for a generation, converts it to a vector, and calculates
 * various metrics such as the mean, median, and standard deviation.
 *
 * @param generation_fitness The parameter `generation_fitness` is a set of double values representing
 * the fitness values of a generation.
 */
void EvoAPI::process_generation_fitness(std::set<double> const& generation_fitness) {
    // get vector from set
    std::vector<double> generation_fitness_vector;
    generation_fitness_vector.reserve(generation_fitness.size());
    std::copy(generation_fitness.begin(), generation_fitness.end(), std::back_inserter(generation_fitness_vector));
    // calculate metrics
    generation_fitness_metrics.push_back(*generation_fitness.begin());
    generation_fitness_metrics.push_back(DescriptiveStatistics::geometric_mean(generation_fitness_vector));
    generation_fitness_metrics.push_back(DescriptiveStatistics::mean(generation_fitness_vector));
    generation_fitness_metrics.push_back(DescriptiveStatistics::median(generation_fitness_vector));
    generation_fitness_metrics.push_back(DescriptiveStatistics::standard_deviation(generation_fitness_vector));
}

void EvoAPI::show_regression_summary() {
    // get matrix with regression summary
    Eigen::MatrixXd regression_result_matrix = get_regression_summary_matrix(titan_result, titan_robust_dataset.predictor, titan_robust_dataset.target);
    //plot it
    Plotter<double> plt = Plotter(
        regression_result_matrix.data(),
        "Regression result summary",
        { "Target", "Prediction", "Difference", "Percentage difference" },
        149,
        regression_result_matrix.size(),
        DataArrangement::ColumnMajor
    );
    plt.print_table();
};

void EvoAPI::show_titan_history() {
    Plotter<double> plt = Plotter(
        titan_history.data(),
        "Best individual history",
        { "Fitness", "Generation" },
        149,
        titan_history.size(),
        DataArrangement::RowMajor
    );
    plt.print_table();
};

void EvoAPI::show_regression_coefficients() {
    Plotter<double> plt = Plotter(
        titan_result.theta.data(),
        "Regression coefficients",
        { "Coefficients" },
        149,
        titan_result.theta.size(),
        DataArrangement::ColumnMajor
    );
    plt.print_table();
};

void EvoAPI::show_genotype() {

    Plotter<std::string> plt = Plotter(
        titan.merge_chromosome_to_string_vector().data(),
        "Merge chromosome",
        { "Alleles" },
        149,
        titan.merge_chromosome_to_string_vector().size(),
        DataArrangement::RowMajor
    );
    plt.print_table();

    plt = Plotter(
        titan.transform_predictor_chromosome_to_string_vector().data(),
        "Transform predictor chromosome",
        { "Alleles" },
        149,
        titan.transform_predictor_chromosome_to_string_vector().size(),
        DataArrangement::RowMajor
    );
    plt.print_table();

    plt = Plotter(
        titan.transform_target_chromosome_to_string_vector().data(),
        "Transform target chromosome",
        { "Alleles" },
        149,
        titan.transform_target_chromosome_to_string_vector().size(),
        DataArrangement::RowMajor
    );
    plt.print_table();

    plt = Plotter(
        titan.robust_chromosome_to_string_vector().data(),
        "Robust chromosome",
        { "Alleles" },
        149,
        titan.robust_chromosome_to_string_vector().size(),
        DataArrangement::RowMajor
    );
    plt.print_table();
};

void EvoAPI::show_formula() {
    std::vector<std::string> formula{ titan.to_math_formula() };
    Plotter<std::string> plt = Plotter(
        formula.data(),
        "Evo-regression formula",
        { "Formula" },
        149,
        formula.size(),
        DataArrangement::RowMajor
    );
    plt.print_table();
};

