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

    for (int row = 0; row < m_output; ++row){
        for (int col = 0; col < n_input; ++col){
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
                    data_transformation_cacheless(
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

void EvoAPI::show_me_result() {

    Eigen::MatrixXd predictor(x);
    Eigen::VectorXd target(y);
    Eigen::MatrixXd predictor_full(x);
    Eigen::VectorXd target_full(y);

    Transform::full_predictor_transform(predictor, titan);
    Transform::full_target_transform(target, titan);
    Transform::half_predictor_transform(predictor_full, titan);
    Transform::half_target_transform(target_full, titan);

    RegressionDetailedResult result = solve_system_by_ldlt_detailed(predictor, target);

    std::cout << "\n\n";
    std::cout << "********************************************REGRESSION RESULT SUMMARY******************************************\n\n";
    std::cout << get_regression_summary_matrix(result, predictor_full, target_full) << "\n\n";
    std::cout << "************************************************REGRESSION HISTORY*********************************************\n\n";
    std::cout << get_regression_history_summary(fitness_history, titan_history) << "\n\n";
    std::cout << "***************************************************TITAN GENOME************************************************\n\n";
    std::cout << titan.to_string();
    std::cout << "***********************************************REGRESSION SUMMARY**********************************************\n\n";
    std::cout << "Regression coefficients: " << result.theta.transpose();
    std::cout << "\n\n";
    std::cout << "Residuals mean: " << result.residuals.mean() << " median: " << DescriptiveStatistics::median(result.residuals.data(), result.residuals.rows());
    std::cout << " standard deviation: " << result.standard_deviation;
    std::cout << "\n\n";
    std::cout << "R-squared: " << result.rsquared << " R-squared Adj: " << result.rsquaredadj << " RMSE: " << result.rmse;
    std::cout << "\n\n";
}

void EvoAPI::generation_postprocessing(std::vector<EvoIndividual> const& generation, int generation_index) {

    // fitness container for statistics
    std::set<double> generation_fitness;

    for (auto& individual : generation) {

        //find & mark titan
        titan_evaluation(individual, generation_index);

        //save fitness for statistics
        generation_fitness.insert(individual.fitness);
    }

}

void EvoAPI::setTitan(EvoIndividual titan, int generation_index) {
    this->titan = titan;
    this->titan_history.push_back(generation_index);
    this->fitness_history.push_back(titan.fitness);
    std::cout << "Great! New titan with fitness " << titan.fitness << " was found ..." << "\n";
}

void EvoAPI::titan_evaluation(EvoIndividual participant, int generation_index) {
    if (participant.fitness < titan.fitness) setTitan(participant, generation_index);
}

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

Transform::EvoDataSet EvoAPI::data_transformation_cacheless(Eigen::MatrixXd predictor, Eigen::VectorXd target, EvoIndividual const& individual) {
    Transform::full_predictor_transform(predictor, individual);
    Transform::full_target_transform(target, individual);
    return { predictor, target };
};

Eigen::MatrixXd EvoAPI::get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd original_x, Eigen::VectorXd original_y) {
    Eigen::MatrixXd summary_regression(original_x.rows(), 4);
    Eigen::VectorXd prediction = original_x * result.theta;

    titan.y_transformer_chromosome.at(0).transformBack(original_y);
    titan.y_transformer_chromosome.at(0).transformBack(prediction);

    summary_regression.col(0) = original_y;
    summary_regression.col(1) = prediction;
    summary_regression.col(2) = summary_regression.col(0) - summary_regression.col(1);
    summary_regression.col(3) = 100. - ((summary_regression.col(1).array() / summary_regression.col(0).array()) * 100);
    return summary_regression;
}

Eigen::MatrixXd EvoAPI::get_regression_history_summary(std::vector<double> fitness_history, std::vector<double> titan_history) {
    Eigen::MatrixXd regression_history_summary(fitness_history.size(), 2);
    regression_history_summary.col(0) = Eigen::Map<Eigen::VectorXd>(fitness_history.data(), fitness_history.size());
    regression_history_summary.col(1) = Eigen::Map<Eigen::VectorXd>(titan_history.data(), fitness_history.size());
    return regression_history_summary;
}

void EvoAPI::report_generation_summary(std::set<double> const& generation_fitness) {
    std::vector<double> generation_fitness_vector(generation_fitness.begin(), generation_fitness.end());
    generation_fitness_median_history.push_back(DescriptiveStatistics::median(generation_fitness_vector));
    generation_fitness_standard_deviation_history.push_back(DescriptiveStatistics::standard_deviation(generation_fitness_vector));

    if (generation_fitness_vector.size() > 10) generation_fitness_vector.erase(generation_fitness_vector.begin() + 10, generation_fitness_vector.end());
    generation_fitness_mean10_history.push_back(DescriptiveStatistics::mean(generation_fitness_vector));
}

