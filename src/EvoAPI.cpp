#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip> 
#include "EvoAPI.hpp"
#include "IOTools.hpp"
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"
#include "Stats.hpp"
#include "omp.h"

EvoAPI::EvoAPI(const std::string& filename) {
    this->filename = filename;
}

void EvoAPI::create_regression_input(std::tuple<int, int, std::vector<double>> input) {

    int m{ std::get<0>(input) };
    int n{ std::get<1>(input) };

    int n_modified = n + interaction_cols - 1; // we exluded target from columns count
    int target_col_index = n - 1; // indexed from 0

    std::vector<double> data = std::get<2>(input);

    x.resize(m, n_modified);
    y.resize(m, 1);

    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n_modified; ++col)
        {

            // last column is always Y or in other words regressant, dependant variable
            if (col == target_col_index) {
                y(row, 0) = data[col + n * row];
            }

            // fill predictor
            if (col < target_col_index) {
                x(row, col) = data[col + n * row];

            }
            else {
                x(row, col) = 1;
            }
        }
    }
    std::cout << "\n" << "Data loading was succesful..." << "\n";
}

void EvoAPI::setBoundaryConditions(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols) {

    this->generation_size_limit = generation_size_limit;
    this->generation_count_limit = generation_count_limit;
    this->interaction_cols = interaction_cols;

    // initialize generation vector with given capacity
    //we want to avoid memory realocation
    std::vector<EvoIndividual> generation{ 0 };
    generation.reserve(generation_size_limit);
    population = std::vector{ 2, generation };

    //silently load data 
    create_regression_input(parse_csv(filename));
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

std::vector<EvoIndividual> EvoAPI::create_random_generation(XoshiroCpp::Xoshiro256Plus& random_engine, int size) {

    std::vector<EvoIndividual> generation;
    generation.reserve(size);

    // add first individual and make him titan
    generation.push_back(Factory::getRandomEvoIndividual(x, y, random_engine));
    setTitan(generation.back(), 0);

    for (int i = 1; i < size; i++)
    {
        EvoIndividual individual = Factory::getRandomEvoIndividual(x, y, random_engine);
        titan_evaluation(individual, 0);
        generation.push_back(individual);
    }

    return generation;
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

void EvoAPI::predict() {

    std::cout << "Prediction started..." << "\n\n";

    //random engines for parallel loop
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines = create_random_engines(12345, omp_get_max_threads());

    //generation zero
    population[0] = create_random_generation(random_engines[0], generation_size_limit);

#pragma omp declare reduction(merge_individuals : std::vector<EvoIndividual> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = omp_orig)
    for (int gen_index = 1; gen_index < generation_count_limit; gen_index++) {

        if (gen_index % 100 == 0) std::cout << "Creating generation " << gen_index << "..." << std::endl;

        std::vector<EvoIndividual> generation = population[1];
        std::vector<EvoIndividual> past_generation = population[0];

#pragma omp parallel for shared(random_engines, past_generation, x, y) reduction (merge_individuals : generation) 
        for (int entity_index = 0; entity_index < generation_size_limit; entity_index++) {

            //selection
            std::vector<EvoIndividual> parents = Selection::tournament_selection(past_generation, random_engines[omp_get_thread_num()]);

            //crossover & mutation 
            EvoIndividual individual = Reproduction::reproduction(parents[0], parents[1], x.cols(), x.rows(), random_engines[omp_get_thread_num()]);

            //transform data
            EvoDataSet evo_data = data_transformation_cacheless(x, y, individual);

            // calculate fitness
            individual.fitness = FitnessEvaluator::get_fitness(evo_data.predictor, evo_data.target);

            //mark titan
            titan_evaluation(individual, gen_index);

            generation.push_back(individual);
        }
        past_generation = generation; //new generation become old generation
    }
}

void EvoAPI::show_me_result() {

    Eigen::MatrixXd predictor = x;
    Eigen::VectorXd target = y;

    Transform::half_predictor_transform(predictor, titan);
    Transform::half_target_transform(target, titan);

    RegressionResult result = solve_system_by_ldlt_detailed(predictor, target);

    titan.y_transformer_chromosome.at(0).transformBack(target);
    titan.y_transformer_chromosome.at(0).transformBack(result.predicton);

    std::cout << "\n\n";
    std::cout << "********************************************REGRESSION RESULT SUMMARY******************************************\n\n";
    std::cout << get_regression_summary_matrix(result) << "\n\n";
    std::cout << "************************************************REGRESSION HISTORY*********************************************\n\n";
    std::cout << get_regression_history_summary(fitness_history, titan_history) << "\n\n";
    std::cout << "***************************************************TITAN GENOME************************************************\n\n";
    std::cout  << titan.to_string();
    std::cout << "***********************************************REGRESSION SUMMARY**********************************************\n\n";
    std::cout << "Regression coefficients: " << result.theta.transpose();
    std::cout << "\n\n";
    std::cout << "Residuals mean: " << result.residuals.mean() << " median: " << DescriptiveStatistics::median(result.residuals.data(), result.residuals.rows());
    std::cout << "\n\n";
    std::cout << "R-squared: " << result.rsquared << " R-squared Adj: " << result.rsquaredadj << " RMSE: " << result.rmse;
    std::cout << "\n\n";
}

EvoDataSet EvoAPI::data_transformation_cacheless(Eigen::MatrixXd predictor, Eigen::VectorXd target, EvoIndividual& individual) {
    EvoDataSet dataset{};
    Transform::full_predictor_transform(predictor, individual);
    Transform::full_target_transform(target, individual);
    dataset.predictor = predictor;
    dataset.target = target;
    return dataset;
};

Eigen::MatrixXd EvoAPI::get_regression_summary_matrix(RegressionResult const& result) {
    Eigen::MatrixXd summary_regression(result.predicton.rows(), 4);
    summary_regression.col(0) = y;
    summary_regression.col(1) = result.predicton;
    summary_regression.col(2) = y - result.predicton;
    summary_regression.col(3) = 100. - ((result.predicton.array() / y.array()) * 100);;
    return summary_regression;
}

Eigen::MatrixXd EvoAPI::get_regression_history_summary(std::vector<double> fitness_history, std::vector<double> titan_history) {
    Eigen::MatrixXd regression_history_summary(fitness_history.size(), 2);
    regression_history_summary.col(0) = Eigen::Map<Eigen::VectorXd>(fitness_history.data(), fitness_history.size());
    regression_history_summary.col(1) = Eigen::Map<Eigen::VectorXd>(titan_history.data(), fitness_history.size());
    return regression_history_summary;
}

void EvoAPI::profiler() {

    // random engines
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines = create_random_engines(12345, omp_get_max_threads());
    // test generation
    std::vector<EvoIndividual> test_generation = create_random_generation(random_engines[0], 500);

    int sample_size = 100;
    double result;

    std::cout << "\n\n" << "Benchmark starting....: \n";
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int j = 0; j < sample_size; j++) {

        for (int i = 0;i < 200000;i++) {
            // add function here
        }

    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    result = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.) / sample_size;
    std::cout << "Mean time difference = " << result << " [s]" << "\n\n";
}
