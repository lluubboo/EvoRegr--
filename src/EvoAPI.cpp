#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <chrono>
#include "EvoAPI.hpp"
#include "IOTools.hpp"
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"
#include "omp.h"

EvoAPI::EvoAPI(const std::string& filename) {
    create_regression_input(parse_csv(filename));
}

void EvoAPI::create_regression_input(std::tuple<int, int, std::vector<double>> input) {

    int m{ std::get<0>(input) };
    int n{ std::get<1>(input) };
    std::vector<double> data = std::get<2>(input);

    x.resize(m, n - 1);
    y.resize(m, 1);

    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            if (col == n - 1) {

                //last column is always Y or in other words regressant, dependant variable
                y(row, 0) = data[col + n * row];
            }
            else {
                x(row, col) = data[col + n * row];
            }
        }
    }
}

void EvoAPI::setBoundaryConditions(unsigned int generation_size_limit, unsigned int generation_count_limit) {

    this->generation_size_limit = generation_size_limit;
    this->generation_count_limit = generation_count_limit;

    // initialize generation vector with given capacity
    //we want to avoid memory realocation
    std::vector<EvoIndividual> generation{ 0 };
    generation.reserve(generation_size_limit);
    population = std::vector{ 2, generation };

    //reserve cache
    cache.reserve(1000000);
}

void EvoAPI::setTitan(EvoIndividual titan, int generation_index) {
    this->titan = titan;
    this->titan_history.push_back(generation_index);
    this->fitness_history.push_back(titan.fitness);
}

void EvoAPI::append_generation_zero(XoshiroCpp::Xoshiro256Plus& random_engine) {

    // add first individual and make him titan
    population[0].push_back(Factory::getRandomEvoIndividual(x, y, random_engine));
    titan = population[0].back();

    for (int i = 1; i < generation_size_limit; i++)
    {
        EvoIndividual individual = Factory::getRandomEvoIndividual(x, y, random_engine);
        if (individual.fitness < titan.fitness) setTitan(individual, 0);
        population[0].push_back(individual);
    }
}

void EvoAPI::predict() {

    const std::uint64_t seed = 12346;
    XoshiroCpp::Xoshiro256Plus master_random_engine(seed);
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines;

    append_generation_zero(master_random_engine); //generate random generation zero

    //for each possible thread create its random engine with unique seed
    for (int i = 0;i < omp_get_max_threads();i++) {
        master_random_engine.longJump();
        random_engines.emplace_back(master_random_engine.serialize());
    }

#pragma omp declare reduction(merge_individuals : std::vector<EvoIndividual> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())) initializer(omp_priv = omp_orig)
#pragma omp declare reduction(merge_cache : std::unordered_map<std::string, double> : omp_out.merge(omp_in)) initializer(omp_priv = omp_orig)
    for (int gen_index = 1; gen_index < generation_count_limit; gen_index++) {

        if (gen_index % 100 == 0) std::cout << "\n\n" << gen_index;

        std::vector<EvoIndividual> generation = population[1];
        std::vector<EvoIndividual> past_generation = population[0];
        std::unordered_map<std::string, double> local_cache;

#pragma omp parallel for shared(random_engines, past_generation, cache, x, y) reduction (merge_individuals : generation) reduction (merge_cache : local_cache)
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
            if (individual.fitness < titan.fitness) setTitan(individual, gen_index);

            generation.push_back(individual);
        }
        cache.merge(local_cache);
        past_generation = generation; //new generation become old generation
    }
}

void EvoAPI::showMeBest() {

    Eigen::MatrixXd predictor = x;
    Eigen::VectorXd target = y;

    Eigen::MatrixXd calculation_history_summary(fitness_history.size(), 2);
    Eigen::MatrixXd summary_regression(x.rows(), 4);

    int chromosomes_size = titan.merger_chromosome.size();

    //merge predictors
    for (int i = 0; i < chromosomes_size; i++)
    {
        titan.merger_chromosome.at(i).transform(predictor);
    }

    //transform predictors
    for (int i = 0; i < chromosomes_size; i++)
    {
        titan.x_transformer_chromosome.at(i).transform(predictor);
    }

    //transform target
    titan.y_transformer_chromosome.at(0).transformVector(target);

    RegressionResult result = solveSystemByLLT(predictor, target);

    titan.y_transformer_chromosome.at(0).transformBack(target);
    titan.y_transformer_chromosome.at(0).transformBack(result.predicton);

    summary_regression.col(0) = target;
    summary_regression.col(1) = result.predicton;
    summary_regression.col(2) = target - result.predicton;
    summary_regression.col(3) = 100 - ((summary_regression.col(1).array() / summary_regression.col(0).array()) * 100);

    calculation_history_summary.col(0) = Eigen::Map<Eigen::VectorXd>(fitness_history.data(), fitness_history.size());
    calculation_history_summary.col(1) = Eigen::Map<Eigen::VectorXd>(titan_history.data(), fitness_history.size());

    std::cout << "\n\n" << "Titan Y comparison:" << "\n\n" << summary_regression;
    std::cout << "\n\n" << "Titan history is:" << "\n\n" << calculation_history_summary;
    std::cout << "\n\n";
    std::cout << titan.to_string();
    std::cout << "\n\n" << "Coefficients: \n" << result.theta;
    std::cout << "\n\n" << "Cache size : \n" << cache.size();
}

EvoDataSet EvoAPI::data_transformation_cacheless(Eigen::MatrixXd predictor, Eigen::VectorXd target, EvoIndividual& individual) {
    EvoDataSet dataset{};
    Transform::half_predictor_transform(predictor, individual);
    Transform::robust_predictor_transform(predictor, individual);
    Transform::full_target_transform(target, individual);
    dataset.predictor = predictor;
    dataset.target = target;
    return dataset;
};
