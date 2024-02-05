#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include "EvoIndividual.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"
#include "EvoDataset.hpp"
#include "RegressionSolver.hpp"
#include "Timer.hpp"
#include "LRUCache.hpp"

void test_to_string_method(std::vector<EvoIndividual> population) {
    {
        volatile int dummy = 0;

        Timer timer(population.size(), "TO STRING CODE");
        for (const auto& individual : population) {
            dummy += individual.to_string_code().size();
        }
    }
}

void test_transforamtion(std::vector<EvoRegression::EvoDataSet> datasets, std::vector<EvoIndividual> population) {
    {
        auto datasets_copy = datasets;
        volatile int dummy = 0;

        Timer timer(population.size(), "TRANSFORM DATASET");
        int i = 0;
        for (const auto& individual : population) {
            dummy += Transform::transform_dataset(datasets_copy[i++], individual, true).predictor.row(0)(0);
        }
    }
}

void test_solve(std::vector<EvoRegression::EvoDataSet> datasets, std::vector<EvoIndividual> population) {
    {
        //solver
        std::function<double(EvoRegression::EvoDataSet const&)> solver = ColPivHouseholderQrSolver();

        volatile double dummy = 0;
        int i = 0;

        Timer timer(population.size(), "SOLVE REGRESSION");
        for (const auto& individual : population) {
            dummy += EvoMath::get_fitness<std::function<double(EvoRegression::EvoDataSet const& dataset)>>(datasets[i++], solver);
        }
    }
}

void test_trans_and_comp(std::vector<EvoRegression::EvoDataSet> datasets, std::vector<EvoIndividual> population) {
    {
        //solver
        std::function<double(EvoRegression::EvoDataSet const&)> solver = ColPivHouseholderQrSolver();

        volatile double dummy = 0;
        int i = 0;

        Timer timer(population.size(), "COMPUTE");
        for (const auto& individual : population) {
            dummy += EvoMath::get_fitness<std::function<double(EvoRegression::EvoDataSet const& dataset)>>(Transform::transform_dataset(datasets[i++], individual, true), solver);
        }
    }
}

void test_get_fitness_chache(std::vector<EvoRegression::EvoDataSet> datasets, std::vector<EvoIndividual> population) {
    {
        //solver
        std::function<double(EvoRegression::EvoDataSet const&)> solver = ColPivHouseholderQrSolver();

        //cache
        LRUCache<std::string, double> cache(population.size());

        //fill cache
        int i = 0;
        for (const auto& individual : population) {
            cache.put(individual.to_string_code(), EvoMath::get_fitness<std::function<double(EvoRegression::EvoDataSet const& dataset)>>(Transform::transform_dataset(datasets[i++], individual, true), solver));
        }

        {
            volatile double dummy = 0;

            Timer timer(population.size(), "GET FROM CACHE");
            for (const auto& individual : population) {
                dummy += cache.get(individual.to_string_code()).value();
            }
        }
    }
}

Eigen::VectorXd gradescent(EvoRegression::EvoDataSet const& dataset) {

    // new theta = old theta - alpha * (X^T * (X * theta - y)) / m

    // boundaries
    int iterations = 1e3;
    double alpha = 0.01;
    double error_norm = 0;
    double old_error_norm = std::numeric_limits<double>::max();
    double threshold = 1e-6;

    // memmory
    Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(dataset.predictor.cols());
    Eigen::VectorXd errors(dataset.predictor.rows());

    for (int i = 0; i < iterations; i++) {

        // compute errors 
        errors.noalias() = dataset.predictor * coefficients - dataset.target;

        // compute coefficients
        coefficients.noalias() -= alpha * (dataset.predictor.transpose() * errors);

        // check error threshold
        error_norm = errors.norm();
        if (abs(old_error_norm - error_norm) < threshold) return coefficients;
        old_error_norm = error_norm;
    }

    return coefficients;
}

void test_gradescent_vs_colpivhoushodler(std::vector<EvoRegression::EvoDataSet> datasets) {

    Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(datasets[0].predictor.cols());

    {
        volatile double dummy = 0;
        Timer timer(datasets.size(), "ColPivHouseholderQrSolver");
        for (auto& dataset : datasets) {
            coefficients = (dataset.predictor.transpose() * dataset.predictor).colPivHouseholderQr().solve(dataset.predictor.transpose() * dataset.target);
            dummy += coefficients(0);
        }
        std::cout << dummy << std::endl;
    }

    coefficients = Eigen::VectorXd::Ones(datasets[0].predictor.cols());

    {
        volatile double dummy = 0;
        Timer timer(datasets.size(), "Gradient descent");
        for (auto& dataset : datasets) {
            coefficients = gradescent(dataset);
            dummy += coefficients(0);
        }
        std::cout << dummy << std::endl;
    }
}

int main() {
    std::cout << "Benchmark started... "  << std::endl;
    
    //population size
    int pop_size = 5000;

    std::cout << "Population size... " << pop_size << std::endl;

    //create random seed for master random engine
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    XoshiroCpp::Xoshiro256Plus master_random_engine(seed);

    std::cout << "Random engine initialized... " << std::endl;

    //solver
    std::function<double(EvoRegression::EvoDataSet const&)> solver = ColPivHouseholderQrSolver();

    std::cout << "Solver engine initialized... " << std::endl;

    // datasets
    std::vector<EvoRegression::EvoDataSet> datasets(pop_size);
    std::for_each(datasets.begin(), datasets.end(), [&](EvoRegression::EvoDataSet& dataset) {dataset = EvoRegression::EvoDataSet(Eigen::MatrixXd::Random(300, 20), Eigen::VectorXd::Random(300, 20));});

    std::cout << "Datasets initialized... " << std::endl;
    
    // cache
    LRUCache<std::string, EvoIndividual> cache(pop_size);

    std::cout << "Cache initialized... " << std::endl;

    //create random population
    EvoBoundaryConditions boundary_conditions = EvoBoundaryConditions(200, 20, 20, 20, 20, 20, 20, 20, 20, 20);
    std::vector<EvoIndividual> population = Factory::generate_random_generation(boundary_conditions, datasets[0], master_random_engine, solver);

    std::cout << "Population initialized... " << std::endl;

    //*********************************************************************************** START TESTS ***********************************************************************************
    
    test_to_string_method(population);

    test_transforamtion(datasets, population);

    test_solve(datasets, population);

    test_trans_and_comp(datasets, population);

    test_get_fitness_chache(datasets, population);

    test_gradescent_vs_colpivhoushodler(datasets);

    std::cout << "Press ENTER to exit...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return 0;
}

