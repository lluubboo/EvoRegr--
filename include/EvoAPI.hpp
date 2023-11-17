#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"

class EvoAPI {

    int generation_size_limit, generation_count_limit, interaction_cols;

    std::string filename;

    EvoIndividual titan;

    Eigen::MatrixXd x, y;
    Transform::EvoDataSet titan_robust_dataset, titan_nonrobust_dataset;
    RegressionDetailedResult titan_result;

    std::vector<std::vector<EvoIndividual>> population;

    std::vector<double> titan_history;
    std::vector<double> generation_fitness_metrics;

    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);

    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd original_x, Eigen::VectorXd original_y);

    void setTitan(EvoIndividual, int);
    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    void titan_evaluation(EvoIndividual participant, int generation_index);
    void titan_postprocessing();
    void process_generation_fitness(std::set<double> const& generation_fitness);
    void generation_postprocessing(std::vector<EvoIndividual> const& generation, int generation_index);

    void show_regression_summary();
    void show_titan_history();
    void show_regression_coefficients();
    void show_genotype();
    void show_formula();

public:

    EvoAPI(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols);
    void load_file();
    void predict();
    void show_result();
};