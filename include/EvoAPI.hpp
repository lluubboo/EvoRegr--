#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include <spdlog/spdlog.h>
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"

class EvoAPI {

    // file logger
    std::shared_ptr<spdlog::logger> logger;

    // algorithm boundary conditions
    int generation_size_limit, generation_count_limit, interaction_cols;

    // inputs
    std::string filename;
    Eigen::MatrixXd x, y;

    // titan 
    EvoIndividual titan;
    Transform::EvoDataSet titan_robust_dataset, titan_nonrobust_dataset;
    RegressionDetailedResult titan_result;

    // algorithm metrics
    std::vector<double> titan_history;
    std::vector<double> generation_fitness_metrics;

    void init_logger();
    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);

    void setTitan(EvoIndividual, int);
    void titan_evaluation(EvoIndividual participant, int generation_index);
    void process_generation_fitness(std::set<double> const& generation_fitness);

    // postprocessing after main loop
    void titan_postprocessing();
    void generation_postprocessing(std::vector<EvoIndividual> const& generation, int generation_index);

    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd original_x, Eigen::VectorXd original_y);

    std::string get_regression_summary_table();
    std::string get_regression_result_table();
    std::string get_titan_history_table();
    std::string get_regression_coefficients_table();
    std::string get_genotype_table();
    std::string get_formula_table();

public:

    EvoAPI(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols);
    void load_file();
    void predict();
    void show_result();
};