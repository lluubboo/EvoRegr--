#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include <spdlog/spdlog.h>
#include <functional>
#include <future>
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"

struct EvoRegressionInput {
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    EvoPopulation& population;
    XoshiroCpp::Xoshiro256Plus& random_engine;
    std::function<RegressionSimpleResult(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver;
    const int mutation_rate;
    const int generation_size_limit;
    const int generation_count_limit;
    const int island_id;
    const int island_count;
};

struct IslandOutput {
    EvoIndividual best_individual;
    int island_id;
};

class EvoAPI {

    // logger
    static std::shared_ptr<spdlog::logger> logger;

    // algorithm boundary conditions
    int generation_size_limit, generation_count_limit, interaction_cols, mutation_rate;

    // preprocessed input data
    Eigen::MatrixXd x, y;

    // solver functor
    std::function<RegressionSimpleResult(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver;

    // titan 
    EvoIndividual titan;
    Transform::EvoDataSet titan_robust_dataset, titan_nonrobust_dataset;
    RegressionDetailedResult titan_result;

    // algorithm metrics
    std::vector<double> titan_history;
    std::vector<double> generation_fitness_metrics;

    // data
    void create_regression_input(std::tuple<int, std::vector<double>>);
    Transform::EvoDataSet get_dataset();

    // predictions
    static IslandOutput run_island(EvoRegressionInput);
    static IslandOutput run_island_async(EvoRegressionInput);
    static std::array<unsigned int, 2> get_island_borders(unsigned int island_id, unsigned int generation_size_limit) noexcept;

    // concurrent random engines
    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(int count);

    // fitness & generation postprocessing
    void setTitan(EvoIndividual);
    void titan_evaluation(EvoIndividual participant);

    // final postprocessing 
    void titan_postprocessing();

    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd const& original_x, Eigen::VectorXd original_y);

    std::string get_regression_summary_table();
    std::string get_regression_result_table();
    std::string get_regression_coefficients_table();
    std::string get_genotype_table();
    std::string get_formula_table();
    std::string get_result_metrics_table();

public:

    EvoAPI();

    void set_solver(std::string const& solver_name);
    void load_file(const std::string& filename);
    void predict();
    void batch_predict();
    void log_result();
    void init_logger();
    void set_boundary_conditions(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols, unsigned int mutation_rate);
    bool is_ready_to_predict();
};