#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include <spdlog/spdlog.h>
#include <functional>
#include <future>
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"

//TODO: delete class, move plots to unique class, vykuchat evoAPI

class EvoAPI {

    // algorithm boundary conditions
    size_t generation_size_limit, generation_count_limit, interaction_cols, mutation_rate, island_count, migration_ratio, migration_interval;
    size_t global_generation_size_limit, migrants_count;

    // preprocessed input data
    Eigen::MatrixXd x, y;

    // solver functor
    std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver;

    // titan 
    EvoIndividual titan;
    EvoRegression::EvoDataSet titan_robust_dataset, titan_nonrobust_dataset;
    RegressionDetailedResult titan_result;

    // data
    void create_regression_input(std::tuple<int, std::vector<double>>);
    EvoRegression::EvoDataSet get_dataset();

    // concurrent random engines 
    // TODO move to EvoLibrary
    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(int count);

    // fitness & generation postprocessing
    void setTitan(EvoIndividual);
    void titan_evaluation(EvoIndividual const& individual);

    // final postprocessing 
    void titan_postprocessing();

    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd const& original_x, Eigen::VectorXd original_y);

    std::string get_regression_summary_table();
    std::string get_regression_result_table();
    std::string get_regression_robust_result_table();
    std::string get_regression_coefficients_table();
    std::string get_genotype_table();
    std::string get_formula_table();
    std::string get_result_metrics_table();

public:

    EvoAPI();

    void set_solver(std::string const& solver_name);
    void load_file(const std::string& filename);
    void batch_predict();
    void log_result();
    void set_boundary_conditions(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols, unsigned int mutation_rate, unsigned int island_count, unsigned int migration_ratio, unsigned int migration_interval);
    bool is_ready_to_predict();
};