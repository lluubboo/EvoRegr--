#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include <spdlog/spdlog.h>
#include <functional>
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"

struct EvoRegressionInput {
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    int mutation_rate;
    int generation_size_limit;
    int generation_count_limit;
    int island_id;
    int island_count;
    std::function<RegressionSimpleResult(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver;
};

class EvoAPI {

    // logger
    static std::shared_ptr<spdlog::logger> logger;

    // 
    static std::mutex population_mutex;

    // algorithm boundary conditions
    int generation_size_limit, generation_count_limit, interaction_cols, mutation_rate;

    // inputs
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

    void create_regression_input(std::tuple<int, std::vector<double>>);

    static EvoIndividual run_island(EvoRegressionInput, std::vector<EvoIndividual>&, XoshiroCpp::Xoshiro256Plus&);

    // concurrent random engines
    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);

    // fitness & generation postprocessing
    void setTitan(EvoIndividual, int);
    void titan_evaluation(EvoIndividual participant, int generation_index);
    void process_generation_fitness(std::set<double> const& generation_fitness);

    // final postprocessing 
    void titan_postprocessing();
    void generation_postprocessing(std::vector<EvoIndividual> const& generation, int generation_index);

    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd const& original_x, Eigen::VectorXd original_y);

    std::string get_regression_summary_table();
    std::string get_regression_result_table();
    std::string get_titan_history_table();
    std::string get_regression_coefficients_table();
    std::string get_genotype_table();
    std::string get_formula_table();
    std::string get_result_metrics_table();

public:

    EvoAPI();

    void create_report_file(std::string const& prefix);
    void set_solver(std::string const& solver_name);
    void reset_api_for_another_calculation();
    void load_file(const std::string& filename);
    void predict();
    void batch_predict();
    void log_result();
    void init_logger();
    void set_boundary_conditions(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols, unsigned int mutation_rate);

    bool is_ready_to_predict();
};