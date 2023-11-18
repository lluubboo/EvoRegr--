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

    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);

    void setTitan(EvoIndividual, int);
    void titan_evaluation(EvoIndividual participant, int generation_index);
    void process_generation_fitness(std::set<double> const& generation_fitness);

    // postprocessing after main loop
    void titan_postprocessing();
    void generation_postprocessing(std::vector<EvoIndividual> const& generation, int generation_index);

    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd original_x, Eigen::VectorXd original_y);

    void print_regression_summary();
    void print_titan_history();
    void print_regression_coefficients();
    void print_genotype();
    void print_formula();

    void show_plots();
    
public:

    EvoAPI(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols);
    void load_file();
    void predict();
    void show_result();
};