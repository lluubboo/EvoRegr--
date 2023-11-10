#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include "RegressionSolver.hpp"
#include "EvoIndividual.hpp"
#include "XoshiroCpp.hpp"

struct EvoDataSet {
    Eigen::MatrixXd predictor;
    Eigen::MatrixXd target;
};

class EvoAPI {

public:

    EvoAPI(const std::string&);
    void setBoundaryConditions(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols);
    void predict();
    void show_me_result();

private:

    //boundary conditions
    int generation_size_limit, generation_count_limit, interaction_cols;

    // best individual
    EvoIndividual titan;

    //absolute path to datafile
    std::string filename;

    // original predictor
    Eigen::MatrixXd x;

    // original target
    Eigen::MatrixXd y;

    // population storage (0=old generation, 1=new generation)
    std::vector<std::vector<EvoIndividual>> population;

    // params usefull for statistics & plotting
    std::vector<double> fitness_history;
    std::vector<double> titan_history;
    std::vector<float> generation_fitness_mean_history;
    std::vector<float> generation_fitness_median_history;
    std::vector<float> generation_fitness_standard_deviation_history;

    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);
    std::vector<EvoIndividual> create_random_generation(XoshiroCpp::Xoshiro256Plus&, int size);
    EvoDataSet data_transformation_cacheless(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual&);
    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd original_x, Eigen::VectorXd original_y);
    Eigen::MatrixXd get_regression_history_summary(std::vector<double> fitness_history, std::vector<double> titan_history);
    void setTitan(EvoIndividual, int);
    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    void titan_evaluation(EvoIndividual participant, int generation_index);
    void report_generation_summary(std::vector<double> const& generation_fitness);
};