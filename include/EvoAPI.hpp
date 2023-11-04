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
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;

    std::vector<std::vector<EvoIndividual>> population;
    std::vector<double> fitness_history;
    std::vector<double> titan_history;

    EvoAPI(const std::string&);
    void setBoundaryConditions(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols);
    void predict();
    void show_me_result();
    void profiler();

private:
    int generation_size_limit;
    int generation_count_limit;
    int interaction_cols;
    EvoIndividual titan;
    std::string filename;

    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);
    std::vector<EvoIndividual> create_random_generation(XoshiroCpp::Xoshiro256Plus&, int size);
    EvoDataSet data_transformation_cacheless(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual&);
    Eigen::MatrixXd get_regression_summary_matrix(RegressionResult const& result);
    Eigen::MatrixXd get_regression_history_summary(std::vector<double> fitness_history, std::vector<double> titan_history);
    void setTitan(EvoIndividual, int);
    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    void titan_evaluation(EvoIndividual participant, int generation_index);
};