#pragma once
#include <Eigen/Dense>
#include "EvoDataSet.hpp"
#include "EvoLibrary.hpp"

struct RegressionDataset {
    Eigen::MatrixXd training_predictors;
    Eigen::VectorXd training_target;
    Eigen::MatrixXd test_predictors;
    Eigen::VectorXd test_target;
};

struct RegressionDetailedResult {
    Eigen::VectorXd theta;
};

RegressionDetailedResult solve_system_detailed(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target, float regularizaton_parameter);

struct LLTSolver {
    double operator()(EvoRegression::EvoDataSet&, int test_ratio, float regularizaton_parameter) const;
};

struct LDLTSolver {
    double operator()(EvoRegression::EvoDataSet&, int test_ratio, float regularizaton_parameter) const;
};

struct ColPivHouseholderQrSolver {
    double operator()(EvoRegression::EvoDataSet&, int test_ratio, float regularizaton_parameter) const;
};

bool validate_coefficients(Eigen::VectorXd const&);

double calculate_fitness(Eigen::VectorXd const&, Eigen::Block<Eigen::MatrixXd>&, Eigen::VectorBlock<Eigen::VectorXd>&);