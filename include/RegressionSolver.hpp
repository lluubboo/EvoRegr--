#pragma once
#include <Eigen/Dense>
#include "EvoDataSet.hpp"
#include "EvoLibrary.hpp"
#include "BoundaryConditions.hpp"

struct RegressionDataset {
    Eigen::MatrixXd training_predictors;
    Eigen::VectorXd training_target;
    Eigen::MatrixXd test_predictors;
    Eigen::VectorXd test_target;
};

struct RegressionDetailedResult {
    Eigen::VectorXd theta;

    RegressionDetailedResult() : theta() {}
    RegressionDetailedResult(Eigen::VectorXd vector) : theta(vector) {}
};

RegressionDetailedResult solve_system_detailed(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&);

struct LLTSolver {
    double operator()(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&) const;
};

struct LDLTSolver {
    double operator()(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&) const;
};

struct ColPivHouseholderQrSolver {
    double operator()(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&) const;
};

bool validate_coefficients(Eigen::VectorXd const&);

double calculate_fitness(Eigen::VectorXd const&, Eigen::Block<Eigen::MatrixXd>&, Eigen::VectorBlock<Eigen::VectorXd>&);