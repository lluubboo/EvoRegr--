#pragma once
#include <Eigen/Dense>

struct RegressionDetailedResult {
    Eigen::VectorXd theta;
    Eigen::VectorXd predicton;
    Eigen::VectorXd residuals;
    Eigen::VectorXd residuals_regression;
    Eigen::VectorXd residuals_total;
    Eigen::VectorXd residuals_squared;
    Eigen::VectorXd residuals_regression_squared;
    Eigen::VectorXd residuals_total_squared;
    Eigen::VectorXd percentage_error;
    double sum_squares_errors;
    double sum_squares_regression;
    double sum_squares_total;
    double mean_sum_squares_errors;
    double variance;
    double standard_deviation;
    double rmse;
    double rsquared;
    double rsquaredadj;
    bool isUsable;
};

struct RegressionSimpleResult {
    Eigen::VectorXd coefficients;
    Eigen::VectorXd residuals;
    Eigen::VectorXd prediction;
    double sum_squares_errors;
    bool isUsable;
};

RegressionDetailedResult solve_system_by_ldlt_detailed(Eigen::MatrixXd const&, Eigen::VectorXd const&);
RegressionSimpleResult solve_system_by_ldlt_simple(Eigen::MatrixXd const&, Eigen::VectorXd const&);

struct LLTSolver {
    RegressionSimpleResult operator()(Eigen::MatrixXd const&, Eigen::VectorXd const&) const;
};

struct LDLTSolver {
    RegressionSimpleResult operator()(Eigen::MatrixXd const&, Eigen::VectorXd const&) const;
};

struct ColPivHouseholderQrSolver {
    RegressionSimpleResult operator()(Eigen::MatrixXd const&, Eigen::VectorXd const&) const;
};