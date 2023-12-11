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

RegressionDetailedResult solve_system_detailed(Eigen::MatrixXd const&, Eigen::VectorXd const&);

struct LLTSolver {
    double operator()(Eigen::MatrixXd const&, Eigen::VectorXd const&) const;
};

struct LDLTSolver {
    double operator()(Eigen::MatrixXd const&, Eigen::VectorXd const&) const;
};

struct ColPivHouseholderQrSolver {
    double operator()(Eigen::MatrixXd const&, Eigen::VectorXd const&) const;
};