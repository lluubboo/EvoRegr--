#pragma once
#include <Eigen/Dense>

struct RegressionResult {
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
    double rmse;
    double rsquared;
    double rsquaredadj;
    bool isUsable;
};

RegressionResult solve_system_by_ldlt_detailed(Eigen::MatrixXd const&, Eigen::VectorXd const&);
RegressionResult solve_system_by_ldlt_simple(Eigen::MatrixXd const&, Eigen::VectorXd const&);
