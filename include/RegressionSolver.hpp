#pragma once
#include <Eigen/Dense>

struct RegressionResult {
    Eigen::VectorXd theta;
    Eigen::VectorXd predicton;
    Eigen::VectorXd residuals;
    Eigen::VectorXd residuals_squared;
    double sum_residuals_squared;
    double mean_sum_residuals_squared;
    bool isUsable;
};

RegressionResult solveSystemByLLT(Eigen::MatrixXd const&, Eigen::VectorXd const&);
