#pragma once
#include <Eigen/Dense>

struct RegressionResult {
    Eigen::VectorXf theta;
    Eigen::VectorXf predicton;
    Eigen::VectorXf residuals;
    Eigen::VectorXf residuals_squared;
    float sum_residuals_squared;
    float mean_sum_residuals_squared;
    bool isUsable;
};

RegressionResult solveSystemByLLT(Eigen::MatrixXf const&, Eigen::VectorXf const&);

Eigen::MatrixXf feature_scale(Eigen::MatrixXf& matrix);