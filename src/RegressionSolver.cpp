#include <Eigen/Dense>
#include <iostream>
#include "RegressionSolver.hpp"

RegressionResult solveSystemByLLT(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) {
    
    RegressionResult result = RegressionResult();

    result.theta = (predictors.transpose() * predictors).ldlt().solve(predictors.transpose() * target);
    result.isUsable = !result.theta.hasNaN() & result.theta.allFinite();

    if (result.isUsable) {
        result.predicton = predictors * result.theta;
        result.residuals = target - result.predicton;
        result.residuals_squared = result.residuals.array().square();
        result.sum_residuals_squared = result.residuals_squared.array().sum();
        result.mean_sum_residuals_squared = result.sum_residuals_squared / target.size();
    }
    else {
        result.mean_sum_residuals_squared = std::numeric_limits<double>::max();
        result.sum_residuals_squared = std::numeric_limits<double>::max();
    }

    return result;
}
