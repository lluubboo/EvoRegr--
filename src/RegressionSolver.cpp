#include <Eigen/Dense>
#include <iostream>
#include "RegressionSolver.hpp"

RegressionResult solve_system_by_llt_detailed(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) {

    RegressionResult result = RegressionResult();

    result.theta = (predictors.transpose() * predictors).ldlt().solve(predictors.transpose() * target);
    result.isUsable = !result.theta.hasNaN() & result.theta.allFinite();

    if (result.isUsable) {
        result.predicton = predictors * result.theta;

        result.residuals = target - result.predicton;
        result.residuals_regression = result.predicton.array() - target.mean();
        result.residuals_total = target.array() - target.mean();
        result.percentage_error = 100. - ((result.predicton.array() / target.array()) * 100);

        result.residuals_squared = result.residuals.array().square();
        result.residuals_regression_squared = result.residuals_regression.array().square();
        result.residuals_total_squared = result.residuals_total.array().square();

        result.sum_squares_errors = result.residuals_squared.array().sum();
        result.sum_squares_regression = result.residuals_regression_squared.array().sum();
        result.sum_squares_total = result.residuals_total_squared.array().sum();

        result.rsquared = 1 - (result.sum_squares_errors / result.sum_squares_total);
        result.rsquaredadj = 1 - ((1 - result.rsquared) * ((predictors.rows() - 1) / (predictors.rows() - predictors.cols() - 1))); //Mordecai Ezekiel
    }
    else {
        result.sum_squares_regression = std::numeric_limits<double>::max();
        result.rsquared = 0.;
    }

    return result;
}

RegressionResult solve_system_by_llt_minimal(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) {

    RegressionResult result = RegressionResult();

    result.theta = (predictors.transpose() * predictors).ldlt().solve(predictors.transpose() * target);
    result.isUsable = !result.theta.hasNaN() & result.theta.allFinite();

    if (result.isUsable) {
        result.predicton = predictors * result.theta;
        result.residuals = target - result.predicton;
        result.sum_squares_errors = result.residuals.array().square().sum();
    }
    else {
        result.sum_squares_errors = std::numeric_limits<double>::max();
    }

    return result;
}
