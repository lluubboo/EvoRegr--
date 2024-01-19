#include <Eigen/Dense>
#include <iostream>
#include "RegressionSolver.hpp"

/**
 * @brief Performs a regression analysis using the LDLT decomposition method.
 *
 * @param predictors An Eigen::MatrixXd object where each row is a different observation and each column is a different predictor variable.
 * @param target An Eigen::VectorXd object where each element is the target variable for a different observation.
 *
 * @return A RegressionDetailedResult object that contains the detailed results of the regression analysis. This includes the coefficients of the regression,
 * a boolean indicating whether the coefficients are usable (i.e., they are not NaN or infinite), the predicted values of the target variable,
 * the residuals of the regression, the sum of squares of the residuals, the variance, the standard deviation, the mean sum of squares errors,
 * the root mean square error (RMSE), the R-squared value, and the adjusted R-squared value.
 */
RegressionDetailedResult solve_system_detailed(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) {

    RegressionDetailedResult result = RegressionDetailedResult();

    result.theta = (predictors.transpose() * predictors).colPivHouseholderQr().solve(predictors.transpose() * target);
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

        result.variance = result.sum_squares_regression / (target.rows() - 1);
        result.standard_deviation = sqrt(result.variance);

        result.mean_sum_squares_errors = result.sum_squares_errors / target.rows();
        result.rmse = sqrt(result.mean_sum_squares_errors);
        result.rsquared = 1 - (result.sum_squares_errors / result.sum_squares_total);
        result.rsquaredadj = 1 - ((1 - result.rsquared) * ((predictors.rows() - 1) / (predictors.rows() - predictors.cols() - 1))); //Mordecai Ezekiel
    }
    else {
        result.sum_squares_regression = std::numeric_limits<double>::max();
        result.rsquared = 0.;
    }

    return result;
}

double LLTSolver::operator()(EvoRegression::EvoDataSet const& dataset) const {
    // calc to save one transpose operation
    Eigen::MatrixXd predictor_transposed = dataset.training_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * dataset.training_predictor).llt().solve(predictor_transposed* dataset.training_target);

    // Check if the result is usable (no NaN or infinite values)
    if (!coefficients.hasNaN() && coefficients.allFinite()) {
        return (dataset.test_target - (dataset.test_predictor * coefficients)).squaredNorm();
    }
    else {
        // Set sum of squares errors to maximum value
        return std::numeric_limits<double>::max();
    }
}

double LDLTSolver::operator()(EvoRegression::EvoDataSet const& dataset) const {
    // calc to save one transpose operation
    Eigen::MatrixXd predictor_transposed = dataset.training_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * dataset.training_predictor).ldlt().solve(predictor_transposed * dataset.training_target);

    // Check if the result is usable (no NaN or infinite values)
    if (!coefficients.hasNaN() && coefficients.allFinite()) {
        return (dataset.test_target - (dataset.test_predictor * coefficients)).squaredNorm();
    }
    else {
        // Set sum of squares errors to maximum value
        return std::numeric_limits<double>::max();
    }
}

double ColPivHouseholderQrSolver::operator()(EvoRegression::EvoDataSet const& dataset) const {
    // calc to save one transpose operation
    Eigen::MatrixXd predictor_transposed = dataset.training_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * dataset.training_predictor).colPivHouseholderQr().solve(predictor_transposed * dataset.training_target);

    // Check if the result is usable (no NaN or infinite values)
    if (!coefficients.hasNaN() && coefficients.allFinite()) {
        return (dataset.test_target - (dataset.test_predictor * coefficients)).squaredNorm();
    }
    else {
        // Set sum of squares errors to maximum value
        return std::numeric_limits<double>::max();
    }
}

