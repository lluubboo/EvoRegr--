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
RegressionDetailedResult solve_system_by_ldlt_detailed(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) {

    RegressionDetailedResult result = RegressionDetailedResult();

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

/**
 * @brief Overloads the function call operator to perform a regression analysis using the LLT decomposition method.
 *
 * @param predictors An Eigen::MatrixXd object where each row is a different observation and each column is a different predictor variable.
 * @param target An Eigen::VectorXd object where each element is the target variable for a different observation.
 *
 * @return A RegressionSimpleResult object that contains the results of the regression analysis. This includes the coefficients of the regression,
 * a boolean indicating whether the coefficients are usable (i.e., they are not NaN or infinite), the predicted values of the target variable,
 * the residuals of the regression, and the sum of squares of the residuals.
 *
 * @throws std::invalid_argument If the dimensions of the predictors and target do not match.
 */
RegressionSimpleResult LLTSolver::operator()(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) const {

    // Check if predictors and target have matching dimensions
    if (predictors.rows() != target.size()) {
        throw std::invalid_argument("Dimensions of predictors and target do not match");
    }

    RegressionSimpleResult result;

    result.coefficients = (predictors.transpose() * predictors).llt().solve(predictors.transpose()* target);

    // Check if the result is usable (no NaN or infinite values)
    result.isUsable = !result.coefficients.hasNaN() && result.coefficients.allFinite();

    if (result.isUsable) {
        result.prediction = predictors * result.coefficients;
        result.residuals = target - result.prediction;
        result.sum_squares_errors = result.residuals.array().square().sum();
    }
    else {
        // Set sum of squares errors to maximum value
        result.sum_squares_errors = std::numeric_limits<double>::max();
    }
    return result;
}

/**
 * @brief Overloads the function call operator to perform a regression analysis using the LDLT decomposition method.
 *
 * @param predictors An Eigen::MatrixXd object where each row is a different observation and each column is a different predictor variable.
 * @param target An Eigen::VectorXd object where each element is the target variable for a different observation.
 *
 * @return A RegressionSimpleResult object that contains the results of the regression analysis. This includes the coefficients of the regression,
 * a boolean indicating whether the coefficients are usable (i.e., they are not NaN or infinite), the predicted values of the target variable,
 * the residuals of the regression, and the sum of squares of the residuals.
 *
 * @throws std::invalid_argument If the dimensions of the predictors and target do not match.
 */
RegressionSimpleResult LDLTSolver::operator()(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) const {

    // Check if predictors and target have matching dimensions
    if (predictors.rows() != target.size()) {
        throw std::invalid_argument("Dimensions of predictors and target do not match");
    }

    RegressionSimpleResult result;

    result.coefficients = (predictors.transpose() * predictors).ldlt().solve(predictors.transpose() * target);

    // Check if the result is usable (no NaN or infinite values)
    result.isUsable = !result.coefficients.hasNaN() && result.coefficients.allFinite();

    if (result.isUsable) {
        result.prediction = predictors * result.coefficients;
        result.residuals = target - result.prediction;
        result.sum_squares_errors = result.residuals.array().square().sum();
    }
    else {
        // Set sum of squares errors to maximum value
        result.sum_squares_errors = std::numeric_limits<double>::max();
    }
    return result;
}

/**
 * @brief Overloads the function call operator to perform a regression analysis using the column-pivoting Householder QR decomposition method.
 *
 * @param predictors An Eigen::MatrixXd object where each row is a different observation and each column is a different predictor variable.
 * @param target An Eigen::VectorXd object where each element is the target variable for a different observation.
 *
 * @return A RegressionSimpleResult object that contains the results of the regression analysis. This includes the coefficients of the regression,
 * a boolean indicating whether the coefficients are usable (i.e., they are not NaN or infinite), the predicted values of the target variable,
 * the residuals of the regression, and the sum of squares of the residuals.
 *
 * @throws std::invalid_argument If the dimensions of the predictors and target do not match.
 */
RegressionSimpleResult ColPivHouseholderQrSolver::operator()(Eigen::MatrixXd const& predictors, Eigen::VectorXd const& target) const {

    // Check if predictors and target have matching dimensions
    if (predictors.rows() != target.size()) {
        throw std::invalid_argument("Dimensions of predictors and target do not match");
    }

    RegressionSimpleResult result;

    result.coefficients = (predictors.transpose() * predictors).colPivHouseholderQr().solve(predictors.transpose() * target);

    // Check if the result is usable (no NaN or infinite values)
    result.isUsable = !result.coefficients.hasNaN() && result.coefficients.allFinite();

    if (result.isUsable) {
        result.prediction = predictors * result.coefficients;
        result.residuals = target - result.prediction;
        result.sum_squares_errors = result.residuals.array().square().sum();
    }
    else {
        // Set sum of squares errors to maximum value
        result.sum_squares_errors = std::numeric_limits<double>::max();
    }
    return result;
}

