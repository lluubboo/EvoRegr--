#pragma once
#include <string>
#include <Eigen/Dense>

namespace EvoRegression {
    /**
     * @brief Represents a dataset for evolutionary regression.
     *
     * This struct contains two Eigen::MatrixXd objects: `predictor` and `target`.
     * The `predictor` matrix represents the input features or predictors of the dataset,
     * while the `target` matrix represents the corresponding target values.
     */
    struct EvoDataSet {
        Eigen::MatrixXd predictor; /**< The matrix representing the input features or predictors of the dataset. */
        Eigen::VectorXd target; /**< The matrix representing the corresponding target values. */

        /**
         * @brief Constructs an EvoDataSet object with the given predictor and target matrices.
         *
         * @param predictor The matrix representing the input features or predictors of the dataset.
         * @param target The matrix representing the corresponding target values.
         */
        EvoDataSet(Eigen::MatrixXd predictor, Eigen::VectorXd target) : predictor(predictor), target(target) {}

        /**
         * @brief Constructs an empty EvoDataSet object with default-initialized predictor and target matrices.
         */
        EvoDataSet() : predictor(Eigen::MatrixXd()), target(Eigen::VectorXd()) {}
    };
}