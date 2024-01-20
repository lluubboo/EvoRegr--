#pragma once
#include <string>
#include <Eigen/Dense>

namespace EvoRegression {

    struct EvoDataSet {

        Eigen::MatrixXd training_predictor;
        Eigen::VectorXd training_target;

        EvoDataSet(Eigen::MatrixXd training_predictor, Eigen::VectorXd training_target) :
            training_predictor(training_predictor),
            training_target(training_target)
        {}

        EvoDataSet() :
            training_predictor(Eigen::MatrixXd()),
            training_target(Eigen::VectorXd())
        {}
    };
}