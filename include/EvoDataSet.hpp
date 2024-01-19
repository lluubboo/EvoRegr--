#pragma once
#include <string>
#include <Eigen/Dense>

namespace EvoRegression {

    struct EvoDataSet {

        Eigen::MatrixXd training_predictor;
        Eigen::VectorXd training_target;

        Eigen::MatrixXd test_predictor;
        Eigen::VectorXd test_target;

        EvoDataSet(Eigen::MatrixXd training_predictor, Eigen::VectorXd training_target, Eigen::MatrixXd test_predictor, Eigen::VectorXd test_target) :
            training_predictor(training_predictor),
            training_target(training_target),
            test_predictor(test_predictor),
            test_target(test_target)
        {}

        EvoDataSet() :
            training_predictor(Eigen::MatrixXd()),
            training_target(Eigen::VectorXd()),
            test_predictor(Eigen::MatrixXd()),
            test_target(Eigen::VectorXd())
        {}
    };
}