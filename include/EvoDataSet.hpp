#pragma once
#include <string>
#include <Eigen/Dense>

namespace EvoRegression {

    struct EvoDataSet {

        Eigen::MatrixXd learn_predictor;
        Eigen::VectorXd learn_target;

        Eigen::MatrixXd test_predictor;
        Eigen::VectorXd test_target;

        EvoDataSet(Eigen::MatrixXd learn_predictor, Eigen::VectorXd learn_target, Eigen::MatrixXd test_predictor, Eigen::VectorXd test_target) :
            learn_predictor(learn_predictor), 
            learn_target(learn_target), 
            test_predictor(test_predictor), 
            test_target(test_target)
        {}

        EvoDataSet() :
            learn_predictor(Eigen::MatrixXd()), 
            learn_target(Eigen::VectorXd()),
            test_predictor(Eigen::MatrixXd()),
            test_target(Eigen::VectorXd())
        {}
    };
}