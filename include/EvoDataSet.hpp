#pragma once
#include <string>
#include <Eigen/Dense>

namespace EvoRegression {

    struct EvoDataSet {

        Eigen::MatrixXd predictor;
        Eigen::VectorXd target;

        EvoDataSet(Eigen::MatrixXd predictor, Eigen::VectorXd target) :
            predictor(predictor),
            target(target)
        {}

        EvoDataSet() :
            predictor(Eigen::MatrixXd()),
            target(Eigen::VectorXd())
        {}
    };
}