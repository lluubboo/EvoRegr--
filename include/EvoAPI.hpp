#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include "EvoIndividual.hpp"
#include "XoshiroCpp.hpp"

struct EvoDataSet {
    Eigen::MatrixXd predictor;
    Eigen::MatrixXd target;
};

class EvoAPI {
public:

    int generation_size_limit;
    int generation_count_limit;

    Eigen::MatrixXd x;
    Eigen::MatrixXd y;

    std::vector<std::vector<EvoIndividual>> population;
    std::vector<double> fitness_history;
    std::vector<double> titan_history;
    std::unordered_map<std::string, double> cache;

    EvoAPI(const std::string&);
    void setBoundaryConditions(unsigned int generation_size_limit, unsigned int generation_count_limit);
    void predict();
    void predict_with_cache();
    void showMeBest();

private:
    EvoIndividual titan;

    void setTitan(EvoIndividual, int);
    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    void append_generation_zero(XoshiroCpp::Xoshiro256Plus&);

    EvoDataSet data_transformation_cacheless(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual&);
};