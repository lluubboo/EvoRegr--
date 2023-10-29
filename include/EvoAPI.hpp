#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include "EvoIndividual.hpp"
#include "XoshiroCpp.hpp"

struct EvoDataSet {
    Eigen::MatrixXf predictor;
    Eigen::MatrixXf target;
};

class EvoAPI {
public:

    int generation_size_limit;
    int generation_count_limit;

    Eigen::MatrixXf x;
    Eigen::MatrixXf y;

    std::vector<std::vector<EvoIndividual>> population;
    std::vector<float> fitness_history;
    std::vector<float> titan_history;

    std::unordered_map<std::string, float> cache;

    EvoAPI(const std::string&);
    void setBoundaryConditions(unsigned int generation_size_limit, unsigned int generation_count_limit);
    void predict();
    void showMeBest();

private:
    EvoIndividual titan;

    void setTitan(EvoIndividual, int);
    void create_regression_input(std::tuple<int, int, std::vector<float>>);
    void append_generation_zero(XoshiroCpp::Xoshiro256Plus&);

    EvoDataSet data_transformation_cacheless(Eigen::MatrixXf, Eigen::VectorXf, EvoIndividual&);
};