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

    EvoAPI(const std::string&);
    void setBoundaryConditions(unsigned int generation_size_limit, unsigned int generation_count_limit);
    void predict();
    void showMeBest();
    void profiler();

private:
    EvoIndividual titan;
    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);
    void setTitan(EvoIndividual, int);
    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    std::vector<EvoIndividual> create_random_generation(XoshiroCpp::Xoshiro256Plus&, int size);
    EvoDataSet data_transformation_cacheless(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual&);
};