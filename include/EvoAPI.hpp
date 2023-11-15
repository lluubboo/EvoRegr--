#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "XoshiroCpp.hpp"

class EvoAPI {

public:

    EvoAPI(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols);
    void load_file();
    void predict();
    void show_result();

private:

    //boundary conditions
    int generation_size_limit, generation_count_limit, interaction_cols;

    // best individual
    EvoIndividual titan;

    //absolute path to datafile
    std::string filename;

    // original predictor
    Eigen::MatrixXd x;

    // original target
    Eigen::MatrixXd y;

    // population storage (0=old generation, 1=new generation)
    std::vector<std::vector<EvoIndividual>> population;

    // params usefull for statistics & plotting
    std::vector<double> titan_history;
    // [titan fitness, titan generation]
    std::vector<double> generation_fitness_metrics;
    // [gen best fitness, gen fitness geometrical mean, gen fitness arithmetic mean, gen fitness median, gen fitness standard deviation, number of ill individuals]] 
    // metrics of healthy individuals, othervise they will be distorted

    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(const std::uint64_t seed, int count);
    Transform::EvoDataSet data_transformation_robust(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual const&);
    Transform::EvoDataSet data_transformation_nonrobust(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual const&);
    Eigen::MatrixXd get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd original_x, Eigen::VectorXd original_y);
    Eigen::MatrixXd get_titan_summary(std::vector<double>);
    void setTitan(EvoIndividual, int);
    void create_regression_input(std::tuple<int, int, std::vector<double>>);
    void titan_evaluation(EvoIndividual participant, int generation_index);
    void process_generation_fitness(std::set<double> const& generation_fitness);
    void generation_postprocessing(std::vector<EvoIndividual> const& generation, int generation_index);
};