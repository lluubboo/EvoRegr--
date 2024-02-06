#pragma once
#include <set>
#include "IEvoAPI.hpp"
#include "BoundaryConditions.hpp"
#include "EvoIndividual.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "EvoDataSet.hpp"
#include "LRUCache.hpp"


class EvoCore : public IEvoAPI {

    // input data (robust = without outliers, nonrobust = with outliers)
    // algorithm is making basic selection of data based on robustness

    EvoRegression::EvoDataSet original_dataset;

    EvoRegression::EvoDataSet titan_dataset_training; // testing dataset part without outliers
    EvoRegression::EvoDataSet titan_dataset_test; // testing dataset part with outliers
    EvoRegression::EvoDataSet titan_dataset_full; // whole dataset 

    Eigen::MatrixXd training_result;
    Eigen::MatrixXd testing_result;

    // settings
    EvoBoundaryConditions boundary_conditions;

    // population containers used in main loop
    std::vector<EvoIndividual> newborns;
    std::vector<EvoIndividual> pensioners; // using only for reading genetic material
    std::vector <std::set<EvoIndividual>> island_gen_elite_groups;

    // tools
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines;
    std::vector<EvoRegression::EvoDataSet> compute_datasets;
    std::vector<LRUCache<std::string, double>> caches;

    // solver functor
    std::function<double(EvoRegression::EvoDataSet const&, int test_ratio, float regularizaton_parameter)> solver;

    // titan 
    EvoIndividual titan;
    std::vector<EvoIndividual> island_titans;
    RegressionDetailedResult titan_result;

    void prepare_input_datasets(std::tuple<int, std::vector<double>>);
    void prepare_for_prediction();
    void predict();
    void rank_past_generation();
    void find_titan();
    void move_elites();
    void clear_elite_groups();

    void log_island_titans(int);
    void setTitan(EvoIndividual);
    void titan_evaluation(EvoIndividual const& individual);
    void titan_postprocessing();

    void log_result();

public:

    EvoCore();

    void set_boundary_conditions(EvoBoundaryConditions const& boundary_conditions) override;
    void set_solver(std::string const& solver_name) override;
    void load_file(std::string const& filepath) override;
    void call_predict_method() override;
    bool is_ready_to_predict() const override;
};