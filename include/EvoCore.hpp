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

    EvoRegression::EvoDataSet original_dataset;

    EvoRegression::EvoDataSet titan_dataset_training, titan_dataset_test, titan_dataset_full;

    Eigen::MatrixXd titan_training_result, titan_testing_result;

    // settings
    EvoBoundaryConditions boundary_conditions;

    // population containers used in main loop
    std::vector<EvoIndividual> newborns, pensioners;
    std::vector <std::set<EvoIndividual>> island_gen_elite_groups;

    // tools
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines;
    std::vector<EvoRegression::EvoDataSet> compute_datasets;
    std::vector<LRUCache<std::string, double>> caches;

    // solver functor
    std::function<double(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&)> solver;

    // titan 
    EvoIndividual titan;
    std::vector<EvoIndividual> island_titans;
    RegressionDetailedResult titan_result;

    void finalize_boundary_conditions();
    void prepare_input_datasets(std::tuple<int, std::vector<double>>);
    void prepare_for_prediction();
    void predict();
    void optimize();
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