#pragma once
#include "IEvoAPI.hpp"
#include "BoundaryConditions.hpp"
#include "EvoPopulation.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include "EvoDataSet.hpp"
#include "LRUCache.hpp"

class EvoCore : public IEvoAPI {

    EvoRegression::EvoDataSet original_dataset; // original dataset
    EvoRegression::EvoDataSet titan_dataset_robust; // dataset of titan with robust features
    EvoRegression::EvoDataSet titan_dataset_nonrobust; // dataset of titan with without robusting (deleting outliers)

    EvoBoundaryConditions boundary_conditions;

    // population containers
    std::vector<EvoIndividual> newborns;
    std::vector<EvoIndividual> pensioners; // using only for reading genetic material

    // tools
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines;
    std::vector<EvoRegression::EvoDataSet> compute_datasets;
    std::vector<LRUCache<std::string, double>> caches;

    // solver functor
    std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver;

    // titan 
    EvoIndividual titan;
    std::vector<EvoIndividual> island_titans;
    RegressionDetailedResult titan_result;

    void create_regression_input(std::tuple<int, std::vector<double>>);

    void prepare_for_prediction();
    void predict();
    void rank_past_generation();
    void rank_island_titans();

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