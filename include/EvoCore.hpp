#include "IEvoAPI.hpp"
#include "BoundaryConditions.hpp"
#include "EvoPopulation.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"

/**
 * @brief Represents a dataset for evolutionary regression.
 *
 * This struct contains two Eigen::MatrixXd objects: `predictor` and `target`.
 * The `predictor` matrix represents the input features or predictors of the dataset,
 * while the `target` matrix represents the corresponding target values.
 */
struct EvoDataSet {
    Eigen::MatrixXd predictor; /**< The matrix representing the input features or predictors of the dataset. */
    Eigen::MatrixXd target; /**< The matrix representing the corresponding target values. */

    /**
     * @brief Constructs an EvoDataSet object with the given predictor and target matrices.
     *
     * @param predictor The matrix representing the input features or predictors of the dataset.
     * @param target The matrix representing the corresponding target values.
     */
    EvoDataSet(Eigen::MatrixXd predictor, Eigen::MatrixXd target) : predictor(predictor), target(target) {}

    /**
     * @brief Constructs an empty EvoDataSet object with default-initialized predictor and target matrices.
     */
    EvoDataSet() : predictor(Eigen::MatrixXd()), target(Eigen::MatrixXd()) {}
};

class EvoCore : public IEvoAPI {

    // datasets
    EvoDataSet original_dataset;
    EvoDataSet titan_dataset_robust;
    EvoDataSet titan_dataset_nonrobust;

    // boundary conditions
    EvoBoundaryConditions boundary_conditions;

    // solver functor
    std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver;

    // titan 
    EvoIndividual titan;
    RegressionDetailedResult titan_result;

    void create_regression_input(std::tuple<int, std::vector<double>>);
    void setTitan(EvoIndividual);
    void titan_evaluation(EvoIndividual const& individual);
    void titan_postprocessing();

public:

    EvoCore();

    void set_boundary_conditions(EvoBoundaryConditions const& boundary_conditions) override;
    void set_solver(std::string const& solver_name) override;
    void load_file(std::string const& filepath) override;
    void call_predict_method() override;
    bool is_ready_to_predict() const override;
};