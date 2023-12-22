#include <Eigen/Dense>
#include "EvoPopulation.hpp"
#include "RegressionSolver.hpp"
#include "EvoLibrary.hpp"
#include <spdlog/spdlog.h>

/**
 * @brief Struct representing the boundary conditions for the evolutionary algorithm.
 */
struct EvoBoundaryConditions {
    size_t island_generation_size;          ///< The island generation size.
    size_t generation_count;                ///< The maximum number of generations.
    size_t interaction_cols;                ///< The number of columns used for interaction.
    size_t mutation_ratio;                  ///< The mutation ratio as a percentage.
    size_t island_count;                    ///< The number of islands in the algorithm.
    size_t migration_ratio;                 ///< The migration ratio as a percentage.
    size_t migration_interval;              ///< The interval at which migration occurs.
    size_t global_generation_size;          ///< The maximum size of the global generation.
    size_t migrants_count;                  ///< The number of migrants in each migration.

    /**
     * @brief Default constructor for EvoBoundaryConditions.
     * Initializes the member variables with default values.
     */
    EvoBoundaryConditions() :
        island_generation_size(100),
        generation_count(100),
        interaction_cols(0),
        mutation_ratio(15),
        island_count(12),
        migration_ratio(5),
        migration_interval(5),
        global_generation_size(island_generation_size * island_count),
        migrants_count(static_cast<size_t>((global_generation_size * migration_ratio) / 100))
    {}
};

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

class EvoCore {

    // logger
    static std::shared_ptr<spdlog::logger> logger;

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

    Transform::EvoDataSet get_dataset();

    void setTitan(EvoIndividual);
    void titan_evaluation(EvoIndividual const& individual);
    void titan_postprocessing();

public:

    EvoCore();

    void init_logger();

    void load_file(const std::string& filename);
    void batch_predict();

    void set_boundary_conditions(unsigned int generation_size_limit, unsigned int generation_count_limit, unsigned int interaction_cols, unsigned int mutation_rate, unsigned int island_count, unsigned int migration_ratio, unsigned int migration_interval);
    void set_solver(std::string const& solver_name);

    bool is_ready_to_predict();
};