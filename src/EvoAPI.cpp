#include <numeric>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fstream>
#include <mutex>
#include <random>
#include "RandomNumberGenerator.hpp"
#include "EvoAPI.hpp"
#include "IOTools.hpp"
#include "EvoPopulation.hpp"
#include "EvoCache.hpp"
#include "Stats.hpp"
#include "Plotter.hpp"
#include "omp.h"

// Define the static logger
std::shared_ptr<spdlog::logger> EvoAPI::logger;

/**
 * @brief Default constructor for the EvoAPI class.
 *
 * This constructor initializes the EvoAPI class with default solver functor.
 */
EvoAPI::EvoAPI() :
    generation_size_limit{ 100 },
    generation_count_limit{ 100 },
    interaction_cols{ 0 },
    mutation_rate{ 15 },
    x{ Eigen::MatrixXd::Zero(0, 0) },
    y{ Eigen::VectorXd::Zero(0) },
    solver{ LDLTSolver() },
    titan{ EvoIndividual() },
    titan_robust_dataset{ Transform::EvoDataSet() },
    titan_nonrobust_dataset{ Transform::EvoDataSet() },
    titan_result{RegressionDetailedResult()} {}

/**
 * @brief Sets the boundary conditions for the evolutionary algorithm.
 *
 * @param generation_size_limit The maximum size of each generation.
 * @param generation_count_limit The maximum number of generations.
 * @param interaction_cols The number of columns used for interaction.
 */
void EvoAPI::set_boundary_conditions(
    unsigned int generation_size_limit,
    unsigned int generation_count_limit,
    unsigned int interaction_cols,
    unsigned int mutation_rate
) {
    this->generation_size_limit = generation_size_limit;
    this->generation_count_limit = generation_count_limit;
    this->interaction_cols = interaction_cols;
    this->mutation_rate = mutation_rate;

    EvoAPI::logger->info("Boundary conditions set generation size limit: {}, generation count limit: {}, interaction cols: {}, mutation rate: {}",
        generation_size_limit, generation_count_limit, interaction_cols, mutation_rate);
}

/**
 * @brief Sets the solver to be used in the EvoAPI.
 *
 * This function sets the solver to be used in the EvoAPI based on the provided solver name.
 * The solver can be one of the following types: "LLT", "LDLT", "ColPivHouseholderQr".
 * If an unrecognized solver name is provided, the solver is set to the default "LDLT".
 *
 * @param solver_name A string representing the name of the solver.
 * It should be one of the following: "LLT", "LDLT", "ColPivHouseholderQr".
 */
void EvoAPI::set_solver(std::string const& solver_name) {
    if (solver_name == "LLT") {
        solver = LLTSolver();
        EvoAPI::logger->info("Solver set to LLT");
    }
    else if (solver_name == "LDLT") {
        solver = LDLTSolver();
        EvoAPI::logger->info("Solver set to LDLT");
    }
    else if (solver_name == "ColPivHouseholderQr") {
        solver = ColPivHouseholderQrSolver();
        EvoAPI::logger->info("Solver set to ColPivHouseholderQr");
    }
    else {
        solver = LDLTSolver();
        EvoAPI::logger->info("Unrecognized solver type. Solver set to default LDLT");
    }
}

/**
 * Initializes the logger for the EvoAPI class.
 * If a logger with the name "EvoRegression++" already exists, it connects to it.
 * Otherwise, it creates a new logger with the name "EvoRegression++" and sets its level to debug.
 * The logger's pattern is set to "[EvoRegression++] [%H:%M:%S.%e] [%^%l%$] [thread %t] %v".
 */
void EvoAPI::init_logger() {

    auto shared_logger = spdlog::get("EvoLogger");
    shared_logger->info("EvoAPI logger trying to connect to existing logger");

    if (shared_logger) {
        EvoAPI::logger = shared_logger;
        EvoAPI::logger->info("EvoAPI logger connected to existing logger");
    }
    else {
        EvoAPI::logger = spdlog::stdout_color_mt("EvoLogger");
        EvoAPI::logger->set_level(spdlog::level::debug);
        EvoAPI::logger->set_pattern("[EvoRegression++] [%H:%M:%S.%e] [%^%l%$] [thread %t] %v");
        EvoAPI::logger->info("EvoAPI logger initialized");
    }
}

/**
 * The function `load_file` prompts the user to enter a filename, opens the file, and processes its
 * contents if it is successfully opened.
 */
void EvoAPI::load_file(const std::string& filename) {
    try {
        create_regression_input(parse_csv<double>(filename));
    }
    catch (const std::exception& e) {
        EvoAPI::logger->error("Error processing file {}: {}", filename, e.what());
    }
    catch (...) {
        EvoAPI::logger->error("An unknown error occurred while processing file {}", filename);
    }
    EvoAPI::logger->info("File {} loaded", filename);
}

/**
 * @brief Creates the regression input matrix for EvoAPI.
 *
 * This function takes a tuple containing the number of rows, number of columns, and a vector of data values.
 * It initializes the predictor matrix and target vector based on the input data.
 * The predictor matrix is initialized with ones, including a column of ones for x0 and additional interaction columns.
 * The target vector is filled with the values from the target column of the input data.
 *
 * @param input A tuple containing the number of rows, number of columns, and a vector of data values.
 */
void EvoAPI::create_regression_input(std::tuple<int, std::vector<double>> input) {

    std::vector<double> data = std::get<1>(input);

    //input matrix columns (with target column)
    int n_input{ std::get<0>(input) };
    int m_input{ static_cast<int>(data.size()) / n_input };

    // predictor matrix column count (n_input - 1 (because of target column) + 1 + interaction columns)
    int n_output = n_input + interaction_cols;

    // + 1 because of header row
    int m_output = m_input + 1;

    // mark y column indexed from 0 (is last every time)
    int target_col_index = n_input - 1;

    // initialize predictors matrix to matrix of ones because of x0 and interaction columns
    x = Eigen::MatrixXd::Ones(m_input, n_output);
    y.resize(m_input, 1);

    for (int row = 0; row < m_input; ++row) {
        for (int col = 0; col < n_input; ++col) {
            // last column is always Y or in other words regressant, dependant variable
            if (col == target_col_index) {
                y(row, 0) = data[col + n_input * row];
            }
            // fil predictors (first is x0 column of 1, last are interaction filled default to 1 too - but they are able to mutate)
            if (col < target_col_index) {
                x(row, col + 1) = data[col + n_input * row];
            }
        }
    }

    EvoAPI::logger->info("Predictor matrix initialized with {} rows and {} columns", m_output, n_output);
    EvoAPI::logger->info("Target vector initialized with {} rows", m_output);
}

/**
 * @brief Get the dataset used by the evolutionary algorithm.
 *
 * This function returns a dataset consisting of the x and y data used by the evolutionary algorithm.
 * The dataset is returned as an instance of the `Transform::EvoDataSet` class.
 *
 * @return Transform::EvoDataSet The dataset used by the evolutionary algorithm.
 */
Transform::EvoDataSet EvoAPI::get_dataset() {
    return Transform::EvoDataSet{ x, y };
}

/**
 * @brief Checks if the model is ready to make predictions.
 *
 * The model is considered ready if both the input (x) and output (y) data sets are not empty.
 *
 * @return true if both x and y data sets are not empty, false otherwise.
 */
bool EvoAPI::is_ready_to_predict() {
    return x.size() > 0 && y.size() > 0;
};

void EvoAPI::batch_predict() {

    EvoAPI::logger->info("Starting batch prediction process...");

    int island_count = omp_get_max_threads();

    // random engine for each island
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines = create_random_engines(island_count);

    // create generation zero
    EvoPopulation population(
        Factory::generate_random_generation(
            island_count * generation_size_limit,
            get_dataset(),
            random_engines[0],
            solver
        )
    );

    std::vector<std::future<IslandOutput>> futures;

    for (int island_index = 0; island_index < island_count; island_index++) {

        // Prepare the input for the function
        EvoRegressionInput input{
            x,
            y,
            population,
            random_engines[island_index],
            solver,
            mutation_rate,
            generation_size_limit,
            generation_count_limit,
            island_index,
            island_count
        };

        // Start a new task for each island
        futures.push_back(std::async(std::launch::async, &EvoAPI::run_island_async, input));
    }

    std::vector<IslandOutput> results;
    for (auto& future : futures) {
        auto result = future.get();
        titan_evaluation(result.best_individual);
        EvoAPI::logger->info("Island {} finished with titan fitness: {}", result.island_id, result.best_individual.fitness);
    }

    log_result();
}

/**
 * @brief Runs the evolutionary algorithm on a single island (subpopulation) and logs the execution time.
 *
 * @param input An EvoRegressionInput object containing the parameters for the evolutionary algorithm.
 *
 * @return An IslandOutput object containing the best individual found on the island and the island's ID.
 *
 * This function performs the following steps:
 * 1. Records the start time.
 * 2. Calls the run_island function to run the evolutionary algorithm on the island.
 * 3. Records the end time.
 * 4. Logs the execution time and the island's ID.
 * 5. Returns the output from the run_island function.
 *
 * The function is designed to be run asynchronously, so it can be used with std::async or similar functions to run the evolutionary algorithm on multiple islands concurrently.
 */
IslandOutput EvoAPI::run_island_async(EvoRegressionInput input) {

    auto start_time = std::chrono::high_resolution_clock::now();

    IslandOutput output = run_island(input);

    auto end_time = std::chrono::high_resolution_clock::now();

    EvoAPI::logger->info("Island {} finished in /s: {}", input.island_id, std::chrono::duration<double>(end_time - start_time).count());
    return output;
}

/**
 * @brief Runs the evolutionary algorithm on a single island (subpopulation).
 *
 * @param input An EvoRegressionInput object containing the parameters for the evolutionary algorithm.
 *
 * @return An IslandOutput object containing the best individual found on the island and the island's ID.
 *
 * This function performs the following steps for a specified number of generations:
 * 1. If the current generation is a multiple of 10 and not the first, it performs a migration operation, swapping a portion of the island's population with individuals from other islands.
 * 2. For each individual in the island's subpopulation, it performs crossover and mutation to produce a new individual, evaluates the new individual's fitness, and adds the new individual to the island's population.
 * 3. It checks each individual in the island's population and updates the island's best individual (the "titan") if it finds an individual with a lower fitness.
 * 4. It moves the island's population to the shared population.
 *
 * The function returns the island's titan and the island's ID.
 */
IslandOutput EvoAPI::run_island(EvoRegressionInput input) {

    // subpopulation borders
    auto island_borders = EvoAPI::get_island_borders(input.island_id, input.generation_size_limit);

    // island cache
    auto island_cache = EvoCache<std::string, double>(input.population.size() * 100);

    EvoIndividual island_titan, newborn;

    // create island population and reserve it to generation size limit
    // its more efficient to move to shared population after each generation only
    EvoPopulation island_population(0, input.generation_size_limit);

    for (int gen_index = 0; gen_index < input.generation_count_limit; gen_index++) {

        if (gen_index != 0 && gen_index % 100 == 0) EvoAPI::logger->info("Started island {} generation {} of {}", input.island_id, gen_index, input.generation_count_limit);

        if (gen_index != 0 && gen_index % 10 == 0) {
            input.population.batch_swap_individuals(input.island_id, input.island_count, island_population.size() * 0.05, input.random_engine);
        }

        for (unsigned int entity_index = island_borders[0]; entity_index <= island_borders[1]; entity_index++) {

            //crossover & mutation [vector sex]
            newborn = Reproduction::reproduction(
                std::move(Selection::tournament_selection(input.population,input.random_engine,island_borders[0],island_borders[1])),
                input.x.cols(),
                input.x.rows(),
                input.mutation_rate,
                input.random_engine
            );

            auto it = island_cache.get(newborn.to_string_code());
            if (it) {
                newborn.fitness = *it;
            }
            else {
                // merge & transform & make robust predictors & target / solve regression problem
                newborn.evaluate(
                    EvoMath::get_fitness<std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)>>(
                        Transform::data_transformation_robust(
                            input.x,
                            input.y,
                            newborn
                        ),
                        input.solver
                    )
                );

                island_cache.put(newborn.to_string_code(), newborn.fitness);
            }

            // newborn to population
            island_population.move_to_end(std::move(newborn));
        }

        for (const auto& individual : island_population) {
            if (individual.fitness < island_titan.fitness) {
                island_titan = individual;
                EvoAPI::logger->info("Island {} new titan with fitness {} in generation {}", input.island_id, island_titan.fitness, gen_index);
            }
        }

        input.population.batch_population_move(std::move(island_population), island_borders[0]);
    }

    return { island_titan, input.island_id };
}

/**
 * @brief Calculates the borders for a given island in a generation.
 *
 * This function calculates the start and end indices for a given island in a generation.
 * The start index is calculated as the island ID multiplied by the generation size limit.
 * The end index is calculated as the start index plus the generation size limit minus one.
 *
 * @param island_id The ID of the island for which to calculate the borders.
 * @param island_count The total number of islands. (Currently unused in the function)
 * @param generation_size_limit The maximum size of a generation.
 * @return A std::array<unsigned int, 2> containing the start and end indices for the island in the generation.
 */
std::array<unsigned int, 2> EvoAPI::get_island_borders(unsigned int island_id, unsigned int generation_size_limit) noexcept {
    return { island_id * generation_size_limit , island_id * generation_size_limit + generation_size_limit - 1 };
};

/**
 * Displays the result of the evolutionary regression analysis.
 * This function performs post-processing, shows the regression summary,
 * the Titan history, regression coefficients, genotype, and formula.
 */
void EvoAPI::log_result() {
    titan_postprocessing();
    EvoAPI::logger->info(get_regression_summary_table());
    EvoAPI::logger->info("Regression results showing...");
}

/**
 * The function sets a new titan and adds its fitness and generation index to the titan history.
 *
 * @param titan The "titan" parameter is an object of type EvoIndividual, which represents a specific
 * individual in the evolutionary algorithm. It contains information about the individual's fitness and
 * other characteristics.
 * @param generation_index The generation index represents the index or number of the current
 * generation in the evolutionary algorithm. It is used to keep track of the progress and history of
 * the algorithm.
 */
void EvoAPI::setTitan(EvoIndividual titan) {
    this->titan = titan;
}

/**
 * The function `titan_evaluation` compares the fitness of a participant with the fitness of the
 * current titan and updates the titan if the participant has a lower fitness.
 *
 * @param participant The participant parameter is an object of the EvoIndividual class, which
 * represents an individual in the evolutionary algorithm. It contains information about the
 * individual's fitness and other attributes.
 * @param generation_index The generation index is an integer value that represents the current
 * generation of the evolutionary algorithm. It is used to keep track of the progress of the algorithm
 * and can be used for various purposes, such as logging or analysis.
 */
void EvoAPI::titan_evaluation(EvoIndividual participant) {
    if (participant.fitness < titan.fitness) setTitan(participant);
}

/**
 * @brief Create a vector of random engines.
 *
 * This method creates a vector of Xoshiro256Plus random engines. Each random engine is initialized with a unique seed.
 * The seed for the first random engine is generated using a random_device, and the seeds for the remaining random engines are generated by performing a long jump on the previous random engine.
 *
 * @param count The number of random engines to create.
 * @return A vector of Xoshiro256Plus random engines.
 */
std::vector<XoshiroCpp::Xoshiro256Plus> EvoAPI::create_random_engines(int count) {

    //create master random engine with random seed
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    XoshiroCpp::Xoshiro256Plus master_random_engine(seed);

    //create n random engines with long jump from master random engine
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines;
    for (int i = 0;i < count;i++) {
        master_random_engine.longJump();
        random_engines.emplace_back(master_random_engine.serialize());
    }

    EvoAPI::logger->info("Created {} random engines", count);
    return random_engines;
}

/**
 * Performs post-processing on the Titan dataset.
 * This function applies data transformation techniques to remove outliers,
 * and then solves the regression system using the LDLT decomposition method.
 */
void EvoAPI::titan_postprocessing() {
    // data without outliers
    titan_robust_dataset = Transform::data_transformation_robust(x, y, titan);
    // data witho outliers
    titan_nonrobust_dataset = Transform::data_transformation_nonrobust(x, y, titan);
    // regression result
    titan_result = solve_system_detailed(titan_robust_dataset.predictor, titan_robust_dataset.target);

    EvoAPI::logger->info("Titan postprocessing finished");
}

/**
 * The function generates a summary matrix for a regression result. The matrix includes the original target values,
 * the predicted values, the difference between the original and predicted values, and the percentage difference.
 *
 * @param result The result is a constant reference to a RegressionDetailedResult object, which contains the regression results.
 * @param original_x The original_x is an Eigen::MatrixXd object representing the original predictor data.
 * @param original_y The original_y is an Eigen::VectorXd object representing the original target data.
 *
 * @return an Eigen::MatrixXd object representing the summary matrix. Each row corresponds to a data point, and the columns are:
 *         - Column 0: Original target values
 *         - Column 1: Predicted target values
 *         - Column 2: Difference between original and predicted values
 *         - Column 3: Percentage difference between original and predicted values
 */
Eigen::MatrixXd EvoAPI::get_regression_summary_matrix(RegressionDetailedResult const& result, Eigen::MatrixXd const& original_x, Eigen::VectorXd original_y) {
    Eigen::MatrixXd regression_result_matrix(original_x.rows(), 4);

    // get target for coefficients not disturbed by outliers
    Eigen::VectorXd prediction = original_x * result.theta;

    // transform target back to original values for presentation p
    titan.y_transformer_chromosome.at(0).transformBack(original_y);
    titan.y_transformer_chromosome.at(0).transformBack(prediction);

    //assembly result matrix
    regression_result_matrix.col(0) = original_y;
    regression_result_matrix.col(1) = prediction;
    regression_result_matrix.col(2) = regression_result_matrix.col(0) - regression_result_matrix.col(1);
    regression_result_matrix.col(3) = 100. - ((regression_result_matrix.col(1).array() / regression_result_matrix.col(0).array()) * 100);

    return regression_result_matrix;
}

/**
 * @brief Returns the regression result table as a string.
 *
 * This function generates a regression result table using the provided data and returns it as a string.
 * The table includes columns for the target, prediction, difference, and percentage difference.
 *
 * @return The regression result table as a string.
 */
std::string EvoAPI::get_regression_result_table() {
    Eigen::MatrixXd regression_result_matrix = get_regression_summary_matrix(
        titan_result, titan_nonrobust_dataset.predictor, titan_nonrobust_dataset.target
    );
    Plotter<double> plt = Plotter(
        regression_result_matrix.data(),
        "Regression result summary with outliers",
        { "Target", "Prediction", "Difference", "Percentage difference" },
        149,
        regression_result_matrix.size(),
        DataArrangement::ColumnMajor
    );
    return plt.get_table();
};

/**
 * @brief Returns the regression result table as a string.
 *
 * This function generates a regression result table using the provided data and returns it as a string.
 * The table includes columns for the target, prediction, difference, and percentage difference.
 *
 * @return The regression result table as a string.
 */
std::string EvoAPI::get_regression_robust_result_table() {
    Eigen::MatrixXd regression_result_matrix = get_regression_summary_matrix(
        titan_result, titan_robust_dataset.predictor, titan_robust_dataset.target
    );
    Plotter<double> plt = Plotter(
        regression_result_matrix.data(),
        "Regression result summary without outliers",
        { "Target", "Prediction", "Difference", "Percentage difference" },
        149,
        regression_result_matrix.size(),
        DataArrangement::ColumnMajor
    );
    return plt.get_table();
};

/**
 * @brief Retrieves the regression coefficients table.
 *
 * This function creates a Plotter object to visualize the regression coefficients
 * and returns the table representation of the plot.
 *
 * @return The regression coefficients table.
 */
std::string EvoAPI::get_regression_coefficients_table() {
    Plotter<double> plt = Plotter(
        titan_result.theta.data(),
        "Regression coefficients",
        { "Coefficients" },
        149,
        titan_result.theta.size(),
        DataArrangement::ColumnMajor
    );
    return plt.get_table();
};

/**
 * @brief Returns the genotype table as a string.
 *
 * This function generates a genotype table using the Plotter class and returns it as a string.
 * The genotype table includes information about merge chromosome, transform predictor chromosome,
 * transform target chromosome, and robust chromosome.
 *
 * @return The genotype table as a string.
 */
std::string EvoAPI::get_genotype_table() {
    std::stringstream genotype_table;

    std::vector<std::string> merge_chromosome = titan.merge_chromosome_to_string_vector();
    Plotter<std::string> plt = Plotter(
        merge_chromosome.data(),
        "Merge chromosome",
        { "Alleles" },
        149,
        merge_chromosome.size(),
        DataArrangement::RowMajor
    );
    genotype_table << plt.get_table();

    std::vector<std::string> transform_predictor_chromosome = titan.transform_predictor_chromosome_to_string_vector();
    plt = Plotter(
        transform_predictor_chromosome.data(),
        "Transform predictor chromosome",
        { "Alleles" },
        149,
        transform_predictor_chromosome.size(),
        DataArrangement::RowMajor
    );
    genotype_table << plt.get_table();

    std::vector<std::string> transform_target_chromosome = titan.transform_target_chromosome_to_string_vector();
    plt = Plotter(
        transform_target_chromosome.data(),
        "Transform target chromosome",
        { "Alleles" },
        149,
        transform_target_chromosome.size(),
        DataArrangement::RowMajor
    );
    genotype_table << plt.get_table();

    std::vector<std::string> robust_chromosome = titan.robust_chromosome_to_string_vector();
    plt = Plotter(
        robust_chromosome.data(),
        "Robust chromosome",
        { "Alleles" },
        149,
        robust_chromosome.size(),
        DataArrangement::RowMajor
    );

    genotype_table << plt.get_table();
    return genotype_table.str();
}

/**
 * @brief Retrieves the formula table.
 *
 * This function returns a string representation of the formula table.
 *
 * @return The formula table as a string.
 */
std::string EvoAPI::get_formula_table() {
    std::vector<std::string> formula{ titan.to_math_formula() };
    Plotter<std::string> plt = Plotter(
        formula.data(),
        "Evo-regression formula",
        { "Formula" },
        149,
        formula.size(),
        DataArrangement::RowMajor
    );
    return plt.get_table();
};

/**
 * @brief Generates a table of regression result metrics.
 *
 * This function calculates the median of the percentage error, the standard deviation, and the coefficient of determination (R2) of the regression results.
 * These metrics are then displayed in a table.
 *
 * @return A string representing a table of the regression result metrics.
 */
std::string EvoAPI::get_result_metrics_table() {

    std::vector<double> metrics{
        DescriptiveStatistics::median(titan_result.percentage_error.data(), titan_result.percentage_error.size()),
        titan_result.standard_deviation,
        titan_result.rsquared
    };

    Plotter<double> plt = Plotter(
        metrics.data(),
        "Regression result metrics [robust]",
        { "Result median", "Result standard deviation", "Robust result COD (R2)" },
        149,
        metrics.size(),
        DataArrangement::RowMajor
    );
    return plt.get_table();
};

/**
 * @brief Returns a summary table for regression analysis.
 *
 * This function generates a summary table for regression analysis by combining
 * various tables including regression result table, titan history table,
 * regression coefficients table, genotype table, and formula table.
 *
 * @return A string containing the regression summary table.
 */
std::string EvoAPI::get_regression_summary_table() {
    std::stringstream table;

    table << get_regression_result_table();
    table << get_regression_robust_result_table();
    table << get_result_metrics_table();
    table << get_regression_coefficients_table();
    table << get_genotype_table();
    table << get_formula_table();
    EvoAPI::logger->info("Regression summary table generated");
    return table.str();
};




