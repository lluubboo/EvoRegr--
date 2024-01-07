#include "EvoCore.hpp"
#include "Stats.hpp"
#include "Log.hpp"
#include "IOTools.hpp"
#include "RandomChoices.hpp"
#include "EvoResultPostprocessing.hpp"
#include "LRUCache.hpp"

EvoCore::EvoCore() :
    original_dataset(),
    titan_dataset_robust(),
    titan_dataset_nonrobust(),
    boundary_conditions(),
    solver(LDLTSolver()),
    titan(),
    titan_result()
{}

void EvoCore::set_boundary_conditions(EvoBoundaryConditions const& boundary_conditions) {
    this->boundary_conditions = boundary_conditions;

    EvoRegression::Log::get_logger()->info(
        "Boundary conditions set to:\n"
        "Island generation size: {}\n"
        "Generation count: {}\n"
        "Interaction columns: {}\n"
        "Mutation ratio: {}\n"
        "Island count: {}\n"
        "Migration ratio: {}\n"
        "Migration interval: {}\n"
        "Global generation size: {}\n"
        "Migrants count: {}",
        boundary_conditions.island_generation_size,
        boundary_conditions.generation_count,
        boundary_conditions.interaction_cols,
        boundary_conditions.mutation_ratio,
        boundary_conditions.island_count,
        boundary_conditions.migration_ratio,
        boundary_conditions.migration_interval,
        boundary_conditions.global_generation_size,
        boundary_conditions.migrants_count
    );
}

/**
 * Sets the solver for EvoCore.
 * 
 * @param solver_name The name of the solver to set.
 * Valid options are "LLT", "LDLT", and "ColPivHouseholderQr".
 * If an unrecognized solver name is provided, the default solver "LDLT" will be set.
 */
void EvoCore::set_solver(std::string const& solver_name) {
    if (solver_name == "LLT") {
        solver = LLTSolver();
        EvoRegression::Log::get_logger()->info("Solver set to LLT");
    }
    else if (solver_name == "LDLT") {
        solver = LDLTSolver();
        EvoRegression::Log::get_logger()->info("Solver set to LDLT");
    }
    else if (solver_name == "ColPivHouseholderQr") {
        solver = ColPivHouseholderQrSolver();
        EvoRegression::Log::get_logger()->info("Solver set to ColPivHouseholderQr");
    }
    else {
        solver = LDLTSolver();
        EvoRegression::Log::get_logger()->info("Unrecognized solver type. Solver set to default LDLT");
    }
}

/**
 * @brief Loads a file and creates regression input from its contents.
 * 
 * This function reads a CSV file specified by the `filename` parameter and parses its contents
 * to create regression input. If an error occurs during the file processing, an appropriate error
 * message is logged. After successful processing, an information message is logged indicating that
 * the file has been loaded.
 * 
 * @param filename The path of the CSV file to be loaded.
 */
void EvoCore::load_file(const std::string& filename) {

    try {
        create_regression_input(parse_csv<double>(filename));
    }
    catch (const std::exception& e) {
        EvoRegression::Log::get_logger()->error(
            "Error processing file {}: {}",
            filename,
            e.what()
        );
    }
    catch (...) {
        EvoRegression::Log::get_logger()->error(
            "An unknown error occurred while processing file {}",
            filename
        );
    }

    EvoRegression::Log::get_logger()->info("File {} loaded",filename);
}

/**
 * Calls the predict method to perform prediction and logs the elapsed time.
 * After prediction, it performs post-processing and logs the result.
 */
void EvoCore::call_predict_method() {
    EvoRegression::Log::get_logger()->info("Starting prediction process...");

    auto start = std::chrono::high_resolution_clock::now();
    predict();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    EvoRegression::Log::get_logger()->info(
        "Prediction process finished in {} seconds",
        elapsed.count()
    );

    titan_postprocessing();
    log_result();
}

void EvoCore::predict() {

    // random engines for each thread
    auto random_engines = Random::create_random_engines(omp_get_max_threads());
    EvoRegression::Log::get_logger()->info("Random engines initilized");

    // caches for each island
    unsigned int cache_size = boundary_conditions.island_generation_size * 10;
    std::vector<LRUCache<std::string, double>> caches(
        boundary_conditions.island_count,
        LRUCache<std::string, double>(cache_size)
    );
    EvoRegression::Log::get_logger()->info("Island caches initialized with size {}", cache_size);

    // create old population as a genofond pool
    std::vector<EvoIndividual> old_population(
        Factory::generate_random_generation(
            boundary_conditions.global_generation_size,
            original_dataset,
            random_engines[0],
            solver
        )
    );

    // create population of newborns
    std::vector<EvoIndividual> population_of_newborns(
        boundary_conditions.global_generation_size
    );

    EvoRegression::Log::get_logger()->info("Starting evolution process...");
    for (unsigned int gen_index = 0; gen_index < boundary_conditions.generation_count; gen_index++) {

        if (gen_index % boundary_conditions.migration_interval == 0 && gen_index != 0) {
            EvoRegression::Log::get_logger()->info("Migration in gen {} started...", gen_index);
            Migration::short_distance_migration(old_population, boundary_conditions.migrants_count, random_engines);
        }

#pragma omp parallel for schedule(guided)
        for (size_t entity_index = 0; entity_index < boundary_conditions.global_generation_size; entity_index++) {

            //get boundaries for island
            size_t thread_id = omp_get_thread_num();
            size_t lower_bound = (entity_index / boundary_conditions.island_generation_size) * boundary_conditions.island_generation_size;

            EvoIndividual newborn = Crossover::cross(
                Selection::tournament_selection(
                    old_population.begin() + lower_bound,
                    boundary_conditions.island_generation_size,
                    random_engines[thread_id]
                ),
                Selection::tournament_selection(
                    old_population.begin() + lower_bound,
                    boundary_conditions.island_generation_size,
                    random_engines[thread_id]
                ),
                original_dataset.predictor.cols(),
                random_engines[thread_id]
            );

            Mutation::mutate(
                newborn,
                original_dataset.predictor.cols(),
                original_dataset.predictor.rows(),
                boundary_conditions.mutation_ratio,
                random_engines[thread_id]
            );

            std::string genotype_key = newborn.to_string_code();

            auto opt_fitness = caches[thread_id].get(genotype_key);
            if (opt_fitness.has_value()) {
                newborn.fitness = opt_fitness.value();
            }
            else {
                newborn.evaluate(
                    EvoMath::get_fitness<std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)>>(
                        Transform::data_transformation_robust(
                            original_dataset.predictor,
                            original_dataset.target,
                            newborn
                        ),
                        solver
                    )
                );
                caches[thread_id].put(genotype_key, newborn.fitness);
            }

            population_of_newborns[entity_index] = std::move(newborn);
        }

        // find best individual in population
        for (const auto& newborn : population_of_newborns)
        {
            titan_evaluation(newborn);
        }

        // move newoborns to old population, they are now old
        old_population = std::move(population_of_newborns);

        // init population of default empty newborns
        population_of_newborns = std::vector<EvoIndividual>(boundary_conditions.global_generation_size);
    }
    EvoRegression::Log::get_logger()->info("Evolution process finished");
}

/**
 * Checks if the EvoCore object is ready to perform predictions.
 * 
 * @return true if the original dataset has predictor and target data, false otherwise.
 */
bool EvoCore::is_ready_to_predict() const {
    return original_dataset.predictor.size() > 0 && original_dataset.target.size() > 0;
}

/**
 * @brief Creates the regression input matrix from the given input tuple.
 * 
 * The regression input matrix consists of a predictor matrix and a target vector.
 * The predictor matrix is initialized with ones and contains the input data columns,
 * excluding the target column. The target vector is filled with the values from the
 * target column of the input data.
 * 
 * @param input The input tuple containing the number of input columns and the input data.
 */
void EvoCore::create_regression_input(std::tuple<int, std::vector<double>> input) {

    std::vector<double> data = std::get<1>(input);

    //input matrix columns (with target column)
    int n_input{ std::get<0>(input) };
    int m_input{ static_cast<int>(data.size()) / n_input };

    // predictor matrix column count (n_input - 1 (because of target column) + 1 + interaction columns)
    int n_output = n_input + boundary_conditions.interaction_cols;

    // mark y column indexed from 0 (is last every time)
    int target_col_index = n_input - 1;

    // initialize predictors matrix to matrix of ones because of x0 and interaction columns
    original_dataset.predictor = Eigen::MatrixXd::Ones(m_input, n_output);
    original_dataset.target.resize(m_input, 1);

    for (int row = 0; row < m_input; ++row) {
        for (int col = 0; col < n_input; ++col) {
            // last column is always Y or in other words regressant, dependant variable
            if (col == target_col_index) {
                original_dataset.target(row, 0) = data[col + n_input * row];
            }
            // fil predictors (first is x0 column of 1, last are interaction filled default to 1 too - but they are able to mutate)
            if (col < target_col_index) {
                original_dataset.predictor(row, col + 1) = data[col + n_input * row];
            }
        }
    }

    EvoRegression::Log::get_logger()->info(
        "Predictor matrix initialized with {} rows and {} columns",
        original_dataset.predictor.rows(),
        original_dataset.predictor.cols()
    );
}

void EvoCore::setTitan(EvoIndividual titan) {
    this->titan = titan;
    EvoRegression::Log::get_logger()->info("New titan found with fitness {}", titan.fitness);
}

void EvoCore::titan_evaluation(EvoIndividual const& individual) {
    if (individual.fitness < titan.fitness) setTitan(individual);
}

/**
 * Performs postprocessing for the Titan dataset.
 * This function applies data transformation to remove outliers from the original dataset,
 * and then solves the regression system using the transformed data.
 * 
 * @remarks This function assumes that the original dataset, titan, and the necessary transformations
 *          have already been initialized.
 */
void EvoCore::titan_postprocessing() {
    EvoRegression::Log::get_logger()->info("Titan postprocessing has begun.");
    // data without outliers
    titan_dataset_robust = Transform::data_transformation_robust(original_dataset.predictor, original_dataset.target, titan);
    // data without outliers
    titan_dataset_nonrobust = Transform::data_transformation_nonrobust(original_dataset.predictor, original_dataset.target, titan);
    // regression result
    titan_result = solve_system_detailed(titan_dataset_robust.predictor, titan_dataset_robust.target);
    EvoRegression::Log::get_logger()->info("Titan postprocessing finished.");
}

/**
 * Logs the regression results.
 * This function generates a summary table of the regression results and logs it using the logger.
 * The table includes the regression result tables, result metrics table, regression coefficients table,
 * genotype table, and the formula table.
 */
void EvoCore::log_result() {
    EvoRegression::Log::get_logger()->info("Logging results...");
    std::stringstream table;
    table << EvoRegression::get_regression_result_table(
        EvoRegression::get_regression_summary_matrix(
            titan,
            titan_result.theta,
            titan_dataset_nonrobust
        ).data(),
        titan_dataset_nonrobust.target.size() * 4
    );
    table << EvoRegression::get_regression_robust_result_table(
        EvoRegression::get_regression_summary_matrix(
            titan,
            titan_result.theta,
            titan_dataset_robust
        ).data(),
        titan_dataset_robust.target.size() * 4
    );
    table << EvoRegression::get_result_metrics_table(
        {
            DescriptiveStatistics::median(titan_dataset_nonrobust.target.data(), titan_dataset_nonrobust.target.size()),
            titan_result.standard_deviation,
            titan_result.rsquared
        }
    );
    table << EvoRegression::get_regression_coefficients_table(titan_result.theta.data(), titan_result.theta.size());
    table << EvoRegression::get_genotype_table(titan);
    table << EvoRegression::get_formula_table({ titan.to_math_formula() });
    EvoRegression::Log::get_logger()->info("Regression summary:\n{}", table.str());
}