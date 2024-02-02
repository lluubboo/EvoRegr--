#include "EvoCore.hpp"
#include "Stats.hpp"
#include "Log.hpp"
#include "IOTools.hpp"
#include "RandomChoices.hpp"
#include "EvoResultPostprocessing.hpp"


EvoCore::EvoCore() :
    original_dataset(),
    titan_dataset_training(),
    titan_dataset_test(),
    boundary_conditions(),
    solver(LDLTSolver()),
    titan(),
    island_titans(),
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
        prepare_input_datasets(parse_csv<double>(filename));
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

    EvoRegression::Log::get_logger()->info("File {} loaded", filename);
}


/**
 * Prepares the input datasets for training by adding ones to the input vector and splitting the data into training and testing sets.
 *
 * @param input A tuple containing the number of columns in the input matrix and the input data as a vector of doubles.
 */
void EvoCore::prepare_input_datasets(std::tuple<int, std::vector<double>> input) {

    // get input data
    std::vector<double> data = std::get<1>(input);

    // input matrix dimensions
    int input_cols = std::get<0>(input);
    int input_rows = data.size() / input_cols;

    // add ones to the input vector as per linear regression convention
    data.insert(data.begin(), input_rows, 1);

    // get result matrix dimensions
    int output_cols = input_cols + 1; // added one column of ones
    int output_rows = input_rows;

    // Map vector to a matrix (last column is the target vector)
    Eigen::MatrixXd raw_data_matrix = Eigen::Map<Eigen::MatrixXd>(data.data(), output_rows, output_cols);

    // populate datasets
    split_dataset(raw_data_matrix, 0.8);

    EvoRegression::Log::get_logger()->info(
        "Predictor training matrix initialized with {} rows and {} columns",
        original_dataset.predictor.rows(),
        original_dataset.predictor.cols()
    );
}

/**
 * Splits the given dataset into training and testing datasets according to the specified ratio.
 *
 * @param dataset The dataset to be split.
 * @param ratio The ratio of training data to total data.
 */
void EvoCore::split_dataset(Eigen::MatrixXd& dataset, double ratio) {
    // split raw matrix to test & training predictor and target matrices
    original_dataset.predictor = dataset.block(0, 0, dataset.rows(), dataset.cols() - 1).eval();
    original_dataset.target = dataset.block(0, dataset.cols() - 1, dataset.rows(), 1).eval();
};

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

/**
 * Prepares the EvoCore object for the prediction process.
 */
void EvoCore::prepare_for_prediction() {

    EvoRegression::Log::get_logger()->info("Preparing for evolution process...");

    // random engines for each thread
    random_engines = Random::create_random_engines(omp_get_max_threads());
    EvoRegression::Log::get_logger()->info("Random engines initilized");

    // allocated memory for matrices
    EvoRegression::Log::get_logger()->info("Trying to allocate memory for matrices...");
    compute_datasets = std::vector(boundary_conditions.island_count, original_dataset);
    EvoRegression::Log::get_logger()->info("Memory allocated for matrices");

    // caches for each island
    unsigned int cache_size = boundary_conditions.island_generation_size * 10;
    caches = std::vector<LRUCache<std::string, double>>(boundary_conditions.island_count, LRUCache<std::string, double>(cache_size));
    EvoRegression::Log::get_logger()->info("Island caches initialized with size {}", cache_size);

    // titans
    island_titans = std::vector<EvoIndividual>(boundary_conditions.island_count);
    EvoRegression::Log::get_logger()->info("Island titans initialized");

    //elites 
    island_gen_elite_groups = std::vector<std::set<EvoIndividual>>(boundary_conditions.island_count);

    // create old population as a genofond pool
    pensioners = std::vector<EvoIndividual>(
        Factory::generate_random_generation(
            boundary_conditions.global_generation_size,
            original_dataset,
            random_engines[0],
            solver
        )
    );
    EvoRegression::Log::get_logger()->info("Generation zero created");

    // create population of newborns
    newborns = std::vector<EvoIndividual>(
        boundary_conditions.global_generation_size
    );
    EvoRegression::Log::get_logger()->info("Population of newborns initialized");
    EvoRegression::Log::get_logger()->info("Preparation finished");
}

void EvoCore::predict() {

    // initialize random engines, caches, containers, create gen zero...
    prepare_for_prediction();

    EvoRegression::Log::get_logger()->info("Starting evolution process...");
    for (unsigned int gen_index = 0; gen_index < boundary_conditions.generation_count; gen_index++) {

        if (gen_index % boundary_conditions.migration_interval == 0 && gen_index != 0) {
            EvoRegression::Log::get_logger()->info("Migration in gen {} started...", gen_index);
            Migration::short_distance_migration(pensioners, boundary_conditions.migrants_count, random_engines);
        }

        if (gen_index % 20 == 0 && gen_index != 0) {
            log_island_titans(gen_index);
        }

        // loop through entities in island
        int cache_hits = 0;

        // loop through islands
#pragma omp parallel for schedule(guided)
        for (size_t island_index = 0; island_index < boundary_conditions.island_count; island_index++) {

            for (size_t entity_index = boundary_conditions.island_borders[island_index][0]; entity_index <= boundary_conditions.island_borders[island_index][1]; entity_index++) {

                // indexes
                size_t thread_id = omp_get_thread_num();

                // reset workspace to original state
                compute_datasets[island_index] = original_dataset;

                // newborn reference
                EvoIndividual& newborn = newborns[entity_index];

                Crossover::cross(
                    newborn,
                    Selection::tournament_selection(
                        pensioners.begin() + boundary_conditions.island_borders[island_index][0],
                        boundary_conditions.island_generation_size,
                        random_engines[thread_id]
                    ),
                    Selection::tournament_selection(
                        pensioners.begin() + boundary_conditions.island_borders[island_index][0],
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

                auto opt_fitness = caches[island_index].get(genotype_key);

                if (opt_fitness.has_value()) {
                    newborn.fitness = opt_fitness.value();
                }
                else {
                    newborn.evaluate(
                        EvoMath::get_fitness<std::function<double(EvoRegression::EvoDataSet const& dataset)>>(Transform::transform_dataset(compute_datasets[island_index], newborn, true), solver)
                    );
                    caches[island_index].put(genotype_key, newborn.fitness);
                }
            }
        }

        if (gen_index % 20 == 0 && gen_index != 0) EvoRegression::Log::get_logger()->info("Cache hit ratio: {}", cache_hits);

        // move newoborns to old population, they are now old
        std::swap(pensioners, newborns);

        // generation postprocessing
        clear_elite_groups();
        rank_past_generation();
        move_elites();
    }

    // postprocessing
    find_titan();
    EvoRegression::Log::get_logger()->info("Evolution process finished");
}

void EvoCore::rank_past_generation() {
    // find group of generation elites for each island

#pragma omp parallel for schedule(guided)
    for (size_t island_index = 0; island_index < boundary_conditions.island_count; island_index++) {

        auto const& island_borders = boundary_conditions.island_borders[island_index];
        auto& island_gen_elite_group = island_gen_elite_groups[island_index];

        for (size_t entity_index = island_borders[0]; entity_index <= island_borders[1]; entity_index++) {

            EvoIndividual const& entity = pensioners[entity_index];

            if (island_gen_elite_group.empty() || entity.fitness < island_gen_elite_group.rbegin()->fitness) { // Check if the current entity is better than the worst elite
                {

                    if (island_gen_elite_group.size() == boundary_conditions.elites_count) { // If the elite group is full, remove the worst elite
                        island_gen_elite_group.erase(--island_gen_elite_group.end());
                    }

                    island_gen_elite_group.insert(entity);
                }
            }
        }
    }
}

void EvoCore::clear_elite_groups() {
    for (auto& elite_group : island_gen_elite_groups) {
        elite_group.clear();
    }
}

void EvoCore::move_elites() {
#pragma omp parallel for schedule(guided)
    for (size_t island_index = 0; island_index < boundary_conditions.island_count; island_index++) {

        auto const& elite_group = island_gen_elite_groups[island_index];

        //copy each elite individual at random position in island

        for (auto const& elite : elite_group) {

            size_t random_island_position = RandomNumbers::rand_interval_int(
                boundary_conditions.island_borders[island_index][0],
                boundary_conditions.island_borders[island_index][1],
                random_engines[0]
            );

            pensioners[random_island_position] = elite;
        }
    }
}

/**
 * Ranks the island titans based on their fitness.
 * For each island titan, it performs titan evaluation and logs the fitness.
 */
void EvoCore::find_titan() {
    for (auto const& elite_group : island_gen_elite_groups) {
        titan_evaluation(*elite_group.begin());
    }
    EvoRegression::Log::get_logger()->info("All-time titan fitness {}", titan.fitness);
}

void EvoCore::log_island_titans(int gen_index) {
    EvoRegression::Log::get_logger()->info("Gen {} island titans: ", gen_index);
    for (auto const& elite_group : island_gen_elite_groups) {
        EvoRegression::Log::get_logger()->info("{}", elite_group.begin()->fitness);
    }
}

/**
 * Checks if the EvoCore object is ready to perform predictions.
 *
 * @return true if the original dataset has predictor and target data, false otherwise.
 */
bool EvoCore::is_ready_to_predict() const {
    return original_dataset.predictor.size() > 0 && original_dataset.target.size() > 0;
}

void EvoCore::setTitan(EvoIndividual titan) {
    this->titan = titan;
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

    // get number of rows
    double splitting_ratio_fraction = static_cast<double>(boundary_conditions.splitting_ratio) / 100.0;
    int result_rows = static_cast<int>(original_dataset.predictor.rows() * splitting_ratio_fraction);

    titan_dataset_training = Transform::transform_dataset_copy(original_dataset, titan, true);
    titan_dataset_training.predictor = titan_dataset_training.predictor.topRows(original_dataset.predictor.rows() - result_rows);
    titan_dataset_training.target = titan_dataset_training.target.topRows(original_dataset.predictor.rows() - result_rows);

    titan_dataset_test = Transform::transform_dataset_copy(original_dataset, titan, true);
    titan_dataset_test.predictor = titan_dataset_test.predictor.bottomRows(result_rows);
    titan_dataset_test.target = titan_dataset_test.target.bottomRows(result_rows);

    // titan detailed result
    titan_result = solve_system_detailed(titan_dataset_training.predictor, titan_dataset_training.target);

    // result matrices
    training_result = EvoRegression::get_regression_summary_matrix(titan, titan_result.theta, titan_dataset_training);
    testing_result = EvoRegression::get_regression_summary_matrix(titan, titan_result.theta, titan_dataset_test);

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

    table << EvoRegression::get_regression_training_table(
        training_result.data(),
        titan_dataset_training.target.size() * 4
    );

    table << EvoRegression::get_regression_testing_table(
        testing_result.data(),
        titan_dataset_test.target.size() * 4
    );

    table << EvoRegression::get_result_metrics_table(
        {
            DescriptiveStatistics::median(testing_result.col(2).data(), testing_result.col(2).size()),
            DescriptiveStatistics::standard_deviation(testing_result.col(2).data(), testing_result.col(2).size()),
            0,
        }
    );

    table << EvoRegression::get_regression_coefficients_table(titan_result.theta.data(), titan_result.theta.size());
    table << EvoRegression::get_genotype_table(titan);
    table << EvoRegression::get_formula_table({ titan.to_math_formula() });
    EvoRegression::Log::get_logger()->info("Regression summary:\n{}", table.str());
}