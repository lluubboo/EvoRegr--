#include "EvoCore.hpp"
#include "Log.hpp"
#include "IOTools.hpp"
#include "RandomChoices.hpp"

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
 *                    Valid options are "LLT", "LDLT", and "ColPivHouseholderQr".
 *                    If an unrecognized solver name is provided, the default solver "LDLT" will be set.
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

void EvoCore::call_predict_method() {

    EvoRegression::Log::get_logger()->info("Starting batch prediction process...");

    auto random_engines = Random::create_random_engines(omp_get_max_threads());

    auto start_time = std::chrono::high_resolution_clock::now();

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
    std::vector<EvoIndividual> newborns_population(
        boundary_conditions.global_generation_size
    );

    for (unsigned int gen_index = 0; gen_index < boundary_conditions.generation_count; gen_index++) {

        if (gen_index % boundary_conditions.migration_interval == 0 && gen_index != 0) {

            EvoRegression::Log::get_logger()->info("Migration in gen {} started...", gen_index);
            Migration::short_distance_migration(old_population, boundary_conditions.migrants_count, random_engines);

        }

#pragma omp parallel for schedule(guided)
        for (size_t entity_index = 0; entity_index < boundary_conditions.global_generation_size; entity_index++) {

            //get boundaries for island
            size_t thread_id = omp_get_thread_num();
            size_t lower_bound = (entity_index / boundary_conditions.island_generation_size) *
                boundary_conditions.island_generation_size;


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

            //evaluate
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

            newborns_population[entity_index] = std::move(newborn);
        }

        // find best individual in population
        for (const auto& newborn : newborns_population)
        {
            titan_evaluation(newborn);
        }

        // move newoborns to old population, they are now old
        old_population = std::move(newborns_population);

        // init population of some default newborns
        newborns_population = std::vector<EvoIndividual>(boundary_conditions.global_generation_size);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    EvoRegression::Log::get_logger()->info("Batch prediction process took {} seconds.", duration / 1000.0);

    //log_result();
}

/**
 * Checks if the EvoCore object is ready to perform predictions.
 * 
 * @return true if the original dataset has predictor and target data, false otherwise.
 */
bool EvoCore::is_ready_to_predict() const {
    return original_dataset.predictor.size() > 0 && original_dataset.target.size() > 0;
}

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

void EvoCore::titan_postprocessing() {
    // data without outliers
    titan_dataset_robust = Transform::data_transformation_robust(original_dataset.predictor, original_dataset.target, titan);
    // data witho outliers
    titan_dataset_nonrobust = Transform::data_transformation_nonrobust(original_dataset.predictor, original_dataset.target, titan);
    // regression result
    titan_result = solve_system_detailed(titan_dataset_robust.predictor, titan_dataset_robust.target);

    EvoRegression::Log::get_logger()->info("Titan postprocessing finished");
}