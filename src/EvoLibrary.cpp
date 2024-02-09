#include <unordered_set>
#include "EvoLibrary.hpp"
#include "RandomChoices.hpp"
#include "RandomNumberGenerator.hpp"

/**
 * @brief Performs tournament selection on a population of individuals.
 *
 * This function selects two random individuals from the population and returns the one with the lower fitness.
 * The selection is performed using a tournament approach, where the "winner" is the individual with the lower fitness.
 *
 * @param begin Iterator pointing to the beginning of the population.
 * @param size The size of the population.
 * @param random_engine A reference to a random number generator.
 * @return A const reference to the selected individual.
 */
EvoIndividual const& Selection::tournament_selection(std::vector<EvoIndividual>::iterator begin, size_t size, XoshiroCpp::Xoshiro256Plus& random_engine) {
    const auto& individual1 = *(begin + RandomNumbers::rand_interval_int(0, size - 1, random_engine));
    const auto& individual2 = *(begin + RandomNumbers::rand_interval_int(0, size - 1, random_engine));
    return (individual1.fitness < individual2.fitness) ? individual1 : individual2;
};

/**
 * Performs short-distance migration on the given population.
 *
 * @param population The population of EvoIndividuals to perform migration on.
 * @param migration_size The number of individuals to migrate.
 * @param random_engines The random engines used for random number generation.
 */
void Migration::short_distance_migration(std::vector<EvoIndividual>& population, size_t migration_size, std::vector<XoshiroCpp::Xoshiro256Plus>& random_engines) {

    size_t num_threads = omp_get_max_threads();

    // calculate batch size, each thread will perform migration on a batch of individuals
    // elements are migrating only inside batches
    // the last thread will perform migration on the remaining individuals
    size_t batch_size = migration_size / num_threads;

    // do short-distance migration
#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < migration_size; i++) {

        unsigned int thread_num = omp_get_thread_num();

        size_t start = thread_num * batch_size;
        size_t end = (thread_num == num_threads - 1) ? migration_size : start + batch_size - 1;

        std::swap(
            population[RandomNumbers::rand_interval_int(start, end, random_engines[thread_num])],
            population[RandomNumbers::rand_interval_int(start, end, random_engines[thread_num])]
        );
    }
};

/**
 * @brief Generates a random EvoIndividual object.
 *
 * @param predictor_row_count The number of rows in the predictor matrix.
 * @param predictor_column_count The number of columns in the predictor matrix.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return An EvoIndividual object with randomly generated chromosomes.
 *
 * This function generates a random EvoIndividual object, which represents a potential solution in a genetic algorithm.
 * The EvoIndividual object contains several chromosomes, each of which is a vector of alleles.
 * The function generates these chromosomes by calling other functions in the Factory class to generate random alleles.
 */
EvoIndividual Factory::get_random_evo_individual(EvoBoundaryConditions boundaries, EvoRegression::EvoDataSet const& dataset, XoshiroCpp::Xoshiro256Plus& random_engine) {
    EvoIndividual individual{};
    int training_row_count = dataset.predictor.rows();
    int column_count = dataset.predictor.cols();
    // create genofond
    for (int i = 0; i < column_count; i++)
    {
        individual.merger_chromosome.push_back(Factory::get_random_merge_allele(i, column_count, boundaries.basis_function_complexity, random_engine));
        individual.x_transformer_chromosome.push_back(Factory::get_random_transform_xallele(i, random_engine));
    }
    individual.tr_robuster_chromosome.push_back(Factory::get_random_robust_allele(training_row_count, boundaries.robustness, random_engine));
    individual.y_transformer_chromosome.push_back(Factory::get_random_transform_yallele(random_engine));
    return individual;
}

/**
 * @brief Generates a random generation of individuals.
 *
 * This function generates a random generation of `EvoIndividual` objects and evaluates their fitness.
 * The size of the generation is determined by the `size` parameter.
 * Each individual is generated using the `getRandomEvoIndividual` function and its fitness is evaluated using the `get_fitness` function.
 *
 * @param size The number of individuals in the generation.
 * @param dataset A `Transform::EvoDataSet` object containing the dataset used to evaluate the fitness of the individuals.
 * @param random_engine A `XoshiroCpp::Xoshiro256Plus` random number generator used to generate random individuals.
 * @param solver A function that calculates the regression result for a given predictor and response matrix.
 *
 * @return std::vector<EvoIndividual> The generated generation of individuals.
 */
std::vector<EvoIndividual> Factory::generate_random_generation(
    EvoBoundaryConditions boundary_conditions,
    EvoRegression::EvoDataSet dataset,
    XoshiroCpp::Xoshiro256Plus& random_engine,
    std::function<double(EvoRegression::EvoDataSet& dataset, EvoBoundaryConditions const&)> solver
) {
    std::vector<EvoIndividual> generation(boundary_conditions.global_generation_size);
    std::generate(generation.begin(), generation.end(), [&]() {return Factory::get_random_evo_individual(boundary_conditions, dataset, random_engine);});
    std::for_each(generation.begin(), generation.end(), [&](EvoIndividual& individual) {individual.evaluate(
        EvoMath::get_fitness<std::function<double(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&)>>(
            Transform::transform_dataset_copy(dataset, individual, true),
            boundary_conditions,
            solver
        )
    );});
    return generation;
};

/**
 * @brief Generates a random MergeAllele object.
 *
 * @param column_index The index of the column in the predictor matrix.
 * @param predictor_column_count The number of columns in the predictor matrix.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return A MergeAllele object with randomly generated alleles.
 *
 * This function generates a random MergeAllele object, which represents a potential solution in a genetic algorithm.
 * The MergeAllele object contains several alleles, each of which is a vector of MergeTwins.
 * The function generates these alleles by calling other functions in the Factory class to generate random MergeTwins.
 */
MergeAllele Factory::get_random_merge_allele(int column_index, int predictor_column_count, int basis_function_complexity, XoshiroCpp::Xoshiro256Plus& random_engine) {
    MergeAllele merge_allele{ column_index };
    // column index 0 marks x0 (no merging)
    if (column_index != 0) {
        merge_allele.allele = EExpression{ predictor_column_count, basis_function_complexity, random_engine };
    }
    return merge_allele;
};

/**
 * @brief Generates a random TransformXAllele object.
 *
 * @param column_index The index of the column in the predictor matrix.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return A TransformXAllele object with randomly generated alleles.
 *
 * This function generates a random TransformXAllele object, which represents a potential solution in a genetic algorithm.
 * The TransformXAllele object contains several alleles, each of which is a Transform_operator.
 * The function generates these alleles by calling other functions in the Factory class to generate random Transform_operators.
 */
TransformXAllele Factory::get_random_transform_xallele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine) {
    TransformXAllele transformx_allele{ column_index };
    if (column_index != 0) {
        transformx_allele.allele = Transform_operator{ RandomNumbers::rand_interval_int(0, transform_operator_maxindex, random_engine) };
        if (transformx_allele.allele == Transform_operator::Pow || transformx_allele.allele == Transform_operator::Wek) transformx_allele.resetCharacteristicNumber(RandomNumbers::rand_interval_decimal_number(1., 3., random_engine));
    }
    else {
        transformx_allele.allele = Transform_operator::Let;
    }
    return transformx_allele;
};

/**
 * @brief Generates a random TransformYAllele object.
 *
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return A TransformYAllele object with randomly generated alleles.
 *
 * This function generates a random TransformYAllele object, which represents a potential solution in a genetic algorithm.
 * The TransformYAllele object contains a Transform_operator allele, which is randomly generated.
 * If the generated Transform_operator is Pow or Wek, the function also generates a characteristic number for the TransformYAllele.
 */
TransformYAllele Factory::get_random_transform_yallele(XoshiroCpp::Xoshiro256Plus& random_engine) {
    TransformYAllele transformy_allele{};
    transformy_allele.allele = Transform_operator{ RandomNumbers::rand_interval_int(0, transform_y_operator_maxindex, random_engine) };
    if (transformy_allele.allele == Transform_operator::Pow || transformy_allele.allele == Transform_operator::Wek) transformy_allele.resetCharacteristicNumber(RandomNumbers::rand_interval_decimal_number(1., 3., random_engine));
    return transformy_allele;
};

/**
 * @brief Generates a random RobustAllele object.
 *
 * @param row_count The number of rows in the predictor matrix.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return A RobustAllele object with randomly generated alleles.
 *
 * This function generates a random RobustAllele object, which represents a potential solution in a genetic algorithm.
 * The RobustAllele object contains a vector of integers representing the rows of the predictor matrix that should be used.
 * The function generates this vector by randomly selecting a subset of the rows from the predictor matrix.
 */
RobustAllele Factory::get_random_robust_allele(int row_count, float robustness, XoshiroCpp::Xoshiro256Plus& random_engine) {
    RobustAllele robust_allele{};
    int rows_to_erase = row_count * robustness;
    std::vector<int> choosen_rows(row_count);
    std::iota(begin(choosen_rows), end(choosen_rows), 0);
    std::shuffle(choosen_rows.begin(), choosen_rows.end(), random_engine);
    choosen_rows.erase(choosen_rows.end() - rows_to_erase, choosen_rows.end());
    std::sort(choosen_rows.begin(), choosen_rows.end());
    robust_allele.allele = choosen_rows;
    return robust_allele;
};

void Crossover::cross(EvoIndividual& child, const EvoIndividual& parent1, const EvoIndividual& parent2, int chromosome_size, XoshiroCpp::Xoshiro256Plus& random_engine) {

    // indexes which points to place of chromosome cut & recombination
    int m_crossover_twist_index = RandomNumbers::rand_interval_int(0, chromosome_size, random_engine);
    int t_crossover_twist_index = RandomNumbers::rand_interval_int(0, chromosome_size, random_engine);
    int rtr_crossover_twist_index = RandomNumbers::rand_interval_int(0, 1, random_engine);
    int y_crossover_twist_index = RandomNumbers::rand_interval_int(0, 1, random_engine);

    // cross single gene chromosomes robuster & ytransformer
    child.tr_robuster_chromosome = (rtr_crossover_twist_index == 0) ? parent1.tr_robuster_chromosome : parent2.tr_robuster_chromosome;
    child.y_transformer_chromosome = (y_crossover_twist_index == 0) ? parent1.y_transformer_chromosome : parent2.y_transformer_chromosome;

    // cross multi gene chromosomes merger & xtransformer
    child.merger_chromosome = parent1.merger_chromosome;
    child.x_transformer_chromosome = parent1.x_transformer_chromosome;

    std::copy(parent2.merger_chromosome.begin() + m_crossover_twist_index, parent2.merger_chromosome.end(), child.merger_chromosome.begin() + m_crossover_twist_index);
    std::copy(parent2.x_transformer_chromosome.begin() + t_crossover_twist_index, parent2.x_transformer_chromosome.end(), child.x_transformer_chromosome.begin() + t_crossover_twist_index);
}

/**
 * @brief Performs mutation operation on an EvoIndividual.
 *
 * @param individual The EvoIndividual object that is to be mutated.
 * @param chromosome_size The size of the chromosomes in the EvoIndividual object.
 * @param predictor_row_count The number of rows in the predictor matrix.
 * @param mutation_rate The mutation rate of the genetic algorithm.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return The mutated EvoIndividual object.
 *
 * This function performs a mutation operation, a method of generating new individuals in a genetic algorithm.
 * It works by selecting a random mutation point in the chromosomes of the individual, and then mutating the allele at that point.
 * The function first selects a random mutation point in the chromosomes of the individual.
 * Then, it selects a random chromosome to mutate, and finally mutates the allele at the selected mutation point.
 */
void Mutation::mutate(EvoIndividual& individual, int chromosome_size, int predictor_row_count, int mutation_rate, int basis_function_complexity, float robustness, XoshiroCpp::Xoshiro256Plus& random_engine) {
    int rand_num = RandomNumbers::rand_interval_int(0, 100, random_engine);
    if (rand_num <= mutation_rate) {
        int mutation_index = RandomNumbers::rand_interval_int(0, 3, random_engine);
        if (mutation_index == 0) {
            int col = RandomNumbers::rand_interval_int(0, chromosome_size - 1, random_engine);
            individual.x_transformer_chromosome.at(col) = Factory::get_random_transform_xallele(col, random_engine);
        }
        if (mutation_index == 1) {
            int col = RandomNumbers::rand_interval_int(0, chromosome_size - 1, random_engine);
            individual.merger_chromosome.at(col) = Factory::get_random_merge_allele(col, chromosome_size, basis_function_complexity, random_engine);
        }
        if (mutation_index == 2) {
            individual.tr_robuster_chromosome.at(0) = Factory::get_random_robust_allele(predictor_row_count, robustness, random_engine);
        }
        if (mutation_index == 3) {
            individual.y_transformer_chromosome.at(0) = Factory::get_random_transform_yallele(random_engine);
        }
    }
}

/**
 * @brief Transforms a predictor matrix based on the characteristics of an EvoIndividual.
 *
 * @param matrix The predictor matrix to be transformed.
 * @param individual The EvoIndividual object that defines the transformation.
 *
 * @return The transformed predictor matrix.
 *
 * This function transforms a predictor matrix based on the characteristics of an EvoIndividual.
 * It first erases some rows based on the robuster_chromosome of the EvoIndividual.
 * Then, it merges predictors based on the merger_chromosome of the EvoIndividual.
 * Finally, it transforms predictors based on the x_transformer_chromosome of the EvoIndividual.
 */
void Transform::full_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual const& individual) {
    individual.tr_robuster_chromosome.at(0).transform(matrix);
    Transform::half_predictor_transform(matrix, individual);
};

/**
 * @brief Performs a partial transformation on a predictor matrix based on the characteristics of an EvoIndividual.
 *
 * @param matrix The predictor matrix to be transformed.
 * @param individual The EvoIndividual object that defines the transformation.
 *
 * @return The partially transformed predictor matrix.
 *
 * This function performs a partial transformation on a predictor matrix based on the characteristics of an EvoIndividual.
 * It first merges predictors based on the merger_chromosome of the EvoIndividual.
 * Then, it transforms predictors based on the x_transformer_chromosome of the EvoIndividual.
 * Unlike the full_predictor_transform function, this function does not erase any rows from the predictor matrix.
 */
void Transform::half_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual const& individual) {
    std::for_each(individual.merger_chromosome.begin(), individual.merger_chromosome.end(), [&matrix](auto& item) { item.transform(matrix); });
    std::for_each(individual.x_transformer_chromosome.begin(), individual.x_transformer_chromosome.end(), [&matrix](auto& item) { item.transform(matrix); });
};

/**
 * @brief Performs a robust transformation on a predictor matrix based on the characteristics of an EvoIndividual.
 *
 * @param matrix The predictor matrix to be transformed.
 * @param individual The EvoIndividual object that defines the transformation.
 *
 * @return The robustly transformed predictor matrix.
 *
 * This function performs a robust transformation on a predictor matrix based on the characteristics of an EvoIndividual.
 * It erases some rows from the predictor matrix based on the robuster_chromosome of the EvoIndividual.
 */
void Transform::robust_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual const& individual) {
    individual.tr_robuster_chromosome.at(0).transform(matrix);
};

/**
 * @brief Performs a full transformation on a target vector based on the characteristics of an EvoIndividual.
 *
 * @param vector The target vector to be transformed.
 * @param individual The EvoIndividual object that defines the transformation.
 *
 * @return The fully transformed target vector.
 *
 * This function performs a full transformation on a target vector based on the characteristics of an EvoIndividual.
 * It first applies a robust transformation on the vector based on the robuster_chromosome of the EvoIndividual.
 * Then, it applies a transformation on the vector based on the y_transformer_chromosome of the EvoIndividual.
 */
void Transform::full_target_transform(Eigen::VectorXd& vector, EvoIndividual const& individual) {
    individual.tr_robuster_chromosome.at(0).transform_vector(vector);
    Transform::half_target_transform(vector, individual);
};

/**
 * @brief Performs a half transformation on a target vector based on the characteristics of an EvoIndividual.
 *
 * @param vector The target vector to be transformed.
 * @param individual The EvoIndividual object that defines the transformation.
 *
 * @return The half transformed target vector.
 *
 * This function performs a half transformation on a target vector based on the characteristics of an EvoIndividual.
 * It applies a transformation on the vector based on the y_transformer_chromosome of the EvoIndividual.
 * Unlike the full_target_transform function, this function does not apply any robust transformation.
 */
void Transform::half_target_transform(Eigen::VectorXd& vector, EvoIndividual const& individual) {
    individual.y_transformer_chromosome.at(0).transform_vector(vector);
};

EvoRegression::EvoDataSet& Transform::transform_dataset(EvoRegression::EvoDataSet& dataset, EvoIndividual const& individual, bool robust) {
    // robust = filtering outliers
    if (robust) {
        Transform::full_predictor_transform(dataset.predictor, individual);
        Transform::full_target_transform(dataset.target, individual);
    }
    else {
        Transform::half_predictor_transform(dataset.predictor, individual);
        Transform::half_target_transform(dataset.target, individual);
    }
    return dataset;
};

EvoRegression::EvoDataSet Transform::transform_dataset_copy(EvoRegression::EvoDataSet dataset, EvoIndividual const& individual, bool robust) {
    // robust = filtering outliers
    if (robust) {
        Transform::full_predictor_transform(dataset.predictor, individual);
        Transform::full_target_transform(dataset.target, individual);
    }
    else {
        Transform::half_predictor_transform(dataset.predictor, individual);
        Transform::half_target_transform(dataset.target, individual);
    }
    return dataset;
};

/**
 * @brief Calculates the fitness of a solver on a given dataset.
 *
 * @tparam T The type of the solver.
 * @param dataset The dataset on which the fitness of the solver is to be calculated.
 * @param solver The solver whose fitness is to be calculated.
 *
 * @return The fitness of the solver, represented as the sum of squares of errors.
 *
 * This function calculates the fitness of a solver on a given dataset.
 * The fitness is calculated as the sum of squares of errors of the solver on the dataset.
 */
template <typename T>
double EvoMath::get_fitness(EvoRegression::EvoDataSet& dataset, EvoBoundaryConditions const& boundary_conditions, T solver) {
    return solver(dataset, boundary_conditions);
}

// Explicit instantiation
template double EvoMath::get_fitness(EvoRegression::EvoDataSet& dataset, EvoBoundaryConditions const&, std::function<double(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&)> solver);

template <typename T>
double EvoMath::get_fitness(EvoRegression::EvoDataSet&& dataset, EvoBoundaryConditions const& boundary_conditions, T solver) {
    return solver(dataset, boundary_conditions);
}

template double EvoMath::get_fitness(EvoRegression::EvoDataSet&& dataset, EvoBoundaryConditions const&, std::function<double(EvoRegression::EvoDataSet&, EvoBoundaryConditions const&)> solver);

/**
 * @brief Extracts a column from a 1D vector representing a 2D matrix.
 *
 * @tparam T The type of the elements in the data vector.
 * @param data The 1D vector representing a 2D matrix.
 * @param column_count The number of columns in the 2D matrix.
 * @param column_index The index of the column to be extracted.
 *
 * @return A vector representing the extracted column.
 *
 * This function extracts a column from a 1D vector that represents a 2D matrix.
 * The 2D matrix is assumed to have a row-major layout in the 1D vector.
 * The function returns a new vector that contains the elements of the extracted column.
 */
template <typename T>
std::vector<T> EvoMath::extract_column(std::vector<T> data, unsigned int column_count, unsigned int column_index) {
    int row_count = data.size() / column_count;
    std::vector<T> column(row_count);
    std::generate(column.begin(), column.end(), [&, i = 0]() mutable {
        return data[i++ * column_count + column_index];}
    );
    return column;
};

//explicit instantiation
template std::vector<int> EvoMath::extract_column(std::vector<int> data, unsigned int column_count, unsigned int column_index);
template std::vector<double> EvoMath::extract_column(std::vector<double> data, unsigned int column_count, unsigned int column_index);
template std::vector<float> EvoMath::extract_column(std::vector<float> data, unsigned int column_count, unsigned int column_index);

std::vector<XoshiroCpp::Xoshiro256Plus> Random::create_random_engines(size_t count) {
    //create random seed for master random engine
    std::random_device rd;
    uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | rd();

    //create master random engine
    XoshiroCpp::Xoshiro256Plus master_random_engine(seed);

    //create n random engines and store them in vector
    //each one is created by long jumping the master random engine
    std::vector<XoshiroCpp::Xoshiro256Plus> random_engines;
    for (unsigned int i = 0;i < count;i++) {
        master_random_engine.longJump();
        random_engines.emplace_back(master_random_engine.serialize());
    }

    return random_engines;
};