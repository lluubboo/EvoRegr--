#include "EvoLibrary.hpp"
#include "RandomChoices.hpp"
#include "RandomNumberGenerator.hpp"

/**
 * @brief Performs tournament selection in a genetic algorithm.
 *
 * @param generation A vector of EvoIndividual objects representing the current generation.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return An array of two EvoIndividual objects that have been selected as parents for the next generation.
 *
 * This function performs tournament selection, a method of selecting individuals in a genetic algorithm.
 * It works by choosing a number of individuals at random from the population, and then selecting the best individual
 * out of that group to become a parent. This process is repeated to select the second parent.
 */
std::array<EvoIndividual, 2> Selection::tournament_selection(std::vector<EvoIndividual> const& generation, XoshiroCpp::Xoshiro256Plus& random_engine) {

    EvoIndividual first = Random::randomChoice(generation, random_engine);
    EvoIndividual second = Random::randomChoice(generation, random_engine);

    if (second.fitness < first.fitness) std::swap(first, second);

    for (int i = 0; i < 2; i++) {
        EvoIndividual entity = Random::randomChoice(generation, random_engine);
        if (entity.fitness < first.fitness) {
            std::swap(first, entity);
            second = std::move(entity);
        }
        else if (entity.fitness < second.fitness) {
            second = std::move(entity);
        }
    }
    return std::array{ std::move(first), std::move(second) };
};

/**
 * @brief Performs tournament selection on a subset of a population.
 *
 * This function performs tournament selection, a method of selecting individuals from a population based on their fitness.
 * It randomly selects two individuals from a specified range within the population and chooses the one with the lower fitness.
 * This process is repeated twice to select two individuals.
 *
 * If the start and end indices of the range are equal, an error message is printed to `std::cerr`.
 *
 * @param population A reference to the `EvoPopulation` from which individuals are selected.
 * @param random_engine A reference to a `XoshiroCpp::Xoshiro256Plus` random number generator.
 * @param start_index The start index of the range within the population from which individuals are selected.
 * @param end_index The end index of the range within the population from which individuals are selected.
 *
 * @return std::array<EvoIndividual, 2> The two selected individuals with the lower fitness.
 */
std::array<EvoIndividual, 2> Selection::tournament_selection(EvoPopulation const& population, XoshiroCpp::Xoshiro256Plus& random_engine, size_t start_index, size_t end_index) {
    if (start_index == end_index) {
        std::cerr << "Tournament selection: start index and end index should not be equal." << std::endl;
    }
    std::array<EvoIndividual, 2> parents, couple;
    for (int i = 0; i < 2; i++) {
        couple = population.get_random_couple_individuals(random_engine, start_index, end_index);
        parents[i] = (couple[0].fitness < couple[1].fitness) ? couple[0] : couple[1];
    }
    return parents;
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
EvoIndividual Factory::getRandomEvoIndividual(int predictor_row_count, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine) {
    EvoIndividual individual{};
    // create genofond
    for (int i = 0; i < predictor_column_count; i++)
    {
        individual.merger_chromosome.push_back(Factory::getRandomMergeAllele(i, predictor_column_count, random_engine));
        individual.x_transformer_chromosome.push_back(Factory::getRandomTransformXAllele(i, random_engine));
    }
    individual.robuster_chromosome.push_back(Factory::getRandomRobustAllele(predictor_row_count, random_engine));
    individual.y_transformer_chromosome.push_back(Factory::getRandomTransformYAllele(random_engine));
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
    int size,
    Transform::EvoDataSet const& dataset,
    XoshiroCpp::Xoshiro256Plus& random_engine,
    std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver
) {
    std::vector<EvoIndividual> generation(size);
    std::generate(generation.begin(), generation.end(), [&]() {return Factory::getRandomEvoIndividual(dataset.predictor.rows(), dataset.predictor.cols(), random_engine);});
    std::for_each(generation.begin(), generation.end(),
        [&](EvoIndividual& individual) {
            individual.evaluate(EvoMath::get_fitness(Transform::data_transformation_robust(dataset.predictor, dataset.target, individual), solver));
        }
    );
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
MergeAllele Factory::getRandomMergeAllele(int column_index, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine) {
    MergeAllele merge_allele{ column_index };
    // column index 0 marks x0 (no merging)
    if (column_index != 0) {
        std::vector<int> free_cols(predictor_column_count);
        std::iota(begin(free_cols), end(free_cols), 0);
        free_cols.erase(std::find(free_cols.begin(), free_cols.end(), column_index));
        std::shuffle(free_cols.begin(), free_cols.end(), random_engine);
        int number_of_cols_to_merge = RandomNumbers::rand_interval_int(0, predictor_column_count - 1, random_engine);
        for (int i = 0; i < number_of_cols_to_merge; i++) {
            merge_allele.allele.emplace_back(free_cols.back(), Merge_operator(RandomNumbers::rand_interval_int(0, merge_operator_maxindex, random_engine)));
            free_cols.pop_back();
        }
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
TransformXAllele Factory::getRandomTransformXAllele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine) {
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
TransformYAllele Factory::getRandomTransformYAllele(XoshiroCpp::Xoshiro256Plus& random_engine) {
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
RobustAllele Factory::getRandomRobustAllele(int row_count, XoshiroCpp::Xoshiro256Plus& random_engine) {
    RobustAllele robust_allele{};
    int rows_to_erase = RandomNumbers::rand_interval_int(0, row_count / 4., random_engine);
    std::vector<int> choosen_rows(row_count);
    std::iota(begin(choosen_rows), end(choosen_rows), 0);
    std::shuffle(choosen_rows.begin(), choosen_rows.end(), random_engine);
    choosen_rows.erase(choosen_rows.end() - rows_to_erase, choosen_rows.end());
    std::sort(choosen_rows.begin(), choosen_rows.end());
    robust_allele.allele = choosen_rows;
    return robust_allele;
};

/**
 * @brief Performs crossover operation on two parent EvoIndividuals to generate a new EvoIndividual.
 *
 * @param parents An array of two EvoIndividual objects that are the parents.
 * @param chromosome_size The size of the chromosomes in the EvoIndividual objects.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return An EvoIndividual object that is the result of the crossover operation.
 *
 * This function performs a crossover operation, a method of generating new individuals in a genetic algorithm.
 * It works by selecting a random crossover point in the chromosomes of the parents, and then creating a new individual
 * that inherits the genes from the first parent up to the crossover point, and the genes from the second parent after the crossover point.
 */
EvoIndividual Crossover::cross(std::array<EvoIndividual, 2> && parents, int chromosome_size, XoshiroCpp::Xoshiro256Plus& random_engine) {
    EvoIndividual youngling{};
    // indexes which points to place of chromosome cut & recombination
    int m_crossover_twist_index = RandomNumbers::rand_interval_int(0, chromosome_size, random_engine);
    int t_crossover_twist_index = RandomNumbers::rand_interval_int(0, chromosome_size, random_engine);
    int r_crossover_twist_index = RandomNumbers::rand_interval_int(0, 1, random_engine);
    int y_crossover_twist_index = RandomNumbers::rand_interval_int(0, 1, random_engine);
    // cross single gene chromosomes robuster & ytransformer
    youngling.robuster_chromosome = (r_crossover_twist_index == 0) ? parents[0].robuster_chromosome : parents[1].robuster_chromosome;
    youngling.y_transformer_chromosome = (y_crossover_twist_index == 0) ? parents[0].y_transformer_chromosome : parents[1].y_transformer_chromosome;
    // cross multi gene chromosomes merger & xtransformer
    youngling.merger_chromosome = parents[0].merger_chromosome;
    youngling.x_transformer_chromosome = parents[0].x_transformer_chromosome;
    std::copy(parents[1].merger_chromosome.begin() + m_crossover_twist_index, parents[1].merger_chromosome.end(), youngling.merger_chromosome.begin() + m_crossover_twist_index);
    std::copy(parents[1].x_transformer_chromosome.begin() + t_crossover_twist_index, parents[1].x_transformer_chromosome.end(), youngling.x_transformer_chromosome.begin() + t_crossover_twist_index);
    return youngling;
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
void Mutation::mutate(EvoIndividual& individual, int chromosome_size, int predictor_row_count, int mutation_rate, XoshiroCpp::Xoshiro256Plus& random_engine) {
    int rand_num = RandomNumbers::rand_interval_int(0, 100, random_engine);
    if (rand_num <= mutation_rate) {
        int mutation_index = RandomNumbers::rand_interval_int(0, 3, random_engine);
        if (mutation_index == 0) {
            int col = RandomNumbers::rand_interval_int(0, chromosome_size - 1, random_engine);
            individual.x_transformer_chromosome.at(col) = Factory::getRandomTransformXAllele(col, random_engine);
        }
        if (mutation_index == 1) {
            int col = RandomNumbers::rand_interval_int(0, chromosome_size - 1, random_engine);
            individual.merger_chromosome.at(col) = Factory::getRandomMergeAllele(col, chromosome_size, random_engine);
        }
        if (mutation_index == 2) {
            individual.robuster_chromosome.at(0) = Factory::getRandomRobustAllele(predictor_row_count, random_engine);
        }
        if (mutation_index == 3) {
            individual.y_transformer_chromosome.at(0) = Factory::getRandomTransformYAllele(random_engine);
        }
    }
}

/**
 * @brief Performs reproduction operation on two parent EvoIndividuals to generate a new EvoIndividual.
 *
 * @param parents An array of two EvoIndividual objects that are the parents.
 * @param chromosomes_size The size of the chromosomes in the EvoIndividual objects.
 * @param predictor_row_count The number of rows in the predictor matrix.
 * @param random_engine A random engine to be used for generating random numbers.
 *
 * @return An EvoIndividual object that is the result of the reproduction operation.
 *
 * This function performs a reproduction operation, a method of generating new individuals in a genetic algorithm.
 * It works by first performing a crossover operation on the parents to generate a new individual,
 * and then performing a mutation operation on the new individual.
 */
EvoIndividual Reproduction::reproduction(
    std::array<EvoIndividual, 2> && parents,
    int chromosomes_size,
    int predictor_row_count,
    int mutation_rate,
    XoshiroCpp::Xoshiro256Plus& random_engine
) {
    // crossover
    EvoIndividual individual = Crossover::cross(std::move(parents), chromosomes_size, random_engine);
    // mutation
    Mutation::mutate(individual, chromosomes_size, predictor_row_count, mutation_rate, random_engine);
    return individual;
};

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

    // erase some rows
    individual.robuster_chromosome.at(0).transform(matrix);

    //merge predictors
    for (int i = 0; i < matrix.cols(); i++)
    {
        individual.merger_chromosome.at(i).transform(matrix);
    }

    //transform predictors
    for (int i = 0; i < matrix.cols(); i++)
    {
        individual.x_transformer_chromosome.at(i).transform(matrix);
    }
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
Eigen::MatrixXd Transform::half_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual const& individual) {
    //merge predictors
    for (int i = 0; i < matrix.cols(); i++)
    {
        individual.merger_chromosome.at(i).transform(matrix);
    }
    //transform predictors
    for (int i = 0; i < matrix.cols(); i++)
    {
        individual.x_transformer_chromosome.at(i).transform(matrix);
    }
    return matrix;
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
Eigen::MatrixXd Transform::robust_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual const& individual) {
    individual.robuster_chromosome.at(0).transform(matrix);
    return matrix;
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
    individual.robuster_chromosome.at(0).transformVector(vector);
    individual.y_transformer_chromosome.at(0).transformVector(vector);
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
Eigen::VectorXd Transform::half_target_transform(Eigen::VectorXd& vector, EvoIndividual const& individual) {
    individual.y_transformer_chromosome.at(0).transformVector(vector);
    return vector;
};

/**
 * The function applies transformations to the predictor and target data based on the provided individual's characteristics.
 * It first transforms the predictor data, then the target data using the Transform::full_predictor_transform and Transform::full_target_transform functions respectively.
 *
 * @param predictor The predictor is an Eigen::MatrixXd object representing the predictor data to be transformed.
 * @param target The target is an Eigen::VectorXd object representing the target data to be transformed.
 * @param individual The individual is a constant reference to an EvoIndividual object, whose characteristics are used to transform the data.
 *
 * @return a Transform::EvoDataSet object containing the transformed predictor and target data.
 */
Transform::EvoDataSet Transform::data_transformation_robust(Eigen::MatrixXd predictor, Eigen::VectorXd target, EvoIndividual const& individual) {
    Transform::full_predictor_transform(predictor, individual);
    Transform::full_target_transform(target, individual);
    return { predictor, target };
};

/**
 * The function applies non-robust transformations to the predictor and target data based on the provided individual's characteristics.
 * It first transforms the predictor data, then the target data using the Transform::half_predictor_transform and Transform::half_target_transform functions respectively.
 *
 * @param predictor The predictor is an Eigen::MatrixXd object representing the predictor data to be transformed.
 * @param target The target is an Eigen::VectorXd object representing the target data to be transformed.
 * @param individual The individual is a constant reference to an EvoIndividual object, whose characteristics are used to transform the data.
 *
 * @return a Transform::EvoDataSet object containing the transformed predictor and target data.
 */
Transform::EvoDataSet Transform::data_transformation_nonrobust(Eigen::MatrixXd predictor, Eigen::VectorXd target, EvoIndividual const& individual) {
    Transform::half_predictor_transform(predictor, individual);
    Transform::half_target_transform(target, individual);
    return { predictor, target };
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
double EvoMath::get_fitness(Transform::EvoDataSet const& dataset, T solver) {
    return solver(dataset.predictor, dataset.target);
}

// Explicit instantiation
template double EvoMath::get_fitness(Transform::EvoDataSet const& dataset, std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)> solver);

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
