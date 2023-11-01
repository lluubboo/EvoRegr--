#include <Eigen/Dense>
#include "EvoLibrary.hpp"
#include "RandomChoices.hpp"
#include "RandomNumberGenerator.hpp"
#include "RegressionSolver.hpp"


std::vector<EvoIndividual> sort_by_fitness_desc(std::vector<EvoIndividual>& source) {
    std::stable_sort(source.begin(), source.end(), [](EvoIndividual const& lhs, EvoIndividual const& rhs) {
        return lhs.fitness < rhs.fitness;});
    return source;
};

EvoIndividual Factory::getRandomEvoIndividual(Eigen::MatrixXd predictor, Eigen::VectorXd target, XoshiroCpp::Xoshiro256Plus& random_engine) {

    EvoIndividual individual{};

    int predictor_count, predictor_entity_count;
    predictor_count = predictor.cols();
    predictor_entity_count = predictor.rows();

    // first create alleles

    for (int i = 0; i < predictor_count; i++)
    {
        individual.merger_chromosome.push_back(Factory::getRandomMergeAllele(i, predictor_count, random_engine));
        individual.x_transformer_chromosome.push_back(Factory::getRandomTransformXAllele(i, random_engine));
    }

    individual.robuster_chromosome.push_back(Factory::getRandomRobustAllele(predictor_entity_count ,random_engine));
    individual.y_transformer_chromosome.push_back(Factory::getRandomTransformYAllele(random_engine));

    // transform data
    Transform::full_predictor_transform(predictor, individual);
    Transform::full_target_transform(target, individual);

    //solve regression
    RegressionResult result;
    result = solveSystemByLLT(predictor, target);
    individual.fitness = result.mean_sum_residuals_squared;

    return individual;
}

MergeAllele Factory::getRandomMergeAllele(int column_index, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine) {
    
    MergeAllele merge_allele{column_index};
    std::vector<int> free_cols(predictor_column_count);
    std::iota(begin(free_cols), end(free_cols), 0);
    std::shuffle(free_cols.begin(), free_cols.end(), random_engine);

    int cols_to_merged = RandomNumbers::rand_interval_int(0, predictor_column_count - 1, random_engine);

    for (int i = 0; i < cols_to_merged; i++) {
        MergeTwin twin = MergeTwin();
        twin.merge_column = free_cols.back();
        twin.merge_operator = Merge_operator(RandomNumbers::rand_interval_int(0, merge_operator_maxindex, random_engine));
        merge_allele.allele.push_back(twin);
        free_cols.pop_back();
    }

    return merge_allele;
};

TransformXAllele Factory::getRandomTransformXAllele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine) {
    TransformXAllele transformx_allele{column_index};
    transformx_allele.allele = Transform_operator{ RandomNumbers::rand_interval_int(0, transform_operator_maxindex, random_engine) };
    if (transformx_allele.allele == Transform_operator::Pow) transformx_allele.resetCharacteristicNumber(RandomNumbers::rand_interval_int(2, 3, random_engine));
    return transformx_allele;
};

TransformYAllele Factory::getRandomTransformYAllele(XoshiroCpp::Xoshiro256Plus& random_engine) {
    TransformYAllele transformy_allele{};
    transformy_allele.allele = Transform_operator{ RandomNumbers::rand_interval_int(0, transform_y_operator_maxindex, random_engine) };
    if (transformy_allele.allele == Transform_operator::Pow) transformy_allele.resetCharacteristicNumber(RandomNumbers::rand_interval_int(2, 3, random_engine));
    return transformy_allele;
};

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

std::vector<EvoIndividual> Selection::tournament_selection(std::vector<EvoIndividual> const& generation, XoshiroCpp::Xoshiro256Plus& random_engine) {
    std::vector<EvoIndividual> sample = Random::randomChoices(generation, 4, random_engine);
    sort_by_fitness_desc(sample);
    return sample;
};

EvoIndividual Crossover::cross(EvoIndividual const& number_one, EvoIndividual const& number_two, int chromosome_size, XoshiroCpp::Xoshiro256Plus& random_engine) {

    EvoIndividual youngling{};

    // indexes which points to place of chromosome cut & recombination
    int m_crossover_twist_index = RandomNumbers::rand_interval_int(0, chromosome_size, random_engine);
    int t_crossover_twist_index = RandomNumbers::rand_interval_int(0, chromosome_size, random_engine);
    int r_crossover_twist_index = RandomNumbers::rand_interval_int(0, 1, random_engine);
    int y_crossover_twist_index = RandomNumbers::rand_interval_int(0, 1, random_engine);

    // cross single gene chromosomes robuster & ytransformer
    (r_crossover_twist_index == 0) ? youngling.robuster_chromosome = number_one.robuster_chromosome
        : youngling.robuster_chromosome = number_two.robuster_chromosome;
    (y_crossover_twist_index == 0) ? youngling.y_transformer_chromosome = number_one.y_transformer_chromosome
        : youngling.y_transformer_chromosome = number_two.y_transformer_chromosome;

    // cross multi gene chromosomes merger & xtransformer
    for (int i = 0; i < chromosome_size; i++) {
        (i < m_crossover_twist_index) ? youngling.merger_chromosome.push_back(number_one.merger_chromosome.at(i))
            : youngling.merger_chromosome.push_back(number_two.merger_chromosome.at(i));
        (i < t_crossover_twist_index) ? youngling.x_transformer_chromosome.push_back(number_one.x_transformer_chromosome.at(i))
            : youngling.x_transformer_chromosome.push_back(number_two.x_transformer_chromosome.at(i));
    }

    return youngling;
}

EvoIndividual Mutation::mutate(EvoIndividual& individual, int chromosome_size, int predictor_row_count, XoshiroCpp::Xoshiro256Plus& random_engine) {
    if (RandomNumbers::rand_interval_int(0, 7, random_engine) == 0) {
        if (RandomNumbers::rand_interval_int(0, 3, random_engine) == 0) {
            int col = RandomNumbers::rand_interval_int(0, chromosome_size - 1, random_engine);
            individual.x_transformer_chromosome.at(col) = Factory::getRandomTransformXAllele(col, random_engine);
        }
        if (RandomNumbers::rand_interval_int(0, 3, random_engine) == 0) {
            int col = RandomNumbers::rand_interval_int(0, chromosome_size - 1, random_engine);
            individual.merger_chromosome.at(col) = Factory::getRandomMergeAllele(col, chromosome_size ,random_engine);
        }
        if (RandomNumbers::rand_interval_int(0, 3, random_engine) == 0) {
            individual.robuster_chromosome.at(0) = Factory::getRandomRobustAllele(predictor_row_count, random_engine);
        }
        if (RandomNumbers::rand_interval_int(0, 3, random_engine) == 0) {
            individual.y_transformer_chromosome.at(0) = Factory::getRandomTransformYAllele(random_engine);
        }
    }
    return individual;
}

EvoIndividual Reproduction::reproduction(EvoIndividual const& parent1, EvoIndividual const& parent2, int chromosomes_size, int predictor_row_count, XoshiroCpp::Xoshiro256Plus& random_engine) {
    // crossover
    EvoIndividual individual = Crossover::cross(parent1, parent2, chromosomes_size, random_engine);
    // mutation
    Mutation::mutate(individual, chromosomes_size, predictor_row_count, random_engine);
    return individual;
};

Eigen::MatrixXd Transform::full_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual& individual) {

    //edit rows
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

    return matrix;
};

Eigen::MatrixXd Transform::half_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual& individual) {
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

Eigen::MatrixXd Transform::robust_predictor_transform(Eigen::MatrixXd& matrix, EvoIndividual& individual) {
    individual.robuster_chromosome.at(0).transform(matrix);
    return matrix;
};

Eigen::VectorXd Transform::full_target_transform(Eigen::VectorXd& vector, EvoIndividual& individual) {
    individual.robuster_chromosome.at(0).transformVector(vector);
    individual.y_transformer_chromosome.at(0).transformVector(vector);
    return vector;
};

double FitnessEvaluator::get_fitness(Eigen::MatrixXd const& predictor, Eigen::VectorXd const& target) {
    RegressionResult result = solveSystemByLLT(predictor, target);
    return result.mean_sum_residuals_squared;
};