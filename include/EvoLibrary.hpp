#pragma once
#include <Eigen/Dense>
#include <vector>
#include "EvoIndividual.hpp"
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"

std::vector<EvoIndividual> sort_by_fitness_desc(std::vector<EvoIndividual>&);

namespace Factory {
    EvoIndividual getRandomEvoIndividual(Eigen::MatrixXd predictor, Eigen::VectorXd predicate, XoshiroCpp::Xoshiro256Plus& random_engine);
    MergeAllele getRandomMergeAllele(int column_index, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformXAllele getRandomTransformXAllele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformYAllele getRandomTransformYAllele(XoshiroCpp::Xoshiro256Plus& random_engine);
    RobustAllele getRandomRobustAllele(int row_count, XoshiroCpp::Xoshiro256Plus& random_engine);
}

namespace Selection {

    std::array<EvoIndividual, 2> tournament_selection(std::vector<EvoIndividual> const&, XoshiroCpp::Xoshiro256Plus&);

}

namespace Crossover {

    EvoIndividual cross(std::array<EvoIndividual, 2> const& parents, int, XoshiroCpp::Xoshiro256Plus&);

}

namespace Mutation {

    EvoIndividual mutate(EvoIndividual&, int, int, XoshiroCpp::Xoshiro256Plus&);

}

namespace Transform {

    Eigen::MatrixXd full_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    Eigen::MatrixXd half_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    Eigen::MatrixXd robust_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    Eigen::VectorXd full_target_transform(Eigen::VectorXd&, EvoIndividual const&);
    Eigen::VectorXd half_target_transform(Eigen::VectorXd&, EvoIndividual const&);
}

namespace Reproduction {
    EvoIndividual reproduction(std::array<EvoIndividual, 2> const& parents, int chromosome_size, int predictor_row_count, XoshiroCpp::Xoshiro256Plus&);
}

namespace FitnessEvaluator {

    double get_fitness(Eigen::MatrixXd const& predictor, Eigen::VectorXd const& target);
    void sort_vector_by_fitness();

}

