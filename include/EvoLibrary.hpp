#pragma once
#include <Eigen/Dense>
#include <vector>
#include "EvoIndividual.hpp"
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"

std::vector<EvoIndividual> sort_by_fitness_desc(std::vector<EvoIndividual>&);

namespace Factory {
    EvoIndividual getRandomEvoIndividual(Eigen::MatrixXf predictor, Eigen::VectorXf predicate, XoshiroCpp::Xoshiro256Plus& random_engine);
    MergeAllele getRandomMergeAllele(int column_index, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformXAllele getRandomTransformXAllele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformYAllele getRandomTransformYAllele(XoshiroCpp::Xoshiro256Plus& random_engine);
    RobustAllele getRandomRobustAllele(int row_count, XoshiroCpp::Xoshiro256Plus& random_engine);
}

namespace Selection {

    std::vector<EvoIndividual> tournament_selection(std::vector<EvoIndividual> const&, XoshiroCpp::Xoshiro256Plus&);

}

namespace Crossover {

    EvoIndividual cross(EvoIndividual const&, EvoIndividual const&, int, XoshiroCpp::Xoshiro256Plus&);

}

namespace Mutation {

    EvoIndividual mutate(EvoIndividual&, int, int, XoshiroCpp::Xoshiro256Plus&);

}

namespace Transform {

    Eigen::MatrixXf full_predictor_transform(Eigen::MatrixXf&, EvoIndividual&);
    Eigen::MatrixXf half_predictor_transform(Eigen::MatrixXf&, EvoIndividual&);
    Eigen::MatrixXf robust_predictor_transform(Eigen::MatrixXf&, EvoIndividual&);
    Eigen::VectorXf full_target_transform(Eigen::VectorXf&, EvoIndividual&);
    
}

namespace Reproduction {
    EvoIndividual reproduction(EvoIndividual const& parent1, EvoIndividual const& parent2, int chromosome_size, int predictor_row_count, XoshiroCpp::Xoshiro256Plus&);
}

namespace FitnessEvaluator {

    float get_fitness(Eigen::MatrixXf const& predictor, Eigen::VectorXf const& target);

}