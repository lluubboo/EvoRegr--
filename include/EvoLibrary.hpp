#pragma once
#include <Eigen/Dense>
#include <vector>
#include <span>
#include <functional>
#include "EvoPopulation.hpp"
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"
#include "RegressionSolver.hpp"

namespace Selection {

    std::array<EvoIndividual, 2> tournament_selection(std::vector<EvoIndividual> const&, XoshiroCpp::Xoshiro256Plus&);
    std::array<EvoIndividual, 2> tournament_selection(EvoPopulation const&, XoshiroCpp::Xoshiro256Plus&, size_t, size_t);

}
namespace Crossover {

    EvoIndividual cross(std::array<EvoIndividual, 2> && parents, int, XoshiroCpp::Xoshiro256Plus&);

}
namespace Mutation {

    void mutate(EvoIndividual&, int, int, int, XoshiroCpp::Xoshiro256Plus&);

}

namespace Transform {

    struct EvoDataSet {
        Eigen::MatrixXd predictor;
        Eigen::MatrixXd target;
    };

    void full_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    Eigen::MatrixXd half_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    Eigen::MatrixXd robust_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    void full_target_transform(Eigen::VectorXd&, EvoIndividual const&);
    Eigen::VectorXd half_target_transform(Eigen::VectorXd&, EvoIndividual const&);
    EvoDataSet data_transformation_robust(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual const&);
    EvoDataSet data_transformation_nonrobust(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual const&);

}

namespace Reproduction {
    EvoIndividual reproduction(std::array<EvoIndividual, 2> && parents, int chromosome_size, int predictor_row_count, int mutation_rate, XoshiroCpp::Xoshiro256Plus&);
}

namespace EvoMath {

    template <typename T>
    double get_fitness(Transform::EvoDataSet const& dataset, T solver);

    template <typename T>
    std::vector<T> extract_column(std::vector<T> data, unsigned int column_count, unsigned int column_index);
}

namespace Factory {
    EvoIndividual getRandomEvoIndividual(int row_count, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    MergeAllele getRandomMergeAllele(int column_index, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformXAllele getRandomTransformXAllele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformYAllele getRandomTransformYAllele(XoshiroCpp::Xoshiro256Plus& random_engine);
    RobustAllele getRandomRobustAllele(int row_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    std::vector<EvoIndividual> generate_random_generation(int, Transform::EvoDataSet const&, XoshiroCpp::Xoshiro256Plus&, std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)>);
}
