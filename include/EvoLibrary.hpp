#pragma once
#include <Eigen/Dense>
#include <vector>
#include <span>
#include <functional>
#include "EvoPopulation.hpp"
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"
#include "RegressionSolver.hpp"
#include "EvoDataSet.hpp"

namespace Selection {

    EvoIndividual const& tournament_selection(std::vector<EvoIndividual>::iterator begin, size_t size, XoshiroCpp::Xoshiro256Plus& random_engine);

}
namespace Crossover {

    EvoIndividual cross(EvoIndividual const&, EvoIndividual const&, int, XoshiroCpp::Xoshiro256Plus&);

}
namespace Mutation {

    void mutate(EvoIndividual&, int, int, int, XoshiroCpp::Xoshiro256Plus&);

}

namespace Migration {

    void short_distance_migration(std::vector<EvoIndividual>& population, size_t migration_size, std::vector<XoshiroCpp::Xoshiro256Plus>& random_engines);

}

namespace Transform {

    void full_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    void full_target_transform(Eigen::VectorXd&, EvoIndividual const&);

    void half_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    void robust_predictor_transform(Eigen::MatrixXd&, EvoIndividual const&);
    void half_target_transform(Eigen::VectorXd&, EvoIndividual const&);

    EvoRegression::EvoDataSet data_transformation_robust(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual const&);
    EvoRegression::EvoDataSet& data_transformation_robust(EvoRegression::EvoDataSet& dataset, EvoIndividual const& individual);

    EvoRegression::EvoDataSet data_transformation_nonrobust(Eigen::MatrixXd, Eigen::VectorXd, EvoIndividual const&);

    EvoIndividual reproduction(std::vector<EvoIndividual> const& old_population, XoshiroCpp::Xoshiro256Plus& random_engine);

}

namespace EvoMath {

    template <typename T>
    double get_fitness(EvoRegression::EvoDataSet const& dataset, T solver);

    template <typename T>
    std::vector<T> extract_column(std::vector<T> data, unsigned int column_count, unsigned int column_index);
}

namespace Factory {

    EvoIndividual getRandomEvoIndividual(int row_count, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    MergeAllele getRandomMergeAllele(int column_index, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformXAllele getRandomTransformXAllele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformYAllele getRandomTransformYAllele(XoshiroCpp::Xoshiro256Plus& random_engine);
    RobustAllele getRandomRobustAllele(int row_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    std::vector<EvoIndividual> generate_random_generation(int, EvoRegression::EvoDataSet const&, XoshiroCpp::Xoshiro256Plus&, std::function<double(Eigen::MatrixXd const&, Eigen::VectorXd const&)>);
    
}

namespace Random {

    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(size_t count);

}
