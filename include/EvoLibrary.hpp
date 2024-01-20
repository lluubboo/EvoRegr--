#pragma once
#include <Eigen/Dense>
#include <vector>
#include <span>
#include <functional>
#include "EvoIndividual.hpp"
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"
#include "RegressionSolver.hpp"
#include "EvoDataSet.hpp"

namespace Selection {

    EvoIndividual const& tournament_selection(std::vector<EvoIndividual>::iterator begin, size_t size, XoshiroCpp::Xoshiro256Plus& random_engine);

}
namespace Crossover {

    void cross(EvoIndividual& child, EvoIndividual const& parent1, EvoIndividual const& parent2, int, XoshiroCpp::Xoshiro256Plus&);

}
namespace Mutation {

    void mutate(EvoIndividual&, int, int, int, int, XoshiroCpp::Xoshiro256Plus&);

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

    EvoRegression::EvoDataSet& transform_dataset(EvoRegression::EvoDataSet& dataset, EvoIndividual const& individual, bool robust = false);
    EvoRegression::EvoDataSet transform_dataset_copy(EvoRegression::EvoDataSet dataset, EvoIndividual const& individual, bool robust = false);
}

namespace EvoMath {

    template <typename T>
    double get_fitness(EvoRegression::EvoDataSet const& dataset, T solver);

    template <typename T>
    std::vector<T> extract_column(std::vector<T> data, unsigned int column_count, unsigned int column_index);
}

namespace Factory {

    EvoIndividual get_random_evo_individual(EvoRegression::EvoDataSet const& dataset, XoshiroCpp::Xoshiro256Plus& random_engine);
    MergeAllele get_random_merge_allele(int column_index, int predictor_column_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformXAllele get_random_transform_xallele(int column_index, XoshiroCpp::Xoshiro256Plus& random_engine);
    TransformYAllele get_random_transform_yallele(XoshiroCpp::Xoshiro256Plus& random_engine);
    RobustAllele get_random_robust_allele(int row_count, XoshiroCpp::Xoshiro256Plus& random_engine);
    std::vector<EvoIndividual> generate_random_generation(int, EvoRegression::EvoDataSet, XoshiroCpp::Xoshiro256Plus&, std::function<double(EvoRegression::EvoDataSet const&)>);

}

namespace Random {

    std::vector<XoshiroCpp::Xoshiro256Plus> create_random_engines(size_t count);

}
