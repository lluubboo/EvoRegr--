#include <Eigen/Dense>
#include <string>
#include <vector>
#include "EvoIndividual.hpp"
#include "Plotter.hpp"
#include "EvoDataSet.hpp"

namespace EvoRegression {

    std::string get_regression_testing_table(double* prediction_dataset_nonrobust, size_t size);

    std::string get_regression_training_table(double* prediction_dataset_robust, size_t size);

    std::string get_regression_coefficients_table(double* regression_coefficients, size_t size);

    std::string get_genotype_table(EvoIndividual titan);

    std::string get_formula_table(std::vector<std::string> formula);

    std::string get_result_metrics_table(std::vector<double> regression_metrics);

    Eigen::MatrixXd get_regression_summary_matrix(EvoIndividual const& titan, Eigen::VectorXd const& regression_coefficcients, EvoRegression::EvoDataSet const& original_dataset);

}