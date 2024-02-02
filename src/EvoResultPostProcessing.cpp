#include "EvoResultPostProcessing.hpp"

/**
 * @brief Returns the regression result table as a string.
 *
 * This function generates a regression result table using the provided data and returns it as a string.
 * The table includes columns for the target, prediction, difference, and percentage difference.
 *
 * @return The regression result table as a string.
 */
std::string EvoRegression::get_regression_testing_table(double* prediction_dataset, size_t size) {
    Plotter<double> plt = Plotter(
        prediction_dataset,
        "Regression result test part",
        { "Target", "Prediction", "Residual", "Residual [%]" },
        149,
        size,
        DataArrangement::ColumnMajor
    );
    return plt.get_table();
};

/**
 * @brief Returns the regression result table as a string.
 *
 * This function generates a regression result table using the provided data and returns it as a string.
 * The table includes columns for the target, prediction, difference, and percentage difference.
 *
 * @return The regression result table as a string.
 */
std::string EvoRegression::get_regression_training_table(double* prediction_dataset_robust, size_t size) {
    Plotter<double> plt = Plotter(
        prediction_dataset_robust,
        "Regression result learning part",
        { "Target", "Prediction", "Residual", "Residual [%]" },
        149,
        size,
        DataArrangement::ColumnMajor
    );
    return plt.get_table();
};

/**
 * @brief Retrieves the regression coefficients table.
 *
 * This function creates a Plotter object to visualize the regression coefficients
 * and returns the table representation of the plot.
 *
 * @return The regression coefficients table.
 */
std::string EvoRegression::get_regression_coefficients_table(double* regression_coefficients, size_t size) {
    Plotter<double> plt = Plotter(
        regression_coefficients,
        "Regression coefficients",
        { "Coefficients" },
        149,
        size,
        DataArrangement::ColumnMajor
    );
    return plt.get_table();
};

/**
 * @brief Returns the genotype table as a string.
 *
 * This function generates a genotype table using the Plotter class and returns it as a string.
 * The genotype table includes information about merge chromosome, transform predictor chromosome,
 * transform target chromosome, and robust chromosome.
 *
 * @return The genotype table as a string.
 */
std::string EvoRegression::get_genotype_table(EvoIndividual titan) {
    std::stringstream genotype_table;

    std::vector<std::string> merge_chromosome = titan.merge_chromosome_to_string_vector();
    Plotter<std::string> plt = Plotter(
        merge_chromosome.data(),
        "Merge chromosome",
        { "Alleles" },
        149,
        merge_chromosome.size(),
        DataArrangement::RowMajor
    );
    genotype_table << plt.get_table();

    std::vector<std::string> transform_predictor_chromosome = titan.transform_predictor_chromosome_to_string_vector();
    plt = Plotter(
        transform_predictor_chromosome.data(),
        "Transform predictor chromosome",
        { "Alleles" },
        149,
        transform_predictor_chromosome.size(),
        DataArrangement::RowMajor
    );
    genotype_table << plt.get_table();

    std::vector<std::string> transform_target_chromosome = titan.transform_target_chromosome_to_string_vector();
    plt = Plotter(
        transform_target_chromosome.data(),
        "Transform target chromosome",
        { "Alleles" },
        149,
        transform_target_chromosome.size(),
        DataArrangement::RowMajor
    );
    genotype_table << plt.get_table();

    std::vector<std::string> robust_chromosome = titan.robust_tr_chromosome_to_string_vector();
    plt = Plotter(
        robust_chromosome.data(),
        "Robust chromosome",
        { "Alleles" },
        149,
        robust_chromosome.size(),
        DataArrangement::RowMajor
    );

    genotype_table << plt.get_table();
    return genotype_table.str();
}

/**
 * @brief Retrieves the formula table.
 *
 * This function returns a string representation of the formula table.
 *
 * @return The formula table as a string.
 */
std::string EvoRegression::get_formula_table(std::vector<std::string> formula) {
    Plotter<std::string> plt = Plotter(
        formula.data(),
        "Evo-regression formula",
        { "Formula" },
        149,
        formula.size(),
        DataArrangement::RowMajor
    );
    return plt.get_table();
};

/**
 * @brief Generates a table of regression result metrics.
 *
 * This function calculates the median of the percentage error, the standard deviation, and the coefficient of determination (R2) of the regression results.
 * These metrics are then displayed in a table.
 *
 * @return A string representing a table of the regression result metrics.
 */
std::string EvoRegression::get_result_metrics_table(std::vector<double> regression_metrics) {
    Plotter<double> plt = Plotter(
        regression_metrics.data(),
        "Regression result metrics [robust]",
        { "Error median", "Error standard deviation", "COD (R2)" },
        149,
        regression_metrics.size(),
        DataArrangement::RowMajor
    );
    return plt.get_table();
};

Eigen::MatrixXd EvoRegression::get_regression_summary_matrix(
    const EvoIndividual& titan,
    const Eigen::VectorXd& regression_coefficients,
    const EvoRegression::EvoDataSet& original
) {
    // get target and predicted target for comparison
    Eigen::VectorXd target = original.target;
    Eigen::VectorXd prediction = original.predictor * regression_coefficients;

    // transform target back to original values for presentation
    auto& transformer = titan.y_transformer_chromosome.at(0);
    transformer.transformBack(target);
    transformer.transformBack(prediction);

    // assembly result matrix
    const int numColumns = 4;
    Eigen::MatrixXd regression_result_matrix(original.predictor.rows(), numColumns);
    regression_result_matrix.col(0) = target;
    regression_result_matrix.col(1) = prediction;
    regression_result_matrix.col(2) = target - prediction;
    regression_result_matrix.col(3) = (regression_result_matrix.col(2).array() / regression_result_matrix.col(0).array()) * 100;

    return regression_result_matrix;
}