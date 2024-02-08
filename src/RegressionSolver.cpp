#include <Eigen/Dense>
#include "RegressionSolver.hpp"

RegressionDetailedResult solve_system_detailed(Eigen::MatrixXd const& predictor, Eigen::VectorXd const& target, float regularizaton_parameter) {
    RegressionDetailedResult result = RegressionDetailedResult();
    // Identity matrix
    Eigen::MatrixXd identity_matrix = Eigen::MatrixXd::Identity(predictor.cols(), predictor.cols());
    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = predictor.transpose();
    result.theta = (predictor_transposed * predictor + regularizaton_parameter * identity_matrix).colPivHouseholderQr().solve(predictor_transposed * target);
    return result;
}

double LLTSolver::operator()(EvoRegression::EvoDataSet& original_dataset, int test_set_size, float regularizaton_parameter) const {

    // Create blocks for training and testing
    int training_set_size = original_dataset.predictor.rows() - test_set_size;

    Transform::TemporarySplittedDataset dataset_view(
        original_dataset,
        test_set_size,
        training_set_size
    );

    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = dataset_view.train_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * dataset_view.train_predictor).llt().solve(predictor_transposed * dataset_view.train_target);

    // Check if the result is usable (no NaN or infinite values)
    if (!coefficients.hasNaN() && coefficients.allFinite()) {
        // Calculate fitness using test data
        return (dataset_view.test_target - (dataset_view.test_predictor * coefficients)).squaredNorm() / dataset_view.test_target.size();
    }
    else {
        // Set sum of squares errors to maximum value
        return std::numeric_limits<double>::max();
    }
}

double LDLTSolver::operator()(EvoRegression::EvoDataSet& dataset, int test_set_size, float regularizaton_parameter) const {

    // Calculate the number of training examples
    int num_train = static_cast<int>(dataset.predictor.rows() * 0.7);

    // Create blocks for training and testing
    auto train_predictor = dataset.predictor.block(0, 0, num_train, dataset.predictor.cols());
    auto train_target = dataset.target.segment(0, num_train);
    auto test_predictor = dataset.predictor.block(num_train, 0, dataset.predictor.rows() - num_train, dataset.predictor.cols());
    auto test_target = dataset.target.segment(num_train, dataset.target.size() - num_train);

    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = train_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * train_predictor).ldlt().solve(predictor_transposed * train_target);

    // Check if the result is usable (no NaN or infinite values)
    if (!coefficients.hasNaN() && coefficients.allFinite()) {
        // Calculate fitness using test data
        return (test_target - (test_predictor * coefficients)).squaredNorm() / test_target.size();
    }
    else {
        // Set sum of squares errors to maximum value
        return std::numeric_limits<double>::max();
    }
}

double ColPivHouseholderQrSolver::operator()(EvoRegression::EvoDataSet& dataset, int test_set_size, float regularizaton_parameter) const {

    // Calculate the number of training examples
    int num_train = static_cast<int>(dataset.predictor.rows() * 0.7);

    // Create blocks for training and testing
    auto train_predictor = dataset.predictor.block(0, 0, num_train, dataset.predictor.cols());
    auto train_target = dataset.target.segment(0, num_train);
    auto test_predictor = dataset.predictor.block(num_train, 0, dataset.predictor.rows() - num_train, dataset.predictor.cols());
    auto test_target = dataset.target.segment(num_train, dataset.target.size() - num_train);

    // Identity matrix
    Eigen::MatrixXd identity_matrix = Eigen::MatrixXd::Identity(train_predictor.cols(), train_predictor.cols());

    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = train_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * train_predictor + regularizaton_parameter * identity_matrix).colPivHouseholderQr().solve(predictor_transposed * train_target);

    // Check if the result is usable (no NaN or infinite values)
    if (!coefficients.hasNaN() && coefficients.allFinite()) {
        // Calculate fitness using test data MSE
        return (test_target - (test_predictor * coefficients)).squaredNorm() / test_target.size();
    }
    else {
        // Set sum of squares errors to maximum value
        return std::numeric_limits<double>::max();
    }
}

bool validate_coefficients(Eigen::VectorXd const& regression_coefficients) {
    return !regression_coefficients.hasNaN() && regression_coefficients.allFinite();
};

double calculate_fitness(Eigen::VectorXd const& regression_coefficients, Eigen::Block<Eigen::MatrixXd>& predictor_view, Eigen::VectorBlock<Eigen::VectorXd>& target_view) {
    if (validate_coefficients(regression_coefficients)) {

        // Calculate fitness using test data MSE - mean square error
        return (target_view - (predictor_view * regression_coefficients)).squaredNorm() / target_view.size();
    }
    else {

        // Set sum of squares errors to maximum value, calculation is thus invalid
        return std::numeric_limits<double>::max();
    }
}



