#include <Eigen/Dense>
#include "RegressionSolver.hpp"

RegressionDetailedResult solve_system_detailed(EvoRegression::EvoDataSet& dataset, float regularization_parameter) {
    // Identity matrix
    Eigen::MatrixXd identity_matrix = Eigen::MatrixXd::Identity(dataset.predictor.cols(), dataset.predictor.cols());
    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = dataset.predictor.transpose();
    return { (predictor_transposed * dataset.predictor + regularization_parameter * identity_matrix).colPivHouseholderQr().solve(predictor_transposed * dataset.target) };
}

double LLTSolver::operator()(EvoRegression::EvoDataSet& original_dataset, EvoBoundaryConditions const& boundaries) const {

    Transform::TemporarySplittedDataset dataset_view(
        original_dataset,
        boundaries.test_set_size,
        boundaries.training_set_size
    );

    // Identity matrix
    Eigen::MatrixXd identity_matrix = Eigen::MatrixXd::Identity(original_dataset.predictor.cols(), original_dataset.predictor.cols());

    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = dataset_view.train_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * dataset_view.train_predictor + boundaries.regularization_parameter * identity_matrix).llt().solve(predictor_transposed * dataset_view.train_target);

    return calculate_fitness(coefficients, dataset_view.test_predictor, dataset_view.test_target);
}

double LDLTSolver::operator()(EvoRegression::EvoDataSet& original_dataset, EvoBoundaryConditions const& boundaries) const {

    Transform::TemporarySplittedDataset dataset_view(
        original_dataset,
        boundaries.test_set_size,
        boundaries.training_set_size
    );

    // Identity matrix
    Eigen::MatrixXd identity_matrix = Eigen::MatrixXd::Identity(original_dataset.predictor.cols(), original_dataset.predictor.cols());

    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = dataset_view.train_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * dataset_view.train_predictor + boundaries.regularization_parameter * identity_matrix).ldlt().solve(predictor_transposed * dataset_view.train_target);

    return calculate_fitness(coefficients, dataset_view.test_predictor, dataset_view.test_target);
}

double ColPivHouseholderQrSolver::operator()(EvoRegression::EvoDataSet& original_dataset, EvoBoundaryConditions const& boundaries) const {

    Transform::TemporarySplittedDataset dataset_view(
        original_dataset,
        boundaries.test_set_size,
        boundaries.training_set_size
    );

    // Identity matrix
    Eigen::MatrixXd identity_matrix = Eigen::MatrixXd::Identity(original_dataset.predictor.cols(), original_dataset.predictor.cols());

    // Calculate coefficients using training data
    Eigen::MatrixXd predictor_transposed = dataset_view.train_predictor.transpose();
    Eigen::VectorXd coefficients = (predictor_transposed * dataset_view.train_predictor + boundaries.regularization_parameter * identity_matrix).colPivHouseholderQr().solve(predictor_transposed * dataset_view.train_target);

    return calculate_fitness(coefficients, dataset_view.test_predictor, dataset_view.test_target);
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



