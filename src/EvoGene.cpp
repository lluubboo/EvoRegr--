#include <numeric>
#include <iostream>
#include <cmath>
#include <sstream>
#include "RandomChoices.hpp"
#include "RandomNumberGenerator.hpp"
#include "EvoGene.hpp"

/*****************************************EvoGene**********************************************/

/**
 * The above code defines a default constructor for the EvoGene class in C++.
 */
EvoGene::EvoGene() {}

/**
 * @brief Constructs an instance of the EvoGene class.
 *
 * @param index The index of the gene.
 */
EvoGene::EvoGene(int index) : column_index{ index }, characteristic_number{ 0 } {}

/**
 * @brief Destructor for the EvoGene class.
 */
EvoGene::~EvoGene() {}

/*****************************************MergeAllele******************************************/

/**
 * @brief Constructs a MergeAllele object with the specified index.
 *
 * @param index The index of the MergeAllele object.
 */
MergeAllele::MergeAllele(int index) : EvoGene(index), allele() {}

/**
 * @brief Destructor for the MergeAllele class.
 *
 * This destructor is responsible for cleaning up any resources
 * allocated by the MergeAllele class.
 */
MergeAllele::~MergeAllele() {}

/**
 * Transforms the given matrix according to the allele.
 *
 * @param matrix The matrix to be transformed.
 */
void MergeAllele::transform(Eigen::MatrixXd& matrix) const {
    if (!allele.empty()) matrix.col(column_index) = allele.evaluate_expression(matrix);
}

/**
 * Converts the MergeAllele object to a string representation.
 * If the allele is empty, it returns a string in the format "(colX)" where X is the column index.
 * If the allele is not empty, it returns a string in the format "(...)(colX)(...)(colY)...",
 * where each "..." represents the merge operator symbol and merge column index.
 * @return The string representation of the MergeAllele object.
 */
std::string MergeAllele::to_string() const {
    return "(" + allele.get_expression() + ")";
}

/**
 * Converts the MergeAllele object to a string representation.
 * The string representation consists of the prefix "MA" followed by the column index,
 * followed by the merge operator names and merge column numbers of each twin in the allele,
 * and ending with the suffix "E".
 *
 * @return The string representation of the MergeAllele object.
 */
std::string MergeAllele::to_string_code() const {
    return allele.get_expression();
}

/*****************************************TransformXAllele*************************************/

/**
 * @brief Constructs a new TransformXAllele object with the given index.
 *
 * @param index The index of the allele.
 */
TransformXAllele::TransformXAllele(int index) : EvoGene(index), allele{} {}

/**
 * @brief Destructor for the TransformXAllele class.
 *
 * This destructor is responsible for cleaning up any resources
 * allocated by the TransformXAllele class.
 */
TransformXAllele::~TransformXAllele() {}

/**
 * Applies a transformation operation to a specific column of a matrix.
 * The type of transformation is determined by the value of the 'allele' parameter.
 *
 * @param matrix The matrix to be transformed.
 */
void TransformXAllele::transform(Eigen::MatrixXd& matrix) const {
    switch (allele) {
    case Transform_operator::Sqr:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return s * s; });
        break;
    case Transform_operator::Cub:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return s * s * s; });
        break;
    case Transform_operator::Pow:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)pow(s, characteristic_number); });
        break;
    case Transform_operator::Wek:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)pow(s, 1 / characteristic_number); });
        break;
    case Transform_operator::Sqt:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)sqrt(s); });
        break;
    case Transform_operator::Csqt:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)cbrt(s); });
        break;
    case Transform_operator::Nul:
        matrix.col(column_index) = Eigen::VectorXd::Zero(matrix.rows());
        break;
    case Transform_operator::Let:;
        break;
    case Transform_operator::Log:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)log(s); });
        break;
    case Transform_operator::Log10:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)log10(s); });
        break;
    case Transform_operator::Log2:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)log2(s); });
        break;
    }
}

/**
 * Converts the TransformXAllele object to a string representation.
 *
 * @return The string representation of the TransformXAllele object.
 */
std::string TransformXAllele::to_string() const {
    std::string sallele;
    sallele = transform_operator_symbols.at(allele) + ((characteristic_number == 0) ? "" : std::to_string(characteristic_number));
    return sallele;
}

/**
 * @brief Converts the TransformXAllele object to a string representation.
 *
 * @return The string representation of the TransformXAllele object.
 */
std::string TransformXAllele::to_string_code() const {
    std::string sallele("TX");
    sallele += std::to_string(column_index);
    sallele += std::to_string(characteristic_number);
    sallele += transform_operator_names.at(allele);
    return sallele;
}

/**
 * @brief Resets the characteristic number of the TransformXAllele object.
 *
 * This function sets the characteristic number of the TransformXAllele object to the specified value.
 *
 * @param number The new value for the characteristic number.
 */
void TransformXAllele::resetCharacteristicNumber(float number) {
    characteristic_number = number;
}

/*****************************************TransformYAllele*************************************/

/**
 * @brief Constructs a new TransformYAllele object.
 *
 * This constructor initializes the TransformYAllele object with a default value of 0 for the EvoGene base class and an empty allele.
 */
TransformYAllele::TransformYAllele() : EvoGene(0), allele{} {}

/**
 * @brief Destructor for the TransformYAllele class.
 */
TransformYAllele::~TransformYAllele() {};

/**
 * @brief Resets the characteristic number of the TransformYAllele object.
 *
 * This function sets the characteristic number of the TransformYAllele object to the specified value.
 *
 * @param number The new value for the characteristic number.
 */
void TransformYAllele::resetCharacteristicNumber(float number) {
    characteristic_number = number;
}

/**
 * Transforms the given matrix.
 *
 * @param matrix The matrix to be transformed.
 */
void TransformYAllele::transform(Eigen::MatrixXd& /*matrix*/) const {}

/**
 * Transforms the elements of the given vector based on the selected allele.
 *
 * @param vector The vector to be transformed.
 */
void TransformYAllele::transform_vector(Eigen::VectorXd& vector) const {
    switch (allele) {
    case Transform_operator::Sqr:
        vector = vector.unaryExpr([&](const auto s) { return s * s; });
        break;
    case Transform_operator::Cub:
        vector = vector.unaryExpr([&](const auto s) { return s * s * s; });
        break;
    case Transform_operator::Pow:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, characteristic_number); });
        break;
    case Transform_operator::Wek:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, 1 / characteristic_number); });
        break;
    case Transform_operator::Sqt:
        vector = vector.unaryExpr([&](const auto s) { return (double)sqrt(s); });
        break;
    case Transform_operator::Csqt:
        vector = vector.unaryExpr([&](const auto s) { return (double)cbrt(s); });
        break;
    case Transform_operator::Let:
        break;
    case Transform_operator::Log:
        vector = vector.unaryExpr([&](const auto s) { return (double)log(s); });
        break;
    case Transform_operator::Log10:
        vector = vector.unaryExpr([&](const auto s) { return (double)log10(s); });
        break;
    case Transform_operator::Log2:
        vector = vector.unaryExpr([&](const auto s) { return (double)log2(s); });
        break;
    case Transform_operator::Nul: //it doesnt happpen
        vector = Eigen::VectorXd::Zero(vector.rows());
        break;
    }
}

/**
 * Transforms the given vector back based on the selected allele.
 *
 * @param vector The vector to be transformed.
 */
void TransformYAllele::transformBack(Eigen::VectorXd& vector) const {
    switch (allele) {
    case Transform_operator::Sqr:
        vector = vector.unaryExpr([&](const auto s) { return (double)sqrt(s); });
        break;
    case Transform_operator::Cub:
        vector = vector.unaryExpr([&](const auto s) { return (double)cbrt(s); });
        break;
    case Transform_operator::Pow:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, (1 / characteristic_number)); });
        break;
    case Transform_operator::Wek:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, (characteristic_number)); });
        break;
    case Transform_operator::Sqt:
        vector = vector.unaryExpr([&](const auto s) { return s * s; });
        break;
    case Transform_operator::Csqt:
        vector = vector.unaryExpr([&](const auto s) { return s * s * s; });
        break;
    case Transform_operator::Let:
        break;
    case Transform_operator::Log:
        vector = vector.unaryExpr([&](const auto s) { return (double)exp(s); });
        break;
    case Transform_operator::Log10:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(10, s); });
        break;
    case Transform_operator::Log2:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(2, s); });
        break;
    case Transform_operator::Nul: //it doesnt happpen
        vector = Eigen::VectorXd::Zero(vector.rows());
        break;
    }
}

/**
 * Converts the TransformYAllele object to a string representation.
 *
 * @return The string representation of the TransformYAllele object.
 */
std::string TransformYAllele::to_string() const {
    std::string sallele;
    sallele = transform_operator_symbols.at(allele) + ((characteristic_number == 0) ? "" : std::to_string(characteristic_number));
    return sallele;
}

/**
 * Converts the TransformYAllele object to a string representation.
 *
 * @return The string representation of the TransformYAllele object.
 */
std::string TransformYAllele::to_string_code() const {
    std::string sallele("TY");
    sallele += std::to_string(characteristic_number);
    sallele += transform_operator_names.at(allele);
    return sallele;
}

/*****************************************RobustAllele*****************************************/

/**
 * @brief Constructs a RobustAllele object.
 *
 * This constructor initializes a RobustAllele object with a default value of 0 for the EvoGene base class,
 * and an empty allele array.
 */
RobustAllele::RobustAllele() : EvoGene(0), allele{} {};

/**
 * @brief Destructor for the RobustAllele class.
 */
RobustAllele::~RobustAllele() {};

/**
 * Transforms the given matrix using the allele values.
 *
 * @param matrix The matrix to be transformed.
 */
void RobustAllele::transform(Eigen::MatrixXd& matrix) const {
    Eigen::MatrixXd transformedMatrix = matrix(Eigen::Map<const Eigen::VectorXi>(allele.data(), allele.size()), Eigen::all);
    matrix.swap(transformedMatrix);
}

/**
 * Transforms the given vector using the allele values.
 * The allele values are used as indices to rearrange the elements of the vector.
 * The transformed vector replaces the original vector.
 *
 * @param vector The vector to be transformed.
 */
void RobustAllele::transform_vector(Eigen::VectorXd& vector) const {
    Eigen::VectorXd transformeVector = vector(Eigen::Map<const Eigen::VectorXi>(allele.data(), allele.size()));
    vector.swap(transformeVector);
}

/**
 * Converts the RobustAllele object to a string representation.
 *
 * @return The string representation of the RobustAllele object.
 */
std::string RobustAllele::to_string() const {
    std::string sallele;
    sallele = "(";
    for (auto const& index : allele) {
        sallele += std::to_string(index) + ((index == allele.back()) ? "" : ",");
    }
    sallele += ")";
    return sallele;
}

/**
 * Converts the RobustAllele object to a string representation.
 * The string representation includes the prefix "RA" followed by the characters in the allele.
 *
 * @return The string representation of the RobustAllele object.
 */
std::string RobustAllele::to_string_code() const {
    std::string result("RA");
    result.reserve(2 + allele.size()); // 2 for "RA"
    result.insert(result.end(), allele.begin(), allele.end());
    return result;
}



