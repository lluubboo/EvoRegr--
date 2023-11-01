#include <vector>
#include <stack>
#include <numeric>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include "RandomChoices.hpp"
#include "RandomNumberGenerator.hpp"
#include "EvoGene.hpp"
#include "XoshiroCpp.hpp"

using namespace std;


EvoGene::EvoGene() {}

EvoGene::EvoGene(int index) : column_index{ index }, characteristic_number{ 0 } {}

EvoGene::~EvoGene() {}




MergeAllele::MergeAllele(int index) : EvoGene(index), allele{} {}

MergeAllele::~MergeAllele() {}

Eigen::MatrixXd& MergeAllele::transform(Eigen::MatrixXd& matrix) {
    for (auto const& twin : allele) {
        modifyMatrixAccordingToTwin(twin, matrix);
    }
    return matrix;
}

Eigen::MatrixXd& MergeAllele::modifyMatrixAccordingToTwin(MergeTwin const& twin, Eigen::MatrixXd& matrix) {
    switch (twin.merge_operator) {
    case Merge_operator::Add:
        matrix.col(column_index) = matrix.col(column_index).array() + matrix.col(twin.merge_column).array();
        break;
    case Merge_operator::Sub:
        matrix.col(column_index) = matrix.col(column_index).array() - matrix.col(twin.merge_column).array();
        break;
    case Merge_operator::Mul:
        matrix.col(column_index) = matrix.col(column_index).array() * matrix.col(twin.merge_column).array();
        break;
    case Merge_operator::Div:
        matrix.col(column_index) = matrix.col(column_index).array() / matrix.col(twin.merge_column).array();
        break;
    }
    return matrix;
}

std::string MergeAllele::to_string() const {
    std::string sallele;
    sallele = "column: " + std::to_string(column_index) + " char_number: " + std::to_string(characteristic_number) + "\n";
    for (auto const& twin : allele) {
        sallele += "merge operator: " + merge_operator_names.at(twin.merge_operator) + " column: " + std::to_string(twin.merge_column) + "\n";
    }
    sallele += "\n";
    return sallele;
}

std::string MergeAllele::to_string_code() const {
    std::string sallele;
    sallele = "MA" + std::to_string(column_index);
    for (auto const& twin : allele) {
        sallele += merge_operator_names.at(twin.merge_operator) + std::to_string(twin.merge_column);
    }
    sallele += "E";
    return sallele;
}





TransformXAllele::TransformXAllele(int index) : EvoGene(index), allele{} {}

TransformXAllele::~TransformXAllele() {}

Eigen::MatrixXd& TransformXAllele::transform(Eigen::MatrixXd& matrix) {
    switch (allele) {
    case Transform_operator::Exp:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)exp(s); });
        break;
    case Transform_operator::Pow:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)pow(s, characteristic_number); });
        break;
    case Transform_operator::Sqt:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)sqrt(s); });
        break;
    case Transform_operator::Csqt:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)cbrt(s); });
        break;
    case Transform_operator::Nul:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return s * 0; });
        break;
    case Transform_operator::Let:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return s * 1; });
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
    case Transform_operator::Sin:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)sin(s); });
        break;
    case Transform_operator::Cos:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (double)cos(s); });
        break;
    }
    return matrix;
}

std::string TransformXAllele::to_string() const {
    std::string sallele;
    sallele = "column: " + std::to_string(column_index) + " char_number: " + std::to_string(characteristic_number);
    sallele += " transform operator: " + transform_operator_names.at(allele) + "\n\n";
    return sallele;
}

std::string TransformXAllele::to_string_code() const {
    std::string sallele;
    sallele = "TX" + std::to_string(column_index) + std::to_string(characteristic_number) + transform_operator_names.at(allele);
    return sallele;
}

void TransformXAllele::resetCharacteristicNumber(int number) {
    characteristic_number = number;
}





TransformYAllele::TransformYAllele() : EvoGene(0), allele{} {}

TransformYAllele::~TransformYAllele() {};

void TransformYAllele::resetCharacteristicNumber(int number) {
    characteristic_number = number;
}

Eigen::MatrixXd& TransformYAllele::transform(Eigen::MatrixXd& matrix) {
    return matrix;
}

Eigen::VectorXd& TransformYAllele::transformVector(Eigen::VectorXd& vector) {
    switch (allele) {
    case Transform_operator::Exp:
        vector = vector.unaryExpr([&](const auto s) { return (double)exp(s); });
        break;
    case Transform_operator::Pow:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, characteristic_number); });
        break;
    case Transform_operator::Sqt:
        vector = vector.unaryExpr([&](const auto s) { return (double)sqrt(s); });
        break;
    case Transform_operator::Csqt:
        vector = vector.unaryExpr([&](const auto s) { return (double)cbrt(s); });
        break;
    case Transform_operator::Let:
        vector = vector.unaryExpr([&](const auto s) { return s * 1; });
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
    case Transform_operator::Sin:
        vector = vector.unaryExpr([&](const auto s) { return (double)sin(s); });
        break;
    case Transform_operator::Cos:
        vector = vector.unaryExpr([&](const auto s) { return (double)cos(s); });
        break;
    case Transform_operator::Nul: //it doesnt happpen
        vector = vector.unaryExpr([&](const auto s) { return s; });
        break;
    }

    return vector;
}

Eigen::VectorXd& TransformYAllele::transformBack(Eigen::VectorXd& vector) {
    switch (allele) {
    case Transform_operator::Exp:
        vector = vector.unaryExpr([&](const auto s) { return (double)log(s); });
        break;
    case Transform_operator::Pow:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, (1 / characteristic_number)); });
        break;
    case Transform_operator::Sqt:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, 2); });
        break;
    case Transform_operator::Csqt:
        vector = vector.unaryExpr([&](const auto s) { return (double)pow(s, 3); });
        break;
    case Transform_operator::Let:
        vector = vector.unaryExpr([&](const auto s) { return s * 1; });
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
    case Transform_operator::Sin:
        vector = vector.unaryExpr([&](const auto s) { return (double)asin(s); });
        break;
    case Transform_operator::Cos:
        vector = vector.unaryExpr([&](const auto s) { return (double)acos(s); });
        break;
    case Transform_operator::Nul: //it doesnt happpen
        vector = vector.unaryExpr([&](const auto s) { return s; });
        break;
    }
    return vector;
}

std::string TransformYAllele::to_string() const {
    std::string sallele;
    sallele += "column: " + std::to_string(column_index) + " char_number: " + std::to_string(characteristic_number);
    sallele += " transform operator: " + transform_operator_names.at(allele) + "\n\n";
    return sallele;
}

std::string TransformYAllele::to_string_code() const {
    std::string sallele;
    sallele += "TY" + std::to_string(characteristic_number) + transform_operator_names.at(allele);
    return sallele;
}





RobustAllele::RobustAllele() : EvoGene(0), allele{} {};

RobustAllele::~RobustAllele() {};

Eigen::MatrixXd& RobustAllele::transform(Eigen::MatrixXd& matrix) {
    Eigen::VectorXi v = Eigen::Map<Eigen::VectorXi>(allele.data(), allele.size());
    Eigen::MatrixXd m = matrix(v, Eigen::all);
    matrix = m;
    return matrix;
}

Eigen::VectorXd& RobustAllele::transformVector(Eigen::VectorXd& matrix) {
    Eigen::VectorXi v = Eigen::Map<Eigen::VectorXi>(allele.data(), allele.size());
    Eigen::VectorXd m = matrix(v);
    matrix = m;
    return matrix;
}

std::string RobustAllele::to_string() const {
    std::string sallele;
    sallele = "column: " + std::to_string(column_index) + " char_number: " + std::to_string(characteristic_number);
    sallele += " robust allele size: " + std::to_string(allele.size()) + "\n\n";
    return sallele;
}

std::string RobustAllele::to_string_code() const {
    std::string sallele;
    sallele = "RA" + std::string(allele.begin(), allele.end());
    return sallele;
}


