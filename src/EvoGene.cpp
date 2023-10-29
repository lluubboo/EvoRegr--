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

EvoGene::EvoGene(int index, XoshiroCpp::Xoshiro256Plus& random_engine) : column_index{ index }, characteristic_number{ RandomNumbers::rand_interval_int(0, 4, random_engine) } {}

EvoGene::~EvoGene() {}




MergeAllele::MergeAllele(int index, XoshiroCpp::Xoshiro256Plus& random_engine) : EvoGene(index, random_engine), allele{} {}

MergeAllele::~MergeAllele() {}

Eigen::MatrixXf& MergeAllele::transform(Eigen::MatrixXf& matrix) {
    for (auto const& twin : allele) {
        modifyMatrixAccordingToTwin(twin, matrix);
    }
    return matrix;
}

Eigen::MatrixXf& MergeAllele::modifyMatrixAccordingToTwin(MergeTwin const& twin, Eigen::MatrixXf& matrix) {
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





TransformXAllele::TransformXAllele(int index, XoshiroCpp::Xoshiro256Plus& random_engine) : EvoGene(index, random_engine), allele{} {}

TransformXAllele::~TransformXAllele() {}

Eigen::MatrixXf& TransformXAllele::transform(Eigen::MatrixXf& matrix) {
    switch (allele) {
    case Transform_operator::Exp:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)exp(s); });
        break;
    case Transform_operator::Pow:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)pow(s, characteristic_number); });
        break;
    case Transform_operator::Sqt:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)sqrt(s); });
        break;
    case Transform_operator::Csqt:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)cbrt(s); });
        break;
    case Transform_operator::Nul:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return s * 0; });
        break;
    case Transform_operator::Let:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return s * 1; });
        break;
    case Transform_operator::Log:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)log(s); });
        break;
    case Transform_operator::Log10:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)log10(s); });
        break;
    case Transform_operator::Log2:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)log2(s); });
        break;
    case Transform_operator::Sin:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)sin(s); });
        break;
    case Transform_operator::Cos:
        matrix.col(column_index) = matrix.col(column_index).unaryExpr([&](const auto s) { return (float)cos(s); });
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





TransformYAllele::TransformYAllele(int index, XoshiroCpp::Xoshiro256Plus& random_engine) : EvoGene(index, random_engine), allele{} {}

TransformYAllele::~TransformYAllele() {};

void TransformYAllele::resetCharacteristicNumber(int number) {
    characteristic_number = number;
}

Eigen::MatrixXf& TransformYAllele::transform(Eigen::MatrixXf& matrix) {
    return matrix;
}

Eigen::VectorXf& TransformYAllele::transformVector(Eigen::VectorXf& vector) {
    switch (allele) {
    case Transform_operator::Exp:
        vector = vector.unaryExpr([&](const auto s) { return (float)exp(s); });
        break;
    case Transform_operator::Pow:
        vector = vector.unaryExpr([&](const auto s) { return (float)pow(s, characteristic_number); });
        break;
    case Transform_operator::Sqt:
        vector = vector.unaryExpr([&](const auto s) { return (float)sqrt(s); });
        break;
    case Transform_operator::Csqt:
        vector = vector.unaryExpr([&](const auto s) { return (float)cbrt(s); });
        break;
    case Transform_operator::Let:
        vector = vector.unaryExpr([&](const auto s) { return s * 1; });
        break;
    case Transform_operator::Log:
        vector = vector.unaryExpr([&](const auto s) { return (float)log(s); });
        break;
    case Transform_operator::Log10:
        vector = vector.unaryExpr([&](const auto s) { return (float)log10(s); });
        break;
    case Transform_operator::Log2:
        vector = vector.unaryExpr([&](const auto s) { return (float)log2(s); });
        break;
    case Transform_operator::Sin:
        vector = vector.unaryExpr([&](const auto s) { return (float)sin(s); });
        break;
    case Transform_operator::Cos:
        vector = vector.unaryExpr([&](const auto s) { return (float)cos(s); });
        break;
    case Transform_operator::Nul: //it doesnt happpen
        vector = vector.unaryExpr([&](const auto s) { return s; });
        break;
    }

    return vector;
}

Eigen::VectorXf& TransformYAllele::transformBack(Eigen::VectorXf& vector) {
    switch (allele) {
    case Transform_operator::Exp:
        vector = vector.unaryExpr([&](const auto s) { return (float)log(s); });
        break;
    case Transform_operator::Pow:
        vector = vector.unaryExpr([&](const auto s) { return (float)pow(s, (1 / characteristic_number)); });
        break;
    case Transform_operator::Sqt:
        vector = vector.unaryExpr([&](const auto s) { return (float)pow(s, 2); });
        break;
    case Transform_operator::Csqt:
        vector = vector.unaryExpr([&](const auto s) { return (float)pow(s, 3); });
        break;
    case Transform_operator::Let:
        vector = vector.unaryExpr([&](const auto s) { return s * 1; });
        break;
    case Transform_operator::Log:
        vector = vector.unaryExpr([&](const auto s) { return (float)exp(s); });
        break;
    case Transform_operator::Log10:
        vector = vector.unaryExpr([&](const auto s) { return (float)pow(10, s); });
        break;
    case Transform_operator::Log2:
        vector = vector.unaryExpr([&](const auto s) { return (float)pow(2, s); });
        break;
    case Transform_operator::Sin:
        vector = vector.unaryExpr([&](const auto s) { return (float)asin(s); });
        break;
    case Transform_operator::Cos:
        vector = vector.unaryExpr([&](const auto s) { return (float)acos(s); });
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





RobustAllele::RobustAllele() : EvoGene(0), allele{} {};

RobustAllele::~RobustAllele() {};

Eigen::MatrixXf& RobustAllele::transform(Eigen::MatrixXf& matrix) {
    Eigen::VectorXi v = Eigen::Map<Eigen::VectorXi>(allele.data(), allele.size());
    Eigen::MatrixXf m = matrix(v, Eigen::all);
    matrix = m;
    return matrix;
}

Eigen::VectorXf& RobustAllele::transformVector(Eigen::VectorXf& matrix) {
    Eigen::VectorXi v = Eigen::Map<Eigen::VectorXi>(allele.data(), allele.size());
    Eigen::VectorXf m = matrix(v);
    matrix = m;
    return matrix;
}

std::string RobustAllele::to_string() const {
    std::string sallele;
    sallele += "column: " + std::to_string(column_index) + " char_number: " + std::to_string(characteristic_number);
    sallele += " robust allele size: " + std::to_string(allele.size()) + "\n\n";
    return sallele;
}


