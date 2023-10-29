#pragma once
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "XoshiroCpp.hpp"

enum class Merge_operator { Add, Sub, Mul, Div };
enum class Transform_operator { Exp, Pow, Sqt, Csqt, Let, Log, Log10, Log2, Sin, Cos, Nul };

const std::map<Merge_operator, std::string> merge_operator_names{
    {Merge_operator::Add, "Add"},
    {Merge_operator::Sub, "Sub"} ,
    {Merge_operator::Mul, "Mul"},
    {Merge_operator::Div, "Div"}
};

const std::map<Transform_operator, std::string> transform_operator_names{
    {Transform_operator::Exp, "Exp"},
    {Transform_operator::Pow, "Pow"} ,
    {Transform_operator::Sqt, "Sqt"},
    {Transform_operator::Csqt, "Csqt"},
    {Transform_operator::Let, "Let"},
    {Transform_operator::Log, "Log"},
    {Transform_operator::Log10, "Log10"},
    {Transform_operator::Log2, "Log2"},
    {Transform_operator::Sin, "Sin"},
    {Transform_operator::Cos, "Cos"},
    {Transform_operator::Nul, "Nul"}
};

inline constexpr int merge_operator_maxindex = 4;
inline constexpr int transform_operator_maxindex = 11;
inline constexpr int transform_y_operator_maxindex = 10;

struct MergeTwin {
    int merge_column;
    Merge_operator merge_operator;
};


class EvoGene {
public:
    EvoGene();
    EvoGene(int);
    EvoGene(int, XoshiroCpp::Xoshiro256Plus& random_engine);
    virtual ~EvoGene();
    virtual Eigen::MatrixXf& transform(Eigen::MatrixXf&) = 0;
    virtual std::string to_string() const = 0;

protected:
    int column_index;
    int characteristic_number;
};


class MergeAllele : EvoGene {
public:
    MergeAllele(int, XoshiroCpp::Xoshiro256Plus&);
    ~MergeAllele();
    Eigen::MatrixXf& modifyMatrixAccordingToTwin(MergeTwin const&, Eigen::MatrixXf&);
    Eigen::MatrixXf& transform(Eigen::MatrixXf&) override;
    std::string to_string() const override;
    std::vector<MergeTwin> allele;
};


class TransformXAllele : EvoGene {
public:
    TransformXAllele(int, XoshiroCpp::Xoshiro256Plus&);
    ~TransformXAllele();
    Eigen::MatrixXf& transform(Eigen::MatrixXf&) override;
    std::string to_string() const override;
    Transform_operator allele;
};

class TransformYAllele : EvoGene {
public:
    TransformYAllele(int, XoshiroCpp::Xoshiro256Plus&);
    ~TransformYAllele();
    Eigen::MatrixXf& transform(Eigen::MatrixXf&) override;
    std::string to_string() const override;
    Eigen::VectorXf& transformVector(Eigen::VectorXf&);
    Eigen::VectorXf& transformBack(Eigen::VectorXf&);
    void resetCharacteristicNumber(int);
    Transform_operator allele;
};

class RobustAllele : EvoGene {
public:
    RobustAllele();
    ~RobustAllele();
    RobustAllele createRandomAllele(int, XoshiroCpp::Xoshiro256Plus&);
    Eigen::MatrixXf& transform(Eigen::MatrixXf&) override;
    std::string to_string() const override;
    Eigen::VectorXf& transformVector(Eigen::VectorXf&);
    std::vector<int> allele;
};