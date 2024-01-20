#pragma once
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "XoshiroCpp.hpp"
#include "EExpression.hpp"

enum class Transform_operator { Sqr, Cub, Pow, Wek, Sqt, Csqt, Let, Log, Log10, Log2, Nul };

const std::map<Transform_operator, std::string> transform_operator_names{
    {Transform_operator::Sqr, "Sqr"} ,
    {Transform_operator::Cub, "Cub"} ,
    {Transform_operator::Pow, "Pow"} ,
    {Transform_operator::Wek, "Wek"} ,
    {Transform_operator::Sqt, "Sqt"},
    {Transform_operator::Csqt, "Csqt"},
    {Transform_operator::Let, "Let"},
    {Transform_operator::Log, "Log"},
    {Transform_operator::Log10, "Log10"},
    {Transform_operator::Log2, "Log2"},
    {Transform_operator::Nul, "Nul"}
};

const std::map<Transform_operator, std::string> transform_operator_symbols{
    {Transform_operator::Sqr, "^2"} ,
    {Transform_operator::Cub, "^3"} ,
    {Transform_operator::Pow, "^"} ,
    {Transform_operator::Wek, "^-1/"} ,
    {Transform_operator::Sqt, "^-1/2"},
    {Transform_operator::Csqt, "^-1/3"},
    {Transform_operator::Let, "*1"},
    {Transform_operator::Log, "Log(e)"},
    {Transform_operator::Log10, "Log(10)"},
    {Transform_operator::Log2, "Log(2)"},
    {Transform_operator::Nul, "*0"}
};

inline constexpr int merge_operator_maxindex = 3;
inline constexpr int transform_operator_maxindex = 10;
inline constexpr int transform_y_operator_maxindex = 9;

class EvoGene {
public:
    EvoGene();
    EvoGene(int);
    virtual ~EvoGene();
    virtual void transform(Eigen::MatrixXd&) const = 0;
    virtual std::string to_string() const = 0;
    virtual std::string to_string_code() const = 0;

protected:
    int column_index;
    float characteristic_number;
};


class MergeAllele : EvoGene {
public:
    MergeAllele(int);
    ~MergeAllele();
    void transform(Eigen::MatrixXd&) const override;
    std::string to_string() const override;
    std::string to_string_code() const override;
    EExpression allele;
};


class TransformXAllele : EvoGene {
public:
    TransformXAllele(int);
    ~TransformXAllele();
    void transform(Eigen::MatrixXd&) const override;
    std::string to_string() const override;
    std::string to_string_code() const override;
    void resetCharacteristicNumber(float);

    Transform_operator allele;
};

class TransformYAllele : EvoGene {
public:
    TransformYAllele();
    ~TransformYAllele();
    void transform(Eigen::MatrixXd&) const override;
    std::string to_string() const override;
    std::string to_string_code() const override;
    void transform_vector(Eigen::VectorXd&) const;
    void transformBack(Eigen::VectorXd&) const;
    void resetCharacteristicNumber(float);

    Transform_operator allele;
};

class RobustAllele : EvoGene {
public:
    RobustAllele();
    ~RobustAllele();
    void transform(Eigen::MatrixXd&) const override;
    std::string to_string() const override;
    std::string to_string_code() const override;
    void transform_vector(Eigen::VectorXd&) const;

    std::vector<int> allele;
};

