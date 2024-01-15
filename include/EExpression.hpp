#pragma once
#include <iostream>
#include <cstddef>
#include <vector>
#include <array>
#include <numeric>
#include <random>
#include <Eigen/Dense>

class EExpression {

    // binary operators used in generated expressions
    static constexpr std::array<std::array<char, 2>, 4> _binary_operators = {
        std::array<char, 2>{'b', '+'},
        std::array<char, 2>{'b', '-'},
        std::array<char, 2>{'b', '*'},
        std::array<char, 2>{'b', '/'}
    };

    // the number shows how many variables I have available
    // it must be known at instance creation
    unsigned int _var_count;

    // it must be known at instance creation
    // possible field for some optimization
    // the number shows how many leafs I have in the expression
    unsigned int _expr_leaf_count;

    // the expression is stored as a vector of array of {type, index}
    std::vector<std::array<char, 2>> _expression;

    // indexes which points to variables
    // variables in expression are indexed from 0 to _var_count - 1
    std::vector<size_t> _variables;
    
    // constants container
    // number of constants is not known at instance creation
    // constants can be changed during some optimization process
    std::vector<double> _constants;

    
    /**
     * Performs the specified operation on two Eigen::VectorXd operands.
     *
     * @param operand1 The first operand.
     * @param operand2 The second operand.
     * @param operation The operation to perform ('+', '-', '*', or '/').
     * @return The result of the operation as an Eigen::VectorXd.
     *         If the operation is not recognized, returns a zero vector of the same size as operand1.
     */
    Eigen::VectorXd perform_operation(const Eigen::VectorXd& operand1, const Eigen::VectorXd& operand2, char operation) const noexcept {
        switch (operation) {
        case '+':
            return operand1.array() + operand2.array();
        case '-':
            return operand1.array() - operand2.array();
        case '*':
            return operand1.array() * operand2.array();
        case '/':
            return operand1.array() / operand2.array();
        default:
            //ignore the operation
            return Eigen::VectorXd::Zero(operand1.size());
        }
    }

    /**
     * @brief Calculates the Arnold and Sleep probability function for generating balanced parentheses strings.
     *
     * This function implements the probability function described in the paper "Uniform Random Number Generation of n Balanced Parenthesis Strings" by Arnold and Sleep (1980).
     * It's used to control the generation of random expressions in the `generate_random` function.
     *
     * @param x The current number of open parentheses that have not been closed.
     * @param n The total number of parentheses pairs in the expression.
     * @param t The current position in the expression.
     * @return The probability of generating an open parenthesis at the current position.
     */
    static inline double arnold_sleep_probability_function(double x, double n, double t) {
        return ((x + 2) / (x + 1)) * (((2 * n) - t - x) / (2 * ((2 * n) - t)));
    }

    /**
     * Generates a random binary operator using the specified engine.
     * The generated operator is in the form of a character array with two elements.
     * The first element is always 'b', and the second element is one of the following: '+', '-', '*', or '/'.
     *
     * @param engine The engine used for random number generation.
     * @return A character array representing the random binary operator.
     */
    template<typename Engine>
    inline std::array<char, 2> get_random_binary_operator(Engine& engine) {
        std::uniform_int_distribution<int> distribution(0, EExpression::_binary_operators.size() - 1);
        return EExpression::_binary_operators[distribution(engine)];
    }

    /**
     * Generates a random constant using the specified engine.
     * The generated constant is in the form of a character array with two elements.
     * The first element is always 'c', and the second element is a digit representing the constant index.
     *
     * @param engine The engine used for random number generation.
     * @return A character array representing the random constant.
     */
    template<typename Engine>
    inline std::array<char, 2> get_random_constant(Engine& engine) {
        std::uniform_real_distribution<double> distribution(std::nextafter(0.0, 1.0), 100000.0);
        _constants.push_back(distribution(engine));
        return { 'c', static_cast<char>('0' + _constants.size() - 1) };
    }

    /**
     * Generates a random operand.
     * 
     * This function generates a random operand for an expression. The operand can be either a constant or a variable.
     * 
     * @tparam Engine The type of the random number engine.
     * @param engine The random number engine used for generating random numbers.
     * @return An array of characters representing the random operand.
     */
    template<typename Engine>
    inline std::array<char, 2> get_random_operand(Engine& engine) {
        std::uniform_int_distribution<int> distribution(0, 10);
        if (distribution(engine) == 0) {
            return get_random_constant(engine);
        }
        else {
            return get_random_variable(engine);
        }
    }

    /**
     * Generates a random variable using the specified engine.
     * The generated variable is in the form of a character array with two elements.
     * The first element is always 'v', and the second element is a digit representing the variable index.
     *
     * @param engine The engine used for random number generation.
     * @return A character array representing the random variable.
     */
    template<typename Engine>
    inline std::array<char, 2> get_random_variable(Engine& engine) {
        std::uniform_int_distribution<int> distribution(0, _var_count - 1);
        return { 'v', static_cast<char>('0' + distribution(engine)) };
    }

    /**
     * @brief Generates a random expression using the provided random number generator.
     *
     * This function generates a random expression based on the method described in
     * "Uniform Random Number Generation of n Balanced Parenthesis Strings" by Arnold and Sleep (1980).
     * The generated expression is stored in the `_expression` member of the `EExpression` object.
     *
     * @tparam Engine The type of the random number generator. This type must meet the requirements
     * of the RandomNumberEngine named requirement from the C++ Standard Library.
     * @param engine A reference to the random number generator to use.
     */
    template<typename Engine>
    void generate_random(Engine& engine) {

        unsigned int nodeCount = _expr_leaf_count - 1;
        unsigned int symbolCount = 2 * nodeCount;

        int currentWalkValue = 0;

        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (unsigned int index = 0; index < symbolCount; ++index) {

            double choice = distribution(engine);

            if (choice <= arnold_sleep_probability_function(currentWalkValue, nodeCount, index)) {
                ++currentWalkValue;
                _expression.push_back(get_random_binary_operator(engine));

            }
            else {
                --currentWalkValue;
                _expression.push_back(get_random_operand(engine));
            }
        }
        _expression.push_back(get_random_operand(engine));
    }

    std::string get_postfix_expression_string() const {
        std::string result;
        for (auto const& symbol : _expression) {
            result += symbol[0];
            result += symbol[1];
        }
        return result;
    }

public:

    // default constructor
    EExpression() :
        _var_count(0),
        _expr_leaf_count(0),
        _expression(0),
        _variables(0),
        _constants(0)
    {}

    // constructor
    template<typename Engine>
    EExpression(unsigned int var_count, unsigned int leaf_count, Engine& engine) :
        _var_count(var_count),
        _expr_leaf_count(leaf_count),
        _expression(0),
        _variables(var_count),
        _constants(0)
    {
        if (var_count > std::numeric_limits<char>::max() || leaf_count > std::numeric_limits<char>::max()) {
            std::cerr << "Error: var_count or leaf_count is too large to fit in a char\n";
        }

        std::iota(_variables.begin(), _variables.end(), 0);
        generate_random(engine);
    }

    // copy constructor
    EExpression(const EExpression& other) :
        _var_count(other._var_count),
        _expr_leaf_count(other._expr_leaf_count),
        _expression(other._expression),
        _variables(other._variables),
        _constants(other._constants)
    {}

    // move constructor
    EExpression(EExpression&& other) noexcept :
        _var_count(other._var_count),
        _expr_leaf_count(other._expr_leaf_count),
        _expression(std::move(other._expression)),
        _variables(std::move(other._variables)),
        _constants(std::move(other._constants))
    {
        // Reset the other object
        other._var_count = 0;
        other._expr_leaf_count = 0;
    }

    EExpression& operator=(const EExpression& other) {
        if (this != &other) {
            _var_count = other._var_count;
            _expr_leaf_count = other._expr_leaf_count;
            _expression = other._expression;
            _variables = other._variables;
            _constants = other._constants;
        }
        return *this;
    }

    Eigen::VectorXd evaluate_expression(Eigen::MatrixXd const& data) const {

        // the stack is used to store the operands
        std::vector<Eigen::VectorXd> stack;

        // reserve the memory for the stack. it cant be bigger than the expression size
        stack.reserve(_expression.size());

        for (int i = _expression.size() - 1; i >= 0; --i) {

            auto const& symbol = _expression[i];

            if (symbol[0] == 'b') {

                Eigen::VectorXd operand1 = std::move(stack.back());
                stack.pop_back();
                Eigen::VectorXd operand2 = std::move(stack.back());
                stack.pop_back();

                stack.push_back(perform_operation(operand1, operand2, symbol[1]));
            }
            else if (symbol[0] == 'c') {
                stack.emplace_back(Eigen::VectorXd::Constant(data.rows(), _constants[symbol[1] - '0']));
            }
            else if (symbol[0] == 'v') {
                stack.push_back(data.col(_variables[symbol[1] - '0']));
            }
        }
        
        return stack[0];
    }

    std::vector<double> get_constants() const noexcept {
        return _constants;
    }

    void set_constants(std::vector<double> constants) noexcept {
        if (constants.size() != _constants.size())
        {
            std::cerr << "EExpression warning: constants size mismatch." << std::endl;
            return;
        }
        _constants = constants;
    }

    void set_constants(std::vector<double>&& constants) noexcept {
        if (constants.size() != _constants.size())
        {
            std::cerr << "EExpression warning: constants size mismatch." << std::endl;
            return;
        }
        _constants = std::move(constants);
    }

    bool empty() const noexcept {
        return _expression.empty();
    }

    std::string get_expression() const {
        std::string result;
        for (const auto& symbol : _expression) {
            if (symbol[0] == 'b') {
                result += std::string(1, symbol[1]);
            }
            else if (symbol[0] == 'c') {
                result += std::to_string(_constants[symbol[1] - '0']);
            }
            else if (symbol[0] == 'v') {
                result += "v" + std::to_string(_variables[symbol[1] - '0']);
            }
        }
        return result;
    }
};