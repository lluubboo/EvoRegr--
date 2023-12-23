#pragma once
#include <string>
#include "BoundaryConditions.hpp"

/**
 * @brief Interface for the EvoAPI class.
 * 
 * This interface defines the methods that need to be implemented by the EvoAPI class.
 * The EvoAPI class provides functionality for setting boundary conditions, selecting solvers,
 * setting decomposition methods, loading files, calling the predict method, and checking if
 * the API is ready to predict.
 */
class IEvoAPI {
public:
    virtual ~IEvoAPI() = default;

    /**
     * @brief Sets the boundary conditions.
     */
    virtual void set_boundary_conditions(EvoBoundaryConditions const&) = 0;

    /**
     * @brief Sets the solver.
     * 
     * @param solver_name The name of the solver.
     */
    virtual void set_solver(std::string const& solver_name) = 0;

    /**
     * @brief Loads a file.
     * 
     * @param filepath The path to the file.
     */
    virtual void load_file(std::string const& filepath) = 0;

    /**
     * @brief Calls the predict method.
     */
    virtual void call_predict_method() = 0;

    /**
     * @brief Checks if the API is ready to predict.
     * 
     * @return True if the API is ready to predict, false otherwise.
     */
    virtual bool is_ready_to_predict() const = 0;
};