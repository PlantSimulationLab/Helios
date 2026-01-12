/** \file "ParameterOptimization.h" Primary header file for the Parameter Optimization plug-in.

Copyright (C) 2016-2026 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#ifndef PARAMETEROPTIMIZATION
#define PARAMETEROPTIMIZATION

#include <functional>
#include <limits>
#include "Context.h"

struct ParameterToOptimize {
    float value; //!< Current value of the parameter
    float min; //!< Minimum allowed value
    float max; //!< Maximum allowed value
};

//! Convenience typedef for a list of parameters to optimize
using ParametersToOptimize = std::unordered_map<std::string, ParameterToOptimize>;

//! Settings structure controlling the optimization
struct OptimizationSettings {
    size_t generations = 10; //!< Number of generations
    size_t population_size = 20; //!< Size of the population
    float crossover_rate = 0.5f; //!< Probability of crossover per gene
    float mutation_rate = 0.1f; //!< Probability of mutation per gene
    float elitism_rate = 0.05f; //!< Fraction of top individuals kept in next generation
    bool print_progress = false; //!< Optional progress printout
    std::string write_result_to_file; //!< Name of file to write final result to, skips writing if empty
    std::string write_progress_to_file; //!< Name of file to write progress to, skips writing if empty
    std::string read_input_from_file; //!< Name of file to read parameter set in from, ignores if empty
};

//! Simple parameter optimization routines
class ParameterOptimization {
public:
    struct Result {
        ParametersToOptimize parameters; //!< Optimized parameters
        float fitness = 0.f; //!< Fitness value of the best individual
    };

    //! Run the plugin for a simulation with a computed objective, parameter set to optimize over, and algorithmic settings.
    /**
     * \param[in] simulation A self contained Helios simulation that takes in parameters, performs computation, and returns a floating point objective value.
     * \param[in] parameters A parameter set to optimize over
     * \param[in] settings Settings of underlying genetic algorithm that performs the optimization
     */
    Result run(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters, const OptimizationSettings &settings);

    //! Runs a self-test of the ParameterOptimization plugin, verifying its functionality.
    /**
     * The self-test executes a series of internal tests, validating critical methods and
     * functionalities of the ParameterOptimization plugin. Errors encountered during the process are counted and reported.
     *
     * \return The number of errors encountered during the self-test. Returns 0 if all tests pass successfully.
     */
    static int selfTest(int argc = 0, char **argv = nullptr);


private:
    bool message_flag = false;
    std::vector<float> fitness_over_time;
};

#endif
