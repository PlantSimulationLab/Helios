/** \file "ParameterOptimization.h" Primary header file for the Parameter Optimization plug-in.

Copyright (C) 2016-2025 Brian Bailey

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
#include <variant>
#include <vector>
#include "Context.h"

//! Parameter type for optimization
enum class ParameterType {
    FLOAT, //!< Continuous float parameter (default)
    INTEGER, //!< Integer parameter (rounded to nearest whole number)
    CATEGORICAL //!< Categorical parameter (picks from a set of allowed values)
};

struct ParameterToOptimize {
    float value; //!< Current value of the parameter
    float min; //!< Minimum allowed value (ignored for CATEGORICAL)
    float max; //!< Maximum allowed value (ignored for CATEGORICAL)
    ParameterType type = ParameterType::FLOAT; //!< Parameter type
    std::vector<float> categories; //!< Allowed values for CATEGORICAL type (ignored otherwise)
};

//! Convenience typedef for a list of parameters to optimize
using ParametersToOptimize = std::unordered_map<std::string, ParameterToOptimize>;

// --- Crossover operator structs ---

//! Standard component-wise blend crossover
struct BLXAlphaCrossover {
    float alpha = 0.5f; //!< Blend parameter controlling extrapolation range
};

//! BLX-alpha in PCA-transformed space (better for non-separable problems)
struct BLXPCACrossover {
    float alpha = 0.5f; //!< Blend parameter controlling extrapolation range
    size_t pca_update_interval = 5; //!< Generations between PCA recomputation
};

// --- Mutation operator structs ---

//! Standard per-gene Gaussian mutation (axis-aligned, default)
struct PerGeneMutation {
    float rate = 0.1f; //!< Per-gene mutation probability
};

//! Single-gate isotropic Gaussian mutation (all genes mutated together)
struct IsotropicMutation {
    float rate = 0.1f; //!< Mutation gate probability
};

//! PCA-Gaussian + PCA-Cauchy + Random Direction hybrid mutation
struct HybridMutation {
    float rate = 0.15f; //!< Mutation gate probability
    size_t pca_update_interval = 5; //!< Generations between PCA recomputation
    float sigma_pca = 0.25f; //!< PCA-Gaussian sigma in whitened space
    float gamma_cauchy = 0.1f; //!< PCA-Cauchy scale parameter
    float sigma_random = 0.3f; //!< Random direction step size as fraction of avg range
    float pca_gaussian_prob = 0.70f; //!< Probability of PCA-Gaussian sub-strategy
    float pca_cauchy_prob = 0.20f; //!< Probability of PCA-Cauchy sub-strategy
};

// --- Algorithm structs ---

//! Genetic Algorithm settings
struct GeneticAlgorithm {
    size_t generations = 100; //!< Number of generations to run
    size_t population_size = 20; //!< Size of the population
    float crossover_rate = 0.5f; //!< Probability of crossover
    float elitism_rate = 0.05f; //!< Fraction of top individuals kept in next generation

    //! Crossover operator selection (default: BLX-alpha)
    std::variant<BLXAlphaCrossover, BLXPCACrossover> crossover;

    //! Mutation operator selection (default: per-gene)
    std::variant<PerGeneMutation, IsotropicMutation, HybridMutation> mutation;

    //! Exploration-biased preset (large population, high mutation, BLXPCA + HYBRID)
    static GeneticAlgorithm explore();

    //! Exploitation-biased preset (smaller population, low mutation, BLXPCA + per-gene)
    static GeneticAlgorithm exploit();
};

//! Bayesian Optimization settings using Gaussian Process
struct BayesianOptimization {
    size_t max_evaluations = 100; //!< Total number of function evaluations
    size_t initial_samples = 0; //!< Number of initial random samples (0 = auto: 2*num_params)
    float ucb_kappa = 2.0f; //!< Exploration parameter for UCB acquisition (higher = more exploration)
    size_t max_gp_samples = 200; //!< Maximum samples to keep in GP (for memory efficiency)
    size_t acquisition_samples = 1000; //!< Random samples for acquisition optimization

    //! Exploration-biased preset (high kappa, many samples)
    static BayesianOptimization explore();

    //! Exploitation-biased preset (low kappa)
    static BayesianOptimization exploit();
};

//! Covariance Matrix Adaptation Evolution Strategy settings
struct CMAES {
    size_t max_evaluations = 200; //!< Total number of function evaluations
    size_t lambda = 0; //!< Population size (0 = auto: 4+floor(3*ln(n)))
    float sigma = 0.3f; //!< Initial step size (typical: 0.3)

    //! Exploration-biased preset (large sigma)
    static CMAES explore();

    //! Exploitation-biased preset (small sigma)
    static CMAES exploit();
};

//! Parameter optimization routines
class ParameterOptimization {
public:
    struct Result {
        ParametersToOptimize parameters; //!< Optimized parameters
        float fitness = 0.f; //!< Fitness value of the best individual
    };

    //! Select the Genetic Algorithm for optimization.
    /**
     * \param[in] algorithm GA hyperparameters including crossover/mutation operator selection
     */
    void setAlgorithm(const GeneticAlgorithm &algorithm);

    //! Select Bayesian Optimization for optimization.
    /**
     * \param[in] algorithm BO hyperparameters including acquisition function parameters
     */
    void setAlgorithm(const BayesianOptimization &algorithm);

    //! Select CMA-ES for optimization.
    /**
     * \param[in] algorithm CMA-ES hyperparameters
     */
    void setAlgorithm(const CMAES &algorithm);

    //! I/O configuration
    bool print_progress = false; //!< Optional progress printout
    std::string write_result_to_file; //!< Name of file to write final result to, skips writing if empty
    std::string write_progress_to_file; //!< Name of file to write progress to, skips writing if empty
    std::string read_input_from_file; //!< Name of file to read parameter set in from, ignores if empty

    //! Run the optimization with stored algorithm settings.
    /**
     * \param[in] simulation A self contained Helios simulation that takes in parameters, performs computation, and returns a floating point objective value.
     * \param[in] parameters A parameter set to optimize over
     */
    Result run(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters);

    //! Runs a self-test of the ParameterOptimization plugin, verifying its functionality.
    /**
     * The self-test executes a series of internal tests, validating critical methods and
     * functionalities of the ParameterOptimization plugin. Errors encountered during the process are counted and reported.
     *
     * \return The number of errors encountered during the self-test. Returns 0 if all tests pass successfully.
     */
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Runs visualization tests that output population data for plotting algorithm performance.
    /**
     * Tests all three optimization methods (GA, BO, CMA-ES) on standard benchmark functions
     * and outputs iteration-by-iteration population data to CSV files for visualization.
     *
     * \param[in] output_dir Directory path where output CSV files will be written
     * \return 0 on success, non-zero on error
     */
    static int selfTestVisualized(const std::string &output_dir = ".");

private:
    std::variant<GeneticAlgorithm, BayesianOptimization, CMAES> algorithm_;
    bool message_flag = false;
    std::vector<float> fitness_over_time;
};

#endif
