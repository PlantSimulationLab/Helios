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

// --- Gradient types ---

//! Gradient of the objective w.r.t. named parameters
using ParameterGradient = std::unordered_map<std::string, float>;

//! Objective function: takes named parameters, returns scalar cost to minimize
using ObjectiveFunction = std::function<float(const ParametersToOptimize&)>;

//! Gradient function: takes named parameters, returns gradient map
using GradientFunction = std::function<ParameterGradient(const ParametersToOptimize&)>;

// --- Gradient composition helpers ---

//! Callback that writes gradient entries for a subset of parameters
using GradientSubsetFunction = std::function<void(const ParametersToOptimize&, ParameterGradient&)>;

//! A named gradient source covering a subset of parameters
struct GradientSource {
    std::string name;                       //!< Human-readable label (e.g. "photosynthesis_ad")
    std::vector<std::string> parameter_names; //!< Parameters this source computes gradients for
    GradientSubsetFunction compute;         //!< Callback that fills gradient entries
};

//! Compose multiple gradient sources into one GradientFunction.
/** Each source writes its parameter entries into the shared gradient map.
 *  The caller decides which parameters use AD vs FD by constructing the appropriate sources.
 */
inline GradientFunction makeGradientFunction(const std::vector<GradientSource>& sources) {
    return [sources](const ParametersToOptimize& params) -> ParameterGradient {
        ParameterGradient gradient;
        for (const auto& source : sources) {
            source.compute(params, gradient);
        }
        return gradient;
    };
}

//! Create a GradientSource that computes centered finite differences for the named parameters.
/**
 * \param[in] name          Human-readable label for this source
 * \param[in] parameter_names  Parameters to differentiate via FD
 * \param[in] objective     The objective function to perturb
 * \param[in] step          Relative perturbation factor (default 1e-5). Actual step per parameter is step * max(|value|, 1).
 */
GradientSource makeFDGradientSource(
    const std::string& name,
    const std::vector<std::string>& parameter_names,
    ObjectiveFunction objective,
    float step = 1e-5f
);

//! Create a GradientFunction that computes centered finite differences for all parameters.
/** Convenience wrapper around \ref makeFDGradientSource for the common case where all parameters
 *  are differentiated via FD. Parameter names are read from the ParametersToOptimize map at each call.
 *
 * \param[in] objective     The objective function to perturb
 * \param[in] step          Relative perturbation factor (default 1e-5). Actual step per parameter is step * max(|value|, 1).
 */
GradientFunction makeFDGradient(ObjectiveFunction objective, float step = 1e-5f);

// --- Constraint types (for SLSQP) ---

//! Constraint function: returns c(x) where c(x) <= 0 is satisfied
using ConstraintFunction = std::function<float(const ParametersToOptimize&)>;

//! Constraint gradient function: returns dc/d(param) for each parameter
using ConstraintGradientFunction = std::function<ParameterGradient(const ParametersToOptimize&)>;

//! A nonlinear inequality constraint: c(x) <= 0
struct Constraint {
    ConstraintFunction function;           //!< Returns c(x); constraint satisfied when c(x) <= 0
    ConstraintGradientFunction gradient;   //!< Returns dc/d(param) for each parameter
    float tolerance = 1e-6f;               //!< Constraint satisfaction tolerance
};

// --- Combined simulation for constrained optimization ---

//! Result from a constrained simulation: objective + constraints + all gradients in one pass.
/** Use this when your simulation computes everything together (e.g., ray tracing +
 *  biophysics produces A, E, T, and all sensitivities in a single forward pass).
 *  The optimizer caches the result internally so the simulation runs once per
 *  parameter point regardless of how many callbacks NLopt makes.
 */
struct ConstrainedResult {
    float objective;                              //!< Objective value (to minimize)
    ParameterGradient obj_gradient;               //!< Gradient of the objective
    std::vector<float> constraints;               //!< Constraint values: c_i(x) <= 0
    std::vector<ParameterGradient> con_gradients; //!< Gradients of each constraint
};

//! A simulation that returns objective, constraints, and all gradients in one call.
using ConstrainedSimulation = std::function<ConstrainedResult(const ParametersToOptimize&)>;

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

//! L-BFGS gradient-based optimization settings (requires NLopt at build time)
struct LBFGS {
    int max_iterations = 200;        //!< Maximum number of optimizer iterations
    double ftol_rel = 1e-6;          //!< Relative tolerance on function value
    double xtol_rel = 1e-6;          //!< Relative tolerance on parameter values
    bool verify_gradients = false;   //!< Compare user-supplied gradients vs FD during optimization
    double fd_step = 1e-5;           //!< FD step size for gradient verification
};

//! AdamW gradient-based optimization (noise-tolerant, no external dependencies)
/** Implements AdamW (Loshchilov & Hutter 2019) with decoupled weight decay.
 *  With weight_decay=0 (default), this is equivalent to standard Adam.
 */
struct Adam {
    int max_iterations = 200;        //!< Maximum number of iterations
    float learning_rate = 0.01f;     //!< Step size (alpha). Typical: 0.001-0.1
    float beta1 = 0.9f;             //!< Exponential decay rate for first moment (momentum)
    float beta2 = 0.999f;           //!< Exponential decay rate for second moment (RMSprop)
    float epsilon = 1e-8f;          //!< Numerical stability constant
    float weight_decay = 0.f;       //!< Decoupled weight decay (AdamW). 0 = standard Adam.
    double ftol_rel = 1e-6;          //!< Relative tolerance on function value for convergence
    double xtol_rel = 1e-6;          //!< Relative tolerance on parameter values for convergence
};

//! BOBYQA derivative-free local optimization (requires NLopt at build time)
/** Builds a local quadratic model from function evaluations — no gradients needed.
 *  Ideal for polishing a GA/CMA-ES result or optimizing noisy/black-box objectives.
 */
struct BOBYQA {
    int max_iterations = 200;        //!< Maximum number of function evaluations
    double ftol_rel = 1e-6;          //!< Relative tolerance on function value
    double xtol_rel = 1e-6;          //!< Relative tolerance on parameter values
    double initial_step = 0.0;       //!< Initial trust region radius (0 = auto: 10% of range)
};

//! SLSQP gradient-based optimization with nonlinear constraints (requires NLopt at build time)
/** Sequential Least Squares Programming — the only algorithm supporting nonlinear
 *  inequality constraints with gradient information. Use for problems like
 *  "maximize A subject to E < budget".
 */
struct SLSQP {
    int max_iterations = 200;        //!< Maximum number of optimizer iterations
    double ftol_rel = 1e-6;          //!< Relative tolerance on function value
    double xtol_rel = 1e-6;          //!< Relative tolerance on parameter values
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

    //! Select L-BFGS gradient-based optimization. Requires NLopt at build time.
    /**
     * \param[in] algorithm LBFGS hyperparameters
     */
    void setAlgorithm(const LBFGS &algorithm);

    //! Select Adam gradient-based optimization. Noise-tolerant, no external dependencies.
    /**
     * \param[in] algorithm Adam hyperparameters
     */
    void setAlgorithm(const Adam &algorithm);

    //! Select BOBYQA derivative-free local optimization. Requires NLopt at build time.
    void setAlgorithm(const BOBYQA &algorithm);

    //! Select SLSQP constrained gradient-based optimization. Requires NLopt at build time.
    void setAlgorithm(const SLSQP &algorithm);

    //! I/O configuration
    bool print_progress = false; //!< Optional progress printout
    std::string write_result_to_file; //!< Name of file to write final result to, skips writing if empty
    std::string write_progress_to_file; //!< Name of file to write progress to, skips writing if empty
    std::string read_input_from_file; //!< Name of file to read parameter set in from, ignores if empty

    //! Run the optimization with stored algorithm settings (derivative-free).
    /**
     * \param[in] simulation A self contained Helios simulation that takes in parameters, performs computation, and returns a floating point objective value.
     * \param[in] parameters A parameter set to optimize over
     */
    Result run(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters);

    //! Run gradient-based optimization with a user-supplied gradient function.
    /**
     * \param[in] objective  Objective function returning scalar cost to minimize
     * \param[in] parameters A parameter set to optimize over (must be all FLOAT type for gradient methods)
     * \param[in] gradient   Gradient function returning df/d(param) for each parameter.
     *                        Can be assembled from multiple sources via makeGradientFunction().
     *
     * Gradient-free algorithms (GA, BO, CMA-ES, BOBYQA) accept this overload but ignore the gradient.
     * Gradient-based algorithms (LBFGS, Adam, SLSQP) require this overload.
     */
    Result run(ObjectiveFunction objective, const ParametersToOptimize &parameters, GradientFunction gradient);

    //! Run constrained gradient-based optimization with separate callbacks (SLSQP only).
    /**
     * \param[in] objective    Objective function returning scalar cost to minimize
     * \param[in] parameters   A parameter set to optimize over (must be all FLOAT type)
     * \param[in] gradient     Gradient function for the objective
     * \param[in] constraints  Nonlinear inequality constraints: c_i(x) <= 0
     */
    Result run(ObjectiveFunction objective, const ParametersToOptimize &parameters,
               GradientFunction gradient, const std::vector<Constraint> &constraints);

    //! Run constrained optimization with a single combined simulation (SLSQP only).
    /** The simulation is called once per parameter point. The optimizer caches the result
     *  internally so NLopt's separate objective/constraint callbacks do not trigger
     *  redundant simulation runs.
     *
     *  Example:
     *  \code
     *  auto simulation = [&](const ParametersToOptimize& p) -> ConstrainedResult {
     *      auto [A, E, T, dA, dE] = run_biophysics(p);
     *      return { -A, dA_gradient, {E - E_budget}, {dE_gradient} };
     *  };
     *  auto result = opt.run(simulation, params);
     *  \endcode
     *
     * \param[in] simulation  Combined simulation returning objective, constraints, and all gradients
     * \param[in] parameters  A parameter set to optimize over (must be all FLOAT type)
     */
    Result run(ConstrainedSimulation simulation, const ParametersToOptimize &parameters);

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
    std::variant<GeneticAlgorithm, BayesianOptimization, CMAES, LBFGS, Adam, BOBYQA, SLSQP> algorithm_;
    bool algorithm_set_ = false; //!< True if setAlgorithm() was called explicitly
    bool message_flag = false;
    std::vector<float> fitness_over_time;

    //! Select default algorithm based on parameter properties and run() overload.
    void selectDefaultAlgorithm(const ParametersToOptimize &parameters, bool has_gradient, bool has_constraints);
};

#endif
