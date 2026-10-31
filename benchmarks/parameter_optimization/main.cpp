/** \file "main.cpp" Benchmark driver for the Parameter Optimization plug-in's global optimizers.

Compares the TPE port against Optuna (see run_optuna.py) and the plug-in's other derivative-free
optimizers. Problem definitions mirror problems.py; the "verify" mode checks they agree.

Usage:
  parameter_optimization_benchmark verify <problem_check.csv>
  parameter_optimization_benchmark quality <out.csv> [--optimizers TPE,Random,GA,CMAES,BO] [--seeds 30] [--budget 200] [--problems a,b]
  parameter_optimization_benchmark trajectory <out.csv> [--seeds 20] [--budget 200] [--problems a,b]
  parameter_optimization_benchmark overhead <out.csv>

Copyright (C) 2016-2025 Brian Bailey

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

SPDX-License-Identifier: LGPL-2.1-or-later

*/

#include "ParameterOptimization.h"
#include "TPESampler.h"
#include <chrono>
#include <numeric>

using namespace helios;

const double PI = 3.14159265358979323846;

struct ParameterSpec {
    std::string name;
    ParameterType type = ParameterType::FLOAT;
    float low = 0.f;
    float high = 0.f;
    std::vector<float> categories;
};

struct Problem {
    std::string name;
    std::vector<ParameterSpec> specs; //!< Sorted by name
    std::function<double(const std::vector<double> &)> function;
    double f_star = 0.0;
    double noise_sd = 0.0;
};

// An off-center optimum location: the prior kernel of TPE is centered, as is the initial mean of CMA-ES
static double shift(size_t index, double low, double high) {
    const double value = 0.31 + 0.6180339887 * static_cast<double>(index);
    const double fraction = value - std::floor(value);
    return low + (high - low) * (0.15 + 0.7 * fraction);
}

static std::vector<ParameterSpec> floatSpecs(size_t dimension, float low, float high) {
    std::vector<ParameterSpec> specs(dimension);
    for (size_t i = 0; i < dimension; i++) {
        specs[i].name = std::string("x") + (i < 10 ? "0" : "") + std::to_string(i);
        specs[i].low = low;
        specs[i].high = high;
    }
    return specs;
}

static double rosenbrock(const std::vector<double> &x) {
    double sum = 0.0;
    for (size_t i = 0; i + 1 < x.size(); i++) {
        sum += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1.0 - x[i]) * (1.0 - x[i]);
    }
    return sum;
}

static double hartmann6(const std::vector<double> &x) {
    static const double alpha[4] = {1.0, 1.2, 3.0, 3.2};
    static const double A[4][6] = {{10, 3, 17, 3.5, 1.7, 8}, {0.05, 10, 17, 0.1, 8, 14}, {3, 3.5, 1.7, 10, 17, 8}, {17, 8, 0.05, 10, 0.1, 14}};
    static const double P[4][6] = {{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886}, {0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991}, {0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650}, {0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}};
    double sum = 0.0;
    for (int i = 0; i < 4; i++) {
        double exponent = 0.0;
        for (int j = 0; j < 6; j++) {
            exponent += A[i][j] * (x[j] - P[i][j]) * (x[j] - P[i][j]);
        }
        sum += alpha[i] * std::exp(-exponent);
    }
    return -sum;
}

static double categoryCost(size_t dimension, size_t category) {
    const double value = static_cast<double>((dimension + 1) * (category + 1)) * 0.7548776662;
    return 3.0 * (value - std::floor(value));
}

static double categoricalPart(size_t c0, size_t c1, size_t c2, size_t c3) {
    double sum = categoryCost(0, c0) + categoryCost(1, c1) + categoryCost(2, c2) + categoryCost(3, c3);
    sum += (c0 + c1) % 5 == 2 ? 0.0 : 2.0;
    sum += (c2 + c3) % 5 == 1 ? 0.0 : 2.0;
    return sum;
}

static std::vector<Problem> buildProblems() {
    std::vector<Problem> problems;

    for (size_t d: {2, 5, 10}) {
        problems.push_back({"sphere_" + std::to_string(d), floatSpecs(d, -5.f, 5.f), [](const std::vector<double> &x) {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); i++) {
                sum += (x[i] - shift(i, -5.0, 5.0)) * (x[i] - shift(i, -5.0, 5.0));
            }
            return sum;
        }});
    }

    for (size_t d: {2, 5, 10}) {
        problems.push_back({"rastrigin_" + std::to_string(d), floatSpecs(d, -5.12f, 5.12f), [](const std::vector<double> &x) {
            double sum = 10.0 * static_cast<double>(x.size());
            for (size_t i = 0; i < x.size(); i++) {
                const double z = x[i] - shift(i, -5.12, 5.12);
                sum += z * z - 10.0 * std::cos(2.0 * PI * z);
            }
            return sum;
        }});
    }

    for (size_t d: {2, 5, 10}) {
        problems.push_back({"rosenbrock_" + std::to_string(d), floatSpecs(d, -2.048f, 2.048f), rosenbrock});
    }

    for (size_t d: {5, 10}) {
        problems.push_back({"ellipsoid_" + std::to_string(d), floatSpecs(d, -5.f, 5.f), [](const std::vector<double> &x) {
            const size_t dimension = x.size();
            std::vector<double> z(dimension);
            for (size_t i = 0; i < dimension; i++) {
                z[i] = x[i] - shift(i, -5.0, 5.0);
            }
            // Fixed sequence of Givens rotations, so that the conditioning is not aligned with the axes
            for (int pass = 0; pass < 2; pass++) {
                for (size_t i = 0; i + 1 < dimension; i++) {
                    const double angle = 0.7 + 0.3 * static_cast<double>(i) + 0.5 * pass;
                    const double rotated = std::cos(angle) * z[i] - std::sin(angle) * z[i + 1];
                    z[i + 1] = std::sin(angle) * z[i] + std::cos(angle) * z[i + 1];
                    z[i] = rotated;
                }
            }
            double sum = 0.0;
            for (size_t i = 0; i < dimension; i++) {
                sum += std::pow(10.0, 3.0 * static_cast<double>(i) / static_cast<double>(dimension - 1)) * z[i] * z[i];
            }
            return sum;
        }});
    }

    {
        std::vector<ParameterSpec> specs = floatSpecs(2, 0.f, 15.f);
        specs[0].low = -5.f;
        specs[0].high = 10.f;
        problems.push_back({"branin", specs, [](const std::vector<double> &x) {
            const double b = 5.1 / (4.0 * PI * PI);
            const double c = 5.0 / PI;
            const double t = 1.0 / (8.0 * PI);
            const double term = x[1] - b * x[0] * x[0] + c * x[0] - 6.0;
            return term * term + 10.0 * (1.0 - t) * std::cos(x[0]) + 10.0;
        }, 0.39788735772973836});
    }

    problems.push_back({"hartmann6", floatSpecs(6, 0.f, 1.f), hartmann6, -3.32236801141551});

    for (size_t d: {2, 10, 20}) {
        problems.push_back({"ackley_" + std::to_string(d), floatSpecs(d, -32.768f, 32.768f), [](const std::vector<double> &x) {
            double squares = 0.0;
            double cosines = 0.0;
            for (size_t i = 0; i < x.size(); i++) {
                const double z = x[i] - shift(i, -32.768, 32.768);
                squares += z * z;
                cosines += std::cos(2.0 * PI * z);
            }
            const double n = static_cast<double>(x.size());
            return -20.0 * std::exp(-0.2 * std::sqrt(squares / n)) - std::exp(cosines / n) + 20.0 + std::exp(1.0);
        }});
    }

    // Stochastic objectives such as ray tracing. Regret is scored on the noiseless function.
    problems.push_back({"noisy_hartmann6", floatSpecs(6, 0.f, 1.f), hartmann6, -3.32236801141551, 0.1});
    problems.push_back({"noisy_rosenbrock_5", floatSpecs(5, -2.048f, 2.048f), rosenbrock, 0.0, 5.0});

    {
        std::vector<ParameterSpec> specs;
        for (int i = 0; i < 3; i++) {
            specs.push_back({"n" + std::to_string(i), ParameterType::INTEGER, -5.f, 10.f, {}});
        }
        for (int i = 0; i < 3; i++) {
            specs.push_back({"x" + std::to_string(i), ParameterType::FLOAT, -5.f, 10.f, {}});
        }
        problems.push_back({"mixed_int_float", specs, [](const std::vector<double> &x) {
            static const double integer_centers[3] = {7.0, -2.0, 3.0};
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); i++) {
                const double z = x[i] - (i < 3 ? integer_centers[i] : shift(i, -5.0, 10.0));
                sum += z * z + 5.0 * (1.0 - std::cos(2.0 * PI * z / 5.0));
            }
            return sum;
        }});
    }

    {
        std::vector<ParameterSpec> specs;
        for (int i = 0; i < 4; i++) {
            specs.push_back({"c" + std::to_string(i), ParameterType::CATEGORICAL, 0.f, 0.f, {0.f, 1.f, 2.f, 3.f, 4.f}});
        }
        for (int i = 0; i < 2; i++) {
            specs.push_back({"x" + std::to_string(i), ParameterType::FLOAT, -2.f, 3.f, {}});
        }
        double f_star = std::numeric_limits<double>::infinity();
        for (size_t c0 = 0; c0 < 5; c0++) {
            for (size_t c1 = 0; c1 < 5; c1++) {
                for (size_t c2 = 0; c2 < 5; c2++) {
                    for (size_t c3 = 0; c3 < 5; c3++) {
                        f_star = std::min(f_star, categoricalPart(c0, c1, c2, c3));
                    }
                }
            }
        }
        problems.push_back({"categorical_float", specs, [](const std::vector<double> &x) {
            double sum = categoricalPart(static_cast<size_t>(x[0]), static_cast<size_t>(x[1]), static_cast<size_t>(x[2]), static_cast<size_t>(x[3]));
            for (size_t i = 4; i < x.size(); i++) {
                sum += (x[i] - shift(i, -2.0, 3.0)) * (x[i] - shift(i, -2.0, 3.0));
            }
            return sum;
        }, f_star});
    }

    return problems;
}

static bool isDiscrete(const Problem &problem) {
    return std::any_of(problem.specs.begin(), problem.specs.end(), [](const ParameterSpec &spec) { return spec.type != ParameterType::FLOAT; });
}

static ParametersToOptimize toParameters(const Problem &problem) {
    ParametersToOptimize parameters;
    for (const ParameterSpec &spec: problem.specs) {
        ParameterToOptimize parameter;
        parameter.type = spec.type;
        parameter.min = spec.low;
        parameter.max = spec.high;
        parameter.categories = spec.categories;
        parameter.value = spec.type == ParameterType::CATEGORICAL ? spec.categories.front() : 0.5f * (spec.low + spec.high);
        parameters[spec.name] = parameter;
    }
    return parameters;
}

struct BudgetReached {};

static void configure(ParameterOptimization &optimizer, const std::string &name, size_t budget, unsigned int seed) {
    if (name == "TPE" || name == "Random") {
        TPE tpe;
        tpe.max_evaluations = budget;
        tpe.random_seed = seed;
        if (name == "Random") {
            tpe.n_startup_trials = budget; // Never leaves the startup phase, so this is random search
        }
        optimizer.setAlgorithm(tpe);
    } else if (name == "BO") {
        BayesianOptimization bo;
        bo.max_evaluations = budget;
        bo.random_seed = seed;
        optimizer.setAlgorithm(bo);
    } else if (name == "CMAES") {
        CMAES cmaes;
        cmaes.max_evaluations = budget;
        cmaes.random_seed = seed;
        optimizer.setAlgorithm(cmaes);
    } else if (name == "GA") {
        // GA caches repeated individuals, so it is run long enough to spend the budget, then stopped by BudgetReached
        GeneticAlgorithm ga;
        ga.generations = 10 * budget / ga.population_size + 1;
        ga.random_seed = seed;
        optimizer.setAlgorithm(ga);
    } else {
        helios_runtime_error("ERROR: Unknown optimizer '" + name + "'. Choose from TPE, Random, BO, CMAES, GA.");
    }
}

static std::vector<std::string> splitList(const std::string &text) {
    std::vector<std::string> items;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        items.push_back(item);
    }
    return items;
}

static int runVerify(const std::vector<Problem> &problems, const std::string &check_file) {
    std::ifstream input(check_file);
    if (!input.is_open()) {
        helios_runtime_error("ERROR: Could not open problem check table '" + check_file + "'.");
    }
    std::string line;
    std::getline(input, line); // Header: problem,f_star,f,x0,x1,...
    size_t n_checked = 0;
    int n_mismatched = 0;
    while (std::getline(input, line)) {
        const std::vector<std::string> fields = splitList(line);
        const auto problem = std::find_if(problems.begin(), problems.end(), [&fields](const Problem &candidate) { return candidate.name == fields[0]; });
        if (problem == problems.end()) {
            helios_runtime_error("ERROR: Problem '" + fields[0] + "' from the check table is not defined in main.cpp.");
        }
        std::vector<double> x;
        for (size_t i = 3; i < fields.size(); i++) {
            x.push_back(std::stod(fields[i]));
        }
        const double f_expected = std::stod(fields[2]);
        const double f = problem->function(x);
        const double f_star_expected = std::stod(fields[1]);
        if (std::abs(f - f_expected) > 1e-9 * std::max(1.0, std::abs(f_expected)) || std::abs(problem->f_star - f_star_expected) > 1e-9 * std::max(1.0, std::abs(f_star_expected))) {
            std::cout << "MISMATCH " << fields[0] << ": C++ f=" << f << " f_star=" << problem->f_star << ", Python f=" << f_expected << " f_star=" << f_star_expected << std::endl;
            n_mismatched++;
        }
        n_checked++;
    }
    std::cout << "Checked " << n_checked << " points: " << n_mismatched << " mismatches" << std::endl;
    return n_mismatched == 0 && n_checked > 0 ? 0 : 1;
}

static void runQuality(const std::vector<Problem> &problems, const std::string &output_file, const std::vector<std::string> &optimizers, int n_seeds, size_t budget) {
    std::ofstream output(output_file);
    output << "optimizer,problem,seed,evaluation,best_regret\n";
    std::ofstream timing(output_file + ".timing.csv");
    timing << "optimizer,problem,seed,evaluations,total_s,objective_s\n";

    for (const Problem &problem: problems) {
        for (const std::string &optimizer_name: optimizers) {
            if (isDiscrete(problem) && (optimizer_name == "BO" || optimizer_name == "CMAES")) {
                continue; // These reject INTEGER and CATEGORICAL parameters
            }
            const auto problem_start = std::chrono::steady_clock::now();
            for (int seed = 1; seed <= n_seeds; seed++) {
                helios_tpe::UniformStream noise_stream(static_cast<unsigned int>(seed) * 1000003u);
                size_t evaluation = 0;
                double best_observed = std::numeric_limits<double>::infinity();
                double true_value_at_best_observed = std::numeric_limits<double>::infinity();
                double objective_seconds = 0.0;

                auto objective = [&](const ParametersToOptimize &p) -> float {
                    const auto objective_start = std::chrono::steady_clock::now();
                    std::vector<double> x(problem.specs.size());
                    for (size_t i = 0; i < x.size(); i++) {
                        const ParameterSpec &spec = problem.specs[i];
                        const float value = p.at(spec.name).value;
                        x[i] = spec.type == ParameterType::CATEGORICAL ? static_cast<double>(std::find(spec.categories.begin(), spec.categories.end(), value) - spec.categories.begin()) : static_cast<double>(value);
                    }
                    const double f_true = problem.function(x);
                    double noisy = f_true;
                    if (problem.noise_sd > 0.0) {
                        // Box-Muller, written out so that it does not depend on the standard library
                        noisy += problem.noise_sd * std::sqrt(-2.0 * std::log(1.0 - noise_stream.next())) * std::cos(2.0 * PI * noise_stream.next());
                    }
                    const float observed = static_cast<float>(noisy);

                    // Regret is that of the point the optimizer would return: the one with the best noisy value, scored on the noiseless function
                    if (observed < best_observed) {
                        best_observed = observed;
                        true_value_at_best_observed = f_true;
                    }
                    evaluation++;
                    if (evaluation == 10 || evaluation == 25 || evaluation % 50 == 0 || evaluation == budget) {
                        output << optimizer_name << "," << problem.name << "," << seed << "," << evaluation << "," << std::setprecision(17) << true_value_at_best_observed - problem.f_star << "\n";
                    }
                    objective_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - objective_start).count();
                    if (evaluation >= budget) {
                        throw BudgetReached{};
                    }
                    return observed;
                };

                ParameterOptimization optimizer;
                configure(optimizer, optimizer_name, budget, static_cast<unsigned int>(seed));
                const auto start = std::chrono::steady_clock::now();
                try {
                    static_cast<void>(optimizer.run(objective, toParameters(problem)));
                } catch (const BudgetReached &) {
                }
                const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                timing << optimizer_name << "," << problem.name << "," << seed << "," << evaluation << "," << total_seconds << "," << objective_seconds << "\n";
            }
            output.flush();
            timing.flush();
            std::cout << optimizer_name << " " << problem.name << ": " << std::chrono::duration<double>(std::chrono::steady_clock::now() - problem_start).count() << " s" << std::endl;
        }
    }
}

static void runTrajectory(const std::vector<Problem> &problems, const std::string &output_file, int n_seeds, size_t budget) {
    std::ofstream output(output_file);
    output << "problem,seed,evaluation,dimension,value\n";
    output << std::setprecision(9);
    for (const Problem &problem: problems) {
        if (problem.noise_sd > 0.0) {
            continue; // The noise streams of the two implementations are independent
        }
        for (int seed = 1; seed <= n_seeds; seed++) {
            size_t evaluation = 0;
            auto objective = [&](const ParametersToOptimize &p) -> float {
                std::vector<double> x(problem.specs.size());
                evaluation++;
                for (size_t i = 0; i < x.size(); i++) {
                    x[i] = p.at(problem.specs[i].name).value;
                    output << problem.name << "," << seed << "," << evaluation << "," << i << "," << x[i] << "\n";
                }
                return static_cast<float>(problem.function(x));
            };
            TPE tpe;
            tpe.max_evaluations = budget;
            tpe.random_seed = static_cast<unsigned int>(seed);
            ParameterOptimization optimizer;
            optimizer.setAlgorithm(tpe);
            static_cast<void>(optimizer.run(objective, toParameters(problem)));
        }
    }
}

static std::vector<helios_tpe::SearchDimension> overheadSpace(const std::string &space, size_t dimension) {
    std::vector<helios_tpe::SearchDimension> dimensions(dimension);
    for (size_t i = 0; i < dimension; i++) {
        if (space == "mixed" && i % 3 == 1) {
            dimensions[i] = {helios_tpe::DimensionKind::INTEGER, 0.0, 20.0, 0};
        } else if (space == "mixed" && i % 3 == 2) {
            dimensions[i] = {helios_tpe::DimensionKind::CATEGORICAL, 0.0, 0.0, 5};
        } else {
            dimensions[i] = {helios_tpe::DimensionKind::CONTINUOUS, 0.0, 1.0, 0};
        }
    }
    return dimensions;
}

static void runOverhead(const std::string &output_file) {
    std::ofstream output(output_file);
    output << "optimizer,space,n_history,dimension,ms_per_suggestion\n";

    const int n_timed = 20;
    for (const std::string space: {"continuous", "mixed"}) {
        for (size_t dimension: {2, 10, 50}) {
            for (size_t n_history: {100, 1000, 5000}) {
                const std::vector<helios_tpe::SearchDimension> dimensions = overheadSpace(space, dimension);
                helios_tpe::TPESampler sampler(dimensions, 1, 24, 1);
                helios_tpe::UniformStream stream(2);
                for (size_t i = 0; i < n_history; i++) {
                    std::vector<double> point(dimension);
                    for (size_t d = 0; d < dimension; d++) {
                        const helios_tpe::SearchDimension &dim = dimensions[d];
                        if (dim.kind == helios_tpe::DimensionKind::CATEGORICAL) {
                            point[d] = std::floor(stream.next() * static_cast<double>(dim.n_choices));
                        } else if (dim.kind == helios_tpe::DimensionKind::INTEGER) {
                            point[d] = std::floor(stream.next() * (dim.high - dim.low + 1.0)) + dim.low;
                        } else {
                            point[d] = dim.low + stream.next() * (dim.high - dim.low);
                        }
                    }
                    sampler.tell(point, stream.next());
                }
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < n_timed; i++) {
                    const std::vector<double> point = sampler.ask();
                    sampler.tell(point, stream.next());
                }
                const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / n_timed;
                output << "TPE," << space << "," << n_history << "," << dimension << "," << milliseconds << "\n";
                output.flush();
                std::cout << "TPE " << space << " N=" << n_history << " d=" << dimension << ": " << milliseconds << " ms/suggestion" << std::endl;
            }
        }
    }

    // BO through the public API, timing the gap between objective calls over the last evaluations
    for (size_t dimension: {2, 10}) {
        for (size_t n_history: {100, 200, 400}) {
            ParametersToOptimize parameters;
            for (const ParameterSpec &spec: floatSpecs(dimension, 0.f, 1.f)) {
                parameters[spec.name] = {0.5f, spec.low, spec.high};
            }
            helios_tpe::UniformStream stream(2);
            size_t evaluation = 0;
            std::chrono::steady_clock::time_point window_start;
            auto objective = [&](const ParametersToOptimize &) -> float {
                evaluation++;
                if (evaluation == n_history - static_cast<size_t>(n_timed) + 1) {
                    window_start = std::chrono::steady_clock::now();
                }
                return static_cast<float>(stream.next());
            };
            BayesianOptimization bo;
            bo.max_evaluations = n_history + 1;
            bo.random_seed = 1;
            bo.max_gp_samples = n_history + 1; // Otherwise history above the default cap of 200 is discarded, hiding the cost of more data
            ParameterOptimization optimizer;
            optimizer.setAlgorithm(bo);
            static_cast<void>(optimizer.run(objective, parameters));
            const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - window_start).count() / n_timed;
            output << "BO,continuous," << n_history << "," << dimension << "," << milliseconds << "\n";
            output.flush();
            std::cout << "BO N=" << n_history << " d=" << dimension << ": " << milliseconds << " ms/suggestion" << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " verify|quality|trajectory|overhead <file> [--optimizers a,b] [--seeds n] [--budget n] [--problems a,b]" << std::endl;
        return 1;
    }
    const std::string mode = argv[1];
    const std::string file = argv[2];

    std::vector<std::string> optimizers = {"TPE", "Random", "GA", "CMAES", "BO"};
    int n_seeds = mode == "trajectory" ? 20 : 30;
    size_t budget = 200;
    std::vector<std::string> selected_problems;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--optimizers") {
            optimizers = splitList(argv[i + 1]);
        } else if (option == "--seeds") {
            n_seeds = std::stoi(argv[i + 1]);
        } else if (option == "--budget") {
            budget = static_cast<size_t>(std::stoul(argv[i + 1]));
        } else if (option == "--problems") {
            selected_problems = splitList(argv[i + 1]);
        } else {
            helios_runtime_error("ERROR: Unknown option '" + option + "'.");
        }
    }

    std::vector<Problem> problems = buildProblems();
    if (!selected_problems.empty()) {
        problems.erase(std::remove_if(problems.begin(), problems.end(), [&selected_problems](const Problem &problem) { return std::find(selected_problems.begin(), selected_problems.end(), problem.name) == selected_problems.end(); }), problems.end());
        if (problems.size() != selected_problems.size()) {
            helios_runtime_error("ERROR: --problems names a problem that does not exist.");
        }
    }

    if (mode == "verify") {
        return runVerify(problems, file);
    }
    if (mode == "quality") {
        runQuality(problems, file, optimizers, n_seeds, budget);
    } else if (mode == "trajectory") {
        runTrajectory(problems, file, n_seeds, budget);
    } else if (mode == "overhead") {
        runOverhead(file);
    } else {
        helios_runtime_error("ERROR: Unknown mode '" + mode + "'.");
    }
    return 0;
}
