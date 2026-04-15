/**
 * \file "selfTest.cpp" Automated tests for the Parameter Optimization plug-in.
 *
 * Copyright (C) 2016-2025 Brian Bailey
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "ParameterOptimization.h"
#include <iomanip>

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "doctest_utils.h"

using namespace helios;

// Retry helper for stochastic tests. Runs `attempt` up to `max_attempts` times.
// Returns true on first success. If all attempts fail, returns false and `attempt`
// will have been called `max_attempts` times — the caller's captured result holds
// the last attempt's values, so DOCTEST_CHECK assertions report useful diagnostics.
static bool retry(int max_attempts, const std::function<bool()> &attempt) {
    for (int i = 0; i < max_attempts; ++i) {
        if (attempt())
            return true;
    }
    return false;
}

int ParameterOptimization::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

// Test 1: GA simple quadratic convergence
// f(x,y) = (x-3)^2 + (y+1)^2, minimum at (3, -1) with f=0
DOCTEST_TEST_CASE("ParameterOptimization quadratic") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 50;
    ga.population_size = 50;
    popt.setAlgorithm(ga);

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));

    DOCTEST_CHECK(result.fitness < 1e-2f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.05f));
}

// Test 2: GA quadratic with Helios primitive data round-trip
DOCTEST_TEST_CASE("ParameterOptimization quadratic with primitive data access") {

    Context context;
    uint UUID = context.addPatch();

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    auto sim = [&](const ParametersToOptimize &p) {
        for (const auto &[name, parameter]: p) {
            context.setPrimitiveData(UUID, name.c_str(), parameter.value);
        }

        float x, y;
        context.getPrimitiveData(UUID, "x", x);
        context.getPrimitiveData(UUID, "y", y);

        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 50;
    ga.population_size = 50;
    popt.setAlgorithm(ga);

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));

    DOCTEST_CHECK(result.fitness < 1e-2f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.05f));
}

// Test 3: GA multi-scale parameter search (4D, targets at 5, 100, 1000, 7)
DOCTEST_TEST_CASE("ParameterOptimization model parameter search") {

    auto sim = [&](const ParametersToOptimize &p) {
        float error = 0.0f;

        error += powi(p.at("Em_BMF").value - 5.f, 2);
        error += powi(p.at("i0_BMF").value - 100.f, 2);
        error += powi(p.at("k_BMF").value - 1000.f, 2);
        error += powi(p.at("b_BMF").value - 7.f, 2);

        return error;
    };

    ParameterOptimization popt;
    ParametersToOptimize params = {{"Em_BMF", {10, 0, 100}}, {"i0_BMF", {10, 0, 1000}}, {"k_BMF", {1e5, 0, 20000}}, {"b_BMF", {0.5, 0, 100}}};

    GeneticAlgorithm ga;
    ga.population_size = 200;
    ga.generations = 125;
    ga.crossover_rate = 0.5;
    popt.setAlgorithm(ga);

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));

    DOCTEST_CHECK(result.fitness < 1e-3f);
    DOCTEST_CHECK(result.parameters.at("Em_BMF").value == doctest::Approx(5.f).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("b_BMF").value == doctest::Approx(7.f).epsilon(0.05f));
}

// Test 4: GA fitness caching reduces simulation calls
DOCTEST_TEST_CASE("ParameterOptimization fitness caching") {
    size_t evaluation_count = 0;

    auto sim = [&](const ParametersToOptimize &p) {
        evaluation_count++;
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 10;
    ga.population_size = 20;
    popt.setAlgorithm(ga);

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));

    // With caching, repeated evaluations of the same individual are avoided.
    // Pre-compute of pop_fitness each generation are cache hits for existing members.
    // Elite individuals carried forward are also cache hits.
    // Real evaluations: initial pop (20) + ~19 new offspring per generation * 10 gens ≈ 210
    // Without caching, evaluatePopulation + pre-compute would double-count everything.
    size_t max_expected = 250;
    size_t min_expected = 100;

    DOCTEST_INFO("Actual evaluations: ", evaluation_count, " (informational: caching effectiveness test)");
    DOCTEST_CHECK(evaluation_count >= min_expected);
    DOCTEST_CHECK(evaluation_count <= max_expected);
}

// Test 5: Caching with more expensive function still converges
DOCTEST_TEST_CASE("ParameterOptimization caching with expensive function") {
    size_t evaluation_count = 0;
    const size_t expensive_iterations = 1000;

    auto expensive_sim = [&](const ParametersToOptimize &p) {
        evaluation_count++;

        // Simulate moderately expensive computation
        float result = 0.0f;
        for (size_t i = 0; i < expensive_iterations; ++i) {
            result += std::sin(static_cast<float>(i) * 0.01f);
        }

        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 2.f) * (x - 2.f) + (y - 1.f) * (y - 1.f) + result * 1e-6f;
    };

    ParametersToOptimize params = {{"x", {0.f, -10.f, 10.f}}, {"y", {0.f, -10.f, 10.f}}};

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 10;
    ga.population_size = 20;
    popt.setAlgorithm(ga);

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(expensive_sim, params));

    DOCTEST_INFO("Expensive function evaluations: ", evaluation_count, " (informational: caching test)");
    DOCTEST_CHECK(evaluation_count <= 250);
    DOCTEST_CHECK(result.fitness < 2.0f);
}

// Test 6: Bayesian Optimization quadratic convergence
DOCTEST_TEST_CASE("ParameterOptimization Bayesian Optimization quadratic") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    ParameterOptimization popt;
    BayesianOptimization bo;
    bo.max_evaluations = 80;
    bo.initial_samples = 15;
    bo.acquisition_samples = 2000;
    popt.setAlgorithm(bo);

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));

    // With corrected normalization, BO should converge well on a smooth quadratic.
    // Threshold 0.1: P(random search passes in 80 evals on [-5,5]^2) < 22%.
    // BO with GP guidance should reliably beat this.
    DOCTEST_CHECK(result.fitness < 0.1f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.15f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.15f));
}

// Test 7: BO vs GA on same problem with comparable evaluation budgets
DOCTEST_TEST_CASE("ParameterOptimization BO vs GA") {
    auto sim = [](const ParametersToOptimize &p) {
        float error = 0.0f;
        error += powi(p.at("a").value - 5.f, 2);
        error += powi(p.at("b").value - 10.f, 2);
        return error;
    };

    ParametersToOptimize params = {{"a", {0, 0, 20}}, {"b", {0, 0, 20}}};

    // GA: 50 generations * 20 pop ≈ 1000 evaluations (pop-based, needs more)
    ParameterOptimization popt_ga;
    GeneticAlgorithm ga;
    ga.generations = 50;
    ga.population_size = 20;
    popt_ga.setAlgorithm(ga);

    ParameterOptimization::Result ga_result;
    DOCTEST_CHECK_NOTHROW(ga_result = popt_ga.run(sim, params));

    // BO: 60 total evaluations (sample-efficient by design)
    ParameterOptimization popt_bo;
    BayesianOptimization bo;
    bo.max_evaluations = 60;
    bo.initial_samples = 10;
    bo.acquisition_samples = 2000;
    popt_bo.setAlgorithm(bo);

    ParameterOptimization::Result bo_result;
    DOCTEST_CHECK_NOTHROW(bo_result = popt_bo.run(sim, params));

    DOCTEST_INFO("GA fitness: ", ga_result.fitness, " (optimum: 0 at a=5, b=10)");
    DOCTEST_INFO("BO fitness: ", bo_result.fitness, " (optimum: 0 at a=5, b=10)");

    // Both should converge well on this simple 2D problem
    DOCTEST_CHECK(ga_result.fitness < 1.0f);
    DOCTEST_CHECK(bo_result.fitness < 1.0f);
}

// Test 8: Backward compatibility with deprecated 'generations' parameter
DOCTEST_TEST_CASE("ParameterOptimization backward compatibility") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        return (x - 2.f) * (x - 2.f);
    };

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 30;
    ga.population_size = 20;
    popt.setAlgorithm(ga);

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));

    DOCTEST_CHECK(result.fitness < 0.1f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(2.f).epsilon(0.1f));
}

// Test 9: CMA-ES simple quadratic (tight convergence expected)
DOCTEST_TEST_CASE("ParameterOptimization CMA-ES quadratic") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    ParameterOptimization popt;
    CMAES cmaes;
    cmaes.max_evaluations = 200;
    cmaes.sigma = 0.3f;
    popt.setAlgorithm(cmaes);

    auto start = std::chrono::high_resolution_clock::now();
    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    DOCTEST_INFO("CMA-ES quadratic time: ", duration.count(), " ms", " | fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 3)", " y=", result.parameters.at("y").value, " (opt: -1)");

    DOCTEST_CHECK(result.fitness < 1e-3f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.05f));
}

// Test 10: CMA-ES Rosenbrock function (classic benchmark)
// f(x,y) = (1-x)^2 + 100*(y-x^2)^2, global minimum at (1,1) with f=0
DOCTEST_TEST_CASE("ParameterOptimization CMA-ES Rosenbrock") {
    ParametersToOptimize params = {{"x", {0.f, -2.f, 2.f}}, {"y", {0.f, -2.f, 2.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float term1 = (1.f - x);
        float term2 = (y - x * x);
        return term1 * term1 + 100.f * term2 * term2;
    };

    ParameterOptimization::Result result;
    retry(3, [&]() {
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 500;
        cmaes.sigma = 0.5f;
        popt.setAlgorithm(cmaes);
        result = popt.run(sim, params);
        return result.fitness < 0.5f && std::abs(result.parameters.at("x").value - 1.f) < 0.2f && std::abs(result.parameters.at("y").value - 1.f) < 0.2f;
    });

    DOCTEST_INFO("CMA-ES Rosenbrock: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 1)", " y=", result.parameters.at("y").value, " (opt: 1)");

    DOCTEST_CHECK(result.fitness < 0.5f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(1.f).epsilon(0.2f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(1.f).epsilon(0.2f));
}

// Test 11: CMA-ES higher-dimensional optimization (4D sphere)
// f(x) = sum(x_i^2), minimum at origin with f=0
DOCTEST_TEST_CASE("ParameterOptimization CMA-ES 4D sphere") {
    ParametersToOptimize params = {{"x1", {0.5f, -5.f, 5.f}}, {"x2", {-0.3f, -5.f, 5.f}}, {"x3", {0.8f, -5.f, 5.f}}, {"x4", {-0.6f, -5.f, 5.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float sum = 0.f;
        sum += p.at("x1").value * p.at("x1").value;
        sum += p.at("x2").value * p.at("x2").value;
        sum += p.at("x3").value * p.at("x3").value;
        sum += p.at("x4").value * p.at("x4").value;
        return sum;
    };

    ParameterOptimization::Result result;
    retry(3, [&]() {
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 250;
        cmaes.sigma = 0.3f;
        popt.setAlgorithm(cmaes);
        result = popt.run(sim, params);
        return result.fitness < 1e-2f;
    });

    DOCTEST_CHECK(result.fitness < 1e-2f);
    DOCTEST_CHECK(result.parameters.at("x1").value == doctest::Approx(0.f).epsilon(0.1f));
    DOCTEST_CHECK(result.parameters.at("x2").value == doctest::Approx(0.f).epsilon(0.1f));
    DOCTEST_CHECK(result.parameters.at("x3").value == doctest::Approx(0.f).epsilon(0.1f));
    DOCTEST_CHECK(result.parameters.at("x4").value == doctest::Approx(0.f).epsilon(0.1f));
}

// Test 12: Three-way comparison GA vs BO vs CMA-ES on 3D quadratic
DOCTEST_TEST_CASE("ParameterOptimization GA vs BO vs CMA-ES comparison") {
    auto sim = [](const ParametersToOptimize &p) {
        float error = 0.0f;
        error += powi(p.at("a").value - 5.f, 2);
        error += powi(p.at("b").value - 10.f, 2);
        error += powi(p.at("c").value - 2.f, 2);
        return error;
    };

    ParametersToOptimize params = {{"a", {0, 0, 20}}, {"b", {0, 0, 20}}, {"c", {0, 0, 10}}};

    // GA: pop-based method, needs more evaluations
    ParameterOptimization popt_ga;
    GeneticAlgorithm ga;
    ga.generations = 60;
    ga.population_size = 30;
    popt_ga.setAlgorithm(ga);

    auto start_ga = std::chrono::high_resolution_clock::now();
    ParameterOptimization::Result ga_result;
    DOCTEST_CHECK_NOTHROW(ga_result = popt_ga.run(sim, params));
    auto end_ga = std::chrono::high_resolution_clock::now();
    auto duration_ga = std::chrono::duration_cast<std::chrono::milliseconds>(end_ga - start_ga);

    // BO: sample-efficient, fewer evaluations needed
    ParameterOptimization popt_bo;
    BayesianOptimization bo;
    bo.max_evaluations = 80;
    bo.initial_samples = 15;
    bo.acquisition_samples = 2000;
    popt_bo.setAlgorithm(bo);

    auto start_bo = std::chrono::high_resolution_clock::now();
    ParameterOptimization::Result bo_result;
    DOCTEST_CHECK_NOTHROW(bo_result = popt_bo.run(sim, params));
    auto end_bo = std::chrono::high_resolution_clock::now();
    auto duration_bo = std::chrono::duration_cast<std::chrono::milliseconds>(end_bo - start_bo);

    // CMA-ES: covariance-adaptive, moderate evaluations
    ParameterOptimization popt_cmaes;
    CMAES cmaes;
    cmaes.max_evaluations = 200;
    cmaes.sigma = 0.3f;
    popt_cmaes.setAlgorithm(cmaes);

    auto start_cmaes = std::chrono::high_resolution_clock::now();
    ParameterOptimization::Result cmaes_result;
    DOCTEST_CHECK_NOTHROW(cmaes_result = popt_cmaes.run(sim, params));
    auto end_cmaes = std::chrono::high_resolution_clock::now();
    auto duration_cmaes = std::chrono::duration_cast<std::chrono::milliseconds>(end_cmaes - start_cmaes);

    DOCTEST_INFO("GA fitness: ", ga_result.fitness, " (optimum: 0 at a=5, b=10, c=2) | time: ", duration_ga.count(), " ms");
    DOCTEST_INFO("BO fitness: ", bo_result.fitness, " (optimum: 0 at a=5, b=10, c=2) | time: ", duration_bo.count(), " ms");
    DOCTEST_INFO("CMA-ES fitness: ", cmaes_result.fitness, " (optimum: 0 at a=5, b=10, c=2) | time: ", duration_cmaes.count(), " ms");

    // All three should converge well on a smooth 3D quadratic
    DOCTEST_CHECK(ga_result.fitness < 1.0f);
    DOCTEST_CHECK(bo_result.fitness < 2.0f);
    DOCTEST_CHECK(cmaes_result.fitness < 0.5f);
}

// Test 13: CMA-ES on Rastrigin (multimodal stress test)
// f(x,y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
// Global minimum at (0,0) with f=0, many local minima
DOCTEST_TEST_CASE("ParameterOptimization CMA-ES multimodal") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        const float A = 10.f;
        const float pi = 3.14159265359f;
        return A * 2.f + (x * x - A * std::cos(2.f * pi * x)) + (y * y - A * std::cos(2.f * pi * y));
    };

    ParametersToOptimize params = {{"x", {0.2f, -5.f, 5.f}}, {"y", {0.3f, -5.f, 5.f}}};

    // Rastrigin is hard; require finding at least a good local minimum
    // Global min is 0, local minima at integer lattice points give f ~ 1-2 per dimension
    ParameterOptimization::Result result;
    retry(4, [&]() {
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 300;
        cmaes.sigma = 0.5f;
        popt.setAlgorithm(cmaes);
        result = popt.run(sim, params);
        return result.fitness < 5.0f;
    });

    DOCTEST_INFO("CMA-ES Rastrigin: fitness=", result.fitness, " (optimum: 0 at x=0, y=0)");
    DOCTEST_CHECK(result.fitness < 5.0f);
}

// Test 14: CMA-ES with automatic population sizing
DOCTEST_TEST_CASE("ParameterOptimization CMA-ES auto population") {
    ParametersToOptimize params = {{"x1", {0.f, -10.f, 10.f}}, {"x2", {0.f, -10.f, 10.f}}, {"x3", {0.f, -10.f, 10.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float sum = 0.f;
        for (const auto &[name, param]: p) {
            sum += (param.value - 5.f) * (param.value - 5.f);
        }
        return sum;
    };

    ParameterOptimization popt;
    CMAES cmaes;
    cmaes.max_evaluations = 150;
    cmaes.lambda = 0; // Automatic: 4 + floor(3*ln(3)) ~ 7
    cmaes.sigma = 0.3f;
    popt.setAlgorithm(cmaes);

    auto start = std::chrono::high_resolution_clock::now();
    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    DOCTEST_INFO("CMA-ES 3D sphere auto population: time=", duration.count(), " ms", " | fitness=", result.fitness, " (optimum: 0)", " | x1=", result.parameters.at("x1").value, " (opt: 5)", " x2=", result.parameters.at("x2").value, " (opt: 5)",
                 " x3=", result.parameters.at("x3").value, " (opt: 5)");

    DOCTEST_CHECK(result.fitness < 1.0f);
    DOCTEST_CHECK(result.parameters.at("x1").value == doctest::Approx(5.f).epsilon(0.2f));
    DOCTEST_CHECK(result.parameters.at("x2").value == doctest::Approx(5.f).epsilon(0.2f));
    DOCTEST_CHECK(result.parameters.at("x3").value == doctest::Approx(5.f).epsilon(0.2f));
}

// Test 15: Three-way scaling on 3D Rosenbrock
// f(x,y,z) = (1-x)^2 + 100*(y-x^2)^2 + (1-y)^2 + 100*(z-y^2)^2
// Global minimum at (1,1,1) with f=0
DOCTEST_TEST_CASE("ParameterOptimization 3D Rosenbrock scaling") {
    auto rosenbrock3d = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float z = p.at("z").value;

        float term1 = (1.0f - x);
        float term2 = (y - x * x);
        float term3 = (1.0f - y);
        float term4 = (z - y * y);

        return term1 * term1 + 100.0f * term2 * term2 + term3 * term3 + 100.0f * term4 * term4;
    };

    ParametersToOptimize params = {{"x", {0.f, -2.f, 2.f}}, {"y", {0.f, -2.f, 2.f}}, {"z", {0.f, -2.f, 2.f}}};

    // 3D Rosenbrock is challenging; each algorithm gets retries independently
    ParameterOptimization::Result ga_result;
    retry(3, [&]() {
        ParameterOptimization popt_ga;
        GeneticAlgorithm ga;
        ga.generations = 300;
        ga.population_size = 30;
        popt_ga.setAlgorithm(ga);
        ga_result = popt_ga.run(rosenbrock3d, params);
        return ga_result.fitness < 2.0f;
    });

    ParameterOptimization::Result bo_result;
    retry(3, [&]() {
        ParameterOptimization popt_bo;
        BayesianOptimization bo;
        bo.max_evaluations = 150;
        bo.initial_samples = 20;
        bo.acquisition_samples = 2000;
        popt_bo.setAlgorithm(bo);
        bo_result = popt_bo.run(rosenbrock3d, params);
        return bo_result.fitness < 2.0f;
    });

    ParameterOptimization::Result cmaes_result;
    retry(3, [&]() {
        ParameterOptimization popt_cmaes;
        CMAES cmaes;
        cmaes.max_evaluations = 500;
        cmaes.sigma = 0.5f;
        popt_cmaes.setAlgorithm(cmaes);
        cmaes_result = popt_cmaes.run(rosenbrock3d, params);
        return cmaes_result.fitness < 2.0f;
    });

    DOCTEST_INFO("GA:     fitness = ", ga_result.fitness, " (optimum: 0 at x=1, y=1, z=1)");
    DOCTEST_INFO("BO:     fitness = ", bo_result.fitness, " (optimum: 0 at x=1, y=1, z=1)");
    DOCTEST_INFO("CMA-ES: fitness = ", cmaes_result.fitness, " (optimum: 0 at x=1, y=1, z=1)");

    DOCTEST_CHECK(ga_result.fitness < 2.0f);
    DOCTEST_CHECK(bo_result.fitness < 2.0f);
    DOCTEST_CHECK(cmaes_result.fitness < 2.0f);
}

// =============================================================================
// Tests 16-21: Algorithmic correctness tests
// These tests verify that the optimization algorithms actually work correctly,
// not just that they converge on easy problems.
// =============================================================================

// Test 16: Result fitness consistency
// Verify that the returned fitness actually matches f(returned_parameters) for all methods.
// This catches bugs where the result struct reports stale/wrong fitness values.
DOCTEST_TEST_CASE("ParameterOptimization result fitness consistency") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return x * x + y * y;
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    // Test GA
    {
        ParameterOptimization popt;
        GeneticAlgorithm ga;
        ga.generations = 50;
        ga.population_size = 15;
        popt.setAlgorithm(ga);
        auto result = popt.run(sim, params);

        float recomputed = sim(result.parameters);
        DOCTEST_INFO("GA: reported=", result.fitness, " recomputed=", recomputed, " (meta-test: consistency check)");
        DOCTEST_CHECK(result.fitness == doctest::Approx(recomputed).epsilon(1e-5f));
    }

    // Test BO
    {
        ParameterOptimization popt;
        BayesianOptimization bo;
        bo.max_evaluations = 50;
        popt.setAlgorithm(bo);
        auto result = popt.run(sim, params);

        float recomputed = sim(result.parameters);
        DOCTEST_INFO("BO: reported=", result.fitness, " recomputed=", recomputed, " (meta-test: consistency check)");
        DOCTEST_CHECK(result.fitness == doctest::Approx(recomputed).epsilon(1e-5f));
    }

    // Test CMA-ES
    {
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 50;
        popt.setAlgorithm(cmaes);
        auto result = popt.run(sim, params);

        float recomputed = sim(result.parameters);
        DOCTEST_INFO("CMA-ES: reported=", result.fitness, " recomputed=", recomputed, " (meta-test: consistency check)");
        DOCTEST_CHECK(result.fitness == doctest::Approx(recomputed).epsilon(1e-5f));
    }
}

// Test 17: GA elitism actually preserves the best individual
// After optimization, the best fitness should be <= the best fitness found at any generation.
// We track this by wrapping the simulation to record the global minimum seen.
DOCTEST_TEST_CASE("ParameterOptimization GA elitism preserves best") {
    float global_min_seen = std::numeric_limits<float>::max();

    auto sim = [&global_min_seen](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float f = (x - 1.f) * (x - 1.f) + (y + 2.f) * (y + 2.f);
        if (f < global_min_seen) {
            global_min_seen = f;
        }
        return f;
    };

    ParametersToOptimize params = {{"x", {5.f, -10.f, 10.f}}, {"y", {5.f, -10.f, 10.f}}};

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 100;
    ga.population_size = 20;
    ga.elitism_rate = 0.1f;
    popt.setAlgorithm(ga);

    auto result = popt.run(sim, params);

    DOCTEST_INFO("GA returned fitness: ", result.fitness, " | global min ever evaluated: ", global_min_seen, " (meta-test: elitism check)");

    // The returned fitness must be the global minimum ever seen (elitism guarantee)
    DOCTEST_CHECK(result.fitness == doctest::Approx(global_min_seen).epsilon(1e-5f));
    // Also verify it actually converged reasonably
    DOCTEST_CHECK(result.fitness < 1.0f);
}

// Test 18: BO beats random search on a tight evaluation budget
// On a 3D quadratic with only 40 evaluations, BO should significantly outperform
// random search. We set a threshold that random search passes <1% of the time.
// For uniform random on [-5,5]^3, E[x^2+y^2+z^2] = 3*25/3 = 25, P(f<0.5) ~ 0.03%
DOCTEST_TEST_CASE("ParameterOptimization BO beats random search") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float z = p.at("z").value;
        return x * x + y * y + z * z;
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}, {"z", {2.f, -5.f, 5.f}}};

    ParameterOptimization::Result result;
    retry(3, [&]() {
        ParameterOptimization popt;
        BayesianOptimization bo;
        bo.max_evaluations = 40;
        bo.initial_samples = 10;
        bo.acquisition_samples = 1000;
        popt.setAlgorithm(bo);

        result = popt.run(sim, params);
        return result.fitness < 0.5f;
    });

    DOCTEST_INFO("BO 3D tight-budget: fitness=", result.fitness, " (optimum: 0 at x=0, y=0, z=0)");

    // Threshold: 0.5. P(best of 40 uniform random < 0.5) ~ 2% on [-5,5]^3
    // A working BO should reliably achieve this.
    DOCTEST_CHECK(result.fitness < 0.5f);
}

// Test 19: CMA-ES on rotated ellipsoid (covariance adaptation verification)
// f(x) = sum_i (10^(i/(n-1)) * z_i)^2 where z = R*x and R is a rotation matrix
// Without covariance adaptation, this is much harder than an axis-aligned ellipsoid.
// CMA-ES should adapt its covariance to the rotated coordinates.
DOCTEST_TEST_CASE("ParameterOptimization CMA-ES rotated ellipsoid") {
    // 45-degree rotation in 2D: z1 = (x+y)/sqrt(2), z2 = (x-y)/sqrt(2)
    // f = z1^2 + 100*z2^2 = 0.5*(x+y)^2 + 50*(x-y)^2
    // Global minimum at (0,0)
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float z1 = (x + y) * 0.7071f; // 1/sqrt(2)
        float z2 = (x - y) * 0.7071f;
        return z1 * z1 + 100.f * z2 * z2; // condition number = 100
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    ParameterOptimization::Result result;
    retry(3, [&]() {
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 200;
        cmaes.sigma = 0.5f;
        popt.setAlgorithm(cmaes);
        result = popt.run(sim, params);
        return result.fitness < 0.1f;
    });

    DOCTEST_INFO("CMA-ES rotated ellipsoid: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 0)", " y=", result.parameters.at("y").value, " (opt: 0)");

    DOCTEST_CHECK(result.fitness < 0.1f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(0.f).epsilon(0.2f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(0.f).epsilon(0.2f));
}

// Test 20: BO normalization with large offset
// f(x,y) = 1e6 + x^2 + y^2
// The GP must normalize y-values correctly; without normalization the kernel
// length-scales would be dominated by the constant offset.
DOCTEST_TEST_CASE("ParameterOptimization BO normalization stress") {
    const float OFFSET = 1e6f;
    auto sim = [OFFSET](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return OFFSET + x * x + y * y;
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    ParameterOptimization::Result result;
    retry(3, [&]() {
        ParameterOptimization popt;
        BayesianOptimization bo;
        bo.max_evaluations = 60;
        bo.initial_samples = 10;
        bo.acquisition_samples = 1000;
        popt.setAlgorithm(bo);
        result = popt.run(sim, params);
        return (result.fitness - OFFSET) < 2.0f;
    });

    float relative_fitness = result.fitness - OFFSET;
    DOCTEST_INFO("BO normalization stress: raw fitness=", result.fitness, " relative=", relative_fitness, " (optimum: relative=0 at x=0, y=0)");

    DOCTEST_CHECK(relative_fitness < 2.0f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(0.f).epsilon(1.0f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(0.f).epsilon(1.0f));
}

// Test 21: Single parameter edge case (n=1)
// Ensure all three methods handle the degenerate case of a single parameter correctly.
DOCTEST_TEST_CASE("ParameterOptimization single parameter") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        return (x - 3.f) * (x - 3.f);
    };

    ParametersToOptimize params = {{"x", {-5.f, -10.f, 10.f}}};

    // Test GA
    {
        ParameterOptimization popt;
        GeneticAlgorithm ga;
        ga.generations = 80;
        ga.population_size = 15;
        popt.setAlgorithm(ga);
        auto result = popt.run(sim, params);

        DOCTEST_INFO("GA single-param: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 3)");
        DOCTEST_CHECK(result.fitness < 0.5f);
        DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.5f));
    }

    // Test BO
    {
        ParameterOptimization popt;
        BayesianOptimization bo;
        bo.max_evaluations = 80;
        popt.setAlgorithm(bo);
        auto result = popt.run(sim, params);

        DOCTEST_INFO("BO single-param: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 3)");
        DOCTEST_CHECK(result.fitness < 0.5f);
        DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.5f));
    }

    // Test CMA-ES
    {
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 80;
        popt.setAlgorithm(cmaes);
        auto result = popt.run(sim, params);

        DOCTEST_INFO("CMA-ES single-param: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 3)");
        DOCTEST_CHECK(result.fitness < 0.5f);
        DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.5f));
    }
}

// =============================================================================
// Tests 22-24: Integer and Categorical parameter support (GA only)
// =============================================================================

// Test 22: GA with integer parameters
// Mixed float + integer: f(x,n) = (x - 3.7)^2 + (n - 5)^2
// x is float in [-10,10], n is integer in [1,10]
// Optimal: x = 3.7, n = 5 (nearest integer to minimum), f = 0 + 0 = 0 at n=5
DOCTEST_TEST_CASE("ParameterOptimization GA integer parameters") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float n = p.at("n").value;
        return (x - 3.7f) * (x - 3.7f) + (n - 5.f) * (n - 5.f);
    };

    ParametersToOptimize params = {
            {"x", {0.f, -10.f, 10.f}}, // float (default)
            {"n", {1.f, 1.f, 10.f, ParameterType::INTEGER}} // integer
    };

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 100;
    ga.population_size = 20;
    popt.setAlgorithm(ga);

    auto result = popt.run(sim, params);

    float n_val = result.parameters.at("n").value;
    float x_val = result.parameters.at("x").value;

    DOCTEST_INFO("GA integer: fitness=", result.fitness, " (optimum: 0)", " | x=", x_val, " (opt: 3.7)", " n=", n_val, " (opt: 5)");

    // n must be a whole number
    DOCTEST_CHECK(n_val == std::round(n_val));
    // n should converge to 5
    DOCTEST_CHECK(n_val == doctest::Approx(5.f).epsilon(0.01f));
    // x should converge near 3.7
    DOCTEST_CHECK(x_val == doctest::Approx(3.7f).epsilon(0.3f));
    // Fitness: GA achieved 2.4e-10 in testing. Tighten well below random search territory.
    DOCTEST_CHECK(result.fitness < 0.1f);
}

// Test 23: GA with categorical parameters
// Mixed float + categorical: f(x,c) = (x - 2.0)^2 + (c - 3.14)^2
// x is float in [-5,5], c is categorical {0.5, 1.7, 3.14, 7.2}
// Non-integer categories ensure BLX-alpha interpolation can't accidentally land on a valid value.
// Optimal: x = 2.0, c = 3.14, f = 0
DOCTEST_TEST_CASE("ParameterOptimization GA categorical parameters") {
    const std::vector<float> cats = {0.5f, 1.7f, 3.14f, 7.2f};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float c = p.at("c").value;
        return (x - 2.0f) * (x - 2.0f) + (c - 3.14f) * (c - 3.14f);
    };

    ParametersToOptimize params = {
            {"x", {0.f, -5.f, 5.f}}, // float
            {"c", {0.5f, 0.f, 0.f, ParameterType::CATEGORICAL, {0.5f, 1.7f, 3.14f, 7.2f}}} // categorical
    };

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 100;
    ga.population_size = 20;
    popt.setAlgorithm(ga);

    auto result = popt.run(sim, params);

    float c_val = result.parameters.at("c").value;
    float x_val = result.parameters.at("x").value;

    DOCTEST_INFO("GA categorical: fitness=", result.fitness, " (optimum: 0)", " | x=", x_val, " (opt: 2.0)", " c=", c_val, " (opt: 3.14)");

    // c must be exactly one of the allowed categories
    bool is_valid_category = false;
    for (float cat: cats) {
        if (c_val == cat) {
            is_valid_category = true;
            break;
        }
    }
    DOCTEST_CHECK(is_valid_category);
    // c should converge to 3.14
    DOCTEST_CHECK(c_val == doctest::Approx(3.14f).epsilon(0.01f));
    // x should converge near 2.0
    DOCTEST_CHECK(x_val == doctest::Approx(2.0f).epsilon(0.3f));
    // Fitness: tightened to reject random search
    DOCTEST_CHECK(result.fitness < 0.1f);
}

// Test 24: GA with all three parameter types mixed
// f(x, n, c) = (x - 1.5)^2 + (n - 3)^2 + (c - 2.71)^2
// x: float [-5,5], n: integer [1,6], c: categorical {0.5, 1.7, 2.71, 7.2}
// Non-integer categories ensure type enforcement is real.
// Optimal: x = 1.5, n = 3, c = 2.71, f = 0
DOCTEST_TEST_CASE("ParameterOptimization GA mixed parameter types") {
    const std::vector<float> cats = {0.5f, 1.7f, 2.71f, 7.2f};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float n = p.at("n").value;
        float c = p.at("c").value;
        return (x - 1.5f) * (x - 1.5f) + (n - 3.f) * (n - 3.f) + (c - 2.71f) * (c - 2.71f);
    };

    ParametersToOptimize params = {
            {"x", {0.f, -5.f, 5.f}}, // float
            {"n", {1.f, 1.f, 6.f, ParameterType::INTEGER}}, // integer
            {"c", {0.5f, 0.f, 0.f, ParameterType::CATEGORICAL, {0.5f, 1.7f, 2.71f, 7.2f}}} // categorical
    };

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 150;
    ga.population_size = 25;
    popt.setAlgorithm(ga);

    auto result = popt.run(sim, params);

    float x_val = result.parameters.at("x").value;
    float n_val = result.parameters.at("n").value;
    float c_val = result.parameters.at("c").value;

    DOCTEST_INFO("GA mixed: fitness=", result.fitness, " (optimum: 0)", " | x=", x_val, " (opt: 1.5)", " n=", n_val, " (opt: 3)", " c=", c_val, " (opt: 2.71)");

    // Verify type constraints
    DOCTEST_CHECK(n_val == std::round(n_val)); // n is integer
    bool is_valid_category = false;
    for (float cat: cats) {
        if (c_val == cat) {
            is_valid_category = true;
            break;
        }
    }
    DOCTEST_CHECK(is_valid_category); // c is a valid category

    // Verify convergence
    DOCTEST_CHECK(x_val == doctest::Approx(1.5f).epsilon(0.3f));
    DOCTEST_CHECK(n_val == doctest::Approx(3.f).epsilon(0.01f));
    DOCTEST_CHECK(c_val == doctest::Approx(2.71f).epsilon(0.01f));
    DOCTEST_CHECK(result.fitness < 0.1f);
}

// Test 25: Validation errors for invalid parameter configurations
DOCTEST_TEST_CASE("ParameterOptimization GA parameter validation") {
    auto sim = [](const ParametersToOptimize &p) { return 0.f; };

    GeneticAlgorithm ga;
    ga.generations = 10;
    ga.population_size = 5;

    // Empty categories should throw
    {
        ParametersToOptimize params = {{"c", {0.f, 0.f, 0.f, ParameterType::CATEGORICAL, {}}}};
        ParameterOptimization popt;
        popt.setAlgorithm(ga);
        DOCTEST_CHECK_THROWS_AS(popt.run(sim, params), std::runtime_error);
    }

    // Non-integer min for INTEGER should throw
    {
        ParametersToOptimize params = {{"n", {1.f, 0.5f, 10.f, ParameterType::INTEGER}}};
        ParameterOptimization popt;
        popt.setAlgorithm(ga);
        DOCTEST_CHECK_THROWS_AS(popt.run(sim, params), std::runtime_error);
    }

    // Non-integer max for INTEGER should throw
    {
        ParametersToOptimize params = {{"n", {1.f, 1.f, 9.7f, ParameterType::INTEGER}}};
        ParameterOptimization popt;
        popt.setAlgorithm(ga);
        DOCTEST_CHECK_THROWS_AS(popt.run(sim, params), std::runtime_error);
    }
}

// =============================================================================
// Tests 26-28: BLXPCA crossover operator
// =============================================================================

// Test 26: BLXPCA on Rosenbrock (non-separable, curved valley)
// f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1) with f=0
// The curved valley couples x and y; BLXPCA should detect this via PCA and
// explore along the valley rather than axis-aligned directions.
DOCTEST_TEST_CASE("ParameterOptimization GA BLXPCA Rosenbrock") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float term1 = (1.f - x);
        float term2 = (y - x * x);
        return term1 * term1 + 100.f * term2 * term2;
    };

    ParametersToOptimize params = {{"x", {0.f, -2.f, 2.f}}, {"y", {0.f, -2.f, 2.f}}};

    ParameterOptimization::Result result;
    retry(3, [&]() {
        ParameterOptimization popt;
        GeneticAlgorithm ga;
        ga.generations = 200;
        ga.population_size = 40;
        ga.crossover_rate = 0.8f;
        ga.crossover = BLXPCACrossover{};
        IsotropicMutation im;
        im.rate = 0.15f;
        ga.mutation = im;
        popt.setAlgorithm(ga);
        result = popt.run(sim, params);
        return result.fitness < 1.0f && std::abs(result.parameters.at("x").value - 1.f) < 0.3f && std::abs(result.parameters.at("y").value - 1.f) < 0.3f;
    });

    DOCTEST_INFO("BLXPCA Rosenbrock: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 1)", " y=", result.parameters.at("y").value, " (opt: 1)");

    DOCTEST_CHECK(result.fitness < 1.0f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(1.f).epsilon(0.3f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(1.f).epsilon(0.3f));
}

// Test 27: BLXPCA on rotated ellipsoid (off-axis conditioning)
// f(x,y) = z1^2 + 100*z2^2 where z = R*x (45-degree rotation)
// Standard BLX-alpha searches along x/y axes, but the ellipsoid is aligned at 45 degrees.
// BLXPCA should rotate its search directions to match the ellipsoid axes.
DOCTEST_TEST_CASE("ParameterOptimization GA BLXPCA rotated ellipsoid") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float z1 = (x + y) * 0.7071f; // 1/sqrt(2)
        float z2 = (x - y) * 0.7071f;
        return z1 * z1 + 100.f * z2 * z2; // condition number = 100
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    IsotropicMutation im;
    im.rate = 0.15f;

    // BLXPCA with retry
    ParameterOptimization::Result pca_result;
    retry(3, [&]() {
        ParameterOptimization popt_pca;
        GeneticAlgorithm ga_pca;
        ga_pca.generations = 200;
        ga_pca.population_size = 40;
        ga_pca.crossover_rate = 0.8f;
        ga_pca.crossover = BLXPCACrossover{};
        ga_pca.mutation = im;
        popt_pca.setAlgorithm(ga_pca);
        pca_result = popt_pca.run(sim, params);
        return pca_result.fitness < 1.0f;
    });

    // Run standard BLX-alpha with same settings for comparison (no retry needed, informational only)
    ParameterOptimization popt_blx;
    GeneticAlgorithm ga_blx;
    ga_blx.generations = 200;
    ga_blx.population_size = 40;
    ga_blx.crossover_rate = 0.8f;
    ga_blx.crossover = BLXAlphaCrossover{};
    ga_blx.mutation = im;
    popt_blx.setAlgorithm(ga_blx);

    auto blx_result = popt_blx.run(sim, params);

    DOCTEST_INFO("BLXPCA rotated ellipsoid: fitness=", pca_result.fitness, " (optimum: 0)", " | x=", pca_result.parameters.at("x").value, " (opt: 0)", " y=", pca_result.parameters.at("y").value, " (opt: 0)");
    DOCTEST_INFO("BLX-alpha rotated ellipsoid: fitness=", blx_result.fitness, " (optimum: 0)", " | x=", blx_result.parameters.at("x").value, " (opt: 0)", " y=", blx_result.parameters.at("y").value, " (opt: 0)");

    // BLXPCA should handle the rotated ellipsoid well
    DOCTEST_CHECK(pca_result.fitness < 1.0f);
    DOCTEST_CHECK(pca_result.parameters.at("x").value == doctest::Approx(0.f).epsilon(0.5f));
    DOCTEST_CHECK(pca_result.parameters.at("y").value == doctest::Approx(0.f).epsilon(0.5f));
}

// Test 28: BLXPCA with mixed parameter types (FLOAT + INTEGER + CATEGORICAL)
// BLXPCA applies PCA-space crossover only to numeric (FLOAT/INTEGER) parameters.
// Categorical parameters use uniform crossover regardless.
// f(x, n, c) = (x - 1.5)^2 + (n - 3)^2 + (c - 2.71)^2
DOCTEST_TEST_CASE("ParameterOptimization GA BLXPCA mixed types") {
    const std::vector<float> cats = {0.5f, 1.7f, 2.71f, 7.2f};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float n = p.at("n").value;
        float c = p.at("c").value;
        return (x - 1.5f) * (x - 1.5f) + (n - 3.f) * (n - 3.f) + (c - 2.71f) * (c - 2.71f);
    };

    ParametersToOptimize params = {
            {"x", {0.f, -5.f, 5.f}}, // float
            {"n", {1.f, 1.f, 6.f, ParameterType::INTEGER}}, // integer
            {"c", {0.5f, 0.f, 0.f, ParameterType::CATEGORICAL, {0.5f, 1.7f, 2.71f, 7.2f}}} // categorical
    };

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 150;
    ga.population_size = 25;
    ga.crossover_rate = 0.8f;
    ga.crossover = BLXPCACrossover{};
    IsotropicMutation im;
    im.rate = 0.15f;
    ga.mutation = im;
    popt.setAlgorithm(ga);

    auto result = popt.run(sim, params);

    float x_val = result.parameters.at("x").value;
    float n_val = result.parameters.at("n").value;
    float c_val = result.parameters.at("c").value;

    DOCTEST_INFO("BLXPCA mixed: fitness=", result.fitness, " (optimum: 0)", " | x=", x_val, " (opt: 1.5)", " n=", n_val, " (opt: 3)", " c=", c_val, " (opt: 2.71)");

    // Verify type constraints still hold with BLXPCA
    DOCTEST_CHECK(n_val == std::round(n_val)); // n is integer
    bool is_valid_category = false;
    for (float cat: cats) {
        if (c_val == cat) {
            is_valid_category = true;
            break;
        }
    }
    DOCTEST_CHECK(is_valid_category); // c is a valid category

    // Verify convergence
    DOCTEST_CHECK(x_val == doctest::Approx(1.5f).epsilon(0.3f));
    DOCTEST_CHECK(n_val == doctest::Approx(3.f).epsilon(0.01f));
    DOCTEST_CHECK(c_val == doctest::Approx(2.71f).epsilon(0.01f));
    DOCTEST_CHECK(result.fitness < 0.1f);
}

// =============================================================================
// Tests 29-31: Mutation type variants
// =============================================================================

// Test 29: ISOTROPIC mutation converges on quadratic
// Single-gate isotropic mutation: all genes mutated together when triggered.
// Should converge similarly to PER_GENE on separable problems.
DOCTEST_TEST_CASE("ParameterOptimization GA isotropic mutation") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 100;
    ga.population_size = 30;
    ga.mutation = IsotropicMutation{}; // uses default rate=0.1f
    popt.setAlgorithm(ga);

    auto result = popt.run(sim, params);

    DOCTEST_INFO("Isotropic mutation: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 3)", " y=", result.parameters.at("y").value, " (opt: -1)");

    DOCTEST_CHECK(result.fitness < 0.1f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.1f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.1f));
}

// Test 30: HYBRID mutation on Rosenbrock (non-separable)
// PCA-Gaussian + PCA-Cauchy + Random Direction hybrid with BLXPCA crossover.
// The Cauchy component provides escape capability on the curved valley.
DOCTEST_TEST_CASE("ParameterOptimization GA hybrid mutation Rosenbrock") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        float term1 = (1.f - x);
        float term2 = (y - x * x);
        return term1 * term1 + 100.f * term2 * term2;
    };

    ParametersToOptimize params = {{"x", {0.f, -2.f, 2.f}}, {"y", {0.f, -2.f, 2.f}}};

    ParameterOptimization::Result result;
    retry(3, [&]() {
        ParameterOptimization popt;
        GeneticAlgorithm ga;
        ga.generations = 200;
        ga.population_size = 40;
        ga.crossover_rate = 0.8f;
        ga.crossover = BLXPCACrossover{};
        HybridMutation hm;
        hm.rate = 0.15f;
        ga.mutation = hm;
        popt.setAlgorithm(ga);
        result = popt.run(sim, params);
        return result.fitness < 1.0f && std::abs(result.parameters.at("x").value - 1.f) < 0.3f && std::abs(result.parameters.at("y").value - 1.f) < 0.3f;
    });

    DOCTEST_INFO("Hybrid mutation Rosenbrock: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 1)", " y=", result.parameters.at("y").value, " (opt: 1)");

    DOCTEST_CHECK(result.fitness < 1.0f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(1.f).epsilon(0.3f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(1.f).epsilon(0.3f));
}

// Test 31: HYBRID mutation with BLX_ALPHA crossover (mix-and-match)
// Verify that HYBRID mutation works with standard BLX_ALPHA crossover.
// PCA is computed for mutation even though crossover doesn't use it.
DOCTEST_TEST_CASE("ParameterOptimization GA hybrid mutation with BLX_ALPHA") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return x * x + y * y;
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 100;
    ga.population_size = 30;
    ga.crossover = BLXAlphaCrossover{}; // standard crossover
    HybridMutation hm;
    hm.rate = 0.15f;
    ga.mutation = hm; // hybrid mutation (PCA computed for mutation)
    popt.setAlgorithm(ga);

    auto result = popt.run(sim, params);

    DOCTEST_INFO("Hybrid+BLX_ALPHA: fitness=", result.fitness, " (optimum: 0)", " | x=", result.parameters.at("x").value, " (opt: 0)", " y=", result.parameters.at("y").value, " (opt: 0)");

    DOCTEST_CHECK(result.fitness < 0.5f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(0.f).epsilon(0.5f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(0.f).epsilon(0.5f));
}

// =============================================================================
// Tests 32-39: Gradient-based optimization (L-BFGS)
// =============================================================================

// Test 32: LBFGS quadratic convergence with exact gradient
// f(x,y) = (x-3)^2 + (y+1)^2, grad = (2(x-3), 2(y+1))
// L-BFGS should converge to near-zero in very few evaluations.
DOCTEST_TEST_CASE("ParameterOptimization LBFGS quadratic with exact gradient") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return {{"x", 2.f * (x - 3.f)}, {"y", 2.f * (y + 1.f)}};
    };

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    LBFGS lbfgs;
    lbfgs.max_iterations = 100;
    popt.setAlgorithm(lbfgs);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(objective, params, gradient);

    DOCTEST_INFO("LBFGS quadratic: fitness=", result.fitness,
                 " | x=", result.parameters.at("x").value, " (opt: 3)",
                 " y=", result.parameters.at("y").value, " (opt: -1)");

    DOCTEST_CHECK(result.fitness < 1e-6f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.001f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.001f));
#else
    // Without NLopt, should return initial params with non-optimal fitness
    auto result = popt.run(objective, params, gradient);
    DOCTEST_INFO("LBFGS without NLopt: fitness=", result.fitness, " (expected: no convergence)");
    DOCTEST_CHECK(result.fitness > 1e-6f);
#endif
}

// Test 33: LBFGS with composed gradient (mixed FD sources)
// Verify makeGradientFunction + makeFDGradientSource produce correct results.
DOCTEST_TEST_CASE("ParameterOptimization LBFGS with composed gradient") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 2.f) * (x - 2.f) + (y - 1.f) * (y - 1.f);
    };

    // Compose: exact gradient for x, FD for y
    auto x_source = GradientSource{
        "x_analytical", {"x"},
        [](const ParametersToOptimize &p, ParameterGradient &g) {
            g["x"] = 2.f * (p.at("x").value - 2.f);
        }
    };

    auto y_source = makeFDGradientSource("y_fd", {"y"}, objective, 1e-5f);

    auto gradient = makeGradientFunction({x_source, y_source});

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    LBFGS lbfgs;
    lbfgs.max_iterations = 100;
    popt.setAlgorithm(lbfgs);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(objective, params, gradient);

    DOCTEST_INFO("LBFGS composed gradient: fitness=", result.fitness,
                 " | x=", result.parameters.at("x").value, " (opt: 2)",
                 " y=", result.parameters.at("y").value, " (opt: 1)");

    DOCTEST_CHECK(result.fitness < 1e-4f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(2.f).epsilon(0.01f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(1.f).epsilon(0.01f));
#endif
}

// Test 34: makeFDGradientSource correctness
// Compare FD gradient against analytical gradient on a smooth function.
DOCTEST_TEST_CASE("ParameterOptimization FD gradient helper correctness") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return x * x * x + 2.f * x * y + y * y;  // f = x^3 + 2xy + y^2
    };

    auto fd_source = makeFDGradientSource("all_fd", {"x", "y"}, objective, 1e-5f);

    ParametersToOptimize test_point = {{"x", {1.5f, -5.f, 5.f}}, {"y", {-0.5f, -5.f, 5.f}}};

    ParameterGradient fd_grad;
    fd_source.compute(test_point, fd_grad);

    // Analytical: df/dx = 3x^2 + 2y = 3(1.5)^2 + 2(-0.5) = 6.75 - 1.0 = 5.75
    // Analytical: df/dy = 2x + 2y = 2(1.5) + 2(-0.5) = 3.0 - 1.0 = 2.0
    // FD truncation error is O(h^2*f''') so wider tolerance for cubic term
    DOCTEST_CHECK(fd_grad.at("x") == doctest::Approx(5.75f).epsilon(0.01f));
    DOCTEST_CHECK(fd_grad.at("y") == doctest::Approx(2.0f).epsilon(0.001f));
}

// Test 35: LBFGS rejects INTEGER parameters
DOCTEST_TEST_CASE("ParameterOptimization LBFGS rejects INTEGER params") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) { return 0.f; };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 0.f}, {"n", 0.f}};
    };

    ParametersToOptimize params = {
        {"x", {0.f, -5.f, 5.f}},
        {"n", {1.f, 1.f, 10.f, ParameterType::INTEGER}}
    };

    ParameterOptimization popt;
    LBFGS lbfgs;
    popt.setAlgorithm(lbfgs);

    DOCTEST_CHECK_THROWS_AS(popt.run(objective, params, gradient), std::runtime_error);
}

// Test 36: LBFGS rejects CATEGORICAL parameters
DOCTEST_TEST_CASE("ParameterOptimization LBFGS rejects CATEGORICAL params") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) { return 0.f; };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 0.f}, {"c", 0.f}};
    };

    ParametersToOptimize params = {
        {"x", {0.f, -5.f, 5.f}},
        {"c", {0.5f, 0.f, 0.f, ParameterType::CATEGORICAL, {0.5f, 1.0f, 2.0f}}}
    };

    ParameterOptimization popt;
    LBFGS lbfgs;
    popt.setAlgorithm(lbfgs);

    DOCTEST_CHECK_THROWS_AS(popt.run(objective, params, gradient), std::runtime_error);
}

// Test 37: LBFGS without gradient (wrong overload) gives clear error
DOCTEST_TEST_CASE("ParameterOptimization LBFGS requires gradient overload") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        return x * x;
    };

    ParametersToOptimize params = {{"x", {1.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    LBFGS lbfgs;
    popt.setAlgorithm(lbfgs);

    // Calling the gradient-free run() with LBFGS should throw
    DOCTEST_CHECK_THROWS_AS(popt.run(sim, params), std::runtime_error);
}

// Test 38: Gradient missing a parameter key throws error
DOCTEST_TEST_CASE("ParameterOptimization LBFGS missing gradient key") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        return p.at("x").value * p.at("x").value + p.at("y").value * p.at("y").value;
    };

    // Gradient only provides "x", missing "y"
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}};
    };

    ParametersToOptimize params = {{"x", {1.f, -5.f, 5.f}}, {"y", {1.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    LBFGS lbfgs;
    popt.setAlgorithm(lbfgs);

#ifdef HELIOS_HAVE_NLOPT
    DOCTEST_CHECK_THROWS_AS(popt.run(objective, params, gradient), std::runtime_error);
#endif
}

// Test 39: LBFGS result fitness consistency
// Verify that the returned fitness matches recomputing f(optimal_params).
DOCTEST_TEST_CASE("ParameterOptimization LBFGS result fitness consistency") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return x * x + y * y;
    };

    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 2.f * p.at("y").value}};
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    LBFGS lbfgs;
    lbfgs.max_iterations = 50;
    popt.setAlgorithm(lbfgs);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(objective, params, gradient);
    float recomputed = objective(result.parameters);

    DOCTEST_INFO("LBFGS consistency: reported=", result.fitness, " recomputed=", recomputed);
    DOCTEST_CHECK(result.fitness == doctest::Approx(recomputed).epsilon(1e-5f));
#endif
}

// =============================================================================
// Tests 40-43: Adam optimizer
// =============================================================================

// Test 40: Adam quadratic convergence with exact gradient
DOCTEST_TEST_CASE("ParameterOptimization Adam quadratic with exact gradient") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * (p.at("x").value - 3.f)}, {"y", 2.f * (p.at("y").value + 1.f)}};
    };
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    Adam adam;
    adam.max_iterations = 500;
    adam.learning_rate = 0.1f;
    popt.setAlgorithm(adam);
    auto result = popt.run(objective, params, gradient);

    DOCTEST_INFO("Adam quadratic: fitness=", result.fitness,
                 " | x=", result.parameters.at("x").value, " (opt: 3)",
                 " y=", result.parameters.at("y").value, " (opt: -1)");
    DOCTEST_CHECK(result.fitness < 0.01f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.1f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.1f));
}

// Test 41: Adam with noisy gradient (simulated noise)
// Verify that Adam's momentum smooths noise and still converges.
DOCTEST_TEST_CASE("ParameterOptimization Adam with noisy gradient") {
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.f, 0.5f);

    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return x * x + y * y;
    };
    GradientFunction gradient = [&](const ParametersToOptimize &p) -> ParameterGradient {
        float x = p.at("x").value, y = p.at("y").value;
        return {{"x", 2.f * x + noise(rng)}, {"y", 2.f * y + noise(rng)}};
    };
    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    Adam adam;
    adam.max_iterations = 500;
    adam.learning_rate = 0.05f;
    popt.setAlgorithm(adam);
    auto result = popt.run(objective, params, gradient);

    DOCTEST_INFO("Adam noisy: fitness=", result.fitness, " (optimum: 0)");
    DOCTEST_CHECK(result.fitness < 1.0f);
}

// Test 42: Adam requires gradient overload
DOCTEST_TEST_CASE("ParameterOptimization Adam requires gradient overload") {
    auto sim = [](const ParametersToOptimize &p) { return p.at("x").value * p.at("x").value; };
    ParametersToOptimize params = {{"x", {1.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    Adam adam;
    popt.setAlgorithm(adam);
    DOCTEST_CHECK_THROWS_AS(popt.run(sim, params), std::runtime_error);
}

// Test 43: Adam result fitness consistency
DOCTEST_TEST_CASE("ParameterOptimization Adam result fitness consistency") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return x * x + y * y;
    };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 2.f * p.at("y").value}};
    };
    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    Adam adam;
    adam.max_iterations = 200;
    adam.learning_rate = 0.05f;
    popt.setAlgorithm(adam);
    auto result = popt.run(objective, params, gradient);
    float recomputed = objective(result.parameters);

    DOCTEST_INFO("Adam consistency: reported=", result.fitness, " recomputed=", recomputed);
    DOCTEST_CHECK(result.fitness == doctest::Approx(recomputed).epsilon(1e-5f));
}

// =============================================================================
// =============================================================================
// Tests 44-49: BOBYQA and SLSQP
// =============================================================================

// Test 44: BOBYQA quadratic convergence (derivative-free local)
DOCTEST_TEST_CASE("ParameterOptimization BOBYQA quadratic") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    BOBYQA bobyqa;
    bobyqa.max_iterations = 100;
    popt.setAlgorithm(bobyqa);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(sim, params);
    DOCTEST_INFO("BOBYQA quadratic: fitness=", result.fitness,
                 " | x=", result.parameters.at("x").value, " (opt: 3)",
                 " y=", result.parameters.at("y").value, " (opt: -1)");
    DOCTEST_CHECK(result.fitness < 1e-6f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(0.001f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.001f));
#else
    auto result = popt.run(sim, params);
    DOCTEST_CHECK(result.fitness > 1e-6f);
#endif
}

// Test 45: BOBYQA Rosenbrock (harder, derivative-free)
DOCTEST_TEST_CASE("ParameterOptimization BOBYQA Rosenbrock") {
    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        float t1 = (1.f - x), t2 = (y - x * x);
        return t1 * t1 + 100.f * t2 * t2;
    };
    ParametersToOptimize params = {{"x", {-1.f, -2.f, 2.f}}, {"y", {-1.f, -2.f, 2.f}}};

    ParameterOptimization popt;
    BOBYQA bobyqa;
    bobyqa.max_iterations = 500;
    popt.setAlgorithm(bobyqa);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(sim, params);
    DOCTEST_INFO("BOBYQA Rosenbrock: fitness=", result.fitness,
                 " | x=", result.parameters.at("x").value, " (opt: 1)",
                 " y=", result.parameters.at("y").value, " (opt: 1)");
    DOCTEST_CHECK(result.fitness < 1e-4f);
#endif
}

// Test 46: SLSQP unconstrained (should behave like L-BFGS)
DOCTEST_TEST_CASE("ParameterOptimization SLSQP unconstrained quadratic") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * (p.at("x").value - 3.f)}, {"y", 2.f * (p.at("y").value + 1.f)}};
    };
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    SLSQP slsqp;
    popt.setAlgorithm(slsqp);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(objective, params, gradient);
    DOCTEST_INFO("SLSQP unconstrained: fitness=", result.fitness);
    DOCTEST_CHECK(result.fitness < 1e-6f);
#endif
}

// Test 47: SLSQP with inequality constraint
// Minimize x^2 + y^2 subject to x + y >= 1 (i.e., -(x+y-1) <= 0)
// Unconstrained optimum is (0,0). Constrained optimum is (0.5, 0.5) with f=0.5
DOCTEST_TEST_CASE("ParameterOptimization SLSQP with constraint") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return x * x + y * y;
    };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 2.f * p.at("y").value}};
    };

    // Constraint: x + y >= 1, written as -(x + y - 1) <= 0, i.e., 1 - x - y <= 0
    Constraint c;
    c.function = [](const ParametersToOptimize &p) -> float {
        return 1.f - p.at("x").value - p.at("y").value;
    };
    c.gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", -1.f}, {"y", -1.f}};
    };
    c.tolerance = 1e-6f;

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    SLSQP slsqp;
    popt.setAlgorithm(slsqp);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(objective, params, gradient, {c});
    DOCTEST_INFO("SLSQP constrained: fitness=", result.fitness,
                 " | x=", result.parameters.at("x").value, " (opt: 0.5)",
                 " y=", result.parameters.at("y").value, " (opt: 0.5)");
    DOCTEST_CHECK(result.fitness == doctest::Approx(0.5f).epsilon(0.01f));
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(0.5f).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(0.5f).epsilon(0.05f));
#endif
}

// Test 48: SLSQP Cowan-Farquhar style: maximize A subject to E < budget
// Simplified: maximize x+y subject to x^2+y^2 <= 1 (unit circle)
// Constrained optimum: x=y=1/sqrt(2), f=sqrt(2)
DOCTEST_TEST_CASE("ParameterOptimization SLSQP maximize with constraint") {
    // Minimize -(x+y) subject to x^2+y^2 - 1 <= 0
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        return -(p.at("x").value + p.at("y").value);
    };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", -1.f}, {"y", -1.f}};
    };

    Constraint c;
    c.function = [](const ParametersToOptimize &p) -> float {
        float x = p.at("x").value, y = p.at("y").value;
        return x * x + y * y - 1.f;  // <= 0
    };
    c.gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 2.f * p.at("y").value}};
    };

    ParametersToOptimize params = {{"x", {0.1f, -2.f, 2.f}}, {"y", {0.1f, -2.f, 2.f}}};

    ParameterOptimization popt;
    SLSQP slsqp;
    popt.setAlgorithm(slsqp);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(objective, params, gradient, {c});
    float expected_f = -std::sqrt(2.f);
    float expected_xy = 1.f / std::sqrt(2.f);
    DOCTEST_INFO("SLSQP maximize+constraint: fitness=", result.fitness, " (opt: ", expected_f, ")",
                 " | x=", result.parameters.at("x").value, " (opt: ", expected_xy, ")",
                 " y=", result.parameters.at("y").value, " (opt: ", expected_xy, ")");
    DOCTEST_CHECK(result.fitness == doctest::Approx(expected_f).epsilon(0.01f));
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(expected_xy).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(expected_xy).epsilon(0.05f));
#endif
}

// Test 49: SLSQP with combined ConstrainedSimulation (single-function API)
// Same problem as test 47: minimize x^2+y^2 s.t. x+y >= 1
DOCTEST_TEST_CASE("ParameterOptimization SLSQP combined simulation") {
    int eval_count = 0;

    ConstrainedSimulation sim = [&](const ParametersToOptimize &p) -> ConstrainedResult {
        eval_count++;
        float x = p.at("x").value, y = p.at("y").value;
        return {
            x * x + y * y,                           // objective
            {{"x", 2.f * x}, {"y", 2.f * y}},        // obj gradient
            {1.f - x - y},                            // constraint: 1-x-y <= 0
            {{{"x", -1.f}, {"y", -1.f}}}              // constraint gradient
        };
    };

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    SLSQP slsqp;
    popt.setAlgorithm(slsqp);

#ifdef HELIOS_HAVE_NLOPT
    auto result = popt.run(sim, params);
    DOCTEST_INFO("SLSQP combined: fitness=", result.fitness,
                 " | x=", result.parameters.at("x").value,
                 " y=", result.parameters.at("y").value,
                 " | sim evals=", eval_count);
    DOCTEST_CHECK(result.fitness == doctest::Approx(0.5f).epsilon(0.01f));
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(0.5f).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(0.5f).epsilon(0.05f));
    // Now run the same problem with separate callbacks and count total callback invocations
    int separate_obj_calls = 0, separate_con_calls = 0;
    ObjectiveFunction obj_sep = [&](const ParametersToOptimize &p) -> float {
        separate_obj_calls++;
        float x = p.at("x").value, y = p.at("y").value;
        return x * x + y * y;
    };
    GradientFunction grad_sep = [&](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 2.f * p.at("y").value}};
    };
    Constraint c_sep;
    c_sep.function = [&](const ParametersToOptimize &p) -> float {
        separate_con_calls++;
        return 1.f - p.at("x").value - p.at("y").value;
    };
    c_sep.gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", -1.f}, {"y", -1.f}};
    };

    ParameterOptimization popt2;
    SLSQP slsqp2;
    popt2.setAlgorithm(slsqp2);
    ParametersToOptimize params2 = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    popt2.run(obj_sep, params2, grad_sep, {c_sep});

    int total_separate_callbacks = separate_obj_calls + separate_con_calls;
    DOCTEST_INFO("Combined sim evals: ", eval_count,
                 " | Separate callbacks: obj=", separate_obj_calls,
                 " con=", separate_con_calls, " total=", total_separate_callbacks);
    // The combined simulation should run strictly fewer times than the total
    // number of separate callbacks (obj + constraint calls), because the cache
    // prevents re-evaluation at the same point.
    DOCTEST_CHECK(eval_count < total_separate_callbacks);
#endif
}

// Test 50: SLSQP requires gradient overload
DOCTEST_TEST_CASE("ParameterOptimization SLSQP requires gradient overload") {
    auto sim = [](const ParametersToOptimize &p) { return p.at("x").value * p.at("x").value; };
    ParametersToOptimize params = {{"x", {1.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    SLSQP slsqp;
    popt.setAlgorithm(slsqp);
    DOCTEST_CHECK_THROWS_AS(popt.run(sim, params), std::runtime_error);
}

// =============================================================================
// Tests 50-52: Algorithm comparison benchmarks (GA vs BO vs CMA-ES vs LBFGS vs Adam)
// These tests run all algorithms on the same problems with timing to show
// the convergence quality and speed differences between derivative-free and
// gradient-based methods.
// =============================================================================

// Helper: run a timed optimization and return {fitness, time_us, evals}
struct BenchmarkResult {
    std::string name;
    float fitness;
    long long time_us; // std::chrono::microseconds::rep is int64_t; using long triggers a narrowing conversion on MSVC where long is 32-bit
    int evals;
};

// Test 40: Four-way comparison on 3D quadratic
// f(a,b,c) = (a-5)^2 + (b-10)^2 + (c-2)^2, minimum at (5,10,2)
DOCTEST_TEST_CASE("ParameterOptimization four-way comparison: 3D quadratic") {
    int eval_count = 0;

    auto make_objective = [&eval_count]() {
        eval_count = 0;
        return [&eval_count](const ParametersToOptimize &p) -> float {
            eval_count++;
            float a = p.at("a").value;
            float b = p.at("b").value;
            float c = p.at("c").value;
            return (a - 5.f) * (a - 5.f) + (b - 10.f) * (b - 10.f) + (c - 2.f) * (c - 2.f);
        };
    };

    ParametersToOptimize params = {{"a", {0.f, 0.f, 20.f}}, {"b", {0.f, 0.f, 20.f}}, {"c", {0.f, 0.f, 10.f}}};

    std::vector<BenchmarkResult> results;

    // GA
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        GeneticAlgorithm ga;
        ga.generations = 60;
        ga.population_size = 30;
        popt.setAlgorithm(ga);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"GA", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // BO
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        BayesianOptimization bo;
        bo.max_evaluations = 80;
        bo.initial_samples = 15;
        popt.setAlgorithm(bo);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"BO", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // CMA-ES
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 200;
        popt.setAlgorithm(cmaes);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"CMA-ES", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

#ifdef HELIOS_HAVE_NLOPT
    // LBFGS with exact gradient
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            return {{"a", 2.f * (p.at("a").value - 5.f)},
                    {"b", 2.f * (p.at("b").value - 10.f)},
                    {"c", 2.f * (p.at("c").value - 2.f)}};
        };
        ParameterOptimization popt;
        LBFGS lbfgs;
        lbfgs.max_iterations = 200;
        popt.setAlgorithm(lbfgs);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"LBFGS(exact)", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // LBFGS with FD gradient
    {
        auto sim = make_objective();
        auto fd_source = makeFDGradientSource("all_fd", {"a", "b", "c"}, sim);
        auto grad = makeGradientFunction({fd_source});
        ParameterOptimization popt;
        LBFGS lbfgs;
        lbfgs.max_iterations = 200;
        popt.setAlgorithm(lbfgs);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"LBFGS(FD)", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }
#endif

    // Adam with exact gradient
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            return {{"a", 2.f * (p.at("a").value - 5.f)},
                    {"b", 2.f * (p.at("b").value - 10.f)},
                    {"c", 2.f * (p.at("c").value - 2.f)}};
        };
        ParameterOptimization popt;
        Adam adam; adam.max_iterations = 500; adam.learning_rate = 0.1f;
        popt.setAlgorithm(adam);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"Adam", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

#ifdef HELIOS_HAVE_NLOPT
    // BOBYQA (derivative-free local)
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        BOBYQA bobyqa; bobyqa.max_iterations = 200;
        popt.setAlgorithm(bobyqa);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"BOBYQA", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // SLSQP (gradient-based, unconstrained)
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            return {{"a", 2.f * (p.at("a").value - 5.f)},
                    {"b", 2.f * (p.at("b").value - 10.f)},
                    {"c", 2.f * (p.at("c").value - 2.f)}};
        };
        ParameterOptimization popt;
        SLSQP slsqp; slsqp.max_iterations = 200;
        popt.setAlgorithm(slsqp);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"SLSQP", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }
#endif

    // Print comparison table
    std::cout << "\n  === 3D Quadratic Benchmark ===\n";
    std::cout << "  " << std::left << std::setw(14) << "Algorithm"
              << std::right << std::setw(12) << "Fitness"
              << std::setw(10) << "Time(us)"
              << std::setw(8) << "Evals" << "\n";
    std::cout << "  " << std::string(44, '-') << "\n";
    for (const auto &r : results) {
        std::cout << "  " << std::left << std::setw(14) << r.name
                  << std::right << std::setw(12) << std::scientific << std::setprecision(3) << r.fitness
                  << std::setw(10) << std::fixed << std::setprecision(0) << r.time_us
                  << std::setw(8) << r.evals << "\n";
    }
    std::cout << std::endl;

    // All should converge
    for (const auto &r : results) {
        DOCTEST_INFO(r.name, ": fitness=", r.fitness, " evals=", r.evals, " time=", r.time_us, "us");
        DOCTEST_CHECK(r.fitness < 1.0f);
    }
}

// Test 41: Four-way comparison on 2D Rosenbrock
// f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)
DOCTEST_TEST_CASE("ParameterOptimization four-way comparison: 2D Rosenbrock") {
    int eval_count = 0;

    auto make_objective = [&eval_count]() {
        eval_count = 0;
        return [&eval_count](const ParametersToOptimize &p) -> float {
            eval_count++;
            float x = p.at("x").value;
            float y = p.at("y").value;
            float t1 = (1.f - x);
            float t2 = (y - x * x);
            return t1 * t1 + 100.f * t2 * t2;
        };
    };

    ParametersToOptimize params = {{"x", {-1.f, -2.f, 2.f}}, {"y", {-1.f, -2.f, 2.f}}};

    std::vector<BenchmarkResult> results;

    // GA
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        GeneticAlgorithm ga;
        ga.generations = 200;
        ga.population_size = 40;
        ga.crossover = BLXPCACrossover{};
        HybridMutation hm;
        hm.rate = 0.15f;
        ga.mutation = hm;
        popt.setAlgorithm(ga);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"GA", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // BO
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        BayesianOptimization bo;
        bo.max_evaluations = 100;
        bo.initial_samples = 15;
        popt.setAlgorithm(bo);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"BO", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // CMA-ES
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 500;
        cmaes.sigma = 0.5f;
        popt.setAlgorithm(cmaes);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"CMA-ES", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

#ifdef HELIOS_HAVE_NLOPT
    // LBFGS with exact gradient
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            float x = p.at("x").value;
            float y = p.at("y").value;
            // df/dx = -2(1-x) + 200*(y-x^2)*(-2x) = 2(x-1) - 400*x*(y-x^2)
            // df/dy = 200*(y-x^2)
            return {{"x", 2.f * (x - 1.f) - 400.f * x * (y - x * x)},
                    {"y", 200.f * (y - x * x)}};
        };
        ParameterOptimization popt;
        LBFGS lbfgs;
        lbfgs.max_iterations = 200;
        popt.setAlgorithm(lbfgs);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"LBFGS(exact)", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // LBFGS with FD gradient
    {
        auto sim = make_objective();
        auto fd_source = makeFDGradientSource("all_fd", {"x", "y"}, sim);
        auto grad = makeGradientFunction({fd_source});
        ParameterOptimization popt;
        LBFGS lbfgs;
        lbfgs.max_iterations = 200;
        popt.setAlgorithm(lbfgs);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"LBFGS(FD)", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }
#endif

    // Adam with exact gradient
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            float x = p.at("x").value, y = p.at("y").value;
            return {{"x", 2.f * (x - 1.f) - 400.f * x * (y - x*x)},
                    {"y", 200.f * (y - x*x)}};
        };
        ParameterOptimization popt;
        Adam adam; adam.max_iterations = 1000; adam.learning_rate = 0.01f;
        popt.setAlgorithm(adam);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"Adam", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

#ifdef HELIOS_HAVE_NLOPT
    // BOBYQA
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        BOBYQA bobyqa; bobyqa.max_iterations = 500;
        popt.setAlgorithm(bobyqa);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"BOBYQA", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // SLSQP
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            float x = p.at("x").value, y = p.at("y").value;
            return {{"x", 2.f * (x - 1.f) - 400.f * x * (y - x*x)},
                    {"y", 200.f * (y - x*x)}};
        };
        ParameterOptimization popt;
        SLSQP slsqp; slsqp.max_iterations = 200;
        popt.setAlgorithm(slsqp);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"SLSQP", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }
#endif

    std::cout << "\n  === 2D Rosenbrock Benchmark ===\n";
    std::cout << "  " << std::left << std::setw(14) << "Algorithm"
              << std::right << std::setw(12) << "Fitness"
              << std::setw(10) << "Time(us)"
              << std::setw(8) << "Evals" << "\n";
    std::cout << "  " << std::string(44, '-') << "\n";
    for (const auto &r : results) {
        std::cout << "  " << std::left << std::setw(14) << r.name
                  << std::right << std::setw(12) << std::scientific << std::setprecision(3) << r.fitness
                  << std::setw(10) << std::fixed << std::setprecision(0) << r.time_us
                  << std::setw(8) << r.evals << "\n";
    }
    std::cout << std::endl;

    for (const auto &r : results) {
        DOCTEST_INFO(r.name, ": fitness=", r.fitness, " evals=", r.evals, " time=", r.time_us, "us");
        DOCTEST_CHECK(r.fitness < 5.0f);
    }
}

// Test 42: Four-way comparison on 3D rotated ellipsoid (non-separable)
// f = sum_i (10^(i/2) * z_i)^2 where z = R*x (rotation)
// Tests how well each algorithm handles correlated parameter landscapes.
DOCTEST_TEST_CASE("ParameterOptimization four-way comparison: 3D rotated ellipsoid") {
    int eval_count = 0;

    // 3D rotation: mix all three dimensions
    auto make_objective = [&eval_count]() {
        eval_count = 0;
        return [&eval_count](const ParametersToOptimize &p) -> float {
            eval_count++;
            float x = p.at("x").value;
            float y = p.at("y").value;
            float z = p.at("z").value;
            // Simple rotation: z1=(x+y)/sqrt(2), z2=(x-y)/sqrt(2), z3=z
            float z1 = (x + y) * 0.7071f;
            float z2 = (x - y) * 0.7071f;
            float z3 = z;
            // Condition: 1, 10, 100
            return z1 * z1 + 10.f * z2 * z2 + 100.f * z3 * z3;
        };
    };

    ParametersToOptimize params = {{"x", {3.f, -5.f, 5.f}}, {"y", {-3.f, -5.f, 5.f}}, {"z", {2.f, -5.f, 5.f}}};

    std::vector<BenchmarkResult> results;

    // GA
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        GeneticAlgorithm ga;
        ga.generations = 200;
        ga.population_size = 40;
        ga.crossover = BLXPCACrossover{};
        HybridMutation hm;
        hm.rate = 0.15f;
        ga.mutation = hm;
        popt.setAlgorithm(ga);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"GA", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // BO
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        BayesianOptimization bo;
        bo.max_evaluations = 100;
        bo.initial_samples = 15;
        popt.setAlgorithm(bo);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"BO", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // CMA-ES
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        CMAES cmaes;
        cmaes.max_evaluations = 500;
        cmaes.sigma = 0.5f;
        popt.setAlgorithm(cmaes);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"CMA-ES", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

#ifdef HELIOS_HAVE_NLOPT
    // LBFGS with exact gradient
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            float x = p.at("x").value;
            float y = p.at("y").value;
            float z = p.at("z").value;
            // z1=(x+y)/sqrt(2), z2=(x-y)/sqrt(2), z3=z
            // f = z1^2 + 10*z2^2 + 100*z3^2
            // df/dx = 2*z1*(1/sqrt(2)) + 20*z2*(1/sqrt(2))
            //       = (x+y) + 10*(x-y) = 11x - 9y
            // df/dy = 2*z1*(1/sqrt(2)) + 20*z2*(-1/sqrt(2))
            //       = (x+y) - 10*(x-y) = -9x + 11y
            // df/dz = 200*z
            return {{"x", 11.f * x - 9.f * y},
                    {"y", -9.f * x + 11.f * y},
                    {"z", 200.f * z}};
        };
        ParameterOptimization popt;
        LBFGS lbfgs;
        lbfgs.max_iterations = 200;
        popt.setAlgorithm(lbfgs);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"LBFGS(exact)", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // LBFGS with FD gradient
    {
        auto sim = make_objective();
        auto fd_source = makeFDGradientSource("all_fd", {"x", "y", "z"}, sim);
        auto grad = makeGradientFunction({fd_source});
        ParameterOptimization popt;
        LBFGS lbfgs;
        lbfgs.max_iterations = 200;
        popt.setAlgorithm(lbfgs);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"LBFGS(FD)", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }
#endif

    // Adam with exact gradient
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            float x = p.at("x").value, y = p.at("y").value, z = p.at("z").value;
            return {{"x", 11.f * x - 9.f * y},
                    {"y", -9.f * x + 11.f * y},
                    {"z", 200.f * z}};
        };
        ParameterOptimization popt;
        Adam adam; adam.max_iterations = 500; adam.learning_rate = 0.05f;
        popt.setAlgorithm(adam);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"Adam", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

#ifdef HELIOS_HAVE_NLOPT
    // BOBYQA
    {
        auto sim = make_objective();
        ParameterOptimization popt;
        BOBYQA bobyqa; bobyqa.max_iterations = 500;
        popt.setAlgorithm(bobyqa);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"BOBYQA", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }

    // SLSQP
    {
        auto sim = make_objective();
        GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
            float x = p.at("x").value, y = p.at("y").value, z = p.at("z").value;
            return {{"x", 11.f * x - 9.f * y},
                    {"y", -9.f * x + 11.f * y},
                    {"z", 200.f * z}};
        };
        ParameterOptimization popt;
        SLSQP slsqp; slsqp.max_iterations = 200;
        popt.setAlgorithm(slsqp);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = popt.run(sim, params, grad);
        auto t1 = std::chrono::high_resolution_clock::now();
        results.push_back({"SLSQP", r.fitness, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), eval_count});
    }
#endif

    std::cout << "\n  === 3D Rotated Ellipsoid Benchmark ===\n";
    std::cout << "  " << std::left << std::setw(14) << "Algorithm"
              << std::right << std::setw(12) << "Fitness"
              << std::setw(10) << "Time(us)"
              << std::setw(8) << "Evals" << "\n";
    std::cout << "  " << std::string(44, '-') << "\n";
    for (const auto &r : results) {
        std::cout << "  " << std::left << std::setw(14) << r.name
                  << std::right << std::setw(12) << std::scientific << std::setprecision(3) << r.fitness
                  << std::setw(10) << std::fixed << std::setprecision(0) << r.time_us
                  << std::setw(8) << r.evals << "\n";
    }
    std::cout << std::endl;

    for (const auto &r : results) {
        DOCTEST_INFO(r.name, ": fitness=", r.fitness, " evals=", r.evals, " time=", r.time_us, "us");
        DOCTEST_CHECK(r.fitness < 2.0f);
    }
}

// ============================================================================
// Explore/Exploit Preset Tests
// ============================================================================

// Shared quadratic for preset tests
static float preset_quadratic(const ParametersToOptimize &p) {
    float x = p.at("x").value, y = p.at("y").value;
    return (x - 2.f) * (x - 2.f) + (y - 3.f) * (y - 3.f);
}

// Test 54: GA explore() preset converges
DOCTEST_TEST_CASE("ParameterOptimization GA explore preset") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    popt.setAlgorithm(GeneticAlgorithm::explore());
    auto result = popt.run(preset_quadratic, params);
    DOCTEST_CHECK(result.fitness < 0.1f);
}

// Test 55: GA exploit() preset converges
DOCTEST_TEST_CASE("ParameterOptimization GA exploit preset") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    popt.setAlgorithm(GeneticAlgorithm::exploit());
    auto result = popt.run(preset_quadratic, params);
    DOCTEST_CHECK(result.fitness < 0.1f);
}

// Test 56: GA explore vs exploit parameter sanity
DOCTEST_TEST_CASE("ParameterOptimization GA explore vs exploit parameters") {
    auto e = GeneticAlgorithm::explore();
    auto x = GeneticAlgorithm::exploit();
    DOCTEST_CHECK(e.population_size >= x.population_size);
    DOCTEST_CHECK(e.crossover_rate >= x.crossover_rate);
    DOCTEST_CHECK(e.elitism_rate <= x.elitism_rate);
}

// Test 57: BO explore() preset converges
DOCTEST_TEST_CASE("ParameterOptimization BO explore preset") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    popt.setAlgorithm(BayesianOptimization::explore());
    auto result = popt.run(preset_quadratic, params);
    DOCTEST_CHECK(result.fitness < 1.0f);
}

// Test 58: BO exploit() preset converges
DOCTEST_TEST_CASE("ParameterOptimization BO exploit preset") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    popt.setAlgorithm(BayesianOptimization::exploit());
    auto result = popt.run(preset_quadratic, params);
    DOCTEST_CHECK(result.fitness < 1.0f);
}

// Test 59: BO explore vs exploit parameter sanity
DOCTEST_TEST_CASE("ParameterOptimization BO explore vs exploit parameters") {
    auto e = BayesianOptimization::explore();
    auto x = BayesianOptimization::exploit();
    DOCTEST_CHECK(e.ucb_kappa > x.ucb_kappa);
    DOCTEST_CHECK(e.max_evaluations >= x.max_evaluations);
}

// Test 60: CMA-ES explore() preset converges
DOCTEST_TEST_CASE("ParameterOptimization CMAES explore preset") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    popt.setAlgorithm(CMAES::explore());
    auto result = popt.run(preset_quadratic, params);
    DOCTEST_CHECK(result.fitness < 0.1f);
}

// Test 61: CMA-ES exploit() preset converges
DOCTEST_TEST_CASE("ParameterOptimization CMAES exploit preset") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    ParameterOptimization popt;
    popt.setAlgorithm(CMAES::exploit());
    auto result = popt.run(preset_quadratic, params);
    DOCTEST_CHECK(result.fitness < 0.5f);
}

// Test 62: CMA-ES explore vs exploit parameter sanity
DOCTEST_TEST_CASE("ParameterOptimization CMAES explore vs exploit parameters") {
    auto e = CMAES::explore();
    auto x = CMAES::exploit();
    DOCTEST_CHECK(e.sigma > x.sigma);
    DOCTEST_CHECK(e.max_evaluations >= x.max_evaluations);
}

// ============================================================================
// I/O Tests
// ============================================================================

// Test 73: write_result_to_file produces valid CSV
DOCTEST_TEST_CASE("ParameterOptimization write_result_to_file") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    std::string tmpfile = "paramopt_test_result.csv";

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 5;
    ga.population_size = 10;
    popt.setAlgorithm(ga);
    popt.write_result_to_file = tmpfile;

    popt.run(preset_quadratic, params);

    // Verify file exists and has expected format
    std::ifstream f(tmpfile);
    DOCTEST_REQUIRE(f.is_open());

    std::string header;
    std::getline(f, header);
    DOCTEST_CHECK(header.find("parameter") != std::string::npos);
    DOCTEST_CHECK(header.find("value") != std::string::npos);

    int row_count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) row_count++;
    }
    DOCTEST_CHECK(row_count == 2);  // x and y

    f.close();
    std::remove(tmpfile.c_str());
}

// Test 74: write_progress_to_file produces valid CSV
DOCTEST_TEST_CASE("ParameterOptimization write_progress_to_file") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};
    std::string tmpfile = "paramopt_test_progress.csv";

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 5;
    ga.population_size = 10;
    popt.setAlgorithm(ga);
    popt.write_progress_to_file = tmpfile;

    popt.run(preset_quadratic, params);

    std::ifstream f(tmpfile);
    DOCTEST_REQUIRE(f.is_open());

    std::string header;
    std::getline(f, header);
    DOCTEST_CHECK(header.find("generation") != std::string::npos);
    DOCTEST_CHECK(header.find("fitness") != std::string::npos);

    int row_count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) row_count++;
    }
    DOCTEST_CHECK(row_count > 0);

    f.close();
    std::remove(tmpfile.c_str());
}

// Test 75: read_input_from_file seeds GA with known parameters
DOCTEST_TEST_CASE("ParameterOptimization read_input_from_file") {
    std::string tmpfile = "paramopt_test_input.csv";

    // Write a parameter file near the optimum
    {
        std::ofstream out(tmpfile);
        out << "x,1.9,-5,5\n";
        out << "y,3.1,-5,5\n";
    }

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    GeneticAlgorithm ga;
    ga.generations = 5;
    ga.population_size = 10;
    popt.setAlgorithm(ga);
    popt.read_input_from_file = tmpfile;

    auto result = popt.run(preset_quadratic, params);
    // Starting near optimum (1.9, 3.1), even 5 generations should yield good result
    DOCTEST_CHECK(result.fitness < 0.5f);

    std::remove(tmpfile.c_str());
}

// ============================================================================
// SLSQP Multi-Constraint Test
// ============================================================================

#ifdef HELIOS_HAVE_NLOPT
// Test 76: SLSQP with two simultaneous inequality constraints
// Minimize x^2 + y^2  subject to: x + y >= 2 (c1: 2-x-y <= 0), y >= 1.5 (c2: 1.5-y <= 0)
// Without c2: optimum is x=y=1, f=2. With c2: y is forced to 1.5, x=0.5, f=0.25+2.25=2.5
DOCTEST_TEST_CASE("ParameterOptimization SLSQP with multiple constraints") {
    ObjectiveFunction objective = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return x * x + y * y;
    };
    GradientFunction gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 2.f * p.at("y").value}};
    };

    // Constraint 1: x + y >= 2  =>  2 - x - y <= 0
    Constraint c1;
    c1.function = [](const ParametersToOptimize &p) -> float {
        return 2.f - p.at("x").value - p.at("y").value;
    };
    c1.gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", -1.f}, {"y", -1.f}};
    };

    // Constraint 2: y >= 1.5  =>  1.5 - y <= 0
    Constraint c2;
    c2.function = [](const ParametersToOptimize &p) -> float {
        return 1.5f - p.at("y").value;
    };
    c2.gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 0.f}, {"y", -1.f}};
    };

    ParametersToOptimize params = {{"x", {0.1f, -5.f, 5.f}}, {"y", {0.1f, -5.f, 5.f}}};

    ParameterOptimization popt;
    SLSQP slsqp;
    popt.setAlgorithm(slsqp);

    auto result = popt.run(objective, params, gradient, {c1, c2});

    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(0.5f).epsilon(0.05f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(1.5f).epsilon(0.05f));
    DOCTEST_CHECK(result.fitness == doctest::Approx(2.5f).epsilon(0.1f));

    // Verify both constraints are satisfied
    float c1_val = 2.f - result.parameters.at("x").value - result.parameters.at("y").value;
    float c2_val = 1.5f - result.parameters.at("y").value;
    DOCTEST_CHECK(c1_val <= 1e-4f);
    DOCTEST_CHECK(c2_val <= 1e-4f);
}
#endif

// ============================================================================
// Adam Weight Decay (AdamW) Test
// ============================================================================

// Test 77: Adam with weight_decay pulls optimum toward zero
DOCTEST_TEST_CASE("ParameterOptimization Adam weight decay (AdamW)") {
    // f(x) = (x-5)^2, true minimum at x=5
    // With L2 penalty: f(x) + lambda*x^2, minimum shifts toward zero
    ObjectiveFunction obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        return (x - 5.f) * (x - 5.f);
    };
    GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * (p.at("x").value - 5.f)}};
    };

    ParametersToOptimize params = {{"x", {0.f, -10.f, 10.f}}};

    // Run without weight decay
    ParameterOptimization popt_nodecay;
    Adam adam_nodecay;
    adam_nodecay.max_iterations = 500;
    adam_nodecay.learning_rate = 0.05f;
    adam_nodecay.weight_decay = 0.f;
    popt_nodecay.setAlgorithm(adam_nodecay);
    auto result_nodecay = popt_nodecay.run(obj, params, grad);

    // Run with weight decay
    ParameterOptimization popt_decay;
    Adam adam_decay;
    adam_decay.max_iterations = 500;
    adam_decay.learning_rate = 0.05f;
    adam_decay.weight_decay = 0.5f;
    popt_decay.setAlgorithm(adam_decay);
    auto result_decay = popt_decay.run(obj, params, grad);

    // Without decay: should converge near x=5
    DOCTEST_CHECK(result_nodecay.parameters.at("x").value == doctest::Approx(5.f).epsilon(0.1f));

    // With decay: optimum should be pulled toward zero (x < 5)
    DOCTEST_CHECK(result_decay.parameters.at("x").value < result_nodecay.parameters.at("x").value);
}

// ============================================================================
// LBFGS Gradient Verification Test
// ============================================================================

#ifdef HELIOS_HAVE_NLOPT
// Test 78: LBFGS with verify_gradients enabled converges normally
DOCTEST_TEST_CASE("ParameterOptimization LBFGS verify_gradients") {
    ObjectiveFunction obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 1.f) * (x - 1.f) + (y - 2.f) * (y - 2.f);
    };
    GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * (p.at("x").value - 1.f)}, {"y", 2.f * (p.at("y").value - 2.f)}};
    };

    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ParameterOptimization popt;
    LBFGS lbfgs;
    lbfgs.verify_gradients = true;
    lbfgs.fd_step = 1e-5;
    popt.setAlgorithm(lbfgs);

    auto result = popt.run(obj, params, grad);
    DOCTEST_CHECK(result.fitness < 1e-6f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(1.f).epsilon(0.01f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(2.f).epsilon(0.01f));
}
#endif

// ============================================================================
// Default Algorithm Selection Tests
// ============================================================================

// Test: No setAlgorithm + continuous FLOAT params → CMA-ES default
DOCTEST_TEST_CASE("ParameterOptimization default algorithm continuous derivative-free") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    auto obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 2.f) * (x - 2.f) + (y - 1.f) * (y - 1.f);
    };

    bool converged = retry(3, [&]() {
        ParameterOptimization popt;
        // No setAlgorithm — should pick CMA-ES for continuous params
        auto result = popt.run(obj, params);
        return result.fitness < 1.0f;
    });
    DOCTEST_CHECK(converged);
}

// Test: No setAlgorithm + INTEGER param → GA default
DOCTEST_TEST_CASE("ParameterOptimization default algorithm discrete derivative-free") {
    ParametersToOptimize params = {
        {"x", {0.f, -5.f, 5.f, ParameterType::INTEGER}},
        {"y", {0.f, -5.f, 5.f}}
    };

    auto obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y - 1.f) * (y - 1.f);
    };

    bool converged = retry(3, [&]() {
        ParameterOptimization popt;
        // No setAlgorithm — should pick GA for discrete params
        auto result = popt.run(obj, params);
        return result.fitness < 2.0f;
    });
    DOCTEST_CHECK(converged);
}

// Test: No setAlgorithm + gradient provided + continuous → Adam default
DOCTEST_TEST_CASE("ParameterOptimization default algorithm gradient") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ObjectiveFunction obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 1.f) * (x - 1.f) + (y + 2.f) * (y + 2.f);
    };
    GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * (p.at("x").value - 1.f)}, {"y", 2.f * (p.at("y").value + 2.f)}};
    };

    ParameterOptimization popt;
    // No setAlgorithm — should pick Adam for gradient + continuous
    auto result = popt.run(obj, params, grad);
    DOCTEST_CHECK(result.fitness < 1.0f);
}

// Test: No setAlgorithm + gradient + constraints + continuous → SLSQP default
#ifdef HELIOS_HAVE_NLOPT
DOCTEST_TEST_CASE("ParameterOptimization default algorithm constrained") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ObjectiveFunction obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return x * x + y * y;
    };
    GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 2.f * p.at("y").value}};
    };

    // Constraint: x + y >= 1 → 1 - x - y <= 0
    Constraint c;
    c.function = [](const ParametersToOptimize &p) -> float {
        return 1.f - p.at("x").value - p.at("y").value;
    };
    c.gradient = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", -1.f}, {"y", -1.f}};
    };

    ParameterOptimization popt;
    // No setAlgorithm — should pick SLSQP for constrained + continuous
    auto result = popt.run(obj, params, grad, {c});
    DOCTEST_CHECK(result.fitness < 1.0f);
    // Optimal: x = y = 0.5, f = 0.5
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(0.5f).epsilon(0.1f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(0.5f).epsilon(0.1f));
}
#endif

// Test: Discrete + constraints → runtime error
DOCTEST_TEST_CASE("ParameterOptimization default algorithm discrete constrained throws") {
    ParametersToOptimize params = {
        {"x", {0.f, -5.f, 5.f, ParameterType::INTEGER}},
        {"y", {0.f, -5.f, 5.f}}
    };

    ObjectiveFunction obj = [](const ParametersToOptimize &p) {
        return p.at("x").value * p.at("x").value;
    };
    GradientFunction grad = [](const ParametersToOptimize &p) -> ParameterGradient {
        return {{"x", 2.f * p.at("x").value}, {"y", 0.f}};
    };
    Constraint c;
    c.function = [](const ParametersToOptimize &p) -> float { return -p.at("x").value; };
    c.gradient = [](const ParametersToOptimize &p) -> ParameterGradient { return {{"x", -1.f}, {"y", 0.f}}; };

    ParameterOptimization popt;
    // No setAlgorithm — discrete + constraints should throw
    DOCTEST_CHECK_THROWS(popt.run(obj, params, grad, {c}));
}

// ============================================================================
// makeFDGradient convenience wrapper test
// ============================================================================

// Test: makeFDGradient produces correct gradients and converges with Adam
DOCTEST_TEST_CASE("ParameterOptimization makeFDGradient convenience") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    ObjectiveFunction obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return (x - 2.f) * (x - 2.f) + (y + 1.f) * (y + 1.f);
    };

    // Use the convenience wrapper — no parameter names, no label needed
    auto gradient = makeFDGradient(obj);

    ParameterOptimization popt;
    Adam adam;
    adam.max_iterations = 500;
    adam.learning_rate = 0.05f;
    popt.setAlgorithm(adam);
    auto result = popt.run(obj, params, gradient);

    DOCTEST_CHECK(result.fitness < 0.5f);
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(2.f).epsilon(0.5f));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(0.5f));
}

// Test: makeFDGradient matches makeFDGradientSource output
DOCTEST_TEST_CASE("ParameterOptimization makeFDGradient matches makeFDGradientSource") {
    ParametersToOptimize params = {{"x", {3.f, -10.f, 10.f}}, {"y", {-2.f, -10.f, 10.f}}};

    ObjectiveFunction obj = [](const ParametersToOptimize &p) {
        float x = p.at("x").value, y = p.at("y").value;
        return x * x * x + 2.f * y * y;
    };

    // Convenience wrapper
    auto grad_convenience = makeFDGradient(obj);
    auto g1 = grad_convenience(params);

    // Explicit source
    auto source = makeFDGradientSource("test", {"x", "y"}, obj);
    auto grad_explicit = makeGradientFunction({source});
    auto g2 = grad_explicit(params);

    DOCTEST_CHECK(g1.at("x") == doctest::Approx(g2.at("x")).epsilon(1e-6f));
    DOCTEST_CHECK(g1.at("y") == doctest::Approx(g2.at("y")).epsilon(1e-6f));
}
