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

    ParameterOptimization popt;
    BayesianOptimization bo;
    bo.max_evaluations = 40;
    bo.initial_samples = 10;
    bo.acquisition_samples = 1000;
    popt.setAlgorithm(bo);

    auto result = popt.run(sim, params);

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
