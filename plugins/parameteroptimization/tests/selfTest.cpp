/**
 * \file "selfTest.cpp" Automated tests for the Parameter Optimization plug-in.

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

using namespace helios;

int ParameterOptimization::selfTest() {

    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) { // important - query flags (and --exit) rely on this
        return res; // propagate the result of the tests
    }

    return res;
}

// Test 1: ParameterOptimization Simple Quadratic Optimization
// simple quadratic fitness function f(x) = (x-3)^2 + (y+1)^2
DOCTEST_TEST_CASE("ParameterOptimization quadratic") {
    ParametersToOptimize params = {{"x", {0.f, -5.f, 5.f}}, {"y", {0.f, -5.f, 5.f}}};

    auto sim = [](const ParametersToOptimize &p) {
        float x = p.at("x").value;
        float y = p.at("y").value;
        return (x - 3.f) * (x - 3.f) + (y + 1.f) * (y + 1.f);
    };

    OptimizationSettings settings;
    settings.generations = 100;
    settings.population_size = 100;

    ParameterOptimization popt;
    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params, settings));

    DOCTEST_CHECK(result.fitness < 1e-2);
}

// Test 2: ParameterOptimization Simple Quadratic Optimization With Primitive Data Access
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

    OptimizationSettings settings;
    settings.generations = 100;
    settings.population_size = 500;

    ParameterOptimization popt;
    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params, settings));

    float tol = 1e-1f;
    DOCTEST_CHECK(result.parameters.at("x").value == doctest::Approx(3.f).epsilon(tol));
    DOCTEST_CHECK(result.parameters.at("y").value == doctest::Approx(-1.f).epsilon(tol));
}

// Test 3
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
    OptimizationSettings settings;
    settings.population_size = 100;
    settings.generations = 100;
    settings.crossover_rate = 0.5;
    settings.mutation_rate = 0.1;

    ParameterOptimization::Result result;
    DOCTEST_CHECK_NOTHROW(result = popt.run(sim, params, settings));

    DOCTEST_CHECK(result.fitness < 1e-3f);
}
