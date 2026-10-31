#!/usr/bin/env python3
"""Generate plugins/parameteroptimization/tests/TPEGoldenData.h from real Optuna.

The golden values let the plugin's self-tests compare the C++ TPE port against upstream without
needing Python. Run with the pinned environment in requirements.txt:

    python generate_tpe_fixtures.py
"""

import os
import platform

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import TPESampler
from optuna.samplers._tpe import _truncnorm
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
from optuna.samplers._tpe.sampler import default_gamma, default_weights

assert optuna.__version__ == "5.0.0", "Fixtures must be generated from the pinned Optuna version."
assert platform.machine().lower() not in ("x86_64", "amd64"), "np.argsort may use a SIMD sort on x86 CPUs, which orders ties differently. Generate fixtures where it uses its generic quicksort, such as Apple silicon."

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins", "parameteroptimization", "tests", "TPEGoldenData.h")
PARZEN_PARAMETERS = _ParzenEstimatorParameters(1.0, True, False, default_weights, True)


def number(value):
    value = float(value)
    if np.isposinf(value):
        return "std::numeric_limits<double>::infinity()"
    if np.isneginf(value):
        return "-std::numeric_limits<double>::infinity()"
    return repr(value) if ("." in repr(value) or "e" in repr(value)) else repr(value) + ".0"


def vector(values):
    return "{" + ", ".join(number(v) for v in values) + "}"


def matrix(rows):
    return "{" + ", ".join(vector(r) for r in rows) + "}"


def table(name, rows):
    lines = [f"    const std::vector<std::vector<double>> {name} = {{"]
    lines += ["        " + vector(row) + "," for row in rows]
    lines.append("    };")
    return "\n".join(lines)


def scalar_tables():
    a_values = [-40.0, -25.0, -20.5, -19.0, -8.0, -1.5, -0.3, 0.0, 0.4, 2.0, 5.9, 6.1, 12.0]
    log_ndtr = [[a, _truncnorm._log_ndtr(np.array([a]))[0]] for a in a_values]

    intervals = [(-30.0, -25.0), (-5.0, -2.0), (-3.0, 0.0), (-2.0, 1.5), (-0.5, 0.5), (-8.0, 8.0), (0.0, 3.0), (1.0, 2.0), (4.0, 9.0), (25.0, 30.0)]
    log_mass = [[a, b, _truncnorm._log_gauss_mass(np.array([a]), np.array([b]))[0]] for a, b in intervals]

    y_values = [-500.0, -50.0, -6.0, -5.0, -1.0, -0.6931, -0.1, -1e-2, -1e-3, -1e-8]
    ndtri = [[y, _truncnorm._ndtri_exp(np.array([y]))[0]] for y in y_values]

    ppf_rows = []
    for a, b in [(-2.0, 1.5), (-10.0, -4.0), (0.5, 3.0), (4.0, 9.0), (-0.1, 0.1), (-30.0, 30.0)]:
        for q in [0.0, 1e-6, 0.1, 0.5, 0.9, 1 - 1e-6, 1.0]:
            ppf_rows.append([q, a, b, _truncnorm.ppf(np.array([q]), np.array([a]), np.array([b]))[0]])

    logpdf_rows = []
    for x, a, b, loc, scale in [(0.3, -2.0, 2.0, 0.0, 1.0), (5.0, -1.0, 3.0, 2.0, 1.5), (-3.0, -4.0, -1.0, 1.0, 2.0), (2.0, 0.5, 8.0, 0.0, 0.25), (9.0, -1.0, 1.0, 0.0, 1.0)]:
        logpdf_rows.append([x, a, b, loc, scale, _truncnorm.logpdf(np.array([x]), np.array([a]), np.array([b]), loc, scale)[0]])

    return "\n".join([table("log_ndtr", log_ndtr), table("log_gauss_mass", log_mass), table("ndtri_exp", ndtri), table("truncnorm_ppf", ppf_rows), table("truncnorm_logpdf", logpdf_rows)])


def argsort_tables():
    """Tied values at sizes spanning the insertion sort threshold, and organ-pipe orderings, the second of which is known to reach the depth-limit heapsort."""
    rng = np.random.RandomState(4)
    rows = []
    for n in (3, 16, 17, 18, 40, 150):
        for cardinality in (2, 6, 1000):
            rows.append(rng.randint(0, cardinality, n).astype(float).tolist())
    values = np.sort(rng.randint(0, 16, 1024).astype(float))
    rows.append(np.concatenate([values[::2], values[1::2][::-1]]).tolist())
    values = np.sort(np.random.RandomState(2).randint(0, 1000, 160).astype(float))
    rows.append(np.concatenate([values[::2], values[1::2][::-1]]).tolist())
    order = [np.argsort(np.array(row)).tolist() for row in rows]
    return table("argsort_values", rows) + "\n" + table("argsort_order", order)


def weight_tables():
    gamma = [[n, default_gamma(n)] for n in [1, 5, 10, 11, 30, 100, 249, 250, 251, 1000]]
    weights = ["    const std::vector<std::vector<double>> default_weights = {"]
    for n in [0, 1, 24, 25, 26, 27, 30, 60]:
        weights.append("        " + vector(default_weights(n)) + ",")
    weights.append("    };")
    return table("default_gamma", gamma) + "\n" + "\n".join(weights)


def estimator_cases():
    rng = np.random.RandomState(20261031)
    spaces = [
        ("continuous", {"a": FloatDistribution(-5.0, 10.0), "b": FloatDistribution(0.0, 1.0)}),
        ("integer", {"a": IntDistribution(-3, 12), "b": FloatDistribution(0.0, 1.0)}),
        ("categorical", {"a": CategoricalDistribution([0, 1, 2, 3]), "b": FloatDistribution(-1.0, 1.0)}),
        ("tied_integers", {"a": IntDistribution(0, 3), "b": FloatDistribution(0.0, 1.0)}),
        ("mixed", {"a": CategoricalDistribution([0, 1, 2]), "b": IntDistribution(0, 7), "c": FloatDistribution(-2.0, 3.0), "d": CategoricalDistribution([0, 1])}),
    ]
    blocks = []
    for name, space in spaces:
        for n_observations in [0, 1, 2, 7, 40]:
            columns = {}
            for param, dist in space.items():
                if isinstance(dist, CategoricalDistribution):
                    columns[param] = rng.randint(0, len(dist.choices), n_observations).astype(float)
                elif isinstance(dist, IntDistribution):
                    columns[param] = rng.randint(dist.low, dist.high + 1, n_observations).astype(float)
                else:
                    columns[param] = dist.low + rng.rand(n_observations) * (dist.high - dist.low)
            estimator = _ParzenEstimator(columns, space, PARZEN_PARAMETERS)
            sample_seed = 7 + n_observations
            samples = estimator.sample(np.random.RandomState(sample_seed), 12)
            log_pdf = estimator.log_pdf(samples)
            observations = [[columns[p][i] for p in space] for i in range(n_observations)]
            dimensions = []
            for dist in space.values():
                if isinstance(dist, CategoricalDistribution):
                    dimensions.append(f"{{helios_tpe::DimensionKind::CATEGORICAL, 0.0, 0.0, {len(dist.choices)}}}")
                elif isinstance(dist, IntDistribution):
                    dimensions.append(f"{{helios_tpe::DimensionKind::INTEGER, {number(dist.low)}, {number(dist.high)}, 0}}")
                else:
                    dimensions.append(f"{{helios_tpe::DimensionKind::CONTINUOUS, {number(dist.low)}, {number(dist.high)}, 0}}")
            mus, sigmas = [], []
            categorical = []
            for d in estimator._mixture_distribution.distributions:
                if hasattr(d, "weights"):
                    mus.append([])
                    sigmas.append([])
                    categorical.append(d.weights.tolist())
                else:
                    mus.append(d.mu.tolist())
                    sigmas.append(d.sigma.tolist())
                    categorical.append([])
            sample_rows = [[samples[p][i] for p in space] for i in range(len(log_pdf))]
            blocks.append(
                "        {\n"
                f'            "{name}_{n_observations}",\n'
                f"            {{{', '.join(dimensions)}}},\n"
                f"            {matrix(observations)},\n"
                f"            {vector(estimator._mixture_distribution.weights)},\n"
                f"            {matrix(mus)},\n"
                f"            {matrix(sigmas)},\n"
                f"            {{{', '.join(matrix(c) for c in categorical)}}},\n"
                f"            {sample_seed},\n"
                f"            {matrix(sample_rows)},\n"
                f"            {vector(log_pdf)},\n"
                "        },"
            )
    return "    const std::vector<EstimatorCase> estimator_cases = {\n" + "\n".join(blocks) + "\n    };"


def run_trajectory(space, centers, seed, n_trials):
    """Full study through the public Optuna API, with values passing through float32 as in Helios."""
    evaluated = []

    def objective(trial):
        row = []
        for name in sorted(space):
            dist = space[name]
            if isinstance(dist, CategoricalDistribution):
                row.append(float(trial.suggest_categorical(name, dist.choices)))
            elif isinstance(dist, IntDistribution):
                row.append(float(trial.suggest_int(name, dist.low, dist.high)))
            else:
                row.append(float(np.float32(trial.suggest_float(name, dist.low, dist.high))))
        evaluated.append(row)
        return float(np.float32(sum((value - center) ** 2 for value, center in zip(row, centers))))

    study = optuna.create_study(sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return evaluated


def trajectory_cases():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cases = [
        ("continuous_2d", {"x0": FloatDistribution(-5.0, 10.0), "x1": FloatDistribution(0.0, 15.0)}, [2.5, 11.0], 3, 60),
        ("continuous_5d", {f"x{i}": FloatDistribution(-2.0, 2.0) for i in range(5)}, [0.7, -1.1, 0.2, 1.4, -0.5], 11, 60),
        ("categorical_float", {"c": CategoricalDistribution([5.0, 12.0, 20.0, 31.0]), "x": FloatDistribution(0.0, 1.0)}, [20.0, 0.3], 5, 40),
        ("integer_float", {"n0": IntDistribution(-5, 10), "n1": IntDistribution(0, 4), "x": FloatDistribution(0.0, 1.0)}, [7.0, 3.0, 0.3], 8, 120),
    ]
    blocks = []
    for name, space, centers, seed, n_trials in cases:
        lows = []
        for key in sorted(space):
            dist = space[key]
            if isinstance(dist, CategoricalDistribution):
                lows.append(f'{{"{key}", {{{number(dist.choices[0])}f, 0.f, 0.f, ParameterType::CATEGORICAL, {{{", ".join(number(c) + "f" for c in dist.choices)}}}}}}}')
            elif isinstance(dist, IntDistribution):
                lows.append(f'{{"{key}", {{{number(dist.low)}f, {number(dist.low)}f, {number(dist.high)}f, ParameterType::INTEGER}}}}')
            else:
                lows.append(f'{{"{key}", {{{number(dist.low)}f, {number(dist.low)}f, {number(dist.high)}f}}}}')
        trajectory = run_trajectory(space, centers, seed, n_trials)
        blocks.append(
            "        {\n"
            f'            "{name}",\n'
            f"            {{{', '.join(lows)}}},\n"
            f"            {vector(centers)},\n"
            f"            {seed},\n"
            f"            {matrix(trajectory)},\n"
            "        },"
        )
    return "    const std::vector<TrajectoryCase> trajectory_cases = {\n" + "\n".join(blocks) + "\n    };"


HEADER = """// Generated by benchmarks/parameter_optimization/generate_tpe_fixtures.py from Optuna {version}. Do not edit by hand.

#ifndef TPE_GOLDEN_DATA_H
#define TPE_GOLDEN_DATA_H

#include "ParameterOptimization.h"
#include "TPESampler.h"

namespace tpe_golden {{

    //! Parzen estimator fitted by Optuna, with the kernels it produced and a seeded sample drawn from it
    struct EstimatorCase {{
        const char *name;
        std::vector<helios_tpe::SearchDimension> dimensions;
        std::vector<std::vector<double>> observations;
        std::vector<double> weights;
        std::vector<std::vector<double>> mus; //!< Per dimension; empty for categorical dimensions
        std::vector<std::vector<double>> sigmas; //!< Per dimension; empty for categorical dimensions
        std::vector<std::vector<std::vector<double>>> category_weights; //!< Per dimension; empty for numerical dimensions
        unsigned int sample_seed;
        std::vector<std::vector<double>> samples;
        std::vector<double> log_pdf;
    }};

    //! Full Optuna study: the parameter values evaluated on each trial, in sorted parameter name order
    struct TrajectoryCase {{
        const char *name;
        ParametersToOptimize parameters;
        std::vector<double> centers; //!< The objective is the sum of squared distances from these
        unsigned int seed;
        std::vector<std::vector<double>> evaluated;
    }};

{body}

}} // namespace tpe_golden

#endif
"""

body = "\n\n".join([scalar_tables(), argsort_tables(), weight_tables(), estimator_cases(), trajectory_cases()])
with open(OUTPUT, "w") as output:
    output.write(HEADER.format(version=optuna.__version__, body=body))
print("Wrote", os.path.normpath(OUTPUT), os.path.getsize(OUTPUT), "bytes")
