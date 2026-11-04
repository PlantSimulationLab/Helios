#!/usr/bin/env python3
"""Run the benchmark problems through Optuna, writing the same CSV files as main.cpp.

    python run_optuna.py quality <out.csv> [--seeds 30] [--budget 200] [--problems a,b]
    python run_optuna.py trajectory <out.csv> [--seeds 20] [--budget 200] [--problems a,b]
    python run_optuna.py overhead <out.csv>

Trials match the C++ port exactly only where np.argsort uses its generic quicksort. As far as we know, NumPy on x86
CPUs with AVX2 or AVX-512 dispatches to a SIMD sort, which orders ties differently, so INTEGER results would differ there.
"""

import argparse
import csv
import time

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import TPESampler
from optuna.trial import TrialState, create_trial

from problems import build_problems

assert optuna.__version__ == "5.0.0", "The benchmark compares against the Optuna version that was ported."
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial, problem):
    """Parameter values as the Helios API would pass them: float32, in sorted name order."""
    x = []
    for spec in problem.specs:
        if spec.kind == "categorical":
            x.append(float(trial.suggest_categorical(spec.name, spec.categories)))
        elif spec.kind == "int":
            x.append(float(trial.suggest_int(spec.name, int(spec.low), int(spec.high))))
        else:
            x.append(float(np.float32(trial.suggest_float(spec.name, spec.low, spec.high))))
    return x


def run_quality(problems, path, n_seeds, budget):
    with open(path, "w", newline="") as output, open(path + ".timing.csv", "w", newline="") as timing_output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["optimizer", "problem", "seed", "evaluation", "best_regret"])
        timing = csv.writer(timing_output, lineterminator="\n")
        timing.writerow(["optimizer", "problem", "seed", "evaluations", "total_s", "objective_s"])
        for problem in problems:
            problem_start = time.perf_counter()
            for seed in range(1, n_seeds + 1):
                noise = np.random.RandomState(seed * 1000003)
                state = {"evaluation": 0, "best_observed": float("inf"), "true_at_best_observed": float("inf"), "objective_s": 0.0}

                def objective(trial):
                    x = suggest(trial, problem)
                    start = time.perf_counter()
                    f_true = problem.function(x)
                    observed = float(np.float32(f_true + (noise.normal(0.0, problem.noise_sd) if problem.noise_sd > 0.0 else 0.0)))
                    # Regret is that of the point the optimizer would return: the one with the best noisy value, scored on the noiseless function
                    if observed < state["best_observed"]:
                        state["best_observed"] = observed
                        state["true_at_best_observed"] = f_true
                    state["evaluation"] += 1
                    n = state["evaluation"]
                    if n == 10 or n == 25 or n % 50 == 0 or n == budget:
                        writer.writerow(["Optuna", problem.name, seed, n, repr(state["true_at_best_observed"] - problem.f_star)])
                    state["objective_s"] += time.perf_counter() - start
                    return observed

                study = optuna.create_study(sampler=TPESampler(seed=seed))
                start = time.perf_counter()
                study.optimize(objective, n_trials=budget)
                timing.writerow(["Optuna", problem.name, seed, state["evaluation"], time.perf_counter() - start, state["objective_s"]])
            output.flush()
            timing_output.flush()
            print(f"Optuna {problem.name}: {time.perf_counter() - problem_start:.1f} s", flush=True)


def run_trajectory(problems, path, n_seeds, budget):
    with open(path, "w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["problem", "seed", "evaluation", "dimension", "value"])
        for problem in problems:
            if problem.noise_sd > 0.0:
                continue
            for seed in range(1, n_seeds + 1):
                state = {"evaluation": 0}

                def objective(trial):
                    x = suggest(trial, problem)
                    state["evaluation"] += 1
                    for i, value in enumerate(x):
                        writer.writerow([problem.name, seed, state["evaluation"], i, f"{value:.9g}"])
                    return float(np.float32(problem.function(x)))

                study = optuna.create_study(sampler=TPESampler(seed=seed))
                study.optimize(objective, n_trials=budget)
            print(f"trajectory {problem.name} done", flush=True)


def overhead_space(space, dimension):
    distributions = {}
    for i in range(dimension):
        name = f"p{i:02d}"
        if space == "mixed" and i % 3 == 1:
            distributions[name] = IntDistribution(0, 20)
        elif space == "mixed" and i % 3 == 2:
            distributions[name] = CategoricalDistribution([0, 1, 2, 3, 4])
        else:
            distributions[name] = FloatDistribution(0.0, 1.0)
    return distributions


def random_value(rng, distribution):
    if isinstance(distribution, CategoricalDistribution):
        return distribution.choices[rng.randint(len(distribution.choices))]
    if isinstance(distribution, IntDistribution):
        return int(rng.randint(distribution.low, distribution.high + 1))
    return float(distribution.low + rng.rand() * (distribution.high - distribution.low))


def run_overhead(path, n_timed=20):
    """ask() + suggest + tell with a pre-loaded history, using Optuna's default in-memory storage."""
    with open(path, "w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["optimizer", "space", "n_history", "dimension", "ms_per_suggestion"])
        for space in ("continuous", "mixed"):
            for dimension in (2, 10, 50):
                for n_history in (100, 1000, 5000):
                    distributions = overhead_space(space, dimension)
                    rng = np.random.RandomState(2)
                    study = optuna.create_study(sampler=TPESampler(seed=1, n_startup_trials=1))
                    study.add_trials(
                        [
                            create_trial(params={name: random_value(rng, d) for name, d in distributions.items()}, distributions=distributions, value=float(rng.rand()), state=TrialState.COMPLETE)
                            for _ in range(n_history)
                        ]
                    )
                    start = time.perf_counter()
                    for _ in range(n_timed):
                        trial = study.ask(distributions)
                        study.tell(trial, float(rng.rand()))
                    milliseconds = 1000.0 * (time.perf_counter() - start) / n_timed
                    writer.writerow(["Optuna", space, n_history, dimension, milliseconds])
                    output.flush()
                    print(f"Optuna {space} N={n_history} d={dimension}: {milliseconds:.2f} ms/suggestion", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["quality", "trajectory", "overhead"])
    parser.add_argument("output")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--problems", default="")
    args = parser.parse_args()

    problems = build_problems()
    if args.problems:
        selected = args.problems.split(",")
        problems = [p for p in problems if p.name in selected]
        assert len(problems) == len(selected), "--problems names a problem that does not exist."

    if args.mode == "quality":
        run_quality(problems, args.output, args.seeds or 30, args.budget)
    elif args.mode == "trajectory":
        run_trajectory(problems, args.output, args.seeds or 20, args.budget)
    else:
        run_overhead(args.output)


if __name__ == "__main__":
    main()
