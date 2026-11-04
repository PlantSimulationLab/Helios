#!/usr/bin/env python3
"""Compare every trial evaluated by the C++ port and by Optuna, from the "trajectory" modes of main.cpp and run_optuna.py.

    python compare_trajectories.py <traj_cpp.csv> <traj_py.csv>

Exits with status 1 if any parameter value differs by more than 1e-6 of its range. Noisy problems are not included in
the trajectory files, since the two implementations draw independent noise.
"""

import sys

import pandas as pd

from problems import build_problems

TOLERANCE = 1e-6


def main():
    cpp = pd.read_csv(sys.argv[1])
    python = pd.read_csv(sys.argv[2])
    merged = cpp.merge(python, on=["problem", "seed", "evaluation", "dimension"], suffixes=("_cpp", "_python"))
    if not len(merged) == len(cpp) == len(python):
        print(f"The files cover different trials: {len(cpp)} rows from C++, {len(python)} from Optuna, {len(merged)} in common.")
        return 1

    ranges = {}
    for problem in build_problems():
        for dimension, spec in enumerate(problem.specs):
            ranges[(problem.name, dimension)] = 1.0 if spec.kind == "categorical" else spec.high - spec.low
    merged["error"] = (merged.value_cpp - merged.value_python).abs() / [ranges[key] for key in zip(merged.problem, merged.dimension)]

    n_diverged_problems = 0
    for name, group in merged.groupby("problem", sort=False):
        diverged = group[group.error > TOLERANCE]
        n_seeds = group.seed.nunique()
        print(f"{name:20s} largest error {group.error.max():.2e} of range, {diverged.seed.nunique()}/{n_seeds} seeds diverged" + (f", first at evaluation {diverged.evaluation.min()}" if len(diverged) else ""))
        n_diverged_problems += len(diverged) > 0
    print(f"{n_diverged_problems} problems diverged.")
    return 1 if n_diverged_problems else 0


if __name__ == "__main__":
    sys.exit(main())
