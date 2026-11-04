#!/usr/bin/env python3
"""Summarize benchmark output as a Markdown report.

    python analyze.py <results_dir> > report.md

Expects quality_cpp.csv, quality_py.csv, overhead_cpp.csv and overhead_py.csv from
main.cpp and run_optuna.py. Regret is that of the point the optimizer would return at each budget, the one with
the best observed value, scored on the noiseless function. It is on a log10 scale, floored at 1e-12.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BUDGETS = (50, 100, 200)
EQUIVALENCE_MARGIN_DECADES = 0.25
OPTIMIZER_ORDER = ["TPE", "Optuna", "Random", "GA", "CMAES", "BO"]


def load_quality(results_dir):
    frames = [pd.read_csv(os.path.join(results_dir, name)) for name in ("quality_cpp.csv", "quality_py.csv")]
    data = pd.concat(frames, ignore_index=True)
    data["log_regret"] = np.log10(data["best_regret"].clip(lower=1e-12))
    return data


def bootstrap_median_difference(a, b, n_resamples=10000, seed=1):
    """Resamples seeds jointly: a seed gives identical trials in both implementations unless noise or tied integers are involved."""
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, a.size, (n_resamples, a.size))
    differences = np.median(a[indices], axis=1) - np.median(b[indices], axis=1)
    return np.percentile(differences, [2.5, 97.5])


def equivalence_table(data, problem_order):
    print("## TPE port vs Optuna 5.0.0\n")
    print(f"Difference in median log10 regret (C++ minus Optuna) with a bootstrap 95% CI over seeds. **Outside** marks a CI that leaves ±{EQUIVALENCE_MARGIN_DECADES} decades.\n")
    print("| Problem | " + " | ".join(f"{b} evals" for b in BUDGETS) + " |")
    print("|---|" + "---|" * len(BUDGETS))
    n_outside = 0
    n_total = 0
    for problem in problem_order:
        cells = []
        for budget in BUDGETS:
            selection = data[(data.problem == problem) & (data.evaluation == budget)]
            cpp_runs = selection[selection.optimizer == "TPE"].sort_values("seed")
            python_runs = selection[selection.optimizer == "Optuna"].sort_values("seed")
            assert cpp_runs.seed.tolist() == python_runs.seed.tolist(), f"The C++ and Optuna seeds differ on {problem}."
            cpp = cpp_runs.log_regret.to_numpy()
            python = python_runs.log_regret.to_numpy()
            low, high = bootstrap_median_difference(cpp, python)
            outside = low < -EQUIVALENCE_MARGIN_DECADES or high > EQUIVALENCE_MARGIN_DECADES
            n_outside += outside
            n_total += 1
            cells.append(f"{np.median(cpp) - np.median(python):+.2f} [{low:+.2f}, {high:+.2f}]" + (" **Outside**" if outside else ""))
        print(f"| {problem} | " + " | ".join(cells) + " |")
    print(f"\n{n_outside} of {n_total} comparisons have a CI outside the margin.\n")


def regret_table(data, problem_order, budget):
    print(f"## Median log10 regret at {budget} evaluations (IQR)\n")
    print("Lower is better. Bold is the best median on the row. `-` means the optimizer rejects the parameter types.\n")
    print("| Problem | " + " | ".join(OPTIMIZER_ORDER) + " |")
    print("|---|" + "---|" * len(OPTIMIZER_ORDER))
    selection = data[data.evaluation == budget]
    for problem in problem_order:
        medians = {}
        cells = {}
        for optimizer in OPTIMIZER_ORDER:
            values = selection[(selection.problem == problem) & (selection.optimizer == optimizer)].log_regret
            if values.empty:
                cells[optimizer] = "-"
                continue
            medians[optimizer] = values.median()
            cells[optimizer] = f"{values.median():.2f} ({values.quantile(0.75) - values.quantile(0.25):.2f})"
        best = min(medians, key=medians.get)
        cells[best] = f"**{cells[best]}**"
        print(f"| {problem} | " + " | ".join(cells[o] for o in OPTIMIZER_ORDER) + " |")
    print()


def win_loss_table(data, problem_order):
    print("## TPE port against each optimizer\n")
    print("Problems where the port is significantly better / not distinguishable / significantly worse (Mann-Whitney U, p < 0.05, not corrected for multiple comparisons). A result that is not distinguishable is not necessarily equal.\n")
    print("| Opponent | " + " | ".join(f"{b} evals" for b in BUDGETS) + " |")
    print("|---|" + "---|" * len(BUDGETS))
    for opponent in OPTIMIZER_ORDER:
        if opponent in ("TPE", "Optuna"):
            continue
        cells = []
        for budget in BUDGETS:
            better = same = worse = 0
            for problem in problem_order:
                selection = data[(data.problem == problem) & (data.evaluation == budget)]
                tpe = selection[selection.optimizer == "TPE"].log_regret
                other = selection[selection.optimizer == opponent].log_regret
                if other.empty:
                    continue
                # The p-value is NaN when both optimizers give identical values on every seed, such as at the regret floor
                p_value = mannwhitneyu(tpe, other).pvalue
                if np.isnan(p_value) or p_value >= 0.05:
                    same += 1
                elif tpe.median() < other.median():
                    better += 1
                else:
                    worse += 1
            cells.append(f"{better} / {same} / {worse}")
        print(f"| {opponent} | " + " | ".join(cells) + " |")
    print()


def timing_table(results_dir, problem_order):
    print("## Optimizer overhead in the quality runs (ms per evaluation, excluding the objective)\n")
    frames = [pd.read_csv(os.path.join(results_dir, name)) for name in ("quality_cpp.csv.timing.csv", "quality_py.csv.timing.csv")]
    timing = pd.concat(frames, ignore_index=True)
    timing["ms"] = 1000.0 * (timing.total_s - timing.objective_s) / timing.evaluations
    table = timing.groupby(["problem", "optimizer"]).ms.median().unstack()
    available = [o for o in OPTIMIZER_ORDER if o in table.columns]
    print("| Problem | " + " | ".join(available) + " |")
    print("|---|" + "---|" * len(available))
    for problem in problem_order:
        cells = ["-" if np.isnan(table.loc[problem, o]) else f"{table.loc[problem, o]:.3g}" for o in available]
        print(f"| {problem} | " + " | ".join(cells) + " |")
    print()


def overhead_table(results_dir):
    print("## Overhead per suggestion against history size (ms)\n")
    print("TPE times the sampler alone. Optuna times `study.ask()` and `tell()`, which includes its storage and trial handling, so the ratio compares TPE with using Optuna as a whole.\n")
    frames = [pd.read_csv(os.path.join(results_dir, name)) for name in ("overhead_cpp.csv", "overhead_py.csv")]
    overhead = pd.concat(frames, ignore_index=True)
    print("| Space | d | N | TPE | Optuna | Optuna / TPE |")
    print("|---|---|---|---|---|---|")
    for (space, dimension, n_history), group in overhead[overhead.optimizer != "BO"].groupby(["space", "dimension", "n_history"]):
        tpe = group[group.optimizer == "TPE"].ms_per_suggestion.iloc[0]
        optuna = group[group.optimizer == "Optuna"].ms_per_suggestion.iloc[0]
        print(f"| {space} | {dimension} | {n_history} | {tpe:.3g} | {optuna:.3g} | {optuna / tpe:.1f} |")
    print("\nBO, through the public API with `max_gp_samples` raised so that all history is kept:\n")
    print("| d | N | BO |")
    print("|---|---|---|")
    for _, row in overhead[overhead.optimizer == "BO"].sort_values(["dimension", "n_history"]).iterrows():
        print(f"| {row.dimension} | {row.n_history} | {row.ms_per_suggestion:.3g} |")
    print()


def main():
    results_dir = sys.argv[1]
    data = load_quality(results_dir)
    problem_order = list(dict.fromkeys(data.problem))
    print("# Parameter optimization benchmark\n")
    equivalence_table(data, problem_order)
    for budget in BUDGETS:
        regret_table(data, problem_order, budget)
    win_loss_table(data, problem_order)
    timing_table(results_dir, problem_order)
    overhead_table(results_dir)


if __name__ == "__main__":
    main()
