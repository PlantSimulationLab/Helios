"""Benchmark problems, mirroring main.cpp. `python problems.py` regenerates problem_check.csv, which the
C++ driver's "verify" mode compares against so that the two definitions cannot drift apart."""

import csv
import math
import os

import numpy as np


class Spec:
    def __init__(self, name, kind="float", low=0.0, high=0.0, categories=None):
        self.name, self.kind, self.low, self.high, self.categories = name, kind, low, high, categories


class Problem:
    def __init__(self, name, specs, function, f_star=0.0, noise_sd=0.0):
        self.name, self.specs, self.function, self.f_star, self.noise_sd = name, specs, function, f_star, noise_sd


def shift(index, low, high):
    value = 0.31 + 0.6180339887 * index
    return low + (high - low) * (0.15 + 0.7 * (value - math.floor(value)))


def float_specs(dimension, low, high):
    return [Spec(f"x{i:02d}", low=low, high=high) for i in range(dimension)]


def sphere(x):
    return sum((v - shift(i, -5.0, 5.0)) ** 2 for i, v in enumerate(x))


def rastrigin(x):
    total = 10.0 * len(x)
    for i, v in enumerate(x):
        z = v - shift(i, -5.12, 5.12)
        total += z * z - 10.0 * math.cos(2.0 * math.pi * z)
    return total


def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))


def ellipsoid(x):
    z = [v - shift(i, -5.0, 5.0) for i, v in enumerate(x)]
    n = len(z)
    for pass_index in range(2):
        for i in range(n - 1):
            angle = 0.7 + 0.3 * i + 0.5 * pass_index
            rotated = math.cos(angle) * z[i] - math.sin(angle) * z[i + 1]
            z[i + 1] = math.sin(angle) * z[i] + math.cos(angle) * z[i + 1]
            z[i] = rotated
    return sum(10.0 ** (3.0 * i / (n - 1)) * z[i] * z[i] for i in range(n))


def branin(x):
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    t = 1.0 / (8.0 * math.pi)
    term = x[1] - b * x[0] * x[0] + c * x[0] - 6.0
    return term * term + 10.0 * (1.0 - t) * math.cos(x[0]) + 10.0


ALPHA = [1.0, 1.2, 3.0, 3.2]
A = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
P = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]


def hartmann6(x):
    return -sum(ALPHA[i] * math.exp(-sum(A[i][j] * (x[j] - P[i][j]) ** 2 for j in range(6))) for i in range(4))


def ackley(x):
    n = len(x)
    z = [v - shift(i, -32.768, 32.768) for i, v in enumerate(x)]
    squares = sum(v * v for v in z)
    cosines = sum(math.cos(2.0 * math.pi * v) for v in z)
    return -20.0 * math.exp(-0.2 * math.sqrt(squares / n)) - math.exp(cosines / n) + 20.0 + math.exp(1.0)


def mixed_int_float(x):
    centers = [7.0, -2.0, 3.0]
    total = 0.0
    for i, v in enumerate(x):
        z = v - (centers[i] if i < 3 else shift(i, -5.0, 10.0))
        total += z * z + 5.0 * (1.0 - math.cos(2.0 * math.pi * z / 5.0))
    return total


def category_cost(dimension, category):
    value = (dimension + 1) * (category + 1) * 0.7548776662
    return 3.0 * (value - math.floor(value))


def categorical_part(c0, c1, c2, c3):
    total = category_cost(0, c0) + category_cost(1, c1) + category_cost(2, c2) + category_cost(3, c3)
    total += 0.0 if (c0 + c1) % 5 == 2 else 2.0
    total += 0.0 if (c2 + c3) % 5 == 1 else 2.0
    return total


def categorical_float(x):
    total = categorical_part(int(x[0]), int(x[1]), int(x[2]), int(x[3]))
    return total + sum((x[i] - shift(i, -2.0, 3.0)) ** 2 for i in range(4, len(x)))


def build_problems():
    problems = []
    problems += [Problem(f"sphere_{d}", float_specs(d, -5.0, 5.0), sphere) for d in (2, 5, 10)]
    problems += [Problem(f"rastrigin_{d}", float_specs(d, float(np.float32(-5.12)), float(np.float32(5.12))), rastrigin) for d in (2, 5, 10)]
    problems += [Problem(f"rosenbrock_{d}", float_specs(d, float(np.float32(-2.048)), float(np.float32(2.048))), rosenbrock) for d in (2, 5, 10)]
    problems += [Problem(f"ellipsoid_{d}", float_specs(d, -5.0, 5.0), ellipsoid) for d in (5, 10)]
    problems.append(Problem("branin", [Spec("x00", low=-5.0, high=10.0), Spec("x01", low=0.0, high=15.0)], branin, 0.39788735772973836))
    problems.append(Problem("hartmann6", float_specs(6, 0.0, 1.0), hartmann6, -3.32236801141551))
    problems += [Problem(f"ackley_{d}", float_specs(d, float(np.float32(-32.768)), float(np.float32(32.768))), ackley) for d in (2, 10, 20)]
    problems.append(Problem("noisy_hartmann6", float_specs(6, 0.0, 1.0), hartmann6, -3.32236801141551, 0.1))
    problems.append(Problem("noisy_rosenbrock_5", float_specs(5, float(np.float32(-2.048)), float(np.float32(2.048))), rosenbrock, 0.0, 5.0))
    problems.append(Problem("mixed_int_float", [Spec(f"n{i}", "int", -5.0, 10.0) for i in range(3)] + [Spec(f"x{i}", low=-5.0, high=10.0) for i in range(3)], mixed_int_float))
    f_star = min(categorical_part(a, b, c, d) for a in range(5) for b in range(5) for c in range(5) for d in range(5))
    problems.append(Problem("categorical_float", [Spec(f"c{i}", "categorical", categories=[0.0, 1.0, 2.0, 3.0, 4.0]) for i in range(4)] + [Spec(f"x{i}", low=-2.0, high=3.0) for i in range(2)], categorical_float, f_star))
    return problems


def write_check_table(path):
    rng = np.random.RandomState(12345)
    with open(path, "w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["problem", "f_star", "f", "x..."])
        for problem in build_problems():
            for _ in range(10):
                x = []
                for spec in problem.specs:
                    if spec.kind == "categorical":
                        x.append(float(rng.randint(len(spec.categories))))
                    elif spec.kind == "int":
                        x.append(float(rng.randint(int(spec.low), int(spec.high) + 1)))
                    else:
                        x.append(float(np.float32(spec.low + rng.rand() * (spec.high - spec.low))))
                writer.writerow([problem.name, repr(problem.f_star), repr(problem.function(x))] + [repr(v) for v in x])


if __name__ == "__main__":
    write_check_table(os.path.join(os.path.dirname(os.path.abspath(__file__)), "problem_check.csv"))
