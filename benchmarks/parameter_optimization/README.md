# Parameter optimization benchmark

Compares the plug-in's TPE port against Optuna 5.0.0 and the plug-in's other global optimizers. Everything runs serially.

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 && cd ..
BIN=build/parameter_optimization_benchmark

# The C++ and Python problem definitions must agree
venv/bin/python problems.py
$BIN verify problem_check.csv

# Conformance: every trial of every seed should match
$BIN trajectory results/traj_cpp.csv
venv/bin/python run_optuna.py trajectory results/traj_py.csv
venv/bin/python compare_trajectories.py results/traj_cpp.csv results/traj_py.csv

# Search quality and overhead, summarized as Markdown. Regret is scored at the point the optimizer would return.
$BIN quality results/quality_cpp.csv
venv/bin/python run_optuna.py quality results/quality_py.csv
$BIN overhead results/overhead_cpp.csv
venv/bin/python run_optuna.py overhead results/overhead_py.csv
venv/bin/python analyze.py results > results/report.md
```

`generate_tpe_fixtures.py` regenerates the golden values in `plugins/parameteroptimization/tests/TPEGoldenData.h` used by the self-tests. It refuses to run on x86 CPUs, where NumPy may use a SIMD `argsort` that orders tied values differently from the port. INTEGER trials from Optuna should not be expected to match the port on such machines either, so `compare_trajectories.py` will report those problems there.
