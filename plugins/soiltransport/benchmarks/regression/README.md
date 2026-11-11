# SoilTransport regression gate

Checks that a change to the plug-in does not make the solvers less accurate, more work or slower. A fixed set of
cases is run and compared with the **baseline**: the results of the last accepted version, recorded on this machine.

## Cases

`soil_regression list quick` (about a minute) / `list full` (adds the 2.5 cm drip and the MODFLOW 6 tests).
Each case reports

- **errors** against an exact or reference solution (lower is better),
- **values** without an exact reference (must not drift),
- **work**: steps, iterations, failed steps (lower is better),
- **time_s**: solve time only.

| Cases | Checks |
|---|---|
| `col_mass_balance`, `col_gardner_steady` | closed-column storage; exact steady Gardner profile |
| `col_infiltration_*`, `col_evaporation_*` | Vanderborght et al. (2005) analytical solutions (DuMux benchmarks) |
| `col_annulus_*` | radial flow to a root: exact steady profile (second order), time to wilting (DuMux) |
| `col_drying`, `3d_drying_*` | a demand the soil can no longer meet: steps must fail cleanly; column and 3D agree |
| `3d_consistency_*`, `3d_thin_cells_*` | the column and the 3D solver (CPU, GPU) give the same answer |
| `3d_drip_*` | 3D drip irrigation vs a grid- and time-converged reference (`reference/drip3d_theta5cm.f32`) |
| `3d_emitter_switch_adaptive_*` | an emitter switched on and off under adaptive stepping (default integrator): the water stored must equal what it delivered |
| `mf6_test2_*`, `mf6_test3_*` | Vázquez-Gasty et al. (2026) Tests 2 and 3 vs MODFLOW 6 (needs `SOIL_REGRESSION_MF6`, see below) |

Cases a version of the plug-in cannot run (features it lacks, detected at compile time) report `unsupported`.

## Use

```
# once per machine/build: build the cases against this tree
./run_regression.py build ../../../.. --build-dir build/current

# after every change
./run_regression.py gate --exe build/current/soil_regression      # exit 1 on a regression
```

The gate lists regressions, improvements and notes. When a change is deliberately accepted (a commit, or an
intended trade-off), make its results the new baseline:

```
./run_regression.py accept results/current.json
```

Keep the baseline fixed between acceptances: small changes that each pass the tolerances then still add up against
the same reference and eventually fail the gate. A known trade-off can instead be recorded in `waivers.json`
(`[{"case": ..., "metric": ..., "baseline": ..., "reason": ...}]`).

Tolerances (options of `gate`/`compare`): errors +5% (and 1e-7), values 0.1%, work +10% (and 1), time +15% (and
2 ms). Times are compared only with a baseline recorded on the same CPU and GPU. Quick cases run three times and
keep the fastest time; run the gate on an otherwise idle machine.

To record a baseline for another version, build the cases against its source tree (e.g. a `git worktree` of the
commit) and run them: `./run_regression.py run --exe <exe> --label <name>`, then `accept` the results.

## Data

- `reference/drip3d_theta5cm.f32`: water content (float32, 20 x 40 x 40, x fastest, bottom layer first) after 48 h,
  averaged onto the 5 cm grid, from `soil_regression reference` (1.25 cm cells, 112.5 s steps, BDF2, the mean of
  arithmetic and upstream-k_r weighting; their difference, at most 1.8e-3, bounds its error).
- MODFLOW 6 tests: `SOIL_REGRESSION_MF6` points to a directory with `test2/` and `test3/` (the case arrays of the
  comparison, plus `mf6_final_psi.bin`, MODFLOW 6's final pressure heads in its node order, float64).
