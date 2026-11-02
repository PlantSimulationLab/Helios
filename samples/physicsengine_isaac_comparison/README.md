# Helios (MuJoCo) and IsaacSim (PhysX) on the same plant

Runs the box sweep of `samples/physicsengine_box_sweep` in both simulators on the same plant and compares them link by link.
The results and what they mean are in `doc/isaac_comparison_report.md`.

| File | Runs in | Does |
|---|---|---|
| `main.cpp` | Helios | Grows the plant, writes it to USD with `writePlantStructureUSD()`, writes the link table and the sweep, and runs the sweep in MuJoCo |
| `isaac_box_sweep.py` | IsaacSim's Python | Loads the USD file, adds a kinematic box, and runs the same sweep in PhysX |
| `isaac_inspect.py` | IsaacSim's Python | Checks the joint frames in the USD file, and compares the stiffness, damping, armature, masses and inertias PhysX parsed with the link table |
| `compare.py` | any Python with numpy and matplotlib | Compares runs: sag, displacement during the sweep by link type, branch order and height, contact force and penetration |

Both simulators write the same CSV: `time, box_x, box_y, box_z, force_x, force_y, force_z, peak_force, peak_penetration, contacts`,
then `x, y, z` of every link in the order of `<plant>_links.csv`. `force` is the force on the plant from the box in the last step,
and `peak_*` are the largest values over the steps since the previous row. A JSON summary is written beside each CSV.

## Running it

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
cd build && mkdir runs
./physicsengine_isaac_comparison dt=0.002 armature_ratio=1 out=runs                 # USD file, link table, sweep, MuJoCo run
C:/isaacsim/python.bat ../isaac_box_sweep.py --usd runs/apple_steps500_ratio1.usda --sweep runs/apple_sweep.json \
    --dt 0.002 --out runs/isaac_apple_dt0.002_ratio1.csv
C:/isaacsim/python.bat ../compare.py --links runs/apple_links.csv --sweep runs/apple_sweep.json \
    --run "MuJoCo=runs/mujoco_apple_dt0.002_noarmature.csv" --run "PhysX=runs/isaac_apple_dt0.002_ratio1.csv" --out runs/apple --plots
```

`main.cpp` takes `key=value` arguments; see the comment at its top. `dt` also sets `USDExportParameters::physics_steps_per_second`,
so the USD armature matches the step it is run at. `timing=1` (and `--timing` for the IsaacSim script) reads nothing back while
stepping, for wall time.

## Things to know

- **Generate every plant on the machine that simulates it.** The same seed grows a different plant on Windows than on macOS.
- **IsaacSim 6 makes Newton the active engine at startup.** The script switches it off and checks that PhysX is active.
- **The GPU tensor API cannot set kinematic targets** in IsaacSim 6.1.0. The script runs PhysX dynamics on the GPU but reads and
  writes state through the CPU (numpy) tensor API, which is slow for a single scene: about 2 minutes of wall time per simulated
  second when contacts are read every step. Timing runs read nothing back.
- **Each contact filter pattern must match exactly one body**, so every link is listed.
