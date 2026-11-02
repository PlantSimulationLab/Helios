# The apple harvesting task in IsaacSim (PhysX)

A port of `../harvest_env.cpp` to IsaacSim 6.1 with PhysX, so that policies trained in one simulator can be evaluated in the other.
`IsaacHarvestVecEnv` has the interface of `harvest_env.AppleHarvestVecEnv`, so the same Stable-Baselines3 policies and the same
`evaluate()`/`report()` code run in both.

| File | Does |
|---|---|
| `harvest_env_export_tree()` in `../harvest_env.cpp` | Writes `tree_<seed>.usda` with `writePlantStructureUSD()` and `tree_<seed>.json` with what the Helios environment does with each link, the candidate fruit, the wood and the task's constants |
| `tree_preparation.py` | Turns an exported tree into one PhysX can simulate as MuJoCo does, at a cost PhysX can afford (see below) |
| `harvest_scene.py` | The arm of `harvest_arm.urdf` as a PhysX articulation, with the servos of `harvest_env.cpp` as joint drives |
| `arm_kinematics.py` | `Robot::solveInverseKinematics()` of the Helios robotics plug-in, in numpy for many arms at once |
| `isaac_harvest_env.py` | The environment |
| `cross_evaluate.py` | Runs a scripted, random or trained controller in either simulator on given trees, with `evaluate.py`'s report |
| `check_env.py`, `bench_tree.py` | Checks and timing used while porting |

## Use

From the sample's build directory, after building it, with IsaacSim's Python (which needs `torch` and `stable-baselines3` installed):

```bash
C:/isaacsim/python.bat -c "import ctypes,os; l=ctypes.CDLL(os.path.abspath('apple_harvest_env.dll')); [l.harvest_env_export_tree(s, b'isaac_trees') for s in range(301, 313)]"
C:/isaacsim/python.bat ../isaac/cross_evaluate.py --simulator isaac --controller scripted --seeds 301-312
C:/isaacsim/python.bat ../isaac/cross_evaluate.py --simulator isaac --controller policy --policy apple_policy --seeds 301-312
```

## How the tree is simulated

In MuJoCo the tree is one articulation in which most links are welded: every internode except 8 behind each candidate fruit, and the
candidate fruit, which are held by detachable welds. A tree of about 2,700 links has about 100 ball joints. As one PhysX articulation,
with every welded link a link on a fixed joint, four trees took 230 ms per step. `tree_preparation.py` merges every welded link into
the body that carries it (exact for rigid bodies): the rigid wood into the base link, and the wood beyond a flexible internode into it.
That leaves about 105 links per tree. Candidate fruit stay separate links on locked joints, so their stem load can be read as the
force in that joint. Fruit collide as MuJoCo's sphere, and every body gets the Helios mass and inertia.

Even so, PhysX takes about 5-7 ms per tree per physics step on the CPU, and about 3.7 ms on the GPU with 64 trees, against about
1 ms in MuJoCo for the whole environment step. About 80% of that is the articulation's dynamics, not collision: removing every collision
shape of the tree saved 20%, and the solver's iteration count made no difference to the time.

## Known differences from the Helios environment

- **The fruit's hold is rigid.** MuJoCo's detachable weld is a soft constraint with a 4 ms time constant; here it is a locked joint.
  When the cup pushes the fruit just after gripping, the stem load spikes to 2-3x the force to detach in PhysX, and only just past it
  in MuJoCo. Both simulators count some such pushes as successes (10 of 36 in PhysX, 6 of 33 in MuJoCo, scripted controller on
  trees 1-3).
- **Simulations finish their control step.** All simulations take all 20 physics steps; one whose episode ended keeps stepping until
  the control step ends and is then reset, where Helios stops it at once. Nothing from those steps is used.
- **Random draws differ.** The fruit, force to detach and start position use numpy, so episodes are not paired with Helios's.
- **The armature is the export's with `armature_stability_ratio = 1`,** which PhysX needs to hold a tree up
  (`doc/isaac_comparison_report.md`). MuJoCo uses no armature.
- **Settling takes 7 minutes for 6 simulations** (20 s of simulated time at 2 ms), against seconds in Helios.
