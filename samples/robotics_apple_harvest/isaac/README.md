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
| `movie_record.py`, `movie_render.py` | Record a controller's episodes on one tree in either simulator, and draw them side by side as a movie (see below) |

## Use

From the sample's build directory, after building it, with IsaacSim's Python (which needs `torch` and `stable-baselines3` installed):

```bash
C:/isaacsim/python.bat -c "import ctypes,os; l=ctypes.CDLL(os.path.abspath('apple_harvest_env.dll')); [l.harvest_env_export_tree(s, b'isaac_trees') for s in range(301, 313)]"
C:/isaacsim/python.bat ../isaac/cross_evaluate.py --simulator isaac --controller scripted --seeds 301-312
C:/isaacsim/python.bat ../isaac/cross_evaluate.py --simulator isaac --controller policy --policy apple_policy --seeds 301-312
```

## Movie

`results/harvester.mp4` shows the same tree (301, never used in training) and the same policy (trained in Helios) in both simulators, side by
side, in real time. Episodes are not paired: each simulator draws its own fruit, force to detach and start position. Both halves are drawn the
same way, by the Helios visualizer, from each simulator's recorded poses (`harvest_env_render_movie_frame()` in `../harvest_env.cpp`): IsaacSim's
bodies are mapped back onto the Helios links they carry. The leaves are drawn for context only; neither simulator simulates them. An inset in
each half shows what the wrist camera of `harvest_env_enable_camera()` would see (`harvest_env_render_wrist_frame()`). The policies shown do
not use it: they are given exact positions of the fruit and wood, which stand in for perception. It needs `imageio-ffmpeg` in IsaacSim's
Python.

Episodes end when the fruit detaches, so for the movie a success goes on for a second (`--follow-through 25`), with the cup backing straight
away from the tree, whatever the policy would do, carrying the freed fruit (`harvest_env_set_follow_through()` in Helios,
`follow_through_steps` here). Nothing from those steps counts: the episode's reward and record are those it had when the fruit detached.
IsaacSim can only free a fruit on a soft hold, by switching its joint off in USD, which PhysX follows while simulating; a rigid hold is a
locked articulation joint, which the tensor API cannot release. So IsaacSim is recorded with `--fruit-hold soft`.

```bash
C:/isaacsim/python.bat ../isaac/movie_record.py --simulator helios --policy apple_policy_helios_seed0 --tree 301 --output helios_301.npz
C:/isaacsim/python.bat ../isaac/movie_record.py --simulator isaac --policy apple_policy_helios_seed0 --tree 301 --output isaac_301.npz
C:/isaacsim/python.bat ../isaac/movie_render.py --helios helios_301.npz --isaac isaac_301.npz --output harvester.mp4 --still harvester.png
```

## How the tree is simulated

In MuJoCo the tree is one articulation in which most links are welded: every internode except 8 behind each candidate fruit, and the
candidate fruit, which are held by detachable welds. A tree of about 2,700 links has about 100 ball joints. As one PhysX articulation,
with every welded link a link on a fixed joint, four trees took 230 ms per step. `tree_preparation.py` merges every welded link into
the body that carries it (exact for rigid bodies): the rigid wood into the base link, and the wood beyond a flexible internode into it.
That leaves about 105 links per tree. Candidate fruit stay separate links on locked joints, so their stem load can be read as the
force in that joint. Fruit collide as MuJoCo's sphere, and every body gets the Helios mass and inertia.

With `--fruit-hold soft` (`cross_evaluate.py`, `isaac_train.py`, `check_env.py`), candidate fruit are instead free bodies outside the
articulation, held to their stem by a D6 joint with drives on all six axes. The drives copy the transient of MuJoCo's weld
(`soft_hold_gains()` in `tree_preparation.py`): MuJoCo makes the weld's acceleration follow `a = (1 - d) a0 + d (-b v - k r)` with
`k = 1 / (dmax tau)^2`, `b = 2 / (dmax tau)`, `tau = 4 ms` and impedance `d = 0.9` (`dmax = 0.95`), so the springs get stiffness
`m d k` (9,300 N/m for a 150 g fruit) and damping `m d b` (71 N s/m), with the fruit's mass `m` as the weld's effective mass (the wood on
the other side is much heavier on this time scale), and the fruit's inertia about the joint for rotation. They do not copy the
`(1 - d) a0` term: MuJoCo's weld passes 90% of a load at once, while a spring builds force over its time constant. In this mode:

- The stem load is the force the springs put on the fruit in a step, from the fruit's change of momentum less gravity, the grip and the
  contact force on it, which is what PhysX applied. Computing it from the springs' stretch, measured between the joint frames, gave
  1.07-1.9x the fruit's weight at rest, from an offset of 0.03-0.14 mm between where the tensor API puts the frames and where PhysX's
  drive balances; the momentum gives 0.998-1.001x.
- The grip reaches the fruit as a change of its velocity before each step (force x step / mass, at its center of mass), the same as
  applying the force over the step. Applying forces through the rigid body view, followed by setting the fruit's state in a reset,
  made PhysX's GPU kernels fail with an illegal memory access.
- A reset puts each fruit back where it settled relative to its stem's link, not at its settled world pose. Setting a tree's joint
  positions, even to the values they already had, does not leave every link where it was: with fruit placed at their settled world
  poses, some stems carried up to 25x the fruit's weight in the first step (a spring stretched by about 4 mm), enough to knock fruit
  off. Placed relative to the stem's link, the load in the first step is 0.99-1.09x the weight.

Most of PhysX's time at first went to writing every body's pose to USD and Fabric after each step, which only rendering needs. The
environment switches that off: with 64 trees on the GPU a step went from 224 to 27 ms. What remains is a floor of about 20 ms per step
inside PhysX's GPU solve that hardly grows with the number of trees, so the GPU pays off only with many simulations (`bench_tree.py
--no-writeback`; see `doc/isaac_comparison_report.md`).

## Known differences from the Helios environment

- **The fruit's hold is rigid by default.** MuJoCo's detachable weld is a soft constraint with a 4 ms time constant; by default here it
  is a locked joint. When the cup pushes the fruit just after gripping, the stem load spikes to 2-5x the force to detach in PhysX, and
  rarely past 1.2x in MuJoCo, so PhysX counts more such pushes as successes (18 against 6 of 240 episodes, scripted controller on
  trees 301-312). With `--fruit-hold soft` there are none (16 of the 18 episodes succeed by pulling instead, 2 end in a protective
  stop), and total success is unchanged (66.2% against 66.7%). Neither hold is MuJoCo's weld, which passes 90% of a load at once: see
  above and `doc/isaac_comparison_report.md`. The default stays rigid, so that earlier results and the IsaacSim-trained policy keep
  their meaning; the soft hold is the closer match to MuJoCo on pushes, and costs a little more (13.6 min for the IsaacSim-trained
  policy's 240 episodes against 11.4 min, with another IsaacSim process sharing the GPU for part of that run).
- **Simulations finish their control step.** All simulations take all 20 physics steps; one whose episode ended keeps stepping until
  the control step ends and is then reset, where Helios stops it at once. Nothing from those steps is used.
- **Random draws differ.** The fruit, force to detach and start position use numpy, so episodes are not paired with Helios's.
- **The armature is the export's with `armature_stability_ratio = 1`,** which PhysX needs to hold a tree up
  (`doc/isaac_comparison_report.md`). MuJoCo uses no armature.
- **Settling takes 7 minutes for 6 simulations** (20 s of simulated time at 2 ms), against seconds in Helios.
