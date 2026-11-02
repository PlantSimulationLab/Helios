# Helios (MuJoCo) and IsaacSim (PhysX) on the same plant

Date: 2026-11-02. Branch `dev/physicsengine-isaac-comparison`, from `dev/physicsengine` at 1f50c5f95. Scripts, raw summaries and plots:
`samples/physicsengine_isaac_comparison/` (see its `README.md`; `make_results.sh` regenerates `results/` from the runs).

Machine: Windows 11, Intel Core Ultra 7 265K (20 cores), 32 GB, NVIDIA GeForce RTX 5070 (12 GB, driver 610.60), MSVC 19.50,
IsaacSim 6.1.0-rc.26 with PhysX (Newton switched off), MuJoCo 3.16.0 built by the plug-in.

Neither engine is ground truth. Agreement below shows consistency, not correctness. No plant stiffness has been checked against a real plant.

## Summary

1. **The USD export had two bugs that made every PhysX result wrong**, now fixed (commit 6349f568e). Every leaf, fruit and flower joint
   had its two ends 4 cm apart, branch bases up to 7 mm, and the fixed joint at the base rotated the whole tree 7.3 degrees. MuJoCo was
   not affected, because the plug-in places joints from the parent end only.
2. **With the export's default armature, PhysX cannot hold the apple tree up.** At 2 ms the tree collapses within a second (median link
   1.1 m from its rest pose, 13x MuJoCo's sag), and the box never touches it. This is also true at 1/60 s (3.8x) and 1/30 s (2.3x), and
   on the small bean (7x). It is not a units or parsing problem: PhysX parsed stiffness, damping and armature exactly.
3. **With 8x less armature frequency (`armature_stability_ratio = 1`, i.e. 64x more armature), PhysX agrees with MuJoCo** at 2 ms,
   1/60 s and 1/30 s. At rest, the median link sag differs by 2-18% depending on PhysX's step, with a per-link correlation of 0.999.
   During the sweep, link displacements correlate at 0.97-0.99, and PhysX moves the tree less (22% less by the median at 2 ms, 14-33% at coarser steps), largely because the export's dynamic
   friction (0.3) is lower than MuJoCo's single coefficient (0.5).
4. **PhysX penetrates the box much less**: 1.0-2.4 mm against MuJoCo's 5-14 mm at 2 ms, and 12-16 mm against 37-65 mm at 1/60-1/30 s.
   This favors PhysX.
5. **For one plant, MuJoCo is 7.5x faster than PhysX on the CPU**, and 30x faster than PhysX with GPU dynamics as this IsaacSim
   version allows it to be driven (see the limits). This says nothing about PhysX with many plants on the GPU, which is Part C.
6. **Windows:** the four test suites pass after two test fixes (commits 88fdfdd78, a80c51edf). Nothing in the plug-ins themselves
   needed changing.

## Part A: build and test on Windows

Run as `.github/workflows/windows_selftests.yaml` does, but with the Visualizer tests run (this machine has a GPU):
`utilities/run_tests.sh --tests "physicsengine,robotics,visualizer,plantarchitecture" --verbose`.

| Suite | Before | After | Fix |
|---|---|---|---|
| physicsengine | 28/28 | 28/28 | |
| robotics | 7/8 | 8/8 | a80c51edf: the test chose the stem segment nearest 2/3 height; on the Windows bean another shoot's petioles were where the tool waits, so it read 2.2 N before the push. It now takes the nearest segment with a clear approach. |
| visualizer | 170/171 | 171/171 | 88fdfdd78: the depth test removed a directory while a file in it was still open, which Windows does not allow. |
| plantarchitecture | 428/428 | 429/429 | One test added with the USD fix. The suite takes about 65 minutes on this machine (known: redundant geometry rebuilds, amplified by MSVC). |

Compiler warnings: 52 in MuJoCo's own sources (MSVC size conversions) and one `APIENTRY` macro redefinition in the Visualizer from the
Windows SDK. None in `physicsengine` or `robotics`. No failure could not be fixed.

## Part B: the same plant, pushed the same way

**Plant.** `apple` at 275 days, `seedRandomGenerator(7)`, grown on this machine: 1.94 m tall, 1001 links (371 internode, 315 petiole and 315 leaf links, no fruit), 0.73 kg.
`elastic_modulus = 5e9`, `wood_density = 750`, `fruit_mass = 0.15`. The sweep of `samples/physicsengine_box_sweep`: a 0.19 m box at 75% of
height, 1 s of settling, then 0.5 m/s across the crown and back (9.9 s in all). A bean at 25 days (295 links, 0.33 m, seed 3) was also run.

**Like for like.** Both engines get the same links, masses, joint stiffness and damping from `getPlantArticulation()`. MuJoCo was run
without armature (its default) and with the armature of the USD export at each `armature_stability_ratio` used. The USD
`physics_steps_per_second` is always the PhysX step. The PhysX box is kinematic, driven by kinematic targets so it has the sweep's velocity;
the MuJoCo box is given its velocity. Positions of all links are recorded every 0.02 s (every step at 1/60 and 1/30 s).

**Differences that remain by construction.** Leaves are a 2 mm thick box in MuJoCo and the convex hull of the blade in PhysX. Joint
limits are a 60 degree cone in MuJoCo and +/-60 degrees on two axes with free twist in PhysX (never reached here). PhysX computes the
inertia of any link with a component below 1e-6 kg m^2 from its collision shape (internodes then get 1.27x the Helios inertia, leaves
0.55x). Friction is 0.5 in MuJoCo, and 0.5 static / 0.3 dynamic with restitution 0.1 in PhysX. MuJoCo's contacts are soft, with a time
constant of max(4 ms, 2 steps).

### The USD export bugs (fixed)

`isaac_inspect.py` compares what PhysX parsed with the Helios link table, and checks that both ends of each joint coincide in the rest pose.
PhysX builds a reduced-coordinate articulation from the joint frames, so an end that is off moves the link when the simulation starts.

| Check | Before | After |
|---|---|---|
| Joints whose ends are more than 1 mm apart | 334 of 1000 (every leaf joint by 40.6 mm, branch bases up to 7.2 mm) | 0 (largest 0.003 mm) |
| Fixed joint at the base | tree turned by 7.3 degrees to make the base segment vertical | matches |
| Largest link movement in PhysX's two warm-up steps | 51 mm | 0.1 mm |
| PhysX stiffness, damping, armature / Helios | 1.000 | 1.000 |
| PhysX mass / Helios | 0.994-1.007 | same |

### Armature decides whether PhysX holds the plant up

PhysX at 2 ms, gravity only, median link displacement from the rest pose (fixed export):

| Variant | at 1 s |
|---|---|
| Export default (`armature_stability_ratio = 8`) | 1107 mm (collapsing) |
| Stiffness and damping x10 | 1011 mm |
| CPU instead of GPU dynamics | 1124 mm |
| PGS instead of TGS | 1099 mm |
| 128 position / 16 velocity iterations | 452 mm |
| 1 ms step, export at 1000 Hz | 1426 mm |
| Ratio 4 / 2 / 1 | 491 / 210 / 121 mm (ratio 1 recovers, as MuJoCo does) |
| Gravity off | 0.0 mm (so nothing is preloaded) |
| MuJoCo, any armature | 89 mm |

Stiffer springs barely help, more iterations help, and more armature fixes it, so the drives are not converging in PhysX's iterative
solver rather than being too soft. A 255/255-iteration run tracked MuJoCo for the 0.4 s it ran before I stopped it, at about 22 minutes of wall time for that 0.4 s
(alongside other runs). **This contradicts the export's documentation**, which says the armature only needs to match the step rate and that a coarser
step risks instability: here a finer step (less armature) is worse, and the default ratio of 8 gives 2.3-13x MuJoCo's sag at every step tried.

### Results, apple

At 2 ms (`results/apple_2ms_*`):

| Run | Sag at 1 s, median / max (mm) | Sweep displacement median / p90 / max (mm) | Peak force on plant (N) | Peak penetration (mm) |
|---|---|---|---|---|
| MuJoCo, no armature | 89 / 253 | 413 / 741 / 1076 | 10.9 | 7.4 |
| MuJoCo, armature ratio 1 | 89 / 253 | 402 / 799 / 1335 | 12.2 | 5.3 |
| MuJoCo, armature ratio 8 | 89 / 253 | 430 / 806 / 1230 | 15.8 | 6.7 |
| PhysX, armature ratio 1 | 121 / 301 | 323 / 624 / 928 | 14.5 | 2.4 |
| PhysX, export default | 1107 / 2388 | box never touches the tree | 0 | 0 |

"Sweep displacement" is each link's largest distance during the sweep from where it was at 1 s. The tree is still swinging at 1 s in both
engines (period about 2-2.5 s, median link motion 19 mm in the last 0.1 s), so the sag at 1 s is a moment in a swing. MuJoCo's own
armature and step choices move the sweep median by 8%, which is the scale below which a cross-engine difference means little.

Sag at rest, after 10 s of settling (`results/apple_static_*`), all still moving under 1 mm:

| Run | Median / max sag (mm) | vs MuJoCo 2 ms: median ratio, per-link correlation, median position difference |
|---|---|---|
| MuJoCo 2 ms | 67.5 / 232 | |
| MuJoCo 1/60 s | 69.8 / 237 | 1.03, 1.000, 2.4 mm |
| PhysX 2 ms, ratio 1 | 79.8 / 268 | 1.18, 0.999, 12.7 mm |
| PhysX 1/60 s, ratio 1 | 68.7 / 235 | 1.02, 1.000, 1.2 mm |
| PhysX 1/60 s, ratio 1, 255/255 iterations | 72.1 / 242 | 1.07, 0.999, 5.1 mm |

Sag grows with height in both: by quarter of tree height, 3 / 31 / 73 / 103 mm in MuJoCo and 3 / 38 / 85 / 121 mm in PhysX at 2 ms.
A static equilibrium should not depend on the step or the armature; PhysX's does, by up to 18%, and more iterations did not bring it
closer. I have not found why.

During the sweep, by height and by branch order (median sweep displacement, 2 ms):

| | 0-25% | 25-50% | 50-75% | 75-100% | order 0 | order 1 | order 2 |
|---|---|---|---|---|---|---|---|
| MuJoCo, no armature | 16 | 182 | 403 | 717 | 49 | 473 | 394 |
| PhysX, ratio 1 | 13 | 143 | 317 | 602 | 39 | 382 | 307 |

The spread is the same shape in both (per-link correlation 0.988); PhysX's links move 16-22% less in every height band and branch order. With PhysX's dynamic
friction raised to MuJoCo's 0.5 at 1/60 s, the median goes from 0.86x to 0.92x of MuJoCo, the p90 and max go from below MuJoCo's to
slightly above, and the per-link correlation from 0.985 to 0.997 (`results/apple_friction_*`). Friction explains most of the gap.

### Results, bean

At 2 ms (`results/bean_2ms_*`): PhysX with ratio 1 sags 1.15x MuJoCo (median; per-link correlation 0.999), its sweep displacements
correlate at 0.97 (medians 11.6 against 10.8 mm, maxima 132 against 133 mm), and its peak force is 3.8 against 3.1 N. With the export
default it sags 7x more and moves half as much. Peak penetration is 1.0 mm in PhysX and 13-14 mm in MuJoCo; the plug-in's documentation
already notes that thin leaf boxes can overlap a box by about a centimeter.

### Stability and time step

No run diverged in either engine. MuJoCo ran from 2 ms to 0.1 s without diverging and with sag within 3%, but contact degrades as the
step grows: peak penetration 7 / 44 / 65 / 68 / 101 mm at 2 ms, 1/60, 1/30, 1/20 and 1/10 s, and the box increasingly passes through
wood instead of pushing it (sweep median 413 to 248 mm). PhysX with ratio 1 was stable at 2 ms, 1/60 s and 1/30 s, with 2-16 mm
penetration. PhysX with the default armature never diverged either; it collapses without becoming non-finite.

### Wall time

One tree, the whole 9.9 s sweep, nothing read back while stepping, idle machine (`results/timing_results.txt`). MuJoCo was run twice
and agreed within 3%.

| Step | MuJoCo (s per simulated s) | PhysX CPU dynamics | PhysX GPU dynamics |
|---|---|---|---|
| 2 ms | 3.54 (7.1 ms per step) | 26.8 (54 ms per step) | 115 (230 ms per step) |
| 1/60 s | 0.43 | 3.26 | 14.0 |
| 1/30 s | 0.21 | 1.64 | 7.0 |

MuJoCo's cost per step does not change with armature (3.53 against 3.54). Both engines cost the same per step at every step size, so
at each engine's coarsest step used here (1/30 s for both; MuJoCo goes coarser but its contact is poor) MuJoCo is 7.8x faster than
PhysX on the CPU. The PhysX loop is driven from Python (one `set_kinematic_targets` and one `SimulationManager.step` per step), which
costs little next to 54 ms.

**The GPU figures are not representative of IsaacSim's GPU pipeline.** IsaacSim 6.1.0's GPU tensor API does not implement kinematic
targets ("GPU Rigid Body View kinematic target setting not implemented"), so GPU dynamics were run with state readback on and the numpy
tensor API, which synchronizes with the GPU every step. One tree is also the worst case for the GPU, which is built for many at once.

## Limits of this comparison

- One apple tree and one bean, one seed each, one sweep. Different trees or a different push could rank things differently.
- The record runs read contacts every step through the CPU tensor API: a 10 s PhysX run at 2 ms took about 20 minutes, which limited how
  many variants could be run.
- PhysX's penetration is from contact separations, MuJoCo's from contact distances; both are deepest-point measures, not geometric checks.
- Leaf and fruit collision shapes differ between the two exports (box against convex hull), which affects penetration and leaf contacts.
- The cause of PhysX's step-dependent static sag is not found.

## Recommendations

- Lower the default `USDExportParameters::armature_stability_ratio` from 8 to 1, and correct its documentation: what limits PhysX here
  is solver convergence, so more armature is needed at fine steps, not less. This needs checking on more plants before changing the default.
- Write the same friction into both exports, or document the difference. MuJoCo has one coefficient, so the USD dynamic friction
  should probably default to 0.5.
- Write the Helios inertia for every link instead of letting PhysX compute it for small links, at full float precision.

## Part C: not started

Waiting for a decision. Isaac Lab is not installed on this machine, and IsaacSim's Python has no PyTorch.
