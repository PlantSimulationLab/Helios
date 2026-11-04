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
7. **Apple picking policies transfer both ways (Part C).** On 12 unseen trees, the Helios-trained policy succeeds 79.2% in Helios and
   75.0% in PhysX. The IsaacSim-trained policy succeeds 75.4% in both. The differences are within the 95% intervals. PhysX records more
   detachments by pushing, because its fruit joint is rigid: holding the fruit with springs that have the time constant of MuJoCo's
   soft weld removes them (18 to 0 for the scripted controller), without changing total success beyond the intervals. Training with the same settings took 25 minutes in Helios and 7.9 hours in
   IsaacSim. PhysX's GPU throughput only catches up with a few hundred trees at once.
8. **With a grip that needs contact, the policies still transfer (Part C, second grip).** The cup must now touch the fruit (5 mm) where
   it gripped from 3 cm. The Helios-trained policy succeeds 69.2% in both simulators (soft hold). The IsaacSim-trained policy, now
   trained with the soft hold, succeeds 75.8% in Helios and 70.8% in PhysX. The scripted controller does worse in PhysX (49.2% against
   57.9%), mostly because it knocks fruit off more often there. Training took 37 minutes in Helios and 11.2 hours in IsaacSim.

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

**The GPU figures are not representative of IsaacSim's GPU pipeline.** Part C found that most of each step was spent writing results
to USD and Fabric for rendering; switching that off made the GPU 3-9x faster (see "Cost, and where PhysX's time went"). IsaacSim 6.1.0's GPU tensor API does not implement kinematic
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

## Part C: the apple picking task in both simulators

Chosen with the user: the IsaacSim side is built on IsaacSim's tensor API rather than Isaac Lab, wrapped as the same Stable-Baselines3
environment as Helios's, so that the same policies, `train.py` settings and `evaluate()`/`report()` code run in both. Isaac Lab is not
installed, and its cloned environments assume every copy has the same structure, which trees do not. PyTorch 2.11 (CUDA 12.8),
Stable-Baselines3 2.9.2 and gymnasium 1.4.0 were installed into IsaacSim's Python (nothing it had was changed). The port, its design
and its known differences are in `samples/robotics_apple_harvest/isaac/README.md`, and the episode records in its `results/`.

### Results

Trees 301-312 (never used in training), 2 simulations per tree, 10 episodes each, 240 episodes per row. A "push" is a success with no
grip force: the cup pushed the fruit off just after gripping, rather than pulling it.

| Controller | Simulator | Success (95% interval) | Pushes | Successes by pulling | Protective stop | Timeout | Peak contact, p90 / max |
|---|---|---|---|---|---|---|---|
| Scripted | Helios (MuJoCo) | 61.3% (55.0-67.2) | 6 | 58.8% | 32.1% | 6.7% | 501 / 1012 N |
| Scripted | IsaacSim (PhysX) | 66.7% (60.5-72.3) | 18 | 59.2% | 32.1% | 1.2% | 493 / 1755 N |
| Policy trained in Helios | Helios (MuJoCo) | 79.2% (73.6-83.8) | 2 | 78.3% | 17.5% | 3.3% | 347 / 1199 N |
| Policy trained in Helios | IsaacSim (PhysX) | 75.0% (69.2-80.1) | 5 | 72.9% | 21.2% | 3.8% | 361 / 2142 N |
| Policy trained in IsaacSim | Helios (MuJoCo) | 75.4% (69.6-80.4) | 3 | 74.2% | 23.3% | 0.4% | 379 / 901 N |
| Policy trained in IsaacSim | IsaacSim (PhysX) | 75.4% (69.6-80.4) | 7 | 72.5% | 24.2% | 0.4% | 411 / 2209 N |

The IsaacSim rows hold the fruit with a rigid joint. The rows below hold it with springs that copy the transient of MuJoCo's soft weld
(`--fruit-hold soft`: the fruit is a free body on a six-axis D6 drive with the weld's 4 ms time constant and damping, and the stem load
is found from the fruit's momentum; the design and its checks are in the port's README):

| Controller | Simulator | Success (95% interval) | Pushes | Successes by pulling | Protective stop | Timeout | Peak contact, p90 / max |
|---|---|---|---|---|---|---|---|
| Scripted | IsaacSim (PhysX), soft hold | 66.2% (60.1-71.9) | 0 | 66.2% | 32.1% | 1.2% | 562 / 1767 N |
| Policy trained in Helios | IsaacSim (PhysX), soft hold | 81.7% (76.3-86.1) | 4 | 80.0% | 12.9% | 5.0% | 293 / 1797 N |
| Policy trained in IsaacSim (rigid hold) | IsaacSim (PhysX), soft hold | 77.5% (71.8-82.3) | 1 | 77.1% | 22.5% | 0.0% | 403 / 2136 N |

- **With the rigid hold, the scripted controller does the same in both once pushes are set aside** (58.8% against 59.2%). PhysX has
  three times as many pushes: the stem load at a push reaches 1.7x the force to detach at the median and 4.5x at most in PhysX, and
  1.1x and 2.0x in MuJoCo.
- **The rigid hold causes the extra pushes.** With the soft hold, PhysX has 0, 4 and 1 pushes (scripted, Helios-trained,
  IsaacSim-trained) against 6, 2 and 3 in MuJoCo and 18, 5 and 7 with the rigid hold. The stem load at a push is at most 1.03-1.14x
  the force to detach, against 1.01-2.0x in MuJoCo and 2.7-4.9x with the rigid hold. The scripted runs draw the same fruit, force to detach and start position for every episode with either hold, so they can be
  compared episode by episode: 16 of the 18 pushes became pulls and 2 became protective stops, and all else is nearly unchanged (138
  pulls and 74 stops in both, 9 episodes changed in other ways). Total success is the same (66.7% and 66.2%), so the scripted
  controller now succeeds by pulling 7 points more often in PhysX than in MuJoCo (66.2% against 58.8%), within the intervals.
- **The soft hold does not make PhysX match MuJoCo better on success.** The Helios-trained policy succeeds 81.7% in PhysX with the soft
  hold against 75.0% with the rigid one and 79.2% in MuJoCo, with fewer protective stops (12.9% against 21.2% and 17.5%). The
  IsaacSim-trained policy, trained with the rigid hold, succeeds 77.5% with the soft one against 75.4%. These runs are not paired
  (the policies followed the scripted controller in one process, so they drew different episodes), and every difference is within the
  intervals.
- **The soft hold is not MuJoCo's weld.** It has the weld's time constant and damping, but MuJoCo's weld also passes 90% of a load to
  the stem at once (the `(1 - d) a0` term of its constraint dynamics), where a spring builds force over its time constant, so the
  springs should be the more forgiving of the two; with so few pushes the counts cannot confirm it. Peak contacts are unchanged: those
  are the arm against wood.
- **A policy trained in Helios transfers to PhysX** with a drop of 4 points (5 by pulling), within the intervals at this sample size.
  It behaves the same way in both: it pulls mostly downward (mean direction -0.29, -0.01/-0.03, -0.86 in both) and detaches the fruit at
  0.94x the grip force needed in both.
- **PhysX's worst contacts are harder.** The protective stop acts at 50 N, but a stiff arm striking wood peaks higher before it acts:
  1755-2142 N in PhysX against 1012-1199 N in MuJoCo.
- **A policy trained in IsaacSim transfers to Helios with no drop**: 75.4% in both (74.2% and 72.5% by pulling). It is no better in its
  own simulator than the Helios-trained policy is in PhysX (75.0%), and 4 points below the Helios-trained policy in Helios, within the
  intervals. Most of its failures are protective stops (23-24%) rather than timeouts.
- **The two policies learned different pulls.** The Helios-trained policy pulls mostly downward (mean direction -0.29, -0.01, -0.86) and
  detaches at 0.94x the grip force needed. The IsaacSim-trained policy pulls toward the robot and upward (-0.59, -0.01, +0.66 in Helios;
  -0.57, -0.03, +0.62 in PhysX) at 1.05x. Each strategy works in both simulators, so the difference comes from training, not from
  which physics it learned in; with one seed each, I cannot say whether it is systematic.
- Each policy was trained once, with seed 0 and the same `train.py` settings, trees (1-8) and 16 simulations, for 500,000 steps. In
  Helios this took 25 minutes; by the end it reached 87.5% on the training trees and 77.5% on 4 held-out trees. In IsaacSim it took
  7.9 hours: 18 steps per second against 330, a factor of 18, with settling taking 5.6 minutes. It reached 86% on the training trees at
  the end, against 83% in Helios. IsaacSim's run learned more slowly at first (2% and 19% success at 50k and 100k steps, against 13% and
  55%). One seed and one tree set: the plan shows that single runs vary by several points, so these curves do not separate the
  simulators.

### With a grip that needs contact (commit c63f6a035)

The movie showed the cup gripping fruit from a distance. The grip was changed so that the cup grips only a fruit whose surface is
within 5 mm of its face (it was 3 cm), and then pulls only along the line from its face to the fruit's center, with its rest length at
the fruit's radius. Both simulators were changed the same way, both policies were retrained with the same settings, and the IsaacSim
policy was trained with the soft hold this time. Same trees and episodes as above; the scripted controller still backs straight away
from the tree once it grips.

| Controller | Simulator | Success (95% interval) | Pushes | Protective stop | Timeout | Knocked off | Peak contact, p90 / max |
|---|---|---|---|---|---|---|---|
| Scripted | Helios (MuJoCo) | 57.9% (51.6-64.0) | 3 | 35.0% | 1.7% | 5.4% | 501 / 1012 N |
| Scripted | IsaacSim (PhysX), rigid hold | 49.6% (43.3-55.9) | 9 | 35.0% | 1.2% | 13.8% | 534 / 1755 N |
| Scripted | IsaacSim (PhysX), soft hold | 49.2% (42.9-55.5) | 2 | 35.4% | 1.2% | 13.8% | 581 / 460868 N |
| Policy trained in Helios | Helios (MuJoCo) | 69.2% (63.1-74.7) | 0 | 15.8% | 15.0% | 0.0% | 308 / 839 N |
| Policy trained in Helios | IsaacSim (PhysX), rigid hold | 62.5% (56.2-68.4) | 1 | 20.4% | 12.1% | 5.0% | 361 / 1260 N |
| Policy trained in Helios | IsaacSim (PhysX), soft hold | 69.2% (63.1-74.7) | 0 | 18.3% | 10.4% | 2.1% | 373 / 1310 N |
| Policy trained in IsaacSim (soft hold) | Helios (MuJoCo) | 75.8% (70.0-80.8) | 0 | 15.0% | 6.2% | 2.9% | 247 / 700 N |
| Policy trained in IsaacSim (soft hold) | IsaacSim (PhysX), soft hold | 70.8% (64.8-76.2) | 0 | 19.6% | 7.1% | 2.5% | 302 / 1350 N |

- **Against the 3 cm grip**, the scripted controller loses 3 points in Helios and 17 in PhysX, and the Helios-trained policy 10 in
  Helios and 12 in PhysX (soft hold). The IsaacSim-trained policy, retrained, is unchanged in Helios (75.4% to 75.8%) and 7 points
  lower in PhysX (soft hold, 77.5% to 70.8%).
- **Fewer pushes**: 0-3 in every row but the scripted controller with the rigid hold (9, against 18 with the 3 cm grip), since the cup
  no longer grips a fruit it is about to push off from a distance.
- **The scripted controller does 9 points worse in PhysX**, near the edge of the intervals (51.6-64.0 against 42.9-55.5). The difference
  is mostly fruit knocked off without being gripped (13.8% against 5.4%), the same with either hold, so it comes from how the cup
  strikes the fruit or wood on its way in, not from the hold. One episode in PhysX with the soft hold peaked at 4.6e5 N on the tree before the fruit was knocked off; the next highest is
  1743 N. It looks like a solver blow-up in that one episode, which I have not investigated.
- **Both policies transfer.** The Helios-trained policy succeeds 69.2% in both simulators with the soft hold (62.5% with the rigid one).
  The IsaacSim-trained policy succeeds 75.8% in Helios and 70.8% in PhysX. All within the intervals.
- **The IsaacSim-trained policy is the better one here**, by 6.6 points in Helios and 1.6 in PhysX, within the intervals. It times out
  less (6-7% against 10-15%). Again the two pull differently: the Helios-trained policy straight toward the robot (-0.89, 0.01, -0.03),
  the IsaacSim-trained one toward the robot and upward (-0.81, -0.04, +0.38 in Helios). One seed each, so not a difference between the
  simulators.
- **Training.** Helios: 37 minutes, 225 steps per second, 78.1% on the training trees at the end. IsaacSim: 11.2 hours, 12.5 steps per
  second, 73.8% on the training trees. IsaacSim's run learned slowly at first (0.5% success at 100k steps, 18% at 200k, 65% at 250k).
  Both runs shared the machine with other work for part of the time, so the times are not a clean comparison of speed; the factor of
  18 measured above is.

Helios's grip has since changed again (commit 0d2a7a5e0: the fruit is held on the face in position and orientation, and the seal
breaks on sliding and peeling as well as pulling). The IsaacSim port has not been brought up to that grip; further changes to the task
are made in Helios only, and IsaacSim will be compared again at chosen checkpoints.

### Cost, and where PhysX's time went

The first measurements made PhysX look 5-7x slower than MuJoCo per tree. Profiling with Tracy (`capture.exe` and `csvexport.exe` from
`omni.kit.profiler.tracy`, run by hand because IsaacSim's `tools/profiling/tracy_capture.py` looks for Linux binary names) showed that
most of it was not physics:

| Per physics step, 4 trees, CPU dynamics | ms |
|---|---|
| `PhysX simulate` and its worker tasks | 7.6 |
| Writing every body's pose to USD and Fabric for rendering (`updateRenderTransforms`, `USDWrite`, USD notices, Fabric hierarchy) | about 20 |

Nothing in these runs renders, and the tensor API reads the state from PhysX directly. With `/physics/updateToUsd`,
`/physics/updateVelocitiesToUsd`, `/physics/fabricUpdateTransformations` off and Fabric disabled, on the GPU (the result is unchanged
to 0.01 mm):

| Trees on the GPU | Before, ms per step | After, ms per step | Tree-steps per second after |
|---|---|---|---|
| 4 | 30 | 20 | 198 |
| 16 | 68 | 22 | 728 |
| 64 | 224 | 27 | 2383 |
| 128 | | 33 | 3869 |
| 256 | | 59 | 4367 |

On the CPU path the same settings changed nothing (28 ms for 4 trees), so the CPU figures above and in Part B still include the
writeback. What remains on the GPU is a floor of about 20 ms per step, almost all inside `PhysX simulate` (one worker task waiting on
the GPU), that hardly grows with the number of trees: latency, not work. I could not see inside it without Nsight Systems, which is
not installed. Changing the articulation's position iterations (32, 16, 8, 4) changed neither the time nor the result on the GPU, so
I cannot tell whether that override took effect there.

For comparison, Helios's training ran 6,000-6,700 physics steps per second on 12 threads. PhysX on the GPU matches that only with
a few hundred trees at once, and is far slower with the 16 simulations `train.py` uses. In the harvest environment, which is now
bound by physics (state reads, inverse kinematics and resets take a few ms per control step):

| | Helios (MuJoCo, 12 threads) | IsaacSim (PhysX, GPU, without writeback) |
|---|---|---|
| Building 24 simulations, including 20 s of settling | 2.2 min | 15-19 min with writeback (not re-measured) |
| Control steps per second, 6 simulations | | 8.3-8.9 |
| Control steps per second, 16 simulations (`train.py`), including PPO's updates | 330 while training | 18 while training (estimated 30 from 22 ms per step) |

The scripted and Helios-trained policy evaluations above were measured with writeback on, and the IsaacSim training and its policy's
evaluation with it off; the physics is the same either way. Evaluating 240 episodes took 0.6 min in Helios and 11.4 min in IsaacSim.

### Not done yet

- More training seeds.
- The IsaacSim port of the current grip (held on the face, with seal limits on sliding and peeling).
