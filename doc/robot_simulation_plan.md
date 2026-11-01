# Robot training simulation in Helios: plan

Status: agreed 2026-10-30. Work happens on branch `dev/physicsengine` in the `Helios_mujoco` worktree.

## Goal

Make Helios a training environment for robotic harvesters and pruners. This is not an Isaac Sim replacement. Isaac Sim
is open source and already covers general robotics, so matching its breadth would not set Helios apart. Helios's
advantage is realistic multi-species plants and physically based cameras, combined with plant-robot physics.

**Out of scope:** the Omniverse/USD runtime, a general scene editor, a general robot asset library, and anything else
that is not specific to plants.

**MuJoCo is the only physics engine.** `dev/mechanicaldeformation`, the in-house physics engine attempt, is abandoned.

## Starting point

- `projects/PruningRobot` does not use MuJoCo. It has hand-written FK/IK and scripted motion, and branches never deflect.
- The proof-of-concept plug-in simulated plants and kinematic boxes only. It had no robot bodies, actuators or sensors,
  plants could not push back, and nothing could be cut or detached after `initialize()`.

## Progress

**Phase 1 complete (2026-10-30).** `plugins/physicsengine` replaces the proof of concept.

- **No behavior change from the split.** Four scenes (bean with a sweeping box, two self-colliding beans, apple with
  per-triangle leaves, bean with `implicitfast` and armature) gave the same MJCF apart from viewer colors, and the same
  link positions and Context vertices to the exact float.
- **No plug-in overhead.** Stepping through the plug-in measured -1% and -2% against direct `mj_step`, which is noise.
- **The engine has no plant dependency.** A project with only `physicsengine` builds and runs without building
  `plantarchitecture`.

Measured with `samples/physicsengine_benchmark` on one Apple Silicon Mac: apple at 275 days, 1053 links, 12720
primitives, Release build, radiation on the Vulkan backend. Rendering times varied about 2x between the two runs,
so treat them as ranges.

| Physics fidelity, 2 ms time step | ms/step | x real time, one core |
|---|---|---|
| Flexible internodes, leaves collide | 7.9-9.7 | 3.9-4.9 slower |
| Rigid internodes, leaves collide | 1.2 | 1.6 faster |
| Rigid internodes, leaves do not collide | 0.6-0.8 | 2.5-3 faster |
| Rigid internodes, organs collide with each other | 4.8 | 2.4 slower |

| Per frame | ms |
|---|---|
| Copy state to Context | 0.6 |
| Visualizer update and render, 1280x720, 4x AA, shadows | 200-490 |
| Visualizer update and render, 640x480, no AA, no shadows | 270 |
| Radiation geometry update (BVH rebuild) | 18-22 |
| Radiation band and camera, 640x480, 8 samples per pixel | 110-210 |

Eight parallel instances gave 7300-10300 steps/s (4.2-5.7x one instance). At a 50 Hz control rate that is 730-1030
control steps/s, or 27-38 hours per 10^8 control steps, without rendering.

**What this changes:**

- **Rigid internodes are the practical training default.** They are 6-8x faster than flexible ones. Flexible wood should
  be limited to the branches near the arm.
- **Most of the Visualizer's time was not spent on pixels.** Profiling found it in per-primitive buffer updates. See "Rendering for phase 4" below.
- **BVH rebuilds are not the bottleneck** at this scene size, contrary to the risk listed below.

**Phase 2 complete (2026-10-30).** `plugins/robotics` and `samples/robotics_pruning`, a URDF of the PruningRobot's arm working on a
dormant grapevine (911 links, flexible stems). `projects/PruningRobot` itself is unchanged: the new sample replaces its role
for physics, but not its appearance, planning, or its choice among many cuts.

Gate, measured in that sample:

- **The arm deflects a branch:** 148 mm, against 0.07 mm of drift before first touch.
- **A sensor reads the force:** 0.62 N on the tool, equal to the force on the plant, falling to 0 N right after the cut. Separately,
  on a rod with a known spring, the force matches theory within the 15% test tolerance. No plant stiffness has been checked against a real plant.
- **A cut completes mid-run:** the cut piece, over 40 cm of shoot, falls and comes to rest on the ground, and the stump stays up.

What this changes or leaves open:

- **Cuts use welds, not `mj_recompile`.** A detachable link is a free body held by a weld that is released per instance, so parallel environments can
  each cut somewhere different with no change to the model. `mj_recompile` would change the model shared by every instance. The cost: cut sites must be
  chosen before compile, each adds 6 degrees of freedom, and a held joint is rigid. Arbitrary cuts anywhere on a plant are not supported.
- **The tool force is an ideal contact sum.** MuJoCo's force sensor read about 0.3 N with no contact while moving at 0.1 m/s, and is only reliable at rest. The
  likely cause, not proven, is the interaction of the discrete integrator and joint damping. Weight, inertia, noise, and bandwidth of a real sensor are not modeled yet.
- **The sample cuts by joint angle.** The blade does not physically shear the shoot, and no cutting force or breaking limit is modeled.
- **Nothing in the sample plans around obstacles.** It fails with an error if the arm hits the vine while driving. During the push, it can touch neighboring shoots.
- **The PlantArchitecture model is not updated by a cut.** Only the physics and the Context change.

**Phase 3 complete (2026-10-30). Decision: stay on CPU MuJoCo for state-based training.** `samples/robotics_apple_harvest` trains a
policy to pick apples on `apple_fruitingwall` trees at day 650, without leaves. Chosen first crop and task: apple, reach and detach. The robot is the
same arm with a vacuum cup, as no specific robot was chosen.

Results on 12 trees never used for training or for setting up the task, 240 episodes each, with 95% intervals:

| Controller | Success | Collision (protective stop) | Timeout | Median peak force on tree |
|---|---|---|---|---|
| Trained, seed 0 | 86.2% (81.3-90.0) | 12.9% | 0.8% | 0 N |
| Trained, seed 1 | 87.5% (82.7-91.1) | 12.1% | 0.0% | 11 N |
| Scripted (straight at the fruit, straight back) | 75-76% | 20-21% | 3-4% | 14-16 N |
| Random actions | 0.0% | 0-0.4% | 99.6-100% | 0 N |

- **Speed:** about 300 control steps/s on 6 threads of one Apple Silicon Mac, including network updates: 28 minutes for 500,000 steps.
  Success on the training trees passed 85% by 200,000 steps (about 12 minutes). This is 0.04 s of simulated time per step.
- **It pulls for real:** at detachment, the grip force is 0.96-0.97x the force needed, and the stem load 1.01x.
- **The first version looked better and was not.** Without joint force limits and a protective stop it scored 89.6%, with contact forces of 272 N at
  the 90th percentile. Check how a policy succeeds, not just that it does.
- **The scripted controller is a high bar:** learning added about 11 points and cut collisions by about 40%. With exact state, this task is not hard. Vision will be.
- **Evaluate on many trees:** scores on the 4 trees watched during training ranged from 62.5% to 87.5% from one report to the next.

What this does not show: see "What is idealized" in the sample's README. The policy has exact positions, there are no leaves, only easy front-facing fruit are targeted, the grip is ideal, and detachment ignores
pull direction. Peak contact can reach about 1000 N for a moment when the arm strikes rigid wood at speed, before the stop acts. Neither a rendered episode nor real hardware has checked any of this. No plant stiffness has been validated.

**Rendering for phase 4 (2026-10-31).** The Visualizer is the default camera. Most of its time went to making 12 separate buffer updates per changed primitive, each of which could make the macOS driver synchronize with the GPU. Grouping them, and leaving unmoved geometry out of what the Context marks as changed, gave:

| Visualizer at 128x128, no antialiasing, no shadows, ms per frame | Before | After |
|---|---|---|
| Whole plant moving, 12,700 primitives | 408 | 12-14 |
| Whole plant moving, 40,000 primitives | 1467 | 52 |
| Harvest-like: 40,000 primitives, rigid wood, a box moving | 26-29 (with the grouped uploads) | 8-10 with dirty flags cleared each frame |

Reading the pixels back adds 1-2 ms, which the table does not include. With it, on the benchmark's smaller scene (12,700 primitives, rigid wood), a frame takes 9-10 ms at 128x128 and 14-15 ms at 1280x720 with antialiasing and shadows, so resolution matters little. The speedup is measured on macOS only, where the driver's cost per call is high, and other drivers may gain less. These are single frames of one scene on an idle machine. A control step costs about 20 ms per thread, so one environment now spends more time on physics than on rendering.

Rendering many environments, with `samples/robotics_apple_harvest` and a camera on the wrist, at 128x128 or 64x64:

| Simulations | Render per simulation, ms | Steps/s with images | Steps/s without |
|---|---|---|---|
| 8 trees x 2, before the changes below | 44-46 | 20-21 | 282 |
| 8 trees x 4, before the changes below | 65 | 15 | 320 |
| 8 trees x 2 | 17 | 48 | 281 |
| 8 trees x 4 | 17 | 49 | 298 |

Those changes were hiding the navigation gizmo, which is rebuilt whenever the camera moves, and `PhysicsEngine::setContextUpdateTolerance()`, which leaves motion too small to see out of the update. Instances are drawn one at a time, so about 50 steps/s is what is left, against about 280 without images: 500,000 steps would take about 3 hours rather than 28 minutes. Most of the remaining time is the driver's work in the draw call. These are measured with a scripted controller and no network, on macOS only.

**Rendering at 180 steps/s (2026-10-31).** Profiling those 17 ms found that about 12.6 ms was CPU time rebuilding geometry, 1 ms was drawing and waiting for the GPU, and about 10,000 primitives, a quarter of the scene, were really moving on every frame. Each tree settled for only 1 s, so every episode released it out of balance, and it swung at up to 0.6 m/s for most of the 10 s episode, even with the arm held still. Each row adds to the one above, for 8 trees x 2 at 128x128:

| Change | Render per simulation, ms | Steps/s with images |
|---|---|---|
| Before | 16-17 | 48-51 |
| Trees settle for 20 s rather than 1 s | 7.4-8.0 | 87-92 |
| No passes over all primitives on an ordinary frame; only the buffers that changed are uploaded | 4.5-4.7 | 121-125 |
| Trees prepared in parallel, with `Visualizer::prepareContextGeometry()` | about 2.7 | about 160 |
| Images read back only after all are drawn | 2.05-2.11 | 178-181 |

The rows before and after the settling show different images, since the tree no longer swings. Below that row the images are identical, by a hash over every frame. Reading back the pixels is included. 8 trees x 4 at 64x64 reaches 199 steps/s. Settling adds about 7 s per tree to the build, which went from 12 s to 70 s for 8 trees. A trained policy succeeded on 87.5% of 240 episodes on unseen trees with either settling, so the swing did not change the outcome of the task, though it did change the images.

With leaves on, physics becomes the limit. A step takes 14 ms, because the leaves add about 3,400 bodies, so rendering at 3.3 ms gives 58 steps/s and the most possible is about 72. Settling takes about 30 s per tree. Leaves that are only visual, not bodies, may be the next large gain, and have not been tried.

The GPU takes about 0.8 ms per image and that barely changes with resolution or with leaves, so frustum culling has little to gain. There is no visible cost in switching OpenGL contexts. By estimate, tiled rendering would reach at most about 200 steps/s, which does not justify the change. All of this was measured on one M2 Pro, and the gain from reading back later is not measured on other drivers.

Visualizers alive together used to draw the wrong scene silently, because only the newest one's OpenGL context was current. They now each make their own current. A Visualizer created after Context::markGeometryClean() still draws only what has changed since, unless Context::markGeometryDirty() is called first.

The radiation camera took about 39 ms per frame at its cheapest setting, whose image quality has not been looked at, and is the option for when physically based images are worth the setup.

**First image policy (phase 4, in progress).** `train_vision.py` imitates the phase 3 state policy (DAgger, about 77,000 frames, 8 training trees) with no exact fruit or wood positions, given a target position with an error of 4 cm standard deviation along each axis for the whole episode. That error is larger than the cup's 3 cm reach. On 12 unseen trees, 240 episodes each, one training seed each:

| Policy | Success (95% interval) | Protective stop | Knocked off | Timeout | Training time |
|---|---|---|---|---|---|
| State policy (exact positions, phase 3) | 86-88% | 12-13% | 0-0.4% | 0-0.8% | 28 min |
| Camera, 64x64 | 60.4% (54.1-66.4) | 36.2% | 0.0% | 3.3% | 51 min, 34 collecting and 17 training |
| Blind, same target error | 46.7% (40.5-53.0) | 43.3% | 3.8% | 6.2% | 4 min |

- **The camera helps:** about 14 points, with intervals that do not overlap. It is one training seed of each, on the same set of trees.
- **Collisions are the main failure.** The image policy cannot see the exact wood positions that the state policy uses, and wood is thin at 64x64.
- **Unseen trees are much harder:** while collecting on the training trees in the last rounds, the camera policy succeeded about 85% and the blind one about 65%, against 60% and 47% on unseen trees. Eight trees are probably too few. Those collection rates are a guide, not a controlled comparison, and each fruit recurs on the training trees, so a policy can partly memorize them.
- **Rate:** 35-43 frames/s while collecting, against about 50 steps/s measured without a network.

These were all trained and evaluated before trees settled for 20 s, so on trees that swung for much of each episode. They have not been repeated on trees at rest, which changes the images and may change these results.

Still to do: more training trees, a second seed, higher resolution to see thin wood, and leaves. Leaves are essential, since they hide fruit and wood, and everything here is without them. The images have not been compared with real ones.

**Image policies on settled trees (phase 4, 2026-11-01).** These replace the earlier results on swinging trees. Each policy imitates the state policy on 24 training trees (154,000 frames, about 16 minutes with a camera), at 64x64, with a target position error of 4 cm standard deviation per axis. Success over 240 episodes per cell, 12 trees per set:

| No leaves | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| Camera, trees 301-312 | 71.2% | 67.1% | 65.4% | 67.9% |
| Blind, trees 301-312 | 48.8% | 57.5% | 44.6% | 50.3% |
| Camera, trees 401-412 | 56.2% | 50.0% | 56.7% | 54.3% |
| Blind, trees 401-412 | 40.8% | 53.3% | 42.9% | 45.7% |

Trees 401-412 played no part in choosing the setup, so +9 points is the fairer estimate of what the camera adds, and trees 301-312 give +18. The camera beat blind in 5 of the 6 comparisons with the same seed. The state policy with exact positions scored 86-88% on trees 301-312.

With leaves (fixed to the wood, the arm passes through them, seed 0 only):

| With leaves | Trees 301-312 | Trees 401-412 |
|---|---|---|
| Camera | 54.6% | 45.0% |
| Blind | 53.3% | 45.4% |

**So the camera helps without leaves, and shows no benefit with them, on one seed.** Other findings:

- **Results move a lot between seeds and tree sets.** Single runs ranged from 50% to 71% for the same setup. One run on one tree set is not enough to compare setups: the first run with 24 trees scored 71%, and looked like a bigger gain than it was.
- **Resolution:** 128x128 was no better than 64x64 (61% vs 63%, 8 training trees, one seed).
- **More training trees:** the gap between training and unseen trees closed with 24 trees, against 12-15 points with 8.
- **Failures are mostly protective stops,** at 27-38% for the camera policy without leaves.
- **Leaves as physics bodies** cap the rate near 72 steps/s, and fixed to the wood they gave 91 steps/s at 64x64, with a startup of about 19 s per tree.

Not done: a rendered episode has not been watched, leaves have one seed, and the images have not been compared with real ones. No target error other than 4 cm has been tried.

## Plug-in structure

Two plug-ins, static libraries in one process, so the boundary itself costs nothing at run time.

| `physicsengine` (general) | robotics plug-in |
|---|---|
| Bodies, joints, contacts, stepping | Robot abstraction (arms, end effectors, mobile bases) |
| Load any MJCF/URDF, combine models | Inverse kinematics and controllers |
| Forward kinematics and Jacobians | Sensor noise and latency models |
| Raw contact and force readings | Task and environment API for training |
| Syncing poses to Context geometry | ROS 2 bridge (possibly its own package) |

### Design rules

1. **Do not hide MuJoCo behind a generic engine interface.** It would reduce the API to what all engines share and add
   cost in the hottest loop. Document the plug-in as MuJoCo-based, and expose `mjModel`/`mjData` directly to code that
   needs them. Add an abstraction only if a second engine ever arrives.
2. **`mjData` is the source of truth; the Context is a copy for rendering.** Controllers and training code never read
   physics state back from the Context. Sync only when a frame is actually rendered, with a bulk path.
3. **Separate the model from its running instances.** Parallel training needs one model with many data objects, only
   some of which have Context bindings.
4. **The engine does not depend on plants.** It has a generic "add articulation" call. A plant adapter is compiled only
   when `plantarchitecture` is in `PLUGINS`. `plantarchitecture` never depends on the engine, or every plant user would
   download and build MuJoCo.

## Phases

Each phase has a gate to pass before moving on.

1. **Consolidate.** Commit the proof of concept on current master, split out `physicsengine`, and benchmark.
   *Gate:* measured physics and render cost per step on a real scene, and stepping through the plug-in shows no
   measurable overhead against raw `mj_step` on the same model.
2. **Physical robot.** Create the robotics plug-in and rebuild PruningRobot on MuJoCo with an imported URDF.
   *Gate:* the arm deflects a branch, a sensor reads the force, and a cut completes mid-run.
3. **One trained task.** Python environment, and state-based fruit reach and detach.
   *Gate:* a policy learns across varied plants at a tolerable speed. This is the go/no-go for CPU MuJoCo.
4. **Vision.** Add randomization and train with images. The Visualizer is the default camera, since it works with almost no setup. The radiation camera gives better images but needs much more setup and input data, so it is the option for when that is worth it, such as perception datasets and validation against real images. This needs a fast path for images into Python;
   PyHelios label maps currently go through a temporary file and `numpy.loadtxt`.
5. **ROS 2.** A bridge for testing real robot software in simulation (`/clock`, tf, joint states, images, point clouds,
   a `ros2_control` interface), optional and Linux-first.
6. **Real-world validation and GPU.** Compare against field measurements. Move to GPU only if phase 3 shows the need.

## Known risks

- **Throughput.** See the measurements above. Flexible wood near the fruit only, as in phase 3, is what makes training practical. Larger trees are unmeasured.
- **Every plant has a different structure.** Batched simulators prefer identical models. `mujoco.rollout` expects
  homogeneous models, MJX shares structural fields across worlds, and MuJoCo Warp needs an NVIDIA GPU for fast simulation.
- **Contact accuracy.** Penetration of 2-3 mm at 0.5 m/s, up to about 1 cm for leaves, is about the size of a thin shoot
  or fruit stem.
- **Cutting and detachment.** See phase 2 above. Fruit detachment is the same mechanism but has not been tried on fruit, and no force threshold triggers it yet.
- **Render sync.** The radiation BVH rebuild takes about 20 ms at 12720 primitives, so it is not urgent, but it was not measured on larger scenes. The Visualizer update is the larger cost.

## Open decisions

- Which robot to model, and what real measurements to validate against. Phase 3 used apple with a generic arm.
- Whether NVIDIA/Linux is acceptable for large-scale training (assumed yes, keeping development working on macOS).

## Related work

- OrchardBench (arXiv 2607.06337): GPU-parallel apple trees on Newton with bending, breaking branches and detaching fruit.
- Find the Fruit (arXiv 2505.16547): sim-trained policy exposing fruit on real plants, up to 86.7% success.
