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
- **The Visualizer is the rendering bottleneck, and pixels are not why.** Going from 1280x720 to 640x480 barely changes
  its time in a run, which points at per-frame geometry handling. It has not been profiled. This must be fixed before
  phase 4, since it currently costs more than the radiation camera.
- **BVH rebuilds are not the bottleneck** at this scene size, contrary to the risk listed below.

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
4. **Vision.** Add randomization and train with the Visualizer. Use radiation cameras for perception datasets and
   validation, not in the training loop, unless benchmarks say otherwise. This needs a fast path for images into Python;
   PyHelios label maps currently go through a temporary file and `numpy.loadtxt`.
5. **ROS 2.** A bridge for testing real robot software in simulation (`/clock`, tf, joint states, images, point clouds,
   a `ros2_control` interface), optional and Linux-first.
6. **Real-world validation and GPU.** Compare against field measurements. Move to GPU only if phase 3 shows the need.

## Known risks

- **Throughput.** See the measurements above. Mitigation: simulate only the branches near the arm. Scaling beyond the
  scene measured here is unknown.
- **Every plant has a different structure.** Batched simulators prefer identical models. `mujoco.rollout` expects
  homogeneous models, MJX shares structural fields across worlds, and MuJoCo Warp needs an NVIDIA GPU for fast simulation.
- **Contact accuracy.** Penetration of 2-3 mm at 0.5 m/s, up to about 1 cm for leaves, is about the size of a thin shoot
  or fruit stem.
- **Cutting and detachment.** Fruit can likely be welded free bodies released with `eq_active`, which is confirmed only
  indirectly. Cuts would use `mj_recompile`, whose cost per cut is unmeasured.
- **Render sync.** The radiation BVH rebuild takes about 20 ms at 12720 primitives, so it is not urgent, but it was not measured on larger scenes. The Visualizer update is the larger cost.

## Open decisions

- First crop, task and robot for phase 3. Pick one where real measurements and hardware are available.
- Whether NVIDIA/Linux is acceptable for large-scale training (assumed yes, keeping development working on macOS).

## Related work

- OrchardBench (arXiv 2607.06337): GPU-parallel apple trees on Newton with bending, breaking branches and detaching fruit.
- Find the Fruit (arXiv 2505.16547): sim-trained policy exposing fruit on real plants, up to 86.7% success.
