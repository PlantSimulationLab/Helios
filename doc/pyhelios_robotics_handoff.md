# Hand-off: PyHelios bindings for the physics engine and robotics plug-ins

Written 2026-11-18 for the agent who adds these bindings to PyHelios. Everything here is in the Helios repository on branch
`dev/physicsengine` (merged with `dev/v1.3.99`), which PyHelios reaches through its `helios-core` submodule. The PyHelios branch for this
work is `dev/robotics`, which so far has no commits of its own.

## Goal

A PyHelios user can set up, run, train on and film the fruit harvest task, and build robot simulations of their own, from Python without a
compiler. Concretely:

1. **`FruitHarvestTask` from Python (first, and the measure of done).** Bind the class and its settings, and give PyHelios a Gymnasium-style
   vector environment around it that Stable-Baselines3 trains on unchanged. `samples/robotics_apple_harvest/harvest_env.py` is a working
   ctypes version of exactly this, over the sample's C interface; port it onto the PyHelios bindings.
2. **The building blocks**, so that users can write other tasks: `PhysicsEngine` (with `addPlantToPhysicsEngine()` and the other plant
   functions of `PlantPhysics.h`), `Robot`, the components (`VacuumCup`, `ToolVelocityAction`, `ProtectiveStop`, `ContactSensor`,
   `JointPositionAction`, `JointVelocityAction`, `ControlLoop`) and `ParallelRobotSimulations`. These can follow once (1) works.
3. **`RobotCamera`** (needs the visualizer), for image observations outside the harvest task.

Decided by the user (2026-11-18): the task logic lives in C++ in the robotics plug-in and is set up from PyHelios. This replaces, for the
harvest task, the plan in `doc/robotics_interface_design.md` of writing each task as a Python class; that document's survey of other
simulators and its list of components are still the background. A generic `pyhelios.robotlearning` package (task base class, evaluate,
record, movies, ONNX export) is not part of this hand-off unless the user asks for it.

## What to read first (Helios side)

| File | What it gives you |
|---|---|
| `plugins/robotics/include/FruitHarvestTask.h` | The class to bind first: settings, outcomes, episode record, camera view, every method documented |
| `plugins/robotics/doc/Robotics.dox` | The plug-in's documentation, including the section "The fruit harvest task" |
| `samples/robotics_apple_harvest/harvest_env.cpp` and `harvest_env.h` | A complete C interface over `FruitHarvestTask`: the shape PyHelios's `native/src/pyhelios_wrapper_robotics.cpp` should have |
| `samples/robotics_apple_harvest/harvest_env.py` | The Python vector environment (`AppleHarvestVecEnv`), `evaluate()` and the scripted controller (`scripted_actions`) to port |
| `samples/robotics_apple_harvest/train.py`, `evaluate.py`, `isaac/movie_record.py`, `isaac/movie_render.py` | Training, evaluation, recording and movies on that environment |
| `samples/robotics_apple_harvest/README.md` | The task: observations, actions, rewards, outcomes, results |
| `plugins/physicsengine/include/PhysicsEngine.h`, `PlantPhysics.h`, `ContactSensor.h` and `plugins/physicsengine/doc/` | The physics engine |
| `plugins/robotics/include/*.h` | Robot and components |
| `doc/robotics_interface_design.md` | Background: how other simulators structure tasks, and the component list |

## The API to bind first

`FruitHarvestTask(tree_seeds, copies_per_tree, thread_count, seed, parameters)` builds every tree (one `PhysicsEngine` each) and its copies.
Copies are numbered tree by tree. Every per-step call covers all copies at once and hands over flat arrays with one row per copy, so a Python
vector environment needs one native call per control step:

- `reset(observations)`: `copies * 26` floats.
- `step(actions, observations, rewards, outcomes, terminal_observations)`: `actions` is `copies * 4` floats in [-1, 1]; `outcomes` are
  `FruitHarvestOutcome` values (0 running, 1 success, 2 timeout, 3 slipped, 4 knocked off, 5 protective stop). A copy whose episode ends is
  reset inside `step()` (Gymnasium's "next-step" autoreset is not what happens: the reset is immediate), its next episode's first observation
  is in `observations`, and its last observation is in its row of `terminal_observations`. Rows of copies that go on are left as passed in.
- `getLastEpisode(copy)` returns a `FruitHarvestEpisodeRecord` (raises if the copy has not finished an episode).
- `setNearStartFraction()`, `setFollowThrough()`, `getTargetFruitPosition()`, the static `treeHasReachableFruit()`,
  `getObservationSize()` (26) and `getActionSize()` (4).
- Cameras and movies: `enableWristCamera()`, `renderWristCameras(images)` (`copies * h * w * 3` bytes, top row first), `saveWristCameraImage()`,
  `measureWristCameraView()`, `getPlantLinkCount/Names()`, `getRestLinkPoses()`, `getDisplayState()`, `addMovieScenery()`,
  `enableMovieCamera()`, `renderMovieFrame()`, `renderWristFrame()`, `enableRadiationMovieCameras()`, `renderRadiationMovieFrame()`.

The C++ methods take and fill `std::vector`s. A native wrapper can copy them into caller-allocated numpy buffers (the sample's `harvest_env.cpp`
does exactly this; the cost is negligible next to a control step of about 3 ms of physics for all copies).

`FruitHarvestTaskParameters` is a plain struct of scalars, a string, a `vec3`, a `std::vector<float>` (`start_pose`, 6 values) and a nested
`VacuumCupParameters`. Expose it as a Python dataclass or keyword arguments, and pass it to the native layer flattened or field by field. Its
defaults are the apple harvest; keep them identical in Python (or read them from the native defaults) so that results match the C++ sample.

Errors: every method raises `helios_runtime_error` (a `std::runtime_error`) with a message naming the method and the fix. Methods that need
the visualizer, or the radiation and leafoptics plug-ins, are always declared and raise an error naming the missing plug-in when Helios was
built without them, so the wrapper can bind them unconditionally.

## How PyHelios binds a plug-in (found in the PyHelios repository)

PyHelios has a 9-phase checklist in `docs/plugin_integration_guide.md` (metadata, build, C++ interface, ctypes wrapper, high-level API, assets,
tests, docs, review); follow it. In short, for LeafOptics:

- **Native layer:** `native/include/pyhelios_wrapper_<plugin>.h` and `native/src/pyhelios_wrapper_<plugin>.cpp`, wrapped in
  `#ifdef <PLUGIN>_PLUGIN_AVAILABLE`, `extern "C"` functions marked `PYHELIOS_API`, named `create<Class>`/`destroy<Class>` and
  `<camelPlugin><Method>`. Each calls `clearError()`, checks pointers, and maps exceptions to error codes read back with `getLastErrorCode()` and
  `getLastErrorMessage()` (`native/include/pyhelios_wrapper_common.h`). Arrays in are pointer plus count; arrays out are caller-allocated or a
  `static thread_local` buffer with an out-parameter size.
- **Python layer:** `pyhelios/wrappers/U<Plugin>Wrapper.py` sets `argtypes`, `restype` and `errcheck` in a try block that records whether the
  symbols exist; `pyhelios/<Plugin>.py` is the class users see, checks `get_plugin_registry().is_plugin_available(...)`, changes to the build
  directory while creating native objects (assets are opened by relative path), and destroys them in `__del__`/`__exit__`.
  `pyhelios/config/plugin_metadata.py` holds each plug-in's dependencies, platforms and `test_symbols` (must match exported names exactly).
  Bulk numpy precedent: `pyhelios/wrappers/UContextWrapper.py`, `np.ctypeslib.as_array(ptr, shape=(count,)).copy()`.
- **Build:** `build_scripts/build_helios.py` (`INTEGRATED_PLUGINS`, `--plugins`, `generate_cmake_plugin_config()` for per-plug-in CMake
  options, as for radiation's `HELIOS_NO_OPTIX`); `pyhelios_build/CMakeLists.txt` needs the wrapper source and the plug-in's include directory
  on both `pyhelios_interface` and `pyhelios_shared`. Wheels: `.github/workflows/build-wheels.yml` (cibuildwheel; Linux, Windows, macOS),
  `build_scripts/prepare_wheel.py` (`plugin_asset_dirs`) and `setup.py` package data.
- **Tests:** `tests/test_<plugin>.py`, markers in `pytest.ini` (`native_only`, `requires_gpu`, `slow`, ...); `tests/conftest.py` skips
  tests of unavailable plug-ins by name.
- **Docs:** `docs/plugin_<plugin>.md`, `docs/Doxyfile.python`, `docs/DoxygenLayout.xml`, `docs/plugin_system.md`, `docs/CHANGELOG.md`.

## Build and packaging specifics of these plug-ins

- **MuJoCo.** `plugins/physicsengine/CMakeLists.txt` pins MuJoCo 3.16.0 exactly (its C API changes between minor releases). It uses an
  installed copy if `mujoco_DIR` points at one, and otherwise downloads and builds it from source (`HELIOS_MUJOCO_FETCH`, on by default). It
  builds as a shared library: on Windows the build directory gets `mujoco.dll` and MuJoCo's plug-in libraries (`actuator.dll`,
  `elasticity.dll`, `sensor.dll`, `sdf_plugin.dll`). These must sit next to `libhelios` and survive wheel repair (delvewheel, auditwheel,
  delocate). Nothing in PyHelios bundles a third-party shared library today except OptiX and Vulkan; follow those.
- **Plug-in dependencies.** `robotics` loads `physicsengine` itself; `physicsengine` builds its plant support (and `FruitHarvestTask` is
  compiled) when `plantarchitecture` is in `PLUGINS`. `RobotCamera` and the task's cameras need `visualizer`; the radiation movie cameras need
  `radiation` and `leafoptics`. These are compile definitions (`HELIOS_ROBOTICS_CAMERA`, `HELIOS_ROBOTICS_RADIATION`) set by the plug-in's CMake.
- **Assets.** The robot model is `plugins/robotics/assets/harvest_arm.urdf`, found with `helios::resolvePluginAsset("robotics",
  "assets/harvest_arm.urdf")`; add `'robotics': ['assets']` to `plugin_asset_dirs`. The trees need the plant architecture plug-in's assets
  (already shipped). `enableRadiationMovieCameras()` loads `plugins/radiation/spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml`
  by a path relative to the working directory, so call it inside PyHelios's usual change to the build directory.
- **OpenGL threads.** Cameras must be created and rendered on the same thread. Each Python call that renders must therefore run on the thread
  that called `enableWristCamera()` or `enableMovieCamera()`; do not move rendering to a worker thread.
- **Threads.** `thread_count` worker threads are kept for the life of the task. MuJoCo and the task do not touch Python objects from them.

## Cost and scale (Windows, 16 threads)

- A mature apple tree (the default) takes about as long to build as to grow, and about 0.7 GB of memory; it settles for 20 s of simulated time
  when built. A 12-tree evaluation with 2 copies each builds in a few minutes. Tests should use a small plant: the robotics self-tests use a
  100-day tomato (`smallHarvestParameters()` in `plugins/robotics/tests/selfTest.cpp`), which builds in seconds.
- Training runs at about 300 control steps per second for state observations; rendering the wrist camera is much slower.
- `enableRadiationMovieCameras()` needs an RTX-class GPU; a 1280 x 720 frame takes about 40 s.

## How to check the port

The C++ refactor was checked by running the scripted controller of `harvest_env.py` on trees 301-312 (2 copies each, 10 episodes per copy,
episode seed 2000) before and after moving the task into the plug-in; every one of the 240 episodes had the same outcome and record:

```
C:/isaacsim/python.bat ../isaac/cross_evaluate.py --simulator helios --controller scripted --seeds 301-312 --threads 16 --output scripted.json
```

(from `samples/robotics_apple_harvest/build`; `harvest_env.py` loads `apple_harvest_env.dll` from there). On the v1.3.99 trees (Windows, MSVC, Release) the scripted
controller succeeds in 146 of 240 episodes (60.8%: 2 copies per tree x 10 episodes x trees 301-312). The C++ sample gave the same 240 episode records, field
for field, before and after the task moved into the robotics plug-in. The per-episode records are not committed (they live in the sample's build directory),
so re-run the command above on the C++ sample first to get a reference on your machine.

The PyHelios port should reproduce these outcomes episode for episode with the same seeds and the scripted controller ported unchanged. If
it does not, the difference is in the port (argument order, the default settings, or the autoreset handling), not in the physics.

## Open questions for the user

- Whether PyHelios should also get a generic task layer (`pyhelios.robotlearning` in `doc/robotics_interface_design.md`), or only the
  harvest task and the building blocks.
- Whether wheels should include the physics engine and robotics plug-ins by default (MuJoCo adds a download and build of several minutes
  and a few shared libraries), or only builds that ask for them.
