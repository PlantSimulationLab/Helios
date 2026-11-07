# Helios robotics interface design

Brian Bailey, 2026-11-06. Exported on 2026-11-07 from the Claude Docs document
https://claude.ai/artifact/BsZpcQbxjk711aZRGNSWKr. That online document is the copy that gets edited; this file is a snapshot of it.

Helios should let a robotics engineer write a task as one Python class in PyHelios. The class builds a scene from reusable parts, then supplies observations, rewards and episode ends as arrays over all parallel copies. Everything that runs every physics substep stays in C++: physics, plants, robot control, grippers, contact sensing and the protective stop. The user configures those parts from Python, so changing a task never means recompiling. Today all of this lives in the apple-harvest sample (about 1,400 lines of C++ plus Python glue), so a new task means copying and rewriting it.

## How other simulators do it

Every simulator below is driven from code, and nearly all of them split a task into the same four parts: build the scene once, reset per episode, act and observe each step, and score and end episodes. They differ mainly in whether those parts are methods you override or named terms you list in a configuration.

| Simulator | How a task is written | Scene and robot | Actions and sensors | Parallel copies |
| --- | --- | --- | --- | --- |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/api_env_design.html) | Two styles. Manager-based: a config class lists named terms for observations, rewards (a weighted sum), events (startup, reset, interval) and terminations. Direct: subclass `DirectRLEnv` and write `_setup_scene`, `_pre_physics_step`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx` | USD assets, cloned once per environment | Action terms, including task-space differential IK and operational-space control; contact sensors filtered by partner, cameras, ray casters | Thousands, on the GPU |
| [ManiSkill3](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html) | Subclass `BaseEnv`: `_load_scene` (once), `_initialize_episode(env_idx)`, `evaluate` (returns success and fail), optional `compute_dense_reward` | Robots and assets loaded in `_load_scene` | Controllers picked by name per robot; reward modes sparse or dense | GPU batches |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | Subclass `MjxEnv`: `reset(rng)` and `step(state, action)` return a `State` with data, obs, reward, done and metrics | MJCF files | Written by hand per task | JAX vectorisation |
| [Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/parallel_simulation.html) | No task layer: the user writes the environment loop around the scene | `scene.add_entity(morph, material)` from URDF, MJCF or meshes, then `scene.build(n_envs=N)` | Batched calls such as `control_dofs_position`, with `envs_idx` for subsets | Batch dimension on every call |
| [dm_control Composer](https://arxiv.org/pdf/2006.12983) | Compose an `Environment` from a `Task` and an `Arena`; reusable `Entity` objects carry their own model and observables | MJCF fragments attached to an arena | Observables declared per entity | One per process |
| [robosuite](https://robosuite.ai/docs/modules/controllers.html) | `robosuite.make(env_name, robots, gripper_types, controller_configs)`; tasks are environment classes | Robots, grippers and arenas as separate MJCF models | Composite controllers per body part: operational-space pose or position, joint position, velocity or torque; gripper controllers separate | One per process |
| [Aerial Gym](https://github.com/ntnu-arl/aerial_gym_simulator) | Python config files for robot, controller and environment | Obstacle assets drawn at random per environment by an asset manager | Attitude, velocity and position controllers; RGB, depth, segmentation and flow cameras | GPU (Isaac Gym) |
| [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md) | Hydra config: a task lists its sensors (observations) and measures (privileged metrics such as distance to goal, success, SPL, and rewards); new ones are registered by name | Scanned or synthetic scenes | Sensors attached to the agent | Many processes |

All of them hand training code a [Gymnasium](https://gymnasium.farama.org/api/vector/)-style vector environment, so any standard RL library can train on them. Gymnasium's default is to reset a finished copy on the next step ([autoreset modes](https://farama.org/Vector-Autoreset-Mode)).

## What Helios should take, and what not

Take the shared shape and the reusable parts. Skip the configuration layer for now. Users work in Python, as they do in Isaac Lab, and Helios's strength is realistic plants, not GPU scale.

- **Take Isaac Lab's direct style as the main way to write a task, in Python.** One class with build, reset, act, observe, score and end methods, each working on all copies at once, as Isaac Lab's do on GPU tensors. A robotics engineer coming from Isaac Lab or ManiSkill reads it at once.
- **Take reusable parts from robosuite and dm_control.** Controllers, grippers and sensors are components that a task attaches by name, not code inside a task. The vacuum cup, the Cartesian tool control and the contact classification are today written into the harvest sample. They belong in the plug-ins.
- **Take Isaac Lab's filtered contact sensor.** Report force on a robot part split by what it touches. For plants the partners are kinds of organ (wood, fruit, leaf, the target fruit), which no other simulator knows about. That is Helios's advantage.
- **Take ManiSkill's and Habitat's split between success and reward.** A task reports named outcomes (success, slipped, protective stop) apart from its reward, so evaluation does not depend on reward shaping.
- **Take Gymnasium's vector environment as the interface to training.** One generic class, so every task trains with Stable-Baselines3, CleanRL or TorchRL unchanged, and users never write a training loop for Helios.
- **Skip, for now, Isaac Lab's manager configs and Habitat's Hydra files.** They pay off with many people sharing many terms. With three or four tasks, plain functions that tasks call (a library of observation and reward helpers) give most of the reuse with none of the indirection. Revisit once there are several tasks.
- **Do not chase GPU-scale batching.** Helios runs MuJoCo on CPU threads, about 300 control steps per second for the harvest task on 16 threads. Its users will need sample-efficient methods (scripted demonstrations, behaviour cloning, then RL), so the framework should make scripted controllers and recorded demonstrations first-class too.

## The proposed layers

Two new layers sit on top of today's plug-ins: reusable C++ components, and a Python package in PyHelios that turns a task class into a vector environment. The user writes only the task class, in Python.

```
Layer (top calls the one below)                                         Who writes it
--------------------------------------------------------------------    ------------------------
Training and evaluation code                                            The user, or a library
  Stable-Baselines3, CleanRL or TorchRL; scripted controllers; behaviour cloning
pyhelios.robotlearning package                                          New in PyHelios, generic
  Gymnasium vector env, sb3 adapter, evaluate, record, movies, ONNX export
>> The user's task class <<                                             The user: all task logic
  build_scene, define_actions, reset, observe, reward, outcome, on all copies
- - - - Python above, C++ below: PyHelios bindings, one call per control step, arrays over all copies - - - -
Robotics components, run every substep                                  Move out of the samples
  IK actions, vacuum cup, gripper, cameras, contact by organ, protective stop
physicsengine and PlantPhysics                                          Exists; needs bindings
  MuJoCo, parallel copies on threads, detachable links, plants as articulations
Helios Context and plantarchitecture                                    Exists in PyHelios
  Geometry, plant models, organ labels, rendering
```

Each layer calls only the one below it. The package calls the task's methods once per control step. The task reads state and sets up components through PyHelios bindings, which hand over arrays covering every copy. The components and the physics engine then run the substeps in C++ on worker threads.

## What runs in Python and what in C++

The line falls by how often code runs. Anything that runs once per control step (25 Hz in the harvest task) can be Python, because it is called once for all copies and works on arrays. Anything that must run every physics substep stays in C++. Python there would be slow, and a check that runs only between control steps misses what happens in between, such as a force peak that should trigger the protective stop.

| Piece | Language | How often it runs | Who writes it |
| --- | --- | --- | --- |
| Physics, plants, robot model, rendering | C++ | Every substep, or when an image is asked for | Exists |
| Components: IK and servos behind each action, vacuum cup seal and breaking, fruit detachment, contact by organ, protective stop | C++, configured from Python | Every substep | Helios developers; a user only to add a new kind |
| Task: scene, reset, observations, reward, outcomes, scripted controllers | Python, numpy over all copies | Every control step | The user |
| Training, evaluation, experiments | Python, any Gymnasium library | Every control step | The user, or a library |
| Movies | Python drives them; the Visualizer draws the frames in C++ | After a run | Generic, in the package |
| A trained policy on a robot or in a C++ sample | Exported to ONNX, run from C++ | Every control step | Exported by the package |

The Python cost per step should be small. The harvest task spends about 3 ms of physics per control step for all copies, and numpy work on arrays of a few dozen copies takes microseconds. This is an estimate, not a measurement. The framework's first step is to time it, and if an observation turns out costly (the nearest-wood search is the likeliest), that piece becomes a C++ component, not the whole task.

## Writing a task

A task is one Python class. The framework calls `build_scene` once per scene variant (here one tree per variant) and copies each variant for the parallel simulations in C++. Every other method gets all copies at once: `reset` gets the indices of the copies that finished, and `observe`, `reward` and `outcome` return arrays with one row per copy. The user never touches threads or substeps. The sketch below is the apple-harvest task rewritten this way; the names are proposals, not existing API.

```python
import numpy as np
from pyhelios.robotlearning import RobotTask, Outcomes, register_task
from pyhelios.robotlearning.actions import ToolVelocityAction, EndEffectorAction
from pyhelios.robotlearning.components import VacuumCup, ContactByOrgan, NearestOrgan, ProtectiveStop

@register_task("apple_harvest")
class AppleHarvestTask(RobotTask):
    control_hz = 25
    max_steps = 250

    # Once per variant; the framework makes the parallel copies in C++
    def build_scene(self, scene, variant):
        self.tree = scene.add_plant_from_library("apple_fruiting_wall", seed=variant, physics=self.config.plant_physics)
        self.robot = scene.add_robot("harvest_arm.urdf", base=self.config.robot_base)
        self.cup = self.robot.add(VacuumCup("tool", face_offset=0.32, **self.config.cup))
        self.contact = self.robot.add(ContactByOrgan("robot", exclude="target"))
        self.robot.add(ProtectiveStop(self.contact, limit=50.0))  # checked every substep, in C++
        self.nearest_wood = self.robot.add(NearestOrgan(self.cup.point, organ="wood"))
        self.candidates = scene.reachable_fruit(self.tree, self.robot)

    def define_actions(self):
        return [ToolVelocityAction(self.robot, "tool", max_speed=0.3), EndEffectorAction(self.cup)]

    # ids: the copies that finished
    def reset(self, ids, rng):
        self.target[ids] = self.candidates.sample(ids, rng)  # from each copy's own tree
        self.tree.set_target(ids, self.target[ids])
        self.tree.set_detach_force(ids, self.target[ids], rng.uniform(14.0, 28.0, len(ids)))
        self.robot.reset_to_home(ids)

    # Each method below returns one row per copy
    def observe(self):
        return {
            "to_fruit": self.tree.organ_position(self.target) - self.cup.point(),
            "nearest_wood": self.nearest_wood.offset(),
            "gripped": self.cup.gripped(),
            "contact": self.contact.peak_force(),
        }

    def reward(self, observation, outcomes):  # shaping only; evaluation reads outcomes
        return (-0.005 - 0.02 * np.minimum(observation["contact"], 50.0)
                + 10.0 * outcomes.is_("success") - 2.0 * outcomes.failed())

    def outcome(self):
        detached, gripped = self.tree.detached(self.target), self.cup.gripped()
        return Outcomes(success=detached & gripped, knocked_off=detached & ~gripped,
                        slipped=self.cup.seal_broken(), protective_stop=self.robot.protective_stopped())
```

The reward above leaves out the harvest task's distance and pull terms to keep the sketch short. Named observation fields let the framework build the flat vector, save its layout with each policy, and catch a changed layout when an old policy is loaded. That last check would have prevented a failed load in this project, when a field was added.

Training, evaluation and movies use one generic package for every task:

```python
from pyhelios import robotlearning as rl
from stable_baselines3 import PPO
from apple_harvest import AppleHarvestTask

env = rl.make_vec_env(AppleHarvestTask, variants=range(1, 9), copies_per_variant=2, threads=16,
                      config={"leaves": False})
model = PPO("MlpPolicy", rl.sb3(env)).learn(1_000_000)

held_out = rl.make_vec_env(AppleHarvestTask, variants=range(301, 313), copies_per_variant=2, threads=16)
print(rl.evaluate(held_out, model, episodes=10))   # outcome rates by name, with intervals
rl.record(held_out, model, "run.npz")              # link poses per step, for any task
rl.render_movie("run.npz", "run.mp4")              # frames drawn by the Visualizer
rl.export_policy(model, "apple_policy.onnx")       # runs from C++ with no Python
```

`evaluate`, `record` and `render_movie` take a trained policy or any Python function of observations, so scripted controllers and demonstrations for behaviour cloning work the same way.

C++ is needed only to add a new kind of component, such as a new gripper, sensor or drone model. That is a change to a plug-in plus its Python binding, done rarely and usually by Helios developers. Writing, changing and training a task never needs a compiler.

## Reusable components

Most of what a new task needs already exists in some form. It is either in the plug-ins without a task-facing wrapper, or written into the harvest sample with raw MuJoCo calls (49 of them). Each component is C++ with a PyHelios binding that sets it up and reads it as arrays over all copies. The table lists each component, where it is today, and where it should live.

| Component | Kind | What it does | Today | Proposed home |
| --- | --- | --- | --- | --- |
| `JointPositionAction`, `JointVelocityAction` | Action | Joint targets, scaled from [-1, 1] | `Robot::setJointTargets` | robotics |
| `ToolVelocityAction` | Action | Cartesian velocity of the tool point, with the limit that stops the target running ahead of a blocked arm | Harvest sample | robotics |
| `ToolPoseAction` | Action | Absolute tool position and axis through inverse kinematics | `Robot::solveInverseKinematics` | robotics |
| `BaseVelocityAction` | Action | Forward, sideways and turn rate of a wheeled or tracked base | None | robotics (new) |
| `MultirotorAction` | Action | Body rates and thrust, or a velocity command through a built-in controller | None | robotics (new) |
| `VacuumCup` | End effector | Seal test, holding force, seal limits on pull, slide and peel | Harvest sample | robotics |
| `ParallelGripper`, `Cutter` | End effector | Finger closing with a force limit; a blade that releases the link it closes on | Pruning sample (jaw) | robotics |
| Protective stop | Safety | Ends the episode when contact force on anything but the target exceeds a limit | Harvest sample | robotics |
| Contact sensor by organ | Sensor | Force on a robot part split by partner: wood, fruit, leaf, target, ground | Harvest sample | physicsengine |
| Nearest organ | Sensor | Offset to the nearest wood, fruit or leaf from a point | Harvest sample | PlantPhysics |
| `Camera` | Sensor | Fast RGB, depth and per-organ segmentation from a mounted camera (Visualizer) | Harvest sample (wrist camera) | robotics |
| Radiation camera | Sensor | Physically based images with automatic bounding boxes and label maps, for perception data | radiation plug-in | radiation (wrap for tasks) |
| Episode recorder | Tooling | Link poses and robot joints per step, for movies and debugging | Harvest sample and `movie_record.py` | task framework |

Each component also works from C++ outside the task framework, so a user who only wants a scripted C++ simulation (like the pruning sample) still benefits.

## How the planned tasks fit

All four fit the same task class. The gantry needs no new components, so it should be the first port and the test of the framework. Navigation and fruit counting need a mobile base or multirotor, a range sensor, and scenes of many plants.

| Task | Scene per variant | Robot and actions | Observations | Outcomes | New pieces it needs |
| --- | --- | --- | --- | --- | --- |
| Apple harvest (today) | One tree, one target fruit per episode | Arm on a drive; `ToolVelocityAction` and `VacuumCup` | Fruit and nearest wood relative to the cup, joints, grip state, contact | success, knocked_off, slipped, protective_stop | None beyond moving its code into components |
| 2D gantry harvester | One tree or a wall section | A URDF with prismatic joints and one rigid tool link; `JointVelocityAction` and `VacuumCup` | The same as harvest, in the gantry's plane | The same as harvest | None |
| Orchard navigation (ground robot or drone) | A row of trees, rigid wood for speed, start and goal | Wheeled base with `BaseVelocityAction`, or a free body with `MultirotorAction` | Pose and velocity, goal offset, ranges from rays (MuJoCo's ray casts) or a depth image | reached_goal, collision, out_of_bounds | Base and multirotor actions, ray sensor, multi-plant scenes |
| Fruit counting | The navigation scene, fruit labelled by ID | As navigation, plus a mounted `Camera` | Camera images (RGB, depth or segmentation) and pose | Count error at the end, or the fraction of fruit seen | Image observations in Python, a per-fruit visibility measure, render speed |

Helios's labelled plant geometry gives fruit counting its ground truth for free: every primitive knows which fruit it belongs to. The harvest sample already measures what its wrist camera sees by organ, so the visibility measure is a generalisation of existing code.

Rendering is the cost to watch. The wrist camera is drawn one copy at a time with OpenGL, so image-based training will be much slower than the 300 control steps per second of the state-based harvest task. A first fruit-counting task could train on state (which fruit are in view) and use images only for evaluation.

## Order of work and open decisions

Build it in four steps. Each step must reproduce the harvest results before the next starts (scripted 65.4%, policy 31.7% on trees 301–312), so the refactor cannot quietly change the physics.

1. **Components into the plug-ins.** Contact by organ, nearest organ, `ToolVelocityAction`, `VacuumCup` and the protective stop, each with tests and docs. Port the harvest sample onto them, still in C++.
2. **PyHelios bindings, then the framework.** PyHelios's README does not list physicsengine or robotics among its plug-ins, so they come first, with array access over all copies. Then `pyhelios.robotlearning`: `RobotTask`, outcomes, registration, the vector environment, the Stable-Baselines3 adapter, evaluate, record, render and ONNX export. Time the Python part of a step before porting. Then port the harvest task to Python; the sample shrinks to the task class and its README.
3. **The 2D gantry as the second task.** Change the framework wherever it pushes back, before more tasks depend on it.
4. **Navigation, then fruit counting.** Base and multirotor actions, the ray sensor, multi-plant scenes, then cameras as observations.

Decided in discussion:

- **Tasks are written in Python**, with C++ only for what runs every substep, so changing a task never needs a compiler.
- **The Python package lives in PyHelios**, as `pyhelios.robotlearning`. The Helios repository keeps the C++ components and their tests.
- **Training stays in Python** with existing RL libraries. Trained policies run from C++ through ONNX export.

Decisions still open:

- [ ] **How physicsengine and robotics are bound:** use the method PyHelios already uses for its other plug-ins. Whatever it is, the components must hand over arrays for all copies in one call, not one call per copy.
- [ ] **A C++ path for whole tasks:** none, unless the timing in step 2 shows the Python part costing more than a few percent of a step. Then move the slow piece into a component rather than the task.
- [ ] **The gantry's geometry:** which plane it moves in, and how the tool reaches into the canopy (a third, telescoping axis, or a fixed reach).
- [ ] **Plug-in name:** grow `robotics`, or add a separate `robotlearning` plug-in for the task framework so the robotics plug-in stays usable without it.
- [ ] **The IsaacSim comparison:** keep it as a sample, or give the framework a second backend. I recommend keeping it a sample; a second backend would double every component.

## Sources

These descriptions come from the projects' documentation as found by web search. The pages were not all read in full, so check exact signatures there before relying on them. The Helios details come from the code on `dev/physicsengine`.

- [Isaac Lab: classes and configs](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/api_env_design.html), [direct workflow tutorial](https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_direct_rl_env.html), [manager-based environment](https://docs.nvidia.com/isaaclab/isaac-lab/how-to-guides/creating-a-manager-based-base-environment), [task-space action configs](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/mdp/actions/actions_cfg.html), [contact sensor](https://docs.nvidia.com/isaaclab/concepts/sensors/contact-sensor)
- [ManiSkill3: introduction to task building](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html), [task template](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/template.html)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- [Genesis: parallel simulation](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/parallel_simulation.html)
- [dm_control: Software and Tasks for Continuous Control](https://arxiv.org/pdf/2006.12983)
- [robosuite: controllers](https://robosuite.ai/docs/modules/controllers.html)
- [Aerial Gym simulator](https://github.com/ntnu-arl/aerial_gym_simulator)
- [Habitat-Lab config keys](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md)
- [Gymnasium vector API](https://gymnasium.farama.org/api/vector/), [autoreset modes](https://farama.org/Vector-Autoreset-Mode)
- [PyHelios](https://github.com/PlantSimulationLab/PyHelios)
