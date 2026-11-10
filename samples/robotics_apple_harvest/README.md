# Apple harvesting: training a policy

A robot arm with a vacuum cup reaches a fruit on a fruiting-wall apple tree, grips it, and pulls until it detaches. Each tree seed
gives a tree with a different structure.

## Setup and use

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j8
./robotics_apple_harvest                 # scripted controller and random actions, without any learning

python3 -m venv venv && venv/bin/pip install stable-baselines3 gymnasium
PYTHONPATH=.. venv/bin/python ../train.py --steps 500000 --threads 6 --output apple_policy
PYTHONPATH=.. venv/bin/python ../evaluate.py --policy apple_policy
```

Run from the build directory, so that the library and the robot model are found. The sample builds Helios without OpenMP, because
PyTorch brings its own OpenMP runtime and two in one process can give wrong results.

## The task

**Observations (26):** fruit position relative to the cup, distance to the fruit's surface, cup position, cup velocity, joint
positions, whether it is gripping, the load on the cup's seal as a fraction of what breaks it, contact force on the tree, time, the
offset to the nearest wood, and the offset of the fruit's center from the cup's axis. The distance is from the point of the cup's lip
farthest from the fruit to the fruit's surface, so it is small only when the cup is in line with the fruit.

**Actions (4):** cup velocity (x, y, z), up to 0.3 m/s, and whether to grip. Inverse kinematics turns the velocity into joint targets at 25 Hz,
with the cup level and pointing away from the robot's turret. The arm cannot turn the cup sideways or tilt it without swinging its wrist,
so the cup is in line with a fruit when the fruit is on the level line from the turret through the cup.

**Episode end:** success if the fruit detaches while gripped. Failure if the force of the robot on the tree passes 50 N (a protective
stop), the cup's seal breaks, the fruit detaches without being gripped, or 10 s pass. The force on the tree leaves out the cup's contact
with the fruit being picked, which it has to touch; pushing that fruit too hard knocks it off.

**Built from plug-in components:** the cup is a `VacuumCup`, the velocity actions go through a `ToolVelocityAction`, the force on the tree comes from
a `ContactSensor` on the robot with the picked fruit as its target, the protective stop is a `ProtectiveStop`, and the offset to the nearest wood
comes from `NearestPlantOrgan` (robotics and physicsengine plug-ins). A `ControlLoop` runs the cup and the stop at every physics step, the stem fails
through `PhysicsEngine::setDetachForce()`, episodes start from the settled tree through `PhysicsEngine::setResetPose()`, and
`ParallelRobotSimulations` steps every simulation of every tree on its threads. The wrist and movie cameras are `RobotCamera`s. `harvest_env.cpp` keeps the task itself: choosing fruit, resets,
observations, rewards and outcomes.

**Start:** the arm tucked, with the robot up to 0.2 m along the row from the fruit. For training, `train.py --near-start 0.4` starts 40% of
episodes with the cup instead 2-12 cm short of where the fruit would be seated and up to 3 cm out of line with it, so that the policy
practises lining the cup up and sealing it; a placement the arm cannot reach without touching anything (about 3%) starts tucked. Held-out
episodes and evaluation always start tucked. On the tree whose parts collide, this did not help a policy trained from scratch (1.7% against
5.0% without it, below).

**Reward:** -0.005 a step; while not gripping, 5 per meter the distance above shrinks; 1 on gripping; while gripping, 3 times the rise in
the stem's load over 28 N; -0.02 per newton of the peak force on the tree in the step, up to 50 N; and at the end, 10 for success and -2 for
a broken seal, a knocked-off fruit or a protective stop.

## Camera

`harvest_env_enable_camera()` adds a camera on the robot's wrist, rendered with the Visualizer, and `harvest_env_render()` returns an image from every simulation. `./robotics_apple_harvest camera` measures the speed.
`train_vision.py` trains an image policy by imitating a state policy from `train.py`. It gets the camera, what the robot knows about itself, and a target position with an error that stays the same for a whole episode (4 cm standard deviation along each axis by default). With `--blind`, it trains the same policy without the camera, as a baseline.

Trees are prepared for rendering in parallel, using the environment's threads, and drawn one at a time. Images are read back only after one instance of every tree has been drawn, so that the GPU can work through them while the next is drawn.

Each tree settles under gravity for 20 s when it is built, which takes about 7 s per tree, or about 30 s with leaves. Every episode starts from the pose reached. A shorter settling leaves the tree out of balance, and it then swings for much of each episode, which also makes rendering about twice as slow because all the wood that swings has to be redrawn.

## What is idealized

- The policy is given exact positions of the fruit and wood, which stand in for perception.
- Leaves are removed unless asked for. With `--leaves`, they hide things from the camera but are fixed to the wood and collide with nothing, so the arm passes through them. Only the 8 reachable fruit nearest the robot on each tree are targeted.
- Wood is rigid except for 8 internodes behind each candidate fruit. Fully flexible wood is about 37x slower.
- The tree's parts collide with each other, so the arm cannot push a fruit through wood or another fruit. The tree is grown with many
  parts already overlapping, though: every target fruit is partly inside the shoot that bears it, and often inside other shoots and fruit,
  by up to 7.7 cm. Pushing them apart would bend the wood against the fruit and load each stem past its force to detach before the arm
  moves (the scripted controller then succeeded 8.8% of the time, with 84% of episodes ending on their first step). So each pair that
  overlaps as grown, 87 to 361 pairs per tree, never collides (`PhysicsEngineParameters::keep_initial_link_overlaps`), and a fruit can
  still pass through the particular wood or fruit it was grown into. These collisions make each step about 60% slower.
- The cup is a bowl 5 cm across whose lip is a ring of spheres 4 mm in radius; its wall does not collide. It seals at once when every point
  of the lip is within 5 mm of the fruit's surface, which means the fruit sits in the mouth with its center within about 8 mm of the cup's
  axis: a lip that only touches the fruit on one side does not seal. Springs then hold the fruit seated in the mouth, with its center on
  the axis, and its orientation, relative to the cup, so it moves with the cup rather than swinging from it. The seal breaks when the
  pull out of the cup passes 40 N, the load across the mouth 20 N, or the moment about its center 1 N m (twisting, and loads that peel the
  lip). The last two were chosen for this task: friction of 0.5 under the vacuum, and the full pull at the rim of the cup. Pulling away
  from the stem at an angle to the cup's axis peels the cup, so where the cup seals on the fruit matters.
- The fruit detaches when the load on its stem passes a force drawn from 14 to 28 N, whichever way it is pulled. Real apples are reported to
  detach more easily with bending and twisting than with a straight pull, so the direction the policy chooses to pull in means nothing yet.
- The 50 N protective stop was chosen for this task, not taken from a particular robot.

## Results as the task changes

Success over 240 episodes (2 simulations of each of trees 301-312, never used in training, 10 episodes each), with the policy trained by
`train.py` (seed 0, 500,000 steps) for that version of the task. The comparison with IsaacSim is in `isaac/` and
`doc/isaac_comparison_report.md`.

| Grip | Scripted | Trained policy | Main failures of the policy |
|---|---|---|---|
| Grips within 3 cm, pulls toward the fruit's center | 61.3% | 79.2% | collision |
| Must touch the fruit (5 mm), pulls toward its center | 57.9% | 69.2% | timeout 15%, collision 16% |
| Must touch, holds the fruit on the face, seal limits on pull, sliding and peeling | 45.0% | 52.1% | slipped 23%, collision 15% |
| Seals only with the whole lip on the fruit, which sits in the cup's mouth | 50.0% | 17.1% (31.7% after 1,500,000 steps) | timeout 48%, collision 23% (after 1.5M: collision 37%, timeout 17%) |
| Same grip, and the tree's parts collide with each other (this version) | 65.4% | 31.7% | collision 29%, knocked off 21%, timeout 16% |

With the face grip the policy pulls mostly downward (mean direction -0.34, -0.09, -0.75), where the earlier ones pulled toward the robot,
and detaches fruit with a pull out of the face of only half the force to detach: the rest of the load goes along the face or into the
moment on the seal.

The seated grip needs the fruit's center within about 8 mm of the cup's axis, and the cup points level and away from the turret, so lining
it up means moving the drive and the arm together. PPO learns this slowly: 17.1% after the usual 500,000 steps, and 31.7% (26.1-37.8%)
after 1,500,000, when its success in training was still rising (47%). Its main failure is then the protective stop, reaching through wood
to line up. The scripted controller of this version lines the cup up 8 cm short of the fruit and then pushes straight in, which the
earlier rows' controller (straight at the fruit) did not do, so its rows are not comparable with this one. The results are in `results/seat_*`.

Until this version a fruit could be pushed through wood and other fruit. Now they collide (see the idealizations below), which leaves the
scripted controller as it was (49.6%) and the seated-grip policy at 22.1% (17.3-27.7%) without retraining. Trained from scratch on the
colliding tree for 1,500,000 steps, PPO learned to hover short of the fruit until time ran out: 5.0% success with 71% timeouts, and 1.7%
with the near start. Training the seated-grip policy for 1,000,000 more steps on the colliding tree (`train.py --start-from
apple_policy_helios_seat_seed0`) brings it back to 31.7% (26.1-37.8%), with its timeouts at 16%. The results are in `results/collide_*`.

The scripted controller used to hold its distance along the line to the fruit until the cup was lined up, moving it only sideways and up or
down. From the tucked pose the arm cannot do that, so in a quarter of episodes it never moved (49.6% success, 26.7% timeouts). It now comes to
8 cm short of the fruit while lining up: 57.5% (51.2-63.6%), with timeouts at 6.2% and protective stops up from 18% to 31%, as it now
reaches through wood where it used to wait (`results/collide_scripted_approach_*`).

Every one of those protective stops is the tool, the 32 cm cone from the wrist to the cup, hitting wood, and most came while the cup was still
moving up or down to the fruit's height, close to the fruit inside the canopy. The scripted controller now stays further back while it is
further off the line, by 8 cm plus three quarters of its distance off the line (up to 15 cm more), so it moves sideways and up or down
outside the canopy and comes in along the line last: 65.4% (59.2-71.1%), with protective stops down to 22% and timeouts at 8.3%
(`results/collide_scripted_funnel_*`). Staying back further than that leaves the arm short of where it can reach, and more episodes time
out (52.1% with twice the distance, capped at 30 cm). About half of the remaining stops come on the final push, lined up, with no contact
before the hit, so from wood near the line from the turret to the fruit. The cup always points level and away from the turret, so with the
cup on the fruit the tool lies along that line, and where wood is on it no controller of this robot can reach the fruit.
