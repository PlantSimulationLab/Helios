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

**Observations (23):** fruit position relative to the cup, distance to the fruit's surface, cup position, cup velocity, joint
positions, whether it is gripping, the load on the cup's seal as a fraction of what breaks it, contact force on the tree, time, and the
offset to the nearest wood.

**Actions (4):** cup velocity (x, y, z), up to 0.3 m/s, and whether to grip. Inverse kinematics turns the velocity into joint targets at 25 Hz.

**Episode end:** success if the fruit detaches while gripped. Failure if the force on the tree passes 50 N (a protective stop), the grip
force passes 40 N, the fruit detaches without being gripped, or 10 s pass.

## Camera

`harvest_env_enable_camera()` adds a camera on the robot's wrist, rendered with the Visualizer, and `harvest_env_render()` returns an image from every simulation. `./robotics_apple_harvest camera` measures the speed.
`train_vision.py` trains an image policy by imitating a state policy from `train.py`. It gets the camera, what the robot knows about itself, and a target position with an error that stays the same for a whole episode (4 cm standard deviation along each axis by default). With `--blind`, it trains the same policy without the camera, as a baseline.

Trees are prepared for rendering in parallel, using the environment's threads, and drawn one at a time. Images are read back only after one instance of every tree has been drawn, so that the GPU can work through them while the next is drawn.

Each tree settles under gravity for 20 s when it is built, which takes about 7 s per tree, or about 30 s with leaves. Every episode starts from the pose reached. A shorter settling leaves the tree out of balance, and it then swings for much of each episode, which also makes rendering about twice as slow because all the wood that swings has to be redrawn.

## What is idealized

- The policy is given exact positions of the fruit and wood, which stand in for perception.
- Leaves are removed unless asked for. With `--leaves`, they hide things from the camera but are fixed to the wood and collide with nothing, so the arm passes through them. Only the 8 reachable fruit nearest the robot on each tree are targeted.
- Wood is rigid except for 8 internodes behind each candidate fruit. Fully flexible wood is about 37x slower.
- The cup grips a fruit whose surface is within 5 mm of its face (it has to touch it, give or take a compliant lip), where it touches it,
  and seals at once, whatever angle it meets the fruit at. Springs then hold the fruit on the face, in position and orientation relative to
  the cup, so it moves with the cup rather than swinging from it. The seal breaks when the pull out of the face passes 40 N, the load
  along it 20 N, or the moment about its center 1 N m (twisting, and loads that peel the cup). The last two were chosen for this task:
  friction of 0.5 under the vacuum, and the full pull at the rim of a 5 cm cup. Pulling away from the stem at an angle to the face peels
  the cup, so where the cup seals on the fruit matters.
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

With the last grip the policy pulls mostly downward (mean direction -0.34, -0.09, -0.75), where the earlier ones pulled toward the robot,
and detaches fruit with a pull out of the face of only half the force to detach: the rest of the load goes along the face or into the
moment on the seal.
