"""Quick checks of the IsaacSim harvest environment: it runs, the arm's kinematics agree with PhysX, and the scripted controller can pick.

C:/isaacsim/python.bat check_env.py --trees ../build/isaac_trees --seeds 1 --instances 2 --episodes 3 --settling-steps 500
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaac_harvest_env as ihe

parser = argparse.ArgumentParser()
parser.add_argument("--trees", required=True)
parser.add_argument("--seeds", default="1")
parser.add_argument("--instances", type=int, default=2)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--settling-steps", type=int, default=None)
args, _ = parser.parse_known_args()

ihe.start_app()
from arm_kinematics import forward

start = time.perf_counter()
env = ihe.IsaacHarvestVecEnv([int(s) for s in args.seeds.split(",")], instances_per_tree=args.instances, trees_directory=os.path.abspath(args.trees), settling_steps=args.settling_steps)
print(f"built and settled in {time.perf_counter() - start:.1f} s (settling {env.settle_seconds:.1f} s)", flush=True)
observations = env.reset()

state = env._gather_state()
joints = state["joints"].cpu().numpy().astype(float)
base = env.env_origins + np.array(env.task["robot_base"])
point, axis, *_ = forward(joints, base, np.array(env.task["cup_face"]), np.array([1.0, 0, 0]))
print("tool from PhysX vs forward kinematics: largest difference", float(np.abs(point - state["tool"].cpu().numpy()).max()), "m; axis", float(np.abs(axis - state["tool_axis"].cpu().numpy()).max()), flush=True)
print("first observation", np.round(observations[0], 3), flush=True)

from harvest_env_scripted import scripted_actions

finished = np.zeros(env.num_envs, int)
totals = {}
steps = 0
start = time.perf_counter()
while finished.min() < args.episodes:
    observations, rewards, dones, infos = env.step(scripted_actions(observations))
    steps += env.num_envs
    for i in np.flatnonzero(dones):
        if finished[i] < args.episodes:
            finished[i] += 1
            name = ihe.OUTCOME_NAMES[infos[i]["outcome"]]
            totals[name] = totals.get(name, 0) + 1
            r = env.last_episode(int(i))
            print(f"sim {i}: {name} after {r.steps} steps, stem load {r.final_stem_load:.1f} / {r.detach_force:.1f} N, grip {r.final_grip_force:.1f} N, peak contact {r.peak_contact_force:.1f} N", flush=True)
seconds = time.perf_counter() - start
print(f"scripted: {totals}, {steps / seconds:.1f} control steps/s", flush=True)
env.close()
ihe._app.close()
