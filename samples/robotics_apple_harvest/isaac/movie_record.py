"""Record episodes of a controller on one tree, in Helios (MuJoCo) or IsaacSim (PhysX), for movie_render.py to show.

Every control step of every episode is kept: the pose of every link of the tree, the robot's joints, and whether the cup grips the fruit,
including the step at which the episode ended (before the simulation is reset for the next one).

From the sample's build directory, with IsaacSim's Python:
    C:/isaacsim/python.bat ../isaac/movie_record.py --simulator helios --policy apple_policy_helios_seed0 --tree 301 --output helios_301.npz
    C:/isaacsim/python.bat ../isaac/movie_record.py --simulator isaac --policy apple_policy_helios_seed0 --tree 301 --output isaac_301.npz
    C:/isaacsim/python.bat ../isaac/movie_record.py --simulator helios --policy scripted --tree 301 --output scripted_301.npz

Helios records each link's world pose in its own link order. IsaacSim records the world pose of each of its bodies (welded links merged into the
body that carries them, see tree_preparation.py) with the map from Helios links to those bodies, and movie_render.py turns them into link poses.
"""

import argparse
import ctypes
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

OUTCOME_NAMES = {1: "success", 2: "timeout", 3: "slipped", 4: "knocked_off", 5: "collision"}


def helios_library():
    """The Helios environment library, with the functions for showing simulations declared."""
    import harvest_env

    lib = harvest_env._lib
    floats = ctypes.POINTER(ctypes.c_float)
    lib.harvest_env_create_for_movie.restype = ctypes.c_void_p
    lib.harvest_env_create_for_movie.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.harvest_env_link_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.harvest_env_link_name.restype = ctypes.c_char_p
    lib.harvest_env_link_name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.harvest_env_rest_link_poses.argtypes = [ctypes.c_void_p, ctypes.c_int, floats]
    lib.harvest_env_display_state.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, floats, floats]
    lib.harvest_env_target_fruit.argtypes = [ctypes.c_void_p, ctypes.c_int, floats]
    lib.harvest_env_enable_movie_camera.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.harvest_env_render_movie_frame.argtypes = [ctypes.c_void_p, ctypes.c_int, floats, floats, floats, floats, ctypes.POINTER(ctypes.c_ubyte)]
    lib.harvest_env_render_wrist_frame.argtypes = [ctypes.c_void_p, ctypes.c_int, floats, floats, ctypes.POINTER(ctypes.c_ubyte)]
    lib.harvest_env_set_follow_through.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return lib


def as_floats(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def helios_link_names(lib, env, index=0):
    return [lib.harvest_env_link_name(env, index, i).decode() for i in range(lib.harvest_env_link_count(env, index))]


def load_policy(path):
    """The controller to record: a PPO policy saved by train.py or isaac_train.py, or, for "scripted", harvest_env.py's scripted controller."""
    if path == "scripted":
        from harvest_env_scripted import scripted_actions

        return scripted_actions
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    with open(path + "_normalization.pkl", "rb") as file:
        normalization = pickle.load(file)
    return lambda o: model.predict(normalization.normalize_obs(o), deterministic=True)[0]


class Recorder:
    """Episodes of every simulation, each a dict of per-step arrays and how it ended."""

    def __init__(self, count):
        self.current = [None] * count
        self.finished = [[] for _ in range(count)]

    def start(self, i, target):
        self.current[i] = {"target": np.asarray(target, np.float32), "poses": [], "joints": [], "gripped": []}

    def add(self, i, poses, joints, gripped):
        episode = self.current[i]
        episode["poses"].append(np.asarray(poses, np.float32))
        episode["joints"].append(np.asarray(joints, np.float32))
        episode["gripped"].append(bool(gripped))

    def end(self, i, outcome, record):
        episode = self.current[i]
        episode["outcome"] = OUTCOME_NAMES[outcome]
        episode["detach_force"] = float(record.detach_force)
        episode["final_grip_force"] = float(record.final_grip_force)
        episode["peak_contact_force"] = float(record.peak_contact_force)
        episode["detach_step"] = int(record.steps)  # the control step at which the episode was decided; later steps are follow-through
        self.finished[i].append(episode)
        self.current[i] = None


def record_helios(args, choose):
    import harvest_env

    lib = helios_library()
    # Without leaves, as in training and evaluation
    env = harvest_env.AppleHarvestVecEnv([args.tree], instances_per_tree=args.instances, threads=args.instances, seed=args.seed, leaves=False)
    lib.harvest_env_set_follow_through(env._env, args.follow_through)
    names = helios_link_names(lib, env._env)
    link_count = len(names)
    poses = np.zeros((link_count, 7), np.float32)
    joints = np.zeros(6, np.float32)
    target = np.zeros(3, np.float32)

    def state(i, last_episode):
        if not lib.harvest_env_display_state(env._env, i, 1 if last_episode else 0, as_floats(poses), as_floats(joints)):
            raise RuntimeError("harvest_env_display_state failed")
        return poses.copy(), joints.copy()

    recorder = Recorder(env.num_envs)
    observations = env.reset()

    def begin(i):
        lib.harvest_env_target_fruit(env._env, i, as_floats(target))
        recorder.start(i, target.copy())
        recorder.add(i, *state(i, False), observations[i, 16] > 0.5)

    for i in range(env.num_envs):
        begin(i)
    while min(len(f) for f in recorder.finished) < args.episodes:
        observations, _, dones, infos = env.step(choose(observations))
        for i in range(env.num_envs):
            if dones[i]:
                gripped = infos[i]["terminal_observation"][16] > 0.5
                recorder.add(i, *state(i, True), gripped)
                recorder.end(i, infos[i]["outcome"], env.last_episode(i))
                print(f"helios sim {i} episode {len(recorder.finished[i])}: {recorder.finished[i][-1]['outcome']} in {len(recorder.finished[i][-1]['poses'])} frames", flush=True)
                begin(i)
            else:
                recorder.add(i, *state(i, False), observations[i, 16] > 0.5)
    env.close()
    return {"simulator": "helios", "link_names": names}, recorder


def record_isaac(args, choose):
    import isaac_harvest_env as ihe
    from harvest_scene import ROBOT_JOINTS

    env = ihe.IsaacHarvestVecEnv([args.tree], instances_per_tree=args.instances, seed=args.seed, trees_directory=os.path.abspath(args.trees), fruit_hold=args.fruit_hold)
    env.follow_through_steps = args.follow_through
    tv = env.tree_views[0]
    view = tv["view"]
    body_names = list(view.shared_metatype.link_names)
    articulation_bodies = len(body_names)
    rows = tv["rows"].cpu().numpy()  # simulation of each row of the tree view
    if env.soft:
        # Candidate fruit are bodies of their own, outside the tree's articulation
        body_names += [c["link"] for c in env.trees[0]["description"]["candidates"]]
        fruit_rows = tv["fruit_rows"].cpu().numpy()  # (view rows, candidates) rows of the fruit view
    owners = env.trees[0]["owners"]
    recorder = Recorder(env.num_envs)
    in_step = [False]

    def snapshot():
        """Per simulation: body poses (bodies, 7: x y z qx qy qz qw) relative to its origin, robot joints in m and degrees as Helios gives them."""
        transforms = view.get_link_transforms().view(view.count, articulation_bodies, 7).cpu().numpy().copy()
        bodies = np.zeros((env.num_envs, len(body_names), 7), np.float32)
        bodies[rows, :articulation_bodies] = transforms
        if env.soft:
            fruit = tv["fruits"].get_transforms().view(-1, 7).cpu().numpy()
            bodies[rows, articulation_bodies:] = fruit[fruit_rows]
        bodies[:, :, :3] -= env.env_origins[:, None, :]
        dofs = env.robots.get_dof_positions().view(env.robots.count, -1)[env.robot_order][:, env.dof_order].cpu().numpy().astype(np.float64)
        dofs[:, 1:] = np.degrees(dofs[:, 1:])
        return bodies, dofs, env.gripped.cpu().numpy().copy()

    original_reset = env._reset
    pending = {}

    def reset_with_snapshot(envs):
        # The pose at the end of an episode, before the simulation is put back for the next one
        if in_step[0]:
            bodies, dofs, gripped = snapshot()
            for i in envs:
                pending[int(i)] = (bodies[i], dofs[i], gripped[i])
        original_reset(envs)

    env._reset = reset_with_snapshot
    observations = env.reset()

    def begin(i):
        recorder.start(i, env.settled_fruit[i].cpu().numpy() - env.env_origins[i])

    bodies, dofs, gripped = snapshot()
    for i in range(env.num_envs):
        begin(i)
        recorder.add(i, bodies[i], dofs[i], gripped[i])
    while min(len(f) for f in recorder.finished) < args.episodes:
        pending.clear()
        in_step[0] = True
        observations, _, dones, infos = env.step(choose(observations))
        in_step[0] = False
        bodies, dofs, gripped = snapshot()
        for i in range(env.num_envs):
            if dones[i]:
                recorder.add(i, *pending[i])
                recorder.end(i, infos[i]["outcome"], env.last_episode(i))
                print(f"isaac sim {i} episode {len(recorder.finished[i])}: {recorder.finished[i][-1]['outcome']} in {len(recorder.finished[i][-1]['poses'])} frames", flush=True)
                begin(i)
            recorder.add(i, bodies[i], dofs[i], gripped[i])
    env.close()
    return {"simulator": "isaac", "body_names": body_names, "owners": owners, "robot_joints": ROBOT_JOINTS}, recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulator", choices=["helios", "isaac"], required=True)
    parser.add_argument("--policy", required=True, help="a policy saved by train.py or isaac_train.py, with its _normalization.pkl, or scripted for the scripted controller")
    parser.add_argument("--tree", type=int, default=301)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=6, help="episodes per simulation")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--trees", default="isaac_trees", help="directory of trees exported for IsaacSim")
    parser.add_argument("--fruit-hold", choices=["rigid", "soft"], default="soft", help="how IsaacSim holds candidate fruit; only a soft hold can release a fruit to be carried away")
    parser.add_argument("--follow-through", type=int, default=25, help="control steps (25 per second) a success goes on for, with the cup carrying the fruit away; 0 to end at detachment")
    parser.add_argument("--output", required=True, help=".npz file")
    args, _ = parser.parse_known_args()

    if args.simulator == "helios":
        meta, recorder = record_helios(args, load_policy(args.policy))
    else:
        import isaac_harvest_env as ihe

        # IsaacSim starts before anything imports torch, as in cross_evaluate.py
        ihe.start_app()
        meta, recorder = record_isaac(args, load_policy(args.policy))

    episodes = [e for simulation in recorder.finished for e in simulation]
    arrays = {}
    summary = []
    for k, e in enumerate(episodes):
        arrays[f"poses_{k}"] = np.stack(e["poses"])
        arrays[f"joints_{k}"] = np.stack(e["joints"])
        arrays[f"gripped_{k}"] = np.array(e["gripped"])
        arrays[f"target_{k}"] = e["target"]
        summary.append({key: e[key] for key in ("outcome", "detach_force", "final_grip_force", "peak_contact_force", "detach_step")})
    meta.update({"tree": args.tree, "policy": os.path.basename(args.policy), "episodes": summary, "follow_through": args.follow_through})
    if args.simulator == "isaac":
        meta["fruit_hold"] = args.fruit_hold
    np.savez_compressed(args.output, meta=json.dumps(meta), **arrays)
    print(f"saved {len(episodes)} episodes to {args.output}", flush=True)
    if args.simulator == "isaac":
        import isaac_harvest_env as ihe

        ihe._app.close()


if __name__ == "__main__":
    main()
