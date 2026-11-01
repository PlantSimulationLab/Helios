"""Measure what the wrist camera sees during episodes, to understand why an image policy does or does not gain from it.

The state policy drives, so that the arm follows the same kind of path with and without leaves, whatever the image policy would have done. The report splits frames by how far the cup is from the fruit.

Usage, from the build directory: python ../diagnose_vision.py --teacher apple_policy [--leaves] [--episodes 5]
"""

import argparse
import pickle

import numpy as np
from stable_baselines3 import PPO

from evaluate import EVALUATION_SEEDS, wilson_interval
from harvest_env import OUTCOME_NAMES, AppleHarvestVecEnv

DISTANCE_BANDS = [(0.5, 10.0, "over 50 cm"), (0.25, 0.5, "25-50 cm"), (0.1, 0.25, "10-25 cm"), (0.0, 0.1, "under 10 cm")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="apple_policy")
    parser.add_argument("--leaves", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    teacher = PPO.load(args.teacher, device="cpu")
    with open(args.teacher + "_normalization.pkl", "rb") as file:
        normalization = pickle.load(file)

    env = AppleHarvestVecEnv(EVALUATION_SEEDS, instances_per_tree=2, threads=args.threads, seed=7000, leaves=args.leaves)
    env.enable_camera(args.image_size)
    observations = env.reset()

    finished = np.zeros(env.num_envs, np.int64)
    outcomes = []
    per_band = {name: [] for _, _, name in DISTANCE_BANDS}
    target_seen = [[] for _ in range(env.num_envs)]
    episode_seen = {}
    while finished.min() < args.episodes:
        for i in range(env.num_envs):
            if finished[i] >= args.episodes:
                continue
            view = env.measure_camera_view(i)
            target_seen[i].append(view.target_fruit)
            distance = float(observations[i, 3])
            for low, high, name in DISTANCE_BANDS:
                if low <= distance < high:
                    per_band[name].append((view.target_fruit, view.other_fruit, view.leaves, view.near_leaves, view.wood))
        actions = np.clip(teacher.predict(normalization.normalize_obs(observations), deterministic=True)[0], -1.0, 1.0)
        observations, _, dones, infos = env.step(actions)
        for i in np.flatnonzero(dones):
            if finished[i] < args.episodes:
                finished[i] += 1
                outcome = OUTCOME_NAMES[infos[i]["outcome"]]
                outcomes.append(outcome)
                seen = np.array(target_seen[i])
                # Before the fruit is touched is when the image has to guide the arm
                episode_seen.setdefault(outcome, []).append((seen.max() if seen.size else 0.0, float(np.mean(seen > 0)) if seen.size else 0.0))
            target_seen[i] = []

    total = len(outcomes)
    successes = outcomes.count("success")
    low, high = wilson_interval(successes, total)
    print(f"State policy, leaves {'on' if args.leaves else 'off'}: {total} episodes. Success {successes / total:.1%} (95% interval {low:.1%} to {high:.1%}), " + ", ".join(f"{name} {outcomes.count(name) / total:.1%}" for name in OUTCOME_NAMES.values() if name != "success"))
    print(f"{'cup to fruit surface':>22} {'frames':>7} {'target visible':>15} {'target':>8} {'other fruit':>12} {'leaves':>8} {'near leaves':>12} {'wood':>7}")
    for _, _, name in DISTANCE_BANDS:
        rows = np.array(per_band[name])
        if rows.size == 0:
            continue
        print(f"{name:>22} {len(rows):7d} {np.mean(rows[:, 0] > 0):15.1%} {rows[:, 0].mean():8.2%} {rows[:, 1].mean():12.2%} {rows[:, 2].mean():8.1%} {rows[:, 3].mean():12.1%} {rows[:, 4].mean():7.1%}")
    all_episodes = [row for rows in episode_seen.values() for row in rows]
    print(f"Episodes in which the target was never visible: {np.mean([row[0] == 0 for row in all_episodes]):.1%}. Share of an episode's frames with it visible: median {np.median([row[1] for row in all_episodes]):.1%}")
    env.close()


if __name__ == "__main__":
    main()
