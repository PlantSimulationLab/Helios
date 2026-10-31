"""Compare a trained policy with a scripted controller and with random actions, on trees it has never seen, and check how each one succeeds.

Usage, from the build directory: python ../evaluate.py [--policy apple_policy] [--episodes 10] [--threads 6]
"""

import argparse
import pickle

import numpy as np
from stable_baselines3 import PPO

from harvest_env import AppleHarvestVecEnv, evaluate, scripted_actions

# Not used in training, nor watched during it, nor used in choosing how the task is set up
EVALUATION_SEEDS = list(range(301, 313))


def wilson_interval(successes, total, z=1.96):
    """95% confidence interval of a proportion."""
    proportion = successes / total
    center = (proportion + z * z / (2 * total)) / (1 + z * z / total)
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return center - half_width, center + half_width


def report(name, records):
    total = len(records)
    successes = [record for outcome, record in records if outcome == "success"]
    low, high = wilson_interval(len(successes), total)
    counts = {outcome: sum(1 for o, _ in records if o == outcome) / total for outcome in ("success", "timeout", "slipped", "knocked_off", "collision")}
    print(f"\n{name}: {total} episodes. Success {counts['success']:.1%} (95% interval {low:.1%} to {high:.1%}), timeout {counts['timeout']:.1%}, slipped {counts['slipped']:.1%}, knocked off {counts['knocked_off']:.1%}, collision {counts['collision']:.1%}")

    def stats(values):
        values = np.array(values)
        return f"median {np.median(values):7.3f}  p90 {np.percentile(values, 90):7.3f}  max {values.max():7.3f}"

    print(f"  all episodes:     peak contact on the tree (N) {stats([r.peak_contact_force for _, r in records])}")
    print(f"                    peak motion of bearing wood (m) {stats([r.peak_wood_motion for _, r in records])}")
    print(f"                    peak tool speed (m/s) {stats([r.peak_tool_speed for _, r in records])}")
    if successes:
        print(f"  successes:        episode length (s) {stats([0.04 * r.steps for r in successes])}")
        print(f"                    grip force at detachment / force to detach {stats([r.final_grip_force / r.detach_force for r in successes])}")
        print(f"                    stem load at detachment / force to detach  {stats([r.final_stem_load / r.detach_force for r in successes])}")
        print(f"                    pull direction: mean x {np.mean([r.pull_x for r in successes]):+.2f} y {np.mean([r.pull_y for r in successes]):+.2f} z {np.mean([r.pull_z for r in successes]):+.2f} (-x is toward the robot)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="apple_policy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    env = AppleHarvestVecEnv(EVALUATION_SEEDS, instances_per_tree=2, threads=args.threads, seed=2000)

    model = PPO.load(args.policy, device="cpu")
    with open(args.policy + "_normalization.pkl", "rb") as file:
        normalization = pickle.load(file)

    random_generator = np.random.default_rng(0)
    controllers = {
        "trained policy": lambda o: model.predict(normalization.normalize_obs(o), deterministic=True)[0],
        "scripted": scripted_actions,
        "random": lambda o: random_generator.uniform(-1.0, 1.0, (o.shape[0], 4)).astype(np.float32),
    }
    for name, choose_actions in controllers.items():
        records = []
        evaluate(env, choose_actions, args.episodes, records)
        report(name, records)
    env.close()


if __name__ == "__main__":
    main()
