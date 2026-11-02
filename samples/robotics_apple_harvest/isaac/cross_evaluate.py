"""Evaluate a controller in Helios (MuJoCo) or IsaacSim (PhysX) on the same trees, with the report of evaluate.py.

Run from the sample's build directory with IsaacSim's Python, which has numpy, torch and Stable-Baselines3:
    C:/isaacsim/python.bat ../isaac/cross_evaluate.py --simulator helios --controller scripted --seeds 1,2,3 --episodes 10
    C:/isaacsim/python.bat ../isaac/cross_evaluate.py --simulator isaac --controller policy --policy apple_policy --seeds 301-312
The controller is "scripted", "random", or "policy" (a PPO policy saved by train.py or isaac_train.py, with its _normalization.pkl).
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulator", choices=["helios", "isaac"], required=True)
    parser.add_argument("--controller", choices=["scripted", "random", "policy"], required=True)
    parser.add_argument("--policy", default="apple_policy")
    parser.add_argument("--seeds", required=True, help="tree seeds, e.g. 1,2,3 or 301-312")
    parser.add_argument("--instances", type=int, default=2, help="simulations per tree")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per simulation")
    parser.add_argument("--threads", type=int, default=12, help="Helios threads")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--trees", default="isaac_trees", help="directory of trees exported for IsaacSim")
    parser.add_argument("--output", default="", help="JSON file for the outcome of every episode")
    args, _ = parser.parse_known_args()
    seeds = parse_seeds(args.seeds)

    if args.simulator == "isaac":
        import isaac_harvest_env

        isaac_harvest_env.start_app()
        start = time.perf_counter()
        env = isaac_harvest_env.IsaacHarvestVecEnv(seeds, instances_per_tree=args.instances, seed=args.seed, trees_directory=os.path.abspath(args.trees))
    else:
        import harvest_env

        start = time.perf_counter()
        env = harvest_env.AppleHarvestVecEnv(seeds, instances_per_tree=args.instances, threads=args.threads, seed=args.seed)
    build_seconds = time.perf_counter() - start

    from harvest_env_scripted import scripted_actions

    if args.controller == "scripted":
        choose = scripted_actions
    elif args.controller == "random":
        generator = np.random.default_rng(0)
        choose = lambda o: generator.uniform(-1.0, 1.0, (o.shape[0], 4)).astype(np.float32)
    else:
        from stable_baselines3 import PPO

        model = PPO.load(args.policy, device="cpu")
        with open(args.policy + "_normalization.pkl", "rb") as file:
            normalization = pickle.load(file)
        choose = lambda o: model.predict(normalization.normalize_obs(o), deterministic=True)[0]

    # evaluate() and report() of the Helios sample, so that both simulators are scored by the same code. Importing harvest_env loads
    # the Helios library, which is why this runs from the sample's build directory.
    from evaluate import report
    from harvest_env import evaluate

    records = []
    start = time.perf_counter()
    _, total = evaluate(env, choose, args.episodes, records)
    seconds = time.perf_counter() - start
    report(f"{args.controller} in {args.simulator}, trees {args.seeds}", records)
    print(f"  {total} episodes in {seconds / 60:.1f} min after {build_seconds / 60:.1f} min to build", flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "simulator": args.simulator,
                    "controller": args.controller,
                    "policy": args.policy if args.controller == "policy" else None,
                    "seeds": seeds,
                    "instances": args.instances,
                    "episodes_per_simulation": args.episodes,
                    "build_seconds": build_seconds,
                    "evaluate_seconds": seconds,
                    "episodes": [dict(outcome=o, **{k: getattr(r, k) for k in ("detach_force", "final_stem_load", "final_grip_force", "pull_x", "pull_y", "pull_z", "peak_contact_force", "peak_wood_motion", "peak_tool_speed", "steps_to_grip", "steps")}) for o, r in records],
                },
                f,
                indent=1,
            )
    env.close()


if __name__ == "__main__":
    main()
