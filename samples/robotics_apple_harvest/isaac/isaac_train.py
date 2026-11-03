"""Train the apple picking policy in IsaacSim (PhysX), exactly as train.py trains it in Helios (MuJoCo).

Same trees, simulations per tree, PPO settings and observation normalization as train.py. One difference: only one IsaacSim environment
can exist in a process, so there is no evaluation on held-out trees during training; evaluate afterwards with cross_evaluate.py.

From the sample's build directory: C:/isaacsim/python.bat ../isaac/isaac_train.py --steps 500000 --output apple_policy_isaac_seed0
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import isaac_harvest_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--report-every", type=int, default=50000)
    parser.add_argument("--output", default="apple_policy_isaac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trees", default="isaac_trees")
    parser.add_argument("--settling-steps", type=int, default=None, help="only for quick checks; by default trees settle as in harvest_env.cpp")
    args, _ = parser.parse_known_args()

    isaac_harvest_env.start_app()
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from train import TRAIN_SEEDS

    class Progress(BaseCallback):
        """Every so often, the success on the training trees and the speed of training."""

        def __init__(self, every_steps):
            super().__init__()
            self.every_steps = every_steps
            self.next_report = every_steps
            self.start = time.time()
            self.recent = []
            self.outcomes = []

        def _on_step(self):
            for info in self.locals["infos"]:
                if "outcome" in info:
                    self.recent.append(info["is_success"])
                    self.outcomes.append(info["outcome"])
            if self.num_timesteps >= self.next_report:
                self.next_report += self.every_steps
                elapsed = time.time() - self.start
                success = float(np.mean(self.recent)) if self.recent else float("nan")
                counts = {name: self.outcomes.count(code) / max(1, len(self.outcomes)) for code, name in isaac_harvest_env.OUTCOME_NAMES.items()}
                self.recent = []
                self.outcomes = []
                print(f"steps {self.num_timesteps:8d}  {elapsed / 60:6.1f} min  {self.num_timesteps / elapsed:5.1f} steps/s  training success {success:5.1%}  " + "  ".join(f"{k} {v:5.1%}" for k, v in counts.items()), flush=True)
            return True

    torch.set_num_threads(2)
    start = time.time()
    raw = isaac_harvest_env.IsaacHarvestVecEnv(TRAIN_SEEDS, instances_per_tree=2, seed=args.seed, trees_directory=os.path.abspath(args.trees), settling_steps=args.settling_steps)
    print(f"built and settled {raw.num_envs} simulations in {(time.time() - start) / 60:.1f} min", flush=True)
    env = VecNormalize(raw, norm_obs=True, norm_reward=True, gamma=0.99)

    # The settings of train.py
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=1024, n_epochs=8, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001, policy_kwargs=dict(net_arch=[128, 128]), seed=args.seed, device="cpu", verbose=0)
    model.learn(args.steps, callback=Progress(args.report_every))
    model.save(args.output)
    env.save(args.output + "_normalization.pkl")
    print(f"saved {args.output} after {(time.time() - start) / 60:.1f} min", flush=True)
    raw.close()


if __name__ == "__main__":
    main()
