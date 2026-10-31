"""Train an image-based policy to pick apples by imitating the state-based policy from train.py, and report how much the camera helps.

The state policy knows exactly where the fruit is. The image policy gets the camera on the wrist, what the robot knows about
itself, and a target position with an error that stays the same for a whole episode, as a rough detection from far away would
have. The error is larger than the reach of the cup, so the target position alone is not enough. A blind twin is trained the same
way with no camera, so the difference between them is what vision adds.

Training follows DAgger: run the current policy, have the state policy label every frame with what it would do, add the frames to
the dataset, train on the whole dataset, and repeat.

Usage, from the build directory: python ../train_vision.py --teacher apple_policy [--target-error 0.04] [--blind] [--output vision_policy]
"""

import argparse
import pickle
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn

from evaluate import EVALUATION_SEEDS, wilson_interval
from harvest_env import OUTCOME_NAMES, AppleHarvestVecEnv
from train import TRAIN_SEEDS

# In the observation: 0-2 the exact fruit position relative to the cup, 3 the exact distance to it, and 20-22 the exact offset to the nearest wood are withheld from the image policy
ROBOT_STATE = list(range(4, 20))


class ImagePolicy(nn.Module):
    def __init__(self, state_size, image_size, use_camera):
        super().__init__()
        self.use_camera = use_camera
        feature_size = 0
        if use_camera:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * (image_size // 16) ** 2, 128), nn.ReLU(),
            )
            feature_size = 128
        self.head = nn.Sequential(nn.Linear(feature_size + state_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4), nn.Tanh())

    def forward(self, images, states):
        if self.use_camera:
            return self.head(torch.cat([self.encoder(images.float() / 255.0 - 0.5), states], dim=1))
        return self.head(states)


class Runner:
    """A set of simulations with the target errors of their episodes, and what the image policy is given."""

    def __init__(self, seeds, instances, threads, image_size, use_camera, target_error, seed):
        self.env = AppleHarvestVecEnv(seeds, instances_per_tree=instances, threads=threads, seed=seed)
        self.use_camera = use_camera
        if use_camera:
            self.env.enable_camera(image_size)
        self.random = np.random.default_rng(seed)
        self.target_error = target_error
        self.errors = self.random.normal(0.0, target_error, (self.env.num_envs, 3)).astype(np.float32)
        self.observations = self.env.reset()
        self.image_size = image_size

    def inputs(self):
        images = self.env.render_images().transpose(0, 3, 1, 2).copy() if self.use_camera else np.zeros((self.env.num_envs, 3, self.image_size, self.image_size), np.uint8)
        states = np.concatenate([self.observations[:, ROBOT_STATE], self.observations[:, :3] + self.errors], axis=1)
        return images, states

    def step(self, actions):
        self.observations, _, dones, infos = self.env.step(actions)
        for i in np.flatnonzero(dones):
            self.errors[i] = self.random.normal(0.0, self.target_error, 3)
        return dones, infos


def run_policy(policy, state_mean, state_std, images, states):
    with torch.no_grad():
        normalized = torch.as_tensor((states - state_mean) / state_std, dtype=torch.float32)
        return policy(torch.as_tensor(images), normalized).numpy()


def evaluate_policy(runner, policy, state_mean, state_std, episodes):
    finished = np.zeros(runner.env.num_envs, np.int64)
    records = []
    while finished.min() < episodes:
        images, states = runner.inputs()
        _, infos = runner.step(run_policy(policy, state_mean, state_std, images, states))
        for i, info in enumerate(infos):
            if "outcome" in info and finished[i] < episodes:
                finished[i] += 1
                records.append((OUTCOME_NAMES[info["outcome"]], runner.env.last_episode(i)))
    return records


def summarize(name, records):
    total = len(records)
    counts = {o: sum(1 for r, _ in records if r == o) for o in OUTCOME_NAMES.values()}
    low, high = wilson_interval(counts["success"], total)
    contact = np.median([r.peak_contact_force for _, r in records])
    print(f"{name}: {total} episodes. Success {counts['success'] / total:.1%} (95% interval {low:.1%} to {high:.1%}), " + ", ".join(f"{o} {c / total:.1%}" for o, c in counts.items() if o != "success") + f", median peak force on the tree {contact:.1f} N", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="apple_policy")
    parser.add_argument("--output", default="vision_policy")
    parser.add_argument("--blind", action="store_true", help="No camera: the baseline")
    parser.add_argument("--target-error", type=float, default=0.04, help="Standard deviation in meters of the error in the target position along each axis")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--steps-per-round", type=int, default=600, help="Control steps of all simulations per round")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    use_camera = not args.blind

    teacher = PPO.load(args.teacher, device="cpu")
    with open(args.teacher + "_normalization.pkl", "rb") as file:
        teacher_normalization = pickle.load(file)

    def teacher_actions(observations):
        return np.clip(teacher.predict(teacher_normalization.normalize_obs(observations), deterministic=True)[0], -1.0, 1.0)

    runner = Runner(TRAIN_SEEDS, 2, args.threads, args.image_size, use_camera, args.target_error, args.seed)
    count = runner.env.num_envs
    state_size = len(ROBOT_STATE) + 3
    capacity = args.rounds * args.steps_per_round * count
    # A blind policy has no images, so it needs no room for them
    stored_size = args.image_size if use_camera else 1
    dataset_images = np.zeros((capacity, 3, stored_size, stored_size), np.uint8)
    dataset_states = np.zeros((capacity, state_size), np.float32)
    dataset_actions = np.zeros((capacity, 4), np.float32)
    stored = 0

    policy = ImagePolicy(state_size, args.image_size, use_camera)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    state_mean = np.zeros(state_size, np.float32)
    state_std = np.ones(state_size, np.float32)
    start = time.time()
    simulate_seconds = 0.0
    train_seconds = 0.0
    random_generator = np.random.default_rng(args.seed)

    for round_index in range(args.rounds):
        # The state policy drives at first, and the image policy takes over, so that the dataset covers the mistakes the image policy makes
        teacher_share = max(0.0, 1.0 - round_index / max(1, args.rounds // 2))
        round_start = time.time()
        successes = 0
        episodes = 0
        for _ in range(args.steps_per_round):
            images, states = runner.inputs()
            labels = teacher_actions(runner.observations)
            dataset_images[stored : stored + count] = images if use_camera else 0
            dataset_states[stored : stored + count] = states
            dataset_actions[stored : stored + count] = labels
            stored += count
            driven_by_teacher = random_generator.random(count) < teacher_share
            actions = labels.copy()
            if round_index > 0 and not driven_by_teacher.all():
                learned = run_policy(policy, state_mean, state_std, images, states)
                actions[~driven_by_teacher] = learned[~driven_by_teacher]
            _, infos = runner.step(actions)
            for info in infos:
                if "outcome" in info:
                    episodes += 1
                    successes += info["is_success"]
        simulate_seconds += time.time() - round_start

        round_start = time.time()
        state_mean = dataset_states[:stored].mean(axis=0)
        state_std = dataset_states[:stored].std(axis=0) + 1e-3
        batch_size = 256
        loss_sum = 0.0
        batches = 0
        for _ in range(args.epochs):
            order = random_generator.permutation(stored)
            for first in range(0, stored - batch_size + 1, batch_size):
                batch = order[first : first + batch_size]
                images = torch.as_tensor(dataset_images[batch]) if use_camera else torch.zeros((batch_size, 3, 1, 1), dtype=torch.uint8)
                states = torch.as_tensor((dataset_states[batch] - state_mean) / state_std, dtype=torch.float32)
                loss = nn.functional.mse_loss(policy(images, states), torch.as_tensor(dataset_actions[batch]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                batches += 1
        train_seconds += time.time() - round_start
        print(f"round {round_index + 1}/{args.rounds}  {stored:7d} frames  {(time.time() - start) / 60:6.1f} min  state policy drove {teacher_share:4.0%}  success while collecting {successes / max(1, episodes):5.1%} of {episodes}  loss {loss_sum / max(1, batches):.4f}  {stored / simulate_seconds:5.0f} frames/s while collecting", flush=True)

    torch.save({"policy": policy.state_dict(), "state_mean": state_mean, "state_std": state_std, "image_size": args.image_size, "use_camera": use_camera, "target_error": args.target_error}, args.output + ".pt")
    print(f"Total {(time.time() - start) / 60:.1f} min: {simulate_seconds / 60:.1f} collecting, {train_seconds / 60:.1f} training", flush=True)
    runner.env.close()

    evaluation = Runner(EVALUATION_SEEDS, 2, args.threads, args.image_size, use_camera, args.target_error, 5000 + args.seed)
    policy.eval()
    summarize(f"{'blind' if args.blind else 'camera'} policy, target error {args.target_error} m, {len(EVALUATION_SEEDS)} unseen trees", evaluate_policy(evaluation, policy, state_mean, state_std, args.evaluation_episodes))
    evaluation.env.close()


if __name__ == "__main__":
    main()
