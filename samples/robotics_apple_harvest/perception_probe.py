"""Measure how well a network can read the fruit and wood positions from the camera, apart from any control.

A network is trained to predict the exact position of the fruit relative to the cup, and of the nearest wood, on unseen trees. Without the camera, it shows what the rest of the input is already worth.

The arm must not give the answer away. If the state policy drove, the arm's motion would point at the true fruit position, and a network could read it from that rather than from the image. So the driver knows only the target with its error, and wanders around it at random, and the network gets only the image, that target, and the cup position, with no velocity, contact, grip, or time.

Usage, from the build directory: python ../perception_probe.py [--leaves] [--blind] [--encoder spatial]
"""

import argparse
import os
import numpy as np
import torch
from torch import nn

from evaluate import EVALUATION_SEEDS
from harvest_env import usable_tree_seeds
from train_vision import Runner

# In the input to the network from Runner.inputs(): the cup position, then the fruit position with its error, which come last
CUP_POSITION = [0, 1, 2]
GIVEN_TARGET = [-3, -2, -1]
INPUTS = CUP_POSITION + GIVEN_TARGET

BANDS = [(0.25, 10.0, "over 25 cm"), (0.1, 0.25, "10-25 cm"), (0.0, 0.1, "under 10 cm")]


class SpatialSoftmax(nn.Module):
    """The image coordinates of where each feature map is strongest, which keeps exact positions that flattening a coarse grid of features loses."""

    def forward(self, features):
        batch, channels, height, width = features.shape
        weights = torch.softmax(features.view(batch, channels, -1), dim=2).view(batch, channels, height, width)
        ys = torch.linspace(-1.0, 1.0, height, device=features.device)
        xs = torch.linspace(-1.0, 1.0, width, device=features.device)
        expected_x = (weights.sum(dim=2) * xs).sum(dim=2)
        expected_y = (weights.sum(dim=3) * ys).sum(dim=2)
        return torch.cat([expected_x, expected_y], dim=1)


class Predictor(nn.Module):
    def __init__(self, state_size, image_size, use_camera, outputs, encoder="plain"):
        super().__init__()
        self.use_camera = use_camera
        features = 0
        if use_camera and encoder == "plain":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * (image_size // 16) ** 2, 128), nn.ReLU(),
            )
            features = 128
        elif use_camera:
            channels = 32
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(64, channels, 1),
                SpatialSoftmax(),
            )
            features = 2 * channels
        self.head = nn.Sequential(nn.Linear(features + state_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, outputs))

    def forward(self, images, states):
        if self.use_camera:
            return self.head(torch.cat([self.encoder(images.float() / 255.0 - 0.5), states], dim=1))
        return self.head(states)


def collect(runner, frames, use_camera, seed):
    """Drive toward the given target and wander around it, knowing nothing more than a network with no camera would."""
    random_generator = np.random.default_rng(seed)
    count = runner.env.num_envs
    steps = frames // count
    size = runner.image_size if use_camera else 1
    images = np.zeros((steps * count, 3, size, size), np.uint8)
    states = np.zeros((steps * count, len(INPUTS)), np.float32)
    # The exact fruit position and the exact offset to the nearest wood
    labels = np.zeros((steps * count, 6), np.float32)
    distances = np.zeros(steps * count, np.float32)
    wander = random_generator.normal(0.0, 0.1, (count, 3))
    for s in range(steps):
        frame_images, frame_states = runner.inputs()
        if use_camera:
            images[s * count : (s + 1) * count] = frame_images
        # Runner.inputs() gives the robot state with the cup position first, then the target with its error
        states[s * count : (s + 1) * count] = frame_states[:, INPUTS]
        labels[s * count : (s + 1) * count, :3] = runner.observations[:, :3]
        labels[s * count : (s + 1) * count, 3:] = runner.observations[:, 20:23]
        distances[s * count : (s + 1) * count] = runner.observations[:, 3]
        wander = 0.97 * wander + random_generator.normal(0.0, 0.03, (count, 3))
        actions = np.zeros((count, 4), np.float32)
        actions[:, :3] = np.clip(6.0 * (frame_states[:, GIVEN_TARGET] + wander), -1.0, 1.0)
        dones, _ = runner.step(actions)
        wander[dones] = random_generator.normal(0.0, 0.1, (int(dones.sum()), 3))
    return images, states, labels, distances


def report(title, predictions, states, labels, distances):
    """Errors in the frame of the camera, which looks along +x from behind the cup: depth is along x, and y and z are across the image."""
    correction = predictions[:, :3] + states[:, 3:6]
    fruit = np.abs(correction - labels[:, :3])
    given = np.abs(states[:, 3:6] - labels[:, :3])
    wood = np.linalg.norm(predictions[:, 3:] - labels[:, 3:], axis=1)
    print(title)
    print(f"{'cup to fruit surface':>22} {'frames':>7} | median error given, cm: {'depth':>6} {'across':>7} {'up':>5} | predicted: {'depth':>6} {'across':>7} {'up':>5} | {'wood':>6} {'wood dist':>9}")
    for low, high, name in BANDS:
        band = (distances >= low) & (distances < high)
        if band.any():
            g = 100 * np.median(given[band], axis=0)
            f = 100 * np.median(fruit[band], axis=0)
            print(f"{name:>22} {band.sum():7d} | {'':24}{g[0]:6.1f} {g[1]:7.1f} {g[2]:5.1f} | {'':11}{f[0]:6.1f} {f[1]:7.1f} {f[2]:5.1f} | {100 * np.median(wood[band]):6.1f} {100 * np.median(np.linalg.norm(labels[band, 3:], axis=1)):9.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaves", action="store_true")
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--encoder", default="plain", choices=["plain", "spatial"])
    parser.add_argument("--target-error", type=float, default=0.04)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--training-frames", type=int, default=80000)
    parser.add_argument("--test-frames", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-trees", type=int, default=24)
    parser.add_argument("--no-position", action="store_true", help="Leave out where the cup is along x and z, by which a network can recognize a fruit it has seen. It keeps y relative to the platform, which sets the direction of the camera.")
    parser.add_argument("--jitter", type=float, default=0.0, help="Random change to the brightness and color of each training image, so that a network cannot lean on how a particular tree looks")
    parser.add_argument("--cache", default=None, help="Directory for the collected frames, since collecting is the slow part")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(4)
    use_camera = not args.blind

    cache_file = None
    if args.cache and use_camera:
        cache_file = os.path.join(args.cache, f"frames_{args.image_size}_{'leaves' if args.leaves else 'bare'}_{args.training_trees}_{args.training_frames}.npz")
    if cache_file and os.path.exists(cache_file):
        data = np.load(cache_file)
        images, states, labels, distances = data["images"], data["states"], data["labels"], data["distances"]
        test_images, test_states, test_labels, test_distances = data["test_images"], data["test_states"], data["test_labels"], data["test_distances"]
    else:
        runner = Runner(usable_tree_seeds(args.training_trees), 1, args.threads, args.image_size, use_camera, args.target_error, 0, args.leaves)
        images, states, labels, distances = collect(runner, args.training_frames, use_camera, 0)
        runner.env.close()
        test_runner = Runner(EVALUATION_SEEDS, 2, args.threads, args.image_size, use_camera, args.target_error, 5000, args.leaves)
        test_images, test_states, test_labels, test_distances = collect(test_runner, args.test_frames, use_camera, 1)
        test_runner.env.close()
        if cache_file:
            np.savez(cache_file, images=images, states=states, labels=labels, distances=distances, test_images=test_images, test_states=test_states, test_labels=test_labels, test_distances=test_distances)

    # The network predicts the correction to the given target, which is what the image has to supply, rather than a position that the input already nearly gives
    def to_targets(state_batch, label_batch):
        targets = label_batch.copy()
        targets[:, :3] = label_batch[:, :3] - state_batch[:, 3:6]
        return targets

    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-3
    targets = to_targets(states, labels)
    target_scale = targets.std(axis=0) + 1e-3
    # Which of the inputs the network gets. The given target stays, since the correction is relative to it.
    used = [1, 3, 4, 5] if args.no_position else list(range(states.shape[1]))
    model = Predictor(len(used), args.image_size, use_camera, 6, args.encoder).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    random_generator = np.random.default_rng(0)
    batch_size = 256
    for epoch in range(args.epochs):
        order = random_generator.permutation(len(states))
        for first in range(0, len(states) - batch_size + 1, batch_size):
            batch = order[first : first + batch_size]
            x = torch.as_tensor(images[batch]).to(args.device)
            if args.jitter > 0.0:
                # Applied here in floating point, then back to bytes, since the network scales its input
                gains = 1.0 + args.jitter * (2.0 * torch.rand(batch_size, 3, 1, 1, device=args.device) - 1.0)
                offsets = 255.0 * 0.5 * args.jitter * (2.0 * torch.rand(batch_size, 3, 1, 1, device=args.device) - 1.0)
                x = torch.clamp(x.float() * gains + offsets, 0.0, 255.0)
            y = torch.as_tensor(((states[batch] - state_mean) / state_std)[:, used], dtype=torch.float32).to(args.device)
            target = torch.as_tensor(targets[batch] / target_scale, dtype=torch.float32).to(args.device)
            loss = nn.functional.mse_loss(model(x, y), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    def predict(image_set, state_set):
        predictions = []
        with torch.no_grad():
            for first in range(0, len(state_set), batch_size):
                x = torch.as_tensor(image_set[first : first + batch_size]).to(args.device)
                y = torch.as_tensor(((state_set[first : first + batch_size] - state_mean) / state_std)[:, used], dtype=torch.float32).to(args.device)
                predictions.append(model(x, y).cpu().numpy() * target_scale)
        return np.concatenate(predictions)

    name = "blind" if args.blind else f"camera, {args.encoder} encoder"
    name += f", {args.training_trees} trees" + (", no cup position" if args.no_position else "") + (f", jitter {args.jitter}" if args.jitter > 0.0 else "")
    leaves = "on" if args.leaves else "off"
    report(f"{name}, leaves {leaves}, TRAINING trees", predict(images[:12000], states[:12000]), states[:12000], labels[:12000], distances[:12000])
    report(f"{name}, leaves {leaves}, unseen trees", predict(test_images, test_states), test_states, test_labels, test_distances)


if __name__ == "__main__":
    main()
