"""The scripted controller of harvest_env.py, without loading the Helios library."""

import numpy as np


def scripted_actions(observations):
    """Head for the fruit and grip it, and once gripped pull straight out of the cup's face, away from the fruit's center, so as not to peel the seal."""
    actions = np.zeros((observations.shape[0], 4), np.float32)
    gripped = observations[:, 16] > 0.5
    actions[:, :3] = 8.0 * observations[:, :3]
    toward_fruit = observations[gripped, :3]
    actions[gripped, :3] = -toward_fruit / np.maximum(np.linalg.norm(toward_fruit, axis=1, keepdims=True), 1e-6)
    actions[:, 3] = 1.0
    return actions
