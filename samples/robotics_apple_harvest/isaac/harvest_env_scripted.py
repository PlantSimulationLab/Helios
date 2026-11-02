"""The scripted controller of harvest_env.py, without loading the Helios library."""

import numpy as np


def scripted_actions(observations):
    """Head for the fruit and grip it, and once gripped back straight away from the tree."""
    actions = np.zeros((observations.shape[0], 4), np.float32)
    gripped = observations[:, 16] > 0.5
    actions[:, :3] = 8.0 * observations[:, :3]
    actions[gripped, :3] = [-1.0, 0.0, 0.0]
    actions[:, 3] = 1.0
    return actions
