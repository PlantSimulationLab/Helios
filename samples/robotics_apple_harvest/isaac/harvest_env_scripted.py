"""The scripted controller of harvest_env.py, without loading the Helios library."""

import numpy as np


def scripted_actions(observations):
    """Line the cup up with the fruit, 8 cm short of it, then push straight in until the lip seals, and once gripped pull straight out of the cup.

    The cup points horizontally away from the robot's turret, so it is lined up when the fruit is on the horizontal line from the turret through
    the cup, at the cup's height.
    """
    actions = np.zeros((observations.shape[0], 4), np.float32)
    toward_fruit = observations[:, :3]
    # The fruit from the turret's column, horizontally: the cup's position from it is in observations 4 and 5
    from_turret = np.stack([observations[:, 4] - 0.15 + toward_fruit[:, 0], observations[:, 5] + toward_fruit[:, 1]], axis=1)
    line = from_turret / np.maximum(np.linalg.norm(from_turret, axis=1, keepdims=True), 1e-6)
    along = np.sum(toward_fruit[:, :2] * line, axis=1)
    sideways = toward_fruit[:, :2] - along[:, None] * line
    lined_up = np.hypot(np.linalg.norm(sideways, axis=1), toward_fruit[:, 2]) < 0.004
    # To 8 cm short of the fruit while lining up (holding back alone would leave the arm trying to move only sideways or up and down from where it
    # is, which from the tucked pose it cannot), then in to where the lip meets it, which stops the cup
    wanted_along = np.where(lined_up, 0.0, 0.08)
    actions[:, :2] = 8.0 * (sideways + (along - wanted_along)[:, None] * line)
    actions[:, 2] = 8.0 * toward_fruit[:, 2]
    gripped = observations[:, 16] > 0.5
    actions[gripped, :3] = -toward_fruit[gripped] / np.maximum(np.linalg.norm(toward_fruit[gripped], axis=1, keepdims=True), 1e-6)
    actions[:, 3] = 1.0
    return actions
