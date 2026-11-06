"""The scripted controller of harvest_env.py, without loading the Helios library."""

import numpy as np


def scripted_actions(observations):
    """Line the cup up with the fruit, short of it, then push straight in until the lip seals, and once gripped pull straight out of the cup.

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
    off_line = np.hypot(np.linalg.norm(sideways, axis=1), toward_fruit[:, 2])
    lined_up = off_line < 0.004
    # While lining up, toward a point short of the fruit by 8 cm plus three quarters of the distance still off the line, up to 15 cm more, then in to
    # where the lip meets the fruit, which stops the cup. Staying back while far off the line moves the cup sideways and up or down outside the
    # canopy, where the 32 cm tool sweeping across would otherwise hit the branches around the fruit; staying back further leaves the arm short of
    # where it can reach. Holding back alone, without coming in, would leave the arm trying to move only sideways or up and down from the tucked
    # pose, which it cannot.
    wanted_along = np.where(lined_up, 0.0, 0.08 + np.minimum(0.75 * off_line, 0.15))
    actions[:, :2] = 8.0 * (sideways + (along - wanted_along)[:, None] * line)
    actions[:, 2] = 8.0 * toward_fruit[:, 2]
    gripped = observations[:, 16] > 0.5
    actions[gripped, :3] = -toward_fruit[gripped] / np.maximum(np.linalg.norm(toward_fruit[gripped], axis=1, keepdims=True), 1e-6)
    actions[:, 3] = 1.0
    return actions
