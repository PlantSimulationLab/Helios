"""Kinematics of harvest_arm.urdf, and Robot::solveInverseKinematics() from the Helios robotics plug-in, for many arms at once.

Joint order and units are those of the model: drive (m along y), then yaw, shoulder, elbow, wrist, roll (rad).
"""

import numpy as np

# Joint origins in the parent link's frame, and axes in it, from harvest_arm.urdf
ORIGINS = np.array([[0, 0, 0], [0.15, 0, 0], [0, 0, 0.75], [0.65, 0, 0], [0.6, 0, 0], [0, 0, 0]], dtype=float)
AXES = np.array([[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, -1, 0], [0, -1, 0], [1, 0, 0]], dtype=float)
IS_HINGE = np.array([False, True, True, True, True, True])
LOWER = np.array([-10.0, -2.5, -0.8, -2.6, -2.2, -3.0])
UPPER = np.array([10.0, 2.5, 2.4, 2.6, 2.2, 3.0])


def _axis_rotation(axis, angle):
    """Rotation matrices (N, 3, 3) by angles (N,) about one fixed unit axis."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    return np.stack(
        [
            np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], -1),
            np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], -1),
            np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], -1),
        ],
        -2,
    )


def forward(q, base, tool_point, tool_axis):
    """Tool point and axis in the world, joint origins and world axes, and the tool frame, for joint positions q (N, 6) and bases (N, 3)."""
    n = q.shape[0]
    rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    position = base.astype(float).copy()
    joint_origins = np.zeros((n, 6, 3))
    joint_axes = np.zeros((n, 6, 3))
    for j in range(6):
        position = position + np.einsum("nij,j->ni", rotation, ORIGINS[j])
        joint_origins[:, j] = position
        joint_axes[:, j] = np.einsum("nij,j->ni", rotation, AXES[j])
        if IS_HINGE[j]:
            rotation = rotation @ _axis_rotation(AXES[j], q[:, j])
        else:
            position = position + joint_axes[:, j] * q[:, j : j + 1]
    point = position + np.einsum("nij,j->ni", rotation, tool_point)
    axis = np.einsum("nij,j->ni", rotation, tool_axis)
    return point, axis, joint_origins, joint_axes, rotation


def solve_inverse_kinematics(q_start, base, tool_point, tool_axis, target_position, target_direction, locked):
    """Robot::solveInverseKinematics() for N arms: damped least squares with a line search, nudges out of singular poses, and joint limits.

    Returns (converged (N,), joint positions (N, 6)). A run that does not converge returns the best pose found, as the C++ does.
    """
    max_iterations = 200
    position_tolerance = 1e-4
    axis_tolerance = 1e-3
    damping = 0.01
    max_step = 0.2
    stall_improvement = 1e-5
    singularity_nudge = 0.1
    max_nudges = 5

    n = q_start.shape[0]
    direction = target_direction / np.linalg.norm(target_direction, axis=1, keepdims=True)
    q = q_start.astype(float).copy()
    accepted = q.copy()
    accepted_error = np.full(n, np.inf)
    line_search_scale = np.ones(n)
    stalled_error = np.full(n, np.inf)
    nudges = np.zeros(n, dtype=int)
    steps = np.zeros((n, 6))
    converged = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    free = ~np.asarray(locked)
    nudge = np.where(np.arange(6) % 2 == 0, singularity_nudge, -singularity_nudge) * (IS_HINGE & free)

    def errors(q_now):
        point, axis, origins, axes, _ = forward(q_now, base, tool_point, tool_axis)
        error = np.concatenate([target_position - point, np.cross(axis, direction)], axis=1)
        opposite = np.einsum("ni,ni->n", axis, direction) < 0
        return error, opposite, point, origins, axes

    for iteration in range(max_iterations + 1):
        active = ~done
        if not active.any():
            break
        error, opposite, _, _, _ = errors(q)
        total = np.linalg.norm(error, axis=1) + opposite
        reached = active & (np.linalg.norm(error[:, :3], axis=1) < position_tolerance) & (np.linalg.norm(error[:, 3:], axis=1) < axis_tolerance) & ~opposite
        converged |= reached
        accepted[reached] = q[reached]
        done |= reached
        if iteration == max_iterations:
            break
        working = active & ~reached

        # A step that made the error worse is taken back and tried at half the length
        retry = working & (total >= accepted_error) & (line_search_scale > 1e-3)
        line_search_scale[retry] *= 0.5
        q[retry] = np.clip(accepted[retry] + line_search_scale[retry, None] * steps[retry], LOWER, UPPER)

        proceed = working & ~retry
        improved = proceed & (total < accepted_error)
        accepted_error[improved] = total[improved]
        accepted[improved] = q[improved]
        q[proceed] = accepted[proceed]  # carry on from the best pose found
        line_search_scale[proceed] = 1.0

        # Stalled at a singular pose: move the hinges, alternating in sign, so the solve can make progress
        stuck = proceed & (stalled_error - accepted_error < stall_improvement) & (nudges < max_nudges)
        nudges[stuck] += 1
        stalled_error[stuck] = np.inf
        accepted_error[stuck] = np.inf
        q[stuck] = np.clip(accepted[stuck] + nudge, LOWER, UPPER)

        step_now = proceed & ~stuck
        stalled_error[step_now] = accepted_error[step_now]
        if not step_now.any():
            continue
        rows = np.flatnonzero(step_now)
        point, axis, origins, axes, _ = forward(q[rows], base[rows], tool_point, tool_axis)
        error = np.concatenate([target_position[rows] - point, np.cross(axis, direction[rows])], axis=1)
        jacobian = np.zeros((rows.size, 6, 6))
        for j in range(6):
            if not free[j]:
                continue
            if IS_HINGE[j]:
                jacobian[:, :3, j] = np.cross(axes[:, j], point - origins[:, j])
                jacobian[:, 3:, j] = axes[:, j]
            else:
                jacobian[:, :3, j] = axes[:, j]
        # Damped least squares: dq = J^T (J J^T + damping^2 I)^-1 error
        normal = jacobian @ np.transpose(jacobian, (0, 2, 1)) + damping * damping * np.eye(6)
        weights = np.linalg.solve(normal, error[:, :, None])[:, :, 0]
        step = np.einsum("nrj,nr->nj", jacobian, weights)
        largest = np.abs(step).max(axis=1)
        step *= np.where(largest > max_step, max_step / np.maximum(largest, 1e-300), 1.0)[:, None]
        steps[rows] = step
        q[rows] = np.clip(accepted[rows] + step, LOWER, UPPER)

    return converged, accepted
