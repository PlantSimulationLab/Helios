"""Apple harvesting environment for reinforcement learning, as a Stable-Baselines3 vectorized environment.

Run from the build directory of this sample, after building it, so that the library and the robot model are found.
"""

import ctypes
import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

RUNNING, SUCCESS, TIMEOUT, SLIPPED, KNOCKED_OFF, COLLISION = 0, 1, 2, 3, 4, 5
OUTCOME_NAMES = {SUCCESS: "success", TIMEOUT: "timeout", SLIPPED: "slipped", KNOCKED_OFF: "knocked_off", COLLISION: "collision"}

# macOS, Linux, and Windows, where CMake puts the DLL beside the executable
_library_names = ["lib/libapple_harvest_env.dylib", "lib/libapple_harvest_env.so", "apple_harvest_env.dll"]
# HARVEST_ENV_LIBRARY names another build of the library, e.g. a copy kept to evaluate policies trained on an earlier version of the task.
# It is still run from the build directory, which holds the assets.
_library_name = os.environ.get("HARVEST_ENV_LIBRARY")
if _library_name is not None:
    if not os.path.exists(_library_name):
        raise FileNotFoundError(f"HARVEST_ENV_LIBRARY is {_library_name}, which does not exist.")
else:
    _library_name = next((name for name in _library_names if os.path.exists(name)), None)
if _library_name is None:
    raise FileNotFoundError(f"None of {_library_names} was found. Run from the sample's build directory after building it.")
_lib = ctypes.CDLL(os.path.abspath(_library_name))
_lib.harvest_env_create.restype = ctypes.c_void_p
_lib.harvest_env_create.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.harvest_env_destroy.argtypes = [ctypes.c_void_p]
_lib.harvest_env_count.argtypes = [ctypes.c_void_p]
_float_pointer = ctypes.POINTER(ctypes.c_float)
_int_pointer = ctypes.POINTER(ctypes.c_int)
_lib.harvest_env_reset.argtypes = [ctypes.c_void_p, _float_pointer]
_lib.harvest_env_enable_camera.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.harvest_env_set_near_start_fraction.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.harvest_env_render.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
_lib.harvest_env_save_camera_image.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
_lib.harvest_env_step.argtypes = [ctypes.c_void_p, _float_pointer, _float_pointer, _float_pointer, _int_pointer, _float_pointer]


class EpisodeRecord(ctypes.Structure):
    """What happened in a simulation's last finished episode. See HarvestEpisodeRecord in harvest_env.h."""

    _fields_ = [(name, ctypes.c_float) for name in ("detach_force", "final_stem_load", "final_grip_force", "pull_x", "pull_y", "pull_z", "peak_contact_force", "peak_wood_motion", "peak_tool_speed")] + [("steps_to_grip", ctypes.c_int), ("steps", ctypes.c_int)]


_lib.harvest_env_last_episode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(EpisodeRecord)]


class CameraView(ctypes.Structure):
    """What a simulation's camera sees, as fractions of the image. See HarvestCameraView in harvest_env.h."""

    _fields_ = [(name, ctypes.c_float) for name in ("target_fruit", "other_fruit", "leaves", "near_leaves", "wood")]


_lib.harvest_env_measure_camera_view.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(CameraView)]


class AppleHarvestVecEnv(VecEnv):
    """Each tree seed gives a tree with a different structure, with instances_per_tree simulations of it."""

    def __init__(self, tree_seeds, instances_per_tree=2, threads=4, seed=0, leaves=False):
        seeds = (ctypes.c_int * len(tree_seeds))(*tree_seeds)
        self._env = _lib.harvest_env_create(seeds, len(tree_seeds), instances_per_tree, threads, seed, 1 if leaves else 0)
        if not self._env:
            raise RuntimeError("The apple harvesting environment could not be created. Is this running from the sample's build directory?")
        self.max_steps = _lib.harvest_env_max_steps()
        observation_size = _lib.harvest_env_observation_size()
        action_size = _lib.harvest_env_action_size()
        count = _lib.harvest_env_count(self._env)
        observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (action_size,), np.float32)
        super().__init__(count, observation_space, action_space)
        self._observations = np.zeros((count, observation_size), np.float32)
        self._terminal = np.zeros((count, observation_size), np.float32)
        self._rewards = np.zeros(count, np.float32)
        self._outcomes = np.zeros(count, np.int32)
        self._actions = None
        self._episode_rewards = np.zeros(count)
        self._episode_lengths = np.zeros(count, np.int64)

    def set_near_start_fraction(self, fraction):
        """Start this share of episodes with the cup near its fruit (see harvest_env_set_near_start_fraction() in harvest_env.h); for training only."""
        if not _lib.harvest_env_set_near_start_fraction(self._env, fraction):
            raise ValueError(f"near start fraction {fraction} was refused; the reason is printed above")

    def reset(self):
        _lib.harvest_env_reset(self._env, self._observations.ctypes.data_as(_float_pointer))
        self._episode_rewards[:] = 0
        self._episode_lengths[:] = 0
        return self._observations.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, np.float32)

    def step_wait(self):
        _lib.harvest_env_step(
            self._env,
            self._actions.ctypes.data_as(_float_pointer),
            self._observations.ctypes.data_as(_float_pointer),
            self._rewards.ctypes.data_as(_float_pointer),
            self._outcomes.ctypes.data_as(_int_pointer),
            self._terminal.ctypes.data_as(_float_pointer),
        )
        self._episode_rewards += self._rewards
        self._episode_lengths += 1
        dones = self._outcomes != RUNNING
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            outcome = int(self._outcomes[i])
            infos[i] = {
                "terminal_observation": self._terminal[i].copy(),
                "outcome": outcome,
                "is_success": outcome == SUCCESS,
                # A timeout is not a failure of the state: the value of what follows should still be estimated
                "TimeLimit.truncated": outcome == TIMEOUT,
                "episode": {"r": float(self._episode_rewards[i]), "l": int(self._episode_lengths[i]), "t": 0.0},
            }
            self._episode_rewards[i] = 0
            self._episode_lengths[i] = 0
        return self._observations.copy(), self._rewards.copy(), dones, infos

    def enable_camera(self, size):
        """Add a square camera on the wrist of the robot in every simulation."""
        if not _lib.harvest_env_enable_camera(self._env, size, size):
            raise RuntimeError("The camera could not be created.")
        self._images = np.zeros((self.num_envs, size, size, 3), np.uint8)

    def render_images(self):
        """The current image of every simulation, as an array of shape (simulations, height, width, 3). The first row is the top of the image."""
        _lib.harvest_env_render(self._env, self._images.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        return self._images

    def save_camera_image(self, index, filename):
        _lib.harvest_env_save_camera_image(self._env, index, filename.encode())

    def measure_camera_view(self, index):
        """What the camera of one simulation sees now. Slow: for understanding a policy, not for training."""
        view = CameraView()
        _lib.harvest_env_measure_camera_view(self._env, index, ctypes.byref(view))
        return view

    def last_episode(self, index):
        record = EpisodeRecord()
        _lib.harvest_env_last_episode(self._env, index, ctypes.byref(record))
        return record

    def close(self):
        if self._env:
            _lib.harvest_env_destroy(self._env)
            self._env = None

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("The simulations are not separate objects, so methods cannot be called on them one at a time.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


def usable_tree_seeds(count, first_seed=1):
    """The first seeds from first_seed that give a tree with fruit within reach of the robot, since not all do."""
    seeds = []
    skipped = []
    seed = first_seed
    while len(seeds) < count:
        if _lib.harvest_env_tree_has_reachable_fruit(seed):
            seeds.append(seed)
        else:
            skipped.append(seed)
        seed += 1
    if skipped:
        print(f"Skipped tree seeds with no fruit within reach: {skipped}", flush=True)
    return seeds


def evaluate(env, choose_actions, episodes_per_simulation, records=None):
    """Success rates over exactly the first episodes of each simulation. Counting all episodes within a number of steps would favor outcomes that end episodes sooner.

    If records is a list, the outcome and record of each counted episode are appended to it.
    """
    observations = env.reset()
    finished = np.zeros(env.num_envs, np.int64)
    outcomes = {name: 0 for name in OUTCOME_NAMES.values()}
    while finished.min() < episodes_per_simulation:
        observations, _, dones, infos = env.step(choose_actions(observations))
        for i in np.flatnonzero(dones):
            if finished[i] < episodes_per_simulation:
                finished[i] += 1
                outcomes[OUTCOME_NAMES[infos[i]["outcome"]]] += 1
                if records is not None:
                    records.append((OUTCOME_NAMES[infos[i]["outcome"]], env.last_episode(int(i))))
    total = int(finished.sum())
    return {name: count / total for name, count in outcomes.items()}, total


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
    # Short of the fruit until lined up, then in to where the lip meets it, which stops the cup
    wanted_along = np.where(lined_up, 0.0, np.maximum(along, 0.08))
    actions[:, :2] = 8.0 * (sideways + (along - wanted_along)[:, None] * line)
    actions[:, 2] = 8.0 * toward_fruit[:, 2]
    gripped = observations[:, 16] > 0.5
    actions[gripped, :3] = -toward_fruit[gripped] / np.maximum(np.linalg.norm(toward_fruit[gripped], axis=1, keepdims=True), 1e-6)
    actions[:, 3] = 1.0
    return actions
