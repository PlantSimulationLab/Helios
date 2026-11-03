"""The apple harvesting environment of harvest_env.cpp, simulated in IsaacSim with PhysX instead of Helios with MuJoCo.

It has the interface of harvest_env.AppleHarvestVecEnv, so that the same policies and Stable-Baselines3 code run in both. The trees
are exported from Helios by harvest_env_export_tree() and prepared by tree_preparation.py. The task follows harvest_env.cpp step for
step: the same observations, actions, inverse kinematics, grip, detachment, protective stop, reward and episode records. What differs:

- Physics: PhysX (TGS, GPU) instead of MuJoCo. Rigid wood is merged into the bodies that carry it, which is exact for rigid bodies.
- Candidate fruit are held either by rigid joints (fruit_hold="rigid", the default), with the stem load the force in that joint, or
  (fruit_hold="soft") by six-axis springs with the time constant and damping of MuJoCo's soft weld (see
  tree_preparation.soft_hold_gains()). There, the stem load is the force the springs put on the fruit in a step, from the change in
  the fruit's momentum less gravity, the grip and contacts.
- All simulations take all 20 physics steps of a control step; one whose episode ended keeps stepping until the control step is over
  and is then reset, where Helios stops it at once. What happens in those steps is not used.
- Random draws (fruit, detach force, start offset) use numpy, so they are not the same numbers as Helios draws.

Only one environment can exist per process, since it owns the USD stage. Start the SimulationApp with start_app() before creating it.
"""

import json
import math
import os
import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

RUNNING, SUCCESS, TIMEOUT, SLIPPED, KNOCKED_OFF, COLLISION = 0, 1, 2, 3, 4, 5
OUTCOME_NAMES = {SUCCESS: "success", TIMEOUT: "timeout", SLIPPED: "slipped", KNOCKED_OFF: "knocked_off", COLLISION: "collision"}
OBSERVATION_SIZE = 23
ACTION_SIZE = 4
ENV_SPACING = 6.0  # m between simulations, so that they cannot touch

_app = None


def start_app():
    """Start IsaacSim headless with PhysX. Must be called before creating an environment."""
    global _app
    if _app is None:
        from isaacsim import SimulationApp

        _app = SimulationApp({"headless": True, "extra_args": ["--/exts/isaacsim.physics.newton/auto_switch_on_startup=false", "--/exts/isaacsim.core.simulation_manager/default_engine=physx"]})
    return _app


class EpisodeRecord:
    """What happened in a simulation's last finished episode, with the fields of HarvestEpisodeRecord in harvest_env.h."""

    def __init__(self):
        self.detach_force = 0.0
        self.final_stem_load = 0.0
        self.final_grip_force = 0.0
        self.pull_x = self.pull_y = self.pull_z = 0.0
        self.peak_contact_force = 0.0
        self.peak_wood_motion = 0.0
        self.peak_tool_speed = 0.0
        self.steps_to_grip = -1
        self.steps = 0


def _quaternion_rotate(q, v):
    """Rotate vectors v (..., 3) by quaternions q (..., 4) given as (x, y, z, w), as the tensor API gives them."""
    import torch

    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quaternion_multiply(a, b):
    """The rotation b followed by a, for quaternions (..., 4) given as (x, y, z, w)."""
    import torch

    a_xyz, a_w = a[..., :3], a[..., 3:4]
    b_xyz, b_w = b[..., :3], b[..., 3:4]
    return torch.cat([a_w * b_xyz + b_w * a_xyz + torch.cross(a_xyz, b_xyz, dim=-1), a_w * b_w - (a_xyz * b_xyz).sum(-1, keepdim=True)], dim=-1)


def _quaternion_conjugate(q):
    import torch

    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


class IsaacHarvestVecEnv(VecEnv):
    """instances_per_tree simulations of each tree seed. threads is accepted for compatibility with AppleHarvestVecEnv and not used."""

    def __init__(self, tree_seeds, instances_per_tree=2, threads=None, seed=0, leaves=False, trees_directory=None, device="cuda:0", settling_steps=None, fruit_hold="rigid"):
        if leaves:
            raise NotImplementedError("Trees with leaves are not exported for IsaacSim yet.")
        if fruit_hold not in ("rigid", "soft"):
            raise ValueError(f'fruit_hold must be "rigid" or "soft", not "{fruit_hold}"')
        self.fruit_hold = fruit_hold
        self.soft = fruit_hold == "soft"
        if _app is None:
            raise RuntimeError("Call isaac_harvest_env.start_app() before creating the environment.")
        import torch

        self.torch = torch
        self.device = device
        self.trees_directory = trees_directory or os.path.join(os.getcwd(), "isaac_trees")
        self.tree_seeds = list(tree_seeds)
        self.settling_steps = settling_steps  # None settles as harvest_env.cpp does; fewer only for quick checks
        self.instances_per_tree = instances_per_tree
        self.max_steps = None
        count = len(self.tree_seeds) * instances_per_tree
        super().__init__(count, gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32), gym.spaces.Box(-1.0, 1.0, (ACTION_SIZE,), np.float32))

        self._load_trees()
        self._build_scene()
        self._start_physics()
        self._settle()

        self.random = [np.random.default_rng(seed + 1000 * t + i) for t in range(len(self.tree_seeds)) for i in range(instances_per_tree)]
        self._observations = np.zeros((count, OBSERVATION_SIZE), np.float32)
        self._episode_rewards = np.zeros(count)
        self._episode_lengths = np.zeros(count, np.int64)
        self._actions = None
        self.last = [EpisodeRecord() for _ in range(count)]

    # ----------------------------------------------------------------------------------------------------------- building

    def _load_trees(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tree_preparation import prepare_tree

        self.trees = []
        for seed in self.tree_seeds:
            base = os.path.join(self.trees_directory, f"tree_{seed}")
            if not os.path.exists(base + ".json"):
                raise FileNotFoundError(f"{base}.json not found. Export the tree with harvest_env_export_tree({seed}, ...) from Helios first.")
            description = json.load(open(base + ".json"))
            suffix = "_isaac" if not self.soft else "_isaac_soft"
            prepared = base + suffix + ".usda"
            sources = [base + ".usda", base + ".json", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_preparation.py")]
            outputs = [prepared, base + suffix + ".json"]
            if not all(os.path.exists(o) for o in outputs) or min(os.path.getmtime(o) for o in outputs) < max(os.path.getmtime(f) for f in sources):
                prepare_tree(base + ".json", base + ".usda", prepared, fruit_hold=self.fruit_hold)
            sidecar = json.load(open(base + suffix + ".json"))
            if sidecar.get("fruit_hold") != self.fruit_hold:
                raise RuntimeError(f"{base + suffix}.json was prepared with fruit hold {sidecar.get('fruit_hold')}, not {self.fruit_hold}; delete it to prepare the tree again")
            owners = sidecar["owners"]
            radius = {link["name"]: link["sphere_radius"] for link in description["links"] if "sphere_radius" in link}
            wood = np.array(description["wood"], dtype=np.float64)
            self.trees.append({"seed": seed, "description": description, "prepared": prepared, "owners": owners, "radius": radius, "wood": wood, "soft_holds": sidecar["soft_holds"]})
        self.task = self.trees[0]["description"]["task"]
        self.max_steps = self.task["max_control_steps"]

    def _build_scene(self):
        from isaacsim.core.experimental.utils import stage as stage_utils
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        from harvest_scene import ROBOT_JOINTS, build_robot

        stage_utils.create_new_stage()
        stage = stage_utils.get_current_stage()
        self.stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateSolverTypeAttr().Set("TGS")
        # Merged welded wood makes bodies with hundreds of shapes, more than PhysX's default GPU buffers hold
        physx_scene.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(2**22)
        physx_scene.CreateGpuTotalAggregatePairsCapacityAttr().Set(2**22)
        physx_scene.CreateGpuFoundLostPairsCapacityAttr().Set(2**22)
        physx_scene.CreateGpuMaxRigidContactCountAttr().Set(2**21)
        physx_scene.CreateGpuMaxRigidPatchCountAttr().Set(2**19)

        columns = max(1, int(math.ceil(math.sqrt(self.num_envs))))
        self.env_origins = np.zeros((self.num_envs, 3))
        self.env_tree = np.zeros(self.num_envs, dtype=int)
        self.env_paths = []
        e = 0
        for t, tree in enumerate(self.trees):
            for k in range(self.instances_per_tree):
                origin = np.array([ENV_SPACING * (e % columns), ENV_SPACING * (e // columns), 0.0])
                path = f"/World/envs/t{tree['seed']}_{k}"
                env = UsdGeom.Xform.Define(stage, path)
                env.AddTranslateOp().Set(Gf.Vec3d(*origin))
                stage.DefinePrim(f"{path}/Tree").GetReferences().AddReference(tree["prepared"])
                build_robot(stage, f"{path}/Robot", tuple(self.task["robot_base"]), tuple(origin + np.array(self.task["robot_base"])), self.task["servos"])
                for name, position in zip(ROBOT_JOINTS, self.task["tucked_pose"]):
                    joint = stage.GetPrimAtPath(f"{path}/Robot/{name}")
                    kind = "linear" if name == "drive" else "angular"
                    PhysxSchema.JointStateAPI.Apply(joint, kind).CreatePositionAttr().Set(position)
                    UsdPhysics.DriveAPI(joint, kind).GetTargetPositionAttr().Set(position)
                self.env_origins[e] = origin
                self.env_tree[e] = t
                self.env_paths.append(path)
                e += 1

    def _start_physics(self):
        import omni.physics.tensors
        import omni.timeline
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.core.simulation_manager import SimulationManager

        from harvest_scene import ROBOT_JOINTS

        torch = self.torch
        self.SimulationManager = SimulationManager
        if SimulationManager.get_active_physics_engine() != "physx":
            SimulationManager.switch_physics_engine("physx")
        SimulationManager.setup_simulation(dt=self.task["physics_timestep"], device=self.device)
        # Nothing here renders, and writing every body's pose to USD and Fabric after each step took most of the time: with 64 trees on
        # the GPU, 224 ms per step with it and 27 ms without. The tensor API reads the state from PhysX directly.
        import carb

        settings = carb.settings.get_settings()
        for setting in ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd", "/physics/updateParticlesToUsd", "/physics/updateForceSensorsToUsd", "/physics/fabricUpdateTransformations"):
            settings.set_bool(setting, False)
        SimulationManager.enable_fabric(False)
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        self.timeline.commit()
        SimulationManager.initialize_physics()
        self.view = omni.physics.tensors.create_simulation_view("torch", stage_id=stage_utils.get_stage_id(self.stage))
        self.view.set_subspace_roots("/")

        def env_index(paths):
            """Rows of a view, as indices into the environment's simulations."""
            by_path = {p: i for i, p in enumerate(self.env_paths)}
            return torch.tensor([by_path["/".join(p.split("/")[:4])] for p in paths], device=self.device)

        self.robots = self.view.create_articulation_view("/World/envs/*/Robot")
        self.robot_rows = env_index(self.robots.prim_paths)
        self.robot_order = torch.argsort(self.robot_rows)  # robot view row of each simulation
        names = list(self.robots.shared_metatype.link_names)
        self.tool_link = names.index("tool")
        dof_names = list(self.robots.shared_metatype.dof_names)
        self.dof_order = [dof_names.index(n) for n in ROBOT_JOINTS]
        if self.robots.count != self.num_envs:
            raise RuntimeError(f"expected {self.num_envs} robots, found {self.robots.count}")

        # Net contact force on every robot link, which is the force between the robot and the tree (it touches nothing else)
        link_names = [n for n in names]
        self.contacts = self.view.create_rigid_contact_view([f"/World/envs/*/Robot/{n}" for n in link_names])
        contact_paths = self.contacts.sensor_paths
        flat = [p for group in contact_paths for p in (group if isinstance(group, (list, tuple)) else [group])]
        self.contact_env = env_index(flat)

        # One view per tree seed: the instances of a tree are identical, trees are not
        self.tree_views = []
        for t, tree in enumerate(self.trees):
            view = self.view.create_articulation_view(f"/World/envs/t{tree['seed']}_*/Tree")
            rows = env_index(view.prim_paths)
            link_names = list(view.shared_metatype.link_names)
            candidates = tree["description"]["candidates"]
            radii = torch.tensor([tree["radius"][c["link"]] for c in candidates], device=self.device, dtype=torch.float32)
            wood_owner = []
            wood_offset = []
            for c in candidates:
                owner, offset = tree["owners"][c["bearing_wood"]]
                wood_owner.append(link_names.index(owner))
                wood_offset.append(offset)
            tv = {
                "view": view,
                "rows": rows,
                "links": len(link_names),
                "radii": radii,
                "wood_owner": torch.tensor(wood_owner, device=self.device),
                "wood_offset": torch.tensor(wood_offset, device=self.device, dtype=torch.float32),
            }
            if view.count != self.instances_per_tree:
                raise RuntimeError(f"expected {self.instances_per_tree} instances of tree {tree['seed']}, found {view.count}")
            if not self.soft:
                tv["candidate_links"] = torch.tensor([link_names.index(c["link"]) for c in candidates], device=self.device)
            else:
                # Fruit are bodies of their own, held to the link that carries them by springs. fruit_rows[i, c] is the row in the fruit
                # view of candidate c of the tree view's instance i.
                instance_paths = ["/".join(p.split("/")[:4]) for p in view.prim_paths]
                paths = [f"{p}/Tree/Links/{c['link']}" for p in instance_paths for c in candidates]
                fruits = self.view.create_rigid_body_view(paths)
                if fruits.count != len(paths):
                    raise RuntimeError(f"expected {len(paths)} fruit bodies for tree {tree['seed']}, found {fruits.count}")
                by_path = {path: k for k, path in enumerate(fruits.prim_paths)}
                fruit_rows = [[by_path[f"{p}/Tree/Links/{c['link']}"] for c in candidates] for p in instance_paths]
                # The net contact force on each fruit, in the fruit view's order
                fruit_contacts = self.view.create_rigid_contact_view(list(fruits.prim_paths))
                sensors = [p for group in fruit_contacts.sensor_paths for p in (group if isinstance(group, (list, tuple)) else [group])]
                if sensors != list(fruits.prim_paths):
                    raise RuntimeError(f"the contact sensors of the fruit of tree {tree['seed']} are not in the order of the fruit bodies")
                holds = [tree["soft_holds"][c["link"]] for c in candidates]
                masses = torch.tensor([h["mass"] for h in holds], device=self.device, dtype=torch.float32)
                fruit_mass = torch.zeros(fruits.count, device=self.device)
                fruit_mass[torch.tensor(fruit_rows, device=self.device).flatten()] = masses.repeat(view.count)
                tv.update(
                    fruits=fruits,
                    fruit_contacts=fruit_contacts,
                    fruit_rows=torch.tensor(fruit_rows, device=self.device),
                    fruit_mass=fruit_mass,
                    carrier_links=torch.tensor([link_names.index(h["carrier"]) for h in holds], device=self.device),
                    all_fruit=torch.arange(fruits.count, dtype=torch.int32, device=self.device),
                    grip_on_fruit=torch.zeros(fruits.count, 3, device=self.device),
                )
            self.tree_views.append(tv)

    def _settle(self):
        """Let every tree settle with the arm tucked, once, and start every episode from the pose reached (as harvest_env.cpp does)."""
        torch = self.torch
        start = time.perf_counter()
        self.SimulationManager.step(steps=self.settling_steps if self.settling_steps is not None else self.task["settling_physics_steps"])
        for tv in self.tree_views:
            view = tv["view"]
            tv["settled_dofs"] = view.get_dof_positions().clone()
            transforms = view.get_link_transforms().view(view.count, tv["links"], 7)
            tv["settled_fruit"] = self._fruit_poses(tv, transforms)[..., :3].clone()
            tv["settled_wood"] = self._points_on(transforms, tv["wood_owner"], tv["wood_offset"]).clone()
            if self.soft:
                # Each fruit's settled pose relative to the link that carries it
                fruit = self._fruit_poses(tv, transforms)
                carrier = transforms[:, tv["carrier_links"]]
                inverse = _quaternion_conjugate(carrier[..., 3:7])
                tv["fruit_in_carrier"] = torch.cat([_quaternion_rotate(inverse, fruit[..., :3] - carrier[..., :3]), _quaternion_multiply(inverse, fruit[..., 3:7])], dim=-1).clone()
        self.settle_seconds = time.perf_counter() - start
        self.robot_settled = self.robots.get_dof_positions().clone()

    # ----------------------------------------------------------------------------------------------------------- state

    def _points_on(self, transforms, link_indices, offsets):
        """World positions (instances, candidates, 3) of points given in the frames of links (candidates,)."""
        frames = transforms[:, link_indices]
        return frames[..., :3] + _quaternion_rotate(frames[..., 3:7], offsets.unsqueeze(0).expand(frames.shape[0], -1, -1))

    def _fruit_poses(self, tv, transforms):
        """Pose (instances, candidates, 7) of every candidate fruit of a tree view, given its links' transforms."""
        if not self.soft:
            return transforms[:, tv["candidate_links"]]
        return tv["fruits"].get_transforms().view(-1, 7)[tv["fruit_rows"]]

    def _fruit_velocities(self, tv, velocities):
        """Velocity (instances, candidates, 6) of every candidate fruit of a tree view, given its links' velocities."""
        if not self.soft:
            return velocities[:, tv["candidate_links"]]
        return tv["fruits"].get_velocities().view(-1, 6)[tv["fruit_rows"]]

    def _gather_state(self):
        """Per simulation: tool point, tool axis, tool center-of-mass velocity, robot joints; fruit position and velocity, wood."""
        torch = self.torch
        n = self.num_envs
        robot_transforms = self.robots.get_link_transforms().view(self.robots.count, -1, 7)[self.robot_order]
        robot_velocities = self.robots.get_link_velocities().view(self.robots.count, -1, 6)[self.robot_order]
        tool_frame = robot_transforms[:, self.tool_link]
        cup = torch.tensor(self.task["cup_face"], device=self.device, dtype=torch.float32).expand(n, 3)
        state = {
            "tool": tool_frame[:, :3] + _quaternion_rotate(tool_frame[:, 3:7], cup),
            "tool_axis": _quaternion_rotate(tool_frame[:, 3:7], torch.tensor([1.0, 0, 0], device=self.device).expand(n, 3)),
            "tool_frame": tool_frame,
            "tool_velocity": robot_velocities[:, self.tool_link, :3],
            "joints": self.robots.get_dof_positions().view(self.robots.count, -1)[self.robot_order][:, self.dof_order],
            "fruit": torch.zeros(n, 3, device=self.device),
            "fruit_velocity": torch.zeros(n, 3, device=self.device),
            "wood": torch.zeros(n, 3, device=self.device),
        }
        for tv in self.tree_views:
            view = tv["view"]
            transforms = view.get_link_transforms().view(view.count, tv["links"], 7)
            velocities = view.get_link_velocities().view(view.count, tv["links"], 6)
            rows = tv["rows"]
            target = self.target[rows]
            arange = torch.arange(view.count, device=self.device)
            state["fruit"][rows] = self._fruit_poses(tv, transforms)[arange, target, :3]
            state["fruit_velocity"][rows] = self._fruit_velocities(tv, velocities)[arange, target, :3]
            wood = self._points_on(transforms, tv["wood_owner"], tv["wood_offset"])
            state["wood"][rows] = wood[arange, target]
        return state

    def _before_physics_step(self):
        """With soft holds, record every fruit's velocity and give the target fruit the grip's impulse for the coming step.

        The grip acts on a free fruit through its velocity (an impulse of force x time step, at its center of mass), which is what
        applying the force over the step does. Applying forces to the fruit through the rigid body view, followed later by setting
        their state in a reset, made PhysX's GPU kernels fail with an illegal memory access in IsaacSim 6.1."""
        if not self.soft:
            return
        dt = self.task["physics_timestep"]
        for tv in self.tree_views:
            fruits = tv["fruits"]
            velocities = fruits.get_velocities().view(-1, 6).clone()
            tv["velocity_before"] = velocities.clone()
            velocities[:, :3] += tv["grip_on_fruit"] * (dt / tv["fruit_mass"]).unsqueeze(1)
            fruits.set_velocities(velocities, tv["all_fruit"])

    def physics_step(self):
        """Take one physics step, with the grip as set by the last _apply_grip()."""
        self._before_physics_step()
        self.SimulationManager.step(steps=1)

    def candidate_stem_loads(self, tv):
        """The force on the stem of every candidate fruit of a tree view (instances, candidates) in the step just taken by physics_step().

        Rigid holds: the force in the fruit's joint. Soft holds: the force the springs put on the fruit over the step, from its change
        of momentum less gravity, the grip and the contact force on it. This is what PhysX applied, whatever the state of the springs
        it was computed from."""
        view = tv["view"]
        if not self.soft:
            forces = view.get_link_incoming_joint_force().view(view.count, tv["links"], 6)
            return self.torch.linalg.norm(forces[:, tv["candidate_links"], :3], dim=2)
        dt = self.task["physics_timestep"]
        velocities = tv["fruits"].get_velocities().view(-1, 6)
        contact = tv["fruit_contacts"].get_net_contact_forces(dt).view(-1, 3)
        gravity = self.torch.tensor([0.0, 0.0, -9.81], device=self.device)
        mass = tv["fruit_mass"].unsqueeze(1)
        stem = mass * (velocities[:, :3] - tv["velocity_before"][:, :3]) / dt - mass * gravity - tv["grip_on_fruit"] - contact
        return self.torch.linalg.norm(stem, dim=1)[tv["fruit_rows"]]

    def _stem_load(self):
        """The force on the stem of each simulation's target fruit in the physics step just taken. Call once after every physics step."""
        torch = self.torch
        load = torch.zeros(self.num_envs, device=self.device)
        for tv in self.tree_views:
            rows = tv["rows"]
            arange = torch.arange(tv["view"].count, device=self.device)
            load[rows] = self.candidate_stem_loads(tv)[arange, self.target[rows]]
        return load

    def _contact_force(self):
        torch = self.torch
        forces = self.contacts.get_net_contact_forces(self.task["physics_timestep"]).view(-1, 3)
        total = torch.zeros(self.num_envs, 3, device=self.device)
        total.index_add_(0, self.contact_env, forces)
        return torch.linalg.norm(total, dim=1)

    def _apply_grip(self, state, dt):
        """The vacuum cup: a spring that can only pull, on the fruit and, in reaction, on the arm at the cup's face."""
        torch = self.torch
        task = self.task
        self.grip_seconds = torch.where(self.gripped, self.grip_seconds + dt, self.grip_seconds)
        ramp = torch.clamp(self.grip_seconds / task["grip_ramp_seconds"], max=1.0)
        toward_tool = state["tool"] - state["fruit"]
        distance = torch.linalg.norm(toward_tool, dim=1)
        toward_tool = toward_tool / torch.clamp(distance, min=1e-6).unsqueeze(1)
        self.pull_direction = torch.where(self.gripped.unsqueeze(1), toward_tool, self.pull_direction)
        stretch = distance - self.grip_rest_length
        separating = ((state["tool_velocity"] - state["fruit_velocity"]) * toward_tool).sum(dim=1)
        force = torch.clamp(ramp * (task["grip_stiffness"] * stretch + task["grip_damping"] * separating), min=0.0)
        force = torch.where(self.gripped & (stretch > 0) & (distance > 1e-6), force, torch.zeros_like(force))
        self.grip_force = force
        on_fruit = force.unsqueeze(1) * toward_tool

        robot_forces = torch.zeros(self.robots.count, self.robots.max_links, 3, device=self.device)
        robot_positions = torch.zeros_like(robot_forces)
        robot_forces[self.robot_order, self.tool_link] = -on_fruit
        robot_positions[self.robot_order, self.tool_link] = state["tool"]
        self.robots.apply_forces_and_torques_at_position(robot_forces, None, robot_positions, self.all_robots, True)
        for tv in self.tree_views:
            view = tv["view"]
            rows = tv["rows"]
            arange = torch.arange(view.count, device=self.device)
            if self.soft:
                # Applied by _before_physics_step()
                grip = torch.zeros_like(tv["grip_on_fruit"])
                grip[tv["fruit_rows"][arange, self.target[rows]]] = on_fruit[rows]
                tv["grip_on_fruit"] = grip
                continue
            forces = torch.zeros(view.count, tv["links"], 3, device=self.device)
            positions = torch.zeros_like(forces)
            links = tv["candidate_links"][self.target[rows]]
            forces[arange, links] = on_fruit[rows]
            positions[arange, links] = state["fruit"][rows]
            view.apply_forces_and_torques_at_position(forces, None, positions, tv["all"], True)

    # ----------------------------------------------------------------------------------------------------------- episodes

    def _surface_distance(self, state):
        return self.torch.linalg.norm(state["fruit"] - state["tool"], dim=1) - self.target_radius

    def _nearest_wood(self, tools):
        """From each tool point to the nearest surface of its tree's wood at rest, as harvest_env.cpp's towardNearestWood()."""
        result = np.zeros((self.num_envs, 3))
        for e in range(self.num_envs):
            wood = self.trees[self.env_tree[e]]["wood"]
            start, end, radius = wood[:, 0:3], wood[:, 3:6], wood[:, 6]
            point = tools[e] - self.env_origins[e]
            along = end - start
            length_squared = (along * along).sum(1)
            t = np.where(length_squared > 0, np.clip(((point - start) * along).sum(1) / np.maximum(length_squared, 1e-30), 0, 1), 0)
            offset = start + t[:, None] * along - point
            distance = np.linalg.norm(offset, axis=1) - radius
            result[e] = offset[np.argmin(distance)]
        return result

    def _observe(self, state, contact):
        torch = self.torch
        tool = state["tool"].cpu().numpy().astype(np.float64)
        fruit = state["fruit"].cpu().numpy().astype(np.float64)
        joints = state["joints"].cpu().numpy().astype(np.float64)
        velocity = state["tool_velocity"].cpu().numpy()
        distance = self._surface_distance(state).cpu().numpy()
        base = self.env_origins + np.array(self.task["robot_base"])
        wood = self._nearest_wood(tool)
        o = np.zeros((self.num_envs, OBSERVATION_SIZE), np.float32)
        o[:, 0:3] = fruit - tool
        o[:, 3] = distance
        o[:, 4] = tool[:, 0] - base[:, 0]
        o[:, 5] = (tool[:, 1] - self.env_origins[:, 1]) - joints[:, 0]
        o[:, 6] = tool[:, 2]
        o[:, 7:10] = velocity
        o[:, 10:16] = joints
        o[:, 16] = self.gripped.cpu().numpy()
        o[:, 17] = self.grip_force.cpu().numpy() / self.task["grip_limit"]
        o[:, 18] = contact.cpu().numpy() / 10.0
        o[:, 19] = self.control_step / self.task["max_control_steps"]
        o[:, 20:23] = wood
        return o

    def _reset(self, envs):
        """Start new episodes in the simulations listed: the settled tree, a random fruit and detach force, the arm tucked at a random place along the row."""
        torch = self.torch
        task = self.task
        if len(envs) == 0:
            return
        envs = np.asarray(envs)
        envs_t = torch.tensor(envs, device=self.device)
        for e in envs:
            tree = self.trees[self.env_tree[e]]
            random = self.random[e]
            self.target_np[e] = random.integers(0, len(tree["description"]["candidates"]))
            self.detach_force_np[e] = task["min_detach_force"] + random.random() * (task["max_detach_force"] - task["min_detach_force"])
            self.start_y[e] = random.random()
        self.target[envs_t] = torch.tensor(self.target_np[envs], device=self.device)
        self.detach_force[envs_t] = torch.tensor(self.detach_force_np[envs], device=self.device, dtype=torch.float32)

        # Trees back to their settled pose, at rest
        for t, tv in enumerate(self.tree_views):
            view = tv["view"]
            rows = tv["rows"]
            members = torch.isin(rows, envs_t)
            if not members.any():
                continue
            indices = torch.nonzero(members).flatten().to(torch.int32)
            view.set_dof_positions(tv["settled_dofs"], indices)
            view.set_dof_velocities(torch.zeros_like(tv["settled_dofs"]), indices)
            arange = torch.arange(view.count, device=self.device)
            settled_fruit = tv["settled_fruit"][arange, self.target[rows]]
            self.target_radius[rows[members]] = tv["radii"][self.target[rows[members]]]
            self.settled_fruit[rows[members]] = settled_fruit[members]
            self.settled_wood[rows[members]] = tv["settled_wood"][arange, self.target[rows]][members]

        # The arm tucked, its platform within start_offset of the fruit along the row
        positions = self.robots.get_dof_positions().view(self.robots.count, -1).clone()
        targets = self.robot_targets.clone()
        tucked = np.radians(np.array(task["tucked_pose"], dtype=np.float64))
        for e in envs:
            pose = tucked.copy()
            fruit_y = float(self.settled_fruit[e, 1]) - self.env_origins[e, 1]
            pose[0] = fruit_y + (2.0 * self.start_y[e] - 1.0) * task["start_offset"]
            row = int(self.robot_order[e])
            for j, dof in enumerate(self.dof_order):
                positions[row, dof] = float(pose[j])
                targets[row, dof] = float(pose[j])
        robot_indices = self.robot_order[envs_t].to(torch.int32)
        self.robots.set_dof_positions(positions, robot_indices)
        self.robots.set_dof_velocities(torch.zeros_like(positions), robot_indices)
        self.robots.set_dof_position_targets(targets, robot_indices)
        self.robot_targets = targets
        self.view.update_articulations_kinematic()

        # Free fruit back on their stems, where they settled relative to the link that carries them. Setting a tree's joint positions,
        # even to the values they already had, does not leave every link exactly where it was, and a fruit placed at its settled world
        # pose then started the episode with a stretched spring: up to 25x its weight on the stem.
        if self.soft:
            for tv in self.tree_views:
                members = torch.isin(tv["rows"], envs_t)
                if not members.any():
                    continue
                view = tv["view"]
                fruits = tv["fruits"]
                carrier = view.get_link_transforms().view(view.count, tv["links"], 7)[:, tv["carrier_links"]]
                relative = tv["fruit_in_carrier"]
                placed = torch.cat([carrier[..., :3] + _quaternion_rotate(carrier[..., 3:7], relative[..., :3]), _quaternion_multiply(carrier[..., 3:7], relative[..., 3:7])], dim=-1)
                reset_bodies = tv["fruit_rows"][members].flatten()
                transforms = fruits.get_transforms().view(-1, 7).clone()
                velocities = fruits.get_velocities().view(-1, 6).clone()
                transforms[reset_bodies] = placed[members].reshape(-1, 7)
                velocities[reset_bodies] = 0.0
                fruits.set_transforms(transforms, tv["all_fruit"])
                fruits.set_velocities(velocities, tv["all_fruit"])

        self.gripped[envs_t] = False
        self.grip_seconds[envs_t] = 0.0
        self.grip_force[envs_t] = 0.0
        self.grip_rest_length[envs_t] = 0.0
        self.pull_direction[envs_t] = 0.0
        self.previous_load[envs] = 0.0
        self.control_step[envs] = 0
        self.current_peak_contact[envs] = 0.0
        self.current_peak_wood[envs] = 0.0
        self.current_peak_speed[envs] = 0.0
        self.steps_to_grip[envs] = -1

    def reset(self):
        torch = self.torch
        n = self.num_envs
        self.target = torch.zeros(n, dtype=torch.long, device=self.device)
        self.target_np = np.zeros(n, dtype=int)
        self.detach_force = torch.zeros(n, device=self.device)
        self.detach_force_np = np.zeros(n)
        self.start_y = np.zeros(n)
        self.target_radius = torch.zeros(n, device=self.device)
        self.settled_fruit = torch.zeros(n, 3, device=self.device)
        self.settled_wood = torch.zeros(n, 3, device=self.device)
        self.gripped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.grip_seconds = torch.zeros(n, device=self.device)
        self.grip_force = torch.zeros(n, device=self.device)
        self.grip_rest_length = torch.zeros(n, device=self.device)
        self.pull_direction = torch.zeros(n, 3, device=self.device)
        self.previous_load = np.zeros(n)
        self.previous_distance = np.zeros(n)
        self.control_step = np.zeros(n, dtype=int)
        self.tool_target = np.zeros((n, 3))
        self.current_peak_contact = np.zeros(n)
        self.current_peak_wood = np.zeros(n)
        self.current_peak_speed = np.zeros(n)
        self.steps_to_grip = np.full(n, -1)
        self.all_robots = torch.arange(self.robots.count, dtype=torch.int32, device=self.device)
        for tv in self.tree_views:
            tv["all"] = torch.arange(tv["view"].count, dtype=torch.int32, device=self.device)
        self.robot_targets = self.robots.get_dof_position_targets().view(self.robots.count, -1).clone()
        self._reset(np.arange(n))
        state = self._gather_state()
        self.tool_target = state["tool"].cpu().numpy().astype(np.float64)
        self.previous_distance = self._surface_distance(state).cpu().numpy()
        self._observations = self._observe(state, torch.zeros(n, device=self.device))
        self._episode_rewards[:] = 0
        self._episode_lengths[:] = 0
        return self._observations.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, np.float32)

    def step_wait(self):
        torch = self.torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from arm_kinematics import solve_inverse_kinematics

        task = self.task
        n = self.num_envs
        actions = self._actions
        dt_control = task["physics_steps_per_control"] * task["physics_timestep"]

        # Move the tool target, and solve for the arm with the cup pointing away from the robot's turret
        state = self._gather_state()
        tool = state["tool"].cpu().numpy().astype(np.float64)
        joints = state["joints"].cpu().numpy().astype(np.float64)
        velocity = task["max_tool_speed"] * np.clip(actions[:, :3], -1.0, 1.0)
        self.tool_target = self.tool_target + dt_control * velocity
        lead = self.tool_target - tool
        lead_length = np.linalg.norm(lead, axis=1)
        too_far = lead_length > task["max_target_lead"]
        self.tool_target[too_far] = tool[too_far] + lead[too_far] / lead_length[too_far, None] * task["max_target_lead"]
        base = self.env_origins + np.array(task["robot_base"])
        pointing = self.tool_target - np.stack([base[:, 0] + 0.15, joints[:, 0] + self.env_origins[:, 1], self.tool_target[:, 2]], axis=1)
        pointing[:, 2] = 0.0
        solve = np.linalg.norm(pointing, axis=1) > 0.05
        if solve.any():
            rows = np.flatnonzero(solve)
            direction = pointing[rows] / np.linalg.norm(pointing[rows], axis=1, keepdims=True)
            locked = np.array([False, False, False, False, False, True])  # roll
            converged, solution = solve_inverse_kinematics(joints[rows], base[rows], np.array(task["cup_face"], dtype=float), np.array([1.0, 0.0, 0.0]), self.tool_target[rows], direction, locked)
            # An unreachable target leaves the arm where it was heading
            apply = rows[converged]
            if apply.size:
                targets = self.robot_targets.clone()
                order = self.robot_order.cpu().numpy()
                for k, e in enumerate(apply):
                    for j, dof in enumerate(self.dof_order):
                        targets[order[e], dof] = float(solution[np.flatnonzero(rows == e)[0], j])
                self.robots.set_dof_position_targets(targets, self.robot_order[torch.tensor(apply, device=self.device)].to(torch.int32))
                self.robot_targets = targets

        distance_before_grip = self._surface_distance(state)
        gripped_now = (~self.gripped) & torch.tensor(actions[:, 3] > 0, device=self.device) & (distance_before_grip < task["grip_reach"])
        self.grip_rest_length = torch.where(gripped_now, distance_before_grip + self.target_radius, self.grip_rest_length)
        self.grip_seconds = torch.where(gripped_now, torch.zeros_like(self.grip_seconds), self.grip_seconds)
        self.gripped = self.gripped | gripped_now
        gripped_now_np = gripped_now.cpu().numpy()
        self.steps_to_grip[gripped_now_np] = self.control_step[gripped_now_np]

        # Simulate, with the stem failing when the load on it passes the force for this fruit
        outcome = torch.zeros(n, dtype=torch.long, device=self.device)
        running = torch.ones(n, dtype=torch.bool, device=self.device)
        load = torch.zeros(n, device=self.device)
        peak_contact = torch.zeros(n, device=self.device)
        final_grip = torch.zeros(n, device=self.device)
        physics_dt = task["physics_timestep"]
        for _ in range(task["physics_steps_per_control"]):
            state = self._gather_state()
            self._apply_grip(state, physics_dt)
            grip_force = self.grip_force
            self.physics_step()
            step_load = self._stem_load()
            contact = self._contact_force()
            detached = running & (step_load > self.detach_force)
            slipped = running & ~detached & (grip_force > task["grip_limit"])
            collided = running & ~detached & ~slipped & (contact > task["protective_stop_force"])
            outcome = torch.where(detached & self.gripped, torch.full_like(outcome, SUCCESS), outcome)
            outcome = torch.where(detached & ~self.gripped, torch.full_like(outcome, KNOCKED_OFF), outcome)
            outcome = torch.where(slipped, torch.full_like(outcome, SLIPPED), outcome)
            outcome = torch.where(collided, torch.full_like(outcome, COLLISION), outcome)
            load = torch.where(running, step_load, load)
            final_grip = torch.where(running, grip_force, final_grip)
            peak_contact = torch.where(running, torch.maximum(peak_contact, contact), peak_contact)
            running = running & ~(detached | slipped | collided)

        self.control_step += 1
        outcome_np = outcome.cpu().numpy()
        outcome_np[(outcome_np == RUNNING) & (self.control_step >= task["max_control_steps"])] = TIMEOUT

        # Reward: approach the fruit, grip it, load the stem, and detach it, without pushing on the tree
        state = self._gather_state()
        distance = self._surface_distance(state).cpu().numpy()
        load_np = load.cpu().numpy()
        gripped_np = self.gripped.cpu().numpy()
        peak_np = peak_contact.cpu().numpy()
        reward = np.full(n, -0.005)
        reward += np.where(~gripped_np, 5.0 * (self.previous_distance - distance), 0.0)
        reward += np.where(gripped_now_np, 1.0, 0.0)
        reward += np.where(gripped_np, 3.0 * (load_np - self.previous_load) / task["max_detach_force"], 0.0)
        reward -= 0.02 * np.minimum(peak_np, task["protective_stop_force"])
        reward += np.where(outcome_np == SUCCESS, 10.0, 0.0)
        reward -= np.where(np.isin(outcome_np, [SLIPPED, KNOCKED_OFF, COLLISION]), 2.0, 0.0)
        self.previous_distance = distance
        self.previous_load = load_np

        wood_motion = torch.linalg.norm(state["wood"] - self.settled_wood, dim=1).cpu().numpy()
        speed = torch.linalg.norm(state["tool_velocity"], dim=1).cpu().numpy()
        self.current_peak_contact = np.maximum(self.current_peak_contact, peak_np)
        self.current_peak_speed = np.maximum(self.current_peak_speed, speed)
        self.current_peak_wood = np.maximum(self.current_peak_wood, wood_motion)
        contact_now = self._contact_force()

        observations = self._observe(state, contact_now)
        dones = outcome_np != RUNNING
        infos = [{} for _ in range(n)]
        self._episode_rewards += reward
        self._episode_lengths += 1
        finished = np.flatnonzero(dones)
        if finished.size:
            pull = self.pull_direction.cpu().numpy()
            grip_np = final_grip.cpu().numpy()
            for i in finished:
                record = EpisodeRecord()
                record.detach_force = float(self.detach_force_np[i])
                record.final_stem_load = float(load_np[i])
                record.final_grip_force = float(grip_np[i])
                record.pull_x, record.pull_y, record.pull_z = (float(v) for v in pull[i])
                record.peak_contact_force = float(self.current_peak_contact[i])
                record.peak_wood_motion = float(self.current_peak_wood[i])
                record.peak_tool_speed = float(self.current_peak_speed[i])
                record.steps_to_grip = int(self.steps_to_grip[i])
                record.steps = int(self.control_step[i])
                self.last[i] = record
                infos[i] = {
                    "terminal_observation": observations[i].copy(),
                    "outcome": int(outcome_np[i]),
                    "is_success": outcome_np[i] == SUCCESS,
                    "TimeLimit.truncated": outcome_np[i] == TIMEOUT,
                    "episode": {"r": float(self._episode_rewards[i]), "l": int(self._episode_lengths[i]), "t": 0.0},
                }
                self._episode_rewards[i] = 0
                self._episode_lengths[i] = 0
            self._reset(finished)
            state = self._gather_state()
            new_tool = state["tool"].cpu().numpy().astype(np.float64)
            self.tool_target[finished] = new_tool[finished]
            new_distance = self._surface_distance(state).cpu().numpy()
            self.previous_distance[finished] = new_distance[finished]
            fresh = self._observe(state, torch.zeros(n, device=self.device))
            observations[finished] = fresh[finished]
        self._observations = observations
        return observations.copy(), reward.astype(np.float32), dones, infos

    def last_episode(self, index):
        return self.last[index]

    def close(self):
        if getattr(self, "timeline", None) is not None:
            self.timeline.stop()
            self.timeline = None

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("The simulations are not separate objects, so methods cannot be called on them one at a time.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
