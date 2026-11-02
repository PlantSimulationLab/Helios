"""Time PhysX on copies of one harvest tree with its robot, to choose how the IsaacSim environment is built.

C:/isaacsim/python.bat bench_tree.py --trees ../build/isaac_trees --seed 1 --copies 4 --device cuda:0
"""

import argparse
import json
import math
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--trees", required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--copies", type=int, default=4)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--strip-tree-shapes", action="store_true", help="diagnosis only: remove every collision shape of the trees")
parser.add_argument("--no-trees", action="store_true", help="diagnosis only: robots without trees")
parser.add_argument("--no-tree-collisions", action="store_true", help="diagnosis only: switch off every collision shape of the trees")
parser.add_argument("--position-iterations", type=int, default=0, help="override the trees' articulation solver position iterations (the export writes 32)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "extra_args": ["--/exts/isaacsim.physics.newton/auto_switch_on_startup=false", "--/exts/isaacsim.core.simulation_manager/default_engine=physx"]})

import numpy as np
import omni.physics.tensors
import omni.timeline
import omni.usd
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_scene import ROBOT_JOINTS, build_robot
from tree_preparation import prepare_tree

json_path = os.path.join(args.trees, f"tree_{args.seed}.json")
usd_path = os.path.join(args.trees, f"tree_{args.seed}.usda")
prepared = os.path.join(args.trees, f"tree_{args.seed}_isaac.usda")
t0 = time.perf_counter()
info = prepare_tree(json_path, usd_path, prepared)
print("prepared", info, f"{time.perf_counter() - t0:.1f} s", flush=True)
task = json.load(open(json_path))["task"]

stage_utils.create_new_stage()
stage = stage_utils.get_current_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physx_scene.CreateSolverTypeAttr().Set("TGS")
# Merged welded links make bodies with hundreds of shapes, more than PhysX's default GPU buffers hold
physx_scene.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(2**22)
physx_scene.CreateGpuTotalAggregatePairsCapacityAttr().Set(2**22)
physx_scene.CreateGpuFoundLostPairsCapacityAttr().Set(2**22)
physx_scene.CreateGpuMaxRigidContactCountAttr().Set(2**21)
physx_scene.CreateGpuMaxRigidPatchCountAttr().Set(2**19)

spacing = 6.0
columns = max(1, int(math.ceil(math.sqrt(args.copies))))
for k in range(args.copies):
    origin = Gf.Vec3d(spacing * (k % columns), spacing * (k // columns), 0)
    env = UsdGeom.Xform.Define(stage, f"/World/envs/env_{k}")
    env.AddTranslateOp().Set(origin)
    tree = stage.DefinePrim(f"/World/envs/env_{k}/Tree")
    if not args.no_trees:
        tree.GetReferences().AddReference(prepared)
    if args.strip_tree_shapes:
        for prim in [p for p in Usd.PrimRange(tree) if p.HasAPI(UsdPhysics.CollisionAPI)]:
            prim.SetActive(False)
    if args.no_tree_collisions:
        for prim in Usd.PrimRange(tree):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    if args.position_iterations > 0:
        for prim in Usd.PrimRange(tree):
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                PhysxSchema.PhysxArticulationAPI(prim).GetSolverPositionIterationCountAttr().Set(args.position_iterations)
    robot_base = Gf.Vec3d(*task["robot_base"]) + origin
    build_robot(stage, f"/World/envs/env_{k}/Robot", tuple(task["robot_base"]), tuple(robot_base), task["servos"])
    for name, position in zip(ROBOT_JOINTS, task["tucked_pose"]):
        joint = stage.GetPrimAtPath(f"/World/envs/env_{k}/Robot/{name}")
        kind = "linear" if name == "drive" else "angular"
        PhysxSchema.JointStateAPI.Apply(joint, kind).CreatePositionAttr().Set(position)
        UsdPhysics.DriveAPI(joint, kind).GetTargetPositionAttr().Set(position)

SimulationManager.setup_simulation(dt=task["physics_timestep"], device=args.device)
timeline = omni.timeline.get_timeline_interface()
timeline.play()
timeline.commit()
SimulationManager.initialize_physics()
view = SimulationManager.get_physics_simulation_view()
trees = view.create_articulation_view("/World/envs/env_*/Tree" if not args.no_trees else "/World/envs/env_*/Robot")
robots = view.create_articulation_view("/World/envs/env_*/Robot")
print("tree view", trees.count, "links", trees.max_links, "dofs", trees.max_dofs, "| robot view", robots.count, robots.max_links, robots.max_dofs, flush=True)

positions_start = trees.get_link_transforms().numpy().reshape(trees.count, -1, 7)[:, :, :3].copy()
SimulationManager.step(steps=10)
_ = trees.get_link_transforms().numpy()
start = time.perf_counter()
SimulationManager.step(steps=args.steps)
positions = trees.get_link_transforms().numpy().reshape(trees.count, -1, 7)[:, :, :3]
seconds = time.perf_counter() - start
moved = np.linalg.norm(positions - positions_start, axis=2)
robot_q = robots.get_dof_positions().numpy().reshape(robots.count, -1)
print(f"{args.copies} copies on {args.device}: {1000 * seconds / args.steps:.2f} ms per step, {args.copies * args.steps / seconds:.0f} env-steps/s; tree links moved median {1000 * np.median(moved):.2f} mm max {1000 * moved.max():.1f} mm, finite {np.all(np.isfinite(positions))}", flush=True)
print("robot joints (m, rad)", np.round(robot_q[0], 3), flush=True)

timeline.stop()
simulation_app.close()
