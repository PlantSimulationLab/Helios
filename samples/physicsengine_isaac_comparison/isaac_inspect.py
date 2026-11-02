"""Check what PhysX makes of a plant written by PlantArchitecture::writePlantStructureUSD().

1. From the USD file alone: whether the two frames of each joint coincide in the exported rest pose. PhysX builds a
   reduced-coordinate articulation from the joint frames and drives each joint towards aligned frames, so a mismatch
   moves links when the simulation starts and preloads the joint springs.
2. After PhysX has parsed the file: the joint stiffness, damping, armature and limits, and the link masses, inertias and
   centers of mass that PhysX actually uses, against the values in the Helios link table (<plant>_links.csv).

Run with IsaacSim's Python:
    C:/isaacsim/python.bat isaac_inspect.py --usd out/apple_steps500.usda --links out/apple_links.csv --out out/inspect_apple_steps500.json
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--usd", required=True)
parser.add_argument("--links", required=True, help="<plant>_links.csv written by physicsengine_isaac_comparison")
parser.add_argument("--out", required=True, help="JSON summary")
parser.add_argument("--dynamics", default="gpu", choices=["gpu", "cpu"])
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/exts/isaacsim.physics.newton/auto_switch_on_startup=false", "--/exts/isaacsim.core.simulation_manager/default_engine=physx"]}
)

import numpy as np
import omni.physics.tensors
import omni.timeline
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom, UsdPhysics

link_rows = [line.split(",") for line in open(args.links).read().splitlines()]
header = link_rows[0]
link_rows = link_rows[1:]
column = {name: i for i, name in enumerate(header)}
helios = {row[column["name"]]: row for row in link_rows}
link_types = ["internode", "petiole", "peduncle", "leaf", "fruit", "flower"]

success, stage = stage_utils.open_stage(args.usd)
if not success:
    print(f"ERROR: could not open {args.usd}", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)
while stage_utils.is_stage_loading():
    simulation_app.update()
plant_path = stage.GetDefaultPrim().GetPath().pathString

# 1. Joint frames in the rest pose, from the USD file alone
xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
position_mismatch = []
rotation_mismatch_deg = []
mismatch_by_type = {}
for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{plant_path}/Joints")):
    joint = UsdPhysics.Joint(prim)
    if not joint:
        continue
    body0 = joint.GetBody0Rel().GetTargets()
    body1 = joint.GetBody1Rel().GetTargets()
    if not body1:
        continue
    child_pose = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body1[0])).RemoveScaleShear()
    parent_pose = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0[0])).RemoveScaleShear() if body0 else Gf.Matrix4d(1.0)
    local_pos0 = Gf.Vec3d(joint.GetLocalPos0Attr().Get())
    local_pos1 = Gf.Vec3d(joint.GetLocalPos1Attr().Get())
    local_rot0 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
    local_rot1 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot1Attr().Get()))
    anchor0 = parent_pose.Transform(local_pos0)
    anchor1 = child_pose.Transform(local_pos1)
    frame0 = local_rot0 * parent_pose.ExtractRotation()
    frame1 = local_rot1 * child_pose.ExtractRotation()
    angle = (frame0 * frame1.GetInverse()).GetAngle()
    angle = min(angle, 360.0 - angle)
    position_mismatch.append((anchor0 - anchor1).GetLength())
    rotation_mismatch_deg.append(angle)
    child_name = body1[0].name
    child_type = link_types[int(helios[child_name][column["type"]])] if child_name in helios else "unknown"
    mismatch_by_type.setdefault(child_type, []).append(angle)

position_mismatch = np.array(position_mismatch)
rotation_mismatch_deg = np.array(rotation_mismatch_deg)
result = {
    "usd": args.usd,
    "joints": int(len(position_mismatch)),
    "joint_anchor_mismatch_m": {"median": float(np.median(position_mismatch)), "max": float(position_mismatch.max())},
    "joint_frame_mismatch_deg": {
        "median": float(np.median(rotation_mismatch_deg)),
        "p90": float(np.percentile(rotation_mismatch_deg, 90)),
        "max": float(rotation_mismatch_deg.max()),
        "count_over_1deg": int(np.sum(rotation_mismatch_deg > 1.0)),
        "median_by_child_type": {k: float(np.median(v)) for k, v in mismatch_by_type.items()},
    },
}

# 2. What PhysX parsed
SimulationManager.setup_simulation(dt=0.002, device="cpu")
SimulationManager.enable_gpu_dynamics(args.dynamics == "gpu")
timeline = omni.timeline.get_timeline_interface()
timeline.play()
timeline.commit()
SimulationManager.initialize_physics()
view = omni.physics.tensors.create_simulation_view("numpy", stage_id=stage_utils.get_stage_id(stage))
plant_view = view.create_articulation_view(plant_path)
metatype = plant_view.shared_metatype
link_names = list(metatype.link_names)

masses = np.asarray(plant_view.get_masses()).reshape(-1)
inertias = np.asarray(plant_view.get_inertias()).reshape(len(link_names), 9)
coms = np.asarray(plant_view.get_coms()).reshape(len(link_names), 7)
helios_mass = np.array([float(helios[name][column["mass"]]) for name in link_names])
helios_inertia = np.array([[float(helios[name][column[f"inertia_{a}"]]) for a in "xyz"] for name in link_names])
link_type = np.array([link_types[int(helios[name][column["type"]])] for name in link_names])
mass_ratio = masses / np.maximum(helios_mass, 1e-12)
inertia_ratio = inertias[:, [0, 4, 8]].sum(axis=1) / np.maximum(helios_inertia.sum(axis=1), 1e-15)
com_offset = np.linalg.norm(coms[:, :3], axis=1)


def ratio_stats(values, mask):
    values = values[mask]
    return {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


result["links"] = {
    "count": len(link_names),
    "physx_over_helios_mass": {t: ratio_stats(mass_ratio, link_type == t) for t in np.unique(link_type)},
    "physx_over_helios_inertia_trace": {t: ratio_stats(inertia_ratio, link_type == t) for t in np.unique(link_type)},
    "com_offset_from_link_frame_m": {t: ratio_stats(com_offset, link_type == t) for t in np.unique(link_type)},
    "total_mass_physx": float(masses.sum()),
    "total_mass_helios": float(helios_mass.sum()),
}

# DOFs: three rotational DOFs per sprung joint; the joint prim is named J_<parent>_to_<child>
dof_paths = list(plant_view.dof_paths[0])
stiffness = np.asarray(plant_view.get_dof_stiffnesses()).reshape(-1)
damping = np.asarray(plant_view.get_dof_dampings()).reshape(-1)
armature = np.asarray(plant_view.get_dof_armatures()).reshape(-1)
limits = np.asarray(plant_view.get_dof_limits()).reshape(-1, 2)
max_velocity = np.asarray(plant_view.get_dof_max_velocities()).reshape(-1)
friction = np.asarray(plant_view.get_dof_friction_coefficients()).reshape(-1)
max_force = np.asarray(plant_view.get_dof_max_forces()).reshape(-1)
drive_types = np.asarray(plant_view.get_drive_types()).reshape(-1)
drive_model = np.asarray(plant_view.get_dof_drive_model_properties()).reshape(len(dof_paths), -1)
dof_types = np.asarray(plant_view.get_dof_types()).reshape(-1)
dof_motions = np.asarray(plant_view.get_dof_motions()).reshape(-1)
child_of_dof = [path.split("/")[-1].split("_to_")[-1] for path in dof_paths]
helios_stiffness = np.array([float(helios[c][column["stiffness"]]) for c in child_of_dof])
helios_damping = np.array([float(helios[c][column["damping"]]) for c in child_of_dof])
helios_armature = np.array([float(helios[c][column["armature"]]) for c in child_of_dof])
positive = helios_stiffness > 0
result["dofs"] = {
    "count": len(dof_paths),
    "physx_over_helios_stiffness": ratio_stats(stiffness / np.where(positive, helios_stiffness, 1), positive),
    "physx_over_helios_damping": ratio_stats(damping / np.where(helios_damping > 0, helios_damping, 1), helios_damping > 0),
    "physx_over_helios_armature": ratio_stats(armature / np.where(helios_armature > 0, helios_armature, 1), helios_armature > 0),
    "limits_rad_or_deg": {"low_median": float(np.median(limits[:, 0])), "high_median": float(np.median(limits[:, 1]))},
    "max_velocity_median": float(np.median(max_velocity)),
    "friction_median": float(np.median(friction)),
    "max_force": {"median": float(np.median(max_force)), "min": float(max_force.min())},
    "drive_types": {str(int(k)): int(v) for k, v in zip(*np.unique(drive_types, return_counts=True))},
    "drive_model_properties_median": [float(v) for v in np.median(drive_model, axis=0)],
    "dof_types": {str(int(k)): int(v) for k, v in zip(*np.unique(dof_types, return_counts=True))},
    "dof_motions": {str(int(k)): int(v) for k, v in zip(*np.unique(dof_motions, return_counts=True))},
    "metatype_dof_count": int(metatype.dof_count),
}

with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)

timeline.stop()
simulation_app.close()
