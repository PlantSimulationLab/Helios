"""IsaacSim (PhysX) side of the comparison with the Helios physics engine plug-in (MuJoCo).

Loads a plant written by PlantArchitecture::writePlantStructureUSD(), adds a kinematic box, and moves it through the
crown exactly as physicsengine_isaac_comparison does in MuJoCo, from the sweep it wrote (<plant>_sweep.json). Writes the
position of every link over time in the same CSV format as the Helios program.

Run with IsaacSim's Python, for example:
    C:/isaacsim/python.bat isaac_box_sweep.py --usd out/apple_steps500.usda --sweep out/apple_sweep.json --dt 0.002 --dynamics gpu --out out/isaac_apple_dt0.002_gpu.csv

Written for and tested with IsaacSim 6.1.0 (Kit 110) on Windows.
"""

import argparse
import json
import sys
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--usd", required=True, help="plant written by writePlantStructureUSD()")
parser.add_argument("--sweep", required=True, help="<plant>_sweep.json written by physicsengine_isaac_comparison")
parser.add_argument("--dt", type=float, required=True, help="physics time step in s; should match the USD export's physics_steps_per_second")
parser.add_argument(
    "--dynamics",
    default="gpu",
    choices=["gpu", "cpu"],
    help="run PhysX dynamics on the GPU, as the USD file asks for, or on the CPU. Either way the state is read and written through the numpy tensor API, because the GPU tensor API cannot set kinematic targets in this IsaacSim version.",
)
parser.add_argument("--out", required=True, help="output CSV of link positions; a JSON summary is written beside it")
parser.add_argument("--record", type=float, default=0.02, help="interval in s between recorded samples")
parser.add_argument("--timing", action="store_true", help="read nothing back while stepping and write no CSV, so that the wall time is for physics alone")
parser.add_argument("--position-iterations", type=int, default=0, help="override the articulation's solver position iteration count (0 keeps the USD value)")
parser.add_argument("--duration", type=float, default=0, help="stop after this many simulated seconds, for diagnosis (0 runs the whole sweep)")
parser.add_argument("--stiffness-scale", type=float, default=1.0, help="multiply every joint stiffness and damping after loading, for diagnosis")
parser.add_argument("--solver", default="", choices=["", "TGS", "PGS"], help="override the PhysX solver (the USD file asks for TGS)")
parser.add_argument("--dynamic-friction", type=float, default=-1, help="override the dynamic friction of the plant's material (the USD export writes 0.3; MuJoCo has a single coefficient, 0.5)")
parser.add_argument("--no-gravity", action="store_true", help="switch gravity off, for diagnosis")
parser.add_argument("--velocity-iterations", type=int, default=0, help="override the articulation's solver velocity iteration count (0 keeps the PhysX default)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

# Newton switches itself on at startup unless told not to, and this comparison is with PhysX
simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/exts/isaacsim.physics.newton/auto_switch_on_startup=false", "--/exts/isaacsim.core.simulation_manager/default_engine=physx"]}
)

import numpy as np
import omni.physics.tensors
import omni.timeline
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade


def fail(message):
    print(f"ERROR (isaac_box_sweep.py): {message}", file=sys.stderr, flush=True)
    simulation_app.close()
    sys.exit(1)


with open(args.sweep) as f:
    sweep = json.load(f)
box_start = np.array(sweep["box_start"], dtype=np.float64)
box_end = np.array(sweep["box_end"], dtype=np.float64)
settle_time = sweep["settle_time"]
travel_time = sweep["travel_time"]
total_time = args.duration if args.duration > 0 else sweep["total_time"]
box_size = sweep["box_size"]


def box_position(t_since_settle):
    """Same schedule as main.cpp: rest, out across the crown, and back."""
    if t_since_settle <= 0:
        fraction = 0.0
    elif t_since_settle < travel_time:
        fraction = t_since_settle / travel_time
    else:
        fraction = max(0.0, 2.0 - t_since_settle / travel_time)
    return box_start + fraction * (box_end - box_start)


# The link table written by main.cpp, in Helios link order
link_table = args.sweep.replace("_sweep.json", "_links.csv")
link_rows = [line.split(",") for line in open(link_table).read().splitlines()[1:]]
helios_names = [row[1] for row in link_rows]
rest = np.array([[float(v) for v in row[5:8]] for row in link_rows])

success, stage = stage_utils.open_stage(args.usd)
if not success:
    fail(f"could not open {args.usd}")
while stage_utils.is_stage_loading():
    simulation_app.update()

if SimulationManager.get_active_physics_engine() != "physx":
    SimulationManager.switch_physics_engine("physx")
if SimulationManager.get_active_physics_engine() != "physx":
    fail(f"PhysX is not the active physics engine ({SimulationManager.get_active_physics_engine()})")

plant_prim = stage.GetDefaultPrim()
plant_path = plant_prim.GetPath().pathString
if args.position_iterations > 0:
    PhysxSchema.PhysxArticulationAPI(plant_prim).GetSolverPositionIterationCountAttr().Set(args.position_iterations)
if args.velocity_iterations > 0:
    PhysxSchema.PhysxArticulationAPI(plant_prim).CreateSolverVelocityIterationCountAttr().Set(args.velocity_iterations)

if args.no_gravity:
    UsdPhysics.Scene(stage.GetPrimAtPath("/PhysicsScene")).GetGravityMagnitudeAttr().Set(0.0)

# The kinematic box, with the plant's collision material
box_path = "/Box"
box = UsdGeom.Cube.Define(stage, box_path)
box.CreateSizeAttr().Set(box_size)
box.AddTranslateOp().Set(Gf.Vec3d(*box_start))
box_prim = box.GetPrim()
UsdPhysics.CollisionAPI.Apply(box_prim)
UsdPhysics.RigidBodyAPI.Apply(box_prim).CreateKinematicEnabledAttr().Set(True)
UsdPhysics.MassAPI.Apply(box_prim).CreateMassAttr().Set(1.0)
PhysxSchema.PhysxContactReportAPI.Apply(box_prim).CreateThresholdAttr().Set(0.0)
material = UsdShade.Material(stage.GetPrimAtPath(f"{plant_path}/PhysicsMaterials/WoodMaterial"))
if not material:
    fail(f"no physics material at {plant_path}/PhysicsMaterials/WoodMaterial")
UsdShade.MaterialBindingAPI.Apply(box_prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
if args.dynamic_friction >= 0:
    UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Set(args.dynamic_friction)

# The CPU device keeps state readback on, so that the numpy tensor API, and with it kinematic targets, works. GPU
# dynamics are then switched back on in the scene if asked for.
SimulationManager.setup_simulation(dt=args.dt, device="cpu")
SimulationManager.enable_gpu_dynamics(args.dynamics == "gpu")
SimulationManager.set_broadphase_type("GPU" if args.dynamics == "gpu" else "MBP")
if args.solver:
    SimulationManager.set_solver_type(args.solver)
timeline = omni.timeline.get_timeline_interface()
timeline.play()
timeline.commit()
SimulationManager.initialize_physics()

view = omni.physics.tensors.create_simulation_view("numpy", stage_id=stage_utils.get_stage_id(stage))
view.set_subspace_roots("/")
plant_view = view.create_articulation_view(plant_path)
box_view = view.create_rigid_body_view(box_path)
# Each filter pattern has to match exactly one body per sensor, so every link is listed
contact_view = view.create_rigid_contact_view(box_path, filter_patterns=[f"{plant_path}/Links/{name}" for name in helios_names], max_contact_data_count=20000)
if plant_view.count != 1 or box_view.count != 1 or contact_view.sensor_count != 1:
    fail(f"expected one plant, one box and one contact sensor, found {plant_view.count}, {box_view.count} and {contact_view.sensor_count}")

link_names = list(plant_view.shared_metatype.link_names)
if len(link_names) != sweep["link_count"]:
    fail(f"the articulation has {len(link_names)} links but Helios exported {sweep['link_count']}")
name_to_column = {name: i for i, name in enumerate(link_names)}
if args.stiffness_scale != 1.0:
    all_plants = np.arange(plant_view.count, dtype=np.uint32)
    plant_view.set_dof_stiffnesses(np.asarray(plant_view.get_dof_stiffnesses()) * args.stiffness_scale, all_plants)
    plant_view.set_dof_dampings(np.asarray(plant_view.get_dof_dampings()) * args.stiffness_scale, all_plants)
missing = [name for name in helios_names if name not in name_to_column]
if missing:
    fail(f"{len(missing)} Helios links are not in the articulation, for example {missing[:3]}")
order = np.array([name_to_column[name] for name in helios_names])

box_index = np.array([0], dtype=np.uint32)
box_target = np.zeros((1, 7), dtype=np.float32)
box_target[0, 6] = 1.0  # identity rotation, (qx, qy, qz, qw)

steps_per_record = max(1, round(args.record / args.dt))
total_steps = int(np.ceil(total_time / args.dt - 1e-6))


def set_box_target(step):
    """The box reaches the position for the end of this step, so PhysX gives it the velocity of the sweep during the step."""
    box_target[0, :3] = box_position((step + 1) * args.dt - settle_time)
    box_view.set_kinematic_targets(box_target, box_index)


def link_positions():
    return np.asarray(plant_view.get_link_transforms()).reshape(-1, 7)[order, :3]


def read_contacts():
    """Net force of the box on the plant, deepest penetration, and contact count, for the last step."""
    force_on_box = np.asarray(contact_view.get_net_contact_forces(args.dt)).reshape(3)
    _, _, _, separations, counts, _ = contact_view.get_contact_data(args.dt)
    count = int(np.asarray(counts).sum())
    separations = np.asarray(separations).reshape(-1)[:count]
    penetration = float(max(0.0, -separations.min())) if count > 0 else 0.0
    return -force_on_box, penetration, count


out = None
if not args.timing:
    out = open(args.out, "w")
    out.write("time,box_x,box_y,box_z,force_x,force_y,force_z,peak_force,peak_penetration,contacts")
    for name in helios_names:
        out.write(f",{name}_x,{name}_y,{name}_z")
    out.write("\n")

peak_force = 0.0
peak_penetration = 0.0
peak_contacts = 0
box_now = box_start.copy()
force = np.zeros(3)


def write_record(time_now):
    global peak_force, peak_penetration, peak_contacts
    positions = link_positions()
    if not np.all(np.isfinite(positions)):
        return False
    row = [time_now, *box_now, *force, peak_force, peak_penetration, peak_contacts]
    out.write(",".join(f"{v:.7g}" for v in row))
    out.write("," + ",".join(f"{v:.7g}" for v in positions.reshape(-1)))
    out.write("\n")
    peak_force = peak_penetration = 0.0
    peak_contacts = 0
    return True


# initialize_physics() steps twice to warm up; report how far the plant moved in those steps
warmup_drift = float(np.max(np.linalg.norm(link_positions() - rest, axis=1)))

diverged = False
physics_seconds = None
step = 0
wall_start = time.perf_counter()
if args.timing:
    while step < total_steps:
        set_box_target(step)
        SimulationManager.step(steps=1)
        step += 1
    final = link_positions()  # waits for the simulation to finish
    physics_seconds = time.perf_counter() - wall_start
    diverged = not np.all(np.isfinite(final))
else:
    write_record(0.0)
    while step < total_steps and not diverged:
        for _ in range(steps_per_record):
            if step >= total_steps:
                break
            set_box_target(step)
            SimulationManager.step(steps=1)
            step += 1
            box_now = box_position(step * args.dt - settle_time)
            force, penetration, count = read_contacts()
            peak_force = max(peak_force, float(np.linalg.norm(force)))
            peak_penetration = max(peak_penetration, penetration)
            peak_contacts = max(peak_contacts, count)
        diverged = not write_record(step * args.dt)
total_wall = time.perf_counter() - wall_start
if out is not None:
    out.close()

summary = {
    "engine": "physx",
    "isaacsim_version": open(r"C:\isaacsim\VERSION").read().strip() if sys.platform == "win32" else "",
    "plant": sweep["plant"],
    "dt": args.dt,
    "dynamics": args.dynamics,
    "gpu_dynamics": SimulationManager.is_gpu_dynamics_enabled(),
    "solver": SimulationManager.get_solver_type(),
    "position_iterations": PhysxSchema.PhysxArticulationAPI(plant_prim).GetSolverPositionIterationCountAttr().Get(),
    "links": len(link_names),
    "steps": step,
    "simulated_seconds": step * args.dt,
    "timing_run": args.timing,
    "physics_wall_seconds": physics_seconds,
    "total_wall_seconds": total_wall,
    "warmup_max_link_drift": warmup_drift,
    "diverged": diverged,
    "gravity": not args.no_gravity,
    "stiffness_scale": args.stiffness_scale,
    "dynamic_friction": UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get(),
    "static_friction": UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Get(),
    "velocity_iterations_override": args.velocity_iterations,
}
summary_file = args.out.replace(".csv", "_timing.json" if args.timing else ".json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2), flush=True)

timeline.stop()
simulation_app.close()
