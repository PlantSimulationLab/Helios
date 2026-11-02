"""Building the apple harvesting scene in IsaacSim (PhysX) so that it matches the Helios (MuJoCo) environment in harvest_env.cpp.

Needs a running SimulationApp, since pxr is only available inside one.
"""


import math
import os

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

ROBOT_JOINTS = ["drive", "yaw", "shoulder", "elbow", "wrist", "roll"]

# harvest_arm.urdf: link name, mass, center of mass, diagonal inertia, collision shapes. Collision shapes are (kind, size, offset, along_x),
# where a cylinder with along_x is rotated from z onto x as the URDF's rpy="0 1.5708 0" does.
ROBOT_LINKS = [
    ("world_base", 1.0, (0, 0, 0), (0.01, 0.01, 0.01), []),
    ("platform", 80.0, (0, 0, 0.22), (5, 5, 5), [("box", (0.55, 0.8, 0.2), (0, 0, 0.22), False)]),
    ("turret", 8.0, (0, 0, 0.5), (0.3, 0.3, 0.05), [("cylinder", (0.075, 0.5), (0, 0, 0.5), False)]),
    ("upper_arm", 4.0, (0.325, 0, 0), (0.01, 0.15, 0.15), [("cylinder", (0.05, 0.65), (0.325, 0, 0), True)]),
    ("forearm", 2.5, (0.3, 0, 0), (0.005, 0.08, 0.08), [("cylinder", (0.038, 0.6), (0.3, 0, 0), True)]),
    ("wrist_link", 0.5, (0.04, 0, 0), (0.0005, 0.001, 0.001), []),
    ("tool", 0.6, (0.14, 0, 0), (0.0005, 0.002, 0.002), [("cylinder", (0.018, 0.28), (0.14, 0, 0), True), ("sphere", (0.02,), (0.3, 0, 0), False)]),
]
# Joint name, parent, child, origin in the parent, axis in the parent frame, type, lower and upper limit (rad or m)
ROBOT_JOINT_TABLE = [
    ("drive", "world_base", "platform", (0, 0, 0), (0, 1, 0), "prismatic", -10.0, 10.0),
    ("yaw", "platform", "turret", (0.15, 0, 0), (0, 0, 1), "revolute", -2.5, 2.5),
    ("shoulder", "turret", "upper_arm", (0, 0, 0.75), (0, -1, 0), "revolute", -0.8, 2.4),
    ("elbow", "upper_arm", "forearm", (0.65, 0, 0), (0, -1, 0), "revolute", -2.6, 2.6),
    ("wrist", "forearm", "wrist_link", (0.6, 0, 0), (0, -1, 0), "revolute", -2.2, 2.2),
    ("roll", "wrist_link", "tool", (0, 0, 0), (1, 0, 0), "revolute", -3.0, 3.0),
]


def _joint_frame(axis):
    """A USD joint has an axis token, X, Y or Z, so a negative axis is made by turning the joint frame half way around."""
    x, y, z = axis
    if (x, y, z) == (0, -1, 0):
        return "Y", Gf.Quatf(0, 0, 0, 1)  # 180 degrees about z turns +y into -y
    if (x, y, z) == (0, 1, 0):
        return "Y", Gf.Quatf(1, 0, 0, 0)
    if (x, y, z) == (0, 0, 1):
        return "Z", Gf.Quatf(1, 0, 0, 0)
    if (x, y, z) == (1, 0, 0):
        return "X", Gf.Quatf(1, 0, 0, 0)
    raise ValueError(f"unsupported joint axis {axis}")


def build_robot(stage, path, local_position, world_position, servos):
    """The harvest arm as a fixed-base articulation in its zero pose, with the servos of harvest_env.cpp as joint drives.

    local_position is the robot's base relative to the prim it is defined under, and world_position the same point in the world, where
    the base is fixed. servos maps a joint name to [stiffness, damping, force limit, armature] in N/m or N m/rad, as in the Helios environment.
    """
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*local_position))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation.CreateEnabledSelfCollisionsAttr().Set(False)

    # Link frames in the zero pose, relative to the robot's base. Every joint origin is a translation, so frames are unrotated.
    offsets = {"world_base": Gf.Vec3d(0, 0, 0)}
    for name, parent, child, origin, *_ in ROBOT_JOINT_TABLE:
        offsets[child] = offsets[parent] + Gf.Vec3d(*origin)

    for name, mass, com, inertia, shapes in ROBOT_LINKS:
        link = UsdGeom.Xform.Define(stage, f"{path}/{name}")
        link.AddTranslateOp().Set(offsets[name])
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass_api.CreateMassAttr().Set(mass)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1, 0, 0, 0))
        PhysxSchema.PhysxContactReportAPI.Apply(link.GetPrim()).CreateThresholdAttr().Set(0.0)
        for k, (kind, size, offset, along_x) in enumerate(shapes):
            shape_path = f"{path}/{name}/collision_{k}"
            if kind == "box":
                shape = UsdGeom.Cube.Define(stage, shape_path)
                shape.CreateSizeAttr().Set(1.0)
                shape.AddTranslateOp().Set(Gf.Vec3d(*offset))
                shape.AddScaleOp().Set(Gf.Vec3f(*size))
            elif kind == "cylinder":
                shape = UsdGeom.Cylinder.Define(stage, shape_path)
                shape.CreateRadiusAttr().Set(size[0])
                shape.CreateHeightAttr().Set(size[1])
                shape.CreateAxisAttr().Set("X" if along_x else "Z")
                shape.AddTranslateOp().Set(Gf.Vec3d(*offset))
            else:
                shape = UsdGeom.Sphere.Define(stage, shape_path)
                shape.CreateRadiusAttr().Set(size[0])
                shape.AddTranslateOp().Set(Gf.Vec3d(*offset))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())

    # The base is fixed to the world, as the URDF's root link is in MuJoCo
    anchor = UsdPhysics.FixedJoint.Define(stage, f"{path}/world_anchor")
    anchor.CreateBody1Rel().SetTargets([f"{path}/world_base"])
    anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_position))
    anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))

    for name, parent, child, origin, axis, kind, lower, upper in ROBOT_JOINT_TABLE:
        token, frame = _joint_frame(axis)
        if kind == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{name}")
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/{name}")
        joint.CreateBody0Rel().SetTargets([f"{path}/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"{path}/{child}"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*origin))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot0Attr().Set(frame)
        joint.CreateLocalRot1Attr().Set(frame)
        joint.CreateAxisAttr().Set(token)
        stiffness, damping, force_limit, armature = servos[name]
        if kind == "prismatic":
            joint.CreateLowerLimitAttr().Set(lower)
            joint.CreateUpperLimitAttr().Set(upper)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateStiffnessAttr().Set(stiffness)
            drive.CreateDampingAttr().Set(damping)
        else:
            # USD angular limits and drives are per degree
            joint.CreateLowerLimitAttr().Set(math.degrees(lower))
            joint.CreateUpperLimitAttr().Set(math.degrees(upper))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateStiffnessAttr().Set(stiffness * math.pi / 180.0)
            drive.CreateDampingAttr().Set(damping * math.pi / 180.0)
        drive.CreateTypeAttr().Set("force")
        drive.CreateMaxForceAttr().Set(force_limit)
        drive.CreateTargetPositionAttr().Set(0.0)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
        physx_joint.CreateArmatureAttr().Set(armature)
        physx_joint.CreateMaxJointVelocityAttr().Set(1e6)
    return root


