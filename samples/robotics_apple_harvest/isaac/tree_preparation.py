"""Preparing a tree exported by harvest_env_export_tree() for PhysX. Needs a running SimulationApp, since pxr is only available inside one."""

import json

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


def _rotation(matrix):
    """The rotation of a Gf.Matrix4d as a numpy matrix acting on column vectors. USD matrices act on row vectors, so this is the transpose."""
    return np.array([[matrix[c][r] for c in range(3)] for r in range(3)])


def _set_mass_properties(prim, parts):
    """Mass, center of mass and principal inertia of a body made of parts (mass, position, rotation, diagonal inertia), each in the body's frame."""
    total = sum(m for m, _, _, _ in parts)
    center = sum(m * p for m, p, _, _ in parts) / total
    inertia = np.zeros((3, 3))
    for m, p, rotation, diagonal in parts:
        d = p - center
        inertia += rotation @ np.diag(diagonal) @ rotation.T + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    values, vectors = np.linalg.eigh(inertia)
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] = -vectors[:, 0]
    # The principal frame's axes, the columns of vectors, as a USD (row-vector) rotation matrix
    principal = Gf.Matrix3d(*vectors.T.flatten().tolist())
    mass_api = UsdPhysics.MassAPI(prim)
    mass_api.GetMassAttr().Set(float(max(total, 1e-9)))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center.tolist()))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(max(v, 1e-14)) for v in values]))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(principal.ExtractRotation().GetQuat()))


def _bind_material(stage, prim, material_path):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)), UsdShade.Tokens.weakerThanDescendants, "physics")


def _lock(joint):
    for axis in ("rotX", "rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(joint, axis)
        limit.CreateLowAttr().Set(1.0)
        limit.CreateHighAttr().Set(-1.0)
        drive = UsdPhysics.DriveAPI(joint, axis)
        if drive:
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(0.0)


def prepare_tree(json_path, usd_path, output_path):
    """Make an exported tree simulate in PhysX as harvest_env.cpp simulates it in MuJoCo, at a cost PhysX can afford.

    In MuJoCo the tree is one articulation in which most links are welded: all internodes except a few behind each candidate fruit,
    and the candidate fruit, held by detachable welds. Welded links make one rigid body, but a PhysX articulation pays for each link,
    and a whole tree as one articulation took 230 ms per step for four trees. So here:

    - Every welded link is merged into the body that carries it: its collision shape moves over, and its mass and inertia are added
      to that body's. Wood welded all the way to the fixed base is merged into the base link.
    - What is left is one articulation, with self-collision off as in MuJoCo, so that no shapes of the tree are ever paired: the base
      link with all the rigid wood, fixed to the world, and below it the links that MuJoCo connects by ball joints, with the export's
      joints.
    - Candidate fruit stay separate links on fixed joints, so that the load on their stems can be read and the grip applied to them.
      MuJoCo holds them with a soft weld (4 ms time constant); here the joint is rigid.
    - Fruit collide as the sphere MuJoCo gives them, and every body gets its mass and inertia from Helios rather than from PhysX.

    Returns counts of what was done.
    """
    description = json.load(open(json_path))
    candidates = {c["link"] for c in description["candidates"]}
    stage = Usd.Stage.Open(usd_path)
    plant_path = stage.GetDefaultPrim().GetPath().pathString
    links = description["links"]
    names = [link["name"] for link in links]
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link_prims = [stage.GetPrimAtPath(f"{plant_path}/Links/{name}") for name in names]
    for name, prim in zip(names, link_prims):
        if not prim:
            raise RuntimeError(f"link {name} is in {json_path} but not in {usd_path}")
    poses = [xform_cache.GetLocalToWorldTransform(prim).RemoveScaleShear() for prim in link_prims]

    # Links are listed parents first. Every welded link belongs to the body that carries it: the base link for wood welded all the
    # way to it, which cannot move, and otherwise the nearest link above it with a ball joint.
    root = next(i for i, link in enumerate(links) if link["parent"] < 0)
    owner = list(range(len(links)))
    for i, link in enumerate(links):
        parent = link["parent"]
        if parent < 0:
            continue
        own_body = link["joint"] == "sprung" or link["name"] in candidates
        if not own_body:
            owner[i] = owner[parent]
    bodies = [i for i in range(len(links)) if owner[i] == i and i != root]
    kept = set(bodies) | {root}

    joints = {}
    for prim in stage.GetPrimAtPath(f"{plant_path}/Joints").GetChildren():
        targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        if targets:
            joints[targets[0].name] = prim
    material = f"{plant_path}/PhysicsMaterials/WoodMaterial"

    # Move the collision shape of each welded link to its body, and collect mass properties in that body's frame
    parts = {i: [] for i in kept}
    for i, link in enumerate(links):
        target = owner[i]
        relative = poses[i] * poses[target].GetInverse()
        parts[target].append((max(link["mass"], 1e-9), np.array(relative.ExtractTranslation()), _rotation(relative), np.array(link["inertia"], dtype=float)))
        if target == i:
            continue
        collision = link_prims[i].GetChild("Collision")
        if collision and collision.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(collision)
            moved = UsdGeom.Capsule.Define(stage, f"{link_prims[target].GetPath()}/Collision_{names[i]}")
            moved.CreateRadiusAttr().Set(capsule.GetRadiusAttr().Get())
            moved.CreateHeightAttr().Set(capsule.GetHeightAttr().Get())
            moved.CreateAxisAttr().Set("Z")
            moved.CreatePurposeAttr().Set("guide")
            moved.AddTranslateOp().Set(relative.ExtractTranslation())
            moved.AddOrientOp().Set(Gf.Quatf(relative.ExtractRotationQuat()))
            UsdPhysics.CollisionAPI.Apply(moved.GetPrim())
            _bind_material(stage, moved.GetPrim(), material)

    for i, link in enumerate(links):
        if "sphere_radius" in link and i in kept and link_prims[i].GetChild("Collision"):
            stage.RemovePrim(link_prims[i].GetChild("Collision").GetPath())
            sphere = UsdGeom.Sphere.Define(stage, f"{link_prims[i].GetPath()}/Collision")
            sphere.CreateRadiusAttr().Set(link["sphere_radius"])
            sphere.AddTranslateOp().Set(Gf.Vec3d(*link["sphere_center"]))
            sphere.CreatePurposeAttr().Set("guide")
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            _bind_material(stage, sphere.GetPrim(), material)

    # Joints of the bodies, re-expressed in the frame of the body that carries the parent; candidate fruit locked
    for i in bodies:
        joint = joints[names[i]]
        parent = links[i]["parent"]
        carrier = owner[parent]
        if carrier != parent:
            joint_api = UsdPhysics.Joint(joint)
            frame = Gf.Matrix4d().SetRotate(Gf.Quatd(joint_api.GetLocalRot0Attr().Get())) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(joint_api.GetLocalPos0Attr().Get()))
            in_carrier = frame * poses[parent] * poses[carrier].GetInverse()
            joint_api.GetBody0Rel().SetTargets([link_prims[carrier].GetPath()])
            joint_api.GetLocalPos0Attr().Set(Gf.Vec3f(in_carrier.ExtractTranslation()))
            joint_api.GetLocalRot0Attr().Set(Gf.Quatf(in_carrier.ExtractRotationQuat()))
        if names[i] in candidates:
            _lock(joint)

    for i in kept:
        _set_mass_properties(link_prims[i], parts[i])

    # The fixed joint at the base and the joints of the bodies stay; everything else was merged
    keep_joints = {names[i] for i in kept}
    for child, joint in joints.items():
        if child not in keep_joints:
            stage.RemovePrim(joint.GetPath())
    for i in range(len(links)):
        if i not in kept:
            stage.RemovePrim(link_prims[i].GetPath())

    scene = stage.GetPrimAtPath("/PhysicsScene")
    if scene:
        stage.RemovePrim(scene.GetPath())
    stage.GetRootLayer().Export(output_path)

    # Where each link of the Helios tree ended up: the body it belongs to, and its origin in that body's frame
    owners = {names[i]: [names[owner[i]], list((poses[i] * poses[owner[i]].GetInverse()).ExtractTranslation())] for i in range(len(links))}
    with open(output_path.replace(".usda", ".json"), "w") as f:
        json.dump({"owners": owners}, f)
    return {
        "links": len(links),
        "articulation_links": len(kept),
        "sprung": sum(1 for i in bodies if names[i] not in candidates),
        "candidates": len(candidates),
        "shapes_on_base": sum(1 for i in range(len(links)) if owner[i] == root),
        "shapes_merged_into_moving_bodies": sum(1 for i in range(len(links)) if owner[i] not in (i, root)),
    }
