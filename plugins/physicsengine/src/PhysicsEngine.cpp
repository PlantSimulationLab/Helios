/** \file "PhysicsEngine.cpp" Rigid-body physics simulation with the MuJoCo engine.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#include "PhysicsEngine.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {

    std::string linkBodyName(const std::string &articulation_name, const std::string &link_name) {
        return articulation_name + "_" + link_name;
    }

    std::string boxBodyName(uint boxID) {
        return "box" + std::to_string(boxID);
    }

    void quaternionToMatrix(float qw, float qx, float qy, float qz, mjtNum (&R)[9]) {
        mjtNum quat[4] = {qw, qx, qy, qz};
        mju_normalize4(quat);
        mju_quat2Mat(R, quat);
    }

    // Rotate a vector by the transpose of a row-major rotation matrix (world frame to body frame)
    vec3 rotateToBodyFrame(const mjtNum (&R)[9], const vec3 &v) {
        return make_vec3(float(R[0] * v.x + R[3] * v.y + R[6] * v.z), float(R[1] * v.x + R[4] * v.y + R[7] * v.z), float(R[2] * v.x + R[5] * v.y + R[8] * v.z));
    }

    vec3 rotateToWorldFrame(const mjtNum (&R)[9], const vec3 &v) {
        return make_vec3(float(R[0] * v.x + R[1] * v.y + R[2] * v.z), float(R[3] * v.x + R[4] * v.y + R[5] * v.z), float(R[6] * v.x + R[7] * v.y + R[8] * v.z));
    }

    // Mass of a kinematic box. It is large enough that what it touches cannot slow the box, whose velocity is reimposed every step regardless.
    constexpr float kinematic_box_mass = 1000.f;

    void writeVec(std::ostringstream &xml, const char *attribute, float a, float b, float c) {
        xml << " " << attribute << "=\"" << a << " " << b << " " << c << "\"";
    }

    void writeColor(std::ostringstream &xml, const RGBcolor &color) {
        xml << " rgba=\"" << color.r << " " << color.g << " " << color.b << " 1\"";
    }

    void writeShapePose(std::ostringstream &xml, const PhysicsCollisionShape &shape) {
        if (shape.position.x != 0.f || shape.position.y != 0.f || shape.position.z != 0.f) {
            writeVec(xml, "pos", shape.position.x, shape.position.y, shape.position.z);
        }
        if (shape.orientation[0] != 1. || shape.orientation[1] != 0. || shape.orientation[2] != 0. || shape.orientation[3] != 0.) {
            xml << " quat=\"" << shape.orientation[0] << " " << shape.orientation[1] << " " << shape.orientation[2] << " " << shape.orientation[3] << "\"";
        }
    }

} // namespace

PhysicsEngine::PhysicsEngine(Context *context) : context(context) {
    if (context == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine): The Context pointer must not be null.");
    }
}

PhysicsEngine::~PhysicsEngine() {
    for (Instance &instance: instances) {
        mj_deleteData(instance.data);
    }
    if (model != nullptr) {
        mj_deleteModel(model);
    }
}

void PhysicsEngine::disableMessages() {
    printmessages = false;
}

void PhysicsEngine::enableMessages() {
    printmessages = true;
}

void PhysicsEngine::requireInitialized(const std::string &method) const {
    if (model == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::" + method + "): The simulation has not been initialized. Call initialize() first.");
    }
}

void PhysicsEngine::requireNotInitialized(const std::string &method) const {
    if (model != nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::" + method + "): This must be done before initialize() is called.");
    }
}

const PhysicsEngine::Instance &PhysicsEngine::getInstance(uint instance, const std::string &method) const {
    requireInitialized(method);
    if (instance >= instances.size()) {
        helios_runtime_error("ERROR (PhysicsEngine::" + method + "): Instance " + std::to_string(instance) + " does not exist. There are " + std::to_string(instances.size()) + " instances.");
    }
    return instances[instance];
}

void PhysicsEngine::requireLink(uint articulationID, uint link_index, const std::string &method) const {
    if (articulationID >= articulations.size()) {
        helios_runtime_error("ERROR (PhysicsEngine::" + method + "): Articulation " + std::to_string(articulationID) + " does not exist.");
    }
    const size_t link_count = articulations[articulationID].description.links.size();
    if (link_index >= link_count) {
        helios_runtime_error("ERROR (PhysicsEngine::" + method + "): Link index " + std::to_string(link_index) + " is out of range for articulation " + std::to_string(articulationID) + ", which has " + std::to_string(link_count) + " links.");
    }
}

uint PhysicsEngine::addArticulation(const PhysicsArticulation &articulation) {
    requireNotInitialized("addArticulation");
    if (articulation.name.empty()) {
        helios_runtime_error("ERROR (PhysicsEngine::addArticulation): The articulation must have a name.");
    }
    if (articulation.links.empty()) {
        helios_runtime_error("ERROR (PhysicsEngine::addArticulation): Articulation " + articulation.name + " has no links.");
    }
    for (const ArticulationRecord &existing: articulations) {
        if (existing.description.name == articulation.name) {
            helios_runtime_error("ERROR (PhysicsEngine::addArticulation): Articulation " + articulation.name + " was already added.");
        }
    }
    for (size_t i = 0; i < articulation.links.size(); i++) {
        if (articulation.links[i].parent_index >= static_cast<int>(i)) {
            helios_runtime_error("ERROR (PhysicsEngine::addArticulation): Link " + articulation.links[i].name + " of articulation " + articulation.name + " comes before its parent. Every parent must precede its children.");
        }
    }
    for (const auto &[link_a, link_b]: articulation.contact_exclusions) {
        if (link_a >= articulation.links.size() || link_b >= articulation.links.size()) {
            helios_runtime_error("ERROR (PhysicsEngine::addArticulation): A contact exclusion of articulation " + articulation.name + " refers to a link that does not exist.");
        }
    }
    ArticulationRecord record;
    record.description = articulation;
    articulations.push_back(std::move(record));
    return static_cast<uint>(articulations.size() - 1);
}

uint PhysicsEngine::addKinematicBox(const vec3 &center, const vec3 &size, const RGBcolor &color) {
    requireNotInitialized("addKinematicBox");
    if (size.x <= 0.f || size.y <= 0.f || size.z <= 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::addKinematicBox): All box dimensions must be positive.");
    }
    BoxRecord box;
    box.center = center;
    box.size = size;
    box.objID = context->addBoxObject(center, size, make_int3(1, 1, 1), color);
    boxes.push_back(box);
    return static_cast<uint>(boxes.size() - 1);
}

void PhysicsEngine::bindObjectToLink(uint objID, uint articulationID, uint link_index) {
    requireNotInitialized("bindObjectToLink");
    requireLink(articulationID, link_index, "bindObjectToLink");
    if (!context->doesObjectExist(objID)) {
        helios_runtime_error("ERROR (PhysicsEngine::bindObjectToLink): Object " + std::to_string(objID) + " does not exist in the Context.");
    }
    object_link_requests.push_back({objID, articulationID, link_index});
}

void PhysicsEngine::bindTubeNodesToLinks(uint objID, uint articulationID, const std::vector<uint> &node_link_indices) {
    requireNotInitialized("bindTubeNodesToLinks");
    if (!context->doesObjectExist(objID) || context->getObjectType(objID) != OBJECT_TYPE_TUBE) {
        helios_runtime_error("ERROR (PhysicsEngine::bindTubeNodesToLinks): Object " + std::to_string(objID) + " is not a tube object in the Context.");
    }
    if (node_link_indices.size() != context->getTubeObjectNodeCount(objID)) {
        helios_runtime_error("ERROR (PhysicsEngine::bindTubeNodesToLinks): Tube object " + std::to_string(objID) + " has " + std::to_string(context->getTubeObjectNodeCount(objID)) + " nodes, but " + std::to_string(node_link_indices.size()) + " link indices were given.");
    }
    for (uint link_index: node_link_indices) {
        requireLink(articulationID, link_index, "bindTubeNodesToLinks");
    }
    tube_link_requests.push_back({objID, articulationID, node_link_indices});
}

std::string PhysicsEngine::buildMJCF(const PhysicsEngineParameters &params) const {

    std::ostringstream xml;
    xml << std::setprecision(9);

    xml << "<mujoco model=\"helios\">\n";
    xml << "  <compiler angle=\"radian\" balanceinertia=\"true\" autolimits=\"true\"/>\n";
    xml << "  <option timestep=\"" << params.timestep << "\" integrator=\"" << params.integrator << "\" solver=\"" << params.solver << "\" gravity=\"0 0 " << (params.gravity ? -9.81f : 0.f) << "\"/>\n";
    // Link geoms collide with kinematic boxes, and with each other only if articulation collisions are enabled: contype/conaffinity bit 1 is the links, bit 0 the boxes.
    xml << "  <default>\n";
    // Contacts between links are frictionless (condim 1), which is all that is needed to push them apart and is several times cheaper to solve; contacts with boxes keep friction, since a contact takes the larger condim of its two geoms.
    xml << "    <geom contype=\"2\" conaffinity=\"" << (params.articulation_collisions ? 3 : 1) << "\" condim=\"" << (params.articulation_collisions ? 1 : 3) << "\" friction=\"" << params.friction << " 0.005 0.0001\" solref=\"" << params.contact_time_constant << " 1\"";
    if (params.articulation_collisions) {
        xml << " margin=\"" << params.articulation_collision_margin << "\"";
    }
    xml << " rgba=\"0.45 0.32 0.2 1\"/>\n";
    xml << "  </default>\n";
    xml << "  <visual><global offwidth=\"1280\" offheight=\"960\"/></visual>\n";
    // MuJoCo exits the program when its memory arena overflows, which collisions between many links can do with
    // the default size. The arena is allocated but not touched until it is used.
    if (params.articulation_collisions) {
        xml << "  <size memory=\"512M\"/>\n";
    }

    // The bodies are written first, since they collect the mesh assets that have to precede them in the file
    std::ostringstream bodies;
    bodies << std::setprecision(9);
    std::ostringstream assets;
    assets << std::setprecision(9);
    bodies << "    <light pos=\"0 0 10\" dir=\"0 0 -1\" directional=\"true\"/>\n";
    bodies << "    <geom name=\"ground\" type=\"plane\" size=\"10 10 0.1\" contype=\"0\" conaffinity=\"0\" rgba=\"0.35 0.45 0.3 1\"/>\n";

    std::ostringstream exclusions;
    for (const ArticulationRecord &record: articulations) {
        const PhysicsArticulation &articulation = record.description;
        const auto &links = articulation.links;
        std::vector<std::vector<size_t>> children(links.size());
        std::vector<size_t> roots;
        for (size_t i = 0; i < links.size(); i++) {
            if (links[i].parent_index < 0) {
                roots.push_back(i);
            } else {
                children.at(links[i].parent_index).push_back(i);
            }
        }
        for (size_t root: roots) {
            writeLinkBody(bodies, assets, articulation, root, children, 2);
        }
        if (params.articulation_collisions) {
            for (const auto &[link_a, link_b]: articulation.contact_exclusions) {
                exclusions << "    <exclude body1=\"" << linkBodyName(articulation.name, links[link_a].name) << "\" body2=\"" << linkBodyName(articulation.name, links[link_b].name) << "\"/>\n";
            }
        }
    }

    for (size_t b = 0; b < boxes.size(); b++) {
        const auto &box = boxes[b];
        // A heavy, gravity-compensated body on three slide joints rather than a mocap body: a mocap body is teleported
        // between steps, so the contact solver sees it at rest and what it pushes sinks into it as it moves.
        bodies << "    <body name=\"" << boxBodyName(static_cast<uint>(b)) << "\" gravcomp=\"1\"";
        writeVec(bodies, "pos", box.center.x, box.center.y, box.center.z);
        bodies << ">\n";
        bodies << "      <joint type=\"slide\" axis=\"1 0 0\"/>\n      <joint type=\"slide\" axis=\"0 1 0\"/>\n      <joint type=\"slide\" axis=\"0 0 1\"/>\n";
        float box_inertia = kinematic_box_mass * (box.size.x * box.size.x + box.size.y * box.size.y + box.size.z * box.size.z) / 12.f;
        bodies << "      <inertial pos=\"0 0 0\" mass=\"" << kinematic_box_mass << "\"";
        writeVec(bodies, "diaginertia", box_inertia, box_inertia, box_inertia);
        bodies << "/>\n      <geom type=\"box\" contype=\"1\" conaffinity=\"2\" condim=\"3\"";
        writeVec(bodies, "size", 0.5f * box.size.x, 0.5f * box.size.y, 0.5f * box.size.z);
        bodies << " rgba=\"0.9 0.35 0.1 1\"/>\n    </body>\n";
    }

    if (!assets.str().empty()) {
        xml << "  <asset>\n" << assets.str() << "  </asset>\n";
    }
    xml << "  <worldbody>\n" << bodies.str() << "  </worldbody>\n";
    if (!exclusions.str().empty()) {
        xml << "  <contact>\n" << exclusions.str() << "  </contact>\n";
    }
    xml << "</mujoco>\n";

    return xml.str();
}

void PhysicsEngine::writeLinkBody(std::ostringstream &xml, std::ostringstream &assets, const PhysicsArticulation &articulation, size_t link_index, const std::vector<std::vector<size_t>> &children, int depth) const {

    const auto &links = articulation.links;
    const PhysicsLink &link = links.at(link_index);
    const std::string indent(2 * depth, ' ');
    const std::string body_name = linkBodyName(articulation.name, link.name);

    mjtNum R_link[9];
    quaternionToMatrix(link.qw, link.qx, link.qy, link.qz, R_link);

    // MJCF bodies are positioned relative to their parent body
    vec3 position = link.position;
    mjtNum quat[4] = {link.qw, link.qx, link.qy, link.qz};
    mju_normalize4(quat);
    if (link.parent_index >= 0) {
        const PhysicsLink &parent = links.at(link.parent_index);
        mjtNum R_parent[9];
        quaternionToMatrix(parent.qw, parent.qx, parent.qy, parent.qz, R_parent);
        position = rotateToBodyFrame(R_parent, link.position - parent.position);
        mjtNum parent_quat[4] = {parent.qw, parent.qx, parent.qy, parent.qz};
        mju_normalize4(parent_quat);
        mjtNum parent_conjugate[4];
        mju_negQuat(parent_conjugate, parent_quat);
        mjtNum relative_quat[4];
        mju_mulQuat(relative_quat, parent_conjugate, quat);
        mju_copy4(quat, relative_quat);
    }

    xml << indent << "<body name=\"" << body_name << "\"";
    writeVec(xml, "pos", position.x, position.y, position.z);
    xml << " quat=\"" << quat[0] << " " << quat[1] << " " << quat[2] << " " << quat[3] << "\">\n";

    float mass = std::max(link.mass, 1e-9f);
    xml << indent << "  <inertial pos=\"0 0 0\" mass=\"" << mass << "\"";
    writeVec(xml, "diaginertia", std::max(link.inertia_diagonal.x, 1e-14f), std::max(link.inertia_diagonal.y, 1e-14f), std::max(link.inertia_diagonal.z, 1e-14f));
    xml << "/>\n";

    // A body without a joint is welded to its parent
    if (link.parent_index >= 0 && link.joint_type == PHYSICS_JOINT_BALL) {
        // The joint sits at the parent-side anchor, which need not be the link's own origin
        const PhysicsLink &parent = links.at(link.parent_index);
        mjtNum R_parent[9];
        quaternionToMatrix(parent.qw, parent.qx, parent.qy, parent.qz, R_parent);
        vec3 anchor_world = parent.position + rotateToWorldFrame(R_parent, link.anchor_in_parent);
        vec3 anchor_local = rotateToBodyFrame(R_link, anchor_world - link.position);

        xml << indent << "  <joint type=\"ball\"";
        writeVec(xml, "pos", anchor_local.x, anchor_local.y, anchor_local.z);
        xml << " stiffness=\"" << link.stiffness << "\" damping=\"" << link.damping << "\"";
        if (link.armature > 0.f) {
            xml << " armature=\"" << link.armature << "\"";
        }
        if (link.max_rotation > 0.f) {
            xml << " range=\"0 " << link.max_rotation << "\"";
        }
        xml << "/>\n";
    }

    uint mesh_count = 0;
    for (const PhysicsCollisionShape &shape: link.shapes) {
        switch (shape.type) {
            case PHYSICS_SHAPE_CAPSULE:
                xml << indent << "  <geom type=\"capsule\" size=\"" << shape.size.x << " " << shape.size.y << "\"";
                break;
            case PHYSICS_SHAPE_SPHERE:
                xml << indent << "  <geom type=\"sphere\" size=\"" << shape.size.x << "\"";
                break;
            case PHYSICS_SHAPE_BOX:
            case PHYSICS_SHAPE_ELLIPSOID:
                xml << indent << "  <geom type=\"" << (shape.type == PHYSICS_SHAPE_BOX ? "box" : "ellipsoid") << "\"";
                writeVec(xml, "size", shape.size.x, shape.size.y, shape.size.z);
                break;
            case PHYSICS_SHAPE_CONVEX_MESH: {
                std::string mesh_name = body_name + "_t" + std::to_string(mesh_count++);
                assets << "    <mesh name=\"" << mesh_name << "\" vertex=\"";
                for (const vec3 &vertex: shape.mesh_vertices) {
                    assets << vertex.x << " " << vertex.y << " " << vertex.z << " ";
                }
                assets << "\"/>\n";
                xml << indent << "  <geom type=\"mesh\" mesh=\"" << mesh_name << "\"";
                break;
            }
        }
        writeShapePose(xml, shape);
        writeColor(xml, shape.color);
        xml << "/>\n";
    }

    for (size_t child: children.at(link_index)) {
        writeLinkBody(xml, assets, articulation, child, children, depth + 1);
    }

    xml << indent << "</body>\n";
}

void PhysicsEngine::initialize(const PhysicsEngineParameters &params) {

    if (model != nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The simulation was already initialized.");
    }
    if (articulations.empty()) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): No articulations were added. Call addArticulation() before initialize().");
    }
    if (params.timestep <= 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The time step must be positive.");
    }
    if (params.articulation_collision_margin < 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The articulation collision margin must not be negative.");
    }
    // MuJoCo silently replaces contacts stiffer than the time step can resolve, so a smaller value would not take effect
    if (params.contact_time_constant < 2.f * params.timestep) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The contact time constant (" + std::to_string(params.contact_time_constant) + " s) must be at least twice the time step (" + std::to_string(params.timestep) +
                             " s). Reduce the time step to allow stiffer contacts.");
    }

    mjcf_text = buildMJCF(params);

    char error[1024] = "";
    mjSpec *spec = mj_parseXMLString(mjcf_text.c_str(), nullptr, error, sizeof(error));
    if (spec == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not parse the generated model: " + std::string(error));
    }
    model = mj_compile(spec, nullptr);
    if (model == nullptr) {
        std::string compile_error = mjs_getError(spec);
        mj_deleteSpec(spec);
        helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not compile the generated model: " + compile_error);
    }
    mj_deleteSpec(spec);

    createInstance();

    for (ArticulationRecord &record: articulations) {
        const auto &links = record.description.links;
        record.link_body_IDs.resize(links.size());
        for (size_t i = 0; i < links.size(); i++) {
            std::string name = linkBodyName(record.description.name, links[i].name);
            int body_ID = mj_name2id(model, mjOBJ_BODY, name.c_str());
            if (body_ID < 0) {
                helios_runtime_error("ERROR (PhysicsEngine::initialize): Body " + name + " is missing from the compiled model.");
            }
            record.link_body_IDs[i] = body_ID;
        }
    }
    body_is_kinematic_box.assign(model->nbody, 0);
    for (size_t b = 0; b < boxes.size(); b++) {
        std::string name = boxBodyName(static_cast<uint>(b));
        boxes[b].body_ID = mj_name2id(model, mjOBJ_BODY, name.c_str());
        if (boxes[b].body_ID < 0 || model->body_jntnum[boxes[b].body_ID] != 3) {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): Kinematic box " + name + " is missing from the compiled model.");
        }
        body_is_kinematic_box[boxes[b].body_ID] = 1;
        int first_joint = model->body_jntadr[boxes[b].body_ID];
        boxes[b].qpos_address = model->jnt_qposadr[first_joint];
        boxes[b].dof_address = model->jnt_dofadr[first_joint];
    }

    bindContextGeometry();

    if (printmessages) {
        std::cout << "MuJoCo model compiled: " << model->nbody << " bodies, " << model->nv << " degrees of freedom, " << model->ngeom << " geoms." << std::endl;
    }
}

uint PhysicsEngine::createInstance() {
    requireInitialized("createInstance");
    Instance instance;
    instance.data = mj_makeData(model);
    if (instance.data == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::createInstance): MuJoCo could not allocate the simulation data.");
    }
    instance.box_velocities.resize(boxes.size());
    mj_forward(model, instance.data);
    instances.push_back(std::move(instance));
    return static_cast<uint>(instances.size() - 1);
}

uint PhysicsEngine::getInstanceCount() const {
    return static_cast<uint>(instances.size());
}

void PhysicsEngine::resetInstance(uint instance) {
    static_cast<void>(getInstance(instance, "resetInstance"));
    mj_resetData(model, instances[instance].data);
    std::fill(instances[instance].box_velocities.begin(), instances[instance].box_velocities.end(), make_vec3(0, 0, 0));
    mj_forward(model, instances[instance].data);
}

size_t PhysicsEngine::addRestPose(int body_ID) {
    const mjData *data = instances.front().data;
    BodyRestPose pose;
    pose.body_ID = body_ID;
    pose.position = make_vec3(float(data->xpos[3 * body_ID]), float(data->xpos[3 * body_ID + 1]), float(data->xpos[3 * body_ID + 2]));
    for (int k = 0; k < 9; k++) {
        pose.rotation[k] = float(data->xmat[9 * body_ID + k]);
    }
    rest_poses.push_back(pose);
    return rest_poses.size() - 1;
}

void PhysicsEngine::bindContextGeometry() {

    for (ArticulationRecord &record: articulations) {
        record.link_rest_pose_indices.resize(record.link_body_IDs.size());
        for (size_t i = 0; i < record.link_body_IDs.size(); i++) {
            record.link_rest_pose_indices[i] = addRestPose(record.link_body_IDs[i]);
        }
    }

    for (const ObjectLinkRequest &request: object_link_requests) {
        bindObjectPrimitives(request.objID, articulations[request.articulationID].link_rest_pose_indices[request.link_index]);
    }

    for (const TubeLinkRequest &request: tube_link_requests) {
        const ArticulationRecord &record = articulations[request.articulationID];
        const std::vector<vec3> nodes = context->getTubeObjectNodes(request.objID);
        TubeBinding tube;
        tube.objID = request.objID;
        for (size_t n = 0; n < nodes.size(); n++) {
            size_t rest_pose_index = record.link_rest_pose_indices[request.node_link_indices[n]];
            const BodyRestPose &pose = rest_poses[rest_pose_index];
            mjtNum R[9];
            for (int k = 0; k < 9; k++) {
                R[k] = pose.rotation[k];
            }
            tube.node_rest_pose_index.push_back(rest_pose_index);
            tube.node_local_position.push_back(rotateToBodyFrame(R, nodes[n] - pose.position));
        }
        tube_bindings.push_back(std::move(tube));
    }

    for (const BoxRecord &box: boxes) {
        bindObjectPrimitives(box.objID, addRestPose(box.body_ID));
    }
}

void PhysicsEngine::bindObjectPrimitives(uint objID, size_t rest_pose_index) {
    for (uint UUID: context->getObjectPrimitiveUUIDs(objID)) {
        PrimitiveBinding binding;
        binding.UUID = UUID;
        binding.rest_pose_index = rest_pose_index;
        context->getPrimitiveTransformationMatrix(UUID, binding.rest_transform);
        primitive_bindings.push_back(binding);
    }
}

void PhysicsEngine::setKinematicBoxPosition(uint boxID, const vec3 &center, const vec3 &velocity, uint instance) {
    static_cast<void>(getInstance(instance, "setKinematicBoxPosition"));
    if (boxID >= boxes.size()) {
        helios_runtime_error("ERROR (PhysicsEngine::setKinematicBoxPosition): Box ID " + std::to_string(boxID) + " does not exist.");
    }
    const BoxRecord &box = boxes[boxID];
    mjData *data = instances[instance].data;
    // The slide joints measure the offset from the position the box was added at
    vec3 offset = center - box.center;
    data->qpos[box.qpos_address] = offset.x;
    data->qpos[box.qpos_address + 1] = offset.y;
    data->qpos[box.qpos_address + 2] = offset.z;
    instances[instance].box_velocities[boxID] = velocity;
}

void PhysicsEngine::stepInstance(Instance &instance, uint step_count) const {
    for (uint s = 0; s < step_count; s++) {
        for (size_t b = 0; b < boxes.size(); b++) {
            instance.data->qvel[boxes[b].dof_address] = instance.box_velocities[b].x;
            instance.data->qvel[boxes[b].dof_address + 1] = instance.box_velocities[b].y;
            instance.data->qvel[boxes[b].dof_address + 2] = instance.box_velocities[b].z;
        }
        mj_step(model, instance.data);
        checkSimulationHealth(instance.data);
    }
}

void PhysicsEngine::step(uint step_count, uint instance) {
    static_cast<void>(getInstance(instance, "step"));
    stepInstance(instances[instance], step_count);
}

void PhysicsEngine::stepInstances(uint step_count, uint thread_count) {
    requireInitialized("stepInstances");

    if (thread_count == 0) {
        thread_count = std::max(1u, std::thread::hardware_concurrency());
    }
    thread_count = std::min(thread_count, static_cast<uint>(instances.size()));

    // Exceptions cannot cross a thread boundary, so the first failure is kept and raised once all threads have finished
    std::atomic<size_t> next_instance{0};
    std::exception_ptr first_failure;
    std::mutex failure_mutex;
    auto worker = [&]() {
        for (size_t i = next_instance++; i < instances.size(); i = next_instance++) {
            try {
                stepInstance(instances[i], step_count);
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!first_failure) {
                    first_failure = std::current_exception();
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (uint t = 1; t < thread_count; t++) {
        threads.emplace_back(worker);
    }
    worker();
    for (std::thread &thread: threads) {
        thread.join();
    }
    if (first_failure) {
        std::rethrow_exception(first_failure);
    }
}

void PhysicsEngine::checkSimulationHealth(const mjData *data) const {
    // MuJoCo silently resets the state when the accelerations become invalid, so a divergence would otherwise look like everything snapping back to rest
    if (data->warning[mjWARN_BADQACC].number > 0 || data->warning[mjWARN_BADQPOS].number > 0 || data->warning[mjWARN_BADQVEL].number > 0) {
        helios_runtime_error("ERROR (PhysicsEngine::step): The simulation diverged at t = " + std::to_string(data->time) + " s. Reduce the time step, use the discrete integrator, or add joint armature.");
    }
    if (data->warning[mjWARN_CONTACTFULL].number > 0 || data->warning[mjWARN_CNSTRFULL].number > 0) {
        helios_runtime_error("ERROR (PhysicsEngine::step): The MuJoCo memory arena is full, so contacts were dropped at t = " + std::to_string(data->time) + " s.");
    }
}

void PhysicsEngine::updateContextGeometry(uint instance) {
    const mjData *data = getInstance(instance, "updateContextGeometry").data;

    // Rigid motion of each bound body from its rest pose: x -> R_delta * x + t_delta, with R_delta = R * R_rest^T
    std::vector<std::array<float, 16>> rest_to_current(rest_poses.size());
    for (size_t i = 0; i < rest_poses.size(); i++) {
        const BodyRestPose &pose = rest_poses[i];
        const mjtNum *R = data->xmat + 9 * pose.body_ID;
        const mjtNum *p = data->xpos + 3 * pose.body_ID;
        std::array<float, 16> &T = rest_to_current[i];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                T[4 * r + c] = float(R[3 * r] * pose.rotation[3 * c] + R[3 * r + 1] * pose.rotation[3 * c + 1] + R[3 * r + 2] * pose.rotation[3 * c + 2]);
            }
            T[4 * r + 3] = float(p[r]) - (T[4 * r] * pose.position.x + T[4 * r + 1] * pose.position.y + T[4 * r + 2] * pose.position.z);
        }
        T[12] = 0.f;
        T[13] = 0.f;
        T[14] = 0.f;
        T[15] = 1.f;
    }

    for (const PrimitiveBinding &binding: primitive_bindings) {
        float transform[16];
        const std::array<float, 16> &T = rest_to_current[binding.rest_pose_index];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                transform[4 * r + c] = T[4 * r] * binding.rest_transform[c] + T[4 * r + 1] * binding.rest_transform[4 + c] + T[4 * r + 2] * binding.rest_transform[8 + c] + T[4 * r + 3] * binding.rest_transform[12 + c];
            }
        }
        context->setPrimitiveTransformationMatrix(binding.UUID, transform);
    }

    for (const TubeBinding &tube: tube_bindings) {
        std::vector<vec3> nodes(tube.node_local_position.size());
        for (size_t n = 0; n < nodes.size(); n++) {
            const BodyRestPose &pose = rest_poses[tube.node_rest_pose_index[n]];
            const mjtNum *R = data->xmat + 9 * pose.body_ID;
            const mjtNum *p = data->xpos + 3 * pose.body_ID;
            const vec3 &local = tube.node_local_position[n];
            nodes[n] = make_vec3(float(R[0] * local.x + R[1] * local.y + R[2] * local.z + p[0]), float(R[3] * local.x + R[4] * local.y + R[5] * local.z + p[1]), float(R[6] * local.x + R[7] * local.y + R[8] * local.z + p[2]));
        }
        context->setTubeNodes(tube.objID, nodes);
    }
}

void PhysicsEngine::writeMJCF(const std::string &filename) const {
    requireInitialized("writeMJCF");
    std::ofstream file(filename);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (PhysicsEngine::writeMJCF): Could not open " + filename + " for writing. Make sure the directory exists and is writable.");
    }
    file << mjcf_text;
}

mjModel *PhysicsEngine::getMuJoCoModel() const {
    requireInitialized("getMuJoCoModel");
    return model;
}

mjData *PhysicsEngine::getMuJoCoData(uint instance) const {
    return getInstance(instance, "getMuJoCoData").data;
}

float PhysicsEngine::getSimulationTime(uint instance) const {
    return float(getInstance(instance, "getSimulationTime").data->time);
}

float PhysicsEngine::getTimestep() const {
    requireInitialized("getTimestep");
    return float(model->opt.timestep);
}

uint PhysicsEngine::getLinkCount() const {
    size_t count = 0;
    for (const ArticulationRecord &record: articulations) {
        count += record.description.links.size();
    }
    return static_cast<uint>(count);
}

uint PhysicsEngine::getContactCount(uint instance) const {
    return static_cast<uint>(getInstance(instance, "getContactCount").data->ncon);
}

float PhysicsEngine::getMaxLinkDisplacement(uint instance) const {
    const mjData *data = getInstance(instance, "getMaxLinkDisplacement").data;
    float max_displacement = 0.f;
    for (const BodyRestPose &pose: rest_poses) {
        if (body_is_kinematic_box[pose.body_ID]) {
            continue;
        }
        const mjtNum *p = data->xpos + 3 * pose.body_ID;
        vec3 displacement = make_vec3(float(p[0]), float(p[1]), float(p[2])) - pose.position;
        max_displacement = std::max(max_displacement, displacement.magnitude());
    }
    return max_displacement;
}

float PhysicsEngine::getMaxLinkSpeed(uint instance) const {
    const mjData *data = getInstance(instance, "getMaxLinkSpeed").data;
    mjtNum max_speed = 0;
    for (const ArticulationRecord &record: articulations) {
        for (int body_ID: record.link_body_IDs) {
            mjtNum velocity[6]; // rotational then translational velocity of the body origin, in the world frame
            mj_objectVelocity(model, data, mjOBJ_BODY, body_ID, velocity, 0);
            max_speed = std::max(max_speed, mju_norm3(velocity + 3));
        }
    }
    return float(max_speed);
}

float PhysicsEngine::getMaxKinematicBoxPenetration(uint instance) const {
    const mjData *data = getInstance(instance, "getMaxKinematicBoxPenetration").data;
    float max_penetration = 0.f;
    for (int c = 0; c < data->ncon; c++) {
        if (!isLinkContact(data, c)) {
            max_penetration = std::max(max_penetration, float(-data->contact[c].dist));
        }
    }
    return max_penetration;
}

bool PhysicsEngine::isLinkContact(const mjData *data, int contact_index) const {
    const mjContact &contact = data->contact[contact_index];
    return !body_is_kinematic_box[model->geom_bodyid[contact.geom[0]]] && !body_is_kinematic_box[model->geom_bodyid[contact.geom[1]]];
}

uint PhysicsEngine::getLinkOverlapCount(float min_depth, uint instance) const {
    const mjData *data = getInstance(instance, "getLinkOverlapCount").data;
    uint count = 0;
    for (int c = 0; c < data->ncon; c++) {
        if (isLinkContact(data, c) && -data->contact[c].dist > min_depth) {
            count++;
        }
    }
    return count;
}

float PhysicsEngine::getMaxLinkOverlapDepth(uint instance) const {
    const mjData *data = getInstance(instance, "getMaxLinkOverlapDepth").data;
    float max_depth = 0.f;
    for (int c = 0; c < data->ncon; c++) {
        if (isLinkContact(data, c)) {
            max_depth = std::max(max_depth, float(-data->contact[c].dist));
        }
    }
    return max_depth;
}

vec3 PhysicsEngine::getLinkPosition(uint articulationID, uint link_index, uint instance) const {
    const mjData *data = getInstance(instance, "getLinkPosition").data;
    requireLink(articulationID, link_index, "getLinkPosition");
    const mjtNum *p = data->xpos + 3 * articulations[articulationID].link_body_IDs[link_index];
    return make_vec3(float(p[0]), float(p[1]), float(p[2]));
}
