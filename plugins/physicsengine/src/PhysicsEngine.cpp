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

    std::string detachEqualityName(const std::string &body_name) {
        return body_name + "_detach";
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

    std::string geomName(const std::string &body_name, size_t shape_index) {
        return body_name + "#" + std::to_string(shape_index);
    }

    // Unit quaternion (w,x,y,z) of a link's orientation
    void linkQuaternion(const PhysicsLink &link, mjtNum (&quat)[4]) {
        quat[0] = link.qw;
        quat[1] = link.qx;
        quat[2] = link.qy;
        quat[3] = link.qz;
        mju_normalize4(quat);
    }

    // Pose of a frame given in the world, expressed in another frame given in the world: the second pose composed with the inverse of the first
    void relativePose(const mjtNum (&frame_pos)[3], const mjtNum (&frame_quat)[4], const mjtNum (&pos)[3], const mjtNum (&quat)[4], mjtNum (&relative_pos)[3], mjtNum (&relative_quat)[4]) {
        mjtNum conjugate[4];
        mju_negQuat(conjugate, frame_quat);
        mjtNum offset[3];
        mju_sub3(offset, pos, frame_pos);
        mju_rotVecQuat(relative_pos, offset, conjugate);
        mju_mulQuat(relative_quat, conjugate, quat);
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
    if (spec != nullptr) {
        mj_deleteSpec(spec);
    }
    for (mjSpec *model_file_spec: model_file_specs) {
        mj_deleteSpec(model_file_spec);
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

void PhysicsEngine::addModelFile(const std::string &filename, const std::string &prefix, const vec3 &position, float yaw_degrees, bool gravity_compensation) {
    requireNotInitialized("addModelFile");
    if (prefix.empty()) {
        helios_runtime_error("ERROR (PhysicsEngine::addModelFile): The prefix must not be empty, since it is what keeps the names of the model apart from everything else in the simulation.");
    }
    for (const ModelFileRecord &existing: model_files) {
        if (existing.prefix == prefix) {
            helios_runtime_error("ERROR (PhysicsEngine::addModelFile): A model with the prefix \"" + prefix + "\" was already added.");
        }
    }
    if (!std::filesystem::exists(filename)) {
        helios_runtime_error("ERROR (PhysicsEngine::addModelFile): The model file " + filename + " does not exist.");
    }
    model_files.push_back({filename, prefix, position, yaw_degrees, gravity_compensation});
}

void PhysicsEngine::addPositionServo(const std::string &joint_name, float stiffness, float damping, float force_limit, float armature) {
    requireNotInitialized("addPositionServo");
    if (stiffness <= 0.f || damping < 0.f || force_limit < 0.f || armature < 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::addPositionServo): The servo on joint " + joint_name + " must have positive stiffness, and damping, force limit, and armature that are not negative.");
    }
    servo_requests.push_back({joint_name, stiffness, damping, force_limit, armature});
}

void PhysicsEngine::addForceTorqueSensor(const std::string &name, const std::string &body_name, const vec3 &position) {
    requireNotInitialized("addForceTorqueSensor");
    sensor_requests.push_back({name, body_name, position});
}

void PhysicsEngine::bindObjectToBody(uint objID, const std::string &body_name) {
    requireNotInitialized("bindObjectToBody");
    if (!context->doesObjectExist(objID)) {
        helios_runtime_error("ERROR (PhysicsEngine::bindObjectToBody): Object " + std::to_string(objID) + " does not exist in the Context.");
    }
    object_body_requests.push_back({objID, body_name});
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
    // A colliding ground uses the same bits as a kinematic box: it collides with articulation links, but not with kinematic boxes or loaded models
    bodies << "    <geom name=\"ground\" type=\"plane\" size=\"10 10 0.1\" contype=\"" << (params.ground_collisions ? 1 : 0) << "\" conaffinity=\"" << (params.ground_collisions ? 2 : 0) << "\" rgba=\"0.35 0.45 0.3 1\"/>\n";

    std::ostringstream exclusions;
    std::ostringstream equalities;
    for (const ArticulationRecord &record: articulations) {
        const PhysicsArticulation &articulation = record.description;
        const auto &links = articulation.links;
        std::vector<std::vector<size_t>> children(links.size());
        std::vector<size_t> roots;
        // The links whose shapes and mass each body carries, and those the world carries
        std::vector<std::vector<size_t>> members(links.size());
        std::vector<size_t> world_members;
        for (size_t i = 0; i < links.size(); i++) {
            // A detachable link is a free body at the top of the tree, which a weld holds to its parent until it is released
            if (links[i].parent_index < 0 || links[i].joint_type == PHYSICS_JOINT_DETACHABLE) {
                roots.push_back(i);
            } else {
                children.at(links[i].parent_index).push_back(i);
            }
            if (record.link_host[i] < 0) {
                world_members.push_back(i);
            } else {
                members.at(record.link_host[i]).push_back(i);
            }
            if (links[i].parent_index >= 0 && links[i].joint_type == PHYSICS_JOINT_DETACHABLE) {
                const std::string body_name = linkBodyName(articulation.name, links[i].name);
                equalities << "    <weld name=\"" << detachEqualityName(body_name) << "\" body1=\"" << body_name << "\"";
                // Without a second body, the weld holds the link to the world
                if (record.link_host[links[i].parent_index] >= 0) {
                    equalities << " body2=\"" << hostBodyName(record, links[i].parent_index) << "\"";
                }
                // The weld is anchored at the parent link, wherever its body is: MuJoCo would otherwise anchor it at the origin of the body that carries the parent, which for the world can be far away,
                // and the weld's small angular give would then move the link by much more
                const vec3 &anchor = record.link_offset[links[i].parent_index];
                writeVec(equalities, "anchor", anchor.x, anchor.y, anchor.z);
                equalities << " solref=\"" << params.contact_time_constant << " 1\"/>\n";
            }
        }
        for (size_t member: world_members) {
            writeLinkShapes(bodies, assets, record, member, -1, "    ");
        }
        for (size_t root: roots) {
            writeLinkBody(bodies, assets, record, root, children, members, 2);
        }
        if (params.articulation_collisions) {
            for (const auto &[link_a, link_b]: articulation.contact_exclusions) {
                // Shapes of one body never collide with each other, and MuJoCo rejects an exclusion of a body from itself
                if (record.link_host[link_a] == record.link_host[link_b]) {
                    continue;
                }
                exclusions << "    <exclude body1=\"" << hostBodyName(record, link_a) << "\" body2=\"" << hostBodyName(record, link_b) << "\"/>\n";
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
    if (!equalities.str().empty()) {
        xml << "  <equality>\n" << equalities.str() << "  </equality>\n";
    }
    xml << "</mujoco>\n";

    return xml.str();
}

void PhysicsEngine::computeLinkHosts(bool merge_welded_links) {
    for (ArticulationRecord &record: articulations) {
        const auto &links = record.description.links;
        record.link_host.assign(links.size(), -2);
        // MuJoCo never lets a body collide with its parent body, unless the parent is the world. A link with a jointed child keeps a body of its own, so that the
        // child still does not collide with it, as links built overlapping what they grow from (a petiole and its internode) should not.
        std::vector<char> has_jointed_child(links.size(), 0);
        for (const PhysicsLink &link: links) {
            if (link.parent_index >= 0 && link.joint_type == PHYSICS_JOINT_BALL) {
                has_jointed_child[link.parent_index] = 1;
            }
        }
        // A link's host is found from its parent's, so each chain is followed up to a link whose host is known
        std::function<int(size_t)> hostOf = [&](size_t i) -> int {
            if (record.link_host[i] != -2) {
                return record.link_host[i];
            }
            const PhysicsLink &link = links[i];
            int host;
            if (!merge_welded_links || has_jointed_child[i] || link.joint_type == PHYSICS_JOINT_BALL || (link.parent_index >= 0 && link.joint_type == PHYSICS_JOINT_DETACHABLE)) {
                host = static_cast<int>(i);
            } else if (link.parent_index < 0) {
                host = -1; // a root is welded to the world
            } else {
                host = hostOf(static_cast<size_t>(link.parent_index));
            }
            record.link_host[i] = host;
            return host;
        };
        record.link_offset.resize(links.size());
        for (size_t i = 0; i < links.size(); i++) {
            const int host = hostOf(i);
            if (host < 0) {
                record.link_offset[i] = links[i].position;
            } else {
                mjtNum R_host[9];
                quaternionToMatrix(links[host].qw, links[host].qx, links[host].qy, links[host].qz, R_host);
                record.link_offset[i] = rotateToBodyFrame(R_host, links[i].position - links[host].position);
            }
        }
    }
}

std::string PhysicsEngine::hostBodyName(const ArticulationRecord &record, size_t link_index) const {
    const int host = record.link_host.at(link_index);
    return host < 0 ? std::string("world") : linkBodyName(record.description.name, record.description.links.at(host).name);
}

void PhysicsEngine::writeLinkShapes(std::ostringstream &xml, std::ostringstream &assets, const ArticulationRecord &record, size_t link_index, int host, const std::string &indent) const {
    const PhysicsLink &link = record.description.links.at(link_index);
    const std::string body_name = linkBodyName(record.description.name, link.name);

    // The pose of the link in the frame of the body that carries it: none for its own body, and its world pose for the world
    const bool own_frame = host == static_cast<int>(link_index);
    mjtNum link_pos[3] = {0, 0, 0};
    mjtNum link_quat[4] = {1, 0, 0, 0};
    if (!own_frame) {
        mjtNum world_pos[3] = {link.position.x, link.position.y, link.position.z};
        mjtNum world_quat[4];
        linkQuaternion(link, world_quat);
        if (host < 0) {
            mju_copy3(link_pos, world_pos);
            mju_copy4(link_quat, world_quat);
        } else {
            const PhysicsLink &host_link = record.description.links.at(host);
            mjtNum host_pos[3] = {host_link.position.x, host_link.position.y, host_link.position.z};
            mjtNum host_quat[4];
            linkQuaternion(host_link, host_quat);
            relativePose(host_pos, host_quat, world_pos, world_quat, link_pos, link_quat);
        }
    }

    uint mesh_count = 0;
    for (size_t k = 0; k < link.shapes.size(); k++) {
        const PhysicsCollisionShape &shape = link.shapes[k];
        xml << indent << "<geom name=\"" << geomName(body_name, k) << "\"";
        switch (shape.type) {
            case PHYSICS_SHAPE_CAPSULE:
                xml << " type=\"capsule\" size=\"" << shape.size.x << " " << shape.size.y << "\"";
                break;
            case PHYSICS_SHAPE_SPHERE:
                xml << " type=\"sphere\" size=\"" << shape.size.x << "\"";
                break;
            case PHYSICS_SHAPE_BOX:
            case PHYSICS_SHAPE_ELLIPSOID:
                xml << " type=\"" << (shape.type == PHYSICS_SHAPE_BOX ? "box" : "ellipsoid") << "\"";
                writeVec(xml, "size", shape.size.x, shape.size.y, shape.size.z);
                break;
            case PHYSICS_SHAPE_CONVEX_MESH: {
                std::string mesh_name = body_name + "_t" + std::to_string(mesh_count++);
                assets << "    <mesh name=\"" << mesh_name << "\" vertex=\"";
                for (const vec3 &vertex: shape.mesh_vertices) {
                    assets << vertex.x << " " << vertex.y << " " << vertex.z << " ";
                }
                assets << "\"/>\n";
                xml << " type=\"mesh\" mesh=\"" << mesh_name << "\"";
                break;
            }
        }
        if (own_frame) {
            writeShapePose(xml, shape);
        } else {
            // The shape's pose in the link, composed with the link's pose in the body that carries it
            mjtNum shape_pos[3] = {shape.position.x, shape.position.y, shape.position.z};
            mjtNum shape_quat[4] = {shape.orientation[0], shape.orientation[1], shape.orientation[2], shape.orientation[3]};
            mju_normalize4(shape_quat);
            mjtNum pos[3];
            mju_rotVecQuat(pos, shape_pos, link_quat);
            mju_addTo3(pos, link_pos);
            mjtNum quat[4];
            mju_mulQuat(quat, link_quat, shape_quat);
            writeVec(xml, "pos", float(pos[0]), float(pos[1]), float(pos[2]));
            xml << " quat=\"" << quat[0] << " " << quat[1] << " " << quat[2] << " " << quat[3] << "\"";
        }
        writeColor(xml, shape.color);
        xml << "/>\n";
    }
}

void PhysicsEngine::writeLinkBody(std::ostringstream &xml, std::ostringstream &assets, const ArticulationRecord &record, size_t link_index, const std::vector<std::vector<size_t>> &children, const std::vector<std::vector<size_t>> &members, int depth) const {

    const auto &links = record.description.links;
    const PhysicsLink &link = links.at(link_index);

    // A link merged into another body has had its shapes and mass written with that body. Its children are nested in that body.
    if (record.link_host[link_index] != static_cast<int>(link_index)) {
        for (size_t child: children.at(link_index)) {
            writeLinkBody(xml, assets, record, child, children, members, depth);
        }
        return;
    }

    const std::string indent(2 * depth, ' ');
    const std::string body_name = linkBodyName(record.description.name, link.name);

    mjtNum R_link[9];
    quaternionToMatrix(link.qw, link.qx, link.qy, link.qz, R_link);

    // MJCF bodies are positioned relative to the body they are nested in, which carries the parent link. A detachable link has no parent body.
    const bool detachable = link.parent_index >= 0 && link.joint_type == PHYSICS_JOINT_DETACHABLE;
    const int frame_host = link.parent_index >= 0 && !detachable ? record.link_host.at(link.parent_index) : -1;
    vec3 position = link.position;
    mjtNum quat[4];
    linkQuaternion(link, quat);
    if (frame_host >= 0) {
        const PhysicsLink &parent = links.at(frame_host);
        mjtNum R_parent[9];
        quaternionToMatrix(parent.qw, parent.qx, parent.qy, parent.qz, R_parent);
        position = rotateToBodyFrame(R_parent, link.position - parent.position);
        mjtNum parent_quat[4];
        linkQuaternion(parent, parent_quat);
        mjtNum parent_conjugate[4];
        mju_negQuat(parent_conjugate, parent_quat);
        mjtNum relative_quat[4];
        mju_mulQuat(relative_quat, parent_conjugate, quat);
        mju_copy4(quat, relative_quat);
    }

    xml << indent << "<body name=\"" << body_name << "\"";
    writeVec(xml, "pos", position.x, position.y, position.z);
    xml << " quat=\"" << quat[0] << " " << quat[1] << " " << quat[2] << " " << quat[3] << "\">\n";

    const std::vector<size_t> &carried = members.at(link_index);
    if (carried.size() == 1) {
        float mass = std::max(link.mass, 1e-9f);
        xml << indent << "  <inertial pos=\"0 0 0\" mass=\"" << mass << "\"";
        writeVec(xml, "diaginertia", std::max(link.inertia_diagonal.x, 1e-14f), std::max(link.inertia_diagonal.y, 1e-14f), std::max(link.inertia_diagonal.z, 1e-14f));
        xml << "/>\n";
    } else {
        // The mass of every link the body carries, about their common center of mass, in the body frame. Each link's mass is centered on its origin, as for a body of its own.
        double total_mass = 0;
        mjtNum center[3] = {0, 0, 0};
        for (size_t member: carried) {
            const double mass = std::max(links[member].mass, 1e-9f);
            total_mass += mass;
            for (int k = 0; k < 3; k++) {
                center[k] += mass * mjtNum(k == 0 ? record.link_offset[member].x : (k == 1 ? record.link_offset[member].y : record.link_offset[member].z));
            }
        }
        mju_scl3(center, center, 1.0 / total_mass);
        mjtNum inertia[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        mjtNum link_quat[4];
        linkQuaternion(link, link_quat);
        mjtNum link_conjugate[4];
        mju_negQuat(link_conjugate, link_quat);
        for (size_t member: carried) {
            const PhysicsLink &carried_link = links[member];
            const double mass = std::max(carried_link.mass, 1e-9f);
            // Rotation of the member in the body frame
            mjtNum member_quat[4];
            linkQuaternion(carried_link, member_quat);
            mjtNum relative_quat[4];
            mju_mulQuat(relative_quat, link_conjugate, member_quat);
            mjtNum R[9];
            mju_quat2Mat(R, relative_quat);
            const mjtNum diagonal[3] = {std::max(carried_link.inertia_diagonal.x, 1e-14f), std::max(carried_link.inertia_diagonal.y, 1e-14f), std::max(carried_link.inertia_diagonal.z, 1e-14f)};
            const mjtNum d[3] = {record.link_offset[member].x - center[0], record.link_offset[member].y - center[1], record.link_offset[member].z - center[2]};
            const mjtNum d_squared = mju_dot3(d, d);
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    // R diag(I) R^T, plus the parallel-axis term m (|d|^2 E - d d^T)
                    mjtNum rotated = 0;
                    for (int k = 0; k < 3; k++) {
                        rotated += R[3 * r + k] * diagonal[k] * R[3 * c + k];
                    }
                    inertia[3 * r + c] += rotated + mass * ((r == c ? d_squared : 0) - d[r] * d[c]);
                }
            }
        }
        xml << indent << "  <inertial";
        writeVec(xml, "pos", float(center[0]), float(center[1]), float(center[2]));
        xml << " mass=\"" << total_mass << "\" fullinertia=\"" << inertia[0] << " " << inertia[4] << " " << inertia[8] << " " << inertia[1] << " " << inertia[2] << " " << inertia[5] << "\"/>\n";
    }

    if (detachable) {
        xml << indent << "  <freejoint/>\n";
    }

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

    for (size_t member: carried) {
        writeLinkShapes(xml, assets, record, member, static_cast<int>(link_index), indent + "  ");
    }

    for (size_t child: children.at(link_index)) {
        writeLinkBody(xml, assets, record, child, children, members, depth + 1);
    }

    xml << indent << "</body>\n";
}

void PhysicsEngine::initialize(const PhysicsEngineParameters &params) {

    if (model != nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The simulation was already initialized.");
    }
    if (articulations.empty() && model_files.empty()) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): Nothing was added to simulate. Call addArticulation() or addModelFile() before initialize().");
    }
    if (params.timestep <= 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The time step must be positive.");
    }
    if (params.articulation_collision_margin < 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The articulation collision margin must not be negative.");
    }
    if (params.memory_arena_megabytes < 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The memory arena size must not be negative; use 0 to choose it automatically.");
    }
    // MuJoCo silently replaces contacts stiffer than the time step can resolve, so a smaller value would not take effect
    if (params.contact_time_constant < 2.f * params.timestep) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): The contact time constant (" + std::to_string(params.contact_time_constant) + " s) must be at least twice the time step (" + std::to_string(params.timestep) +
                             " s). Reduce the time step to allow stiffer contacts.");
    }

    computeLinkHosts(params.merge_welded_links);
    mjcf_text = buildMJCF(params);

    char error[1024] = "";
    spec = mj_parseXMLString(mjcf_text.c_str(), nullptr, error, sizeof(error));
    if (spec == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not parse the generated model: " + std::string(error));
    }
    attachModelFiles(params);
    addServosAndSensors();

    // MuJoCo ends the program when its memory arena overflows, and its collision search needs memory that grows with the square of the number of shapes.
    // Articulation collisions already have a large arena (see buildMJCF), which is allocated but not touched until it is used.
    if (!params.articulation_collisions) {
        mjtSize shape_count = 0;
        for (mjsElement *element = mjs_firstElement(spec, mjOBJ_GEOM); element != nullptr; element = mjs_nextElement(spec, element)) {
            shape_count++;
        }
        spec->memory = 16 * 1024 * 1024 + 8 * shape_count * shape_count;
    }
    if (params.memory_arena_megabytes > 0.f) {
        spec->memory = static_cast<mjtSize>(double(params.memory_arena_megabytes) * 1024.0 * 1024.0);
    }

    model = mj_compile(spec, nullptr);
    if (model == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not compile the model: " + std::string(mjs_getError(spec)));
    }
    if (params.articulation_collisions && params.keep_initial_link_overlaps) {
        excludeInitialLinkOverlaps();
    }

    createInstance();

    for (ArticulationRecord &record: articulations) {
        const auto &links = record.description.links;
        record.link_body_IDs.resize(links.size());
        for (size_t i = 0; i < links.size(); i++) {
            const std::string name = hostBodyName(record, i);
            int body_ID = mj_name2id(model, mjOBJ_BODY, name.c_str());
            if (body_ID < 0) {
                helios_runtime_error("ERROR (PhysicsEngine::initialize): Body " + name + " is missing from the compiled model.");
            }
            record.link_body_IDs[i] = body_ID;
        }
        record.link_detach_equality_IDs.assign(links.size(), -1);
        for (size_t i = 0; i < links.size(); i++) {
            if (links[i].parent_index >= 0 && links[i].joint_type == PHYSICS_JOINT_DETACHABLE) {
                record.link_detach_equality_IDs[i] = mj_name2id(model, mjOBJ_EQUALITY, detachEqualityName(linkBodyName(record.description.name, links[i].name)).c_str());
                if (record.link_detach_equality_IDs[i] < 0) {
                    helios_runtime_error("ERROR (PhysicsEngine::initialize): The constraint holding detachable link " + links[i].name + " is missing from the compiled model.");
                }
            }
        }
    }
    geom_link_articulation.assign(model->ngeom, -1);
    geom_link_index.assign(model->ngeom, -1);
    body_is_external.assign(model->nbody, 1);
    body_is_external[0] = 0;
    for (size_t a = 0; a < articulations.size(); a++) {
        const ArticulationRecord &record = articulations[a];
        for (size_t i = 0; i < record.description.links.size(); i++) {
            body_is_external[record.link_body_IDs[i]] = 0;
            const std::string body_name = linkBodyName(record.description.name, record.description.links[i].name);
            for (size_t k = 0; k < record.description.links[i].shapes.size(); k++) {
                const int geom_ID = mj_name2id(model, mjOBJ_GEOM, geomName(body_name, k).c_str());
                if (geom_ID < 0) {
                    helios_runtime_error("ERROR (PhysicsEngine::initialize): Collision shape " + geomName(body_name, k) + " is missing from the compiled model.");
                }
                geom_link_articulation[geom_ID] = static_cast<int>(a);
                geom_link_index[geom_ID] = static_cast<int>(i);
            }
        }
    }
    for (size_t b = 0; b < boxes.size(); b++) {
        std::string name = boxBodyName(static_cast<uint>(b));
        boxes[b].body_ID = mj_name2id(model, mjOBJ_BODY, name.c_str());
        if (boxes[b].body_ID < 0 || model->body_jntnum[boxes[b].body_ID] != 3) {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): Kinematic box " + name + " is missing from the compiled model.");
        }
        int first_joint = model->body_jntadr[boxes[b].body_ID];
        boxes[b].qpos_address = model->jnt_qposadr[first_joint];
        boxes[b].dof_address = model->jnt_dofadr[first_joint];
    }

    bindContextGeometry();
    for (const BodyRestPose &pose: rest_poses) {
        if (std::find(display_body_IDs.begin(), display_body_IDs.end(), pose.body_ID) == display_body_IDs.end()) {
            display_body_IDs.push_back(pose.body_ID);
        }
    }

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
    applyResetPose(instance);
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
    instances[instance].load_detachments = 0;
    applyResetPose(instances[instance]);
    mj_forward(model, instances[instance].data);
}

void PhysicsEngine::applyResetPose(Instance &instance) const {
    if (!reset_qpos.empty()) {
        mju_copy(instance.data->qpos, reset_qpos.data(), model->nq);
    }
}

void PhysicsEngine::setResetPose(uint instance) {
    const mjData *data = getInstance(instance, "setResetPose").data;
    reset_qpos.assign(data->qpos, data->qpos + model->nq);
}

void PhysicsEngine::clearResetPose() {
    requireInitialized("clearResetPose");
    reset_qpos.clear();
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

    for (const ObjectBodyRequest &request: object_body_requests) {
        int body_ID = mj_name2id(model, mjOBJ_BODY, request.body_name.c_str());
        if (body_ID < 0) {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): Object " + std::to_string(request.objID) + " was bound to body " + request.body_name + ", which is not in the compiled model.");
        }
        bindObjectPrimitives(request.objID, addRestPose(body_ID));
    }
}

void PhysicsEngine::attachModelFiles(const PhysicsEngineParameters &params) {
    mjsBody *world = mjs_findBody(spec, "world");
    for (const ModelFileRecord &record: model_files) {
        char error[1024] = "";
        mjSpec *model_file_spec = mj_parseXML(record.filename.c_str(), nullptr, error, sizeof(error));
        if (model_file_spec == nullptr) {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not parse the model file " + record.filename + ": " + std::string(error));
        }
        model_file_specs.push_back(model_file_spec);

        // Geometry of a loaded model collides like a kinematic box: with articulations only (see the contype and conaffinity comment in buildMJCF)
        for (mjsElement *element = mjs_firstElement(model_file_spec, mjOBJ_GEOM); element != nullptr; element = mjs_nextElement(model_file_spec, element)) {
            mjsGeom *geom = mjs_asGeom(element);
            geom->contype = 1;
            geom->conaffinity = 2;
            geom->condim = 3;
            geom->friction[0] = params.friction;
            geom->solref[0] = params.contact_time_constant;
            geom->solref[1] = 1;
        }
        if (record.gravity_compensation) {
            for (mjsElement *element = mjs_firstElement(model_file_spec, mjOBJ_BODY); element != nullptr; element = mjs_nextElement(model_file_spec, element)) {
                mjs_asBody(element)->gravcomp = 1;
            }
        }

        mjsFrame *frame = mjs_addFrame(world, nullptr);
        frame->pos[0] = record.position.x;
        frame->pos[1] = record.position.y;
        frame->pos[2] = record.position.z;
        const double half_yaw = 0.5 * double(deg2rad(record.yaw_degrees));
        frame->quat[0] = std::cos(half_yaw);
        frame->quat[1] = 0;
        frame->quat[2] = 0;
        frame->quat[3] = std::sin(half_yaw);
        if (mjs_attach(frame->element, model_file_spec->element, record.prefix.c_str(), "") == nullptr) {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not add the model file " + record.filename + " to the simulation: " + std::string(mjs_getError(spec)));
        }
    }
}

void PhysicsEngine::addServosAndSensors() {
    for (const ServoRequest &request: servo_requests) {
        mjsElement *joint_element = mjs_findElement(spec, mjOBJ_JOINT, request.joint_name.c_str());
        if (joint_element == nullptr) {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): A servo was added to joint " + request.joint_name + ", which is not in the model. Joint names include the prefix given to addModelFile().");
        }
        mjs_asJoint(joint_element)->armature = request.armature;
        mjsActuator *actuator = mjs_addActuator(spec, nullptr);
        mjs_setName(actuator->element, (request.joint_name + "_servo").c_str());
        actuator->trntype = mjTRN_JOINT;
        mjs_setString(actuator->target, request.joint_name.c_str());
        double kv[1] = {request.damping};
        const char *servo_error = mjs_setToPosition(actuator, request.stiffness, kv, nullptr, nullptr, 0);
        if (servo_error != nullptr && servo_error[0] != '\0') {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not make a servo on joint " + request.joint_name + ": " + std::string(servo_error));
        }
        if (request.force_limit > 0.f) {
            actuator->forcelimited = mjLIMITED_TRUE;
            actuator->forcerange[0] = -request.force_limit;
            actuator->forcerange[1] = request.force_limit;
        }
    }

    for (const SensorRequest &request: sensor_requests) {
        mjsBody *body = mjs_findBody(spec, request.body_name.c_str());
        if (body == nullptr) {
            helios_runtime_error("ERROR (PhysicsEngine::initialize): Force/torque sensor " + request.name + " was added to body " + request.body_name + ", which is not in the model. Body names include the prefix given to addModelFile().");
        }
        mjsSite *site = mjs_addSite(body, nullptr);
        mjs_setName(site->element, request.name.c_str());
        site->pos[0] = request.position.x;
        site->pos[1] = request.position.y;
        site->pos[2] = request.position.z;
        for (const auto &[sensor_type, suffix]: {std::pair<mjtSensor, std::string>{mjSENS_FORCE, "_force"}, std::pair<mjtSensor, std::string>{mjSENS_TORQUE, "_torque"}}) {
            mjsSensor *sensor = mjs_addSensor(spec);
            mjs_setName(sensor->element, (request.name + suffix).c_str());
            sensor->type = sensor_type;
            sensor->objtype = mjOBJ_SITE;
            mjs_setString(sensor->objname, request.name.c_str());
        }
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

int PhysicsEngine::detachEqualityID(uint articulationID, uint link_index, const std::string &method) const {
    requireLink(articulationID, link_index, method);
    const int equality_ID = articulations[articulationID].link_detach_equality_IDs[link_index];
    if (equality_ID < 0) {
        helios_runtime_error("ERROR (PhysicsEngine::" + method + "): Link " + std::to_string(link_index) + " of articulation " + std::to_string(articulationID) + " does not have a detachable joint. Set its joint type to PHYSICS_JOINT_DETACHABLE before initialize().");
    }
    return equality_ID;
}

float PhysicsEngine::equalityLoad(const mjData *data, int equality_ID) const {
    // The first three constraint rows of a weld are its force, and the other three its torque
    for (int row = 0; row < data->nefc; row++) {
        if (data->efc_type[row] == mjCNSTR_EQUALITY && data->efc_id[row] == equality_ID) {
            return float(mju_norm3(data->efc_force + row));
        }
    }
    return 0.f;
}

void PhysicsEngine::detachLink(uint articulationID, uint link_index, uint instance) {
    static_cast<void>(getInstance(instance, "detachLink"));
    const int equality_ID = detachEqualityID(articulationID, link_index, "detachLink");
    instances[instance].data->eq_active[equality_ID] = 0;
}

void PhysicsEngine::setDetachForce(uint articulationID, uint link_index, float force, uint instance) {
    static_cast<void>(getInstance(instance, "setDetachForce"));
    const int equality_ID = detachEqualityID(articulationID, link_index, "setDetachForce");
    if (!(force > 0.f)) {
        helios_runtime_error("ERROR (PhysicsEngine::setDetachForce): The force must be positive, but was " + std::to_string(force) + " N. Use clearDetachForces() to stop releasing links by load.");
    }
    std::vector<DetachForce> &detach_forces = instances[instance].detach_forces;
    for (DetachForce &existing: detach_forces) {
        if (existing.equality_ID == equality_ID) {
            existing.force = force;
            return;
        }
    }
    detach_forces.push_back({equality_ID, force});
}

void PhysicsEngine::clearDetachForces(uint instance) {
    static_cast<void>(getInstance(instance, "clearDetachForces"));
    instances[instance].detach_forces.clear();
}

uint PhysicsEngine::getLoadDetachmentCount(uint instance) const {
    return getInstance(instance, "getLoadDetachmentCount").load_detachments;
}

bool PhysicsEngine::isLinkDetached(uint articulationID, uint link_index, uint instance) const {
    const mjData *data = getInstance(instance, "isLinkDetached").data;
    const int equality_ID = detachEqualityID(articulationID, link_index, "isLinkDetached");
    return data->eq_active[equality_ID] == 0;
}

float PhysicsEngine::getDetachableLinkLoad(uint articulationID, uint link_index, uint instance) const {
    const mjData *data = getInstance(instance, "getDetachableLinkLoad").data;
    const int equality_ID = detachEqualityID(articulationID, link_index, "getDetachableLinkLoad");
    return equalityLoad(data, equality_ID);
}

void PhysicsEngine::splitTubesAtLink(uint articulationID, uint link_index) {
    requireInitialized("splitTubesAtLink");
    requireLink(articulationID, link_index, "splitTubesAtLink");
    const ArticulationRecord &record = articulations[articulationID];
    const PhysicsLink &link = record.description.links[link_index];
    if (link.parent_index < 0) {
        helios_runtime_error("ERROR (PhysicsEngine::splitTubesAtLink): Link " + std::to_string(link_index) + " of articulation " + std::to_string(articulationID) + " has no parent to part from.");
    }
    const size_t link_rest_pose = record.link_rest_pose_indices[link_index];

    const PhysicsLink &parent = record.description.links[link.parent_index];
    mjtNum R_parent[9];
    quaternionToMatrix(parent.qw, parent.qx, parent.qy, parent.qz, R_parent);
    const vec3 anchor_world = parent.position + rotateToWorldFrame(R_parent, link.anchor_in_parent);

    // Splitting appends to the list, and appended tubes begin at the link, so they need no further splitting
    const size_t tube_count = tube_bindings.size();
    for (size_t t = 0; t < tube_count; t++) {
        const std::vector<size_t> &node_rest_poses = tube_bindings[t].node_rest_pose_index;
        auto first_on_link = std::find(node_rest_poses.begin(), node_rest_poses.end(), link_rest_pose);
        if (first_on_link == node_rest_poses.end() || first_on_link == node_rest_poses.begin()) {
            continue;
        }
        const size_t first_node = static_cast<size_t>(first_on_link - node_rest_poses.begin());

        auto nodeRestPosition = [&](size_t node) {
            const BodyRestPose &pose = rest_poses[node_rest_poses[node]];
            mjtNum R[9];
            for (int k = 0; k < 9; k++) {
                R[k] = pose.rotation[k];
            }
            return rotateToWorldFrame(R, tube_bindings[t].node_local_position[node]) + pose.position;
        };

        // The tube parts at the node where the link meets its parent. That is the first node following the link or the one
        // before it, depending on which link a node at the junction of two links was bound to.
        const size_t junction_node = (nodeRestPosition(first_node) - anchor_world).magnitude() < (nodeRestPosition(first_node - 1) - anchor_world).magnitude() ? first_node : first_node - 1;
        const vec3 junction_rest = nodeRestPosition(junction_node);
        const BodyRestPose &link_pose = rest_poses[link_rest_pose];
        mjtNum R_link[9];
        for (int k = 0; k < 9; k++) {
            R_link[k] = link_pose.rotation[k];
        }
        const vec3 junction_local = rotateToBodyFrame(R_link, junction_rest - link_pose.position);
        const size_t parent_rest_pose = record.link_rest_pose_indices[link.parent_index];
        const BodyRestPose &parent_pose = rest_poses[parent_rest_pose];
        mjtNum R_parent_rest[9];
        for (int k = 0; k < 9; k++) {
            R_parent_rest[k] = parent_pose.rotation[k];
        }
        const vec3 junction_local_to_parent = rotateToBodyFrame(R_parent_rest, junction_rest - parent_pose.position);

        TubeBinding &tube = tube_bindings[t];
        if (junction_node == 0) {
            // The tube begins at the junction, so there is nothing to leave behind: the whole tube follows the link
            tube.node_rest_pose_index[0] = link_rest_pose;
            tube.node_local_position[0] = junction_local;
            continue;
        }

        TubeBinding upper;
        upper.objID = context->splitTubeObject(tube.objID, static_cast<uint>(junction_node + 1));
        upper.node_rest_pose_index.assign(tube.node_rest_pose_index.begin() + static_cast<std::ptrdiff_t>(junction_node), tube.node_rest_pose_index.end());
        upper.node_local_position.assign(tube.node_local_position.begin() + static_cast<std::ptrdiff_t>(junction_node), tube.node_local_position.end());
        upper.node_rest_pose_index[0] = link_rest_pose;
        upper.node_local_position[0] = junction_local;
        // The junction ends the piece left behind, so there it stays with the parent whichever link it was bound to
        tube.node_rest_pose_index.resize(junction_node + 1);
        tube.node_local_position.resize(junction_node + 1);
        tube.node_rest_pose_index[junction_node] = parent_rest_pose;
        tube.node_local_position[junction_node] = junction_local_to_parent;
        tube_bindings.push_back(std::move(upper));
    }

    // The nodes have new bindings, whether or not their bodies move
    synced_transforms.clear();
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
        for (const DetachForce &detach: instance.detach_forces) {
            if (instance.data->eq_active[detach.equality_ID] && equalityLoad(instance.data, detach.equality_ID) > detach.force) {
                instance.data->eq_active[detach.equality_ID] = 0;
                instance.load_detachments++;
            }
        }
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

    // Bodies that have not moved since the last update leave their geometry alone, so that its dirty flags in the Context mean something
    const bool first_update = synced_transforms.size() != rest_to_current.size();
    std::vector<char> moved(rest_poses.size(), 1);
    if (first_update) {
        synced_transforms = rest_to_current;
    } else {
        for (size_t i = 0; i < rest_poses.size(); i++) {
            bool body_moved = false;
            for (size_t k = 0; k < 12 && !body_moved; k++) {
                // Inverting the test also counts a NaN as a change
                body_moved = !(std::abs(rest_to_current[i][k] - synced_transforms[i][k]) <= context_update_tolerance);
            }
            moved[i] = body_moved;
            // A body within tolerance keeps the transform it is shown at, so that slow drift adds up until it does move
            if (body_moved) {
                synced_transforms[i] = rest_to_current[i];
            } else {
                rest_to_current[i] = synced_transforms[i];
            }
        }
    }

    for (const PrimitiveBinding &binding: primitive_bindings) {
        if (!moved[binding.rest_pose_index]) {
            continue;
        }
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
        bool tube_moved = false;
        for (size_t rest_pose_index: tube.node_rest_pose_index) {
            tube_moved = tube_moved || moved[rest_pose_index];
        }
        if (!tube_moved) {
            continue;
        }
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

uint PhysicsEngine::getDisplayBodyCount() const {
    requireInitialized("getDisplayBodyCount");
    return static_cast<uint>(display_body_IDs.size());
}

std::vector<std::string> PhysicsEngine::getDisplayBodyNames() const {
    requireInitialized("getDisplayBodyNames");
    std::vector<std::string> names;
    for (int body_ID: display_body_IDs) {
        const char *name = mj_id2name(model, mjOBJ_BODY, body_ID);
        names.emplace_back(name != nullptr ? name : "");
    }
    return names;
}

void PhysicsEngine::getDisplayBodyPoses(float *poses, uint instance) const {
    const mjData *data = getInstance(instance, "getDisplayBodyPoses").data;
    for (size_t i = 0; i < display_body_IDs.size(); i++) {
        const int body_ID = display_body_IDs[i];
        for (int k = 0; k < 3; k++) {
            poses[7 * i + k] = float(data->xpos[3 * body_ID + k]);
        }
        for (int k = 0; k < 4; k++) {
            poses[7 * i + 3 + k] = float(data->xquat[4 * body_ID + k]);
        }
    }
}

void PhysicsEngine::showDisplayBodyPoses(const float *poses, uint instance) {
    mjData *data = getInstance(instance, "showDisplayBodyPoses").data;
    for (size_t i = 0; i < display_body_IDs.size(); i++) {
        const int body_ID = display_body_IDs[i];
        mjtNum quaternion[4] = {poses[7 * i + 3], poses[7 * i + 4], poses[7 * i + 5], poses[7 * i + 6]};
        if (mju_normalize4(quaternion) < 1e-9) {
            helios_runtime_error("ERROR (PhysicsEngine::showDisplayBodyPoses): The orientation of body " + std::to_string(i) + " is a zero quaternion.");
        }
        for (int k = 0; k < 3; k++) {
            data->xpos[3 * body_ID + k] = poses[7 * i + k];
        }
        mju_copy4(data->xquat + 4 * body_ID, quaternion);
        mju_quat2Mat(data->xmat + 9 * body_ID, quaternion);
    }
    updateContextGeometry(instance);
}

void PhysicsEngine::setContextUpdateTolerance(float tolerance) {
    if (tolerance < 0.f) {
        helios_runtime_error("ERROR (PhysicsEngine::setContextUpdateTolerance): The tolerance must not be negative.");
    }
    context_update_tolerance = tolerance;
}

void PhysicsEngine::writeMJCF(const std::string &filename) const {
    requireInitialized("writeMJCF");
    std::ofstream file(filename);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (PhysicsEngine::writeMJCF): Could not open " + filename + " for writing. Make sure the directory exists and is writable.");
    }
    if (model_files.empty() && servo_requests.empty() && sensor_requests.empty() && excluded_initial_overlaps == 0) {
        file << mjcf_text;
        return;
    }
    // Loaded models and the exclusions of links that overlap as built are only in the spec, not in the generated text. MuJoCo returns the size needed when the buffer is too small.
    char error[1024] = "";
    std::string xml(1 << 20, '\0');
    int required_size = mj_saveXMLString(spec, xml.data(), static_cast<int>(xml.size()), error, sizeof(error));
    if (required_size > 0) {
        xml.resize(required_size + 1);
        required_size = mj_saveXMLString(spec, xml.data(), static_cast<int>(xml.size()), error, sizeof(error));
    }
    if (required_size != 0) {
        helios_runtime_error("ERROR (PhysicsEngine::writeMJCF): MuJoCo could not write the model: " + std::string(error));
    }
    file << xml.c_str();
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

uint PhysicsEngine::getArticulationCount() const {
    return static_cast<uint>(articulations.size());
}

uint PhysicsEngine::getArticulationLinkCount(uint articulationID) const {
    if (articulationID >= articulations.size()) {
        helios_runtime_error("ERROR (PhysicsEngine::getArticulationLinkCount): Articulation " + std::to_string(articulationID) + " does not exist.");
    }
    return static_cast<uint>(articulations[articulationID].description.links.size());
}

std::string PhysicsEngine::getLinkLabel(uint articulationID, uint link_index) const {
    requireLink(articulationID, link_index, "getLinkLabel");
    return articulations[articulationID].description.links[link_index].label;
}

int PhysicsEngine::getLinkBodyID(uint articulationID, uint link_index) const {
    requireInitialized("getLinkBodyID");
    requireLink(articulationID, link_index, "getLinkBodyID");
    return articulations[articulationID].link_body_IDs[link_index];
}

bool PhysicsEngine::hasOwnBody(uint articulationID, uint link_index) const {
    requireInitialized("hasOwnBody");
    requireLink(articulationID, link_index, "hasOwnBody");
    return articulations[articulationID].link_host[link_index] == static_cast<int>(link_index);
}

bool PhysicsEngine::getGeomLink(int geom_ID, uint &articulationID, uint &link_index) const {
    requireInitialized("getGeomLink");
    if (geom_ID < 0 || geom_ID >= model->ngeom) {
        helios_runtime_error("ERROR (PhysicsEngine::getGeomLink): Geom " + std::to_string(geom_ID) + " does not exist; the model has " + std::to_string(model->ngeom) + " geoms.");
    }
    if (geom_link_articulation[geom_ID] < 0) {
        return false;
    }
    articulationID = static_cast<uint>(geom_link_articulation[geom_ID]);
    link_index = static_cast<uint>(geom_link_index[geom_ID]);
    return true;
}

vec3 PhysicsEngine::linkWorldPosition(const mjData *data, const ArticulationRecord &record, size_t link_index) const {
    const int body_ID = record.link_body_IDs[link_index];
    const vec3 &offset = record.link_offset[link_index];
    const mjtNum local[3] = {offset.x, offset.y, offset.z};
    mjtNum world[3];
    mju_mulMatVec3(world, data->xmat + 9 * body_ID, local);
    mju_addTo3(world, data->xpos + 3 * body_ID);
    return make_vec3(float(world[0]), float(world[1]), float(world[2]));
}

uint PhysicsEngine::getContactCount(uint instance) const {
    return static_cast<uint>(getInstance(instance, "getContactCount").data->ncon);
}

float PhysicsEngine::getMaxLinkDisplacement(uint instance) const {
    const mjData *data = getInstance(instance, "getMaxLinkDisplacement").data;
    float max_displacement = 0.f;
    for (const ArticulationRecord &record: articulations) {
        for (size_t i = 0; i < record.description.links.size(); i++) {
            const vec3 displacement = linkWorldPosition(data, record, i) - record.description.links[i].position;
            max_displacement = std::max(max_displacement, displacement.magnitude());
        }
    }
    return max_displacement;
}

float PhysicsEngine::getMaxLinkSpeed(uint instance) const {
    const mjData *data = getInstance(instance, "getMaxLinkSpeed").data;
    mjtNum max_speed = 0;
    for (const ArticulationRecord &record: articulations) {
        for (size_t i = 0; i < record.description.links.size(); i++) {
            const int body_ID = record.link_body_IDs[i];
            mjtNum velocity[6]; // rotational then translational velocity of the body origin, in the world frame
            mj_objectVelocity(model, data, mjOBJ_BODY, body_ID, velocity, 0);
            // A link merged into a body moves with it, so its velocity adds the body's rotation about the body origin
            const vec3 link_position = linkWorldPosition(data, record, i);
            const mjtNum arm[3] = {link_position.x - data->xpos[3 * body_ID], link_position.y - data->xpos[3 * body_ID + 1], link_position.z - data->xpos[3 * body_ID + 2]};
            mjtNum turning[3];
            mju_cross(turning, velocity, arm);
            mju_addTo3(turning, velocity + 3);
            max_speed = std::max(max_speed, mju_norm3(turning));
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
    return !body_is_external[model->geom_bodyid[contact.geom[0]]] && !body_is_external[model->geom_bodyid[contact.geom[1]]];
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

// Exclude from collisions every pair of articulation links that overlap in the compiled model as built, and compile it again
void PhysicsEngine::excludeInitialLinkOverlaps() {
    // The bodies that carry links: their own, those they were merged into, and the world for links welded to it
    std::unordered_map<int, std::string> link_body_names;
    for (const ArticulationRecord &record: articulations) {
        for (size_t i = 0; i < record.description.links.size(); i++) {
            const std::string name = hostBodyName(record, i);
            const int body_ID = mj_name2id(model, mjOBJ_BODY, name.c_str());
            if (body_ID < 0) {
                helios_runtime_error("ERROR (PhysicsEngine::initialize): Body " + name + " is missing from the compiled model.");
            }
            link_body_names[body_ID] = name;
        }
    }
    // Only the shapes of links count: the world body also carries the ground
    std::unordered_set<int> link_geom_IDs;
    for (const ArticulationRecord &record: articulations) {
        for (const PhysicsLink &link: record.description.links) {
            for (size_t k = 0; k < link.shapes.size(); k++) {
                link_geom_IDs.insert(mj_name2id(model, mjOBJ_GEOM, geomName(linkBodyName(record.description.name, link.name), k).c_str()));
            }
        }
    }

    mjData *data = mj_makeData(model);
    if (data == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not allocate the data for finding links that overlap as built.");
    }
    mj_forward(model, data);
    std::set<std::pair<int, int>> overlapping;
    for (int c = 0; c < data->ncon; c++) {
        const mjContact &contact = data->contact[c];
        // A contact within the collision margin is between links that do not touch yet; they keep colliding
        if (contact.dist >= 0) {
            continue;
        }
        const int body_0 = model->geom_bodyid[contact.geom[0]];
        const int body_1 = model->geom_bodyid[contact.geom[1]];
        if (body_0 != body_1 && link_geom_IDs.count(contact.geom[0]) > 0 && link_geom_IDs.count(contact.geom[1]) > 0) {
            overlapping.emplace(std::min(body_0, body_1), std::max(body_0, body_1));
        }
    }
    mj_deleteData(data);

    excluded_initial_overlaps = static_cast<uint>(overlapping.size());
    if (overlapping.empty()) {
        return;
    }
    for (const auto &[body_0, body_1]: overlapping) {
        mjsExclude *exclusion = mjs_addExclude(spec);
        mjs_setString(exclusion->bodyname1, link_body_names.at(body_0).c_str());
        mjs_setString(exclusion->bodyname2, link_body_names.at(body_1).c_str());
    }
    mj_deleteModel(model);
    model = mj_compile(spec, nullptr);
    if (model == nullptr) {
        helios_runtime_error("ERROR (PhysicsEngine::initialize): MuJoCo could not compile the model after excluding links that overlap as built: " + std::string(mjs_getError(spec)));
    }
}

uint PhysicsEngine::getExcludedInitialOverlapCount() const {
    requireInitialized("getExcludedInitialOverlapCount");
    return excluded_initial_overlaps;
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

vec3 PhysicsEngine::getExternalContactForce(uint articulationID, uint instance) const {
    const mjData *data = getInstance(instance, "getExternalContactForce").data;
    if (articulationID >= articulations.size()) {
        helios_runtime_error("ERROR (PhysicsEngine::getExternalContactForce): Articulation " + std::to_string(articulationID) + " does not exist.");
    }
    mjtNum total[3] = {0, 0, 0};
    for (int c = 0; c < data->ncon; c++) {
        const mjContact &contact = data->contact[c];
        if (contact.efc_address < 0) {
            continue;
        }
        const int body_0 = model->geom_bodyid[contact.geom[0]];
        const int body_1 = model->geom_bodyid[contact.geom[1]];
        // MuJoCo gives the force that geom 0 exerts on geom 1, with the contact normal as the first axis of the contact frame
        mjtNum sign = 0;
        if (geom_link_articulation[contact.geom[1]] == static_cast<int>(articulationID) && body_is_external[body_0]) {
            sign = 1;
        } else if (geom_link_articulation[contact.geom[0]] == static_cast<int>(articulationID) && body_is_external[body_1]) {
            sign = -1;
        } else {
            continue;
        }
        mjtNum contact_force[6];
        mj_contactForce(model, data, c, contact_force);
        mjtNum world_force[3];
        mju_mulMatTVec3(world_force, contact.frame, contact_force);
        for (int k = 0; k < 3; k++) {
            total[k] += sign * world_force[k];
        }
    }
    return make_vec3(float(total[0]), float(total[1]), float(total[2]));
}

vec3 PhysicsEngine::getLinkPosition(uint articulationID, uint link_index, uint instance) const {
    const mjData *data = getInstance(instance, "getLinkPosition").data;
    requireLink(articulationID, link_index, "getLinkPosition");
    return linkWorldPosition(data, articulations[articulationID], link_index);
}
