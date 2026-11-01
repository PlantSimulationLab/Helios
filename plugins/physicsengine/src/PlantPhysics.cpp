/** \file "PlantPhysics.cpp" Adds plants from the plant architecture plug-in to the physics engine.

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

#include "PlantPhysics.h"

#include <mujoco/mujoco.h>

using namespace helios;

namespace {

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

    // Distance from a point to the axis segment of a plant link (the link origin itself for organ links)
    float distanceToLink(const PlantArticulationLink &link, const vec3 &point) {
        mjtNum R[9];
        quaternionToMatrix(link.qw, link.qx, link.qy, link.qz, R);
        vec3 axis = make_vec3(float(R[2]), float(R[5]), float(R[8]));
        vec3 start = link.position - axis * link.half_length;
        float segment_length = 2.f * link.half_length;
        float t = segment_length > 0.f ? std::clamp((point - start) * axis, 0.f, segment_length) : 0.f;
        return (point - (start + axis * t)).magnitude();
    }

    uint findNearestLink(const std::vector<PlantArticulationLink> &links, const vec3 &point) {
        uint nearest_link = 0;
        float nearest_distance = std::numeric_limits<float>::max();
        for (uint i = 0; i < links.size(); i++) {
            float distance = distanceToLink(links[i], point);
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest_link = i;
            }
        }
        return nearest_link;
    }

    PhysicsCollisionShape makeShape(PhysicsShapeType type, const RGBcolor &color) {
        PhysicsCollisionShape shape;
        shape.type = type;
        shape.color = color;
        return shape;
    }

    // Fit the collision shapes of a fruit, leaf, or flower to its vertices, expressed in the link frame
    void addOrganShapes(PhysicsLink &physics_link, const PlantArticulationLink &link, const Context &context, const PlantPhysicsParameters &params) {
        mjtNum R_link[9];
        quaternionToMatrix(link.qw, link.qx, link.qy, link.qz, R_link);

        std::vector<vec3> vertices;
        for (uint UUID: context.getObjectPrimitiveUUIDs(link.source_objID)) {
            for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                vertices.push_back(rotateToBodyFrame(R_link, vertex - link.position));
            }
        }

        if (link.type == ARTICULATION_LINK_FRUIT) {
            vec3 lower = vertices.front();
            vec3 upper = vertices.front();
            for (const vec3 &v: vertices) {
                lower = make_vec3(std::min(lower.x, v.x), std::min(lower.y, v.y), std::min(lower.z, v.z));
                upper = make_vec3(std::max(upper.x, v.x), std::max(upper.y, v.y), std::max(upper.z, v.z));
            }
            vec3 extent = upper - lower;
            PhysicsCollisionShape sphere = makeShape(PHYSICS_SHAPE_SPHERE, make_RGBcolor(0.75f, 0.1f, 0.1f));
            sphere.size.x = (extent.x + extent.y + extent.z) / 6.f;
            sphere.position = 0.5f * (lower + upper);
            physics_link.shapes.push_back(sphere);
        } else if (params.leaf_collision_shape == "triangles") {
            // A thin prism for every triangle of the blade, so that the collision shape follows a folded or curved blade
            const float half_thickness = 0.5f * params.leaf_collision_thickness;
            for (uint UUID: context.getObjectPrimitiveUUIDs(link.source_objID)) {
                std::vector<vec3> corners;
                for (const vec3 &vertex: context.getPrimitiveVertices(UUID)) {
                    corners.push_back(rotateToBodyFrame(R_link, vertex - link.position));
                }
                for (size_t k = 1; k + 1 < corners.size(); k++) {
                    vec3 normal = cross(corners[k] - corners[0], corners[k + 1] - corners[0]);
                    if (normal.magnitude() < 1e-12f) {
                        continue;
                    }
                    normal.normalize();
                    PhysicsCollisionShape prism = makeShape(PHYSICS_SHAPE_CONVEX_MESH, make_RGBcolor(0.3f, 0.6f, 0.2f));
                    for (const vec3 &corner: {corners[0], corners[k], corners[k + 1]}) {
                        for (float side: {-1.f, 1.f}) {
                            prism.mesh_vertices.push_back(corner + side * half_thickness * normal);
                        }
                    }
                    physics_link.shapes.push_back(std::move(prism));
                }
            }
        } else {
            // Thin box or ellipsoid along the principal axes of the leaf blade
            mjtNum covariance[9] = {0};
            vec3 mean = make_vec3(0, 0, 0);
            for (const vec3 &v: vertices) {
                mean = mean + v;
            }
            mean = mean / float(vertices.size());
            for (const vec3 &v: vertices) {
                vec3 d = v - mean;
                mjtNum dv[3] = {d.x, d.y, d.z};
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        covariance[3 * r + c] += dv[r] * dv[c];
                    }
                }
            }
            mjtNum eigenvalues[3], eigenvectors[9], principal_quat[4];
            mju_eig3(eigenvalues, eigenvectors, principal_quat, covariance);
            mjtNum R_principal[9];
            mju_quat2Mat(R_principal, principal_quat);

            vec3 lower = make_vec3(1e30f, 1e30f, 1e30f);
            vec3 upper = make_vec3(-1e30f, -1e30f, -1e30f);
            for (const vec3 &v: vertices) {
                vec3 p = rotateToBodyFrame(R_principal, v);
                lower = make_vec3(std::min(lower.x, p.x), std::min(lower.y, p.y), std::min(lower.z, p.z));
                upper = make_vec3(std::max(upper.x, p.x), std::max(upper.y, p.y), std::max(upper.z, p.z));
            }
            vec3 half_size = 0.5f * (upper - lower);
            // mju_eig3 sorts eigenvalues in decreasing order, so the third principal axis is the leaf normal
            half_size.z = 0.5f * params.leaf_collision_thickness;
            half_size.x = std::max(half_size.x, half_size.z);
            half_size.y = std::max(half_size.y, half_size.z);

            PhysicsCollisionShape blade = makeShape(params.leaf_collision_shape == "box" ? PHYSICS_SHAPE_BOX : PHYSICS_SHAPE_ELLIPSOID, make_RGBcolor(0.3f, 0.6f, 0.2f));
            blade.size = half_size;
            blade.position = rotateToWorldFrame(R_principal, 0.5f * (lower + upper));
            for (int k = 0; k < 4; k++) {
                blade.orientation[k] = principal_quat[k];
            }
            physics_link.shapes.push_back(blade);
        }
    }

    // Stem, petiole, and peduncle segments meeting at or near the same node overlap there by construction: a branch
    // starts inside the stem it grows from, and short segments overlap their neighbors. Those pairs are excluded when
    // they are within a few links of each other. Leaves, fruit, and flowers are excluded only from the link they hang
    // from, so they are kept out of the stems and of each other. Each pair with the parent is listed explicitly
    // because MuJoCo does not filter contacts with a parent that is fixed to the world.
    std::vector<std::pair<uint, uint>> findContactExclusions(const std::vector<PlantArticulationLink> &links) {
        constexpr int axial_exclusion_links = 4;

        std::vector<std::vector<size_t>> children(links.size());
        for (size_t i = 0; i < links.size(); i++) {
            if (links[i].parent_index >= 0) {
                children.at(links[i].parent_index).push_back(i);
            }
        }
        auto isAxial = [&](size_t i) { return links[i].type == ARTICULATION_LINK_INTERNODE || links[i].type == ARTICULATION_LINK_PETIOLE || links[i].type == ARTICULATION_LINK_PEDUNCLE; };

        std::vector<std::pair<uint, uint>> exclusions;
        std::vector<int> distance(links.size(), -1);
        std::vector<size_t> visited;
        for (size_t start = 0; start < links.size(); start++) {
            if (links[start].parent_index >= 0) {
                exclusions.emplace_back(static_cast<uint>(links[start].parent_index), static_cast<uint>(start));
            }
            if (!isAxial(start)) {
                continue;
            }
            // Breadth-first walk through the axial links within reach of this one, along the tree in both directions
            distance[start] = 0;
            visited.assign(1, start);
            for (size_t v = 0; v < visited.size(); v++) {
                size_t link = visited[v];
                if (distance[link] == axial_exclusion_links) {
                    continue;
                }
                std::vector<size_t> neighbors = children[link];
                if (links[link].parent_index >= 0) {
                    neighbors.push_back(static_cast<size_t>(links[link].parent_index));
                }
                for (size_t neighbor: neighbors) {
                    if (distance[neighbor] < 0 && isAxial(neighbor)) {
                        distance[neighbor] = distance[link] + 1;
                        visited.push_back(neighbor);
                        if (neighbor > start && distance[neighbor] > 1) {
                            exclusions.emplace_back(static_cast<uint>(start), static_cast<uint>(neighbor));
                        }
                    }
                }
            }
            for (size_t link: visited) {
                distance[link] = -1;
            }
        }
        return exclusions;
    }

} // namespace

uint addPlantToPhysicsEngine(PhysicsEngine &engine, const Context &context, const PlantArchitecture &plantarchitecture, uint plantID, const USDExportParameters &plant_physics, const PlantPhysicsParameters &params) {

    if (params.leaf_collision_shape != "box" && params.leaf_collision_shape != "ellipsoid" && params.leaf_collision_shape != "triangles") {
        helios_runtime_error("ERROR (addPlantToPhysicsEngine): Unknown leaf collision shape \"" + params.leaf_collision_shape + "\". Use \"box\", \"ellipsoid\", or \"triangles\".");
    }

    const PlantArticulation plant = plantarchitecture.getPlantArticulation(plantID, plant_physics);
    const std::vector<PlantArticulationLink> &links = plant.links;

    std::vector<int> joint_of_child(links.size(), -1);
    for (size_t j = 0; j < plant.joints.size(); j++) {
        joint_of_child.at(plant.joints[j].child_index) = static_cast<int>(j);
    }

    std::vector<char> is_flexible(links.size(), 0);
    for (uint link_index: params.flexible_links) {
        if (link_index >= links.size() || links[link_index].type != ARTICULATION_LINK_INTERNODE) {
            helios_runtime_error("ERROR (addPlantToPhysicsEngine): Flexible link " + std::to_string(link_index) + " of plant " + std::to_string(plantID) + " does not exist or is not an internode.");
        }
        is_flexible[link_index] = 1;
    }

    PhysicsArticulation articulation;
    articulation.name = "p" + std::to_string(plantID);
    articulation.links.resize(links.size());
    for (size_t i = 0; i < links.size(); i++) {
        const PlantArticulationLink &link = links[i];
        PhysicsLink &physics_link = articulation.links[i];
        physics_link.name = link.name;
        physics_link.parent_index = link.parent_index;
        physics_link.position = link.position;
        physics_link.qw = link.qw;
        physics_link.qx = link.qx;
        physics_link.qy = link.qy;
        physics_link.qz = link.qz;
        physics_link.mass = link.mass;
        physics_link.inertia_diagonal = link.inertia_diagonal;

        if (link.parent_index >= 0 && joint_of_child[i] < 0) {
            helios_runtime_error("ERROR (addPlantToPhysicsEngine): Link " + link.name + " of plant " + std::to_string(plantID) + " has a parent but no joint.");
        }
        if (link.parent_index >= 0) {
            physics_link.anchor_in_parent = plant.joints.at(joint_of_child[i]).anchor_in_parent;
        }

        // A link without a joint is welded to its parent
        bool welded = (params.rigid_internodes && link.type == ARTICULATION_LINK_INTERNODE && !is_flexible[i]) || (params.rigid_leaves && (link.type == ARTICULATION_LINK_PETIOLE || link.type == ARTICULATION_LINK_LEAF));
        if (link.parent_index >= 0 && !welded) {
            const PlantArticulationJoint &joint = plant.joints.at(joint_of_child[i]);
            physics_link.joint_type = PHYSICS_JOINT_BALL;
            physics_link.stiffness = joint.stiffness;
            physics_link.damping = joint.damping;
            physics_link.armature = params.apply_joint_armature ? joint.armature : 0.f;
            physics_link.max_rotation = params.max_joint_bend_degrees * float(M_PI) / 180.f;
        }

        if (link.type == ARTICULATION_LINK_INTERNODE || link.type == ARTICULATION_LINK_PETIOLE || link.type == ARTICULATION_LINK_PEDUNCLE) {
            PhysicsCollisionShape capsule = makeShape(PHYSICS_SHAPE_CAPSULE, link.type == ARTICULATION_LINK_INTERNODE ? make_RGBcolor(0.45f, 0.32f, 0.2f) : make_RGBcolor(0.4f, 0.6f, 0.25f));
            capsule.size.x = link.radius;
            capsule.size.y = link.half_length;
            physics_link.shapes.push_back(capsule);
        } else if (link.source_objID != 0 && context.doesObjectExist(link.source_objID) && (link.type == ARTICULATION_LINK_FRUIT || (link.type == ARTICULATION_LINK_LEAF && params.leaf_collisions))) {
            addOrganShapes(physics_link, link, context, params);
        }
    }
    articulation.contact_exclusions = findContactExclusions(links);

    for (uint link_index: params.detachable_links) {
        if (link_index >= links.size() || links[link_index].parent_index < 0) {
            helios_runtime_error("ERROR (addPlantToPhysicsEngine): Detachable link " + std::to_string(link_index) + " of plant " + std::to_string(plantID) + " does not exist or has no parent to detach from.");
        }
        articulation.links[link_index].joint_type = PHYSICS_JOINT_DETACHABLE;
    }

    uint articulationID = engine.addArticulation(articulation);

    std::map<uint, std::vector<uint>> links_by_object;
    for (uint i = 0; i < links.size(); i++) {
        if (links[i].source_objID != 0) {
            links_by_object[links[i].source_objID].push_back(i);
        }
    }

    // Every object of the plant follows a link: the link built from it when there is one, otherwise the nearest link
    for (uint objID: plantarchitecture.getAllPlantObjectIDs(plantID)) {
        if (!context.doesObjectExist(objID)) {
            continue;
        }
        auto bound_links = links_by_object.find(objID);

        if (context.getObjectType(objID) == OBJECT_TYPE_TUBE) {
            std::vector<uint> node_link_indices;
            for (const vec3 &node: context.getTubeObjectNodes(objID)) {
                uint link_index;
                if (bound_links != links_by_object.end()) {
                    link_index = bound_links->second.front();
                    float nearest = distanceToLink(links[link_index], node);
                    for (uint candidate: bound_links->second) {
                        float distance = distanceToLink(links[candidate], node);
                        if (distance < nearest) {
                            nearest = distance;
                            link_index = candidate;
                        }
                    }
                } else {
                    link_index = findNearestLink(links, node);
                }
                node_link_indices.push_back(link_index);
            }
            engine.bindTubeNodesToLinks(objID, articulationID, node_link_indices);
        } else {
            uint link_index = bound_links != links_by_object.end() ? bound_links->second.front() : findNearestLink(links, context.getObjectCenter(objID));
            engine.bindObjectToLink(objID, articulationID, link_index);
        }
    }

    return articulationID;
}
