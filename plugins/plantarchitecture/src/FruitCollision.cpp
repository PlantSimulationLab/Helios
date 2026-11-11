/** \file "FruitCollision.cpp" Fruit-branch collision avoidance: keeps growing fruit out of the internodes of their own plant by swinging their peduncles about the node.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.0 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
*/

#include "CollisionGeometry.h"
#include "PlantArchitecture.h"
#include "PlantArchitectureInternal.h"

using namespace helios;

namespace {

    // ---- Rotations ---- //
    // The small rotations of one swing are composed as unit quaternions and stored on the floral bud as a single axis and angle.

    struct Quaternion {
        float w = 1;
        float x = 0;
        float y = 0;
        float z = 0;
    };

    Quaternion quaternionFromAxisAngle(const vec3 &axis, float angle) {
        vec3 unit_axis = axis;
        unit_axis.normalize();
        const float half_sine = std::sin(0.5f * angle);
        return {std::cos(0.5f * angle), unit_axis.x * half_sine, unit_axis.y * half_sine, unit_axis.z * half_sine};
    }

    //! Rotation that applies \a second first and \a first after it
    Quaternion quaternionProduct(const Quaternion &first, const Quaternion &second) {
        return {first.w * second.w - first.x * second.x - first.y * second.y - first.z * second.z, first.w * second.x + first.x * second.w + first.y * second.z - first.z * second.y,
                first.w * second.y - first.x * second.z + first.y * second.w + first.z * second.x, first.w * second.z + first.x * second.y - first.y * second.x + first.z * second.w};
    }

    //! Axis and angle (0 to pi) of a quaternion; the axis is undefined for a zero rotation, so \a fallback_axis is returned then
    void quaternionToAxisAngle(Quaternion q, const vec3 &fallback_axis, vec3 &axis, float &angle) {
        const float norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        if (norm < 1e-12f) {
            axis = fallback_axis;
            angle = 0.f;
            return;
        }
        q = {q.w / norm, q.x / norm, q.y / norm, q.z / norm};
        if (q.w < 0.f) {
            // q and -q are the same rotation; the one with w >= 0 has the angle in [0, pi]
            q = {-q.w, -q.x, -q.y, -q.z};
        }
        // The sine of the half angle is taken from the vector part directly, and the angle from atan2 of it and the cosine. Recovering either from w alone (1 - w^2, acos(w)) loses everything below about
        // 1e-3 rad in single precision, because w rounds to 1: small swings would be recorded as zero, and a nearly settled one would never reach zero.
        const float sine_half = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        if (sine_half < 1e-12f) {
            axis = fallback_axis;
            angle = 0.f;
            return;
        }
        angle = 2.f * std::atan2(sine_half, q.w);
        axis = make_vec3(q.x / sine_half, q.y / sine_half, q.z / sine_half);
    }

    // ---- Branch capsules ---- //

    using fruit_contact::BranchSegment;
    using fruit_contact::Contact;
    using FruitWorldProxy = fruit_contact::WorldProxy;

    //! Radius of a branch segment at the point of its axis nearest \a on_axis
    float branchRadiusAt(const BranchSegment &segment, const vec3 &on_axis) {
        const vec3 axis = segment.end - segment.start;
        const float length_squared = axis * axis;
        const float t = (length_squared > 0.f) ? clamp(((on_axis - segment.start) * axis) / length_squared, 0.f, 1.f) : 0.f;
        return segment.start_radius + t * (segment.end_radius - segment.start_radius);
    }

    //! Distance from a point to a branch segment's axis, with the nearest axis point and the branch radius there
    float branchAxisDistance(const vec3 &point, const BranchSegment &segment, vec3 &on_axis, float &radius_at) {
        vec3 on_point;
        const float distance_squared = collision::closestPointsOnSegments(point, point, segment.start, segment.end, on_point, on_axis);
        radius_at = branchRadiusAt(segment, on_axis);
        return std::sqrt(distance_squared);
    }

    //! Number of internode segments of a phytomer: its own tube nodes, preceded (after the first phytomer) by the last node of the phytomer below, as Phytomer::getInternodeNodePositions() lists them
    size_t internodeSegmentCount(const Shoot &shoot, uint node) {
        const size_t own_nodes = shoot.shoot_internode_vertices.at(node).size();
        if (node == 0) {
            return own_nodes > 0 ? own_nodes - 1 : 0;
        }
        return own_nodes;
    }

    //! Segment \a s of a phytomer's internode, without copying the node lists
    BranchSegment internodeSegment(const Shoot &shoot, uint node, uint s) {
        const std::vector<vec3> &vertices = shoot.shoot_internode_vertices.at(node);
        const std::vector<float> &radii = shoot.shoot_internode_radii.at(node);
        if (node == 0) {
            return {vertices.at(s), vertices.at(s + 1), radii.at(s), radii.at(s + 1)};
        }
        if (s == 0) {
            return {shoot.shoot_internode_vertices.at(node - 1).back(), vertices.at(0), shoot.shoot_internode_radii.at(node - 1).back(), radii.at(0)};
        }
        return {vertices.at(s - 1), vertices.at(s), radii.at(s - 1), radii.at(s)};
    }

    //! Current geometry of a cached candidate segment; false if the shoot, phytomer or segment no longer exists
    bool fetchBranchSegment(const PlantInstance &plant, const std::array<uint, 3> &key, BranchSegment &segment) {
        if (key[0] >= plant.shoot_tree.size()) {
            return false;
        }
        const Shoot &shoot = *plant.shoot_tree.at(key[0]);
        if (key[1] >= shoot.shoot_internode_vertices.size() || key[1] >= shoot.shoot_internode_radii.size() || shoot.shoot_internode_vertices.at(key[1]).size() != shoot.shoot_internode_radii.at(key[1]).size()) {
            return false;
        }
        if (key[2] >= internodeSegmentCount(shoot, key[1])) {
            return false;
        }
        segment = internodeSegment(shoot, key[1], key[2]);
        return true;
    }

    //! Distance from a point to an axis-aligned box (zero inside)
    float boxDistance(const vec3 &point, const vec3 &lower, const vec3 &upper) {
        const float dx = std::max({0.f, lower.x - point.x, point.x - upper.x});
        const float dy = std::max({0.f, lower.y - point.y, point.y - upper.y});
        const float dz = std::max({0.f, lower.z - point.z, point.z - upper.z});
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    //! Direction in which a fruit is pushed off a branch: from the nearest axis point toward the fruit's center, or across the branch if the two coincide
    vec3 pushDirection(const vec3 &fruit_center, const vec3 &on_axis, const BranchSegment &segment) {
        vec3 direction = fruit_center - on_axis;
        if (direction.magnitude() < 1e-6f) {
            direction = cross(segment.end - segment.start, make_vec3(0, 0, 1));
            if (direction.magnitude() < 1e-6f) {
                direction = make_vec3(1, 0, 0);
            }
        }
        direction.normalize();
        return direction;
    }

    //! Hinge about which a contact is moved out: the smallest rotation about \a pivot that moves \a contact.point along \a contact.direction by its penetration
    struct Hinge {
        bool usable = false; // false when the contact lies along the lever, so that no rotation about the pivot moves it out
        vec3 axis;
        float angle = 0; // rotation that clears the contact (before capping)
        float efficiency = 0; // distance the contact moves per radian: lever length times the sine of the angle between lever and push direction
    };

    Hinge hingeFor(const Contact &contact, const vec3 &pivot, float min_lever_sine) {
        Hinge hinge;
        vec3 lever = contact.point - pivot;
        const float lever_length = lever.magnitude();
        if (lever_length < 1e-4f) {
            return hinge;
        }
        lever = lever / lever_length;
        vec3 axis = cross(lever, contact.direction);
        const float lever_sine = axis.magnitude();
        if (lever_sine < min_lever_sine) {
            return hinge;
        }
        hinge.usable = true;
        hinge.axis = axis / lever_sine;
        hinge.efficiency = lever_length * lever_sine;
        hinge.angle = contact.penetration / hinge.efficiency;
        return hinge;
    }

    //! The rotation \a q expressed after the frame it acts in has been turned by \a turn (its axis is turned, its angle kept)
    Quaternion turnQuaternion(const Quaternion &turn, const Quaternion &q) {
        const Quaternion turn_inverse{turn.w, -turn.x, -turn.y, -turn.z};
        return quaternionProduct(turn, quaternionProduct(q, turn_inverse));
    }

    //! Angle of the rotation \a step composed onto the rotation \a accumulated
    float composedAngle(const Quaternion &step, const Quaternion &accumulated) {
        vec3 axis;
        float angle;
        quaternionToAxisAngle(quaternionProduct(step, accumulated), make_vec3(0, 0, 1), axis, angle);
        return angle;
    }

    void rotateWorldProxy(FruitWorldProxy &proxy, const vec3 &pivot, const vec3 &axis, float angle) {
        proxy.bounding_center = rotatePointAboutLine(proxy.bounding_center, pivot, axis, angle);
        for (vec3 &center: proxy.sphere_centers) {
            center = rotatePointAboutLine(center, pivot, axis, angle);
        }
        for (vec3 &vertex: proxy.triangle_vertices) {
            vertex = rotatePointAboutLine(vertex, pivot, axis, angle);
        }
    }

    //! Per-iteration cap on the swing, so that a contact is approached rather than overshot
    constexpr float swing_step_max_rad = 5.f * PI_F / 180.f;
    //! Iterations of the swing per time step
    constexpr int swing_iterations_max = 20;
    //! Iterations without the deepest penetration improving before a bud is considered wedged
    constexpr int swing_stall_iterations = 4;
    //! Below this sine of the angle between lever and push direction a rotation about the hinge moves the contact mostly sideways rather than out, and is not attempted. This also sets aside the
    //! contact every panicle has with the culm it continues: the push there is along the panicle itself, so no hinge could clear it and any attempt would only turn the head.
    constexpr float swing_min_lever_sine = 0.3f;

} // namespace

// ------------------------------------------------------------------------------------------------------------------ //

namespace {

    //! Deepest contact between one fruit and one branch segment; zero penetration when they are clear
    fruit_contact::Contact pairContact(const FruitWorldProxy &proxy, size_t fruit_index, const BranchSegment &segment, float clearance) {
        Contact worst;
        worst.fruit_index = fruit_index;
        vec3 on_axis;
        float radius_at;
        const float center_distance = branchAxisDistance(proxy.bounding_center, segment, on_axis, radius_at);
        if (center_distance - radius_at - clearance - proxy.bounding_radius > 0.f) {
            return worst;
        }
        auto consider = [&](float penetration, const vec3 &point, const vec3 &direction) {
            if (penetration > worst.penetration) {
                worst.penetration = penetration;
                worst.point = point;
                worst.direction = direction;
            }
        };
        if (!proxy.triangle_vertices.empty()) {
            for (size_t v = 0; v + 2 < proxy.triangle_vertices.size(); v += 3) {
                vec3 on_segment, on_triangle;
                const float distance_squared = collision::closestPointsSegmentTriangle(segment.start, segment.end, proxy.triangle_vertices[v], proxy.triangle_vertices[v + 1], proxy.triangle_vertices[v + 2], on_segment, on_triangle);
                consider(branchRadiusAt(segment, on_segment) + clearance - std::sqrt(distance_squared), on_triangle, pushDirection(proxy.bounding_center, on_segment, segment));
            }
            // The triangles see a branch only where it reaches the fruit's surface, so a segment lying wholly inside the fruit would pass unseen. Its ends are therefore also tested against the chain
            // spheres, which enclose the fruit. This is conservative: an end just outside the fruit but inside a sphere also counts.
            for (const auto &end: {std::make_pair(segment.start, segment.start_radius), std::make_pair(segment.end, segment.end_radius)}) {
                for (size_t k = 0; k < proxy.sphere_centers.size(); k++) {
                    const float distance = (end.first - proxy.sphere_centers[k]).magnitude();
                    consider(proxy.sphere_radii[k] + end.second + clearance - distance, proxy.sphere_centers[k], pushDirection(proxy.sphere_centers[k], end.first, segment));
                }
            }
        } else {
            for (size_t k = 0; k < proxy.sphere_centers.size(); k++) {
                vec3 sphere_on_axis;
                float sphere_radius_at;
                const float distance = branchAxisDistance(proxy.sphere_centers[k], segment, sphere_on_axis, sphere_radius_at);
                consider(sphere_radius_at + clearance + proxy.sphere_radii[k] - distance, proxy.sphere_centers[k], pushDirection(proxy.sphere_centers[k], sphere_on_axis, segment));
            }
        }
        return worst;
    }

    //! Every contact between a bud's fruit and the branch segments, one per fruit and segment, deepest first
    std::vector<Contact> contactsByDepth(const std::vector<FruitWorldProxy> &fruit, const std::vector<BranchSegment> &segments, float clearance) {
        std::vector<Contact> contacts;
        for (size_t fruit_index = 0; fruit_index < fruit.size(); fruit_index++) {
            for (const BranchSegment &segment: segments) {
                const Contact contact = pairContact(fruit[fruit_index], fruit_index, segment, clearance);
                if (contact.penetration > 0.f) {
                    contacts.push_back(contact);
                }
            }
        }
        std::sort(contacts.begin(), contacts.end(), [](const Contact &a, const Contact &b) { return a.penetration > b.penetration; });
        return contacts;
    }

} // namespace

fruit_contact::Contact fruit_contact::deepestContact(const std::vector<WorldProxy> &fruit, const std::vector<BranchSegment> &segments, float clearance) {
    Contact worst;
    for (size_t fruit_index = 0; fruit_index < fruit.size(); fruit_index++) {
        for (const BranchSegment &segment: segments) {
            const Contact contact = pairContact(fruit[fruit_index], fruit_index, segment, clearance);
            if (contact.penetration > worst.penetration) {
                worst = contact;
            }
        }
    }
    return worst;
}

bool fruit_contact::pushOut(const std::vector<WorldProxy> &fruit, const std::vector<vec3> &fruit_bases, const vec3 &pivot, const std::vector<BranchSegment> &segments, float clearance, const HingeControls &controls) {
    // Progress is judged on the total penetration of all contacts, so that clearing a shallow contact while a deeper one that no hinge can move stays put still counts
    float best_total = std::numeric_limits<float>::max();
    int stalled_iterations = 0;
    for (int iteration = 0; iteration < swing_iterations_max; iteration++) {
        const std::vector<Contact> contacts = contactsByDepth(fruit, segments, clearance);
        if (contacts.empty()) {
            return true;
        }
        float total = 0.f;
        for (const Contact &contact: contacts) {
            total += contact.penetration;
        }
        if (total < best_total - 1e-5f) {
            best_total = total;
            stalled_iterations = 0;
        } else if (++stalled_iterations >= swing_stall_iterations) {
            return false; // wedged between branches: the least-penetrating pose reached is kept
        }

        // The deepest contact that some hinge can move out, at whichever hinge does it with the smaller rotation. A contact pushed along both levers, which no rotation about either pivot relieves,
        // is passed over rather than ending the push-out for every other contact of the bud.
        bool moved = false;
        for (const Contact &contact: contacts) {
            Hinge swing = hingeFor(contact, pivot, swing_min_lever_sine);
            if (swing.usable) {
                swing.angle = std::min(swing.angle, swing_step_max_rad);
                swing.usable = controls.swing_allowed(swing.axis, swing.angle);
            }
            const size_t f = contact.fruit_index;
            Hinge tilt = hingeFor(contact, fruit_bases.at(f), swing_min_lever_sine);
            if (tilt.usable) {
                tilt.angle = std::min(tilt.angle, swing_step_max_rad);
                tilt.usable = controls.tilt_allowed(f, tilt.axis, tilt.angle);
            }
            if (!swing.usable && !tilt.usable) {
                continue;
            }
            if (swing.usable && (!tilt.usable || swing.efficiency >= tilt.efficiency)) {
                controls.apply_swing(swing.axis, swing.angle);
            } else {
                controls.apply_tilt(f, tilt.axis, tilt.angle);
            }
            moved = true;
            break;
        }
        if (!moved) {
            return false; // no hinge can move any remaining contact out any further
        }
    }
    return false;
}

void Phytomer::applyPeduncleSwing(FloralBud &fbud, const helios::vec3 &axis, float angle) {
    if (angle == 0.f) {
        return;
    }
    if (!std::isfinite(angle) || !std::isfinite(axis.x) || !std::isfinite(axis.y) || !std::isfinite(axis.z) || axis.magnitude() < 1e-6f) {
        helios_runtime_error("ERROR (Phytomer::applyPeduncleSwing): The swing axis must be a non-zero finite vector and the angle finite.");
    }
    vec3 unit_axis = axis;
    unit_axis.normalize();
    const vec3 pivot = fbud.base_position;

    for (uint objID: fbud.inflorescence_objIDs) {
        if (context_ptr->doesObjectExist(objID)) {
            context_ptr->rotateObject(objID, angle, pivot, unit_axis);
        }
    }
    for (uint objID: fbud.peduncle_objIDs) {
        if (context_ptr->doesObjectExist(objID)) {
            context_ptr->rotateObject(objID, angle, pivot, unit_axis);
        }
    }

    const uint2 storage_index = getFloralBudStorageIndex(fbud);
    if (storage_index.x < peduncle_vertices.size() && storage_index.y < peduncle_vertices.at(storage_index.x).size()) {
        for (vec3 &vertex: peduncle_vertices.at(storage_index.x).at(storage_index.y)) {
            vertex = rotatePointAboutLine(vertex, pivot, unit_axis, angle);
        }
    }
    for (vec3 &base: fbud.inflorescence_bases) {
        base = rotatePointAboutLine(base, pivot, unit_axis, angle);
    }
    for (AxisRotation &rotation: fbud.inflorescence_rotation) {
        rotation.peduncle_axis = rotatePointAboutLine(rotation.peduncle_axis, nullorigin, unit_axis, angle);
        // A tilt about the fruit's attachment point, carried along with the peduncle, is the same tilt about an axis turned with it
        rotation.contact_tilt_axis = rotatePointAboutLine(rotation.contact_tilt_axis, nullorigin, unit_axis, angle);
    }

    // The proxies are kept in the frame of the fruit's attachment point, which turns with the swing
    for (uint objID: fbud.inflorescence_objIDs) {
        auto proxy = plantarchitecture_ptr->fruit_contact_proxies.find(objID);
        if (proxy == plantarchitecture_ptr->fruit_contact_proxies.end()) {
            continue;
        }
        proxy->second.bounding_center = rotatePointAboutLine(proxy->second.bounding_center, nullorigin, unit_axis, angle);
        for (vec3 &center: proxy->second.sphere_centers) {
            center = rotatePointAboutLine(center, nullorigin, unit_axis, angle);
        }
        for (vec3 &vertex: proxy->second.triangle_vertices) {
            vertex = rotatePointAboutLine(vertex, nullorigin, unit_axis, angle);
        }
    }

    // Compose into the bud's accumulated swing
    const Quaternion previous = (fbud.contact_swing_angle != 0.f) ? quaternionFromAxisAngle(fbud.contact_swing_axis, fbud.contact_swing_angle) : Quaternion{};
    const Quaternion total = quaternionProduct(quaternionFromAxisAngle(unit_axis, angle), previous);
    quaternionToAxisAngle(total, unit_axis, fbud.contact_swing_axis, fbud.contact_swing_angle);
}

void Phytomer::applyFruitContactTilt(FloralBud &fbud, size_t fruit_index, const helios::vec3 &axis, float angle) {
    if (angle == 0.f) {
        return;
    }
    if (fruit_index >= fbud.inflorescence_objIDs.size() || fruit_index >= fbud.inflorescence_bases.size() || fruit_index >= fbud.inflorescence_rotation.size()) {
        helios_runtime_error("ERROR (Phytomer::applyFruitContactTilt): Fruit index " + std::to_string(fruit_index) + " is out of range for a floral bud carrying " + std::to_string(fbud.inflorescence_objIDs.size()) + " flowers/fruit.");
    }
    if (!std::isfinite(angle) || !std::isfinite(axis.x) || !std::isfinite(axis.y) || !std::isfinite(axis.z) || axis.magnitude() < 1e-6f) {
        helios_runtime_error("ERROR (Phytomer::applyFruitContactTilt): The tilt axis must be a non-zero finite vector and the angle finite.");
    }
    vec3 unit_axis = axis;
    unit_axis.normalize();
    const uint objID = fbud.inflorescence_objIDs.at(fruit_index);
    if (context_ptr->doesObjectExist(objID)) {
        context_ptr->rotateObject(objID, angle, fbud.inflorescence_bases.at(fruit_index), unit_axis);
    }

    auto proxy = plantarchitecture_ptr->fruit_contact_proxies.find(objID);
    if (proxy != plantarchitecture_ptr->fruit_contact_proxies.end()) {
        proxy->second.bounding_center = rotatePointAboutLine(proxy->second.bounding_center, nullorigin, unit_axis, angle);
        for (vec3 &center: proxy->second.sphere_centers) {
            center = rotatePointAboutLine(center, nullorigin, unit_axis, angle);
        }
        for (vec3 &vertex: proxy->second.triangle_vertices) {
            vertex = rotatePointAboutLine(vertex, nullorigin, unit_axis, angle);
        }
    }

    AxisRotation &rotation = fbud.inflorescence_rotation.at(fruit_index);
    const Quaternion previous = (rotation.contact_tilt_angle != 0.f) ? quaternionFromAxisAngle(rotation.contact_tilt_axis, rotation.contact_tilt_angle) : Quaternion{};
    const Quaternion total = quaternionProduct(quaternionFromAxisAngle(unit_axis, angle), previous);
    quaternionToAxisAngle(total, unit_axis, rotation.contact_tilt_axis, rotation.contact_tilt_angle);
}

// ------------------------------------------------------------------------------------------------------------------ //

void PlantArchitecture::enableFruitBranchCollisionAvoidance() {
    fruit_branch_collision_enabled = true;
    if (printmessages) {
        std::cout << "Fruit-branch collision avoidance enabled: fruit will be swung out of the internodes of their own plant as they grow" << std::endl;
    }
}

void PlantArchitecture::disableFruitBranchCollisionAvoidance() {
    fruit_branch_collision_enabled = false;
    fruit_contact_proxies.clear();
    fruit_contact_candidates.clear();
}

void PlantArchitecture::setFruitBranchCollisionParameters(float max_swing_deg, float candidate_refresh_days, float clearance, float settle_rate_deg_per_day) {
    if (!(max_swing_deg > 0.f) || max_swing_deg > 180.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setFruitBranchCollisionParameters): max_swing_deg must be greater than 0 and at most 180 degrees, but was " + std::to_string(max_swing_deg) + ".");
    }
    if (!(candidate_refresh_days > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::setFruitBranchCollisionParameters): candidate_refresh_days must be positive, but was " + std::to_string(candidate_refresh_days) + ".");
    }
    if (!(clearance >= 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::setFruitBranchCollisionParameters): clearance must be zero or positive, but was " + std::to_string(clearance) + ".");
    }
    if (!(settle_rate_deg_per_day >= 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::setFruitBranchCollisionParameters): settle_rate_deg_per_day must be zero or positive, but was " + std::to_string(settle_rate_deg_per_day) + ".");
    }
    fruit_branch_collision_max_swing_rad = deg2rad(max_swing_deg);
    fruit_branch_collision_refresh_days = candidate_refresh_days;
    fruit_branch_collision_clearance = clearance;
    fruit_branch_collision_settle_rad_per_day = deg2rad(settle_rate_deg_per_day);
    // Reach of the candidate lists depends on the clearance, so they are rebuilt at the next check
    fruit_contact_candidates.clear();
}

FruitContactProxy PlantArchitecture::buildFruitContactProxy(const FloralBud &fbud, size_t fruit_index) const {
    const uint objID = fbud.inflorescence_objIDs.at(fruit_index);
    const vec3 base = fbud.inflorescence_bases.at(fruit_index);
    const std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);

    // The proxy is stored at the scale it was built at and rescaled by current/build scale from then on, so that scale must be a usable divisor
    if (!(fbud.current_fruit_scale_factor > 0.f) || !std::isfinite(fbud.current_fruit_scale_factor)) {
        helios_runtime_error("ERROR (PlantArchitecture::buildFruitContactProxy): Fruit object " + std::to_string(objID) + " has a fruit scale factor of " + std::to_string(fbud.current_fruit_scale_factor) +
                             ", so its size is undefined and no contact proxy can be built for it. A fruiting floral bud's scale factor must be positive; check the value read from the plant structure XML or set by setInflorescenceScaleFraction().");
    }
    FruitContactProxy proxy;
    proxy.build_scale = fbud.current_fruit_scale_factor;

    // Vertices relative to the attachment point; triangles kept when the mesh is small enough to test exactly
    std::vector<vec3> vertices;
    const bool keep_triangles = UUIDs.size() <= fruit_contact_exact_primitive_limit;
    for (uint UUID: UUIDs) {
        const std::vector<vec3> primitive_vertices = context_ptr->getPrimitiveVertices(UUID);
        for (const vec3 &vertex: primitive_vertices) {
            vertices.push_back(vertex - base);
        }
        if (keep_triangles) {
            if (primitive_vertices.size() == 3) {
                for (const vec3 &vertex: primitive_vertices) {
                    proxy.triangle_vertices.push_back(vertex - base);
                }
            } else if (primitive_vertices.size() == 4) {
                const vec3 corners[4] = {primitive_vertices[0] - base, primitive_vertices[1] - base, primitive_vertices[2] - base, primitive_vertices[3] - base};
                proxy.triangle_vertices.insert(proxy.triangle_vertices.end(), {corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]});
            }
        }
    }
    if (vertices.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::buildFruitContactProxy): Fruit object " + std::to_string(objID) + " has no primitives with vertices, so no contact proxy can be built for it.");
    }

    // Enclosing sphere about the centroid
    vec3 centroid = nullorigin;
    for (const vec3 &vertex: vertices) {
        centroid = centroid + vertex;
    }
    centroid = centroid / float(vertices.size());
    proxy.bounding_center = centroid;
    for (const vec3 &vertex: vertices) {
        proxy.bounding_radius = std::max(proxy.bounding_radius, (vertex - centroid).magnitude());
        proxy.max_extent = std::max(proxy.max_extent, vertex.magnitude());
    }

    // Chain of spheres along the fruit's axis, one per slab of the mesh. A round fruit gets one sphere; an elongated cluster gets several, each its slab's enclosing sphere, so that the chain does not
    // stick out far beyond the ends of the fruit the way a single sphere would.
    vec3 axis = centroid;
    if (axis.magnitude() < 1e-6f) {
        axis = make_vec3(0, 0, -1);
    }
    axis.normalize();
    float axial_min = std::numeric_limits<float>::max();
    float axial_max = -std::numeric_limits<float>::max();
    float radial_max = 0.f;
    std::vector<float> axial(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        axial[i] = vertices[i] * axis;
        axial_min = std::min(axial_min, axial[i]);
        axial_max = std::max(axial_max, axial[i]);
        radial_max = std::max(radial_max, (vertices[i] - axial[i] * axis).magnitude());
    }
    const float length = std::max(axial_max - axial_min, 1e-6f);
    const int slab_count = (radial_max > 0.f) ? std::clamp(int(std::round(length / (2.f * radial_max))), 1, 8) : 1;
    std::vector<vec3> slab_centroid(slab_count, nullorigin);
    std::vector<int> slab_population(slab_count, 0);
    std::vector<int> slab_of(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        const int slab = std::clamp(int((axial[i] - axial_min) / length * float(slab_count)), 0, slab_count - 1);
        slab_of[i] = slab;
        slab_centroid[slab] = slab_centroid[slab] + vertices[i];
        slab_population[slab]++;
    }
    for (int slab = 0; slab < slab_count; slab++) {
        if (slab_population[slab] == 0) {
            continue;
        }
        const vec3 center = slab_centroid[slab] / float(slab_population[slab]);
        float radius = 0.f;
        for (size_t i = 0; i < vertices.size(); i++) {
            if (slab_of[i] == slab) {
                radius = std::max(radius, (vertices[i] - center).magnitude());
            }
        }
        proxy.sphere_centers.push_back(center);
        proxy.sphere_radii.push_back(radius);
    }
    return proxy;
}

FruitContactCandidates PlantArchitecture::collectFruitContactCandidates(uint plantID, const Phytomer &phytomer, const FloralBud &fbud, float reach, const std::vector<helios::vec3> &shoot_lower, const std::vector<helios::vec3> &shoot_upper) const {
    const PlantInstance &plant = plant_instances.at(plantID);
    // A grass panicle or tassel continues the culm it stands on, so it necessarily meets that internode at its base; that is not a contact to resolve, and no hinge could.
    const bool exclude_own_internode = phytomer.peduncleShouldMatchCulm();
    FruitContactCandidates candidates;
    candidates.refreshed_at_age = plant.current_age;
    candidates.shoot_count = plant.shoot_tree.size();
    for (uint shootID = 0; shootID < plant.shoot_tree.size(); shootID++) {
        const std::shared_ptr<Shoot> &shoot = plant.shoot_tree.at(shootID);
        if (shoot->phytomers.empty() || boxDistance(fbud.base_position, shoot_lower.at(shootID), shoot_upper.at(shootID)) > reach) {
            continue;
        }
        const size_t node_count = std::min(shoot->shoot_internode_vertices.size(), shoot->shoot_internode_radii.size());
        for (uint node = 0; node < node_count; node++) {
            if (exclude_own_internode && shootID == phytomer.parent_shoot_ID && node == uint(phytomer.shoot_index.x)) {
                continue;
            }
            if (shoot->shoot_internode_vertices.at(node).size() != shoot->shoot_internode_radii.at(node).size()) {
                continue;
            }
            const size_t segment_count = internodeSegmentCount(*shoot, node);
            for (uint s = 0; s < segment_count; s++) {
                const BranchSegment segment = internodeSegment(*shoot, node, s);
                vec3 on_axis;
                float radius_at;
                if (branchAxisDistance(fbud.base_position, segment, on_axis, radius_at) - radius_at <= reach) {
                    candidates.segments.push_back({shootID, node, s});
                }
            }
        }
    }
    return candidates;
}

void PlantArchitecture::resolveFruitBranchCollisions(uint plantID, float dt_days) {
    PlantInstance &plant = plant_instances.at(plantID);
    const size_t shoot_count = plant.shoot_tree.size();

    // Bounding boxes of the shoots' internode capsules, built only if some bud's candidate list has to be rebuilt this step
    std::vector<vec3> shoot_lower;
    std::vector<vec3> shoot_upper;
    auto ensureShootBoxes = [&]() {
        if (!shoot_lower.empty()) {
            return;
        }
        shoot_lower.assign(shoot_count, make_vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()));
        shoot_upper.assign(shoot_count, make_vec3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()));
        for (uint shootID = 0; shootID < shoot_count; shootID++) {
            const std::shared_ptr<Shoot> &shoot = plant.shoot_tree.at(shootID);
            for (size_t node = 0; node < shoot->shoot_internode_vertices.size(); node++) {
                float radius_max = 0.f;
                if (node < shoot->shoot_internode_radii.size()) {
                    for (float radius: shoot->shoot_internode_radii.at(node)) {
                        radius_max = std::max(radius_max, radius);
                    }
                }
                for (const vec3 &vertex: shoot->shoot_internode_vertices.at(node)) {
                    shoot_lower[shootID] = make_vec3(std::min(shoot_lower[shootID].x, vertex.x - radius_max), std::min(shoot_lower[shootID].y, vertex.y - radius_max), std::min(shoot_lower[shootID].z, vertex.z - radius_max));
                    shoot_upper[shootID] = make_vec3(std::max(shoot_upper[shootID].x, vertex.x + radius_max), std::max(shoot_upper[shootID].y, vertex.y + radius_max), std::max(shoot_upper[shootID].z, vertex.z + radius_max));
                }
            }
        }
    };

    for (const std::shared_ptr<Shoot> &shoot: plant.shoot_tree) {
        for (const std::shared_ptr<Phytomer> &phytomer: shoot->phytomers) {
            for (std::vector<FloralBud> &petiole_buds: phytomer->floral_buds) {
                for (FloralBud &fbud: petiole_buds) {
                    if (fbud.state != BUD_FRUITING || fbud.inflorescence_objIDs.empty()) {
                        continue;
                    }

                    // ---- proxies of this bud's fruit ---- //
                    bool all_fruit_exist = true;
                    for (size_t i = 0; i < fbud.inflorescence_objIDs.size(); i++) {
                        const uint objID = fbud.inflorescence_objIDs.at(i);
                        if (!context_ptr->doesObjectExist(objID)) {
                            all_fruit_exist = false;
                            break;
                        }
                        if (fruit_contact_proxies.find(objID) == fruit_contact_proxies.end()) {
                            fruit_contact_proxies.emplace(objID, buildFruitContactProxy(fbud, i));
                        }
                    }
                    if (!all_fruit_exist) {
                        continue;
                    }

                    // ---- candidate branches ---- //
                    const uint candidate_key = fbud.inflorescence_objIDs.front();
                    auto candidate_it = fruit_contact_candidates.find(candidate_key);
                    const bool refresh = candidate_it == fruit_contact_candidates.end() || plant.current_age - candidate_it->second.refreshed_at_age >= fruit_branch_collision_refresh_days || candidate_it->second.shoot_count != shoot_count;
                    if (refresh) {
                        // Everything the swing can bring within reach of the node: the peduncle, then the fruit at full size, then the clearance; widened for branch thickening between refreshes
                        float reach = 0.f;
                        for (size_t i = 0; i < fbud.inflorescence_objIDs.size(); i++) {
                            const FruitContactProxy &proxy = fruit_contact_proxies.at(fbud.inflorescence_objIDs.at(i));
                            reach = std::max(reach, (fbud.inflorescence_bases.at(i) - fbud.base_position).magnitude() + proxy.max_extent / proxy.build_scale);
                        }
                        reach = 1.2f * (reach + fruit_branch_collision_clearance) + 0.02f;
                        ensureShootBoxes();
                        candidate_it = fruit_contact_candidates.insert_or_assign(candidate_key, collectFruitContactCandidates(plantID, *phytomer, fbud, reach, shoot_lower, shoot_upper)).first;
                    }
                    if (candidate_it->second.segments.empty()) {
                        continue;
                    }
                    std::vector<BranchSegment> segments;
                    segments.reserve(candidate_it->second.segments.size());
                    for (const std::array<uint, 3> &key: candidate_it->second.segments) {
                        BranchSegment segment;
                        if (fetchBranchSegment(plant, key, segment)) {
                            segments.push_back(segment);
                        }
                    }
                    if (segments.empty()) {
                        continue;
                    }

                    // ---- broad phase: is any fruit's enclosing sphere within reach of a branch? ---- //
                    bool possible_contact = false;
                    for (size_t i = 0; i < fbud.inflorescence_objIDs.size() && !possible_contact; i++) {
                        const FruitContactProxy &proxy = fruit_contact_proxies.at(fbud.inflorescence_objIDs.at(i));
                        const float growth = fbud.current_fruit_scale_factor / proxy.build_scale;
                        const vec3 bounding_center = fbud.inflorescence_bases.at(i) + growth * proxy.bounding_center;
                        const float bounding_radius = growth * proxy.bounding_radius;
                        for (const BranchSegment &segment: segments) {
                            vec3 on_axis;
                            float radius_at;
                            if (branchAxisDistance(bounding_center, segment, on_axis, radius_at) - radius_at - fruit_branch_collision_clearance - bounding_radius <= 0.f) {
                                possible_contact = true;
                                break;
                            }
                        }
                    }
                    // A bud that has been swung or tilted may settle back toward its grown pose if it is clear
                    bool has_pose = fbud.contact_swing_angle > 0.f;
                    for (const AxisRotation &rotation: fbud.inflorescence_rotation) {
                        has_pose = has_pose || rotation.contact_tilt_angle > 0.f;
                    }
                    // A bud found unable to settle (resting against wood) is not retried until its fruit grow or its candidate list is rebuilt, which also catches a branch that has since moved away
                    const bool settle_blocked = candidate_it->second.settle_blocked_at_scale == fbud.current_fruit_scale_factor;
                    const bool settle_active = has_pose && fruit_branch_collision_settle_rad_per_day > 0.f && dt_days > 0.f && !settle_blocked;
                    if (!possible_contact && !settle_active) {
                        continue;
                    }
                    // A bud left wedged is not retried until its fruit have grown (its candidate list is rebuilt with unresolved_at_scale reset on the next refresh)
                    const bool wedged = candidate_it->second.unresolved_at_scale >= 0.f && std::abs(fbud.current_fruit_scale_factor - candidate_it->second.unresolved_at_scale) < 0.05f;
                    if (wedged && !settle_active) {
                        continue;
                    }

                    // ---- world-space proxies at the fruit's current size ---- //
                    std::vector<FruitWorldProxy> fruit(fbud.inflorescence_objIDs.size());
                    std::vector<vec3> fruit_bases = fbud.inflorescence_bases;
                    for (size_t i = 0; i < fbud.inflorescence_objIDs.size(); i++) {
                        const FruitContactProxy &proxy = fruit_contact_proxies.at(fbud.inflorescence_objIDs.at(i));
                        const float growth = fbud.current_fruit_scale_factor / proxy.build_scale;
                        const vec3 base = fruit_bases.at(i);
                        FruitWorldProxy &world = fruit[i];
                        world.bounding_center = base + growth * proxy.bounding_center;
                        world.bounding_radius = growth * proxy.bounding_radius;
                        world.sphere_centers.reserve(proxy.sphere_centers.size());
                        for (const vec3 &center: proxy.sphere_centers) {
                            world.sphere_centers.push_back(base + growth * center);
                        }
                        world.sphere_radii.reserve(proxy.sphere_radii.size());
                        for (float radius: proxy.sphere_radii) {
                            world.sphere_radii.push_back(growth * radius);
                        }
                        world.triangle_vertices.reserve(proxy.triangle_vertices.size());
                        for (const vec3 &vertex: proxy.triangle_vertices) {
                            world.triangle_vertices.push_back(base + growth * vertex);
                        }
                    }

                    // ---- rotations of this step, kept as quaternions and applied to the proxies as they are decided ---- //
                    // The peduncle swings about the node (all fruit of the bud together) or one fruit tilts about its attachment point. Rotations act on the world-space proxies while iterating and are
                    // applied to the Context objects once, at the end.
                    const vec3 pivot = fbud.base_position;
                    const Quaternion existing_swing_initial = (fbud.contact_swing_angle != 0.f) ? quaternionFromAxisAngle(fbud.contact_swing_axis, fbud.contact_swing_angle) : Quaternion{};
                    Quaternion existing_swing = existing_swing_initial;
                    Quaternion session_swing;
                    vec3 last_swing_axis = fbud.contact_swing_axis;
                    std::vector<Quaternion> existing_tilt(fruit.size());
                    std::vector<Quaternion> session_tilt(fruit.size());
                    std::vector<vec3> last_tilt_axis(fruit.size());
                    for (size_t i = 0; i < fruit.size(); i++) {
                        const AxisRotation &rotation = fbud.inflorescence_rotation.at(i);
                        existing_tilt[i] = (rotation.contact_tilt_angle != 0.f) ? quaternionFromAxisAngle(rotation.contact_tilt_axis, rotation.contact_tilt_angle) : Quaternion{};
                        last_tilt_axis[i] = rotation.contact_tilt_axis;
                    }
                    auto applySwingStep = [&](const vec3 &axis, float angle) {
                        const Quaternion step_swing = quaternionFromAxisAngle(axis, angle);
                        session_swing = quaternionProduct(step_swing, session_swing);
                        last_swing_axis = axis;
                        for (FruitWorldProxy &world: fruit) {
                            rotateWorldProxy(world, pivot, axis, angle);
                        }
                        // The fruit attachment points move with the peduncle, and so do the axes of the tilts already made about them
                        for (size_t i = 0; i < fruit.size(); i++) {
                            fruit_bases[i] = rotatePointAboutLine(fruit_bases[i], pivot, axis, angle);
                            session_tilt[i] = turnQuaternion(step_swing, session_tilt[i]);
                            existing_tilt[i] = turnQuaternion(step_swing, existing_tilt[i]);
                            last_tilt_axis[i] = rotatePointAboutLine(last_tilt_axis[i], nullorigin, axis, angle);
                        }
                    };
                    auto applyTiltStep = [&](size_t i, const vec3 &axis, float angle) {
                        session_tilt[i] = quaternionProduct(quaternionFromAxisAngle(axis, angle), session_tilt[i]);
                        last_tilt_axis[i] = axis;
                        rotateWorldProxy(fruit[i], fruit_bases[i], axis, angle);
                    };

                    // ---- move the deepest contact out, at whichever hinge does it with the smaller rotation, until the fruit are clear or the bud is wedged ---- //
                    bool pushed = false;
                    if (possible_contact && !wedged) {
                        fruit_contact::HingeControls controls;
                        controls.swing_allowed = [&](const vec3 &axis, float angle) {
                            return composedAngle(quaternionFromAxisAngle(axis, angle), quaternionProduct(session_swing, existing_swing)) <= fruit_branch_collision_max_swing_rad;
                        };
                        controls.apply_swing = [&](const vec3 &axis, float angle) {
                            applySwingStep(axis, angle);
                            pushed = true;
                        };
                        controls.tilt_allowed = [&](size_t i, const vec3 &axis, float angle) {
                            return composedAngle(quaternionFromAxisAngle(axis, angle), quaternionProduct(session_tilt[i], existing_tilt[i])) <= fruit_branch_collision_max_swing_rad;
                        };
                        controls.apply_tilt = [&](size_t i, const vec3 &axis, float angle) {
                            applyTiltStep(i, axis, angle);
                            pushed = true;
                        };
                        const bool resolved = fruit_contact::pushOut(fruit, fruit_bases, pivot, segments, fruit_branch_collision_clearance, controls);
                        candidate_it->second.unresolved_at_scale = resolved ? -1.f : fbud.current_fruit_scale_factor;
                    }

                    // ---- settle back toward the grown pose while clear ---- //
                    // The grown pose, hanging as the gravity term left it, is where a fruit returns to when nothing holds it away. Each hinge is turned back by up to the settling rate, and only as far as
                    // the fruit stays clear: the step is tried at full, half and quarter size, and the first that leaves the fruit clear is kept. A fruit resting against a limb thus stays against it.
                    if (settle_active && fruit_contact::deepestContact(fruit, segments, fruit_branch_collision_clearance).penetration <= 0.f) {
                        const float settle_step = fruit_branch_collision_settle_rad_per_day * dt_days;
                        bool settled = false;
                        auto trySettle = [&](const vec3 &axis, float angle, const std::function<void(const vec3 &, float)> &apply) {
                            const float step = std::min(settle_step, angle);
                            for (float fraction: {1.f, 0.5f, 0.25f}) {
                                apply(axis, -step * fraction);
                                if (fruit_contact::deepestContact(fruit, segments, fruit_branch_collision_clearance).penetration <= 0.f) {
                                    settled = true;
                                    return;
                                }
                                apply(axis, step * fraction);
                            }
                        };
                        vec3 total_axis;
                        float total_angle;
                        quaternionToAxisAngle(quaternionProduct(session_swing, existing_swing), last_swing_axis, total_axis, total_angle);
                        if (total_angle > 1e-6f) {
                            trySettle(total_axis, total_angle, applySwingStep);
                        }
                        for (size_t i = 0; i < fruit.size(); i++) {
                            quaternionToAxisAngle(quaternionProduct(session_tilt[i], existing_tilt[i]), last_tilt_axis[i], total_axis, total_angle);
                            if (total_angle > 1e-6f) {
                                trySettle(total_axis, total_angle, [&](const vec3 &axis, float angle) { applyTiltStep(i, axis, angle); });
                            }
                        }
                        candidate_it->second.settle_blocked_at_scale = settled ? -1.f : fbud.current_fruit_scale_factor;
                    }
                    if (pushed) {
                        candidate_it->second.settle_blocked_at_scale = -1.f; // a new pose may have room to settle
                    }

                    // ---- apply to the Context and the bud: the swing first, then the tilts in the swung frame ---- //
                    vec3 swing_axis;
                    float swing_angle;
                    quaternionToAxisAngle(session_swing, last_swing_axis, swing_axis, swing_angle);
                    if (swing_angle > 1e-6f) {
                        phytomer->applyPeduncleSwing(fbud, swing_axis, swing_angle);
                    }
                    for (size_t i = 0; i < fruit.size(); i++) {
                        vec3 tilt_axis;
                        float tilt_angle;
                        quaternionToAxisAngle(session_tilt[i], last_tilt_axis[i], tilt_axis, tilt_angle);
                        if (tilt_angle > 1e-6f) {
                            phytomer->applyFruitContactTilt(fbud, i, tilt_axis, tilt_angle);
                        }
                    }
                }
            }
        }
    }
}

void PlantArchitecture::pruneFruitContactCaches() {
    for (auto it = fruit_contact_proxies.begin(); it != fruit_contact_proxies.end();) {
        if (context_ptr->doesObjectExist(it->first)) {
            ++it;
        } else {
            it = fruit_contact_proxies.erase(it);
        }
    }
    for (auto it = fruit_contact_candidates.begin(); it != fruit_contact_candidates.end();) {
        if (context_ptr->doesObjectExist(it->first)) {
            ++it;
        } else {
            it = fruit_contact_candidates.erase(it);
        }
    }
}
