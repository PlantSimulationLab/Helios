/** \file "PlantPose.cpp" Skeleton of a plant, and re-posing a grown plant at the joints of that skeleton.

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

#include "PlantArchitecture.h"
#include "PlantArchitectureInternal.h"

using namespace helios;

namespace {

    //! Total length of a centerline
    float centerlineLength(const std::vector<vec3> &centerline_vertices) {
        float length = 0;
        for (size_t vertex = 1; vertex < centerline_vertices.size(); vertex++) {
            length += (centerline_vertices.at(vertex) - centerline_vertices.at(vertex - 1)).magnitude();
        }
        return length;
    }

    //! Whether the segment between two centerline vertices is long enough to have a direction
    /**
     * Longer than the rounding error of single-precision coordinates of this size. An exact comparison with zero is not enough: posing
     * transforms each organ's vertices through its own chain of rotations, so two vertices that coincide can come out a few nanometres
     * apart, and the direction between them is then noise.
     */
    bool segmentHasDirection(const vec3 &segment_start, const vec3 &segment_end, float segment_length) {
        const float coordinate_size = std::max(segment_start.magnitude(), segment_end.magnitude());
        const float rounding_error = 32.f * std::numeric_limits<float>::epsilon() * coordinate_size;
        return segment_length > rounding_error;
    }

    //! Unit direction of the first segment of a centerline that has a length, which is the direction in which the organ leaves its base
    vec3 centerlineBaseAxis(const std::vector<vec3> &centerline_vertices, const std::string &organ_description) {
        for (size_t vertex = 1; vertex < centerline_vertices.size(); vertex++) {
            const vec3 segment = centerline_vertices.at(vertex) - centerline_vertices.at(vertex - 1);
            const float segment_length = segment.magnitude();
            if (segmentHasDirection(centerline_vertices.at(vertex - 1), centerline_vertices.at(vertex), segment_length)) {
                return segment / segment_length;
            }
        }
        helios_runtime_error("ERROR (PlantArchitecture::getPlantJointFrame): The centerline of " + organ_description + " has no length, so it has no direction.");
        return nullorigin;
    }

    //! Unit direction of the last segment of a centerline that has a length, which is the direction of the organ where something grows from its tip
    vec3 centerlineTipAxis(const std::vector<vec3> &centerline_vertices, const std::string &organ_description) {
        for (size_t vertex = centerline_vertices.size(); vertex-- > 1;) {
            const vec3 segment = centerline_vertices.at(vertex) - centerline_vertices.at(vertex - 1);
            const float segment_length = segment.magnitude();
            if (segmentHasDirection(centerline_vertices.at(vertex - 1), centerline_vertices.at(vertex), segment_length)) {
                return segment / segment_length;
            }
        }
        helios_runtime_error("ERROR (PlantArchitecture::getPlantJointFrame): The centerline of " + organ_description + " has no length, so it has no direction.");
        return nullorigin;
    }

    //! Unit axis perpendicular to both the parent and the organ axes of a joint, or an arbitrary perpendicular to the parent axis when the two are parallel
    vec3 jointPitchAxis(const vec3 &parent_axis, const vec3 &organ_axis) {
        vec3 pitch_axis = cross(parent_axis, organ_axis);
        if (pitch_axis.magnitude() < 1e-6f) {
            // The organ continues straight on from its parent, so there is no plane of tilt to be perpendicular to. Any perpendicular serves; take the one from whichever
            // coordinate axis is further from the parent axis so that the cross product is well conditioned.
            const vec3 reference_axis = (std::fabs(parent_axis.x) < 0.9f) ? make_vec3(1, 0, 0) : make_vec3(0, 1, 0);
            pitch_axis = cross(parent_axis, reference_axis);
        }
        pitch_axis.normalize();
        return pitch_axis;
    }

    // ---- Rigid motions and transformation matrices ---- //

    //! Rotation matrix, in row-major order, of a turn about a unit axis (Rodrigues' formula)
    void makeAxisRotation(const vec3 &axis, double angle_radians, double rotation[9]) {
        const double axis_length = std::sqrt(double(axis.x) * axis.x + double(axis.y) * axis.y + double(axis.z) * axis.z);
        const double x = axis.x / axis_length;
        const double y = axis.y / axis_length;
        const double z = axis.z / axis_length;
        const double cosine = std::cos(angle_radians);
        const double sine = std::sin(angle_radians);
        const double versine = 1.0 - cosine;
        rotation[0] = cosine + x * x * versine;
        rotation[1] = x * y * versine - z * sine;
        rotation[2] = x * z * versine + y * sine;
        rotation[3] = y * x * versine + z * sine;
        rotation[4] = cosine + y * y * versine;
        rotation[5] = y * z * versine - x * sine;
        rotation[6] = z * x * versine - y * sine;
        rotation[7] = z * y * versine + x * sine;
        rotation[8] = cosine + z * z * versine;
    }

    //! Product of two 3x3 matrices in row-major order
    void multiplyRotations(const double left[9], const double right[9], double product[9]) {
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                product[row * 3 + column] = left[row * 3] * right[column] + left[row * 3 + 1] * right[3 + column] + left[row * 3 + 2] * right[6 + column];
            }
        }
    }

    //! Rigid motion of everything a joint carries when the joint is turned from the way the plant grew
    /**
     * The roll is applied first, about the organ's own axis, which leaves the organ's direction alone; then the pitch, which tilts it away from its parent; then the yaw, which carries it around its parent.
     * All three axes are those of the plant as it grew, so the result does not depend on the pose the joint had before.
     */
    PoseRigidTransform makeJointRotation(const PlantJointFrame &rest_frame, const PlantJointPose &pose) {
        double roll_rotation[9], pitch_rotation[9], yaw_rotation[9], pitch_and_roll[9];
        makeAxisRotation(rest_frame.organ_axis, deg2rad(double(pose.roll_degrees)), roll_rotation);
        makeAxisRotation(rest_frame.pitch_axis, deg2rad(double(pose.pitch_degrees)), pitch_rotation);
        makeAxisRotation(rest_frame.parent_axis, deg2rad(double(pose.yaw_degrees)), yaw_rotation);
        multiplyRotations(pitch_rotation, roll_rotation, pitch_and_roll);

        PoseRigidTransform joint_transform;
        multiplyRotations(yaw_rotation, pitch_and_roll, joint_transform.rotation);
        // Turning about the pivot: p -> pivot + R (p - pivot)
        const double pivot[3] = {rest_frame.pivot.x, rest_frame.pivot.y, rest_frame.pivot.z};
        for (int row = 0; row < 3; row++) {
            joint_transform.translation[row] = pivot[row] - (joint_transform.rotation[row * 3] * pivot[0] + joint_transform.rotation[row * 3 + 1] * pivot[1] + joint_transform.rotation[row * 3 + 2] * pivot[2]);
        }
        return joint_transform;
    }

    //! Stretch of an organ's centerline about its joint when the joint is given a length: p -> pivot + length_scale (p - pivot)
    PoseRigidTransform makeJointStretch(const PlantJointFrame &rest_frame, const PlantJointPose &pose) {
        PoseRigidTransform stretch;
        stretch.scale = pose.length_scale;
        stretch.translation[0] = (1.0 - stretch.scale) * rest_frame.pivot.x;
        stretch.translation[1] = (1.0 - stretch.scale) * rest_frame.pivot.y;
        stretch.translation[2] = (1.0 - stretch.scale) * rest_frame.pivot.z;
        return stretch;
    }

    //! Translation of what grows from the tip of an organ when the organ's centerline is stretched about its joint
    PoseRigidTransform makeTipTranslation(const PlantJointFrame &rest_frame, const PlantJointPose &pose, const vec3 &rest_tip) {
        PoseRigidTransform tip_translation;
        tip_translation.translation[0] = (double(pose.length_scale) - 1.0) * (double(rest_tip.x) - rest_frame.pivot.x);
        tip_translation.translation[1] = (double(pose.length_scale) - 1.0) * (double(rest_tip.y) - rest_frame.pivot.y);
        tip_translation.translation[2] = (double(pose.length_scale) - 1.0) * (double(rest_tip.z) - rest_frame.pivot.z);
        return tip_translation;
    }

    //! Pose of a joint that carries the chord of its organ, from the joint to the organ's tip, onto another chord
    /**
     * The length scale is the ratio of the two chords' lengths. The direction is reached with a pitch, which sets the chord's angle from the parent axis, and then a yaw, which carries it around the
     * parent axis; of the two pitches that give the right angle, the smaller is taken. A roll is used only when no pitch can reach that angle, which happens when the organ is curved, so that its chord
     * lies out of the plane the pitch turns in, and the target is nearly in line with the parent axis.
     */
    PlantJointPose solveJointPoseForChord(const PlantJointFrame &rest_frame, const vec3 &rest_chord, const vec3 &target_chord) {
        const double rest_length = rest_chord.magnitude();
        const double target_length = target_chord.magnitude();

        // Unit vectors in double precision
        auto unit = [](const vec3 &v, double out[3]) {
            const double length = std::sqrt(double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
            out[0] = v.x / length;
            out[1] = v.y / length;
            out[2] = v.z / length;
        };
        auto dot = [](const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
        auto crossProduct = [](const double a[3], const double b[3], double out[3]) {
            out[0] = a[1] * b[2] - a[2] * b[1];
            out[1] = a[2] * b[0] - a[0] * b[2];
            out[2] = a[0] * b[1] - a[1] * b[0];
        };
        auto rotate = [](const double rotation[9], const double v[3], double out[3]) {
            for (int row = 0; row < 3; row++) {
                out[row] = rotation[row * 3] * v[0] + rotation[row * 3 + 1] * v[1] + rotation[row * 3 + 2] * v[2];
            }
        };

        double parent_axis[3], pitch_axis[3], organ_axis[3], chord[3], target[3];
        unit(rest_frame.parent_axis, parent_axis);
        unit(rest_frame.pitch_axis, pitch_axis);
        unit(rest_frame.organ_axis, organ_axis);
        unit(rest_chord, chord);
        unit(target_chord, target);

        const double target_cosine_from_parent = dot(target, parent_axis);

        // The chord after a pitch of theta about the pitch axis has a component A cos(theta) + B sin(theta) along the parent axis, the pitch axis being perpendicular to the parent axis
        double roll = 0;
        double pitch_axis_cross_chord[3];
        crossProduct(pitch_axis, chord, pitch_axis_cross_chord);
        double A = dot(chord, parent_axis);
        double B = dot(pitch_axis_cross_chord, parent_axis);
        double reach = std::sqrt(A * A + B * B);
        if (std::fabs(target_cosine_from_parent) > reach) {
            // Roll the organ about its own axis until its chord lies in the plane the pitch turns in, from where every angle from the parent axis can be reached
            const double along_organ = dot(chord, organ_axis);
            double across_organ[3] = {chord[0] - along_organ * organ_axis[0], chord[1] - along_organ * organ_axis[1], chord[2] - along_organ * organ_axis[2]};
            double in_plane[3];
            crossProduct(organ_axis, pitch_axis, in_plane); // perpendicular to both the organ axis and the pitch axis
            // Angle, about the organ axis, from the chord's component across the organ to the nearer of the two directions in the plane
            double across_cross_in_plane[3];
            crossProduct(across_organ, in_plane, across_cross_in_plane);
            double roll_to_plane = std::atan2(dot(across_cross_in_plane, organ_axis), dot(across_organ, in_plane));
            if (roll_to_plane > 0.5 * PI_F) {
                roll_to_plane -= PI_F;
            } else if (roll_to_plane < -0.5 * PI_F) {
                roll_to_plane += PI_F;
            }
            roll = roll_to_plane;
            double roll_rotation[9], rolled_chord[3];
            makeAxisRotation(rest_frame.organ_axis, roll, roll_rotation);
            rotate(roll_rotation, chord, rolled_chord);
            chord[0] = rolled_chord[0];
            chord[1] = rolled_chord[1];
            chord[2] = rolled_chord[2];
            crossProduct(pitch_axis, chord, pitch_axis_cross_chord);
            A = dot(chord, parent_axis);
            B = dot(pitch_axis_cross_chord, parent_axis);
            reach = std::sqrt(A * A + B * B);
        }

        // A cos(theta) + B sin(theta) = reach * cos(theta - phase)
        const double phase = std::atan2(B, A);
        const double spread = std::acos(std::max(-1.0, std::min(1.0, target_cosine_from_parent / reach)));
        const double pitch = (std::fabs(phase + spread) < std::fabs(phase - spread)) ? phase + spread : phase - spread;

        double pitch_rotation[9], pitched_chord[3];
        makeAxisRotation(rest_frame.pitch_axis, pitch, pitch_rotation);
        rotate(pitch_rotation, chord, pitched_chord);

        // The yaw is the angle, about the parent axis, from the pitched chord to the target. When either is in line with the parent axis there is nothing for a yaw to do.
        double yaw = 0;
        double pitched_across[3], target_across[3];
        const double pitched_along = dot(pitched_chord, parent_axis);
        for (int i = 0; i < 3; i++) {
            pitched_across[i] = pitched_chord[i] - pitched_along * parent_axis[i];
            target_across[i] = target[i] - target_cosine_from_parent * parent_axis[i];
        }
        if (dot(pitched_across, pitched_across) > 1e-24 && dot(target_across, target_across) > 1e-24) {
            double across_cross[3];
            crossProduct(pitched_across, target_across, across_cross);
            yaw = std::atan2(dot(across_cross, parent_axis), dot(pitched_across, target_across));
        }

        PlantJointPose pose;
        pose.roll_degrees = static_cast<float>(rad2deg(roll));
        pose.pitch_degrees = static_cast<float>(rad2deg(pitch));
        pose.yaw_degrees = static_cast<float>(rad2deg(yaw));
        pose.length_scale = static_cast<float>(target_length / rest_length);
        // Angles and a stretch too small to move anything are left out, so that a shoot given the positions it already has is not thereby posed
        if (std::fabs(pose.roll_degrees) < 1e-6f) {
            pose.roll_degrees = 0;
        }
        if (std::fabs(pose.pitch_degrees) < 1e-6f) {
            pose.pitch_degrees = 0;
        }
        if (std::fabs(pose.yaw_degrees) < 1e-6f) {
            pose.yaw_degrees = 0;
        }
        if (std::fabs(pose.length_scale - 1.f) < 1e-7f) {
            pose.length_scale = 1;
        }
        return pose;
    }

    //! A transformation matrix moved by a rigid motion: motion * matrix
    std::array<float, 16> transformMatrix(const PoseRigidTransform &motion, const std::array<float, 16> &matrix) {
        std::array<float, 16> moved = matrix;
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 4; column++) {
                double value = motion.rotation[row * 3] * matrix[column] + motion.rotation[row * 3 + 1] * matrix[4 + column] + motion.rotation[row * 3 + 2] * matrix[8 + column];
                if (column == 3) {
                    value += motion.translation[row];
                }
                moved[row * 4 + column] = static_cast<float>(value);
            }
        }
        return moved;
    }

    //! Product of two affine transformation matrices (bottom row 0 0 0 1), in double precision
    std::array<double, 16> multiplyAffine(const std::array<double, 16> &left, const std::array<double, 16> &right) {
        std::array<double, 16> product{};
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 4; column++) {
                product[row * 4 + column] = left[row * 4] * right[column] + left[row * 4 + 1] * right[4 + column] + left[row * 4 + 2] * right[8 + column];
            }
            product[row * 4 + 3] += left[row * 4 + 3];
        }
        product[15] = 1.0;
        return product;
    }

    //! Inverse of an affine transformation matrix (bottom row 0 0 0 1), in double precision
    std::array<double, 16> invertAffine(const std::array<double, 16> &m, uint objID) {
        const double determinant = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
        if (determinant == 0.0 || !std::isfinite(determinant)) {
            helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPose): The transformation matrix of object " + std::to_string(objID) +
                                 " of a posed plant cannot be inverted, so a change made to that object since the plant was posed cannot be kept. The object has probably been scaled to zero size.");
        }
        std::array<double, 16> inverse{};
        inverse[0] = (m[5] * m[10] - m[6] * m[9]) / determinant;
        inverse[1] = (m[2] * m[9] - m[1] * m[10]) / determinant;
        inverse[2] = (m[1] * m[6] - m[2] * m[5]) / determinant;
        inverse[4] = (m[6] * m[8] - m[4] * m[10]) / determinant;
        inverse[5] = (m[0] * m[10] - m[2] * m[8]) / determinant;
        inverse[6] = (m[2] * m[4] - m[0] * m[6]) / determinant;
        inverse[8] = (m[4] * m[9] - m[5] * m[8]) / determinant;
        inverse[9] = (m[1] * m[8] - m[0] * m[9]) / determinant;
        inverse[10] = (m[0] * m[5] - m[1] * m[4]) / determinant;
        for (int row = 0; row < 3; row++) {
            inverse[row * 4 + 3] = -(inverse[row * 4] * m[3] + inverse[row * 4 + 1] * m[7] + inverse[row * 4 + 2] * m[11]);
        }
        inverse[15] = 1.0;
        return inverse;
    }

    std::array<double, 16> toDouble(const std::array<float, 16> &matrix) {
        std::array<double, 16> converted{};
        for (int i = 0; i < 16; i++) {
            converted[i] = matrix[i];
        }
        return converted;
    }

    std::array<float, 16> toFloat(const std::array<double, 16> &matrix) {
        std::array<float, 16> converted{};
        for (int i = 0; i < 16; i++) {
            converted[i] = static_cast<float>(matrix[i]);
        }
        return converted;
    }

    // ---- Context objects ---- //

    std::array<float, 16> getObjectMatrix(const Context *context_ptr, uint objID) {
        float matrix[16];
        context_ptr->getObjectTransformationMatrix(objID, matrix);
        std::array<float, 16> copy{};
        std::memcpy(copy.data(), matrix, sizeof(matrix));
        return copy;
    }

    void setObjectMatrix(Context *context_ptr, uint objID, const std::array<float, 16> &matrix) {
        float matrix_copy[16];
        std::memcpy(matrix_copy, matrix.data(), sizeof(matrix_copy));
        context_ptr->setObjectTransformationMatrix(objID, matrix_copy);
    }

    std::vector<std::array<float, 16>> getPrimitiveMatrices(const Context *context_ptr, uint objID) {
        const std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);
        std::vector<std::array<float, 16>> matrices(UUIDs.size());
        float matrix[16];
        for (size_t primitive = 0; primitive < UUIDs.size(); primitive++) {
            context_ptr->getPrimitiveTransformationMatrix(UUIDs.at(primitive), matrix);
            std::memcpy(matrices.at(primitive).data(), matrix, sizeof(matrix));
        }
        return matrices;
    }

    void setPrimitiveMatrices(Context *context_ptr, uint objID, const std::vector<std::array<float, 16>> &matrices) {
        const std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);
        float matrix[16];
        for (size_t primitive = 0; primitive < UUIDs.size(); primitive++) {
            std::memcpy(matrix, matrices.at(primitive).data(), sizeof(matrix));
            context_ptr->setPrimitiveTransformationMatrix(UUIDs.at(primitive), matrix);
        }
    }

    //! Hash of a set of transformation matrices (FNV-1a over their bytes), used to notice that an object's primitives have been changed
    uint64_t hashMatrices(const std::vector<std::array<float, 16>> &matrices) {
        uint64_t hash = 14695981039346656037ULL;
        for (const std::array<float, 16> &matrix: matrices) {
            unsigned char bytes[sizeof(float) * 16];
            std::memcpy(bytes, matrix.data(), sizeof(bytes));
            for (const unsigned char byte: bytes) {
                hash ^= byte;
                hash *= 1099511628211ULL;
            }
        }
        return hash;
    }

    PoseObjectState captureObjectState(const Context *context_ptr, uint objID) {
        PoseObjectState object_state;
        object_state.objID = objID;
        object_state.rest_object_matrix = getObjectMatrix(context_ptr, objID);
        object_state.rest_primitive_matrices = getPrimitiveMatrices(context_ptr, objID);
        object_state.posed_object_matrix = object_state.rest_object_matrix;
        object_state.posed_primitive_hash = hashMatrices(object_state.rest_primitive_matrices);
        return object_state;
    }

    void recordPosedObjectState(const Context *context_ptr, PoseObjectState &object_state) {
        object_state.posed_object_matrix = getObjectMatrix(context_ptr, object_state.objID);
        object_state.posed_primitive_hash = hashMatrices(getPrimitiveMatrices(context_ptr, object_state.objID));
    }

    //! Move an object that keeps its shape to where a rigid motion carries it from the way the plant grew
    void writeRigidObject(Context *context_ptr, const PoseObjectState &object_state, const PoseRigidTransform &motion) {
        if (motion.isIdentity()) {
            // Written back as recorded rather than multiplied by an identity, so that the object is restored bit for bit
            setObjectMatrix(context_ptr, object_state.objID, object_state.rest_object_matrix);
            if (!object_state.rest_shared_vertices.empty()) {
                context_ptr->setPolymeshObjectVertices(object_state.objID, object_state.rest_shared_vertices);
            }
            setPrimitiveMatrices(context_ptr, object_state.objID, object_state.rest_primitive_matrices);
            return;
        }
        setObjectMatrix(context_ptr, object_state.objID, transformMatrix(motion, object_state.rest_object_matrix));
        // The shared vertices of a leaf that droops go with it, before its primitives are written: the mesh re-derives its primitives from them, and the primitives written afterwards are the exact ones
        if (!object_state.rest_shared_vertices.empty()) {
            std::vector<vec3> moved_vertices(object_state.rest_shared_vertices.size());
            for (size_t vertex = 0; vertex < moved_vertices.size(); vertex++) {
                moved_vertices.at(vertex) = motion.applyToPoint(object_state.rest_shared_vertices.at(vertex));
            }
            context_ptr->setPolymeshObjectVertices(object_state.objID, moved_vertices);
        }
        std::vector<std::array<float, 16>> moved_matrices(object_state.rest_primitive_matrices.size());
        for (size_t primitive = 0; primitive < moved_matrices.size(); primitive++) {
            moved_matrices.at(primitive) = transformMatrix(motion, object_state.rest_primitive_matrices.at(primitive));
        }
        setPrimitiveMatrices(context_ptr, object_state.objID, moved_matrices);
    }

    //! Put a tube on a new centerline. If that is the centerline the plant grew with, its primitives are also put back exactly as they were recorded.
    /**
     * A tube is given its centerline rather than moved by a matrix, because the tube object keeps its own copy of the nodes, which a later change of radius or of shape is computed from. Rebuilding the
     * surface from the nodes does not necessarily reproduce the recorded primitives to the last bit, hence the second step.
     */
    void writeCenterlineTube(Context *context_ptr, const PoseObjectState &object_state, const std::vector<vec3> &centerline_vertices, bool centerline_is_as_grown) {
        context_ptr->setTubeNodes(object_state.objID, centerline_vertices);
        if (centerline_is_as_grown && object_state.rest_primitive_matrices.size() == context_ptr->getObjectPrimitiveCount(object_state.objID)) {
            setPrimitiveMatrices(context_ptr, object_state.objID, object_state.rest_primitive_matrices);
        }
    }

    // ---- Operations over a plant's stored geometry ---- //

    //! Records every value visited
    struct CaptureVisitor : PlantSpatialStateVisitor {
        std::vector<vec3> values;
        void point(vec3 &position) override {
            values.push_back(position);
        }
        void direction(vec3 &direction) override {
            values.push_back(direction);
        }
        void offset(vec3 &offset) override {
            values.push_back(offset);
        }
        void length(float &length) override {
            // Held in the same list as the vectors so that one list, in one order, describes the part
            values.push_back(make_vec3(length, 0, 0));
        }
    };

    //! Counts the values visited
    struct CountVisitor : PlantSpatialStateVisitor {
        size_t count = 0;
        void point(vec3 &) override {
            count++;
        }
        void direction(vec3 &) override {
            count++;
        }
        void offset(vec3 &) override {
            count++;
        }
        void length(float &) override {
            count++;
        }
    };

    //! Puts recorded values back, in the order they were recorded
    struct RestoreVisitor : PlantSpatialStateVisitor {
        explicit RestoreVisitor(const std::vector<vec3> &recorded_values) : values(recorded_values) {
        }
        const std::vector<vec3> &values;
        size_t next = 0;
        void point(vec3 &position) override {
            position = values.at(next++);
        }
        void direction(vec3 &direction) override {
            direction = values.at(next++);
        }
        void offset(vec3 &offset) override {
            offset = values.at(next++);
        }
        void length(float &length) override {
            length = values.at(next++).x;
        }
    };

    //! Moves every value by a rigid motion
    struct TransformVisitor : PlantSpatialStateVisitor {
        explicit TransformVisitor(const PoseRigidTransform &rigid_motion) : motion(rigid_motion) {
        }
        const PoseRigidTransform &motion;
        void point(vec3 &position) override {
            position = motion.applyToPoint(position);
        }
        void direction(vec3 &direction) override {
            direction = motion.applyToDirection(direction);
        }
        void offset(vec3 &offset) override {
            offset = motion.applyToOffset(offset);
        }
        void length(float &length) override {
            length = static_cast<float>(motion.scale * length);
        }
    };

    //! Folds a change made to a value since the pose last left it into the value recorded for the plant as it grew
    /**
     * The change is turned back by the rotation the part currently has, which is what the same change would have been had it been made to the unposed plant. A value that has not changed is left
     * exactly as recorded.
     */
    struct RebindVisitor : PlantSpatialStateVisitor {
        explicit RebindVisitor(PoseGroupState &group_state) : state(group_state) {
        }
        PoseGroupState &state;
        size_t next = 0;
        //! \param[in] stretched Whether the value is one the part's stretch applies to, so that a change to it is shrunk back as well as turned back
        void fold(const vec3 &current_value, bool stretched) {
            vec3 &posed_value = state.posed.at(next);
            if (current_value.x != posed_value.x || current_value.y != posed_value.y || current_value.z != posed_value.z) {
                vec3 change_as_grown = state.transform.applyInverseToDirection(current_value - posed_value);
                if (stretched) {
                    change_as_grown = change_as_grown / static_cast<float>(state.transform.scale);
                }
                state.rest.at(next) = state.rest.at(next) + change_as_grown;
                posed_value = current_value;
            }
            next++;
        }
        void point(vec3 &position) override {
            fold(position, true);
        }
        void direction(vec3 &direction) override {
            fold(direction, false);
        }
        void offset(vec3 &offset) override {
            fold(offset, true);
        }
        void length(float &length) override {
            vec3 &posed_value = state.posed.at(next);
            if (length != posed_value.x) {
                // A length is changed by a factor, which is the same factor on the unposed plant
                state.rest.at(next).x = (posed_value.x != 0.f) ? state.rest.at(next).x * (length / posed_value.x) : length / static_cast<float>(state.transform.scale);
                posed_value.x = length;
            }
            next++;
        }
    };

    //! Carries a translation of the whole plant, already applied to it, into the values recorded for the plant as it grew
    struct TranslateRestVisitor : PlantSpatialStateVisitor {
        TranslateRestVisitor(PoseGroupState &group_state, const vec3 &translation) : state(group_state), shift(translation) {
        }
        PoseGroupState &state;
        vec3 shift;
        size_t next = 0;
        void point(vec3 &position) override {
            state.rest.at(next) = state.rest.at(next) + shift;
            state.posed.at(next) = position;
            next++;
        }
        void direction(vec3 &direction) override {
            state.posed.at(next) = direction;
            next++;
        }
        void offset(vec3 &offset) override {
            state.posed.at(next) = offset;
            next++;
        }
        void length(float &length) override {
            state.posed.at(next) = make_vec3(length, 0, 0);
            next++;
        }
    };

    //! Apply an operation to the stored geometry of one rigid part of a plant
    void visitPoseGroup(const PoseGroupReference &group, PlantSpatialStateVisitor &visitor) {
        switch (group.kind) {
            case PoseGroupReference::SHOOT_ATTACHMENT:
                group.shoot->visitAttachmentSpatialState(visitor);
                break;
            case PoseGroupReference::SHOOT:
                group.shoot->visitShootSpatialState(visitor);
                break;
            case PoseGroupReference::INTERNODE:
                group.shoot->visitInternodeSpatialState(group.node_index, visitor);
                break;
            case PoseGroupReference::NODE:
                group.phytomer->visitNodeSpatialState(visitor);
                break;
            case PoseGroupReference::PETIOLE:
                group.phytomer->visitPetioleSpatialState(group.petiole_index, visitor);
                break;
            case PoseGroupReference::FLORAL_BUD:
                group.phytomer->visitFloralBudSpatialState(group.petiole_index, group.bud_index, visitor);
                break;
        }
    }

    //! A Context object that moves with one part of a plant
    struct PoseGroupObject {
        PoseGroupObject(uint a_objID, bool a_is_centerline_tube, int a_leaf_index) : objID(a_objID), is_centerline_tube(a_is_centerline_tube), leaf_index(a_leaf_index) {
        }
        uint objID;
        //! Whether the object is a tube drawn along a centerline the plant stores, which is redrawn on that centerline rather than moved
        bool is_centerline_tube;
        //! For a leaf, its index within its petiole, whose leaf base is where it is attached. -1 for any other object.
        int leaf_index;
    };

    //! Context objects that move with one part of a plant
    std::vector<PoseGroupObject> poseGroupObjects(const PoseGroupReference &group, const Context *context_ptr) {
        std::vector<PoseGroupObject> objects;
        if (group.kind == PoseGroupReference::PETIOLE) {
            const Phytomer &phytomer = *group.phytomer;
            if (group.petiole_index < phytomer.petiole_objIDs.size() && group.petiole_index < phytomer.petiole_vertices.size() && context_ptr->doesObjectExist(phytomer.petiole_objIDs.at(group.petiole_index))) {
                objects.emplace_back(phytomer.petiole_objIDs.at(group.petiole_index), true, -1);
            }
            if (group.petiole_index < phytomer.leaf_objIDs.size()) {
                const std::vector<uint> &petiole_leaf_objIDs = phytomer.leaf_objIDs.at(group.petiole_index);
                for (size_t leaf_index = 0; leaf_index < petiole_leaf_objIDs.size(); leaf_index++) {
                    if (context_ptr->doesObjectExist(petiole_leaf_objIDs.at(leaf_index))) {
                        objects.emplace_back(petiole_leaf_objIDs.at(leaf_index), false, static_cast<int>(leaf_index));
                    }
                }
            }
        } else if (group.kind == PoseGroupReference::FLORAL_BUD) {
            const Phytomer &phytomer = *group.phytomer;
            const FloralBud &fbud = phytomer.floral_buds.at(group.petiole_index).at(group.bud_index);
            const bool has_stored_centerline = group.petiole_index < phytomer.peduncle_vertices.size() && group.bud_index < phytomer.peduncle_vertices.at(group.petiole_index).size();
            for (const uint peduncle_objID: fbud.peduncle_objIDs) {
                if (context_ptr->doesObjectExist(peduncle_objID)) {
                    // A peduncle is one tube along the stored centerline. Were a bud ever to have more than one peduncle object, only the first could be that tube, and the rest are moved as rigid objects.
                    objects.emplace_back(peduncle_objID, has_stored_centerline && peduncle_objID == fbud.peduncle_objIDs.front(), -1);
                }
            }
            for (const uint inflorescence_objID: fbud.inflorescence_objIDs) {
                if (context_ptr->doesObjectExist(inflorescence_objID)) {
                    objects.emplace_back(inflorescence_objID, false, -1);
                }
            }
        }
        return objects;
    }

    //! Centerline, as the plant stores it, of the tube of a petiole or peduncle
    const std::vector<vec3> &poseGroupCenterline(const PoseGroupReference &group) {
        if (group.kind == PoseGroupReference::PETIOLE) {
            return group.phytomer->petiole_vertices.at(group.petiole_index);
        }
        return group.phytomer->peduncle_vertices.at(group.petiole_index).at(group.bud_index);
    }

} // namespace

bool Phytomer::petioleHasJoint(uint petiole_index) const {
    if (petiole_index >= petiole_vertices.size() || petiole_index >= petiole_length.size()) {
        return false;
    }
    // The stored length is what reports a petiole as gone: removeLeaf() clears it while leaving the centerline in place (see getPetioleLength(uint)).
    if (petiole_length.at(petiole_index) <= 0.f) {
        return false;
    }
    return centerlineLength(petiole_vertices.at(petiole_index)) > 0.f;
}

bool Phytomer::peduncleHasJoint(uint petiole_index, uint bud_index) const {
    if (petiole_index >= floral_buds.size() || bud_index >= floral_buds.at(petiole_index).size()) {
        return false;
    }
    // The peduncle centerline is kept after the flowers or fruit it carried are gone, so the state of the bud, not the centerline, says whether there is a peduncle.
    const BudState bud_state = floral_buds.at(petiole_index).at(bud_index).state;
    if (bud_state != BUD_FLOWER_CLOSED && bud_state != BUD_FLOWER_OPEN && bud_state != BUD_FRUITING) {
        return false;
    }
    if (petiole_index >= peduncle_vertices.size() || bud_index >= peduncle_vertices.at(petiole_index).size()) {
        return false;
    }
    return centerlineLength(peduncle_vertices.at(petiole_index).at(bud_index)) > 0.f;
}

Phytomer *PlantArchitecture::validatePlantJoint(uint plantID, const PlantJointID &joint, const std::string &function_name) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    const PlantInstance &plant_instance = plant_instances.at(plantID);

    // An index the joint's type does not use has to be zero. Two identifiers that differ only in such an index would otherwise name the same joint while being told apart wherever poses are kept,
    // and a pose set through one of them would be recorded without moving anything.
    const bool uses_shoot = joint.type != PLANT_JOINT_ROOT;
    const bool uses_node = joint.type == PLANT_JOINT_INTERNODE || joint.type == PLANT_JOINT_PETIOLE || joint.type == PLANT_JOINT_PEDUNCLE;
    const bool uses_petiole = joint.type == PLANT_JOINT_PETIOLE || joint.type == PLANT_JOINT_PEDUNCLE;
    const bool uses_bud = joint.type == PLANT_JOINT_PEDUNCLE;
    if ((!uses_shoot && joint.shootID != 0) || (!uses_node && joint.node_index != 0) || (!uses_petiole && joint.petiole_index != 0) || (!uses_bud && joint.bud_index != 0)) {
        helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): The joint identifier has a non-zero index that its joint type does not use (shootID " + std::to_string(joint.shootID) + ", node_index " +
                             std::to_string(joint.node_index) + ", petiole_index " + std::to_string(joint.petiole_index) + ", bud_index " + std::to_string(joint.bud_index) +
                             "). Indices that do not apply to the joint's type must be zero; this usually means an identifier was reused for a joint of another type without being cleared.");
    }

    if (joint.type == PLANT_JOINT_ROOT) {
        if (plant_instance.shoot_tree.empty()) {
            helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Plant " + std::to_string(plantID) + " has no shoots, so it has no joints.");
        }
        return nullptr;
    }

    validateShootID(plantID, joint.shootID, function_name);
    const std::shared_ptr<Shoot> &shoot = plant_instance.shoot_tree.at(joint.shootID);
    if (shoot->isPruned()) {
        helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Shoot " + std::to_string(joint.shootID) + " of plant " + std::to_string(plantID) + " has been pruned, so it has no joints.");
    }

    if (joint.type == PLANT_JOINT_SHOOT) {
        if (shoot->parent_shoot_ID < 0) {
            helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Shoot " + std::to_string(joint.shootID) + " of plant " + std::to_string(plantID) +
                                 " is a base shoot, which is not attached to a parent shoot and so has no PLANT_JOINT_SHOOT. Use PLANT_JOINT_ROOT to turn the plant at its base.");
        }
        return nullptr;
    }

    if (joint.node_index >= shoot->phytomers.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Shoot " + std::to_string(joint.shootID) + " does not have a node " + std::to_string(joint.node_index) + "; it has " + std::to_string(shoot->phytomers.size()) +
                             " nodes.");
    }
    Phytomer *phytomer = shoot->phytomers.at(joint.node_index).get();

    if (joint.type == PLANT_JOINT_PETIOLE) {
        if (!phytomer->petioleHasJoint(joint.petiole_index)) {
            helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Node " + std::to_string(joint.node_index) + " of shoot " + std::to_string(joint.shootID) + " has no petiole " + std::to_string(joint.petiole_index) +
                                 " with a length. A sessile leaf, or one that has been shed, has no petiole joint.");
        }
    } else if (joint.type == PLANT_JOINT_PEDUNCLE) {
        if (!phytomer->peduncleHasJoint(joint.petiole_index, joint.bud_index)) {
            helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Node " + std::to_string(joint.node_index) + " of shoot " + std::to_string(joint.shootID) + " has no peduncle for floral bud " + std::to_string(joint.bud_index) +
                                 " of petiole " + std::to_string(joint.petiole_index) + ". Only a bud that is flowering or fruiting has a peduncle joint.");
        }
    }
    return phytomer;
}

std::vector<PlantJointID> PlantArchitecture::getPlantJoints(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantJoints): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    const PlantInstance &plant_instance = plant_instances.at(plantID);

    std::vector<PlantJointID> joints;
    if (plant_instance.shoot_tree.empty()) {
        return joints;
    }

    PlantJointID root_joint;
    root_joint.type = PLANT_JOINT_ROOT;
    joints.push_back(root_joint);

    // Shoots are walked from each base shoot outward, descending into the child shoots of a node as soon as that node's own joints have been listed, so that every joint follows the ones that carry it.
    // The walk keeps its own stack rather than recursing so that a deeply branched tree cannot exhaust the call stack. Each entry is a shoot and the node at which to resume it.
    std::vector<std::pair<int, uint>> shoots_in_progress;
    for (auto shoot_iterator = plant_instance.shoot_tree.rbegin(); shoot_iterator != plant_instance.shoot_tree.rend(); ++shoot_iterator) {
        if ((*shoot_iterator)->parent_shoot_ID < 0) {
            shoots_in_progress.emplace_back((*shoot_iterator)->ID, 0u);
        }
    }

    while (!shoots_in_progress.empty()) {
        const int shootID = shoots_in_progress.back().first;
        const uint node_index = shoots_in_progress.back().second;
        const std::shared_ptr<Shoot> &shoot = plant_instance.shoot_tree.at(shootID);

        if (shoot->isPruned() || node_index >= shoot->phytomers.size()) {
            shoots_in_progress.pop_back();
            continue;
        }
        shoots_in_progress.back().second = node_index + 1;

        PlantJointID joint;
        joint.shootID = static_cast<uint>(shootID);

        if (node_index == 0 && shoot->parent_shoot_ID >= 0) {
            joint.type = PLANT_JOINT_SHOOT;
            joints.push_back(joint);
        }

        joint.type = PLANT_JOINT_INTERNODE;
        joint.node_index = node_index;
        joints.push_back(joint);

        const Phytomer &phytomer = *shoot->phytomers.at(node_index);

        joint.type = PLANT_JOINT_PETIOLE;
        for (uint petiole_index = 0; petiole_index < phytomer.petiole_vertices.size(); petiole_index++) {
            if (phytomer.petioleHasJoint(petiole_index)) {
                joint.petiole_index = petiole_index;
                joints.push_back(joint);
            }
        }

        joint.type = PLANT_JOINT_PEDUNCLE;
        for (uint petiole_index = 0; petiole_index < phytomer.floral_buds.size(); petiole_index++) {
            for (uint bud_index = 0; bud_index < phytomer.floral_buds.at(petiole_index).size(); bud_index++) {
                if (phytomer.peduncleHasJoint(petiole_index, bud_index)) {
                    joint.petiole_index = petiole_index;
                    joint.bud_index = bud_index;
                    joints.push_back(joint);
                }
            }
        }

        const auto children_of_node = shoot->childIDs.find(static_cast<int>(node_index));
        if (children_of_node != shoot->childIDs.end()) {
            for (auto child_iterator = children_of_node->second.rbegin(); child_iterator != children_of_node->second.rend(); ++child_iterator) {
                shoots_in_progress.emplace_back(*child_iterator, 0u);
            }
        }
    }

    return joints;
}

PlantJointFrame PlantArchitecture::getPlantJointFrame(uint plantID, const PlantJointID &joint) const {
    const Phytomer *phytomer = validatePlantJoint(plantID, joint, "getPlantJointFrame");
    const PlantInstance &plant_instance = plant_instances.at(plantID);

    // Direction of what a shoot grows from: the internode of the parent node it is attached to, or the vertical for a base shoot, which grows from the ground.
    auto shootParentAxis = [&plant_instance](const Shoot &shoot) {
        if (shoot.parent_shoot_ID < 0) {
            return make_vec3(0, 0, 1);
        }
        const std::shared_ptr<Shoot> &parent_shoot = plant_instance.shoot_tree.at(shoot.parent_shoot_ID);
        return centerlineTipAxis(parent_shoot->phytomers.at(shoot.parent_node_index)->getInternodeNodePositions(), "the internode shoot " + std::to_string(shoot.ID) + " grows from");
    };

    PlantJointFrame frame;

    if (joint.type == PLANT_JOINT_ROOT) {
        const Shoot *base_shoot = nullptr;
        for (const std::shared_ptr<Shoot> &shoot: plant_instance.shoot_tree) {
            if (shoot->parent_shoot_ID < 0 && !shoot->isPruned()) {
                base_shoot = shoot.get();
                break;
            }
        }
        if (base_shoot == nullptr) {
            helios_runtime_error("ERROR (PlantArchitecture::getPlantJointFrame): Every base shoot of plant " + std::to_string(plantID) + " has been pruned, so the plant has no direction at its base.");
        }
        frame.pivot = plant_instance.base_position;
        frame.parent_axis = make_vec3(0, 0, 1);
        frame.organ_axis = centerlineBaseAxis(base_shoot->phytomers.front()->getInternodeNodePositions(), "the first internode of the base shoot");
    } else if (joint.type == PLANT_JOINT_SHOOT) {
        const Shoot &shoot = *plant_instance.shoot_tree.at(joint.shootID);
        frame.pivot = shoot.shoot_internode_vertices.front().front();
        frame.parent_axis = shootParentAxis(shoot);
        frame.organ_axis = centerlineBaseAxis(shoot.phytomers.front()->getInternodeNodePositions(), "the first internode of shoot " + std::to_string(joint.shootID));
    } else if (joint.type == PLANT_JOINT_INTERNODE) {
        const Shoot &shoot = *plant_instance.shoot_tree.at(joint.shootID);
        const std::vector<vec3> internode_nodes = phytomer->getInternodeNodePositions();
        const std::string internode_description = "the internode of node " + std::to_string(joint.node_index) + " of shoot " + std::to_string(joint.shootID);
        frame.pivot = internode_nodes.front();
        if (joint.node_index == 0) {
            frame.parent_axis = shootParentAxis(shoot);
        } else {
            frame.parent_axis = centerlineTipAxis(shoot.phytomers.at(joint.node_index - 1)->getInternodeNodePositions(), "the internode before " + internode_description);
        }
        frame.organ_axis = centerlineBaseAxis(internode_nodes, internode_description);
        frame.length = centerlineLength(internode_nodes);
    } else if (joint.type == PLANT_JOINT_PETIOLE) {
        const std::vector<vec3> &petiole_nodes = phytomer->petiole_vertices.at(joint.petiole_index);
        frame.pivot = petiole_nodes.front();
        frame.parent_axis = centerlineTipAxis(phytomer->getInternodeNodePositions(), "the internode of node " + std::to_string(joint.node_index) + " of shoot " + std::to_string(joint.shootID));
        frame.organ_axis = centerlineBaseAxis(petiole_nodes, "petiole " + std::to_string(joint.petiole_index));
        frame.length = centerlineLength(petiole_nodes);
    } else if (joint.type == PLANT_JOINT_PEDUNCLE) {
        const std::vector<vec3> &peduncle_nodes = phytomer->peduncle_vertices.at(joint.petiole_index).at(joint.bud_index);
        // The floral bud's base is what a peduncle is swung about and re-seated to as the stem thickens, so it, rather than the first vertex of the centerline, is the pivot.
        frame.pivot = phytomer->floral_buds.at(joint.petiole_index).at(joint.bud_index).base_position;
        frame.parent_axis = centerlineTipAxis(phytomer->getInternodeNodePositions(), "the internode of node " + std::to_string(joint.node_index) + " of shoot " + std::to_string(joint.shootID));
        frame.organ_axis = centerlineBaseAxis(peduncle_nodes, "the peduncle of floral bud " + std::to_string(joint.bud_index));
        frame.length = centerlineLength(peduncle_nodes);
    } else {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantJointFrame): Unknown joint type.");
    }

    frame.pitch_axis = jointPitchAxis(frame.parent_axis, frame.organ_axis);
    return frame;
}

// ---- PoseRigidTransform ---- //

bool PoseRigidTransform::isRotationIdentity() const {
    return rotation[0] == 1.0 && rotation[1] == 0.0 && rotation[2] == 0.0 && rotation[3] == 0.0 && rotation[4] == 1.0 && rotation[5] == 0.0 && rotation[6] == 0.0 && rotation[7] == 0.0 && rotation[8] == 1.0;
}

bool PoseRigidTransform::isIdentity() const {
    return isRotationIdentity() && scale == 1.0 && translation[0] == 0.0 && translation[1] == 0.0 && translation[2] == 0.0;
}

bool PoseRigidTransform::operator==(const PoseRigidTransform &other) const {
    for (int i = 0; i < 9; i++) {
        if (rotation[i] != other.rotation[i]) {
            return false;
        }
    }
    return scale == other.scale && translation[0] == other.translation[0] && translation[1] == other.translation[1] && translation[2] == other.translation[2];
}

helios::vec3 PoseRigidTransform::applyToPoint(const helios::vec3 &position) const {
    return make_vec3(static_cast<float>(scale * (rotation[0] * position.x + rotation[1] * position.y + rotation[2] * position.z) + translation[0]),
                     static_cast<float>(scale * (rotation[3] * position.x + rotation[4] * position.y + rotation[5] * position.z) + translation[1]),
                     static_cast<float>(scale * (rotation[6] * position.x + rotation[7] * position.y + rotation[8] * position.z) + translation[2]));
}

helios::vec3 PoseRigidTransform::applyToOffset(const helios::vec3 &offset) const {
    return make_vec3(static_cast<float>(scale * (rotation[0] * offset.x + rotation[1] * offset.y + rotation[2] * offset.z)), static_cast<float>(scale * (rotation[3] * offset.x + rotation[4] * offset.y + rotation[5] * offset.z)),
                     static_cast<float>(scale * (rotation[6] * offset.x + rotation[7] * offset.y + rotation[8] * offset.z)));
}

PoseRigidTransform PoseRigidTransform::rigidMotionOfBodyAt(const helios::vec3 &anchor) const {
    // The body is turned about the origin and then translated so that its anchor lands where the stretched organ carries it: R x + (s R a + t - R a)
    PoseRigidTransform body_motion = *this;
    body_motion.scale = 1;
    for (int row = 0; row < 3; row++) {
        body_motion.translation[row] = translation[row] + (scale - 1.0) * (rotation[row * 3] * anchor.x + rotation[row * 3 + 1] * anchor.y + rotation[row * 3 + 2] * anchor.z);
    }
    return body_motion;
}

helios::vec3 PoseRigidTransform::applyToDirection(const helios::vec3 &direction) const {
    return make_vec3(static_cast<float>(rotation[0] * direction.x + rotation[1] * direction.y + rotation[2] * direction.z), static_cast<float>(rotation[3] * direction.x + rotation[4] * direction.y + rotation[5] * direction.z),
                     static_cast<float>(rotation[6] * direction.x + rotation[7] * direction.y + rotation[8] * direction.z));
}

helios::vec3 PoseRigidTransform::applyInverseToDirection(const helios::vec3 &direction) const {
    // The inverse of a rotation is its transpose
    return make_vec3(static_cast<float>(rotation[0] * direction.x + rotation[3] * direction.y + rotation[6] * direction.z), static_cast<float>(rotation[1] * direction.x + rotation[4] * direction.y + rotation[7] * direction.z),
                     static_cast<float>(rotation[2] * direction.x + rotation[5] * direction.y + rotation[8] * direction.z));
}

PoseRigidTransform PoseRigidTransform::composedWith(const PoseRigidTransform &inner) const {
    PoseRigidTransform composed;
    multiplyRotations(rotation, inner.rotation, composed.rotation);
    // Uniform stretches commute with rotations, so the two simply multiply
    composed.scale = scale * inner.scale;
    for (int row = 0; row < 3; row++) {
        composed.translation[row] = scale * (rotation[row * 3] * inner.translation[0] + rotation[row * 3 + 1] * inner.translation[1] + rotation[row * 3 + 2] * inner.translation[2]) + translation[row];
    }
    return composed;
}

// ---- Stored geometry of a plant ---- //

void Phytomer::visitPetioleSpatialState(uint petiole_index, PlantSpatialStateVisitor &visitor) {
    if (petiole_index < petiole_vertices.size()) {
        for (vec3 &vertex: petiole_vertices.at(petiole_index)) {
            visitor.point(vertex);
        }
    }
    if (petiole_index < leaf_bases.size()) {
        for (vec3 &leaf_base: leaf_bases.at(petiole_index)) {
            visitor.point(leaf_base);
        }
    }
    if (petiole_index < petiole_axis_initial.size()) {
        visitor.direction(petiole_axis_initial.at(petiole_index));
    }
    if (petiole_index < petiole_rotation_axis.size()) {
        visitor.direction(petiole_rotation_axis.at(petiole_index));
    }
    // The rest shape is held as offsets from the petiole base, so it turns and stretches with the petiole but is not translated
    if (petiole_index < petiole_rest_offsets.size()) {
        for (vec3 &rest_offset: petiole_rest_offsets.at(petiole_index)) {
            visitor.offset(rest_offset);
        }
    }
    if (petiole_index < petiole_length.size()) {
        visitor.length(petiole_length.at(petiole_index));
    }
    if (petiole_index < leaf_objIDs.size()) {
        for (const uint leaf_objID: leaf_objIDs.at(petiole_index)) {
            const auto stalk_base = leaf_stalk_base_placed.find(leaf_objID);
            if (stalk_base != leaf_stalk_base_placed.end()) {
                visitor.point(stalk_base->second);
            }
        }
    }
}

void Phytomer::visitFloralBudSpatialState(uint petiole_index, uint bud_index, PlantSpatialStateVisitor &visitor) {
    FloralBud &fbud = floral_buds.at(petiole_index).at(bud_index);
    visitor.point(fbud.base_position);
    for (vec3 &inflorescence_base: fbud.inflorescence_bases) {
        visitor.point(inflorescence_base);
    }
    visitor.direction(fbud.bending_axis);
    visitor.direction(fbud.contact_swing_axis);
    for (AxisRotation &fruit_rotation: fbud.inflorescence_rotation) {
        visitor.direction(fruit_rotation.peduncle_axis);
        visitor.direction(fruit_rotation.contact_tilt_axis);
    }
    if (petiole_index < peduncle_vertices.size() && bud_index < peduncle_vertices.at(petiole_index).size()) {
        for (vec3 &vertex: peduncle_vertices.at(petiole_index).at(bud_index)) {
            visitor.point(vertex);
        }
    }
}

void Phytomer::visitNodeSpatialState(PlantSpatialStateVisitor &visitor) {
    visitor.direction(inflorescence_bending_axis);
    visitor.point(seated_node_position);
}

void Phytomer::resettlePetioleUnderGravity(uint petiole_index) {
    if (petiole_index >= petiole_vertices.size() || petiole_index >= leaf_objIDs.size()) {
        return;
    }
    // Both steps skip an organ that has not grown since it was last deflected, which is the case here, so what they compare against is cleared first.
    if (petiole_index < petiole_bend_state.size()) {
        petiole_bend_state.at(petiole_index) = make_vec3(-1, 0, 0);
    }
    if (petiole_index < leaf_last_deformed_scale.size()) {
        for (float &last_deformed_scale: leaf_last_deformed_scale.at(petiole_index)) {
            last_deformed_scale = -1.f;
        }
    }
    // The petiole first: it carries its leaves to where they hang from, and a blade that had already drooped would be turned by that as a rigid shape rather than drooping where it ends up.
    bendPetioleUnderLeafWeight(petiole_index);
    for (uint leaf_index = 0; leaf_index < leaf_objIDs.at(petiole_index).size(); leaf_index++) {
        deformLeafUnderSelfWeight(petiole_index, leaf_index);
    }
}

void Phytomer::rotatePetiole(uint petiole_index, const AxisRotation &rotation) {
    if (petiole_index >= petiole_vertices.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer::rotatePetiole): Invalid petiole index.");
    }
    if (petiole_vertices.at(petiole_index).empty()) {
        return;
    }
    if (!std::isfinite(rotation.pitch) || !std::isfinite(rotation.yaw) || !std::isfinite(rotation.roll)) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer::rotatePetiole): The rotation has an angle that is not a finite number.");
    }
    if (rotation.pitch == 0.f && rotation.yaw == 0.f && rotation.roll == 0.f) {
        return;
    }

    const vec3 base = petiole_vertices.at(petiole_index).front();

    // One turn of the petiole and its leaves, as a rigid body, about a line through the petiole base
    auto turnAboutBase = [this, petiole_index, &base](float angle, vec3 axis) {
        if (angle == 0.f) {
            return;
        }
        axis.normalize();

        if (petiole_index < petiole_objIDs.size() && context_ptr->doesObjectExist(petiole_objIDs.at(petiole_index))) {
            context_ptr->rotateObject(petiole_objIDs.at(petiole_index), angle, base, axis);
        }
        if (petiole_index < leaf_objIDs.size()) {
            for (const uint leaf_objID: leaf_objIDs.at(petiole_index)) {
                if (context_ptr->doesObjectExist(leaf_objID)) {
                    context_ptr->rotateObject(leaf_objID, angle, base, axis);
                }
            }
        }

        // Everything the phytomer stores for the petiole is turned through the one function that lists it (visitPetioleSpatialState()), so that nothing stored for the petiole can be left behind
        PoseRigidTransform turn;
        makeAxisRotation(axis, angle, turn.rotation);
        const double pivot[3] = {base.x, base.y, base.z};
        for (int row = 0; row < 3; row++) {
            turn.translation[row] = pivot[row] - (turn.rotation[row * 3] * pivot[0] + turn.rotation[row * 3 + 1] * pivot[1] + turn.rotation[row * 3 + 2] * pivot[2]);
        }
        TransformVisitor transform(turn);
        visitPetioleSpatialState(petiole_index, transform);

        // The rest shape has turned with the petiole, and the bent shape has to be recomputed from it: gravity does not turn with the petiole, so the rigidly rotated bent shape is no longer in balance.
        if (petiole_index < petiole_bend_state.size()) {
            petiole_bend_state.at(petiole_index) = make_vec3(-1, 0, 0);
        }
    };

    // pitch -- about the axis the petiole was tilted away from the internode about when it was built, which turns with the petiole through any yaw or roll. A positive angle about that axis is, as at
    // construction, away from the internode, so the sign of the pitch says which way the petiole goes.
    if (rotation.pitch != 0.f) {
        if (petiole_index >= petiole_rotation_axis.size() || petiole_rotation_axis.at(petiole_index).magnitude() < 1e-6f) {
            helios_runtime_error("ERROR (PlantArchitecture::Phytomer::rotatePetiole): Petiole " + std::to_string(petiole_index) + " has no pitch axis, so it cannot be pitched. This is the case for a phytomer built without petioles.");
        }
        turnAboutBase(rotation.pitch, petiole_rotation_axis.at(petiole_index));
        if (petiole_index < petiole_pitch.size()) {
            // The stored pitch is the angle from the internode, carried with either sign (the construction uses its magnitude), so the new angle is given the sign the old one had
            const float pitch_from_internode = std::fabs(std::fabs(petiole_pitch.at(petiole_index)) + rotation.pitch);
            petiole_pitch.at(petiole_index) = std::copysign(pitch_from_internode, petiole_pitch.at(petiole_index));
        }
    }

    // yaw -- around the internode (azimuth around the stem)
    if (rotation.yaw != 0.f) {
        const vec3 internode_axis = getInternodeAxisVector(1.f);
        if (internode_axis.magnitude() < 1e-6f) {
            helios_runtime_error("ERROR (PlantArchitecture::Phytomer::rotatePetiole): The internode has no direction to turn the petiole around.");
        }
        turnAboutBase(rotation.yaw, internode_axis);
    }

    // roll -- about the petiole's own length axis, as it now is
    if (rotation.roll != 0.f) {
        const vec3 petiole_axis = getPetioleAxisVector(1.f, petiole_index);
        if (petiole_axis.magnitude() < 1e-6f) {
            helios_runtime_error("ERROR (PlantArchitecture::Phytomer::rotatePetiole): The petiole has no length, so it has no axis to roll about.");
        }
        turnAboutBase(rotation.roll, petiole_axis);
    }
}

void Shoot::visitInternodeSpatialState(uint node_index, PlantSpatialStateVisitor &visitor) {
    for (vec3 &vertex: shoot_internode_vertices.at(node_index)) {
        visitor.point(vertex);
    }
}

void Shoot::visitAttachmentSpatialState(PlantSpatialStateVisitor &visitor) {
    // The offset of the shoot's base from the axis of its parent, and the parent direction it was computed for (see updateShootNodes()). The offset is a fraction of the parent's radius, which no pose
    // changes, so it turns without stretching.
    visitor.direction(cached_base_offset);
    visitor.direction(last_parent_internode_axis);
}

void Shoot::visitShootSpatialState(PlantSpatialStateVisitor &visitor) {
    visitor.point(base_position);
    visitor.direction(initial_heading_horizontal);
    visitor.direction(radial_outward_axis);
}

// ---- Pose of a plant ---- //

bool PlantArchitecture::forEachPoseGroup(PlantInstance &plant_instance, PlantPoseState &pose_state, bool create_missing, const std::function<void(PoseGroupReference &)> &callback) const {
    // Resize a level of the pose's structure to match the plant, or report that it does not match
    auto matchSize = [create_missing](auto &pose_entries, size_t plant_size) {
        if (pose_entries.size() == plant_size) {
            return true;
        }
        if (!create_missing) {
            return false;
        }
        pose_entries.resize(plant_size);
        return true;
    };

    if (!matchSize(pose_state.shoots, plant_instance.shoot_tree.size())) {
        return false;
    }

    for (size_t shootID = 0; shootID < plant_instance.shoot_tree.size(); shootID++) {
        Shoot &shoot = *plant_instance.shoot_tree.at(shootID);
        PoseShootState &shoot_state = pose_state.shoots.at(shootID);

        PoseGroupReference group;
        group.shoot = &shoot;

        group.kind = PoseGroupReference::SHOOT_ATTACHMENT;
        group.state = &shoot_state.attachment;
        callback(group);

        group.kind = PoseGroupReference::SHOOT;
        group.state = &shoot_state.shoot;
        callback(group);

        const size_t node_count = shoot.phytomers.size();
        if (!matchSize(shoot_state.internodes, node_count) || !matchSize(shoot_state.nodes, node_count) || !matchSize(shoot_state.petioles, node_count) || !matchSize(shoot_state.floral_buds, node_count)) {
            return false;
        }

        for (uint node_index = 0; node_index < node_count; node_index++) {
            Phytomer &phytomer = *shoot.phytomers.at(node_index);
            group.phytomer = &phytomer;
            group.node_index = node_index;
            group.petiole_index = 0;
            group.bud_index = 0;

            group.kind = PoseGroupReference::INTERNODE;
            group.state = &shoot_state.internodes.at(node_index);
            callback(group);

            group.kind = PoseGroupReference::NODE;
            group.state = &shoot_state.nodes.at(node_index);
            callback(group);

            // A phytomer whose leaves have been removed can keep its petiole centerlines, and one with sessile leaves has leaves without petiole tubes, so the longer of the two lists is the one that covers every petiole
            const size_t petiole_count = std::max(phytomer.petiole_vertices.size(), phytomer.leaf_objIDs.size());
            if (!matchSize(shoot_state.petioles.at(node_index), petiole_count)) {
                return false;
            }
            group.kind = PoseGroupReference::PETIOLE;
            for (uint petiole_index = 0; petiole_index < petiole_count; petiole_index++) {
                group.petiole_index = petiole_index;
                group.state = &shoot_state.petioles.at(node_index).at(petiole_index);
                callback(group);
            }

            if (!matchSize(shoot_state.floral_buds.at(node_index), phytomer.floral_buds.size())) {
                return false;
            }
            group.kind = PoseGroupReference::FLORAL_BUD;
            for (uint petiole_index = 0; petiole_index < phytomer.floral_buds.size(); petiole_index++) {
                if (!matchSize(shoot_state.floral_buds.at(node_index).at(petiole_index), phytomer.floral_buds.at(petiole_index).size())) {
                    return false;
                }
                group.petiole_index = petiole_index;
                for (uint bud_index = 0; bud_index < phytomer.floral_buds.at(petiole_index).size(); bud_index++) {
                    group.bud_index = bud_index;
                    group.state = &shoot_state.floral_buds.at(node_index).at(petiole_index).at(bud_index);
                    callback(group);
                }
            }
        }
    }
    return true;
}

void PlantArchitecture::capturePlantRestPose(PlantInstance &plant_instance) const {
    // While the plant is posed each branch is kept on the side of its parent it is on, by carrying its recorded offset from the parent's axis with the parent (see Shoot::base_offset_pinned). A
    // branch that has not been re-seated since it was created has no offset recorded yet, so it is recorded here from where the branch actually is, before the plant is recorded as it grew.
    for (const std::shared_ptr<Shoot> &shoot: plant_instance.shoot_tree) {
        if (shoot->isPruned() || shoot->parent_shoot_ID < 0 || shoot->is_axis_continuation || shoot->last_parent_internode_axis.magnitude() < 2.f) {
            continue;
        }
        const std::shared_ptr<Shoot> &parent_shoot = plant_instance.shoot_tree.at(shoot->parent_shoot_ID);
        if (parent_shoot->isPruned() || shoot->parent_node_index >= parent_shoot->phytomers.size()) {
            continue;
        }
        shoot->cached_base_offset = shoot->shoot_internode_vertices.front().front() - parent_shoot->shoot_internode_vertices.at(shoot->parent_node_index).back();
        shoot->last_parent_internode_axis = parent_shoot->phytomers.at(shoot->parent_node_index)->getInternodeAxisVector(1.f);
        shoot->last_parent_node_radius = parent_shoot->phytomers.at(shoot->parent_node_index)->getInternodeRadius(1.f);
    }

    plant_instance.pose_state = std::make_shared<PlantPoseState>();
    PlantPoseState &pose_state = *plant_instance.pose_state;

    forEachPoseGroup(plant_instance, pose_state, true, [this](PoseGroupReference &group) {
        CaptureVisitor capture;
        visitPoseGroup(group, capture);
        group.state->rest = capture.values;
        group.state->posed = capture.values;
        for (const PoseGroupObject &object: poseGroupObjects(group, context_ptr)) {
            group.state->objects.push_back(captureObjectState(context_ptr, object.objID));
            // A leaf that droops, or hangs from a petiole that bends, is rewritten through its mesh's shared vertices each time it settles, so those are recorded too
            const bool leaf_can_resettle = object.leaf_index >= 0 && (group.phytomer->leaf_flexibility > 0.f || group.phytomer->petiole_flexibility > 0.f);
            if (leaf_can_resettle && context_ptr->getObjectType(object.objID) == OBJECT_TYPE_POLYMESH && !context_ptr->getPolymeshObjectFaces(object.objID).empty()) {
                group.state->objects.back().rest_shared_vertices = context_ptr->getPolymeshObjectVertices(object.objID);
            }
        }
    });

    // Each branch keeps the side of its parent it is on for as long as the plant is posed (see Shoot::base_offset_pinned)
    pose_state.base_offset_pinned_before.resize(plant_instance.shoot_tree.size());
    for (size_t shootID = 0; shootID < plant_instance.shoot_tree.size(); shootID++) {
        pose_state.base_offset_pinned_before.at(shootID) = plant_instance.shoot_tree.at(shootID)->base_offset_pinned;
        plant_instance.shoot_tree.at(shootID)->base_offset_pinned = true;
    }

    for (size_t shootID = 0; shootID < plant_instance.shoot_tree.size(); shootID++) {
        const Shoot &shoot = *plant_instance.shoot_tree.at(shootID);
        if (!shoot.isPruned() && context_ptr->doesObjectExist(shoot.internode_tube_objID)) {
            pose_state.shoots.at(shootID).internode_tube = captureObjectState(context_ptr, shoot.internode_tube_objID);
            pose_state.shoots.at(shootID).has_internode_tube = true;
        }
    }
}

bool PlantArchitecture::rebindPlantRestPose(PlantInstance &plant_instance) const {
    PlantPoseState &pose_state = *plant_instance.pose_state;

    bool structure_unchanged = true;
    const bool group_structure_unchanged = forEachPoseGroup(plant_instance, pose_state, false, [this, &structure_unchanged](PoseGroupReference &group) {
        if (!structure_unchanged) {
            return;
        }
        PoseGroupState &state = *group.state;

        CountVisitor counter;
        visitPoseGroup(group, counter);
        const std::vector<PoseGroupObject> objects = poseGroupObjects(group, context_ptr);
        if (counter.count != state.rest.size() || objects.size() != state.objects.size()) {
            structure_unchanged = false;
            return;
        }
        for (size_t object = 0; object < objects.size(); object++) {
            if (objects.at(object).objID != state.objects.at(object).objID) {
                structure_unchanged = false;
                return;
            }
        }

        RebindVisitor rebind(state);
        visitPoseGroup(group, rebind);

        for (size_t object = 0; object < objects.size(); object++) {
            PoseObjectState &object_state = state.objects.at(object);
            const std::array<float, 16> current_object_matrix = getObjectMatrix(context_ptr, object_state.objID);
            const std::vector<std::array<float, 16>> current_primitive_matrices = getPrimitiveMatrices(context_ptr, object_state.objID);
            const uint64_t current_primitive_hash = hashMatrices(current_primitive_matrices);
            if (current_object_matrix == object_state.posed_object_matrix && current_primitive_hash == object_state.posed_primitive_hash) {
                continue;
            }

            if (objects.at(object).is_centerline_tube) {
                // The tube is redrawn from the stored centerline, which has just been folded in. The primitives recorded for the plant as it grew no longer describe it, so they are dropped, and the tube
                // of the unposed plant is from now on the one its centerline gives.
                object_state.rest_primitive_matrices.clear();
            } else {
                // Everything done to the object by the pose and by gravity is in the change of its own matrix, so undoing that change gives the object as it would be on the unposed plant
                const std::array<double, 16> motion_since_rest = multiplyAffine(toDouble(object_state.posed_object_matrix), invertAffine(toDouble(object_state.rest_object_matrix), object_state.objID));
                const std::array<double, 16> motion_back_to_rest = invertAffine(motion_since_rest, object_state.objID);
                object_state.rest_object_matrix = toFloat(multiplyAffine(motion_back_to_rest, toDouble(current_object_matrix)));
                object_state.rest_primitive_matrices.resize(current_primitive_matrices.size());
                for (size_t primitive = 0; primitive < current_primitive_matrices.size(); primitive++) {
                    object_state.rest_primitive_matrices.at(primitive) = toFloat(multiplyAffine(motion_back_to_rest, toDouble(current_primitive_matrices.at(primitive))));
                }
                // The droop of a blade is not in its matrix, so what was just recorded is the changed leaf with the droop of its posed orientation. The shared vertices recorded before the change no
                // longer describe it, and the leaf has to be left to settle wherever the pose next puts it, including back where it grew.
                object_state.rest_shared_vertices.clear();
                state.resettle_at_rest = true;
            }
            object_state.posed_object_matrix = current_object_matrix;
            object_state.posed_primitive_hash = current_primitive_hash;
        }
    });
    if (!group_structure_unchanged || !structure_unchanged) {
        return false;
    }

    // Internode tubes are redrawn from the stored node positions. One that has been changed by something else, for example by a change of radius, keeps that change, and only loses its recorded primitives.
    for (size_t shootID = 0; shootID < plant_instance.shoot_tree.size(); shootID++) {
        const Shoot &shoot = *plant_instance.shoot_tree.at(shootID);
        PoseObjectState &tube_state = pose_state.shoots.at(shootID).internode_tube;
        const bool tube_exists = !shoot.isPruned() && context_ptr->doesObjectExist(shoot.internode_tube_objID);
        if (tube_exists != pose_state.shoots.at(shootID).has_internode_tube || (tube_exists && shoot.internode_tube_objID != tube_state.objID)) {
            return false;
        }
        if (!tube_exists) {
            continue;
        }
        const std::array<float, 16> current_object_matrix = getObjectMatrix(context_ptr, tube_state.objID);
        const uint64_t current_primitive_hash = hashMatrices(getPrimitiveMatrices(context_ptr, tube_state.objID));
        if (!(current_object_matrix == tube_state.posed_object_matrix) || current_primitive_hash != tube_state.posed_primitive_hash) {
            tube_state.rest_primitive_matrices.clear();
            tube_state.posed_object_matrix = current_object_matrix;
            tube_state.posed_primitive_hash = current_primitive_hash;
        }
    }
    return true;
}

void PlantArchitecture::computeShootPoseTransforms(uint plantID, uint shootID, const PoseRigidTransform &carrier_transform) const {
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    const PlantPoseState &pose_state = *plant_instance.pose_state;
    const Shoot &shoot = *plant_instance.shoot_tree.at(shootID);
    PoseShootState &shoot_state = plant_instance.pose_state->shoots.at(shootID);

    // Motion of an organ's own centerline, and of what grows from its tip, when a pose has been set for the joint at its base. The joint's axes are read from the plant, which the caller has put back
    // as it grew. The centerline is stretched about the joint and then turned; what grows from the tip is carried to where the tip goes and turned, but not stretched.
    auto withJoint = [this, plantID, &pose_state](const PoseRigidTransform &carrier, const PlantJointID &joint, const vec3 &rest_tip, PoseRigidTransform &beyond_tip) {
        const auto joint_pose = pose_state.joint_poses.find(joint);
        if (joint_pose == pose_state.joint_poses.end()) {
            beyond_tip = carrier;
            return carrier;
        }
        const PlantJointFrame rest_frame = getPlantJointFrame(plantID, joint);
        const PoseRigidTransform turned = carrier.composedWith(makeJointRotation(rest_frame, joint_pose->second));
        if (joint_pose->second.length_scale == 1.f) {
            beyond_tip = turned;
            return turned;
        }
        beyond_tip = turned.composedWith(makeTipTranslation(rest_frame, joint_pose->second, rest_tip));
        return turned.composedWith(makeJointStretch(rest_frame, joint_pose->second));
    };

    PlantJointID joint;
    joint.shootID = shootID;

    // How the shoot sits on its parent turns with the parent, not with the shoot's own joint
    shoot_state.attachment.pending_transform = carrier_transform;

    // Motion of everything from the current node outward
    PoseRigidTransform transform = carrier_transform;
    PoseRigidTransform unused_beyond_tip;
    if (shoot.parent_shoot_ID >= 0 && !shoot.isPruned()) {
        joint.type = PLANT_JOINT_SHOOT;
        transform = withJoint(transform, joint, nullorigin, unused_beyond_tip);
    }
    shoot_state.shoot.pending_transform = transform;

    for (uint node_index = 0; node_index < shoot.phytomers.size(); node_index++) {
        const Phytomer &phytomer = *shoot.phytomers.at(node_index);
        joint.node_index = node_index;
        joint.petiole_index = 0;
        joint.bud_index = 0;

        // The internode's own nodes are stretched if its length is posed; the node at its tip, and everything from there on, is carried to where the tip goes
        joint.type = PLANT_JOINT_INTERNODE;
        PoseRigidTransform beyond_internode_tip;
        shoot_state.internodes.at(node_index).pending_transform = withJoint(transform, joint, shoot.shoot_internode_vertices.at(node_index).back(), beyond_internode_tip);
        transform = beyond_internode_tip;
        shoot_state.nodes.at(node_index).pending_transform = transform;

        // A petiole's leaves are attached along it, not only at its tip, so each is carried by where its own base goes (see evaluatePlantPose())
        joint.type = PLANT_JOINT_PETIOLE;
        for (uint petiole_index = 0; petiole_index < shoot_state.petioles.at(node_index).size(); petiole_index++) {
            joint.petiole_index = petiole_index;
            shoot_state.petioles.at(node_index).at(petiole_index).pending_transform = withJoint(transform, joint, nullorigin, unused_beyond_tip);
        }

        joint.type = PLANT_JOINT_PEDUNCLE;
        for (uint petiole_index = 0; petiole_index < phytomer.floral_buds.size(); petiole_index++) {
            joint.petiole_index = petiole_index;
            for (uint bud_index = 0; bud_index < phytomer.floral_buds.at(petiole_index).size(); bud_index++) {
                joint.bud_index = bud_index;
                shoot_state.floral_buds.at(node_index).at(petiole_index).at(bud_index).pending_transform = withJoint(transform, joint, nullorigin, unused_beyond_tip);
            }
        }

        const auto children_of_node = shoot.childIDs.find(static_cast<int>(node_index));
        if (children_of_node != shoot.childIDs.end()) {
            for (const int child_shootID: children_of_node->second) {
                computeShootPoseTransforms(plantID, static_cast<uint>(child_shootID), transform);
            }
        }
    }
}

void PlantArchitecture::computePlantPoseTransforms(uint plantID) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    PlantPoseState &pose_state = *plant_instance.pose_state;

    // The stored geometry is cheap to put back, so all of it is put back as the plant grew, which is what the axes of the joints have to be read from
    forEachPoseGroup(plant_instance, pose_state, false, [](PoseGroupReference &group) {
        RestoreVisitor restore(group.state->rest);
        visitPoseGroup(group, restore);
        group.state->pending_transform = PoseRigidTransform();
    });

    PoseRigidTransform root_transform;
    PlantJointID root_joint;
    root_joint.type = PLANT_JOINT_ROOT;
    const auto root_pose = pose_state.joint_poses.find(root_joint);
    if (root_pose != pose_state.joint_poses.end()) {
        root_transform = makeJointRotation(getPlantJointFrame(plantID, root_joint), root_pose->second);
    }
    for (size_t shootID = 0; shootID < plant_instance.shoot_tree.size(); shootID++) {
        if (plant_instance.shoot_tree.at(shootID)->parent_shoot_ID < 0) {
            computeShootPoseTransforms(plantID, static_cast<uint>(shootID), root_transform);
        }
    }
}

void PlantArchitecture::evaluatePlantPose(uint plantID) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    PlantPoseState &pose_state = *plant_instance.pose_state;

    // Every part of the plant is moved from where it grew, never from where it currently is, so that the result depends only on the joint poses that are set and not on the poses the plant passed
    // through.
    computePlantPoseTransforms(plantID);

    // Move each part. A part whose motion is the one it already has is put back as the pose last left it and is not written to the Context, so that changing one joint rewrites only the geometry
    // that joint carries. (Checking the whole plant for changes made by other calls, before this, still reads every primitive of the plant.)
    std::vector<PoseGroupReference> moved_groups;
    std::set<Shoot *> shoots_with_moved_internodes;
    forEachPoseGroup(plant_instance, pose_state, false, [this, &moved_groups, &shoots_with_moved_internodes](PoseGroupReference &group) {
        PoseGroupState &state = *group.state;
        if (state.pending_transform == state.transform) {
            RestoreVisitor restore_posed(state.posed);
            visitPoseGroup(group, restore_posed);
            return;
        }

        const bool is_as_grown = state.pending_transform.isIdentity();

        // A leaf is carried by where its base goes. The bases are read here, while the stored geometry is still as the plant grew.
        const std::vector<PoseGroupObject> objects = poseGroupObjects(group, context_ptr);
        std::vector<PoseRigidTransform> object_motions(objects.size(), state.pending_transform);
        if (state.pending_transform.scale != 1.0) {
            for (size_t object = 0; object < objects.size(); object++) {
                if (objects.at(object).is_centerline_tube) {
                    continue;
                }
                if (group.kind != PoseGroupReference::PETIOLE || objects.at(object).leaf_index < 0) {
                    helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPose): An object other than a leaf is attached to an organ whose length is posed, which should not be possible. This is a bug in the plant architecture plug-in.");
                }
                object_motions.at(object) = state.pending_transform.rigidMotionOfBodyAt(group.phytomer->leaf_bases.at(group.petiole_index).at(objects.at(object).leaf_index));
            }
        }

        if (!is_as_grown) {
            TransformVisitor transform(state.pending_transform);
            visitPoseGroup(group, transform);
            if (group.kind == PoseGroupReference::SHOOT) {
                // The heading the shoot's growth is steered back toward is by definition horizontal, so a tilt is taken out of it again
                vec3 &heading = group.shoot->initial_heading_horizontal;
                heading.z = 0;
                if (heading.magnitude() > 1e-6f) {
                    heading.normalize();
                } else {
                    heading = make_vec3(0, 0, 0);
                }
            }
        }

        for (size_t object = 0; object < objects.size(); object++) {
            if (objects.at(object).is_centerline_tube) {
                writeCenterlineTube(context_ptr, state.objects.at(object), poseGroupCenterline(group), is_as_grown);
            } else {
                writeRigidObject(context_ptr, state.objects.at(object), object_motions.at(object));
            }
        }

        state.transform = state.pending_transform;
        moved_groups.push_back(group);
        if (group.kind == PoseGroupReference::INTERNODE) {
            shoots_with_moved_internodes.insert(group.shoot);
        }
    });

    // Internode tubes span every internode of their shoot, so they are redrawn once per shoot from the node positions just moved
    for (Shoot *shoot: shoots_with_moved_internodes) {
        PoseShootState &shoot_state = pose_state.shoots.at(shoot->ID);
        if (!shoot_state.has_internode_tube || !context_ptr->doesObjectExist(shoot_state.internode_tube.objID)) {
            continue;
        }
        bool every_internode_is_as_grown = true;
        for (const PoseGroupState &internode_state: shoot_state.internodes) {
            if (!internode_state.transform.isIdentity()) {
                every_internode_is_as_grown = false;
                break;
            }
        }
        writeCenterlineTube(context_ptr, shoot_state.internode_tube, flatten(shoot->shoot_internode_vertices), every_internode_is_as_grown);
        recordPosedObjectState(context_ptr, shoot_state.internode_tube);
    }

    // Petioles and leaves that bend under their own weight were in balance where they grew, and are no longer once they have been turned
    for (const PoseGroupReference &group: moved_groups) {
        if (group.kind == PoseGroupReference::PETIOLE && (!group.state->transform.isRotationIdentity() || group.state->transform.scale != 1.0 || group.state->resettle_at_rest)) {
            group.phytomer->resettlePetioleUnderGravity(group.petiole_index);
        }
    }

    // Record how the pose left each part it moved, against which later changes made by anything else are recognized
    for (const PoseGroupReference &group: moved_groups) {
        CaptureVisitor capture;
        visitPoseGroup(group, capture);
        group.state->posed = capture.values;
        for (PoseObjectState &object_state: group.state->objects) {
            recordPosedObjectState(context_ptr, object_state);
        }
    }

    if (moved_groups.empty()) {
        return;
    }

    // Caches built from where the plant was
    clearBVHCache();
    plant_instance.shadow_grid_leaf_count = SIZE_MAX;
    plant_instance.leaf_angle_tracker.blade_primitive_cache.clear();
    for (const std::shared_ptr<Shoot> &shoot: plant_instance.shoot_tree) {
        for (const std::shared_ptr<Phytomer> &phytomer: shoot->phytomers) {
            for (const std::vector<FloralBud> &petiole_buds: phytomer->floral_buds) {
                for (const FloralBud &fbud: petiole_buds) {
                    for (const uint fruit_objID: fbud.inflorescence_objIDs) {
                        fruit_contact_proxies.erase(fruit_objID);
                        fruit_contact_candidates.erase(fruit_objID);
                    }
                }
            }
        }
    }
}

void PlantArchitecture::translatePlantRestPose(PlantInstance &plant_instance, const helios::vec3 &shift) const {
    PlantPoseState &pose_state = *plant_instance.pose_state;

    auto translateObjectState = [this, &shift](PoseObjectState &object_state) {
        object_state.rest_object_matrix[3] += shift.x;
        object_state.rest_object_matrix[7] += shift.y;
        object_state.rest_object_matrix[11] += shift.z;
        for (std::array<float, 16> &primitive_matrix: object_state.rest_primitive_matrices) {
            primitive_matrix[3] += shift.x;
            primitive_matrix[7] += shift.y;
            primitive_matrix[11] += shift.z;
        }
        for (vec3 &shared_vertex: object_state.rest_shared_vertices) {
            shared_vertex = shared_vertex + shift;
        }
        recordPosedObjectState(context_ptr, object_state);
    };

    const bool structure_unchanged = forEachPoseGroup(plant_instance, pose_state, false, [&shift, &translateObjectState](PoseGroupReference &group) {
        PoseGroupState &state = *group.state;
        TranslateRestVisitor translate(state, shift);
        visitPoseGroup(group, translate);
        for (PoseObjectState &object_state: state.objects) {
            translateObjectState(object_state);
        }
        // The part now turns, and stretches, about a pivot that has moved by the shift: k R (p + s) + t' = k R p + t + s
        const vec3 rotated_shift = state.transform.applyToOffset(shift);
        state.transform.translation[0] += double(shift.x) - double(rotated_shift.x);
        state.transform.translation[1] += double(shift.y) - double(rotated_shift.y);
        state.transform.translation[2] += double(shift.z) - double(rotated_shift.z);
    });
    if (!structure_unchanged) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): The structure of the posed plant changed while it was being moved, which should not be possible. This is a bug in the plant architecture plug-in.");
    }
    for (PoseShootState &shoot_state: pose_state.shoots) {
        if (shoot_state.has_internode_tube && context_ptr->doesObjectExist(shoot_state.internode_tube.objID)) {
            translateObjectState(shoot_state.internode_tube);
        }
    }
}

void PlantArchitecture::rejectPosedPlant(uint plantID, const std::string &function_name) const {
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    if (plant_instance.pose_baked || (plant_instance.pose_state != nullptr && !plant_instance.pose_state->joint_poses.empty())) {
        helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Plant " + std::to_string(plantID) +
                             " has been re-posed with setPlantJointPose(). This function rebuilds a plant from its growth parameters, which do not yet describe a pose, so the result would not be the posed plant. "
                             "Call resetPlantPose() first if the plant is still posed; a plant whose pose has been made permanent by bakePlantPose() or advanceTime() cannot be used here.");
    }
}

bool PlantArchitecture::isPlantPosed(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::isPlantPosed): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    return plant_instance.pose_state != nullptr && !plant_instance.pose_state->joint_poses.empty();
}

PlantJointPose PlantArchitecture::getPlantJointPose(uint plantID, const PlantJointID &joint) const {
    validatePlantJoint(plantID, joint, "getPlantJointPose");
    const PlantInstance &plant_instance = plant_instances.at(plantID);
    if (plant_instance.pose_state == nullptr) {
        return {};
    }
    const auto joint_pose = plant_instance.pose_state->joint_poses.find(joint);
    if (joint_pose == plant_instance.pose_state->joint_poses.end()) {
        return {};
    }
    return joint_pose->second;
}

void PlantArchitecture::setPlantJointPose(uint plantID, const PlantJointID &joint, const PlantJointPose &pose) {
    setPlantJointPoses(plantID, std::vector<PlantJointID>{joint}, std::vector<PlantJointPose>{pose});
}

void PlantArchitecture::setPlantJointPoses(uint plantID, const std::vector<PlantJointID> &joints, const std::vector<PlantJointPose> &poses) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPoses): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    if (joints.size() != poses.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPoses): " + std::to_string(joints.size()) + " joints were given but " + std::to_string(poses.size()) + " poses; one pose is needed for each joint.");
    }
    if (isInsidePhytomerCallback()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPoses): A plant cannot be re-posed inside a phytomer callback (PhytomerParameters::phytomer_creation_function or phytomer_callback_function), because the shoot being grown "
                             "holds positions that would not be moved with it. Call it from outside the callback, for example between calls to advanceTime().");
    }
    for (size_t i = 0; i < joints.size(); i++) {
        validatePlantJoint(plantID, joints.at(i), "setPlantJointPoses");
        if (!std::isfinite(poses.at(i).pitch_degrees) || !std::isfinite(poses.at(i).yaw_degrees) || !std::isfinite(poses.at(i).roll_degrees)) {
            helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPoses): The pose given for joint " + std::to_string(i) + " has an angle that is not a finite number.");
        }
        if (!std::isfinite(poses.at(i).length_scale) || poses.at(i).length_scale <= 0.f) {
            helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPoses): The pose given for joint " + std::to_string(i) + " has a length scale of " + std::to_string(poses.at(i).length_scale) +
                                 ", but it must be a positive number: 1 leaves the organ at the length it grew to.");
        }
        if (poses.at(i).length_scale != 1.f && joints.at(i).type != PLANT_JOINT_INTERNODE && joints.at(i).type != PLANT_JOINT_PETIOLE) {
            helios_runtime_error("ERROR (PlantArchitecture::setPlantJointPoses): The pose given for joint " + std::to_string(i) + " has a length scale of " + std::to_string(poses.at(i).length_scale) +
                                 ", but only an internode or a petiole joint (PLANT_JOINT_INTERNODE, PLANT_JOINT_PETIOLE) has a length to pose.");
        }
    }

    PlantInstance &plant_instance = plant_instances.at(plantID);
    preparePlantForPosing(plantID);

    std::map<PlantJointID, PlantJointPose> &joint_poses = plant_instance.pose_state->joint_poses;
    const std::map<PlantJointID, PlantJointPose> previous_joint_poses = joint_poses;
    for (size_t i = 0; i < joints.size(); i++) {
        const PlantJointPose &pose = poses.at(i);
        if (pose.pitch_degrees == 0.f && pose.yaw_degrees == 0.f && pose.roll_degrees == 0.f && pose.length_scale == 1.f) {
            joint_poses.erase(joints.at(i));
        } else {
            joint_poses[joints.at(i)] = pose;
        }
    }

    evaluatePlantPoseOrRestore(plantID, previous_joint_poses);

    // With every joint back at zero the plant stands exactly as recorded, and the record is no longer needed
    if (joint_poses.empty()) {
        releasePlantPoseState(plant_instance);
    }
}

void PlantArchitecture::evaluatePlantPoseOrRestore(uint plantID, const std::map<PlantJointID, PlantJointPose> &previous_joint_poses) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    try {
        evaluatePlantPose(plantID);
    } catch (...) {
        // A pose can turn out not to be applicable only once the plant's stored geometry has been put back as it grew to read the joints from, for example a turn about an internode that has no
        // length. Left there, the plant's stored positions would describe the unposed plant while its geometry in the Context was still posed, and the pose that failed would be tried again by every
        // later call. The poses the plant had are put back instead, which returns its stored positions to what they were, and the error is passed on.
        plant_instance.pose_state->joint_poses = previous_joint_poses;
        evaluatePlantPose(plantID);
        if (plant_instance.pose_state->joint_poses.empty()) {
            releasePlantPoseState(plant_instance);
        }
        throw;
    }
}

void PlantArchitecture::releasePlantPoseState(PlantInstance &plant_instance) const {
    // The branches go back to being seated as they were before the plant was posed (see Shoot::base_offset_pinned)
    const std::vector<bool> &pinned_before = plant_instance.pose_state->base_offset_pinned_before;
    for (size_t shootID = 0; shootID < plant_instance.shoot_tree.size() && shootID < pinned_before.size(); shootID++) {
        plant_instance.shoot_tree.at(shootID)->base_offset_pinned = pinned_before.at(shootID);
    }
    plant_instance.pose_state.reset();
}

void PlantArchitecture::preparePlantForPosing(uint plantID) {
    PlantInstance &plant_instance = plant_instances.at(plantID);
    // Keep whatever else has been done to the plant since it was last posed. If its structure has changed, the plant as it now stands is all there is to measure from, so the old pose becomes permanent.
    if (plant_instance.pose_state != nullptr && !rebindPlantRestPose(plant_instance)) {
        bakePlantPose(plantID);
    }
    if (plant_instance.pose_state == nullptr) {
        capturePlantRestPose(plant_instance);
    }
}

void PlantArchitecture::setShootNodePositions(uint plantID, uint shootID, const std::vector<helios::vec3> &node_positions) {
    validateShootID(plantID, shootID, "setShootNodePositions");
    if (isInsidePhytomerCallback()) {
        helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): A plant cannot be re-posed inside a phytomer callback. Call it from outside the callback, for example between calls to advanceTime().");
    }
    PlantInstance &plant_instance = plant_instances.at(plantID);
    const Shoot &shoot = *plant_instance.shoot_tree.at(shootID);
    if (shoot.isPruned()) {
        helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): Shoot " + std::to_string(shootID) + " of plant " + std::to_string(plantID) + " has been pruned, so it has no nodes to position.");
    }
    const size_t node_count = shoot.phytomers.size();
    if (node_positions.size() != node_count + 1) {
        helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): " + std::to_string(node_positions.size()) + " positions were given for shoot " + std::to_string(shootID) + ", which has " + std::to_string(node_count) +
                             " nodes. One position is needed for the base of the shoot and one for each of its nodes (" + std::to_string(node_count + 1) + ").");
    }
    for (size_t position = 0; position < node_positions.size(); position++) {
        if (!std::isfinite(node_positions.at(position).x) || !std::isfinite(node_positions.at(position).y) || !std::isfinite(node_positions.at(position).z)) {
            helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): Position " + std::to_string(position) + " is not finite.");
        }
        if (position > 0 && (node_positions.at(position) - node_positions.at(position - 1)).magnitude() < 1e-9f) {
            helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): Positions " + std::to_string(position - 1) + " and " + std::to_string(position) + " are coincident, which leaves the internode between them without a direction.");
        }
    }

    // The shoot stays attached where it is, so the first position has to be its base. As for a petiole, a small discrepancy is taken to be measurement error and the path is moved onto the base.
    const vec3 shoot_base = shoot.shoot_internode_vertices.front().front();
    const float attachment_distance = (node_positions.front() - shoot_base).magnitude();
    const float attachment_tolerance = std::fmax(1e-3f, 2.f * shoot.phytomers.front()->getInternodeRadius());
    if (attachment_distance > attachment_tolerance) {
        helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): The first position is " + std::to_string(attachment_distance) + " m from the base of shoot " + std::to_string(shootID) + ", which exceeds the attachment tolerance of " +
                             std::to_string(attachment_tolerance) + " m. The first position must be where the shoot is attached; to move that, pose the joints of the shoot it grows from.");
    }
    const vec3 base_shift = shoot_base - node_positions.front();

    preparePlantForPosing(plantID);
    PlantPoseState &pose_state = *plant_instance.pose_state;

    // The shoot's own internode poses are replaced; those of every other joint stay, including the one where this shoot is attached, and are what the shoot is carried by
    const std::map<PlantJointID, PlantJointPose> previous_joint_poses = pose_state.joint_poses;
    PlantJointID joint;
    joint.type = PLANT_JOINT_INTERNODE;
    joint.shootID = shootID;
    for (uint node_index = 0; node_index < node_count; node_index++) {
        joint.node_index = node_index;
        pose_state.joint_poses.erase(joint);
    }

    // Working the poses out puts the plant's stored geometry back as it grew. If they cannot be worked out, the plant is put back as it was posed before the error is passed on.
    try {

    // With the plant's stored geometry as it grew, the motion of this shoot from everything before its first internode
    computePlantPoseTransforms(plantID);
    PoseRigidTransform carrier = pose_state.shoots.at(shootID).shoot.pending_transform;

    // Each internode in turn, from the base outward, is turned and stretched about its base so that its tip lands on the next position. Its base is already where the previous internode's tip was put.
    for (uint node_index = 0; node_index < node_count; node_index++) {
        joint.node_index = node_index;
        const PlantJointFrame rest_frame = getPlantJointFrame(plantID, joint);
        const vec3 rest_tip = shoot.shoot_internode_vertices.at(node_index).back();
        const vec3 rest_chord = rest_tip - rest_frame.pivot;
        if (rest_chord.magnitude() < 1e-9f) {
            helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): The internode of node " + std::to_string(node_index) + " of shoot " + std::to_string(shootID) + " has no length as it grew, so it cannot be pointed at a position.");
        }

        // Where the tip has to go, in the frame of the plant as it grew: the target with the motion of everything before this internode taken back out
        const vec3 target = node_positions.at(node_index + 1) + base_shift;
        const vec3 target_offset_world = make_vec3(float(double(target.x) - carrier.translation[0]), float(double(target.y) - carrier.translation[1]), float(double(target.z) - carrier.translation[2]));
        const vec3 target_as_grown = carrier.applyInverseToDirection(target_offset_world);
        const vec3 target_chord = target_as_grown - rest_frame.pivot;
        if (target_chord.magnitude() < 1e-9f) {
            helios_runtime_error("ERROR (PlantArchitecture::setShootNodePositions): Position " + std::to_string(node_index + 1) + " coincides with the base of its internode.");
        }

        PlantJointPose pose = solveJointPoseForChord(rest_frame, rest_chord, target_chord);
        if (pose.pitch_degrees != 0.f || pose.yaw_degrees != 0.f || pose.roll_degrees != 0.f || pose.length_scale != 1.f) {
            pose_state.joint_poses[joint] = pose;
        }

        // The motion of everything from this internode's tip outward, exactly as the pose will be applied
        carrier = carrier.composedWith(makeJointRotation(rest_frame, pose));
        if (pose.length_scale != 1.f) {
            carrier = carrier.composedWith(makeTipTranslation(rest_frame, pose, rest_tip));
        }
    }
    } catch (...) {
        pose_state.joint_poses = previous_joint_poses;
        evaluatePlantPose(plantID);
        if (pose_state.joint_poses.empty()) {
            releasePlantPoseState(plant_instance);
        }
        throw;
    }

    evaluatePlantPoseOrRestore(plantID, previous_joint_poses);
    if (pose_state.joint_poses.empty()) {
        releasePlantPoseState(plant_instance);
    }
}

void PlantArchitecture::setPlantJointLength(uint plantID, const PlantJointID &joint, float length) {
    validatePlantJoint(plantID, joint, "setPlantJointLength");
    if (joint.type != PLANT_JOINT_INTERNODE && joint.type != PLANT_JOINT_PETIOLE) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantJointLength): Only an internode or a petiole joint (PLANT_JOINT_INTERNODE, PLANT_JOINT_PETIOLE) has a length to set.");
    }
    if (!std::isfinite(length) || length <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantJointLength): A length of " + std::to_string(length) + " m was given, but it must be a positive number.");
    }

    // The length the organ grew to is its present length with the present pose taken back out
    PlantJointPose pose = getPlantJointPose(plantID, joint);
    const float grown_length = getPlantJointFrame(plantID, joint).length / pose.length_scale;
    if (!(grown_length > 0.f)) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantJointLength): The organ has no length as it grew, so it cannot be scaled to a length of " + std::to_string(length) + " m.");
    }
    pose.length_scale = length / grown_length;
    setPlantJointPose(plantID, joint, pose);
}

void PlantArchitecture::resetPlantPose(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::resetPlantPose): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    if (isInsidePhytomerCallback()) {
        helios_runtime_error("ERROR (PlantArchitecture::resetPlantPose): A plant cannot be re-posed inside a phytomer callback. Call it from outside the callback, for example between calls to advanceTime().");
    }
    PlantInstance &plant_instance = plant_instances.at(plantID);
    if (plant_instance.pose_state == nullptr) {
        return;
    }
    if (!rebindPlantRestPose(plant_instance)) {
        helios_runtime_error("ERROR (PlantArchitecture::resetPlantPose): The structure of plant " + std::to_string(plantID) +
                             " has changed since it was posed (organs were added or removed), so the plant as it grew can no longer be restored. Reset the pose before changing the plant's structure.");
    }
    plant_instance.pose_state->joint_poses.clear();
    evaluatePlantPose(plantID);
    releasePlantPoseState(plant_instance);
}

void PlantArchitecture::bakePlantPose(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::bakePlantPose): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    PlantInstance &plant_instance = plant_instances.at(plantID);
    if (plant_instance.pose_state == nullptr) {
        return;
    }
    if (!plant_instance.pose_state->joint_poses.empty()) {
        plant_instance.pose_baked = true;
        // Each branch is from now on seated where the pose left it (see Shoot::base_offset_pinned)
        for (const std::shared_ptr<Shoot> &shoot: plant_instance.shoot_tree) {
            shoot->base_offset_pinned = true;
        }
    } else {
        // Nothing is posed, so there is nothing to make permanent and the plant goes back to being as it was before it was posed
        releasePlantPoseState(plant_instance);
        return;
    }
    // An organ whose length was posed is growing toward a length changed by the same factor, so that the fraction of it that has elongated is what it was and later growth does not undo the pose
    for (const auto &joint_pose: plant_instance.pose_state->joint_poses) {
        const PlantJointID &joint = joint_pose.first;
        const float length_scale = joint_pose.second.length_scale;
        if (length_scale == 1.f || joint.shootID >= plant_instance.shoot_tree.size() || joint.node_index >= plant_instance.shoot_tree.at(joint.shootID)->phytomers.size()) {
            continue;
        }
        Phytomer &phytomer = *plant_instance.shoot_tree.at(joint.shootID)->phytomers.at(joint.node_index);
        if (joint.type == PLANT_JOINT_INTERNODE) {
            phytomer.internode_length_max *= length_scale;
        } else if (joint.type == PLANT_JOINT_PETIOLE && joint.petiole_index < phytomer.petiole_length_max.size()) {
            phytomer.petiole_length_max.at(joint.petiole_index) *= length_scale;
        }
    }
    plant_instance.pose_state.reset();
}
