/** \file "Assets.cpp" Function definitions for plant organ prototypes plant architecture plug-in.

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
#include "global.h"

using namespace helios;

// The stalk is a cylinder followed by a join onto the blade. Both are rings of vertices around the leaflet's axis, and the join closes to a single point on the midrib.
static constexpr int petiolule_radial_subdivisions = 8;
static constexpr int petiolule_join_rings = 6;

uint petioluleVertexCount() {
    // Two rings bound the cylinder, the join adds one ring per step short of its tip, and the tip is a single vertex.
    return uint((2 + petiolule_join_rings - 1) * petiolule_radial_subdivisions + 1);
}

std::vector<helios::vec3> deformLeafLattice(const std::vector<helios::vec3> &rest_vertices, uint Nx, uint Ny, float scale, float mature_scale, float flexibility, float taper) {

    const size_t vertex_count = (size_t(Nx) + 1) * (size_t(Ny) + 1);
    if (rest_vertices.size() != vertex_count) {
        helios_runtime_error("ERROR (deformLeafLattice): Expected " + std::to_string(vertex_count) + " vertices for a " + std::to_string(Nx) + "x" + std::to_string(Ny) + " leaf lattice, but " +
                             std::to_string(rest_vertices.size()) + " were given.");
    } else if (scale < 0.f) {
        helios_runtime_error("ERROR (deformLeafLattice): Leaf scale must be non-negative, but " + std::to_string(scale) + " was given.");
    } else if (flexibility < 0.f) {
        helios_runtime_error("ERROR (deformLeafLattice): Leaf flexibility must be non-negative, but " + std::to_string(flexibility) + " was given.");
    } else if (mature_scale <= 0.f && flexibility > 0.f) {
        helios_runtime_error("ERROR (deformLeafLattice): Mature leaf scale must be positive when the leaf is flexible, but " + std::to_string(mature_scale) +
                             " was given. The mature size sets the length that the dimensionless flexibility is measured against.");
    }

    // A rigid leaf keeps its rest shape exactly. This is the path every non-drooping species takes, so it must cost nothing beyond the copy.
    if (flexibility == 0.f || scale == 0.f) {
        return rest_vertices;
    }

    // The parameter is dimensionless: it says how far the leaf droops once it is FULLY GROWN, rather than being a compliance in physical units. Raw compliance would have to be set per species across orders
    // of magnitude, because deflection grows with the fifth power of blade length - calibrating the three grasses against their previous shapes gave 32, 131 and 5414 for mature blades of 0.90 m, 0.65 m and
    // 0.22 m. Dividing out that length dependence collapses the 167x spread to under 2x, so a value near 20 is a sensible droopiness for any grass and means the same thing on a wheat leaf as on a maize one.
    //
    // Crucially the reference length is the leaf's MATURE size, which is fixed for a given leaf, and not its current size. Normalizing by the current size would divide out the very thing being modelled and
    // leave a leaf drooping just as far when it is small as when it is full-grown; dividing by a constant instead only fixes the units, leaving the growth-driven droop intact.
    const float compliance = flexibility / powf(std::max(mature_scale, 1e-6f), 4);

    auto latticeVertexIndex = [Nx](uint i, uint j) { return size_t(j) * (size_t(Nx) + 1) + size_t(i); };

    // ---- 1. Extract the rest midrib and the mass distribution along it ----

    // The midrib is the lattice column at the mid-span row. The blade is built symmetric about y=0, so this row is the leaf's structural axis and the one the bending moment acts along.
    const uint j_mid = Ny / 2;

    std::vector<vec3> midrib(Nx + 1);
    for (uint i = 0; i <= Nx; i++) {
        midrib.at(i) = rest_vertices.at(latticeVertexIndex(i, j_mid));
    }

    // Arclength along the rest midrib, and the local blade width, which sets how much mass each station carries. Width is measured across the full lattice row rather than assumed, so a tapered or lobed
    // blade loads the cantilever correctly without needing its profile described separately.
    std::vector<float> ds(Nx + 1, 0.f);
    std::vector<float> width(Nx + 1, 0.f);
    for (uint i = 0; i <= Nx; i++) {
        if (i > 0) {
            ds.at(i) = (midrib.at(i) - midrib.at(i - 1)).magnitude();
        }
        const vec3 &edge_low = rest_vertices.at(latticeVertexIndex(i, 0));
        const vec3 &edge_high = rest_vertices.at(latticeVertexIndex(i, Ny));
        width.at(i) = (edge_high - edge_low).magnitude();
    }

    // ---- 2. Rest tangent angles ----

    // The rest shape already carries the turgor curvature the prototype was built with; the deflection is added on top of it rather than replacing it, so a leaf that is curved at rest stays curved and simply
    // droops further.
    std::vector<float> theta_rest(Nx + 1, 0.f);
    for (uint i = 1; i <= Nx; i++) {
        const vec3 segment = midrib.at(i) - midrib.at(i - 1);
        const float horizontal = sqrtf(segment.x * segment.x + segment.y * segment.y);
        theta_rest.at(i) = std::atan2(segment.z, horizontal);
    }
    theta_rest.at(0) = theta_rest.at(1);

    // ---- 3. Iterate the moment balance ----

    // The bending moment at a station is the weight of everything distal to it acting through its horizontal lever arm. That lever arm depends on the DEFLECTED shape, not the rest shape, so the balance is
    // solved by iteration: deflect using the current lever arms, recompute them, repeat. Deflection shortens the lever arms, so the iteration is a contraction and converges in a handful of passes; a fixed
    // small iteration count is used rather than a convergence test to keep the cost per leaf bounded and predictable.
    //
    // Everything is computed in units of the SCALED leaf. Scale enters twice - once through the lever arm and once through the distal mass - so the moment grows with the cube of leaf length at fixed shape,
    // which is what makes a leaf that doubles in length droop roughly eight times as much with no age term anywhere in the model.
    constexpr int moment_iterations = 4;

    std::vector<float> theta(theta_rest);
    std::vector<vec3> deflected_midrib(Nx + 1);
    std::vector<float> moment(Nx + 1, 0.f);

    for (int iteration = 0; iteration < moment_iterations; iteration++) {

        // Integrate the current tangent angles into a centreline. The base station is clamped: it stays where the rest shape put it, which is what makes this a cantilever rather than a free body.
        deflected_midrib.at(0) = midrib.at(0) * scale;
        for (uint i = 1; i <= Nx; i++) {
            const vec3 rest_segment = midrib.at(i) - midrib.at(i - 1);
            // The horizontal heading of each rest segment is preserved and only its inclination changes, so the leaf droops in its own vertical plane instead of swinging sideways.
            const float rest_horizontal = sqrtf(rest_segment.x * rest_segment.x + rest_segment.y * rest_segment.y);
            const float segment_length = ds.at(i) * scale;
            vec3 heading;
            if (rest_horizontal > 1e-9f) {
                heading = make_vec3(rest_segment.x / rest_horizontal, rest_segment.y / rest_horizontal, 0.f);
            } else {
                heading = make_vec3(1.f, 0.f, 0.f);
            }
            deflected_midrib.at(i) = deflected_midrib.at(i - 1) + segment_length * (cosf(theta.at(i)) * heading + make_vec3(0.f, 0.f, sinf(theta.at(i))));
        }

        // Accumulate the moment from the tip inward in a single pass. Stepping inward by one segment adds that segment's own weight (acting through half its length) and swings all the mass already counted
        // further out through the extra lever arm, so the running distal mass is carried along rather than re-summed at each station - the difference between an O(Nx) sweep and an O(Nx^2) one.
        moment.assign(Nx + 1, 0.f);
        float distal_mass = 0.f;
        for (int i = int(Nx) - 1; i >= 0; i--) {
            const uint i_outer = uint(i) + 1;
            const float segment_mass = width.at(i_outer) * ds.at(i_outer) * scale * scale;
            const vec3 lever = deflected_midrib.at(i_outer) - deflected_midrib.at(uint(i));
            const float lever_arm = sqrtf(lever.x * lever.x + lever.y * lever.y);

            // Everything already counted acts through this segment's full lever arm; this segment's own weight acts through half of it, being distributed along the segment rather than concentrated at its end.
            moment.at(uint(i)) = moment.at(i_outer) + distal_mass * lever_arm + segment_mass * 0.5f * lever_arm;
            distal_mass += segment_mass;
        }

        // Curvature is proportional to moment; integrating it along the arc gives how far the blade has bent. Integrating curvature along arclength (rather than solving the linear beam equation for a vertical
        // deflection field) is what keeps the tip well-behaved once it droops past the small-angle regime - the leaf curls over instead of running off to infinity.
        //
        // The bending is accumulated on its own and then ADDED to the rest angle, rather than integrated into the tangent angle directly. The prototype already carries the blade's turgor shape - its
        // longitudinal curvature, and the arc a grass leaf holds even when rigid - in theta_rest, and building the tangent angle purely from the bending would discard all of it and straighten the leaf out
        // into a flat board. A leaf that is curved at rest must stay curved and simply droop further.
        float accumulated_bend = 0.f;
        for (uint i = 1; i <= Nx; i++) {
            // Compliance rises along the blade rather than being constant, because the midrib that carries the bending thins out toward the tip. Without this the blade hinges at its base and runs straight
            // from there, since that is where the moment is greatest; with it, the stiffness falls off faster than the moment does and the blade curves hardest where a real leaf does, near the tip.
            const float station = float(i) / float(Nx);
            const float local_compliance = compliance * powf(std::max(taper, 1e-3f), station);
            const float curvature = -local_compliance * moment.at(i);
            accumulated_bend += curvature * ds.at(i) * scale;
            // Clamp to just short of straight down. A leaf hanging vertically carries no further lever arm, so this is a physical limit rather than an arbitrary cutoff, and it stops a very flexible leaf from
            // curling back underneath itself.
            //
            // The accumulator is clamped along with the angle, not just the angle it produces. Letting it run past vertical while only the angle was limited pinned every station beyond that point at the
            // clamp, which drew the blade as a sharp hinge near the base followed by a dead-straight remainder - and no amount of stiffness taper could relieve it, because the excess bend had already been
            // banked into the accumulator further in.
            theta.at(i) = std::clamp(theta_rest.at(i) + accumulated_bend, -0.5f * PI_F + 1e-3f, 0.5f * PI_F);
            accumulated_bend = theta.at(i) - theta_rest.at(i);
        }
        theta.at(0) = theta_rest.at(0);
    }

    // ---- 4. Carry the lattice with the deflected midrib ----

    // Each lattice row is rigidly attached to its midrib station: the offset from the midrib is rotated by how much that station's tangent turned. This keeps the blade's width and its cross-sectional shape
    // (the midrib fold, the lateral curvature, the waves) intact while the leaf as a whole bends, which is what distinguishes bending from stretching.
    std::vector<vec3> deformed(rest_vertices.size());
    for (uint i = 0; i <= Nx; i++) {
        const float turn = theta.at(i) - theta_rest.at(i);
        const float cos_turn = cosf(turn);
        const float sin_turn = sinf(turn);

        // The rotation is about the local lateral axis - horizontally perpendicular to the midrib heading - so the blade tips forward and down rather than twisting about its own length.
        const vec3 rest_segment = (i > 0) ? (midrib.at(i) - midrib.at(i - 1)) : (midrib.at(1) - midrib.at(0));
        const float rest_horizontal = sqrtf(rest_segment.x * rest_segment.x + rest_segment.y * rest_segment.y);
        vec3 heading = (rest_horizontal > 1e-9f) ? make_vec3(rest_segment.x / rest_horizontal, rest_segment.y / rest_horizontal, 0.f) : make_vec3(1.f, 0.f, 0.f);

        for (uint j = 0; j <= Ny; j++) {
            const size_t index = latticeVertexIndex(i, j);
            const vec3 offset = (rest_vertices.at(index) - midrib.at(i)) * scale;

            // Split the offset into the component along the midrib heading and the vertical one; those are the two the bend rotates. The lateral component rides along unchanged.
            const float along = offset.x * heading.x + offset.y * heading.y;
            const vec3 lateral = offset - along * heading - make_vec3(0.f, 0.f, offset.z);

            const float along_rotated = along * cos_turn - offset.z * sin_turn;
            const float vertical_rotated = along * sin_turn + offset.z * cos_turn;

            deformed.at(index) = deflected_midrib.at(i) + along_rotated * heading + lateral + make_vec3(0.f, 0.f, vertical_rotated);
        }
    }

    return deformed;
}

std::vector<helios::vec3> bendPetioleCenterline(const std::vector<helios::vec3> &rest_offsets, const std::vector<float> &radii, const std::vector<float> &load_arclength_fractions, const std::vector<float> &load_weights,
                                                float reference_load, float compliance, float *equilibrium_residual) {

    const size_t node_count = rest_offsets.size();
    if (node_count < 2) {
        helios_runtime_error("ERROR (bendPetioleCenterline): A centerline needs at least 2 nodes, but " + std::to_string(node_count) + " were given.");
    } else if (radii.size() != node_count) {
        helios_runtime_error("ERROR (bendPetioleCenterline): " + std::to_string(radii.size()) + " radii were given for " + std::to_string(node_count) + " centerline nodes; there must be one radius per node.");
    } else if (load_arclength_fractions.size() != load_weights.size()) {
        helios_runtime_error("ERROR (bendPetioleCenterline): " + std::to_string(load_arclength_fractions.size()) + " load positions were given with " + std::to_string(load_weights.size()) +
                             " load weights; every load needs both.");
    } else if (!(compliance >= 0.f) || !std::isfinite(compliance)) {
        helios_runtime_error("ERROR (bendPetioleCenterline): Compliance must be finite and non-negative, but " + std::to_string(compliance) + " was given.");
    } else if (!std::isfinite(reference_load)) {
        helios_runtime_error("ERROR (bendPetioleCenterline): Reference load must be finite, but " + std::to_string(reference_load) + " was given.");
    }
    for (size_t load = 0; load < load_weights.size(); load++) {
        if (!(load_arclength_fractions.at(load) >= 0.f && load_arclength_fractions.at(load) <= 1.f)) {
            helios_runtime_error("ERROR (bendPetioleCenterline): Load " + std::to_string(load) + " is at arclength fraction " + std::to_string(load_arclength_fractions.at(load)) + ", which is outside [0, 1].");
        } else if (!(load_weights.at(load) >= 0.f) || !std::isfinite(load_weights.at(load))) {
            helios_runtime_error("ERROR (bendPetioleCenterline): Load " + std::to_string(load) + " has weight " + std::to_string(load_weights.at(load)) + ", but weights must be finite and non-negative.");
        }
    }

    if (equilibrium_residual != nullptr) {
        *equilibrium_residual = 0.f;
    }
    if (compliance == 0.f || reference_load <= 0.f || load_weights.empty()) {
        return rest_offsets;
    }

    const size_t segment_count = node_count - 1;

    // ---- Rest geometry of each segment: length, horizontal heading and elevation ----

    // Worked in double precision: the iteration below compares successive shapes to decide when it has converged, and single precision noise in the lever arms would otherwise set a floor on that test.
    std::vector<double> segment_length(segment_count);
    std::vector<double> node_arclength(node_count, 0.0);
    std::vector<double> rest_elevation(segment_count);
    std::vector<double> heading_x(segment_count), heading_y(segment_count);
    std::vector<bool> has_heading(segment_count, false);
    for (size_t j = 0; j < segment_count; j++) {
        const double dx = double(rest_offsets.at(j + 1).x) - double(rest_offsets.at(j).x);
        const double dy = double(rest_offsets.at(j + 1).y) - double(rest_offsets.at(j).y);
        const double dz = double(rest_offsets.at(j + 1).z) - double(rest_offsets.at(j).z);
        segment_length.at(j) = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (segment_length.at(j) <= 0.0) {
            helios_runtime_error("ERROR (bendPetioleCenterline): Centerline nodes " + std::to_string(j) + " and " + std::to_string(j + 1) + " coincide, so the segment between them has no direction to bend.");
        }
        node_arclength.at(j + 1) = node_arclength.at(j) + segment_length.at(j);
        const double horizontal = std::sqrt(dx * dx + dy * dy);
        rest_elevation.at(j) = std::atan2(dz, horizontal);
        if (horizontal > 1e-6 * segment_length.at(j)) {
            heading_x.at(j) = dx / horizontal;
            heading_y.at(j) = dy / horizontal;
            has_heading.at(j) = true;
        }
    }
    const double total_length = node_arclength.back();

    // A segment standing exactly vertical has no heading of its own to bend toward; it bends toward the heading of the petiole as a whole, taken from its rest chord or failing that from the nearest segment
    // that has one. A centerline that is vertical throughout has no horizontal lever arm anywhere, so its loads exert no moment and it keeps its rest shape.
    double fallback_x = double(rest_offsets.back().x) - double(rest_offsets.front().x);
    double fallback_y = double(rest_offsets.back().y) - double(rest_offsets.front().y);
    double fallback_horizontal = std::sqrt(fallback_x * fallback_x + fallback_y * fallback_y);
    if (fallback_horizontal <= 1e-6 * total_length) {
        fallback_horizontal = 0.0;
        for (size_t j = 0; j < segment_count; j++) {
            if (has_heading.at(j)) {
                fallback_x = heading_x.at(j);
                fallback_y = heading_y.at(j);
                fallback_horizontal = 1.0;
                break;
            }
        }
        if (fallback_horizontal == 0.0) {
            return rest_offsets;
        }
    }
    for (size_t j = 0; j < segment_count; j++) {
        if (!has_heading.at(j)) {
            heading_x.at(j) = fallback_x / fallback_horizontal;
            heading_y.at(j) = fallback_y / fallback_horizontal;
        }
    }

    // ---- Stiffness taper ----

    // Bending stiffness goes with the fourth power of radius. The ratio is floored because the tube's taper parameter can take its tip radius to zero, which would make the last segment infinitely compliant; a
    // real petiole tip carries the terminal leaflet's petiolule and is never that thin.
    constexpr double minimum_radius_ratio = 0.5;
    std::vector<double> segment_compliance_factor(segment_count, 1.0);
    if (radii.front() > 0.f) {
        for (size_t j = 0; j < segment_count; j++) {
            const double ratio = std::max(minimum_radius_ratio, 0.5 * (double(radii.at(j)) + double(radii.at(j + 1))) / double(radii.front()));
            segment_compliance_factor.at(j) = 1.0 / (ratio * ratio * ratio * ratio);
        }
    }

    const double gain = 2.0 * double(compliance) / (double(reference_load) * total_length * total_length);

    const size_t load_count = load_weights.size();
    std::vector<double> load_arclength(load_count);
    std::vector<size_t> load_segment(load_count);
    for (size_t load = 0; load < load_count; load++) {
        load_arclength.at(load) = double(load_arclength_fractions.at(load)) * total_length;
        size_t segment = 0;
        while (segment + 1 < segment_count && load_arclength.at(load) > node_arclength.at(segment + 1)) {
            segment++;
        }
        load_segment.at(load) = segment;
    }

    // ---- Shape and moments ----

    // Nodes and unit directions of the shape given by a set of per-segment rotations. Each segment keeps its heading and is lowered by the rotations of the segments BEFORE it - the curvature accumulated up to its
    // own base, which for the first segment is nothing - and is stopped at hanging straight down.
    //
    // The first segment is therefore clamped: the petiole leaves the stem at the pitch it was generated with, however heavily it is loaded, and the bend appears beyond the insertion as curvature accumulating
    // along the length. That is what a cantilever built in at its support does, and it is what real petioles do. Measured Pheno4D tomato petioles hold a base elevation of 33.7, 20.9, 19.8 and 12.0 degrees at
    // leaf ages 3, 6, 9 and 12 days while their base-to-tip arch grows from 13 to 19 degrees: the outer petiole arches over while the insertion is roughly held. Rotating each segment through its own share as
    // well - which the bending moment makes largest at the base, where the moment is - swung the insertion itself down instead, to 9.9, 8.3, 1.9 and 2.8 degrees over the same ages, and no refinement of the
    // discretization recovered it (raising petiole.length_segments from 5 to 30 moved the base elevation only from 9.9 to 11.3 degrees).
    const double half_pi = 0.5 * M_PI;
    std::vector<double> elevation(segment_count);
    std::vector<double> node_x(node_count), node_y(node_count), node_z(node_count);
    std::vector<double> direction_x(segment_count), direction_y(segment_count), direction_z(segment_count);
    auto buildShape = [&](const std::vector<double> &rotation) {
        node_x.at(0) = rest_offsets.front().x;
        node_y.at(0) = rest_offsets.front().y;
        node_z.at(0) = rest_offsets.front().z;
        double accumulated = 0.0;
        for (size_t j = 0; j < segment_count; j++) {
            elevation.at(j) = std::max(rest_elevation.at(j) - accumulated, -half_pi);
            accumulated += rotation.at(j);
            const double horizontal = std::cos(elevation.at(j));
            direction_x.at(j) = horizontal * heading_x.at(j);
            direction_y.at(j) = horizontal * heading_y.at(j);
            direction_z.at(j) = std::sin(elevation.at(j));
            node_x.at(j + 1) = node_x.at(j) + segment_length.at(j) * direction_x.at(j);
            node_y.at(j + 1) = node_y.at(j) + segment_length.at(j) * direction_y.at(j);
            node_z.at(j + 1) = node_z.at(j) + segment_length.at(j) * direction_z.at(j);
        }
    };

    // Rotation of each segment called for by the moments of the shape last built. The moment at arclength s within segment j is the sum over loads beyond s of the load times the horizontal distance from s to
    // the load along the segment's heading; that distance falls linearly along the segment, so its integral over the part of the segment short of the load is exact in closed form.
    // horizontal_bound replaces every lever arm by the arclength to the load, which bounds the moment of any shape and gives the iteration's relaxation factor.
    auto momentRotations = [&](std::vector<double> &rotation, bool horizontal_bound) {
        for (size_t j = 0; j < segment_count; j++) {
            double moment_integral = 0.0;
            for (size_t load = 0; load < load_count; load++) {
                if (load_arclength.at(load) <= node_arclength.at(j) || load_weights.at(load) == 0.f) {
                    continue;
                }
                const double span = std::min(load_arclength.at(load), node_arclength.at(j + 1)) - node_arclength.at(j);
                double contribution;
                if (horizontal_bound) {
                    contribution = span * (load_arclength.at(load) - node_arclength.at(j)) - 0.5 * span * span;
                } else {
                    const size_t k = load_segment.at(load);
                    const double along_load_segment = load_arclength.at(load) - node_arclength.at(k);
                    const double load_x = node_x.at(k) + along_load_segment * direction_x.at(k);
                    const double load_y = node_y.at(k) + along_load_segment * direction_y.at(k);
                    const double lever_at_node = (load_x - node_x.at(j)) * heading_x.at(j) + (load_y - node_y.at(j)) * heading_y.at(j);
                    const double lever_decrease_rate = direction_x.at(j) * heading_x.at(j) + direction_y.at(j) * heading_y.at(j);
                    contribution = span * lever_at_node - 0.5 * lever_decrease_rate * span * span;
                }
                moment_integral += double(load_weights.at(load)) * std::max(0.0, contribution);
            }
            rotation.at(j) = gain * segment_compliance_factor.at(j) * moment_integral;
        }
    };

    // ---- Fixed-point iteration ----

    // Bending shortens the lever arms, so the rotations a shape calls for fall as the shape bends further, and the balance is found by iterating from the rest shape. Plain iteration overshoots and oscillates
    // once the bend is large, so each step moves only part of the way: with G bounding how strongly the rotations respond to the shape (the total rotation of the centerline laid out horizontally), a relaxation
    // factor of 2/(2+G) makes every step contract. The iteration stops when the elevations stop changing, or after a fixed number of steps, which only an extreme compliance reaches and which then leaves a
    // bounded, nearly vertical shape. The result depends only on the inputs, never on any earlier call.
    std::vector<double> rotation(segment_count, 0.0);
    std::vector<double> target(segment_count, 0.0);
    momentRotations(target, true);
    double response_bound = 0.0;
    for (double value: target) {
        response_bound += value;
    }
    const double relaxation = 2.0 / (2.0 + response_bound);

    constexpr int maximum_iterations = 256;
    constexpr double elevation_tolerance = 1e-9;
    std::vector<double> previous_elevation(segment_count);
    buildShape(rotation);
    for (int iteration = 0; iteration < maximum_iterations; iteration++) {
        momentRotations(target, false);
        previous_elevation = elevation;
        for (size_t j = 0; j < segment_count; j++) {
            rotation.at(j) += relaxation * (target.at(j) - rotation.at(j));
        }
        buildShape(rotation);
        double largest_change = 0.0;
        for (size_t j = 0; j < segment_count; j++) {
            largest_change = std::max(largest_change, std::fabs(elevation.at(j) - previous_elevation.at(j)));
        }
        if (largest_change < elevation_tolerance) {
            break;
        }
    }

    if (equilibrium_residual != nullptr) {
        // Compare the returned shape's elevations with those its own moments call for, both stopped at vertical, so a segment legitimately held at hanging straight down does not count as unconverged.
        momentRotations(target, false);
        double accumulated_target = 0.0;
        double accumulated_applied = 0.0;
        double largest_residual = 0.0;
        for (size_t j = 0; j < segment_count; j++) {
            // The rotations are accumulated exclusively here for the same reason buildShape() accumulates them that way: a segment is placed by the curvature accumulated up to its own base.
            const double applied = std::max(rest_elevation.at(j) - accumulated_applied, -half_pi);
            const double called_for = std::max(rest_elevation.at(j) - accumulated_target, -half_pi);
            largest_residual = std::max(largest_residual, std::fabs(applied - called_for));
            accumulated_target += target.at(j);
            accumulated_applied += rotation.at(j);
        }
        *equilibrium_residual = float(largest_residual);
    }

    std::vector<vec3> bent(node_count);
    for (size_t node = 0; node < node_count; node++) {
        bent.at(node) = make_vec3(float(node_x.at(node)), float(node_y.at(node)), float(node_z.at(node)));
    }
    return bent;
}

uint GenericLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {

    // If OBJ model file is specified, load it and return the object ID
    if (!prototype_parameters->OBJ_model_file.empty()) {
        // Resolve OBJ file path (allows users to specify simple paths like "MyLeaf.obj")
        std::string resolved_obj = PlantArchitecture::resolveTextureFile(prototype_parameters->OBJ_model_file);
        return context_ptr->addPolymeshObject(context_ptr->loadOBJ(resolved_obj.c_str(), prototype_parameters->leaf_offset, 0, nullrotation, RGB::black, "ZUP", true));
    }

    std::string leaf_texture;
    if (prototype_parameters->leaf_texture_file.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture): Leaf prototype texture file was not specified.");
    } else if (prototype_parameters->leaf_texture_file.size() == 1) {
        leaf_texture = prototype_parameters->leaf_texture_file.begin()->second;
    } else if (prototype_parameters->leaf_texture_file.find(compound_leaf_index) == prototype_parameters->leaf_texture_file.end()) {
        helios_runtime_error("ERROR (PlantArchitecture): Leaf prototype texture file for compound leaf index " + std::to_string(compound_leaf_index) + " was not found.");
    } else {
        leaf_texture = prototype_parameters->leaf_texture_file[compound_leaf_index];
    }

    // Resolve leaf texture path (allows users to specify simple paths like "AlmondLeaf.png")
    leaf_texture = PlantArchitecture::resolveTextureFile(leaf_texture);

    // -- main leaf generation code -- //

    std::vector<uint> UUIDs;

    uint Nx = prototype_parameters->subdivisions; // number of leaf subdivisions in the x-direction (longitudinal)
    uint Ny = ceil(prototype_parameters->leaf_aspect_ratio.val() * float(Nx)); // number of leaf subdivisions in the y-direction (lateral)

    if (Ny % 2 != 0) { // Ny must be even
        Ny = Ny + 1;
    }

    const float dx = 1.f / float(Nx); // length of leaf subdivision in the x-direction
    const float dy = prototype_parameters->leaf_aspect_ratio.val() / float(Ny); // length of leaf subdivision in the y-direction

    std::vector<std::vector<vec3>> vertices;
    resize_vector(vertices, Nx + 1, Ny + 1);

    for (int j = 0; j <= Ny; j++) {
        float dtheta = 0;
        for (int i = 0; i <= Nx; i++) {

            const float x = float(i) * dx; // x-coordinate of leaf subdivision
            const float y = float(j) * dy - 0.5f * prototype_parameters->leaf_aspect_ratio.val(); // y-coordinate of leaf subdivision

            // Midrib folding, tapering along the blade. A leaf is most strongly channelled where its midrib is a prominent keel, near the base, and flattens out toward the tip as the midrib thins away;
            // folding the blade by the same amount along its whole length instead gives a leaf that is still V-shaped at the tip, which reads as a stiff gutter rather than a leaf.
            //
            // The taper is normalised to average one over the blade, so the parameter keeps meaning the same average fold it always did and an existing species is not silently flattened - only the
            // distribution of that fold along the leaf changes, from uniform to base-weighted.
            const float fold_taper = 1.5f * (1.f - x * x);
            const float local_fold = prototype_parameters->midrib_fold_fraction.val() * fold_taper;
            const float y_fold = cosf(0.5f * local_fold * M_PI) * y;
            const float z_fold = sinf(0.5f * local_fold * M_PI) * fabs(y);

            // x-curvature. The exponent sets how the bending is distributed along the blade rather than how much of it there is - see LeafPrototype::longitudinal_curvature_exponent - so the tip
            // deflection is the same whatever it is, and it is clamped to at least 1 because a smaller value would put a cusp at the leaf base.
            const float curvature_exponent = fmaxf(1.f, prototype_parameters->longitudinal_curvature_exponent.val());
            float z_xcurve = prototype_parameters->longitudinal_curvature.val() * powf(x, curvature_exponent);

            // y-curvature
            float z_ycurve = prototype_parameters->lateral_curvature.val() * powf(y / prototype_parameters->leaf_aspect_ratio.val(), 4);

            // petiole roll
            float z_petiole = 0;
            if (prototype_parameters->petiole_roll.val() != 0.0f) {
                z_petiole = fmin(0.1f, prototype_parameters->petiole_roll.val() * powf(7.f * y / prototype_parameters->leaf_aspect_ratio.val(), 4) * exp(-70.f * (x))) -
                            0.01 * prototype_parameters->petiole_roll.val() / fabs(prototype_parameters->petiole_roll.val());
            }

            // vertical displacement for leaf wave at each of the four subdivision vertices.
            // The wave is applied along the local surface normal *after* the longitudinal
            // curvature and buckle rotations, so the bumps don't get tilted into the leaf
            // tangent direction (which would visually flatten the wave past the buckle).
            float z_wave = 0;
            if (prototype_parameters->wave_period.val() > 0.0f && prototype_parameters->wave_amplitude.val() > 0.0f) {
                const float wave_phase = (x + prototype_parameters->wave_period.val() * float(j >= 0.5 * Ny)) * M_PI / prototype_parameters->wave_period.val();
                z_wave = 2.f * fabs(y) * prototype_parameters->wave_amplitude.val() * sinf(wave_phase);
            }

            vertices.at(j).at(i) = make_vec3(x, y_fold, z_fold + z_ycurve + z_petiole);

            float rot_angle = 0.f;

            if (prototype_parameters->longitudinal_curvature.val() != 0.0f && i > 0) {
                // Derivative of the curvature term above, which must track its exponent: this is what orients the wave displacement along the leaf surface.
                dtheta -= atan(curvature_exponent * prototype_parameters->longitudinal_curvature.val() * powf(x, curvature_exponent - 1.f) * dx);
                vertices.at(j).at(i) = rotatePointAboutLine(vertices.at(j).at(i), nullorigin, make_vec3(0, 1, 0), dtheta);
                rot_angle += dtheta;
            }

            // apply wave displacement along the rotated leaf-surface normal
            if (z_wave != 0.f) {
                vertices.at(j).at(i).x += z_wave * sinf(rot_angle);
                vertices.at(j).at(i).z += z_wave * cosf(rot_angle);
            }
        }
    }

    // The leaf is a regular (Nx+1) x (Ny+1) lattice, so which facets meet at each vertex is known exactly. Recording it as an indexed face set lets consumers that reconstruct a field across the mesh - camera
    // flux smoothing in the radiation model, vertex normals, shared-vertex OBJ export - see the leaf as one connected surface rather than as a bag of loose triangles.
    std::vector<vec3> mesh_vertices;
    std::vector<vec2> mesh_vertex_uv;
    mesh_vertices.reserve(size_t(Nx + 1) * size_t(Ny + 1));
    mesh_vertex_uv.reserve(size_t(Nx + 1) * size_t(Ny + 1));
    for (int j = 0; j <= Ny; j++) {
        for (int i = 0; i <= Nx; i++) {
            mesh_vertices.push_back(vertices.at(j).at(i));
            // The texture coordinate of a lattice vertex is a function of its indices alone, so neighbouring facets agree on it and it can be stored per vertex rather than per corner.
            mesh_vertex_uv.push_back(make_vec2(float(i) * dx, float(j) * dy / prototype_parameters->leaf_aspect_ratio.val()));
        }
    }
    auto latticeVertexIndex = [Nx](int i, int j) { return j * (int(Nx) + 1) + i; };

    std::vector<int3> mesh_faces;
    std::vector<uint> mesh_face_UUIDs;

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {

            const float x = float(i) * dx;
            const float y = float(j) * dy - 0.5f * prototype_parameters->leaf_aspect_ratio.val();
            vec2 uv0(x, (y + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());
            vec2 uv1(x + dx, (y + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());
            vec2 uv2(x + dx, (y + dy + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());
            vec2 uv3(x, (y + dy + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());

            vec3 v0 = vertices.at(j).at(i);
            vec3 v1 = vertices.at(j).at(i + 1);
            vec3 v2 = vertices.at(j + 1).at(i + 1);
            vec3 v3 = vertices.at(j + 1).at(i);

            // Add triangle 1 and check if it has effective area (including texture transparency)
            uint uuid1 = context_ptr->addTriangle(v0, v1, v2, leaf_texture.c_str(), uv0, uv1, uv2);
            if (context_ptr->getPrimitiveArea(uuid1) > 0) {
                UUIDs.push_back(uuid1);
                // Only the facets actually kept get a face entry, so the table continues to describe the geometry after the fully-transparent ones have been dropped.
                mesh_faces.push_back(make_int3(latticeVertexIndex(i, j), latticeVertexIndex(i + 1, j), latticeVertexIndex(i + 1, j + 1)));
                mesh_face_UUIDs.push_back(uuid1);
            } else {
                context_ptr->deletePrimitive(uuid1);
            }

            // Add triangle 2 and check if it has effective area (including texture transparency)
            uint uuid2 = context_ptr->addTriangle(v0, v2, v3, leaf_texture.c_str(), uv0, uv2, uv3);
            if (context_ptr->getPrimitiveArea(uuid2) > 0) {
                UUIDs.push_back(uuid2);
                mesh_faces.push_back(make_int3(latticeVertexIndex(i, j), latticeVertexIndex(i + 1, j + 1), latticeVertexIndex(i, j + 1)));
                mesh_face_UUIDs.push_back(uuid2);
            } else {
                context_ptr->deletePrimitive(uuid2);
            }
        }
    }

    // The lattice was built in the leaf's own frame, so every translation applied to the primitives has to be applied to the stored vertices as well or the face set would describe geometry that has moved.
    context_ptr->translatePrimitive(UUIDs, prototype_parameters->leaf_offset);
    for (vec3 &mesh_vertex: mesh_vertices) {
        mesh_vertex = mesh_vertex + prototype_parameters->leaf_offset;
    }

    if (prototype_parameters->build_petiolule) {
        // The petiolule length is looked up by compound-leaf index the same way as the texture, so the terminal leaflet can carry a longer stalk than the laterals.
        float petiolule_length = 0.f;
        if (prototype_parameters->petiolule_length.empty()) {
            helios_runtime_error("ERROR (PlantArchitecture): Leaf prototype builds a petiolule but LeafPrototype::petiolule_length is empty.");
        } else if (prototype_parameters->petiolule_length.size() == 1) {
            petiolule_length = prototype_parameters->petiolule_length.begin()->second;
        } else if (prototype_parameters->petiolule_length.find(compound_leaf_index) == prototype_parameters->petiolule_length.end()) {
            helios_runtime_error("ERROR (PlantArchitecture): LeafPrototype::petiolule_length has no entry for compound leaf index " + std::to_string(compound_leaf_index) +
                                 ". With more than one entry, every leaflet index needs its own; give a single entry to use one length for all leaflets.");
        } else {
            petiolule_length = prototype_parameters->petiolule_length.at(compound_leaf_index);
        }
        // Set by Phytomer::buildLeafPrototype() from the petiole radius where this leaflet attaches, so that the stalk continues the petiole without a step.
        const float petiolule_radius = prototype_parameters->petiolule_radius_fraction;
        if (petiolule_length <= 0.f || petiolule_radius <= 0.f) {
            helios_runtime_error("ERROR (PlantArchitecture): A petiolule needs a positive length and radius (got length " + std::to_string(petiolule_length) + ", radius " + std::to_string(petiolule_radius) +
                                 "). The length is LeafPrototype::petiolule_length; the radius is taken from the petiole, so the petiole must have a positive radius.");
        }

        // The blade starts where the cylindrical part of the stalk ends.
        const vec3 blade_shift = make_vec3(petiolule_length, 0, 0);
        context_ptr->translatePrimitive(UUIDs, blade_shift);
        for (vec3 &mesh_vertex: mesh_vertices) {
            mesh_vertex = mesh_vertex + blade_shift;
        }

        // The stalk is a cylinder along the midrib axis from the leaflet base to the blade base, followed by a join under the blade base. A cylinder simply
        // stopping against a blade of zero thickness leaves a visible step, so over the join the stalk's cross-section flattens onto the blade plane: its
        // thickness falls to zero with zero slope at both ends, so it leaves the cylinder and meets the blade tangentially, while its width narrows to a
        // rounded tip on the midrib. Built directly into the leaf's face set, as rings of vertices, so that the combined object describes both surfaces.
        constexpr float petiolule_join_length_in_radii = 4.f;
        const float join_length = petiolule_join_length_in_radii * petiolule_radius;

        std::vector<float> ring_x = {0.f, petiolule_length};
        std::vector<float> ring_half_width = {petiolule_radius, petiolule_radius};
        std::vector<float> ring_half_thickness = {petiolule_radius, petiolule_radius};
        for (int ring = 1; ring < petiolule_join_rings; ring++) {
            const float t = float(ring) / float(petiolule_join_rings);
            ring_x.push_back(petiolule_length + t * join_length);
            ring_half_width.push_back(petiolule_radius * std::sqrt(1.f - t * t));
            ring_half_thickness.push_back(petiolule_radius * 0.5f * (1.f + cosf(float(M_PI) * t)));
        }

        const int first_ring_vertex = scast<int>(mesh_vertices.size());
        for (size_t ring = 0; ring < ring_x.size(); ring++) {
            for (int k = 0; k < petiolule_radial_subdivisions; k++) {
                const float theta = 2.f * float(M_PI) * float(k) / float(petiolule_radial_subdivisions);
                mesh_vertices.push_back(prototype_parameters->leaf_offset + make_vec3(ring_x.at(ring), ring_half_width.at(ring) * cosf(theta), ring_half_thickness.at(ring) * sinf(theta)));
                // The texture coordinate array has to stay parallel to the vertices. The petiolule is drawn in a flat colour rather than from a texture, so the entry
                // only keeps the arrays aligned and is never sampled.
                mesh_vertex_uv.push_back(make_vec2(0, 0));
            }
        }
        // The join closes to a single point on the midrib rather than a ring of coincident vertices, which would give zero-area facets.
        const int tip_vertex = scast<int>(mesh_vertices.size());
        mesh_vertices.push_back(prototype_parameters->leaf_offset + make_vec3(petiolule_length + join_length, 0, 0));
        mesh_vertex_uv.push_back(make_vec2(0, 0));

        std::vector<uint> UUIDs_petiolule;
        auto addPetioluleFacet = [&](int i0, int i1, int i2) {
            const uint UUID = context_ptr->addTriangle(mesh_vertices.at(i0), mesh_vertices.at(i1), mesh_vertices.at(i2));
            UUIDs_petiolule.push_back(UUID);
            mesh_faces.push_back(make_int3(i0, i1, i2));
            mesh_face_UUIDs.push_back(UUID);
        };
        // Wound so the facet normals point away from the stalk's axis.
        for (size_t ring = 0; ring + 1 < ring_x.size(); ring++) {
            for (int k = 0; k < petiolule_radial_subdivisions; k++) {
                const int k_next = (k + 1) % petiolule_radial_subdivisions;
                const int near_k = first_ring_vertex + int(ring) * petiolule_radial_subdivisions + k;
                const int near_next = first_ring_vertex + int(ring) * petiolule_radial_subdivisions + k_next;
                const int far_k = near_k + petiolule_radial_subdivisions;
                const int far_next = near_next + petiolule_radial_subdivisions;
                addPetioluleFacet(near_k, near_next, far_next);
                addPetioluleFacet(near_k, far_next, far_k);
            }
        }
        const int last_ring_start = first_ring_vertex + int(ring_x.size() - 1) * petiolule_radial_subdivisions;
        for (int k = 0; k < petiolule_radial_subdivisions; k++) {
            addPetioluleFacet(last_ring_start + k, last_ring_start + (k + 1) % petiolule_radial_subdivisions, tip_vertex);
        }
        // labelLeafPrototype() keeps this label when it labels the rest of the leaf "leaf", and gives these primitives the petiole's material.
        context_ptr->setPrimitiveData(UUIDs_petiolule, "object_label", "petiolule");

        UUIDs.insert(UUIDs.end(), UUIDs_petiolule.begin(), UUIDs_petiolule.end());
    }

    prototype_parameters->leaf_aspect_ratio.resample();
    prototype_parameters->midrib_fold_fraction.resample();
    prototype_parameters->longitudinal_curvature.resample();
    prototype_parameters->longitudinal_curvature_exponent.resample();
    prototype_parameters->lateral_curvature.resample();
    prototype_parameters->petiole_roll.resample();
    prototype_parameters->wave_period.resample();
    prototype_parameters->wave_amplitude.resample();


    const uint objID = context_ptr->addPolymeshObject(UUIDs);

    // The face set covers every primitive of the object: the blade from the lattice above, and the petiolule from its rings. It is attached only when that is actually true, because
    // getFaceIndexForPrimitive() raises for a member the table omits, and writeOBJ() calls it for every primitive of any object reporting a non-zero face count.
    if (!mesh_faces.empty() && mesh_face_UUIDs.size() == context_ptr->getObjectPrimitiveUUIDs(objID).size()) {
        context_ptr->setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, mesh_face_UUIDs, {}, mesh_vertex_uv, helios::NORMAL_SOURCE_NONE);

        // Generate vertex normals from that topology. The blade is assembled from individual triangles,
        // which carry only a face normal, and Helios never synthesizes vertex normals implicitly -- so
        // without this a generated leaf renders faceted however the shading model is configured, and
        // the midrib fold and lateral curl that the lattice went to the trouble of describing read as
        // flat panels.
        //
        // The crease angle is 180 degrees, which smooths across every edge. A narrower angle splits the
        // shared vertices along any fold sharper than it -- a grass leaf's midrib is exactly such a
        // fold -- and a split rewrites the vertex array, which breaks consumers that read
        // getPolymeshObjectVertices() expecting the lattice's own row-major ordering. Smoothing the
        // whole blade is also what is wanted visually: the fold is a curve in the surface, not a
        // structural crease that should catch a hard highlight.
        constexpr float leaf_crease_angle_degrees = 180.f;
        context_ptr->computePolymeshObjectVertexNormals(objID, leaf_crease_angle_degrees);
    }

    return objID;
}

uint GeneralSphericalFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    return context_ptr->addSphereObject(5, make_vec3(0.5f, 0, 0), 0.5f, RGB::red);
}

uint AlmondFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AlmondHull.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint AlmondFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AlmondFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

//! Grow spurs from the lateral buds of a limb that did not grow a long shoot.
/** In almond and pistachio most lateral vegetative buds grow short shoots carrying a rosette of leaves, and these clothe the older wood
    and carry most of the tree's foliage. The bud-break probabilities decide which buds grow long shoots when a shoot leaves dormancy; a
    bud that lost that draw is dead, and here a fraction of those grow a shoot of type "spur" instead, so that the number of spurs can be
    set without changing the number of long shoots. A bud is considered once: its shoot type is set to "spur" whether or not it grows
    one, which is what marks it as considered. Buds of the current season's nodes are still dormant and are left for next spring's draw. */
static void growSpursFromUnbrokenBuds(const std::shared_ptr<Phytomer> &phytomer, float spur_fraction) {
    const Shoot &shoot = *phytomer->parent_shoot_ptr;
    if (spur_fraction <= 0.f || shoot.isdormant || shoot.shoot_type_label == "spur" || phytomer->shoot_index.x == 0) {
        return;
    }
    for (auto &petiole: phytomer->axillary_vegetative_buds) {
        for (auto &vbud: petiole) {
            if (vbud.state == BUD_DEAD && vbud.shoot_ID == uint(-1) && vbud.shoot_type_label != "spur") {
                vbud.shoot_type_label = "spur";
                if (shoot.context_ptr->randu() < spur_fraction) {
                    Phytomer::setVegetativeBudState(BUD_ACTIVE, vbud);
                }
            }
        }
    }
}

//! Fraction of the lateral vegetative buds left unbroken on an almond limb's year-old wood that grow a spur (see growSpursFromUnbrokenBuds()).
static constexpr float almond_spur_fraction = 0.2f;

void AlmondPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (phytomer->internode_length_max < 0.01) { // spurs
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->scaleLeafPrototypeScale(0.8);
        phytomer->setFloralBudState(BUD_DEAD);
        // A shoot of type "spur" sets its own number of leaves per season.
        if (phytomer->parent_shoot_ptr->shoot_type_label != "spur") {
            phytomer->parent_shoot_ptr->growth_state.max_nodes_per_season_growth_cap = 7;
        }
    }

    // blind nodes
    //    if( shoot_node_index<3 ){
    //        phytomer->setVegetativeBudState( BUD_DEAD );
    //        phytomer->setFloralBudState( BUD_DEAD );
    //    }
}

void AlmondPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
}

void AlmondSpurPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
    growSpursFromUnbrokenBuds(phytomer, almond_spur_fraction);
}

uint AppleFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AppleFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint AppleFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AlmondFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

//! Fraction of the lateral vegetative buds left unbroken on an apple limb's year-old wood that grow a spur (see growSpursFromUnbrokenBuds()).
static constexpr float apple_spur_fraction = 0.2f;
//! Nodes at the base of each season's growth unit of a long shoot whose axillary buds never flower.
/** The basal buds of an apple growth unit, in the axils of its small first leaves, mostly stay latent. */
static constexpr uint apple_axillary_flowering_basal_nodes = 4;
//! Nodes below the tip of a long shoot whose axillary buds never flower.
/** The most distal lateral buds of a one-year-old apple shoot grow into long shoots (acrotony), and its axillary flower buds lie below
    them. These are the nodes where the proleptic bud-break probability is still well above its minimum. */
static constexpr uint apple_axillary_flowering_distal_nodes = 3;
//! Probability that a node of a long shoot between the basal and distal zones carries axillary flower buds.
static constexpr float apple_axillary_flowering_probability = 0.3f;
//! Light exposure, under the past summer's canopy, below which an apple flower bud does not form.
/** Flower bud formation in apple falls off steeply in the shade of the crown interior, which is why a mature tree bears its fruit in the
    outer, well-lit shell of the canopy. Exposure is that of PlantArchitecture::getSeasonPeakLightExposureAtPoint(), from 0 (fully
    shaded) to 1 (unshaded). */
static constexpr float apple_flowering_light_exposure_min = 0.45f;
//! Light exposure above which an apple flower bud holds its full cluster of flowers.
static constexpr float apple_flowering_light_exposure_full = 0.8f;

//! Remove the flowers that a node's light would not have supported
/** Between the two exposure thresholds a cluster keeps a proportional share of its flowers. The flowers kept are always the first ones of
    the cluster and the light is that of the past summer, so applying this on every dormant time step changes nothing after the first.
    \param[in] phytomer Phytomer whose dormant floral buds are thinned.
    \param[in] terminal Whether to thin the terminal cluster rather than the axillary ones.
    \param[in] light_exposure Light exposure at the node.
*/
static void thinAppleFlowerClusterForLight(const std::shared_ptr<Phytomer> &phytomer, bool terminal, float light_exposure) {
    float kept_fraction = std::clamp((light_exposure - apple_flowering_light_exposure_min) / (apple_flowering_light_exposure_full - apple_flowering_light_exposure_min), 0.f, 1.f);
    // The terminal buds of a shoot are each stored as their own petiole group, so they are counted as one cluster across groups; the
    // axillary buds of a petiole are a cluster of their own.
    std::vector<std::vector<FloralBud *>> clusters;
    if (terminal) {
        clusters.emplace_back();
    }
    for (auto &petiole: phytomer->floral_buds) {
        if (!terminal) {
            clusters.emplace_back();
        }
        for (auto &fbud: petiole) {
            if (fbud.isterminal == terminal) {
                clusters.back().push_back(&fbud);
            }
        }
    }
    for (auto &cluster: clusters) {
        size_t kept = size_t(std::ceil(kept_fraction * float(cluster.size())));
        for (size_t i = kept; i < cluster.size(); i++) {
            if (cluster.at(i)->state == BUD_DORMANT) {
                phytomer->setFloralBudState(BUD_DEAD, *cluster.at(i));
            }
        }
    }
}

void ApplePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    const Shoot &shoot = *phytomer->parent_shoot_ptr;

    // An apple spur flowers only from its terminal mixed bud, which is added when the spur goes dormant and is not touched here.
    if (shoot.shoot_type_label == "spur") {
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->scaleLeafPrototypeScale(0.8);
        phytomer->setFloralBudState(BUD_DEAD);
        return;
    }

    // A lateral that breaks far below its parent's tip is held at the proleptic internode_length_min of 1 cm (the threshold sits between
    // that and the next length the 4 mm/node decay gives, 1.2 cm). It grows as a short shoot: a few nodes a season, ending each season in
    // the terminal flower cluster of a long shoot, and never branching. Its unbroken buds are marked as already considered for a spur, so
    // that growSpursFromUnbrokenBuds() does not grow spurs out of a short shoot.
    if (phytomer->internode_length_max < 0.011f) {
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        for (auto &petiole: phytomer->axillary_vegetative_buds) {
            for (auto &vbud: petiole) {
                vbud.shoot_type_label = "spur";
            }
        }
        phytomer->scaleLeafPrototypeScale(0.8);
        phytomer->parent_shoot_ptr->growth_state.max_nodes_per_season_growth_cap = 6;
    }

    // Axillary flower buds of a long shoot. The basal latent zone is decided here, by position within the season's growth unit; the
    // distal zone depends on where the shoot ends the season and is applied in ApplePhytomerCallbackFunction(). nodes_this_season is
    // only counted from the shoot's first dormancy onward, so during the shoot's first season its node index is used instead.
    uint growth_unit_node_index = shoot.dormancy_cycles == 0 ? shoot_node_index : shoot.nodes_this_season;
    if (growth_unit_node_index < apple_axillary_flowering_basal_nodes || shoot.context_ptr->randu() >= apple_axillary_flowering_probability) {
        phytomer->setFloralBudState(BUD_DEAD);
    }
}

void ApplePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    growSpursFromUnbrokenBuds(phytomer, apple_spur_fraction);

    const Shoot &shoot = *phytomer->parent_shoot_ptr;
    if (!shoot.isdormant) {
        return;
    }

    // Flower buds are thinned by the light the node stood in over the past summer, spur clusters and long shoots alike.
    float light_exposure = shoot.plantarchitecture_ptr->getSeasonPeakLightExposureAtPoint(shoot.plantID, phytomer->getInternodeNodePositions().back());
    thinAppleFlowerClusterForLight(phytomer, false, light_exposure);
    if (phytomer->shoot_index.x == phytomer->shoot_index.y - 1) {
        thinAppleFlowerClusterForLight(phytomer, true, light_exposure);
    }

    // Once a long shoot has stopped for the season its tip is known, and the axillary flower buds in the distal zone are removed. Only
    // this season's nodes still hold dormant axillary flower buds, so the rule does not reach older wood and applying it on every
    // dormant time step changes nothing after the first.
    if (shoot.shoot_type_label == "spur") {
        return;
    }
    int nodes_below_tip = phytomer->shoot_index.y - phytomer->shoot_index.x - 1;
    if (nodes_below_tip >= int(apple_axillary_flowering_distal_nodes)) {
        return;
    }
    for (auto &petiole: phytomer->floral_buds) {
        for (auto &fbud: petiole) {
            if (!fbud.isterminal && fbud.state == BUD_DORMANT) {
                phytomer->setFloralBudState(BUD_DEAD, fbud);
            }
        }
    }
}

uint AsparagusLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {

    // Drawn from the prototype's own generator rather than the Context's, so that the shape of a cladode belongs to
    // the prototype the way every other species' blade shape does. Building the unique prototypes redirects that
    // generator at a private stream, and a cladode drawn straight from the Context would not follow it - a reloaded
    // plant would then rebuild differently-curved cladodes than the ones that were saved.
    std::minstd_rand0 *generator = prototype_parameters->getRandomGenerator();
    if (generator == nullptr) {
        helios_runtime_error("ERROR (AsparagusLeafPrototype): Leaf prototype has no random number generator. Construct LeafPrototype with the Context's generator, which can be retrieved with Context::getRandomGenerator().");
    }
    std::uniform_real_distribution<float> unif_distribution;
    auto randu = [&](float minval, float maxval) { return minval + unif_distribution(*generator) * (maxval - minval); };

    float curve_magnitude = randu(0.f, 0.2f);

    std::vector<vec3> nodes;
    nodes.push_back(make_vec3(0, 0, 0));
    nodes.push_back(make_vec3(randu(0.4f, 0.7f), 0, -0.25f * curve_magnitude));
    nodes.push_back(make_vec3(0.95, 0, -0.9f * curve_magnitude));
    nodes.push_back(make_vec3(1, 0, -curve_magnitude));

    std::vector<float> radius;
    radius.push_back(0.015);
    radius.push_back(0.015);
    radius.push_back(0.015);
    radius.push_back(0.0);

    std::vector<RGBcolor> colors;
    colors.push_back(RGB::forestgreen);
    colors.push_back(RGB::forestgreen);
    colors.push_back(RGB::forestgreen);
    colors.push_back(RGB::forestgreen);

    uint objID = context_ptr->addTubeObject(8, nodes, radius, colors);
    context_ptr->rotateObject(objID, randu(0.f, 2.f * PI_F), "x");
    return objID;
}

void AsparagusPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // blind nodes
    if (shoot_node_index <= 2) {
        phytomer->scaleLeafPrototypeScale(0.6);
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
}

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanLeaf_unifoliate.obj").string().c_str(), true);

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs;
    if (compound_leaf_index == 0) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanLeaf_tip.obj").string().c_str(), true);
    } else if (compound_leaf_index < 0) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanLeaf_right.obj").string().c_str(), true);
    } else {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanLeaf_left.obj").string().c_str(), true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanPod.obj").string().c_str(), true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs;
    if (flower_is_open) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanFlower_open_white.obj").string().c_str(), true);
    } else {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanFlower_closed_white.obj").string().c_str(), true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void BeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index > 5 || phytomer->rank > 1) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    } else {
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.6 + 0.4 * plant_age / 8.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    if (phytomer->rank == 0) {
        float inode_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 10.f);
        phytomer->scaleInternodeMaxLength(inode_scale);
    }
}

uint BindweedFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BindweedFlower.obj").string().c_str(), true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BougainvilleaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BougainvilleaFlower.obj").string().c_str(), true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CapsicumFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::string OBJ_file;
    if (context_ptr->randn() < 0.4) {
        OBJ_file = helios::resolvePluginAsset("plantarchitecture", "assets/obj/CapsicumFruit_green.obj").string().c_str();
    } else {
        OBJ_file = helios::resolvePluginAsset("plantarchitecture", "assets/obj/CapsicumFruit_red.obj").string().c_str();
    }

    std::vector<uint> UUIDs = context_ptr->loadOBJ(OBJ_file.c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

void CapsicumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index < 6 && phytomer->rank == 0) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->removeLeaf();
    }

    if (phytomer->rank >= 2) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = std::min(1.f, 0.6f + 0.4f * shoot_node_index / 5.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    if (phytomer->rank == 0) {
        float inode_scale = std::min(1.f, 0.05f + 0.95f * plant_age / 15.f);
        phytomer->scaleInternodeMaxLength(inode_scale);
    }
}

uint CheeseweedLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CheeseweedLeaf.obj").string().c_str(), true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaLeaf_unifoliate.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs;
    if (compound_leaf_index < 0) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaLeaf_right_highres.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else if (compound_leaf_index == 0) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaLeaf_tip_highres.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaLeaf_left_highres.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaPod.obj").string().c_str(), make_vec3(0., 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs;
    if (flower_is_open) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaFlower_open_yellow.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaFlower_closed_yellow.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void CowpeaPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // Cowpea branches from its basal nodes and flowers from the upper nodes of every axis, which is what holds
    // the pods above the foliage (IBPGR "peduncle position: mostly above canopy"). Floral buds were previously
    // allowed on every node of a rank-2 shoot and on every node above the fifth of the main stem and its
    // branches, so most peduncles started at about half the canopy height and the pods they carried hung
    // back down into the leaves. Floral buds are now confined to the upper half of each shoot, on every rank;
    // the basal nodes carry vegetative buds on the main stem and its primary branches (this is where the
    // branches come from) and nothing on higher-order shoots.
    const uint first_floral_node = std::max(5u, shoot_max_nodes / 2);
    const bool can_branch = phytomer->rank <= 1;
    if (shoot_node_index >= first_floral_node) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    } else if (can_branch) {
        phytomer->setFloralBudState(BUD_DEAD);
    } else {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.6 + 0.4 * plant_age / 8.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    if (phytomer->rank == 0) {
        float inode_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 10.f);
        phytomer->scaleInternodeMaxLength(inode_scale);
    }
}

GrapeClusterParameters GrapeClusterParameters::fromBuildParameters(const std::map<std::string, float> &build_parameters) {
    GrapeClusterParameters parameters;
    parameters.berry_diameter = getBuildParameterValue(build_parameters, "berry_diameter", parameters.berry_diameter, 0.004f, 0.03f, "grape berry diameter in meters");
    parameters.cluster_berry_count = getBuildParameterValue(build_parameters, "cluster_berry_count", parameters.cluster_berry_count, 10.f, 400.f, "mean number of berries in a grape cluster");
    return parameters;
}

std::vector<std::string> GrapeClusterParameters::buildParameterNames() {
    return {"berry_diameter", "cluster_berry_count"};
}

uint GrapevineFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    // Mature wine-grape cluster (Cabernet Sauvignon), built along +x from its attachment at the origin, in meters, so the fruit_prototype_scale that goes with it is 1. Defaults target ~90 berries of
    // ~12 mm (about 1 g; Przic & Markovic 2014, Travanic-Fuentes et al. 2024) in a conical cluster ~12 cm long and ~7 cm wide with medium compactness and 0-2 wings (OIV 202/204/208/209). Berries are
    // seeded on a rachis with ~15 primary branches (Dunn & Martin 2007) whose reach shrinks toward the tip, then relaxed until they pack against their neighbors.
    //
    // The cluster is laid out in units in which a cluster of the reference berry count is one unit long and holds berries of radius berry_radius_mean, and is scaled to meters at the end. Its size
    // therefore follows from its berries: a cluster grows with the cube root of its berry count, and in proportion to berry diameter.

    const GrapeClusterParameters cluster_parameters = GrapeClusterParameters::fromBuildParameters(build_parameters);
    const float berry_diameter = cluster_parameters.berry_diameter;
    const float cluster_berry_count = cluster_parameters.cluster_berry_count;

    constexpr int reference_berry_count = 90;
    constexpr float berry_radius_mean = 0.05f; // 12 mm berries in a 12 cm cluster
    constexpr float berry_radius_cv = 0.07f; // ~20% CV in berry mass
    constexpr uint berry_subdivisions = 8;
    const float meters_per_unit = 0.5f * berry_diameter / berry_radius_mean;

    // Berry count varies by -17% to +17% about the mean between clusters
    const int berry_count = context_ptr->randu(int(std::round(cluster_berry_count * 75.f / 90.f)), int(std::round(cluster_berry_count * 105.f / 90.f)));
    const float cluster_length = std::cbrt(float(berry_count) / float(reference_berry_count));

    // ---- Rachis skeleton ---- //

    const float shoulder_radius = 0.33f * cluster_length; // envelope radius at the shoulders
    const int primary_branch_count = context_ptr->randu(13, 17);
    const float first_branch_position = 0.06f * cluster_length;
    const float last_branch_position = 0.85f * cluster_length;

    std::vector<float> branch_position(primary_branch_count);
    std::vector<float> branch_reach(primary_branch_count);
    std::vector<vec3> branch_direction(primary_branch_count);
    std::vector<float> branch_length(primary_branch_count);
    std::vector<float> branch_berry_weight(primary_branch_count);

    const bool has_wing = context_ptr->randu() < 0.5f;
    const float azimuth_offset = context_ptr->randu(0.f, 2.f * PI_F);
    for (int branch = 0; branch < primary_branch_count; branch++) {
        branch_position.at(branch) = first_branch_position + (last_branch_position - first_branch_position) * (float(branch) + 0.5f * context_ptr->randu()) / float(primary_branch_count);
        // Conical outline: basal branches reach farthest from the rachis
        float reach = (shoulder_radius - berry_radius_mean) * (1.f - (branch_position.at(branch) - first_branch_position) / (cluster_length - first_branch_position));
        reach *= 1.f + 0.15f * context_ptr->randn();
        reach = std::max(reach, 0.3f * berry_radius_mean);

        float insertion_angle = deg2rad(context_ptr->randu(45.f, 65.f)); // from the rachis axis
        float azimuth = deg2rad(137.5f) * float(branch) + context_ptr->randu(-0.3f, 0.3f) + azimuth_offset;
        float length = reach / sin(insertion_angle);
        float berry_weight = reach * reach + std::pow(1.2f * berry_radius_mean, 2);

        // A wing is an enlarged basal branch that hangs alongside the main cluster
        if (has_wing && branch == 0) {
            insertion_angle = deg2rad(22.f);
            length = 0.4f * cluster_length;
            berry_weight *= 2.5f;
        }

        branch_reach.at(branch) = reach;
        branch_length.at(branch) = length;
        branch_berry_weight.at(branch) = berry_weight;
        branch_direction.at(branch) = make_vec3(cos(insertion_angle), sin(insertion_angle) * cos(azimuth), sin(insertion_angle) * sin(azimuth));
    }

    // One berry terminates the rachis; the rest are shared among branches by weight (largest remainder)
    const int branch_berries_total = berry_count - 1;
    float total_weight = 0.f;
    for (float weight: branch_berry_weight) {
        total_weight += weight;
    }
    std::vector<int> branch_berry_count(primary_branch_count);
    std::vector<std::pair<float, int>> remainders(primary_branch_count);
    int assigned = 0;
    for (int branch = 0; branch < primary_branch_count; branch++) {
        float share = branch_berry_weight.at(branch) / total_weight * float(branch_berries_total);
        branch_berry_count.at(branch) = int(std::floor(share));
        remainders.at(branch) = {share - std::floor(share), branch};
        assigned += branch_berry_count.at(branch);
    }
    std::sort(remainders.rbegin(), remainders.rend()); // largest remainder first
    for (int i = 0; i < branch_berries_total - assigned; i++) {
        branch_berry_count.at(remainders.at(i).second)++;
    }

    // ---- Seed berries on the skeleton ---- //

    // Each berry's anchor is its pedicel tip; secondary branching is represented by scatter about the primary branch
    std::vector<vec3> anchors;
    anchors.reserve(berry_count);
    anchors.push_back(make_vec3(last_branch_position + berry_radius_mean, 0, 0));
    for (int branch = 0; branch < primary_branch_count; branch++) {
        const vec3 &direction = branch_direction.at(branch);
        for (int berry = 0; berry < branch_berry_count.at(branch); berry++) {
            float along = 1.f - 0.75f * std::pow(context_ptr->randu(), 1.5f); // biased toward the distal end of the branch
            vec3 scatter = make_vec3(context_ptr->randn(), context_ptr->randn(), context_ptr->randn());
            scatter = scatter - direction * (scatter * direction);
            vec3 anchor = make_vec3(branch_position.at(branch), 0, 0) + direction * (branch_length.at(branch) * along) + scatter * (0.35f * branch_length.at(branch) * along);
            anchors.push_back(anchor);
        }
    }

    std::vector<float> berry_radii(berry_count);
    for (float &radius: berry_radii) {
        radius = berry_radius_mean * clamp(1.f + berry_radius_cv * context_ptr->randn(), 0.8f, 1.2f);
    }

    // ---- Pack the berries ---- //

    // Berries are drawn in toward the rachis and slightly toward the tip (hanging), held loosely to their pedicel tips, and pushed apart wherever they
    // overlap. The final iterations only resolve overlaps so the berries end up touching but not interpenetrating.
    constexpr int packing_iterations = 300;
    constexpr int settling_iterations = 100;
    constexpr float compaction_rate = 0.012f;
    constexpr float pedicel_spring = 0.02f;

    std::vector<vec3> centers = anchors;
    for (int iteration = 0; iteration < packing_iterations + settling_iterations; iteration++) {
        if (iteration < packing_iterations) {
            for (int berry = 0; berry < berry_count; berry++) {
                vec3 &center = centers.at(berry);
                center.y *= 1.f - compaction_rate;
                center.z *= 1.f - compaction_rate;
                center.x += 0.3f * compaction_rate * berry_radius_mean;
                center = center + (anchors.at(berry) - center) * pedicel_spring;
            }
        }
        for (int pass = 0; pass < 2; pass++) {
            std::vector<vec3> displacement(berry_count, make_vec3(0, 0, 0));
            for (int i = 0; i < berry_count; i++) {
                for (int j = i + 1; j < berry_count; j++) {
                    vec3 separation = centers.at(i) - centers.at(j);
                    float distance = separation.magnitude();
                    float overlap = berry_radii.at(i) + berry_radii.at(j) - distance;
                    if (overlap > 0.f && distance > 0.f) {
                        vec3 push = separation * (0.5f * overlap / distance);
                        displacement.at(i) = displacement.at(i) + push;
                        displacement.at(j) = displacement.at(j) - push;
                    }
                }
            }
            for (int berry = 0; berry < berry_count; berry++) {
                centers.at(berry) = centers.at(berry) + displacement.at(berry);
            }
        }
        // Keep berries below the attachment point and within the cluster length
        for (int berry = 0; berry < berry_count; berry++) {
            centers.at(berry).x = clamp(centers.at(berry).x, berry_radii.at(berry), cluster_length - 0.5f * berry_radii.at(berry));
        }
    }

    // ---- Geometry ---- //

    std::vector<uint> UUIDs;
    UUIDs.reserve(berry_count * 2 * berry_subdivisions * (berry_subdivisions - 1));
    for (int berry = 0; berry < berry_count; berry++) {
        std::vector<uint> UUIDs_berry = context_ptr->addSphere(berry_subdivisions, meters_per_unit * centers.at(berry), meters_per_unit * berry_radii.at(berry), make_RGBcolor(0.053, 0.076, 0.098));
        UUIDs.insert(UUIDs.end(), UUIDs_berry.begin(), UUIDs_berry.end());
    }

    context_ptr->setPrimitiveData(UUIDs, "object_label", "fruit");

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

// uint GrapevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
//     std::vector<uint> UUIDs = context_ptr->loadOBJ( helios::resolvePluginAsset("plantarchitecture", "assets/obj/OliveFlower_open.obj").string().c_str(), make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
//     uint objID = context_ptr->addPolymeshObject( UUIDs );
//     return objID;
// }

void GrapevinePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // A grapevine shoot bears its clusters opposite the leaves at the third to sixth nodes from its base; the two basal
    // nodes carry none, and above the sixth the same positions bear tendrils. A shoot carries one to three clusters, and
    // with three, two sit on adjacent nodes, the next node has none and the following node has the third (Hellman,
    // Grapevine Structure and Function). The candidate nodes are therefore the third, fourth and sixth.
    const bool is_cluster_node = shoot_node_index == 2 || shoot_node_index == 3 || shoot_node_index == 5;
    if (!is_cluster_node) {
        phytomer->setFloralBudState(BUD_DEAD);
    }
    // Summer laterals arise at most nodes of a fruiting shoot and are most vigorous basally, dying out
    // toward the tip. Keeping them below node 8 is what gives a vine the 27-35% of its leaf area that
    // destructive defoliation finds on laterals (Navarrete 2015, VSP Pinot noir).
    if (phytomer->rank >= 1 && shoot_node_index >= 8) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
    // A lateral does not fruit. A lateral is recognised by what bears it, a fruiting shoot or another lateral, rather than
    // by its rank: the fruiting shoots of a cane-trained vine grow from canes that themselves come off the trunk, so they are
    // of rank 2, and a test on rank killed every floral bud on the vine.
    const Shoot &shoot = *phytomer->parent_shoot_ptr;
    if (shoot.parent_shoot_ID >= 0) {
        const std::string &bearer_type = shoot.plantarchitecture_ptr->getPlantShoot(shoot.plantID, uint(shoot.parent_shoot_ID))->shoot_type_label;
        if (bearer_type == "grapevine_shoot" || bearer_type == "grapevine_shoot_mirrored" || bearer_type == "grapevine_lateral") {
            phytomer->setFloralBudState(BUD_DEAD);
        }
    }
    // Nor does a lateral itself bear further laterals: second-order branching is what would run the canopy away. As above, a
    // lateral is recognised by its type rather than its rank, which differs between training systems: the fruiting shoots of a
    // Wye vine grow from cordons borne on arms off the trunk, a rank deeper than those of a VSP vine.
    if (shoot.shoot_type_label == "grapevine_lateral") {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
}

// ---- Grapevine fruit-zone leaf removal ---- //
//
// Leaves around the clusters are pulled by hand or machine at about fruit set, to open the fruit zone to light and air. The leaves of
// the basal nodes of each fruiting shoot, up to and including the node of the highest cluster, are removed.

//! Number of basal nodes of a fruiting shoot whose leaves are pulled. Clusters are borne up to the sixth node (see GrapevinePhytomerCreationFunction()).
static constexpr uint grapevine_leaf_pulling_nodes = 6;
//! Age (days) of a fruiting shoot at which its fruit-zone leaves are pulled. Fruit set of the library vines is 45 days after a shoot's buds become active.
static constexpr float grapevine_leaf_pulling_shoot_age = 45.f;

void GrapevinePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    const Shoot &shoot = *phytomer->parent_shoot_ptr;
    if (shoot.shoot_type_label != "grapevine_shoot" && shoot.shoot_type_label != "grapevine_shoot_mirrored") {
        return;
    }
    // A lateral borne on a fruiting shoot keeps its leaves. Where laterals have no shoot type of their own they carry the label of the shoot that bears them.
    if (shoot.parent_shoot_ID >= 0) {
        const std::string &bearer_type = shoot.plantarchitecture_ptr->getPlantShoot(shoot.plantID, uint(shoot.parent_shoot_ID))->shoot_type_label;
        if (bearer_type == "grapevine_shoot" || bearer_type == "grapevine_shoot_mirrored") {
            return;
        }
    }

    // The whole fruit zone is pulled in one pass, timed from the age of the shoot rather than of each leaf.
    if (uint(phytomer->shoot_index.x) < grapevine_leaf_pulling_nodes && phytomer->hasLeaf() && shoot.phytomers.front()->age >= grapevine_leaf_pulling_shoot_age) {
        phytomer->removeLeaf();
    }
}

uint MaizeTasselPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {

    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/MaizeTassel.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

uint MaizeEarPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {

    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/MaizeEar.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

//! Turns the flag leaf of a library grass to a given angle off its culm
/**
 * A grass leaf has no petiole, so it leaves the culm at the leaf pitch it was drawn with, which Phytomer::petiole_pitch holds. The flag leaf is turned from there to the given angle with
 * Phytomer::rotateLeaf(), whose pitch the leaf keeps when it is re-posed, as it is when it droops or is read back from XML.
 * \param[in] phytomer Phytomer bearing the flag leaf.
 * \param[in] angle_off_culm_degrees Angle between the culm and the base of the flag leaf blade, in degrees.
 */
static void setGrassFlagLeafAngle(const std::shared_ptr<Phytomer> &phytomer, float angle_off_culm_degrees) {
    if (phytomer->leaf_objIDs.empty() || phytomer->leaf_objIDs.front().empty()) {
        return;
    }
    // A positive pitch turns the blade toward its upper face, which on a leaf without a petiole is toward the culm.
    const float drawn_angle_off_culm = std::fabs(phytomer->petiole_pitch.front());
    phytomer->rotateLeaf(0, 0, make_AxisRotation(drawn_angle_off_culm - deg2rad(angle_off_culm_degrees), 0.f, 0.f));
}

void MaizePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // set leaf scale based on position along the shoot
    float scale = 1.f;
    if (shoot_node_index <= 5) {
        scale = fmin(1.f, 0.7 + 0.3 * float(shoot_node_index) / 5.f);
        phytomer->scaleInternodeMaxLength(scale);
    }

    // Maize leaf area follows a bell-shaped profile that peaks at or just below the ear and falls away sharply over the last few ranks, so the upper leaves are much smaller than the ear leaf and the flag
    // leaf is the smallest blade on the plant. Fan et al. (2021, Journal of Agricultural Science 158:676) fit area = y0*exp(-(rank - x0)^2 / 2a^2) to two commercial hybrids: ZD958 (22 leaves, x0 = 13.8,
    // a = 4.5) and XY335 (21 leaves, x0 = 13.9, a = 4.3). Evaluating those over the top five ranks and taking the square root -- the prototype scale is linear, area goes as its square -- gives a decline
    // that is very nearly straight in ranks-from-the-top, and the ramp below is a least-squares fit to it (residual 6e-5). It reaches full size about six ranks down, which is where the two fitted profiles
    // are within a few percent of their peak.
    //
    // The previous ramp was far too shallow: it left the top three leaves at full size and the flag leaf at 0.767, which is 0.59 of the largest leaf's AREA against the 0.19-0.26 the two hybrids give.
    const float ranks_from_top = float(phytomer->shoot_index.z - 1 - int(shoot_node_index));
    scale = fmin(scale, fmin(1.f, 0.475f + 0.092f * ranks_from_top));

    phytomer->scaleLeafPrototypeScale(scale);

    // Upper leaves are stiffer than lower ones, so they hold the erect attitude their insertion angle gives them instead of arcing over to the horizontal. This is the erectophile canopy commercial maize
    // has been bred toward - roughly 20-50 degrees from vertical above the ear against 60-80 below it - and it is what lets a dense stand intercept light without the top of the canopy shading the bottom.
    //
    // Stiffness is the lever rather than insertion angle: the leaves already leave the stalk at a reasonably erect 50-55 degrees, and it is the blade's own curvature that takes 35-40 degrees back out of
    // that. Making the upper leaves less compliant leaves the lower ones free to keep the arc they should have.
    if (phytomer->shoot_index.z > 1) {
        const float rank_fraction = float(shoot_node_index) / float(phytomer->shoot_index.z - 1);
        // Full stiffening at the top of the canopy, fading out below the ear so the lower leaves are untouched.
        phytomer->leaf_flexibility *= 1.f - 0.75f * fmax(0.f, rank_fraction - 0.45f) / 0.55f;
    }

    // Maize forms an axillary ear meristem at every above-ground node except the upper six to eight
    // below the tassel, and apical dominance means only the uppermost eligible node develops a
    // harvestable ear. Anchoring the ear to the TOP of the shoot rather than to fixed node indices
    // keeps it in the right part of the canopy when max_nodes is changed; an absolute window is only
    // correct for one particular plant height.
    //
    // Computed in signed int deliberately: shoot_node_index and shoot_max_nodes are uint, so a plant
    // shorter than nodes_below_tassel would wrap to a huge value in unsigned arithmetic. In signed
    // arithmetic apical_ear_node simply goes negative and never matches, leaving a short shoot earless.
    constexpr int nodes_below_tassel = 6;
    const int apical_ear_node = int(shoot_max_nodes) - nodes_below_tassel;

    // Number of ears borne on the plant. Commercial hybrids are near-strictly single-eared, so this
    // is 1; a second sub-apical ear (seen at low planting density and high nitrogen) is obtained by
    // setting this to 2, which additionally bears an ear at the node just below the apical one.
    constexpr int ears_per_plant = 1;

    const bool is_ear = shoot_max_nodes > 0 && (int(shoot_node_index) == apical_ear_node || (ears_per_plant >= 2 && int(shoot_node_index) == apical_ear_node - 1));

    if (is_ear) {
        phytomer->phytomer_parameters.inflorescence.flowers_per_peduncle = 1;
        phytomer->phytomer_parameters.inflorescence.fruit_prototype_function = MaizeEarPrototype;
        phytomer->phytomer_parameters.inflorescence.fruit_prototype_scale = 0.2;
        phytomer->phytomer_parameters.peduncle.length = 0.05f;
        phytomer->phytomer_parameters.peduncle.radius = 0.01;
        phytomer->phytomer_parameters.peduncle.pitch = 5;
        phytomer->setFloralBudState(BUD_ACTIVE);
    } else {
        phytomer->phytomer_parameters.inflorescence.fruit_prototype_function = MaizeTasselPrototype;
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // The terminal phytomer bears the flag leaf, and the terminal floral bud carrying the tassel is anchored to the same node tip (Shoot::addTerminalFloralBud) with maize's peduncle.pitch of 0, so a
    // flag leaf leaving the culm nearly along it would pass through the tassel. It is set to 23 degrees off the culm.
    //
    // The angle stays smaller than the 30 degrees of sorghum, because the two species hold their flag leaves differently: sorghum's is the most horizontal leaf on the plant,
    // whereas maize is bred for an erectophile upper canopy -- 22.9 degrees from vertical for the leaves above the ear against 41.9 for those below it (Tang et al. 2021, Euphytica 217:75, measured on the
    // hybrid Pioneer 335), a gradient that does not reverse at the top.
    //
    // What has to be preserved is that the flag leaf is the most erect BLADE, which is what Tang et al. measure and not the same thing as the most erect insertion. In this model the erectophile gradient
    // comes almost entirely from the leaf_flexibility taper above rather than from insertion angle: every rank is inserted from the same U(16,30) distribution, and it is the stiffening of the upper leaves
    // that keeps their blades from arcing over. The flag leaf is the stiffest leaf on the plant, so it holds a blade inclination near 27 degrees from vertical against 38 for the leaf below it and 53 near
    // the base -- an 11 degree margin that an angle of this size does not close.
    if (shoot_max_nodes > 0 && shoot_node_index + 1 == shoot_max_nodes) {
        setGrassFlagLeafAngle(phytomer, 23.f);
    }
}

uint OliveLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {

    std::vector<uint> UUIDs_upper = context_ptr->addTile(make_vec3(0.5, 0, 0), make_vec2(1, 0.2), nullrotation, make_int2(prototype_parameters->subdivisions, prototype_parameters->subdivisions),
                                                         helios::resolvePluginAsset("plantarchitecture", "assets/textures/OliveLeaf_upper.png").string().c_str());
    std::vector<uint> UUIDs_lower = context_ptr->addTile(make_vec3(0.5, 0, -1e-4), make_vec2(1, 0.2), nullrotation, make_int2(prototype_parameters->subdivisions, prototype_parameters->subdivisions),
                                                         helios::resolvePluginAsset("plantarchitecture", "assets/textures/OliveLeaf_lower.png").string().c_str());
    context_ptr->rotatePrimitive(UUIDs_lower, M_PI, "x");

    UUIDs_upper.insert(UUIDs_upper.end(), UUIDs_lower.begin(), UUIDs_lower.end());
    uint objID = context_ptr->addPolymeshObject(UUIDs_upper);
    return objID;
}

uint OliveFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/OliveFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    std::vector<uint> UUIDs_fruit = context_ptr->filterPrimitivesByData(context_ptr->getObjectPrimitiveUUIDs(objID), "object_label", "fruit");
    context_ptr->setPrimitiveColor(UUIDs_fruit, make_RGBcolor(0.65, 0.7, 0.4)); // green
    return objID;
}

uint OliveFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/OliveFlower_open.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void OlivePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {
}

void OlivePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    if (phytomer->isdormant) {
        if (phytomer->shoot_index.x < phytomer->shoot_index.y - 8) {
            phytomer->setFloralBudState(BUD_DEAD);
        }
    }
}

uint PistachioFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/PistachioNut.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint PistachioFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/OliveFlower_open.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void PistachioPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // blind nodes
    if (shoot_node_index == 0) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // A lateral that breaks well below its parent's tip gets short internodes and is a spur: a few leaves a season, and no branching.
    if (phytomer->internode_length_max < 0.01) {
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->parent_shoot_ptr->growth_state.max_nodes_per_season_growth_cap = 6;
    }
}

// ---- Pistachio orchard training ---- //
//
// California pistachio is trained in its first seasons after the scion is headed (UC ANR, Sample Costs to Establish and Produce
// Pistachios, San Joaquin Valley South, 2020): the primary scaffolds are tipped during their first growing season to make them branch,
// and in the following dormant season the secondaries are thinned to 2-3 per primary and headed to 11-13 in; after that the tree is left
// to grow. The values below reproduce the trees of the QSM calibration in projects/QSMcalibration.

//! Length (m) at which a scaffold is tipped in its first growing season.
static constexpr float pistachio_scaffold_tip_length = 0.45f;
//! Length (m) to which the secondaries are headed in the first winter.
static constexpr float pistachio_secondary_head_length = 0.30f;
//! Secondaries kept on each scaffold in the first winter.
static constexpr size_t pistachio_secondaries_per_scaffold = 3;
//! Nodes below a cut on which one bud each is made to break.
static constexpr uint pistachio_buds_released_per_cut = 2;
//! Nodes below a scaffold's tipping cut on which one bud each is made to break. More than the secondaries kept, so that the first-winter
//! thinning has a choice: the scaffold's other buds do not break until the following spring.
static constexpr uint pistachio_buds_released_at_tipping = 5;
//! Fraction of the lateral vegetative buds left unbroken on a limb's year-old wood that grow a spur (see growSpursFromUnbrokenBuds()).
static constexpr float pistachio_spur_fraction = 0.f;
//! Number of nodes to keep so that a shoot is cut back to about `length` metres (the first node whose far end reaches that length is kept).
static uint pistachioNodesToKeep(const Shoot &shoot, float length) {
    float arc = 0;
    bool first_vertex = true;
    helios::vec3 previous;
    for (uint node = 0; node < shoot.shoot_internode_vertices.size(); node++) {
        for (const helios::vec3 &vertex: shoot.shoot_internode_vertices.at(node)) {
            if (!first_vertex) {
                arc += (vertex - previous).magnitude();
            }
            previous = vertex;
            first_vertex = false;
        }
        if (arc >= length) {
            return node + 1;
        }
    }
    return uint(shoot.shoot_internode_vertices.size());
}

//! Cut a shoot back to about `length` metres, ending its growth. Returns the number of nodes kept.
/** The cut is deferred to the end of the time step (see PlantArchCallbackEdits), but the tip is stopped at once: removing phytomers only
    ends a shoot's growth when the removed tip already has its geometry, so a young tip could otherwise survive the cut and regrow. */
static uint pistachioCutToLength(Shoot &shoot, float length) {
    uint keep = pistachioNodesToKeep(shoot, length);
    if (keep < shoot.phytomers.size()) {
        shoot.plantarchitecture_ptr->pruneBranch(shoot.plantID, uint(shoot.ID), keep);
        shoot.meristem_is_alive = false;
    }
    return keep;
}

//! Make one vegetative bud on each of the top `node_count` nodes below `keep` break, standing in for the release from apical dominance a cut causes.
/** The bud-break probabilities cannot tell a cut from a natural shoot tip, so without this a headed shoot usually regrows from a single bud
    or none. Only valid while the shoot is growing: a bud set active in dormancy is drawn again when dormancy breaks. Node 0 is skipped
    (a blind node), and so are buds that have already grown a shoot. */
static void pistachioReleaseBudsBelow(Shoot &shoot, uint keep, uint node_count) {
    uint released = 0;
    for (int node = int(keep) - 1; node >= 1 && released < node_count; node--) {
        const std::shared_ptr<Phytomer> &phytomer = shoot.phytomers.at(node);
        bool released_here = false;
        for (auto &petiole: phytomer->axillary_vegetative_buds) {
            for (auto &vbud: petiole) {
                if (!released_here && vbud.shoot_ID == uint(-1)) {
                    // A bud already passed over for a spur (see growSpursFromUnbrokenBuds()) is growing a limb after all.
                    vbud.shoot_type_label = "proleptic";
                    Phytomer::setVegetativeBudState(BUD_ACTIVE, vbud);
                    released_here = true;
                }
            }
        }
        released++;
    }
}

void PistachioPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    growSpursFromUnbrokenBuds(phytomer, pistachio_spur_fraction);

    if (phytomer->isdormant) {
        if (phytomer->shoot_index.x <= phytomer->shoot_index.y - 4) {
            phytomer->setFloralBudState(BUD_DORMANT);
        }
    }

    // The training rules act once per shoot per time step, from its tip phytomer. Seasons are read from the shoot's own dormancy flag, not
    // the phytomer's: a tip phytomer is also marked dormant when its shoot reaches its node cap for the season, while the plant grows on.
    Shoot &shoot = *phytomer->parent_shoot_ptr;
    if (shoot.phytomers.empty() || phytomer != shoot.phytomers.back()) {
        return;
    }
    PlantArchitecture *plantarchitecture = shoot.plantarchitecture_ptr;

    const bool is_scaffold = shoot.shoot_type_label == "scaffold";
    const bool is_secondary = shoot.shoot_type_label != "spur" && shoot.parent_shoot_ID >= 0 && plantarchitecture->getPlantShoot(shoot.plantID, uint(shoot.parent_shoot_ID))->shoot_type_label == "scaffold";

    // Regrowth below a training cut: a scaffold tipped in its first season, or a secondary headed in its first winter, regrows that
    // growing season from the buds just below the cut. The release waits until the cut has been applied (cuts from a callback take
    // effect at the end of the time step), because a new shoot's internode length is set by its distance below the parent's tip at the
    // moment it grows: released while the cut tip is still there, the buds would count as lying below it and start as short stubs. It
    // also waits for the shoot to be out of dormancy, since the dormancy-break draw overwrites any bud set earlier. It stops by itself
    // once those buds have grown shoots. (Scaffolds are in their first dormancy cycle from planting, secondaries from their first winter.)
    if ((is_scaffold || is_secondary) && !shoot.meristem_is_alive && !shoot.isdormant && shoot.dormancy_cycles == 1) {
        const uint buds_released = is_scaffold ? pistachio_buds_released_at_tipping : pistachio_buds_released_per_cut;
        pistachioReleaseBudsBelow(shoot, uint(shoot.phytomers.size()), buds_released);

        // The shoots that grow from those buds are released from apical dominance, so they grow as leaders of the fork rather than as
        // laterals: each gets the full internode length of its shoot type. Without this only the top one did (a new shoot's internode
        // length shrinks with its distance below the parent's tip), and the one a node lower grew as a short stub, leaving each scaffold
        // with a single limb and the crown with gaps between them. The shoot's first internode, already built, keeps its length.
        for (uint childID: plantarchitecture->getChildShootIDs(shoot.plantID, uint(shoot.ID))) {
            const std::shared_ptr<Shoot> &child = plantarchitecture->getPlantShoot(shoot.plantID, childID);
            if (child->dormancy_cycles == 0 && child->parent_node_index + buds_released >= shoot.phytomers.size()) {
                child->internode_length_max_shoot_initial = child->shoot_parameters.internode_length_max.val();
            }
        }
    }

    if (is_scaffold) {
        // Tip each scaffold once, in the season it reaches the tipping length. Tipping removes the growing tip even when the node reaching
        // the tipping length is the tip itself and nothing is cut off.
        if (shoot.meristem_is_alive && !shoot.isdormant && shoot.calculateShootLength() >= pistachio_scaffold_tip_length) {
            static_cast<void>(pistachioCutToLength(shoot, pistachio_scaffold_tip_length));
            shoot.meristem_is_alive = false;
        }

        // First winter after the scaffolds grew (the builder's own dormancy at planting is the first cycle): thin the secondaries to the
        // longest few and head them. Run on every day of that winter, but a no-op once the cuts have been applied.
        if (shoot.isdormant && shoot.dormancy_cycles == 2) {
            std::vector<std::pair<float, uint>> secondaries;
            for (uint childID: plantarchitecture->getChildShootIDs(shoot.plantID, uint(shoot.ID))) {
                secondaries.emplace_back(plantarchitecture->getPlantShoot(shoot.plantID, childID)->calculateShootLength(), childID);
            }
            std::sort(secondaries.begin(), secondaries.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
            for (size_t i = 0; i < secondaries.size(); i++) {
                const std::shared_ptr<Shoot> &secondary = plantarchitecture->getPlantShoot(shoot.plantID, secondaries.at(i).second);
                if (i >= pistachio_secondaries_per_scaffold) {
                    plantarchitecture->pruneBranch(shoot.plantID, secondaries.at(i).second, 0);
                } else {
                    static_cast<void>(pistachioCutToLength(*secondary, pistachio_secondary_head_length));
                }
            }
        }
    }
}

uint PuncturevineFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/PuncturevineFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint RedbudFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/RedbudFlower_open.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

uint RedbudFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/RedbudPod.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

void RedbudPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {
}

void RedbudPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    // redbud has the shoot pattern that the first few nodes on the shoot are vegetative, then the rest are floral
    if (phytomer->isdormant) {
        int Nchild_shoots = randu(2, 4);
        if (phytomer->shoot_index.x < phytomer->shoot_index.y - Nchild_shoots) {
            phytomer->setVegetativeBudState(BUD_DEAD);
        } else {
            phytomer->setFloralBudState(BUD_DEAD);
        }
    }
}

uint RiceSpikePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/RiceGrain.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void RicePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // The flag leaf stands nearly along the culm.
    if (shoot_max_nodes > 0 && shoot_node_index + 1 == shoot_max_nodes) {
        setGrassFlagLeafAngle(phytomer, 5.f);
    }
}

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    float fact = float(shoot_max_nodes - shoot_node_index) / float(shoot_max_nodes);

    // set leaf scale based on position along the shoot
    //    float scale = fmin(1.f, 1 + 0.1*fact);
    //    phytomer->scaleLeafPrototypeScale(scale);

    //    phytomer->rotateLeaf( 0, 0, make_AxisRotation(-deg2rad(15)*fact, 0, 0));
    phytomer->rotateLeaf(0, 0, make_AxisRotation(-deg2rad(60) * fact, 0, 0));
}

SorghumPanicleParameters SorghumPanicleParameters::compact() {
    SorghumPanicleParameters parameters;
    parameters.length = 0.22f;
    parameters.compactness = 0.9f;
    parameters.primary_branch_count = 80;
    parameters.branch_length_base = 0.15f;
    parameters.branch_length_middle = 0.16f;
    parameters.branch_length_tip = 0.1f;
    parameters.branch_angle_base = 40.f;
    parameters.branch_angle_tip = 20.f;
    parameters.branch_upcurve = 10.f;
    parameters.branch_bare_fraction = 0.05f;
    parameters.splayed_branch_fraction = 0.02f;
    parameters.raceme_length_fraction = 0.25f;
    parameters.grain_count = 2000;
    return parameters;
}

SorghumPanicleParameters SorghumPanicleParameters::loose() {
    SorghumPanicleParameters parameters;
    parameters.length = 0.30f;
    parameters.compactness = 0.f;
    parameters.rachis_length_fraction = 0.8f;
    parameters.primary_branch_count = 50;
    parameters.branch_length_base = 0.35f;
    parameters.branch_length_middle = 0.25f;
    parameters.branch_length_tip = 0.12f;
    parameters.branch_length_variation = 0.25f;
    parameters.branch_angle_base = 50.f;
    parameters.branch_angle_tip = 25.f;
    parameters.branch_upcurve = 30.f;
    parameters.branch_bare_fraction = 0.35f;
    parameters.splayed_branch_fraction = 0.f;
    parameters.raceme_length_fraction = 0.15f;
    parameters.grains_per_raceme = 4;
    parameters.grain_count = 1100;
    return parameters;
}

// ---- Sorghum panicle build parameters ---- //
//
// Each build parameter of the library sorghum maps onto one field of SorghumPanicleParameters. The tables below are the single list of those keys: they are read by
// SorghumPanicleParameters::fromBuildParameters(), blended by the panicle_type preset, and declared to the plant library by SorghumPanicleParameters::buildParameterNames(). The rachis radius, branch
// color and tube subdivisions are left out, since they are matched to the library peduncle, which is not built from the panicle parameters.

struct SorghumPanicleFloatKey {
    const char *name;
    float SorghumPanicleParameters::*field;
    float min_value;
    float max_value;
    const char *description;
};

struct SorghumPanicleIntegerKey {
    const char *name;
    int SorghumPanicleParameters::*field;
    int min_value;
    int max_value;
    const char *description;
};

struct SorghumPanicleColorKey {
    const char *name;
    helios::RGBcolor SorghumPanicleParameters::*field;
    float helios::RGBcolor::*channel;
    const char *description;
};

static const std::vector<SorghumPanicleFloatKey> sorghum_panicle_float_keys = {
        {"panicle_length", &SorghumPanicleParameters::length, 0.05f, 0.6f, "sorghum panicle length in meters"},
        {"panicle_length_variation", &SorghumPanicleParameters::length_variation, 0.f, 0.3f, "coefficient of variation of sorghum panicle length"},
        {"panicle_rachis_length_fraction", &SorghumPanicleParameters::rachis_length_fraction, 0.3f, 1.f, "fraction of the sorghum panicle length spanned by the rachis"},
        {"panicle_rachis_curvature", &SorghumPanicleParameters::rachis_curvature, 0.f, 180.f, "total bend of the sorghum rachis in degrees"},
        {"panicle_compactness", &SorghumPanicleParameters::compactness, 0.f, 1.f, "compactness of the sorghum panicle"},
        {"panicle_branch_length_base", &SorghumPanicleParameters::branch_length_base, 0.02f, 0.6f, "length of the basal sorghum primary branches as a fraction of panicle length"},
        {"panicle_branch_length_middle", &SorghumPanicleParameters::branch_length_middle, 0.02f, 0.6f, "length of the middle sorghum primary branches as a fraction of panicle length"},
        {"panicle_branch_length_tip", &SorghumPanicleParameters::branch_length_tip, 0.02f, 0.6f, "length of the apical sorghum primary branches as a fraction of panicle length"},
        {"panicle_branch_length_variation", &SorghumPanicleParameters::branch_length_variation, 0.f, 0.5f, "coefficient of variation of sorghum primary branch length"},
        {"panicle_branch_angle_base", &SorghumPanicleParameters::branch_angle_base, 0.f, 90.f, "insertion angle of the basal sorghum primary branches in degrees"},
        {"panicle_branch_angle_tip", &SorghumPanicleParameters::branch_angle_tip, 0.f, 90.f, "insertion angle of the apical sorghum primary branches in degrees"},
        {"panicle_branch_upcurve", &SorghumPanicleParameters::branch_upcurve, 0.f, 90.f, "upward turn of the sorghum primary branches in degrees"},
        {"panicle_branch_bare_fraction", &SorghumPanicleParameters::branch_bare_fraction, 0.f, 0.9f, "fraction of each sorghum primary branch that bears no racemes"},
        {"panicle_splayed_branch_fraction", &SorghumPanicleParameters::splayed_branch_fraction, 0.f, 1.f, "fraction of sorghum primary branches that splay out from the head"},
        {"panicle_splayed_branch_angle", &SorghumPanicleParameters::splayed_branch_angle, 0.f, 120.f, "insertion angle of a splayed sorghum primary branch in degrees"},
        {"panicle_branch_radius", &SorghumPanicleParameters::branch_radius, 0.0002f, 0.003f, "radius of a sorghum primary branch in meters"},
        {"panicle_raceme_length_fraction", &SorghumPanicleParameters::raceme_length_fraction, 0.02f, 1.f, "sorghum raceme length as a fraction of its primary branch"},
        {"panicle_raceme_angle", &SorghumPanicleParameters::raceme_angle, 0.f, 90.f, "angle of a sorghum raceme from its primary branch in degrees"},
        {"grain_length", &SorghumPanicleParameters::grain_length, 0.001f, 0.01f, "sorghum grain length in meters"},
        {"grain_width", &SorghumPanicleParameters::grain_width, 0.001f, 0.01f, "sorghum grain width in meters"},
        {"grain_thickness", &SorghumPanicleParameters::grain_thickness, 0.001f, 0.01f, "sorghum grain thickness in meters"},
        {"grain_size_variation", &SorghumPanicleParameters::grain_size_variation, 0.f, 0.3f, "coefficient of variation of sorghum grain size"},
        {"grain_color_variation", &SorghumPanicleParameters::grain_color_variation, 0.f, 0.5f, "coefficient of variation of sorghum grain brightness"},
        {"glume_length_fraction", &SorghumPanicleParameters::glume_length_fraction, 0.f, 1.2f, "sorghum glume length as a fraction of grain length"},
        {"glume_spread", &SorghumPanicleParameters::glume_spread, 0.f, 90.f, "gape of the sorghum glumes in degrees"},
};

static const std::vector<SorghumPanicleIntegerKey> sorghum_panicle_integer_keys = {
        {"panicle_branch_count", &SorghumPanicleParameters::primary_branch_count, 5, 200, "number of sorghum primary branches"},
        {"panicle_grains_per_raceme", &SorghumPanicleParameters::grains_per_raceme, 1, 20, "mean number of grains on a sorghum raceme"},
        {"grain_count", &SorghumPanicleParameters::grain_count, 100, 10000, "mean number of grains on a sorghum panicle"},
};

static const std::vector<SorghumPanicleColorKey> sorghum_panicle_color_keys = {
        {"grain_color_r", &SorghumPanicleParameters::grain_color, &helios::RGBcolor::r, "red component of the sorghum grain color"},
        {"grain_color_g", &SorghumPanicleParameters::grain_color, &helios::RGBcolor::g, "green component of the sorghum grain color"},
        {"grain_color_b", &SorghumPanicleParameters::grain_color, &helios::RGBcolor::b, "blue component of the sorghum grain color"},
        {"glume_color_r", &SorghumPanicleParameters::glume_color, &helios::RGBcolor::r, "red component of the sorghum glume color"},
        {"glume_color_g", &SorghumPanicleParameters::glume_color, &helios::RGBcolor::g, "green component of the sorghum glume color"},
        {"glume_color_b", &SorghumPanicleParameters::glume_color, &helios::RGBcolor::b, "blue component of the sorghum glume color"},
};

// Blend of two panicle parameter sets, weighted by fraction toward the second. Only the fields in the key tables are blended; the presets differ in no other field.
static SorghumPanicleParameters blendSorghumPanicleParameters(const SorghumPanicleParameters &from, const SorghumPanicleParameters &to, float fraction) {
    SorghumPanicleParameters blended = from;
    for (const SorghumPanicleFloatKey &key: sorghum_panicle_float_keys) {
        blended.*key.field = (1.f - fraction) * from.*key.field + fraction * to.*key.field;
    }
    for (const SorghumPanicleIntegerKey &key: sorghum_panicle_integer_keys) {
        blended.*key.field = int(std::round((1.f - fraction) * float(from.*key.field) + fraction * float(to.*key.field)));
    }
    for (const SorghumPanicleColorKey &key: sorghum_panicle_color_keys) {
        blended.*key.field.*key.channel = (1.f - fraction) * from.*key.field.*key.channel + fraction * to.*key.field.*key.channel;
    }
    return blended;
}

SorghumPanicleParameters SorghumPanicleParameters::fromBuildParameters(const std::map<std::string, float> &build_parameters) {

    // The preset is applied first, so that individual parameters given alongside it override it
    SorghumPanicleParameters parameters;
    if (build_parameters.find("panicle_type") != build_parameters.end()) {
        const float panicle_type = getBuildParameterValue(build_parameters, "panicle_type", 0.5f, 0.f, 1.f, "sorghum panicle type from 0 (compact) through 0.5 (semi-compact) to 1 (loose)");
        if (panicle_type <= 0.5f) {
            parameters = blendSorghumPanicleParameters(compact(), SorghumPanicleParameters(), 2.f * panicle_type);
        } else {
            parameters = blendSorghumPanicleParameters(SorghumPanicleParameters(), loose(), 2.f * panicle_type - 1.f);
        }
    }

    for (const SorghumPanicleFloatKey &key: sorghum_panicle_float_keys) {
        parameters.*key.field = getBuildParameterValue(build_parameters, key.name, parameters.*key.field, key.min_value, key.max_value, key.description);
    }
    for (const SorghumPanicleIntegerKey &key: sorghum_panicle_integer_keys) {
        parameters.*key.field = getBuildParameterIntegerValue(build_parameters, key.name, parameters.*key.field, key.min_value, key.max_value, key.description);
    }
    for (const SorghumPanicleColorKey &key: sorghum_panicle_color_keys) {
        parameters.*key.field.*key.channel = getBuildParameterValue(build_parameters, key.name, parameters.*key.field.*key.channel, 0.f, 1.f, key.description);
    }
    parameters.grain_subdivisions = uint(getBuildParameterIntegerValue(build_parameters, "grain_subdivisions", int(parameters.grain_subdivisions), 3, 12, "number of facets around a sorghum grain"));

    return parameters;
}

std::vector<std::string> SorghumPanicleParameters::buildParameterNames() {
    std::vector<std::string> names = {"panicle_type", "grain_subdivisions"};
    for (const SorghumPanicleFloatKey &key: sorghum_panicle_float_keys) {
        names.emplace_back(key.name);
    }
    for (const SorghumPanicleIntegerKey &key: sorghum_panicle_integer_keys) {
        names.emplace_back(key.name);
    }
    for (const SorghumPanicleColorKey &key: sorghum_panicle_color_keys) {
        names.emplace_back(key.name);
    }
    return names;
}

// Quadratic through the base, middle and tip values of a profile along the rachis, evaluated at a normalized position from 0 at the base to 1 at the tip.
static float interpolateRachisProfile(float base_value, float middle_value, float tip_value, float position) {
    return base_value * (1.f - position) * (1.f - 2.f * position) + middle_value * 4.f * position * (1.f - position) + tip_value * position * (2.f * position - 1.f);
}

// Any unit vector perpendicular to the given unit vector.
static vec3 perpendicularTo(const vec3 &direction) {
    vec3 reference = std::fabs(direction.x) < 0.9f ? make_vec3(1, 0, 0) : make_vec3(0, 1, 0);
    return normalize(cross(direction, reference));
}

// Mesh of one grain: an obovoid with a pole at each end and two staggered rings of vertices, broader toward its apex than its base. The long axis is grain_axis; width and thickness lie along
// width_axis and the third, perpendicular axis.
static void addSorghumGrain(helios::Context *context_ptr, const vec3 &center, const vec3 &grain_axis, const vec3 &width_axis, float length, float width, float thickness, uint facets,
                            const RGBcolor &color, std::vector<uint> &UUIDs) {
    const vec3 thickness_axis = cross(grain_axis, width_axis);
    const float half_length = 0.5f * length;
    const float ring_height[2] = {-0.4f * half_length, 0.25f * half_length};

    const vec3 base_pole = center - grain_axis * half_length;
    const vec3 apex_pole = center + grain_axis * half_length;
    std::vector<vec3> rings[2];
    for (int ring = 0; ring < 2; ring++) {
        const float profile = std::sqrt(1.f - ring_height[ring] * ring_height[ring] / (half_length * half_length));
        rings[ring].resize(facets);
        for (uint facet = 0; facet < facets; facet++) {
            const float angle = 2.f * PI_F * (float(facet) + 0.5f * float(ring)) / float(facets);
            rings[ring].at(facet) = center + grain_axis * ring_height[ring] + width_axis * (0.5f * width * profile * cosf(angle)) + thickness_axis * (0.5f * thickness * profile * sinf(angle));
        }
    }

    for (uint facet = 0; facet < facets; facet++) {
        const uint next = (facet + 1) % facets;
        UUIDs.push_back(context_ptr->addTriangle(base_pole, rings[0].at(next), rings[0].at(facet), color));
        UUIDs.push_back(context_ptr->addTriangle(rings[0].at(facet), rings[0].at(next), rings[1].at(facet), color));
        UUIDs.push_back(context_ptr->addTriangle(rings[1].at(facet), rings[0].at(next), rings[1].at(next), color));
        UUIDs.push_back(context_ptr->addTriangle(apex_pole, rings[1].at(facet), rings[1].at(next), color));
    }
}

// The two glumes clasping the base of a grain, one over each flattened face. Each is a pointed scale of two triangles folded along its midline, which follows the grain surface outward.
static void addSorghumGlumes(helios::Context *context_ptr, const vec3 &center, const vec3 &grain_axis, const vec3 &width_axis, float length, float width, float thickness, float glume_length_fraction,
                             float glume_spread, const RGBcolor &color, std::vector<uint> &UUIDs) {
    const vec3 thickness_axis = cross(grain_axis, width_axis);
    const float half_length = 0.5f * length;
    const float glume_length = glume_length_fraction * length;

    // Half-width of the grain cross-section at a height above its center, as a fraction of the equatorial half-width.
    auto grain_profile = [&](float height) { return std::sqrt(std::max(0.f, 1.f - height * height / (half_length * half_length))); };

    const float tip_height = std::min(-half_length + glume_length, 1.05f * half_length);
    const float side_height = -half_length + 0.55f * glume_length;
    const vec3 base = center - grain_axis * (1.05f * half_length);
    for (float face: {-1.f, 1.f}) {
        const float tip_offset = 0.5f * thickness * (1.05f * grain_profile(tip_height) + 0.1f) + 0.5f * std::sin(deg2rad(glume_spread)) * glume_length;
        const vec3 tip = center + grain_axis * tip_height + thickness_axis * (face * tip_offset);
        const vec3 side_offset = thickness_axis * (face * 0.3f * thickness * grain_profile(side_height));
        const vec3 side_center = center + grain_axis * side_height + side_offset;
        const vec3 side_extent = width_axis * (0.48f * width * grain_profile(side_height) + 0.05f * width);
        UUIDs.push_back(context_ptr->addTriangle(base, side_center + side_extent, tip, color));
        UUIDs.push_back(context_ptr->addTriangle(base, tip, side_center - side_extent, color));
    }
}

uint buildSorghumPanicle(helios::Context *context_ptr, const SorghumPanicleParameters &parameters) {

    const SorghumPanicleParameters &p = parameters;
    if (p.length <= 0.f || p.grain_length <= 0.f || p.grain_width <= 0.f || p.grain_thickness <= 0.f || p.rachis_radius <= 0.f || p.branch_radius <= 0.f) {
        helios_runtime_error("ERROR (buildSorghumPanicle): Panicle length, rachis and branch radii, and grain dimensions must all be positive.");
    } else if (p.rachis_length_fraction <= 0.f || p.rachis_length_fraction > 1.f) {
        helios_runtime_error("ERROR (buildSorghumPanicle): rachis_length_fraction must be greater than 0 and at most 1, but " + std::to_string(p.rachis_length_fraction) + " was given.");
    } else if (p.primary_branch_count < 1 || p.grain_count < 1 || p.grains_per_raceme < 1) {
        helios_runtime_error("ERROR (buildSorghumPanicle): primary_branch_count, grain_count and grains_per_raceme must each be at least 1.");
    } else if (p.branch_bare_fraction < 0.f || p.branch_bare_fraction >= 1.f) {
        helios_runtime_error("ERROR (buildSorghumPanicle): branch_bare_fraction must be in the range [0,1), but " + std::to_string(p.branch_bare_fraction) + " was given.");
    } else if (p.compactness < 0.f || p.compactness > 1.f) {
        helios_runtime_error("ERROR (buildSorghumPanicle): compactness must be in the range [0,1], but " + std::to_string(p.compactness) + " was given.");
    } else if (p.glume_length_fraction < 0.f) {
        helios_runtime_error("ERROR (buildSorghumPanicle): glume_length_fraction cannot be negative.");
    } else if (p.grain_subdivisions < 3 || p.branch_radial_subdivisions < 3 || p.rachis_radial_subdivisions < 3) {
        helios_runtime_error("ERROR (buildSorghumPanicle): grain_subdivisions, branch_radial_subdivisions and rachis_radial_subdivisions must each be at least 3.");
    }

    const float size_factor = clamp(1.f + p.length_variation * context_ptr->randn(), 0.7f, 1.3f);
    const float panicle_length = p.length * size_factor;
    const int grain_count = std::max(1, int(std::round(float(p.grain_count) * size_factor)));

    // ---- Rachis ---- //

    // The rachis bends at constant curvature in a plane of random orientation through the x-axis. Each point along it carries a frame: the tangent and two normals spanning the plane in which the
    // primary branches radiate.
    const float rachis_length = p.rachis_length_fraction * panicle_length;
    const float total_bend = deg2rad(p.rachis_curvature);
    const float bend_azimuth = context_ptr->randu(0.f, 2.f * PI_F);
    const vec3 axis = make_vec3(1, 0, 0);
    const vec3 bend_direction = make_vec3(0, cosf(bend_azimuth), sinf(bend_azimuth));
    const vec3 binormal = cross(axis, bend_direction);

    auto rachis_position = [&](float arc_length) {
        if (std::fabs(total_bend) < 1e-4f) {
            return axis * arc_length;
        }
        const float bend_radius = rachis_length / total_bend;
        const float angle = arc_length / bend_radius;
        return axis * (bend_radius * sinf(angle)) + bend_direction * (bend_radius * (1.f - cosf(angle)));
    };
    auto rachis_tangent = [&](float arc_length) {
        const float angle = total_bend * arc_length / rachis_length;
        return axis * cosf(angle) + bend_direction * sinf(angle);
    };
    auto rachis_normal = [&](float arc_length) {
        const float angle = total_bend * arc_length / rachis_length;
        return bend_direction * cosf(angle) - axis * sinf(angle);
    };

    // Component of a point's offset from the panicle axis that is perpendicular to the axis. The axis is the rachis, continued straight past its tip. arc_length_hint is the arc length of a nearby
    // rachis point, from which the nearest one is found by a single projection.
    auto offset_from_axis = [&](const vec3 &point, float arc_length_hint) {
        float arc_length = arc_length_hint + (point - rachis_position(arc_length_hint)) * rachis_tangent(arc_length_hint);
        arc_length = std::max(0.f, arc_length);
        vec3 axis_point, tangent;
        if (arc_length <= rachis_length) {
            axis_point = rachis_position(arc_length);
            tangent = rachis_tangent(arc_length);
        } else {
            tangent = rachis_tangent(rachis_length);
            axis_point = rachis_position(rachis_length) + tangent * (arc_length - rachis_length);
        }
        const vec3 offset = point - axis_point;
        return offset - tangent * (offset * tangent);
    };

    // ---- Primary branches ---- //

    // Each primary branch leaves the rachis at its insertion angle and turns progressively back toward the rachis direction, so the branch tips point up the head as they do in the field. Branches are
    // spaced up the rachis with a divergence near the golden angle, which spreads them evenly around it without visible ranks.
    constexpr int branch_segments = 5;
    struct PrimaryBranch {
        std::vector<vec3> nodes;
        float length;
        float arc_length; // insertion point on the rachis
        vec3 outward; // radial direction from the rachis at the insertion point
    };
    std::vector<PrimaryBranch> branches(p.primary_branch_count);
    const float azimuth_offset = context_ptr->randu(0.f, 2.f * PI_F);
    for (int b = 0; b < p.primary_branch_count; b++) {
        PrimaryBranch &branch = branches.at(b);
        const float position = clamp((float(b) + 0.5f + context_ptr->randu(-0.3f, 0.3f)) / float(p.primary_branch_count), 0.f, 1.f);
        branch.arc_length = position * rachis_length;

        const float length_fraction = std::max(0.02f, interpolateRachisProfile(p.branch_length_base, p.branch_length_middle, p.branch_length_tip, position));
        branch.length = length_fraction * panicle_length * clamp(1.f + p.branch_length_variation * context_ptr->randn(), 0.4f, 1.6f);

        float insertion_angle = p.branch_angle_base + (p.branch_angle_tip - p.branch_angle_base) * position + context_ptr->randu(-5.f, 5.f);
        float upcurve = p.branch_upcurve;
        if (context_ptr->randu() < p.splayed_branch_fraction) {
            insertion_angle = p.splayed_branch_angle;
            upcurve *= 0.5f;
        }
        insertion_angle = deg2rad(clamp(insertion_angle, 0.f, 90.f));
        upcurve = deg2rad(upcurve);

        const float azimuth = azimuth_offset + deg2rad(137.5f) * float(b) + context_ptr->randu(-0.3f, 0.3f);
        const vec3 tangent = rachis_tangent(branch.arc_length);
        branch.outward = rachis_normal(branch.arc_length) * cosf(azimuth) + binormal * sinf(azimuth);

        branch.nodes.resize(branch_segments + 1);
        branch.nodes.at(0) = rachis_position(branch.arc_length);
        const float segment_length = branch.length / float(branch_segments);
        for (int segment = 0; segment < branch_segments; segment++) {
            const float angle = std::max(0.f, insertion_angle - upcurve * (float(segment) + 0.5f) / float(branch_segments));
            const vec3 direction = tangent * cosf(angle) + branch.outward * sinf(angle);
            branch.nodes.at(segment + 1) = branch.nodes.at(segment) + direction * segment_length;
        }
    }

    // Point and direction at a fraction of the way along a primary branch
    auto branch_point = [&](const PrimaryBranch &branch, float fraction, vec3 &point, vec3 &direction) {
        const float scaled = clamp(fraction, 0.f, 1.f) * float(branch_segments);
        const int segment = std::min(int(scaled), branch_segments - 1);
        const float local = scaled - float(segment);
        direction = normalize(branch.nodes.at(segment + 1) - branch.nodes.at(segment));
        point = branch.nodes.at(segment) + (branch.nodes.at(segment + 1) - branch.nodes.at(segment)) * local;
    };

    // ---- Racemes and grain placement ---- //

    // Racemes are shared among the primary branches in proportion to the fertile length of each (largest remainder), and grains among the racemes as evenly as the total allows.
    const int raceme_count = std::max(1, int(std::round(float(grain_count) / float(p.grains_per_raceme))));
    std::vector<float> fertile_length(p.primary_branch_count);
    float total_fertile_length = 0.f;
    for (int b = 0; b < p.primary_branch_count; b++) {
        fertile_length.at(b) = (1.f - p.branch_bare_fraction) * branches.at(b).length;
        total_fertile_length += fertile_length.at(b);
    }
    std::vector<int> branch_raceme_count(p.primary_branch_count);
    std::vector<std::pair<float, int>> remainders(p.primary_branch_count);
    int assigned = 0;
    for (int b = 0; b < p.primary_branch_count; b++) {
        const float share = fertile_length.at(b) / total_fertile_length * float(raceme_count);
        branch_raceme_count.at(b) = int(std::floor(share));
        remainders.at(b) = {share - std::floor(share), b};
        assigned += branch_raceme_count.at(b);
    }
    std::sort(remainders.rbegin(), remainders.rend());
    for (int i = 0; i < raceme_count - assigned; i++) {
        branch_raceme_count.at(remainders.at(i).second)++;
    }

    struct Raceme {
        int branch;
        float fraction; // position along the primary branch
        int first_grain;
        int grain_count;
    };
    struct Grain {
        vec3 anchor;
        vec3 center;
        vec3 axis;
        vec3 width_axis;
        float scale;
        float arc_length; // insertion of the bearing primary branch on the rachis
    };
    std::vector<Raceme> racemes;
    racemes.reserve(raceme_count);
    std::vector<Grain> grains;
    grains.reserve(grain_count);

    for (int b = 0; b < p.primary_branch_count; b++) {
        const PrimaryBranch &branch = branches.at(b);
        for (int r = 0; r < branch_raceme_count.at(b); r++) {
            // Grains are dealt out so that the first (grain_count mod raceme_count) racemes carry one extra.
            const int raceme_index = int(racemes.size());
            const int raceme_grains = grain_count / raceme_count + (raceme_index < grain_count % raceme_count ? 1 : 0);

            Raceme raceme;
            raceme.branch = b;
            raceme.fraction = p.branch_bare_fraction + (1.f - p.branch_bare_fraction) * (float(r) + context_ptr->randu(0.2f, 0.8f)) / float(branch_raceme_count.at(b));
            raceme.first_grain = int(grains.size());
            raceme.grain_count = raceme_grains;
            racemes.push_back(raceme);

            vec3 base, branch_direction;
            branch_point(branch, raceme.fraction, base, branch_direction);

            // Racemes leave the branch on all sides, but those facing back into the head are turned outward.
            const vec3 side_reference = perpendicularTo(branch_direction);
            const float side_angle = context_ptr->randu(0.f, 2.f * PI_F);
            vec3 side = side_reference * cosf(side_angle) + cross(branch_direction, side_reference) * sinf(side_angle);
            if (side * branch.outward < 0.f) {
                side = side - branch.outward * (1.5f * (side * branch.outward));
                side = normalize(side - branch_direction * (side * branch_direction));
            }
            const float raceme_angle = deg2rad(p.raceme_angle);
            const vec3 raceme_direction = branch_direction * cosf(raceme_angle) + side * sinf(raceme_angle);
            const float raceme_length = std::max(p.raceme_length_fraction * branch.length, 0.6f * float(raceme_grains) * p.grain_length);

            // Spikelets alternate from one side of the raceme to the other, each grain tilted outward from it.
            const vec3 alternation_axis = normalize(cross(raceme_direction, side));
            for (int g = 0; g < raceme_grains; g++) {
                const float along = (float(g) + 0.75f) / float(raceme_grains);
                const float alternation = (g % 2 == 0) ? 1.f : -1.f;
                const vec3 lateral = normalize(side + alternation_axis * (0.8f * alternation));

                Grain grain;
                grain.scale = clamp(1.f + p.grain_size_variation * context_ptr->randn(), 0.7f, 1.3f);
                grain.anchor = base + raceme_direction * (raceme_length * along) + lateral * (0.5f * p.grain_width * grain.scale);
                grain.center = grain.anchor;
                grain.axis = normalize(raceme_direction * 0.8f + lateral * 0.6f);
                grain.width_axis = normalize(cross(grain.axis, lateral));
                grain.arc_length = branch.arc_length;
                grains.push_back(grain);
            }
        }
    }

    // ---- Pack the grains ---- //

    // Grains placed on neighbouring racemes overlap freely. Overlapping grains are pushed apart while each is held to its place on its raceme by a weak spring, which leaves them touching.
    //
    // A real head is not solid grain: a compact head 7 cm across would need several times the grain it carries to fill, and instead holds its grain in a layer a few grains deep over the branches, with
    // the rachis and branches inside. Compactness draws the grains into that layer, the depth of which is set by the branch envelope at each height, from both sides: grains standing out beyond the
    // envelope are pulled in, and grains on the inner, hidden parts of the branches are moved out to where they can be seen. Within the layer the overlap resolution spreads them into a continuous
    // surface of grain. The spring holding grains to their racemes weakens in proportion so it does not undo the compaction. The final iterations only resolve overlaps. A uniform grid with cells about
    // one grain wide limits the overlap search to neighbouring cells.
    constexpr int packing_iterations = 24;
    constexpr int settling_iterations = 10;
    constexpr float compaction_rate = 0.25f;
    const float anchor_spring = 0.1f * (1.f - p.compactness);
    const float contact_radius = 0.25f * (p.grain_width + p.grain_thickness);
    const float layer_depth = 3.f * contact_radius;
    const float cell_size = 2.f * 1.3f * contact_radius;
    auto cell_index = [&](const vec3 &position) { return make_int3(int(std::floor(position.x / cell_size)), int(std::floor(position.y / cell_size)), int(std::floor(position.z / cell_size))); };
    auto cell_key = [](int ix, int iy, int iz) { return (int64_t(ix + (1 << 20)) << 42) | (int64_t(iy + (1 << 20)) << 21) | int64_t(iz + (1 << 20)); };

    // Envelope radius of the head along its length: a high percentile of the grain radial distances in each height band, smoothed between bands so the outline has no steps.
    constexpr int envelope_bands = 24;
    std::vector<int> grain_band(grains.size());
    std::vector<std::vector<float>> band_radii(envelope_bands);
    for (int i = 0; i < int(grains.size()); i++) {
        const Grain &grain = grains.at(i);
        const float axial = grain.arc_length + (grain.anchor - rachis_position(grain.arc_length)) * rachis_tangent(grain.arc_length);
        grain_band.at(i) = clamp(int(float(envelope_bands) * axial / panicle_length), 0, envelope_bands - 1);
        band_radii.at(grain_band.at(i)).push_back(offset_from_axis(grain.anchor, grain.arc_length).magnitude());
    }
    std::vector<float> band_envelope(envelope_bands, 0.f);
    for (int band = 0; band < envelope_bands; band++) {
        std::vector<float> &radii = band_radii.at(band);
        if (!radii.empty()) {
            std::sort(radii.begin(), radii.end());
            band_envelope.at(band) = radii.at(std::min(radii.size() - 1, size_t(0.85f * float(radii.size()))));
        }
    }
    std::vector<float> envelope = band_envelope;
    for (int band = 0; band < envelope_bands; band++) {
        float sum = 0.f;
        int count = 0;
        for (int neighbour = std::max(0, band - 1); neighbour <= std::min(envelope_bands - 1, band + 1); neighbour++) {
            if (!band_radii.at(neighbour).empty()) {
                sum += band_envelope.at(neighbour);
                count++;
            }
        }
        envelope.at(band) = count > 0 ? sum / float(count) : 0.f;
    }

    std::unordered_map<int64_t, std::vector<int>> grid;
    std::vector<vec3> displacement(grains.size());
    for (int iteration = 0; iteration < packing_iterations + settling_iterations; iteration++) {
        if (iteration < packing_iterations) {
            for (int i = 0; i < int(grains.size()); i++) {
                Grain &grain = grains.at(i);
                const vec3 radial = offset_from_axis(grain.center, grain.arc_length);
                const float radial_distance = radial.magnitude();
                const float outer_radius = envelope.at(grain_band.at(i));
                const float inner_radius = std::max(p.rachis_radius + contact_radius * grain.scale, outer_radius - layer_depth);
                float target_distance = radial_distance;
                if (radial_distance > outer_radius) {
                    target_distance = outer_radius;
                } else if (radial_distance < inner_radius) {
                    target_distance = inner_radius;
                }
                if (radial_distance > 1e-6f) {
                    grain.center = grain.center + radial * (p.compactness * compaction_rate * (target_distance / radial_distance - 1.f));
                }
                grain.center = grain.center + (grain.anchor - grain.center) * anchor_spring;
            }
        }
        for (int pass = 0; pass < 3; pass++) {
            grid.clear();
            for (int i = 0; i < int(grains.size()); i++) {
                const int3 cell = cell_index(grains.at(i).center);
                grid[cell_key(cell.x, cell.y, cell.z)].push_back(i);
            }
            std::fill(displacement.begin(), displacement.end(), make_vec3(0, 0, 0));
            for (int i = 0; i < int(grains.size()); i++) {
                const int3 home = cell_index(grains.at(i).center);
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dz = -1; dz <= 1; dz++) {
                            auto cell = grid.find(cell_key(home.x + dx, home.y + dy, home.z + dz));
                            if (cell == grid.end()) {
                                continue;
                            }
                            for (int j: cell->second) {
                                if (j <= i) {
                                    continue;
                                }
                                const vec3 separation = grains.at(i).center - grains.at(j).center;
                                const float distance = separation.magnitude();
                                const float overlap = contact_radius * (grains.at(i).scale + grains.at(j).scale) - distance;
                                if (overlap > 0.f && distance > 1e-9f) {
                                    const vec3 push = separation * (0.5f * overlap / distance);
                                    displacement.at(i) = displacement.at(i) + push;
                                    displacement.at(j) = displacement.at(j) - push;
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < int(grains.size()); i++) {
                grains.at(i).center = grains.at(i).center + displacement.at(i);
            }
        }
    }

    // ---- Branch geometry ---- //

    // Compaction moved the grains but not the branches that bear them, which would leave bare branch tips standing out of the head. Each primary branch is drawn in toward the axis by the mean factor
    // by which its own grains were, and its racemes are then run from the branch to the grains they carry.
    std::vector<uint> UUIDs;

    constexpr int rachis_segments = 10;
    std::vector<vec3> rachis_nodes(rachis_segments + 1);
    std::vector<float> rachis_radii(rachis_segments + 1);
    for (int node = 0; node <= rachis_segments; node++) {
        const float fraction = float(node) / float(rachis_segments);
        rachis_nodes.at(node) = rachis_position(fraction * rachis_length);
        rachis_radii.at(node) = p.rachis_radius * (1.f - 0.67f * fraction);
    }
    std::vector<uint> UUIDs_rachis = context_ptr->addTube(p.rachis_radial_subdivisions, rachis_nodes, rachis_radii, std::vector<RGBcolor>(rachis_nodes.size(), p.branch_color));
    UUIDs.insert(UUIDs.end(), UUIDs_rachis.begin(), UUIDs_rachis.end());

    std::vector<float> anchor_radial_sum(p.primary_branch_count, 0.f);
    std::vector<float> final_radial_sum(p.primary_branch_count, 0.f);
    for (const Raceme &raceme: racemes) {
        for (int g = raceme.first_grain; g < raceme.first_grain + raceme.grain_count; g++) {
            anchor_radial_sum.at(raceme.branch) += offset_from_axis(grains.at(g).anchor, grains.at(g).arc_length).magnitude();
            final_radial_sum.at(raceme.branch) += offset_from_axis(grains.at(g).center, grains.at(g).arc_length).magnitude();
        }
    }
    for (int b = 0; b < p.primary_branch_count; b++) {
        PrimaryBranch &branch = branches.at(b);
        if (anchor_radial_sum.at(b) > 0.f) {
            const float radial_ratio = std::min(1.f, final_radial_sum.at(b) / anchor_radial_sum.at(b));
            for (int node = 1; node <= branch_segments; node++) {
                const vec3 radial = offset_from_axis(branch.nodes.at(node), branch.arc_length);
                branch.nodes.at(node) = branch.nodes.at(node) - radial * (1.f - radial_ratio);
            }
        }
        std::vector<float> radii(branch_segments + 1);
        for (int node = 0; node <= branch_segments; node++) {
            radii.at(node) = p.branch_radius * (1.f - 0.5f * float(node) / float(branch_segments));
        }
        std::vector<uint> UUIDs_branch = context_ptr->addTube(p.branch_radial_subdivisions, branch.nodes, radii, std::vector<RGBcolor>(branch.nodes.size(), p.branch_color));
        UUIDs.insert(UUIDs.end(), UUIDs_branch.begin(), UUIDs_branch.end());
    }

    for (const Raceme &raceme: racemes) {
        if (raceme.grain_count == 0) {
            continue;
        }
        vec3 base, branch_direction;
        branch_point(branches.at(raceme.branch), raceme.fraction, base, branch_direction);
        const vec3 tip = grains.at(raceme.first_grain + raceme.grain_count - 1).center;
        if ((tip - base).magnitude() < 1e-5f) {
            continue;
        }
        std::vector<vec3> raceme_nodes = {base, tip};
        std::vector<float> raceme_radii = {0.6f * p.branch_radius, 0.4f * p.branch_radius};
        std::vector<uint> UUIDs_raceme = context_ptr->addTube(p.branch_radial_subdivisions, raceme_nodes, raceme_radii, std::vector<RGBcolor>(2, p.branch_color));
        UUIDs.insert(UUIDs.end(), UUIDs_raceme.begin(), UUIDs_raceme.end());
    }

    // ---- Grain geometry ---- //

    for (const Grain &grain: grains) {
        const float brightness = clamp(1.f + p.grain_color_variation * context_ptr->randn(), 0.6f, 1.4f);
        RGBcolor grain_color = p.grain_color;
        grain_color.scale(brightness);
        const float length = p.grain_length * grain.scale;
        const float width = p.grain_width * grain.scale;
        const float thickness = p.grain_thickness * grain.scale;
        addSorghumGrain(context_ptr, grain.center, grain.axis, grain.width_axis, length, width, thickness, p.grain_subdivisions, grain_color, UUIDs);
        if (p.glume_length_fraction > 0.f) {
            RGBcolor glume_color = p.glume_color;
            glume_color.scale(clamp(1.f + 0.5f * p.grain_color_variation * context_ptr->randn(), 0.6f, 1.4f));
            addSorghumGlumes(context_ptr, grain.center, grain.axis, grain.width_axis, length, width, thickness, p.glume_length_fraction, p.glume_spread, glume_color, UUIDs);
        }
    }

    context_ptr->setPrimitiveData(UUIDs, "object_label", "fruit");

    return context_ptr->addPolymeshObject(UUIDs);
}

uint SorghumPaniclePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    return buildSorghumPanicle(context_ptr, SorghumPanicleParameters::fromBuildParameters(build_parameters));
}

int sorghumTillerCount(const std::map<std::string, float> &build_parameters) {
    return getBuildParameterIntegerValue(build_parameters, "tiller_count", 0, 0, 6, "number of tillers of a sorghum plant");
}

void SorghumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    Shoot *shoot = phytomer->parent_shoot_ptr;
    const PlantArchitecture *plantarchitecture = shoot->plantarchitecture_ptr;
    if (phytomer->rank == 0) {
        // Tillers grow from the axils of the lowest main-stem leaves, and are added in the order of the hierarchy of tiller emergence, fertility and grain number T3 > T4 > T2 > T1 > T5 > T6 (Lafarge,
        // Broad & Hammer 2002, Annals of Botany 90:87). Fertile tillers of field hybrids are likewise mostly T2-T4 (Kim et al. 2010, Annals of Botany 106:57). The bud in every other axil is killed.
        constexpr std::array<uint, 6> tiller_leaf_numbers = {3, 4, 2, 1, 5, 6};
        const int tiller_count = sorghumTillerCount(plantarchitecture->getPlantBuildParameters(phytomer->plantID));
        const auto tillering_end = tiller_leaf_numbers.begin() + tiller_count;
        if (std::find(tiller_leaf_numbers.begin(), tillering_end, shoot_node_index + 1) == tillering_end) {
            phytomer->setVegetativeBudState(BUD_DEAD);
        }
    } else if (shoot_node_index == 0) {
        // The final leaf number of a tiller falls by one for each main-stem leaf below its own, from about 15 on tiller 1 to 10 on tiller 6 of a main stem of about 16 leaves, which brings the final
        // leaves of every tiller out together (Lafarge et al. 2002). Set on the tiller's own shoot parameters, so that its panicle forms at that node.
        const std::shared_ptr<Shoot> &main_stem = plantarchitecture->getPlantShoot(phytomer->plantID, shoot->parent_shoot_ID);
        const int tiller_max_nodes = std::max(1, main_stem->shoot_parameters.max_nodes.val() - int(parent_shoot_node_index + 1));
        shoot->shoot_parameters.max_nodes = tiller_max_nodes;
        phytomer->shoot_index.z = tiller_max_nodes;
        shoot_max_nodes = uint(tiller_max_nodes);
    }

    // Leaf size and internode length are scaled separately along the shoot. A single ramp cannot serve both: the lowest leaves of a sorghum plant are already close to full length while the internodes
    // carrying them are the shortest on the plant, so one factor either stunts the bottom leaves or spreads the base of the stem out too far.
    //
    // Blade length rises from the base, peaks several ranks below the top, and then falls away over the last few leaves. The rank of the largest leaf follows xs = 4.64 + 0.46*TLN for a total leaf number
    // below 20.5 (Demarco, van Oosterom, Kholova & Hammer 2026, Annals of Botany 137(4):920), which places it three to four ranks below the flag leaf on a plant of this size and agrees with the profiles
    // of Lafarge & Hammer (2002, Field Crops Research 77:137). Without the decline every leaf from the fourth node upward came out at the same maximum size, leaving the flag leaf fractionally the largest
    // blade on the plant -- the opposite of the real canopy, where it is conspicuously the smallest.
    float leaf_scale = fmin(1.f, 0.82f + 0.18f * float(shoot_node_index) / 3.f);
    if (shoot_max_nodes > 0) {
        const float largest_leaf_rank = 4.64f + 0.46f * float(shoot_max_nodes);
        const float ranks_above_largest = float(shoot_node_index + 1) - largest_leaf_rank;
        if (ranks_above_largest > 0.f) {
            // Linear in blade length, so leaf area -- which goes as roughly its square -- falls off faster, leaving the flag leaf near 40% of the largest blade's area.
            const float ranks_to_flag = fmax(1.f, float(shoot_max_nodes) - largest_leaf_rank);
            leaf_scale *= fmax(0.80f, 1.f - 0.20f * ranks_above_largest / ranks_to_flag);
        }
    }

    // Applied relative to the size the blade is currently built at, rather than as an absolute target. leaf_size_max holds that size, and current_leaf_scale_factor -- how far the leaf has expanded toward
    // it -- is left alone, because both setters divide it by whatever factor they apply and a reduced fraction would apply the shrink a second time.
    if (!phytomer->leaf_size_max.empty() && !phytomer->leaf_size_max.front().empty()) {
        const float leaf_size_target = leaf_scale * phytomer->phytomer_parameters.leaf.prototype_scale.val();
        const float leaf_size_current = phytomer->leaf_size_max.front().front();
        if (leaf_size_current > 0.f && std::fabs(leaf_size_target - leaf_size_current) > 1e-5f) {
            phytomer->scaleLeafPrototypeScale(leaf_size_target / leaf_size_current);
        }
    }

    // The lowest internodes do not elongate. They form a crown at the soil surface from which the lowest leaves and the tillers emerge. In sweet sorghum, internodes 3-6 lack an intercalary meristem and
    // stay shorter than 1 cm, internodes 7-8 may or may not elongate, and those from 9 upward elongate (Nakamura & Goto 1996, Japanese Journal of Crop Science 65:324). The shorter-season grain types
    // modelled here elongate few internodes before floral initiation, so the crown is taken as the first six, which also bear all six tiller positions, with the next two transitional.
    constexpr uint crown_internodes = 6;
    if (shoot_node_index < crown_internodes) {
        phytomer->setInternodeMaxLength(0.008f);
    } else {
        // Above the crown, internodes lengthen with height rather than shortening, so the upper leaves are not bunched together at the apex.
        const float internode_scale = fmin(1.f, 0.60f + 0.20f * float(shoot_node_index - crown_internodes));
        phytomer->scaleInternodeMaxLength(internode_scale);
    }

    // The terminal phytomer carries the panicle; its leaf is the flag leaf, which is set 30 degrees off the culm so that it clears the panicle.
    const bool is_flag_leaf = (shoot_max_nodes > 0 && shoot_node_index + 1 == shoot_max_nodes);
    if (is_flag_leaf) {
        setGrassFlagLeafAngle(phytomer, 30.f);

        // Sorghum bears a conspicuously long internode between the last normal leaf and the flag leaf, so the flag leaf stands clear of the canopy below it rather than sitting among the upper leaves. The
        // internode ramp above saturates at 1.0 by node 5, which left this internode the same length as the eleven below it and removed the segment entirely.
        //
        // Its length is tied to the peduncle rather than set outright, so that the last-normal-leaf-to-flag-leaf distance matches the flag-leaf-to-panicle-base distance. That keeps the relationship intact
        // if the peduncle length is changed, and anchors the elongation to a quantity that is itself measured: the sorghum peduncle is 35-39 cm in three-dwarf grain types (Wang et al. 2024, Genes 15(1):83),
        // which the 0.30 m used here sits just below. No published length for this internode itself was found, so the tie to the peduncle is the calibration.
        phytomer->setInternodeMaxLength(phytomer->phytomer_parameters.peduncle.length.val());
    }
}

uint SoybeanFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/SoybeanPod.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint SoybeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs;
    if (flower_is_open) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/SoybeanFlower_open_white.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanFlower_closed_white.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void SoybeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index > 5 || phytomer->rank > 1) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    } else {
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.1 + 0.9 * plant_age / 15.f);
    phytomer->scaleInternodeMaxLength(inode_scale);
}

uint StrawberryFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/StrawberryFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint StrawberryFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/StrawberryFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint TomatoFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/TomatoFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint TomatoFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/TomatoFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void TomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index < 8 && phytomer->rank == 0) {
        phytomer->setFloralBudState(BUD_DEAD);
    }
    if (phytomer->rank > 1) {
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->setVegetativeBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.5 + 0.5 * plant_age / 10.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 10.f);
    phytomer->scaleInternodeMaxLength(inode_scale);
}

void CherryTomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index < 8 || phytomer->rank > 1) {
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->setVegetativeBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 10.f);
    phytomer->scaleInternodeMaxLength(inode_scale);
}

void CherryTomatoPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    float pruning_height = 1.f;
    float pruning_day = 101.f;

    float plant_age = phytomer->parent_shoot_ptr->plantarchitecture_ptr->getPlantAge(phytomer->plantID);

    if (phytomer->hasLeaf() && plant_age >= pruning_day) {
        float height = phytomer->getInternodeNodePositions().at(0).z;
        if (height < pruning_height) {
            phytomer->removeLeaf();
        }
    }
}

uint WalnutFruitPrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/WalnutHull.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint WalnutFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AlmondFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void WalnutPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // blind nodes
    if (shoot_node_index < 4) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }
}

void WalnutPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
}

uint WheatSpikePrototype(helios::Context *context_ptr, uint subdivisions, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/WheatSpike.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void WheatPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // Leaf size and internode length are scaled separately along the shoot, because they run in opposite directions: the lowest internodes of a wheat culm are the shortest on the plant while the leaves
    // they carry are already well grown, so a single ramp either stunts the bottom leaves or spreads the base of the stem out too far.
    //
    // Leaf length peaks in the middle of the canopy and falls away again toward the flag leaf, which is shorter than the leaves below it.
    float leaf_scale = std::fmin(1.f, 0.62f + 0.38f * float(shoot_node_index) / 2.f);
    if (phytomer->shoot_index.z > 1) {
        // The flag leaf is shorter than the leaves below it, but only modestly so - roughly three quarters of the longest leaf on the plant rather than a stub. Only the topmost leaf is trimmed, since the
        // leaf immediately below it is already close to full length in a real plant.
        const float ranks_from_top = float(phytomer->shoot_index.z - 1 - int(shoot_node_index));
        leaf_scale = std::fmin(leaf_scale, 0.88f + 0.12f * std::fmin(1.f, ranks_from_top));
    }
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // Internodes are very short at the base and lengthen up the culm, which is what puts the lower leaves close together above the crown and carries the ear clear of them.
    const float internode_scale = std::fmin(1.f, 0.25f + 0.75f * float(shoot_node_index) / 5.f);
    phytomer->scaleInternodeMaxLength(internode_scale);

    // The terminal phytomer carries the ear, and its leaf is the flag leaf. It is set 33 degrees off the culm, so that it swings clear of the ear rather than lying along the culm through the spike, as on a
    // real plant, where the flag leaf is the most horizontal leaf of the canopy rather than the most upright.
    if (shoot_max_nodes > 0 && shoot_node_index + 1 == shoot_max_nodes) {
        setGrassFlagLeafAngle(phytomer, 33.f);
    }
}
