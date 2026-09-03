/** \file "Assets.cpp" Function definitions for plant organ prototypes plant architecture plug-in.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "PlantArchitecture.h"
#include "global.h"

using namespace helios;

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
        // loadOBJ() returns nothing, and creates no object, if every face of the asset was rejected as degenerate. Reading the parent of the first primitive would then be a read past the end.
        std::vector<uint> UUIDs_petiolule = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/PetiolulePrototype.obj").string().c_str(), make_vec3(0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
        context_ptr->translatePrimitive(UUIDs, make_vec3(0.07, 0, 0.005));
        for (vec3 &mesh_vertex: mesh_vertices) {
            mesh_vertex = mesh_vertex + make_vec3(0.07, 0, 0.005);
        }

        // loadOBJ() worked out the petiolule's own vertex-facet connectivity and grouped it into a polymesh object of its own. Carry that face set into the leaf's before releasing the primitives, so the
        // combined object describes both surfaces; simply detaching them would delete the object and throw the connectivity away.
        const uint petiolule_ObjID = UUIDs_petiolule.empty() ? 0 : context_ptr->getPrimitiveParentObjectID(UUIDs_petiolule.front());
        if (petiolule_ObjID != 0 && context_ptr->doesObjectExist(petiolule_ObjID)) {
            const std::vector<vec3> petiolule_vertices = context_ptr->getPolymeshObjectVertices(petiolule_ObjID);
            const std::vector<int3> petiolule_faces = context_ptr->getPolymeshObjectFaces(petiolule_ObjID);
            const int petiolule_vertex_offset = scast<int>(mesh_vertices.size());

            for (const vec3 &petiolule_vertex: petiolule_vertices) {
                mesh_vertices.push_back(petiolule_vertex);
                // The texture coordinate array has to stay parallel to the vertices. The petiolule is drawn in a flat color rather than from a texture, so it has no coordinates to carry over and none that
                // could ever be sampled; the entry is a placeholder keeping the arrays aligned, not a coordinate anyone should read.
                mesh_vertex_uv.push_back(make_vec2(0, 0));
            }
            for (size_t petiolule_face = 0; petiolule_face < petiolule_faces.size(); petiolule_face++) {
                const int3 &face = petiolule_faces.at(petiolule_face);
                mesh_faces.push_back(make_int3(face.x + petiolule_vertex_offset, face.y + petiolule_vertex_offset, face.z + petiolule_vertex_offset));
                mesh_face_UUIDs.push_back(context_ptr->getPolymeshObjectPrimitiveUUIDForFace(petiolule_ObjID, petiolule_face));
            }
        }

        // The petiolule's primitives already have a parent, and addPolymeshObject() below skips anything that does. Without this the leaf copied to every phytomer would carry no petiolule, and the
        // prototype's own would be left behind at the origin at prototype scale as geometry no plant owns.
        context_ptr->setPrimitiveParentObjectID(UUIDs_petiolule, 0);

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

    // The face set covers every primitive of the object: the blade from the lattice above, and the petiolule from the connectivity loadOBJ() retained. It is attached only when that is actually true, because
    // getFaceIndexForPrimitive() raises for a member the table omits, and writeOBJ() calls it for every primitive of any object reporting a non-zero face count.
    if (!mesh_faces.empty() && mesh_face_UUIDs.size() == context_ptr->getObjectPrimitiveUUIDs(objID).size()) {
        context_ptr->setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, mesh_face_UUIDs, {}, mesh_vertex_uv, helios::NORMAL_SOURCE_NONE);
    }

    return objID;
}

uint GeneralSphericalFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    return context_ptr->addSphereObject(5, make_vec3(0.5f, 0, 0), 0.5f, RGB::red);
}

uint AlmondFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AlmondHull.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint AlmondFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AlmondFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void AlmondPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (phytomer->internode_length_max < 0.01) { // spurs
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->scaleLeafPrototypeScale(0.8);
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->parent_shoot_ptr->shoot_parameters.max_nodes_per_season = 7;
    }

    // blind nodes
    //    if( shoot_node_index<3 ){
    //        phytomer->setVegetativeBudState( BUD_DEAD );
    //        phytomer->setFloralBudState( BUD_DEAD );
    //    }
}

void AlmondPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
}

uint AppleFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AppleFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint AppleFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/AlmondFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void ApplePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (phytomer->internode_length_max < 0.01) { // spurs
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->scaleLeafPrototypeScale(0.8);
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->parent_shoot_ptr->shoot_parameters.max_nodes_per_season = 6;
    }
}

void ApplePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
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
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanLeaf_left.obj").string().c_str(), true);
    } else {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanLeaf_right.obj").string().c_str(), true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BeanPod.obj").string().c_str(), true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
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

uint BindweedFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BindweedFlower.obj").string().c_str(), true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BougainvilleaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/BougainvilleaFlower.obj").string().c_str(), true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CapsicumFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
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
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaLeaf_left_highres.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else if (compound_leaf_index == 0) {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaLeaf_tip_highres.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else {
        UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaLeaf_right_highres.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/CowpeaPod.obj").string().c_str(), make_vec3(0., 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
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

// Function to generate random float between min and max
float random_float(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (max - min));
}

// Function to check if two spheres overlap
bool spheres_overlap(const helios::vec3 &center1, float radius1, const helios::vec3 &center2, float radius2) {
    float distance = std::sqrt(std::pow(center1.x - center2.x, 2) + std::pow(center1.y - center2.y, 2) + std::pow(center1.z - center2.z, 2));
    return distance < (radius1 + radius2);
}

uint GrapevineFruitPrototype(helios::Context *context_ptr, uint subdivisions) {

    int num_grapes = 60;
    float height = 5.0f; // Height of the cluster
    float base_radius = 2.f; // Base radius of the cluster
    float taper_factor = 0.6f; // Taper factor (higher means more taper)
    float grape_radius = 0.25f; // Fixed radius for each grape

    std::vector<std::pair<helios::vec3, float>> grapes;
    float z_step = height / num_grapes;

    // Place the first grape at the bottom center
    helios::vec3 first_center(0.0f, 0.0f, 0.0f);
    grapes.push_back({first_center, grape_radius});

    // Attempt to place each subsequent grape close to an existing grape
    int max_attempts = 100; // Number of retries to find a tight fit

    for (int i = 1; i < num_grapes; ++i) {
        float z = i * z_step;
        // Tapered radius based on height (denser at the top, sparser at the bottom)
        float taper_radius = base_radius * (1.0f - taper_factor * (z / height));

        bool placed = false;
        int attempts = 0;
        while (!placed && attempts < max_attempts) {
            // Randomly select an existing grape as the reference point
            int reference_idx = rand() % grapes.size();
            const helios::vec3 &reference_center = grapes[reference_idx].first;

            // Pick a random offset direction from the reference grape
            float angle = random_float(0, 2 * M_PI);
            float distance = random_float(1.2 * grape_radius, 1.3 * grape_radius); // Keep grapes close but not overlapping

            // Compute the new potential center for the grape
            helios::vec3 new_center(reference_center.x + distance * cos(angle), reference_center.y + distance * sin(angle), random_float(z - 0.5f * z_step, z + 0.5f * z_step));

            // Check that the new center is within the allowable radius (for tapering)
            float new_center_distance = std::sqrt(new_center.x * new_center.x + new_center.y * new_center.y);
            if (new_center_distance > taper_radius) {
                attempts++;
                continue; // Skip if the new position exceeds the tapered radius
            }

            // Check for collisions with existing grapes
            bool collision = false;
            for (const auto &grape: grapes) {
                if (spheres_overlap(new_center, grape_radius, grape.first, grape.second)) {
                    collision = true;
                    break;
                }
            }

            // If no collision, place the grape
            if (!collision) {
                grapes.push_back({new_center, grape_radius});
                placed = true;
            }

            attempts++;
        }
    }

    std::vector<uint> UUIDs;
    for (const auto &grape: grapes) {
        //        std::vector<uint> UUIDs_tmp = context_ptr->addSphere( 10, grape.first, grape.second, "../../../plugins/plantarchitecture/assets/textures/GrapeBerry.jpg" );
        std::vector<uint> UUIDs_tmp = context_ptr->addSphere(10, grape.first, grape.second, make_RGBcolor(0.053, 0.076, 0.098));
        UUIDs.insert(UUIDs.end(), UUIDs_tmp.begin(), UUIDs_tmp.end());
    }

    context_ptr->rotatePrimitive(UUIDs, 0.5 * M_PI, "y");

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

    // blind nodes
    if (shoot_node_index >= 2) {
        phytomer->setFloralBudState(BUD_DEAD);
    }
    if (phytomer->rank >= 1 && shoot_node_index >= 8) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
    if (phytomer->rank >= 2) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }
}

// void GrapevinePhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer ){
//
//     if( phytomer->isdormant ){
//         if( phytomer->shoot_index.x >= phytomer->shoot_index.y-1  ){
//             phytomer->setVegetativeBudState( BUD_DORMANT ); //first vegetative buds always break
//         }
//         if( phytomer->shoot_index.x <= phytomer->shoot_index.y-4  ){
//             phytomer->setFloralBudState( BUD_DORMANT ); //first vegetative buds always break
//         }
//     }
//
// }

uint MaizeTasselPrototype(helios::Context *context_ptr, uint subdivisions) {

    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/MaizeTassel.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

uint MaizeEarPrototype(helios::Context *context_ptr, uint subdivisions) {

    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/MaizeEar.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

void MaizePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // set leaf scale based on position along the shoot
    float scale = 1.f;
    if (shoot_node_index <= 5) {
        scale = fmin(1.f, 0.7 + 0.3 * float(shoot_node_index) / 5.f);
        phytomer->scaleInternodeMaxLength(scale);
    } else if (shoot_node_index >= phytomer->shoot_index.z - 5) {
        scale = fmin(1.f, 0.65 + 0.35 * float(phytomer->shoot_index.z - shoot_node_index) / 3.f);
    }

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

    //    phytomer->setFloralBudState( BUD_DEAD );
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

uint OliveFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/OliveFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    std::vector<uint> UUIDs_fruit = context_ptr->filterPrimitivesByData(context_ptr->getObjectPrimitiveUUIDs(objID), "object_label", "fruit");
    context_ptr->setPrimitiveColor(UUIDs_fruit, make_RGBcolor(0.65, 0.7, 0.4)); // green
    return objID;
}

uint OliveFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
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

uint PistachioFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/PistachioNut.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint PistachioFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
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
}

void PistachioPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    if (phytomer->isdormant) {
        if (phytomer->shoot_index.x <= phytomer->shoot_index.y - 4) {
            phytomer->setFloralBudState(BUD_DORMANT);
        }
    }
}

uint PuncturevineFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/PuncturevineFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint RedbudFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/RedbudFlower_open.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

uint RedbudFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
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

uint RiceSpikePrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/RiceGrain.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void RicePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // Leaf size and internode length are scaled separately along the shoot. A single ramp cannot serve both: the lowest leaves of a sorghum plant are already close to full length while the internodes
    // carrying them are the shortest on the plant, so one factor either stunts the bottom leaves or spreads the base of the stem out too far.
    const float leaf_scale = fmin(1.f, 0.82f + 0.18f * float(shoot_node_index) / 3.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // Internodes lengthen with height rather than shortening, so the upper leaves are not bunched together at the apex.
    const float internode_scale = fmin(1.f, 0.60f + 0.40f * float(shoot_node_index) / 5.f);
    phytomer->scaleInternodeMaxLength(internode_scale);
}

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    float fact = float(shoot_max_nodes - shoot_node_index) / float(shoot_max_nodes);

    // set leaf scale based on position along the shoot
    //    float scale = fmin(1.f, 1 + 0.1*fact);
    //    phytomer->scaleLeafPrototypeScale(scale);

    //    phytomer->rotateLeaf( 0, 0, make_AxisRotation(-deg2rad(15)*fact, 0, 0));
    phytomer->rotateLeaf(0, 0, make_AxisRotation(-deg2rad(60) * fact, 0, 0));
}

uint SorghumPaniclePrototype(helios::Context *context_ptr, uint subdivisions) {

    if (subdivisions <= 1) {
        subdivisions = 3;
    }

    float panicle_height = 1;
    float panicle_width = 0.08;
    float width_seed = 0.08;
    float height_seed = 0.25;
    float seed_tilt = 50;
    subdivisions = 6;

    std::string seed_texture_file = helios::resolvePluginAsset("plantarchitecture", "assets/textures/SorghumSeed.jpeg").string().c_str();
    RGBcolor stem_color(0.45, 0.55, 0.42);

    std::vector<uint> UUIDs;

    panicle_height -= 0.8 * height_seed;

    std::vector<vec3> nodes_panicle;
    std::vector<float> radius_panicle;

    for (int n = 0; n < subdivisions; n++) {
        float x = 0;
        float y = 0;
        float z;
        if (n == 0) {
            z = 0.5f * height_seed / float(subdivisions - 1);
        } else if (n == subdivisions - 1) {
            z = (subdivisions - 1.5f) * height_seed / float(subdivisions - 1);
        } else {
            z = n * height_seed / float(subdivisions - 1);
        }

        float angle = float(n) * M_PI / float(subdivisions - 1);
        float dr = std::fmax(0.f, 0.5f * width_seed * sin(angle));

        nodes_panicle.push_back(make_vec3(x, y, z));
        radius_panicle.push_back(dr);
    }

    std::vector<uint> UUIDs_seed_ptype = context_ptr->addTube(subdivisions, nodes_panicle, radius_panicle, seed_texture_file.c_str());

    int Ntheta = ceil(6.f * panicle_height / height_seed);
    int Nphi = ceil(2.f * M_PI * panicle_width / width_seed);

    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i < Ntheta; i++) {

            if (i == 0 && j == 0) {
                continue;
            }

            std::vector<uint> UUIDs_copy = context_ptr->copyPrimitive(UUIDs_seed_ptype);
            context_ptr->scalePrimitive(UUIDs_copy, make_vec3(1, 1, 1) * context_ptr->randu(0.9f, 1.1f));

            float phi = 2.f * M_PI * float(j + 0.5f * float(i % 2)) / float(Nphi);
            float theta = acos(1 - 2 * float(i + float(j) / float(Nphi)) / float(Ntheta));
            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = 0.5f + 0.5f * cos(theta);

            x *= 0.5f * panicle_width;
            y *= 0.5f * panicle_width;
            z *= panicle_height;

            float tilt = -deg2rad(seed_tilt) * sqrtf(1.f - z / panicle_height);

            context_ptr->rotatePrimitive(UUIDs_copy, tilt, "x");
            context_ptr->rotatePrimitive(UUIDs_copy, phi - 0.5f * M_PI, "z");

            context_ptr->translatePrimitive(UUIDs_copy, make_vec3(x, y, z));
            UUIDs.insert(UUIDs.end(), UUIDs_copy.begin(), UUIDs_copy.end());
        }
    }

    context_ptr->deletePrimitive(UUIDs_seed_ptype);

    std::vector<uint> UUIDs_sphere = context_ptr->addSphere(10, make_vec3(0, 0, 0.5 * panicle_height), 0.5f, seed_texture_file.c_str());
    context_ptr->scalePrimitiveAboutPoint(UUIDs_sphere, make_vec3(1.9 * panicle_width, 1.9 * panicle_width, 0.8 * panicle_height), make_vec3(0, 0, 0.5 * panicle_height));
    UUIDs.insert(UUIDs.end(), UUIDs_sphere.begin(), UUIDs_sphere.end());

    context_ptr->rotatePrimitive(UUIDs, 0.5f * M_PI, "y");
    context_ptr->translatePrimitive(UUIDs, make_vec3(-0.2, 0, 0));

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void SorghumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

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

    // Internodes lengthen with height rather than shortening, so the upper leaves are not bunched together at the apex.
    const float internode_scale = fmin(1.f, 0.60f + 0.40f * float(shoot_node_index) / 5.f);
    phytomer->scaleInternodeMaxLength(internode_scale);

    // The terminal phytomer carries the panicle; its leaf is the flag leaf. The global
    // rule sets the last-phytomer petiole to a small near-zero pitch (5°), which keeps
    // the petiole axis well-defined but leaves the leaf almost parallel to the panicle.
    // Add the rest of the tilt as a solid-body petiole rotation so the leaf comes along
    // and clears the panicle.
    const bool is_flag_leaf = (shoot_max_nodes > 0 && shoot_node_index + 1 == shoot_max_nodes);
    if (is_flag_leaf) {
        phytomer->rotatePetiole(0, make_AxisRotation(deg2rad(25.f), 0.f, 0.f));

        // Sorghum bears a conspicuously long internode between the last normal leaf and the flag leaf, so the flag leaf stands clear of the canopy below it rather than sitting among the upper leaves. The
        // internode ramp above saturates at 1.0 by node 5, which left this internode the same length as the eleven below it and removed the segment entirely.
        //
        // Its length is tied to the peduncle rather than set outright, so that the last-normal-leaf-to-flag-leaf distance matches the flag-leaf-to-panicle-base distance. That keeps the relationship intact
        // if the peduncle length is changed, and anchors the elongation to a quantity that is itself measured: the sorghum peduncle is 35-39 cm in three-dwarf grain types (Wang et al. 2024, Genes 15(1):83),
        // which the 0.30 m used here sits just below. No published length for this internode itself was found, so the tie to the peduncle is the calibration.
        phytomer->setInternodeMaxLength(phytomer->phytomer_parameters.peduncle.length.val());
    }
}

uint SoybeanFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/SoybeanPod.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint SoybeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
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

uint StrawberryFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/StrawberryFlower.obj").string().c_str(), make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint StrawberryFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/StrawberryFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint TomatoFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/TomatoFruit.obj").string().c_str(), make_vec3(0., 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint TomatoFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
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

uint WalnutFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ(helios::resolvePluginAsset("plantarchitecture", "assets/obj/WalnutHull.obj").string().c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint WalnutFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
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

uint WheatSpikePrototype(helios::Context *context_ptr, uint subdivisions) {
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

    // The terminal phytomer carries the ear, and its leaf is the flag leaf. The global rule gives the last phytomer on a shoot a near-zero petiole pitch, which keeps the petiole axis well defined but
    // leaves the flag leaf lying along the culm and passing straight through the spike. The rest of the tilt is added here as a solid-body rotation so the leaf swings clear of the ear, as it does on a real
    // plant, where the flag leaf is the most horizontal leaf of the canopy rather than the most upright.
    if (shoot_max_nodes > 0 && shoot_node_index + 1 == shoot_max_nodes) {
        phytomer->rotatePetiole(0, make_AxisRotation(deg2rad(28.f), 0.f, 0.f));
    }
}
