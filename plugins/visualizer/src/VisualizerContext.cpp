/** \file "VisualizerContext.cpp" Visualizer Context geometry handling functions.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "Visualizer.h"

using namespace helios;

void Visualizer::clearContextGeometry() {
    geometry_handler.clearContextGeometry();

    contextUUIDs_build.clear();
    contextUUIDs_uploaded.clear();
    colorPrimitives_UUIDs.clear();
    colorPrimitives_objIDs.clear();
    color_primitive_set_stale = true;
    depth_buffer_data.clear();
    colorbar_min = 0;
    colorbar_max = 0;
    colorbar_range_set = false;
}

void Visualizer::buildContextGeometry(helios::Context *context_ptr) {
    context = context_ptr;
    // The data-colored set belongs to the Context it was settled from; a handed-over Context may hold different primitives
    // even at the same count, so the next build settles it again.
    color_primitive_set_stale = true;

    // Asset directory registration removed - now using HELIOS_BUILD resolution

    build_all_context_geometry = true;

    // Geometry from this Context has not been rendered yet, so a capture must render rather than
    // reuse whatever frame is on the GPU.
    rendered_frame_is_current = false;

    // Restore navigation gizmo if it was enabled before displaying an image
    if (navigation_gizmo_was_enabled_before_image_display) {
        this->showNavigationGizmo();
        navigation_gizmo_was_enabled_before_image_display = false; // Reset the flag
    }
}

void Visualizer::buildContextGeometry(helios::Context *context_ptr, const std::vector<uint> &UUIDs) {
    if (UUIDs.empty()) {
        std::cerr << "WARNING (Visualizer::buildContextGeometry): There is no Context geometry to build...exiting." << std::endl;
        return;
    }

    context = context_ptr;
    color_primitive_set_stale = true; // see buildContextGeometry(Context*)

    // Asset directory registration removed - now using HELIOS_BUILD resolution

    build_all_context_geometry = false;
    contextUUIDs_build = UUIDs;

    rendered_frame_is_current = false;

    // Restore navigation gizmo if it was enabled before displaying an image
    if (navigation_gizmo_was_enabled_before_image_display) {
        this->showNavigationGizmo();
        navigation_gizmo_was_enabled_before_image_display = false; // Reset the flag
    }
}

void Visualizer::buildContextGeometry_private() {

    context_primitives_added_last_build = 0;
    context_primitives_recolored_last_build = 0;

    if (context == nullptr) { // If Visualizer::buildContextGeometry() was never called, nothing to do
        return;
    }

    // If building all context geometry, get all dirty UUIDs from the Context
    if (build_all_context_geometry) {
        bool include_deleted_UUIDs = true;
        if (contextUUIDs_build.empty()) {
            include_deleted_UUIDs = false;
        }
        contextUUIDs_build = context->getDirtyUUIDs(include_deleted_UUIDs);
    }

    // Populate contextUUIDs_needupdate based on dirty primitives in the Context
    std::vector<uint> contextUUIDs_needupdate;
    contextUUIDs_needupdate.reserve(contextUUIDs_build.size());

    // Partition contextUUIDs_build in place with a write cursor: primitives still present in the
    // Context are kept and collected for update, and primitives deleted from the Context since the
    // last build are dropped from this list and from the geometry handler. The cursor replaces a
    // range-for that swap-and-popped the very vector it was iterating - which invalidated the loop's
    // cached end iterator - and that located the element it was already positioned on with a
    // std::find, making a scene-wide deletion quadratic.
    bool context_geometry_deleted = false;
    size_t surviving_count = 0;
    for (size_t read_index = 0; read_index < contextUUIDs_build.size(); read_index++) {
        const uint UUID = contextUUIDs_build[read_index];

        if (context->doesPrimitiveExist(UUID)) {
            contextUUIDs_build[surviving_count] = UUID;
            surviving_count++;
            contextUUIDs_needupdate.push_back(UUID);
        } else {
            if (geometry_handler.doesGeometryExist(UUID)) {
                geometry_handler.deleteGeometry(UUID);
            }
            if (contextUUIDs_uploaded.erase(UUID) > 0) {
                context_geometry_deleted = true;
            }
        }
    }
    contextUUIDs_build.resize(surviving_count);

    // Settle which primitives are colored by data before fingerprinting. A data-colored primitive's fingerprint includes its
    // color-by value, so fingerprints taken before this set is filled do not match the ones the next build takes, and the
    // whole scene was rebuilt a second time on the frame after coloring by data was switched on.
    //
    // Settling the set walks every Context primitive, so it runs only when the set can have changed: the color source
    // changed, or primitives were added or deleted. Running it on every build made each idle frame of a data-colored scene
    // a pass over the whole Context.
    const size_t context_primitive_count = context->getPrimitiveCount();
    const bool color_primitive_set_may_have_changed = color_primitive_set_stale || context_geometry_deleted || context_primitive_count != color_primitive_set_primitive_count;
    if (color_primitive_set_may_have_changed && (!colorPrimitivesByData.empty() || !colorPrimitivesByObjectData.empty())) {
        color_primitive_set_refills++;
    }
    if (!color_primitive_set_may_have_changed) {
        // The set is current.
    } else if (!colorPrimitivesByData.empty()) {
        if (colorPrimitives_UUIDs.empty()) { // load all primitives
            std::vector<uint> all_UUIDs = context->getAllUUIDs();
            for (uint UUID: all_UUIDs) {
                if (context->doesPrimitiveExist(UUID)) {
                    colorPrimitives_UUIDs[UUID] = UUID;
                }
            }
        } else { // double check that primitives exist
            std::vector<uint> all_UUIDs = context->getAllUUIDs();
            for (uint UUID: all_UUIDs) {
                if (!context->doesPrimitiveExist(UUID)) {
                    auto it = colorPrimitives_UUIDs.find(UUID);
                    colorPrimitives_UUIDs.erase(it);
                }
            }
        }
    } else if (!colorPrimitivesByObjectData.empty()) {
        if (colorPrimitives_objIDs.empty()) { // load all primitives, including orphans (no parent object) which are assigned a default object-data value of 0
            std::vector<uint> all_UUIDs = context->getAllUUIDs();
            for (uint UUID: all_UUIDs) {
                if (context->doesPrimitiveExist(UUID)) {
                    colorPrimitives_UUIDs[UUID] = UUID;
                }
            }
        } else { // load primitives specified by user
            for (const auto &objID: colorPrimitives_objIDs) {
                if (context->doesObjectExist(objID.first)) {
                    std::vector<uint> UUIDs = context->getObjectPrimitiveUUIDs(objID.first);
                    for (uint UUID: UUIDs) {
                        if (context->doesPrimitiveExist(UUID)) {
                            colorPrimitives_UUIDs[UUID] = UUID;
                        }
                    }
                }
            }
        }
    }

    if (color_primitive_set_may_have_changed) {
        color_primitive_set_stale = false;
        color_primitive_set_primitive_count = context_primitive_count;
    }

    if (!colorPrimitives_UUIDs.empty() && colorbar_flag == 0) {
        enableColorbar();
    }

    // Context dirty flags are sticky - only the user's call to Context::markGeometryClean() clears them, once every
    // plug-in has processed the change - so getDirtyUUIDs() keeps reporting the whole scene on every frame. Rebuilding
    // and re-uploading a primitive that is already in the geometry handler in its current form would restore identical
    // contents, so skip the ones already uploaded.
    //
    // A sticky flag cannot distinguish "still dirty from when it was uploaded" from "modified again since", so each
    // uploaded primitive is remembered with a fingerprint of the Context state it was built from. A primitive whose
    // fingerprint no longer matches was translated, rotated, scaled, recolored, or given new color-by data in place, and
    // must be rebuilt; skipping it left the displayed scene frozen at whatever the primitive looked like when first shown.
    //
    // A change to the mapping from data to color alone (primitiveColorsNeedUpdate, or an automatic colorbar range that
    // moved) changes no primitive, so it rebuilds none: it is applied after the build by one recoloring pass over the
    // displayed geometry.
    std::vector<std::pair<uint, uint64_t>> fingerprints_to_record;
    fingerprints_to_record.reserve(contextUUIDs_needupdate.size());
    {
        std::vector<uint> contextUUIDs_changed;
        contextUUIDs_changed.reserve(contextUUIDs_needupdate.size());
        for (uint UUID: contextUUIDs_needupdate) {
            const uint64_t fingerprint = computeContextPrimitiveFingerprint(UUID);
            const auto uploaded = contextUUIDs_uploaded.find(UUID);
            if (uploaded != contextUUIDs_uploaded.end() && uploaded->second == fingerprint && geometry_handler.doesGeometryExist(UUID)) {
                continue;
            }
            contextUUIDs_changed.push_back(UUID);
            fingerprints_to_record.emplace_back(UUID, fingerprint);
        }
        contextUUIDs_needupdate = std::move(contextUUIDs_changed);
    }

    const bool coloring_by_data = !colorPrimitivesByData.empty() || !colorPrimitivesByObjectData.empty();
    // A deleted primitive may have held an extreme of an automatic colorbar range, so a deletion alone can recolor the rest.
    const bool colorbar_range_may_have_changed = context_geometry_deleted && coloring_by_data && !colorbar_range_set;

    if (contextUUIDs_needupdate.empty() && !primitiveColorsNeedUpdate && !colorbar_range_may_have_changed) {
        // No geometry work to do, but material data may still have changed without dirtying any
        // primitive, so the per-material Phong pass must still run against the existing geometry.
        updatePhongMaterialIndices();
        return;
    }

    //------ Colormap ------//

    uint psize = contextUUIDs_needupdate.size();
    if (message_flag) {
        if (psize > 0) {
            if (psize >= 1e3 && psize < 1e6) {
                std::cout << "updating " << psize / 1e3 << "K Context primitives to visualizer...." << std::flush;
            } else if (psize >= 1e6) {
                std::cout << "updating " << psize / 1e6 << "M Context primitives to visualizer...." << std::flush;
            } else {
                std::cout << "updating " << psize << " Context primitives to visualizer...." << std::flush;
            }
        } else if (context->getPrimitiveCount() == 0) {
            std::cerr << "WARNING (Visualizer::buildContextGeometry): No primitives were found in the Context..." << std::endl;
        }
    }

    // figure out colorbar range

    colormap_current.setRange(colorbar_min, colorbar_max);
    if (coloring_by_data && !colorbar_range_set) { // range was not set by user, use full range of values
        // The range spans every data-colored primitive on display as well as those about to be built. Taken over the dirty
        // primitives alone, it collapsed onto whatever had changed once the caller marked the Context geometry clean.
        std::vector<uint> range_UUIDs = contextUUIDs_needupdate;
        std::unordered_set<uint> range_UUID_set(range_UUIDs.begin(), range_UUIDs.end());
        for (uint UUID: getDisplayedContextPrimitiveUUIDs()) {
            if (range_UUID_set.insert(UUID).second) {
                range_UUIDs.push_back(UUID);
            }
        }
        updateColorbarRangeFromData(range_UUIDs);
    }

    if (!colorPrimitivesByData.empty()) {
        assert(colorbar_min <= colorbar_max);
    }

    // A new mapping from data to color - a colormap, a user range, a different color source, or an automatic range that
    // moved with the data - changes the color of primitives that did not themselves change. They are recolored after the
    // build in one pass rather than rebuilt.
    const helios::vec2 colorbar_range_this_build = make_vec2(colorbar_min, colorbar_max);
    const bool recolor_displayed_geometry = primitiveColorsNeedUpdate || (coloring_by_data && colorbar_range_this_build != colorbar_range_last_build);
    colorbar_range_last_build = colorbar_range_this_build;

    //------- Simulation Geometry -------//

    // add primitives

    size_t patch_count = context->getPatchCount();
    geometry_handler.allocateBufferSize(patch_count, GeometryHandler::GEOMETRY_TYPE_RECTANGLE);
    size_t triangle_count = context->getTriangleCount();
    geometry_handler.allocateBufferSize(triangle_count, GeometryHandler::GEOMETRY_TYPE_TRIANGLE);

    // Vertex normals belong to the parent object rather than to the individual triangle, and resolving them costs work proportional to the size of that object. Triangles are therefore collected by parent
    // object here and resolved once per object after the loop, rather than resolving the parent again for every one of its triangles. Applying them after the loop is also required for correctness, because
    // addGeometry() resets a primitive's vertex normals to its face normal, so a primitive re-added later in the loop would otherwise discard normals set earlier.
    std::unordered_map<uint, std::vector<uint>> smooth_shaded_triangles;

    for (unsigned int UUID: contextUUIDs_needupdate) {

        if (!context->doesPrimitiveExist(UUID)) {
            std::cerr << "WARNING (Visualizer::buildContextGeometry): UUID vector contains ID(s) that do not exist in the Context...they will be ignored." << std::endl;
            continue;
        }
        context_primitives_added_last_build++;


        helios::PrimitiveType ptype = context->getPrimitiveType(UUID);

        const std::vector<vec3> verts = context->getPrimitiveVertices(UUID);
        const std::string texture_file = context->getPrimitiveTextureFile(UUID);

        RGBAcolor color;
        float colorValue;
        if (!colorPrimitivesByData.empty()) {
            if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                if (context->doesPrimitiveDataExist(UUID, colorPrimitivesByData.c_str())) {
                    HeliosDataType type = context->getPrimitiveDataType(colorPrimitivesByData.c_str());
                    if (type == HELIOS_TYPE_FLOAT) {
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), colorValue);
                    } else if (type == HELIOS_TYPE_INT) {
                        int cv;
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_UINT) {
                        uint cv;
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_DOUBLE) {
                        double cv;
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else {
                        colorValue = 0;
                    }
                } else {
                    colorValue = 0;
                }

                if (std::isnan(colorValue) || std::isinf(colorValue)) { // check for NaN or infinity
                    colorValue = 0;
                }

                color = make_RGBAcolor(colormap_current.query(colorValue), 1);
            } else {
                color = context->getPrimitiveColorRGBA(UUID);
            }
        } else if (!colorPrimitivesByObjectData.empty()) {
            if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                uint ObjID = context->getPrimitiveParentObjectID(UUID);
                if (ObjID != 0 && context->doesObjectDataExist(ObjID, colorPrimitivesByObjectData.c_str())) {
                    HeliosDataType type = context->getObjectDataType(colorPrimitivesByObjectData.c_str());
                    if (type == HELIOS_TYPE_FLOAT) {
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), colorValue);
                    } else if (type == HELIOS_TYPE_INT) {
                        int cv;
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_UINT) {
                        uint cv;
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_DOUBLE) {
                        double cv;
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                        colorValue = float(cv);
                    } else {
                        colorValue = 0;
                    }
                } else {
                    colorValue = 0;
                }

                if (std::isnan(colorValue) || std::isinf(colorValue)) { // check for NaN or infinity
                    colorValue = 0;
                }

                color = make_RGBAcolor(colormap_current.query(colorValue), 1);
            } else {
                color = context->getPrimitiveColorRGBA(UUID);
            }
        } else {
            color = context->getPrimitiveColorRGBA(UUID);
        }

        int textureID = -1;
        if (!texture_file.empty()) {
            textureID = registerTextureImage(texture_file);
        }

        // ---- PATCHES ---- //
        if (ptype == helios::PRIMITIVE_TYPE_PATCH) {
            // - Patch does not have an associated texture or we are ignoring texture
            if (texture_file.empty()) {
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, verts, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
            }
            // - Patch has a texture
            else {
                std::vector<vec2> uvs = context->getPrimitiveTextureUV(UUID);

                // - coloring primitive based on texture
                if (!isContextPrimitiveTextureColorOverridden(UUID)) {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, verts, color, uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                }
                // - coloring primitive based on primitive data
                else {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, verts, color, uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                }
            }
        }
        // ---- TRIANGLES ---- //
        else if (ptype == helios::PRIMITIVE_TYPE_TRIANGLE) {
            // - Triangle does not have an associated texture or we are ignoring texture
            if (texture_file.empty()) {
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, verts, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
            }
            // - Triangle has a texture
            else {
                std::vector<vec2> uvs = context->getPrimitiveTextureUV(UUID);

                // - coloring primitive based on texture
                if (!isContextPrimitiveTextureColorOverridden(UUID)) {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, verts, color, uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                }
                // - coloring primitive based on RGB color but mask using texture
                else {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, verts, color, uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                }
            }

            // A triangle belonging to an object that can supply per-vertex normals is shaded smoothly rather than faceted: a Polymesh that retains authored or computed normals, or a curved object whose
            // normal is known analytically. Primitives with neither keep the face normal that addGeometry() already replicated across their vertices, so nothing needs to be done for them. The normals
            // themselves are resolved after the loop, once per parent object.
            const uint parent_ObjID = context->getPrimitiveParentObjectID(UUID);
            if (parent_ObjID != 0 && context->doesObjectExist(parent_ObjID)) {
                const bool polymesh_has_normals =
                        context->getObjectType(parent_ObjID) == helios::OBJECT_TYPE_POLYMESH && context->getPolymeshObjectVertexNormalSource(parent_ObjID) != helios::NORMAL_SOURCE_NONE;
                if (polymesh_has_normals || context->doesObjectHaveAnalyticVertexNormals(parent_ObjID)) {
                    smooth_shaded_triangles[parent_ObjID].push_back(UUID);
                }
            }
        }
        // ---- VOXELS ---- //
        else if (ptype == helios::PRIMITIVE_TYPE_VOXEL) {
            std::vector<vec3> v_vertices = context->getPrimitiveVertices(UUID);

            // bottom
            const std::vector<vec3> bottom_vertices{v_vertices.at(0), v_vertices.at(1), v_vertices.at(2), v_vertices.at(3)};

            // top
            const std::vector<vec3> top_vertices{v_vertices.at(4), v_vertices.at(5), v_vertices.at(6), v_vertices.at(7)};

            //-x
            const std::vector<vec3> mx_vertices{v_vertices.at(0), v_vertices.at(3), v_vertices.at(7), v_vertices.at(4)};

            //+x
            const std::vector<vec3> px_vertices{v_vertices.at(1), v_vertices.at(2), v_vertices.at(6), v_vertices.at(5)};

            //-y
            const std::vector<vec3> my_vertices{v_vertices.at(0), v_vertices.at(1), v_vertices.at(5), v_vertices.at(4)};

            //+y
            const std::vector<vec3> py_vertices{v_vertices.at(2), v_vertices.at(3), v_vertices.at(7), v_vertices.at(6)};

            // Voxel does not have an associated texture or we are ignoring texture
            if (texture_file.empty()) {
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, bottom_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, top_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, mx_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, px_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, my_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, py_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
            }
            // Voxel has a texture
            else {
                const std::vector<helios::vec2> voxel_uvs = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};

                // coloring primitive based on texture
                if (!isContextPrimitiveTextureColorOverridden(UUID)) {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, bottom_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, top_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, mx_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, px_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, my_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, py_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                }
                // coloring primitive based on primitive data
                else {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, bottom_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, top_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, mx_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, px_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, my_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, py_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                }
            }
        }
    }

    // Apply per-vertex normals, one parent object at a time. Everything expensive - a Polymesh's vertex and face tables, a tube's node tangents and taper rates - is fetched once per object here instead of
    // once per triangle.
    for (const auto &[parent_ObjID, object_UUIDs]: smooth_shaded_triangles) {

        if (context->getObjectType(parent_ObjID) == helios::OBJECT_TYPE_POLYMESH) {
            const std::vector<helios::vec3> mesh_vertex_normals = context->getPolymeshObjectVertexNormals(parent_ObjID);
            const std::vector<helios::int3> mesh_faces = context->getPolymeshObjectFaces(parent_ObjID);
            for (uint UUID: object_UUIDs) {
                const size_t face_index = context->getPolymeshObjectFaceIndexForPrimitive(parent_ObjID, UUID);
                const helios::int3 &face = mesh_faces.at(face_index);
                geometry_handler.setVertexNormals(UUID, {mesh_vertex_normals.at(face.x), mesh_vertex_normals.at(face.y), mesh_vertex_normals.at(face.z)});
            }
        } else {
            const std::vector<std::vector<helios::vec3>> object_normals = context->getObjectPrimitiveVertexNormals(parent_ObjID, object_UUIDs);
            for (size_t k = 0; k < object_UUIDs.size(); k++) {
                // An object whose transformation is degenerate returns no normals, and the face normal already in place remains the best available.
                if (object_normals.at(k).size() == 3) {
                    geometry_handler.setVertexNormals(object_UUIDs.at(k), object_normals.at(k));
                }
            }
        }
    }

    if (recolor_displayed_geometry) {
        // The primitives just built already carry the new mapping, so they are skipped rather than colored twice.
        const std::unordered_set<uint> UUIDs_just_built(contextUUIDs_needupdate.begin(), contextUUIDs_needupdate.end());
        recolorContextGeometry(UUIDs_just_built);
    }
    primitiveColorsNeedUpdate = false;

    // Record what was just built, and the Context state it was built from, so that the next frame does not rebuild these
    // primitives again purely because their Context dirty flags are still set - but does rebuild them if they change.
    for (const auto &[UUID, fingerprint]: fingerprints_to_record) {
        contextUUIDs_uploaded[UUID] = fingerprint;
    }

    updatePhongMaterialIndices();
}

bool Visualizer::getContextPrimitiveColorValue(uint UUID, float &value) const {
    // Mirrors how buildContextGeometry_private() picks the value it maps through the colormap.
    if (colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end()) {
        return false;
    }

    value = 0.f;
    if (!colorPrimitivesByData.empty()) {
        if (!context->doesPrimitiveDataExist(UUID, colorPrimitivesByData.c_str())) {
            return true;
        }
        const HeliosDataType type = context->getPrimitiveDataType(colorPrimitivesByData.c_str());
        if (type == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), value);
        } else if (type == HELIOS_TYPE_INT) {
            int cv;
            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
            value = float(cv);
        } else if (type == HELIOS_TYPE_UINT) {
            uint cv;
            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
            value = float(cv);
        } else if (type == HELIOS_TYPE_DOUBLE) {
            double cv;
            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
            value = float(cv);
        }
        return true;
    }

    if (!colorPrimitivesByObjectData.empty()) {
        const uint ObjID = context->getPrimitiveParentObjectID(UUID);
        if (ObjID == 0 || !context->doesObjectDataExist(ObjID, colorPrimitivesByObjectData.c_str())) {
            return true;
        }
        const HeliosDataType type = context->getObjectDataType(colorPrimitivesByObjectData.c_str());
        if (type == HELIOS_TYPE_FLOAT) {
            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), value);
        } else if (type == HELIOS_TYPE_INT) {
            int cv;
            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
            value = float(cv);
        } else if (type == HELIOS_TYPE_UINT) {
            uint cv;
            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
            value = float(cv);
        } else if (type == HELIOS_TYPE_DOUBLE) {
            double cv;
            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
            value = float(cv);
        }
        return true;
    }

    return false;
}

// FNV-1a parameters. These live at file scope rather than inside computeContextPrimitiveFingerprint() because MSVC
// refuses to read a constexpr local from a lambda that has an explicit capture list (C3493), which GCC and Clang allow.
static constexpr uint64_t FNV_offset_basis = 14695981039346656037ull;
static constexpr uint64_t FNV_prime = 1099511628211ull;

uint64_t Visualizer::computeContextPrimitiveFingerprint(uint UUID) const {
    // FNV-1a over the raw bytes of everything the build reads from the primitive. The transformation matrix determines
    // the vertices of a patch, triangle, or voxel, so a translate/rotate/scale shows up here without fetching vertices.
    uint64_t hash = FNV_offset_basis;
    auto hash_bytes = [&hash](const void *data, size_t byte_count) {
        const auto *bytes = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < byte_count; i++) {
            hash ^= bytes[i];
            hash *= FNV_prime;
        }
    };

    float transform[16];
    context->getPrimitiveTransformationMatrix(UUID, transform);
    hash_bytes(transform, sizeof(transform));

    const RGBAcolor color = context->getPrimitiveColorRGBA(UUID);
    const float color_components[4] = {color.r, color.g, color.b, color.a};
    hash_bytes(color_components, sizeof(color_components));

    const unsigned char texture_color_overridden = context->isPrimitiveTextureColorOverridden(UUID) ? 1 : 0;
    hash_bytes(&texture_color_overridden, sizeof(texture_color_overridden));

    // A different texture file or texture coordinates change what is drawn without moving or recoloring the primitive.
    const std::string texture_file = context->getPrimitiveTextureFile(UUID);
    hash_bytes(texture_file.data(), texture_file.size());
    if (!texture_file.empty()) {
        const std::vector<helios::vec2> uvs = context->getPrimitiveTextureUV(UUID);
        hash_bytes(uvs.data(), uvs.size() * sizeof(helios::vec2));
    }

    float color_value;
    if (getContextPrimitiveColorValue(UUID, color_value)) {
        if (std::isnan(color_value) || std::isinf(color_value)) {
            color_value = 0.f;
        }
        hash_bytes(&color_value, sizeof(color_value));
    }

    return hash;
}

void Visualizer::updatePhongMaterialIndices() {

    if (context == nullptr) {
        return;
    }

    // The geometry handler's live set is the true set of displayed primitives. contextUUIDs_build
    // holds only the dirty UUIDs when building all context geometry, so it would miss primitives
    // whose material data changed while the primitives themselves stayed clean. Geometry added
    // directly to the Visualizer rather than from the Context is skipped: it has no material. It has to
    // be skipped by its flag, not by asking the Context whether the ID exists: such geometry carries a
    // random 64-bit ID, and truncated to a UUID that can equal an unrelated Context primitive.
    const std::vector<size_t> live_geometry = geometry_handler.getAllGeometryIDs();

    std::set<uint> referenced_material_IDs;
    for (size_t geometry_UUID: live_geometry) {
        if (!geometry_handler.isContextGeometry(geometry_UUID)) {
            continue;
        }
        const uint UUID = static_cast<uint>(geometry_UUID);
        if (context->doesPrimitiveExist(UUID)) {
            referenced_material_IDs.insert(context->getPrimitiveMaterialID(UUID));
        }
    }

    const std::unordered_map<uint, int> material_ID_to_index = buildPhongMaterialTable(context, referenced_material_IDs);

    // Reassigning indices walks every displayed primitive, so it is skipped when the resolved Phong
    // data is identical to the previous build. Building the table itself is O(materials) and cheap
    // enough to redo each time, and it is what produces the fingerprint.
    std::vector<float> fingerprint;
    fingerprint.reserve(material_ID_to_index.size() * 2);
    for (const auto &[materialID, table_index]: material_ID_to_index) {
        fingerprint.push_back(float(materialID));
        fingerprint.push_back(float(table_index));
    }
    std::sort(fingerprint.begin(), fingerprint.end());

    if (phong_material_indices_assigned && fingerprint == phong_material_fingerprint) {
        return;
    }

    for (size_t geometry_UUID: live_geometry) {
        if (!geometry_handler.isContextGeometry(geometry_UUID)) {
            continue;
        }
        const uint UUID = static_cast<uint>(geometry_UUID);
        if (!context->doesPrimitiveExist(UUID)) {
            continue;
        }
        const auto it = material_ID_to_index.find(context->getPrimitiveMaterialID(UUID));
        // A material with no Phong data is absent from the map; -1 means "use the global material".
        geometry_handler.setMaterialIndex(UUID, it != material_ID_to_index.end() ? it->second : -1);
    }

    phong_material_fingerprint = fingerprint;
    phong_material_indices_assigned = true;
}

void Visualizer::updateContextPrimitiveColors() {
    // Only Context geometry is recolored, and without a Context there is none. The flag is left set, so geometry built once
    // a Context is attached is still colored with the current mapping.
    if (context == nullptr) {
        return;
    }

    colormap_current.setRange(colorbar_min, colorbar_max);
    if ((!colorPrimitivesByData.empty() || !colorPrimitivesByObjectData.empty()) && !colorbar_range_set) {
        updateColorbarRangeFromData(getDisplayedContextPrimitiveUUIDs());
    }

    recolorContextGeometry({});

    colorbar_range_last_build = make_vec2(colorbar_min, colorbar_max);
    primitiveColorsNeedUpdate = false;
}

std::vector<uint> Visualizer::getDisplayedContextPrimitiveUUIDs() const {
    std::vector<uint> UUIDs;
    for (const size_t geometry_ID: geometry_handler.getAllGeometryIDs()) {
        // Geometry added directly to the Visualizer carries a random 64-bit ID that, truncated, can equal a Context UUID, so
        // it is told apart by its flag. Context geometry is keyed by its UUID, so for it the cast is exact.
        if (!geometry_handler.isContextGeometry(geometry_ID)) {
            continue;
        }
        const uint UUID = static_cast<uint>(geometry_ID);
        if (context->doesPrimitiveExist(UUID)) {
            UUIDs.push_back(UUID);
        }
    }
    return UUIDs;
}

void Visualizer::updateColorbarRangeFromData(const std::vector<uint> &UUIDs) {
    colorbar_min = (std::numeric_limits<float>::max)();
    colorbar_max = (std::numeric_limits<float>::lowest)();

    // Initialize integer data flag (will be set on first data encounter)
    colorbar_integer_data = false;
    bool data_type_detected = false;
    bool data_value_found = false;

    for (const uint UUID: UUIDs) {
        float color_value;
        if (!getContextPrimitiveColorValue(UUID, color_value)) {
            continue;
        }

        // Detect data type on first encounter
        if (!data_type_detected) {
            if (!colorPrimitivesByData.empty()) {
                if (context->doesPrimitiveDataExist(UUID, colorPrimitivesByData.c_str())) {
                    const HeliosDataType type = context->getPrimitiveDataType(colorPrimitivesByData.c_str());
                    colorbar_integer_data = (type == HELIOS_TYPE_INT || type == HELIOS_TYPE_UINT);
                    data_type_detected = true;
                }
            } else {
                const uint ObjID = context->getPrimitiveParentObjectID(UUID);
                if (ObjID != 0 && context->doesObjectDataExist(ObjID, colorPrimitivesByObjectData.c_str())) {
                    const HeliosDataType type = context->getObjectDataType(colorPrimitivesByObjectData.c_str());
                    colorbar_integer_data = (type == HELIOS_TYPE_INT || type == HELIOS_TYPE_UINT);
                    data_type_detected = true;
                }
            }
        }

        if (std::isnan(color_value) || std::isinf(color_value)) {
            color_value = 0.f;
        }

        data_value_found = true;
        colorbar_min = std::min(colorbar_min, color_value);
        colorbar_max = std::max(colorbar_max, color_value);
    }

    // Test that a value was actually found rather than testing the accumulators for infinity: they are seeded with
    // FLT_MAX/lowest(), which are finite, so an empty scan passed the old isinf() check and handed the colormap an inverted
    // range.
    if (data_value_found) {
        colormap_current.setRange(colorbar_min, colorbar_max);
    } else {
        // Nothing contributed a value, so leave the colormap range as it was rather than publishing the sentinels.
        colorbar_min = 0.f;
        colorbar_max = 0.f;
    }
}

void Visualizer::recolorContextGeometry(const std::unordered_set<uint> &UUIDs_to_skip) {
    for (const size_t geometry_ID: geometry_handler.getAllGeometryIDs()) {
        // See getDisplayedContextPrimitiveUUIDs(): non-Context geometry must be skipped by its flag, not by its truncated ID.
        if (!geometry_handler.isContextGeometry(geometry_ID)) {
            continue;
        }
        const uint UUID = static_cast<uint>(geometry_ID);
        if (UUIDs_to_skip.find(UUID) != UUIDs_to_skip.end() || !context->doesPrimitiveExist(UUID)) {
            continue;
        }

        // The same color and texture rules the build applies when it adds the primitive.
        RGBAcolor color = context->getPrimitiveColorRGBA(UUID);
        float color_value;
        if (getContextPrimitiveColorValue(UUID, color_value)) {
            if (std::isnan(color_value) || std::isinf(color_value)) {
                color_value = 0.f;
            }
            color = make_RGBAcolor(colormap_current.query(color_value), 1.f);
        }

        if (!context->getPrimitiveTextureFile(UUID).empty()) {
            if (isContextPrimitiveTextureColorOverridden(UUID)) {
                geometry_handler.overrideTextureColor(geometry_ID);
            } else {
                geometry_handler.useTextureColor(geometry_ID);
            }
        }

        geometry_handler.setColor(geometry_ID, color);
        context_primitives_recolored_last_build++;
    }
}

bool Visualizer::isContextPrimitiveTextureColorOverridden(uint UUID) const {
    return colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end() || context->isPrimitiveTextureColorOverridden(UUID);
}

void Visualizer::colorContextPrimitivesByData(const char *data_name) {
    colorPrimitivesByData = data_name;
    colorPrimitivesByObjectData = "";
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
    color_primitive_set_stale = true;
}

void Visualizer::colorContextPrimitivesByData(const char *data_name, const std::vector<uint> &UUIDs) {
    colorPrimitivesByData = data_name;
    colorPrimitivesByObjectData = "";
    for (uint UUID: UUIDs) {
        colorPrimitives_UUIDs[UUID] = UUID;
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
    color_primitive_set_stale = true;
}

void Visualizer::colorContextPrimitivesByObjectData(const char *data_name) {
    colorPrimitivesByObjectData = data_name;
    colorPrimitivesByData = "";
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
    color_primitive_set_stale = true;
}

void Visualizer::colorContextPrimitivesByObjectData(const char *data_name, const std::vector<uint> &ObjIDs) {
    colorPrimitivesByObjectData = data_name;
    colorPrimitivesByData = "";
    for (uint objID: ObjIDs) {
        colorPrimitives_objIDs[objID] = objID;
    }
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
    color_primitive_set_stale = true;
}

void Visualizer::colorContextPrimitivesRandomly(const std::vector<uint> &UUIDs) {
    disableColorbar();
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    for (uint UUID: UUIDs) {
        float rc = randu();
        context->setPrimitiveData(UUID, "random_color", rc);
    }

    colorPrimitivesByData = "random_color";
    colorPrimitivesByObjectData = "";
    for (uint UUID: UUIDs) {
        colorPrimitives_UUIDs[UUID] = UUID;
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
    color_primitive_set_stale = true;
}

void Visualizer::colorContextPrimitivesRandomly() {
    disableColorbar();

    std::vector<uint> all_UUIDs = context->getAllUUIDs();
    for (uint UUID: all_UUIDs) {
        float rc = randu();
        context->setPrimitiveData(UUID, "random_color", rc);
    }

    colorPrimitivesByData = "random_color";
    colorPrimitivesByObjectData = "";
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
    color_primitive_set_stale = true;
}


void Visualizer::colorContextObjectsRandomly(const std::vector<uint> &ObjIDs) {
    disableColorbar();
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    for (uint ObjID: ObjIDs) {
        float rc = randu();
        context->setObjectData(ObjID, "random_color", rc);
    }

    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "random_color";
    primitiveColorsNeedUpdate = true;
}

void Visualizer::colorContextObjectsRandomly() {
    std::vector<uint> all_ObjIDs = context->getAllObjectIDs();
    disableColorbar();
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    for (uint ObjID: all_ObjIDs) {
        float rc = randu();
        context->setObjectData(ObjID, "random_color", rc);
    }

    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "random_color";
    primitiveColorsNeedUpdate = true;
}

void Visualizer::clearColor() {
    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "";
    colorbar_integer_data = false;
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    disableColorbar();
    colorbar_min = 0;
    colorbar_max = 0;
    colorbar_range_set = false;
    colorbar_flag = 0;
    primitiveColorsNeedUpdate = true;
}
