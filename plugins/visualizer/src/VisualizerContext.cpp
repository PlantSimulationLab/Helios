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
    depth_buffer_data.clear();
    colorbar_min = 0;
    colorbar_max = 0;
    colorbar_range_set = false;
}

void Visualizer::buildContextGeometry(helios::Context *context_ptr) {
    context = context_ptr;

    // Asset directory registration removed - now using HELIOS_BUILD resolution

    build_all_context_geometry = true;

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

    // Asset directory registration removed - now using HELIOS_BUILD resolution

    build_all_context_geometry = false;
    contextUUIDs_build = UUIDs;

    // Restore navigation gizmo if it was enabled before displaying an image
    if (navigation_gizmo_was_enabled_before_image_display) {
        this->showNavigationGizmo();
        navigation_gizmo_was_enabled_before_image_display = false; // Reset the flag
    }
}

void Visualizer::buildContextGeometry_private() {

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

    for (uint UUID: contextUUIDs_build) {

        // Check if primitives in contextUUIDs_build have since been deleted from the Context. If so, remove them from contextUUIDs_build and from the geometry handler
        if (!context->doesPrimitiveExist(UUID)) {
            auto it = std::find(contextUUIDs_build.begin(), contextUUIDs_build.end(), UUID);
            if (it != contextUUIDs_build.end()) {
                // swap-and-pop delete from contextUUIDs_build
                *it = contextUUIDs_build.back();
                contextUUIDs_build.pop_back();
                // delete from the geometry handler
                if (geometry_handler.doesGeometryExist(UUID)) {
                    geometry_handler.deleteGeometry(UUID);
                }
                contextUUIDs_uploaded.erase(UUID);
            }
        }
        // check if the primitive is dirty, if so, add it to contextUUIDs_needupdate
        else {
            contextUUIDs_needupdate.push_back(UUID);
        }
    }

    // Context dirty flags are sticky - only the user's call to Context::markGeometryClean() clears them, once every
    // plug-in has processed the change - so getDirtyUUIDs() keeps reporting the whole scene on every frame. Rebuilding
    // and re-uploading a primitive that is already in the geometry handler in its current form would restore identical
    // contents, so skip the ones already uploaded.
    //
    // primitiveColorsNeedUpdate means the mapping from primitive/object data to color has changed, which changes the
    // color of primitives whose geometry did not change at all, so in that case every dirty primitive must be rebuilt.
    if (!contextUUIDs_needupdate.empty() && !primitiveColorsNeedUpdate) {
        std::vector<uint> contextUUIDs_notyetuploaded;
        contextUUIDs_notyetuploaded.reserve(contextUUIDs_needupdate.size());
        for (uint UUID: contextUUIDs_needupdate) {
            if (contextUUIDs_uploaded.find(UUID) == contextUUIDs_uploaded.end() || !geometry_handler.doesGeometryExist(UUID)) {
                contextUUIDs_notyetuploaded.push_back(UUID);
            }
        }
        contextUUIDs_needupdate = std::move(contextUUIDs_notyetuploaded);
    }

    if (contextUUIDs_needupdate.empty() && !primitiveColorsNeedUpdate) {
        // No geometry work to do, but material data may still have changed without dirtying any
        // primitive, so the per-material Phong pass must still run against the existing geometry.
        updatePhongMaterialIndices();
        return;
    }

    if (!colorPrimitivesByData.empty()) {
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

    if (!colorPrimitives_UUIDs.empty() && colorbar_flag == 0) {
        enableColorbar();
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
        } else {
            std::cerr << "WARNING (Visualizer::buildContextGeometry): No primitives were found in the Context..." << std::endl;
        }
    }

    // figure out colorbar range
    //  \todo Figure out how to avoid doing this when not necessary

    colormap_current.setRange(colorbar_min, colorbar_max);
    if ((!colorPrimitivesByData.empty() || !colorPrimitivesByObjectData.empty()) && !colorbar_range_set) { // range was not set by user, use full range of values

        colorbar_min = (std::numeric_limits<float>::max)();
        colorbar_max = (std::numeric_limits<float>::lowest)();

        // Initialize integer data flag (will be set on first data encounter)
        colorbar_integer_data = false;
        bool data_type_detected = false;
        bool data_value_found = false;

        for (uint UUID: contextUUIDs_build) {
            float colorValue = -9999;
            if (!colorPrimitivesByData.empty()) {
                if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                    if (context->doesPrimitiveDataExist(UUID, colorPrimitivesByData.c_str())) {
                        HeliosDataType type = context->getPrimitiveDataType(colorPrimitivesByData.c_str());

                        // Detect data type on first encounter
                        if (!data_type_detected) {
                            colorbar_integer_data = (type == HELIOS_TYPE_INT || type == HELIOS_TYPE_UINT);
                            data_type_detected = true;
                        }

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
                }
            } else if (!colorPrimitivesByObjectData.empty()) {
                if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                    uint ObjID = context->getPrimitiveParentObjectID(UUID);
                    if (ObjID != 0 && context->doesObjectDataExist(ObjID, colorPrimitivesByObjectData.c_str())) {
                        HeliosDataType type = context->getObjectDataType(colorPrimitivesByObjectData.c_str());

                        // Detect data type on first encounter
                        if (!data_type_detected) {
                            colorbar_integer_data = (type == HELIOS_TYPE_INT || type == HELIOS_TYPE_UINT);
                            data_type_detected = true;
                        }

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
                }
            }

            if (std::isnan(colorValue) || std::isinf(colorValue)) { // check for NaN or infinity
                colorValue = 0;
            }

            if (colorValue != -9999) {
                data_value_found = true;
                if (colorValue < colorbar_min) {
                    colorbar_min = colorValue;
                }
                if (colorValue > colorbar_max) {
                    colorbar_max = colorValue;
                }
            }
        }

        // Test that a value was actually found rather than testing the accumulators for infinity:
        // they are seeded with FLT_MAX/lowest(), which are finite, so an empty scan passed the old
        // isinf() check and handed the colormap an inverted range.
        if (data_value_found) {
            colormap_current.setRange(colorbar_min, colorbar_max);
        } else {
            // Nothing contributed a value, so leave the range as it was rather than publishing the
            // sentinels.
            colorbar_min = 0.f;
            colorbar_max = 0.f;
        }
    }

    if (!colorPrimitivesByData.empty()) {
        assert(colorbar_min <= colorbar_max);
    }

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
                if ((colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.empty()) && !context->isPrimitiveTextureColorOverridden(UUID)) {
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
                if ((colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.empty()) && !context->isPrimitiveTextureColorOverridden(UUID)) {
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
                if ((colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.empty()) && context->isPrimitiveTextureColorOverridden(UUID)) {
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

    if (primitiveColorsNeedUpdate) {
        updateContextPrimitiveColors();
    }

    // Record what was just built, so that the next frame does not rebuild these primitives again purely because their
    // Context dirty flags are still set.
    contextUUIDs_uploaded.insert(contextUUIDs_needupdate.begin(), contextUUIDs_needupdate.end());

    updatePhongMaterialIndices();
}

void Visualizer::updatePhongMaterialIndices() {

    if (context == nullptr) {
        return;
    }

    // The geometry handler's live set is the true set of displayed primitives. contextUUIDs_build
    // holds only the dirty UUIDs when building all context geometry, so it would miss primitives
    // whose material data changed while the primitives themselves stayed clean. Geometry added
    // directly to the Visualizer rather than from the Context is skipped: it has no material.
    const std::vector<size_t> live_geometry = geometry_handler.getAllGeometryIDs();

    std::set<uint> referenced_material_IDs;
    for (size_t geometry_UUID: live_geometry) {
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

    std::vector<size_t> geometry_UUIDs = geometry_handler.getAllGeometryIDs();

    if (geometry_UUIDs.empty()) {
        primitiveColorsNeedUpdate = false;
        return;
    }

    colormap_current.setRange(colorbar_min, colorbar_max);

    if ((!colorPrimitivesByData.empty() || !colorPrimitivesByObjectData.empty()) && !colorbar_range_set) {
        colorbar_min = (std::numeric_limits<float>::max)();
        colorbar_max = (std::numeric_limits<float>::lowest)();

        // Initialize integer data flag (will be set on first data encounter)
        colorbar_integer_data = false;
        bool data_value_found = false;
        bool data_type_detected = false;

        for (auto UUID: geometry_UUIDs) {
            if (!context->doesPrimitiveExist(static_cast<uint>(UUID))) {
                continue;
            }

            float colorValue = -9999.f;
            if (!colorPrimitivesByData.empty()) {
                if (colorPrimitives_UUIDs.find(static_cast<uint>(UUID)) != colorPrimitives_UUIDs.end()) {
                    if (context->doesPrimitiveDataExist(static_cast<uint>(UUID), colorPrimitivesByData.c_str())) {
                        HeliosDataType type = context->getPrimitiveDataType(colorPrimitivesByData.c_str());

                        // Detect data type on first encounter
                        if (!data_type_detected) {
                            colorbar_integer_data = (type == HELIOS_TYPE_INT || type == HELIOS_TYPE_UINT);
                            data_type_detected = true;
                        }

                        if (type == HELIOS_TYPE_FLOAT) {
                            context->getPrimitiveData(static_cast<uint>(UUID), colorPrimitivesByData.c_str(), colorValue);
                        } else if (type == HELIOS_TYPE_INT) {
                            int cv;
                            context->getPrimitiveData(static_cast<uint>(UUID), colorPrimitivesByData.c_str(), cv);
                            colorValue = float(cv);
                        } else if (type == HELIOS_TYPE_UINT) {
                            uint cv;
                            context->getPrimitiveData(static_cast<uint>(UUID), colorPrimitivesByData.c_str(), cv);
                            colorValue = float(cv);
                        } else if (type == HELIOS_TYPE_DOUBLE) {
                            double cv;
                            context->getPrimitiveData(static_cast<uint>(UUID), colorPrimitivesByData.c_str(), cv);
                            colorValue = float(cv);
                        } else {
                            colorValue = 0.f;
                        }
                    } else {
                        colorValue = 0.f;
                    }
                }
            } else if (!colorPrimitivesByObjectData.empty()) {
                if (colorPrimitives_UUIDs.find(static_cast<uint>(UUID)) != colorPrimitives_UUIDs.end()) {
                    uint ObjID = context->getPrimitiveParentObjectID(static_cast<uint>(UUID));
                    if (ObjID != 0 && context->doesObjectDataExist(ObjID, colorPrimitivesByObjectData.c_str())) {
                        HeliosDataType type = context->getObjectDataType(colorPrimitivesByObjectData.c_str());

                        // Detect data type on first encounter
                        if (!data_type_detected) {
                            colorbar_integer_data = (type == HELIOS_TYPE_INT || type == HELIOS_TYPE_UINT);
                            data_type_detected = true;
                        }

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
                            colorValue = 0.f;
                        }
                    } else {
                        colorValue = 0.f;
                    }
                }
            }

            if (std::isnan(colorValue) || std::isinf(colorValue)) {
                colorValue = 0.f;
            }

            if (colorValue != -9999.f) {
                data_value_found = true;
                if (colorValue < colorbar_min) {
                    colorbar_min = colorValue;
                }
                if (colorValue > colorbar_max) {
                    colorbar_max = colorValue;
                }
            }
        }

        // See the matching comment in buildContextGeometry(): the accumulators are seeded with
        // finite sentinels, so isinf() never detected an empty scan.
        if (data_value_found) {
            colormap_current.setRange(colorbar_min, colorbar_max);
        } else {
            colorbar_min = 0.f;
            colorbar_max = 0.f;
        }
    }

    for (auto UUID: geometry_UUIDs) {
        uint uid = static_cast<uint>(UUID);
        if (!context->doesPrimitiveExist(uid) || !geometry_handler.doesGeometryExist(UUID)) {
            continue;
        }

        RGBAcolor color = context->getPrimitiveColorRGBA(uid);

        const std::string texture_file = context->getPrimitiveTextureFile(uid);

        if (!colorPrimitivesByData.empty()) {
            if (colorPrimitives_UUIDs.find(uid) != colorPrimitives_UUIDs.end()) {
                float colorValue = 0.f;
                if (context->doesPrimitiveDataExist(uid, colorPrimitivesByData.c_str())) {
                    HeliosDataType type = context->getPrimitiveDataType(colorPrimitivesByData.c_str());
                    if (type == HELIOS_TYPE_FLOAT) {
                        context->getPrimitiveData(uid, colorPrimitivesByData.c_str(), colorValue);
                    } else if (type == HELIOS_TYPE_INT) {
                        int cv;
                        context->getPrimitiveData(uid, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_UINT) {
                        uint cv;
                        context->getPrimitiveData(uid, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_DOUBLE) {
                        double cv;
                        context->getPrimitiveData(uid, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else {
                        colorValue = 0.f;
                    }
                }

                if (std::isnan(colorValue) || std::isinf(colorValue)) {
                    colorValue = 0.f;
                }

                color = make_RGBAcolor(colormap_current.query(colorValue), 1.f);

                if (!texture_file.empty()) {
                    geometry_handler.overrideTextureColor(UUID);
                }
            } else if (!texture_file.empty()) {
                geometry_handler.useTextureColor(UUID);
            }
        } else if (!colorPrimitivesByObjectData.empty()) {
            if (colorPrimitives_UUIDs.find(uid) != colorPrimitives_UUIDs.end()) {
                float colorValue = 0.f;
                uint ObjID = context->getPrimitiveParentObjectID(uid);
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
                        colorValue = 0.f;
                    }
                }

                if (std::isnan(colorValue) || std::isinf(colorValue)) {
                    colorValue = 0.f;
                }

                color = make_RGBAcolor(colormap_current.query(colorValue), 1.f);

                if (!texture_file.empty()) {
                    geometry_handler.overrideTextureColor(UUID);
                }
            } else if (!texture_file.empty()) {
                geometry_handler.useTextureColor(UUID);
            }
        } else {
            if (!texture_file.empty()) {
                geometry_handler.useTextureColor(UUID);
            }
        }

        geometry_handler.setColor(UUID, color);
    }

    primitiveColorsNeedUpdate = false;
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
