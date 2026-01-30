/** \file "RayTracingTypes.cpp" Implementation of validation for ray tracing types.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RayTracingTypes.h"
#include "global.h"
#include <algorithm>
#include <climits>

namespace helios {

    void RayTracingGeometry::validate() const {
#ifndef NDEBUG

        // ========== UUID / Position Validation ==========

        if (primitive_UUIDs.size() != primitive_count) {
            helios_runtime_error("RayTracingGeometry validation failed: primitive_UUIDs.size()=" + std::to_string(primitive_UUIDs.size()) + " != primitive_count=" + std::to_string(primitive_count));
        }

        // UUID lookup table validation
        // NOTE: primitive_positions now includes bbox UUIDs (safe due to bbox_UUID_base = max_UUID + 1)
        if (!primitive_UUIDs.empty()) {
            uint max_uuid = *std::max_element(primitive_UUIDs.begin(), primitive_UUIDs.end());

            // Include bbox UUIDs in expected size if present
            if (bbox_count > 0) {
                uint bbox_max_uuid = bbox_UUID_base + bbox_count - 1;
                if (bbox_max_uuid > max_uuid) {
                    max_uuid = bbox_max_uuid;
                }
            }

            size_t expected_size = max_uuid + 1;

            if (primitive_positions.size() != expected_size) {
                helios_runtime_error("RayTracingGeometry validation failed: primitive_positions.size()=" + std::to_string(primitive_positions.size()) + " != max_UUID+1=" + std::to_string(expected_size) + " (max_UUID=" + std::to_string(max_uuid) +
                                     ")");
            }

            // Validate UUID→position mapping consistency
            for (size_t pos = 0; pos < primitive_UUIDs.size(); pos++) {
                uint UUID = primitive_UUIDs[pos];

                if (UUID >= primitive_positions.size()) {
                    helios_runtime_error("RayTracingGeometry validation failed: primitive_UUIDs[" + std::to_string(pos) + "]=" + std::to_string(UUID) + " >= primitive_positions.size()=" + std::to_string(primitive_positions.size()));
                }

                uint mapped_pos = primitive_positions[UUID];
                if (mapped_pos == UINT_MAX) {
                    helios_runtime_error("RayTracingGeometry validation failed: primitive_positions[" + std::to_string(UUID) + "] == UINT_MAX (should map to position " + std::to_string(pos) + ")");
                }

                if (mapped_pos != pos) {
                    helios_runtime_error("RayTracingGeometry validation failed: primitive_positions[" + std::to_string(UUID) + "]=" + std::to_string(mapped_pos) + " but UUID is at position " + std::to_string(pos) +
                                         " (bidirectional mapping inconsistent)");
                }
            }
        }

        // ========== Per-Primitive Buffer Sizing ==========

        // Calculate expected buffer size including bboxes
        size_t expected_types_size = primitive_count + bbox_count;

        // transform_matrices can be larger than primitive_count*16 if bboxes are present
        if (transform_matrices.size() != expected_types_size * 16) {
            helios_runtime_error("RayTracingGeometry validation failed: transform_matrices.size()=" + std::to_string(transform_matrices.size()) + " != (primitive_count+bbox_count)*16=" + std::to_string(expected_types_size * 16));
        }

        // primitive_types can be larger than primitive_count if bboxes are present
        if (primitive_types.size() != expected_types_size) {
            helios_runtime_error("RayTracingGeometry validation failed: primitive_types.size()=" + std::to_string(primitive_types.size()) + " != primitive_count+bbox_count=" + std::to_string(expected_types_size));
        }

        // object_subdivisions can be larger than primitive_count if bboxes are present
        if (object_subdivisions.size() != expected_types_size) {
            helios_runtime_error("RayTracingGeometry validation failed: object_subdivisions.size()=" + std::to_string(object_subdivisions.size()) + " != primitive_count+bbox_count=" + std::to_string(expected_types_size));
        }

        // twosided_flags can be larger than primitive_count if bboxes are present
        if (twosided_flags.size() != expected_types_size) {
            helios_runtime_error("RayTracingGeometry validation failed: twosided_flags.size()=" + std::to_string(twosided_flags.size()) + " != primitive_count+bbox_count=" + std::to_string(expected_types_size));
        }

        // solid_fractions can be larger than primitive_count if bboxes are present
        if (solid_fractions.size() != expected_types_size) {
            helios_runtime_error("RayTracingGeometry validation failed: solid_fractions.size()=" + std::to_string(solid_fractions.size()) + " != primitive_count+bbox_count=" + std::to_string(expected_types_size));
        }

        // ========== Object ID Validation ==========

        // object_IDs can be larger than primitive_count if bboxes are present
        if (object_IDs.size() != expected_types_size) {
            helios_runtime_error("RayTracingGeometry validation failed: object_IDs.size()=" + std::to_string(object_IDs.size()) + " != primitive_count+bbox_count=" + std::to_string(expected_types_size));
        }

        // primitive_IDs must include bbox entries when periodic boundaries are enabled
        size_t expected_primitive_IDs_size = primitive_count + bbox_count;
        if (primitive_IDs.size() != expected_primitive_IDs_size) {
            helios_runtime_error("RayTracingGeometry validation failed: primitive_IDs.size()=" + std::to_string(primitive_IDs.size()) + " != primitive_count+bbox_count=" + std::to_string(expected_primitive_IDs_size) +
                                 " (COMMON BUG: Did you size by Nobjects instead of Nprimitives?)");
        }

        // ========== Per-Type Buffer Sizing ==========

        if (patches.UUIDs.size() != patch_count) {
            helios_runtime_error("RayTracingGeometry validation failed: patches.UUIDs.size()=" + std::to_string(patches.UUIDs.size()) + " != patch_count=" + std::to_string(patch_count));
        }

        if (patches.vertices.size() != patch_count * 4) {
            helios_runtime_error("RayTracingGeometry validation failed: patches.vertices.size()=" + std::to_string(patches.vertices.size()) + " != patch_count*4=" + std::to_string(patch_count * 4));
        }

        if (triangles.UUIDs.size() != triangle_count) {
            helios_runtime_error("RayTracingGeometry validation failed: triangles.UUIDs.size()=" + std::to_string(triangles.UUIDs.size()) + " != triangle_count=" + std::to_string(triangle_count));
        }

        if (triangles.vertices.size() != triangle_count * 3) {
            helios_runtime_error("RayTracingGeometry validation failed: triangles.vertices.size()=" + std::to_string(triangles.vertices.size()) + " != triangle_count*3=" + std::to_string(triangle_count * 3));
        }

        if (tiles.UUIDs.size() != tile_count) {
            helios_runtime_error("RayTracingGeometry validation failed: tiles.UUIDs.size()=" + std::to_string(tiles.UUIDs.size()) + " != tile_count=" + std::to_string(tile_count));
        }

        if (tiles.vertices.size() != tile_count * 4) {
            helios_runtime_error("RayTracingGeometry validation failed: tiles.vertices.size()=" + std::to_string(tiles.vertices.size()) + " != tile_count*4=" + std::to_string(tile_count * 4));
        }

        if (voxels.UUIDs.size() != voxel_count) {
            helios_runtime_error("RayTracingGeometry validation failed: voxels.UUIDs.size()=" + std::to_string(voxels.UUIDs.size()) + " != voxel_count=" + std::to_string(voxel_count));
        }

        if (voxels.vertices.size() != voxel_count * 8) {
            helios_runtime_error("RayTracingGeometry validation failed: voxels.vertices.size()=" + std::to_string(voxels.vertices.size()) + " != voxel_count*8=" + std::to_string(voxel_count * 8));
        }

        if (bboxes.UUIDs.size() != bbox_count) {
            helios_runtime_error("RayTracingGeometry validation failed: bboxes.UUIDs.size()=" + std::to_string(bboxes.UUIDs.size()) + " != bbox_count=" + std::to_string(bbox_count));
        }

        if (bboxes.vertices.size() != bbox_count * 4) {
            helios_runtime_error("RayTracingGeometry validation failed: bboxes.vertices.size()=" + std::to_string(bboxes.vertices.size()) + " != bbox_count*4=" + std::to_string(bbox_count * 4));
        }

        if (disk_UUIDs.size() != disk_count) {
            helios_runtime_error("RayTracingGeometry validation failed: disk_UUIDs.size()=" + std::to_string(disk_UUIDs.size()) + " != disk_count=" + std::to_string(disk_count));
        }

        if (disk_centers.size() != disk_count) {
            helios_runtime_error("RayTracingGeometry validation failed: disk_centers.size()=" + std::to_string(disk_centers.size()) + " != disk_count=" + std::to_string(disk_count));
        }

        if (disk_radii.size() != disk_count) {
            helios_runtime_error("RayTracingGeometry validation failed: disk_radii.size()=" + std::to_string(disk_radii.size()) + " != disk_count=" + std::to_string(disk_count));
        }

        if (disk_normals.size() != disk_count) {
            helios_runtime_error("RayTracingGeometry validation failed: disk_normals.size()=" + std::to_string(disk_normals.size()) + " != disk_count=" + std::to_string(disk_count));
        }

        // ========== Subdivision Validation ==========

        for (size_t i = 0; i < object_subdivisions.size(); i++) {
            helios::int2 subdiv = object_subdivisions[i];
            if (subdiv.x < 1 || subdiv.y < 1) {
                helios_runtime_error("RayTracingGeometry validation failed: object_subdivisions[" + std::to_string(i) + "] = (" + std::to_string(subdiv.x) + "," + std::to_string(subdiv.y) + ") has zero or negative subdivision count");
            }
        }

        // Heuristic check: warn if many primitives have subdivisions > (1,1)
        // This often indicates subdivision inheritance bug (subpatches should be (1,1))
        size_t subdivision_anomalies = 0;
        for (size_t i = 0; i < object_subdivisions.size(); i++) {
            if (object_subdivisions[i].x > 1 || object_subdivisions[i].y > 1) {
                subdivision_anomalies++;
            }
        }

        if (subdivision_anomalies > primitive_count / 2) {
            std::cerr << "WARNING [RayTracingGeometry]: " << subdivision_anomalies << " / " << primitive_count << " primitives have subdivisions > (1,1). "
                      << "Check for subdivision inheritance bug - subpatches should have (1,1), "
                      << "only parent geometry should have actual subdivision counts.\n";
        }

#endif // NDEBUG
    }

} // namespace helios
