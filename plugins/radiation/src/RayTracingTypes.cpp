/** \file "RayTracingTypes.cpp" Implementation of validation for ray tracing types.

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
        // Shared per-primitive arrays are sized to primitive_count only.
        // Bbox data is stored separately in bboxes.UUIDs / bboxes.vertices and is NOT
        // included in the shared arrays (see RadiationModel::buildGeometryData()).

        if (transform_matrices.size() != primitive_count * 16) {
            helios_runtime_error("RayTracingGeometry validation failed: transform_matrices.size()=" + std::to_string(transform_matrices.size()) + " != primitive_count*16=" + std::to_string(primitive_count * 16));
        }

        if (primitive_types.size() != primitive_count) {
            helios_runtime_error("RayTracingGeometry validation failed: primitive_types.size()=" + std::to_string(primitive_types.size()) + " != primitive_count=" + std::to_string(primitive_count));
        }

        if (object_subdivisions.size() != primitive_count) {
            helios_runtime_error("RayTracingGeometry validation failed: object_subdivisions.size()=" + std::to_string(object_subdivisions.size()) + " != primitive_count=" + std::to_string(primitive_count));
        }

        if (twosided_flags.size() != primitive_count) {
            helios_runtime_error("RayTracingGeometry validation failed: twosided_flags.size()=" + std::to_string(twosided_flags.size()) + " != primitive_count=" + std::to_string(primitive_count));
        }

        if (solid_fractions.size() != primitive_count) {
            helios_runtime_error("RayTracingGeometry validation failed: solid_fractions.size()=" + std::to_string(solid_fractions.size()) + " != primitive_count=" + std::to_string(primitive_count));
        }

        // ========== Object ID Validation ==========

        if (object_IDs.size() != primitive_count) {
            helios_runtime_error("RayTracingGeometry validation failed: object_IDs.size()=" + std::to_string(object_IDs.size()) + " != primitive_count=" + std::to_string(primitive_count));
        }

        if (primitive_IDs.size() != primitive_count) {
            helios_runtime_error("RayTracingGeometry validation failed: primitive_IDs.size()=" + std::to_string(primitive_IDs.size()) + " != primitive_count=" + std::to_string(primitive_count) +
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

    std::vector<float> computeSurfaceLayerAlbedoTable(float refractive_index, float roughness, uint32_t table_size) {
        if (refractive_index < 1.f || roughness <= 0.f || table_size < 2) {
            helios_runtime_error("ERROR (computeSurfaceLayerAlbedoTable): The refractive index must be at least 1, the roughness positive and the table at least 2 long (got n = " + std::to_string(refractive_index) +
                                 ", alpha = " + std::to_string(roughness) + ", " + std::to_string(table_size) + " values).");
        }
        const double n = refractive_index;
        const double alpha2 = double(roughness) * double(roughness);

        // Unpolarized Fresnel reflectance of the interface for the cosine c of the angle to the microfacet normal
        auto fresnel = [n](double c) {
            c = std::clamp(c, 0.0, 1.0);
            const double sin_t2 = (1.0 - c * c) / (n * n);
            const double cos_t = std::sqrt(std::max(0.0, 1.0 - sin_t2));
            const double rs = (c - n * cos_t) / (c + n * cos_t);
            const double rp = (n * c - cos_t) / (n * c + cos_t);
            return 0.5 * (rs * rs + rp * rp);
        };
        // Smith Lambda of the GGX distribution
        auto lambda = [alpha2](double mu) {
            const double tan2 = (1.0 - mu * mu) / std::max(mu * mu, 1e-12);
            return 0.5 * (-1.0 + std::sqrt(1.0 + alpha2 * tan2));
        };

        // The integral is taken over the microfacet normals h rather than the incident directions, since the lobe is only about
        // 2 alpha wide: with the incident direction l = 2 (v.h) h - v, d(omega_l) = 4 (v.h) d(omega_h), and with
        // tan^2(theta_h) = alpha^2 xi / (1 - xi), D(h) cos(theta_h) d(omega_h) = d(xi) d(phi) / (2 pi), so that
        //   a(mu_o) = 1/(2 pi) int int F(v.h) G2(l, v) (v.h) / (mu_o cos(theta_h)) d(xi) d(phi)   over the h with l above the surface.
        // Gauss-Legendre nodes in xi and equally spaced azimuths resolve the lobe at every roughness and viewing angle.
        constexpr int n_xi = 256;
        constexpr int n_phi = 256;
        std::vector<double> xi_node(n_xi), xi_weight(n_xi);
        for (int i = 0; i < n_xi; i++) {
            // Newton iteration on the Legendre polynomial P_n for the i-th root on [-1, 1]
            double x = std::cos(M_PI * (i + 0.75) / (n_xi + 0.5));
            double dp = 0.0;
            for (int iteration = 0; iteration < 100; iteration++) {
                double p0 = 1.0, p1 = x;
                for (int k = 2; k <= n_xi; k++) {
                    const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                    p0 = p1;
                    p1 = p2;
                }
                dp = n_xi * (x * p1 - p0) / (x * x - 1.0);
                const double dx = p1 / dp;
                x -= dx;
                if (std::fabs(dx) < 1e-15) {
                    break;
                }
            }
            xi_node.at(i) = 0.5 * (x + 1.0);
            xi_weight.at(i) = 1.0 / ((1.0 - x * x) * dp * dp); // half the weight on [-1, 1], for the interval [0, 1]
        }

        std::vector<float> table(table_size + 1);
        for (uint32_t m = 0; m < table_size; m++) {
            // The integrand vanishes as mu_out -> 0 only in the limit, so the first entry is evaluated just above grazing
            const double mu_out = std::max(double(m) / double(table_size - 1), 1e-3);
            const double sin_out = std::sqrt(1.0 - mu_out * mu_out);
            const double lambda_out = lambda(mu_out);
            double albedo = 0.0;
            for (int i = 0; i < n_xi; i++) {
                const double tan2_h = alpha2 * xi_node.at(i) / (1.0 - xi_node.at(i));
                const double mu_h = 1.0 / std::sqrt(1.0 + tan2_h);
                const double sin_h = std::sqrt(1.0 - mu_h * mu_h);
                double azimuth_sum = 0.0;
                for (int j = 0; j < n_phi; j++) {
                    const double phi = 2.0 * M_PI * (j + 0.5) / n_phi;
                    // Viewing direction v = (sin_out, 0, mu_out), microfacet normal h = (sin_h cos phi, sin_h sin phi, mu_h)
                    const double v_dot_h = sin_out * sin_h * std::cos(phi) + mu_out * mu_h;
                    const double mu_in = 2.0 * v_dot_h * mu_h - mu_out; // z of the incident direction l = 2 (v.h) h - v
                    if (v_dot_h <= 0.0 || mu_in <= 0.0) {
                        continue;
                    }
                    const double G2 = 1.0 / (1.0 + lambda(mu_in) + lambda_out);
                    azimuth_sum += fresnel(v_dot_h) * G2 * v_dot_h / (mu_out * mu_h);
                }
                albedo += xi_weight.at(i) * azimuth_sum / n_phi;
            }
            table.at(m) = float(std::min(albedo, 1.0));
        }

        // Hemispherical mean by the trapezoidal rule on the table
        double mean = 0.0;
        for (uint32_t m = 0; m + 1 < table_size; m++) {
            const double mu0 = double(m) / double(table_size - 1);
            const double mu1 = double(m + 1) / double(table_size - 1);
            mean += (table.at(m) * mu0 + table.at(m + 1) * mu1) * (mu1 - mu0);
        }
        table.at(table_size) = float(mean);
        return table;
    }

} // namespace helios
