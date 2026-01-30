/** \file "OptiX6Backend.h" OptiX 6.5 ray tracing backend implementation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef OPTIX6_BACKEND_H
#define OPTIX6_BACKEND_H

#include "RayTracingBackend.h"

// OptiX 6.5 includes
#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_vector_functions.h>
#include <optixu/optixu_vector_types.h>

namespace helios {

    // Ray type enumeration
    enum RayType { RAYTYPE_DIRECT = 0, RAYTYPE_DIFFUSE = 1, RAYTYPE_CAMERA = 2, RAYTYPE_PIXEL_LABEL = 3 };

// OptiX error checking macros
#define RT_CHECK_ERROR(func)                                                                                                                                                                                                                             \
    do {                                                                                                                                                                                                                                                 \
        RTresult code = func;                                                                                                                                                                                                                            \
        if (code != RT_SUCCESS)                                                                                                                                                                                                                          \
            sutilHandleError(OptiX_Context, code, __FILE__, __LINE__);                                                                                                                                                                                   \
    } while (0)

#define RT_CHECK_ERROR_NOEXIT(func)                                                                                                                                                                                                                      \
    do {                                                                                                                                                                                                                                                 \
        RTresult code = func;                                                                                                                                                                                                                            \
        if (code != RT_SUCCESS) {                                                                                                                                                                                                                        \
            const char *message;                                                                                                                                                                                                                         \
            rtContextGetErrorString(OptiX_Context, code, &message);                                                                                                                                                                                      \
            std::cerr << "WARNING (OptiX cleanup): " << message << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;                                                                                                                              \
        }                                                                                                                                                                                                                                                \
    } while (0)

    /**
     * @brief OptiX 6.5 implementation of the ray tracing backend
     *
     * This backend wraps the existing OptiX 6.5 code from RadiationModel,
     * providing the same functionality through the RayTracingBackend interface.
     */
    class OptiX6Backend : public RayTracingBackend {
    public:
        OptiX6Backend();
        ~OptiX6Backend() override;

        // Lifecycle
        void initialize() override;
        void shutdown() override;

        // Geometry management
        void updateGeometry(const RayTracingGeometry &geometry) override;
        void buildAccelerationStructure() override;

        // Material/optical properties
        void updateMaterials(const RayTracingMaterial &materials) override;

        // Radiation sources
        void updateSources(const std::vector<RayTracingSource> &sources) override;

        // Diffuse/sky radiation
        void updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm, const std::vector<float> &sky_energy) override;

        void updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction, const std::vector<float> &solar_disk_radiance, float solar_disk_cos_angle) override;

        // Ray launching
        void launchDirectRays(const RayTracingLaunchParams &params) override;
        void launchDiffuseRays(const RayTracingLaunchParams &params) override;
        void launchCameraRays(const RayTracingLaunchParams &params) override;
        void launchPixelLabelRays(const RayTracingLaunchParams &params) override;

        // Results retrieval
        void getRadiationResults(RayTracingResults &results) override;
        void getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels, std::vector<float> &pixel_depths, uint camera_id, const helios::int2 &resolution) override;

        // Buffer utilities
        void zeroRadiationBuffers(size_t launch_band_count) override;
        void zeroScatterBuffers() override;
        void zeroCameraPixelBuffers(const helios::int2 &resolution) override;
        void copyScatterToRadiation() override;
        void uploadRadiationOut(const std::vector<float> &radiation_out_top, const std::vector<float> &radiation_out_bottom) override;
        void uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam, const std::vector<float> &scatter_bottom_cam) override;
        void zeroCameraScatterBuffers(size_t launch_band_count) override;
        void uploadSourceFluxes(const std::vector<float> &fluxes) override;

        // Diagnostics
        void queryGPUMemory() const override;
        std::string getBackendName() const override;

    private:
        // OptiX context and core objects
        RTcontext OptiX_Context;

        // Ray generation programs (4 ray types)
        RTprogram direct_raygen;
        RTprogram diffuse_raygen;
        RTprogram camera_raygen;
        RTprogram pixel_label_raygen;

        // Geometry and acceleration structures
        RTgeometrygroup base_geometry_group;
        RTacceleration base_acceleration;
        RTgroup top_level_group;
        RTacceleration top_level_acceleration;
        RTtransform transform;

        // Geometry objects (6 primitive types)
        RTgeometry patch_geometry;
        RTgeometry triangle_geometry;
        RTgeometry disk_geometry;
        RTgeometry tile_geometry;
        RTgeometry voxel_geometry;
        RTgeometry bbox_geometry;

        // Geometry instances
        RTgeometryinstance patch_geometryinstance;
        RTgeometryinstance triangle_geometryinstance;
        RTgeometryinstance disk_geometryinstance;
        RTgeometryinstance tile_geometryinstance;
        RTgeometryinstance voxel_geometryinstance;
        RTgeometryinstance bbox_geometryinstance;

        // Materials (one per primitive type)
        RTmaterial patch_material;
        RTmaterial triangle_material;
        RTmaterial disk_material;
        RTmaterial tile_material;
        RTmaterial voxel_material;
        RTmaterial bbox_material;

        // Intersection programs
        RTprogram rectangle_intersect;
        RTprogram rectangle_bounds;
        RTprogram triangle_intersect;
        RTprogram triangle_bounds;
        RTprogram disk_intersect;
        RTprogram disk_bounds;
        RTprogram tile_intersect;
        RTprogram tile_bounds;
        RTprogram voxel_intersect;
        RTprogram voxel_bounds;
        RTprogram bbox_intersect;
        RTprogram bbox_bounds;

        // Hit programs
        RTprogram closest_hit_direct;
        RTprogram closest_hit_diffuse;
        RTprogram closest_hit_camera;
        RTprogram closest_hit_pixel_label;
        RTprogram miss_direct;
        RTprogram miss_diffuse;
        RTprogram miss_camera;

        // Buffers: Geometry/Topology (17 buffers)
        RTbuffer patch_vertices_RTbuffer;
        RTvariable patch_vertices_RTvariable;
        RTbuffer triangle_vertices_RTbuffer;
        RTvariable triangle_vertices_RTvariable;
        RTbuffer disk_centers_RTbuffer;
        RTvariable disk_centers_RTvariable;
        RTbuffer disk_radii_RTbuffer;
        RTvariable disk_radii_RTvariable;
        RTbuffer disk_normals_RTbuffer;
        RTvariable disk_normals_RTvariable;
        RTbuffer tile_vertices_RTbuffer;
        RTvariable tile_vertices_RTvariable;
        RTbuffer voxel_vertices_RTbuffer;
        RTvariable voxel_vertices_RTvariable;
        RTbuffer bbox_vertices_RTbuffer;
        RTvariable bbox_vertices_RTvariable;
        RTbuffer object_subdivisions_RTbuffer;
        RTvariable object_subdivisions_RTvariable;
        RTbuffer primitive_type_RTbuffer;
        RTvariable primitive_type_RTvariable;
        RTbuffer primitive_solid_fraction_RTbuffer;
        RTvariable primitive_solid_fraction_RTvariable;

        // Buffers: UUIDs/Mapping (12 buffers)
        RTbuffer patch_UUID_RTbuffer;
        RTvariable patch_UUID_RTvariable;
        RTbuffer triangle_UUID_RTbuffer;
        RTvariable triangle_UUID_RTvariable;
        RTbuffer disk_UUID_RTbuffer;
        RTvariable disk_UUID_RTvariable;
        RTbuffer tile_UUID_RTbuffer;
        RTvariable tile_UUID_RTvariable;
        RTbuffer voxel_UUID_RTbuffer;
        RTvariable voxel_UUID_RTvariable;
        RTbuffer bbox_UUID_RTbuffer;
        RTvariable bbox_UUID_RTvariable;
        RTbuffer objectID_RTbuffer;
        RTvariable objectID_RTvariable;
        RTbuffer primitiveID_RTbuffer;
        RTvariable primitiveID_RTvariable;
        RTbuffer primitive_positions_RTbuffer;
        RTvariable primitive_positions_RTvariable;
        RTbuffer twosided_flag_RTbuffer;
        RTvariable twosided_flag_RTvariable;

        // Buffers: Material Properties (8 buffers)
        RTbuffer rho_RTbuffer;
        RTvariable rho_RTvariable;
        RTbuffer tau_RTbuffer;
        RTvariable tau_RTvariable;
        RTbuffer rho_cam_RTbuffer;
        RTvariable rho_cam_RTvariable;
        RTbuffer tau_cam_RTbuffer;
        RTvariable tau_cam_RTvariable;
        RTbuffer specular_exponent_RTbuffer;
        RTvariable specular_exponent_RTvariable;
        RTbuffer specular_scale_RTbuffer;
        RTvariable specular_scale_RTvariable;
        RTbuffer transform_matrix_RTbuffer;
        RTvariable transform_matrix_RTvariable;

        // Buffers: Radiation Energy (10 buffers)
        RTbuffer radiation_in_RTbuffer;
        RTvariable radiation_in_RTvariable;
        RTbuffer radiation_out_top_RTbuffer;
        RTvariable radiation_out_top_RTvariable;
        RTbuffer radiation_out_bottom_RTbuffer;
        RTvariable radiation_out_bottom_RTvariable;
        RTbuffer Rsky_RTbuffer;
        RTvariable Rsky_RTvariable;
        RTbuffer radiation_specular_RTbuffer;
        RTvariable radiation_specular_RTvariable;
        RTbuffer scatter_buff_top_RTbuffer;
        RTvariable scatter_buff_top_RTvariable;
        RTbuffer scatter_buff_bottom_RTbuffer;
        RTvariable scatter_buff_bottom_RTvariable;
        RTbuffer radiation_in_camera_RTbuffer;
        RTvariable radiation_in_camera_RTvariable;
        RTbuffer scatter_buff_top_cam_RTbuffer;
        RTvariable scatter_buff_top_cam_RTvariable;
        RTbuffer scatter_buff_bottom_cam_RTbuffer;
        RTvariable scatter_buff_bottom_cam_RTvariable;

        // Buffers: Camera (3 buffers)
        RTbuffer camera_pixel_label_RTbuffer;
        RTvariable camera_pixel_label_RTvariable;
        RTbuffer camera_pixel_depth_RTbuffer;
        RTvariable camera_pixel_depth_RTvariable;

        // Buffers: Diffuse/Sky (5 buffers)
        RTbuffer diffuse_flux_RTbuffer;
        RTvariable diffuse_flux_RTvariable;
        RTbuffer diffuse_extinction_RTbuffer;
        RTvariable diffuse_extinction_RTvariable;
        RTbuffer diffuse_peak_dir_RTbuffer;
        RTvariable diffuse_peak_dir_RTvariable;
        RTbuffer diffuse_dist_norm_RTbuffer;
        RTvariable diffuse_dist_norm_RTvariable;
        RTbuffer sky_radiance_params_RTbuffer;
        RTvariable sky_radiance_params_RTvariable;
        RTbuffer camera_sky_radiance_RTbuffer;
        RTvariable camera_sky_radiance_RTvariable;
        RTbuffer solar_disk_radiance_RTbuffer;
        RTvariable solar_disk_radiance_RTvariable;

        // Buffers: Sources/Texture (9 buffers)
        RTbuffer source_positions_RTbuffer;
        RTvariable source_positions_RTvariable;
        RTbuffer source_types_RTbuffer;
        RTvariable source_types_RTvariable;
        RTbuffer source_fluxes_RTbuffer;
        RTvariable source_fluxes_RTvariable;
        RTbuffer source_fluxes_cam_RTbuffer;
        RTvariable source_fluxes_cam_RTvariable;
        RTbuffer source_widths_RTbuffer;
        RTvariable source_widths_RTvariable;
        RTbuffer source_rotations_RTbuffer;
        RTvariable source_rotations_RTvariable;
        RTbuffer band_launch_flag_RTbuffer;
        RTvariable band_launch_flag_RTvariable;
        RTbuffer max_scatters_RTbuffer;
        RTvariable max_scatters_RTvariable;

        // Buffers: Texture/Masking (5 buffers)
        RTbuffer maskdata_RTbuffer;
        RTvariable maskdata_RTvariable;
        RTbuffer masksize_RTbuffer;
        RTvariable masksize_RTvariable;
        RTbuffer maskID_RTbuffer;
        RTvariable maskID_RTvariable;
        RTbuffer uvdata_RTbuffer;
        RTvariable uvdata_RTvariable;
        RTbuffer uvID_RTbuffer;
        RTvariable uvID_RTvariable;

        // RT Variables (non-buffer parameters)
        RTvariable direct_ray_type_RTvariable;
        RTvariable diffuse_ray_type_RTvariable;
        RTvariable camera_ray_type_RTvariable;
        RTvariable pixel_label_ray_type_RTvariable;
        RTvariable random_seed_RTvariable;
        RTvariable launch_offset_RTvariable;
        RTvariable launch_face_RTvariable;
        RTvariable Nprimitives_RTvariable;
        RTvariable bbox_UUID_base_RTvariable;
        RTvariable Nsources_RTvariable;
        RTvariable Nbands_global_RTvariable;
        RTvariable Nbands_launch_RTvariable;
        RTvariable Ncameras_RTvariable;
        RTvariable periodic_flag_RTvariable;
        RTvariable sun_direction_RTvariable;
        RTvariable solar_disk_cos_angle_RTvariable;
        RTvariable camera_position_RTvariable;
        RTvariable camera_direction_RTvariable;
        RTvariable camera_lens_diameter_RTvariable;
        RTvariable camera_focal_length_RTvariable;
        RTvariable FOV_aspect_ratio_RTvariable;
        RTvariable camera_HFOV_RTvariable;
        RTvariable camera_resolution_RTvariable;
        RTvariable camera_viewplane_length_RTvariable;
        RTvariable camera_pixel_solid_angle_RTvariable;
        RTvariable camera_pixel_offset_x_RTvariable;
        RTvariable camera_pixel_offset_y_RTvariable;
        RTvariable camera_ID_RTvariable;
        RTvariable camera_resolution_full_RTvariable;
        RTvariable specular_reflection_enabled_RTvariable;
        RTvariable scattering_iteration_RTvariable;

        // Helper methods for buffer management (will be moved from RadiationModel)
        void addBuffer(const char *name, RTbuffer &buffer, RTvariable &variable, RTbuffertype type, RTformat format, size_t dimension);
        void zeroBuffer1D(RTbuffer &buffer, size_t bsize);
        void zeroBuffer2D(RTbuffer &buffer, const helios::int2 &bsize);
        void initializeBuffer1Df(RTbuffer &buffer, const std::vector<float> &array);
        void initializeBuffer1Dui(RTbuffer &buffer, const std::vector<uint> &array);
        void initializeBuffer1Di(RTbuffer &buffer, const std::vector<int> &array);
        void initializeBuffer1Dchar(RTbuffer &buffer, const std::vector<char> &array);
        void initializeBuffer1Dbool(RTbuffer &buffer, const std::vector<bool> &array);
        void initializeBuffer1Dfloat2(RTbuffer &buffer, const std::vector<helios::vec2> &array);
        void initializeBuffer1Dfloat3(RTbuffer &buffer, const std::vector<helios::vec3> &array);
        void initializeBuffer1Dfloat4(RTbuffer &buffer, const std::vector<helios::vec4> &array);
        void initializeBuffer1Dint2(RTbuffer &buffer, const std::vector<helios::int2> &array);
        void initializeBuffer2Df(RTbuffer &buffer, const std::vector<std::vector<float>> &array);
        void initializeBuffer2Dui(RTbuffer &buffer, const std::vector<std::vector<uint>> &array);
        void initializeBuffer2Di(RTbuffer &buffer, const std::vector<std::vector<int>> &array);
        void initializeBuffer2Dfloat2(RTbuffer &buffer, const std::vector<std::vector<helios::vec2>> &array);
        void initializeBuffer2Dfloat3(RTbuffer &buffer, const std::vector<std::vector<helios::vec3>> &array);
        void initializeBuffer2Dfloat3(RTbuffer &buffer, const std::vector<std::vector<optix::float3>> &array);
        void initializeBuffer3Dbool(RTbuffer &buffer, const std::vector<std::vector<std::vector<bool>>> &array);
        void copyBuffer1D(RTbuffer &source, RTbuffer &dest);
        std::vector<float> getOptiXbufferData(RTbuffer buffer);
        std::vector<uint> getOptiXbufferData_ui(RTbuffer buffer);

        // Conversion methods: backend-agnostic → OptiX
        void geometryToBuffers(const RayTracingGeometry &geometry);
        void materialsToBuffers(const RayTracingMaterial &materials);
        void sourcesToBuffers(const std::vector<RayTracingSource> &sources);
        void diffuseToBuffers(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm, const std::vector<float> &sky_energy);
        void skyModelToBuffers(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction, const std::vector<float> &solar_disk_radiance, float solar_disk_cos_angle);
        void launchParamsToVariables(const RayTracingLaunchParams &params);

        // Extraction methods: OptiX → backend-agnostic
        void buffersToResults(RayTracingResults &results);

        // Internal state tracking
        bool is_initialized = false;
        size_t current_primitive_count = 0;
        size_t current_patch_count = 0;
        size_t current_triangle_count = 0;
        size_t current_disk_count = 0;
        size_t current_tile_count = 0;
        size_t current_voxel_count = 0;
        size_t current_bbox_count = 0;
        size_t current_source_count = 0;
        size_t current_band_count = 0;
        size_t current_camera_count = 0;
    };

} // namespace helios

#endif // OPTIX6_BACKEND_H
