/** \file "OptiX6Backend.cpp" OptiX 6.5 ray tracing backend implementation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "OptiX6Backend.h"
#include "Context.h"
#include <chrono>

using namespace helios;

// Forward declaration of error handling functions
static void sutilReportError(const char *message);
static void sutilHandleError(RTcontext context, RTresult code, const char *file, int line);

OptiX6Backend::OptiX6Backend() {
    // Constructor - initialization happens in initialize()
}

OptiX6Backend::~OptiX6Backend() {
    if (is_initialized) {
        shutdown();
    }
}

void OptiX6Backend::initialize() {

    if (is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::initialize): Backend already initialized.");
    }

    /* Create OptiX Context */
    RT_CHECK_ERROR(rtContextCreate(&OptiX_Context));
    RT_CHECK_ERROR(rtContextSetPrintEnabled(OptiX_Context, 1));

    /* Set ray type and entry point counts */
    RT_CHECK_ERROR(rtContextSetRayTypeCount(OptiX_Context, 4));
    // ray types: 0=direct, 1=diffuse, 2=camera, 3=pixel_label

    RT_CHECK_ERROR(rtContextSetEntryPointCount(OptiX_Context, 4));
    // entry points: 0=direct_raygen, 1=diffuse_raygen, 2=camera_raygen, 3=pixel_label_raygen

    /* Declare ray type variables */
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "direct_ray_type", &direct_ray_type_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(direct_ray_type_RTvariable, RAYTYPE_DIRECT));
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "diffuse_ray_type", &diffuse_ray_type_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(diffuse_ray_type_RTvariable, RAYTYPE_DIFFUSE));
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_ray_type", &camera_ray_type_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(camera_ray_type_RTvariable, RAYTYPE_CAMERA));
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "pixel_label_ray_type", &pixel_label_ray_type_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(pixel_label_ray_type_RTvariable, RAYTYPE_PIXEL_LABEL));

    /* Load ray generation programs from PTX */
    std::string ptx_path = helios::resolvePluginAsset("radiation", "cuda_compile_ptx_generated_rayGeneration.cu.ptx").string();
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, ptx_path.c_str(), "direct_raygen", &direct_raygen));
    RT_CHECK_ERROR(rtContextSetRayGenerationProgram(OptiX_Context, RAYTYPE_DIRECT, direct_raygen));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, ptx_path.c_str(), "diffuse_raygen", &diffuse_raygen));
    RT_CHECK_ERROR(rtContextSetRayGenerationProgram(OptiX_Context, RAYTYPE_DIFFUSE, diffuse_raygen));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, ptx_path.c_str(), "camera_raygen", &camera_raygen));
    RT_CHECK_ERROR(rtContextSetRayGenerationProgram(OptiX_Context, RAYTYPE_CAMERA, camera_raygen));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, ptx_path.c_str(), "pixel_label_raygen", &pixel_label_raygen));
    RT_CHECK_ERROR(rtContextSetRayGenerationProgram(OptiX_Context, RAYTYPE_PIXEL_LABEL, pixel_label_raygen));

    /* Load hit programs from PTX */
    std::string hit_ptx_path = helios::resolvePluginAsset("radiation", "cuda_compile_ptx_generated_rayHit.cu.ptx").string();
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, hit_ptx_path.c_str(), "closest_hit_direct", &closest_hit_direct));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, hit_ptx_path.c_str(), "closest_hit_diffuse", &closest_hit_diffuse));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, hit_ptx_path.c_str(), "closest_hit_camera", &closest_hit_camera));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, hit_ptx_path.c_str(), "closest_hit_pixel_label", &closest_hit_pixel_label));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, hit_ptx_path.c_str(), "miss_direct", &miss_direct));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, hit_ptx_path.c_str(), "miss_diffuse", &miss_diffuse));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, hit_ptx_path.c_str(), "miss_camera", &miss_camera));

    /* Set miss programs */
    RT_CHECK_ERROR(rtContextSetMissProgram(OptiX_Context, RAYTYPE_DIRECT, miss_direct));
    RT_CHECK_ERROR(rtContextSetMissProgram(OptiX_Context, RAYTYPE_DIFFUSE, miss_diffuse));
    RT_CHECK_ERROR(rtContextSetMissProgram(OptiX_Context, RAYTYPE_CAMERA, miss_camera));

    /* Load intersection programs from PTX */
    std::string intersect_ptx_path = helios::resolvePluginAsset("radiation", "cuda_compile_ptx_generated_primitiveIntersection.cu.ptx").string();

    // Patch (rectangle) programs
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "rectangle_intersect", &rectangle_intersect));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "rectangle_bounds", &rectangle_bounds));

    // Triangle programs
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "triangle_intersect", &triangle_intersect));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "triangle_bounds", &triangle_bounds));

    // Disk programs
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "disk_intersect", &disk_intersect));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "disk_bounds", &disk_bounds));

    // Tile programs
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "tile_intersect", &tile_intersect));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "tile_bounds", &tile_bounds));

    // Voxel programs
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "voxel_intersect", &voxel_intersect));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "voxel_bounds", &voxel_bounds));

    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "bbox_intersect", &bbox_intersect));
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, intersect_ptx_path.c_str(), "bbox_bounds", &bbox_bounds));

    /* Create geometry objects for each primitive type */

    // Patch geometry
    RT_CHECK_ERROR(rtGeometryCreate(OptiX_Context, &patch_geometry));
    RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(patch_geometry, rectangle_bounds));
    RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(patch_geometry, rectangle_intersect));

    // Triangle geometry
    RT_CHECK_ERROR(rtGeometryCreate(OptiX_Context, &triangle_geometry));
    RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(triangle_geometry, triangle_bounds));
    RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(triangle_geometry, triangle_intersect));

    // Disk geometry
    RT_CHECK_ERROR(rtGeometryCreate(OptiX_Context, &disk_geometry));
    RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(disk_geometry, disk_bounds));
    RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(disk_geometry, disk_intersect));

    // Tile geometry
    RT_CHECK_ERROR(rtGeometryCreate(OptiX_Context, &tile_geometry));
    RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(tile_geometry, tile_bounds));
    RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(tile_geometry, tile_intersect));

    // Voxel geometry
    RT_CHECK_ERROR(rtGeometryCreate(OptiX_Context, &voxel_geometry));
    RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(voxel_geometry, voxel_bounds));
    RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(voxel_geometry, voxel_intersect));

    // Bbox geometry
    RT_CHECK_ERROR(rtGeometryCreate(OptiX_Context, &bbox_geometry));
    RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(bbox_geometry, bbox_bounds));
    RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(bbox_geometry, bbox_intersect));

    /* Create materials for each primitive type */

    // Patch material
    RT_CHECK_ERROR(rtMaterialCreate(OptiX_Context, &patch_material));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(patch_material, RAYTYPE_DIRECT, closest_hit_direct));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(patch_material, RAYTYPE_DIFFUSE, closest_hit_diffuse));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(patch_material, RAYTYPE_CAMERA, closest_hit_camera));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(patch_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label));

    // Triangle material
    RT_CHECK_ERROR(rtMaterialCreate(OptiX_Context, &triangle_material));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(triangle_material, RAYTYPE_DIRECT, closest_hit_direct));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(triangle_material, RAYTYPE_DIFFUSE, closest_hit_diffuse));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(triangle_material, RAYTYPE_CAMERA, closest_hit_camera));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(triangle_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label));

    // Disk material
    RT_CHECK_ERROR(rtMaterialCreate(OptiX_Context, &disk_material));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(disk_material, RAYTYPE_DIRECT, closest_hit_direct));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(disk_material, RAYTYPE_DIFFUSE, closest_hit_diffuse));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(disk_material, RAYTYPE_CAMERA, closest_hit_camera));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(disk_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label));

    // Tile material
    RT_CHECK_ERROR(rtMaterialCreate(OptiX_Context, &tile_material));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(tile_material, RAYTYPE_DIRECT, closest_hit_direct));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(tile_material, RAYTYPE_DIFFUSE, closest_hit_diffuse));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(tile_material, RAYTYPE_CAMERA, closest_hit_camera));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(tile_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label));

    // Voxel material
    RT_CHECK_ERROR(rtMaterialCreate(OptiX_Context, &voxel_material));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(voxel_material, RAYTYPE_DIRECT, closest_hit_direct));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(voxel_material, RAYTYPE_DIFFUSE, closest_hit_diffuse));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(voxel_material, RAYTYPE_CAMERA, closest_hit_camera));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(voxel_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label));

    // Bbox material
    RT_CHECK_ERROR(rtMaterialCreate(OptiX_Context, &bbox_material));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(bbox_material, RAYTYPE_DIRECT, closest_hit_direct));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(bbox_material, RAYTYPE_DIFFUSE, closest_hit_diffuse));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(bbox_material, RAYTYPE_CAMERA, closest_hit_camera));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(bbox_material, RAYTYPE_PIXEL_LABEL, closest_hit_pixel_label));

    /* Create OptiX scene graph structure */

    // Create top level group
    RT_CHECK_ERROR(rtGroupCreate(OptiX_Context, &top_level_group));
    RT_CHECK_ERROR(rtGroupSetChildCount(top_level_group, 1));

    // Create top level acceleration (NoAccel for minimal overhead)
    RT_CHECK_ERROR(rtAccelerationCreate(OptiX_Context, &top_level_acceleration));
    RT_CHECK_ERROR(rtAccelerationSetBuilder(top_level_acceleration, "NoAccel"));
    RT_CHECK_ERROR(rtAccelerationSetTraverser(top_level_acceleration, "NoAccel"));
    RT_CHECK_ERROR(rtGroupSetAcceleration(top_level_group, top_level_acceleration));
    RT_CHECK_ERROR(rtAccelerationMarkDirty(top_level_acceleration));

    // Create transform node (identity matrix)
    RT_CHECK_ERROR(rtTransformCreate(OptiX_Context, &transform));
    float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    RT_CHECK_ERROR(rtTransformSetMatrix(transform, 0, identity, nullptr));
    RT_CHECK_ERROR(rtGroupSetChild(top_level_group, 0, transform));

    // Create geometry group
    RT_CHECK_ERROR(rtGeometryGroupCreate(OptiX_Context, &base_geometry_group));
    RT_CHECK_ERROR(rtGeometryGroupSetChildCount(base_geometry_group, 6)); // 6 primitive types
    RT_CHECK_ERROR(rtTransformSetChild(transform, base_geometry_group));

    // Create geometry acceleration (Trbvh for fast BVH building)
    RT_CHECK_ERROR(rtAccelerationCreate(OptiX_Context, &base_acceleration));
    RT_CHECK_ERROR(rtAccelerationSetBuilder(base_acceleration, "Trbvh"));
    RT_CHECK_ERROR(rtAccelerationSetTraverser(base_acceleration, "Bvh"));
    RT_CHECK_ERROR(rtGeometryGroupSetAcceleration(base_geometry_group, base_acceleration));
    RT_CHECK_ERROR(rtAccelerationMarkDirty(base_acceleration));

    // Create geometry instances
    RT_CHECK_ERROR(rtGeometryInstanceCreate(OptiX_Context, &patch_geometryinstance));
    RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(patch_geometryinstance, patch_geometry));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(patch_geometryinstance, 1));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(patch_geometryinstance, 0, patch_material));
    RT_CHECK_ERROR(rtGeometryGroupSetChild(base_geometry_group, 0, patch_geometryinstance));

    RT_CHECK_ERROR(rtGeometryInstanceCreate(OptiX_Context, &triangle_geometryinstance));
    RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(triangle_geometryinstance, triangle_geometry));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(triangle_geometryinstance, 1));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(triangle_geometryinstance, 0, triangle_material));
    RT_CHECK_ERROR(rtGeometryGroupSetChild(base_geometry_group, 1, triangle_geometryinstance));

    RT_CHECK_ERROR(rtGeometryInstanceCreate(OptiX_Context, &disk_geometryinstance));
    RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(disk_geometryinstance, disk_geometry));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(disk_geometryinstance, 1));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(disk_geometryinstance, 0, disk_material));
    RT_CHECK_ERROR(rtGeometryGroupSetChild(base_geometry_group, 2, disk_geometryinstance));

    RT_CHECK_ERROR(rtGeometryInstanceCreate(OptiX_Context, &tile_geometryinstance));
    RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(tile_geometryinstance, tile_geometry));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(tile_geometryinstance, 1));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(tile_geometryinstance, 0, tile_material));
    RT_CHECK_ERROR(rtGeometryGroupSetChild(base_geometry_group, 3, tile_geometryinstance));

    RT_CHECK_ERROR(rtGeometryInstanceCreate(OptiX_Context, &voxel_geometryinstance));
    RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(voxel_geometryinstance, voxel_geometry));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(voxel_geometryinstance, 1));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(voxel_geometryinstance, 0, voxel_material));
    RT_CHECK_ERROR(rtGeometryGroupSetChild(base_geometry_group, 4, voxel_geometryinstance));

    RT_CHECK_ERROR(rtGeometryInstanceCreate(OptiX_Context, &bbox_geometryinstance));
    RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(bbox_geometryinstance, bbox_geometry));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(bbox_geometryinstance, 1));
    RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(bbox_geometryinstance, 0, bbox_material));
    RT_CHECK_ERROR(rtGeometryGroupSetChild(base_geometry_group, 5, bbox_geometryinstance));

    // Set top_object variable
    RTvariable top_object;
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "top_object", &top_object));
    RT_CHECK_ERROR(rtVariableSetObject(top_object, top_level_group));

    /* Create all required buffers and variables */

    // Geometry buffers
    addBuffer("patch_vertices", patch_vertices_RTbuffer, patch_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2);
    addBuffer("triangle_vertices", triangle_vertices_RTbuffer, triangle_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2);
    addBuffer("disk_centers", disk_centers_RTbuffer, disk_centers_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1);
    addBuffer("disk_radii", disk_radii_RTbuffer, disk_radii_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("disk_normals", disk_normals_RTbuffer, disk_normals_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1);
    addBuffer("tile_vertices", tile_vertices_RTbuffer, tile_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2);
    addBuffer("voxel_vertices", voxel_vertices_RTbuffer, voxel_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2);
    addBuffer("bbox_vertices", bbox_vertices_RTbuffer, bbox_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2);

    // Primitive data buffers
    addBuffer("transform_matrix", transform_matrix_RTbuffer, transform_matrix_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 2);
    addBuffer("primitive_type", primitive_type_RTbuffer, primitive_type_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("primitive_solid_fraction", primitive_solid_fraction_RTbuffer, primitive_solid_fraction_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("twosided_flag", twosided_flag_RTbuffer, twosided_flag_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_BYTE, 1);

    // UUID buffers
    addBuffer("patch_UUID", patch_UUID_RTbuffer, patch_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("triangle_UUID", triangle_UUID_RTbuffer, triangle_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("disk_UUID", disk_UUID_RTbuffer, disk_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("tile_UUID", tile_UUID_RTbuffer, tile_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("voxel_UUID", voxel_UUID_RTbuffer, voxel_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("bbox_UUID", bbox_UUID_RTbuffer, bbox_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);

    // Mapping buffers
    addBuffer("objectID", objectID_RTbuffer, objectID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("primitiveID", primitiveID_RTbuffer, primitiveID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("primitive_positions", primitive_positions_RTbuffer, primitive_positions_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("object_subdivisions", object_subdivisions_RTbuffer, object_subdivisions_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT2, 1);

    // Material property buffers
    addBuffer("rho", rho_RTbuffer, rho_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("tau", tau_RTbuffer, tau_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("rho_cam", rho_cam_RTbuffer, rho_cam_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("tau_cam", tau_cam_RTbuffer, tau_cam_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("specular_exponent", specular_exponent_RTbuffer, specular_exponent_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("specular_scale", specular_scale_RTbuffer, specular_scale_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);

    // Radiation energy buffers
    addBuffer("radiation_in", radiation_in_RTbuffer, radiation_in_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("radiation_out_top", radiation_out_top_RTbuffer, radiation_out_top_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("radiation_out_bottom", radiation_out_bottom_RTbuffer, radiation_out_bottom_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("radiation_in_camera", radiation_in_camera_RTbuffer, radiation_in_camera_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("camera_pixel_label", camera_pixel_label_RTbuffer, camera_pixel_label_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("camera_pixel_depth", camera_pixel_depth_RTbuffer, camera_pixel_depth_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("scatter_buff_top", scatter_buff_top_RTbuffer, scatter_buff_top_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("scatter_buff_bottom", scatter_buff_bottom_RTbuffer, scatter_buff_bottom_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("radiation_specular", radiation_specular_RTbuffer, radiation_specular_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("Rsky", Rsky_RTbuffer, Rsky_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("scatter_buff_top_cam", scatter_buff_top_cam_RTbuffer, scatter_buff_top_cam_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("scatter_buff_bottom_cam", scatter_buff_bottom_cam_RTbuffer, scatter_buff_bottom_cam_RTvariable, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1);

    // Source buffers
    addBuffer("source_positions", source_positions_RTbuffer, source_positions_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1);
    addBuffer("source_widths", source_widths_RTbuffer, source_widths_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 1);
    addBuffer("source_rotations", source_rotations_RTbuffer, source_rotations_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1);
    addBuffer("source_types", source_types_RTbuffer, source_types_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
    addBuffer("source_fluxes", source_fluxes_RTbuffer, source_fluxes_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("source_fluxes_cam", source_fluxes_cam_RTbuffer, source_fluxes_cam_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);

    // Diffuse radiation buffers
    addBuffer("diffuse_flux", diffuse_flux_RTbuffer, diffuse_flux_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("diffuse_extinction", diffuse_extinction_RTbuffer, diffuse_extinction_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("diffuse_peak_dir", diffuse_peak_dir_RTbuffer, diffuse_peak_dir_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1);
    addBuffer("diffuse_dist_norm", diffuse_dist_norm_RTbuffer, diffuse_dist_norm_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);

    // Sky model buffers
    addBuffer("sky_radiance_params", sky_radiance_params_RTbuffer, sky_radiance_params_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1);
    addBuffer("camera_sky_radiance", camera_sky_radiance_RTbuffer, camera_sky_radiance_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);
    addBuffer("solar_disk_radiance", solar_disk_radiance_RTbuffer, solar_disk_radiance_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);

    // Band control buffers
    addBuffer("band_launch_flag", band_launch_flag_RTbuffer, band_launch_flag_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_BYTE, 1);
    addBuffer("max_scatters", max_scatters_RTbuffer, max_scatters_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);

    // Texture/masking buffers
    addBuffer("masksize", masksize_RTbuffer, masksize_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT2, 1);
    addBuffer("maskID", maskID_RTbuffer, maskID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT, 1);
    addBuffer("uvdata", uvdata_RTbuffer, uvdata_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 2);
    addBuffer("uvID", uvID_RTbuffer, uvID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_INT, 1);

    // Special handling for 3D mask buffer
    RT_CHECK_ERROR(rtBufferCreate(OptiX_Context, RT_BUFFER_INPUT, &maskdata_RTbuffer));
    RT_CHECK_ERROR(rtBufferSetFormat(maskdata_RTbuffer, RT_FORMAT_BYTE));
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "maskdata", &maskdata_RTvariable));
    RT_CHECK_ERROR(rtVariableSetObject(maskdata_RTvariable, maskdata_RTbuffer));
    std::vector<std::vector<std::vector<bool>>> dummydata;
    initializeBuffer3Dbool(maskdata_RTbuffer, dummydata);

    // Context variables
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "Nprimitives", &Nprimitives_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(Nprimitives_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "bbox_UUID_base", &bbox_UUID_base_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(bbox_UUID_base_RTvariable, UINT_MAX)); // Initialize to sentinel (no bboxes)

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "Nsources", &Nsources_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(Nsources_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "Nbands_global", &Nbands_global_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(Nbands_global_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "Nbands_launch", &Nbands_launch_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(Nbands_launch_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "Ncameras", &Ncameras_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(Ncameras_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "periodic_flag", &periodic_flag_RTvariable));
    RT_CHECK_ERROR(rtVariableSet2f(periodic_flag_RTvariable, 0.f, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "specular_reflection_enabled", &specular_reflection_enabled_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(specular_reflection_enabled_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "scattering_iteration", &scattering_iteration_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(scattering_iteration_RTvariable, 0));

    // Launch control variables
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "random_seed", &random_seed_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(random_seed_RTvariable, std::chrono::system_clock::now().time_since_epoch().count()));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "launch_offset", &launch_offset_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(launch_offset_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "launch_face", &launch_face_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(launch_face_RTvariable, 0));

    // Camera variables
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_position", &camera_position_RTvariable));
    RT_CHECK_ERROR(rtVariableSet3f(camera_position_RTvariable, 0.f, 0.f, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_direction", &camera_direction_RTvariable));
    RT_CHECK_ERROR(rtVariableSet2f(camera_direction_RTvariable, 0.f, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_lens_diameter", &camera_lens_diameter_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1f(camera_lens_diameter_RTvariable, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "FOV_aspect_ratio", &FOV_aspect_ratio_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1f(FOV_aspect_ratio_RTvariable, 1.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_HFOV", &camera_HFOV_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1f(camera_HFOV_RTvariable, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_focal_length", &camera_focal_length_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1f(camera_focal_length_RTvariable, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_viewplane_length", &camera_viewplane_length_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1f(camera_viewplane_length_RTvariable, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_pixel_solid_angle", &camera_pixel_solid_angle_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1f(camera_pixel_solid_angle_RTvariable, 0.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_pixel_offset_x", &camera_pixel_offset_x_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(camera_pixel_offset_x_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_pixel_offset_y", &camera_pixel_offset_y_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(camera_pixel_offset_y_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_ID", &camera_ID_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1ui(camera_ID_RTvariable, 0));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "camera_resolution_full", &camera_resolution_full_RTvariable));
    RT_CHECK_ERROR(rtVariableSet2i(camera_resolution_full_RTvariable, 0, 0));

    // Sun direction for sky model
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "sun_direction", &sun_direction_RTvariable));
    RT_CHECK_ERROR(rtVariableSet3f(sun_direction_RTvariable, 0.f, 0.f, 1.f));

    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "solar_disk_cos_angle", &solar_disk_cos_angle_RTvariable));
    RT_CHECK_ERROR(rtVariableSet1f(solar_disk_cos_angle_RTvariable, 0.f));

    is_initialized = true;
}

void OptiX6Backend::shutdown() {

    if (!is_initialized) {
        return; // Already shut down or never initialized
    }

    // Destroy OptiX context (this destroys all child objects automatically)
    if (OptiX_Context) {
        RT_CHECK_ERROR_NOEXIT(rtContextDestroy(OptiX_Context));
        OptiX_Context = nullptr;
    }

    is_initialized = false;
}

void OptiX6Backend::updateGeometry(const RayTracingGeometry &geometry) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::updateGeometry): Backend not initialized.");
    }

    // Validate geometry before upload (debug builds only)
    // Catches buffer sizing errors that cause 90% of backend debugging issues
    validateGeometryBeforeUpload(geometry);

    // Convert geometry data to OptiX buffers
    geometryToBuffers(geometry);

    // Update primitive counts
    current_primitive_count = geometry.primitive_count;
    current_patch_count = geometry.patch_count;
    current_triangle_count = geometry.triangle_count;
    current_disk_count = geometry.disk_count;
    current_tile_count = geometry.tile_count;
    current_voxel_count = geometry.voxel_count;
    current_bbox_count = geometry.bbox_count;

    RT_CHECK_ERROR(rtVariableSet1ui(Nprimitives_RTvariable, geometry.primitive_count));
    RT_CHECK_ERROR(rtVariableSet1ui(bbox_UUID_base_RTvariable, geometry.bbox_UUID_base));

    // Update periodic boundary flags
    RT_CHECK_ERROR(rtVariableSet2f(periodic_flag_RTvariable, geometry.periodic_flag.x, geometry.periodic_flag.y));

    // Mark acceleration structure as dirty (needs rebuild)
    RT_CHECK_ERROR(rtAccelerationMarkDirty(base_acceleration));
}

void OptiX6Backend::buildAccelerationStructure() {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::buildAccelerationStructure): Backend not initialized.");
    }

    // Set primitive counts for each geometry type (use type-specific counts, not total)
    RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(patch_geometry, current_patch_count));
    RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(triangle_geometry, current_triangle_count));
    RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(disk_geometry, current_disk_count));
    RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(tile_geometry, current_tile_count));
    RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(voxel_geometry, current_voxel_count));
    RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(bbox_geometry, current_bbox_count));

    // OptiX will automatically rebuild the acceleration structure on next launch
}

void OptiX6Backend::updateMaterials(const RayTracingMaterial &materials) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::updateMaterials): Backend not initialized.");
    }

    materialsToBuffers(materials);

    // Update material counts
    current_band_count = materials.num_bands;
    current_source_count = materials.num_sources;
    current_camera_count = materials.num_cameras;

    // Update Ncameras variable in backend's OptiX context
    RT_CHECK_ERROR(rtVariableSet1ui(Ncameras_RTvariable, materials.num_cameras));
}

void OptiX6Backend::updateSources(const std::vector<RayTracingSource> &sources) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::updateSources): Backend not initialized.");
    }

    sourcesToBuffers(sources);

    // Update source count variable
    current_source_count = sources.size();
    RT_CHECK_ERROR(rtVariableSet1ui(Nsources_RTvariable, sources.size()));
}

void OptiX6Backend::updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm, const std::vector<float> &sky_energy) {

    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::updateDiffuseRadiation): Backend not initialized.");
    }

    diffuseToBuffers(flux, extinction, peak_dir, dist_norm, sky_energy);
}

void OptiX6Backend::updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction, const std::vector<float> &solar_disk_radiance, float solar_disk_cos_angle) {

    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::updateSkyModel): Backend not initialized.");
    }

    skyModelToBuffers(sky_radiance_params, camera_sky_radiance, sun_direction, solar_disk_radiance, solar_disk_cos_angle);
}

void OptiX6Backend::launchDirectRays(const RayTracingLaunchParams &params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::launchDirectRays): Backend not initialized.");
    }

    // Validate context to ensure acceleration structure is built and buffers are synchronized
    RT_CHECK_ERROR(rtContextValidate(OptiX_Context));

    // OptiX 6.5 batching: limit total rays per launch to avoid GPU timeout/memory issues
    // Maximum rays per launch (1 billion = OptiX practical limit)
    size_t maxRays = 1024 * 1024 * 1024;
    uint n = std::ceil(std::sqrt(static_cast<double>(params.rays_per_primitive)));
    size_t rays_per_primitive = n * n;

    // Calculate batching parameters
    size_t maxPrims = std::floor(static_cast<float>(maxRays) / static_cast<float>(rays_per_primitive));
    size_t Nlaunches = std::ceil(rays_per_primitive * params.launch_count / static_cast<float>(maxRays));
    size_t prims_per_launch = std::min(static_cast<size_t>(params.launch_count), maxPrims);

    // Batch launches if primitive count exceeds limit
    for (size_t launch = 0; launch < Nlaunches; launch++) {
        size_t prims_this_launch;
        if ((launch + 1) * prims_per_launch > params.launch_count) {
            prims_this_launch = params.launch_count - launch * prims_per_launch;
        } else {
            prims_this_launch = prims_per_launch;
        }

        // Set launch offset for this batch
        uint launch_offset = params.launch_offset + launch * prims_per_launch;
        RT_CHECK_ERROR(rtVariableSet1ui(launch_offset_RTvariable, launch_offset));

        // Set launch parameters for this batch (excluding launch_offset which is set above)
        RT_CHECK_ERROR(rtVariableSet1ui(random_seed_RTvariable, params.random_seed));
        RT_CHECK_ERROR(rtVariableSet1ui(Nbands_global_RTvariable, params.num_bands_global));
        RT_CHECK_ERROR(rtVariableSet1ui(Nbands_launch_RTvariable, params.num_bands_launch));
        RT_CHECK_ERROR(rtVariableSet1ui(launch_face_RTvariable, params.launch_face));
        RT_CHECK_ERROR(rtVariableSet1ui(scattering_iteration_RTvariable, params.scattering_iteration));

        // Band launch flags (same for all batches)
        if (!params.band_launch_flag.empty()) {
            initializeBuffer1Dbool(band_launch_flag_RTbuffer, params.band_launch_flag);
        }

        // Specular reflection flag
        uint specular_enabled = params.specular_reflection_enabled ? 1 : 0;
        RT_CHECK_ERROR(rtVariableSet1ui(specular_reflection_enabled_RTvariable, specular_enabled));

        // Launch this batch: dimension = (n, n, primitives_this_batch)
        RT_CHECK_ERROR(rtContextLaunch3D(OptiX_Context, RAYTYPE_DIRECT, n, n, prims_this_launch));
    }
}

void OptiX6Backend::launchDiffuseRays(const RayTracingLaunchParams &params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::launchDiffuseRays): Backend not initialized.");
    }

    // Upload emission/outgoing radiation if provided (upload once for all batches)
    if (!params.radiation_out_top.empty()) {
        initializeBuffer1Df(radiation_out_top_RTbuffer, params.radiation_out_top);
    }
    if (!params.radiation_out_bottom.empty()) {
        initializeBuffer1Df(radiation_out_bottom_RTbuffer, params.radiation_out_bottom);
    }

    // Upload diffuse parameters if provided (upload once for all batches)
    if (!params.diffuse_flux.empty()) {
        initializeBuffer1Df(diffuse_flux_RTbuffer, params.diffuse_flux);
    }
    if (!params.diffuse_extinction.empty()) {
        initializeBuffer1Df(diffuse_extinction_RTbuffer, params.diffuse_extinction);
    }
    if (!params.diffuse_peak_dir.empty()) {
        initializeBuffer1Dfloat3(diffuse_peak_dir_RTbuffer, params.diffuse_peak_dir);
    }
    if (!params.diffuse_dist_norm.empty()) {
        initializeBuffer1Df(diffuse_dist_norm_RTbuffer, params.diffuse_dist_norm);
    }

    // Validate context to ensure acceleration structure is built
    RT_CHECK_ERROR(rtContextValidate(OptiX_Context));

    // OptiX 6.5 batching: limit total rays per launch to avoid GPU timeout/memory issues
    // Maximum rays per launch (1 billion = OptiX practical limit)
    size_t maxRays = 1024 * 1024 * 1024;
    uint n = std::ceil(std::sqrt(static_cast<double>(params.rays_per_primitive)));
    size_t rays_per_primitive = n * n;

    // Calculate batching parameters
    size_t maxPrims = std::floor(static_cast<float>(maxRays) / static_cast<float>(rays_per_primitive));
    size_t Nlaunches = std::ceil(rays_per_primitive * params.launch_count / static_cast<float>(maxRays));
    size_t prims_per_launch = std::min(static_cast<size_t>(params.launch_count), maxPrims);

    // Batch launches if primitive count exceeds limit
    for (size_t launch = 0; launch < Nlaunches; launch++) {
        size_t prims_this_launch;
        if ((launch + 1) * prims_per_launch > params.launch_count) {
            prims_this_launch = params.launch_count - launch * prims_per_launch;
        } else {
            prims_this_launch = prims_per_launch;
        }

        // Set launch offset for this batch
        uint launch_offset = params.launch_offset + launch * prims_per_launch;
        RT_CHECK_ERROR(rtVariableSet1ui(launch_offset_RTvariable, launch_offset));

        // Set launch parameters for this batch (excluding launch_offset which is set above)
        RT_CHECK_ERROR(rtVariableSet1ui(random_seed_RTvariable, params.random_seed));
        RT_CHECK_ERROR(rtVariableSet1ui(Nbands_global_RTvariable, params.num_bands_global));
        RT_CHECK_ERROR(rtVariableSet1ui(Nbands_launch_RTvariable, params.num_bands_launch));
        RT_CHECK_ERROR(rtVariableSet1ui(launch_face_RTvariable, params.launch_face));
        RT_CHECK_ERROR(rtVariableSet1ui(scattering_iteration_RTvariable, params.scattering_iteration));

        // Band launch flags (same for all batches)
        if (!params.band_launch_flag.empty()) {
            initializeBuffer1Dbool(band_launch_flag_RTbuffer, params.band_launch_flag);
        }

        // Specular reflection flag
        uint specular_enabled = params.specular_reflection_enabled ? 1 : 0;
        RT_CHECK_ERROR(rtVariableSet1ui(specular_reflection_enabled_RTvariable, specular_enabled));

        // Launch this batch: dimension = (n, n, primitives_this_batch)
        RT_CHECK_ERROR(rtContextLaunch3D(OptiX_Context, RAYTYPE_DIFFUSE, n, n, prims_this_launch));
    }
}

void OptiX6Backend::launchCameraRays(const RayTracingLaunchParams &params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::launchCameraRays): Backend not initialized.");
    }


    // Set common launch parameters
    launchParamsToVariables(params);

    // Set camera-specific parameters
    RT_CHECK_ERROR(rtVariableSet3f(camera_position_RTvariable, params.camera_position.x, params.camera_position.y, params.camera_position.z));

    RT_CHECK_ERROR(rtVariableSet2f(camera_direction_RTvariable, params.camera_direction.x, params.camera_direction.y));

    RT_CHECK_ERROR(rtVariableSet1f(camera_focal_length_RTvariable, params.camera_focal_length));
    RT_CHECK_ERROR(rtVariableSet1f(camera_lens_diameter_RTvariable, params.camera_lens_diameter));
    RT_CHECK_ERROR(rtVariableSet1f(FOV_aspect_ratio_RTvariable, params.camera_fov_aspect));

    // Debug: check camera_HFOV value
    if (std::isnan(params.camera_HFOV) || std::isinf(params.camera_HFOV)) {
    }

    RT_CHECK_ERROR(rtVariableSet1f(camera_HFOV_RTvariable, params.camera_HFOV));

    RT_CHECK_ERROR(rtVariableSet1ui(camera_pixel_offset_x_RTvariable, params.camera_pixel_offset.x));
    RT_CHECK_ERROR(rtVariableSet1ui(camera_pixel_offset_y_RTvariable, params.camera_pixel_offset.y));
    RT_CHECK_ERROR(rtVariableSet1ui(camera_ID_RTvariable, params.camera_id));

    // Set the 3 new camera parameters
    RT_CHECK_ERROR(rtVariableSet1f(camera_viewplane_length_RTvariable, params.camera_viewplane_length));
    RT_CHECK_ERROR(rtVariableSet1f(camera_pixel_solid_angle_RTvariable, params.camera_pixel_solid_angle));
    RT_CHECK_ERROR(rtVariableSet2i(camera_resolution_full_RTvariable, params.camera_resolution_full.x, params.camera_resolution_full.y));

    // Resize camera buffer for full resolution (not tile resolution!)
    size_t total_pixels = params.camera_resolution_full.x * params.camera_resolution_full.y;
    size_t buffer_size = total_pixels * params.num_bands_launch;
    if (buffer_size > 0) {
        zeroBuffer1D(radiation_in_camera_RTbuffer, buffer_size);
    }

    // Validate context to ensure acceleration structure is built and buffers are synchronized
    RT_CHECK_ERROR(rtContextValidate(OptiX_Context));

    // Launch camera rays: dimension = (antialiasing_samples, resolution.x, resolution.y) for pixel sampling
    RT_CHECK_ERROR(rtContextLaunch3D(OptiX_Context, RAYTYPE_CAMERA, params.antialiasing_samples, params.camera_resolution.x, params.camera_resolution.y));
}

void OptiX6Backend::launchPixelLabelRays(const RayTracingLaunchParams &params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::launchPixelLabelRays): Backend not initialized.");
    }

    // Set launch parameters
    launchParamsToVariables(params);

    // Set only essential parameters for pixel coordinate calculations
    // Camera orientation (position/direction) is inherited from camera rendering
    // (This matches master's behavior where pixel labeling reuses camera settings)
    RT_CHECK_ERROR(rtVariableSet1f(camera_viewplane_length_RTvariable, params.camera_viewplane_length));
    RT_CHECK_ERROR(rtVariableSet1f(camera_pixel_solid_angle_RTvariable, params.camera_pixel_solid_angle));
    RT_CHECK_ERROR(rtVariableSet2i(camera_resolution_full_RTvariable, params.camera_resolution_full.x, params.camera_resolution_full.y));

    // Set camera pixel offset for tiling
    RT_CHECK_ERROR(rtVariableSet1ui(camera_pixel_offset_x_RTvariable, params.camera_pixel_offset.x));
    RT_CHECK_ERROR(rtVariableSet1ui(camera_pixel_offset_y_RTvariable, params.camera_pixel_offset.y));

    // NOTE: Camera pixel buffers must be zeroed BEFORE the tile loop, not here!
    // Zeroing happens in zeroCameraPixelBuffers() called from RadiationModel.cpp

    // Launch pixel label rays: dimension = (1, resolution.x, resolution.y) - no antialiasing
    RT_CHECK_ERROR(rtContextLaunch3D(OptiX_Context, RAYTYPE_PIXEL_LABEL, 1, params.camera_resolution.x, params.camera_resolution.y));
}

void OptiX6Backend::getRadiationResults(RayTracingResults &results) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::getRadiationResults): Backend not initialized.");
    }

    // Extract results from OptiX buffers
    buffersToResults(results);

    // Set dimension information
    results.num_primitives = current_primitive_count;
    results.num_bands = current_band_count;
    results.num_sources = current_source_count;
    results.num_cameras = current_camera_count;
}

void OptiX6Backend::getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels, std::vector<float> &pixel_depths, uint camera_id, const helios::int2 &resolution) {

    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::getCameraResults): Backend not initialized.");
    }

    // Extract camera pixel data from buffers
    pixel_data = getOptiXbufferData(radiation_in_camera_RTbuffer);
    pixel_labels = getOptiXbufferData_ui(camera_pixel_label_RTbuffer);
    pixel_depths = getOptiXbufferData(camera_pixel_depth_RTbuffer);
}

void OptiX6Backend::zeroRadiationBuffers(size_t launch_band_count) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::zeroRadiationBuffers): Backend not initialized.");
    }

    // Validation: launch bands cannot exceed global bands
    if (launch_band_count > current_band_count) {
        helios_runtime_error("ERROR (OptiX6Backend::zeroRadiationBuffers): launch_band_count (" + std::to_string(launch_band_count) + ") exceeds current_band_count (" + std::to_string(current_band_count) + ").");
    }

    // Zero all radiation result buffers (use current_band_count for global accumulation)
    // Note: Bbox primitives don't accumulate radiation (they only wrap rays),
    // so buffers are sized for real primitives only
    size_t buffer_size = current_primitive_count * current_band_count;
    if (buffer_size > 0) {
        zeroBuffer1D(radiation_in_RTbuffer, buffer_size);
        zeroBuffer1D(radiation_out_top_RTbuffer, buffer_size);
        zeroBuffer1D(radiation_out_bottom_RTbuffer, buffer_size);
        zeroBuffer1D(scatter_buff_top_RTbuffer, buffer_size);
        zeroBuffer1D(scatter_buff_bottom_RTbuffer, buffer_size);
    }

    // Zero camera scatter buffers (use launch_band_count for per-launch sizing)
    // Camera scatter uses same indexing as regular scatter: [primitive][band]
    if (current_camera_count > 0) {
        size_t cam_scatter_size = current_primitive_count * launch_band_count;
        if (cam_scatter_size > 0) {
            zeroBuffer1D(scatter_buff_top_cam_RTbuffer, cam_scatter_size);
            zeroBuffer1D(scatter_buff_bottom_cam_RTbuffer, cam_scatter_size);
        }
    }

    // Zero specular buffer (use current_band_count for global accumulation)
    size_t specular_size = current_source_count * current_camera_count * current_primitive_count * current_band_count;
    if (specular_size > 0) {
        zeroBuffer1D(radiation_specular_RTbuffer, specular_size);
    }

    // Zero sky energy buffer
    if (current_band_count > 0) {
        zeroBuffer1D(Rsky_RTbuffer, current_band_count);
    }
}

void OptiX6Backend::zeroScatterBuffers() {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::zeroScatterBuffers): Backend not initialized.");
    }

    // Zero primitive scatter buffers (between iterations)
    size_t buffer_size = current_primitive_count * current_band_count;
    if (buffer_size > 0) {
        zeroBuffer1D(scatter_buff_top_RTbuffer, buffer_size);
        zeroBuffer1D(scatter_buff_bottom_RTbuffer, buffer_size);
    }

    // NOTE: Camera scatter buffers are NOT zeroed here
    // They accumulate across all scatter iterations and are only zeroed once in zeroRadiationBuffers()
}

void OptiX6Backend::zeroCameraScatterBuffers(size_t launch_band_count) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::zeroCameraScatterBuffers): Backend not initialized.");
    }

    // Validation: launch bands cannot exceed global bands
    if (launch_band_count > current_band_count) {
        helios_runtime_error("ERROR (OptiX6Backend::zeroCameraScatterBuffers): launch_band_count (" + std::to_string(launch_band_count) + ") exceeds current_band_count (" + std::to_string(current_band_count) + ").");
    }

    // Zero camera scatter buffers (use launch_band_count for per-launch sizing)
    if (current_camera_count > 0) {
        size_t buffer_size = current_primitive_count * launch_band_count;
        if (buffer_size > 0) {
            zeroBuffer1D(scatter_buff_top_cam_RTbuffer, buffer_size);
            zeroBuffer1D(scatter_buff_bottom_cam_RTbuffer, buffer_size);
        }
    }
}

void OptiX6Backend::zeroCameraPixelBuffers(const helios::int2 &resolution) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::zeroCameraPixelBuffers): Backend not initialized.");
    }

    // Zero pixel label and depth buffers for full resolution
    size_t total_pixels = resolution.x * resolution.y;
    if (total_pixels > 0) {
        zeroBuffer1D(camera_pixel_label_RTbuffer, total_pixels);
        zeroBuffer1D(camera_pixel_depth_RTbuffer, total_pixels);
    }
}

void OptiX6Backend::copyScatterToRadiation() {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::copyScatterToRadiation): Backend not initialized.");
    }

    // Copy scatter buffer contents to radiation_out buffers
    copyBuffer1D(scatter_buff_top_RTbuffer, radiation_out_top_RTbuffer);
    copyBuffer1D(scatter_buff_bottom_RTbuffer, radiation_out_bottom_RTbuffer);
}

void OptiX6Backend::uploadRadiationOut(const std::vector<float> &radiation_out_top, const std::vector<float> &radiation_out_bottom) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::uploadRadiationOut): Backend not initialized.");
    }

    if (!radiation_out_top.empty()) {
        initializeBuffer1Df(radiation_out_top_RTbuffer, radiation_out_top);
    }
    if (!radiation_out_bottom.empty()) {
        initializeBuffer1Df(radiation_out_bottom_RTbuffer, radiation_out_bottom);
    }
}

void OptiX6Backend::uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam, const std::vector<float> &scatter_bottom_cam) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::uploadCameraScatterBuffers): Backend not initialized.");
    }

    if (!scatter_top_cam.empty()) {
        initializeBuffer1Df(scatter_buff_top_cam_RTbuffer, scatter_top_cam);
    }
    if (!scatter_bottom_cam.empty()) {
        initializeBuffer1Df(scatter_buff_bottom_cam_RTbuffer, scatter_bottom_cam);
    }
}

void OptiX6Backend::uploadSourceFluxes(const std::vector<float> &fluxes) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX6Backend::uploadSourceFluxes): Backend not initialized.");
    }

    if (!fluxes.empty()) {
        initializeBuffer1Df(source_fluxes_RTbuffer, fluxes);
    }
}

void OptiX6Backend::queryGPUMemory() const {
    if (!is_initialized) {
        std::cout << "Backend not initialized - cannot query GPU memory." << std::endl;
        return;
    }

    // Query OptiX memory usage
    RTsize memory_used;
    RT_CHECK_ERROR(rtContextGetAttribute(OptiX_Context, RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY, sizeof(RTsize), &memory_used));

    // Memory info available via backend->queryGPUMemory() - removed automatic output for cleaner tests
}

std::string OptiX6Backend::getBackendName() const {
    return "OptiX 6.5";
}

// ========== Error Handling Helpers ==========

static void sutilReportError(const char *message) {
    fprintf(stderr, "OptiX Error: %s\n", message);
#if defined(_WIN32) && defined(RELEASE_PUBLIC)
    {
        char s[2048];
        sprintf(s, "OptiX Error: %s", message);
        MessageBox(0, s, "OptiX Error", MB_OK | MB_ICONWARNING | MB_SYSTEMMODAL);
    }
#endif
}

static void sutilHandleError(RTcontext context, RTresult code, const char *file, int line) {
    const char *message;
    char s[2048];
    rtContextGetErrorString(context, code, &message);
    sprintf(s, "%s\n(%s:%d)", message, file, line);
    sutilReportError(s);
    exit(1);
}

// ========== Private Helper Methods: Buffer Management ==========

void OptiX6Backend::addBuffer(const char *name, RTbuffer &buffer, RTvariable &variable, RTbuffertype type, RTformat format, size_t dimension) {
    RT_CHECK_ERROR(rtBufferCreate(OptiX_Context, type, &buffer));
    RT_CHECK_ERROR(rtBufferSetFormat(buffer, format));
    RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, name, &variable));
    RT_CHECK_ERROR(rtVariableSetObject(variable, buffer));
    if (dimension == 1) {
        zeroBuffer1D(buffer, 1);
    } else if (dimension == 2) {
        zeroBuffer2D(buffer, helios::make_int2(1, 1));
    } else {
        helios_runtime_error("ERROR (OptiX6Backend::addBuffer): invalid buffer dimension of " + std::to_string(dimension) + ", must be 1 or 2.");
    }
}

void OptiX6Backend::zeroBuffer1D(RTbuffer &buffer, size_t bsize) {
    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));

    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));

    if (format == RT_FORMAT_FLOAT) {
        float *data = (float *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = 0.0f;
        }
    } else if (format == RT_FORMAT_FLOAT2) {
        optix::float2 *data = (optix::float2 *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = optix::make_float2(0, 0);
        }
    } else if (format == RT_FORMAT_FLOAT3) {
        optix::float3 *data = (optix::float3 *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = optix::make_float3(0, 0, 0);
        }
    } else if (format == RT_FORMAT_FLOAT4) {
        optix::float4 *data = (optix::float4 *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = optix::make_float4(0, 0, 0, 0);
        }
    } else if (format == RT_FORMAT_UNSIGNED_INT) {
        uint *data = (uint *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = 0;
        }
    } else if (format == RT_FORMAT_INT) {
        int *data = (int *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = 0;
        }
    } else if (format == RT_FORMAT_INT2) {
        optix::int2 *data = (optix::int2 *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = optix::make_int2(0, 0);
        }
    } else if (format == RT_FORMAT_BYTE) {
        char *data = (char *) ptr;
        for (size_t i = 0; i < bsize; i++) {
            data[i] = 0;
        }
    } else {
        RT_CHECK_ERROR(rtBufferUnmap(buffer));
        helios_runtime_error("ERROR (OptiX6Backend::zeroBuffer1D): Unsupported buffer format.");
    }

    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::zeroBuffer2D(RTbuffer &buffer, const helios::int2 &bsize) {
    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));

    if (format == RT_FORMAT_FLOAT) {
        std::vector<std::vector<float>> array(bsize.y, std::vector<float>(bsize.x, 0.0f));
        initializeBuffer2Df(buffer, array);
    } else if (format == RT_FORMAT_FLOAT2) {
        std::vector<std::vector<helios::vec2>> array(bsize.y, std::vector<helios::vec2>(bsize.x, helios::make_vec2(0, 0)));
        initializeBuffer2Dfloat2(buffer, array);
    } else if (format == RT_FORMAT_FLOAT3) {
        std::vector<std::vector<optix::float3>> array(bsize.y, std::vector<optix::float3>(bsize.x, optix::make_float3(0, 0, 0)));
        initializeBuffer2Dfloat3(buffer, array);
    } else {
        helios_runtime_error("ERROR (OptiX6Backend::zeroBuffer2D): Unsupported buffer format.");
    }
}

void OptiX6Backend::initializeBuffer1Df(RTbuffer &buffer, const std::vector<float> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_FLOAT) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer1Df): Buffer must have type float.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    float *data = (float *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = array[i];
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Dui(RTbuffer &buffer, const std::vector<uint> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_UNSIGNED_INT) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer1Dui): Buffer must have type unsigned int.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    uint *data = (uint *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = array[i];
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Di(RTbuffer &buffer, const std::vector<int> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_INT) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer1Di): Buffer must have type int.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    int *data = (int *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = array[i];
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Dchar(RTbuffer &buffer, const std::vector<char> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_BYTE) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer1Dchar): Buffer must have type char.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    char *data = (char *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = array[i];
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Dbool(RTbuffer &buffer, const std::vector<bool> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    char *data = (char *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = array[i] ? 1 : 0;
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Dfloat2(RTbuffer &buffer, const std::vector<helios::vec2> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    optix::float2 *data = (optix::float2 *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = optix::make_float2(array[i].x, array[i].y);
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Dfloat3(RTbuffer &buffer, const std::vector<helios::vec3> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    optix::float3 *data = (optix::float3 *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = optix::make_float3(array[i].x, array[i].y, array[i].z);
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Dfloat4(RTbuffer &buffer, const std::vector<helios::vec4> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    optix::float4 *data = (optix::float4 *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = optix::make_float4(array[i].x, array[i].y, array[i].z, array[i].w);
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer1Dint2(RTbuffer &buffer, const std::vector<helios::int2> &array) {
    size_t bsize = array.size();
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    optix::int2 *data = (optix::int2 *) ptr;
    for (size_t i = 0; i < bsize; i++) {
        data[i] = optix::make_int2(array[i].x, array[i].y);
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer2Dfloat2(RTbuffer &buffer, const std::vector<std::vector<helios::vec2>> &array) {
    helios::int2 bsize;
    bsize.y = array.size();
    bsize.x = (bsize.y == 0) ? 0 : array.front().size();

    RT_CHECK_ERROR(rtBufferSetSize2D(buffer, bsize.x, bsize.y));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    optix::float2 *data = (optix::float2 *) ptr;
    for (int j = 0; j < bsize.y; j++) {
        for (int i = 0; i < bsize.x; i++) {
            data[i + j * bsize.x] = optix::make_float2(array[j][i].x, array[j][i].y);
        }
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer2Df(RTbuffer &buffer, const std::vector<std::vector<float>> &array) {
    helios::int2 bsize;
    bsize.y = array.size();
    bsize.x = (bsize.y == 0) ? 0 : array.front().size();

    RT_CHECK_ERROR(rtBufferSetSize2D(buffer, bsize.x, bsize.y));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_FLOAT) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer2Df): Buffer must have type float.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    float *data = (float *) ptr;
    for (int j = 0; j < bsize.y; j++) {
        for (int i = 0; i < bsize.x; i++) {
            data[i + j * bsize.x] = array[j][i];
        }
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer2Dui(RTbuffer &buffer, const std::vector<std::vector<uint>> &array) {
    helios::int2 bsize;
    bsize.y = array.size();
    bsize.x = (bsize.y == 0) ? 0 : array.front().size();

    RT_CHECK_ERROR(rtBufferSetSize2D(buffer, bsize.x, bsize.y));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_UNSIGNED_INT) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer2Dui): Buffer must have type unsigned int.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    uint *data = (uint *) ptr;
    for (int j = 0; j < bsize.y; j++) {
        for (int i = 0; i < bsize.x; i++) {
            data[i + j * bsize.x] = array[j][i];
        }
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer2Di(RTbuffer &buffer, const std::vector<std::vector<int>> &array) {
    helios::int2 bsize;
    bsize.y = array.size();
    bsize.x = (bsize.y == 0) ? 0 : array.front().size();

    RT_CHECK_ERROR(rtBufferSetSize2D(buffer, bsize.x, bsize.y));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_INT) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer2Di): Buffer must have type int.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    int *data = (int *) ptr;
    for (int j = 0; j < bsize.y; j++) {
        for (int i = 0; i < bsize.x; i++) {
            data[i + j * bsize.x] = array[j][i];
        }
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer2Dfloat3(RTbuffer &buffer, const std::vector<std::vector<helios::vec3>> &array) {
    // Convert helios::vec3 to optix::float3
    std::vector<std::vector<optix::float3>> optix_array;
    optix_array.resize(array.size());
    for (size_t j = 0; j < array.size(); j++) {
        optix_array[j].resize(array[j].size());
        for (size_t i = 0; i < array[j].size(); i++) {
            optix_array[j][i] = optix::make_float3(array[j][i].x, array[j][i].y, array[j][i].z);
        }
    }
    initializeBuffer2Dfloat3(buffer, optix_array);
}

void OptiX6Backend::initializeBuffer2Dfloat3(RTbuffer &buffer, const std::vector<std::vector<optix::float3>> &array) {
    helios::int2 bsize;
    bsize.y = array.size();
    bsize.x = (bsize.y == 0) ? 0 : array.front().size();

    RT_CHECK_ERROR(rtBufferSetSize2D(buffer, bsize.x, bsize.y));

    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    if (format != RT_FORMAT_FLOAT3) {
        helios_runtime_error("ERROR (OptiX6Backend::initializeBuffer2Dfloat3): Buffer must have type float3.");
    }

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    optix::float3 *data = (optix::float3 *) ptr;
    for (int j = 0; j < bsize.y; j++) {
        for (int i = 0; i < bsize.x; i++) {
            data[i + j * bsize.x] = array[j][i];
        }
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::initializeBuffer3Dbool(RTbuffer &buffer, const std::vector<std::vector<std::vector<bool>>> &array) {
    // Template implementation for 3D buffers
    helios::int3 bsize;
    bsize.z = array.size();
    bsize.y = (bsize.z == 0) ? 0 : array.front().size();
    bsize.x = (bsize.y == 0) ? 0 : array.front().front().size();

    RT_CHECK_ERROR(rtBufferSetSize3D(buffer, bsize.x, bsize.y, bsize.z));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    char *data = (char *) ptr;
    for (int k = 0; k < bsize.z; k++) {
        for (int j = 0; j < bsize.y; j++) {
            for (int i = 0; i < bsize.x; i++) {
                data[i + j * bsize.x + k * bsize.x * bsize.y] = array[k][j][i] ? 1 : 0;
            }
        }
    }
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

void OptiX6Backend::copyBuffer1D(RTbuffer &source, RTbuffer &dest) {
    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(source, &format));

    RTsize bsize;
    rtBufferGetSize1D(source, &bsize);
    rtBufferSetSize1D(dest, bsize);

    if (format == RT_FORMAT_FLOAT) {
        void *ptr_src;
        RT_CHECK_ERROR(rtBufferMap(source, &ptr_src));
        float *data_src = (float *) ptr_src;

        void *ptr_dest;
        RT_CHECK_ERROR(rtBufferMap(dest, &ptr_dest));
        float *data_dest = (float *) ptr_dest;

        for (size_t i = 0; i < bsize; i++) {
            data_dest[i] = data_src[i];
        }

        RT_CHECK_ERROR(rtBufferUnmap(source));
        RT_CHECK_ERROR(rtBufferUnmap(dest));
    } else {
        helios_runtime_error("ERROR (OptiX6Backend::copyBuffer1D): Only float buffers supported currently.");
    }
}

std::vector<float> OptiX6Backend::getOptiXbufferData(RTbuffer buffer) {
    RTsize bsize;
    RT_CHECK_ERROR(rtBufferGetSize1D(buffer, &bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    float *data = (float *) ptr;

    std::vector<float> result(bsize);
    for (size_t i = 0; i < bsize; i++) {
        result[i] = data[i];
    }

    RT_CHECK_ERROR(rtBufferUnmap(buffer));
    return result;
}

std::vector<uint> OptiX6Backend::getOptiXbufferData_ui(RTbuffer buffer) {
    RTsize bsize;
    RT_CHECK_ERROR(rtBufferGetSize1D(buffer, &bsize));

    void *ptr;
    RT_CHECK_ERROR(rtBufferMap(buffer, &ptr));
    uint *data = (uint *) ptr;

    std::vector<uint> result(bsize);
    for (size_t i = 0; i < bsize; i++) {
        result[i] = data[i];
    }

    RT_CHECK_ERROR(rtBufferUnmap(buffer));
    return result;
}

void OptiX6Backend::geometryToBuffers(const RayTracingGeometry &geometry) {
    // Convert backend-agnostic geometry data to OptiX buffers

    // Transform matrices: 1D vector → 2D buffer [primitive+bbox][16]
    // Geometry data only contains transforms for real primitives (not bboxes)
    // We append identity transforms for bboxes here
    if (!geometry.transform_matrices.empty()) {
        size_t total_count = geometry.primitive_count + geometry.bbox_count;
        std::vector<std::vector<float>> transform_2d(total_count);

        // Copy real primitive transforms
        for (size_t p = 0; p < geometry.primitive_count; p++) {
            transform_2d[p].resize(16);
            for (int i = 0; i < 16; i++) {
                transform_2d[p][i] = geometry.transform_matrices[p * 16 + i];
            }
        }

        // Append identity transforms for bboxes
        for (size_t b = 0; b < geometry.bbox_count; b++) {
            size_t bbox_idx = geometry.primitive_count + b;
            transform_2d[bbox_idx].resize(16, 0.0f);
            transform_2d[bbox_idx][0] = 1.0f;   // m00
            transform_2d[bbox_idx][5] = 1.0f;   // m11
            transform_2d[bbox_idx][10] = 1.0f;  // m22
            transform_2d[bbox_idx][15] = 1.0f;  // m33
        }

        initializeBuffer2Df(transform_matrix_RTbuffer, transform_2d);
    }

    // Primitive types: append bbox entries for OptiX
    if (!geometry.primitive_types.empty()) {
        std::vector<uint> types_with_bbox = geometry.primitive_types;
        for (size_t i = 0; i < geometry.bbox_count; i++) {
            types_with_bbox.push_back(5); // type=5 for bbox
        }
        initializeBuffer1Dui(primitive_type_RTbuffer, types_with_bbox);
    }

    // Primitive IDs: append bbox UUIDs for OptiX
    if (!geometry.primitive_IDs.empty()) {
        std::vector<uint> ids_with_bbox = geometry.primitive_IDs;
        for (size_t i = 0; i < geometry.bbox_count; i++) {
            ids_with_bbox.push_back(geometry.bbox_UUID_base + i);
        }
        initializeBuffer1Dui(primitiveID_RTbuffer, ids_with_bbox);
    }

    // Primitive positions (UUID → array position lookup)
    if (!geometry.primitive_positions.empty()) {
        initializeBuffer1Dui(primitive_positions_RTbuffer, geometry.primitive_positions);
    }

    // Object IDs: append bbox object IDs for OptiX
    if (!geometry.object_IDs.empty()) {
        std::vector<uint> obj_ids_with_bbox = geometry.object_IDs;
        for (size_t i = 0; i < geometry.bbox_count; i++) {
            obj_ids_with_bbox.push_back(geometry.primitive_count + i);
        }
        initializeBuffer1Dui(objectID_RTbuffer, obj_ids_with_bbox);
    }

    // Two-sided flags: append bbox flags for OptiX
    if (!geometry.twosided_flags.empty()) {
        std::vector<char> flags_with_bbox = geometry.twosided_flags;
        for (size_t i = 0; i < geometry.bbox_count; i++) {
            flags_with_bbox.push_back(1); // bboxes are two-sided
        }
        initializeBuffer1Dchar(twosided_flag_RTbuffer, flags_with_bbox);
    }

    // Solid fractions: append bbox fractions for OptiX
    if (!geometry.solid_fractions.empty()) {
        std::vector<float> fractions_with_bbox = geometry.solid_fractions;
        for (size_t i = 0; i < geometry.bbox_count; i++) {
            fractions_with_bbox.push_back(1.0f); // bboxes are fully solid
        }
        initializeBuffer1Df(primitive_solid_fraction_RTbuffer, fractions_with_bbox);
    }

    // Object subdivisions: append bbox subdivisions for OptiX
    if (!geometry.object_subdivisions.empty()) {
        std::vector<helios::int2> subdivs_with_bbox = geometry.object_subdivisions;
        for (size_t i = 0; i < geometry.bbox_count; i++) {
            subdivs_with_bbox.push_back(helios::make_int2(1, 1)); // no subdivisions
        }
        initializeBuffer1Dint2(object_subdivisions_RTbuffer, subdivs_with_bbox);
    }

    // Patch vertices: std::vector<vec3> → 2D buffer [patch][4]
    if (geometry.patch_count > 0 && !geometry.patches.vertices.empty()) {
        std::vector<std::vector<helios::vec3>> patch_verts_2d(geometry.patch_count);
        for (size_t p = 0; p < geometry.patch_count; p++) {
            patch_verts_2d[p].resize(4);
            for (int v = 0; v < 4; v++) {
                patch_verts_2d[p][v] = geometry.patches.vertices[p * 4 + v];
            }
        }
        initializeBuffer2Dfloat3(patch_vertices_RTbuffer, patch_verts_2d);

        // Patch UUIDs
        initializeBuffer1Dui(patch_UUID_RTbuffer, geometry.patches.UUIDs);
    }

    // Triangle vertices: std::vector<vec3> → 2D buffer [triangle][3]
    if (geometry.triangle_count > 0 && !geometry.triangles.vertices.empty()) {
        std::vector<std::vector<helios::vec3>> tri_verts_2d(geometry.triangle_count);
        for (size_t t = 0; t < geometry.triangle_count; t++) {
            tri_verts_2d[t].resize(3);
            for (int v = 0; v < 3; v++) {
                tri_verts_2d[t][v] = geometry.triangles.vertices[t * 3 + v];
            }
        }
        initializeBuffer2Dfloat3(triangle_vertices_RTbuffer, tri_verts_2d);

        // Triangle UUIDs
        initializeBuffer1Dui(triangle_UUID_RTbuffer, geometry.triangles.UUIDs);
    }

    // Disk geometry
    if (geometry.disk_count > 0) {
        initializeBuffer1Dfloat3(disk_centers_RTbuffer, geometry.disk_centers);
        initializeBuffer1Df(disk_radii_RTbuffer, geometry.disk_radii);
        initializeBuffer1Dfloat3(disk_normals_RTbuffer, geometry.disk_normals);
        initializeBuffer1Dui(disk_UUID_RTbuffer, geometry.disk_UUIDs);
    }

    // Tile vertices: std::vector<vec3> → 2D buffer [tile][4]
    if (geometry.tile_count > 0 && !geometry.tiles.vertices.empty()) {
        std::vector<std::vector<helios::vec3>> tile_verts_2d(geometry.tile_count);
        for (size_t t = 0; t < geometry.tile_count; t++) {
            tile_verts_2d[t].resize(4);
            for (int v = 0; v < 4; v++) {
                tile_verts_2d[t][v] = geometry.tiles.vertices[t * 4 + v];
            }
        }
        initializeBuffer2Dfloat3(tile_vertices_RTbuffer, tile_verts_2d);

        initializeBuffer1Dui(tile_UUID_RTbuffer, geometry.tiles.UUIDs);
    }

    // Voxel vertices: std::vector<vec3> → 2D buffer [voxel][8]
    if (geometry.voxel_count > 0 && !geometry.voxels.vertices.empty()) {
        std::vector<std::vector<helios::vec3>> voxel_verts_2d(geometry.voxel_count);
        for (size_t v = 0; v < geometry.voxel_count; v++) {
            voxel_verts_2d[v].resize(8);
            for (int vtx = 0; vtx < 8; vtx++) {
                voxel_verts_2d[v][vtx] = geometry.voxels.vertices[v * 8 + vtx];
            }
        }
        initializeBuffer2Dfloat3(voxel_vertices_RTbuffer, voxel_verts_2d);
        initializeBuffer1Dui(voxel_UUID_RTbuffer, geometry.voxels.UUIDs);
    }

    // Bbox vertices: std::vector<vec3> → 2D buffer [bbox][4]
    // Bbox faces are 4-vertex rectangles (verified in primitiveIntersection.cu:390-393)
    if (geometry.bbox_count > 0 && !geometry.bboxes.vertices.empty()) {
        std::vector<std::vector<helios::vec3>> bbox_verts_2d(geometry.bbox_count);
        for (size_t b = 0; b < geometry.bbox_count; b++) {
            bbox_verts_2d[b].resize(4);
            for (int v = 0; v < 4; v++) {
                bbox_verts_2d[b][v] = geometry.bboxes.vertices[b * 4 + v];
            }
        }
        initializeBuffer2Dfloat3(bbox_vertices_RTbuffer, bbox_verts_2d);
        initializeBuffer1Dui(bbox_UUID_RTbuffer, geometry.bboxes.UUIDs);
    }

    // Object subdivisions
    if (!geometry.object_subdivisions.empty()) {
        initializeBuffer1Dint2(object_subdivisions_RTbuffer, geometry.object_subdivisions);
    }

    // Texture masks
    if (!geometry.mask_data.empty()) {
        // Convert 1D bool array to 3D structure
        // CRITICAL: All masks must have same dimensions for 3D buffer, so pad to max size
        int max_width = 0, max_height = 0;
        for (const auto &size: geometry.mask_sizes) {
            max_width = std::max(max_width, size.x);
            max_height = std::max(max_height, size.y);
        }

        std::vector<std::vector<std::vector<bool>>> mask_3d;
        size_t offset = 0;
        for (size_t m = 0; m < geometry.mask_sizes.size(); m++) {
            int width = geometry.mask_sizes[m].x;
            int height = geometry.mask_sizes[m].y;
            // Pad to max dimensions (padded regions will be false)
            std::vector<std::vector<bool>> mask_2d(max_height, std::vector<bool>(max_width, false));
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    mask_2d[y][x] = geometry.mask_data[offset++];
                }
            }
            mask_3d.push_back(mask_2d);
        }
        initializeBuffer3Dbool(maskdata_RTbuffer, mask_3d);
        initializeBuffer1Dint2(masksize_RTbuffer, geometry.mask_sizes);
    }

    if (!geometry.mask_IDs.empty()) {
        initializeBuffer1Di(maskID_RTbuffer, geometry.mask_IDs);
    }

    // UV data
    // uv_IDs contains position indices (not offsets), used by CUDA to access uvdata[vertex][position]
    // uv_data is stored sequentially: 4 UVs per primitive that has UVs
    if (!geometry.uv_data.empty()) {
        // Convert 1D vec2 array to 2D structure: uv_2d[position][vertex]
        std::vector<std::vector<helios::vec2>> uv_2d(geometry.primitive_count);
        size_t uv_offset = 0;
        for (size_t p = 0; p < geometry.primitive_count; p++) {
            int uv_id = geometry.uv_IDs[p];
            if (uv_id >= 0) {
                // This primitive has UVs - read next 4 from uv_data
                uv_2d[p].resize(4);
                for (int v = 0; v < 4 && uv_offset < geometry.uv_data.size(); v++) {
                    uv_2d[p][v] = geometry.uv_data[uv_offset++];
                }
            } else {
                // No UVs - use default
                uv_2d[p] = {helios::make_vec2(0, 0), helios::make_vec2(1, 0), helios::make_vec2(1, 1), helios::make_vec2(0, 1)};
            }
        }
        initializeBuffer2Dfloat2(uvdata_RTbuffer, uv_2d);
    }

    if (!geometry.uv_IDs.empty()) {
        initializeBuffer1Di(uvID_RTbuffer, geometry.uv_IDs);
    }
}

void OptiX6Backend::materialsToBuffers(const RayTracingMaterial &materials) {
    // Upload material properties to OptiX buffers
    // Indexing: [source * Nbands * Nprims + prim * Nbands + band]
    // This matches CUDA formula: Nprimitives * Nbands_global * source_ID + Nbands_global * origin_UUID + b_global

    if (!materials.reflectivity.empty()) {
        initializeBuffer1Df(rho_RTbuffer, materials.reflectivity);
    }

    if (!materials.transmissivity.empty()) {
        initializeBuffer1Df(tau_RTbuffer, materials.transmissivity);
    }

    if (!materials.reflectivity_cam.empty()) {
        initializeBuffer1Df(rho_cam_RTbuffer, materials.reflectivity_cam);
    }

    if (!materials.transmissivity_cam.empty()) {
        initializeBuffer1Df(tau_cam_RTbuffer, materials.transmissivity_cam);
    }

    if (!materials.specular_exponent.empty()) {
        initializeBuffer1Df(specular_exponent_RTbuffer, materials.specular_exponent);
    }

    if (!materials.specular_scale.empty()) {
        initializeBuffer1Df(specular_scale_RTbuffer, materials.specular_scale);
    }
}

void OptiX6Backend::sourcesToBuffers(const std::vector<RayTracingSource> &sources) {
    // Convert source data to OptiX buffers

    if (sources.empty()) {
        return;
    }

    std::vector<helios::vec3> positions;
    std::vector<helios::vec2> widths;
    std::vector<helios::vec3> rotations;
    std::vector<uint> types;
    std::vector<float> fluxes;
    std::vector<float> fluxes_cam;

    for (const auto &source: sources) {
        positions.push_back(source.position);
        widths.push_back(source.width);
        rotations.push_back(source.rotation);
        types.push_back(source.type);

        // Flatten flux arrays
        for (float flux: source.fluxes) {
            fluxes.push_back(flux);
        }
        for (float flux: source.fluxes_cam) {
            fluxes_cam.push_back(flux);
        }
    }

    initializeBuffer1Dfloat3(source_positions_RTbuffer, positions);
    initializeBuffer1Dfloat2(source_widths_RTbuffer, widths);
    initializeBuffer1Dfloat3(source_rotations_RTbuffer, rotations);
    initializeBuffer1Dui(source_types_RTbuffer, types);
    initializeBuffer1Df(source_fluxes_RTbuffer, fluxes);
    initializeBuffer1Df(source_fluxes_cam_RTbuffer, fluxes_cam);
}

void OptiX6Backend::diffuseToBuffers(const std::vector<float> &flux, const std::vector<float> &extinction, const std::vector<helios::vec3> &peak_dir, const std::vector<float> &dist_norm, const std::vector<float> &sky_energy) {
    // Upload diffuse radiation parameters

    if (!flux.empty()) {
        initializeBuffer1Df(diffuse_flux_RTbuffer, flux);
    }

    if (!extinction.empty()) {
        initializeBuffer1Df(diffuse_extinction_RTbuffer, extinction);
    }

    if (!peak_dir.empty()) {
        initializeBuffer1Dfloat3(diffuse_peak_dir_RTbuffer, peak_dir);
    }

    if (!dist_norm.empty()) {
        initializeBuffer1Df(diffuse_dist_norm_RTbuffer, dist_norm);
    }

    if (!sky_energy.empty()) {
        initializeBuffer1Df(Rsky_RTbuffer, sky_energy);
    }
}

void OptiX6Backend::skyModelToBuffers(const std::vector<helios::vec4> &sky_radiance_params, const std::vector<float> &camera_sky_radiance, const helios::vec3 &sun_direction, const std::vector<float> &solar_disk_radiance, float solar_disk_cos_angle) {
    // Upload sky model parameters for camera rendering

    if (!sky_radiance_params.empty()) {
        initializeBuffer1Dfloat4(sky_radiance_params_RTbuffer, sky_radiance_params);
    }

    if (!camera_sky_radiance.empty()) {
        initializeBuffer1Df(camera_sky_radiance_RTbuffer, camera_sky_radiance);
    }

    // Set sun direction variable
    RT_CHECK_ERROR(rtVariableSet3f(sun_direction_RTvariable, sun_direction.x, sun_direction.y, sun_direction.z));

    if (!solar_disk_radiance.empty()) {
        initializeBuffer1Df(solar_disk_radiance_RTbuffer, solar_disk_radiance);
    }

    // Set solar disk angular size
    RT_CHECK_ERROR(rtVariableSet1f(solar_disk_cos_angle_RTvariable, solar_disk_cos_angle));
}

void OptiX6Backend::launchParamsToVariables(const RayTracingLaunchParams &params) {
    // Set common launch parameters as OptiX variables

    RT_CHECK_ERROR(rtVariableSet1ui(random_seed_RTvariable, params.random_seed));
    RT_CHECK_ERROR(rtVariableSet1ui(launch_offset_RTvariable, params.launch_offset));
    RT_CHECK_ERROR(rtVariableSet1ui(Nbands_global_RTvariable, params.num_bands_global));
    RT_CHECK_ERROR(rtVariableSet1ui(Nbands_launch_RTvariable, params.num_bands_launch));
    RT_CHECK_ERROR(rtVariableSet1ui(launch_face_RTvariable, params.launch_face));
    RT_CHECK_ERROR(rtVariableSet1ui(scattering_iteration_RTvariable, params.scattering_iteration));

    // Band launch flags
    if (!params.band_launch_flag.empty()) {
        initializeBuffer1Dbool(band_launch_flag_RTbuffer, params.band_launch_flag);
    }

    // Specular reflection flag
    uint specular_enabled = params.specular_reflection_enabled ? 1 : 0;
    RT_CHECK_ERROR(rtVariableSet1ui(specular_reflection_enabled_RTvariable, specular_enabled));
}

void OptiX6Backend::buffersToResults(RayTracingResults &results) {
    // Extract radiation results from OptiX buffers

    results.radiation_in = getOptiXbufferData(radiation_in_RTbuffer);
    results.radiation_out_top = getOptiXbufferData(radiation_out_top_RTbuffer);
    results.radiation_out_bottom = getOptiXbufferData(radiation_out_bottom_RTbuffer);
    results.scatter_buff_top = getOptiXbufferData(scatter_buff_top_RTbuffer);
    results.scatter_buff_bottom = getOptiXbufferData(scatter_buff_bottom_RTbuffer);

    // Extract camera scatter buffers (if cameras present)
    // Use current_camera_count since results.num_cameras not set yet
    if (current_camera_count > 0) {
        results.scatter_buff_top_cam = getOptiXbufferData(scatter_buff_top_cam_RTbuffer);
        results.scatter_buff_bottom_cam = getOptiXbufferData(scatter_buff_bottom_cam_RTbuffer);
    }

    results.radiation_specular = getOptiXbufferData(radiation_specular_RTbuffer);
    results.sky_energy = getOptiXbufferData(Rsky_RTbuffer);
}
