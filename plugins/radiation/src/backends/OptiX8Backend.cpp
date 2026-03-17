/** \file "OptiX8Backend.cpp" OptiX 8.1 ray tracing backend implementation.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "OptiX8Backend.h"

// OptiX function table definition (must be in exactly one .cpp)
#include <optix_function_table_definition.h>

#include "Context.h"

#include <algorithm>
#include <cfloat>
#include <fstream>

namespace helios {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

OptiX8Backend::OptiX8Backend() = default;

bool OptiX8Backend::probe() noexcept {
    try {
        int device_count = 0;
        cudaError_t rc = cudaGetDeviceCount(&device_count);
        if (rc != cudaSuccess || device_count == 0) {
            return false;
        }
        OptixResult optix_rc = optixInit();
        return (optix_rc == OPTIX_SUCCESS);
    } catch (...) {
        return false;
    }
}

OptiX8Backend::~OptiX8Backend() {
    if (is_initialized) {
        shutdown();
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void OptiX8Backend::initialize() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(nullptr)); // Force CUDA context initialization

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&cuda_stream));

    // Initialize OptiX function table from the loaded driver
    OPTIX_CHECK(optixInit());

    // Create OptiX device context
    CUcontext cuda_context = nullptr; // use current context
    OptixDeviceContextOptions ctx_options = {};
    ctx_options.logCallbackFunction = [](unsigned int level, const char *tag, const char *message, void *) {
        if (level <= 2) {
            std::cerr << "[OptiX][" << tag << "] " << message << "\n";
        }
    };
    ctx_options.logCallbackLevel = 2;
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, &ctx_options, &optix_context));

    // Load device code (PTX or OptixIR) and compile module
    const std::string device_code_path = findDeviceCodeFile();
    std::ifstream file(device_code_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (OptiX8Backend::initialize): Could not open device code file: " + device_code_path);
    }
    const std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> device_code(file_size);
    if (!file.read(device_code.data(), file_size)) {
        helios_runtime_error("ERROR (OptiX8Backend::initialize): Could not read device code file: " + device_code_path);
    }

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    module_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur                   = 0;
    pipeline_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues                 = 2;   // two uint32 for pointer-in-registers
    pipeline_options.numAttributeValues               = 2;   // UUID + face in attributes
    pipeline_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char log[4096];
    size_t log_size = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(
        optix_context,
        &module_options,
        &pipeline_options,
        device_code.data(),
        static_cast<size_t>(file_size),
        log, &log_size,
        &optix_module));

    // ---- Create program groups ----
    OptixProgramGroupOptions pg_options = {};

    // Raygen programs
    auto createRaygen = [&](const char *entry, OptixProgramGroup &pg) {
        OptixProgramGroupDesc desc = {};
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = optix_module;
        desc.raygen.entryFunctionName = entry;
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &desc, 1, &pg_options, log, &log_size, &pg));
        (void)log_size;
    };

    createRaygen("__raygen__direct",       pg_raygen_direct);
    createRaygen("__raygen__diffuse",      pg_raygen_diffuse);
    createRaygen("__raygen__camera",       pg_raygen_camera);
    createRaygen("__raygen__pixel_label",  pg_raygen_pixel_label);

    // Miss programs
    auto createMiss = [&](const char *entry, OptixProgramGroup &pg) {
        OptixProgramGroupDesc desc = {};
        desc.kind                    = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module             = optix_module;
        desc.miss.entryFunctionName  = entry;
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &desc, 1, &pg_options, log, &log_size, &pg));
        (void)log_size;
    };

    createMiss("__miss__direct",       pg_miss_direct);
    createMiss("__miss__diffuse",      pg_miss_diffuse);
    createMiss("__miss__camera",       pg_miss_camera);
    createMiss("__miss__pixel_label",  pg_miss_pixel_label);

    // Hit groups: one per ray type (4 total). All geometry uses __intersection__patch,
    // which dispatches internally on primitive type. With a single GAS and numSbtRecords=1,
    // the SBT hit record index = sbt_offset from optixTrace (stride=0 for all ray types).

    auto createHitGroup = [&](const char *ch_entry, const char *is_entry, OptixProgramGroup &pg) {
        OptixProgramGroupDesc desc = {};
        desc.kind                                   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH                      = optix_module;
        desc.hitgroup.entryFunctionNameCH           = ch_entry;
        desc.hitgroup.moduleIS                      = optix_module;
        desc.hitgroup.entryFunctionNameIS           = is_entry;
        desc.hitgroup.moduleAH                      = nullptr;
        desc.hitgroup.entryFunctionNameAH           = nullptr;
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context, &desc, 1, &pg_options, log, &log_size, &pg));
        (void)log_size;
    };

    createHitGroup("__closesthit__direct",       "__intersection__patch", pg_hit_direct);
    createHitGroup("__closesthit__diffuse",      "__intersection__patch", pg_hit_diffuse);
    createHitGroup("__closesthit__camera",       "__intersection__patch", pg_hit_camera);
    createHitGroup("__closesthit__pixel_label",  "__intersection__patch", pg_hit_pixel_label);

    // ---- Create pipeline ----
    OptixProgramGroup all_groups[] = {
        pg_raygen_direct, pg_raygen_diffuse, pg_raygen_camera, pg_raygen_pixel_label,
        pg_miss_direct,   pg_miss_diffuse,   pg_miss_camera,   pg_miss_pixel_label,
        pg_hit_direct,    pg_hit_diffuse,    pg_hit_camera,    pg_hit_pixel_label
    };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;

    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        optix_context,
        &pipeline_options,
        &link_options,
        all_groups,
        sizeof(all_groups) / sizeof(all_groups[0]),
        log, &log_size,
        &optix_pipeline));

    // Set pipeline stack sizes using OptiX utilities
    OptixStackSizes stack_sizes = {};
    for (auto &pg : all_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, optix_pipeline));
    }
    uint32_t max_trace_depth = 1;
    uint32_t direct_callable_stack_size_from_traversal = 0;
    uint32_t direct_callable_stack_size_from_state     = 0;
    uint32_t continuation_stack_size                   = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        0, // maxCCDepth (no continuation callables)
        0, // maxDCDepth (no direct callables)
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        optix_pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        1 /* maxTraversableGraphDepth */));

    // Allocate device-side launch params buffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(OptiX8LaunchParams)));
    memset(&h_params, 0, sizeof(h_params));

    is_initialized = true;
}

void OptiX8Backend::shutdown() {
    if (!is_initialized) {
        return;
    }

    // Synchronize before cleanup
    if (cuda_stream) {
        cudaStreamSynchronize(cuda_stream);
    }

    // Free device buffers
    freeGeometryBuffers();
    freeMaterialBuffers();

    auto freePtr = [](CUdeviceptr &ptr) {
        if (ptr) { cudaFree(reinterpret_cast<void *>(ptr)); ptr = 0; }
    };

    freePtr(d_radiation_in);
    freePtr(d_radiation_out_top);
    freePtr(d_radiation_out_bottom);
    freePtr(d_scatter_buff_top);
    freePtr(d_scatter_buff_bottom);
    freePtr(d_radiation_in_camera);
    freePtr(d_scatter_buff_top_cam);
    freePtr(d_scatter_buff_bottom_cam);
    freePtr(d_radiation_specular);
    freePtr(d_Rsky);
    freePtr(d_camera_pixel_label);
    freePtr(d_camera_pixel_depth);
    freePtr(d_source_positions);
    freePtr(d_source_rotations);
    freePtr(d_source_widths);
    freePtr(d_source_types);
    freePtr(d_source_fluxes);
    freePtr(d_source_fluxes_cam);
    freePtr(d_diffuse_flux);
    freePtr(d_diffuse_extinction);
    freePtr(d_diffuse_peak_dir);
    freePtr(d_diffuse_dist_norm);
    freePtr(d_sky_radiance_params);
    freePtr(d_camera_sky_radiance);
    freePtr(d_solar_disk_radiance);
    freePtr(d_band_launch_flag);
    freePtr(d_mask_data);
    freePtr(d_mask_sizes);
    freePtr(d_mask_IDs);
    freePtr(d_uv_data);
    freePtr(d_uv_IDs);
    freePtr(d_params);

    // Free SBT device memory
    if (d_raygen_records)   { cudaFree(reinterpret_cast<void *>(d_raygen_records));   d_raygen_records   = 0; }
    if (d_miss_records)     { cudaFree(reinterpret_cast<void *>(d_miss_records));     d_miss_records     = 0; }
    if (d_hitgroup_records) { cudaFree(reinterpret_cast<void *>(d_hitgroup_records)); d_hitgroup_records = 0; }

    // Free GAS
    if (d_gas_output) { cudaFree(reinterpret_cast<void *>(d_gas_output)); d_gas_output = 0; }

    // Destroy program groups
    auto destroyPG = [](OptixProgramGroup &pg) {
        if (pg) { optixProgramGroupDestroy(pg); pg = nullptr; }
    };
    destroyPG(pg_raygen_direct); destroyPG(pg_raygen_diffuse);
    destroyPG(pg_raygen_camera); destroyPG(pg_raygen_pixel_label);
    destroyPG(pg_miss_direct);   destroyPG(pg_miss_diffuse);
    destroyPG(pg_miss_camera);   destroyPG(pg_miss_pixel_label);
    destroyPG(pg_hit_direct);    destroyPG(pg_hit_diffuse);
    destroyPG(pg_hit_camera);    destroyPG(pg_hit_pixel_label);

    if (optix_pipeline) { optixPipelineDestroy(optix_pipeline); optix_pipeline = nullptr; }
    if (optix_module)   { optixModuleDestroy(optix_module);     optix_module   = nullptr; }
    if (optix_context)  { optixDeviceContextDestroy(optix_context); optix_context = nullptr; }
    if (cuda_stream)    { cudaStreamDestroy(cuda_stream); cuda_stream = nullptr; }

    is_initialized = false;
}

// ---------------------------------------------------------------------------
// Geometry management
// ---------------------------------------------------------------------------

void OptiX8Backend::updateGeometry(const RayTracingGeometry &geometry) {
    validateGeometryBeforeUpload(geometry);

    // Validate that all primitive types are supported by the OptiX 8.1 backend.
    // Supported: patch (0), triangle (1), tile (3). Disk (2), voxel (4), and
    // bbox (5) intersection programs are not yet implemented.
    if (geometry.disk_count > 0) {
        helios_runtime_error("ERROR (OptiX8Backend::updateGeometry): Scene contains disk primitives, "
                             "which are not yet supported by the OptiX 8.1 backend.");
    }
    if (geometry.voxel_count > 0) {
        helios_runtime_error("ERROR (OptiX8Backend::updateGeometry): Scene contains voxel primitives, "
                             "which are not yet supported by the OptiX 8.1 backend.");
    }

    freeGeometryBuffers();

    auto upload = [this](CUdeviceptr &d_ptr, const void *src, size_t bytes) {
        if (bytes > 0 && src) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ptr), bytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_ptr), src, bytes, cudaMemcpyHostToDevice));
        }
    };

    upload(d_transform_matrix,      geometry.transform_matrices.data(),  geometry.transform_matrices.size()  * sizeof(float));
    upload(d_primitive_type,        geometry.primitive_types.data(),      geometry.primitive_types.size()      * sizeof(uint32_t));
    upload(d_primitive_positions,   geometry.primitive_positions.data(),  geometry.primitive_positions.size()  * sizeof(uint32_t));
    upload(d_primitive_uuid_arr,    geometry.primitive_UUIDs.data(),      geometry.primitive_UUIDs.size()      * sizeof(uint32_t));
    upload(d_primitiveID,           geometry.primitive_IDs.data(),        geometry.primitive_IDs.size()        * sizeof(uint32_t));
    upload(d_objectID,              geometry.object_IDs.data(),           geometry.object_IDs.size()           * sizeof(uint32_t));
    upload(d_twosided_flag,         geometry.twosided_flags.data(),       geometry.twosided_flags.size()       * sizeof(char));
    upload(d_primitive_solid_fraction, geometry.solid_fractions.data(),   geometry.solid_fractions.size()      * sizeof(float));

    // object_subdivisions: vector<helios::int2> → flat int32 array (2 ints per prim)
    if (!geometry.object_subdivisions.empty()) {
        const size_t bytes = geometry.object_subdivisions.size() * sizeof(helios::int2);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_object_subdivisions), bytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_object_subdivisions),
                              geometry.object_subdivisions.data(), bytes, cudaMemcpyHostToDevice));
    }

    // Per-type geometry
    if (geometry.patch_count > 0) {
        upload(d_patch_vertices, geometry.patches.vertices.data(),
               geometry.patches.vertices.size() * sizeof(helios::vec3));
        upload(d_patch_UUIDs, geometry.patches.UUIDs.data(),
               geometry.patches.UUIDs.size() * sizeof(uint32_t));
    }
    if (geometry.triangles.count > 0) {
        upload(d_triangle_vertices, geometry.triangles.vertices.data(),
               geometry.triangles.vertices.size() * sizeof(helios::vec3));
        upload(d_triangle_UUIDs, geometry.triangles.UUIDs.data(),
               geometry.triangles.UUIDs.size() * sizeof(uint32_t));
    }
    if (geometry.disk_count > 0) {
        upload(d_disk_centers, geometry.disk_centers.data(), geometry.disk_centers.size() * sizeof(helios::vec3));
        upload(d_disk_radii,   geometry.disk_radii.data(),   geometry.disk_radii.size()   * sizeof(float));
        upload(d_disk_normals, geometry.disk_normals.data(), geometry.disk_normals.size() * sizeof(helios::vec3));
        upload(d_disk_UUIDs,   geometry.disk_UUIDs.data(),   geometry.disk_UUIDs.size()   * sizeof(uint32_t));
    }
    if (geometry.tiles.count > 0) {
        upload(d_tile_vertices, geometry.tiles.vertices.data(),
               geometry.tiles.vertices.size() * sizeof(helios::vec3));
        upload(d_tile_UUIDs, geometry.tiles.UUIDs.data(),
               geometry.tiles.UUIDs.size() * sizeof(uint32_t));
    }
    if (geometry.voxels.count > 0) {
        upload(d_voxel_vertices, geometry.voxels.vertices.data(),
               geometry.voxels.vertices.size() * sizeof(helios::vec3));
        upload(d_voxel_UUIDs, geometry.voxels.UUIDs.data(),
               geometry.voxels.UUIDs.size() * sizeof(uint32_t));
    }
    if (geometry.bbox_count > 0) {
        upload(d_bbox_vertices, geometry.bboxes.vertices.data(),
               geometry.bboxes.vertices.size() * sizeof(helios::vec3));
        upload(d_bbox_UUIDs, geometry.bboxes.UUIDs.data(),
               geometry.bboxes.UUIDs.size() * sizeof(uint32_t));
    }

    // Extend primitive_type and primitive_uuid arrays to include bbox entries (type=5).
    // OptiX AABB indices are global: indices [0, Nprims) are real primitives,
    // indices [Nprims, Nprims+Nbboxes) are bbox faces.  The intersection dispatch
    // program reads params.primitive_type[optixGetPrimitiveIndex()] and needs
    // type-5 entries at those positions.
    if (geometry.bbox_count > 0) {
        const size_t Nprims  = geometry.primitive_count;
        const size_t Nbboxes = geometry.bbox_count;

        freeCUdeviceptr(d_primitive_type);
        std::vector<uint32_t> ext_types(geometry.primitive_types);
        ext_types.resize(Nprims + Nbboxes, 5u);
        upload(d_primitive_type, ext_types.data(), ext_types.size() * sizeof(uint32_t));

        freeCUdeviceptr(d_primitive_uuid_arr);
        std::vector<uint32_t> ext_uuids(geometry.primitive_UUIDs);
        ext_uuids.insert(ext_uuids.end(),
                         geometry.bboxes.UUIDs.begin(), geometry.bboxes.UUIDs.end());
        upload(d_primitive_uuid_arr, ext_uuids.data(), ext_uuids.size() * sizeof(uint32_t));
    }

    // ---- Texture mask and UV data ----
    {
        // Compute per-mask offsets (cumulative start index into mask_data)
        std::vector<uint32_t> mask_offsets;
        uint32_t cumulative = 0;
        for (const auto &sz : geometry.mask_sizes) {
            mask_offsets.push_back(cumulative);
            cumulative += static_cast<uint32_t>(sz.x) * static_cast<uint32_t>(sz.y);
        }

        // Convert vector<bool> to flat uint8 array (1=opaque, 0=transparent)
        std::vector<uint8_t> mask_data_u8(cumulative);
        for (uint32_t i = 0; i < cumulative; ++i) {
            mask_data_u8[i] = geometry.mask_data[i] ? 1u : 0u;
        }

        // Reformat UV data to flat [Nprims * 4] array (4 UV vertices per primitive).
        // uv_IDs[p] >= 0 flags whether primitive p has custom UV data; all textured
        // primitives store exactly 4 UV vertices in uv_data (triangles are padded).
        const size_t Np = geometry.primitive_count;
        std::vector<helios::vec2> uv_flat(Np * 4, helios::make_vec2(0.f, 0.f));
        size_t uv_read = 0;
        for (size_t p = 0; p < Np; ++p) {
            if (!geometry.uv_IDs.empty() && geometry.uv_IDs[p] >= 0) {
                for (int v = 0; v < 4 && uv_read < geometry.uv_data.size(); ++v) {
                    uv_flat[p * 4 + v] = geometry.uv_data[uv_read++];
                }
            }
        }

        freeCUdeviceptr(d_mask_data);
        freeCUdeviceptr(d_mask_offsets);
        freeCUdeviceptr(d_mask_sizes);
        freeCUdeviceptr(d_mask_IDs);
        freeCUdeviceptr(d_uv_data);
        freeCUdeviceptr(d_uv_IDs);

        upload(d_mask_data,    mask_data_u8.data(),                 mask_data_u8.size()  * sizeof(uint8_t));
        upload(d_mask_offsets, mask_offsets.data(),                 mask_offsets.size()  * sizeof(uint32_t));
        upload(d_mask_sizes,   geometry.mask_sizes.data(),          geometry.mask_sizes.size()  * sizeof(helios::int2));
        upload(d_mask_IDs,     geometry.mask_IDs.data(),            geometry.mask_IDs.size()    * sizeof(int32_t));
        upload(d_uv_data,      uv_flat.data(),                      uv_flat.size()       * sizeof(helios::vec2));
        upload(d_uv_IDs,       geometry.uv_IDs.data(),              geometry.uv_IDs.size()      * sizeof(int32_t));
    }

    // Update h_params device pointers
    const uint32_t Nprims = static_cast<uint32_t>(geometry.primitive_count);
    h_params.transform_matrix         = reinterpret_cast<float *>(d_transform_matrix);
    h_params.primitive_type           = reinterpret_cast<uint32_t *>(d_primitive_type);
    h_params.primitive_positions      = reinterpret_cast<uint32_t *>(d_primitive_positions);
    h_params.primitive_uuid           = reinterpret_cast<uint32_t *>(d_primitive_uuid_arr);
    h_params.primitiveID              = reinterpret_cast<uint32_t *>(d_primitiveID);
    h_params.objectID                 = reinterpret_cast<uint32_t *>(d_objectID);
    h_params.object_subdivisions      = reinterpret_cast<int32_t *>(d_object_subdivisions);
    h_params.twosided_flag            = reinterpret_cast<int8_t *>(d_twosided_flag);
    h_params.primitive_solid_fraction = reinterpret_cast<float *>(d_primitive_solid_fraction);
    h_params.patch_vertices           = reinterpret_cast<float3 *>(d_patch_vertices);
    h_params.patch_UUIDs              = reinterpret_cast<uint32_t *>(d_patch_UUIDs);
    h_params.triangle_vertices        = reinterpret_cast<float3 *>(d_triangle_vertices);
    h_params.triangle_UUIDs           = reinterpret_cast<uint32_t *>(d_triangle_UUIDs);
    h_params.disk_centers             = reinterpret_cast<float3 *>(d_disk_centers);
    h_params.disk_radii               = reinterpret_cast<float *>(d_disk_radii);
    h_params.disk_normals             = reinterpret_cast<float3 *>(d_disk_normals);
    h_params.disk_UUIDs               = reinterpret_cast<uint32_t *>(d_disk_UUIDs);
    h_params.tile_vertices            = reinterpret_cast<float3 *>(d_tile_vertices);
    h_params.tile_UUIDs               = reinterpret_cast<uint32_t *>(d_tile_UUIDs);
    h_params.voxel_vertices           = reinterpret_cast<float3 *>(d_voxel_vertices);
    h_params.voxel_UUIDs              = reinterpret_cast<uint32_t *>(d_voxel_UUIDs);
    h_params.bbox_vertices            = reinterpret_cast<float3 *>(d_bbox_vertices);
    h_params.bbox_UUIDs               = reinterpret_cast<uint32_t *>(d_bbox_UUIDs);
    h_params.Nprimitives              = Nprims;
    h_params.bbox_UUID_base           = geometry.bbox_UUID_base;
    h_params.periodic_flag            = make_float2(geometry.periodic_flag.x, geometry.periodic_flag.y);
    h_params.mask_data                = reinterpret_cast<uint8_t *>(d_mask_data);
    h_params.mask_offsets             = reinterpret_cast<uint32_t *>(d_mask_offsets);
    h_params.mask_sizes               = reinterpret_cast<int32_t *>(d_mask_sizes);
    h_params.mask_IDs                 = reinterpret_cast<int32_t *>(d_mask_IDs);
    h_params.uv_data                  = reinterpret_cast<float2 *>(d_uv_data);
    h_params.uv_IDs                   = reinterpret_cast<int32_t *>(d_uv_IDs);

    // Store counts
    current_primitive_count = geometry.primitive_count;
    current_patch_count     = geometry.patch_count;
    current_triangle_count  = geometry.triangle_count;
    current_disk_count      = geometry.disk_count;
    current_tile_count      = geometry.tile_count;
    current_voxel_count     = geometry.voxel_count;
    current_bbox_count      = geometry.bbox_count;

    buildAABBs(geometry);
}

void OptiX8Backend::buildAccelerationStructure() {
    if (current_primitive_count == 0) {
        helios_runtime_error("ERROR (OptiX8Backend::buildAccelerationStructure): No geometry uploaded. Call updateGeometry() first.");
    }
    buildGAS(static_cast<uint32_t>(current_primitive_count + current_bbox_count));
    buildSBT();
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

void OptiX8Backend::updateMaterials(const RayTracingMaterial &materials) {
    freeMaterialBuffers();

    auto upload = [this](CUdeviceptr &d_ptr, const void *src, size_t bytes) {
        if (bytes > 0 && src) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ptr), bytes));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_ptr), src, bytes, cudaMemcpyHostToDevice));
        }
    };

    // rho/tau: always allocate at least Nprims*Nbands elements (zeroed) to prevent null pointer
    // dereference in device code when Nsources==0 (diffuse-only scenarios).
    {
        const size_t Nprims = current_primitive_count;
        const size_t Nbands = materials.num_bands;
        const size_t alloc  = std::max(materials.reflectivity.size(), std::max(Nprims * Nbands, (size_t)1));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rho), alloc * sizeof(float)));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_rho), 0, alloc * sizeof(float)));
        if (!materials.reflectivity.empty()) {
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rho), materials.reflectivity.data(),
                                  materials.reflectivity.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    {
        const size_t Nprims = current_primitive_count;
        const size_t Nbands = materials.num_bands;
        const size_t alloc  = std::max(materials.transmissivity.size(), std::max(Nprims * Nbands, (size_t)1));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), alloc * sizeof(float)));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_tau), 0, alloc * sizeof(float)));
        if (!materials.transmissivity.empty()) {
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_tau), materials.transmissivity.data(),
                                  materials.transmissivity.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    if (!materials.reflectivity_cam.empty())
        upload(d_rho_cam, materials.reflectivity_cam.data(),  materials.reflectivity_cam.size()  * sizeof(float));
    if (!materials.transmissivity_cam.empty())
        upload(d_tau_cam, materials.transmissivity_cam.data(), materials.transmissivity_cam.size() * sizeof(float));
    if (!materials.specular_exponent.empty())
        upload(d_specular_exponent, materials.specular_exponent.data(), materials.specular_exponent.size() * sizeof(float));
    if (!materials.specular_scale.empty())
        upload(d_specular_scale, materials.specular_scale.data(), materials.specular_scale.size() * sizeof(float));

    // Allocate radiation energy buffers (Nprims × Nbands_global)
    const size_t Nprims = current_primitive_count;
    const size_t Nbands = materials.num_bands;
    const size_t rad_bytes = Nprims * Nbands * sizeof(float);

    reallocDevice(d_radiation_in,         rad_bytes);
    reallocDevice(d_radiation_out_top,    rad_bytes);
    reallocDevice(d_radiation_out_bottom, rad_bytes);
    reallocDevice(d_scatter_buff_top,     rad_bytes);
    reallocDevice(d_scatter_buff_bottom,  rad_bytes);

    current_band_count   = Nbands;
    current_source_count = materials.num_sources;
    current_camera_count = materials.num_cameras;

    // Update h_params
    h_params.rho                  = reinterpret_cast<float *>(d_rho);
    h_params.tau                  = reinterpret_cast<float *>(d_tau);
    h_params.rho_cam              = reinterpret_cast<float *>(d_rho_cam);
    h_params.tau_cam              = reinterpret_cast<float *>(d_tau_cam);
    h_params.specular_exponent    = reinterpret_cast<float *>(d_specular_exponent);
    h_params.specular_scale       = reinterpret_cast<float *>(d_specular_scale);
    h_params.radiation_in         = reinterpret_cast<float *>(d_radiation_in);
    h_params.radiation_out_top    = reinterpret_cast<float *>(d_radiation_out_top);
    h_params.radiation_out_bottom = reinterpret_cast<float *>(d_radiation_out_bottom);
    h_params.scatter_buff_top     = reinterpret_cast<float *>(d_scatter_buff_top);
    h_params.scatter_buff_bottom  = reinterpret_cast<float *>(d_scatter_buff_bottom);
    h_params.Nsources             = static_cast<uint32_t>(materials.num_sources);
    h_params.Ncameras             = static_cast<uint32_t>(materials.num_cameras);
    h_params.Nbands_global        = static_cast<uint32_t>(Nbands);
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

void OptiX8Backend::updateSources(const std::vector<RayTracingSource> &sources) {
    auto freePtr = [this](CUdeviceptr &ptr) { freeCUdeviceptr(ptr); };
    freePtr(d_source_positions);
    freePtr(d_source_rotations);
    freePtr(d_source_widths);
    freePtr(d_source_types);
    // d_source_fluxes is managed by uploadSourceFluxes()

    const size_t Nsources = sources.size();
    if (Nsources == 0) {
        current_source_count = 0;
        h_params.Nsources = 0;
        return;
    }

    std::vector<float3>   positions(Nsources);
    std::vector<float3>   rotations(Nsources);
    std::vector<float2>   widths(Nsources);
    std::vector<uint32_t> types(Nsources);

    for (size_t i = 0; i < Nsources; i++) {
        positions[i] = make_float3(sources[i].position.x, sources[i].position.y, sources[i].position.z);
        rotations[i] = make_float3(sources[i].rotation.x, sources[i].rotation.y, sources[i].rotation.z);
        widths[i]    = make_float2(sources[i].width.x,    sources[i].width.y);
        types[i]     = sources[i].type;
    }

    auto upload = [this](CUdeviceptr &d_ptr, const void *src, size_t bytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ptr), bytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_ptr), src, bytes, cudaMemcpyHostToDevice));
    };

    upload(d_source_positions, positions.data(), Nsources * sizeof(float3));
    upload(d_source_rotations, rotations.data(), Nsources * sizeof(float3));
    upload(d_source_widths,    widths.data(),    Nsources * sizeof(float2));
    upload(d_source_types,     types.data(),     Nsources * sizeof(uint32_t));

    // Upload camera-weighted source fluxes (full 3D buffer [source][band][camera])
    // This is needed during direct ray tracing for specular accumulation in __miss__direct.
    // Layout matches OptiX 6: flattened as [src0_band0_cam0, src0_band0_cam1, ..., src0_band1_cam0, ...]
    freeCUdeviceptr(d_source_fluxes_cam);
    std::vector<float> fluxes_cam;
    for (size_t i = 0; i < Nsources; i++) {
        for (float f : sources[i].fluxes_cam) {
            fluxes_cam.push_back(f);
        }
    }
    if (!fluxes_cam.empty()) {
        upload(d_source_fluxes_cam, fluxes_cam.data(), fluxes_cam.size() * sizeof(float));
        h_params.source_fluxes_cam = reinterpret_cast<float *>(d_source_fluxes_cam);
    }

    current_source_count = Nsources;

    h_params.Nsources         = static_cast<uint32_t>(Nsources);
    h_params.source_positions = reinterpret_cast<float3 *>(d_source_positions);
    h_params.source_rotations = reinterpret_cast<float3 *>(d_source_rotations);
    h_params.source_widths    = reinterpret_cast<float2 *>(d_source_widths);
    h_params.source_types     = reinterpret_cast<uint32_t *>(d_source_types);
}

// ---------------------------------------------------------------------------
// Diffuse / sky
// ---------------------------------------------------------------------------

void OptiX8Backend::updateDiffuseRadiation(const std::vector<float> &flux, const std::vector<float> &extinction,
                                            const std::vector<helios::vec3> &peak_dir,
                                            const std::vector<float> &dist_norm,
                                            const std::vector<float> &sky_energy) {
    freeCUdeviceptr(d_diffuse_flux);
    freeCUdeviceptr(d_diffuse_extinction);
    freeCUdeviceptr(d_diffuse_peak_dir);
    freeCUdeviceptr(d_diffuse_dist_norm);
    freeCUdeviceptr(d_Rsky);

    auto upload_f = [this](CUdeviceptr &ptr, const std::vector<float> &v) {
        if (!v.empty()) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), v.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(ptr), v.data(),
                                  v.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    };
    upload_f(d_diffuse_flux,       flux);
    upload_f(d_diffuse_extinction, extinction);
    upload_f(d_diffuse_dist_norm,  dist_norm);
    upload_f(d_Rsky,               sky_energy);

    if (!peak_dir.empty()) {
        std::vector<float3> pd_f3(peak_dir.size());
        for (size_t i = 0; i < peak_dir.size(); i++) {
            pd_f3[i] = make_float3(peak_dir[i].x, peak_dir[i].y, peak_dir[i].z);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_diffuse_peak_dir),
                              pd_f3.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_diffuse_peak_dir), pd_f3.data(),
                              pd_f3.size() * sizeof(float3), cudaMemcpyHostToDevice));
    }

    h_params.diffuse_flux       = reinterpret_cast<float *>(d_diffuse_flux);
    h_params.diffuse_extinction = reinterpret_cast<float *>(d_diffuse_extinction);
    h_params.diffuse_peak_dir   = reinterpret_cast<float3 *>(d_diffuse_peak_dir);
    h_params.diffuse_dist_norm  = reinterpret_cast<float *>(d_diffuse_dist_norm);
    h_params.Rsky               = reinterpret_cast<float *>(d_Rsky);
}

void OptiX8Backend::updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params,
                                    const std::vector<float> &camera_sky_radiance,
                                    const helios::vec3 &sun_direction,
                                    const std::vector<float> &solar_disk_radiance,
                                    float solar_disk_cos_angle) {
    // Upload sky_radiance_params (helios::vec4 → float4)
    freeCUdeviceptr(d_sky_radiance_params);
    if (!sky_radiance_params.empty()) {
        std::vector<float4> f4(sky_radiance_params.size());
        for (size_t i = 0; i < sky_radiance_params.size(); i++) {
            f4[i] = make_float4(sky_radiance_params[i].x, sky_radiance_params[i].y,
                                sky_radiance_params[i].z, sky_radiance_params[i].w);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_sky_radiance_params), f4.size() * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_sky_radiance_params), f4.data(),
                              f4.size() * sizeof(float4), cudaMemcpyHostToDevice));
    }

    freeCUdeviceptr(d_camera_sky_radiance);
    if (!camera_sky_radiance.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_camera_sky_radiance),
                              camera_sky_radiance.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_camera_sky_radiance), camera_sky_radiance.data(),
                              camera_sky_radiance.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    freeCUdeviceptr(d_solar_disk_radiance);
    if (!solar_disk_radiance.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_solar_disk_radiance),
                              solar_disk_radiance.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_solar_disk_radiance), solar_disk_radiance.data(),
                              solar_disk_radiance.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    h_params.sky_radiance_params  = reinterpret_cast<float4 *>(d_sky_radiance_params);
    h_params.camera_sky_radiance  = reinterpret_cast<float *>(d_camera_sky_radiance);
    h_params.solar_disk_radiance  = reinterpret_cast<float *>(d_solar_disk_radiance);
    h_params.sun_direction        = make_float3(sun_direction.x, sun_direction.y, sun_direction.z);
    h_params.solar_disk_cos_angle = solar_disk_cos_angle;
}

// ---------------------------------------------------------------------------
// Ray launching
// ---------------------------------------------------------------------------

void OptiX8Backend::launchDirectRays(const RayTracingLaunchParams &launch_params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX8Backend::launchDirectRays): Backend not initialized.");
    }
    if (gas_handle == 0) {
        helios_runtime_error("ERROR (OptiX8Backend::launchDirectRays): No acceleration structure. Call buildAccelerationStructure() first.");
    }

    applyLaunchParams(launch_params);

    // Upload band_launch_flag (vector<bool> → device bool array)
    if (d_band_launch_flag) { freeCUdeviceptr(d_band_launch_flag); }
    const size_t Nbands_g = launch_params.band_launch_flag.size();
    if (Nbands_g > 0) {
        std::vector<uint8_t> flags_u8(Nbands_g);
        for (size_t i = 0; i < Nbands_g; i++) {
            flags_u8[i] = launch_params.band_launch_flag[i] ? 1u : 0u;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_band_launch_flag), Nbands_g * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_band_launch_flag), flags_u8.data(),
                              Nbands_g * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
    h_params.band_launch_flag = reinterpret_cast<bool *>(d_band_launch_flag);
    h_params.traversable      = gas_handle;

    // 2D stratification: split rays_per_primitive into dim_x × dim_y grid (same as diffuse)
    // rays_per_primitive is always n*n (RadiationModel sets it as ceil(sqrt(N))^2)
    const uint32_t launch_count  = launch_params.launch_count;
    const uint32_t rpp   = launch_params.rays_per_primitive;
    const uint32_t dim_x = static_cast<uint32_t>(sqrtf(static_cast<float>(rpp)));
    const uint32_t dim_y = (dim_x > 0) ? (rpp / dim_x) : 1u;
    h_params.launch_dim_x        = dim_x;
    h_params.launch_dim_y        = dim_y;

    // prd_pool is unused (PRD allocated on thread stack in raygen)
    h_params.prd_pool = nullptr;

    // Direct launch uses raygen record 0 (d_raygen_records + 0)
    OptixShaderBindingTable direct_sbt = sbt;
    direct_sbt.raygenRecord = d_raygen_records;

    // OptiX depth dimension is limited to 65535; batch if needed
    const uint32_t MAX_DEPTH = 65535u;
    uint32_t offset = launch_params.launch_offset;
    uint32_t remaining = launch_count;
    while (remaining > 0) {
        const uint32_t batch = std::min(remaining, MAX_DEPTH);
        h_params.launch_offset = offset;
        h_params.launch_count  = batch;
        uploadLaunchParams();
        OPTIX_CHECK(optixLaunch(optix_pipeline, cuda_stream, d_params,
                                sizeof(OptiX8LaunchParams), &direct_sbt,
                                dim_x, dim_y, batch));
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        offset    += batch;
        remaining -= batch;
    }
}

void OptiX8Backend::launchDiffuseRays(const RayTracingLaunchParams &launch_params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX8Backend::launchDiffuseRays): Backend not initialized.");
    }
    if (gas_handle == 0) {
        helios_runtime_error("ERROR (OptiX8Backend::launchDiffuseRays): No acceleration structure. "
                             "Call buildAccelerationStructure() first.");
    }

    applyLaunchParams(launch_params);

    // Upload radiation_out buffers (emission + scattered energy from previous iteration).
    // This must happen here because RadiationModel adds emission to flux_top/bottom AFTER
    // calling uploadRadiationOut() for the direct-ray scatter, so the params always carry
    // the most up-to-date radiation_out data.
    if (!launch_params.radiation_out_top.empty() && d_radiation_out_top) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_radiation_out_top),
                              launch_params.radiation_out_top.data(),
                              launch_params.radiation_out_top.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    if (!launch_params.radiation_out_bottom.empty() && d_radiation_out_bottom) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_radiation_out_bottom),
                              launch_params.radiation_out_bottom.data(),
                              launch_params.radiation_out_bottom.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Upload band_launch_flag
    if (d_band_launch_flag) { freeCUdeviceptr(d_band_launch_flag); }
    const size_t Nbands_g = launch_params.band_launch_flag.size();
    if (Nbands_g > 0) {
        std::vector<uint8_t> flags_u8(Nbands_g);
        for (size_t i = 0; i < Nbands_g; i++) {
            flags_u8[i] = launch_params.band_launch_flag[i] ? 1u : 0u;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_band_launch_flag),
                              Nbands_g * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_band_launch_flag), flags_u8.data(),
                              Nbands_g * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
    h_params.band_launch_flag = reinterpret_cast<bool *>(d_band_launch_flag);
    h_params.traversable      = gas_handle;

    // Upload diffuse params from launch_params (re-upload each launch since params may vary)
    freeCUdeviceptr(d_diffuse_flux);
    freeCUdeviceptr(d_diffuse_extinction);
    freeCUdeviceptr(d_diffuse_peak_dir);
    freeCUdeviceptr(d_diffuse_dist_norm);
    freeCUdeviceptr(d_sky_radiance_params);

    auto upload_f = [this](CUdeviceptr &ptr, const std::vector<float> &v) {
        if (!v.empty()) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), v.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(ptr), v.data(),
                                  v.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    };
    upload_f(d_diffuse_flux,       launch_params.diffuse_flux);
    upload_f(d_diffuse_extinction, launch_params.diffuse_extinction);
    upload_f(d_diffuse_dist_norm,  launch_params.diffuse_dist_norm);

    if (!launch_params.diffuse_peak_dir.empty()) {
        std::vector<float3> pd_f3(launch_params.diffuse_peak_dir.size());
        for (size_t i = 0; i < launch_params.diffuse_peak_dir.size(); i++) {
            pd_f3[i] = make_float3(launch_params.diffuse_peak_dir[i].x,
                                   launch_params.diffuse_peak_dir[i].y,
                                   launch_params.diffuse_peak_dir[i].z);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_diffuse_peak_dir),
                              pd_f3.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_diffuse_peak_dir), pd_f3.data(),
                              pd_f3.size() * sizeof(float3), cudaMemcpyHostToDevice));
    }

    if (!launch_params.sky_radiance_params.empty()) {
        std::vector<float4> sky_f4(launch_params.sky_radiance_params.size());
        for (size_t i = 0; i < launch_params.sky_radiance_params.size(); i++) {
            sky_f4[i] = make_float4(launch_params.sky_radiance_params[i].x,
                                    launch_params.sky_radiance_params[i].y,
                                    launch_params.sky_radiance_params[i].z,
                                    launch_params.sky_radiance_params[i].w);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_sky_radiance_params),
                              sky_f4.size() * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_sky_radiance_params), sky_f4.data(),
                              sky_f4.size() * sizeof(float4), cudaMemcpyHostToDevice));
    }

    h_params.diffuse_flux        = reinterpret_cast<float *>(d_diffuse_flux);
    h_params.diffuse_extinction  = reinterpret_cast<float *>(d_diffuse_extinction);
    h_params.diffuse_peak_dir    = reinterpret_cast<float3 *>(d_diffuse_peak_dir);
    h_params.diffuse_dist_norm   = reinterpret_cast<float *>(d_diffuse_dist_norm);
    h_params.sky_radiance_params = reinterpret_cast<float4 *>(d_sky_radiance_params);

    // Early return when there are no rays to launch (e.g. diffuseRayCount=0 during scattering)
    if (launch_params.rays_per_primitive == 0 || launch_params.launch_count == 0) {
        return;
    }

    // Set up 2D stratification launch dimensions
    // rays_per_primitive is always n*n (RadiationModel sets it as ceil(sqrt(N))^2)
    const uint32_t rpp   = launch_params.rays_per_primitive;
    const uint32_t dim_x = static_cast<uint32_t>(sqrtf(static_cast<float>(rpp)));
    const uint32_t dim_y = (dim_x > 0) ? (rpp / dim_x) : 1u;
    h_params.launch_dim_x = dim_x;
    h_params.launch_dim_y = dim_y;
    h_params.prd_pool     = nullptr;

    // Use diffuse raygen record (index 1 in the raygen records array)
    OptixShaderBindingTable diffuse_sbt = sbt;
    diffuse_sbt.raygenRecord = d_raygen_record_diffuse;

    // OptiX depth dimension is limited to 65535; batch if needed
    const uint32_t MAX_DEPTH    = 65535u;
    const uint32_t launch_count = launch_params.launch_count;
    uint32_t offset    = launch_params.launch_offset;
    uint32_t remaining = launch_count;
    while (remaining > 0) {
        const uint32_t batch = std::min(remaining, MAX_DEPTH);
        h_params.launch_offset = offset;
        h_params.launch_count  = batch;
        uploadLaunchParams();
        OPTIX_CHECK(optixLaunch(optix_pipeline, cuda_stream, d_params,
                                sizeof(OptiX8LaunchParams), &diffuse_sbt,
                                dim_x, dim_y, batch));
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        offset    += batch;
        remaining -= batch;
    }
}

void OptiX8Backend::launchCameraRays(const RayTracingLaunchParams &launch_params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX8Backend::launchCameraRays): Backend not initialized.");
    }
    if (gas_handle == 0) {
        helios_runtime_error("ERROR (OptiX8Backend::launchCameraRays): No acceleration structure. "
                             "Call buildAccelerationStructure() first.");
    }

    applyLaunchParams(launch_params);

    const uint32_t tile_w       = launch_params.camera_resolution.x;
    const uint32_t tile_h       = launch_params.camera_resolution.y;
    const uint32_t full_w       = launch_params.camera_resolution_full.x;
    const uint32_t full_h       = launch_params.camera_resolution_full.y;
    const uint32_t anti_samples = launch_params.antialiasing_samples;
    const uint32_t Nbands_l     = launch_params.num_bands_launch;
    const uint32_t cam_id       = launch_params.camera_id;

    // Allocate/zero radiation_in_camera when starting a new camera or when band count changes.
    // Multiple tiles for the same camera accumulate into the same buffer without re-zeroing.
    const size_t Npixels   = (size_t)full_w * full_h;
    const size_t cam_bytes = Npixels * Nbands_l * sizeof(float);
    if (cam_id != current_camera_launch_id || current_launch_band_count != Nbands_l) {
        reallocDevice(d_radiation_in_camera, cam_bytes);
        CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_in_camera), 0, cam_bytes));
        // Free pixel label/depth buffers from previous camera so getCameraResults skips them
        // until zeroCameraPixelBuffers() is called for the new camera.
        freeCUdeviceptr(d_camera_pixel_label);
        freeCUdeviceptr(d_camera_pixel_depth);
        h_params.camera_pixel_label = nullptr;
        h_params.camera_pixel_depth = nullptr;
        current_camera_launch_id  = cam_id;
        current_launch_band_count = Nbands_l;
    }
    h_params.radiation_in_camera = reinterpret_cast<float *>(d_radiation_in_camera);

    // Upload band_launch_flag
    if (d_band_launch_flag) { freeCUdeviceptr(d_band_launch_flag); }
    const size_t Nbands_g = launch_params.band_launch_flag.size();
    if (Nbands_g > 0) {
        std::vector<uint8_t> flags_u8(Nbands_g);
        for (size_t i = 0; i < Nbands_g; i++) {
            flags_u8[i] = launch_params.band_launch_flag[i] ? 1u : 0u;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_band_launch_flag), Nbands_g * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_band_launch_flag), flags_u8.data(),
                              Nbands_g * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
    h_params.band_launch_flag = reinterpret_cast<bool *>(d_band_launch_flag);
    h_params.traversable      = gas_handle;

    // Camera launch: x=antialiasing_samples, y=tile_width, z=tile_height
    h_params.launch_dim_x = anti_samples;
    h_params.launch_dim_y = tile_w;

    uploadLaunchParams();

    OptixShaderBindingTable camera_sbt = sbt;
    camera_sbt.raygenRecord = d_raygen_record_camera;

    OPTIX_CHECK(optixLaunch(optix_pipeline, cuda_stream, d_params,
                            sizeof(OptiX8LaunchParams), &camera_sbt,
                            anti_samples, tile_w, tile_h));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void OptiX8Backend::launchPixelLabelRays(const RayTracingLaunchParams &launch_params) {
    if (!is_initialized) {
        helios_runtime_error("ERROR (OptiX8Backend::launchPixelLabelRays): Backend not initialized.");
    }
    if (gas_handle == 0) {
        helios_runtime_error("ERROR (OptiX8Backend::launchPixelLabelRays): No acceleration structure. "
                             "Call buildAccelerationStructure() first.");
    }

    applyLaunchParams(launch_params);

    const uint32_t tile_w = launch_params.camera_resolution.x;
    const uint32_t tile_h = launch_params.camera_resolution.y;

    // Upload band_launch_flag
    if (d_band_launch_flag) { freeCUdeviceptr(d_band_launch_flag); }
    const size_t Nbands_g = launch_params.band_launch_flag.size();
    if (Nbands_g > 0) {
        std::vector<uint8_t> flags_u8(Nbands_g);
        for (size_t i = 0; i < Nbands_g; i++) {
            flags_u8[i] = launch_params.band_launch_flag[i] ? 1u : 0u;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_band_launch_flag), Nbands_g * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_band_launch_flag), flags_u8.data(),
                              Nbands_g * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }
    h_params.band_launch_flag = reinterpret_cast<bool *>(d_band_launch_flag);
    h_params.traversable      = gas_handle;

    // Pixel label launch: 1 ray per pixel center, no antialiasing
    h_params.launch_dim_x = 1u;
    h_params.launch_dim_y = tile_w;

    uploadLaunchParams();

    OptixShaderBindingTable pixel_label_sbt = sbt;
    pixel_label_sbt.raygenRecord = d_raygen_record_pixel_label;

    OPTIX_CHECK(optixLaunch(optix_pipeline, cuda_stream, d_params,
                            sizeof(OptiX8LaunchParams), &pixel_label_sbt,
                            1u, tile_w, tile_h));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

// ---------------------------------------------------------------------------
// Results retrieval
// ---------------------------------------------------------------------------

void OptiX8Backend::getRadiationResults(RayTracingResults &results) {
    const size_t Nprims = current_primitive_count;
    const size_t Nbands = current_band_count;
    const size_t total  = Nprims * Nbands;

    results.num_primitives = Nprims;
    results.num_bands      = Nbands;
    results.num_sources    = current_source_count;
    results.num_cameras    = current_camera_count;

    if (total > 0) {
        if (d_radiation_in)         results.radiation_in         = downloadFloat(d_radiation_in,         total);
        if (d_radiation_out_top)    results.radiation_out_top    = downloadFloat(d_radiation_out_top,    total);
        if (d_radiation_out_bottom) results.radiation_out_bottom = downloadFloat(d_radiation_out_bottom, total);
        if (d_scatter_buff_top)     results.scatter_buff_top     = downloadFloat(d_scatter_buff_top,     total);
        if (d_scatter_buff_bottom)  results.scatter_buff_bottom  = downloadFloat(d_scatter_buff_bottom,  total);
    }

    // Camera scatter buffers: sized Nprims × Nbands_launch (may differ from Nbands_global)
    if (Nprims > 0 && current_launch_band_count > 0) {
        const size_t cam_total = Nprims * current_launch_band_count;
        if (d_scatter_buff_top_cam)    results.scatter_buff_top_cam    = downloadFloat(d_scatter_buff_top_cam,    cam_total);
        if (d_scatter_buff_bottom_cam) results.scatter_buff_bottom_cam = downloadFloat(d_scatter_buff_bottom_cam, cam_total);
    }
}

void OptiX8Backend::getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels,
                                      std::vector<float> &pixel_depths, uint camera_id,
                                      const helios::int2 &resolution) {
    const size_t Npixels = (size_t)resolution.x * resolution.y;

    if (d_radiation_in_camera && Npixels > 0 && current_launch_band_count > 0) {
        pixel_data = downloadFloat(d_radiation_in_camera, Npixels * current_launch_band_count);
    }

    if (d_camera_pixel_label && Npixels > 0) {
        auto labels_u32 = downloadUInt32(d_camera_pixel_label, Npixels);
        pixel_labels.assign(labels_u32.begin(), labels_u32.end());
    }

    if (d_camera_pixel_depth && Npixels > 0) {
        pixel_depths = downloadFloat(d_camera_pixel_depth, Npixels);
    }
}

// ---------------------------------------------------------------------------
// Buffer management utilities
// ---------------------------------------------------------------------------

void OptiX8Backend::zeroRadiationBuffers(size_t launch_band_count) {
    const size_t Nprims = current_primitive_count;
    if (Nprims == 0 || launch_band_count == 0) return;

    if (launch_band_count > current_band_count) {
        helios_runtime_error("ERROR (OptiX8Backend::zeroRadiationBuffers): launch_band_count (" +
                             std::to_string(launch_band_count) + ") exceeds current_band_count (" +
                             std::to_string(current_band_count) + "). Call updateMaterials() first.");
    }

    // Reset camera launch ID so camera pixel buffers get re-zeroed in launchCameraRays()
    current_camera_launch_id = 0xFFFFFFFFu;

    const size_t bytes = Nprims * launch_band_count * sizeof(float);

    if (d_radiation_in)         CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_in),         0, bytes));
    if (d_radiation_out_top)    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_out_top),    0, bytes));
    if (d_radiation_out_bottom) CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_out_bottom), 0, bytes));
    if (d_scatter_buff_top)     CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_top),     0, bytes));
    if (d_scatter_buff_bottom)  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_bottom),  0, bytes));

    // Zero specular buffer: [source × camera × primitive × launch_band_count]
    const size_t specular_size = current_source_count * current_camera_count * Nprims * launch_band_count;
    if (specular_size > 0) {
        const size_t specular_bytes = specular_size * sizeof(float);
        reallocDevice(d_radiation_specular, specular_bytes);
        CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_specular), 0, specular_bytes));
        h_params.radiation_specular = reinterpret_cast<float *>(d_radiation_specular);
    }
}

void OptiX8Backend::zeroScatterBuffers() {
    const size_t total_bytes = current_primitive_count * current_band_count * sizeof(float);
    if (total_bytes == 0) return;

    if (d_scatter_buff_top)    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_top),    0, total_bytes));
    if (d_scatter_buff_bottom) CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_bottom), 0, total_bytes));
}

void OptiX8Backend::zeroCameraPixelBuffers(const helios::int2 &resolution) {
    const size_t Npixels = (size_t)resolution.x * resolution.y;
    if (Npixels == 0) return;

    reallocDevice(d_camera_pixel_label, Npixels * sizeof(uint32_t));
    reallocDevice(d_camera_pixel_depth,  Npixels * sizeof(float));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_camera_pixel_label), 0, Npixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_camera_pixel_depth),  0, Npixels * sizeof(float)));

    h_params.camera_pixel_label = reinterpret_cast<uint32_t *>(d_camera_pixel_label);
    h_params.camera_pixel_depth = reinterpret_cast<float *>(d_camera_pixel_depth);
}

void OptiX8Backend::copyScatterToRadiation() {
    const size_t total = current_primitive_count * current_band_count * sizeof(float);
    if (total == 0) return;
    if (d_scatter_buff_top && d_radiation_out_top) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_radiation_out_top),
                              reinterpret_cast<const void *>(d_scatter_buff_top),
                              total, cudaMemcpyDeviceToDevice));
    }
    if (d_scatter_buff_bottom && d_radiation_out_bottom) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_radiation_out_bottom),
                              reinterpret_cast<const void *>(d_scatter_buff_bottom),
                              total, cudaMemcpyDeviceToDevice));
    }
}

void OptiX8Backend::uploadRadiationOut(const std::vector<float> &radiation_out_top,
                                        const std::vector<float> &radiation_out_bottom) {
    if (!radiation_out_top.empty() && d_radiation_out_top) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_radiation_out_top),
                              radiation_out_top.data(),
                              radiation_out_top.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    if (!radiation_out_bottom.empty() && d_radiation_out_bottom) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_radiation_out_bottom),
                              radiation_out_bottom.data(),
                              radiation_out_bottom.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

void OptiX8Backend::uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam,
                                                const std::vector<float> &scatter_bottom_cam) {
    if (!scatter_top_cam.empty() && d_scatter_buff_top_cam) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_scatter_buff_top_cam),
                              scatter_top_cam.data(),
                              scatter_top_cam.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    if (!scatter_bottom_cam.empty() && d_scatter_buff_bottom_cam) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_scatter_buff_bottom_cam),
                              scatter_bottom_cam.data(),
                              scatter_bottom_cam.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

void OptiX8Backend::zeroCameraScatterBuffers(size_t launch_band_count) {
    const size_t Nprims = current_primitive_count;
    if (Nprims == 0 || launch_band_count == 0) return;

    const size_t bytes = Nprims * launch_band_count * sizeof(float);
    reallocDevice(d_scatter_buff_top_cam,    bytes);
    reallocDevice(d_scatter_buff_bottom_cam, bytes);
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_top_cam),    0, bytes));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_bottom_cam), 0, bytes));

    h_params.scatter_buff_top_cam    = reinterpret_cast<float *>(d_scatter_buff_top_cam);
    h_params.scatter_buff_bottom_cam = reinterpret_cast<float *>(d_scatter_buff_bottom_cam);
    current_launch_band_count        = launch_band_count;
}

void OptiX8Backend::uploadSourceFluxes(const std::vector<float> &fluxes) {
    freeCUdeviceptr(d_source_fluxes);
    if (fluxes.empty()) return;

    const size_t bytes = fluxes.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_source_fluxes), bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_source_fluxes), fluxes.data(), bytes, cudaMemcpyHostToDevice));
    h_params.source_fluxes = reinterpret_cast<float *>(d_source_fluxes);
}

void OptiX8Backend::uploadSourceFluxesCam(const std::vector<float> &fluxes_cam) {
    freeCUdeviceptr(d_source_fluxes_cam);
    if (fluxes_cam.empty()) return;

    const size_t bytes = fluxes_cam.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_source_fluxes_cam), bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_source_fluxes_cam), fluxes_cam.data(), bytes,
                          cudaMemcpyHostToDevice));
    h_params.source_fluxes_cam = reinterpret_cast<float *>(d_source_fluxes_cam);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

void OptiX8Backend::queryGPUMemory() const {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const float free_mb  = static_cast<float>(free_bytes)  / (1024.0f * 1024.0f);
    const float total_mb = static_cast<float>(total_bytes) / (1024.0f * 1024.0f);
    std::cout << "GPU memory: " << free_mb << " MB free / " << total_mb << " MB total" << std::endl;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void OptiX8Backend::freeCUdeviceptr(CUdeviceptr &ptr) {
    if (ptr) {
        cudaFree(reinterpret_cast<void *>(ptr));
        ptr = 0;
    }
}

void OptiX8Backend::freeGeometryBuffers() {
    freeCUdeviceptr(d_transform_matrix);
    freeCUdeviceptr(d_primitive_type);
    freeCUdeviceptr(d_primitive_positions);
    freeCUdeviceptr(d_primitive_uuid_arr);
    freeCUdeviceptr(d_primitiveID);
    freeCUdeviceptr(d_objectID);
    freeCUdeviceptr(d_object_subdivisions);
    freeCUdeviceptr(d_twosided_flag);
    freeCUdeviceptr(d_primitive_solid_fraction);
    freeCUdeviceptr(d_patch_vertices);
    freeCUdeviceptr(d_patch_UUIDs);
    freeCUdeviceptr(d_triangle_vertices);
    freeCUdeviceptr(d_triangle_UUIDs);
    freeCUdeviceptr(d_disk_centers);
    freeCUdeviceptr(d_disk_radii);
    freeCUdeviceptr(d_disk_normals);
    freeCUdeviceptr(d_disk_UUIDs);
    freeCUdeviceptr(d_tile_vertices);
    freeCUdeviceptr(d_tile_UUIDs);
    freeCUdeviceptr(d_voxel_vertices);
    freeCUdeviceptr(d_voxel_UUIDs);
    freeCUdeviceptr(d_bbox_vertices);
    freeCUdeviceptr(d_bbox_UUIDs);
    freeCUdeviceptr(d_mask_data);
    freeCUdeviceptr(d_mask_offsets);
    freeCUdeviceptr(d_mask_sizes);
    freeCUdeviceptr(d_mask_IDs);
    freeCUdeviceptr(d_uv_data);
    freeCUdeviceptr(d_uv_IDs);
    freeCUdeviceptr(d_aabbs);
    freeCUdeviceptr(d_gas_output);
    gas_handle = 0;
}

void OptiX8Backend::freeMaterialBuffers() {
    freeCUdeviceptr(d_rho);
    freeCUdeviceptr(d_tau);
    freeCUdeviceptr(d_rho_cam);
    freeCUdeviceptr(d_tau_cam);
    freeCUdeviceptr(d_specular_exponent);
    freeCUdeviceptr(d_specular_scale);
}

void OptiX8Backend::reallocDevice(CUdeviceptr &ptr, size_t bytes) {
    freeCUdeviceptr(ptr);
    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), bytes));
    }
}

std::vector<float> OptiX8Backend::downloadFloat(CUdeviceptr ptr, size_t count) const {
    std::vector<float> result(count);
    CUDA_CHECK(cudaMemcpy(result.data(), reinterpret_cast<const void *>(ptr),
                          count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

std::vector<uint32_t> OptiX8Backend::downloadUInt32(CUdeviceptr ptr, size_t count) const {
    std::vector<uint32_t> result(count);
    CUDA_CHECK(cudaMemcpy(result.data(), reinterpret_cast<const void *>(ptr),
                          count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return result;
}

void OptiX8Backend::buildAABBs(const RayTracingGeometry &geometry) {
    const uint32_t Nprims  = static_cast<uint32_t>(geometry.primitive_count);
    const uint32_t Nbboxes = static_cast<uint32_t>(geometry.bboxes.UUIDs.size());
    const uint32_t Ntotal  = Nprims + Nbboxes;
    if (Ntotal == 0) return;

    std::vector<OptixAabb> aabbs(Ntotal);
    const float eps = 1e-5f;

    auto store_aabb = [&](uint32_t pos, float mn_x, float mn_y, float mn_z,
                                        float mx_x, float mx_y, float mx_z) {
        if (mx_x - mn_x < eps) { mn_x -= eps; mx_x += eps; }
        if (mx_y - mn_y < eps) { mn_y -= eps; mx_y += eps; }
        if (mx_z - mn_z < eps) { mn_z -= eps; mx_z += eps; }
        aabbs[pos] = {mn_x, mn_y, mn_z, mx_x, mx_y, mx_z};
    };

    // Real primitives — AABB computed via canonical-space vertices + transform matrix
    for (uint32_t pos = 0; pos < Nprims; pos++) {
        const float *T    = &geometry.transform_matrices[pos * 16];
        const uint32_t pt = geometry.primitive_types[pos];

        float mn_x = FLT_MAX, mn_y = FLT_MAX, mn_z = FLT_MAX;
        float mx_x = -FLT_MAX, mx_y = -FLT_MAX, mx_z = -FLT_MAX;

        auto expand = [&](float x, float y, float z) {
            const float wx = T[0]*x + T[1]*y + T[2]*z + T[3];
            const float wy = T[4]*x + T[5]*y + T[6]*z + T[7];
            const float wz = T[8]*x + T[9]*y + T[10]*z + T[11];
            if (wx < mn_x) mn_x = wx; if (wx > mx_x) mx_x = wx;
            if (wy < mn_y) mn_y = wy; if (wy > mx_y) mx_y = wy;
            if (wz < mn_z) mn_z = wz; if (wz > mx_z) mx_z = wz;
        };

        if (pt == 0 || pt == 3) { // Patch or Tile: canonical space [-0.5, 0.5]^2
            expand(-0.5f, -0.5f, 0.f); expand( 0.5f, -0.5f, 0.f);
            expand(-0.5f,  0.5f, 0.f); expand( 0.5f,  0.5f, 0.f);
        } else if (pt == 1) { // Triangle: (0,0,0)-(0,1,0)-(1,1,0)
            expand(0.f, 0.f, 0.f); expand(0.f, 1.f, 0.f); expand(1.f, 1.f, 0.f);
        } else if (pt == 2) { // Disk: bounding box of unit circle
            expand(-0.5f, -0.5f, 0.f); expand( 0.5f, -0.5f, 0.f);
            expand(-0.5f,  0.5f, 0.f); expand( 0.5f,  0.5f, 0.f);
        } else if (pt == 4) { // Voxel: unit cube [0,1]^3
            for (float fx : {0.f, 1.f})
                for (float fy : {0.f, 1.f})
                    for (float fz : {0.f, 1.f})
                        expand(fx, fy, fz);
        } else { // Unknown type: unit box fallback
            expand(-0.5f, -0.5f, -0.5f); expand( 0.5f,  0.5f,  0.5f);
        }

        store_aabb(pos, mn_x, mn_y, mn_z, mx_x, mx_y, mx_z);
    }

    // Bbox faces: AABB from actual world-space vertices (4 vertices per face)
    for (uint32_t b = 0; b < Nbboxes; b++) {
        float mn_x = FLT_MAX, mn_y = FLT_MAX, mn_z = FLT_MAX;
        float mx_x = -FLT_MAX, mx_y = -FLT_MAX, mx_z = -FLT_MAX;
        for (int v = 0; v < 4; v++) {
            const helios::vec3 &vtx = geometry.bboxes.vertices[b * 4 + v];
            mn_x = std::min(mn_x, vtx.x); mx_x = std::max(mx_x, vtx.x);
            mn_y = std::min(mn_y, vtx.y); mx_y = std::max(mx_y, vtx.y);
            mn_z = std::min(mn_z, vtx.z); mx_z = std::max(mx_z, vtx.z);
        }
        store_aabb(Nprims + b, mn_x, mn_y, mn_z, mx_x, mx_y, mx_z);
    }

    const size_t aabb_bytes = Ntotal * sizeof(OptixAabb);
    reallocDevice(d_aabbs, aabb_bytes);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_aabbs), aabbs.data(), aabb_bytes, cudaMemcpyHostToDevice));
}

void OptiX8Backend::buildGAS(uint32_t Nprimitives) {
    if (d_gas_output) { cudaFree(reinterpret_cast<void *>(d_gas_output)); d_gas_output = 0; }
    gas_handle = 0;

    if (Nprimitives == 0 || !d_aabbs) return;

    const unsigned int build_flags[]       = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInputCustomPrimitiveArray ca = {};
    ca.aabbBuffers    = &d_aabbs;
    ca.numPrimitives  = Nprimitives;
    ca.strideInBytes  = sizeof(OptixAabb);
    ca.flags          = build_flags;
    ca.numSbtRecords  = 1;

    OptixBuildInput bi = {};
    bi.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray    = ca;

    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context, &opts, &bi, 1, &sizes));

    CUdeviceptr d_temp = 0, d_pre_compact = 0, d_compact_size = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp),         sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_pre_compact),  sizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_compact_size), sizeof(uint64_t)));

    OptixAccelEmitDesc emit = {};
    emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit.result = d_compact_size;

    OPTIX_CHECK(optixAccelBuild(optix_context, cuda_stream, &opts, &bi, 1,
                                d_temp, sizes.tempSizeInBytes,
                                d_pre_compact, sizes.outputSizeInBytes,
                                &gas_handle, &emit, 1));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    uint64_t compact_size = 0;
    CUDA_CHECK(cudaMemcpy(&compact_size, reinterpret_cast<const void *>(d_compact_size),
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output), compact_size));
    OPTIX_CHECK(optixAccelCompact(optix_context, cuda_stream, gas_handle, d_gas_output, compact_size, &gas_handle));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_pre_compact)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_compact_size)));

    h_params.traversable = gas_handle;
}

void OptiX8Backend::buildSBT() {
    if (d_raygen_records)   { cudaFree(reinterpret_cast<void *>(d_raygen_records));   d_raygen_records   = 0; }
    if (d_miss_records)     { cudaFree(reinterpret_cast<void *>(d_miss_records));     d_miss_records     = 0; }
    if (d_hitgroup_records) { cudaFree(reinterpret_cast<void *>(d_hitgroup_records)); d_hitgroup_records = 0; }

    // Raygen records: header only (no data), 4 records (direct, diffuse, camera, pixel_label)
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    constexpr int N_rg = 4;
    std::vector<RaygenRecord> rg_recs(N_rg);
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_raygen_direct,      &rg_recs[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_raygen_diffuse,     &rg_recs[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_raygen_camera,      &rg_recs[2]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_raygen_pixel_label, &rg_recs[3]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_records), N_rg * sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygen_records), rg_recs.data(),
                          N_rg * sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    // Cache individual record device pointers (used to select raygen per launch type)
    d_raygen_record_direct      = d_raygen_records;
    d_raygen_record_diffuse     = d_raygen_records + 1 * sizeof(RaygenRecord);
    d_raygen_record_camera      = d_raygen_records + 2 * sizeof(RaygenRecord);
    d_raygen_record_pixel_label = d_raygen_records + 3 * sizeof(RaygenRecord);

    // Miss records: header only, 4 records
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    constexpr int N_ms = 4;
    std::vector<MissRecord> ms_recs(N_ms);
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_miss_direct,      &ms_recs[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_miss_diffuse,     &ms_recs[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_miss_camera,      &ms_recs[2]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_miss_pixel_label, &ms_recs[3]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_records), N_ms * sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_miss_records), ms_recs.data(),
                          N_ms * sizeof(MissRecord), cudaMemcpyHostToDevice));

    // Hit group records: header + HitGroupData, 4 records (one per ray type)
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        HitGroupData data;
    };
    constexpr int N_hg = 4;
    std::vector<HitRecord> hg_recs(N_hg);
    for (auto &r : hg_recs) {
        r.data.vertices  = reinterpret_cast<float3 *>(d_patch_vertices);
        r.data.UUIDs     = reinterpret_cast<uint32_t *>(d_patch_UUIDs);
        r.data.prim_type = 0; // patch
    }
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_hit_direct,      &hg_recs[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_hit_diffuse,     &hg_recs[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_hit_camera,      &hg_recs[2]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pg_hit_pixel_label, &hg_recs[3]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitgroup_records), N_hg * sizeof(HitRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitgroup_records), hg_recs.data(),
                          N_hg * sizeof(HitRecord), cudaMemcpyHostToDevice));

    sbt = {};
    sbt.raygenRecord                = d_raygen_records; // updated per-launch type
    sbt.missRecordBase              = d_miss_records;
    sbt.missRecordStrideInBytes     = static_cast<uint32_t>(sizeof(MissRecord));
    sbt.missRecordCount             = N_ms;
    sbt.hitgroupRecordBase          = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitRecord));
    sbt.hitgroupRecordCount         = N_hg;
}

void OptiX8Backend::uploadLaunchParams() {
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(d_params), &h_params,
                               sizeof(OptiX8LaunchParams), cudaMemcpyHostToDevice, cuda_stream));
}

void OptiX8Backend::applyLaunchParams(const RayTracingLaunchParams &params) {
    h_params.launch_offset          = params.launch_offset;
    h_params.launch_count           = params.launch_count;
    h_params.rays_per_primitive     = params.rays_per_primitive;
    h_params.random_seed            = params.random_seed;
    h_params.Nbands_global          = params.num_bands_global;
    h_params.Nbands_launch          = params.num_bands_launch;
    h_params.launch_face            = params.launch_face;
    h_params.scattering_iteration   = params.scattering_iteration;
    h_params.specular_reflection_enabled = params.specular_reflection_enabled;
    h_params.camera_ID              = params.camera_id;
    h_params.camera_position        = make_float3(params.camera_position.x, params.camera_position.y, params.camera_position.z);
    h_params.camera_direction       = make_float2(params.camera_direction.x, params.camera_direction.y);
    h_params.camera_focal_length    = params.camera_focal_length;
    h_params.camera_lens_diameter   = params.camera_lens_diameter;
    h_params.FOV_aspect_ratio       = params.camera_fov_aspect;
    h_params.camera_HFOV            = params.camera_HFOV;
    h_params.camera_resolution.x      = params.camera_resolution.x;
    h_params.camera_resolution.y      = params.camera_resolution.y;
    h_params.camera_viewplane_length= params.camera_viewplane_length;
    h_params.camera_pixel_solid_angle = params.camera_pixel_solid_angle;
    h_params.camera_pixel_offset.x    = params.camera_pixel_offset.x;
    h_params.camera_pixel_offset.y    = params.camera_pixel_offset.y;
    h_params.camera_resolution_full.x = params.camera_resolution_full.x;
    h_params.camera_resolution_full.y = params.camera_resolution_full.y;
}

std::string OptiX8Backend::findDeviceCodeFile() const {
    // OptiX 8 uses either PTX or OptixIR (.optixir)
    // Use the non-throwing resolver so we can probe each candidate in order
    const std::vector<std::string> candidate_names = {
        "OptiX8DeviceCode.optixir",
        "OptiX8DeviceCode.ptx",
    };
    for (const auto &name : candidate_names) {
        auto path = helios::tryResolveFilePath("plugins/radiation/" + name);
        if (!path.empty()) {
            return path.string();
        }
    }
    helios_runtime_error(
        "ERROR (OptiX8Backend::findDeviceCodeFile): Could not find OptiX8DeviceCode.optixir or "
        "OptiX8DeviceCode.ptx in the radiation plugin asset directory. "
        "Ensure the radiation plugin was built with OptiX 8 support (HELIOS_HAVE_OPTIX8).");
    return ""; // unreachable
}

} // namespace helios
