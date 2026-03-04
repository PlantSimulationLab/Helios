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
    ctx_options.logCallbackLevel = 4;
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

    if (log_size > 1) {
        std::cerr << "[OptiX8 module log] " << log << "\n";
    }

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
        if (log_size > 1) { std::cerr << "[OptiX8 pg log] " << log << "\n"; }
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
        if (log_size > 1) { std::cerr << "[OptiX8 pg log] " << log << "\n"; }
    };

    createMiss("__miss__direct",       pg_miss_direct);
    createMiss("__miss__diffuse",      pg_miss_diffuse);
    createMiss("__miss__camera",       pg_miss_camera);
    createMiss("__miss__pixel_label",  pg_miss_pixel_label);

    // Hit groups: one per primitive type (patch=0, tri=1, disk=2, tile=3, voxel=4, bbox=5)
    // All 4 ray types share the same intersection programs but have different closest-hit programs.
    // We create 4 hit groups (one per ray type); the intersection program is on each.
    // With a single GAS, the SBT index = ray_type_index, so 4 hit groups suffice (ray type
    // is used as SBT offset). Each type shares all 4 closest-hit programs.
    //
    // Actually, with custom primitives in OptiX 8, the SBT stride/offset determines which
    // hit group record to use. We use: sbt_offset = primitive_type, sbt_stride = Nraytypes.
    // So we need Ntypes * Nraytypes = 6 * 4 = 24 hit group records.
    // For Phase 0, we create minimal hit groups (just patch type, all 4 ray types).

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
        if (log_size > 1) { std::cerr << "[OptiX8 pg log] " << log << "\n"; }
    };

    // For now create one hit group per ray type (using patch intersection).
    // Phase 6+ will expand to all primitive types.
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

    if (log_size > 1) {
        std::cerr << "[OptiX8 pipeline log] " << log << "\n";
    }

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
    freePtr(d_prd_pool);
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
    if (geometry.bboxes.count > 0) {
        upload(d_bbox_vertices, geometry.bboxes.vertices.data(),
               geometry.bboxes.vertices.size() * sizeof(helios::vec3));
        upload(d_bbox_UUIDs, geometry.bboxes.UUIDs.data(),
               geometry.bboxes.UUIDs.size() * sizeof(uint32_t));
    }

    // Update h_params device pointers
    const uint32_t Nprims = static_cast<uint32_t>(geometry.primitive_count);
    h_params.transform_matrix         = reinterpret_cast<float *>(d_transform_matrix);
    h_params.primitive_type           = reinterpret_cast<uint32_t *>(d_primitive_type);
    h_params.primitive_positions      = reinterpret_cast<uint32_t *>(d_primitive_positions);
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
    buildGAS(static_cast<uint32_t>(current_primitive_count));
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

    upload(d_rho, materials.reflectivity.data(),   materials.reflectivity.size()   * sizeof(float));
    upload(d_tau, materials.transmissivity.data(),  materials.transmissivity.size() * sizeof(float));
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
    helios_runtime_error("ERROR (OptiX8Backend::updateDiffuseRadiation): Not yet implemented.");
}

void OptiX8Backend::updateSkyModel(const std::vector<helios::vec4> &sky_radiance_params,
                                    const std::vector<float> &camera_sky_radiance,
                                    const helios::vec3 &sun_direction,
                                    const std::vector<float> &solar_disk_radiance,
                                    float solar_disk_cos_angle) {
    helios_runtime_error("ERROR (OptiX8Backend::updateSkyModel): Not yet implemented.");
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

    const uint32_t Nrays         = launch_params.rays_per_primitive;
    const uint32_t launch_count  = launch_params.launch_count;
    h_params.launch_dim_x        = Nrays;
    h_params.launch_dim_y        = 1;

    // prd_pool is unused (PRD allocated on thread stack in raygen)
    h_params.prd_pool = nullptr;

    uploadLaunchParams();
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    // Direct launch uses raygen record 0 (d_raygen_records + 0)
    OptixShaderBindingTable direct_sbt = sbt;
    direct_sbt.raygenRecord = d_raygen_records;

    OPTIX_CHECK(optixLaunch(
        optix_pipeline,
        cuda_stream,
        d_params,
        sizeof(OptiX8LaunchParams),
        &direct_sbt,
        Nrays,         // width
        1,             // height
        launch_count   // depth
    ));

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void OptiX8Backend::launchDiffuseRays(const RayTracingLaunchParams &params) {
    helios_runtime_error("ERROR (OptiX8Backend::launchDiffuseRays): Not yet implemented.");
}

void OptiX8Backend::launchCameraRays(const RayTracingLaunchParams &params) {
    helios_runtime_error("ERROR (OptiX8Backend::launchCameraRays): Not yet implemented.");
}

void OptiX8Backend::launchPixelLabelRays(const RayTracingLaunchParams &params) {
    helios_runtime_error("ERROR (OptiX8Backend::launchPixelLabelRays): Not yet implemented.");
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
}

void OptiX8Backend::getCameraResults(std::vector<float> &pixel_data, std::vector<uint> &pixel_labels,
                                      std::vector<float> &pixel_depths, uint camera_id,
                                      const helios::int2 &resolution) {
    helios_runtime_error("ERROR (OptiX8Backend::getCameraResults): Not yet implemented.");
}

// ---------------------------------------------------------------------------
// Buffer management utilities
// ---------------------------------------------------------------------------

void OptiX8Backend::zeroRadiationBuffers(size_t launch_band_count) {
    const size_t Nprims = current_primitive_count;
    if (Nprims == 0 || launch_band_count == 0) return;

    const size_t bands = std::min(launch_band_count, current_band_count);
    const size_t bytes = Nprims * bands * sizeof(float);

    if (d_radiation_in)         CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_in),         0, bytes));
    if (d_radiation_out_top)    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_out_top),    0, bytes));
    if (d_radiation_out_bottom) CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_radiation_out_bottom), 0, bytes));
    if (d_scatter_buff_top)     CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_top),     0, bytes));
    if (d_scatter_buff_bottom)  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_bottom),  0, bytes));
}

void OptiX8Backend::zeroScatterBuffers() {
    const size_t total_bytes = current_primitive_count * current_band_count * sizeof(float);
    if (total_bytes == 0) return;

    if (d_scatter_buff_top)    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_top),    0, total_bytes));
    if (d_scatter_buff_bottom) CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(d_scatter_buff_bottom), 0, total_bytes));
}

void OptiX8Backend::zeroCameraPixelBuffers(const helios::int2 &resolution) {
    helios_runtime_error("ERROR (OptiX8Backend::zeroCameraPixelBuffers): Not yet implemented.");
}

void OptiX8Backend::copyScatterToRadiation() {
    helios_runtime_error("ERROR (OptiX8Backend::copyScatterToRadiation): Not yet implemented.");
}

void OptiX8Backend::uploadRadiationOut(const std::vector<float> &radiation_out_top,
                                        const std::vector<float> &radiation_out_bottom) {
    helios_runtime_error("ERROR (OptiX8Backend::uploadRadiationOut): Not yet implemented.");
}

void OptiX8Backend::uploadCameraScatterBuffers(const std::vector<float> &scatter_top_cam,
                                                const std::vector<float> &scatter_bottom_cam) {
    helios_runtime_error("ERROR (OptiX8Backend::uploadCameraScatterBuffers): Not yet implemented.");
}

void OptiX8Backend::zeroCameraScatterBuffers(size_t launch_band_count) {
    helios_runtime_error("ERROR (OptiX8Backend::zeroCameraScatterBuffers): Not yet implemented.");
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
    helios_runtime_error("ERROR (OptiX8Backend::uploadSourceFluxesCam): Not yet implemented.");
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
    std::cout << "[OptiX8] GPU memory: " << free_mb << " MB free / " << total_mb << " MB total\n";
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
    const uint32_t Nprims = static_cast<uint32_t>(geometry.primitive_count);
    if (Nprims == 0) return;

    std::vector<OptixAabb> aabbs(Nprims);

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
        } else { // Tile (pt==3 above), Bbox, or unknown: unit box
            expand(-0.5f, -0.5f, -0.5f); expand( 0.5f,  0.5f,  0.5f);
        }

        // Pad thin AABBs so OptiX doesn't reject them
        const float eps = 1e-5f;
        if (mx_x - mn_x < eps) { mn_x -= eps; mx_x += eps; }
        if (mx_y - mn_y < eps) { mn_y -= eps; mx_y += eps; }
        if (mx_z - mn_z < eps) { mn_z -= eps; mx_z += eps; }

        aabbs[pos] = {mn_x, mn_y, mn_z, mx_x, mx_y, mx_z};
    }

    const size_t aabb_bytes = Nprims * sizeof(OptixAabb);
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

    cudaFree(reinterpret_cast<void *>(d_temp));
    cudaFree(reinterpret_cast<void *>(d_pre_compact));
    cudaFree(reinterpret_cast<void *>(d_compact_size));

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
