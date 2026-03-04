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
    helios_runtime_error("ERROR (OptiX8Backend::updateGeometry): Not yet implemented.");
}

void OptiX8Backend::buildAccelerationStructure() {
    helios_runtime_error("ERROR (OptiX8Backend::buildAccelerationStructure): Not yet implemented.");
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

void OptiX8Backend::updateMaterials(const RayTracingMaterial &materials) {
    helios_runtime_error("ERROR (OptiX8Backend::updateMaterials): Not yet implemented.");
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

void OptiX8Backend::updateSources(const std::vector<RayTracingSource> &sources) {
    helios_runtime_error("ERROR (OptiX8Backend::updateSources): Not yet implemented.");
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

void OptiX8Backend::launchDirectRays(const RayTracingLaunchParams &params) {
    helios_runtime_error("ERROR (OptiX8Backend::launchDirectRays): Not yet implemented.");
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
    helios_runtime_error("ERROR (OptiX8Backend::getRadiationResults): Not yet implemented.");
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
    helios_runtime_error("ERROR (OptiX8Backend::zeroRadiationBuffers): Not yet implemented.");
}

void OptiX8Backend::zeroScatterBuffers() {
    helios_runtime_error("ERROR (OptiX8Backend::zeroScatterBuffers): Not yet implemented.");
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
    helios_runtime_error("ERROR (OptiX8Backend::uploadSourceFluxes): Not yet implemented.");
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
    helios_runtime_error("ERROR (OptiX8Backend::buildAABBs): Not yet implemented.");
}

void OptiX8Backend::buildGAS(uint32_t Nprimitives) {
    helios_runtime_error("ERROR (OptiX8Backend::buildGAS): Not yet implemented.");
}

void OptiX8Backend::buildSBT() {
    helios_runtime_error("ERROR (OptiX8Backend::buildSBT): Not yet implemented.");
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
    // Look for file in the same directory as the radiation plugin's PTX files
    const std::vector<std::string> candidates = {
        helios::resolvePluginAsset("radiation", "OptiX8DeviceCode.optixir").string(),
        helios::resolvePluginAsset("radiation", "OptiX8DeviceCode.ptx").string(),
    };
    for (const auto &path : candidates) {
        std::ifstream f(path);
        if (f.good()) {
            return path;
        }
    }
    helios_runtime_error(
        "ERROR (OptiX8Backend::findDeviceCodeFile): Could not find OptiX8DeviceCode.optixir or "
        "OptiX8DeviceCode.ptx in the radiation plugin asset directory. "
        "Ensure the radiation plugin was built with OptiX 8 support (HELIOS_HAVE_OPTIX8).");
    return ""; // unreachable
}

} // namespace helios
