# Helios Radiation Plugin Ray-Tracing Migration Plan

## Executive Summary

This document outlines a comprehensive three-phase plan to modernize the Helios radiation plugin's ray-tracing infrastructure:

1. **Phase 1: Code Isolation** - Decouple OptiX-specific code to create a maintainable abstraction layer
2. **Phase 2: OptiX 7.7 Migration** - Migrate from OptiX 6.5 to OptiX 7.7 with bundled headers
3. **Phase 3: Vulkan Fallback** - Implement Vulkan ray-tracing as a cross-platform fallback

This migration is substantial but necessary for long-term maintainability, cross-platform support, and access to modern ray-tracing features.

---

## Current State Analysis

### Existing Architecture

The radiation plugin currently uses **OptiX 5.1.0/6.5.0** with:

- **~60+ RTbuffer objects** for geometry, materials, sources, and results
- **4 ray types**: direct, diffuse, camera, pixel_label
- **3 CUDA files** compiled to PTX: `rayGeneration.cu`, `primitiveIntersection.cu`, `rayHit.cu`
- **5 custom primitive types**: patch, triangle, disk, tile, voxel
- **No abstraction layer** between business logic and OptiX API

### Files Requiring Changes

| File | Lines | OptiX Coupling | Migration Impact |
|------|-------|----------------|------------------|
| `RadiationModel.h` | ~2575 | Heavy (~273 RT* types) | Major rewrite |
| `RadiationModel.cpp` | ~5000+ | Heavy | Major rewrite |
| `rayGeneration.cu` | ~656 | Complete | Complete rewrite |
| `primitiveIntersection.cu` | ~500+ | Complete | Complete rewrite |
| `rayHit.cu` | ~746 | Complete | Complete rewrite |
| `RayTracing.cuh` | ~807 | Heavy | Major rewrite |
| `CMakeLists.txt` | ~237 | Moderate | Moderate changes |

### Files Minimally Affected

- `RadiationCamera.cpp` - Pure C++ image processing
- `LensFlare.cpp` - CPU-based post-processing
- `CameraCalibration.cpp` - Calibration algorithms

---

## Phase 1: OptiX Code Isolation

### Objective

Create a clean abstraction layer that isolates OptiX-specific code, making it easier to implement Phase 2 (OptiX 7.7) and Phase 3 (Vulkan) backends without touching business logic.

### 1.1 Design the Backend Interface

Create an abstract interface that encapsulates all ray-tracing operations:

```cpp
// File: plugins/radiation/include/RayTracingBackend.h

#pragma once
#include <memory>
#include <vector>
#include <string>

namespace helios {

// Forward declarations
struct RayTracingGeometry;
struct RayTracingMaterial;
struct RayTracingSource;
struct RayTracingLaunchParams;

/**
 * @brief Abstract interface for ray-tracing backends
 *
 * Implementations: OptiX6Backend, OptiX7Backend, VulkanBackend
 */
class RayTracingBackend {
public:
    virtual ~RayTracingBackend() = default;

    // Lifecycle
    virtual void initialize() = 0;
    virtual void shutdown() = 0;

    // Geometry management
    virtual void updateGeometry(const RayTracingGeometry& geometry) = 0;
    virtual void buildAccelerationStructure() = 0;

    // Material/optical properties
    virtual void updateMaterials(const RayTracingMaterial& materials) = 0;

    // Radiation sources
    virtual void updateSources(const std::vector<RayTracingSource>& sources) = 0;

    // Ray launching
    virtual void launchDirectRays(const RayTracingLaunchParams& params) = 0;
    virtual void launchDiffuseRays(const RayTracingLaunchParams& params) = 0;
    virtual void launchCameraRays(const RayTracingLaunchParams& params) = 0;
    virtual void launchPixelLabelRays(const RayTracingLaunchParams& params) = 0;

    // Results retrieval
    virtual void getRadiationResults(std::vector<float>& radiation_in,
                                      std::vector<float>& radiation_out_top,
                                      std::vector<float>& radiation_out_bottom) = 0;
    virtual void getCameraResults(std::vector<float>& pixel_data,
                                   std::vector<uint>& pixel_labels,
                                   std::vector<float>& pixel_depths) = 0;

    // Buffer management
    virtual void syncBuffersToDevice() = 0;
    virtual void syncBuffersFromDevice() = 0;

    // Factory method
    static std::unique_ptr<RayTracingBackend> create(const std::string& backend_type);
};

} // namespace helios
```

### 1.2 Define Backend-Agnostic Data Structures

```cpp
// File: plugins/radiation/include/RayTracingTypes.h

#pragma once
#include <vector>
#include "Context.h"

namespace helios {

/**
 * @brief Geometry data for ray tracing (backend-agnostic)
 */
struct RayTracingGeometry {
    // Per-primitive data
    std::vector<float> transform_matrices;  // 16 floats per primitive
    std::vector<uint> primitive_types;      // 0=patch, 1=triangle, 2=disk, 3=tile, 4=voxel
    std::vector<uint> primitive_UUIDs;
    std::vector<uint> object_IDs;
    std::vector<int2> object_subdivisions;

    // Vertex data by type
    std::vector<vec3> patch_vertices;
    std::vector<vec3> triangle_vertices;
    std::vector<vec3> tile_vertices;
    std::vector<vec3> voxel_vertices;
    std::vector<vec3> bbox_vertices;

    // Texture/masking
    std::vector<bool> mask_data;
    std::vector<int2> mask_sizes;
    std::vector<int> mask_IDs;
    std::vector<vec2> uv_data;
    std::vector<int> uv_IDs;

    // Flags
    std::vector<char> twosided_flags;
    std::vector<float> solid_fractions;

    size_t primitive_count = 0;
};

/**
 * @brief Material properties for ray tracing
 */
struct RayTracingMaterial {
    std::vector<float> reflectivity;      // [source * band * primitive]
    std::vector<float> transmissivity;    // [source * band * primitive]
    std::vector<float> reflectivity_cam;  // Camera-weighted
    std::vector<float> transmissivity_cam;
    std::vector<float> specular_exponent;
    std::vector<float> specular_scale;

    size_t num_bands = 0;
    size_t num_sources = 0;
    size_t num_primitives = 0;
    size_t num_cameras = 0;
};

/**
 * @brief Radiation source definition
 */
struct RayTracingSource {
    vec3 position;
    vec3 rotation;
    vec2 width;
    uint type;  // 0=collimated, 1=sphere, 2=sun_sphere, 3=rectangle, 4=disk
    std::vector<float> fluxes;  // Per-band fluxes
    std::vector<float> fluxes_cam;  // Camera-weighted fluxes
};

/**
 * @brief Launch parameters for ray tracing
 */
struct RayTracingLaunchParams {
    uint launch_offset;
    uint launch_count;
    uint rays_per_primitive;
    uint random_seed;
    uint current_band;
    uint num_bands;
    uint scattering_iteration;

    // Camera-specific
    uint camera_id;
    vec3 camera_position;
    vec2 camera_direction;
    float camera_focal_length;
    float camera_lens_diameter;
    float camera_fov_aspect;
    int2 camera_resolution;

    // Diffuse parameters
    std::vector<float> diffuse_flux;
    std::vector<float> diffuse_extinction;
    std::vector<vec3> diffuse_peak_dir;

    // Periodic boundary
    vec2 periodic_flag;
};

} // namespace helios
```

### 1.3 Refactor RadiationModel to Use Abstraction

**Current Pattern (to refactor):**
```cpp
// Direct OptiX API calls scattered throughout
RT_CHECK_ERROR(rtContextCreate(&OptiX_Context));
RT_CHECK_ERROR(rtBufferCreate(OptiX_Context, RT_BUFFER_INPUT, &buffer));
RT_CHECK_ERROR(rtContextLaunch3D(OptiX_Context, RAYTYPE_DIRECT, ...));
```

**Target Pattern:**
```cpp
// Business logic uses abstraction
backend = RayTracingBackend::create("optix6");  // or "optix7", "vulkan"
backend->initialize();
backend->updateGeometry(geometry_data);
backend->updateMaterials(material_data);
backend->launchDirectRays(launch_params);
backend->getRadiationResults(results);
```

### 1.4 Create OptiX 6.5 Backend Wrapper

Wrap the existing OptiX 6.5 code into a concrete `OptiX6Backend` class:

```cpp
// File: plugins/radiation/src/OptiX6Backend.cpp

class OptiX6Backend : public RayTracingBackend {
private:
    RTcontext context;

    // All existing RTbuffer, RTvariable, RTprogram members
    RTbuffer transform_matrix_RTbuffer;
    RTbuffer primitive_type_RTbuffer;
    // ... (60+ buffers)

    // Existing helper methods
    void initializeBuffer1Dfloat(...);
    void initializeBuffer2Dfloat(...);
    // ...

public:
    void initialize() override {
        // Move existing initializeOptiX() code here
    }

    void launchDirectRays(const RayTracingLaunchParams& params) override {
        // Move existing direct ray launch code here
        RT_CHECK_ERROR(rtContextLaunch3D(context, RAYTYPE_DIRECT, ...));
    }

    // ... implement all interface methods
};
```

### 1.5 Phase 1 Testing Milestones

| Milestone | Validation Criteria | Test Method |
|-----------|---------------------|-------------|
| **M1.1**: Interface compiles | All header files compile without errors | `./run_tests.sh --test radiation` |
| **M1.2**: OptiX6Backend wraps existing code | Existing tests pass unchanged | Compare output against baseline |
| **M1.3**: Backend selection works | Can switch backend via configuration | Unit test backend factory |
| **M1.4**: No regression | All radiation plugin tests pass | Full test suite |
| **M1.5**: Performance baseline | Document current performance | Benchmark suite |

### 1.6 Phase 1 File Structure

```
plugins/radiation/
├── include/
│   ├── RadiationModel.h          # Updated to use backend interface
│   ├── RayTracingBackend.h       # NEW: Abstract interface
│   ├── RayTracingTypes.h         # NEW: Backend-agnostic types
│   └── RayTracing.cuh            # Unchanged (Phase 2 updates)
├── src/
│   ├── RadiationModel.cpp        # Refactored to use backend
│   ├── backends/                 # NEW: Backend implementations
│   │   ├── OptiX6Backend.cpp     # NEW: Wrapper for existing code
│   │   ├── OptiX6Backend.h       # NEW: OptiX 6 specific headers
│   │   └── BackendFactory.cpp    # NEW: Backend selection
│   ├── rayGeneration.cu          # Unchanged (Phase 2 updates)
│   ├── primitiveIntersection.cu  # Unchanged (Phase 2 updates)
│   └── rayHit.cu                 # Unchanged (Phase 2 updates)
```

---

## Phase 2: OptiX 6.5 to 7.7 Migration

### Objective

Implement a new `OptiX7Backend` using the OptiX 7.7 API, replacing the legacy OptiX 6.5 code with modern, header-only OptiX.

### 2.1 Key Architectural Changes

OptiX 7.x represents a fundamental redesign from OptiX 6.x:

| Aspect | OptiX 6.5 | OptiX 7.7 |
|--------|-----------|-----------|
| **API Style** | High-level, managed | Low-level, explicit |
| **Memory** | Automatic (RTbuffer) | Manual (cudaMalloc) |
| **Programs** | Runtime PTX loading | Explicit pipeline + SBT |
| **Accel Struct** | Automatic building | Manual build workflow |
| **Variables** | rtDeclareVariable | Launch parameters struct |
| **Distribution** | Library files | Header-only (driver provides runtime) |

### 2.2 New OptiX 7 Concepts to Implement

#### Shader Binding Table (SBT)

The SBT replaces OptiX 6's automatic variable binding:

```cpp
// SBT Record structure with alignment requirements
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RayGenSbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Custom data here
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissSbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Custom data here
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupSbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Per-geometry custom data
};
```

#### Pipeline Creation

```cpp
// Pipeline compile options
OptixPipelineCompileOptions pipelineCompileOptions = {};
pipelineCompileOptions.usesMotionBlur = false;
pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
pipelineCompileOptions.numPayloadValues = 8;  // Match PerRayData size
pipelineCompileOptions.numAttributeValues = 2;
pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

// Module creation from PTX/OptiX IR
OptixModule module;
optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions,
                  ptx_code, ptx_size, log, &log_size, &module);

// Program group creation
OptixProgramGroupDesc pgDesc = {};
pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
pgDesc.raygen.module = module;
pgDesc.raygen.entryFunctionName = "__raygen__direct";
optixProgramGroupCreate(context, &pgDesc, 1, &pgOptions, log, &log_size, &raygen_pg);
```

#### Acceleration Structure Building

```cpp
// Build input for triangles
OptixBuildInput buildInput = {};
buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
buildInput.triangleArray.vertexBuffers = &d_vertices;
buildInput.triangleArray.numVertices = vertex_count;
buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);

// Query memory requirements
OptixAccelBufferSizes bufferSizes;
optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes);

// Allocate and build
CUdeviceptr d_temp, d_output;
cudaMalloc(&d_temp, bufferSizes.tempSizeInBytes);
cudaMalloc(&d_output, bufferSizes.outputSizeInBytes);

optixAccelBuild(context, stream, &accelOptions, &buildInput, 1,
                d_temp, bufferSizes.tempSizeInBytes,
                d_output, bufferSizes.outputSizeInBytes,
                &gas_handle, nullptr, 0);
```

### 2.3 Device Code Migration

**OptiX 6.5 Pattern:**
```cuda
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtBuffer<float, 1> radiation_in;

RT_PROGRAM void direct_raygen() {
    PerRayData prd;
    prd.seed = tea<16>(launch_index.x, random_seed);

    optix::Ray ray = optix::make_Ray(origin, direction, ray_type, tmin, tmax);
    rtTrace(top_object, ray, prd);

    atomicAdd(&radiation_in[index], prd.strength);
}
```

**OptiX 7.7 Pattern:**
```cuda
// Launch parameters structure (replaces rtDeclareVariable)
struct LaunchParams {
    OptixTraversableHandle traversable;
    float* radiation_in;
    uint random_seed;
    // ... all parameters
};

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __raygen__direct() {
    const uint3 idx = optixGetLaunchIndex();

    // Payload passed via registers (not struct)
    uint32_t p0, p1, p2, p3;  // Pack PerRayData into uint32 slots

    optixTrace(
        params.traversable,
        origin, direction,
        tmin, tmax,
        0.0f,  // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,  // SBT offset, stride, miss index
        p0, p1, p2, p3  // payload registers
    );

    // Unpack and process results
    float strength = __uint_as_float(p0);
    atomicAdd(&params.radiation_in[index], strength);
}
```

### 2.4 OptiX 7.7 Header Distribution

OptiX 7.x is header-only - the runtime library ships with NVIDIA drivers:

```cmake
# CMakeLists.txt for OptiX 7.7
set(OPTIX7_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX7/include")

# Headers to bundle (check license for redistribution)
# Required headers:
#   optix.h
#   optix_device.h
#   optix_host.h
#   optix_stack_size.h
#   optix_stubs.h
#   optix_types.h
#   optix_function_table.h
#   optix_function_table_definition.h
#   internal/optix_7_device_impl.h
#   internal/optix_7_device_impl_exception.h
#   internal/optix_7_device_impl_transformations.h

# No library linking needed - driver provides runtime
target_link_libraries(radiation PUBLIC CUDA::cudart)
```

**Licensing Note:** Most OptiX 7 SDK headers can be redistributed. Exception: `optixPaging.*` files have restricted redistribution. Always verify against current SDK license.

### 2.5 Phase 2 Implementation Steps

#### Step 2.5.1: OptiX 7 Context Setup
```cpp
class OptiX7Backend : public RayTracingBackend {
private:
    OptixDeviceContext optix_context;
    CUcontext cuda_context;
    CUstream cuda_stream;

    // Pipeline components
    OptixPipeline pipeline;
    OptixModule module;
    OptixProgramGroup raygen_direct_pg;
    OptixProgramGroup raygen_diffuse_pg;
    OptixProgramGroup raygen_camera_pg;
    OptixProgramGroup miss_pg;
    OptixProgramGroup hitgroup_pg;

    // Shader Binding Table
    CUdeviceptr d_raygen_record;
    CUdeviceptr d_miss_record;
    CUdeviceptr d_hitgroup_records;
    OptixShaderBindingTable sbt;

    // Launch parameters
    LaunchParams h_params;
    CUdeviceptr d_params;

    // Acceleration structures
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output;

public:
    void initialize() override {
        // 1. Initialize CUDA
        cudaFree(0);  // Initialize CUDA context
        cuCtxGetCurrent(&cuda_context);
        cuStreamCreate(&cuda_stream, CU_STREAM_DEFAULT);

        // 2. Initialize OptiX
        optixInit();
        OptixDeviceContextOptions options = {};
        optixDeviceContextCreate(cuda_context, &options, &optix_context);

        // 3. Create modules and pipeline
        createModules();
        createProgramGroups();
        createPipeline();
        createSBT();
    }
};
```

#### Step 2.5.2: Port Each Ray Type

For each of the 4 ray types, create corresponding OptiX 7 programs:

| OptiX 6 Program | OptiX 7 Entry Point |
|-----------------|---------------------|
| `direct_raygen` | `__raygen__direct` |
| `diffuse_raygen` | `__raygen__diffuse` |
| `camera_raygen` | `__raygen__camera` |
| `pixel_label_raygen` | `__raygen__pixel_label` |
| `closest_hit_direct` | `__closesthit__direct` |
| `closest_hit_diffuse` | `__closesthit__diffuse` |
| `closest_hit_camera` | `__closesthit__camera` |
| `closest_hit_pixel_label` | `__closesthit__pixel_label` |
| `miss_direct` | `__miss__direct` |
| `miss_diffuse` | `__miss__diffuse` |
| `miss_camera` | `__miss__camera` |
| `rectangle_intersect` | `__intersection__rectangle` |
| `triangle_intersect` | Built-in triangle intersection |
| `disk_intersect` | `__intersection__disk` |

#### Step 2.5.3: Custom Primitive Intersection

OptiX 7 supports custom intersection programs via AABB primitives:

```cuda
extern "C" __global__ void __intersection__rectangle() {
    const uint prim_idx = optixGetPrimitiveIndex();

    // Get ray parameters
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();

    // Load transform matrix from launch params
    float m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = params.transform_matrix[16 * prim_idx + i];
    }

    // Compute intersection (same math as OptiX 6)
    // ...

    if (hit) {
        optixReportIntersection(t_hit, 0, /* attributes */);
    }
}
```

### 2.6 Phase 2 Testing Milestones

| Milestone | Validation Criteria | Test Method |
|-----------|---------------------|-------------|
| **M2.1**: OptiX 7 context initializes | No errors during initialization | Unit test |
| **M2.2**: Module compilation works | PTX compiles to OptiX module | Build test |
| **M2.3**: Pipeline creation works | All program groups link | Unit test |
| **M2.4**: GAS builds correctly | Acceleration structure builds | Unit test with simple geometry |
| **M2.5**: Direct rays work | Direct radiation matches OptiX 6 | Numerical comparison (<0.1% error) |
| **M2.6**: Diffuse rays work | Diffuse radiation matches OptiX 6 | Numerical comparison (<0.1% error) |
| **M2.7**: Camera rays work | Camera images match OptiX 6 | Visual + PSNR comparison |
| **M2.8**: Full regression pass | All tests pass with OptiX 7 backend | Full test suite |
| **M2.9**: Performance benchmark | Document performance vs OptiX 6 | Benchmark suite |

### 2.7 OptiX 7.7 Feature Opportunities

After basic migration, consider these OptiX 7.7 enhancements:

- **Opacity Micromaps (OMMs)**: Hardware-accelerated alpha testing for vegetation
- **Displacement Micromaps (DMMs)**: Detailed surface displacement without mesh complexity
- **OptiX IR compilation**: Faster module loading than PTX
- **Shader Execution Reordering**: Better coherence for complex materials
- **Temporal denoising**: If implementing path tracing or camera effects

---

## Phase 3: Vulkan Ray-Tracing Fallback

### Objective

Implement a `VulkanBackend` that provides cross-platform ray-tracing support for AMD and Intel GPUs when NVIDIA hardware is unavailable.

### 3.1 Vulkan Ray-Tracing Overview

Vulkan ray-tracing is built on three core extensions:

- **VK_KHR_acceleration_structure**: BVH building and management
- **VK_KHR_ray_tracing_pipeline**: Dedicated RT pipeline with shader stages
- **VK_KHR_ray_query**: Inline ray tracing from any shader (simpler, often faster)

**Hardware Support:**
| Vendor | Hardware | Driver Requirements |
|--------|----------|---------------------|
| NVIDIA | Turing+ (RTX 20/30/40) | R460+ |
| AMD | RDNA 2+ (RX 6000/7000) | Adrenalin 20.11.3+ |
| Intel | Arc Alchemist+ | Latest drivers |

### 3.2 Vulkan vs OptiX Comparison

| Aspect | OptiX 7 | Vulkan RT |
|--------|---------|-----------|
| **Complexity** | Moderate | High (~5x more code) |
| **Platform** | NVIDIA only | Cross-vendor |
| **Memory** | Manual (CUDA) | Manual (Vulkan allocator) |
| **Shaders** | CUDA (.cu) | GLSL/HLSL (.rgen, .rchit, etc.) |
| **AS Building** | Explicit | Explicit (similar) |
| **Performance** | Highly optimized for NVIDIA | Varies by vendor/driver |

### 3.3 Implementation Strategy: Ray Query First

**Recommendation:** Start with `VK_KHR_ray_query` rather than `VK_KHR_ray_tracing_pipeline`:

**Advantages:**
- Simpler API (inline from compute shaders)
- Better performance for simple use cases (shadows, visibility)
- Mobile support (Mali GPUs only support ray query)
- Easier debugging

**Ray Query Pattern:**
```glsl
#version 460
#extension GL_EXT_ray_query : require

layout(binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1) buffer RadiationBuffer { float radiation_in[]; };

void main() {
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0xFF, origin, tMin, direction, tMax);

    while (rayQueryProceedEXT(rayQuery)) {
        // Handle intersection
        if (rayQueryGetIntersectionTypeEXT(rayQuery, false) ==
            gl_RayQueryCandidateIntersectionTriangleEXT) {
            rayQueryConfirmIntersectionEXT(rayQuery);
        }
    }

    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        // Process hit
        float t = rayQueryGetIntersectionTEXT(rayQuery, true);
        int primID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
        // ...
    }
}
```

### 3.4 Vulkan Backend Architecture

```cpp
class VulkanBackend : public RayTracingBackend {
private:
    // Vulkan core
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    VkCommandPool command_pool;

    // Ray tracing extensions
    VkAccelerationStructureKHR blas;
    VkAccelerationStructureKHR tlas;
    VkBuffer blas_buffer;
    VkBuffer tlas_buffer;

    // Compute pipeline (for ray query)
    VkPipeline direct_pipeline;
    VkPipeline diffuse_pipeline;
    VkPipeline camera_pipeline;
    VkPipelineLayout pipeline_layout;
    VkDescriptorSetLayout descriptor_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;

    // Buffers
    VkBuffer geometry_buffer;
    VkBuffer material_buffer;
    VkBuffer radiation_buffer;
    VkDeviceMemory device_memory;

    // Extension function pointers
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    // ...

public:
    void initialize() override;
    void buildAccelerationStructure() override;
    void launchDirectRays(const RayTracingLaunchParams& params) override;
    // ...
};
```

### 3.5 Shader Porting: CUDA to GLSL

**CUDA (OptiX):**
```cuda
extern "C" __global__ void __raygen__direct() {
    const uint3 idx = optixGetLaunchIndex();
    float3 origin = computeRayOrigin(idx);
    float3 direction = computeRayDirection(idx);

    optixTrace(params.traversable, origin, direction, ...);
}
```

**GLSL (Vulkan ray query):**
```glsl
#version 460
#extension GL_EXT_ray_query : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, std430) buffer Params { LaunchParams params; };
layout(binding = 2, std430) buffer RadiationIn { float radiation_in[]; };

void main() {
    uvec3 idx = gl_GlobalInvocationID;
    vec3 origin = computeRayOrigin(idx);
    vec3 direction = computeRayDirection(idx);

    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0xFF, origin, 1e-4, direction, 1e30);

    while (rayQueryProceedEXT(rayQuery)) {
        // Process candidates
    }

    // Process committed intersection
}
```

### 3.6 BLAS/TLAS Construction

```cpp
void VulkanBackend::buildAccelerationStructure() {
    // 1. Build BLAS for each geometry type
    buildBLAS(patch_vertices, patch_indices, &blas_patch);
    buildBLAS(triangle_vertices, triangle_indices, &blas_triangle);
    buildBLAS(disk_aabbs, &blas_disk);  // Custom AABB for disks

    // 2. Create instances for TLAS
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    for (size_t i = 0; i < primitive_count; i++) {
        VkAccelerationStructureInstanceKHR instance = {};
        // Set transform from transform_matrices[i]
        memcpy(instance.transform.matrix, &transform_matrices[i * 16],
               sizeof(VkTransformMatrixKHR));
        instance.instanceCustomIndex = i;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = getBlasDeviceAddress(primitive_types[i]);
        instances.push_back(instance);
    }

    // 3. Build TLAS
    buildTLAS(instances, &tlas);
}

void VulkanBackend::buildBLAS(const std::vector<float3>& vertices,
                               const std::vector<uint3>& indices,
                               VkAccelerationStructureKHR* blas) {
    // Query build sizes
    VkAccelerationStructureBuildSizesInfoKHR buildSizes = {};
    buildSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress = vertexBufferAddress;
    geometry.geometry.triangles.vertexStride = sizeof(float3);
    geometry.geometry.triangles.maxVertex = vertices.size() - 1;
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.indexData.deviceAddress = indexBufferAddress;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t primitiveCount = indices.size();
    vkGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &buildSizes);

    // Allocate buffers and build...
}
```

### 3.7 Phase 3 Testing Milestones

| Milestone | Validation Criteria | Test Method |
|-----------|---------------------|-------------|
| **M3.1**: Vulkan instance + device | Initialization succeeds on AMD/Intel | Unit test |
| **M3.2**: RT extensions available | Extensions enumerated | Capability check |
| **M3.3**: BLAS builds | Triangle BLAS builds correctly | Unit test |
| **M3.4**: TLAS builds | Instance hierarchy works | Unit test |
| **M3.5**: Ray query works | Simple visibility test passes | Unit test |
| **M3.6**: Direct rays work | Results within 1% of OptiX | Numerical comparison |
| **M3.7**: Diffuse rays work | Results within 1% of OptiX | Numerical comparison |
| **M3.8**: Camera rays work | Images visually match | Visual + PSNR |
| **M3.9**: Cross-vendor test | Works on AMD, Intel, NVIDIA | Hardware matrix testing |

### 3.8 Runtime Backend Selection

```cpp
// In RadiationModel constructor or configuration
std::unique_ptr<RayTracingBackend> selectBackend() {
    // Check environment override
    const char* env_backend = std::getenv("HELIOS_RT_BACKEND");
    if (env_backend) {
        return RayTracingBackend::create(env_backend);
    }

    // Auto-detection
    if (isNVIDIAGPU() && hasOptiX7Support()) {
        return RayTracingBackend::create("optix7");
    } else if (hasVulkanRayTracing()) {
        return RayTracingBackend::create("vulkan");
    } else if (isNVIDIAGPU() && hasOptiX6Support()) {
        return RayTracingBackend::create("optix6");  // Legacy fallback
    }

    helios_runtime_error("No ray tracing backend available. "
                         "NVIDIA GPU with OptiX or Vulkan RT-capable GPU required.");
}
```

---

## Testing Strategy

### Continuous Validation

Throughout all phases, maintain numerical and visual validation:

#### Numerical Validation
```cpp
// Test helper for comparing radiation results
bool validateRadiationResults(const std::vector<float>& reference,
                               const std::vector<float>& test,
                               float tolerance = 0.001f) {
    if (reference.size() != test.size()) return false;

    for (size_t i = 0; i < reference.size(); i++) {
        float rel_error = std::abs(reference[i] - test[i]) /
                          std::max(std::abs(reference[i]), 1e-10f);
        if (rel_error > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": ref=" << reference[i]
                      << " test=" << test[i]
                      << " error=" << rel_error << std::endl;
            return false;
        }
    }
    return true;
}
```

#### Visual Validation (Camera Images)
```cpp
// PSNR comparison for camera renders
float computePSNR(const std::vector<float>& ref, const std::vector<float>& test) {
    float mse = 0.0f;
    for (size_t i = 0; i < ref.size(); i++) {
        float diff = ref[i] - test[i];
        mse += diff * diff;
    }
    mse /= ref.size();

    float max_val = 1.0f;  // Assuming normalized [0,1] images
    return 10.0f * std::log10(max_val * max_val / mse);
}

// Require PSNR > 40 dB for "visually identical"
bool validateCameraImage(const std::vector<float>& ref,
                          const std::vector<float>& test) {
    return computePSNR(ref, test) > 40.0f;
}
```

### Benchmark Suite

Create comprehensive benchmarks to track performance across phases:

```cpp
// benchmarks/ray_tracing_performance/main.cpp
struct BenchmarkResult {
    std::string backend;
    std::string scene;
    double build_time_ms;
    double trace_time_ms;
    size_t rays_per_second;
    size_t memory_bytes;
};

void runBenchmark(RayTracingBackend* backend, const std::string& scene_path) {
    // Load scene
    Context context;
    loadScene(context, scene_path);

    // Warm-up
    backend->launchDirectRays(params);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        backend->launchDirectRays(params);
    }
    cudaDeviceSynchronize();  // or vkQueueWaitIdle()
    auto end = std::chrono::high_resolution_clock::now();

    // Report
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    // ...
}
```

---

## Risk Assessment

### High Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| OptiX 7 API learning curve | Schedule delay | Study optix7course examples first |
| Vulkan complexity | 3-4 month delay | Use ray query (simpler) first |
| Numerical precision differences | Validation failures | Allow configurable tolerance |
| Custom primitive intersection | Functionality loss | Thoroughly test all 5 primitive types |
| Multi-GPU support | Feature regression | Test single-GPU first, add multi later |

### Medium Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| Driver compatibility | User issues | Document minimum driver versions |
| Memory management bugs | Crashes | Use CUDA/Vulkan validation layers |
| SBT alignment errors | Crashes | Use alignment macros consistently |
| Build system complexity | CI failures | Test on multiple platforms early |

### Contingency Plans

1. **If OptiX 7 migration stalls**: Keep OptiX 6.5 as primary, make OptiX 7 optional
2. **If Vulkan proves too complex**: Consider Intel Embree as CPU fallback instead
3. **If performance degrades significantly**: Profile and optimize specific bottlenecks

---

## Resource Requirements

### Development Time Estimates

| Phase | Estimated Duration | Dependencies |
|-------|-------------------|--------------|
| Phase 1: Isolation | 4-6 weeks | None |
| Phase 2: OptiX 7.7 | 8-12 weeks | Phase 1 complete |
| Phase 3: Vulkan | 12-16 weeks | Phase 1 complete |
| Integration testing | 4 weeks | All phases |
| **Total** | **28-38 weeks** | |

*Note: Phases 2 and 3 can be developed in parallel after Phase 1.*

### Hardware Requirements for Testing

- NVIDIA GPU: Turing or newer (RTX 2000+) for OptiX 7
- AMD GPU: RDNA 2 or newer (RX 6000+) for Vulkan testing
- Intel GPU: Arc Alchemist for Intel Vulkan testing (optional)

### Software Requirements

- CUDA Toolkit 11.7+ (for OptiX IR)
- Vulkan SDK 1.3+ with ray tracing extensions
- CMake 3.18+
- C++17 compiler

---

## References

### OptiX Resources
- [OptiX 7.7 Programming Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html)
- [NVIDIA OptiX_Apps Repository](https://github.com/NVIDIA/OptiX_Apps)
- [Ingo Wald's OptiX 7 Course](https://github.com/ingowald/optix7course)
- [How to Get Started with OptiX 7](https://developer.nvidia.com/blog/how-to-get-started-with-optix-7/)

### Vulkan Resources
- [Vulkan Ray Tracing Tutorial](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
- [Ray Tracing in Vulkan (Khronos)](https://www.khronos.org/blog/ray-tracing-in-vulkan)
- [Vulkan RT Best Practices](https://www.khronos.org/blog/vulkan-ray-tracing-best-practices-for-hybrid-rendering)

### Cross-Platform References
- [ChameleonRT](https://github.com/Twinklebear/ChameleonRT) - Multi-backend path tracer
- [OWL (OptiX Wrapper Library)](https://github.com/owl-project/owl)

---

## Appendix A: Current OptiX 6.5 API Usage Summary

### Buffer Types (60+)
```
Geometry: patch_vertices, triangle_vertices, disk_*, tile_*, voxel_*, bbox_*
Materials: rho, tau, rho_cam, tau_cam, specular_*
Sources: source_positions, source_types, source_fluxes, source_widths, source_rotations
Diffuse: diffuse_flux, diffuse_extinction, diffuse_peak_dir, sky_radiance_params
Output: radiation_in, radiation_out_top, radiation_out_bottom, scatter_buff_*
Camera: camera_pixel_label, camera_pixel_depth, radiation_in_camera
Texture: maskdata, masksize, maskID, uvdata, uvID
```

### Ray Types (4)
```
RAYTYPE_DIRECT = 0
RAYTYPE_DIFFUSE = 1
RAYTYPE_CAMERA = 2
RAYTYPE_PIXEL_LABEL = 3
```

### Program Entry Points (16)
```
Ray Generation: direct_raygen, diffuse_raygen, camera_raygen, pixel_label_raygen
Closest Hit: closest_hit_direct, closest_hit_diffuse, closest_hit_camera, closest_hit_pixel_label
Miss: miss_direct, miss_diffuse, miss_camera, miss_pixel_label
Intersection: rectangle_intersect, triangle_intersect, disk_intersect, tile_intersect, voxel_intersect
Bounding Box: rectangle_bounds, triangle_bounds, disk_bounds, tile_bounds, voxel_bounds
```

---

## Appendix B: OptiX 7.7 Headers for Distribution

Required headers (check license for redistribution rights):
```
optix.h
optix_device.h
optix_host.h
optix_stack_size.h
optix_stubs.h
optix_types.h
optix_function_table.h
optix_function_table_definition.h
internal/optix_7_device_impl.h
internal/optix_7_device_impl_exception.h
internal/optix_7_device_impl_transformations.h
```

Driver requirements:
- OptiX 7.7: NVIDIA Driver 530.41+
- CUDA 11.7+ recommended for OptiX IR

---

*Document Version: 1.0*
*Created: December 2025*
*Author: Generated with Claude Code*
