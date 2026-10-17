# Radiation backend parity — handoff to a CUDA/OptiX machine

**Date:** 2026-10-16 (v1.3.92 development)
**Status:** The Vulkan side is done and tested on macOS (Vulkan/MoltenVK). The OptiX 8 and OptiX 6 items below need an NVIDIA GPU.
**Before committing your changes to this branch, delete this handoff file (`git rm plugins/radiation/doc/backend_parity_HANDOFF.md`)
so it is not merged into master.** Anything still unresolved at that point goes in your report to the user, not in this file.

## Background

A read-only audit compared the three ray-tracing backends of the radiation plug-in:

- **OptiX 8:** `src/optix8/OptiX8DeviceCode.cu`, `src/backends/OptiX8Backend.cpp`
- **OptiX 6:** `src/rayGeneration.cu`, `src/rayHit.cu`, `src/primitiveIntersection.cu`, `include/RayTracing.cuh`, `src/backends/OptiX6Backend.cpp`
- **Vulkan:** `shaders/*.comp`, `shaders/common/*.glsl`, `src/backends/VulkanComputeBackend.cpp`

Every divergence the Mac could fix and test has been fixed in Vulkan. The rest are OptiX-side or need a design decision. Line numbers
are as of the audit and may have drifted.

**Workflow (from CLAUDE.md):**

- For each bug, write a regression test that FAILS on the current OptiX code, then fix it, then confirm it passes.
- Use `GPU_TEST_CASE` for every test.
- Run `./run_tests.sh --nogpu --test radiation` before pushing.
- Grep the build log for `warning:`.
- Add a CHANGELOG bullet per user-visible fix. Several Vulkan bullets are already under `# [1.3.92]` → `## Radiation`; widen them rather than adding duplicates where the same fix now covers OptiX too.

## Already done on Vulkan (for reference)

Each fix has a regression test in `tests/selfTest.cpp` that failed before the fix and passes after it (Vulkan). These tests run on every
backend, so run them on OptiX first: they show where OptiX already agrees and where it does not.

| Test (search the name) | Vulkan fix | Expected on OptiX |
|---|---|---|
| `a repeated runBand without direct rays gives the same result` | `zeroRadiationBuffers` now zeroes scatter + radiation_out buffers | should pass (OptiX already zeroes them) |
| `a specular highlight appears on the bottom face of a surface lit and viewed from below` | specular uses the normal of the viewed face, and skips sources behind that face | **expected to FAIL on OptiX 6**; should pass on OptiX 8 |
| `rays from a sparsely masked primitive are not lost` | masked-origin resampling launches from the last sample instead of dropping the ray | should pass |
| `an object behind an extended light source does not shadow it` | direct shadow ray ends at the sampled source point, minus distance travelled through wraps/covers | should pass |
| `a ray that reaches the periodic wrap limit is treated as blocked` | Vulkan now abandons a ray at the 10th boundary hit (deposits nothing), for direct, diffuse and camera rays | should pass |
| `periodic boundaries wrap rays in a domain far from the origin` | none: coverage only, see item 13 | should pass |

**Other Vulkan changes, not separately tested:**

- The band-map identity fallback (when `band_launch_flag` was empty) now raises an error.
- The voxel error message is corrected: only OptiX 6 supports voxels.
- When the 32-segment limit runs out, direct, diffuse and camera rays deposit nothing.

**Decision made by the user:** the wrap limit stays at 10 boundary hits in all backends, and a ray that reaches it is treated as
blocked (deposits nothing). This deliberately under-counts rays needing more wraps (low sun, tall periodic canopy). If the cap is
ever raised, it must change in all three backends together: `MAX_PERIODIC_WRAPS` in `RayTracing.cuh`, `OptiX8DeviceCode.cu` and
`shaders/common/glass_cover.glsl`.

## OptiX work, in priority order

### 1. OptiX 8 fires direct rays from the back face of `twosided_flag == 2` primitives (bug)

- **Location:** `OptiX8DeviceCode.cu` ~1616-1618, which skips only `!face && tsf==0` and `tsf==3`.
- **Rule in OptiX 6 (`rayGeneration.cu:240`) and Vulkan (`direct_raygen.comp`):** a ray fires if `(face_top || tsf == 1) && tsf != 3`.
- **Effect:** a transparent primitive lit from behind absorbs and scatters direct flux on OptiX 8 only.
- **Test:** a horizontal `twosided_flag=2` patch lit from below by a collimated source, so it should absorb 0.
- Check how flag 2 is meant to behave; see the `invisibility_flag` CHANGELOG entry in 1.3.92.

### 2. OptiX 8 has no self-intersection exclusion (bug, depends on coordinates)

- **The gap:** `__intersection__patch` (the unified dispatcher, ~323-500) never compares the hit to `prd->origin_UUID`. The only self-check is in `coverAnyHitBody` (~985), which returns early when no glass is in the scene. Protection is therefore only `tmin = 1e-4`.
- **The other backends always skip the origin primitive:**
  - OptiX 6: `isRayOrigin()` in `primitiveIntersection.cu`
  - Vulkan: `bvh_traversal.glsl`, the `origin_prim_idx` skip
- **Fix:** reject `hit UUID == origin_UUID` in the intersection program for direct and diffuse rays. Camera and pixel-label rays carry a pixel index in `origin_UUID`, so they must be excluded from this check.
- **Test:** a single tilted patch far from the origin (e.g. x = 5000 m) under a grazing sun. Its direct flux should equal `flux * cos θ`.

### 3. Specular normal and same-side guard (bug)

- **OptiX 6 (`rayHit.cu` ~528-558, used ~733):** uses the fixed geometric normal. Flip it to the face the camera sees, as OptiX 8 does at ~1167-1170.
- **OptiX 6 and OptiX 8:** add the guard Vulkan now has. Skip a source when `dot(light_dir, viewed_normal) <= 0`, because that light falls on the other face. This is not covered by the new test; it only matters for grazing light from behind.
- **NaN when the light direction equals the view direction:** OptiX 6 (~728) and OptiX 8 (~1322) call `normalize(light_dir - ray_dir)` with no zero check, which gives a NaN pixel. Vulkan uses `safe_normalize` and skips the source. Match Vulkan.
- **Voxels in OptiX 6:** the voxel branch leaves `normal` uninitialised (~552-558). See item 8.

### 4. When wraps or segments run out, pixel-label depth differs (minor)

- Vulkan writes depth −1 with label 0. OptiX writes nothing, so depth keeps its zeroed value of 0. Make OptiX write −1.
- Locations: OptiX 6 `rayGeneration.cu` ~701-704; OptiX 8 ~2033.

### 5. Direct shadow ray length after a periodic wrap (minor)

- **OptiX:** keeps the original `tmax` (the distance to the sampled source point) for every segment after a wrap.
- **Vulkan:** now subtracts the distance already travelled, and treats a boundary beyond the source point as reaching the source.
- **Effect:** only matters for a finite source inside a periodic domain.
- **Locations:** OptiX 8 ~1525-1645; OptiX 6 `rayGeneration.cu` ~157-249.

### 6. OptiX 8 may pick the opposite hit face for patches with a mirrored transform (possible bug, unverified)

- **OptiX 8:** `__intersection__patch` takes the face from transform column 2, `(T[2],T[6],T[10])` (~382, 427).
- **Everything else:** OptiX 8 raygen, OptiX 6 and Vulkan all use `cross(col0, col1)`.
- **When they disagree:** for a transform with a negative determinant, e.g. a patch scaled by −1 on one axis.
- **Test:** a mirrored two-sided patch with different top and bottom outgoing flux, checking which value a diffuse ray from a receiver picks up.

### 7. Sampling statistics (minor, no bias)

- **Direct ray count:**
  - OptiX 8 (`OptiX8Backend.cpp` ~874-878) uses `floor(sqrt(N))` × `N/dim_x` (N = 50 gives 49 rays).
  - OptiX 6 and Vulkan use `ceil(sqrt(N))²` (N = 50 gives 64 rays).
  - Align OptiX 8 to the others.
- **RNG seed:**
  - OptiX 8 seeds with the batch-local `prim_local` (direct ~1415, diffuse ~1680), and batches of 65535 restart it. Primitives k and k+65535 get identical samples.
  - Vulkan uses the global primitive index.
  - Use `launch_offset + prim_local` in OptiX 8. OptiX 6 has the same pattern (`rayGeneration.cu` ~263), but its batches are about 1e9 rays.

### 8. Voxels (decision needed: fix or drop)

- Only OptiX 6 intersects voxels. OptiX 8 and Vulkan refuse them with an error.
- OptiX 6's direct raygen for type 4 (`rayGeneration.cu` ~128-131) never sets the sample point `sp` or `normal`, but still traces and deposits energy. The result is garbage direct flux.
- Either implement voxel sampling properly in OptiX 6, or refuse voxels there too and remove voxel support. **Ask the user.**

### 9. `getSkyEnergy()` returns 0 on every backend (decision needed)

- No device code in any backend writes `Rsky`, and `updateDiffuseRadiation()` is never called.
- `RadiationModel::getSkyEnergy()` (`RadiationModel.cpp` ~5435) therefore always returns 0.0, a silent fake value that breaks the fail-fast rule.
- Either implement it (accumulate escaped scattered energy in the diffuse miss programs of all three backends), or make it raise an error and deprecate it. **Ask the user.**

### 10. Triangle texture-mask texel lookup (minor inconsistency)

- **Hit test:**
  - OptiX 8 `sampleMask` (~180-190) uses `floor`.
  - OptiX 6 hit (`primitiveIntersection.cu` ~212-227) and Vulkan (`texture_mask.glsl` ~66-74) use `round(|uv|)`.
- **OptiX 6 origin sampling** (`RayTracing.cuh` ~627-632) uses `floor`, so OptiX 6 disagrees with itself.
- **Which is right?** Determine which convention matches how the Context computes the mask and solid fraction, then make all four sites agree.
- **Parametric triangle UVs (no custom UVs):**
  - OptiX 8 maps to (β+γ, γ); Vulkan maps to (γ, β+γ).
  - OptiX 6 reads `uvdata[-1]`.
  - This is probably unreachable, since a textured triangle always has UVs. Confirm, then unify or guard.
- **OptiX 6 triangle origin barycentrics** come from a ray cast from the world origin (`RayTracing.cuh` ~608-623). That gives NaN when the triangle's plane passes through (0,0,0).

### 11. Small robustness gaps (minor)

- **Prague sky with zero circumsolar width:** OptiX 6 (`rayHit.cu` ~1015) and OptiX 8 (~161) divide by it. Vulkan guards it (`sky_distribution.glsl:98`).
- **`updateSkyModel` with empty vectors:**
  - OptiX 8 frees the buffers.
  - OptiX 6 and Vulkan keep the previous ones.
  - Make empty input an error in all three, or make them behave the same.
- **OptiX 6 `launchPixelLabelRays`** does not set the camera pose (`OptiX6Backend.cpp` ~804-831). It relies on the preceding `launchCameraRays` call. Set the pose from the launch params, as the other backends do.
- **OptiX 6 `queryGPUMemory`** queries but prints nothing (~1059-1070).
- **Non-finite camera pixels:** Vulkan raises an error on readback; OptiX returns NaN silently.

### 12. Dead or misleading host fields (cleanup)

- **`RayTracingSource::fluxes`:** no backend reads it.
- **`RayTracingSource::fluxes_cam`:**
  - `buildSourceData()` (`RadiationModel.cpp` ~7055, "Same for now") fills it in the wrong layout.
  - It is overwritten with the correct `[source][launch band][camera]` layout before every direct launch that has cameras (~4609-4617).
- **OptiX 8 `uploadSourceFluxesCam`** overwrites the 3D buffer with a 2D slice. That is harmless only because of the re-upload above.
- **`solid_fractions`:** only OptiX 6 reads it.
- **Unused by all device code:** `camera_HFOV`, `camera_pixel_solid_angle`, `max_scatters`, `current_band`.
- **Comment error:** `RayTracingTypes.h:73` says "8 vertices per bbox"; the code uses 4 per face.

### 13. Not a bug after testing: periodic wrap tolerance

- **The claim:** the audit said OptiX 6 and Vulkan pick the wrapped face with an absolute tolerance of 1e-5 (`rayHit.cu` ~235-243, `periodic_boundary.glsl`), which should fail far from the origin.
- **Not reproduced:** on Vulkan the wrap works at offsets of 1e3, 1e4 and 1e5 m, because a hit on an axis-aligned face lands exactly on the face.
- **No action** unless it reproduces on OptiX 6. The test above checks it.
