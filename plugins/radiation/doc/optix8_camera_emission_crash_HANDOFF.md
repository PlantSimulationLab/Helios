# Handoff: Pre-existing OptiX 8 illegal-memory-access crash in camera + emission-band rendering

> **STATUS: RESOLVED (2026-06-15).** Root-caused and fixed. The camera `__miss__camera` source
> loop dereferenced a **null `source_fluxes`** while `Nsources == 1`: `runBand()` auto-adds a
> default zero-flux collimated source, so `rundirect == false` and `uploadSourceFluxes` (gated on
> `rundirect`) never ran, leaving `source_fluxes` null while `Nsources` was still uploaded as 1.
> Fix: `RadiationModel::runBand` now uploads `source_fluxes` whenever `Nsources > 0` (not only when
> the direct pass runs), and `OptiX8Backend::uploadSourceFluxes`/`uploadSourceFluxesCam` null the
> host pointer on empty input instead of leaving it dangling. Verified: full radiation suite
> **98/98 cases, 13,217 assertions, 0 failures, 0 skipped on OptiX 8.1** with NO test-case exclusion.
> See `optix8_camera_emission_crash_INVESTIGATION.md` for the full diagnosis. The historical
> problem statement below is retained for context.

## TL;DR

On an NVIDIA machine where the **OptiX 8** backend is selected (driver ≥ 560, e.g. RTX 6000 Ada /
driver 580), the radiation self-test
**`RadiationModel - Camera samples sky longwave on emission band miss`**
(`plugins/radiation/tests/selfTest.cpp:5553`) throws:

```
CUDA error in plugins/radiation/src/backends/OptiX8Backend.cpp:1053:
an illegal memory access was encountered
[OptiX][PIPELINE] Error synching on OptixPipeline event (CUDA error code: 700)
```

`OptiX8Backend.cpp:1053` is the `optixLaunch(...)` inside `OptiX8Backend::launchCameraRays()`
(function starts at line 989). The illegal access happens **inside the camera ray-tracing kernel**.

This is a **pre-existing OptiX 8 bug, unrelated to the translucent-glass-cover feature** — see "Why this
is not the glass feature" below. It is the **only** failure on OptiX 8: with this one test excluded, the
full radiation suite passes **96/96 test cases, 13,212 assertions, 0 failures**. Because the illegal
access poisons the CUDA context (sticky error), if this test is *not* excluded it cascades into ~46
spurious downstream failures (all reported at `OptiX8Backend.cpp:61`, the next `CUDA_CHECK` —
`cudaFree(nullptr)` in `initialize()`), which are red herrings.

## Why this matters

OptiX 8 is the default backend on modern NVIDIA drivers (≥ 560). Any user who renders a **radiation camera
bound to an emission (thermal/longwave) band** is likely to hit this crash. The Vulkan and OptiX 6 backends
are unaffected (the test passes there).

## Reproduce

```bash
cd utilities
# Default build selects OptiX 8 on driver >= 560. Reproduces in isolation:
./run_tests.sh --test radiation --project-dir cam_crash \
  --testcase "RadiationModel - Camera samples sky longwave on emission band miss" --verbose 2>&1 | tee /tmp/cam_crash.log
# Confirm the rest of the suite is otherwise green (exclude the crash; note the INNER quotes,
# run_tests.sh eval-splits --doctestargs on spaces):
./run_tests.sh --test radiation --project-dir cam_crash \
  --doctestargs '--test-case-exclude="*sky longwave*"' --verbose 2>&1 | tee /tmp/cam_excl.log
rm -rf cam_crash
```
Expect: first command FAILS (illegal memory access); second SUCCEEDS (96/96).

For a faster device-side signal, build Debug and run under `compute-sanitizer`:
```bash
# Build a persistent project, then run the test binary under compute-sanitizer to get the exact
# kernel + buffer + thread of the out-of-bounds access:
./run_tests.sh --test radiation --project-dir cam_crash --debugbuild --testcase "RadiationModel - Camera samples sky longwave on emission band miss"
# then locate the built radiation_tests binary under cam_crash/ and:
#   compute-sanitizer --tool memcheck ./radiation_tests --test-case="RadiationModel - Camera samples sky longwave on emission band miss"
```
`compute-sanitizer` (or `cuda-memcheck` on older toolkits) is the fastest path to the exact device buffer
and index being over-read/written.

## The triggering scenario (what's unusual about this test)

`plugins/radiation/tests/selfTest.cpp:5553`:
- A camera (`addRadiationCamera`, 8×8, HFOV 30°, looking up `+z`) is bound to band **"LW"**.
- **`emission` is ENABLED** for LW (it's a thermal band; per-band `emission_flag` uploaded as 1).
- **`setDirectRayCount("LW", 0)` and `setDiffuseRayCount("LW", 0)`** — zero scene rays launched.
- `setScatteringDepth("LW", 1)`, `setDiffuseRadiationFlux("LW", 400)`.
- Geometry is a single tiny patch placed far below the camera (so the frustum sees only sky).
- `runBand("LW")` → the **camera** launch crashes.

The smoking gun is the combination **camera + emission band + zero direct/diffuse ray counts**. Strong
hypothesis: a device buffer that the camera kernel dereferences is **null or zero-sized** because no
direct/diffuse passes ran to allocate/populate it, yet the camera closest-hit / miss / emission path indexes
into it unconditionally.

## Suspected root-cause candidates (investigate in order)

1. **Buffers sized by ray-count or by the diffuse/direct pass.** With direct=diffuse=0, any buffer that is
   allocated or filled only when rays are launched may be null/empty while the camera kernel still reads it.
   Audit `OptiX8Backend::launchCameraRays()` (line 989+) and what it binds in `OptiX8LaunchParams` vs what
   the camera device programs (`__raygen__camera`, `__closesthit__camera`, `__miss__camera` in
   `plugins/radiation/src/optix8/OptiX8DeviceCode.cu`) dereference. Look for pointers used without a null
   guard: `scatter_buff_top_cam` / `scatter_buff_bottom_cam`, `radiation_out_top/bottom`,
   `source_fluxes_cam`, `radiation_specular`, `sky_radiance_params`, `diffuse_peak_dir`,
   `diffuse_extinction`, `diffuse_dist_norm`, `rho_cam`/`tau_cam`.
2. **Emission path on the camera.** Emission-enabled bands populate per-primitive `radiation_out_*` from
   `εσT⁴`. The camera closest-hit reads `radiation_out_top/bottom[ind_hit]`. Confirm those buffers are sized
   `Nprims * Nbands_launch` and that the camera kernel's `ind_hit`/band indexing matches the layout used when
   emission filled them. An off-by-one in band indexing (`Nbands_launch` vs `Nbands_global`) would walk off
   the end. (`__miss__diffuse`/`__miss__direct` use `b_global` for rho/tau and `b` (launch) for
   radiation_in — verify the camera path is consistent.)
3. **Camera bound to a band with no scene rays.** `__miss__camera` samples sky longwave on miss
   (the test's purpose). Check `__miss__camera` for use of `diffuse_flux[b]`, `sky_radiance_params[b]`,
   `diffuse_peak_dir[b]`, etc., and whether those device buffers are non-null/populated when only emission +
   diffuse-flux (not diffuse *rays*) are configured. `setDiffuseRadiationFlux` was called but
   `setDiffuseRayCount=0` — is `diffuse_flux` uploaded in that case?
4. **`radiation_specular` for cameras.** `__miss__direct` (and possibly the camera path) writes
   `radiation_specular` indexed `[source][camera][prim][band]`. With a camera present but unusual
   source/ray-count config, confirm this buffer is allocated to the full size the kernel indexes.
5. **Tile/launch-dim math.** `launchCameraRays` sets `launch_dim_x = anti_samples`, `launch_dim_y = tile_w`,
   and launches `optixLaunch(..., anti_samples, tile_w, tile_h)`. Verify per-pixel buffer indexing in
   `__raygen__camera` against these dims for an 8×8 camera with `samples=1` — an indexing error here would be
   an OOB write into the pixel/accumulation buffer.

## Method

- Run under `compute-sanitizer --tool memcheck` first — it will name the exact kernel, the buffer, and the
  out-of-bounds offset/thread, which collapses the candidate list immediately.
- Add temporary `printf` guards in the suspected camera device program(s) (`OptiX8DeviceCode.cu`) printing
  the index and the buffer's expected size just before the dereference; remove before finishing.
- Cross-check the **OptiX 6** (`rayHit.cu` / `rayGeneration.cu`, build with
  `--cmake-args "-DOPTIX_VERSION_LEGACY=ON"`) and **Vulkan** (`camera_raygen.comp`) camera/emission paths,
  which DON'T crash, to see what guard or sizing OptiX 8 is missing.

## Constraints / conventions (Helios)

- Build/test ONLY via `utilities/run_tests.sh` (never build from `core/` or `plugins/` directly, never
  hand-create build dirs). Use `--project-dir <name>` for fast iteration; `rm -rf <name>` to clean up.
  Always `--verbose`; redirect to a log file.
- Fail-fast philosophy: if a buffer is legitimately absent in this configuration, the camera kernel must not
  silently read garbage — either guard the access (null-check + skip) or raise a clear `helios_runtime_error`
  on the host before launch. No silent fallbacks / fake values.
- The fix must keep parity with OptiX 6 and Vulkan (the camera-emission-sky result should match across
  backends).
- After fixing: the named test must pass, AND the full radiation suite must be 100% green on OptiX 8
  (`./run_tests.sh --test radiation --verbose`, no `--test-case-exclude`).

## Evidence summary

- First illegal access: `selfTest.cpp:5553` → `OptiX8Backend.cpp:1053` (`launchCameraRays` → `optixLaunch`),
  CUDA error 700. Reproduces **in isolation** (not a state-accumulation artifact).
- All other 46 full-suite "failures" are sticky-error cascades at `OptiX8Backend.cpp:61`
  (`cudaFree(nullptr)` in `initialize()`); 0 of 11,979 assertions actually failed.
- With `*sky longwave*` excluded: **96/96 cases, 13,212 assertions, SUCCESS** on OptiX 8.

## Why this is NOT the glass-cover feature

- `git show <glass-commit> --stat` touches no camera/emission code: the glass diff in `OptiX8DeviceCode.cu`
  adds only `glass_tau_rho_alpha`/`coverCosTheta`/`coverAnyHitBody`/`__anyhit__*` and per-band
  `cover_transmittance` handling in the **direct/diffuse** miss programs. The only "emission" token in the
  diff is a doc comment.
- The crashing test **pre-exists** the glass branch (not in the glass commit's `selfTest.cpp` additions).
- The crash occurs at `OptiX8Backend.cpp:1053` **before and after** the glass branch's only OptiX 8 device
  edit (the one-sided back-face occlusion fix at `OptiX8DeviceCode.cu:~418`/`~482`), i.e. independent of it.
- OptiX 8 had **never been executed** before this verification work (it was compile-verified only), so this
  latent bug simply surfaced the first time the suite ran on real OptiX 8 hardware.
