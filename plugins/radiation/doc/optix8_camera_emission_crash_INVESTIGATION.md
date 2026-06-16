# Investigation progress + next steps: OptiX 8 camera + emission-band illegal-memory-access

> Companion to `optix8_camera_emission_crash_HANDOFF.md` (the original problem statement).
> This doc records what was done in the follow-up session and exactly what the next agent
> (on a **working GPU node**) needs to do to finish the fix.

## RESOLUTION (2026-06-15)

**Fixed and verified on a working GPU node (TITAN V, OptiX 8.1).**

**Environment unblock:** the GPU was never broken. On CHPC SLURM, the cgroup binds only the
*allocated* device (`SLURM_STEP_GPUS=4` → `/dev/nvidia4`, remapped to `CUDA_VISIBLE_DEVICES=0`).
The prior session tested `/dev/nvidia0` (never granted → EPERM) and wrongly concluded compute was
dead. `cudaGetDeviceCount`/`cudaMalloc`, OptiX 8.1, and Vulkan all work. Always probe CUDA directly
or open `/dev/nvidia${SLURM_STEP_GPUS}` — not `/dev/nvidia0`.

**Diagnosis:** compute-sanitizer can't instrument OptiX programs (only reported the host
`cuStreamSynchronize` detection point even with validation mode + cache disabled). A device-`printf`
bisection in `__miss__camera` did localize it: `Nsources=1` but `source_fluxes=(nil)`.

**Root cause:** `runBand()` auto-adds a default collimated source (flux unset → 0) when none exist
(`RadiationModel.cpp` ~3536), so `Nsources == 1`. But `uploadSourceFluxes` was only called inside
`if (Nsources>0 && rundirect)`; with no positive flux `rundirect == false`, so `source_fluxes` was
left null while `Nsources == 1` was still uploaded. The camera trace launches regardless, and
`__miss__camera`'s `for (s<Nsources)` loop dereferenced the null `source_fluxes[0]` → CUDA 700.
Candidate #1/#3 below were on the right track; the *executed* faulting deref was `source_fluxes`.

**Fix:**
1. `RadiationModel::runBand` — upload `source_fluxes` whenever `Nsources > 0` (zeros are skipped
   device-side by the `flux <= 0` guard), independent of `rundirect`. Backend-agnostic.
2. `OptiX8Backend::uploadSourceFluxes` / `uploadSourceFluxesCam` — assign `h_params.*`
   unconditionally so an empty upload yields `nullptr`, not a dangling freed pointer.
3. `selfTest.cpp` — strengthened the test (all pixels finite + positive sky signal).

**Note on the pixel value:** raw camera radiance is correctly `400/π ≈ 127.32`; the final ~0.7 pixel
is just `applyCameraExposure` scaling, NOT a bug — so the test does NOT assert `400/π`.

**Verification:** `samples/optix8_cam_repro` runs clean; full radiation suite **98/98 cases,
13,217 assertions, 0 failed, 0 skipped on OptiX 8.1**, no `--test-case-exclude`.

---

## Current status (historical — pre-resolution)

- **Root cause NOT yet pinpointed at runtime.** The session that wrote this doc could not run any
  CUDA/OptiX code because **GPU compute was unavailable on the cluster** (see "Environment blocker"
  below). So `compute-sanitizer` — the decisive step — was never able to run.
- **Static analysis is complete** and has narrowed the faulting access to a very small set of
  candidates (see "Static analysis findings").
- **A standalone reproducer is built and committed** (`samples/optix8_cam_repro/`) that drives the
  OptiX 8 backend directly, bypassing a test-harness gating bug that otherwise skips the test.
- **An `sbatch` harness is committed** to capture the sanitizer output on a GPU node without needing
  interactive GPU access.

## FIRST THING TO DO on a working GPU node

Run the reproducer under compute-sanitizer and read the output. Two ways:

**A) Directly (if your shell has working GPU compute):**
```bash
cd /group/bnbaileygrp/bnbailey/CLionProjects/Helios
# rebuild the reproducer if needed (see "Reproducer" below), then:
compute-sanitizer --tool memcheck --print-limit 0 ./samples/optix8_cam_repro/build/optix8_cam_repro
```

**B) Via batch job (works even if interactive GPU access is flaky):**
```bash
cd /group/bnbaileygrp/bnbailey/CLionProjects/Helios
sbatch samples/optix8_cam_repro/run_sanitizer.sbatch     # edit partition/account first if needed
# then read samples/optix8_cam_repro/sanitizer_<jobid>.log
```

compute-sanitizer will name the exact **kernel** (expected: `__miss__camera` or `__raygen__camera`),
the **buffer**, and the **out-of-bounds thread/offset**. That collapses the candidate list below to a
single fix. (For device-side line numbers, rebuild the reproducer Debug: `-DCMAKE_BUILD_TYPE=Debug`.)

## Reproducer (`samples/optix8_cam_repro/`)

- `main.cpp` — mirrors the failing self-test exactly, but constructs `RadiationModel(&context)`
  directly so the **OptiX 8** backend is used (the default ctor selects OptiX 8 on a CUDA build).
- Build:
  ```bash
  cd samples/optix8_cam_repro
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # or Debug for sanitizer line info
  cmake --build build --target optix8_cam_repro -j 8
  ```
- **Run from the repo root** (asset resolution needs `plugins/radiation/spectral_data/...` and the
  compiled device code `plugins/radiation/OptiX8DeviceCode.optixir`/`.ptx`). If the device code isn't
  found, set `HELIOS_BUILD` to the build dir or run with the repo root as CWD.
- Expected on the broken build: illegal memory access (CUDA error 700) at the camera `optixLaunch`.
  Expected after the fix: prints `pixels_LW.size() = 64`, pixel value ≈ `400/π ≈ 127.3`, and
  `REPRO COMPLETED WITHOUT CRASH`.
- `run_sanitizer.sbatch` / `probe.sbatch` — batch wrappers; edit `--partition`/`--account` to a healthy
  GPU partition. `probe.sbatch` is a minimal CUDA sanity check (`cudaGetDeviceCount`).

## Test-harness gotcha (why the self-test "passes" by skipping)

`plugins/radiation/tests/selfTest.cpp` `RadiationModelTestHelper::isGPUAvailable()` short-circuits to
`false` when `HELIOS_HAVE_VULKAN` is defined **and** `TestVulkanDeviceManager::isVulkanAvailable()` is
false — *before* it ever probes OptiX/CUDA. On a machine where Vulkan can't init but OptiX 8 works, every
GPU test (including this crash) is **silently skipped**, so `run_tests.sh` reports SUCCESS without
exercising OptiX 8. Watch for `MESSAGE: SKIPPED: No GPU/Vulkan device available` in the output — that
means the test did NOT run. The standalone reproducer exists precisely to dodge this gating. (Consider
fixing `isGPUAvailable()` to return true when `probeAnyGPUBackend()` succeeds regardless of Vulkan — but
that's a separate harness bug, not the crash.)

## Static analysis findings (what the executed path actually does)

The failing scenario (`selfTest.cpp:5553`): single band `"LW"`, emission ON, direct=diffuse ray counts
= 0, **no radiation sources** (`Nsources==0`), 8×8 camera looking `+z`, only primitive a tiny patch at
`z=-1000`. All 64 camera rays therefore **miss** → only `__raygen__camera` + `__miss__camera` execute
(`plugins/radiation/src/optix8/OptiX8DeviceCode.cu`). `Nbands_launch == Nbands_global == 1`, so the
launch-vs-global band-stride mismatch is NOT the trigger here.

Tracing every buffer access reachable in this exact configuration:
- `__raygen__camera` (`OptiX8DeviceCode.cu:1665`): only scalar params + `optixTrace`. Safe.
- `__miss__camera` (`OptiX8DeviceCode.cu:673-741`):
  - source loop skipped (`Nsources==0`),
  - sky-radiance and emission branches are **null-guarded** (e.g.
    `if (params.camera_diffuse_flux && params.camera_diffuse_flux[b] > 0)`),
  - `radiation_in_camera[pixel_idx*Nbands_l + b]` is freshly allocated to `full_w*full_h*Nbands_l = 64`
    floats (`OptiX8Backend.cpp:1010-1024`) and indexed in-bounds.

So in this config every reachable access is either guarded or size-1/size-64 and in-bounds — which means
the fault is subtle and **needs the sanitizer to localize**. Key architectural insight that frames the
fix: the OptiX 8 device guards (`if (params.ptr && params.ptr[b]...)`) are what OptiX 6 / Vulkan lack
(those rely on always-resident buffers). **The guards only work if the host sets `h_params.<ptr> =
nullptr` whenever it frees the underlying device buffer.**

### Candidate root causes (priority order)

1. **Dangling `h_params` pointers from early-return uploads (TOP SUSPECT, confirmed real bug):**
   `OptiX8Backend::uploadSourceFluxes` (`OptiX8Backend.cpp:1267-1275`) and `uploadSourceFluxesCam`
   (`:1277-1286`) do `freeCUdeviceptr(d_*)` then `return` on empty input **without** setting
   `h_params.source_fluxes(_cam) = nullptr`. That leaves a dangling (freed, non-null) pointer that
   defeats the device-side guards. `uploadSourceFluxesCam([])` is called immediately before **every**
   camera launch (`RadiationModel.cpp:4414`, empty when `Nsources==0`). Contrast the SAFE pattern in
   `updateSkyModel` (`:737-789`) and `updateDiffuseRadiation` (`:706-726`), which assign
   `h_params.* = d_*` unconditionally at the end (so empty ⇒ null).
   - Caveat: for the *isolated* repro, `__miss__camera` reads `source_fluxes` only when `Nsources>0`,
     so this exact bug may not be the faulting access for THIS test — but it IS a real defect and must
     be fixed regardless. The sanitizer will confirm whether it is the one.
2. **A buffer that is null/zero-sized in the camera-only-emission config** read without a guard. If the
   sanitizer points here, add the missing device-side null guard (match existing style) AND null the
   host pointer when absent.
3. **Stale `Nsources`** reaching `__miss__camera` with a freed `source_fluxes`.

### The fix (once the sanitizer confirms which buffer)

- Normalize **every** `freeCUdeviceptr(...)` + conditional-`cudaMalloc` pair in `OptiX8Backend.cpp` to
  the safe pattern: always set `h_params.<ptr> = reinterpret_cast<...>(d_<ptr>)` at the end so an empty
  upload yields `nullptr` (not a dangling pointer). Start with `uploadSourceFluxes` /
  `uploadSourceFluxesCam`, then audit the whole file.
- Keep parity with OptiX 6 (`rayHit.cu`/`rayGeneration.cu`, build `-DOPTIX_VERSION_LEGACY=ON`) and
  Vulkan (`shaders/camera_raygen.comp`). Fail-fast: absent buffers must be `nullptr` + guarded, never a
  dangling/garbage pointer (no silent fallbacks).

## Verification (after the fix)

1. Reproducer runs clean under compute-sanitizer (0 errors, `REPRO COMPLETED WITHOUT CRASH`).
2. Strengthen the self-test to assert the LW pixel ≈ `400/π` (currently only a size check),
   `selfTest.cpp:5553`.
3. Full radiation suite 100% green on OptiX 8, **no** `--test-case-exclude`:
   `cd utilities && ./run_tests.sh --test radiation --verbose`. Note: on a Vulkan-less machine the
   harness will skip GPU tests (see gotcha above) — verify on a node where the OptiX 8 path actually
   runs, or temporarily relax `isGPUAvailable()`.

## Environment blocker (why this couldn't be finished here)

On `gpu-10-50` (RTX 6000 Ada) and `gpu-4-54` (RTX A5500), during 2026-06-13:
- `nvidia-smi` works and SLURM reports a GPU allocated, but opening the compute device fails:
  `head -c0 /dev/nvidia0` → **"Operation not permitted"** (EPERM on a `0666` node), and
  `cudaGetDeviceCount` → **error 100/999**. The radiation backend therefore falls back to **Vulkan**
  (which then fails to find its `.spv` shaders), or reports "No compatible GPU backend".
- Driver versions are **consistent** (kernel module + `libcuda` + `libnvidia-ml` all `580.105.08`), so
  it is NOT a userspace/kernel version skew. The signature points to the **SLURM cgroup not binding the
  allocated GPU device into the job step** (`SLURM_JOB_GPUS` sometimes empty, `SLURM_STEP_GPUS=0`).
- This affected both the interactive shell and `sbatch` jobs. System maintenance was scheduled for the
  following day; re-test after the nodes are healthy (open `/dev/nvidia0` and `cudaGetDeviceCount`
  should succeed before attempting the OptiX 8 run).

## Files added/changed in this session

- `plugins/radiation/doc/optix8_camera_emission_crash_INVESTIGATION.md` (this file).
- `samples/optix8_cam_repro/` — standalone OptiX 8 reproducer + sbatch harness (sources only; build
  artifacts and logs are git-ignored).
- The original `plugins/radiation/doc/optix8_camera_emission_crash_HANDOFF.md` (problem statement) is
  also committed for the next agent.
