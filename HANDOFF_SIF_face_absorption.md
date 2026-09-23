# Handoff: per-face excitation absorption for SIF (Fluspect-B Mf/Mb face assignment)

**Branch:** `dev/sif-photosynthesis-coupling` (based on master at v1.3.88). Delete this file before merging to master.

Read the repo `CLAUDE.md` first. In particular: write a failing regression test **before** the fix, `GPU_TEST_CASE` for anything that constructs a `RadiationModel`, zero new compiler warnings, `--verbose` + absolute `--project-dir` under `utilities/.scratch/`, clean up scratch dirs.

## Background: what is already on this branch

A previous session audited the SIF chain (PhotosynthesisModel → `RadiationModel::calculateFluorescenceYield()` → `RadiationModel::computeSIFEmission()`) against van der Tol et al. (2014, open access: PMC4852699) and the SCOPE source (`src/fluxes/biochemical.m`, `src/RTMs/fluspect_B_CX.m`, `src/RTMs/RTMf.m` in github.com/Christiaanvandertol/SCOPE), and fixed:

- `electron_transport_ratio` is now Ja/Je (van der Tol's relative light saturation), not J/Jmax; the C4 model writes it too.
- `calculateFluorescenceYield()` uses the paper's Eq. 19 (cotton set Kn0 = 2.48, α = 2.83, β = 0.114) and Φ_P0 from the rate constants.
- `computeSIFEmission()` applies the kernel per nm of excitation (÷ bin width, interpolated to band centres), converts photons→energy (λe/λf), and acts on **incident** excitation. `populateExcitationIncidentFlux()` recovers incident from absorbed flux: at excitation scattering depth 0 Helios retains would-be-scattered energy in `radiation_flux`, so reported == incident; at depth ≥ 1 it divides by the 1−ρ−τ the ray tracer used (`material_data`).
- Missing `electron_transport_ratio` on a leaf with `fluspect_spectrum` is now an error; `integrateSpectrum(w1, w2)` clips exactly to its bounds.

See `doc/CHANGELOG.md` `[1.3.89]` and `plugins/radiation/doc/SIFCamera.dox` (section "Fluspect-B Excitation–Emission Kernel", which carries a `\note` describing the limitation you are fixing).

Note: a separate branch `origin/dev/nitrogen-senescence` has overlapping (different) fixes for Ja/Je, the bin-width scaling and `integrateSpectrum`. This branch supersedes them; do not merge them in.

## The problem to fix

Fluspect-B returns two excitation–emission matrices per leaf: **Mb** (backward) is the fluorescence leaving the face that *received* the excitation, **Mf** (forward) leaves the opposite face (`fluspect_B_CX.m` lines 53–54; `FluspectB.h` documents this correctly). `computeSIFEmission()` currently always puts Mf on the top face and Mb on the bottom face, because the ray tracer only reports the total absorbed flux per primitive. For a leaf lit on its top face (the common case) the two spectra are swapped. The difference is largest near 685 nm, where backward emission is much stronger than forward because red fluorescence is reabsorbed crossing the leaf.

Correct per-face emission, with I_top / I_bot the excitation incident on each face:

    F_top = Mb·I_top + Mf·I_bot
    F_bot = Mf·I_top + Mb·I_bot

## Agreed design (the user's constraint: zero cost when SIF is not running)

1. **One extra GPU buffer, `radiation_in_top`** [primitive × band_launch], same layout as `radiation_in`: absorbed flux that arrived on the top face. Bottom = `radiation_in` − `radiation_in_top`, computed on the host. `radiation_in` itself is unchanged.
2. **Auto-toggled.** `runBand()` sets a launch flag only when the launched band set contains `_SIF_exc_*` excitation bands (including when they are piggybacked onto another band's launch; see the piggyback logic around `runBand()` / `populateExcitationIncidentFlux()`). When the flag is off, allocate a 1-element placeholder so the Vulkan descriptor layout stays fixed, and skip the write with a uniform branch. Memory cost when off: 4 bytes; compute cost: one uniform branch per absorption write.
3. **Write sites** — surface absorption only (not glass-cover `cov_*` sites, not voxels). Each site already knows the face because the adjacent scatter-buffer code splits reflection/transmission by face:
   - OptiX 6.5: `plugins/radiation/src/rayHit.cu` ~899 (direct) and ~1042 (diffuse/scattering, non-voxel branch); face is `prd.face`.
   - OptiX 8: `plugins/radiation/src/optix8/OptiX8DeviceCode.cu` ~572 and ~671.
   - Vulkan: `plugins/radiation/shaders/direct_raygen.comp` ~753 and `shaders/diffuse_raygen.comp` ~700. Both accumulate with `subgroupAdd` then one atomic per subgroup; the top-face sum needs its own `subgroupAdd(is_top ? energy : 0.0)`, done only when the flag is set (keep the reduction uniform across the subgroup — every thread must call it).
   - Backend plumbing: allocation/zeroing/readback and `RayTracingResults` (`plugins/radiation/include/RayTracingTypes.h`) in `OptiX6Backend.cpp`, `OptiX8Backend.cpp` (+ `OptiX8LaunchParams.h`), `VulkanComputeBackend.cpp` (+ descriptor set layout).
4. **Depth-0 retention per face.** At scattering depth 0 the retired scattered energy is credited back into `radiation_flux` (`runBand()`, near `R.at(u) = radiation_flux_data + TBS_top + TBS_bottom`). The retired energy is already split into `retired_top` / `retired_bottom` by the face it will leave from, which is *not* the face it arrived on (transmitted energy leaves the other face). Work out the per-face incident attribution for the retained energy carefully; the invariant to satisfy is that, for an isolated leaf, per-face incident is independent of ρ and τ at any depth.
5. **Host side (also agreed): shrink the per-leaf buffer.** Today `ExcitationSet::incident_flux_buffer` stores `N_excitation_bands` floats per leaf (35 at 10 nm, ~140 MB per 1 M leaves) until the SIF band runs. Emission is linear in Φ_F·fqe, so at the end of the excitation launch compute, per leaf and per SIF emission band bound to that bin width, the Φ_F-free face emissions F_top and F_bot (with the per-face formula above), store only those (2 floats per band per leaf), and multiply by Φ_F·fqe in `computeSIFEmission()` once photosynthesis has written `electron_transport_ratio`. Keep the kernel cache (`getOrComputeFluspectKernel`) as is. Be careful that emission bands can be registered after the excitation set was populated (a later `addSIFCamera()`); either compute lazily or invalidate/recompute — do not return stale or zero emission silently.

Before implementing, confirm with a quick look at how `prd.face` / the Vulkan face test is defined relative to the primitive normal, so "top" means the same thing as `sif_emission_buffer` (top) / `sif_emission_buffer_bottom` in the emission loop.

## Tests (write first, watch them fail on the current branch)

Helpers already exist in `plugins/radiation/tests/selfTest.cpp`: `sif_stamp_biochem()`, `sifLeafEmission()` / `SIFLeafEmissionSetup`, and `RadiationModelTestHelper::getSIFEmission(model, band, UUID, top_face)` and `::calculateFluorescenceYield()`. Extend `sifLeafEmission` to return both faces and to take the sun direction.

1. **Face assignment, lit from above**: horizontal leaf, flat 1 W/m²/nm sun from +z, 10 nm bins, depth 0. Top-face emission must equal the Mb product and bottom the Mf product (compute the expected values from `computeFluspectKernel()` exactly as the existing test "SIF regression: leaf emission equals the Fluspect-B matrix product on incident photon flux" does). Fails today (top = Mf). Use an emission band where Mb and Mf differ a lot (e.g. 680–700 nm) so the check is decisive.
2. **Lit from below**: same leaf, sun from −z → top = Mf, bottom = Mb.
3. **Both faces lit**: two collimated sources, +z and −z, with different fluxes (e.g. 1.0 and 0.3 W/m²/nm) → F_top = Mb·I_top + Mf·I_bot, F_bot = Mf·I_top + Mb·I_bot.
4. **Optics invariance per face**: repeat 1 and 2 with leaf ρ = 0.3, τ = 0.2 at depth 0 and depth 1; results unchanged (extends the existing optics-invariance test).
5. **Flag off costs nothing**: a non-SIF `runBand()` must not allocate the full-size buffer — expose a test-helper query of the allocated size, or equivalent, and assert it is the placeholder.
6. Update the existing matrix-product test (it currently expects Mf on the top face for a top-lit leaf; it must become Mb) and re-check the Tier 2 test's red/far-red expectations, which were written for the swapped assignment.

Remove the `\note` about the fixed Mf/Mb assignment from `SIFCamera.dox` and the comment in `computeSIFEmission()`, and add a CHANGELOG bullet under `[1.3.89]` → `## Radiation`.

## Running on all three backends

On an NVIDIA machine with driver ≥ 560 (default build picks OptiX 8):

```bash
cd utilities
# OptiX 8 (default)
./run_tests.sh --test radiation --requiregpu --verbose --project-dir /abs/path/utilities/.scratch/sif_optix8 > /tmp/sif_optix8.log 2>&1
# OptiX 6.5 (legacy)
./run_tests.sh --test radiation --requiregpu --verbose --cmake-args "-DOPTIX_VERSION_LEGACY=ON" --project-dir /abs/path/utilities/.scratch/sif_optix6 > /tmp/sif_optix6.log 2>&1
# Vulkan
./run_tests.sh --test radiation --requiregpu --verbose --force-vulkan --project-dir /abs/path/utilities/.scratch/sif_vulkan > /tmp/sif_vulkan.log 2>&1
# No-GPU CI configuration
./run_tests.sh --test radiation --nogpu --verbose --project-dir /abs/path/utilities/.scratch/sif_nogpu > /tmp/sif_nogpu.log 2>&1
./lint_gpu_tests.sh
```

Confirm from each configure log (`[Radiation] ...` status lines) which backend was actually built — do not assume. `--requiregpu` makes a silently skipped GPU test a failure. `grep -n "warning:"` each log. Also run the full suites of plugins that depend on radiation: `energybalance`, `leafoptics`, `syntheticannotation`, `projectbuilder`, `photosynthesis`.

**Performance check (the user's main concern):** time a non-SIF radiation workload before and after on each backend (e.g. `benchmarks/radiation_homogeneous_canopy`, or the radiation suite's heaviest cases) and report the numbers; the flag-off path should be indistinguishable. Also report peak GPU memory for a SIF run at 10 nm bins with and without the change, and host memory of the per-leaf SIF buffers.

## Deliverable

Commit to this branch (do not push to master), and report: what changed per backend, test results per backend (pass counts, zero warnings), the performance and memory numbers, and anything you could not verify. Clean up `utilities/.scratch/` directories you created.
