# SCOPE v2.0 reference data for Helios SIF V&V Tier 2

This directory contains frozen outputs from a single-point run of the SCOPE v2.0
vegetation radiative transfer and fluorescence model. Helios's Tier 2 SIF test
(`plugins/radiation/tests/selfTest.cpp`, test case "SIF V&V Tier 2: SCOPE
intercomparison at LAI=3") reads these values at test time and compares
Helios's simulated top-of-canopy SIF against them.

The reference is **frozen** — committed once, used every test run. No MATLAB
is required to run the Tier 2 test. See below to regenerate if SCOPE changes.

## Files

| File | Purpose |
|------|---------|
| `scope_v2_homogeneous_lai3.csv` | key/value reference data consumed by the test |
| `generate_scope_reference.m`    | MATLAB driver that regenerates the CSV from a local SCOPE clone |
| `README.md`                     | this file |

## Provenance

- **SCOPE version**: git commit `e4c2e51` ("correct error in RTM_sb.m"), GitHub: <https://github.com/Christiaanvandertol/SCOPE>
- **MATLAB version**: R2026a
- **Run date**: 2026-04-20
- **Operator**: bnbailey
- **SCOPE input deck**: the stock `input/input_data_default.csv`, unchanged. Relevant parameters:
  - `LAI=3`, `hc=2`, `LIDFa=-0.35`, `LIDFb=-0.15` (spherical leaf-angle distribution)
  - PROSPECT-D/CX leaf biochemistry: `Cab=40`, `Cca=10`, `Cdm=0.012`, `Cw=0.009`, `N=1.5`
  - Solar position: `tts=30°`, `tto=0°`, `psi=0°`
  - Forcing: `Rin=600 W/m²`, `Ta=20 °C`, `ea=15 mbar`, `u=2 m/s`, `Ca=410 ppm`
  - Fluorescence: `fqe=0.01`
  - Soil: default `soilnew.txt` (`spectrum=1`, `rss=500`, `BSMBrightness=0.5`)
- **SCOPE options** (`input/setoptions.csv`): `lite=1`, `calc_fluor=1`, `calc_planck=1`, `calc_xanthophyllabs=1`, `applTcorr=1`, `verify=0`.
- **Working directory dispatch** (`set_parameter_filenames.csv`): `setoptions.csv, filenames.csv, input_data_default.csv`

## How to regenerate

If SCOPE is updated and these values need refreshing:

1. Clone SCOPE somewhere outside the Helios repo (e.g. `~/Downloads/SCOPE/`).
   ```
   git clone https://github.com/Christiaanvandertol/SCOPE.git ~/Downloads/SCOPE
   ```
2. In SCOPE's top-level `set_parameter_filenames.csv`, confirm the line is:
   ```
   setoptions.csv, filenames.csv, input_data_default.csv
   ```
3. In SCOPE's `input/setoptions.csv`, set `0,verify` (disables the LHC verification ensemble — we want a single-point run).
4. Run the driver:
   ```
   /Applications/MATLAB_R2026a.app/bin/matlab -nodisplay -nosplash -nodesktop \
     -batch "run('generate_scope_reference.m');"
   ```
   The driver runs SCOPE, extracts the values from SCOPE's output directory, and
   overwrites `scope_v2_homogeneous_lai3.csv`.
5. Update the git commit hash and run date at the top of the CSV.
6. Re-run the Tier 2 test locally — it should still pass at the documented
   tolerances. If it fails substantially, investigate before committing.

## What's in the reference CSV

The CSV is a flat key/value table with columns `key,value,units,note`. Consumers:

- **Scene parameters** (LAI, SZA, PROSPECT inputs) — documented so the Helios
  scene in the Tier 2 test can match SCOPE's setup exactly.
- **Leaf optics at SIF bands** (`leaf_rho_*`, `leaf_tau_*`) — the Helios test
  stamps these directly onto leaf primitives as `reflectivity_SIF_*` and
  `transmissivity_SIF_*` primitive data. This removes LeafOptics from the
  comparison and isolates the 3D radiative transfer test.
- **Soil optics at SIF bands** (`soil_rho_*`) — applied as `reflectivity_SIF_*`
  on the ground tile, so soil scattering is matched between SCOPE and Helios.
- **SCOPE fluorescence outputs** (`sif_hemis_band_*`, `sif_emitted_band_*`,
  `f_esc_band_*`) — the Helios test primarily compares **escape fractions**,
  computed as `hemispheric canopy-leaving flux / total SIF produced by leaves`.
  Escape fractions are the cleanest metric because they are insensitive to the
  absolute SIF source magnitude (Helios v1 uses a fixed 25/75 red/far-red
  spectral split; SCOPE uses the more realistic Fluspect-B PSII/PSI emission
  spectrum, so source magnitudes per band differ non-trivially between the
  two models).

## Why escape fraction, not absolute flux

Helios v1 partitions each leaf's total fluorescence source (`APAR × Φ_F`) as
25% into SIF_red and 75% into SIF_farred — a fixed simple approximation. SCOPE
instead uses the Fluspect-B spectral emission, where the red PSII peak is much
taller than the far-red PSI peak. As a result, for the LAI=3 reference run:

- **SCOPE red-band emission** (pre-reabsorption): 1.63 W/m²
- **SCOPE farred-band emission** (pre-reabsorption): 1.42 W/m²
- Source ratio red/farred ≈ 1.15, NOT 0.33 as Helios assumes.

Direct absolute comparison is therefore unfair: even with a perfect ray tracer,
Helios would produce a red-band canopy-leaving flux that is too small by a
factor of ~3 relative to SCOPE's red source strength. The canopy **escape
fraction** (what Helios's 3D ray tracing actually computes) is the isolatable
piece of physics we can compare:

- `f_esc = F_canopy_top / F_total_emitted_by_leaves`

Both sides can report this quantity, and it's the core thing we want to
verify. The expected match: within ~15% (far-red) / ~20% (red) per the DART vs
SCOPE homogeneous-canopy literature (IGARSS 2021 — see bottom of this file).

The absolute-flux reference values (`sif_hemis_band_*`) are retained in the CSV
for diagnostic use, but are NOT asserted in the Tier 2 test.

## Known 1D-vs-3D differences to expect

SCOPE uses a 1D SAIL-family canopy transport with homogeneous layers and the
hotspot parameterised analytically. Helios uses full 3D ray tracing through a
Poisson-like point cloud of leaves with periodic boundaries. Even with matched
leaf optics and LAI, the two models should not agree exactly:

- Expected red/far-red ratio agreement: within ~15–25% (literature tolerance
  for DART vs SCOPE homogeneous-canopy comparisons — IGARSS 2021,
  <https://ieeexplore.ieee.org/document/9323616/>).
- Absolute SIF_farred agreement: tighter than red, since far-red is less
  sensitive to structural details (reabsorption is weak).
- Absolute SIF_red agreement: looser (≈ 20%), since red is more sensitive
  to canopy gap fraction and leaf-angle-specific absorption.

The test currently asserts 15% (far-red) / 20% (red) per the V&V plan.

---

# Homogeneous canopy fixture (uniform leaf angle distribution)

`homogeneous_canopy_uniform_LAI1.obj` (+ `.mtl`) is the leaf geometry for the test
case "RadiationModel Homogeneous Canopy with Periodic Boundaries". The test checks canopy
interception against Beer's law and the probability distributions of absorbed direct
and diffuse flux against the theory of Bailey and Fu (2022), Agric. For. Meteorol.
322:109009.

## Geometry

- Domain 10 x 10 m, canopy height 3 m, LAI = 1 (one-sided leaf area / ground area)
- 10,000 square leaves of width 0.1 m, centers uniformly random in the domain
- Uniform leaf angle distribution: leaf normal zenith angle uniform on [0, 90°],
  azimuth uniform on [0, 360°)
- Context RNG seeded with `seedRandomGenerator(5)`

The leaves were sliced at the domain boundary with
`CollisionDetection::slicePrimitivesUsingGrid()` (one cell, `grid_center=(0,0,1.5)`,
`grid_size=(10,10,6)`). Every piece outside the cell (`cell_ID == -1`) was moved by
±10 m in x and/or y to the opposite side of the domain, based on its centroid. The canopy
therefore fills the box exactly, with no leaf area lost or overhanging, and the periodic
domain is homogeneous right up to its edges. The result has 10,691 primitives (patches and
slice triangles). After `loadOBJ()` these become 20,466 triangles.

To regenerate it, run this snippet in a project with the `collisiondetection` plug-in,
then call `writeOBJ()` on the sliced UUIDs:

```cpp
context.seedRandomGenerator(5);
for (int i = 0; i < 10000; i++) {
    vec3 position((-0.5f + context.randu()) * 10.f, (-0.5f + context.randu()) * 10.f, 0.05f + context.randu() * 2.9f);
    SphericalCoord rotation(1.f, 0.5f * float(M_PI) * context.randu(), 2.f * float(M_PI) * context.randu());
    UUIDs.push_back(context.addPatch(position, make_vec2(0.1f, 0.1f), rotation));
}
std::vector<uint> sliced = collisiondetection.slicePrimitivesUsingGrid(UUIDs, make_vec3(0, 0, 1.5f), make_vec3(10, 10, 6), make_int3(1, 1, 1));
// translate each primitive with cell_ID == -1 by +/-10 m back into the domain, then clear "cell_ID"
```
