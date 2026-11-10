## What this project is

A user project built on Helios, a C++ library for 3D plant and environmental simulation. You write
the simulation in `main.cpp` (and other files in `SOURCE_FILES`); the library lives in the Helios
install named above. Do not edit Helios library sources to make a project work. If the API seems
to be missing something or to misbehave, stop and tell the user.

## Setting up a simulation case

1. **Pin down the question first.** Confirm with the user what is simulated, at what scale (leaf,
   plant, canopy), over what period, and which outputs they need. The outputs decide the plugins.
2. **Work from the docs and headers, not memory.** Helios changes between versions and your training
   data is likely out of date. Read the `.dox` file of every plugin you use (units, primitive data
   labels, required call order) and follow its worked example's call sequence. Check every class,
   method, and argument against the headers. Tutorials and samples are compilable starting points.
3. **Enable plugins in two places:** add the name to `PLUGINS` in `CMakeLists.txt`
   (semicolon-separated) and `#include` its header.
4. **Build only through the project's standard `CMakeLists.txt`.** Never compile with `g++`/`clang++`
   directly or write your own CMake file; edit only the variables in its `USER INPUTS` block.
   ```
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..   # the default, Debug, runs several times slower
   cmake --build .                       # add --config Release with Visual Studio generators
   ./<executable name>
   ```
5. **Read all build and run output.** Read stdout and stderr in full, every time. Helios prints a
   `WARNING` when an input is suspicious or a result may be wrong, and still runs to completion.
   Do not report success until every warning is resolved or reported to the user.
6. **Start small, then scale.** Run a reduced case first (few plants, coarse resolution, few time
   steps) and sanity-check the outputs physically before the full run.
7. **Look at every image before handing it over.** Open each image the simulation writes (Visualizer
   `printWindow()`, radiation camera images) and inspect it: scene in frame, not black, blank, or
   saturated, plausible sky, lighting, and colors. Fix problems before reporting.

## Conventions

- **Coordinates:** right-handed, z up, +y North, +x East; azimuth clockwise from North.
- **Angles:** core geometry uses **radians** (`SphericalCoord`, `make_SphericalCoord()`,
  `Context::rotatePrimitive()`, `Context::rotateObject()`), and `SphericalCoord` uses *elevation*,
  not zenith. Plugin setters often take **degrees**. Check each parameter name and docstring.
- **Units:** SI unless the docs say otherwise. Check, do not assume.
- **Plugins communicate through named primitive data.** A misspelled label or wrong data type fails
  at run time. Each plugin doc tabulates its input and output labels.

## Pitfalls

- **Errors are fatal on purpose.** Read the `helios_runtime_error` message and fix the cause. Never
  catch it to continue, and never substitute made-up default values.
- **Longitude and UTC offset are positive WEST** (`SolarPosition`, `Context::setLocation()`), the
  opposite of maps, GPS, and ISO 6709: Davis, CA (121.76 W, UTC-8) is +121.76 and +8.
- **The radiation plugin needs a GPU backend** (CUDA/OptiX or Vulkan); constructing a
  `RadiationModel` throws without one.
- **Periodic scenes need `RadiationModel::enforcePeriodicBoundary()`** (rows, a plot cut from a
  larger field). Without it, edge plants get extra side light and see a dark void below the horizon.
- **Radiation camera images need `SolarPosition` for realistic light:** the sun direction plus
  `calculateDirectSolarSpectrum()` and `calculateDiffuseSolarSpectrum()`, and for the most realistic
  sky `enablePragueSkyModel()` / `updatePragueSkyModel()`. Without a diffuse flux the sky renders
  black; a hand-made source has an unrealistic spectrum.
- **Photorealistic RGB renders need `RadiationModel::enableCameraFluxSmoothing()`**, or faceted
  curved surfaces render as flat panels. Leave it off when pixel values are used quantitatively:
  smoothed pixels no longer match the `radiation_flux_*` primitive data.
- **A default `CameraProperties` is a shallow-focus lens:** f/1.0 (`lens_diameter = 0.05`,
  `lens_focal_length = 0.05`) focused at 1 m, so anything at another distance renders blurred. Set
  `lens_diameter = 0` for a pinhole camera, or set `focal_plane_distance` to the subject distance.
- **Camera flux smoothing does not smooth direct-sun shadows.** They are still resolved facet by
  facet, so a coarse curved mesh that is partly shaded renders faceted. Tessellate curved organs
  finely if close-ups matter.
- **Run the executable from `build/`** (or one directory below it), or set the `HELIOS_BUILD`
  environment variable to the build directory. Plugin assets are resolved relative to the working
  directory, so running from elsewhere fails with "Could not resolve asset file".
- **Assign a material before setting a primitive's color.** The color is stored on the material, so
  setting it first and then assigning a new material leaves the primitive black.
- **No display:** construct the `Visualizer` with `headless = true`.
