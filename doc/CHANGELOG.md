#Changelog

# [1.0.0] - 2018-11-15

**** Initial commit ****

# [1.0.1] - 2018-11-26

## Radiation Model
- Running radiation model for only a subset of UUIDs was not working properly (not implemented correctly).
- Texture-masked patches were causing an OptiX error under certain cases. Cause was an indexing typo in rayGeneration.cu

## Visualizer
- Visualizing texture-masked patches by primitive data was not working properly
- File needed in selfTest() function was missing (Helios_logo.jpg)

# [1.0.2] - 2018-11-26

## Visualizer
- Error corrected with adding texture-mapped triangles in buildContextGeometry() function.

# [1.0.3] - 2018-12-13

## Context
- Errors in timeseries data querying were corrected in which some date/time combinations did not work correctly. Also, if a time/date is queried that is outside the range of all the data in the timeseries, the error will be caught rather than producing a segmentation fault.

## Visualizer
- colorPrimitivesByData() now supports primitive data types of int, uint, and double. Note: it will convert these data types to float.

## LiDAR
- Functionality added to be able to visualize triangulated fill groups used to create the leaf reconstruction: see function addReconstructedTriangleGroupsToContext().
- Calling addLeafReconstructionToContext() now creates primitive data for each leaf called ``directFlag" which equals 1 if the leaf came from the direct reconstruction, and equal to 0 if the leaf was backfilled.
- An initial implementation of functions to deal with aerial LiDAR data has been implemented, but has not been fully tested.

# [1.0.4] - 2018-12-17

## Context
- Further errors corrected in timeseries data query due to precision issues near beginning/end of timeseries
- Errors corrected in calculating the area of primitives. Areas were not properly calculated for patches with custom (u,v) coordinates, and for primitives where a scaling transformation is applied, or when primitives were read from an xml file.

# [1.0.5] - 2019-1-24

## Photosynthesis
- Added temperature response functions to Farquhar model following Bernacchi et al. (2001)
- Error corrected in where PAR was not converted from W/m^2 to umol/m^2/s

## Energy Balance
- Error corrected which caused a segmentation fault in the energy balance model if any primitives are deleted from the Context.

## Weber-Penn Tree
- Default tree library updated. Currently contains nine trees: almond, apple, avocado, lemon, olive, orange, peach, pistachio, and walnut.
- Error fixed that caused tree height to vary with "Shape" parameter if "BaseSplits">0.

## Voxel Intersection
- Error corrected that caused compilation to fail if voxel intersection plugin and lidar plugin are both used in the same build.

# [1.0.6] - 2019-04-02

## Context
- Added file core/lib/libpng/pnglibconf.h. Not having this file causes problems on some systems.
- Error fixed with Patch::getNormal() function.
- selfTest() tests added for rotate patches, and boxes.

## LiDAR
- Aerial LiDAR scan can now have a coneangle of zero without causing problems
- Aerial LiDAR updated to use advanced method for inverting Beer's law

# [1.0.7] - 2019-04-03

## Context
- Implemented acos_safe() and asin_safe() functions to catch instances where round-off errors cause normal acos() and asin() to return NaN. The new functions were implemented with the LiDAR and solarposition plug-ins where acos() or asin() was used previously.
- Error was corrected from v1.0.6 in which voxels sizes were not properly set when using addVoxel().

# [1.0.8] - 2019-05-09

## LiDAR
- Aerial LiDAR reverted to before v1.0.6, except that acos_safe() function is still implemented. There are still apparently problems with the changes implemented in v1.0.6.

# [1.0.9] - 2019-05-24

## Radiation
- Having a radiation band name longer than about 8 characters would cause a segmentation fault. You can now have names of up to 80 characters.

## Energy Balance
- There is now a function optionalOutputPrimitiveData() that allows you to output the calculated boundary-layer conductance and vapor pressure deficit to primitive data.

## LiDAR
- Added new scheme for inversion of LAD from aerial LiDAR data. When vegetation in a voxel is very dense, a method is used that fits to the hit point distance PDF.
- Added aerial LiDAR self-test and sample to run the self-test.

# [1.0.10] - 2019-05-30

## Context
- Error fixed when calling addTile() with a non-zero spherical rotation.

## LiDAR
- Certain compiler versions would cause a compiler error with the 'fmin' function, which has been corrected.

# [1.0.11] - 2019-07-24

## Context
- Error fixed with texture mapping of tiles.

## Radiation
- Error fixed that could cause an infinite loop in certain cases where the primitive has a texture mask.

# [1.0.12] - 2019-08-08

## Context
- Adding texture-masked tiles with a transparency channel was previously excessively slow. A new implementation was added such that there is no slow-down when texturing a tile.

## Radiation
- Error fixed that caused errors in absorbed radiation when the scence contains two different texture masks of different size.

## LiDAR
- Stability improvements.

# [1.0.13] - 2019-08-09

## Context
- Error corrected causing incorrect rotation of tile sub-patches.
- Additional selfTest() cases added for rotated and texture-masked tiles.

# [1.0.14] - 2019-09-14

## Context
- Minor change to randu() functions for case when the minimum range and maximum range arguments are equal.
- calculateTriangleArea() function added to calculate area of a triangle given its three vertices (in global.h/global.cpp).
- Copying texture-mapped triangles is much faster (similar implementation as for patches added.
- Functions added to translate, rotate, and scale primitives or groups of primitives directly from the Context without having to get their individual pointers (see translatePrimitive(), rotatePrimitive(), scalePrimitive()).

## Radiation Model
- Calculations for diffuse radiation in a canopy were not correct, causing the flux on leaves to be under predicted (but the flux reaching the ground was correct).
- Self-tests added based on a homogeneous "canopy" of patches and comparing against the exact Beer's law solution for collimated and diffuse radiation.

*Energy Balance Model*
- Some changes were needed based on the above changes to the radiation model.
- Functions added to disable or enable command-line output messages.

## Visualizer
- Error corrected to function colorContextPrimitivesByData() with a UUID vector argument, which did not properly separate primitives by UUID.

# [1.0.15] - 2019-10-04

## Context
- Minor changes to standard CMakeLists.txt file to allow for custom CXX compiler flags.

## Radiation Model
- Freeing GPU memory was previously accomplished via the finalize() function. Failing to call the finalize() function, say inside a loop, results in a GPU memory leak. This function has been removed, and now GPU memory is automatically done in the RadiationModel class destructor.
- selfTest() cases added for an enclosure (furnace) filled with a non-scattering participating medium, and infinite parallel plates filled with a purely scattering participating medium.

## Visualizer
- Very minor change to the command-line output messages (easier to read).

## Weber Penn Tree
- Option to set the the branch, trunk, and leaf subdivision resolutions (see functions setBranchSegmentResolution(), setTrunkSegmentResolution(), and setLeafSubdivisions()).
- Option to dynamically modify the architectural parameters for a tree in the library (see functions getTreeParameters() and setTreeParameters()).

# [1.0.16] - 2019-11-07

## Context
- Primitives with a texture can now be colored based on the R-G-B color set in the Context, but have its shape masked using the transparency channel of the texture. This is accomplished with the new Context::overrideTextureColor() function.
- Functions added to calculate bounding box (getDomainBoundingBox()) or bounding sphere (getDomainBoundingSphere()) of a subset of primitives given a vector of their UUIDs.

## Visualizer
- Minor bug fixed that would not update colorbar ticks if buildContextGeometry() was never called.
- Performance enhancements when rendering a subset of primitives based on a UUID vector.

## Weber-Penn Tree
- Leaf UUIDs were not properly assigned when leaves are segmented with sub-patches.

## LiDAR
- First attempt to add several new features to aerial LiDAR data processing:
	+ Use RANSAC to determine the ground height and effective canopy height
	+ Calculate maximum canopy height
	+ Shift grid in vertical direction to line up with ground (see alignGridToGround() funciton)
	+ selfTest() case to check these calculations
- Added option to addLeafReconstructionToContext() to all for sub-patch distretization (tile) of leaves.
- If the total number of grid cells (aerial LiDAR) was larger than 2,097,120 then leaf area calculation would error out without any useful error message. A catch for this has been implemented.

# [1.0.17] - 2019-11-08

## Context
- "Swizzle" capability for vec3 removed.
- Error corrected in getDomainBoundingBox() that could cause an error if primitives are deleted from the Context.

## Weber Penn Tree
- Re-implemented leaf sub-patch tiling to be much faster.

## LiDAR
- Re-implemented leaf sub-patch tiling to be much faster.

# [1.0.18] - 2019-11-14

## Context
- addBox() functions added to allow for texture mapping of boxes.
- useTextureColor() function added, which can be used to reverse a previous call to overrideTextureColor().

## Radiation
- Previously, the maximum total number of rays for a scene could not exceed ~50 billion, otherwise the radiation model would not actually launch any rays. If a ray launch exceeds 50 billion rays, the launch is now segmented into multiple launches to avoid this problem.
- There was some inconsistency between references for the "twosided flag" primitive data, where in some instances the primitive data 'twosided-flag' was referenced or 'twosided_flag' in others. It is now always 'twosided_flag'.

## Weber-Penn Tree
- Added selfTest() function, which builds all trees in the default library.
- Option to manually seed the random number generator for reproducable trees.

## Visualizer
- Stability improvements.

# [1.1.0] - 2020-02-07

🚨+ NEW PLUG-IN + 🚨
- Aerial LiDAR was separated from the terrestrial LiDAR plug-in and moved to its own plug-in.

## Context
- Error corrected in which textures became flipped about x- and y-axes.
- Voxels can now be texture-mapped.
- Error corrected in which the area of texture-mapped triangles could potentially be calculated incorrectly.

## Visualizer
- Error corrected in which textures became flipped about x- and y-axes.
- Visualization of voxels was temporarily disabled due to an error, which has been fixed.

## LiDAR
- Stability improvements for terrestrial LiDAR leaf reconstruction.
- Added capability to perform synthetic scans of texture-mapped patches and triangles.

## Photosynthesis
- Optional outputs added to allow writing C_i and photosynthesis limitation state to primitive data.
- An error was corrected that resulted in no response of Vc,max to temperature variation.

## Radiation
-Error corrected in which textures became flipped about x- and y-axes.

## Energy Balance
- Capability added to solve unsteady energy balance equation with heat storage.

# [1.1.1] - 2020-04-10

## Visualizer
- Correction was not picked up in previous version that caused Helios watermark to be flipped.

# [1.1.2] - 2020-04-14

## Radiation
- Periodic boundary condition added. See function RadiationModel::enforcePeriodicBoundary().

## Aerial LiDAR
- Documentation page added, but still under construction.

# [1.1.3] - 2020-06-16

## Context
- Further problems with patch texturing corrected. A lingering error was causing patches with custom (u,v) coordinates to have zero surface area. This was likely introduced in v1.1.0 in correcting a similar error with triangles.

## LiDAR
- For full-waveform scans, leaf area calculations now use equal weighting of hit points.
- Function added to gap fill missing scan points due to "sky" hit points (see function gapfillMisses()). This function is only incorporated within the "testing" leaf area calculation function calculateLeafAreaGPU_testing().

## Aerial LiDAR
- Aerial LiDAR sample was not compiling because of an error in the CMakeLists.txt file.

# [1.1.4] - 2020-06-18

## Aerial LiDAR
- Functions added to get the beam mean free path and numerator and denominator of the probability of interception calculated in calculateLeafAreaGPU(). See functions getCellTransmissionProbability() and getCellRbar().

# [1.1.5] - 2020-08-07

🚨+ NEW PLUG-IN + 🚨
- Canopy Generator: Initial testing version of Canopy Generator plug-in added.

## LiDAR
- Files were not properly copied to allow running the selfTest() function from a directory other than samples/lidar_selftest. You should now be able to run the selfTest() from any directory in which the lidar plug-in is built.
- Error corrected in which sine weighting of G(theta) in calculateLeafAreaGPU() was not performed correctly.

## Radiation
- There was a problem with the periodic boundary condition with diffuse radiation that could cause unexpected results. This was due to rare instances when a ray would hit the periodic boundaries many times and apparently cause a stack overflow (although no error is actually thrown). This was corrected by limiting the maximum number of times a ray could hit the periodic boundary.

## Photosynthesis
- Error corrected in FvCB photosynthesis model solution.

## Visualizer
- Helios watermark corrected to scale properly as aspect ratio of visualizer window is changed.

## Weber-Penn Tree
- Issue corrected in which leaves not positioned correctly with respect to parent branch.

# [1.1.6] - 2020-08-19

## Context
- Overloaded functions added to set primitive data based on a 2D or 3D vector of UUIDs (see setPrimitiveData() functions).
- If a patch is created with size of 0, an explicit error message is thrown rather than triggering an ambiguous "assert".
- Functions added to flatten a 2D or 3D vector into a 1D vector (see "flatten" function).

## Weber-Penn Tree
- Issue with leaf rotations from v1.1.5 was not completed fixed, but appears to be working now.

## Radiation
- If the user mistakenly sets emissivity<1 but keeps scattering iterations at 0, the model would not satisfy the second law of thermodynamics because there would be missing energy. This has been changed such that the model will default to emisssivity=1 if scattering iterations is 0.
- If the user sets emissivity<1 but keeps the default reflectivity and transmissivity for the band, it is now automatically assumed that reflectivity=1-emissivity and transmissivity=0.

## Visualizer
- Issue fixed in which there could be white "speckles" on the edges of texture-mappped primitives with transparency.

## Canopy Generator
- Function added to get UUIDs for ground primitives.
- Error in grapevine geometries corrected that caused the radius of the end of each shoot to be very large.
- Default shoot radius changed for grapevine plants.

# [1.1.7] - 2020-08-24

## Context
- Functions added to crop the domain based on some axis-aligned bounding box. This is especially helpful when using periodic boundaries with the radiation model to delete any primitives that may accidentally lie beyond the ground surface.

## Canopy Generator
- canopygenerator_selftest case added to 'samples' directory, and is now included in the run_samples.sh script.
- Added variable for grapevine canopies to change the color of the grapes.
- Corrected error in getBranchUUIDs(), getLeafUUIDs(), and getFruitUUIDs(), which could incorrectly throw an error if a trunk does not exist.
- If primitives created by CanopyGenerator are deleted from the context, their UUIDs will now not be returned by getTrunkUUIDs(), getBranchUUIDs(), getLeafUUIDs(), getFruitUUIDs(), and getAllUUIDs() functions.

*Energy Balance Model*
- Documentation added for unsteady energy balance model.
- Input primitive data for unsteady energy balance model changed. Users now specify the object heat capacity instead of the object heat capacity.

# [1.2.0] - 2020-09-11

+++++ NEW FEATURE ++++++
- Compound Objects: Functionality to group primitives into compound object has been added (tiles, spheres, tubes, boxes, disks). When a compound object is added, information about the object as a whole is retained, rather than just adding each primitive separately. For example, when a tile object is added (see Context::addTileObject() function), the original dimension of the tile and subdivision resolution can be later queried. The current implementation is an initial draft with further development needed. It is not yet documented in the User's Guide.

+++++ PLUG-IN RELEASE ++++++
- Canopy Generator: The Canopy Generator plug-in has been tested by several users, and appears to be relatively stable. The base functionality has been implemented, and documentation is relatively thorough.

## Context
- Function added to add a "default" patch to the Context (see Context::addPatch( void ))
- Function added to clear the primitive data for a given primitive (see Primitive::clearPrimitiveData() and Context::clearPrimitiveData()).
- Error corrected in which Triangle::getVertices() returned a 4-element vector, with the last element equal to (0,0,0).
- deletePrimitive() function now clears primitive data for the primitive(s) to free up the memory.
- Copying a primitive now also copies the primitive data for that primitive.

## Energy Balance
- Error in selfTest() causing failure of Case #5 (actual source code was fine).

## Canopy Generator
- samples/canopygenerator_selftest had not been added to git control, and thus was missing in the repository.
- Added capability to change fruit color.
- Added capability to change the number of triangles used to make fruit spheres.

## Radiation
- CMakeLists.txt modified to be more robust - it was possible if there was an erroneous file in the src/ directory with a .cu extension, CMake would try to build it and fail.

## Visualizer
- Upgraded GLEW library to v2.2.0
- Upgraded GLFW library to v3.3.2
- Fixed issue in which writing an image to file (printWindow()) would render a blank image.

# [1.2.1] - 2020-11-18

## Context
- addTile() function will automatically delete sub-patches with zero solid area due to transparency mask.
- An error was corrected in the getVertices() function for a Tile object.

## Canopy Generator
- Fixed issue in which seeding the random generator still resulted in inconsistent random generaton of the grape clusters.
- Fixed issue with "Split" vineyard canopy that resulted in odd behavior if the canopy was rotated.

## Visualizer
- The size/with of pixels is now actually set when passed as an argument to addPoint(). However, one limitation is that the size of points is constant within a scene and set based on the last value passed to addPoint(). Hopefully a future release will allow for variable point size within a scene.

## Energy Balance
- There was a discrepancy between the documentation and source code behavior for setting the storage heat flux, which has been made consistent.

## LiDAR
- Fixed an error that could cause out of bounds indexing if primitives have been deleted from the Context.

## Aerial LiDAR
- Fixed an error that could cause out of bounds indexing if primitives have been deleted from the Context.

# [1.2.2] - 2020-11-25

## Canopy Generator
- An initial tomato model was added to the canopy generator in v1.2.1, but the file src/tomato.cpp was not added to the git repository, which will cause an error while building the canopy generator.

# [1.2.3] - 2020-12-02

## Context
- Function was added to get the normal vector (getNormal()) of a compound tile object.

## Canopy Generator
- The canopy generator now uses compound tile objects for leaves.
- Leaves are now created by copying a prototype element, which greatly improves efficiency when building in the case that leaves are transparency masked.
- There was an error in the spherical crowns canopy causing the leaf angle distribution to be non-spherical.
- There was an error in the "split" vineyard canopy in which the UUIDs for cross-arms were not being stored.

## Weber Penn Tree
- Leaves are now represented by compound tile objects.

# [1.2.4] - 2020-12-10

## Context
- Error handling has been improved. Errors in the Context no longer call "exit(EXIT_FAILURE)", but rather call "throw(1)". This allows for catching of errors in a debugger to provide a stack trace and line numbers for the errors. Future releases will throw more specific and informative error codes, but for now "1" is always thrown.

## Visualizer
- An error was corrected related to window sizing on a Mac, which could result in a segmentation faullt.

# [1.2.5] - 2021-01-20

## Context
- Updating to latest LLMV compiler on Mac now causes build of ZLIB to fail due to the 'LSEEK' function. Adding the header "#include <unistd.h>" to the file gzguts.h solved this problem.
- On Mac, ZLIB was giving a compiler warning related to shifting a negative signed value. This was corrected by changing line 1507 of inflate.c to -(1L<<16)

## Visualizer
- setColorbarFontsize() function was not properly setting the font size.
- The problem with the image only filling 1/4 of the window on Mac Retina displays has finally been fixed!
- New way of throwing errors for improved debugging was added to Visualizer.

## Canopy Generator
- The user can now set the leaf angle distribution for homogeneous and spherical crowns canopies.
- There was an error in the homogeneous and spherical crowns canopies where the specified leaf area index or leaf area density would not be correct if leaves had a transparency mask.

## Radiation
- Function created to add sun sphere radiation source with default sun direction (see addSunSphereRadiationSource( void ) ).

# [1.2.6] - 2021-03-02

## Context
- Improvements made to writeOBJ() function for cases with multiple materials (acknowledgement to Clark Zha)
- Capability to set/get compound object data. This allows data to be associated with a given compound object, similar to how primitive data is associated with a given primitive.
- Added selfTest() routines for testing compound object data functions.

## Solar Position Model
- Functions were added to calculate just the PAR or NIR component of incoming solar radiation (see getSolarFluxPAR() and getSolarFluxNIR()).

## Visualizer
- The default camera view calculation was improved to make a better guess of where to put the camera.
- An error was corrected that would cause the camera settings to be overwritten if buildContextGeometry() was called after setCameraPosition().

## Canopy Generator
- Added initial draft of strawberry model and documentation.

*Radiation, LiDAR, AerialLiDAR, Energy Balance, Voxel Intersection*
- Architecture-specific CUDA compiler flags were removed from CMakeLists.txt file, which could cause build failure depending on architecture.
- Upgraded OptiX to version 5.1.0.

## LiDAR
- Overloaded version of loadXML() function added that can allow for loading details of the grid only.

# [1.2.7] - 2021-04-09

+++++ PLUG-IN RELEASE ++++++
- Boundary-Layer Conductance Model: The Boundary-Layer Conductance plug-in provides several models to calculate the primitive boundary-layer conductance. Currently, 4 models are available, and more will be added periodically.
- Added selftest in samples directory.
- Added selftest to utilities/run_samples.sh script.

## Context
- Issues corrected with transformations and compound objects, where things like the whole-object size, nodes, etc. were not being transformed.
- Functions added to get the whole-object center position of compound objects Tiles, Spheres, Boxes, and Disks.

## Energy Balance
- Minor modification to avoid getting wind speed or object length if boundary-layer conductance is specified through primitive data.

# [1.2.8] - 2021-04-26

## Context
- Added texture mapping for spheres and sphere compound objects.
- Added "Cone" compound object and self-test.

# [1.2.9] - 2021-06-17

## Context
- Fix to cone objects to use more stable acos_safe() function.
- Error corrected with Tube Objects that caused translation to be applied twice.
- The warning was removed when calling Primitive::setColor() or Primitive::overrideTextureColor() for primitives part of an object. This was removed because a warning would be issued whenever changing the color or texture override from an object.

## Radiation
- An error was fixed that would apparently cause rays to intersect the primitive from which they were launched in some instances.
- There seemed to be a problem when rays would intersect the periodic boundaries 10 or more times, which would cause a large amount of energy to accumulate in the first primitive added to the Context. This may have been associated with an inherent recursion depth limit in OptiX. The maxumum ray recursion depth was limited to 9, which appeared to fix the problem.

## Energy Balance
- A variable associated with the net radiation was not explicitly initiallized to 0, which could cause anomalous behavior.

## Weber-Penn Tree
- Checks were added to throw an error if UUIDs are requested for a tree ID that does not exist.

## Visualizer
- addCoordinateAxes() function was added to label coordinate axes (credit to Eric Kent).
- Added capability to disable messaged to the command line (see disableMessages(), enableMessages())
- Added capability to color context primitives based on compound object data (see colorContextPrimitivesByObjectData())

## Canopy Generator
- Functions added to get UUIDs for all primitives of a given type (see getLeafUUIDs(), getTrunkUUIDs(), getBranchUUIDs(), getFruitUUIDs()).

## Stomatal Conductance
- There was a correction to the units of Em in the documentation, which should be mol/m^2/s.

# [1.2.10] - 2021-06-21

## Visualizer
- An error was corrected that caused shadows to not be rendered when using plotUpdate().

## Canopy Generator
- Walnut tree model added with documentation.

## Stomatal Conductance
- There was not actually an error with the units of Em - they were changed back as before.

# [1.2.11] - 2021-06-30

## Context
- Documentation added for Compound Objects

## Radiation
- Radiation model adapted to more efficiently handle sub-divided patches via Tile Objects. When tile objects are used to represent sub-divided patches, there is now a dramatic reduction in the amount of GPU memory used, and ray-tracing computations are more efficient.

# [1.2.12] - 2021-06-30

## Canopy Generator
- The file walnut.cpp was not added to the repository, so it has been missing for the past two versions.

## Radiation
- There was an error in the selfTest() function, causing the compile to fail.

# [1.2.13] - 2021-07-13

## Context
- There was an error in the vecmult function, causing incorrect values.
- There were errors with the getSize() functions for sphere, box, and disk compound objects that caused them to return incorrect values when rotation was applied.
*Credit to Eric Kent for implementing the updates below
- Order of (u,v) texture coordinates for patches changed to be consistent with documentation.
- Function added to copy all primitive data from one primitive to another (see copyPrimitiveData()).
- Changed the way solid fraction was calculated for triangles so that the order of vertices doesn't matter - otherwise solid fraction would get set to zero erroneously for some triangles because the vertices were not always specified in a counterclockwise direction.
- Function added to get the solid fraction of a texture masked triangle (see Triangle::getSolidFraction()).
- Overloaded function to crop the domain based on a subset of UUIDs (see cropDomain()).
- Operators added for to test whether vec2, vec3, and vec4 vectors are equal.

## Radiation
- Anisotropic diffuse radiation capability was added (see setDiffuseRadiationExtinctionCoeff()).

## Voxel Intersection
*Credit to Eric Kent for implementing these updates
- Numerous functions added to slice primitives into two sub-elements when they land on the edge of a voxel. This is useful for calculating primitive area within a voxel when the primitives intersect the walls of the voxel.

## Canopy Generator
*Credit to Eric Kent for implementing these updates
- Option added for homogeneous canopy to change behavior related to the buffer on the edge of the domain to ensure leaves are completely inside the specified canopy volume.

# [1.2.14] - 2021-07-21

## Radiation
- Correction in the normalization of anisotropic diffuse distribution. For a horizontal, unobstructed patch, it should always give a normalized diffuse flux of 1 regardless of the value of K.

## Canopy Generator
- The 'spherical crowns' canopy has been generalized to include ellipsoidal crowns with different principal radii.

## Visualizer
- There is apparently an issue with the printWindow() function on Linux that causes a segmentation fault. A work-around has been implemented to remove the window "decorations", which seems to fix the problem. A new overloaded constructor has been added to allow for disabling of the window decorations. When a fix for the issue is eventually found, this constructor will likely be removed.

# [1.2.15] - 2021-07-28
## Context
- Errror in copyPrimitive() function was fixed in which the primitive parent object ID, primitive texture color override, and object transformation matrix were not copied.
- Fixed issue with tubes, cones, and voxels in which the radii/size/center may have been incorrect if a rotation was applied.

## Visualizer
- There were some potential issues with aliasing_samples set to 0 or 1 on Linux systems that would still do multisampling. A fix for this was implemented.
- There was an error when using colorPrimitivesByObjectData() when the object data was not of type float.

## Voxel Intersection
- Bug fixes and corrections for cases where primitives lie on corners or exactly on the edges of voxels.
- Overloaded version of approxSame() function added for vec3's.
- Two new self-tests added.

# [1.2.16] - 2021-09-07

## Context
- Header file for pugi XML library (pugixml.hpp) was moved to global.h (was previously just included in Context.cpp).
- Functions added to read and convert a value from a pugi XML node (see, e.g., XMLloadfloat, XMLloadvec3, etc.)
- string2vec4 and string2int4 functions added in global.cpp

## Canopy Generator
- loadXML() function added to allow for setting of canopy geometry parameters from an XML file. Currently, this is only implemented for the homogeneous and VSP grapevine canopies.

## LiDAR
- calculateSyntheticGtheta() and calculateSyntheticLeafArea() functions now outputs the synthetic G(theta) values as a vector and writes primitive data rather than writing them to file.
- Function added to add a grid programmatically rather than just reading from an XML file (see addGrid()).
- Function added to add a wireframe of the grid to the visualizer (see addGridWireframeToVisualizer())
- Rays for synthetic full-waveform beams are now sampled according to a radial Gaussian distribution, rather than uniformly across the beam diameter. This is more consistent with the radial distribution of a real LiDAR beam.
- The syntheticScan() function no longer takes an XML file as an argument, but rather the parameters of the scan are now set using the normal loadXML() function.
- Option added to synthetic scans to record "miss" points.
- calculateLeafAreaGPU_synthetic() function added to allow for more in-depth analysis of synthetic LiDAR data processing.
- Hit points can be labeled according to primitive data 'object_label'
- exportPointCloud() function was upgraded to allow more control over the format using the <ASCII_format> tag in XML file.

# [1.2.17] - 2021-09-15

## Canopy Generator
- Visual improvements to VSP grapevine canopy.

## Visualizer
- If buildContextGeometry() is called and there is no geometry in the Context, it could mess with the camera view. This behavior was changed to simply skip the function call if there is no geometry to build.
- A version of plotUpdate() has been added to not open the graphics window.

# [1.2.18] - 2021-09-19

- The project creation script (utilities/create_project.sh), and default project CMakeLists.txt file (version 1.4), has been modified to make the executable name the same as the build target. This makes using Helios with IDEs easier.

## Context
- The functions getGlobalDataType() and getGlobalDataSize() were declared but never implemented.
- The functions string2int2, string2vec2, etc. have been updated to be more robust by using the stoi() and stof() functions to convert parsed strings into values.

## Canopy Generator
- Ability to generate the other canopy types from XML file has been implemented. Credit to Dario Guevara for finishing these.

## LiDAR
- Change made to synthetic waveform data generation to improve delineation of adjacent hit points.

# [1.2.19] - 2021-10-14

* All projects in the samples/ directory have been moved to Helios CMakeLists.txt version 1.4, which includes Windows support and avoids the *_exe build target issue in CLion.
* Working on Windows/PC support: Several changes have been made to support PC, which will be tested and eventually released in v1.3 when it appears stable.

## Context
- Many changes in the core/ files to clean up the code based on clang-tidy suggestions
- Removed strdup function in favor of strcpy function in global.cpp, because strdup could cause problems on some architectures
- Added define of M_PI in global.h in case it is not defined on some architectures

## LiDAR
- Checks were added to addScan() function to avoid specifying invalid scan angle ranges.

# [1.2.20] - 2021-10-19

* Further improvements for Windows support - all plug-ins should now work. Continued testing needed..
* Helios project CMakeLists.txt updated to version 1.5, which has updates for Windows support

## Context
- There was an error in which two of the arguments for one overloaded version of addTileObject() were reversed.

## Visualizer
- Stability improvements for cases in which Visualizer functions are called out of order. For example, previously calling colorContextPrimitivesByData() before buildContextGeometry() would likely result in a runtime error.
- Check added to colorbar initializer to ensure that the minimum colorbar value is less than the maximum value.

## Radiation
- 'to_be_scattered' variable was not initialized, which could cause an error on some platforms.
- Type of size argument to function rtGetBufferSize1D() changed to RTsize to avoid possible compile errors on some platforms.
- Definition of M_PI and uint added to RayTracing.cu.h file, which were not defined when using MSVC compiler.
- Explicit type conversion added to argument of pow() function in rayHit.cu, which caused and error when using MSVC compiler.
- Some unneeded OptiX .so library files were deleted for Linux to speed up Git code clones and checkouts.
- OptiX library files added for Windows.

## LiDAR
- Fix to selfTest() in which a variable was uninitialized, which could cause memory errors.

## Weber Penn Tree
- Fix that caused build failure on Windows

# [1.2.21] - 2021-11-1

* utilities/create_project.sh script changed to not put 'void' in main function argument, and to not include Context.h header if a plug-in is used.

## Context
- Build directory for samples/context_selftest was accidentally deleted from the git repository
- Timer was broken, and would not print the elapsed time properly.
- Added global function to apply Newton-Raphson method to find the zero of a function (see fzero).

## Photosynthesis
- Improved selfTest() to actually check against known result.
- Farquhar model now solved using fzero function (see above).

## Stomatal Conductance
- New stomatal conductance model has been added, but is still in testing phase and thus has not been added to the documentation.

## Canopy Generator
- Now uses fzero function (see above) for leaf angle distribution sampling.
- Added option to create primitive data labels for each element type (VSP canopy only for now).
- General cleaning up of the code based on clang-tidy suggestions.

## Visualizer
- Error corrected that could cause visualizer to fail if not coloring primitives by data.

# [1.2.22] - 2021-11-19

* Helper functions added to global.cpp to calculate median and standard deviation of float values in a vector.

* Standard project CMakeLists.txt updated to v1.6 to correct potential build issues on certain systems.

# [1.2.23] - 2021-11-28

*Documentation added for set-up on Windows systems.

## Context
- Include file in core/include/pugixml.h changed from <cstring> to <string>, which could cause build issues on some systems.
- Issues with inconsistent treatment of (u,v) texture coordinates corrected.
- Added "nullrotation" global variable to avoid having to type out make_SphericalCoord(0,0).

# [1.2.24] - 2021-12-14

## Context
- interp1 function added for linear interpolation of a vector of vec2's
- Explicit definition of 'blend' function was missing in global.h
- Compound Object functions getRGBColor() and getRGBAColor() were changed to getColorRGB() and getColorRGBA() to be consistent with how things are done for primitives.
- Compound Object functions hasTexture() and getTextureFile() added.
- cropDomain*() functions updated to also delete objects when cropping.
- loadXML() and writeXML() updated for compound objects.
- Handling of textures within the Context backend re-worked to avoid storing pointers to textures.
- CompoundObject::setPrimitiveUUIDs() function added to allow for updating of object child UUIDs.
- setSubdivisionCount() functions added for all compound object types to allow for updating of subdivisions.

## Canopy Generator
- Tomato and strawberry models updated to properly store element UUIDs.

## Visualizer
- Version of write_JPEG_file() function added to write JPEG image based on pixel data instead of visualizer window.

## Radiation
- "--use_fast_math" flag was removed in a previous version, but it has been re-added as this caused issues on some systems.

# [1.2.25] - 2021-12-15

## Context
- Added getter functions related to primitive texture properties: getPrimitiveTextureSize(), primitiveTextureHasTransparencyChannel(), getPrimitiveTextureTransparencyData()

## Radiation
- Fixed issues in v1.2.24 related to textures.
- Some updates made to avoid use of getPrimitivePointer().

## LiDAR
- Fixed issues in v1.2.24 related to textures.
- Some updates made to avoid use of getPrimitivePointer().

## Aerial LiDAR
- Fixed issues in v1.2.24 related to textures.
- Some updates made to avoid use of getPrimitivePointer().

# [1.2.26] - 2021-12-31

* Documentation "Dependent Software" updated for CLion 2021.3
* Utah teapot PLY model added *
* CLion Live Templates for Helios added (see utilities/CLion_Helios_settings.zip) *

## Context
- Previous implementation of loadXML for objects was not correct, and led to object primitives not being properly transformed.
- flatten() function added for 4D vectors.
- flatten() functions re-written to improve efficiency.
- Tile::setTileObjectAreaRatio() and Tile::getTileObjectAreaRatio() functions added (credit to Eric Kent).
- Tile::setTileObjectSubdivisionCount() function added to easily change the tile sub-patch resolution (credit to Eric Kent).
- Additional selfTest() added for Tile Objects (credit to Eric Kent).
- getObjectCount() function added.

## Visualizer
- Default light direction changed to give a reasonable value.

# [1.2.27] - 2022-01-07

## Context
- Functions to get pointers to primitives (getPrimitivePointer(), getPatchPointer(), getTrianglePointer(), getVoxelPointer()) have been deprecated and removed from the documentation, and will be removed in a future version. Replace primitive-specific functions with direct calls from the Context (e.g., instead of context.getPrimitivePointer(UUID)->getArea() use context.getPrimitiveArea(UUID)).
- std::cout support added for Helios vector types (credit to jannessm).
- Function writePrimitiveData() added to make writing of primitive data to ASCII text file easier.
- getPrimitiveSolidFraction() function added.
- isPrimitiveTextureOverridden() function added.
- selfTest() function was getting very long, so it was moved to its own file (selfTest.cpp).
- Correction was made to loadXML() to correct an error occurring when object primitives had multiple primitive data fields.

## Visualizer
- Minor edits to Visualizer based on clangtidy suggestions.
- Updates to remove usage of getPrimitivePointer(), getPatchPointer(), getTrianglePointer(), getVoxelPointer().

*LiDAR, Aerial LiDAR, Boundary-Layer Conductance, Energy Balance, Radiation, and Voxel Intersection*
- Updates to remove usage of getPrimitivePointer(), getPatchPointer(), getTrianglePointer(), getVoxelPointer().

# [1.2.28] - 2022-01-13

## Visualizer
- Error corrected that could result in GLFW_INVALID_OPERATION error on Windows when the visualizer window is closed.
- The default filename when printWindow() is called with no arguments is not valid on Windows machines.

## Energy Balance
- Documentation updated to better clarify the meaning of the "moisture conductance".

# [1.2.29] - 2022-01-30

## Context
- The normals were reversed for triangles at the top and bottom of spheres (sphere objects were already correct).
- Error was introduced in v1.2.27, which caused scaling of tile objects not to work.

## Canopy Generator
- Walnut, Tomato, and Strawberry models: if fruit_radius=0, don't add any fruit.
- Tomato textures were inadvertently removed from Git. They have been re-added.
- Definition of fruit_radius for walnut model was missing from documentation.

# [1.2.30] - 2022-03-06

## Context
- Correction made for Cones in which color was not properly being set.
- Correction made for Cones in which reading from XML would incorrectly assign a texture file of "none" if no texture file was specified.
- Tube Objects now store the RGB color values for each node, and is properly written and read to/from XML files.
- If a primitive belonging to a compound object is deleted, now the object is deleted but all other primitives will still remain but not belong to any object.
- Functions added to get compound object primitive UUIDs directly from a context function without using pointers (credit to Eric Kent).
- Function added to filter compound objects based on an object data threshold (credit to Eric Kent).

## Visualizer
- When rendering with the visualizer, the window will now remain hidden until after the geometry is built. This should eliminate the 'window is not responding' warning message on some systems.

## Radiation
- Correction made to CLion Live Template when adding longwave band.
- Maximum number of rays in a launch reduced to 1024^3, which is a new (but undocumented) requirement in OptiX 6+.
- OptiX 6.5 libraries for Windows added to Git.
- Option added to allow switching between OptiX 5.1 and 6.5 depending on available system hardware.

## Stomatal Conductance
- Error corrected that could cause a segmentation fault when setting model coefficients based on a subset of UUIDs.
- Input parameters specified through primitive data are now checked to make sure they have the correct datatype (usually 'float').

## Energy Balance
- Input parameters specified through primitive data are now checked to make sure they have the correct datatype (usually 'float').

## Photosynthesis
- Input parameters specified through primitive data are now checked to make sure they have the correct datatype (usually 'float').

# [1.2.31] - 2022-03-08

## LiDAR
- Error corrected causing build failure on Windows systems.

# [1.2.32] - 2022-03-29

## Context
- Added functions to rotate a primitive and compound object about an arbitrary line not necessarily passing through the origin (see Context::rotatePrimitive() functions).
- Work started to eliminate the need for "getObjectPointer()" function. The following functions have been added so far: translateObject(), rotateObject(), getObjectType().
- The getPrimitiveCenter() function was added, which will now return the centroid of all primitive types.
- Function added to easily print information about a primitive for debugging purposes (see printPrimitiveInfo() function).

## Visualizer
- Visualizer now checks primitive data for infinity or NaN, which previously would cause an ambiguous error.

## Voxel Intersection
- Miscellaneous edits to slicePrimitive() function to improve stability and handle edge cases.

## LiDAR
- Certain architectures could cause an error related to the size of the GPU kernel launch for full-waveform simulations. This was fixed by reducing the launch dimension.

## Aerial LiDAR
- Certain architectures could cause an error related to the size of the GPU kernel launch for full-waveform simulations. This was fixed by reducing the launch dimension.

# [1.2.33] - 2022-04-11

## Context
- There was an error in OBJ texture writing that caused texture coordinates to be flipped about y-axis.
- There was an error in OBJ texture writing that gave invalid texture coordinates for patches.
- If a primitive that belongs to a compound object is deleted (via Context::deletePrimitive()), it will delete the primitive and remove its UUID from the list stored with the object. A function was added Context::areObjectPrimitivesComplete() that allows for querying of whether some primitives have been deleted from the object.

## Radiation
- Tile objects are now checked to see if any child primitives have been deleted. If so, it will treat it as individual primitives and ignore the fact that it is an object.

## Visualizer
- If a UUID for a primitive that does not exist was passed to the Visualizer, it would previously cause a segmentation fault.

# [1.2.34] - 2022-04-26

## Context
- Added warning messages to setTileObjectCount() and getTileObjectAreaRatio() related to incomplete objects, minor related documentation changes.
- Changed Tile::getVerticies so that a missing corner primitive doesn't break it.
- Added many functions to set/get object information without using pointers.

## Visualizer
- added Visualizer::colorContextPrimitivesRandomly to easily visualize object sub-primitives.

# [1.2.35] - 2022-04-29

## Context
- Added functions to calculate the axis-aligned bounding box for a primitive or group of primitives (see getPrimitiveBoundingBox()).
- Added functions to calculate the axis-aligned bounding box for a compound object or group of compound objects (see getObjectBoundingBox()).

# [1.2.36] - 2022-05-13

## Context
- Added numerous checks for texture image files to ensure they are either PNG or JPEG files.
- loadPLY() and loadOBJ() functions check to be sure the input files have the correct extensions.
- When loading compound objects using loadXML(), it will first look to see if there are any primitives in the file assigned to that object, and if so it will build the object based on those primitives. If no primitives are assigned to the object, it will build a fresh/complete object from scratch.
- Corrected an issue with cropDomain() functions to be consistent with new treatment of incomplete compound objects (i.e., primitives can be deleted from within objects).

## Visualizer
- Functions for reading PNG and JPEG files now check to ensure they have the correct file extension.

## Radiation
- Users can now set primitive data "twosided_flag" equal to 2 to make the primitive a one-sided "sensor" that does not attenuate or emit any radiation.

# [1.2.37] - 2022-05-27

## Context
- Building was failing on M1 Macs. A fix was implemented into the libpng CMakeLists.txt file.

## Photosynthesis
- Users can now optionally output 'Gamma' (CO2 compensation point) as primitive data.

## Canopy Generator
- First version of a sorghum model is now included. (credit to Ismael Mayanja)

# [1.2.38] - 2022-06-20

## Canopy Generator
- Improvement to sorghum panicle shape
- Bug fixed in sorghum model that caused jagged leaf edge when subpatch resolution was changed

# [1.2.39] - 2022-07-19

++++NEW PLUGIN++++
- First version of synthetic data annotation plug-in is now included.

## Canopy Generator
- Tomato canopy was missing from loadXML() function.

# [1.2.40] - 2022-07-25

## Context
- Error fixed that caused incorrect transparent texture mapping in the y-direction, which could result in incorrect areas when there are sub-patches/triangles or custom (u,v) coordinates.
- Changes related to deleting objects when all constituent primitives have been deleted in deletePrimitive, deleteObject, setTileObjectSubdivisionsCount, loadXML, and setPrimitiveParentObjectID.
- Added related self tests.

## Canopy Generator
- Fixed formatting error for documentation.
- Edited default parameter values to match documentation.

# [1.2.41] - 2022-08-05

*Copyright updated to (C) 2016-2022 for all files
*Files in doc/ needed to build documentation have been added to the Git repository.

## Stomatal Conductance
- Default BMF model coefficients (almond) have been changed.
- Added Ball, Woodrow, Berry, Ball-Berry-Leuning, and Medlyn optimality-based stomatal conductance models.
- The vapor pressure deficit used in now calculated based on the leaf surface vapor pressure rather than the ambient vapor pressure.
- Self-test case has been added for all models.

## Photosynthesis
- Changed default parameters to match parameter set given in stomatalconductance plug-in.
- Added model parameters to documentation for various tree species.
- Added optional output primitive data for Gamma_CO2, which is the CO2 compensation point including dark respiration.
- Model coefficients can be specified differently for different primitives based on UUIDs.

## Weber-Penn Tree
- Wood textures are now packaged with the plug-in, so that references to textures from the visualizer plug-in could be removed (which caused failure if visualizer plug-in was not also built).

# [1.2.42] - 2022-09-02

## Context
- Minor updates to helios_vector_types.h
- Bug fixed that would cause segmentation fault if texture (u,v) coordinates are out of bounds. These will now be truncated to [0,1]
- Change made to Context::loadOBJ() such that if a 'height' value of 0 is given, the model will not be scaled
- Overloaded versions of Context::loadOBJ() added to only write geometry for a subset of UUIDs, and to write a .dat file containing primitive data for Unity visualization tool
- Feature added to Context::writePrimitiveData() to allow for writing primitive UUID without explicitly creating primitive data for UUID

## LiDAR
- Bug fixed that caused triangle textures to be flipped about the y-axis in synthetic scans

# [1.2.43] - 2022-09-07

## Context
- Overloaded loadOBJ() function added to scale the model in the x,y,z directions.
- Messages can now be disabled when loading OBJ files.
- Primitive solid area fraction is written and read from XML so that it doesn't need to be re-calculated when loading.
- Change made to Context::loadPLY() such that if a 'height' value of 0 is given, the model will not be scaled.
- An error was introduced in v1.2.41 associated with texture indexing/truncating.
- The member variable 'solid_fraction' was erroneously declared for Triangles, which was redundant and conflicting with the definition for Primitive.
- Added additional self-test case to test solid fraction for triangles.

* Energy Balance *
- Check added to only add a radiation band if it has not already been added previously to avoid duplicates.

* Canopy Generator *
- Functions to add individual plants now return the plant ID.
- Functions added to clean up deleted UUIDs.
- Issues were fixed with get[*]UUIDs() functions that would return UUIDs for deleted primitives in some cases.

# [1.2.44] 2022-09-16 ** revised 2022-09-23

**This commit originally pushed 2022-09-16 introduced an error that would cause build of the Context to fail on many systems. This commit was reverted and re-committed with a correction on 2022-09-23 *

* Context *
- Added functions to read and write PNG images from the Context (see readPNG() and writePNG()).
- Added equality operators for int2, int3, int4, RGBcolor, and RGBAcolor
- Separate member functions added to compute solid fraction for Patches and Triangles.
- Update to loadXML() to make it output all added UUIDs for both individual primitives and primitives belonging to compound objects.
- If patches have "default" texture u,v coordinates that encompass the entire texture, the solid fraction will be looked up from the texture rather than re-calculating it.
- writeOBJ() function has been substantially re-written:
    + More efficient arrangement of materials to eliminate duplicate materials and frequent switching between materials.
    + Texture image files are all copied to a new directory and referenced in the .mtl file to make things more portable.
    + Filename function argument changed from const char* to strings.
    + 'map_d' property is only written for .png textures with transparency, otherwise this apparently causes a translucent appearance for solid textures.
- Added functions to read and write JPEG images (see readJPEG(), writeJPEG()).
- Added Primitive::setTextureUV() function.

* Radiation *
- CMakeLists.txt file improved so that the *.ptx files will only be re-built if the source *.cu file is changed.

# [1.2.45] 2022-09-26

* Context *
- If any primitives have (u,v) coordinates, writeOBJ() will now write 'dummy' (u,v)'s for primitives that are not UV mapped. This is apparently needed to correctly read the .obj file into Blender.
- writeOBJ() function was not working properly based on updates in v.1.2.44.
- Deblank function added for strings (previously was only for character arrays).

* Canopy Generator *
- Draft model for common bean added. It is not documented yet, and bean pods are not yet supported.

# [1.2.46] 2022-10-07

* Context *
- All Compound Objects besides Tiles were not being properly read from XML files.
- There was an error in the Context::writeJPEG() file, which caused garbage pixel values to be written.
- Added explicit instantiation of template functions in global.cpp. Not having these could cause build errors with high compiler optimization enabled.

* Canopy Generator *
- Updates to bean model to speed up generation time.

# [1.2.47] 2022-10-21

* Context *
- Error fixed in loadXML() where patches that are members of a tile objects were added twice to the output UUID vector in some cases.
- Error fixed in loadXML() where the elevation angle of loaded tile objects was not correct.
- Error fixed in writeXML() where primitives that were members of "complete" tile objects were getting written to the XML file, which could cause the member patches to be created twice in loadXML().

* LiDAR, Aerial LiDAR, Energy Balance, Voxel Intersection, Radiation *
- Added explicit setting of CUDA compiler optimization flag for Debug/Release build types.

* Radiation *
- Added check to addBand() function to prevent users from adding duplicate bands.
- For each ray launch 'batch', the previous batch output message will be cleared so it will not continue printing a new line for each batch launch.

* Energy Balance *
- If no object length is specified via primitive data, the model will first check to see if the primitive is a member of an object, in which case it will use the object area to calculate the characteristic length for boundary-layer conductance calculations.

* Boundary-Layer Conductance *
- If no object length is specified via primitive data, the model will first check to see if the primitive is a member of an object, in which case it will use the object area to calculate the characteristic length for boundary-layer conductance calculations.

* LiDAR *
- Revised LiDAR self-test to use geometries read from file to achieve more consistent results.
- laszip and libLAS libraries are included in the repository, but are not yet used.

* Voxel Intersection *
- Minor changes to avoid using primitive pointers directly.
- Error fixed where sliced triangles that were previously members of a tile would retain the object ID of the previous tile (created triangles should not be a member of an object).

# [1.2.48] 2022-10-28

## LiDAR
- Building with liblas and laszip libraries disabled for now, as this seemed to cause build errors on some systems.

# [1.2.49] 2022-11-07

## Context
- Error corrected in Context::loadOBJ() that would cause geometry to not be loaded of (u,v) coordinates are not supplied.
- Error corrected in helios::getFileExtension() that would cause a crash if the file has no extension.
- Added helios::importVectorFromFile() function to read values from a text file into a 1D float vector.
- Added Context::calculatePrimitiveDataMean() functions to calculate average of primitive data values.
- Added Context::calculatePrimitiveDataAreaWeightedMean() functions to calculate area-weighted average of primitive data values.
- Added Context::calculatePrimitiveDataSum() functions to calculate sum of primitive data values.
- Changed all file extension checking to use new, more robust helios::getFileExtension() function.
- self-tests added for primitive data calculation functions and file extension parsing functions.
- Added overloaded version of Context::writeXML() that can export a subset of primitives based on a UUID vector.

## Energy Balance
- Added checks when loading primitive data to issue a warning if values are out of the expected range of possibilities.
- Fixed error in conversion factor between conductance of heat to vapor in air - changed from 0.97 to 1.08.
- Capability added to change the number of convective surfaces based on the value of primitive data ``twosided_flag".

## Canopy Generator
- Issue fixed in bean.cpp that could cause node positions of NaN if leaflet_length was too short.

* LiDAR *
- Fixed bug in distanceFilter, reflectanceFilter, and scalarFilter, which caused them not to apply filtering.
- Added xyzFilter() function to filter out points outside of a specified bounding.

# [1.2.50] 2022-11-10

## Context
- Added check to primitive data calculation functions to make sure that primitives referenced in UUID vector exist.
- Added function to sum surface area of a set of UUIDs (see sumPrimitiveSurfaceArea()).
- Fixed issue with Context::writeOBJ() where it was not writing primitive data values to the optional *.dat file in the correct order.
- Context::writeOBJ() no longer writes "map_Ks" texture files.

## Radiation
- Print messages for iterative launches were not clearing the previous line. More spaces were added.

## Energy Balance
- Added option to set the number of transpiring primitive surfaces to account for the fact that you may have a two-sided primitive but with only one side transpiring.

## LiDAR
- Fixed error introduced in last commit in LiDAR.h causing build failure
- Modified xyzFilter() function to add option for filtering out points within the provided box
- Added overloaded xyzFilter() function that defaults to filtering out points outside of the provided box
- Fixed bug impacting the azimuthal angle range in syntheticScan() function

# [1.2.51] 2022-11-17

## PLY Models
- All chess piece models except King.ply deleted from Git repo. These were also removed from the samples/visualizer main.cpp file.

## Tutorials
- Tutorials #8, #10, and #11 added in samples/ folder and in documentation.

## Context
- calculatePrimitiveDataAreaWeightedSum() functions added.
- Modified Context::getPrimitiveBoundingBox() to be more robust when you only have 1 primitive in the UUID vector.
- Gave calls to 'fread' function in global.cpp a result assignment to get rid of compiler warnings.

## Energy Balance
- Error introduced in last version causing src/EnergyBalance.cu not to build.

*Aerial LiDAR, LiDAR, Energy Balance, Radiation, Voxel Intersection*
- Option in CMakeList.txt CUDA-based plug-ins added to build for GPU architectures of 3.5 or greater (--gpu-architecture=compute_35 -Wno-deprecated-gpu-targets).

## Radiation
- Fixed an issue with ray generation that caused self-test #8b to fail. This was causing radiative fluxes to be accumulated only in the first sub-patch of a tile object.

# [1.2.52] 2022-12-12

* General *
- Tutorial 2 and 5 were modified to be able to run as part of the standard set of tests in utilities/run_samples.sh
- The standard project CMakeLists.txt file has been changed to remove all non-user-defined code. Instead, this was moved to a separate file core/CMake_project.txt, which is reference from the project CMakeLists.txt file. Old project CMakeLists.txt files should still work.
- Sample project CMakeList.txt files updated to reflect new format.
- utilities/run_samples.sh script updated to be able to only run test cases that do not require a GPU

## Context
- Modifications made to core/CMakeLists.txt and core/lib/libpng/CMakeLists.txt to clean up dependent library build structure and avoid potential circular dependencies.
- Updated pugixml library to use snprintf instead of deprecated sprintf.
- Functions added to global.cpp to perform checking of string to value conversions (see parse_float(), parse_double(), parse_int(), parse_uint()).
- Optional arguments added to loadPLY() function to specify up-direction and to silence output messages.
- Overall cleaning up of loadPLY() code, including checks of file parsing.
- loadOBJ() now supports object groups in .obj files.
- Self-test for loadOBJ() added. This included adding a sample model file in core/lib/models. The core/CMakeLists.txt file was modified to copy this model into the project build directory.
- Functions added to filter primitives based on their primitive data values (see filterPrimitivesByData() functions).
- Added != operators for RGBcolor and RGBAcolor.

## Aerial LiDAR
- Error corrected in gridindex2ijk() function that would cause an error if the grid size in the x and y directions are not the same.

## LiDAR
- When a synthetic scan is being run, it now checks for primitive data "reflectivity_lidar" for calculation of the reflected intensity.
- Reading of point cloud input data files was moved to a separate function loadASCIIFile().
- Some overall cleaning up of LiDAR code.

## Visualizer
- Updated to use snprintf instead of deprecated sprintf.

## Radiation
- Overloaded version of setSourcePosition() added to accept a spherical coordinate instead of Cartesian vec3.

# [1.2.53] 2022-12-22

## General
- Added GitHub Actions workflows to automatically run utilities/run_samples.sh on push or pull requests.
- Fixed issue on Windows in core/lib/libpng/CMakeLists.txt file where it could not find the zlib dll.
- Fixed issue on Windows in project CMakeLists.txt files that would cause a build error because 'cmake_minimum_required' was not specified directly.
- Modified core/CMake_project.txt file to be sure that it puts .dll's in the lib directory and not in a sub-directory based on the build type
- utilities/run_samples.sh script modified such that it can take an argument '--visbuildonly' to skip runs for visualizer-based projects, and so that this script now works on Windows

## Context
- All Context functions/methods related to file I/O were moved from Context.cpp to a separate Context_fileIO.cpp file.
- String conversion when parsing text files has been improved to check for errors in the conversion using 'parse_[*]()' functions.

## LiDAR
- Documentation edited to correct minor error related to azimuth rotation reference coordinate system.

## Visualizer
- Modified CMakeLists.txt to only copy glfw3.dll file if it exists (Windows only).

## Weber-Penn Tree
- selfTest() routine modified to not build all trees at the same time in order to save memory.

# [1.2.54] 2022-12-23

## General
- Added 'cmake_minimum_required' at the beginning of all project CMakeLists.txt files for all platforms, as well as 'project(helios)'.
- Suppressed many of the compiler warnings that occur on Windows.

## Context
- Fixed an error in Context::loadPLY() introduced in version 1.2.52 that caused PLY models to not be read correctly.

# [1.2.55] 2023-01-06

## General
- The CMakeLists.txt for tutorial1 was not updated to be consistent with all the rest.

## Context
- All declarations related to primitive, object, and global data were moved from core/src/Context.cpp to core/src/Context_data.cpp
- Parsing of XML files has been re-written to use functions from a new XMLparser class.
- Error fixed in "parse_[*]" functions that would cause an error if strings have whitespace.

## LiDAR
- LiDARcloud::addScan() now returns a uint corresponding to the scanID.
- Number of hit points in output message after synthetic scan did not match the actual number of hit points generated.
- Scan azimuth range is no longer truncated to be less than 2*pi (it will only issue a warning if >4*pi).

## Canopy Generator
- All primitives in the grapevine and bean models now belong to compound objects.

# [1.2.56] 2023-01-18

## Context
- There were some minor memory leaks associated with the PNG texture/image reading/writing. These were generally very small leaks.

## Visualizer
- There were some minor memory leaks associated with the PNG texture/image reading/writing. These were generally very small leaks.

## Synthetic Annotation
- Visualizer geometry was being updated for every view when generating labels, which resulted in slow performance and building of large amounts of memory.

# [1.2.57] 2023-02-23

## Context
- Some temporary fixes applied related to reading/writing of Tile Objects to XML. There are still issues when the tiles have been rotated about their normal axis that need to be resolved.

## Synthetic Annotation
- Error fixed that could cause a character buffer overflow for very long file paths. Strings are now used instead of character arrays, which allows for arbitrarily long paths.

## Energy Balance
- Documentation updated to clarify that primitive outgoing emission depends on whether the primitive is 'two-sided'

*Boundary-Layer Conductance*
- Minor errors corrected in documentation

# [1.2.58] 2023-03-14

## Context
- Context::copyObject() function did not copy object data. Function Context::copyObjectData() was added and is now called.
- When writing primitive data files from Context::writeOBJ() function, the .dat file would always be written to the build directory rather than to the same folder where the .obj and .mtl files should be written.
- There were cases in which objects read from file could end up with child primitives that had the wrong object ID.

## Canopy Generator
- Changing bean model to use Polymesh objects for leaves broke the model. Reverting back to triangle primitives for now.

*VoxelIntersection*
- Credit to Eric Kent for these updates
- redid UV interpolation code in slicePrimitive to fix bug occurring when x and y coordinates of a vertex were the same and pulled out into a separate function (interpolate_texture_UV_to_slice_point)
- changed some naming conventions for clarity inside slicePrimitive
- changed order of some if statements and removed some repetitive ones inside slicePrimitive
- fixed issue in slicePrimitive when slice points fell on primitive vertices for triangles. Note that it hasn't been fixed for patches yet.
- added check in slicePrimitive to make sure the sliced primitive normals match the original normal
- added check in slicePrimitive to make sure the sum of the sliced primitive surface areas match the original primitive surface area

## LiDAR
- Credit to Eric Kent for these updates
- added non-idealized intensity-based weighting to calculateLeafAreaGPU_synthetic
- added calculateLeafAreaGPU_equal_weighting that pulls out only the equal weighting method from the _synthetic version for use with actual LiDAR data
- fixed bug in calculateLeafAreaGPU_synthetic related to indexing when the first beam of a scan was a "miss"
- calculateLeafAreaGPU_synthetic can now print LAD and G in addition to P to console
- gapfillMisses now adds hitpoints directly to the point cloud and adds timestamp and target_index as data
- an overloaded version of gapfillMisses allows for extrapolation of only beams that intersect the axis-aligned bounding box of the voxel grid to save time and reduce unwanted gap filling

## Visualizer
- Bug fixed in Visualizer::clearGeometry() function, which was not completely clearing all geometric information.

# [1.2.59] 2023-05-12

## Context
- Some efficiency improvements in Helios vector types.
- writeOBJ() updated to write optional .dat files to same directory that the .obj and .mtl files will be written.
- scaleObject() function was added to allow for scaling of compound objects after they have been created. As a result, scaling of objects is now done the same regardless of the object type.
- 'nullorigin' global variable created to avoid having to type out make_vec3(0,0,0).

*Solar Position*
- Error corrected in calculation of diffuse fraction using Gueymard model (credit to Alejandra for finding this)
- Added function setSunDirection() to manually set the sun direction.

## Photosynthesis
- Error in documentation corrected that had column labels for photosynthesis model parameters swapped for theta and Rd.
- Optional output data for limitation state was outputting the wrong value.

# [1.2.60] 2023-06-03

* Copyrights updated

## Context
- Compound objects no longer manage color in order to allow object primitives to have different colors.

## LiDAR
- Error corrected when writing synthetic point clouds that would incorrectly add a line carriage return after a point that was un-labeled.
- Numerical precision issues corrected in synthetic scans where hit points lying exactly on the seam where two primitives meet could randomly result in a miss.
- When calling LiDARcloud::syntheticScan() with record_misses=true now generates miss points for beams that do not intersect the voxel grid.
- The 'timestamp' field for synthetic scans was revised to be globally consistent for all pulses including misses.
- The 'intensity' field for synthetic scans was revised to have a value of 0 for misses (was previously an arbitrarily large number).

# [1.2.61] 2023-06-19

## Context
- Fixed Context::writeXML( const char*, std::vector<uint> ) method, which would write out primitives that were part of compound objects regardless of whether their UUID was specified in the input vector argument.
- Added Context::writeXML_byobjects() method to write geometry to XML based on a vector of compound object IDs.
- Made temporary fix to Context::writeXML() file to correct an issue where primitives part of a Tile were not loaded correctly if the Tile is complete. The fix was to write all primitives part of an object to XML regardless of whether the object is complete.

## LiDAR
- Added LiDARcloud::exportPointCloudPTX() method to export to the PTX file format.

## Canopy Generator
- Fixed error caused by canopy_rotation is not equal to zero in the SplitGrapevine canopy.
- Added a catch for when leaf_width = 0 in grapevine canopies, which would cause an infinite loop.

# [1.2.62] 2023-07-13

## Context
- Option added for Context::addDisk() and Context::addDiskObject() to specify disk subdivisions in the radial direction.

## Photosynthesis
- Boundary-layer conductance was not properly incorporated into CO2 diffusion equation.
- Default moisture conductance in the code did not match documentation.
- The photosynthesis model will now check for the primitive data "boundarylayer_conductance_out" to use for the boundary-layer conductance value. If the energy balance plug-in is being used to calculate the boundary-layer conductance, this optional output primitive data should be enabled so that it can be used by other plug-ins.

## Stomatal Conductance
- Factor converting boundary-layer conductance to heat to moisture changed from 0.97 to 1.08 to be consistent with energy balance plug-in.
- The stomatal conductance model will now check for the primitive data "boundarylayer_conductance_out" to use for the boundary-layer conductance value. If the energy balance plug-in is being used to calculate the boundary-layer conductance, this optional output primitive data should be enabled so that it can be used by other plug-ins.

## LiDAR
- Added overloaded version of calculateLeafAreaGPU_equal_weighting() that accepts a vector of G values. A separate LAD inversion will be performed for each G value, setting G in all voxels to this value.
- Modified the overloaded version of gapfillMisses to accept an additional argument, add_flags, which adds a flag as hitpoint data (0: original point, 1: gapfilled, 2: extrapolated at downward edge, 3: extrapolated at upward edge) - added checks to catch issue when one sweep of the LiDAR had zero points - added a check so that if gapfill_grid_only is true and the range is outside of the one provided in the XML file, the XML file range will be used - printed a little more info to screen about the extrapolation ranges and the input arguments.
- Added in overloaded version of triangulateHitPoints() that applies a scalar filter before triangulating.
- Implemented cropBeamToGridAngleRange() to reduce memory needed to process point clouds.

## Visualizer
- Added colorContextObjectsRandomly().

# [1.2.63] 2023-07-15

## Context
- Primitive transformation functions when called with a vector of UUIDs will now calculate the transformation matrix once, and apply it to all primitives for improved efficiency.
- Added primitive data processing methods Context::scalePrimitiveData(), Context::aggregatePrimitiveDataSum(), and Context::aggregatePrimitiveDataProduct().

## Stomatal Conductance
- printDefaultValueReport() method was added to print a summary of default input value usage to help with debugging.
- Very minor change to not get primitive data for photosynthesis or Gamma if not using a photosynthesis-based stomatal conductance model.
- Made selfTest() static so it can be called without declaring the StomatalConductanceModel class.

## Photosynthesis
- printDefaultValueReport() method was added to print a summary of default input value usage to help with debugging.
- Made selfTest() static so it can be called without declaring the PhotosynthesisModel class.

## Energy Balance
- There was an error in the documentation which said that twosided_flag=1 was for one-sided heat transfer and twosided_flag=2 for two-sided heat transfer. It should be twosided_flag=0 for one-sided heat transfer and twosided_flag=1 for two-sided heat transfer.
- The 'surface_humidity' input listed in the documentation was not actually implemented.
- A warning message is now issued if the boundary-layer conductance is calculated from the length/area of a primitive instead of an object.
- Made sure warning messages are disabled when disableMessages() is called.
- printDefaultValueReport() method was added to print a summary of default input value usage to help with debugging.
- Made selfTest() static so it can be called without declaring the EnergyBalanceModel class.

# [1.2.64] 2023-07-20

## Context
- Increment operator added for Helios vector types int2, int3, int4, vec2, vec3, vec4, and general improvements to helios_vector_types.h
- Overloaded versions of Context::overridePrimitiveTextureColor() and Context::usePrimitiveTextureColor() added to accept a vector of primitive UUIDs.
- Overloaded versions of Context::overrideObjectTextureColor() and Context::useObjectTextureColor() added to accept a vector of object IDs.

# [1.2.65] 2023-07-25

## Context
- Added Context::loadTabularTimeseriesData() method to directly load a tabular text file containing weather data.
- Self-test added for Context::loadTabularTimeseriesData().
- Output streams to print a 'Date' or 'Time' vector modified to print leading zero for month, day, minute, and second.
- Added Date::incrementDay() method to date vector to increment date by one day.
- Added helios::separate_string_by_delimiter() function to separate a delimited string into a vector of strings.

## Radiation
- Upgrading to CUDA 12.0+ caused build errors. Users with GPU compute capability of 3.5 now need to enable the OPTIX_VERSION_LEGACY build option.

# [1.2.66] 2023-08-03

## Context
- Method added to get the random number generator from the Context, such that it can be used by other plug-ins. This keeps the seed consistent across plugins.

## Visualizer
- Issue fixed that caused glfw3.dll to not be properly copied after build on Windows platforms.

# [1.2.67] 2023-08-08

* Changed core/CMake_project.txt to set CMAKE_RUNTIME_OUTPUT_DIRECTORY and CMAKE_LIBRARY_OUTPUT_DIRECTORY variables, which seems to resolve some intermittent issues with not finding .dll locations on Windows.
* `<iomanip>` header added to helios_vector_types.h, as it can be needed in some cases.

## Aerial LiDAR
- Updated CMakeLists.txt to deprecate compute capability 3.5.

## Energy Balance
- Updated CMakeLists.txt to deprecate compute capability 3.5.
- Changed selfTest() back to non-static because it could cause issues running it outside of the class.

## LiDAR
- Updated CMakeLists.txt to deprecate compute capability 3.5.

## Voxel Intersection
- Updated CMakeLists.txt to deprecate compute capability 3.5.

## Photosynthesis
- Changed selfTest() back to non-static because it could cause issues running it outside of the class.

## Stomatal Conductance
- Changed selfTest() back to non-static because it could cause issues running it outside of the class.

# [1.2.68] 2023-08-17

* Added new error handling function 'helios_runtime_error', which allows for greater control over error messages. If running in debug mode, it will output the error message to stderr as well as throwing an exception.

## Context
- Incorporated new error handling function 'helios_runtime_error' into all Context-related files.
- Polymesh objects were not being properly written to XML files (Context::writeXML()). The \<polymesh\> tag was not being closed properly.
- The overloaded += operator for helios vector types were changed to return a reference to the object instead of void, which allows for chaining of operations.
- If a material was not specified in an OBJ file (Context::loadOBJ()), the default material was not being applied to the object and an error would be thrown.

## LiDAR
- References to laszip in CMakeLists.txt completely commented out for now, as this was causing build errors on some Windows systems.
- Check added to LiDARcloud::loadASCIIfile() to automatically compute hit direction if it is not specified in the ASCII file. Without this, an assertion error is thrown since v1.2.60.

*Boundary-Layer Conductance*
- Fixed UUID indexing error.

# [1.2.69] 2023-08-28

* utilities/create_project.sh will now figure out the current directory name and use it as the executable name if run inside the project directory.
* Memory leak checking option added to utilities/run_samples.sh for Linux and MacOS.
* Added documentation about setting Timeout Delay (TDR) on Windows systems to avoid GPU timeout errors.

## Context
- Polymeshes were not implemented in Context::loadXML().
- Removed 'remapped_ObjIDs' variable from Context::loadXML() function, as this was not used.
- Memory leak when creating Tile objects fixed.
- Changed push_back() to emplace_back() in Context::getTileObjectAreaRatio() and Context::setTileObjectSubdivisionCount() methods.
- Context copy constructor and assignment operator deleted to prevent copying of the Context.

## LiDAR
- Point cloud rotations specified in input XML or ScanMetadata was not working correctly.

## Visualizer
- Visualizer::setColormap() with custom colormaps was not working correctly.
- Visualizer now uses helios_runtime_error() for error handling.
- Cleaning of the Visualizer code throughout.

## Energy Balance
- Energy balance now uses helios_runtime_error() for error handling.

# [1.2.70] 2023-09-05

*Changed core/CMake_project.txt to set HELIOS_DEBUG variable instead of _DEBUG, which could cause build conflicts on Windows.
*Set CMake policy CMP0079 to "NEW" in core/CMake_project.txt to allow for target_link_libraries() to be used with imported targets.
*Add target_link_libraries() between all plug-ins and helios target in core/CMake_project.txt to avoid build errors on some systems.

## Context
- Fixed error in Context::loadTabularTimeseriesData() that caused incorrect reading of 'minute' values.

## Aerial LiDAR
- Fixed and error with implementation of helios_runtime_error() that caused a compile error.

# [1.3.0] 2023-09-06

---- Additions since v1.2.0 ----

## Radiation Model
The radiation model has been re-designed, with the following primary additions:
- Multiple radiation bands are run within a single ray trace by calling RadiationModel::runBand() with a vector of band labels.
- The model can simulate multiple camera sensors for any radiative band, as well as depth and longwave emission (thermal).
- Support was added for easily calculating source fluxes and surface radiative properties based on spectral data.

## Leaf Optics
- Leaf optics plug-in added to calculate leaf optical properties based on the PROSPECT model.

*Boundary-Layer Conductance*
- Separate plug-in was added to calculate boundary-layer conductance based on several models.

## LiDAR
- Support for generating synthetic full-waveform data added.
- Support for auto-labeling synthetic point clouds added.

## Synthetic Annotation
- Synthetic annotation plug-in added to create annotations for images generated by the Visualizer.

# [1.3.1] 2023-09-14

* Changed copyright header information for most plug-ins to remove authorship information.

## Synthetic Annotation
- Include paths in CMakeLists.txt changed to relative paths to avoid build errors on some systems.
- Error fixed in SyntheticAnnotation::setCameraPosition() where arguments were not being correctly assigned.
- Converted to using helios_runtime_error() for error handling.

* LiDAR *
- Converted to using helios_runtime_error() for error handling.
- Removed 'Nhits' field from ScanMetadata struct, as this was not used.
- LiDARcloud::syntheticScan: Changed the method of grouping rays into hit points so that it is based on detecting peaks in the histogram of intensity weighted distance, closer to what a real LiDAR scanner would do. A pulse distance threshold is then used to merge hit points that are close together.
- LiDARcloud::syntheticScan: scanner range was set to 1000 m instead of 1e6. Miss points are now assigned a value of 1001 m.
- LiDARcloud::calculateLeafAreaGPU_synthetic: removed unused components, including weighting by sine of zenith angle in transmission probability calculation
- LiDARcloud::calculateLeafAreaGPU_synthetic: fixed issue with intensity weighting that caused total miss beams to not be accounted for in transmission estimates. This issue was was introduced in version # [1.2.60] when intensity of miss points was switched to zero in syntheticScan.
- Minor code cleaning.

* Radiation *
- Calibration "calibrated_CREE6500K_NikonD700_spectral_response_*" in camera_spectral_library.xml edited to be consistent with normalized spectrum for CREE LED light source.
- Added example to documentation of writing image pixel labels.

# [1.3.2] 2023-10-20

## Context
- Try making Context::selfTest() static (again) so it can be run without declaring the Context class.
- Fixed and issue in helios::parse_float() that could cause and out of bounds error if the parsed value requires more precision than can be stored in a float.

## LiDAR
- Added error checking to LiDARcloud::export[*] methods to check whether the output file was successfully created.
- Added method LiDARcloud::exportTriangleInclinationDistribution() to easily calculate and export triangulation angle distribution.
- If users specify a scan thetaMax that is out of range and thetaMax is truncated to pi, the thetaMin value will also be set to 0.
- ScanMetadata::direction2rc() function will truncate any points outside of the specified scan range (thetaMin - thetaMax; phiMin - phiMax).

# [1.3.3] 2023-10-24

## Context
- Simpler overloaded versions of Context::loadPLY() and Context::loadOBJ() added that only require a filename argument.

## Stomatal Conductance
- Any steady-state stomatal conductance model can now be run in dynamic mode based on stomatal time constants for opening and closing.

*Solar Position*
- Added method calibrateTurbidityFromTimeseries() to calibrate turbidity based on a timeseries of measured solar irradiance. For now, this method only uses the maximum solar flux in the dataset to calibrate the turbidity.
- Several methods were made const to have const-correctness.

# [1.3.4] 2023-11-10

## Context
- Context::getTimeriesLength() was giving a warning that there was no return value from non-void function. This was fixed by adding a return statement after the helios_runtime_error() call, although it is not technically needed.
- sum() function added to sum values in a vector.
- Added a check to Context::addPolymeshObject() to ensure that all primitives in input UUID vector exist.
- Added overloaded version of Context::doesPrimitiveExist() that accepts a vector of UUIDs to check all UUIDs in the vector at once.

## Canopy Generator
- Added canopy of conical crowns filled with homogeneous vegetation.

## Visualizer
- Only apply GLFW hints "GLFW_OPENGL_FORWARD_COMPAT" and "GLFW_OPENGL_PROFILE" on MacOS. They are needed on Mac, but may cause issues in certain cases on Linux.

## Energy Balance
- There was a memory leak on the host and GPU associated with the variables 'surfacehumidity' and 'd_surfacehumidity' that has now been fixed. It appears this issue was introduced in v1.2.63.

# [1.3.5] 2023-12-13

- Some updates to documentation and README

## Context
- There was an error in the documentation for reading .obj files. When object groups are specified in the .obj file, the object label/name is assigned to primitive data called 'object_label'. Previously, the documentation said 'object_group'.

## Radiation
- Images written by RadiationModel::writeCameraImage() were upside-down.
- Added direct writing of label bounding boxes (see RadiationModel::writeImageBoundingBoxes).
- Method RadiationModel::writeNormCameraImage() was not working correctly. It did not actually normalize by the maximum image value.

## Visualizer
- There was an error that could cause the Helios watermark to get cut off if the window width is too narrow.

# [1.3.6] 2023-12-21

## Radiation
- Fixed issue in RadiationModel::writePrimitiveDataLabelMap() that caused an error thinking that the file could not be opened.
- In some rare edge cases, there was a segmentation fault in direct_raygen() and diffuse_raygen() for textured triangle ray generation.
- There was an error in one version of RadiationModel::integrateSpectrum() causing an index out of bounds.
- If primitives had area of NaN, this would cause an NaN warning from the radiation model. These primitives are now excluded in RadiationModel::updateGeometry().
- There was an error in the camera model for diffuse radiation where the flux was always equal to 1.0.
- Modified input arguments of RadiationModel::writePrimitiveDataLabelMap(), RadiationModel::writeDepthImage(), and RadiationModel::writeImageBoundingBoxes() to be consistent with the convention used by RadiationModel::writeCameraImage().
- Modified RadiationModel::writeImageBoundingBoxes() to be able to specify the object class ID as an argument, and an option to append the label file so that multiple classes can be written.

## LiDAR
- Triangulation could in rare cases produce triangles with a surface area of NaN, which would cause problems when calculating the area-weighted angle distribution. These triangles are now automatically removed.

# [1.3.7] 2024-01-23

* Updated copyrights to 2024 *

## Context
- Changed methods Context::setCurrentTimeseriesPoint(), Context::queryTimeseriesData(), Context::queryTimeseriesDate(), and Context::queryTimeseriesTime() to error out if the timeseries data does not exist. Previously, only a warning was issued, which caused a segmentation fault.
- Added explicit error message to geometry generation functions when the specified texture file dies not exist.

## Radiation
- Error corrected in RadiationModel::setSourcePosition() causing source positions to be incorrect for rectangle and disk source types.
- Added overloaded version of RadiationModel::setSourcePosition() to accept a spherical coordinate instead of Cartesian vec3.
- Added method to scale a spectrum by a constant factor (see RadiationModel::scaleSpectrum() and RadiationModel::scaleSpectrumRandomly()).
- Added method to blend multiple spectra together (see RadiationModel::blendSpectra() and RadiationModel::blendSpectraRandomly()).
- Added additional cowpea spectra to the default spectral library.

## Leaf Optics
- The global data labels for generated spectra now appends an underscore between the "leaf_reflectivity_" and "leaf_transmissivity_" and the label.
- Added LeafOptics::selfTest().
- Added overloaded version of LeafOptics::run() that generates the spectra, but does not assign to any UUIDs.
- Fixed an issue where a file was reference from the RadiationModel plug-in, making it so that the LeafOptics plug-in could not be used without the RadiationModel plug-in.

# [1.3.8] 2024-03-06

* Documentation updates

## Context
- Added method to easily increment (sum) global data (see Context::incrementGlobalData()).

## Radiation
- Added check to RadiationModel::writePrimitiveDataLabelMap() to print a warning if the primitive data was empty for all pixels.
- There was an error with the radiation camera field of view in that the actual field of view in the image was half that of the specified HFOV value.

# [1.3.9] 2024-04-17

* Many documentation updates
* Leaf Optics plug-in was missing from utilities/create_project.sh script

## Context
- Added check to Context::doesPrimitiveExist() for an empty UUID vector, which could cause undefined behavior.
- Added check to Context::addPolymeshObject() for an empty UUID vector.
- Added overloaded version of Context::getObjectPrimitiveUUIDs() that accepts a 2D vector of object IDs.
- Added inequality operators for vec2, vec3, vec4, int2, int3, int4, Date, Time, and SphericalCoord.
- Changed SphericalCoord to make elements read-only to make sure that elevation and zenith angles remain linked.

## Radiation
- In selfTest() test #17, camera HFOV was increased (by factor of 2) to account for correction from version 1.3.8.

## LiDAR
- Degrees vs. radians was not handled correctly if zenith/azimuth is specified from an input XML file.

*AerialLiDAR*
- Degrees vs. radians was not handled correctly if zenith/azimuth is specified from an input XML file.

# [1.3.10] 2024-05-14

## Context
- Added Context::duplicatePrimitiveData(), Context::duplicateObjectData(), and Context::duplicateGlobalData() methods to duplicate existing primitive, object, and global data.
- Added Context::renamePrimitiveData(), Context::renameObjectData(), and Context::renameGlobalData() methods to rename existing primitive, object, and global data.
- Added Context::clearGlobalData() method to delete global data.

## Synthetic Annotation
- Added synthetic annotation self-test sample.
- Added synthetic annotation to utilities/run_samples.sh automated testing script.
- Error fixed in SyntheticAnnotation::setCameraPosition() where arguments were not being correctly assigned.
- Error fixed in SyntheticAnnotation::render() where the 'outputdir' string needed to be converted to string in the error message.

# [1.3.11] 2024-05-24

## Context
- Context::filterPrimitivesByData() was not properly filtering out primitives that did not have the specified primitive data field.

## Radiation
- Added overloaded version of RadiationModel::scaleSpectrum() that performs scaling in-place without creating new global data.
- Periodic boundary conditions were not working correctly with the radiation camera.
- Output radiation images were incorrectly flipped about the vertical axis.
- There was an error with texture coordinates. For Patches, textures were flipped about the x- and y-directions, and for Triangles, textures were flipped about the y-direction.
- Combined DGK colorboard spectra in plugins/radiationmodel/spectral_data/color_board/ into a single file named DGK_DKK_colorboard.xml. Also added a JPEG image reference of the colorboard.
- Added Calibrite ColorChecker colorboard in plugins/radiationmodel/spectral_data/color_board/.
- ColorCalibration::addColorboard() was changed to accept everything that is needed to fully define the colorboard. Accordingly, ColorCalibration::setColorboardReflectivity() was removed.
- Argument order of ColorCalibration::addDefaultColorboard() was changed to be consistent with ColorCalibration::addColorboard() and typical Helios convention.
- Added method to output optional primitive data (reflectivity and transmissivity values).
- Updated calibrated_CREE6500K_Basler-acA2500-20gc_spectral_response_[*] spectra in the camera library to be properly normalized.

# [1.3.12] 2024-06-03

*utilities/create_project.sh script*
- syntheticannotation plug-in was missing from this script, which also caused run_samples.sh to fail to run its self-test.

## Context
- Added Context::cleanDeletedUUIDs() to delete UUIDs from a vector when the corresponding primitive does not exist. This is useful, for example, to update UUID vectors when deleting some primitives, such as after calling cropDomain().
- Added fix for edge case in getFileExtension() to consider case where the filename starts with a '.' and had no extension.
- Performance improvements when adding geometry with textures. Existence of the texture file is now only checked when a new texture is added.
- Performance improvements when loading geometry with textures from XML file. The solid fraction is not re-calculated if it was previously written to the XML file being read.

## Radiation
- In-place version of RadiationModel::scaleSpectrum() was not actually added in the last commit.
- Added a check to make sure that the radiation camera position and 'lookat' coordinates are not the same.
- Added a check in RadiationModel::writeNormCameraImage() to throw an error if the camera data does not exist. This check was there for RadiationModel::writeCameraImage(), but was missing in RadiationModel::writeNormCameraImage() which could cause an uncaught error.
- Added support for periodic boundary conditions with depth images.
- Changed previous version of RadiationModel::writeDepthImage() to RadiationModel::writeDepthImageData() (writes ASCII files with depth image data).
- Added new RadiationModel::writeDepthImage(), which writes a JPEG version of (normalized) depth images.
- Removed previously deprecated version of RadiationModel::addRadiationCamera(). Use versions that take a CameraProperties struct instead.
- Behavior of twosided_flag = 0 was changed for direct sources. Previously, primitives could receive direct radiation from the back side when twosided_flag = 0, but now they cannot.
- twosided_flag = 2 was not implemented correctly and was not working in most cases.
- Added case for twosided_flag = 3 to make the primitive completely invisible to any radiation.
- Added ability to visualize radiation sources by adding their geometry to the Context. See RadiationModel::enableLightModelVisualization().
- Added ability to visualize radiation cameras by adding their geometry to the Context. See RadiationModel::enableCameraModelVisualization().

# [1.3.13] 2024-06-10

*Fixed compiler warning in pugixml (convert_number_to_mantissa_exponent).
*Disabled CMake deprecation warnings.
*Upgraded zlib to v1.3.1 to get rid of compiler warnings
*Changed standard project CMakeList.txt to require cmake v3.15 on all systems.

## Context
- Added Context::incrementPrimitiveData() methods.
- Added overloaded version of Context::scalePrimitiveData() that does not take a UUID argument and scales data for all primitives.
- Added overloaded version of Context::queryTimeseriesData() that queries at the time currently set in the Context.

## Radiation
- **** significant change **** - 'sensor_size' was removed from the CameraProperties struct and replaced with FOV_aspect_ratio. The actual sensor_size values did not matter, and it was only the ratio of the size that determined the vertical field of view.
- Added RadiationModel::getCameraPosition(), RadiationModel::getCameraLookat(), and RadiationModel::getCameraOrientation() methods.
- Geometry for the light source model visualizations was not being correctly rotated.
- Added check to RadiationModel::setCameraSpectralResponse() to make sure band actually exists.
- Added Basler 730nm and 850nm filtered mono camera spectra to camera_spectral_library.xml.
- Correction to fix camera calibration in CameraCalibration::updateCameraResponseSpectra().
- Added RadiationModel::integrateSourceSpectrum() to easily integrate a source spectrum over a given wavelength range.
- Added a filter to RadiationModel::writeImageBoundingBoxes() to not write bounding boxes with a width of 0 pixels.
- Added RadiationModel::writeCameraImageData() to write out raw camera pixel flux data to ASCII text file.
- 'flux_to_pixel_conversion' argument of RadiationModel::writeCameraImage() was not implemented.
- File was not closed at the end of writing in RadiationModel::writeDepthImageData().
- Added RadiationModel::deleteRadiationSource().
- Deleted some overloaded versions of RadiationModel::addSunSphereSource() that take a sourceID as input to change the source ID. Instead we can use RadiationModel::deleteRadiationSource() and re-add a new source.
- Fixed an error in RadiationModel::getSourceFlux() for sun sphere sources that caused the flux to be incorrect.

# [1.3.14] 2024-06-22

🚨+ NEW PLUG-IN + 🚨
- Plant Architecture plug-in merged from development branch to master repo. This plug-in is still in beta testing and is not yet fully documented and is likely to still have a number of bugs. Please report bugs as you find them.

## Context
- Cone object girth was not being properly scaled.

*Solar Position*
- Added sub-model to calculate solar fluxes and diffuse fraction for cloudy conditions based on radiometer measurements (see SolarPosition::enableCloudCalibration()).
- Changed many variable names to make units more explicit.

## Radiation
- Added "ActiveGrow_LED_RedBloom" light to light spectral library.

# [1.3.15] 2024-07-01

*Added Contributing.md file to the root directory of the repository to help guide users on how to contribute to the Helios project.

## Context
- Added checks to all primitive and object rotation functions to return immediately if a zero rotation is specified.
- Added checks to all primitive and object scaling functions to return immediately if no scaling is to be applied.
- Added checks to all primitive and object translation functions to return immediately if no translation is to be applied.
- Using JPEG textures with tiles was broken in v1.3.12.
- helios::readJPEG() was not working and would cause a segmentation fault.
- Added helios::getImageResolutionJPEG() function to query the resolution of a JPEG image without loading it.
- Changed UUID argument of Context::cropDomain() to be automatically trimmed of any UUIDs that were deleted as a result of cropping.
- vertex0, vertex1, and vertex2 data members of Triangle primitives were removed, as they are not used.
- Added Context::filterObjectsByData() methods to give object analogue to Context::filterPrimitivesByData().

## Radiation
- Added ASTMG173 diffuse reference spectrum ("solar_spectrum_diffuse_ASTMG173") to solar_spectrum_ASTMG173.xml.
- Renamed ASTMG173 direct reference spectrum to "solar_spectrum_direct_ASTMG173" in solar_spectrum_ASTMG173.xml. This should be used instead of the old "solar_spectrum_ASTMG173".
- Setting band fluxes based on the source spectrum was not working correctly.
- Changed self-test Case #9 to have a 'sun sphere' source instead of collimated source, since sun sphere sources are never checked in the self-test.

## Plant Architecture
- Added selfTest(). This is empty for now, but will be run to compile the plug-in when tests are run.
- Added PlantArchitecture::deletePlantInstance() to make sure everything gets cleaned up properly when deleting a plant. Just deleting the object IDs from the Context causes issues.
- Added check in PlantArchitecture::detectGroundCollision() to ignore objects that don't exist in the Context.
- Error corrected in parsing of plant string (credit to Heesup Yun for fixing this).
- Changed output tags/labels (e.g., plantID, peduncleID, etc.) to be object data instead of primitive data to decrease memory usage.

## Visualizer
- Fixed compiler warning about truncation when calling snprintf within Visualizer::addColorbarByCenter.

# [1.3.16] 2024-07-11

## Context
- Added Context::scanXMLForTag() to scan an XML file to see if it contains a specific tag.
- Added capability to scale a primitive about an arbitrary point in space rather than about the origin (see Context::scalePrimitiveAboutPoint()).
- Added capability to scale a compound object about its center or an arbitrary point in space rather than about the origin (see Context::scaleObjectAboutCenter() and Context::scaleObjectAboutPoint()).

## Radiation
- Added capability to set diffuse flux spectrum (see RadiationModel::setDiffuseSpectrum(), RadiationModel::setDiffuseSpectrumIntegral()). For now, the model does not calculate separate surface reflectivity and transmissivity for diffuse radiation, it assumes the same values as for the external source. This will be updated in the future.
- Added RadiationModel::getDiffuseFlux() to provide a consistent means for getting the diffuse flux for a band.
- Added cowpea stem spectra to the default surface spectral library.
- Changed the "ActiveGrow_LED_RedBloom" spectrum in the default light spectral library, as the one initially added in v1.3.14 was not correct.
- Added RadiationModel::writeObjectDataLabelMap() to write out object data pixel labels to an image.
- Error fixed that was causing incorrect pixel labels when there are multiple cameras.
- If global data for a spectrum from any of the spectral library files in plugins/radiation/spectral_data/ are referenced, the corresponding XML file will be automatically loaded.
- Labels for color board spectra in file plugins/radiation/spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml and plugins/radiation/spectral_data/color_board/DGK_DKK_colorboard.xml were changed to make them unique.

*Solar Position*
- Moved self-tests to separate selfTest.cpp file. Experimenting with new way of doing self-tests based on catch-try blocks and lambda functions.

## Plant Architecture
- Changed initial fruit scaling to be 25% of full size when it is created.

# [1.3.17] 2024-08-19

## Context
- Warning added to Context::loadOBJ() to warn if voxels exist in the Context (not supported for OBJ files). Previously, this would cause the program to crash.

## Visualizer
- Fixed a warning that could be issued if no primitives have textures AND the watermark is hidden.

## Plant Architecture
- The overall dimensions of several plant library models was increased to be more realistic.
- Behavior of axillary and terminal buds were separated such that they can be controlled independently.
- Fixed an error with leaf yaw when the leaf is compound.
- Fixed an error in which some primitives were not being deleted when ground collision was enabled.
- Sorghum model can now produce a panicle.
- Optional output object data is now finished and should be working correctly. This avoids creating large amounts of object data when not needed.
- When manually adding shoots via addBaseStemShoot(), appendShoot() or addChildShoot(), an option was added to give a taper to the shoot.

# [1.3.18] 2024-08-28

## Context
- Added parse_vec2 and parse_vec3 functions in global.h/global.cpp to parse vec2 and vec3 from a string.
- Added parse_xml_node_[*] functions in global.h/global.cpp to parse XML nodes of different types.
- Added the capability to calculate the volume of closed compound objects.
- Added the capability to hide primitives and objects within the Context such that they won't be visualized or used in plug-in calculations.
- Many performance improvements in the Context.

## Radiation
- Added warning if users add more than one sun source.

## Plant Architecture
- Changed name and behavior of shoot parameters: "insertion_angle_tip", "insertion_angle_decay_rate", "internode_length_max", "internode_length_min", "internode_length_decay_rate" (previously "child_insertion_angle_tip", "child_insertion_angle_decay_rate", "child_internode_length_max", "child_internode_length_min", "child_internode_length_decay_rate"). All of these parameters now apply to the shoot type corresponding to the ShootParameters structure and not the child.
- Added new phytomer parameter 'leaf.unique_prototypes', which controls copying of leaf prototypes to improve efficiency. By default, leaf.unique_prototypes = 1, which means that all leaves will look identical. Increase this value to increase random variation.
- Added Eastern redbud and asparagus models to the library.
- Base framework for carbohydrate model added. These methods were moved to src/CarbohydrateModel.cpp.

## Visualizer
- There was an error in the case 'samples/visualizer' related to the sun position. Thanks to Jan Graefe for pointing this out.

## Energy Balance
- Removed use of "evaporating_faces" input primitive data to specify whether the leaf is hypostomatous or amphistomatous. Instead, a more general version using 'stomatal_sidedness' was implemented which allows continuous variation in stomatal sidedness.

## Photosynthesis
- Treatment of twosided_flag and stomatal sidedness in the moisture/CO2 conductance is now consistent with the energy balance model.

# [1.3.19] 2024-09-20

**Changed to c++17 standard**

## Context
- Added capability of manually setting triangle vertices (see Context::setTriangleVertices()).
- OBJ writer checks if output directory exists, and if not it creates it (and if it can't create it an error is thrown).
- OBJ writer automatically copies all model texture files to the output directory.
- Specular material property in OBJ output models was not being set when there is a texture.
- Added version of Context::addTubeObject() that allows for explicit specification of texture mapping coordinates.
- Added method to append a segment to an existing tube object (see Context::appendTubeSegment()).
- Added methods to scale tube objects in the radial and axial directions (see Context::scaleTubeGirth() and Context::scaleTubeLength()).
- Added decrement (-=) operator for vec2, vec3, vec4, int2, int3, and int4 vector types.
- Added inequality (!-) operator for int2, int3, and int4 vector types.
- Added unary minus (multiply by -1) operator for vec2, vec3, vec4, int2, int3, and int4 vector types.
- A memory leak was fixed in helios::getImageResolutionJPEG() that would occur when JPEG image textures are used.

## Plant Architecture
* Many significant updates to the plant architecture model. Some parameter names have changed.
- Shoots are now made up of tube objects rather than cone segments.
- Shoot tubes can be textured.
- Peduncle tube color can be explicitly specified.
- Context geometry is not updated until the end of the specified timestep (i.e., the argument of PlantArchitecture::advanceTime()). For example, if advanceTime() is given a timestep of 50 days, the Context geometry will only be updated at the end of 50 days. This makes the model run much faster.
- Library models added for lettuce and wheat.

## Photosynthesis
*Credit to Kyle Rizzo for these updates
- Temperature response functions have been overhauled in the photosynthesis model. Users have the option to select from several different types of temperature response functions.
- Users can now choose between a rectangular hyperbolic or non-rectangular hyperbolic J light response function.
- Added a photosynthesis model (FvCB) parameter library for a range of species.
- Added helper functions to assist setting the parameters for the temperature response functions.

## Visualizer
- A memory leak was fixed in Visualizer::addTextboxByCenter due to freetype library not being properly cleaned up.

## Radiation
- Reflected and transmitted diffuse radiation flux values were not correct. They were not being properly scaled by the ambient diffuse flux.

## Stomatal Conductance
*Credit to Kyle Rizzo for these updates
- Added a stomatal conductance model (BMF) parameter library for a range of species.

# [1.3.20] 2024-10-21

## Context
- Texture mapping for sphere objects was not correct. The top cap was not being properly mapped, resulting in a dark spot.
- Added Context::listGlobalData() to list out all global data in the Context.
- There were still issues with tube twisting when using Context::appendTubeSegment() based on the implementation from v1.3.19. This should now be fixed.
- Modified helios vector types vec2, vec3, and vec4 normalize methods to also return the normalized vector. This allows for chaining of operations.

## Energy Balance
- Added overloaded version of EnergyBalanceModel::addRadiationBand() to add multiple bands at the same time by passing a vector.

## Stomatal Conductance
- Error in StomatalConductanceModel::setBMFCoefficientsFromLibrary(const std::string &species_name) function fixed.

## Plant Architecture
- Added capability of having evergreen plants. This is controlled through PlantArchitecture::setPlantPhenologicalThresholds();
- Added olive, pistachio, and apple trees and grapevine (VSP) to the library.
- 'flower_arrangement_pattern' parameter of the internode has been removed. It currently now follows the phyllotaxy of the parent shoot.
- An error was corrected in which the first peduncle segment was being set to the internode color.
- An error was corrected that would cause inconsistent behavior of the shoot internode when there were multiple child shoots originating from the same node.
- Changed how girth scaling is handled. The shoot girth is now calculated based on the downstream leaf area. The new parameter is called "girth_area_factor", which is cm^2 branch area / m^2 downstream leaf area.
- The parameter 'flowers_per_rachis' was changed to be 'flowers_per_peduncle'.

## Radiation Model
- When writing output images, the radiation model now checks to make sure the output directory exists and creates it if it does not.

## Synthetic Annotation
- Error (Windows OS) fixed to use the filesystem library for file management instead of mkdir. Credit to Sean Banks for this edit.

## Weber-Penn Tree
* Credit to Corentin LEROY for these edits
- Addition of variable range parameters for WeberPenn tree generation
- Addition of a sample project to demonstrate how one could build tree orchards based on a WeberPenn XML configuration

## Canopy Generator
* Credit to Corentin LEROY for these edits
- Handling of parameters that were not parsed from a CanopyGenerator XML configuration for grapevine canopies
- A refactoring of the way canopy types are handled. Now all canopy classes inherit from BaseCanopyParameters, and all grapevine canopy classes inherit from BaseGrapeVineParameters. On top of that, parameters read from an XML file are stored so that canopies or individual plants can be built later on. This allows client code to have no knowledge at all about the types of canopies or plants that can be built; this information is exclusively written in the configuration file.
- Addition of variable range parameters for grapevine canopies
- Addition of parameters to create dead plants or to have missing plants (holes) in grapevine canopies
- Addition of parameters to set the cordon length independently from the plant spacing for grapevine canopies. Until now, the plant spacing was used so that there was no discontinuity between neighbor vine stocks. These new parameters allow to potentially have gaps between vine stocks, or even have them overlap with each other (if the cordon length is greater than the plant spacing).
- Addition of a sample project to demonstrate how one could build realistic vineyards based on CanopyGenerator XML configuration.

# [1.3.21] 2024-10-24

## Context
- There was an error introduced in the previous version for writing OBJ files. If no output directory was explicitly specified, the write would fail.
- Added validateOutputPath() function to global.cpp to validate output files and directory paths.
- Replaced getFile*() functions in global.cpp with more robust std::filesystem functions.
- Added improved file validation to Context::writeXML().

## Radiation
- There was an error introduced in the previous version for writing image output files. If no output directory was explicitly specified, the write would fail. Removed the validateOutputPath() function implemented in the previous version, and replaced it with a better version implemented in global.cpp.

## Plant Architecture
- The 'tortuosity' shoot parameter was multiplied by a hard-coded factor of 5.0. This was removed, and all tortuosity values in the plant library were scaled by a factor of 5.
- There were some errors with the parameters of some plants introduced in the previous version.
- Added file validation to PlantArchitecture::writePlantStructureXML().

# [1.3.22] 2024-11-01

## Context
- Methods Context::sumPrimitiveSurfaceArea(), Context::calculatePrimitiveDataAreaWeightedMean(), and Context::calculatePrimitiveDataAreaWeightedSum() to check if primitive area is NaN, and if so exclude it from calculations.
- Added parse_int2(), parse_int3, and parse_RGBcolor functions to global.cpp.
- Added open_xml_file() function to global.cpp that opens an XML file and checks for basic validity.
- Added 'Location' type to helios_vector_types.h to store latitude, longitude, and UTC offset.
- Added Context::listTimeseriesVariables() to return a list of all existing timeseries variables.

## Plant Architecture
- Some updates to soybean model parameters.
- Added methods to query UUIDs and Object IDs for all plants in the model to avoid having to loop over each plant instance.
- Some additional checks were needed to make sure the tube internode object actually exists in the Context, otherwise there could be an out-of-bounds error.
- Removed the Shoot Parameter 'internode_radius_max', as it is not needed anymore after the pipe-model-based internode girth scaling was added.
- Corrected some issues with reading/writing plants using strings or XML. Namely, some parameters like the phyllotactic angle were not being applied correctly across shoots.
- Added PlantArchitecture::getCurrentPytomerParameters() to make it easy to get all the phytomer parameters structures to pass to PlantArchitecture::generatePlantFromString().

## Radiation
- Split spectral_data/surface_spectral_library.xml into separate files for soil, leaves, bark, and fruit, and added many new species. Credit to Kyle Rizzo for these additions.
- Some default values were set in RadiationModel.cpp while others were set in RadiationModel.h. Everything was moved to be set in the RadiationModel constructor, which is in RadiationModel.cpp.

*Solar Position*
- Default constructor changed to load the location based on the location set in the Context.
- UTC offset variable changed from int to float type.

## Visualizer
- Visualizer::printWindow() now creates the output directory if it does not already exist.

# [1.3.23] 2024-11-19

## Context
- Still some lingering issues with Context::addTubeObject() in which tube creation could fail.

## Visualizer
- Added warning in Visualizer::buildContextGeometry() if there is existing Context geometry already in the Visualizer that may have needed to be cleared.
- When calling Visualizer::clearGeometry(), the colorbar range was not being reset, which could cause unexpected behavior.
- Changed default lighting intensity to be brighter in order to better match the default behavior of 3rd party renderers such as Blender.
- Added method Visualizer::setLightIntensityFactor() to allow users to adjust the intensity of the lighting in the scene.

## Plant Architecture
- Some updates to bean, cowpea, and apple model parameters and assets.
- Changed PlantArchitecture::buildPlantCanopyFromLibrary() to accept an optional argument specifying the germination rate.
- Fixed error in which fruit could potentially become disconnected from their peducle over time.
- Fixed error that was causing flowers not to open.
- Added optional phenological parameter 'max_leaf_lifespan' to allow leaves to die based on age, which is especially important for evergreen plants.
- Updates to carbohydrate model. Credit to Ethan Frehner

## Photosynthesis
- Error corrected that could cause an incorrect temperature response depending on how model parameters are set. Thanks to Kyle Rizzo for this fix.
- Added methods to disable output messages.

## Stomatal Conductance
- Added methods to disable output messages.

## Canopy Generator
- Error fixed when parsing XML files for strawberry canopies. Thanks to Heesup Yun for the fix.

## Radiation
- Error corrected in the camera model related to the field of view and aspect ratio. Thanks to Peng Wei for this fix.

# [1.3.24] 2024-11-27

* Added utilities/dependencies.sh script to automatically install all dependent libraries on Linux and MacOS. Credit to Sean Banks for creating this script.

## Context
- Seemed to still have an issue with tube objects, where there was severe twisting.

## Plant Architecture
- Numerous model improvements via parameter tuning and texture adjustments
- Added rice, maize, and walnut models
- Changed map structures that hold organ prototypes to use the prototype function pointer as the key rather than a string, which allows different prototypes along a single shoot.
- Peduncles were not correctly lining up with their parent petioles.

## Radiation
- Error corrected in radiation camera documentation example code.

## Photosynthesis
- Updated code to include stomatal sidedness option (this was already described in the documentation, although it had not been implemented).

# [1.3.25] 2025-01-03

*Updated copyright dates to 2025*

## Plant Architecture
!!! Many significant changes in this release !!!
- The way leaf prototypes are generated has changed. There is now a modifiable parameter set that controls procedural leaf generation (or loading a model from an OBJ file).
- Changed generic leaf prototype function to be able to represent leaf "buckling", such as what happens to long leaves of grasses like maize.
- Removed shoot parameter 'leaf_flush_count'.
- Added shoot parameter 'max_nodes_per_season' to allow the shoot to stop growing for the season, but continue adding nodes in future seasons.
- Phenological thresholds for plants have been changed to be realistic (e.g., the phenological cycle for perennial plants corresponds to days in a year).
- A maximum age limit for plants has been added so users don't inadvertantly specify a very old plant that will cause the program to hang.
- Changed the way that vegetative bud break probability is handled such that it can vary along the shoot. The relevant parameters are now 'vegetative_bud_break_probability_min' and 'vegetative_bud_break_probability_decay_rate'.
- Changed the way that downstream leaf area is calculated to improve computational efficiency. This required some changes in the 'girth_area_factor` parameter.
- Committed all Blender project files used to create organ OBJ models: plugins/plantarchitecture/assets/Blender_organ_models/
- Moved initial internode radius from being a shoot parameter to a phytomer parameter, since it is constant along the shoot and over time.
- Added overloaded PlantArchitecture::advanceTime() method that allows to specify the time step in years and days for advancing over long time periods.
- Changed the name of the shoot parameter 'phyllochron' to 'phyllochron_min' to be consistent with how it will be used in the carbohydrate model.
- Added actual self-test that builds all plants in the library.

## Visualizer
- Problem fixed in freetype zconf.h file that caused build errors on latest MacOS compilers.
- All output messages were not being properly suppressed after calling disableOutputMessages().

## Canopy Generator
- Many edits made regarding how parameters with random variation are handled, mainly to ensure proper constraints on values (e.g., always positive).

# [1.3.26] 2025-01-17

## Context
- Definition of global PI_F variable in global.h was causing a build issue on some compilers

## Visualizer
- Added mouse-based camera controls in the interactive visualizer. Credit to Sean Banks for this update.

## Plant Architecture
- Added methods for querying bulk properties of a plant (total leaf count, stem height, plant height, leaf angle distribution) and write plant mesh vertices to file.

# [1.3.27] 2025-02-11

**Some experimental OpenMP parallelization has been added. This is not enabled by default yet, but can be turned on by setting the CMake option ENABLE_OPENMP to ON.**
**Updates to core CMakeLists.txt to use more modern CMake features.**

## Context
- Context::pruneTubeNodes() was added to allow for removal of part of a tube object.

## Plant Architecture
- Added ground cherry weed model.
- Minor fix in PlantArchitecture::getPlsantLeafInclinationAngleDistribution() to prevent rare out of range index.
- Changed PlantArchitecture::getPlantLeafInclinationAngleDistribution() to area-weight the distribution, and removed the 'normalize' optional argument.
- Added PlantArchitecture::prunBranch() method to remove all or part of a branch and its downstream branches.
- Removed shoot parameter 'elongation_rate'. There is now an 'elongation_rate_max' shoot parameter that can be set by the user. The actual elongation rate can be reduced dynamically if the carbohydrate model is enabled.
- Many updates to carbohydrate model. Credit to Ethan Frehner for these updates.

## LiDAR
- Added exportTriangleAzimuthDistribution() to write the triangulated azimuthal angle distribution to a file. Credit to Alejandra Ponce de Leon for this addition.
- The output distribution from exportTriangleInclinationDistribution() was not being normalized.

## Radiation
- Added bindweed spectra to the default library.
- There was an error in the writeObjectDataLabelMap() method that could cause undefined behavior if the UUID in the pixel label map did not exist in the Context.
- There was an error in the writeObjectDataLabelMap() method where the primitive UUID was being used instead of the object ID.
- There was an error in the model that could cause incorrect assignment of radiative properties if runBand() is called with fewer bands than a previous call to runBand().
- The previous version could cause unnecessary updating of radiative properties, resulting in a performance hit. This has been fixed.
- Revised the radiation plug-in CMakeLists.txt to use a modern CMake approach for building CUDA source files. This update should also enable indexing of .cu source files in IDEs such as CLion.

# [1.3.28] 2025-02-25

🚨+ NEW PLUG-IN + 🚨
- project builder plug-in added. GUI and XML interface for creating projects. This is still in beta testing. Credit to Sean Banks for this addition.

++Substantial expansion of self-tests for boundarylayerconductance, energybalance, photosynthesis, solarposition, stomatalconductance, syntheticannotation, and visualizer plug-ins
++New documentation theme

## Context
- Many upgrades to substantially improve performance for CPU code.
- Upgrades to OBJ file writing, including making Gazebo compatable and adding option to write normals to OBJ files. Credit to LeRoy Corentin for this addition.

## Radiation
- Minor change in RadiationModel::runBand() to set all primitive data values in a single Context method call for efficiency.
- Changed up logic for radiation property setting. If a spectrum is specified for reflectivity or transmissivity, this can be overridden for a single band by specifying the primitive data "reflectivity_[bandname]" or "transmissivity_[bandname]".

## Plant Architecture
- cowpeaLeafPrototype_unifoliate_OBJ() asset function was incorrect in the last commit, and would cause an error if used.
- The plugin now writes primitive data "object_label" for all organs it creates.
- Some minor fixes to bean OBJ assets.
- Removed erroneous organ labels in MaizeTassel.obj and TomatoFlower.obj assets.

## Boundary-Layer Conductance
- Corrected spelling of 'Pohlhausen'. This should still be backward compatible, as it also accepts the spelling "Polhausen".

## Photosynthesis
- The way parameters are set for the FvCB model has been clarified to be more explicitly clear about how temperature response is being treated.

# [1.3.29] 2025-04-14

*Tutorial #12 added for canopy water-use efficiency example. (credit to Ethan Frehner for this addition)

## Context
- Added a check to Context::getUniquePrimitiveParentObjectIDs() to return an empty output objID vector if given an empty input UUID vector.
- Error corrected with OBJ file loading that could cause incorrect material depending on the ordering of values in the material file.
- The latest Xcode version on MacOS caused an error in libpng, which was fixed.

## Radiation
- There were many instances where calling setter methods after calling updateGeometry() would cause issues. This has been fixed by making sure the setters force recalculation of the radiative properties.

# [1.3.30] 2025-05-01

*Added automated performance benchmarking
*Fixed documentation sidebar navigation collapse issue

## Context
- Added Context::setPrimitiveNormal(), Context::setPrimitiveElevation(), and Context::setPrimitiveAzimuth() methods.
- Added sample_Beta_distribution() funciton to randomly sample from a Beta distribution for inclination angle.
- Added sample_ellipsoidal_distribution() funciton to randomly sample from an ellipsoidal distribution for azimuth angle.
- Added explicit normalize() function for vec2, vec3, and vec4 types.
- Minor performance improvements for Helios vector types and Context operations.
- Added OpenMP parallelization for primitive area calculations from transparency textures.
- Changed the type of 'primitives' and 'objects' from std::map to std::unordered_map to improve performance.
- Changed Context::getObjectPrimitiveUUIDs( uint ObjID ) method to support passing an ObjID of 0.
- Context::getObjectPrimitiveUUIDs( const std::vector<uint> &ObjIDs ) was missing a check for existence of the ObjID object.
- Overloaded getPrimitiveParentObjectID( const std::vector<uint> &UUIDs  ) added to take a vector of UUIDs.
- When tile objects are created, completely transparent sub-patch tiles are no longer deleted because this causes problems in the radiation model.
- Context primitive and object class constructors were made 'protected' so that they cannot be instantiated outside of the Context.
- Many new self-tests were added (credit to Ismael Mayanja for this addition).
- Refactoring of PNG and JPEG image to improve robustness.

## Radiation
- For two-sided primitives, diffuse ray launches were split into two passes to dramatically improve thread coherence and thus performance.
- An issue with texture mapping for tile objects was fixed.
- OptiX variable declarations were moved to RayTracing.cuh header file.
- Moved texture sampling in rayGeneration.cu into a separate function to avoid redundant code.
- New method for handling periodic boundaries was implemented to improve performance.
- Completely transparent sub-patch tiles are no longer excluded from calculations.

## Plant Architecture
- Many updates to carbohydrate model code. Credit to Ethan Frehner for these updates.
- Added methods to calculate the leaf azimuth distribution of a plant (see PlantArchitecture::getPlantLeafAzimuthDistribution()), and overloaded versions of leaf angle distribution calculation functions that take a vector of multiple plant IDs.
- Added method to set the leaf angle distribution of a plant.

## Visualizer
- Visualizer::setColorbarRange() was not properly setting the colorbar range.

## Project Builder
- Many updates and improvements to the project builder plug-in. Credit to Sean Banks for these updates.

# [1.3.31] 2025-05-10

## Context
- Fixed minor compiler warning that appeared within Context::readPNGAlpha().
- Added overloaded Context::writePLY() to take a vector of UUIDs. Also added optional parameter to silence output printing.
- Some minor refactoring was done for the existing Context::writePLY() method to better validate file names and paths.
- helios::CalendarDay() function was not working correctly. The bug appeared to be introduced in the last version.

## Radiation
- Changed RadiationModel::setDiffuseSpectrum() such that it always needs to be associated with a band. This prevents confusion about the required order of the method call in the code.
- The spectral data file plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml, the diffuse spectrum contained negative values which caused an error.

## Plant Architecture
- Added PlantArchitecture::isPlantDormant().
- Carbohydrate model updated to take a parameter structure rather than having hard-coded model parameters. Thanks to Ethan Frehner for these updates.
- Minor refactoring improvements throughtout plant architecture code.
- Error corrected that was causing plants to flower too quickly if starting with closed flowers.

## Visualizer
- Minor refactoring improvements throughout visualizer code.
- There was an error in the mouse-based camera controls that could cause phantom camera rotation while zooming.

## Project Builder
- Lots of project builder updates and bug fixes. Credit to Sean Banks for these updates.

# [1.3.32] 2025-05-30

## Context
- Minor refactoring to improve code clarity.
- Added pre-defined RGBA colors, which mirror the existing RGB definitions.
- Added Context::getPatchCount() and Context::getTriangleCount() methods.
- Context::getPrimitiveCount() method now has optional argument to control whether to include hidden primitives in the count.
- Added overloaded Context::hidePrimitive() and Context::hideObject() that takes a single UUID or object ID.
- Implemented caching of primitive solid fraction calculations, which should dramatically speed up build times when there are a lot of Tiles/Triangles with transparency masking.
- Added uint2, uint3, and uint4 vector types.
- Added macro 'scast' to avoid having to type out static_cast.
- Added many additional debug checks for object and primitive methods to catch index out-of-bounds errors.
- Added object type enumeration OBJECT_TYPE_NONE to handle the case when the object type is queried but the object ID = 0 (no object assigned).
- Changed how 'dirty' flags are handled in the Context. Dirty flags are handled on a per-primitive basis. If a primitive's geometry or primitive data are modified, it will get flagged as dirty.
- Small issue with libjpeg and libpng include file paths was fixed to avoid possible unusual build errors on some systems.

## Visualizer
- Core of Visualizer plug-in was re-written. This should dramatically improve performance, and allows for updating of geometry on a per-primitive basis. It should be backward compatible, except for the API modifications given below.
- Visualizer::addDisk() methods have been removed.
- Line size argument for Visualizer::addLine() was removed because it didn't do anything.
- Removed Visualizer::plotInit() as this seemed to be dead code that was not used anywhere.
- Significant improvements to shadows, which should remove the issue of pixelated shadows when the domain is very large, and erroneous shadows cast on the ground when the aspect ratio of the domain is very high.
- Added Visualizer::displayImage() methods to display a PNG or JPEG image in the visualizer window.
- Depth images were re-worked. Calling Visualizer::plotDepthMap() works without calling other methods. Calculation of physical depth should now be correct.

## Plant Architecture
- Fixed CMakeLists.txt to automatically copy assets into the build directory if they are modified.
- Re-defined interpolateTube() function as a private member of the PlantArchitecture class to avoid naming clash with the interpolateTube() function of the canopy generator plug-in.
- Updates to the carbohydrate model to better represent carbon costs of wood growth. Credit to Ethan Frehner for these updates.

## Canopy Generator
- Minor change needed to achieve compatability with the new way that textures are handled in the Context.

## Energy Balance
- Removed 'using namespace helios' from EnergyBalance.cu because it was causing a naming clash with the new uint3 vector type.
- Reconfigured the plugin CMakeLists.txt to automatically detect all supported GPU compute capabilities and build them all. This should likely result in some performance improvements.

## LiDAR
- Reconfigured the plugin CMakeLists.txt to automatically detect all supported GPU compute capabilities and build them all. This should likely result in some performance improvements.
- Calls to Visualizer::addLine() were updated to reflect removal of line width argument.

## Voxel Intersection
- Removed 'using namespace helios' from VoxelIntersection.cu because it was causing a naming clash with the new uint3 vector type.
- Reconfigured the plugin CMakeLists.txt to automatically detect all supported GPU compute capabilities and build them all. This should likely result in some performance improvements.

## Project Builder
- Many more updates to the project builder plug-in, including ability to change the lighting model, texture tiling of the ground, and many others. Credit to Sean Banks for these updates.

## Radiation
- Added specular reflection in synthetic images for collimated and sun sphere sources (still need to implement disk, rectangle, and sphere sources).

Co-authored by: Ethan Frehner <ehfrehner@users.noreply.github.com>
Co-authored by: Sean Banks <smbanx@users.noreply.github.com>

# [1.3.33] 2025-06-08

## Context
- Added Context::getDirtyUUIDs() and Context::getDeletedUUIDs().
- dirty_flag was not being properly set to true in Primitive::setTransformationMatrix.
- validateOutputPath() function should have been marked [[nodiscard]].
- Fixed out of bounds error in helios::vecmult().
- Patch for helios::deblank() to avoid possible buffer overflow issues.
- Patch for helios::separate_string_by_delimiter() so it always returns the final token and works when no delimiter is found.
- Added a robust file-extension check in helios::readPNGAlpha() to handle filenames without dots safely.

## Visualizer
- Changed GeometryHandler::allocateBufferSize() to be general for any primitive type rather than being specific for rectangles and triangles in a single call.
- Added GeometryHandler::doesGeometryExist(), GeometryHandler::getAllGeometryIDs(), GeometryHandler::getPrimitiveCount(), and GeometryHandler::getDeleteCount().
- The visualizer now only updates Context geometry that has changed since Context::markGeometryClean() was last called.
- Some errors were fixed in the defragmentation routine.
- Visualizer::printWindow() was not properly validating the output file. It now auto-appends .jpeg if no file extension is given, and properly validates the output file path.

## Radiation
- Inside RadiationModel::updateGeometry(), removal of primitives deleted from Context changed from using std::erase() to swap-and-pop to improve performance.

## Energy Balance
- Output primitive data 'storage_flux' moved to optional output primtitive data.
- Net radiation flux added as optional output primitive data.
- Added a simple air temperature and humidity transport model that calculates the evolution of a spatially uniform air temperature and humidity based on an energy balance.

## Radiation Model
- Added specular reflectance scaling factor option, which is enabled by setting the primitive data value "specular_scale".

## Plant Architecture
- Updates to carbohydrate model to prevent shoot concentrations from going negative at the beginning of the simulation. Credit to Ethan Frehner for this update.

## Project Builder
- Several updates including changing how canopies are handled, and improving the rig tab. Credit to Sean Banks for this update.

Co-authored by: Ethan Frehner <ehfrehner@users.noreply.github.com>
Co-authored by: Sean Banks <smbanx@users.noreply.github.com>

# [1.3.34] 2025-06-15

- Added .clang-format file for consistent code formatting across IDEs and agents.
- Added AGENTS.md to give some basic instructions for AI agents.

## Context
 - Reworked Context::cleanDeletedUUIDs() and Context::cleanDeletedObjectIDs() to erase invalid IDs using iterators, preventing unsigned underflow issues when vectors are empty.
 - Fixed a loop in Context::loadMTL() that could cause unsigned underflow.
 - Context::writeOBJ() can now maintain object groups based on the value of the primitive data "object_label".
 - Context::writeOBJ() was defining normals (if enabled), but they were not being assigned to faces. This has been fixed.
 - Improved stability of Tube::appendTubeSegment(), Context::addConeObject(), Context::addTubeObject(), and helios::rotatePointAboutLine() to prevent dividing by nearly zero when the vector length is very small.
 - Added helios::powi() function to calculate integer powers of a number, which is more efficient than using std::pow() for integer exponents and avoids having to do "a = b*b*b*b", for example.
 - Added methods Context::scaleObjectAboutOrigin() and Context::rotateObjectAboutOrigin() to scale and rotate an object about its origin, which can be set with Context::setObjectOrigin().

 ## Radiation
 - Added many new camera response spectra to the spectral library (spectral_data/camera_spectral_library.xml).
 - The spectral labels were not correct for the default DGK colorboard (CameraCalibration::addDefaultColorboard()).
 - Added the ability to create the Calibrite ColorChecker Classic in the scene. Accordingly added explicit functions `CameraCalibration::addDGKColorboard()` and CameraCalibration::addCalibriteColorboard() to add the two available color boards.
 - Added `RadiationModel::applyImageProcessingPipeline()` to apply a digital camera-like processing pipeline to an image.

 ## Plant Architecture
 - Added capability to scale petioles of the same internode independently.

# [1.3.35] 2025-06-20

* If the user changes their Helios version, CMake will now automatically rebuild the entire project. This avoids some issues where updating the code version causes unexpected CMake errors because the user is using an old build and forgot to reconfigure.
* Many minor fixes to documentation and docstrings.
* Added github workflow for testing on self-hosted GPU runner.
* Project CMakeLists.txt files should now reference core/CMake_project.cmake instead of `core/CMake_project.txt`. The old file is still present for backward compatability.
* Tutorials 3, 4, and 9 added in `Tutorials.dox`

## Context
- Added `Context::getTubeObjectNodeCount()` to return the number of nodes in a tube object.
- Added clamping of (u,v) coordinates to [0,1] range in Patch and Triangle constructors.
- Added out of memory handler so that the program will exit gracefully if it runs out of memory.

## Plant Architecture
- In self-tests, enabled the ground collision detection option so that it will be tested.
- Multiple petioles on the same internode can now be scaled independently.
- Fixed an error that was causing leafand roll and yaw rotations to not be applied for non-compound leaves.
- Fixed an out of bounds error in `PlantArchitecture::readPlantStructureXML()` when the internode has no leaf.
- Shoot pruning was not correct when the internode tube has multiple segments.
- Ground collision detection now prunes the shoot if all downstream phytomers had their leaves removed from ground collision detection.

## Radiation
- Added Datacolor SpyderCHECKR 24 color calibration board, and associated CameraCalibration::addSpyderCHECKRColorboard() method.

## Energy Balance
- The energy balance self-tests were failing because of tiny differences introduced in the previous version 1.3.33.

## Visualizer
- The visualizer was running very slow when there were a lot of deleted primitives in the Context.
- Increased the threshold for buffer defragmentation from 10,000 to 250,000 so that it won't happen so often.
- Shadows were not working properly on some Linux systems. This was fixed.
- There have been persisting issues where exported images were black on Linux systems. This has hopefully been fixed.

## Project Builder
- Added a project builder self-test to test the project builder plug-in.
- Many stability and performance improvements to the project builder plug-in. Credit to Sean Banks for these updates.
- Fixed SIGBUS error that happened when closing the project builder window.

# [1.3.36] 2025-06-25

* Many minor fixes to documentation and docstrings.
* Moved `AGENTS.md` to `doc/AGENTS.md`

## Context
- When reading OBJ files with both `map_Kd` and `map_d` specified, ensured `map_Kd` is preserved when map_d specifies a different file and only use `map_d` when no diffuse texture is provided.
- Added overloaded `Context::addTile()` and `Context::addTileObject()` that allow periodic tiling of the texture image.
- Fixed `Context::readPNGAlpha()` to properly handle the case when the PNG file has no alpha channel.
- Changed `Context::cleanDeletedUUIDs()` back to looping backward rather than using an iterator.

## Plant Architecture
- Fixed an error with the leaf pitch and roll angles when calling `Phytomer::rotateLeaf()` with a non-compound leaf.
- Some minor tweaks to model parameters for cowpea, bean, and tomato plants.

## Visualizer
- Changed default lighting parameters in shader, which should give a bit more ambient lighting and make plant visualizations look a little better.
- Fixed `read_png_file()` to properly hand the case when the PNG file has no alpha channel.
- Fixed an issue with the sky dome where the sky texture mapping was not correct.
- Changed clipping planes for camera and shadow rendering to be more robust, which should fix some issues with shadows not being rendered correctly.
- Changed Helios watermark to version with white border, which should be more visible on dark backgrounds.
- Fixed a bug where if you update the visualizer window with RGB coloring, then call `colorContextPrimitivesByData()`/`colorContextObjectsByData()`, then re-render, it would break the coloring.
- Fixed an issue where the back side of primitives was being rendered way too dark.

## Radiation
- Minor fix in `RadiationModel::normalizePixels()` to avoid a compiler error in some pixels (wrap `(std::numeric_limits<float>::max)()` in parentheses).

## Energy Balance
- Made header variables cp_air_mol and lambda_mol constexpr instead of const, hopefully to fix some issues where they would not get properly defined in `EnergyBalance.cu`.

## Project Builder
- Several updates to the project builder, including fixing the ground updating, changing some default values, adding option to set the light direction based on spherical coordinates, and some other features. Credit to Sean Banks for these updates.

# [1.3.37] 2025-06-27

* Updated `utilities/run_samples.sh` to have an option to redirect command outputs to a user-specified log file.
* Updated all testing workflows to print the output log if the run fails.
* Many documentation updates and fixes.
* Renamed this file to `CHANGELOG.md` and added markdown formatting.

🚨+ NEW PLUG-IN + 🚨
- Added a new plant hydraulics plug-in. Credit to Kyle Rizzo for developing this plug-in.

## Context
- Added overloaded + and - operators for `helios::RGBcolor` and `helios::RGBAcolor` vector types.

## Plant Architecture
- Fixed a bug in `PlantArchitecture::interpolateTube()`. 
- Added 10 self-tests.

## Radiation
- Refined `RadiationModel::applyImageProcessingPipeline()`. It now performs a standard set of processing steps, and has an option to apply HDR toning.