# Changelog

# [1.3.83] 2026-08-27

## Core

- `Polymesh` compound objects now store an indexed face set (deduplicated vertices, faces, optional per-vertex normals and texture coordinates) with a bidirectional mapping between member primitive UUIDs and faces. New accessors `getPolymeshObjectVertices()`, `getPolymeshObjectFaces()`, `getPolymeshObjectVertexNormals()`, `getPolymeshObjectVertexUV()`, `getPolymeshObjectVertexCount()`, `getPolymeshObjectFaceCount()`, `getPolymeshObjectFaceIndexForPrimitive()` and `getPolymeshObjectPrimitiveUUIDForFace()` expose it, and `setPolymeshObjectTopology()` attaches a face set to a mesh built programmatically.
- `Context::loadOBJ()` now reads `vn` vertex normal records, which were previously parsed as unrecognized lines and silently discarded. Normals are attached to the mesh and reported as `NORMAL_SOURCE_AUTHORED` by `getPolymeshObjectVertexNormalSource()`; Helios never synthesizes normals implicitly, and `computePolymeshObjectVertexNormals()` must be called explicitly to generate them.
- `Context::loadOBJ()` face parsing now handles all four OBJ vertex reference forms (`v`, `v/vt`, `v//vn`, `v/vt/vn`) and negative (relative) indices. Previously a negative index was silently dropped from the face, producing a malformed shorter face, and a third index field could not be read at all.
- `Context::loadPLY()` now reads `nx`, `ny` and `nz` vertex properties as per-vertex normals, which were previously discarded, and applies the same up-axis permutation used for vertex coordinates.
- `Context::loadOBJ()` and `Context::loadPLY()` now group the triangles they create into a `Polymesh` object that retains the connectivity of the source file. The returned UUID vector is unchanged; the object ID is obtained with `getPrimitiveParentObjectID()`.
- `Context::writeOBJ()` now writes geometry belonging to a `Polymesh` against the mesh's shared vertices, emitting each vertex once and referencing it by index from every face that uses it, rather than three independent vertices per triangle. With `write_normals=true` it likewise writes one true per-vertex normal per shared vertex instead of one flattened face normal per primitive, so a smooth mesh survives a load-write-load round trip instead of degrading to a faceted one. Output for geometry that is not part of a `Polymesh` with retained connectivity is unchanged.
- Deleting a primitive belonging to a `Polymesh` now repairs the mesh topology: the corresponding face is removed, vertices left unreferenced are dropped, and remaining faces are reindexed. Note that repairing a hole makes area-weighted vertex normals adjacent to the new boundary subtly incorrect.
- `Context::getPolymeshObjectVolume()` now verifies that a mesh carrying a face table is closed before applying the divergence theorem, and raises an error identifying the number of boundary edges instead of returning a meaningless number for an open surface. Meshes without a face table retain the previous behavior.
- Added `Context::computePolymeshObjectVertexNormals()`, which computes per-vertex normals by area-weighted averaging and splits vertices across edges exceeding a given crease angle so that hard edges stay hard.
- Added the topological queries `isPolymeshObjectClosed()`, `getPolymeshObjectBoundaryEdges()`, `getPolymeshObjectConnectedComponents()` and `getPolymeshObjectSurfaceArea()`.
- `Context::copyObject()` now carries `Polymesh` mesh topology to the copy, remapping the face table onto the copied primitives.
- `Polymesh` objects are now deformable: their individual member primitives can be translated, rotated and scaled, which other compound object types still refuse because doing so would violate the shape that defines them (a tile would no longer be planar, a tube no longer coherent). New virtual `CompoundObject::isDeformable()` reports which types allow this. Transforming a member primitive of a mesh that carries an indexed face set also updates the mesh vertices that facet references, so the stored topology continues to describe the geometry; neighbouring primitives sharing those vertices are not rewritten, so transform them too if the rendered geometry must follow. Vertex normals computed before the deformation are left unchanged and `computePolymeshObjectVertexNormals()` should be called again if exact normals are needed.
- `Context::writePLY()` now writes geometry belonging to a `Polymesh` against the mesh's shared vertices instead of three independent vertices per triangle, and emits `nx`, `ny` and `nz` vertex properties when every vertex in the file carries a normal. Previously a closed solid written to PLY reloaded as a triangle soup whose every edge was a boundary edge, so its volume could not be computed. Output for geometry that is not part of a `Polymesh` with retained connectivity is unchanged.
- `Context::deletePrimitive()` given a vector of UUIDs now detaches the primitives from their parent objects as a single batch rather than one at a time. Deleting all 500,000 primitives of a large mesh takes roughly half as long as before, and an object that maintains per-primitive bookkeeping now repairs it once for the batch instead of once per primitive.
- `Context::addPolymeshObject()` now returns the existing object when every supplied UUID already belongs to the same polymesh, rather than warning and attempting to build an object from an empty primitive list. It now also raises an error when all supplied primitives belong to other objects, a case that previously read past the end of an empty vector.
- `Context::appendTubeSegment()` now inserts the new segment's triangles into the tube's existing radial ordering rather than adding them to the end of the object's primitive list. The tube's geometry is rebuilt by assigning vertices to primitives by position, so appending previously left each primitive holding another primitive's share of the surface: the shape remained correct, but per-primitive properties did not follow it, and a tube whose nodes had different colors came out with those colors on the wrong segments.
- `Context::getConeObjectNodeRadii()` and `Context::getConeObjectNodeRadius()` now apply the object's transformation, so they describe the cone as it is currently transformed rather than as it was constructed. Scaling a cone with `Context::scaleObject()` previously left them reporting the original radii, which also made `Context::getConeObjectVolume()` combine un-scaled radii with a scaled length. `Context::scaleConeObjectGirth()` is unaffected.
- `Context::writeXML()` now records the nodes and radii of `Tube` and `Cone` objects in the object's local coordinate frame, matching how `Context::loadXML()` consumes them. They were previously written with the object transformation already applied, and since loading rebuilds the object from those values and then applies the transformation matrix recorded alongside them, a transformed tube or cone came back with the transformation applied twice: a uniformly doubled tube reloaded at four times its original size. Objects that were never transformed are unaffected, as are the other object types, which record their shape entirely in the transformation matrix.
- `Sphere`, `Tube` and `Cone` objects can now report the true surface normal of the curved shape they approximate at each vertex of their member primitives, via the new `Context::getObjectPrimitiveVertexNormals()` and `Context::doesObjectHaveAnalyticVertexNormals()`. Normals are evaluated from each shape's own definition rather than stored, so they account for taper and remain correct after the object is transformed or its nodes and radii are changed. Object types built from flat faces, such as a tile or box, report that they have no analytic normals.
- Fixed `Context::loadXML()` corrupting compound object ownership when object IDs recorded in the file collide with object IDs already assigned earlier in the same load. The `<sphere>`, `<tube>`, `<box>`, `<disk>` and `<cone>` blocks reassigned the file's object ID before using it to look up which primitives belong to that object, so an object could claim a different object's primitives and the rightful owner was left empty and destroyed. Reloading a scene written by `Context::writeXML()` reported "primitives were not added to polymesh object because they already belong to another object" warnings and produced a Context missing objects; an 11-plant maize canopy of 561 objects reloaded as 396.
- `Context::loadXML()` now fails fast when two object blocks in a file claim the same object ID, naming the duplicated ID. Both blocks were previously handed the same primitives, and reparenting them to the second object left the first empty and destroyed it, so the file loaded without complaint but produced a Context short one object. This is reachable by concatenating the bodies of two `Context::writeXML()` outputs to merge two scenes, since each numbers its objects from 1.
- Fixed object member primitive data being restored onto the wrong sub-primitives by `Context::loadXML()`. `Context::writeXML()` records each value's position within the object in the `<data>` element's `label` attribute and omits sub-primitives that do not carry the label, but the reader ignored that index and consumed the values in order, so data set on only some sub-primitives was shifted onto the first N of them -- overwriting values the per-primitive blocks had already restored correctly. `Context::writeXML()` also wrote an object's data from its full member list even when only some of those primitives were being written, which shifted every later index and recorded values for primitives absent from the file; it now writes only the members present. An object with a hidden sub-primitive, or written through the UUID-subset overload of `Context::writeXML()`, previously reloaded with shifted data and an `osubpdata_length_mismatch` warning.
- `Context::writeXML()` and `Context::loadXML()` now round-trip `Polymesh` mesh topology and the object's transformation matrix. The `<polymesh>` block previously carried neither, so a mesh that had retained its connectivity came back as a triangle soup: `getPolymeshObjectFaceCount()` returned 0, `getPolymeshObjectFaceIndexForPrimitive()` threw, a closed solid was no longer closed so its volume could not be computed, and `Context::writePLY()`/`Context::writeOBJ()` fell back to independent per-triangle vertices. Vertices, faces, per-vertex normals and their provenance, and texture coordinates are all preserved. Topology is written only when the file contains every primitive the face set refers to, so a partial write reloads as an unconnected mesh rather than one referring to absent primitives.
- Fixed `helios::parse_uint()` being unable to parse the upper half of the `uint` range. It converted through `std::stoi`, which cannot represent a value above `INT_MAX` and threw `std::out_of_range`; because only `std::invalid_argument` was caught, that escaped to the caller as a raw STL exception rather than a failed parse. A `uint` primitive, object or global data value above 2147483647 written by `Context::writeXML()` therefore could not be read back by `Context::loadXML()`, which terminated with `stoi: out of range` naming no file, block or label. `helios::parse_int()`, `helios::parse_float()` and `helios::parse_double()` shared the uncaught-`std::out_of_range` half of the bug and now report an out-of-range value as a failed parse.
- `Context::loadXML()` now fails fast when a primitive names an object ID that no object block in the file declares, naming the ID. Such primitives were added to the Context but omitted from the vector of UUIDs the function returns, and were left with no parent object, so a truncated or hand-edited file produced geometry that rendered and ray-traced but that the caller had no handle on.
- Fixed `Context::loadXML()` recoloring geometry loaded from an earlier file when reading the legacy v2 material format. A v2 `<material id="N">` was mapped onto the Context-global label `__auto_material_N`, but the numeric ID is scoped to the file and every v2 file numbers its materials from 1, so loading a second one found the label already taken and mutated the existing material in place. Colliding legacy IDs are now given a distinct material, as `Context::loadOBJ()` already does for colliding `.mtl` names.
- Fixed `Context::renamePrimitiveData()` and `Context::duplicatePrimitiveData()` leaving the new label unregistered in the Context-wide primitive data type registry. The data was set on the primitive, but the label-only `Context::getPrimitiveDataType()` reported that it did not exist, which broke every caller that resolves a type by label -- `Context::writeXML()` among them, so writing a Context in which an object member's data had been renamed failed with "Primitive data ... does not exist".
- Fixed `Context::copyPrimitive()` giving the copy the source primitive's parent object ID without adding it to that object. The copy claimed a membership the object did not list, so the object neither wrote nor deleted it: deleting the object left the copy behind naming an object that no longer existed, and `Context::writeXML()` then terminated with an uncaught `std::out_of_range`. A copy is now a standalone primitive with no parent object; `Context::copyObject()` is unaffected, as it assigns the parent of each copy itself.
- `Context::writeXML()` now raises a descriptive error when a primitive being written names a parent object that does not exist in the Context, instead of terminating with an uncaught `std::out_of_range` from an unchecked map lookup.

## Radiation

- `RadiationModel::writeImageSegmentationMasks()` now fails fast when the camera's pixel-to-primitive map refers to primitives that no longer exist, instead of writing annotations against it. The map holds Context UUIDs and is published as global data, which `Context::writeXML()` stores verbatim while `Context::loadXML()` assigns every primitive a fresh UUID, so a map arriving with a reloaded scene describes different primitives than the ones rendered and silently mislabeled the output. Re-run the radiation model for the camera after reloading a scene.
- Fixed the Vulkan compute backend delivering zero flux from every radiation source after the first. The shaders declared the source position and source rotation arrays as `vec3 positions[]` / `vec3 rotations[]`, which std430 gives a 16-byte stride, while the host uploads tightly-packed 12-byte `helios::vec3`; element 0 read correctly and every later element read the wrong 12 bytes, with the last reading past the end of the allocation. A scene with two collimated sources absorbed exactly the flux of the first, and the second contributed nothing. Affects any Vulkan-backend scene with more than one source.
- Fixed the Vulkan compute backend sampling the diffuse sky around the wrong peak direction for every radiation band after the first, caused by the same std430 stride mismatch on the per-band `diffuse_peak_dir` array. With a directional sky model (`setDiffuseRadiationExtinctionCoeff()`) and three bands, the absorbed flux in bands 1 and 2 was in error by 21% and 27% respectively. Isotropic-sky runs are unaffected, since the peak direction is not read.
- Non-finite results are no longer returned silently. `VulkanComputeBackend` now checks the radiation, scatter and camera pixel buffers as they are read back and raises `helios_runtime_error` naming the backend, the buffer, the offending primitive and band, and which launch produced it. The previous check tested `radiation_in[0] < 0.0f`, which cannot detect NaN, so a failed trace propagated through the scattering iterations into every camera pixel and surfaced only as a uniformly black image.
- `RadiationCamera::applyCameraExposure()` now rejects non-finite pixel data with an explicit error instead of scaling the whole image by a non-finite gain. Auto-exposure divides its target by `std::max(<statistic>, 1e-6f)`, and `std::max(NaN, 1e-6f)` returns NaN, so one bad pixel turned the entire frame of every band black; sorting a range containing NaN was also undefined behavior.
- Fixed the Vulkan compute backend dropping geometry from the acceleration structure when a BVH leaf held more than 255 primitives. `CWBVH_Node::child_prim_count` was 8 bits, and `recursiveBuild()` creates a leaf of unbounded size whenever it reaches `MAX_DEPTH`, which dense overlapping geometry does: a 300-primitive leaf was traversed as 44 primitives and a 256-primitive leaf as none at all. The count is now 16 bits, occupying the bytes of the former `child_prim_type` array, which nothing read.
- Fixed the Vulkan traversal shader silently discarding BVH nodes when its fixed traversal stack was exhausted, which made geometry invisible to the trace and read as a light leak. The stack is now sized for the deepest tree the builder can emit, and `BVHBuilder::convertToCWBVH()` verifies that bound on the host and raises an error rather than leaving the shader to drop nodes.
- The Vulkan compute backend now raises an error when the scene contains voxel primitives instead of tracing as though they were not there. Its software BVH intersects only patches, tiles and triangles, so voxels occupied acceleration-structure leaves that no ray could hit and neither intercepted nor absorbed radiation. Voxel ray tracing remains available on the OptiX backends.
- Degenerate primitives no longer produce non-finite intersection results on the Vulkan backend. The ray-patch and periodic-boundary face tests normalized the surface normal before testing for degeneracy, so a zero-area primitive produced a NaN normal that then slipped past the parallel-ray guard (`abs(NaN) < eps` is false); the ray-triangle test normalized an unguarded cross product. Guards were also added for the source-distance divisions in the direct shader, the Blinn-Phong half-vector in the camera shader, the circumsolar-width division in the Prague sky model, and `atan(0,0)`, which GLSL leaves undefined and which arises for every perfectly horizontal primitive.
- Fixed textured triangles with no texture coordinates sampling a single texel of their transparency mask on the Vulkan backend, which made the whole triangle either fully solid or fully invisible depending on that one texel. Such triangles now use the default triangle UV mapping, matching how the patch path already treats missing UVs. The ray-origin and ray-hit mask lookups also now share one implementation; they previously rounded differently and disagreed by one texel on each axis.
- The Vulkan compute backend now validates transparency mask and UV data on upload (mask dimensions, mask indices in range, sufficient UV data) and initializes its placeholder texture buffers, which were left holding uninitialized device memory. It also raises an error when a device memory mapping fails or when the device does not support the compute-stage subgroup operations the ray-generation shaders require, rather than continuing with a null pointer, stale results, or unsupported shader capabilities.
- Fixed the Vulkan compute backend building camera images from band-weighted scattered radiance instead of camera-weighted radiance. A camera integrates a surface's scattered radiance against its own spectral response, which is a different integral over wavelength than the band weighting that drives the scattering iterations, so the backend maintains a parallel `rho_cam`/`tau_cam` material set and a parallel pair of camera scatter buffers. The Vulkan backend uploaded neither and aliased `scatter_buff_top_cam` onto `scatter_buff_top`, which is exact only when every camera spectral response is uniform. On a scene combining a spectral surface reflectivity, spectral sources and per-band camera responses, the rendered image was in error by 60-75% per band and the foliage colour ratios by 18-24%; with fewer than all three of those present the aliasing is exact, which is why it went unnoticed. The direct and diffuse shaders now accumulate camera-weighted scatter alongside the band-weighted scatter, against `rho_cam`/`tau_cam` uploaded as material-set bindings 14 and 15.
- Fixed the Vulkan compute backend applying the first radiation source's reflectivity and transmissivity to every source. Radiative properties are per source -- a surface's reflectivity is integrated against each source's own spectrum -- but the direct-ray shader indexed the material buffers as `[primitive][band]`, omitting the leading source dimension that the host writes and that the OptiX backends read. Two sources whose spectra gave reflectivities of 0.12 and 0.20 on the same surface deposited identical flux; the correct ratio is `(1 - rho_1)/(1 - rho_0)`. Only the direct pass is affected: the diffuse and scattering passes carry no source of their own and correctly read source 0.

## Project Builder

- Fixed `ProjectBuilder::ground_objID` being read before it was ever assigned. `BuildGeometry()` is a free function and had no way to report the ground object it created back to the `ProjectBuilder`, so `updateGround()` tested an indeterminate value with `doesObjectExist()` and passed it to `deleteObject()`, which could delete an unrelated object that happened to share the value. The ground object ID is now handed back through global data, as its primitives already were.

## Visualizer

- Rendering now uses a physically-based linear-light pipeline by default: albedo is decoded from sRGB to linear light before shading, and the shaded radiance is tone-mapped through an ACES filmic curve and re-encoded to sRGB. Previously all lighting arithmetic was performed directly on non-linear sRGB values, producing mid-tones that were too dark and abrupt shadow terminators, while bright surfaces clipped flat against the 8-bit framebuffer instead of rolling off. **Rendered images therefore differ from previous versions.** The previous behavior is available with the new `Visualizer::disableLinearPipeline()`, and `Visualizer::enableLinearPipeline()` restores the default. The pipeline is automatically disabled by `Visualizer::enableExactColorMode()`, whose framebuffer carries data rather than an image, and restored by `Visualizer::disableExactColorMode()`.
- Added `Visualizer::setExposure()` and `Visualizer::getExposure()`, controlling the linear exposure multiplier applied before tone mapping. A non-positive exposure now raises an error rather than producing a black or sign-flipped image.
- The shading model for Context primitives now includes a Blinn-Phong specular highlight, which was previously absent entirely despite the model being described as Phong. Its strength and tightness are controlled with the new `Visualizer::setPhongMaterial()` and the `Visualizer::PhongMaterial` struct, which also expose the ambient and diffuse reflectance weights. Surfaces without a highlight can be recovered by setting `specular` to zero.
- Ambient light is now modeled hemispherically, blending a sky color from above with a ground-bounce color from below according to surface orientation, rather than applying a single constant ambient term to every surface regardless of which way it faces. The two colors are set with the new `Visualizer::setAmbientColors()`. Surfaces facing away from the light source are shaded slightly differently than before as a result.
- A single-sided primitive viewed from the side its normal points away from now picks up ambient light from the side being viewed, rather than rendering black. Direct illumination is unaffected by camera position: the diffuse and specular terms and the shadow-map lookup all use the true surface normal, so a leaf turned away from the sun receives no direct light no matter where it is viewed from, while the same leaf viewed from its lit side is fully illuminated.
- Surfaces are now shaded with smooth, interpolated per-vertex normals rather than a single flat normal per face, removing the faceted appearance of tessellated curved geometry such as stems, trunks and fruit. `Visualizer::disableSmoothShading()` selects flat shading and `Visualizer::enableSmoothShading()` restores the default. Only geometry that supplies distinct vertex normals is affected: primitives added without them carry the face normal replicated across their vertices and are unchanged, so patches, triangles and voxels added directly to the Context render exactly as before.
- Vertex normals retained by a `Polymesh` object are now used by the Visualizer, so a mesh loaded from an OBJ or PLY file that supplies them is rendered smooth rather than faceted.
- `Sphere`, `Tube` and `Cone` objects are now rendered smooth rather than faceted, using the analytic surface normal of the shape each one approximates. Stems, branches, trunks and fruit built from these object types previously read as a series of flat panels regardless of how finely they were tessellated. Objects whose faces are genuinely flat, such as a tile or box, are unaffected and render exactly as before.
- Headless rendering is now anti-aliased. Geometry is rendered into a multisampled framebuffer and resolved before readback, using the sample count given to the `Visualizer` constructor; previously the offscreen framebuffer used in headless mode was single-sampled, so every saved image had hard, aliased edges while the equivalent windowed render did not. The requested sample count is clamped to the driver maximum, and `Visualizer::isHeadlessMultisamplingActive()` reports whether multisampled attachments were obtained. Note that macOS drives OpenGL through a translation layer that accepts the attachments but does not rasterize into them, so headless anti-aliasing has no effect there.
- `GL_MULTISAMPLE` is now enabled when a positive anti-aliasing sample count is requested. It was only ever disabled, never enabled, which had no visible effect in windowed mode because the window's own framebuffer is multisampled, but left rendering into any framebuffer created by the Visualizer unantialiased.
- Phong material parameters can now be set per Context material group, rather than only for the whole scene, by attaching `phong_ambient`, `phong_diffuse`, `phong_specular` or `phong_shininess` material data with `Context::setMaterialData()`. Each parameter falls back individually to the value set by `Visualizer::setPhongMaterial()`, so a material can override only its specular lobe, and a material specifying none of them renders exactly as before. Parameters are resolved once per material rather than once per primitive.
- Rendering a scene no longer rebuilds and re-uploads every primitive on every frame. Context dirty flags are only cleared by `Context::markGeometryClean()`, which is the user's call to make once every plug-in has processed a change, so `plotUpdate()`, `plotInteractive()` and `printWindow()` previously treated the entire scene as changed on every call and re-sent it to the GPU one primitive at a time. The cost of a frame is now proportional to what actually changed rather than to the size of the scene: rendering and annotating a 900,000-primitive scene went from roughly 90 seconds to 15. Rendered images and annotations are unchanged.

## Plant Architecture

- Petioles are now built as a single `Tube` compound object, matching internodes and peduncles, rather than as a chain of independent `Cone` objects. Each cone previously chose its own azimuthal orientation, so adjacent segments of a multi-segment petiole met with their vertex rings out of phase and left a visible crease at every joint; the tube's surface is continuous. `Phytomer::petiole_objIDs` is correspondingly now a `std::vector<uint>` holding one object ID per petiole instead of a `std::vector<std::vector<uint>>` of per-segment IDs, with `Phytomer::no_petiole_objID` marking a petiole that has no geometry. Plants whose petioles use a single length segment are visually unchanged.

## Solar Position

- Cloud calibration no longer destroys the PAR/NIR partitioning of the solar flux. `SolarPosition::getSolarFluxPAR()` and `getSolarFluxNIR()` each derived the cloud attenuation factor from their own band, which cancelled algebraically and made both return the same value, equal to the measured horizontal flux divided by the cosine of the solar zenith angle. With calibration enabled the two bands were therefore identical, each was inflated to the full broadband flux, and they summed to twice the value returned by `getSolarFlux()`, so a two-band radiation simulation driven from them received double the intended shortwave energy. The factor is now computed once from the all-wave flux and applied to both bands, preserving their clear-sky ratio. `getDiffuseFraction()` was likewise computing the diffuse fraction from a PAR-only horizontal flux, which biased it upward and could saturate it at 1.
- The diffuse fraction returned by `SolarPosition::getDiffuseFraction()` under cloud calibration is now estimated from the clearness index using the correlation of Erbs et al. (1982). The previous expression reduced to `2 - R_meas/R_clear`, which exceeds 1 whenever the measured flux falls below the clear-sky prediction, so it clamped to a fully-diffuse sky for every real cloud condition and reported overcast conditions for light haze; it varied only when the measurement exceeded the clear-sky value. The diffuse fraction now increases monotonically with cloud thickness.
- Cloud calibration now also applies to the spectral model, i.e. to `SolarPosition::calculateDirectSolarSpectrum()`, `calculateDiffuseSolarSpectrum()` and `calculateGlobalSolarSpectrum()`, which previously ignored `enableCloudCalibration()` entirely and silently returned clear-sky spectra. The modeled spectrum is normalized to the measured broadband flux and corrected for the preferential transmission of short wavelengths through cloud, following Bird, Riordan & Myers (1987); no modification is applied when the modeled and measured broadband fluxes agree to within 5%. The modeled photosynthetically active fraction of the irradiance in the model's 300-2600 nm window consequently rises from about 0.45 under clear sky to about 0.51 under heavy overcast. See the plug-in documentation for the method and its limitations.

# [1.3.82] 2026-08-23

## Core
- Added `annotation_io.h`, a set of utilities for writing machine-learning image annotations in the standard YOLO and COCO formats. These operate on plain binary masks and bounding boxes rather than on any particular camera or renderer, so any plug-in that can produce a per-pixel object label can write annotations in the same formats. The radiation and synthetic annotation plug-ins now share this implementation.
- The `nlohmann/json` header is now bundled with the core library at `core/lib/json/` and is on the include path of every plug-in, rather than being private to the radiation plug-in. Code that includes `json.hpp` is unaffected.
- Fixed `loadXML()` throwing "ERROR (Context::addVoxel): Voxel has size of zero." on any file containing a `<voxel>` element. The loader constructed the placeholder voxel with a size of zero before applying its transformation matrix, so a scene containing voxels could be written by `writeXML()` but never read back. Voxel center and size now survive the round-trip.
- Removed the declaration of `point_distance()` from `global.h`. It was declared but never defined anywhere in the library, so any user who called it got a link error rather than a distance.

## Radiation
- Removed the deprecated overloads of `writeImageBoundingBoxes()` and `writeImageBoundingBoxes_ObjectData()` that took `(..., imagefile_base, image_path, append_label_file, frame)`. They silently captured calls written against the current `(..., image_file, classes_txt_file, image_path)` signature whenever the trailing arguments were string literals, because their by-value `uint` class-ID parameter was a better match than the current overloads' `const uint &`. The class name file was then treated as the output directory and the call failed with "Expected a directory path but got a file path for argument 'image_path'", with no deprecation warning to indicate what had happened.
- Fixed the first two values of each bounding-box annotation written by `writeImageBoundingBoxes()` being formatted differently from the last two, because the fixed-precision format was applied partway through the output expression. The centre coordinates could be written in scientific notation while the width and height were not.
- The COCO and YOLO writing, connected-component labeling and boundary tracing now come from the shared core annotation utilities rather than being private to `RadiationModel`. Output is unchanged.
- Fixed `writeImageSegmentationMasks()` and `writeImageSegmentationMasks_ObjectData()` writing at most one annotation per label value, silently omitting every other separate region carrying that value. Any object split into more than one visible region — most commonly by an occluder in front of it, such as a leaf behind a stem — had all but one of its pieces dropped from the COCO file with no warning or error, so the output looked complete while objects plainly visible in the image were missing. Files regenerated with this version will contain more annotations than before wherever this occurred.
- Fixed connected components in the segmentation mask writers being found with 4-connectivity while their boundaries were traced with 8-connectivity. A diagonally-connected region was therefore split into one component per pixel, which combined with the above to drop it from the output entirely.
- `createWithBackend()`, `exportColorCorrectionMatrixXML()` and `loadColorCorrectionMatrixXML()` are now private. All three were public solely so the test suite could reach them — the first is documented as "INTERNAL TEST-ONLY … not for production use", and the latter two are implementation details of `autoCalibrateCameraImage()` that were marked "public for testing" — so none had any user-facing meaning. The tests reach them through a `RadiationModelTestHelper` friend class defined solely in the plug-in's `selfTest.cpp`, so nothing test-related is exposed to library users.

## LiDAR
- Removed the public `forceBruteForceLeafArea()`. It existed only so the self-tests could A/B the brute-force leaf-area path against the fast DDA path, has no effect on results, and was documented as "not needed in normal use"; the tests now set the underlying private flag through a `LiDARTestHelper` friend class defined in the plug-in's `selfTest.cpp`.

## Visualizer
- Fixed the interactive display functions returning immediately after the first one was used. Closing the window with its close button latches GLFW's should-close flag, which nothing ever cleared, so a program calling `displayImage()`, `displayImageWithBoundingBoxes()` and `displayImageWithSegmentationMasks()` in sequence blocked on the first and then flashed the remaining images past in a single frame each. Dismissing the window with the escape key did not have this effect, so the behavior depended on how the window was closed.
- Added `enableExactColorMode()` and `disableExactColorMode()`, which control whether primitive colors are reproduced exactly in the rendered image. The fragment shader multiplies vertex-interpolated colors by 1.5 to brighten ordinary renders, and did so unconditionally -- including under `LIGHTING_NONE` -- so a color set with `Context::setPrimitiveColor()` never matched the color read back with `getWindowPixelsRGB()`, and any channel above 170 saturated to 255. This only matters when the rendered image is used to carry data rather than to be looked at; the default behavior is unchanged.
- Fixed `clearGeometry()` leaving the GPU-side geometry buffers untouched. Clearing all geometry marks no individual primitive as dirty, and the buffer upload is skipped when nothing is dirty, so the previously uploaded geometry was retained and continued to be rendered.
- Added `displayImageWithSegmentationMasks()`, which displays an image with the segmentation masks from a COCO JSON annotation file overlaid, each drawn as a translucent filled polygon with a solid outline and its class name on a filled chip. This is the format written by `RadiationModel::writeImageSegmentationMasks()`; class names are read from the file's own `categories` array, so no separate class name file is needed, and masks are colored per annotation rather than per class so that touching objects of the same class stay distinguishable. The fill opacity and the class labels can each be turned off with optional arguments, and the fill is computed by an even-odd scanline fill so that it is correct for the self-intersecting contours the writer produces wherever a mask narrows to a one-pixel neck. The new `readSegmentationMaskFile()` exposes the parsing on its own.
- Added `displayImageWithBoundingBoxes()`, which displays an image with the bounding boxes from a YOLO-format annotation file overlaid as colored outlines, each labeled with its class name on a filled chip inside the box's top-left corner. Class names are read from a file in either the `<class_ID> <class_name>` form written by `RadiationModel::writeImageBoundingBoxes()` or the standard one-name-per-line Ultralytics form; when no class file is given, `classes.txt` beside the annotation file is used if present, and boxes are otherwise labeled with their numeric class ID. The new `readBoundingBoxFile()` and `readBoundingBoxClassNames()` expose the parsing on its own.
- Added `getTextboxSize()`, which returns the extent a string would occupy if rendered by `addTextboxByCenter()`, in window-normalized units. `addTextboxByCenter()` centers text and discarded the width it computed internally, so there was previously no way to align text by any edge.
- Fixed `displayImage()` and `clearGeometry()` throwing `std::out_of_range` when called after anything had already been plotted. Both clear all geometry but left the visualizer's cached identifiers for the watermark, background rectangle, background sky, coordinate axes, navigation gizmo and colorbar pointing at geometry that no longer existed; deleting by one of those afterwards looked up a map key that had been erased. The sequence `plotUpdate(); displayImage(file);` aborted instead of displaying the image.
- Corrected the documentation of `COORDINATES_WINDOW_NORMALIZED`, which stated that an object at z=0.5 would be in front of an object at z=0. Smaller z is nearer the viewer, as the plug-in documentation already stated.
- `createShadowFramebuffer()`, `setupOffscreenFramebuffer()`, `cleanupOffscreenFramebuffer()` and `renderToOffscreenBuffer()` are now private. They are internal framebuffer lifecycle machinery driven by headless/offscreen rendering, with no user-facing meaning; the first three had no caller outside the plug-in at all, and the tests reach the last through the existing `VisualizerTestHelper` friend class.
- Fixed `Visualizer` construction changing the working directory of the host process on macOS. `Visualizer::initialize()` calls `glfwInit()`, whose `GLFW_COCOA_CHDIR_RESOURCES` init hint defaults to enabled and moves the process to the `Contents/Resources` directory of the host application's bundle; the hint is now disabled beforehand. A bare Helios executable has no bundle and was unaffected, but a `.app`-packaged host — most commonly a macOS framework-build Python interpreter, whose real executable lives inside `Python.app` — had its working directory silently relocated, after which construction failed with `ERROR (Context::readJPEG): File plugins/visualizer/textures/gradient_background.jpg could not be opened` for a file that exists and is readable, and every subsequent relative path in the host application resolved against the wrong directory.
- The six remaining visualizer textures loaded by a path relative to the process working directory — the gradient and transparent backgrounds used by `setBackgroundGradient()`, `setBackgroundTransparent()` and `printWindow()`, and the three navigation gizmo bubbles — now resolve through `helios::resolvePluginAsset()`, as the shaders, fonts, skydome and watermark already did. Any host that had changed the working directory before using the visualizer, for any reason, previously failed to load them.

## Plant Architecture
- Fixed `pruneBranch()` failing partway through when pruning a whole branch system. Pruning a shoot also empties every shoot descending from it, so a loop over shoot IDs reaches those descendants again later; `pruneBranch()` rejected them with `Node index 0 is out of range for shoot N`, aborting the loop and leaving the plant half-pruned. Pruning a shoot that has already been pruned away is now a no-op, while a node index that is genuinely out of range on a live shoot is still an error.
- Fixed `advanceTime()` segfaulting on a plant containing a pruned branch when internode context geometry was disabled with `disableInternodeContextBuild()`. A pruned shoot's apical meristem was killed only as a side effect of deleting its internode tube object, so with no tube objects to delete the emptied shoot was still treated as growable and `advanceTime()` dereferenced its (now non-existent) last phytomer. `pruneBranch()` now terminates the apical bud of any shoot it empties.
- Fixed `getShootTaper()` segfaulting when called on a pruned shoot, which has no internode geometry to read a radius from. It now throws an error naming the shoot and pointing at the new `isShootPruned()`.
- Fixed `Shoot::sumShootLeafArea()` and `Shoot::sumChildVolume()` throwing `Start node index out of range` for a pruned shoot, which has no nodes for index 0 to be in range of. Both now return zero. This also affected live shoots: because a pruned shoot was left in its parent's child list, summing the leaf area of an ancestor descended into the pruned shoot and threw, so the leaf area of any branch containing a pruned shoot could not be obtained at all.
- A shoot pruned away by `pruneBranch()` is now unlinked from its parent's child list and its deleted internode tube object ID is reset to the sentinel, matching `pruneGroundCollisions()`. Previously the shoot remained reachable as a child of a live shoot despite having no phytomers or geometry, so recursive traversals of the plant structure descended into it, and it retained the object ID of a tube object that had been deleted from the Context.
- Added `PlantArchitecture::isShootPruned()` and `Shoot::isPruned()`, which report whether a shoot has been pruned away entirely. A pruned shoot keeps its slot in the plant's shoot tree so that shoot IDs are not renumbered by a prune, so its ID stays valid and is still returned by `getAllShootIDs()` even though the shoot no longer forms part of the plant; there was previously no way to tell the two apart.
- Added `setPlantMaxAge()` and `getPlantMaxAge()` to set and query the age at which a plant stops growing. `PlantInstance::max_age` defaults to 999 days and every library builder overrides it, but a plant assembled through the manual API (`addPlantInstance()` with `addBaseStemShoot()`/`appendShoot()`/`addChildShoot()`) kept the default and so silently stopped growing after roughly 1000 days, with no message and no public way to raise the limit.
- Maize ear position is now defined relative to the tassel rather than at fixed node indices, and a maize plant now bears one ear by default instead of three. `MaizePhytomerCreationFunction()` placed ears at absolute nodes 9-11, which only approximated the biology at the default `max_nodes` of 17; raising `max_nodes` left the ears stranded low in the canopy, so a taller plant produced fewer ears than a shorter one. The ear now forms at `max_nodes - 6` (node 14 of the new 20-node default, node 19 of 25), matching the observation that ear meristems form at every node except the upper six to eight and only the uppermost develops. A maize plant now bears 8 fruit objects and 2 peduncles where it previously bore 10 and 4.
- Maize main-stem node count raised from 17 to 20 and the phyllochron from 2 to 3 days per node. Corn Belt grain hybrids carry 19-21 main-stem leaves (range 16-23) and add a leaf collar roughly every 3 days over most of vegetative development; the previous values produced a 17-leaf plant that finished growing about twice as fast as a field crop. Tasseling now occurs near 60 days after emergence rather than 32, within the reported 54-63 day range. A default maize plant is correspondingly taller and takes longer to reach its final size, so simulations that advanced maize by a fixed number of days may now see a less mature plant than before.
- Maize `time_to_fruit_maturity` raised from 10 to 58 days. Grain fill from silking to physiological maturity takes 55-65 days in the field, so ears previously reached full size roughly five times too quickly. This value is the divisor controlling how fast the ear scales up, and it also sets the sink duration seen by the carbohydrate and nitrogen models.
- Fixed the maize mainstem ignoring `ShootParameters::internode_length_max`. `buildMaizePlant()` passed a hardcoded 0.08 m to `addBaseStemShoot()`, which takes precedence over the value stored on the shoot type, so the 0.22 m set in `initializeMaizeShoots()` was never used and editing it had no effect on the plant. The builder now forwards the shoot-type value, as the bougainvillea and capsicum builders already did. A default maize plant is now about 2.2 m tall rather than 1.4 m, within the 2-3 m range of a field hybrid.
- Sorghum phyllochron raised from 2 to 3.5 days per node and `time_to_fruit_maturity` from 15 to 35 days. A grain sorghum plant reaches half-bloom about 60 days after emergence and fills grain over roughly 35 days (Vanderlip, Kansas State S-3; Gerik et al., Texas A&M B-6137); the previous values completed the plant by day 30 and filled grain in 15. The main-stem node count of 16 is unchanged, being already within the 15-19 leaf range spanned by early to late maturity classes, as is plant height, which was already correct for a dwarfed grain hybrid.
- Fixed the sorghum mainstem ignoring `ShootParameters::internode_length_max`, the same defect as maize: `buildSorghumPlant()` passed a hardcoded 0.06 m to `addBaseStemShoot()` in preference to the stored value. The builder now forwards the shoot-type value. Note the stored value was simultaneously corrected from 0.26 m to the 0.06 m that was actually in force, so sorghum geometry is unchanged by this fix; the previously inert 0.26 m would have produced a roughly 5.5 m plant, far outside the 0.6-1.5 m range of a grain hybrid.
- Corrected the plant phenology table in the Plant Architecture documentation, which had the `time_to_fruit_maturity` values for "rice" and "sorghum" transposed (listing 15 and 10 where the library uses 10 and 15).
- `writePlantStructureXML()` now writes the plant's maximum age as a `<max_age>` tag and `readPlantStructureXML()` restores it. It was previously not persisted, so any plant written and read back silently reverted to the default of 999 days regardless of the value its builder or `setPlantMaxAge()` had set. The tag is optional on read, so files written by earlier versions still load.
- Fixed `duplicatePlantInstance()` producing a plant that differed from the one it copied. It rebuilt the shoot structure but carried over only the shoot-parameter snapshot, so the duplicate silently fell back to the `PlantInstance` defaults for everything not derivable from that structure: a copy of an apple tree reported its name as "custom", took a 999-day maximum age instead of 1460, and lost its phenological thresholds, epicormic-shoot probability, and carbohydrate and nitrogen parameters — so it entered dormancy, flowered and fruited on a different schedule from the original. All of these are now copied.
- Fixed `duplicatePlantInstance()` attaching child shoots at the wrong node. It passed the node at which the *parent* shoot joins the grandparent where the attachment node of the shoot being copied was required, so on any plant whose branches leave the trunk at different heights every branch was relocated onto a single wrong node and the copy's architecture did not match the original's.
- Fixed `duplicatePlantInstance()` returning a dormant copy of an actively-growing plant. Shoots are constructed dormant, and the duplicate's dormancy was never reconciled with its source, so the copy stalled until its dormancy broke while the plant it was copied from kept growing. Dormancy state and time since dormancy are now matched to the source.
- `duplicatePlantInstance()` now translates per-plant attraction points onto the duplicate's base position. They are absolute world coordinates, so the trellis of a trained plant (for example the apple fruiting wall) was previously not carried over at all; copying them unchanged would instead have steered the duplicate's growth toward the original plant's trellis.
- Removed `setPlantAge()`. Its body was commented out, so it silently did nothing when called, and it was documented as "Don't use this". `getPlantAge()` is unaffected.
- Fixed `disablePlantPhenology()` corrupting fruit geometry without bound. It set `dd_to_fruit_maturity` to `-1`, matching the flower and fruit-set stages beside it, but that field is not a "skip this stage" sentinel — it is used only as a divisor during fruit growth, so the scale factor became `0.25 - 0.75 * time_counter` and went negative within the first day. A debug build then aborted on the assertion in `Phytomer::setInflorescenceScaleFraction()`, while a release build mirror-scaled the fruit by a negative factor and compounded the error on every subsequent step: a sorghum panicle reached roughly 550 m², against a mature size of 0.16 m², after 100 days. It is now `1e6`, matching the `PlantInstance` default, so fruit simply never matures. The fruit-growth block in `advanceTime()` is additionally guarded against a non-positive `dd_to_fruit_maturity`, which `setPlantPhenologicalThresholds()` accepts and `readPlantStructureXML()` restores without validation. Only plants with a fruiting bud were affected, which requires a model defining a fruit prototype.
- Added `ShootParameters::inheritCustomFunctionsFrom()`, which copies the five user-defined function pointers held in a shoot type's phytomer parameters (the phytomer creation and callback hooks, and the leaf, flower, and fruit prototype functions) from another `ShootParameters`. `defineShootType()` and `updateCurrentShootParameters()` replace a shoot type entry in its entirety, which is harmless when the structure was copied but strips the species' customizations when it was reconstructed from plain values, as a scripting-language binding that serializes the parameters must do. For maize, committing such a reconstruction previously produced a tassel at nearly every node and no ears.
- Corrected the documentation example for modifying the parameters of all shoot types, which assigned the result of `getCurrentShootParameters(shoot_type_label)` to a `std::map` and so did not compile. It now calls the no-argument overload that returns the map.
- Fixed `ShootParameters::operator=` not copying `elongation_rate_max`. It was the only member missing from the assignment operator, so assigning a `ShootParameters` silently reset the shoot's maximum elongation rate to the default of 0.2, discarding the value set by the library or the user. Every library model sets a different value, so any path that assigned a `ShootParameters` — including reading a shoot type with `getCurrentShootParameters()` and writing it back with `updateCurrentShootParameters()` — lost it and the plant grew at the wrong rate.
- `defineShootType()` and `updateCurrentShootParameters()` now store the shoot-level parameter values they are given instead of re-drawing the random ones. `ShootParameters::operator=` resampled every distributed shoot-level parameter, while defining a shoot type under a label that did not yet exist did not, so the same input produced different stored values depending only on whether the label already existed, and reading a shoot type and writing it straight back silently perturbed parameters the caller never edited. Per-shoot variation is unaffected, as it comes from the growth engine resampling when each shoot is created. The phytomer parameters nested inside `ShootParameters` still resample on assignment, which is what varies phytomers along a shoot.

## LeafOptics
- `PROSPECT()` now validates its inputs, matching `getLeafSpectra()`. It is a spectrum-computing entry point in its own right but bypassed `validateProperties()` entirely, so calling it directly with a structure parameter of `numberlayers = 0` divided by zero when computing the mean absorption coefficient, and negative constituent contents contributed negative absorption — both returning non-finite or unphysical spectra with no diagnostic instead of the documented runtime error.

## Energy Balance
- Fixed primitive data `surface_humidity` being applied to the air vapor pressure instead of the surface vapor pressure in the latent heat flux. The term was computed as `e_s(T_s) - e_s(T_a)*h*f_s` rather than `e_s(T_s)*f_s - e_s(T_a)*h`, so lowering `surface_humidity` below 1 to represent a drying surface increased evaporation instead of suppressing it, and could flip the sign of `latent_flux`. Results are unchanged at the default `surface_humidity` of 1, which is why this went unnoticed; any simulation that set it below 1 had incorrect latent and sensible fluxes and surface temperatures. Both the CPU and GPU paths were affected.
- Fixed the reported `latent_flux` primitive data omitting `surface_humidity` entirely, so the value written out disagreed with the flux the energy balance actually solved whenever `surface_humidity` was not 1.

## Synthetic Annotation
- `render()` now writes bounding boxes in the YOLO annotation format, as a single file per view named after the rendered image, with every class in the one file and a `classes.txt` beside it. This replaces the previous `rectangular_labels_<label>.txt` files, which held one label per file in a format no training pipeline reads.
- `render()` now writes instance segmentation masks as a COCO JSON file per view (`instances.json`). This replaces the previous per-object `instance_segmentation_<label>_<object>.txt` pixel grids.
- Both new outputs are in the formats read by `Visualizer::displayImageWithBoundingBoxes()` and `Visualizer::displayImageWithSegmentationMasks()`, so annotations can be displayed over the rendered image directly, and match what the radiation plug-in writes for its cameras.
- The instance masks now describe each object's **visible** extent rather than its full un-occluded extent, which is the convention the COCO format uses. As a result the masks no longer require a separate rendering pass per object: rendering a scene with many labeled objects is substantially faster, and the cost no longer grows with the number of objects.
- Fixed the plug-in failing to configure unless the visualizer plug-in also happened to be listed in the project's `PLUGINS`. It declared a build dependency on `visualizer` without causing it to be loaded, so building with only `syntheticannotation` selected aborted at the CMake configure step with "The dependency target visualizer of target syntheticannotation does not exist". The visualizer is now loaded automatically when it is not already present.
- Fixed the `object_detection`, `semantic_segmentation` and `instance_segmentation` flags read from a loaded XML file being inverted in `render()`. The values were tested with `std::string::compare()` as though it returned a boolean, and it returns 0 on a match, so specifying "enabled" turned the output off and "disabled" turned it on. An unrecognized value is now a runtime error rather than being silently ignored.
- Fixed `render()` writing every object of a label group to the same instance segmentation file. The output filename omitted the object counter, so the field width intended for it padded the ".txt" extension instead and each object overwrote its predecessor, leaving only the last object of each label on disk.
- Fixed the label ID codes recovered from the rendered image being wrong, which made every annotation the plug-in produced meaningless. IDs are encoded as RGB colors and decoded back from the rendered pixels, but the visualizer brightened those colors by a factor of 1.5, so an object labeled 1 was recovered as 2 and IDs whose color channels exceeded 170 saturated and became indistinguishable from one another. `render()` now renders the ID pass with `Visualizer::enableExactColorMode()`. The semantic segmentation mask was the most visible symptom: no decoded ID ever matched a label group, so the mask came out uniformly blank.
- Fixed each per-object instance segmentation mask containing the *previous* object rather than its own. The ID pass was rendered to a window, and the windowed pixel readback picks between the front and back buffer by sampling nine pixels and taking whichever has more non-black content; against the white background used for the ID pass both score identically and the stale buffer wins. The ID pass now renders headless, to an offscreen framebuffer that is read back directly.
- Implemented `labelUnlabeledPrimitives()`, which was an empty stub that silently did nothing. Every primitive not already labeled is now assigned the given label as its own object, so primitives the user expected to be annotated are no longer omitted from the output.
- Implemented `addSkyDome()`, which was an empty stub. The sky dome texture used as the background of the RGB rendering is now configurable rather than hardcoded, and a path that does not exist is a runtime error rather than being silently ignored.
- Added `setMinimumLabelPixels()` to set how many pixels an object must cover to be written to the annotations, replacing a hardcoded value of 10 that could not be changed. Values below 3 are rejected, since an object covering fewer pixels has no traceable outline and would be dropped from the segmentation masks without a diagnostic.
- Added `disableMessages()` and `enableMessages()`. The plug-in printed progress messages unconditionally with no way to silence them, including two debug lines that printed a label name for every label on every view.
- Fixed `ID_mapping.txt` being written as a zero-byte file, which left the integer object IDs appearing in `pixelID_combined.txt` and the instance masks impossible to trace back to a label. It now contains one row per object giving its object ID, label, and class index. A `classes.txt` listing class names in class-index order is also written.
- Fixed the bounding-box annotations recording a class index of 0 for every object regardless of its label, which made the output unusable for training a multi-class detector. Each label now writes its own class index.
- Fixed `render()` leaving the Context modified when it threw. Primitive colors are overwritten with label ID codes during rendering and were only restored at the end of the function, so an error partway through left the caller's entire scene colored in ID codes. Colors are now restored on every exit path, and the internal `object_label` primitive data the plug-in adds is removed rather than left behind.
- Removed a hardcoded 5-second delay per camera view in `render()`. Rendering six views took 39 seconds, of which 30 were spent sleeping; the same run now takes 8 seconds and produces byte-identical images.
- Fixed the bounding-box annotations measuring `y_center` from the bottom of the image. The YOLO format requires the origin at the top-left, so every box was vertically mirrored with respect to the rendered image it describes; an object in the upper half of the frame was annotated as being in the lower half. `getGroupRectangularBBox()` now reports the box in top-down image coordinates.
- Fixed `pixelID_combined.txt` being written vertically mirrored relative to `RGB_rendering.jpeg` and the segmentation masks, so that a pixel at a given row in one file did not correspond to the same row in the others.
- Fixed the instance segmentation masks, where the bounding box on the first line was measured bottom-up while the mask below it was written top-down, so the two did not describe the same rows of the image.
- Fixed the first two values of each bounding-box annotation being written in a different numeric format from the last two, because the fixed-precision format was applied partway through the output expression.
- The header of `semantic_segmentation_ID_mapping.txt` is now `label class_ID`, matching the order of the columns beneath it. It previously read `Element Label`, which named the columns in the opposite order to the data.
- Fixed the pixel indexing used to write the semantic and instance segmentation masks in `render()`. The offset was computed such that the first row read addressed a full row past the end of the pixel buffer while the last row was never read at all, so the written masks were shifted by one row and their first row was filled from out-of-bounds memory.

# [1.3.81] 2026-08-13

## Visualizer
- Fixed text rendering grainy and blocky on high-DPI (Retina) displays, most visibly in the colorbar title and tick labels. Glyphs were rasterized at `fontsize` pixels and converted to normalized coordinates using the window dimensions, but every `glViewport()` call site renders into the framebuffer — twice the window size per axis on such a display, so a 12-pixel glyph bitmap was stretched over roughly 24 physical pixels. Glyphs are now rasterized at `fontsize * getDPIScale()` and measured against the framebuffer dimensions; their on-screen size is unchanged.
- Fixed `windowResizeCallback()` corrupting the stored framebuffer size on every resize. On seeing the framebuffer and window sizes differ it called `glfwSetWindowSize()` to force them equal — a no-op, since their ratio is the monitor's content scale — and then assigned the window size as though that had worked, halving the recorded framebuffer size per axis. Because the viewport is set from those members, the scene had been rendering into a quarter of the drawable area after any resize. The size is now queried and trusted, guarded against the zero size a minimized window reports, which also removes a race with `framebufferResizeCallback()`.
- Fixed the glyph fragment shader discarding the antialiasing FreeType had already computed. It applied `smoothstep()` with an `fwidth()`-derived width to what is an 8-bit coverage mask rather than a signed distance field, which under `GL_NEAREST` degenerates into a hard binary threshold, and it premultiplied the color by an alpha the blend function applies again, darkening glyph edges. Coverage is now used directly as alpha. Glyphs are additionally drawn through a sampler object applying `GL_LINEAR` and carry a one-texel transparent border; image textures keep `GL_NEAREST`, which their alpha-mask cutouts require.
- Fixed the watermark rendering with jagged edges, both around its outer border and along the black-on-white boundary inside it. The cause was an alpha test rather than filtering: `Helios_watermark.png` contains no fully opaque pixels — 72% of it sits at exactly `alpha=128`, half of one 8-bit step above the `textureFlag==1` shader branch's 0.5 discard threshold — so every antialiased edge pixel fell below it and was discarded outright. A new `textureFlag==4` blends the texture's alpha rather than testing it, selected by a new private `addAlphaBlendedRectangleByCenter()`, and the watermark is drawn after the opaque batch with depth writes disabled and with linear filtering. The alpha test on `textureFlag==1` is unchanged: it is what turns a leaf or bark texture's alpha channel into a cutout, and blending it would give every leaf a translucent fringe.
- Texture array layers are now cleared to fully transparent when allocated. A texture smaller than its layer left the remainder undefined, which nearest-neighbour sampling never reads but a linear filter does; this is what allows linear filtering to be enabled for any texture in the array at all.
- Reduced the visualizer's texture memory by roughly 24x. Every layer of the shared texture array was allocated at the hardcoded 2048x2048 `maximum_texture_size` regardless of contents, and because `addTextboxByCenter()` registers one texture per letter, a 7x9-texel glyph received a 16 MB RGBA8 layer — a colorbar with a title and tick labels cost on the order of 480 MB of VRAM, reallocated from scratch on every refresh. `maximum_texture_size` is a clamp applied to oversized images as they load, not a statement of what must be allocated; layers are now sized to the largest texture actually present, and the shader's UV rescale factors are computed against the size actually allocated. The standing `\todo` to bake a whole string into one texture remains the right long-term fix.
- Fixed three defects in `addTextboxByCenter()`: text was measured for centering by summing `bitmap.width` while the layout pass advanced by `advance.x`, so every string was centered slightly off; the subscript/superscript baseline shift scaled the glyph's ascent rather than translating it and was expressed as a fraction of the window rather than of the em size, so it changed with both window size and DPI scale; and the scaled bearings omitted the sub/superscript scale factor. `getTextureResolution()` now raises `helios_runtime_error()` for an unknown texture ID rather than falling through an empty branch to a bare `std::out_of_range`.
- Fixed a heap-buffer-overflow that made the visualizer self-test crash intermittently on macOS, in roughly 40% of full-suite runs. `getWindowPixelsRGB(uint*)` writes one entry per framebuffer subpixel, but three sites in the test suite sized their buffer from the window dimensions passed to the `Visualizer` constructor; on a high-DPI display that is a quarter of what is needed, and the resulting 1.4 MB overwrite corrupted the Objective-C class hash table, so the process died later inside an unrelated test. No production code was affected — every non-test caller already sizes from `getFramebufferSize()` — and it was invisible in CI because headless mode makes the two sizes equal. The docstring, which stated the required size as "3*width*height" without saying which width, now says explicitly that the dimensions come from `getFramebufferSize()`, and a new `getWindowPixelsRGB(std::vector<uint>&, uint&, uint&)` overload sizes the buffer for the caller.
- Fixed three colorbar range defects: an explicit `setColorbarRange(0, 0)` was treated as "no range was set" and silently replaced by the data's own range, since intent was inferred from `colorbar_min == 0 && colorbar_max == 0` rather than tracked in a flag; the auto-range scan's empty-data guard tested its `FLT_MAX`/`lowest()` sentinels with `std::isinf()`, which is never true, so a scan matching no data published the inverted range `[FLT_MAX, -FLT_MAX]` into tick generation and the value-to-position mapping; and `addColorbarByCenter()` recorded only the upper of the two tick marks it draws per interior tick, so colorbar geometry accumulated on every refresh for the lifetime of the visualizer.
- Fixed an auto-ranged colorbar frequently rendering with only a single tick label, leaving no sense of scale. `generateNiceTicks()` deliberately extends the tick range outward past the data and guarantees at least two ticks, but `addColorbarByCenter()` then deleted every tick outside `[cmin, cmax]` — exactly the endpoints just added — with nothing rechecking the count. Ticks are now confined to the colorbar range as they are generated, by the new `generateColorbarTicks()`, which steps to progressively finer nice spacings until at least two multiples fall inside the range and falls back to the range endpoints where no nice grid fits. `max_ticks` was also computed from `colorbar_size.y`, the bar's thickness, rather than `size.x`, along which the labels are laid out, so it never adapted to the bar's width and sat at its floor of 3 — the target at which the collapse fires most often.
- `tick_spacing` is now the spacing actually used to generate the ticks, reported through an out-parameter, rather than being re-derived from the surviving ticks, where zero or one left it silently at its default of `1.0`. `formatTickLabel()` chose decimal places from the spacing alone, which made labels state values different from the ticks they marked — a tick of `2.5` printed as `"2"` at a spacing of 1.0 — so precision is now increased until the label round-trips to the tick value, and a spacing of zero no longer reaches `std::log10()` and casts an infinity to `int`.
- `setColorbarTicks()` now warns when it widens the colorbar range. The behavior is documented and intended, but it moves the colormap limits as well as the labels, so a caller who set an explicit range and then supplied a tick slightly outside it had their range rewritten with no diagnostic; calling `setColorbarRange()` afterwards restores an explicit range. Separately, `setColorbarRange()` gated its own validation on the messaging flag — the guard read `message_flag && cmin > cmax` — so with messages disabled, the default, an inverted range was silently accepted and stored. The range is now always rejected and only the warning depends on `message_flag`.
- `niceNumber()`, `formatTickLabel()` and `generateNiceTicks()` are now private, along with the new `generateColorbarTicks()`. They were public members of `Visualizer` but undocumented internal helpers with no user-facing meaning that nothing outside `addColorbarByCenter()` called; the unit tests reach them through a `VisualizerTestHelper` friend class defined solely in the plug-in's `selfTest.cpp`, so nothing test-related is exposed to library users.
- Fixed `plotOnce()` crashing with SIGSEGV on a freshly constructed `Visualizer`, in both windowed and headless mode, with no geometry required to reproduce. It was the only render entry point that never uploaded the CPU-side geometry before drawing — it called `transferBufferData()` only after rendering, inside a nested branch — so with the default gradient background that `initialize()` installs, `render()` issued a `glMultiDrawArrays()` against the zero-byte vertex buffers left by `glGenBuffers()`. Any caller that added geometry and then called `plotOnce()` without an intervening `plotUpdate()` hit the same fault; the `setBackgroundColor()` workaround being used downstream worked only by removing the offending rectangle. `plotOnce()` now transfers pending geometry first, while still not re-walking the Context, since it is the cheap re-render path.
- `plotOnce()` also now branches on headless mode as `plotUpdate()` does. It bound the default framebuffer rather than `offscreenFramebufferID` and unconditionally overwrote `Wframebuffer`/`Hframebuffer` from `glfwGetFramebufferSize()`, which in headless mode queries the 1x1 dummy window backing the offscreen context and thereby corrupted every subsequent `getWindowPixelsRGB()`, `printWindow()` and depth-buffer call.
- `transferBufferData()` now checks for texture-array recovery before its early return. The `texArray == 0` repair path sat after the return taken whenever no geometry was dirty, making it unreachable in the one situation it exists for; the expensive per-buffer and per-UUID geometry uploads are still skipped as before.

## Context
- Added adaptive-resolution tile objects, via a new `addAdaptiveTileObject()` with the same four overloads as `addTileObject()` (plain, color, texture, texture with repeat count), a new `OBJECT_TYPE_ADAPTIVE_TILE`, and an `AdaptiveTile` compound object class. Sub-patches are generated by recursive quadtree subdivision so that they are finest at a user-specified target point and grow progressively coarser away from it. This addresses the tradeoff a ground plane forces today: resolving the shadows an object casts requires centimeter-scale sub-patches, but the rest of the domain only needs to occlude the horizon, so a uniform tile must either run the fine resolution out to the domain edge or be hand-nested from several tiles of differing resolution. Refinement is specified through the new `AdaptiveTileRefinement` struct — target point, minimum and maximum sub-patch size, and a transition exponent — rather than a subdivision count. Every field documents its own default and useful range, since the transition exponent in particular has no intuitive units and changes only the sub-patch count, not the achieved resolution; it defaults to 0.35, which suits a typical ground plane.
- Fixed `setTileObjectSubdivisionCount()` silently resetting a tile object's texture repeat to 1x1, so that a ground tile built with `texture_repeat = (5,5)` rendered as a single stretched copy of the image after any subdivision change. `regenerateTileObjectSubpatches()` rebuilds the sub-patches by creating a canonical unit-tile template and copying its primitives, and it built that template with the five-argument `addTileObject()` overload, which delegates with a repeat of 1x1. The repeat could not be forwarded because nothing retained it: `Tile` stored only the subdivision count, and `texture_repeat` was consumed by `addTileObject()` and discarded. A `texture_repeat` member has been added to `Tile` and is now threaded through the constructor, both `addTileObject()` overloads, `copyObject()` and the XML reader/writer.
- Tile objects now write an optional `<texture_repeat>` element to XML. Without it the fix would be half a fix: sub-patch texture coordinates already round-tripped, so a reloaded scene rendered correctly, but the reader recreated the tile with an implicit 1x1 repeat and the defect returned in full on the next subdivision change. The element is emitted only for textured tiles whose repeat is not 1x1, so files containing no tiled textures are unchanged, and its absence is treated as the 1x1 default without a warning — every file written before it existed lacks it. Older builds ignore it, since the tile reader queries named children rather than validating against a schema.
- Removed the OBJ performance benchmarks from the core self-tests. `Test_OBJ.h` asserted throughput floors on OBJ loading and writing — `triangles_per_second > 1000`, `prims_per_sec > 100`, a 10-second wall-clock cap and five others — which measure the speed of whatever machine runs the suite rather than the correctness of the code. The loading floor was failing on a developer machine at 243 triangles/second under full-suite load while passing at 964 in isolation, against a threshold of 1000: a real failure signal for a defect that does not exist. Two whole test cases ("OBJ File I/O - Performance Benchmarking" and "OBJ WriteOBJ - Performance Benchmarking Suite") were purely timing assertions and are gone; the correctness checks they carried duplicate the loading tests already present. The one subcase worth keeping from the second, which validates written file size rather than speed, is retained under the renamed "OBJ WriteOBJ - Output Size Validation". The timing assertion inside "OBJ WriteOBJ - Stress Testing and Edge Cases" was removed while keeping the case, which still verifies that a 10,000-primitive write succeeds. A stray timing measurement in "Load complex large model" that was recorded and then never asserted on, along with the now-unused `<chrono>` include, are also gone. Benchmarking belongs in `benchmarks/`, and the file header now says so to keep the pattern from returning.

## Plant Architecture
- Fixed `advanceTime()` destroying every leaf and petiole on a manually-built plant. `PlantInstance` declared all six phenological thresholds with a default of `0`, so for any plant that never had `setPlantPhenologicalThresholds()` called on it the dormancy check in `advanceTime()` read `time_since_dormancy > 0` — satisfied on the very first timestep, and on every step after, since the branch resets `time_since_dormancy` to `0`. Each firing called `Shoot::makeDormant()`, which removes every leaf and marks all non-dormant buds `BUD_DEAD`; because `breakDormancy()` only revives buds that are not `BUD_DEAD`, the plant was permanently defoliated and sterilized rather than merely losing a season's growth. The plant shrank instead of growing, with no error or warning. `disablePlantPhenology()` has always used `dd_to_dormancy = 1e6` rather than `0`, which is good evidence `0` was never intended as a "no dormancy" sentinel.
- The defaults now encode "no phenology scheduled", mirroring `disablePlantPhenology()`: `dd_to_dormancy = 1e6`, the flower and fruit-set stages `-1` (the plug-in's existing "skip this stage" sentinel, tested with `>= 0.f`), and `dd_to_dormancy_break` unchanged at `0` so a shoot restored in a dormant state can still break dormancy. `dd_to_fruit_maturity` is deliberately `1e6` rather than `-1`: unlike the other stages it is used as a divisor during fruit growth, and `appendPhytomerToShoot()` can set a bud `BUD_FRUITING` from shoot structure alone without consulting `dd_to_fruit_set`, so a negative value would reach that divisor and trip the assertion in `Phytomer::setInflorescenceScaleFraction()`. The dormancy check itself is additionally guarded so that a non-positive dormancy period cannot trigger a cycle, which makes the failure unreachable even if the defaults are bypassed.
- Phenological thresholds are now written to and read from the plant structure XML. They are not derivable from the shoot structure and `readPlantStructureXML()` never set them, so a restored plant silently fell back to the defaults and hit the identical defect — a gap the test suite had documented in a comment but left unfixed. The new tags are optional on read, so XML files written before they existed still load and now grow correctly rather than defoliating.
- Corrected 28 `setPlantPhenologicalThresholds()` call sites in the plant library that passed a literal `false` into the eighth parameter, `float max_leaf_lifespan`, rather than the ninth, `bool is_evergreen`. This was harmless only by accident — `false` converts to `0.0f`, which the function remaps to `1e6`, exactly the value the parameter defaults to — but it read as a defect to anyone auditing the file. The two correct call sites (olive, the only evergreen, and cherry tomato) are unchanged.

## Parameter Optimization
- CMA-ES and Bayesian Optimization now reject INTEGER and CATEGORICAL parameters instead of silently returning a meaningless result. Both are continuous optimizers with no representation for integer rounding or category lookup — neither function referenced `ParameterType`, `.type` or `categories` anywhere in its body — so they treated every parameter as a real number bounded by `min`/`max`. The `validateContinuousParameters()` guard that enforces the FLOAT-only requirement already existed and was already applied to L-BFGS, Adam, BOBYQA and SLSQP; it was simply never called from these two. The documentation has been unambiguous throughout that discrete types are GA-only.

# [1.3.80] 2026-08-06

## Visualizer
- Fixed shadows being silently absent from every headless render. `initialize()` created the shadow-map framebuffer and its depth texture inside an `if (!headless)` guard, so in headless mode `framebufferID` and `depthTexture` both stayed 0 — but the shadow pass in `plotUpdate()` is gated only on the lighting model, never on `headless`. With `LIGHTING_PHONG_SHADOWED` it therefore bound framebuffer 0, rendering the shadow depth pass into the default framebuffer with an 8192x8192 viewport, and bound texture 0 as the shadow map. Sampling an unbound texture returns 0.0, and the depth comparison in `primaryShader.frag` is `0.0 < proj.z`, which holds for essentially all geometry — so all four Poisson taps reported "occluded" and the entire scene rendered at the fully-shadowed brightness rather than merely losing its shadows. This is the other half of the v1.3.73 fix: zero-initializing those handles stopped the macOS crash from binding a stale GL name, but never made headless mode create the framebuffer. Reported with a reproducer by Heesup Yun (@heesup, GitHub #100), who also contributed the original patch.
- The shadow-map framebuffer is now created lazily, on the first render that actually uses shadowed lighting, in both windowed and headless modes. Creating it unconditionally at initialization would have imposed the 8192x8192 depth texture — 256 MB — on every `Visualizer` instance, including the many that never enable shadows. Its size is now also clamped to `GL_MAX_TEXTURE_SIZE`; nothing previously validated `shadow_buffer_size` against the driver limit, even though `initialize()` already anticipates contexts reporting a maximum texture size below 1024.
- The destructor released the shadow framebuffer and depth texture only in its windowed branch, so once headless mode could allocate them they would have leaked on every instance. Both are now released regardless of mode, guarded on being non-zero so the lazy path is safe.
- The initial light direction is now sent to the shader in headless mode as well. It was set only when `!headless`, and the only other unconditional call site sits inside `render()` behind a `line_count > 0` check that a scene made of patches never satisfies. Because `Shader::initialize()` seeds the uniform with `(0,0,1)`, this did not blank the lighting outright — instead a headless `Visualizer` that never called `setLightDirection()` silently rendered with a directly-overhead light while a windowed one used the intended `(1,1,1)`, so the two modes shaded the same scene differently. Relatedly, `setLightDirection()` stored the normalized direction but forwarded the caller's raw vector to the shader, scaling the diffuse term by its magnitude; it now forwards the normalized vector.
- `updateDepthBuffer()` no longer renders into the shadow-map framebuffer. It reused that framebuffer and re-specified the shared depth texture at the window resolution, so any shadowed render following a `plotDepthMap()` call set an 8192x8192 shadow viewport against a texture the size of the window. It now owns a separate framebuffer and texture, checks them for completeness, reads back without selecting a front/back buffer (framebuffer objects have no such concept), and restores the default framebuffer binding when done.
- Added regression tests covering all of the above. None existed: `LIGHTING_PHONG_SHADOWED` appeared in the test suite only inside `CHECK_NOTHROW` setter calls, so no test had ever rendered a shadow. The new tests measure the ratio of a region's brightness rendered without versus with an occluder above it. That contrast metric is deliberate rather than an absolute darkness threshold — because the broken code rendered the whole scene uniformly dark, a test asserting "the shadowed region is dark" would have passed against the bug. The ratio is 3.60 with a working shadow map and exactly 1.00 without one, since the two renders come out byte-identical.
- Documentation: the headless section promised "identical visual output to windowed mode" and support for all visual effects, which was untrue for shadows for as long as headless mode has existed; it now states shadow support explicitly with a version note. The macOS `"using zero texture because texture unloadable"` warning was documented as harmless and safe to ignore — in headless mode with shadowed lighting it was in fact this bug reporting itself, and the note now says so.

## Leaf Optics
- Fixed the elementary-layer transmittance returning 0 instead of 1 when the absorption coefficient `k` is exactly zero, which produced NaN reflectance and transmittance across large parts of the spectrum. The reference PROSPECT implementation initializes `tau` to 1 and overwrites it only where `k > 0`, since zero absorption means the layer transmits everything; `transmittance()` instead guarded on `k < 0` and let `k == 0` fall past both polynomial branches to the final `return 0.0`. With `tau = 0` the internal transmissivity `t` is exactly zero, so the Stokes solution's `b = (1-r²+t²+D)/(2t)` divided by zero and every downstream quantity became NaN. This is reachable with documented inputs: the protein absorption spectrum is identically zero over 400–1432 nm and PROSPECT-PRO mode forces `drymass` to zero, so a PROSPECT-PRO leaf specified with no water absorbs nothing at 652 of the 2101 wavelengths. The NaN spectra were written to global data and consumed by the radiation plug-in with no diagnostic. Every existing spectrum test kept `watermass` and `drymass` non-zero — including the "Zero Values" test, which zeroes only the four pigments — so `k` never reached zero, and no test asserted finiteness.
- Fixed `transmittance()` dropping the boundary values of its approximation intervals. The two exponential-integral polynomial branches were guarded by independent range tests (`k > 0 && k < 4` and `k > 4 && k < 85`) rather than being chained, so `k` exactly equal to 4 matched neither and returned 0 — a jump from ≈0.0055 to 0 at a single point, where the two approximations otherwise agree to four decimal places. The interval tests are now chained so every non-negative `k` is covered. The `k >= 85` case still returns 0, which is correct rather than accidental: `exp(-85)` underflows to zero in double precision, so the layer is fully absorbing to within machine precision.
- Fixed the zero-absorption branch of the Stokes N-layer solution testing `r + t > 1` where the reference tests `r + t >= 1`. That branch exists to catch the degenerate non-absorbing case in which `D = sqrt((1+r+t)(1+r-t)(1-r+t)(1-r-t))` collapses to zero; at `r + t == 1` exactly the factor `(1-r-t)` vanishes, so the analytic solution above is degenerate and the boundary must be included. This matters in combination with the fix above, which makes the non-absorbing limit reachable.
- `getLeafSpectra()` and `PROSPECT()` now overwrite their output vectors instead of appending to them. Both took the spectra as `[out]` parameters but only ever called `push_back`, so calling either twice with the same vectors produced a 4202-element result while the conversion loop kept indexing the first 2101 entries — silently returning the first call's spectrum from the second call. The existing repeated-call test passed only because it used four separate fresh vectors and never exercised reuse.
- PROSPECT inputs are now validated rather than used as given. The structure parameter `numberlayers` (N) counts the elementary layers making up the leaf and must be at least 1: `N < 1` makes the Stokes exponent `N-1` negative, which inverts the layer stack and can drive the zero-absorption denominator `t + (1-t)(N-1)` through zero and change its sign, while `N = 0` divides by zero when computing the per-layer absorption coefficient. Negative constituent contents, which would contribute negative absorption, are likewise rejected. Both errors name the offending parameter, its value, and how to correct it.
- Documentation: the instructions for extending the species library pointed at `getPropertiesFromLibrary()` and described an if-else chain, but species are registered in `initializeSpeciesLibrary()` in a `std::map` keyed by lower-case name; the section now shows a correct worked example and notes the lower-casing requirement. The species table was also missing `common_bean` and `cowpea`, which have been in the library but undocumented and therefore undiscoverable for a user reading only the documentation. Added a section documenting the new input validation, and a note that a leaf with no dry matter at all warns rather than erroring.

## Radiation
- Fixed camera images reading a factor of two too high by double-counting the emitted flux already held in the host camera-scatter accumulator. The device camera-scatter buffers are write-only `atomicFloatAdd` targets that nothing reads back during a launch, so they must be zero going into each launch for the post-launch download to yield only that launch's own contribution — but `runBand()` uploaded the host accumulator into them instead, so the download returned base + new and the base was added on top of itself. Cameras are now zeroed rather than seeded before the emission launch (what the cameras actually read is `radiation_out`, uploaded from the same host accumulator further down, which is unaffected). On the OptiX8 backend the first `runBand()` of an emission-only scene was accidentally correct, because with no sources the earlier zeroing call is skipped and the upload is guarded on device pointers that nothing had allocated yet, so only the second and later renders doubled; on OptiX 6.5 the buffers are allocated eagerly and every render doubled. Zeroing here also allocates the buffers, removing that first-call/later-call asymmetry.
- `uploadCameraScatterBuffers()` has been removed from the `RayTracingBackend` interface and from the OptiX 6.5, OptiX8 and Vulkan implementations. Seeding the device camera-scatter buffers from the host accumulator was the double-counting described above, and now that `runBand()` zeroes them instead the method has no callers; leaving it on the interface would have obliged every future backend to implement an upload path that must never be used, and made it available to be called again by mistake.
- Fixed the OptiX 6.5 backend tracing a second render of the same camera on top of the previous image, doubling it again on every repeat `runBand()`. `radiation_in_camera_RTbuffer` accumulates across the tiles of one camera launch, so `launchCameraRays()` zeroes it only when the camera ID or launch band count changes; nothing ever reset that latch between runs, so an unchanged camera matched the existing ID and skipped the zero. `zeroRadiationBuffers()` now resets `current_camera_launch_id` at the start of every `runBand()`, which is what the OptiX8 backend already did and the OptiX 6.5 path never had.
- Added a camera self-test that compares a pixel value against a quantity known independently of Helios, which is what let both defects above survive from v1.3.64: every pre-existing camera test asserts only on the shape of the output (pixel data exists, has the right size, printed no error), so a camera rendering the correct image multiplied by a constant passed all of them. A single isolated blackbody patch with no other geometry and no sources has no inter-primitive scattering, so a pinhole camera with manual exposure — which bypasses the auto-exposure gain — must read exactly the Stefan-Boltzmann radiance `sigma*T^4/pi`. The same camera is rendered three times and every run is checked against the analytic value rather than against run 1, so neither a backend that doubles on repeat renders nor one that doubles on every render can pass.

## Weber-Penn Tree
- The `getTrunkUUIDs()`, `getBranchUUIDs()`, `getLeafUUIDs()`, and `getAllUUIDs()` getters now prune UUIDs of primitives that have since been deleted from the Context before returning, via `cleanDeletedUUIDs()`. The plug-in caches the UUIDs it creates at build time and never learned about deletions made directly through the Context, so deleting tree geometry left the getters handing back dangling UUIDs — and since `getAllUUIDs()` is typically passed straight to another plug-in, that surfaced as a `does not exist in the Context` error from a call site far from the deletion.
- A `<LeafAngleDist>` tag whose probability density does not integrate to 1 is now rejected with an error from `loadXML()` instead of being discarded in favor of the default leaf placement. The old behavior printed a warning and then built the tree anyway with branch-relative leaf orientations, so a user who mistyped or forgot to normalize their distribution got a tree that silently ignored it. The error message reports what the given values integrate to, what they must sum to for the number of inclination classes provided, and the factor to divide by to normalize them. The (commented-out) almond distribution shipped in `WeberPennTreeLibrary.xml` integrated to 0.9982 and has been renormalized so that it no longer trips the check if uncommented.
- Fixed first-level branches being attached at the wrong length whenever exactly one branch is placed per trunk segment (i.e. `nBranches[1]` equal to `nCurveRes[0]`). The offset of each branch along the trunk divided by `stems_per_segment-1`, which is a division by zero in that case; the resulting infinity made the position ratio passed down into `recursiveBranch()` negative infinity, which was then clamped to 0, so every branch was built at the length the shape function gives at the very base of the crown rather than at its true height. No NaN ever reached the geometry, which is why the tree still looked plausible. The divisor is now `stems_per_segment`, matching the fraction already used to place the branch's attachment point on the trunk.
- Tree parameters are now validated wherever they enter the plug-in — both when a library is read by `loadXML()` and when `setTreeParameters()` is called — instead of being used as given. The geometry routines index the per-level parameter arrays by recursion level and divide by several of the parameters, so an out-of-range set produced a division by zero, a bare `std::out_of_range` from deep inside the recursion, or a silently empty or degenerate tree with no diagnostic at all: `nCurveRes` of 0 built a tree with no primitives, and `Levels` of 4 indexed one past the fixed four-entry arrays and built a tree with no leaves. The check requires `Levels` between 1 and 3, every per-level array to hold at least `Levels+1` entries, `nCurveRes` of at least 1 and non-negative `nBranches` at every level, positive `Scale`, `nLength[0]`, `LeafScale` and `LeafScaleX`, and `BaseSize` in [0,1), and it names the offending parameter and its value. `setTreeParameters()` validates before storing, so a rejected set does not overwrite the library entry. Relatedly, `recursiveBranch()` no longer silently remaps a recursion level that is past the end of the parameter arrays onto `Levels-1`, and `getAllUUIDs()` now bounds-checks the trunk and branch containers it indexes rather than only the leaf container.

## Photosynthesis
- The empirical model now applies the temperature response function `f_T` that its documentation has always specified. `evaluateEmpiricalModel()` computed the light response `f_L` and the CO<sub>2</sub> response `f_C` but never evaluated `f_T` at all, so assimilation was `Asat*fL*fC - Rd` and the only temperature dependence in the model came through the respiration term. The `Tmin`, `Topt`, `Tref` and `q` coefficients of `EmpiricalModelCoefficients` were therefore settable, serialized to material data, and completely inert — a leaf at 5 °C and a leaf at 45 °C returned the same gross assimilation rate. `f_T` is clamped to zero at or below `Tmin` and above the upper temperature at which its second factor changes sign, so it never goes negative, and coefficients that make the reference denominator zero (`Tref <= Tmin`, or `(1+q)*Topt - Tmin - q*Tref == 0`) are now rejected with an error rather than producing an infinite or NaN assimilation rate. That coefficient check is made before the `TL <= Tmin` early return rather than after it: the coefficients are a property of the set and not of the current leaf temperature, so validating afterwards skipped the check entirely on every timestep where the leaf sat at or below `Tmin` and returned a well-formed `A = -Rd`, hiding an invalid configuration for part of a diurnal or seasonal run and surfacing it only once the leaf warmed up.
- `setModelCoefficients()` now serializes the TPU and light-response-curvature `theta` temperature responses to material data, and `getCoefficientsForPrimitive_Farquhar()` reconstructs them. Both were omitted from the material round-trip, so a material saved with TPU limitation enabled or with a non-rectangular-hyperbola light response came back with those parameters reset to the struct defaults, silently changing the modeled rates. The `TPU_flag` restore was also moved to after the `setTPU()` call, since `setTPU()` enables the flag as a side effect and would otherwise overwrite a deserialized value of 0. Both new label groups are optional on read, so materials written by earlier versions still load (with the struct defaults for these two parameters).
- The Farquhar species library values for Almond, Walnut and Pistachio now match the parameters published in the plug-in documentation, and Prune's `Rd25` was corrected from 1.56 to 1.65. The shipped values had diverged substantially from the documented table — Almond, for instance, used `Vcmax25` = 105.9 against a documented 72.6 — so `getFarquharCoefficientsFromLibrary()` returned a different parameter set than the one a user reading the documentation expected. The three re-fitted species now use the peaked temperature response with TPU limitation as documented. A self-test checks every library species against the documented table so the two cannot drift apart again.
- A four-argument peaked temperature response constructed with `dHd <= dHa` is now rejected with an error. The peaked Arrhenius form evaluates `ln(dHd/dHa - 1)`, which is undefined in that case, so the response returned NaN and propagated it into `net_photosynthesis` with no diagnostic. The three-argument setters are unaffected, as they default `dHd` to `10*dHa`.
- Calling `setVcmax()`, `setJmax()`, `setRd()` or `setQuantumEfficiency_alpha()` now resets the corresponding deprecated public scalar field (`Vcmax`, `Jmax`, `Rd`, `alpha`) to its `-1` sentinel. Those fields are still honored when assigned directly, and a value greater than zero selects the legacy non-peaked Arrhenius path in preference to the temperature-response object — so a struct that had one assigned directly and was then updated through the setter kept using the stale legacy value, and the two representations disagreed about which number the model actually used. The setter is now always authoritative.

## Stomatal Conductance
- The Bailey (`BB`) model is now coupled to the boundary layer in the same way as the other models. It previously used the air-to-cavity vapor pressure deficit directly instead of solving the water vapor flux balance at the leaf surface for `Ds`, so it was the only model in the plug-in that did not respond to the boundary-layer conductance at all. It also never applied the soil moisture factor `beta`, making it the only model that ignored `beta_soil` primitive data. `run()` now solves for the surface vapor pressure with `fzero()` as the BMF model does, and the guard-cell relation — which is implicit in `gs` but linear in it — is solved in closed form by the new `evaluate_BBconductance()` rather than by a second nested iteration. The model theory and the newly required `xylem_water_potential` and `radiation_flux_PAR` primitive data are now documented.
- Non-physical inputs to `run()` now raise an error instead of being silently clipped or propagated. A negative boundary-layer conductance was clipped to zero with a warning, which then divided by zero in the surface CO<sub>2</sub> balance; it is now rejected, as are a non-positive ambient CO<sub>2</sub> concentration, a surface CO<sub>2</sub> concentration that evaluates non-positive (which flips the sign of `gs` in all three `A`-based models, and happens when the assimilation rate is too large to be supplied through the given boundary-layer conductance), and `Cs <= Gamma` in the Ball-Berry-Leuning model, which sits on that model's pole. Note that a negative `A` from dark respiration legitimately gives `Cs > Ca` and remains valid.
- A boundary-layer conductance of exactly zero — which the BLConductance plug-in produces for zero wind speed or a zero-size primitive — is now handled as the physical limit it represents rather than dividing by zero: no vapor or CO<sub>2</sub> can be exchanged with the air outside the boundary layer, so the steady-state conductance is taken to be zero and a warning is issued.
- `setDynamicTimeConstants()` now rejects a time constant that is zero, negative or non-finite. `tau` appears in the denominator of the forward Euler update, so zero gave a non-finite conductance and a negative value inverted the relaxation so that stomata diverged away from the steady-state value.
- Added a deleted `run(std::initializer_list<uint>)` overload so that `run({UUID})` fails to compile rather than converting the braced integer to a float and being silently interpreted as `run(dt)` — a timestep — instead of as a single-primitive UUID list.
- `optionalOutputPrimitiveData()` no longer appends a duplicate entry when called twice with the same label, and the Ball-Berry-Leuning missing-`Gamma_CO2` warning now reports the count of primitives missing Gamma rather than the count missing `net_photosynthesis`.

## Parameter Optimization
- An exception thrown by a user's objective, gradient or constraint function during an L-BFGS, BOBYQA or SLSQP run is no longer discarded. All three algorithms ended their catch chains with `catch (const std::exception &e)`, which folded the error into a `result_message` string that is only ever printed inside an `if (print_progress)` block — and `print_progress` defaults to `false`. Execution then fell through to build a `Result` from `optimal_params` and `optimal_value`, which were seeded with the initial parameters and `0.0`, so a failed optimization returned a fitness of exactly zero: for a minimization, the one value most likely to be read as perfect convergence. An objective that failed on bad input, a failed sub-simulation or an unopenable file therefore produced silently wrong scientific output with no indication anything had gone wrong. This was specific to the three NLopt-backed algorithms; Adam, GA and CMA-ES have no catch blocks and always propagated.
- NLopt's own configuration and internal failures are now converted to `helios_runtime_error` instead of escaping as whatever the library happened to throw. `nlopt-in.hpp`'s `mythrow()` maps `NLOPT_INVALID_ARGS` to `std::invalid_argument` and `NLOPT_OUT_OF_MEMORY` to `std::bad_alloc`, and the `nlopt::opt` constructor throws `std::bad_alloc` directly — so a genuine failure could surface as a `std::logic_error`, which slips past a caller catching `std::runtime_error` as the rest of Helios does, carrying a message that names NLopt rather than anything the caller can act on. This affected not just `optimize()` but every setup call, all of which sat outside the try block: the constructor, the bound and tolerance setters, and SLSQP's constraint registration. Exceptions originating in user code are deliberately not translated, since those are captured in the callbacks and rethrown verbatim.
- SLSQP now validates its constraints up front rather than letting malformed ones reach NLopt. `Constraint` is an aggregate whose fields the user assigns directly, and none of them were checked anywhere in the plug-in. A negative `tolerance` was caught only by NLopt, as `std::invalid_argument`, reported far from the offending field and identifying no constraint index. A NaN `tolerance` was not caught at all — NLopt tests only `tol < 0`, and a NaN comparison is false — so the optimization ran to completion against a corrupt feasibility test and returned a plausible-looking result. An unset `function` or `gradient`, which is simply a default-constructed `Constraint` that the user forgot to finish populating, raised `std::bad_function_call` from inside a callback; like `std::invalid_argument`, that derives from `std::exception` but not `std::runtime_error`, so it escaped a caller following the convention used everywhere else in Helios. All four are now rejected before the optimizer is configured, naming the constraint's index. A tolerance of exactly zero remains valid and means exact satisfaction.
- The `catch (const GradientValidationError &)` clauses in `runLBFGS()` and `runSLSQP()` were dead code for the case they appeared to handle — a validation error raised inside a callback was swallowed by NLopt's `catch (...)` long before reaching them, which is why the eager pre-validation before the optimization loop exists. They have been removed; the eager check still covers the missing-gradient-key case. A `nlopt::forced_stop` handler was added to BOBYQA and SLSQP, not as a user-facing cancellation feature (nothing in the plug-in calls `force_stop()`) but as the path by which an exception captured from user code now surfaces.
- Existing tests could not catch any of this: every assertion in the suite used an objective that always succeeds, so no test ever entered the swallowing branch. Five new regression tests cover an objective that throws under each of the three algorithms, plus a gradient and a constraint function that throw, and each asserts on the propagated message rather than only its type — an assertion on type alone would also pass against the weaker fix that lets NLopt's own generic error escape. The gradient test deliberately succeeds on its first call, because L-BFGS evaluates the gradient once before entering the optimization loop and a gradient that always threw would propagate even on the unfixed code, proving nothing.
- Added the CMake option `HELIOS_NLOPT_LUKSAN` (default `ON`, so no existing build changes) to exclude NLopt's LGPL-2.1 Luksan solvers. NLopt is otherwise MIT, and per its own `COPYING` the compiled library carries the conjunction of its components' licenses, so a default build is effectively LGPL; `src/algs/luksan/COPYRIGHT` additionally carries an ACM notice restricting redistribution of some subroutines. Building with `-DHELIOS_NLOPT_LUKSAN=OFF` compiles none of those sources — verified empirically, with 104 Luksan symbols in the default `libnlopt.a` and none in the MIT build — at the cost of L-BFGS, which is implemented entirely by that code. BOBYQA (MIT), SLSQP (BSD-3) and the four algorithms needing no NLopt are unaffected.
- L-BFGS availability is now a distinct compile-time flag, `HELIOS_HAVE_LBFGS`, since `HELIOS_HAVE_NLOPT` no longer implies it. Selecting L-BFGS without it raises an error naming the CMake option and suggesting Adam or BOBYQA, rather than reaching NLopt's dispatcher, which prints to stdout and returns `NLOPT_INVALID_ARGS` — a diagnostic that surfaced far from its cause and, before the exception fix above, was itself swallowed into the same silent zero-fitness result. Whether a *system-installed* NLopt includes the Luksan solvers cannot be detected, since the flag is private to NLopt's build and `nlopt.h` declares `NLOPT_LD_LBFGS` unconditionally regardless of what was compiled; the build assumes they are present, which is correct for every mainstream distribution, and `HELIOS_SYSTEM_NLOPT_HAS_LUKSAN=OFF` asserts otherwise.
- The no-NLopt fallbacks in `runLBFGS()`, `runBOBYQA()` and `runSLSQP()` now raise an error instead of writing to `std::cerr` and returning the objective evaluated at the initial point. That fabricated result was the same class of defect as the swallowed exceptions — a caller could not distinguish it from a converged optimization — and `std::cerr` is invisible under a test harness or a redirected log.

# [1.3.79] 2026-07-29

## Core
- Fixed `getObjectBoundingBox()` returning a degenerate box for an object made of a single primitive. The loop seeded `min_corner`/`max_corner` from the first primitive's `vertices.front()` and then `continue`d to the next primitive, so the remaining vertices of that first primitive were never compared against the seed — a one-patch tile object reported `min == max ==` its first vertex, and in the multi-object overload the first object's unique extremes were dropped (only the *first* primitive of the *first* object was affected, since the `p == 0 && o == 0` guard could match only once). The seed no longer skips its own primitive, primitives with no vertices are stepped over, and a call covering no primitives at all now fails fast rather than returning the caller's untouched output corners as if they were a valid box. The two pre-existing bounding-box self-tests could not detect this because both used a box object, whose six faces cover each other's extremes; verified by new single-primitive-object, first-object-in-list, and empty-request self-tests that fail on the unfixed code.
- Added `gpuRequiredByEnvironment()` and `requireGPUOrFail()`, the mirror image of the `HELIOS_NO_GPU` veto: setting `HELIOS_REQUIRE_GPU` makes a test that would skip for lack of a GPU fail instead. Because every GPU test skips gracefully when no device is found, a CI runner whose driver has died reports a green build having tested nothing at all — which is what happened to the Linux EC2 GPU runner starting in 1.3.78. A kernel upgrade from 6.14 to 6.17 left the NVIDIA DKMS module built only for the old kernel (`modprobe nvidia` reporting no such module, and driver 575.57.08 failing to rebuild against 6.17 on a removed DMA-BUF helper), so `nvidia-smi` could not reach the driver. The CUDA *toolkit* was untouched and still compiled, so the build succeeded while all 100 radiation GPU tests and the collision-detection GPU-path assertion silently skipped, and the job reported success. `requireGPUOrFail()` is inert unless the variable is set, so developer machines and the non-GPU CI runners keep skipping exactly as before, and it names the specific test that would have skipped rather than only reporting that some GPU was missing. Setting both `HELIOS_REQUIRE_GPU` and `HELIOS_NO_GPU` is contradictory (the veto guarantees the demanded GPU can never be found) and is reported as such rather than letting one silently win. Unlike the veto, which the backend probes consult on hot paths and therefore cache, this reads the environment on every call: it is invoked at most once per test case, where the cost is irrelevant and a live read keeps both branches reachable within a single test binary. Verified by new self-tests covering the unset, explicit `"0"`, set, and arbitrary-value cases, the caller's context string appearing in the failure, and the both-variables-set contradiction.
- `Location` now validates its fields, which `Date` and `Time` in the same header have always done and `Location` alone did not. Latitude must lie in [-90,90], longitude in [-180,180], the UTC offset in [-14,+12], and the altitude must be finite; every check is written in negated form so a NaN is rejected rather than passing as "not out of range". The UTC offset range is asymmetric because Helios counts the offset positive moving West, which inverts the real-world span of UTC-12 through UTC+14 (Kiribati keeps the latter) to +12 through -14. Validation lives in a new public `Location::validate()` called from both parameterized constructors, so `make_Location()` inherits it through the constructor rather than duplicating the checks as `make_Date()` does. `Context::setLocation()` calls `validate()` again on the way in, which is not redundant: `Location`'s fields are public, so a caller can hold a valid `Location`, assign nonsense to one member, and hand it straight back — which is precisely what the ISO-8601 branch of `loadTabularTimeseriesData()` does with `UTC_offset`, using a value read out of an external file that no constructor can intercept. A rejected `setLocation()` leaves the Context's location unchanged. Note that these constructors throw `std::runtime_error` directly rather than calling `helios_runtime_error()`: `helios_vector_types.h` is included by `global.h` before that function is declared, so it is not visible, and every other validating type in the header (`Date`, `Time`, `int2`, `vec3`, `RGBcolor`, ...) throws directly for the same reason. Verified by new self-tests covering each range boundary as valid, each field just outside its range as invalid, NaN and infinity in every field, the default constructor satisfying its own invariant, the offending value and field name appearing in the message, and the mutate-then-`setLocation()` path including the no-change-on-rejection guarantee — all 18 assertions fail without the validation.

## Radiation
- Fixed `runBand()` tracing against stale geometry when primitives were added to or deleted from the Context after a previous run, so calling `updateGeometry()` is no longer required at all in the common case. The `isgeometryinitialized` flag latched `true` on the first build and was never reset, so subsequent runs skipped the rebuild and the acceleration structure still held only the original primitives: newly added primitives silently received 0.0 flux on the Vulkan backend (whose `buildAccelerationStructure()` is a no-op), or the launch tripped the OptiX8 "No acceleration structure" guard when the first build covered an empty Context. The flag was also left uninitialized in the constructor, so it could read as `true` on a fresh model and suppress the initial build entirely. `runBand()` now calls the new `needsGeometryRebuild()`, which rebuilds when geometry has never been built or when the Context primitive count differs from the count recorded at the last build; the flag is initialized to `false`. A build made through the `updateGeometry(UUIDs)` overload that covers fewer primitives than the Context holds is latched as an explicit user-specified subset and is never auto-rebuilt, since rebuilding from the full Context would silently discard the requested subset (verified by new self-tests for the flag default, geometry auto-initialization without an explicit `updateGeometry()` call, tracing of a patch added after a first `runBand()`, and non-rebuild of an explicit subset). The OptiX8 backend's "No acceleration structure" errors now reference `updateGeometry()` rather than the internal `buildAccelerationStructure()`. Corrected the documentation's claim that the geometry is rebuilt whenever primitives are added or deleted: because the check compares primitive counts, it cannot see an in-place modification of existing primitives, nor a deletion and addition of equal numbers between runs, and `updateGeometry()` remains required in those cases.
- Fixed a SIGSEGV during `RadiationModel` construction on macOS runners where MoltenVK loads but cannot execute compute, and fixed the spurious "No compatible GPU backend found" that preceded it. Two independent defects: (1) `VulkanDevice::createInstance()` requested `VK_KHR_portability_enumeration` and set `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` unconditionally under `__APPLE__` without ever querying `vkEnumerateInstanceExtensionProperties`, relying on the driver to reject an unsupported name. That extension is implemented by the *Vulkan loader*, not by MoltenVK — so whenever the build links `libMoltenVK.dylib` directly (which `plugins/radiation/CMakeLists.txt` does when no loader is found), every `vkCreateInstance()` call failed with `VK_ERROR_EXTENSION_NOT_PRESENT` and a perfectly usable driver was reported as absent. Instance extensions and layers are now enumerated first and only requested when supported: behind the loader the portability extension is still requested (MoltenVK's portability-subset device is hidden without it), and when linked directly it is skipped, which makes the instance create succeed and enumerate the physical device. The same query now gates the validation layer and `VK_EXT_debug_utils`, replacing the create-then-retry-without-validation dance, and the instance handle is explicitly reset to `VK_NULL_HANDLE` on failure so a driver that writes a non-null handle into a failed create cannot lead the destructor into `vkDestroyInstance()` on a handle that was never created. Enumeration alone is not sufficient, because the loader and the ICD can disagree about which instance extensions exist and the disagreement is only observable by trying: the macOS loader advertises `VK_KHR_portability_enumeration` (it implements the extension itself) but MoltenVK rejects the name once it is forwarded to the driver. Since every extension requested for headless compute is optional, a create that fails with `VK_ERROR_EXTENSION_NOT_PRESENT` or `VK_ERROR_LAYER_NOT_PRESENT` is now retried once with no optional extensions or layers at all rather than reporting a usable driver as missing; if that retry also fails, the error names both result codes so neither failure is swallowed. (2) `RayTracingBackend::create("auto")` re-probed on every call and each probe ran a full `vkCreateInstance`/`vkDestroyInstance` cycle (plus a logical device and a compute dispatch when the instance was created), so every `RadiationModel` construction and every GPU availability check churned the driver — the same failure mode the test suite's shared-device singleton already worked around for NVIDIA loaders, which start returning `VK_ERROR_INCOMPATIBLE_DRIVER` after roughly 25 cycles. `OptiX8Backend::probe()`, `OptiX6Backend::probe()`, and `VulkanComputeBackend::probe()` now compute their result once per process and cache it, so a process performs at most one driver initialization; the accepted trade-off is that a transient failure is latched until the process restarts. A new `VulkanDevice::getInitializationAttemptCount()` diagnostic makes the churn observable, and a new self-test asserts that 25 consecutive probes produce at most one initialization — it fails on the unfixed code, which records 27. Two further self-tests cover the acceptance path: 25 consecutive `RadiationModel` constructions against an unusable driver must produce 25 clean `helios_runtime_error` throws and no crash, and the same loop must hold in a forked child (the macOS harness forks per test, and Metal is not fork-safe — with the probe cached the child answers from inherited state without re-entering the driver). Verified against a genuinely broken driver by running the suite with the Vulkan ICD forced absent.
- Made `HELIOS_NO_GPU` a real veto across the whole radiation backend layer rather than a `RadiationModel::isGPUBackendAvailable()`-only flag. Every backend `probe()` now consults the new `gpuBackendsDisabledByEnvironment()`, so setting the variable makes a GPU-equipped machine behave exactly like one with no compatible hardware: `probeAnyGPUBackend()` returns false and `RayTracingBackend::create("auto")` throws with a message naming the environment variable instead of listing driver troubleshooting steps that do not apply. Previously the variable suppressed the availability *query* while `create("auto")` still probed and happily built a backend, so the reported availability could diverge from the path actually taken; `isGPUBackendAvailable()` now simply forwards to `probeAnyGPUBackend()` so there is a single source of truth. Explicitly requesting a backend by name (e.g. `create("vulkan_compute")`) still bypasses the veto.
- Fixed `writeCameraImage()` and `autoCalibrateCameraImage()` throwing a raw `std::out_of_range` ("invalid map<K, T> key") for a camera that exists but has never been rendered. Both guarded on `band_labels`, which is populated when the camera is *created* by `addRadiationCamera()`, and then indexed `pixel_data`, which is populated only when the camera is *rendered* by `runBand()` — so a camera added after the last `runBand()` call, or whose bands were never dispatched, passed the guard with no image data behind it. Both now check `pixel_data` itself and report which camera and band needs a `runBand()` call, `writeCameraImage()` by skipping the write as it already does for a missing camera or band and `autoCalibrateCameraImage()` by failing fast as it already does for a missing band. `writeNormCameraImage()`, which had no camera-existence check at all, now performs the same check as its siblings rather than letting `cameras.at()` throw. Separately, `updateCameraParameters()` left the per-pixel buffers (`pixel_data`, `pixel_label_UUID`, `pixel_depth`, and the corresponding camera global data) at their previous size when the resolution changed, while every reader indexes them using the camera's *current* resolution — so a resolution change followed by an image write read past the end of the buffer; the stale render is now discarded on a resolution change (leaving the camera reporting as not-yet-rendered, which callers already handle) and `writeCameraImage()` additionally rejects pixel data whose size does not match the current resolution. Non-resolution parameter changes still leave an existing render intact. Verified by new self-tests for the unrendered-camera, never-dispatched-band, nonexistent-camera, resolution-change-discards, and non-resolution-change-preserves cases, all of which fail on the unfixed code.
- Fixed `RadiationModelTestHelper::isGPUAvailable()` reporting no GPU whenever the shared Vulkan test device failed to initialize, even on builds where an OptiX backend is also compiled in and `createWithSharedDevice()` would return an OptiX model without ever touching the shared Vulkan device. On such a build — the configuration the Linux EC2 GPU runner uses — a broken Vulkan installation alone was enough to skip every radiation GPU test on a machine with a perfectly healthy NVIDIA GPU. The Vulkan health check is now applied only on builds where `createWithSharedDevice()` actually hands out a Vulkan backend, mirroring that function's own backend selection, which restores the behavior the docstring always claimed ("always true for OptiX, Vulkan-dependent otherwise").
- Every self-test that constructs a `RadiationModel` is now declared `GPU_TEST_CASE`, including those whose subject is entirely CPU-side (the `isgeometryinitialized` default, and the unrendered-camera and nonexistent-camera guards in `writeCameraImage()`/`writeNormCameraImage()`): reaching the behavior under test needs no ray tracing, but the constructor calls `RayTracingBackend::create("auto")`, which throws `No compatible GPU backend found` on any runner without a usable device, so a plain `DOCTEST_TEST_CASE` passes on a GPU-equipped development machine and fails at construction — before a single assertion runs — on the non-GPU runners. The camera tests build through `RadiationModelTestHelper::createWithSharedDevice()` rather than the plain constructor, matching the dominant pattern in the file and avoiding the per-model Vulkan device churn the shared-device singleton exists to prevent. The `GPU_TEST_CASE` skip path now calls `helios::requireGPUOrFail()` first, so the skip becomes a failure on a runner dedicated to GPU coverage.
- Fixed `updateGeometry()` throwing `ERROR (OptiX8Backend::buildAccelerationStructure): No geometry uploaded. Call updateGeometry() first.` on the OptiX8 backend whenever it was called while the Context held no primitives. `buildAccelerationStructure()` rejected a zero primitive count, but `updateGeometry()` is its only caller, so the advice to call `updateGeometry()` was emitted from inside the very function it was asking for. The guard was also the only one of its kind: `OptiX6Backend::buildAccelerationStructure()` sets primitive counts of zero without complaint, `VulkanComputeBackend::buildAccelerationStructure()` is a no-op, and `buildGAS()` — the function immediately below the guard — already resets `gas_handle` to 0 and returns early for zero primitives, so the empty build was supported everywhere except at the point that refused to attempt it. Removing the check restores cross-backend parity and makes the sequence introduced with `needsGeometryRebuild()` work on OptiX8: build over an empty Context, add primitives, then call `runBand()`, which notices the changed Context primitive count and rebuilds before tracing. Nothing is silently permitted as a result — the four launch entry points (`launchDirectRays()`, `launchDiffuseRays()`, `launchCameraRays()`, `launchPixelLabelRays()`) each already refuse a `gas_handle` of 0 with a message that correctly tells the caller to add geometry and then call `updateGeometry()`, which is the point at which tracing against an empty scene actually becomes an error. This was caught by the existing "RadiationModel Multi-Band Stale Backend Geometry" self-test on a CUDA runner; that test cannot fail on the Vulkan backend, where the rejected call is a no-op, which is why it went unnoticed on non-CUDA development machines.
- Fixed the EXIF `OffsetTime` tag being written as a silently truncated string when the Context's UTC offset falls outside the range of real time zones, which also produced a `-Wformat-truncation` warning on the Linux GCC build. `formatStandardOffset()` formatted the sign, hours, and minutes into a `char[8]` with `snprintf()` — exactly the size of `+HH:MM` and its terminator, correct for every real offset, but `Location::UTC_offset` is a plain `float` that nothing validates, so an offset of, say, 100 hours was truncated into `-100:0` and embedded in the image as if it were a valid tag. GCC flagged the call because value-range propagation can bound the hour field only by `INT_MAX`, making the warning a true report of the malformed-input case rather than a false positive. The offset is now validated against `[-14,+12]` in Helios's +West convention (i.e. UTC-12 through UTC+14, covering every zone in use), and an out-of-range or NaN value fails fast naming the offending value and `Context::setLocation()` instead of writing a corrupt tag; the formatting itself moved to `std::ostringstream`, matching the neighboring `formatExifDateTime()` and removing the fixed buffer the warning was about. Verified by a new self-test asserting that a fractional zone East of UTC (Helios -5.75, i.e. UTC+5:45) reaches the JPEG as `+05:45`. With `Location` validation now in place (see Core, above), a malformed offset can no longer reach the radiation plugin at all, so the check in `formatStandardOffset()` is retained only as a backstop against a future path that sets the Context's location without passing through `Location`'s constructors or `Context::setLocation()`; its regression coverage lives in the core `Location` tests rather than here, since the throw is no longer reachable through `writeCameraImage()`.

## Plant Architecture
- Fixed `advanceTime()` throwing `ERROR (Tube::setTubeRadii): Number of radii in input vector must match number of tube nodes.` on any plant restored with `readPlantStructureXML()`, which made the save/reload/continue-growing workflow impossible. `Shoot::shoot_internode_vertices`/`shoot_internode_radii` store one inner vector per phytomer under a vertex-sharing convention set by the `Phytomer` constructor: the first phytomer on a shoot holds all `Ndiv+1` nodes, while every later phytomer omits its node 0 because that node is the previous phytomer's last node, so a shoot of `P` phytomers flattens to exactly `(Ndiv+1) + (P-1)*Ndiv` nodes — the node count of the internode `Tube` that `Shoot::updateShootNodes()` pushes the flattened arrays into. The XML writer already honored this convention, but the reader reconstructed `Ndiv+1` vertices for *every* phytomer and seeded each one with the previous phytomer's last vertex, duplicating each shared node; both flattened arrays came out `P-1` entries too long while the `Tube` (built through the normal growth API during the restore) had the correct count, so the next `setTubeRadii()` failed the size check. The reader now mirrors the constructor's convention and drops the leading shared node for phytomers after the first. This also closes two latent defects on the same data: `Phytomer::setInternodeLengthScaleFraction()` differenced the duplicated node against its own twin and divided by the resulting zero-magnitude axis, writing NaN into every downstream vertex of the shoot (reachable as soon as a restored plant is elongated with a scale factor below 1), and `Phytomer::getInternodeNodePositions()` prepended a second copy of the shared node, giving a zero-length leading segment that makes `interpolateTube()` return NaN at fraction 0. `readPlantStructureXML()` additionally now applies the saved `<internode_length_segments>` (to the plant's shoot-type snapshot as well as the local parameters, since `addBaseStemShoot()`/`addChildShoot()` re-read the snapshot and ignore caller-modified parameters) and the saved `<internode_length_max>`, both of which were parsed and discarded, and calls `updateShootNodes()` once per restored plant so the Context tube geometry reflects the reconstructed geometry rather than the library-parameter geometry the constructor built — previously a restored plant rendered the wrong shape until the first `advanceTime()`. The dead backward-compatibility branch that assigned `<internode_vertices>`/`<internode_radii>` was removed (unreachable, since the segment count defaults to 1 and those tags are no longer written) along with their now-unused parsing, and a zero `<internode_length_segments>` now fails fast instead of dividing by zero. Verified by a new self-test that round-trips a bean plant and asserts, before growth, that each shoot's flattened node/radius counts equal its `Tube` node count, that no phytomer's first node duplicates the previous phytomer's last, and that the Context tube nodes match the reconstructed vertices; the test fails on the unfixed code. Note that phenological thresholds remain absent from the XML schema, so a restored plant keeps the `PlantInstance` defaults where `dd_to_dormancy_break + dd_to_dormancy == 0`, which satisfies the dormancy check on the first `advanceTime()` step and drops the plant's leaves — a separate pre-existing gap.
- Fixed `pruneGroundCollisions()` leaving a freed object ID in `Shoot::internode_tube_objID`. When a shoot's internode geometry dipped below the plane set by `enableGroundClipping()`, the tube object was deleted from the Context but the field kept the now-dangling ID, so anything reading it directly addressed a deleted object (a clean `helios_runtime_error` in a debug build, but an opaque `std::out_of_range` from `std::map::at` in a release build, which is the default). The ID is now reset to the sentinel — exposed as the named constant `Shoot::no_internode_tube_objID` rather than a bare `4294967294` literal — immediately after the delete, so the dangling ID never exists in the first place; the existing `doesObjectExist()` guards in `getShootInternodeObjectIDs()`, `getPlantInternodeObjectIDs()`, and `getAllPlantObjectIDs()` are retained as a backstop (verified by a new self-test that drives a plant into ground clipping and asserts every shoot either has a live tube object or holds the sentinel, which fails on the unfixed code). The other organ getters (leaves, petioles, peduncles, flowers, fruit) were audited and need no such guard: those ID containers are cleared in lockstep with the geometry deletion (`Phytomer::removeLeaf()` clears `leaf_objIDs` and `leaf_bases`; `setFloralBudState()` resizes the inflorescence and peduncle vectors), which a second new self-test pins in place.
- Made the leaf object-ID/attachment-base pairing in `setPlantLeafAngleDistribution()` structurally safe. The implementation zipped the results of two independent traversals, `getPlantLeafObjectIDs()` and `getPlantLeafBases()`, index-for-index and rotated each leaf about `bases[i]` — an invariant guarded only by an `assert` that compiles out in release builds. Had either getter ever started filtering its output, every leaf after the first dropped one would have been silently rotated about a different leaf's base point rather than failing loudly. Both are now gathered in a single traversal by the new private `getPlantLeafObjectIDsAndBases()`, so the two can never desynchronize; the public getters are unchanged. Also guarded an unchecked `leaf_bases.front()` in `getPlantLeafBases()` that was undefined behavior for a phytomer whose leaves had been removed.
- Fixed `Phytomer::build_context_geometry_petiole` and `build_context_geometry_peduncle` never being assigned. The `Phytomer` constructor and `updateInflorescence()` declared locals that shadowed the members of the same name, so the locals governed whether geometry was *built* while the permanently-`true` members governed whether it was *deleted* and *transformed* — meaning `disablePetioleContextBuild()`/`disablePeduncleContextBuild()` were not honored on those paths. The members are now assigned directly. Currently self-cancelling (a disabled build leaves the ID vectors empty, so the deletes were no-ops), but it would have become a live dangling-ID bug the moment either path started doing real work.
- Corrected the function name in three error messages that reported `getPlantInflorescenceObjectIDs`, a function that does not exist: `getPlantFlowerObjectIDs()`, `getPlantFruitObjectIDs()`, and `updateShootFruitCounts()`.

## Collision Detection
- The CPU/GPU equivalence test's GPU-path assertion, which skips when no usable CUDA device is present at runtime and lets the comparison run CPU-vs-CPU, now calls `helios::requireGPUOrFail()` first so that skip is a failure on a runner dedicated to GPU coverage.

## Project Builder
- `SKIP_IF_NO_GPU()` now calls `helios::requireGPUOrFail()` before skipping, for the same reason.

## Build System
- Added `run_tests.sh --nogpu` and `--requiregpu`, which export `HELIOS_NO_GPU=1` and `HELIOS_REQUIRE_GPU=1` respectively, so a GPU-equipped development machine can reproduce the non-GPU CI runners exactly, or a GPU-dedicated runner can be made to fail rather than skip. `--nogpu` was documented in `CLAUDE.md` in three places but had never been implemented — the script rejected it as an unknown option — so the one prescribed way to catch an unguarded GPU test locally did not exist; it is only faithful because this release makes `HELIOS_NO_GPU` a veto across the whole backend layer (previously the variable suppressed the availability query while `create("auto")` still built a backend). The two flags are mutually exclusive and the script rejects the combination rather than letting one win. `.gitignore` now covers `utilities/.scratch/`, the prescribed location for throwaway `--project-dir` debug builds.
- Added `utilities/lint_gpu_tests.sh`, which detects test cases that construct a `RadiationModel` without being declared `GPU_TEST_CASE`, and runs in the Linux workflow before the build so this class of failure is reported in seconds instead of after the full build-and-test cycle. Detection covers subclass probes that derive from `RadiationModel` to reach a protected member and so never name the type in a declaration of their own. A test that must genuinely run without a GPU — one asserting that construction against a broken driver throws cleanly, or reporting which backends are compiled in — is exempted with a `GPU-LINT-OK: <reason>` comment in its body; the three such tests are annotated.
- The Linux and Windows GPU workflows now verify the GPU before building (`nvidia-smi`, failing with an explanation that the CUDA toolkit can still compile in this state) and run the suite with `--requiregpu`. The first check catches a dead driver in seconds; the second catches a healthy driver that Helios still cannot use, which no hardware probe would reveal.

# [1.3.78] 2026-07-26

## Core
- Fixed three `writeXML()`/`loadXML()` corruptions of primitive/global data. `writeDataToXMLstream()` wrote `HELIOS_TYPE_INT4` data inside a `<data_int3>` opening tag closed by `</data_int4>`, producing malformed XML that failed to reload; the tag is now `<data_int4>`. The per-object primitive-data writer in `writeXML()` emitted the elements of array-valued and multi-component data (`float`/`double`/`uint`/`int`/`vec2`–`vec4`/`int2`–`int4`/`string`) with no delimiter between consecutive values, silently concatenating them into unparseable numbers on reload; a trailing space now separates every element. And reloading a file whose `<polymesh>` object ID was already present in the target Context remapped `objID` to a fresh Context ID *before* using it to key `object_prim_UUIDs`, throwing `std::out_of_range`; the remap is removed (the Context object ID is assigned by `addPolymeshObject()`), and a `<polymesh>` block referencing no primitives now fails fast with a clear error instead (verified by new int4/vector-data and polymesh-reload round-trip self-tests).
- Fixed `CompoundObject::rotate()` rotating objects about the z-axis in the opposite direction to `rotatePrimitive()` and to the `vec3`-axis `rotateObject()` overload: the "z" branch alone negated the rotation angle. The negation is removed so all three rotation paths share one handedness (verified by a new z-axis rotation-consistency self-test); callers that were compensating for the flipped azimuth should be corrected.
- Fixed non-z-axis rotations being silently applied to voxels despite the "Voxels can only be rotated about the z-axis" warning. The three string/`vec3`-axis batch `rotatePrimitive()` overloads warned but fell through and applied the transform anyway (a missing `continue`), and `Voxel::rotate()` ignored the requested axis entirely and always rotated about z; both now skip the rotation for a non-z axis (and `Voxel::rotate()` fails fast on an axis string other than x/y/z), while z-axis rotations are still applied (verified by a new voxel rotation-restriction self-test).
- Fixed the color overloads of `addDisk()`/`addDiskObject()` leaving the first triangle of each outer disk ring (`Ndivs.y >= 2`) at the world origin. Each ring iteration adds two triangles but only the last was rotated/translated into place; the transform is now applied to every triangle added in the iteration (verified by a new multi-ring disk self-test).
- Fixed `getDomainBoundingBox()` (over an explicit UUID list) failing to expand the upper bound of an axis when the same vertex set a new lower bound, because the min and max tests were chained with `else if`; they are now independent `if`s.
- Fixed `addTimeseriesData()` terminating with an "unknown reason" runtime error when inserting a point at a date/time already present in the series. The duplicate check was buried in the interior-insertion loop (which never runs for a duplicate matching the first/last point) and used `continue` in a context that fell through to the error path; it now checks the whole sorted series up front and warns-and-skips the duplicate.
- Fixed several `Context` data operations mishandling non-`float` types: `aggregatePrimitiveDataProduct()` had the initialize-vs-accumulate branches inverted for every type except `float` (so `double`/`int`/`uint`/`vec2`–`vec4`/`int2`–`int4` products were formed from a zero/garbage seed or overwritten to the last value); the all-primitives `duplicatePrimitiveData()` overload was missing the `INT4` branch (registering the copied label's type without copying its value, and now failing fast on any other unsupported type); and `calculatePrimitiveDataAreaWeightedMean()` counted a NaN-area primitive into the running sums (contradicting its own "excluded from calculations" warning) and left the output unassigned when a NaN was seen — the NaN primitive is now `continue`d over and the warning decoupled from the empty-set case (verified by new non-float-product, `int4`-duplicate, and NaN-area self-tests).
- Fixed textured primitives written with the v3 `<material>` XML format losing their texture `(u,v)` coordinates across a `writeXML()`/`loadXML()` round-trip. The material-format read branches in `loadXML()` create the patch/triangle without UVs, so the UVs parsed from the file were dropped on load; `loadXML()` now re-applies them with `setTextureUV()` after the primitive is created. Also made `writeXML()` emit primitives in ascending UUID order (rather than the nondeterministic order of the underlying `unordered_map`) so the output is deterministic and a compound object's sub-primitives — created with contiguous ascending UUIDs — keep their relative order (e.g. the row-major layout of tile sub-patches) across a round-trip; `CompoundObject::deleteChildPrimitive()` was correspondingly changed from swap-and-pop to an order-preserving `erase()` so a deletion leaves the remaining sub-primitive UUIDs sorted (the linear `find()` it already performs makes this no more than the existing O(n), verified by new texture-UV-preservation and post-deletion sub-primitive-ordering round-trip self-tests).

## Collision Detection
- Fixed `buildBVH()` over a caller-specified UUID subset silently widening back to all Context geometry on the next automatic rebuild. Because this plugin never clears the Context dirty flags, every primitive outside the requested subset stayed flagged dirty; `ensureBVHCurrent()` read that as "geometry changed" and rebuilt over `context->getAllUUIDs()`, so a ray could return a primitive the caller had explicitly excluded. The build now records whether it was restricted and which primitives that subset deliberately left out (`bvh_geometry_restricted`/`bvh_excluded_geometry`), and an automatic rebuild carries the exclusion set forward — still absorbing primitives added to the Context afterward, still dropping deleted ones, but never resurrecting an excluded primitive; `markBVHDirty()` clears the restriction so a subsequent unrestricted `rebuildBVH()` rebuilds over everything, and `isBVHValid()` no longer counts a deliberately-excluded primitive's absence as staleness (verified by new restricted-survives-rebuild and unrestricted-absorbs-new-geometry self-tests).
- Fixed the BVH failing to rebuild when an existing primitive was moved, rotated, or scaled in place. Such a primitive keeps its UUID and its stale build-time entry in `primitive_aabbs_cache`, so `buildBVH()`'s cache-refresh test (which consulted only the never-populated `dirty_primitive_cache`) reused the old bounding box and the BVH kept bounding the geometry's former position; `isBVHValid()` likewise compared only UUID sets and reported a moved-geometry BVH as valid. `buildBVH()` now also refreshes any cache entry the Context reports dirty via `getDirtyUUIDs()`, and `isBVHValid()` compares each cached build-time AABB against the primitive's current bounding box so an in-place move is detected directly (verified by a new moved-geometry-reported-stale self-test).
- Fixed `CollisionDetection` being implicitly copyable despite owning eleven raw CUDA device pointers that its destructor frees. The class declared no copy or move operations, so the compiler-generated ones did a memberwise pointer copy: copy-construction left two objects holding the same device pointers with `gpu_memory_allocated` true, so the second destructor double-freed them, while copy-assignment overwrote the target's live pointers and cleared its allocation flag, orphaning those device buffers behind `freeGPUMemory()`'s `if (!gpu_memory_allocated) return;` early exit. All four operations are now explicitly `= delete`d, so a copy is a compile-time error rather than a runtime double-free — hold instances by pointer or reference, as the LiDAR and plant architecture plugins already do (verified by a new non-copyable-device-ownership self-test).
- Fixed the voxel ray-path-length GPU fallback leaving the resident scene buffers allocated for the object's remaining lifetime. On `calculateVoxelRayPathLengths_GPU()` failure the fallback cleared `gpu_acceleration_enabled` directly instead of calling `disableGPUAcceleration()`, so the device-side BVH, geometry, mask, and UV buffers were never released, and the now-false flag also wedged `transferBVHToGPU()` into its early return so GPU residency could never be restored. The fallback now routes through `disableGPUAcceleration()`, which owns both halves of that transition.
- Made `optimizeLayout()` fail fast via `helios_runtime_error()` instead of printing a warning and returning `0`. The function is unimplemented, and its documented return value is the post-optimization collision count, so a `0` return was indistinguishable from a genuine "optimization eliminated every collision" — and was silenced entirely by `disableMessages()`; it now throws a message directing callers to `findCollisions()` in the meantime (verified by a new fail-fast self-test). Also removed dead unused ray-tracing helpers (`countRayIntersections()`, `findNearestRayIntersection()`, and the never-called SIMD ray-AABB/BVH-traversal paths).

## Radiation
- Fixed two-sided emitting primitives double-counting their top face's scattered-direct energy onto their bottom face. In `runBand()` the two-sided branch deposited `flux_bottom += flux_top`, but `flux_top` already accumulates the top face's scattered direct radiation in addition to its emission, so the bottom face re-emitted the top face's reflected/scattered energy into the scene, violating energy conservation. The bottom face now emits only its own emission term (`flux_bottom += out_top`), matching the SIF branch above and the camera-scatter accumulator below (verified by a new self-test that a black sensor fully shadowed beneath a two-sided reflecting patch reads a source-independent flux — it jumped ~10x with the source on under the bug, and is now source-independent).
- Fixed the direct source-flux upload skipping the auto-generated SIF excitation bands when a regular band is dispatched with a SIF camera registered. The fill loop iterated `b < label.size()` but wrote into a per-source flux array sized and indexed by `Nbands_launch` (the full alphabetically-ordered launch set, into which `runBand()` piggybacks the `_SIF_exc_*` excitation bands); when `Nbands_launch > label.size()` the trailing excitation-band slots were left at zero, uploading zero direct flux for every excitation band, giving zero APAR and zero fluorescence. The loop now iterates the full `Nbands_launch` (verified by a new self-test that runs a regular band before the SIF bands and confirms the sensor receives nonzero SIF flux in both channels).
- Fixed the Vulkan BVH traversal shader silently pruning subtrees a ray actually passes through when the ray is axis-aligned. The slab test precomputed `1.0/ray_dir`, which is `±inf` for a zero direction component; when the ray origin lies exactly on a slab plane the `(plane − origin) * inf` term becomes `0*inf = NaN` and the `t_near <= t_far` comparison evaluates false, dropping the node — common with axis-aligned geometry (flat ground at z=0, horizontal leaves). A new `safe_inv_dir()` substitutes a large finite reciprocal for near-zero direction components (so `0*large = 0` keeps the term finite), applied in both `traverse_bvh()` and `traverse_cwbvh()`, mirroring the CUDA backend's `safeRayInvDir()` fix.
- Fixed the Vulkan radiation backend aborting a large scene with `VK_ERROR_DEVICE_LOST` on macOS/Metal, where the GPU execution watchdog kills a single command buffer that covers every primitive in one `vkCmdDispatch` (most exposed on diffuse launches, which trace `rays_per_primitive` hemisphere rays for every primitive). `launchDirectRays()` and `launchDiffuseRays()` now split the launch into batches of primitives sized by a fixed `max_rays_per_launch` ray budget (via the new `primitivesPerLaunchBatch()`), recording and submitting one command buffer per batch — mirroring the OptiX 6 backend's batching. The batching is numerically transparent (results accumulate into the same buffers via atomic adds and both ray-generation shaders already resolve their primitive from `launch_offset + batch index`); the budget is chosen to keep typical scenes in a low-batch regime (each batch adds ~0.6 ms of fixed record/submit/fence-wait overhead) while staying far below any launch size observed to trigger device loss (verified by a new self-test that a uniformly-lit tile of many patches receives uniform flux, so no batch is skipped or double-counted).
- Added a terrain-following overload `LiDARcloud::addGrid(center, size, ndiv, rotation, column_z_offsets)` that shifts each vertical voxel column in z by a per-`(x,y)`-column offset (row-major as `[j*ndiv.x + i]`, one value per column) so a grid can track an external terrain surface such as a DEM. The offset is applied as a pure vertical shift to the stored un-rotated lattice center (rotation is applied downstream) and is also recorded on each cell as its `ground_height`; the offsets vector is validated to have length `ndiv.x*ndiv.y` and fails fast otherwise. The original four-argument `addGrid()` now delegates to this overload with an empty offsets vector, which is byte-for-byte identical to the axis-regular grid built previously.
- Guarded the OpenMP thread-introspection calls (`omp_get_max_threads()`/`omp_get_thread_num()`) in the `syntheticScan()` hit-data-label collection with `#ifdef _OPENMP` so the plugin compiles when built without OpenMP (the parallel label-union step falls back to a single thread-local set), rather than failing to compile on the missing intrinsics.
- Made `LiDARcloud::getCellCenter()` and `getCellRotation()` report values in the public world frame and unit convention, rather than the internal lattice frame they previously leaked. For a grid created with a non-zero azimuthal rotation, `getCellCenter()` previously returned the raw un-rotated lattice center stored in `GridCell.center`, which did not lie in the same rotated world frame as the hit points, scan origins, and grid bounding box; it now rotates that center about the grid anchor (about +z) so the returned coordinate matches every other exposed coordinate (the un-rotated grid keeps a fast path). `getCellRotation()` now returns the rotation in degrees — matching the degrees expected by `addGrid()` — instead of the radians it was previously stored and returned in. The un-rotated lattice center and the radian rotation remain a load-bearing internal convention: the many hot paths that build an axis-aligned cell AABB and inverse-rotate the query point (`getContainingGridCell()`, `getGridBoundingBox()`, `calculateHitGridCell()`, `calculateLeafArea_inner()`, `calculateSyntheticLeafArea()`, `backfillLeavesAlphaMask()`) were repointed to two new private getters, `getCellCenterUnrotated()` and `getCellRotationRadians()`, so they read the stored values directly and are unaffected by the public-getter change. The grid wireframe (`addGridWireFrametoVisualizer()`) was also rewritten to rotate the eight box corners about the anchor so it lines up with the rotated voxels (verified by new rotated-grid center, anchor round-trip, degrees round-trip, wireframe-corner, and LAD rotation-invariance self-tests). Also corrected a copy-paste function name in `getCellSize()`'s out-of-range error message (it reported `getCellCenter`).
- Fixed `LiDARcloud::getGridBoundingBox()` returning an under-sized bounding box for an azimuthally-rotated grid. It rotated only the cell box's min and max corners and min/maxed over those two points, but rotating an axis-aligned box's two extreme corners does not bound the rotated box — for a 0.6 m cell at 30 degrees the chosen diagonal maps onto the y-extremes, leaving the reported x-extent 3.7x too small (and min/max swap outright past 90 degrees). Because `syntheticScan()` derives its ray cull from this box when `scan_grid_only` is set, the missing extent silently discarded most of the beams approaching a rotated voxel along the clipped axis, biasing the transmission ratio, path-length samples, and `Gtheta` that feed the LAD inversion. All eight corners are now rotated and enclosed, matching `addGridWireFrametoVisualizer()` (whose comment already incorrectly claimed the two agreed). Also corrected the strict guard in the LAD rotation-invariance self-test, whose voxel was a 0.6 m sub-volume of the 1 m leaf cube rather than the enclosure its own comment claimed, so it was measuring sub-volume sampling noise on top of the real defect; at the enclosing 1.5 x 1.5 x 1.2 m it now reports a rotation-invariant LAD and fails by 10x the tolerance if the eight-corner enclosure is reverted.
- Recorded per-band thermal-emission state so downstream plug-ins can identify which band governs longwave emission. `runBand()` now writes global data `emission_enabled_<band>` (`uint`, 1/0) for every radiation band from that band's `emissionFlag`, which the energy balance model reads to select the emitting band's emissivity.

## Energy Balance
- Fixed the surface-energy-balance solver writing an uninitialized temperature back to the primitive when the user-supplied initial `temperature` equaled the solver's hardcoded second initial guess of 400 K. The secant method seeds two initial guesses (`To[p]` and 400 K); when they coincide the two residuals are identical, the degenerate-case branch is taken on the first iteration, and the (previously uninitialized on both the CPU and GPU paths) result variable is written out as the surface temperature. The second guess is now offset when it collides with the first, and the result variable is initialized to the current best guess (verified by a new self-test that an initial temperature of 400 K yields the finite radiative-equilibrium temperature). Also replaced the NaN-only (`T != T`) writeback guard with an `std::isfinite()` check so an infinite result is likewise reset to the default temperature.
- Fixed the moisture-conductance term evaluating `0/0` (NaN latent flux) when the boundary-layer conductance `gH` is zero (e.g. zero wind) with a non-zero stomatal conductance and a stomatal sidedness of 0 or 1, which makes one denominator of the series-conductance expression vanish. The previous guard only zeroed the conductance when both `gH` and `gS` were zero; it now short-circuits to zero whenever either is zero (a zero boundary-layer or stomatal conductance means no vapor pathway) before evaluating the expression, on both the CPU and GPU paths (verified by a new self-test that `gH = 0` with a non-zero `gS` gives a finite, zero latent flux).
- Fixed thermal emission using an ambiguous or wrong emissivity when the energy balance ran over multiple radiation bands. The prior code overwrote a single per-primitive emissivity once per band, so the last band's `emissivity_<band>` silently won regardless of which band actually emits. It now uses the emissivity of the single band for which emission is enabled (read from the `emission_enabled_<band>` global data written by the RadiationModel), warning and using the first emission-enabled band if more than one emits; when no emission information is available (fluxes set manually without running the RadiationModel) it falls back to the first band that defines an emissivity, warning if more than one does (verified by new single-band, multiple-emission-band, and manual-flux fallback self-tests).

# [1.3.77] 2026-06-24

## Core
- Fixed `setTileObjectSubdivisionCount()` flipping or mis-rotating a tile object when changing its subdivision count. Both overloads reconstructed the tile's pose from its normal via `cart2sphere()` and replayed it as elevation/azimuth rotations — a lossy round-trip that discarded any in-plane rotation and, for a horizontal (ground) tile, produced `elevation=90°` and tilted the regenerated tile to vertical. The regeneration was unified into a new private `regenerateTileObjectSubpatches()` that builds the new sub-patches in canonical unit-tile space and maps them through the tile's existing transformation matrix, so position, size, orientation (including in-plane rotation) and color/texture are preserved exactly, with the textured and non-textured paths now sharing one code path (the prior duplicate per-texture template machinery removed) and the subdivision count read back from the template in case it is down-corrected for a low-resolution texture. The `float area_ratio` overload now fails fast via `helios_runtime_error()` when given a ratio below 1 (it is the ratio of whole-tile area to sub-patch area and so cannot be less than 1) instead of producing a degenerate subdivision (verified by new flat-tile and arbitrary-orientation preservation, area-ratio, and fail-fast self-tests).
- Fixed `setTileObjectSubdivisionCount()` silently discarding per-sub-patch primitive data when changing a tile object's subdivision count. The regeneration built the new sub-patches as blank copies of a fresh template, so any data set on individual sub-patches via `setPrimitiveData()` (e.g. the radiation colorboard's per-patch `reflectivity_spectrum`/`twosided_flag`, or LiDAR per-primitive fields) was lost. `regenerateTileObjectSubpatches()` now transfers per-sub-patch primitive data from old to new sub-patches: because the subdivision count may change, each new sub-patch inherits all data from the old sub-patch whose grid cell contains the new sub-patch's center (an exact index-for-index copy when the count is unchanged), computed from the canonical-space template patch centers without inverting the transform. A fast path skips the transfer entirely when no old sub-patch carries any data, so uniform/no-data tiles are unchanged in cost and behavior (verified by new 1:1, refine, coarsen, and multi-type primitive-data preservation self-tests).

## Collision Detection
- Sped up the CPU BVH ray-tracing path that backs the LiDAR synthetic scan by ~15x on a multi-return scan (12.7s → 0.86s at 100 rays/pulse on a deterministic homogeneous canopy) with a bit-identical point cloud (position and intensity checksums match). The longest-axis median BVH split was replaced with a binned-SAH build (leaf cap 8, depth backstop 64, previously depth 6/10 with 100/500-primitive leaves) — result-neutral on closest hit but the single biggest contributor; the per-primitive `unordered_map::find` was removed from the innermost traversal loop via a dense, BVH-leaf-ordered primitive cache (`primitive_cache_dense`, indexed directly by leaf slot, carrying each entry's UUID, and kept in lockstep with the UUID-keyed cache by `ensurePrimitiveCacheCurrent()`/`rebuildDensePrimitiveCache()`); `castRaySoATraversal` gained a fixed-size traversal stack (no per-ray heap allocation) with near-child-first ordering; and a new coherent packet path (`castRaysSoA_packets()`/`castPacketSoATraversal()`) traverses a pulse's contiguous sub-rays against one shared BVH stack with a per-leaf active-ray mask (each leaf primitive fetched once and tested only against rays whose AABB reaches the node — only ~1/3 of a pulse's sub-rays reach a given leaf) and near-first child ordering, amortizing node/primitive fetches for ~1.4x over per-ray (bit-identical results). The finer SAH leaves surfaced a latent `0*inf=NaN` slab-test bug for axis-aligned rays lying exactly on an AABB face (spuriously rejecting the box), now fixed with explicit parallel-axis handling in `aabbIntersectSoA` (scalar and SSE) and `aabbEntryDistanceSoA` (verified by a new packet-vs-per-ray equivalence self-test).
- Fixed two GPU ray-tracing correctness regressions exposed by the deeper SAH BVH trees (the GPU synthetic-scan path delegates to these kernels): the per-thread BVH traversal stack was moved to thread-local memory at capacity 128 (`BVH_TRAVERSAL_STACK_CAPACITY`) — the previous 32-deep block-shared stack silently dropped children on the deeper trees, missing or mis-reporting hits — with a device overflow flag now raised atomically and checked host-side after each launch (`rayPrimitiveBVHKernel`/`bvhTraversalKernel`) so a deeper-than-expected tree fails fast via `helios_runtime_error()` instead of silently dropping primitives; and `safeRayInvDir()` now takes a sign-preserving reciprocal that fixes the same `0*inf=NaN` hazard for axis-aligned rays in `warpRayAABBIntersect`/`rayVoxelIntersect`.
- Added GPU texture-transparency sampling to the ray-tracing kernel so a ray striking the transparent background of a textured primitive (e.g. a leaf-patch PNG with an alpha channel) is rejected and traversal continues to the geometry behind it, matching the CPU `isHitTexelOpaque()` path; previously the GPU kernel did purely geometric intersection and registered such hits as solid, diverging from the CPU path and corrupting the LiDAR GPU synthetic scan. A per-scene texture-transparency Structure-of-Arrays (masks de-duplicated by texture file, plus per-primitive mask/UV indices) is emitted by `buildGPUGeometrySoA()` and uploaded alongside the resident geometry by `transferBVHToGPU()` — the mask pixel data is skipped entirely on scenes with no transparency — and sampled on the device by new `isHitOpaqueGPU()`/`sampleMaskOpaqueGPU()` helpers that reproduce `isHitTexelOpaque()` exactly (patch basis projection, triangle barycentric, UV wrap, Y-flip, clamp), verified by a 1.05M-ray CPU-vs-GPU parity test over a wall of transparent leaf patches (0 mismatches, with 40% of rays correctly passing through to a backstop).
- Added a `CollisionDetection::isGPUAvailable()` static query that reports whether a usable GPU is actually present, complementing the existing `isGPUAccelerationEnabled()` toggle-state getter (the plugin previously exposed no way to distinguish "compiled without CUDA" from "CUDA compiled but no device found"). Mirroring the radiation plugin's `isGPUBackendAvailable()`, it returns true only when the plugin was built with CUDA support, `cudaGetDeviceCount()` finds at least one device, and the `HELIOS_NO_GPU` environment variable is unset; the result is cached after the first probe. The plugin now also actually honors `HELIOS_NO_GPU` (previously only the radiation plugin did): the constructor default, `enableGPUAcceleration()`, `allocateGPUMemory()`, and `calculateVoxelRayPathLengths_GPU()` were all routed through `isGPUAvailable()` so the env-var veto and hardware presence gate the real execution path through a single source of truth — the reported availability therefore can never diverge from the path actually taken (setting `HELIOS_NO_GPU=1` makes `enableGPUAcceleration()` a no-op and forces the CPU path). The query reports GPU capability/permission; whether a given call uses the GPU additionally depends on the existing batch-size/scene-complexity thresholds in `shouldUseGPU()` (verified by a new self-test covering determinism and the post-enable `isGPUAccelerationEnabled() == isGPUAvailable()` invariant).

## LiDAR
- Redefined the synthetic-scan `deviation` hit field as a physically-meaningful, dimensionless pulse-shape deviation metric analogous to the RIEGL "pulse shape deviation" confidence value (small for a clean single-surface return, large for a broadened/sloped/mixed return). It was previously the absolute difference between a return's range and the unweighted mean range of all of a pulse's returns — a quantity with no physical meaning that was identically zero for the common single-return case. It is now computed per return from that return's `echo_width` (the pulse-width-convolved range spread, `echo_width = sqrt(range_resolution² + var)`) as the excess broadening beyond the transmit pulse, `sqrt(echo_width² − range_resolution²)`, normalized by `range_resolution` to make it dimensionless; misses, clean returns, and the degenerate no-pulse-width case report 0 (verified by a new self-test that a flat perpendicular wall yields ~0 deviation while a steeply tilted wall whose footprint samples a range interval yields a meaningfully larger value).
- Reworked the synthetic-scan beam-footprint sampler from "draw sub-rays uniformly over a disk, then re-weight each by a Gaussian envelope" to importance-sampling the Gaussian directly, and made the default detection threshold a non-zero noise floor. Each sub-ray's radial offset is now drawn FROM the Gaussian irradiance profile \f$I(r)\sim\exp(-2(r/r_0)^2)\f$ via its Rayleigh inverse-CDF — the profile lives in the sample density, so every sub-ray carries unit weight (`subray_weight` stays 1, retained only as a hook for future non-Gaussian profiles) and the beam wings are sampled in proportion to their energy instead of being wasted on near-zero-weight outer rays or hard-truncated at the \f$1/e^2\f$ radius. Coverage is kept even and stripe-free with stratification: linear radial strata + golden-angle azimuth for the divergence cone, a base-2 van der Corput radial sequence + golden-angle azimuth for the exit aperture (the two independent radial dimensions use different sequences so the joint footprint develops no diagonal artifact), with per-beam Cranley-Patterson rotations so the pattern decorrelates pulse-to-pulse. The radial CDF is truncated at the detectability radius (where the irradiance falls below the detection threshold, or the negligible outer 0.1% — ~1.86 \f$r_0\f$ — when no threshold is set), removing the old hard \f$3 r_0\f$ clamp and its tail pile-up. The RNG is now consumed per-beam (two draws each for the divergence and aperture dimensions: a radial stratum shift and an azimuth base) rather than per-sub-ray, drawn serially before the OpenMP generation loop so seeded scans stay reproducible across thread counts — this intentionally changes the sub-ray pattern and is NOT bit-identical to the previous sampler. The default `ScanMetadata::detectionThreshold` was changed from `0` to `0.05`, a ~5% noise floor that pairs with the recommended ~40 rays/pulse to suppress the single-sub-ray "phantom" returns whose Bernoulli appearance/disappearance otherwise forced very high ray counts to converge (set to `0` to restore the previous "report every return" behavior). Verified by new stratified-sampler-invariant and default-threshold self-tests; existing leaf-area-inversion and geometry self-tests that need complete returns now set the threshold to `0` explicitly.
- Added `LiDARcloud::setSyntheticScanProgressPointer(volatile int *)` so a host can poll per-scan `syntheticScan()` progress from another thread: the caller-owned counter is written with the 0-based index of the scan currently being ray-traced (updated at the top of the scan loop, before any early "no rays hit the bounding box" continue, so it advances even for scans that hit nothing) and set to `getScanCount()` when the batch finishes; `nullptr` clears it and the counter's lifetime is the caller's (verified by a new self-test covering ordered advancement, the no-geometry early-continue path, and that clearing leaves a caller-owned counter untouched).
- Parallelized the `syntheticScan()` host post-processing pipeline, previously serial and the dominant cost once the GPU traces a scan in milliseconds. Per-beam post-processing now runs in parallel into per-beam buffers that are merged into the columnar hit storage via a pre-sized, beam-order parallel scatter (each hit owns a distinct precomputed row and each label a distinct column slot, so the writes are race-free and order-preserving), ray generation (per-sub-ray divergence/aperture trig) is parallelized with the divergence-cone/aperture RNG samples pre-drawn serially in the original order so seeded scans remain bit-identical, and `addHitPoint()` now takes `ScanMetadata` by const reference to avoid a per-hit copy. The coherent CPU packet traversal is wired in through a new `castRaysUnified(..., packet_size)` overload so a pulse's contiguous sub-rays (`packet_size = Npulse`) route to `CollisionDetection::castRaysSoA_packets()`.
- Made `LiDARcloud::triangulateHitPoints()` cancellable via the existing `setCancelFlag(volatile int *)` mechanism so a long, multi-million-point triangulation can be aborted from another thread instead of running to completion (previously it was one uninterruptible blocking call). Both overloads now poll the caller-owned flag at every loop boundary — between scans, inside the per-scan hit-gather loop (coarsely, ~every 128k hits), immediately before and after the Delaunay `triangulate_CDT()` call (which bounds the worst-case latency to a single scan's tessellation without touching the cross-platform-deterministic CDT internals), and inside the triangle-build loop. On cancellation the partially-built mesh is discarded and the call returns an empty triangulation (`getTriangleCount() == 0`, `triangulationcomputed` left false) rather than a meaningless half-mesh; a non-cancelled run produces the identical triangle set as before (verified by a new self-test that a flag set before the call yields zero triangles while the same input with the flag clear matches the unflagged baseline). The per-iteration `volatile` read is effectively free and `nullptr` clears it.
- Reduced the transient memory footprint of `LiDARcloud::triangulateHitPoints()` with no change to the output mesh, diagnostics, or public API. The per-scan point buffer was being fully duplicated into a `pts_copy` that was never read afterward (a dead copy of the largest live buffer, present identically in both overloads); it is removed, and the now-dead de-duplication index list and the point buffer itself are released (swap-to-empty) before the triangle-build loop rather than lingering through it, and the CDT adapter (`triangulate_CDT()`) frees its local vertex-coordinate copy immediately after `insertVertices()` copies it into the library's own buffer. To make these changes once instead of in two ~400-line near-identical bodies (the `pts_copy` dead copy had been duplicated across both overloads for exactly this reason), the shared per-scan "second pass" — gather, wrap/snap, de-duplicate, Delaunay call, and edge-length/aspect/separation filtering — was extracted into a single private `triangulateScanSecondPass()` that both overloads call, with the only per-overload difference (the two-argument overload's first-return auto-filter vs. the filtered overload's scalar-field gate) selected by a `scalar_field == nullptr` argument. Peak transient memory per scan is unchanged in asymptotics (still bounded by the largest single scan's Delaunay tessellation) but lower in constant factor; the output is bit-identical (verified by a new self-test asserting a deterministic scene reproduces an exact triangle count and vertex/area checksum across independent runs, that the diagnostics reconcile `candidates == kept + dropped_lmax + dropped_aspect + dropped_degenerate`, and that both overloads agree).
- Fixed the flaky "LiDAR Exit Diameter - Comparative Spread Test" self-test, which intermittently failed its `extent_exit > extent_point * 1.01` assertion. The test ran a point-source and an exit-diameter `syntheticScan` without seeding the Context RNG, so it compared two independent random beam-footprint realizations against a hard threshold; because the spread metric is a bounding box over a full hemispherical scan (driven by extreme outliers), an unlucky draw could put the point-source spread above the exit-diameter spread. The Context RNG is now seeded with the same seed before each scan so both share an identical sampling realization and the comparison is deterministic (ratio ~1.026, comfortably above the 1.01 threshold), isolating the physical effect of the exit diameter from RNG noise.
- Added a rotating-Risley-prism (Livox-style rosette) synthetic-scan pattern (`SCAN_PATTERN_RISLEY_PRISM`/`SCAN_MODE_RISLEY_PRISM`), the optical mechanism of Livox rosette sensors such as the Mid-40/Mid-70/Avia: a single beam is refracted through a stack of continuously rotating wedge prisms (each a new `RisleyPrism` with a wedge angle, glass refractive index, signed rotor rate, and phase), so that a counter-rotating pair with incommensurate rates traces the characteristic non-repetitive rosette filling a circular field of view. Each pulse's body-frame direction is computed by non-paraxial ray tracing — the vector form of Snell's law applied at both faces of every rotating wedge — so the field of view is an emergent property of the wedge angles and refractive indices rather than a specified parameter (validated against the Livox Mid-40 datasheet, ~38° circular FoV). Because the pattern is non-separable (each pulse has its own direction as a function of time, not a row×column angular grid) it is stored as a single-row `Ntheta=1`, `Nphi=Npulses` table and is always trajectory-driven, riding the existing moving-platform path; a stationary capture is a trajectory of two coincident poses separated in time by the acquisition duration, exactly as for a spinning scan. Added `LiDARcloud::addScanRisley()` (quaternion and roll/pitch/yaw-trajectory overloads, mirroring `addScanSpinning()`) which derives the pulse count from the PRF and trajectory duration and delegates to the moving-platform binder, the `getScanRisleyPrisms()`/`getScanRisleyRefractiveIndexAir()` getters, the `rc2direction()`/ray-generator branches (with `direction2rc()` returning `(0,0)` since the pattern has no invertible row/column grid), and the `risley` scanPattern with `<prism>` and `<refractiveIndexAir>` XML tags that round-trip through `loadXML()`/`exportScans()`. Only the rotating-Risley-prism rosette is modeled; Livox's deterministic line-scan mode and the newer MEMS/galvanometer non-repetitive patterns are different mechanisms left for future scan patterns (verified by new FoV-containment, non-repetition, fail-fast, and XML round-trip self-tests).
- Made the synthetic raster scan model the continuous azimuth rotation of a real terrestrial scanner. Previously each zenith column was swept at a single fixed azimuth before stepping to the next discrete azimuth, producing perfectly vertical columns; real instruments sweep the beam vertically with a fast mirror while the head rotates continuously in azimuth, so the azimuth advances during each zenith sweep and the columns are slightly skewed (tilted). The raster ray generator in `syntheticScan()` now drifts the azimuth across each zenith column by exactly one column step (`dphi = (phiMax-phiMin)/(Nphi-1)`, per-row increment `dphi/Ntheta`), so column `j` tiles seamlessly into column `j+1` with no azimuth gap or overlap. The drift applies only to the raster pattern (spinning-multibeam columns still fire at one azimuth and step between firings; Risley is non-separable) and composes naturally with the moving-platform path since it is part of the body-frame beam direction. It is intentionally not reflected in the nominal `rc2direction()`/`direction2rc()` grid mapping (the drift is sub-cell, so hit-point binning is unaffected); the `gapfillMisses()` per-row `azimuth = intercept[row] + slope[row]*column` reconstruction absorbs the constant per-row offset and is unchanged. Single-row (`Ntheta=1`) and single-column (`Nphi=1`) scans produce no skew. Verified by a new self-test asserting the per-row azimuth drift, seamless column-to-column tiling, and the two degenerate no-skew cases.
- Extended synthetic-scan hit-point labeling to object data. A non-standard label listed in the scan column format (`<ASCII_format>`) is now resolved from the intersected primitive's primitive data first, and on a miss falls back to the object data of the primitive's parent compound object, so a hit can be labeled with a field carried per-primitive or by the parent object (e.g. a per-object classification or branch order). Previously only primitive data was queried, silently leaving such labels unrecorded. When the same label exists on both, the more specific per-primitive value wins; only scalar types (float/double/int/uint) are transferable and non-scalar types or labels absent from both sources are skipped for that hit. The per-label resolution was factored into a `resolveScalarHitData()` helper shared by both sources, with the object-data lookup guarded by `doesObjectExist()` so object-less primitives (parent id 0) are handled without throwing (verified by new object-data-labeling and primitive-over-object precedence self-tests).
- Added `LiDARcloud::isGPUAvailable()` and `LiDARcloud::isGPUAccelerationEnabled()` so the GPU state is queryable from the LiDAR API (which previously exposed only the `enableGPUAcceleration()`/`disableGPUAcceleration()` setters). `isGPUAvailable()` forwards to the new `CollisionDetection::isGPUAvailable()` static probe — valid even before the collision-detection instance is lazily created — reporting whether a usable CUDA GPU is present (and not vetoed by `HELIOS_NO_GPU`), while `isGPUAccelerationEnabled()` reports whether acceleration is currently toggled on (falling back to the availability-based default when the instance does not yet exist). Because the underlying collision-detection plugin now honors `HELIOS_NO_GPU`, setting that variable forces the LiDAR synthetic scan onto the CPU path and `isGPUAvailable()` reports it accordingly (verified by a new self-test asserting the forward matches the static probe and the enabled-state default is consistent with availability).
- Fixed the scanner azimuth offset leaning a tilted synthetic scan the wrong way once a non-zero heading was set. The roll/pitch body frame was built with `heading = phiMin + scanAzimuthOffset`, but the azimuth offset is applied to the ray directions as a right-hand (CCW) rotation about world +z while the azimuth angle `phi` is measured CW-from-+y, so advancing the heading by the offset must *subtract* it (rotating the azimuth-zero direction `(sin phiMin, cos phiMin)` CCW by the offset gives `(sin(phiMin−offset), cos(phiMin−offset))`). The `+` form reflected the body frame about the un-offset heading, so any roll/pitch tilt was displaced in the wrong direction relative to the rays; `heading` is now `phiMin − scanAzimuthOffset` so the tilt frame and the rays rotate together (the affected self-test now pins the handedness, asserting a 90° offset plus a right-hand pitch displaces the near-nadir floor hit to −y rather than +y).
- Made `prepareUnifiedRayTracing()`, `finishUnifiedRayTracing()`, and `castRaysUnified()` private, as they are internal helpers of the synthetic-scan ray-tracing pipeline and not part of the public `LiDARcloud` API.

## Plant Architecture
- Fixed a crash in `advanceTime()` when a plant contained a branch pruned at node 0 by `pruneBranch()`. Pruning deletes a shoot's phytomers and its internode tube object but leaves the now-empty `Shoot` in the `shoot_tree` (and a stale entry in the parent's `childIDs`), and several paths then dereferenced the cleared geometry or the dangling internode tube object: the per-shoot volume bookkeeping loop in `advanceTime()` and the recursive `Shoot::updateShootNodes()` both touched the deleted object, and `getShootInternodeObjectIDs()` returned the dangling object ID to the caller. `updateShootNodes()` now returns early when a shoot has no phytomers (its empty descendants are likewise skipped), and the `advanceTime()` loop and `getShootInternodeObjectIDs()` now guard on `doesObjectExist()` before touching or returning a shoot's internode tube object (verified by a new self-test that prunes a branch and confirms `advanceTime()` no longer throws and the dangling ID is not returned). Also corrected the function name in `getShootInternodeObjectIDs()`'s error message (it reported `getPlantInflorescenceObjectIDs`).

# [1.3.76] 2026-06-20

## Radiation
- Fixed `RadiationCamera::applyCameraExposure()` over-amplifying small or distant subjects so that the auto-exposure depended on how much of the frame the subject occupied (camera distance, field of view) rather than on its radiance. The 95th-percentile gain anchor was measured over the whole frame, so when the subject filled less than the top 5% of pixels the percentile landed on the near-zero background, the gain floor took over, and the real signal was amplified by ~1/floor (≈7e5). The percentile is now taken over the signal pixels only (those above `1e-4` of the per-band peak), excluding the background, which makes the resulting exposure invariant to subject size; a band that is entirely background is left unscaled.

## Core
- Fixed `ProgressBar::finish()` not driving a disabled bar to completion through its progress callback. The terminal jump to 100% was gated on the `enabled` flag, but the callback is a separate output channel from console messages — `update()` already suppresses console output internally when the bar is disabled while still firing the callback — so a caller that disabled console messages and listened only via the callback never observed completion. `finish()` now advances the bar (and thus fires the callback at 1.0) whenever it is below total regardless of `enabled`, and remains a no-op once already complete.

## Plant Architecture
- Added `PlantArchitecture::setCancelFlag(volatile int *)` to abort a long plant/canopy build mid-flight: the caller-owned flag is polled at loop boundaries — between plants in both `buildPlantCanopyFromLibrary()` overloads and between timesteps in `advanceTime()` — so a cancelled build stops early and returns whatever has been built so far (a partially-built canopy, or plants aged less far) instead of running to completion, then falls through to the normal progress-bar finish. The flag is owned by the caller (e.g. a ctypes int shared with Python, written under the GIL while the build runs GIL-free) and must outlive the build; the per-iteration `volatile` read is effectively free, and `nullptr` clears it.

## Collision Detection
- Added a low-memory Structure-of-Arrays batch ray cast `castRaysSoA(origins, directions, count, max_distance, out_distance, out_normal, out_primitive_UUID, stats)` that writes results directly into caller-owned arrays (miss signalled by `out_primitive_UUID == 0xFFFFFFFF`), eliminating the intermediate `std::vector<RayQuery>` input and `std::vector<HitResult>` output (~96 bytes/ray of transient storage, including an unused per-ray target list) that the existing vector-based `castRays()` materializes — the path the LiDAR synthetic scan needed to stay memory-bounded over billions of sub-rays. The GPU dispatch decision was factored into a single shared `shouldUseGPU(ray_count)` predicate called by both `castRays()` and `castRaysSoA()` so the two never drift, and the GPU ray-tracing path was reworked to keep the scene geometry device-resident across calls: a new `buildGPUGeometrySoA()` packs the per-primitive type codes, vertices (4 per primitive), and vertex offsets once, `transferBVHToGPU()` uploads them alongside the BVH (freed by `freeGPUMemory()`), and the renamed `launchRaysOnResidentScene()` (replacing `launchRayPrimitiveIntersection()`) uploads/downloads only the per-call ray and result buffers instead of re-extracting and re-uploading the entire scene every batch — so a chunked synthetic scan no longer re-uploads the geometry per chunk. The intersection kernel now also computes and returns the face-forwarded surface normal on the device (matching the CPU `cross(edge1, edge2)` convention for triangles/patches and the axis-aligned face normal for voxels) rather than the host recomputing it per hit, accepts an optional uniform max-distance (no per-ray distance array needed when the cutoff is constant), and CUDA failures now raise `helios_runtime_error()` (fail-fast) instead of the previous `fprintf(stderr)`+return that left output arrays partially written (verified by a new SoA-vs-`castRays` equivalence self-test).
- Added an external cancellation flag (`setCancelFlag(volatile int *)`) polled inside the `castRaysSoA()` OpenMP ray loop so a long batch trace can be aborted mid-flight: each thread re-reads the caller-owned `volatile` flag per iteration and, once it flips non-zero, cheaply drains its remaining loop indices (OpenMP forbids breaking out of a worksharing loop), leaving skipped rays default-constructed (no hit) for the caller to discard. The per-ray `volatile` load is effectively free next to a BVH traversal, and `nullptr` clears the flag.

## LiDAR
- Added `LiDARcloud::setCancelFlag(volatile int *)` to abort a long `syntheticScan()` mid-trace: the caller-owned flag is forwarded to the collision-detection ray loop (re-applied when the collision-detection object is lazily created, so it works whether set before or after first use), and once it flips non-zero the scan returns early with whatever hits were recorded so far. This lets a caller (e.g. a ctypes int shared with Python, written under the GIL while the trace runs GIL-free) interrupt an in-flight scan and free its memory instead of running to completion; `nullptr` clears it (verified by a new self-test that the flag suppresses all hits and clearing it restores normal output).
- Added `setExternalTriangulation()` so an externally-supplied world-space mesh (e.g. a re-used Helios triangulation or a per-scan open3d Ball-Pivot mesh) can drive the leaf-area inversion in place of the internal Constrained-Delaunay triangulation. Because the inversion consumes triangulation only through the per-voxel G(theta) leaf-angle term — which needs each triangle's three vertices, its source scan (to recover the ray zenith via `getScanOrigin()`), and the grid cell its centroid falls in, not the triangulation topology — the method ingests a flat triangle-vertex list plus a required per-triangle scan ID, bins each triangle into a grid cell by centroid containment, drops degenerate (zero/NaN-area) triangles, and sets the triangulation-computed flag so `calculateLeafArea()` runs unchanged. A grid must already be defined and every scan ID must be valid, failing fast otherwise. The new private `getContainingGridCell()` helper reuses the same point-in-cell containment test (with inverse rotation for rotated cells) as `calculateHitGridCell()` so external-triangle binning and hit-point binning agree.
- Made `syntheticScan()` memory-bounded for large multi-return scans by processing each scan's beam fan-out in chunks instead of tracing all `beams * rays_per_pulse` sub-rays at once (which consumed ~135 GB resident on a 2500×4500 grid × 100 sub-rays scan). A new `setSyntheticScanMemoryBudget()`/`getSyntheticScanMemoryBudget()` soft cap sizes the per-chunk scratch buffers so only `chunk_beams * rays_per_pulse` sub-rays are live at once; the budget defaults to an automatic path-dependent cap (8 GiB on a GPU build, 4 GiB otherwise) and a small scan still runs as a single chunk, preserving the exact original behavior and RNG draw order. `performUnifiedRayTracing()` was split into a `prepareUnifiedRayTracing()`/`castRaysUnified()`/`finishUnifiedRayTracing()` bracket so the BVH is built once per scan (with automatic rebuilds disabled) and reused across chunks rather than rebuilt per chunk, with the per-chunk cast routed through `CollisionDetection::castRaysSoA()` (verified by a new self-test confirming a many-chunk run produces a byte-identical cloud to the single-chunk run).
- Added a `SINGLE_RETURN_STRONGEST_PLUS_LAST` single-return selection policy (XML alias `strongest_plus_last` / `dual`) that keeps the strongest echo plus the last (farthest) return of a pulse, deduplicated to one point when they coincide, modeling the "strongest + last" dual-return mode of real discrete-return scanners. Unlike the existing strongest/first/last policies it is a pick-these-two selection rather than a top-N ranking, so it intrinsically yields 1 or 2 returns and ignores `maxReturns`; it round-trips through `loadXML()`/`exportScans()` (verified by a new self-test).
- Replaced the per-hit `std::map<std::string,double>` scalar-data store with cloud-level columnar storage (one contiguous value array plus a presence mask per label, indexed by hit position), eliminating the last bulk-export hot spot: extracting a whole scalar field is now a single cache-linear pass instead of N cache-cold red-black-tree descents, matching XYZ/RGB export speed. The columns are kept in lockstep with the `hits` vector everywhere it is mutated (`addHitPoint`, `deleteHitPoint`'s swap-and-pop, a new `clearHits()`), new labels introduced mid-cloud are back-filled absent for prior hits, and the moving-platform per-pulse origin helpers (`origin_x`/`origin_y`/`origin_z`) now operate by hit index. New `getHitDataColumnIndex()` and `getHitDataColumn()` accessors expose the fast bulk-read path, and `exportPointCloud()` now resolves every output column once to a built-in field code or a column slot so its per-hit inner loop does no string comparisons or per-cell lookups. `getHitData()`/`doesHitDataExist()`/`setHitData()` are observably unchanged (verified by new bulk-getter-vs-per-hit, delete-lockstep, mid-cloud-back-fill, and origin-survives-`coordinateShift` self-tests).
- Added an optional progress callback to `syntheticScan()` via a new `LiDARcloud::setProgressCallback()` that receives `(progress_fraction, message)` per scan; it is driven by an internal `ProgressBar` whose step is updated at the top of each scan loop iteration so the callback fires on every code path (including the early "no rays hit the bounding box" continue) and is clamped to 1.0 on completion (verified by a new self-test, including the no-geometry early-continue case).
- Added high-level, physical-parameter entry points for spinning-multibeam and moving-platform scans so a caller specifies the instrument's real parameters instead of hand-flattening them into an `Ntheta`×`Nphi` grid. `LiDARcloud::addScanSpinning()` (quaternion and roll/pitch/yaw trajectory overloads) takes per-channel beam elevation angles, an azimuth resolution, a pulse repetition rate, and a 6-DOF trajectory, and derives the internal sampling grid, rotation rate, and revolution count (`steps_per_rev = round(2π/azimuthStep)`, `rotation_rate = PRF/(channels·steps_per_rev)`, `n_revolutions = rotation_rate·duration`, `Nphi = round(steps_per_rev·n_revolutions)`); it is the only way to set up a spinning scan, with a stationary "spin in place" capture (e.g. a tripod) expressed as a trajectory of two coincident poses separated in time by the acquisition duration, and `addScanMovingRaster()` wraps `addScanMoving()` for a fixed angular fan swept along a trajectory. Spinning scans fire each pulse at its exact per-channel elevation and sample a periodic azimuth (`dphi = (phiMax−phiMin)/Nphi`, no duplicated wrap column) matching `rc2direction()`/`direction2rc()`, whereas raster scans keep inclusive-endpoint sampling. A new self-describing `ScanMode` descriptor (`SCAN_MODE_STATIC_RASTER`/`SCAN_MODE_MOVING_RASTER`/`SCAN_MODE_SPINNING`) plus `steps_per_rev`/`rotation_rate`/`n_revolutions` fields on `ScanMetadata` make a scan introspectable through new `getScanMode()`/`getScanStepsPerRev()`/`getScanRotationRate()`/`getScanRevolutions()` accessors, and the `addScan()` degrees-vs-radians azimuth warning is now gated to `SCAN_MODE_STATIC_RASTER` so a legitimate multi-revolution `phiMax` no longer trips it. `loadXML()` gained `<azimuthStep>`, `<PRF>`, `<leverArm>`, `<boresight>`, `<t0>`, and inline `<trajectory>`/`<pose>` or referenced `<trajectoryFile>` elements (each pose row auto-detected as 8-number quaternion `t x y z qx qy qz qw` or 7-number Euler `t x y z roll pitch yaw` in degrees, failing fast on ragged rows); a spinning scan now always loads through this physical-parameter path (requiring `<azimuthStep>`, `<PRF>`, and a trajectory and failing fast otherwise), and the legacy static-grid spinning form (`<Nphi>`/`<phiMax>`/`<revolutions>`/`<orientation>` and the corresponding `ScanMetadata` spinning constructor) was removed. `exportScans()` writes the physical parameters plus a trajectory sidecar CSV so spinning and moving-raster scans round-trip through the same physical-parameter path they were created with (verified by new multi-revolution-derivation, stationary-seam, fail-fast-validation, azimuth-warning-gating, and XML round-trip self-tests).
- Generalized synthetic-scan single-return mode into a configurable N-return (single/dual/N-return) mode via a new per-scan `maxReturns` parameter. In `RETURN_MODE_SINGLE` a pulse now reports up to `maxReturns` real returns (1 = classic single-return, 2 = dual-return, N = N-return) instead of exactly one, with the subset chosen by the existing strongest/first/last `singleReturnSelection` policy and always re-ordered nearest-first so the downstream `target_index` assignment stays in range order; `RETURN_MODE_MULTI` is unchanged (every detected return reported). `detectReturnsFromSubrays()` was reworked from a single-best selection to a partial-sort-and-keep over the real returns, preserving the miss-handling invariant (a full-miss pulse keeps one transmitted-beam sentinel for leaf-area inversion, while a pulse with any real return drops the miss sentinel so only real points are reported). Configured through a new `ScanMetadata::maxReturns` field (default 1, must be ≥1) with `LiDARcloud::getScanMaxReturns()`/`setScanMaxReturns()` accessors and a `<maxReturns>` XML element that round-trips through `loadXML()`/`exportScans()` (written only when non-default), verified by new default/validation and exact-N-kept self-tests.
- Reworked multi-ray synthetic-scan return detection onto an analytic sum-of-Gaussians waveform model, replacing the previous fixed-distance sub-ray grouping, and exposed single- vs. multi-return reporting. Each sub-ray now carries a Gaussian footprint weight \f$w=\exp(-2(r/r_0)^2)\f$ from its offset off the beam axis (independent divergence-cone and exit-aperture factors), giving the beam a realistic Gaussian transverse profile instead of a uniform top-hat; the sorted sub-ray hits of a pulse are treated as a waveform from which discrete returns are detected by a new `detectReturnsFromSubrays()` helper. Hits within one range-resolution merge into a single return at the energy-weighted range (so two surfaces closer than a pulse width blend into one intermediate-range "ghost"/"mixed-pixel" point), returns below a detection threshold are discarded (noise floor), and a single-return mode reports one return per pulse selected by strongest/first/last policy; each return additionally records an `echo_width` (pulse range-extent in quadrature with the merged-surface range spread). These are configured per scan through new `ScanMetadata` fields (`returnMode`, `singleReturnSelection`, `pulseWidth`, `detectionThreshold`) with `LiDARcloud` get/set accessors, a new `syntheticScan()` overload taking an explicit `ReturnMode`, and `<returnMode>`/`<singleReturnSelection>`/`<pulseWidth>` (or `<pulseDuration>`)/`<detectionThreshold>` XML elements that round-trip through `loadXML()`/`exportScans()`. The range resolution used for merging is the scan's `pulseWidth` when set, otherwise the `pulse_distance_threshold` argument, and a single ray per pulse still produces the idealized exact intersection; the full-miss intensity sentinel and leaf-area-inversion miss handling are preserved (verified by new idealized, ghost-point, resolved-multi-return, selection-policy, detection-threshold, and XML round-trip self-tests).
- Sped up the leaf-area inversion without changing results by replacing the brute-force per-cell slab loop (which tested every hit against every voxel, O(Nscans·Ncells·Nhits)) with a per-beam 3D-DDA traversal that walks each beam through only the voxels it actually pierces (O(Nscans·Nbeams·cells-pierced)). A new `detectVoxelLattice()` recognizes the regular axis-aligned lattice produced by `addGrid()` (shared anchor/size/division-count/azimuthal-rotation, dense `ijk`→cell map) and, when found, runs the Amanatides-Woo DDA fast path; grids assembled cell-by-cell with mixed sizes or rotations fall back to the original brute-force per-cell loop. Both paths now share a single `accumulateBeamCell()` helper and compute each voxel's entry/exit parameters from the identical slab expression, so the DDA result is bit-for-bit equivalent to the brute-force result (verified by new A/B self-tests across cubic, non-cubic, single-return, and multi-return grids, plus a non-lattice-fallback test); a `forceBruteForceLeafArea()` diagnostic hook forces the fallback for that comparison. The DDA path is parallelized over beams with per-thread scratch buffers (no per-beam heap allocation in the hot loop) and a deterministic ascending-thread-id reduction. Supporting changes: the `BeamGrouping` structure was switched from a vector-of-vectors to a compressed-sparse-row (`beam_members`/`beam_offsets`) layout to avoid one small heap allocation per beam (tens of millions in a dense scan), `groupHitsByTimestamp()` now caches each hit's timestamp once and builds the CSR grouping in a single pass instead of calling `getHitData()` O(N log N) times during the sort.

# [1.3.75] 2026-06-15

## Radiation
- Added a translucent-cover (glass/plastic) material for modeling greenhouse/shadehouse glazing (glass, polyethylene film, polycarbonate, acrylic). Setting primitive data `glass_n_<band>` (refractive index, ≥1) activates an on-device Fresnel+Bouguer angular model that computes the angle-dependent transmittance \f$\tau(\theta)\f$, reflectance, and absorptance of a single dielectric sheet (Fresnel reflection at both interfaces summed over internal reflections for both polarizations, plus Bouguer absorption through the sheet from the optional `glass_KL_<band>` product K·L, default 0 = lossless), overriding any constant `reflectivity_<band>`/`transmissivity_<band>` for that primitive+band and exempting it from the ε+ρ+τ=1 conservation check. A glass primitive lets direct-source and diffuse-sky rays pass through (nearly undeviated) attenuated by \f$\tau(\theta)\f$ rather than being occluded, implemented via newly-added any-hit programs (OptiX 8 `__anyhit__direct`/`__anyhit__diffuse`, OptiX 6, and a per-ray `cover_transmittance` accumulator in the Vulkan direct/diffuse compute shaders) that multiply a per-launch-band per-ray transmittance accumulator by the cover's `tau` and deposit the absorbed/reflected fraction, with the accumulator consumed in the miss programs; the model is gated by a cheap `glass_enabled` flag so non-glass scenes pay no cost, requires a non-zero scattering depth, and is supported identically across the OptiX 8, OptiX 6, and Vulkan backends (new `shaders/common/glass_cover.glsl` GLSL port). A new `RadGlass` documentation section, the `glass_n_*`/`glass_KL_*` primitive-data table entries, and representative literature refractive indices were added.
- Fixed a pre-existing OptiX 8 illegal-memory-access crash (CUDA error 700) when a radiation camera is bound to an emission (thermal/longwave) band with no positive-flux sources. `runBand()` auto-adds a default zero-flux collimated source, so `Nsources == 1` while `rundirect == false`; `source_fluxes` was uploaded only inside the gated direct pass, leaving the device pointer null while `Nsources` was uploaded as 1, so the camera miss program's source loop dereferenced a null `source_fluxes`. `runBand()` now uploads the per-band source fluxes whenever `Nsources > 0` (zero fluxes are skipped device-side), and `OptiX8Backend::uploadSourceFluxes`/`uploadSourceFluxesCam` now assign the host pointer unconditionally so an empty upload yields `nullptr` rather than a dangling freed pointer that would defeat the device-side null guards.
- Fixed OptiX 8 one-sided primitives (`twosided_flag == 0`) failing to occlude rays striking their back face, which let direct/diffuse rays leak through the back of one-sided geometry inconsistently with the OptiX 6 and Vulkan backends. The `__intersection__patch` program now reports the intersection for both faces (recording the hit face in `face_attr`) and one-sidedness is enforced at ray launch, and `buildGeometryData()` now stores the raw 0/1/2/3 two-sided flag value rather than collapsing it to a boolean so the backends receive the full semantics.

## Core
- Fixed undefined behavior in the EXIF writer (`exif_writer.cpp`): an ASCII entry built from an empty string called `std::memcpy` with a null source pointer, and the APP1 payload was assembled with an unnecessary intermediate header buffer. Both were hardened.

## LiDAR
- Fixed `syntheticScan()` recording every return as a miss for planar/flat scene geometry (a single patch, a flat wall, or a flat ground plane with no thickness). The AABB cull that pre-filters rays against the scene bounding box rejected all rays when the box was degenerate (zero extent) along an axis — a zero-thickness axis forced the slab test's entry and exit parameters to coincide, failing the strict interval test for every ray actually pointing at the plane. Degenerate bounding-box axes (for both the whole-domain and voxel-grid culls) are now padded to a tiny finite thickness before the slab test, leaving non-degenerate scenes unchanged.
- Sped up the leaf-area inversion and hit-to-voxel assignment without changing results. `calculateLeafArea()`'s per-scan `global_to_local` hit index was switched from a `std::map` to a flat `std::vector`, making the per-beam lookups in the classification loop (which runs ~Ncells×Nbeams times per scan) O(1) and cache-friendly instead of a red-black-tree traversal. In `calculateHitGridCell()` the constant per-voxel geometry (bounds, anchor, rotation) is now hoisted into contiguous arrays once before the parallel per-hit loop rather than re-fetched through bounds-checked getters on every hit×cell iteration, and the per-voxel ray-from-origin slab test was replaced by the equivalent point-in-AABB containment test it reduces to (the ray passes through the hit point P at parameter T=|P|, so the entry/exit interval contains T iff P lies inside the box bounds).
- Added an optional global scanner azimuth (heading) offset for synthetic scans — the yaw component that completes the roll/pitch/yaw scanner pose alongside the existing `<scanTilt>` — configured per scan via a new `<scanAzimuthOffset>` XML element (degrees; default `0` = no offset) and exposed through a new `scanTilt_azimuth` field on `ScanMetadata`, a trailing constructor argument on both `ScanMetadata` constructors, and a `LiDARcloud::getScanAzimuthOffset()` accessor. During `syntheticScan()` the offset applies a right-hand rotation of the entire fan of ray directions about the world vertical (`+z`) axis on top of the per-scan azimuth sweep, and it also rotates the roll/pitch tilt body frame (the azimuth-zero heading from which the forward/lateral axes are derived), so the combined orientation is applied in yaw→pitch→roll order. The new field round-trips through `exportScans()`/`loadXML()`.
- Added moving-platform (mobile/airborne) synthetic-scan support, in which the scanner pose changes during the sweep instead of firing from a single static origin. A new `LiDARcloud::addScanMoving()` (with both a quaternion and a roll/pitch/yaw Euler-angle trajectory overload) registers a scan driven by a dense timestamped 6-DOF pose trajectory, a body-frame lever arm, a fixed boresight misalignment, and a pulse repetition rate; `ScanMetadata` gains the trajectory storage and a `poseAt()` method that linearly interpolates position and SLERPs orientation (Hamilton body→world quaternions, clamped at the endpoints) at each pulse's acquisition time `t = t0 + ordinal·(1/pulse_rate_hz)`. During `syntheticScan()` each pulse now computes its own emission origin (`pos + R(q)·lever_arm`) and direction (`R(q)·R(boresight)·dir_body`), records its real timestamp and firing index, and stores the per-pulse origin on every hit and miss as `origin_x`/`origin_y`/`origin_z` data; the static `scanTilt` is not applied in this mode and must be zero. The leaf-area inversion, ray-direction recovery (`getHitRaydir()`), distance filter, and coordinate shift/rotation transforms were reworked to measure beam geometry from each hit's own emission origin via a new `getHitOrigin()` accessor (falling back to the single scan origin for static scans), so they remain correct under platform motion; the timestamp gap-filling path emits synthesized misses from the interpolated platform pose. Because moving scans have no fixed theta-phi grid to triangulate, a new `calculateLeafArea()` overload accepts a caller-supplied `G(theta)` instead of deriving it from triangulation, and the functions that assume a single static origin (`triangulateHitPoints()`, `validateRayDirections()`, the row/column gap-fill path, `exportPointCloudPTX()`) now fail fast on a moving scan. `loadXML()`/`exportScans()` were updated so a scan defines its origin via either a static `<origin>` tag or per-point `origin_x`/`origin_y`/`origin_z` columns (one is now required), and the PTX/static-origin paths reject moving scans rather than write a misleading single origin.

# [1.3.74] 2026-06-10

## Plant Architecture
- Added two read-only shoot-topology accessors: `PlantArchitecture::getAllShootIDs()` returns the contiguous, 0-based list of shoot IDs for a plant (shoot 0 is always the base stem), and `PlantArchitecture::getPlantShoot()` returns a const reference to the `shared_ptr<Shoot>` for a given shoot, exposing its topology (rank, parent/child IDs, parent node index) and woody internode geometry for inspecting a plant's ground-truth structure. Both fail fast with `helios_runtime_error()` on a nonexistent plant ID, and `getPlantShoot()` also on an out-of-range shoot ID.
- Fixed terminal fruiting buds being created at full size and then abruptly snapping to 25% scale on the first fruit-growth step: `appendPhytomerToShoot()` now initializes a terminal bud entering the `BUD_FRUITING` state at a 0.25 inflorescence scale fraction (matching the existing `setFloralBudState()` behavior), so the fruit/panicle grows in gradually instead of regrowing after a sudden shrink.

## LiDAR
- Fixed `syntheticScan()` producing NaN ray directions for a single-row or single-column scan: the inclusive-endpoint angular step now guards its `N-1` denominator so a scan with `Nphi==1` or `Ntheta==1` samples once at the minimum angle instead of dividing by zero.
- Added an optional global scanner tilt for synthetic scans, configured per scan via a new `<scanTilt>` XML element (`roll pitch` in degrees; default `0 0` = perfectly level) and exposed through new `scanTilt_roll`/`scanTilt_pitch` fields on `ScanMetadata`, two trailing constructor arguments, and `LiDARcloud::getScanTiltRoll()` / `getScanTiltPitch()` accessors. This models the residual tilt of the scanner spin axis away from plumb that a real terrestrial scanner's dual-axis inclinometer reports. During `syntheticScan()` the entire fan of ray directions is rotated about the scanner origin using right-hand rotations in a right-handed, Z-up body frame (matching commercial scanners such as RIEGL SOCS): the forward axis is the horizontal projection of the azimuth-zero (`phiMin`) heading, the lateral axis completes the frame, roll rotates about the lateral axis and pitch about the forward axis, with roll applied first. The new field round-trips through `exportScans()`/`loadXML()`.
- Reworked leaf-area inversion onto a single miss-aware, beam-based equal-weighting algorithm that handles both single- and multi-return data, replacing the previous auto-detected split between a single-return "standard" path and a multi-return "equal weighting" path (the now-unused `filterRaysByBoundingBox()`, `calculateVoxelPathLengths()`, and `findRayIndexByDirection()` helpers and the entire single-return path were removed). Misses (fired pulses that returned nothing — the transmitted beams that form the denominator of the per-voxel transmission probability) are now the canonical input: they are flagged with a per-hit `is_miss` data field (set by `syntheticScan()` and `gapfillMisses()`, carried through imported data when present), queryable via new `LiDARcloud::isHitMiss()` / `hasMisses()` methods and the `LIDAR_MISS_DISTANCE` constant. `calculateLeafArea()` now fails fast with an explicit error if the cloud contains no misses, rather than silently producing biased LAD. Also fixed `invertLAD()` to no longer report a stalled secant solve (which left the estimate near the 0.1 initial guess) as converged — convergence now requires the achieved error to fall below tolerance, so a stalled solve correctly falls back to the average-`dr` formulation.
- Made the leaf-area inversion's per-beam voxel classification independent of how far out a miss point is placed along its beam. Previously a transmitted beam was counted toward the transmission probability only when its hit point fell within a hard-coded 5000 m `scanner_range` (so a miss placed at the synthetic-scan distance and a miss placed at `LIDAR_MISS_DISTANCE` by `gapfillMisses()` were classified differently, and moving the placement distance silently doubled the inverted LAD). A beam is now classified purely geometrically: any beam that passes through the voxel exit is "after voxel" (transmitted) regardless of placement distance, folding misses into the same `E_after` count as any other transmitted return. The `scanner_range` threshold and the separate all-miss-only probability branch were removed. Results are unchanged for existing data; only the previously fragile coupling between the miss placement distance and the inversion is eliminated.
- Documented and made explicit the synthetic-scan return-intensity radiometric model: the `intensity` recorded by `syntheticScan()` is the *range-normalized* return amplitude \f$\rho\cos\theta\f$ (per-primitive `reflectivity_lidar` reflectivity times the incidence-angle cosine) with the \f$1/R^2\f$ range loss of the LiDAR range equation divided back out, so a surface returns the same intensity regardless of scanner-to-target range. The normalization is now routed through a new `LiDARcloud::applyRangeIntensityCorrection()` helper (identity on the value, since the synthetic intensity is generated directly as \f$\rho\cos\theta\f$) that is the single place to switch to a raw range-dependent convention. Full-waveform partial-footprint attenuation is deliberately preserved (it is carried by the fraction of beam sub-rays striking the target — a target property, not a range-geometry loss).
- Added an optional `reflectance` synthetic-scan output column: when `reflectance` is listed in a scan's `ASCII_format`, `syntheticScan()` records the return reflectance in decibels, \f$10\log_{10}|I|\f$ of the range-normalized intensity, following the terrestrial-laser-scanner convention (e.g. RIEGL) in which a perfect Lambertian reflector at normal incidence is 0 dB and all real returns are negative. Returns with no detectable signal (misses, fully grazing, or back-facing hits) are floored at -999 dB rather than \f$-\infty\f$. The `reflectance` token is treated as a computed output rather than a primitive-data field, so it is not overwritten by same-named primitive data.
- Added a robust row/column miss gap-filling path and made `gapfillMisses()` auto-select between it and the existing timestamp path. When returns carry native scan-grid `row`/`column` hit data the new `gapfillMisses_rowcolumn()` fits a per-row generative model (median zenith per row; Theil-Sen azimuth line `intercept + slope·column`, with cross-row Theil-Sen extrapolation of the per-row parameters so blank near-zenith rows are extrapolated rather than only interpolated), which is robust to scanner tilt, azimuth shear, and angular noise without requiring a level or regular scan; the timestamp path is renamed `gapfillMisses_timestamp()` and used as the fallback. Row/column is preferred when both are present; a scan with returns carrying neither now fails fast, while a scan with no returns at all is skipped gracefully. The ASCII loader now retains `row`/`column` columns as hit data (previously discarded) so imported clouds can use this path.
- Made exported ASCII point clouds self-describing: `exportPointCloud()` now writes a leading `#`-prefixed comment-line header listing the column field names (matching the `ASCII_format` columns, including user-defined fields), following the convention recognized by tools such as CloudCompare. A new optional `write_header` argument (default true) suppresses it, and the loader now skips any `#`-prefixed line, so headered files round-trip through `loadXML()` unchanged.
- Added read-only triangulation diagnostics to `triangulateHitPoints()`: the run now tallies the number of candidate triangles the Delaunay pass produced and how many were dropped by each filter, exposed through new `LiDARcloud::getTriangulationCandidateCount()`, `getTriangulationDroppedByLmax()`, `getTriangulationDroppedByAspect()`, and `getTriangulationDroppedByDegenerate()` accessors. The filtering logic was refactored so each dropped triangle is attributed to exactly one primary reason in priority order (`Lmax`, then aspect/separation-ratio, then degenerate area), so the counts reconcile as `candidates == kept + dropped_lmax + dropped_aspect + dropped_degenerate`. No change to which triangles are kept.
- Added per-voxel statistical sampling uncertainty for the leaf-area inversion, following Pimont et al. (2018, RSE 215:343-370). A new `calculateLeafArea()` overload taking a characteristic vegetation `element_width` computes, alongside the unchanged leaf-area point estimate, the sampling variance of LAD via a Beer-Lambert delta-method propagation with two terms: a finite-beam term that decays as 1/N (the binomial variance of the relative density index, guarded against the larger empirical spread of multi-return per-beam transmittance fractions), and an N-independent element-position-variability term derived from the single-element optical depth (omitted, leaving a sampling-only variance, when `element_width <= 0`); the existing two-argument overload delegates to it with a default 5 cm width. The per-voxel sufficient statistics and variance are stored on each `GridCell` (beam count N, relative density index, mean/variance of beam path length, single-element optical depth, LAD variance, and a precomputed 95% CI-validity flag) and exposed through new `getCellBeamCount()`, `getCellRelativeDensityIndex()`, `getCellMeanPathLength()`, `getCellLADVariance()`, single-voxel `getCellLeafAreaConfidenceInterval()`, and group-scale `getGroupLADConfidenceInterval()` (the recommended path; Eq. 39, assuming voxel independence) accessors. Confidence intervals are gated by the Pimont Table-3 validity envelope (`ciValidPimont()`) so untrustworthy single-voxel intervals are refused rather than emitted, with the two-sided z-multiplier obtained from an Acklam normal-quantile approximation. A new `exportLeafAreaUncertainty()` writes a self-describing per-cell file (`cell_index leaf_area beam_count I_rdi LAD_std_error ci_valid`). The uncertainty is conditional on the beams that entered each voxel and does not capture occlusion/coverage bias.
- Added a spinning multibeam scan pattern modeling a rotating multi-channel sensor (e.g. Velodyne, Ouster, Hesai), in addition to the existing uniform-angular-grid raster pattern. A new `ScanPattern` enum (`SCAN_PATTERN_RASTER`/`SCAN_PATTERN_SPINNING_MULTIBEAM`) and `scanPattern`/`beamZenithAngles` fields on `ScanMetadata` are set through a new `ScanMetadata` constructor that takes a vector of per-channel zenith angles in place of `Ntheta` and the zenith range. Each scan-table row is a laser channel fired at its own fixed (generally non-uniformly spaced) zenith angle while each column is a uniform azimuth step, so the pattern reuses the same `Ntheta`×`Nphi` table storage and all downstream processing (ray tracing, hit tables, leaf-area/leaf-angle inversion) is shared with raster scans; `rc2direction()`/`direction2rc()` map a multibeam row to the nearest channel rather than interpolating, and `syntheticScan()` fires each row at its channel angle and records the firing channel index as a `channel` hit-data column. The pattern round-trips through `exportScans()`/`loadXML()` via a new `<scanPattern>` element plus `<beamElevationAngles>` (space-separated per-channel elevation angles in degrees above the horizon, converted internally to zenith) and `<Nphi>` (azimuth-step count), and is queryable through new `LiDARcloud::getScanPattern()` / `getScanBeamZenithAngles()` accessors.

# [1.3.73] 2026-06-06

## Core
- Migrated spammy per-iteration warnings to the existing `WarningAggregator` machinery throughout the core and several plugins, so that repeated identical warnings (per-primitive, per-band, per-voxel, per-CSV-row, per-data-label) collapse to a single deduplicated message with an occurrence count instead of flooding stderr. Affected core paths include `Context::filterObjectsByData`, both `Context::setTileObjectSubdivisionCount` overloads, `Context::loadXML` (missing-subdivision warnings for tile/sphere/tube/box/disk/cone), `Context::loadOsubPData` (per-data-label length-mismatch warnings — now takes a `WarningAggregator&` argument so its messages join the enclosing `loadXML` aggregation), and `Context::loadTabularTimeseriesData`. Plugin paths covered: `AerialLiDARcloud::syntheticScan`, `LiDARcloud::syntheticScan` / `backfillLeavesAlphaMask` / `calculateLeafArea` (`invertLAD` now also takes a `WarningAggregator&`), `CollisionDetection::castRaysGPU` and `slicePrimitivesUsingGrid`, `EnergyBalanceModel::evaluateSurfaceEnergyBalance_{CPU,GPU}` (air-pressure validation), `LeafOptics::groupPrimitivesByObject` and `updateNitrogenBasedSpectra`, `RadiationModel::updateRadiativeProperties` (spectral-type checks) and `runBand` (scattering-disabled-for-band warning), `RadiationModel::writeImageSegmentationMasks{,_ObjectData}` label checks, and `CameraCalibration::preprocessSpectra`. The five `Patch::rotate` / `Triangle::rotate` "cannot rotate primitives in a compound object" warnings and the `spline_interp3` clamp warning were instead given a once-per-process `static bool` guard since they are not called from a single bounded loop.
- Added two `Context::deleteTimeseriesDataPoint()` overloads to delete a single timeseries data point identified by (date, time): one that targets a named variable, and one that removes the matching point from every existing timeseries variable. Both issue a non-fatal warning to stderr if no matching point is found (or, for the labelled overload, if the variable does not exist) and are otherwise no-ops; matching uses the same floating-point date/time encoding as `addTimeseriesData()` and `updateTimeseriesData()`.
- Added a metadata-aware `helios::writeJPEG()` overload that takes a new `helios::ImageEXIFData` struct (`core/include/image_metadata.h`) and embeds EXIF (APP1 / TIFF IFDs) and XMP (APP1 / Pix4D `Camera` namespace) segments directly into the JPEG output. The EXIF serializer (`core/include/exif_writer.h`, `core/src/exif_writer.cpp`) is hand-rolled — no new external dependency — and writes a TIFF header + IFD0 + ExifSubIFD + GPS IFD using bundled `libjpeg-9a`'s `jpeg_write_marker(JPEG_APP0+1, …)`. XMP is generated as RDF/XML via the already-vendored pugixml. The existing 4-arg `writeJPEG()` overload is unchanged, so this is additive.
- Added a `helios::Location::altitude_m` field (default `0.f`) with a new 4-arg constructor and `make_Location(lat, lon, utc, alt)` overload. Existing 3-arg callers continue to work because the new field defaults to 0; equality and stream operators now include altitude. This is so the radiation plugin can write a meaningful `GPSAltitude` EXIF tag relative to a user-specified scene origin altitude rather than treating the local Cartesian origin as sea level. The struct docstring also now flags Helios's non-standard +W longitude convention and notes that EXIF emission flips the sign automatically.
- Added label-only `Context::clearPrimitiveData(label)` and `Context::clearObjectData(label)` overloads that remove a data label from *every* primitive/object (including hidden ones) and release the registered type for that label, so it may subsequently be re-registered with a different type. The per-label type lock is now also released automatically when the last primitive/object holding a label is cleared via the existing UUID-vector overloads (the label reference counter reaching zero now erases the type-registry entry). The associated type-mismatch error and casting-warning messages thrown from `setPrimitiveData()`/`setObjectData()` were rewritten to name the actual public method and to point users at the new `clearPrimitiveData()`/`clearObjectData()` overloads as the way to change an existing label's type.

## Radiation
- Every JPEG written by `RadiationModel::writeCameraImage()` now embeds EXIF (Make, Model, Software, DateTime, FocalLength, FocalLengthIn35mmFilm, FocalPlaneX/YResolution, FNumber, MaxApertureValue, SubjectDistance, ExposureTime, ExposureBiasValue, ExposureMode, WhiteBalance, ISOSpeedRatings, DigitalZoomRatio, PixelXDimension, PixelYDimension, LensMake, LensModel, LensSpecification, Orientation) and a full GPS IFD (lat/lon/alt with hemisphere refs, ImgDirection, DateStamp, TimeStamp), plus an XMP packet in the Pix4D `Camera` namespace with `Camera:Yaw`/`Pitch`/`Roll`. This makes simulated images directly consumable by photogrammetry / structure-from-motion pipelines (OpenDroneMap, COLMAP, Pix4D, Agisoft Metashape) without a sidecar file; the existing JSON sidecar metadata pathway is unchanged and continues to be gated by `enableCameraMetadata()`. GPS coordinates use a flat-earth approximation centered on the Context's `Location` and an explicit Cartesian convention of +Y = geographic North, +X = East, +Z = up; UTC date/time is computed from the Context date+time and Helios's `+W` UTC offset via Julian-day arithmetic to handle year/leap-year rollovers. Camera roll is always written as 0 because `RadiationCamera` does not currently store an up-vector — flagged as a follow-up.
- Added a `CameraProperties::manufacturer` field (and the matching `RadiationCamera::manufacturer` storage) and updated `addRadiationCameraFromLibrary()` to populate it from the camera library's `<manufacturer>` XML element. This decouples Make from Model for EXIF embedding so that a library camera like `Canon_20D` writes `EXIF:Make = "Canon"` and `EXIF:Model = "EOS 20D"` rather than a hardcoded `Make = "Helios"` with `Model = "Canon EOS 20D"` — photogrammetry sensor-calibration databases key on the two tags separately. The legacy `model = "<manufacturer> <model>"` concatenation is preserved in the JSON sidecar metadata path, so JSON consumers see no behavior change; the EXIF populator strips the manufacturer prefix when emitting the Model tag.
- Cameras now sample isotropic sky longwave/thermal emission on a ray miss. `RayTracingBackend::updateSkyModel` gained two new per-band inputs (`camera_diffuse_flux`, `band_emission_flag`) plumbed through the OptiX 6, OptiX 8, and Vulkan backends and their respective camera miss/raygen programs (`rayHit.cu::miss_camera`, `OptiX8DeviceCode.cu::__miss__camera`, `shaders/camera_raygen.comp`); for bands with emission enabled, the camera miss now adds `diffuse_flux / π` to the per-pixel radiance so user-set hemispherical sky thermal flux contributes to camera images instead of returning zero. The diffuse flux is held in a dedicated camera-side buffer rather than reusing the global diffuse-flux buffer because `launchCameraRays` does not refresh that buffer in camera-only dispatches. `RadiationModel::runBand` builds the new per-band vectors from `getDiffuseFlux()` and the per-band `emissionFlag` and uploads them alongside the existing sky-model parameters; the Vulkan descriptor pool was bumped from 62 to 64 storage-buffer slots and the sky descriptor set gained bindings 7-8 to accommodate.

## Plant Architecture
- Fixed USD export for IsaacSim/PhysX consumption: `PhysicsScene` now declares the `PhysxSceneAPI` schema and sets `physxScene:enableGPUDynamics`, `broadphaseType = "GPU"`, and `solverType = "TGS"`; per-link `xformOp:translate` (in both `writePlantStructureUSD()` link bodies and the `writePlantGrowthUSD()` time-sampled animation) is now declared as `float3` rather than `point3f` so it matches the `xformOpOrder` token type expected by USD/PhysX runtimes; and collision meshes now apply `PhysicsMeshCollisionAPI` alongside the existing collision schemas so mesh-based colliders are recognised correctly.
- Reworked the carbohydrate model to split each shoot's single `carbohydrate_pool_molC` into separate `sugar_pool_molC` (mobile) and `starch_pool_molC` (stored) reserves, with `total_carbohydrate_pool_molC` as their sum. Sugar above the upstream-transfer threshold now sequesters a fixed fraction (`starch_sequestration_ratio`) into starch each step, and stored starch is fully mobilized back to sugar on dormancy break (new `Shoot::mobilizeStarch()`, called from `breakPlantDormancy()` and the dormancy-break branch of `advanceTime()`). The carbohydrate parameter defaults were reparameterized to literature values (DeJong & Walton 1989, DeJong & Goudriaan 1989, AL Pica 2022) — notably `stem_density` 54000 → 675000 g m⁻³, `maturity_age` 180 → 120 d, a derived SLA, and a carbohydrate-fraction reformulation. Object data written by the model now exposes `sugar_pool_molC` and `total_carbohydrate_pool_molC` in place of the old `carbohydrate_pool_molC`.
- Made plant maintenance respiration temperature-dependent: rates are now stored per-plant at a 20 °C reference (`r_m_w_20`/`r_m_r_20`) and scaled by a Q10 of 1.8 against the prevailing air temperature each step via the new `PlantArchitecture::adjustPlantMaintenanceRespiration(Ta)`, replacing the previous constant rate and the separate dormant-respiration multiplier.
- Fixed carbohydrate accounting in the growth model: `calculateShootInternodeVolume()` now sums per-phytomer volumes (consistent with the growth/abortion accounting) instead of multiplying the whole-tube volume by the phytomer count; the per-phytomer construction cost that was double-charged in `appendPhytomer()` was removed; growth-respiration carbon is now charged explicitly per phytomer and reported as new `daily_growth_respiration` object data; a phytomer whose own pool cannot cover its construction cost now draws from its parent shoot rather than silently driving the pool negative; and shoots whose total pool reaches zero (or that have negative/degenerate volumes) are now pruned immediately.
- Retuned several library models: the almond tree gained longer internodes and more nodes per scaffold/shoot, higher gravitropic curvature, and adjusted bud-break probabilities for a fuller canopy, and its leaf/bark textures are now referenced by bare filename (resolved through the normal asset search) rather than an absolute `resolvePluginAsset()` path; the maize and sorghum phyllotactic angle, wave period/amplitude, leaf-buckle angle, and sorghum gravitropic curvature were re-narrowed to their pre-1.3.72 values.

## Collision Detection
- The CPU ray tracer now respects texture transparency, matching the GPU radiation ray tracer: when a ray strikes a primitive textured with an image that has a transparency (alpha) channel, the texel at the intersection point is tested against the texture's transparency mask, and hits on transparent texels are rejected so the ray passes through to geometry behind. Implemented as a new `isHitTexelOpaque()` helper plus cached `transparency_mask`/`texture_size`/`uv` fields on `CachedPrimitive` (populated during `buildPrimitiveCache()` to keep parallel traversal thread-safe); the fallback uncached and single-primitive paths build the mask from the Context on demand. Primitives without a transparency channel (untextured, or e.g. JPEG textures) keep a null mask and are treated as fully solid, preserving previous behavior.

## LiDAR
- Synthetic-scan hit points are now colored by sampling the primitive's texture image at the intersection point (decoding the PNG/JPEG once per texture into a cached RGB map and interpolating the patch/triangle UV at the hit) rather than using the primitive's flat base color, so textured leaves and other primitives produce realistically colored point clouds. Also fixed the recorded hit UUID, which was being remapped a second time through the `ID_mapping` table even though the ray tracer already returns the true Context UUID.
- Added `LiDARcloud::exportScans()`, which writes an XML metadata file describing all scans plus one ASCII point-cloud data file per scan (named `<stem>_<scanID>.xyz` beside the XML); the XML can be re-loaded with `loadXML()`. Relatedly, `loadXML()` now resolves a scan's data file by trying `input/<file>`, then a cwd-relative/absolute path, then the XML file's own parent directory, so a metadata+data bundle loads correctly regardless of working directory.
- Added optional Gaussian range (along-beam) measurement noise for synthetic scans, configured per scan via a new `<rangeNoiseStdDev>` XML element (meters; default 0 = no noise) and exposed through a new `rangeNoiseStdDev` field on `ScanMetadata`, the corresponding `ScanMetadata` constructor argument, and a `LiDARcloud::getScanRangeNoiseStdDev()` accessor. During `syntheticScan()` the measured range of each return is perturbed by an independent zero-mean Gaussian draw (using the Context RNG, so `Context::seedRandomGenerator()` makes scans reproducible) and the hit point is reconstructed along the nominal beam direction — producing physically realistic *anisotropic* positional error (displacement along the beam, not isotropic in x/y/z) consistent with how established LiDAR simulators (HELIOS++, BlenSor) model ranging error. Misses are not perturbed. The new field round-trips through `exportScans()`/`loadXML()`.
- Added optional Gaussian angular (beam-pointing) jitter for synthetic scans, the across-beam complement of the range noise above, configured per scan via a new `<angleNoiseStdDev>` XML element (radians; default 0 = no jitter) and exposed through a new `angleNoiseStdDev` field on `ScanMetadata`, the constructor argument, and a `LiDARcloud::getScanAngleNoiseStdDev()` accessor. During `syntheticScan()` the nominal direction of each pulse is perturbed by an independent zero-mean Gaussian angular offset (small-angle tilt in the plane perpendicular to the beam) before ray tracing, so the whole beam — including any divergence cone and finite aperture — rotates together; the resulting lateral positional error scales with range (≈ range·σ), giving the across-beam axis of the per-point error ellipsoid (distinct from beam divergence, which models within-beam footprint spread). Misses are not perturbed. The new field round-trips through `exportScans()`/`loadXML()`.
- Generalized synthetic-scan primitive-data transfer: `syntheticScan()` now copies *any* scalar primitive-data field (`float`/`double`/`int`/`uint`) named in a scan's ASCII column format onto its hit points, rather than only the hardcoded `object_label` and `reflectivity_lidar` labels. The scan column format is the source of truth — adding a non-standard label to it makes the scanner sample that primitive data per hit — and the special `reflectivity_lidar` semantics (modulating hit intensity) are preserved. Geometry/standard tokens (coordinates, colors, row/column, zenith/azimuth, `raydir`) continue to be handled by the file I/O path and are never treated as primitive-data labels.
- Hardened `gapfillMisses()` against out-of-bounds reads and runaway fills on ASCII point clouds that lack row/column indices (whose angular grid is reconstructed from timestamps). Unsigned `size() - 1` loop bounds that could underflow on empty/single-hit tables were rewritten as `r + 1 < size()`; per-beam `dt`/`dtheta` arrays are now value-initialized; the routine bails out early (returning no filled points) when fewer than two cleaned hits remain or when the reconstructed timestamp spacing is non-finite or non-positive; the per-gap fill count is capped at the scan's row count; and reconstructed grid positions outside the scan's row/column range are skipped. This fixes a non-deterministic crash on the multi-return file-import path.
- Replaced the s_hull_pro 2D Delaunay triangulation library with [CDT](https://github.com/artem-ogre/CDT) (v1.4.4, bundled header-only in `plugins/lidar/lib/CDT`) as the engine behind `triangulateHitPoints()`. CDT uses robust geometric predicates, so the previous per-scan rotate-and-retry recovery loop (which retried triangulation up to three times, rotating the azimuth coordinate by 0.25·π each time to dodge s_hull's non-robust-predicate failures) was removed; a triangulation failure is now deterministic and the affected scan is skipped. The `Shx`/`Triad` point/triangle types and the `de_duplicate()` helper that `LiDAR.cpp` still relies on were reduced to minimal self-contained definitions in the new `triangulation_cdt.h` adapter, and `s_hull_pro` was deleted entirely. The 1e-6 input-coordinate snapping (for cross-architecture determinism of the `cart2sphere` inputs) is retained. This also resolves a latent licensing mismatch — s_hull_pro is GPL-3.0 while Helios is GPL-2.0; CDT is MPL-2.0. No public API change.

## Solar Position
- Fixed `SolarPosition` to honor fractional UTC offsets: the local standard time meridian (`LSTM`) was truncated to an integer, so half-hour zones (e.g. India, +05:30) snapped to the nearest whole hour. `LSTM` is now computed in floating point.

## Project Builder
- The GUI UTC-offset field is now a `float` (was `int`), so fractional-hour time zones can be entered and are passed through to the Context `Location` without truncation.

## Visualizer
- Zero-initialized the `Visualizer` OpenGL handle members (`framebufferID`, `depthTexture`, `texArray`, `texture_array_layers`, and the offscreen FBO/texture handles) at declaration. In headless mode the windowed-only shadow-map framebuffer and depth texture are never created, so `depthTexture`/`framebufferID` were left uninitialized; the render path unconditionally binds `depthTexture` as the shadow map, so a headless instance constructed after a windowed instance was destroyed could bind a stale GL texture name left in freed memory, producing `GL_INVALID_OPERATION` and a driver-side crash on macOS (observed as a segfault in the `Visualizer::PNG with transparent background` self-test when run after the windowed `printWindow` regression test). Known remaining issue: the windowed-mode `printWindow` self-test (`selfTest.cpp:673`) can still intermittently crash on macOS due to a separate, timing-sensitive GL-driver worker-thread race that only surfaces after many Visualizer create/destroy cycles in one process; this does not affect normal single-Visualizer usage and is tracked separately.

# [1.3.72] 2026-05-06

## Core
- Added `Context::deleteTimeseriesVariable()` to delete a single timeseries variable and all of its data points, identified by label. Issues a non-fatal warning if the variable does not exist; complements the existing `clearTimeseriesData()` (which removes all variables) and `updateTimeseriesData()` (which modifies a single point).

## Plant Architecture
- Added leaf-to-fruit nitrogen translocation in `removeFruitNitrogen()`. Per-plant fruit demand is now aggregated across all fruiting buds and first satisfied from the available pool; any remaining shortfall is drawn from leaves, with old leaves (age ≥ `remobilization_age_threshold`) donating first and young leaves serving as a fallback. Per-leaf availability uses the same `(current_N_area − minimum_leaf_N_area) × leaf_remobilization_efficiency` reserve as leaf-to-leaf remobilization, distributed proportionally across source leaves; no stress-acceleration multiplier is applied because withdrawal is governed by fruit demand rather than stress signaling. Residual demand that even leaves cannot cover is silently absorbed, preserving the preexisting fail-soft accounting.
- Petiole geometry is now suppressed when either `petiole.radius` or `petiole.length` is zero (previously only radius=0 suppressed it); in that case no petiole tube object is created and leaves attach directly at the internode tip. Vertices are still computed so leaf orientation is well-defined.
- Added `Phytomer::rotatePetiole(petiole_index, AxisRotation)` for solid-body rotation of a petiole and all leaves anchored to it (the terminal floral bud and its peduncle/inflorescence, anchored to the internode, are unaffected). Pitch rotates about the stored `petiole_rotation_axis` using the construction-time `abs(pitch)` convention, yaw rotates about the internode axis, and roll rotates about the petiole's current length axis. Used internally by the sorghum flag-leaf rule, which now sets the terminal phytomer petiole pitch to a small non-zero value (5°) to keep the rotation axis well-defined and applies an additional +25° solid-body petiole rotation so the flag leaf clears the panicle.
- Reworked leaf orientation on curved stems: the legacy roll term `(acos(internode_axis.z) − roll)` is removed and replaced by a curvature-aware blade-up correction applied after the pitch/yaw chain. The leaf is rolled about its own length axis (the petiole tip axis) by an angle scaled by `min(petiole_length / leaf_size, 1)` and clamped below 90°, so leaves on long petioles hang at their natural angle while leaves that hug the stem follow stem curvature. Also fixed leaf-wave vertex displacement to be applied along the post-rotation surface normal rather than along world Z, so wave bumps no longer flatten visually past the buckle.
- Tightened maize and sorghum library parameters: widened phyllotactic angle (170–190° → 155–205°), increased leaf buckle angles, increased sorghum wave amplitude, lengthened maize wave period (0.1 → 0.15), and relaxed sorghum gravitropic curvature (−1000…−400 → −800…−200) for less-extreme stem droop.

## Leaf Optics
- Added Fluspect-B SIF parameters `V2Z` (violaxanthin↔zeaxanthin de-epoxidation state) and `fqe` (intrinsic fluorescence quantum efficiency scalar) to `LeafOpticsProperties` and `LeafOpticsProperties_Nauto`. They are ignored by the PROSPECT reflectance/transmittance calculation; on every `LeafOptics::run()` call (and per-bin in the nitrogen-auto path), Helios now writes an 11-element `fluspect_biochem_<label>` global-data vector containing `[Cab, Cca, Cw, Cdm, Cs, Cant, Cp, Cbc, N, V2Z, fqe]` and stamps the primitive data label `fluspect_spectrum` on processed UUIDs so the radiation plug-in's SIF pipeline can identify fluorescing leaves and look up their biochemistry by label.

## Photosynthesis
- Added an implementation of the von Caemmerer (2021) steady-state C4 photosynthesis model, exposed via `setModelType_C4()` and a new `C4ModelCoefficients` struct that mirrors the C3 Farquhar API (four temperature-response setter overloads per rate parameter `Vpmax`/`Vcmax`/`Jmax`/`Rd`/`gm`, plus directly-editable kinetic constants and structural scalars). Includes a fail-fast species library (`SetariaViridis_vC2021`, `GenericC4_vC2000`, `Maize_Massad2007`) accessed via `setC4CoefficientsFromLibrary()`, per-UUID and per-material coefficient assignment with full material-data serialization, a `setCm()` override for direct mesophyll-CO2 prescription (testing/validation against the paper's reference spreadsheet), and new optional output primitive data `Cm` (mesophyll cytosolic CO2) and `Vp` (PEP carboxylation rate). The `limitation_state` output uses 1 = enzyme-limited and 2 = electron-transport-limited for the C4 model.
- Added mesophyll conductance `gm` to the Farquhar C3 model via four `setMesophyllConductance_gm()` overloads on `FarquharModelCoefficients` (constant / Arrhenius / peaked / peaked+dHd) with the same temperature-response framework as the other rate parameters. When `gm` is finite, net assimilation is obtained by solving the Ethier & Livingston (2004) / Sharkey et al. (2007) quadratics derived from `Cc = Ci - A/gm` for the Rubisco- and electron-transport-limited branches; the default value is +infinity, which short-circuits to the legacy `Cc ≡ Ci` path and reproduces the original Farquhar `A` bit-for-bit so existing code and parameter libraries that do not set `gm` are unaffected.

## Radiation
- Added a solar-induced chlorophyll fluorescence (SIF) modelling pipeline to `RadiationModel`, exposed through a new SIF camera type. Users add a camera via `addSIFCamera()` (taking a list of user-defined emission bands, a `SIFCameraProperties` struct with `excitation_bin_width_nm` and `excitation_scattering_depth`, and standard camera geometry); `isSIFCamera()` queries whether a label was registered as SIF. Helios auto-creates internal excitation bands spanning 400–750 nm at the requested bin width, ray-traces them to populate per-leaf APAR (with optional inter-leaf scattering when the user wants accurate NIR APAR), evaluates a per-leaf Fluspect-B excitation→emission kernel (Vilfan 2016, ported in `FluspectB.cpp`) cached by biochemistry label + bin width, scales by the van der Tol (2014) rate-coefficient quantum yield Φ_F (read from `electron_transport_ratio` and `temperature` primitive data), integrates per emission-band wavelength range, and injects the resulting per-leaf top/bottom source flux (Mf/Mb) into the standard `runBand()` emission loop so escape and reabsorption are 3D-faithful. Excitation bands are piggy-backed onto regular non-SIF dispatches when present, and cached APAR is invalidated when radiative properties change. SIF-flagged emission bands bypass the Stefan-Boltzmann ε+ρ+τ=1 check (still enforcing ρ+τ ≤ 1) and skip the broadband σ·ε·T⁴ fallback for primitives without an SIF source flux, so non-emitting primitives (ground, sensors) do not inject thermal energy into visible/red SIF bands. Each emission band is locked to one excitation bin width — re-flagging it from a second camera with a different width is an error.
- CMake now disables the Vulkan radiation backend (rather than emitting an unresolvable `-lvulkan` linker flag that broke downstream linking) when no Vulkan loader is installed on Linux/macOS, with a status message pointing at the appropriate install command.
- `VulkanComputeBackend::copyScatterToRadiation()` now sizes the scatter→radiation memcpy by `launch_band_count` rather than the global `band_count`, fixing an OOB read crash whenever a `runBand` dispatch is a strict subset of all bands (e.g., a SIF emission-only dispatch following an excitation-band dispatch that raised the global band count).
- `addRadiationCamera()` now warns once per camera (aggregated across all bound bands) when a camera is bound to a band with `scatteringDepth == 0`, since camera pixels for that band would otherwise read as zero with no diagnostic.
- Spectral-camera auto-exposure now anchors to the 95th percentile mapped to ~0.7 instead of the median mapped to 0.18, so narrow scientific bands (fluorescence, thermal IR, Fraunhofer-line retrievals) where most of the frame reads near zero no longer drive the gain to astronomical values and saturate the subject.
- `runBand()` now skips cameras whose bound bands don't intersect the current dispatch, avoiding wasted full-camera launches when (e.g.) running PAR with a SIF camera registered.
- BVH traversal kernels (`traverse_bvh`, `traverse_cwbvh`) drop the unused `hit_prim_type` and `hit_uv` outputs and the corresponding leaf-data load in cwBVH, removing dead reads from every camera/diffuse/direct/pixel-label raygen path.

## Visualizer
- Fixed `colorContextPrimitivesByObjectData()` (no-arg overload) so that orphan primitives (those with no parent compound object) are colored by the colormap at value 0 — matching the existing behaviour for compound-object primitives whose parent object lacks the data label, and matching the behaviour of `colorContextPrimitivesByData()` for primitives without primitive data. Previously orphans were rendered with their base RGBA, which left ground tiles and other standalone primitives showing through as their default color (e.g., the green from `Context::addTile`) instead of participating in the colormap.
- Colorbar ticks are now filtered to the colorbar's `[cmin, cmax]` range (with a small epsilon) before placement, so a fixed-range colorbar no longer shows orphan labels at "nice" tick values that `generateNiceTicks()` extends past the data range.

# [1.3.71] 2026-04-14

## Core
- OBJ loader now generates unique material names when loading multiple OBJ files with colliding MTL material names (e.g., Blender's default "Material.001"), instead of silently reusing the first registered material.
- Added `Context::updateTimeseriesData()` method to modify the value of an existing timeseries data point, identified by its (Date, Time) key. Throws a runtime error if the variable or specified timestamp does not exist.
- `helios::writeEXR()` now maps common channel name aliases (`red`/`green`/`blue`/`alpha`, case-insensitive) to the standard EXR channel names `R`/`G`/`B`/`A` so that images written with descriptive band labels are recognized as RGB by external EXR viewers.

## Plant Architecture
- Added `writePlantStructureUSD()` method to export plant structures as USDA files with PhysX articulated rigid bodies for NVIDIA IsaacSim physics simulation. Each tube segment becomes a capsule-shaped rigid link connected by spherical joints whose local frames encode the rest-pose orientation (zero joint angle reproduces the authored pose), with spring/damper drives derived from beam bending stiffness. Visual meshes use smooth area-weighted faceVarying normals and indexed primvars, materials are nested under the default prim, and a single shared angular drive replaces the previous per-axis rotX/rotY drives.
- Added `registerGrowthFrame()`, `writePlantGrowthUSD()`, `clearGrowthFrames()`, and `getGrowthFrameCount()` methods to capture per-step plant geometry snapshots and export them as a time-sampled USDA animation for Blender. Organs that appear during growth are toggled visible via scale/visibility keyframes, with step-style hold keyframes preventing interpolation between growth frames.
- Fixed `readPlantStructureXML()` to use the plant instance's base position instead of the origin when reconstructing the first internode of the base shoot.
- Fixed operator-precedence bug in `random_float()` helper used by Cowpea phytomer generation, where `RAND_MAX / (max - min)` was being divided before the cast instead of after, producing skewed random ranges.
- Floral bud `age` is now tracked from the time of bud break and written as object data on inflorescence and peduncle objects (when the `age` output is enabled), so flower/fruit organs report a meaningful age rather than inheriting the parent phytomer's age.

## Leaf Optics
- `LeafOptics::run()` now assigns spectra to newly added primitives within an existing object, instead of skipping them when the object's nitrogen content has not changed enough to trigger a bin reassignment.

## Photosynthesis
- Reformulated TPU limitation in the Farquhar model to compare at the assimilation level (`A_TPU = 3*TPU - Rd`) rather than via `Wp = 3*TPU*Ci/(Ci - Gamma*)`, eliminating the singularity at `Ci = Gamma*` that produced non-physical assimilation values when intercellular CO2 approached the compensation point. Also fixed `evaluateCi_Farquhar()` to read `TPUflag` from the model coefficients struct instead of the (uninitialised) variables vector.

## Plant Hydraulics
- Fixed `computeCapacitance()` to use the first-derivative finite-difference stencil `(f2 - f0)/(2h)` instead of the second-derivative stencil `(f0 - 2f1 + f2)/h^2`, since hydraulic capacitance is defined as `Wsat / (dpsi/dw)`.
- Fixed `setStemHydraulicConductanceTemperatureDependence()` to modify the stem conductance struct instead of the root conductance struct.
- Fixed `setLeafHydraulicCapacitanceFromLibrary()` parameters: `PistachioFemale` had a positive (physically impossible) osmotic potential at full turgor and used Walnut's RWC at turgor loss; the unknown-species fallback claimed to be Walnut but used Western Redbud values.
- Fixed per-primitive coefficient lookup in `updateLeafWaterPotentialsOfPlant()`, which indexed `modelcoeffs_map` by the loop counter instead of the primitive UUID, silently discarding any per-primitive coefficients set via `setModelCoefficients(coeffs, UUIDs)`.
- Fixed non-steady-state leaf RWC update to divide the net flux by `saturated_specific_water_content`, so that leaves with a larger water reservoir respond more slowly (previously the dimensionless RWC was incremented by raw `mol/m^2` flux).
- Fixed `computeConductance()` and `updateRootAndStemWaterPotentialsOfPlant()` convergence test to use `fabs()` instead of `abs()`, which on float arguments could resolve to the integer overload and truncate intermediate values (also caused `delta_psi` in the root/stem solver to be signed, allowing premature exit).
- Added `setLeafHydraulicCapacitance()` overload that accepts an explicit `saturated_specific_water_content` argument.

## Parameter Optimization
- Added four new optimization algorithms: L-BFGS and BOBYQA (local, NLopt-backed), SLSQP (local constrained, NLopt-backed), and Adam/AdamW (noise-tolerant, no external dependency). The existing GA, BO, and CMA-ES algorithms are unchanged.
- Added a composable gradient architecture (`GradientSource`, `makeGradientFunction()`, `makeFDGradientSource()`, `makeFDGradient()`) that lets users mix analytic/AD gradients with centered finite differences on a per-parameter basis, plus a `ConstrainedSimulation` overload that returns the objective, constraints, and all sensitivities in a single cached forward pass.
- Added three new `ParameterOptimization::run()` overloads for gradient-based, constrained, and combined-simulation optimization, and an automatic default-algorithm selector chosen from parameter properties and the overload called.
- NLopt 2.10.1 is now bundled under `plugins/parameteroptimization/lib/nlopt-2.10.1/` and built as a static library when CMake is configured; a system-installed NLopt is used as a fallback. If NLopt is unavailable, L-BFGS/BOBYQA/SLSQP are disabled and the remaining algorithms still build. Credit to Kyle Rizzo for developing the kernel-extraction/AD-enabling extensions.

## Radiation
- Added Basler acA2500-20gc (color, GigE) and acA2500-20gm (monochrome, GigE, base + 730 nm and 850 nm bandpass variants) to the camera library.
- Fixed `RayTracingGeometry::validate()` to size shared per-primitive arrays (`primitive_types`, `transform_matrices`, `object_subdivisions`, `twosided_flags`, `solid_fractions`, `object_IDs`, `primitive_IDs`) by `primitive_count` only; bbox geometry is stored in its own `bboxes.UUIDs` / `bboxes.vertices` buffers, so the previous `primitive_count + bbox_count` expectations crashed validation in Debug builds with periodic boundaries enabled.
- Vulkan backend now destroys the `band_map_buffer` during teardown, fixing a Vulkan resource leak on RadiationModel destruction.
- `updateRadiativeProperties()` now writes reflectivity/transmissivity directly into the flat `material_data` buffers via index helpers, eliminating large nested `std::vector` temporaries (previously `O(Nsources * Nprimitives * Nbands * Ncameras)` of nested allocation followed by a redundant `flatten()` copy).
- Camera output now records the actually applied exposure gain and white balance factors on the `RadiationCamera` and writes them to the camera metadata JSON, so downstream tools can recover the linear-radiance scaling that was applied to a saved image.
- Fixed `blendSpectra()` weight-sum check to allow floating-point rounding (tolerance `1e-5`) instead of requiring exact equality to 1, which previously rejected mathematically valid weight combinations.

## Visualizer
- Fixed `addPoint()` argument list, which was passing the point size into the wrong parameter slot of `GeometryHandler::addGeometry()` after a recent signature change.

## Build System
- CMake now checks at configure time that the installed CUDA Toolkit version does not exceed the NVIDIA driver's supported CUDA version, preventing cryptic `OPTIX_ERROR_INVALID_INPUT` failures at runtime due to incompatible device code.

# [1.3.70] 2026-03-19

## Plant Architecture
- `deletePlantInstance()` now cleans up hidden prototype primitives from the Context when all plant instances have been deleted, preventing orphaned hidden primitives and materials that could never be freed. Added optional `include_hidden` parameter to `getAllPlantUUIDs()` to allow querying hidden prototype primitives.

## Radiation
- Fixed camera pixel label UUID mapping where the CPU-side conversion from internal primitive indices to Helios UUIDs produced incorrect labels when `buildGeometryData` reordered primitives by parent object. The GPU intersection programs already store actual UUIDs directly, so the redundant conversion was removed.

## Build System
- CMake now verifies that the CUDA language can actually be enabled (e.g., Visual Studio MSBuild integration is installed) before attempting a GPU build in the radiation, collision detection, and energy balance plugins, preventing cryptic build failures when the CUDA toolkit is installed without Visual Studio integration.
- Bundled Vulkan SDK headers and glslang 16.2.0 shader compiler source with the radiation plugin.

# [1.3.69] 2026-03-16

## Core
- Removed redundant `setColor()` call in `addConeObject()` that was re-applying the color already set on individual triangles during construction.

## Plant Architecture
- Renamed generic `Material.002` material in `PetiolulePrototype` OBJ/MTL assets to descriptive `petiolule` label.

## Radiation
- Removed bundled OptiX 5.1 libraries and headers (linux64-5.1.0, windows64-5.1.1) and all legacy `OPTIX_VERSION_LEGACY` CMake code paths, since OptiX 5.1 support was superseded by OptiX 6.5 and 8.1 backends.
- Fixed multi-tile camera rendering in the OptiX 8 backend where `__raygen__camera()` and `__raygen__pixel_label()` used per-tile resolution instead of full image resolution for ray direction computation, causing black pixels in tiles beyond the first (e.g., iPhone 12 Pro Max at 3024×4032 with 100 AA samples).
- Fixed camera closest-hit for triangle primitives in the OptiX 8 backend to use the intersection program's face attribute instead of recomputing the surface normal from canonical patch vertices, which produced incorrect face orientation and black triangle pixels.
- Fixed camera pixel buffer zeroing in the OptiX 6 backend so that multi-tile renders for the same camera accumulate correctly rather than re-zeroing between tiles.
- Specular reflection was missing in OptiX 8 camera rendering path: camera-weighted source fluxes are now uploaded and specular contributions are accumulated per-camera in the direct miss program.

# [1.3.68] 2026-03-15

## Core
- OBJ loader now registers MTL materials in the Context material system and assigns them to loaded primitives, enabling material-based queries and rendering workflows for imported OBJ meshes.
- Added `renameMaterial()` method to allow renaming material labels while preserving all properties and deduplication aliases for auto-generated materials.
- Added `clearTimeseriesData()` method to remove all timeseries variables and their associated date/time values from the Context.

## Plant Architecture
- Plant organ materials (stems, petioles, leaves, flowers, fruit, peduncles) are now given descriptive names based on plant name and shoot type (e.g., "bean_trifoliate_leaf") instead of opaque auto-generated hash labels.
- Added `setProgressCallback()` method to `PlantArchitecture` for receiving real-time `(float progress, std::string message)` updates during `advanceTime()` and `adjustFruitForObstacleCollision()`, enabling GUI and Python binding integration. Added `setCallback()` to `ProgressBar` to support this.

## Visualizer
- Reverted colorbar tick clamping that was incorrectly removing valid tick marks at the data range boundaries.

# [1.3.67] 2026-03-11

## Radiation
- Replaced compile-time backend selection with runtime GPU hardware probing: the radiation model now auto-detects the best available backend (OptiX 8 → OptiX 6 → Vulkan) at startup, with clear diagnostic errors when no compatible GPU is found.
- Bundled Vulkan headers and the glslang shader compiler with the plugin, eliminating the requirement to install the Vulkan SDK on all platforms (Windows now has zero external Vulkan dependencies).

# [1.3.66] 2026-03-05

## Core
- Added `writeEXR()` functions for writing single-channel and multi-channel float images to OpenEXR files with lossless ZIP compression, using the tinyexr header-only library.

## Radiation
- Added `writeCameraImageDataEXR()` and `writeDepthImageDataEXR()` methods for exporting camera and depth data to EXR files, preserving full floating-point precision.

## Visualizer
- Fixed bug in colorbar ticks where ticks could extend past the colorbar.

## Radiation
- Added OptiX 8.1 ray tracing backend for NVIDIA systems with driver ≥ 560. This resolves the driver 590+ incompatibility that prevented the OptiX 6.5 backend from working on modern NVIDIA drivers. The backend is selected automatically at build time: driver ≥ 560 uses OptiX 8.1; driver < 560 continues to use OptiX 6.5.

# [1.3.65] 2026-02-27

## Core
- Expanded `loadTabularTimeseriesData()` to support combined "datetime" columns (ISO-8601 with timezone offsets, compact YYYYMMDDHH/YYYYMMDDHHMM, space-separated date+time), a "time" column (HH:MM, HH:MM:SS), compact 8-digit date strings, and date format synonyms (e.g., "YYYY-MM-DD", "DD/MM/YYYY").
- Fixed `Tube::pruneTubeNodes()` to correctly partition and delete primitives using the proper UUID ordering, and fixed `Tube::updateTriangleVertices()` loop order to match `addTubeObject()` so that colors stay mapped to the correct segments.

## Radiation
- Added Vulkan compute backend with software BVH traversal, enabling GPU-accelerated ray tracing on AMD, Intel, and Apple Silicon GPUs without requiring CUDA or OptiX. Backend selection is now automatic based on available hardware, with `FORCE_VULKAN_BACKEND` CMake option for testing. The OptiX backend now handles its own launch batching internally rather than in `RadiationModel`. Tile sub-patches are treated as individual patches for cross-backend compatibility, and the Prague sky model parameters are now properly passed to all backends.

## Parameter Optimization
- Major overhaul: added Bayesian Optimization (Gaussian Process with UCB acquisition) and CMA-ES algorithms alongside the existing Genetic Algorithm. New `setAlgorithm()` API replaces the old `OptimizationSettings` struct, with configurable crossover operators (BLX-alpha, BLX-PCA), mutation operators (per-gene, isotropic, hybrid PCA), integer/categorical parameter types, fitness caching, and `explore()`/`exploit()` factory presets for each algorithm.

## Plant Hydraulics
- Fixed sign error in turgor pressure term of the water potential equation in the documentation.

## LiDAR
- Updated multi-return triangulation filter documentation to reflect first-return filtering and corrected separation ratio threshold.

# [1.3.64] 2026-01-30

## Core
- Added non-throwing path resolution functions `tryResolveFilePath()` and `tryResolvePluginAsset()` that return empty paths instead of throwing exceptions when files are not found.

## Plant Architecture
- Refactored `resolveTextureFile()` to use non-throwing path resolution functions, eliminating exception handling for file probing.
- Added overloaded `PlantArchitecture::getPlantInternodeObjectIDs()` that can take a shoot type label string in order to only get object IDs for that shoot type.

## Radiation
- Refactored ray tracing architecture with backend abstraction layer to enable future support for OptiX 7.7 and Vulkan backends while maintaining backward compatibility with OptiX 6.5. This is a major overhaul, but should be fully backward compatible - please report any errors you encounter.
- Added type-safe buffer indexing utilities (`BufferIndexing.h`, `IndexTypes.h`) to eliminate manual index calculations and prevent indexing errors in multi-dimensional GPU buffers.

# [1.3.63] 2026-01-21

## Core
- Updated docs to be mobile friendly with slide-out sidebar navigation and hamburger menu.

## LiDAR
- Removed direct CUDA dependency; ray tracing now uses the CollisionDetection plugin for GPU acceleration.
- Renamed `calculateLeafAreaGPU()` to `calculateLeafArea()` (old methods deprecated but still available).
- Added `enableGPUAcceleration()` and `disableGPUAcceleration()` methods to control CollisionDetection GPU usage.
- Added adaptive triangulation filtering for multi-return data using separation ratio thresholds.
- Improved `gapfillMisses()` to track filled grid positions and avoid duplicate gap-filled points.

## Plant Architecture
- Refined carbohydrate model with new parameters for structural vs non-structural carbon tracking (`carbohydrate_percentage`, `stem_structural_carbon_percentage`, `living_wood_fraction`, `dormant_respiration_fraction`) and separate upward/downward transfer thresholds.
- Added `vegetative_bud_break_probability_max` parameter to ShootParameters for controlling maximum bud break probability.
- Added `updateShootFruitCounts()` and `getShootInternodeObjectIDs()` methods.
- Added additional internode object data outputs for carbon model diagnostics (`carbohydrate_pool_molC`, `daily_net_photosynthesis`, `daily_respiration`, `daily_growth`).
- Added two new almond tree model variants: `almond_aldrich` and `almond_wood_colony`.

# [1.3.62] 2026-01-12

## Core
- Fixed `Tube::appendTubeSegment()` to correctly add triangles for new segments using proper vertex indices instead of hardcoded values.
- Refactored `readPNGAlpha()` to use RAII patterns for proper resource management and added PNG signature validation.
- Fixed `cart2sphere()` singularity when converting vectors pointing straight up or down (gimbal lock), which caused camera orientation issues.

## Voxel Intersection
- The voxel intersection plug-in has been deprecated. Use the collision detection instead.

## Collision Detection
- All functionality from the voxel intersection plug-in has been incorporated in the collision detection plug-in. This also includes an OpenMP fallback if no GPU is available.

## Photosynthesis
- Added `electron_transport_ratio` (J/Jmax) as optional output primitive data for computing fluorescence quantum yield.

## Plant Architecture
- Added `listShootTypeLabels()` method with three overloads to discover available shoot types: query the currently loaded model (no parameters), query any model by name (string parameter), or query a plant instance by ID (uint parameter).
- Added `resolveTextureFile()` static method for simplified asset path resolution when loading textures and OBJ models.
- Updates to pistachio and walnut model parameters.
- Fixed nitrogen model to track all leaves in the nitrogen map even when no nitrogen is available in the pool.

## Leaf Optics
- Added optional primitive data output for biochemical properties (chlorophyll, carotenoid, etc.) when using nitrogen-based automatic mode.

## Radiation
- Fixed horizontal mirroring in camera output functions: `writePrimitiveDataLabelMap()`, `writeObjectDataLabelMap()`, `writeImageBoundingBoxes()`, `writeImageSegmentationMasks()`, `writeCameraImage()`, and related methods.
- Refactored COCO annotation generation in `writeImageSegmentationMasks_ObjectData()` to ensure 1:1 correspondence between annotations and their attribute values.
- Fixed `runBand()` to validate band labels from the input parameter rather than an internal variable, which caused incorrect error messages when invalid bands were passed.

# [1.3.61] 2025-12-17

## Core
- Fixed CMake linkage visibility for `stdc++fs` library on older GNU compilers (< 9.0) by specifying `PRIVATE` visibility.
- Added `compact` parameter to `WarningAggregator::report()` for single-line warning summaries.

## Energy Balance
- CUDA is now optional. Plugin uses three-tier execution: GPU (CUDA), OpenMP (parallel CPU), or serial CPU fallback. OpenMP is recommended for most workloads.
- Added `enableGPUAcceleration()`, `disableGPUAcceleration()`, and `isGPUAccelerationEnabled()` methods to control GPU usage at runtime (only available when compiled with CUDA).

## Collision Detection
- Added runtime warning when OpenMP is not available, recommending its installation for better performance.

## Plant Architecture
- Added nitrogen model for simulating nitrogen uptake, allocation, remobilization, and stress effects on plant growth. Uses area-based nitrogen tracking (g N/m²) with three-level pool structure (root → available → per-leaf). Calculates nitrogen stress factor (0-1) for use by other plugins.

## Leaf Optics
- Added nitrogen-based automatic leaf optics mode that computes chlorophyll and carotenoid content from leaf nitrogen concentration. Uses adaptive binning to limit computational overhead while maintaining spectral fidelity across nitrogen gradients. Integrates with PlantArchitecture nitrogen model via `leaf_nitrogen_gN_m2` object data. 

# [1.3.60] 2025-12-08

## Core
- Added vector-scalar arithmetic operators for `std::vector<float>` to enable element-wise operations: addition (`vec + scalar`, `scalar + vec`), subtraction (`vec - scalar`, `scalar - vec`), multiplication (`vec * scalar`, `scalar * vec`), and division (`vec / scalar`, `scalar / vec`).
- Fixed an issue with shared materials. If duplicate materials were merged, and the color/texture/etc. is changed for a subset of primitives, all primitives sharing that merged material would be affected.

## Photosynthesis
- Added `setCi()` method to manually override intercellular CO2 concentration for specified primitives, bypassing the normal iterative calculation that couples photosynthesis with stomatal conductance. This feature is primarily intended for testing and validation purposes.
- CRITICAL BUG: Fixed bug in `evaluateFarquharModel()` where incorrect variable name was used when calling `fzero()`. This could result in parameters reverting to the default values in certain cases.
- Added check for Farquhar `Topt` parameters to catch instances where users specify them in units of Kelvin instead of Celsius.

## Plant Architecture
- Substantially refactored XML reading/writing to avoid writing position and orientation vectors to XML. Instead, they are auto-computed based on bulk parameters.

## Radiation
- Added zoom parameter to the radiation camera parameters structure. By default zoom is 1x, so should be backward compatible.
- Added lens flare model to radiation camera.
- Radiation sources are now rendered in the radiation camera images if in view of the camera.

# [1.3.59] 2025-12-02

## Core
- Removed public object pointer getter methods (`getObjectPointer()`, `getTileObjectPointer()`, `getSphereObjectPointer()`, `getTubeObjectPointer()`, `getBoxObjectPointer()`, `getDiskObjectPointer()`, `getPolymeshObjectPointer()`, `getConeObjectPointer()`) to enforce encapsulation. Users should use the existing public getter/setter methods for object properties instead.
- Added `scaleConeObjectLength()` and `scaleConeObjectGirth()` methods to scale Cone object dimensions without requiring direct pointer access.
- Added `getGlobalDataVersion()` method for detecting when global data has been modified.

## Leaf Optics
- Added `optionalOutputPrimitiveData()` method to selectively control which biochemical properties are written as primitive data, improving performance when only specific properties are needed.

## Solar Position
- Added Prague Sky Model interface for computing physically-based sky radiance distributions: `enablePragueSkyModel()`, `updatePragueSkyModel()`, `isPragueSkyModelEnabled()`, `pragueSkyModelNeedsUpdate()`. The model accounts for Rayleigh and Mie scattering to produce spectral-angular sky radiance parameters stored in Context global data.

## Radiation
- Moved Prague Sky Model from radiation plugin to solar position plugin; radiation now reads pre-computed Prague spectral parameters from Context global data.
- Changed `setDiffuseSpectrum()` to apply globally to all bands (existing and future) rather than per-band.
- Adopted `WarningAggregator` for property validation warnings during radiative property updates.

## Collision Detection
- Adopted `WarningAggregator` for BVH construction and collision detection warnings.

## Energy Balance
- Adopted `WarningAggregator` for air energy balance warnings; removed debug print statements.

## Voxel Intersection
- Adopted `WarningAggregator` for primitive slicing warnings; replaced manual error throws with `helios_runtime_error()`.

# [1.3.58] 2025-11-28

## Core
- Added materials to avoid duplicating surface-level data. This Phase 1 implementation focused only on the primitive/object color or texture.
- Added `WarningAggregator` class, a thread-safe utility for aggregating repetitive warnings to prevent console flooding when processing large numbers of primitives. Used throughout core functions like `fzero()`, `incrementPrimitiveData()`, and `rotatePrimitive()` to collect and report warning summaries instead of per-iteration messages.

## Plant Architecture
- Users can now configure canopy-level plant parameters, such as trellis dimensions, number of tree scaffolds, etc.
- Added support for "all" keyword in `optionalOutputObjectData()` to enable all optional output data fields simultaneously (case-insensitive). Invalid data labels now throw `helios_runtime_error` instead of printing warnings.

## Radiation
- Added camera library with pre-configured real-world cameras including intrinsic parameters and manufacturer-measured spectral response curves. Library includes Canon EOS 20D, Nikon D700, Nikon D50, Apple iPhone 11, and Apple iPhone 12 Pro Max with full spectral response data for RGB channels.
- Added `addRadiationCameraFromLibrary()` method to create cameras directly from library entries, automatically loading all camera properties, spectral responses, and creating radiation bands if needed. Supports custom band name mapping.
- Enhanced camera metadata with lens properties (make, model, specification), exposure settings (mode, shutter speed), white balance mode, and agronomic properties (plant species, counts, heights, phenological stages, leaf area). Metadata now includes image_processing parameters when `applyCameraImageCorrections()` is applied.
- Added `updateCameraParameters()` and `getCameraParameters()` methods for runtime modification and retrieval of camera intrinsic parameters.
- Added automatic exposure and white balance application during rendering based on camera settings. New `applyCameraExposure()` and `applyCameraWhiteBalance()` methods are called automatically. Supports "auto", "manual", and ISO-based exposure modes (e.g., "ISO100") calibrated to match auto-exposure at reference settings.
- Enhanced metadata API with `enableCameraMetadata()` and `getCameraMetadata()` methods for better control over automatic JSON metadata export alongside camera images.
- Renamed `applyImageProcessingPipeline()` to `applyCameraImageCorrections()` for clarity. Old method is deprecated but still functional for backward compatibility.
- Fixed camera ray generation pixel coordinate calculations when using camera tiling. Now correctly computes global pixel coordinates using `camera_pixel_offset_x`, `camera_pixel_offset_y`, and `camera_resolution_full` variables in OptiX kernel.

## Stomatal Conductance
- Model parameters can now be set on a per-material basis

## Photosynthesis
- Model parameters can now be set on a per-material basis
- `twosided_flag` is now managed as part of materials

## Boundary-layer Conductance
- `twosided_flag` is now managed as part of materials

## Visualizer
- Fix to a visualizer test that was causing a segmentation fault when tests are run in non-headless mode

##	Solar Position
- Implemented SSolar-SOA model to predict full solar spectrum as a function of atmospheric conditions

# [1.3.57] 2025-11-18

## Radiation
- **NEW FEATURE:** Integrated Prague Sky Model for physically-based atmospheric sky radiance in camera rendering. When camera "miss" rays don't hit any geometry, sky radiance is now computed using the advanced atmospheric physics model from Wilkie et al. (ACM SIGGRAPH 2021, CGF 2022) based on Rayleigh and Mie scattering.
- Prague Sky Model data file (27 MB) and Apache 2.0 license added to `plugins/radiation/spectral_data/prague_sky_model/`

## Solar Position
- Clarified turbidity parameter documentation across all methods to explicitly state it represents Ångström's aerosol turbidity coefficient (β), which is aerosol optical depth (AOD) at 500 nm wavelength
- Updated documentation for `setAtmosphericConditions()`, `getAtmosphericConditions()`, `getSolarFlux()`, `getSolarFluxPAR()`, `getSolarFluxNIR()`, and `getDiffuseFraction()` to include typical turbidity values and clarify the difference from Linke turbidity.

## Leaf Optics
- Added built-in species library with pre-configured optical properties for common plant species to simplify model usage and eliminate the need to manually specify PROSPECT-D biochemical parameters
- Added `LeafOptics::getPropertiesFromLibrary()` method to populate `LeafOpticsProperties` structures with species-specific PROSPECT-D parameters. The method accepts case-insensitive species names (e.g., "corn", "CORN", "Corn" are equivalent) and gracefully falls back to default values with a warning if an unknown species is provided.
- Species library includes 11 species fitted to LOPEX93 spectral library samples using robust optimization: default (original Helios defaults), garden_lettuce, alfalfa, corn, sunflower, english_walnut, rice, soybean, wine_grape, tomato, common_bean (from GEMINI field experiments), and cowpea (from GEMINI field experiments)
- All species use PROSPECT-D mode with fitted parameters for numberlayers (N), chlorophyllcontent (Cab), carotenoidcontent (Car), anthocyancontent (Ant), brownpigments (Cbrown), watermass (Cw), and drymass (Cm)

## Plant Architecture
- Added `PlantArchitecture::determinePhenologyStage()` method to determine if a plant is in dormant, vegetative, reproductive, or senescent phenological stage based on the presence and state of floral buds and leaf senescence
- Added optional object data output fields for improved plant identification and analysis: `plant_name` (string), `plant_height` (float), `plant_type` (string: "herbaceous", "deciduous", or "evergreen"), and `phenology_stage` (string: "dormant", "vegetative", "reproductive", or "senescent")
- Plant type classification (herbaceous, deciduous, evergreen) is now tracked internally via `plant_type_map` and automatically set during plant model registration, enabling proper phenological stage determination and classification
- Fixed bug when generating plant from XML, caused by not re-generating prototype models

## Visualizer
- Fixed bug where navigation gizmo would not reappear after calling `Visualizer::displayImage()` followed by `Visualizer::buildContextGeometry()`. The gizmo's enabled state is now tracked and properly restored when building geometry.

# [1.3.56] 2025-11-12

- Made a change to `.clang-format` to not sort header includes, which was persistently causing issues in the project builder plug-in

## Core
- Fixed critical bug in `Texture::computeSolidFraction()` where triangle winding order (clockwise vs counter-clockwise) was not handled correctly. The half-space test for point-in-triangle determination assumed counter-clockwise winding, causing all pixels in clockwise-wound triangles to be rejected, resulting in zero solid fraction values. Now uses shoelace formula to detect winding order and flip half-space coefficients as needed.
- Fixed bug in `Context::copyPrimitive()` for textured triangles where solid fraction was incorrectly recalculated using geometric area instead of being directly copied from the source primitive. This caused zero solid fractions to perpetuate through primitive copying operations.
- Fixed bug in `Context::addTubeObject()` where very small or zero radii at tube ends created degenerate triangles with near-zero surface area, generating numerous warnings. Now clamps radii to a minimum threshold (1e-5) that is large enough to avoid degenerate triangles but small enough to appear as a point visually.
- Fixed bug in `validateOutputPath()` where directory paths without trailing slashes were not properly handled. Now ensures directory paths always end with a trailing slash for consistent path concatenation behavior. Added regression test for this fix.

## CanopyGenerator
- Fixed incorrect triangle winding order in VSP grapevine `leafPrototype()` function. The bottom half of leaves (y<0) had reversed vertex ordering, causing normals to point in the opposite direction from the top half. Both halves now use consistent counter-clockwise winding order following the right-hand rule.

## Plant Architecture
- Fixed bug where child shoot insertion angles were incorrect when using multiple petioles per internode. The angular offset calculation now properly accounts for the number of petioles per internode rather than the number of axillary buds. Added regression test for this fix.
- Added comprehensive support for exact geometry restoration from XML files:
  - Added `Phytomer::scalePetioleGeometry()` method to restore exact petiole dimensions that may differ from parameter-based values
  - Added `PlantArchitecture::comparePlantGeometry()` debugging method for validating XML read/write operations by comparing geometry between original and loaded plants
  - Inflorescence rotation parameters (pitch, yaw, roll, azimuth, peduncle_axis) are now stored in `FloralBud::inflorescence_rotation` and restored from XML
  - Individual flower/fruit base scales are now stored in `FloralBud::inflorescence_base_scales` and restored from XML
  - Peduncle radii are now stored alongside vertices in `Phytomer::peduncle_radii` for exact reconstruction
  - Refactored inflorescence geometry creation into `Phytomer::createInflorescenceGeometry()` helper method to enable deterministic reproduction during XML restoration
- Made `RandomParameter` distribution members (`distribution`, `distribution_parameters`) public to support XML serialization
- Optimized all parameter assignment operators to check if distribution is constant before resampling, avoiding unnecessary random number generation
- Fixed peduncle curvature calculation to bend toward or away from vertical axis using horizontal bending plane, preventing incorrect curvature behavior when peduncle orientation changes
- Fixed bug where peduncle vertices were not being updated when phytomer position was translated via `setPetioleBase()`

## Radiation
- Enhanced camera calibration to support multiple colorboards simultaneously in the same scene. `CameraCalibration::detectColorBoardType()` has been replaced with `CameraCalibration::detectColorBoardTypes()` which returns a vector of all detected colorboard types. The internal storage has been updated from a single vector to a map organized by colorboard type.
- `CameraCalibration::getColorBoardUUIDs()` has been renamed to `CameraCalibration::getAllColorBoardUUIDs()` and made const. It now returns UUIDs from all colorboards in the scene.
- `RadiationModel::autoCalibrateCameraImage()` now processes all colorboards in the scene for calibration, combining patches from multiple colorboards (e.g., DGK, Calibrite, SpyderCHECKR) to improve calibration accuracy when multiple colorboards are present.
- Adding a colorboard of the same type now replaces the previous colorboard of that type with a warning message, rather than clearing all colorboards.
- Added UTF-8 encoding compiler flag (`/utf-8`) for MSVC in CMakeLists.txt to properly handle Unicode characters in source files
- Fixed Unicode em-dash characters in `RayTracing.cuh` comment headers by replacing with ASCII dashes to avoid encoding issues
- Updated Calibrite ColorChecker Classic colorboard spectral data
- Updated leaf surface spectral library with expanded spectral measurements
- Added automatic JSON metadata export for camera images. When `RadiationModel::setCameraMetadata()` is called for a camera, a JSON metadata file is automatically written alongside the image when `RadiationModel::writeCameraImage()` is called. The metadata includes camera properties (resolution, focal length, aperture, sensor dimensions, model name), geographic location, acquisition date/time, and lighting conditions.
- Added `RadiationModel::populateCameraMetadata()` method to automatically populate camera metadata from camera parameters and simulation context (date, time, location, lighting sources). Optical focal length, sensor dimensions, aperture f-stop, camera tilt angle, and light source type are all calculated automatically.
- Added `CameraProperties::sensor_width_mm` parameter to specify physical sensor width in mm (default 35mm for full-frame sensors). This is used to calculate optical focal length when metadata is auto-populated.
- Added `CameraProperties::model` parameter to specify camera model name (e.g., "Nikon D700", "Canon EOS 5D") for documentation purposes in exported metadata.
- **BREAKING CHANGE:** `CameraProperties::FOV_aspect_ratio` is now deprecated and automatically calculated from `camera_resolution` to ensure square pixels (FOV_aspect_ratio = horizontal_resolution / vertical_resolution). Explicitly setting this parameter will trigger a warning and the value will be ignored.
- Renamed `RadiationCamera::applyCameraSpectralCorrection()` to `RadiationCamera::whiteBalanceSpectral()` to better reflect its purpose as a spectral-based white balance method. The method now requires spectral response data and will throw an error if not available, rather than silently returning.
- Removed `RadiationCamera::whiteBalanceAuto()` method. Image processing pipeline now uses `whiteBalanceSpectral()` for more consistent and physically-based white balance.

## Visualizer
- Fixed error causing line widths from `Visualizer::addLine` to be fixed at 1.0 regardless of the line width passed

# [1.3.55] 2025-10-18

## Core
- Added `Context::setObjectDataFromPrimitiveDataMean()` method to calculate the mean of primitive data values across all child primitives of an object and set it as object data. Supports float, double, vec2, vec3, and vec4 data types. Only primitives with the specified data label are included in the calculation.
- Improved XML formatting for vector global data types (vec2, vec3, vec4, int2, int3, int4) - now writes each value on a separate line with proper indentation for better readability instead of all values on one line

## Visualizer
- Fixed issue with navigation gizmo where it could be blocked by objects extending past the camera near plane by moving it closer to the near plane (z-position changed from 0.01 to -0.9999)

## Leaf Optics
- Added `LeafOptics::getPropertiesFromSpectrum()` methods to retrieve PROSPECT model parameters from primitives based on their assigned reflectivity spectra. Available as both single primitive and vector overloads. The method queries primitives for their "reflectivity_spectrum" data and assigns the corresponding PROSPECT parameters (chlorophyll, carotenoid, anthocyanin, water, etc.) as primitive data if the spectrum matches one generated by the LeafOptics instance.

## Plant Architecture
- Fixed a bug when `petioles_per_internode` is set to 0, which caused child shoot insertion angles to be incorrect. This caused major issues in the weed models "puncturevine" and "bindweed"
- Fixed a bug where internodes that start out extremely small were only being scaled in the radius but not length. This caused major issues in the weed model "cheeseweed"
- Improved `ShootParameters` documentation comments for `insertion_angle_tip` and `insertion_angle_decay_rate` for clarity

## Radiation
- Cleaned up camera_spectral_library.xml by removing duplicate copyright comments for Canon 20D, Nikon D700, Nikon D50, and SONY NEX-5N camera spectral data

# [1.3.54] 2025-10-16

## Radiation
- Added automatic spectral interpolation methods for dynamic radiative property assignment:
  - `RadiationModel::interpolateSpectrumFromPrimitiveData()` - Interpolates spectra based on primitive data values using nearest-neighbor selection
  - `RadiationModel::interpolateSpectrumFromObjectData()` - Interpolates spectra based on object data values for all primitives in objects
- Enhanced image segmentation mask with optional parameter to calculate and write mean values of specified primitive/object data labels as mask attributes

## Visualizer
- Fixed colorbar aspect ratio distortion that occurred when window aspect ratio changed - colorbar now maintains intended proportions across different window sizes
- Added `colorbar_intended_aspect_ratio` member variable to track design proportions
- Added `updateColorbar()` method to apply aspect ratio corrections during rendering
- Fixed cascading deprecation warnings in `addSkyDomeByCenter()` by implementing the deprecated overload directly with appropriate warning message
- Fixed issue with navigation gizmo where it could be blocked by objects extending past the camera near plane

## Collision Detection
- Enhanced GPU acceleration fallback handling to gracefully disable GPU mode when CUDA is unavailable at runtime instead of crashing
- Added runtime device count check before attempting GPU memory allocation
- GPU memory allocation failures now fall back to CPU-only mode with informative warnings instead of throwing fatal errors

## Project Builder
- Updated segmentation mask writing to reflect the new Radiation API changes above

# [1.3.53] 2025-10-10

# Visualizer
- The visualizer can now render images with a transparent background. This is enabled with `Visualizer::setBackgroundTransparent`
- `Visualizer::printWindow` can now write images in PNG format (necessary for transparent backgrounds)
- Added navigation gizmo to visualizer window to make it easier to orient the view. This can be disabled with `Visualizer::hideNavigationGizmo()`.
- The line width argument to `Visualizer::addLine()` now works properly on all platforms.
- Fixed a bug where rectangle vertices were not returning the correct values when queried due to improper application of transformations.
- By default, the visualizer now uses a gradient background texture image. The user can also manually set their own using `Visualizer::setBackgroundImage()`.
- Added `Visualizer::setBackgroundSkyTexture()` to add a skybox background that automatically scales with the scene size.
- Deprecated `Visualizer::addSkyDomeByCenter()` in favor of `Visualizer::setBackgroundSkyTexture()`.
- Implemented Heckbert's "Nice Numbers" algorithm that generates clean, evenly-spaced tick values for the colorbar.

## Plant Architecture
- Implemented more efficient setting of optional object data `age`, which yields a huge speedup when setting age for many thousands of compound objects.

# [1.3.52] 2025-10-02

## Plant Architecture
- Fixed a bug in the maize phytomer creation function where the scale could be left uninitialized, leading to undefined behavior.
- Removed a duplicate code block in the tomato phytomer creation function.
- Added new apple fruiting wall plant model (`apple_fruitingwall`)
- Fixed fruit set probability calculation when skipping flower stages - now applies compound probability (flower bud break × fruit set) when jumping directly from BUD_ACTIVE to BUD_FRUITING
- Fixed child shoot insertion angle calculation to properly account for parent petiole position using angular offset
- Updated pistachio tree parameters for improved canopy structure

## Project Builder
- Fixed vector bounds checking for write flags (`write_depth`, `write_norm_depth`, `write_segmentation_mask`)
- Fixed initialization of write flag vectors in `xmlGetValues()` - vectors are now properly sized when no XML values are present
- Fixed initialization of write flag vectors when adding new rigs via `addRig()`
- Updated bounding box writing calls to include `classes.txt` filename parameter matching new Radiation API
- Merged in many project builder bug fixes and updates

# [1.3.51] 2025-09-25

- Added 'validation' page in documentation

## Plant Architecture
- Added collision detection documentation
- Many assets were not properly using file resolution system

## Radiation
- Updates to camera calibration and radiation model implementations
- Many assets were not properly using file resolution system

## Visualizer
- Font and shader assets were not properly using file resolution system
- Further improvements to fix `Visualizer::printWindow()` producing black images in some edge cases

# [1.3.50] 2025-09-22

- Updated PugiXML library from version 1.7 to 1.15 with compact mode and XPath disabled for improved performance

## Context
- Fixed a bug in `Context::loadOBJ()` where materials that have a texture mask (`map_d`) and a diffuse color `Kd` were not being loaded correctly. The texture was being applied but it was using the default color.
- Fixed a bug where copied primitives with texture color overridden were not copying the constant RGB color.
- Enhanced OBJ/MTL loading system with improved default color handling in `loadMTL()` function
- Added `isDirectoryPath()` utility function to distinguish between directory and file paths using multiple heuristics

## Plant Architecture
- Fixed a bug that could cause plant parameter contamination between different plant types in the scene.
- Added Bougainvillea flower 3D model assets (OBJ/MTL files and Blender source)
- Updated strawberry and tomato fruit models with improved geometry
- Enhanced documentation with collision detection schematics and perception cone diagrams

## Visualizer
- Fixes to headless rendering mode, including ability to save images without opening a window
- Significant changes to front/back buffer detection hopefully to fix issues with `Visualizer::printWindow()` producing black images. Automated tests added for this in headless and windowed mode.
- Improved shader management and geometry handling
- Enhanced documentation with updated API references

## Radiation
- Updates to RadiationCamera file path discovery to check whether the file is a directory or file, and handle accordingly

# [1.3.49] 2025-09-11

- Ran `.clang-format` on all source files to ensure consistent code formatting.

## Context
- Fixes for zlib and pnglib building on old linux systems

## Radiation 
- Fixed assignment of image IDs in JSON image segmentation mask output files

## Project Builder
- Fixed some issues with include file order
- Loading of XML files now properly uses Helios' file path resolution system
- Copied `dirt.jpg` asset from the visualizer to the project builder so that the default XML works when not building with the visualizer

# [1.3.48] 2025-09-07

- Split tutorials into separate files for better organization

## Context
- Added OpenMP parallelization to `Context::loadOBJ()` for faster processing of large models
- Added comprehensive bounds checking for face and texture indices with detailed error messages
- Better handling of missing texture files and UV coordinate validation
- Corrected vertex scaling to apply individual scale factors per axis instead of uniform scaling
- Skip triangles with area below threshold to avoid rendering issues
- Enhanced `Context::writeOBJ()` with optional `silent` parameter to suppress output messages
- Improved path resolution hierarchy: checks current working directory first, then HELIOS_BUILD environment variable path
- Enhanced `resolveFilePath()` function with better error messages showing all checked paths

## Plant Architecture
- Many updates to the collision detection integration
- Moved attraction points code from collision detection plug-in to plant architecture plug-in, as it doesn't use any of the collision detection functionality like BVH's

## Collision Detection
- Code cleanup and optimization with removal of deprecated functionality
- Streamlined API and improved documentation organization
- Enhanced test suite maintenance and cleanup

## Visualizer
- Minor fixes and improvements to core rendering functionality
- The issue with `Visualizer::printWindow()` outputting black images has seemed to persist. A more robust fix has been implemented that should hopefully resolve this issue once and for all.

## Weber-Penn Tree
- Updated error handling to use modern `helios_runtime_error`.
- Added `WeberPennTree::disableMessages` and `WeberPennTree::enableMessages()`

# [1.3.47] 2025-09-02

- Enhanced `dependencies.sh` script with comprehensive OpenMP support
- Added GitHub Actions workflow for auto-generating and deploying documentation to GitHub Pages with automatic version synchronization
- Added comprehensive OpenMP documentation and build instructions to User Guide
- Added Tutorial 12: Radiation camera and annotation example

## Core
- Added `ProgressBar` class for console progress indication with customizable width and messaging
- Enhanced global utilities with progress tracking capabilities for long-running operations

## Context
- Fixed object data value caching for bulk data operations (`Context::setObjectData` for multiple objects)
- Enhanced object data caching to properly handle vector-based data operations with comprehensive cache management
- Enhanced MTL file loading with improved path resolution for both absolute and relative paths
- Fixed file resolution logic to properly handle relative paths in MTL files relative to OBJ file directories

## CollisionDetection
- Major performance optimizations for gap and solid object detection in plant architecture model:
  - Implemented rasterization-based collision detection with angular binning for optimal path finding
  - Added fast cone-AABB intersection filtering using aggressive early rejection tests
  - Introduced spatial hash grids and OpenMP parallelization for improved scaling
  - Enhanced BVH management with tree-based isolation for spatially separated objects
- Added tests to verify the accuracy of `CollisionDetection::findOptimalConePath()`

## Radiation
- Enhanced camera image writing functions to return output filename for better integration
- Modernized image annotation API with improved vector-based interfaces:
  - Updated `writeImageBoundingBoxes()` and `writeImageBoundingBoxes_ObjectData()` to support multiple data labels and class IDs
  - Enhanced `writeImageSegmentationMasks()` and `writeImageSegmentationMasks_ObjectData()` with vector-based parameters
  - Deprecated single-value versions of annotation functions in favor of more flexible vector-based approaches
- Added automatic image calibration pipeline (`RadiationModel::autoCalibrateCameraImage()`).

## Plant Architecture
- Updated integration with enhanced collision detection optimizations for improved geometric operations

## Visualizer
- Some OpenGL buffers were not being properly deleted when the Visualizer was destroyed, which could lead to memory leaks. This has been fixed.
- Fixed an issue where the visualizer could hang if `Visualizer::plotUpdate()` is called after `Visualizer::plotInteractive()`.

# [1.3.46] 2025-08-25

* Enhanced GitHub Actions workflows with new Windows GPU self-tests and improved CI configuration
* Upgraded documentation build system with improved cross-referencing and layout enhancements

## Context
- Implemented filesystem-based asset path resolution with fallback mechanisms for different installation types
- Added extensive new utility functions in `global.h` for asset management and path validation
- Added comprehensive test coverage for new asset resolution functionality in `Test_functions.h`

## Visualizer
- Fixed some potential segfaults that could occur when calling `Visualizer::plotInteractive()` or `Visualizer::plotUpdate()` in headless mode.

# [1.3.45] 2025-08-21

* Fixed errors in the `radiation_StanfordBunny` and `energybalance_StanfordBunny` samples that were using the deprecated `Context::setPrimitiveData()` signature with explicit type and size parameters.
* Added automatic version change detection and CUDA object file cleanup improvements to the CMake build system
* Enhanced benchmark collection and reporting in `utilities/run_benchmarks.sh`
* Enhanced OpenMP configuration for macOS with automatic Apple Clang + libomp setup via Homebrew detection
* Added OpenMP installation step to GitHub Actions macOS workflow for improved CI compatibility

## CollisionDetection
- Added comprehensive ray-tracing implementation with new `CollisionDetection_RayTracing.cpp` module
- Implemented advanced ray-tracing data structures including `RayQuery`, `HitResult`, `RayTracingStats`, `RayPacket`, and `RayStream` for efficient batch ray processing
- Added GPU-accelerated ray casting with warp-efficient CUDA kernels for optimal performance
- Introduced Structure-of-Arrays (SoA) BVH layout (`BVHNodesSoA`) for improved memory access patterns and cache efficiency
- Added SIMD-optimized ray-AABB intersection tests using AVX2/SSE instructions
- Implemented streaming ray tracer interface for optimal GPU utilization with memory usage statistics
- Enhanced spatial query capabilities including cone intersection queries, voxel ray path length calculations, and proximity-based collision detection
- Added advanced geometry operations for gap detection, optimal path finding, and attraction point detection within perception cones
- Implemented thread-safe primitive caching system for safe multi-threaded ray casting operations
- Added comprehensive GPU memory management with allocation and transfer optimizations
- Enhanced CMakeLists.txt with improved CUDA support and compilation handling
- Extended `HitResult` structure with `path_length` field for unobstructed voxel path length calculations in LiDAR processing
- Added `calculateVoxelPathLengths()` method for efficient ray path length calculations through individual voxels with OpenMP parallelization
- Enhanced `performGridRayIntersection()` to return detailed `HitResult` objects for each hit instead of simple hit counts
- Added ray classification methods `getRayClassificationCounts()` for Beer's law calculations with hit_before/hit_after statistics

## Visualizer
- Fixed an error with `Visualizer::setColorbarRange()` where it was setting the color of values clipped below the lower bound incorrectly by implementing proper bounds checking for normalized color positions.

## Radiation
- Fixed `RadiationModel::updateGeometry()` to properly handle cases when called with an empty Context by adding a check for zero primitives and returning early with a warning message.

## Plant Architecture
- Updated integration with enhanced collision detection capabilities for improved geometric operations

# [1.3.44] 2025-08-10

* Added `BUILD_TESTS` CMake option to conditionally build test executables instead of building them by default
* Test file renamed to `run_tests.sh`, and now runs tests without needing any of the sample projects (`samples/`)
* Deleted most of the `samples/[*]_selftest` directories, as they are no longer needed for testing. `context_selftest`, `energybalance_selftest`, and `radiation_selftest` were left because they are referenced in YouTube tutorials
* Added compiler warning suppression for third-party libraries (zlib, libpng, libjpeg) to reduce build noise
* Helios now builds with OpenMP parallelization enabled by default if available

## Context
- Added automatic filtering of zero-area primitives in addSphere(), addTube(), addDisk(), addCone(), and other geometric construction methods
- Improved tube generation robustness by handling degenerate cross products when axis vectors are parallel or near-vertical
- Added minimum size validation (1e-6) for patch primitives to prevent numerical precision issues
- Added debug warnings for malformed triangles with near-zero area
- Deprecated `Context::getPrimitiveDataType(uint, const char*)` and `Context::getObjectDataType(uint, const char*)` in favor of `Context::getPrimitiveDataType(const char*)` and `Context::getObjectDataType(const char*)`
- Added new `fzero()` function overload that returns convergence status
- Implemented linspace utility functions in `global.h`/`global.cpp` for float, vec2, vec3, and vec4 types

## Plant Architecture
- Fixed leaf prototype scale validation to ensure minimum positive values
- Updated petiole handling to preserve intended leaf generation

## Radiation
- Enhanced camera image processing pipeline with spectral correction and improved white balance workflow
- Added Gray Edge white balance algorithm implementation for color constancy
- Fixed gamma compression application in image processing pipeline
- Improved gain adjustment and histogram equalization for better image quality
- Substantially refactored `RadiationModel::updateGeometry()` and `RadiationModel::updateRadiativeProperties()` methods for improved efficiency:
  - Replaced O(n) std::find operations with O(1) unordered_set lookups in geometry updates
  - Added spectral integration caching to avoid redundant computations
  - Implemented OpenMP parallelization for radiative property calculations
  - Optimized map access patterns and reduced memory allocations

## Photosynthesis
- Improved temporal continuity in Farquhar model by initializing Ci with previous timestep values
- Enhanced convergence behavior with better initial guesses for intercellular CO2 concentration

## Stomatal Conductance
- Enhanced Ball-Woodrow-Berry model convergence with improved initial surface vapor pressure estimates
- Increased numerical solver tolerance and iteration limits for better stability in vapor pressure calculations

## Project Builder
- Removed `linspace` function from `ProjectBuilder.cpp` since it is now available in the Helios core
- `CMakeLists.txt` was missing the conditional include guards for the canopy generator plug-in, which would cause a linker error if the plug-in was not built

# [1.3.43] 2025-08-04

- Added `doctest_utils.h` to help with parsing test command-line arguments.
- Converted all tests to be able to accept command line arguments, e.g., to only run specific tests.

## Context
- Added cached data type lookup methods and automatic type consistency enforcement with casting support for primitive and object data labels.
- Implemented generic value-level caching system with unified template methods `getUniquePrimitiveDataValues()` and `getUniqueObjectDataValues()` for string, int, and uint data types.
- Optimized `getAllUUIDs()` performance with caching to avoid repeated expensive iteration over primitives.

## Plant Architecture
- Fixed invalid petioles_per_internode handling to preserve leaf generation instead of setting leaves_per_petiole to 0
- Enhanced leaf prototype scale validation to prevent zero-area primitives by setting minimum positive values (1e-6) with warning messages for negative scales

## Radiation
- Updated CameraProperties documentation to clarify that FOV_aspect_ratio represents the ratio of horizontal to vertical field of view (HFOV/VFOV)
- Removed restrictive 3-channel requirement for camera image processing pipeline to support cameras with arbitrary channel counts
- Fixed camera ray generation calculations to use proper VFOV scaling based on aspect ratio (1/FOV_aspect_ratio)
- Enhanced histogram equalization bounds checking to prevent array overflow errors
- Improved geometry update logic to only include existing primitives that weren't filtered out for zero area
- Fixed camera ray tracing to properly handle cases where scattering is disabled or all radiation sources have zero flux
- Improved warning message formatting by replacing std::flush with std::endl for proper line termination
- Added conditional checks to ensure camera ray tracing only runs when scattering is enabled with active radiation sources

## Plant Architecture
- Fixed bugs in `PlantArchitecture::pruneBranch()` that could cause a segmentation fault.
- Added many methods for querying branch IDs.

# [1.3.42] 2025-07-30

- Split tutorials into separate files for better organization.
- `detect_GPU_compute.cmake` script updated to fix some issues on Windows systems.

🚨+ NEW PLUG-IN + 🚨
- Added a new `collisiondetection` plug-in for efficiently calculating primitive and object collisions.

# Context
- `Context::generateColormap()` is now public. An error with the "cool" colormap was also fixed.

# LiDAR 
- Added tutorials for several LiDAR application examples.
- `LiDARcloud::addGridWireframeToVisualizer()` now has an optional argument to specify the line width.
- All of the file exporting methods now try to create the output directory if it does not exist.
- Updated error throwing to use `helios_runtime_error()`.
- Added `LiDARcloud::loadTreeQSM()` and `LiDARcloud::loadTreeQSMColormap()` methods to load a TreeQSM output file and build it in the Context using tube objects.
- Made `LiDARcloud::loadASCIIFile()` public, and modified it to take a file name rather than a parameters structure.
- Added overloaded `LiDARcloud::addHitsToVisualizer()` that allows specification of the point color.

# Visualizer
- Added option to `Visualizer::addLine()` to be able to specify the line width.
- Added option to `Visualizer::addPoint()` to be able to specify the point size.
- Refactored how points are visualized to add automatic culling that speeds up rendering of large point clouds.
- Broke `Visualizer.cpp` into four different files, as it was getting huge.
- Fixed an issue introduced in v1.3.40 causing images to be written up-side down when using the `Visualizer::printWindow()` method.
- Fixed an issue that could cause `Visualizer::printWindow()` to write black images on Linux.

# Plant Architecture
- Added `PlantArchitecture::writeQSMCylinderFile()` method to export the plant geometry as a TreeQSM cylinder file.
- Added collision detection capabilities within the plant architecture plug-in. This is in the intial testing phase for now, and will be further developed in future versions.

# Radiation
- Added `RadiationModel::writeImageSegmentationMasks()` and `RadiationModel::writeImageSegmentationMasks_ObjectData()` to directly write COCO JSON segmentation masks for objects in the scene.
- Moved camera-related code to separate file `RadiationCamera.cpp` to improve organization.
- Many modifications to `CMakeLists.txt` file to fix an issue on Windows systems introduced in v1.3.40.
- Fixed an error checking issue with camera processing pipeline functions to allow cameras with more than 3 channels.

# Aerial LiDAR, LiDAR, Energy Balance, and Voxel Intersection
- Some modifications to `CMakeLists.txt` to allow building with the project builder on Windows systems.

# [1.3.41] 2025-07-22

🚨+ NEW PLUG-IN + 🚨
- Added a new `parameteroptimization` plug-in. Credit to Kyle Rizzo for developing this plug-in.

- Applied `.clang-format` to all source files. As such, there are a lot of small changes to many files in this commit.
- Converted the following plug-ins to use the doctest framework: `energybalance`, `photosynthesis`, `radiation`, `stomatalconductance`, `canopygenerator`, `leafoptics`, `lidar`, `planthydraulics`, `plantarchitecture`, `projectbuilder`, `voxelintersection`, `weberpenntree`, and `boundarylayerconductance`.

# Context
- Added `Context::listAllPrimitiveDataLabels()` and `Context::listAllObjectDataLabels()` methods to efficiently get a list of all primitive or object data labels that exist.
- Added `Context::showPrimitive()` and `Context::showObject()` methods to show a previously hidden primitive or object.
- Fixed an error in `Tube::scaleTubeGirth(float S)` where the stored radius value was not being updated (although the actual primitive geometry was).
- Split doctest tests into separate files for better organization.
- An error was corrected in `Context::cleanDeletedUUIDs(std::vector<std::vector<uint>> &UUIDs)` and `Context::cleanDeletedUUIDs(std::vector<std::vector<std::vector<uint>>> &UUIDs)`, which were checking for object IDs instead of primitive UUIDs.

# Boundary-Layer Conductance
- Modified inclined plate model code to fall back to purely free convection when wind speed is zero, rather than giving NaN.

# Plant Architecture
- Re-factored the plug-in to streamline creation of new plant models in the library (see documentation).

# Visualizer 
- Functions for reading/writing of PNG and JPEG files have been removed from the Visualizer. Functions from the Context are now used instead, which reduces redundant code.

# [1.3.40] 2025-07-17

- Improvements to `utilities/run_benchmarks.sh` script with corresponding changes in `utilities/plot_benchmarks.py`.
- Added benchmarks to documentation.
- Added benchmark cases `energy_balance_dragon` and `plant_architecture_bean`.
- Updated required CMake version to 3.15 for plug-ins, and made many updates to plug-in CMakeLists.txt files to use more modern CMake features.
- Started transitioning tests to using the "doctest" framework. This will be a gradual process, and the old self-test framework will still be supported for now.

## Context
- Reverted a previous change in `Context::setPrimitiveData()` and `Context::setObjectData()` to not use range-based for loops when it is an openmp paralell loop.
- There was an issue in `Context::setObjectData()` and `Context::getObjectData()` where the openmp preprocessor directive needed to be placed after the static assert.
- Moved third-party library pugixml into the `core/lib` directory for better organization.
- Improved robustness of geometric intersection functions (`lineIntersection`, `pointInPolygon`, `fzero`).
- Added new geometric utility function `pointOnSegment`.
- Converted the self-tests to use the doctest framework.

## Visualizer
- Updated visualizer code to only transfer buffer data to the GPU that has changed, rather than re-building all geometry every time any geometry changes.
- Changed the GLFW code to continuously update the window even when there are no new input events. This prevents the usual lag that can occur if the user doesn't move the mouse in the window.
- Converted the self-tests to use the doctest framework.

## Radiation
- Removed MacOS OptiX files from the repository.

## Solar Position
- Minor updates based on edge cases found in the self-tests.
- Converted the self-tests to use the doctest framework.

# [1.3.39] 2025-07-09

- More minor documentation fixes.

## Context
- Implemented some improvements for file path handling on Windows systems to improve robustness.
- Added *= and /= operators for `helios::vec2`, `helios::vec3` and `helios::vec4` vector types.

## Visualizer
- Fixed 'haloing' around text.
- Efficiency of changing the visualization type was improved (e.g., via `Visualizer::colorContextPrimitivesByData()` or `Visualizer::colorContextObjectsByData()`), whereby only the color data is changed rather than rebuilding the entire geometry.
- Added `Visualizer::deleteGeometry()` method to delete a geometric element based on its ID.
- The visualizer window can now be freely resized with arbitrary aspect ratio, including making the visualizer window full screen.

## Project Builder
- Console now auto-scrolls to the bottom when new text is added.
- Disabled keyboard controls so that rotating the view via keyboard does not conflict.

## Plant Architecture
- There was an unhandled error when the combination of `leaves_per_petiole` and `leaflet_offset` are too large such that there is no more space along the petiole to place leaflets. There was a similar issue with flowers for the parameters `flowers_per_peduncle` and `flower_offset`, potentially resulting in too many flowers per peduncle.
- Fixed a minor issue where the max leaf age was not properly enforced when the plant hits its maximum age and stops growing.

## Radiation
- Changed how the camera image post-processing pipeline works. It now applies global histogram equalization, followed by an optional saturation, brightness, and contrast adjustment.

## Leaf Optics
- Added documentation page.

## Stomatal Conductance
- Added optional output primitive data to write stomatal conductance model parameters to primitive data.
- 
# [1.3.38] 2025-07-05

- Added `.gitattributes` file and modified build options for Windows to deal with potential file encoding issues on Windows.
- Edit to `utilities/run_samples.sh` to hopefully fix issues with log file outputting within GitHub Actions workflows.
- Moved `AGENTD.md` back to the base directory because it was not being applied globally when in `doc`.

## Context
- Added a check to `Context::addPolymeshObject()` to ensure that the UUIDs don't already belong to another object.
- Refactored `Primitive::setPrimitiveData()`, `Primitive::getPrimitiveData()`, `Context::setPrimitiveData()`, and `Context::getPrimitiveData()` to use templates to avoid a lot of redundant code.
- Refactored `Context::setGlobalData()` and `Context::getGlobalData()` to use templates to avoid a lot of redundant code.
- Re-wrote how templates are handled for `helios::resize_vector()`, `helios::flatten()` and `helios::powi()` to be more robust and explicitly correct.
- Removed overloaded version of `Context::setPrimitiveData( uint UUID, const char* label, HeliosDataType type, uint size, void* data )`, as it is not needed. The vector-based version of `Context::setPrimitiveData()` should be used instead.
- Re-wrote libjpeg error handling to me more robust.

## Project Builder
- Some updates to object geometry updating.

## Plant Architecture
- Added check for very small petiole length or radius so that it can be set by the user to be exactly 0.
- Fixed asset copy commands in `CMakeLists.txt` that could cause an error on some Windows systems.

## Visualizer
- Re-wrote libjpeg error handling to me more robust.
- Fixed texture and shader copy commands in `CMakeLists.txt` that could cause an error on some Windows systems.
- Fixed an error in `Visualizer::displayImage()` where the arguments to `read_JPEG_file()` and `read_png_file()` were swapped.
- Fixed issues with `Visualizer::clearColor()` where the visualizer was not properly reverting to RGB coloring.
- Fixed where the colorbar range was not being automatically updated when switching between different primitive/object data visualizations.
- Fixed an error where tiled textures were showing ghost lines at the seams between adjacent tiles.
- Fixed 'haloing' around texture-masked primitives.
- Implemented a (temporary) fix to make sure that colors are updated if the user successively calls `Visualizer::colorContextPrimitivesByData()`, `Visualizer::colorContextObjectsByData()`, or `Visualizer::clearColor()`. A more efficient long-term fix is still needed.

## Plant Hydraulics
- `plantID` can now be set as primitive or object data.
- Added an example to the documentation of manually grouping leaves into a plant ID
- Added self-test to check `plantID` assignment.

## Radiation, Energy Balance, LiDAR, Aerial LiDAR
- CMake now uses consistent c++ standard for CUDA code based on the standard used for regular c++ code. It also now explicitly sets the c++ standard for Windows CUDA code.

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

# [1.3.35] 2025-06-20

* If the user changes their Helios version, CMake will now automatically rebuild the entire project. This avoids some issues where updating the code version causes unexpected CMake errors because the user is using an old build and forgot to reconfigure.
* Many minor fixes to documentation and docstrings.
* Added github workflow for testing on self-hosted GPU runner.
* Project `CMakeLists.txt` files should now reference `core/CMake_project.cmake` instead of `core/CMake_project.txt`. The old file is still present for backward compatability.
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
- Added Datacolor SpyderCHECKR 24 color calibration board, and associated `CameraCalibration::addSpyderCHECKRColorboard()` method.

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

# [1.3.34] 2025-06-15

- Added `.clang-format` file for consistent code formatting across IDEs and agents.
- Added `AGENTS.md` to give some basic instructions for AI agents.

## Context
 - Reworked `Context::cleanDeletedUUIDs()` and `Context::cleanDeletedObjectIDs()` to erase invalid IDs using iterators, preventing unsigned underflow issues when vectors are empty.
 - Fixed a loop in `Context::loadMTL()` that could cause unsigned underflow.
 - `Context::writeOBJ()` can now maintain object groups based on the value of the primitive data "object_label".
 - `Context::writeOBJ()` was defining normals (if enabled), but they were not being assigned to faces. This has been fixed.
 - Improved stability of `Tube::appendTubeSegment()`, `Context::addConeObject()`, `Context::addTubeObject()`, and `helios::rotatePointAboutLine()` to prevent dividing by nearly zero when the vector length is very small.
 - Added `helios::powi()` function to calculate integer powers of a number, which is more efficient than using `std::pow()` for integer exponents and avoids having to do "a = b*b*b*b", for example.
 - Added methods `Context::scaleObjectAboutOrigin()` and `Context::rotateObjectAboutOrigin()` to scale and rotate an object about its origin, which can be set with `Context::setObjectOrigin()`.

 ## Radiation
 - Added many new camera response spectra to the spectral library (`spectral_data/camera_spectral_library.xml`).
 - The spectral labels were not correct for the default DGK colorboard (`CameraCalibration::addDefaultColorboard()`).
 - Added the ability to create the Calibrite ColorChecker Classic in the scene. Accordingly added explicit functions `CameraCalibration::addDGKColorboard()` and CameraCalibration::addCalibriteColorboard() to add the two available color boards.
 - Added `RadiationModel::applyImageProcessingPipeline()` to apply a digital camera-like processing pipeline to an image.

 ## Plant Architecture
 - Added capability to scale petioles of the same internode independently.

# [1.3.33] 2025-06-08

## Context
- Added `Context::getDirtyUUIDs()` and `Context::getDeletedUUIDs()`.
- dirty_flag was not being properly set to true in `Primitive::setTransformationMatrix`.
- `validateOutputPath()` function should have been marked [[nodiscard]].
- Fixed out of bounds error in `helios::vecmult()`.
- Patch for `helios::deblank()` to avoid possible buffer overflow issues.
- Patch for `helios::separate_string_by_delimiter()` so it always returns the final token and works when no delimiter is found.
- Added a robust file-extension check in `helios::readPNGAlpha()` to handle filenames without dots safely.

## Visualizer
- Changed `GeometryHandler::allocateBufferSize()` to be general for any primitive type rather than being specific for rectangles and triangles in a single call.
- Added `GeometryHandler::doesGeometryExist()`, `GeometryHandler::getAllGeometryIDs()`, `GeometryHandler::getPrimitiveCount()`, and `GeometryHandler::getDeleteCount()`.
- The visualizer now only updates Context geometry that has changed since `Context::markGeometryClean()` was last called.
- Some errors were fixed in the defragmentation routine.
- `Visualizer::printWindow()` was not properly validating the output file. It now auto-appends `.jpeg` if no file extension is given, and properly validates the output file path.

## Radiation
- Inside `RadiationModel::updateGeometry()`, removal of primitives deleted from Context changed from using `std::erase()` to swap-and-pop to improve performance.

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

# [1.3.32] 2025-05-30

## Context
- Minor refactoring to improve code clarity.
- Added pre-defined RGBA colors, which mirror the existing RGB definitions.
- Added `Context::getPatchCount()` and `Context::getTriangleCount()` methods.
- `Context::getPrimitiveCount()` method now has optional argument to control whether to include hidden primitives in the count.
- Added overloaded `Context::hidePrimitive()` and `Context::hideObject()` that takes a single UUID or object ID.
- Implemented caching of primitive solid fraction calculations, which should dramatically speed up build times when there are a lot of Tiles/Triangles with transparency masking.
- Added `uint2`, `uint3`, and `uint4` vector types.
- Added macro `scast` to avoid having to type out `static_cast`.
- Added many additional debug checks for object and primitive methods to catch index out-of-bounds errors.
- Added object type enumeration `OBJECT_TYPE_NONE` to handle the case when the object type is queried but the object ID = 0 (no object assigned).
- Changed how 'dirty' flags are handled in the `Context`. Dirty flags are handled on a per-primitive basis. If a primitive's geometry or primitive data are modified, it will get flagged as dirty.
- Small issue with `libjpeg` and `libpng` include file paths was fixed to avoid possible unusual build errors on some systems.

## Visualizer
- Core of `Visualizer` plug-in was re-written. This should dramatically improve performance, and allows for updating of geometry on a per-primitive basis. It should be backward compatible, except for the API modifications given below.
- `Visualizer::addDisk()` methods have been removed.
- Line size argument for `Visualizer::addLine()` was removed because it didn't do anything.
- Removed `Visualizer::plotInit()` as this seemed to be dead code that was not used anywhere.
- Significant improvements to shadows, which should remove the issue of pixelated shadows when the domain is very large, and erroneous shadows cast on the ground when the aspect ratio of the domain is very high.
- Added `Visualizer::displayImage()` methods to display a PNG or JPEG image in the visualizer window.
- Depth images were re-worked. Calling `Visualizer::plotDepthMap()` works without calling other methods. Calculation of physical depth should now be correct.

## Plant Architecture
- Fixed `CMakeLists.txt` to automatically copy assets into the build directory if they are modified.
- Re-defined `interpolateTube()` function as a private member of the `PlantArchitecture` class to avoid naming clash with the `interpolateTube()` function of the canopy generator plug-in.
- Updates to the carbohydrate model to better represent carbon costs of wood growth. Credit to Ethan Frehner for these updates.

## Canopy Generator
- Minor change needed to achieve compatability with the new way that textures are handled in the `Context`.

## Energy Balance
- Removed `using namespace helios` from `EnergyBalance.cu` because it was causing a naming clash with the new `uint3` vector type.
- Reconfigured the plugin `CMakeLists.txt` to automatically detect all supported GPU compute capabilities and build them all. This should likely result in some performance improvements.

## LiDAR
- Reconfigured the plugin `CMakeLists.txt` to automatically detect all supported GPU compute capabilities and build them all. This should likely result in some performance improvements.
- Calls to `Visualizer::addLine()` were updated to reflect removal of line width argument.

## Voxel Intersection
- Removed `using namespace helios` from `VoxelIntersection.cu` because it was causing a naming clash with the new `uint3` vector type.
- Reconfigured the plugin `CMakeLists.txt` to automatically detect all supported GPU compute capabilities and build them all. This should likely result in some performance improvements.

## Project Builder
- Many more updates to the project builder plug-in, including ability to change the lighting model, texture tiling of the ground, and many others. Credit to Sean Banks for these updates.

## Radiation
- Added specular reflection in synthetic images for collimated and sun sphere sources (still need to implement disk, rectangle, and sphere sources).

# [1.3.31] 2025-05-10

## Context
- Fixed minor compiler warning that appeared within `Context::readPNGAlpha()`.
- Added overloaded `Context::writePLY()` to take a vector of UUIDs. Also added optional parameter to silence output printing.
- Some minor refactoring was done for the existing `Context::writePLY()` method to better validate file names and paths.
- `helios::CalendarDay()` function was not working correctly. The bug appeared to be introduced in the last version.

## Radiation
- Changed `RadiationModel::setDiffuseSpectrum()` such that it always needs to be associated with a band. This prevents confusion about the required order of the method call in the code.
- The spectral data file `plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml`, the diffuse spectrum contained negative values which caused an error.

## Plant Architecture
- Added `PlantArchitecture::isPlantDormant()`.
- Carbohydrate model updated to take a parameter structure rather than having hard-coded model parameters. Thanks to Ethan Frehner for these updates.
- Minor refactoring improvements throughout plant architecture code.
- Error corrected that was causing plants to flower too quickly if starting with closed flowers.

## Visualizer
- Minor refactoring improvements throughout visualizer code.
- There was an error in the mouse-based camera controls that could cause phantom camera rotation while zooming.

## Project Builder
- Lots of project builder updates and bug fixes. Credit to Sean Banks for these updates.

# [1.3.30] 2025-05-01

*Added automated performance benchmarking
*Fixed documentation sidebar navigation collapse issue

## Context
- Added `Context::setPrimitiveNormal()`, `Context::setPrimitiveElevation()`, and `Context::setPrimitiveAzimuth()` methods.
- Added `sample_Beta_distribution()` funciton to randomly sample from a Beta distribution for inclination angle.
- Added `sample_ellipsoidal_distribution()` funciton to randomly sample from an ellipsoidal distribution for azimuth angle.
- Added explicit `normalize()` function for `vec2`, `vec3`, and `vec4` types.
- Minor performance improvements for Helios vector types and `Context` operations.
- Added `OpenMP` parallelization for primitive area calculations from transparency textures.
- Changed the type of 'primitives' and 'objects' from `std::map` to `std::unordered_map` to improve performance.
- Changed `Context::getObjectPrimitiveUUIDs( uint ObjID )` method to support passing an ObjID of 0.
- `Context::getObjectPrimitiveUUIDs( const std::vector<uint> &ObjIDs )` was missing a check for existence of the ObjID object.
- Overloaded `getPrimitiveParentObjectID( const std::vector<uint> &UUIDs  )` added to take a vector of UUIDs.
- When tile objects are created, completely transparent sub-patch tiles are no longer deleted because this causes problems in the radiation model.
- `Context` primitive and object class constructors were made 'protected' so that they cannot be instantiated outside of the `Context`.
- Many new self-tests were added (credit to Ismael Mayanja for this addition).
- Refactoring of PNG and JPEG image to improve robustness.

## Radiation
- For two-sided primitives, diffuse ray launches were split into two passes to dramatically improve thread coherence and thus performance.
- An issue with texture mapping for tile objects was fixed.
- `OptiX` variable declarations were moved to `RayTracing.cuh` header file.
- Moved texture sampling in `rayGeneration.cu` into a separate function to avoid redundant code.
- New method for handling periodic boundaries was implemented to improve performance.
- Completely transparent sub-patch tiles are no longer excluded from calculations.

## Plant Architecture
- Many updates to carbohydrate model code. Credit to Ethan Frehner for these updates.
- Added methods to calculate the leaf azimuth distribution of a plant (see `PlantArchitecture::getPlantLeafAzimuthDistribution()`), and overloaded versions of leaf angle distribution calculation functions that take a vector of multiple plant IDs.
- Added method to set the leaf angle distribution of a plant.

## Visualizer
- `Visualizer::setColorbarRange()` was not properly setting the colorbar range.

## Project Builder
- Many updates and improvements to the project builder plug-in. Credit to Sean Banks for these updates.

# [1.3.29] 2025-04-14

*Tutorial #12 added for canopy water-use efficiency example. (credit to Ethan Frehner for this addition)

## Context
- Added a check to `Context::getUniquePrimitiveParentObjectIDs()` to return an empty output objID vector if given an empty input UUID vector.
- Error corrected with `OBJ` file loading that could cause incorrect material depending on the ordering of values in the material file.
- The latest Xcode version on MacOS caused an error in `libpng`, which was fixed.

## Radiation
- There were many instances where calling setter methods after calling `updateGeometry()` would cause issues. This has been fixed by making sure the setters force recalculation of the radiative properties.

# [1.3.28] 2025-02-25

🚨+ NEW PLUG-IN + 🚨
- project builder plug-in added. GUI and XML interface for creating projects. This is still in beta testing. Credit to Sean Banks for this addition.

++Substantial expansion of self-tests for `boundarylayerconductance`, `energybalance`, `photosynthesis`, `solarposition`, `stomatalconductance`, `syntheticannotation`, and `visualizer` plug-ins
++New documentation theme

## Context
- Many upgrades to substantially improve performance for CPU code.
- Upgrades to `OBJ` file writing, including making Gazebo compatable and adding option to write normals to `OBJ` files. Credit to LeRoy Corentin for this addition.

## Radiation
- Minor change in `RadiationModel::runBand()` to set all primitive data values in a single `Context` method call for efficiency.
- Changed up logic for radiation property setting. If a spectrum is specified for reflectivity or transmissivity, this can be overridden for a single band by specifying the primitive data `"reflectivity_[bandname]"` or `"transmissivity_[bandname]"`.

## Plant Architecture
- `cowpeaLeafPrototype_unifoliate_OBJ()` asset function was incorrect in the last commit, and would cause an error if used.
- The plugin now writes primitive data `"object_label"` for all organs it creates.
- Some minor fixes to bean `OBJ` assets.
- Removed erroneous organ labels in `MaizeTassel.obj` and `TomatoFlower.obj` assets.

## Boundary-Layer Conductance
- Corrected spelling of 'Pohlhausen'. This should still be backward compatible, as it also accepts the spelling "Polhausen".

## Photosynthesis
- The way parameters are set for the `FvCB` model has been clarified to be more explicitly clear about how temperature response is being treated.

# [1.3.27] 2025-02-11

**Some experimental `OpenMP` parallelization has been added. This is not enabled by default yet, but can be turned on by setting the CMake option `ENABLE_OPENMP` to `ON`.**
**Updates to core `CMakeLists.txt` to use more modern CMake features.**

## Context
- `Context::pruneTubeNodes()` was added to allow for removal of part of a tube object.

## Plant Architecture
- Added ground cherry weed model.
- Minor fix in `PlantArchitecture::getPlsantLeafInclinationAngleDistribution()` to prevent rare out of range index.
- Changed `PlantArchitecture::getPlantLeafInclinationAngleDistribution()` to area-weight the distribution, and removed the 'normalize' optional argument.
- Added `PlantArchitecture::prunBranch()` method to remove all or part of a branch and its downstream branches.
- Removed shoot parameter `'elongation_rate'`. There is now an `'elongation_rate_max'` shoot parameter that can be set by the user. The actual elongation rate can be reduced dynamically if the carbohydrate model is enabled.
- Many updates to carbohydrate model. Credit to Ethan Frehner for these updates.

## LiDAR
- Added `exportTriangleAzimuthDistribution()` to write the triangulated azimuthal angle distribution to a file. Credit to Alejandra Ponce de Leon for this addition.
- The output distribution from `exportTriangleInclinationDistribution()` was not being normalized.

## Radiation
- Added bindweed spectra to the default library.
- There was an error in the `writeObjectDataLabelMap()` method that could cause undefined behavior if the UUID in the pixel label map did not exist in the `Context`.
- There was an error in the `writeObjectDataLabelMap()` method where the primitive UUID was being used instead of the object ID.
- There was an error in the model that could cause incorrect assignment of radiative properties if `runBand()` is called with fewer bands than a previous call to `runBand()`.
- The previous version could cause unnecessary updating of radiative properties, resulting in a performance hit. This has been fixed.
- Revised the radiation plug-in `CMakeLists.txt` to use a modern CMake approach for building CUDA source files. This update should also enable indexing of `.cu` source files in IDEs such as CLion.

# [1.3.26] 2025-01-17

## Context
- Definition of global `PI_F` variable in `global.h` was causing a build issue on some compilers

## Visualizer
- Added mouse-based camera controls in the interactive visualizer. Credit to Sean Banks for this update.

## Plant Architecture
- Added methods for querying bulk properties of a plant (total leaf count, stem height, plant height, leaf angle distribution) and write plant mesh vertices to file.

# [1.3.25] 2025-01-03

*Updated copyright dates to 2025*

## Plant Architecture
!!! Many significant changes in this release !!!
- The way leaf prototypes are generated has changed. There is now a modifiable parameter set that controls procedural leaf generation (or loading a model from an `OBJ` file).
- Changed generic leaf prototype function to be able to represent leaf "buckling", such as what happens to long leaves of grasses like maize.
- Removed shoot parameter `'leaf_flush_count'`.
- Added shoot parameter `'max_nodes_per_season'` to allow the shoot to stop growing for the season, but continue adding nodes in future seasons.
- Phenological thresholds for plants have been changed to be realistic (e.g., the phenological cycle for perennial plants corresponds to days in a year).
- A maximum age limit for plants has been added so users don't inadvertantly specify a very old plant that will cause the program to hang.
- Changed the way that vegetative bud break probability is handled such that it can vary along the shoot. The relevant parameters are now `'vegetative_bud_break_probability_min'` and `'vegetative_bud_break_probability_decay_rate'`.
- Changed the way that downstream leaf area is calculated to improve computational efficiency. This required some changes in the 'girth_area_factor` parameter.
- Committed all Blender project files used to create organ `OBJ` models: `plugins/plantarchitecture/assets/Blender_organ_models/`
- Moved initial internode radius from being a shoot parameter to a phytomer parameter, since it is constant along the shoot and over time.
- Added overloaded `PlantArchitecture::advanceTime()` method that allows to specify the time step in years and days for advancing over long time periods.
- Changed the name of the shoot parameter `'phyllochron'` to `'phyllochron_min'` to be consistent with how it will be used in the carbohydrate model.
- Added actual self-test that builds all plants in the library.

## Visualizer
- Problem fixed in freetype `zconf.h` file that caused build errors on latest MacOS compilers.
- All output messages were not being properly suppressed after calling `disableOutputMessages()`.

## Canopy Generator
- Many edits made regarding how parameters with random variation are handled, mainly to ensure proper constraints on values (e.g., always positive).

# [1.3.24] 2024-11-27

* Added `utilities/dependencies.sh` script to automatically install all dependent libraries on Linux and MacOS. Credit to Sean Banks for creating this script.

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

# [1.3.23] 2024-11-19

## Context
- Still some lingering issues with `Context::addTubeObject()` in which tube creation could fail.

## Visualizer
- Added warning in `Visualizer::buildContextGeometry()` if there is existing `Context` geometry already in the `Visualizer` that may have needed to be cleared.
- When calling `Visualizer::clearGeometry()`, the colorbar range was not being reset, which could cause unexpected behavior.
- Changed default lighting intensity to be brighter in order to better match the default behavior of 3rd party renderers such as Blender.
- Added method `Visualizer::setLightIntensityFactor()` to allow users to adjust the intensity of the lighting in the scene.

## Plant Architecture
- Some updates to bean, cowpea, and apple model parameters and assets.
- Changed `PlantArchitecture::buildPlantCanopyFromLibrary()` to accept an optional argument specifying the germination rate.
- Fixed error in which fruit could potentially become disconnected from their peducle over time.
- Fixed error that was causing flowers not to open.
- Added optional phenological parameter `'max_leaf_lifespan'` to allow leaves to die based on age, which is especially important for evergreen plants.
- Updates to carbohydrate model. Credit to Ethan Frehner

## Photosynthesis
- Error corrected that could cause an incorrect temperature response depending on how model parameters are set. Thanks to Kyle Rizzo for this fix.
- Added methods to disable output messages.

## Stomatal Conductance
- Added methods to disable output messages.

## Canopy Generator
- Error fixed when parsing `XML` files for strawberry canopies. Thanks to Heesup Yun for the fix.

## Radiation
- Error corrected in the camera model related to the field of view and aspect ratio. Thanks to Peng Wei for this fix.

# [1.3.22] 2024-11-01

## Context
- Methods `Context::sumPrimitiveSurfaceArea()`, `Context::calculatePrimitiveDataAreaWeightedMean()`, and `Context::calculatePrimitiveDataAreaWeightedSum()` to check if primitive area is NaN, and if so exclude it from calculations.
- Added `parse_int2()`, `parse_int3()`, and `parse_RGBcolor()` functions to `global.cpp`.
- Added `open_xml_file()` function to `global.cpp` that opens an `XML` file and checks for basic validity.
- Added `'Location'` type to `helios_vector_types.h` to store latitude, longitude, and UTC offset.
- Added `Context::listTimeseriesVariables()` to return a list of all existing timeseries variables.

## Plant Architecture
- Some updates to soybean model parameters.
- Added methods to query UUIDs and Object IDs for all plants in the model to avoid having to loop over each plant instance.
- Some additional checks were needed to make sure the tube internode object actually exists in the Context, otherwise there could be an out-of-bounds error.
- Removed the Shoot Parameter 'internode_radius_max', as it is not needed anymore after the pipe-model-based internode girth scaling was added.
- Corrected some issues with reading/writing plants using strings or XML. Namely, some parameters like the phyllotactic angle were not being applied correctly across shoots.
- Added `PlantArchitecture::getCurrentPytomerParameters()` to make it easy to get all the phytomer parameters structures to pass to `PlantArchitecture::generatePlantFromString()`.

## Radiation
- Split spectral_data/surface_spectral_library.xml into separate files for soil, leaves, bark, and fruit, and added many new species. Credit to Kyle Rizzo for these additions.
- Some default values were set in `RadiationModel.cpp` while others were set in `RadiationModel.h`. Everything was moved to be set in the RadiationModel constructor, which is in `RadiationModel.cpp`.

## Solar Position
- Default constructor changed to load the location based on the location set in the Context.
- UTC offset variable changed from int to float type.

## Visualizer
- Visualizer::printWindow() now creates the output directory if it does not already exist.

# [1.3.21] 2024-10-24

## Context
- There was an error introduced in the previous version for writing OBJ files. If no output directory was explicitly specified, the write would fail.
- Added `validateOutputPath()` function to `global.cpp` to validate output files and directory paths.
- Replaced `getFile*()` functions in `global.cpp` with more robust `std::filesystem` functions.
- Added improved file validation to `Context::writeXML()`.

## Radiation
- There was an error introduced in the previous version for writing image output files. If no output directory was explicitly specified, the write would fail. Removed the `validateOutputPath()` function implemented in the previous version, and replaced it with a better version implemented in `global.cpp`.

## Plant Architecture
- The 'tortuosity' shoot parameter was multiplied by a hard-coded factor of 5.0. This was removed, and all tortuosity values in the plant library were scaled by a factor of 5.
- There were some errors with the parameters of some plants introduced in the previous version.
- Added file validation to `PlantArchitecture::writePlantStructureXML()`.

# [1.3.20] 2024-10-21

## Context
- Texture mapping for sphere objects was not correct. The top cap was not being properly mapped, resulting in a dark spot.
- Added `Context::listGlobalData()` to list out all global data in the Context.
- There were still issues with tube twisting when using `Context::appendTubeSegment()` based on the implementation from v1.3.19. This should now be fixed.
- Modified helios vector types `vec2`, `vec3`, and `vec4` normalize methods to also return the normalized vector. This allows for chaining of operations.

## Energy Balance
- Added overloaded version of `EnergyBalanceModel::addRadiationBand()` to add multiple bands at the same time by passing a vector.

## Stomatal Conductance
- Error in `StomatalConductanceModel::setBMFCoefficientsFromLibrary(const std::string &species_name)` function fixed.

## Plant Architecture
- Added capability of having evergreen plants. This is controlled through `PlantArchitecture::setPlantPhenologicalThresholds()`;
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

## Solar Position
- Moved self-tests to separate selfTest.cpp file. Experimenting with new way of doing self-tests based on catch-try blocks and lambda functions.

## Plant Architecture
- Changed initial fruit scaling to be 25% of full size when it is created.

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

# [1.3.14] 2024-06-22

🚨+ NEW PLUG-IN + 🚨
- Plant Architecture plug-in merged from development branch to master repo. This plug-in is still in beta testing and is not yet fully documented and is likely to still have a number of bugs. Please report bugs as you find them.

## Context
- Cone object girth was not being properly scaled.

## Solar Position
- Added sub-model to calculate solar fluxes and diffuse fraction for cloudy conditions based on radiometer measurements (see SolarPosition::enableCloudCalibration()).
- Changed many variable names to make units more explicit.

## Radiation
- Added "ActiveGrow_LED_RedBloom" light to light spectral library.

# [1.3.13] 2024-06-10

- Fixed compiler warning in pugixml (convert_number_to_mantissa_exponent).
- Disabled CMake deprecation warnings.
- Upgraded zlib to v1.3.1 to get rid of compiler warnings
- Changed standard project CMakeList.txt to require cmake v3.15 on all systems.

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

# [1.3.8] 2024-03-06

* Documentation updates

## Context
- Added method to easily increment (sum) global data (see Context::incrementGlobalData()).

## Radiation
- Added check to RadiationModel::writePrimitiveDataLabelMap() to print a warning if the primitive data was empty for all pixels.
- There was an error with the radiation camera field of view in that the actual field of view in the image was half that of the specified HFOV value.

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

# [1.3.3] 2023-10-24

## Context
- Simpler overloaded versions of Context::loadPLY() and Context::loadOBJ() added that only require a filename argument.

## Stomatal Conductance
- Any steady-state stomatal conductance model can now be run in dynamic mode based on stomatal time constants for opening and closing.

## Solar Position
- Added method calibrateTurbidityFromTimeseries() to calibrate turbidity based on a timeseries of measured solar irradiance. For now, this method only uses the maximum solar flux in the dataset to calibrate the turbidity.
- Several methods were made const to have const-correctness.

# [1.3.2] 2023-10-20

## Context
- Try making Context::selfTest() static (again) so it can be run without declaring the Context class.
- Fixed and issue in helios::parse_float() that could cause and out of bounds error if the parsed value requires more precision than can be stored in a float.

## LiDAR
- Added error checking to LiDARcloud::export[*] methods to check whether the output file was successfully created.
- Added method LiDARcloud::exportTriangleInclinationDistribution() to easily calculate and export triangulation angle distribution.
- If users specify a scan thetaMax that is out of range and thetaMax is truncated to pi, the thetaMin value will also be set to 0.
- ScanMetadata::direction2rc() function will truncate any points outside of the specified scan range (thetaMin - thetaMax; phiMin - phiMax).

# [1.3.1] 2023-09-14

* Changed copyright header information for most plug-ins to remove authorship information.

## Synthetic Annotation
- Include paths in CMakeLists.txt changed to relative paths to avoid build errors on some systems.
- Error fixed in SyntheticAnnotation::setCameraPosition() where arguments were not being correctly assigned.
- Converted to using helios_runtime_error() for error handling.

## LiDAR
- Converted to using helios_runtime_error() for error handling.
- Removed 'Nhits' field from ScanMetadata struct, as this was not used.
- LiDARcloud::syntheticScan: Changed the method of grouping rays into hit points so that it is based on detecting peaks in the histogram of intensity weighted distance, closer to what a real LiDAR scanner would do. A pulse distance threshold is then used to merge hit points that are close together.
- LiDARcloud::syntheticScan: scanner range was set to 1000 m instead of 1e6. Miss points are now assigned a value of 1001 m.
- LiDARcloud::calculateLeafAreaGPU_synthetic: removed unused components, including weighting by sine of zenith angle in transmission probability calculation
- LiDARcloud::calculateLeafAreaGPU_synthetic: fixed issue with intensity weighting that caused total miss beams to not be accounted for in transmission estimates. This issue was was introduced in version # [1.2.60] when intensity of miss points was switched to zero in syntheticScan.
- Minor code cleaning.

## Radiation
- Calibration "calibrated_CREE6500K_NikonD700_spectral_response_*" in camera_spectral_library.xml edited to be consistent with normalized spectrum for CREE LED light source.
- Added example to documentation of writing image pixel labels.

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

# [1.2.70] 2023-09-05

*Changed core/CMake_project.txt to set HELIOS_DEBUG variable instead of _DEBUG, which could cause build conflicts on Windows.
*Set CMake policy CMP0079 to "NEW" in core/CMake_project.txt to allow for target_link_libraries() to be used with imported targets.
*Add target_link_libraries() between all plug-ins and helios target in core/CMake_project.txt to avoid build errors on some systems.

## Context
- Fixed error in Context::loadTabularTimeseriesData() that caused incorrect reading of 'minute' values.

## Aerial LiDAR
- Fixed and error with implementation of helios_runtime_error() that caused a compile error.

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

# [1.2.66] 2023-08-03

## Context
- Method added to get the random number generator from the Context, such that it can be used by other plug-ins. This keeps the seed consistent across plugins.

## Visualizer
- Issue fixed that caused glfw3.dll to not be properly copied after build on Windows platforms.

# [1.2.65] 2023-07-25

## Context
- Added Context::loadTabularTimeseriesData() method to directly load a tabular text file containing weather data.
- Self-test added for Context::loadTabularTimeseriesData().
- Output streams to print a 'Date' or 'Time' vector modified to print leading zero for month, day, minute, and second.
- Added Date::incrementDay() method to date vector to increment date by one day.
- Added helios::separate_string_by_delimiter() function to separate a delimited string into a vector of strings.

## Radiation
- Upgrading to CUDA 12.0+ caused build errors. Users with GPU compute capability of 3.5 now need to enable the OPTIX_VERSION_LEGACY build option.

# [1.2.64] 2023-07-20

## Context
- Increment operator added for Helios vector types int2, int3, int4, vec2, vec3, vec4, and general improvements to helios_vector_types.h
- Overloaded versions of Context::overridePrimitiveTextureColor() and Context::usePrimitiveTextureColor() added to accept a vector of primitive UUIDs.
- Overloaded versions of Context::overrideObjectTextureColor() and Context::useObjectTextureColor() added to accept a vector of object IDs.

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

# [1.2.59] 2023-05-12

## Context
- Some efficiency improvements in Helios vector types.
- writeOBJ() updated to write optional .dat files to same directory that the .obj and .mtl files will be written.
- scaleObject() function was added to allow for scaling of compound objects after they have been created. As a result, scaling of objects is now done the same regardless of the object type.
- 'nullorigin' global variable created to avoid having to type out make_vec3(0,0,0).

## Solar Position
- Error corrected in calculation of diffuse fraction using Gueymard model (credit to Alejandra for finding this)
- Added function setSunDirection() to manually set the sun direction.

## Photosynthesis
- Error in documentation corrected that had column labels for photosynthesis model parameters swapped for theta and Rd.
- Optional output data for limitation state was outputting the wrong value.

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

# [1.2.57] 2023-02-23

## Context
- Some temporary fixes applied related to reading/writing of Tile Objects to XML. There are still issues when the tiles have been rotated about their normal axis that need to be resolved.

## Synthetic Annotation
- Error fixed that could cause a character buffer overflow for very long file paths. Strings are now used instead of character arrays, which allows for arbitrarily long paths.

## Energy Balance
- Documentation updated to clarify that primitive outgoing emission depends on whether the primitive is 'two-sided'

*Boundary-Layer Conductance*
- Minor errors corrected in documentation

# [1.2.56] 2023-01-18

## Context
- There were some minor memory leaks associated with the PNG texture/image reading/writing. These were generally very small leaks.

## Visualizer
- There were some minor memory leaks associated with the PNG texture/image reading/writing. These were generally very small leaks.

## Synthetic Annotation
- Visualizer geometry was being updated for every view when generating labels, which resulted in slow performance and building of large amounts of memory.

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

# [1.2.54] 2022-12-23

## General
- Added 'cmake_minimum_required' at the beginning of all project CMakeLists.txt files for all platforms, as well as 'project(helios)'.
- Suppressed many of the compiler warnings that occur on Windows.

## Context
- Fixed an error in Context::loadPLY() introduced in version 1.2.52 that caused PLY models to not be read correctly.

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

# [1.2.52] 2022-12-12

## General
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

## LiDAR
- Fixed bug in distanceFilter, reflectanceFilter, and scalarFilter, which caused them not to apply filtering.
- Added xyzFilter() function to filter out points outside of a specified bounding.

# [1.2.48] 2022-10-28

## LiDAR
- Building with liblas and laszip libraries disabled for now, as this seemed to cause build errors on some systems.

# [1.2.47] 2022-10-21

## Context
- Error fixed in loadXML() where patches that are members of a tile objects were added twice to the output UUID vector in some cases.
- Error fixed in loadXML() where the elevation angle of loaded tile objects was not correct.
- Error fixed in writeXML() where primitives that were members of "complete" tile objects were getting written to the XML file, which could cause the member patches to be created twice in loadXML().

## LiDAR, Aerial LiDAR, Energy Balance, Voxel Intersection, Radiation *
- Added explicit setting of CUDA compiler optimization flag for Debug/Release build types.

## Radiation
- Added check to addBand() function to prevent users from adding duplicate bands.
- For each ray launch 'batch', the previous batch output message will be cleared so it will not continue printing a new line for each batch launch.

## Energy Balance
- If no object length is specified via primitive data, the model will first check to see if the primitive is a member of an object, in which case it will use the object area to calculate the characteristic length for boundary-layer conductance calculations.

## Boundary-Layer Conductance
- If no object length is specified via primitive data, the model will first check to see if the primitive is a member of an object, in which case it will use the object area to calculate the characteristic length for boundary-layer conductance calculations.

## LiDAR
- Revised LiDAR self-test to use geometries read from file to achieve more consistent results.
- laszip and libLAS libraries are included in the repository, but are not yet used.

* Voxel Intersection *
- Minor changes to avoid using primitive pointers directly.
- Error fixed where sliced triangles that were previously members of a tile would retain the object ID of the previous tile (created triangles should not be a member of an object).

# [1.2.46] 2022-10-07

## Context
- All Compound Objects besides Tiles were not being properly read from XML files.
- There was an error in the Context::writeJPEG() file, which caused garbage pixel values to be written.
- Added explicit instantiation of template functions in global.cpp. Not having these could cause build errors with high compiler optimization enabled.

## Canopy Generator
- Updates to bean model to speed up generation time.

# [1.2.45] 2022-09-26

## Context
- If any primitives have (u,v) coordinates, writeOBJ() will now write 'dummy' (u,v)'s for primitives that are not UV mapped. This is apparently needed to correctly read the .obj file into Blender.
- writeOBJ() function was not working properly based on updates in v.1.2.44.
- Deblank function added for strings (previously was only for character arrays).

## Canopy Generator
- Draft model for common bean added. It is not documented yet, and bean pods are not yet supported.

# [1.2.44] 2022-09-16 ** revised 2022-09-23

**This commit originally pushed 2022-09-16 introduced an error that would cause build of the Context to fail on many systems. This commit was reverted and re-committed with a correction on 2022-09-23 *

## Context
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

## Radiation
- CMakeLists.txt file improved so that the *.ptx files will only be re-built if the source *.cu file is changed.

# [1.2.43] - 2022-09-07

## Context
- Overloaded loadOBJ() function added to scale the model in the x,y,z directions.
- Messages can now be disabled when loading OBJ files.
- Primitive solid area fraction is written and read from XML so that it doesn't need to be re-calculated when loading.
- Change made to Context::loadPLY() such that if a 'height' value of 0 is given, the model will not be scaled.
- An error was introduced in v1.2.41 associated with texture indexing/truncating.
- The member variable 'solid_fraction' was erroneously declared for Triangles, which was redundant and conflicting with the definition for Primitive.
- Added additional self-test case to test solid fraction for triangles.

## Energy Balance
- Check added to only add a radiation band if it has not already been added previously to avoid duplicates.

## Canopy Generator
- Functions to add individual plants now return the plant ID.
- Functions added to clean up deleted UUIDs.
- Issues were fixed with get[*]UUIDs() functions that would return UUIDs for deleted primitives in some cases.

# [1.2.42] - 2022-09-02

## Context
- Minor updates to helios_vector_types.h
- Bug fixed that would cause segmentation fault if texture (u,v) coordinates are out of bounds. These will now be truncated to [0,1]
- Change made to Context::loadOBJ() such that if a 'height' value of 0 is given, the model will not be scaled
- Overloaded versions of Context::loadOBJ() added to only write geometry for a subset of UUIDs, and to write a .dat file containing primitive data for Unity visualization tool
- Feature added to Context::writePrimitiveData() to allow for writing primitive UUID without explicitly creating primitive data for UUID

## LiDAR
- Bug fixed that caused triangle textures to be flipped about the y-axis in synthetic scans

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

# [1.2.40] - 2022-07-25

## Context
- Error fixed that caused incorrect transparent texture mapping in the y-direction, which could result in incorrect areas when there are sub-patches/triangles or custom (u,v) coordinates.
- Changes related to deleting objects when all constituent primitives have been deleted in deletePrimitive, deleteObject, setTileObjectSubdivisionsCount, loadXML, and setPrimitiveParentObjectID.
- Added related self tests.

## Canopy Generator
- Fixed formatting error for documentation.
- Edited default parameter values to match documentation.

# [1.2.39] - 2022-07-19

🚨+ NEW PLUG-IN + 🚨
- First version of synthetic data annotation plug-in is now included.

## Canopy Generator
- Tomato canopy was missing from loadXML() function.

# [1.2.38] - 2022-06-20

## Canopy Generator
- Improvement to sorghum panicle shape
- Bug fixed in sorghum model that caused jagged leaf edge when subpatch resolution was changed

# [1.2.37] - 2022-05-27

## Context
- Building was failing on M1 Macs. A fix was implemented into the libpng CMakeLists.txt file.

## Photosynthesis
- Users can now optionally output 'Gamma' (CO2 compensation point) as primitive data.

## Canopy Generator
- First version of a sorghum model is now included. (credit to Ismael Mayanja)

# [1.2.36] - 2022-05-13

## Context
- Added numerous checks for texture image files to ensure they are either PNG or JPEG files.
- `loadPLY()` and `loadOBJ()` functions check to be sure the input files have the correct extensions.
- When loading compound objects using loadXML(), it will first look to see if there are any primitives in the file assigned to that object, and if so it will build the object based on those primitives. If no primitives are assigned to the object, it will build a fresh/complete object from scratch.
- Corrected an issue with cropDomain() functions to be consistent with new treatment of incomplete compound objects (i.e., primitives can be deleted from within objects).

## Visualizer
- Functions for reading PNG and JPEG files now check to ensure they have the correct file extension.

## Radiation
- Users can now set primitive data "twosided_flag" equal to 2 to make the primitive a one-sided "sensor" that does not attenuate or emit any radiation.

# [1.2.35] - 2022-04-29

## Context
- Added functions to calculate the axis-aligned bounding box for a primitive or group of primitives (see getPrimitiveBoundingBox()).
- Added functions to calculate the axis-aligned bounding box for a compound object or group of compound objects (see getObjectBoundingBox()).

# [1.2.34] - 2022-04-26

## Context
- Added warning messages to setTileObjectCount() and getTileObjectAreaRatio() related to incomplete objects, minor related documentation changes.
- Changed Tile::getVerticies so that a missing corner primitive doesn't break it.
- Added many functions to set/get object information without using pointers.

## Visualizer
- added Visualizer::colorContextPrimitivesRandomly to easily visualize object sub-primitives.

# [1.2.33] - 2022-04-11

## Context
- There was an error in OBJ texture writing that caused texture coordinates to be flipped about y-axis.
- There was an error in OBJ texture writing that gave invalid texture coordinates for patches.
- If a primitive that belongs to a compound object is deleted (via Context::deletePrimitive()), it will delete the primitive and remove its UUID from the list stored with the object. A function was added Context::areObjectPrimitivesComplete() that allows for querying of whether some primitives have been deleted from the object.

## Radiation
- Tile objects are now checked to see if any child primitives have been deleted. If so, it will treat it as individual primitives and ignore the fact that it is an object.

## Visualizer
- If a UUID for a primitive that does not exist was passed to the Visualizer, it would previously cause a segmentation fault.

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

# [1.2.31] - 2022-03-08

## LiDAR
- Error corrected causing build failure on Windows systems.

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

# [1.2.29] - 2022-01-30

## Context
- The normals were reversed for triangles at the top and bottom of spheres (sphere objects were already correct).
- Error was introduced in v1.2.27, which caused scaling of tile objects not to work.

## Canopy Generator
- Walnut, Tomato, and Strawberry models: if fruit_radius=0, don't add any fruit.
- Tomato textures were inadvertently removed from Git. They have been re-added.
- Definition of fruit_radius for walnut model was missing from documentation.

# [1.2.28] - 2022-01-13

## Visualizer
- Error corrected that could result in GLFW_INVALID_OPERATION error on Windows when the visualizer window is closed.
- The default filename when printWindow() is called with no arguments is not valid on Windows machines.

## Energy Balance
- Documentation updated to better clarify the meaning of the "moisture conductance".

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

# [1.2.23] - 2021-11-28

*Documentation added for set-up on Windows systems.

## Context
- Include file in core/include/pugixml.h changed from <cstring> to <string>, which could cause build issues on some systems.
- Issues with inconsistent treatment of (u,v) texture coordinates corrected.
- Added "nullrotation" global variable to avoid having to type out make_SphericalCoord(0,0).

# [1.2.22] - 2021-11-19

* Helper functions added to global.cpp to calculate median and standard deviation of float values in a vector.

* Standard project CMakeLists.txt updated to v1.6 to correct potential build issues on certain systems.

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

# [1.2.19] - 2021-10-14

* All projects in the samples/ directory have been moved to Helios CMakeLists.txt version 1.4, which includes Windows support and avoids the *_exe build target issue in CLion.
* Working on Windows/PC support: Several changes have been made to support PC, which will be tested and eventually released in v1.3 when it appears stable.

## Context
- Many changes in the core/ files to clean up the code based on clang-tidy suggestions
- Removed strdup function in favor of strcpy function in global.cpp, because strdup could cause problems on some architectures
- Added define of M_PI in global.h in case it is not defined on some architectures

## LiDAR
- Checks were added to addScan() function to avoid specifying invalid scan angle ranges.

# [1.2.18] - 2021-09-19

- The project creation script (utilities/create_project.sh), and default project CMakeLists.txt file (version 1.4), has been modified to make the executable name the same as the build target. This makes using Helios with IDEs easier.

## Context
- The functions getGlobalDataType() and getGlobalDataSize() were declared but never implemented.
- The functions string2int2, string2vec2, etc. have been updated to be more robust by using the stoi() and stof() functions to convert parsed strings into values.

## Canopy Generator
- Ability to generate the other canopy types from XML file has been implemented. Credit to Dario Guevara for finishing these.

## LiDAR
- Change made to synthetic waveform data generation to improve delineation of adjacent hit points.

# [1.2.17] - 2021-09-15

## Canopy Generator
- Visual improvements to VSP grapevine canopy.

## Visualizer
- If buildContextGeometry() is called and there is no geometry in the Context, it could mess with the camera view. This behavior was changed to simply skip the function call if there is no geometry to build.
- A version of plotUpdate() has been added to not open the graphics window.

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

# [1.2.14] - 2021-07-21

## Radiation
- Correction in the normalization of anisotropic diffuse distribution. For a horizontal, unobstructed patch, it should always give a normalized diffuse flux of 1 regardless of the value of K.

## Canopy Generator
- The 'spherical crowns' canopy has been generalized to include ellipsoidal crowns with different principal radii.

## Visualizer
- There is apparently an issue with the printWindow() function on Linux that causes a segmentation fault. A work-around has been implemented to remove the window "decorations", which seems to fix the problem. A new overloaded constructor has been added to allow for disabling of the window decorations. When a fix for the issue is eventually found, this constructor will likely be removed.

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

# [1.2.12] - 2021-06-30

## Canopy Generator
- The file walnut.cpp was not added to the repository, so it has been missing for the past two versions.

## Radiation
- There was an error in the selfTest() function, causing the compile to fail.

# [1.2.11] - 2021-06-30

## Context
- Documentation added for Compound Objects

## Radiation
- Radiation model adapted to more efficiently handle sub-divided patches via Tile Objects. When tile objects are used to represent sub-divided patches, there is now a dramatic reduction in the amount of GPU memory used, and ray-tracing computations are more efficient.

# [1.2.10] - 2021-06-21

## Visualizer
- An error was corrected that caused shadows to not be rendered when using plotUpdate().

## Canopy Generator
- Walnut tree model added with documentation.

## Stomatal Conductance
- There was not actually an error with the units of Em - they were changed back as before.

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

# [1.2.8] - 2021-04-26

## Context
- Added texture mapping for spheres and sphere compound objects.
- Added "Cone" compound object and self-test.

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

# [1.2.4] - 2020-12-10

## Context
- Error handling has been improved. Errors in the Context no longer call "exit(EXIT_FAILURE)", but rather call "throw(1)". This allows for catching of errors in a debugger to provide a stack trace and line numbers for the errors. Future releases will throw more specific and informative error codes, but for now "1" is always thrown.

## Visualizer
- An error was corrected related to window sizing on a Mac, which could result in a segmentation faullt.

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

# [1.2.2] - 2020-11-25

## Canopy Generator
- An initial tomato model was added to the canopy generator in v1.2.1, but the file src/tomato.cpp was not added to the git repository, which will cause an error while building the canopy generator.

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

# [1.1.4] - 2020-06-18

## Aerial LiDAR
- Functions added to get the beam mean free path and numerator and denominator of the probability of interception calculated in calculateLeafAreaGPU(). See functions getCellTransmissionProbability() and getCellRbar().

# [1.1.3] - 2020-06-16

## Context
- Further problems with patch texturing corrected. A lingering error was causing patches with custom (u,v) coordinates to have zero surface area. This was likely introduced in v1.1.0 in correcting a similar error with triangles.

## LiDAR
- For full-waveform scans, leaf area calculations now use equal weighting of hit points.
- Function added to gap fill missing scan points due to "sky" hit points (see function gapfillMisses()). This function is only incorporated within the "testing" leaf area calculation function calculateLeafAreaGPU_testing().

## Aerial LiDAR
- Aerial LiDAR sample was not compiling because of an error in the CMakeLists.txt file.

# [1.1.2] - 2020-04-14

## Radiation
- Periodic boundary condition added. See function RadiationModel::enforcePeriodicBoundary().

## Aerial LiDAR
- Documentation page added, but still under construction.

# [1.1.1] - 2020-04-10

## Visualizer
- Correction was not picked up in previous version that caused Helios watermark to be flipped.

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

# [1.0.17] - 2019-11-08

## Context
- "Swizzle" capability for vec3 removed.
- Error corrected in getDomainBoundingBox() that could cause an error if primitives are deleted from the Context.

## Weber Penn Tree
- Re-implemented leaf sub-patch tiling to be much faster.

## LiDAR
- Re-implemented leaf sub-patch tiling to be much faster.

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

# [1.0.13] - 2019-08-09

## Context
- Error corrected causing incorrect rotation of tile sub-patches.
- Additional selfTest() cases added for rotated and texture-masked tiles.

# [1.0.12] - 2019-08-08

## Context
- Adding texture-masked tiles with a transparency channel was previously excessively slow. A new implementation was added such that there is no slow-down when texturing a tile.

## Radiation
- Error fixed that caused errors in absorbed radiation when the scence contains two different texture masks of different size.

## LiDAR
- Stability improvements.

# [1.0.11] - 2019-07-24

## Context
- Error fixed with texture mapping of tiles.

## Radiation
- Error fixed that could cause an infinite loop in certain cases where the primitive has a texture mask.

# [1.0.10] - 2019-05-30

## Context
- Error fixed when calling addTile() with a non-zero spherical rotation.

## LiDAR
- Certain compiler versions would cause a compiler error with the 'fmin' function, which has been corrected.

# [1.0.9] - 2019-05-24

## Radiation
- Having a radiation band name longer than about 8 characters would cause a segmentation fault. You can now have names of up to 80 characters.

## Energy Balance
- There is now a function optionalOutputPrimitiveData() that allows you to output the calculated boundary-layer conductance and vapor pressure deficit to primitive data.

## LiDAR
- Added new scheme for inversion of LAD from aerial LiDAR data. When vegetation in a voxel is very dense, a method is used that fits to the hit point distance PDF.
- Added aerial LiDAR self-test and sample to run the self-test.

# [1.0.8] - 2019-05-09

## LiDAR
- Aerial LiDAR reverted to before v1.0.6, except that acos_safe() function is still implemented. There are still apparently problems with the changes implemented in v1.0.6.

# [1.0.7] - 2019-04-03

## Context
- Implemented acos_safe() and asin_safe() functions to catch instances where round-off errors cause normal acos() and asin() to return NaN. The new functions were implemented with the LiDAR and solarposition plug-ins where acos() or asin() was used previously.
- Error was corrected from v1.0.6 in which voxels sizes were not properly set when using addVoxel().

# [1.0.6] - 2019-04-02

## Context
- Added file core/lib/libpng/pnglibconf.h. Not having this file causes problems on some systems.
- Error fixed with Patch::getNormal() function.
- selfTest() tests added for rotate patches, and boxes.

## LiDAR
- Aerial LiDAR scan can now have a coneangle of zero without causing problems
- Aerial LiDAR updated to use advanced method for inverting Beer's law

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

# [1.0.4] - 2018-12-17

## Context
- Further errors corrected in timeseries data query due to precision issues near beginning/end of timeseries
- Errors corrected in calculating the area of primitives. Areas were not properly calculated for patches with custom (u,v) coordinates, and for primitives where a scaling transformation is applied, or when primitives were read from an xml file.

# [1.0.3] - 2018-12-13

## Context
- Errors in timeseries data querying were corrected in which some date/time combinations did not work correctly. Also, if a time/date is queried that is outside the range of all the data in the timeseries, the error will be caught rather than producing a segmentation fault.

## Visualizer
- colorPrimitivesByData() now supports primitive data types of int, uint, and double. Note: it will convert these data types to float.

## LiDAR
- Functionality added to be able to visualize triangulated fill groups used to create the leaf reconstruction: see function addReconstructedTriangleGroupsToContext().
- Calling addLeafReconstructionToContext() now creates primitive data for each leaf called ``directFlag" which equals 1 if the leaf came from the direct reconstruction, and equal to 0 if the leaf was backfilled.
- An initial implementation of functions to deal with aerial LiDAR data has been implemented, but has not been fully tested.

# [1.0.2] - 2018-11-26

## Visualizer
- Error corrected with adding texture-mapped triangles in buildContextGeometry() function.

# [1.0.1] - 2018-11-26

## Radiation Model
- Running radiation model for only a subset of UUIDs was not working properly (not implemented correctly).
- Texture-masked patches were causing an OptiX error under certain cases. Cause was an indexing typo in rayGeneration.cu

## Visualizer
- Visualizing texture-masked patches by primitive data was not working properly
- File needed in selfTest() function was missing (Helios_logo.jpg)

# [1.0.0] - 2018-11-15

**** Initial commit ****
