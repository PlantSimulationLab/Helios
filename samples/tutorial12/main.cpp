#include "RadiationModel.h"
#include "Visualizer.h"

using namespace helios;

int main() {

    Context context;

    // STEP 1: Load in model geometry for our synthetic scene

    // Load the first Stanford Bunny model and position it to the right
    std::vector<uint> UUIDs_bunny_1 = context.loadPLY("../../../PLY/StanfordBunny.ply");
    context.translatePrimitive(UUIDs_bunny_1, make_vec3(0.15, 0, 0));

    // Load the second Stanford Bunny model and position it to the left
    std::vector<uint> UUIDs_bunny_2 = context.loadPLY("../../../PLY/StanfordBunny.ply");
    context.translatePrimitive(UUIDs_bunny_2, make_vec3(-0.15, 0, 0));

    // Load and scale the Stanford Dragon model, positioned at center
    std::vector<uint> UUIDs_dragon = context.loadPLY("../../../PLY/StanfordDragon.ply", make_vec3(0.025, 0, 0), 0.2, nullrotation);

    // Create a ground plane to provide realistic lighting and context
    std::vector<uint> UUIDs_tile = context.addTile(nullorigin, make_vec2(1, 1), nullrotation, make_int2(1000, 1000));

    // Make the ground plane single-sided (only visible from above)
    context.setPrimitiveData(UUIDs_tile, "twosided_flag", 0u);

    // STEP 2: Set material properties using spectrally accurate color references

    // Load the Calibrite ColorChecker Classic color board data
    context.loadXML("plugins/radiation/spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml", true);

    // Assign realistic colors to each object using Calibrite color references
    context.setPrimitiveData(UUIDs_bunny_1, "reflectivity_spectrum", "ColorReference_Calibrite_15"); // red
    context.setPrimitiveData(UUIDs_bunny_2, "reflectivity_spectrum", "ColorReference_Calibrite_04"); // green
    context.setPrimitiveData(UUIDs_dragon, "reflectivity_spectrum", "ColorReference_Calibrite_10"); // purple
    context.setPrimitiveData(UUIDs_tile, "reflectivity_spectrum", "ColorReference_Calibrite_13"); // dark blue

    // STEP 3: Set up object labels for segmentation and object detection
    // These labels will be used to generate COCO-format annotations

    // Label the first bunny with class "bunny" and instance ID 0
    context.setPrimitiveData(UUIDs_bunny_1, "bunny", 0u);

    // Label the second bunny with class "bunny" and instance ID 8
    // Note: The ID value doesn't matter, only that all primitives of the same object instance have the same ID
    context.setPrimitiveData(UUIDs_bunny_2, "bunny", 8u);

    // Label the dragon with a different class "dragon" and instance ID 6
    context.setPrimitiveData(UUIDs_dragon, "dragon", 6u);

    // Set specular properties to make objects appear slightly shiny
    context.setPrimitiveData(UUIDs_bunny_1, "specular_exponent", 10.f);
    context.setPrimitiveData(UUIDs_bunny_2, "specular_exponent", 10.f);
    context.setPrimitiveData(UUIDs_dragon, "specular_exponent", 10.f);

    // STEP 4: Set up the radiation model for RGB imaging
    RadiationModel radiation(&context);

    // Optional: Add a color calibration target for accurate color reproduction
    CameraCalibration calibration(&context);
    calibration.addCalibriteColorboard(make_vec3(0, 0.23, 0.001), 0.025);

    // Configure sun lighting - position at 45 degrees elevation and azimuth
    SphericalCoord sun_dir = make_SphericalCoord(deg2rad(45), -deg2rad(45));
    uint sunID = radiation.addSunSphereRadiationSource(sun_dir);
    radiation.setSourceSpectrum(sunID, "solar_spectrum_ASTMG173");

    // Set up RGB radiation bands for color imaging
    radiation.addRadiationBand("red");
    radiation.disableEmission("red"); // Disable thermal emission for visible bands
    radiation.setDiffuseRadiationExtinctionCoeff("red", 0.2f, sun_dir);
    radiation.setScatteringDepth("red", 3); // Allow 3 bounces for realistic lighting

    // Copy red band settings to green and blue bands
    radiation.copyRadiationBand("red", "green");
    radiation.copyRadiationBand("red", "blue");

    std::vector<std::string> bandlabels = {"red", "green", "blue"};

    // Set diffuse sky spectrum for all bands
    radiation.setDiffuseSpectrum("solar_spectrum_ASTMG173");
    radiation.setDiffuseSpectrumIntegral(100.f);

    // STEP 5: Configure the radiation camera for synthetic image generation
    std::string cameralabel = "bunnycam";

    // Position camera to capture all objects in the scene
    vec3 camera_position = make_vec3(-0.01, 0.05, 0.6f); // Positioned back and slightly elevated
    vec3 camera_lookat = make_vec3(0, 0.05, 0); // Looking at the center of the scene

    // Configure camera properties for high-quality imaging
    CameraProperties cameraproperties;
    cameraproperties.camera_resolution = make_int2(1024, 1024); // Square 1024x1024 resolution
    cameraproperties.focal_plane_distance = 0.5; // Focus distance
    cameraproperties.lens_diameter = 0.002f; // Small aperture for sharp focus
    cameraproperties.HFOV = 50.f; // 50-degree horizontal field of view

    // Add the camera to the radiation model with 100 rays per pixel for quality
    radiation.addRadiationCamera(cameralabel, bandlabels, camera_position, camera_lookat, cameraproperties, 100);

    // Load camera spectral response data and simulate iPhone 12 Pro Max camera
    context.loadXML("plugins/radiation/spectral_data/camera_spectral_library.xml", true);
    radiation.setCameraSpectralResponse(cameralabel, "red", "iPhone12ProMAX_red");
    radiation.setCameraSpectralResponse(cameralabel, "green", "iPhone12ProMAX_green");
    radiation.setCameraSpectralResponse(cameralabel, "blue", "iPhone12ProMAX_blue");

    // STEP 6: Run the radiation simulation and generate outputs

    // Update the radiation model geometry (required before running simulation)
    radiation.updateGeometry();

    // Run the radiation simulation for all RGB bands
    radiation.runBand(bandlabels);

    // Apply standard image processing pipeline (tone mapping, gamma correction, etc.)
    radiation.applyCameraImageCorrections("bunnycam", "red", "green", "blue", 1, 1, 1);

    // STEP 7: Export results for computer vision applications

    // Save the rendered RGB image
    std::string image_file = radiation.writeCameraImage(cameralabel, bandlabels, "RGB", "../output/");

    // Export object detection data in COCO JSON format
    // This creates bounding boxes around each labeled object
    radiation.writeImageBoundingBoxes(cameralabel, {"bunny", "dragon"}, {0, 1}, image_file, "classes.txt", "../output/");

    // Export segmentation masks in COCO JSON format
    // This creates precise pixel-level masks for each object
    radiation.writeImageSegmentationMasks(cameralabel, {"bunny", "dragon"}, {0, 1}, "../output/bunnycam_segmentation.json", image_file);

    // STEP 8: Display the results

    // Initialize visualizer and display the rendered image
    Visualizer vis(1000);
    vis.displayImage("../output/bunnycam_RGB.jpeg");

    // Auto-calibrate camera image using colorboard reference values with quality report
    std::string corrected_image = radiation.autoCalibrateCameraImage("bunnycam", "red", "green", "blue", "../output/auto_calibrated_bunnycam.jpeg", true);

    // Note: Use the Python script 'visualize_segmentation.py' to view segmentation masks

    return 0;
}
