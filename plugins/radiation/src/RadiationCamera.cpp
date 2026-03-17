/**
 * \file "RadiationCamera.cpp" Definitions for methods related to the radiation camera.

Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "RadiationModel.h"
#include "LensFlare.h"

#include <queue>
#include <set>
#include <stack>
#include <sstream>
#include <filesystem>

#include "global.h"

using namespace helios;

void RadiationModel::addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::vec3 &lookat, const CameraProperties &camera_properties, uint antialiasing_samples) {

    if (antialiasing_samples == 0) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): The model requires at least 1 antialiasing sample to run.");
    } else if (camera_properties.camera_resolution.x <= 0 || camera_properties.camera_resolution.y <= 0) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera resolution must be at least 1x1.");
    } else if (camera_properties.HFOV < 0 || camera_properties.HFOV > 180.f) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera horizontal field of view must be between 0 and 180 degrees.");
    }

    // Auto-calculate FOV_aspect_ratio from camera resolution to ensure square pixels
    CameraProperties modified_properties = camera_properties;
    if (camera_properties.FOV_aspect_ratio != 0.f) {
        std::cerr << "WARNING (RadiationModel::addRadiationCamera): FOV_aspect_ratio is deprecated and will be ignored. The value is auto-calculated from camera_resolution to ensure square pixels." << std::endl;
    }
    modified_properties.FOV_aspect_ratio = float(camera_properties.camera_resolution.x) / float(camera_properties.camera_resolution.y);

    RadiationCamera camera(camera_label, band_label, position, lookat, modified_properties, antialiasing_samples);
    if (cameras.find(camera_label) == cameras.end()) {
        cameras.emplace(camera_label, camera);
    } else {
        if (message_flag) {
            std::cout << "Camera with label " << camera_label << "already exists. Existing properties will be replaced by new inputs." << std::endl;
        }
        cameras.erase(camera_label);
        cameras.emplace(camera_label, camera);
    }

    if (iscameravisualizationenabled) {
        buildCameraModelGeometry(camera_label);
    }

    // Auto-populate camera metadata (does not enable JSON writing)
    CameraMetadata metadata;
    populateCameraMetadata(camera_label, metadata);
    camera_metadata[camera_label] = metadata;

    radiativepropertiesneedupdate = true;
}

void RadiationModel::addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::SphericalCoord &viewing_direction, const CameraProperties &camera_properties,
                                        uint antialiasing_samples) {

    vec3 lookat = position + sphere2cart(viewing_direction);
    addRadiationCamera(camera_label, band_label, position, lookat, camera_properties, antialiasing_samples);
}

void RadiationModel::setCameraSpectralResponse(const std::string &camera_label, const std::string &band_label, const std::string &global_data) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (setCameraSpectralResponse): Camera '" + camera_label + "' does not exist.");
    } else if (!doesBandExist(band_label)) {
        helios_runtime_error("ERROR (setCameraSpectralResponse): Band '" + band_label + "' does not exist.");
    }

    cameras.at(camera_label).band_spectral_response[band_label] = global_data;

    radiativepropertiesneedupdate = true;
}

void RadiationModel::setCameraSpectralResponseFromLibrary(const std::string &camera_label, const std::string &camera_library_name) {

    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (setCameraSpectralResponseFromLibrary): Camera '" + camera_label + "' does not exist.");
    }

    const auto &band_labels = cameras.at(camera_label).band_labels;

    if (!context->doesGlobalDataExist("spectral_library_loaded")) {
        context->loadXML(helios::resolvePluginAsset("radiation", "spectral_data/camera_spectral_library.xml").string().c_str());
    }

    for (const auto &band: band_labels) {
        std::string response_spectrum = camera_library_name + "_" + band;
        if (!context->doesGlobalDataExist(response_spectrum.c_str()) || context->getGlobalDataType(response_spectrum.c_str()) != HELIOS_TYPE_VEC2) {
            helios_runtime_error("ERROR (setCameraSpectralResponseFromLibrary): Band '" + band + "' referenced in spectral library camera " + camera_library_name + " does not exist for camera '" + camera_label + "'.");
        }

        cameras.at(camera_label).band_spectral_response[band] = response_spectrum;
    }

    radiativepropertiesneedupdate = true;
}

void RadiationModel::addRadiationCameraFromLibrary(const std::string &camera_label, const std::string &library_camera_label, const helios::vec3 &position, const helios::vec3 &lookat, uint antialiasing_samples) {
    // Call the overloaded version with empty band_labels to use XML labels
    addRadiationCameraFromLibrary(camera_label, library_camera_label, position, lookat, antialiasing_samples, std::vector<std::string>());
}

void RadiationModel::addRadiationCameraFromLibrary(const std::string &camera_label, const std::string &library_camera_label, const helios::vec3 &position, const helios::vec3 &lookat, uint antialiasing_samples,
                                                   const std::vector<std::string> &custom_band_labels) {

    // Resolve library file path
    std::filesystem::path library_path = helios::resolvePluginAsset("radiation", "camera_library/camera_library.xml");

    // Load and parse XML file using pugixml
    pugi::xml_document xmldoc;
    pugi::xml_parse_result result = xmldoc.load_file(library_path.string().c_str());

    if (!result) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Failed to load camera library file '" + library_path.string() + "'. " + result.description());
    }

    pugi::xml_node helios_node = xmldoc.child("helios");
    if (helios_node.empty()) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Camera library XML must have '<helios>' root tag.");
    }

    // Find the camera node with matching label
    pugi::xml_node camera_node;
    for (pugi::xml_node cam = helios_node.child("camera"); cam; cam = cam.next_sibling("camera")) {
        std::string label = cam.attribute("label").value();
        if (label == library_camera_label) {
            camera_node = cam;
            break;
        }
    }

    if (camera_node.empty()) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Camera '" + library_camera_label + "' not found in camera library.");
    }

    // Parse camera parameters
    std::string manufacturer = camera_node.child("manufacturer").child_value();
    std::string model = camera_node.child("model").child_value();

    // Parse camera type (required field)
    std::string camera_type = camera_node.child("type").child_value();
    if (camera_type.empty()) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Missing required 'type' field for camera '" + library_camera_label + "'.");
    }
    if (camera_type != "rgb" && camera_type != "spectral" && camera_type != "thermal") {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Invalid camera type '" + camera_type + "' for camera '" + library_camera_label + "'. Must be one of: 'rgb', 'spectral', or 'thermal'.");
    }

    float sensor_width_mm;
    if (!helios::parse_float(camera_node.child("sensor_width_mm").child_value(), sensor_width_mm)) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Invalid or missing sensor_width_mm for camera '" + library_camera_label + "'.");
    }

    int resolution_width;
    if (!helios::parse_int(camera_node.child("resolution_width").child_value(), resolution_width)) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Invalid or missing resolution_width for camera '" + library_camera_label + "'.");
    }

    int resolution_height;
    if (!helios::parse_int(camera_node.child("resolution_height").child_value(), resolution_height)) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Invalid or missing resolution_height for camera '" + library_camera_label + "'.");
    }

    // Parse lens optical focal length (physical focal length, not 35mm equivalent)
    float focal_length_mm;
    if (!helios::parse_float(camera_node.child("focal_length_mm").child_value(), focal_length_mm)) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Invalid or missing focal_length_mm for camera '" + library_camera_label + "'.");
    }

    float lens_diameter_mm;
    if (!helios::parse_float(camera_node.child("lens_diameter_mm").child_value(), lens_diameter_mm)) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Invalid or missing lens_diameter_mm for camera '" + library_camera_label + "'.");
    }

    // Parse optional focal plane distance (working distance), default to 2.0m if not specified
    float focal_plane_distance_m = 2.0f;
    if (camera_node.child("focal_plane_distance_m")) {
        if (!helios::parse_float(camera_node.child("focal_plane_distance_m").child_value(), focal_plane_distance_m)) {
            helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Invalid focal_plane_distance_m for camera '" + library_camera_label + "'.");
        }
    }

    // Parse optional lens metadata
    std::string lens_make = camera_node.child("lens_make").child_value();
    std::string lens_model = camera_node.child("lens_model").child_value();
    std::string lens_specification = camera_node.child("lens_specification").child_value();

    // Parse optional exposure mode (default: "auto")
    std::string exposure_mode = camera_node.child("exposure").child_value();
    if (exposure_mode.empty()) {
        exposure_mode = "auto"; // Default to auto exposure
    }

    // Parse optional shutter speed (default: 1/125 second)
    float shutter_speed = 1.0f / 125.0f;
    if (camera_node.child("shutter_speed")) {
        if (!helios::parse_float(camera_node.child("shutter_speed").child_value(), shutter_speed)) {
            std::cerr << "WARNING (RadiationModel::addRadiationCameraFromLibrary): Invalid shutter_speed for camera '" << library_camera_label << "'. Using default 1/125 second." << std::endl;
            shutter_speed = 1.0f / 125.0f;
        }
    }

    // Parse optional white balance mode (default: "auto")
    std::string white_balance_mode = camera_node.child("white_balance").child_value();
    if (white_balance_mode.empty()) {
        white_balance_mode = "auto"; // Default to auto white balance
    } else if (white_balance_mode != "auto" && white_balance_mode != "off") {
        std::cerr << "WARNING (RadiationModel::addRadiationCameraFromLibrary): Invalid white_balance mode '" << white_balance_mode << "' for camera '" << library_camera_label << "'. Must be 'auto' or 'off'. Using default 'auto'." << std::endl;
        white_balance_mode = "auto";
    }

    // Build CameraProperties struct
    CameraProperties camera_properties;
    camera_properties.camera_resolution = helios::make_int2(resolution_width, resolution_height);
    camera_properties.sensor_width_mm = sensor_width_mm;

    // Calculate HFOV from lens optical focal length and sensor width
    // HFOV = 2 * atan(sensor_width / (2 * optical_focal_length))
    float HFOV_rad = 2.0f * atan(sensor_width_mm / (2.0f * focal_length_mm));
    camera_properties.HFOV = HFOV_rad * 180.0f / M_PI;

    // Set focal plane distance (working distance for ray generation)
    camera_properties.focal_plane_distance = focal_plane_distance_m;

    // Convert lens optical focal length from mm to meters (for f-number calculations)
    camera_properties.lens_focal_length = focal_length_mm / 1000.0f;

    // Convert lens diameter from mm to meters
    camera_properties.lens_diameter = lens_diameter_mm / 1000.0f;

    // FOV aspect ratio will be auto-calculated
    camera_properties.FOV_aspect_ratio = 0.0f;

    // Set model name
    camera_properties.model = manufacturer + " " + model;

    // Set lens metadata
    camera_properties.lens_make = lens_make;
    camera_properties.lens_model = lens_model;
    camera_properties.lens_specification = lens_specification;

    // Set exposure settings
    camera_properties.exposure = exposure_mode;
    camera_properties.shutter_speed = shutter_speed;

    // Set white balance mode
    camera_properties.white_balance = white_balance_mode;

    // Parse spectral response data and store in global data
    // xml_band_labels stores the band labels from the XML file (used for global data naming)
    std::vector<std::string> xml_band_labels;
    // spectral_wavelength_ranges stores the wavelength range for each band (for auto-creating bands)
    std::vector<std::pair<float, float>> spectral_wavelength_ranges;

    for (pugi::xml_node spectral_node = camera_node.child("spectral_response"); spectral_node; spectral_node = spectral_node.next_sibling("spectral_response")) {

        std::string xml_band_label = spectral_node.attribute("label").value();
        if (xml_band_label.empty()) {
            helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): spectral_response node missing 'label' attribute for camera '" + library_camera_label + "'.");
        }

        xml_band_labels.push_back(xml_band_label);

        // Parse wavelength-response pairs
        std::vector<helios::vec2> spectral_data;
        std::string data_str = spectral_node.child_value();

        if (!data_str.empty()) {
            std::istringstream data_stream(data_str);
            float wavelength, response;
            while (data_stream >> wavelength >> response) {
                spectral_data.push_back(helios::make_vec2(wavelength, response));
            }
        }

        if (spectral_data.empty()) {
            helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): Empty spectral response data for band '" + xml_band_label + "' in camera '" + library_camera_label + "'.");
        }

        // Store wavelength range for potential band creation
        spectral_wavelength_ranges.emplace_back(spectral_data.front().x, spectral_data.back().x);

        // Store spectral response in global data with naming convention using XML labels
        std::string global_data_label = library_camera_label + "_" + xml_band_label;
        context->setGlobalData(global_data_label.c_str(), spectral_data);
    }

    if (xml_band_labels.empty()) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): No spectral response data found for camera '" + library_camera_label + "'.");
    }

    // Determine effective band labels: use custom labels if provided, otherwise use XML labels
    std::vector<std::string> effective_band_labels;
    if (!custom_band_labels.empty()) {
        if (custom_band_labels.size() != xml_band_labels.size()) {
            helios_runtime_error("ERROR (RadiationModel::addRadiationCameraFromLibrary): custom_band_labels size (" + std::to_string(custom_band_labels.size()) + ") does not match number of spectral responses in library (" +
                                 std::to_string(xml_band_labels.size()) + ") for camera '" + library_camera_label + "'.");
        }
        effective_band_labels = custom_band_labels;
    } else {
        effective_band_labels = xml_band_labels;
    }

    // Add radiation bands if they don't exist (using effective band labels)
    for (size_t i = 0; i < effective_band_labels.size(); i++) {
        const std::string &band_label = effective_band_labels[i];
        if (!doesBandExist(band_label)) {
            float min_wavelength = spectral_wavelength_ranges[i].first;
            float max_wavelength = spectral_wavelength_ranges[i].second;
            addRadiationBand(band_label, min_wavelength, max_wavelength);

            // Disable emission for camera bands
            disableEmission(band_label);

            // Set scattering depth
            setScatteringDepth(band_label, 3);

            std::cout << "WARNING (RadiationModel::addRadiationCameraFromLibrary): Band '" << band_label << "' did not exist and was automatically created with wavelength range [" << min_wavelength << ", " << max_wavelength << "] nm." << std::endl;
        }
    }

    // Create the camera using existing addRadiationCamera method (with effective band labels)
    addRadiationCamera(camera_label, effective_band_labels, position, lookat, camera_properties, antialiasing_samples);

    // Set the camera type
    cameras.at(camera_label).camera_type = camera_type;

    // Set spectral responses from the global data we created (mapping effective labels to XML labels)
    for (size_t i = 0; i < effective_band_labels.size(); i++) {
        std::string global_data_label = library_camera_label + "_" + xml_band_labels[i];
        setCameraSpectralResponse(camera_label, effective_band_labels[i], global_data_label);
    }
}

void RadiationModel::setCameraPosition(const std::string &camera_label, const helios::vec3 &position) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::setCameraPosition): Camera '" + camera_label + "' does not exist.");
    } else if (position == cameras.at(camera_label).lookat) {
        helios_runtime_error("ERROR (RadiationModel::setCameraPosition): Camera position cannot be equal to the 'lookat' position.");
    }

    cameras.at(camera_label).position = position;

    if (iscameravisualizationenabled) {
        updateCameraModelPosition(camera_label);
    }
}

helios::vec3 RadiationModel::getCameraPosition(const std::string &camera_label) const {

    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraPosition): Camera '" + camera_label + "' does not exist.");
    }

    return cameras.at(camera_label).position;
}

void RadiationModel::setCameraLookat(const std::string &camera_label, const helios::vec3 &lookat) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLookat): Camera '" + camera_label + "' does not exist.");
    }

    cameras.at(camera_label).lookat = lookat;

    if (iscameravisualizationenabled) {
        updateCameraModelPosition(camera_label);
    }
}

helios::vec3 RadiationModel::getCameraLookat(const std::string &camera_label) const {

    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraLookat): Camera '" + camera_label + "' does not exist.");
    }

    return cameras.at(camera_label).lookat;
}

void RadiationModel::setCameraOrientation(const std::string &camera_label, const helios::vec3 &direction) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::setCameraOrientation): Camera '" + camera_label + "' does not exist.");
    }

    cameras.at(camera_label).lookat = cameras.at(camera_label).position + direction;

    if (iscameravisualizationenabled) {
        updateCameraModelPosition(camera_label);
    }
}

helios::SphericalCoord RadiationModel::getCameraOrientation(const std::string &camera_label) const {

    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraOrientation): Camera '" + camera_label + "' does not exist.");
    }

    return cart2sphere(cameras.at(camera_label).lookat - cameras.at(camera_label).position);
}

void RadiationModel::setCameraOrientation(const std::string &camera_label, const helios::SphericalCoord &direction) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::setCameraOrientation): Camera '" + camera_label + "' does not exist.");
    }

    cameras.at(camera_label).lookat = cameras.at(camera_label).position + sphere2cart(direction);

    if (iscameravisualizationenabled) {
        updateCameraModelPosition(camera_label);
    }
}

CameraProperties RadiationModel::getCameraParameters(const std::string &camera_label) const {

    // Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraParameters): Camera '" + camera_label + "' does not exist.");
    }

    // Get reference to camera
    const auto &camera = cameras.at(camera_label);

    // Create and populate CameraProperties struct
    CameraProperties camera_properties;
    camera_properties.camera_resolution = camera.resolution;
    camera_properties.HFOV = camera.HFOV_degrees;
    camera_properties.lens_diameter = camera.lens_diameter;
    camera_properties.focal_plane_distance = camera.focal_length;
    camera_properties.lens_focal_length = camera.lens_focal_length;
    camera_properties.sensor_width_mm = camera.sensor_width_mm;
    camera_properties.model = camera.model;
    camera_properties.lens_make = camera.lens_make;
    camera_properties.lens_model = camera.lens_model;
    camera_properties.lens_specification = camera.lens_specification;
    camera_properties.exposure = camera.exposure;
    camera_properties.shutter_speed = camera.shutter_speed;
    camera_properties.white_balance = camera.white_balance;
    camera_properties.camera_zoom = camera.camera_zoom;
    camera_properties.FOV_aspect_ratio = camera.FOV_aspect_ratio;

    return camera_properties;
}

void RadiationModel::updateCameraParameters(const std::string &camera_label, const CameraProperties &camera_properties) {

    // Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::updateCameraParameters): Camera '" + camera_label + "' does not exist.");
    }

    // Validate camera properties
    if (camera_properties.camera_resolution.x <= 0 || camera_properties.camera_resolution.y <= 0) {
        helios_runtime_error("ERROR (RadiationModel::updateCameraParameters): Camera resolution must be at least 1x1.");
    } else if (camera_properties.HFOV <= 0 || camera_properties.HFOV >= 180.f) {
        helios_runtime_error("ERROR (RadiationModel::updateCameraParameters): Camera horizontal field of view must be between 0 and 180 degrees.");
    } else if (camera_properties.camera_zoom <= 0.0f) {
        helios_runtime_error("ERROR (RadiationModel::updateCameraParameters): camera_zoom must be greater than 0.");
    }

    // Get reference to camera
    auto &camera = cameras.at(camera_label);

    // Update camera parameters
    camera.resolution = camera_properties.camera_resolution;
    camera.HFOV_degrees = camera_properties.HFOV;
    camera.lens_diameter = camera_properties.lens_diameter;
    camera.focal_length = camera_properties.focal_plane_distance;
    camera.lens_focal_length = camera_properties.lens_focal_length;
    camera.sensor_width_mm = camera_properties.sensor_width_mm;
    camera.model = camera_properties.model;
    camera.exposure = camera_properties.exposure;
    camera.shutter_speed = camera_properties.shutter_speed;
    camera.white_balance = camera_properties.white_balance;
    camera.camera_zoom = camera_properties.camera_zoom;

    // Recalculate FOV_aspect_ratio to ensure square pixels
    camera.FOV_aspect_ratio = float(camera.resolution.x) / float(camera.resolution.y);

    // Flag that radiative properties need to be updated
    radiativepropertiesneedupdate = true;

    // Update camera visualization if enabled
    if (iscameravisualizationenabled) {
        updateCameraModelPosition(camera_label);
    }
}

std::vector<std::string> RadiationModel::getAllCameraLabels() {
    std::vector<std::string> labels(cameras.size());
    uint cam = 0;
    for (const auto &camera: cameras) {
        labels.at(cam) = camera.second.label;
        cam++;
    }
    return labels;
}

std::string RadiationModel::writeCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path, int frame, float flux_to_pixel_conversion) {

    // check if camera exists
    if (cameras.find(camera) == cameras.end()) {
        std::cout << "ERROR (RadiationModel::writeCameraImage): camera with label " << camera << " does not exist. Skipping image write for this camera." << std::endl;
        return "";
    }

    if (bands.size() != 1 && bands.size() != 3) {
        helios_runtime_error("ERROR (RadiationModel::writeCameraImage): input vector of band labels should either have length of 1 (grayscale image) or length of 3 (RGB image). Skipping image write for this camera.");
    }

    std::vector<std::vector<float>> camera_data(bands.size());

    uint b = 0;
    for (const auto &band: bands) {

        // check if band exists
        if (std::find(cameras.at(camera).band_labels.begin(), cameras.at(camera).band_labels.end(), band) == cameras.at(camera).band_labels.end()) {
            std::cout << "ERROR (RadiationModel::writeCameraImage): camera " << camera << " band with label " << band << " does not exist. Skipping image write for this camera." << std::endl;
            return "";
        }

        camera_data.at(b) = cameras.at(camera).pixel_data.at(band);

        b++;
    }

    // Apply sRGB gamma compression for 3-channel (RGB) images
    // This is done on the copy, preserving the original linear data
    bool is_rgb = (camera_data.size() == 3);
    if (is_rgb) {
        for (auto &band_data: camera_data) {
            for (float &v: band_data) {
                v = RadiationCamera::lin_to_srgb(std::fmaxf(0.0f, v));
            }
        }
    }

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeCameraImage): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeCameraImage): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << camera << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".jpeg";
    } else {
        outfile << camera << "_" << imagefile_base << ".jpeg";
    }
    std::ofstream testfile(outfile.str());

    if (!testfile.is_open()) {
        std::cout << "ERROR (RadiationModel::writeCameraImage): image file " << outfile.str() << " could not be opened. Check that the path exists and that you have write permission. Skipping image write for this camera." << std::endl;
        return "";
    }
    testfile.close();

    int2 camera_resolution = cameras.at(camera).resolution;

    std::vector<RGBcolor> pixel_data_RGB(camera_resolution.x * camera_resolution.y);

    RGBcolor pixel_color;
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            if (camera_data.size() == 1) {
                float c = camera_data.front().at(j * camera_resolution.x + i);
                pixel_color = make_RGBcolor(c, c, c);
            } else {
                pixel_color = make_RGBcolor(camera_data.at(0).at(j * camera_resolution.x + i), camera_data.at(1).at(j * camera_resolution.x + i), camera_data.at(2).at(j * camera_resolution.x + i));
            }
            pixel_color.scale(flux_to_pixel_conversion);
            uint ii = camera_resolution.x - i - 1;
            uint jj = camera_resolution.y - j - 1;
            pixel_data_RGB.at(jj * camera_resolution.x + ii) = pixel_color;
        }
    }

    writeJPEG(outfile.str(), camera_resolution.x, camera_resolution.y, pixel_data_RGB);

    std::string image_filepath = outfile.str();

    // Write JSON metadata if enabled for this camera
    if (metadata_enabled_cameras.find(camera) != metadata_enabled_cameras.end()) {
        // Preserve any existing image_processing parameters (e.g., from applyCameraImageCorrections)
        CameraMetadata::ImageProcessingProperties saved_image_processing;
        if (camera_metadata.find(camera) != camera_metadata.end()) {
            saved_image_processing = camera_metadata.at(camera).image_processing;
        }

        // Re-populate metadata to capture any new data (e.g., agronomic properties)
        CameraMetadata metadata;
        populateCameraMetadata(camera, metadata);

        // Restore image_processing parameters
        metadata.image_processing = saved_image_processing;

        // Set color space based on channel count (sRGB for RGB, linear for grayscale)
        metadata.image_processing.color_space = is_rgb ? "sRGB" : "linear";

        // Extract just the filename (without directory path) for portability
        size_t last_slash = image_filepath.find_last_of("/\\");
        std::string filename_only = (last_slash != std::string::npos) ? image_filepath.substr(last_slash + 1) : image_filepath;
        metadata.path = filename_only;

        // Store updated metadata and write JSON file
        camera_metadata[camera] = metadata;
        writeCameraMetadataFile(camera, output_path);
    }

    return image_filepath;
}

std::string RadiationModel::writeNormCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path, int frame) {
    float maxval = 0;
    // Find maximum mean value over all bands
    for (const std::string &band: bands) {
        std::string global_data_label = "camera_" + camera + "_" + band;
        if (std::find(cameras.at(camera).band_labels.begin(), cameras.at(camera).band_labels.end(), band) == cameras.at(camera).band_labels.end()) {
            std::cout << "ERROR (RadiationModel::writeNormCameraImage): camera " << camera << " band with label " << band << " does not exist. Skipping image write for this camera." << std::endl;
            return "";
        } else if (!context->doesGlobalDataExist(global_data_label.c_str())) {
            std::cout << "ERROR (RadiationModel::writeNormCameraImage): image data for camera " << camera << ", band " << band << " has not been created. Did you run the radiation model? Skipping image write for this camera." << std::endl;
            return "";
        }
        std::vector<float> cameradata;
        context->getGlobalData(global_data_label.c_str(), cameradata);
        for (float val: cameradata) {
            if (val > maxval) {
                maxval = val;
            }
        }
    }
    // Normalize all bands
    for (const std::string &band: bands) {
        std::string global_data_label = "camera_" + camera + "_" + band;
        std::vector<float> cameradata;
        context->getGlobalData(global_data_label.c_str(), cameradata);
        for (float &val: cameradata) {
            val = val / maxval;
        }
        context->setGlobalData(global_data_label.c_str(), cameradata);
    }

    return RadiationModel::writeCameraImage(camera, bands, imagefile_base, image_path, frame);
}

void RadiationModel::writeCameraImageData(const std::string &camera, const std::string &band, const std::string &imagefile_base, const std::string &image_path, int frame) {

    // check if camera exists
    if (cameras.find(camera) == cameras.end()) {
        std::cout << "ERROR (RadiationModel::writeCameraImageData): camera with label " << camera << " does not exist. Skipping image write for this camera." << std::endl;
        return;
    }

    std::vector<float> camera_data;

    // check if band exists
    if (std::find(cameras.at(camera).band_labels.begin(), cameras.at(camera).band_labels.end(), band) == cameras.at(camera).band_labels.end()) {
        std::cout << "ERROR (RadiationModel::writeCameraImageData): camera " << camera << " band with label " << band << " does not exist. Skipping image write for this camera." << std::endl;
        return;
    }

    std::string global_data_label = "camera_" + camera + "_" + band;

    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        std::cout << "ERROR (RadiationModel::writeCameraImageData): image data for camera " << camera << ", band " << band << " has not been created. Did you run the radiation model? Skipping image write for this camera." << std::endl;
        return;
    }

    context->getGlobalData(global_data_label.c_str(), camera_data);

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeCameraImage): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeCameraImage): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << camera << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".txt";
    } else {
        outfile << camera << "_" << imagefile_base << ".txt";
    }

    std::ofstream outfilestream(outfile.str());

    if (!outfilestream.is_open()) {
        std::cout << "ERROR (RadiationModel::writeCameraImageData): image file " << outfile.str() << " could not be opened. Check that the path exists and that you have write permission. Skipping image write for this camera." << std::endl;
        return;
    }

    int2 camera_resolution = cameras.at(camera).resolution;

    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = camera_resolution.x - 1; i >= 0; i--) {
            outfilestream << camera_data.at(j * camera_resolution.x + i) << " ";
        }
        outfilestream << "\n";
    }

    outfilestream.close();
}

void RadiationModel::setCameraCalibration(CameraCalibration *CameraCalibration) {
    cameracalibration = CameraCalibration;
    calibration_flag = true;
}

void RadiationModel::updateCameraResponse(const std::string &orginalcameralabel, const std::vector<std::string> &sourcelabels_raw, const std::vector<std::string> &cameraresponselabels, vec2 &wavelengthrange,
                                          const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark) {

    std::vector<std::string> objectlabels;
    vec2 wavelengthrange_c = wavelengthrange;
    cameracalibration->preprocessSpectra(sourcelabels_raw, cameraresponselabels, objectlabels, wavelengthrange_c);

    RadiationCamera calibratecamera = cameras.at(orginalcameralabel);
    CameraProperties cameraproperties;
    cameraproperties.HFOV = calibratecamera.HFOV_degrees;
    cameraproperties.camera_resolution = calibratecamera.resolution;
    cameraproperties.focal_plane_distance = calibratecamera.focal_length; // Working distance for ray generation
    cameraproperties.lens_focal_length = calibratecamera.lens_focal_length; // Optical focal length for aperture
    cameraproperties.lens_diameter = calibratecamera.lens_diameter;
    cameraproperties.FOV_aspect_ratio = calibratecamera.FOV_aspect_ratio;
    cameraproperties.exposure = calibratecamera.exposure;
    cameraproperties.shutter_speed = calibratecamera.shutter_speed;

    std::vector<uint> UUIDs_target = cameracalibration->getAllColorBoardUUIDs();
    std::string cameralabel = "calibration";
    std::map<uint, std::vector<vec2>> simulatedcolorboardspectra;
    for (uint UUID: UUIDs_target) {
        simulatedcolorboardspectra.emplace(UUID, NULL);
    }

    for (uint ID = 0; ID < radiation_sources.size(); ID++) {
        RadiationModel::setSourceSpectrumIntegral(ID, 1);
    }

    std::vector<float> wavelengths;
    context->getGlobalData("wavelengths", wavelengths);
    int numberwavelengths = wavelengths.size();

    for (int iw = 0; iw < numberwavelengths; iw++) {
        std::string wavelengthlabel = std::to_string(wavelengths.at(iw));

        std::vector<std::string> sourcelabels;
        for (std::string sourcelabel_raw: sourcelabels_raw) {
            std::vector<vec2> icalsource;
            icalsource.push_back(cameracalibration->processedspectra.at("source").at(sourcelabel_raw).at(iw));
            icalsource.push_back(cameracalibration->processedspectra.at("source").at(sourcelabel_raw).at(iw));
            icalsource.at(1).x += 1;
            std::string sourcelable = "Cal_source_" + sourcelabel_raw;
            sourcelabels.push_back(sourcelable);
            context->setGlobalData(sourcelable.c_str(), icalsource);
        }

        std::vector<vec2> icalcamera(2);
        icalcamera.at(0).y = 1;
        icalcamera.at(1).y = 1;
        icalcamera.at(0).x = wavelengths.at(iw);
        icalcamera.at(1).x = wavelengths.at(iw) + 1;
        std::string camlable = "Cal_cameraresponse";
        context->setGlobalData(camlable.c_str(), icalcamera);

        for (auto objectpair: cameracalibration->processedspectra.at("object")) {
            std::vector<vec2> spectrum_obj;
            spectrum_obj.push_back(objectpair.second.at(iw));
            spectrum_obj.push_back(objectpair.second.at(iw));
            spectrum_obj.at(1).x += 1;
            context->setGlobalData(objectpair.first.c_str(), spectrum_obj);
        }

        RadiationModel::addRadiationBand(wavelengthlabel, std::stof(wavelengthlabel), std::stof(wavelengthlabel) + 1);
        RadiationModel::disableEmission(wavelengthlabel);

        uint ID = 0;
        for (std::string sourcelabel_raw: sourcelabels_raw) {
            RadiationModel::setSourceSpectrum(ID, sourcelabels.at(ID).c_str());
            RadiationModel::setSourceFlux(ID, wavelengthlabel, 1);
            ID++;
        }
        RadiationModel::setScatteringDepth(wavelengthlabel, 1);
        RadiationModel::setDiffuseRadiationFlux(wavelengthlabel, 0);
        RadiationModel::setDiffuseRadiationExtinctionCoeff(wavelengthlabel, 0.f, make_vec3(-0.5, 0.5, 1));

        RadiationModel::addRadiationCamera(cameralabel, {wavelengthlabel}, calibratecamera.position, calibratecamera.lookat, cameraproperties, 10);
        RadiationModel::setCameraSpectralResponse(cameralabel, wavelengthlabel, camlable);
        RadiationModel::updateGeometry();
        RadiationModel::runBand({wavelengthlabel});

        std::vector<float> camera_data;
        std::string global_data_label = "camera_" + cameralabel + "_" + wavelengthlabel;
        context->getGlobalData(global_data_label.c_str(), camera_data);

        std::vector<uint> pixel_labels;
        std::string global_data_label_UUID = "camera_" + cameralabel + "_pixel_UUID";
        context->getGlobalData(global_data_label_UUID.c_str(), pixel_labels);

        for (uint j = 0; j < calibratecamera.resolution.y; j++) {
            for (uint i = 0; i < calibratecamera.resolution.x; i++) {
                float icdata = camera_data.at(j * calibratecamera.resolution.x + i);

                uint UUID = pixel_labels.at(j * calibratecamera.resolution.x + i) - 1;
                if (find(UUIDs_target.begin(), UUIDs_target.end(), UUID) != UUIDs_target.end()) {
                    if (simulatedcolorboardspectra.at(UUID).empty()) {
                        simulatedcolorboardspectra.at(UUID).push_back(make_vec2(wavelengths.at(iw), icdata / float(numberwavelengths)));
                    } else if (simulatedcolorboardspectra.at(UUID).back().x == wavelengths.at(iw)) {
                        simulatedcolorboardspectra.at(UUID).back().y += icdata / float(numberwavelengths);
                    } else if (simulatedcolorboardspectra.at(UUID).back().x != wavelengths.at(iw)) {
                        simulatedcolorboardspectra.at(UUID).push_back(make_vec2(wavelengths.at(iw), icdata / float(numberwavelengths)));
                    }
                }
            }
        }
    }
    // Update camera response spectra
    cameracalibration->updateCameraResponseSpectra(cameraresponselabels, calibratedmark, simulatedcolorboardspectra, truevalues);
    // Reset color board spectra
    std::vector<uint> UUIDs_colorbd = cameracalibration->getAllColorBoardUUIDs();
    for (uint UUID: UUIDs_colorbd) {
        std::string colorboardspectra;
        context->getPrimitiveData(UUID, "reflectivity_spectrum", colorboardspectra);
        context->setPrimitiveData(UUID, "reflectivity_spectrum", colorboardspectra + "_raw");
    }
}

void RadiationModel::runRadiationImaging(const std::string &cameralabel, const std::vector<std::string> &sourcelabels, const std::vector<std::string> &bandlabels, const std::vector<std::string> &cameraresponselabels, helios::vec2 wavelengthrange,
                                         float fluxscale, float diffusefactor, uint scatteringdepth) {

    float sources_fluxsum = 0;
    std::vector<float> sources_fluxes;
    for (uint ID = 0; ID < sourcelabels.size(); ID++) {
        std::vector<vec2> Source_spectrum = loadSpectralData(sourcelabels.at(ID).c_str());
        sources_fluxes.push_back(RadiationModel::integrateSpectrum(Source_spectrum, wavelengthrange.x, wavelengthrange.y));
        RadiationModel::setSourceSpectrum(ID, sourcelabels.at(ID).c_str());
        RadiationModel::setSourceSpectrumIntegral(ID, sources_fluxes.at(ID));
        sources_fluxsum += sources_fluxes.at(ID);
    }

    RadiationModel::addRadiationBand(bandlabels.at(0), wavelengthrange.x, wavelengthrange.y);
    RadiationModel::disableEmission(bandlabels.at(0));
    for (uint ID = 0; ID < radiation_sources.size(); ID++) {
        RadiationModel::setSourceFlux(ID, bandlabels.at(0), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
    }
    RadiationModel::setScatteringDepth(bandlabels.at(0), scatteringdepth);
    RadiationModel::setDiffuseRadiationFlux(bandlabels.at(0), diffusefactor * sources_fluxsum);
    RadiationModel::setDiffuseRadiationExtinctionCoeff(bandlabels.at(0), 1.f, make_vec3(-0.5, 0.5, 1));

    if (bandlabels.size() > 1) {
        for (int iband = 1; iband < bandlabels.size(); iband++) {
            RadiationModel::copyRadiationBand(bandlabels.at(iband - 1), bandlabels.at(iband), wavelengthrange.x, wavelengthrange.y);
            for (uint ID = 0; ID < radiation_sources.size(); ID++) {
                RadiationModel::setSourceFlux(ID, bandlabels.at(iband), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
            }
            RadiationModel::setDiffuseRadiationFlux(bandlabels.at(iband), diffusefactor * sources_fluxsum);
        }
    }

    for (int iband = 0; iband < bandlabels.size(); iband++) {
        RadiationModel::setCameraSpectralResponse(cameralabel, bandlabels.at(iband), cameraresponselabels.at(iband));
    }

    RadiationModel::updateGeometry();
    RadiationModel::runBand(bandlabels);
}

void RadiationModel::runRadiationImaging(const std::vector<std::string> &cameralabels, const std::vector<std::string> &sourcelabels, const std::vector<std::string> &bandlabels, const std::vector<std::string> &cameraresponselabels,
                                         helios::vec2 wavelengthrange, float fluxscale, float diffusefactor, uint scatteringdepth) {

    float sources_fluxsum = 0;
    std::vector<float> sources_fluxes;
    for (uint ID = 0; ID < sourcelabels.size(); ID++) {
        std::vector<vec2> Source_spectrum = loadSpectralData(sourcelabels.at(ID).c_str());
        sources_fluxes.push_back(RadiationModel::integrateSpectrum(Source_spectrum, wavelengthrange.x, wavelengthrange.y));
        RadiationModel::setSourceSpectrum(ID, sourcelabels.at(ID).c_str());
        RadiationModel::setSourceSpectrumIntegral(ID, sources_fluxes.at(ID));
        sources_fluxsum += sources_fluxes.at(ID);
    }

    RadiationModel::addRadiationBand(bandlabels.at(0), wavelengthrange.x, wavelengthrange.y);
    RadiationModel::disableEmission(bandlabels.at(0));
    for (uint ID = 0; ID < radiation_sources.size(); ID++) {
        RadiationModel::setSourceFlux(ID, bandlabels.at(0), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
    }
    RadiationModel::setScatteringDepth(bandlabels.at(0), scatteringdepth);
    RadiationModel::setDiffuseRadiationFlux(bandlabels.at(0), diffusefactor * sources_fluxsum);
    RadiationModel::setDiffuseRadiationExtinctionCoeff(bandlabels.at(0), 1.f, make_vec3(-0.5, 0.5, 1));

    if (bandlabels.size() > 1) {
        for (int iband = 1; iband < bandlabels.size(); iband++) {
            RadiationModel::copyRadiationBand(bandlabels.at(iband - 1), bandlabels.at(iband), wavelengthrange.x, wavelengthrange.y);
            for (uint ID = 0; ID < radiation_sources.size(); ID++) {
                RadiationModel::setSourceFlux(ID, bandlabels.at(iband), (1 - diffusefactor) * sources_fluxes.at(ID) * fluxscale);
            }
            RadiationModel::setDiffuseRadiationFlux(bandlabels.at(iband), diffusefactor * sources_fluxsum);
        }
    }

    for (int ic = 0; ic < cameralabels.size(); ic++) {
        for (int iband = 0; iband < bandlabels.size(); iband++) {
            RadiationModel::setCameraSpectralResponse(cameralabels.at(ic), bandlabels.at(iband), cameraresponselabels.at(iband));
        }
    }


    RadiationModel::updateGeometry();
    RadiationModel::runBand(bandlabels);
}

float RadiationModel::getCameraResponseScale(const std::string &orginalcameralabel, const std::vector<std::string> &cameraresponselabels, const std::vector<std::string> &bandlabels, const std::vector<std::string> &sourcelabels, vec2 &wavelengthrange,
                                             const std::vector<std::vector<float>> &truevalues) {


    RadiationCamera calibratecamera = cameras.at(orginalcameralabel);
    CameraProperties cameraproperties;
    cameraproperties.HFOV = calibratecamera.HFOV_degrees;
    cameraproperties.camera_resolution = calibratecamera.resolution;
    cameraproperties.focal_plane_distance = calibratecamera.focal_length; // Working distance for ray generation
    cameraproperties.lens_focal_length = calibratecamera.lens_focal_length; // Optical focal length for aperture
    cameraproperties.lens_diameter = calibratecamera.lens_diameter;
    cameraproperties.FOV_aspect_ratio = calibratecamera.FOV_aspect_ratio;
    cameraproperties.exposure = calibratecamera.exposure;
    cameraproperties.shutter_speed = calibratecamera.shutter_speed;

    std::string cameralabel = orginalcameralabel + "Scale";
    RadiationModel::addRadiationCamera(cameralabel, bandlabels, calibratecamera.position, calibratecamera.lookat, cameraproperties, 20);
    RadiationModel::runRadiationImaging(cameralabel, sourcelabels, bandlabels, cameraresponselabels, wavelengthrange, 1, 0);

    // Get camera spectral response scale based on comparing true values and calibrated image
    float camerascale = cameracalibration->getCameraResponseScale(cameralabel, cameraproperties.camera_resolution, bandlabels, truevalues);
    return camerascale;
}


void RadiationModel::writePrimitiveDataLabelMap(const std::string &cameralabel, const std::string &primitive_data_label, const std::string &imagefile_base, const std::string &image_path, int frame, float padvalue) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writePrimitiveDataLabelMap): Camera '" + cameralabel + "' does not exist.");
    }

    // Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writePrimitiveDataLabelMap): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writePrimitiveDataLabelMap): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writePrimitiveDataLabelMap): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".txt";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".txt";
    }

    // Output label image in ".txt" format
    std::ofstream pixel_data(outfile.str());

    if (!pixel_data.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writePrimitiveDataLabelMap): Could not open file '" + outfile.str() + "' for writing.");
    }

    bool empty_flag = true;
    // Apply horizontal flip to match mask coordinate system
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1; // horizontal flip
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID, primitive_data_label.c_str())) {
                HeliosDataType datatype = context->getPrimitiveDataType(primitive_data_label.c_str());
                if (datatype == HELIOS_TYPE_FLOAT) {
                    float labeldata;
                    context->getPrimitiveData(UUID, primitive_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else if (datatype == HELIOS_TYPE_UINT) {
                    uint labeldata;
                    context->getPrimitiveData(UUID, primitive_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else if (datatype == HELIOS_TYPE_INT) {
                    int labeldata;
                    context->getPrimitiveData(UUID, primitive_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else if (datatype == HELIOS_TYPE_DOUBLE) {
                    double labeldata;
                    context->getPrimitiveData(UUID, primitive_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else {
                    pixel_data << padvalue << " ";
                }
            } else {
                pixel_data << padvalue << " ";
            }
        }
        pixel_data << "\n";
    }
    pixel_data.close();

    if (empty_flag) {
        std::cerr << "WARNING (RadiationModel::writePrimitiveDataLabelMap): No primitive data of " << primitive_data_label << " found in camera image. Primitive data map contains only padded values." << std::endl;
    }
}

void RadiationModel::writeObjectDataLabelMap(const std::string &cameralabel, const std::string &object_data_label, const std::string &imagefile_base, const std::string &image_path, int frame, float padvalue) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeObjectDataLabelMap): Camera '" + cameralabel + "' does not exist.");
    }

    // Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeObjectDataLabelMap): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeObjectDataLabelMap): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeObjectDataLabelMap): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".txt";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".txt";
    }

    // Output label image in ".txt" format
    std::ofstream pixel_data(outfile.str());

    if (!pixel_data.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeObjectDataLabelMap): Could not open file '" + outfile.str() + "' for writing.");
    }

    bool empty_flag = true;
    // Apply horizontal flip to match mask coordinate system
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1; // horizontal flip
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;
            if (!context->doesPrimitiveExist(UUID)) {
                pixel_data << padvalue << " ";
                continue;
            }
            uint objID = context->getPrimitiveParentObjectID(UUID);
            if (context->doesObjectExist(objID) && context->doesObjectDataExist(objID, object_data_label.c_str())) {
                HeliosDataType datatype = context->getObjectDataType(object_data_label.c_str());
                if (datatype == HELIOS_TYPE_FLOAT) {
                    float labeldata;
                    context->getObjectData(objID, object_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else if (datatype == HELIOS_TYPE_UINT) {
                    uint labeldata;
                    context->getObjectData(objID, object_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else if (datatype == HELIOS_TYPE_INT) {
                    int labeldata;
                    context->getObjectData(objID, object_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else if (datatype == HELIOS_TYPE_DOUBLE) {
                    double labeldata;
                    context->getObjectData(objID, object_data_label.c_str(), labeldata);
                    pixel_data << labeldata << " ";
                    empty_flag = false;
                } else {
                    pixel_data << padvalue << " ";
                }
            } else {
                pixel_data << padvalue << " ";
            }
        }
        pixel_data << "\n";
    }
    pixel_data.close();

    if (empty_flag) {
        std::cerr << "WARNING (RadiationModel::writeObjectDataLabelMap): No object data of " << object_data_label << " found in camera image. Object data map contains only padded values." << std::endl;
    }
}

void RadiationModel::writeDepthImageData(const std::string &cameralabel, const std::string &imagefile_base, const std::string &image_path, int frame) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeDepthImageData): Camera '" + cameralabel + "' does not exist.");
    }

    std::string global_data_label = "camera_" + cameralabel + "_pixel_depth";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeDepthImageData): Depth data for camera '" + cameralabel + "' does not exist. Was the radiation model run for the camera?");
    }
    std::vector<float> camera_depth;
    context->getGlobalData(global_data_label.c_str(), camera_depth);
    helios::vec3 camera_position = cameras.at(cameralabel).position;
    helios::vec3 camera_lookat = cameras.at(cameralabel).lookat;

    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeDepthImageData): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeDepthImageData): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".txt";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".txt";
    }

    // Output label image in ".txt" format
    std::ofstream pixel_data(outfile.str());

    if (!pixel_data.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeDepthImageData): Could not open file '" + outfile.str() + "' for writing.");
    }

    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = camera_resolution.x - 1; i >= 0; i--) {
            pixel_data << camera_depth.at(j * camera_resolution.x + i) << " ";
        }
        pixel_data << "\n";
    }

    pixel_data.close();
}

void RadiationModel::writeNormDepthImage(const std::string &cameralabel, const std::string &imagefile_base, float max_depth, const std::string &image_path, int frame) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeNormDepthImage): Camera '" + cameralabel + "' does not exist.");
    }

    std::string global_data_label = "camera_" + cameralabel + "_pixel_depth";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeNormDepthImage): Depth data for camera '" + cameralabel + "' does not exist. Was the radiation model run for the camera?");
    }
    std::vector<float> camera_depth;
    context->getGlobalData(global_data_label.c_str(), camera_depth);
    helios::vec3 camera_position = cameras.at(cameralabel).position;
    helios::vec3 camera_lookat = cameras.at(cameralabel).lookat;

    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeNormDepthImage): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeNormDepthImage): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".jpeg";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".jpeg";
    }

    float min_depth = 99999;
    for (int i = 0; i < camera_depth.size(); i++) {
        if (camera_depth.at(i) < 0 || camera_depth.at(i) > max_depth) {
            camera_depth.at(i) = max_depth;
        }
        if (camera_depth.at(i) < min_depth) {
            min_depth = camera_depth.at(i);
        }
    }
    for (int i = 0; i < camera_depth.size(); i++) {
        camera_depth.at(i) = 1.f - (camera_depth.at(i) - min_depth) / (max_depth - min_depth);
    }

    std::vector<RGBcolor> pixel_data(camera_resolution.x * camera_resolution.y);

    RGBcolor pixel_color;
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {

            float c = camera_depth.at(j * camera_resolution.x + i);
            pixel_color = make_RGBcolor(c, c, c);

            uint ii = camera_resolution.x - i - 1;
            uint jj = camera_resolution.y - j - 1;
            pixel_data.at(jj * camera_resolution.x + ii) = pixel_color;
        }
    }

    writeJPEG(outfile.str(), camera_resolution.x, camera_resolution.y, pixel_data);
}

// DEPRECATED
void RadiationModel::writeImageBoundingBoxes(const std::string &cameralabel, const std::string &primitive_data_label, uint object_class_ID, const std::string &imagefile_base, const std::string &image_path, bool append_label_file, int frame) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Camera '" + cameralabel + "' does not exist.");
    }

    // Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeImageBoundingBoxes): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".txt";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".txt";
    }

    // Output label image in ".txt" format
    std::ofstream label_file;
    if (append_label_file) {
        label_file.open(outfile.str(), std::ios::out | std::ios::app);
    } else {
        label_file.open(outfile.str());
    }

    if (!label_file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Could not open file '" + outfile.str() + "'.");
    }

    std::map<int, vec4> pdata_bounds;

    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + i) - 1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID, primitive_data_label.c_str())) {

                uint labeldata;

                HeliosDataType datatype = context->getPrimitiveDataType(primitive_data_label.c_str());
                if (datatype == HELIOS_TYPE_UINT) {
                    uint labeldata_ui;
                    context->getPrimitiveData(UUID, primitive_data_label.c_str(), labeldata_ui);
                    labeldata = labeldata_ui;
                } else if (datatype == HELIOS_TYPE_INT) {
                    int labeldata_i;
                    context->getPrimitiveData(UUID, primitive_data_label.c_str(), labeldata_i);
                    labeldata = (uint) labeldata_i;
                } else {
                    continue;
                }

                if (pdata_bounds.find(labeldata) == pdata_bounds.end()) {
                    pdata_bounds[labeldata] = make_vec4(1e6, -1, 1e6, -1);
                }

                if (i < pdata_bounds[labeldata].x) {
                    pdata_bounds[labeldata].x = i;
                }
                if (i > pdata_bounds[labeldata].y) {
                    pdata_bounds[labeldata].y = i;
                }
                if (j < pdata_bounds[labeldata].z) {
                    pdata_bounds[labeldata].z = j;
                }
                if (j > pdata_bounds[labeldata].w) {
                    pdata_bounds[labeldata].w = j;
                }
            }
        }
    }

    for (auto box: pdata_bounds) {
        vec4 bbox = box.second;
        if (bbox.x == bbox.y || bbox.z == bbox.w) { // filter boxes of zeros size
            continue;
        }
        label_file << object_class_ID << " " << (bbox.x + 0.5 * (bbox.y - bbox.x)) / float(camera_resolution.x) << " " << (bbox.z + 0.5 * (bbox.w - bbox.z)) / float(camera_resolution.y) << " " << std::setprecision(6) << std::fixed
                   << (bbox.y - bbox.x) / float(camera_resolution.x) << " " << (bbox.w - bbox.z) / float(camera_resolution.y) << std::endl;
    }

    label_file.close();
}

// DEPRECATED
void RadiationModel::writeImageBoundingBoxes_ObjectData(const std::string &cameralabel, const std::string &object_data_label, uint object_class_ID, const std::string &imagefile_base, const std::string &image_path, bool append_label_file, int frame) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Camera '" + cameralabel + "' does not exist.");
    }

    // Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeImageBoundingBoxes_ObjectData): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::ostringstream outfile;
    outfile << output_path;

    if (frame >= 0) {
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".txt";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".txt";
    }

    // Output label image in ".txt" format
    std::ofstream label_file;
    if (append_label_file) {
        label_file.open(outfile.str(), std::ios::out | std::ios::app);
    } else {
        label_file.open(outfile.str());
    }

    if (!label_file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Could not open file '" + outfile.str() + "'.");
    }

    std::map<int, vec4> pdata_bounds;

    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1;
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;

            if (!context->doesPrimitiveExist(UUID)) {
                continue;
            }

            uint objID = context->getPrimitiveParentObjectID(UUID);

            if (!context->doesObjectExist(objID) || !context->doesObjectDataExist(objID, object_data_label.c_str())) {
                continue;
            }

            uint labeldata;

            HeliosDataType datatype = context->getObjectDataType(object_data_label.c_str());
            if (datatype == HELIOS_TYPE_UINT) {
                uint labeldata_ui;
                context->getObjectData(objID, object_data_label.c_str(), labeldata_ui);
                labeldata = labeldata_ui;
            } else if (datatype == HELIOS_TYPE_INT) {
                int labeldata_i;
                context->getObjectData(objID, object_data_label.c_str(), labeldata_i);
                labeldata = (uint) labeldata_i;
            } else {
                continue;
            }

            if (pdata_bounds.find(labeldata) == pdata_bounds.end()) {
                pdata_bounds[labeldata] = make_vec4(1e6, -1, 1e6, -1);
            }

            if (i < pdata_bounds[labeldata].x) {
                pdata_bounds[labeldata].x = i;
            }
            if (i > pdata_bounds[labeldata].y) {
                pdata_bounds[labeldata].y = i;
            }
            if (j < pdata_bounds[labeldata].z) {
                pdata_bounds[labeldata].z = j;
            }
            if (j > pdata_bounds[labeldata].w) {
                pdata_bounds[labeldata].w = j;
            }
        }
    }

    for (auto box: pdata_bounds) {
        vec4 bbox = box.second;
        if (bbox.x == bbox.y || bbox.z == bbox.w) { // filter boxes of zeros size
            continue;
        }
        label_file << object_class_ID << " " << (bbox.x + 0.5 * (bbox.y - bbox.x)) / float(camera_resolution.x) << " " << (bbox.z + 0.5 * (bbox.w - bbox.z)) / float(camera_resolution.y) << " " << std::setprecision(6) << std::fixed
                   << (bbox.y - bbox.x) / float(camera_resolution.x) << " " << (bbox.w - bbox.z) / float(camera_resolution.y) << std::endl;
    }

    label_file.close();
}

void RadiationModel::writeImageBoundingBoxes(const std::string &cameralabel, const std::string &primitive_data_label, const uint &object_class_ID, const std::string &image_file, const std::string &classes_txt_file, const std::string &image_path) {
    writeImageBoundingBoxes(cameralabel, std::vector<std::string>{primitive_data_label}, std::vector<uint>{object_class_ID}, image_file, classes_txt_file, image_path);
}

void RadiationModel::writeImageBoundingBoxes(const std::string &cameralabel, const std::vector<std::string> &primitive_data_label, const std::vector<uint> &object_class_ID, const std::string &image_file, const std::string &classes_txt_file,
                                             const std::string &image_path) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Camera '" + cameralabel + "' does not exist.");
    }

    if (primitive_data_label.size() != object_class_ID.size()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): The lengths of primitive_data_label and object_class_ID vectors must be the same.");
    }

    // Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeImageBoundingBoxes): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::string outfile_txt = output_path + std::filesystem::path(image_file).stem().string() + ".txt";

    std::ofstream label_file(outfile_txt);

    if (!label_file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Could not open output bounding box file '" + outfile_txt + "'.");
    }

    // Map to store bounding boxes for each data label class combination
    std::map<std::pair<uint, uint>, vec4> pdata_bounds; // (class_id, label_value) -> bbox

    // Iterate through all pixels
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1;
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;

            if (context->doesPrimitiveExist(UUID)) {
                // Check each primitive data label
                for (size_t label_idx = 0; label_idx < primitive_data_label.size(); label_idx++) {
                    const std::string &data_label = primitive_data_label[label_idx];
                    uint class_id = object_class_ID[label_idx];

                    if (context->doesPrimitiveDataExist(UUID, data_label.c_str())) {
                        uint labeldata;
                        bool has_data = false;

                        HeliosDataType datatype = context->getPrimitiveDataType(data_label.c_str());
                        if (datatype == HELIOS_TYPE_UINT) {
                            uint labeldata_ui;
                            context->getPrimitiveData(UUID, data_label.c_str(), labeldata_ui);
                            labeldata = labeldata_ui;
                            has_data = true;
                        } else if (datatype == HELIOS_TYPE_INT) {
                            int labeldata_i;
                            context->getPrimitiveData(UUID, data_label.c_str(), labeldata_i);
                            labeldata = (uint) labeldata_i;
                            has_data = true;
                        }

                        if (has_data) {
                            std::pair<uint, uint> key = std::make_pair(class_id, labeldata);

                            if (pdata_bounds.find(key) == pdata_bounds.end()) {
                                pdata_bounds[key] = make_vec4(1e6, -1, 1e6, -1);
                            }

                            if (i < pdata_bounds[key].x) {
                                pdata_bounds[key].x = i;
                            }
                            if (i > pdata_bounds[key].y) {
                                pdata_bounds[key].y = i;
                            }
                            if (j < pdata_bounds[key].z) {
                                pdata_bounds[key].z = j;
                            }
                            if (j > pdata_bounds[key].w) {
                                pdata_bounds[key].w = j;
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto box: pdata_bounds) {
        uint class_id = box.first.first;
        vec4 bbox = box.second;
        if (bbox.x == bbox.y || bbox.z == bbox.w) { // filter boxes of zero size
            continue;
        }
        label_file << class_id << " " << (bbox.x + 0.5 * (bbox.y - bbox.x)) / float(camera_resolution.x) << " " << (bbox.z + 0.5 * (bbox.w - bbox.z)) / float(camera_resolution.y) << " " << std::setprecision(6) << std::fixed
                   << (bbox.y - bbox.x) / float(camera_resolution.x) << " " << (bbox.w - bbox.z) / float(camera_resolution.y) << std::endl;
    }

    label_file.close();

    std::ofstream classes_txt_stream(output_path + classes_txt_file);
    if (!classes_txt_stream.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes): Could not open output classes file '" + output_path + classes_txt_file + ".");
    }
    for (int i = 0; i < object_class_ID.size(); i++) {
        classes_txt_stream << object_class_ID.at(i) << " " << primitive_data_label.at(i) << std::endl;
    }
    classes_txt_stream.close();
}

void RadiationModel::writeImageBoundingBoxes_ObjectData(const std::string &cameralabel, const std::string &object_data_label, const uint &object_class_ID, const std::string &image_file, const std::string &classes_txt_file,
                                                        const std::string &image_path) {
    writeImageBoundingBoxes_ObjectData(cameralabel, std::vector<std::string>{object_data_label}, std::vector<uint>{object_class_ID}, image_file, classes_txt_file, image_path);
}

void RadiationModel::writeImageBoundingBoxes_ObjectData(const std::string &cameralabel, const std::vector<std::string> &object_data_label, const std::vector<uint> &object_class_ID, const std::string &image_file, const std::string &classes_txt_file,
                                                        const std::string &image_path) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Camera '" + cameralabel + "' does not exist.");
    }

    if (object_data_label.size() != object_class_ID.size()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): The lengths of object_data_label and object_class_ID vectors must be the same.");
    }

    // Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!isDirectoryPath(output_path)) {
        helios_runtime_error("ERROR(RadiationModel::writeImageBoundingBoxes_ObjectData): Expected a directory path but got a file path for argument 'image_path'.");
    }

    std::string outfile_txt = output_path + std::filesystem::path(image_file).stem().string() + ".txt";

    std::ofstream label_file(outfile_txt);

    if (!label_file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Could not open output bounding box file '" + outfile_txt + "'.");
    }

    // Map to store bounding boxes for each data label class combination
    std::map<std::pair<uint, uint>, vec4> pdata_bounds; // (class_id, label_value) -> bbox

    // Iterate through all pixels
    // Apply horizontal flip to match mask coordinate system
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1; // horizontal flip
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;

            if (!context->doesPrimitiveExist(UUID)) {
                continue;
            }

            uint objID = context->getPrimitiveParentObjectID(UUID);

            if (!context->doesObjectExist(objID)) {
                continue;
            }

            // Check each object data label
            for (size_t label_idx = 0; label_idx < object_data_label.size(); label_idx++) {
                const std::string &data_label = object_data_label[label_idx];
                uint class_id = object_class_ID[label_idx];

                if (context->doesObjectDataExist(objID, data_label.c_str())) {
                    uint labeldata;
                    bool has_data = false;

                    HeliosDataType datatype = context->getObjectDataType(data_label.c_str());
                    if (datatype == HELIOS_TYPE_UINT) {
                        uint labeldata_ui;
                        context->getObjectData(objID, data_label.c_str(), labeldata_ui);
                        labeldata = labeldata_ui;
                        has_data = true;
                    } else if (datatype == HELIOS_TYPE_INT) {
                        int labeldata_i;
                        context->getObjectData(objID, data_label.c_str(), labeldata_i);
                        labeldata = (uint) labeldata_i;
                        has_data = true;
                    }

                    if (has_data) {
                        std::pair<uint, uint> key = std::make_pair(class_id, labeldata);

                        if (pdata_bounds.find(key) == pdata_bounds.end()) {
                            pdata_bounds[key] = make_vec4(1e6, -1, 1e6, -1);
                        }

                        if (i < pdata_bounds[key].x) {
                            pdata_bounds[key].x = i;
                        }
                        if (i > pdata_bounds[key].y) {
                            pdata_bounds[key].y = i;
                        }
                        if (j < pdata_bounds[key].z) {
                            pdata_bounds[key].z = j;
                        }
                        if (j > pdata_bounds[key].w) {
                            pdata_bounds[key].w = j;
                        }
                    }
                }
            }
        }
    }

    for (auto box: pdata_bounds) {
        uint class_id = box.first.first;
        vec4 bbox = box.second;
        if (bbox.x == bbox.y || bbox.z == bbox.w) { // filter boxes of zero size
            continue;
        }
        label_file << class_id << " " << (bbox.x + 0.5 * (bbox.y - bbox.x)) / float(camera_resolution.x) << " " << (bbox.z + 0.5 * (bbox.w - bbox.z)) / float(camera_resolution.y) << " " << std::setprecision(6) << std::fixed
                   << (bbox.y - bbox.x) / float(camera_resolution.x) << " " << (bbox.w - bbox.z) / float(camera_resolution.y) << std::endl;
    }

    label_file.close();

    std::ofstream classes_txt_stream(output_path + classes_txt_file);
    if (!classes_txt_stream.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageBoundingBoxes_ObjectData): Could not open output classes file '" + output_path + classes_txt_file + ".");
    }
    for (int i = 0; i < object_class_ID.size(); i++) {
        classes_txt_stream << object_class_ID.at(i) << " " << object_data_label.at(i) << std::endl;
    }
    classes_txt_stream.close();
}

// Helper function to initialize or load existing COCO JSON structure and get image ID
std::pair<nlohmann::json, int> RadiationModel::initializeCOCOJsonWithImageId(const std::string &filename, bool append_file, const std::string &cameralabel, const helios::int2 &camera_resolution, const std::string &image_file) {
    nlohmann::json coco_json;
    int image_id = 0;

    if (append_file) {
        std::ifstream existing_file(filename);
        if (existing_file.is_open()) {
            try {
                existing_file >> coco_json;
            } catch (const std::exception &e) {
                coco_json.clear();
            }
            existing_file.close();
        }
    }

    // Initialize JSON structure if empty
    if (coco_json.empty()) {
        coco_json["categories"] = nlohmann::json::array();
        coco_json["images"] = nlohmann::json::array();
        coco_json["annotations"] = nlohmann::json::array();
    }

    // Extract just the filename (no path) from the image file
    std::filesystem::path image_path_obj(image_file);
    std::string filename_only = image_path_obj.filename().string();

    // Check if this image already exists in the JSON
    bool image_exists = false;
    for (const auto &img: coco_json["images"]) {
        if (img["file_name"] == filename_only) {
            image_id = img["id"];
            image_exists = true;
            break;
        }
    }

    // If image doesn't exist, add it with a new unique ID
    if (!image_exists) {
        // Find the next available image ID
        int max_image_id = -1;
        for (const auto &img: coco_json["images"]) {
            if (img["id"] > max_image_id) {
                max_image_id = img["id"];
            }
        }
        image_id = max_image_id + 1;

        // Add the new image entry
        nlohmann::json image_entry;
        image_entry["id"] = image_id;
        image_entry["file_name"] = filename_only;
        image_entry["height"] = camera_resolution.y;
        image_entry["width"] = camera_resolution.x;
        coco_json["images"].push_back(image_entry);
    }

    return std::make_pair(coco_json, image_id);
}

// Helper function to initialize or load existing COCO JSON structure (backward compatibility)
nlohmann::json RadiationModel::initializeCOCOJson(const std::string &filename, bool append_file, const std::string &cameralabel, const helios::int2 &camera_resolution, const std::string &image_file) {
    return initializeCOCOJsonWithImageId(filename, append_file, cameralabel, camera_resolution, image_file).first;
}

// Helper function to add category to COCO JSON if it doesn't exist
void RadiationModel::addCategoryToCOCO(nlohmann::json &coco_json, const std::vector<uint> &object_class_ID, const std::vector<std::string> &category_name) {
    if (object_class_ID.size() != category_name.size()) {
        helios_runtime_error("ERROR (RadiationModel::addCategoryToCOCO): The lengths of object_class_ID and category_name vectors must be the same.");
    }

    for (size_t i = 0; i < object_class_ID.size(); ++i) {
        bool category_exists = false;
        for (auto &cat: coco_json["categories"]) {
            if (cat["id"] == object_class_ID[i]) {
                category_exists = true;
                break;
            }
        }
        if (!category_exists) {
            nlohmann::json category;
            category["id"] = object_class_ID[i];
            category["name"] = category_name[i];
            category["supercategory"] = "none";
            coco_json["categories"].push_back(category);
        }
    }
}

// Helper function to write COCO JSON with proper formatting
void RadiationModel::writeCOCOJson(const nlohmann::json &coco_json, const std::string &filename) {
    std::ofstream json_file(filename);
    if (!json_file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel): Could not open file '" + filename + "'.");
    }

    // Use standard JSON formatting for now (can optimize array formatting later)
    json_file << coco_json.dump(2) << std::endl;
    json_file.close();
}

// Helper function to generate label masks from either primitive or object data
std::map<int, std::vector<std::vector<bool>>> RadiationModel::generateLabelMasks(const std::string &cameralabel, const std::string &data_label, bool use_object_data) {
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    std::vector<uint> pixel_UUIDs = camera_UUIDs;
    int2 camera_resolution = cameras.at(cameralabel).resolution;

    std::map<int, std::vector<std::vector<bool>>> label_masks;

    // First pass: identify all unique labels and create binary masks
    // Apply horizontal flip to match JPEG coordinate system
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1; // horizontal flip to match JPEG
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;

            if (context->doesPrimitiveExist(UUID)) {
                uint labeldata;
                bool has_data = false;

                if (use_object_data) {
                    // Object data version
                    uint objID = context->getPrimitiveParentObjectID(UUID);
                    if (objID != 0 && context->doesObjectDataExist(objID, data_label.c_str())) {
                        HeliosDataType datatype = context->getObjectDataType(data_label.c_str());
                        if (datatype == HELIOS_TYPE_UINT) {
                            uint labeldata_ui;
                            context->getObjectData(objID, data_label.c_str(), labeldata_ui);
                            labeldata = labeldata_ui;
                            has_data = true;
                        } else if (datatype == HELIOS_TYPE_INT) {
                            int labeldata_i;
                            context->getObjectData(objID, data_label.c_str(), labeldata_i);
                            labeldata = (uint) labeldata_i;
                            has_data = true;
                        }
                    }
                } else {
                    // Primitive data version
                    if (context->doesPrimitiveDataExist(UUID, data_label.c_str())) {
                        HeliosDataType datatype = context->getPrimitiveDataType(data_label.c_str());
                        if (datatype == HELIOS_TYPE_UINT) {
                            uint labeldata_ui;
                            context->getPrimitiveData(UUID, data_label.c_str(), labeldata_ui);
                            labeldata = labeldata_ui;
                            has_data = true;
                        } else if (datatype == HELIOS_TYPE_INT) {
                            int labeldata_i;
                            context->getPrimitiveData(UUID, data_label.c_str(), labeldata_i);
                            labeldata = (uint) labeldata_i;
                            has_data = true;
                        }
                    }
                }

                if (has_data) {
                    // Initialize mask for this label if not exists
                    if (label_masks.find(labeldata) == label_masks.end()) {
                        label_masks[labeldata] = std::vector<std::vector<bool>>(camera_resolution.y, std::vector<bool>(camera_resolution.x, false));
                    }
                    label_masks[labeldata][j][i] = true;
                }
            }
        }
    }

    return label_masks;
}

// Helper function to find starting boundary pixel (topmost-leftmost)
std::pair<int, int> RadiationModel::findStartingBoundaryPixel(const std::vector<std::vector<bool>> &mask, const helios::int2 &camera_resolution) {
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            if (mask[j][i]) {
                // Check if this pixel is on the boundary
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0)
                            continue;
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni < 0 || ni >= camera_resolution.x || nj < 0 || nj >= camera_resolution.y || !mask[nj][ni]) {
                            return {i, j}; // Found boundary pixel
                        }
                    }
                }
            }
        }
    }
    return {-1, -1}; // No boundary found
}

// Helper function to trace boundary using Moore neighborhood algorithm
std::vector<std::pair<int, int>> RadiationModel::traceBoundaryMoore(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &camera_resolution) {
    std::vector<std::pair<int, int>> contour;

    // 8-connected neighbors in clockwise order starting from East
    int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

    int x = start_x, y = start_y;
    int dir = 6; // Start looking West (opposite of East)

    do {
        contour.push_back({x, y});

        // Look for next boundary pixel
        int start_dir = (dir + 6) % 8; // Start looking 3 positions counter-clockwise from where we came
        bool found = false;

        for (int i = 0; i < 8; i++) {
            int check_dir = (start_dir + i) % 8;
            int nx = x + dx[check_dir];
            int ny = y + dy[check_dir];

            // Check if this neighbor is inside bounds and inside the mask
            if (nx >= 0 && nx < camera_resolution.x && ny >= 0 && ny < camera_resolution.y && mask[ny][nx]) {
                x = nx;
                y = ny;
                dir = check_dir;
                found = true;
                break;
            }
        }

        if (!found)
            break; // No next boundary pixel found

    } while (!(x == start_x && y == start_y) && contour.size() < camera_resolution.x * camera_resolution.y);

    return contour;
}

// Helper function to trace boundary using simple connected components
std::vector<std::pair<int, int>> RadiationModel::traceBoundarySimple(const std::vector<std::vector<bool>> &mask, int start_x, int start_y, const helios::int2 &camera_resolution) {
    std::vector<std::pair<int, int>> contour;
    std::set<std::pair<int, int>> visited_boundary;

    // Use a simple approach: walk along the boundary
    std::queue<std::pair<int, int>> boundary_queue;
    boundary_queue.push({start_x, start_y});
    visited_boundary.insert({start_x, start_y});

    while (!boundary_queue.empty()) {
        auto [x, y] = boundary_queue.front();
        boundary_queue.pop();
        contour.push_back({x, y});

        // 8-connected neighbors
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                if (di == 0 && dj == 0)
                    continue;
                int nx = x + di;
                int ny = y + dj;

                if (nx >= 0 && nx < camera_resolution.x && ny >= 0 && ny < camera_resolution.y && mask[ny][nx] && visited_boundary.find({nx, ny}) == visited_boundary.end()) {

                    // Check if this pixel is on the boundary
                    bool is_boundary = false;
                    for (int ddi = -1; ddi <= 1; ddi++) {
                        for (int ddj = -1; ddj <= 1; ddj++) {
                            if (ddi == 0 && ddj == 0)
                                continue;
                            int nnx = nx + ddi;
                            int nny = ny + ddj;
                            if (nnx < 0 || nnx >= camera_resolution.x || nny < 0 || nny >= camera_resolution.y || !mask[nny][nnx]) {
                                is_boundary = true;
                                break;
                            }
                        }
                        if (is_boundary)
                            break;
                    }

                    if (is_boundary) {
                        boundary_queue.push({nx, ny});
                        visited_boundary.insert({nx, ny});
                    }
                }
            }
        }
    }

    return contour;
}

// Helper function to generate annotations from label masks
std::vector<std::map<std::string, std::vector<float>>> RadiationModel::generateAnnotationsFromMasks(const std::map<int, std::vector<std::vector<bool>>> &label_masks, uint object_class_ID, const helios::int2 &camera_resolution, int image_id) {
    std::vector<std::map<std::string, std::vector<float>>> annotations;
    int annotation_id = 0;

    for (const auto &label_pair: label_masks) {
        int label_value = label_pair.first;
        const auto &mask = label_pair.second;

        // Create a visited mask for connected components
        std::vector<std::vector<bool>> visited(camera_resolution.y, std::vector<bool>(camera_resolution.x, false));

        // Find all connected components for this label
        for (int j = 0; j < camera_resolution.y; j++) {
            for (int i = 0; i < camera_resolution.x; i++) {
                if (mask[j][i] && !visited[j][i]) {
                    // Find boundary pixel for this component
                    int boundary_i = i, boundary_j = j;
                    bool is_boundary = false;

                    // Check if this pixel is on the boundary
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni < 0 || ni >= camera_resolution.x || nj < 0 || nj >= camera_resolution.y || !mask[nj][ni]) {
                                is_boundary = true;
                                boundary_i = i;
                                boundary_j = j;
                                break;
                            }
                        }
                        if (is_boundary)
                            break;
                    }

                    if (is_boundary) {
                        // First, mark all pixels in this connected component using flood fill
                        std::stack<std::pair<int, int>> stack;
                        std::vector<std::pair<int, int>> component_pixels;
                        stack.push({i, j});
                        visited[j][i] = true;

                        int min_x = i, max_x = i, min_y = j, max_y = j;
                        int area = 0;

                        while (!stack.empty()) {
                            auto [ci, cj] = stack.top();
                            stack.pop();
                            area++;
                            component_pixels.push_back({ci, cj});

                            min_x = std::min(min_x, ci);
                            max_x = std::max(max_x, ci);
                            min_y = std::min(min_y, cj);
                            max_y = std::max(max_y, cj);

                            // Check 4-connected neighbors
                            for (int di = -1; di <= 1; di++) {
                                for (int dj = -1; dj <= 1; dj++) {
                                    if (abs(di) + abs(dj) != 1)
                                        continue; // Only 4-connected
                                    int ni = ci + di;
                                    int nj = cj + dj;
                                    if (ni >= 0 && ni < camera_resolution.x && nj >= 0 && nj < camera_resolution.y && mask[nj][ni] && !visited[nj][ni]) {
                                        stack.push({ni, nj});
                                        visited[nj][ni] = true;
                                    }
                                }
                            }
                        }

                        // Now trace the boundary of this component
                        auto start_pixel = findStartingBoundaryPixel(mask, camera_resolution);
                        bool is_boundary_start = false;

                        if (start_pixel.first >= min_x && start_pixel.first <= max_x && start_pixel.second >= min_y && start_pixel.second <= max_y) {
                            is_boundary_start = true;
                        }

                        if (is_boundary_start) {
                            // Try Moore neighborhood boundary tracing first
                            auto contour = traceBoundaryMoore(mask, start_pixel.first, start_pixel.second, camera_resolution);

                            // If Moore tracing didn't work well, fall back to simple boundary collection
                            if (contour.size() < 10) {
                                contour = traceBoundarySimple(mask, start_pixel.first, start_pixel.second, camera_resolution);
                            }

                            if (contour.size() >= 3) {
                                // Create annotation
                                std::map<std::string, std::vector<float>> annotation;
                                annotation["id"] = {(float) annotation_id++};
                                annotation["image_id"] = {(float) image_id};
                                annotation["category_id"] = {(float) object_class_ID};
                                annotation["bbox"] = {(float) min_x, (float) min_y, (float) (max_x - min_x), (float) (max_y - min_y)};
                                annotation["area"] = {(float) area};
                                annotation["iscrowd"] = {0.0f};

                                // Convert contour to segmentation format (flatten coordinates)
                                std::vector<float> segmentation;
                                for (const auto &point: contour) {
                                    segmentation.push_back((float) point.first); // x coordinate
                                    segmentation.push_back((float) point.second); // y coordinate
                                }
                                annotation["segmentation"] = segmentation;

                                annotations.push_back(annotation);
                            }
                        }
                    }
                }
            }
        }
    }

    return annotations;
}

void RadiationModel::writeImageSegmentationMasks(const std::string &cameralabel, const std::string &primitive_data_label, const uint &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                                 const std::vector<std::string> &data_attribute_labels, bool append_file) {
    writeImageSegmentationMasks(cameralabel, std::vector<std::string>{primitive_data_label}, std::vector<uint>{object_class_ID}, json_filename, image_file, data_attribute_labels, append_file);
}

void RadiationModel::writeImageSegmentationMasks(const std::string &cameralabel, const std::vector<std::string> &primitive_data_label, const std::vector<uint> &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                                 const std::vector<std::string> &data_attribute_labels, bool append_file) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks): Camera '" + cameralabel + "' does not exist.");
    }

    if (primitive_data_label.size() != object_class_ID.size()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks): The lengths of primitive_data_label and object_class_ID vectors must be the same.");
    }

    // Check that camera pixel data exists
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }

    // Check that all primitive data labels exist
    std::vector<std::string> all_primitive_data = context->listAllPrimitiveDataLabels();
    for (const auto &data_label: primitive_data_label) {
        if (std::find(all_primitive_data.begin(), all_primitive_data.end(), data_label) == all_primitive_data.end()) {
            std::cerr << "WARNING (RadiationModel::writeImageSegmentationMasks): Primitive data label '" << data_label << "' does not exist in the context." << std::endl;
        }
    }

    // Check that image file exists
    if (!std::filesystem::exists(image_file)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks): Image file '" + image_file + "' does not exist.");
    }

    // Validate and ensure JSON filename has .json extension
    std::string validated_json_filename = json_filename;
    if (validated_json_filename.length() < 5 || validated_json_filename.substr(validated_json_filename.length() - 5) != ".json") {
        validated_json_filename += ".json";
    }

    // Use the validated filename directly
    std::string outfile = validated_json_filename;

    // Write annotations to JSON file
    int2 camera_resolution = cameras.at(cameralabel).resolution;
    auto coco_json_pair = initializeCOCOJsonWithImageId(outfile, append_file, cameralabel, camera_resolution, image_file);
    nlohmann::json coco_json = coco_json_pair.first;
    int image_id = coco_json_pair.second;
    addCategoryToCOCO(coco_json, object_class_ID, primitive_data_label);

    // Check which data_attribute_labels exist in primitive or object data
    struct AttributeInfo {
        std::string label;
        bool is_primitive_data;
        bool exists;
    };
    std::vector<AttributeInfo> attribute_info;

    if (!data_attribute_labels.empty()) {
        std::vector<std::string> all_primitive_data = context->listAllPrimitiveDataLabels();
        std::vector<std::string> all_object_data = context->listAllObjectDataLabels();

        for (const auto &attr_label: data_attribute_labels) {
            AttributeInfo info;
            info.label = attr_label;
            info.exists = false;

            if (std::find(all_primitive_data.begin(), all_primitive_data.end(), attr_label) != all_primitive_data.end()) {
                info.is_primitive_data = true;
                info.exists = true;
            } else if (std::find(all_object_data.begin(), all_object_data.end(), attr_label) != all_object_data.end()) {
                info.is_primitive_data = false;
                info.exists = true;
            }

            if (info.exists) {
                attribute_info.push_back(info);
            }
        }
    }

    bool use_attributes = !attribute_info.empty();

    // Get pixel UUID data
    std::vector<uint> pixel_UUIDs;
    std::string pixel_UUID_label = "camera_" + cameralabel + "_pixel_UUID";
    context->getGlobalData(pixel_UUID_label.c_str(), pixel_UUIDs);

    // Process each data label and class ID pair
    for (size_t i = 0; i < primitive_data_label.size(); ++i) {
        // Generate label masks using helper function (primitive data version)
        std::map<int, std::vector<std::vector<bool>>> label_masks = generateLabelMasks(cameralabel, primitive_data_label[i], false);

        // Generate annotations from masks using helper function
        std::vector<std::map<std::string, std::vector<float>>> annotations = generateAnnotationsFromMasks(label_masks, object_class_ID[i], camera_resolution, image_id);

        // Calculate mean attribute values for each mask if requested
        std::vector<std::map<std::string, double>> mean_attribute_values_per_component;
        if (use_attributes) {
            // For each label mask, find connected components and calculate mean attribute values
            for (const auto &label_pair: label_masks) {
                const auto &mask = label_pair.second;
                std::vector<std::vector<bool>> visited(camera_resolution.y, std::vector<bool>(camera_resolution.x, false));

                for (int j = 0; j < camera_resolution.y; j++) {
                    for (int i_px = 0; i_px < camera_resolution.x; i_px++) {
                        if (mask[j][i_px] && !visited[j][i_px]) {
                            // Found a new connected component - gather all pixels
                            std::stack<std::pair<int, int>> stack;
                            std::vector<std::pair<int, int>> component_pixels;
                            stack.push({i_px, j});
                            visited[j][i_px] = true;

                            while (!stack.empty()) {
                                auto [ci, cj] = stack.top();
                                stack.pop();
                                component_pixels.push_back({ci, cj});

                                // Check 4-connected neighbors
                                for (int di = -1; di <= 1; di++) {
                                    for (int dj = -1; dj <= 1; dj++) {
                                        if (abs(di) + abs(dj) != 1)
                                            continue;
                                        int ni = ci + di;
                                        int nj = cj + dj;
                                        if (ni >= 0 && ni < camera_resolution.x && nj >= 0 && nj < camera_resolution.y && mask[nj][ni] && !visited[nj][ni]) {
                                            stack.push({ni, nj});
                                            visited[nj][ni] = true;
                                        }
                                    }
                                }
                            }

                            // Calculate mean attribute values for this component (for all attributes)
                            std::map<std::string, double> component_attributes;
                            for (const auto &attr: attribute_info) {
                                double sum = 0.0;
                                int count = 0;

                                for (const auto &[px_i, px_j]: component_pixels) {
                                    uint ii = camera_resolution.x - px_i - 1; // horizontal flip because component_pixels are in mask space
                                    uint UUID = pixel_UUIDs.at(px_j * camera_resolution.x + ii) - 1;

                                    if (context->doesPrimitiveExist(UUID)) {
                                        double value = 0.0;
                                        bool has_value = false;

                                        if (attr.is_primitive_data) {
                                            if (context->doesPrimitiveDataExist(UUID, attr.label.c_str())) {
                                                HeliosDataType datatype = context->getPrimitiveDataType(attr.label.c_str());
                                                if (datatype == HELIOS_TYPE_INT) {
                                                    int val;
                                                    context->getPrimitiveData(UUID, attr.label.c_str(), val);
                                                    value = static_cast<double>(val);
                                                    has_value = true;
                                                } else if (datatype == HELIOS_TYPE_UINT) {
                                                    uint val;
                                                    context->getPrimitiveData(UUID, attr.label.c_str(), val);
                                                    value = static_cast<double>(val);
                                                    has_value = true;
                                                } else if (datatype == HELIOS_TYPE_FLOAT) {
                                                    float val;
                                                    context->getPrimitiveData(UUID, attr.label.c_str(), val);
                                                    value = static_cast<double>(val);
                                                    has_value = true;
                                                } else if (datatype == HELIOS_TYPE_DOUBLE) {
                                                    context->getPrimitiveData(UUID, attr.label.c_str(), value);
                                                    has_value = true;
                                                }
                                            }
                                        } else {
                                            uint objID = context->getPrimitiveParentObjectID(UUID);
                                            if (objID != 0 && context->doesObjectDataExist(objID, attr.label.c_str())) {
                                                HeliosDataType datatype = context->getObjectDataType(attr.label.c_str());
                                                if (datatype == HELIOS_TYPE_INT) {
                                                    int val;
                                                    context->getObjectData(objID, attr.label.c_str(), val);
                                                    value = static_cast<double>(val);
                                                    has_value = true;
                                                } else if (datatype == HELIOS_TYPE_UINT) {
                                                    uint val;
                                                    context->getObjectData(objID, attr.label.c_str(), val);
                                                    value = static_cast<double>(val);
                                                    has_value = true;
                                                } else if (datatype == HELIOS_TYPE_FLOAT) {
                                                    float val;
                                                    context->getObjectData(objID, attr.label.c_str(), val);
                                                    value = static_cast<double>(val);
                                                    has_value = true;
                                                } else if (datatype == HELIOS_TYPE_DOUBLE) {
                                                    context->getObjectData(objID, attr.label.c_str(), value);
                                                    has_value = true;
                                                }
                                            }
                                        }

                                        if (has_value) {
                                            sum += value;
                                            count++;
                                        }
                                    }
                                }

                                if (count > 0) {
                                    component_attributes[attr.label] = sum / count;
                                } else {
                                    component_attributes[attr.label] = 0.0; // Default if no valid data
                                }
                            }

                            mean_attribute_values_per_component.push_back(component_attributes);
                        }
                    }
                }
            }
        }

        // Find the highest existing annotation ID to avoid conflicts
        int max_annotation_id = -1;
        for (const auto &existing_ann: coco_json["annotations"]) {
            if (existing_ann["id"] > max_annotation_id) {
                max_annotation_id = existing_ann["id"];
            }
        }

        // Add new annotations for this data label
        size_t ann_idx = 0;
        for (const auto &ann: annotations) {
            nlohmann::json json_annotation;
            json_annotation["id"] = max_annotation_id + 1;
            json_annotation["image_id"] = (int) ann.at("image_id")[0];
            json_annotation["category_id"] = (int) ann.at("category_id")[0];

            const auto &bbox = ann.at("bbox");
            json_annotation["bbox"] = {(int) bbox[0], (int) bbox[1], (int) bbox[2], (int) bbox[3]};
            json_annotation["area"] = (int) ann.at("area")[0];

            const auto &seg = ann.at("segmentation");
            std::vector<int> segmentation_coords;
            for (float coord: seg) {
                segmentation_coords.push_back((int) coord);
            }
            json_annotation["segmentation"] = {segmentation_coords};
            json_annotation["iscrowd"] = (int) ann.at("iscrowd")[0];

            // Add attributes if requested
            if (use_attributes && ann_idx < mean_attribute_values_per_component.size()) {
                json_annotation["attributes"] = mean_attribute_values_per_component[ann_idx];
            }

            coco_json["annotations"].push_back(json_annotation);
            max_annotation_id++;
            ann_idx++;
        }
    }

    // Write JSON to file
    writeCOCOJson(coco_json, outfile);
}

void RadiationModel::writeImageSegmentationMasks_ObjectData(const std::string &cameralabel, const std::string &object_data_label, const uint &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                                            const std::vector<std::string> &data_attribute_labels, bool append_file) {
    writeImageSegmentationMasks_ObjectData(cameralabel, std::vector<std::string>{object_data_label}, std::vector<uint>{object_class_ID}, json_filename, image_file, data_attribute_labels, append_file);
}

void RadiationModel::writeImageSegmentationMasks_ObjectData(const std::string &cameralabel, const std::vector<std::string> &object_data_label, const std::vector<uint> &object_class_ID, const std::string &json_filename, const std::string &image_file,
                                                            const std::vector<std::string> &data_attribute_labels, bool append_file) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks_ObjectData): Camera '" + cameralabel + "' does not exist.");
    }

    if (object_data_label.size() != object_class_ID.size()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks_ObjectData): The lengths of object_data_label and object_class_ID vectors must be the same.");
    }

    // Check that camera pixel data exists
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks_ObjectData): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }

    // Check that all object data labels exist
    std::vector<std::string> all_object_data = context->listAllObjectDataLabels();
    for (const auto &data_label: object_data_label) {
        if (std::find(all_object_data.begin(), all_object_data.end(), data_label) == all_object_data.end()) {
            std::cerr << "WARNING (RadiationModel::writeImageSegmentationMasks_ObjectData): Object data label '" << data_label << "' does not exist in the context." << std::endl;
        }
    }

    // Check that image file exists
    if (!std::filesystem::exists(image_file)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks_ObjectData): Image file '" + image_file + "' does not exist.");
    }

    // Validate and ensure JSON filename has .json extension
    std::string validated_json_filename = json_filename;
    if (validated_json_filename.length() < 5 || validated_json_filename.substr(validated_json_filename.length() - 5) != ".json") {
        validated_json_filename += ".json";
    }

    // Use the validated filename directly
    std::string outfile = validated_json_filename;

    // Write annotations to JSON file
    int2 camera_resolution = cameras.at(cameralabel).resolution;
    auto coco_json_pair = initializeCOCOJsonWithImageId(outfile, append_file, cameralabel, camera_resolution, image_file);
    nlohmann::json coco_json = coco_json_pair.first;
    int image_id = coco_json_pair.second;
    addCategoryToCOCO(coco_json, object_class_ID, object_data_label);

    // Check which data_attribute_labels exist in primitive or object data
    struct AttributeInfo {
        std::string label;
        bool is_primitive_data;
        bool exists;
    };
    std::vector<AttributeInfo> attribute_info;

    if (!data_attribute_labels.empty()) {
        std::vector<std::string> all_primitive_data = context->listAllPrimitiveDataLabels();
        std::vector<std::string> all_object_data = context->listAllObjectDataLabels();

        for (const auto &attr_label: data_attribute_labels) {
            AttributeInfo info;
            info.label = attr_label;
            info.exists = false;

            if (std::find(all_primitive_data.begin(), all_primitive_data.end(), attr_label) != all_primitive_data.end()) {
                info.is_primitive_data = true;
                info.exists = true;
            } else if (std::find(all_object_data.begin(), all_object_data.end(), attr_label) != all_object_data.end()) {
                info.is_primitive_data = false;
                info.exists = true;
            }

            if (info.exists) {
                attribute_info.push_back(info);
            }
        }
    }

    bool use_attributes = !attribute_info.empty();

    // Get pixel UUID data
    std::vector<uint> pixel_UUIDs;
    std::string pixel_UUID_label = "camera_" + cameralabel + "_pixel_UUID";
    context->getGlobalData(pixel_UUID_label.c_str(), pixel_UUIDs);

    // Process each data label and class ID pair
    for (size_t i = 0; i < object_data_label.size(); ++i) {
        // Generate label masks using helper function (object data version)
        std::map<int, std::vector<std::vector<bool>>> label_masks = generateLabelMasks(cameralabel, object_data_label[i], true);

        // Find the highest existing annotation ID to avoid conflicts
        int max_annotation_id = -1;
        for (const auto &existing_ann: coco_json["annotations"]) {
            if (existing_ann["id"] > max_annotation_id) {
                max_annotation_id = existing_ann["id"];
            }
        }

        // Generate annotations from masks and calculate attributes together
        // This ensures 1:1 correspondence between annotations and their attributes
        for (const auto &label_pair: label_masks) {
            const auto &mask = label_pair.second;

            // Create a visited mask for connected components
            std::vector<std::vector<bool>> visited(camera_resolution.y, std::vector<bool>(camera_resolution.x, false));

            // Find all connected components for this label
            for (int j = 0; j < camera_resolution.y; j++) {
                for (int i_px = 0; i_px < camera_resolution.x; i_px++) {
                    if (mask[j][i_px] && !visited[j][i_px]) {
                        // Find boundary pixel for this component
                        int boundary_i = i_px, boundary_j = j;
                        bool is_boundary = false;

                        // Check if this pixel is on the boundary
                        for (int di = -1; di <= 1; di++) {
                            for (int dj = -1; dj <= 1; dj++) {
                                int ni = i_px + di;
                                int nj = j + dj;
                                if (ni < 0 || ni >= camera_resolution.x || nj < 0 || nj >= camera_resolution.y || !mask[nj][ni]) {
                                    is_boundary = true;
                                    boundary_i = i_px;
                                    boundary_j = j;
                                    break;
                                }
                            }
                            if (is_boundary)
                                break;
                        }

                        if (is_boundary) {
                            // First, mark all pixels in this connected component using flood fill
                            std::stack<std::pair<int, int>> stack;
                            std::vector<std::pair<int, int>> component_pixels;
                            stack.push({i_px, j});
                            visited[j][i_px] = true;

                            int min_x = i_px, max_x = i_px, min_y = j, max_y = j;
                            int area = 0;

                            while (!stack.empty()) {
                                auto [ci, cj] = stack.top();
                                stack.pop();
                                area++;
                                component_pixels.push_back({ci, cj});

                                min_x = std::min(min_x, ci);
                                max_x = std::max(max_x, ci);
                                min_y = std::min(min_y, cj);
                                max_y = std::max(max_y, cj);

                                // Check 4-connected neighbors
                                for (int di = -1; di <= 1; di++) {
                                    for (int dj = -1; dj <= 1; dj++) {
                                        if (abs(di) + abs(dj) != 1)
                                            continue; // Only 4-connected
                                        int ni = ci + di;
                                        int nj = cj + dj;
                                        if (ni >= 0 && ni < camera_resolution.x && nj >= 0 && nj < camera_resolution.y && mask[nj][ni] && !visited[nj][ni]) {
                                            stack.push({ni, nj});
                                            visited[nj][ni] = true;
                                        }
                                    }
                                }
                            }

                            // Now trace the boundary of this component
                            auto start_pixel = findStartingBoundaryPixel(mask, camera_resolution);
                            bool is_boundary_start = false;

                            if (start_pixel.first >= min_x && start_pixel.first <= max_x && start_pixel.second >= min_y && start_pixel.second <= max_y) {
                                is_boundary_start = true;
                            }

                            if (is_boundary_start) {
                                // Try Moore neighborhood boundary tracing first
                                auto contour = traceBoundaryMoore(mask, start_pixel.first, start_pixel.second, camera_resolution);

                                // If Moore tracing didn't work well, fall back to simple boundary collection
                                if (contour.size() < 10) {
                                    contour = traceBoundarySimple(mask, start_pixel.first, start_pixel.second, camera_resolution);
                                }

                                if (contour.size() >= 3) {
                                    // Calculate mean attribute values for this component (for all attributes)
                                    std::map<std::string, double> component_attributes;
                                    if (use_attributes) {
                                        for (const auto &attr: attribute_info) {
                                            double sum = 0.0;
                                            int count = 0;

                                            for (const auto &[px_i, px_j]: component_pixels) {
                                                uint ii = camera_resolution.x - px_i - 1;
                                                uint UUID = pixel_UUIDs.at(px_j * camera_resolution.x + ii) - 1;

                                                if (context->doesPrimitiveExist(UUID)) {
                                                    double value = 0.0;
                                                    bool has_value = false;

                                                    if (attr.is_primitive_data) {
                                                        if (context->doesPrimitiveDataExist(UUID, attr.label.c_str())) {
                                                            HeliosDataType datatype = context->getPrimitiveDataType(attr.label.c_str());
                                                            if (datatype == HELIOS_TYPE_INT) {
                                                                int val;
                                                                context->getPrimitiveData(UUID, attr.label.c_str(), val);
                                                                value = static_cast<double>(val);
                                                                has_value = true;
                                                            } else if (datatype == HELIOS_TYPE_UINT) {
                                                                uint val;
                                                                context->getPrimitiveData(UUID, attr.label.c_str(), val);
                                                                value = static_cast<double>(val);
                                                                has_value = true;
                                                            } else if (datatype == HELIOS_TYPE_FLOAT) {
                                                                float val;
                                                                context->getPrimitiveData(UUID, attr.label.c_str(), val);
                                                                value = static_cast<double>(val);
                                                                has_value = true;
                                                            } else if (datatype == HELIOS_TYPE_DOUBLE) {
                                                                context->getPrimitiveData(UUID, attr.label.c_str(), value);
                                                                has_value = true;
                                                            }
                                                        }
                                                    } else {
                                                        uint objID = context->getPrimitiveParentObjectID(UUID);
                                                        if (objID != 0 && context->doesObjectDataExist(objID, attr.label.c_str())) {
                                                            HeliosDataType datatype = context->getObjectDataType(attr.label.c_str());
                                                            if (datatype == HELIOS_TYPE_INT) {
                                                                int val;
                                                                context->getObjectData(objID, attr.label.c_str(), val);
                                                                value = static_cast<double>(val);
                                                                has_value = true;
                                                            } else if (datatype == HELIOS_TYPE_UINT) {
                                                                uint val;
                                                                context->getObjectData(objID, attr.label.c_str(), val);
                                                                value = static_cast<double>(val);
                                                                has_value = true;
                                                            } else if (datatype == HELIOS_TYPE_FLOAT) {
                                                                float val;
                                                                context->getObjectData(objID, attr.label.c_str(), val);
                                                                value = static_cast<double>(val);
                                                                has_value = true;
                                                            } else if (datatype == HELIOS_TYPE_DOUBLE) {
                                                                context->getObjectData(objID, attr.label.c_str(), value);
                                                                has_value = true;
                                                            }
                                                        }
                                                    }

                                                    if (has_value) {
                                                        sum += value;
                                                        count++;
                                                    }
                                                }
                                            }

                                            if (count > 0) {
                                                component_attributes[attr.label] = sum / count;
                                            } else {
                                                component_attributes[attr.label] = 0.0; // Default if no valid data
                                            }
                                        }
                                    }

                                    // Create annotation with attributes
                                    nlohmann::json json_annotation;
                                    json_annotation["id"] = max_annotation_id + 1;
                                    json_annotation["image_id"] = image_id;
                                    json_annotation["category_id"] = (int) object_class_ID[i];
                                    json_annotation["bbox"] = {min_x, min_y, max_x - min_x, max_y - min_y};
                                    json_annotation["area"] = area;
                                    json_annotation["iscrowd"] = 0;

                                    // Convert contour to segmentation format (flatten coordinates)
                                    std::vector<int> segmentation_coords;
                                    for (const auto &point: contour) {
                                        segmentation_coords.push_back(point.first); // x coordinate
                                        segmentation_coords.push_back(point.second); // y coordinate
                                    }
                                    json_annotation["segmentation"] = {segmentation_coords};

                                    // Add attributes if requested
                                    if (use_attributes) {
                                        json_annotation["attributes"] = component_attributes;
                                    }

                                    coco_json["annotations"].push_back(json_annotation);
                                    max_annotation_id++;
                                }
                            }
                        } else {
                            // Mark all pixels in this non-boundary component as visited
                            std::stack<std::pair<int, int>> stack;
                            stack.push({i_px, j});
                            visited[j][i_px] = true;

                            while (!stack.empty()) {
                                auto [ci, cj] = stack.top();
                                stack.pop();

                                // Check 4-connected neighbors
                                for (int di = -1; di <= 1; di++) {
                                    for (int dj = -1; dj <= 1; dj++) {
                                        if (abs(di) + abs(dj) != 1)
                                            continue;
                                        int ni = ci + di;
                                        int nj = cj + dj;
                                        if (ni >= 0 && ni < camera_resolution.x && nj >= 0 && nj < camera_resolution.y && mask[nj][ni] && !visited[nj][ni]) {
                                            stack.push({ni, nj});
                                            visited[nj][ni] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Write JSON to file
    writeCOCOJson(coco_json, outfile);
}

void RadiationModel::setPadValue(const std::string &cameralabel, const std::vector<std::string> &bandlabels, const std::vector<float> &padvalues) {
    for (uint b = 0; b < bandlabels.size(); b++) {
        std::string bandlabel = bandlabels.at(b);

        std::string image_value_label = "camera_" + cameralabel + "_" + bandlabel;
        std::vector<float> cameradata;
        context->getGlobalData(image_value_label.c_str(), cameradata);

        std::vector<uint> camera_UUIDs;
        std::string image_UUID_label = "camera_" + cameralabel + "_pixel_UUID";
        context->getGlobalData(image_UUID_label.c_str(), camera_UUIDs);

        for (uint i = 0; i < cameradata.size(); i++) {
            uint UUID = camera_UUIDs.at(i) - 1;
            if (!context->doesPrimitiveExist(UUID)) {
                cameradata.at(i) = padvalues.at(b);
            }
        }
        context->setGlobalData(image_value_label.c_str(), cameradata);
    }
}

void RadiationModel::calibrateCamera(const std::string &originalcameralabel, const std::vector<std::string> &sourcelabels, const std::vector<std::string> &cameraresplabels_raw, const std::vector<std::string> &bandlabels, const float scalefactor,
                                     const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark) {

    if (cameras.find(originalcameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): Camera " + originalcameralabel + " does not exist.");
    } else if (radiation_sources.empty()) {
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): No radiation sources were added to the radiation model. Cannot perform calibration.");
    }

    CameraCalibration cameracalibration_(context);
    if (!calibration_flag) {
        std::cout << "No color board added, use default color calibration." << std::endl;
        cameracalibration = &cameracalibration_;
        vec3 centrelocation = make_vec3(0, 0, 0.2); // Location of color board
        vec3 rotationrad = make_vec3(0, 0, 1.5705); // Rotation angle of color board
        cameracalibration->addDefaultColorboard(centrelocation, 0.1, rotationrad);
    }
    vec2 wavelengthrange = make_vec2(-10000, 10000);

    // Calibrated camera response labels
    std::vector<std::string> cameraresplabels_cal(cameraresplabels_raw.size());

    for (int iband = 0; iband < bandlabels.size(); iband++) {
        cameraresplabels_cal.at(iband) = calibratedmark + "_" + cameraresplabels_raw.at(iband);
    }

    RadiationModel::runRadiationImaging(originalcameralabel, sourcelabels, bandlabels, cameraresplabels_raw, wavelengthrange, 1, 0);
    // Update camera responses
    RadiationModel::updateCameraResponse(originalcameralabel, sourcelabels, cameraresplabels_raw, wavelengthrange, truevalues, calibratedmark);

    float camerascale = RadiationModel::getCameraResponseScale(originalcameralabel, cameraresplabels_cal, bandlabels, sourcelabels, wavelengthrange, truevalues);

    std::cout << "Camera response scale: " << camerascale << std::endl;
    // Scale and write calibrated camera responses
    cameracalibration->writeCalibratedCameraResponses(cameraresplabels_raw, calibratedmark, camerascale * scalefactor);
}

void RadiationModel::calibrateCamera(const std::string &originalcameralabel, const float scalefactor, const std::vector<std::vector<float>> &truevalues, const std::string &calibratedmark) {

    if (cameras.find(originalcameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): Camera " + originalcameralabel + " does not exist.");
    } else if (radiation_sources.empty()) {
        helios_runtime_error("ERROR (RadiationModel::calibrateCamera): No radiation sources were added to the radiation model. Cannot perform calibration.");
    }

    CameraCalibration cameracalibration_(context);
    if (!calibration_flag) {
        std::cout << "No color board added, use default color calibration." << std::endl;
        vec3 centrelocation = make_vec3(0, 0, 0.2); // Location of color board
        vec3 rotationrad = make_vec3(0, 0, 1.5705); // Rotation angle of color board
        cameracalibration_.addDefaultColorboard(centrelocation, 0.1, rotationrad);
        RadiationModel::setCameraCalibration(&cameracalibration_);
    }

    vec2 wavelengthrange = make_vec2(-10000, 10000);

    std::vector<std::string> bandlabels = cameras.at(originalcameralabel).band_labels;

    // Get camera response spectra labels from camera
    std::vector<std::string> cameraresplabels_cal(cameras.at(originalcameralabel).band_spectral_response.size());
    std::vector<std::string> cameraresplabels_raw = cameraresplabels_cal;

    int iband = 0;
    for (auto &band: cameras.at(originalcameralabel).band_spectral_response) {
        cameraresplabels_raw.at(iband) = band.second;
        cameraresplabels_cal.at(iband) = calibratedmark + "_" + band.second;
        iband++;
    }

    // Get labels of radiation sources from camera
    std::vector<std::string> sourcelabels(radiation_sources.size());
    int isource = 0;
    for (auto &source: radiation_sources) {
        if (source.source_spectrum.empty()) {
            helios_runtime_error("ERROR (RadiationModel::calibrateCamera): A spectral distribution was not specified for source " + source.source_spectrum_label + ". Cannot perform camera calibration.");
        }
        sourcelabels.at(isource) = source.source_spectrum_label;
        isource++;
    }

    RadiationModel::updateGeometry();
    RadiationModel::runBand(bandlabels);
    // Update camera responses
    RadiationModel::updateCameraResponse(originalcameralabel, sourcelabels, cameraresplabels_raw, wavelengthrange, truevalues, calibratedmark);

    float camerascale = RadiationModel::getCameraResponseScale(originalcameralabel, cameraresplabels_cal, bandlabels, sourcelabels, wavelengthrange, truevalues);

    std::cout << "Camera response scale: " << camerascale << std::endl;
    // Scale and write calibrated camera responses
    cameracalibration->writeCalibratedCameraResponses(cameraresplabels_raw, calibratedmark, camerascale * scalefactor);
}

std::vector<helios::vec2> RadiationModel::generateGaussianCameraResponse(float FWHM, float mu, float centrawavelength, const helios::int2 &wavebandrange) {

    // Convert FWHM to sigma
    float sigma = FWHM / (2 * std::sqrt(2 * std::log(2)));

    size_t lenspectra = wavebandrange.y - wavebandrange.x;
    std::vector<helios::vec2> cameraresponse(lenspectra);


    for (int i = 0; i < lenspectra; ++i) {
        cameraresponse.at(i).x = float(wavebandrange.x + i);
    }

    // Gaussian function
    for (size_t i = 0; i < lenspectra; ++i) {
        cameraresponse.at(i).y = centrawavelength * std::exp(-std::pow((cameraresponse.at(i).x - mu), 2) / (2 * std::pow(sigma, 2)));
    }


    return cameraresponse;
}

void RadiationModel::applyCameraImageCorrections(const std::string &cameralabel, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation_adjustment, float brightness_adjustment,
                                                 float contrast_adjustment) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraImageCorrections): Camera '" + cameralabel + "' does not exist.");
    }
    RadiationCamera &camera = cameras.at(cameralabel);
    if (camera.pixel_data.find(red_band_label) == camera.pixel_data.end() || camera.pixel_data.find(green_band_label) == camera.pixel_data.end() || camera.pixel_data.find(blue_band_label) == camera.pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyCameraImageCorrections): One or more specified band labels do not exist for the camera pixel data.");
    }

    // Store parameters for metadata output
    if (camera_metadata.find(cameralabel) == camera_metadata.end()) {
        camera_metadata[cameralabel] = CameraMetadata();
    }
    camera_metadata[cameralabel].image_processing.saturation_adjustment = saturation_adjustment;
    camera_metadata[cameralabel].image_processing.brightness_adjustment = brightness_adjustment;
    camera_metadata[cameralabel].image_processing.contrast_adjustment = contrast_adjustment;

    // NOTE: Auto-exposure is now automatically applied during rendering based on camera exposure setting
    // NOTE: White balance is now automatically applied during rendering based on camera white_balance setting
    // NOTE: sRGB gamma compression is now applied during image export in writeCameraImage()

    // Step 0: Apply lens flare effect if enabled (before other adjustments)
    if (camera.lens_flare_enabled) {
        LensFlare lens_flare(camera.lens_flare_properties, camera.resolution);
        lens_flare.apply(camera.pixel_data, camera.resolution);
    }

    // Step 1: Brightness and contrast adjustments in linear space
    if (brightness_adjustment != 1.f || contrast_adjustment != 1.f) {
        camera.adjustBrightnessContrast(red_band_label, green_band_label, blue_band_label, brightness_adjustment, contrast_adjustment);
    }

    // Step 2: Saturation adjustment
    if (saturation_adjustment != 1.f) {
        camera.adjustSaturation(red_band_label, green_band_label, blue_band_label, saturation_adjustment);
    }
}

void RadiationModel::applyImageProcessingPipeline(const std::string &cameralabel, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation_adjustment, float brightness_adjustment,
                                                  float contrast_adjustment, float gain_adjustment) {
    applyCameraImageCorrections(cameralabel, red_band_label, green_band_label, blue_band_label, saturation_adjustment, brightness_adjustment, contrast_adjustment);
}

void RadiationCamera::normalizePixels() {

    float min_P = (std::numeric_limits<float>::max)();
    float max_P = 0.0f;
    for (const auto &[channel_label, data]: pixel_data) {
        for (float v: data) {
            if (v < min_P) {
                min_P = v;
            }
            if (v > max_P) {
                max_P = v;
            }
        }
    }

    for (auto &[channel_label, data]: pixel_data) {
        for (float &v: data) {
            v = (v - min_P) / (max_P - min_P); // Normalize to [0, 1]
        }
    }
}

void RadiationCamera::whiteBalance(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float p) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::whiteBalance): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    const std::size_t N = data_red.size();
    if (data_green.size() != N || data_blue.size() != N) {
        throw std::invalid_argument("All channels must have the same length");
    }
    if (p < 1.0f) {
        throw std::invalid_argument("Minkowski exponent p must satisfy p >= 1");
    }

    // Compute Minkowski means:
    // \[ M_R = \Bigl(\frac{1}{N}\sum_{i=1}^{N}R_i^p\Bigr)^{1/p},\quad
    //    M_G = \Bigl(\frac{1}{N}\sum_{i=1}^{N}G_i^p\Bigr)^{1/p},\quad
    //    M_B = \Bigl(\frac{1}{N}\sum_{i=1}^{N}B_i^p\Bigr)^{1/p} \]
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
        acc_r += std::pow(data_red[i], p);
        acc_g += std::pow(data_green[i], p);
        acc_b += std::pow(data_blue[i], p);
    }
    float mean_r_p = acc_r / static_cast<float>(N);
    float mean_g_p = acc_g / static_cast<float>(N);
    float mean_b_p = acc_b / static_cast<float>(N);

    float M_R = std::pow(mean_r_p, 1.0f / p);
    float M_G = std::pow(mean_g_p, 1.0f / p);
    float M_B = std::pow(mean_b_p, 1.0f / p);

    // Avoid division by zero
    const float eps = 1e-6f;
    if (M_R < eps || M_G < eps || M_B < eps) {
        throw std::runtime_error("Channel Minkowski mean too small");
    }

    // Compute gray reference:
    // \[ M = \frac{M_R + M_G + M_B}{3} \]
    float M = (M_R + M_G + M_B) / 3.0f;

    // Derive per-channel gains:
    // \[ s_R = M / M_R,\quad s_G = M / M_G,\quad s_B = M / M_B \]
    helios::vec3 scale;
    scale.x = M / M_R;
    scale.y = M / M_G;
    scale.z = M / M_B;

    // Apply gains to each pixel:
    // \[ R'_i = s_R\,R_i,\quad G'_i = s_G\,G_i,\quad B'_i = s_B\,B_i \]
    for (std::size_t i = 0; i < N; ++i) {
        data_red[i] *= scale.x;
        data_green[i] *= scale.y;
        data_blue[i] *= scale.z;
    }
}

void RadiationCamera::whiteBalanceGrayEdge(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, int derivative_order, float p) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::whiteBalanceGrayEdge): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    const int width = resolution.x;
    const int height = resolution.y;
    const std::size_t N = width * height;

    if (p < 1.0f) {
        throw std::invalid_argument("Minkowski exponent p must satisfy p >= 1");
    }
    if (derivative_order < 1 || derivative_order > 2) {
        throw std::invalid_argument("Derivative order must be 1 or 2");
    }

    // Compute derivatives using simple finite differences
    std::vector<float> deriv_red(N, 0.0f);
    std::vector<float> deriv_green(N, 0.0f);
    std::vector<float> deriv_blue(N, 0.0f);

    if (derivative_order == 1) {
        // First-order derivatives (gradient magnitude)
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;

                // Sobel operator for gradient estimation
                float dx_r = (data_red[(y - 1) * width + (x + 1)] + 2 * data_red[y * width + (x + 1)] + data_red[(y + 1) * width + (x + 1)]) -
                             (data_red[(y - 1) * width + (x - 1)] + 2 * data_red[y * width + (x - 1)] + data_red[(y + 1) * width + (x - 1)]) / 8.0f;
                float dy_r = (data_red[(y + 1) * width + (x - 1)] + 2 * data_red[(y + 1) * width + x] + data_red[(y + 1) * width + (x + 1)]) -
                             (data_red[(y - 1) * width + (x - 1)] + 2 * data_red[(y - 1) * width + x] + data_red[(y - 1) * width + (x + 1)]) / 8.0f;
                deriv_red[idx] = std::sqrt(dx_r * dx_r + dy_r * dy_r);

                float dx_g = (data_green[(y - 1) * width + (x + 1)] + 2 * data_green[y * width + (x + 1)] + data_green[(y + 1) * width + (x + 1)]) -
                             (data_green[(y - 1) * width + (x - 1)] + 2 * data_green[y * width + (x - 1)] + data_green[(y + 1) * width + (x - 1)]) / 8.0f;
                float dy_g = (data_green[(y + 1) * width + (x - 1)] + 2 * data_green[(y + 1) * width + x] + data_green[(y + 1) * width + (x + 1)]) -
                             (data_green[(y - 1) * width + (x - 1)] + 2 * data_green[(y - 1) * width + x] + data_green[(y - 1) * width + (x + 1)]) / 8.0f;
                deriv_green[idx] = std::sqrt(dx_g * dx_g + dy_g * dy_g);

                float dx_b = (data_blue[(y - 1) * width + (x + 1)] + 2 * data_blue[y * width + (x + 1)] + data_blue[(y + 1) * width + (x + 1)]) -
                             (data_blue[(y - 1) * width + (x - 1)] + 2 * data_blue[y * width + (x - 1)] + data_blue[(y + 1) * width + (x - 1)]) / 8.0f;
                float dy_b = (data_blue[(y + 1) * width + (x - 1)] + 2 * data_blue[(y + 1) * width + x] + data_blue[(y + 1) * width + (x + 1)]) -
                             (data_blue[(y - 1) * width + (x - 1)] + 2 * data_blue[(y - 1) * width + x] + data_blue[(y - 1) * width + (x + 1)]) / 8.0f;
                deriv_blue[idx] = std::sqrt(dx_b * dx_b + dy_b * dy_b);
            }
        }
    } else {
        // Second-order derivatives (Laplacian)
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;

                deriv_red[idx] = std::abs(data_red[(y - 1) * width + x] + data_red[(y + 1) * width + x] + data_red[y * width + (x - 1)] + data_red[y * width + (x + 1)] - 4 * data_red[idx]);

                deriv_green[idx] = std::abs(data_green[(y - 1) * width + x] + data_green[(y + 1) * width + x] + data_green[y * width + (x - 1)] + data_green[y * width + (x + 1)] - 4 * data_green[idx]);

                deriv_blue[idx] = std::abs(data_blue[(y - 1) * width + x] + data_blue[(y + 1) * width + x] + data_blue[y * width + (x - 1)] + data_blue[y * width + (x + 1)] - 4 * data_blue[idx]);
            }
        }
    }

    // Compute Minkowski means of derivatives
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
    int valid_pixels = 0;

    for (std::size_t i = 0; i < N; ++i) {
        if (deriv_red[i] > 0 || deriv_green[i] > 0 || deriv_blue[i] > 0) {
            acc_r += std::pow(deriv_red[i], p);
            acc_g += std::pow(deriv_green[i], p);
            acc_b += std::pow(deriv_blue[i], p);
            valid_pixels++;
        }
    }

    if (valid_pixels == 0) {
        // No edges detected, fall back to standard white balance
        whiteBalance(red_band_label, green_band_label, blue_band_label, p);
        return;
    }

    float mean_r_p = acc_r / static_cast<float>(valid_pixels);
    float mean_g_p = acc_g / static_cast<float>(valid_pixels);
    float mean_b_p = acc_b / static_cast<float>(valid_pixels);

    float M_R = std::pow(mean_r_p, 1.0f / p);
    float M_G = std::pow(mean_g_p, 1.0f / p);
    float M_B = std::pow(mean_b_p, 1.0f / p);

    // Avoid division by zero
    const float eps = 1e-6f;
    if (M_R < eps || M_G < eps || M_B < eps) {
        // Fall back to standard white balance
        whiteBalance(red_band_label, green_band_label, blue_band_label, p);
        return;
    }

    // Compute gray reference
    float M = (M_R + M_G + M_B) / 3.0f;

    // Derive per-channel gains
    helios::vec3 scale;
    scale.x = M / M_R;
    scale.y = M / M_G;
    scale.z = M / M_B;

    // Apply gains to each pixel
    for (std::size_t i = 0; i < N; ++i) {
        data_red[i] *= scale.x;
        data_green[i] *= scale.y;
        data_blue[i] *= scale.z;
    }
}

void RadiationCamera::whiteBalanceWhitePatch(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float percentile) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::whiteBalanceWhitePatch): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    if (percentile <= 0.0f || percentile > 1.0f) {
        throw std::invalid_argument("Percentile must be in range (0, 1]");
    }

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    const std::size_t N = data_red.size();

    // Find the percentile values for each channel
    std::vector<float> sorted_red = data_red;
    std::vector<float> sorted_green = data_green;
    std::vector<float> sorted_blue = data_blue;

    std::size_t k = static_cast<std::size_t>(percentile * (N - 1));

    std::nth_element(sorted_red.begin(), sorted_red.begin() + k, sorted_red.end());
    std::nth_element(sorted_green.begin(), sorted_green.begin() + k, sorted_green.end());
    std::nth_element(sorted_blue.begin(), sorted_blue.begin() + k, sorted_blue.end());

    float white_r = sorted_red[k];
    float white_g = sorted_green[k];
    float white_b = sorted_blue[k];

    // Avoid division by zero
    const float eps = 1e-6f;
    if (white_r < eps || white_g < eps || white_b < eps) {
        throw std::runtime_error("White patch values too small");
    }

    // Apply gains to normalize to white
    for (std::size_t i = 0; i < N; ++i) {
        data_red[i] /= white_r;
        data_green[i] /= white_g;
        data_blue[i] /= white_b;
    }
}


void RadiationCamera::whiteBalanceSpectral(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, helios::Context *context) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationCamera::whiteBalanceSpectral): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    // Check if spectral response data exists for all bands
    if (band_spectral_response.find(red_band_label) == band_spectral_response.end() || band_spectral_response.find(green_band_label) == band_spectral_response.end() || band_spectral_response.find(blue_band_label) == band_spectral_response.end()) {
        helios_runtime_error("ERROR (RadiationCamera::whiteBalanceSpectral): Spectral response data not found for one or more bands. Ensure camera spectral responses are properly initialized.");
    }

    // Get spectral response identifiers
    std::string red_response_id = band_spectral_response.at(red_band_label);
    std::string green_response_id = band_spectral_response.at(green_band_label);
    std::string blue_response_id = band_spectral_response.at(blue_band_label);

    // Skip if using uniform response (cannot apply spectral white balance)
    if (red_response_id == "uniform" && green_response_id == "uniform" && blue_response_id == "uniform") {
        return;
    }

    // Access spectral response data from global data (assuming vec2 format: wavelength, response)
    std::vector<helios::vec2> red_spectrum, green_spectrum, blue_spectrum;

    if (red_response_id != "uniform" && context->doesGlobalDataExist(red_response_id.c_str())) {
        context->getGlobalData(red_response_id.c_str(), red_spectrum);
    }
    if (green_response_id != "uniform" && context->doesGlobalDataExist(green_response_id.c_str())) {
        context->getGlobalData(green_response_id.c_str(), green_spectrum);
    }
    if (blue_response_id != "uniform" && context->doesGlobalDataExist(blue_response_id.c_str())) {
        context->getGlobalData(blue_response_id.c_str(), blue_spectrum);
    }

    // Verify we have spectral data for all channels
    if (red_spectrum.empty() || green_spectrum.empty() || blue_spectrum.empty()) {
        helios_runtime_error("ERROR (RadiationCamera::whiteBalanceSpectral): Could not retrieve spectral response curves for all bands from global data.");
    }

    // Compute integrated response (area under curve) for each channel using trapezoidal integration
    // This represents the total sensitivity of each channel assuming a flat light source spectrum
    float red_integrated = 0.0f, green_integrated = 0.0f, blue_integrated = 0.0f;

    for (size_t i = 1; i < red_spectrum.size(); ++i) {
        float dw = red_spectrum[i].x - red_spectrum[i - 1].x;
        red_integrated += 0.5f * (red_spectrum[i].y + red_spectrum[i - 1].y) * dw;
    }
    for (size_t i = 1; i < green_spectrum.size(); ++i) {
        float dw = green_spectrum[i].x - green_spectrum[i - 1].x;
        green_integrated += 0.5f * (green_spectrum[i].y + green_spectrum[i - 1].y) * dw;
    }
    for (size_t i = 1; i < blue_spectrum.size(); ++i) {
        float dw = blue_spectrum[i].x - blue_spectrum[i - 1].x;
        blue_integrated += 0.5f * (blue_spectrum[i].y + blue_spectrum[i - 1].y) * dw;
    }

    // Check for valid integrated values
    if (red_integrated <= 0 || green_integrated <= 0 || blue_integrated <= 0) {
        helios_runtime_error("ERROR (RadiationCamera::whiteBalanceSpectral): Invalid integrated spectral response (non-positive value). Check spectral response data.");
    }

    // Compute white balance factors relative to each channel's integrated spectral response
    // Normalize relative to the maximum integrated response to preserve brightness
    // This ensures that an object with flat spectral reflectance appears correctly white balanced
    // while keeping the brightest channel at unity gain (factor = 1.0)
    float max_integrated = std::max({red_integrated, green_integrated, blue_integrated});

    helios::vec3 white_balance_factors;
    white_balance_factors.x = max_integrated / red_integrated;
    white_balance_factors.y = max_integrated / green_integrated;
    white_balance_factors.z = max_integrated / blue_integrated;

    // Apply white balance factors to pixel data
    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    const std::size_t N = data_red.size();
    for (std::size_t i = 0; i < N; ++i) {
        data_red[i] *= white_balance_factors.x;
        data_green[i] *= white_balance_factors.y;
        data_blue[i] *= white_balance_factors.z;
    }
}

void RadiationCamera::reinhardToneMapping(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::reinhardToneMapping): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    const std::size_t N = resolution.x * resolution.y;
    constexpr float eps = 1e-6f;

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);
    for (std::size_t i = 0; i < N; ++i) {
        float R = data_red[i], G = data_green[i], B = data_blue[i];
        float L = luminance(R, G, B);
        float s = (L > eps) ? (L / (1.0f + L)) / L : 0.0f;

        data_red[i] = R * s;
        data_green[i] = G * s;
        data_blue[i] = B * s;
    }
}

void RadiationCamera::applyGain(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float percentile) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyGain): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    const std::size_t N = resolution.x * resolution.y;

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    std::vector<float> luminance_pixel;
    luminance_pixel.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        luminance_pixel.push_back(luminance(data_red[i], data_green[i], data_blue[i]));
    }

    std::size_t k = std::size_t(percentile * (luminance_pixel.size() - 1));
    std::nth_element(luminance_pixel.begin(), luminance_pixel.begin() + k, luminance_pixel.end());
    float peak = luminance_pixel[k];
    float gain = (peak > 0.0f) ? 1.0f / peak : 1.0f;

    for (auto &[channel, data]: pixel_data) {
        for (float &v: data) {
            v *= gain;
        }
    }
}

void RadiationCamera::globalHistogramEqualization(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::globalHistogramEquilization): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    const size_t N = resolution.x * resolution.y;
    const float eps = 1e-6f;

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    /* luminance array and store original chromaticity */
    std::vector<float> lum(N);
    std::vector<float> chroma_r(N), chroma_g(N), chroma_b(N);

    for (size_t i = 0; i < N; ++i) {
        vec3 p(data_red[i], data_green[i], data_blue[i]);
        lum[i] = 0.2126f * p.x + 0.7152f * p.y + 0.0722f * p.z;

        // Store chromaticity ratios (color information)
        if (lum[i] > eps) {
            chroma_r[i] = p.x / lum[i];
            chroma_g[i] = p.y / lum[i];
            chroma_b[i] = p.z / lum[i];
        } else {
            chroma_r[i] = 1.0f;
            chroma_g[i] = 1.0f;
            chroma_b[i] = 1.0f;
        }
    }

    /* build CDF on 2048-bin histogram */
    const int B = 2048;
    std::vector<int> hist(B, 0);
    for (float v: lum) {
        int b = int(std::clamp(v, 0.0f, 1.0f - eps) * B);
        if (b >= 0 && b < 2048) {
            hist[b]++;
        }
    }
    std::vector<float> cdf(B);
    int acc = 0;
    for (int b = 0; b < B; ++b) {
        acc += hist[b];
        cdf[b] = float(acc) / float(N);
    }

    /* remap - only adjust luminance, preserve chromaticity */
    for (size_t i = 0; i < N; ++i) {
        // Handle bright pixels (> 1.0) specially
        if (lum[i] >= 1.0f) {
            data_red[i] = std::min(1.0f, data_red[i]);
            data_green[i] = std::min(1.0f, data_green[i]);
            data_blue[i] = std::min(1.0f, data_blue[i]);
            continue;
        }

        int b = int(std::clamp(lum[i], 0.0f, 1.0f - eps) * B);

        if (b < 0 || b >= 2048) {
            continue;
        }

        constexpr float k = 0.2f; // how far to pull towards equalised value  (0.2–0.3 OK)
        constexpr float cs = 0.2f; // S-curve strength   (0.4–0.7 recommended)

        float Yeq = cdf[b]; // equalised luminance  ∈[0,1]
        float Ynew = (1.0f - k) * lum[i] + k * Yeq; // partial equalisation

        /* symmetric S-curve centred at 0.5  :  y = ½ + (x–½)*(1+cs–2·cs·|x–½|)   */
        float t = Ynew - 0.5f;
        Ynew = 0.5f + t * (1.0f + cs - 2.0f * cs * std::fabs(t));

        // Reconstruct RGB using new luminance but original chromaticity
        data_red[i] = Ynew * chroma_r[i];
        data_green[i] = Ynew * chroma_g[i];
        data_blue[i] = Ynew * chroma_b[i];
    }
}

void RadiationCamera::adjustSBC(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation, float brightness, float contrast) {
#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::adjustSBC): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    constexpr float kRedW = 0.2126f;
    constexpr float kGreenW = 0.7152f;
    constexpr float kBlueW = 0.0722f;

    const size_t N = resolution.x * resolution.y;

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    for (int i = 0; i < N; ++i) {

        helios::vec3 p(data_red[i], data_green[i], data_blue[i]);

        /* ----- 1. luminance ----- */
        float Y = kRedW * p.x + kGreenW * p.y + kBlueW * p.z;

        /* ----- 2. saturation ----- */
        p = helios::vec3(Y, Y, Y) + (p - helios::vec3(Y, Y, Y)) * saturation;

        /* ----- 3. brightness (gain) ----- */
        p *= brightness;

        /* ----- 4. contrast ----- */
        p = (p - helios::vec3(0.5f, 0.5f, 0.5f)) * contrast + helios::vec3(0.5f, 0.5f, 0.5f);

        /* ----- 5. clamp to valid range ----- */
        data_red[i] = clamp(p.x, 0.0f, 1.0f);
        data_green[i] = clamp(p.y, 0.0f, 1.0f);
        data_blue[i] = clamp(p.z, 0.0f, 1.0f);
    }
}

// void RadiationCamera::applyCCM(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label) {
//
//     const std::size_t N = resolution.x * resolution.y;
//     auto &data_red = pixel_data.at(red_band_label);
//     auto &data_green = pixel_data.at(green_band_label);
//     auto &data_blue = pixel_data.at(blue_band_label);
//     for (std::size_t i = 0; i < N; ++i) {
//         float R = data_red[i], G = data_green[i], B = data_blue[i];
//         data_red[i] = color_correction_matrix[0] * R + color_correction_matrix[1] * G + color_correction_matrix[2] * B + color_correction_matrix[9];
//         data_green[i] = color_correction_matrix[3] * R + color_correction_matrix[4] * G + color_correction_matrix[5] * B + color_correction_matrix[10];
//         data_blue[i] = color_correction_matrix[6] * R + color_correction_matrix[7] * G + color_correction_matrix[8] * B + color_correction_matrix[11];
//     }
// }

void RadiationCamera::gammaCompress(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::gammaCompress): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    for (float &v: pixel_data.at(red_band_label)) {
        v = lin_to_srgb(std::fmaxf(0.0f, v));
    }
    for (float &v: pixel_data.at(green_band_label)) {
        v = lin_to_srgb(std::fmaxf(0.0f, v));
    }
    for (float &v: pixel_data.at(blue_band_label)) {
        v = lin_to_srgb(std::fmaxf(0.0f, v));
    }
}

// New methods for improved image processing pipeline

void RadiationCamera::autoExposure(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float gain_multiplier) {
#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::autoExposure): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    const std::size_t N = data_red.size();

    // Calculate luminance for each pixel
    std::vector<float> luminance_values(N);
    for (std::size_t i = 0; i < N; ++i) {
        luminance_values[i] = luminance(data_red[i], data_green[i], data_blue[i]);
    }

    // Sort luminance values to find percentiles
    std::vector<float> sorted_luminance = luminance_values;
    std::sort(sorted_luminance.begin(), sorted_luminance.end());

    // Calculate 95th percentile for exposure (prevents bright outliers from under-exposing scene)
    std::size_t p95_idx = static_cast<std::size_t>(0.95f * (N - 1));
    float p95_luminance = sorted_luminance[p95_idx];

    // Calculate median luminance for scene analysis
    std::size_t median_idx = N / 2;
    float median_luminance = sorted_luminance[median_idx];

    // Target median luminance scaled appropriately for the data range
    // Since RGB data is not normalized to [0,1], we need to scale the target accordingly
    float target_median = 0.18f; // Calibrated based on empirical testing
    float auto_gain = target_median / std::max(median_luminance, 1e-6f);

    // Clamp auto-gain to reasonable range to prevent over/under exposure
    // auto_gain = std::clamp(auto_gain, 0.0005f, 0.5f);

    // Apply final gain (auto-exposure * manual adjustment)
    float final_gain = auto_gain * gain_multiplier;

    // Apply gain to all channels
    for (std::size_t i = 0; i < N; ++i) {
        data_red[i] *= final_gain;
        data_green[i] *= final_gain;
        data_blue[i] *= final_gain;
    }
}

void RadiationCamera::applyCameraExposure(helios::Context *context) {
    // Skip if pixel_data is empty (camera hasn't been rendered yet)
    if (pixel_data.empty()) {
        return;
    }

    // Verify that all expected bands exist in pixel_data
    for (const auto &band: band_labels) {
        if (pixel_data.find(band) == pixel_data.end()) {
            return; // Skip exposure if not all bands are populated yet
        }
    }

    // Parse exposure mode
    std::string exposure_mode = exposure;

    // Manual mode: no automatic exposure scaling
    if (exposure_mode == "manual") {
        return;
    }

    // Auto mode: apply automatic exposure based on camera type
    if (exposure_mode == "auto") {
        // Determine camera type: if not set explicitly, infer from band count
        std::string cam_type;
        if (!camera_type.empty()) {
            cam_type = camera_type;
        } else {
            // Infer type for manually created cameras
            cam_type = (band_labels.size() >= 3) ? "rgb" : "spectral";
        }

        if (cam_type == "thermal") {
            // Thermal cameras: skip exposure adjustment
            return;
        } else if (cam_type == "rgb" && band_labels.size() >= 3) {
            // RGB cameras: luminance-based auto-exposure (18% gray target)

            // Use the first 3 bands as RGB (or find bands named "red", "green", "blue")
            std::string red_band, green_band, blue_band;
            for (const auto &band: band_labels) {
                if (band.find("red") != std::string::npos || band.find("Red") != std::string::npos || band.find("RED") != std::string::npos) {
                    red_band = band;
                } else if (band.find("green") != std::string::npos || band.find("Green") != std::string::npos || band.find("GREEN") != std::string::npos) {
                    green_band = band;
                } else if (band.find("blue") != std::string::npos || band.find("Blue") != std::string::npos || band.find("BLUE") != std::string::npos) {
                    blue_band = band;
                }
            }

            // Fallback to first 3 bands if named bands not found
            if (red_band.empty())
                red_band = band_labels[0];
            if (green_band.empty())
                green_band = band_labels[1];
            if (blue_band.empty())
                blue_band = band_labels[2];

            auto &data_red = pixel_data.at(red_band);
            auto &data_green = pixel_data.at(green_band);
            auto &data_blue = pixel_data.at(blue_band);

            const std::size_t N = data_red.size();

            // Calculate luminance for each pixel
            std::vector<float> luminance_values(N);
            for (std::size_t i = 0; i < N; ++i) {
                luminance_values[i] = luminance(data_red[i], data_green[i], data_blue[i]);
            }

            // Sort to find median
            std::vector<float> sorted_luminance = luminance_values;
            std::sort(sorted_luminance.begin(), sorted_luminance.end());

            std::size_t median_idx = N / 2;
            float median_luminance = sorted_luminance[median_idx];

            // Target 18% gray
            float target_median = 0.18f;
            float auto_gain = target_median / std::max(median_luminance, 1e-6f);

            // Apply gain to all bands
            for (auto &band_pair: pixel_data) {
                auto &data = band_pair.second;
                for (std::size_t i = 0; i < N; ++i) {
                    data[i] *= auto_gain;
                }
            }

        } else if (cam_type == "spectral") {
            // Spectral cameras: per-band normalization
            for (auto &band_pair: pixel_data) {
                auto &data = band_pair.second;
                const std::size_t N = data.size();

                // Calculate median for this band
                std::vector<float> sorted_data = data;
                std::sort(sorted_data.begin(), sorted_data.end());

                std::size_t median_idx = N / 2;
                float median_value = sorted_data[median_idx];

                // Target 18% gray for each band independently
                float target_median = 0.18f;
                float band_gain = target_median / std::max(median_value, 1e-6f);

                // Apply gain to this band
                for (std::size_t i = 0; i < N; ++i) {
                    data[i] *= band_gain;
                }
            }
        } else {
            helios_runtime_error("ERROR (RadiationCamera::applyCameraExposure): Unknown camera_type '" + cam_type + "'. Must be 'rgb', 'spectral', or 'thermal'.");
        }
        return;
    }

    // ISO mode: "ISOXXX" (e.g., "ISO100", "ISO200", etc.)
    if (exposure_mode.substr(0, 3) == "ISO" || exposure_mode.substr(0, 3) == "iso") {
        // Parse ISO value
        int iso_value;
        try {
            iso_value = std::stoi(exposure_mode.substr(3));
        } catch (...) {
            helios_runtime_error("ERROR (RadiationCamera::applyCameraExposure): Invalid ISO format '" + exposure_mode + "'. Expected format: 'ISOXXX' (e.g., 'ISO100').");
        }

        if (iso_value <= 0) {
            helios_runtime_error("ERROR (RadiationCamera::applyCameraExposure): ISO value must be positive. Got: " + std::to_string(iso_value));
        }

        // Validate that lens_focal_length is set (required for ISO mode)
        if (lens_focal_length <= 0) {
            helios_runtime_error("ERROR (RadiationCamera::applyCameraExposure): ISO mode requires lens_focal_length to be set. Camera '" + label + "' has lens_focal_length = " + std::to_string(lens_focal_length) +
                                 ". Either set it explicitly or use 'auto' or 'manual' exposure mode.");
        }

        // Calculate f-number from lens diameter and optical focal length
        // f-number = lens_focal_length / lens_diameter
        float f_number = lens_focal_length / std::max(lens_diameter, 1e-6f);

        // Reference camera settings (chosen to match typical photography)
        const float ref_iso = 100.0f;
        const float ref_shutter = 1.0f / 125.0f;
        const float ref_f_number = 2.8f;

        // Calibration to match auto-exposure behavior
        // Typical Helios scenes have raw median ~10, auto-exposure targets 0.0675
        // At reference settings (ISO 100, 1/125s, f/2.8), we want the same result as auto
        const float typical_scene_median = 10.0f;
        const float target_median = 0.0675f;

        // Calculate exposure from camera settings (proportional to ISO × t / N²)
        // Higher ISO → brighter, longer shutter → brighter, wider aperture (smaller N) → brighter
        float exposure = (float(iso_value) * shutter_speed) / (f_number * f_number);
        float ref_exposure = (ref_iso * ref_shutter) / (ref_f_number * ref_f_number);

        // Calibration: at reference settings with typical scene, achieve target_median
        // Required gain at reference: target_median / typical_scene_median
        // This gain must equal: ref_exposure × calibration_factor
        // Therefore: calibration_factor = (target_median / typical_scene_median) / ref_exposure
        float ref_gain = target_median / typical_scene_median;
        float calibration_factor = ref_gain / ref_exposure;

        // Final exposure multiplier
        float exposure_multiplier = exposure * calibration_factor;

        // Apply exposure to all bands
        const std::size_t N = pixel_data.begin()->second.size();
        for (auto &band_pair: pixel_data) {
            auto &data = band_pair.second;
            for (std::size_t i = 0; i < N; ++i) {
                data[i] *= exposure_multiplier;
            }
        }
        return;
    }

    // Unknown exposure mode
    helios_runtime_error("ERROR (RadiationCamera::applyCameraExposure): Unknown exposure mode '" + exposure_mode + "'. Must be 'auto', 'ISOXXX' (e.g., 'ISO100'), or 'manual'.");
}

void RadiationCamera::applyCameraWhiteBalance(helios::Context *context) {
    // Skip if pixel_data is empty (camera hasn't been rendered yet)
    if (pixel_data.empty()) {
        return;
    }

    // Verify that all expected bands exist in pixel_data
    for (const auto &band: band_labels) {
        if (pixel_data.find(band) == pixel_data.end()) {
            return; // Skip white balance if not all bands are populated yet
        }
    }

    // Parse white balance mode
    std::string wb_mode = white_balance;

    // "off" mode: no white balance correction
    if (wb_mode == "off") {
        return;
    }

    // Skip white balance for single-channel images (grayscale/thermal)
    if (band_labels.size() < 3) {
        return;
    }

    // "auto" mode: apply spectral white balance
    if (wb_mode == "auto") {
        // For 3+ channel images, apply white balance to first 3 channels
        // Assume standard RGB ordering for the first 3 bands
        std::string red_band = band_labels[0];
        std::string green_band = band_labels[1];
        std::string blue_band = band_labels[2];

        try {
            whiteBalanceSpectral(red_band, green_band, blue_band, context);
        } catch (const std::exception &e) {
            // If spectral white balance fails (e.g., no spectral data), silently skip
            // This matches the behavior of whiteBalanceSpectral which returns early
            // when all bands use "uniform" response
        }
        return;
    }

    // Unknown white balance mode
    helios_runtime_error("ERROR (RadiationCamera::applyCameraWhiteBalance): Unknown white_balance mode '" + wb_mode + "'. Must be 'auto' or 'off'.");
}

void RadiationCamera::adjustBrightnessContrast(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float brightness, float contrast) {
#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::adjustBrightnessContrast): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    const std::size_t N = data_red.size();

    for (std::size_t i = 0; i < N; ++i) {
        // Apply brightness adjustment
        float r = data_red[i] * brightness;
        float g = data_green[i] * brightness;
        float b = data_blue[i] * brightness;

        // Apply contrast adjustment (around 0.5 midpoint in linear space)
        r = 0.5f + (r - 0.5f) * contrast;
        g = 0.5f + (g - 0.5f) * contrast;
        b = 0.5f + (b - 0.5f) * contrast;

        // Store results (allow values outside [0,1] range for HDR processing)
        data_red[i] = r;
        data_green[i] = g;
        data_blue[i] = b;
    }
}

void RadiationCamera::adjustSaturation(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation) {
#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::adjustSaturation): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    auto &data_red = pixel_data.at(red_band_label);
    auto &data_green = pixel_data.at(green_band_label);
    auto &data_blue = pixel_data.at(blue_band_label);

    const std::size_t N = data_red.size();

    for (std::size_t i = 0; i < N; ++i) {
        float r = data_red[i];
        float g = data_green[i];
        float b = data_blue[i];

        // Calculate luminance for this pixel
        float lum = luminance(r, g, b);

        // Apply saturation adjustment by interpolating between luminance (grayscale) and original color
        data_red[i] = lum + saturation * (r - lum);
        data_green[i] = lum + saturation * (g - lum);
        data_blue[i] = lum + saturation * (b - lum);
    }
}

// -------------------- Camera Metadata Export Methods -------------------- //

std::string RadiationModel::detectLightingType() const {
    if (radiation_sources.empty()) {
        return "none";
    }

    bool has_sun = false;
    bool has_artificial = false;

    for (const auto &source: radiation_sources) {
        if (source.source_type == RADIATION_SOURCE_TYPE_COLLIMATED || source.source_type == RADIATION_SOURCE_TYPE_SUN_SPHERE) {
            has_sun = true;
        } else if (source.source_type == RADIATION_SOURCE_TYPE_SPHERE || source.source_type == RADIATION_SOURCE_TYPE_RECTANGLE || source.source_type == RADIATION_SOURCE_TYPE_DISK) {
            has_artificial = true;
        }
    }

    if (has_sun && has_artificial) {
        return "mixed";
    } else if (has_sun) {
        return "sunlight";
    } else if (has_artificial) {
        return "artificial";
    } else {
        return "none";
    }
}

float RadiationModel::calculateCameraTiltAngle(const helios::vec3 &position, const helios::vec3 &lookat) const {
    // Calculate viewing direction vector
    helios::vec3 direction = lookat - position;
    direction.normalize();

    // Calculate tilt from horizontal (0 = horizontal, 90 = straight down, -90 = straight up)
    // The z component gives us the vertical component of the direction
    // Tilt angle = -asin(direction.z) in degrees
    float tilt_angle_deg = -asin(direction.z) * 180.0f / M_PI;

    return tilt_angle_deg;
}

void RadiationModel::computeAgronomicProperties(const std::string &camera_label, CameraMetadata::AgronomicProperties &props) const {
    // Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::computeAgronomicProperties): Camera '" + camera_label + "' does not exist.");
    }

    const auto &cam = cameras.at(camera_label);

    // Clear any existing data
    props.plant_species.clear();
    props.plant_count.clear();
    props.plant_height_m.clear();
    props.plant_age_days.clear();
    props.plant_stage.clear();
    props.leaf_area_m2.clear();
    props.weed_pressure = "";

    // Load pixel UUID map from global data
    std::vector<uint> pixel_UUIDs;
    std::string pixel_UUID_label = "camera_" + camera_label + "_pixel_UUID";
    if (!context->doesGlobalDataExist(pixel_UUID_label.c_str())) {
        // No pixel UUID data available - skip agronomic properties
        return;
    }
    context->getGlobalData(pixel_UUID_label.c_str(), pixel_UUIDs);

    // Map: species_name -> set of unique plantIDs for that species
    std::map<std::string, std::set<int>> species_to_plantIDs;

    // Set of all plantIDs that are weeds
    std::set<int> weed_plantIDs;

    // Set of all unique plantIDs (for weed pressure calculation)
    std::set<int> all_plantIDs;

    // Maps for new agronomic properties (per species, per plantID)
    std::map<std::string, std::map<int, float>> species_plant_heights; // species -> (plantID -> height)
    std::map<std::string, std::map<int, float>> species_plant_ages; // species -> (plantID -> age)
    std::map<std::string, std::map<int, std::string>> species_plant_stages; // species -> (plantID -> stage)
    std::map<std::string, std::map<int, float>> species_plant_leaf_areas; // species -> (plantID -> leaf area)
    std::map<std::string, std::map<int, int>> species_plant_pixel_counts; // species -> (plantID -> pixel count) for weighted averaging

    // Iterate through all pixels to find unique objects and query their data
    for (uint j = 0; j < cam.resolution.y; j++) {
        for (uint i = 0; i < cam.resolution.x; i++) {
            uint pixel_index = j * cam.resolution.x + i;

            if (pixel_index >= pixel_UUIDs.size()) {
                continue;
            }

            uint UUID_plus_one = pixel_UUIDs.at(pixel_index);
            if (UUID_plus_one == 0) {
                // Sky pixel, skip
                continue;
            }

            uint UUID = UUID_plus_one - 1;

            // Check if primitive exists
            if (!context->doesPrimitiveExist(UUID)) {
                continue;
            }

            // Get parent object ID
            uint objID = context->getPrimitiveParentObjectID(UUID);
            if (objID == 0) {
                // Primitive has no parent object, skip
                continue;
            }

            // Query plant_name (species)
            std::string plant_name;
            bool has_plant_name = false;
            if (context->doesObjectDataExist(objID, "plant_name")) {
                HeliosDataType datatype = context->getObjectDataType("plant_name");
                if (datatype == HELIOS_TYPE_STRING) {
                    context->getObjectData(objID, "plant_name", plant_name);
                    has_plant_name = true;
                }
            }

            // Query plantID
            int plantID = -1;
            bool has_plantID = false;
            if (context->doesObjectDataExist(objID, "plantID")) {
                HeliosDataType datatype = context->getObjectDataType("plantID");
                if (datatype == HELIOS_TYPE_INT) {
                    context->getObjectData(objID, "plantID", plantID);
                    has_plantID = true;
                } else if (datatype == HELIOS_TYPE_UINT) {
                    uint plantID_uint;
                    context->getObjectData(objID, "plantID", plantID_uint);
                    plantID = static_cast<int>(plantID_uint);
                    has_plantID = true;
                }
            }

            // Query plant_type (to identify weeds)
            std::string plant_type;
            bool has_plant_type = false;
            if (context->doesObjectDataExist(objID, "plant_type")) {
                HeliosDataType datatype = context->getObjectDataType("plant_type");
                if (datatype == HELIOS_TYPE_STRING) {
                    context->getObjectData(objID, "plant_type", plant_type);
                    has_plant_type = true;
                }
            }

            // Query plant_height (for new agronomic metadata)
            float plant_height = 0.0f;
            bool has_plant_height = false;
            if (context->doesObjectDataExist(objID, "plant_height")) {
                HeliosDataType datatype = context->getObjectDataType("plant_height");
                if (datatype == HELIOS_TYPE_FLOAT) {
                    context->getObjectData(objID, "plant_height", plant_height);
                    has_plant_height = true;
                }
            }

            // Query age (for new agronomic metadata)
            float age = 0.0f;
            bool has_age = false;
            if (context->doesObjectDataExist(objID, "age")) {
                HeliosDataType datatype = context->getObjectDataType("age");
                if (datatype == HELIOS_TYPE_FLOAT) {
                    context->getObjectData(objID, "age", age);
                    has_age = true;
                }
            }

            // Query phenology_stage (for new agronomic metadata)
            std::string phenology_stage;
            bool has_phenology_stage = false;
            if (context->doesObjectDataExist(objID, "phenology_stage")) {
                HeliosDataType datatype = context->getObjectDataType("phenology_stage");
                if (datatype == HELIOS_TYPE_STRING) {
                    context->getObjectData(objID, "phenology_stage", phenology_stage);
                    has_phenology_stage = true;
                }
            }

            // Get primitive surface area (for leaf area calculation)
            float primitive_area = context->getPrimitiveArea(UUID);

            // Only process if we have the required data
            if (has_plant_name && has_plantID) {
                // Add plantID to the species set
                species_to_plantIDs[plant_name].insert(plantID);

                // Add to all plantIDs set
                all_plantIDs.insert(plantID);

                // Check if this plant is a weed
                if (has_plant_type && plant_type == "weed") {
                    weed_plantIDs.insert(plantID);
                }

                // Accumulate new agronomic data per species and plantID
                if (has_plant_height) {
                    species_plant_heights[plant_name][plantID] = plant_height;
                }
                if (has_age) {
                    species_plant_ages[plant_name][plantID] = age;
                }
                if (has_phenology_stage) {
                    species_plant_stages[plant_name][plantID] = phenology_stage;
                }

                // Accumulate leaf area for this plant
                species_plant_leaf_areas[plant_name][plantID] += primitive_area;

                // Track pixel count for weighted averaging
                species_plant_pixel_counts[plant_name][plantID]++;
            }
        }
    }

    // If no valid data was found, leave properties empty
    if (species_to_plantIDs.empty()) {
        return;
    }

    // Build plant_species and plant_count vectors
    for (const auto &species_pair: species_to_plantIDs) {
        props.plant_species.push_back(species_pair.first);
        props.plant_count.push_back(static_cast<int>(species_pair.second.size()));
    }

    // Compute new agronomic properties per species (parallel to plant_species vector)
    for (const auto &species_pair: species_to_plantIDs) {
        const std::string &species = species_pair.first;
        const std::set<int> &plantIDs = species_pair.second;

        // --- Plant Height (weighted average by pixel count) ---
        if (species_plant_heights.find(species) != species_plant_heights.end()) {
            float total_weighted_height = 0.0f;
            int total_pixels = 0;
            for (int plantID: plantIDs) {
                if (species_plant_heights.at(species).find(plantID) != species_plant_heights.at(species).end()) {
                    float height = species_plant_heights.at(species).at(plantID);
                    int pixel_count = species_plant_pixel_counts.at(species).at(plantID);
                    total_weighted_height += height * static_cast<float>(pixel_count);
                    total_pixels += pixel_count;
                }
            }
            if (total_pixels > 0) {
                props.plant_height_m.push_back(total_weighted_height / static_cast<float>(total_pixels));
            } else {
                props.plant_height_m.push_back(0.0f);
            }
        } else {
            props.plant_height_m.push_back(0.0f);
        }

        // --- Plant Age (weighted average by pixel count) ---
        if (species_plant_ages.find(species) != species_plant_ages.end()) {
            float total_weighted_age = 0.0f;
            int total_pixels = 0;
            for (int plantID: plantIDs) {
                if (species_plant_ages.at(species).find(plantID) != species_plant_ages.at(species).end()) {
                    float age = species_plant_ages.at(species).at(plantID);
                    int pixel_count = species_plant_pixel_counts.at(species).at(plantID);
                    total_weighted_age += age * static_cast<float>(pixel_count);
                    total_pixels += pixel_count;
                }
            }
            if (total_pixels > 0) {
                props.plant_age_days.push_back(total_weighted_age / static_cast<float>(total_pixels));
            } else {
                props.plant_age_days.push_back(0.0f);
            }
        } else {
            props.plant_age_days.push_back(0.0f);
        }

        // --- Plant Stage (mode - most common stage) ---
        if (species_plant_stages.find(species) != species_plant_stages.end()) {
            std::map<std::string, int> stage_counts;
            for (int plantID: plantIDs) {
                if (species_plant_stages.at(species).find(plantID) != species_plant_stages.at(species).end()) {
                    std::string stage = species_plant_stages.at(species).at(plantID);
                    stage_counts[stage]++;
                }
            }
            // Find most common stage
            std::string mode_stage;
            int max_count = 0;
            for (const auto &stage_pair: stage_counts) {
                if (stage_pair.second > max_count) {
                    max_count = stage_pair.second;
                    mode_stage = stage_pair.first;
                }
            }
            props.plant_stage.push_back(mode_stage);
        } else {
            props.plant_stage.push_back("");
        }

        // --- Leaf Area (sum of all leaf areas for this species) ---
        if (species_plant_leaf_areas.find(species) != species_plant_leaf_areas.end()) {
            float total_leaf_area = 0.0f;
            for (int plantID: plantIDs) {
                if (species_plant_leaf_areas.at(species).find(plantID) != species_plant_leaf_areas.at(species).end()) {
                    total_leaf_area += species_plant_leaf_areas.at(species).at(plantID);
                }
            }
            props.leaf_area_m2.push_back(total_leaf_area);
        } else {
            props.leaf_area_m2.push_back(0.0f);
        }
    }

    // Calculate weed pressure
    if (!all_plantIDs.empty()) {
        float weed_fraction = static_cast<float>(weed_plantIDs.size()) / static_cast<float>(all_plantIDs.size());
        float weed_percentage = weed_fraction * 100.0f;

        if (weed_percentage <= 20.0f) {
            props.weed_pressure = "low";
        } else if (weed_percentage <= 40.0f) {
            props.weed_pressure = "moderate";
        } else {
            props.weed_pressure = "high";
        }
    }
}

void RadiationModel::populateCameraMetadata(const std::string &camera_label, CameraMetadata &metadata) const {
    // Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::populateCameraMetadata): Camera '" + camera_label + "' does not exist.");
    }

    const auto &cam = cameras.at(camera_label);

    // --- Camera Properties --- //
    metadata.camera_properties.width = cam.resolution.x;
    metadata.camera_properties.height = cam.resolution.y;
    metadata.camera_properties.channels = static_cast<int>(cam.band_labels.size());
    metadata.camera_properties.type = cam.camera_type;

    // Calculate sensor dimensions from HFOV and sensor_width_mm
    // sensor_width is already in mm, stored in the camera
    metadata.camera_properties.sensor_width = cam.sensor_width_mm;

    // Calculate VFOV from HFOV and aspect ratio
    float VFOV_degrees = cam.HFOV_degrees / cam.FOV_aspect_ratio;

    // Calculate sensor height from sensor width and aspect ratio
    metadata.camera_properties.sensor_height = cam.sensor_width_mm / cam.FOV_aspect_ratio;

    // Back-calculate optical focal length from HFOV and sensor width for metadata export
    // IMPORTANT: All metadata values reflect the REFERENCE state at zoom=1.0, not the zoomed state.
    // - focal_length is calculated from the base HFOV (cam.HFOV_degrees), not effective HFOV
    // - sensor dimensions are physical properties unaffected by zoom
    // - The zoom value itself is written separately so users can reconstruct effective parameters
    // This ensures metadata accurately reflects the configured camera geometry (HFOV)
    // Note: For ISO exposure calculations, we use lens_focal_length which may differ from this value
    float HFOV_rad = cam.HFOV_degrees * M_PI / 180.0f;
    float optical_focal_length_mm = cam.sensor_width_mm / (2.0f * tan(HFOV_rad / 2.0f));
    metadata.camera_properties.focal_length = optical_focal_length_mm;

    // Calculate aperture (f-number = optical_focal_length / lens_diameter)
    if (cam.lens_diameter > 0) {
        float lens_diameter_mm = cam.lens_diameter * 1000.0f; // Convert meters to mm
        float f_number = optical_focal_length_mm / lens_diameter_mm;
        std::ostringstream aperture_str;
        aperture_str << "f/" << std::fixed << std::setprecision(1) << f_number;
        metadata.camera_properties.aperture = aperture_str.str();
    } else {
        metadata.camera_properties.aperture = "pinhole";
    }

    // Camera model
    metadata.camera_properties.model = cam.model;

    // Lens metadata
    metadata.camera_properties.lens_make = cam.lens_make;
    metadata.camera_properties.lens_model = cam.lens_model;
    metadata.camera_properties.lens_specification = cam.lens_specification;

    // Exposure settings
    metadata.camera_properties.exposure = cam.exposure;
    metadata.camera_properties.shutter_speed = cam.shutter_speed;

    // White balance mode
    metadata.camera_properties.white_balance = cam.white_balance;

    // Zoom setting
    metadata.camera_properties.camera_zoom = cam.camera_zoom;

    // --- Location Properties --- //
    helios::Location loc = context->getLocation();
    metadata.location_properties.latitude = loc.latitude_deg;
    metadata.location_properties.longitude = loc.longitude_deg;

    // --- Acquisition Properties --- //
    helios::Date date = context->getDate();
    helios::Time time = context->getTime();

    // Format date as YYYY-MM-DD
    std::ostringstream date_str;
    date_str << date.year << "-" << std::setw(2) << std::setfill('0') << date.month << "-" << std::setw(2) << std::setfill('0') << date.day;
    metadata.acquisition_properties.date = date_str.str();

    // Format time as HH:MM:SS
    std::ostringstream time_str;
    time_str << std::setw(2) << std::setfill('0') << time.hour << ":" << std::setw(2) << std::setfill('0') << time.minute << ":" << std::setw(2) << std::setfill('0') << time.second;
    metadata.acquisition_properties.time = time_str.str();

    metadata.acquisition_properties.UTC_offset = loc.UTC_offset;
    metadata.acquisition_properties.camera_height_m = cam.position.z;
    metadata.acquisition_properties.camera_angle_deg = calculateCameraTiltAngle(cam.position, cam.lookat);
    metadata.acquisition_properties.light_source = detectLightingType();

    // --- Agronomic Properties --- //
    computeAgronomicProperties(camera_label, metadata.agronomic_properties);

    // Note: path field is left empty and will be set when image is written
    metadata.path = "";
}

void RadiationModel::enableCameraMetadata(const std::string &camera_label) {
    // Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::enableCameraMetadata): Camera '" + camera_label + "' does not exist.");
    }

    // Preserve any existing image_processing parameters (e.g., from applyCameraImageCorrections)
    CameraMetadata::ImageProcessingProperties saved_image_processing;
    if (camera_metadata.find(camera_label) != camera_metadata.end()) {
        saved_image_processing = camera_metadata.at(camera_label).image_processing;
    }

    // Populate metadata from camera properties and context
    CameraMetadata metadata;
    populateCameraMetadata(camera_label, metadata);

    // Restore image_processing parameters
    metadata.image_processing = saved_image_processing;

    // Store metadata and mark camera as enabled for metadata writing
    camera_metadata[camera_label] = metadata;
    metadata_enabled_cameras.insert(camera_label);
}

void RadiationModel::enableCameraMetadata(const std::vector<std::string> &camera_labels) {
    // Enable metadata for each camera in the vector
    for (const auto &camera_label: camera_labels) {
        enableCameraMetadata(camera_label);
    }
}

CameraMetadata RadiationModel::getCameraMetadata(const std::string &camera_label) const {
    // Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraMetadata): Camera '" + camera_label + "' does not exist.");
    }

    // Re-populate metadata to ensure it reflects current state (e.g., light sources, updated context data)
    CameraMetadata metadata;
    populateCameraMetadata(camera_label, metadata);

    return metadata;
}

void RadiationModel::setCameraMetadata(const std::string &camera_label, const CameraMetadata &metadata) {
    // Validate camera exists
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::setCameraMetadata): Camera '" + camera_label + "' does not exist.");
    }

    camera_metadata[camera_label] = metadata;
}

std::string RadiationModel::writeCameraMetadataFile(const std::string &camera_label, const std::string &output_path) const {
    // Validate camera has metadata
    if (camera_metadata.find(camera_label) == camera_metadata.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeCameraMetadataFile): No metadata set for camera '" + camera_label + "'.");
    }

    const auto &metadata = camera_metadata.at(camera_label);

    // Helper lambda to format floats with specific decimal precision for clean JSON output
    // Converts to string with fixed precision, then parses as double to avoid float representation artifacts
    auto format_float = [](float value, int decimals) -> double {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(decimals) << value;
        return std::stod(oss.str());
    };

    // Build JSON structure matching schema
    nlohmann::json j;
    j["path"] = metadata.path;

    j["camera_properties"]["height"] = metadata.camera_properties.height;
    j["camera_properties"]["width"] = metadata.camera_properties.width;
    j["camera_properties"]["channels"] = metadata.camera_properties.channels;
    j["camera_properties"]["type"] = metadata.camera_properties.type;
    j["camera_properties"]["focal_length"] = format_float(metadata.camera_properties.focal_length, 2);
    j["camera_properties"]["aperture"] = metadata.camera_properties.aperture;
    j["camera_properties"]["sensor_width"] = format_float(metadata.camera_properties.sensor_width, 2);
    j["camera_properties"]["sensor_height"] = format_float(metadata.camera_properties.sensor_height, 2);
    j["camera_properties"]["model"] = metadata.camera_properties.model;

    // Only include lens fields if they're not empty
    if (!metadata.camera_properties.lens_make.empty()) {
        j["camera_properties"]["lens_make"] = metadata.camera_properties.lens_make;
    }
    if (!metadata.camera_properties.lens_model.empty()) {
        j["camera_properties"]["lens_model"] = metadata.camera_properties.lens_model;
    }
    if (!metadata.camera_properties.lens_specification.empty()) {
        j["camera_properties"]["lens_specification"] = metadata.camera_properties.lens_specification;
    }

    // Exposure settings
    j["camera_properties"]["exposure"] = metadata.camera_properties.exposure;
    j["camera_properties"]["shutter_speed"] = format_float(metadata.camera_properties.shutter_speed, 6);

    // White balance mode
    j["camera_properties"]["white_balance"] = metadata.camera_properties.white_balance;

    // Camera zoom setting
    j["camera_properties"]["zoom"] = format_float(metadata.camera_properties.camera_zoom, 2);

    j["location_properties"]["latitude"] = format_float(metadata.location_properties.latitude, 6);
    j["location_properties"]["longitude"] = format_float(metadata.location_properties.longitude, 6);

    j["acquisition_properties"]["date"] = metadata.acquisition_properties.date;
    j["acquisition_properties"]["time"] = metadata.acquisition_properties.time;
    j["acquisition_properties"]["UTC_offset"] = format_float(metadata.acquisition_properties.UTC_offset, 1);
    j["acquisition_properties"]["camera_height_m"] = format_float(metadata.acquisition_properties.camera_height_m, 2);
    j["acquisition_properties"]["camera_angle_deg"] = format_float(metadata.acquisition_properties.camera_angle_deg, 2);
    j["acquisition_properties"]["light_source"] = metadata.acquisition_properties.light_source;

    // Always include image_processing section with color_space
    const auto &img_proc = metadata.image_processing;
    j["image_processing"]["saturation_adjustment"] = format_float(img_proc.saturation_adjustment, 2);
    j["image_processing"]["brightness_adjustment"] = format_float(img_proc.brightness_adjustment, 2);
    j["image_processing"]["contrast_adjustment"] = format_float(img_proc.contrast_adjustment, 2);
    j["image_processing"]["color_space"] = img_proc.color_space;

    // Only include agronomic_properties if data is available
    if (!metadata.agronomic_properties.plant_species.empty()) {
        j["agronomic_properties"]["plant_species"] = metadata.agronomic_properties.plant_species;
        j["agronomic_properties"]["plant_count"] = metadata.agronomic_properties.plant_count;

        // Format new agronomic fields with appropriate precision
        if (!metadata.agronomic_properties.plant_height_m.empty()) {
            std::vector<double> formatted_heights;
            for (float height: metadata.agronomic_properties.plant_height_m) {
                formatted_heights.push_back(format_float(height, 2));
            }
            j["agronomic_properties"]["plant_height_m"] = formatted_heights;
        }

        if (!metadata.agronomic_properties.plant_age_days.empty()) {
            std::vector<double> formatted_ages;
            for (float age: metadata.agronomic_properties.plant_age_days) {
                formatted_ages.push_back(format_float(age, 1));
            }
            j["agronomic_properties"]["plant_age_days"] = formatted_ages;
        }

        if (!metadata.agronomic_properties.plant_stage.empty()) {
            j["agronomic_properties"]["plant_stage"] = metadata.agronomic_properties.plant_stage;
        }

        if (!metadata.agronomic_properties.leaf_area_m2.empty()) {
            std::vector<double> formatted_leaf_areas;
            for (float area: metadata.agronomic_properties.leaf_area_m2) {
                formatted_leaf_areas.push_back(format_float(area, 4));
            }
            j["agronomic_properties"]["leaf_area_m2"] = formatted_leaf_areas;
        }

        j["agronomic_properties"]["weed_pressure"] = metadata.agronomic_properties.weed_pressure;
    }

    // Generate JSON filename (replace image extension with .json)
    std::string json_filename = metadata.path;
    size_t ext_pos = json_filename.find_last_of(".");
    if (ext_pos != std::string::npos) {
        json_filename = json_filename.substr(0, ext_pos) + ".json";
    } else {
        json_filename += ".json";
    }

    // Construct full path with output directory
    std::string json_path = output_path + json_filename;

    // Write to file
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
        helios_runtime_error("ERROR (RadiationModel::writeCameraMetadataFile): Failed to open file '" + json_path + "' for writing.");
    }
    json_file << j.dump(2) << std::endl; // Pretty print with 2-space indentation
    json_file.close();

    return json_path;
}

void RadiationModel::enableCameraLensFlare(const std::string &camera_label) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::enableCameraLensFlare): Camera '" + camera_label + "' does not exist.");
    }
    cameras.at(camera_label).lens_flare_enabled = true;
}

void RadiationModel::disableCameraLensFlare(const std::string &camera_label) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::disableCameraLensFlare): Camera '" + camera_label + "' does not exist.");
    }
    cameras.at(camera_label).lens_flare_enabled = false;
}

bool RadiationModel::isCameraLensFlareEnabled(const std::string &camera_label) const {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::isCameraLensFlareEnabled): Camera '" + camera_label + "' does not exist.");
    }
    return cameras.at(camera_label).lens_flare_enabled;
}

void RadiationModel::setCameraLensFlareProperties(const std::string &camera_label, const LensFlareProperties &properties) {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLensFlareProperties): Camera '" + camera_label + "' does not exist.");
    }

    // Validate properties
    if (properties.aperture_blade_count < 3) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLensFlareProperties): aperture_blade_count must be at least 3.");
    }
    if (properties.coating_efficiency < 0.0f || properties.coating_efficiency > 1.0f) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLensFlareProperties): coating_efficiency must be in range [0.0, 1.0].");
    }
    if (properties.ghost_intensity < 0.0f) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLensFlareProperties): ghost_intensity must be non-negative.");
    }
    if (properties.starburst_intensity < 0.0f) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLensFlareProperties): starburst_intensity must be non-negative.");
    }
    if (properties.intensity_threshold < 0.0f || properties.intensity_threshold > 1.0f) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLensFlareProperties): intensity_threshold must be in range [0.0, 1.0].");
    }
    if (properties.ghost_count < 1) {
        helios_runtime_error("ERROR (RadiationModel::setCameraLensFlareProperties): ghost_count must be at least 1.");
    }

    cameras.at(camera_label).lens_flare_properties = properties;
}

LensFlareProperties RadiationModel::getCameraLensFlareProperties(const std::string &camera_label) const {
    if (cameras.find(camera_label) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::getCameraLensFlareProperties): Camera '" + camera_label + "' does not exist.");
    }
    return cameras.at(camera_label).lens_flare_properties;
}
