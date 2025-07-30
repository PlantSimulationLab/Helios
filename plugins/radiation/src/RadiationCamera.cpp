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

#include <queue>
#include <set>
#include <stack>

#include "global.h"

using namespace helios;

void RadiationModel::addRadiationCamera(const std::string &camera_label, const std::vector<std::string> &band_label, const helios::vec3 &position, const helios::vec3 &lookat, const CameraProperties &camera_properties, uint antialiasing_samples) {

    if (camera_properties.FOV_aspect_ratio <= 0) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Field of view aspect ratio must be greater than 0.");
    } else if (antialiasing_samples == 0) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): The model requires at least 1 antialiasing sample to run.");
    } else if (camera_properties.camera_resolution.x <= 0 || camera_properties.camera_resolution.y <= 0) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera resolution must be at least 1x1.");
    } else if (camera_properties.HFOV < 0 || camera_properties.HFOV > 180.f) {
        helios_runtime_error("ERROR (RadiationModel::addRadiationCamera): Camera horizontal field of view must be between 0 and 180 degrees.");
    }

    RadiationCamera camera(camera_label, band_label, position, lookat, camera_properties, antialiasing_samples);
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
        context->loadXML("plugins/radiation/spectral_data/camera_spectral_library.xml");
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

std::vector<std::string> RadiationModel::getAllCameraLabels() {
    std::vector<std::string> labels(cameras.size());
    uint cam = 0;
    for (const auto &camera: cameras) {
        labels.at(cam) = camera.second.label;
        cam++;
    }
    return labels;
}

void RadiationModel::writeCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path, int frame, float flux_to_pixel_conversion) {

    // check if camera exists
    if (cameras.find(camera) == cameras.end()) {
        std::cout << "ERROR (RadiationModel::writeCameraImage): camera with label " << camera << " does not exist. Skipping image write for this camera." << std::endl;
        return;
    }

    if (bands.size() != 1 && bands.size() != 3) {
        helios_runtime_error("ERROR (RadiationModel::writeCameraImage): input vector of band labels (" + bands + ") should either have length of 1 (grayscale image) or length of 3 (RGB image). Skipping image write for this camera.");
    }

    std::vector<std::vector<float>> camera_data(bands.size());

    uint b = 0;
    for (const auto &band: bands) {

        // check if band exists
        if (std::find(cameras.at(camera).band_labels.begin(), cameras.at(camera).band_labels.end(), band) == cameras.at(camera).band_labels.end()) {
            std::cout << "ERROR (RadiationModel::writeCameraImage): camera " << camera << " band with label " << band << " does not exist. Skipping image write for this camera." << std::endl;
            return;
        }

        // std::string global_data_label = "camera_" + camera + "_" + bands.at(b);
        //
        // if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        //     std::cout << "ERROR (RadiationModel::writeCameraImage): image data for camera " << camera << ", band " << bands.at(b) << " has not been created. Did you run the radiation model? Skipping image write for this camera." << std::endl;
        //     return;
        // }
        //
        // context->getGlobalData(global_data_label.c_str(), camera_data.at(b));

        camera_data.at(b) = cameras.at(camera).pixel_data.at(band);

        b++;
    }

    std::string frame_str;
    if (frame >= 0) {
        frame_str = std::to_string(frame);
    }

    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeCameraImage): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeCameraImage): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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
        return;
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
}

void RadiationModel::writeNormCameraImage(const std::string &camera, const std::vector<std::string> &bands, const std::string &imagefile_base, const std::string &image_path, int frame) {
    float maxval = 0;
    // Find maximum mean value over all bands
    for (const std::string &band: bands) {
        std::string global_data_label = "camera_" + camera + "_" + band;
        if (std::find(cameras.at(camera).band_labels.begin(), cameras.at(camera).band_labels.end(), band) == cameras.at(camera).band_labels.end()) {
            std::cout << "ERROR (RadiationModel::writeNormCameraImage): camera " << camera << " band with label " << band << " does not exist. Skipping image write for this camera." << std::endl;
            return;
        } else if (!context->doesGlobalDataExist(global_data_label.c_str())) {
            std::cout << "ERROR (RadiationModel::writeNormCameraImage): image data for camera " << camera << ", band " << band << " has not been created. Did you run the radiation model? Skipping image write for this camera." << std::endl;
            return;
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

    RadiationModel::writeCameraImage(camera, bands, imagefile_base, image_path, frame);
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
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeCameraImage): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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
    cameraproperties.focal_plane_distance = calibratecamera.focal_length;
    cameraproperties.lens_diameter = calibratecamera.lens_diameter;
    cameraproperties.FOV_aspect_ratio = calibratecamera.FOV_aspect_ratio;

    std::vector<uint> UUIDs_target = cameracalibration->getColorBoardUUIDs();
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
    std::vector<uint> UUIDs_colorbd = cameracalibration->getColorBoardUUIDs();
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
    cameraproperties.focal_plane_distance = calibratecamera.focal_length;
    cameraproperties.lens_diameter = calibratecamera.lens_diameter;
    cameraproperties.FOV_aspect_ratio = calibratecamera.FOV_aspect_ratio;

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
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writePrimitiveDataLabelMap): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1;
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID, primitive_data_label.c_str())) {
                HeliosDataType datatype = context->getPrimitiveDataType(UUID, primitive_data_label.c_str());
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
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeObjectDataLabelMap): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1;
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;
            if (!context->doesPrimitiveExist(UUID)) {
                pixel_data << padvalue << " ";
                continue;
            }
            uint objID = context->getPrimitiveParentObjectID(UUID);
            if (context->doesObjectExist(objID) && context->doesObjectDataExist(objID, object_data_label.c_str())) {
                HeliosDataType datatype = context->getObjectDataType(objID, object_data_label.c_str());
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
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeDepthImageData): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeNormDepthImage): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeImageBoundingBoxes): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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
            uint ii = camera_resolution.x - i - 1;
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID, primitive_data_label.c_str())) {

                uint labeldata;

                HeliosDataType datatype = context->getPrimitiveDataType(UUID, primitive_data_label.c_str());
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
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeImageBoundingBoxes_ObjectData): Image output directory contains a filename. This argument should be the path to a directory not a file.");
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

            HeliosDataType datatype = context->getObjectDataType(objID, object_data_label.c_str());
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

// Helper function to initialize or load existing COCO JSON structure
nlohmann::json RadiationModel::initializeCOCOJson(const std::string &filename, bool append_file, const std::string &cameralabel, const helios::int2 &camera_resolution) {
    nlohmann::json coco_json;

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

        // Add the image entry (only once)
        nlohmann::json image_entry;
        image_entry["id"] = 0;
        image_entry["license"] = 1;
        image_entry["file_name"] = cameralabel + "_RGB.jpeg";
        image_entry["height"] = camera_resolution.y;
        image_entry["width"] = camera_resolution.x;
        image_entry["date_captured"] = "2025-07-28T00:00:00+00:00";
        coco_json["images"].push_back(image_entry);
    }

    return coco_json;
}

// Helper function to add category to COCO JSON if it doesn't exist
void RadiationModel::addCategoryToCOCO(nlohmann::json &coco_json, uint object_class_ID, const std::string &category_name) {
    bool category_exists = false;
    for (auto &cat: coco_json["categories"]) {
        if (cat["id"] == object_class_ID) {
            category_exists = true;
            break;
        }
    }
    if (!category_exists) {
        nlohmann::json category;
        category["id"] = object_class_ID;
        category["name"] = category_name;
        category["supercategory"] = "none";
        coco_json["categories"].push_back(category);
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
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1;
            uint UUID = pixel_UUIDs.at(j * camera_resolution.x + ii) - 1;

            if (context->doesPrimitiveExist(UUID)) {
                uint labeldata;
                bool has_data = false;

                if (use_object_data) {
                    // Object data version
                    uint objID = context->getPrimitiveParentObjectID(UUID);
                    if (objID != 0 && context->doesObjectDataExist(objID, data_label.c_str())) {
                        HeliosDataType datatype = context->getObjectDataType(objID, data_label.c_str());
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
                        HeliosDataType datatype = context->getPrimitiveDataType(UUID, data_label.c_str());
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
std::vector<std::map<std::string, std::vector<float>>> RadiationModel::generateAnnotationsFromMasks(const std::map<int, std::vector<std::vector<bool>>> &label_masks, uint object_class_ID, const helios::int2 &camera_resolution) {
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
                                annotation["image_id"] = {0.0f}; // Will be updated when writing
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

void RadiationModel::writeImageSegmentationMasks(const std::string &cameralabel, const std::string &primitive_data_label, uint object_class_ID, const std::string &imagefile_base, const std::string &image_path, bool append_file, int frame) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks): Camera '" + cameralabel + "' does not exist.");
    }

    // Check that camera pixel data exists
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }

    // Validate output path
    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeImageSegmentationMasks): Image output directory contains a filename. This argument should be the path to a directory not a file.");
    }

    // Create output filename
    std::ostringstream outfile;
    outfile << output_path;
    if (frame >= 0) {
        std::string frame_str = std::to_string(frame);
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".json";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".json";
    }

    // Generate label masks using helper function (primitive data version)
    std::map<int, std::vector<std::vector<bool>>> label_masks = generateLabelMasks(cameralabel, primitive_data_label, false);

    // Generate annotations from masks using helper function
    int2 camera_resolution = cameras.at(cameralabel).resolution;
    std::vector<std::map<std::string, std::vector<float>>> annotations = generateAnnotationsFromMasks(label_masks, object_class_ID, camera_resolution);

    // Write annotations to JSON file
    nlohmann::json coco_json = initializeCOCOJson(outfile.str(), append_file, cameralabel, camera_resolution);
    addCategoryToCOCO(coco_json, object_class_ID, primitive_data_label);

    // Find the highest existing annotation ID to avoid conflicts
    int max_annotation_id = -1;
    for (const auto &existing_ann: coco_json["annotations"]) {
        if (existing_ann["id"] > max_annotation_id) {
            max_annotation_id = existing_ann["id"];
        }
    }

    // Add new annotations
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

        coco_json["annotations"].push_back(json_annotation);
        max_annotation_id++;
    }

    // Write JSON to file
    writeCOCOJson(coco_json, outfile.str());
}

void RadiationModel::writeImageSegmentationMasks_ObjectData(const std::string &cameralabel, const std::string &object_data_label, uint object_class_ID, const std::string &imagefile_base, const std::string &image_path, bool append_file, int frame) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks_ObjectData): Camera '" + cameralabel + "' does not exist.");
    }

    // Check that camera pixel data exists
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks_ObjectData): Pixel labels for camera '" + cameralabel + "' do not exist. Was the radiation model run to generate labels?");
    }

    // Validate output path
    std::string output_path = image_path;
    if (!image_path.empty() && !validateOutputPath(output_path)) {
        helios_runtime_error("ERROR (RadiationModel::writeImageSegmentationMasks_ObjectData): Invalid image output directory '" + image_path + "'. Check that the path exists and that you have write permission.");
    } else if (!getFileName(output_path).empty()) {
        helios_runtime_error("ERROR(RadiationModel::writeImageSegmentationMasks_ObjectData): Image output directory contains a filename. This argument should be the path to a directory not a file.");
    }

    // Create output filename
    std::ostringstream outfile;
    outfile << output_path;
    if (frame >= 0) {
        std::string frame_str = std::to_string(frame);
        outfile << cameralabel << "_" << imagefile_base << "_" << std::setw(5) << std::setfill('0') << frame_str << ".json";
    } else {
        outfile << cameralabel << "_" << imagefile_base << ".json";
    }

    // Generate label masks using helper function (object data version)
    std::map<int, std::vector<std::vector<bool>>> label_masks = generateLabelMasks(cameralabel, object_data_label, true);

    // Generate annotations from masks using helper function
    int2 camera_resolution = cameras.at(cameralabel).resolution;
    std::vector<std::map<std::string, std::vector<float>>> annotations = generateAnnotationsFromMasks(label_masks, object_class_ID, camera_resolution);

    // Write annotations to JSON file
    nlohmann::json coco_json = initializeCOCOJson(outfile.str(), append_file, cameralabel, camera_resolution);
    addCategoryToCOCO(coco_json, object_class_ID, object_data_label);

    // Find the highest existing annotation ID to avoid conflicts
    int max_annotation_id = -1;
    for (const auto &existing_ann: coco_json["annotations"]) {
        if (existing_ann["id"] > max_annotation_id) {
            max_annotation_id = existing_ann["id"];
        }
    }

    // Add new annotations
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

        coco_json["annotations"].push_back(json_annotation);
        max_annotation_id++;
    }

    // Write JSON to file
    writeCOCOJson(coco_json, outfile.str());
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

std::vector<helios::vec2> RadiationModel::generateGaussianCameraResponse(float FWHM, float mu, float centrawavelength, const helios::int2 &wavebanrange) {

    // Convert FWHM to sigma
    float sigma = FWHM / (2 * std::sqrt(2 * std::log(2)));

    size_t lenspectra = wavebanrange.y - wavebanrange.x;
    std::vector<helios::vec2> cameraresponse(lenspectra);


    for (int i = 0; i < lenspectra; ++i) {
        cameraresponse.at(i).x = float(wavebanrange.x + i);
    }

    // Gaussian function
    for (size_t i = 0; i < lenspectra; ++i) {
        cameraresponse.at(i).y = centrawavelength * std::exp(-std::pow((cameraresponse.at(i).x - mu), 2) / (2 * std::pow(sigma, 2)));
    }


    return cameraresponse;
}

void RadiationModel::applyImageProcessingPipeline(const std::string &cameralabel, const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float saturation_adjustment, float brightness_adjustment,
                                                  float contrast_adjustment) {

    if (cameras.find(cameralabel) == cameras.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyImageProcessingPipeline): Camera '" + cameralabel + "' does not exist.");
    }
    RadiationCamera &camera = cameras.at(cameralabel);
    if (camera.pixel_data.size() != 3) {
        helios_runtime_error("ERROR (RadiationModel::applyImageProcessingPipeline): Image data must have 3 channels (RGB). This camera has " + std::to_string(camera.pixel_data.size()) + " channels.");
    }
    if (camera.pixel_data.find(red_band_label) == camera.pixel_data.end() || camera.pixel_data.find(green_band_label) == camera.pixel_data.end() || camera.pixel_data.find(blue_band_label) == camera.pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::applyImageProcessingPipeline): One or more specified band labels do not exist for the camera pixel data.");
    }

    camera.normalizePixels();

    camera.scaleToGreyTarget(red_band_label, green_band_label, blue_band_label, 0.18);

    camera.whiteBalance(red_band_label, green_band_label, blue_band_label, 5.0);

    camera.applyGain(red_band_label, green_band_label, blue_band_label, 0.9f);

    camera.globalHistogramEqualization(red_band_label, green_band_label, blue_band_label);

    if (saturation_adjustment != 1.f || brightness_adjustment != 1.f || contrast_adjustment != 1.f) {
        camera.adjustSBC(red_band_label, green_band_label, blue_band_label, saturation_adjustment, brightness_adjustment, contrast_adjustment);
    }

    // camera.applyCCM(red_band_label, green_band_label, blue_band_label);

    // camera.gammaCompress(red_band_label, green_band_label, blue_band_label);
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

void RadiationCamera::scaleToGreyTarget(const std::string &red_band_label, const std::string &green_band_label, const std::string &blue_band_label, float target) {

#ifdef HELIOS_DEBUG
    if (pixel_data.find(red_band_label) == pixel_data.end() || pixel_data.find(green_band_label) == pixel_data.end() || pixel_data.find(blue_band_label) == pixel_data.end()) {
        helios_runtime_error("ERROR (RadiationModel::scaleToGreyTarget): One or more specified band labels do not exist for the camera pixel data.");
    }
#endif

    const std::size_t N = resolution.x * resolution.y;
    constexpr float eps = 1e-6f;

    double logSum = 0.0;
    const auto &data_red = pixel_data.at(red_band_label);
    const auto &data_green = pixel_data.at(green_band_label);
    const auto &data_blue = pixel_data.at(blue_band_label);
    for (std::size_t i = 0; i < N; ++i) {
        logSum += std::log(luminance(data_red[i], data_green[i], data_blue[i]) + eps);
    }

    const float Llog = std::exp(logSum / static_cast<double>(N));
    const float k = target / Llog;

    for (auto &[channel, data]: pixel_data) {
        for (float &v: data) {
            v *= k;
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

    /* luminance array */
    std::vector<float> lum(N);
    for (size_t i = 0; i < N; ++i) {
        // vec3 p;
        // p.x = srgb_to_lin(data_red[i]);
        // p.y = srgb_to_lin(data_green[i]);
        // p.z = srgb_to_lin(data_blue[i]);
        vec3 p(data_red[i], data_green[i], data_blue[i]);
        lum[i] = 0.2126f * p.x + 0.7152f * p.y + 0.0722f * p.z;
    }

    /* build CDF on 2048-bin histogram */
    const int B = 2048;
    std::vector<int> hist(B, 0);
    for (float v: lum) {
        int b = int(std::clamp(v, 0.0f, 1.0f - eps) * B);
        hist[b]++;
    }
    std::vector<float> cdf(B);
    int acc = 0;
    for (int b = 0; b < B; ++b) {
        acc += hist[b];
        cdf[b] = float(acc) / float(N);
    }

    /* remap */
    for (size_t i = 0; i < N; ++i) {
        int b = int(std::clamp(lum[i], 0.0f, 1.0f - eps) * B);

        constexpr float k = 0.2f; // how far to pull towards equalised value  (0.2–0.3 OK)
        constexpr float cs = 0.2f; // S-curve strength   (0.4–0.7 recommended)

        float Yeq = cdf[b]; // equalised luminance  ∈[0,1]
        float Ynew = (1.0f - k) * lum[i] + k * Yeq; // partial equalisation

        /* symmetric S-curve centred at 0.5  :  y = ½ + (x–½)*(1+cs–2·cs·|x–½|)   */
        float t = Ynew - 0.5f;
        Ynew = 0.5f + t * (1.0f + cs - 2.0f * cs * std::fabs(t));

        float scale = lum[i] > 0.0f ? Ynew / lum[i] : 0;
        data_red[i] = lin_to_srgb(data_red[i] * scale);
        data_green[i] = lin_to_srgb(data_green[i] * scale);
        data_blue[i] = lin_to_srgb(data_blue[i] * scale);
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
