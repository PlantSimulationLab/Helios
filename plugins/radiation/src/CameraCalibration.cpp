/** \file "CameraCalibration.cpp" Routines for performing synthetic radiation camera calibration.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#include "CameraCalibration.h"

using namespace helios;

CameraCalibration::CameraCalibration(helios::Context *context) : context(context) {
}

std::vector<uint> CameraCalibration::addCheckerboard(const helios::int2 &boardsidesize, const float &patchsize, const helios::vec3 &centrelocation, const helios::vec3 &rotationrad, bool firstblack) {
    helios::vec2 patchfullsize = make_vec2(patchsize, patchsize); // size of each patch
    std::vector<float> bwv(2, 0);
    float bw;
    if (firstblack) {
        bwv[0] = 1;
    } else {
        bwv[1] = 1;
    }
    std::vector<uint> UUIDs;
    for (int inpr = 0; inpr < boardsidesize.x; inpr += 1) {
        for (int inpc = 0; inpc < boardsidesize.y; inpc += 1) {
            if ((inpr % 2 == 0 & inpc % 2 == 0) | (inpr % 2 != 0 & inpc % 2 != 0)) { // black - white - black - white -...
                bw = bwv[0];
            } else {
                bw = bwv[1];
            }
            helios::RGBcolor binarycolor(bw, bw, bw);

            float xp = 0 - patchsize * (float(boardsidesize.x) - 1) / 2 + patchsize * float(inpr);
            float yp = 0 - patchsize * (float(boardsidesize.y) - 1) / 2 + patchsize * float(inpc);

            uint UUID = context->addPatch(make_vec3(xp, yp, 0), patchfullsize, nullrotation, binarycolor);

            if (bw == 1) {
                UUIDs_white.push_back(UUID);
                context->setPrimitiveData(UUID, "reflectivity_spectrum", "white");
            } else {
                UUIDs_black.push_back(UUID);
                context->setPrimitiveData(UUID, "reflectivity_spectrum", "");
            }

            context->setPrimitiveData(UUID, "transmissivity_spectrum", "");
            context->setPrimitiveData(UUID, "twosided_flag", uint(0));
            UUIDs.push_back(UUID);
        }
    }
    std::vector<helios::vec2> whitespectra(2200);
    for (int i = 0; i < whitespectra.size(); i++) {
        whitespectra.at(i).x = 301 + i;
        whitespectra.at(i).y = 1;
    }
    context->setGlobalData("white", whitespectra);
    context->rotatePrimitive(UUIDs, rotationrad.x, make_vec3(1, 0, 0));
    context->rotatePrimitive(UUIDs, rotationrad.y, make_vec3(0, 1, 0));
    context->rotatePrimitive(UUIDs, rotationrad.z, make_vec3(0, 0, 1));
    context->translatePrimitive(UUIDs, centrelocation);
    return UUIDs;
}

std::vector<uint> CameraCalibration::addDefaultCheckerboard(const helios::vec3 &centrelocation, const helios::vec3 &rotationrad) {
    helios::int2 boardsidesize = make_int2(10, 7);
    float patchsize = 0.029;
    bool firstblack = true;
    std::vector<uint> UUID_checkboard = CameraCalibration::addCheckerboard(boardsidesize, patchsize, centrelocation, rotationrad, firstblack);
    return UUID_checkboard;
}

std::vector<uint> CameraCalibration::addColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad, const std::vector<std::vector<helios::RGBcolor>> &colorassignment,
                                                   const std::vector<std::vector<std::string>> &spectrumassignment) {

    uint Nrow;
    uint Ncol;
    if (!colorassignment.empty()) {
        Nrow = colorassignment.size();
        Ncol = colorassignment.back().size();
    } else if (!spectrumassignment.empty()) {
        Nrow = spectrumassignment.size();
        Ncol = spectrumassignment.back().size();
    } else {
        helios_runtime_error("ERROR (CameraCalibration::addColorboard): No color or spectrum assignment provided.");
    }

    std::vector<uint> UUIDs;
    uint UUID;
    int patch_index = 0; // Linear patch index for labeling
    for (int irow = 0; irow < Nrow; irow++) {
        for (int icolumn = 0; icolumn < Ncol; icolumn++) {

            float xp = centrelocation.x - patchsize * (float(Ncol) - 1) / 2.f + patchsize * float(icolumn);
            float yp = centrelocation.y + patchsize * (float(Nrow) - 1) / 2.f - patchsize * float(irow);

            if (!colorassignment.empty()) {
                if (irow >= colorassignment.size() || icolumn >= colorassignment.at(irow).size()) {
                    helios_runtime_error("ERROR (CameraCalibration::addColorboard): Dimensions of color assignment array are not consistent. This should be a square matrix.");
                }
                UUID = context->addPatch(make_vec3(xp, yp, centrelocation.z), make_vec2(patchsize, patchsize), nullrotation, colorassignment.at(irow).at(icolumn));
            } else {
                UUID = context->addPatch(make_vec3(xp, yp, centrelocation.z), make_vec2(patchsize, patchsize), nullrotation);
            }
            context->setPrimitiveData(UUID, "twosided_flag", uint(0));
            UUIDs.push_back(UUID);

            if (!spectrumassignment.empty()) {
                if (irow >= spectrumassignment.size() || icolumn >= spectrumassignment.at(irow).size()) {
                    helios_runtime_error("ERROR (CameraCalibration::addColorboard): Dimensions of spectrum assignment array are not consistent. This should be a square matrix.");
                } else if (!context->doesGlobalDataExist(spectrumassignment.at(irow).at(icolumn).c_str())) {
                    helios_runtime_error("ERROR (CameraCalibration::addColorboard): Spectrum assignment label of "
                                         "" +
                                         spectrumassignment.at(irow).at(icolumn) +
                                         ""
                                         " does not exist in global data.");
                }
                context->setPrimitiveData(UUID, "reflectivity_spectrum", spectrumassignment.at(irow).at(icolumn));
            }

            patch_index++;
        }
    }

    context->rotatePrimitive(UUIDs, rotationrad.x, "x"); // rotate color board
    context->rotatePrimitive(UUIDs, rotationrad.y, "y");
    context->rotatePrimitive(UUIDs, rotationrad.z, "z");
    // Note: This version does not track colorboards in the map (no type specified)
    return UUIDs;
}

std::vector<uint> CameraCalibration::addColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad, const std::vector<std::vector<helios::RGBcolor>> &colorassignment,
                                                   const std::vector<std::vector<std::string>> &spectrumassignment, const std::string &colorboard_type) {

    uint Nrow = colorassignment.size();
    uint Ncol = colorassignment.back().size();

    if (spectrumassignment.size() != Nrow || spectrumassignment.back().size() != Ncol) {
        helios_runtime_error("ERROR (CameraCalibration::addColorboard): Color and spectrum assignment dimensions must match.");
    }

    std::vector<uint> UUIDs;
    uint UUID;
    int patch_index = 0; // Linear patch index for labeling
    for (int irow = 0; irow < Nrow; irow++) {
        for (int icolumn = 0; icolumn < Ncol; icolumn++) {

            float xp = centrelocation.x - patchsize * (float(Ncol) - 1) / 2.f + patchsize * float(icolumn);
            float yp = centrelocation.y + patchsize * (float(Nrow) - 1) / 2.f - patchsize * float(irow);

            if (irow >= colorassignment.size() || icolumn >= colorassignment.at(irow).size()) {
                helios_runtime_error("ERROR (CameraCalibration::addColorboard): Dimensions of color assignment array are not consistent. This should be a square matrix.");
            }
            UUID = context->addPatch(make_vec3(xp, yp, centrelocation.z), make_vec2(patchsize, patchsize), nullrotation, colorassignment.at(irow).at(icolumn));
            context->setPrimitiveData(UUID, "twosided_flag", uint(0));
            UUIDs.push_back(UUID);

            if (irow >= spectrumassignment.size() || icolumn >= spectrumassignment.at(irow).size()) {
                helios_runtime_error("ERROR (CameraCalibration::addColorboard): Dimensions of spectrum assignment array are not consistent. This should be a square matrix.");
            } else if (!context->doesGlobalDataExist(spectrumassignment.at(irow).at(icolumn).c_str())) {
                helios_runtime_error("ERROR (CameraCalibration::addColorboard): Spectrum assignment label of " + spectrumassignment.at(irow).at(icolumn) + " does not exist in global data.");
            }
            context->setPrimitiveData(UUID, "reflectivity_spectrum", spectrumassignment.at(irow).at(icolumn));

            // Add colorboard type and patch index labeling for auto-calibration
            std::string colorboard_label = "colorboard_" + colorboard_type;
            context->setPrimitiveData(UUID, colorboard_label.c_str(), uint(patch_index));

            patch_index++;
        }
    }

    context->rotatePrimitive(UUIDs, rotationrad.x, "x"); // rotate color board
    context->rotatePrimitive(UUIDs, rotationrad.y, "y");
    context->rotatePrimitive(UUIDs, rotationrad.z, "z");
    // Store colorboard UUIDs in map by type
    UUIDs_colorboards[colorboard_type] = UUIDs;
    return UUIDs;
}

// void CameraCalibration::setColorboardReflectivity(const uint &UUID, const std::string &filename, const std::string &labelname) {
//     //\todo this needs to be updated
//     std::vector<vec2> spectraldata;
//     CameraCalibration::loadXMLlabeldata(filename,labelname,spectraldata);
//     context->setPrimitiveData(UUID, "reflectivity_spectrum", labelname);
//     context->setPrimitiveData(UUID, "reflectivity_spectrum_raw", labelname+"_raw");
//     context->setPrimitiveData(UUID, "transmissivity_spectrum", "");
//     context->setPrimitiveData(UUID, "twosided_flag", uint(0) );
//     context->setGlobalData(labelname.c_str(),spectraldata);
//     context->setGlobalData((labelname+"_raw").c_str(),spectraldata);
// }

std::vector<uint> CameraCalibration::addDefaultColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad) {

    return addDGKColorboard(centrelocation, patchsize, rotationrad);
}

std::vector<uint> CameraCalibration::addDGKColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad) {

    // Check if a DGK colorboard already exists
    if (UUIDs_colorboards.find("DGK") != UUIDs_colorboards.end()) {
        context->deletePrimitive(UUIDs_colorboards["DGK"]);
        std::cout << "WARNING (CameraCalibration::addDGKColorboard): Replacing existing DGK colorboard." << std::endl;
    }

    context->loadXML(helios::resolvePluginAsset("radiation", "spectral_data/color_board/DGK_DKK_colorboard.xml").string().c_str(), true);

    assert(context->doesGlobalDataExist("ColorReference_DGK_01"));

    return CameraCalibration::addColorboard(centrelocation, patchsize, rotationrad, colorassignment_DGK, spectrumassignment_DGK, "DGK");
}

std::vector<uint> CameraCalibration::addCalibriteColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad) {

    // Check if a Calibrite colorboard already exists
    if (UUIDs_colorboards.find("Calibrite") != UUIDs_colorboards.end()) {
        context->deletePrimitive(UUIDs_colorboards["Calibrite"]);
        std::cout << "WARNING (CameraCalibration::addCalibriteColorboard): Replacing existing Calibrite colorboard." << std::endl;
    }

    context->loadXML(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Calibrite_ColorChecker_Classic_colorboard.xml").string().c_str(), true);

    assert(context->doesGlobalDataExist("ColorReference_Calibrite_01"));

    return CameraCalibration::addColorboard(centrelocation, patchsize, rotationrad, colorassignment_Calibrite, spectrumassignment_Calibrite, "Calibrite");
}

std::vector<uint> CameraCalibration::addSpyderCHECKRColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad) {

    // Check if a SpyderCHECKR colorboard already exists
    if (UUIDs_colorboards.find("SpyderCHECKR") != UUIDs_colorboards.end()) {
        context->deletePrimitive(UUIDs_colorboards["SpyderCHECKR"]);
        std::cout << "WARNING (CameraCalibration::addSpyderCHECKRColorboard): Replacing existing SpyderCHECKR colorboard." << std::endl;
    }

    context->loadXML(helios::resolvePluginAsset("radiation", "spectral_data/color_board/Datacolor_SpyderCHECKR_24_colorboard.xml").string().c_str(), true);

    assert(context->doesGlobalDataExist("ColorReference_SpyderCHECKR_01"));

    return CameraCalibration::addColorboard(centrelocation, patchsize, rotationrad, colorassignment_SpyderCHECKR, spectrumassignment_SpyderCHECKR, "SpyderCHECKR");
}

std::vector<uint> CameraCalibration::getAllColorBoardUUIDs() const {
    std::vector<uint> all_UUIDs;
    for (const auto &[type, UUIDs]: UUIDs_colorboards) {
        all_UUIDs.insert(all_UUIDs.end(), UUIDs.begin(), UUIDs.end());
    }
    return all_UUIDs;
}

bool CameraCalibration::writeSpectralXMLfile(const std::string &filename, const std::string &note, const std::string &label, std::vector<vec2> *spectrum) {

    std::ofstream newspectraldata(filename);

    if (newspectraldata.is_open()) {

        newspectraldata << "<helios>\n\n";
        newspectraldata << "\t<!-- ";
        newspectraldata << note;
        newspectraldata << " -->\n";
        newspectraldata << "\t<globaldata_vec2 label=\"";
        newspectraldata << label;
        newspectraldata << "\">";

        for (int i = 0; i < spectrum->size(); ++i) {
            newspectraldata << "\n\t\t";
            newspectraldata << spectrum->at(i).x;
            newspectraldata << " ";
            newspectraldata << std::to_string(spectrum->at(i).y);
        }
        //            spectrum->clear();
        newspectraldata << "\n\t</globaldata_vec2>\n";
        newspectraldata << "\n</helios>";
        newspectraldata.close();
        return true;
    }

    else {
        std::cerr << "\n(CameraCalibration::writeSpectralXMLfile) Unable to open file";
        return false;
    }
}

// Load XML file and save data in spectral vectors containing both wavelengths and spectral values
bool CameraCalibration::loadXMLlabeldata(const std::string &filename, const std::string &labelname, std::vector<vec2> &spectraldata) {

    Context context_temporary;
    context_temporary.loadXML(filename.c_str(), true);

    if (context_temporary.doesGlobalDataExist(labelname.c_str())) {
        context_temporary.getGlobalData(labelname.c_str(), spectraldata);
        return true;
    } else {
        std::cerr << "\n(CameraCalibration::loadXMLlabeldata) Cannot find the label " << labelname << " in XML file:" << filename;
        return false;
    }
}

float CameraCalibration::GradientDescent(std::vector<std::vector<float>> *expandedcameraspectra, const std::vector<std::vector<float>> &expandedconstinput, const float &learningrate, const std::vector<std::vector<float>> &truevalues) {

    size_t boardnumber = expandedconstinput.size();
    size_t wavelengthnumber = expandedconstinput.at(0).size();
    size_t bandnumber = expandedcameraspectra->size();

    float iloss = 0;
    for (int iband = 0; iband < expandedcameraspectra->size(); iband++) {

        // Calculate errors
        std::vector<float> output(boardnumber, 0);
        std::vector<float> errors(boardnumber);
        for (int iboardn = 0; iboardn < boardnumber; ++iboardn) {
            for (int ispecn = 0; ispecn < wavelengthnumber; ++ispecn) {
                output.at(iboardn) += expandedcameraspectra->at(iband).at(ispecn) * expandedconstinput.at(iboardn).at(ispecn);
            }
            errors.at(iboardn) = truevalues.at(iband).at(iboardn) - output.at(iboardn);

            // Calculate root mean square error as loss function
            iloss += (errors.at(iboardn) * errors.at(iboardn)) / float(boardnumber) / float(bandnumber);
        }

        // Update extended spectrum
        std::vector<float> despectrum(wavelengthnumber);
        for (int ispecn = 0; ispecn < wavelengthnumber; ++ispecn) {
            for (int iboardn = 0; iboardn < boardnumber; ++iboardn) {
                despectrum.at(ispecn) += errors.at(iboardn) * expandedconstinput.at(iboardn).at(ispecn);
            }
            expandedcameraspectra->at(iband).at(ispecn) += learningrate * despectrum.at(ispecn);

            // Non-negative constrain
            if (expandedcameraspectra->at(iband).at(ispecn) < 0) {
                expandedcameraspectra->at(iband).at(ispecn) -= learningrate * despectrum.at(ispecn);
            }
        }
    }
    return iloss;
}

std::vector<float> CameraCalibration::expandSpectrum(const std::vector<helios::vec2> &targetspectrum, float normvalue = 1) {

    std::vector<float> extendedspectrum;
    extendedspectrum.reserve(targetspectrum.size());

    for (vec2 spectralvalue: targetspectrum) {
        extendedspectrum.push_back(spectralvalue.y / normvalue);
    }

    extendedspectrum.insert(extendedspectrum.end() - 1, extendedspectrum.begin() + 1, extendedspectrum.end() - 1);
    return extendedspectrum;
}

static float normalizevalue(std::vector<std::vector<helios::vec2>> cameraresponsespectra, const std::map<uint, std::vector<vec2>> &simulatedinputspectra) {

    // Find the maximum value of the simulated input spectra multiplied by the camera response spectra
    float normvalue = 0;
    for (const auto &cameraspectrum: cameraresponsespectra) {
        for (const auto &inputspectrum: simulatedinputspectra) {
            float outputvalue = 0;
            for (int iwave = 1; iwave < inputspectrum.second.size() - 1; iwave++) {
                outputvalue += inputspectrum.second.at(iwave).y * cameraspectrum.at(iwave).y * 2;
            }
            outputvalue += inputspectrum.second.at(0).y * cameraspectrum.at(0).y;
            outputvalue += inputspectrum.second.back().y * cameraspectrum.back().y;
            if (outputvalue > normvalue) {
                normvalue = outputvalue;
            }
        }
    }
    return normvalue;
}


std::vector<float> CameraCalibration::updateCameraResponseSpectra(const std::vector<std::string> &camerareponselabels, const std::string &label, const std::map<uint, std::vector<vec2>> &simulatedinputspectra,
                                                                  const std::vector<std::vector<float>> &truevalues) {

    float learningrate = responseupdateparameters.learningrate;
    int maxiteration = responseupdateparameters.maxiteration;
    float minloss = responseupdateparameters.minloss;
    std::vector<float> camerarescales = responseupdateparameters.camerarescales;

    std::vector<std::vector<helios::vec2>> cameraresponsespectra;
    for (std::string cameraresponselabel: camerareponselabels) {
        std::vector<vec2> cameraresponsespectrum;
        context->getGlobalData(cameraresponselabel.c_str(), cameraresponsespectrum);
        cameraresponsespectra.push_back(cameraresponsespectrum);
    }

    // Get the highest value of color board by using the original camera response used for normalization
    float normvalue = normalizevalue(cameraresponsespectra, simulatedinputspectra);

    std::vector<std::vector<float>> expandedcameraspectra;
    uint bandsnumber = cameraresponsespectra.size();
    for (const auto &cameraspectrum: cameraresponsespectra) {
        expandedcameraspectra.push_back(CameraCalibration::expandSpectrum(cameraspectrum, normvalue));
    }

    std::vector<std::vector<float>> expandedinputspectra;
    expandedinputspectra.reserve(simulatedinputspectra.size());
    for (const auto &inputspectrum: simulatedinputspectra) {
        expandedinputspectra.push_back(CameraCalibration::expandSpectrum(inputspectrum.second));
    }

    // Update expanded camera response spectra
    std::vector<float> loss;
    float initialloss;
    loss.reserve(maxiteration);
    float stopiteration = maxiteration;
    for (int iloop = 0; iloop < maxiteration; ++iloop) {
        float iloss = CameraCalibration::GradientDescent(&expandedcameraspectra, expandedinputspectra, learningrate, truevalues) / float(bandsnumber);
        loss.push_back(iloss);
        if (iloss < minloss) {
            stopiteration = iloop;
            break;
        }
        // Automatically change learning rate
        if (iloop == 0) {
            initialloss = iloss;
        } else if (iloss > 0.5 * initialloss) {
            learningrate = learningrate * 2;
        }
    }

    // Get the calibrated camera response spectra

    for (int iband = 0; iband < camerareponselabels.size(); iband++) {
        calibratedcameraspectra[camerareponselabels.at(iband)] = cameraresponsespectra.at(iband);
        for (int ispec = 0; ispec < cameraresponsespectra.at(iband).size() - 1; ispec++) {
            calibratedcameraspectra[camerareponselabels.at(iband)].at(ispec).y = expandedcameraspectra.at(iband).at(ispec) * camerarescales.at(iband);
        }
        calibratedcameraspectra[camerareponselabels.at(iband)].back().y = expandedcameraspectra.at(iband).back() * camerarescales.at(iband);
        std::string calibratedlabel = label + "_" + camerareponselabels.at(iband);
        context->setGlobalData(calibratedlabel.c_str(), calibratedcameraspectra.at(camerareponselabels.at(iband)));
    }

    // Calculate the final loss
    float iloss = CameraCalibration::GradientDescent(&expandedcameraspectra, expandedinputspectra, learningrate, truevalues) / float(bandsnumber);
    std::cout << "The final loss after " << stopiteration << " iteration is: " << iloss << std::endl;
    loss.push_back(iloss);
    return loss;
}

void CameraCalibration::writeCalibratedCameraResponses(const std::vector<std::string> &camerareponselabels, const std::string &calibratemark, float scale) {
    // write the calibrated camera response spectra in xml files and set them as global data

    for (const std::string &cameraresponselabel: camerareponselabels) {
        std::vector<vec2> cameraresponsespectrum = calibratedcameraspectra[cameraresponselabel];
        std::string calibratedlabel = calibratemark + "_" + cameraresponselabel;
        for (int ispec = 0; ispec < cameraresponsespectrum.size(); ispec++) {
            cameraresponsespectrum.at(ispec).y = cameraresponsespectrum.at(ispec).y * scale;
        }
        context->setGlobalData(calibratedlabel.c_str(), cameraresponsespectrum);
        CameraCalibration::writeSpectralXMLfile(calibratedlabel + ".xml", "", calibratedlabel, &cameraresponsespectrum);
    }
}

void CameraCalibration::distortImage(const std::string &cameralabel, const std::vector<std::string> &bandlabels, const helios::vec2 &focalxy, std::vector<double> &distCoeffs, helios::int2 cameraresolution) {

    helios::int2 camerareoslutionR = cameraresolution;

    // Original image dimensions
    int cols = cameraresolution.x;
    int rows = cameraresolution.y;

    //    float PPointsRatiox =1.052f;
    //    float PPointsRatioy =0.999f;

    // Distorted image dimensions diff
    int cols_dif = (cols - camerareoslutionR.x) / 2;
    int rows_dif = (rows - camerareoslutionR.y) / 2;

    for (int ib = 0; ib < bandlabels.size(); ib++) {
        std::string global_data_label = "camera_" + cameralabel + "_" + bandlabels.at(ib);
        std::vector<float> cameradata;

        context->getGlobalData(global_data_label.c_str(), cameradata);

        std::vector<float> distorted_cameradata(camerareoslutionR.x * camerareoslutionR.y, 0);

        // Compute the undistorted image
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < cols; i++) {
                // Compute the undistorted pixel coordinates
                double x = (i - cols / 2) / focalxy.x;
                double y = (j - rows / 2) / focalxy.y;
                // Apply distortion
                double r2 = x * x + y * y;
                double xDistorted = x * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[2] * x * y + distCoeffs[3] * (r2 + 2 * x * x);
                double yDistorted = y * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[3] * x * y + distCoeffs[2] * (r2 + 2 * y * y);

                // Compute the distorted pixel coordinates
                int xPixel = int(round(xDistorted * focalxy.x + cols / 2));
                int yPixel = int(round(yDistorted * focalxy.y + rows / 2));

                // Set the distorted pixel value
                if (xPixel >= cols_dif && xPixel < cols - cols_dif && yPixel >= rows_dif && yPixel < rows - rows_dif) {
                    int xPos = xPixel - cols_dif;
                    int yPos = yPixel - rows_dif;
                    if (yPos * camerareoslutionR.x + xPos < distorted_cameradata.size()) {
                        distorted_cameradata.at(yPos * camerareoslutionR.x + xPos) = cameradata.at(j * cameraresolution.x + i);
                    }
                }
            }
        }
        context->setGlobalData(global_data_label.c_str(), distorted_cameradata);
    }
}

void undistortImage(std::vector<std::vector<unsigned char>> &image, std::vector<std::vector<unsigned char>> &undistorted, std::vector<std::vector<double>> &K, std::vector<double> &distCoeffs) {
    // Image dimensions
    int rows = image.size();
    int cols = image[0].size();

    // Compute the undistorted image
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // Compute the undistorted pixel coordinates
            double x = (c - K[0][2]) / K[0][0];
            double y = (r - K[1][2]) / K[1][1];

            // Apply distortion
            double r2 = x * x + y * y;
            double xDistorted = x * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[2] * x * y + distCoeffs[3] * (r2 + 2 * x * x);
            double yDistorted = y * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[3] * x * y + distCoeffs[2] * (r2 + 2 * y * y);

            // Compute the distorted pixel coordinates
            int xPixel = round(xDistorted * K[0][0] + K[0][2]);
            int yPixel = round(yDistorted * K[1][1] + K[1][2]);

            // Set the undistorted pixel value
            if (xPixel >= 0 && xPixel < cols && yPixel >= 0 && yPixel < rows) {
                undistorted[r][c] = image[yPixel][xPixel];
            }
        }
    }
}

static void wavelengthboundary(float &lowwavelength, float &highwavelength, const std::vector<vec2> &spectrum) {

    if (spectrum.back().x < highwavelength) {
        highwavelength = spectrum.back().x;
    }
    if (spectrum.at(0).x > lowwavelength) {
        lowwavelength = spectrum.at(0).x;
    }
}

void CameraCalibration::preprocessSpectra(const std::vector<std::string> &sourcelabels, const std::vector<std::string> &cameralabels, std::vector<std::string> &objectlabels, vec2 &wavelengthrange, const std::string &targetlabel) {

    std::map<std::string, std::map<std::string, std::vector<vec2>>> allspectra;

    helios::WarningAggregator warnings;

    // Extract and check source spectra from global data
    std::map<std::string, std::vector<vec2>> Source_spectra;
    for (const std::string &sourcelable: sourcelabels) {
        if (context->doesGlobalDataExist(sourcelable.c_str())) {
            std::vector<vec2> Source_spectrum;
            context->getGlobalData(sourcelable.c_str(), Source_spectrum);
            Source_spectra.emplace(sourcelable, Source_spectrum);
            wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Source_spectrum);
        } else {
            warnings.addWarning("missing_source_spectrum", "Source (" + sourcelable + ") does not exist in global data");
        }
    }
    allspectra.emplace("source", Source_spectra);

    // Extract and check camera spectra from global data
    std::map<std::string, std::vector<vec2>> Camera_spectra;
    for (const std::string &cameralabel: cameralabels) {
        if (context->doesGlobalDataExist(cameralabel.c_str())) {
            std::vector<vec2> Camera_spectrum;
            context->getGlobalData(cameralabel.c_str(), Camera_spectrum);
            Camera_spectra.emplace(cameralabel, Camera_spectrum);
            wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Camera_spectrum);
        } else {
            warnings.addWarning("missing_camera_spectrum", "Camera (" + cameralabel + ") does not exist in global data");
        }
    }
    allspectra.emplace("camera", Camera_spectra);

    // Extract and check object spectra from global data
    std::map<std::string, std::vector<vec2>> Object_spectra;
    if (!objectlabels.empty()) {
        for (const std::string &objectlable: objectlabels) {
            if (context->doesGlobalDataExist(objectlable.c_str())) {
                std::vector<vec2> Object_spectrum;
                context->getGlobalData(objectlable.c_str(), Object_spectrum);
                Object_spectra.emplace(objectlable, Object_spectrum);
                wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Object_spectrum);
            } else {
                warnings.addWarning("missing_object_spectrum", "Object (" + objectlable + ") does not exist in global data");
            }
        }
    }

    // Check if object spectra in global data has been added to UUIDs but are not in the provided object labels;
    std::vector<uint> exist_UUIDs = context->getAllUUIDs();
    for (uint UUID: exist_UUIDs) {
        if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
            std::string spectralreflectivitylabel;
            context->getPrimitiveData(UUID, "reflectivity_spectrum", spectralreflectivitylabel);
            if (context->doesGlobalDataExist(spectralreflectivitylabel.c_str())) {
                if (std::find(objectlabels.begin(), objectlabels.end(), spectralreflectivitylabel) == objectlabels.end()) {
                    objectlabels.push_back(spectralreflectivitylabel);
                    std::vector<vec2> Object_spectrum;
                    context->getGlobalData(spectralreflectivitylabel.c_str(), Object_spectrum);
                    Object_spectra.emplace(spectralreflectivitylabel, Object_spectrum);
                    wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Object_spectrum);
                }
            }
        }

        if (context->doesPrimitiveDataExist(UUID, "transmissivity_spectrum")) {
            std::string spectraltransmissivitylabel;
            context->getPrimitiveData(UUID, "transmissivity_spectrum", spectraltransmissivitylabel);
            if (context->doesGlobalDataExist(spectraltransmissivitylabel.c_str())) {
                if (std::find(objectlabels.begin(), objectlabels.end(), spectraltransmissivitylabel) == objectlabels.end()) {
                    objectlabels.push_back(spectraltransmissivitylabel);
                    warnings.addWarning("spectrum_not_in_object_labels", "Spectrum (" + spectraltransmissivitylabel + ") has been added to UUID (" + std::to_string(UUID) + ") but is not in the provided object labels");
                    std::vector<vec2> Object_spectrum;
                    context->getGlobalData(spectraltransmissivitylabel.c_str(), Object_spectrum);
                    Object_spectra.emplace(spectraltransmissivitylabel, Object_spectrum);
                    wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Object_spectrum);
                }
            }
        }
    }
    allspectra.emplace("object", Object_spectra);

    warnings.report(std::cerr);

    // interpolate spectra
    // set unifying wavelength resolution for all spectra according to resolution of target spectrum
    std::vector<vec2> target_spectrum;
    if (targetlabel.empty()) {
        target_spectrum = allspectra.at("object").at(objectlabels.at(0));
    } else if (std::find(objectlabels.begin(), objectlabels.end(), targetlabel) == objectlabels.end()) {
        std::cout << "WARNING (CameraCalibration::preprocessSpectra()): target label (" << targetlabel << ") is not a member of object labels" << std::endl;
        target_spectrum = allspectra.at("object").at(objectlabels.at(0));
    } else {
        target_spectrum = allspectra.at("object").at(targetlabel);
    }

    for (const auto &spectralgrouppair: allspectra) {
        for (const auto &spectrumpair: spectralgrouppair.second) {
            std::vector<vec2> cal_spectrum;
            for (auto ispectralvalue: target_spectrum) {
                if (ispectralvalue.x > wavelengthrange.y) {
                    context->setGlobalData(spectrumpair.first.c_str(), cal_spectrum);
                    break;
                }
                if (ispectralvalue.x >= wavelengthrange.x) {
                    cal_spectrum.push_back(make_vec2(ispectralvalue.x, interp1(spectrumpair.second, ispectralvalue.x)));
                }
            }
            allspectra[spectralgrouppair.first][spectrumpair.first] = cal_spectrum;
        }
    }
    processedspectra = allspectra;

    // store wavelengths into global data
    std::vector<float> wavelengths;
    for (auto ispectralvalue: target_spectrum) {
        if (ispectralvalue.x > wavelengthrange.y || ispectralvalue.x == target_spectrum.back().x) {
            context->setGlobalData("wavelengths", wavelengths);
            break;
        }
        if (ispectralvalue.x >= wavelengthrange.x) {
            wavelengths.push_back(ispectralvalue.x);
        }
    }
}

float CameraCalibration::getCameraResponseScale(const std::string &cameralabel, const helios::int2 cameraresolution, const std::vector<std::string> &bandlabels, const std::vector<std::vector<float>> &truevalues) {

    std::vector<uint> camera_UUIDs;
    std::string global_UUID_label = "camera_" + cameralabel + "_pixel_UUID";
    context->getGlobalData(global_UUID_label.c_str(), camera_UUIDs);

    // Collect all colorboard UUIDs from all colorboards
    std::vector<uint> all_colorboard_UUIDs;
    for (const auto &[type, UUIDs]: UUIDs_colorboards) {
        all_colorboard_UUIDs.insert(all_colorboard_UUIDs.end(), UUIDs.begin(), UUIDs.end());
    }

    float dotsimreal = 0;
    float dotsimsim = 0;
    for (int ib = 0; ib < bandlabels.size(); ib++) {

        std::vector<float> camera_data;
        std::string global_data_label = "camera_" + cameralabel + "_" + bandlabels.at(ib);
        context->getGlobalData(global_data_label.c_str(), camera_data);

        for (int icu = 0; icu < all_colorboard_UUIDs.size(); icu++) {

            float count = 0;
            float sum = 0;
            for (uint j = 0; j < cameraresolution.y; j++) {
                for (uint i = 0; i < cameraresolution.x; i++) {
                    uint UUID = camera_UUIDs.at(j * cameraresolution.x + i) - 1;
                    if (UUID == all_colorboard_UUIDs.at(icu)) {
                        float value = camera_data.at(j * cameraresolution.x + i);
                        sum += value;
                        count += 1;
                    }
                }
            }

            float sim_cv = sum / count;
            float real_cv = truevalues.at(ib).at(icu);
            dotsimreal += (sim_cv * real_cv);
            dotsimsim += (sim_cv * sim_cv);
        }
    }
    float fluxscale = dotsimreal / dotsimsim;
    return fluxscale;
}

std::vector<uint> CameraCalibration::readROMCCanopy() {
    std::filesystem::path file_path = context->resolveFilePath("plugins/radiation/spectral_data/external_data/HET01_UNI_scene.def");
    std::ifstream inputFile(file_path);
    std::vector<uint> iCUUIDs;
    float idata;
    // read the data from the file
    while (!inputFile.eof()) {
        std::vector<float> poslf;
        std::vector<float> rotatelf_cos;
        std::vector<float> rotatelf_sc(2);
        auto widthlengthlf = float(sqrt(0.01f * M_PI));
        for (int i = 0; i < 7; i++) {
            inputFile >> idata;

            if (i > 0 && i < 4) {
                poslf.push_back(idata);
            } else if (i > 3) {
                rotatelf_cos.push_back(idata);
            }
        }

        rotatelf_sc.at(0) = float(asin(rotatelf_cos.at(2)));
        rotatelf_sc.at(1) = float(atan2(rotatelf_cos.at(1), rotatelf_cos.at(0)));
        if (rotatelf_sc.at(1) < 0) {
            rotatelf_sc.at(1) += 2 * M_PI;
        }
        vec3 iposlf = make_vec3(poslf.at(0), poslf.at(1), poslf.at(2));
        iCUUIDs.push_back(context->addPatch(iposlf, vec2(widthlengthlf, widthlengthlf), make_SphericalCoord(float(0.5 * M_PI + rotatelf_sc.at(0)), rotatelf_sc.at(1))));
    }
    inputFile.close();
    return iCUUIDs;
}

// Auto-calibration implementation methods

std::vector<std::string> CameraCalibration::detectColorBoardTypes() const {
    // Check for each colorboard type by looking for primitive data
    std::vector<uint> all_UUIDs = context->getAllUUIDs();
    std::vector<std::string> detected_types;
    bool found_DGK = false, found_Calibrite = false, found_SpyderCHECKR = false;

    for (uint UUID: all_UUIDs) {
        if (!found_DGK && context->doesPrimitiveDataExist(UUID, "colorboard_DGK")) {
            detected_types.push_back("DGK");
            found_DGK = true;
        }
        if (!found_Calibrite && context->doesPrimitiveDataExist(UUID, "colorboard_Calibrite")) {
            detected_types.push_back("Calibrite");
            found_Calibrite = true;
        }
        if (!found_SpyderCHECKR && context->doesPrimitiveDataExist(UUID, "colorboard_SpyderCHECKR")) {
            detected_types.push_back("SpyderCHECKR");
            found_SpyderCHECKR = true;
        }
        // Early exit if all types have been found
        if (found_DGK && found_Calibrite && found_SpyderCHECKR) {
            break;
        }
    }

    if (detected_types.empty()) {
        helios_runtime_error("ERROR (CameraCalibration::detectColorBoardTypes): No colorboard detected in the scene. Make sure to add a colorboard using addDGKColorboard(), addCalibriteColorboard(), or addSpyderCHECKRColorboard().");
    }

    return detected_types;
}

std::map<int, std::vector<std::vector<bool>>> CameraCalibration::generateColorBoardSegmentationMasks(const std::string &camera_label, const std::string &colorboard_type) const {
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + camera_label + "_pixel_UUID";

    if (!context->doesGlobalDataExist(global_data_label.c_str())) {
        helios_runtime_error("ERROR (CameraCalibration::generateColorBoardSegmentationMasks): Camera pixel UUID data for camera '" + camera_label + "' does not exist. Make sure the radiation model has been run.");
    }

    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);

    // Get camera resolution from global data
    std::string resolution_label = "camera_" + camera_label + "_resolution";
    if (!context->doesGlobalDataExist(resolution_label.c_str())) {
        helios_runtime_error("ERROR (CameraCalibration::generateColorBoardSegmentationMasks): Camera resolution data for camera '" + camera_label + "' does not exist. Make sure the radiation model has been run.");
    }
    helios::int2 camera_resolution;
    context->getGlobalData(resolution_label.c_str(), camera_resolution);

    std::string colorboard_label = "colorboard_" + colorboard_type;
    std::map<int, std::vector<std::vector<bool>>> label_masks;

    // First pass: identify all unique labels and create binary masks
    for (int j = 0; j < camera_resolution.y; j++) {
        for (int i = 0; i < camera_resolution.x; i++) {
            uint ii = camera_resolution.x - i - 1;
            uint pixel_idx = j * camera_resolution.x + ii;
            if (pixel_idx < camera_UUIDs.size()) {
                uint UUID = camera_UUIDs.at(pixel_idx) - 1;
                if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID, colorboard_label.c_str())) {
                    uint patch_id;
                    context->getPrimitiveData(UUID, colorboard_label.c_str(), patch_id);

                    // Initialize mask for this patch if it doesn't exist
                    if (label_masks.find(patch_id) == label_masks.end()) {
                        label_masks[patch_id] = std::vector<std::vector<bool>>(camera_resolution.y, std::vector<bool>(camera_resolution.x, false));
                    }

                    label_masks[patch_id][j][i] = true;
                }
            }
        }
    }

    return label_masks;
}


CameraCalibration::LabColor CameraCalibration::rgbToLab(const helios::vec3 &rgb) const {
    // Decode the sRGB transfer function, then convert the linear values
    auto gamma_correct = [](float c) { return (c <= 0.04045f) ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); };
    return linearRgbToLab(helios::make_vec3(gamma_correct(rgb.x), gamma_correct(rgb.y), gamma_correct(rgb.z)));
}

CameraCalibration::LabColor CameraCalibration::linearRgbToLab(const helios::vec3 &linear_rgb) const {
    // Linear sRGB to XYZ (D65)
    float X = 0.4124564f * linear_rgb.x + 0.3575761f * linear_rgb.y + 0.1804375f * linear_rgb.z;
    float Y = 0.2126729f * linear_rgb.x + 0.7151522f * linear_rgb.y + 0.0721750f * linear_rgb.z;
    float Z = 0.0193339f * linear_rgb.x + 0.1191920f * linear_rgb.y + 0.9503041f * linear_rgb.z;

    // Normalize by the D65 reference white
    X /= 0.95047f;
    Z /= 1.08883f;

    auto f_transform = [](float t) { return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f); };
    const float fx = f_transform(X);
    const float fy = f_transform(Y);
    const float fz = f_transform(Z);

    return LabColor(116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz));
}

helios::vec3 CameraCalibration::labToRgb(const LabColor &lab) const {
    // Convert Lab to XYZ
    float fy = (lab.L + 16.0f) / 116.0f;
    float fx = lab.a / 500.0f + fy;
    float fz = fy - lab.b / 200.0f;

    // Reverse f transform
    auto f_reverse = [](float t) {
        float t3 = t * t * t;
        return (t3 > 0.008856f) ? t3 : (t - 16.0f / 116.0f) / 7.787f;
    };

    float X = f_reverse(fx);
    float Y = f_reverse(fy);
    float Z = f_reverse(fz);

    // Reference white point D65
    const float Xn = 0.95047f;
    const float Yn = 1.00000f;
    const float Zn = 1.08883f;

    // Scale by reference white
    X *= Xn;
    Y *= Yn;
    Z *= Zn;

    // XYZ to sRGB transformation matrix
    float r_linear = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float g_linear = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float b_linear = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

    // Apply gamma correction
    auto gamma_uncorrect = [](float c) { return (c <= 0.0031308f) ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f; };

    float r = gamma_uncorrect(r_linear);
    float g = gamma_uncorrect(g_linear);
    float b = gamma_uncorrect(b_linear);

    // Clamp to [0, 1]
    r = std::max(0.0f, std::min(1.0f, r));
    g = std::max(0.0f, std::min(1.0f, g));
    b = std::max(0.0f, std::min(1.0f, b));

    return helios::make_vec3(r, g, b);
}

double CameraCalibration::deltaE76(const LabColor &lab1, const LabColor &lab2) const {
    // CIE76 Delta E formula: simple Euclidean distance in Lab space
    double delta_L = lab1.L - lab2.L;
    double delta_a = lab1.a - lab2.a;
    double delta_b = lab1.b - lab2.b;

    return sqrt(delta_L * delta_L + delta_a * delta_a + delta_b * delta_b);
}

double CameraCalibration::deltaE2000(const LabColor &lab1, const LabColor &lab2) const {
    // Implementation of CIE Delta E 2000 formula
    // More perceptually accurate than CIE76

    const double kL = 1.0; // Lightness weighting factor
    const double kC = 1.0; // Chroma weighting factor
    const double kH = 1.0; // Hue weighting factor

    double L1 = lab1.L, a1 = lab1.a, b1 = lab1.b;
    double L2 = lab2.L, a2 = lab2.a, b2 = lab2.b;

    // Calculate C and h for each color
    double C1 = sqrt(a1 * a1 + b1 * b1);
    double C2 = sqrt(a2 * a2 + b2 * b2);
    double C_bar = (C1 + C2) / 2.0;

    // G factor for a' calculation
    double G = 0.5 * (1.0 - sqrt(pow(C_bar, 7) / (pow(C_bar, 7) + pow(25.0, 7))));

    // Calculate a' values
    double a1_prime = a1 * (1.0 + G);
    double a2_prime = a2 * (1.0 + G);

    // Calculate C' values
    double C1_prime = sqrt(a1_prime * a1_prime + b1 * b1);
    double C2_prime = sqrt(a2_prime * a2_prime + b2 * b2);

    // Calculate h' values (hue angles in degrees)
    double h1_prime = 0, h2_prime = 0;
    if (C1_prime > 0) {
        h1_prime = atan2(b1, a1_prime) * 180.0 / M_PI;
        if (h1_prime < 0)
            h1_prime += 360.0;
    }
    if (C2_prime > 0) {
        h2_prime = atan2(b2, a2_prime) * 180.0 / M_PI;
        if (h2_prime < 0)
            h2_prime += 360.0;
    }

    // Calculate differences
    double delta_L_prime = L2 - L1;
    double delta_C_prime = C2_prime - C1_prime;

    double delta_h_prime = 0;
    if (C1_prime * C2_prime > 0) {
        double diff = h2_prime - h1_prime;
        if (abs(diff) <= 180) {
            delta_h_prime = diff;
        } else if (diff > 180) {
            delta_h_prime = diff - 360;
        } else {
            delta_h_prime = diff + 360;
        }
    }

    double delta_H_prime = 2.0 * sqrt(C1_prime * C2_prime) * sin(delta_h_prime * M_PI / 360.0);

    // Calculate averages
    double L_bar_prime = (L1 + L2) / 2.0;
    double C_bar_prime = (C1_prime + C2_prime) / 2.0;

    double h_bar_prime = 0;
    if (C1_prime * C2_prime > 0) {
        double sum = h1_prime + h2_prime;
        double diff = abs(h1_prime - h2_prime);
        if (diff <= 180) {
            h_bar_prime = sum / 2.0;
        } else if (sum < 360) {
            h_bar_prime = (sum + 360) / 2.0;
        } else {
            h_bar_prime = (sum - 360) / 2.0;
        }
    }

    // Calculate T
    double T = 1.0 - 0.17 * cos((h_bar_prime - 30) * M_PI / 180.0) + 0.24 * cos(2 * h_bar_prime * M_PI / 180.0) + 0.32 * cos((3 * h_bar_prime + 6) * M_PI / 180.0) - 0.20 * cos((4 * h_bar_prime - 63) * M_PI / 180.0);

    // Calculate delta_theta
    double delta_theta = 30.0 * exp(-pow((h_bar_prime - 275) / 25.0, 2));

    // Calculate RC
    double RC = 2.0 * sqrt(pow(C_bar_prime, 7) / (pow(C_bar_prime, 7) + pow(25.0, 7)));

    // Calculate SL, SC, SH
    double SL = 1.0 + (0.015 * pow(L_bar_prime - 50, 2)) / sqrt(20 + pow(L_bar_prime - 50, 2));
    double SC = 1.0 + 0.045 * C_bar_prime;
    double SH = 1.0 + 0.015 * C_bar_prime * T;

    // Calculate RT
    double RT = -sin(2 * delta_theta * M_PI / 180.0) * RC;

    // Calculate Delta E 2000
    double term1 = delta_L_prime / (kL * SL);
    double term2 = delta_C_prime / (kC * SC);
    double term3 = delta_H_prime / (kH * SH);
    double term4 = RT * term2 * term3;

    return sqrt(term1 * term1 + term2 * term2 + term3 * term3 + term4);
}
