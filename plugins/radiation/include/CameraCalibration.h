/** \file "CameraCalibration.h" Primary header file for synthetic radiation camera calibration.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_CAMERACALIBRATION_H
#define HELIOS_CAMERACALIBRATION_H
#include "Context.h"

//! Camera calibration structure used for camera calibration tasks
struct CameraCalibration {
    explicit CameraCalibration(helios::Context *context);

    //! Add checker board geometry into context
    /**
     * \param[in] boardsidesize size of check board
     * \param[in] patchsize size of each square patch
     * \param[in] centrelocation location of the board (vec3 on coordinates x,y,z)
     * \param[in] rotationrad radians of rotation (vec3 on coordinates x,y,z)
     * \param[in] firstblack initial the color of first square patch (black or white)
     */
    std::vector<uint> addCheckerboard(const helios::int2 &boardsidesize, const float &patchsize, const helios::vec3 &centrelocation, const helios::vec3 &rotationrad, bool firstblack = true);


    //! Add default checker board geometry into context
    /**
     * \param[in] centrelocation location of the board (vec3 on coordinates x,y,z)
     * \param[in] rotationrad radians of rotation (vec3 on coordinates x,y,z)
     */
    std::vector<uint> addDefaultCheckerboard(const helios::vec3 &centrelocation, const helios::vec3 &rotationrad);

    //! Add color board geometry into context
    /**
     * \param[in] centrelocation location of the board center (vec3 on coordinates x,y,z)
     * \param[in] patchsize size of each square color patch on the board
     * \param[in] rotationrad rotation the board about x-, y-, and z-axes
     * \param[in] colorassignment [OPTIONAL] Color of each patch and size of color board. First index is the row and second index is the column.
     * \param[in] spectrumassignment [OPTIONAL] Label of global data corresponding to the reflectance spectrum of each color patch on the color board. First index is the row and second index is the column.
     */
    std::vector<uint> addColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad, const std::vector<std::vector<helios::RGBcolor>> &colorassignment = {},
                                    const std::vector<std::vector<std::string>> &spectrumassignment = {});

    //! Add color board geometry into context with colorboard type labeling
    /**
     * \param[in] centrelocation location of the board center (vec3 on coordinates x,y,z)
     * \param[in] patchsize size of each square color patch on the board
     * \param[in] rotationrad rotation the board about x-, y-, and z-axes
     * \param[in] colorassignment Color of each patch and size of color board. First index is the row and second index is the column.
     * \param[in] spectrumassignment Label of global data corresponding to the reflectance spectrum of each color patch on the color board. First index is the row and second index is the column.
     * \param[in] colorboard_type String identifier for the colorboard type (e.g., "DGK", "Calibrite", "SpyderCHECKR")
     */
    std::vector<uint> addColorboard(const helios::vec3 &centrelocation, float patchsize, const helios::vec3 &rotationrad, const std::vector<std::vector<helios::RGBcolor>> &colorassignment,
                                    const std::vector<std::vector<std::string>> &spectrumassignment, const std::string &colorboard_type);

    // //! Set reflectivity for a specific UUID
    // /**
    //  * \param[in] UUID: Corresponding UUID
    //  * \param[in] filename: name with path of XML file
    //  * \param[in] labelname: label name of data
    // */
    //    void setColorboardReflectivity(const uint &UUID, const std::string &filename, const std::string &labelname);

    //! Add default color board (DGK-DKK) with spectral reflectivity values
    /**
     * \param[in] centrelocation Location of the board (vec3 on coordinates x,y,z)
     * \param[in] patchsize Size of each color patch in color board
     * \param[in] rotationrad Radians of rotation (vec3 on coordinates x,y,z)
     * \return A vector of UUIDs making up the color board patches
     */
    std::vector<uint> addDefaultColorboard(const helios::vec3 &centrelocation, float patchsize = 0.5, const helios::vec3 &rotationrad = helios::make_vec3(0, 0, 0));

    //! Add DGK-DKK calibration color board
    /**
     * \param[in] centrelocation Location of the board (vec3 on coordinates x,y,z)
     * \param[in] patchsize Size of each color patch in color board
     * \param[in] rotationrad Radians of rotation (vec3 on coordinates x,y,z)
     * \return A vector of UUIDs making up the color board patches
     */
    std::vector<uint> addDGKColorboard(const helios::vec3 &centrelocation, float patchsize = 0.5, const helios::vec3 &rotationrad = helios::make_vec3(0, 0, 0));

    //! Add Calibrite ColorChecker Classic calibration color board
    /**
     * \param[in] centrelocation Location of the board (vec3 on coordinates x,y,z)
     * \param[in] patchsize Size of each color patch in color board
     * \param[in] rotationrad Radians of rotation (vec3 on coordinates x,y,z)
     * \return A vector of UUIDs making up the color board patches
     */
    std::vector<uint> addCalibriteColorboard(const helios::vec3 &centrelocation, float patchsize = 0.5, const helios::vec3 &rotationrad = helios::make_vec3(0, 0, 0));

    //! Add Datacolor SpyderCHECKR 24 calibration color board
    /**
     * \param[in] centrelocation Location of the board (vec3 on coordinates x,y,z)
     * \param[in] patchsize Size of each color patch in color board
     * \param[in] rotationrad Radians of rotation (vec3 on coordinates x,y,z)
     * \return A vector of UUIDs making up the color board patches
     */
    std::vector<uint> addSpyderCHECKRColorboard(const helios::vec3 &centrelocation, float patchsize = 0.5, const helios::vec3 &rotationrad = helios::make_vec3(0, 0, 0));

    /**
     * \brief Retrieves all color board UUIDs from all colorboards.
     *
     * \return A vector containing the UUIDs of all color board patches from all colorboards.
     */
    std::vector<uint> getAllColorBoardUUIDs() const;

    //! Write XML file from spectral vectors containing both wavelengths and spectral values
    /**
     * \param[in] filename Name with path of XML file
     * \param[in] note Note to write in XML file
     * \param[in] label Label of spectral data to write in XML file
     * \param[in] spectrum Pointer of spectrum to write in XML file
     */
    bool writeSpectralXMLfile(const std::string &filename, const std::string &note, const std::string &label, std::vector<helios::vec2> *spectrum);

    //! Load XML file and save data in spectral vectors containing both wavelengths and spectral values
    /**
     * \param[in] filename Name with path of XML file
     * \param[in] labelname Label name of data
     * \param[in] spectraldata An empty spectral vector to be filled
     */
    bool loadXMLlabeldata(const std::string &filename, const std::string &labelname, std::vector<helios::vec2> &spectraldata);

    //! Reduce calibration error based on gradient descent
    /**
     * \param[in] expandedcameraspectra Expanded camera response spectra to be updated.
     * \param[in] expandedconstinput Expanded input spectra.
     * \param[in] learningrate Learning rate.
     * \param[in] truevalues True color board values.
     * \return A float of loss.
     */
    float GradientDescent(std::vector<std::vector<float>> *expandedcameraspectra, const std::vector<std::vector<float>> &expandedconstinput, const float &learningrate, const std::vector<std::vector<float>> &truevalues);

    //! Update camera response spectra
    /**
     * \param[in] camerareponselabels Label vector of camera response spectra to be updated.
     * \param[in] cameralabel Label of camera to be used for simulation.
     * \param[in] simulatedinputspectra Input spectra.
     * \param[in] truevalues True color board values.
     * \return A vector of training losses.
     */
    std::vector<float> updateCameraResponseSpectra(const std::vector<std::string> &camerareponselabels, const std::string &cameralabel, const std::map<uint, std::vector<helios::vec2>> &simulatedinputspectra,
                                                   const std::vector<std::vector<float>> &truevalues);

    //! Parameter struct for gradient descent.
    struct GradientDescentParameters {
        float learningrate = 0.000001;
        int maxiteration = 500;
        float minloss = 0.01;
        std::vector<float> camerarescales = {1, 1, 1};
        GradientDescentParameters() = default;
    };

    GradientDescentParameters responseupdateparameters;

    //! Preprocess all spectra for modelling.
    /**
     * \param[in] sourcelabels Label vector of source spectra.
     * \param[in] cameralabels Label vector of camera spectra.
     * \param[in] objectlabels Label vector of object spectra.
     * \param[inout] wavelengthrange Min and max of wavelength boundary.
     * \param[in] targetlabel Label of target spectrum used for interpolation.
     */
    void preprocessSpectra(const std::vector<std::string> &sourcelabels, const std::vector<std::string> &cameralabels, std::vector<std::string> &objectlabels, helios::vec2 &wavelengthrange, const std::string &targetlabel = "");

    //! Get distorted image.
    /**
     * \param[in] cameralabel Label of camera.
     * \param[in] bandlabels Label vector of bands.
     * \param[in] focalxy Focal length xy of camera.
     * \param[in] distCoeffs Distortion coefficients.
     * \param[in] cameraresolution Resolution of camera.
     * \param[in] cameraresolution Resolution of camera for distortion.
     */
    void distortImage(const std::string &cameralabel, const std::vector<std::string> &bandlabels, const helios::vec2 &focalxy, std::vector<double> &distCoeffs, helios::int2 cameraresolution);

    //! Get camera spectral response scale.
    /**
     * \param[in] cameralabel Label of camera.
     * \param[in] cameraresolution Resolution of camera.
     * \param[in] bandlabels Label vector of bands.
     * \param[in] truevalues True color board values.
     */
    float getCameraResponseScale(const std::string &cameralabel, const helios::int2 cameraresolution, const std::vector<std::string> &bandlabels, const std::vector<std::vector<float>> &truevalues);

    std::map<std::string, std::map<std::string, std::vector<helios::vec2>>> processedspectra;

    //! Read ROMC canopy file (Used for self test).
    /**
     * \return A vector of ROMC canopy UUIDs.
     */
    std::vector<uint> readROMCCanopy();


    //! Write calibrated camera response spectra.
    /**
     * \param[in] camerareponselabels Label vector of original camera response spectra to be written.
     * \param[in] calibratemark Calibration mark.
     * \param[in] scale Manually adjustable Scaling factor for calibrated camera response spectra.
     */
    void writeCalibratedCameraResponses(const std::vector<std::string> &camerareponselabels, const std::string &calibratemark, float scale);

    //    void resetCameraResponses(std::string camerareponselabels, float scale);

    // === PUBLIC METHODS FOR AUTO-CALIBRATION (used by RadiationModel) ===

    //! Structure to store Lab color values
    struct LabColor {
        float L, a, b;
        LabColor(float L_val, float a_val, float b_val) : L(L_val), a(a_val), b(b_val) {
        }
    };

    //! Detect which colorboard types are present in the scene
    std::vector<std::string> detectColorBoardTypes() const;

    //! Get reference Lab values for DGK colorboard (18 patches)
    std::vector<LabColor> getReferenceLab_DGK() const;

    //! Get reference Lab values for Calibrite ColorChecker Classic (24 patches)
    std::vector<LabColor> getReferenceLab_Calibrite() const;

    //! Get reference Lab values for Datacolor SpyderCHECKR 24 (24 patches)
    std::vector<LabColor> getReferenceLab_SpyderCHECKR() const;

    //! Convert RGB color to Lab color space (RGB as vec3 [0,1])
    LabColor rgbToLab(const helios::vec3 &rgb) const;

    //! Convert Lab color to RGB color space (returns vec3 [0,1])
    helios::vec3 labToRgb(const LabColor &lab) const;

    //! Calculate Delta E 76 (CIE76) color difference between two Lab colors
    double deltaE76(const LabColor &lab1, const LabColor &lab2) const;

    //! Calculate Delta E 2000 color difference between two Lab colors (more perceptually accurate)
    double deltaE2000(const LabColor &lab1, const LabColor &lab2) const;

protected:
    std::map<std::string, std::vector<helios::vec2>> calibratedcameraspectra;

    helios::Context *context;

    //! Generate segmentation masks for colorboard patches
    std::map<int, std::vector<std::vector<bool>>> generateColorBoardSegmentationMasks(const std::string &camera_label, const std::string &colorboard_type) const;


    //! Expand vector for integral.
    /**
     * \param[in] targetspectrum Spectrum to be expanded.
     * \param[in] scale Scale value for spectrum.
     * \return Expanded spectrum.
     */
    std::vector<float> expandSpectrum(const std::vector<helios::vec2> &targetspectrum, float scale);

    //! UUIDs of colorboard patches organized by type
    std::map<std::string, std::vector<uint>> UUIDs_colorboards;

    std::vector<uint> UUIDs_black;

    std::vector<uint> UUIDs_white;

    //! RGB values of DGK DKK color board
    helios::RGBcolor white_DGK_01 = helios::make_RGBcolor(1.f, 1.f, 1.f);
    helios::RGBcolor lightgray_DGK_02 = helios::make_RGBcolor(0.8, 0.8, 0.8);
    helios::RGBcolor mediumlightgray_DGK_03 = helios::make_RGBcolor(0.6, 0.6, 0.6);
    helios::RGBcolor mediumdarkgray_DGK_04 = helios::make_RGBcolor(0.4, 0.4, 0.4);
    helios::RGBcolor darkgray_DGK_05 = helios::make_RGBcolor(0.2, 0.2, 0.2);
    helios::RGBcolor black_DGK_06 = helios::make_RGBcolor(0., 0., 0.);
    helios::RGBcolor red_DGK_07 = helios::make_RGBcolor(1, 0.1, 0);
    helios::RGBcolor yellow_DGK_08 = helios::make_RGBcolor(1, 1, 0);
    helios::RGBcolor green_DGK_09 = helios::make_RGBcolor(0.4118, 0.7490, 0.2706);
    helios::RGBcolor skyblue_DGK_10 = helios::make_RGBcolor(0, 0.5647, 1);
    helios::RGBcolor darkblue_DGK_11 = helios::make_RGBcolor(0.1882, 0.2314, 0.3608);
    helios::RGBcolor magenta_DGK_12 = helios::make_RGBcolor(1, 0, 0.5647);
    helios::RGBcolor brickred_DGK_13 = helios::make_RGBcolor(0.7294, 0.0118, 0.1216);
    helios::RGBcolor orange_DGK_14 = helios::make_RGBcolor(0.9569, 0.6, 0.0078);
    helios::RGBcolor teal_DGK_15 = helios::make_RGBcolor(0, 0.4549, 0.7412);
    helios::RGBcolor mauve_DGK_16 = helios::make_RGBcolor(0.66, 0.34, 0.53);
    helios::RGBcolor lighttan_DGK_17 = helios::make_RGBcolor(0.73, 0.60, 0.51);
    helios::RGBcolor darktan_DGK_18 = helios::make_RGBcolor(0.66, 0.55, 0.45);

    const std::vector<std::vector<helios::RGBcolor>> colorassignment_DGK = {{white_DGK_01, lightgray_DGK_02, mediumlightgray_DGK_03, mediumdarkgray_DGK_04, darkgray_DGK_05, black_DGK_06},
                                                                            {red_DGK_07, yellow_DGK_08, green_DGK_09, skyblue_DGK_10, darkblue_DGK_11, magenta_DGK_12},
                                                                            {brickred_DGK_13, orange_DGK_14, teal_DGK_15, mauve_DGK_16, lighttan_DGK_17, darktan_DGK_18}};

    const std::vector<std::vector<std::string>> spectrumassignment_DGK = {{"ColorReference_DGK_01", "ColorReference_DGK_02", "ColorReference_DGK_03", "ColorReference_DGK_04", "ColorReference_DGK_05", "ColorReference_DGK_06"},
                                                                          {"ColorReference_DGK_07", "ColorReference_DGK_08", "ColorReference_DGK_09", "ColorReference_DGK_10", "ColorReference_DGK_11", "ColorReference_DGK_12"},
                                                                          {"ColorReference_DGK_13", "ColorReference_DGK_14", "ColorReference_DGK_15", "ColorReference_DGK_16", "ColorReference_DGK_17", "ColorReference_DGK_18"}};

    //! RGB values of Calibrite ColorChecker Classic color board
    helios::RGBcolor brown_Calibrite_01 = helios::make_RGBcolor(0.38, 0.25, 0.16);
    helios::RGBcolor lighttan_Calibrite_02 = helios::make_RGBcolor(0.79, 0.59, 0.54);
    helios::RGBcolor bluegray_Calibrite_03 = helios::make_RGBcolor(0.34, 0.43, 0.64);
    helios::RGBcolor olive_Calibrite_04 = helios::make_RGBcolor(0.34, 0.43, 0.22);
    helios::RGBcolor lavender_Calibrite_05 = helios::make_RGBcolor(0.50, 0.47, 0.70);
    helios::RGBcolor bluegreen_Calibrite_06 = helios::make_RGBcolor(0.54, 0.75, 0.72);
    helios::RGBcolor orange_Calibrite_07 = helios::make_RGBcolor(0.85, 0.52, 0.24);
    helios::RGBcolor midblue_Calibrite_08 = helios::make_RGBcolor(0.22, 0.28, 0.60);
    helios::RGBcolor lightred_Calibrite_09 = helios::make_RGBcolor(0.78, 0.30, 0.33);
    helios::RGBcolor violet_Calibrite_10 = helios::make_RGBcolor(0.25, 0.14, 0.36);
    helios::RGBcolor yellowgreen_Calibrite_11 = helios::make_RGBcolor(0.72, 0.79, 0.32);
    helios::RGBcolor lightorange_Calibrite_12 = helios::make_RGBcolor(0.87, 0.72, 0.27);
    helios::RGBcolor blue_Calibrite_13 = helios::make_RGBcolor(0.16, 0.19, 0.48);
    helios::RGBcolor green_Calibrite_14 = helios::make_RGBcolor(0.36, 0.61, 0.32);
    helios::RGBcolor red_Calibrite_15 = helios::make_RGBcolor(0.65, 0.19, 0.17);
    helios::RGBcolor yellow_Calibrite_16 = helios::make_RGBcolor(0.93, 0.86, 0.29);
    helios::RGBcolor magenta_Calibrite_17 = helios::make_RGBcolor(0.70, 0.27, 0.58);
    helios::RGBcolor lightblue_Calibrite_18 = helios::make_RGBcolor(0.21, 0.48, 0.71);
    helios::RGBcolor white_Calibrite_19 = helios::make_RGBcolor(1.0, 1.0, 1.0);
    helios::RGBcolor lightgray_Calibrite_20 = helios::make_RGBcolor(0.81, 0.81, 0.81);
    helios::RGBcolor midlightgray_Calibrite_21 = helios::make_RGBcolor(0.68, 0.68, 0.68);
    helios::RGBcolor middarkgray_Calibrite_22 = helios::make_RGBcolor(0.48, 0.48, 0.48);
    helios::RGBcolor darkgray_Calibrite_23 = helios::make_RGBcolor(0.27, 0.27, 0.27);
    helios::RGBcolor black_Calibrite_24 = helios::make_RGBcolor(0.0, 0.0, 0.0);

    const std::vector<std::vector<helios::RGBcolor>> colorassignment_Calibrite = {{brown_Calibrite_01, lighttan_Calibrite_02, bluegray_Calibrite_03, olive_Calibrite_04, lavender_Calibrite_05, bluegreen_Calibrite_06},
                                                                                  {orange_Calibrite_07, midblue_Calibrite_08, lightred_Calibrite_09, violet_Calibrite_10, yellowgreen_Calibrite_11, lightorange_Calibrite_12},
                                                                                  {blue_Calibrite_13, green_Calibrite_14, red_Calibrite_15, yellow_Calibrite_16, magenta_Calibrite_17, lightblue_Calibrite_18},
                                                                                  {white_Calibrite_19, lightgray_Calibrite_20, midlightgray_Calibrite_21, middarkgray_Calibrite_22, darkgray_Calibrite_23, black_Calibrite_24}};

    const std::vector<std::vector<std::string>> spectrumassignment_Calibrite = {
            {"ColorReference_Calibrite_01", "ColorReference_Calibrite_02", "ColorReference_Calibrite_03", "ColorReference_Calibrite_04", "ColorReference_Calibrite_05", "ColorReference_Calibrite_06"},
            {"ColorReference_Calibrite_07", "ColorReference_Calibrite_08", "ColorReference_Calibrite_09", "ColorReference_Calibrite_10", "ColorReference_Calibrite_11", "ColorReference_Calibrite_12"},
            {"ColorReference_Calibrite_13", "ColorReference_Calibrite_14", "ColorReference_Calibrite_15", "ColorReference_Calibrite_16", "ColorReference_Calibrite_17", "ColorReference_Calibrite_18"},
            {"ColorReference_Calibrite_19", "ColorReference_Calibrite_20", "ColorReference_Calibrite_21", "ColorReference_Calibrite_22", "ColorReference_Calibrite_23", "ColorReference_Calibrite_24"}};

    //! RGB values of Datacolor SpyderCHECKR 24 color board
    helios::RGBcolor bluegreen_SpyderCHECKR_01 = helios::make_RGBcolor(0.54, 0.75, 0.72);
    helios::RGBcolor lavender_SpyderCHECKR_02 = helios::make_RGBcolor(0.50, 0.47, 0.70);
    helios::RGBcolor olive_SpyderCHECKR_03 = helios::make_RGBcolor(0.34, 0.43, 0.22);
    helios::RGBcolor bluegray_SpyderCHECKR_04 = helios::make_RGBcolor(0.34, 0.43, 0.64);
    helios::RGBcolor lighttan_SpyderCHECKR_05 = helios::make_RGBcolor(0.79, 0.59, 0.54);
    helios::RGBcolor brown_SpyderCHECKR_06 = helios::make_RGBcolor(0.38, 0.25, 0.16);
    helios::RGBcolor orange_SpyderCHECKR_07 = helios::make_RGBcolor(0.85, 0.52, 0.24);
    helios::RGBcolor midblue_SpyderCHECKR_08 = helios::make_RGBcolor(0.22, 0.28, 0.60);
    helios::RGBcolor lightred_SpyderCHECKR_09 = helios::make_RGBcolor(0.78, 0.30, 0.33);
    helios::RGBcolor violet_SpyderCHECKR_10 = helios::make_RGBcolor(0.25, 0.14, 0.36);
    helios::RGBcolor yellowgreen_SpyderCHECKR_11 = helios::make_RGBcolor(0.72, 0.79, 0.32);
    helios::RGBcolor lightorange_SpyderCHECKR_12 = helios::make_RGBcolor(0.87, 0.72, 0.27);
    helios::RGBcolor lightblue_SpyderCHECKR_13 = helios::make_RGBcolor(0.21, 0.48, 0.71);
    helios::RGBcolor magenta_SpyderCHECKR_14 = helios::make_RGBcolor(0.70, 0.27, 0.58);
    helios::RGBcolor yellow_SpyderCHECKR_15 = helios::make_RGBcolor(0.93, 0.86, 0.29);
    helios::RGBcolor red_SpyderCHECKR_16 = helios::make_RGBcolor(0.65, 0.19, 0.17);
    helios::RGBcolor green_SpyderCHECKR_17 = helios::make_RGBcolor(0.36, 0.61, 0.32);
    helios::RGBcolor blue_SpyderCHECKR_18 = helios::make_RGBcolor(0.16, 0.19, 0.48);
    helios::RGBcolor white_SpyderCHECKR_19 = helios::make_RGBcolor(1.0, 1.0, 1.0);
    helios::RGBcolor lightgray_SpyderCHECKR_20 = helios::make_RGBcolor(0.81, 0.81, 0.81);
    helios::RGBcolor midlightgray_SpyderCHECKR_21 = helios::make_RGBcolor(0.68, 0.68, 0.68);
    helios::RGBcolor middarkgray_SpyderCHECKR_22 = helios::make_RGBcolor(0.48, 0.48, 0.48);
    helios::RGBcolor darkgray_SpyderCHECKR_23 = helios::make_RGBcolor(0.27, 0.27, 0.27);
    helios::RGBcolor black_SpyderCHECKR_24 = helios::make_RGBcolor(0.0, 0.0, 0.0);

    const std::vector<std::vector<helios::RGBcolor>> colorassignment_SpyderCHECKR = {{bluegreen_SpyderCHECKR_01, lavender_SpyderCHECKR_02, olive_SpyderCHECKR_03, bluegray_SpyderCHECKR_04, lighttan_SpyderCHECKR_05, brown_SpyderCHECKR_06},
                                                                                     {orange_SpyderCHECKR_07, midblue_SpyderCHECKR_08, lightred_SpyderCHECKR_09, violet_SpyderCHECKR_10, yellowgreen_SpyderCHECKR_11, lightorange_SpyderCHECKR_12},
                                                                                     {lightblue_SpyderCHECKR_13, magenta_SpyderCHECKR_14, yellow_SpyderCHECKR_15, red_SpyderCHECKR_16, green_SpyderCHECKR_17, blue_SpyderCHECKR_18},
                                                                                     {white_SpyderCHECKR_19, lightgray_SpyderCHECKR_20, midlightgray_SpyderCHECKR_21, middarkgray_SpyderCHECKR_22, darkgray_SpyderCHECKR_23, black_SpyderCHECKR_24}};

    const std::vector<std::vector<std::string>> spectrumassignment_SpyderCHECKR = {
            {"ColorReference_SpyderCHECKR_01", "ColorReference_SpyderCHECKR_02", "ColorReference_SpyderCHECKR_03", "ColorReference_SpyderCHECKR_04", "ColorReference_SpyderCHECKR_05", "ColorReference_SpyderCHECKR_06"},
            {"ColorReference_SpyderCHECKR_07", "ColorReference_SpyderCHECKR_08", "ColorReference_SpyderCHECKR_09", "ColorReference_SpyderCHECKR_10", "ColorReference_SpyderCHECKR_11", "ColorReference_SpyderCHECKR_12"},
            {"ColorReference_SpyderCHECKR_13", "ColorReference_SpyderCHECKR_14", "ColorReference_SpyderCHECKR_15", "ColorReference_SpyderCHECKR_16", "ColorReference_SpyderCHECKR_17", "ColorReference_SpyderCHECKR_18"},
            {"ColorReference_SpyderCHECKR_19", "ColorReference_SpyderCHECKR_20", "ColorReference_SpyderCHECKR_21", "ColorReference_SpyderCHECKR_22", "ColorReference_SpyderCHECKR_23", "ColorReference_SpyderCHECKR_24"}};
};

#endif // HELIOS_CAMERACALIBRATION_H
