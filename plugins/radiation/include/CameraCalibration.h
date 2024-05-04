/** \file "CameraCalibration.h" Primary header file for synthetic radiation camera calibration.

    Copyright (C) 2016-2024 Brian Bailey

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
struct CameraCalibration{
    explicit CameraCalibration(helios::Context *context);

    //! Add checker board geometry into context
    /**
     * \param[in] boardsidesize: size of check board
     * \param[in] patchsize: size of each square patch
     * \param[in] centrelocation: location of the board (vec3 on coordinates x,y,z)
     * \param[in] rotationrad: radians of rotation (vec3 on coordinates x,y,z)
     * \param[in] firstblack: initial the color of first square patch (black or white)
    */
    std::vector<uint> addCheckerboard(const helios::int2 &boardsidesize, const float &patchsize, const helios::vec3 &centrelocation,
                                      const helios::vec3 &rotationrad, bool firstblack = true);


    //! Add default checker board geometry into context
    /**
     * \param[in] centrelocation: location of the board (vec3 on coordinates x,y,z)
     * \param[in] rotationrad: radians of rotation (vec3 on coordinates x,y,z)
    */
    std::vector<uint> addDefaultCheckerboard(const helios::vec3 &centrelocation,
                                             const helios::vec3 &rotationrad);

    //! Add color board geometry into context
    /**
     * \param[in] centrelocation: location of the board (vec3 on coordinates x,y,z)
     * \param[in] rotationrad: radians of rotation (vec3 on coordinates x,y,z)
     * \param[in] colorassignment: color of each patch and size of color board
     * \param[in] patchsize: size of each square patch
    */
    std::vector<uint> addColorboard(const helios::vec3 &centrelocation, const helios::vec3 &rotationrad,
                       const std::vector<std::vector<helios::RGBcolor>> &colorassignment, float patchsize);

    //! Set reflectivity for a specific UUID
    /**
     * \param[in] UUID: Corresponding UUID
     * \param[in] filename: name with path of XML file
     * \param[in] labelname: label name of data
    */
    void setColorboardReflectivity(const uint &UUID, const std::string &filename, const std::string &labelname);

    //! Add default color board (DKC-RPO) with spectral reflectivity values
    /**
     * \param[in] centrelocation: Location of the board (vec3 on coordinates x,y,z)
     * \param[in] rotationrad: Radians of rotaion (vec3 on coordinates x,y,z)
     * \param[in] patchsize: Size of each patch in color board
     * \return A vector of default color board UUIDs.
    */
    std::vector<uint> addDefaultColorboard(const helios::vec3 &centrelocation, const helios::vec3 &rotationrad, float patchsize=0.5);

    void setDefaultColorBoardSpectra();

    std::vector<uint> getColorBoardUUIDs();

    //! Write XML file from spectral vectors containing both wavelengths and spectral values
    /**
     * \param[in] filename: Name with path of XML file
     * \param[in] note: Note to write in XML file
     * \param[in] label: Label of spectral data to write in XML file
     * \param[in] spectrum: Pointer of spectrum to write in XML file
    */
    bool writeSpectralXMLfile(const std::string &filename, const std::string &note, const std::string &label, std::vector<helios::vec2> *spectrum);

    //! Load XML file and save data in spectral vectors containing both wavelengths and spectral values
    /**
     * \param[in] filename: Name with path of XML file
     * \param[in] labelname: Label name of data
     * \param[in] spectraldata: An empty spectral vector to be filled
    */
    bool loadXMLlabeldata(const std::string &filename, const std::string &labelname,std::vector<helios::vec2> &spectraldata);

    //! Reduce calibration error based on gradient descent
    /**
     * \param[in] expandedcameraspectra: Expanded camera response spectra to be updated.
     * \param[in] expandedconstinput: Expanded input spectra.
     * \param[in] learningrate: Learning rate.
     * \param[in] truevalues: True color board values.
     * \return A float of loss.
    */
    float GradientDescent(std::vector<std::vector<float>> *expandedcameraspectra, const std::vector<std::vector<float>> &expandedconstinput,
                          const float &learningrate, const std::vector<std::vector<float>> &truevalues);

    //! Update camera response spectra
    /**
     * \param[in] camerareponselabels: Label vector of camera response spectra to be updated.
     * \param[in] simulatedinputspectra: Input spectra.
     * \param[in] truevalues: True color board values.
     * \param[in] Econst: Constant of sources.
     * \param[in] camerarescales: Scaling factors for calibrated camera response spectra.
     * \param[in] learningrate: Learning rate.
     * \return A vector of training losses.
    */
    std::vector<float> updateCameraResponseSpectra(const std::vector<std::string>& camerareponselabels, const std::string &label,
                                                   const std::map<uint,std::vector<helios::vec2>>& simulatedinputspectra,
                                                   const std::vector<std::vector<float>> &truevalues);

    //! Parameter struct for gradient descent.
    struct GradientDescentParameters
    {
        float learningrate=0.000001;
        int maxiteration = 500;
        float minloss = 0.01;
        std::vector<float> camerarescales = {1,1,1};
        GradientDescentParameters() = default;
    };

    GradientDescentParameters responseupdateparameters;

    //! Preprocess all spectra for modelling.
    /**
     * \param[in] sourcelabels: Label vector of source spectra.
     * \param[in] cameralabels: Label vector of camera spectra.
     * \param[in] objectlabels: Label vector of object spectra.
     * \param[inout] lowwavelength: Low wavelength boundary.
     * \param[inout] highwavelength: High wavelength boundary.
     * \param[in] targetlabel: Label of target spectrum used for interpolation.
    */
    void preprocessSpectra(const std::vector<std::string>& sourcelabels, const std::vector<std::string>& cameralabels,
                           std::vector<std::string>& objectlabels, helios::vec2 &wavelengthrange, const std::string& targetlabel="");

    //! Get distorted image.
    /**
     * \param[in] cameralabel: Label of camera.
     * \param[in] bandlabels: Label vector of bands.
     * \param[in] focalxy: Focal length xy of camera.
     * \param[in] distCoeffs: Distortion coefficients.
     * \param[in] camerareoslution: Resolution of camera.
     * \param[in] camerareoslutionR: Resolution of camera for distortion.
    */
    void distortImage(const std::string& cameralabel, const std::vector<std::string>& bandlabels, const helios::vec2 &focalxy,
                      std::vector<double> &distCoeffs, helios::int2 camerareoslution);

    //! Get camera spectral response scale.
    /**
     * \param[in] cameralabel: Label of camera.
     * \param[in] cameraresolution: Resolution of camera.
     * \param[in] bandlabels: Label vector of bands.
     * \param[in] truevalues: True color board values.
    */
    float getCameraResponseScale(const std::string &cameralabel, const helios::int2 cameraresolution, const std::vector<std::string> &bandlabels,
                       const std::vector<std::vector<float>> &truevalues);

    std::map<std::string,std::map<std::string, std::vector<helios::vec2>>> processedspectra;

    //! Read ROMC canopy file (Used for self test).
    /**
     * \return A vector of ROMC canopy UUIDs.
    */
    std::vector<uint> readROMCCanopy();

    //! Write calibrated camera response spectra.
    /**
     * \param[in] camerareponselabels: Label vector of original camera response spectra to be written.
     * \param[in] calibratemark: Calibration mark.
     * \param[in] scale: Manually adjustable Scaling factor for calibrated camera response spectra.
    */
    void writeCalibratedCameraResponses(const std::vector<std::string>& camerareponselabels, const std::string &calibratemark, float scale);

//    void resetCameraResponses(std::string camerareponselabels, float scale);

protected:

    std::map<std::string, std::vector<helios::vec2>> calibratedcameraspectra;

    helios::Context *context;

    //! Expand vector for integral.
    /**
     * \param[in] targetspectrum: Spectrum to be expanded.
     * \param[in] scale: Scale value for spectrum.
     * \return Expanded spectrum.
    */
    std::vector<float> expandSpectrum(const std::vector<helios::vec2>& targetspectrum, float scale);

    //! UUIDs of colorboard patches
    std::vector<uint> UUIDs_colorboard;

    std::vector<uint> UUIDs_black;

    std::vector<uint> UUIDs_white;

    //! Size and RGB values of default color board
    const std::vector<std::vector<helios::RGBcolor>> colorassignment_default=
            {{helios::make_RGBcolor(0.161,0.141,0.169), helios::make_RGBcolor(1,0,0.5647), helios::make_RGBcolor(0.898,0.596,0.4784)},
             {helios::make_RGBcolor(0.31,0.282,0.314), helios::make_RGBcolor(0.1882,0.2314,0.3608),helios::make_RGBcolor(0.9961,0.6667,0.5569)},
             {helios::make_RGBcolor(0.463,0.416,0.463), helios::make_RGBcolor(0,0.5647,1), helios::make_RGBcolor(0.8784,0.0039,0.6039)},
             {helios::make_RGBcolor(0.647,0.573,0.651), helios::make_RGBcolor(0.4118,0.7490,0.2706),helios::make_RGBcolor(0,0.4549,0.7412)},
             {helios::make_RGBcolor(0.828,0.573,0.651), helios::make_RGBcolor(1,1,0), helios::make_RGBcolor(0.9569,0.6,0.0078)},
             {helios::make_RGBcolor(0.984,0.977,0.988), helios::make_RGBcolor(1,0.1255,0), helios::make_RGBcolor(0.7294,0.0118,0.1216)}};

};

#endif //HELIOS_CAMERACALIBRATION_H
