/** \file "CameraCalibration.cpp" Routines for performing synthetic radiation camera calibration.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "CameraCalibration.h"

using namespace helios;

CameraCalibration::CameraCalibration(helios::Context *context):context(context){
}

std::vector<uint> CameraCalibration::addCheckerboard(const helios::int2 &boardsidesize, const float &patchsize, const helios::vec3 &centrelocation,
                                                     const helios::vec3 &rotationrad, bool firstblack) {
    helios::vec2 patchfullsize = make_vec2(patchsize,patchsize); //size of each patch
    std::vector<float> bwv(2,0);
    float bw;
    if (firstblack){
        bwv[0] = 1;
    }
    else{
        bwv[1] = 1;
    }
    std::vector<uint> UUIDs;
    for (int inpr=0;inpr<boardsidesize.x;inpr+=1){
        for (int inpc=0;inpc<boardsidesize.y;inpc+=1){
            if ( (inpr % 2 == 0 & inpc % 2 == 0)|(inpr % 2 != 0 & inpc % 2 != 0)){   //black - white - black - white -...
                bw = bwv[0];
            }
            else{
                bw = bwv[1];
            }
            helios::RGBcolor binarycolor(bw,bw,bw);

            float xp = 0-patchsize*(float(boardsidesize.x)-1)/2+patchsize*float(inpr);
            float yp = 0-patchsize*(float(boardsidesize.y)-1)/2+patchsize*float(inpc);

            uint UUID = context->addPatch(make_vec3(xp,yp,0),patchfullsize,nullrotation,binarycolor);

            if (bw==1){
                UUIDs_white.push_back(UUID);
                context->setPrimitiveData( UUID, "reflectivity_spectrum", "white");
            }
            else{
                UUIDs_black.push_back(UUID);
                context->setPrimitiveData( UUID, "reflectivity_spectrum", "");
            }

            context->setPrimitiveData(UUID, "transmissivity_spectrum", "");
            context->setPrimitiveData(UUID, "twosided_flag", uint(0));
            UUIDs.push_back(UUID);
        }
    }
    std::vector<helios::vec2> whitespectra(2200);
    for (int i=0; i<whitespectra.size(); i++){
        whitespectra.at(i).x=301+i;
        whitespectra.at(i).y=1;
    }
    context->setGlobalData("white",HELIOS_TYPE_VEC2,whitespectra.size(),&whitespectra[0]);
    context->rotatePrimitive(UUIDs,rotationrad.x, make_vec3(1,0,0));
    context->rotatePrimitive(UUIDs,rotationrad.y, make_vec3(0,1,0));
    context->rotatePrimitive(UUIDs,rotationrad.z, make_vec3(0,0,1));
    context->translatePrimitive(UUIDs,centrelocation);
    return UUIDs;
}

std::vector<uint> CameraCalibration::addDefaultCheckerboard(const helios::vec3 &centrelocation, const helios::vec3 &rotationrad){
    helios::int2 boardsidesize = make_int2(10,7);
    float patchsize = 0.029;
    bool firstblack = true;
    std::vector<uint> UUID_checkboard = CameraCalibration::addCheckerboard(boardsidesize, patchsize, centrelocation, rotationrad, firstblack);
    return UUID_checkboard;

}

std::vector<uint> CameraCalibration::addColorboard(const helios::vec3 &centrelocation, const helios::vec3 &rotationrad,
                                      const std::vector<std::vector<helios::RGBcolor>> &colorassignment, const float patchsize){
    std::vector<uint> UUIDs;
    helios::vec2 boardfullsize;
    boardfullsize.x=float(colorassignment.size());
    boardfullsize.y=float(colorassignment.back().size());
    uint UUID;
    for (int irow=0; float(irow) < boardfullsize.y; irow+=1){
        for (int icolumn=0; float(icolumn) < boardfullsize.x; icolumn+=1){

            float xp = centrelocation.x-patchsize*(float(boardfullsize.x)-1)/2+patchsize*float(icolumn);
            float yp = centrelocation.y-patchsize*(float(boardfullsize.y)-1)/2+patchsize*float(irow);

            UUID = context->addPatch(make_vec3(xp,yp,centrelocation.z),make_vec2(patchsize,patchsize),nullrotation,colorassignment.at(icolumn).at(irow));
            UUIDs.push_back(UUID); //get UUIDs
        }
    }

    context->rotatePrimitive(UUIDs,rotationrad.x, make_vec3(1,0,0));  //rotate color board
    context->rotatePrimitive(UUIDs,rotationrad.y, make_vec3(0,1,0));
    context->rotatePrimitive(UUIDs,rotationrad.z, make_vec3(0,0,1));
    UUIDs_colorboard = UUIDs;
    return UUIDs;

}

// Set reflectivity for a specific UUID
void CameraCalibration::setColorboardReflectivity(const uint &UUID, const std::string &filename, const std::string &labelname) {
    std::vector<vec2> spectraldata;
    CameraCalibration::loadXMLlabeldata(filename,labelname,spectraldata);
    context->setPrimitiveData(UUID, "reflectivity_spectrum", labelname);
    context->setPrimitiveData(UUID, "reflectivity_spectrum_raw", labelname+"_raw");
    context->setPrimitiveData(UUID, "transmissivity_spectrum", "");
    context->setPrimitiveData(UUID, "twosided_flag", uint(0) );
    context->setGlobalData(labelname.c_str(),HELIOS_TYPE_VEC2,spectraldata.size(),&spectraldata[0]);
    context->setGlobalData((labelname+"_raw").c_str(),HELIOS_TYPE_VEC2,spectraldata.size(),&spectraldata[0]);
}

// Add default color board (DKC-RPO) with spectral reflectivity values
std::vector<uint> CameraCalibration::addDefaultColorboard(const helios::vec3 &centrelocation, const helios::vec3 &rotationrad, float patchsize){

    if (!UUIDs_colorboard.empty()){
        context->deletePrimitive(UUIDs_colorboard);
        std::cout << "WARNING: Default color board has been reset"<< std::endl;
    }

    std::vector<uint> UUIDs = CameraCalibration::addColorboard(centrelocation, rotationrad ,colorassignment_default,patchsize);

    CameraCalibration::setDefaultColorBoardSpectra();

    return UUIDs;
}

void CameraCalibration::setDefaultColorBoardSpectra(){

    uint colornumber=colorassignment_default.size()*colorassignment_default.at(0).size();
    std::string numberstr;
    std::vector<vec2> spectraldata;
    std::string filename;
    std::string labelname;

    // Color board spectra XML file ID
    // white to black:  1,  2,  3,  4,  5,  6
    // basic colors:    7,  8,  9, 10, 11, 12
    // bottom colors:   13, 14, 15, 16, 17, 18

    for (uint UUID:UUIDs_colorboard){
        numberstr = std::to_string(colornumber);
        if (numberstr.size() < 2) {
            numberstr = "0" + numberstr;
        }
        filename= "plugins/radiation/spectral_data/color_board/ColorReference_"+numberstr+".xml";
        labelname= "ColorReference_"+numberstr;
        CameraCalibration::setColorboardReflectivity(UUID, filename, labelname);
        colornumber=colornumber-1;
    }
}

std::vector<uint> CameraCalibration::getColorBoardUUIDs(){
    return UUIDs_colorboard;
}

bool CameraCalibration::writeSpectralXMLfile(const std::string &filename, const std::string &note, const std::string &label, std::vector<vec2> *spectrum){

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
            newspectraldata <<"\n\t\t";
            newspectraldata << spectrum->at(i).x;
            newspectraldata << " ";
            newspectraldata << std::to_string(spectrum->at(i).y);
        }
//            spectrum->clear();
        newspectraldata <<"\n\t</globaldata_vec2>\n";
        newspectraldata <<"\n</helios>";
        newspectraldata.close();
        return true;
    }

    else{
        std::cerr << "\n(CameraCalibration::writeSpectralXMLfile) Unable to open file";
        return false;

    }
}

// Load XML file and save data in spectral vectors containing both wavelengths and spectral values
bool CameraCalibration::loadXMLlabeldata(const std::string &filename,const std::string &labelname,std::vector<vec2> &spectraldata){

    Context context_temporary;
    context_temporary.loadXML(filename.c_str(), true);

    if (context_temporary.doesGlobalDataExist(labelname.c_str())){
        context_temporary.getGlobalData(labelname.c_str(),spectraldata);
        return true;
    }
    else{
        std::cerr << "\n(CameraCalibration::loadXMLlabeldata) Cannot find the label "<<labelname<<" in XML file:"<<filename;
        return false;
    }
}

float CameraCalibration::GradientDescent(std::vector<std::vector<float>> *expandedcameraspectra, const std::vector<std::vector<float>> &expandedconstinput,
                                         const float &learningrate, const std::vector<std::vector<float>> &truevalues) {

    size_t boardnumber=expandedconstinput.size();
    size_t wavelengthnumber = expandedconstinput.at(0).size();
    size_t bandnumber = expandedcameraspectra->size();

    float iloss = 0;
    for (int iband = 0; iband < expandedcameraspectra->size(); iband++){

        //Calculate errors
        std::vector<float> output(boardnumber,0);
        std::vector<float> errors(boardnumber);
        for (int iboardn = 0; iboardn < boardnumber; ++iboardn){
            for (int ispecn = 0; ispecn < wavelengthnumber; ++ispecn){
                output.at(iboardn) += expandedcameraspectra->at(iband).at(ispecn) * expandedconstinput.at(iboardn).at(ispecn);
            }
            errors.at(iboardn)=truevalues.at(iband).at(iboardn)-output.at(iboardn);

            //Calculate root mean square error as loss function
            iloss += (errors.at(iboardn) * errors.at(iboardn)) / float(boardnumber)/float(bandnumber);
        }

        //Update extended spectrum
        std::vector<float> despectrum(wavelengthnumber);
        for (int ispecn = 0; ispecn < wavelengthnumber; ++ispecn){
            for (int iboardn = 0; iboardn < boardnumber; ++iboardn){
                despectrum.at(ispecn) += errors.at(iboardn) * expandedconstinput.at(iboardn).at(ispecn);
            }
            expandedcameraspectra->at(iband).at(ispecn) += learningrate * despectrum.at(ispecn);

            // Non-negative constrain
            if (expandedcameraspectra->at(iband).at(ispecn) < 0){
                expandedcameraspectra->at(iband).at(ispecn) -= learningrate * despectrum.at(ispecn);
            }
        }
    }
    return iloss;
}

std::vector<float> CameraCalibration::expandSpectrum(const std::vector<helios::vec2>& targetspectrum, float normvalue=1) {

    std::vector<float> extendedspectrum;
    extendedspectrum.reserve(targetspectrum.size());

    for (vec2 spectralvalue : targetspectrum){
        extendedspectrum.push_back(spectralvalue.y/normvalue);
    }

    extendedspectrum.insert(extendedspectrum.end()-1, extendedspectrum.begin()+1, extendedspectrum.end()-1);
    return extendedspectrum;
}

static float normalizevalue( std::vector<std::vector<helios::vec2>> cameraresponsespectra, const std::map<uint, std::vector<vec2>> &simulatedinputspectra){

    // Find the maximum value of the simulated input spectra multiplied by the camera response spectra
    float normvalue = 0;
    for (const auto & cameraspectrum : cameraresponsespectra){
        for(const auto& inputspectrum : simulatedinputspectra){
            float outputvalue = 0;
            for(int iwave = 1; iwave<inputspectrum.second.size()-1; iwave++){
                outputvalue+=inputspectrum.second.at(iwave).y*cameraspectrum.at(iwave).y*2;
            }
            outputvalue+=inputspectrum.second.at(0).y*cameraspectrum.at(0).y;
            outputvalue+=inputspectrum.second.back().y*cameraspectrum.back().y;
            if(outputvalue>normvalue){
                normvalue= outputvalue;
            }
        }
    }
    return normvalue;
}


std::vector<float> CameraCalibration::updateCameraResponseSpectra(const std::vector<std::string>& camerareponselabels, const std::string &label,
                                                                  const std::map<uint, std::vector<vec2>> &simulatedinputspectra,
                                                                  const std::vector<std::vector<float>> &truevalues) {

    float learningrate = responseupdateparameters.learningrate;
    int maxiteration = responseupdateparameters.maxiteration;
    float minloss = responseupdateparameters.minloss;
    std::vector<float> camerarescales = responseupdateparameters.camerarescales;

    std::vector<std::vector<helios::vec2>> cameraresponsespectra;
    for (std::string cameraresponselabel:camerareponselabels){
        std::vector<vec2> cameraresponsespectrum;
        context->getGlobalData(cameraresponselabel.c_str(),cameraresponsespectrum);
        cameraresponsespectra.push_back(cameraresponsespectrum);
    }

    // Get the highest value of color board by using the original camera response used for normalization
    float normvalue = normalizevalue(cameraresponsespectra, simulatedinputspectra);

    std::vector<std::vector<float>> expandedcameraspectra;
    uint bandsnumber = cameraresponsespectra.size();
    for (const auto& cameraspectrum : cameraresponsespectra){
        expandedcameraspectra.push_back(CameraCalibration::expandSpectrum(cameraspectrum, normvalue));
    }

    std::vector<std::vector<float>> expandedinputspectra;
    expandedinputspectra.reserve(simulatedinputspectra.size());
    for (const auto& inputspectrum : simulatedinputspectra){
        expandedinputspectra.push_back(CameraCalibration::expandSpectrum(inputspectrum.second));
    }

    // Update expanded camera response spectra
    std::vector<float> loss;
    float initialloss;
    loss.reserve(maxiteration);
    float stopiteration = maxiteration;
    for (int iloop=0; iloop < maxiteration; ++iloop){
        float iloss = CameraCalibration::GradientDescent(&expandedcameraspectra, expandedinputspectra, learningrate, truevalues) / float(bandsnumber);
        loss.push_back(iloss);
        if (iloss<minloss){
            stopiteration = iloop;
            break;
        }
        // Automatically change learning rate
        if (iloop==0){
            initialloss = iloss;
        }
        else if(iloss>0.5*initialloss){
            learningrate = learningrate * 2;
        }
    }

    // Get the calibrated camera response spectra

    for (int iband=0; iband < camerareponselabels.size(); iband++){
        calibratedcameraspectra[camerareponselabels.at(iband)] = cameraresponsespectra.at(iband);
        for ( int ispec = 0; ispec < cameraresponsespectra.at(iband).size()-1; ispec ++){
            calibratedcameraspectra[camerareponselabels.at(iband)].at(ispec).y= expandedcameraspectra.at(iband).at(ispec) * camerarescales.at(iband);
        }
        calibratedcameraspectra[camerareponselabels.at(iband)].back().y = expandedcameraspectra.at(iband).back() * camerarescales.at(iband);
        std::string calibratedlabel = label+ "_" + camerareponselabels.at(iband);
        context->setGlobalData(calibratedlabel.c_str(),HELIOS_TYPE_VEC2,calibratedcameraspectra.at(camerareponselabels.at(iband)).size(),
                               &calibratedcameraspectra.at(camerareponselabels.at(iband))[0]);
    }

    // Calculate the final loss
    float iloss = CameraCalibration::GradientDescent(&expandedcameraspectra, expandedinputspectra, learningrate, truevalues) / float(bandsnumber);
    std::cout<<"The final loss after " << stopiteration << " iteration is: "<<iloss <<std::endl;
    loss.push_back(iloss);
    return loss;
}

void CameraCalibration::writeCalibratedCameraResponses(const std::vector<std::string>& camerareponselabels, const std::string &calibratemark, float scale){
    // write the calibrated camera response spectra in xml files and set them as global data

    for (const std::string& cameraresponselabel:camerareponselabels){
        std::vector<vec2> cameraresponsespectrum = calibratedcameraspectra[cameraresponselabel];
        std::string calibratedlabel = calibratemark+ "_" + cameraresponselabel;
        for (int ispec = 0; ispec < cameraresponsespectrum.size(); ispec ++){
            cameraresponsespectrum.at(ispec).y= cameraresponsespectrum.at(ispec).y * scale;
        }
        context->setGlobalData(calibratedlabel.c_str(),HELIOS_TYPE_VEC2,cameraresponsespectrum.size(),&cameraresponsespectrum[0]);
        CameraCalibration::writeSpectralXMLfile(calibratedlabel + ".xml", "", calibratedlabel, &cameraresponsespectrum);
    }
}

void CameraCalibration::distortImage(const std::string& cameralabel, const std::vector<std::string>& bandlabels,
                                     const helios::vec2 &focalxy, std::vector<double> &distCoeffs, helios::int2 cameraresolution) {

    helios::int2 camerareoslutionR = cameraresolution;

    // Original image dimensions
    int cols = cameraresolution.x;
    int rows = cameraresolution.y;

//    float PPointsRatiox =1.052f;
//    float PPointsRatioy =0.999f;

    // Distorted image dimensions diff
    int cols_dif = (cols - camerareoslutionR.x)/2;
    int rows_dif = (rows - camerareoslutionR.y)/2;

    for (int ib = 0; ib<bandlabels.size(); ib++){
        std::string global_data_label = "camera_" + cameralabel + "_" + bandlabels.at(ib);
        std::vector<float> cameradata;

        context->getGlobalData(global_data_label.c_str(), cameradata);

        std::vector<float> distorted_cameradata(camerareoslutionR.x * camerareoslutionR.y, 0);

        // Compute the undistorted image
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < cols; i++) {
                // Compute the undistorted pixel coordinates
                double x = (i - cols/2) / focalxy.x;
                double y = (j - rows/2) / focalxy.y;
                // Apply distortion
                double r2 = x*x + y*y;
                double xDistorted = x * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[2] * x * y + distCoeffs[3] * (r2 + 2 * x * x);
                double yDistorted = y * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[3] * x * y + distCoeffs[2] * (r2 + 2 * y * y);

                // Compute the distorted pixel coordinates
                int xPixel = int(round(xDistorted * focalxy.x + cols/2));
                int yPixel = int(round(yDistorted * focalxy.y + rows/2));

                // Set the distorted pixel value
                if (xPixel >= cols_dif && xPixel < cols-cols_dif  && yPixel >= rows_dif && yPixel <  rows-rows_dif) {
                    int xPos = xPixel - cols_dif;
                    int yPos = yPixel - rows_dif;
                    if(yPos * camerareoslutionR.x + xPos < distorted_cameradata.size()){
                        distorted_cameradata.at(yPos * camerareoslutionR.x + xPos) = cameradata.at(j * cameraresolution.x + i);
                    }
                }
            }
        }
        context->setGlobalData(global_data_label.c_str(), HELIOS_TYPE_FLOAT, distorted_cameradata.size(), &distorted_cameradata[0]);
    }

}

void undistortImage(std::vector<std::vector<unsigned char>> &image,
                    std::vector<std::vector<unsigned char>> &undistorted,
                    std::vector<std::vector<double>> &K,
                    std::vector<double> &distCoeffs) {
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
            double r2 = x*x + y*y;
            double xDistorted = x * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[2] * x * y + distCoeffs[3] * (r2 + 2 * x * x);
            double yDistorted = y * (1 + distCoeffs[0] * r2 + distCoeffs[1] * r2 * r2) + 2 * distCoeffs[3] * x * y + distCoeffs[2] * (r2 + 2 * y * y);

            // Compute the distorted pixel coordinates
            int xPixel = round(xDistorted * K[0][0] + K[0][2]);
            int yPixel = round(yDistorted * K[1][1] + K[1][2]);

            // Set the undistorted pixel value
            if (xPixel >= 0 && xPixel < cols && yPixel >= 0 && yPixel < rows) {
                undistorted[r][c] = image[yPixel][xPixel] ;
            }
        }
    }
}

static void wavelengthboundary(float &lowwavelength, float &highwavelength, const std::vector<vec2>& spectrum){

    if (spectrum.back().x<highwavelength){
        highwavelength = spectrum.back().x;
    }
    if (spectrum.at(0).x>lowwavelength){
        lowwavelength = spectrum.at(0).x;
    }
}

void CameraCalibration::preprocessSpectra(const std::vector<std::string>& sourcelabels, const std::vector<std::string>& cameralabels,
                                          std::vector<std::string>& objectlabels, vec2 &wavelengthrange, const std::string& targetlabel){

    std::map<std::string,std::map<std::string, std::vector<vec2>>> allspectra;

    // Extract and check source spectra from global data
    std::map<std::string, std::vector<vec2>> Source_spectra;
    for (const std::string& sourcelable:sourcelabels){
        if (context->doesGlobalDataExist(sourcelable.c_str())){
            std::vector<vec2> Source_spectrum;
            context->getGlobalData(sourcelable.c_str(), Source_spectrum);
            Source_spectra.emplace(sourcelable, Source_spectrum);
            wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Source_spectrum);
        }
        else {std::cout << "WARNING: Source ("<< sourcelable <<") does not exist in global data"<< std::endl;}
    }
    allspectra.emplace("source", Source_spectra);

    // Extract and check camera spectra from global data
    std::map<std::string, std::vector<vec2>> Camera_spectra;
    for (const std::string& cameralabel:cameralabels){
        if (context->doesGlobalDataExist(cameralabel.c_str())) {
            std::vector<vec2> Camera_spectrum;
            context->getGlobalData(cameralabel.c_str(), Camera_spectrum);
            Camera_spectra.emplace(cameralabel, Camera_spectrum);
            wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Camera_spectrum);
        }
        else {std::cout << "WARNING: Camera ("<< cameralabel <<") does not exist in global data"<< std::endl;}
    }
    allspectra.emplace("camera", Camera_spectra);

    // Extract and check object spectra from global data
    std::map<std::string, std::vector<vec2>> Object_spectra;
    if (!objectlabels.empty()){
        for (const std::string& objectlable:objectlabels){
            if (context->doesGlobalDataExist(objectlable.c_str())) {
                std::vector<vec2> Object_spectrum;
                context->getGlobalData(objectlable.c_str(), Object_spectrum);
                Object_spectra.emplace(objectlable, Object_spectrum);
                wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Object_spectrum);
            }
            else {std::cout << "WARNING: Object ("<< objectlable <<") does not exist in global data"<< std::endl;}
        }
    }

    // Check if object spectra in global data has been added to UUIDs but are not in the provided object labels;
    std::vector<uint> exist_UUIDs = context->getAllUUIDs();
    for (uint UUID:exist_UUIDs){
        if (context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
            std::string spectralreflectivitylabel;
            context->getPrimitiveData(UUID, "reflectivity_spectrum", spectralreflectivitylabel);
            if (context->doesGlobalDataExist(spectralreflectivitylabel.c_str())) {
                if (std::find(objectlabels.begin(), objectlabels.end(), spectralreflectivitylabel) == objectlabels.end()) {
                    objectlabels.push_back(spectralreflectivitylabel);
//                    std::cout << "WARNING: Spectrum (" << spectralreflectivitylabel << ") has been added to UUID (" << UUID << ") but is not in the provided object spectral labels" << std::endl;
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
                    std::cout << "WARNING: Spectrum (" << spectraltransmissivitylabel << ") has been added to UUID (" << UUID << ") but is not in the provided object labels" << std::endl;
                    std::vector<vec2> Object_spectrum;
                    context->getGlobalData(spectraltransmissivitylabel.c_str(), Object_spectrum);
                    Object_spectra.emplace(spectraltransmissivitylabel, Object_spectrum);
                    wavelengthboundary(wavelengthrange.x, wavelengthrange.y, Object_spectrum);
                }
            }
        }
    }
    allspectra.emplace("object", Object_spectra);

    // interpolate spectra
    // set unifying wavelength resolution for all spectra according to resolution of target spectrum
    std::vector<vec2> target_spectrum;
    if (targetlabel.empty()){
        target_spectrum = allspectra.at("object").at(objectlabels.at(0));
    }
    else if (std::find(objectlabels.begin(),objectlabels.end(),targetlabel)==objectlabels.end()){
        std::cout << "WARNING (CameraCalibration::preprocessSpectra()): target label ("<< targetlabel <<") is not a member of object labels"<< std::endl;
        target_spectrum = allspectra.at("object").at(objectlabels.at(0));
    }
    else{target_spectrum = allspectra.at("object").at(targetlabel);}

    for (const auto& spectralgrouppair: allspectra){
        for (const auto& spectrumpair : spectralgrouppair.second){
            std::vector<vec2> cal_spectrum;
            for (auto ispectralvalue:target_spectrum){
                if(ispectralvalue.x>wavelengthrange.y){
                    context->setGlobalData(spectrumpair.first.c_str(), HELIOS_TYPE_VEC2, cal_spectrum.size(), &cal_spectrum[0]);
                    break;
                }
                if(ispectralvalue.x>=wavelengthrange.x){
                    cal_spectrum.push_back(make_vec2(ispectralvalue.x,interp1(spectrumpair.second, ispectralvalue.x)));
                }
            }
            allspectra[spectralgrouppair.first][spectrumpair.first] = cal_spectrum;
        }
    }
    processedspectra = allspectra;

    // store wavelengths into global data
    std::vector<float> wavelengths;
    for (auto ispectralvalue:target_spectrum){
        if(ispectralvalue.x>wavelengthrange.y || ispectralvalue.x == target_spectrum.back().x){
            context->setGlobalData("wavelengths", HELIOS_TYPE_FLOAT, wavelengths.size(), &wavelengths[0]);
            break;
        }
        if(ispectralvalue.x>=wavelengthrange.x){
            wavelengths.push_back(ispectralvalue.x);
        }
    }

}

float CameraCalibration::getCameraResponseScale(const std::string &cameralabel, const helios::int2 cameraresolution, const std::vector<std::string> &bandlabels,
                                                const std::vector<std::vector<float>> &truevalues) {

    std::vector<uint> camera_UUIDs;
    std::string global_UUID_label = "camera_" + cameralabel + "_pixel_UUID";
    context->getGlobalData(global_UUID_label.c_str(), camera_UUIDs);

    float dotsimreal = 0;
    float dotsimsim = 0;
    for (int ib = 0; ib<bandlabels.size(); ib++){

        std::vector<float> camera_data;
        std::string global_data_label = "camera_" + cameralabel + "_" + bandlabels.at(ib);
        context->getGlobalData(global_data_label.c_str(), camera_data);

        for (int icu = 0; icu<UUIDs_colorboard.size(); icu++){

            float count = 0;
            float sum = 0;
            for (uint j = 0; j < cameraresolution.y; j++) {
                for (uint i = 0; i < cameraresolution.x; i++) {
                    uint UUID = camera_UUIDs.at(j*cameraresolution.x + i)-1;
                    if (UUID == UUIDs_colorboard.at(icu)){
                        float value = camera_data.at(j*cameraresolution.x + i);
                        sum += value;
                        count += 1;
                    }
                }
            }

            float sim_cv = sum / count;
            float real_cv = truevalues.at(ib).at(icu);
            dotsimreal += (sim_cv*real_cv);
            dotsimsim += (sim_cv*sim_cv);
        }
    }
    float fluxscale = dotsimreal / dotsimsim;
    return fluxscale;
}

std::vector<uint> CameraCalibration::readROMCCanopy(){
    std::ifstream inputFile("plugins/radiation/spectral_data/external_data/HET01_UNI_scene.def");
    std::vector<uint> iCUUIDs;
    float idata;
    // read the data from the file
    while (!inputFile.eof()) {
        std::vector<float> poslf;
        std::vector<float> rotatelf_cos;
        std::vector<float> rotatelf_sc(2);
        auto widthlengthlf = float(sqrt(0.01f*M_PI));
        for (int i = 0; i < 7; i++) {
            inputFile >> idata;

            if(i>0 && i<4){
                poslf.push_back(idata);
            }
            else if (i>3){
                rotatelf_cos.push_back(idata);
            }
        }

        rotatelf_sc.at(0) = float(asin(rotatelf_cos.at(2)));
        rotatelf_sc.at(1) =  float(atan2(rotatelf_cos.at(1), rotatelf_cos.at(0)));
        if ( rotatelf_sc.at(1) < 0) {
            rotatelf_sc.at(1) += 2 * M_PI;
        }
        vec3 iposlf = make_vec3(poslf.at(0),poslf.at(1),poslf.at(2));
        iCUUIDs.push_back(context->addPatch(iposlf,vec2(widthlengthlf,widthlengthlf), make_SphericalCoord(float(0.5*M_PI+rotatelf_sc.at(0)), rotatelf_sc.at(1))));
    }
    inputFile.close();
    return iCUUIDs;
}

