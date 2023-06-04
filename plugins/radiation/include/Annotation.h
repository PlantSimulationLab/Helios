//
// Created by tlei on 5/25/23.
//

#ifndef HELIOS_ANNOTATION_H
#define HELIOS_ANNOTATION_H
#include "Context.h"
#include <fstream>

class Annotation {

public:
    Annotation(helios::Context *context, const std::string &cameralabel);

    //! Function to write the label image of all primitives to a ".txt" file
    /**
     * \param[in] filename Name of the output file
     * \param[in] labelname Name of the primitive data
     * \param[in] camera_resolution Resolution of the camera
     * \param[in] datatype Type of the primitive data
    */
    void getBasicLabel(const std::string &filename, const std::string &labelname, helios::HeliosDataType datatype, float padvalue = NAN);

    void setCameraResolution(const helios::int2 &camera_resolution);

    //! Parameter struct constructor for net photosynthesis labelling
    struct PhotosynthesisParameters{
        float Vcmax = 103.9861;
        float Jmax = 179.0330;
        float Ci = 665.0820;
        float Kc = 394.6921;
        float Ko = 273.3198;
        float Oi = 213.5000;
        float alpha = 0.24;
        float Gamma_star = 42.3161;
        float Rd = 1.0582;

        float avc = 0.54;
        float bvc = 55.28;
        float ajv = 0.89;
        float bjv = 1.01;
        float dVcmax = 0.9717;
        float dJmax = 0.9986;

        PhotosynthesisParameters()= default;
    };

    //! Parameters for net photosynthesis labelling
    PhotosynthesisParameters photosynthesisParameters;

    //! Function to write the net photosynthesis label image of all primitives to a ".txt" file (link chlorophyll content to Vcmax))
    /**
     * \param[in] filename Name of the output file
     * \param[in] leaflabel Name of the leaf label
     * \param[in] chllabel Name of the chlorophyll label
    */
    void getNetPhotosynthesis(const std::string &filename, const std::string &leaflabel, const std::string &chllabel, float sourcefluxscale = 1.0f);

    //! Function to write the net photosynthesis label image of all primitives to a ".txt" file (with given Vcmax and Jmax)
    /**
     * \param[in] filename Name of the output file
     * \param[in] leaflabel Name of the leaf label
    */
    void getNetPhotosynthesis(const std::string &filename, const std::string &leaflabel, float sourcefluxscale = 1.0f);

    //! Function to write the depth image of all primitives to a ".txt" file
    /**
     * \param[in] filename Name of the output file
     * \param[in] camera_position Position of the camera
     * \param[in] camera_lookat Look at vector of the camera
    */
    void getDepthImage(const std::string &filename, const helios::vec3 &camera_position, const helios::vec3 &camera_lookat);

protected:

    float runNetPhotosynthesis( float sourceflux, float chlorophyllcontent, Annotation::PhotosynthesisParameters photosynthesisParameters);

    float runNetPhotosynthesis( float sourceflux, Annotation::PhotosynthesisParameters photosynthesisParameters);

    float e = 2.71828182845904523536;

    helios::Context *context;

    std::vector<uint> pixel_UUIDs;

    helios::int2 camera_resolution;
};
#endif //HELIOS_ANNOTATION_H
