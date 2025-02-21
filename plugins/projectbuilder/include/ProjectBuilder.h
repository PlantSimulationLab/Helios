/** \file "ProjectBuilder.h" ProjectBuilder header.

Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef PROJECT_BUILDER
#define PROJECT_BUILDER

//#pragma once

#include <chrono>
#include <set>
#include <thread>
#include <iostream>

#include "Context.h"
#include <pugixml.hpp>
#include "InitializeSimulation.h"

// Forward Declaration
class BLConductanceModel;
class EnergyBalanceModel;
class PlantArchitecture;
class RadiationModel;
class SolarPosition;
class Visualizer;
class CameraProperties;
void BuildGeometry(const std::string &xml_input_file, PlantArchitecture *plant_architecture_ptr, helios::Context *context_ptr);
void InitializeRadiation(const std::string &xml_input_file, SolarPosition *solarposition_ptr, RadiationModel *radiation_ptr, helios::Context *context_ptr );
void InitializeEnergyBalance(const std::string &xml_input_file, BLConductanceModel *boundarylayerconductancemodel, EnergyBalanceModel *energybalancemodel, helios::Context *context_ptr);
void InitializeSimulation(const std::string &xml_input_file, helios::Context *context_ptr );


#ifdef ENABLE_BOUNDARYLAYERCONDUCTANCEMODEL
    #include "BoundaryLayerConductanceModel.h"
#endif //BOUNDARYLAYERCONDUCTANCEMODEL

#ifdef ENABLE_ENERGYBALANCEMODEL
    #include "EnergyBalanceModel.h"
#endif //ENERGYBALANCEMODEL

#if defined(ENABLE_BOUNDARYLAYERCONDUCTANCEMODEL) && defined(ENABLE_ENERGYBALANCEMODEL)
    #include "InitializeEnergyBalance.h"
#endif //BOUNDARYLAYERCONDUCTANCEMODEL && ENERGYBALANCEMODEL

#ifdef ENABLE_PLANT_ARCHITECTURE
    #include "PlantArchitecture.h"
    #include "BuildGeometry.h"
#endif //PLANT_ARCHITECTURE

#ifdef ENABLE_RADIATION_MODEL
    #include "RadiationModel.h"
    #include "InitializeRadiation.h"
#endif //RADIATION_MODEL

#ifdef ENABLE_SOLARPOSITION
    #include "SolarPosition.h"
#endif //SOLARPOSITION

#ifdef ENABLE_HELIOS_VISUALIZER
    #include "glew.h"
    #include "Visualizer.h"
    // IMGUI
    #include "imgui.h"
    #include "imgui_internal.h"
    #include "backends/imgui_impl_glfw.h"
    #include "backends/imgui_impl_opengl3.h"
    #include "misc/cpp/imgui_stdlib.h"
    #include "GLFW/glfw3.h"
#endif //HELIOS_VISUALIZER


//! Function to convert vector to string
/**
 * \param[in] v Vector input
*/
std::string vec_to_string(const helios::vec2& v);

//! Function to convert vector to string
/**
 * \param[in] v Vector input
*/
std::string vec_to_string(const helios::vec3& v);

//! Function to convert vector to string
/**
 * \param[in] v Vector input
*/
std::string vec_to_string(const helios::int2& v);

//! Function to linearly space between
/**
 * \param[in] a Start point coordinates
 * \param[in] b End point coordinates
 * \param[in] num_points Total number of points (including a & b)
*/
std::vector<helios::vec3> linspace(helios::vec3 a, helios::vec3 b, int num_points);

//! Function to return interpolated vector based on keypoints
/**
 * \param[in] keypoints Vector of keypoints
 * \param[in] positions Vector of position vectors
 * \param[in] num_points Total number of points (including points in positions)
*/
std::vector<helios::vec3> interpolate(std::vector<int> keypoints, std::vector<helios::vec3> positions, int num_points);

//! ImGUI toggle button function
/**
 * \param[in] str_id String ID
 * \param[in] v Boolean representing on or off
*/
void toggle_button(const char* str_id, bool* v);

//! Function to open file dialog
std::string file_dialog();

//! Function to get node labels for a given set of nodes
/**
 * \param[in] xml_file Path to the XML file
 * \param[in] label_name Name of the label
 * \param[in] node_name Name of the XML nodes to get labels from
*/
std::vector<std::string> get_xml_node_values(std::string xml_file, const std::string& label_name, const std::string& node_name);


class ProjectBuilder {
  private:
    //! XML Document
    pugi::xml_document xmldoc;

    //! XML Document path
    std::string xml_input_file;

    //! User input
    bool user_input;

    //! Absorbed PAR value
    float PAR_absorbed;

    //! Absorbed NIR value
    float NIR_absorbed;

    //! Absorbed LW value
    float LW_absorbed;

    //! Turbidity
    float turbidity;

    //! Diffuse extinction coefficient
    float diffuse_extinction_coeff = 0.1;

    //! Sun ID
    uint sun_ID;

    //! Context
    helios::Context *context = nullptr;

    //! Visualizer
    Visualizer *visualizer = nullptr;

    //! Plant Architecture
    PlantArchitecture *plantarchitecture = nullptr;

    //! Radiation Model
    RadiationModel *radiation = nullptr;

    //! Solar Position
    SolarPosition *solarposition = nullptr;

    //! Energy Balance Model
    EnergyBalanceModel *energybalancemodel = nullptr;

    //! Boundary Layer Conductance
    BLConductanceModel *boundarylayerconductance = nullptr;

    //! Camera Properties
    CameraProperties *cameraproperties = nullptr;

    //! Band Labels
    std::vector<std::string> bandlabels;

    //! Ground UUIDs
    std::vector<uint> ground_UUIDs;

    //! Leaf UUIDs
    std::vector<uint> leaf_UUIDs;

    //! Petiolule UUIDs
    std::vector<uint> petiolule_UUIDs;

    //! Petiole UUIDs
    std::vector<uint> petiole_UUIDs;

    //! Internode UUIDs
    std::vector<uint> internode_UUIDs;

    //! Peduncle UUIDs
    std::vector<uint> peduncle_UUIDs;

    //! Petal UUIDs
    std::vector<uint> petal_UUIDs;

    //! Pedicel UUIDs
    std::vector<uint> pedicel_UUIDs;

    //! Fruit UUIDs
    std::vector<uint> fruit_UUIDs;

    //! Primitive names
    std::vector<std::string> primitive_names = {"All", "Ground", "Leaf", "Petiolule", "Petiole", "Internode",
                                                "Peduncle", "Petal", "Pedicel", "Fruit"};

    //! Map keyed by primitive names that returns a vector of UUIDs corresponding to the primitive name
    std::map<std::string, std::vector<uint>*> primitive_types;

    //! Map keyed by primitive names that returns a bool representing whether the primitive has continuous spectra (reflectivity, transmissivity, emissivity)
    std::map<std::string, std::vector<bool>> primitive_continuous;

    //! Map keyed by primitive names that returns spectra (reflectivity, transmissivity, emissivity)
    std::map<std::string,  std::vector<std::string*>> primitive_spectra;

    //! Primitive values map: band -> primitive type -> {reflectivity, transmissivity, emissivity}
    std::map<std::string, std::map<std::string, std::vector<float*>>> primitive_values;

    //! Ground area
    float ground_area;

    //! Timeseries variables
    std::vector<std::string> timeseries_variables;

    //! Air temperature
    float air_temperature = 300.f;

    //! Air humidity
    float air_humidity = 0.5f;

    //! Sun direction vector
    helios::vec3 sun_dir_vec;

    //! R PAR dir
    float R_PAR_dir;

    //! R NIR dir
    float R_NIR_dir;

    //! fdiff
    float fdiff;

    //! XML Error String
    std::string xml_error_string;

    //! Rig labels
    std::vector<std::string> rig_labels;

    //! Camera positions
    std::vector<helios::vec3> camera_positions;

    //! Camera lookats
    std::vector<helios::vec3> camera_lookats;

    //! Vector containting the *first* camera label of every rig.
    std::vector<std::string> camera_labels;

    //! Vector of camera resolutions
    std::vector<helios::int2> camera_resolutions;

    //! Vector of focal plane distances for every camera
    std::vector<float> focal_plane_distances;

    //! Vector of lens diameters for every camera
    std::vector<float> lens_diameters;

    //! Vector of FOV aspect ratios for every camera
    std::vector<float> FOV_aspect_ratios;

    //! Vector of HFOVs for every camera
    std::vector<float> HFOVs;

    //! Map keyed by rig name that returns rig index
    std::map<std::string, int> rig_dict;

    //! Rig position
    helios::vec3 camera_position = {0,0,0};

    //! Vector of keypoint frames for every rig
    std::vector<std::vector<int>> keypoint_frames;

    //! Vector of rig positions for every rig
    std::vector<std::vector<helios::vec3>> camera_position_vec;

    //! Vector of rig lookat positions for every rig
    std::vector<std::vector<helios::vec3>> camera_lookat_vec;

    //! Rig lookat
    helios::vec3 camera_lookat = {0,0,0};

    //! Camera label
    std::string camera_label = "RGB";

    //! Number of images/frames
    int num_images = 5;

    //! Number of images/frames per rig
    std::vector<int> num_images_vec;

    //! Vector of camera names
    std::vector<std::string> camera_names;

    //! Camera resolution
    helios::int2 camera_resolution = {1024, 1024};

    //! Focal plane distance
    float focal_plane_distance = 0.4;

    //! Lens diameter
    float lens_diameter = 0.02;

    //! FOV aspect ratio
    float FOV_aspect_ratio = 1.4;

    //! HFOV
    float HFOV = 50.0;

    //! Dictionary keyed by camera name that returns camera index.
    std::map<std::string, int> camera_dict;

    //! Set of camera labels for every rig
    std::vector<std::set<std::string>> rig_camera_labels;

     //! Vector of light names.
    std::vector<std::string> light_names;

    //! Vector of light types (e.g. sphere, rectangle, etc.).
    std::vector<std::string> light_types;

    //! Vector of all possible light types.
    // std::vector<std::string> all_light_types = {"collimated", "sphere", "sunsphere", "rectangle", "disk"};
    std::vector<std::string> all_light_types = {"sphere", "rectangle", "disk"};

    //! Vector of light positions for each light.
    std::vector<helios::vec3> light_direction_vec;

    //! Vector of light spherical directions for each light.
    std::vector<helios::SphericalCoord> light_direction_sph_vec;

    //! Vector of rotations for each light.
    std::vector<helios::vec3> light_rotation_vec;

    //! Vector of sizes for each light.
    std::vector<helios::vec2> light_size_vec;

    //! Vector of sizes for each light.
    std::vector<float> light_radius_vec;

    //! Dictionary keyed by light name that returns light index (in light_names).
    std::map<std::string, int> light_dict;

    //! Vector of sets of lights. The i-th set in the vector contains the light names of the i-th rig.
    std::vector<std::set<std::string>> rig_light_labels;

    //! Dictionary containing arrow UUIDs for every arrow
    std::map<int, std::vector<uint>> arrow_dict;

    //! Arrow count
    int arrow_count = 0;

    //! Helios XML node
    pugi::xml_node helios;

    //! Latitude
    float latitude = 38.55;

    //! Longitude
    float longitude = 121.76;

    //! UTC offset
    int UTC_offset = 8;

    //! CSV weather file path
    std::string csv_weather_file = "../inputs/weather_data.csv";

    //! Domain origin
    helios::vec3 domain_origin = {0,0,0};

    //! Domain extent
    helios::vec2 domain_extent = {10,10};

    //! Ground resolution
    helios::int2 ground_resolution = {1,1};

    //! Ground texture file
    std::string ground_texture_file = "plugins/visualizer/textures/dirt.jpg";

    //! Vector of canopy labels
    std::vector<std::string> labels;

    //! Canopy origin
    helios::vec3 canopy_origin = {0,0,0};

    //! Vector of canopy origins for every canopy
    std::vector<helios::vec3> canopy_origins;

    //! Plant count
    helios::int2 plant_count = {1,1};

    //! Vector of plant counts for every canopy
    std::vector<helios::int2> plant_counts;

    //! Plant spacing
    helios::vec2 plant_spacing = {0.5,0.5};

    //! Vector of plant spacings for every canopy
    std::vector<helios::vec2> plant_spacings;

    //! Plant library name
    std::string plant_library_name = "cowpea";

    //! Vector of plant library names for every canopy
    std::vector<std::string> plant_library_names;

    //! Plant age
    float plant_age = 0;

    //! Vector of plant ages for all canopies
    std::vector<float> plant_ages;

    //! Ground clipping height
    float ground_clipping_height = 0;

    //! Vector of ground clipping heights for all canopies
    std::vector<float> ground_clipping_heights;

    //! Map of canopy name to canopy index
    std::map<std::string, int> canopy_labels;

    //! Direct ray count
    int direct_ray_count = 100;

    //! Diffuse ray count
    int diffuse_ray_count = 1000;

    //! Scattering depth
    int scattering_depth = 2;

    //! Air turbidity
    float air_turbidity = 0.05;

    //! XML spectral library files
    std::set<std::string> xml_library_files = {"plugins/radiation/spectral_data/leaf_surface_spectral_library.xml",
                                                 "plugins/radiation/spectral_data/soil_surface_spectral_library.xml"};

    //! Possible spectra vector from spectral library files
    std::vector<std::string> possible_spectra;

    //! Camera XML library files
    std::set<std::string> camera_xml_library_files = {"plugins/radiation/spectral_data/camera_spectral_library.xml"};

    //! Possible camera calibrations vector from camera library files
    std::vector<std::string> possible_camera_calibrations;

    //! Camera calibration selection for each camera
    std::vector<std::string> camera_calibrations;

    //! Light XML library files
    std::set<std::string> light_xml_library_files = {"plugins/radiation/spectral_data/light_spectral_library.xml"};

    //! Possible light spectra vector from light library files
    std::vector<std::string> possible_light_spectra;

    //! Spectra selection for each light
    std::vector<std::string> light_spectra;

    //! Solar direct spectrum
    std::string solar_direct_spectrum = "ASTMG173";

    //! Reflectivity (apply to all)
    float reflectivity = 0.0;

    //! Transmissivity (apply to all)
    float transmissivity = 0.0;

    //! Emissivity (apply to all)
    float emissivity = 0.0;

    //! Leaf reflectivity
    float leaf_reflectivity = 0.0;

    //! Leaf transmissivity
    float leaf_transmissivity = 0.0;

    //! Leaf emissivity
    float leaf_emissivity = 0.0;

    //! Ground reflectivity
    float ground_reflectivity = 0.0;

    //! Ground transmissivity
    float ground_transmissivity = 0.0;

    //! Ground emissivity
    float ground_emissivity = 0.0;

    //! Petiolule reflectivity
    float petiolule_reflectivity = 0.0;

    //! Petiolule transmissivity
    float petiolule_transmissivity = 0.0;

    //! Petiolule emissivity
    float petiolule_emissivity = 0.0;

    //! Petiole reflectivity
    float petiole_reflectivity = 0.0;

    //! Petiole transmissivity
    float petiole_transmissivity = 0.0;

    //! Petiole emissivity
    float petiole_emissivity = 0.0;

    //! Internode reflectivity
    float internode_reflectivity = 0.0;

    //! Internode transmissivity
    float internode_transmissivity = 0.0;

    //! Internode emissivity
    float internode_emissivity = 0.0;

    //! Peduncle reflectivity
    float peduncle_reflectivity = 0.0;

    //! Peduncle transmissivity
    float peduncle_transmissivity = 0.0;

    //! Peduncle emissivity
    float peduncle_emissivity = 0.0;

    //! Petal reflectivity
    float petal_reflectivity = 0.0;

    //! Petal transmissivity
    float petal_transmissivity = 0.0;

    //! Petal emissivity
    float petal_emissivity = 0.0;

    //! Pedicel reflectivity
    float pedicel_reflectivity = 0.0;

    //! Pedicel transmissivity
    float pedicel_transmissivity = 0.0;

    //! Pedicel emissivity
    float pedicel_emissivity = 0.0;

    //! Fruit reflectivity
    float fruit_reflectivity = 0.0;

    //! Fruit transmissivity
    float fruit_transmissivity = 0.0;

    //! Fruit emissivity
    float fruit_emissivity = 0.0;

    //! Reflectivity spectrum (applies to all)
    std::string reflectivity_spectrum = "";

    //! Transmissivity spectrum (applies to all)
    std::string transmissivity_spectrum = "";

    //! Emissivity spectrum (applies to all)
    std::string emissivity_spectrum = "";

    //! Leaf reflectivity spectrum
    std::string leaf_reflectivity_spectrum = "grape_leaf_reflectivity_0000";

    //! Leaf transmissivity spectrum
    std::string leaf_transmissivity_spectrum = "grape_leaf_transmissivity_0000";

    //! Leaf emissivity spectrum
    std::string leaf_emissivity_spectrum = "";

    //! Ground reflectivity spectrum
    std::string ground_reflectivity_spectrum = "soil_reflectivity_0000";

    //! Ground transmissivity spectrum
    std::string ground_transmissivity_spectrum = "";

    //! Ground emissivity spectrum
    std::string ground_emissivity_spectrum = "";

    //! Petiolule reflectivity spectrum
    std::string petiolule_reflectivity_spectrum = "";

    //! Petiolule transmissivity spectrum
    std::string petiolule_transmissivity_spectrum = "";

    //! Petiolule emissivity spectrum
    std::string petiolule_emissivity_spectrum = "";

    //! Petiole reflectivity spectrum
    std::string petiole_reflectivity_spectrum = "";

    //! Petiole transmissivity spectrum
    std::string petiole_transmissivity_spectrum = "";

    //! Petiole emissivity spectrum
    std::string petiole_emissivity_spectrum = "";

    //! Internode reflectivity spectrum
    std::string internode_reflectivity_spectrum = "";

    //! Internode transmissivity spectrum
    std::string internode_transmissivity_spectrum = "";

    //! Internode emissivity spectrum
    std::string internode_emissivity_spectrum = "";

    //! Peduncle reflectivity spectrum
    std::string peduncle_reflectivity_spectrum = "";

    //! Peduncle transmissivity spectrum
    std::string peduncle_transmissivity_spectrum = "";

    //! Peduncle emissivity spectrum
    std::string peduncle_emissivity_spectrum = "";

    //! Petal reflectivity spectrum
    std::string petal_reflectivity_spectrum = "";

    //! Petal transmissivity spectrum
    std::string petal_transmissivity_spectrum = "";

    //! Petal emissivity spectrum
    std::string petal_emissivity_spectrum = "";

    //! Pedicel reflectivity spectrum
    std::string pedicel_reflectivity_spectrum = "";

    //! Pedicel transmissivity spectrum
    std::string pedicel_transmissivity_spectrum = "";

    //! Pedicel emissivity spectrum
    std::string pedicel_emissivity_spectrum = "";

    //! Fruit reflectivity spectrum
    std::string fruit_reflectivity_spectrum = "";

    //! Fruit transmissivity spectrum
    std::string fruit_transmissivity_spectrum = "";

    //! Fruit emissivity spectrum
    std::string fruit_emissivity_spectrum = "";

    //! All possible visualization types
    std::set<std::string> visualization_types = {"radiation_flux_PAR", "radiation_flux_NIR", "radiation_flux_LW"};

    //! Visualization type
    std::string visualization_type = "RGB";

    //! Currently selected canopy in the GUI
    std::string current_canopy;

    //! Currently selected rig in the GUI
    std::string current_rig;

    //! Currently selected camera in the GUI
    std::string current_cam;

    //! Currently selected light in the GUI
    std::string current_light;

    //! Currently selected keypoint in the GUI
    std::string current_keypoint;

    //! Currently selected primitive in the GUI
    std::string current_primitive = "All";

    //! Currently selected radiation band for reflectivity in the GUI
    std::string current_band_reflectivity = "red";

    //! Currently selected radiation band for transmissivity in the GUI
    std::string current_band_transmissivity = "red";

    //! Currently selected radiation band for emissivity in the GUI
    std::string current_band_emissivity = "red";

    //! Depth MVP matrix
//    #ifdef HELIOS_VISUALIZER
//        glm::mat4 depthMVP;
//    #endif //HELIOS_VISUALIZER

    //! Function to delete arrows denoting rig movement
    void deleteArrows();

    //! Function to update arrows for rig movement
    void updateArrows();

  public:
    //! Function to update spectra based on saved information
    void updateSpectra();

    //! Function to update cameras based on saved information
    void updateCameras();

    //! Function to "record", or save camera images with bounding boxes for each rig
    void record();

    //! Function to build context from XML
    void buildFromXML();

    //! Function to build context from XML
    /**
     * \param[in] xml_input_file Name of XML input file
    */
    void buildFromXML(std::string xml_path);

    //! Function to visualize XML plot
    void visualize();

    //! Function to visualize XML plot
    /**
     * \param[in] xml_input_file Name of XML input file
    */
    void buildAndVisualize(std::string xml_path);

    //! Function to set all values in GUI from XML
    void xmlSetValues();

    //! Function to set all values in GUI from XML
    /**
     * \param[in] xml_input_file Name of XML input file
    */
    void xmlSetValues(std::string xml_path);

    //! Function to get all values from current XML
    void xmlGetValues();

    //! Function to get all values from current XML
    /**
     * \param[in] xml_input_file Name of XML input file
    */
    void xmlGetValues(std::string xml_path);

    //! Function to get node labels for a given set of nodes
    /**
     * \param[in] label_name Name of the label
     * \param[in] node_name Name of the XML nodes to get labels from
     * \param[out] labels_vec Vector of labels of XML "parent" nodes
    */
    std::map<std::string, int> getNodeLabels(const std::string& label_name, const std::string& node_name,
                                               std::vector<std::string>& labels_vec);

    //! Function to get keypoints for every rig
    /**
     * \param[in] name Name of the label (e.g. name="keypoint")
     * \param[in] parent Name of the XML fields to get labels from (e.g. field="camera_position")
     * \param[out] keypoints Vector of keypoint (int) vectors
    */
    void getKeypoints(const std::string& name, const std::string& field, std::vector<std::vector<int>>& keypoints);

    //! Function to set keypoints for every rig
    /**
     * \param[in] name Name of the label (e.g. name="keypoint")
     * \param[in] parent Name of the XML fields to get labels from (e.g. field="camera_position")
     * \param[out] keypoints Vector of keypoint (int) vectors
    */
    void setKeypoints(const std::string& name, const std::string& field, std::vector<std::vector<int>>& keypoints);

    //! Function to get value of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value if one exists
    */
    void xmlGetValue(const std::string& name, const std::string& parent, int& default_value);

    //! Function to get value of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value if one exists
    */
    void xmlGetValue(const std::string& name, const std::string& parent, float& default_value);

    //! Function to get value of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value if one exists
    */
    void xmlGetValue(const std::string& name, const std::string& parent, std::string& default_value);

    //! Function to get value of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value if one exists
    */
    void xmlGetValue(const std::string& name, const std::string& parent, helios::vec2& default_value);

    //! Function to get value of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value if one exists
    */
    void xmlGetValue(const std::string& name, const std::string& parent, helios::vec3& default_value);

    //! Function to get value of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value if one exists
    */
    void xmlGetValue(const std::string& name, const std::string& parent, helios::int2& default_value);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<helios::vec2>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<helios::vec3>& default_vec);

     //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<helios::int2>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::string>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<float>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<int>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::vector<helios::vec3>>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::set<std::string>>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_set Set of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::set<std::string>& default_set);

    //! Function to set value of an XML field in the XML file
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value to set
    */
    void xmlSetValue(const std::string& name, const std::string& parent, int& default_value);

    //! Function to set value of an XML field in the XML file
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value to set
    */
    void xmlSetValue(const std::string& name, const std::string& parent, float& default_value);

    //! Function to set value of an XML field in the XML file
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value to set
    */
    void xmlSetValue(const std::string& name, const std::string& parent, std::string& default_value);

    //! Function to set value of an XML field in the XML file
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value to set
    */
    void xmlSetValue(const std::string& name, const std::string& parent, helios::vec2& default_value);

    //! Function to set value of an XML field in the XML file
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value to set
    */
    void xmlSetValue(const std::string& name, const std::string& parent, helios::vec3& default_value);

    //! Function to set value of an XML field in the XML file
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML node
     * \param[out] default_value Field value to set
    */
    void xmlSetValue(const std::string& name, const std::string& parent, helios::int2& default_value);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string& name, const std::string& parent, std::vector<helios::vec2>& default_vec);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string& name, const std::string& parent, std::vector<helios::vec3>& default_vec);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string& name, const std::string& parent, std::vector<helios::int2>& default_vec);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string& name, const std::string& parent, std::vector<std::string>&);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string& name, const std::string& parent, std::vector<int>&);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string& name, const std::string& parent, std::vector<float>&);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string&, const std::string&, std::vector<std::vector<helios::vec3>>&);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string&, const std::string&, std::vector<std::set<std::string>>&);

    //! Function to set values to an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes to set
    */
    void xmlSetValues(const std::string&, const std::string&, std::set<std::string>&);

    //! Function to set node labels for a given set of nodes
    /**
     * \param[in] label_name Name of the label
     * \param[in] node_name Name of the XML nodes to get labels from
     * \param[out] labels_vec Vector of labels of XML "parent" nodes to set
    */
    std::map<std::string, int> setNodeLabels(const std::string&, const std::string&, std::vector<std::string>&);

    //! Constructor
    ProjectBuilder(){
      primitive_types = {{"Ground", &ground_UUIDs}, {"Leaf", &leaf_UUIDs}, {"Petiolule", &petiolule_UUIDs},
                         {"Petiole", &petiole_UUIDs}, {"Internode", &internode_UUIDs}, {"Peduncle", &peduncle_UUIDs},
                         {"Petal", &petal_UUIDs}, {"Pedicel", &pedicel_UUIDs}, {"Fruit", &fruit_UUIDs}};
      primitive_continuous = {{"All", {false, false, false}}, {"Ground", {false, false, false}}, {"Leaf", {false, false, false}},
                              {"Petiolule", {false, false, false}}, {"Petiole", {false, false, false}},
                              {"Internode", {false, false, false}}, {"Peduncle", {false, false, false}},
                              {"Petal", {false, false, false}}, {"Pedicel", {false, false, false}},
                              {"Fruit", {false, false, false}}};
      bandlabels = {"red", "green", "blue"};
      for (std::string band : bandlabels){
       primitive_values[band] = {{"Ground", {&ground_reflectivity, &ground_transmissivity, &ground_emissivity}},
                                 {"Leaf", {&leaf_reflectivity, &leaf_transmissivity, &leaf_emissivity}},
                                 {"Petiolule", {&petiolule_reflectivity, &petiolule_transmissivity, &petiolule_emissivity}},
                                 {"Petiole", {&petiole_reflectivity, &petiole_transmissivity, &petiole_emissivity}},
                                 {"Internode", {&internode_reflectivity, &internode_transmissivity, &internode_emissivity}},
                                 {"Peduncle", {&peduncle_reflectivity, &peduncle_transmissivity, &peduncle_emissivity}},
                                 {"Petal", {&petal_reflectivity, &petal_transmissivity, &petal_emissivity}},
                                 {"Pedicel", {&pedicel_reflectivity, &pedicel_transmissivity, &pedicel_emissivity}},
                                 {"Fruit", {&fruit_reflectivity, &fruit_transmissivity, &fruit_emissivity}}};
      }
      primitive_spectra = {{"All", {&reflectivity_spectrum, &transmissivity_spectrum, &emissivity_spectrum}},
                             {"Ground", {&ground_reflectivity_spectrum, &ground_transmissivity_spectrum, &ground_emissivity_spectrum}},
                             {"Leaf", {&leaf_reflectivity_spectrum, &leaf_transmissivity_spectrum, &leaf_emissivity_spectrum}},
                             {"Petiolule", {&petiolule_reflectivity_spectrum, &petiolule_transmissivity_spectrum, &petiolule_emissivity_spectrum}},
                             {"Petiole", {&petiole_reflectivity_spectrum, &petiole_transmissivity_spectrum, &petiole_emissivity_spectrum}},
                             {"Internode", {&internode_reflectivity_spectrum, &internode_transmissivity_spectrum, &internode_emissivity_spectrum}},
                             {"Peduncle", {&peduncle_reflectivity_spectrum, &peduncle_transmissivity_spectrum, &peduncle_emissivity_spectrum}},
                             {"Petal", {&petal_reflectivity_spectrum, &petal_transmissivity_spectrum, &petal_emissivity_spectrum}},
                             {"Pedicel", {&pedicel_reflectivity_spectrum, &pedicel_transmissivity_spectrum, &pedicel_emissivity_spectrum}},
                             {"Fruit", {&fruit_reflectivity_spectrum, &fruit_transmissivity_spectrum, &fruit_emissivity_spectrum}}};
    }

    //! Destructor
    ~ProjectBuilder(){
      delete context;

      #ifdef HELIOS_VISUALIZER
          delete visualizer;
      #endif //HELIOS_VISUALIZER

      #ifdef PLANT_ARCHITECTURE
          delete plantarchitecture;
      #endif //PLANT_ARCHITECTURE

      #ifdef RADIATION_MODEL
          delete radiation;
          delete cameraproperties;
      #endif //RADIATION_MODEL

      #ifdef SOLARPOSITION
          delete solarposition;
      #endif //SOLARPOSITION

      #ifdef ENERGYBALANCEMODEL
          delete energybalancemodel;
      #endif //ENERGYBALANCEMODEL

      #ifdef BOUNDARYLAYERCONDUCTANCEMODEL
          delete boundarylayerconductance;
      #endif //BOUNDARYLAYERCONDUCTANCEMODEL
    }
};

#endif // PROJECT_BUILDER
