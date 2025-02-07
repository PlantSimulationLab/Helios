/** \file "ProjectBuilder.h" Visualizer header.

Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_PROJECTBUILDER_H
#define HELIOS_PROJECTBUILDER_H


#include "Context.h"
#include <pugixml.hpp>
#include "glew.h"
#include "BoundaryLayerConductanceModel.h"
#include "EnergyBalanceModel.h"
#include "PlantArchitecture.h"
#include "RadiationModel.h"
#include "SolarPosition.h"
#include "Visualizer.h"

#include "InitializeSimulation.h"
#include "BuildGeometry.h"
#include "InitializeRadiation.h"
#include "InitializeEnergyBalance.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "misc/cpp/imgui_stdlib.h"
#include "GLFW/glfw3.h"

#include <chrono>
#include <set>
#include <thread>

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

//! Function to open file dialog
std::string file_dialog();

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
#ifdef HELIOS_VISUALIZER
    Visualizer *visualizer = nullptr;
#endif

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
    std::vector<std::string> primitive_names = {"Ground", "Leaf", "Petiolule", "Petiole", "Internode", "Peduncle",
                                                "Petal", "Pedicel", "Fruit"};

    //! Map keyed by primitive names that returns a vector of UUIDs corresponding to the primitive name
    std::map<std::string, std::vector<uint>> primitive_types = {{"Ground", ground_UUIDs}, {"Leaf", leaf_UUIDs},
                                                                {"Petiolule", petiolule_UUIDs}, {"Petiole", petiole_UUIDs},
                                                                {"Internode", internode_UUIDs}, {"Peduncle", peduncle_UUIDs},
                                                                {"Petal", petal_UUIDs}, {"Pedicel", pedicel_UUIDs},
                                                                {"Fruit", fruit_UUIDs}};

    //! Primitive values map: band -> primitive type -> {reflectivity, transmissivity, emissivity}
    std::map<std::string, std::map<std::string, std::vector<float>>> primitive_values;

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

    //! XML library file
    std::string load_xml_library_file = "plugins/radiation/spectral_data/leaf_surface_spectral_library.xml";

    //! Solar direct spectrum
    std::string solar_direct_spectrum = "ASTMG173";

    //! Leaf reflectivity spectrum
    std::string leaf_reflectivity_spectrum = "grape_leaf_reflectivity_0000";

    //! Leaf transmissivity spectrum
    std::string leaf_transmissivity_spectrum = "grape_leaf_transmissivity_0000";

    //! Leaf emissivity
    float leaf_emissivity = 0.95;

    //! Ground reflectivity spectrum
    std::string ground_reflectivity_spectrum = "soil_reflectivity_0000";

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
    std::string current_primitive = "Leaf";

    //! Currently selected radiation band in the GUI
    std::string current_band = "red";

    //! Depth MVP matrix
    glm::mat4 depthMVP;

  public:
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
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::vector<helios::vec3>>& default_vec);

    //! Function to get values of an XML field
    /**
     * \param[in] name Name of the XML field
     * \param[in] parent Name of the parent XML nodes
     * \param[out] default_vec Vector of field values for all parent nodes
    */
    void xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::set<std::string>>& default_vec);

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

    //! Function to set node labels for a given set of nodes
    /**
     * \param[in] label_name Name of the label
     * \param[in] node_name Name of the XML nodes to get labels from
     * \param[out] labels_vec Vector of labels of XML "parent" nodes to set
    */
    std::map<std::string, int> setNodeLabels(const std::string&, const std::string&, std::vector<std::string>&);

    //! Destructor
    ~ProjectBuilder(){
      delete context;
      delete visualizer;
      delete plantarchitecture;
      delete radiation;
      delete solarposition;
      delete energybalancemodel;
      delete boundarylayerconductance;
      delete cameraproperties;
    }
};

#endif //HELIOS_PROJECTBUILDER_H
