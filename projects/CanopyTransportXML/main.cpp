//#include "RadiationModel.h"
//#include "EnergyBalanceModel.h"
//#include "BoundaryLayerConductanceModel.h"
//#include "StomatalConductanceModel.h"
//#include "PhotosynthesisModel.h"
//#include "SolarPosition.h"
#include "glew.h"
#include "PlantArchitecture.h"
#include "Visualizer.h"

#include "InitializeSimulation.h"
#include "BuildGeometry.h"
#include "InitializeRadiation.h"
#include "InitializeEnergyBalance.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "misc/cpp/imgui_stdlib.h"
#include "GLFW/glfw3.h"

#include <chrono>
#include <set>
#include <thread>
// #include "tinyfiledialogs.h"

using namespace helios;

void key_callback(GLFWwindow*, int, int, int, int);
std::map<std::string, int> get_node_labels(const std::string&, const std::string&, std::vector<std::string>&);
void get_xml_value(const std::string&, const std::string&, int&);
void get_xml_value(const std::string&, const std::string&, float&);
void get_xml_value(const std::string&, const std::string&, std::string&);
void get_xml_value(const std::string&, const std::string&, vec2&);
void get_xml_value(const std::string&, const std::string&, vec3&);
void get_xml_value(const std::string&, const std::string&, int2&);
void get_xml_values(const std::string&, const std::string&, std::vector<vec2>&);
void get_xml_values(const std::string&, const std::string&, std::vector<vec3>&);
void get_xml_values(const std::string&, const std::string&, std::vector<int2>&);
void get_xml_values(const std::string&, const std::string&, std::vector<std::string>&);
void get_xml_values(const std::string&, const std::string&, std::vector<float>&);
void get_xml_values(const std::string&, const std::string&, std::vector<std::vector<vec3>>&);
void get_xml_values(const std::string&, const std::string&, std::vector<std::set<std::string>>&);
std::string vec_to_string(const vec2&);
std::string vec_to_string(const vec3&);
std::string vec_to_string(const int2&);
std::map<std::string, int> set_node_labels(const std::string&, const std::string&, std::vector<std::string>&);
void set_xml_value(const std::string&, const std::string&, int&);
void set_xml_value(const std::string&, const std::string&, float&);
void set_xml_value(const std::string&, const std::string&, std::string&);
void set_xml_value(const std::string&, const std::string&, vec2&);
void set_xml_value(const std::string&, const std::string&, vec3&);
void set_xml_value(const std::string&, const std::string&, int2&);
void set_xml_values(const std::string&, const std::string&, std::vector<vec2>&);
void set_xml_values(const std::string&, const std::string&, std::vector<vec3>&);
void set_xml_values(const std::string&, const std::string&, std::vector<int2>&);
void set_xml_values(const std::string&, const std::string&, int2&);
void set_xml_values(const std::string&, const std::string&, std::vector<std::string>&);
void set_xml_values(const std::string&, const std::string&, std::vector<int>&);
void set_xml_values(const std::string&, const std::string&, std::vector<float>&);
void recalculate_values(Context&, float&, float&, float&, std::vector<std::string>&, std::vector<vec3>&,
                        std::vector<vec3>&, std::vector<std::string>&, std::vector<int2>&, std::vector<float>&,
                        std::vector<float>&, std::vector<float>&, std::vector<float>&);
std::vector<vec3> linspace(vec3, vec3, int);
// void OpenFileDialog();

pugi::xml_document xmldoc;
bool user_input;
float PAR_absorbed;
float NIR_absorbed;
float LW_absorbed;

std::string xml_input_file = "../inputs/inputs_2.xml";

int main(){

    xml_input_file = "../inputs/inputs_2.xml"; //\todo Will eventually make this passable from a command-line argument

    // std::vector<std::string> labels;
    // get_node_labels("label", "canopy_block", labels);

    Context context;

    PlantArchitecture plantarchitecture(&context);

    InitializeSimulation(xml_input_file, &context);

    BuildGeometry(xml_input_file, &plantarchitecture, &context);

    RadiationModel radiation(&context);
    SolarPosition solarposition(&context);

    InitializeRadiation(xml_input_file, &solarposition, &radiation, &context);

    EnergyBalanceModel energybalancemodel(&context);
    BLConductanceModel boundarylayerconductance(&context);

    InitializeEnergyBalance(xml_input_file, &boundarylayerconductance, &energybalancemodel, &context);

    // -- main time loop -- //

    float turbidity;
    assert( context.doesGlobalDataExist( "air_turbidity" ) );
    context.getGlobalData( "air_turbidity", turbidity );

    float diffuse_extinction_coeff;
    assert( context.doesGlobalDataExist( "diffuse_extinction_coeff" ) );
    context.getGlobalData( "diffuse_extinction_coeff", diffuse_extinction_coeff );

    uint sun_ID;
    assert( context.doesGlobalDataExist( "sun_ID" ) );
    context.getGlobalData( "sun_ID", sun_ID );

    std::vector<uint> ground_UUIDs, leaf_UUIDs, petiolule_UUIDs, petiole_UUIDs, internode_UUIDs, peduncle_UUIDs, petal_UUIDs, pedicel_UUIDs, fruit_UUIDs;
    context.getGlobalData( "ground_UUIDs", ground_UUIDs );
    assert( !ground_UUIDs.empty() );
    context.getGlobalData( "leaf_UUIDs", leaf_UUIDs );
    assert( !leaf_UUIDs.empty() );

    float ground_area = context.sumPrimitiveSurfaceArea( ground_UUIDs );

    std::vector<std::string> timeseries_variables = context.listTimeseriesVariables();

    if( timeseries_variables.empty() ){
        std::cout << "No timeseries data was loaded. Skipping time loop." << std::endl;
    }else{

        uint num_time_points = context.getTimeseriesLength( timeseries_variables.front().c_str() );
        for( uint timestep = 0; timestep<num_time_points; timestep++ ){

            context.setCurrentTimeseriesPoint( timeseries_variables.front().c_str(), timestep );

            std::cout << "Timestep " << timestep << ": " << context.getDate() << " " << context.getTime() << std::endl;

            float air_temperature = 300.f;
            if( context.doesTimeseriesVariableExist( "air_temperature" ) ){
                air_temperature = context.queryTimeseriesData( "air_temperature", timestep );
            }
            context.setPrimitiveData( context.getAllUUIDs(), "air_temperature", air_temperature );

            float air_humidity = 0.5f;
            if( context.doesTimeseriesVariableExist( "air_humidity" ) ){
                air_humidity = context.queryTimeseriesData( "air_humidity", timestep );
                if( air_humidity > 1 ){
                    //try dividing by 100
                    air_humidity /= 100.f;
                    if( air_humidity > 1 ){
                        std::cout << "WARNING: air_humidity must be between 0 and 1. Setting to default value of 0.5." << std::endl;
                        air_humidity = 0.5f;
                    }else{
                        std::cout << "WARNING: air_humidity read from timeseries was greater than 1.0. It was assumed that the given value was in percent and was automatically divided by 100." << std::endl;
                    }
                }
            }
            context.setPrimitiveData( context.getAllUUIDs(), "air_humidity", air_humidity );

            vec3 sun_dir_vec = solarposition.getSunDirectionVector();

            radiation.setSourcePosition( sun_ID, sun_dir_vec );

            if( diffuse_extinction_coeff > 0 ){
                radiation.setDiffuseRadiationExtinctionCoeff("PAR", diffuse_extinction_coeff, sun_dir_vec);
                radiation.setDiffuseRadiationExtinctionCoeff("NIR", diffuse_extinction_coeff, sun_dir_vec);
            }

            float R_PAR_dir = solarposition.getSolarFluxPAR(101000, air_temperature, air_humidity, turbidity);
            float R_NIR_dir = solarposition.getSolarFluxNIR(101000, air_temperature, air_humidity, turbidity);
            float fdiff = solarposition.getDiffuseFraction(101000, air_temperature, air_humidity, turbidity);

            radiation.setSourceFlux(sun_ID, "PAR", R_PAR_dir * (1.f - fdiff));
            radiation.setDiffuseRadiationFlux("PAR", R_PAR_dir * fdiff);
            radiation.setSourceFlux(sun_ID, "NIR", R_NIR_dir * (1.f - fdiff));
            radiation.setDiffuseRadiationFlux("NIR", R_NIR_dir * fdiff);


            // Run the radiation model
            radiation.runBand({"PAR","NIR","LW"});

            context.calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_PAR", PAR_absorbed );
            PAR_absorbed /= ground_area;

            context.calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_NIR", NIR_absorbed );
            NIR_absorbed /= ground_area;

            context.calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_LW", LW_absorbed );
            PAR_absorbed /= ground_area;

            std::cout << "Absorbed PAR: " << PAR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed NIR: " << NIR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed LW: " << LW_absorbed << " W/m^2" << std::endl;
        }
        // RIG BLOCK
        context.loadXML( "plugins/radiation/spectral_data/camera_spectral_library.xml", true);
        radiation.setSourceSpectrum( sun_ID, "solar_spectrum_ASTMG173");

        radiation.addRadiationBand("red");
        radiation.disableEmission("red");
        radiation.setSourceFlux(sun_ID, "red", 2.f);
        radiation.setScatteringDepth("red", 2);

        radiation.copyRadiationBand("red", "green");
        radiation.copyRadiationBand("red", "blue");

        radiation.enforcePeriodicBoundary("xy");

        std::vector<std::string> bandlabels = {"red", "green", "blue"};

        std::string xml_error_string;
        if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
            helios_runtime_error(xml_error_string);
        }

        std::vector<std::string> rig_labels;
        std::vector<vec3> camera_positions;
        std::vector<vec3> camera_lookats;
        std::vector<std::string> camera_labels;
        std::vector<int2> camera_resolutions;
        std::vector<float> focal_plane_distances;
        std::vector<float> lens_diameters;
        std::vector<float> FOV_aspect_ratios;
        std::vector<float> HFOVs;
        std::map<std::string, int> rig_dict = get_node_labels("label", "rig", rig_labels);
        get_xml_values("camera_position", "rig", camera_positions);
        get_xml_values("camera_lookat", "rig", camera_lookats);
        get_xml_values("camera_label", "rig", camera_labels);
        // get_xml_values("camera_resolution", "rig", camera_resolutions);
        // get_xml_values("focal_plane_distance", "rig", focal_plane_distances);
        // get_xml_values("lens_diameter", "rig", lens_diameters);
        // get_xml_values("FOV_aspect_ratio", "rig", FOV_aspect_ratios);
        // get_xml_values("HFOV", "rig", HFOVs);

        /*
        for (int i = 0; i < rig_labels.size(); i++)
        {
            CameraProperties cameraproperties;
            cameraproperties.camera_resolution = camera_resolutions[i];
            cameraproperties.focal_plane_distance = focal_plane_distances[i];
            cameraproperties.lens_diameter = lens_diameters[i];
            cameraproperties.FOV_aspect_ratio = FOV_aspect_ratios[i];
            cameraproperties.HFOV = HFOVs[i];

            radiation.addRadiationCamera(camera_labels[i], bandlabels, camera_positions[i], camera_lookats[i], cameraproperties, 100);

            context.loadXML( "plugins/radiation/spectral_data/camera_spectral_library.xml", true);
            radiation.setCameraSpectralResponse(camera_labels[i], "red", "calibrated_sun_NikonB500_spectral_response_red");
            radiation.setCameraSpectralResponse(camera_labels[i], "green","calibrated_sun_NikonB500_spectral_response_green");
            radiation.setCameraSpectralResponse(camera_labels[i], "blue", "calibrated_sun_NikonB500_spectral_response_blue");
            radiation.updateGeometry();
            radiation.runBand({"red", "green", "blue"});
        }
        */
        // RIG BLOCK END
    }
    // RIG BLOCK
    std::vector<std::string> rig_labels;
    vec3 camera_position(0,0,0); std::vector<vec3> camera_positions;
    // camera_positions = {camera_position};
    std::vector<std::vector<int>> keypoint_frames;
    std::vector<std::vector<vec3>> camera_position_vec;
    std::vector<std::vector<vec3>> camera_lookat_vec;
    vec3 camera_lookat(0,0,0); std::vector<vec3> camera_lookats;
    std::string camera_label = "RGB"; std::vector<std::string> camera_labels;
    std::map<std::string, int> rig_dict = get_node_labels("label", "rig", rig_labels);
    int num_images = 10;
    get_xml_value("camera_position", "rig", camera_position);
    get_xml_value("camera_lookat", "rig", camera_lookat);
    get_xml_value("camera_label", "rig", camera_label);
    get_xml_values("camera_position", "rig", camera_positions);
    get_xml_values("camera_position", "rig", camera_position_vec);
    get_xml_values("camera_lookat", "rig", camera_lookats);
    get_xml_values("camera_lookat", "rig", camera_lookat_vec);
    get_xml_values("camera_label", "rig", camera_labels);
    for (int i = 0; i < camera_position_vec.size(); i++){
        std::vector<int> curr_vec{};
        for (int j = 0; j < camera_position_vec[i].size(); j++){
            curr_vec.push_back(j);
        }
        keypoint_frames.push_back( curr_vec );
    }
    // CAMERA BLOCK
    std::vector<std::string> camera_names;
    int2 camera_resolution(1024, 1024); std::vector<int2> camera_resolutions;
    float focal_plane_distance = 0.4; std::vector<float> focal_plane_distances;
    float lens_diameter = 0.02; std::vector<float> lens_diameters;
    float FOV_aspect_ratio = 1.4; std::vector<float> FOV_aspect_ratios;
    float HFOV = 50.0; std::vector<float> HFOVs;
    std::map<std::string, int> camera_dict = get_node_labels("label", "camera", camera_names);
    get_xml_value("camera_resolution", "camera", camera_resolution);
    get_xml_value("focal_plane_distance", "camera", focal_plane_distance);
    get_xml_value("lens_diameter", "camera", lens_diameter);
    get_xml_value("FOV_aspect_ratio", "camera", FOV_aspect_ratio);
    get_xml_value("HFOV", "camera", HFOV);
    get_xml_values("camera_resolution", "camera", camera_resolutions);
    get_xml_values("focal_plane_distance", "camera", focal_plane_distances);
    get_xml_values("lens_diameter", "camera", lens_diameters);
    get_xml_values("FOV_aspect_ratio", "camera", FOV_aspect_ratios);
    get_xml_values("HFOV", "camera", HFOVs);


    std::vector<std::set<std::string>> rig_camera_labels;
    get_xml_values("camera_label", "rig", rig_camera_labels);

    std::vector<std::vector<uint>> arrow_vec;
    Visualizer visualizer(800);

    std::string current_rig;
    for (int n = 0; n < rig_labels.size(); n++){
        current_rig = rig_labels[n];
        vec3 camera_lookat_ = camera_lookats[rig_dict[(std::string) current_rig]];
        vec3 camera_position_ = camera_positions[rig_dict[(std::string) current_rig]];
        vec3 scale(1,1,1);
        vec3 initial(0, 1, 0);
        SphericalCoord rotation = cart2sphere(camera_lookat_ - camera_position_);
        // SphericalCoord rotation = cart2sphere(camera_position_ - camera_lookat_);
        // SphericalCoord rotation = cart2sphere(camera_lookat_ - camera_position_);
        // SphericalCoord rotation(0,0,0);
        RGBcolor color(255,0,0);
        // context.loadOBJ("../../../plugins/radiation/camera_light_models/Camera.obj", camera_position_,
        //                 scale, rotation, color, "ZUP", true);
        for (int i = 1; i < camera_position_vec[rig_dict[(std::string) current_rig]].size(); i++){
            vec3 arrow_pos = camera_position_vec[rig_dict[(std::string) current_rig]][0];
            vec3 arrow_direction_vec = arrow_pos - camera_position_vec[rig_dict[(std::string) current_rig]][1];
            SphericalCoord arrow_direction_sph = cart2sphere(arrow_direction_vec);
            vec3 arrow_scale(0.35, 0.35, 0.35);
            std::vector<uint> arrow = context.loadOBJ("../../../plugins/radiation/camera_light_models/Arrow.obj",
                                                    nullorigin, arrow_scale, nullrotation, RGB::blue, "YUP", true);
            context.rotatePrimitive(arrow, arrow_direction_sph.elevation, "x");
            context.rotatePrimitive(arrow, -arrow_direction_sph.azimuth, "z");
            context.translatePrimitive(arrow, arrow_pos);
            context.setPrimitiveData(arrow, "twosided_flag", uint(3));
            arrow_vec.push_back(arrow);
        }
    }
    std::vector<std::string> bandlabels = {"red", "green", "blue"};
    for (std::string rig_label : rig_labels){
        CameraProperties cameraproperties;
        int rig_index = rig_dict[rig_label];
        int camera_index = camera_dict[camera_labels[rig_index]];
        cameraproperties.camera_resolution = camera_resolutions[camera_index];
        cameraproperties.focal_plane_distance = focal_plane_distances[camera_index];
        cameraproperties.lens_diameter = lens_diameters[camera_index];
        cameraproperties.FOV_aspect_ratio = FOV_aspect_ratios[camera_index];
        cameraproperties.HFOV = HFOVs[camera_index];
        std::string camera_label_ = rig_label + "_" + camera_labels[rig_index];
        vec3 camera_position_ = camera_positions[rig_index];
        vec3 camera_lookat_ = camera_lookats[rig_index];
        radiation.addRadiationCamera(camera_label_, bandlabels, camera_position_, camera_lookat_, cameraproperties, 100);
        radiation.setCameraSpectralResponse(camera_label_, "red", "calibrated_sun_NikonB500_spectral_response_red");
        radiation.setCameraSpectralResponse(camera_label_, "green","calibrated_sun_NikonB500_spectral_response_green");
        radiation.setCameraSpectralResponse(camera_label_, "blue", "calibrated_sun_NikonB500_spectral_response_blue");
        radiation.updateGeometry();
    }

    radiation.enableCameraModelVisualization();
    visualizer.buildContextGeometry(&context);
    // visualizer.colorContextPrimitivesByData("radiation_flux_PAR");

    // Uncomment below for interactive
    // visualizer.plotInteractive();

    visualizer.addCoordinateAxes();
    visualizer.plotUpdate();

    // visualizer.openWindow();

    std::string xml_error_string;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }

    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;

    // ####### DEFAULT VALUES ####### //
    // MAIN BLOCK
    float latitude = 38.55;
    float longitude = 121.76;
    int UTC_offset = 8;
    std::string csv_weather_file = "../inputs/weather_data.csv";
    vec3 domain_origin(0,0,0);
    vec2 domain_extent(10,10);
    int2 ground_resolution(1,1);
    std::string ground_texture_file = "plugins/visualizer/textures/dirt.jpg";
    get_xml_value("latitude", "helios", latitude);
    get_xml_value("longitude", "helios", longitude);
    get_xml_value("UTC_offset", "helios", UTC_offset);
    get_xml_value("csv_weather_file", "helios", csv_weather_file);
    get_xml_value("domain_origin", "helios", domain_origin);
    get_xml_value("domain_extent", "helios", domain_extent);
    get_xml_value("ground_resolution", "helios", ground_resolution);
    get_xml_value("ground_texture_file", "helios", ground_texture_file);
    // CANOPY BLOCK
    std::vector<std::string> labels;
    vec3 canopy_origin(0,0,0); std::vector<vec3> canopy_origins;
    int2 plant_count(1,1); std::vector<int2> plant_counts;
    vec2 plant_spacing(0.5,0.5); std::vector<vec2> plant_spacings;
    std::string plant_library_name = "cowpea"; std::vector<std::string> plant_library_names;
    float plant_age = 0; std::vector<float> plant_ages;
    float ground_clipping_height = 0; std::vector<float> ground_clipping_heights;
    std::map<std::string, int> canopy_labels = get_node_labels("label", "canopy_block", labels);
    get_xml_value("canopy_origin", "canopy_block", canopy_origin);
    get_xml_value("plant_count", "canopy_block", plant_count);
    get_xml_value("plant_spacing", "canopy_block", plant_spacing);
    get_xml_value("plant_library_name", "canopy_block", plant_library_name);
    get_xml_value("plant_age", "canopy_block", plant_age);
    get_xml_value("ground_clipping_height", "canopy_block", ground_clipping_height);
    get_xml_values("canopy_origin", "canopy_block", canopy_origins);
    get_xml_values("plant_count", "canopy_block", plant_counts);
    get_xml_values("plant_spacing", "canopy_block", plant_spacings);
    get_xml_values("plant_library_name", "canopy_block", plant_library_names);
    get_xml_values("plant_age", "canopy_block", plant_ages);
    get_xml_values("ground_clipping_height", "canopy_block", ground_clipping_heights);
    // RADIATION BLOCK
    int direct_ray_count = 100;
    int diffuse_ray_count = 1000;
    diffuse_extinction_coeff = 0.1;
    int scattering_depth = 2;
    float air_turbidity = 0.05;
    std::string load_xml_library_file = "plugins/radiation/spectral_data/leaf_surface_spectral_library.xml";
    std::string solar_direct_spectrum = "ASTMG173";
    std::string leaf_reflectivity_spectrum = "grape_leaf_reflectivity_0000";
    std::string leaf_transmissivity_spectrum = "grape_leaf_transmissivity_0000";
    float leaf_emissivity = 0.95;
    std::string ground_reflectivity_spectrum = "soil_reflectivity_0000";
    get_xml_value("direct_ray_count", "radiation", direct_ray_count);
    get_xml_value("diffuse_ray_count", "radiation", diffuse_ray_count);
    get_xml_value("diffuse_extinction_coeff", "radiation", diffuse_extinction_coeff);
    get_xml_value("scattering_depth", "radiation", scattering_depth);
    get_xml_value("air_turbidity", "radiation", air_turbidity);
    get_xml_value("load_xml_library_file", "radiation", load_xml_library_file);
    get_xml_value("solar_direct_spectrum", "radiation", solar_direct_spectrum);
    get_xml_value("leaf_reflectivity_spectrum", "radiation", leaf_reflectivity_spectrum);
    get_xml_value("leaf_transmissivity_spectrum", "radiation", leaf_transmissivity_spectrum);
    get_xml_value("leaf_emissivity", "radiation", leaf_emissivity);
    get_xml_value("ground_reflectivity_spectrum", "radiation", ground_reflectivity_spectrum);
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // void* window;

    // glfwShowWindow((GLFWwindow *) window);

    // GLFWwindow* window = glfwCreateWindow(640, 480, "My Title", NULL, NULL);
    GLFWwindow* window = (GLFWwindow *)visualizer.getWindow();

    glfwShowWindow(window);

    bool show_demo_window = false;
    bool my_tool_active = true;
    // bool user_input = false;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();
    // glfwSwapInterval(1); // enable V-Sync

    ImVec2 current_position;
    ImVec2 last_position;
    bool currently_collapsed;
    bool previously_collapsed = false;
    std::string current_tab = "General";
    std::string previous_tab = "General";

    // (Your code calls glfwPollEvents())
    // ...
    // Start the Dear ImGui frame

    std::string visualization_type = "RGB";
    bool switch_visualization = false;

    std::string current_cam_position = "0";

    while ( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose((GLFWwindow*)window) == 0 ) {
        // Poll and handle events
        // glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // std::vector<uint> frameBufferSize = visualizer.getFrameBufferSize();
        // glViewport(0,0,frameBufferSize[0],frameBufferSize[1]);
        //
        // helios::RGBcolor backgroundColor = visualizer.getBackgroundColor();
        // glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 0.0f);
        //
        // Shader primaryShader = visualizer.getPrimaryShader();
        // primaryShader.useShader();
        //
        // std::vector<helios::vec3> cameraPosition = visualizer.getCameraPosition();
        // helios::vec3 camera_lookat_center = cameraPosition[0];
        // helios::vec3 camera_eye_location = cameraPosition[1];
        // visualizer.updatePerspectiveTransformation( camera_lookat_center, camera_eye_location );
        //
        // glm::mat4 biasMatrix(
        //         0.5, 0.0, 0.0, 0.0,
        //         0.0, 0.5, 0.0, 0.0,
        //         0.0, 0.0, 0.5, 0.0,
        //         0.5, 0.5, 0.5, 1.0
        // );
        //
        // glm::mat4 depthMVP = glm::mat4(1.0);
        //
        // glm::mat4 DepthBiasMVP = biasMatrix*depthMVP;
        //
        // primaryShader.setDepthBiasMatrix( DepthBiasMVP );
        //
        // glm::mat4 perspectiveTransformationMatrix = visualizer.getPerspectiveTransformationMatrix();
        // primaryShader.setTransformationMatrix( perspectiveTransformationMatrix );
        //
        // primaryShader.enableTextureMaps();
        // primaryShader.enableTextureMasks();
        //
        // std::vector<Visualizer::LightingModel> primaryLightingModel = visualizer.getPrimaryLightingModel();
        // primaryShader.setLightingModel( primaryLightingModel.at(0) );
        //
        // uint depthTexture = visualizer.getDepthTexture();
        // glBindTexture(GL_TEXTURE_2D, depthTexture);
        // glUniform1i(primaryShader.shadowmapUniform,1);
        //
        // visualizer.render( 0 );
        //
        // glfwPollEvents();
        // visualizer.getViewKeystrokes( camera_eye_location, camera_lookat_center );
        //
        //
        // glfwSwapBuffers((GLFWwindow*)window);
        //
        // glfwWaitEvents();
        //
        // int width, height;
        // glfwGetFramebufferSize(window, &width, &height );
        // // Wframebuffer = width;
        // // Hframebuffer = height;
        // visualizer.setFrameBufferSize(width, height);
        //
        // glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );


        // glBegin(GL_POINTS);
        // glVertex3f((GLfloat) canopy_origin.x, (GLfloat) canopy_origin.y, (GLfloat) canopy_origin.z);
        // glEnd();

        // glm::mat4 perspectiveTransformationMatrix = visualizer.getPerspectiveTransformationMatrix();
        // glm::vec4 canopy_origin_position = glm::vec4(camera_position.x, camera_position.y, camera_position.z, 1.0);
        // canopy_origin_position = perspectiveTransformationMatrix * canopy_origin_position;

        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        // ImGui::ShowDemoWindow(&show_demo_window); // Show demo window! :)

        // glfwSetKeyCallback(window, key_callback);

        // Check for key press or mouse movement
        // if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
        //     glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        //     user_input = true;
        // }

        if (ImGui::IsKeyDown(ImGuiKey_Q)) {
            user_input = true;
        }

        if (user_input)
            visualizer.plotUpdate();

        user_input = false;
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;

        // ImGui::SetNextWindowSize(ImVec2(500, 400));
        ImVec2 windowSize = ImGui::GetWindowSize();

        // TEST
        glm::mat4 perspectiveTransformationMatrix = visualizer.getPerspectiveTransformationMatrix();
        glm::vec4 origin_position;
        std::string current_label;
        glm::mat4 depthMVP = visualizer.getDepthMVP();
        for (int n = 0; n < labels.size(); n++){
            current_label = labels[n];
            vec3 canopy_origin_ = canopy_origins[canopy_labels[(std::string) current_label]];
            origin_position = glm::vec4(canopy_origin_.x, canopy_origin_.y, canopy_origin_.z + 0.5, 1.0);
            origin_position = perspectiveTransformationMatrix * origin_position;
            ImGui::SetNextWindowPos(ImVec2(windowSize.x + (origin_position.x / origin_position.w) * windowSize.x,
                                            windowSize.y - (origin_position.y / origin_position.w) * windowSize.y), ImGuiCond_Always);
            // double check above
            ImGui::Begin(current_label.c_str(), &my_tool_active);
            ImGui::End();
        }
        for (int n = 0; n < rig_labels.size(); n++){
            current_label = rig_labels[n];
            vec3 camera_position_ = camera_positions[rig_dict[(std::string) current_label]];
            origin_position = glm::vec4(camera_position_.x, camera_position_.y, camera_position_.z + 0.5, 1.0);
            origin_position = perspectiveTransformationMatrix * origin_position;
            ImGui::SetNextWindowPos(ImVec2(windowSize.x + (origin_position.x / origin_position.w) * windowSize.x,
                                            windowSize.y - (origin_position.y / origin_position.w) * windowSize.y), ImGuiCond_Always);
            ImGui::Begin(current_label.c_str(), &my_tool_active);
            ImGui::End();
            // vec3 scale(1,1,1);
            // SphericalCoord rotation(0,0,0);
            // RGBcolor color(1,0,0);
            // context.loadOBJ("../../../plugins/radiation/camera_light_models/Camera.obj", camera_position_, scale, rotation, color, "ZUP", true);
            // visualizer.buildContextGeometry(&context);
        }
        //

        // ImGui::Begin("Editor", &my_tool_active, ImGuiWindowFlags_MenuBar);  // Begin a new window
        ImGui::Begin("Editor", &my_tool_active, window_flags);  // Begin a new window
        ImGui::SetNextWindowPos(ImVec2(windowSize.x - 100.0f, 0), ImGuiCond_Always); // flag -> can't move window with mouse
        current_position = ImGui::GetWindowPos();
        currently_collapsed = ImGui::IsWindowCollapsed();

        if (current_tab != previous_tab || current_position.x != last_position.x || current_position.y != last_position.y || currently_collapsed != previously_collapsed) {
            user_input = true;
            previous_tab = current_tab;
        }
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
                if (ImGui::MenuItem("Save", "Ctrl+S")){
                    // MAIN BLOCK
                    set_xml_value("latitude", "helios", latitude);
                    set_xml_value("longitude", "helios", longitude);
                    set_xml_value("UTC_offset", "helios", UTC_offset);
                    set_xml_value("csv_weather_file", "helios", csv_weather_file);
                    set_xml_value("domain_origin", "helios", domain_origin);
                    set_xml_value("domain_extent", "helios", domain_extent);
                    set_xml_value("ground_resolution", "helios", ground_resolution);
                    set_xml_value("ground_texture_file", "helios", ground_texture_file);
                    // Canopy Block
                    canopy_labels = set_node_labels("label", "canopy_block", labels);
                    set_xml_values("canopy_origin", "canopy_block", canopy_origins);
                    set_xml_values("plant_count", "canopy_block", plant_counts);
                    set_xml_values("plant_spacing", "canopy_block", plant_spacings);
                    set_xml_values("plant_library_name", "canopy_block", plant_library_names);
                    set_xml_values("plant_age", "canopy_block", plant_ages);
                    set_xml_values("ground_clipping_height", "canopy_block", ground_clipping_heights);
                    // Radiation Block
                    set_xml_value("diffuse_ray_count", "radiation", diffuse_ray_count);
                    set_xml_value("direct_ray_count", "radiation", direct_ray_count);
                    set_xml_value("diffuse_extinction_coeff", "radiation", diffuse_extinction_coeff);
                    set_xml_value("scattering_depth", "radiation", scattering_depth);
                    set_xml_value("air_turbidity", "radiation", air_turbidity);
                    set_xml_value("load_xml_library_file", "radiation", load_xml_library_file);
                    set_xml_value("solar_direct_spectrum", "radiation", solar_direct_spectrum);
                    set_xml_value("leaf_reflectivity_spectrum", "radiation", leaf_reflectivity_spectrum);
                    set_xml_value("leaf_transmissivity_spectrum", "radiation", leaf_transmissivity_spectrum);
                    set_xml_value("leaf_emissivity", "radiation", leaf_emissivity);
                    set_xml_value("ground_reflectivity_spectrum", "radiation", ground_reflectivity_spectrum);
                    // RIG BLOCK
                    rig_dict = set_node_labels("label", "rig", rig_labels);
                    set_xml_values("camera_position", "rig", camera_positions);
                    set_xml_values("camera_lookat", "rig", camera_lookats);
                    set_xml_values("camera_label", "rig", camera_labels);
                    set_xml_values("camera_resolution", "rig", camera_resolutions);
                    set_xml_values("focal_plane_distances", "rig", focal_plane_distances);
                    set_xml_values("lens_diameter", "rig", lens_diameters);
                    set_xml_values("FOV_aspect_ratio", "rig", FOV_aspect_ratios);
                    set_xml_values("HFOV", "rig", HFOVs);
                    // BuildGeometry(xml_input_file, &plantarchitecture, &context);
                    // xmldoc.save_file("../inputs/inputs.xml");
                    xmldoc.save_file(xml_input_file.c_str());
                }
                if (ImGui::MenuItem("Close", "Ctrl+W"))  { my_tool_active = false; }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Visualization"))
            {
                if (ImGui::MenuItem("RGB (Default)") && visualization_type != "RGB")
                {
                    visualization_type = "RGB";
                    switch_visualization = true;
                }
                if (ImGui::MenuItem("PAR") && visualization_type != "radiation_flux_PAR")
                {
                    visualization_type = "radiation_flux_PAR";
                    switch_visualization = true;
                }
                if (ImGui::MenuItem("NIR") && visualization_type != "radiation_flux_NIR")  {
                    visualization_type = "radiation_flux_NIR";
                    switch_visualization = true;
                }
                if (ImGui::MenuItem("LW") && visualization_type != "radiation_flux_LW")  {
                    visualization_type = "radiation_flux_LW";
                    switch_visualization = true;
                }
                if (switch_visualization)
                {
                    visualizer.clearGeometry();
                    if (visualization_type != "RGB") {
                        visualizer.colorContextPrimitivesByData(visualization_type.c_str());
                    } else{
                        visualizer.clearColor();
                    }
                    visualizer.buildContextGeometry(&context);
                    visualizer.plotUpdate();
                    switch_visualization = false;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
        if (ImGui::Button("Reload")) {
            // domain_extent = vec2( domain_extent.x, domain_extent.y );
            // node = helios.child("domain_extent");
            // node.text().set(vec_to_string(domain_extent).c_str());

            // MAIN BLOCK
            set_xml_value("latitude", "helios", latitude);
            set_xml_value("longitude", "helios", longitude);
            set_xml_value("UTC_offset", "helios", UTC_offset);
            set_xml_value("csv_weather_file", "helios", csv_weather_file);
            set_xml_value("domain_origin", "helios", domain_origin);
            set_xml_value("domain_extent", "helios", domain_extent);
            set_xml_value("ground_resolution", "helios", ground_resolution);
            set_xml_value("ground_texture_file", "helios", ground_texture_file);
            // Canopy Block
            canopy_labels = set_node_labels("label", "canopy_block", labels);
            set_xml_values("canopy_origin", "canopy_block", canopy_origins);
            set_xml_values("plant_count", "canopy_block", plant_counts);
            set_xml_values("plant_spacing", "canopy_block", plant_spacings);
            set_xml_values("plant_library_name", "canopy_block", plant_library_names);
            set_xml_values("plant_age", "canopy_block", plant_ages);
            set_xml_values("ground_clipping_height", "canopy_block", ground_clipping_heights);
            // Radiation Block
            set_xml_value("diffuse_ray_count", "radiation", diffuse_ray_count);
            set_xml_value("direct_ray_count", "radiation", direct_ray_count);
            set_xml_value("diffuse_extinction_coeff", "radiation", diffuse_extinction_coeff);
            set_xml_value("scattering_depth", "radiation", scattering_depth);
            set_xml_value("air_turbidity", "radiation", air_turbidity);
            set_xml_value("load_xml_library_file", "radiation", load_xml_library_file);
            set_xml_value("solar_direct_spectrum", "radiation", solar_direct_spectrum);
            set_xml_value("leaf_reflectivity_spectrum", "radiation", leaf_reflectivity_spectrum);
            set_xml_value("leaf_transmissivity_spectrum", "radiation", leaf_transmissivity_spectrum);
            set_xml_value("leaf_emissivity", "radiation", leaf_emissivity);
            set_xml_value("ground_reflectivity_spectrum", "radiation", ground_reflectivity_spectrum);
            // RIG BLOCK
            rig_dict = set_node_labels("label", "rig", rig_labels);
            set_xml_values("camera_position", "rig", camera_positions);
            set_xml_values("camera_lookat", "rig", camera_lookats);
            set_xml_values("camera_label", "rig", camera_labels);
            set_xml_values("camera_resolution", "rig", camera_resolutions);
            set_xml_values("focal_plane_distances", "rig", focal_plane_distances);
            set_xml_values("lens_diameter", "rig", lens_diameters);
            set_xml_values("FOV_aspect_ratio", "rig", FOV_aspect_ratios);
            set_xml_values("HFOV", "rig", HFOVs);
            // BuildGeometry(xml_input_file, &plantarchitecture, &context);
            // xmldoc.save_file("../inputs/inputs.xml");
            xmldoc.save_file(xml_input_file.c_str());
            context.~Context(); // clear Geometry
            Context context;
            visualizer.clearGeometry();
            // xml_input_file = "../inputs/inputs_2.xml";
            recalculate_values(context, PAR_absorbed, NIR_absorbed, LW_absorbed, rig_labels, camera_positions,
                                camera_lookats, camera_labels, camera_resolutions, focal_plane_distances,
                                lens_diameters, FOV_aspect_ratios, HFOVs);
            // visualizer.clearGeometry();
            // visualizer.colorContextPrimitivesByData("radiation_flux_PAR");
            visualizer.buildContextGeometry(&context);
            visualizer.plotUpdate();
        }
        ImGui::SameLine();
        std::string image_dir = "./saved/";
        bool dir = std::filesystem::create_directories(image_dir);
        if (!dir && !std::filesystem::exists(image_dir)){
            helios_runtime_error("Error: image output directory " + image_dir + " could not be created. Exiting...");
        }
        // context.setPrimitiveData(UUIDs_bunny, "bunny", uint(0));
        // plant segmentation bounding boxes
        // plant ID bounding boxes (plant architecture->optional plant output data)
        if (ImGui::Button("Record")){
            for (int i = 0; i < rig_labels.size(); i++){
                CameraProperties cameraproperties;
                int rig_index = rig_dict[rig_labels[i]];
                int camera_index = camera_dict[camera_labels[rig_index]];
                cameraproperties.camera_resolution = camera_resolutions[camera_index];
                cameraproperties.focal_plane_distance = focal_plane_distances[camera_index];
                cameraproperties.lens_diameter = lens_diameters[camera_index];
                cameraproperties.FOV_aspect_ratio = FOV_aspect_ratios[camera_index];
                cameraproperties.HFOV = HFOVs[camera_index];
                std::vector<vec3> interpolated_camera_positions;
                std::vector<vec3> interpolated_camera_lookats;
                for (int k = 0; k < camera_position_vec[camera_index].size() - 1; k++){
                    interpolated_camera_positions = linspace(camera_position_vec[camera_index][k], camera_position_vec[camera_index][k + 1], num_images);
                    interpolated_camera_lookats = linspace(camera_lookat_vec[camera_index][k], camera_lookat_vec[camera_index][k + 1], num_images);
                    for (int j = 0; j < interpolated_camera_positions.size(); j++){
                        std::string cameralabel = camera_labels[rig_index] + "_" + std::to_string(j);
                        radiation.addRadiationCamera(cameralabel, bandlabels, interpolated_camera_positions[j],
                            interpolated_camera_lookats[j], cameraproperties, 100);
                        radiation.setCameraSpectralResponse(cameralabel, "red", "calibrated_sun_NikonB500_spectral_response_red");
                        radiation.setCameraSpectralResponse(cameralabel, "green","calibrated_sun_NikonB500_spectral_response_green");
                        radiation.setCameraSpectralResponse(cameralabel, "blue", "calibrated_sun_NikonB500_spectral_response_blue");
                        radiation.updateGeometry();
                        /* Everything above move outside of the loop. Call when creating camera. */
                        // radiation.setCameraPosition();
                        // radiation.setCameraLookat();
                        radiation.runBand({"red", "green", "blue"});
                        // radiation.writeCameraImage( cameralabel, bandlabels, "RGB", image_dir);
                        radiation.writeNormCameraImage( cameralabel, bandlabels, "RGB", image_dir);
                        radiation.writeDepthImageData( cameralabel, "depth", image_dir);
                        radiation.writeNormDepthImage( cameralabel, "normdepth", 3, image_dir);
                        radiation.writeImageBoundingBoxes( cameralabel, "bunny", 0, "bbox", image_dir);
                    }
                }
            }
            // for (std::string cameralabel : camera_labels){
            //     radiation.writeCameraImage( cameralabel, bandlabels, "RGB", image_dir);
            //     radiation.writeDepthImageData( cameralabel, "depth", image_dir);
            //     radiation.writeNormDepthImage( cameralabel, "normdepth", 3, image_dir);
            //     radiation.writeImageBoundingBoxes( cameralabel, "bunny", 0, "bbox", image_dir);
            // }
        }
        // ####### RESULTS ####### //
        ImGui::Text("Absorbed PAR: %f W/m^2", PAR_absorbed);
        ImGui::Text("Absorbed NIR: %f W/m^2", NIR_absorbed);
        ImGui::Text("Absorbed  LW: %f W/m^2", LW_absorbed);
        if (ImGui::BeginTabBar("Settings#left_tabs_bar")){
            if (ImGui::BeginTabItem("General")){
                current_tab = "General";
                // ####### LATITUDE ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Latitude", &latitude);
                // ####### LONGITUDE ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Longitude", &longitude);
                // ####### UTC OFFSET ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputInt("UTC Offset", &UTC_offset);
                // ####### CSV Weather File ####### //
                ImGui::SetNextItemWidth(60);
                if (ImGui::Button("CSV Weather File")){
                    // OpenFileDialog();
                }
                ImGui::SameLine();
                ImGui::SameLine();
                std::string shorten_weather_file = csv_weather_file;
                size_t last_weather_file = csv_weather_file.rfind('/');
                if (last_weather_file != std::string::npos){
                    shorten_weather_file = csv_weather_file.substr(last_weather_file + 1);
                }
                ImGui::Text(shorten_weather_file.c_str());
                // ####### DOMAIN ORIGIN ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##domain_origin_x", &domain_origin.x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##domain_origin_y", &domain_origin.y);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##domain_origin_z", &domain_origin.z);
                ImGui::SameLine();
                ImGui::Text("Domain Origin");
                // ####### DOMAIN EXTENT ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("##domain_extent_x", &domain_extent.x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("##domain_extent_y", &domain_extent.y);
                ImGui::SameLine();
                ImGui::Text("Domain Extent");
                // ####### GROUND RESOLUTION ####### //
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("##ground_resolution_x", &ground_resolution.x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("##ground_resolution_y", &ground_resolution.y);
                ImGui::SameLine();
                ImGui::Text("Ground Resolution");
                // ####### GROUND TEXTURE File ####### //
                ImGui::SetNextItemWidth(60);
                if (ImGui::Button("Ground Texture File")){
                    // OpenFileDialog();
                }
                ImGui::SameLine();
                std::string shorten = ground_texture_file;
                size_t last = ground_texture_file.rfind('/');
                if (last != std::string::npos){
                    shorten = ground_texture_file.substr(last + 1);
                }
                ImGui::Text(shorten.c_str());

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Canopy")){
                current_tab = "Canopy";
                // ####### CANOPY ORIGIN ####### //
                static const char* current_item = "canopy_0";
                if (ImGui::BeginCombo("##combo", current_item)) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < labels.size(); n++)
                    {
                        bool is_selected = (current_item == labels[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(labels[n].c_str(), is_selected))
                            current_item = labels[n].c_str();
                        if (is_selected)
                        ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                }
                    ImGui::EndCombo();
                }
                ImGui::SetNextItemWidth(100);
                ImGui::InputText("##canopy_name", &labels[canopy_labels[(std::string) current_item]]);
                ImGui::SameLine();
                if (ImGui::Button("Add Canopy")){
                    std::string default_canopy_label = "canopy";
                    std::string new_canopy_label = "canopy_0";
                    int count = 0;
                    while (canopy_labels.find(new_canopy_label) != canopy_labels.end()){
                        count++;
                        new_canopy_label = default_canopy_label + "_" + std::to_string(count);
                    }
                    canopy_labels.insert({new_canopy_label, labels.size()});
                    canopy_origins.push_back(canopy_origin);
                    plant_counts.push_back(plant_count);
                    plant_spacings.push_back(plant_spacing);
                    plant_library_names.push_back(plant_library_name);
                    plant_ages.push_back(plant_age);
                    ground_clipping_heights.push_back(ground_clipping_height);
                    labels.push_back(new_canopy_label);
                    current_item = new_canopy_label.c_str();
                    std::string parent = "canopy_block";
                    pugi::xml_node canopy_block = helios.child(parent.c_str());
                    pugi::xml_node new_canopy_node = helios.append_copy(canopy_block);
                    // pugi::xml_node new_canopy_node = helios.append_child(pugi::node_pcdata);
                    // pugi::xml_node node;
                    std::string name = "canopy_origin";
                    // node = new_canopy_node.child(name.c_str());
                    // node.text().set(vec_to_string(canopy_origin).c_str());
                    // name = "plant_count";
                    // node = new_canopy_node.child(name.c_str());
                    // node.text().set(vec_to_string(plant_count).c_str());
                    // name = "plant_spacing";
                    // node = new_canopy_node.child(name.c_str());
                    // node.text().set(vec_to_string(plant_spacing).c_str());
                    // name = "plant_library_name";
                    // node = new_canopy_node.child(name.c_str());
                    // node.text().set(plant_library_name.c_str());
                    // name = "plant_age";
                    // node = new_canopy_node.child(name.c_str());
                    // node.text().set(std::to_string(plant_age).c_str());
                    // name = "ground_clipping_height";
                    // node = new_canopy_node.child(name.c_str());
                    // node.text().set(std::to_string(ground_clipping_height).c_str());
                    name = "label";
                    pugi::xml_attribute node_label = new_canopy_node.attribute(name.c_str());
                    node_label.set_value(new_canopy_label.c_str());
                }
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##canopy_origin_x", &canopy_origins[canopy_labels[(std::string) current_item]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##canopy_origin_y", &canopy_origins[canopy_labels[(std::string) current_item]].y);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##canopy_origin_z", &canopy_origins[canopy_labels[(std::string) current_item]].z);
                ImGui::SameLine();
                ImGui::Text("Canopy Origin");
                // ####### PLANT COUNT ####### //
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("##plant_count_x", &plant_counts[canopy_labels[current_item]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("##plant_count_y", &plant_counts[canopy_labels[current_item]].y);
                ImGui::SameLine();
                ImGui::Text("Plant Count");
                // ####### PLANT SPACING ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("##plant_spacing_x", &plant_spacings[canopy_labels[current_item]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("##plant_spacing_y", &plant_spacings[canopy_labels[current_item]].y);
                ImGui::SameLine();
                ImGui::Text("Plant Spacing");
                // ####### PLANT LIBRARY NAME ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputText("Plant Library", &plant_library_names[rig_dict[(std::string) current_item]]);
                // ####### PLANT AGE ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Plant Age", &plant_ages[canopy_labels[current_item]]);
                // ####### GROUND CLIPPING HEIGHT ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Ground Clipping Height", &ground_clipping_heights[canopy_labels[current_item]]);

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Radiation")){
                current_tab = "Radiation";
                // ####### DIRECT RAY COUNT ####### //
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Direct Ray Count", &direct_ray_count);
                // ####### DIFFUSE RAY COUNT ####### //
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Diffuse Ray Count", &diffuse_ray_count);
                // ####### DIFFUSE EXTINCTION COEFFICIENT ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Diffuse Extinction Coefficient", &diffuse_extinction_coeff);
                // ####### SCATTERING DEPTH ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputInt("Scattering Depth", &scattering_depth);
                // ####### AIR TURBIDITY ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Air Turbidity", &air_turbidity);
                // ####### LOAD XML LIBRARY FILE ####### //
                ImGui::SetNextItemWidth(60);
                if (ImGui::Button("XML Library File")){
                    // OpenFileDialog();
                }
                ImGui::SameLine();
                std::string shorten_ground_texture_file = ground_texture_file;
                size_t last_ground_texture_file = ground_texture_file.rfind('/');
                if (last_ground_texture_file != std::string::npos){
                    shorten_ground_texture_file = ground_texture_file.substr(last_ground_texture_file + 1);
                }
                ImGui::Text(shorten_ground_texture_file.c_str());
                // ####### SOLAR DIRECT SPECTRUM ####### //
                // ImGui::SetNextItemWidth(60);
                // ImGui::InputText("Solar Direct Spectrum", &solar_direct_spectrum);
                // ####### LEAF REFLECTIVITY SPECTRUM ####### //
                // ImGui::SetNextItemWidth(60);
                // ImGui::InputText("Solar Direct Spectrum", &solar_direct_spectrum);
                // ####### LEAF TRANSMISSIVITY SPECTRUM ####### //
                // ImGui::SetNextItemWidth(60);
                // ImGui::InputText("Solar Direct Spectrum", &solar_direct_spectrum);
                // ####### LEAF EMISSIVITY ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Leaf Emissivity", &leaf_emissivity);
                // ####### GROUND REFLECTIVITY SPECTRUM ####### //
                // ImGui::SetNextItemWidth(60);
                // ImGui::InputText("Ground Reflectivity Spectrum", &solar_direct_spectrum);

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Rig")){
                current_tab = "Rig";
                static const char* current_rig = "rig_0";
                if (ImGui::BeginCombo("##rig_combo", current_rig)) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < rig_labels.size(); n++){
                        bool is_rig_selected = (current_rig == rig_labels[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(rig_labels[n].c_str(), is_rig_selected))
                            current_rig = rig_labels[n].c_str();
                            current_cam_position = "0";
                        if (is_rig_selected)
                        ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                if (ImGui::Button("Add Rig")){
                    std::string default_rig_label = "rig";
                    std::string new_rig_label = "rig_0";
                    int count = 0;
                    while (rig_dict.find(new_rig_label) != rig_dict.end()){
                        count++;
                        new_rig_label = default_rig_label + "_" + std::to_string(count);
                    }
                    rig_dict.insert({new_rig_label, rig_labels.size()});
                    camera_positions.push_back(camera_position);
                    camera_lookats.push_back(camera_lookat);
                    camera_labels.push_back(camera_label);
                    camera_position_vec.push_back(camera_position_vec[rig_dict[(std::string) current_rig]]);
                    camera_lookat_vec.push_back(camera_lookat_vec[rig_dict[(std::string) current_rig]]);
                    // camera_resolutions.push_back(camera_resolution);
                    // focal_plane_distances.push_back(focal_plane_distance);
                    // lens_diameters.push_back(lens_diameter);
                    // FOV_aspect_ratios.push_back(FOV_aspect_ratio);
                    // HFOVs.push_back(HFOV);
                    rig_labels.push_back(new_rig_label);
                    current_rig = new_rig_label.c_str();
                    std::string parent = "rig";
                    pugi::xml_node rig_block = helios.child(parent.c_str());
                    pugi::xml_node new_rig_node = helios.append_copy(rig_block);
                    std::string name = "label";
                    pugi::xml_attribute node_label = new_rig_node.attribute(name.c_str());
                    node_label.set_value(new_rig_label.c_str());
                }
                ImGui::SetNextItemWidth(100);
                ImGui::InputText("##rig_name", &rig_labels[rig_dict[(std::string) current_rig]]);
                ImGui::SameLine();
                ImGui::Text("Rig Name");
                // ####### CAMERA LABEL ####### //
                /* SINGLE CAMERA VERSION
                ImGui::SetNextItemWidth(60);
                // ImGui::InputText("Camera Label", &camera_labels[rig_dict[(std::string) current_rig]]);
                if (ImGui::BeginCombo("##cam_label_combo", camera_labels[rig_dict[(std::string) current_rig]].c_str())){
                    for (int n = 0; n < camera_names.size(); n++){
                        bool is_cam_label_selected = (camera_labels[rig_dict[(std::string) current_rig]] == camera_names[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(camera_names[n].c_str(), is_cam_label_selected)){
                            camera_labels[rig_dict[(std::string) current_rig]] = camera_names[n];
                        }
                        if (is_cam_label_selected)
                            ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                ImGui::Text("Camera Label");
                ImGui::EndTabItem();
                */
                ImGui::Text("Cameras:");
                for (int i = 0; i < camera_names.size(); i++){
                    std::string& camera_name = camera_names[i];

                    ImGui::SetNextItemWidth(60);

                    std::set curr_set = rig_camera_labels[rig_dict[(std::string) current_rig]];
                    // if (i % 3 != 0){
                    //     ImGui::SameLine();
                    // }
                    bool isCameraSelected = curr_set.find(camera_name) != curr_set.end();
                    ImGui::PushID(i);
                    if(ImGui::Checkbox(camera_name.c_str(), &isCameraSelected)){
                        if (isCameraSelected){
                            rig_camera_labels[rig_dict[(std::string) current_rig]].insert(camera_name);
                        }else{
                            rig_camera_labels[rig_dict[(std::string) current_rig]].erase(camera_name);
                        }
                    }
                    ImGui::PopID();
                }
                // ####### ADD KEYPOINT ####### //
                std::stringstream cam_pos_value;
                cam_pos_value << current_cam_position.c_str();
                int current_cam_position_;
                cam_pos_value >> current_cam_position_;
                std::string current_keypoint_ = std::to_string(keypoint_frames[rig_dict[(std::string) current_rig]][current_cam_position_]);
                if (ImGui::BeginCombo("##cam_combo", current_keypoint_.c_str())){
                    for (int n = 0; n < camera_position_vec[rig_dict[(std::string) current_rig]].size(); n++){
                        std::string select_cam_position = std::to_string(n);
                        std::string selected_keypoint = std::to_string(keypoint_frames[rig_dict[(std::string) current_rig]][n]);
                        bool is_pos_selected = (current_cam_position == select_cam_position); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(selected_keypoint.c_str(), is_pos_selected)){
                            current_cam_position = std::to_string(n);
                        }
                        if (is_pos_selected)
                        ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                    ImGui::EndCombo();
                }
                cam_pos_value << current_cam_position.c_str();
                cam_pos_value >> current_cam_position_;
                ImGui::SameLine();
                if (ImGui::Button("Add Keypoint")){
                    camera_position_vec[rig_dict[(std::string) current_rig]].push_back(camera_position_vec[rig_dict[(std::string) current_rig]][current_cam_position_]);
                    camera_lookat_vec[rig_dict[(std::string) current_rig]].push_back(camera_lookat_vec[rig_dict[(std::string) current_rig]][current_cam_position_]);
                    keypoint_frames[rig_dict[(std::string) current_rig]].push_back(keypoint_frames[rig_dict[(std::string) current_rig]].back() + 1);
                }
                // ####### KEYPOINT FRAME ####### //
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Keypoint Frame", &keypoint_frames[rig_dict[(std::string) current_rig]][current_cam_position_]);
                // ####### CAMERA POSITION ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_position_x", &camera_position_vec[rig_dict[(std::string) current_rig]][current_cam_position_].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_position_y", &camera_position_vec[rig_dict[(std::string) current_rig]][current_cam_position_].y);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_position_z", &camera_position_vec[rig_dict[(std::string) current_rig]][current_cam_position_].z);
                ImGui::SameLine();
                ImGui::Text("Camera Position");
                // ####### CAMERA LOOKAT ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_lookat_x", &camera_lookat_vec[rig_dict[(std::string) current_rig]][current_cam_position_].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_lookat_y", &camera_lookat_vec[rig_dict[(std::string) current_rig]][current_cam_position_].y);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_lookat_z", &camera_lookat_vec[rig_dict[(std::string) current_rig]][current_cam_position_].z);
                ImGui::SameLine();
                ImGui::Text("Camera Lookat");
                // ####### NUMBER OF IMAGES ####### //
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Number of Images", &num_images);
                num_images = std::max(num_images,
                    *std::max_element(keypoint_frames[rig_dict[(std::string) current_rig]].begin(), keypoint_frames[rig_dict[(std::string) current_rig]].end()) + 1);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Camera")){
                current_tab = "Camera";
                static const char* current_cam = camera_names[0].c_str();
                if (ImGui::BeginCombo("##camera_combo", current_cam)){
                    for (int n = 0; n < camera_names.size(); n++){
                        bool is_cam_selected = (current_cam == camera_names[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(camera_names[n].c_str(), is_cam_selected))
                            current_cam = camera_names[n].c_str();
                        if (is_cam_selected)
                        ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                if (ImGui::Button("Add Camera")){
                    std::string default_cam_name = "camera";
                    std::string new_cam_name = "camera_0";
                    int count = 0;
                    while (camera_dict.find(new_cam_name) != camera_dict.end()){
                        count++;
                        new_cam_name = default_cam_name + "_" + std::to_string(count);
                    }
                    camera_dict.insert({new_cam_name, camera_names.size()});
                    camera_resolutions.push_back(camera_resolution);
                    focal_plane_distances.push_back(focal_plane_distance);
                    lens_diameters.push_back(lens_diameter);
                    FOV_aspect_ratios.push_back(FOV_aspect_ratio);
                    HFOVs.push_back(HFOV);
                    camera_names.push_back(new_cam_name);
                    std::string parent = "camera";
                    pugi::xml_node camera_block = helios.child(parent.c_str());
                    pugi::xml_node new_cam_node = helios.append_copy(camera_block);
                    std::string name = "label";
                    pugi::xml_attribute node_label = new_cam_node.attribute(name.c_str());
                    node_label.set_value(new_cam_name.c_str());
                }
                ImGui::SetNextItemWidth(100);
                ImGui::InputText("##cam_name", &camera_names[camera_dict[(std::string) current_cam]]);
                ImGui::SameLine();
                ImGui::Text("Camera Label");
               // ####### CAMERA RESOLUTION ####### //
                ImGui::SetNextItemWidth(90);
                ImGui::InputInt("##camera_resolution_x", &camera_resolutions[camera_dict[(std::string) current_cam]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90);
                ImGui::InputInt("##camera_resolution_y", &camera_resolutions[camera_dict[(std::string) current_cam]].y);
                ImGui::SameLine();
                ImGui::Text("Camera Resolution");
                // ####### FOCAL PLANE DISTANCE ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Focal Plane Distance", &focal_plane_distances[camera_dict[(std::string) current_cam]]);
                // ####### LENS DIAMETER ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Lens Diameter", &lens_diameters[camera_dict[(std::string) current_cam]]);
                // ####### FOV ASPECT RATIO ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("FOV Aspect Ratio", &FOV_aspect_ratios[camera_dict[(std::string) current_cam]]);
                // ####### HFOV ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("HFOV", &HFOVs[camera_dict[(std::string) current_cam]]);
                //
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        // ImGui::Text("Hello, world %d", 123);
        // ImGui::Button("Save");
        // if (ImGui::Button("Save"))
        //     std::cout << "here" << std::endl;
        last_position = current_position;
        previously_collapsed = currently_collapsed;
        ImGui::End();

        // Rendering
        // (Your code clears your framebuffer, renders your other stuff etc.)
        // glClearColor(0.1f, 0.1f, 0.1f, 1.0f);  // Set a background color (e.g., dark grey)
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        // (Your code calls glfwSwapBuffers() etc.)

        std::this_thread::sleep_for(std::chrono::milliseconds(100/6));
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    }


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        user_input = true;
    }
}

std::string vec_to_string(const int2& v) {
    std::ostringstream oss;
    oss << v.x << " " << v.y;
    return oss.str();
}

std::string vec_to_string(const vec2& v) {
    std::ostringstream oss;
    oss << v.x << " " << v.y;
    return oss.str();
}

std::string vec_to_string(const vec3& v) {
    std::ostringstream oss;
    oss << v.x << " " << v.y << " " << v.z;
    return oss.str();
}


std::map<std::string, int> get_node_labels(const std::string& name, const std::string& parent, std::vector<std::string>& labels_vec){
    int counter = 0;
    std::map<std::string, int> labels_dict = {};
    pugi::xml_node helios = xmldoc.child("helios");
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        std::string default_value = "canopy_0";
        if (!p.attribute(name.c_str()).empty()){
            const char *node_str = p.attribute(name.c_str()).value();
            default_value = (std::string) node_str;
        }
        labels_vec.push_back(default_value);
        labels_dict.insert({default_value, counter});
        counter++;
    }
    return labels_dict;
}


void get_xml_value(const std::string& name, const std::string& parent, int &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if( node.empty() ){
        std::cout << "WARNING: No value given for '" << name << "'. Using default value of " << default_value << std::endl;
    }else {
        const char *node_str = node.child_value();
        if (!parse_int(node_str, default_value)) {
            helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
        } else if( default_value<0 ){
            helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than or equal to 0.");
        }
    }
}

void get_xml_value(const std::string& name, const std::string& parent, float &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if( node.empty() ){
        std::cout << "WARNING: No value given for '" << name << "'. Using default value of " << default_value << std::endl;
    }else {
        const char *node_str = node.child_value();
        if (!parse_float(node_str, default_value)) {
            helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
        } else if( default_value<0 ){
            helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than or equal to 0.");
        }
    }
}

void get_xml_value(const std::string& name, const std::string& parent, std::string &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if( node.empty() ){
        std::cout << "WARNING: No value given for '" << name << "'. Using default value of " << default_value << std::endl;
    }else {
        const char *node_str = node.child_value();
        default_value = node_str;
    }
}

void get_xml_value(const std::string& name, const std::string& parent, vec2 &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if( node.empty() ){
        std::cout << "WARNING: No value given for '" << name << "'. Using default value of " << default_value << std::endl;
    }else {
        const char *node_str = node.child_value();
        if (!parse_vec2(node_str, default_value)) {
            helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
        }else if( default_value.x<=0 || default_value.y<=0 ){
            helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than 0.");
        }
    }
}

void get_xml_value(const std::string& name, const std::string& parent, vec3 &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if( node.empty() ){
        std::cout << "WARNING: No value given for '" << name << "'. Using default value of " << default_value << std::endl;
    }else {
        const char *node_str = node.child_value();
        if (!parse_vec3(node_str, default_value)) {
            helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
        }
    }
}

void get_xml_value(const std::string& name, const std::string& parent, int2 &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if( node.empty() ){
        std::cout << "WARNING: No value given for '" << name << "'. Using default value of " << default_value << std::endl;
    }else {
        const char *node_str = node.child_value();
        if (!parse_int2(node_str, default_value)) {
            helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
        }else if( default_value.x<=0 || default_value.y<=0 ){
            helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than 0.");
        }
    }
}


void get_xml_values(const std::string& name, const std::string& parent, std::vector<vec2>& default_vec){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        node = p.child(name.c_str());
        if( node.empty() ){
            std::cout << "WARNING: No value given for '" << name << "'.";
        } else {
            const char *node_str = node.child_value();
            vec2 default_value;
            if (!parse_vec2(node_str, default_value)) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else if( default_value.x<=0 || default_value.y<=0 ){
                helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than 0.");
            }else{
                default_vec.push_back(default_value);
            }
        }
    }
}


void get_xml_values(const std::string& name, const std::string& parent, std::vector<vec3>& default_vec){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        node = p.child(name.c_str());
        if( node.empty() ){
            std::cout << "WARNING: No value given for '" << name << "'.";
        } else {
            const char *node_str = node.child_value();
            vec3 default_value;
            if (!parse_vec3(node_str, default_value)) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else{
                default_vec.push_back(default_value);
            }
        }
    }
}


void get_xml_values(const std::string& name, const std::string& parent, std::vector<std::string>& default_vec){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        node = p.child(name.c_str());
        if( node.empty() ){
            std::cout << "WARNING: No value given for '" << name << "'.";
        } else {
            const char *node_str = node.child_value();
            std::string default_value = node_str;
            default_vec.push_back(default_value);
        }
    }
}


void get_xml_values(const std::string& name, const std::string& parent, std::vector<float>& default_vec){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        node = p.child(name.c_str());
        if( node.empty() ){
            std::cout << "WARNING: No value given for '" << name << "'.";
        } else {
            const char *node_str = node.child_value();
            float default_value;
            if (!parse_float(node_str, default_value)) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else if( default_value<0 ){
                helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than or equal to 0.");
            }else{
                default_vec.push_back(default_value);
            }
        }
    }
}


void get_xml_values(const std::string& name, const std::string& parent, std::vector<int2>& default_vec){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        node = p.child(name.c_str());
        if( node.empty() ){
            std::cout << "WARNING: No value given for '" << name << "'.";
        } else {
            const char *node_str = node.child_value();
            int2 default_value;
            if (!parse_int2(node_str, default_value)) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else if( default_value.x<=0 || default_value.y<=0 ){
                helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than or equal to 0.");
            }else{
                default_vec.push_back(default_value);
            }
        }
    }
}


void get_xml_values(const std::string& name, const std::string& parent, std::vector<std::vector<vec3>>& default_vec){
    pugi::xml_node helios = xmldoc.child("helios");
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        std::vector<vec3> curr_vec = {};
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            const char *node_str = node.child_value();
            vec3 default_value;
            if (!parse_vec3(node_str, default_value)) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else{
                curr_vec.push_back(default_value);
            }
        }
        default_vec.push_back(curr_vec);
    }
}


void get_xml_values(const std::string& name, const std::string& parent, std::vector<std::set<std::string>>& default_vec){
    pugi::xml_node helios = xmldoc.child("helios");
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        std::set<std::string> curr_vec = {};
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            const char *node_str = node.child_value();
            std::string default_value;
            if ( node.empty() ) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else{
                default_value = node_str;
                curr_vec.insert(default_value);
            }
        }
        default_vec.push_back(curr_vec);
    }
}


std::map<std::string, int> set_node_labels(const std::string& name, const std::string& parent, std::vector<std::string>& labels_vec){
    int i = 0;
    pugi::xml_node helios = xmldoc.child("helios");
    std::map<std::string, int> labels_dict = {};
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_attribute node_label = p.attribute(name.c_str());
        node_label.set_value(labels_vec[i].c_str());
        labels_dict.insert({labels_vec[i], i});
        i++;
    }
    return labels_dict;
}


void set_xml_value(const std::string& name, const std::string& parent, int &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(std::to_string(default_value).c_str());
}

void set_xml_value(const std::string& name, const std::string& parent, float &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(std::to_string(default_value).c_str());
}

void set_xml_value(const std::string& name, const std::string& parent, std::string &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(default_value.c_str());
}

void set_xml_value(const std::string& name, const std::string& parent, int2 &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void set_xml_value(const std::string& name, const std::string& parent, vec2 &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void set_xml_value(const std::string& name, const std::string& parent, vec3 &default_value) {
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}


void set_xml_values(const std::string& name, const std::string& parent, std::vector<vec2>& default_values){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void set_xml_values(const std::string& name, const std::string& parent, std::vector<vec3>& default_values){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void set_xml_values(const std::string& name, const std::string& parent, std::vector<int2>& default_values){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void set_xml_values(const std::string& name, const std::string& parent, std::vector<std::string>& default_values){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(default_values[i].c_str());
        i++;
    }
}


void set_xml_values(const std::string& name, const std::string& parent, std::vector<int>& default_values){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(std::to_string(default_values[i]).c_str());
        i++;
    }
}


void set_xml_values(const std::string& name, const std::string& parent, std::vector<float>& default_values){
    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(std::to_string(default_values[i]).c_str());
        i++;
    }
}

// std::vector<std::string> rig_labels;
// vec3 camera_position(0,0,0); std::vector<vec3> camera_positions;
// // camera_positions = {camera_position}; std::vector<std::vector<vec3>> camera_position_list;
// vec3 camera_lookat(0,0,0); std::vector<vec3> camera_lookats;
// std::string camera_label = "RGB"; std::vector<std::string> camera_labels;
// int2 camera_resolution(1024, 1024); std::vector<int2> camera_resolutions;
// float focal_plane_distance = 0.4; std::vector<float> focal_plane_distances;
// float lens_diameter = 0.02; std::vector<float> lens_diameters;
// float FOV_aspect_ratio = 1.4; std::vector<float> FOV_aspect_ratios;
// float HFOV = 1.4; std::vector<float> HFOVs;

void recalculate_values(Context& context, float &PAR_absorbed, float &NIR_absorbed, float &LW_absorbed,
                        std::vector<std::string> &rig_labels, std::vector<vec3> &camera_positions,
                        std::vector<vec3> &camera_lookats, std::vector<std::string> &camera_labels,
                        std::vector<int2> &camera_resolutions, std::vector<float> &focal_plane_distances,
                        std::vector<float> &lens_diameters, std::vector<float> &FOV_aspect_ratios, std::vector<float> &HFOVs) {
    PlantArchitecture plantarchitecture(&context);

    InitializeSimulation(xml_input_file, &context);

    BuildGeometry(xml_input_file, &plantarchitecture, &context);

    RadiationModel radiation(&context);
    SolarPosition solarposition(&context);

    InitializeRadiation(xml_input_file, &solarposition, &radiation, &context);

    EnergyBalanceModel energybalancemodel(&context);
    BLConductanceModel boundarylayerconductance(&context);

    InitializeEnergyBalance(xml_input_file, &boundarylayerconductance, &energybalancemodel, &context);

    // -- main time loop -- //

    float turbidity;
    assert( context.doesGlobalDataExist( "air_turbidity" ) );
    context.getGlobalData( "air_turbidity", turbidity );

    float diffuse_extinction_coeff;
    assert( context.doesGlobalDataExist( "diffuse_extinction_coeff" ) );
    context.getGlobalData( "diffuse_extinction_coeff", diffuse_extinction_coeff );

    uint sun_ID;
    assert( context.doesGlobalDataExist( "sun_ID" ) );
    context.getGlobalData( "sun_ID", sun_ID );

    std::vector<uint> ground_UUIDs, leaf_UUIDs, petiolule_UUIDs, petiole_UUIDs, internode_UUIDs, peduncle_UUIDs, petal_UUIDs, pedicel_UUIDs, fruit_UUIDs;
    context.getGlobalData( "ground_UUIDs", ground_UUIDs );
    assert( !ground_UUIDs.empty() );
    context.getGlobalData( "leaf_UUIDs", leaf_UUIDs );
    assert( !leaf_UUIDs.empty() );

    float ground_area = context.sumPrimitiveSurfaceArea( ground_UUIDs );

    std::vector<std::string> timeseries_variables = context.listTimeseriesVariables();

    if( timeseries_variables.empty() ){
        std::cout << "No timeseries data was loaded. Skipping time loop." << std::endl;
    }else{

        uint num_time_points = context.getTimeseriesLength( timeseries_variables.front().c_str() );
        for( uint timestep = 0; timestep<num_time_points; timestep++ ){

            context.setCurrentTimeseriesPoint( timeseries_variables.front().c_str(), timestep );

            std::cout << "Timestep " << timestep << ": " << context.getDate() << " " << context.getTime() << std::endl;

            float air_temperature = 300.f;
            if( context.doesTimeseriesVariableExist( "air_temperature" ) ){
                air_temperature = context.queryTimeseriesData( "air_temperature", timestep );
            }
            context.setPrimitiveData( context.getAllUUIDs(), "air_temperature", air_temperature );

            float air_humidity = 0.5f;
            if( context.doesTimeseriesVariableExist( "air_humidity" ) ){
                air_humidity = context.queryTimeseriesData( "air_humidity", timestep );
                if( air_humidity > 1 ){
                    //try dividing by 100
                    air_humidity /= 100.f;
                    if( air_humidity > 1 ){
                        std::cout << "WARNING: air_humidity must be between 0 and 1. Setting to default value of 0.5." << std::endl;
                        air_humidity = 0.5f;
                    }else{
                        std::cout << "WARNING: air_humidity read from timeseries was greater than 1.0. It was assumed that the given value was in percent and was automatically divided by 100." << std::endl;
                    }
                }
            }
            context.setPrimitiveData( context.getAllUUIDs(), "air_humidity", air_humidity );

            vec3 sun_dir_vec = solarposition.getSunDirectionVector();

            radiation.setSourcePosition( sun_ID, sun_dir_vec );

            if( diffuse_extinction_coeff > 0 ){
                radiation.setDiffuseRadiationExtinctionCoeff("PAR", diffuse_extinction_coeff, sun_dir_vec);
                radiation.setDiffuseRadiationExtinctionCoeff("NIR", diffuse_extinction_coeff, sun_dir_vec);
            }

            float R_PAR_dir = solarposition.getSolarFluxPAR(101000, air_temperature, air_humidity, turbidity);
            float R_NIR_dir = solarposition.getSolarFluxNIR(101000, air_temperature, air_humidity, turbidity);
            float fdiff = solarposition.getDiffuseFraction(101000, air_temperature, air_humidity, turbidity);

            radiation.setSourceFlux(sun_ID, "PAR", R_PAR_dir * (1.f - fdiff));
            radiation.setDiffuseRadiationFlux("PAR", R_PAR_dir * fdiff);
            radiation.setSourceFlux(sun_ID, "NIR", R_NIR_dir * (1.f - fdiff));
            radiation.setDiffuseRadiationFlux("NIR", R_NIR_dir * fdiff);

            // Run the radiation model
            radiation.runBand({"PAR","NIR","LW"});

            context.calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_PAR", PAR_absorbed );
            PAR_absorbed /= ground_area;

            context.calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_NIR", NIR_absorbed );
            NIR_absorbed /= ground_area;

            context.calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_LW", LW_absorbed );
            PAR_absorbed /= ground_area;

            std::cout << "Absorbed PAR: " << PAR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed NIR: " << NIR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed LW: " << LW_absorbed << " W/m^2" << std::endl;
        }
        // RIG BLOCK
        radiation.addRadiationBand("red");
        radiation.disableEmission("red");
        radiation.setSourceFlux(sun_ID, "red", 2.f);
        radiation.setScatteringDepth("red", 2);

        radiation.copyRadiationBand("red", "green");
        radiation.copyRadiationBand("red", "blue");

        std::vector<std::string> bandlabels = {"red", "green", "blue"};

        for (int n = 0; n < rig_labels.size(); n++){
            std::string cameralabel = camera_labels[n];

            vec3 camera_position = camera_positions[n];
            vec3 camera_lookat = camera_lookats[n];
            CameraProperties cameraproperties;
            cameraproperties.camera_resolution = camera_resolutions[n];
            cameraproperties.focal_plane_distance = focal_plane_distances[n];
            cameraproperties.lens_diameter = lens_diameters[n];
            cameraproperties.FOV_aspect_ratio = FOV_aspect_ratios[n];
            cameraproperties.HFOV = HFOVs[n];


            radiation.addRadiationCamera(cameralabel, bandlabels, camera_position, camera_lookat, cameraproperties, 100);

            // context.loadXML( "plugins/radiation/spectral_data/camera_spectral_library.xml", true);
            radiation.setCameraSpectralResponse(cameralabel, "red", "calibrated_sun_NikonB500_spectral_response_red");
            radiation.setCameraSpectralResponse(cameralabel, "green","calibrated_sun_NikonB500_spectral_response_green");
            radiation.setCameraSpectralResponse(cameralabel, "blue", "calibrated_sun_NikonB500_spectral_response_blue");
        }
        radiation.updateGeometry();

        radiation.runBand({"red", "green", "blue"});
        // RIG BLOCK END
    }
}

std::vector<vec3> linspace(vec3 a, vec3 b, int num_points){
    std::vector<vec3> result(num_points);
    for (int i = 0; i < num_points; i++){
        result[i].x = a.x + i * ( (b.x - a.x) / (float)num_points );
        result[i].y = a.y + i * ( (b.y - a.y) / (float)num_points );
        result[i].z = a.z + i * ( (b.z - a.z) / (float)num_points );
    }
    return result;
}

// void OpenFileDialog(){
//     const char* filePath = tinyfd_openFileDialog("Select a File", "", 0, NULL, NULL, 0);
//     if (filePath)
//     {
//         std::cout << "Selected file: " << filePath << std::endl;
//     }
//     else
//     {
//         std::cout << "No file selected." << std::endl;
//     }
// }
