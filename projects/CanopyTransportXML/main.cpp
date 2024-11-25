//#include "RadiationModel.h"
//#include "EnergyBalanceModel.h"
//#include "BoundaryLayerConductanceModel.h"
//#include "StomatalConductanceModel.h"
//#include "PhotosynthesisModel.h"
//#include "SolarPosition.h"
#include "PlantArchitecture.h"
#include "Visualizer.h"

#include "InitializeSimulation.h"
#include "BuildGeometry.h"
#include "InitializeRadiation.h"
#include "InitializeEnergyBalance.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "GLFW/glfw3.h"

#include <chrono>
#include <thread>

using namespace helios;

void key_callback(GLFWwindow*, int, int, int, int);
void get_xml_value(const std::string&, const std::string&, float&);
void get_xml_value(const std::string&, const std::string&, vec2&);
void get_xml_value(const std::string&, const std::string&, vec3&);
void get_xml_value(const std::string&, const std::string&, int2&);
std::string vec_to_string(const vec2&);
std::string vec_to_string(const vec3&);
std::string vec_to_string(const int2&);
void set_xml_value(const std::string&, const std::string&, float&);
void set_xml_value(const std::string&, const std::string&, vec2&);
void set_xml_value(const std::string&, const std::string&, vec3&);
void set_xml_value(const std::string&, const std::string&, int2&);
void recalculate_values(Context&, float&, float&, float&);

pugi::xml_document xmldoc;
bool user_input;
float PAR_absorbed;
float NIR_absorbed;
float LW_absorbed;

std::string xml_input_file = "../inputs/inputs.xml";

int main(){

    xml_input_file = "../inputs/inputs.xml"; //\todo Will eventually make this passable from a command-line argument

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

//
//        // Run the energy balance model
//        EnergyBalanceModel energybalance(&context);
//        energybalance.run();
//
//        // Run the stomatal conductance model
//        StomatalConductanceModel stomatalconductance(&context);
//        stomatalconductance.run(leaf_UUIDs);
//
//        // Run the photosynthesis model
//        PhotosynthesisModel photosynthesis(&context);
//        photosynthesis.run(leaf_UUIDs);
//
//        // Output results
//        context.writePrimitiveData("output.txt", {"net_photosynthesis", "latent_flux"}, leaf_UUIDs);



        }

    }

    Visualizer visualizer(800);

    visualizer.buildContextGeometry(&context);
    // visualizer.colorContextPrimitivesByData("radiation_flux_PAR");

    // visualizer.plotInteractive();

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
    vec3 domain_origin(0,0,0);
    vec2 domain_extent(10,10);
    int2 ground_resolution(1,1);
    get_xml_value("domain_origin", "helios", domain_origin);
    get_xml_value("domain_extent", "helios", domain_extent);
    get_xml_value("ground_resolution", "helios", ground_resolution);
    // CANOPY BLOCK
    vec3 canopy_origin(0,0,0);
    int2 plant_count(1,1);
    vec2 plant_spacing(0.5,0.5);
    float plant_age = 0;
    float ground_clipping_height = 0;
    get_xml_value("canopy_origin", "canopy_block", canopy_origin);
    get_xml_value("plant_count", "canopy_block", plant_count);
    get_xml_value("plant_spacing", "canopy_block", plant_spacing);
    get_xml_value("plant_age", "canopy_block", plant_age);
    get_xml_value("ground_clipping_height", "canopy_block", ground_clipping_height);

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

    // (Your code calls glfwPollEvents())
    // ...
    // Start the Dear ImGui frame

    std::string visualization_type = "color";
    bool switch_visualization = false;

    while (!glfwWindowShouldClose(window)) {
        // Poll and handle events
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
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;

        ImVec2 windowSize = ImGui::GetWindowSize();
        // ImGui::Begin("Editor", &my_tool_active, ImGuiWindowFlags_MenuBar);  // Begin a new window
        ImGui::Begin("Editor", &my_tool_active, window_flags);  // Begin a new window
        ImGui::SetNextWindowPos(ImVec2(windowSize.x, 0), ImGuiCond_Always);
        current_position = ImGui::GetWindowPos();
        currently_collapsed = ImGui::IsWindowCollapsed();

        if (current_position.x != last_position.x || current_position.y != last_position.y || currently_collapsed != previously_collapsed) {
            user_input = true;
        }
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
                if (ImGui::MenuItem("Save", "Ctrl+S"))   { /* Do stuff */ }
                if (ImGui::MenuItem("Close", "Ctrl+W"))  { my_tool_active = false; }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Visualization"))
            {
                if (ImGui::MenuItem("RGB (Default)") && visualization_type != "RGBAcolor")
                {
                    visualization_type = "RGBA";
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
                    // if (visualization_type != "color") {
                        visualizer.colorContextPrimitivesByData(visualization_type.c_str());
                    // }
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
            set_xml_value("domain_origin", "helios", domain_origin);
            set_xml_value("domain_extent", "helios", domain_extent);
            set_xml_value("ground_resolution", "helios", ground_resolution);
            set_xml_value("canopy_origin", "canopy_block", canopy_origin);
            set_xml_value("plant_count", "canopy_block", plant_count);
            set_xml_value("plant_spacing", "canopy_block", plant_spacing);
            set_xml_value("plant_age", "canopy_block", plant_age);
            set_xml_value("ground_clipping_height", "canopy_block", ground_clipping_height);
            // BuildGeometry(xml_input_file, &plantarchitecture, &context);
            // xmldoc.save_file("../inputs/inputs.xml");
            xmldoc.save_file(xml_input_file.c_str());
            context.~Context(); // clear Geometry
            Context context;
            visualizer.clearGeometry();
            recalculate_values(context, PAR_absorbed, NIR_absorbed, LW_absorbed);
            // visualizer.clearGeometry();
            visualizer.colorContextPrimitivesByData("radiation_flux_PAR");
            visualizer.buildContextGeometry(&context);
            visualizer.plotUpdate();
        }
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
        // ####### CANOPY ORIGIN ####### //
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("##canopy_origin_x", &canopy_origin.x);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("##canopy_origin_y", &canopy_origin.y);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputFloat("##canopy_origin_z", &canopy_origin.z);
        ImGui::SameLine();
        ImGui::Text("Canopy Origin");
        // ####### PLANT COUNT ####### //
        ImGui::SetNextItemWidth(70);
        ImGui::InputInt("##plant_count_x", &plant_count.x);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(70);
        ImGui::InputInt("##plant_count_y", &plant_count.y);
        ImGui::SameLine();
        ImGui::Text("Plant Count");
        // ####### PLANT SPACING ####### //
        ImGui::SetNextItemWidth(50);
        ImGui::InputFloat("##plant_spacing_x", &plant_spacing.x);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(50);
        ImGui::InputFloat("##plant_spacing_y", &plant_spacing.y);
        ImGui::SameLine();
        ImGui::Text("Plant Spacing");
        // ####### PLANT AGE ####### //
        ImGui::SetNextItemWidth(50);
        ImGui::InputFloat("Plant Age", &plant_age);
        // ####### GROUND CLIPPING HEIGHT ####### //
        ImGui::SetNextItemWidth(50);
        ImGui::InputFloat("Ground Clipping Height", &ground_clipping_height);

        // ####### RESULTS ####### //
        ImGui::Text("Absorbed PAR: %f W/m^2", PAR_absorbed);
        ImGui::Text("Absorbed NIR: %f W/m^2", NIR_absorbed);
        ImGui::Text("Absorbed LW: %f W/m^2", LW_absorbed);
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

void recalculate_values(Context& context, float &PAR_absorbed, float &NIR_absorbed, float &LW_absorbed) {
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
    }
}
