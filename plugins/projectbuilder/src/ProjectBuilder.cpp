#include "../include/ProjectBuilder.h"
#ifdef _WIN32
    #include <windows.h>
    #include <iostream>
    #include <commdlg.h>
#endif

using namespace helios;


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

std::vector<vec3> linspace(vec3 a, vec3 b, int num_points){
    std::vector<vec3> result(num_points);
    result[0] = a;
    for (int i = 1; i < num_points - 1; i++){
        result[i].x = a.x + i * ( (b.x - a.x) / ((float)num_points - 1.0) );
        result[i].y = a.y + i * ( (b.y - a.y) / ((float)num_points - 1.0) );
        result[i].z = a.z + i * ( (b.z - a.z) / ((float)num_points - 1.0) );
    }
    result[num_points - 1] = b;
    return result;
}

std::vector<vec3> interpolate(std::vector<int> keypoints, std::vector<vec3> positions, int num_points){
    std::vector<vec3> result(num_points);
    std::vector<int> keypoints_sorted = keypoints;
    std::sort(keypoints_sorted.begin(), keypoints_sorted.end());
    std::map<int, int> keypoints_loc;
    for (int i = 0; i < keypoints.size(); i++){
        keypoints_loc.insert({keypoints[i], i});
    }
    if (keypoints_sorted[keypoints_sorted.size() - 1] != num_points - 1){
        keypoints_sorted.push_back(num_points - 1);
        keypoints_loc.insert({num_points - 1, keypoints.size()});
        positions.push_back(positions[positions.size() - 1]);
    }
    for (int i = 0; i < keypoints_sorted.size() - 1; i++){
        int keypoint = keypoints_sorted[i];
        int keypoint_idx = keypoints_loc[keypoints_sorted[i]];
        int next_keypoint = keypoints_sorted[i + 1];
        int next_keypoint_idx = keypoints_loc[keypoints_sorted[i + 1]];
        std::vector<vec3> curr_positions = linspace(positions[keypoint_idx], positions[next_keypoint_idx], next_keypoint - keypoint + 1);
        for (int j = 0; j < next_keypoint - keypoint + 1; j++){
            int idx = j + keypoint;
            result[idx] = curr_positions[j];
        }
    }
    return result;
}

std::string file_dialog(){
    std::string file_name;
    #ifdef _WIN32
        // save CWD
        char CWD[MAX_PATH];
        GetCurrentDirectory(MAX_PATH, CWD);

        OPENFILENAME ofn;
        char szFile[260] = {0};

        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL;
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        ofn.lpstrFilter = "All Files\0*.*\0Text Files\0*.txt\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFileTitle = NULL;
        ofn.nMaxFileTitle = 0;
        ofn.lpstrInitialDir = NULL;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

        if (GetOpenFileName(&ofn)) {
            std::cout << "Selected file: " << ofn.lpstrFile << std::endl;
        } else {
            std::cout << "No file selected." << std::endl;
            return "";
        }

        // correct CWD
        SetCurrentDirectory(CWD);

        file_name = (std::string)ofn.lpstrFile;
    #endif

    return file_name;
}


void ProjectBuilder::build_from_xml(){
    context = new Context();

    plantarchitecture = new PlantArchitecture(context);

    InitializeSimulation(xml_input_file, context);

    BuildGeometry(xml_input_file, plantarchitecture, context);

    radiation = new RadiationModel(context);
    solarposition = new SolarPosition(context);

    InitializeRadiation(xml_input_file, solarposition, radiation, context);

    energybalancemodel = new EnergyBalanceModel(context);
    boundarylayerconductance = new BLConductanceModel(context);

    InitializeEnergyBalance(xml_input_file, boundarylayerconductance, energybalancemodel, context);

    // -- main time loop -- //
    assert( context->doesGlobalDataExist( "air_turbidity" ) );
    context->getGlobalData( "air_turbidity", turbidity );

    assert( context->doesGlobalDataExist( "diffuse_extinction_coeff" ) );
    context->getGlobalData( "diffuse_extinction_coeff", diffuse_extinction_coeff );

    assert( context->doesGlobalDataExist( "sun_ID" ) );
    context->getGlobalData( "sun_ID", sun_ID );

    bandlabels = {"red", "green", "blue"};

    context->getGlobalData( "ground_UUIDs", ground_UUIDs );
    assert( !ground_UUIDs.empty() );
    context->getGlobalData( "leaf_UUIDs", leaf_UUIDs );
    assert( !leaf_UUIDs.empty() );

    for (std::string& band : bandlabels){
        std::map<std::string, std::vector<float>> curr;
        for (std::pair<std::string, std::vector<uint>> primitive_pair : primitive_types){
            // curr[primitive_pair.first] = std::vector<float>{0.0, 0.0, 0.0};
            curr[primitive_pair.first] = std::vector<float>{0.25, 0.0, 0.0};
        }
        primitive_values[band] = curr;
    }

    // Update reflectivity, transmissivity, & emissivity for each band / primitive_type
    for (std::string band : bandlabels){
        for (std::pair<std::string, std::vector<uint>> primitive_pair : primitive_types){
            float reflectivity = primitive_values[band][primitive_pair.first][0];
            float transmissivity = primitive_values[band][primitive_pair.first][1];
            float emissivity = primitive_values[band][primitive_pair.first][2];
            std::string reflectivity_band = "reflectivity_" + band;
            std::string transmissivity_band = "transmissivity_" + band;
            std::string emissivity_band = "emissivity_" + band;
            context->setPrimitiveData(primitive_pair.second, reflectivity_band.c_str(), reflectivity);
            context->setPrimitiveData(primitive_pair.second, transmissivity_band.c_str(), transmissivity);
            context->setPrimitiveData(primitive_pair.second, emissivity_band.c_str(), emissivity);
        }
    }

    ground_area = context->sumPrimitiveSurfaceArea( ground_UUIDs );

    timeseries_variables = context->listTimeseriesVariables();

    if( timeseries_variables.empty() ){
        std::cout << "No timeseries data was loaded. Skipping time loop." << std::endl;
    }else{

        uint num_time_points = context->getTimeseriesLength( timeseries_variables.front().c_str() );
        for( uint timestep = 0; timestep<num_time_points; timestep++ ){

            context->setCurrentTimeseriesPoint( timeseries_variables.front().c_str(), timestep );

            std::cout << "Timestep " << timestep << ": " << context->getDate() << " " << context->getTime() << std::endl;

            if( context->doesTimeseriesVariableExist( "air_temperature" ) ){
                air_temperature = context->queryTimeseriesData( "air_temperature", timestep );
            }
            context->setPrimitiveData( context->getAllUUIDs(), "air_temperature", air_temperature );

            if( context->doesTimeseriesVariableExist( "air_humidity" ) ){
                air_humidity = context->queryTimeseriesData( "air_humidity", timestep );
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
            context->setPrimitiveData( context->getAllUUIDs(), "air_humidity", air_humidity );

            sun_dir_vec = solarposition->getSunDirectionVector();

            radiation->setSourcePosition( sun_ID, sun_dir_vec );

            if( diffuse_extinction_coeff > 0 ){
                radiation->setDiffuseRadiationExtinctionCoeff("PAR", diffuse_extinction_coeff, sun_dir_vec);
                radiation->setDiffuseRadiationExtinctionCoeff("NIR", diffuse_extinction_coeff, sun_dir_vec);
            }

            R_PAR_dir = solarposition->getSolarFluxPAR(101000, air_temperature, air_humidity, turbidity);
            R_NIR_dir = solarposition->getSolarFluxNIR(101000, air_temperature, air_humidity, turbidity);
            fdiff = solarposition->getDiffuseFraction(101000, air_temperature, air_humidity, turbidity);

            radiation->setSourceFlux(sun_ID, "PAR", R_PAR_dir * (1.f - fdiff));
            radiation->setDiffuseRadiationFlux("PAR", R_PAR_dir * fdiff);
            radiation->setSourceFlux(sun_ID, "NIR", R_NIR_dir * (1.f - fdiff));
            radiation->setDiffuseRadiationFlux("NIR", R_NIR_dir * fdiff);

            // Run the radiation model
            radiation->runBand({"PAR","NIR","LW"});

            context->calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_PAR", PAR_absorbed );
            PAR_absorbed /= ground_area;

            context->calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_NIR", NIR_absorbed );
            NIR_absorbed /= ground_area;

            context->calculatePrimitiveDataAreaWeightedSum( leaf_UUIDs, "radiation_flux_LW", LW_absorbed );
            PAR_absorbed /= ground_area;

            std::cout << "Absorbed PAR: " << PAR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed NIR: " << NIR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed LW: " << LW_absorbed << " W/m^2" << std::endl;
        }
        // RIG BLOCK
        context->loadXML( "plugins/radiation/spectral_data/camera_spectral_library.xml", true);
        radiation->setSourceSpectrum( sun_ID, "solar_spectrum_ASTMG173");

        radiation->addRadiationBand("red");
        radiation->disableEmission("red");
        radiation->setSourceFlux(sun_ID, "red", 2.f);
        radiation->setScatteringDepth("red", 2);

        radiation->copyRadiationBand("red", "green");
        radiation->copyRadiationBand("red", "blue");

        radiation->enforcePeriodicBoundary("xy");

        if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
            helios_runtime_error(xml_error_string);
        }

        rig_dict = get_node_labels("label", "rig", rig_labels);
        get_xml_values("camera_position", "rig", camera_positions);
        get_xml_values("camera_lookat", "rig", camera_lookats);
        get_xml_values("camera_label", "rig", camera_labels);
    }
    // RIG BLOCK
    num_images = 5;
    get_xml_values();
    for (int n = 0; n < rig_labels.size(); n++){
        std::string current_rig = rig_labels[n];
        for (int i = 1; i < camera_position_vec[rig_dict[(std::string) current_rig]].size(); i++){
            vec3 arrow_pos = camera_position_vec[rig_dict[(std::string) current_rig]][i - 1];
            vec3 arrow_direction_vec = arrow_pos - camera_position_vec[rig_dict[(std::string) current_rig]][i];
            SphericalCoord arrow_direction_sph = cart2sphere(arrow_direction_vec);
            vec3 arrow_scale(0.35, 0.35, 0.35);
            arrow_dict[arrow_count] = context->loadOBJ("../../../plugins/radiation/camera_light_models/Arrow.obj",
                                                    nullorigin, arrow_scale, nullrotation, RGB::blue, "YUP", true);
            context->rotatePrimitive(arrow_dict[arrow_count], arrow_direction_sph.elevation, "x");
            context->rotatePrimitive(arrow_dict[arrow_count], -arrow_direction_sph.azimuth, "z");
            context->translatePrimitive(arrow_dict[arrow_count], arrow_pos);
            context->setPrimitiveData(arrow_dict[arrow_count], "twosided_flag", uint(3));
            arrow_count++;
        }
    }
    for (std::string rig_label : rig_labels){
        int rig_index = rig_dict[rig_label];
        for (std::string rig_camera_label : rig_camera_labels[rig_index]){
            int camera_index = camera_dict[rig_camera_label];

            /* Load properties of camera */
            cameraproperties = new CameraProperties();
            cameraproperties->camera_resolution = camera_resolutions[camera_index];
            cameraproperties->focal_plane_distance = focal_plane_distances[camera_index];
            cameraproperties->lens_diameter = lens_diameters[camera_index];
            cameraproperties->FOV_aspect_ratio = FOV_aspect_ratios[camera_index];
            cameraproperties->HFOV = HFOVs[camera_index];

            /* Create new camera */
            std::string camera_label_ = rig_label + "_" + rig_camera_label;
            vec3 camera_position_ = camera_positions[rig_index];
            vec3 camera_lookat_ = camera_lookats[rig_index];
            radiation->addRadiationCamera(camera_label_, bandlabels, camera_position_, camera_lookat_, *cameraproperties, 100);
            radiation->setCameraSpectralResponse(camera_label_, "red", "calibrated_sun_NikonB500_spectral_response_red");
            radiation->setCameraSpectralResponse(camera_label_, "green","calibrated_sun_NikonB500_spectral_response_green");
            radiation->setCameraSpectralResponse(camera_label_, "blue", "calibrated_sun_NikonB500_spectral_response_blue");
            radiation->updateGeometry();
        }
    }
    radiation->runBand(bandlabels);

    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }
    helios = xmldoc.child("helios");
    pugi::xml_node node;
}


void ProjectBuilder::build_from_xml(std::string xml_path){
     xml_input_file = xml_path;
     build_from_xml();
}


std::map<std::string, int> ProjectBuilder::get_node_labels(const std::string& name, const std::string& parent, std::vector<std::string>& labels_vec){
    int counter = 0;
    std::map<std::string, int> labels_dict = {};
    helios = xmldoc.child("helios");
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


void ProjectBuilder::get_keypoints(const std::string& name, const std::string& parent, std::vector<std::vector<int>>& keypoints){
    helios = xmldoc.child("helios");
    const char *rig_ = "rig";
    for (pugi::xml_node rig = helios.child(rig_); rig; rig = rig.next_sibling(rig_)){
        int count = 0;
        std::vector<int> curr_keypoints;
        for (pugi::xml_node p = rig.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
            std::string default_value = std::to_string(count);
            if (!p.attribute(name.c_str()).empty()){
                const char *node_str = p.attribute(name.c_str()).value();
                default_value = (std::string) node_str;
            }
            std::stringstream keypoint_value;
            keypoint_value << default_value.c_str();
            int keypoint_;
            keypoint_value >> keypoint_;
            curr_keypoints.push_back(keypoint_);
            count++;
        }
        keypoints.push_back(curr_keypoints);
    }
}

void ProjectBuilder::set_keypoints(const std::string& name, const std::string& parent, std::vector<std::vector<int>>& keypoints){
    helios = xmldoc.child("helios");
    const char *rig_ = "rig";
    int rig_count = 0;
    for (pugi::xml_node rig = helios.child(rig_); rig; rig = rig.next_sibling(rig_)){
        int count = 0;
        for (pugi::xml_node p = rig.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
            std::string default_value = std::to_string(count);
            if (!p.attribute(name.c_str()).empty()){
                p.attribute(name.c_str()).set_value(keypoints[rig_count][count]);
            }
            count++;
        }
        rig_count++;
    }
} // TODO: test this function

void ProjectBuilder::get_xml_value(const std::string& name, const std::string& parent, int &default_value) {
    helios = xmldoc.child("helios");
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

void ProjectBuilder::get_xml_value(const std::string& name, const std::string& parent, float &default_value) {
    helios = xmldoc.child("helios");
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
        }
        // else if( default_value<0 ){
        //     helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than or equal to 0.");
        // }
    }
}

void ProjectBuilder::get_xml_value(const std::string& name, const std::string& parent, std::string &default_value) {
    helios = xmldoc.child("helios");
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

void ProjectBuilder::get_xml_value(const std::string& name, const std::string& parent, vec2 &default_value) {
    helios = xmldoc.child("helios");
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

void ProjectBuilder::get_xml_value(const std::string& name, const std::string& parent, vec3 &default_value) {
    helios = xmldoc.child("helios");
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

void ProjectBuilder::get_xml_value(const std::string& name, const std::string& parent, int2 &default_value) {
    helios = xmldoc.child("helios");
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


void ProjectBuilder::get_xml_values(const std::string& name, const std::string& parent, std::vector<vec2>& default_vec){
    helios = xmldoc.child("helios");
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
            }
            // else if( default_value.x<=0 || default_value.y<=0 ){
            //     helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than 0.");
            // }
            else{
                default_vec.push_back(default_value);
            }
        }
    }
}


void ProjectBuilder::get_xml_values(const std::string& name, const std::string& parent, std::vector<vec3>& default_vec){
    helios = xmldoc.child("helios");
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


void ProjectBuilder::get_xml_values(const std::string& name, const std::string& parent, std::vector<std::string>& default_vec){
    helios = xmldoc.child("helios");
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


void ProjectBuilder::get_xml_values(const std::string& name, const std::string& parent, std::vector<float>& default_vec){
    helios = xmldoc.child("helios");
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


void ProjectBuilder::get_xml_values(const std::string& name, const std::string& parent, std::vector<int2>& default_vec){
    helios = xmldoc.child("helios");
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


void ProjectBuilder::get_xml_values(const std::string& name, const std::string& parent, std::vector<std::vector<vec3>>& default_vec){
    helios = xmldoc.child("helios");
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


void ProjectBuilder::get_xml_values(const std::string& name, const std::string& parent, std::vector<std::set<std::string>>& default_vec){
    helios = xmldoc.child("helios");
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

std::map<std::string, int> ProjectBuilder::set_node_labels(const std::string& name, const std::string& parent, std::vector<std::string>& labels_vec){
    int i = 0;
    helios = xmldoc.child("helios");
    std::map<std::string, int> labels_dict = {};
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_attribute node_label = p.attribute(name.c_str());
        node_label.set_value(labels_vec[i].c_str());
        labels_dict.insert({labels_vec[i], i});
        i++;
    }
    return labels_dict;
}


void ProjectBuilder::set_xml_value(const std::string& name, const std::string& parent, int &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(std::to_string(default_value).c_str());
}

void ProjectBuilder::set_xml_value(const std::string& name, const std::string& parent, float &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(std::to_string(default_value).c_str());
}

void ProjectBuilder::set_xml_value(const std::string& name, const std::string& parent, std::string &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(default_value.c_str());
}

void ProjectBuilder::set_xml_value(const std::string& name, const std::string& parent, int2 &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void ProjectBuilder::set_xml_value(const std::string& name, const std::string& parent, vec2 &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void ProjectBuilder::set_xml_value(const std::string& name, const std::string& parent, vec3 &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<vec2>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<vec3>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<int2>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<std::string>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(default_values[i].c_str());
        i++;
    }
}


void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<int>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(std::to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<float>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(std::to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<std::vector<vec3>>& default_vec){
    helios = xmldoc.child("helios");
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        int j = 0;
        std::vector<vec3> curr_vec = {};
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            node.text().set(vec_to_string(default_vec[i][j]).c_str());
            j++;
        }
        i++;
    }
}


void ProjectBuilder::set_xml_values(const std::string& name, const std::string& parent, std::vector<std::set<std::string>>& default_vec){
    helios = xmldoc.child("helios");
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            p.remove_child(node);
        }
        for (std::string s : default_vec[i]){
            p.append_child(name.c_str()).set_value(s.c_str());
        }
        i++;
    }
} // TODO: test this function


void ProjectBuilder::visualize(){
    visualizer = new Visualizer(800);
    radiation->enableCameraModelVisualization();
    visualizer->buildContextGeometry(context);

    // Uncomment below for interactive
    // visualizer.plotInteractive();

    visualizer->addCoordinateAxes();
    visualizer->plotUpdate();

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // void* window;

    // glfwShowWindow((GLFWwindow *) window);

    // GLFWwindow* window = glfwCreateWindow(640, 480, "My Title", NULL, NULL);
    GLFWwindow* window = (GLFWwindow *)visualizer->getWindow();

    glfwShowWindow(window);

    bool show_demo_window = false;
    bool my_tool_active = true;
    // bool user_input = false;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
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

    bool switch_visualization = false;

    std::string current_cam_position = "0";
    depthMVP = visualizer->plotInit();
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

        // if (ImGui::IsKeyDown(ImGuiKey_Q)) {
        //     user_input = true;
        // }
        //
        // if (user_input)
        //     visualizer->plotUpdate();

        user_input = false;
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;
        ImGuiWindowFlags window_flags2 = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;

        // ImGui::SetNextWindowSize(ImVec2(500, 400));
        ImVec2 windowSize = ImGui::GetWindowSize();

        // TEST
        glm::mat4 perspectiveTransformationMatrix = visualizer->getPerspectiveTransformationMatrix();
        glm::vec4 origin_position;
        std::string current_label;
        //glm::mat4 depthMVP = visualizer->getDepthMVP();
        for (int n = 0; n < labels.size(); n++){
            current_label = labels[n];
            vec3 canopy_origin_ = canopy_origins[canopy_labels[current_label]];
            origin_position = glm::vec4(canopy_origin_.x, canopy_origin_.y, canopy_origin_.z + 0.5, 1.0);
            origin_position = perspectiveTransformationMatrix * origin_position;
            ImGui::SetNextWindowPos(ImVec2(windowSize.x + (origin_position.x / origin_position.w) * windowSize.x,
                                            windowSize.y - (origin_position.y / origin_position.w) * windowSize.y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(110, 10), ImGuiCond_Always);
            // double check above
            ImGui::Begin(current_label.c_str(), &my_tool_active);
            ImGui::End();
        }
        for (int n = 0; n < rig_labels.size(); n++){
            current_label = rig_labels[n];
            vec3 camera_position_ = camera_positions[rig_dict[current_label]];
            origin_position = glm::vec4(camera_position_.x, camera_position_.y, camera_position_.z + 0.5, 1.0);
            origin_position = perspectiveTransformationMatrix * origin_position;
            ImGui::SetNextWindowPos(ImVec2(windowSize.x + (origin_position.x / origin_position.w) * windowSize.x,
                                            windowSize.y - (origin_position.y / origin_position.w) * windowSize.y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(110, 10), ImGuiCond_Always);
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
        ImGui::Begin("Editor", &my_tool_active, window_flags2);  // Begin a new window
        // ImGui::SetNextWindowPos(ImVec2(windowSize.x - 400.0f, 0), ImGuiCond_Always); // flag -> can't move window with mouse
        ImGui::SetNextWindowPos(ImVec2(windowSize.x - 400.0f, 0), ImGuiCond_Always);
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
                if (ImGui::MenuItem("Open..", "Ctrl+O")){
                    std::string file_name = file_dialog();
                    if (!file_name.empty()){
                        get_xml_values(file_name);
                    }
                }
                if (ImGui::MenuItem("Save XML", "Ctrl+S")){
                    set_xml_values();
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
                if (switch_visualization){
                    const char* font_name = "LCD";
                    visualizer->addTextboxByCenter("LOADING...", vec3(.5,.5,0), make_SphericalCoord(0, 0),
                        RGB::red, 40, font_name, Visualizer::COORDINATES_WINDOW_NORMALIZED);
                    visualizer->plotUpdate();
                    visualizer->clearGeometry();
                    if (visualization_type != "RGB") {
                        visualizer->colorContextPrimitivesByData(visualization_type.c_str());
                        visualizer->addCoordinateAxes();
                    } else{
                        visualizer->clearColor();
                        visualizer->disableColorbar();
                        visualizer->addCoordinateAxes();
                    }
                    visualizer->buildContextGeometry(context);
                    visualizer->plotUpdate();
                    switch_visualization = false;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
        if (ImGui::Button("Reload")) {
            set_xml_values();
            const char* font_name = "LCD";
            visualizer->addTextboxByCenter("LOADING...", vec3(.5,.5,0), make_SphericalCoord(0, 0),
                RGB::red, 40, font_name, Visualizer::COORDINATES_WINDOW_NORMALIZED);
            visualizer->plotUpdate();
            visualizer->clearGeometry();
            delete context;
            delete plantarchitecture;
            delete radiation;
            delete solarposition;
            delete energybalancemodel;
            delete boundarylayerconductance;
            delete cameraproperties;
            build_from_xml();
            radiation->enableCameraModelVisualization();
            visualizer->buildContextGeometry(context);
            visualizer->addCoordinateAxes();
            visualizer->plotUpdate();
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
            // Update reflectivity, transmissivity, & emissivity for each band / primitive_type
            for (std::string band : bandlabels){
                for (std::pair<std::string, std::vector<uint>> primitive_pair : primitive_types){
                    float reflectivity = primitive_values[band][primitive_pair.first][0];
                    float transmissivity = primitive_values[band][primitive_pair.first][1];
                    float emissivity = primitive_values[band][primitive_pair.first][2];
                    std::string reflectivity_band = "reflectivity_" + band;
                    std::string transmissivity_band = "transmissivity_" + band;
                    std::string emissivity_band = "emissivity_" + band;
                    context->setPrimitiveData(primitive_pair.second, reflectivity_band.c_str(), reflectivity);
                    context->setPrimitiveData(primitive_pair.second, transmissivity_band.c_str(), transmissivity);
                    context->setPrimitiveData(primitive_pair.second, emissivity_band.c_str(), emissivity);
                }
            }
            // delete_arrows(context, arrow_dict);
            for (std::string rig_label : rig_labels){
                int rig_index = rig_dict[rig_label];
                for (std::string rig_camera_label : rig_camera_labels[rig_index]){
                    int camera_index = camera_dict[rig_camera_label];
                    std::string cameralabel = rig_label + "_" + rig_camera_label;
                    std::vector<vec3> interpolated_camera_positions = interpolate(keypoint_frames[rig_index], camera_position_vec[rig_index], num_images);
                    std::vector<vec3> interpolated_camera_lookats = interpolate(keypoint_frames[rig_index], camera_lookat_vec[rig_index], num_images);
                    for (int i = 0; i < interpolated_camera_positions.size(); i++){
                        radiation->setCameraPosition(cameralabel, interpolated_camera_positions[i]);
                        radiation->setCameraLookat(cameralabel, interpolated_camera_lookats[i]);
                        radiation->runBand({"red", "green", "blue"});
                        radiation->writeCameraImage( cameralabel, bandlabels, "RGB" + std::to_string(i), image_dir + rig_label + '/');
                        radiation->writeNormCameraImage( cameralabel, bandlabels, "norm" + std::to_string(i), image_dir + rig_label + '/');
                        radiation->writeDepthImageData( cameralabel, "depth" + std::to_string(i), image_dir + rig_label + '/');
                        radiation->writeNormDepthImage( cameralabel, "normdepth" + std::to_string(i), 3, image_dir + rig_label + '/');
                        radiation->writeImageBoundingBoxes( cameralabel, "bunny" + std::to_string(i), 0, "bbox", image_dir + rig_label + '/');
                    }
                }
            }
            // update_arrows(context, arrow_dict, camera_position_vec, rig_labels, rig_dict);
            visualizer->plotUpdate();
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
                    std::string csv_weather_file_ = file_dialog();
                    if (!csv_weather_file_.empty()){
                        csv_weather_file = csv_weather_file_;
                    }
                }
                ImGui::SameLine();
                std::string shorten_weather_file = csv_weather_file;
                size_t last_weather_file = csv_weather_file.rfind('/');
                if (last_weather_file != std::string::npos){
                    shorten_weather_file = csv_weather_file.substr(last_weather_file + 1);
                }
                ImGui::Text("%s", shorten_weather_file.c_str());
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
                    std::string ground_texture_file_ = file_dialog();
                    if (!ground_texture_file_.empty()){
                        ground_texture_file = ground_texture_file_;
                    }
                }
                ImGui::SameLine();
                std::string shorten = ground_texture_file;
                size_t last = ground_texture_file.rfind('/');
                if (last != std::string::npos){
                    shorten = ground_texture_file.substr(last + 1);
                }
                ImGui::Text("%s", shorten.c_str());

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Canopy")){
                current_tab = "Canopy";
                // ####### CANOPY ORIGIN ####### //
                if (ImGui::BeginCombo("##combo", current_canopy.c_str()))
                {
                    for (int n = 0; n < labels.size(); n++){
                        bool is_selected = (current_canopy == labels[n]);
                        if (ImGui::Selectable(labels[n].c_str(), is_selected))
                            current_canopy = labels[n];
                        if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                    ImGui::EndCombo();
                }
                ImGui::SetNextItemWidth(100);
                std::string prev_canopy_name = labels[canopy_labels[current_canopy]];
                ImGui::InputText("##canopy_name", &labels[canopy_labels[current_canopy]]);
                if (labels[canopy_labels[current_canopy]] != prev_canopy_name){
                    int temp = canopy_labels[current_canopy];
                    current_canopy = labels[canopy_labels[current_canopy]];
                    std::map<std::string, int>::iterator current_canopy_iter = canopy_labels.find(prev_canopy_name);
                    if (current_canopy_iter != canopy_labels.end()){
                        canopy_labels.erase(current_canopy_iter);
                    }
                    canopy_labels[current_canopy] = temp;
                }
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
                    // current_canopy = new_canopy_label;
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
                ImGui::InputFloat("##canopy_origin_x", &canopy_origins[canopy_labels[current_canopy]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##canopy_origin_y", &canopy_origins[canopy_labels[current_canopy]].y);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##canopy_origin_z", &canopy_origins[canopy_labels[current_canopy]].z);
                ImGui::SameLine();
                ImGui::Text("Canopy Origin");
                // ####### PLANT COUNT ####### //
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("##plant_count_x", &plant_counts[canopy_labels[current_canopy]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(70);
                ImGui::InputInt("##plant_count_y", &plant_counts[canopy_labels[current_canopy]].y);
                ImGui::SameLine();
                ImGui::Text("Plant Count");
                // ####### PLANT SPACING ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("##plant_spacing_x", &plant_spacings[canopy_labels[current_canopy]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("##plant_spacing_y", &plant_spacings[canopy_labels[current_canopy]].y);
                ImGui::SameLine();
                ImGui::Text("Plant Spacing");
                // ####### PLANT LIBRARY NAME ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputText("Plant Library", &plant_library_names[canopy_labels[current_canopy]]);
                // ####### PLANT AGE ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Plant Age", &plant_ages[canopy_labels[current_canopy]]);
                // ####### GROUND CLIPPING HEIGHT ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Ground Clipping Height", &ground_clipping_heights[canopy_labels[current_canopy]]);

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
                    std::string load_xml_library_file_ = file_dialog();
                    if (!load_xml_library_file_.empty()){
                        load_xml_library_file = load_xml_library_file_;
                    }
                }
                ImGui::SameLine();
                std::string shorten_xml_library_file = load_xml_library_file;
                size_t last_xml_library_file = load_xml_library_file.rfind('/');
                if (last_xml_library_file != std::string::npos){
                    shorten_xml_library_file = load_xml_library_file.substr(last_xml_library_file + 1);
                }
                ImGui::Text("%s", shorten_xml_library_file.c_str());
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
                // ImGui::SetNextItemWidth(60);
                // ImGui::InputFloat("Leaf Emissivity", &leaf_emissivity);
                // ####### GROUND REFLECTIVITY SPECTRUM ####### //
                // ImGui::SetNextItemWidth(60);
                // ImGui::InputText("Ground Reflectivity Spectrum", &solar_direct_spectrum);

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Spectra")){
                current_tab = "Spectra";
                // std::vector<uint> ground_UUIDs, leaf_UUIDs, petiolule_UUIDs, petiole_UUIDs, internode_UUIDs, peduncle_UUIDs, petal_UUIDs, pedicel_UUIDs, fruit_UUIDs;
                if (ImGui::BeginCombo("##combo_primitive", current_primitive.c_str()))
                {
                    for (int m = 0; m < primitive_names.size(); m++){
                        bool is_primitive_selected = (current_primitive == primitive_names[m]);
                        if (ImGui::Selectable(primitive_names[m].c_str(), is_primitive_selected))
                            current_primitive = primitive_names[m];
                        if (is_primitive_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                ImGui::Text("Primitive Types");
                if (ImGui::BeginCombo("##combo_band", current_band.c_str())){
                    for (int n = 0; n < bandlabels.size(); n++){
                        bool is_band_selected = (current_band == bandlabels[n]);
                        if (ImGui::Selectable(bandlabels[n].c_str(), is_band_selected))
                            current_band = bandlabels[n];
                        if (is_band_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                ImGui::Text("Bands");
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Reflectivity", &primitive_values[current_band][current_primitive][0]);
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Transmissivity", &primitive_values[current_band][current_primitive][1]);
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("Emissivity", &primitive_values[current_band][current_primitive][2]);
                ImGui::EndTabItem();
            }
            // RIG TAB
            if (ImGui::BeginTabItem("Rig")){
                current_tab = "Rig";
                if (ImGui::BeginCombo("##rig_combo", current_rig.c_str())) // The second parameter is the label previewed before opening the combo.
                {
                    for (int n = 0; n < rig_labels.size(); n++){
                        bool is_rig_selected = (current_rig == rig_labels[n]); // You can store your selection however you want, outside or inside your objects
                        if (ImGui::Selectable(rig_labels[n].c_str(), is_rig_selected))
                            current_rig = rig_labels[n];
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
                    camera_position_vec.push_back(camera_position_vec[rig_dict[current_rig]]);
                    camera_lookat_vec.push_back(camera_lookat_vec[rig_dict[current_rig]]);
                    // camera_resolutions.push_back(camera_resolution);
                    // focal_plane_distances.push_back(focal_plane_distance);
                    // lens_diameters.push_back(lens_diameter);
                    // FOV_aspect_ratios.push_back(FOV_aspect_ratio);
                    // HFOVs.push_back(HFOV);
                    rig_labels.push_back(new_rig_label);
                    rig_camera_labels.push_back(rig_camera_labels[rig_dict[current_rig]]);
                    rig_light_labels.push_back(rig_light_labels[rig_dict[current_rig]]);
                    keypoint_frames.push_back(keypoint_frames[rig_dict[current_rig]]);
                    // current_rig = new_rig_label;
                    std::string parent = "rig";
                    pugi::xml_node rig_block = helios.child(parent.c_str());
                    pugi::xml_node new_rig_node = helios.append_copy(rig_block);
                    std::string name = "label";
                    pugi::xml_attribute node_label = new_rig_node.attribute(name.c_str());
                    node_label.set_value(new_rig_label.c_str());
                }
                ImGui::SetNextItemWidth(100);
                std::string prev_rig_name = rig_labels[rig_dict[current_rig]];
                ImGui::InputText("##rig_name", &rig_labels[rig_dict[current_rig]]);
                if (rig_labels[rig_dict[current_rig]] != prev_rig_name){
                    int temp = rig_dict[current_rig];
                    current_rig = rig_labels[rig_dict[current_rig]];
                    std::map<std::string, int>::iterator current_rig_iter = rig_dict.find(prev_rig_name);
                    if (current_rig_iter != rig_dict.end()){
                        rig_dict.erase(current_rig_iter);
                    }
                    rig_dict[current_rig] = temp;
                }
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
                // ####### CAMERA CHECKBOX ####### //
                ImGui::Text("Cameras:");
                for (int i = 0; i < camera_names.size(); i++){
                    std::string& camera_name = camera_names[i];

                    ImGui::SetNextItemWidth(60);

                    std::set curr_set = rig_camera_labels[rig_dict[current_rig]];
                    // if (i % 3 != 0){
                    //     ImGui::SameLine();
                    // }
                    bool isCameraSelected = curr_set.find(camera_name) != curr_set.end();
                    ImGui::PushID(i);
                    if(ImGui::Checkbox(camera_name.c_str(), &isCameraSelected)){
                        if (isCameraSelected){
                            rig_camera_labels[rig_dict[current_rig]].insert(camera_name);
                        }else{
                            rig_camera_labels[rig_dict[current_rig]].erase(camera_name);
                        }
                    }
                    ImGui::PopID();
                }
                // ####### LIGHT CHECKBOX ####### //
                ImGui::Text("Lights:");
                for (int i = 0; i < light_names.size(); i++){
                    std::string& light_name = light_names[i];

                    ImGui::SetNextItemWidth(60);

                    std::set curr_rig_light = rig_light_labels[rig_dict[current_rig]];
                    bool isLightSelected = curr_rig_light.find(light_name) != curr_rig_light.end();
                    ImGui::PushID(i);
                    if(ImGui::Checkbox(light_name.c_str(), &isLightSelected)){
                        if (isLightSelected){
                            rig_light_labels[rig_dict[current_rig]].insert(light_name);
                        }else{
                            rig_light_labels[rig_dict[current_rig]].erase(light_name);
                        }
                    }
                    ImGui::PopID();
                }
                // ####### ADD KEYPOINT ####### //
                std::stringstream cam_pos_value;
                cam_pos_value << current_cam_position.c_str();
                int current_cam_position_;
                cam_pos_value >> current_cam_position_;
                current_keypoint = std::to_string(keypoint_frames[rig_dict[current_rig]][current_cam_position_]);
                if (ImGui::BeginCombo("##cam_combo", current_keypoint.c_str())){
                    for (int n = 0; n < camera_position_vec[rig_dict[current_rig]].size(); n++){
                        std::string select_cam_position = std::to_string(n);
                        std::string selected_keypoint = std::to_string(keypoint_frames[rig_dict[current_rig]][n]);
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
                    camera_position_vec[rig_dict[current_rig]].push_back(camera_position_vec[rig_dict[current_rig]][current_cam_position_]);
                    camera_lookat_vec[rig_dict[current_rig]].push_back(camera_lookat_vec[rig_dict[current_rig]][current_cam_position_]);
                    keypoint_frames[rig_dict[current_rig]].push_back(keypoint_frames[rig_dict[current_rig]].back() + 1);
                }
                // ####### KEYPOINT FRAME ####### //
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Keypoint Frame", &keypoint_frames[rig_dict[current_rig]][current_cam_position_]);
                // ####### CAMERA POSITION ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_position_x", &camera_position_vec[rig_dict[current_rig]][current_cam_position_].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_position_y", &camera_position_vec[rig_dict[current_rig]][current_cam_position_].y);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_position_z", &camera_position_vec[rig_dict[current_rig]][current_cam_position_].z);
                ImGui::SameLine();
                ImGui::Text("Rig Position");
                // ####### CAMERA LOOKAT ####### //
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_lookat_x", &camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_lookat_y", &camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].y);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(60);
                ImGui::InputFloat("##camera_lookat_z", &camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].z);
                ImGui::SameLine();
                ImGui::Text("Rig Lookat");
                // ####### NUMBER OF IMAGES ####### //
                ImGui::SetNextItemWidth(80);
                ImGui::InputInt("Total Number of Frames", &num_images);
                num_images = std::max(num_images,
                    *std::max_element(keypoint_frames[rig_dict[current_rig]].begin(), keypoint_frames[rig_dict[(std::string) current_rig]].end()) + 1);
                ImGui::EndTabItem();
            }
            // CAMERA TAB
            if (ImGui::BeginTabItem("Camera")){
                current_tab = "Camera";
                if (ImGui::BeginCombo("##camera_combo", current_cam.c_str())){
                    for (int n = 0; n < camera_names.size(); n++){
                        bool is_cam_selected = (current_cam == camera_names[n]);
                        if (ImGui::Selectable(camera_names[n].c_str(), is_cam_selected))
                            current_cam = camera_names[n];
                        if (is_cam_selected)
                        ImGui::SetItemDefaultFocus();
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
                std::string prev_cam_name = camera_names[camera_dict[current_cam]];
                ImGui::InputText("##cam_name", &camera_names[camera_dict[current_cam]]);
                if (camera_names[camera_dict[current_cam]] != prev_cam_name){
                    int temp = camera_dict[current_cam];
                    current_cam = camera_names[camera_dict[current_cam]];
                    std::map<std::string, int>::iterator current_cam_iter = camera_dict.find(prev_cam_name);
                    if (current_cam_iter != camera_dict.end()){
                        camera_dict.erase(current_cam_iter);
                    }
                    camera_dict[current_cam] = temp;
                }
                ImGui::SameLine();
                ImGui::Text("Camera Label");
               // ####### CAMERA RESOLUTION ####### //
                ImGui::SetNextItemWidth(90);
                ImGui::InputInt("##camera_resolution_x", &camera_resolutions[camera_dict[current_cam]].x);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90);
                ImGui::InputInt("##camera_resolution_y", &camera_resolutions[camera_dict[current_cam]].y);
                ImGui::SameLine();
                ImGui::Text("Camera Resolution");
                // ####### FOCAL PLANE DISTANCE ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Focal Plane Distance", &focal_plane_distances[camera_dict[current_cam]]);
                // ####### LENS DIAMETER ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("Lens Diameter", &lens_diameters[camera_dict[current_cam]]);
                // ####### FOV ASPECT RATIO ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("FOV Aspect Ratio", &FOV_aspect_ratios[camera_dict[current_cam]]);
                // ####### HFOV ####### //
                ImGui::SetNextItemWidth(50);
                ImGui::InputFloat("HFOV", &HFOVs[camera_dict[current_cam]]);
                //
                ImGui::EndTabItem();
            }
            // LIGHT TAB
            if (ImGui::BeginTabItem("Light")){
                current_tab = "Light";
                if (ImGui::BeginCombo("##light_combo", current_light.c_str())){
                    for (int n = 0; n < light_names.size(); n++){
                        bool is_light_selected = (current_light == light_names[n]);
                        if (ImGui::Selectable(light_names[n].c_str(), is_light_selected))
                            current_light = light_names[n];
                        if (is_light_selected)
                        ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                if (ImGui::Button("Add Light")){
                    std::string default_light_name = "light";
                    std::string new_light_name = "light_0";
                    int count = 0;
                    while (light_dict.find(new_light_name) != light_dict.end()){
                        count++;
                        new_light_name = default_light_name + "_" + std::to_string(count);
                    }
                    light_dict.insert({new_light_name, light_names.size()});
                    light_types.push_back(light_types[light_dict[current_light]]);
                    light_direction_vec.push_back(light_direction_vec[light_dict[current_light]]);
                    light_direction_sph_vec.push_back(light_direction_sph_vec[light_dict[current_light]]);
                    light_rotation_vec.push_back(light_rotation_vec[light_dict[current_light]]);
                    light_size_vec.push_back(light_size_vec[light_dict[current_light]]);
                    light_radius_vec.push_back(light_radius_vec[light_dict[current_light]]);
                    light_names.push_back(new_light_name);
                    std::string parent = "light";
                    pugi::xml_node light_block = helios.child(parent.c_str());
                    pugi::xml_node new_light_node = helios.append_copy(light_block);
                    std::string name = "label";
                    pugi::xml_attribute node_label = new_light_node.attribute(name.c_str());
                    node_label.set_value(new_light_name.c_str());
                }
                ImGui::SetNextItemWidth(100);
                std::string prev_light_name = light_names[light_dict[current_light]];
                ImGui::InputText("##light_name", &light_names[light_dict[current_light]]);
                if (light_names[light_dict[current_light]] != prev_light_name){
                    int temp = light_dict[current_light];
                    current_light = light_names[light_dict[current_light]];
                    std::map<std::string, int>::iterator current_light_iter = light_dict.find(prev_light_name);
                    if (current_light_iter != light_dict.end()){
                        light_dict.erase(current_light_iter);
                    }
                    light_dict[current_light] = temp;
                }
                ImGui::SameLine();
                ImGui::Text("Light Label");
                // ####### LIGHT TYPE ############ //
                if (ImGui::BeginCombo("##light_type_combo", light_types[light_dict[current_light]].c_str())){
                    for (int n = 0; n < all_light_types.size(); n++){
                        bool is_type_selected = (light_types[light_dict[current_light]] == all_light_types[n]);
                        if (ImGui::Selectable(all_light_types[n].c_str(), is_type_selected)){
                            light_types[light_dict[current_light]] = all_light_types[n];
                        }
                        if (is_type_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                ImGui::Text("Light Type");
                // collimated -> direction
                // disk       -> position, radius, rotation
                // sphere     -> position, radius
                // sunsphere  -> direction
                // rectangle  -> position, size, rotation
                // ####### LIGHT DIRECTION ####### //
                if (light_types[light_dict[(std::string) current_light]] == "collimated" ||
                    light_types[light_dict[(std::string) current_light]] == "sunsphere"){
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_direction_x", &light_direction_vec[light_dict[current_light]].x);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_direction_y", &light_direction_vec[light_dict[current_light]].y);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_direction_z", &light_direction_vec[light_dict[current_light]].z);
                    ImGui::SameLine();
                    ImGui::Text("Light Direction");
                }
                // ####### LIGHT ROTATION ####### //
                if (light_types[light_dict[current_light]] == "disk" ||
                    light_types[light_dict[current_light]] == "rectangle"){
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_rotation_x", &light_rotation_vec[light_dict[current_light]].x);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_rotation_y", &light_rotation_vec[light_dict[current_light]].y);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_rotation_z", &light_rotation_vec[light_dict[current_light]].z);
                    ImGui::SameLine();
                    ImGui::Text("Light Rotation");
                }
                // ####### LIGHT SIZE ####### //
                if (light_types[light_dict[current_light]] == "rectangle"){
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_size_x", &light_size_vec[light_dict[current_light]].x);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_size_y", &light_size_vec[light_dict[current_light]].y);
                    ImGui::SameLine();
                    ImGui::Text("Light Size");
                }
                // ####### LIGHT RADIUS ####### //
                if (light_types[light_dict[current_light]] == "disk" ||
                    light_types[light_dict[current_light]] == "sphere"){
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##light_radius", &light_radius_vec[light_dict[current_light]]);
                    ImGui::SameLine();
                    ImGui::Text("Light Radius");
                }
                // LIGHT END
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
        visualizer->plotOnce(depthMVP);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwWaitEvents();
        // (Your code calls glfwSwapBuffers() etc.)

        std::this_thread::sleep_for(std::chrono::milliseconds(100/6));
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}


void ProjectBuilder::build_and_visualize(std::string xml_path){
    build_from_xml(xml_path);
    visualize();
}


void ProjectBuilder::set_xml_values(){
    // MAIN BLOCK
    set_xml_value("latitude", "helios", latitude);
    set_xml_value("longitude", "helios", longitude);
    set_xml_value("UTC_offset", "helios", UTC_offset);
    set_xml_value("csv_weather_file", "helios", csv_weather_file);
    set_xml_value("domain_origin", "helios", domain_origin);
    set_xml_value("domain_extent", "helios", domain_extent);
    set_xml_value("ground_resolution", "helios", ground_resolution);
    set_xml_value("ground_texture_file", "helios", ground_texture_file);
    // CANOPY BLOCK
    canopy_labels = set_node_labels("label", "canopy_block", labels);
    set_xml_value("canopy_origin", "canopy_block", canopy_origin);
    set_xml_value("plant_count", "canopy_block", plant_count);
    set_xml_value("plant_spacing", "canopy_block", plant_spacing);
    set_xml_value("plant_library_name", "canopy_block", plant_library_name);
    set_xml_value("plant_age", "canopy_block", plant_age);
    set_xml_value("ground_clipping_height", "canopy_block", ground_clipping_height);
    set_xml_values("canopy_origin", "canopy_block", canopy_origins);
    set_xml_values("plant_count", "canopy_block", plant_counts);
    set_xml_values("plant_spacing", "canopy_block", plant_spacings);
    set_xml_values("plant_library_name", "canopy_block", plant_library_names);
    set_xml_values("plant_age", "canopy_block", plant_ages);
    set_xml_values("ground_clipping_height", "canopy_block", ground_clipping_heights);
    // RIG BLOCK
    rig_dict = set_node_labels("label", "rig", rig_labels);
    // set_xml_value("camera_position", "rig", camera_position);
    // set_xml_value("camera_lookat", "rig", camera_lookat);
    set_xml_value("camera_label", "rig", camera_label);
    // set_xml_values("camera_position", "rig", camera_positions);
    set_xml_values("camera_position", "rig", camera_position_vec);
    // set_xml_values("camera_lookat", "rig", camera_lookats);
    set_xml_values("camera_lookat", "rig", camera_lookat_vec);
    // set_xml_values("camera_label", "rig", camera_labels);
    set_xml_values("camera_label", "rig", rig_camera_labels);
    set_keypoints("keypoint", "camera_position", keypoint_frames);
    // CAMERA BLOCK
    camera_dict = set_node_labels("label", "camera", camera_names);
    set_xml_value("camera_resolution", "camera", camera_resolution);
    set_xml_value("focal_plane_distance", "camera", focal_plane_distance);
    set_xml_value("lens_diameter", "camera", lens_diameter);
    set_xml_value("FOV_aspect_ratio", "camera", FOV_aspect_ratio);
    set_xml_value("HFOV", "camera", HFOV);
    set_xml_values("camera_resolution", "camera", camera_resolutions);
    set_xml_values("focal_plane_distance", "camera", focal_plane_distances);
    set_xml_values("lens_diameter", "camera", lens_diameters);
    set_xml_values("FOV_aspect_ratio", "camera", FOV_aspect_ratios);
    set_xml_values("HFOV", "camera", HFOVs);
    // LIGHT BLOCK
    set_xml_values("light_type", "light", light_types);
    set_xml_values("light_direction", "light", light_direction_vec);
    set_xml_values("light_rotation", "light", light_rotation_vec);
    set_xml_values("light_size", "light", light_size_vec);
    set_xml_values("light_radius", "light", light_radius_vec);
    light_dict = set_node_labels("label", "light", light_names);
    set_xml_values("light_label", "rig", rig_light_labels);
    // RADIATION BLOCK
    set_xml_value("direct_ray_count", "radiation", direct_ray_count);
    set_xml_value("diffuse_ray_count", "radiation", diffuse_ray_count);
    set_xml_value("diffuse_extinction_coeff", "radiation", diffuse_extinction_coeff);
    set_xml_value("scattering_depth", "radiation", scattering_depth);
    set_xml_value("air_turbidity", "radiation", air_turbidity);
    set_xml_value("load_xml_library_file", "radiation", load_xml_library_file);
    set_xml_value("solar_direct_spectrum", "radiation", solar_direct_spectrum);
    set_xml_value("leaf_reflectivity_spectrum", "radiation", leaf_reflectivity_spectrum);
    set_xml_value("leaf_transmissivity_spectrum", "radiation", leaf_transmissivity_spectrum);
    set_xml_value("leaf_emissivity", "radiation", leaf_emissivity);
    set_xml_value("ground_reflectivity_spectrum", "radiation", ground_reflectivity_spectrum);
    xmldoc.save_file(xml_input_file.c_str());
}


void ProjectBuilder::set_xml_values(std::string xml_path){
    xml_input_file = xml_path;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }
    set_xml_values();
}


void ProjectBuilder::get_xml_values(){
    // MAIN BLOCK
    get_xml_value("latitude", "helios", latitude);
    get_xml_value("longitude", "helios", longitude);
    get_xml_value("UTC_offset", "helios", UTC_offset);
    get_xml_value("csv_weather_file", "helios", csv_weather_file);
    get_xml_value("domain_origin", "helios", domain_origin);
    get_xml_value("domain_extent", "helios", domain_extent);
    get_xml_value("ground_resolution", "helios", ground_resolution);
    get_xml_value("ground_texture_file", "helios", ground_texture_file);
    // CANOPY BLOCK
    labels.clear();
    canopy_labels = get_node_labels("label", "canopy_block", labels);
    current_canopy = labels[0];
    get_xml_value("canopy_origin", "canopy_block", canopy_origin);
    get_xml_value("plant_count", "canopy_block", plant_count);
    get_xml_value("plant_spacing", "canopy_block", plant_spacing);
    get_xml_value("plant_library_name", "canopy_block", plant_library_name);
    get_xml_value("plant_age", "canopy_block", plant_age);
    get_xml_value("ground_clipping_height", "canopy_block", ground_clipping_height);
    canopy_origins.clear();
    get_xml_values("canopy_origin", "canopy_block", canopy_origins);
    plant_counts.clear();
    get_xml_values("plant_count", "canopy_block", plant_counts);
    plant_spacings.clear();
    get_xml_values("plant_spacing", "canopy_block", plant_spacings);
    plant_library_names.clear();
    get_xml_values("plant_library_name", "canopy_block", plant_library_names);
    plant_ages.clear();
    get_xml_values("plant_age", "canopy_block", plant_ages);
    ground_clipping_heights.clear();
    get_xml_values("ground_clipping_height", "canopy_block", ground_clipping_heights);
    // RIG BLOCK
    rig_labels.clear();
    rig_dict = get_node_labels("label", "rig", rig_labels);
    current_rig = rig_labels[0];
    get_xml_value("camera_position", "rig", camera_position);
    get_xml_value("camera_lookat", "rig", camera_lookat);
    get_xml_value("camera_label", "rig", camera_label);
    camera_positions.clear();
    get_xml_values("camera_position", "rig", camera_positions);
    camera_position_vec.clear();
    get_xml_values("camera_position", "rig", camera_position_vec);
    camera_lookats.clear();
    get_xml_values("camera_lookat", "rig", camera_lookats);
    camera_lookat_vec.clear();
    get_xml_values("camera_lookat", "rig", camera_lookat_vec);
    camera_labels.clear();
    get_xml_values("camera_label", "rig", camera_labels);
    rig_camera_labels.clear();
    get_xml_values("camera_label", "rig", rig_camera_labels);
    keypoint_frames.clear();
    get_keypoints("keypoint", "camera_position", keypoint_frames);
    current_keypoint = std::to_string(keypoint_frames[0][0]);
    // CAMERA BLOCK
    camera_names.clear();
    camera_dict = get_node_labels("label", "camera", camera_names);
    current_cam = camera_names[0];
    get_xml_value("camera_resolution", "camera", camera_resolution);
    get_xml_value("focal_plane_distance", "camera", focal_plane_distance);
    get_xml_value("lens_diameter", "camera", lens_diameter);
    get_xml_value("FOV_aspect_ratio", "camera", FOV_aspect_ratio);
    get_xml_value("HFOV", "camera", HFOV);
    camera_resolutions.clear();
    get_xml_values("camera_resolution", "camera", camera_resolutions);
    focal_plane_distances.clear();
    get_xml_values("focal_plane_distance", "camera", focal_plane_distances);
    lens_diameters.clear();
    get_xml_values("lens_diameter", "camera", lens_diameters);
    FOV_aspect_ratios.clear();
    get_xml_values("FOV_aspect_ratio", "camera", FOV_aspect_ratios);
    HFOVs.clear();
    get_xml_values("HFOV", "camera", HFOVs);
    // LIGHT BLOCK
    light_types.clear();
    get_xml_values("light_type", "light", light_types);
    light_direction_vec.clear();
    get_xml_values("light_direction", "light", light_direction_vec);
    light_direction_sph_vec.clear();
    for (vec3 vec : light_direction_vec){
        light_direction_sph_vec.push_back(cart2sphere(vec));
    }
    light_rotation_vec.clear();
    get_xml_values("light_rotation", "light", light_rotation_vec);
    light_size_vec.clear();
    get_xml_values("light_size", "light", light_size_vec);
    light_radius_vec.clear();
    get_xml_values("light_radius", "light", light_radius_vec);
    light_names.clear();
    light_dict = get_node_labels("label", "light", light_names);
    current_light = light_names[0];
    rig_light_labels.clear();
    get_xml_values("light_label", "rig", rig_light_labels);
    // RADIATION BLOCK
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
}

void ProjectBuilder::get_xml_values(std::string xml_path){
    xml_input_file = xml_path;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }
    get_xml_values();
}


