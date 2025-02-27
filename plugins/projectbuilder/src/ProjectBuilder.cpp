#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <iostream>
    #include <commdlg.h>
#elif defined(__APPLE__)
    #include <nfd.h>
    #include <stdio.h>
    #include <stdlib.h>
#elif defined(__linux__)
    #include <nfd.h>
    #include <stdio.h>
    #include <stdlib.h>
#endif

#include "ProjectBuilder.h"

#ifdef ENABLE_BOUNDARYLAYERCONDUCTANCEMODEL
    const bool enable_blconductance = true;
#else
    const bool enable_blconductance = false;
#endif //BOUNDARYLAYERCONDUCTANCEMODEL

#ifdef ENABLE_ENERGYBALANCEMODEL
    const bool enable_energybalance = true;
#else
    const bool enable_energybalance = false;
#endif //ENERGYBALANCEMODEL

#ifdef ENABLE_PLANT_ARCHITECTURE
    const bool enable_plantarchitecture = true;
#else
    const bool enable_plantarchitecture = false;
#endif //PLANT_ARCHITECTURE

#ifdef ENABLE_RADIATION_MODEL
    const bool enable_radiation = true;
#else
    const bool enable_radiation = false;
#endif //RADIATION_MODEL

#ifdef ENABLE_SOLARPOSITION
    const bool enable_solarposition = true;
#else
    const bool enable_solarposition = false;
#endif //SOLARPOSITION

#ifdef ENABLE_HELIOS_VISUALIZER
    const bool enable_visualizer = true;
#else
    const bool enable_visualizer = false;
#endif //HELIOS_VISUALIZER


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
    if (keypoints.size() == 1){
        std::fill(result.begin(), result.end(), positions[0]);
        return result;
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

void toggle_button(const char* str_id, bool* v){
    #ifdef ENABLE_HELIOS_VISUALIZER
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float height = ImGui::GetFrameHeight();
    float width = height * 1.55f;
    float radius = height * 0.50f;

    ImGui::InvisibleButton(str_id, ImVec2(width, height));
    if (ImGui::IsItemClicked())
        *v = !*v;

    float t = *v ? 1.0f : 0.0f;

    ImGuiContext& g = *GImGui;
    float ANIM_SPEED = 0.08f;
    if (g.LastActiveId == g.CurrentWindow->GetID(str_id))// && g.LastActiveIdTimer < ANIM_SPEED)
    {
        float t_anim = ImSaturate(g.LastActiveIdTimer / ANIM_SPEED);
        t = *v ? (t_anim) : (1.0f - t_anim);
    }

    ImU32 col_bg;
    if (ImGui::IsItemHovered())
        col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.78f, 0.78f, 0.78f, 1.0f), ImVec4(0.64f, 0.83f, 0.34f, 1.0f), t));
    else
        col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), ImVec4(0.56f, 0.83f, 0.26f, 1.0f), t));

    draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
    draw_list->AddCircleFilled(ImVec2(p.x + radius + t * (width - radius * 2.0f), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));
    #endif //HELIOS_VISUALIZER
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
    #elif defined(__APPLE__)
        nfdchar_t *outPath = NULL;
        nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath );

        if ( result == NFD_OKAY ) {
            puts("Success!");
            puts(outPath);
            free(outPath);
        }
        else if ( result == NFD_CANCEL ) {
            puts("User pressed cancel.");
        }
        else {
            // printf("Error: %s\n", NFD_GetError() );
            std::cout << "Error: " << NFD_GetError() << std::endl;
        }
    #elif defined(__linux__)
        nfdchar_t *outPath = NULL;
        nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath );

        if ( result == NFD_OKAY ) {
            puts("Success!");
            puts(outPath);
            free(outPath);
        }
        else if ( result == NFD_CANCEL ) {
            puts("User pressed cancel.");
        }
        else {
            // printf("Error: %s\n", NFD_GetError() );
            std::cout << "Error: " << NFD_GetError() << std::endl;
        }
    #endif
    // TODO: make sure file dialog works on macOS and Linux

    return file_name;
}


std::vector<std::string> get_xml_node_values(std::string xml_input, const std::string& name,
                                            const std::string& parent){
    int counter = 0;
    pugi::xml_document xml_library_doc;
    std::vector<std::string> labels_vec{};
    std::string xml_library_error = "Failed to load XML Library file";
    if( !open_xml_file(xml_input, xml_library_doc, xml_library_error) ) {
        helios_runtime_error(xml_library_error);
    }
    pugi::xml_node helios = xml_library_doc.child("helios");
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        std::string default_value = "";
        if (!p.attribute(name.c_str()).empty()){
            const char *node_str = p.attribute(name.c_str()).value();
            default_value = (std::string) node_str;
        }
        labels_vec.push_back(default_value);
        counter++;
    }
    return labels_vec;
}


void ProjectBuilder::deleteArrows(){
    for (auto& arrow : arrow_dict){
        context->deletePrimitive(arrow_dict.at(arrow.first));
    }
    arrow_dict.clear();
}


void ProjectBuilder::updateArrows(){
    #ifdef ENABLE_RADIATION_MODEL
    arrow_count = 0;
    for (int n = 0; n < rig_labels.size(); n++){
        std::string current_rig = rig_labels[n];
        for (int i = 1; i < camera_position_vec[rig_dict[current_rig]].size(); i++){
            vec3 arrow_pos = camera_position_vec[rig_dict[current_rig]][i - 1];
            vec3 arrow_direction_vec = arrow_pos - camera_position_vec[rig_dict[current_rig]][i];
            SphericalCoord arrow_direction_sph = cart2sphere(arrow_direction_vec);
            vec3 arrow_scale(0.35, 0.35, 0.35);
            arrow_dict[arrow_count] = context->loadOBJ("plugins/radiation/camera_light_models/Arrow.obj",
                                                    nullorigin, arrow_scale, nullrotation, RGB::blue, "YUP", true);
            context->rotatePrimitive(arrow_dict.at(arrow_count), arrow_direction_sph.elevation, "x");
            context->rotatePrimitive(arrow_dict.at(arrow_count), -arrow_direction_sph.azimuth, "z");
            context->translatePrimitive(arrow_dict.at(arrow_count), arrow_pos);
            context->setPrimitiveData(arrow_dict.at(arrow_count), "twosided_flag", uint(3));
            arrow_count++;
        }
    }
    #endif
}

void ProjectBuilder::updatePrimitiveTypes(){
    std::vector<uint> allUUIDs = context->getAllUUIDs();
    // Clear current primitive data
    primitive_names.clear();
    primitive_names_set.clear();
    primitive_UUIDs.clear();
    primitive_continuous.clear();
    primitive_values.clear();
    primitive_spectra.clear();
    //
    primitive_names.push_back("All");
    primitive_names_set.insert("All");
    primitive_continuous.insert({"All", {false, false, false}});
    primitive_spectra.insert({"All", {reflectivity_spectrum, transmissivity_spectrum, emissivity_spectrum}});
    for (auto &primitive_UUID : allUUIDs){
        std::string default_value;
        if(context->doesPrimitiveDataExist(primitive_UUID, "object_label")){
            context->getPrimitiveData(primitive_UUID, "object_label", default_value);
            if (primitive_names_set.find(default_value) == primitive_names_set.end()){
                primitive_names.push_back(default_value);
                primitive_names_set.insert(default_value);
            }
            if ( primitive_UUIDs.find(default_value) == primitive_UUIDs.end() ){
                std::vector<uint> new_UUIDs;
                // primitive_addresses[default_value] = &new_UUIDs;
                primitive_UUIDs.insert({default_value, new_UUIDs});
                // primitive_continuous[default_value] = primitive_continuous["All"];
                primitive_continuous.insert({default_value, {false, false, false}});
                for (std::string band : bandlabels){
                    primitive_values[band].insert({default_value, {reflectivity, transmissivity, emissivity}});
                }
                primitive_spectra.insert({default_value, {reflectivity_spectrum, transmissivity_spectrum, emissivity_spectrum}});
            }
            primitive_UUIDs[default_value].push_back(primitive_UUID);
        }
        current_primitive = "All";
    }
    for (std::pair<std::string, std::vector<uint>*> prim : primitive_addresses){
        if ( primitive_UUIDs.find(prim.first) == primitive_UUIDs.end() ){
            primitive_addresses[prim.first]->clear();
        }
    }
    for (std::pair<std::string, std::vector<uint>> prim : primitive_UUIDs){
        if ( primitive_addresses.find(prim.first) == primitive_addresses.end() ){
            primitive_addresses.insert({prim.first, &primitive_UUIDs[prim.first]});
        }else{
            primitive_addresses[prim.first] = &primitive_UUIDs[prim.first];
            // *primitive_addresses[prim.first] = prim.second;
        }
    }
}



void ProjectBuilder::updateSpectra(){
    for (std::pair<std::string, std::vector<uint>*> primitive_pair : primitive_addresses){
        if (!primitive_continuous[primitive_pair.first][0]){
            for (std::string band : bandlabels){
                float reflectivity = primitive_values[band][primitive_pair.first][0];
                std::string reflectivity_band = "reflectivity_" + band;
                context->setPrimitiveData(*primitive_pair.second, reflectivity_band.c_str(), reflectivity);
            }
        }else{
            std::string reflectivity_spectrum = primitive_spectra[primitive_pair.first][0];
            if( !reflectivity_spectrum.empty() ){
                context->setPrimitiveData( *primitive_addresses[primitive_pair.first], "reflectivity_spectrum", reflectivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << primitive_pair.first << "_reflectivity_spectrum'. Assuming " << primitive_pair.first << " primitives are black across all shortwave bands." << std::endl;
            }
        }
        if (!primitive_continuous[primitive_pair.first][1]){
            for (std::string band : bandlabels){
                float transmissivity = primitive_values[band][primitive_pair.first][1];
                std::string transmissivity_band = "transmissivity_" + band;
                context->setPrimitiveData(*primitive_pair.second, transmissivity_band.c_str(), transmissivity);
            }
        }else{
            std::string transmissivity_spectrum = primitive_spectra[primitive_pair.first][1];
            if( !transmissivity_spectrum.empty() ){
                context->setPrimitiveData( *primitive_addresses[primitive_pair.first], "transmissivity_spectrum", transmissivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << primitive_pair.first << "_transmissivity_spectrum'. Assuming " << primitive_pair.first << " primitives are black across all shortwave bands." << std::endl;
            }
        }
        if (!primitive_continuous[primitive_pair.first][2]){
            for (std::string band : bandlabels){
                float emissivity = primitive_values[band][primitive_pair.first][2];
                std::string emissivity_band = "emissivity_" + band;
                context->setPrimitiveData(*primitive_pair.second, emissivity_band.c_str(), emissivity);
            }
        }else{
            std::string emissivity_spectrum = primitive_spectra[primitive_pair.first][2];
            if( !emissivity_spectrum.empty() ){
                context->setPrimitiveData( *primitive_addresses[primitive_pair.first], "emissivity_spectrum", emissivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << primitive_pair.first << "_emissivity_spectrum'. Assuming " << primitive_pair.first << " primitives are black across all shortwave bands." << std::endl;
            }
        }
    }
}

void ProjectBuilder::updateCameras(){
    #ifdef ENABLE_RADIATION_MODEL
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
            for (auto &band : bandlabels){
                radiation->setCameraSpectralResponse(camera_label_, band, camera_calibrations[camera_index] + "_" + band);
            }
            radiation->updateGeometry();
        }
    }
    radiation->runBand(bandlabels);
    #endif //RADIATION_MODEL
}

void ProjectBuilder::record(){
    #ifdef ENABLE_RADIATION_MODEL
    std::string image_dir = "./saved/";
    // deleteArrows();
    std::vector<uint> temp_lights{};
    for (std::string rig_label : rig_labels){
        int rig_index = rig_dict[rig_label];
        std::vector<vec3> interpolated_camera_positions = interpolate(keypoint_frames[rig_index], camera_position_vec[rig_index], num_images_vec[rig_index]);
        std::vector<vec3> interpolated_camera_lookats = interpolate(keypoint_frames[rig_index], camera_lookat_vec[rig_index], num_images_vec[rig_index]);
        for (int i = 0; i < interpolated_camera_positions.size(); i++){
            // ADD RIG LIGHTS
            for (std::string light : rig_light_labels[rig_dict[rig_label]]){
                int light_idx = light_dict[light];
                uint new_light_UUID;
                if (light_types[light_idx] == "sphere"){
                    new_light_UUID = radiation->addSphereRadiationSource(interpolated_camera_positions[i], light_radius_vec[light_idx]);
                    temp_lights.push_back(new_light_UUID);
                }else if (light_types[light_dict[light]] == "rectangle"){
                    new_light_UUID = radiation->addRectangleRadiationSource(interpolated_camera_positions[i],
                        light_size_vec[light_idx], light_rotation_vec[light_idx]);
                    temp_lights.push_back(new_light_UUID);
                }else if (light_types[light_dict[light]] == "disk"){
                    new_light_UUID = radiation->addDiskRadiationSource(interpolated_camera_positions[i],
                        light_radius_vec[light_idx], light_rotation_vec[light_idx]);
                    temp_lights.push_back(new_light_UUID);
                }
                for (auto &band : bandlabels){
                    radiation->setSourceFlux(new_light_UUID, band, light_flux_vec[light_idx]);
                }
            }
            // radiation->setSourceFlux(light_UUID, band, flux_value)
            //
            for (std::string rig_camera_label : rig_camera_labels[rig_index]){
                int camera_index = camera_dict[rig_camera_label];
                std::string cameralabel = rig_label + "_" + rig_camera_label;
                radiation->setCameraPosition(cameralabel, interpolated_camera_positions[i]);
                radiation->setCameraLookat(cameralabel, interpolated_camera_lookats[i]);
            }
            radiation->runBand({"red", "green", "blue"});
            for (std::string rig_camera_label : rig_camera_labels[rig_index]){
                std::string cameralabel = rig_label + "_" + rig_camera_label;
                // Write Images
                radiation->writeCameraImage( cameralabel, bandlabels, "RGB" + std::to_string(i), image_dir + rig_label + '/');
                radiation->writeNormCameraImage( cameralabel, bandlabels, "norm" + std::to_string(i), image_dir + rig_label + '/');
                radiation->writeDepthImageData( cameralabel, "depth" + std::to_string(i), image_dir + rig_label + '/');
                radiation->writeNormDepthImage( cameralabel, "normdepth" + std::to_string(i), 3, image_dir + rig_label + '/');
                //
                // Bounding boxes for all primitive types
                for (std::string primitive_name : primitive_names){
                    if (!primitive_name.empty()){
                        primitive_name[0] = std::tolower(static_cast<unsigned char>(primitive_name[0]));
                    }
                    radiation->writeImageBoundingBoxes( cameralabel, primitive_name, 0, "bbox_" + primitive_name + std::to_string(i), image_dir + rig_label + '/');
                }
                //
            }
            // REMOVE RIG LIGHTS
            for (uint temp_light : temp_lights){
                radiation->deleteRadiationSource(temp_light);
            }
            temp_lights.clear();
            //
        }
    }
    // updateArrows();
    visualizer->plotUpdate();
    #endif //RADIATION_MODEL
}

void ProjectBuilder::buildFromXML(){
    context = new Context();

    // if (enable_plantarchitecture){
    #ifdef ENABLE_PLANT_ARCHITECTURE
        plantarchitecture = new PlantArchitecture(context);
        std::cout << "Loaded PlantArchitecture plugin." << std::endl;
    // }else{
    #else
        std::cout << "Excluding PlantArchitecture plugin." << std::endl;
    // } //PLANT_ARCHITECTURE
    #endif //PLANT_ARCHITECTURE

    InitializeSimulation(xml_input_file, context);

    // if (enable_plantarchitecture){
    #ifdef ENABLE_PLANT_ARCHITECTURE
        BuildGeometry(xml_input_file, plantarchitecture, context);
    // } //PLANT_ARCHITECTURE
    #endif //PLANT_ARCHITECTURE

    // if (enable_radiation){
    #ifdef ENABLE_RADIATION_MODEL
        radiation = new RadiationModel(context);
        std::cout << "Loaded Radiation plugin." << std::endl;
    // }else{
    #else
        std::cout << "Excluding Radiation plugin." << std::endl;
    // } //RADIATION_MODEL
    #endif //RADIATION_MODEL

    // if (enable_solarposition){
    #ifdef ENABLE_SOLARPOSITION
        solarposition = new SolarPosition(context);
        std::cout << "Loaded SolarPosition plugin." << std::endl;
    // }else{
    #else
        std::cout << "Excluding SolarPosition plugin." << std::endl;
    // } //SOLARPOSITION
    #endif //SOLARPOSITION

    // if (enable_radiation){
    #ifdef ENABLE_RADIATION_MODEL
        InitializeRadiation(xml_input_file, solarposition, radiation, context);
    // } //RADIATION_MODEL
    #endif //RADIATION_MODEL

    // if (enable_energybalance){
    #ifdef ENABLE_ENERGYBALANCEMODEL
        energybalancemodel = new EnergyBalanceModel(context);
        std::cout << "Loaded EnergyBalance plugin." << std::endl;
    // }else{
    #else
        std::cout << "Excluding EnergyBalance plugin." << std::endl;
    // } //ENERGYBALANCEMODEL
    #endif //ENERGYBALANCEMODEL

    // if (enable_blconductance){
    #ifdef ENABLE_BOUNDARYLAYERCONDUCTANCEMODEL
        boundarylayerconductance = new BLConductanceModel(context);
        std::cout << "Loaded BoundaryLayerConductance plugin." << std::endl;
    // }else{
    #else
        std::cout << "Excluding BoundaryLayerConductance plugin." << std::endl;
    // } //BOUNDARYLAYERCONDUCTANCEMODEL
    #endif //BOUNDARYLAYERCONDUCTANCEMODEL

    // if (enable_blconductance && enable_energybalance){
    #if defined(ENABLE_ENERGYBALANCEMODEL) && defined(ENABLE_BOUNDARYLAYERCONDUCTANCEMODEL)
        InitializeEnergyBalance(xml_input_file, boundarylayerconductance, energybalancemodel, context);
    // } //BOUNDARYLAYERCONDUCTANCEMODE && ENERGYBALANCEMODEL
    #endif //BOUNDARYLAYERCONDUCTANCEMODE && ENERGYBALANCEMODEL

    // -- main time loop -- //
    if (enable_radiation){
        assert( context->doesGlobalDataExist( "air_turbidity" ) );
        context->getGlobalData( "air_turbidity", turbidity );
    }

    if (enable_radiation){
        assert( context->doesGlobalDataExist( "diffuse_extinction_coeff" ) );
        context->getGlobalData( "diffuse_extinction_coeff", diffuse_extinction_coeff );
    }

    if (enable_radiation){
        assert( context->doesGlobalDataExist( "sun_ID" ) );
        context->getGlobalData( "sun_ID", sun_ID );
    }

    // bandlabels = {"red", "green", "blue"};

    if (enable_plantarchitecture){
        // context->getGlobalData( "ground_UUIDs", ground_UUIDs );
        // context->getGlobalData( "leaf_UUIDs", leaf_UUIDs );
        for (std::string primitive_name : primitive_names){
            if (primitive_name != "All" && primitive_name != "all"){
                std::string primitive_name_lower = primitive_name;
                primitive_name_lower[0] = std::tolower(static_cast<unsigned char>(primitive_name_lower[0]));
                std::string primitive_UUIDs_name = primitive_name_lower + "_UUIDs";
                if ( context->doesGlobalDataExist( primitive_UUIDs_name.c_str() ) ){
                    context->getGlobalData( primitive_UUIDs_name.c_str(), *primitive_addresses[primitive_name] );
                    std::vector<uint> primitive_UUIDs = *primitive_addresses[primitive_name];
                    if (! primitive_UUIDs.empty()){
                        context->setPrimitiveData(*primitive_addresses[primitive_name], "object_label", primitive_name_lower);
                    }
                }
            }
        }
        assert( !ground_UUIDs.empty() );
        assert( !leaf_UUIDs.empty() );
        context->setPrimitiveData(ground_UUIDs, "object_label", "ground");
        context->setPrimitiveData(leaf_UUIDs, "object_label", "leaf");

    }

    // Update reflectivity, transmissivity, & emissivity for each band / primitive_type
    updateSpectra();

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

            // if (enable_solarposition && enable_radiation){
            #ifdef ENABLE_RADIATION_MODEL
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
            // } //SOLARPOSITION && RADIATION_MODEL
            #endif //SOLARPOSITION && RADIATION_MODEL

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
        if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
            helios_runtime_error(xml_error_string);
        }
        xmlGetValues();
        // RIG BLOCK
        // *** Loading any XML files needed for cameras *** //
        for (auto &xml_file : camera_xml_library_files) {
            if( xml_file.empty() || !std::filesystem::exists(xml_file) ){
                std::cout << "WARNING: Could not find camera XML library file: " + xml_file << ". Skipping..." << std::endl;
                continue;
            }
            context->loadXML( xml_file.c_str() );
        }
        // *** Loading any XML files needed for lights *** //
        for (auto &xml_file : light_xml_library_files) {
            if( xml_file.empty() || !std::filesystem::exists(xml_file) ){
                std::cout << "WARNING: Could not find light XML library file: " + xml_file << ". Skipping..." << std::endl;
                continue;
            }
            context->loadXML( xml_file.c_str() );
        }

        // if (enable_solarposition && enable_radiation){
        #ifdef ENABLE_RADIATION_MODEL

            radiation->setSourceSpectrum( sun_ID, solar_direct_spectrum );

            radiation->addRadiationBand("red");
            radiation->disableEmission("red");
            // radiation->setSourceFlux(sun_ID, "red", 2.f);
            radiation->setScatteringDepth("red", 2);

            radiation->copyRadiationBand("red", "green");
            radiation->copyRadiationBand("red", "blue");

            radiation->enforcePeriodicBoundary("xy");
        // } //SOLARPOSITION && RADIATION_MODEL
        #endif //SOLARPOSITION && RADIATION_MODEL
    }
    // RIG BLOCK
    num_images = 5;
    updateArrows();
    updateCameras();

    helios = xmldoc.child("helios");
}


void ProjectBuilder::buildFromXML(std::string xml_path){
     xml_input_file = xml_path;
     buildFromXML();
}


std::map<std::string, int> ProjectBuilder::getNodeLabels(const std::string& name, const std::string& parent, std::vector<std::string>& labels_vec){
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


void ProjectBuilder::getKeypoints(const std::string& name, const std::string& parent, std::vector<std::vector<int>>& keypoints){
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

void ProjectBuilder::setKeypoints(const std::string& name, const std::string& parent, std::vector<std::vector<int>>& keypoints){
    helios = xmldoc.child("helios");
    const char *rig_ = "rig";
    int rig_count = 0;
    for (pugi::xml_node rig = helios.child(rig_); rig; rig = rig.next_sibling(rig_)){
        int count = 0;
        for (pugi::xml_node p = rig.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
            std::string default_value = std::to_string(count);
            p.append_attribute(name.c_str());
            p.attribute(name.c_str()).set_value(keypoints[rig_count][count]);
            count++;
        }
        rig_count++;
    }
} // TODO: test this function

void ProjectBuilder::xmlGetValue(const std::string& name, const std::string& parent, int &default_value) {
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

void ProjectBuilder::xmlGetValue(const std::string& name, const std::string& parent, float &default_value) {
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

void ProjectBuilder::xmlGetValue(const std::string& name, const std::string& parent, std::string &default_value) {
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

void ProjectBuilder::xmlGetValue(const std::string& name, const std::string& parent, vec2 &default_value) {
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

void ProjectBuilder::xmlGetValue(const std::string& name, const std::string& parent, vec3 &default_value) {
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

void ProjectBuilder::xmlGetValue(const std::string& name, const std::string& parent, int2 &default_value) {
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<vec2>& default_vec){
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<vec3>& default_vec){
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::string>& default_vec){
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<float>& default_vec){
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<int>& default_vec){
    helios = xmldoc.child("helios");
    pugi::xml_node node;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        node = p.child(name.c_str());
        if( node.empty() ){
            std::cout << "WARNING: No value given for '" << name << "'.";
        } else {
            const char *node_str = node.child_value();
            int default_value;
            if (!parse_int(node_str, default_value)) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else if( default_value<0 ){
                helios_runtime_error("ERROR: Value given for '" + name + "' must be greater than or equal to 0.");
            }else{
                default_vec.push_back(default_value);
            }
        }
    }
}


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<int2>& default_vec){
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::vector<vec3>>& default_vec){
    helios = xmldoc.child("helios");
    pugi::xml_node p_;
    if (parent != "helios") {
        p_ = helios.child(parent.c_str());
    }else{
        p_ = helios;
    }
    for (pugi::xml_node p = p_; p; p = p.next_sibling(parent.c_str())){
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<std::set<std::string>>& default_vec){
    helios = xmldoc.child("helios");
    pugi::xml_node p_;
    if (parent != "helios") {
        p_ = helios.child(parent.c_str());
    }else{
        p_ = helios;
    }
    for (pugi::xml_node p = p_; p; p = p.next_sibling(parent.c_str())){
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::set<std::string>& default_set){
    helios = xmldoc.child("helios");
    pugi::xml_node p_;
    if (parent != "helios") {
        p_ = helios.child(parent.c_str());
    }else{
        p_ = helios;
    }
    for (pugi::xml_node p = p_; p; p = p.next_sibling(parent.c_str())){
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            const char *node_str = node.child_value();
            std::string default_value;
            if ( node.empty() ) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
            }else{
                default_value = node_str;
                default_set.insert(default_value);
            }
        }
    }
}


std::map<std::string, int> ProjectBuilder::setNodeLabels(const std::string& name, const std::string& parent, std::vector<std::string>& labels_vec){
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


void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, int &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(std::to_string(default_value).c_str());
}

void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, float &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(std::to_string(default_value).c_str());
}

void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, std::string &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(default_value.c_str());
}

void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, int2 &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, vec2 &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, vec3 &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    node.text().set(vec_to_string(default_value).c_str());
}

void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<vec2>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<vec3>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<int2>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(vec_to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<std::string>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(default_values[i].c_str());
        i++;
    }
}


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<int>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(std::to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<float>& default_values){
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        pugi::xml_node node = p.child(name.c_str());
        node.text().set(std::to_string(default_values[i]).c_str());
        i++;
    }
}


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<std::vector<vec3>>& default_vec){
    helios = xmldoc.child("helios");
    int i = 0;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        std::vector<pugi::xml_node> remove = {};
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            remove.push_back(node);
        }
        for (pugi::xml_node &node : remove){
            p.remove_child(node);
        }
        for (int j = 0; j < default_vec[i].size(); j++){
            // p.append_child(name.c_str()).set_value(vec_to_string(default_vec[i][j]).c_str());
            pugi::xml_node new_node = p.append_child(name.c_str());
            new_node.text().set(vec_to_string(default_vec[i][j]).c_str());
        }
        i++;
    }
}


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::vector<std::set<std::string>>& default_vec){
    helios = xmldoc.child("helios");
    int i = 0;
    pugi::xml_node p_;
    if (parent != "helios") {
        p_ = helios.child(parent.c_str());
    }else{
        p_ = helios;
    }
    for (pugi::xml_node p = p_; p; p = p.next_sibling(parent.c_str())){
        std::vector<pugi::xml_node> remove{};
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            remove.push_back(node);
        }
        for (pugi::xml_node &node : remove){
            p.remove_child(node);
        }
        for (std::string s : default_vec[i]){
            // p.append_child(name.c_str()).set_value(s.c_str());
            pugi::xml_node new_node = p.append_child(name.c_str());
            new_node.text().set(s.c_str());
        }
        i++;
    }
} // TODO: test this function


void ProjectBuilder::xmlSetValues(const std::string& name, const std::string& parent, std::set<std::string>& default_set){
    helios = xmldoc.child("helios");
    int i = 0;
    pugi::xml_node p_;
    if (parent != "helios") {
        p_ = helios.child(parent.c_str());
    }else{
        p_ = helios;
    }
    for (pugi::xml_node p = p_; p; p = p.next_sibling(parent.c_str())){
        std::vector<pugi::xml_node> remove{};
        for (pugi::xml_node node = p.child(name.c_str()); node; node = node.next_sibling(name.c_str())){
            remove.push_back(node);
        }
        for (pugi::xml_node &node : remove){
            p.remove_child(node);
        }
        for (std::string s : default_set){
            // p.append_child(name.c_str()).set_value(s.c_str());
            pugi::xml_node new_node = p.append_child(name.c_str());
            new_node.text().set(s.c_str());
        }
        i++;
    }
} // TODO: test this function


void ProjectBuilder::visualize(){
    // if (enable_visualizer){
    #ifdef ENABLE_HELIOS_VISUALIZER
        visualizer = new Visualizer(800);
        #ifdef ENABLE_RADIATION_MODEL
        radiation->enableCameraModelVisualization();
        #endif //RADIATION_MODEL
        visualizer->buildContextGeometry(context);

        // Uncomment below for interactive
        // visualizer.plotInteractive();

        visualizer->addCoordinateAxes();
        visualizer->plotUpdate();

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        // ImFont* font_awesome = io.Fonts->AddFontFromFileTTF("plugins/visualizer/fonts/FontAwesome.ttf", 16.0f);
        ImFont* arial = io.Fonts->AddFontFromFileTTF("plugins/visualizer/fonts/Arial.ttf", 16.0f);
        io.Fonts->Build();
        // ImGui::PushFont(arial);
        io.FontDefault = arial;
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
        glm::mat4 depthMVP = visualizer->plotInit();
        while ( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose((GLFWwindow*)window) == 0 ) {
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
            // ImVec2 windowSize = ImGui::GetWindowSize();
            int windowSize_x = 0;
            int windowSize_y = 0;
            // visualizer->getWindowSize(windowSize_x, windowSize_y);
            glfwGetWindowSize((GLFWwindow*)window, &windowSize_x, &windowSize_y );
            int2 windowSize(windowSize_x, windowSize_y);
            // TEST
            glm::mat4 perspectiveTransformationMatrix = visualizer->getPerspectiveTransformationMatrix();
            glm::vec4 origin_position;
            std::string current_label;
            //glm::mat4 depthMVP = visualizer->getDepthMVP();
            for (int n = 0; n < labels.size(); n++){
                current_label = labels[n];
                vec3 canopy_origin_ = canopy_origins[canopy_labels[current_label]];
                origin_position = glm::vec4(canopy_origin_.x, canopy_origin_.y, canopy_origin_.z, 1.0);
                origin_position = perspectiveTransformationMatrix * origin_position;
                ImGui::SetNextWindowPos(ImVec2(((origin_position.x / origin_position.w) * 0.5f + 0.5f) * windowSize.x,
                                                (1.0f - ((origin_position.y / origin_position.w) * 0.5f + 0.5f)) * windowSize.y), ImGuiCond_Always);
                // ImGui::SetNextWindowPos(ImVec2(windowSize.x + (origin_position.x / origin_position.w) * windowSize.x,
                //                                 windowSize.y - (origin_position.y / origin_position.w) * windowSize.y), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(150, 10), ImGuiCond_Always);
                // double check above
                ImGui::Begin(current_label.c_str(), &my_tool_active);
                ImGui::End();
            }
            for (int n = 0; n < rig_labels.size(); n++){
                current_label = rig_labels[n];
                vec3 camera_position_ = camera_positions[rig_dict[current_label]];
                origin_position = glm::vec4(camera_position_.x, camera_position_.y, camera_position_.z, 1.0);
                origin_position = perspectiveTransformationMatrix * origin_position;
                ImGui::SetNextWindowPos(ImVec2(((origin_position.x / origin_position.w) * 0.5f + 0.5f) * windowSize.x,
                                                (1.0f - ((origin_position.y / origin_position.w) * 0.5f + 0.5f)) * windowSize.y), ImGuiCond_Always);
                // ImGui::SetNextWindowPos(ImVec2(windowSize.x + (origin_position.x / origin_position.w) * windowSize.x,
                //                                 windowSize.y - (origin_position.y / origin_position.w) * windowSize.y), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(150, 10), ImGuiCond_Always);
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
            // ImGui::SetNextWindowPos(ImVec2(windowSize.x - 400.0f, 0), ImGuiCond_Always);
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
                            xmlGetValues(file_name);
                        }
                    }
                    if (ImGui::MenuItem("Save XML", "Ctrl+S")){
                        xmlSetValues();
                    }
                    if (ImGui::MenuItem("Close", "Ctrl+W"))  { my_tool_active = false; }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Visualization")){
                    // ImGui::PushFont(font_awesome);
                    // io.FontDefault = font_awesome;
                    if (ImGui::MenuItem("! REFRESH LIST !")){
                        visualization_types.clear();
                        std::vector<uint> allUUIDs = context->getAllUUIDs();
                        for (auto &UUID : allUUIDs){
                            std::vector<std::string> primitiveData = context->listPrimitiveData(UUID);
                            for (auto &data : primitiveData){
                                visualization_types.insert(data);
                            }
                        }
                    }
                    // ImGui::PopFont();
                    // io.FontDefault = arial;
                    if (ImGui::MenuItem("RGB (Default)") && visualization_type != "RGB"){
                        visualization_type = "RGB";
                        switch_visualization = true;
                    }
                    for (auto &type : visualization_types){
                        if (ImGui::MenuItem(type.c_str()) && visualization_type != type)  {
                            visualization_type = type;
                            switch_visualization = true;
                        }
                    }
                    if (switch_visualization){
                        const char* font_name = "LCD";
                        visualizer->addTextboxByCenter("LOADING...", vec3(.5,.5,0), make_SphericalCoord(0, 0),
                            RGB::red, 40, font_name, Visualizer::COORDINATES_WINDOW_NORMALIZED);
                        visualizer->plotUpdate();
                        visualizer->clearGeometry();
                        if (visualization_type != "RGB") {
                            visualizer->colorContextPrimitivesByData(visualization_type.c_str());
                            visualizer->enableColorbar();
                            visualizer->addCoordinateAxes();
                        }else{
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
                xmlSetValues();
                const char* font_name = "LCD";
                visualizer->addTextboxByCenter("LOADING...", vec3(.5,.5,0), make_SphericalCoord(0, 0),
                    RGB::red, 40, font_name, Visualizer::COORDINATES_WINDOW_NORMALIZED);
                visualizer->plotUpdate();
                visualizer->clearGeometry();
                delete context;
                #ifdef ENABLE_PLANT_ARCHITECTURE
                delete plantarchitecture;
                #endif //PLANT_ARCHITECTURE
                #ifdef ENABLE_RADIATION_MODEL
                delete radiation;
                #endif //RADIATION_MODEL
                #ifdef ENABLE_SOLARPOSITION
                delete solarposition;
                #endif //SOLARPOSITION
                #ifdef ENABLE_ENERGYBALANCEMODEL
                delete energybalancemodel;
                #endif //ENERGYBALANCEMODEL
                #ifdef ENABLE_BOUNDARYLAYERCONDUCTANCEMODEL
                delete boundarylayerconductance;
                #endif //BOUNDARYLAYERCONDUCTANCEMODEL
                #ifdef ENABLE_RADIATION_MODEL
                delete cameraproperties;
                #endif //RADIATION_MODEL
                buildFromXML();
                #ifdef ENABLE_RADIATION_MODEL
                radiation->enableCameraModelVisualization();
                #endif //RADIATION_MODEL
                #ifdef ENABLE_PLANT_ARCHITECTURE
                visualizer->buildContextGeometry(context);
                #endif //PLANT_ARCHITECTURE
                visualizer->addCoordinateAxes();
                visualizer->plotUpdate();
            }
            std::string image_dir = "./saved/";
            bool dir = std::filesystem::create_directories(image_dir);
            if (!dir && !std::filesystem::exists(image_dir)){
                helios_runtime_error("Error: image output directory " + image_dir + " could not be created. Exiting...");
            }
            // context.setPrimitiveData(UUIDs_bunny, "bunny", uint(0));
            // plant segmentation bounding boxes
            // plant ID bounding boxes (plant architecture->optional plant output data)
            #ifdef ENABLE_RADIATION_MODEL
            ImGui::SameLine();
            if (ImGui::Button("Record")){
                // Update reflectivity, transmissivity, & emissivity for each band / primitive_type
                updateSpectra();
                // updateCameras(); //TODO: figure out why this causes an error
                record();
            }
            #endif //RADIATION_MODEL
            // ####### RESULTS ####### //
            ImGui::Text("Absorbed PAR: %f W/m^2", PAR_absorbed);
            ImGui::Text("Absorbed NIR: %f W/m^2", NIR_absorbed);
            ImGui::Text("Absorbed  LW: %f W/m^2", LW_absorbed);
            if (ImGui::BeginTabBar("Settings#left_tabs_bar")){
                if (ImGui::BeginTabItem("General")){
                    current_tab = "General";
                    // ####### LATITUDE ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputFloat("Latitude", &latitude);
                    // ####### LONGITUDE ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputFloat("Longitude", &longitude);
                    // ####### UTC OFFSET ####### //
                    ImGui::SetNextItemWidth(100);
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
                    for (char &c : shorten_weather_file){
                        if (c == '\\'){
                            c = '/';
                        }
                    }
                    size_t last_weather_file = shorten_weather_file.rfind('/');
                    if (last_weather_file != std::string::npos){
                        shorten_weather_file = shorten_weather_file.substr(last_weather_file + 1);
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
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("##ground_resolution_x", &ground_resolution.x);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(100);
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
                    for (char &c : shorten){
                        if (c == '\\'){
                            c = '/';
                        }
                    }
                    size_t last = shorten.rfind('/');
                    if (last != std::string::npos){
                        shorten = shorten.substr(last + 1);
                    }
                    ImGui::Text("%s", shorten.c_str());

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Object")){
                    current_tab = "Object";
                    if (ImGui::Button("Load Object File")){
                        std::string new_obj_file = file_dialog();
                        if ( !new_obj_file.empty() && std::filesystem::exists(new_obj_file) ){
                            if( std::filesystem::path(new_obj_file).extension() != ".obj" && std::filesystem::path(new_obj_file).extension() != ".ply" ){
                                std::cout << "Object file must have .obj or .ply extension." << std::endl;
                            }
                            else{
                                std::vector<uint> new_UUIDs;
                                if( std::filesystem::path(new_obj_file).extension() == ".obj" ){
                                    new_UUIDs = context->loadOBJ(new_obj_file.c_str());
                                } else if ( std::filesystem::path(new_obj_file).extension() == ".ply" ){
                                    new_UUIDs = context->loadPLY(new_obj_file.c_str());
                                }
                                visualizer->buildContextGeometry(context);
                                visualizer->plotUpdate();std::string default_object_label = "object";
                                std::string new_obj_label = "object_0";
                                int count = 0;
                                while (obj_names_dict.find(new_obj_label) != obj_names_dict.end()){
                                    count++;
                                    new_obj_label = default_object_label + "_" + std::to_string(count);
                                }
                                obj_names_dict.insert({new_obj_label, obj_files.size()});
                                obj_names.push_back(new_obj_label);
                                obj_files.push_back(new_obj_file);
                                obj_UUIDs.push_back(new_UUIDs);
                            }
                        }
                    }
                    if (ImGui::BeginCombo("##obj_combo", current_obj.c_str())){
                        for (int n = 0; n < obj_names.size(); n++){
                            bool is_obj_selected = (current_obj == obj_names[n]);
                            if (ImGui::Selectable(obj_names[n].c_str(), is_obj_selected))
                                current_obj = obj_names[n];
                            if (is_obj_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    if ( !current_obj.empty() ){
                        ImGui::SetNextItemWidth(100);
                        std::string prev_obj_name = obj_names[obj_names_dict[current_obj]];
                        ImGui::InputText("##obj_name", &obj_names[obj_names_dict[current_obj]]);
                        if (obj_names[obj_names_dict[current_obj]] != prev_obj_name){
                            int idx = obj_names_dict[current_obj];
                            current_obj = obj_names[obj_names_dict[current_obj]];
                            std::map<std::string, int>::iterator current_obj_iter = obj_names_dict.find(prev_obj_name);
                            if (current_obj_iter != obj_names_dict.end()){
                                obj_names_dict.erase(current_obj_iter);
                            }
                            obj_names_dict[current_obj] = idx;
                        }
                    }

                    ImGui::EndTabItem();
                }
            // if (enable_plantarchitecture){
            #ifdef ENABLE_PLANT_ARCHITECTURE
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
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("##plant_count_x", &plant_counts[canopy_labels[current_canopy]].x);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(100);
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
                    ImGui::SetNextItemWidth(80);
                    ImGui::InputText("Plant Library", &plant_library_names[canopy_labels[current_canopy]]);
                    // ####### PLANT AGE ####### //
                    ImGui::SetNextItemWidth(80);
                    ImGui::InputFloat("Plant Age", &plant_ages[canopy_labels[current_canopy]]);
                    // ####### GROUND CLIPPING HEIGHT ####### //
                    ImGui::SetNextItemWidth(80);
                    ImGui::InputFloat("Ground Clipping Height", &ground_clipping_heights[canopy_labels[current_canopy]]);

                    ImGui::EndTabItem();
                }
            // } //PLANT_ARCHITECTURE
            #endif //PLANT_ARCHITECTURE
            #ifdef ENABLE_RADIATION_MODEL
            if (enable_radiation){
                if (ImGui::BeginTabItem("Radiation")){
                    current_tab = "Radiation";
                    // LOAD XML LIBRARY FILE
                    ImGui::SetNextItemWidth(60);
                    if (ImGui::Button("Load XML Library File")){
                        std::string new_xml_library_file = file_dialog();
                        if (!new_xml_library_file.empty() && std::filesystem::exists(new_xml_library_file)){
                            if ( xml_library_files.find(new_xml_library_file) == xml_library_files.end() ){
                                xml_library_files.insert(new_xml_library_file);
                                std::vector<std::string> current_spectra_file = get_xml_node_values(new_xml_library_file, "label", "globaldata_vec2");
                                possible_spectra.insert(possible_spectra.end(), current_spectra_file.begin(), current_spectra_file.end());
                            }
                            context->loadXML( new_xml_library_file.c_str() );
                        }
                    }
                    // ####### DIRECT RAY COUNT ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("Direct Ray Count", &direct_ray_count);
                    // ####### DIFFUSE RAY COUNT ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("Diffuse Ray Count", &diffuse_ray_count);
                    // ####### DIFFUSE EXTINCTION COEFFICIENT ####### //
                    ImGui::SetNextItemWidth(60);
                    ImGui::InputFloat("Diffuse Extinction Coefficient", &diffuse_extinction_coeff);
                    // ####### SCATTERING DEPTH ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("Scattering Depth", &scattering_depth);
                    // ####### AIR TURBIDITY ####### //
                    ImGui::SetNextItemWidth(60);
                    ImGui::InputFloat("Air Turbidity", &air_turbidity);
                    // ####### SOLAR DIRECT SPECTRUM ####### //
                    ImGui::SetNextItemWidth(250);
                    // ImGui::InputText("Solar Direct Spectrum", &solar_direct_spectrum);
                    if (ImGui::BeginCombo("##combo_solar_direct_spectrum", solar_direct_spectrum.c_str())){
                        for (int n = 0; n < possible_spectra.size(); n++){
                            bool is_solar_direct_spectrum_selected = (solar_direct_spectrum == possible_spectra[n]);
                            if (ImGui::Selectable(possible_spectra[n].c_str(), is_solar_direct_spectrum_selected))
                                solar_direct_spectrum = possible_spectra[n];
                            if (is_solar_direct_spectrum_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::Text("Solar Direct Spectrum");
                    // ####### RADIATIVE PROPERTIES ####### //
                    ImGui::Text("Radiative Properties:");
                    ImGui::SetNextItemWidth(100);
                    if (ImGui::Button("Refresh")){
                        updatePrimitiveTypes();
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::BeginCombo("##combo_primitive", current_primitive.c_str())){
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
                    ImGui::SetNextItemWidth(100);
                    // default primitive data group
                    ImGui::Text("Select Primitive Type");
                    // REFLECTIVITY
                    ImGui::Text("Reflectivity:");
                    std::string toggle_display_reflectivity = "Manual Entry";
                    bool reflectivity_continuous = primitive_continuous[current_primitive][0];
                    toggle_button("##reflectivity_toggle", &reflectivity_continuous);
                    if (reflectivity_continuous != primitive_continuous[current_primitive][0]){
                        if (current_primitive == "All"){
                            for (auto &prim_values : primitive_continuous){
                                primitive_continuous[prim_values.first][0] = reflectivity_continuous;
                            }
                        }
                        primitive_continuous[current_primitive][0] = reflectivity_continuous;
                    }
                    if (primitive_continuous[current_primitive][0]){
                        toggle_display_reflectivity = "File Entry";
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(250);
                    if (!primitive_continuous[current_primitive][0]){
                        ImGui::Text("Select band:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        if (ImGui::BeginCombo("##combo_band_reflectivity", current_band_reflectivity.c_str())){
                            for (int n = 0; n < bandlabels.size(); n++){
                                bool is_band_selected = (current_band_reflectivity == bandlabels[n]);
                                if (ImGui::Selectable(bandlabels[n].c_str(), is_band_selected))
                                    current_band_reflectivity = bandlabels[n];
                                if (is_band_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                        ImGui::SameLine();
                        ImGui::Text("Enter value:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(80);
                        if (current_primitive == "All"){
                            float prev_reflectivity = reflectivity;
                            ImGui::InputFloat("##reflectivity_all", &reflectivity);
                            if (reflectivity != prev_reflectivity){
                                for (auto &prim_values : primitive_values[current_band_reflectivity]){
                                    primitive_values[current_band_reflectivity][prim_values.first][0] = reflectivity;
                                }
                            }
                        }else{
                            ImGui::InputFloat("##reflectivity", &primitive_values[current_band_reflectivity][current_primitive][0]);
                        }
                    }else{
                        std::string reflectivity_prev = primitive_spectra[current_primitive][0];
                        if (ImGui::BeginCombo("##reflectivity_combo", reflectivity_prev.c_str())){
                            for (int n = 0; n < possible_spectra.size(); n++){
                                bool is_spectra_selected = (primitive_spectra[current_primitive][0] == possible_spectra[n]);
                                if (ImGui::Selectable(possible_spectra[n].c_str(), is_spectra_selected))
                                    primitive_spectra[current_primitive][0] = possible_spectra[n];
                                if (is_spectra_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                        if (current_primitive == "All" && reflectivity_prev != primitive_spectra[current_primitive][0]){
                            for (auto &prim_spectrum : primitive_spectra){
                                primitive_spectra[prim_spectrum.first][0] = primitive_spectra[current_primitive][0];
                            }
                        }
                    }
                    ImGui::SameLine();
                    ImGui::Text(toggle_display_reflectivity.c_str());
                    // TRANSMISSIVITY
                    ImGui::Text("Transmissivity:");
                    std::string toggle_display_transmissivity = "Manual Entry";
                    bool transmissivity_continuous = primitive_continuous[current_primitive][1];
                    toggle_button("##transmissivity_toggle", &transmissivity_continuous);
                    if (transmissivity_continuous != primitive_continuous[current_primitive][1]){
                        if (current_primitive == "All"){
                            for (auto &prim_values : primitive_continuous){
                                primitive_continuous[prim_values.first][1] = transmissivity_continuous;
                            }
                        }
                        primitive_continuous[current_primitive][1] = transmissivity_continuous;
                    }
                    if (primitive_continuous[current_primitive][1]){
                        toggle_display_transmissivity = "File Entry";
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(250);
                    if (!primitive_continuous[current_primitive][1]){
                        ImGui::Text("Select band:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        if (ImGui::BeginCombo("##combo_band_transmissivity", current_band_transmissivity.c_str())){
                            for (int n = 0; n < bandlabels.size(); n++){
                                bool is_band_selected = (current_band_transmissivity == bandlabels[n]);
                                if (ImGui::Selectable(bandlabels[n].c_str(), is_band_selected))
                                    current_band_transmissivity = bandlabels[n];
                                if (is_band_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                        ImGui::SameLine();
                        ImGui::Text("Enter value:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(80);
                        if (current_primitive == "All"){
                            float prev_transmissivity = transmissivity;
                            ImGui::InputFloat("##transmissivity_all", &transmissivity);
                            if (transmissivity != prev_transmissivity){
                                for (auto &prim_values : primitive_values[current_band_transmissivity]){
                                    primitive_values[current_band_transmissivity][prim_values.first][1] = transmissivity;
                                }
                            }
                        }else{
                            ImGui::InputFloat("##transmissivity", &primitive_values[current_band_transmissivity][current_primitive][1]);
                        }
                    }else{
                        std::string transmissivity_prev = primitive_spectra[current_primitive][1];
                        if (ImGui::BeginCombo("##transmissivity_combo", transmissivity_prev.c_str())){
                            for (int n = 0; n < possible_spectra.size(); n++){
                                bool is_spectra_selected = (primitive_spectra[current_primitive][1] == possible_spectra[n]);
                                if (ImGui::Selectable(possible_spectra[n].c_str(), is_spectra_selected))
                                    primitive_spectra[current_primitive][1] = possible_spectra[n];
                                if (is_spectra_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                        if (current_primitive == "All" && transmissivity_prev != primitive_spectra[current_primitive][1]){
                            for (auto &prim_spectrum : primitive_spectra){
                                primitive_spectra[prim_spectrum.first][1] = primitive_spectra[current_primitive][1];
                            }
                        }
                    }
                    ImGui::SameLine();
                    ImGui::Text(toggle_display_transmissivity.c_str());
                    // EMISSIVITY
                    ImGui::Text("Emissivity:");
                    // ImGui::SetNextItemWidth(250);
                    // ImGui::Text("");
                    ImGui::Dummy(ImVec2(35.f, 0.f));
                    ImGui::SameLine();
                    ImGui::Text("Select band:");
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(60);
                    if (ImGui::BeginCombo("##combo_band_emissivity", current_band_emissivity.c_str())){
                        for (int n = 0; n < bandlabels.size(); n++){
                            bool is_band_selected = (current_band_emissivity == bandlabels[n]);
                            if (ImGui::Selectable(bandlabels[n].c_str(), is_band_selected))
                                current_band_emissivity = bandlabels[n];
                            if (is_band_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::Text("Enter value:");
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80);
                    if (current_primitive == "All"){
                        float prev_emissivity = emissivity;
                        ImGui::InputFloat("##emissivity_all", &emissivity);
                        if (emissivity != prev_emissivity){
                            for (auto &prim_values : primitive_values[current_band_emissivity]){
                                primitive_values[current_band_emissivity][prim_values.first][2] = emissivity;
                            }
                        }
                    }else{
                        ImGui::InputFloat("##emissivity", &primitive_values[current_band_emissivity][current_primitive][2]);
                    }
                    ImGui::SameLine();
                    ImGui::Text("Manual Entry");

                    ImGui::EndTabItem();
                }
            } //RADIATION_MODEL
            #endif //RADIATION_MODEL
            if (enable_radiation){
                // RIG TAB
                if (ImGui::BeginTabItem("Rig")){
                    current_tab = "Rig";
                    if (ImGui::BeginCombo("##rig_combo", current_rig.c_str())){
                        for (int n = 0; n < rig_labels.size(); n++){
                            bool is_rig_selected = (current_rig == rig_labels[n]);
                            if (ImGui::Selectable(rig_labels[n].c_str(), is_rig_selected))
                                current_rig = rig_labels[n];
                            current_cam_position = "0";
                            if (is_rig_selected)
                                ImGui::SetItemDefaultFocus();
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
                        num_images_vec.push_back(num_images_vec[rig_dict[current_rig]]);
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
                    std::string modified_current_keypoint = std::to_string(keypoint_frames[rig_dict[current_rig]][current_cam_position_] + 1); // 1-indexed value
                    if (ImGui::BeginCombo("##cam_combo", modified_current_keypoint.c_str())){
                        for (int n = 1; n <= camera_position_vec[rig_dict[current_rig]].size(); n++){
                            std::string select_cam_position = std::to_string(n - 1);
                            std::string selected_keypoint = std::to_string(keypoint_frames[rig_dict[current_rig]][n - 1]);
                            bool is_pos_selected = (current_cam_position == select_cam_position);
                            std::string modified_selected_keypoint = std::to_string(keypoint_frames[rig_dict[current_rig]][n - 1] + 1); // 1-indexed value
                            if (ImGui::Selectable(modified_selected_keypoint.c_str(), is_pos_selected)){
                                current_cam_position = std::to_string(n - 1);
                            }
                            if (is_pos_selected)
                                ImGui::SetItemDefaultFocus();
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
                    int modified_keypoint_frame = keypoint_frames[rig_dict[current_rig]][current_cam_position_] + 1; // 1-indexed value
                    ImGui::InputInt("Keypoint Frame", &modified_keypoint_frame);
                    if (modified_keypoint_frame != keypoint_frames[rig_dict[current_rig]][current_cam_position_] + 1){
                        keypoint_frames[rig_dict[current_rig]][current_cam_position_] = modified_keypoint_frame - 1;
                    }
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
                    ImGui::InputInt("Total Number of Frames", &num_images_vec[rig_dict[current_rig]]);
                    num_images_vec[rig_dict[current_rig]] = std::max(num_images_vec[rig_dict[current_rig]], *std::max_element(keypoint_frames[rig_dict[current_rig]].begin(), keypoint_frames[rig_dict[(std::string) current_rig]].end()) + 1);
                    ImGui::EndTabItem();
                }
                // CAMERA TAB
                if (ImGui::BeginTabItem("Camera")){
                    current_tab = "Camera";
                    // LOAD XML LIBRARY FILE
                    ImGui::SetNextItemWidth(60);
                    if (ImGui::Button("Load XML Library File")){
                        std::string new_xml_library_file = file_dialog();
                        if (!new_xml_library_file.empty() && std::filesystem::exists(new_xml_library_file)){
                            if ( camera_xml_library_files.find(new_xml_library_file) == camera_xml_library_files.end() ){
                                camera_xml_library_files.insert(new_xml_library_file);
                                std::vector<std::string> current_camera_file = get_xml_node_values(new_xml_library_file, "label", "globaldata_vec2");
                                possible_camera_calibrations.insert(possible_camera_calibrations.end(), current_camera_file.begin(), current_camera_file.end());
                            }
                            context->loadXML( new_xml_library_file.c_str() );
                        }
                    }
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
                        camera_calibrations.push_back(camera_calibrations[camera_dict[current_cam]]);
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
                    // ####### CAMERA CALIBRATION ####### //
                    std::string prev_cam_calibration = camera_calibrations[camera_dict[current_cam]];
                    if (ImGui::BeginCombo("##camera_calibration_combo", camera_calibrations[camera_dict[current_cam]].c_str())){
                        for (int n = 0; n < possible_camera_calibrations.size(); n++){
                            bool is_cam_calibration_selected = (camera_calibrations[camera_dict[current_cam]] == possible_camera_calibrations[n]);
                            if (ImGui::Selectable(possible_camera_calibrations[n].c_str(), is_cam_calibration_selected))
                                camera_calibrations[camera_dict[current_cam]] = possible_camera_calibrations[n];
                            if (is_cam_calibration_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::Text("Camera Calibration");
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
                    // LOAD XML LIBRARY FILE
                    ImGui::SetNextItemWidth(60);
                    if (ImGui::Button("Load XML Library File")){
                        std::string new_xml_library_file = file_dialog();
                        if (!new_xml_library_file.empty() && std::filesystem::exists(new_xml_library_file)){
                            if ( light_xml_library_files.find(new_xml_library_file) == light_xml_library_files.end() ){
                                light_xml_library_files.insert(new_xml_library_file);
                                std::vector<std::string> current_light_file = get_xml_node_values(new_xml_library_file, "label", "globaldata_vec2");
                                possible_light_spectra.insert(possible_light_spectra.end(), current_light_file.begin(), current_light_file.end());
                            }
                            context->loadXML( new_xml_library_file.c_str() );
                        }
                    }
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
                        light_spectra.push_back(light_spectra[light_dict[current_light]]);
                        light_types.push_back(light_types[light_dict[current_light]]);
                        light_direction_vec.push_back(light_direction_vec[light_dict[current_light]]);
                        light_direction_sph_vec.push_back(light_direction_sph_vec[light_dict[current_light]]);
                        light_rotation_vec.push_back(light_rotation_vec[light_dict[current_light]]);
                        light_size_vec.push_back(light_size_vec[light_dict[current_light]]);
                        light_radius_vec.push_back(light_radius_vec[light_dict[current_light]]);
                        light_names.push_back(new_light_name);
                        light_flux_vec.push_back(light_flux_vec[light_dict[current_light]]);
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
                    // ####### LIGHT SPECTRA ####### //
                    std::string prev_light_spectra = light_spectra[light_dict[current_light]];
                    if (ImGui::BeginCombo("##light_spectra_combo", light_spectra[light_dict[current_light]].c_str())){
                        for (int n = 0; n < possible_light_spectra.size(); n++){
                            bool is_light_spectra_selected = (light_spectra[light_dict[current_light]] == possible_light_spectra[n]);
                            if (ImGui::Selectable(possible_light_spectra[n].c_str(), is_light_spectra_selected))
                                light_spectra[light_dict[current_light]] = possible_light_spectra[n];
                            if (is_light_spectra_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::Text("Light Spectrum");
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
                    // ####### LIGHT SOURCE FLUX ####### //
                    ImGui::SetNextItemWidth(80);
                    ImGui::InputFloat("##source_flux", &light_flux_vec[light_dict[current_light]]);
                    ImGui::SameLine();
                    ImGui::Text("Source Flux");
                    // radiation->setSourceFlux(light_UUID, band, flux_value);
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
            visualizer->plotOnce(depthMVP, !io.WantCaptureMouse);
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
            if (!io.WantCaptureMouse){
                glfwWaitEvents();
            }
            // (Your code calls glfwSwapBuffers() etc.)

            std::this_thread::sleep_for(std::chrono::milliseconds(100/6));
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    // }else{
    #else
        std::cout << "Visualizer plugin is required for this function." << std::endl;
    // } //HELIOS_VISUALIZER
    #endif //HELIOS_VISUALIZER
}


void ProjectBuilder::buildAndVisualize(std::string xml_path){
    buildFromXML(xml_path);
    visualize();
}


void ProjectBuilder::buildAndVisualize(){
    buildFromXML();
    visualize();
}


void ProjectBuilder::xmlSetValues(){
    // MAIN BLOCK
    xmlSetValue("latitude", "helios", latitude);
    xmlSetValue("longitude", "helios", longitude);
    xmlSetValue("UTC_offset", "helios", UTC_offset);
    xmlSetValue("csv_weather_file", "helios", csv_weather_file);
    xmlSetValue("domain_origin", "helios", domain_origin);
    xmlSetValue("domain_extent", "helios", domain_extent);
    xmlSetValue("ground_resolution", "helios", ground_resolution);
    xmlSetValue("ground_texture_file", "helios", ground_texture_file);
    xmlSetValues("camera_xml_library_file", "helios", camera_xml_library_files);
    xmlSetValues("light_xml_library_file", "helios", light_xml_library_files);
    // CANOPY BLOCK
    canopy_labels = setNodeLabels("label", "canopy_block", labels);
    xmlSetValue("canopy_origin", "canopy_block", canopy_origin);
    xmlSetValue("plant_count", "canopy_block", plant_count);
    xmlSetValue("plant_spacing", "canopy_block", plant_spacing);
    xmlSetValue("plant_library_name", "canopy_block", plant_library_name);
    xmlSetValue("plant_age", "canopy_block", plant_age);
    xmlSetValue("ground_clipping_height", "canopy_block", ground_clipping_height);
    xmlSetValues("canopy_origin", "canopy_block", canopy_origins);
    xmlSetValues("plant_count", "canopy_block", plant_counts);
    xmlSetValues("plant_spacing", "canopy_block", plant_spacings);
    xmlSetValues("plant_library_name", "canopy_block", plant_library_names);
    xmlSetValues("plant_age", "canopy_block", plant_ages);
    xmlSetValues("ground_clipping_height", "canopy_block", ground_clipping_heights);
    // RIG BLOCK
    rig_dict = setNodeLabels("label", "rig", rig_labels);
    // xmlSetValue("camera_position", "rig", camera_position);
    // xmlSetValue("camera_lookat", "rig", camera_lookat);
    xmlSetValue("camera_label", "rig", camera_label);
    // xmlSetValues("camera_position", "rig", camera_positions);
    xmlSetValues("camera_position", "rig", camera_position_vec);
    // xmlSetValues("camera_lookat", "rig", camera_lookats);
    xmlSetValues("camera_lookat", "rig", camera_lookat_vec);
    // xmlSetValues("camera_label", "rig", camera_labels);
    xmlSetValues("camera_label", "rig", rig_camera_labels);
    setKeypoints("keypoint", "camera_position", keypoint_frames);
    xmlSetValues("images", "rig", num_images_vec);
    // CAMERA BLOCK
    camera_dict = setNodeLabels("label", "camera", camera_names);
    xmlSetValue("camera_resolution", "camera", camera_resolution);
    xmlSetValue("focal_plane_distance", "camera", focal_plane_distance);
    xmlSetValue("lens_diameter", "camera", lens_diameter);
    xmlSetValue("FOV_aspect_ratio", "camera", FOV_aspect_ratio);
    xmlSetValue("HFOV", "camera", HFOV);
    xmlSetValues("camera_resolution", "camera", camera_resolutions);
    xmlSetValues("camera_calibration", "camera", camera_calibrations);
    xmlSetValues("focal_plane_distance", "camera", focal_plane_distances);
    xmlSetValues("lens_diameter", "camera", lens_diameters);
    xmlSetValues("FOV_aspect_ratio", "camera", FOV_aspect_ratios);
    xmlSetValues("HFOV", "camera", HFOVs);
    // LIGHT BLOCK
    xmlSetValues("light_type", "light", light_types);
    xmlSetValues("light_spectra", "light", light_spectra);
    xmlSetValues("light_direction", "light", light_direction_vec);
    xmlSetValues("light_rotation", "light", light_rotation_vec);
    xmlSetValues("light_size", "light", light_size_vec);
    xmlSetValues("light_source_flux", "light", light_flux_vec);
    xmlSetValues("light_radius", "light", light_radius_vec);
    light_dict = setNodeLabels("label", "light", light_names);
    xmlSetValues("light_label", "rig", rig_light_labels);
    // RADIATION BLOCK
    xmlSetValue("direct_ray_count", "radiation", direct_ray_count);
    xmlSetValue("diffuse_ray_count", "radiation", diffuse_ray_count);
    xmlSetValue("diffuse_extinction_coeff", "radiation", diffuse_extinction_coeff);
    xmlSetValue("scattering_depth", "radiation", scattering_depth);
    xmlSetValue("air_turbidity", "radiation", air_turbidity);
    xmlSetValues("load_xml_library_file", "radiation", xml_library_files);
    xmlSetValue("solar_direct_spectrum", "radiation", solar_direct_spectrum);
    xmlSetValue("leaf_reflectivity_spectrum", "radiation", leaf_reflectivity_spectrum);
    xmlSetValue("leaf_transmissivity_spectrum", "radiation", leaf_transmissivity_spectrum);
    xmlSetValue("leaf_emissivity", "radiation", leaf_emissivity);
    xmlSetValue("ground_reflectivity_spectrum", "radiation", ground_reflectivity_spectrum);
    xmldoc.save_file(xml_input_file.c_str());
}


void ProjectBuilder::xmlSetValues(std::string xml_path){
    xml_input_file = xml_path;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }
    xmlSetValues();
}


void ProjectBuilder::xmlGetValues(){
    // MAIN BLOCK
    xmlGetValue("latitude", "helios", latitude);
    xmlGetValue("longitude", "helios", longitude);
    xmlGetValue("UTC_offset", "helios", UTC_offset);
    xmlGetValue("csv_weather_file", "helios", csv_weather_file);
    xmlGetValue("domain_origin", "helios", domain_origin);
    xmlGetValue("domain_extent", "helios", domain_extent);
    xmlGetValue("ground_resolution", "helios", ground_resolution);
    xmlGetValue("ground_texture_file", "helios", ground_texture_file);
    xmlGetValues("camera_xml_library_file", "helios", camera_xml_library_files);
    possible_camera_calibrations.clear();
    for (auto &xml_library_file : camera_xml_library_files){
        if( xml_library_file.empty() || !std::filesystem::exists(xml_library_file) ){
            continue;
        }
        std::vector<std::string> current_camera_file = get_xml_node_values(xml_library_file, "label", "globaldata_vec2");
        possible_camera_calibrations.insert(possible_camera_calibrations.end(), current_camera_file.begin(), current_camera_file.end());
    }
    xmlGetValues("light_xml_library_file", "helios", light_xml_library_files);
    possible_light_spectra.clear();
    for (auto &xml_library_file : light_xml_library_files){
        if( xml_library_file.empty() || !std::filesystem::exists(xml_library_file) ){
            continue;
        }
        std::vector<std::string> current_light_file = get_xml_node_values(xml_library_file, "label", "globaldata_vec2");
        possible_light_spectra.insert(possible_light_spectra.end(), current_light_file.begin(), current_light_file.end());
    }
    // CANOPY BLOCK
    labels.clear();
    canopy_labels = getNodeLabels("label", "canopy_block", labels);
    current_canopy = labels[0];
    xmlGetValue("canopy_origin", "canopy_block", canopy_origin);
    xmlGetValue("plant_count", "canopy_block", plant_count);
    xmlGetValue("plant_spacing", "canopy_block", plant_spacing);
    xmlGetValue("plant_library_name", "canopy_block", plant_library_name);
    xmlGetValue("plant_age", "canopy_block", plant_age);
    xmlGetValue("ground_clipping_height", "canopy_block", ground_clipping_height);
    canopy_origins.clear();
    xmlGetValues("canopy_origin", "canopy_block", canopy_origins);
    plant_counts.clear();
    xmlGetValues("plant_count", "canopy_block", plant_counts);
    plant_spacings.clear();
    xmlGetValues("plant_spacing", "canopy_block", plant_spacings);
    plant_library_names.clear();
    xmlGetValues("plant_library_name", "canopy_block", plant_library_names);
    plant_ages.clear();
    xmlGetValues("plant_age", "canopy_block", plant_ages);
    ground_clipping_heights.clear();
    xmlGetValues("ground_clipping_height", "canopy_block", ground_clipping_heights);
    // RIG BLOCK
    rig_labels.clear();
    rig_dict = getNodeLabels("label", "rig", rig_labels);
    current_rig = rig_labels[0];
    xmlGetValue("camera_position", "rig", camera_position);
    xmlGetValue("camera_lookat", "rig", camera_lookat);
    xmlGetValue("camera_label", "rig", camera_label);
    camera_positions.clear();
    xmlGetValues("camera_position", "rig", camera_positions);
    camera_position_vec.clear();
    xmlGetValues("camera_position", "rig", camera_position_vec);
    camera_lookats.clear();
    xmlGetValues("camera_lookat", "rig", camera_lookats);
    camera_lookat_vec.clear();
    xmlGetValues("camera_lookat", "rig", camera_lookat_vec);
    camera_labels.clear();
    xmlGetValues("camera_label", "rig", camera_labels);
    rig_camera_labels.clear();
    xmlGetValues("camera_label", "rig", rig_camera_labels);
    keypoint_frames.clear();
    getKeypoints("keypoint", "camera_position", keypoint_frames);
    current_keypoint = std::to_string(keypoint_frames[0][0]);
    num_images_vec.clear();
    xmlGetValues("images", "rig", num_images_vec);
    // CAMERA BLOCK
    camera_names.clear();
    camera_dict = getNodeLabels("label", "camera", camera_names);
    current_cam = camera_names[0];
    xmlGetValue("camera_resolution", "camera", camera_resolution);
    xmlGetValue("focal_plane_distance", "camera", focal_plane_distance);
    xmlGetValue("lens_diameter", "camera", lens_diameter);
    xmlGetValue("FOV_aspect_ratio", "camera", FOV_aspect_ratio);
    xmlGetValue("HFOV", "camera", HFOV);
    camera_resolutions.clear();
    xmlGetValues("camera_resolution", "camera", camera_resolutions);
    camera_calibrations.clear();
    xmlGetValues("camera_calibration", "camera", camera_calibrations);
    focal_plane_distances.clear();
    xmlGetValues("focal_plane_distance", "camera", focal_plane_distances);
    lens_diameters.clear();
    xmlGetValues("lens_diameter", "camera", lens_diameters);
    FOV_aspect_ratios.clear();
    xmlGetValues("FOV_aspect_ratio", "camera", FOV_aspect_ratios);
    HFOVs.clear();
    xmlGetValues("HFOV", "camera", HFOVs);
    // LIGHT BLOCK
    light_types.clear();
    xmlGetValues("light_type", "light", light_types);
    light_direction_vec.clear();
    xmlGetValues("light_direction", "light", light_direction_vec);
    light_direction_sph_vec.clear();
    for (vec3 vec : light_direction_vec){
        light_direction_sph_vec.push_back(cart2sphere(vec));
    }
    light_rotation_vec.clear();
    xmlGetValues("light_rotation", "light", light_rotation_vec);
    light_size_vec.clear();
    xmlGetValues("light_size", "light", light_size_vec);
    light_flux_vec.clear();
    xmlGetValues("light_source_flux", "light", light_flux_vec);
    light_radius_vec.clear();
    xmlGetValues("light_radius", "light", light_radius_vec);
    light_spectra.clear();
    xmlGetValues("light_spectra", "light", light_spectra);
    light_names.clear();
    light_dict = getNodeLabels("label", "light", light_names);
    current_light = light_names[0];
    rig_light_labels.clear();
    xmlGetValues("light_label", "rig", rig_light_labels);
    // RADIATION BLOCK
    xmlGetValue("direct_ray_count", "radiation", direct_ray_count);
    xmlGetValue("diffuse_ray_count", "radiation", diffuse_ray_count);
    xmlGetValue("diffuse_extinction_coeff", "radiation", diffuse_extinction_coeff);
    xmlGetValue("scattering_depth", "radiation", scattering_depth);
    xmlGetValue("air_turbidity", "radiation", air_turbidity);
    xmlGetValues("load_xml_library_file", "radiation", xml_library_files);
    possible_spectra.clear();
    for (auto &xml_library_file : xml_library_files){
        if( xml_library_file.empty() || !std::filesystem::exists(xml_library_file) ){
            continue;
        }
        std::vector<std::string> current_spectra_file = get_xml_node_values(xml_library_file, "label", "globaldata_vec2");
        possible_spectra.insert(possible_spectra.end(), current_spectra_file.begin(), current_spectra_file.end());
    }
    xmlGetValue("solar_direct_spectrum", "radiation", solar_direct_spectrum);
    xmlGetValue("leaf_reflectivity_spectrum", "radiation", leaf_reflectivity_spectrum);
    xmlGetValue("leaf_transmissivity_spectrum", "radiation", leaf_transmissivity_spectrum);
    xmlGetValue("leaf_emissivity", "radiation", leaf_emissivity);
    xmlGetValue("ground_reflectivity_spectrum", "radiation", ground_reflectivity_spectrum);
    primitive_values.clear();
    for (std::string band : bandlabels){
        primitive_values[band] = {{"ground", {ground_reflectivity, ground_transmissivity, ground_emissivity}},
                                  {"leaf", {leaf_reflectivity, leaf_transmissivity, leaf_emissivity}},
                                  {"petiolule", {petiolule_reflectivity, petiolule_transmissivity, petiolule_emissivity}},
                                  {"petiole", {petiole_reflectivity, petiole_transmissivity, petiole_emissivity}},
                                  {"internode", {internode_reflectivity, internode_transmissivity, internode_emissivity}},
                                  {"peduncle", {peduncle_reflectivity, peduncle_transmissivity, peduncle_emissivity}},
                                  {"petal", {petal_reflectivity, petal_transmissivity, petal_emissivity}},
                                  {"pedicel", {pedicel_reflectivity, pedicel_transmissivity, pedicel_emissivity}},
                                  {"fruit", {fruit_reflectivity, fruit_transmissivity, fruit_emissivity}}};
    }
    primitive_spectra.clear();
    primitive_spectra = {{"All", {reflectivity_spectrum, transmissivity_spectrum, emissivity_spectrum}},
                           {"ground", {ground_reflectivity_spectrum, ground_transmissivity_spectrum, ground_emissivity_spectrum}},
                           {"leaf", {leaf_reflectivity_spectrum, leaf_transmissivity_spectrum, leaf_emissivity_spectrum}},
                           {"petiolule", {petiolule_reflectivity_spectrum, petiolule_transmissivity_spectrum, petiolule_emissivity_spectrum}},
                           {"petiole", {petiole_reflectivity_spectrum, petiole_transmissivity_spectrum, petiole_emissivity_spectrum}},
                           {"internode", {internode_reflectivity_spectrum, internode_transmissivity_spectrum, internode_emissivity_spectrum}},
                           {"peduncle", {peduncle_reflectivity_spectrum, peduncle_transmissivity_spectrum, peduncle_emissivity_spectrum}},
                           {"petal", {petal_reflectivity_spectrum, petal_transmissivity_spectrum, petal_emissivity_spectrum}},
                           {"pedicel", {pedicel_reflectivity_spectrum, pedicel_transmissivity_spectrum, pedicel_emissivity_spectrum}},
                           {"fruit", {fruit_reflectivity_spectrum, fruit_transmissivity_spectrum, fruit_emissivity_spectrum}}};
}

void ProjectBuilder::xmlGetValues(std::string xml_path){
    xml_input_file = xml_path;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }
    xmlGetValues();
}


