#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <iostream>
    #include <commdlg.h>
#elif defined(__APPLE__)
    #include <nfd.h>
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

std::vector<vec3> linspace(const helios::vec3 &a, const helios::vec3 &b, int num_points){
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

std::vector<vec3> interpolate(const std::vector<int> &keypoints, const std::vector<helios::vec3> &positions, int num_points){
    std::vector<vec3> pos = positions;
    std::vector<vec3> result(num_points);
    std::vector<int> keypoints_sorted = keypoints;
    std::sort(keypoints_sorted.begin(), keypoints_sorted.end());
    std::map<int, int> keypoints_loc;
    for (int i = 0; i < keypoints.size(); i++){
        keypoints_loc.insert({keypoints[i], i});
    }
    if (keypoints.size() == 1){
        std::fill(result.begin(), result.end(), pos[0]);
        return result;
    }
    if (keypoints_sorted[keypoints_sorted.size() - 1] != num_points - 1){
        keypoints_sorted.push_back(num_points - 1);
        keypoints_loc.insert({num_points - 1, keypoints.size()});
        pos.push_back(pos[pos.size() - 1]);
    }
    for (int i = 0; i < keypoints_sorted.size() - 1; i++){
        int keypoint = keypoints_sorted[i];
        int keypoint_idx = keypoints_loc[keypoints_sorted[i]];
        int next_keypoint = keypoints_sorted[i + 1];
        int next_keypoint_idx = keypoints_loc[keypoints_sorted[i + 1]];
        std::vector<vec3> curr_positions = linspace(pos[keypoint_idx], pos[next_keypoint_idx], next_keypoint - keypoint + 1);
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
    const float ANIM_SPEED = 0.08f;
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
        ofn.hwndOwner = nullptr;
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        ofn.lpstrFilter = "All Files\0*.*\0Text Files\0*.txt\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFileTitle = nullptr;
        ofn.nMaxFileTitle = 0;
        ofn.lpstrInitialDir = nullptr;
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
        nfdchar_t *outPath = nullptr;
        nfdresult_t result = NFD_OpenDialog( nullptr, nullptr, &outPath );

        if ( result == NFD_OKAY ) {
            //puts("Success!");
            //puts(outPath);
            file_name = std::string(outPath);
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
        nfdchar_t *outPath = nullptr;
        nfdresult_t result = NFD_OpenDialog( nullptr, nullptr, &outPath );

        if ( result == NFD_OKAY ) {
            puts("Success!");
            puts(outPath);
            file_name = std::string(outPath);
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


std::string save_as_file_dialog(std::vector<std::string> extensions){
    std::string file_name;
    #ifdef _WIN32
        // save CWD
        char CWD[MAX_PATH];
        GetCurrentDirectory(MAX_PATH, CWD);

        OPENFILENAME ofn;
        char szFile[260] = {0};

        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = nullptr;
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        std::string filterList = "";
        for (std::string extension : extensions){
            std::string ext_lower = extension;
            std::string ext_upper = extension;
            std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
            std::transform(ext_upper.begin(), ext_upper.end(), ext_upper.begin(), ::toupper);
            filterList += ext_upper + " Files (*." + ext_lower + ")";
            filterList += '\0';
            filterList += "*." + ext_lower;
            filterList += '\0';
        }
        filterList += '\0';
        ofn.lpstrFilter = filterList.c_str();
        ofn.nFilterIndex = 1;
        ofn.lpstrFileTitle = nullptr;
        ofn.nMaxFileTitle = 0;
        ofn.lpstrInitialDir = nullptr;
        ofn.Flags = OFN_PATHMUSTEXIST;

        if (GetSaveFileName(&ofn)) {
            std::cout << "Selected file: " << ofn.lpstrFile << std::endl;
        } else {
            std::cout << "No file selected." << std::endl;
            return "";
        }

        std::string ext_ = extensions[ofn.nFilterIndex - 1];
        std::transform(ext_.begin(), ext_.end(), ext_.begin(), ::tolower);
        std::string ext = "." + ext_;

        // correct CWD
        SetCurrentDirectory(CWD);

        file_name = (std::string)ofn.lpstrFile;

        std::filesystem::path file_path(file_name);
        if (file_path.extension().empty()) {
            file_path.replace_extension(ext);
            file_name = file_path.string();
        }
    #elif defined(__APPLE__)
        nfdchar_t *outPath = nullptr;
        std::string filterList_ = "";
        for (std::string extension : extensions){
            std::string ext_lower = extension;
            std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
            filterList_ += ext_lower + ",";
        }
        if (!filterList_.empty() && filterList_.back() == ',') filterList_.pop_back();
        const nfdchar_t *filterList = filterList_.c_str();
        nfdresult_t result = NFD_SaveDialog( filterList, nullptr, &outPath );

        if ( result == NFD_OKAY ) {
            puts("Success!");
            puts(outPath);
            file_name = std::string(outPath);
            free(outPath);

            std::string ext_ = extensions[0];
            std::transform(ext_.begin(), ext_.end(), ext_.begin(), ::tolower);
            std::string ext = "." + ext_;
            if (file_name.find('.') == std::string::npos) file_name += ext;
        }
        else if ( result == NFD_CANCEL ) {
            puts("User pressed cancel.");
        }
        else {
            // printf("Error: %s\n", NFD_GetError() );
            std::cout << "Error: " << NFD_GetError() << std::endl;
        }
    #elif defined(__linux__)
        nfdchar_t *outPath = nullptr;
        std::string filterList_ = "";
        for (std::string extension : extensions){
            std::string ext_lower = extension;
            std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
            filterList_ += ext_lower + ",";
        }
        if (!filterList_.empty() && filterList_.back() == ',') filterList_.pop_back();
        const nfdchar_t *filterList = filterList_.c_str();
        nfdresult_t result = NFD_SaveDialog( filterList, nullptr, &outPath );

        if ( result == NFD_OKAY ) {
            puts("Success!");
            puts(outPath);
            file_name = std::string(outPath);
            free(outPath);

            std::string ext_ = extensions[0];
            std::transform(ext_.begin(), ext_.end(), ext_.begin(), ::tolower);
            std::string ext = "." + ext_;
            if (file_name.find('.') == std::string::npos) file_name += ext;
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


std::vector<std::string> get_xml_node_values(const std::string &xml_input, const std::string& name,
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
    for (auto& arrow_vec : arrow_dict){
        for (std::vector<uint> arrow : arrow_vec.second){
            context->deletePrimitive(arrow);
        }
    }
    arrow_dict.clear();
}


void ProjectBuilder::updateArrows(){
    #ifdef ENABLE_RADIATION_MODEL
    for (auto rig_label_ : rig_labels_set){
        arrow_count = 0;
        std::string current_rig = rig_label_;
        arrow_dict[current_rig] = std::vector<std::vector<uint>>{};
        for (int i = 1; i < camera_position_vec[rig_dict[current_rig]].size(); i++){
            vec3 arrow_pos = camera_position_vec[rig_dict[current_rig]][i - 1];
            vec3 arrow_direction_vec = arrow_pos - camera_position_vec[rig_dict[current_rig]][i];
            SphericalCoord arrow_direction_sph = cart2sphere(arrow_direction_vec);
            vec3 arrow_scale(0.35, 0.35, 0.35);
            arrow_dict[current_rig].push_back(context->loadOBJ("plugins/radiation/camera_light_models/Arrow.obj",
                                            nullorigin, arrow_scale, nullrotation, RGB::blue, "YUP", true));
            context->rotatePrimitive(arrow_dict.at(current_rig)[arrow_count], arrow_direction_sph.elevation, "x");
            context->rotatePrimitive(arrow_dict.at(current_rig)[arrow_count], -arrow_direction_sph.azimuth, "z");
            context->translatePrimitive(arrow_dict.at(current_rig)[arrow_count], arrow_pos);
            context->setPrimitiveData(arrow_dict.at(current_rig)[arrow_count], "twosided_flag", uint(3));
            arrow_count++;
        }
        float col[3];
        RGBcolor rig_color = rig_colors[rig_dict[current_rig]];
        col[0] = rig_color.r;
        col[1] = rig_color.g;
        col[2] = rig_color.b;
        updateColor(current_rig, "arrow", col);
    }
    #endif
}


void ProjectBuilder::deleteCameraModels(){
    for (auto& camera_model : camera_models_dict){
        context->deletePrimitive(camera_model.second);
    }
    camera_models_dict.clear();
}


void ProjectBuilder::updateCameraModels(){
#ifdef ENABLE_RADIATION_MODEL
    for (auto rig_label_ : rig_labels_set){
        int camera_idx = rig_dict[rig_label_];
        camera_models_dict[rig_label_] = std::vector<uint>{};
        vec3 camera_pos = camera_position_vec[camera_idx][0];
        vec3 camera_direction_vec = camera_lookat_vec[camera_idx][0] - camera_pos;
        SphericalCoord camera_direction_sph = cart2sphere(camera_direction_vec);
        vec3 camera_scale(1.0, 1.0, 1.0);
        camera_models_dict[rig_label_] = context->loadOBJ("plugins/radiation/camera_light_models/Camera.obj",
                                        nullorigin, camera_scale, nullrotation, RGB::blue, "ZUP", true);
        context->rotatePrimitive(camera_models_dict.at(rig_label_), camera_direction_sph.elevation, "x");
        context->rotatePrimitive(camera_models_dict.at(rig_label_), -camera_direction_sph.azimuth, "z");
        context->translatePrimitive(camera_models_dict.at(rig_label_), camera_pos);
        context->setPrimitiveData(camera_models_dict.at(rig_label_), "twosided_flag", uint(3));
        float col[3];
        RGBcolor rig_color = rig_colors[camera_idx];
        col[0] = rig_color.r;
        col[1] = rig_color.g;
        col[2] = rig_color.b;
        updateColor(rig_label_, "camera", col);
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
    for (auto& box_pair : bounding_boxes){
        if (primitive_names_set.find(box_pair.first) == primitive_names_set.end()){
            bounding_boxes.erase(box_pair.first);
        }
    }
    for (auto& prim : primitive_names_set){
        if (prim == "All"){
            continue;
        }
        if (bounding_boxes.find(prim) == bounding_boxes.end()){
            bounding_boxes[prim] = false;
        }
    }
    //context->setPrimitiveData
    // context->setPrimitiveData(); type uint or int
}


void ProjectBuilder::updateDataGroups(){
    data_groups_set.clear();
    data_groups_set.insert("All");
    std::vector<uint> all_UUIDs = context->getAllUUIDs();
    for (uint UUID : all_UUIDs){
        std::string curr_data_group = "";
        if (!context->doesPrimitiveDataExist(UUID, "data_group")) continue;
        context->getPrimitiveData(UUID, "data_group", curr_data_group);
        if (!curr_data_group.empty()){
            data_groups_set.insert(curr_data_group);
        }
    }
    for (std::string data_group : data_groups_set){ //initialize new data_groups if necessary
        if (primitive_continuous_dict.find(data_group) == primitive_continuous_dict.end()){
            primitive_continuous_dict[data_group] = primitive_continuous;
        }
        if (primitive_spectra_dict.find(data_group) == primitive_spectra_dict.end()){
            primitive_spectra_dict[data_group] = primitive_spectra;
        }
        if (primitive_values_dict.find(data_group) == primitive_values_dict.end()){
            primitive_values_dict[data_group] = primitive_values;
        }
    }
}



void ProjectBuilder::updateSpectra(){
    for (std::pair<std::string, std::vector<uint>> primitive_pair : primitive_UUIDs){
        if (primitive_continuous[primitive_pair.first].empty()) continue;
        if (!primitive_continuous[primitive_pair.first][0]){
            for (std::string band : bandlabels){
                float reflectivity = primitive_values[band][primitive_pair.first][0];
                std::string reflectivity_band = "reflectivity_" + band;
                context->setPrimitiveData(primitive_pair.second, reflectivity_band.c_str(), reflectivity);
            }
        }else{
            std::string reflectivity_spectrum = primitive_spectra[primitive_pair.first][0];
            if( !reflectivity_spectrum.empty() ){
                context->setPrimitiveData( primitive_UUIDs[primitive_pair.first], "reflectivity_spectrum", reflectivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << primitive_pair.first << "_reflectivity_spectrum'. Assuming " << primitive_pair.first << " primitives are black across all shortwave bands." << std::endl;
            }
        }
        if (!primitive_continuous[primitive_pair.first][1]){
            for (std::string band : bandlabels){
                float transmissivity = primitive_values[band][primitive_pair.first][1];
                std::string transmissivity_band = "transmissivity_" + band;
                context->setPrimitiveData(primitive_pair.second, transmissivity_band.c_str(), transmissivity);
            }
        }else{
            std::string transmissivity_spectrum = primitive_spectra[primitive_pair.first][1];
            if( !transmissivity_spectrum.empty() ){
                context->setPrimitiveData( primitive_UUIDs[primitive_pair.first], "transmissivity_spectrum", transmissivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << primitive_pair.first << "_transmissivity_spectrum'. Assuming " << primitive_pair.first << " primitives are black across all shortwave bands." << std::endl;
            }
        }
        if (!primitive_continuous[primitive_pair.first][2]){
            for (std::string band : bandlabels){
                if (bandlabels_set_emissivity.find(band) != bandlabels_set_emissivity.end()) continue;
                float emissivity = primitive_values[band][primitive_pair.first][2];
                std::string emissivity_band = "emissivity_" + band;
                context->setPrimitiveData(primitive_pair.second, emissivity_band.c_str(), emissivity);
            }
        }else{
            std::string emissivity_spectrum = primitive_spectra[primitive_pair.first][2];
            if( !emissivity_spectrum.empty() ){
                context->setPrimitiveData( primitive_UUIDs[primitive_pair.first], "emissivity_spectrum", emissivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << primitive_pair.first << "_emissivity_spectrum'. Assuming " << primitive_pair.first << " primitives are black across all shortwave bands." << std::endl;
            }
        }
    }
    std::vector<uint> all_UUIDs = context->getAllUUIDs();
    for (uint UUID : all_UUIDs){
        if (!context->doesPrimitiveDataExist(UUID, "data_group")) continue;
        // Get data group of UUID
        std::string curr_data_group;
        context->getPrimitiveData(UUID, "data_group", curr_data_group);
        // Get primitive type of UUID
        std::string curr_prim_type;
        if ( context->doesPrimitiveDataExist(UUID, "object_label") ){
            context->getPrimitiveData(UUID, "object_label", curr_prim_type);
        } else{
            curr_prim_type = "All";
        }
        if (primitive_continuous_dict[curr_data_group][curr_prim_type].empty()) continue;
        if (!primitive_continuous_dict[curr_data_group][curr_prim_type][0]){
            for (std::string band : bandlabels){
                float reflectivity = primitive_values_dict[curr_data_group][band][curr_prim_type][0];
                std::string reflectivity_band = "reflectivity_" + band;
                context->setPrimitiveData(UUID, reflectivity_band.c_str(), reflectivity);
            }
        }else{
            std::string reflectivity_spectrum = primitive_spectra_dict[curr_data_group][curr_prim_type][0];
            if( !reflectivity_spectrum.empty() ){
                context->setPrimitiveData( UUID, "reflectivity_spectrum", reflectivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << curr_prim_type << "_reflectivity_spectrum'. Assuming " << curr_prim_type << " primitives are black across all shortwave bands." << std::endl;
            }
        }
        if (!primitive_continuous_dict[curr_data_group][curr_prim_type][1]){
            for (std::string band : bandlabels){
                float transmissivity = primitive_values_dict[curr_data_group][band][curr_prim_type][1];
                std::string transmissivity_band = "transmissivity_" + band;
                context->setPrimitiveData(UUID, transmissivity_band.c_str(), transmissivity);
            }
        }else{
            std::string transmissivity_spectrum = primitive_spectra_dict[curr_data_group][curr_prim_type][1];
            if( !transmissivity_spectrum.empty() ){
                context->setPrimitiveData( UUID, "transmissivity_spectrum", transmissivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << curr_prim_type << "_transmissivity_spectrum'. Assuming " << curr_prim_type << " primitives are black across all shortwave bands." << std::endl;
            }
        }
        if (!primitive_continuous_dict[curr_data_group][curr_prim_type][2]){
            for (std::string band : bandlabels){
                if (bandlabels_set_emissivity.find(band) != bandlabels_set_emissivity.end()) continue;
                float emissivity = primitive_values_dict[curr_data_group][band][curr_prim_type][2];
                std::string emissivity_band = "emissivity_" + band;
                context->setPrimitiveData(UUID, emissivity_band.c_str(), emissivity);
            }
        }else{
            std::string emissivity_spectrum = primitive_spectra_dict[curr_data_group][curr_prim_type][2];
            if( !emissivity_spectrum.empty() ){
                context->setPrimitiveData( UUID, "emissivity_spectrum", emissivity_spectrum );
            }else{
                std::cout << "WARNING: No value given for '" << curr_prim_type << "_emissivity_spectrum'. Assuming " << curr_prim_type << " primitives are black across all shortwave bands." << std::endl;
            }
        }
    }
}

void ProjectBuilder::updateCameras(){
    #ifdef ENABLE_RADIATION_MODEL
    for (std::string rig_label : rig_labels_set){
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
                radiation->setCameraSpectralResponse(camera_label_, band, camera_calibrations[camera_index][band]);
            }
            radiation->updateGeometry();
        }
    }
    for (std::string band_group_name : band_group_names){
        bandGroup curr_band_group = band_group_lookup[band_group_name];
        if (!curr_band_group.grayscale){
            radiation->runBand(curr_band_group.bands);
        } else{
            radiation->runBand(std::vector<std::string>{curr_band_group.bands[0]});
        }
    }
    #endif //RADIATION_MODEL
}

void ProjectBuilder::record(){
    #ifdef ENABLE_RADIATION_MODEL
    deleteArrows();
    deleteCameraModels();
    randomize(true);
    setBoundingBoxObjects();
    for (int _ = 0; _ < num_recordings; _++){
        updateContext();
        // std::time_t now = std::time(nullptr);
        // std::tm* tm_ptr = std::localtime(&now);
        // std::ostringstream oss;
        // oss << std::put_time(tm_ptr, "%Y-%m-%d %H:%M:%S");
        // std::string image_dir = "./saved-" + oss.str() + "/";
        // std::filesystem::create_directory("saved-" + oss.str());
        std::string image_dir = "./saved/";
        std::string image_dir_base = "./saved_";
        int image_dir_idx = 0;
        while (std::filesystem::exists(image_dir)){
            image_dir = image_dir_base + std::to_string(image_dir_idx) + "/";
            image_dir_idx++;
        }
        std::filesystem::create_directory(image_dir);
        std::vector<uint> temp_lights{};
        for (std::string rig_label : rig_labels_set){
            int rig_index = rig_dict[rig_label];
            std::vector<vec3> interpolated_camera_positions = interpolate(keypoint_frames[rig_index], camera_position_vec[rig_index], num_images_vec[rig_index]);
            std::vector<vec3> interpolated_camera_lookats = interpolate(keypoint_frames[rig_index], camera_lookat_vec[rig_index], num_images_vec[rig_index]);
            // ADD RIG LIGHTS
            for (std::string light : rig_light_labels[rig_dict[rig_label]]){
                int light_idx = light_dict[light];
                uint new_light_UUID;
                if (light_types[light_idx] == "sphere"){
                    new_light_UUID = radiation->addSphereRadiationSource(interpolated_camera_positions[0], light_radius_vec[light_idx]);
                    temp_lights.push_back(new_light_UUID);
                }else if (light_types[light_dict[light]] == "rectangle"){
                    new_light_UUID = radiation->addRectangleRadiationSource(interpolated_camera_positions[0],
                        light_size_vec[light_idx], light_rotation_vec[light_idx]);
                    temp_lights.push_back(new_light_UUID);
                }else if (light_types[light_dict[light]] == "disk"){
                    new_light_UUID = radiation->addDiskRadiationSource(interpolated_camera_positions[0],
                        light_radius_vec[light_idx], light_rotation_vec[light_idx]);
                    temp_lights.push_back(new_light_UUID);
                }
                for (auto &band : bandlabels){
                    radiation->setSourceFlux(new_light_UUID, band, light_flux_vec[light_idx]);
                }
            }
            // radiation->updateGeometry(); // TODO: figure out why we can't move updateGeometry here
            // radiation->setSourceFlux(light_UUID, band, flux_value)
            //
            for (int i = 0; i < interpolated_camera_positions.size(); i++){
                // SET LIGHT POSITIONS
                for (uint light_ID : temp_lights){
                    radiation->setSourcePosition(light_ID, interpolated_camera_positions[i]);
                }
                // radiation->setSourceFlux(light_UUID, band, flux_value)
                //
                for (std::string rig_camera_label : rig_camera_labels[rig_index]){
                    int camera_index = camera_dict[rig_camera_label];
                    std::string cameralabel = rig_label + "_" + rig_camera_label;
                    radiation->setCameraPosition(cameralabel, interpolated_camera_positions[i]);
                    radiation->setCameraLookat(cameralabel, interpolated_camera_lookats[i]);
                }
                radiation->updateGeometry();
                for (std::string band_group_name : band_group_names){
                    bandGroup curr_band_group = band_group_lookup[band_group_name];
                    if (!curr_band_group.grayscale){
                        radiation->runBand(curr_band_group.bands);
                    } else{
                        radiation->runBand(std::vector<std::string>{curr_band_group.bands[0]});
                    }
                }
                for (std::string rig_camera_label : rig_camera_labels[rig_index]){
                    std::string cameralabel = rig_label + "_" + rig_camera_label;
                    // Write Images
                    for (std::string band_group_name : band_group_names){
                        bandGroup curr_band_group = band_group_lookup[band_group_name];
                        std::vector<std::string> band_group_vec;
                        if (!curr_band_group.grayscale){
                            band_group_vec = curr_band_group.bands;
                        } else{
                            band_group_vec = std::vector<std::string>{curr_band_group.bands[0]};
                        }
                        radiation->writeCameraImage( cameralabel, band_group_vec, band_group_name + std::to_string(i), image_dir + rig_label + '/');
                        if (band_group_lookup[band_group_name].norm){
                            radiation->writeNormCameraImage( cameralabel, band_group_vec, band_group_name + "_norm" + std::to_string(i), image_dir + rig_label + '/');
                        }
                    }
                    if (write_depth[rig_dict[current_rig]])
                    radiation->writeDepthImageData( cameralabel, "depth" + std::to_string(i), image_dir + rig_label + '/');
                    if (write_norm_depth[rig_dict[current_rig]])
                    radiation->writeNormDepthImage( cameralabel, "normdepth" + std::to_string(i), 3, image_dir + rig_label + '/');
                    //
                    // Bounding boxes for all primitive types
                    for (std::string primitive_name : primitive_names){
                        if (!primitive_name.empty()){
                            primitive_name[0] = std::tolower(static_cast<unsigned char>(primitive_name[0]));
                        }
                        if (bounding_boxes_map.find(primitive_name) != bounding_boxes_map.end())
                            radiation->writeImageBoundingBoxes( cameralabel, "object_number", bounding_boxes_map[primitive_name], "bbox_" + primitive_name + std::to_string(i), image_dir + rig_label + '/');
                        // radiation->writeImageBoundingBoxes( cameralabel, primitive_name, 0, "bbox_" + primitive_name + std::to_string(i), image_dir + rig_label + '/');
                        // radiation->writeImageBoundingBoxes_ObjectData();
                    }
                    //
                }
            }
            // REMOVE RIG LIGHTS
            for (uint temp_light : temp_lights){
                radiation->deleteRadiationSource(temp_light);
            }
            temp_lights.clear();
            //
        }
        randomize(false);
    }
    updateArrows();
    updateCameraModels();
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
        BuildGeometry(xml_input_file, plantarchitecture, context, canopy_IDs, individual_plant_locations);
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

    #ifdef ENABLE_CANOPY_GENERATOR
        canopygenerator = new CanopyGenerator(context);
        std::cout << "Loaded CanopyGenerator plugin." << std::endl;
    // }else{
    #else
        std::cout << "Excluding CanopyGenerator plugin." << std::endl;
    // } //CANOPYGENERATOR
    #endif //CANOPYGENERATOR

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
                bounding_boxes[primitive_name] = false;
                std::string primitive_name_lower = primitive_name;
                primitive_name_lower[0] = std::tolower(static_cast<unsigned char>(primitive_name_lower[0]));
                std::string primitive_UUIDs_name = primitive_name_lower + "_UUIDs";
                if ( context->doesGlobalDataExist( primitive_UUIDs_name.c_str() ) ){
                    context->getGlobalData( primitive_UUIDs_name.c_str(), primitive_UUIDs[primitive_name] );
                    std::vector<uint> primitive_UUIDs_ = primitive_UUIDs[primitive_name];
                    if ( !primitive_UUIDs_.empty()){
                        context->setPrimitiveData(primitive_UUIDs[primitive_name], "object_label", primitive_name_lower);
                    }
                }
            }
        }
        ground_UUIDs = primitive_UUIDs["ground"];
        leaf_UUIDs = primitive_UUIDs["leaf"];
        // assert( !ground_UUIDs.empty() );
        // assert( !leaf_UUIDs.empty() );
        context->setPrimitiveData(ground_UUIDs, "object_label", "ground");
        context->setPrimitiveData(leaf_UUIDs, "object_label", "leaf");

    }

    // Update reflectivity, transmissivity, & emissivity for each band / primitive_type
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }
    xmlGetValues();
    updateGround(); // TODO: add repeat ground to buildGeometry
    updateSpectra(); // TODO: add update geometry at end
    radiation->updateGeometry();

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
            LW_absorbed /= ground_area;

            std::cout << "Absorbed PAR: " << PAR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed NIR: " << NIR_absorbed << " W/m^2" << std::endl;
            std::cout << "Absorbed LW: " << LW_absorbed << " W/m^2" << std::endl;
        }
        // OBJ BLOCK
        for (int i = 0; i < obj_files.size(); i++){
            object new_object;
            std::vector<uint> new_UUIDs;
            std::string new_obj_file = obj_files[i];
            if( std::filesystem::path(new_obj_file).extension() == ".obj" ){
                new_UUIDs = context->loadOBJ(new_obj_file.c_str());
            } else if ( std::filesystem::path(new_obj_file).extension() == ".ply" ){
                new_UUIDs = context->loadPLY(new_obj_file.c_str());
            } else {
                std::cout << "Failed to load object file " << new_obj_file << "." << std::endl;
            }
            // check for MTL file
            std::filesystem::path mtl_path(new_obj_file);
            mtl_path.replace_extension("mtl");
            if (std::filesystem::exists(mtl_path)){
                new_object.use_texture_file = true;
            } else{
                new_object.use_texture_file = false;
                context->setPrimitiveColor(new_UUIDs, obj_colors[i]);
            }
            context->scalePrimitive(new_UUIDs, obj_scales[i]);
            context->rotatePrimitive(new_UUIDs, deg2rad(obj_orientations[i].x), "x");
            context->rotatePrimitive(new_UUIDs, deg2rad(obj_orientations[i].y), "y");
            context->rotatePrimitive(new_UUIDs, deg2rad(obj_orientations[i].z), "z");
            context->translatePrimitive(new_UUIDs, obj_positions[i]);
            obj_UUIDs.push_back(new_UUIDs);
            new_object.index = obj_idx;
            obj_idx++;
            new_object.name = obj_names[i];
            new_object.file = obj_files[i];
            new_object.data_group = obj_data_groups[i];
            new_object.UUIDs = new_UUIDs;
            new_object.position = obj_positions[i];
            new_object.prev_position = obj_positions[i];
            new_object.orientation = obj_orientations[i];
            new_object.prev_orientation = obj_orientations[i];
            new_object.scale = obj_scales[i];
            new_object.prev_scale = obj_scales[i];
            new_object.color = obj_colors[i];
            new_object.prev_color = obj_colors[i];
            new_object.is_dirty = false;
            objects_dict[new_object.name] = new_object;
        }
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

            radiation->enforcePeriodicBoundary("xy"); // TODO: make this a user setting
        // } //SOLARPOSITION && RADIATION_MODEL
        #endif //SOLARPOSITION && RADIATION_MODEL
    }
    // RIG BLOCK
    num_images = 5;
    updateArrows();
    updateCameras();
    updateCameraModels();

    helios = xmldoc.child("helios");

    refreshVisualizationTypes();

    built = true;
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


void ProjectBuilder::xmlGetDistribution(const std::string& name, const std::string& parent, distribution& distribution) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if( node.empty() ){
        std::cout << "WARNING: No distribution given for '" << name << "'. Using default distribution of N/A." << std::endl;
        distribution.flag = -1;
    }else {
        const char *node_str = node.child_value();
        if (!parse_distribution(node_str, distribution)) {
            helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
        }
    }
}


void ProjectBuilder::xmlSetDistribution(const std::string& name, const std::string& parent, distribution& distribution) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if (!node){
        node = p.append_child(name.c_str());
    }
    std::string dist_type;
    std::string param_1;
    std::string param_2;
    std::string repeat;
    if (distribution.flag == 0){
        dist_type = "normal";
        param_1 = std::to_string(distribution.dist.normal->mean());
        param_2 = std::to_string(distribution.dist.normal->stddev());
        repeat = std::to_string(distribution.repeat);
    } else if (distribution.flag == 1){
        dist_type = "uniform";
        param_1 = std::to_string(distribution.dist.uniform->a());
        param_2 = std::to_string(distribution.dist.uniform->b());
        repeat = std::to_string(distribution.repeat);
    } else if (distribution.flag == 2){
        dist_type = "weibull";
        param_1 = std::to_string(distribution.dist.weibull->a());
        param_2 = std::to_string(distribution.dist.weibull->b());
        repeat = std::to_string(distribution.repeat);
    } else if (distribution.flag == -1){
        dist_type = "N/A";
        param_1 = "0";
        param_2 = "0";
        repeat = "0";
    }
    node.text().set((dist_type + " " + param_1 + " " + param_2 + " " + repeat).c_str());
}


bool parse_distribution( const std::string &input_string, distribution &converted_distribution ){
    std::istringstream vecstream(input_string);
    std::vector<std::string> tmp_s(4);
    vecstream >> tmp_s[0];
    vecstream >> tmp_s[1];
    vecstream >> tmp_s[2];
    vecstream >> tmp_s[3];
    std::string dist_type;
    std::vector<float> dist_params = {0.0, 0.0};
    int repeat = 0;
    if (!parse_float(tmp_s[1], dist_params[0]) || !parse_float(tmp_s[2], dist_params[1]) || !parse_int(tmp_s[3], repeat) ) {
        return false;
    }
    distUnion dist{};
    if (dist_type == "normal"){
        dist.normal = new std::normal_distribution<float>;
        *dist.normal = std::normal_distribution<float>(dist_params[0], dist_params[1]);
        converted_distribution.dist = dist;
        converted_distribution.flag = 0;
        converted_distribution.repeat = (bool) repeat;
    } else if (dist_type == "uniform"){
        dist.uniform = new std::uniform_real_distribution<float>;
        *dist.uniform = std::uniform_real_distribution<float>(dist_params[0], dist_params[1]);
        converted_distribution.dist = dist;
        converted_distribution.flag = 1;
        converted_distribution.repeat = (bool) repeat;
    } else if (dist_type == "weibull"){
        dist.weibull = new std::weibull_distribution<float>;
        *dist.weibull = std::weibull_distribution<float>(dist_params[0], dist_params[1]);
        converted_distribution.dist = dist;
        converted_distribution.flag = 2;
        converted_distribution.repeat = (bool) repeat;
    } else{
        converted_distribution.flag = -1;
    }
    return true;
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


void ProjectBuilder::xmlGetValues(const std::string& name, const std::string& parent, std::vector<helios::RGBcolor>& default_vec){
    helios = xmldoc.child("helios");
    pugi::xml_node node;
    for (pugi::xml_node p = helios.child(parent.c_str()); p; p = p.next_sibling(parent.c_str())){
        node = p.child(name.c_str());
        if( node.empty() ){
            std::cout << "WARNING: No value given for '" << name << "'.";
        } else {
            const char *node_str = node.child_value();
            helios::RGBcolor default_value;
            if (!parse_RGBcolor(node_str, default_value)) {
                helios_runtime_error("ERROR: Value given for '" + name + "' could not be parsed.");
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


void ProjectBuilder::setNodeLabels(const std::string& name, const std::string& parent, std::set<std::string>& labels_vec){
    int i = 0;
    helios = xmldoc.child("helios");

    std::set<pugi::xml_node> curr_nodes_set;
    for (pugi::xml_node node = helios.child(parent.c_str()); node; node = node.next_sibling(parent.c_str())){
        curr_nodes_set.insert(node);
    }
    for (auto node : curr_nodes_set){
        node.parent().remove_child(node);
    }

    std::map<std::string, int> labels_dict = {};
    for (auto new_node_label : labels_vec){
        pugi::xml_node new_node = helios.append_child(parent.c_str());
        pugi::xml_attribute node_label = new_node.attribute(name.c_str());
        node_label.set_value(new_node_label.c_str());
        new_node.append_attribute(name.c_str()).set_value(new_node_label.c_str());
    }

    return;
}


void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, int &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if (!node){
        node = p.append_child(name.c_str());
    }
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
    if (!node){
        node = p.append_child(name.c_str());
    }
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
    if (!node){
        node = p.append_child(name.c_str());
    }
    node.text().set(default_value.c_str());
}

void ProjectBuilder::xmlRemoveField(const std::string& name, const std::string& parent) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if (node){
        p.remove_child(name.c_str());
    }
}

void ProjectBuilder::xmlSetValue(const std::string& name, const std::string& parent, int2 &default_value) {
    helios = xmldoc.child("helios");
    pugi::xml_node p = helios;
    if (parent != "helios") {
        p = helios.child(parent.c_str());
    }
    pugi::xml_node node;
    node = p.child(name.c_str());
    if (!node){
        node = p.append_child(name.c_str());
    }
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
    if (!node){
        node = p.append_child(name.c_str());
    }
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
    if (!node){
        node = p.append_child(name.c_str());
    }
    node.text().set(vec_to_string(default_value).c_str());
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<vec2>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        pugi::xml_node node = p.child(field_name.c_str());
        if (!node){
            node = p.append_child(field_name.c_str());
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        node.text().set(vec_to_string(values_vec[idx]).c_str());
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<vec3>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        pugi::xml_node node = p.child(field_name.c_str());
        if (!node){
            node = p.append_child(field_name.c_str());
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        node.text().set(vec_to_string(values_vec[idx]).c_str());
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<helios::RGBcolor>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        pugi::xml_node node = p.child(field_name.c_str());
        if (!node){
            node = p.append_child(field_name.c_str());
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        helios::vec3 default_values_vec;
        default_values_vec.x = values_vec[idx].r;
        default_values_vec.y = values_vec[idx].g;
        default_values_vec.z = values_vec[idx].b;
        node.text().set(vec_to_string(default_values_vec).c_str());
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<int2>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        pugi::xml_node node = p.child(field_name.c_str());
        if (!node){
            node = p.append_child(field_name.c_str());
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        node.text().set(vec_to_string(values_vec[idx]).c_str());
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<std::string>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        pugi::xml_node node = p.child(field_name.c_str());
        if (!node){
            node = p.append_child(field_name.c_str());
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        node.text().set(values_vec[idx].c_str());
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<int>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        pugi::xml_node node = p.child(field_name.c_str());
        if (!node){
            node = p.append_child(field_name.c_str());
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        node.text().set(std::to_string(values_vec[idx]).c_str());
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<float>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        pugi::xml_node node = p.child(field_name.c_str());
        if (!node){
            node = p.append_child(field_name.c_str());
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        node.text().set(std::to_string(values_vec[idx]).c_str());
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<std::vector<vec3>>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");

    for (pugi::xml_node p = helios.child(node_name.c_str()); p; p = p.next_sibling(node_name.c_str())){
        std::vector<pugi::xml_node> remove = {};
        for (pugi::xml_node node = p.child(field_name.c_str()); node; node = node.next_sibling(field_name.c_str())){
            remove.push_back(node);
        }
        for (pugi::xml_node &node : remove){
            p.remove_child(node);
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        for (int j = 0; j < values_vec[idx].size(); j++){
            // p.append_child(name.c_str()).set_value(vec_to_string(default_vec[i][j]).c_str());
            pugi::xml_node new_node = p.append_child(field_name.c_str());
            new_node.text().set(vec_to_string(values_vec[idx][j]).c_str());
        }
    }
}


void ProjectBuilder::xmlSetValues(const std::string& field_name, const std::string& node_name, std::vector<std::set<std::string>>& values_vec, std::map<std::string, int>& node_map){
    helios = xmldoc.child("helios");
    int i = 0;
    pugi::xml_node p_;
    if (node_name != "helios") {
        p_ = helios.child(node_name.c_str());
    }else{
        p_ = helios;
    }
    for (pugi::xml_node p = p_; p; p = p.next_sibling(node_name.c_str())){
        std::vector<pugi::xml_node> remove{};
        for (pugi::xml_node node = p.child(field_name.c_str()); node; node = node.next_sibling(field_name.c_str())){
            remove.push_back(node);
        }
        for (pugi::xml_node &node : remove){
            p.remove_child(node);
        }
        pugi::xml_attribute label = p.attribute("label");
        int idx = node_map[label.as_string()];
        for (std::string s : values_vec[idx]){
            // p.append_child(name.c_str()).set_value(s.c_str());
            pugi::xml_node new_node = p.append_child(field_name.c_str());
            new_node.text().set(s.c_str());
        }
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
    if (!built){
        std::cout << "Project must be built before running visualize." << std::endl;
        return;
    }
    // if (enable_visualizer){
    #ifdef ENABLE_HELIOS_VISUALIZER
        visualizer = new Visualizer(800);
        // #ifdef ENABLE_RADIATION_MODEL
        // radiation->enableCameraModelVisualization();
        // #endif //RADIATION_MODEL
        visualizer->buildContextGeometry(context);

        // Uncomment below for interactive
        // visualizer.plotInteractive();

        visualizer->addCoordinateAxes(helios::make_vec3(0,0,0.05), helios::make_vec3(1,1,1), "positive");
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

        // GLFWwindow* window = glfwCreateWindow(640, 480, "My Title", nullptr, nullptr);
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
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)){
                currently_dragging = "";
                disable_dragging = true;
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)){
                if (!currently_dragging.empty()){
                    // ImVec2 mouse_pos = ImGui::GetMousePos();
                    // float ndcX = (2.0f * mouse_pos.x / windowSize.x) - 1.0f;
                    // float ndcY = 1.0f - (2.0f * mouse_pos.y / windowSize.y);
                    // std::vector<vec3> camera_lookat_pos = visualizer->getCameraPosition();
                    // vec3 camera_pos = camera_lookat_pos[0];
                    // vec3 lookat_pos = camera_lookat_pos[1];
                    // float depth = (camera_pos - lookat_pos).magnitude();
                    // glm::vec4 clip_pos(ndcX, ndcY, depth, 1.0f);
                    // glm::vec4 view_pos = glm::inverse(perspectiveTransformationMatrix) * clip_pos;
                    // view_pos /= view_pos.w;
                    // glm::mat4 viewMatrix = visualizer->getViewMatrix();
                    // glm::vec4 world_pos = glm::inverse(viewMatrix) * view_pos;
                    // glm::vec3 final_pos = glm::vec3(world_pos.x, world_pos.y, world_pos.z);
                    ImVec2 mouse_pos = ImGui::GetMousePos();
                    std::vector<vec3> camera_lookat_pos = visualizer->getCameraPosition();
                    vec3 camera_pos = camera_lookat_pos[0];
                    vec3 lookat_pos = camera_lookat_pos[1];
                    float drag_distance = std::sqrt(std::pow(mouse_pos.x - dragging_start_position.x, 2) +
                                          std::pow(mouse_pos.y - dragging_start_position.y, 2));;
                    float depth = (camera_pos - lookat_pos).magnitude();
                    vec3 view_dir =  (camera_pos - lookat_pos).normalize();

                    vec3 world_up = vec3(0, 1, 0);

                    vec3 right_dir = cross(world_up, view_dir);
                    right_dir = right_dir.normalize();

                    vec3 up_dir = cross(view_dir, right_dir).normalize();

                    right_dir = cross(view_dir, world_up).normalize();
                    float base_offset = 0.001f;
                    float offset = base_offset * depth;

                    if (currently_dragging_type == "canopy"){
                        canopy_origins[canopy_labels_dict[currently_dragging]] += drag_distance * offset * (up_dir + right_dir);
                    } else if (currently_dragging_type == "rig"){
                        camera_positions[rig_dict[currently_dragging]] += drag_distance * offset * (up_dir + right_dir);
                    }
                }
                currently_dragging = "";
                disable_dragging = false;
                dragging_start_position = int2(0, 0);
            }
            int object_window_count = 0;
            for (auto current_label : canopy_labels_set){
                vec3 canopy_origin_ = canopy_origins[canopy_labels_dict[current_label]];
                origin_position = glm::vec4(canopy_origin_.x, canopy_origin_.y, canopy_origin_.z, 1.0);
                origin_position = perspectiveTransformationMatrix * origin_position;
                // ImGui::SetNextWindowSize(ImVec2(150, 10), ImGuiCond_Always);
                // ImGui::SetNextWindowSize(ImVec2(150, 10));
                ImVec2 next_window_pos = ImVec2(((origin_position.x / origin_position.w) * 0.5f + 0.5f) * windowSize.x,
                                                (1.0f - ((origin_position.y / origin_position.w) * 0.5f + 0.5f)) * windowSize.y);
                // WINDOW MOUSE CLICK AND DRAG
                // TODO: re-enable this feature when it is improved
                // ImVec2 mouse_pos = ImGui::GetMousePos();
                // bool drag_window = (mouse_pos.x >= next_window_pos.x && mouse_pos.x <= next_window_pos.x + 150 &&
                //                     mouse_pos.y >= next_window_pos.y && mouse_pos.y <= next_window_pos.y + 10);
                // if (drag_window && ImGui::IsMouseDown(ImGuiMouseButton_Left) && currently_dragging.empty() && !disable_dragging){
                //     currently_dragging = current_label;
                //     currently_dragging_type = "canopy";
                //     dragging_start_position = int2(mouse_pos.x, mouse_pos.y);
                // }
                // if (!disable_dragging && currently_dragging == current_label){
                //     ImGui::SetNextWindowPos(mouse_pos);
                // } else{
                ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
                ImGui::SetNextWindowPos(next_window_pos);
                // }
                ImGui::Begin((current_label + "###" + std::to_string(object_window_count)).c_str(), nullptr, //&my_tool_active,
                                ImGui::IsWindowCollapsed() ? 0 : ImGuiWindowFlags_AlwaysAutoResize);
                if (ImGui::IsWindowCollapsed()){
                    ImGui::SetWindowSize(ImVec2(150, 10));
                }
                canopyTab(current_label, object_window_count);
                ImGui::End();
                object_window_count++;
            }
            for (auto current_label : rig_labels_set){
                vec3 camera_position_ = camera_positions[rig_dict[current_label]];
                origin_position = glm::vec4(camera_position_.x, camera_position_.y, camera_position_.z, 1.0);
                origin_position = perspectiveTransformationMatrix * origin_position;
                // ImGui::SetNextWindowSize(ImVec2(150, 10), ImGuiCond_Always);
                // ImGui::SetNextWindowSize(ImVec2(150, 10));
                ImVec2 next_window_pos = ImVec2(((origin_position.x / origin_position.w) * 0.5f + 0.5f) * windowSize.x,
                                                (1.0f - ((origin_position.y / origin_position.w) * 0.5f + 0.5f)) * windowSize.y);
                // WINDOW MOUSE CLICK AND DRAG
                // TODO: re-enable this feature when it is improved
                // ImVec2 mouse_pos = ImGui::GetMousePos();
                // bool drag_window = (mouse_pos.x >= next_window_pos.x && mouse_pos.x <= next_window_pos.x + 150 &&
                //                     mouse_pos.y >= next_window_pos.y && mouse_pos.y <= next_window_pos.y + 10);
                // if (drag_window && ImGui::IsMouseDown(ImGuiMouseButton_Left) && currently_dragging.empty() && !disable_dragging){
                //     currently_dragging = current_label;
                //     currently_dragging_type = "rig";
                //     dragging_start_position = int2(mouse_pos.x, mouse_pos.y);
                // }
                // if (!disable_dragging && currently_dragging == current_label){
                //     ImGui::SetNextWindowPos(mouse_pos);
                // } else{
                ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
                ImGui::SetNextWindowPos(next_window_pos);
                // }
                ImGui::Begin((current_label + "###" + std::to_string(object_window_count)).c_str(), nullptr, // &my_tool_active,
                                ImGui::IsWindowCollapsed() ? 0 : ImGuiWindowFlags_AlwaysAutoResize);
                if (ImGui::IsWindowCollapsed()){
                    ImGui::SetWindowSize(ImVec2(150, 10));
                }
                rigTab(current_label, object_window_count);
                ImGui::End();
                object_window_count++;
            }
            for (std::string obj_name : obj_names_set){
                current_label = obj_name;
                vec3 obj_position_ = objects_dict[current_label].position;
                origin_position = glm::vec4(obj_position_.x, obj_position_.y, obj_position_.z, 1.0);
                origin_position = perspectiveTransformationMatrix * origin_position;
                // ImGui::SetNextWindowSize(ImVec2(150, 10), ImGuiCond_Always);
                // ImGui::SetNextWindowSize(ImVec2(150, 10));
                ImVec2 next_window_pos = ImVec2(((origin_position.x / origin_position.w) * 0.5f + 0.5f) * windowSize.x,
                                                (1.0f - ((origin_position.y / origin_position.w) * 0.5f + 0.5f)) * windowSize.y);
                // WINDOW MOUSE CLICK AND DRAG
                // TODO: re-enable this feature when it is improved
                // ImVec2 mouse_pos = ImGui::GetMousePos();
                // bool drag_window = (mouse_pos.x >= next_window_pos.x && mouse_pos.x <= next_window_pos.x + 150 &&
                //                     mouse_pos.y >= next_window_pos.y && mouse_pos.y <= next_window_pos.y + 10);
                // if (drag_window && ImGui::IsMouseDown(ImGuiMouseButton_Left) && currently_dragging.empty() && !disable_dragging){
                //     currently_dragging = current_label;
                //     currently_dragging_type = "object";
                //     dragging_start_position = int2(mouse_pos.x, mouse_pos.y);
                // }
                // if (!disable_dragging && currently_dragging == current_label){
                //     ImGui::SetNextWindowPos(mouse_pos);
                // } else{
                ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
                ImGui::SetNextWindowPos(next_window_pos);
                // }
                ImGui::Begin((current_label + "###" + std::to_string(object_window_count)).c_str(), nullptr, // &my_tool_active,
                                ImGui::IsWindowCollapsed() ? 0 : ImGuiWindowFlags_AlwaysAutoResize);
                if (ImGui::IsWindowCollapsed()){
                    ImGui::SetWindowSize(ImVec2(150, 10));
                }
                objectTab(current_label, object_window_count);
                ImGui::End();
                object_window_count++;
            }
            //
            ImGui::SetNextWindowCollapsed(false, ImGuiCond_Once);
            ImGui::Begin("Editor", nullptr, window_flags2);  // Begin editor window
            // ImGui::SetNextWindowPos(ImVec2(windowSize.x - 400.0f, 0), ImGuiCond_Always); // flag -> can't move window with mouse
            // ImGui::SetNextWindowPos(ImVec2(windowSize.x - 400.0f, 0), ImGuiCond_Always);
            current_position = ImGui::GetWindowPos();
            currently_collapsed = ImGui::IsWindowCollapsed();

            if (current_tab != previous_tab || current_position.x != last_position.x || current_position.y != last_position.y || currently_collapsed != previously_collapsed) {
                user_input = true;
                previous_tab = current_tab;
            }
            if (ImGui::BeginMenuBar()){
                if (ImGui::BeginMenu("File")){
                    if (ImGui::MenuItem("Open..", "Ctrl+O")){
                        std::string file_name = file_dialog();
                        if (!file_name.empty()){
                            xmlGetValues(file_name);
                        }
                    }
                    if (ImGui::MenuItem("Save XML", "Ctrl+S")){
                        xmlSetValues();
                    }
                    if (ImGui::MenuItem("Save As", "Ctrl+S")){
                        std::string new_xml_file = save_as_file_dialog(std::vector<std::string>{"XML"});
                        if (!new_xml_file.empty()){
                            std::string file_extension = new_xml_file;
                            size_t last_obj_file = file_extension.rfind('.');
                            if (last_obj_file != std::string::npos){
                                file_extension = file_extension.substr(last_obj_file + 1);
                            }
                            if (file_extension == "xml"){
                                if (!std::filesystem::exists(new_xml_file)){
                                    // Create file
                                    std::filesystem::copy(xml_input_file, new_xml_file, std::filesystem::copy_options::overwrite_existing);
                                }
                                // Change XML input file
                                std::string xml_input_file_ = xml_input_file;
                                pugi::xml_node helios_ = helios;
                                xml_input_file = new_xml_file;
                                if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
                                    helios_runtime_error(xml_error_string);
                                }
                                xmlSetValues();
                                // Change XML input file back to original
                                // xml_input_file = xml_input_file_;
                                // helios = helios_;
                                // if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
                                //     helios_runtime_error(xml_error_string);
                                // }
                            } else{
                                // Needs to be a obj or ply file
                                std::cout << "Not a valid file type. Project must be saved to a XML file." << std::endl;
                            }
                        } else{
                            // Not a valid file
                            std::cout << "Not a valid file." << std::endl;
                        }
                    }
                    if (ImGui::MenuItem("Close", "Ctrl+W"))  { my_tool_active = false; }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Visualization")){
                    // ImGui::PushFont(font_awesome);
                    // io.FontDefault = font_awesome;
                    if (ImGui::MenuItem("! REFRESH LIST !")){
                        refreshVisualizationTypes();
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
                            visualizer->addCoordinateAxes(helios::make_vec3(0,0,0.05), helios::make_vec3(1,1,1), "positive");
                        }else{
                            visualizer->clearColor();
                            visualizer->disableColorbar();
                            visualizer->addCoordinateAxes(helios::make_vec3(0,0,0.05), helios::make_vec3(1,1,1), "positive");
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
                // #ifdef ENABLE_RADIATION_MODEL
                // radiation->enableCameraModelVisualization();
                // #endif //RADIATION_MODEL
                #ifdef ENABLE_PLANT_ARCHITECTURE
                visualizer->buildContextGeometry(context);
                #endif //PLANT_ARCHITECTURE
                visualizer->addCoordinateAxes(helios::make_vec3(0,0,0.05), helios::make_vec3(1,1,1), "positive");
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
                const char* font_name = "LCD";
                visualizer->addTextboxByCenter("LOADING...", vec3(.5,.5,0), make_SphericalCoord(0, 0),
                    RGB::red, 40, font_name, Visualizer::COORDINATES_WINDOW_NORMALIZED);
                visualizer->plotUpdate();
                updateSpectra();
                updateCameras(); //TODO: figure out why this causes an error
                record();
                visualizer->clearGeometry();
                visualizer->buildContextGeometry(context);
                visualizer->plotUpdate();
            }
            recordPopup();
            ImGui::OpenPopupOnItemClick("repeat_record", ImGuiPopupFlags_MouseButtonRight);
            #endif //RADIATION_MODEL
            // ####### RESULTS ####### //
            // ImGui::Text("Absorbed PAR: %f W/m^2", PAR_absorbed);
            // ImGui::Text("Absorbed NIR: %f W/m^2", NIR_absorbed);
            // ImGui::Text("Absorbed  LW: %f W/m^2", LW_absorbed);
            ImGui::Text("Console:");
            outputConsole();
            if (ImGui::BeginTabBar("Settings#left_tabs_bar")){
                if (ImGui::BeginTabItem("General")){
                    current_tab = "General";
                    // ####### LOCATION ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Visualization:");
                    ImGui::SetWindowFontScale(1.0f);
                    // ####### COORDINATE AXES ####### //
                    bool enable_coords_ = enable_coordinate_axes;
                    toggle_button("##coordinate_axes", &enable_coordinate_axes);
                    if (enable_coords_ != enable_coordinate_axes){
                        if (enable_coordinate_axes){
                            visualizer->addCoordinateAxes(helios::make_vec3(0,0,0.05), helios::make_vec3(1,1,1), "positive");
                            visualizer->plotUpdate();
                        } else{
                            refreshVisualization();
                        }
                    }
                    ImGui::SameLine();
                    if (enable_coordinate_axes){
                        ImGui::Text("Coordinate Axes Enabled");
                    } else{
                        ImGui::Text("Coordinate Axes Disabled");
                    }
                    // ####### LIGHTING MODEL ####### //
                    std::string prev_lighting_model = lighting_model;
                    ImGui::SetNextItemWidth(120);
                    dropDown("Lighting Model", lighting_model, lighting_models);
                    if (prev_lighting_model != lighting_model){
                        if (lighting_model == "None") visualizer->setLightingModel(Visualizer::LIGHTING_NONE);
                        if (lighting_model == "Phong") visualizer->setLightingModel(Visualizer::LIGHTING_PHONG);
                        if (lighting_model == "Phong Shadowed") visualizer->setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
                    }
                    ImGui::SetNextItemWidth(120);
                    // ####### LIGHTING INTENSITY ####### //
                    ImGui::InputFloat("Light Intensity Factor", &light_intensity);
                    visualizer->setLightIntensityFactor(light_intensity);
                    // ####### LIGHTING DIRECTION ####### //
                    float light_dir[3];
                    light_dir[0] = light_direction.x;
                    light_dir[1] = light_direction.y;
                    light_dir[2] = light_direction.z;
                    ImGui::InputFloat3("Light Direction", light_dir);
                    light_direction.x = light_dir[0];
                    light_direction.y = light_dir[1];
                    light_direction.z = light_dir[2];
                    visualizer->setLightDirection(light_direction);
                    // ####### LOCATION ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Location:");
                    ImGui::SetWindowFontScale(1.0f);
                    if (ImGui::Button("Update Location")){
                        updateLocation();
                    }
                    // ####### LATITUDE ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputFloat("Latitude", &latitude);
                    randomizePopup("latitude", createTaggedPtr(&latitude));
                    randomizerParams("latitude");
                    ImGui::OpenPopupOnItemClick("randomize_latitude", ImGuiPopupFlags_MouseButtonRight);
                    ImGui::SameLine();
                    // ####### LONGITUDE ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputFloat("Longitude", &longitude);
                    randomizePopup("longitude", createTaggedPtr(&longitude));
                    randomizerParams("longitude");
                    ImGui::OpenPopupOnItemClick("randomize_longitude", ImGuiPopupFlags_MouseButtonRight);
                    // ####### UTC OFFSET ####### //
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("UTC Offset", &UTC_offset);
                    randomizePopup("UTC_offset", createTaggedPtr(&UTC_offset));
                    randomizerParams("UTC_offset");
                    ImGui::OpenPopupOnItemClick("randomize_UTC_offset", ImGuiPopupFlags_MouseButtonRight);
                     // ####### Weather File ####### //
                    ImGui::SetNextItemWidth(60);
                    ImGui::RadioButton("CSV", is_weather_file_csv); if (ImGui::IsItemClicked()) is_weather_file_csv = true;
                    ImGui::SameLine();
                    ImGui::RadioButton("CIMIS", !is_weather_file_csv); if (ImGui::IsItemClicked()) is_weather_file_csv = false;
                    std::string prev_weather_file;
                    std::string *weather_file;
                    if (is_weather_file_csv){
                        ImGui::Text("CSV");
                        weather_file = &csv_weather_file;
                        prev_weather_file = csv_weather_file;
                    } else{
                        ImGui::Text("CIMIS");
                        weather_file = &cimis_weather_file;
                        prev_weather_file = cimis_weather_file;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Weather File")){
                        std::string weather_file_ = file_dialog();
                        if (!weather_file_.empty()){
                            *weather_file = weather_file_;
                            try{
                                if (is_weather_file_csv){
                                    context->loadTabularTimeseriesData(*weather_file, {}, ",", "YYYYMMDD", 1 );
                                } else{
                                    context->loadTabularTimeseriesData(*weather_file, {"CIMIS"}, ",");
                                }
                            } catch(...) {
                                std::cout << "Failed to load weather file: " << *weather_file << std::endl;
                                *weather_file = prev_weather_file;
                            }
                        } else{
                            *weather_file = prev_weather_file;
                        }
                    }
                    ImGui::SameLine();
                    std::string shorten_weather_file = *weather_file;
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
                    // ####### GROUND ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Ground:");
                    ImGui::SetWindowFontScale(1.0f);
                    if (ImGui::Button("Update Ground")){
                        updateGround();
                        updateSpectra();
                        refreshVisualization();
                    }
                    // ImGui::RadioButton("Manually Set Color", ground_flag == 0); if (ImGui::IsItemClicked()) ground_flag = 0;
                    // ImGui::SameLine();
                    ImGui::RadioButton("Use Texture File", ground_flag == 1); if (ImGui::IsItemClicked()) ground_flag = 1;
                    ImGui::SameLine();
                    ImGui::RadioButton("Use Model File", ground_flag == 2); if (ImGui::IsItemClicked()) ground_flag = 2;
                    if (ground_flag == 0) {
                        // ####### GROUND COLOR ####### //
                        ImGui::ColorEdit3("##ground_color_edit", ground_color);
                    } else if (ground_flag == 1){
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
                    } else if (ground_flag == 2){
                        // ####### GROUND Model File ####### //
                        ImGui::SetNextItemWidth(60);
                        if (ImGui::Button("Ground Model File")){
                            std::string ground_model_file_ = file_dialog();
                            if (!ground_model_file_.empty()){
                                ground_model_file = ground_model_file_;
                            }
                        }
                        ImGui::SameLine();
                        ImGui::Text("%s", shortenPath(ground_model_file).c_str());
                    }
                    toggle_button("##use_ground_texture_color", &use_ground_texture);
                    // ####### GROUND COLOR ####### //
                    if (use_ground_texture){
                        ImGui::SameLine();
                        ImGui::Text("Use Ground Texture Color");
                    } else{
                        ImGui::SameLine();
                        ImGui::ColorEdit3("##ground_color_edit", ground_color);
                        ImGui::SameLine();
                        ImGui::Text("Manually Set Ground Color");
                    }
                    if (ground_flag == 1){
                        // ####### GROUND RESOLUTION ####### //
                        ImGui::SetNextItemWidth(100);
                        ImGui::InputInt("##ground_resolution_x", &ground_resolution.x);
                        randomizePopup("ground_resolution_x", createTaggedPtr(&ground_resolution.x));
                        randomizerParams("ground_resolution_x");
                        ImGui::OpenPopupOnItemClick("randomize_ground_resolution_x", ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(100);
                        ImGui::InputInt("##ground_resolution_y", &ground_resolution.y);
                        randomizePopup("ground_resolution_y", createTaggedPtr(&ground_resolution.y));
                        randomizerParams("ground_resolution_y");
                        ImGui::OpenPopupOnItemClick("randomize_ground_resolution_y", ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Ground Resolution");
                        // ####### DOMAIN EXTENT ####### //
                        ImGui::SetNextItemWidth(50);
                        ImGui::InputFloat("##domain_extent_x", &domain_extent.x);
                        randomizePopup("domain_extent_x", createTaggedPtr(&domain_extent.x));
                        randomizerParams("domain_extent_x");
                        ImGui::OpenPopupOnItemClick("randomize_domain_extent_x", ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(50);
                        ImGui::InputFloat("##domain_extent_y", &domain_extent.y);
                        randomizePopup("domain_extent_y", createTaggedPtr(&domain_extent.y));
                        randomizerParams("domain_extent_y");
                        ImGui::OpenPopupOnItemClick("randomize_domain_extent_y", ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Domain Extent");
                    }
                    // ####### DOMAIN ORIGIN ####### //
                    ImGui::SetNextItemWidth(60);
                    ImGui::InputFloat("##domain_origin_x", &domain_origin.x);
                    randomizePopup("domain_origin_x", createTaggedPtr(&domain_origin.x));
                    randomizerParams("domain_origin_x");
                    ImGui::OpenPopupOnItemClick("randomize_domain_origin_x", ImGuiPopupFlags_MouseButtonRight);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(60);
                    ImGui::InputFloat("##domain_origin_y", &domain_origin.y);
                    randomizePopup("domain_origin_y", createTaggedPtr(&domain_origin.y));
                    randomizerParams("domain_origin_y");
                    ImGui::OpenPopupOnItemClick("randomize_domain_origin_y", ImGuiPopupFlags_MouseButtonRight);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(60);
                    ImGui::InputFloat("##domain_origin_z", &domain_origin.z);
                    randomizePopup("domain_origin_z", createTaggedPtr(&domain_origin.z));
                    randomizerParams("domain_origin_z");
                    ImGui::OpenPopupOnItemClick("randomize_domain_origin_z", ImGuiPopupFlags_MouseButtonRight);
                    ImGui::SameLine();
                    ImGui::Text("Domain Origin");
                    #ifdef ENABLE_CANOPY_GENERATOR
                    // ####### NUMBER OF TILES ####### //
                    ImGui::SetNextItemWidth(60);
                    int temp[2];
                    temp[0] = num_tiles.x;
                    temp[1] = num_tiles.y;
                    ImGui::InputInt2("Number of Tiles", temp);
                    num_tiles.x = temp[0];
                    num_tiles.y = temp[1];
                    // ####### SUBPATCHES ####### //
                    ImGui::SetNextItemWidth(60);
                    temp[0] = subpatches.x;
                    temp[1] = subpatches.y;
                    ImGui::InputInt2("Subpatches", temp);
                    subpatches.x = temp[0];
                    subpatches.y = temp[1];
                    #endif

                    ImGui::EndTabItem();
                }
                // OBJECT TAB
                if (ImGui::BeginTabItem("Object")){
                    current_tab = "Object";
                    if (ImGui::Button("Load Object File")){
                        std::string new_obj_file = file_dialog();
                        if ( !new_obj_file.empty() && std::filesystem::exists(new_obj_file) ){
                            if( std::filesystem::path(new_obj_file).extension() != ".obj" && std::filesystem::path(new_obj_file).extension() != ".ply" ){
                                std::cout << "Object file must have .obj or .ply extension." << std::endl;
                            } else{
                                object new_obj;
                                std::vector<uint> new_UUIDs;
                                if( std::filesystem::path(new_obj_file).extension() == ".obj" ){
                                    new_UUIDs = context->loadOBJ(new_obj_file.c_str());
                                } else if ( std::filesystem::path(new_obj_file).extension() == ".ply" ){
                                    new_UUIDs = context->loadPLY(new_obj_file.c_str());
                                }
                                // check for MTL file
                                std::filesystem::path mtl_path(new_obj_file);
                                mtl_path.replace_extension("mtl");
                                if (std::filesystem::exists(mtl_path)){
                                    new_obj.use_texture_file = true;
                                } else{
                                    new_obj.use_texture_file = false;
                                }
                                visualizer->buildContextGeometry(context);
                                visualizer->plotUpdate();
                                std::string default_object_label = "object";
                                std::string new_obj_label = "object_0";
                                int count = 0;
                                while (objects_dict.find(new_obj_label) != objects_dict.end()){
                                    count++;
                                    new_obj_label = default_object_label + "_" + std::to_string(count);
                                }
                                obj_names_set.insert(new_obj_label);
                                new_obj.index = obj_idx;
                                obj_idx++;
                                new_obj.name = new_obj_label;
                                new_obj.file = new_obj_file;
                                new_obj.UUIDs = new_UUIDs;
                                new_obj.position = vec3(0,0,0);
                                new_obj.prev_position = vec3(0,0,0);
                                new_obj.orientation = vec3(0,0,0);
                                new_obj.prev_orientation = vec3(0,0,0);
                                new_obj.scale = vec3(1,1,1);
                                new_obj.prev_scale = vec3(1,1,1);
                                new_obj.color = RGBcolor(0, 0, 1);
                                new_obj.prev_color = RGBcolor(0, 0, 1);
                                new_obj.data_group = "";
                                new_obj.is_dirty = false;
                                current_obj = new_obj_label;
                                objects_dict[new_obj_label] = new_obj;
                            }
                        }
                    }
                    if (ImGui::BeginCombo("##obj_combo", current_obj.c_str())){
                        for (std::string obj_name : obj_names_set){
                            bool is_obj_selected = (current_obj == obj_name);
                            if (ImGui::Selectable(obj_name.c_str(), is_obj_selected))
                                current_obj = obj_name;
                            if (is_obj_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::Text("Select Object");
                    if ( !current_obj.empty() ){
                        if (ImGui::Button("Update Object")){
                            updateObject(current_obj);
                            refreshVisualization();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Delete Object")){
                            deleteObject(current_obj);
                        }
                        if (objects_dict[current_obj].is_dirty){
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255)); // Red text
                            ImGui::Text("update required");
                            ImGui::PopStyleColor();
                        }
                        ImGui::SetNextItemWidth(100);
                        std::string prev_obj_name = objects_dict[current_obj].name;
                        ImGui::InputText("##obj_name", &objects_dict[current_obj].name);
                        if (objects_dict[current_obj].name != prev_obj_name && !objects_dict[current_obj].name.empty() && obj_names_set.find(objects_dict[current_obj].name) == obj_names_set.end()){
                            object temp_obj = objects_dict[prev_obj_name];
                            current_obj = objects_dict[current_obj].name;
                            std::map<std::string, object>::iterator delete_obj_iter = objects_dict.find(prev_obj_name);
                            if (delete_obj_iter != objects_dict.end()){
                                objects_dict.erase(delete_obj_iter);
                            }
                            objects_dict[current_obj] = temp_obj;
                            obj_names_set.erase(prev_obj_name);
                            obj_names_set.insert(current_obj);
                        } else {
                            objects_dict[current_obj].name = prev_obj_name;
                        }
                        ImGui::SameLine();
                        ImGui::Text("Object Name");
                        if (!current_obj.empty()){
                            // ####### OBJECT DATA GROUP ####### //
                            ImGui::SetNextItemWidth(100);
                            std::string prev_obj_data_group = objects_dict[current_obj].data_group;
                            ImGui::InputText("##obj_data_group", &objects_dict[current_obj].data_group);
                            if (objects_dict[current_obj].data_group == "All" || objects_dict[current_obj].data_group.empty()){
                                objects_dict[current_obj].data_group = prev_obj_data_group;
                            }
                            if (!objects_dict[current_obj].data_group.empty() && prev_obj_data_group != objects_dict[current_obj].data_group){
                                std::string new_data_group = objects_dict[current_obj].data_group;
                                context->setPrimitiveData(objects_dict[current_obj].UUIDs, "data_group", new_data_group);
                            }
                            ImGui::SameLine();
                            ImGui::Text("Data Group");
                            bool use_obj_texture = objects_dict[current_obj].use_texture_file;
                            toggle_button("##use_texture_file", &use_obj_texture);
                            if (use_obj_texture != objects_dict[current_obj].use_texture_file){
                                objects_dict[current_obj].use_texture_file = use_obj_texture;
                                objects_dict[current_obj].is_dirty = true;
                            }
                            ImGui::SameLine();
                            if (!use_obj_texture){
                                // ####### OBJECT COLOR ####### //
                                float col[3];
                                col[0] = objects_dict[current_obj].color.r;
                                col[1] = objects_dict[current_obj].color.g;
                                col[2] = objects_dict[current_obj].color.b;
                                ImGui::ColorEdit3("##obj_color_edit", col);
                                if (objects_dict[current_obj].color.r != col[0] ||
                                    objects_dict[current_obj].color.g != col[1] ||
                                    objects_dict[current_obj].color.b != col[2]){
                                    updateColor(current_obj, "obj", col);
                                }
                                ImGui::SameLine();
                                ImGui::Text("Object Color");
                            } else{
                                // ####### OBJECT TEXTURE FILE ####### //
                                ImGui::Text("Use Object Texture File");
                            }
                            // ####### OBJECT SCALE ####### //
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_scale_x", &objects_dict[current_obj].scale.x);
                            randomizePopup("obj_scale_x_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].scale.x, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_scale_x_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_scale_x_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_scale_y", &objects_dict[current_obj].scale.y);
                            randomizePopup("obj_scale_y_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].scale.y, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_scale_y_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_scale_y_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_scale_z", &objects_dict[current_obj].scale.z);
                            randomizePopup("obj_scale_z_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].scale.z, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_scale_z_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_scale_z_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::Text("Object Scale");
                            // ####### OBJECT POSITION ####### //
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_position_x", &objects_dict[current_obj].position.x);
                            randomizePopup("obj_position_x_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].position.x, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_position_x_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_position_x_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_position_y", &objects_dict[current_obj].position.y);
                            randomizePopup("obj_position_y_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].position.y, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_position_y_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_position_y_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_position_z", &objects_dict[current_obj].position.z);
                            randomizePopup("obj_position_z_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].position.z, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_position_z_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_position_z_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::Text("Object Position");
                            // ####### OBJECT ORIENTATION ####### //
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_orientation_x", &objects_dict[current_obj].orientation.x);
                            randomizePopup("obj_orientation_x_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].orientation.x, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_orientation_x_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_orientation_x_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_orientation_y", &objects_dict[current_obj].orientation.y);
                            randomizePopup("obj_orientation_y_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].orientation.y, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_orientation_y_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_orientation_y_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(60);
                            ImGui::InputFloat("##obj_orientation_z", &objects_dict[current_obj].orientation.z);
                            randomizePopup("obj_orientation_z_" + std::to_string(objects_dict[current_obj].index), createTaggedPtr(&objects_dict[current_obj].orientation.z, &objects_dict[current_obj].is_dirty));
                            randomizerParams("obj_orientation_z_" + std::to_string(objects_dict[current_obj].index));
                            ImGui::OpenPopupOnItemClick(("randomize_obj_orientation_z_" + std::to_string(objects_dict[current_obj].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                            ImGui::SameLine();
                            ImGui::Text("Object Orientation");
                        }
                        if (objects_dict[current_obj].position != objects_dict[current_obj].prev_position ||
                            objects_dict[current_obj].orientation != objects_dict[current_obj].prev_orientation ||
                            objects_dict[current_obj].scale != objects_dict[current_obj].prev_scale ||
                            objects_dict[current_obj].color != objects_dict[current_obj].prev_color){
                            objects_dict[current_obj].is_dirty = true;
                        }
                    }
                    ImGui::EndTabItem();
                }
            // if (enable_plantarchitecture){
            #ifdef ENABLE_PLANT_ARCHITECTURE
                // CANOPY TAB
                if (ImGui::BeginTabItem("Canopy")){
                    current_tab = "Canopy";
                    if (ImGui::BeginCombo("##combo", current_canopy.c_str()))
                    {
                        for (auto canopy_label_ : canopy_labels_set){
                            bool is_selected = (current_canopy == canopy_label_);
                            if (ImGui::Selectable(canopy_label_.c_str(), is_selected))
                                current_canopy = canopy_label_;
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Add Canopy")){
                        addCanopy();
                    }
                    if (!current_canopy.empty()){
                        if (ImGui::Button("Update Canopy")){
                            updateCanopy(current_canopy);
                            refreshVisualization();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Delete Canopy")){
                            deleteCanopy(current_canopy);
                            updatePrimitiveTypes();
                            refreshVisualization();
                        }
                        ImGui::SetNextItemWidth(100);
                        std::string prev_canopy_name = canopy_labels[canopy_labels_dict[current_canopy]];
                        ImGui::InputText("##canopy_name", &canopy_labels[canopy_labels_dict[current_canopy]]);
                        if (canopy_labels[canopy_labels_dict[current_canopy]] != prev_canopy_name && canopy_labels_dict.find(canopy_labels[canopy_labels_dict[current_canopy]]) == canopy_labels_dict.end() && !canopy_labels[canopy_labels_dict[current_canopy]].empty()){
                            int temp = canopy_labels_dict[current_canopy];
                            current_canopy = canopy_labels[canopy_labels_dict[current_canopy]];
                            std::map<std::string, int>::iterator current_canopy_iter = canopy_labels_dict.find(prev_canopy_name);
                            if (current_canopy_iter != canopy_labels_dict.end()){
                                canopy_labels_dict.erase(current_canopy_iter);
                            }
                            canopy_labels_dict[current_canopy] = temp;

                            canopy_labels_set.erase(prev_canopy_name);
                            canopy_labels_set.insert(current_canopy);
                        } else{
                            canopy_labels[canopy_labels_dict[current_canopy]] = prev_canopy_name;
                        }
                        ImGui::SameLine();
                        ImGui::Text("Canopy Name");
                        // ####### PLANT LIBRARY NAME ####### //
                        ImGui::SetNextItemWidth(250);
                        // ImGui::InputText("Plant Library", &plant_library_names[canopy_labels_dict[current_canopy]]);
                        dropDown("Plant Library###dropdown", plant_library_names_verbose[canopy_labels_dict[current_canopy]], plant_types_verbose);
                        plant_library_names[canopy_labels_dict[current_canopy]] = plant_type_lookup[plant_library_names_verbose[canopy_labels_dict[current_canopy]]];
                        // ######### CANOPY DATA GROUP ####### //
                        ImGui::SetNextItemWidth(100);
                        std::string prev_canopy_data_group = canopy_data_groups[canopy_labels_dict[current_canopy]];
                        ImGui::InputText("##canopy_data_group", &canopy_data_groups[canopy_labels_dict[current_canopy]]);
                        if (canopy_data_groups[canopy_labels_dict[current_canopy]] == "All" || canopy_data_groups[canopy_labels_dict[current_canopy]].empty()){
                            canopy_data_groups[canopy_labels_dict[current_canopy]] = prev_canopy_data_group;
                        }
                        if (!canopy_data_groups[canopy_labels_dict[current_canopy]].empty() && prev_canopy_data_group != canopy_data_groups[canopy_labels_dict[current_canopy]]){
                            std::string new_data_group = canopy_data_groups[canopy_labels_dict[current_canopy]];
                            std::vector<uint> canopy_primID_vec;
                            for (int i = 0; i < canopy_IDs[canopy_labels_dict[current_canopy]].size(); i++){
                                std::vector<uint> new_canopy_primIDs = plantarchitecture->getAllPlantUUIDs(canopy_IDs[canopy_labels_dict[current_canopy]][i]);
                                canopy_primID_vec.insert(canopy_primID_vec.end(), new_canopy_primIDs.begin(), new_canopy_primIDs.end());
                            }
                            context->setPrimitiveData(canopy_primID_vec, "data_group", new_data_group);
                        }
                        ImGui::SameLine();
                        ImGui::Text("Data Group");
                        // ####### CANOPY ORIGIN ####### //
                        obj curr_plant = obj{canopy_labels_dict[current_canopy], true};
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##canopy_origin_x", &canopy_origins[canopy_labels_dict[current_canopy]].x);
                        randomizePopup("canopy_origin_x_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&canopy_origins[canopy_labels_dict[current_canopy]].x));
                        randomizerParams("canopy_origin_x_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_canopy_origin_x_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##canopy_origin_y", &canopy_origins[canopy_labels_dict[current_canopy]].y);
                        randomizePopup("canopy_origin_y_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&canopy_origins[canopy_labels_dict[current_canopy]].y));
                        randomizerParams("canopy_origin_y_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_canopy_origin_y_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##canopy_origin_z", &canopy_origins[canopy_labels_dict[current_canopy]].z);
                        randomizePopup("canopy_origin_z_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&canopy_origins[canopy_labels_dict[current_canopy]].z));
                        randomizerParams("canopy_origin_z_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_canopy_origin_z_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Canopy Origin");
                        // ####### PLANT COUNT ####### //
                        ImGui::SetNextItemWidth(100);
                        ImGui::InputInt("##plant_count_x", &plant_counts[canopy_labels_dict[current_canopy]].x);
                        plant_counts[canopy_labels_dict[current_canopy]].x = std::max(plant_counts[canopy_labels_dict[current_canopy]].x, 1);
                        randomizePopup("plant_count_x_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&plant_counts[canopy_labels_dict[current_canopy]].x));
                        randomizerParams("plant_count_x_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_plant_count_x_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(100);
                        ImGui::InputInt("##plant_count_y", &plant_counts[canopy_labels_dict[current_canopy]].y);
                        plant_counts[canopy_labels_dict[current_canopy]].y = std::max(plant_counts[canopy_labels_dict[current_canopy]].y, 1);
                        randomizePopup("plant_count_y_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&plant_counts[canopy_labels_dict[current_canopy]].y));
                        randomizerParams("plant_count_y_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_plant_count_y_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Plant Count");
                        // ####### PLANT SPACING ####### //
                        ImGui::SetNextItemWidth(50);
                        ImGui::InputFloat("##plant_spacing_x", &plant_spacings[canopy_labels_dict[current_canopy]].x);
                        randomizePopup("plant_spacing_x_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&plant_spacings[canopy_labels_dict[current_canopy]].x));
                        randomizerParams("plant_spacing_x_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_plant_spacing_x_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(50);
                        ImGui::InputFloat("##plant_spacing_y", &plant_spacings[canopy_labels_dict[current_canopy]].y);
                        randomizePopup("plant_spacing_y_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&plant_spacings[canopy_labels_dict[current_canopy]].y));
                        randomizerParams("plant_spacing_y_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_plant_spacing_y_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Plant Spacing");
                        // ####### PLANT AGE ####### //
                        ImGui::SetNextItemWidth(80);
                        ImGui::InputFloat("Plant Age", &plant_ages[canopy_labels_dict[current_canopy]]);
                        randomizePopup("plant_age_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&plant_ages[canopy_labels_dict[current_canopy]]));
                        randomizerParams("plant_age_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_plant_age_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        // ####### GROUND CLIPPING HEIGHT ####### //
                        ImGui::SetNextItemWidth(80);
                        ImGui::InputFloat("Ground Clipping Height", &ground_clipping_heights[canopy_labels_dict[current_canopy]]);
                        randomizePopup("ground_clipping_height_" + std::to_string(canopy_labels_dict[current_canopy]), createTaggedPtr(&ground_clipping_heights[canopy_labels_dict[current_canopy]]));
                        randomizerParams("ground_clipping_height_" + std::to_string(canopy_labels_dict[current_canopy]));
                        ImGui::OpenPopupOnItemClick(("randomize_ground_clipping_height_" + std::to_string(canopy_labels_dict[current_canopy])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        if (ImGui::Button("Save Canopy to OBJ/PLY File")){
                            std::string new_obj_file = save_as_file_dialog(std::vector<std::string>{"OBJ", "PLY"});
                            if (!new_obj_file.empty()){
                                std::string file_extension = new_obj_file;
                                size_t last_obj_file = file_extension.rfind('.');
                                if (last_obj_file != std::string::npos){
                                    file_extension = file_extension.substr(last_obj_file + 1);
                                }
                                if (file_extension == "obj" || file_extension == "ply"){
                                    if (!std::filesystem::exists(new_obj_file)){
                                        // Create file
                                        std::ofstream outFile(new_obj_file);
                                    }
                                    if (!save_plants_individually){
                                        saveCanopy(new_obj_file, canopy_IDs[canopy_labels_dict[current_canopy]],
                                                    canopy_origins[canopy_labels_dict[current_canopy]], file_extension);
                                    } else{
                                        saveCanopy(new_obj_file, canopy_IDs[canopy_labels_dict[current_canopy]],
                                                    individual_plant_locations[canopy_labels_dict[current_canopy]], file_extension);
                                    }
                                } else{
                                    // Needs to be a obj or ply file
                                    std::cout << "Not a valid file type. Object must be saved to .obj or .ply file." << std::endl;
                                }
                            } else{
                                // Not a valid file
                                std::cout << "Not a valid file." << std::endl;
                            }
                        }
                        ImGui::SameLine();
                        ImGui::Checkbox("Save plants individually", &save_plants_individually);
                    }
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
                                for (int i = 0; i < current_spectra_file.size(); i++){
                                    possible_spectra.insert(current_spectra_file[i]);
                                }
                            }
                            context->loadXML( new_xml_library_file.c_str() );
                        }
                    }
                    // ####### GLOBAL PROPERTIES ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Global Properties:");
                    ImGui::SetWindowFontScale(1.0f);
                    // ####### ENFORCE PERIODIC BOUNDARY CONDITION ####### //
                    ImGui::Text("Enforce Periodic Boundary Condition:");
                    ImGui::SameLine();
                    bool prev_cond_x = enforce_periodic_boundary_x;
                    ImGui::Checkbox("x###periodic_boundary_x", &enforce_periodic_boundary_x);
                    ImGui::SameLine();
                    bool prev_cond_y = enforce_periodic_boundary_y;
                    ImGui::Checkbox("y###periodic_boundary_y", &enforce_periodic_boundary_y);
                    if (prev_cond_x != enforce_periodic_boundary_x || prev_cond_y != enforce_periodic_boundary_y){
                        if (enforce_periodic_boundary_x && enforce_periodic_boundary_y)
                            radiation->enforcePeriodicBoundary("xy");
                        else if (enforce_periodic_boundary_x)
                            radiation->enforcePeriodicBoundary("x");
                        else if (enforce_periodic_boundary_y)
                            radiation->enforcePeriodicBoundary("y");
                        else
                            radiation->enforcePeriodicBoundary("");
                    }
                    // ####### DIFFUSE EXTINCTION COEFFICIENT ####### //
                    ImGui::SetNextItemWidth(60);
                    float prev_value = diffuse_extinction_coeff;
                    ImGui::InputFloat("Diffuse Extinction Coefficient", &diffuse_extinction_coeff);
                    if (prev_value != diffuse_extinction_coeff){
                        context->setGlobalData( "diffuse_extinction_coeff", diffuse_extinction_coeff );
                    }
                    randomizePopup("diffuse_extinction_coeff", createTaggedPtr(&diffuse_extinction_coeff));
                    randomizerParams("diffuse_extinction_coeff");
                    ImGui::OpenPopupOnItemClick("randomize_diffuse_extinction_coeff", ImGuiPopupFlags_MouseButtonRight);
                    // ####### AIR TURBIDITY ####### //
                    ImGui::SetNextItemWidth(60);
                    prev_value = air_turbidity;
                    ImGui::InputFloat("Air Turbidity", &air_turbidity);
                    if (prev_value != air_turbidity){
                        if( air_turbidity > 0 ){
                            context->setGlobalData( "air_turbidity", air_turbidity );
                        }else if( air_turbidity < 0 ){ //try calibration
                            if( context->doesTimeseriesVariableExist( "net_radiation" ) ){
                                air_turbidity = solarposition->calibrateTurbidityFromTimeseries( "net_radiation" );
                                if( air_turbidity>0  && air_turbidity < 1 ){
                                    context->setGlobalData( "air_turbidity", air_turbidity );
                                }
                            }
                        }
                    }
                    randomizePopup("air_turbidity", createTaggedPtr(&air_turbidity));
                    randomizerParams("air_turbidity");
                    ImGui::OpenPopupOnItemClick("randomize_air_turbidity", ImGuiPopupFlags_MouseButtonRight);
                    // ####### SOLAR DIRECT SPECTRUM ####### //
                    ImGui::SetNextItemWidth(250);
                    // ImGui::InputText("Solar Direct Spectrum", &solar_direct_spectrum);
                    if (ImGui::BeginCombo("##combo_solar_direct_spectrum", solar_direct_spectrum.c_str())){
                        for (auto &spectra : possible_spectra){
                            bool is_solar_direct_spectrum_selected = (solar_direct_spectrum == spectra);
                            if (ImGui::Selectable(spectra.c_str(), is_solar_direct_spectrum_selected))
                                solar_direct_spectrum = spectra;
                            if (is_solar_direct_spectrum_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::Text("Solar Direct Spectrum");
                    // ####### BAND PROPERTIES ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Add Band:");
                    ImGui::SetWindowFontScale(1.0f);
                    // ####### ADD BAND ####### //
                    toggle_button("##enable_wavelength", &enable_wavelength);
                    ImGui::SameLine();
                    if (enable_wavelength){
                        ImGui::Text("Wavelength Min:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##wavelength_min", &wavelength_min);
                        ImGui::SameLine();
                        ImGui::Text("Max:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##wavelength_max", &wavelength_max);
                    } else{
                        ImGui::Text("No Specified Wavelength");
                    }
                    //
                    ImGui::Text("Label:");
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputText("##new_band_label", &new_band_label);
                    ImGui::SameLine();
                    ImGui::Text("Emission:");
                    ImGui::SameLine();
                    ImGui::Checkbox("##enable_emission", &enable_emission);
                    ImGui::SameLine();
                    if (ImGui::Button("Add Band")){
                        if (enable_wavelength) {
                            addBand(new_band_label, wavelength_min, wavelength_max, enable_emission);
                        } else{
                            addBand(new_band_label, enable_emission);
                            bandlabels_set_wavelength.insert(new_band_label);
                        }
                    }
                    // ####### BAND PROPERTIES ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Band Properties:");
                    ImGui::SetWindowFontScale(1.0f);
                    // ####### SELECT BAND ####### //
                    if (ImGui::BeginCombo("##combo_current_band", current_band.c_str())){
                        for (std::string band : bandlabels_set){
                            bool is_current_band_selected = (current_band == band);
                            if (ImGui::Selectable(band.c_str(), is_current_band_selected))
                                current_band = band;
                            if (is_current_band_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::Text("Select Band");
                    // ####### DIRECT RAY COUNT ####### //
                    int prev_direct_ray_count;
                    ImGui::SetNextItemWidth(100);
                    if (current_band == "All"){
                        prev_direct_ray_count = direct_ray_count;
                        ImGui::InputInt("Direct Ray Count", &direct_ray_count);
                        randomizePopup("direct_ray_count", createTaggedPtr(&direct_ray_count));
                        randomizerParams("direct_ray_count");
                        ImGui::OpenPopupOnItemClick("randomize_direct_ray_count", ImGuiPopupFlags_MouseButtonRight);
                        if (direct_ray_count != prev_direct_ray_count){
                            for (std::string band : bandlabels){
                                radiation->setDirectRayCount(band, direct_ray_count);
                                direct_ray_count_dict[band] = direct_ray_count;
                            }
                        }
                    } else{
                        prev_direct_ray_count = direct_ray_count_dict[current_band];
                        ImGui::InputInt("Direct Ray Count", &direct_ray_count_dict[current_band]);
                        randomizePopup("direct_ray_count_" + current_band, createTaggedPtr(&direct_ray_count_dict[current_band]));
                        randomizerParams("direct_ray_count_" + current_band);
                        ImGui::OpenPopupOnItemClick(("randomize_direct_ray_count_" + current_band).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        if (direct_ray_count_dict[current_band] != prev_direct_ray_count){
                            radiation->setDirectRayCount(current_band, direct_ray_count_dict[current_band]);
                        }
                    }
                    // ####### DIFFUSE RAY COUNT ####### //
                    ImGui::SetNextItemWidth(100);
                    int prev_diffuse_ray_count;
                    if (current_band == "All"){
                        prev_diffuse_ray_count = diffuse_ray_count;
                        ImGui::InputInt("Diffuse Ray Count", &diffuse_ray_count);
                        randomizePopup("diffuse_ray_count", createTaggedPtr(&diffuse_ray_count));
                        randomizerParams("diffuse_ray_count");
                        ImGui::OpenPopupOnItemClick("randomize_diffuse_ray_count", ImGuiPopupFlags_MouseButtonRight);
                        if (diffuse_ray_count != prev_diffuse_ray_count){
                            for (std::string band : bandlabels){
                                radiation->setDiffuseRayCount(band, diffuse_ray_count);
                                diffuse_ray_count_dict[band] = diffuse_ray_count;
                            }
                        }
                    } else{
                        prev_diffuse_ray_count = diffuse_ray_count_dict[current_band];
                        ImGui::InputInt("Diffuse Ray Count", &diffuse_ray_count_dict[current_band]);
                        randomizePopup("diffuse_ray_count_" + current_band, createTaggedPtr(&diffuse_ray_count_dict[current_band]));
                        randomizerParams("diffuse_ray_count_" + current_band);
                        ImGui::OpenPopupOnItemClick(("randomize_diffuse_ray_count_" + current_band).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        if (diffuse_ray_count_dict[current_band] != prev_diffuse_ray_count){
                            radiation->setDiffuseRayCount(current_band, diffuse_ray_count_dict[current_band]);
                        }
                    }
                    // ####### SCATTERING DEPTH ####### //
                    ImGui::SetNextItemWidth(100);
                    int prev_scattering_depth;
                    if (current_band == "All"){
                        prev_scattering_depth = scattering_depth;
                        ImGui::InputInt("Scattering Depth", &scattering_depth);
                        randomizePopup("scattering_depth", createTaggedPtr(&scattering_depth));
                        randomizerParams("scattering_depth");
                        ImGui::OpenPopupOnItemClick("randomize_scattering_depth", ImGuiPopupFlags_MouseButtonRight);
                        if (scattering_depth <= 0){
                            scattering_depth = prev_scattering_depth;
                        }
                        if (prev_scattering_depth != scattering_depth){
                            for (std::string band : bandlabels){
                                radiation->setScatteringDepth(band, scattering_depth);
                                scattering_depth_dict[band] = scattering_depth;
                            }
                        }
                    } else{
                        prev_scattering_depth = scattering_depth_dict[current_band];
                        ImGui::InputInt("Scattering Depth", &scattering_depth_dict[current_band]);
                        randomizePopup("scattering_depth_" + current_band, createTaggedPtr(&scattering_depth_dict[current_band]));
                        randomizerParams("scattering_depth_" + current_band);
                        ImGui::OpenPopupOnItemClick(("randomize_scattering_depth_" + current_band).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        if (scattering_depth_dict[current_band] <= 0){ // scattering depth must be >0
                            scattering_depth_dict[current_band] = prev_scattering_depth;
                        }
                        if (prev_scattering_depth != scattering_depth_dict[current_band]){
                            radiation->setScatteringDepth(current_band, scattering_depth_dict[current_band]);
                        }
                    }
                    // ####### RADIATIVE PROPERTIES ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Radiative Properties:");
                    ImGui::SetWindowFontScale(1.0f);
                    ImGui::SetNextItemWidth(100);
                    // ######### SELECT DATA GROUP ############//
                    if (ImGui::Button("Refresh###data_groups_refresh")){
                        updatePrimitiveTypes();
                        updateDataGroups();
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::BeginCombo("##data_group_primitive", current_data_group.c_str())){
                        for (std::string data_group : data_groups_set){
                            bool is_data_group_selected = (current_data_group == data_group);
                            if (ImGui::Selectable(data_group.c_str(), is_data_group_selected))
                                current_data_group = data_group;
                            if (is_data_group_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(100);
                    ImGui::Text("Select Data Group");
                    // default primitive data group
                    // ######### SELECT PRIMITIVE ############//
                    if (ImGui::Button("Refresh")){
                        updatePrimitiveTypes();
                        updateDataGroups();
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
                    ImGui::Text("Select Primitive Type");
                    if (current_data_group == "All"){
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
                            if (ImGui::BeginCombo("##reflectivity_combo_all", reflectivity_prev.c_str())){
                                for (auto& spectra : possible_spectra){
                                    bool is_spectra_selected = (primitive_spectra[current_primitive][0] == spectra);
                                    if (ImGui::Selectable(spectra.c_str(), is_spectra_selected))
                                        primitive_spectra[current_primitive][0] = spectra;
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
                        ImGui::Text("%s",toggle_display_reflectivity.c_str());
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
                                for (auto &spectra : possible_spectra){
                                    bool is_spectra_selected = (primitive_spectra[current_primitive][1] == spectra);
                                    if (ImGui::Selectable(spectra.c_str(), is_spectra_selected))
                                        primitive_spectra[current_primitive][1] = spectra;
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
                        ImGui::Text("%s",toggle_display_transmissivity.c_str());
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
                            for (std::string band : bandlabels_set_emissivity){
                                bool is_band_selected = (current_band_emissivity == band);
                                if (ImGui::Selectable(band.c_str(), is_band_selected))
                                    current_band_emissivity = band;
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
                    } else{ // specific data group
                        // REFLECTIVITY
                        ImGui::Text("Reflectivity:");
                        std::string toggle_display_reflectivity = "Manual Entry";
                        bool reflectivity_continuous = primitive_continuous_dict[current_data_group][current_primitive][0];
                        toggle_button("##reflectivity_toggle", &reflectivity_continuous);
                        if (reflectivity_continuous != primitive_continuous_dict[current_data_group][current_primitive][0]){
                            if (current_primitive == "All"){
                                for (auto &prim_values : primitive_continuous_dict[current_data_group]){
                                    primitive_continuous_dict[current_data_group][prim_values.first][0] = reflectivity_continuous;
                                }
                            }
                            primitive_continuous_dict[current_data_group][current_primitive][0] = reflectivity_continuous;
                        }
                        if (primitive_continuous_dict[current_data_group][current_primitive][0]){
                            toggle_display_reflectivity = "File Entry";
                        }
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(250);
                        if (!primitive_continuous_dict[current_data_group][current_primitive][0]){
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
                                    for (auto &prim_values : primitive_values_dict[current_data_group][current_band_reflectivity]){
                                        primitive_values_dict[current_data_group][current_band_reflectivity][prim_values.first][0] = reflectivity;
                                    }
                                }
                            }else{
                                ImGui::InputFloat("##reflectivity", &primitive_values_dict[current_data_group][current_band_reflectivity][current_primitive][0]);
                            }
                        }else{
                            std::string reflectivity_prev = primitive_spectra_dict[current_data_group][current_primitive][0];
                            if (ImGui::BeginCombo("##reflectivity_combo", reflectivity_prev.c_str())){
                                for (auto &spectra : possible_spectra){
                                    bool is_spectra_selected = (primitive_spectra_dict[current_data_group][current_primitive][0] == spectra);
                                    if (ImGui::Selectable(spectra.c_str(), is_spectra_selected))
                                        primitive_spectra_dict[current_data_group][current_primitive][0] = spectra;
                                    if (is_spectra_selected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }
                            if (current_primitive == "All" && reflectivity_prev != primitive_spectra_dict[current_data_group][current_primitive][0]){
                                for (auto &prim_spectrum : primitive_spectra_dict[current_data_group]){
                                    primitive_spectra_dict[current_data_group][prim_spectrum.first][0] = primitive_spectra_dict[current_data_group][current_primitive][0];
                                }
                            }
                        }
                        ImGui::SameLine();
                        ImGui::TextUnformatted("%s",toggle_display_reflectivity.c_str());
                        // TRANSMISSIVITY
                        ImGui::Text("Transmissivity:");
                        std::string toggle_display_transmissivity = "Manual Entry";
                        bool transmissivity_continuous = primitive_continuous_dict[current_data_group][current_primitive][1];
                        toggle_button("##transmissivity_toggle", &transmissivity_continuous);
                        if (transmissivity_continuous != primitive_continuous_dict[current_data_group][current_primitive][1]){
                            if (current_primitive == "All"){
                                for (auto &prim_values : primitive_continuous_dict[current_data_group]){
                                    primitive_continuous_dict[current_data_group][prim_values.first][1] = transmissivity_continuous;
                                }
                            }
                            primitive_continuous_dict[current_data_group][current_primitive][1] = transmissivity_continuous;
                        }
                        if (primitive_continuous_dict[current_data_group][current_primitive][1]){
                            toggle_display_transmissivity = "File Entry";
                        }
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(250);
                        if (!primitive_continuous_dict[current_data_group][current_primitive][1]){
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
                                    for (auto &prim_values : primitive_values_dict[current_data_group][current_band_transmissivity]){
                                        primitive_values_dict[current_data_group][current_band_transmissivity][prim_values.first][1] = transmissivity;
                                    }
                                }
                            }else{
                                ImGui::InputFloat("##transmissivity", &primitive_values_dict[current_data_group][current_band_transmissivity][current_primitive][1]);
                            }
                        }else{
                            std::string transmissivity_prev = primitive_spectra_dict[current_data_group][current_primitive][1];
                            if (ImGui::BeginCombo("##transmissivity_combo", transmissivity_prev.c_str())){
                                for (auto &spectra : possible_spectra){
                                    bool is_spectra_selected = (primitive_spectra_dict[current_data_group][current_primitive][1] == spectra);
                                    if (ImGui::Selectable(spectra.c_str(), is_spectra_selected))
                                        primitive_spectra_dict[current_data_group][current_primitive][1] = spectra;
                                    if (is_spectra_selected)
                                        ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }
                            if (current_primitive == "All" && transmissivity_prev != primitive_spectra_dict[current_data_group][current_primitive][1]){
                                for (auto &prim_spectrum : primitive_spectra_dict[current_data_group]){
                                    primitive_spectra_dict[current_data_group][prim_spectrum.first][1] = primitive_spectra_dict[current_data_group][current_primitive][1];
                                }
                            }
                        }
                        ImGui::SameLine();
                        ImGui::TextUnformatted("%s",toggle_display_transmissivity.c_str());
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
                            for (std::string band : bandlabels_set_emissivity){
                                bool is_band_selected = (current_band_emissivity == band);
                                if (ImGui::Selectable(band.c_str(), is_band_selected))
                                    current_band_emissivity = band;
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
                                for (auto &prim_values : primitive_values_dict[current_data_group][current_band_emissivity]){
                                    primitive_values_dict[current_data_group][current_band_emissivity][prim_values.first][2] = emissivity;
                                }
                            }
                        }else{
                            ImGui::InputFloat("##emissivity", &primitive_values_dict[current_data_group][current_band_emissivity][current_primitive][2]);
                        }
                        ImGui::SameLine();
                        ImGui::Text("Manual Entry");
                    }

                    ImGui::EndTabItem();
                }
            } //RADIATION_MODEL
            #endif //RADIATION_MODEL
            if (enable_radiation){
                // RIG TAB
                if (ImGui::BeginTabItem("Rig")){
                    current_tab = "Rig";
                    if (ImGui::BeginCombo("##rig_combo", current_rig.c_str())){
                        for (auto rig_label : rig_labels_set){
                            bool is_rig_selected = (current_rig == rig_label);
                            if (ImGui::Selectable(rig_label.c_str(), is_rig_selected))
                                current_rig = rig_label;
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
                        rig_labels.push_back(new_rig_label);
                        rig_labels_set.insert(new_rig_label);
                        rig_camera_labels.push_back(rig_camera_labels[rig_dict[current_rig]]);
                        rig_light_labels.push_back(rig_light_labels[rig_dict[current_rig]]);
                        keypoint_frames.push_back(keypoint_frames[rig_dict[current_rig]]);
                        num_images_vec.push_back(num_images_vec[rig_dict[current_rig]]);
                        rig_colors.push_back(rig_colors[rig_dict[current_rig]]);
                        rig_position_noise.push_back(std::vector<distribution>{distribution{}, distribution{}, distribution{}});
                        rig_lookat_noise.push_back(std::vector<distribution>{distribution{}, distribution{}, distribution{}});
                        // current_rig = new_rig_label;
                        std::string parent = "rig";
                        pugi::xml_node rig_block = helios.child(parent.c_str());
                        pugi::xml_node new_rig_node = helios.append_copy(rig_block);
                        std::string name = "label";
                        pugi::xml_attribute node_label = new_rig_node.attribute(name.c_str());
                        node_label.set_value(new_rig_label.c_str());
                        current_rig = new_rig_label;
                    }
                    if (!current_rig.empty()){
                        // ##### UPDATE RIG ######//
                        if (ImGui::Button("Update Rig")){
                            updateRigs();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Delete Rig")){
                            deleteRig(current_rig);
                        }
                        // ##### RIG NAME ######//
                        ImGui::SetNextItemWidth(100);
                        std::string prev_rig_name = rig_labels[rig_dict[current_rig]];
                        ImGui::InputText("##rig_name", &rig_labels[rig_dict[current_rig]]);
                        if (rig_labels[rig_dict[current_rig]] != prev_rig_name && rig_dict.find(rig_labels[rig_dict[current_rig]]) == rig_dict.end() && !rig_labels[rig_dict[current_rig]].empty()){
                            int temp = rig_dict[current_rig];
                            current_rig = rig_labels[rig_dict[current_rig]];
                            std::map<std::string, int>::iterator current_rig_iter = rig_dict.find(prev_rig_name);
                            if (current_rig_iter != rig_dict.end()){
                                rig_dict.erase(current_rig_iter);
                            }
                            rig_dict[current_rig] = temp;
                            rig_labels_set.erase(prev_rig_name);
                            rig_labels_set.insert(rig_labels[rig_dict[current_rig]]);
                        } else{
                            rig_labels[rig_dict[current_rig]] = prev_rig_name;
                        }
                        ImGui::SameLine();
                        ImGui::Text("Rig Name");
                        // ####### WRITE DEPTH ####### //
                        bool write_depth_ = write_depth[rig_dict[current_rig]];
                        ImGui::Checkbox("Write Depth Images", &write_depth_);
                        write_depth[rig_dict[current_rig]] = write_depth_;
                        ImGui::SameLine();
                        bool write_norm_depth_ = write_norm_depth[rig_dict[current_rig]];
                        ImGui::Checkbox("Write Norm Depth Images", &write_norm_depth_);
                        write_norm_depth[rig_dict[current_rig]] = write_norm_depth_;
                        // ####### BOUNDING BOXES ####### //
                        if (ImGui::BeginPopup("multi_select_popup")) {
                            for (auto &box_pair : bounding_boxes) {
                                ImGui::Selectable(box_pair.first.c_str(), &box_pair.second, ImGuiSelectableFlags_DontClosePopups);
                            }
                            ImGui::EndPopup();
                        }
                        if (ImGui::Button("Select Bounding Box Objects")) {
                            ImGui::OpenPopup("multi_select_popup");
                        }
                        // ImGui::OpenPopupOnItemClick(("rig_position_noise_" + std::to_string(rig_dict[current_rig])).c_str(), ImGuiPopupFlags_MouseButtonLeft);
                        // Display selected items
                        ImGui::Text("Objects:");
                        int idx = 0;
                        for (auto &box_pair : bounding_boxes) {
                            if (box_pair.second){
                                ImGui::SameLine(), ImGui::Text("%i. %s", idx, box_pair.first.c_str());
                                idx++;
                            }
                        }
                        // ####### RIG COLOR ####### //
                        float col[3];
                        col[0] = rig_colors[rig_dict[current_rig]].r;
                        col[1] = rig_colors[rig_dict[current_rig]].g;
                        col[2] = rig_colors[rig_dict[current_rig]].b;
                        ImGui::ColorEdit3("##rig_color_edit", col);
                        updateColor(current_rig, "rig", col);
                        ImGui::SameLine();
                        ImGui::Text("Rig Color");
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
                        randomizePopup("camera_position_x_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_), createTaggedPtr(&camera_position_vec[rig_dict[current_rig]][current_cam_position_].x));
                        randomizerParams("camera_position_x_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_));
                        ImGui::OpenPopupOnItemClick(("randomize_camera_position_x_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##camera_position_y", &camera_position_vec[rig_dict[current_rig]][current_cam_position_].y);
                        randomizePopup("camera_position_y_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_), createTaggedPtr(&camera_position_vec[rig_dict[current_rig]][current_cam_position_].y));
                        randomizerParams("camera_position_y_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_));
                        ImGui::OpenPopupOnItemClick(("randomize_camera_position_y_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##camera_position_z", &camera_position_vec[rig_dict[current_rig]][current_cam_position_].z);
                        randomizePopup("camera_position_z_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_), createTaggedPtr(&camera_position_vec[rig_dict[current_rig]][current_cam_position_].z));
                        randomizerParams("camera_position_z_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_));
                        ImGui::OpenPopupOnItemClick(("randomize_camera_position_z_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Rig Position");
                        ImGui::SameLine();
                        ImGui::Button("Add Noise###position");
                        noisePopup("rig_position_noise_" + std::to_string(rig_dict[current_rig]), rig_lookat_noise[rig_dict[current_rig]]);
                        ImGui::OpenPopupOnItemClick(("rig_position_noise_" + std::to_string(rig_dict[current_rig])).c_str(), ImGuiPopupFlags_MouseButtonLeft);
                        // ####### CAMERA LOOKAT ####### //
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##camera_lookat_x", &camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].x);
                        randomizePopup("camera_lookat_x_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_), createTaggedPtr(&camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].x));
                        randomizerParams("camera_lookat_x_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_));
                        ImGui::OpenPopupOnItemClick(("randomize_camera_lookat_x_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##camera_lookat_y", &camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].y);
                        randomizePopup("camera_lookat_y_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_), createTaggedPtr(&camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].y));
                        randomizerParams("camera_lookat_y_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_));
                        ImGui::OpenPopupOnItemClick(("randomize_camera_lookat_y_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(60);
                        ImGui::InputFloat("##camera_lookat_z", &camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].z);
                        randomizePopup("camera_lookat_z_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_), createTaggedPtr(&camera_lookat_vec[rig_dict[current_rig]][current_cam_position_].z));
                        randomizerParams("camera_lookat_z_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_));
                        ImGui::OpenPopupOnItemClick(("randomize_camera_lookat_z_" + std::to_string(rig_dict[current_rig]) + std::to_string(current_cam_position_)).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Rig Lookat");
                        ImGui::SameLine();
                        ImGui::Button("Add Noise###lookat");
                        noisePopup("rig_lookat_noise_" + std::to_string(rig_dict[current_rig]), rig_lookat_noise[rig_dict[current_rig]]);
                        ImGui::OpenPopupOnItemClick(("rig_lookat_noise_" + std::to_string(rig_dict[current_rig])).c_str(), ImGuiPopupFlags_MouseButtonLeft);
                        // ####### NUMBER OF IMAGES ####### //
                        ImGui::SetNextItemWidth(80);
                        ImGui::InputInt("Total Number of Frames", &num_images_vec[rig_dict[current_rig]]);
                        num_images_vec[rig_dict[current_rig]] = std::max(num_images_vec[rig_dict[current_rig]], *std::max_element(keypoint_frames[rig_dict[current_rig]].begin(), keypoint_frames[rig_dict[(std::string) current_rig]].end()) + 1);
                    }
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

                    // ####### ADD BAND GROUP ####### //
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Band Groups:");
                    ImGui::SetWindowFontScale(1.0f);
                    dropDown("##band_group_combo", current_band_group, band_group_names);
                    ImGui::SameLine();
                    if (ImGui::Button("Add Band Group")){
                        std::string default_band_group_label = "band_group";
                        std::string new_band_group_label = "band_group_0";
                        int count = 0;
                        while (band_group_lookup.find(new_band_group_label) != band_group_lookup.end()){
                            count++;
                            new_band_group_label = default_band_group_label + "_" + std::to_string(count);
                        }
                        std::vector<std::string> new_band_group_vector;
                        new_band_group_vector.push_back("red");
                        new_band_group_vector.push_back("green");
                        new_band_group_vector.push_back("blue");
                        bandGroup new_band_group{new_band_group_vector, false, false};
                        band_group_lookup[new_band_group_label] = new_band_group;
                        band_group_names.insert(new_band_group_label);
                        current_band_group = new_band_group_label;
                    }
                    if (!current_band_group.empty()){
                        ImGui::SetNextItemWidth(100);
                        std::string prev_group_name = current_band_group;
                        ImGui::InputText("Group Name", &current_band_group);
                        if (current_band_group.empty() || band_group_lookup.find(current_band_group) != band_group_lookup.end()){
                            current_band_group = prev_group_name;
                        }
                        if (current_band_group != prev_group_name){
                            bandGroup temp = band_group_lookup[prev_group_name];
                            std::map<std::string, bandGroup>::iterator current_band_group_iter = band_group_lookup.find(prev_group_name);
                            if (current_band_group_iter != band_group_lookup.end()){
                                band_group_lookup.erase(current_band_group_iter);
                            }
                            band_group_lookup[current_band_group] = temp;
                            band_group_names.erase(prev_group_name);
                            band_group_names.insert(current_band_group);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Delete Group")){
                            band_group_names.erase(current_band_group);
                            band_group_lookup.erase(current_band_group);
                            current_band_group = "";
                        }
                        ImGui::Checkbox("Grayscale", &band_group_lookup[current_band_group].grayscale);
                        ImGui::SameLine();
                        ImGui::Checkbox("Norm", &band_group_lookup[current_band_group].norm);
                        // Band 1
                        ImGui::SetNextItemWidth(100);
                        dropDown("##band_1_combo", band_group_lookup[current_band_group].bands[0], bandlabels);
                        if (!band_group_lookup[current_band_group].grayscale){
                            // Band 2
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(100);
                            dropDown("##band_2_combo", band_group_lookup[current_band_group].bands[1], bandlabels);
                            // Band 3
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(100);
                            dropDown("##band_3_combo", band_group_lookup[current_band_group].bands[2], bandlabels);
                            ImGui::SameLine();
                            ImGui::Text("Select Bands");
                        } else{
                            ImGui::SameLine();
                            ImGui::Text("Select Band");
                        }
                    }
                    ImGui::SetWindowFontScale(1.25f);
                    ImGui::Text("Edit Camera:");
                    ImGui::SetWindowFontScale(1.0f);
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
                    if (ImGui::Button("Add New Camera")){
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
                        current_cam = new_cam_name;
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
                    ImGui::SetNextItemWidth(100);
                    dropDown("##camera_calibration_band", current_calibration_band, bandlabels);
                    ImGui::SameLine();
                    ImGui::Text("Band");
                    ImGui::SetNextItemWidth(250);
                    ImGui::SameLine();
                    dropDown("##camera_band_group_combo", camera_calibrations[camera_dict[current_cam]][current_calibration_band], possible_camera_calibrations);
                    ImGui::SameLine();
                    ImGui::Text("Calibration");
                    // ####### CAMERA CALIBRATION ####### //
                    // std::string prev_cam_calibration = camera_calibrations[camera_dict[current_cam]];
                    // if (ImGui::BeginCombo("##camera_calibration_combo", camera_calibrations[camera_dict[current_cam]].c_str())){
                    //     for (int n = 0; n < possible_camera_calibrations.size(); n++){
                    //         bool is_cam_calibration_selected = (camera_calibrations[camera_dict[current_cam]] == possible_camera_calibrations[n]);
                    //         if (ImGui::Selectable(possible_camera_calibrations[n].c_str(), is_cam_calibration_selected))
                    //             camera_calibrations[camera_dict[current_cam]] = possible_camera_calibrations[n];
                    //         if (is_cam_calibration_selected)
                    //             ImGui::SetItemDefaultFocus();
                    //     }
                    //     ImGui::EndCombo();
                    // }
                    // ImGui::SameLine();
                    // ImGui::Text("Camera Calibration");
                    // ####### CAMERA RESOLUTION ####### //
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputInt("##camera_resolution_x", &camera_resolutions[camera_dict[current_cam]].x);
                    randomizePopup("camera_resolution_x_" + std::to_string(camera_dict[current_cam]), createTaggedPtr(&camera_resolutions[camera_dict[current_cam]].x));
                    randomizerParams("camera_resolution_x_" + std::to_string(camera_dict[current_cam]));
                    ImGui::OpenPopupOnItemClick(("randomize_camera_resolution_x_" + std::to_string(camera_dict[current_cam])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputInt("##camera_resolution_y", &camera_resolutions[camera_dict[current_cam]].y);
                    randomizePopup("camera_resolution_y_" + std::to_string(camera_dict[current_cam]), createTaggedPtr(&camera_resolutions[camera_dict[current_cam]].y));
                    randomizerParams("camera_resolution_y_" + std::to_string(camera_dict[current_cam]));
                    ImGui::OpenPopupOnItemClick(("randomize_camera_resolution_y_" + std::to_string(camera_dict[current_cam])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                    ImGui::SameLine();
                    ImGui::Text("Camera Resolution");
                    // ####### FOCAL PLANE DISTANCE ####### //
                    ImGui::SetNextItemWidth(50);
                    ImGui::InputFloat("Focal Plane Distance", &focal_plane_distances[camera_dict[current_cam]]);
                    randomizePopup("focal_plane_distance_" + std::to_string(camera_dict[current_cam]), createTaggedPtr(&focal_plane_distances[camera_dict[current_cam]]));
                    randomizerParams("focal_plane_distance_" + std::to_string(camera_dict[current_cam]));
                    ImGui::OpenPopupOnItemClick(("randomize_focal_plane_distance_" + std::to_string(camera_dict[current_cam])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                    // ####### LENS DIAMETER ####### //
                    ImGui::SetNextItemWidth(50);
                    ImGui::InputFloat("Lens Diameter", &lens_diameters[camera_dict[current_cam]]);
                    randomizePopup("lens_diameter_" + std::to_string(camera_dict[current_cam]), createTaggedPtr(&lens_diameters[camera_dict[current_cam]]));
                    randomizerParams("lens_diameter_" + std::to_string(camera_dict[current_cam]));
                    ImGui::OpenPopupOnItemClick(("randomize_lens_diameter_" + std::to_string(camera_dict[current_cam])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                    // ####### FOV ASPECT RATIO ####### //
                    ImGui::SetNextItemWidth(50);
                    ImGui::InputFloat("FOV Aspect Ratio", &FOV_aspect_ratios[camera_dict[current_cam]]);
                    randomizePopup("FOV_aspect_ratio_" + std::to_string(camera_dict[current_cam]), createTaggedPtr(&FOV_aspect_ratios[camera_dict[current_cam]]));
                    randomizerParams("FOV_aspect_ratio_" + std::to_string(camera_dict[current_cam]));
                    ImGui::OpenPopupOnItemClick(("randomize_FOV_aspect_ratio_" + std::to_string(camera_dict[current_cam])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                    // ####### HFOV ####### //
                    ImGui::SetNextItemWidth(50);
                    ImGui::InputFloat("HFOV", &HFOVs[camera_dict[current_cam]]);
                    randomizePopup("HFOV_" + std::to_string(camera_dict[current_cam]), createTaggedPtr(&HFOVs[camera_dict[current_cam]]));
                    randomizerParams("HFOV_" + std::to_string(camera_dict[current_cam]));
                    ImGui::OpenPopupOnItemClick(("HFOV_" + std::to_string(camera_dict[current_cam])).c_str(), ImGuiPopupFlags_MouseButtonRight);
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
                        current_light = new_light_name;
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
                        randomizePopup("light_direction_x_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_direction_vec[light_dict[current_light]].x));
                        randomizerParams("light_direction_x_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_direction_x_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_direction_y", &light_direction_vec[light_dict[current_light]].y);
                        randomizePopup("light_direction_y_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_direction_vec[light_dict[current_light]].y));
                        randomizerParams("light_direction_y_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_direction_y_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_direction_z", &light_direction_vec[light_dict[current_light]].z);
                        randomizePopup("light_direction_z_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_direction_vec[light_dict[current_light]].z));
                        randomizerParams("light_direction_z_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_direction_z_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Light Direction");
                    }
                    // ####### LIGHT SOURCE FLUX ####### //
                    ImGui::SetNextItemWidth(90);
                    ImGui::InputFloat("##source_flux", &light_flux_vec[light_dict[current_light]]);
                    randomizePopup("source_flux_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_flux_vec[light_dict[current_light]]));
                    randomizerParams("source_flux_" + std::to_string(light_dict[current_light]));
                    ImGui::OpenPopupOnItemClick(("source_flux_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                    ImGui::SameLine();
                    ImGui::Text("Source Flux");
                    // radiation->setSourceFlux(light_UUID, band, flux_value);
                    // ####### LIGHT ROTATION ####### //
                    if (light_types[light_dict[current_light]] == "disk" ||
                        light_types[light_dict[current_light]] == "rectangle"){
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_rotation_x", &light_rotation_vec[light_dict[current_light]].x);
                        randomizePopup("light_rotation_x_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_rotation_vec[light_dict[current_light]].x));
                        randomizerParams("light_rotation_x_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_rotation_x_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_rotation_y", &light_rotation_vec[light_dict[current_light]].y);
                        randomizePopup("light_rotation_y_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_rotation_vec[light_dict[current_light]].y));
                        randomizerParams("light_rotation_y_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_rotation_y_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_rotation_z", &light_rotation_vec[light_dict[current_light]].z);
                        randomizePopup("light_rotation_z_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_rotation_vec[light_dict[current_light]].z));
                        randomizerParams("light_rotation_z_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_rotation_z_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Light Rotation");
                        }
                    // ####### LIGHT SIZE ####### //
                    if (light_types[light_dict[current_light]] == "rectangle"){
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_size_x", &light_size_vec[light_dict[current_light]].x);
                        randomizePopup("light_size_x_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_size_vec[light_dict[current_light]].x));
                        randomizerParams("light_size_x_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_size_x_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_size_y", &light_size_vec[light_dict[current_light]].y);
                        randomizePopup("light_size_y_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_size_vec[light_dict[current_light]].y));
                        randomizerParams("light_size_y_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_size_y_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
                        ImGui::SameLine();
                        ImGui::Text("Light Size");
                    }
                    // ####### LIGHT RADIUS ####### //
                    if (light_types[light_dict[current_light]] == "disk" ||
                        light_types[light_dict[current_light]] == "sphere"){
                        ImGui::SetNextItemWidth(90);
                        ImGui::InputFloat("##light_radius", &light_radius_vec[light_dict[current_light]]);
                        randomizePopup("light_radius_" + std::to_string(light_dict[current_light]), createTaggedPtr(&light_radius_vec[light_dict[current_light]]));
                        randomizerParams("light_radius_" + std::to_string(light_dict[current_light]));
                        ImGui::OpenPopupOnItemClick(("light_radius_" + std::to_string(light_dict[current_light])).c_str(), ImGuiPopupFlags_MouseButtonRight);
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
    // *** Latitude *** //
    xmlSetValue("latitude", "helios", latitude);
    xmlSetDistribution("latitude_dist", "helios", distributions[distribution_dict["latitude"]]);
    // *** Longitude *** //
    xmlSetValue("longitude", "helios", longitude);
    xmlSetDistribution("longitude_dist", "helios", distributions[distribution_dict["longitude"]]);
    // *** UTC Offset *** //
    xmlSetValue("UTC_offset", "helios", UTC_offset);
    xmlSetDistribution("UTC_offset_dist", "helios", distributions[distribution_dict["UTC_offset"]]);
    // *** CSV Weather File *** //
    xmlSetValue("csv_weather_file", "helios", csv_weather_file);
    // *** Domain Origin *** //
    xmlSetValue("domain_origin", "helios", domain_origin);
    xmlSetDistribution("domain_origin_x_dist", "helios", distributions[distribution_dict["domain_origin_x"]]);
    xmlSetDistribution("domain_origin_y_dist", "helios", distributions[distribution_dict["domain_origin_y"]]);
    xmlSetDistribution("domain_origin_z_dist", "helios", distributions[distribution_dict["domain_origin_z"]]);
    // *** Domain Extent *** //
    xmlSetValue("domain_extent", "helios", domain_extent);
    xmlSetDistribution("domain_extent_x_dist", "helios", distributions[distribution_dict["domain_extent_x"]]);
    xmlSetDistribution("domain_extent_y_dist", "helios", distributions[distribution_dict["domain_extent_y"]]);
    // *** Ground Resolution *** //
    xmlSetValue("ground_resolution", "helios", ground_resolution);
    xmlSetDistribution("ground_resolution_x_dist", "helios", distributions[distribution_dict["ground_resolution_x"]]);
    xmlSetDistribution("ground_resolution_y_dist", "helios", distributions[distribution_dict["ground_resolution_y"]]);
    // *** Ground Texture File *** //
    xmlSetValue("ground_texture_file", "helios", ground_texture_file);
    // *** Camera XML Library File *** //
    xmlSetValues("camera_xml_library_file", "helios", camera_xml_library_files);
    // *** Light XML Library File *** //
    xmlSetValues("light_xml_library_file", "helios", light_xml_library_files);
    // OBJECT BLOCK
    // Delete from XML doc
    helios = xmldoc.child("helios");
    pugi::xml_node node;
    node = helios.child("object");
    if (node){
        node.parent().remove_child(node);
    }
    // Refresh lists
    obj_names.clear();
    obj_files.clear();
    obj_positions.clear();
    obj_orientations.clear();
    obj_scales.clear();
    obj_colors.clear();
    obj_data_groups.clear();
    for (std::string obj_name : obj_names_set){
        object curr_obj = objects_dict[obj_name];
        obj_names.push_back(curr_obj.name);
        obj_files.push_back(curr_obj.file);
        obj_positions.push_back(curr_obj.position);
        obj_orientations.push_back(curr_obj.orientation);
        obj_scales.push_back(curr_obj.scale);
        obj_colors.push_back(curr_obj.color);
        obj_data_groups.push_back(curr_obj.data_group);
    }
    setNodeLabels("label", "object", obj_names_set);
    xmlSetValues("file", "object", obj_files, obj_names_dict);
    xmlSetValues("position", "object", obj_positions, obj_names_dict);
    xmlSetValues("orientation", "object", obj_orientations, obj_names_dict);
    xmlSetValues("scale", "object", obj_scales, obj_names_dict);
    xmlSetValues("color", "object", obj_colors, obj_names_dict);
    xmlSetValues("data_group", "object", obj_data_groups, obj_names_dict);
    // CANOPY BLOCK
    // Delete from XML doc
    helios = xmldoc.child("helios");
    node = helios.child("canopy");
    if (node){
        node.parent().remove_child(node);
    }
    setNodeLabels("label", "canopy_block", canopy_labels_set);
    xmlSetValue("canopy_origin", "canopy_block", canopy_origin);
    xmlSetValue("plant_count", "canopy_block", plant_count);
    xmlSetValue("plant_spacing", "canopy_block", plant_spacing);
    xmlSetValue("plant_library_name", "canopy_block", plant_library_name);
    xmlSetValue("plant_age", "canopy_block", plant_age);
    xmlSetValue("ground_clipping_height", "canopy_block", ground_clipping_height);
    xmlSetValues("canopy_origin", "canopy_block", canopy_origins, canopy_labels_dict);
    xmlSetValues("plant_count", "canopy_block", plant_counts, canopy_labels_dict);
    xmlSetValues("plant_spacing", "canopy_block", plant_spacings, canopy_labels_dict);
    xmlSetValues("plant_library_name", "canopy_block", plant_library_names, canopy_labels_dict);
    xmlSetValues("plant_age", "canopy_block", plant_ages, canopy_labels_dict);
    xmlSetValues("ground_clipping_height", "canopy_block", ground_clipping_heights, canopy_labels_dict);
    xmlSetValues("data_group", "canopy", canopy_data_groups, canopy_labels_dict);
    // RIG BLOCK
    // Delete from XML doc
    helios = xmldoc.child("helios");
    node = helios.child("rig");
    if (node){
        node.parent().remove_child(node);
    }
    setNodeLabels("label", "rig", rig_labels_set);
    xmlSetValues("color", "rig", rig_colors, rig_dict);
    // xmlSetValue("camera_position", "rig", camera_position);
    // xmlSetValue("camera_lookat", "rig", camera_lookat);
    xmlSetValue("camera_label", "rig", camera_label);
    // xmlSetValues("camera_position", "rig", camera_positions);
    xmlSetValues("camera_position", "rig", camera_position_vec, rig_dict);
    // xmlSetValues("camera_lookat", "rig", camera_lookats);
    xmlSetValues("camera_lookat", "rig", camera_lookat_vec, rig_dict);
    // xmlSetValues("camera_label", "rig", camera_labels);
    xmlSetValues("camera_label", "rig", rig_camera_labels, rig_dict);
    setKeypoints("keypoint", "camera_position", keypoint_frames);
    xmlSetValues("images", "rig", num_images_vec, rig_dict);
    std::vector<int> write_depth_{};
    std::vector<int> write_norm_depth_{};
    for (int i = 0; i < write_depth.size(); i++){
        if (write_depth[i]){
            write_depth_.push_back(1);
        } else{
            write_depth_.push_back(0);
        }
        if (write_norm_depth[i]){
            write_norm_depth_.push_back(1);
        } else{
            write_norm_depth_.push_back(0);
        }
    }
    xmlSetValues("depth", "rig", write_depth_, rig_dict);
    xmlSetValues("normdepth", "rig", write_norm_depth_, rig_dict);
    // CAMERA BLOCK
    setNodeLabels("label", "camera", camera_names_set);
    xmlSetValue("camera_resolution", "camera", camera_resolution);
    xmlSetValue("focal_plane_distance", "camera", focal_plane_distance);
    xmlSetValue("lens_diameter", "camera", lens_diameter);
    xmlSetValue("FOV_aspect_ratio", "camera", FOV_aspect_ratio);
    xmlSetValue("HFOV", "camera", HFOV);
    xmlSetValues("camera_resolution", "camera", camera_resolutions, camera_dict);
    for (std::string& band : bandlabels){
        for (std::string& camera : camera_names){
            xmlSetValue("camera_calibration_" + band, "camera", camera_calibrations[camera_dict[camera]][band]);
        }
    }
    xmlSetValues("focal_plane_distance", "camera", focal_plane_distances, camera_dict);
    xmlSetValues("lens_diameter", "camera", lens_diameters, camera_dict);
    xmlSetValues("FOV_aspect_ratio", "camera", FOV_aspect_ratios, camera_dict);
    xmlSetValues("HFOV", "camera", HFOVs, camera_dict);
    // LIGHT BLOCK
    setNodeLabels("label", "light", light_names_set);
    xmlSetValues("light_type", "light", light_types, light_dict);
    xmlSetValues("light_spectra", "light", light_spectra, light_dict);
    xmlSetValues("light_direction", "light", light_direction_vec, light_dict);
    xmlSetValues("light_rotation", "light", light_rotation_vec, light_dict);
    xmlSetValues("light_size", "light", light_size_vec, light_dict);
    xmlSetValues("light_source_flux", "light", light_flux_vec, light_dict);
    xmlSetValues("light_radius", "light", light_radius_vec, light_dict);
    xmlSetValues("light_label", "rig", rig_light_labels, light_dict);
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
    for (std::string prim : primitive_names){
        if (prim == "All") continue;
        // Reflectivity
        if (primitive_continuous[prim][0]){
            xmlSetValue(prim + "_reflectivity_spectrum", "radiation", primitive_spectra[prim][0]);
        } else{
            xmlRemoveField(prim + "_reflectivity_spectrum", "radiation");
        }
        // Transmissivity
        if (primitive_continuous[prim][1]){
            xmlSetValue(prim + "_transmissivity_spectrum", "radiation", primitive_spectra[prim][1]);
        } else{
            xmlRemoveField(prim + "_transmissivity_spectrum", "radiation");
        }
        for (std::string band : bandlabels){
            if (band == "All") continue;
            xmlSetValue(prim + "_reflectivity", "radiation", primitive_values[band][prim][0]);
            xmlSetValue(prim + "_transmissivity", "radiation", primitive_values[band][prim][1]);
            xmlSetValue(prim + "_emissivity", "radiation", primitive_values[band][prim][2]);
        }
    }
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
    distribution curr_distribution = distribution{};
    // MAIN BLOCK
    // *** Latitude *** //
    xmlGetValue("latitude", "helios", latitude);
    curr_distribution = distribution{};
    xmlGetDistribution("latitude_dist", "helios", curr_distribution);
    applyDistribution("latitude", curr_distribution, createTaggedPtr(&latitude));
    // *** Longitude *** //
    xmlGetValue("longitude", "helios", longitude);
    curr_distribution = distribution{};
    xmlGetDistribution("longitude_dist", "helios", curr_distribution);
    applyDistribution("longitude", curr_distribution, createTaggedPtr(&longitude));
    // *** UTC Offset *** //
    xmlGetValue("UTC_offset", "helios", UTC_offset);
    curr_distribution = distribution{};
    xmlGetDistribution("UTC_offset_dist", "helios", curr_distribution);
    applyDistribution("UTC_offset", curr_distribution, createTaggedPtr(&UTC_offset));
    // *** CSV Weather File *** //
    xmlGetValue("csv_weather_file", "helios", csv_weather_file);
    // *** Domain Origin *** //
    xmlGetValue("domain_origin", "helios", domain_origin);
    curr_distribution = distribution{};
    xmlGetDistribution("domain_origin_x_dist", "helios", curr_distribution);
    applyDistribution("domain_origin_x", curr_distribution, createTaggedPtr(&domain_origin.x));
    curr_distribution = distribution{};
    xmlGetDistribution("domain_origin_y_dist", "helios", curr_distribution);
    applyDistribution("domain_origin_y", curr_distribution, createTaggedPtr(&domain_origin.y));
    curr_distribution = distribution{};
    xmlGetDistribution("domain_origin_z_dist", "helios", curr_distribution);
    applyDistribution("domain_origin_z", curr_distribution, createTaggedPtr(&domain_origin.z));
    // *** Domain Extent *** //
    xmlGetValue("domain_extent", "helios", domain_extent);
    curr_distribution = distribution{};
    xmlGetDistribution("domain_extent_x_dist", "helios", curr_distribution);
    applyDistribution("domain_extent_x", curr_distribution, createTaggedPtr(&domain_extent.x));
    curr_distribution = distribution{};
    xmlGetDistribution("domain_extent_y_dist", "helios", curr_distribution);
    applyDistribution("domain_extent_y", curr_distribution, createTaggedPtr(&domain_extent.y));
    // *** Ground Resolution *** //
    xmlGetValue("ground_resolution", "helios", ground_resolution);
    curr_distribution = distribution{};
    xmlGetDistribution("ground_resolution_x_dist", "helios", curr_distribution);
    applyDistribution("ground_resolution_x", curr_distribution, createTaggedPtr(&ground_resolution.x));
    curr_distribution = distribution{};
    xmlGetDistribution("ground_resolution_y_dist", "helios", curr_distribution);
    applyDistribution("ground_resolution_y", curr_distribution, createTaggedPtr(&ground_resolution.y));
    // *** Ground Texture File *** //
    xmlGetValue("ground_texture_file", "helios", ground_texture_file);
    // *** Camera XML Library Files *** //
    xmlGetValues("camera_xml_library_file", "helios", camera_xml_library_files);
    possible_camera_calibrations.clear();
    for (auto &xml_library_file : camera_xml_library_files){
        if( xml_library_file.empty() || !std::filesystem::exists(xml_library_file) ){
            continue;
        }
        std::vector<std::string> current_camera_file = get_xml_node_values(xml_library_file, "label", "globaldata_vec2");
        possible_camera_calibrations.insert(possible_camera_calibrations.end(), current_camera_file.begin(), current_camera_file.end());
    }
    // *** Light XML Library Files *** //
    xmlGetValues("light_xml_library_file", "helios", light_xml_library_files);
    possible_light_spectra.clear();
    for (auto &xml_library_file : light_xml_library_files){
        if( xml_library_file.empty() || !std::filesystem::exists(xml_library_file) ){
            continue;
        }
        std::vector<std::string> current_light_file = get_xml_node_values(xml_library_file, "label", "globaldata_vec2");
        possible_light_spectra.insert(possible_light_spectra.end(), current_light_file.begin(), current_light_file.end());
    }
    // OBJECT BLOCK
    obj_names_dict.clear();
    obj_names_dict = getNodeLabels("label", "object", obj_names);
    obj_names_set = std::set<std::string>{obj_names.begin(), obj_names.end()};
    if (!obj_names.empty()) current_obj = obj_names[0];
    xmlGetValues("file", "object", obj_files);
    xmlGetValues("position", "object", obj_positions);
    xmlGetValues("orientation", "object", obj_orientations);
    xmlGetValues("scale", "object", obj_scales);
    prev_obj_positions = obj_positions;
    prev_obj_orientations = obj_orientations;
    prev_obj_scales = obj_scales;
    obj_data_groups.clear();
    xmlGetValues("data_group", "object", obj_data_groups);
    obj_colors.clear();
    xmlGetValues("color", "object", obj_colors);
    // CANOPY BLOCK
    canopy_labels.clear();
    canopy_labels_dict = getNodeLabels("label", "canopy_block", canopy_labels);
    for (auto canopy_label_ : canopy_labels){
        canopy_labels_set.insert(canopy_label_);
    }
    if (!canopy_labels.empty()) current_canopy = canopy_labels[0];
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
    for (int i = 0; i < plant_library_names.size(); i++){
        plant_library_names_verbose.push_back(plant_type_verbose_lookup[plant_library_names[i]]);
    }
    plant_ages.clear();
    xmlGetValues("plant_age", "canopy_block", plant_ages);
    ground_clipping_heights.clear();
    xmlGetValues("ground_clipping_height", "canopy_block", ground_clipping_heights);
    canopy_data_groups.clear();
    xmlGetValues("data_group", "canopy_block", canopy_data_groups);
    #ifdef ENABLE_RADIATION_MODEL
    // RIG BLOCK
    rig_labels.clear();
    rig_labels_set.clear();
    rig_dict = getNodeLabels("label", "rig", rig_labels);
    for (auto rig : rig_labels){
        rig_labels_set.insert(rig);
        rig_position_noise.push_back(std::vector<distribution>{distribution{}, distribution{}, distribution{}});
        rig_lookat_noise.push_back(std::vector<distribution>{distribution{}, distribution{}, distribution{}});
    }
    current_rig = rig_labels[0];
    xmlGetValues("color", "rig", rig_colors);
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
    write_depth.clear();
    write_norm_depth.clear();
    std::vector<int> write_depth_{};
    std::vector<int> write_norm_depth_{};
    xmlGetValues("depth", "rig", write_depth_);
    xmlGetValues("depth", "rig", write_norm_depth_);
    for (int i = 0; i < write_depth_.size(); i++){
        if (write_depth_[i] == 1){
            write_depth.push_back(true);
        } else{
            write_depth.push_back(false);
        }
        if (write_norm_depth_[i] == 1){
            write_norm_depth.push_back(true);
        } else{
            write_norm_depth.push_back(false);
        }
    }
    // CAMERA BLOCK
    camera_names.clear();
    camera_names_set.clear();
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
    for (std::string& camera : camera_names){
        camera_calibrations.push_back(std::map<std::string, std::string>{});
        for (std::string& band : bandlabels){
            xmlGetValue("camera_calibration_" + band, "camera", camera_calibrations[camera_dict[camera]][band]);
        }
        camera_names_set.insert(camera);
    }
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
    light_names_set.clear();
    light_dict = getNodeLabels("label", "light", light_names);
    for (auto &light : light_names){
        light_names_set.insert(light);
    }
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
        for (int i = 0; i < current_spectra_file.size(); i++){
            possible_spectra.insert(current_spectra_file[i]);
        }
    }
    xmlGetValue("solar_direct_spectrum", "radiation", solar_direct_spectrum);
    xmlGetValue("leaf_reflectivity_spectrum", "radiation", leaf_reflectivity_spectrum);
    xmlGetValue("leaf_transmissivity_spectrum", "radiation", leaf_transmissivity_spectrum);
    xmlGetValue("leaf_emissivity", "radiation", leaf_emissivity);
    xmlGetValue("ground_reflectivity_spectrum", "radiation", ground_reflectivity_spectrum);
    primitive_values.clear();
    for (std::string band : bandlabels){
        primitive_values[band];
        for (std::string prim : primitive_names){
            primitive_values[band][prim] = {ground_reflectivity, ground_transmissivity, ground_emissivity};
        }
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
    for (std::string prim : primitive_names){
        if (prim == "All") continue;
        std::string default_spectrum = "";
        // Reflectivity
        xmlGetValue(prim + "_reflectivity_spectrum", "radiation", default_spectrum);
        if (!default_spectrum.empty()){
            primitive_spectra[prim][0] = default_spectrum;
            primitive_continuous[prim][0] = true;
        } else{
            primitive_continuous[prim][0] = false;
        }
        // Transmissivity
        default_spectrum = "";
        xmlGetValue(prim + "_transmissivity_spectrum", "radiation", default_spectrum);
        if (!default_spectrum.empty()){
            primitive_spectra[prim][1] = default_spectrum;
            primitive_continuous[prim][1] = true;
        } else{
            primitive_continuous[prim][1] = false;
        }
        for (std::string band : bandlabels){
            if (band == "All") continue;
            xmlGetValue(prim + "_reflectivity", "radiation", primitive_values[band][prim][0]);
            xmlGetValue(prim + "_transmissivity", "radiation", primitive_values[band][prim][1]);
            xmlGetValue(prim + "_emissivity", "radiation", primitive_values[band][prim][2]);
        }
    }
    #endif
}

void ProjectBuilder::xmlGetValues(std::string xml_path){
    xml_input_file = xml_path;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }
    xmlGetValues();
}

void ProjectBuilder::objectTab(std::string curr_obj_name, int id){
    if (ImGui::Button("Update Object")){
        updateObject(curr_obj_name);
        refreshVisualization();
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete Object")){
        deleteObject(curr_obj_name);
    }
    ImGui::SetNextItemWidth(100);
    std::string prev_obj_name = objects_dict[curr_obj_name].name;
    ImGui::InputText(("##obj_name_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].name);
    if (objects_dict[curr_obj_name].name != prev_obj_name && objects_dict.find(curr_obj_name) == objects_dict.end() && !objects_dict[curr_obj_name].name.empty()){
        object temp_obj = objects_dict[curr_obj_name];
        curr_obj_name = objects_dict[curr_obj_name].name;
        std::map<std::string, object>::iterator delete_obj_iter = objects_dict.find(prev_obj_name);
        if (delete_obj_iter != objects_dict.end()){
            objects_dict.erase(delete_obj_iter);
        }

        objects_dict[objects_dict[curr_obj_name].name] = temp_obj;

        obj_names_set.erase(prev_obj_name);
        obj_names_set.insert(objects_dict[curr_obj_name].name);
    } else{
        objects_dict[curr_obj_name].name = prev_obj_name;
    }
    ImGui::SameLine();
    ImGui::Text("Object Name");
    // ####### OBJECT SCALE ####### //
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_scale_x_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].scale.x);
    randomizePopup("obj_scale_x_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].scale.x, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_scale_x_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_scale_x_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_scale_y_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].scale.y);
    randomizePopup("obj_scale_y_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].scale.y, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_scale_y_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_scale_y_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_scale_z_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].scale.z);
    randomizePopup("obj_scale_z_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].scale.z, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_scale_z_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_scale_z_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::Text("Object Scale");
    // ####### OBJECT POSITION ####### //
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_position_x_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].position.x);
    randomizePopup("obj_position_x_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].position.x, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_position_x_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_position_x_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_position_y_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].position.y);
    randomizePopup("obj_position_y_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].position.y, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_position_y_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_position_y_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_position_z_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].position.z);
    randomizePopup("obj_position_z_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].position.z, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_position_z_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_position_z_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::Text("Object Position");
    // ####### OBJECT ORIENTATION ####### //
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_orientation_x_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].orientation.x);
    randomizePopup("obj_orientation_x_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].orientation.x, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_orientation_x_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_orientation_x_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_orientation_y_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].orientation.y);
    randomizePopup("obj_orientation_y_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].orientation.y, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_orientation_y_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_orientation_y_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat(("##obj_orientation_z_" + std::to_string(id)).c_str(), &objects_dict[curr_obj_name].orientation.z);
    randomizePopup("obj_orientation_z_" + std::to_string(objects_dict[curr_obj_name].index), createTaggedPtr(&objects_dict[curr_obj_name].orientation.z, &objects_dict[curr_obj_name].is_dirty));
    randomizerParams("obj_orientation_z_" + std::to_string(objects_dict[curr_obj_name].index));
    ImGui::OpenPopupOnItemClick(("randomize_obj_orientation_z_" + std::to_string(objects_dict[curr_obj_name].index)).c_str(), ImGuiPopupFlags_MouseButtonRight);
    ImGui::SameLine();
    ImGui::Text("Object Orientation");
}

void ProjectBuilder::rigTab(std::string curr_rig_name, int id){
    ImGui::SetNextItemWidth(100);
    std::string prev_rig_name = rig_labels[rig_dict[curr_rig_name]];
    ImGui::InputText("##rig_name", &rig_labels[rig_dict[curr_rig_name]]);
    if (rig_labels[rig_dict[curr_rig_name]] != prev_rig_name && rig_labels_set.find(rig_labels[rig_dict[curr_rig_name]]) == rig_labels_set.end() && !rig_labels[rig_dict[curr_rig_name]].empty()){
        rig_labels_set.erase(prev_rig_name);
        int temp = rig_dict[curr_rig_name];
        rig_labels_set.insert(rig_labels[temp]);
        curr_rig_name = rig_labels[rig_dict[curr_rig_name]];
        std::map<std::string, int>::iterator current_rig_iter = rig_dict.find(prev_rig_name);
        if (current_rig_iter != rig_dict.end()){
            rig_dict.erase(current_rig_iter);
        }
        rig_dict[curr_rig_name] = temp;

        rig_labels_set.erase(prev_rig_name);
        rig_labels_set.insert(rig_labels[rig_dict[curr_rig_name]]);
    } else{
        rig_labels[rig_dict[curr_rig_name]] = prev_rig_name;
    }
    ImGui::SameLine();
    ImGui::Text("Rig Name");
    int current_cam_position_ = 0; // TODO: make this dynamic
    // ####### CAMERA POSITION ####### //
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##camera_position_x", &camera_position_vec[rig_dict[curr_rig_name]][current_cam_position_].x);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##camera_position_y", &camera_position_vec[rig_dict[curr_rig_name]][current_cam_position_].y);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##camera_position_z", &camera_position_vec[rig_dict[curr_rig_name]][current_cam_position_].z);
    ImGui::SameLine();
    ImGui::Text("Rig Position");
    // ####### CAMERA LOOKAT ####### //
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##camera_lookat_x", &camera_lookat_vec[rig_dict[curr_rig_name]][current_cam_position_].x);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##camera_lookat_y", &camera_lookat_vec[rig_dict[curr_rig_name]][current_cam_position_].y);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##camera_lookat_z", &camera_lookat_vec[rig_dict[curr_rig_name]][current_cam_position_].z);
    ImGui::SameLine();
    ImGui::Text("Rig Lookat");
}

void ProjectBuilder::canopyTab(std::string curr_canopy_name, int id){
    #ifdef ENABLE_PLANT_ARCHITECTURE
    if (ImGui::Button("Update Canopy")){
        updateCanopy(curr_canopy_name);
        refreshVisualization();
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete Canopy")){
        deleteCanopy(curr_canopy_name);
        refreshVisualization();
    }
    ImGui::SetNextItemWidth(100);
    std::string prev_canopy_name = canopy_labels[canopy_labels_dict[curr_canopy_name]];
    ImGui::InputText("##canopy_name", &canopy_labels[canopy_labels_dict[curr_canopy_name]]);
    if (canopy_labels[canopy_labels_dict[curr_canopy_name]] != prev_canopy_name && canopy_labels_dict.find(canopy_labels[canopy_labels_dict[curr_canopy_name]]) == canopy_labels_dict.end() && !canopy_labels[canopy_labels_dict[curr_canopy_name]].empty()){
        int temp = canopy_labels_dict[curr_canopy_name];
        curr_canopy_name = canopy_labels[canopy_labels_dict[curr_canopy_name]];
        std::map<std::string, int>::iterator current_canopy_iter = canopy_labels_dict.find(prev_canopy_name);
        if (current_canopy_iter != canopy_labels_dict.end()){
            canopy_labels_dict.erase(current_canopy_iter);
        }
        canopy_labels_dict[curr_canopy_name] = temp;

        canopy_labels_set.erase(prev_canopy_name);
        canopy_labels_set.insert(canopy_labels[canopy_labels_dict[curr_canopy_name]]);
    } else{
        canopy_labels[canopy_labels_dict[curr_canopy_name]] = prev_canopy_name;
    }
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##canopy_origin_x", &canopy_origins[canopy_labels_dict[curr_canopy_name]].x);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##canopy_origin_y", &canopy_origins[canopy_labels_dict[curr_canopy_name]].y);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::InputFloat("##canopy_origin_z", &canopy_origins[canopy_labels_dict[curr_canopy_name]].z);
    ImGui::SameLine();
    ImGui::Text("Canopy Origin");
    // ####### PLANT COUNT ####### //
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("##plant_count_x", &plant_counts[canopy_labels_dict[curr_canopy_name]].x);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("##plant_count_y", &plant_counts[canopy_labels_dict[curr_canopy_name]].y);
    ImGui::SameLine();
    ImGui::Text("Plant Count");
    // ####### PLANT SPACING ####### //
    ImGui::SetNextItemWidth(50);
    ImGui::InputFloat("##plant_spacing_x", &plant_spacings[canopy_labels_dict[curr_canopy_name]].x);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    ImGui::InputFloat("##plant_spacing_y", &plant_spacings[canopy_labels_dict[curr_canopy_name]].y);
    ImGui::SameLine();
    ImGui::Text("Plant Spacing");
    // ####### PLANT LIBRARY NAME ####### //
    ImGui::SetNextItemWidth(80);
    ImGui::InputText("Plant Library", &plant_library_names[canopy_labels_dict[curr_canopy_name]]);
    // ####### PLANT AGE ####### //
    ImGui::SetNextItemWidth(80);
    ImGui::InputFloat("Plant Age", &plant_ages[canopy_labels_dict[curr_canopy_name]]);
    // ####### GROUND CLIPPING HEIGHT ####### //
    ImGui::SetNextItemWidth(80);
    ImGui::InputFloat("Ground Clipping Height", &ground_clipping_heights[canopy_labels_dict[curr_canopy_name]]);
    #endif //PLANT_ARCHITECTURE
}

void ProjectBuilder::saveCanopy(std::string file_name, std::vector<uint> canopy_ID_vec, vec3 position, std::string file_extension) const{
    #ifdef ENABLE_PLANT_ARCHITECTURE
    std::vector<std::string> primitive_data_vec = {"object_label"};
    std::vector<uint> canopy_primID_vec;
    std::vector<uint> canopy_objID_vec;
    // plantarchitecture->getAllPlantObjectIDs(plantID)
    for (int i = 0; i < canopy_ID_vec.size(); i++){
        std::vector<uint> canopy_prim_UUIDs =  plantarchitecture->getAllPlantUUIDs(canopy_ID_vec[i]);
        std::vector<uint> canopy_obj_UUIDs = plantarchitecture->getAllPlantObjectIDs(canopy_ID_vec[i]);
        canopy_primID_vec.insert(canopy_primID_vec.end(), canopy_prim_UUIDs.begin(), canopy_prim_UUIDs.end());
        canopy_objID_vec.insert(canopy_objID_vec.end(), canopy_obj_UUIDs.begin(), canopy_obj_UUIDs.end());
    }
    for (uint objID : canopy_objID_vec){
        context->translateObject(objID, -position);
    }
    if (file_extension == "obj"){
        context->writeOBJ(file_name, canopy_primID_vec, primitive_data_vec, true);
    } else if (file_extension == "ply"){
        // context->writePLY(file_name, obj_UUID_vec, primitive_data_vec);
    }
    for (uint objID : canopy_objID_vec){
        context->translateObject(objID, position);
    }
    #endif
}

void ProjectBuilder::saveCanopy(std::string file_name_base, std::vector<uint> canopy_ID_vec, std::vector<helios::vec3> positions, std::string file_extension) const{
    #ifdef ENABLE_PLANT_ARCHITECTURE
    std::vector<std::string> primitive_data_vec = {"object_label"};
    std::vector<std::vector<uint>> canopy_primID_vec;
    std::vector<std::vector<uint>> canopy_objID_vec;
    // plantarchitecture->getAllPlantObjectIDs(plantID)
    for (int i = 0; i < canopy_ID_vec.size(); i++){
        std::vector<uint> canopy_prim_UUIDs =  plantarchitecture->getAllPlantUUIDs(canopy_ID_vec[i]);
        std::vector<uint> canopy_obj_UUIDs = plantarchitecture->getAllPlantObjectIDs(canopy_ID_vec[i]);
        canopy_primID_vec.push_back(canopy_prim_UUIDs);
        canopy_objID_vec.push_back(canopy_obj_UUIDs);
    }
    for (int i = 0; i < canopy_objID_vec.size(); i++){
        for (uint objID : canopy_objID_vec[i]){
            context->translateObject(objID, -positions[i]);
        }
    }
    std::filesystem::path file_path(file_name_base);
    std::string ext = file_path.extension().string();
    file_path.replace_extension("");
    std::string file_name = file_path.string();
    if (file_extension == "obj"){
        for (int i = 0; i < canopy_primID_vec.size(); i++){
            context->writeOBJ(file_name + "_" + std::to_string(i) + ext, canopy_primID_vec[i], primitive_data_vec, true);
        }
    } else if (file_extension == "ply"){
        // context->writePLY(file_name, obj_UUID_vec, primitive_data_vec);
    }
    for (int i = 0; i < canopy_objID_vec.size(); i++){
        for (uint objID : canopy_objID_vec[i]){
            context->translateObject(objID, positions[i]);
        }
    }
    #endif
}

void ProjectBuilder::addBand(std::string label, float wavelength_min, float wavelength_max, bool enable_emission){
#ifdef ENABLE_RADIATION_MODEL
    if (label.empty()){
        std::cout << "Failed to add band. Please specify a band label." << std::endl;
        return;
    }
    if (bandlabels_set.find(label) != bandlabels_set.end()){
        std::cout << "Failed to add band. Band with the specified name already exists." << std::endl;
        return;
    }
    if (wavelength_min > wavelength_max){
        std::cout << "Failed to add band. Invalid wavelength minimum and maximum." << std::endl;
        return;
    }
    bandlabels.push_back(label);
    bandlabels_set.insert(label);
    primitive_values[label] = {{"All", {reflectivity, transmissivity, emissivity}},
                                 {"ground", {ground_reflectivity, ground_transmissivity, ground_emissivity}},
                                 {"leaf", {leaf_reflectivity, leaf_transmissivity, leaf_emissivity}},
                                 {"petiolule", {petiolule_reflectivity, petiolule_transmissivity, petiolule_emissivity}},
                                 {"petiole", {petiole_reflectivity, petiole_transmissivity, petiole_emissivity}},
                                 {"internode", {internode_reflectivity, internode_transmissivity, internode_emissivity}},
                                 {"peduncle", {peduncle_reflectivity, peduncle_transmissivity, peduncle_emissivity}},
                                 {"petal", {petal_reflectivity, petal_transmissivity, petal_emissivity}},
                                 {"pedicel", {pedicel_reflectivity, pedicel_transmissivity, pedicel_emissivity}},
                                 {"fruit", {fruit_reflectivity, fruit_transmissivity, fruit_emissivity}}};
    for (auto &primitive_values_pair : primitive_values_dict){
        primitive_values_dict[primitive_values_pair.first][label] = {{"All", {reflectivity, transmissivity, emissivity}},
                                 {"ground", {ground_reflectivity, ground_transmissivity, ground_emissivity}},
                                 {"leaf", {leaf_reflectivity, leaf_transmissivity, leaf_emissivity}},
                                 {"petiolule", {petiolule_reflectivity, petiolule_transmissivity, petiolule_emissivity}},
                                 {"petiole", {petiole_reflectivity, petiole_transmissivity, petiole_emissivity}},
                                 {"internode", {internode_reflectivity, internode_transmissivity, internode_emissivity}},
                                 {"peduncle", {peduncle_reflectivity, peduncle_transmissivity, peduncle_emissivity}},
                                 {"petal", {petal_reflectivity, petal_transmissivity, petal_emissivity}},
                                 {"pedicel", {pedicel_reflectivity, pedicel_transmissivity, pedicel_emissivity}},
                                 {"fruit", {fruit_reflectivity, fruit_transmissivity, fruit_emissivity}}};
    }
    radiation->addRadiationBand(label,wavelength_min,wavelength_max);
    // radiation->addRadiationBand(label);
    if (!enable_emission){
        radiation->disableEmission(label);
    } else{
        bandlabels_set_emissivity.insert(label);
    }
    radiation->setDirectRayCount(label, direct_ray_count);
    direct_ray_count_dict.insert({label, direct_ray_count});
    radiation->setDiffuseRayCount(label, diffuse_ray_count);
    diffuse_ray_count_dict.insert({label, diffuse_ray_count});
    radiation->setScatteringDepth(label, scattering_depth);
    scattering_depth_dict.insert({label, scattering_depth});
#endif
}

void ProjectBuilder::addBand(std::string label, bool enable_emission){
#ifdef ENABLE_RADIATION_MODEL
    if (label.empty()){
        std::cout << "Failed to add band. Please specify a band label." << std::endl;
        return;
    }
    if (bandlabels_set.find(label) != bandlabels_set.end()){
        std::cout << "Failed to add band. Band with the specified name already exists." << std::endl;
        return;
    }
    bandlabels.push_back(label);
    bandlabels_set.insert(label);
    primitive_values[label] = {{"All", {reflectivity, transmissivity, emissivity}},
                                 {"ground", {ground_reflectivity, ground_transmissivity, ground_emissivity}},
                                 {"leaf", {leaf_reflectivity, leaf_transmissivity, leaf_emissivity}},
                                 {"petiolule", {petiolule_reflectivity, petiolule_transmissivity, petiolule_emissivity}},
                                 {"petiole", {petiole_reflectivity, petiole_transmissivity, petiole_emissivity}},
                                 {"internode", {internode_reflectivity, internode_transmissivity, internode_emissivity}},
                                 {"peduncle", {peduncle_reflectivity, peduncle_transmissivity, peduncle_emissivity}},
                                 {"petal", {petal_reflectivity, petal_transmissivity, petal_emissivity}},
                                 {"pedicel", {pedicel_reflectivity, pedicel_transmissivity, pedicel_emissivity}},
                                 {"fruit", {fruit_reflectivity, fruit_transmissivity, fruit_emissivity}}};
    for (auto &primitive_values_pair : primitive_values_dict){
        primitive_values_dict[primitive_values_pair.first][label] = {{"All", {reflectivity, transmissivity, emissivity}},
                                 {"ground", {ground_reflectivity, ground_transmissivity, ground_emissivity}},
                                 {"leaf", {leaf_reflectivity, leaf_transmissivity, leaf_emissivity}},
                                 {"petiolule", {petiolule_reflectivity, petiolule_transmissivity, petiolule_emissivity}},
                                 {"petiole", {petiole_reflectivity, petiole_transmissivity, petiole_emissivity}},
                                 {"internode", {internode_reflectivity, internode_transmissivity, internode_emissivity}},
                                 {"peduncle", {peduncle_reflectivity, peduncle_transmissivity, peduncle_emissivity}},
                                 {"petal", {petal_reflectivity, petal_transmissivity, petal_emissivity}},
                                 {"pedicel", {pedicel_reflectivity, pedicel_transmissivity, pedicel_emissivity}},
                                 {"fruit", {fruit_reflectivity, fruit_transmissivity, fruit_emissivity}}};
    }
    radiation->addRadiationBand(label);
    // radiation->addRadiationBand(label);
    if (!enable_emission){
        radiation->disableEmission(label);
    } else{
        bandlabels_set_emissivity.insert(label);
    }
    radiation->setDirectRayCount(label, direct_ray_count);
    direct_ray_count_dict.insert({label, direct_ray_count});
    radiation->setDiffuseRayCount(label, diffuse_ray_count);
    diffuse_ray_count_dict.insert({label, diffuse_ray_count});
    radiation->setScatteringDepth(label, scattering_depth);
    scattering_depth_dict.insert({label, scattering_depth});
#endif
}


void ProjectBuilder::randomizePopup(std::string popup_name, taggedPtr ptr){
    std::string popup = "randomize_" + popup_name;
    if (ImGui::BeginPopup(popup.c_str())){
        ImGui::Text("Random Distribution");
        ImGui::SetNextItemWidth(150);
        if (ImGui::BeginCombo("##combo_distribution", current_distribution.c_str())){
            for (int n = 0; n < distribution_names.size(); n++){
                bool is_dist_selected = (current_distribution == distribution_names[n]);
                if (ImGui::Selectable(distribution_names[n].c_str(), is_dist_selected))
                    current_distribution = distribution_names[n];
                if (is_dist_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (current_distribution == "Normal (Gaussian)"){
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Mean", &curr_distribution_params[0]);
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Variance", &curr_distribution_params[1]);
        }
        if (current_distribution == "Uniform"){
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Lower Bound", &curr_distribution_params[0]);
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Upper Bound", &curr_distribution_params[1]);
        }
        if (current_distribution == "Weibull"){
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Shape (k)", &curr_distribution_params[0]);
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat(u8"Scale (\u03bb)", &curr_distribution_params[1]);
        }
        if (current_distribution != "N/A"){
            ImGui::Checkbox("Randomize for Every Image", &randomize_repeatedly);
        }
        if (ImGui::Button("Apply")){
            applyDistribution(popup_name, ptr);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}


void ProjectBuilder::noisePopup(std::string popup_name, std::vector<distribution>& dist_vec){
    std::string popup = popup_name;
    if (ImGui::BeginPopup(popup.c_str())){
        ImGui::Text("Add Random Noise Along Path");
        ImGui::SetNextItemWidth(100);
        if (ImGui::BeginCombo("##axis_combo", current_axis.c_str())){
            for (auto axis : possible_axes){
                bool is_axis_selected = (current_axis == axis);
                if (ImGui::Selectable(axis.c_str(), is_axis_selected))
                    current_axis = axis;
                if (is_axis_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::Text("Axis");
        ImGui::SetNextItemWidth(150);
        if (ImGui::BeginCombo("##combo_distribution", current_distribution.c_str())){
            for (int n = 0; n < distribution_names.size(); n++){
                bool is_dist_selected = (current_distribution == distribution_names[n]);
                if (ImGui::Selectable(distribution_names[n].c_str(), is_dist_selected))
                    current_distribution = distribution_names[n];
                if (is_dist_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::Text("Distribution");
        if (current_distribution == "Normal (Gaussian)"){
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Mean", &curr_distribution_params[0]);
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Variance", &curr_distribution_params[1]);
        }
        if (current_distribution == "Uniform"){
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Lower Bound", &curr_distribution_params[0]);
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Upper Bound", &curr_distribution_params[1]);
        }
        if (current_distribution == "Weibull"){
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat("Shape (k)", &curr_distribution_params[0]);
            ImGui::SetNextItemWidth(150);
            ImGui::InputFloat(u8"Scale (\u03bb)", &curr_distribution_params[1]);
        }
        int idx;
        if (current_axis == "X"){
            idx = 0;
        } else if (current_axis == "Y"){
            idx = 1;
        } else{
            idx = 2;
        }
        if (ImGui::Button("Apply")){
            distUnion dist_union{};
            if (current_distribution == "Normal (Gaussian)"){
                dist_union.normal = new std::normal_distribution<float>;
                *dist_union.normal = std::normal_distribution<float>(curr_distribution_params[0], curr_distribution_params[1]);
                dist_vec[idx].dist = dist_union;
                dist_vec[idx].flag = 0;
                dist_vec[idx].repeat = 0;
            }
            if (current_distribution == "Uniform"){
                dist_union.uniform = new std::uniform_real_distribution<float>;
                *dist_union.uniform = std::uniform_real_distribution<float>(curr_distribution_params[0], curr_distribution_params[1]);
                dist_vec[idx].dist = dist_union;
                dist_vec[idx].flag = 1;
                dist_vec[idx].repeat = 0;
            }
            if (current_distribution == "Weibull"){
                dist_union.weibull = new std::weibull_distribution<float>;
                *dist_union.weibull = std::weibull_distribution<float>(curr_distribution_params[0], curr_distribution_params[1]);
                dist_vec[idx].dist = dist_union;
                dist_vec[idx].flag = 2;
                dist_vec[idx].repeat = 0;
            }
            if (current_distribution == "N/A"){
                dist_vec[idx].flag = -1;
                dist_vec[idx].repeat = 0;
            }

            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}



void ProjectBuilder::applyDistribution(std::string var_name, taggedPtr ptr){
    if (current_distribution == "N/A"){
        distribution_types[var_name] = "N/A";
    }
    if (current_distribution == "Normal (Gaussian)"){
        std::normal_distribution<float> curr_dist_normal(curr_distribution_params[0],curr_distribution_params[1]);
        distribution_dict[var_name] = distributions.size();
        distributions.push_back(createDistribution(curr_dist_normal, randomize_repeatedly));
        distribution_types[var_name] = "Normal (Gaussian)";
    }
    if (current_distribution == "Uniform"){
        std::uniform_real_distribution<float> curr_dist_uniform(curr_distribution_params[0],curr_distribution_params[1]);
        distribution_dict[var_name] = distributions.size();
        distributions.push_back(createDistribution(curr_dist_uniform, randomize_repeatedly));
        distribution_types[var_name] = "Uniform";
    }
    if (current_distribution == "Weibull"){
        std::weibull_distribution<float> curr_dist_weibull(curr_distribution_params[0],curr_distribution_params[1]);
        distribution_dict[var_name] = distributions.size();
        distributions.push_back(createDistribution(curr_dist_weibull, randomize_repeatedly));
        distribution_types[var_name] = "Weibull";
    }
    distribution_params[var_name] = curr_distribution_params;
    randomized_variable_lookup[var_name] = ptr;
    sample(var_name);
}


void ProjectBuilder::applyDistribution(std::string var_name, distribution dist, taggedPtr ptr){
    if (dist.flag == -1){
        distribution_dict[var_name] = distributions.size();
        distributions.push_back(dist);
        distribution_types[var_name] = "N/A";
        distribution_params[var_name] = std::vector<float>{0.0, 0.0};
    }
    if (dist.flag == 0){
        distribution_dict[var_name] = distributions.size();
        distributions.push_back(dist);
        distribution_types[var_name] = "Normal (Gaussian)";
        distribution_params[var_name] = std::vector<float>{dist.dist.normal->mean(), dist.dist.normal->stddev()};
    }
    if (current_distribution == "Uniform"){
        distribution_dict[var_name] = distributions.size();
        distributions.push_back(dist);
        distribution_types[var_name] = "Uniform";
        distribution_params[var_name] = std::vector<float>{dist.dist.uniform->a(), dist.dist.uniform->b()};
    }
    if (current_distribution == "Weibull"){
        distribution_dict[var_name] = distributions.size();
        distributions.push_back(dist);
        distribution_types[var_name] = "Weibull";
        distribution_params[var_name] = std::vector<float>{dist.dist.weibull->a(), dist.dist.weibull->b()};
    }
    randomized_variable_lookup[var_name] = ptr;
    sample(var_name);
}


void ProjectBuilder::randomize(bool randomize_all){
    for (std::pair<std::string, int> dist_pair : distribution_dict){
        std::string var_name = dist_pair.first;
        distribution dist = distributions[dist_pair.second];
        if (randomize_all || dist.repeat){
            sample(var_name);
        }
    }
}


taggedPtr createTaggedPtr(float* ptr){
    digitPtr p;
    p.f = ptr;
    taggedPtr t{p, false};
    return t;
}


taggedPtr createTaggedPtr(int* ptr){
    digitPtr p;
    p.i = ptr;
    taggedPtr t{p, true};
    return t;
}


taggedPtr createTaggedPtr(float* ptr, bool* dirty){
    digitPtr p;
    p.f = ptr;
    taggedPtr t{p, false, dirty};
    return t;
}


taggedPtr createTaggedPtr(int* ptr, bool* dirty){
    digitPtr p;
    p.i = ptr;
    taggedPtr t{p, true, dirty};
    return t;
}


distribution createDistribution(const std::normal_distribution<float> &dist, bool randomize_repeat){
    distUnion u;
    u.normal = new std::normal_distribution<float>;
    *u.normal = dist;
    distribution d{u, 0, randomize_repeat};
    return d;
}

distribution createDistribution(const std::uniform_real_distribution<float> &dist, bool randomize_repeat){
    distUnion u;
    u.uniform = new std::uniform_real_distribution<float>;
    *u.uniform = dist;
    distribution d{u, 1, randomize_repeat};
    return d;
}

distribution createDistribution(const std::weibull_distribution<float> &dist, bool randomize_repeat){
    distUnion u;
    u.weibull = new std::weibull_distribution<float>;
    *u.weibull = dist;
    distribution d{u, 2, randomize_repeat};
    return d;
}

void ProjectBuilder::randomizerParams(std::string var_name){
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)){
        if (distribution_params.find(var_name) != distribution_params.end()){
            curr_distribution_params = distribution_params[var_name];
        } else{
            curr_distribution_params = std::vector<float>{0, 0};
        }
        if (distribution_types.find(var_name) != distribution_types.end()){
            current_distribution = distribution_types[var_name];
        } else{
            current_distribution = "N/A";
        }
        if (current_distribution != "N/A"){
            randomize_repeatedly = distributions[distribution_dict[var_name]].repeat;
        }
    }
}

void ProjectBuilder::sample(std::string var_name){
    if (distribution_types.find(var_name) == distribution_types.end() || distribution_types[var_name] == "N/A"){
        return;
    }
    distribution d = distributions[distribution_dict[var_name]];
    taggedPtr t = randomized_variable_lookup[var_name];
    float sampled_value;
    if (d.flag == 0){
        std::normal_distribution<float> normal = *d.dist.normal;
        sampled_value = normal(generator);
        if (t.isInt){
            *t.ptr.i = (int) sampled_value;
        } else{
            *t.ptr.f = sampled_value;
        }
    } else if (d.flag == 1){
        std::uniform_real_distribution<float> uniform = *d.dist.uniform;
        sampled_value = uniform(generator);
        if (t.isInt){
            *t.ptr.i = (int) sampled_value;
        } else{
            *t.ptr.f = sampled_value;
        }
    } else if (d.flag == 2){
        std::weibull_distribution<float> weibull = *d.dist.weibull;
        sampled_value = weibull(generator);
        if (t.isInt){
            *t.ptr.i = (int) sampled_value;
        } else{
            *t.ptr.f = sampled_value;
        }
    }
    // if (t.object.idx != -1){
    //     if (t.object.isCanopy){
    //         dirty_canopies.insert(t.object.idx);
    //     } else{
    //         dirty_objects.insert(t.object.idx);
    //     }
    // }
}


void ProjectBuilder::updateContext(){
    for (int canopy_idx : dirty_canopies){
        // updateCanopy(canopy_idx);
    }
    for (int object_idx : dirty_objects){
        updateObject(obj_names[object_idx]);
    }
    dirty_canopies.clear();
    dirty_objects.clear();
}


void ProjectBuilder::sampleAll(){
    for (std::pair<std::string, std::string> var_pair : distribution_types){
        sample(var_pair.first);
    }
}

void ProjectBuilder::outputConsole(){
    old_cout_stream_buf = std::cout.rdbuf();
    std::string buffer = captured_cout.str();
    ImGui::InputTextMultiline("##console", &buffer[0], buffer.size() + 1, ImVec2(-FLT_MIN,
                                ImGui::GetTextLineHeight() * 5), ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_AllowTabInput);
    std::cout.rdbuf(old_cout_stream_buf);
}

void ProjectBuilder::updateColor(std::string curr_obj, std::string obj_type, float* new_color){
    helios::RGBcolor *curr_color = nullptr;
    if (obj_type == "obj"){
        curr_color = &objects_dict[curr_obj].color;
    }
    if (obj_type == "rig" || obj_type == "arrow" || obj_type == "camera"){
        curr_color = &rig_colors[rig_dict[curr_obj]];
    }
    // if (curr_color->r == new_color[0] && curr_color->g == new_color[1] && curr_color->b == new_color[2]){
    //     return;
    // }
    curr_color->r = new_color[0];
    curr_color->g = new_color[1];
    curr_color->b = new_color[2];
    if (obj_type == "obj"){
        context->setPrimitiveColor(objects_dict[curr_obj].UUIDs, *curr_color);
    }
    if (obj_type == "rig"){
        if (arrow_dict.find(curr_obj) != arrow_dict.end()){
            for (std::vector<uint> &arrow : arrow_dict.at(curr_obj)){
                context->setPrimitiveColor(arrow, *curr_color);
            }
        }
        if (camera_models_dict.find(curr_obj) != camera_models_dict.end()){
            context->setPrimitiveColor(camera_models_dict.at(curr_obj), *curr_color);
        }
    }
    if (obj_type == "arrow"){
        if (arrow_dict.find(curr_obj) != arrow_dict.end()){
            for (std::vector<uint> &arrow : arrow_dict.at(curr_obj)){
                context->setPrimitiveColor(arrow, *curr_color);
            }
        }
    }
    if (obj_type == "camera"){
        if (camera_models_dict.find(curr_obj) != camera_models_dict.end()){
            context->setPrimitiveColor(camera_models_dict.at(curr_obj), *curr_color);
        }
    }
}


void ProjectBuilder::updateObject(std::string curr_obj){
    // Scale, rotate, and translate object
    if (objects_dict[curr_obj].use_texture_file && objects_dict[curr_obj].is_dirty){
        context->deletePrimitive(objects_dict[curr_obj].UUIDs);
        if( std::filesystem::path(objects_dict[curr_obj].file).extension() == ".obj" ){
            objects_dict[curr_obj].UUIDs = context->loadOBJ(objects_dict[curr_obj].file.c_str());
        } else if ( std::filesystem::path(objects_dict[curr_obj].file).extension() == ".ply" ){
            objects_dict[curr_obj].UUIDs = context->loadPLY(objects_dict[curr_obj].file.c_str());
        }
        context->scalePrimitive(objects_dict[curr_obj].UUIDs, objects_dict[curr_obj].scale);
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].orientation.x), "x");
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].orientation.y), "y");
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].orientation.z), "z");
        context->translatePrimitive(objects_dict[curr_obj].UUIDs, objects_dict[curr_obj].position);

        objects_dict[curr_obj].prev_scale = objects_dict[curr_obj].scale;
        objects_dict[curr_obj].prev_orientation = objects_dict[curr_obj].orientation;
        objects_dict[curr_obj].prev_position = objects_dict[curr_obj].position;
    }
    if (objects_dict[curr_obj].scale != objects_dict[curr_obj].prev_scale){
        vec3 obj_scale_;
        obj_scale_.x = objects_dict[curr_obj].scale.x / objects_dict[curr_obj].prev_scale.x;
        obj_scale_.y = objects_dict[curr_obj].scale.y / objects_dict[curr_obj].prev_scale.y;
        obj_scale_.z = objects_dict[curr_obj].scale.z / objects_dict[curr_obj].prev_scale.z;
        // context->scalePrimitiveAboutPoint();
        context->translatePrimitive(objects_dict[curr_obj].UUIDs, -objects_dict[curr_obj].prev_position); // translate back to origin
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, -deg2rad(objects_dict[curr_obj].prev_orientation.x), "x");
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, -deg2rad(objects_dict[curr_obj].prev_orientation.y), "y");
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, -deg2rad(objects_dict[curr_obj].prev_orientation.z), "z");

        context->scalePrimitive(objects_dict[curr_obj].UUIDs, obj_scale_);
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].prev_orientation.x), "x");
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].prev_orientation.y), "y");
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].prev_orientation.z), "z");
        context->translatePrimitive(objects_dict[curr_obj].UUIDs, objects_dict[curr_obj].prev_position); // restore translation
        objects_dict[curr_obj].prev_scale = objects_dict[curr_obj].scale;
    }
    if (objects_dict[curr_obj].orientation != objects_dict[curr_obj].prev_orientation){
        // context->rotatePrimitive(origin = prev_position, axis = (1,0,0));
        //rotate about x
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].orientation.x - objects_dict[curr_obj].prev_orientation.x), objects_dict[curr_obj].prev_position, make_vec3(1, 0, 0));
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].orientation.y - objects_dict[curr_obj].prev_orientation.y), objects_dict[curr_obj].prev_position, make_vec3(0, 1, 0));
        context->rotatePrimitive(objects_dict[curr_obj].UUIDs, deg2rad(objects_dict[curr_obj].orientation.z - objects_dict[curr_obj].prev_orientation.z), objects_dict[curr_obj].prev_position, make_vec3(0, 0, 1));
        objects_dict[curr_obj].prev_orientation = objects_dict[curr_obj].orientation;
    }
    if (objects_dict[curr_obj].position != objects_dict[curr_obj].prev_position){
        context->translatePrimitive(objects_dict[curr_obj].UUIDs, objects_dict[curr_obj].position - objects_dict[curr_obj].prev_position);
        objects_dict[curr_obj].prev_position = objects_dict[curr_obj].position;
    }
    objects_dict[curr_obj].prev_color = objects_dict[curr_obj].color;
    objects_dict[curr_obj].is_dirty = false;
}


void ProjectBuilder::updateRigs(){
    deleteArrows();
    arrow_dict.clear();
    updateArrows();
    deleteCameraModels();
    camera_models_dict.clear();
    updateCameraModels();
    // Update visualizer
    refreshVisualization();
}


void ProjectBuilder::deleteRig(std::string curr_rig){
    int delete_idx = rig_dict[curr_rig];
    rig_dict.erase(rig_dict.find(curr_rig));
    rig_labels_set.erase(curr_rig);
    updateRigs();
    if (!rig_labels_set.empty() && current_rig == curr_rig){
        current_rig = *rig_labels_set.begin();
    } else{
        current_rig = "";
    }
}



void ProjectBuilder::dropDown(std::string widget_name, std::string& selected, std::vector<std::string> choices){
    if (ImGui::BeginCombo(widget_name.c_str(), selected.c_str())){
        for (int n = 0; n < choices.size(); n++){
            bool is_selected = (selected == choices[n]);
            if (ImGui::Selectable(choices[n].c_str(), is_selected))
                selected = choices[n];
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}


void ProjectBuilder::dropDown(std::string widget_name, std::string& selected, std::set<std::string> choices){
    if (ImGui::BeginCombo(widget_name.c_str(), selected.c_str())){
        for (std::string choice : choices){
            bool is_selected = (selected == choice);
            if (ImGui::Selectable(choice.c_str(), is_selected))
                selected = choice;
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}

void ProjectBuilder::deleteCanopy(const std::string &canopy){
    #ifdef ENABLE_PLANT_ARCHITECTURE
    int delete_idx = canopy_labels_dict[canopy];
    for (auto plant_instance : canopy_IDs[delete_idx]){
        plantarchitecture->deletePlantInstance(plant_instance);
    }
    canopy_labels_dict.erase(canopy);
    canopy_labels_set.erase(canopy);
    if (!canopy_labels_set.empty() && current_canopy == canopy){
        current_canopy = *canopy_labels_set.begin();
    } else{
        current_canopy = "";
    }
    #endif
}


void ProjectBuilder::deleteObject(const std::string& obj){
    context->deletePrimitive(objects_dict[obj].UUIDs);
    refreshVisualization();
    objects_dict.erase(obj);
    obj_names_set.erase(obj);
    if (!obj_names_set.empty()){
        current_obj = *obj_names_set.begin();
    } else{
        current_obj = "";
    }
}



void ProjectBuilder::updateCanopy(const std::string &canopy){
    #ifdef ENABLE_PLANT_ARCHITECTURE
    int update_idx = canopy_labels_dict[canopy];
    for (auto plant_instance : canopy_IDs[update_idx]){
        plantarchitecture->deletePlantInstance(plant_instance);
    }
    plantarchitecture->loadPlantModelFromLibrary( plant_library_names[update_idx] );
    plantarchitecture->enableGroundClipping( ground_clipping_height );

    std::vector<uint> new_canopy_IDs = plantarchitecture->buildPlantCanopyFromLibrary( canopy_origins[update_idx], plant_spacings[update_idx],
                                                                                        plant_counts[update_idx], plant_ages[update_idx]);
    std::vector<vec3> curr_plant_locations = plantarchitecture->getPlantBasePosition(new_canopy_IDs);
    individual_plant_locations.push_back(curr_plant_locations);

    leaf_UUIDs = plantarchitecture->getAllLeafUUIDs();
    primitive_UUIDs["leaf"] = leaf_UUIDs;
    internode_UUIDs = plantarchitecture->getAllInternodeUUIDs();
    primitive_UUIDs["internode"] = internode_UUIDs;
    petiole_UUIDs = plantarchitecture->getAllPetioleUUIDs();
    primitive_UUIDs["petiole"] = petiole_UUIDs;
    peduncle_UUIDs = plantarchitecture->getAllPeduncleUUIDs();
    primitive_UUIDs["peduncle"] = peduncle_UUIDs;
    std::vector<uint> flower_UUIDs = plantarchitecture->getAllFlowerUUIDs();
    petal_UUIDs = context->filterPrimitivesByData(flower_UUIDs, "object_label", "petal");
    sepal_UUIDs = context->filterPrimitivesByData(flower_UUIDs, "object_label", "sepal");
    if( petal_UUIDs.empty() && sepal_UUIDs.empty() ){
        petal_UUIDs = flower_UUIDs;
        sepal_UUIDs.clear();
    }
    fruit_UUIDs = plantarchitecture->getAllFruitUUIDs();
    primitive_UUIDs["petal"] = petal_UUIDs;
    primitive_UUIDs["sepal"] = sepal_UUIDs;
    primitive_UUIDs["flower"] = flower_UUIDs;

    canopy_IDs[canopy_labels_dict[canopy]] = new_canopy_IDs;
    #endif
}



void ProjectBuilder::addCanopy(){
    std::string default_canopy_label = "canopy";
    std::string new_canopy_label = "canopy_0";
    int count = 0;
    while (canopy_labels_dict.find(new_canopy_label) != canopy_labels_dict.end()){
        count++;
        new_canopy_label = default_canopy_label + "_" + std::to_string(count);
    }
    canopy_labels_dict.insert({new_canopy_label, canopy_labels.size()});
    canopy_labels_set.insert(new_canopy_label);
    canopy_origins.push_back(canopy_origin);
    canopy_data_groups.push_back("");
    plant_counts.push_back(plant_count);
    plant_spacings.push_back(plant_spacing);
    plant_library_names.push_back(plant_library_name);
    plant_library_names_verbose.push_back(plant_library_name_verbose);
    plant_ages.push_back(plant_age);
    ground_clipping_heights.push_back(ground_clipping_height);
    canopy_labels.push_back(new_canopy_label);
    canopy_IDs.push_back(std::vector<unsigned int>{});
    current_canopy = new_canopy_label;
}


void ProjectBuilder::refreshVisualization(){
    const char* font_name = "LCD";
    visualizer->addTextboxByCenter("LOADING...", vec3(.5,.5,0), make_SphericalCoord(0, 0),
        RGB::red, 40, font_name, Visualizer::COORDINATES_WINDOW_NORMALIZED);
    visualizer->plotUpdate();
    visualizer->clearGeometry();
    visualizer->buildContextGeometry(context);
    if (enable_coordinate_axes){
        visualizer->addCoordinateAxes(helios::make_vec3(0,0,0.05), helios::make_vec3(1,1,1), "positive");
    }
    visualizer->plotUpdate();
}


void ProjectBuilder::recordPopup(){
    if (ImGui::BeginPopup("repeat_record")){
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("Number of Recordings", &num_recordings);
        ImGui::EndPopup();
    }
}


void ProjectBuilder::updateLocation(){
    Location location;
    location.latitude_deg = latitude;
    location.longitude_deg = longitude;
    location.UTC_offset = static_cast<float>(UTC_offset);
    context->setLocation(location);
}


void ProjectBuilder::updateGround(){
    context->deletePrimitive(primitive_UUIDs["ground"]);
    // uint ground_objID = context->addTileObject( domain_origin, domain_extent, nullptr, ground_resolution, ground_texture_file.c_str() );
    uint ground_objID;
    if( !ground_model_file.empty() && ground_flag == 2 && use_ground_texture ) {
        ground_UUIDs = context->loadOBJ(ground_model_file.c_str() );
        context->translatePrimitive( ground_UUIDs, domain_origin );
        ground_objID = context->addPolymeshObject( ground_UUIDs );
    }else if( !ground_texture_file.empty() && ground_flag == 1 && use_ground_texture ){
        #ifdef ENABLE_CANOPY_GENERATOR
        ground_UUIDs.clear();
        canopygenerator->buildGround( domain_origin, domain_extent, num_tiles, subpatches, ground_texture_file.c_str() );
        ground_UUIDs = canopygenerator->getGroundUUIDs();
        context->setPrimitiveData( ground_UUIDs, "twosided_flag", uint(0) );
        context->setGlobalData( "ground_UUIDs", HELIOS_TYPE_UINT, ground_UUIDs.size(), ground_UUIDs.data() );
        primitive_UUIDs["ground"] = ground_UUIDs;

        return;
        #else
        ground_objID = context->addTileObject( domain_origin, domain_extent, nullrotation, ground_resolution, ground_texture_file.c_str() );
        ground_UUIDs = context->getObjectPrimitiveUUIDs(ground_objID);
        #endif
    }else if( ground_flag == 1  && !use_ground_texture ){
        RGBcolor ground_color_;
        ground_color_.r = ground_color[0];
        ground_color_.g = ground_color[1];
        ground_color_.b = ground_color[2];

        #ifdef ENABLE_CANOPY_GENERATOR
        ground_UUIDs.clear();
        canopygenerator->buildGround( domain_origin, domain_extent, num_tiles, subpatches, ground_texture_file.c_str() );
        ground_UUIDs = canopygenerator->getGroundUUIDs();
        context->setPrimitiveColor(ground_UUIDs, ground_color_);
        context->setPrimitiveData( ground_UUIDs, "twosided_flag", uint(0) );
        context->setGlobalData( "ground_UUIDs", HELIOS_TYPE_UINT, ground_UUIDs.size(), ground_UUIDs.data() );
        primitive_UUIDs["ground"] = ground_UUIDs;

        return;
        #else
        ground_objID = context->addTileObject( domain_origin, domain_extent, nullrotation, ground_resolution, ground_color_ );
        ground_UUIDs = context->getObjectPrimitiveUUIDs(ground_objID);
        #endif
    }else if( ground_flag == 2  && !use_ground_texture ){
        RGBcolor ground_color_;
        ground_color_.r = ground_color[0];
        ground_color_.g = ground_color[1];
        ground_color_.b = ground_color[2];
        ground_UUIDs = context->loadOBJ(ground_model_file.c_str());
        ground_objID = context->addPolymeshObject( ground_UUIDs );
        context->setObjectColor(ground_objID, ground_color_);
    }
    // else {
    //     ground_objID = context->addTileObject(domain_origin, domain_extent, nullrotation, ground_resolution);
    //     ground_UUIDs = context->getObjectPrimitiveUUIDs(ground_objID);
    // }
    ground_UUIDs.clear();
    ground_UUIDs = context->getObjectPrimitiveUUIDs(ground_objID);
    context->setPrimitiveData( ground_UUIDs, "twosided_flag", uint(0) );
    primitive_UUIDs["ground"] = ground_UUIDs;
}


void ProjectBuilder::refreshVisualizationTypes(){
    visualization_types.clear();
    std::vector<uint> allUUIDs = context->getAllUUIDs();
    for (auto &UUID : allUUIDs){
        std::vector<std::string> primitiveData = context->listPrimitiveData(UUID);
        for (auto &data : primitiveData){
            visualization_types.insert(data);
        }
    }
}


std::string ProjectBuilder::shortenPath(std::string path_name){
    std::string shorten_path = path_name;
    for (char &c : shorten_path){
        if (c == '\\'){
            c = '/';
        }
    }
    size_t last_file = shorten_path.rfind('/');
    if (last_file != std::string::npos){
        shorten_path = shorten_path.substr(last_file + 1);
    }
    return shorten_path;
}


void ProjectBuilder::setBoundingBoxObjects(){
    bounding_boxes_map.clear();
    int idx = 0;
    for (auto &box_pair : bounding_boxes) {
        if (box_pair.second){
            bounding_boxes_map[box_pair.first] = idx;
            idx++;
        }
    }
    std::vector<uint> all_UUIDs = context->getAllUUIDs();
    context->clearPrimitiveData(all_UUIDs, "object_number");
    for (auto &UUID : all_UUIDs){
        if (!context->doesPrimitiveDataExist(UUID, "object_label")) continue;
        std::string obj_label;
        context->getPrimitiveData(UUID, "object_label", obj_label);
        if (bounding_boxes_map.find(obj_label) != bounding_boxes_map.end()){
            context->setPrimitiveData(UUID, "object_number", HELIOS_TYPE_UINT, 1, &bounding_boxes_map[obj_label]);
        }
    }
}


