#include "BuildGeometry/BuildGeometry.h"
#include "PlantArchitecture.h"

using namespace helios;

void BuildGeometry(const std::string &xml_input_file, PlantArchitecture *plant_architecture_ptr, Context *context_ptr, std::vector<std::vector<uint>> &canopy_UUID_vector, std::vector<std::vector<helios::vec3>> &individual_plant_locations) {

    std::vector<uint> ground_UUIDs;
    std::vector<uint> leaf_UUIDs;
    std::vector<uint> petiolule_UUIDs;
    std::vector<uint> petiole_UUIDs;
    std::vector<uint> internode_UUIDs;
    std::vector<uint> peduncle_UUIDs;
    std::vector<uint> petal_UUIDs;
    std::vector<uint> sepal_UUIDs;
    std::vector<uint> fruit_UUIDs;

    pugi::xml_document xmldoc;

    std::string xml_error_string;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }

    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;

    // ####### BUILDING THE GROUND ####### //

    vec2 domain_extent(10,10);
    bool domain_extent_read = false;
    node = helios.child("domain_extent");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'domain_extent'. Using default value of " << domain_extent << std::endl;
    }else {

        const char *domain_extent_str = node.child_value();
        if (!parse_vec2(domain_extent_str, domain_extent)) {
            helios_runtime_error("ERROR: Value given for 'domain_extent' could not be parsed.");
        }else if( domain_extent.x<=0 || domain_extent.y<=0 ){
            helios_runtime_error("ERROR: Value given for 'domain_extent' must be greater than 0.");
        }else{
            domain_extent_read = true;
        }

    }

    vec3 domain_origin(0,0,0);
    bool domain_origin_read = false;
    node = helios.child("domain_origin");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'domain_origin'. Using default value of " << domain_origin << std::endl;
    }else {

        const char *domain_origin_str = node.child_value();
        if (!parse_vec3(domain_origin_str, domain_origin)) {
            helios_runtime_error("ERROR: Value given for 'domain_origin' could not be parsed.");
        }else{
            domain_origin_read = true;
        }

    }

    int2 ground_resolution(1,1);
    bool ground_resolution_read = false;
    node = helios.child("ground_resolution");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'ground_resolution'. Using default value of " << ground_resolution << std::endl;
    }else {

        const char *ground_resolution_str = node.child_value();
        if (!parse_int2(ground_resolution_str, ground_resolution)) {
            helios_runtime_error("ERROR: Value given for 'ground_resolution' could not be parsed.");
        }else if( ground_resolution.x<=0 || ground_resolution.y<=0 ){
            helios_runtime_error("ERROR: Value given for 'ground_resolution' must be greater than 0.");
        }else{
            ground_resolution_read = true;
        }

    }

    RGBcolor ground_color;
    bool ground_color_read = false;
    node = helios.child("ground_color");
    if( !node.empty() ){

        const char *ground_color_str = node.child_value();
        if (!parse_RGBcolor(ground_color_str, ground_color)) {
            helios_runtime_error("ERROR: Value given for 'ground_color' could not be parsed.");
        }else{
            ground_color_read = true;
        }

    }

    std::string ground_texture_file;
    node = helios.child("ground_texture_file");
    if( !node.empty() ){

        const char *ground_texture_file_str = node.child_value();
        ground_texture_file = trim_whitespace(std::string(ground_texture_file_str));

        if( ground_texture_file.empty() ){
            helios_runtime_error("ERROR: Value given for 'ground_texture_file' is empty.");
        }

    }

    std::string ground_model_file;
    node = helios.child("ground_model_obj_file");
    if( !node.empty() ){

        const char *ground_model_file_str = node.child_value();
        ground_model_file = trim_whitespace(std::string(ground_model_file_str));

        std::string ext = getFileExtension(ground_model_file);

        if( ground_model_file.empty() ){
            helios_runtime_error("ERROR: Value given for 'ground_model_obj_file' is empty.");
        }else if( ext != ".obj" && ext != ".OBJ" ){
            helios_runtime_error("ERROR: File given for 'ground_model_obj_file' is not an .obj file.");
        }else if( !std::filesystem::exists(ground_model_file) ){
            helios_runtime_error("ERROR: File given for 'ground_model_obj_file' does not exist.");
        }

    }

    if( !ground_model_file.empty() ){
        if( domain_extent_read ){
            std::cout << "WARNING: Both 'domain_extent' and 'ground_model_obj_file' were defined. The domain extent will not be used because it is determined by the geometry in the ground model file." << std::endl;
        }else if( domain_origin_read ) {
            std::cout << "WARNING: Both 'domain_origin' and 'ground_model_obj_file' were defined. The domain origin will not be used because it is determined by the geometry in the ground model file." << std::endl;
        }else if( ground_resolution_read ) {
            std::cout << "WARNING: Both 'ground_resolution' and 'ground_model_obj_file' were defined. The ground resolution will not be used because it is determined by the geometry in the ground model file." << std::endl;
        }else if( !ground_texture_file.empty() ){
            std::cout << "WARNING: Both 'ground_texture_file' and 'ground_model_obj_file' were defined. The ground texture file will not be used because all ground geometry comes from the ground model file." << std::endl;
        }else if( ground_color_read ) {
            std::cout << "WARNING: Both 'ground_color' and 'ground_model_obj_file' were defined. The ground color will not be used because it is determined by the geometry in the ground model file." << std::endl;
        }
    }else if( ground_color_read && !ground_texture_file.empty() ){
        std::cout << "WARNING: Both 'ground_color' and 'ground_texture_file' were defined. The ground color will be overridden by the texture image." << std::endl;
    }

    uint ground_objID;
    if( !ground_model_file.empty() ) {
        ground_UUIDs = context_ptr->loadOBJ(ground_model_file.c_str());
        ground_objID = context_ptr->addPolymeshObject( ground_UUIDs );
    }else if( !ground_texture_file.empty() ){
        ground_objID = context_ptr->addTileObject( domain_origin, domain_extent, nullrotation, ground_resolution, ground_texture_file.c_str() );
        ground_UUIDs = context_ptr->getObjectPrimitiveUUIDs(ground_objID);
    }else if( ground_color_read ){
        ground_objID = context_ptr->addTileObject( domain_origin, domain_extent, nullrotation, ground_resolution, ground_color );
        ground_UUIDs = context_ptr->getObjectPrimitiveUUIDs(ground_objID);
    }else {
        ground_objID = context_ptr->addTileObject(domain_origin, domain_extent, nullrotation, ground_resolution);
        ground_UUIDs = context_ptr->getObjectPrimitiveUUIDs(ground_objID);
    }

    context_ptr->setPrimitiveData( ground_UUIDs, "twosided_flag", uint(0) );

    // ####### BUILDING THE CANOPY ####### //

    for (pugi::xml_node p = helios.child("canopy_block"); p; p = p.next_sibling("canopy_block")) {

        std::string canopy_model_file;
        node = p.child("canopy_model_obj_file");
        if (!node.empty()) {

            const char *canopy_model_file_str = node.child_value();
            canopy_model_file = trim_whitespace(std::string(canopy_model_file_str));

            std::string ext = getFileExtension(canopy_model_file);

            if (canopy_model_file.empty()) {
                helios_runtime_error("ERROR: Value given for 'canopy_model_obj_file' is empty.");
            } else if (ext != ".obj" && ext != ".OBJ") {
                helios_runtime_error("ERROR: File given for 'canopy_model_obj_file' is not an .obj file.");
            } else if (!std::filesystem::exists(ground_model_file)) {
                helios_runtime_error("ERROR: File given for 'canopy_model_obj_file' does not exist.");
            }

        }

        vec3 canopy_origin(0,0,0);
        bool canopy_origin_read = false;
        node = p.child("canopy_origin");
        if( node.empty() ){
            std::cout << "WARNING: No value given for 'canopy_origin'. Using default value of " << canopy_origin << std::endl;
        }else {

            const char *canopy_origin_str = node.child_value();
            if (!parse_vec3(canopy_origin_str, canopy_origin)) {
                helios_runtime_error("ERROR: Value given for 'canopy_origin' could not be parsed.");
            }else{
                canopy_origin_read = true;
            }

        }

        int2 plant_count(1,1);
        bool plant_count_read = false;
        node = p.child("plant_count");
        if( node.empty() ){
            std::cout << "WARNING: No value given for 'plant_count'. Using default value of " << plant_count << std::endl;
        }else {

            const char *plant_count_str = node.child_value();
            if (!parse_int2(plant_count_str, plant_count)) {
                helios_runtime_error("ERROR: Value given for 'plant_count' could not be parsed.");
            }else if( plant_count.x<1 || plant_count.y<1 ){
                helios_runtime_error("ERROR: Value given for 'plant_count' must be greater than or equal to 1.");
            }else{
                plant_count_read = true;
            }

        }

        vec2 plant_spacing(0.5,0.5);
        bool plant_spacing_read = false;
        node = p.child("plant_spacing");
        if( node.empty() ){
            std::cout << "WARNING: No value given for 'plant_spacing'. Using default value of " << plant_spacing << std::endl;
        }else {

            const char *plant_spacing_str = node.child_value();
            if (!parse_vec2(plant_spacing_str, plant_spacing)) {
                helios_runtime_error("ERROR: Value given for 'plant_spacing' could not be parsed.");
            }else if( plant_spacing.x<=0 || plant_spacing.y<=0 ){
                helios_runtime_error("ERROR: Value given for 'plant_spacing' must be greater than 0.");
            }else{
                plant_spacing_read = true;
            }

        }

        std::string plant_library_name;
        node = p.child("plant_library_name");
        if (!node.empty()) {

            const char *plant_library_name_str = node.child_value();
            plant_library_name = trim_whitespace(std::string(plant_library_name_str));

        }

        float plant_age = 0;
        bool plant_age_read = false;
        node = p.child("plant_age");
        if( node.empty() ){
            std::cout << "WARNING: No value given for 'plant_age'. Using default value of " << plant_age << std::endl;
        }else {

            const char *plant_age_str = node.child_value();
            if (!parse_float(plant_age_str, plant_age)) {
                helios_runtime_error("ERROR: Value given for 'plant_age' could not be parsed.");
            }else if( plant_age<0 ){
                helios_runtime_error("ERROR: Value given for 'plant_age' must be greater than or equal to 0.");
            }else{
                plant_age_read = true;
            }

        }

        float ground_clipping_height = 0;
        bool ground_clipping_height_read = false;
        node = p.child("ground_clipping_height");
        if( !node.empty() ){

            const char *ground_clipping_height_str = node.child_value();
            if (!parse_float(ground_clipping_height_str, ground_clipping_height)) {
                helios_runtime_error("ERROR: Value given for 'ground_clipping_height' could not be parsed.");
            }else{
                ground_clipping_height_read = true;
            }

        }


        if (!canopy_model_file.empty()) { //canopy geometry will be read from OBJ file specified by 'canopy_model_obj_file'

            if( plant_count_read ) {
                std::cout << "WARNING: Both 'plant_count' and 'ground_model_obj_file' were defined. The plant count value will not be used because it is determined by the geometry in the canopy model file." << std::endl;
            }else if( plant_spacing_read ) {
                std::cout << "WARNING: Both 'plant_spacing' and 'ground_model_obj_file' were defined. The plant spacing value will not be used because it is determined by the geometry in the canopy model file." << std::endl;
            }

            std::vector<uint> canopy_UUIDs = context_ptr->loadOBJ(canopy_model_file.c_str());

            std::vector<helios::vec3> curr_plant_locations{};
            individual_plant_locations.push_back(curr_plant_locations);
            // TODO: save canopy_UUIDs for individual canopies here
            if( canopy_origin_read ) {
                context_ptr->translatePrimitive(canopy_UUIDs, canopy_origin);
            }

            leaf_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "leaf");
            petiolule_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "petiolule");
            petiole_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "petiole");
            internode_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "internode");
            peduncle_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "peduncle");
            petal_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "petal");
            sepal_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "sepal");
            fruit_UUIDs = context_ptr->filterPrimitivesByData(canopy_UUIDs, "object_label", "fruit");

            canopy_UUID_vector.push_back(canopy_UUIDs);

        }else{ //canopy geometry will be generated using the plant architecture plug-in

            if( plant_library_name.empty() ){
                helios_runtime_error("ERROR: No value given for 'plant_library_name'.");
            }

            plant_architecture_ptr->loadPlantModelFromLibrary( plant_library_name );

            if( ground_clipping_height_read ){
                plant_architecture_ptr->enableGroundClipping( ground_clipping_height );
            }

            std::vector<uint> canopy_UUIDs = plant_architecture_ptr->buildPlantCanopyFromLibrary( canopy_origin, plant_spacing, plant_count, plant_age);
            std::vector<helios::vec3> curr_plant_locations = plant_architecture_ptr->getPlantBasePosition(canopy_UUIDs);
            individual_plant_locations.push_back(curr_plant_locations);

            leaf_UUIDs = plant_architecture_ptr->getAllLeafUUIDs();
            internode_UUIDs = plant_architecture_ptr->getAllInternodeUUIDs();
            petiole_UUIDs = plant_architecture_ptr->getAllPetioleUUIDs();
            peduncle_UUIDs = plant_architecture_ptr->getAllPeduncleUUIDs();
            std::vector<uint> flower_UUIDs = plant_architecture_ptr->getAllFlowerUUIDs();
            petal_UUIDs = context_ptr->filterPrimitivesByData(flower_UUIDs, "object_label", "petal");
            sepal_UUIDs = context_ptr->filterPrimitivesByData(flower_UUIDs, "object_label", "sepal");
            if( petal_UUIDs.empty() && sepal_UUIDs.empty() ){
                petal_UUIDs = flower_UUIDs;
                sepal_UUIDs.clear();
            }
            fruit_UUIDs = plant_architecture_ptr->getAllFruitUUIDs();

            canopy_UUID_vector.push_back(canopy_UUIDs);

        }

    }

    // ####### CROP ALL GEOMETRY TO THE GROUND ####### //

    uint primitive_count = context_ptr->getPrimitiveCount();

    context_ptr->cropDomainX( domain_origin.x + make_vec2(-0.5f*domain_extent.x,0.5f*domain_extent.x) );
    context_ptr->cropDomainY( domain_origin.y + make_vec2(-0.5f*domain_extent.y,0.5f*domain_extent.y) );

    uint deleted_primitives = primitive_count - context_ptr->getPrimitiveCount();
    if( deleted_primitives>0 ){
        std::cout << "WARNING: " << deleted_primitives << " primitives were deleted because they were overhanging the ground." << std::endl;

        context_ptr->cleanDeletedUUIDs(leaf_UUIDs);
        context_ptr->cleanDeletedUUIDs(petiolule_UUIDs);
        context_ptr->cleanDeletedUUIDs(petiole_UUIDs);
        context_ptr->cleanDeletedUUIDs(internode_UUIDs);
        context_ptr->cleanDeletedUUIDs(peduncle_UUIDs);
        context_ptr->cleanDeletedUUIDs(petal_UUIDs);
        context_ptr->cleanDeletedUUIDs(sepal_UUIDs);
        context_ptr->cleanDeletedUUIDs(fruit_UUIDs);
        context_ptr->cleanDeletedUUIDs(ground_UUIDs);

    }

    // ####### SET UUID GLOBAL DATA ####### //

    context_ptr->setGlobalData( "ground_UUIDs", ground_UUIDs );
    context_ptr->setGlobalData( "leaf_UUIDs", leaf_UUIDs );
    context_ptr->setGlobalData( "petiolule_UUIDs", petiolule_UUIDs );
    context_ptr->setGlobalData( "petiole_UUIDs", petiole_UUIDs );
    context_ptr->setGlobalData( "internode_UUIDs", internode_UUIDs );
    context_ptr->setGlobalData( "peduncle_UUIDs", peduncle_UUIDs );
    context_ptr->setGlobalData( "petal_UUIDs", petal_UUIDs );
    context_ptr->setGlobalData( "sepal_UUIDs", sepal_UUIDs );
    context_ptr->setGlobalData( "fruit_UUIDs", fruit_UUIDs );

}
