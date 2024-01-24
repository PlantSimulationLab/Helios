/** \file "CanopyGenerator.cpp" Primary source file for canopy geometry generator plug-in.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "CanopyGenerator.h"
#include "pugixml.hpp"

using namespace helios;

HomogeneousCanopyParameters::HomogeneousCanopyParameters(){

    leaf_size = make_vec2(0.1,0.1);

    leaf_subdivisions = make_int2(1,1);

    leaf_color = RGB::green;

    leaf_angle_distribution = "spherical";

    leaf_area_index = 1.f;

    canopy_height = 1.f;

    canopy_extent = make_vec2(5,5);

    canopy_origin = make_vec3(0,0,0);

    buffer = "z";

}

SphericalCrownsCanopyParameters::SphericalCrownsCanopyParameters(){

    leaf_size = make_vec2(0.025,0.025);

    leaf_subdivisions = make_int2(1,1);

    leaf_color = RGB::green;

    leaf_angle_distribution = "spherical";

    leaf_area_density = 1.f;

    crown_radius = make_vec3(0.5f,0.5f,0.5f);

    canopy_configuration = "uniform";

    plant_spacing = make_vec2(2.f,2.f);

    plant_count = make_int2(5,5);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0.f;

}

ConicalCrownsCanopyParameters::ConicalCrownsCanopyParameters(){

    leaf_size = make_vec2(0.025,0.025);

    leaf_subdivisions = make_int2(1,1);

    leaf_color = RGB::green;

    leaf_angle_distribution = "spherical";

    leaf_area_density = 1.f;

    crown_radius = 0.5f;

    crown_height = 1.f;

    canopy_configuration = "uniform";

    plant_spacing = make_vec2(2.f,2.f);

    plant_count = make_int2(5,5);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0.f;

}

VSPGrapevineParameters::VSPGrapevineParameters(){

    leaf_width = 0.18;

    leaf_subdivisions = make_int2(1,1);

    leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

    wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

    wood_subdivisions = 10;

    trunk_height = 0.7;

    trunk_radius = 0.05;

    cordon_height = 0.9;

    cordon_radius = 0.02;

    shoot_length = 0.9;

    shoot_radius = 0.005;

    shoots_per_cordon = 10;

    leaf_spacing_fraction = 0.5;

    grape_radius = 0.0075;

    cluster_radius = 0.03;

    cluster_height_max = 0.15;

    grape_color = make_RGBcolor(0.18,0.2,0.25);

    grape_subdivisions = 8;

    plant_spacing = 2;

    row_spacing = 2;

    plant_count = make_int2(3,3);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

SplitGrapevineParameters::SplitGrapevineParameters(){

    leaf_width = 0.18;

    leaf_subdivisions = make_int2(1,1);

    leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

    wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

    wood_subdivisions = 10;

    trunk_height = 1.3;

    trunk_radius = 0.05;

    cordon_height = 1.5;

    cordon_radius = 0.02;

    cordon_spacing = 1.f;

    shoot_length = 1.2;

    shoot_radius = 0.0025;

    shoots_per_cordon = 10;

    shoot_angle_tip = 0.4*M_PI;

    shoot_angle_base = 0.;

    leaf_spacing_fraction = 0.6;

    grape_radius = 0.0075;

    cluster_radius = 0.03;

    cluster_height_max = 0.1;

    grape_color = make_RGBcolor(0.18,0.2,0.25);

    grape_subdivisions = 8;

    plant_spacing = 2;

    row_spacing = 4;

    plant_count = make_int2(3,3);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

UnilateralGrapevineParameters::UnilateralGrapevineParameters(){

    leaf_width = 0.18;

    leaf_subdivisions = make_int2(1,1);

    leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

    wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

    wood_subdivisions = 10;

    trunk_height = 0.7;

    trunk_radius = 0.05;

    cordon_height = 0.9;

    cordon_radius = 0.04;

    shoot_length = 0.9;

    shoot_radius = 0.0025;

    shoots_per_cordon = 20;

    leaf_spacing_fraction = 0.6;

    grape_radius = 0.0075;

    cluster_radius = 0.03;

    cluster_height_max = 0.1;

    grape_color = make_RGBcolor(0.18,0.2,0.25);

    grape_subdivisions = 8;

    plant_spacing = 1.5;

    row_spacing = 2;

    plant_count = make_int2(3,3);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

GobletGrapevineParameters::GobletGrapevineParameters(){

    leaf_width = 0.18;

    leaf_subdivisions = make_int2(1,1);

    leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

    wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

    wood_subdivisions = 10;

    trunk_height = 0.7;

    trunk_radius = 0.05;

    cordon_height = 0.9;

    cordon_radius = 0.02;

    shoot_length = 0.9;

    shoot_radius = 0.0025;

    shoots_per_cordon = 10;

    leaf_spacing_fraction = 0.6;

    grape_radius = 0.0075;

    cluster_radius = 0.03;

    cluster_height_max = 0.1;

    grape_color = make_RGBcolor(0.18,0.2,0.25);

    grape_subdivisions = 8;

    plant_spacing = 2;

    row_spacing = 2;

    plant_count = make_int2(3,3);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

WhiteSpruceCanopyParameters::WhiteSpruceCanopyParameters(){

    needle_width = 0.0005;

    needle_length = 0.05;

    needle_color = make_RGBcolor(0.67,0.9,0.56);

    needle_subdivisions = make_int2(1,1);

    wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

    wood_subdivisions = 10;

    trunk_height = 10;

    trunk_radius = 0.15;

    base_height = 2.0;

    crown_radius = 0.65;

    shoot_radius = 0.02;

    level_spacing = 0.35;

    branches_per_level = 8;

    shoot_angle = 0.3*M_PI;

    canopy_configuration = "random";

    plant_spacing = make_vec2(10,10);

    plant_count = make_int2(3,3);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

TomatoParameters::TomatoParameters(){

    leaf_length = 0.2;

    leaf_subdivisions = make_int2(4,3);

    leaf_texture_file = "plugins/canopygenerator/textures/TomatoLeaf_big.png";

    shoot_color = make_RGBcolor(0.35,0.45,0.2);

    shoot_subdivisions = 10;

    plant_height = 1.;

    fruit_radius = 0.03;

    fruit_color = make_RGBcolor(0.7,0.28,0.2);

    fruit_subdivisions = 8;

    plant_spacing = 2;

    row_spacing = 2;

    plant_count = make_int2(3,3);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

StrawberryParameters::StrawberryParameters(){

    leaf_length = 0.1;

    leaf_subdivisions = make_int2(4,4);

    leaf_texture_file = "plugins/canopygenerator/textures/StrawberryLeaf.png";

    stem_color = make_RGBcolor(0.35,0.45,0.2);

    stem_subdivisions = 10;

    stems_per_plant = 50;

    stem_radius = 0.005;

    plant_height = 0.4;

    fruit_radius = 0.025;

    fruit_texture_file = "plugins/canopygenerator/textures/StrawberryTexture.jpg";

    fruit_subdivisions = 12;

    clusters_per_stem = 0.6;

    plant_spacing = 0.5;

    row_spacing = 1.5;

    plant_count = make_int2(4,2);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

WalnutCanopyParameters::WalnutCanopyParameters(){

    leaf_length = 0.15;

    leaf_subdivisions = make_int2(1,2);

    leaf_texture_file = "plugins/canopygenerator/textures/WalnutLeaf.png";

    wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

    wood_subdivisions = 10;

    trunk_radius = 0.15;

    trunk_height = 4.f;

    branch_length = make_vec3(4,0.75,0.75);

    fruit_radius = 0.04;

    fruit_texture_file = "plugins/canopygenerator/textures/WalnutTexture.png";

    fruit_subdivisions = 16;

    plant_spacing = 6;

    row_spacing = 8;

    plant_count = make_int2(4,2);

    canopy_origin = make_vec3(0,0,0);

    canopy_rotation = 0;

}

SorghumCanopyParameters::SorghumCanopyParameters(){
    sorghum_stage = 5;

    // stage 1
    s1_stem_length = 0.04;

    s1_stem_radius = 0.003;

    s1_stem_subdivisions = 10;

    s1_leaf_size1 = make_vec2(0.14,0.012);

    s1_leaf_size2 = make_vec2(0.12,0.012);

    s1_leaf_size3 = make_vec2(0.06,0.008);

    s1_leaf1_angle = 50;

    s1_leaf2_angle = 50;

    s1_leaf3_angle = 50;

    s1_leaf_subdivisions = make_int2(20,2);

    s1_leaf_texture_file = "plugins/canopygenerator/textures/s1_Sorghum_leaf.png";


    // stage 2
    s2_stem_length = 0.2;

    s2_stem_radius = 0.003;

    s2_stem_subdivisions = 10;

    s2_leaf_size1 = make_vec2(0.25,0.02);

    s2_leaf_size2 = make_vec2(0.14,0.02);

    s2_leaf_size3 = make_vec2(0.2,0.015);

    s2_leaf_size4 = make_vec2(0.12,0.012);

    s2_leaf_size5 = make_vec2(0.08,0.01);

    s2_leaf1_angle = 25;

    s2_leaf2_angle = 50;

    s2_leaf3_angle = 15;

    s2_leaf4_angle = 25;

    s2_leaf5_angle = 10;

    s2_leaf_subdivisions = make_int2(30,2);

    s2_leaf_texture_file = "plugins/canopygenerator/textures/s2_Sorghum_leaf.png";

    // stage 3
    s3_stem_length = 1.2;

    s3_stem_radius = 0.01;

    s3_stem_subdivisions = 10;

    s3_leaf_size = make_vec2(0.8,0.08);

    s3_leaf_subdivisions = make_int2(30,2);

    s3_number_of_leaves = 15;

    s3_mean_leaf_angle = 45;

    s3_leaf_texture_file = "plugins/canopygenerator/textures/s3_Sorghum_leaf.png";

    // stage 4
    s4_stem_length = 1.6;

    s4_stem_radius = 0.01;

    s4_stem_subdivisions = 10;

    s4_panicle_size = make_vec2(0.2,0.06);

    s4_panicle_subdivisions = 5;

    s4_seed_texture_file = "plugins/canopygenerator/textures/s4_Sorghum_seed.png";

    s4_leaf_size = make_vec2(0.8,0.08);

    s4_leaf_subdivisions = make_int2(30,2);

    s4_number_of_leaves = 15;

    s4_mean_leaf_angle = 40;

    s4_leaf_texture_file = "plugins/canopygenerator/textures/s4_Sorghum_leaf.png";

    // stage 5
    s5_stem_length = 2.5;

    s5_stem_radius = 0.01;

    s5_stem_bend = 0.15;

    s5_stem_subdivisions = 10;

    s5_panicle_size = make_vec2(0.3,0.08);

    s5_panicle_subdivisions = 5;

    s5_seed_texture_file = "plugins/canopygenerator/textures/s5_Sorghum_seed.png";

    s5_leaf_size = make_vec2(0.9,0.125);

    s5_leaf_subdivisions = make_int2(30,2);

    s5_number_of_leaves = 24;

    s5_mean_leaf_angle = 20;  // std = 10 degrees

    s5_leaf_texture_file = "plugins/canopygenerator/textures/s5_Sorghum_leaf.png";

    // Canopy
    plant_spacing = 0.45;

    row_spacing = 0.45;

    plant_count = make_int2(10 ,10);

    canopy_origin = make_vec3(0,0,0);
}

BeanParameters::BeanParameters() {

  leaf_length = 0.075;

  leaf_subdivisions = make_int2(6, 4);

  leaf_texture_file = "plugins/canopygenerator/textures/BeanLeaf.png";

  shoot_color = make_RGBcolor(0.6471, 0.7333, 0.1176);

  shoot_subdivisions = 10;

  stem_radius = 0.004;

  stem_length = 0.075;

  leaflet_length = 0.075;

  pod_length = 0; //pods not supported yet

  pod_color = make_RGBcolor(0.7, 0.28, 0.2);

  pod_subdivisions = 8;

  plant_spacing = 0.15;

  row_spacing = 0.6;

  plant_count = make_int2(3, 3);

  germination_probability = 1.f;

  canopy_origin = make_vec3(0, 0, 0);

  canopy_rotation = 0;

}

  CanopyGenerator::CanopyGenerator( helios::Context* m_context ){

    context = m_context;

    //seed the random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);

    printmessages = true;

    enable_element_labels = false;

}

int CanopyGenerator::selfTest(){

    std::cout << "Running canopy generator plug-in self-test..." << std::endl;

    Context context_test;

    std::cout << "Generating default homogeneous canopy..." << std::flush;

    CanopyGenerator canopygenerator_0(&context_test);
    canopygenerator_0.disableMessages();
    HomogeneousCanopyParameters params_0;
    canopygenerator_0.buildCanopy(params_0);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default spherical crowns canopy..." << std::flush;

    CanopyGenerator canopygenerator_1(&context_test);
    canopygenerator_1.disableMessages();
    SphericalCrownsCanopyParameters params_1;
    canopygenerator_1.buildCanopy(params_1);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default VSP grapevine canopy..." << std::flush;

    CanopyGenerator canopygenerator_2(&context_test);
    canopygenerator_2.disableMessages();
    VSPGrapevineParameters params_2;
    params_2.grape_radius = 0;
    canopygenerator_2.buildCanopy(params_2);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default split trellis grapevine canopy..." << std::flush;

    CanopyGenerator canopygenerator_3(&context_test);
    canopygenerator_3.disableMessages();
    SplitGrapevineParameters params_3;
    params_3.grape_radius = 0;
    canopygenerator_3.buildCanopy(params_3);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default unilateral trellis grapevine canopy..." << std::flush;

    CanopyGenerator canopygenerator_4(&context_test);
    canopygenerator_4.disableMessages();
    UnilateralGrapevineParameters params_4;
    params_4.grape_radius = 0;
    canopygenerator_4.buildCanopy(params_4);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default goblet trellis grapevine canopy..." << std::flush;

    CanopyGenerator canopygenerator_5(&context_test);
    canopygenerator_5.disableMessages();
    GobletGrapevineParameters params_5;
    params_5.grape_radius = 0;
    canopygenerator_5.buildCanopy(params_5);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default strawberry canopy..." << std::flush;

    CanopyGenerator canopygenerator_7(&context_test);
    canopygenerator_7.disableMessages();
    StrawberryParameters params_7;
    canopygenerator_7.buildCanopy(params_7);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default walnut tree canopy (minus nuts)..." << std::flush;

    CanopyGenerator canopygenerator_8(&context_test);
    canopygenerator_8.disableMessages();
    WalnutCanopyParameters params_8;
    params_8.fruit_radius = 0.f;
    canopygenerator_8.buildCanopy(params_8);
    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    std::cout << "Generating default sorghum plant canopy..." << std::flush;

    CanopyGenerator canopygenerator_9(&context_test);
    canopygenerator_9.disableMessages();
    SorghumCanopyParameters params_9;
    canopygenerator_9.buildCanopy(params_9);
    context_test.deletePrimitive( context_test.getAllUUIDs() );

    std::cout << "done." << std::endl;

    std::cout << "Generating homogeneous canopy with randomly deleted primitives..." << std::flush;

    CanopyGenerator canopygenerator_6(&context_test);
    canopygenerator_6.disableMessages();
    HomogeneousCanopyParameters params_6;
    canopygenerator_6.buildCanopy(params_6);

    std::vector<uint> UUIDs_leaves = flatten( canopygenerator_6.getLeafUUIDs(0) );

    context_test.deletePrimitive(UUIDs_leaves.at(0));
    context_test.deletePrimitive(UUIDs_leaves.at(11));
    context_test.deletePrimitive(UUIDs_leaves.at(23));
    context_test.deletePrimitive(UUIDs_leaves.back());

    UUIDs_leaves = flatten( canopygenerator_6.getLeafUUIDs(0) );

    bool fail_flag = false;
    for( uint UUID : UUIDs_leaves ){
        if( !context_test.doesPrimitiveExist( UUID ) ){
            fail_flag = true;
        }
    }

    std::vector<uint> UUIDs_all = canopygenerator_6.getAllUUIDs(0);

    fail_flag = false;
    for( uint p : UUIDs_all ){
        if( !context_test.doesPrimitiveExist( p ) ){
            fail_flag = true;
        }
    }

    context_test.deletePrimitive(context_test.getAllUUIDs());

    std::cout << "done." << std::endl;

    if( fail_flag ){
        std::cout << "failed." << std::endl;
        return 1;
    }else{
        std::cout << "passed." << std::endl;
        return 0;
    }

}

void CanopyGenerator::loadXML( const char* filename ){

    if( printmessages ){
        std::cout << "Reading XML file: " << filename << "..." << std::flush;
    }

    float nullvalue_f = 99999;
    int nullvalue_i = 99999;
    std::string nullvalue_s = "99999";

    //Check if file exists
    std::ifstream f(filename);
    if( !f.good() ){
        std::cerr << "failed." << std::endl;
        throw( std::runtime_error("XML file " + std::string(filename) + " does not exist.") );
    }

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    //load file
    pugi::xml_parse_result result = xmldoc.load_file(filename);

    //error checking
    if (!result){
        std::cout << "failed." << std::endl;
        throw( std::runtime_error("XML  file " + std::string(filename) + " parsed with errors, attribute value: [" + xmldoc.child("node").attribute("attr").value() + "]\nError description: " + result.description() + "\n"));
    }

    pugi::xml_node helios = xmldoc.child("helios");

    if( helios.empty() ){
        std::cout << "failed." << std::endl;
        throw(std::runtime_error("ERROR (loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags."));
    }

    //looping over any Canopy Generator blocks specified in XML file
    for (pugi::xml_node cgen = helios.child("canopygenerator"); cgen; cgen = cgen.next_sibling("CanopyGenerator")) {

        //looping over any canopy types specified

        //Homogeneous Canopy
        for (pugi::xml_node s = cgen.child("HomogeneousCanopyParameters"); s; s = s.next_sibling(
                "HomogeneousCanopyParameters")) {

            HomogeneousCanopyParameters homogeneouscanopyparameters;

            // ----- leaf size ------//
            vec2 leaf_size = XMLloadvec2(s, "leaf_size");
            if (leaf_size.x != nullvalue_f && leaf_size.y != nullvalue_f) {
                homogeneouscanopyparameters.leaf_size = leaf_size;
            }

            // ----- leaf subdivisions ------//
            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                homogeneouscanopyparameters.leaf_subdivisions = leaf_subdivisions;
            }

            // ----- leaf color ------//
            RGBAcolor leaf_color = XMLloadrgba(s, "leaf_color");
            if (leaf_color.a != 0) {
                homogeneouscanopyparameters.leaf_color = make_RGBcolor(leaf_color.r,leaf_color.g,leaf_color.b);
            }

            // ----- leaf texture file ------//
            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if (leaf_texture_file != nullvalue_s) {
                homogeneouscanopyparameters.leaf_texture_file = leaf_texture_file;
            }

            // ----- leaf area index ------//
            float LAI = XMLloadfloat(s, "leaf_area_index");
            if (LAI != nullvalue_f) {
                homogeneouscanopyparameters.leaf_area_index = LAI;
            }

            // ----- canopy height ------//
            float h = XMLloadfloat(s, "canopy_height");
            if (h != nullvalue_f) {
                homogeneouscanopyparameters.canopy_height = h;
            }

            // ----- canopy extent ------//
            vec2 canopy_extent = XMLloadvec2(s, "canopy_extent");
            if (canopy_extent.x != nullvalue_f && canopy_extent.y != nullvalue_f) {
                homogeneouscanopyparameters.canopy_extent = canopy_extent;
            }

            // ----- canopy origin ------//
            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                homogeneouscanopyparameters.canopy_origin = canopy_origin;
            }

            // ----- buffer ------//
            std::string buffer = XMLloadstring(s, "buffer");
            if ( buffer != nullvalue_s ) {
                homogeneouscanopyparameters.buffer = buffer;
            }

            buildCanopy(homogeneouscanopyparameters);

        }


        //VSP Grapevine Canopy
        for (pugi::xml_node s = cgen.child("VSPGrapevineParameters"); s; s = s.next_sibling("VSPGrapevineParameters")) {

            VSPGrapevineParameters vspgrapevineparameters;

            float leaf_width = XMLloadfloat(s, "leaf_width");
            if (leaf_width != nullvalue_f) {
                vspgrapevineparameters.leaf_width = leaf_width;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                vspgrapevineparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                vspgrapevineparameters.leaf_texture_file = leaf_texture_file;
            }

            std::string wood_texture_file = XMLloadstring(s, "wood_texture_file");
            if ( wood_texture_file != nullvalue_s ) {
                vspgrapevineparameters.wood_texture_file = wood_texture_file;
            }

            int wood_subdivisions = XMLloadint(s, "wood_subdivisions");
            if ( wood_subdivisions != nullvalue_i) {
                vspgrapevineparameters.wood_subdivisions = wood_subdivisions;
            }

            float h = XMLloadfloat(s, "trunk_height");
            if (h != nullvalue_f) {
                vspgrapevineparameters.trunk_height = h;
            }

            float r = XMLloadfloat(s, "trunk_radius");
            if (r != nullvalue_f) {
                vspgrapevineparameters.trunk_radius = r;
            }

            float ch = XMLloadfloat(s, "cordon_height");
            if (ch != nullvalue_f) {
                vspgrapevineparameters.cordon_height = ch;
            }

            float cr = XMLloadfloat(s, "cordon_radius");
            if (cr != nullvalue_f) {
                vspgrapevineparameters.cordon_radius = cr;
            }

            float sl = XMLloadfloat(s, "shoot_length");
            if (sl != nullvalue_f) {
                vspgrapevineparameters.shoot_length = sl;
            }

            float sr = XMLloadfloat(s, "shoot_radius");
            if (sr != nullvalue_f) {
                vspgrapevineparameters.shoot_radius = sr;
            }

            int spc = XMLloadint(s, "shoots_per_cordon");
            if (spc != nullvalue_i) {
                vspgrapevineparameters.shoots_per_cordon = uint(spc);
            }

            float lsf = XMLloadfloat(s, "leaf_spacing_fraction");
            if (lsf != nullvalue_f) {
                vspgrapevineparameters.leaf_spacing_fraction = lsf;
            }

            float gr = XMLloadfloat(s, "grape_radius");
            if (gr != nullvalue_f) {
                vspgrapevineparameters.grape_radius = gr;
            }

            float clr = XMLloadfloat(s, "cluster_radius");
            if (clr != nullvalue_f) {
                vspgrapevineparameters.cluster_radius = clr;
            }

            RGBAcolor grape_color = XMLloadrgba(s, "grape_color");
            if ( grape_color.a != 0 ) {
                vspgrapevineparameters.grape_color = make_RGBcolor(grape_color.r,grape_color.g,grape_color.b);
            }

            int grape_subdivisions = XMLloadint(s, "grape_subdivisions");
            if (grape_subdivisions != nullvalue_i) {
                vspgrapevineparameters.grape_subdivisions = uint(grape_subdivisions);
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                vspgrapevineparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                vspgrapevineparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                vspgrapevineparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                vspgrapevineparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                vspgrapevineparameters.canopy_rotation = canopy_rotation;
            }

            buildCanopy(vspgrapevineparameters);

        }

        //Split Grapevine Canopy
        for (pugi::xml_node s = cgen.child("SplitGrapevineParameters"); s; s = s.next_sibling(
                "SplitGrapevineParameters")) {

            SplitGrapevineParameters splitgrapevineparameters;

            float leaf_width = XMLloadfloat(s, "leaf_width");
            if (leaf_width != nullvalue_f) {
                splitgrapevineparameters.leaf_width = leaf_width;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                splitgrapevineparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                splitgrapevineparameters.leaf_texture_file = leaf_texture_file;
            }

            std::string wood_texture_file = XMLloadstring(s, "wood_texture_file");
            if ( wood_texture_file != nullvalue_s ) {
                splitgrapevineparameters.wood_texture_file = wood_texture_file;
            }

            int wood_subdivisions = XMLloadint(s, "wood_subdivisions");
            if (wood_subdivisions != nullvalue_i) {
                splitgrapevineparameters.wood_subdivisions = wood_subdivisions;
            }

            float h = XMLloadfloat(s, "trunk_height");
            if (h != nullvalue_f) {
                splitgrapevineparameters.trunk_height = h;
            }

            float r = XMLloadfloat(s, "trunk_radius");
            if (r != nullvalue_f) {
                splitgrapevineparameters.trunk_radius = r;
            }

            float ch = XMLloadfloat(s, "cordon_height");
            if (ch != nullvalue_f) {
                splitgrapevineparameters.cordon_height = ch;
            }

            float cr = XMLloadfloat(s, "cordon_radius");
            if (cr != nullvalue_f) {
                splitgrapevineparameters.cordon_radius = cr;
            }

            float cs = XMLloadfloat(s, "cordon_spacing");
            if (cs != nullvalue_f) {
                splitgrapevineparameters.cordon_spacing = cs;
            }

            float sl = XMLloadfloat(s, "shoot_length");
            if (sl != nullvalue_f) {
                splitgrapevineparameters.shoot_length = sl;
            }

            float sr = XMLloadfloat(s, "shoot_radius");
            if (sr != nullvalue_f) {
                splitgrapevineparameters.shoot_radius = sr;
            }

            int spc = XMLloadint(s, "shoots_per_cordon");
            if (spc != nullvalue_i) {
                splitgrapevineparameters.shoots_per_cordon = uint(spc);
            }

/*
      float spa = XMLloadfloat( s, "shoots_tip_angle" );
      if( spa != nullvalue_f ){
	splitgrapevineparameters.shoots_tip_angle = uint(spa);
*/
            float lsf = XMLloadfloat(s, "leaf_spacing_fraction");
            if (lsf != nullvalue_f) {
                splitgrapevineparameters.leaf_spacing_fraction = lsf;
            }

            float gr = XMLloadfloat(s, "grape_radius");
            if (gr != nullvalue_f) {
                splitgrapevineparameters.grape_radius = gr;
            }

            float clr = XMLloadfloat(s, "cluster_radius");
            if (clr != nullvalue_f) {
                splitgrapevineparameters.cluster_radius = clr;
            }

            RGBAcolor grape_color = XMLloadrgba(s, "grape_color");
            if (grape_color.a!=0 ) {
                splitgrapevineparameters.grape_color = make_RGBcolor(grape_color.r,grape_color.g,grape_color.b);
            }

            int grape_subdivisions = XMLloadint(s, "grape_subdivisions");
            if (grape_subdivisions != nullvalue_i) {
                splitgrapevineparameters.grape_subdivisions = uint(grape_subdivisions);
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                splitgrapevineparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                splitgrapevineparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                splitgrapevineparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                splitgrapevineparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                splitgrapevineparameters.canopy_rotation = canopy_rotation;
            }


            buildCanopy(splitgrapevineparameters);

        }


        //UnilateralGrapevineParameters Canopy
        for (pugi::xml_node s = cgen.child("UnilateralGrapevineParameters"); s; s = s.next_sibling(
                "UnilateralGrapevineParameters")) {

            UnilateralGrapevineParameters unilateralgrapevineparameters;

            float leaf_width = XMLloadfloat(s, "leaf_width");
            if (leaf_width != nullvalue_f) {
                unilateralgrapevineparameters.leaf_width = leaf_width;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                unilateralgrapevineparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                unilateralgrapevineparameters.leaf_texture_file = leaf_texture_file;
            }

            std::string wood_texture_file = XMLloadstring(s, "wood_texture_file");
            if ( wood_texture_file != nullvalue_s ) {
                unilateralgrapevineparameters.wood_texture_file = wood_texture_file;
            }

            int wood_subdivisions = XMLloadint(s, "wood_subdivisions");
            if ( wood_subdivisions != nullvalue_i) {
                unilateralgrapevineparameters.wood_subdivisions = wood_subdivisions;
            }

            float h = XMLloadfloat(s, "trunk_height");
            if (h != nullvalue_f) {
                unilateralgrapevineparameters.trunk_height = h;
            }

            float r = XMLloadfloat(s, "trunk_radius");
            if (r != nullvalue_f) {
                unilateralgrapevineparameters.trunk_radius = r;
            }

            float ch = XMLloadfloat(s, "cordon_height");
            if (ch != nullvalue_f) {
                unilateralgrapevineparameters.cordon_height = ch;
            }

            float cr = XMLloadfloat(s, "cordon_radius");
            if (cr != nullvalue_f) {
                unilateralgrapevineparameters.cordon_radius = cr;
            }

            float sl = XMLloadfloat(s, "shoot_length");
            if (sl != nullvalue_f) {
                unilateralgrapevineparameters.shoot_length = sl;
            }

            float sr = XMLloadfloat(s, "shoot_radius");
            if (sr != nullvalue_f) {
                unilateralgrapevineparameters.shoot_radius = sr;
            }

            int spc = XMLloadint(s, "shoots_per_cordon");
            if (spc != nullvalue_i) {
                unilateralgrapevineparameters.shoots_per_cordon = uint(spc);
            }

/*
      float spa = XMLloadfloat( s, "shoots_tip_angle" );
      if( spa != nullvalue_f ){
	unilateralgrapevineparameters.shoots_tip_angle = uint(spa);
      }*/

            float lsf = XMLloadfloat(s, "leaf_spacing_fraction");
            if (lsf != nullvalue_f) {
                unilateralgrapevineparameters.leaf_spacing_fraction = lsf;
            }

            float gr = XMLloadfloat(s, "grape_radius");
            if (gr != nullvalue_f) {
                unilateralgrapevineparameters.grape_radius = gr;
            }

            float clr = XMLloadfloat(s, "cluster_radius");
            if (clr != nullvalue_f) {
                unilateralgrapevineparameters.cluster_radius = clr;
            }

            RGBAcolor grape_color = XMLloadrgba(s, "grape_color");
            if (grape_color.a!=0 ) {
                unilateralgrapevineparameters.grape_color = make_RGBcolor(grape_color.r,grape_color.g,grape_color.b);
            }

            int grape_subdivisions = XMLloadint(s, "grape_subdivisions");
            if (grape_subdivisions != nullvalue_i) {
                unilateralgrapevineparameters.grape_subdivisions = uint(grape_subdivisions);
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                unilateralgrapevineparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                unilateralgrapevineparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                unilateralgrapevineparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                unilateralgrapevineparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                unilateralgrapevineparameters.canopy_rotation = canopy_rotation;
            }


            buildCanopy(unilateralgrapevineparameters);

        }


        //GobletGrapevineParameters Canopy
        for (pugi::xml_node s = cgen.child("GobletGrapevineParameters"); s; s = s.next_sibling(
                "GobletGrapevineParameters")) {

            GobletGrapevineParameters gobletgrapevineparameters;

            float leaf_width = XMLloadfloat(s, "leaf_width");
            if (leaf_width != nullvalue_f) {
                gobletgrapevineparameters.leaf_width = leaf_width;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                gobletgrapevineparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                gobletgrapevineparameters.leaf_texture_file = leaf_texture_file;
            }

            std::string wood_texture_file = XMLloadstring(s, "wood_texture_file");
            if ( wood_texture_file != nullvalue_s ) {
                gobletgrapevineparameters.wood_texture_file = wood_texture_file;
            }

            int wood_subdivisions = XMLloadint(s, "wood_subdivisions");
            if (wood_subdivisions != nullvalue_i) {
                gobletgrapevineparameters.wood_subdivisions = wood_subdivisions;
            }

            float h = XMLloadfloat(s, "trunk_height");
            if (h != nullvalue_f) {
                gobletgrapevineparameters.trunk_height = h;
            }

            float r = XMLloadfloat(s, "trunk_radius");
            if (r != nullvalue_f) {
                gobletgrapevineparameters.trunk_radius = r;
            }

            float ch = XMLloadfloat(s, "cordon_height");
            if (ch != nullvalue_f) {
                gobletgrapevineparameters.cordon_height = ch;
            }

            float cr = XMLloadfloat(s, "cordon_radius");
            if (cr != nullvalue_f) {
                gobletgrapevineparameters.cordon_radius = cr;
            }

            float sl = XMLloadfloat(s, "shoot_length");
            if (sl != nullvalue_f) {
                gobletgrapevineparameters.shoot_length = sl;
            }

            float sr = XMLloadfloat(s, "shoot_radius");
            if (sr != nullvalue_f) {
                gobletgrapevineparameters.shoot_radius = sr;
            }

            int spc = XMLloadint(s, "shoots_per_cordon");
            if (spc != nullvalue_i) {
                gobletgrapevineparameters.shoots_per_cordon = uint(spc);
            }
/*
      float spa = XMLloadfloat( s, "shoots_tip_angle" );
      if( spa != nullvalue_f ){
	gobletgrapevineparameters.shoots_tip_angle = uint(spa);
      }*/

            float lsf = XMLloadfloat(s, "leaf_spacing_fraction");
            if (lsf != nullvalue_f) {
                gobletgrapevineparameters.leaf_spacing_fraction = lsf;
            }

            float gr = XMLloadfloat(s, "grape_radius");
            if (gr != nullvalue_f) {
                gobletgrapevineparameters.grape_radius = gr;
            }

            float clr = XMLloadfloat(s, "cluster_radius");
            if (clr != nullvalue_f) {
                gobletgrapevineparameters.cluster_radius = clr;
            }

            RGBAcolor grape_color = XMLloadrgba(s, "grape_color");
            if (grape_color.a != 0 ) {
                gobletgrapevineparameters.grape_color = make_RGBcolor(grape_color.r,grape_color.g,grape_color.b);
            }

            int grape_subdivisions = XMLloadint(s, "grape_subdivisions");
            if (grape_subdivisions != nullvalue_i) {
                gobletgrapevineparameters.grape_subdivisions = uint(grape_subdivisions);
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                gobletgrapevineparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                gobletgrapevineparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                gobletgrapevineparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                gobletgrapevineparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                gobletgrapevineparameters.canopy_rotation = canopy_rotation;
            }


            buildCanopy(gobletgrapevineparameters);

        }

        //StrawberryParameters Canopy
        for (pugi::xml_node s = cgen.child("StrawberryParameters"); s; s = s.next_sibling("StrawberryParameters")) {

            StrawberryParameters strawberryparameters;

            float leaf_length = XMLloadfloat(s, "leaf_length");
            if (leaf_length != nullvalue_f) {
                strawberryparameters.leaf_length = leaf_length;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                strawberryparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                strawberryparameters.leaf_texture_file = leaf_texture_file;
            }

            int stem_subdivisions = XMLloadint(s, "stem_subdivisions");
            if (stem_subdivisions != nullvalue_i) {
                strawberryparameters.stem_subdivisions = stem_subdivisions;
            }

            float stem_radius = XMLloadfloat(s, "stem_radius");
            if (stem_radius != nullvalue_f) {
                strawberryparameters.stem_radius = stem_radius;
            }

            float h = XMLloadfloat(s, "plant_height");
            if (h != nullvalue_f) {
                strawberryparameters.plant_height = h;
            }

            int r = XMLloadint(s, "stems_per_plant");
            if (r != nullvalue_i) {
                strawberryparameters.stems_per_plant = r;
            }

            float gr = XMLloadfloat(s, "fruit_radius");
            if (gr != nullvalue_f) {
                strawberryparameters.fruit_radius = gr;
            }

            float clr = XMLloadfloat(s, "clusters_per_stem");
            if (clr != nullvalue_f) {
                strawberryparameters.clusters_per_stem = clr;
            }

            int fruit_subdivisions = XMLloadint(s, "fruit_subdivisions");
            if (fruit_subdivisions != nullvalue_i) {
                strawberryparameters.fruit_subdivisions = uint(fruit_subdivisions);
            }

            std::string fruit_texture_file = XMLloadstring(s, "fruit_texture_file");
            if ( fruit_texture_file != nullvalue_s ) {
                strawberryparameters.fruit_texture_file = fruit_texture_file;
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                strawberryparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                strawberryparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                strawberryparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                strawberryparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                strawberryparameters.canopy_rotation = canopy_rotation;
            }


            buildCanopy(strawberryparameters);

        }

        //TomatoParameters Canopy
        for (pugi::xml_node s = cgen.child("TomatoParameters"); s; s = s.next_sibling("TomatoParameters")) {

            TomatoParameters tomatoparameters;

            float leaf_length = XMLloadfloat(s, "leaf_length");
            if (leaf_length != nullvalue_f) {
                tomatoparameters.leaf_length = leaf_length;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                tomatoparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                tomatoparameters.leaf_texture_file = leaf_texture_file;
            }

            RGBAcolor shoot_color = XMLloadrgba(s, "shoot_color");
            if (shoot_color.a != 0 ) {
                tomatoparameters.shoot_color = make_RGBcolor(shoot_color.r,shoot_color.g,shoot_color.b);
            }

            int shoot_subdivisions = XMLloadint(s, "shoot_subdivisions");
            if (shoot_subdivisions != nullvalue_i) {
                tomatoparameters.shoot_subdivisions = shoot_subdivisions;
            }

            float h = XMLloadfloat(s, "plant_height");
            if (h != nullvalue_f) {
                tomatoparameters.plant_height = h;
            }

            float gr = XMLloadfloat(s, "fruit_radius");
            if (gr != nullvalue_f) {
                tomatoparameters.fruit_radius = gr;
            }

            RGBAcolor fruit_color = XMLloadrgba(s, "fruit_color");
            if (fruit_color.a != 0 ) {
                tomatoparameters.fruit_color = make_RGBcolor(fruit_color.r,fruit_color.g,fruit_color.b);
            }


            int fruit_subdivisions = XMLloadint(s, "fruit_subdivisions");
            if (fruit_subdivisions != nullvalue_i) {
                tomatoparameters.fruit_subdivisions = uint(fruit_subdivisions);
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                tomatoparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                tomatoparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                tomatoparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                tomatoparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                tomatoparameters.canopy_rotation = canopy_rotation;
            }


            buildCanopy(tomatoparameters);

        }

        //WalnutCanopyParameters Canopy
        for (pugi::xml_node s = cgen.child("WalnutCanopyParameters"); s; s = s.next_sibling("WalnutCanopyParameters")) {

            WalnutCanopyParameters walnutcanopyparameters;

            float leaf_length = XMLloadfloat(s, "leaf_length");
            if (leaf_length != nullvalue_f) {
                walnutcanopyparameters.leaf_length = leaf_length;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                walnutcanopyparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                walnutcanopyparameters.leaf_texture_file = leaf_texture_file;
            }

            int wood_subdivisions = XMLloadint(s, "wood_subdivisions");
            if (wood_subdivisions != nullvalue_i) {
                walnutcanopyparameters.wood_subdivisions = wood_subdivisions;
            }


            float trunk_radius = XMLloadfloat(s, "trunk_radius");
            if (trunk_radius != nullvalue_f) {
                walnutcanopyparameters.trunk_radius = trunk_radius;
            }

            float trunk_height = XMLloadfloat(s, "trunk_height");
            if (trunk_height != nullvalue_f) {
                walnutcanopyparameters.trunk_height = trunk_height;
            }

            vec3 branch_length = XMLloadvec3(s, "branch_length");
            if (branch_length.x != nullvalue_f && branch_length.y != nullvalue_f) {
                walnutcanopyparameters.branch_length = branch_length;
            }

            std::string fruit_texture_file = XMLloadstring(s, "fruit_texture_file");
            if ( fruit_texture_file != nullvalue_s ) {
                walnutcanopyparameters.fruit_texture_file = fruit_texture_file;
            }

            int fruit_subdivisions = XMLloadint(s, "fruit_subdivisions");
            if (fruit_subdivisions != nullvalue_i) {
                walnutcanopyparameters.fruit_subdivisions = uint(fruit_subdivisions);
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                walnutcanopyparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                walnutcanopyparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                walnutcanopyparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                walnutcanopyparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                walnutcanopyparameters.canopy_rotation = canopy_rotation;
            }


            buildCanopy(walnutcanopyparameters);

        }

        //SorghumCanopyParameters Canopy
        for (pugi::xml_node s = cgen.child("SorghumCanopyParameters"); s; s = s.next_sibling("SorghumCanopyParameters")) {

            SorghumCanopyParameters sorghumcanopyparameters;

            int sorghum_stage = XMLloadint(s, "sorghum_stage");
            if (sorghum_stage != nullvalue_i) {
                sorghumcanopyparameters.sorghum_stage = sorghum_stage;
            }
            // STAGE 1
            float s1_stem_length = XMLloadint(s, "s1_stem_length");
            if (s1_stem_length != nullvalue_i) {
                sorghumcanopyparameters.s1_stem_length = s1_stem_length;
            }

            float s1_stem_radius = XMLloadfloat(s, "s1_stem_radius");
            if (s1_stem_radius != nullvalue_f) {
                sorghumcanopyparameters.s1_stem_radius = s1_stem_radius;
            }

            int s1_stem_subdivisions = XMLloadint(s, "s1_stem_subdivisions");
            if (s1_stem_subdivisions != nullvalue_f) {
                sorghumcanopyparameters.s1_stem_subdivisions = uint(s1_stem_subdivisions);
            }

            vec2 s1_leaf_size1 = XMLloadvec2(s, "s1_leaf_size1");
            if (s1_leaf_size1.x != nullvalue_f && s1_leaf_size1.y != nullvalue_f) {
                sorghumcanopyparameters.s1_leaf_size1 = s1_leaf_size1;
            }

            vec2 s1_leaf_size2 = XMLloadvec2(s, "s1_leaf_size2");
            if (s1_leaf_size2.x != nullvalue_f && s1_leaf_size2.y != nullvalue_f) {
                sorghumcanopyparameters.s1_leaf_size2 = s1_leaf_size2;
            }

            vec2 s1_leaf_size3 = XMLloadvec2(s, "s1_leaf_size3");
            if (s1_leaf_size3.x != nullvalue_f && s1_leaf_size3.y != nullvalue_f) {
                sorghumcanopyparameters.s1_leaf_size3 = s1_leaf_size3;
            }

            float s1_leaf1_angle = XMLloadfloat(s, "s1_leaf1_angle");
            if (s1_leaf1_angle != nullvalue_f) {
                sorghumcanopyparameters.s1_leaf1_angle = s1_leaf1_angle;
            }

            float s1_leaf2_angle = XMLloadfloat(s, "s1_leaf2_angle");
            if (s1_leaf2_angle != nullvalue_f) {
                sorghumcanopyparameters.s1_leaf2_angle = s1_leaf2_angle;
            }

            float s1_leaf3_angle = XMLloadfloat(s, "s1_leaf3_angle");
            if (s1_leaf3_angle != nullvalue_f) {
                sorghumcanopyparameters.s1_leaf3_angle = s1_leaf3_angle;
            }

            int2 s1_leaf_subdivisions = XMLloadint2(s, "s1_leaf_subdivisions");
            if (s1_leaf_subdivisions.x != nullvalue_i && s1_leaf_subdivisions.y != nullvalue_i) {
                sorghumcanopyparameters.s1_leaf_subdivisions = s1_leaf_subdivisions;
            }

            std::string s1_leaf_texture_file = XMLloadstring(s, "s1_leaf_texture_file");
            if (s1_leaf_texture_file.compare(nullvalue_s) != 0) {
                sorghumcanopyparameters.s1_leaf_texture_file = s1_leaf_texture_file;
            }
            // STAGE 2
            float s2_stem_length = XMLloadint(s, "s2_stem_length");
            if (s2_stem_length != nullvalue_i) {
                sorghumcanopyparameters.s2_stem_length = s2_stem_length;
            }

            float s2_stem_radius = XMLloadfloat(s, "s2_stem_radius");
            if (s2_stem_radius != nullvalue_f) {
                sorghumcanopyparameters.s2_stem_radius = s2_stem_radius;
            }

            int s2_stem_subdivisions = XMLloadint(s, "s2_stem_subdivisions");
            if (s2_stem_subdivisions != nullvalue_f) {
                sorghumcanopyparameters.s2_stem_subdivisions = uint(s2_stem_subdivisions);
            }

            vec2 s2_leaf_size1 = XMLloadvec2(s, "s2_leaf_size1");
            if (s2_leaf_size1.x != nullvalue_f && s2_leaf_size1.y != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf_size1 = s2_leaf_size1;
            }

            vec2 s2_leaf_size2 = XMLloadvec2(s, "s2_leaf_size2");
            if (s2_leaf_size2.x != nullvalue_f && s2_leaf_size2.y != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf_size2 = s2_leaf_size2;
            }

            vec2 s2_leaf_size3 = XMLloadvec2(s, "s2_leaf_size3");
            if (s2_leaf_size3.x != nullvalue_f && s2_leaf_size3.y != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf_size3 = s2_leaf_size3;
            }

            vec2 s2_leaf_size4 = XMLloadvec2(s, "s2_leaf_size4");
            if (s2_leaf_size4.x != nullvalue_f && s2_leaf_size4.y != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf_size4 = s2_leaf_size4;
            }

            vec2 s2_leaf_size5 = XMLloadvec2(s, "s2_leaf_size5");
            if (s2_leaf_size5.x != nullvalue_f && s2_leaf_size5.y != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf_size5 = s2_leaf_size5;
            }

            float s2_leaf1_angle = XMLloadfloat(s, "s2_leaf1_angle");
            if (s2_leaf1_angle != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf1_angle = s2_leaf1_angle;
            }

            float s2_leaf2_angle = XMLloadfloat(s, "s2_leaf2_angle");
            if (s2_leaf2_angle != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf2_angle = s2_leaf2_angle;
            }

            float s2_leaf3_angle = XMLloadfloat(s, "s2_leaf3_angle");
            if (s2_leaf3_angle != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf3_angle = s2_leaf3_angle;
            }

            float s2_leaf4_angle = XMLloadfloat(s, "s2_leaf4_angle");
            if (s2_leaf4_angle != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf4_angle = s2_leaf4_angle;
            }

            float s2_leaf5_angle = XMLloadfloat(s, "s2_leaf5_angle");
            if (s2_leaf3_angle != nullvalue_f) {
                sorghumcanopyparameters.s2_leaf5_angle = s2_leaf5_angle;
            }

            int2 s2_leaf_subdivisions = XMLloadint2(s, "s2_leaf_subdivisions");
            if (s2_leaf_subdivisions.x != nullvalue_i && s2_leaf_subdivisions.y != nullvalue_i) {
                sorghumcanopyparameters.s2_leaf_subdivisions = s2_leaf_subdivisions;
            }

            std::string s2_leaf_texture_file = XMLloadstring(s, "s2_leaf_texture_file");
            if (s2_leaf_texture_file.compare(nullvalue_s) != 0) {
                sorghumcanopyparameters.s2_leaf_texture_file = s2_leaf_texture_file;
            }
            // STAGE 3
            float s3_stem_length = XMLloadint(s, "s3_stem_length");
            if (s3_stem_length != nullvalue_i) {
                sorghumcanopyparameters.s3_stem_length = s3_stem_length;
            }

            float s3_stem_radius = XMLloadfloat(s, "s3_stem_radius");
            if (s3_stem_radius != nullvalue_f) {
                sorghumcanopyparameters.s3_stem_radius = s3_stem_radius;
            }

            int s3_stem_subdivisions = XMLloadint(s, "s3_stem_subdivisions");
            if (s3_stem_subdivisions != nullvalue_f) {
                sorghumcanopyparameters.s3_stem_subdivisions = uint(s3_stem_subdivisions);
            }

            vec2 s3_leaf_size = XMLloadvec2(s, "s3_leaf_size");
            if (s3_leaf_size.x != nullvalue_f && s3_leaf_size.y != nullvalue_f) {
                sorghumcanopyparameters.s3_leaf_size = s3_leaf_size;
            }

            int2 s3_leaf_subdivisions = XMLloadint2(s, "s3_leaf_subdivisions");
            if (s3_leaf_subdivisions.x != nullvalue_i && s3_leaf_subdivisions.y != nullvalue_i) {
                sorghumcanopyparameters.s3_leaf_subdivisions = s3_leaf_subdivisions;
            }

            int s3_number_of_leaves = XMLloadint(s, "s3_number_of_leaves");
            if (s3_number_of_leaves != nullvalue_i) {
                sorghumcanopyparameters.s3_number_of_leaves = s3_number_of_leaves;
            }

            float s3_mean_leaf_angle = XMLloadfloat(s, "s3_mean_leaf_angle");
            if (s3_mean_leaf_angle != nullvalue_f) {
                sorghumcanopyparameters.s3_mean_leaf_angle = s3_mean_leaf_angle;
            }

            std::string s3_leaf_texture_file = XMLloadstring(s, "s3_leaf_texture_file");
            if (s3_leaf_texture_file.compare(nullvalue_s) != 0) {
                sorghumcanopyparameters.s3_leaf_texture_file = s3_leaf_texture_file;
            }

            // STAGE 4
            float s4_stem_length = XMLloadint(s, "s4_stem_length");
            if (s4_stem_length != nullvalue_i) {
                sorghumcanopyparameters.s4_stem_length = s4_stem_length;
            }

            float s4_stem_radius = XMLloadfloat(s, "s4_stem_radius");
            if (s4_stem_radius != nullvalue_f) {
                sorghumcanopyparameters.s4_stem_radius = s4_stem_radius;
            }

            int s4_stem_subdivisions = XMLloadint(s, "s4_stem_subdivisions");
            if (s4_stem_subdivisions != nullvalue_f) {
                sorghumcanopyparameters.s4_stem_subdivisions = uint(s4_stem_subdivisions);
            }

            vec2 s4_panicle_size = XMLloadvec2(s, "s4_panicle_size");
            if (s4_panicle_size.x != nullvalue_f && s4_panicle_size.y != nullvalue_f) {
                sorghumcanopyparameters.s4_panicle_size = s4_panicle_size;
            }

            int s4_panicle_subdivisions = XMLloadint(s, "s4_panicle_subdivisions");
            if (s4_panicle_subdivisions != nullvalue_f) {
                sorghumcanopyparameters.s4_panicle_subdivisions = uint(s4_panicle_subdivisions);
            }

            std::string s4_seed_texture_file = XMLloadstring(s, "s4_seed_texture_file");
            if (s4_seed_texture_file.compare(nullvalue_s) != 0) {
                sorghumcanopyparameters.s4_seed_texture_file = s4_seed_texture_file;
            }

            vec2 s4_leaf_size = XMLloadvec2(s, "s4_leaf_size");
            if (s4_leaf_size.x != nullvalue_f && s4_leaf_size.y != nullvalue_f) {
                sorghumcanopyparameters.s4_leaf_size = s4_leaf_size;
            }

            int2 s4_leaf_subdivisions = XMLloadint2(s, "s4_leaf_subdivisions");
            if (s4_leaf_subdivisions.x != nullvalue_i && s4_leaf_subdivisions.y != nullvalue_i) {
                sorghumcanopyparameters.s4_leaf_subdivisions = s4_leaf_subdivisions;
            }

            int s4_number_of_leaves = XMLloadint(s, "s4_number_of_leaves");
            if (s4_number_of_leaves != nullvalue_i) {
                sorghumcanopyparameters.s4_number_of_leaves = s4_number_of_leaves;
            }

            float s4_mean_leaf_angle = XMLloadfloat(s, "s4_mean_leaf_angle");
            if (s4_mean_leaf_angle != nullvalue_f) {
                sorghumcanopyparameters.s4_mean_leaf_angle = s4_mean_leaf_angle;
            }

            std::string s4_leaf_texture_file = XMLloadstring(s, "s4_leaf_texture_file");
            if (s4_leaf_texture_file.compare(nullvalue_s) != 0) {
                sorghumcanopyparameters.s4_leaf_texture_file = s4_leaf_texture_file;
            }

            // STAGE 5
            float s5_stem_length = XMLloadint(s, "s5_stem_length");
            if (s5_stem_length != nullvalue_i) {
                sorghumcanopyparameters.s5_stem_length = s5_stem_length;
            }

            float s5_stem_radius = XMLloadfloat(s, "s5_stem_radius");
            if (s5_stem_radius != nullvalue_f) {
                sorghumcanopyparameters.s5_stem_radius = s5_stem_radius;
            }

            float s5_stem_bend = XMLloadfloat(s, "s5_stem_bend");
            if (s5_stem_bend != nullvalue_f) {
                sorghumcanopyparameters.s5_stem_bend = s5_stem_bend;
            }

            int s5_stem_subdivisions = XMLloadint(s, "s5_stem_subdivisions");
            if (s5_stem_subdivisions != nullvalue_f) {
                sorghumcanopyparameters.s5_stem_subdivisions = uint(s5_stem_subdivisions);
            }

            vec2 s5_panicle_size = XMLloadvec2(s, "s5_panicle_size");
            if (s5_panicle_size.x != nullvalue_f && s5_panicle_size.y != nullvalue_f) {
                sorghumcanopyparameters.s5_panicle_size = s5_panicle_size;
            }

            int s5_panicle_subdivisions = XMLloadint(s, "s5_panicle_subdivisions");
            if (s5_panicle_subdivisions != nullvalue_f) {
                sorghumcanopyparameters.s5_panicle_subdivisions = uint(s5_panicle_subdivisions);
            }

            std::string s5_seed_texture_file = XMLloadstring(s, "s5_seed_texture_file");
            if (s5_seed_texture_file.compare(nullvalue_s) != 0) {
                sorghumcanopyparameters.s5_seed_texture_file = s5_seed_texture_file;
            }

            vec2 s5_leaf_size = XMLloadvec2(s, "s5_leaf_size");
            if (s5_leaf_size.x != nullvalue_f && s5_leaf_size.y != nullvalue_f) {
                sorghumcanopyparameters.s5_leaf_size = s5_leaf_size;
            }

            int2 s5_leaf_subdivisions = XMLloadint2(s, "s5_leaf_subdivisions");
            if (s5_leaf_subdivisions.x != nullvalue_i && s5_leaf_subdivisions.y != nullvalue_i) {
                sorghumcanopyparameters.s5_leaf_subdivisions = s5_leaf_subdivisions;
            }

            int s5_number_of_leaves = XMLloadint(s, "s5_number_of_leaves");
            if (s5_number_of_leaves != nullvalue_i) {
                sorghumcanopyparameters.s5_number_of_leaves = s5_number_of_leaves;
            }

            float s5_mean_leaf_angle = XMLloadfloat(s, "s5_mean_leaf_angle");
            if (s5_mean_leaf_angle != nullvalue_f) {
                sorghumcanopyparameters.s5_mean_leaf_angle = s5_mean_leaf_angle;
            }

            std::string s5_leaf_texture_file = XMLloadstring(s, "s5_leaf_texture_file");
            if (s5_leaf_texture_file.compare(nullvalue_s) != 0) {
                sorghumcanopyparameters.s5_leaf_texture_file = s5_leaf_texture_file;
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                sorghumcanopyparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                sorghumcanopyparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                sorghumcanopyparameters.plant_count = plant_count;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                sorghumcanopyparameters.canopy_origin = canopy_origin;
            }

            buildCanopy(sorghumcanopyparameters);

        }

        //BeanParameters Canopy
        for (pugi::xml_node s = cgen.child("BeanParameters"); s; s = s.next_sibling("BeanParameters")) {

            BeanParameters beanparameters;

            float leaf_length = XMLloadfloat(s, "leaf_length");
            if (leaf_length != nullvalue_f) {
                beanparameters.leaf_length = leaf_length;
            }

            int2 leaf_subdivisions = XMLloadint2(s, "leaf_subdivisions");
            if (leaf_subdivisions.x != nullvalue_i && leaf_subdivisions.y != nullvalue_i) {
                beanparameters.leaf_subdivisions = leaf_subdivisions;
            }

            std::string leaf_texture_file = XMLloadstring(s, "leaf_texture_file");
            if ( leaf_texture_file != nullvalue_s ) {
                beanparameters.leaf_texture_file = leaf_texture_file;
            }

            int shoot_subdivisions = XMLloadint(s, "shoot_subdivisions");
            if (shoot_subdivisions != nullvalue_i) {
                beanparameters.shoot_subdivisions = shoot_subdivisions;
            }

            float stem_radius = XMLloadfloat(s, "stem_radius");
            if (stem_radius != nullvalue_f) {
                beanparameters.stem_radius = stem_radius;
            }

            RGBAcolor shoot_color = XMLloadrgba(s, "shoot_color");
            if( shoot_color.a != 0 ){
                beanparameters.shoot_color = make_RGBcolor(shoot_color.r,shoot_color.g,shoot_color.b);
            }

            float stem_length = XMLloadfloat(s, "stem_length");
            if (stem_length != nullvalue_f ) {
                beanparameters.stem_length = stem_length;
            }

            float leaflet_length = XMLloadfloat(s, "leaflet_length");
            if (leaflet_length != nullvalue_f ) {
                beanparameters.leaflet_length = leaflet_length;
            }

            float pod_length = XMLloadfloat(s, "pod_length");
            if (pod_length != nullvalue_f ) {
                beanparameters.pod_length = pod_length;
            }

            RGBAcolor pod_color = XMLloadrgba(s, "pod_color");
            if( pod_color.a != 0 ){
                beanparameters.pod_color = make_RGBcolor(pod_color.r,pod_color.g,pod_color.b);
            }

            int pod_subdivisions = XMLloadint(s, "pod_subdivisions");
            if (pod_subdivisions != nullvalue_i ) {
                beanparameters.pod_subdivisions = pod_subdivisions;
            }

            float plant_spacing = XMLloadfloat(s, "plant_spacing");
            if (plant_spacing != nullvalue_f) {
                beanparameters.plant_spacing = plant_spacing;
            }

            float row_spacing = XMLloadfloat(s, "row_spacing");
            if (row_spacing != nullvalue_f) {
                beanparameters.row_spacing = row_spacing;
            }

            int2 plant_count = XMLloadint2(s, "plant_count");
            if (plant_count.x != nullvalue_i && plant_count.y != nullvalue_i) {
                beanparameters.plant_count = plant_count;
            }

            float germination_probability = XMLloadfloat(s, "germination_probability");
            if (germination_probability != nullvalue_f) {
                beanparameters.germination_probability = germination_probability;
            }

            vec3 canopy_origin = XMLloadvec3(s, "canopy_origin");
            if (canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f) {
                beanparameters.canopy_origin = canopy_origin;
            }

            float canopy_rotation = XMLloadfloat(s, "canopy_rotation");
            if (canopy_rotation != nullvalue_f) {
                beanparameters.canopy_rotation = canopy_rotation;
            }

            buildCanopy(beanparameters);

        }

        //Ground
        for (pugi::xml_node s = cgen.child("Ground"); s; s = s.next_sibling("Ground")) {

            vec3 origin = XMLloadvec3(s, "origin");
            if (origin.x == nullvalue_f || origin.y == nullvalue_f || origin.z == nullvalue_f) {
                origin = make_vec3(0, 0, 0);
                if (printmessages) {
                    std::cout << "WARNING: origin not provided for ground in file " << filename << std::endl;
                }
            }

            vec2 extent = XMLloadvec2(s, "extent");
            if (extent.x == nullvalue_f || extent.y == nullvalue_f) {
                extent = make_vec2(1, 1);
                if (printmessages) {
                    std::cout << "WARNING: horizontal extent not provided for ground in file " << filename << std::endl;
                }
            }

            int2 texture_subtiles = XMLloadint2(s, "texture_subtiles");
            if (texture_subtiles.x == nullvalue_i || texture_subtiles.y == nullvalue_i) {
                texture_subtiles = make_int2(1, 1);
            }

            int2 texture_subpatches = XMLloadint2(s, "texture_subpatches");
            if (texture_subpatches.x == nullvalue_i || texture_subpatches.y == nullvalue_i) {
                texture_subpatches = make_int2(1, 1);
            }

            std::string texturefile = XMLloadstring(s, "ground_texture_file");
            if ( texturefile == nullvalue_s ) {
                texturefile = "plugins/canopygenerator/textures/dirt.jpg";
                if (printmessages) {
                    std::cout << "WARNING: texture map file not provided for ground in file " << filename << std::endl;
                }
            }

            float rotation = XMLloadfloat(s, "rotation");
            if (rotation == nullvalue_f) {
                rotation = 0;
            }

            buildGround(origin, extent, texture_subtiles, texture_subpatches, texturefile.c_str(), rotation);

        }

    }

    std::cout << "done." << std::endl;

}

void CanopyGenerator::buildGround(const vec3 &ground_origin, const vec2 &ground_extent, const int2 &texture_subtiles, const int2 &texture_subpatches, const char* ground_texture_file  ){
    buildGround( ground_origin, ground_extent, texture_subtiles, texture_subpatches, ground_texture_file, 0.f );
}

void CanopyGenerator::buildGround(const vec3 &ground_origin, const vec2 &ground_extent, const int2 &texture_subtiles, const int2 &texture_subpatches, const char* ground_texture_file, float ground_rotation  ){

    if( printmessages ){
        std::cout << "Ground geometry..." << std::flush;
    }

    vec2 dx_tile( ground_extent.x/float(texture_subtiles.x), ground_extent.y/float(texture_subtiles.y) );

    vec2 dx_subpatch( dx_tile.x/float(texture_subpatches.x), dx_tile.y/float(texture_subpatches.y) );

    std::vector<uint> UUIDs;
    for( int j=0; j<texture_subtiles.y; j++ ){
        for( int i=0; i<texture_subtiles.x; i++ ){

            vec3 center = ground_origin + make_vec3( -0.5f*ground_extent.x+(float(i)+0.5f)*dx_tile.x, -0.5f*ground_extent.y+(float(j)+0.5f)*dx_tile.y, 0 );

            if( ground_rotation!=0 ){
                center = rotatePointAboutLine( center, ground_origin, make_vec3(0,0,1), ground_rotation );
            }

            UUIDs = context->addTile( center, dx_tile, make_SphericalCoord(0,-ground_rotation), texture_subpatches, ground_texture_file );

            UUID_ground.insert( UUID_ground.begin(), UUIDs.begin(), UUIDs.end() );

        }
    }

    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Ground consists of " << UUID_ground.size() << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const HomogeneousCanopyParameters &params ){

    if( printmessages ){
        std::cout << "Building homogeneous canopy..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    UUID_leaf.resize(1);

    float solidFractionx;
    if(params.leaf_texture_file.empty()){
        solidFractionx = 1.0;
    }else{
        helios::Texture texture(params.leaf_texture_file.c_str());
        solidFractionx = texture.getSolidFraction();
    }

    float leafArea = params.leaf_size.x*params.leaf_size.y*solidFractionx;
    int Nleaves = (int)lroundf(params.leaf_area_index*params.canopy_extent.x*params.canopy_extent.y/leafArea);

    float Lmax = sqrtf(params.leaf_size.x*params.leaf_size.x + params.leaf_size.y*params.leaf_size.y);

    uint ID0;
    if( params.leaf_texture_file.empty() ){
        ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_color );
    }else{
        ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() );
    }

    for( int i=0; i<Nleaves; i++ ){

        float rx = unif_distribution(generator);
        float ry = unif_distribution(generator);
        float rz = unif_distribution(generator);

        float rp = unif_distribution(generator);

        vec3 position;


        if(params.buffer == "z"){
            position = params.canopy_origin + make_vec3( (-0.5f+rx)*params.canopy_extent.x, (-0.5f+ry)*params.canopy_extent.y,0.5f*Lmax+rz*(params.canopy_height-Lmax) );
        }else if(params.buffer == "xyz"){
            position = params.canopy_origin + make_vec3( 0.5f*Lmax + (rx)*(params.canopy_extent.x-Lmax) + -0.5f*params.canopy_extent.x,0.5f*Lmax + (ry)*(params.canopy_extent.y-Lmax) + -0.5f*params.canopy_extent.y,0.5f*Lmax + rz*(params.canopy_height-Lmax) );
        }else{
            position = params.canopy_origin + make_vec3( (-0.5f+rx)*params.canopy_extent.x,(-0.5f+ry)*params.canopy_extent.y,(rz)*(params.canopy_height) );
        }

        SphericalCoord rotation( 1.f, sampleLeafPDF(params.leaf_angle_distribution.c_str()), 2.f*float(M_PI)*rp );

        uint ID = context->copyObject(ID0);
        context->getObjectPointer(ID)->rotate(-rotation.elevation,"y");
        context->getObjectPointer(ID)->rotate(rotation.azimuth,"z");
        context->getObjectPointer(ID)->translate(position);

        std::vector<uint> UUID = context->getObjectPointer(ID)->getPrimitiveUUIDs();

        UUID_leaf.front().push_back(UUID);

    }

    context->deleteObject(ID0);


    if( printmessages ){
        std::cout << "done." << std::endl;
        std::vector<uint> UUIDs_all = getAllUUIDs(0);
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << UUIDs_all.size() << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const SphericalCrownsCanopyParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of spherical crowns..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec3 r = params.crown_radius;

    float solidFractionx;
    if(params.leaf_texture_file.empty()){
        solidFractionx = 1.0;
    }else{
        helios::Texture texture(params.leaf_texture_file.c_str());
        solidFractionx = texture.getSolidFraction();
    }

    float leafArea = params.leaf_size.x*params.leaf_size.y*solidFractionx;
    int Nleaves = (int)lroundf(4.f/3.f*float(M_PI)*r.x*r.y*r.z*params.leaf_area_density/leafArea);

    vec2 canopy_extent( params.plant_spacing.x*float(params.plant_count.x), params.plant_spacing.y*float(params.plant_count.y) );

    std::string cconfig = params.canopy_configuration;
    if( cconfig !="uniform" && cconfig !="random" ){
        std::cout << "WARNING: Unknown canopy configuration parameter for spherical crowns canopy: " << cconfig << ". Using uniformly spaced configuration." << std::endl;
        cconfig = "uniform";
    }

    UUID_leaf.resize(params.plant_count.x*params.plant_count.y);

    uint ID0;
    if( params.leaf_texture_file.empty() ){
        ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_color );
    }else{
        ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() );
    }

    uint plant_ID = 0;
    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center;
            if( cconfig=="uniform" ){
                center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing.x, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.plant_spacing.y, r.z );
            }else if( cconfig=="random" ){
                float rx = unif_distribution(generator);
                float ry = unif_distribution(generator);
                center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+float(i)*params.plant_spacing.x+r.x+(params.plant_spacing.x-2.f*r.x)*rx, -0.5f*canopy_extent.y+float(j)*params.plant_spacing.y+r.y+(params.plant_spacing.y-2.f*r.y)*ry, r.z );
            }

            if( params.canopy_rotation!=0 ){
                center = rotatePointAboutLine( center, params.canopy_origin, make_vec3(0,0,1), params.canopy_rotation );
            }

            for (int l=0; l<Nleaves; l++ ){

                vec3 position(-9999,-9999,-9999);

                while( pow(position.x,2)/pow(params.crown_radius.x,2)+pow(position.y,2)/pow(params.crown_radius.y,2)+pow(position.z,2)/pow(params.crown_radius.z,2) > 1.f ){

                    float u = unif_distribution(generator);
                    float v = unif_distribution(generator);
                    float w = unif_distribution(generator);

                    position = make_vec3( (-1+2.f*u)*r.x, (-1+2.f*v)*r.y, (-1+2.f*w)*r.z );

                }

                float theta = sampleLeafPDF(params.leaf_angle_distribution.c_str());
                float phi = 2.f*float(M_PI)*unif_distribution(generator);

                uint ID = context->copyObject(ID0);
                context->getObjectPointer(ID)->rotate(-theta,"y");
                context->getObjectPointer(ID)->rotate(phi,"z");
                context->getObjectPointer(ID)->translate(center+position);

                std::vector<uint> UUID = context->getObjectPointer(ID)->getPrimitiveUUIDs();

                UUID_leaf.at(plant_ID).push_back(UUID);

            }

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

            plant_ID++;

        }
    }

    context->deleteObject(ID0);

    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const ConicalCrownsCanopyParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of conical crowns..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    float r = params.crown_radius;
    float h = params.crown_height;

    float solidFractionx;
    if(params.leaf_texture_file.empty()){
        solidFractionx = 1.0;
    }else{
        helios::Texture texture(params.leaf_texture_file.c_str());
        solidFractionx = texture.getSolidFraction();
    }

    float leafArea = params.leaf_size.x*params.leaf_size.y*solidFractionx;
    int Nleaves = (int)lroundf(1.f/3.f*float(M_PI)*r*r*h*params.leaf_area_density/leafArea);

    vec2 canopy_extent( params.plant_spacing.x*float(params.plant_count.x), params.plant_spacing.y*float(params.plant_count.y) );

    std::string cconfig = params.canopy_configuration;
    if( cconfig !="uniform" && cconfig !="random" ){
        std::cout << "WARNING: Unknown canopy configuration parameter for conical crowns canopy: " << cconfig << ". Using uniformly spaced configuration." << std::endl;
        cconfig = "uniform";
    }

    UUID_leaf.resize(params.plant_count.x*params.plant_count.y);

    uint ID0;
    if( params.leaf_texture_file.empty() ){
        ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_color );
    }else{
        ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() );
    }

    uint plant_ID = 0;
    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center;
            if( cconfig=="uniform" ){
                center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing.x, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.plant_spacing.y, 0 );
            }else if( cconfig=="random" ){
                float rx = unif_distribution(generator);
                float ry = unif_distribution(generator);
                center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+float(i)*params.plant_spacing.x+r+(params.plant_spacing.x-2.f*r)*rx, -0.5f*canopy_extent.y+float(j)*params.plant_spacing.y+r+(params.plant_spacing.y-2.f*r)*ry, 0 );
            }

            if( params.canopy_rotation!=0 ){
                center = rotatePointAboutLine( center, params.canopy_origin, make_vec3(0,0,1), params.canopy_rotation );
            }

            for (int l=0; l<Nleaves; l++ ){

                vec3 position(-9999,-9999,-9999);

                while( (powf(position.x,2) + powf(position.y,2) )/powf(r/h,2) > powf(h-position.z,2) ){

                    float u = unif_distribution(generator);
                    float v = unif_distribution(generator);
                    float w = unif_distribution(generator);

                    position = make_vec3( (-1+2.f*u)*r, (-1+2.f*v)*r, w*h );

                }

                float theta = sampleLeafPDF(params.leaf_angle_distribution.c_str());
                float phi = 2.f*float(M_PI)*unif_distribution(generator);

                uint ID = context->copyObject(ID0);
                context->getObjectPointer(ID)->rotate(-theta,"y");
                context->getObjectPointer(ID)->rotate(phi,"z");
                context->getObjectPointer(ID)->translate(center+position);

                std::vector<uint> UUID = context->getObjectPointer(ID)->getPrimitiveUUIDs();

                UUID_leaf.at(plant_ID).push_back(UUID);

            }

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

            plant_ID++;

        }
    }

    context->deleteObject(ID0);

    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const VSPGrapevineParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of VSP grapevine..." << std::flush;
    }

    if( params.cordon_height<params.trunk_height ){
        std::cout << "failed." << std::endl;
        throw(std::runtime_error("ERROR (CanopyGenerator::buildCanopy)): Cannot build VSP grapevine canopy. Cordon height cannot be less than the trunk height."));
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            uint plant_ID = grapevineVSP( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }


    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const SplitGrapevineParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of split trellis grapevine..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            uint plant_ID = grapevineSplit( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }


    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const UnilateralGrapevineParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of unilateral trellis grapevine..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            uint plant_ID = grapevineUnilateral( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }


    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const GobletGrapevineParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of goblet trellis grapevine..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            uint plant_ID = grapevineGoblet( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }


    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const WhiteSpruceCanopyParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of white spruce trees..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing.x*float(params.plant_count.x), params.plant_spacing.y*float(params.plant_count.y) );

    std::string cconfig = params.canopy_configuration;
    if( cconfig !="uniform" && cconfig !="random" ){
        std::cout << "WARNING: Unknown canopy configuration parameter for white spruce canopy: " << cconfig << ". Using uniformly spaced configuration." << std::endl;
        cconfig = "uniform";
    }

    float r = params.crown_radius;

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center;
            if( cconfig !="uniform" ){
                center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing.x, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.plant_spacing.y, 0 );
            }else if( cconfig !="random" ){
                float rx = unif_distribution(generator);
                float ry = unif_distribution(generator);
                center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+float(i)*params.plant_spacing.x+r+(params.plant_spacing.x-2.f*r)*rx, -0.5f*canopy_extent.y+float(j)*params.plant_spacing.y+r+(params.plant_spacing.y-2.f*r)*ry, 0 );
            }

            if( params.canopy_rotation!=0 ){
                center = rotatePointAboutLine( center, params.canopy_origin, make_vec3(0,0,1), params.canopy_rotation );
            }

            uint plant_ID = whitespruce( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }

    if( printmessages ){
        std::cout << "done." << std::endl;
        std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const TomatoParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of tomato plants..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            uint plant_ID = tomato( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }

    if( printmessages ){
        std::cout << "done." << std::endl;
        //std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const StrawberryParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of strawberry plants..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            uint plant_ID = strawberry( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }

    if( printmessages ){
        std::cout << "done." << std::endl;
        //std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const WalnutCanopyParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of walnut trees..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            uint plant_ID = walnut( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }

    if( printmessages ){
        std::cout << "done." << std::endl;
        //std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy( const SorghumCanopyParameters &params ){

    if( printmessages ){
        std::cout << "Building canopy of sorghum plants..." << std::flush;
    }

    std::uniform_real_distribution<float> unif_distribution;

    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

    uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

            uint plant_ID = sorghum( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

        }
    }

    if( printmessages ){
        std::cout << "done." << std::endl;
        //std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
    }

}

void CanopyGenerator::buildCanopy(const BeanParameters &params ){

  if( printmessages ){
    std::cout << "Building canopy of bean plants..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
    for( int j=0; j<params.plant_count.y; j++ ){
        for( int i=0; i<params.plant_count.x; i++ ){

            if ( unif_distribution(generator) > params.germination_probability) {
                continue;
            }

            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*params.plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*params.row_spacing, 0 );

            if( params.canopy_rotation!=0 ){
                center = rotatePointAboutLine( center, params.canopy_origin, make_vec3(0,0,1), params.canopy_rotation );
            }

            bean( params, center );

            std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();

            plant_ID++;

    }
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }

}

float getVariation( float V, std::minstd_rand0& generator ){

    std::uniform_real_distribution<float> unif_distribution;

    return -V + 2.f*unif_distribution(generator)*V;

}

float CanopyGenerator::sampleLeafAngle(const std::vector<float> &leafAngleDist ){

    std::vector<float> gL = leafAngleDist;

    float dTheta = 0.5f*float(M_PI)/float(gL.size());

    //make sure PDF is properly normalized
    float norm = 0;
    for( float i : gL){
        norm += i*dTheta;
    }
    for( float & i : gL){
        i /= norm;
    }
    norm = 0;
    for( float i : gL){
        norm += i*dTheta;
    }
    assert( fabs(norm-1)<0.001 );

    //calculate the leaf angle CDF
    std::vector<float> leafAngleCDF;
    leafAngleCDF.resize( gL.size() );
    float tsum = 0;
    for( int i=0; i<gL.size(); i++ ){
        tsum += gL.at(i)*dTheta;
        leafAngleCDF.at(i) = tsum;
    }

    assert( fabs(tsum-1.f)<0.001 );

    //draw from leaf angle PDF
    std::uniform_real_distribution<float> unif_distribution;
    float rt = unif_distribution(generator);

    float theta = -1;
    for( int i=0; i<gL.size(); i++ ){
        if( rt<leafAngleCDF.at(i) ){
            theta = (float(i)+unif_distribution(generator))*dTheta;
            break;
        }
    }

    assert( theta!=-1 );

    return theta;

}

void CanopyGenerator::cleanDeletedUUIDs( std::vector<uint> &UUIDs ){
    for( int p=UUIDs.size()-1; p>=0; p-- ){
        if( !context->doesPrimitiveExist(UUIDs.at(p)) ){
            std::swap( UUIDs.at(p), UUIDs.back() );
            UUIDs.pop_back();
        }
    }
}

void CanopyGenerator::cleanDeletedUUIDs( std::vector<std::vector<uint> > &UUIDs ){
    for( auto & UUID : UUIDs){
        cleanDeletedUUIDs( UUID );
    }
}

void CanopyGenerator::cleanDeletedUUIDs( std::vector<std::vector<std::vector<uint> > > &UUIDs ){
    for( auto & UUID : UUIDs){
        cleanDeletedUUIDs( UUID );
    }
}

std::vector<uint> CanopyGenerator::getTrunkUUIDs(uint PlantID ){
    if( PlantID>=UUID_trunk.size() ){
        throw( std::runtime_error("ERROR (CanopyGenerator::getTrunkUUIDs): Cannot get UUIDs for plant " + std::to_string(PlantID) + " because only " + std::to_string(UUID_trunk.size()) + " plants have been built.") );
    }

    cleanDeletedUUIDs(UUID_trunk.at(PlantID));

    return UUID_trunk.at(PlantID);

}

std::vector<uint> CanopyGenerator::getTrunkUUIDs(){
    std::vector<uint> UUID_flat = flatten( UUID_trunk );
    cleanDeletedUUIDs(UUID_flat);
    return UUID_flat;
}

std::vector<uint> CanopyGenerator::getBranchUUIDs(uint PlantID ){
    if( PlantID>=UUID_branch.size() ){
        throw(std::runtime_error("ERROR (CanopyGenerator::getBranchUUIDs): Cannot get UUIDs for plant " + std::to_string(PlantID) + " because only " + std::to_string(UUID_branch.size()) + " plants have been built."));
    }

    cleanDeletedUUIDs(UUID_branch.at(PlantID));

    return UUID_branch.at(PlantID);

}

std::vector<uint> CanopyGenerator::getBranchUUIDs(){
    std::vector<uint> UUID_flat = flatten( UUID_branch );
    cleanDeletedUUIDs(UUID_flat);
    return UUID_flat;
}

std::vector<std::vector<uint> > CanopyGenerator::getLeafUUIDs(uint PlantID ){
    if( PlantID>=UUID_leaf.size() ){
        throw(std::runtime_error("ERROR (CanopyGenerator::getLeafUUIDs): Cannot get UUIDs for plant " + std::to_string(PlantID) + " because only " + std::to_string(UUID_leaf.size()) + " plants have been built."));
    }

    cleanDeletedUUIDs(UUID_leaf.at(PlantID));

    return UUID_leaf.at(PlantID);
}

std::vector<uint> CanopyGenerator::getLeafUUIDs(){
    std::vector<uint> UUID_flat = flatten( UUID_leaf );
    cleanDeletedUUIDs(UUID_flat);
    return UUID_flat;
}

std::vector<std::vector<std::vector<uint> > > CanopyGenerator::getFruitUUIDs(uint PlantID ){
    if( PlantID>=UUID_fruit.size() ){
        throw(std::runtime_error("ERROR (CanopyGenerator::getFruitUUIDs): Cannot get UUIDs for plant " + std::to_string(PlantID) + " because only " + std::to_string(UUID_fruit.size()) + " plants have been built."));
    }

    cleanDeletedUUIDs(UUID_fruit.at(PlantID));

    return UUID_fruit.at(PlantID);

}

std::vector<uint> CanopyGenerator::getFruitUUIDs(){

    std::vector<uint> UUIDs_flat, U;

    for( auto & p : UUID_fruit){
        U = flatten( p );
        UUIDs_flat.insert( UUIDs_flat.end(), U.begin(), U.end() );
    }

    cleanDeletedUUIDs(UUIDs_flat);
    return UUIDs_flat;
}

std::vector<uint> CanopyGenerator::getGroundUUIDs(){

    std::vector<uint> UUID;

    for( uint i : UUID_ground){

        if( context->doesPrimitiveExist(i) ){
            UUID.push_back(i);
        }

    }

    return UUID;

}

std::vector<uint> CanopyGenerator::getAllUUIDs(uint PlantID ){
    std::vector<uint> UUIDs;
    if( UUID_trunk.size()>PlantID ){
        UUIDs.insert(UUIDs.end(),UUID_trunk.at(PlantID).begin(),UUID_trunk.at(PlantID).end());
    }
    if( UUID_branch.size()>PlantID ){
        UUIDs.insert(UUIDs.end(),UUID_branch.at(PlantID).begin(),UUID_branch.at(PlantID).end());
    }
    if( UUID_leaf.size()>PlantID ){
        for( auto & i : UUID_leaf.at(PlantID)){
            UUIDs.insert(UUIDs.end(),i.begin(),i.end());
        }
    }
    if( UUID_fruit.size()>PlantID ){
        for( auto & j : UUID_fruit.at(PlantID)){
            for( auto & i : j ){
                UUIDs.insert(UUIDs.end(),i.begin(),i.end());
            }
        }
    }

    cleanDeletedUUIDs(UUIDs);

    return UUIDs;
}

uint CanopyGenerator::getPlantCount(){
    return UUID_leaf.size();
}

void CanopyGenerator::seedRandomGenerator(uint seed ){
    generator.seed(seed);
}

void CanopyGenerator::disableMessages(){
    printmessages=false;
}

void CanopyGenerator::enableMessages(){
    printmessages=true;
}

helios::vec3 interpolateTube(const std::vector<vec3> &P, float frac ){

    assert( frac>=0 && frac<=1 );
    assert( !P.empty() );

    float dl=0.f;
    for( int i=0; i<P.size()-1; i++ ){
        dl+=(P.at(i+1)-P.at(i)).magnitude();
    }

    float f = 0;
    for( int i=0; i<P.size()-1; i++ ){

        float dseg = (P.at(i+1)-P.at(i)).magnitude();

        float fplus = f+dseg/dl;

        if( fplus>=1.f ){
            fplus = 1.f+1e-3;
        }

        if( frac>=f && (frac<=fplus || fabs(frac-fplus)<0.0001 ) ){

            vec3 V = P.at(i) + (frac-f)/(fplus-f)*(P.at(i+1)-P.at(i));

            return V;
        }

        f=fplus;
    }

    return P.front();

}

float interpolateTube(const std::vector<float> &P, float frac ){

    assert( frac>=0 && frac<=1 );
    assert( !P.empty() );

    float dl=0.f;
    for( int i=0; i<P.size()-1; i++ ){
        dl+=(P.at(i+1)-P.at(i));
    }

    float f = 0;
    for( int i=0; i<P.size()-1; i++ ){

        float dseg = (P.at(i+1)-P.at(i));

        float fplus = f+dseg/dl;

        if( fplus>=1.f ){
            fplus = 1.f+1e-3;
        }

        if( frac>=f && (frac<=fplus || fabs(frac-fplus)<0.0001 ) ){

            float V = P.at(i) + (frac-f)/(fplus-f)*(P.at(i+1)-P.at(i));

            return V;
        }

        f=fplus;
    }

    return P.front();

}

float evaluateCDFresid(float thetaL, std::vector<float> &ru_v, const void *a_distribution) {

    assert(!ru_v.empty());
    assert( ru_v.front()>=0.f && ru_v.front()<=1.f );

    float ru = ru_v.front();

    const char* distribution = reinterpret_cast<const char*>(a_distribution);

    float CDFresid = INFINITY;
    if( strcmp(distribution,"planophile")==0 ){
        CDFresid = ru - 2.f/float(M_PI)*(thetaL+0.5f*sinf(2.f*thetaL));
    }else if( strcmp(distribution,"erectophile")==0 ){
        CDFresid = ru - 2.f/float(M_PI)*(thetaL-0.5f*sinf(2.f*thetaL));
    }else if( strcmp(distribution,"plagiophile")==0 ){
        CDFresid = ru - 2.f/float(M_PI)*(thetaL-0.25f*sinf(4.f*thetaL));
    }else if( strcmp(distribution,"extremophile")==0 ){
        CDFresid = ru - 2.f/float(M_PI)*(thetaL+0.25f*sinf(4.f*thetaL));
    }

    return CDFresid;

}

float CanopyGenerator::sampleLeafPDF( const char* distribution ){

    float thetaL;

    std::uniform_real_distribution<float> unif_distribution;

    float ru = unif_distribution(generator);

    if( strcmp(distribution,"spherical")==0 ){
        thetaL = acos_safe(1.f-ru);
    }else if( strcmp(distribution,"uniform")==0 ){
        thetaL = ru*0.5f*float(M_PI);
    }else if( strcmp(distribution,"planophile")==0 || strcmp(distribution,"erectophile")==0 || strcmp(distribution,"plagiophile")==0 || strcmp(distribution,"extremophile")==0  ){

        std::vector<float> args{ru};

        thetaL = fzero( evaluateCDFresid, args, distribution, 0.25f );

    }else{
        throw(std::runtime_error("ERROR (sampleLeafPDF): Invalid leaf angle distribution of " + std::string(distribution) + " specified."));
    }

    return thetaL;

}

void CanopyGenerator::createElementLabels() {
    enable_element_labels = true;
}

void CanopyGenerator::disableElementLabels() {
    enable_element_labels = false;
}
