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

//! Float null value in XML parameters
static float nullvalue_f = 99999;
//! Int null value in XML parameters
static int nullvalue_i = 99999;
//! String null value in XML parameters
static std::string nullvalue_s = "99999";

BaseCanopyParameters::BaseCanopyParameters(){
    canopy_origin = make_vec3(0,0,0);
    canopy_rotation = 0;
}

BaseCanopyParameters::BaseCanopyParameters(const pugi::xml_node canopy_node) : BaseCanopyParameters(){
    readParametersFromXML(canopy_node);
}

void BaseCanopyParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    // ----- canopy origin ------ //
    vec3 new_canopy_origin = XMLloadvec3(canopy_node, "canopy_origin");
    if (new_canopy_origin.x != nullvalue_f && new_canopy_origin.y != nullvalue_f) {
        canopy_origin = new_canopy_origin;
    }

    // ----- canopy rotation ------ //
    float new_canopy_rotation = XMLloadfloat(canopy_node, "canopy_rotation");
    if (new_canopy_rotation != nullvalue_f) {
        canopy_rotation = new_canopy_rotation;
    }
}

HomogeneousCanopyParameters::HomogeneousCanopyParameters() : BaseCanopyParameters(){

    leaf_size = make_vec2(0.1,0.1);

    leaf_subdivisions = make_int2(1,1);

    leaf_color = RGB::green;

    leaf_angle_distribution = "spherical";

    leaf_area_index = 1.f;

    canopy_height = 1.f;

    canopy_extent = make_vec2(5,5);

    buffer = "z";

}

HomogeneousCanopyParameters::HomogeneousCanopyParameters(const pugi::xml_node canopy_node) : HomogeneousCanopyParameters(){
    readParametersFromXML(canopy_node);
}

void HomogeneousCanopyParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    // ----- leaf size ------//
    vec2 new_leaf_size = XMLloadvec2(canopy_node, "leaf_size");
    if (new_leaf_size.x != nullvalue_f && new_leaf_size.y != nullvalue_f) {
        leaf_size = new_leaf_size;
    }

    // ----- leaf subdivisions ------//
    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    // ----- leaf color ------//
    RGBAcolor new_leaf_color = XMLloadrgba(canopy_node, "leaf_color");
    if (new_leaf_color.a != 0) {
        leaf_color = make_RGBcolor(new_leaf_color.r, new_leaf_color.g, new_leaf_color.b);
    }

    // ----- leaf texture file ------//
    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if (new_leaf_texture_file != nullvalue_s) {
        leaf_texture_file = new_leaf_texture_file;
    }

    // ----- leaf area index ------//
    float LAI = XMLloadfloat(canopy_node, "leaf_area_index");
    if (LAI != nullvalue_f) {
        leaf_area_index = LAI;
    }

    // ----- canopy height ------//
    float h = XMLloadfloat(canopy_node, "canopy_height");
    if (h != nullvalue_f) {
        canopy_height = h;
    }

    // ----- canopy extent ------//
    vec2 new_canopy_extent = XMLloadvec2(canopy_node, "canopy_extent");
    if (new_canopy_extent.x != nullvalue_f && new_canopy_extent.y != nullvalue_f) {
        canopy_extent = new_canopy_extent;
    }

    // ----- buffer ------//
    std::string new_buffer = XMLloadstring(canopy_node, "buffer");
    if ( new_buffer != nullvalue_s ) {
        buffer = new_buffer;
    }
}

void HomogeneousCanopyParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    std::cout << "HomogeneousCanopyParameters::buildPlant: Cannot build a single plant of canopy type HomogeneousCanopyParameters" << std::endl;
}

void HomogeneousCanopyParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

SphericalCrownsCanopyParameters::SphericalCrownsCanopyParameters() : BaseCanopyParameters(){

    leaf_size = make_vec2(0.025,0.025);

    leaf_subdivisions = make_int2(1,1);

    leaf_color = RGB::green;

    leaf_angle_distribution = "spherical";

    leaf_area_density = 1.f;

    crown_radius = make_vec3(0.5f,0.5f,0.5f);

    canopy_configuration = "uniform";

    plant_spacing = make_vec2(2.f,2.f);

    plant_count = make_int2(5,5);

}

SphericalCrownsCanopyParameters::SphericalCrownsCanopyParameters(const pugi::xml_node canopy_node) : SphericalCrownsCanopyParameters(){
    readParametersFromXML(canopy_node);
}

void SphericalCrownsCanopyParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    // ----- leaf size ------//
    vec2 new_leaf_size = XMLloadvec2(canopy_node, "leaf_size");
    if (new_leaf_size.x != nullvalue_f && new_leaf_size.y != nullvalue_f) {
        leaf_size = new_leaf_size;
    }

    // ----- leaf subdivisions ------//
    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    // ----- leaf color ------//
    RGBAcolor new_leaf_color = XMLloadrgba(canopy_node, "leaf_color");
    if (new_leaf_color.a != 0) {
        leaf_color = make_RGBcolor(new_leaf_color.r, new_leaf_color.g, new_leaf_color.b);
    }

    // ----- leaf texture file ------//
    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if (new_leaf_texture_file != nullvalue_s) {
        leaf_texture_file = new_leaf_texture_file;
    }

    // ----- leaf angle distribution ------//
    std::string new_leaf_angle_distribution = XMLloadstring(canopy_node, "leaf_angle_distribution");
    if (new_leaf_angle_distribution != nullvalue_s) {
        leaf_angle_distribution = new_leaf_angle_distribution;
    }

    // ----- leaf area density ------//
    float new_leaf_area_density = XMLloadfloat(canopy_node, "leaf_area_density");
    if (new_leaf_area_density != nullvalue_f) {
        leaf_area_density = new_leaf_area_density;
    }

    // ----- crown radius ------//
    vec3 new_crown_radius = XMLloadvec3(canopy_node, "crown_radius");
    if (new_crown_radius.x != nullvalue_f && new_crown_radius.y != nullvalue_f) {
        crown_radius = new_crown_radius;
    }

    // ----- canopy configuration ------//
    std::string new_canopy_configuration = XMLloadstring(canopy_node, "canopy_configuration");
    if (new_canopy_configuration != nullvalue_s) {
        canopy_configuration = new_canopy_configuration;
    }

    // ----- plant spacing ------//
    vec2 new_plant_spacing = XMLloadvec2(canopy_node, "plant_spacing");
    if (new_plant_spacing.x != nullvalue_f && new_plant_spacing.y != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    // ----- plant count ------//
    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }
}

void SphericalCrownsCanopyParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    std::cout << "SphericalCrownsCanopyParameters::buildPlant: Cannot build a single plant of canopy type SphericalCrownsCanopyParameters" << std::endl;
}

void SphericalCrownsCanopyParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

ConicalCrownsCanopyParameters::ConicalCrownsCanopyParameters() : BaseCanopyParameters(){

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

}

ConicalCrownsCanopyParameters::ConicalCrownsCanopyParameters(const pugi::xml_node canopy_node) : ConicalCrownsCanopyParameters(){
    readParametersFromXML(canopy_node);
}

void ConicalCrownsCanopyParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    // ----- leaf size ------//
    vec2 new_leaf_size = XMLloadvec2(canopy_node, "leaf_size");
    if (new_leaf_size.x != nullvalue_f && new_leaf_size.y != nullvalue_f) {
        leaf_size = new_leaf_size;
    }

    // ----- leaf subdivisions ------//
    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    // ----- leaf color ------//
    RGBAcolor new_leaf_color = XMLloadrgba(canopy_node, "leaf_color");
    if (new_leaf_color.a != 0) {
        leaf_color = make_RGBcolor(new_leaf_color.r, new_leaf_color.g, new_leaf_color.b);
    }

    // ----- leaf texture file ------//
    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if (new_leaf_texture_file != nullvalue_s) {
        leaf_texture_file = new_leaf_texture_file;
    }

    // ----- leaf angle distribution ------//
    std::string new_leaf_angle_distribution = XMLloadstring(canopy_node, "leaf_angle_distribution");
    if (new_leaf_angle_distribution != nullvalue_s) {
        leaf_angle_distribution = new_leaf_angle_distribution;
    }

    // ----- leaf area density ------//
    float new_leaf_area_density = XMLloadfloat(canopy_node, "leaf_area_density");
    if (new_leaf_area_density != nullvalue_f) {
        leaf_area_density = new_leaf_area_density;
    }

    // ----- crown radius ------//
    float new_crown_radius = XMLloadfloat(canopy_node, "crown_radius");
    if (new_crown_radius != nullvalue_f) {
        crown_radius = new_crown_radius;
    }

    // ----- crown height ------//
    float new_crown_height = XMLloadfloat(canopy_node, "crown_height");
    if (new_crown_height != nullvalue_f) {
        crown_height = new_crown_height;
    }

    // ----- canopy configuration ------//
    std::string new_canopy_configuration = XMLloadstring(canopy_node, "canopy_configuration");
    if (new_canopy_configuration != nullvalue_s) {
        canopy_configuration = new_canopy_configuration;
    }

    // ----- plant spacing ------//
    vec2 new_plant_spacing = XMLloadvec2(canopy_node, "plant_spacing");
    if (new_plant_spacing.x != nullvalue_f && new_plant_spacing.y != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    // ----- plant count ------//
    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }
}

void ConicalCrownsCanopyParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    std::cout << "ConicalCrownsCanopyParameters::buildPlant: Cannot build a single plant of canopy type ConicalCrownsCanopyParameters" << std::endl;
}

void ConicalCrownsCanopyParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

BaseGrapeVineParameters::BaseGrapeVineParameters() : BaseCanopyParameters(){

    leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

    wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

    leaf_width = 0.18;
    leaf_width_spread = 0;

    leaf_subdivisions = make_int2(1,1);

    leaf_spacing_fraction_spread = 0;

    wood_subdivisions = 10;
    wood_subdivisions_spread = 0;

    trunk_height_spread = 0;

    cordon_height_spread = 0;
    cordon_radius_spread = 0;

    shoot_length_spread = 0;
    shoot_radius_spread = 0;
    shoots_per_cordon_spread = 0;

    grape_color = make_RGBcolor(0.18,0.2,0.25);

    grape_radius_spread = 0;

    cluster_radius_spread = 0;
    cluster_height_max_spread = 0;

    grape_subdivisions_spread = 0;

    plant_count = make_int2(3,3);

    canopy_rotation_spread = 0;

    plant_spacing_spread = 0;

    row_spacing_spread = 0;

    dead_probability = 0;
    missing_plant_probability = 0;

}

BaseGrapeVineParameters::BaseGrapeVineParameters(const pugi::xml_node canopy_node) : BaseGrapeVineParameters(){
    readParametersFromXML(canopy_node);
}

void BaseGrapeVineParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    float new_leaf_width = XMLloadfloat(canopy_node, "leaf_width");
    if (new_leaf_width != nullvalue_f) {
        leaf_width = new_leaf_width;
    }

    float new_leaf_width_spread = XMLloadfloat(canopy_node, "leaf_width_spread");
    if (new_leaf_width_spread != nullvalue_f) {
        leaf_width_spread = new_leaf_width_spread;
    }

    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if ( new_leaf_texture_file != nullvalue_s ) {
        leaf_texture_file = new_leaf_texture_file;
    }

    std::string new_wood_texture_file = XMLloadstring(canopy_node, "wood_texture_file");
    if ( new_wood_texture_file != nullvalue_s ) {
        wood_texture_file = new_wood_texture_file;
    }

    int new_wood_subdivisions = XMLloadint(canopy_node, "wood_subdivisions");
    if ( new_wood_subdivisions != nullvalue_i) {
        wood_subdivisions = new_wood_subdivisions;
    }

    int new_wood_subdivisions_spread = XMLloadint(canopy_node, "wood_subdivisions_spread");
    if ( new_wood_subdivisions_spread != nullvalue_i) {
        wood_subdivisions_spread = new_wood_subdivisions_spread;
    }

    float h = XMLloadfloat(canopy_node, "trunk_height");
    if (h != nullvalue_f) {
        trunk_height = h;
    }

    float new_trunk_height_spread = XMLloadfloat(canopy_node, "trunk_height_spread");
    if (new_trunk_height_spread != nullvalue_f) {
        trunk_height_spread = new_trunk_height_spread;
    }

    float r = XMLloadfloat(canopy_node, "trunk_radius");
    if (r != nullvalue_f) {
        trunk_radius = r;
    }

    float new_trunk_radius_spread = XMLloadfloat(canopy_node, "trunk_radius_spread");
    if (new_trunk_radius_spread != nullvalue_f) {
        trunk_radius_spread = new_trunk_radius_spread;
    }

    float ch = XMLloadfloat(canopy_node, "cordon_height");
    if (ch != nullvalue_f) {
        cordon_height = ch;
    }

    float new_cordon_height_spread = XMLloadfloat(canopy_node, "cordon_height_spread");
    if (new_cordon_height_spread != nullvalue_f) {
        cordon_height_spread = new_cordon_height_spread;
    }

    float cr = XMLloadfloat(canopy_node, "cordon_radius");
    if (cr != nullvalue_f) {
        cordon_radius = cr;
    }

    float new_cordon_radius_spread = XMLloadfloat(canopy_node, "cordon_radius_spread");
    if (new_cordon_radius_spread != nullvalue_f) {
        cordon_radius_spread = new_cordon_radius_spread;
    }

    float sl = XMLloadfloat(canopy_node, "shoot_length");
    if (sl != nullvalue_f) {
        shoot_length = sl;
    }

    float new_shoot_length_spread = XMLloadfloat(canopy_node, "shoot_length_spread");
    if (new_shoot_length_spread != nullvalue_f) {
        shoot_length_spread = new_shoot_length_spread;
    }

    float sr = XMLloadfloat(canopy_node, "shoot_radius");
    if (sr != nullvalue_f) {
        shoot_radius = sr;
    }

    float new_shoot_radius_spread = XMLloadfloat(canopy_node, "shoot_radius_spread");
    if (new_shoot_radius_spread != nullvalue_f) {
        shoot_radius_spread = new_shoot_radius_spread;
    }

    int spc = XMLloadint(canopy_node, "shoots_per_cordon");
    if (spc != nullvalue_i) {
        shoots_per_cordon = uint(spc);
    }

    int new_shoots_per_cordon_spread = XMLloadint(canopy_node, "shoots_per_cordon_spread");
    if (new_shoots_per_cordon_spread != nullvalue_i) {
        shoots_per_cordon_spread = uint(new_shoots_per_cordon_spread);
    }

    float lsf = XMLloadfloat(canopy_node, "leaf_spacing_fraction");
    if (lsf != nullvalue_f) {
        leaf_spacing_fraction = lsf;
    }

    float new_leaf_spacing_fraction_spread = XMLloadfloat(canopy_node, "leaf_spacing_fraction_spread");
    if (new_leaf_spacing_fraction_spread != nullvalue_f) {
        leaf_spacing_fraction_spread = new_leaf_spacing_fraction_spread;
    }

    float gr = XMLloadfloat(canopy_node, "grape_radius");
    if (gr != nullvalue_f) {
        grape_radius = gr;
    }

    float new_grape_radius_spread = XMLloadfloat(canopy_node, "grape_radius_spread");
    if (new_grape_radius_spread != nullvalue_f) {
        grape_radius_spread = new_grape_radius_spread;
    }

    float clr = XMLloadfloat(canopy_node, "cluster_radius");
    if (clr != nullvalue_f) {
        cluster_radius = clr;
    }

    float new_cluster_radius_spread = XMLloadfloat(canopy_node, "cluster_radius_spread");
    if (new_cluster_radius_spread != nullvalue_f) {
        cluster_radius_spread = new_cluster_radius_spread;
    }

    float clhm = XMLloadfloat(canopy_node, "cluster_height_max");
    if (clhm != nullvalue_f) {
        cluster_height_max = clhm;
    }

    float new_cluster_height_max_spread = XMLloadfloat(canopy_node, "cluster_height_max_spread");
    if (new_cluster_height_max_spread != nullvalue_f) {
        cluster_height_max_spread = new_cluster_height_max_spread;
    }

    RGBAcolor new_grape_color = XMLloadrgba(canopy_node, "grape_color");
    if ( new_grape_color.a != 0 ) {
        grape_color = make_RGBcolor(new_grape_color.r, new_grape_color.g, new_grape_color.b);
    }

    int new_grape_subdivisions = XMLloadint(canopy_node, "grape_subdivisions");
    if (new_grape_subdivisions != nullvalue_i) {
        grape_subdivisions = uint(new_grape_subdivisions);
    }

    int new_grape_subdivisions_spread = XMLloadint(canopy_node, "grape_subdivisions_spread");
    if (new_grape_subdivisions_spread != nullvalue_i) {
        grape_subdivisions_spread = uint(new_grape_subdivisions_spread);
    }

    float new_plant_spacing = XMLloadfloat(canopy_node, "plant_spacing");
    if (new_plant_spacing != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    float new_plant_spacing_spread = XMLloadfloat(canopy_node, "plant_spacing_spread");
    if (new_plant_spacing_spread != nullvalue_f) {
        plant_spacing_spread = new_plant_spacing_spread;
    }

    float new_row_spacing = XMLloadfloat(canopy_node, "row_spacing");
    if (new_row_spacing != nullvalue_f) {
        row_spacing = new_row_spacing;
    }

    float new_row_spacing_spread = XMLloadfloat(canopy_node, "row_spacing_spread");
    if (new_row_spacing_spread != nullvalue_f) {
        row_spacing_spread = new_row_spacing_spread;
    }

    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }

    float new_dead_probability = XMLloadfloat(canopy_node, "dead_probability");
    if (new_dead_probability != nullvalue_f) {
        if (new_dead_probability < 0 || new_dead_probability > 1)
            std::cout << "BaseGrapeVineParameters::readParametersFromXML: dead_probability value must be between 0 and 1" << std::endl;
        else
            dead_probability = new_dead_probability;
    }

    float new_missing_plant_probability = XMLloadfloat(canopy_node, "missing_plant_probability");
    if (new_missing_plant_probability != nullvalue_f) {
        if (new_missing_plant_probability < 0 || new_missing_plant_probability > 1)
            std::cout << "BaseGrapeVineParameters::readParametersFromXML: missing_plant_probability value must be between 0 and 1" << std::endl;
        else
            missing_plant_probability = new_missing_plant_probability;
    }

    float new_canopy_rotation_spread = XMLloadfloat(canopy_node, "canopy_rotation_spread");
    if (new_canopy_rotation_spread != nullvalue_f) {
        canopy_rotation_spread = new_canopy_rotation_spread;
    }
}

void BaseGrapeVineParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    std::cout << "BaseGrapeVineParameters::buildPlant: Cannot build a single plant of canopy type BaseGrapeVineParameters" << std::endl;
}

void BaseGrapeVineParameters::buildCanopy(CanopyGenerator& canopy_generator){
    std::cout << "BaseGrapeVineParameters::buildPlant: Cannot build a canopy of type BaseGrapeVineParameters" << std::endl;
}

VSPGrapevineParameters::VSPGrapevineParameters() : BaseGrapeVineParameters(){

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

    grape_subdivisions = 8;

    plant_spacing = 2;

    row_spacing = 2;

}

VSPGrapevineParameters::VSPGrapevineParameters(const pugi::xml_node canopy_node) : VSPGrapevineParameters(){
    readParametersFromXML(canopy_node);
}

void VSPGrapevineParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseGrapeVineParameters::readParametersFromXML(canopy_node);
}

void VSPGrapevineParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.grapevineVSP(*this, origin);
}

void VSPGrapevineParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

SplitGrapevineParameters::SplitGrapevineParameters() : BaseGrapeVineParameters(){

    trunk_height = 1.3;

    trunk_radius = 0.05;

    cordon_height = 1.5;

    cordon_radius = 0.02;

    cordon_spacing = 1.f;
    cordon_spacing_spread = 0;

    shoot_length = 1.2;

    shoot_radius = 0.0025;

    shoots_per_cordon = 10;

    shoot_angle_tip = 0.4*M_PI;
    shoot_angle_tip_spread = 0;

    shoot_angle_base = 0.;
    shoot_angle_base_spread = 0;

    leaf_spacing_fraction = 0.6;

    grape_radius = 0.0075;

    cluster_radius = 0.03;

    cluster_height_max = 0.1;

    grape_subdivisions = 8;

    plant_spacing = 2;

    row_spacing = 4;

}

SplitGrapevineParameters::SplitGrapevineParameters(const pugi::xml_node canopy_node) : SplitGrapevineParameters(){
    readParametersFromXML(canopy_node);
}

void SplitGrapevineParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseGrapeVineParameters::readParametersFromXML(canopy_node);

    float cs = XMLloadfloat(canopy_node, "cordon_spacing");
    if (cs != nullvalue_f) {
        cordon_spacing = cs;
    }

    float sat = XMLloadfloat( canopy_node, "shoot_angle_tip" );
    if( sat != nullvalue_f ){
        shoot_angle_tip = sat;
    }

    float sab = XMLloadfloat( canopy_node, "shoot_angle_base" );
    if( sab != nullvalue_f ){
        shoot_angle_base = sab;
    }
}

void SplitGrapevineParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.grapevineSplit(*this, origin);
}

void SplitGrapevineParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

UnilateralGrapevineParameters::UnilateralGrapevineParameters() : BaseGrapeVineParameters(){

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

    grape_subdivisions = 8;

    plant_spacing = 1.5;

    row_spacing = 2;

}

UnilateralGrapevineParameters::UnilateralGrapevineParameters(const pugi::xml_node canopy_node) : UnilateralGrapevineParameters(){
    readParametersFromXML(canopy_node);
}

void UnilateralGrapevineParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseGrapeVineParameters::readParametersFromXML(canopy_node);
}

void UnilateralGrapevineParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.grapevineUnilateral(*this, origin);
}

void UnilateralGrapevineParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

GobletGrapevineParameters::GobletGrapevineParameters() : BaseGrapeVineParameters(){

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

    grape_subdivisions = 8;

    plant_spacing = 2;

    row_spacing = 2;

}

GobletGrapevineParameters::GobletGrapevineParameters(const pugi::xml_node canopy_node) : GobletGrapevineParameters(){
    readParametersFromXML(canopy_node);
}

void GobletGrapevineParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseGrapeVineParameters::readParametersFromXML(canopy_node);
}

void GobletGrapevineParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.grapevineGoblet(*this, origin);
}

void GobletGrapevineParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

WhiteSpruceCanopyParameters::WhiteSpruceCanopyParameters() : BaseCanopyParameters(){

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

}

WhiteSpruceCanopyParameters::WhiteSpruceCanopyParameters(const pugi::xml_node canopy_node) : WhiteSpruceCanopyParameters(){
    readParametersFromXML(canopy_node);
}

void WhiteSpruceCanopyParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    float new_needle_width = XMLloadfloat(canopy_node, "needle_width");
    if (new_needle_width != nullvalue_f) {
        needle_width = new_needle_width;
    }

    float new_needle_length = XMLloadfloat(canopy_node, "needle_length");
    if (new_needle_length != nullvalue_f) {
        needle_length = new_needle_length;
    }

    int2 new_needle_subdivisions = XMLloadint2(canopy_node, "needle_subdivisions");
    if (new_needle_subdivisions.x != nullvalue_i && new_needle_subdivisions.y != nullvalue_i) {
        needle_subdivisions = new_needle_subdivisions;
    }

    RGBAcolor new_needle_color = XMLloadrgba(canopy_node, "needle_color");
    if (new_needle_color.a != 0) {
        needle_color = make_RGBcolor(new_needle_color.r, new_needle_color.g, new_needle_color.b);
    }

    std::string new_wood_texture_file = XMLloadstring(canopy_node, "wood_texture_file");
    if ( new_wood_texture_file != nullvalue_s ) {
        wood_texture_file = new_wood_texture_file;
    }

    int new_wood_subdivisions = XMLloadint(canopy_node, "wood_subdivisions");
    if (new_wood_subdivisions != nullvalue_i) {
        wood_subdivisions = new_wood_subdivisions;
    }

    float new_trunk_height = XMLloadfloat(canopy_node, "trunk_height");
    if (new_trunk_height != nullvalue_f) {
        trunk_height = new_trunk_height;
    }

    float new_trunk_radius = XMLloadfloat(canopy_node, "trunk_radius");
    if (new_trunk_radius != nullvalue_f) {
        trunk_radius = new_trunk_radius;
    }

    float new_crown_radius = XMLloadfloat(canopy_node, "crown_radius");
    if (new_crown_radius != nullvalue_f) {
        crown_radius = new_crown_radius;
    }

    float new_shoot_radius = XMLloadfloat(canopy_node, "shoot_radius");
    if (new_shoot_radius != nullvalue_f) {
        shoot_radius = new_shoot_radius;
    }

    float new_level_spacing = XMLloadfloat(canopy_node, "level_spacing");
    if (new_level_spacing != nullvalue_f) {
        level_spacing = new_level_spacing;
    }

    int new_branches_per_level = XMLloadint(canopy_node, "branches_per_level");
    if (new_branches_per_level != nullvalue_i) {
        branches_per_level = new_branches_per_level;
    }

    float new_shoot_angle = XMLloadfloat(canopy_node, "shoot_angle");
    if (new_shoot_angle != nullvalue_f) {
        shoot_angle = new_shoot_angle;
    }

    std::string new_canopy_configuration = XMLloadstring(canopy_node, "canopy_configuration");
    if (new_canopy_configuration != nullvalue_s) {
        canopy_configuration = new_canopy_configuration;
    }

    vec2 new_plant_spacing = XMLloadvec2(canopy_node, "plant_spacing");
    if (new_plant_spacing.x != nullvalue_f && new_plant_spacing.y != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }
}

void WhiteSpruceCanopyParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.whitespruce(*this, origin);
}

void WhiteSpruceCanopyParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

TomatoParameters::TomatoParameters() : BaseCanopyParameters(){

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

}

TomatoParameters::TomatoParameters(const pugi::xml_node canopy_node) : TomatoParameters(){
    readParametersFromXML(canopy_node);
}

void TomatoParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    float new_leaf_length = XMLloadfloat(canopy_node, "leaf_length");
    if (new_leaf_length != nullvalue_f) {
        leaf_length = new_leaf_length;
    }

    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if ( new_leaf_texture_file != nullvalue_s ) {
        leaf_texture_file = new_leaf_texture_file;
    }

    RGBAcolor new_shoot_color = XMLloadrgba(canopy_node, "shoot_color");
    if (new_shoot_color.a != 0 ) {
        shoot_color = make_RGBcolor(new_shoot_color.r, new_shoot_color.g, new_shoot_color.b);
    }

    int new_shoot_subdivisions = XMLloadint(canopy_node, "shoot_subdivisions");
    if (new_shoot_subdivisions != nullvalue_i) {
        shoot_subdivisions = new_shoot_subdivisions;
    }

    float h = XMLloadfloat(canopy_node, "plant_height");
    if (h != nullvalue_f) {
        plant_height = h;
    }

    float gr = XMLloadfloat(canopy_node, "fruit_radius");
    if (gr != nullvalue_f) {
        fruit_radius = gr;
    }

    RGBAcolor new_fruit_color = XMLloadrgba(canopy_node, "fruit_color");
    if (new_fruit_color.a != 0 ) {
        fruit_color = make_RGBcolor(new_fruit_color.r, new_fruit_color.g, new_fruit_color.b);
    }


    int new_fruit_subdivisions = XMLloadint(canopy_node, "fruit_subdivisions");
    if (new_fruit_subdivisions != nullvalue_i) {
        fruit_subdivisions = uint(new_fruit_subdivisions);
    }

    float new_plant_spacing = XMLloadfloat(canopy_node, "plant_spacing");
    if (new_plant_spacing != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    float new_row_spacing = XMLloadfloat(canopy_node, "row_spacing");
    if (new_row_spacing != nullvalue_f) {
        row_spacing = new_row_spacing;
    }

    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }
}

void TomatoParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.tomato(*this, origin);
}

void TomatoParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

StrawberryParameters::StrawberryParameters() : BaseCanopyParameters(){

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

}

StrawberryParameters::StrawberryParameters(const pugi::xml_node canopy_node) : StrawberryParameters(){
    readParametersFromXML(canopy_node);
}

void StrawberryParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    float new_leaf_length = XMLloadfloat(canopy_node, "leaf_length");
    if (new_leaf_length != nullvalue_f) {
        leaf_length = new_leaf_length;
    }

    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if ( new_leaf_texture_file != nullvalue_s ) {
        leaf_texture_file = new_leaf_texture_file;
    }

    int new_stem_subdivisions = XMLloadint(canopy_node, "stem_subdivisions");
    if (new_stem_subdivisions != nullvalue_i) {
        stem_subdivisions = new_stem_subdivisions;
    }

    float new_stem_radius = XMLloadfloat(canopy_node, "stem_radius");
    if (new_stem_radius != nullvalue_f) {
        stem_radius = new_stem_radius;
    }

    float h = XMLloadfloat(canopy_node, "plant_height");
    if (h != nullvalue_f) {
        plant_height = h;
    }

    int r = XMLloadint(canopy_node, "stems_per_plant");
    if (r != nullvalue_i) {
        stems_per_plant = r;
    }

    float gr = XMLloadfloat(canopy_node, "fruit_radius");
    if (gr != nullvalue_f) {
        fruit_radius = gr;
    }

    float clr = XMLloadfloat(canopy_node, "clusters_per_stem");
    if (clr != nullvalue_f) {
        clusters_per_stem = clr;
    }

    int new_fruit_subdivisions = XMLloadint(canopy_node, "fruit_subdivisions");
    if (new_fruit_subdivisions != nullvalue_i) {
        fruit_subdivisions = uint(new_fruit_subdivisions);
    }

    std::string new_fruit_texture_file = XMLloadstring(canopy_node, "fruit_texture_file");
    if ( new_fruit_texture_file != nullvalue_s ) {
        fruit_texture_file = new_fruit_texture_file;
    }

    float new_plant_spacing = XMLloadfloat(canopy_node, "plant_spacing");
    if (new_plant_spacing != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    float new_row_spacing = XMLloadfloat(canopy_node, "row_spacing");
    if (new_row_spacing != nullvalue_f) {
        row_spacing = new_row_spacing;
    }

    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = plant_count;
    }
}

void StrawberryParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.strawberry(*this, origin);
}

void StrawberryParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

WalnutCanopyParameters::WalnutCanopyParameters() : BaseCanopyParameters(){

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

}

WalnutCanopyParameters::WalnutCanopyParameters(const pugi::xml_node canopy_node) : WalnutCanopyParameters(){
    readParametersFromXML(canopy_node);
}

void WalnutCanopyParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    float new_leaf_length = XMLloadfloat(canopy_node, "leaf_length");
    if (new_leaf_length != nullvalue_f) {
        leaf_length = new_leaf_length;
    }

    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if ( new_leaf_texture_file != nullvalue_s ) {
        leaf_texture_file = new_leaf_texture_file;
    }

    int new_wood_subdivisions = XMLloadint(canopy_node, "wood_subdivisions");
    if (new_wood_subdivisions != nullvalue_i) {
        wood_subdivisions = new_wood_subdivisions;
    }


    float new_trunk_radius = XMLloadfloat(canopy_node, "trunk_radius");
    if (new_trunk_radius != nullvalue_f) {
        trunk_radius = new_trunk_radius;
    }

    float new_trunk_height = XMLloadfloat(canopy_node, "trunk_height");
    if (new_trunk_height != nullvalue_f) {
        trunk_height = new_trunk_height;
    }

    vec3 new_branch_length = XMLloadvec3(canopy_node, "branch_length");
    if (new_branch_length.x != nullvalue_f && new_branch_length.y != nullvalue_f) {
        branch_length = new_branch_length;
    }

    std::string new_fruit_texture_file = XMLloadstring(canopy_node, "fruit_texture_file");
    if ( new_fruit_texture_file != nullvalue_s ) {
        fruit_texture_file = new_fruit_texture_file;
    }

    int new_fruit_subdivisions = XMLloadint(canopy_node, "fruit_subdivisions");
    if (new_fruit_subdivisions != nullvalue_i) {
        fruit_subdivisions = uint(new_fruit_subdivisions);
    }

    float new_plant_spacing = XMLloadfloat(canopy_node, "plant_spacing");
    if (new_plant_spacing != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    float new_row_spacing = XMLloadfloat(canopy_node, "row_spacing");
    if (new_row_spacing != nullvalue_f) {
        row_spacing = new_row_spacing;
    }

    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }
}

void WalnutCanopyParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.walnut(*this, origin);
}

void WalnutCanopyParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

SorghumCanopyParameters::SorghumCanopyParameters() : BaseCanopyParameters(){
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

}

SorghumCanopyParameters::SorghumCanopyParameters(const pugi::xml_node canopy_node) : SorghumCanopyParameters(){
    readParametersFromXML(canopy_node);
}

void SorghumCanopyParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    int new_sorghum_stage = XMLloadint(canopy_node, "sorghum_stage");
    if (new_sorghum_stage != nullvalue_i) {
        sorghum_stage = new_sorghum_stage;
    }

    // STAGE 1
    float new_s1_stem_length = XMLloadint(canopy_node, "s1_stem_length");
    if (new_s1_stem_length != nullvalue_i) {
        s1_stem_length = new_s1_stem_length;
    }

    float new_s1_stem_radius = XMLloadfloat(canopy_node, "s1_stem_radius");
    if (new_s1_stem_radius != nullvalue_f) {
        s1_stem_radius = new_s1_stem_radius;
    }

    int new_s1_stem_subdivisions = XMLloadint(canopy_node, "s1_stem_subdivisions");
    if (new_s1_stem_subdivisions != nullvalue_f) {
        s1_stem_subdivisions = uint(new_s1_stem_subdivisions);
    }

    vec2 new_s1_leaf_size1 = XMLloadvec2(canopy_node, "s1_leaf_size1");
    if (new_s1_leaf_size1.x != nullvalue_f && new_s1_leaf_size1.y != nullvalue_f) {
        s1_leaf_size1 = new_s1_leaf_size1;
    }

    vec2 new_s1_leaf_size2 = XMLloadvec2(canopy_node, "s1_leaf_size2");
    if (new_s1_leaf_size2.x != nullvalue_f && new_s1_leaf_size2.y != nullvalue_f) {
        s1_leaf_size2 = new_s1_leaf_size2;
    }

    vec2 new_s1_leaf_size3 = XMLloadvec2(canopy_node, "s1_leaf_size3");
    if (new_s1_leaf_size3.x != nullvalue_f && new_s1_leaf_size3.y != nullvalue_f) {
        s1_leaf_size3 = new_s1_leaf_size3;
    }

    float new_s1_leaf1_angle = XMLloadfloat(canopy_node, "s1_leaf1_angle");
    if (new_s1_leaf1_angle != nullvalue_f) {
        s1_leaf1_angle = new_s1_leaf1_angle;
    }

    float new_s1_leaf2_angle = XMLloadfloat(canopy_node, "s1_leaf2_angle");
    if (new_s1_leaf2_angle != nullvalue_f) {
        s1_leaf2_angle = new_s1_leaf2_angle;
    }

    float new_s1_leaf3_angle = XMLloadfloat(canopy_node, "s1_leaf3_angle");
    if (new_s1_leaf3_angle != nullvalue_f) {
        s1_leaf3_angle = new_s1_leaf3_angle;
    }

    int2 new_s1_leaf_subdivisions = XMLloadint2(canopy_node, "s1_leaf_subdivisions");
    if (new_s1_leaf_subdivisions.x != nullvalue_i && new_s1_leaf_subdivisions.y != nullvalue_i) {
        s1_leaf_subdivisions = new_s1_leaf_subdivisions;
    }

    std::string new_s1_leaf_texture_file = XMLloadstring(canopy_node, "s1_leaf_texture_file");
    if (new_s1_leaf_texture_file.compare(nullvalue_s) != 0) {
        s1_leaf_texture_file = new_s1_leaf_texture_file;
    }

    // STAGE 2
    float new_s2_stem_length = XMLloadint(canopy_node, "s2_stem_length");
    if (new_s2_stem_length != nullvalue_i) {
        s2_stem_length = new_s2_stem_length;
    }

    float new_s2_stem_radius = XMLloadfloat(canopy_node, "s2_stem_radius");
    if (new_s2_stem_radius != nullvalue_f) {
        s2_stem_radius = new_s2_stem_radius;
    }

    int new_s2_stem_subdivisions = XMLloadint(canopy_node, "s2_stem_subdivisions");
    if (new_s2_stem_subdivisions != nullvalue_f) {
        s2_stem_subdivisions = uint(new_s2_stem_subdivisions);
    }

    vec2 new_s2_leaf_size1 = XMLloadvec2(canopy_node, "s2_leaf_size1");
    if (new_s2_leaf_size1.x != nullvalue_f && new_s2_leaf_size1.y != nullvalue_f) {
        s2_leaf_size1 = new_s2_leaf_size1;
    }

    vec2 new_s2_leaf_size2 = XMLloadvec2(canopy_node, "s2_leaf_size2");
    if (new_s2_leaf_size2.x != nullvalue_f && new_s2_leaf_size2.y != nullvalue_f) {
        s2_leaf_size2 = new_s2_leaf_size2;
    }

    vec2 new_s2_leaf_size3 = XMLloadvec2(canopy_node, "s2_leaf_size3");
    if (new_s2_leaf_size3.x != nullvalue_f && new_s2_leaf_size3.y != nullvalue_f) {
        s2_leaf_size3 = new_s2_leaf_size3;
    }

    vec2 new_s2_leaf_size4 = XMLloadvec2(canopy_node, "s2_leaf_size4");
    if (new_s2_leaf_size4.x != nullvalue_f && new_s2_leaf_size4.y != nullvalue_f) {
        s2_leaf_size4 = new_s2_leaf_size4;
    }

    vec2 new_s2_leaf_size5 = XMLloadvec2(canopy_node, "s2_leaf_size5");
    if (new_s2_leaf_size5.x != nullvalue_f && new_s2_leaf_size5.y != nullvalue_f) {
        s2_leaf_size5 = new_s2_leaf_size5;
    }

    float new_s2_leaf1_angle = XMLloadfloat(canopy_node, "s2_leaf1_angle");
    if (new_s2_leaf1_angle != nullvalue_f) {
        s2_leaf1_angle = new_s2_leaf1_angle;
    }

    float new_s2_leaf2_angle = XMLloadfloat(canopy_node, "s2_leaf2_angle");
    if (new_s2_leaf2_angle != nullvalue_f) {
        s2_leaf2_angle = new_s2_leaf2_angle;
    }

    float new_s2_leaf3_angle = XMLloadfloat(canopy_node, "s2_leaf3_angle");
    if (new_s2_leaf3_angle != nullvalue_f) {
        s2_leaf3_angle = new_s2_leaf3_angle;
    }

    float new_s2_leaf4_angle = XMLloadfloat(canopy_node, "s2_leaf4_angle");
    if (new_s2_leaf4_angle != nullvalue_f) {
        s2_leaf4_angle = new_s2_leaf4_angle;
    }

    float new_s2_leaf5_angle = XMLloadfloat(canopy_node, "s2_leaf5_angle");
    if (new_s2_leaf3_angle != nullvalue_f) {
        s2_leaf5_angle = new_s2_leaf5_angle;
    }

    int2 new_s2_leaf_subdivisions = XMLloadint2(canopy_node, "s2_leaf_subdivisions");
    if (new_s2_leaf_subdivisions.x != nullvalue_i && new_s2_leaf_subdivisions.y != nullvalue_i) {
        s2_leaf_subdivisions = new_s2_leaf_subdivisions;
    }

    std::string new_s2_leaf_texture_file = XMLloadstring(canopy_node, "s2_leaf_texture_file");
    if (new_s2_leaf_texture_file.compare(nullvalue_s) != 0) {
        s2_leaf_texture_file = new_s2_leaf_texture_file;
    }

    // STAGE 3
    float new_s3_stem_length = XMLloadint(canopy_node, "s3_stem_length");
    if (new_s3_stem_length != nullvalue_i) {
        s3_stem_length = new_s3_stem_length;
    }

    float new_s3_stem_radius = XMLloadfloat(canopy_node, "s3_stem_radius");
    if (new_s3_stem_radius != nullvalue_f) {
        s3_stem_radius = new_s3_stem_radius;
    }

    int new_s3_stem_subdivisions = XMLloadint(canopy_node, "s3_stem_subdivisions");
    if (new_s3_stem_subdivisions != nullvalue_f) {
        s3_stem_subdivisions = uint(new_s3_stem_subdivisions);
    }

    vec2 new_s3_leaf_size = XMLloadvec2(canopy_node, "s3_leaf_size");
    if (new_s3_leaf_size.x != nullvalue_f && new_s3_leaf_size.y != nullvalue_f) {
        s3_leaf_size = new_s3_leaf_size;
    }

    int2 new_s3_leaf_subdivisions = XMLloadint2(canopy_node, "s3_leaf_subdivisions");
    if (new_s3_leaf_subdivisions.x != nullvalue_i && new_s3_leaf_subdivisions.y != nullvalue_i) {
        s3_leaf_subdivisions = new_s3_leaf_subdivisions;
    }

    int new_s3_number_of_leaves = XMLloadint(canopy_node, "s3_number_of_leaves");
    if (new_s3_number_of_leaves != nullvalue_i) {
        s3_number_of_leaves = new_s3_number_of_leaves;
    }

    float new_s3_mean_leaf_angle = XMLloadfloat(canopy_node, "s3_mean_leaf_angle");
    if (new_s3_mean_leaf_angle != nullvalue_f) {
        s3_mean_leaf_angle = new_s3_mean_leaf_angle;
    }

    std::string new_s3_leaf_texture_file = XMLloadstring(canopy_node, "s3_leaf_texture_file");
    if (new_s3_leaf_texture_file.compare(nullvalue_s) != 0) {
        s3_leaf_texture_file = new_s3_leaf_texture_file;
    }

    // STAGE 4
    float new_s4_stem_length = XMLloadint(canopy_node, "s4_stem_length");
    if (new_s4_stem_length != nullvalue_i) {
        s4_stem_length = new_s4_stem_length;
    }

    float new_s4_stem_radius = XMLloadfloat(canopy_node, "s4_stem_radius");
    if (new_s4_stem_radius != nullvalue_f) {
        s4_stem_radius = new_s4_stem_radius;
    }

    int new_s4_stem_subdivisions = XMLloadint(canopy_node, "s4_stem_subdivisions");
    if (new_s4_stem_subdivisions != nullvalue_f) {
        s4_stem_subdivisions = uint(new_s4_stem_subdivisions);
    }

    vec2 new_s4_panicle_size = XMLloadvec2(canopy_node, "s4_panicle_size");
    if (new_s4_panicle_size.x != nullvalue_f && new_s4_panicle_size.y != nullvalue_f) {
        s4_panicle_size = new_s4_panicle_size;
    }

    int new_s4_panicle_subdivisions = XMLloadint(canopy_node, "s4_panicle_subdivisions");
    if (new_s4_panicle_subdivisions != nullvalue_f) {
        s4_panicle_subdivisions = uint(new_s4_panicle_subdivisions);
    }

    std::string new_s4_seed_texture_file = XMLloadstring(canopy_node, "s4_seed_texture_file");
    if (new_s4_seed_texture_file.compare(nullvalue_s) != 0) {
        s4_seed_texture_file = new_s4_seed_texture_file;
    }

    vec2 new_s4_leaf_size = XMLloadvec2(canopy_node, "s4_leaf_size");
    if (new_s4_leaf_size.x != nullvalue_f && new_s4_leaf_size.y != nullvalue_f) {
        s4_leaf_size = new_s4_leaf_size;
    }

    int2 new_s4_leaf_subdivisions = XMLloadint2(canopy_node, "s4_leaf_subdivisions");
    if (new_s4_leaf_subdivisions.x != nullvalue_i && new_s4_leaf_subdivisions.y != nullvalue_i) {
        s4_leaf_subdivisions = new_s4_leaf_subdivisions;
    }

    int new_s4_number_of_leaves = XMLloadint(canopy_node, "s4_number_of_leaves");
    if (new_s4_number_of_leaves != nullvalue_i) {
        s4_number_of_leaves = new_s4_number_of_leaves;
    }

    float new_s4_mean_leaf_angle = XMLloadfloat(canopy_node, "s4_mean_leaf_angle");
    if (new_s4_mean_leaf_angle != nullvalue_f) {
        s4_mean_leaf_angle = new_s4_mean_leaf_angle;
    }

    std::string new_s4_leaf_texture_file = XMLloadstring(canopy_node, "s4_leaf_texture_file");
    if (new_s4_leaf_texture_file.compare(nullvalue_s) != 0) {
        s4_leaf_texture_file = new_s4_leaf_texture_file;
    }

    // STAGE 5
    float new_s5_stem_length = XMLloadint(canopy_node, "s5_stem_length");
    if (new_s5_stem_length != nullvalue_i) {
        s5_stem_length = new_s5_stem_length;
    }

    float new_s5_stem_radius = XMLloadfloat(canopy_node, "s5_stem_radius");
    if (new_s5_stem_radius != nullvalue_f) {
        s5_stem_radius = new_s5_stem_radius;
    }

    float new_s5_stem_bend = XMLloadfloat(canopy_node, "s5_stem_bend");
    if (new_s5_stem_bend != nullvalue_f) {
        s5_stem_bend = new_s5_stem_bend;
    }

    int new_s5_stem_subdivisions = XMLloadint(canopy_node, "s5_stem_subdivisions");
    if (new_s5_stem_subdivisions != nullvalue_f) {
        s5_stem_subdivisions = uint(new_s5_stem_subdivisions);
    }

    vec2 new_s5_panicle_size = XMLloadvec2(canopy_node, "s5_panicle_size");
    if (new_s5_panicle_size.x != nullvalue_f && new_s5_panicle_size.y != nullvalue_f) {
        s5_panicle_size = new_s5_panicle_size;
    }

    int new_s5_panicle_subdivisions = XMLloadint(canopy_node, "s5_panicle_subdivisions");
    if (new_s5_panicle_subdivisions != nullvalue_f) {
        s5_panicle_subdivisions = uint(new_s5_panicle_subdivisions);
    }

    std::string new_s5_seed_texture_file = XMLloadstring(canopy_node, "s5_seed_texture_file");
    if (new_s5_seed_texture_file.compare(nullvalue_s) != 0) {
        s5_seed_texture_file = new_s5_seed_texture_file;
    }

    vec2 new_s5_leaf_size = XMLloadvec2(canopy_node, "s5_leaf_size");
    if (new_s5_leaf_size.x != nullvalue_f && new_s5_leaf_size.y != nullvalue_f) {
        s5_leaf_size = new_s5_leaf_size;
    }

    int2 new_s5_leaf_subdivisions = XMLloadint2(canopy_node, "s5_leaf_subdivisions");
    if (new_s5_leaf_subdivisions.x != nullvalue_i && new_s5_leaf_subdivisions.y != nullvalue_i) {
        s5_leaf_subdivisions = new_s5_leaf_subdivisions;
    }

    int new_s5_number_of_leaves = XMLloadint(canopy_node, "s5_number_of_leaves");
    if (new_s5_number_of_leaves != nullvalue_i) {
        s5_number_of_leaves = new_s5_number_of_leaves;
    }

    float new_s5_mean_leaf_angle = XMLloadfloat(canopy_node, "s5_mean_leaf_angle");
    if (new_s5_mean_leaf_angle != nullvalue_f) {
        s5_mean_leaf_angle = new_s5_mean_leaf_angle;
    }

    std::string new_s5_leaf_texture_file = XMLloadstring(canopy_node, "s5_leaf_texture_file");
    if (new_s5_leaf_texture_file.compare(nullvalue_s) != 0) {
        s5_leaf_texture_file = new_s5_leaf_texture_file;
    }

    float new_plant_spacing = XMLloadfloat(canopy_node, "plant_spacing");
    if (new_plant_spacing != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    float new_row_spacing = XMLloadfloat(canopy_node, "row_spacing");
    if (new_row_spacing != nullvalue_f) {
        row_spacing = new_row_spacing;
    }

    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }
}

void SorghumCanopyParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.sorghum(*this, origin);
}

void SorghumCanopyParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
}

BeanParameters::BeanParameters() : BaseCanopyParameters() {

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

}

BeanParameters::BeanParameters(const pugi::xml_node canopy_node) : BeanParameters(){
    readParametersFromXML(canopy_node);
}

void BeanParameters::readParametersFromXML(const pugi::xml_node canopy_node){
    BaseCanopyParameters::readParametersFromXML(canopy_node);

    float new_leaf_length = XMLloadfloat(canopy_node, "leaf_length");
    if (new_leaf_length != nullvalue_f) {
        leaf_length = new_leaf_length;
    }

    int2 new_leaf_subdivisions = XMLloadint2(canopy_node, "leaf_subdivisions");
    if (new_leaf_subdivisions.x != nullvalue_i && new_leaf_subdivisions.y != nullvalue_i) {
        leaf_subdivisions = new_leaf_subdivisions;
    }

    std::string new_leaf_texture_file = XMLloadstring(canopy_node, "leaf_texture_file");
    if ( new_leaf_texture_file != nullvalue_s ) {
        leaf_texture_file = new_leaf_texture_file;
    }

    int new_shoot_subdivisions = XMLloadint(canopy_node, "shoot_subdivisions");
    if (new_shoot_subdivisions != nullvalue_i) {
        shoot_subdivisions = new_shoot_subdivisions;
    }

    float new_stem_radius = XMLloadfloat(canopy_node, "stem_radius");
    if (new_stem_radius != nullvalue_f) {
        stem_radius = new_stem_radius;
    }

    RGBAcolor new_shoot_color = XMLloadrgba(canopy_node, "shoot_color");
    if( new_shoot_color.a != 0 ){
        shoot_color = make_RGBcolor(new_shoot_color.r, new_shoot_color.g, new_shoot_color.b);
    }

    float new_stem_length = XMLloadfloat(canopy_node, "stem_length");
    if (new_stem_length != nullvalue_f ) {
        stem_length = new_stem_length;
    }

    float new_leaflet_length = XMLloadfloat(canopy_node, "leaflet_length");
    if (new_leaflet_length != nullvalue_f ) {
        leaflet_length = new_leaflet_length;
    }

    float new_pod_length = XMLloadfloat(canopy_node, "pod_length");
    if (new_pod_length != nullvalue_f ) {
        pod_length = new_pod_length;
    }

    RGBAcolor new_pod_color = XMLloadrgba(canopy_node, "pod_color");
    if( new_pod_color.a != 0 ){
        pod_color = make_RGBcolor(new_pod_color.r, new_pod_color.g, new_pod_color.b);
    }

    int new_pod_subdivisions = XMLloadint(canopy_node, "pod_subdivisions");
    if (new_pod_subdivisions != nullvalue_i ) {
        pod_subdivisions = new_pod_subdivisions;
    }

    float new_plant_spacing = XMLloadfloat(canopy_node, "plant_spacing");
    if (new_plant_spacing != nullvalue_f) {
        plant_spacing = new_plant_spacing;
    }

    float new_row_spacing = XMLloadfloat(canopy_node, "row_spacing");
    if (new_row_spacing != nullvalue_f) {
        row_spacing = new_row_spacing;
    }

    int2 new_plant_count = XMLloadint2(canopy_node, "plant_count");
    if (new_plant_count.x != nullvalue_i && new_plant_count.y != nullvalue_i) {
        plant_count = new_plant_count;
    }

    float new_germination_probability = XMLloadfloat(canopy_node, "germination_probability");
    if (new_germination_probability != nullvalue_f) {
        germination_probability = new_germination_probability;
    }
}

void BeanParameters::buildPlant(CanopyGenerator& canopy_generator, helios::vec3 origin){
    canopy_generator.bean(*this, origin);
}

void BeanParameters::buildCanopy(CanopyGenerator& canopy_generator){
    canopy_generator.buildCanopy(*this);
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

template <typename CanopyType, typename... Args>
void CanopyGenerator::storeCanopyParameters(Args&&... args){
    static_assert(std::is_base_of<BaseCanopyParameters, CanopyType>::value, "CanopyType must inherit from BaseCanopyParameters");
    canopy_parameters_list.push_back(std::make_shared<CanopyType>(std::forward<Args>(args)...));
}

std::vector<std::shared_ptr<BaseCanopyParameters>> CanopyGenerator::getCanopyParametersList(){
    std::vector<std::shared_ptr<BaseCanopyParameters>> params_list;
    for (const auto& params : canopy_parameters_list) {
        params_list.push_back(params);
    }
    return params_list;
}

void CanopyGenerator::loadXML( const char* filename, bool build ){

    if( printmessages ){
        std::cout << "Reading XML file: " << filename << "..." << std::flush;
    }

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

            HomogeneousCanopyParameters homogeneouscanopyparameters(s);
            storeCanopyParameters<HomogeneousCanopyParameters>(homogeneouscanopyparameters);
            if (build)
                buildCanopy(homogeneouscanopyparameters);

        }

        //Spherical Canopy
        for (pugi::xml_node s = cgen.child("SphericalCrownsCanopyParameters"); s; s = s.next_sibling("SphericalCrownsCanopyParameters")) {

            SphericalCrownsCanopyParameters sphericalcrownscanopyparameters(s);
            storeCanopyParameters<SphericalCrownsCanopyParameters>(sphericalcrownscanopyparameters);
            if (build)
                buildCanopy(sphericalcrownscanopyparameters);

        }

        //Conical Canopy
        for (pugi::xml_node s = cgen.child("ConicalCrownsCanopyParameters"); s; s = s.next_sibling("ConicalCrownsCanopyParameters")) {

            ConicalCrownsCanopyParameters conicalcrownscanopyparameters(s);
            storeCanopyParameters<ConicalCrownsCanopyParameters>(conicalcrownscanopyparameters);
            if (build)
                buildCanopy(conicalcrownscanopyparameters);

        }


        //VSP Grapevine Canopy
        for (pugi::xml_node s = cgen.child("VSPGrapevineParameters"); s; s = s.next_sibling("VSPGrapevineParameters")) {

            VSPGrapevineParameters vspgrapevineparameters(s);
            storeCanopyParameters<VSPGrapevineParameters>(vspgrapevineparameters);
            if (build)
                buildCanopy(vspgrapevineparameters);

        }

        //Split Grapevine Canopy
        for (pugi::xml_node s = cgen.child("SplitGrapevineParameters"); s; s = s.next_sibling(
                "SplitGrapevineParameters")) {

            SplitGrapevineParameters splitgrapevineparameters(s);
            storeCanopyParameters<SplitGrapevineParameters>(splitgrapevineparameters);
            if (build)
                buildCanopy(splitgrapevineparameters);

        }


        //UnilateralGrapevineParameters Canopy
        for (pugi::xml_node s = cgen.child("UnilateralGrapevineParameters"); s; s = s.next_sibling(
                "UnilateralGrapevineParameters")) {

            UnilateralGrapevineParameters unilateralgrapevineparameters(s);
            storeCanopyParameters<UnilateralGrapevineParameters>(unilateralgrapevineparameters);
            if (build)
                buildCanopy(unilateralgrapevineparameters);

        }


        //GobletGrapevineParameters Canopy
        for (pugi::xml_node s = cgen.child("GobletGrapevineParameters"); s; s = s.next_sibling(
                "GobletGrapevineParameters")) {

            GobletGrapevineParameters gobletgrapevineparameters(s);
            storeCanopyParameters<GobletGrapevineParameters>(gobletgrapevineparameters);
            if (build)
                buildCanopy(gobletgrapevineparameters);

        }


        //WhiteSpruceCanopyParameters Canopy
        for (pugi::xml_node s = cgen.child("WhiteSpruceCanopyParameters"); s; s = s.next_sibling("WhiteSpruceCanopyParameters")) {

            WhiteSpruceCanopyParameters whitesprucecanopyparameters(s);
            storeCanopyParameters<WhiteSpruceCanopyParameters>(whitesprucecanopyparameters);
            if (build)
                buildCanopy(whitesprucecanopyparameters);

        }

        //StrawberryParameters Canopy
        for (pugi::xml_node s = cgen.child("StrawberryParameters"); s; s = s.next_sibling("StrawberryParameters")) {

            StrawberryParameters strawberryparameters(s);
            storeCanopyParameters<StrawberryParameters>(strawberryparameters);
            if (build)
                buildCanopy(strawberryparameters);

        }

        //TomatoParameters Canopy
        for (pugi::xml_node s = cgen.child("TomatoParameters"); s; s = s.next_sibling("TomatoParameters")) {

            TomatoParameters tomatoparameters(s);
            storeCanopyParameters<TomatoParameters>(tomatoparameters);
            if (build)
                buildCanopy(tomatoparameters);

        }

        //WalnutCanopyParameters Canopy
        for (pugi::xml_node s = cgen.child("WalnutCanopyParameters"); s; s = s.next_sibling("WalnutCanopyParameters")) {

            WalnutCanopyParameters walnutcanopyparameters(s);
            storeCanopyParameters<WalnutCanopyParameters>(walnutcanopyparameters);
            if (build)
                buildCanopy(walnutcanopyparameters);

        }

        //SorghumCanopyParameters Canopy
        for (pugi::xml_node s = cgen.child("SorghumCanopyParameters"); s; s = s.next_sibling("SorghumCanopyParameters")) {

            SorghumCanopyParameters sorghumcanopyparameters(s);
            storeCanopyParameters<SorghumCanopyParameters>(sorghumcanopyparameters);
            if (build)
                buildCanopy(sorghumcanopyparameters);

        }

        //BeanParameters Canopy
        for (pugi::xml_node s = cgen.child("BeanParameters"); s; s = s.next_sibling("BeanParameters")) {

            BeanParameters beanparameters(s);
            storeCanopyParameters<BeanParameters>(beanparameters);
            if (build)
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

            if (build)
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

void CanopyGenerator::buildCanopies(){
    for (const auto& canopy_parameters : canopy_parameters_list) {
        canopy_parameters->buildCanopy(*this);
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

            if( params.missing_plant_probability > 0 ){
                float random_draw = context->randu();
                if( random_draw <= params.missing_plant_probability ){
                    // Don't add the plant
                    continue;
                }
            }

            float plant_spacing = params.plant_spacing + getVariation(params.plant_spacing_spread, generator);
            float row_spacing = params.row_spacing + getVariation(params.row_spacing_spread, generator);
            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*row_spacing, 0 );

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

            if( params.missing_plant_probability > 0 ){
                float random_draw = context->randu();
                if( random_draw <= params.missing_plant_probability ){
                    // Don't add the plant
                    continue;
                }
            }

            float plant_spacing = params.plant_spacing + getVariation(params.plant_spacing_spread, generator);
            float row_spacing = params.row_spacing + getVariation(params.row_spacing_spread, generator);
            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*row_spacing, 0 );

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

            if( params.missing_plant_probability > 0 ){
                float random_draw = context->randu();
                if( random_draw <= params.missing_plant_probability ){
                    // Don't add the plant
                    continue;
                }
            }

            float plant_spacing = params.plant_spacing + getVariation(params.plant_spacing_spread, generator);
            float row_spacing = params.row_spacing + getVariation(params.row_spacing_spread, generator);
            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*row_spacing, 0 );

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

            if( params.missing_plant_probability > 0 ){
                float random_draw = context->randu();
                if( random_draw <= params.missing_plant_probability ){
                    // Don't add the plant
                    continue;
                }
            }

            float plant_spacing = params.plant_spacing + getVariation(params.plant_spacing_spread, generator);
            float row_spacing = params.row_spacing + getVariation(params.row_spacing_spread, generator);
            vec3 center = params.canopy_origin+make_vec3(-0.5f*canopy_extent.x+(float(i)+0.5f)*plant_spacing, -0.5f*canopy_extent.y+(float(j)+0.5f)*row_spacing, 0 );

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

void CanopyGenerator::buildIndividualPlants(helios::vec3 position){
    for (const auto& params : canopy_parameters_list) {
        params->buildPlant(*this, position);
    }
}

void CanopyGenerator::buildIndividualPlants(){
    for (const auto& params : canopy_parameters_list) {
        vec3 position = params->canopy_origin;
        params->buildPlant(*this, position);
    }
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
