/** \file "PlantArchitecture.cpp" Primary source file for plant architecture plug-in.

    Copyright (C) 2016-2023 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "PlantArchitecture.h"

#include <utility>
#include "Assets.h"

using namespace helios;

float interpolateTube( const std::vector<float> &P, float frac ){

    assert( frac>=0 && frac<=1 );
    assert( !P.empty() );

    float dl=1.f/float(P.size());

    float f = 0;
    for( int i=0; i<P.size()-1; i++ ){

        float fplus = f+dl;

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

helios::vec3 interpolateTube( const std::vector<vec3> &P, float frac ){

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

PlantArchitecture::PlantArchitecture( helios::Context* context_ptr ) : context_ptr(context_ptr){
    generator = context_ptr->getRandomGenerator();
}

PhytomerParameters::PhytomerParameters() : PhytomerParameters(nullptr){}

PhytomerParameters::PhytomerParameters( std::minstd_rand0 *generator ) {

    internode.origin = make_vec3(0,0,0);
    internode.pitch.initialize( 0.1*M_PI, generator );
    internode.radius = 0.005;
    internode.color = RGB::forestgreen;
    internode.length.initialize(0.05,generator);
    internode.tube_subdivisions = 1;
    internode.curvature.initialize( 0, generator );
    internode.petioles_per_internode = 1;

    petiole.pitch.initialize( 0.5*M_PI, generator );
    petiole.yaw.initialize( 0, generator);
    petiole.roll.initialize( 0, generator );
    petiole.radius.initialize( 0.001, generator );
    petiole.length.initialize( 0.05, generator );
    petiole.curvature.initialize(0, generator);
    petiole.taper.initialize( 0, generator );
    petiole.tube_subdivisions = 1;
    petiole.leaves_per_petiole = 1;

    leaf.pitch.initialize( 0, generator );
    leaf.yaw.initialize( 0, generator );
    leaf.roll.initialize( 0, generator );
    leaf.leaflet_offset.initialize( 0, generator );
    leaf.leaflet_scale = 1;
    leaf.prototype_scale = make_vec3(0.05,0.025, 1.f);

    //--- inflorescence ---//
    inflorescence.length.initialize(0.05,generator);
    inflorescence.fruit_per_inflorescence.initialize(1,generator);
    inflorescence.fruit_offset.initialize(0,generator);
    inflorescence.rachis_radius.initialize(0.001,generator);
    inflorescence.curvature.initialize(0,generator);
    inflorescence.fruit_arrangement_pattern = "alternate";
    inflorescence.tube_subdivisions = 1;
    inflorescence.fruit_pitch.initialize(0,generator);
    inflorescence.fruit_roll.initialize(0,generator);
    inflorescence.fruit_prototype_scale = make_vec3(0.0075,0.0075,0.0075);
    inflorescence.flower_prototype_scale = make_vec3(0.0075,0.0075,0.0075);

}

PhytomerParameters::PhytomerParameters( const PhytomerParameters& parameters_copy ){
    inflorescence = parameters_copy.inflorescence;
    internode = parameters_copy.internode;
    petiole = parameters_copy.petiole;
    leaf = parameters_copy.leaf;
}

ShootParameters::ShootParameters() : ShootParameters(nullptr) {}

ShootParameters::ShootParameters( std::minstd_rand0 *generator ) {
    max_nodes = 5;
    shoot_internode_taper.initialize(0.5,generator);

    phyllochron.initialize(0.0001,generator);
    growth_rate.initialize(0,generator);

    bud_break_probability = 0;
    flower_probability = 1;
    fruit_set_probability = 1;

    bud_time.initialize(0,generator);

    child_insertion_angle_tip.initialize(deg2rad(20), generator);
    child_insertion_angle_decay_rate.initialize(deg2rad(10), generator);

    child_internode_length_max.initialize(0.02, generator);
    child_internode_length_min.initialize(0.002, generator);
    child_internode_length_decay_rate.initialize(0.005, generator);
}

void ShootParameters::defineChildShootTypes( const std::vector<std::string> &a_child_shoot_type_labels, const std::vector<float> &a_child_shoot_type_probabilities ){

    if( a_child_shoot_type_labels.size()!=a_child_shoot_type_probabilities.size() ){
        helios_runtime_error("ERROR (ShootParameters::defineChildShootTypes): Child shoot type labels and probabilities must be the same size.");
    }else if( a_child_shoot_type_labels.empty() ){
        helios_runtime_error("ERROR (ShootParameters::defineChildShootTypes): Input argument vectors were empty.");
    }else if( sum(a_child_shoot_type_probabilities)!=1.f ){
        helios_runtime_error("ERROR (ShootParameters::defineChildShootTypes): Child shoot type probabilities must sum to 1.");
    }

    child_shoot_type_labels = a_child_shoot_type_labels;
    child_shoot_type_probabilities = a_child_shoot_type_probabilities;

}


void PlantArchitecture::defineShootType( const std::string &shoot_type_label, const ShootParameters &shoot_params ) {
    shoot_types.emplace(shoot_type_label, shoot_params );
}

helios::vec3 Phytomer::getInternodeAxisVector(float stem_fraction) const{
    return getAxisVector( stem_fraction, internode_vertices );
}

helios::vec3 Phytomer::getPetioleAxisVector(float stem_fraction) const{
    return getAxisVector( stem_fraction, petiole_vertices );
}

helios::vec3 Phytomer::getAxisVector( float stem_fraction, const std::vector<helios::vec3> &axis_vertices ) const{

    assert( stem_fraction>=0 && stem_fraction<=1 );

    float df = 0.1f;
    float frac_plus, frac_minus;
    if( stem_fraction+df<=1 ) {
        frac_minus = stem_fraction;
        frac_plus = stem_fraction + df;
    }else{
        frac_minus = stem_fraction-df;
        frac_plus = stem_fraction;
    }

    vec3 node_minus = interpolateTube( axis_vertices, frac_minus );
    vec3 node_plus = interpolateTube( axis_vertices, frac_plus );

    vec3 norm = node_plus-node_minus;
    norm.normalize();

    return norm;

}

float Phytomer::getInternodeLength() const{

    // \todo
    return 0;
}

float Phytomer::getPetioleLength() const{

    // \todo
    return 0;
}

float Phytomer::getInternodeRadius( float stem_fraction ) const{

    return interpolateTube( internode_radii, stem_fraction );

}

float Phytomer::getPetioleRadius( float stem_fraction ) const{

    return interpolateTube( petiole_radii, stem_fraction );

}

int Shoot::addPhytomer(const PhytomerParameters &params, const AxisRotation &shoot_base_rotation, float internode_scale_factor_fraction, float leaf_scale_factor_fraction) {

    vec3 parent_internode_axis;
    vec3 parent_petiole_axis;
    if( phytomers.empty() ) { //very first phytomer on shoot
        if( parentID==-1 ) { //very first shoot
            parent_internode_axis = make_vec3(0, 0, 1);
            parent_petiole_axis = make_vec3(0, 1, 0);
        }else{
            assert( parentID < shoot_tree_ptr->size() && parentNode < shoot_tree_ptr->at(parentID)->phytomers.size() );
            parent_internode_axis = shoot_tree_ptr->at(parentID)->phytomers.at(parentNode)->getInternodeAxisVector(1.f);
            parent_petiole_axis = shoot_tree_ptr->at(parentID)->phytomers.at(parentNode)->getPetioleAxisVector(0.f);
        }
    }else {
        parent_internode_axis = phytomers.back()->getInternodeAxisVector(1.f);
        parent_petiole_axis = phytomers.back()->getPetioleAxisVector(0.f);
    }

    std::shared_ptr<Phytomer> phytomer = std::make_shared<Phytomer>(params, shoot_parameters, phytomers.size(), parent_internode_axis, parent_petiole_axis, shoot_base_rotation, internode_scale_factor_fraction, leaf_scale_factor_fraction, rank, context_ptr);

    phytomer->flower_bud_state = BUD_DORMANT;
    phytomer->vegetative_bud_state = BUD_DORMANT;

    shoot_tree_ptr->at(ID)->phytomers.push_back(phytomer);

    return (int)phytomers.size()-1;

}

void Shoot::breakDormancy(){

    dormant = false;

    for( auto &phytomer : phytomers ) {
        phytomer->inflorescence_age = 0;
        if (phytomer->flower_bud_state != BUD_DEAD) {
            phytomer->flower_bud_state = BUD_ACTIVE;
        }
        if( phytomer->vegetative_bud_state!=BUD_DEAD) {
            phytomer->vegetative_bud_state = BUD_ACTIVE;
        }
    }

}

void Shoot::makeDormant(){

    dormant = true;
    dormancy_cycles++;

    for( auto &phytomer : phytomers ){
        if (phytomer->flower_bud_state != BUD_DEAD) {
            phytomer->flower_bud_state = BUD_DORMANT;
        }
        if( phytomer->vegetative_bud_state!=BUD_DEAD) {
            phytomer->vegetative_bud_state = BUD_DORMANT;
        }
    }

}

Phytomer::Phytomer(const PhytomerParameters &params, const ShootParameters &parent_shoot_parameters, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, const AxisRotation &shoot_base_rotation,
                   float internode_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, helios::Context *context_ptr) : phytomer_parameters(params), context_ptr(context_ptr), rank(rank) {

    //Number of longitudinal segments for internode and petiole
    //if Ndiv=0, use Ndiv=1 (but don't add any primitives to Context)
    uint Ndiv_internode = std::max(uint(1), phytomer_parameters.internode.tube_subdivisions);
    uint Ndiv_petiole = std::max(uint(1), phytomer_parameters.petiole.tube_subdivisions);

    current_internode_scale_factor = internode_scale_factor_fraction;
    current_leaf_scale_factor = leaf_scale_factor_fraction;

    //Length of longitudinal segments
    internode_length = internode_scale_factor_fraction * phytomer_parameters.internode.length.val();
    float dr_internode = internode_length / float(phytomer_parameters.internode.tube_subdivisions);
    float dr_internode_max = phytomer_parameters.internode.length.val() / float(phytomer_parameters.internode.tube_subdivisions);
    petiole_length = leaf_scale_factor_fraction * phytomer_parameters.petiole.length.val();
    float dr_petiole = petiole_length / float(phytomer_parameters.petiole.tube_subdivisions);
    float dr_petiole_max = phytomer_parameters.petiole.length.val() / float(phytomer_parameters.petiole.tube_subdivisions);

    float internode_pitch = phytomer_parameters.internode.pitch.val();

    //Initialize segment vertices vector
    internode_vertices.resize(Ndiv_internode+1);
    internode_vertices.at(0) = phytomer_parameters.internode.origin;
    petiole_vertices.resize(Ndiv_petiole+1 );

    internode_radii.resize( Ndiv_internode+1 );
    internode_radii.at(0) = internode_scale_factor_fraction * phytomer_parameters.internode.radius;
    petiole_radii.resize( Ndiv_petiole+1 );
    petiole_radii.at(0) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val();

    internode_colors.resize( Ndiv_internode+1 );
    internode_colors.at(0) = phytomer_parameters.internode.color;
    petiole_colors.resize( Ndiv_petiole+1 );
    petiole_colors.at(0) = phytomer_parameters.internode.color;

    vec3 internode_axis = parent_internode_axis;

    vec3 petiole_rotation_axis = cross(parent_internode_axis, parent_petiole_axis );
    if(petiole_rotation_axis == make_vec3(0, 0, 0) ){
        petiole_rotation_axis = make_vec3(1, 0, 0);
    }

    vec3 shoot_bending_axis;

    if( phytomer_index==0 ){ //if this is the first phytomer along a shoot, apply the origin rotation about the parent axis

        shoot_bending_axis = make_vec3(-1,0,0);

        //pitch rotation for shoot base rotation
        if( shoot_base_rotation.pitch!=0 ) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, shoot_bending_axis,-shoot_base_rotation.pitch);
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, shoot_bending_axis,-shoot_base_rotation.pitch);
        }

        //roll rotation for shoot base rotation
        if( shoot_base_rotation.roll!=0 ) {
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis,-shoot_base_rotation.roll);
         }

        //yaw rotation for shoot base rotation
        if( shoot_base_rotation.yaw!=0 ) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis,shoot_base_rotation.yaw);
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis,shoot_base_rotation.yaw);
            shoot_bending_axis = rotatePointAboutLine(shoot_bending_axis, nullorigin, parent_internode_axis,shoot_base_rotation.yaw);
        }

        //pitch rotation for phytomer base
        if( internode_pitch!=0 ) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, -0.5f*internode_pitch );
        }

    }else {

        if( parent_internode_axis == make_vec3(0,0,1) ){
            shoot_bending_axis = make_vec3(1,0,0);
        }else {
            shoot_bending_axis = -1 * cross(parent_internode_axis, make_vec3(0, 0, 1));
        }

        //pitch rotation for phytomer base
        if ( internode_pitch != 0) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis,-1.25f*internode_pitch );
        }

    }

    // create internode
    for( int i=1; i<=Ndiv_internode; i++ ){

        //apply curvature
        if( phytomer_parameters.internode.curvature.val()>0 ) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, shoot_bending_axis, -deg2rad(phytomer_parameters.internode.curvature.val() * dr_internode_max));
        }

        internode_vertices.at(i) = internode_vertices.at(i - 1) + dr_internode * internode_axis;

        internode_radii.at(i) = internode_scale_factor_fraction * phytomer_parameters.internode.radius;
        internode_colors.at(i) = phytomer_parameters.internode.color;

    }

    internode_objIDs = makeTubeFromCones( 10, internode_vertices, internode_radii, internode_colors, context_ptr );

    //--- create petiole ---//

    petiole_vertices.at(0) = internode_vertices.back();

    vec3 petiole_axis = internode_axis;

    //petiole pitch rotation
    if( phytomer_parameters.petiole.pitch.val()!=0 ) {
        petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, petiole_rotation_axis, std::abs(phytomer_parameters.petiole.pitch.val()) );
    }

    //petiole yaw rotation
    if( phytomer_parameters.petiole.yaw.val()!=0 ) {
        petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, internode_axis, phytomer_parameters.petiole.yaw.val() );
        petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis, phytomer_parameters.petiole.yaw.val() );
    }

    petiole_objIDs.resize(phytomer_parameters.internode.petioles_per_internode);

    for(int bud=0; bud < phytomer_parameters.internode.petioles_per_internode; bud++ ) {

        if( bud>0 ) {
            float budrot = float(bud)*2.f*M_PI/float(phytomer_parameters.internode.petioles_per_internode);
            petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, internode_axis, budrot );
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis, budrot );
        }

        for (int j = 1; j <= Ndiv_petiole; j++) {

            if( phytomer_parameters.petiole.curvature.val()>0 ) {
                petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, petiole_rotation_axis, -deg2rad(phytomer_parameters.petiole.curvature.val() * dr_petiole_max));
            }

            petiole_vertices.at(j) = petiole_vertices.at(j - 1) + dr_petiole * petiole_axis;

            petiole_radii.at(j) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val() * (1.f - phytomer_parameters.petiole.taper.val() / float(Ndiv_petiole - 1) * float(j) );
            petiole_colors.at(j) = phytomer_parameters.internode.color;

        }

        petiole_objIDs.at(bud) = makeTubeFromCones(10, petiole_vertices, petiole_radii, petiole_colors, context_ptr);

        //--- create leaves ---//

        vec3 petiole_tip_axis = getPetioleAxisVector(1.f);

        vec3 leaf_rotation_axis = cross(internode_axis, petiole_tip_axis );

        for(int leaf=0; leaf < phytomer_parameters.petiole.leaves_per_petiole; leaf++ ){

            float ind_from_tip = float(leaf)-float(phytomer_parameters.petiole.leaves_per_petiole-1)/2.f;

            uint objID_leaf;
            if( phytomer_parameters.internode.petioles_per_internode==1 ){
                objID_leaf = phytomer_parameters.leaf.prototype_function(context_ptr,1,(int)ind_from_tip);
            }else{
                objID_leaf = phytomer_parameters.leaf.prototype_function(context_ptr,1,bud);
            }

            // -- scaling -- //

            vec3 leaf_scale = leaf_scale_factor_fraction * phytomer_parameters.leaf.prototype_scale;
            if( phytomer_parameters.petiole.leaves_per_petiole>0 && phytomer_parameters.leaf.leaflet_scale.val()!=1.f && ind_from_tip!=0 ){
                leaf_scale = powf(phytomer_parameters.leaf.leaflet_scale.val(),fabs(ind_from_tip))*leaf_scale;
            }

            context_ptr->scaleObject( objID_leaf, leaf_scale );

            float compound_rotation = 0;
            if( phytomer_parameters.petiole.leaves_per_petiole>1 ) {
                if (phytomer_parameters.leaf.leaflet_offset.val() == 0) {
                    float dphi = M_PI / (floor(0.5 * float(phytomer_parameters.petiole.leaves_per_petiole - 1)) + 1);
                    compound_rotation = -float(M_PI) + dphi * (leaf + 0.5f);
                } else {
                    if( leaf == float(phytomer_parameters.petiole.leaves_per_petiole-1)/2.f ){ //tip leaf
                        compound_rotation = 0;
                    }else if( leaf < float(phytomer_parameters.petiole.leaves_per_petiole-1)/2.f ) {
                        compound_rotation = -0.5*M_PI;
                    }else{
                        compound_rotation = 0.5*M_PI;
                    }
                }
            }

            // -- rotations -- //

            //\todo All the rotations below should be based on local_petiole_axis, but it doesn't seem to be working
//            vec3 local_petiole_axis = interpolateTube( petiole_vertices, 1.f-fabs(ind_from_tip)*phytomer_parameters.leaf.leaflet_offset.val() );

            //pitch rotation
            float pitch_rot = phytomer_parameters.leaf.pitch.val();
            phytomer_parameters.leaf.pitch.resample();
            if( ind_from_tip==0 ){
                pitch_rot += asin_safe(petiole_tip_axis.z);
            }
            context_ptr->rotateObject(objID_leaf, -pitch_rot , "y" );

            //yaw rotation
            if( ind_from_tip!=0 ){
                float yaw_rot = -phytomer_parameters.leaf.yaw.val()*compound_rotation/fabs(compound_rotation);
                phytomer_parameters.leaf.yaw.resample();
                context_ptr->rotateObject( objID_leaf, yaw_rot, "z" );
            }

            //roll rotation
            if( ind_from_tip!= 0){
                float roll_rot = (asin_safe(petiole_tip_axis.z)+phytomer_parameters.leaf.roll.val())*compound_rotation/std::fabs(compound_rotation);
                phytomer_parameters.leaf.roll.resample();
                context_ptr->rotateObject(objID_leaf, roll_rot, "x" );
            }

            //rotate to azimuth of petiole
            context_ptr->rotateObject( objID_leaf, -std::atan2(petiole_tip_axis.y, petiole_tip_axis.x)+compound_rotation, "z" );


            // -- translation -- //

            vec3 leaf_base = petiole_vertices.back();
            if( phytomer_parameters.petiole.leaves_per_petiole>1 && phytomer_parameters.leaf.leaflet_offset.val()>0 ){
                if( ind_from_tip != 0 ) {
                    float offset = (fabs(ind_from_tip) - 0.5f) * phytomer_parameters.leaf.leaflet_offset.val() * phytomer_parameters.petiole.length.val();
                    leaf_base = interpolateTube(petiole_vertices, 1.f - offset / phytomer_parameters.petiole.length.val() );
                }
            }

            context_ptr->translateObject( objID_leaf, leaf_base );

            leaf_objIDs.push_back( objID_leaf );
            leaf_bases.push_back( leaf_base );

        }

        if( petiole_axis==make_vec3(0,0,1) ) {
            inflorescence_bending_axis = make_vec3(1, 0, 0);
        }else{
            inflorescence_bending_axis = cross(make_vec3(0, 0, 1), petiole_axis);
        }

    }

}

void Phytomer::addInfluorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation,
                                 const helios::vec3 &a_inflorescence_bending_axis) {

    float dr_rachis = phytomer_parameters.inflorescence.length.val()/float(phytomer_parameters.inflorescence.tube_subdivisions);

    std::vector<vec3> rachis_vertices(phytomer_parameters.inflorescence.tube_subdivisions+1);
    rachis_vertices.at(0) = base_position;
    std::vector<float> rachis_radii(phytomer_parameters.inflorescence.tube_subdivisions+1);
    rachis_radii.at(0) = phytomer_parameters.inflorescence.rachis_radius.val();
    std::vector<RGBcolor> rachis_colors(phytomer_parameters.inflorescence.tube_subdivisions+1);
    rachis_colors.at(0) = phytomer_parameters.internode.color;

    vec3 rachis_axis = getAxisVector( 1.f, internode_vertices );

    float theta_base = fabs(cart2sphere(rachis_axis).zenith);

    for( int i=1; i<=phytomer_parameters.inflorescence.tube_subdivisions; i++ ){

        float theta_curvature = fabs(deg2rad(phytomer_parameters.inflorescence.curvature.val()*dr_rachis));
         if( theta_curvature*float(i) < M_PI-theta_base ) {
            rachis_axis = rotatePointAboutLine(rachis_axis, nullorigin, inflorescence_bending_axis, theta_curvature);
        }else{
            rachis_axis = make_vec3(0,0,-1);
        }

        rachis_vertices.at(i) = rachis_vertices.at(i - 1) + dr_rachis * rachis_axis;

        rachis_radii.at(i) = phytomer_parameters.inflorescence.rachis_radius.val();
        rachis_colors.at(i) = phytomer_parameters.internode.color;

    }

    //\todo This should be in separate vector from flowers/fruit
    //inflorescence_objIDs.push_back( context_ptr->addTubeObject(10, rachis_vertices, rachis_radii, rachis_colors ) );

    for(int fruit=0; fruit < phytomer_parameters.inflorescence.fruit_per_inflorescence.val(); fruit++ ){

        uint objID_fruit;
        vec3 fruit_scale;

        if( flower_bud_state == BUD_FLOWERING ){
            objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr,1,0);
            fruit_scale = phytomer_parameters.inflorescence.flower_prototype_scale;
        }else{
            objID_fruit = phytomer_parameters.inflorescence.fruit_prototype_function(context_ptr,1,0);
            fruit_scale = phytomer_parameters.inflorescence.fruit_prototype_scale;
        }
        inflorescence_age = 0;

        float ind_from_tip = fabs(fruit-float(phytomer_parameters.inflorescence.fruit_per_inflorescence.val()-1)/2.f);

        context_ptr->scaleObject( objID_fruit, fruit_scale );

        float compound_rotation = 0;
        if( phytomer_parameters.inflorescence.fruit_per_inflorescence.val()>1 ) {
            if (phytomer_parameters.inflorescence.fruit_offset.val() == 0) {
                float dphi = M_PI / (floor(0.5 * float(phytomer_parameters.inflorescence.fruit_per_inflorescence.val() - 1)) + 1);
                compound_rotation = -float(M_PI) + dphi * (fruit + 0.5f);
            } else {
                if( fruit == float(phytomer_parameters.inflorescence.fruit_per_inflorescence.val()-1)/2.f ){ //tip leaf
                    compound_rotation = 0;
                }else if( fruit < float(phytomer_parameters.inflorescence.fruit_per_inflorescence.val()-1)/2.f ) {
                    compound_rotation = -0.5*M_PI;
                }else{
                    compound_rotation = 0.5*M_PI;
                }
            }
        }

        //\todo Once rachis curvature is added, rachis_axis needs to become rachis_tip_axis
        rachis_axis = inflorescence_bending_axis; //\todo This is needed to get rotation of fruit without an actual rachis
        //pitch rotation
        context_ptr->rotateObject(objID_fruit, -asin_safe(rachis_axis.z) + phytomer_parameters.inflorescence.fruit_pitch.val(), "y" );
        phytomer_parameters.inflorescence.fruit_pitch.resample();

        //azimuth rotation
        context_ptr->rotateObject( objID_fruit, -std::atan2(rachis_axis.y, rachis_axis.x)+compound_rotation, "z" );

        //roll rotation
        context_ptr->rotateObject(objID_fruit, phytomer_parameters.inflorescence.fruit_roll.val(), make_vec3(0,0,0), rachis_axis );
        phytomer_parameters.inflorescence.fruit_roll.resample();

        vec3 fruit_base = rachis_vertices.back();
        if( phytomer_parameters.inflorescence.fruit_per_inflorescence.val()>1 && phytomer_parameters.inflorescence.fruit_offset.val()>0 ){
            if( ind_from_tip != 0 ) {
                float offset = 0;
                if( phytomer_parameters.inflorescence.fruit_arrangement_pattern == "opposite" ){
                    offset = (ind_from_tip - 0.5f) * phytomer_parameters.inflorescence.fruit_offset.val() * phytomer_parameters.inflorescence.length.val();
                }else if( phytomer_parameters.inflorescence.fruit_arrangement_pattern == "alternate" ){
                    offset = (ind_from_tip - 0.5f + 0.5f*float(fruit>float(phytomer_parameters.inflorescence.fruit_per_inflorescence.val()-1)/2.f) ) * phytomer_parameters.inflorescence.fruit_offset.val() * phytomer_parameters.inflorescence.length.val();
                }else{
                    helios_runtime_error("ERROR (PlantArchitecture::addInfluorescence): Invalid fruit arrangement pattern.");
                }
                float frac = 1;
                if( phytomer_parameters.inflorescence.length.val()>0 ){
                    frac = 1.f - offset / phytomer_parameters.inflorescence.length.val();
                }
                fruit_base = interpolateTube(rachis_vertices, frac);
            }
        }

        context_ptr->translateObject( objID_fruit, fruit_base );

        inflorescence_bases.push_back( fruit_base );

        inflorescence_objIDs.push_back( objID_fruit );

    }

    context_ptr->setPrimitiveData( context_ptr->getObjectPrimitiveUUIDs( inflorescence_objIDs ), "rank", rank );

}

void Phytomer::scaleInternode( float girth_scale_factor, float length_scale_factor ){

    if( girth_scale_factor!=1.f || length_scale_factor!=1.f ){

        if( length_scale_factor!=1.f ){
            internode_length *= length_scale_factor;
            current_internode_scale_factor *= length_scale_factor;
        }

        int node = 0;
        vec3 last_base = internode_vertices.front();
        for( uint objID : internode_objIDs ) {
            if( girth_scale_factor!=1.f ) {
                context_ptr->getConeObjectPointer(objID)->scaleGirth(girth_scale_factor);
                internode_radii.at(node) *= girth_scale_factor;
            }
            if( length_scale_factor!=1.f ){
                context_ptr->getConeObjectPointer(objID)->scaleLength(length_scale_factor);
                if( node>0 ) {
                    vec3 new_base = context_ptr->getConeObjectNode(objID, 0);
                    context_ptr->translateObject(objID, last_base - new_base);
                }
                last_base = context_ptr->getConeObjectNode( objID, 1 );
                internode_vertices.at(node+1 ) = last_base;
            }
            node++;
        }

    }


}

void Phytomer::setPetioleBase( const helios::vec3 &base_position ){

    vec3 old_base = petiole_vertices.front();
    vec3 shift = base_position - old_base;

    for( auto & vertex : petiole_vertices){
        vertex += shift;
    }

    context_ptr->translateObject( flatten(petiole_objIDs), shift );
    context_ptr->translateObject( leaf_objIDs, shift );
    for(auto & leaf_base : leaf_bases){
        leaf_base += shift;
    }

}

void Phytomer::setPhytomerBase( const helios::vec3 &base_position ){

    vec3 old_base = internode_vertices.front();
    vec3 shift = base_position - old_base;

    for( auto & vertex : internode_vertices){
        vertex += shift;
    }

    for( auto & vertex : petiole_vertices){
        vertex += shift;
    }

    context_ptr->translateObject( internode_objIDs, shift );
    context_ptr->translateObject( flatten(petiole_objIDs), shift );
    context_ptr->translateObject( leaf_objIDs, shift );
    for(auto & leaf_base : leaf_bases){
        leaf_base += shift;
    }
    context_ptr->translateObject( inflorescence_objIDs, shift );

}

void Phytomer::setInternodeScale( float internode_scale_factor_fraction ){

    assert(internode_scale_factor_fraction >= 0 && internode_scale_factor_fraction <= 1 );

    if(internode_scale_factor_fraction == current_internode_scale_factor ){
        return;
    }

    float delta_scale = internode_scale_factor_fraction / current_internode_scale_factor;

    internode_length = internode_length*delta_scale;
    current_internode_scale_factor = internode_scale_factor_fraction;

    int node = 0;
    vec3 last_base = internode_vertices.front();
    for( uint objID : internode_objIDs ) {
        context_ptr->getConeObjectPointer(objID)->scaleLength(delta_scale );
        context_ptr->getConeObjectPointer(objID)->scaleGirth( delta_scale );
        internode_radii.at(node) *= delta_scale;
        if( node>0 ) {
            vec3 new_base = context_ptr->getConeObjectNode(objID, 0);
            context_ptr->translateObject(objID, last_base - new_base);
        }
        last_base = context_ptr->getConeObjectNode( objID, 1 );
        internode_vertices.at(node+1 ) = last_base;
        node++;
    }

    //translate leaf to new internode position
    setPetioleBase( internode_vertices.back() );

}

void Phytomer::setLeafScale( float leaf_scale_factor_fraction ){

    assert(leaf_scale_factor_fraction >= 0 && leaf_scale_factor_fraction <= 1 );

    if(leaf_scale_factor_fraction == current_leaf_scale_factor ){
        return;
    }

    float delta_scale = leaf_scale_factor_fraction / current_leaf_scale_factor;

    petiole_length *= delta_scale;
    current_leaf_scale_factor = leaf_scale_factor_fraction;

    //scale the petiole
    int node = 0;
    vec3 old_tip = petiole_vertices.back();
    vec3 last_base = petiole_vertices.front();
    for( auto &petiole : petiole_objIDs ) {
        for (uint objID : petiole) {
            context_ptr->getConeObjectPointer(objID)->scaleLength(delta_scale);
            context_ptr->getConeObjectPointer(objID)->scaleGirth(delta_scale);
            petiole_radii.at(node) *= delta_scale;
            if (node > 0) {
                vec3 new_base = context_ptr->getConeObjectNode(objID, 0);
                context_ptr->translateObject(objID, last_base - new_base);
            }else{
                petiole_vertices.at(0) = context_ptr->getConeObjectNode(objID, 0);
            }
            last_base = context_ptr->getConeObjectNode(objID, 1);
            petiole_vertices.at(node + 1) = last_base;
            node++;
        }
    }

    //scale and translate leaves
    assert( leaf_objIDs.size()==leaf_bases.size() );
    for(int leaf=0; leaf < leaf_objIDs.size(); leaf++ ) {

        float ind_from_tip = float(leaf) - float(phytomer_parameters.petiole.leaves_per_petiole - 1)/2.f;

        context_ptr->translateObject(leaf_objIDs.at(leaf), -1 * leaf_bases.at(leaf));
        context_ptr->scaleObject(leaf_objIDs.at(leaf), delta_scale * make_vec3(1, 1, 1));
        if( ind_from_tip == 0 ) {
            context_ptr->translateObject(leaf_objIDs.at(leaf), petiole_vertices.back());
            leaf_bases.at(leaf) = petiole_vertices.back();
        }else{
            float offset = (fabs(ind_from_tip) - 0.5f) * phytomer_parameters.leaf.leaflet_offset.val() * phytomer_parameters.petiole.length.val();
            vec3 leaf_base = interpolateTube(petiole_vertices, 1.f - offset / phytomer_parameters.petiole.length.val() );
            context_ptr->translateObject(leaf_objIDs.at(leaf), leaf_base);
            leaf_bases.at(leaf) = leaf_base;
        }

    }

}

void Phytomer::setInflorescenceScale( float inflorescence_scale_factor_fraction ){

    assert(inflorescence_scale_factor_fraction >= 0 && inflorescence_scale_factor_fraction <= 1 );

    if(inflorescence_scale_factor_fraction == current_inflorescence_scale_factor ){
        return;
    }

    float delta_scale = inflorescence_scale_factor_fraction / current_inflorescence_scale_factor;

    current_inflorescence_scale_factor = inflorescence_scale_factor_fraction;

    //scale the rachis
//    int node = 0;
//    vec3 old_tip = petiole_vertices.back();
//    vec3 last_base = petiole_vertices.front();
//    for( auto &petiole : petiole_objIDs ) {
//        for (uint objID : petiole) {
//            context_ptr->getConeObjectPointer(objID)->scaleLength(delta_scale);
//            context_ptr->getConeObjectPointer(objID)->scaleGirth(delta_scale);
//            petiole_radii.at(node) *= delta_scale;
//            if (node > 0) {
//                vec3 new_base = context_ptr->getConeObjectNode(objID, 0);
//                context_ptr->translateObject(objID, last_base - new_base);
//            }else{
//                petiole_vertices.at(0) = context_ptr->getConeObjectNode(objID, 0);
//            }
//            last_base = context_ptr->getConeObjectNode(objID, 1);
//            petiole_vertices.at(node + 1) = last_base;
//            node++;
//        }
//    }

    //scale and translate flowers/fruit
    assert( inflorescence_objIDs.size()==inflorescence_bases.size() );
    for(int inflorescence=0; inflorescence < inflorescence_objIDs.size(); inflorescence++ ) {

        float ind_from_tip = float(inflorescence) - float(phytomer_parameters.inflorescence.fruit_per_inflorescence.val() - 1)/2.f;

        context_ptr->translateObject(inflorescence_objIDs.at(inflorescence), -1 * inflorescence_bases.at(inflorescence));
        context_ptr->scaleObject(inflorescence_objIDs.at(inflorescence), delta_scale * make_vec3(1, 1, 1));
        context_ptr->translateObject(inflorescence_objIDs.at(inflorescence), inflorescence_bases.at(inflorescence));

    }

}

void Phytomer::setPhytomerScale(float internode_scale_factor_fraction, float leaf_scale_factor_fraction) {

    setInternodeScale(internode_scale_factor_fraction );
    setLeafScale(leaf_scale_factor_fraction );


}

void Phytomer::changeReproductiveState( BudState a_state ){

    // If state is already at the desired state, do nothing
    if( this->flower_bud_state == a_state ){
        return;
    }

    // Delete geometry from previous reproductive state
    if( this->flower_bud_state==BUD_FLOWERING || this->flower_bud_state==BUD_FRUITING ) {
        context_ptr->deleteObject(inflorescence_objIDs);
        inflorescence_objIDs.resize(0);
        inflorescence_bases.resize(0);
    }

    this->flower_bud_state = a_state;

    if( this->flower_bud_state==BUD_FLOWERING || this->flower_bud_state==BUD_FRUITING ) {
        //\todo Here need to set rotation based on the parent petiole rotation
        addInfluorescence(internode_vertices.back(), make_AxisRotation(0, 0, 0), make_vec3(0, 0, 1));
        if( this->flower_bud_state==BUD_FRUITING ){
            setInflorescenceScale( 0.01 );
        }
    }

}

void Phytomer::removeLeaf(){

    context_ptr->deleteObject(leaf_objIDs);
    context_ptr->deleteObject(flatten(petiole_objIDs));

    leaf_objIDs.resize(0);
    petiole_objIDs.resize(0);

}

Shoot::Shoot(int ID, int parentID, uint parent_node, uint rank, const helios::vec3 &origin, const AxisRotation &shoot_base_rotation, uint current_node_number, ShootParameters shoot_params, const std::string &shoot_type_label,
             std::vector<std::shared_ptr<Shoot> > *shoot_tree_ptr, helios::Context *context_ptr) :
        ID(ID), parentID(parentID), parentNode(parent_node), rank(rank), origin(origin), base_rotation(shoot_base_rotation), current_node_number(current_node_number), shoot_parameters(std::move(shoot_params)), shoot_type_label(shoot_type_label), shoot_tree_ptr(shoot_tree_ptr), context_ptr(context_ptr) {
    assimilate_pool = 0;
    phyllochron_counter = 0;
}

void Shoot::initializePhytomer(){

    PhytomerParameters phytomer_parameters(shoot_parameters.phytomer_parameters);

    for( int i=0; i<current_node_number; i++ ) {

        if( i==0 ){ //first phytomer on shoot
            phytomer_parameters.internode.origin = origin;
            phytomer_parameters.petiole.roll = 0;
        }

        float parent_radius = 1e6;
        if( parentID>=0 ){
            parent_radius = shoot_tree_ptr->at(parentID)->phytomers.at(parentNode)->phytomer_parameters.internode.radius;
        }
        phytomer_parameters.internode.radius = std::min(parent_radius,shoot_parameters.phytomer_parameters.internode.radius)*(1.f-shoot_parameters.shoot_internode_taper.val()*float(i)/float(current_node_number) );

        int pID = addPhytomer(phytomer_parameters, base_rotation, 1.f, 1.f);

        auto phytomer = phytomers.at(pID);

        phytomer_parameters.internode.origin = phytomer->internode_vertices.back();

        context_ptr->setPrimitiveData( context_ptr->getObjectPrimitiveUUIDs(phytomer->internode_objIDs), "rank", rank );
        context_ptr->setPrimitiveData( context_ptr->getObjectPrimitiveUUIDs( flatten(phytomer->petiole_objIDs)), "rank", rank );
        context_ptr->setPrimitiveData( context_ptr->getObjectPrimitiveUUIDs(phytomer->leaf_objIDs), "rank", rank );
        context_ptr->setPrimitiveData( context_ptr->getObjectPrimitiveUUIDs(phytomer->inflorescence_objIDs), "rank", rank );

    }

//    if( parentID!=-1 ) {
//        shoot_tree_ptr->at(parentID)->childIDs[i](ID);
//    }

}

bool PlantArchitecture::sampleChildShootType( uint plantID, uint shootID, std::string &child_shoot_type_label ) const{

    auto shoot_ptr = plant_instances.at(plantID).shoot_tree.at(shootID);

    assert( shoot_ptr->shoot_parameters.child_shoot_type_labels.size() == shoot_ptr->shoot_parameters.child_shoot_type_probabilities.size() );

    child_shoot_type_label = "";

    if (shoot_ptr->shoot_parameters.child_shoot_type_labels.empty()) {
        child_shoot_type_label = shoot_ptr->shoot_type_label;
    }else{
        float randf = context_ptr->randu();
        int shoot_type_index = -1;
        float cumulative_probability = 0;
        for (int s = 0; s < shoot_ptr->shoot_parameters.child_shoot_type_labels.size(); s++) {
            cumulative_probability += shoot_ptr->shoot_parameters.child_shoot_type_probabilities.at(s);
            if (randf < cumulative_probability ) {
                shoot_type_index = s;
                break;
            }
        }
        if (shoot_type_index < 0) {
            shoot_type_index = shoot_ptr->shoot_parameters.child_shoot_type_labels.size() - 1;
        }
        child_shoot_type_label = shoot_ptr->shoot_type_label;
        if (shoot_type_index >= 0) {
            child_shoot_type_label = shoot_ptr->shoot_parameters.child_shoot_type_labels.at(shoot_type_index);
        }
    }

    bool bud_break = true;
    if (context_ptr->randu() > shoot_types.at(child_shoot_type_label).bud_break_probability) {
        bud_break = false;
        child_shoot_type_label = "";
    }

    return bud_break;

}

uint PlantArchitecture::addBaseShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label) {

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);

    validateShootTypes(shoot_parameters);

    if(current_node_number > shoot_parameters.max_nodes ){
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_parameters.max_nodes) + ".");
    }else if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shoot_types.find(shoot_type_label) == shoot_types.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    uint shootID = shoot_tree_ptr->size();

    auto* shoot_new = (new Shoot(shootID, -1, 0, 0, plant_instances.at(plantID).base_position, base_rotation, current_node_number, shoot_parameters, shoot_type_label, shoot_tree_ptr, context_ptr));
    shoot_tree_ptr->emplace_back(shoot_new);
    shoot_new->initializePhytomer();

    return shootID;

}

uint PlantArchitecture::appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label) {

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);

    validateShootTypes(shoot_parameters);

    if( shoot_tree_ptr->empty() ){
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot append shoot to empty shoot. You must call addBaseShoot() first for each plant.");
    }else if( parent_shoot_ID >= int(shoot_tree_ptr->size()) ){
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    }else if(current_node_number > shoot_parameters.max_nodes ){
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_parameters.max_nodes) + ".");
    }else if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shoot_tree_ptr->at(parent_shoot_ID)->phytomers.empty() ){
        std::cout << "WARNING (PlantArchitecture::appendShoot): Shoot does not have any phytomers to append." << std::endl;
    }else if( shoot_types.find(shoot_type_label) == shoot_types.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    //stop parent shoot from producing new phytomers at the apex
    shoot_tree_ptr->at(parent_shoot_ID)->shoot_parameters.max_nodes = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number;

    int shootID = shoot_tree_ptr->size();

    uint parent_node = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number-1;

    uint rank = shoot_tree_ptr->at(parent_shoot_ID)->rank;

    vec3 base_position = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.back()->internode_vertices.back();

    auto * shoot_new = (new Shoot(shootID, parent_shoot_ID, parent_node, rank, base_position, base_rotation, current_node_number, shoot_parameters, shoot_type_label, shoot_tree_ptr, context_ptr));
    shoot_tree_ptr->emplace_back(shoot_new);
    shoot_new->initializePhytomer();

    return shootID;

}

uint PlantArchitecture::addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shoot_types.find(shoot_type_label) == shoot_types.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);

    validateShootTypes(shoot_parameters);

    if(parent_shoot_ID < -1 || parent_shoot_ID >= shoot_tree_ptr->size() ){
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    }

    int parent_rank = -1;
    uint parent_node_count = 0;
    if(parent_shoot_ID != -1 ){
        parent_rank = (int)shoot_tree_ptr->at(parent_shoot_ID)->rank;
        parent_node_count = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number;
    }

    //scale the shoot internode based on proximity from the tip
    if( shoot_parameters.growth_requires_dormancy ){
        shoot_parameters.phytomer_parameters.internode.length = fmax(shoot_parameters.child_internode_length_max.val() - shoot_parameters.child_internode_length_decay_rate.val() * float(parent_node_count-parent_node-1), shoot_parameters.child_internode_length_min.val());
    }

    //set the insertion angle based on proximity from the tip
    float insertion_angle_adjustment = fmin(shoot_parameters.child_insertion_angle_decay_rate.val() * float(parent_node_count-parent_node-1), M_PI/2.f-base_rotation.pitch );

    vec3 node_position;

    if(parent_shoot_ID > -1 ){
        auto shoot_phytomers = &shoot_tree_ptr->at(parent_shoot_ID)->phytomers;

        if( parent_node>=shoot_phytomers->size() ){
            helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Requested to place child shoot on node " + std::to_string(parent_node) + " but parent only has " + std::to_string(shoot_phytomers->size()) + " nodes." );
        }

        node_position = shoot_phytomers->at(parent_node)->internode_vertices.back();

    }

    int childID = shoot_tree_ptr->size();

    auto* shoot_new = (new Shoot(childID, parent_shoot_ID, parent_node, parent_rank + 1, node_position, base_rotation+ make_AxisRotation(insertion_angle_adjustment,0,0), current_node_number, shoot_parameters, shoot_type_label, shoot_tree_ptr, context_ptr));
    shoot_tree_ptr->emplace_back(shoot_new);
    shoot_new->initializePhytomer();

    shoot_tree_ptr->at(parent_shoot_ID)->childIDs[(int)parent_node] = childID;

    return childID;

}

void PlantArchitecture::validateShootTypes( ShootParameters &shoot_parameters ) const{

    assert( shoot_parameters.child_shoot_type_probabilities.size() == shoot_parameters.child_shoot_type_labels.size() );

    for( int ind = shoot_parameters.child_shoot_type_labels.size()-1; ind>=0; ind-- ){
        if( shoot_types.find(shoot_parameters.child_shoot_type_labels.at(ind)) == shoot_types.end() ){
            shoot_parameters.child_shoot_type_labels.erase(shoot_parameters.child_shoot_type_labels.begin()+ind);
            shoot_parameters.child_shoot_type_probabilities.erase(shoot_parameters.child_shoot_type_probabilities.begin()+ind);
        }
    }

}

int PlantArchitecture::addPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_params, float internode_scale_factor_fraction, float leaf_scale_factor_fraction) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::addPhytomerToShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    if(shootID >= shoot_tree_ptr->size() ){
        helios_runtime_error("ERROR (PlantArchitecture::addPhytomerToShoot): Parent with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto parent_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    PhytomerParameters phytomer_parameters(phytomer_params);

    phytomer_parameters.internode.origin = parent_shoot->phytomers.back()->internode_vertices.back();

    phytomer_parameters.internode.radius = phytomer_parameters.internode.radius * (1.f - parent_shoot->shoot_parameters.shoot_internode_taper.val() * float(parent_shoot->current_node_number) / float(parent_shoot->shoot_parameters.max_nodes) );

    int pID = parent_shoot->addPhytomer(phytomer_parameters, parent_shoot->base_rotation, internode_scale_factor_fraction, leaf_scale_factor_fraction);

    parent_shoot->current_node_number ++;

    return pID;

}

void PlantArchitecture::scalePhytomerInternode(uint plantID, uint shootID, uint node_number, float girth_scale_factor, float length_scale_factor) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::scalePhytomerInternode): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto parent_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::scalePhytomerInternode): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if( node_number>=parent_shoot->current_node_number ){
        helios_runtime_error("ERROR (PlantArchitecture::scalePhytomerInternode): Cannot scale internode " + std::to_string(node_number) + " because there are only " + std::to_string(parent_shoot->current_node_number) + " nodes in this shoot.");
    }

    auto phytomer = parent_shoot->phytomers.at(node_number);

    phytomer->scaleInternode(girth_scale_factor, length_scale_factor );


    if( length_scale_factor!=1.f ){

        //shift this phytomer's petiole(s)
        parent_shoot->phytomers.at(node_number)->setPetioleBase( parent_shoot->phytomers.at(node_number)->internode_vertices.back() );

        //shift all downstream phytomers
        for( int node=node_number+1; node<parent_shoot->phytomers.size(); node++ ){
            vec3 upstream_base = parent_shoot->phytomers.at(node-1)->internode_vertices.back();
            parent_shoot->phytomers.at(node)->setPhytomerBase(upstream_base);
        }
    }

}

void PlantArchitecture::setPhytomerInternodeScale(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeScale): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto parent_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeScale): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if( node_number>=parent_shoot->current_node_number ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeScale): Cannot scale internode " + std::to_string(node_number) + " because there are only " + std::to_string(parent_shoot->current_node_number) + " nodes in this shoot.");
    }
    if(internode_scale_factor_fraction < 0 || internode_scale_factor_fraction > 1 ){
        std::cout << "WARNING (PlantArchitecture::setPhytomerInternodeScale): Internode scaling factor was outside the range of 0 to 1. No scaling was applied." << std::endl;
        return;
    }

    parent_shoot->phytomers.at(node_number)->setInternodeScale(internode_scale_factor_fraction);

    //shift all downstream phytomers
    for( int node=node_number+1; node<parent_shoot->phytomers.size(); node++ ){
        vec3 upstream_base = parent_shoot->phytomers.at(node-1)->internode_vertices.back();
        parent_shoot->phytomers.at(node)->setPhytomerBase(upstream_base);
    }

}

void PlantArchitecture::setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeScale): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto parent_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerLeafScale): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if( node_number>=parent_shoot->current_node_number ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerLeafScale): Cannot scale leaf " + std::to_string(node_number) + " because there are only " + std::to_string(parent_shoot->current_node_number) + " nodes in this shoot.");
    }else if( parent_shoot->phytomers.at(node_number)->petiole_objIDs.empty() ){
        return;
    }
    if(leaf_scale_factor_fraction < 0 || leaf_scale_factor_fraction > 1 ){
        std::cout << "WARNING (PlantArchitecture::setPhytomerLeafScale): Leaf scaling factor was outside the range of 0 to 1. No scaling was applied." << std::endl;
        return;
    }

    parent_shoot->phytomers.at(node_number)->setLeafScale(leaf_scale_factor_fraction);

}

void PlantArchitecture::setPhytomerScale(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction, float leaf_scale_factor_fraction) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerScale): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    setPhytomerInternodeScale(plantID, shootID, node_number, internode_scale_factor_fraction);
    setPhytomerLeafScale(plantID, shootID, node_number, leaf_scale_factor_fraction);

}

void PlantArchitecture::setPlantBasePosition(uint plantID, const helios::vec3 &base_position) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).base_position = base_position;

    //\todo Does not work after shoots have been added to the plant.
    if( !plant_instances.at(plantID).shoot_tree.empty() ){
        std::cout << "WARNING (PlantArchitecture::setPlantBasePosition): This function does not work after shoots have been added to the plant." << std::endl;
    }

}

helios::vec3 PlantArchitecture::getPlantBasePosition(uint plantID) const{
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( plant_instances.at(plantID).shoot_tree.empty() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): Plant with ID of " + std::to_string(plantID) + " has no shoots, so could not get a base position.");
    }
    return plant_instances.at(plantID).base_position;
}

void PlantArchitecture::setPlantAge(uint plantID, float a_current_age) {
    //\todo
//    this->current_age = current_age;
}

float PlantArchitecture::getPlantAge(uint plantID) const{
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAge): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( plant_instances.at(plantID).shoot_tree.empty() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAge): Plant with ID of " + std::to_string(plantID) + " has no shoots, so could not get a base position.");
    }
    return plant_instances.at(plantID).current_age;
}


uint PlantArchitecture::getShootNodeCount( uint plantID, uint shootID ) const{
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getShootNodeCount): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( plant_instances.at(plantID).shoot_tree.size()<=shootID ){
        helios_runtime_error("ERROR (PlantArchitecture::getShootNodeCount): Shoot ID is out of range.");
    }
    return plant_instances.at(plantID).shoot_tree.at(shootID)->current_node_number;
}

std::vector<uint> PlantArchitecture::getAllPlantObjectIDs(uint plantID) const{

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getAllPlantObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    for( const auto& shoot: plant_instances.at(plantID).shoot_tree ){
        for( const auto& phytomer: shoot->phytomers ){
            objIDs.insert(objIDs.end(), phytomer->internode_objIDs.begin(), phytomer->internode_objIDs.end() );
            std::vector<uint> petiole_objIDs_flat = flatten(phytomer->petiole_objIDs);
            objIDs.insert(objIDs.end(), petiole_objIDs_flat.begin(), petiole_objIDs_flat.end() );
            objIDs.insert(objIDs.end(), phytomer->leaf_objIDs.begin(), phytomer->leaf_objIDs.end() );
            objIDs.insert(objIDs.end(), phytomer->inflorescence_objIDs.begin(), phytomer->inflorescence_objIDs.end() );
        }
    }

    return objIDs;

}

std::vector<uint> PlantArchitecture::getAllPlantUUIDs(uint plantID) const{
    return context_ptr->getObjectPrimitiveUUIDs(getAllPlantObjectIDs(plantID));
}

PhytomerParameters PlantArchitecture::getPhytomerParametersFromLibrary(const std::string &phytomer_label ){

    PhytomerParameters phytomer_parameters_current(generator);
    
    if( phytomer_label=="bean" ){

        phytomer_parameters_current.internode.pitch = 0.1 * M_PI; //pitch>0 creates zig-zagging
        phytomer_parameters_current.internode.radius = 0.002;
        phytomer_parameters_current.internode.length = 0.015;
        phytomer_parameters_current.internode.curvature = -100;
        phytomer_parameters_current.internode.petioles_per_internode = 1;
        phytomer_parameters_current.internode.color = make_RGBcolor(0.38, 0.48, 0.1);
        phytomer_parameters_current.internode.tube_subdivisions = 5;

        phytomer_parameters_current.petiole.pitch = 0.25 * M_PI;
        phytomer_parameters_current.petiole.yaw = M_PI;
        phytomer_parameters_current.petiole.roll = 0;
        phytomer_parameters_current.petiole.radius = 0.001;
        phytomer_parameters_current.petiole.length = 0.03;
        phytomer_parameters_current.petiole.taper = 0.15;
        phytomer_parameters_current.petiole.curvature = -600;
        phytomer_parameters_current.petiole.tube_subdivisions = 5;
        phytomer_parameters_current.petiole.leaves_per_petiole = 3;

        phytomer_parameters_current.leaf.pitch.normalDistribution( 0, 0.1 * M_PI);
        phytomer_parameters_current.leaf.yaw = 0;
        phytomer_parameters_current.leaf.roll.normalDistribution( 0, 0.05 * M_PI);
        phytomer_parameters_current.leaf.leaflet_offset = 0.3;
        phytomer_parameters_current.leaf.leaflet_scale = 0.9;
        phytomer_parameters_current.leaf.prototype_function = BeanLeafPrototype;
        phytomer_parameters_current.leaf.prototype_scale = 0.04 * make_vec3(1, 1, 1.);

        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = BeanFruitPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.05 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.flower_prototype_scale = 0.05 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.fruit_arrangement_pattern = "opposite";
        phytomer_parameters_current.inflorescence.fruit_per_inflorescence = 4;
        phytomer_parameters_current.inflorescence.fruit_offset = 0.2;
        phytomer_parameters_current.inflorescence.curvature = -200;
        phytomer_parameters_current.inflorescence.length = 0.025;
        phytomer_parameters_current.inflorescence.rachis_radius = 0.001;
        
    }else if( phytomer_label=="cowpea" ){

        phytomer_parameters_current.internode.pitch = 0.1 * M_PI; //pitch>0 creates zig-zagging
        phytomer_parameters_current.internode.tube_subdivisions = 5;
        phytomer_parameters_current.internode.curvature = -100;
        phytomer_parameters_current.internode.radius = 0.0025;
        phytomer_parameters_current.internode.length = 0.015;
        phytomer_parameters_current.internode.color = make_RGBcolor(0.38, 0.48, 0.1);

        phytomer_parameters_current.petiole.pitch = 0.25 * M_PI;
        phytomer_parameters_current.petiole.yaw = M_PI;
        phytomer_parameters_current.petiole.taper = 0.1;
        phytomer_parameters_current.petiole.tube_subdivisions = 5;
        phytomer_parameters_current.petiole.curvature = -600;
        phytomer_parameters_current.petiole.length = 0.02;
        phytomer_parameters_current.petiole.leaves_per_petiole = 3;

        phytomer_parameters_current.leaf.leaflet_offset = 0.3;
        phytomer_parameters_current.leaf.leaflet_scale = 0.9;
        phytomer_parameters_current.leaf.prototype_function = CowpeaLeafPrototype;
        phytomer_parameters_current.leaf.prototype_scale = 0.05 * make_vec3(1, 1, 1);

        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = CowpeaFruitPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.1 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.flower_prototype_scale = 0.1 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.fruit_arrangement_pattern = "opposite";
        phytomer_parameters_current.inflorescence.fruit_per_inflorescence = 4;
        phytomer_parameters_current.inflorescence.fruit_offset = 0.;
        phytomer_parameters_current.inflorescence.curvature = -200;
        phytomer_parameters_current.inflorescence.length = 0.075;
        phytomer_parameters_current.inflorescence.rachis_radius = 0.001;
        
    }else if( phytomer_label=="tomato" ){

        phytomer_parameters_current.internode.pitch = 0.05 * M_PI; //pitch>0 creates zig-zagging
        phytomer_parameters_current.internode.tube_subdivisions = 5;
        phytomer_parameters_current.internode.curvature = -00;
        phytomer_parameters_current.internode.radius = 0.0025;
        phytomer_parameters_current.internode.length.uniformDistribution(0.03,0.05);
        phytomer_parameters_current.internode.color = make_RGBcolor(0.26, 0.38, 0.10);
        phytomer_parameters_current.internode.petioles_per_internode = 1;

        phytomer_parameters_current.petiole.pitch = 0.3 * M_PI;
        phytomer_parameters_current.petiole.taper = 0.1;
        phytomer_parameters_current.petiole.curvature = -400;
        phytomer_parameters_current.petiole.length = 0.1;
        phytomer_parameters_current.petiole.tube_subdivisions = 7;

        phytomer_parameters_current.leaf.pitch = -0.05 * M_PI;
        phytomer_parameters_current.leaf.yaw = 0.05*M_PI;
        phytomer_parameters_current.petiole.leaves_per_petiole = 7;
        phytomer_parameters_current.leaf.leaflet_offset = 0.2;
        phytomer_parameters_current.leaf.leaflet_scale = 0.7;
        phytomer_parameters_current.leaf.prototype_function = TomatoLeafPrototype;
        phytomer_parameters_current.leaf.prototype_scale = 0.05 * make_vec3(1, 1, 1);

        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = TomatoFruitPrototype;
        phytomer_parameters_current.inflorescence.flower_prototype_function = TomatoFlowerPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.01 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.flower_prototype_scale = 0.01 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.fruit_arrangement_pattern = "alternate";
        phytomer_parameters_current.inflorescence.fruit_per_inflorescence = 9;
        phytomer_parameters_current.inflorescence.fruit_offset = 0.14;
        phytomer_parameters_current.inflorescence.curvature = -300;
        phytomer_parameters_current.inflorescence.length = 0.075;
        phytomer_parameters_current.inflorescence.rachis_radius = 0.0005;

    }else if( phytomer_label=="almond" ){

        phytomer_parameters_current.internode.pitch = 0.01 * M_PI;
        phytomer_parameters_current.internode.tube_subdivisions = 5;
        phytomer_parameters_current.internode.curvature = -100;
        phytomer_parameters_current.internode.radius = 0.0025;
        phytomer_parameters_current.internode.length.uniformDistribution(0.01,0.015);
        phytomer_parameters_current.internode.color = make_RGBcolor(0.42,0.27,0.09);
        phytomer_parameters_current.internode.petioles_per_internode = 1;

        phytomer_parameters_current.petiole.pitch =-0.3 * M_PI;
        phytomer_parameters_current.petiole.yaw = 0.5 * M_PI;
        phytomer_parameters_current.petiole.roll = 0. * M_PI;
        phytomer_parameters_current.petiole.taper = 0.1;
        phytomer_parameters_current.petiole.curvature.uniformDistribution(-3000,3000);
        phytomer_parameters_current.petiole.length = 0.01;
        phytomer_parameters_current.petiole.radius = 0.00025;
        phytomer_parameters_current.petiole.tube_subdivisions = 7;

        phytomer_parameters_current.petiole.leaves_per_petiole = 1;
        phytomer_parameters_current.leaf.leaflet_offset = 0.2;
        phytomer_parameters_current.leaf.leaflet_scale = 0.7;
        phytomer_parameters_current.leaf.prototype_function = AlmondLeafPrototype;
        phytomer_parameters_current.leaf.prototype_scale = 0.03 * make_vec3(1, 1, 1);

        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = AlmondFruitPrototype;
        phytomer_parameters_current.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.01 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.flower_prototype_scale = 0.0075 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.fruit_arrangement_pattern = "alternate";
        phytomer_parameters_current.inflorescence.fruit_per_inflorescence = 1;
        phytomer_parameters_current.inflorescence.fruit_offset = 0.14;
        phytomer_parameters_current.inflorescence.curvature = -300;
        phytomer_parameters_current.inflorescence.length = 0.0;
        phytomer_parameters_current.inflorescence.rachis_radius = 0.0005;
        phytomer_parameters_current.inflorescence.fruit_pitch = -0.5*M_PI;
        
    }else{
        helios_runtime_error("ERROR (PlantArchitecture::setCurrentPhytomerParameters): " + phytomer_label + " is not a valid phytomer in the library.");
    }

    return phytomer_parameters_current;
    
}

uint PlantArchitecture::addPlantInstance(const helios::vec3 &base_position, float current_age) {

    if( current_age<0 ){
        helios_runtime_error("ERROR (PlantArchitecture::addPlantInstance): Current age must be greater than or equal to zero.");
    }

    PlantInstance instance(base_position, current_age);

    plant_instances.emplace(plant_count, instance);

    plant_count++;

    return plant_count-1;


}

uint PlantArchitecture::duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, float current_age ){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::duplicatePlantInstance): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto plant_shoot_tree = &plant_instances.at(plantID).shoot_tree;

    uint plantID_new = addPlantInstance(base_position, current_age);

    if( plant_shoot_tree->empty() ){ //no shoots to add
        return plantID_new;
    }

    //add the first shoot
    auto first_shoot = plant_shoot_tree->at(0);
    first_shoot->base_rotation.roll += context_ptr->randu(0.f,2.f*M_PI);
    addBaseShoot(plantID_new, first_shoot->current_node_number, first_shoot->base_rotation, first_shoot->shoot_type_label);

    for( auto &shoot: *plant_shoot_tree ){
        if( shoot->parentID==-1 ){
            continue;
        }
        uint sID = appendShoot(plantID_new, shoot->parentID, shoot->current_node_number, shoot->base_rotation+make_AxisRotation(0,context_ptr->randu(0.f,2.f*M_PI), context_ptr->randu(0.f,2.f*M_PI)), shoot->shoot_type_label);
        for( int i=0; i<shoot->current_node_number; i++ ){
            setPhytomerScale(plantID_new, sID, i, shoot->phytomers.at(i)->current_internode_scale_factor, shoot->phytomers.at(i)->current_leaf_scale_factor);
        }
    }

    return plantID_new;

}

void PlantArchitecture::setPlantPhenologicalThresholds(uint plantID, float assimilate_dormancy_threshold, float dd_to_flowering, float dd_to_fruit_set, float dd_to_fruit_maturity, float dd_to_senescence) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantPhenologicalThresholds): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).assimilate_dormancy_threshold = assimilate_dormancy_threshold;
    plant_instances.at(plantID).dd_to_flowering = dd_to_flowering;
    plant_instances.at(plantID).dd_to_fruit_set = dd_to_fruit_set;
    plant_instances.at(plantID).dd_to_fruit_maturity = dd_to_fruit_maturity;
    plant_instances.at(plantID).dd_to_senescence = dd_to_senescence;

}


void PlantArchitecture::advanceTime( float dt ) {

    for (auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        //\todo placeholder
        incrementAssimilatePool(plantID, -10);

        size_t shoot_count = shoot_tree->size();
        for ( int i=0; i<shoot_count; i++ ){

            auto shoot = shoot_tree->at(i);

            // breaking dormancy
            bool dormancy_broken_this_timestep = false;
            if( shoot->dormancy_cycles==1 && shoot->dormant && shoot->assimilate_pool < plant.second.assimilate_dormancy_threshold ){
                shoot->breakDormancy();
                dormancy_broken_this_timestep = true;
                shoot->assimilate_pool = 1e6;
            }

            if( shoot->dormant ){ //dormant, don't do anything
                continue;
            }

            for( auto &phytomer : shoot->phytomers ) {

                if (phytomer->flower_bud_state == BUD_FLOWERING || phytomer->flower_bud_state == BUD_FRUITING) {
                    phytomer->inflorescence_age += dt;
                }

                // -- Flowering -- //
                if ( phytomer->flower_bud_state == BUD_ACTIVE && ( (!shoot->shoot_parameters.flowers_require_dormancy && phytomer->inflorescence_age >= plant.second.dd_to_flowering) || dormancy_broken_this_timestep ) ) {
                    phytomer->inflorescence_age = 0;
                    if( context_ptr->randu() < shoot->shoot_parameters.flower_probability ) {
                        phytomer->changeReproductiveState(BUD_FLOWERING);
                    } else {
                        phytomer->changeReproductiveState(BUD_DEAD);
                    }
                }

                // -- Fruit Set -- //
                // If the flower bud is in a 'flowering' state, the fruit set occurs after a certain amount of time
                else if ( phytomer->flower_bud_state == BUD_FLOWERING && phytomer->inflorescence_age >= plant.second.dd_to_fruit_set ) {
                    phytomer->inflorescence_age = 0;
                    if( context_ptr->randu() < shoot->shoot_parameters.fruit_set_probability ) {
                        phytomer->changeReproductiveState(BUD_FRUITING);
                    }else{
                        phytomer->changeReproductiveState(BUD_DEAD);
                    }
                }

                // -- Fruit Growth -- //
                // If the flower bud it in a 'fruiting' state, the fruit grows with time
                else if ( phytomer->flower_bud_state == BUD_FRUITING ){
                    float scale = fmin(1,phytomer->inflorescence_age / plant.second.dd_to_fruit_maturity );
                    phytomer->setInflorescenceScale(scale);
                }

            }

            // if shoot has reached max_nodes, don't do anything more with the shoot
            if (shoot->current_node_number >= shoot->shoot_parameters.max_nodes) {
                continue;
            }

            int node_number = 0;
            for (auto &phytomer: shoot->phytomers) {

                // Scale phytomers based on the growth rate

                float dL = dt * shoot->shoot_parameters.growth_rate.val();

                 //internode
                if (phytomer->current_internode_scale_factor < 1) {
                    float scale = fmin(1.f, (phytomer->internode_length + dL) / phytomer->phytomer_parameters.internode.length.val());
                    //std::cout << "dL: " << dL << ", " << phytomer.current_internode_scale_factor << " " << scale << " " << phytomer.internode_length << " " << phytomer.internode_length + dL << " " << phytomer.phytomer_parameters.internode.length.val() << " " << dL / phytomer.internode_length << std::endl;
                    phytomer->setInternodeScale(scale);
                }

                //petiole/leaves
                if (phytomer->current_leaf_scale_factor < 1) {
                    float scale = fmin(1.f, (phytomer->petiole_length + dL) / phytomer->phytomer_parameters.petiole.length.val());
                    phytomer->setLeafScale(scale);
                }

                //shift all downstream phytomers
                for (int node = node_number + 1; node < shoot->phytomers.size(); node++) {
                    vec3 upstream_base = shoot->phytomers.at(node - 1)->internode_vertices.back();
                    shoot->phytomers.at(node)->setPhytomerBase(upstream_base);
                }

                // -- Add shoots at buds based on bud probability -- //
                if (shoot->shoot_parameters.bud_break_probability > 0 && phytomer->vegetative_bud_state == BUD_ACTIVE ) {

                    if (phytomer->age <= shoot->shoot_parameters.bud_time.val() && phytomer->age + dt > shoot->shoot_parameters.bud_time.val() ) {

                       std::string new_shoot_type_label;
                        if(sampleChildShootType(plantID,shoot->ID, new_shoot_type_label) ){
                            uint childID = addChildShoot(plantID, shoot->ID, node_number, 1, make_AxisRotation(shoot->shoot_parameters.child_insertion_angle_tip.val(), context_ptr->randu(0.f, 2.f * M_PI), -0. * M_PI), new_shoot_type_label);
                            setPhytomerScale(plantID, childID, 0, 0.01, 0.01);
                            phytomer->vegetative_bud_state = BUD_DEAD;
                            std::cout << "Adding child shoot to phytomer " << node_number << " of type " << new_shoot_type_label << " on shoot " << shoot->ID << std::endl;
                        }
                    }

                }

                phytomer->age += dt;

                node_number++;
            }

            // If the apical bud is dead, don't do anything more with the shoot
            if( !shoot->meristem_is_alive ){
                continue;
            }

            // -- Add new phytomer at terminal bud based on the phyllochron -- //
            shoot->phyllochron_counter += dt;
            if ( shoot->phyllochron_counter >= 1.f / shoot->shoot_parameters.phyllochron.val()) {
                 int pID = addPhytomerToShoot(plantID, shoot->ID, shoot->phytomers.back()->phytomer_parameters, 0.01, 0.01);
                shoot->phyllochron_counter = 0;
            }

        }

    }
}

void PlantArchitecture::incrementAssimilatePool( uint plantID, uint shootID, float assimilate_increment_mg_g ){

        if( plant_instances.find(plantID) == plant_instances.end() ){
            helios_runtime_error("ERROR (PlantArchitecture::incrementAssimilatePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
        }else if( plant_instances.at(plantID).shoot_tree.size()<=shootID ){
            helios_runtime_error("ERROR (PlantArchitecture::incrementAssimilatePool): Shoot ID is out of range.");
        }

        plant_instances.at(plantID).shoot_tree.at(shootID)->assimilate_pool += assimilate_increment_mg_g;

}

void PlantArchitecture::incrementAssimilatePool( uint plantID, float assimilate_increment_mg_g ){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::incrementAssimilatePool): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for( auto &shoot: plant_instances.at(plantID).shoot_tree ){
        shoot->assimilate_pool += assimilate_increment_mg_g;
    }

}

std::string PlantArchitecture::makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> & shoot_tree) const{

    std::string lstring = current_string;

    if( shoot->parentID!=-1 ) {
        lstring += "[";
    }

    uint node_number = 0;
    for( auto &phytomer: shoot->phytomers ){

        float length = phytomer->internode_length;
        float radius = phytomer->phytomer_parameters.internode.radius;

        if( node_number<shoot->phytomers.size()-1 ) {
            lstring += "Internode(" + std::to_string(length) + "," + std::to_string(radius) + ")";
        }else{
            lstring += "Apex(" + std::to_string(length) + "," + std::to_string(radius) + ")";
        }

        for( int l=0; l<phytomer->phytomer_parameters.internode.petioles_per_internode; l++ ){
            lstring += "~l";
        }

        if( shoot->childIDs.find(node_number)!=shoot->childIDs.end() ){
            lstring = makeShootString(lstring, shoot_tree.at(shoot->childIDs.at(node_number)), shoot_tree );
        }

        node_number++;
    }

    if( shoot->parentID!=-1 ) {
        lstring += "]";
    }

    return lstring;

}

std::string PlantArchitecture::getLSystemsString(uint plantID) const{

    auto plant_shoot_tree = &plant_instances.at(plantID).shoot_tree;

    std::string lsystems_string;

    for( auto &shoot: *plant_shoot_tree ){
        lsystems_string = makeShootString(lsystems_string, shoot, *plant_shoot_tree );
    }

    return lsystems_string;

}

void PlantArchitecture::accumulateShootPhotosynthesis( float dt ){

    uint A_prim_data_missing = 0;

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){

            float net_photosynthesis = 0;

            for( auto &phytomer: shoot->phytomers ){

                for( auto &leaf_objID: phytomer->leaf_objIDs ){
                    for( uint UUID : context_ptr->getObjectPrimitiveUUIDs(leaf_objID) ){
                        if( context_ptr->doesPrimitiveDataExist(UUID, "net_photosynthesis") && context_ptr->getPrimitiveDataType(UUID,"net_photosynthesis")==HELIOS_TYPE_FLOAT ){
                            float A;
                            context_ptr->getPrimitiveData(UUID,"net_photosynthesis",A);
                            net_photosynthesis += A*context_ptr->getPrimitiveArea(UUID)*dt;
                        }else{
                            A_prim_data_missing++;
                        }
                    }
                }

            }

            shoot->assimilate_pool += net_photosynthesis;

        }

    }

    if( A_prim_data_missing>0 ){
        std::cout << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): " << A_prim_data_missing << " leaf primitives were missing net_photosynthesis primitive data. Did you run the photosynthesis model?" << std::endl;
    }

}


std::vector<uint> makeTubeFromCones(uint Ndivs, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr) {

    uint Nverts = vertices.size();

    if( radii.size()!=Nverts || colors.size()!=Nverts ){
        helios_runtime_error("ERROR (makeTubeFromCones): Length of vertex vectors is not consistent.");
    }

    std::vector<uint> objIDs(Nverts-1);

    for( uint v=0; v<Nverts-1; v++ ){

        objIDs.at(v) = context_ptr->addConeObject(Ndivs, vertices.at(v), vertices.at(v + 1), radii.at(v), radii.at(v + 1), colors.at(v) );

    }

    return objIDs;

}
