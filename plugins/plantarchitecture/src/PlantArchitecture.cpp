/** \file "PlantArchitecture.cpp" Primary source file for plant architecture plug-in.
    \author Brian Bailey

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
#include "Assets.h"

using namespace helios;

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

}

PhytomerParameters::PhytomerParameters() {

    internode.origin = make_vec3(0,0,0);
    internode.pitch.initialize( 0.1*M_PI, &generator );
    internode.radius = 0.005;
    internode.color = RGB::forestgreen;
    internode.length.initialize(0.05,&generator);
    internode.tube_subdivisions = 1;
    internode.curvature.initialize( 0, &generator );
    internode.petioles_per_internode = 1;

    petiole.pitch.initialize( 0.5*M_PI, &generator );
    petiole.roll.initialize( 0, &generator );
    petiole.radius.initialize( 0.001, &generator );
    petiole.length.initialize( 0.05, &generator );
    petiole.curvature.initialize(0, &generator);
    petiole.taper.initialize( 0, &generator );
    petiole.tube_subdivisions = 1;
    petiole.leaves_per_petiole = 1;

    leaf.pitch.initialize( 0, &generator );
    leaf.yaw.initialize( 0, &generator );
    leaf.roll.initialize( 0, &generator );
    leaf.leaflet_offset.initialize( 0, &generator );
    leaf.leaflet_scale = 1;
    leaf.prototype_scale = make_vec3(0.05,0.025, 1.f);

    //--- inflorescence ---//
    inflorescence.reproductive_state = 0;
    inflorescence.length = 0.05;
    inflorescence.fruit_per_inflorescence = 1;
    inflorescence.fruit_offset = 0;
    inflorescence.rachis_radius = 0.001;
    inflorescence.curvature = 0;
    inflorescence.fruit_arrangement_pattern = "alternate";
    inflorescence.tube_subdivisions = 1;
    inflorescence.fruit_rotation = make_AxisRotation(0,0,0);
    inflorescence.fruit_prototype_scale = make_vec3(0.0075,0.0075,0.0075);

}

void PhytomerParameters::seedRandomGenerator(uint seed ){
    generator.seed(seed);
}

ShootParameters::ShootParameters() {
    max_nodes = 5;
    shoot_internode_taper = 0.5;
}

helios::vec3 Phytomer::getInternodeAxisVector(float stem_fraction) const{
    return getAxisVector( stem_fraction, internode_vertices );
}

helios::vec3 Phytomer::getPetioleAxisVector(float stem_fraction) const{
    return getAxisVector( stem_fraction, petiole_vertices );
}

helios::vec3 Phytomer::getAxisVector( float stem_fraction, const std::vector<helios::vec3> &axis_vertices ) const{

    if( stem_fraction<0 || stem_fraction>1 ){
        std::cerr << "ERROR: Stem fraction must be between 0 and 1." << std::endl;
    }

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

int Shoot::addPhytomer(PhytomerParameters &params, const AxisRotation &shoot_base_rotation) {

    vec3 parent_internode_axis;
    vec3 parent_petiole_axis;
    if( phytomers.empty() ) { //very first phytomer on shoot
        if( parentID==-1 ) { //very first shoot
            parent_internode_axis = make_vec3(0, 0, 1);
            parent_petiole_axis = make_vec3(0, 1, 0);
        }else{
            parent_internode_axis = shoot_tree_ptr->at(parentID).phytomers.at(parentNode).getInternodeAxisVector(1.f); //\todo Need checks to avoid index out of bounds
            parent_petiole_axis = shoot_tree_ptr->at(parentID).phytomers.at(parentNode).getPetioleAxisVector(0.f);
        }
    }else {
        parent_internode_axis = phytomers.back().getInternodeAxisVector(1.f);
        parent_petiole_axis = phytomers.back().getPetioleAxisVector(0.f);
    }

    Phytomer phytomer(params, phytomers.size(), parent_internode_axis, parent_petiole_axis, shoot_base_rotation, 1, rank, context_ptr);

    phytomers.emplace_back(phytomer);

    return (int)phytomers.size()-1;

}

Phytomer::Phytomer(const PhytomerParameters &params, uint phytomer_index, const helios::vec3 &parent_internode_axis,
                   const helios::vec3 &parent_petiole_axis, const AxisRotation &shoot_base_rotation, float scale,
                   uint rank,
                   helios::Context *context_ptr)
        : phytomer_parameters(params), context_ptr(context_ptr), rank(rank) {

    //Number of longitudinal segments for internode and petiole
    //if Ndiv=0, use Ndiv=1 (but don't add any primitives to Context)
    uint Ndiv_internode = std::max(uint(1), phytomer_parameters.internode.tube_subdivisions);
    uint Ndiv_petiole = std::max(uint(1), phytomer_parameters.petiole.tube_subdivisions);

    //Length of longitudinal segments
    float dr_internode = scale * phytomer_parameters.internode.length.val() / float(phytomer_parameters.internode.tube_subdivisions);
    float dr_petiole = scale * phytomer_parameters.petiole.length.val() / float(phytomer_parameters.petiole.tube_subdivisions);

    float internode_pitch = phytomer_parameters.internode.pitch.val();

    //Initialize segment vertices vector
    internode_vertices.resize(Ndiv_internode+1);
    internode_vertices.at(0) = phytomer_parameters.internode.origin;
    petiole_vertices.resize(Ndiv_petiole+1 );

    internode_radii.resize( Ndiv_internode+1 );
    internode_radii.at(0) = phytomer_parameters.internode.radius;
    petiole_radii.resize( Ndiv_petiole+1 );
    petiole_radii.at(0) = phytomer_parameters.petiole.radius.val();

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
            internode_axis = rotatePointAboutLine(internode_axis, phytomer_parameters.internode.origin, shoot_bending_axis,-shoot_base_rotation.pitch);
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, phytomer_parameters.internode.origin, shoot_bending_axis,-shoot_base_rotation.pitch);
        }

        //roll rotation for shoot base rotation
        if( shoot_base_rotation.roll!=0 ) {
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, phytomer_parameters.internode.origin, internode_axis,-shoot_base_rotation.roll);
         }

        //yaw rotation for shoot base rotation
        if( shoot_base_rotation.yaw!=0 ) {
            internode_axis = rotatePointAboutLine(internode_axis, phytomer_parameters.internode.origin, parent_internode_axis,shoot_base_rotation.yaw);
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, phytomer_parameters.internode.origin, parent_internode_axis,shoot_base_rotation.yaw);
            shoot_bending_axis = rotatePointAboutLine(shoot_bending_axis, phytomer_parameters.internode.origin, parent_internode_axis,shoot_base_rotation.yaw);
        }

        //pitch rotation for phytomer base
        if( internode_pitch!=0 ) {
            internode_axis = rotatePointAboutLine(internode_axis, phytomer_parameters.internode.origin, petiole_rotation_axis, -0.5f*internode_pitch );
        }

    }else {

        if( parent_internode_axis == make_vec3(0,0,1) ){
            shoot_bending_axis = make_vec3(1,0,0);
        }else {
            shoot_bending_axis = -1 * cross(parent_internode_axis, make_vec3(0, 0, 1));
        }

        //pitch rotation for phytomer base
        if ( internode_pitch != 0) {
            internode_axis = rotatePointAboutLine(internode_axis, phytomer_parameters.internode.origin, petiole_rotation_axis,-1.25f*internode_pitch );
        }

    }

    // create internode
    for( int i=1; i<=Ndiv_internode; i++ ){

        internode_axis = rotatePointAboutLine(internode_axis, internode_vertices.at(i - 1), shoot_bending_axis, -deg2rad(phytomer_parameters.internode.curvature.val() * dr_internode) );

        internode_vertices.at(i) = internode_vertices.at(i - 1) + dr_internode * internode_axis;

        internode_radii.at(i) = phytomer_parameters.internode.radius;
        internode_colors.at(i) = phytomer_parameters.internode.color;

    }

    uint objID = context_ptr->addTubeObject(10, internode_vertices, internode_radii, internode_colors );
    internode_UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);

    //--- create petiole ---//

    petiole_vertices.at(0) = internode_vertices.back(); //\todo shift this position outward by one internode radius

    vec3 petiole_axis = internode_axis;

    //petiole pitch rotation
    if( phytomer_parameters.petiole.pitch.val()!=0 ) {
        petiole_axis = rotatePointAboutLine(petiole_axis, petiole_vertices.at(0), petiole_rotation_axis, std::abs(phytomer_parameters.petiole.pitch.val()) );
    }

    //petiole yaw rotation
    if( phytomer_parameters.petiole.yaw.val()!=0 ) {
        petiole_axis = rotatePointAboutLine(petiole_axis, petiole_vertices.at(0), internode_axis, phytomer_parameters.petiole.yaw.val() );
        petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, petiole_vertices.at(0), internode_axis, phytomer_parameters.petiole.yaw.val() );
    }

    for(int bud=0; bud < phytomer_parameters.internode.petioles_per_internode; bud++ ) {

        if( bud>0 ) {
            float budrot = float(bud)*2.f*M_PI/float(phytomer_parameters.internode.petioles_per_internode);
            petiole_axis = rotatePointAboutLine(petiole_axis, petiole_vertices.at(0), internode_axis, budrot );
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, petiole_vertices.at(0), internode_axis, budrot );
        }

        for (int j = 1; j <= Ndiv_petiole; j++) {

            petiole_axis = rotatePointAboutLine(petiole_axis, petiole_vertices.at(j-1), petiole_rotation_axis, -deg2rad(phytomer_parameters.petiole.curvature.val()*dr_petiole) );

            petiole_vertices.at(j) = petiole_vertices.at(j - 1) + dr_petiole * petiole_axis;

            petiole_radii.at(j) = phytomer_parameters.petiole.radius.val()*( 1.f-phytomer_parameters.petiole.taper.val()/float(Ndiv_petiole-1)*float(j) );
            petiole_colors.at(j) = phytomer_parameters.internode.color;

        }

        objID = context_ptr->addTubeObject(10, petiole_vertices, petiole_radii, petiole_colors);
        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);
        petiole_UUIDs.insert( petiole_UUIDs.end(), UUIDs.begin(), UUIDs.end() );

        //--- create leaves ---//

        vec3 petiole_tip_axis = getPetioleAxisVector(1.f);

        vec3 leaf_rotation_axis = cross(internode_axis, petiole_tip_axis );

        for(int leaf=0; leaf < phytomer_parameters.petiole.leaves_per_petiole; leaf++ ){

            float ind_from_tip = leaf-float(phytomer_parameters.petiole.leaves_per_petiole-1)/2.f;

            uint objID_leaf = phytomer_parameters.leaf.prototype_function(context_ptr,1,(int)ind_from_tip);

            // -- scaling -- //

            vec3 leaf_scale = phytomer_parameters.leaf.prototype_scale;
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
            if( ind_from_tip==0 ){
                pitch_rot += asin_safe(petiole_tip_axis.z);
            }
            context_ptr->rotateObject(objID_leaf, -pitch_rot , "y" );

            //yaw rotation
            if( ind_from_tip!=0 ){
                float yaw_rot = -phytomer_parameters.leaf.yaw.val()*compound_rotation/fabs(compound_rotation);
                context_ptr->rotateObject( objID_leaf, yaw_rot, "z" );
            }

            //roll rotation
            if( ind_from_tip!= 0){
                float roll_rot = (asin_safe(petiole_tip_axis.z)+phytomer_parameters.leaf.roll.val())*compound_rotation/std::fabs(compound_rotation);
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

            UUIDs = context_ptr->getObjectPrimitiveUUIDs( objID_leaf );
            leaf_UUIDs.insert( leaf_UUIDs.end(), UUIDs.begin(), UUIDs.end() );

        }

        //--- create inflorescence ---//

        if( phytomer_parameters.inflorescence.reproductive_state!=0 ) {
            vec3 inflorescence_bending_axis;
            if( petiole_axis==make_vec3(0,0,1) ) {
                inflorescence_bending_axis = make_vec3(1, 0, 0);
            }else{
                inflorescence_bending_axis = cross(make_vec3(0, 0, 1), petiole_axis);
            }
            addInfluorescence(internode_vertices.back(), make_AxisRotation(0, 0, 0), inflorescence_bending_axis);
        }

    }

}

void Phytomer::addInfluorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation,
                                 const helios::vec3 &inflorescence_bending_axis) {

    float dr_rachis = phytomer_parameters.inflorescence.length/float(phytomer_parameters.inflorescence.tube_subdivisions);

    std::vector<vec3> rachis_vertices(phytomer_parameters.inflorescence.tube_subdivisions+1);
    rachis_vertices.at(0) = base_position;
    std::vector<float> rachis_radii(phytomer_parameters.inflorescence.tube_subdivisions+1);
    rachis_radii.at(0) = phytomer_parameters.inflorescence.rachis_radius;
    std::vector<RGBcolor> rachis_colors(phytomer_parameters.inflorescence.tube_subdivisions+1);
    rachis_colors.at(0) = phytomer_parameters.internode.color;

    vec3 rachis_axis = getAxisVector( 1.f, internode_vertices );

    float theta_base = fabs(cart2sphere(rachis_axis).zenith);

    for( int i=1; i<=phytomer_parameters.inflorescence.tube_subdivisions; i++ ){

        float theta_curvature = fabs(deg2rad(phytomer_parameters.inflorescence.curvature*dr_rachis));
         if( theta_curvature*float(i) < M_PI-theta_base ) {
            rachis_axis = rotatePointAboutLine(rachis_axis, rachis_vertices.at(i - 1), inflorescence_bending_axis, theta_curvature);
        }else{
            rachis_axis = make_vec3(0,0,-1);
        }

        rachis_vertices.at(i) = rachis_vertices.at(i - 1) + dr_rachis * rachis_axis;

        rachis_radii.at(i) = phytomer_parameters.inflorescence.rachis_radius;
        rachis_colors.at(i) = phytomer_parameters.internode.color;

    }

    uint objID = context_ptr->addTubeObject(10, rachis_vertices, rachis_radii, rachis_colors );
    inflorescence_UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);

    for(int fruit=0; fruit < phytomer_parameters.inflorescence.fruit_per_inflorescence; fruit++ ){

        uint objID_fruit;
        if( phytomer_parameters.inflorescence.reproductive_state==1 ){
            objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr,1,0);
        }else{
            objID_fruit = phytomer_parameters.inflorescence.fruit_prototype_function(context_ptr,1,0);
        }

        float ind_from_tip = fabs(fruit-float(phytomer_parameters.inflorescence.fruit_per_inflorescence-1)/2.f);

        vec3 fruit_scale = phytomer_parameters.inflorescence.fruit_prototype_scale;

        context_ptr->scaleObject( objID_fruit, fruit_scale );

        float compound_rotation = 0;
        if( phytomer_parameters.inflorescence.fruit_per_inflorescence>1 ) {
            if (phytomer_parameters.inflorescence.fruit_offset == 0) {
                float dphi = M_PI / (floor(0.5 * float(phytomer_parameters.inflorescence.fruit_per_inflorescence - 1)) + 1);
                compound_rotation = -float(M_PI) + dphi * (fruit + 0.5f);
            } else {
                if( fruit == float(phytomer_parameters.inflorescence.fruit_per_inflorescence-1)/2.f ){ //tip leaf
                    compound_rotation = 0;
                }else if( fruit < float(phytomer_parameters.inflorescence.fruit_per_inflorescence-1)/2.f ) {
                    compound_rotation = -0.5*M_PI;
                }else{
                    compound_rotation = 0.5*M_PI;
                }
            }
        }

        //\todo Once rachis curvature is added, rachis_axis needs to become rachis_tip_axis
        context_ptr->rotateObject( objID_fruit, -std::atan2(rachis_axis.y, rachis_axis.x)+compound_rotation, "z" );

        //pitch rotation
//        context_ptr->rotateObject(objID_fruit, -asin_safe(rachis_axis.z) - phytomer_parameters.leaf_rotation.pitch, make_vec3(0,0,0), rachis_axis );

//        //roll rotation
//        context_ptr->rotateObject(objID_leaf, phytomer_parameters.leaf_rotation.roll, make_vec3(0,0,0), petiole_tip_axis );

        vec3 fruit_base = rachis_vertices.back();
        if( phytomer_parameters.inflorescence.fruit_per_inflorescence>1 && phytomer_parameters.inflorescence.fruit_offset>0 ){
            if( ind_from_tip != 0 ) {
                float offset;
                if( phytomer_parameters.inflorescence.fruit_arrangement_pattern == "opposite" ){
                    offset = (ind_from_tip - 0.5f) * phytomer_parameters.inflorescence.fruit_offset * phytomer_parameters.inflorescence.length;
                }else if( phytomer_parameters.inflorescence.fruit_arrangement_pattern == "alternate" ){
                    offset = (ind_from_tip - 0.5f + 0.5f*float(fruit>float(phytomer_parameters.inflorescence.fruit_per_inflorescence-1)/2.f) ) * phytomer_parameters.inflorescence.fruit_offset * phytomer_parameters.inflorescence.length;
                }else{
                    throw(std::runtime_error("ERROR (PlantArchitecture::addInfluorescence): Invalid fruit arrangement pattern."));
                }
                fruit_base = interpolateTube(rachis_vertices, 1.f - offset / phytomer_parameters.inflorescence.length);
            }
        }

        context_ptr->translateObject( objID_fruit, fruit_base );

        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs( objID_fruit );
        inflorescence_UUIDs.insert( inflorescence_UUIDs.end(), UUIDs.begin(), UUIDs.end() );

    }

    context_ptr->setPrimitiveData( inflorescence_UUIDs, "rank", rank );

}

uint PlantArchitecture::addShoot(int parentID, uint parent_node, uint rank, uint current_node_number,
                                 const helios::vec3 &base_position, const AxisRotation &base_rotation, const ShootParameters &shoot_params) {

    int shootID = (int)shoot_tree.size();

    if( parentID<-1 || parentID>=int(shoot_tree.size()) ){
        throw( std::runtime_error("ERROR (PlantArchitecture::addShoot): Parent with ID of " + std::to_string(parentID) + " does not exist.") );
    }else if( current_node_number>shoot_params.max_nodes ){
        throw( std::runtime_error("ERROR (PlantArchitecture::addShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_params.max_nodes) + ".") );
    }

    Shoot shoot(shootID, parentID, parent_node, rank, base_position, base_rotation, current_node_number, phytomer_parameters_current,  shoot_params, &shoot_tree, context_ptr);

    shoot_tree.emplace_back(shoot);

    return shootID;

}

Shoot::Shoot(int ID, int parentID, uint parent_node, uint rank, const helios::vec3 &origin,
             const AxisRotation &shoot_base_rotation, uint current_node_number, const PhytomerParameters &phytomer_params,
             const ShootParameters &shoot_params, std::vector<Shoot> *shoot_tree_ptr, helios::Context *context_ptr) :
        ID(ID), parentID(parentID), parentNode(parent_node), rank(rank), origin(origin), base_rotation(shoot_base_rotation), current_node_number(current_node_number), shoot_tree_ptr(shoot_tree_ptr), context_ptr(context_ptr)
{

    PhytomerParameters phytomer_parameters = phytomer_params;

    std::vector<uint> phytomer_UUIDs;

    for( int i=0; i<current_node_number; i++ ) {

        if( i==0 ){ //first phytomer on shoot
            phytomer_parameters.internode.origin = origin;
            phytomer_parameters.petiole.roll = 0;
        }

        phytomer_parameters.internode.radius = phytomer_params.internode.radius*(1.f-shoot_params.shoot_internode_taper*float(i)/float(current_node_number) );

        int pID = addPhytomer(phytomer_parameters, shoot_base_rotation);

        phytomer_parameters.internode.origin = phytomers.at(pID).internode_vertices.back();

        phytomer_UUIDs.insert( phytomer_UUIDs.end(), phytomers.at(pID).internode_UUIDs.begin(), phytomers.at(pID).internode_UUIDs.end() );
        phytomer_UUIDs.insert( phytomer_UUIDs.end(), phytomers.at(pID).petiole_UUIDs.begin(), phytomers.at(pID).petiole_UUIDs.end() );
        phytomer_UUIDs.insert( phytomer_UUIDs.end(), phytomers.at(pID).leaf_UUIDs.begin(), phytomers.at(pID).leaf_UUIDs.end() );

    }

    context_ptr->setPrimitiveData( phytomer_UUIDs, "rank", rank );

    if( parentID!=-1 ) {
        shoot_tree_ptr->at(parentID).childIDs.push_back(ID);
    }

//    vec3 shoot_bending_axis = make_vec3(-1,0,0);
//
//    std::vector<uint> phytomer_objIDs = context_ptr->getUniquePrimitiveParentObjectIDs( phytomer_UUIDs );

    //pitch rotation for shoot base rotation
//    if( shoot_base_rotation.pitch!=0 ) {
//        context_ptr->rotateObject( phytomer_objIDs, -shoot_base_rotation.pitch, origin, shoot_bending_axis );
//    }
//
//    vec3 parent_internode_axis;
//    if( parentID==-1 ) { //very first shoot
//        parent_internode_axis = make_vec3(0, 0, 1);
//    }else{
//        parent_internode_axis = shoot_tree_ptr->at(parentID).phytomers.at(parentNode).getInternodeAxisVector(1.f); //\todo Need checks to avoid index out of bounds
//    }
//
//    //yaw rotation for shoot base rotation
//    if( shoot_base_rotation.yaw!=0 ) {
//        context_ptr->rotateObject( phytomer_objIDs, shoot_base_rotation.yaw, origin, parent_internode_axis );
//    }

}

uint PlantArchitecture::addChildShoot(int parentID, uint parent_node, uint current_node_number,
                                      const AxisRotation &base_rotation, const ShootParameters &shoot_params) {

    if( parentID<-1 || parentID>=shoot_tree.size() ){
        throw( std::runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent with ID of " + std::to_string(parentID) + " does not exist.") );
    }

    int parent_rank;
    if( parentID==-1 ){
        parent_rank = -1;
    }else{
        parent_rank = (int)shoot_tree.at(parentID).rank;
    }

    vec3 node_position;

    if( parentID>-1 ){
        std::vector<Phytomer> *shoot_phytomers = &shoot_tree.at(parentID).phytomers;

        if( parent_node>=shoot_phytomers->size() ){
            throw( std::runtime_error("ERROR (PlantArchitecture::addChildShoot): Requested to place child shoot on node " + std::to_string(parent_node) + " but parent only has " + std::to_string(shoot_phytomers->size()) + " nodes." ) );
        }

        node_position = shoot_phytomers->at(parent_node).internode_vertices.back();

    }

    uint childID = addShoot(parentID, parent_node, parent_rank + 1, current_node_number, node_position, base_rotation, shoot_params);

    shoot_tree.at(parentID).childIDs.push_back(childID);

    return childID;

}

void PlantArchitecture::setCurrentPhytomerParameters( const PhytomerParameters &phytomer_parameters_new ){
    phytomer_parameters_current = phytomer_parameters_new;
}

void PlantArchitecture::setCurrentPhytomerParameters( const std::string &phytomer_label ){
    
    if( phytomer_label=="bean" ){

        phytomer_parameters_current.internode.pitch = 0.1 * M_PI; //pitch>0 creates zig-zagging
        phytomer_parameters_current.internode.radius = 0.0025;
        phytomer_parameters_current.internode.length = 0.015;
        phytomer_parameters_current.internode.curvature = -100;
        phytomer_parameters_current.internode.petioles_per_internode = 1;
        phytomer_parameters_current.internode.color = make_RGBcolor(0.38, 0.48, 0.1);
        phytomer_parameters_current.internode.tube_subdivisions = 5;

        phytomer_parameters_current.petiole.pitch = 0.25 * M_PI;
        phytomer_parameters_current.petiole.yaw = M_PI;
        phytomer_parameters_current.petiole.roll = 0;
        phytomer_parameters_current.petiole.radius = 0.0015;
        phytomer_parameters_current.petiole.length = 0.03;
        phytomer_parameters_current.petiole.taper = 0.1;
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

        phytomer_parameters_current.inflorescence.reproductive_state = 0;
        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = BeanFruitPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.05 * make_vec3(1, 1, 1);
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

        phytomer_parameters_current.inflorescence.reproductive_state = 0;
        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = CowpeaFruitPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.1 * make_vec3(1, 1, 1);
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

        phytomer_parameters_current.inflorescence.reproductive_state = 1;
        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = TomatoFruitPrototype;
        phytomer_parameters_current.inflorescence.flower_prototype_function = TomatoFlowerPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.015 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.fruit_arrangement_pattern = "alternate";
        phytomer_parameters_current.inflorescence.fruit_per_inflorescence = 9;
        phytomer_parameters_current.inflorescence.fruit_offset = 0.14;
        phytomer_parameters_current.inflorescence.curvature = -300;
        phytomer_parameters_current.inflorescence.length = 0.075;
        phytomer_parameters_current.inflorescence.rachis_radius = 0.0005;

    }else if( phytomer_label=="almond" ){

        phytomer_parameters_current.internode.pitch = 0.05 * M_PI;
        phytomer_parameters_current.internode.tube_subdivisions = 5;
        phytomer_parameters_current.internode.curvature = -100;
        phytomer_parameters_current.internode.radius = 0.0025;
        phytomer_parameters_current.internode.length.uniformDistribution(0.01,0.015);
        phytomer_parameters_current.internode.color = make_RGBcolor(0.42,0.27,0.09);
        phytomer_parameters_current.internode.petioles_per_internode = 1;

        phytomer_parameters_current.petiole.pitch =-0.5 * M_PI;
        phytomer_parameters_current.petiole.roll = 0.5 * M_PI;
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

        phytomer_parameters_current.inflorescence.reproductive_state = 1;
        phytomer_parameters_current.inflorescence.tube_subdivisions = 10;
        phytomer_parameters_current.inflorescence.fruit_prototype_function = AlmondFruitPrototype;
        phytomer_parameters_current.inflorescence.flower_prototype_function = AlmondFlowerPrototype;
        phytomer_parameters_current.inflorescence.fruit_prototype_scale = 0.01 * make_vec3(1, 1, 1);
        phytomer_parameters_current.inflorescence.fruit_arrangement_pattern = "alternate";
        phytomer_parameters_current.inflorescence.fruit_per_inflorescence = 1;
        phytomer_parameters_current.inflorescence.fruit_offset = 0.14;
        phytomer_parameters_current.inflorescence.curvature = -300;
        phytomer_parameters_current.inflorescence.length = 0.0;
        phytomer_parameters_current.inflorescence.rachis_radius = 0.0005;
        phytomer_parameters_current.inflorescence.fruit_rotation = make_AxisRotation(-0.5*M_PI,0,0);
        
    }else{
        throw( std::runtime_error("ERROR (PlantArchitecture::setCurrentPhytomerParameters): " + phytomer_label + " is not a valid phytomer in the library.") );
    }
    
}
