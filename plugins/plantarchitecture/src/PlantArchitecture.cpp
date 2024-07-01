/** \file "PlantArchitecture.cpp" Primary source file for plant architecture plug-in.

    Copyright (C) 2016-2024 Brian Bailey

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

    //--- internode ---//
    internode.pitch.initialize( 20, generator );
    internode.phyllotactic_angle.initialize(137.5, generator );
    internode.color = RGB::forestgreen;
    internode.length_segments = 1;
    internode.radial_subdivisions = 7;

    //--- petiole ---//
    petiole.petioles_per_internode = 1;
    petiole.pitch.initialize( 90, generator );
    petiole.radius.initialize( 0.001, generator );
    petiole.length.initialize( 0.05, generator );
    petiole.curvature.initialize(0, generator);
    petiole.taper.initialize( 0, generator );
    petiole.color = RGB::forestgreen;
    petiole.length_segments = 1;
    petiole.radial_subdivisions = 7;

    //--- leaf ---//
    leaf.leaves_per_petiole = 1;
    leaf.pitch.initialize( 0, generator );
    leaf.yaw.initialize( 0, generator );
    leaf.roll.initialize( 0, generator );
    leaf.leaflet_offset.initialize( 0, generator );
    leaf.leaflet_scale = 1;
    leaf.prototype_scale.initialize(0.05,generator);
    leaf.subdivisions = 1;

    //--- peduncle ---//
    peduncle.length.initialize(0.05,generator);
    peduncle.radius.initialize(0.001, generator);
    peduncle.pitch.initialize(0,generator);
    peduncle.roll.initialize(0,generator);
    peduncle.curvature.initialize(0,generator);
    peduncle.length_segments = 3;
    peduncle.radial_subdivisions = 7;

    //--- inflorescence ---//
    inflorescence.flowers_per_rachis.initialize(1, generator);
    inflorescence.flower_offset.initialize(0, generator);
    inflorescence.flower_arrangement_pattern = "alternate";
    inflorescence.pitch.initialize(0,generator);
    inflorescence.roll.initialize(0,generator);
    inflorescence.flower_prototype_scale.initialize(0.0075,generator);
    inflorescence.fruit_prototype_scale.initialize(0.0075,generator);
    inflorescence.fruit_gravity_factor_fraction.initialize(0, generator);

}

ShootParameters::ShootParameters() : ShootParameters(nullptr) {}

ShootParameters::ShootParameters( std::minstd_rand0 *generator ) {

    // ---- Geometric Parameters ---- //

    max_nodes.initialize( 10, generator );

    internode_radius_initial.initialize(0.001,generator);
    internode_radius_max.initialize(1e6, generator);

    child_insertion_angle_tip.initialize(20, generator);
    child_insertion_angle_decay_rate.initialize(0, generator);

    child_internode_length_max.initialize(0.02, generator);
    child_internode_length_min.initialize(0.002, generator);
    child_internode_length_decay_rate.initialize(0, generator);

    base_roll.initialize(0, generator);
    base_yaw.initialize(0, generator);

    gravitropic_curvature.initialize(0, generator );
    tortuosity.initialize(0, generator );

    // ---- Growth Parameters ---- //

    phyllochron.initialize(1, generator);
    leaf_flush_count = 1;

    elongation_rate.initialize(0.2, generator);
    girth_growth_rate.initialize(0, generator);

    vegetative_bud_break_time.initialize(5, generator);
    vegetative_bud_break_probability.initialize(0,generator);
    flower_bud_break_probability.initialize(0,generator);
    fruit_set_probability.initialize(0,generator);


    flowers_require_dormancy = false;
    growth_requires_dormancy = false;

    determinate_shoot_growth = true;

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
    if( shoot_types.find(shoot_type_label)!=shoot_types.end() ){
        //std::cerr <<"WARNING (PlantArchitecture::defineShootType): Shoot type label of " << shoot_type_label << " already exists." << std::endl;
        shoot_types.at(shoot_type_label) = shoot_params;
    }else {
        shoot_types.emplace(shoot_type_label, shoot_params);
    }
}

helios::vec3 Phytomer::getInternodeAxisVector(float stem_fraction) const{
    return getAxisVector( stem_fraction, internode_vertices );
}

helios::vec3 Phytomer::getPetioleAxisVector(float stem_fraction, uint petiole_index) const {
    if( petiole_index>=petiole_vertices.size() ){
        helios_runtime_error("ERROR (Phytomer::getPetioleAxisVector): Petiole index out of range.");
    }
    return getAxisVector( stem_fraction, petiole_vertices.at(petiole_index) );
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

void Phytomer::setVegetativeBudState( BudState state ){
    for( auto& petiole : vegetative_buds ){
        for( auto& bud : petiole ) {
            setVegetativeBudState(state, bud);
        }
    }
}

void Phytomer::setVegetativeBudState(BudState state, uint petiole_index, uint bud_index) {
    if( petiole_index>=vegetative_buds.size() ){
        helios_runtime_error("ERROR (Phytomer::setVegetativeBudState): Petiole index out of range.");
    }
    if( bud_index>=vegetative_buds.at(petiole_index).size() ){
        helios_runtime_error("ERROR (Phytomer::setVegetativeBudState): Bud index out of range.");
    }
    setVegetativeBudState( state, vegetative_buds.at(petiole_index).at(bud_index) );
}

void Phytomer::setVegetativeBudState( BudState state, VegetativeBud &vbud ){
    vbud.state = state;
}

void Phytomer::setFloralBudState( BudState state ) {
    for( auto &petiole :floral_buds ) {
        for ( auto &fbud : petiole ) {
            setFloralBudState(state, fbud);
        }
    }
}

void Phytomer::setFloralBudState(BudState state, uint petiole_index, uint bud_index) {
    if (petiole_index >= floral_buds.size()) {
        helios_runtime_error("ERROR (Phytomer::setFloralBudState): Petiole index out of range.");
    }
    if (bud_index >= floral_buds.at(petiole_index).size()) {
        helios_runtime_error("ERROR (Phytomer::setFloralBudState): Bud index out of range.");
    }
    setFloralBudState(state, floral_buds.at(petiole_index).at(bud_index));
}

void Phytomer::setFloralBudState(BudState a_state, FloralBud &fbud ) {

    // If state is already at the desired state, do nothing
    if (fbud.state == a_state) {
        return;
    } else if (a_state == BUD_DORMANT || a_state == BUD_ACTIVE ) {
        fbud.state = a_state;
        return;
    }

    // Delete geometry from previous reproductive state (if present)
    context_ptr->deleteObject(inflorescence_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index));
    inflorescence_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index).resize(0);
    inflorescence_bases.at(fbud.parent_petiole_index).at(fbud.bud_index).resize(0);

    if( build_context_geometry_peduncle ) {
        context_ptr->deleteObject(peduncle_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index));
        peduncle_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index).resize(0);
    }

    fbud.state = a_state;

    if( a_state != BUD_DEAD ) { //add new reproductive organs

        float pitch_adjustment;
        float yaw_adjustment;
        if (vegetative_buds.empty()) {
            pitch_adjustment = 0;
            yaw_adjustment = 0;
        } else {
            pitch_adjustment = fbud.bud_index * 0.1f * M_PI / float(vegetative_buds.size());
            yaw_adjustment = -0.25f * M_PI + fbud.bud_index * 0.5f * M_PI / float(vegetative_buds.size());
        }
        addInflorescence(internode_vertices.back(), make_AxisRotation(pitch_adjustment, yaw_adjustment, 0), make_vec3(0, 0, 1), fbud);
        fbud.time_counter = 0;
        if (fbud.state == BUD_FRUITING) {
            setInflorescenceScaleFraction(fbud, 0.1);
        }

    }


}

int Shoot::addPhytomer(const PhytomerParameters &params, const helios::vec3 internode_base_position, const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction,
                       float leaf_scale_factor_fraction) {

    auto shoot_tree_ptr = &plant_architecture_ptr->plant_instances.at(plantID).shoot_tree;

    //Determine the parent internode and petiole axes for rotation of the new phytomer
    vec3 parent_internode_axis;
    vec3 parent_petiole_axis;
    if( phytomers.empty() ) { //very first phytomer on shoot
        if(parent_shoot_ID == -1 ) { //very first shoot of the plant
            parent_internode_axis = make_vec3(0, 0, 1);
            parent_petiole_axis = make_vec3(0, -1, 0);
        }else{ //first phytomer of a new shoot
            assert(parent_shoot_ID < shoot_tree_ptr->size() && parent_node_index < shoot_tree_ptr->at(parent_shoot_ID)->phytomers.size() );
            parent_internode_axis = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->getInternodeAxisVector(1.f);
            parent_petiole_axis = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->getPetioleAxisVector(0.f, parent_petiole_index);
        }
    }else{ //additional phytomer being added to an existing shoot
        parent_internode_axis = phytomers.back()->getInternodeAxisVector(1.f);
        parent_petiole_axis = phytomers.back()->getPetioleAxisVector(0.f, 0);
    }

    std::shared_ptr<Phytomer> phytomer = std::make_shared<Phytomer>(params, this, phytomers.size(), parent_internode_axis, parent_petiole_axis, internode_base_position, shoot_base_rotation, internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, rank, plant_architecture_ptr->build_context_geometry_internode, plant_architecture_ptr->build_context_geometry_petiole, plant_architecture_ptr->build_context_geometry_peduncle, context_ptr);

    //Initialize phytomer vegetative bud types and state
    for( auto& petiole : phytomer->vegetative_buds ) {
        for (auto &vbud: petiole) {

            //sample the bud shoot type and initialize its state
            std::string child_shoot_type_label;
            if (sampleChildShootType(child_shoot_type_label)) {
                phytomer->setVegetativeBudState( BUD_DORMANT, vbud );
                vbud.shoot_type_label = child_shoot_type_label;
            } else {
                phytomer->setVegetativeBudState( BUD_DEAD, vbud );
            }

            // if the shoot type does not require dormancy, bud should be set to active
            if (!shoot_parameters.growth_requires_dormancy && vbud.state != BUD_DEAD) {
                phytomer->setVegetativeBudState( BUD_ACTIVE, vbud );
            }

        }
    }

    //Initialize phytomer floral bud types and state
    uint petiole_index = 0;
    for( auto& petiole : phytomer->floral_buds ) {
        uint bud_index = 0;
        for (auto &fbud: petiole) {

            //Set state of phytomer buds
            phytomer->setFloralBudState(BUD_DORMANT, fbud);

            // if the shoot type does not require dormancy, bud should be set to active
            if (!shoot_parameters.flowers_require_dormancy && fbud.state != BUD_DEAD) {
                phytomer->setFloralBudState(BUD_ACTIVE, fbud);
            }

            fbud.parent_petiole_index = petiole_index;
            fbud.bud_index = bud_index;

            bud_index++;
        }
        petiole_index++;
    }

    shoot_tree_ptr->at(ID)->phytomers.push_back(phytomer);

    //Set output object data 'age'
    phytomer->age = 0;
    if( phytomer->build_context_geometry_internode ) {
        context_ptr->setObjectData(phytomer->internode_objIDs, "age", phytomer->age);
        context_ptr->setObjectData(phytomer->internode_objIDs, "rank", rank);
        context_ptr->setObjectData( phytomer->internode_objIDs, "plantID", (int)plantID );
    }
    if( phytomer->build_context_geometry_petiole ) {
        context_ptr->setObjectData(phytomer->petiole_objIDs, "age", phytomer->age);
        context_ptr->setObjectData(phytomer->petiole_objIDs, "rank", phytomer->rank);
        context_ptr->setObjectData( phytomer->petiole_objIDs, "plantID", (int)plantID );
    }
    context_ptr->setObjectData(phytomer->leaf_objIDs, "age", phytomer->age);
    context_ptr->setObjectData(phytomer->leaf_objIDs, "rank", phytomer->rank);
    context_ptr->setObjectData( phytomer->leaf_objIDs, "plantID", (int)plantID );

    for( auto &petiole : phytomer->leaf_objIDs ){
        for( uint objID : petiole ){
            context_ptr->setObjectData( objID, "leafID", (int)objID );
        }
    }

    if( phytomer->phytomer_parameters.phytomer_creation_function != nullptr ) {
        phytomer->phytomer_parameters.phytomer_creation_function(phytomer, current_node_number, this->parent_node_index, shoot_parameters.max_nodes.val(), rank, plant_architecture_ptr->plant_instances.at(plantID).current_age);
    }

    return (int)phytomers.size()-1;

}

void Shoot::breakDormancy(){

    dormant = false;

    int phytomer_ind = 0;
    for( auto &phytomer : phytomers ) {

        bool is_terminal_phytomer = (phytomer_ind == current_node_number - 1);
        for( auto& petiole : phytomer->floral_buds ) {
            for (auto &fbud: petiole) {
                if (fbud.state != BUD_DEAD) {
                    phytomer->setFloralBudState( BUD_ACTIVE, fbud );
                }
                if (meristem_is_alive && is_terminal_phytomer) {
                    phytomer->setFloralBudState( BUD_ACTIVE, fbud );
                }
                fbud.time_counter = 0;
            }
        }
        for( auto& petiole : phytomer->vegetative_buds ) {
            for (auto &vbud: petiole) {
                if (vbud.state != BUD_DEAD) {
                    phytomer->setVegetativeBudState(BUD_ACTIVE, vbud);
                }
            }
        }

        phytomer_ind++;
    }

}

void Shoot::makeDormant(){

    dormant = true;
    dormancy_cycles++;

    for( auto &phytomer : phytomers ){
        for( auto& petiole : phytomer->floral_buds ) {
            //all currently active lateral buds die at dormancy
            for (auto &fbud: petiole) {
                if (fbud.state != BUD_DORMANT) {
                    phytomer->setFloralBudState( BUD_DEAD, fbud );
                }
            }
        }
        for( auto& petiole : phytomer->vegetative_buds ) {
            for (auto &vbud: petiole) {
                if (vbud.state != BUD_DORMANT) {
                    phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                }
            }
        }
        phytomer->removeLeaf();
        phytomer->time_since_dormancy = 0;
    }

}

void Shoot::terminateApicalBud(){
    this->meristem_is_alive = false;
}

void Shoot::terminateAxillaryVegetativeBuds() {

    for( auto &phytomer : phytomers ){
        for( auto& petiole : phytomer->vegetative_buds ) {
            for (auto &vbud: petiole) {
                phytomer->setVegetativeBudState( BUD_DEAD, vbud );
            }
        }
    }

}

Phytomer::Phytomer(const PhytomerParameters &params, Shoot *parent_shoot, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, helios::vec3 internode_base_origin,
                   const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, bool build_context_geometry_internode, bool build_context_geometry_petiole,
                   bool build_context_geometry_peduncle, helios::Context *context_ptr)
        : context_ptr(context_ptr), rank(rank), build_context_geometry_internode(build_context_geometry_internode), build_context_geometry_petiole(build_context_geometry_petiole), build_context_geometry_peduncle(build_context_geometry_peduncle){

    phytomer_parameters = params; //note this needs to be an assignment operation not a copy in order to re-randomize all the parameters

    ShootParameters parent_shoot_parameters = parent_shoot->shoot_parameters;

    this->internode_radius_initial = internode_radius;
    this->internode_radius_max = parent_shoot_parameters.internode_radius_max.val();
    this->internode_length_max = internode_length_max;
    this->shoot_index = make_int2(phytomer_index, parent_shoot_parameters.max_nodes.val()); //.x is the index of the phytomer along the shoot, .y is the maximum number of phytomers on the parent shoot.
    this->rank = parent_shoot->rank;

//    if( internode_radius==0.f || internode_length_max==0.f || parent_shoot_parameters.internode_radius_max.val()==0.f ){
//        build_context_geometry_internode = false;
//    }

    //Number of longitudinal segments for internode and petiole
    //if Ndiv=0, use Ndiv=1 (but don't add any primitives to Context)
    uint Ndiv_internode_length = std::max(uint(1), phytomer_parameters.internode.length_segments);
    uint Ndiv_internode_radius = std::max(uint(3), phytomer_parameters.internode.radial_subdivisions);
    uint Ndiv_petiole_length = std::max(uint(1), phytomer_parameters.petiole.length_segments);
    uint Ndiv_petiole_radius = std::max(uint(3), phytomer_parameters.petiole.radial_subdivisions);

    //Flags to determine whether internode geometry should be built in the Context. Not building all geometry can save memory and computation time.
    if( phytomer_parameters.internode.length_segments==0 || phytomer_parameters.internode.radial_subdivisions<3 ){
        build_context_geometry_internode = false;
    }
    if( phytomer_parameters.petiole.length_segments==0 || phytomer_parameters.petiole.radial_subdivisions<3 ){
        build_context_geometry_petiole = false;
    }

    if( phytomer_parameters.petiole.petioles_per_internode<1 ){
        build_context_geometry_petiole = false;
        phytomer_parameters.petiole.petioles_per_internode = 1;
        phytomer_parameters.leaf.leaves_per_petiole = 0;
    }

    if( phytomer_parameters.petiole.petioles_per_internode==0 ){
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Number of petioles per internode must be greater than zero.");
    }

    current_internode_scale_factor = internode_length_scale_factor_fraction;
    current_leaf_scale_factor = leaf_scale_factor_fraction;

    //Initialize internode variables
    internode_length = internode_length_scale_factor_fraction * internode_length_max;
    float dr_internode = internode_length / float(phytomer_parameters.internode.length_segments);
    float dr_internode_max = internode_length_max / float(phytomer_parameters.internode.length_segments);
    internode_vertices.resize(Ndiv_internode_length + 1);
    internode_vertices.at(0) = internode_base_origin;
    internode_radii.resize(Ndiv_internode_length + 1 );
    internode_radii.at(0) = internode_radius;
    internode_pitch = deg2rad(phytomer_parameters.internode.pitch.val());
    phytomer_parameters.internode.pitch.resample();
    internode_phyllotactic_angle = deg2rad(phytomer_parameters.internode.phyllotactic_angle.val());
    phytomer_parameters.internode.phyllotactic_angle.resample();

    //initialize petiole variables
    petiole_length.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_vertices.resize( phytomer_parameters.petiole.petioles_per_internode );
    petiole_radii.resize( phytomer_parameters.petiole.petioles_per_internode );
    std::vector<float> dr_petiole(phytomer_parameters.petiole.petioles_per_internode);
    std::vector<float> dr_petiole_max(phytomer_parameters.petiole.petioles_per_internode);
    for( int p=0; p<phytomer_parameters.petiole.petioles_per_internode; p++ ) {

        petiole_vertices.at(p).resize(Ndiv_petiole_length + 1);
        petiole_radii.at(p).resize(Ndiv_petiole_length + 1);

        petiole_length.at(p) = leaf_scale_factor_fraction * phytomer_parameters.petiole.length.val();
        dr_petiole.at(p) = petiole_length.at(p) / float(phytomer_parameters.petiole.length_segments);
        dr_petiole_max.at(p) = phytomer_parameters.petiole.length.val() / float(phytomer_parameters.petiole.length_segments);
        phytomer_parameters.petiole.length.resample();

        petiole_radii.at(p).at(0) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val();

    }
    if( build_context_geometry_petiole ) {
        petiole_objIDs.resize(phytomer_parameters.petiole.petioles_per_internode);
    }

    //initialize leaf variables
    leaf_bases.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_objIDs.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_size_max.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_rotation.resize(phytomer_parameters.petiole.petioles_per_internode);
    for( uint petiole=0; petiole<phytomer_parameters.petiole.petioles_per_internode; petiole++ ) {
        leaf_rotation.at(petiole).resize(phytomer_parameters.leaf.leaves_per_petiole);
    }

    internode_colors.resize(Ndiv_internode_length + 1 );
    internode_colors.at(0) = phytomer_parameters.internode.color;
    petiole_colors.resize(Ndiv_petiole_length + 1 );
    petiole_colors.at(0) = phytomer_parameters.petiole.color;

    vec3 internode_axis = parent_internode_axis;

    vec3 petiole_rotation_axis = cross(parent_internode_axis, parent_petiole_axis );
    if(petiole_rotation_axis == make_vec3(0, 0, 0) ){
        petiole_rotation_axis = make_vec3(1, 0, 0);
    }

    if( phytomer_index==0 ){ //if this is the first phytomer along a shoot, apply the origin rotation about the parent axis)

         //pitch rotation for phytomer base
        if( internode_pitch!=0.f ) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, 0.5f*internode_pitch );
        }

        //roll rotation for shoot base rotation
        float roll_nudge = 0.f;
        if( shoot_base_rotation.roll/180.f == floor(shoot_base_rotation.roll/180.f) ) {
            roll_nudge = 0.2;
        }
        if( shoot_base_rotation.roll!=0.f || roll_nudge!=0.f ){
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis,shoot_base_rotation.roll + roll_nudge ); //small additional rotation is to make sure the petiole is not exactly vertical
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis,shoot_base_rotation.roll + roll_nudge );
        }

        vec3 base_pitch_axis = -1*cross(parent_internode_axis, parent_petiole_axis );

        //pitch rotation for shoot base rotation
        if( shoot_base_rotation.pitch!=0.f ) {
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
        }

        //yaw rotation for shoot base rotation
        if( shoot_base_rotation.yaw!=0 ){
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw );
        }

    }else {

        //pitch rotation for phytomer base
        if ( internode_pitch != 0) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis,-1.25f*internode_pitch );
        }

    }

    vec3 shoot_bending_axis = cross( internode_axis, make_vec3(0,0,1) );

    internode_axis.normalize();
    if( internode_axis==make_vec3(0,0,1) ){
        shoot_bending_axis = make_vec3(0,1,0);
    }

    // create internode
    for(int i=1; i <= Ndiv_internode_length; i++ ){

        //apply curvature
        if( fabs(parent_shoot_parameters.gravitropic_curvature.val()) > 0 ) {
            float dt = 1.f / float(Ndiv_internode_length);
            parent_shoot->curvature_perturbation += - 0.5f*parent_shoot->curvature_perturbation*dt + 5*parent_shoot_parameters.tortuosity.val()*context_ptr->randn()*sqrt(dt);
            float curvature_angle = deg2rad((parent_shoot_parameters.gravitropic_curvature.val()+parent_shoot->curvature_perturbation) * dr_internode_max);
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, shoot_bending_axis, curvature_angle);
        }

        internode_vertices.at(i) = internode_vertices.at(i - 1) + dr_internode * internode_axis;

        internode_radii.at(i) = internode_radius;
        internode_colors.at(i) = phytomer_parameters.internode.color;

    }

    if( build_context_geometry_internode ) {
        internode_objIDs = makeTubeFromCones(Ndiv_internode_radius, internode_vertices, internode_radii, internode_colors, context_ptr);
    }

    //--- create petiole ---//

    vec3 petiole_axis = internode_axis;

    //petiole pitch rotation
    petiole_pitch = deg2rad(phytomer_parameters.petiole.pitch.val());
    phytomer_parameters.petiole.pitch.resample();
    if( fabs(petiole_pitch)<deg2rad(5.f) ) {
        petiole_pitch = deg2rad(5.f);
    }
    petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, petiole_rotation_axis, std::abs(petiole_pitch) );

    //petiole yaw rotation
    if( phytomer_index!=0 && internode_phyllotactic_angle!=0 ){ //not first phytomer along shoot
        petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, internode_axis, internode_phyllotactic_angle );
        petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis, internode_phyllotactic_angle );
    }

    for(int petiole=0; petiole < phytomer_parameters.petiole.petioles_per_internode; petiole++ ) { //looping over petioles

        if( petiole > 0 ) {
            float budrot = float(petiole) * 2.f * M_PI / float(phytomer_parameters.petiole.petioles_per_internode);
            petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, internode_axis, budrot );
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis, budrot );
        }

        petiole_vertices.at(petiole).at(0) = internode_vertices.back();

        for (int j = 1; j <= Ndiv_petiole_length; j++) {

            if( fabs(phytomer_parameters.petiole.curvature.val())>0 ) {
                petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, petiole_rotation_axis, -deg2rad(phytomer_parameters.petiole.curvature.val() * dr_petiole_max.at(petiole)));
                phytomer_parameters.petiole.curvature.resample();
            }

            petiole_vertices.at(petiole).at(j) = petiole_vertices.at(petiole).at(j - 1) + dr_petiole.at(petiole) * petiole_axis;

            petiole_radii.at(petiole).at(j) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val() * (1.f - phytomer_parameters.petiole.taper.val() / float(Ndiv_petiole_length) * float(j) );
            petiole_colors.at(j) = phytomer_parameters.petiole.color;

        }

        if( build_context_geometry_petiole && petiole_radii.at(petiole).front() > 0.f ) {
            petiole_objIDs.at(petiole) = makeTubeFromCones(Ndiv_petiole_radius, petiole_vertices.at(petiole), petiole_radii.at(petiole), petiole_colors, context_ptr);
        }

        //--- create buds ---//

        std::vector<VegetativeBud> vegetative_buds_new;
        vegetative_buds_new.resize( phytomer_parameters.internode.max_vegetative_buds_per_petiole.val() );
        phytomer_parameters.internode.max_vegetative_buds_per_petiole.resample();

        vegetative_buds.push_back(vegetative_buds_new);

        std::vector<FloralBud> floral_buds_new;
        floral_buds_new.resize( phytomer_parameters.internode.max_floral_buds_per_petiole.val() );
        phytomer_parameters.internode.max_floral_buds_per_petiole.resample();

        floral_buds.push_back(floral_buds_new);
        resize_vector(inflorescence_objIDs, 0, phytomer_parameters.internode.max_floral_buds_per_petiole.val(), phytomer_parameters.petiole.petioles_per_internode);
        if( build_context_geometry_peduncle ) {
            resize_vector(peduncle_objIDs, 0, phytomer_parameters.internode.max_floral_buds_per_petiole.val(), phytomer_parameters.petiole.petioles_per_internode);
        }
        resize_vector( inflorescence_bases, 0, phytomer_parameters.internode.max_floral_buds_per_petiole.val(), phytomer_parameters.petiole.petioles_per_internode );

        //--- create leaves ---//

        if( phytomer_parameters.leaf.prototype_function == nullptr ){
            helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Leaf prototype function was not defined for shoot type " + parent_shoot->shoot_type_label + ".");
        }

        vec3 petiole_tip_axis = getPetioleAxisVector(1.f, petiole);

        vec3 leaf_rotation_axis = cross(internode_axis, petiole_tip_axis );
        leaf_size_max.at(petiole) = phytomer_parameters.leaf.prototype_scale.val();

        for(int leaf=0; leaf < phytomer_parameters.leaf.leaves_per_petiole; leaf++ ){

            float ind_from_tip = float(leaf)-float(phytomer_parameters.leaf.leaves_per_petiole-1)/2.f;

            uint objID_leaf = phytomer_parameters.leaf.prototype_function(context_ptr, phytomer_parameters.leaf.subdivisions, ind_from_tip, shoot_index.x, shoot_index.y );

            // -- scaling -- //

            vec3 leaf_scale = leaf_scale_factor_fraction * leaf_size_max.at(petiole) * make_vec3(1,1,1);
            if( phytomer_parameters.leaf.leaves_per_petiole>0 && phytomer_parameters.leaf.leaflet_scale.val()!=1.f && ind_from_tip!=0 ){
                leaf_scale = powf(phytomer_parameters.leaf.leaflet_scale.val(),fabs(ind_from_tip))*leaf_scale;
            }

            context_ptr->scaleObject( objID_leaf, leaf_scale );

            float compound_rotation = 0;
            if( phytomer_parameters.leaf.leaves_per_petiole>1 ) {
                if (phytomer_parameters.leaf.leaflet_offset.val() == 0) {
                    float dphi = M_PI / (floor(0.5 * float(phytomer_parameters.leaf.leaves_per_petiole - 1)) + 1);
                    compound_rotation = -float(M_PI) + dphi * (leaf + 0.5f);
                } else {
                    if( leaf == float(phytomer_parameters.leaf.leaves_per_petiole-1)/2.f ){ //tip leaf
                        compound_rotation = 0;
                    }else if( leaf < float(phytomer_parameters.leaf.leaves_per_petiole-1)/2.f ) {
                        compound_rotation = -0.5*M_PI;
                    }else{
                        compound_rotation = 0.5*M_PI;
                    }
                }
            }

            // -- rotations -- //

            //roll rotation
            float roll_rot = 0;
            if( phytomer_parameters.leaf.leaves_per_petiole==1 ){
                int sign = (shoot_index.x%2==0) ? 1 : -1;
                roll_rot = (acos_safe(internode_axis.z)-deg2rad(phytomer_parameters.leaf.roll.val()))*sign;
            } else if( ind_from_tip!= 0){
                roll_rot = (asin_safe(petiole_tip_axis.z)+deg2rad(phytomer_parameters.leaf.roll.val()))*compound_rotation/std::fabs(compound_rotation);
            }
            leaf_rotation.at(petiole).at(leaf).roll = roll_rot;
            phytomer_parameters.leaf.roll.resample();
            context_ptr->rotateObject(objID_leaf, roll_rot, "x" );

            //pitch rotation
            leaf_rotation.at(petiole).at(leaf).pitch = deg2rad(phytomer_parameters.leaf.pitch.val());
            float pitch_rot = leaf_rotation.at(petiole).at(leaf).pitch;
            phytomer_parameters.leaf.pitch.resample();
            if( ind_from_tip==0 ){
                pitch_rot += asin_safe(petiole_tip_axis.z);
            }
            context_ptr->rotateObject(objID_leaf, -pitch_rot , "y" );

            //yaw rotation
            if( ind_from_tip!=0 ){
                leaf_rotation.at(petiole).at(leaf).yaw = -deg2rad(phytomer_parameters.leaf.yaw.val()/fabs(compound_rotation));
                float yaw_rot = leaf_rotation.at(petiole).at(leaf).yaw;
                phytomer_parameters.leaf.yaw.resample();
                context_ptr->rotateObject( objID_leaf, yaw_rot, "z" );
            }else{
                leaf_rotation.at(petiole).at(leaf).yaw = 0;
            }

            //rotate to azimuth of petiole
            context_ptr->rotateObject( objID_leaf, -std::atan2(petiole_tip_axis.y, petiole_tip_axis.x)+compound_rotation, "z" );


            // -- translation -- //

            vec3 leaf_base = petiole_vertices.at(petiole).back();
            if( phytomer_parameters.leaf.leaves_per_petiole>1 && phytomer_parameters.leaf.leaflet_offset.val()>0 ){
                if( ind_from_tip != 0 ) {
                    float offset = (fabs(ind_from_tip) - 0.5f) * phytomer_parameters.leaf.leaflet_offset.val() * phytomer_parameters.petiole.length.val();
                    leaf_base = interpolateTube(petiole_vertices.at(petiole), 1.f - offset / phytomer_parameters.petiole.length.val() );
                }
            }

            context_ptr->translateObject( objID_leaf, leaf_base );

            leaf_objIDs.at(petiole).push_back(objID_leaf );
            leaf_bases.at(petiole).push_back(leaf_base );

        }
        phytomer_parameters.leaf.prototype_scale.resample();

        if( petiole_axis==make_vec3(0,0,1) ) {
            inflorescence_bending_axis = make_vec3(1, 0, 0);
        }else{
            inflorescence_bending_axis = cross(make_vec3(0, 0, 1), petiole_axis);
        }

    }

}

void Phytomer::addInflorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation, const helios::vec3 &a_inflorescence_bending_axis, FloralBud &fbud) {

    uint Ndiv_rachis_length = std::max(uint(1), phytomer_parameters.peduncle.length_segments);
    uint Ndiv_rachis_radius = std::max(uint(3), phytomer_parameters.peduncle.radial_subdivisions);
    if( phytomer_parameters.peduncle.length_segments==0 || phytomer_parameters.peduncle.radial_subdivisions<3 ){
        build_context_geometry_peduncle = false;
    }

    float dr_peduncle = phytomer_parameters.peduncle.length.val() / float(Ndiv_rachis_length);
    phytomer_parameters.peduncle.length.resample();

    std::vector<vec3> peduncle_vertices(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_vertices.at(0) = base_position;
    std::vector<float> peduncle_radii(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_radii.at(0) = phytomer_parameters.peduncle.radius.val();
    std::vector<RGBcolor> peduncle_colors(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_colors.at(0) = phytomer_parameters.internode.color;

    vec3 peduncle_axis = getAxisVector(1.f, internode_vertices );

    //peduncle pitch rotation
    if( phytomer_parameters.peduncle.pitch.val()!=0.f || base_rotation.pitch!=0.f ) {
        peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, inflorescence_bending_axis, deg2rad(phytomer_parameters.peduncle.pitch.val())+base_rotation.pitch );
        phytomer_parameters.peduncle.pitch.resample();
    }

    //rotate peduncle to azimuth of petiole and apply peduncle base yaw rotation
    vec3 internode_axis = getAxisVector(1.f, internode_vertices );
    vec3 parent_petiole_base_axis = getPetioleAxisVector(0.f, fbud.parent_petiole_index);
    float parent_petiole_azimuth = -std::atan2(parent_petiole_base_axis.y, parent_petiole_base_axis.x);
    float current_peduncle_azimuth = -std::atan2(peduncle_axis.y, peduncle_axis.x);
    peduncle_axis = rotatePointAboutLine( peduncle_axis, nullorigin, internode_axis, (current_peduncle_azimuth-parent_petiole_azimuth) + base_rotation.yaw );

    float theta_base = fabs(cart2sphere(peduncle_axis).zenith);

    for(int i=1; i<=phytomer_parameters.peduncle.length_segments; i++ ){

        if( phytomer_parameters.peduncle.curvature.val()!=0.f ) {
            float theta_curvature = -deg2rad(phytomer_parameters.peduncle.curvature.val() * dr_peduncle);
            phytomer_parameters.peduncle.curvature.resample();
            if (fabs(theta_curvature) * float(i) < M_PI - theta_base) {
                peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, inflorescence_bending_axis, theta_curvature);
            } else {
                peduncle_axis = make_vec3(0, 0, -1);
            }
        }

        peduncle_vertices.at(i) = peduncle_vertices.at(i - 1) + dr_peduncle * peduncle_axis;

        peduncle_radii.at(i) = phytomer_parameters.peduncle.radius.val();
        peduncle_colors.at(i) = phytomer_parameters.internode.color;

    }
    phytomer_parameters.peduncle.radius.resample();

    if( build_context_geometry_peduncle) {
        peduncle_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index).push_back(context_ptr->addTubeObject(Ndiv_rachis_radius, peduncle_vertices, peduncle_radii, peduncle_colors));
    }

    for(int fruit=0; fruit < phytomer_parameters.inflorescence.flowers_per_rachis.val(); fruit++ ){

        uint objID_fruit;
        helios::vec3 fruit_scale;

        if(fbud.state == BUD_FRUITING ){
            objID_fruit = phytomer_parameters.inflorescence.fruit_prototype_function(context_ptr,1,fbud.time_counter);
            fruit_scale = phytomer_parameters.inflorescence.fruit_prototype_scale.val()*make_vec3(1,1,1);
            phytomer_parameters.inflorescence.fruit_prototype_scale.resample();
        }else{
            bool flower_is_open;
            if(fbud.state == BUD_FLOWER_CLOSED ) {
                flower_is_open = false;
            }else{
                flower_is_open = true;
            }
            objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr,1,flower_is_open);
            fruit_scale = phytomer_parameters.inflorescence.flower_prototype_scale.val()*make_vec3(1,1,1);
            phytomer_parameters.inflorescence.flower_prototype_scale.resample();
        }

        float ind_from_tip = fabs(fruit- float(phytomer_parameters.inflorescence.flowers_per_rachis.val() - 1) / 2.f);

        context_ptr->scaleObject( objID_fruit, fruit_scale );

        //if we have more than one flower/fruit, we need to adjust the base position of the fruit
        vec3 fruit_base = peduncle_vertices.back();
        float frac = 1;
        if(phytomer_parameters.inflorescence.flowers_per_rachis.val() > 1 && phytomer_parameters.inflorescence.flower_offset.val() > 0 ){
            if( ind_from_tip != 0 ) {
                float offset = 0;
                if(phytomer_parameters.inflorescence.flower_arrangement_pattern == "opposite" ){
                    offset = (ind_from_tip - 0.5f) * phytomer_parameters.inflorescence.flower_offset.val() * phytomer_parameters.peduncle.length.val();
                }else if(phytomer_parameters.inflorescence.flower_arrangement_pattern == "alternate" ){
                    offset = (ind_from_tip - 0.5f + 0.5f*float(fruit> float(phytomer_parameters.inflorescence.flowers_per_rachis.val() - 1) / 2.f) ) * phytomer_parameters.inflorescence.flower_offset.val() * phytomer_parameters.peduncle.length.val();
                }else{
                    helios_runtime_error("ERROR (PlantArchitecture::addInflorescence): Invalid fruit arrangement pattern.");
                }
                if( phytomer_parameters.peduncle.length.val()>0 ){
                    frac = 1.f - offset / phytomer_parameters.peduncle.length.val();
                }
                fruit_base = interpolateTube(peduncle_vertices, frac);
            }
        }

        //if we have more than one flower/fruit, we need to adjust the rotation about the peduncle
        float compound_rotation = 0;
        if(phytomer_parameters.inflorescence.flowers_per_rachis.val() > 1 ) {
            if (phytomer_parameters.inflorescence.flower_offset.val() == 0) { //flowers/fruit are all at the tip, so just equally distribute them about the azimuth
                float dphi = M_PI / (floor(0.5 * float(phytomer_parameters.inflorescence.flowers_per_rachis.val() - 1)) + 1);
                compound_rotation = -float(M_PI) + dphi * (fruit + 0.5f);
            } else {
                if( fruit < float(phytomer_parameters.inflorescence.flowers_per_rachis.val() - 1) / 2.f ) {
                    compound_rotation = 0;
                }else {
                    compound_rotation = M_PI;
                }
            }
        }

        peduncle_axis = getAxisVector(frac, peduncle_vertices );

        vec3 fruit_axis = peduncle_axis;

        //roll rotation
        if( phytomer_parameters.inflorescence.roll.val()!=0.f ) {
            context_ptr->rotateObject(objID_fruit, deg2rad(phytomer_parameters.inflorescence.roll.val()), "x" );
            phytomer_parameters.inflorescence.roll.resample();
        }

        //pitch rotation
        float pitch_inflorescence = -asin_safe(peduncle_axis.z) + deg2rad(phytomer_parameters.inflorescence.pitch.val());
        phytomer_parameters.inflorescence.pitch.resample();
        if(fbud.state == BUD_FRUITING ) { //gravity effect for fruit
            pitch_inflorescence = pitch_inflorescence + phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.val() * (0.5f * M_PI - pitch_inflorescence);
        }
        context_ptr->rotateObject(objID_fruit, pitch_inflorescence, "y");
        fruit_axis = rotatePointAboutLine(fruit_axis, nullorigin, make_vec3(1, 0, 0), pitch_inflorescence);

        //rotate flower/fruit to azimuth of peduncle
        context_ptr->rotateObject(objID_fruit, -std::atan2(peduncle_axis.y, peduncle_axis.x), "z" );
        fruit_axis = rotatePointAboutLine( fruit_axis, nullorigin, make_vec3(0,0,1), -std::atan2(peduncle_axis.y, peduncle_axis.x) );

        context_ptr->translateObject( objID_fruit, fruit_base );

        //rotate flower/fruit about peduncle (roll)
        if( phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.val()!=0 && fbud.state == BUD_FRUITING ) {
            context_ptr->rotateObject(objID_fruit, deg2rad(phytomer_parameters.peduncle.roll.val()) + compound_rotation, fruit_base, make_vec3(0, 0, 1));
        }else{
            context_ptr->rotateObject(objID_fruit, deg2rad(phytomer_parameters.peduncle.roll.val()) + compound_rotation, fruit_base, peduncle_axis);
            fruit_axis = rotatePointAboutLine(fruit_axis, nullorigin, peduncle_axis, deg2rad(phytomer_parameters.peduncle.roll.val()) + compound_rotation);
        }
        phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.resample();

        inflorescence_bases.at(fbud.parent_petiole_index).at(fbud.bud_index).push_back( fruit_base );

        inflorescence_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index).push_back( objID_fruit );

    }
    phytomer_parameters.inflorescence.flowers_per_rachis.resample();
    phytomer_parameters.peduncle.roll.resample();

    context_ptr->setObjectData( peduncle_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index), "rank", rank );

    context_ptr->setObjectData( inflorescence_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index), "rank", rank );

    for( uint objID : peduncle_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index) ) {
        context_ptr->setObjectData( objID, "peduncleID", (int)objID );
    }
    for( uint objID : inflorescence_objIDs.at(fbud.parent_petiole_index).at(fbud.bud_index) ) {
        if( fbud.state == BUD_FLOWER_CLOSED ) {
            context_ptr->setObjectData( objID, "closedflowerID", (int) objID);
        }else if( fbud.state == BUD_FLOWER_OPEN ) {
            context_ptr->clearObjectData( objID, "closedflowerID" );
            context_ptr->setObjectData( objID, "openflowerID", (int) objID);
        }else{
            context_ptr->setObjectData( objID, "fruitID", (int) objID);
        }
    }

}

void Phytomer::setPetioleBase( const helios::vec3 &base_position ){

    vec3 old_base = petiole_vertices.front().front();
    vec3 shift = base_position - old_base;

    for( int petiole=0; petiole<phytomer_parameters.petiole.petioles_per_internode; petiole++ ) {
        for (auto &vertex: petiole_vertices.at(petiole) ) {
            vertex += shift;
        }
    }

    if( build_context_geometry_petiole ) {
        context_ptr->translateObject(flatten(petiole_objIDs), shift);
    }
    context_ptr->translateObject( flatten(leaf_objIDs), shift );

    for( int petiole=0; petiole<leaf_bases.size(); petiole++ ) {
        for (auto &leaf_base: leaf_bases.at(petiole)) {
            leaf_base += shift;
        }
        if( !inflorescence_objIDs.empty() ) {
            for (int bud = 0; bud < inflorescence_objIDs.at(petiole).size(); bud++) {
                context_ptr->translateObject(inflorescence_objIDs.at(petiole).at(bud), shift);
                for (auto &inflorescence_base: inflorescence_bases.at(petiole).at(bud)) {
                    inflorescence_base += shift;
                }
                if( build_context_geometry_peduncle ) {
                    context_ptr->translateObject(peduncle_objIDs.at(petiole).at(bud), shift);
                }
            }
        }
    }

}

void Phytomer::setPhytomerBase( const helios::vec3 &base_position ){

    vec3 old_base = internode_vertices.front();
    vec3 shift = base_position - old_base;

    for( auto & vertex : internode_vertices){
        vertex += shift;
    }

    for( int petiole=0; petiole<phytomer_parameters.petiole.petioles_per_internode; petiole++ ) {
        for (auto &vertex: petiole_vertices.at(petiole) ) {
            vertex += shift;
        }
    }

    if( build_context_geometry_internode ) {
        context_ptr->translateObject(internode_objIDs, shift);
    }
    if( build_context_geometry_petiole ) {
        context_ptr->translateObject(flatten(petiole_objIDs), shift);
    }
    context_ptr->translateObject( flatten(leaf_objIDs), shift );
    for( int petiole=0; petiole<leaf_bases.size(); petiole++ ) {
        for (auto &leaf_base: leaf_bases.at(petiole)) {
            leaf_base += shift;
        }
        if( !inflorescence_objIDs.empty() ) {
            for (int bud = 0; bud < inflorescence_objIDs.at(petiole).size(); bud++) {
                context_ptr->translateObject(inflorescence_objIDs.at(petiole).at(bud), shift);
                for (auto &inflorescence_base: inflorescence_bases.at(petiole).at(bud)) {
                    inflorescence_base += shift;
                }
                if( build_context_geometry_peduncle ) {
                    context_ptr->translateObject(peduncle_objIDs.at(petiole).at(bud), shift);
                }
            }
        }
    }

}

void Phytomer::setInternodeLengthScaleFraction(float internode_scale_factor_fraction ){

    assert(internode_scale_factor_fraction >= 0 && internode_scale_factor_fraction <= 1 );

    if(internode_scale_factor_fraction == current_internode_scale_factor ){
        return;
    }

    float delta_scale = internode_scale_factor_fraction / current_internode_scale_factor;

    internode_length = internode_length*delta_scale;
    current_internode_scale_factor = internode_scale_factor_fraction;

    if( build_context_geometry_internode ) {
        int node = 0;
        vec3 last_base = internode_vertices.front();
        for (uint objID: internode_objIDs) { //should be automatically skipped if internode geometry was not built
            context_ptr->getConeObjectPointer(objID)->scaleLength(delta_scale);
            if (node > 0) {
                vec3 new_base = context_ptr->getConeObjectNode(objID, 0);
                context_ptr->translateObject(objID, last_base - new_base);
            }
            last_base = context_ptr->getConeObjectNode(objID, 1);
            internode_vertices.at(node + 1) = last_base;
            node++;
        }
    }else{
        for( int i=1; i<internode_vertices.size(); i++ ) {
            vec3 axis_vector = internode_vertices.at(i) - internode_vertices.at(i-1);
            internode_vertices.at(i) = internode_vertices.at(i-1) + delta_scale*axis_vector;
        }
    }

    //translate leaf to new internode position
    setPetioleBase( internode_vertices.back() );


}

void Phytomer::setInternodeMaxLength( float internode_length_max ){
    this->internode_length_max = internode_length_max;

    current_internode_scale_factor = current_internode_scale_factor*this->internode_length_max/internode_length_max;

    if( current_internode_scale_factor>=1.f ){
        setInternodeLengthScaleFraction(1.f);
        current_internode_scale_factor = 1.f;
    }
}

void Phytomer::scaleInternodeMaxLength( float scale_factor ){
    this->internode_length_max *= scale_factor;

    current_internode_scale_factor = current_internode_scale_factor/scale_factor;

    if( current_internode_scale_factor>=1.f ){
        setInternodeLengthScaleFraction(1.f);
        current_internode_scale_factor = 1.f;
    }
}

void Phytomer::setInternodeMaxRadius( float internode_radius_max ){

    this->internode_radius_max = internode_radius_max;

}

void Phytomer::setLeafScaleFraction(float leaf_scale_factor_fraction ){

    assert(leaf_scale_factor_fraction >= 0 && leaf_scale_factor_fraction <= 1 );

    if(leaf_scale_factor_fraction == current_leaf_scale_factor || (leaf_objIDs.empty() && petiole_objIDs.empty())  ){
        return;
    }

    float delta_scale = leaf_scale_factor_fraction / current_leaf_scale_factor;

    for( int petiole=0; petiole<phytomer_parameters.petiole.petioles_per_internode; petiole++ ) {
        petiole_length.at(petiole) *= delta_scale;
    }
    current_leaf_scale_factor = leaf_scale_factor_fraction;

    assert(leaf_objIDs.size() == leaf_bases.size());

    //scale the petiole
    for( int petiole=0; petiole<phytomer_parameters.petiole.petioles_per_internode; petiole++ ) {

        if( !petiole_objIDs.empty() ) {
            int node = 0;
            vec3 old_tip = petiole_vertices.at(petiole).back();
            vec3 last_base = petiole_vertices.at(petiole).front();//looping over petioles
            for (uint objID: petiole_objIDs.at(petiole)) { //looping over cones/segments within petiole
                context_ptr->getConeObjectPointer(objID)->scaleLength(delta_scale);
                context_ptr->getConeObjectPointer(objID)->scaleGirth(delta_scale);
                petiole_radii.at(petiole).at(node) *= delta_scale;
                if (node > 0) {
                    vec3 new_base = context_ptr->getConeObjectNode(objID, 0);
                    context_ptr->translateObject(objID, last_base - new_base);
                } else {
                    petiole_vertices.at(petiole).at(0) = context_ptr->getConeObjectNode(objID, 0);
                }
                last_base = context_ptr->getConeObjectNode(objID, 1);
                petiole_vertices.at(petiole).at(node + 1) = last_base;
                node++;
            }
        }else{
            for( int i=1; i<petiole_vertices.at(petiole).size(); i++ ) {
                vec3 axis_vector = petiole_vertices.at(petiole).at(i) - petiole_vertices.at(petiole).at(i-1);
                petiole_vertices.at(petiole).at(i) = petiole_vertices.at(petiole).at(i-1) + delta_scale*axis_vector;
                petiole_radii.at(petiole).at(i) *= delta_scale;
            }
        }

        //scale and translate leaves
        assert(leaf_objIDs.at(petiole).size() == leaf_bases.at(petiole).size());
        for (int leaf = 0; leaf < leaf_objIDs.at(petiole).size(); leaf++) {

            float ind_from_tip = float(leaf) - float(phytomer_parameters.leaf.leaves_per_petiole - 1) / 2.f;

            context_ptr->translateObject(leaf_objIDs.at(petiole).at(leaf), -1 * leaf_bases.at(petiole).at(leaf));
            context_ptr->scaleObject(leaf_objIDs.at(petiole).at(leaf), delta_scale * make_vec3(1, 1, 1));
            if (ind_from_tip == 0) {
                context_ptr->translateObject(leaf_objIDs.at(petiole).at(leaf), petiole_vertices.at(petiole).back());
                leaf_bases.at(petiole).at(leaf) = petiole_vertices.at(petiole).back();
            } else {
                float offset = (fabs(ind_from_tip) - 0.5f) * phytomer_parameters.leaf.leaflet_offset.val() * phytomer_parameters.petiole.length.val();
                vec3 leaf_base = interpolateTube(petiole_vertices.at(petiole), 1.f - offset / phytomer_parameters.petiole.length.val());
                context_ptr->translateObject(leaf_objIDs.at(petiole).at(leaf), leaf_base);
                leaf_bases.at(petiole).at(leaf) = leaf_base;
            }

        }

    }

}

void Phytomer::setLeafPrototypeScale( float leaf_prototype_scale ){

    current_leaf_scale_factor = current_leaf_scale_factor*leaf_size_max.front()/leaf_prototype_scale;

    if( current_leaf_scale_factor>=1.f ){
        setLeafScaleFraction(1.f);
        current_leaf_scale_factor = 1.f;
    }

}
void Phytomer::scaleLeafPrototypeScale( float scale_factor ){

    current_leaf_scale_factor = current_leaf_scale_factor/scale_factor;

    if( current_leaf_scale_factor>=1.f ){
        setLeafScaleFraction(1.f);
        current_leaf_scale_factor = 1.f;
    }

//    for( int petiole=0; petiole<phytomer_parameters.petiole.petioles_per_internode; petiole++ ) {
//        leaf_size_max.at(petiole) *= scale_factor;
//        for( int leaf=0; leaf<leaf_objIDs.at(petiole).size(); leaf++ ) {
//            context_ptr->translateObject( leaf_objIDs.at(petiole).at(leaf), -1 * leaf_bases.at(petiole).at(leaf) );
//            context_ptr->scaleObject( leaf_objIDs.at(petiole).at(leaf), make_vec3(1,1,1)*scale_factor );
//            context_ptr->translateObject( leaf_objIDs.at(petiole).at(leaf), leaf_bases.at(petiole).at(leaf) );
//        }
//    }

}


void Phytomer::setInflorescenceScaleFraction(FloralBud &fbud, float inflorescence_scale_factor_fraction) {

    assert(inflorescence_scale_factor_fraction >= 0 && inflorescence_scale_factor_fraction <= 1 );

    if(inflorescence_scale_factor_fraction == fbud.current_fruit_scale_factor ){
        return;
    }

    float delta_scale = inflorescence_scale_factor_fraction / fbud.current_fruit_scale_factor;

    fbud.current_fruit_scale_factor = inflorescence_scale_factor_fraction;

    //scale and translate flowers/fruit
    for (int inflorescence = 0; inflorescence < inflorescence_objIDs.at( fbud.parent_petiole_index ).at( fbud.bud_index ).size(); inflorescence++) {

        uint objID = inflorescence_objIDs.at( fbud.parent_petiole_index ).at( fbud.bud_index ).at(inflorescence);

        context_ptr->translateObject(objID, -1 * inflorescence_bases.at( fbud.parent_petiole_index ).at( fbud.bud_index ).at(inflorescence));
        context_ptr->scaleObject(objID, delta_scale * make_vec3(1, 1, 1));
        context_ptr->translateObject(objID, inflorescence_bases.at( fbud.parent_petiole_index ).at( fbud.bud_index ).at(inflorescence));

    }

}

void Phytomer::removeLeaf(){

    context_ptr->deleteObject(flatten(leaf_objIDs));
    leaf_objIDs.resize(0);
    leaf_bases.resize(0);

    if( build_context_geometry_petiole ) {
        context_ptr->deleteObject(flatten(petiole_objIDs));
        petiole_objIDs.resize(0);
    }

}

bool Phytomer::hasLeaf() const{
    return !leaf_objIDs.empty();
}

bool Phytomer::hasInflorescence() const {
    return !inflorescence_objIDs.empty();
}

Shoot::Shoot(uint plant_ID, int shoot_ID, int parent_shoot_ID, uint parent_node, uint parent_petiole_index, uint rank, const helios::vec3 &origin, const AxisRotation &shoot_base_rotation, uint current_node_number,
             float internode_length_shoot_initial, const ShootParameters& shoot_params, std::string shoot_type_label, PlantArchitecture *plant_architecture_ptr) :
        plantID(plant_ID), ID(shoot_ID), parent_shoot_ID(parent_shoot_ID), parent_node_index(parent_node), parent_petiole_index(parent_petiole_index), rank(rank), origin(origin), base_rotation(shoot_base_rotation), current_node_number(current_node_number), internode_length_max_shoot_initial(internode_length_shoot_initial), shoot_parameters(shoot_params), shoot_type_label(std::move(shoot_type_label)), plant_architecture_ptr(plant_architecture_ptr) {
    assimilate_pool = 0;
    phyllochron_counter = 0;
    dormant = true;
    context_ptr = plant_architecture_ptr->context_ptr;
}

void Shoot::buildShootPhytomers(float internode_radius, float internode_length, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction) {

    for( int i=0; i<current_node_number; i++ ) { //loop over phytomers to build up the shoot

        //Determine position of internode base
        vec3 internode_base_position;
        if( i==0 ){ //first phytomer on shoot
            internode_base_position = origin;
        }else{ // not the first phytomer on the shoot
            internode_base_position = phytomers.at(i-1)->internode_vertices.back();
        }

        //Adding the phytomer(s) to the shoot
        int pID = addPhytomer(shoot_parameters.phytomer_parameters, internode_base_position, this->base_rotation, internode_radius, internode_length, internode_length_scale_factor_fraction, leaf_scale_factor_fraction);

    }

}

bool Shoot::sampleChildShootType(std::string &child_shoot_type_label) const{

    auto shoot_ptr = this;

    assert( shoot_ptr->shoot_parameters.child_shoot_type_labels.size() == shoot_ptr->shoot_parameters.child_shoot_type_probabilities.size() );

    child_shoot_type_label = "";

    if ( shoot_ptr->shoot_parameters.child_shoot_type_labels.size()==0 ) { //if user doesn't specify child shoot types, generate the same type by default
        child_shoot_type_label = shoot_ptr->shoot_type_label;
    }else if( shoot_ptr->shoot_parameters.child_shoot_type_labels.size()==1 ){ //if only one child shoot types was specified, use it
        child_shoot_type_label = shoot_ptr->shoot_parameters.child_shoot_type_labels.at(0);
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
    if (context_ptr->randu() > plant_architecture_ptr->shoot_types.at(shoot_ptr->shoot_type_label).vegetative_bud_break_probability.val() ) {
        bud_break = false;
        child_shoot_type_label = "";
    }


    return bud_break;

}

uint PlantArchitecture::addBaseStemShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction,
                                         const std::string &shoot_type_label) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shoot_types.find(shoot_type_label) == shoot_types.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);

    validateShootTypes(shoot_parameters);

    if(current_node_number > shoot_parameters.max_nodes.val() ){
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_parameters.max_nodes.val()) + ".");
    }

    uint shootID = shoot_tree_ptr->size();

    auto* shoot_new = (new Shoot(plantID, shootID, -1, 0, 0, 0, plant_instances.at(plantID).base_position, base_rotation, current_node_number, internode_length_max, shoot_parameters, shoot_type_label, this));
    shoot_tree_ptr->emplace_back(shoot_new);
    shoot_new->buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction);

    return shootID;

}

uint PlantArchitecture::appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction,
                                    float leaf_scale_factor_fraction, const std::string &shoot_type_label) {

    if( plant_instances.find(plantID) == plant_instances.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shoot_types.find(shoot_type_label) == shoot_types.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);

    validateShootTypes(shoot_parameters);

    if( shoot_tree_ptr->empty() ){
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot append shoot to empty shoot. You must call addBaseStemShoot() first for each plant.");
    }else if( parent_shoot_ID >= int(shoot_tree_ptr->size()) ){
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    }else if(current_node_number > shoot_parameters.max_nodes.val() ){
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_parameters.max_nodes.val()) + ".");
    }else if( shoot_tree_ptr->at(parent_shoot_ID)->phytomers.empty() ){
        std::cerr << "WARNING (PlantArchitecture::appendShoot): Shoot does not have any phytomers to append." << std::endl;
    }

    //stop parent shoot from producing new phytomers at the apex
    shoot_tree_ptr->at(parent_shoot_ID)->shoot_parameters.max_nodes = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number;

    shoot_tree_ptr->at(parent_shoot_ID)->terminateApicalBud(); //meristem should not keep growing after appending shoot

    int appended_shootID = shoot_tree_ptr->size();

    uint parent_node = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number-1;

    uint rank = shoot_tree_ptr->at(parent_shoot_ID)->rank;

    vec3 base_position = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.back()->internode_vertices.back();

    auto * shoot_new = (new Shoot(plantID, appended_shootID, parent_shoot_ID, parent_node, 0, rank, base_position, base_rotation, current_node_number, internode_length_max, shoot_parameters, shoot_type_label, this));
    shoot_tree_ptr->emplace_back(shoot_new);
    shoot_new->buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction);

    shoot_tree_ptr->at(parent_shoot_ID)->childIDs[(int)shoot_tree_ptr->at(parent_shoot_ID)->current_node_number] = appended_shootID;

    return appended_shootID;

}

uint PlantArchitecture::addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node_index, uint current_node_number, const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max,
                                      float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, const std::string &shoot_type_label, uint petiole_index) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shoot_types.find(shoot_type_label) == shoot_types.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::addShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);

    validateShootTypes(shoot_parameters);

    if(parent_shoot_ID <= -1 || parent_shoot_ID >= shoot_tree_ptr->size() ){
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    }else if(shoot_tree_ptr->at(parent_shoot_ID)->phytomers.size() <= parent_node_index ) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent shoot does not have a node " + std::to_string(parent_node_index) + ".");
    }

    uint parent_rank = (int)shoot_tree_ptr->at(parent_shoot_ID)->rank;
    int parent_node_count = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number;

    vec3 shoot_base_position;
    if(parent_shoot_ID > -1 ){
        auto shoot_phytomers = &shoot_tree_ptr->at(parent_shoot_ID)->phytomers;

        if(parent_node_index >= shoot_phytomers->size() ){
            helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Requested to place child shoot on node " + std::to_string(parent_node_index) + " but parent only has " + std::to_string(shoot_phytomers->size()) + " nodes." );
        }

        shoot_base_position = shoot_phytomers->at(parent_node_index)->internode_vertices.back();

        //\todo Shift the shoot base position outward by the parent internode radius

    }else{
        helios_runtime_error("PlantArchitecture::addChildShoot: Should not be here.");
    }

    int childID = shoot_tree_ptr->size();

    auto* shoot_new = (new Shoot(plantID, childID, parent_shoot_ID, parent_node_index, petiole_index, parent_rank + 1, shoot_base_position, shoot_base_rotation, current_node_number, internode_length_max, shoot_parameters, shoot_type_label, this));
    shoot_tree_ptr->emplace_back(shoot_new);
    shoot_new->buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction);

    shoot_tree_ptr->at(parent_shoot_ID)->childIDs[(int)parent_node_index] = childID;

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

int PlantArchitecture::addPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_parameters, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::addPhytomerToShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    if(shootID >= shoot_tree_ptr->size() ){
        helios_runtime_error("ERROR (PlantArchitecture::addPhytomerToShoot): Parent with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto current_shoot_ptr = plant_instances.at(plantID).shoot_tree.at(shootID);

    //The base position of this phytomer is the last vertex position of the prior phytomer on the shoot
    vec3 base_position = current_shoot_ptr->phytomers.back()->internode_vertices.back();
    //The base rotation of this phytomer is the base rotation of the current shoot (the phytomer will be further rotated in the Phytomer constructor)
    AxisRotation base_rotation = current_shoot_ptr->base_rotation;

    int pID = current_shoot_ptr->addPhytomer(phytomer_parameters, base_position, base_rotation, internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction);

    current_shoot_ptr->current_node_number ++;

    return pID;

}

void PlantArchitecture::disableInternodeContextBuild(){
    build_context_geometry_internode = false;
}

void PlantArchitecture::disablePetioleContextBuild(){
    build_context_geometry_petiole = false;
}

void PlantArchitecture::disablePeduncleContextBuild(){
    build_context_geometry_peduncle = false;
}

void PlantArchitecture::enableGroundClipping( float ground_height ){
    ground_clipping_height = ground_height;
}

void PlantArchitecture::incrementPhytomerInternodeGirth(uint plantID, uint shootID, uint node_number, float girth_change){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto parent_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if( node_number>=parent_shoot->current_node_number ){
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Cannot scale internode " + std::to_string(node_number) + " because there are only " + std::to_string(parent_shoot->current_node_number) + " nodes in this shoot.");
    }

    auto phytomer = parent_shoot->phytomers.at(node_number);

    if( girth_change != 0.f ) {

        int node = 0;
        for (uint objID: phytomer->internode_objIDs) { //should be automatically skipped if internode geometry was not built

            float radius = context_ptr->getConeObjectNodeRadius(objID, 0);

            if( radius*girth_change > phytomer->internode_radius_max ){
                girth_change = phytomer->internode_radius_max/radius;
                radius = phytomer->internode_radius_max;
            }

            context_ptr->getConeObjectPointer(objID)->scaleGirth(girth_change);

            phytomer->internode_radii.at(node) = context_ptr->getConeObjectNodeRadius(objID, 0);

            node++;
        }

    }

    // \todo Shift all shoot bases outward to account for the girth scaling

}

void PlantArchitecture::shiftDownstreamShoots(uint plantID, std::vector<std::shared_ptr<Shoot>> &shoot_tree, std::shared_ptr<Shoot> parent_shoot_ptr, const vec3 &base_position ){

    for(int node=0; node < parent_shoot_ptr->phytomers.size(); node++ ){

        if(parent_shoot_ptr->childIDs.find(node) != parent_shoot_ptr->childIDs.end() ){
            auto child_shoot = shoot_tree.at(parent_shoot_ptr->childIDs.at(node));
            setShootOrigin(plantID, parent_shoot_ptr->childIDs.at(node), parent_shoot_ptr->phytomers.at(node)->internode_vertices.back());
            shiftDownstreamShoots(plantID, shoot_tree, child_shoot, parent_shoot_ptr->phytomers.at(node)->internode_vertices.back() );
        }

    }

}

void PlantArchitecture::setPhytomerInternodeLengthScaleFraction(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeLengthScaleFraction): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto current_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeLengthScaleFraction): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if(node_number >= current_shoot->current_node_number ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeLengthScaleFraction): Cannot scale internode " + std::to_string(node_number) + " because there are only " + std::to_string(current_shoot->current_node_number) + " nodes in this shoot.");
    }
    if(internode_scale_factor_fraction < 0 || internode_scale_factor_fraction > 1 ){
        std::cerr << "WARNING (PlantArchitecture::setPhytomerInternodeLengthScaleFraction): Internode scaling factor was outside the range of 0 to 1. No scaling was applied." << std::endl;
        return;
    }

    current_shoot->phytomers.at(node_number)->setInternodeLengthScaleFraction(internode_scale_factor_fraction);

    for(int node=node_number; node < current_shoot->phytomers.size(); node++ ){

        //shift all downstream phytomers
        if( node>node_number ) {
            vec3 upstream_base = current_shoot->phytomers.at(node - 1)->internode_vertices.back();
            current_shoot->phytomers.at(node)->setPhytomerBase(upstream_base);
        }

        //shift all downstream shoots
        if( current_shoot->childIDs.find(node) != current_shoot->childIDs.end() ) {
            auto child_shoot = plant_instances.at(plantID).shoot_tree.at(current_shoot->childIDs.at(node));
            setShootOrigin(plantID, current_shoot->childIDs.at(node), current_shoot->phytomers.at(node)->internode_vertices.back());
            shiftDownstreamShoots(plantID, plant_instances.at(plantID).shoot_tree, child_shoot, current_shoot->phytomers.at(node)->internode_vertices.back());
        }

    }

    //shift appended shoot (if present)
    int node = current_shoot->phytomers.size();
    if( current_shoot->childIDs.find(node) != current_shoot->childIDs.end() ) {
        auto child_shoot = plant_instances.at(plantID).shoot_tree.at(current_shoot->childIDs.at(node));
        setShootOrigin(plantID, current_shoot->childIDs.at(node), current_shoot->phytomers.back()->internode_vertices.back());
        shiftDownstreamShoots(plantID, plant_instances.at(plantID).shoot_tree, child_shoot, current_shoot->phytomers.back()->internode_vertices.back());
    }

}

void PlantArchitecture::setShootOrigin(uint plantID, uint shootID, const helios::vec3 &origin){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setShootOrigin): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }else if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::setShootOrigin): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shoot->phytomers.empty() ){
        return;
    }

    uint node_count = shoot->phytomers.size();

    shoot->phytomers.front()->setPhytomerBase(origin);

    vec3 upstream_base = shoot->phytomers.front()->internode_vertices.front();
    for( int node=0; node<node_count; node++ ) {
        shoot->phytomers.at(node)->setPhytomerBase(upstream_base);
        if( shoot->childIDs.find(node) != shoot->childIDs.end() ){
            auto child_shoot = plant_instances.at(plantID).shoot_tree.at(shoot->childIDs.at(node));
            setShootOrigin(plantID, shoot->childIDs.at(node), upstream_base);
        }
        upstream_base = shoot->phytomers.at(node)->internode_vertices.back();
    }

}

void PlantArchitecture::setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerInternodeLengthScaleFraction): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto parent_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerLeafScale): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }else if( node_number>=parent_shoot->current_node_number ){
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerLeafScale): Cannot scale leaf " + std::to_string(node_number) + " because there are only " + std::to_string(parent_shoot->current_node_number) + " nodes in this shoot.");
    }
    if(leaf_scale_factor_fraction < 0 || leaf_scale_factor_fraction > 1 ){
        std::cerr << "WARNING (PlantArchitecture::setPhytomerLeafScale): Leaf scaling factor was outside the range of 0 to 1. No scaling was applied." << std::endl;
        return;
    }

    parent_shoot->phytomers.at(node_number)->setLeafScaleFraction(leaf_scale_factor_fraction);

}

void PlantArchitecture::setPlantBasePosition(uint plantID, const helios::vec3 &base_position) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).base_position = base_position;

    //\todo Does not work after shoots have been added to the plant.
    if( !plant_instances.at(plantID).shoot_tree.empty() ){
        std::cerr << "WARNING (PlantArchitecture::setPlantBasePosition): This function does not work after shoots have been added to the plant." << std::endl;
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

void PlantArchitecture::harvestPlant(uint plantID){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::harvestPlant): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for( auto& shoot: plant_instances.at(plantID).shoot_tree ){
        for( auto& phytomer: shoot->phytomers ){

            phytomer->setFloralBudState( BUD_DEAD );

        }
    }

}

void PlantArchitecture::removeShootLeaves(uint plantID, uint shootID){
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::removePlantLeaves): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    if( shootID>=plant_instances.at(plantID).shoot_tree.size() ){
        helios_runtime_error("ERROR (PlantArchitecture::removeShootLeaves): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto& shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    for( auto& phytomer: shoot->phytomers ){
        phytomer->removeLeaf();
    }

}

void PlantArchitecture::removePlantLeaves(uint plantID ){
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::removePlantLeaves): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for( auto& shoot: plant_instances.at(plantID).shoot_tree ){
        for( auto& phytomer: shoot->phytomers ){
            phytomer->removeLeaf();
        }
    }
}

void PlantArchitecture::makePlantDormant( uint plantID ){
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::makePlantDormant): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for( auto& shoot: plant_instances.at(plantID).shoot_tree ){
        shoot->makeDormant();
    }
}

void PlantArchitecture::breakPlantDormancy( uint plantID ){
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::breakPlantDormancy): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for( auto& shoot: plant_instances.at(plantID).shoot_tree ){
        shoot->breakDormancy();
    }
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
            std::vector<uint> leaf_objIDs_flat = flatten(phytomer->leaf_objIDs);
            objIDs.insert(objIDs.end(), leaf_objIDs_flat.begin(), leaf_objIDs_flat.end() );
            std::vector<uint> inflorescence_objIDs_flat = flatten(phytomer->inflorescence_objIDs);
            objIDs.insert(objIDs.end(), inflorescence_objIDs_flat.begin(), inflorescence_objIDs_flat.end() );
            std::vector<uint> rachis_objIDs_flat = flatten(phytomer->peduncle_objIDs);
            objIDs.insert(objIDs.end(), rachis_objIDs_flat.begin(), rachis_objIDs_flat.end() );
        }
    }

    return objIDs;

}

std::vector<uint> PlantArchitecture::getAllPlantUUIDs(uint plantID) const{
    return context_ptr->getObjectPrimitiveUUIDs(getAllPlantObjectIDs(plantID));
}

std::vector<uint> PlantArchitecture::getPlantInternodeObjectIDs(uint plantID) const{

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInternodeObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for( auto &shoot : shoot_tree ){
        for( auto &phytomer : shoot->phytomers ){
            objIDs.insert(objIDs.end(), phytomer->internode_objIDs.begin(), phytomer->internode_objIDs.end() );
        }
    }

    return objIDs;

}

std::vector<uint> PlantArchitecture::getPlantPetioleObjectIDs(uint plantID) const{
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getPlantPetioleObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for( auto &shoot : shoot_tree ){
        for( auto &phytomer : shoot->phytomers ){
            for( auto &petiole : phytomer->petiole_objIDs ){
                objIDs.insert(objIDs.end(), petiole.begin(), petiole.end() );
            }
        }
    }

    return objIDs;

}

std::vector<uint> PlantArchitecture::getPlantLeafObjectIDs(uint plantID) const{
    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getPlantLeafObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for( auto &shoot : shoot_tree ){
        for( auto &phytomer : shoot->phytomers ){
            for( int petiole=0; petiole<phytomer->leaf_objIDs.size(); petiole++ ) {
                objIDs.insert(objIDs.end(), phytomer->leaf_objIDs.at(petiole).begin(), phytomer->leaf_objIDs.at(petiole).end());
            }
        }
    }

    return objIDs;

}

std::vector<uint> PlantArchitecture::getPlantPeduncleObjectIDs(uint plantID) const{

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getPlantPeduncleObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for( auto &shoot : shoot_tree ){
        for( auto &phytomer : shoot->phytomers ){
            for( auto &fbud : phytomer->peduncle_objIDs ){
                for( auto &peduncle : fbud ) {
                    objIDs.insert(objIDs.end(), peduncle.begin(), peduncle.end());
                }
            }
        }
    }

    return objIDs;

}

std::vector<uint> PlantArchitecture::getPlantFlowerObjectIDs(uint plantID) const{

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInflorescenceObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for( auto &shoot : shoot_tree ){
        for( auto &phytomer : shoot->phytomers ){
            for( int petiole=0; petiole<phytomer->floral_buds.size(); petiole++ ){
                for( int bud=0; bud<phytomer->floral_buds.at(petiole).size(); bud++ ){
                    if( phytomer->floral_buds.at(petiole).at(bud).state == BUD_FLOWER_OPEN || phytomer->floral_buds.at(petiole).at(bud).state == BUD_FLOWER_CLOSED ) {
                        objIDs.insert(objIDs.end(), phytomer->inflorescence_objIDs.at(petiole).at(bud).begin(), phytomer->inflorescence_objIDs.at(petiole).at(bud).end());
                    }
                }
            }
        }
    }

    return objIDs;


}

std::vector<uint> PlantArchitecture::getPlantFruitObjectIDs(uint plantID) const{

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInflorescenceObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for( auto &shoot : shoot_tree ){
        for( auto &phytomer : shoot->phytomers ){
            for( int petiole=0; petiole<phytomer->floral_buds.size(); petiole++ ){
                for( int bud=0; bud<phytomer->floral_buds.at(petiole).size(); bud++ ){
                    if( phytomer->floral_buds.at(petiole).at(bud).state == BUD_FRUITING ) {
                        objIDs.insert(objIDs.end(), phytomer->inflorescence_objIDs.at(petiole).at(bud).begin(), phytomer->inflorescence_objIDs.at(petiole).at(bud).end());
                    }
                }
            }
        }
    }

    return objIDs;


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

uint PlantArchitecture::duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, const AxisRotation &base_rotation, float current_age) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::duplicatePlantInstance): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto plant_shoot_tree = &plant_instances.at(plantID).shoot_tree;

    uint plantID_new = addPlantInstance(base_position, current_age);

    if( plant_shoot_tree->empty() ){ //no shoots to add
        return plantID_new;
    }
    if( plant_shoot_tree->front()->phytomers.empty() ){ //no phytomers to add
        return plantID_new;
    }

    for( auto shoot: *plant_shoot_tree ) {

        uint shootID_new; //ID of the new shoot; will be set once the shoot is created on the first loop iteration
        for (int node = 0; node < shoot->current_node_number; node++) {

            auto phytomer = shoot->phytomers.at(node);
            float internode_radius = phytomer->internode_radius_initial;
            float internode_length_max = phytomer->internode_length_max;
            float internode_scale_factor_fraction = phytomer->current_internode_scale_factor;
            float leaf_scale_factor_fraction = phytomer->current_leaf_scale_factor;

            if (node == 0) {//first phytomer on shoot
                AxisRotation original_base_rotation = shoot->base_rotation;
                if(shoot->parent_shoot_ID == -1 ) { //first shoot on plant
                    shootID_new = addBaseStemShoot(plantID_new, 1, original_base_rotation+base_rotation, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction, shoot->shoot_type_label);
                }else{ //child shoot
                    uint parent_node = plant_shoot_tree->at(shoot->parent_shoot_ID)->parent_node_index;
                    uint parent_petiole_index = 0;
                    for( auto &petiole : phytomer->vegetative_buds ) {
                        shootID_new = addChildShoot(plantID_new, shoot->parent_shoot_ID, parent_node, 1, original_base_rotation, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction,
                                                    shoot->shoot_type_label,parent_petiole_index);
                        parent_petiole_index++;
                    }
                }
            } else {
                //each phytomer needs to be added one-by-one to account for possible internodes/leaves that are not fully elongated
                addPhytomerToShoot(plantID_new, shootID_new, shoot_types.at(shoot->shoot_type_label).phytomer_parameters, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction);
            }

        }

    }

    return plantID_new;

}

void PlantArchitecture::deletePlantInstance(uint plantID){

    if( plant_instances.find(plantID) == plant_instances.end() ){
        return;
    }

    context_ptr->deleteObject(getAllPlantObjectIDs(plantID));

    plant_instances.erase(plantID);

}

void PlantArchitecture::deletePlantInstance( const std::vector<uint> &plantIDs ){

    for( uint ID : plantIDs ){
        deletePlantInstance(ID);
    }

}

void PlantArchitecture::setPlantPhenologicalThresholds(uint plantID, float time_to_dormancy_break, float time_to_flower_initiation, float time_to_flower_opening, float time_to_fruit_set, float time_to_fruit_maturity, float time_to_senescence) {

    if( plant_instances.find(plantID) == plant_instances.end() ){
        helios_runtime_error("ERROR (PlantArchitecture::setPlantPhenologicalThresholds): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).dd_to_dormancy_break = time_to_dormancy_break;
    plant_instances.at(plantID).dd_to_flower_initiation = time_to_flower_initiation;
    plant_instances.at(plantID).dd_to_flower_opening = time_to_flower_opening;
    plant_instances.at(plantID).dd_to_fruit_set = time_to_fruit_set;
    plant_instances.at(plantID).dd_to_fruit_maturity = time_to_fruit_maturity;
    plant_instances.at(plantID).dd_to_senescence = time_to_senescence;

}


void PlantArchitecture::advanceTime( float dt ) {

    for (auto &plant: plant_instances ){

        uint plantID = plant.first;
        PlantInstance& plant_instance = plant.second;

        auto shoot_tree = &plant_instance.shoot_tree;

        if( shoot_tree->empty() ){
            continue;
        }

        //\todo placeholder
        incrementAssimilatePool(plantID, -10);

        if( plant_instance.current_age > plant_instance.dd_to_senescence ){
            for (const auto& shoot : *shoot_tree) {
                shoot->makeDormant();
                shoot->assimilate_pool = 100;
                plant_instance.current_age = 0;
            }
            harvestPlant(plantID);
            std::cout << "Going dormant" << std::endl;
            continue;
        }

        //accounting for case of dt>phyllochron
        float phyllochron_min = shoot_tree->front()->shoot_parameters.phyllochron.val();
        for ( int i=1; i<shoot_tree->size(); i++ ){
            if( shoot_tree->at(i)->shoot_parameters.phyllochron.val() < phyllochron_min ){
                phyllochron_min = shoot_tree->at(i)->shoot_parameters.phyllochron.val();
            }
        }

        int Nsteps = std::floor(dt/phyllochron_min);
        float remainder_time = dt-phyllochron_min*float(Nsteps);
        if( remainder_time>0.f ){
            Nsteps++;
        }

        for( int timestep=0; timestep<Nsteps; timestep++ ) {

            float dt_max = phyllochron_min;
            if (timestep == Nsteps - 1 && remainder_time != 0.f ) {
                dt_max = remainder_time;
            }

            plant_instance.current_age += dt_max;

            size_t shoot_count = shoot_tree->size();
            for ( int i=0; i<shoot_count; i++ ){

                auto shoot = shoot_tree->at(i);

                // ****** PHENOLOGICAL TRANSITIONS ****** //

                // breaking dormancy
                bool dormancy_broken_this_timestep = false;
                if (shoot->dormancy_cycles >= 1 && shoot->dormant && plant_instance.current_age >= plant_instance.dd_to_dormancy_break) {
                    shoot->breakDormancy();
                    dormancy_broken_this_timestep = true;
                    shoot->assimilate_pool = 1e6;
//                std::cout << "Shoot " << shoot->ID << " breaking dormancy" << std::endl;
                }

                if (shoot->dormant) { //dormant, don't do anything
                    continue;
                }

                for (auto &phytomer: shoot->phytomers) {

                    if (!shoot->dormant) {
                        phytomer->time_since_dormancy += dt_max;
                    }

                    if (phytomer->floral_buds.empty()) { //no floral buds - skip this phytomer
                        continue;
                    }

                    for (auto &petiole: phytomer->floral_buds) {
                        for (auto &fbud: petiole) {

                            if (fbud.state != BUD_DORMANT && fbud.state != BUD_DEAD) {
                                fbud.time_counter += dt_max;
                            }

                            // -- Flowering -- //
                            if (shoot->shoot_parameters.phytomer_parameters.inflorescence.flower_prototype_function != nullptr) { //user defined a flower prototype function
                                // -- Flower initiation (closed flowers) -- //
                                if (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation >= 0.f) { //bud is active and flower initiation is enabled
                                    if ((!shoot->shoot_parameters.flowers_require_dormancy && plant_instance.current_age >= plant_instance.dd_to_flower_initiation) ||
                                        (shoot->shoot_parameters.flowers_require_dormancy && phytomer->time_since_dormancy >= plant_instance.dd_to_flower_initiation)) {
                                        fbud.time_counter = 0;
                                        if (context_ptr->randu() < shoot->shoot_parameters.flower_bud_break_probability.val() ) {
                                            phytomer->setFloralBudState(BUD_FLOWER_CLOSED, fbud);
                                        } else {
                                            phytomer->setFloralBudState(BUD_DEAD, fbud);
                                        }
                                        if (shoot->shoot_parameters.determinate_shoot_growth) {
                                            shoot->terminateApicalBud();
                                            shoot->terminateAxillaryVegetativeBuds();
                                        }
                                    }

                                    // -- Flower opening -- //
                                } else if ((fbud.state == BUD_FLOWER_CLOSED && plant_instance.dd_to_flower_opening >= 0.f) ||
                                           (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation < 0.f && plant_instance.dd_to_flower_opening >= 0.f)) {
                                    if (fbud.time_counter >= plant_instance.dd_to_flower_opening) {
                                        fbud.time_counter = 0;
                                        if( fbud.state == BUD_FLOWER_CLOSED ) {
                                            phytomer->setFloralBudState(BUD_FLOWER_OPEN, fbud);
                                        }else{
                                            if (context_ptr->randu() < shoot->shoot_parameters.flower_bud_break_probability.val() ) {
                                                phytomer->setFloralBudState(BUD_FLOWER_OPEN, fbud);
                                            } else {
                                                phytomer->setFloralBudState(BUD_DEAD, fbud);
                                            }
                                        }
                                        if (shoot->shoot_parameters.determinate_shoot_growth) {
                                            shoot->terminateApicalBud();
                                            shoot->terminateAxillaryVegetativeBuds();
                                        }
                                    }
                                }
                            }

                            // -- Fruit Set -- //
                            // If the flower bud is in a 'flowering' state, the fruit set occurs after a certain amount of time
                            if (shoot->shoot_parameters.phytomer_parameters.inflorescence.fruit_prototype_function != nullptr) {
                                if ((fbud.state == BUD_FLOWER_OPEN && plant_instance.dd_to_fruit_set >= 0.f) || //flower opened and fruit set is enabled
                                    (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation < 0.f && plant_instance.dd_to_flower_opening < 0.f && plant_instance.dd_to_fruit_set >= 0.f) || //jumped straight to fruit set with no flowering
                                    (fbud.state == BUD_FLOWER_CLOSED && plant_instance.dd_to_flower_opening < 0.f && plant_instance.dd_to_fruit_set >= 0.f)) { //jumped from closed flower to fruit set with no flower opening
                                    if (fbud.time_counter >= plant_instance.dd_to_fruit_set) {
                                        fbud.time_counter = 0;
                                        if (context_ptr->randu() < shoot->shoot_parameters.fruit_set_probability.val() ) {
                                            phytomer->setFloralBudState(BUD_FRUITING, fbud);
                                        } else {
                                            phytomer->setFloralBudState(BUD_DEAD, fbud);
                                        }
                                        if (shoot->shoot_parameters.determinate_shoot_growth) {
                                            shoot->terminateApicalBud();
                                            shoot->terminateAxillaryVegetativeBuds();
                                        }
                                    }
                                }
                            }

                        }
                    }

                }

                int node_index = 0;
                for (auto &phytomer: shoot->phytomers) {

                    // ****** GROWTH/SCALING OF CURRENT PHYTOMERS/FRUIT ****** //

                    //scale internode length
                    if (phytomer->current_internode_scale_factor < 1) {
                        float dL_internode = dt_max * shoot->shoot_parameters.elongation_rate.val() * phytomer->internode_length_max;
                        float length_scale = fmin(1.f, (phytomer->internode_length + dL_internode) / phytomer->internode_length_max);
                        setPhytomerInternodeLengthScaleFraction(plantID, shoot->ID, node_index, length_scale);
                    }

                    //scale internode girth
                    if( phytomer->internode_radii.front() < phytomer->internode_radius_max ) {
                        float dR = ( 1.f + dt_max * shoot->shoot_parameters.girth_growth_rate.val()/phytomer->internode_radii.front() );
                        incrementPhytomerInternodeGirth(plantID, shoot->ID, node_index, dR);
                    }

                    //scale petiole/leaves
                    if (phytomer->hasLeaf() && phytomer->current_leaf_scale_factor <= 1) {
                        float leaf_length = phytomer->current_leaf_scale_factor * phytomer->leaf_size_max.front();
                        float dL_leaf = dt_max * shoot->shoot_parameters.elongation_rate.val() * phytomer->leaf_size_max.front();
                        float scale = fmin(1.f, (leaf_length + dL_leaf) / phytomer->phytomer_parameters.leaf.prototype_scale.val() );
                        phytomer->setLeafScaleFraction(scale);
                    }

                    //Fruit Growth
                    for (auto &petiole: phytomer->floral_buds) {
                        for (auto &fbud: petiole) {

                            // If the floral bud it in a 'fruiting' state, the fruit grows with time
                            if (fbud.state == BUD_FRUITING && fbud.time_counter > 0) {
                                float scale = fmin(1, fbud.time_counter / plant_instance.dd_to_fruit_maturity);
                                phytomer->setInflorescenceScaleFraction(fbud, scale);
                            }
                        }
                    }

                    // ****** NEW CHILD SHOOTS FROM VEGETATIVE BUDS ****** //
                    uint parent_petiole_index = 0;
                    for (auto &petiole: phytomer->vegetative_buds) {
                        for (auto &vbud: petiole) {

                            if (vbud.state == BUD_ACTIVE && phytomer->age + dt_max > shoot->shoot_parameters.vegetative_bud_break_time.val()) {

                                ShootParameters *new_shoot_parameters = &shoot_types.at(vbud.shoot_type_label);
                                int parent_node_count = shoot->current_node_number;

                                float insertion_angle_adjustment = fmin(shoot->shoot_parameters.child_insertion_angle_tip.val() + shoot->shoot_parameters.child_insertion_angle_decay_rate.val() * float(parent_node_count - phytomer->shoot_index.x - 1),90.f);
                                AxisRotation base_rotation = make_AxisRotation(deg2rad(insertion_angle_adjustment), deg2rad(new_shoot_parameters->base_yaw.val()), deg2rad(new_shoot_parameters->base_roll.val()));
                                new_shoot_parameters->base_yaw.resample();

                                //scale the shoot internode length based on proximity from the tip
                                float internode_length_max;
                                if (new_shoot_parameters->growth_requires_dormancy) {
                                    internode_length_max = fmax(new_shoot_parameters->child_internode_length_max.val() - new_shoot_parameters->child_internode_length_decay_rate.val() * float(parent_node_count - phytomer->shoot_index.x - 1),
                                                                new_shoot_parameters->child_internode_length_min.val());
                                } else {
                                    internode_length_max = new_shoot_parameters->child_internode_length_max.val();
                                }

                                float internode_radius = shoot_types.at(vbud.shoot_type_label).internode_radius_initial.val();

//                                std::cout << "Adding child shoot of type " << vbud.shoot_type_label << std::endl;

                                uint childID = addChildShoot(plantID, shoot->ID, node_index, 1, base_rotation, internode_radius, internode_length_max, 0.01, 0.01, vbud.shoot_type_label, parent_petiole_index);

                                phytomer->setVegetativeBudState( BUD_DEAD, vbud );
                                vbud.shoot_ID = childID;
                                shoot_tree->at(childID)->dormant = false;

                            }

                        }
                        parent_petiole_index++;
                    }

                    // check for ground collisions
                    if( ground_clipping_height!=-99999  ){

                        // internode
                        if ( phytomer->shoot_index.x>0 && detectGroundCollision( phytomer->internode_objIDs ) ) {
                            context_ptr->deleteObject(phytomer->internode_objIDs);
                            phytomer->internode_objIDs.clear();
                        }

                        // leaves
                        for( uint petiole=0; petiole<phytomer->leaf_objIDs.size(); petiole++ ) {

                            if ( detectGroundCollision(phytomer->leaf_objIDs.at(petiole)) ) {
                                context_ptr->deleteObject(phytomer->leaf_objIDs.at(petiole));
                                phytomer->leaf_objIDs.at(petiole).clear();
                                phytomer->leaf_bases.at(petiole).clear();
                                context_ptr->deleteObject(phytomer->petiole_objIDs.at(petiole));
                                phytomer->petiole_objIDs.at(petiole).clear();
                            }

                        }

                        //inflorescence
                        for( uint petiole=0; petiole<phytomer->inflorescence_objIDs.size(); petiole++ ){
                            for( uint bud=0; bud<phytomer->inflorescence_objIDs.at(petiole).size(); bud++ ) {
                                for (uint fruit = 0; fruit < phytomer->inflorescence_objIDs.at(petiole).at(bud).size(); fruit++) {
                                    if (detectGroundCollision(phytomer->inflorescence_objIDs.at(petiole).at(bud).at(fruit))) {
                                        context_ptr->deleteObject(phytomer->inflorescence_objIDs.at(petiole).at(bud).at(fruit));
                                        phytomer->inflorescence_objIDs.at(petiole).at(bud).erase(phytomer->inflorescence_objIDs.at(petiole).at(bud).begin() + fruit);
                                    }
                                }
                            }
                        }

                    }



                    phytomer->age += dt_max;

                    if( phytomer->build_context_geometry_internode ) {
                        context_ptr->setObjectData(phytomer->internode_objIDs, "age", phytomer->age);
                    }
                    if( phytomer->build_context_geometry_petiole ) {
                        context_ptr->setObjectData(phytomer->petiole_objIDs, "age", phytomer->age);
                    }
                    context_ptr->setObjectData(phytomer->leaf_objIDs, "age", phytomer->age);

                    node_index++;
                }

                // if shoot has reached max_nodes, stop apical growth
                if (shoot->current_node_number >= shoot->shoot_parameters.max_nodes.val()) {
                    shoot->terminateApicalBud();
                }

                // If the apical bud is dead, don't do anything more with the shoot
                if (!shoot->meristem_is_alive) {
                    continue;
                }

                // ****** PHYLLOCHRON - NEW PHYTOMERS ****** //
                shoot->phyllochron_counter += dt_max;
                if (shoot->phyllochron_counter >= float(shoot->shoot_parameters.leaf_flush_count) * shoot->shoot_parameters.phyllochron.val()) {
                    float internode_radius = shoot->shoot_parameters.internode_radius_initial.val();
                    float internode_length_max = shoot->internode_length_max_shoot_initial;
                    for (int leaf = 0; leaf < shoot->shoot_parameters.leaf_flush_count; leaf++) {
                        addPhytomerToShoot(plantID, shoot->ID, shoot_types.at(shoot->shoot_type_label).phytomer_parameters, internode_radius, internode_length_max, 0.01, 0.01); //\todo These factors should be set to be consistent with the shoot
                    }
                    shoot->shoot_parameters.phyllochron.resample();
                    shoot->phyllochron_counter = 0;
                }

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

void PlantArchitecture::accumulateShootPhotosynthesis( float dt ){

    uint A_prim_data_missing = 0;

    for( auto &plant: plant_instances ){

        uint plantID = plant.first;
        auto shoot_tree = &plant.second.shoot_tree;

        for( auto &shoot: *shoot_tree ){

            float net_photosynthesis = 0;

            for( auto &phytomer: shoot->phytomers ){

                for( auto &leaf_objID: flatten(phytomer->leaf_objIDs) ){
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
        std::cerr << "WARNING (PlantArchitecture::accumulateShootPhotosynthesis): " << A_prim_data_missing << " leaf primitives were missing net_photosynthesis primitive data. Did you run the photosynthesis model?" << std::endl;
    }

}


std::vector<uint> makeTubeFromCones(uint radial_subdivisions, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr) {

    uint Nverts = vertices.size();

    if( radii.size()!=Nverts || colors.size()!=Nverts ){
        helios_runtime_error("ERROR (makeTubeFromCones): Length of vertex vectors is not consistent.");
    }

    std::vector<uint> objIDs(Nverts-1);

    for( uint v=0; v<Nverts-1; v++ ){

        objIDs.at(v) = context_ptr->addConeObject(radial_subdivisions, vertices.at(v), vertices.at(v + 1), radii.at(v), radii.at(v + 1), colors.at(v) );

    }

    return objIDs;

}

bool PlantArchitecture::detectGroundCollision(uint objID) {
    std::vector<uint> objIDs = {objID};
    return detectGroundCollision(objIDs);
}

bool PlantArchitecture::detectGroundCollision(const std::vector<uint> &objID) {

    for( uint ID : objID ){
        if( context_ptr->doesObjectExist(ID) ) {
            const std::vector<uint> &UUIDs = context_ptr->getObjectPrimitiveUUIDs(ID);
            for (uint UUID: UUIDs) {
                const std::vector<vec3> &vertices = context_ptr->getPrimitiveVertices(UUID);
                for (const vec3 &v: vertices) {
                    if (v.z < ground_clipping_height) {
                        return true;
                    }
                }
            }
        }
    }
    return false;

}
