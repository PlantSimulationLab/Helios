/** \file "PlantArchitecture.cpp" Primary source file for plant architecture plug-in.

    Copyright (C) 2016-2025 Brian Bailey

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

float PlantArchitecture::interpolateTube(const std::vector<float> &P, const float frac) {
    assert(frac >= 0 && frac <= 1);
    assert(!P.empty());

    float dl = 1.f / float(P.size() - 1);

    float f = 0;
    for (int i = 0; i < P.size() - 1; i++) {
        float fplus = f + dl;

        if (fplus >= 1.f) {
            fplus = 1.f + 1e-3;
        }

        if (frac >= f && (frac <= fplus || fabs(frac - fplus) < 0.0001)) {
            float V = P.at(i) + (frac - f) / (fplus - f) * (P.at(i + 1) - P.at(i));

            return V;
        }

        f = fplus;
    }

    return P.front();
}

vec3 PlantArchitecture::interpolateTube(const std::vector<vec3> &P, const float frac) {
    assert(frac >= 0 && frac <= 1);
    assert(!P.empty());

    float dl = 0.f;
    for (int i = 0; i < P.size() - 1; i++) {
        dl += (P.at(i + 1) - P.at(i)).magnitude();
    }

    float f = 0;
    for (int i = 0; i < P.size() - 1; i++) {
        float dseg = (P.at(i + 1) - P.at(i)).magnitude();

        float fplus = f + dseg / dl;

        if (fplus >= 1.f) {
            fplus = 1.f + 1e-3;
        }

        if (frac >= f && (frac <= fplus || fabs(frac - fplus) < 0.0001)) {
            vec3 V = P.at(i) + (frac - f) / (fplus - f) * (P.at(i + 1) - P.at(i));

            return V;
        }

        f = fplus;
    }

    return P.front();
}

PlantArchitecture::PlantArchitecture(helios::Context *context_ptr) : context_ptr(context_ptr) {
    generator = context_ptr->getRandomGenerator();

    output_object_data["age"] = false;
    output_object_data["rank"] = false;
    output_object_data["plantID"] = false;
    output_object_data["leafID"] = false;
    output_object_data["peduncleID"] = false;
    output_object_data["closedflowerID"] = false;
    output_object_data["openflowerID"] = false;
    output_object_data["fruitID"] = false;
    output_object_data["carbohydrate_concentration"] = false;
}

LeafPrototype::LeafPrototype(std::minstd_rand0 *generator) : generator(generator) {
    leaf_aspect_ratio.initialize(1.f, generator);
    midrib_fold_fraction.initialize(0.f, generator);
    longitudinal_curvature.initialize(0.f, generator);
    lateral_curvature.initialize(0.f, generator);
    petiole_roll.initialize(0.f, generator);
    wave_period.initialize(0.f, generator);
    wave_amplitude.initialize(0.f, generator);
    leaf_buckle_length.initialize(0.f, generator);
    leaf_buckle_angle.initialize(0.f, generator);
    subdivisions = 1;
    unique_prototypes = 1;
    leaf_offset = make_vec3(0, 0, 0);
    prototype_function = GenericLeafPrototype;
    build_petiolule = false;
    if (generator != nullptr) {
        sampleIdentifier();
    }
}

PhytomerParameters::PhytomerParameters() : PhytomerParameters(nullptr) {}

PhytomerParameters::PhytomerParameters(std::minstd_rand0 *generator) {
    //--- internode ---//
    internode.pitch.initialize(20, generator);
    internode.phyllotactic_angle.initialize(137.5, generator);
    internode.radius_initial.initialize(0.001, generator);
    internode.color = RGB::forestgreen;
    internode.length_segments = 1;
    internode.radial_subdivisions = 7;

    //--- petiole ---//
    petiole.petioles_per_internode = 1;
    petiole.pitch.initialize(90, generator);
    petiole.radius.initialize(0.001, generator);
    petiole.length.initialize(0.05, generator);
    petiole.curvature.initialize(0, generator);
    petiole.taper.initialize(0, generator);
    petiole.color = RGB::forestgreen;
    petiole.length_segments = 1;
    petiole.radial_subdivisions = 7;

    //--- leaf ---//
    leaf.leaves_per_petiole.initialize(1, generator);
    leaf.pitch.initialize(0, generator);
    leaf.yaw.initialize(0, generator);
    leaf.roll.initialize(0, generator);
    leaf.leaflet_offset.initialize(0, generator);
    leaf.leaflet_scale = 1;
    leaf.prototype_scale.initialize(0.05, generator);
    leaf.prototype = LeafPrototype(generator);

    //--- peduncle ---//
    peduncle.length.initialize(0.05, generator);
    peduncle.radius.initialize(0.001, generator);
    peduncle.pitch.initialize(0, generator);
    peduncle.roll.initialize(0, generator);
    peduncle.curvature.initialize(0, generator);
    petiole.color = RGB::forestgreen;
    peduncle.length_segments = 3;
    peduncle.radial_subdivisions = 7;

    //--- inflorescence ---//
    inflorescence.flowers_per_peduncle.initialize(1, generator);
    inflorescence.flower_offset.initialize(0, generator);
    inflorescence.pitch.initialize(0, generator);
    inflorescence.roll.initialize(0, generator);
    inflorescence.flower_prototype_scale.initialize(0.0075, generator);
    inflorescence.fruit_prototype_scale.initialize(0.0075, generator);
    inflorescence.fruit_gravity_factor_fraction.initialize(0, generator);
    inflorescence.unique_prototypes = 1;
}

ShootParameters::ShootParameters() : ShootParameters(nullptr) {}

ShootParameters::ShootParameters(std::minstd_rand0 *generator) {
    // ---- Geometric Parameters ---- //

    max_nodes.initialize(10, generator);

    max_nodes_per_season.initialize(9999, generator);

    insertion_angle_tip.initialize(20, generator);
    insertion_angle_decay_rate.initialize(0, generator);

    internode_length_max.initialize(0.02, generator);
    internode_length_min.initialize(0.002, generator);
    internode_length_decay_rate.initialize(0, generator);

    base_roll.initialize(0, generator);
    base_yaw.initialize(0, generator);

    gravitropic_curvature.initialize(0, generator);
    tortuosity.initialize(0, generator);

    // ---- Growth Parameters ---- //

    phyllochron_min.initialize(2, generator);

    elongation_rate_max.initialize(0.2, generator);
    girth_area_factor.initialize(0, generator);

    vegetative_bud_break_time.initialize(5, generator);
    vegetative_bud_break_probability_min.initialize(0, generator);
    vegetative_bud_break_probability_decay_rate.initialize(-0.5, generator);
    max_terminal_floral_buds.initialize(0, generator);
    flower_bud_break_probability.initialize(0, generator);
    fruit_set_probability.initialize(0, generator);

    flowers_require_dormancy = false;
    growth_requires_dormancy = false;

    determinate_shoot_growth = true;
}

void ShootParameters::defineChildShootTypes(const std::vector<std::string> &a_child_shoot_type_labels, const std::vector<float> &a_child_shoot_type_probabilities) {
    if (a_child_shoot_type_labels.size() != a_child_shoot_type_probabilities.size()) {
        helios_runtime_error("ERROR (ShootParameters::defineChildShootTypes): Child shoot type labels and probabilities must be the same size.");
    } else if (a_child_shoot_type_labels.empty()) {
        helios_runtime_error("ERROR (ShootParameters::defineChildShootTypes): Input argument vectors were empty.");
    } else if (sum(a_child_shoot_type_probabilities) != 1.f) {
        helios_runtime_error("ERROR (ShootParameters::defineChildShootTypes): Child shoot type probabilities must sum to 1.");
    }

    this->child_shoot_type_labels = a_child_shoot_type_labels;
    this->child_shoot_type_probabilities = a_child_shoot_type_probabilities;
}

std::vector<uint> PlantArchitecture::buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &plant_spacing_xy, const helios::int2 &plant_count_xy, const float age, const float germination_rate) {
    if (plant_count_xy.x <= 0 || plant_count_xy.y <= 0) {
        helios_runtime_error("ERROR (PlantArchitecture::buildPlantCanopyFromLibrary): Plant count must be greater than zero.");
    }

    vec2 canopy_extent(plant_spacing_xy.x * float(plant_count_xy.x - 1), plant_spacing_xy.y * float(plant_count_xy.y - 1));

    std::vector<uint> plantIDs;
    plantIDs.reserve(plant_count_xy.x * plant_count_xy.y);
    for (int j = 0; j < plant_count_xy.y; j++) {
        for (int i = 0; i < plant_count_xy.x; i++) {
            if (context_ptr->randu() < germination_rate) {
                plantIDs.push_back(buildPlantInstanceFromLibrary(canopy_center_position + make_vec3(-0.5f * canopy_extent.x + float(i) * plant_spacing_xy.x, -0.5f * canopy_extent.y + float(j) * plant_spacing_xy.y, 0), age));
            }
        }
    }

    return plantIDs;
}

std::vector<uint> PlantArchitecture::buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &canopy_extent_xy, const uint plant_count, const float age) {
    std::vector<uint> plantIDs;
    plantIDs.reserve(plant_count);
    for (int i = 0; i < plant_count; i++) {
        vec3 plant_origin = canopy_center_position + make_vec3((-0.5f + context_ptr->randu()) * canopy_extent_xy.x, (-0.5f + context_ptr->randu()) * canopy_extent_xy.y, 0);
        plantIDs.push_back(buildPlantInstanceFromLibrary(plant_origin, age));
    }

    return plantIDs;
}


void PlantArchitecture::defineShootType(const std::string &shoot_type_label, const ShootParameters &shoot_params) {
    if (this->shoot_types.find(shoot_type_label) != this->shoot_types.end()) { // shoot type already exists
        this->shoot_types.at(shoot_type_label) = shoot_params;
    } else {
        this->shoot_types.emplace(shoot_type_label, shoot_params);
    }
}

std::vector<helios::vec3> Phytomer::getInternodeNodePositions() const {
    std::vector<vec3> nodes = parent_shoot_ptr->shoot_internode_vertices.at(shoot_index.x);
    if (shoot_index.x > 0) {
        int p_minus = shoot_index.x - 1;
        int s_minus = parent_shoot_ptr->shoot_internode_vertices.at(p_minus).size() - 1;
        nodes.insert(nodes.begin(), parent_shoot_ptr->shoot_internode_vertices.at(p_minus).at(s_minus));
    }
    return nodes;
}

std::vector<float> Phytomer::getInternodeNodeRadii() const {
    std::vector<float> node_radii = parent_shoot_ptr->shoot_internode_radii.at(shoot_index.x);
    if (shoot_index.x > 0) {
        int p_minus = shoot_index.x - 1;
        int s_minus = parent_shoot_ptr->shoot_internode_radii.at(p_minus).size() - 1;
        node_radii.insert(node_radii.begin(), parent_shoot_ptr->shoot_internode_radii.at(p_minus).at(s_minus));
    }
    return node_radii;
}

helios::vec3 Phytomer::getInternodeAxisVector(const float stem_fraction) const { return getAxisVector(stem_fraction, getInternodeNodePositions()); }

helios::vec3 Phytomer::getPetioleAxisVector(const float stem_fraction, const uint petiole_index) const {
    if (petiole_index >= this->petiole_vertices.size()) {
        helios_runtime_error("ERROR (Phytomer::getPetioleAxisVector): Petiole index out of range.");
    }
    return getAxisVector(stem_fraction, this->petiole_vertices.at(petiole_index));
}

helios::vec3 Phytomer::getAxisVector(const float stem_fraction, const std::vector<helios::vec3> &axis_vertices) {
    assert(stem_fraction >= 0 && stem_fraction <= 1);

    float df = 0.1f;
    float frac_plus, frac_minus;
    if (stem_fraction + df <= 1) {
        frac_minus = stem_fraction;
        frac_plus = stem_fraction + df;
    } else {
        frac_minus = stem_fraction - df;
        frac_plus = stem_fraction;
    }

    const vec3 node_minus = PlantArchitecture::interpolateTube(axis_vertices, frac_minus);
    const vec3 node_plus = PlantArchitecture::interpolateTube(axis_vertices, frac_plus);

    vec3 norm = node_plus - node_minus;
    norm.normalize();

    return norm;
}

float Phytomer::getInternodeRadius() const { return parent_shoot_ptr->shoot_internode_radii.at(shoot_index.x).front(); }

float Phytomer::getInternodeLength() const {
    std::vector<vec3> node_vertices = this->getInternodeNodePositions();
    float length = 0;
    for (int i = 0; i < node_vertices.size() - 1; i++) {
        length += (node_vertices.at(i + 1) - node_vertices.at(i)).magnitude();
    }
    return length;
}

float Phytomer::getPetioleLength() const {
    // \todo
    return 0;
}

float Phytomer::getInternodeRadius(const float stem_fraction) const { return PlantArchitecture::interpolateTube(parent_shoot_ptr->shoot_internode_radii.at(shoot_index.x), stem_fraction); }

float Phytomer::getLeafArea() const {
    float leaf_area = 0;
    uint p = 0;
    for (auto &petiole: leaf_objIDs) {
        for (auto &leaf_objID: petiole) {
            if (context_ptr->doesObjectExist(leaf_objID)) {
                leaf_area += context_ptr->getObjectArea(leaf_objID) / powi(current_leaf_scale_factor.at(p), 2);
            }
        }
        p++;
    }

    return leaf_area;
}

helios::vec3 Phytomer::getLeafBasePosition(const uint petiole_index, const uint leaf_index) const {
#ifdef HELIOS_DEBUG
    if (petiole_index >= leaf_bases.size()) {
        helios_runtime_error("ERROR (Phytomer::getLeafBasePosition): Petiole index out of range.");
    } else if (leaf_index >= leaf_bases.at(petiole_index).size()) {
        helios_runtime_error("ERROR (Phytomer::getLeafBasePosition): Leaf index out of range.");
    }
#endif
    return leaf_bases.at(petiole_index).at(leaf_index);
}

void Phytomer::setVegetativeBudState(BudState state) {
    for (auto &petiole: axillary_vegetative_buds) {
        for (auto &bud: petiole) {
            bud.state = state;
        }
    }
}

void Phytomer::setVegetativeBudState(BudState state, uint petiole_index, uint bud_index) {
    if (petiole_index >= axillary_vegetative_buds.size()) {
        helios_runtime_error("ERROR (Phytomer::setVegetativeBudState): Petiole index out of range.");
    }
    if (bud_index >= axillary_vegetative_buds.at(petiole_index).size()) {
        helios_runtime_error("ERROR (Phytomer::setVegetativeBudState): Bud index out of range.");
    }
    setVegetativeBudState(state, axillary_vegetative_buds.at(petiole_index).at(bud_index));
}

void Phytomer::setVegetativeBudState(BudState state, VegetativeBud &vbud) { vbud.state = state; }

void Phytomer::setFloralBudState(BudState state) {
    for (auto &petiole: floral_buds) {
        for (auto &fbud: petiole) {
            if (!fbud.isterminal) {
                setFloralBudState(state, fbud);
            }
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

void Phytomer::setFloralBudState(BudState state, FloralBud &fbud) {
    // If state is already at the desired state, do nothing
    if (fbud.state == state) {
        return;
    } else if (state == BUD_DORMANT || state == BUD_ACTIVE) {
        fbud.state = state;
        return;
    }

    // Calculate carbon cost
    if (plantarchitecture_ptr->carbon_model_enabled) {
        if (state == BUD_FLOWER_CLOSED || (fbud.state == BUD_ACTIVE && state == BUD_FLOWER_OPEN)) { // state went from active to closed flower or open flower
            float flower_cost = calculateFlowerConstructionCosts(fbud);
            plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(this->parent_shoot_ID)->carbohydrate_pool_molC -= flower_cost;
        } else if (state == BUD_FRUITING) { // adding a fruit
            float fruit_cost = calculateFruitConstructionCosts(fbud);
            fbud.previous_fruit_scale_factor = fbud.current_fruit_scale_factor;
            if (plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(this->parent_shoot_ID)->carbohydrate_pool_molC > fruit_cost) {
                plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(this->parent_shoot_ID)->carbohydrate_pool_molC -= fruit_cost;
            } else {
                setFloralBudState(BUD_DEAD, fbud);
            }
        }
    }

    // Delete geometry from previous reproductive state (if present)
    context_ptr->deleteObject(fbud.inflorescence_objIDs);
    fbud.inflorescence_objIDs.resize(0);
    fbud.inflorescence_bases.resize(0);

    if (plantarchitecture_ptr->build_context_geometry_peduncle) {
        context_ptr->deleteObject(fbud.peduncle_objIDs);
        fbud.peduncle_objIDs.resize(0);
    }

    fbud.state = state;

    if (state != BUD_DEAD) { // add new reproductive organs

        updateInflorescence(fbud);
        fbud.time_counter = 0;
        if (fbud.state == BUD_FRUITING) {
            setInflorescenceScaleFraction(fbud, 0.25);
        }
    }
}

int Shoot::appendPhytomer(float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, const PhytomerParameters &phytomer_parameters) {
    auto shoot_tree_ptr = &plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree;

    // Determine the parent internode and petiole axes for rotation of the new phytomer
    vec3 parent_internode_axis;
    vec3 parent_petiole_axis;
    vec3 internode_base_position;
    if (phytomers.empty()) { // very first phytomer on shoot
        if (parent_shoot_ID == -1) { // very first shoot of the plant
            parent_internode_axis = make_vec3(0, 0, 1);
            parent_petiole_axis = make_vec3(0, -1, 0);
        } else { // first phytomer of a new shoot
            assert(parent_shoot_ID < shoot_tree_ptr->size() && parent_node_index < shoot_tree_ptr->at(parent_shoot_ID)->phytomers.size());
            parent_internode_axis = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->getInternodeAxisVector(1.f);
            parent_petiole_axis = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->getPetioleAxisVector(0.f, parent_petiole_index);
        }
        internode_base_position = base_position;
    } else { // additional phytomer being added to an existing shoot
        parent_internode_axis = phytomers.back()->getInternodeAxisVector(1.f);
        parent_petiole_axis = phytomers.back()->getPetioleAxisVector(0.f, 0);
        internode_base_position = shoot_internode_vertices.back().back();
    }

    std::shared_ptr<Phytomer> phytomer = std::make_shared<Phytomer>(phytomer_parameters, this, phytomers.size(), parent_internode_axis, parent_petiole_axis, internode_base_position, this->base_rotation, internode_radius, internode_length_max,
                                                                    internode_length_scale_factor_fraction, leaf_scale_factor_fraction, rank, plantarchitecture_ptr, context_ptr);
    shoot_tree_ptr->at(ID)->phytomers.push_back(phytomer);
    phytomer = shoot_tree_ptr->at(ID)->phytomers.back(); // change to point to phytomer stored in shoot

    // Initialize phytomer vegetative bud types and state
    for (auto &petiole: phytomer->axillary_vegetative_buds) {
        // sample the bud shoot type
        std::string child_shoot_type_label = sampleChildShootType();
        for (auto &vbud: petiole) {
            phytomer->setVegetativeBudState(BUD_DORMANT, vbud);
            vbud.shoot_type_label = child_shoot_type_label;

            // if the shoot type does not require dormancy, bud should be set to active
            if (!shoot_parameters.growth_requires_dormancy) {
                if (plantarchitecture_ptr->carbon_model_enabled) {
                    if (sampleVegetativeBudBreak_carb(phytomer->shoot_index.x)) { // randomly sample bud
                        phytomer->setVegetativeBudState(BUD_ACTIVE, vbud);
                    } else {
                        phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                    }
                } else {
                    if (sampleVegetativeBudBreak(phytomer->shoot_index.x)) { // randomly sample bud
                        phytomer->setVegetativeBudState(BUD_ACTIVE, vbud);
                    } else {
                        phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                    }
                }
            }
        }
    }

    // Initialize phytomer floral bud types and state
    uint petiole_index = 0;
    for (auto &petiole: phytomer->floral_buds) {
        uint bud_index = 0;
        for (auto &fbud: petiole) {
            // Set state of phytomer buds
            phytomer->setFloralBudState(BUD_DORMANT, fbud);

            // if the shoot type does not require dormancy, bud should be set to active
            if (!shoot_parameters.flowers_require_dormancy && fbud.state != BUD_DEAD) {
                phytomer->setFloralBudState(BUD_ACTIVE, fbud);
            }

            fbud.parent_index = petiole_index;
            fbud.bud_index = bud_index;

            bud_index++;
        }
        petiole_index++;
    }

    // Update the downstream leaf area for all upstream phytomers
    propagateDownstreamLeafArea(this, phytomer->shoot_index.x, phytomer->getLeafArea());

    // Set output object data 'age'
    phytomer->age = 0;
    if (plantarchitecture_ptr->build_context_geometry_internode && context_ptr->doesObjectExist(internode_tube_objID)) {
        //\todo This really only needs to be done once when the shoot is first created.
        if (plantarchitecture_ptr->output_object_data.at("age")) {
            context_ptr->setObjectData(internode_tube_objID, "age", phytomer->age);
        }
        if (plantarchitecture_ptr->output_object_data.at("rank")) {
            context_ptr->setObjectData(internode_tube_objID, "rank", rank);
        }
        if (plantarchitecture_ptr->output_object_data.at("plantID")) {
            context_ptr->setObjectData(internode_tube_objID, "plantID", (int) plantID);
        }
    }
    if (plantarchitecture_ptr->build_context_geometry_petiole) {
        if (plantarchitecture_ptr->output_object_data.at("age")) {
            context_ptr->setObjectData(phytomer->petiole_objIDs, "age", phytomer->age);
        }
        if (plantarchitecture_ptr->output_object_data.at("rank")) {
            context_ptr->setObjectData(phytomer->petiole_objIDs, "rank", phytomer->rank);
        }
        if (plantarchitecture_ptr->output_object_data.at("plantID")) {
            context_ptr->setObjectData(phytomer->petiole_objIDs, "plantID", (int) plantID);
        }
    }
    if (plantarchitecture_ptr->output_object_data.at("age")) {
        context_ptr->setObjectData(phytomer->leaf_objIDs, "age", phytomer->age);
    }
    if (plantarchitecture_ptr->output_object_data.at("rank")) {
        context_ptr->setObjectData(phytomer->leaf_objIDs, "rank", phytomer->rank);
    }
    if (plantarchitecture_ptr->output_object_data.at("plantID")) {
        context_ptr->setObjectData(phytomer->leaf_objIDs, "plantID", (int) plantID);
    }

    if (plantarchitecture_ptr->output_object_data.at("leafID")) {
        for (auto &petiole: phytomer->leaf_objIDs) {
            for (uint objID: petiole) {
                context_ptr->setObjectData(objID, "leafID", (int) objID);
            }
        }
    }

    if (phytomer_parameters.phytomer_creation_function != nullptr) {
        phytomer_parameters.phytomer_creation_function(phytomer, current_node_number, this->parent_node_index, shoot_parameters.max_nodes.val(), plantarchitecture_ptr->plant_instances.at(plantID).current_age);
    }

    // calculate fully expanded/elongated carbon costs
    if (plantarchitecture_ptr->carbon_model_enabled) {
        this->carbohydrate_pool_molC -= phytomer->calculatePhytomerConstructionCosts();
    }

    return (int) phytomers.size() - 1;
}

void Shoot::breakDormancy() {
    isdormant = false;

    int phytomer_ind = 0;
    for (auto &phytomer: phytomers) {
        for (auto &petiole: phytomer->floral_buds) {
            for (auto &fbud: petiole) {
                if (fbud.state != BUD_DEAD) {
                    phytomer->setFloralBudState(BUD_ACTIVE, fbud);
                }
                if (meristem_is_alive && fbud.isterminal) {
                    phytomer->setFloralBudState(BUD_ACTIVE, fbud);
                }
                fbud.time_counter = 0;
            }
        }
        for (auto &petiole: phytomer->axillary_vegetative_buds) {
            for (auto &vbud: petiole) {
                if (vbud.state != BUD_DEAD) {
                    if (plantarchitecture_ptr->carbon_model_enabled) {
                        if (sampleVegetativeBudBreak_carb(phytomer_ind)) {
                            // randomly sample bud
                            phytomer->setVegetativeBudState(BUD_ACTIVE, vbud);
                        } else {
                            phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                        }
                    } else {
                        if (sampleVegetativeBudBreak(phytomer_ind)) {
                            // randomly sample bud
                            phytomer->setVegetativeBudState(BUD_ACTIVE, vbud);
                        } else {
                            phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                        }
                    }
                }
            }
        }

        phytomer->isdormant = false;
        phytomer_ind++;
    }
}

void Shoot::makeDormant() {
    isdormant = true;
    dormancy_cycles++;
    nodes_this_season = 0;

    for (auto &phytomer: phytomers) {
        for (auto &petiole: phytomer->floral_buds) {
            // all currently active lateral buds die at dormancy
            for (auto &fbud: petiole) {
                if (fbud.state != BUD_DORMANT) {
                    phytomer->setFloralBudState(BUD_DEAD, fbud);
                }
            }
        }
        for (auto &petiole: phytomer->axillary_vegetative_buds) {
            for (auto &vbud: petiole) {
                if (vbud.state != BUD_DORMANT) {
                    phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                }
            }
        }
        if (!plantarchitecture_ptr->plant_instances.at(plantID).is_evergreen) {
            phytomer->removeLeaf();
        }
        phytomer->isdormant = true;
    }

    if (meristem_is_alive && shoot_parameters.flowers_require_dormancy && shoot_parameters.max_terminal_floral_buds.val() > 0) {
        addTerminalFloralBud();
    }
}

void Shoot::terminateApicalBud() {
    this->meristem_is_alive = false;
    this->phyllochron_counter = 0;
}

void Shoot::terminateAxillaryVegetativeBuds() {
    for (auto &phytomer: phytomers) {
        for (auto &petiole: phytomer->axillary_vegetative_buds) {
            for (auto &vbud: petiole) {
                phytomer->setVegetativeBudState(BUD_DEAD, vbud);
            }
        }
    }
}

void Shoot::addTerminalFloralBud() {
    int Nbuds = shoot_parameters.max_terminal_floral_buds.val();
    for (int bud = 0; bud < Nbuds; bud++) {
        FloralBud bud_new;
        bud_new.isterminal = true;
        bud_new.parent_index = 0;
        bud_new.bud_index = bud;
        bud_new.base_position = shoot_internode_vertices.back().back();
        float pitch_adjustment = 0;
        if (Nbuds > 1) {
            pitch_adjustment = deg2rad(30);
        }
        float yaw_adjustment = static_cast<float>(bud_new.bud_index) * 2.f * PI_F / float(Nbuds);
        //-0.25f * PI_F + bud_new.bud_index * 0.5f * PI_F / float(Nbuds);
        bud_new.base_rotation = make_AxisRotation(pitch_adjustment, yaw_adjustment, 0);
        bud_new.bending_axis = make_vec3(1, 0, 0);

        phytomers.back()->floral_buds.push_back({bud_new});
    }

    shoot_parameters.max_terminal_floral_buds.resample();
}

float Shoot::calculateShootInternodeVolume() const {
    float shoot_volume = 0;
    for (const auto &phytomer: phytomers) {
        if (context_ptr->doesObjectExist(internode_tube_objID)) {
            shoot_volume += context_ptr->getTubeObjectVolume(internode_tube_objID);
        }
    }
    return shoot_volume;
}

float Shoot::calculateShootLength() const {
    float shoot_length = 0;
    for (const auto &phytomer: phytomers) {
        shoot_length += phytomer->getInternodeLength();
    }
    return shoot_length;
}

void Shoot::updateShootNodes(bool update_context_geometry) {
    // make shoot origin consistent with parent shoot node position
    if (parent_shoot_ID >= 0) { // only if not the base shoot

        auto parent_shoot = plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID);

        const vec3 current_origin = shoot_internode_vertices.front().front();
        const vec3 updated_origin = parent_shoot->shoot_internode_vertices.at(this->parent_node_index).back();
        vec3 shift = updated_origin - current_origin;

        // shift shoot based outward by the radius of the parent internode
        //         shift += radial_outward_axis * parent_shoot->shoot_internode_radii.at(this->parent_node_index).back();

        if (shift != nullorigin) {
            for (auto &phytomer: shoot_internode_vertices) {
                for (vec3 &node: phytomer) {
                    node += shift;
                }
            }
        }
    }

    if (update_context_geometry && plantarchitecture_ptr->build_context_geometry_internode && context_ptr->doesObjectExist(internode_tube_objID)) {
        context_ptr->setTubeRadii(internode_tube_objID, flatten(shoot_internode_radii));
        context_ptr->setTubeNodes(internode_tube_objID, flatten(shoot_internode_vertices));
    }

    // update petiole/leaf positions
    for (int p = 0; p < phytomers.size(); p++) {
        vec3 petiole_base = shoot_internode_vertices.at(p).back();
        if (parent_shoot_ID >= 0) { // shift petiole base outward by the parent internode radius
            auto parent_shoot = plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID);
            //            petiole_base += radial_outward_axis * parent_shoot->shoot_internode_radii.at(this->parent_node_index).back();
        }
        phytomers.at(p)->setPetioleBase(petiole_base);
    }

    // update child shoot origins
    for (const auto &node: childIDs) {
        for (int child_shoot_ID: node.second) {
            plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->updateShootNodes(update_context_geometry);
        }
    }
}

helios::vec3 Shoot::getShootAxisVector(float shoot_fraction) const {
    uint phytomer_count = this->phytomers.size();

    uint phytomer_index = 0;
    if (shoot_fraction > 0) {
        phytomer_index = std::ceil(shoot_fraction * float(phytomer_count)) - 1;
    }

    assert(phytomer_index < phytomer_count);

    return this->phytomers.at(phytomer_index)->getInternodeAxisVector(0.5);
}

void Shoot::propagateDownstreamLeafArea(const Shoot *shoot, uint node_index, float leaf_area) {
    for (int i = node_index; i >= 0; i--) {
        shoot->phytomers.at(i)->downstream_leaf_area += leaf_area;
        shoot->phytomers.at(i)->downstream_leaf_area = std::max(0.f, shoot->phytomers.at(i)->downstream_leaf_area);
    }

    if (shoot->parent_shoot_ID >= 0) {
        Shoot *parent_shoot = plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(shoot->parent_shoot_ID).get();
        propagateDownstreamLeafArea(parent_shoot, shoot->parent_node_index, leaf_area);
    }
}

float Shoot::sumShootLeafArea(uint start_node_index) const {
    if (start_node_index >= phytomers.size()) {
        helios_runtime_error("ERROR (Shoot::sumShootLeafArea): Start node index out of range.");
    }

    float area = 0;

    for (uint p = start_node_index; p < phytomers.size(); p++) {
        // sum up leaves directly connected to this shoot
        auto phytomer = phytomers.at(p);
        for (auto &petiole: phytomer->leaf_objIDs) {
            for (uint objID: petiole) {
                if (context_ptr->doesObjectExist(objID)) {
                    area += context_ptr->getObjectArea(objID);
                }
            }
        }

        // call recursively for child shoots
        if (childIDs.find(p) != childIDs.end()) {
            for (int child_shoot_ID: childIDs.at(p)) {
                area += plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->sumShootLeafArea(0);
            }
        }
    }

    return area;
}


float Shoot::sumChildVolume(uint start_node_index) const {
    if (start_node_index >= phytomers.size()) {
        helios_runtime_error("ERROR (Shoot::sumChildVolume): Start node index out of range.");
    }

    float volume = 0;

    for (uint p = start_node_index; p < phytomers.size(); p++) {
        // call recursively for child shoots
        if (childIDs.find(p) != childIDs.end()) {
            for (int child_shoot_ID: childIDs.at(p)) {
                volume += plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->calculateShootInternodeVolume();
            }
        }
    }

    return volume;
}

Phytomer::Phytomer(const PhytomerParameters &params, Shoot *parent_shoot, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, helios::vec3 internode_base_origin,
                   const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, PlantArchitecture *plantarchitecture_ptr,
                   helios::Context *context_ptr) : rank(rank), context_ptr(context_ptr), plantarchitecture_ptr(plantarchitecture_ptr) {
    this->phytomer_parameters = params; // note this needs to be an assignment operation not a copy in order to re-randomize all the parameters

    ShootParameters parent_shoot_parameters = parent_shoot->shoot_parameters;

    this->internode_radius_initial = internode_radius;
    this->internode_length_max = internode_length_max;
    this->shoot_index = make_int3(phytomer_index, parent_shoot->current_node_number, parent_shoot_parameters.max_nodes.val());
    //.x is the index of the phytomer along the shoot, .y is the current number of phytomers on the parent shoot, .z is the maximum number of phytomers on the parent shoot.
    this->rank = parent_shoot->rank;
    this->plantID = parent_shoot->plantID;
    this->parent_shoot_ID = parent_shoot->ID;
    this->parent_shoot_ptr = parent_shoot;

    bool build_context_geometry_internode = plantarchitecture_ptr->build_context_geometry_internode;
    bool build_context_geometry_petiole = plantarchitecture_ptr->build_context_geometry_petiole;
    bool build_context_geometry_peduncle = plantarchitecture_ptr->build_context_geometry_peduncle;

    //    if( internode_radius==0.f || internode_length_max==0.f || parent_shoot_parameters.internode_radius_max.val()==0.f ){
    //        build_context_geometry_internode = false;
    //    }

    // Number of longitudinal segments for internode and petiole
    // if Ndiv=0, use Ndiv=1 (but don't add any primitives to Context)
    uint Ndiv_internode_length = std::max(uint(1), phytomer_parameters.internode.length_segments);
    uint Ndiv_internode_radius = std::max(uint(3), phytomer_parameters.internode.radial_subdivisions);
    uint Ndiv_petiole_length = std::max(uint(1), phytomer_parameters.petiole.length_segments);
    uint Ndiv_petiole_radius = std::max(uint(3), phytomer_parameters.petiole.radial_subdivisions);

    // Flags to determine whether internode geometry should be built in the Context. Not building all geometry can save memory and computation time.
    if (phytomer_parameters.internode.length_segments == 0 || phytomer_parameters.internode.radial_subdivisions < 3) {
        build_context_geometry_internode = false;
    }
    if (phytomer_parameters.petiole.length_segments == 0 || phytomer_parameters.petiole.radial_subdivisions < 3) {
        build_context_geometry_petiole = false;
    }

    if (phytomer_parameters.petiole.petioles_per_internode < 1) {
        build_context_geometry_petiole = false;
        phytomer_parameters.petiole.petioles_per_internode = 1;
        phytomer_parameters.leaf.leaves_per_petiole = 0;
    }

    if (phytomer_parameters.petiole.petioles_per_internode == 0) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Number of petioles per internode must be greater than zero.");
    }

    current_internode_scale_factor = internode_length_scale_factor_fraction;
    current_leaf_scale_factor.resize(phytomer_parameters.petiole.petioles_per_internode);
    std::fill(current_leaf_scale_factor.begin(), current_leaf_scale_factor.end(), leaf_scale_factor_fraction);

    if (internode_radius == 0.f) {
        internode_radius = 1e-5;
    }

    // Initialize internode variables
    float internode_length = internode_length_scale_factor_fraction * internode_length_max;
    float dr_internode = internode_length / float(phytomer_parameters.internode.length_segments);
    float dr_internode_max = internode_length_max / float(phytomer_parameters.internode.length_segments);
    std::vector<vec3> phytomer_internode_vertices;
    std::vector<float> phytomer_internode_radii;
    phytomer_internode_vertices.resize(Ndiv_internode_length + 1);
    phytomer_internode_vertices.at(0) = internode_base_origin;
    phytomer_internode_radii.resize(Ndiv_internode_length + 1);
    phytomer_internode_radii.at(0) = internode_radius;
    internode_pitch = deg2rad(phytomer_parameters.internode.pitch.val());
    phytomer_parameters.internode.pitch.resample();
    internode_phyllotactic_angle = deg2rad(phytomer_parameters.internode.phyllotactic_angle.val());
    phytomer_parameters.internode.phyllotactic_angle.resample();

    // initialize petiole variables
    petiole_length.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_vertices.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_radii.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_pitch.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_curvature.resize(phytomer_parameters.petiole.petioles_per_internode);
    std::vector<float> dr_petiole(phytomer_parameters.petiole.petioles_per_internode);
    std::vector<float> dr_petiole_max(phytomer_parameters.petiole.petioles_per_internode);
    for (int p = 0; p < phytomer_parameters.petiole.petioles_per_internode; p++) {
        petiole_vertices.at(p).resize(Ndiv_petiole_length + 1);
        petiole_radii.at(p).resize(Ndiv_petiole_length + 1);

        petiole_length.at(p) = leaf_scale_factor_fraction * phytomer_parameters.petiole.length.val();
        dr_petiole.at(p) = petiole_length.at(p) / float(phytomer_parameters.petiole.length_segments);
        dr_petiole_max.at(p) = phytomer_parameters.petiole.length.val() / float(phytomer_parameters.petiole.length_segments);

        petiole_radii.at(p).at(0) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val();
    }
    phytomer_parameters.petiole.length.resample();
    if (build_context_geometry_petiole) {
        petiole_objIDs.resize(phytomer_parameters.petiole.petioles_per_internode);
    }

    // initialize leaf variables
    leaf_bases.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_objIDs.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_size_max.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_rotation.resize(phytomer_parameters.petiole.petioles_per_internode);
    int leaves_per_petiole = phytomer_parameters.leaf.leaves_per_petiole.val();
    phytomer_parameters.leaf.leaves_per_petiole.resample();
    for (uint petiole = 0; petiole < phytomer_parameters.petiole.petioles_per_internode; petiole++) {
        leaf_size_max.at(petiole).resize(leaves_per_petiole);
        leaf_rotation.at(petiole).resize(leaves_per_petiole);
    }

    internode_colors.resize(Ndiv_internode_length + 1);
    internode_colors.at(0) = phytomer_parameters.internode.color;
    petiole_colors.resize(Ndiv_petiole_length + 1);
    petiole_colors.at(0) = phytomer_parameters.petiole.color;

    vec3 internode_axis = parent_internode_axis;

    vec3 petiole_rotation_axis = cross(parent_internode_axis, parent_petiole_axis);
    if (petiole_rotation_axis == make_vec3(0, 0, 0)) {
        petiole_rotation_axis = make_vec3(1, 0, 0);
    }

    if (phytomer_index == 0) { // if this is the first phytomer along a shoot, apply the origin rotation about the parent axis

        // internode pitch rotation for phytomer base
        if (internode_pitch != 0.f) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, 0.5f * internode_pitch);
        }

        // internode roll rotation for shoot base rotation
        float roll_nudge = 0.f;
        //\todo Not clear if this is still needed. It causes problems when you want to plant base roll to be exactly 0.
        //        if( shoot_base_rotation.roll/180.f == floor(shoot_base_rotation.roll/180.f) ) {
        //            roll_nudge = 0.2;
        //        }
        if (shoot_base_rotation.roll != 0.f || roll_nudge != 0.f) {
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll + roll_nudge); // small additional rotation is to make sure the petiole is not exactly vertical
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll + roll_nudge);
        }

        vec3 base_pitch_axis = -1 * cross(parent_internode_axis, parent_petiole_axis);

        // internode pitch rotation for shoot base rotation
        if (shoot_base_rotation.pitch != 0.f) {
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
        }

        // internode yaw rotation for shoot base rotation
        if (shoot_base_rotation.yaw != 0) {
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
        }

        parent_shoot->radial_outward_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, 0.5f * PI_F);

        //        if( parent_shoot->parent_shoot_ID>=0 ) { //if this is not the first shoot on the plant (i.e. it has a parent shoot
        //            auto parent_of_parent_shoot = plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(parent_shoot->parent_shoot_ID);
        //            phytomer_internode_vertices.at(0) += parent_shoot->radial_outward_axis * parent_of_parent_shoot->shoot_internode_radii.at(parent_shoot->parent_node_index).back();
        //        }
    } else {
        // internode pitch rotation for phytomer base
        if (internode_pitch != 0) {
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, -1.25f * internode_pitch);
        }
    }

    vec3 shoot_bending_axis = cross(internode_axis, make_vec3(0, 0, 1));

    internode_axis.normalize();
    if (internode_axis == make_vec3(0, 0, 1)) {
        shoot_bending_axis = make_vec3(0, 1, 0);
    }

    // create internode tube
    for (int inode_segment = 1; inode_segment <= Ndiv_internode_length; inode_segment++) {
        // apply curvature and tortuosity
        if ((fabs(parent_shoot->gravitropic_curvature) > 0 || parent_shoot_parameters.tortuosity.val() > 0) && shoot_index.x > 0) {
            // note: curvature is not applied to the first phytomer because if scaling is performed in the phytomer creation function it messes things up

            float current_curvature_fact = 0.5f - internode_axis.z / 2.f;
            if (internode_axis.z < 0) {
                current_curvature_fact *= 2.f;
            }

            float dt = dr_internode_max / float(Ndiv_internode_length);

            parent_shoot->curvature_perturbation += -0.5f * parent_shoot->curvature_perturbation * dt + parent_shoot_parameters.tortuosity.val() * context_ptr->randn() * sqrt(dt);
            float curvature_angle = deg2rad((parent_shoot->gravitropic_curvature * current_curvature_fact * dr_internode_max + parent_shoot->curvature_perturbation));
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, shoot_bending_axis, curvature_angle);

            parent_shoot->yaw_perturbation += -0.5f * parent_shoot->yaw_perturbation * dt + parent_shoot_parameters.tortuosity.val() * context_ptr->randn() * sqrt(dt);
            float yaw_angle = deg2rad((parent_shoot->yaw_perturbation));
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, make_vec3(0, 0, 1), yaw_angle);
        }

        phytomer_internode_vertices.at(inode_segment) = phytomer_internode_vertices.at(inode_segment - 1) + dr_internode * internode_axis;

        phytomer_internode_radii.at(inode_segment) = internode_radius;
        internode_colors.at(inode_segment) = phytomer_parameters.internode.color;
    }

    if (shoot_index.x == 0) { // first phytomer on shoot
        parent_shoot_ptr->shoot_internode_vertices.push_back(phytomer_internode_vertices);
        parent_shoot_ptr->shoot_internode_radii.push_back(phytomer_internode_radii);
    } else { // if not the first phytomer on shoot, don't insert the first node because it's already defined on the previous phytomer
        parent_shoot_ptr->shoot_internode_vertices.emplace_back(phytomer_internode_vertices.begin() + 1, phytomer_internode_vertices.end());
        parent_shoot_ptr->shoot_internode_radii.emplace_back(phytomer_internode_radii.begin() + 1, phytomer_internode_radii.end());
    }

    // build internode context geometry
    if (build_context_geometry_internode) {
        // calculate texture coordinates
        float texture_repeat_length = 0.25f; // meters
        float length = 0; // shoot length prior to this phytomer
        for (auto &phytomer: parent_shoot_ptr->phytomers) {
            length += phytomer->internode_length_max;
        }
        std::vector<float> uv_y(phytomer_internode_vertices.size());
        float dy = internode_length_max / float(uv_y.size() - 1);
        for (int j = 0; j < uv_y.size(); j++) {
            uv_y.at(j) = (length + j * dy) / texture_repeat_length - std::floor((length + j * dy) / texture_repeat_length);
        }

        if (!context_ptr->doesObjectExist(parent_shoot->internode_tube_objID)) { // first internode on shoot
            if (!phytomer_parameters.internode.image_texture.empty()) {
                parent_shoot->internode_tube_objID = context_ptr->addTubeObject(Ndiv_internode_radius, phytomer_internode_vertices, phytomer_internode_radii, phytomer_parameters.internode.image_texture.c_str(), uv_y);
            } else {
                parent_shoot->internode_tube_objID = context_ptr->addTubeObject(Ndiv_internode_radius, phytomer_internode_vertices, phytomer_internode_radii, internode_colors);
            }
            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(parent_shoot->internode_tube_objID), "object_label", "shoot");
        } else { // appending internode to shoot
            for (int inode_segment = 1; inode_segment <= Ndiv_internode_length; inode_segment++) {
                if (!phytomer_parameters.internode.image_texture.empty()) {
                    context_ptr->appendTubeSegment(parent_shoot->internode_tube_objID, phytomer_internode_vertices.at(inode_segment), phytomer_internode_radii.at(inode_segment), phytomer_parameters.internode.image_texture.c_str(),
                                                   {uv_y.at(inode_segment - 1), uv_y.at(inode_segment)});
                } else {
                    context_ptr->appendTubeSegment(parent_shoot->internode_tube_objID, phytomer_internode_vertices.at(inode_segment), phytomer_internode_radii.at(inode_segment), internode_colors.at(inode_segment));
                }
            }
            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(parent_shoot->internode_tube_objID), "object_label", "shoot");
        }
    }

    //--- create petiole ---//

    for (int petiole = 0; petiole < phytomer_parameters.petiole.petioles_per_internode; petiole++) { // looping over petioles

        vec3 petiole_axis = internode_axis;

        // petiole pitch rotation
        petiole_pitch.at(petiole) = deg2rad(phytomer_parameters.petiole.pitch.val());
        phytomer_parameters.petiole.pitch.resample();
        if (fabs(petiole_pitch.at(petiole)) < deg2rad(5.f)) {
            petiole_pitch.at(petiole) = deg2rad(5.f);
        }
        petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, petiole_rotation_axis, std::abs(petiole_pitch.at(petiole)));

        // petiole yaw rotation
        if (phytomer_index != 0 && internode_phyllotactic_angle != 0) { // not first phytomer along shoot
            petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, internode_axis, internode_phyllotactic_angle);
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, internode_axis, internode_phyllotactic_angle);
        }

        // petiole curvature
        petiole_curvature.at(petiole) = phytomer_parameters.petiole.curvature.val();
        phytomer_parameters.petiole.curvature.resample();

        vec3 petiole_rotation_axis_actual = petiole_rotation_axis;
        vec3 petiole_axis_actual = petiole_axis;

        if (petiole > 0) {
            float budrot = float(petiole) * 2.f * PI_F / float(phytomer_parameters.petiole.petioles_per_internode);
            petiole_axis_actual = rotatePointAboutLine(petiole_axis_actual, nullorigin, internode_axis, budrot);
            petiole_rotation_axis_actual = rotatePointAboutLine(petiole_rotation_axis_actual, nullorigin, internode_axis, budrot);
        }

        petiole_vertices.at(petiole).at(0) = phytomer_internode_vertices.back();

        for (int j = 1; j <= Ndiv_petiole_length; j++) {
            if (fabs(petiole_curvature.at(petiole)) > 0) {
                petiole_axis_actual = rotatePointAboutLine(petiole_axis_actual, nullorigin, petiole_rotation_axis_actual, -deg2rad(petiole_curvature.at(petiole) * dr_petiole_max.at(petiole)));
            }

            petiole_vertices.at(petiole).at(j) = petiole_vertices.at(petiole).at(j - 1) + dr_petiole.at(petiole) * petiole_axis_actual;

            petiole_radii.at(petiole).at(j) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val() * (1.f - phytomer_parameters.petiole.taper.val() / float(Ndiv_petiole_length) * float(j));
            petiole_colors.at(j) = phytomer_parameters.petiole.color;

            assert( !std::isnan(petiole_vertices.at(petiole).at(j).x) && std::isfinite(petiole_vertices.at(petiole).at(j).x) );
            assert( !std::isnan(petiole_radii.at(petiole).at(j)) && std::isfinite(petiole_radii.at(petiole).at(j)) );
        }

        if (build_context_geometry_petiole && petiole_radii.at(petiole).front() > 0.f) {
            petiole_objIDs.at(petiole) = makeTubeFromCones(Ndiv_petiole_radius, petiole_vertices.at(petiole), petiole_radii.at(petiole), petiole_colors, context_ptr);
            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(petiole_objIDs.at(petiole)), "object_label", "petiole");
        }

        //--- create buds ---//

        std::vector<VegetativeBud> vegetative_buds_new;
        vegetative_buds_new.resize(phytomer_parameters.internode.max_vegetative_buds_per_petiole.val());
        phytomer_parameters.internode.max_vegetative_buds_per_petiole.resample();

        axillary_vegetative_buds.push_back(vegetative_buds_new);

        std::vector<FloralBud> floral_buds_new;
        floral_buds_new.resize(phytomer_parameters.internode.max_floral_buds_per_petiole.val());
        phytomer_parameters.internode.max_floral_buds_per_petiole.resample();

        uint index = 0;
        for (auto &fbud: floral_buds_new) {
            fbud.bud_index = index;
            fbud.parent_index = petiole;
            float pitch_adjustment = fbud.bud_index * 0.1f * PI_F / float(axillary_vegetative_buds.size());
            float yaw_adjustment = -0.25f * PI_F + fbud.bud_index * 0.5f * PI_F / float(axillary_vegetative_buds.size());
            fbud.base_rotation = make_AxisRotation(pitch_adjustment, yaw_adjustment, 0);
            fbud.base_position = phytomer_internode_vertices.back();
            fbud.bending_axis = shoot_bending_axis;
            index++;
        }

        floral_buds.push_back(floral_buds_new);

        //--- create leaves ---//

        if (phytomer_parameters.leaf.prototype.prototype_function == nullptr) {
            helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Leaf prototype function was not defined for shoot type " + parent_shoot->shoot_type_label + ".");
        }

        vec3 petiole_tip_axis = getPetioleAxisVector(1.f, petiole);

        // Create unique leaf prototypes for each shoot type so we can simply copy them for each leaf
        assert(phytomer_parameters.leaf.prototype.unique_prototype_identifier != 0);
        if (phytomer_parameters.leaf.prototype.unique_prototypes > 0 &&
            plantarchitecture_ptr->unique_leaf_prototype_objIDs.find(phytomer_parameters.leaf.prototype.unique_prototype_identifier) == plantarchitecture_ptr->unique_leaf_prototype_objIDs.end()) {
            plantarchitecture_ptr->unique_leaf_prototype_objIDs[phytomer_parameters.leaf.prototype.unique_prototype_identifier].resize(phytomer_parameters.leaf.prototype.unique_prototypes);
            for (int prototype = 0; prototype < phytomer_parameters.leaf.prototype.unique_prototypes; prototype++) {
                for (int leaf = 0; leaf < leaves_per_petiole; leaf++) {
                    float ind_from_tip = float(leaf) - float(leaves_per_petiole - 1) / 2.f;
                    uint objID_leaf = phytomer_parameters.leaf.prototype.prototype_function(context_ptr, &phytomer_parameters.leaf.prototype, ind_from_tip);
                    if (phytomer_parameters.leaf.prototype.prototype_function == GenericLeafPrototype) {
                        context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(objID_leaf), "object_label", "leaf");
                    }
                    plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).push_back(objID_leaf);
                    std::vector<uint> petiolule_UUIDs = context_ptr->filterPrimitivesByData(context_ptr->getObjectPrimitiveUUIDs(objID_leaf), "object_label", "petiolule");
                    context_ptr->setPrimitiveColor(petiolule_UUIDs, phytomer_parameters.petiole.color);
                    context_ptr->hideObject(objID_leaf);
                }
            }
        }

        for (int leaf = 0; leaf < leaves_per_petiole; leaf++) {
            float ind_from_tip = float(leaf) - float(leaves_per_petiole - 1) / 2.f;

            uint objID_leaf;
            if (phytomer_parameters.leaf.prototype.unique_prototypes > 0) { // copy the existing prototype
                int prototype = context_ptr->randu(0, phytomer_parameters.leaf.prototype.unique_prototypes - 1);
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.find(phytomer_parameters.leaf.prototype.unique_prototype_identifier) != plantarchitecture_ptr->unique_leaf_prototype_objIDs.end());
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).size() > prototype);
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).size() > leaf);
                objID_leaf = context_ptr->copyObject(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).at(leaf));
            } else { // load a new prototype
                objID_leaf = phytomer_parameters.leaf.prototype.prototype_function(context_ptr, &phytomer_parameters.leaf.prototype, ind_from_tip);
            }

            // -- leaf scaling -- //

            if (leaves_per_petiole > 0 && phytomer_parameters.leaf.leaflet_scale.val() != 1.f && ind_from_tip != 0) {
                leaf_size_max.at(petiole).at(leaf) = powf(phytomer_parameters.leaf.leaflet_scale.val(), fabs(ind_from_tip)) * phytomer_parameters.leaf.prototype_scale.val();
            } else {
                leaf_size_max.at(petiole).at(leaf) = phytomer_parameters.leaf.prototype_scale.val();
            }
            vec3 leaf_scale = leaf_scale_factor_fraction * leaf_size_max.at(petiole).at(leaf) * make_vec3(1, 1, 1);

            context_ptr->scaleObject(objID_leaf, leaf_scale);

            float compound_rotation = 0;
            if (leaves_per_petiole > 1) {
                if (phytomer_parameters.leaf.leaflet_offset.val() == 0) {
                    float dphi = PI_F / (floor(0.5 * float(leaves_per_petiole - 1)) + 1);
                    compound_rotation = -float(PI_F) + dphi * (leaf + 0.5f);
                } else {
                    if (leaf == float(leaves_per_petiole - 1) / 2.f) { // tip leaf
                        compound_rotation = 0;
                    } else if (leaf < float(leaves_per_petiole - 1) / 2.f) {
                        compound_rotation = -0.5 * PI_F;
                    } else {
                        compound_rotation = 0.5 * PI_F;
                    }
                }
            }

            // -- leaf rotations -- //

            // leaf roll rotation
            float roll_rot = 0;
            if (leaves_per_petiole == 1) {
                int sign = (shoot_index.x % 2 == 0) ? 1 : -1;
                roll_rot = (acos_safe(internode_axis.z) - deg2rad(phytomer_parameters.leaf.roll.val())) * sign;
            } else if (ind_from_tip != 0) {
                roll_rot = (asin_safe(petiole_tip_axis.z) + deg2rad(phytomer_parameters.leaf.roll.val())) * compound_rotation / std::fabs(compound_rotation);
            }
            leaf_rotation.at(petiole).at(leaf).roll = deg2rad(phytomer_parameters.leaf.roll.val());
            phytomer_parameters.leaf.roll.resample();
            context_ptr->rotateObject(objID_leaf, roll_rot, "x");

            // leaf pitch rotation
            leaf_rotation.at(petiole).at(leaf).pitch = deg2rad(phytomer_parameters.leaf.pitch.val());
            float pitch_rot = leaf_rotation.at(petiole).at(leaf).pitch;
            phytomer_parameters.leaf.pitch.resample();
            if (ind_from_tip == 0) {
                pitch_rot += asin_safe(petiole_tip_axis.z);
            }
            context_ptr->rotateObject(objID_leaf, -pitch_rot, "y");

            // leaf yaw rotation
            if (ind_from_tip != 0) {
                float sign = -compound_rotation / fabs(compound_rotation);
                leaf_rotation.at(petiole).at(leaf).yaw = sign * deg2rad(phytomer_parameters.leaf.yaw.val());
                float yaw_rot = leaf_rotation.at(petiole).at(leaf).yaw;
                phytomer_parameters.leaf.yaw.resample();
                context_ptr->rotateObject(objID_leaf, yaw_rot, "z");
            } else {
                leaf_rotation.at(petiole).at(leaf).yaw = 0;
            }

            // rotate leaf to azimuth of petiole
            context_ptr->rotateObject(objID_leaf, -std::atan2(petiole_tip_axis.y, petiole_tip_axis.x) + compound_rotation, "z");


            // -- leaf translation -- //

            vec3 leaf_base = petiole_vertices.at(petiole).back();
            if (leaves_per_petiole > 1 && phytomer_parameters.leaf.leaflet_offset.val() > 0) {
                if (ind_from_tip != 0) {
                    float offset = (fabs(ind_from_tip) - 0.5f) * phytomer_parameters.leaf.leaflet_offset.val() * phytomer_parameters.petiole.length.val();
                    leaf_base = PlantArchitecture::interpolateTube(petiole_vertices.at(petiole), 1.f - offset / phytomer_parameters.petiole.length.val());
                }
            }

            context_ptr->translateObject(objID_leaf, leaf_base);

            leaf_objIDs.at(petiole).push_back(objID_leaf);
            leaf_bases.at(petiole).push_back(leaf_base);
        }
        phytomer_parameters.leaf.prototype_scale.resample();

        if (petiole_axis_actual == make_vec3(0, 0, 1)) {
            inflorescence_bending_axis = make_vec3(1, 0, 0);
        } else {
            inflorescence_bending_axis = cross(make_vec3(0, 0, 1), petiole_axis_actual);
        }
    }
}

float Phytomer::calculatePhytomerVolume(uint node_number) const {
    // Get the radii of this phytomer from the parent shoot
    const auto &segment = parent_shoot_ptr->shoot_internode_radii.at(node_number);

    // Find the average radius
    float avg_radius = 0.0f;
    for (float radius: segment) {
        avg_radius += radius;
    }
    avg_radius /= scast<float>(segment.size());

    // Get the length of the phytomer
    float length = getInternodeLength();

    // Calculate the volume of the cylinder
    float volume = PI_F * avg_radius * avg_radius * length;

    return volume;
}

void Phytomer::updateInflorescence(FloralBud &fbud) {
    bool build_context_geometry_peduncle = plantarchitecture_ptr->build_context_geometry_peduncle;

    uint Ndiv_peduncle_length = std::max(uint(1), phytomer_parameters.peduncle.length_segments);
    uint Ndiv_peduncle_radius = std::max(uint(3), phytomer_parameters.peduncle.radial_subdivisions);
    if (phytomer_parameters.peduncle.length_segments == 0 || phytomer_parameters.peduncle.radial_subdivisions < 3) {
        build_context_geometry_peduncle = false;
    }

    float dr_peduncle = phytomer_parameters.peduncle.length.val() / float(Ndiv_peduncle_length);
    phytomer_parameters.peduncle.length.resample();

    std::vector<vec3> peduncle_vertices(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_vertices.at(0) = fbud.base_position;
    std::vector<float> peduncle_radii(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_radii.at(0) = phytomer_parameters.peduncle.radius.val();
    std::vector<RGBcolor> peduncle_colors(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_colors.at(0) = phytomer_parameters.peduncle.color;

    vec3 peduncle_axis = getAxisVector(1.f, getInternodeNodePositions());

    // peduncle pitch rotation
    if (phytomer_parameters.peduncle.pitch.val() != 0.f || fbud.base_rotation.pitch != 0.f) {
        peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, inflorescence_bending_axis, deg2rad(phytomer_parameters.peduncle.pitch.val()) + fbud.base_rotation.pitch);
        phytomer_parameters.peduncle.pitch.resample();
    }

    // rotate peduncle to azimuth of petiole and apply peduncle base yaw rotation
    vec3 internode_axis = getAxisVector(1.f, getInternodeNodePositions());
    vec3 parent_petiole_base_axis = getPetioleAxisVector(0.f, fbud.parent_index);
    float parent_petiole_azimuth = -std::atan2(parent_petiole_base_axis.y, parent_petiole_base_axis.x);
    float current_peduncle_azimuth = -std::atan2(peduncle_axis.y, peduncle_axis.x);
    peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, internode_axis, (current_peduncle_azimuth - parent_petiole_azimuth));


    float theta_base = fabs(cart2sphere(peduncle_axis).zenith);

    for (int i = 1; i <= phytomer_parameters.peduncle.length_segments; i++) {
        if (phytomer_parameters.peduncle.curvature.val() != 0.f) {
            float theta_curvature = -deg2rad(phytomer_parameters.peduncle.curvature.val() * dr_peduncle);
            phytomer_parameters.peduncle.curvature.resample();
            if (fabs(theta_curvature) * float(i) < PI_F - theta_base) {
                peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, inflorescence_bending_axis, theta_curvature);
            } else {
                peduncle_axis = make_vec3(0, 0, -1);
            }
        }

        peduncle_vertices.at(i) = peduncle_vertices.at(i - 1) + dr_peduncle * peduncle_axis;

        peduncle_radii.at(i) = phytomer_parameters.peduncle.radius.val();
        peduncle_colors.at(i) = phytomer_parameters.peduncle.color;
    }
    phytomer_parameters.peduncle.radius.resample();

    if (build_context_geometry_peduncle) {
        fbud.peduncle_objIDs.push_back(context_ptr->addTubeObject(Ndiv_peduncle_radius, peduncle_vertices, peduncle_radii, peduncle_colors));
        context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(fbud.peduncle_objIDs.back()), "object_label", "peduncle");
    }

    // Create unique inflorescence prototypes for each shoot type so we can simply copy them for each leaf
    std::string parent_shoot_type_label = plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(parent_shoot_ID)->shoot_type_label;
    if (phytomer_parameters.inflorescence.unique_prototypes > 0) {
        // closed flowers
        if (phytomer_parameters.inflorescence.flower_prototype_function != nullptr &&
            plantarchitecture_ptr->unique_closed_flower_prototype_objIDs.find(phytomer_parameters.inflorescence.flower_prototype_function) == plantarchitecture_ptr->unique_closed_flower_prototype_objIDs.end()) {
            plantarchitecture_ptr->unique_closed_flower_prototype_objIDs[phytomer_parameters.inflorescence.flower_prototype_function].resize(phytomer_parameters.inflorescence.unique_prototypes);
            for (int prototype = 0; prototype < phytomer_parameters.inflorescence.unique_prototypes; prototype++) {
                uint objID_flower = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr, 1, false);
                plantarchitecture_ptr->unique_closed_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype) = objID_flower;
                context_ptr->hideObject(objID_flower);
            }
        }
        // open flowers
        if (phytomer_parameters.inflorescence.flower_prototype_function != nullptr &&
            plantarchitecture_ptr->unique_open_flower_prototype_objIDs.find(phytomer_parameters.inflorescence.flower_prototype_function) == plantarchitecture_ptr->unique_open_flower_prototype_objIDs.end()) {
            plantarchitecture_ptr->unique_open_flower_prototype_objIDs[phytomer_parameters.inflorescence.flower_prototype_function].resize(phytomer_parameters.inflorescence.unique_prototypes);
            for (int prototype = 0; prototype < phytomer_parameters.inflorescence.unique_prototypes; prototype++) {
                uint objID_flower = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr, 1, true);
                plantarchitecture_ptr->unique_open_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype) = objID_flower;
                context_ptr->hideObject(objID_flower);
            }
        }
        // fruit
        if (phytomer_parameters.inflorescence.fruit_prototype_function != nullptr &&
            plantarchitecture_ptr->unique_fruit_prototype_objIDs.find(phytomer_parameters.inflorescence.fruit_prototype_function) == plantarchitecture_ptr->unique_fruit_prototype_objIDs.end()) {
            plantarchitecture_ptr->unique_fruit_prototype_objIDs[phytomer_parameters.inflorescence.fruit_prototype_function].resize(phytomer_parameters.inflorescence.unique_prototypes);
            for (int prototype = 0; prototype < phytomer_parameters.inflorescence.unique_prototypes; prototype++) {
                uint objID_fruit = phytomer_parameters.inflorescence.fruit_prototype_function(context_ptr, 1);
                plantarchitecture_ptr->unique_fruit_prototype_objIDs.at(phytomer_parameters.inflorescence.fruit_prototype_function).at(prototype) = objID_fruit;
                context_ptr->hideObject(objID_fruit);
            }
        }
    }

    for (int fruit = 0; fruit < phytomer_parameters.inflorescence.flowers_per_peduncle.val(); fruit++) {
        uint objID_fruit;
        helios::vec3 fruit_scale;

        if (fbud.state == BUD_FRUITING) {
            if (phytomer_parameters.inflorescence.unique_prototypes > 0) { // copy existing prototype
                int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
                objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_fruit_prototype_objIDs.at(phytomer_parameters.inflorescence.fruit_prototype_function).at(prototype));
            } else { // load new prototype
                objID_fruit = phytomer_parameters.inflorescence.fruit_prototype_function(context_ptr, 1);
            }
            fruit_scale = phytomer_parameters.inflorescence.fruit_prototype_scale.val() * make_vec3(1, 1, 1);
            phytomer_parameters.inflorescence.fruit_prototype_scale.resample();
        } else {
            bool flower_is_open;
            if (fbud.state == BUD_FLOWER_CLOSED) {
                flower_is_open = false;
                if (phytomer_parameters.inflorescence.unique_prototypes > 0) { // copy existing prototype
                    int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
                    objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_closed_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype));
                } else { // load new prototype
                    objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr, 1, flower_is_open);
                }
            } else {
                flower_is_open = true;
                if (phytomer_parameters.inflorescence.unique_prototypes > 0) { // copy existing prototype
                    int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
                    objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_open_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype));
                } else { // load new prototype
                    objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr, 1, flower_is_open);
                }
            }
            fruit_scale = phytomer_parameters.inflorescence.flower_prototype_scale.val() * make_vec3(1, 1, 1);
            phytomer_parameters.inflorescence.flower_prototype_scale.resample();
        }

        float ind_from_tip = fabs(fruit - float(phytomer_parameters.inflorescence.flowers_per_peduncle.val() - 1) / float(phytomer_parameters.petiole.petioles_per_internode));

        context_ptr->scaleObject(objID_fruit, fruit_scale);

        // if we have more than one flower/fruit, we need to adjust the base position of the fruit
        vec3 fruit_base = peduncle_vertices.back();
        float frac = 1;
        if (phytomer_parameters.inflorescence.flowers_per_peduncle.val() > 1 && phytomer_parameters.inflorescence.flower_offset.val() > 0) {
            if (ind_from_tip != 0) {
                float offset = offset = (ind_from_tip - 0.5f) * phytomer_parameters.inflorescence.flower_offset.val() * phytomer_parameters.peduncle.length.val();
                if (phytomer_parameters.peduncle.length.val() > 0) {
                    frac = 1.f - offset / phytomer_parameters.peduncle.length.val();
                }
                fruit_base = PlantArchitecture::interpolateTube(peduncle_vertices, frac);
            }
        }

        // if we have more than one flower/fruit, we need to adjust the rotation about the peduncle
        float compound_rotation = 0;
        if (phytomer_parameters.inflorescence.flowers_per_peduncle.val() > 1) {
            if (phytomer_parameters.inflorescence.flower_offset.val() == 0) { // flowers/fruit are all at the tip, so just equally distribute them about the azimuth
                float dphi = PI_F / (floor(0.5 * float(phytomer_parameters.inflorescence.flowers_per_peduncle.val() - 1)) + 1);
                compound_rotation = -float(PI_F) + dphi * (fruit + 0.5f);
            } else {
                compound_rotation = deg2rad(phytomer_parameters.internode.phyllotactic_angle.val()) * float(ind_from_tip) + 2.f * PI_F / float(phytomer_parameters.petiole.petioles_per_internode) * float(fruit);
                phytomer_parameters.internode.phyllotactic_angle.resample();
            }
        }

        peduncle_axis = getAxisVector(frac, peduncle_vertices);

        vec3 fruit_axis = peduncle_axis;

        // roll rotation
        if (phytomer_parameters.inflorescence.roll.val() != 0.f) {
            context_ptr->rotateObject(objID_fruit, deg2rad(phytomer_parameters.inflorescence.roll.val()), "x");
            phytomer_parameters.inflorescence.roll.resample();
        }

        // pitch rotation
        float pitch_inflorescence = -asin_safe(peduncle_axis.z) + deg2rad(phytomer_parameters.inflorescence.pitch.val());
        phytomer_parameters.inflorescence.pitch.resample();
        if (fbud.state == BUD_FRUITING) { // gravity effect for fruit
            pitch_inflorescence = pitch_inflorescence + phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.val() * (0.5f * PI_F - pitch_inflorescence);
        }
        context_ptr->rotateObject(objID_fruit, pitch_inflorescence, "y");
        fruit_axis = rotatePointAboutLine(fruit_axis, nullorigin, make_vec3(1, 0, 0), pitch_inflorescence);

        // rotate flower/fruit to azimuth of peduncle
        context_ptr->rotateObject(objID_fruit, -std::atan2(peduncle_axis.y, peduncle_axis.x), "z");
        fruit_axis = rotatePointAboutLine(fruit_axis, nullorigin, make_vec3(0, 0, 1), -std::atan2(peduncle_axis.y, peduncle_axis.x));

        context_ptr->translateObject(objID_fruit, fruit_base);

        // rotate flower/fruit about peduncle (roll)
        if (phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.val() != 0 && fbud.state == BUD_FRUITING) {
            context_ptr->rotateObject(objID_fruit, deg2rad(phytomer_parameters.peduncle.roll.val()) + compound_rotation, fruit_base, make_vec3(0, 0, 1));
        } else {
            context_ptr->rotateObject(objID_fruit, deg2rad(phytomer_parameters.peduncle.roll.val()) + compound_rotation, fruit_base, peduncle_axis);
            fruit_axis = rotatePointAboutLine(fruit_axis, nullorigin, peduncle_axis, deg2rad(phytomer_parameters.peduncle.roll.val()) + compound_rotation);
        }
        phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.resample();

        fbud.inflorescence_bases.push_back(fruit_base);

        fbud.inflorescence_objIDs.push_back(objID_fruit);
    }
    phytomer_parameters.inflorescence.flowers_per_peduncle.resample();
    phytomer_parameters.peduncle.roll.resample();

    if (plantarchitecture_ptr->output_object_data.at("rank")) {
        context_ptr->setObjectData(fbud.peduncle_objIDs, "rank", rank);
        context_ptr->setObjectData(fbud.inflorescence_objIDs, "rank", rank);
    }

    if (plantarchitecture_ptr->output_object_data.at("peduncleID")) {
        for (uint objID: fbud.peduncle_objIDs) {
            context_ptr->setObjectData(objID, "peduncleID", (int) objID);
        }
    }
    for (uint objID: fbud.inflorescence_objIDs) {
        if (fbud.state == BUD_FLOWER_CLOSED && plantarchitecture_ptr->output_object_data.at("closedflowerID")) {
            context_ptr->setObjectData(objID, "closedflowerID", (int) objID);
        } else if (fbud.state == BUD_FLOWER_OPEN && plantarchitecture_ptr->output_object_data.at("openflowerID")) {
            context_ptr->clearObjectData(objID, "closedflowerID");
            context_ptr->setObjectData(objID, "openflowerID", (int) objID);
        } else if (plantarchitecture_ptr->output_object_data.at("fruitID")) {
            context_ptr->setObjectData(objID, "fruitID", (int) objID);
        }
    }
}

void Phytomer::setPetioleBase(const helios::vec3 &base_position) {
    vec3 old_base = petiole_vertices.front().front();
    vec3 shift = base_position - old_base;

    for (auto &petiole_vertice: petiole_vertices) {
        for (auto &vertex: petiole_vertice) {
            vertex += shift;
        }
    }

    if (build_context_geometry_petiole) {
        context_ptr->translateObject(flatten(petiole_objIDs), shift);
    }
    context_ptr->translateObject(flatten(leaf_objIDs), shift);

    for (auto &petiole: leaf_bases) {
        for (auto &leaf_base: petiole) {
            leaf_base += shift;
        }
    }
    for (auto &floral_bud: floral_buds) {
        for (auto &fbud: floral_bud) {
            fbud.base_position = petiole_vertices.front().front();
            context_ptr->translateObject(fbud.inflorescence_objIDs, shift);
            for (auto &base: fbud.inflorescence_bases) {
                base += shift;
            }
            if (build_context_geometry_peduncle) {
                context_ptr->translateObject(fbud.peduncle_objIDs, shift);
            }
        }
    }
}

void Phytomer::rotateLeaf(uint petiole_index, uint leaf_index, const AxisRotation &rotation) {
    if (petiole_index >= leaf_objIDs.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Invalid petiole index.");
    } else if (leaf_index >= leaf_objIDs.at(petiole_index).size()) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Invalid leaf index.");
    }

    vec3 petiole_axis = getPetioleAxisVector(1.f, petiole_index);
    // note: this is not exactly correct because it should get the axis at the leaf position and not the tip

    vec3 internode_axis = getInternodeAxisVector(1.f);

    vec3 pitch_axis = -1 * cross(internode_axis, petiole_axis);

    int leaves_per_petiole = leaf_rotation.at(petiole_index).size();
    float yaw;
    float roll;
    float compound_rotation = 0;
    if (leaves_per_petiole > 1 && leaf_index == float(leaves_per_petiole - 1) / 2.f) { // tip leaflet of compound leaf
        roll = 0;
        yaw = 0;
        compound_rotation = 0;
    } else if (leaves_per_petiole > 1 && leaf_index < float(leaves_per_petiole - 1) / 2.f) { // lateral leaflet of compound leaf
        yaw = -rotation.yaw;
        roll = -rotation.roll;
        compound_rotation = -0.5 * PI_F;
    } else { // not a compound leaf
        yaw = -rotation.yaw;
        roll = rotation.roll;
        compound_rotation = 0;
    }

    // roll
    if (roll != 0.f) {
        vec3 roll_axis = rotatePointAboutLine({petiole_axis.x, petiole_axis.y, 0}, nullorigin, {0, 0, 1}, leaf_rotation.at(petiole_index).at(leaf_index).yaw + compound_rotation);
        context_ptr->rotateObject(leaf_objIDs.at(petiole_index).at(leaf_index), roll, leaf_bases.at(petiole_index).at(leaf_index), roll_axis);
        leaf_rotation.at(petiole_index).at(leaf_index).roll += roll;
    }

    // pitch
    if (rotation.pitch != 0) {
        pitch_axis = rotatePointAboutLine(pitch_axis, nullorigin, {0, 0, 1}, -compound_rotation);
        context_ptr->rotateObject(leaf_objIDs.at(petiole_index).at(leaf_index), rotation.pitch, leaf_bases.at(petiole_index).at(leaf_index), pitch_axis);
        leaf_rotation.at(petiole_index).at(leaf_index).pitch += rotation.pitch;
    }

    // yaw
    if (yaw != 0.f) {
        context_ptr->rotateObject(leaf_objIDs.at(petiole_index).at(leaf_index), yaw, leaf_bases.at(petiole_index).at(leaf_index), {0, 0, 1});
        leaf_rotation.at(petiole_index).at(leaf_index).yaw += yaw;
    }
}

void Phytomer::setInternodeLengthScaleFraction(const float internode_scale_factor_fraction, const bool update_context_geometry) {
    assert(internode_scale_factor_fraction >= 0 && internode_scale_factor_fraction <= 1);

    if (internode_scale_factor_fraction == current_internode_scale_factor) {
        return;
    }

    float delta_scale = internode_scale_factor_fraction / current_internode_scale_factor;

    current_internode_scale_factor = internode_scale_factor_fraction;

    int p = shoot_index.x;
    int s_start = (p == 0) ? 1 : 0; // skip the first node at the base of the shoot

    for (int s = s_start; s < parent_shoot_ptr->shoot_internode_vertices.at(p).size(); s++) { // looping over all segments within this phytomer internode

        int p_minus = p;
        int s_minus = s - 1;
        if (s_minus < 0) {
            p_minus--;
            s_minus = static_cast<int>(parent_shoot_ptr->shoot_internode_vertices.at(p_minus).size() - 1);
        }

        vec3 central_axis = (parent_shoot_ptr->shoot_internode_vertices.at(p).at(s) - parent_shoot_ptr->shoot_internode_vertices.at(p_minus).at(s_minus));
        float current_length = central_axis.magnitude();
        central_axis = central_axis / current_length;
        vec3 dL = central_axis * current_length * (delta_scale - 1);

        // apply shift to all downstream nodes
        for (int p_downstream = p; p_downstream < parent_shoot_ptr->shoot_internode_vertices.size(); p_downstream++) {
            int sd_start = (p_downstream == p) ? s : 0;
            for (int s_downstream = sd_start; s_downstream < parent_shoot_ptr->shoot_internode_vertices.at(p_downstream).size(); s_downstream++) {
                parent_shoot_ptr->shoot_internode_vertices.at(p_downstream).at(s_downstream) += dL;
            }
        }
    }

    parent_shoot_ptr->updateShootNodes(update_context_geometry);
}

void Phytomer::scaleInternodeMaxLength(const float scale_factor) {
    this->internode_length_max *= scale_factor;

    current_internode_scale_factor = current_internode_scale_factor / scale_factor;

    if (current_internode_scale_factor >= 1.f) {
        setInternodeLengthScaleFraction(1.f, true);
        current_internode_scale_factor = 1.f;
    }
}

void Phytomer::setInternodeMaxLength(const float internode_length_max_new) {
    float scale_factor = internode_length_max_new / this->internode_length_max;
    scaleInternodeMaxLength(scale_factor);
}

void Phytomer::setInternodeMaxRadius(float internode_radius_max_new) { this->internode_radius_max = internode_radius_max_new; }

void Phytomer::setLeafScaleFraction(uint petiole_index, float leaf_scale_factor_fraction) {

    assert(leaf_scale_factor_fraction >= 0 && leaf_scale_factor_fraction <= 1);

    if (current_leaf_scale_factor.size() <= petiole_index) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Invalid petiole index for leaf scale factor.");
    }

    // If the leaf is already at leaf_scale_factor_fraction, or there are no petioles/leaves, nothing to do.
    if (leaf_scale_factor_fraction == current_leaf_scale_factor.at(petiole_index) || (leaf_objIDs.at(petiole_index).empty() && petiole_objIDs.at(petiole_index).empty())) {
        return;
    }

    float delta_scale = leaf_scale_factor_fraction / current_leaf_scale_factor.at(petiole_index);

    petiole_length.at(petiole_index) *= delta_scale;

    current_leaf_scale_factor.at(petiole_index) = leaf_scale_factor_fraction;

    assert(leaf_objIDs.size() == leaf_bases.size());

    // scale the petiole

    if (!petiole_objIDs.at(petiole_index).empty()) {
        int node = 0;
        vec3 last_base = petiole_vertices.at(petiole_index).front(); // looping over petioles
        for (uint objID: petiole_objIDs.at(petiole_index)) { // looping over cones/segments within petiole
            context_ptr->getConeObjectPointer(objID)->scaleLength(delta_scale);
            context_ptr->getConeObjectPointer(objID)->scaleGirth(delta_scale);
            petiole_radii.at(petiole_index).at(node) *= delta_scale;
            if (node > 0) {
                vec3 new_base = context_ptr->getConeObjectNode(objID, 0);
                context_ptr->translateObject(objID, last_base - new_base);
            } else {
                petiole_vertices.at(petiole_index).at(0) = context_ptr->getConeObjectNode(objID, 0);
            }
            last_base = context_ptr->getConeObjectNode(objID, 1);
            petiole_vertices.at(petiole_index).at(node + 1) = last_base;
            node++;
        }
    }

    // scale and translate leaves
    assert(leaf_objIDs.at(petiole_index).size() == leaf_bases.at(petiole_index).size());
    for (int leaf = 0; leaf < leaf_objIDs.at(petiole_index).size(); leaf++) {
        float ind_from_tip = float(leaf) - float(leaf_objIDs.at(petiole_index).size() - 1) / 2.f;

        context_ptr->translateObject(leaf_objIDs.at(petiole_index).at(leaf), -1 * leaf_bases.at(petiole_index).at(leaf));
        context_ptr->scaleObject(leaf_objIDs.at(petiole_index).at(leaf), delta_scale * make_vec3(1, 1, 1));
        if (ind_from_tip == 0) {
            context_ptr->translateObject(leaf_objIDs.at(petiole_index).at(leaf), petiole_vertices.at(petiole_index).back());
            leaf_bases.at(petiole_index).at(leaf) = petiole_vertices.at(petiole_index).back();
        } else {
            float offset = (fabs(ind_from_tip) - 0.5f) * phytomer_parameters.leaf.leaflet_offset.val() * phytomer_parameters.petiole.length.val();
            vec3 leaf_base = PlantArchitecture::interpolateTube(petiole_vertices.at(petiole_index), 1.f - offset / phytomer_parameters.petiole.length.val());
            context_ptr->translateObject(leaf_objIDs.at(petiole_index).at(leaf), leaf_base);
            leaf_bases.at(petiole_index).at(leaf) = leaf_base;
        }
    }
}

void Phytomer::setLeafScaleFraction(float leaf_scale_factor_fraction) {
    for (uint petiole_index = 0; petiole_index < leaf_objIDs.size(); petiole_index++) {
        setLeafScaleFraction(petiole_index, leaf_scale_factor_fraction);
    }
}

void Phytomer::setLeafPrototypeScale(uint petiole_index, float leaf_prototype_scale) {

    if (leaf_objIDs.size() <= petiole_index) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Invalid petiole index for leaf prototype scale.");
    }
    if (leaf_prototype_scale < 0.f) {
        leaf_prototype_scale = 0;
    }

    float tip_ind = ceil(scast<float>(leaf_size_max.at(petiole_index).size() - 1) / 2.f);
    float scale_factor = leaf_prototype_scale / leaf_size_max.at(petiole_index).at(tip_ind);
    current_leaf_scale_factor.at(petiole_index) *= scale_factor;

    for (int leaf = 0; leaf < leaf_objIDs.at(petiole_index).size(); leaf++) {
        leaf_size_max.at(petiole_index).at(leaf) *= scale_factor;
        context_ptr->scaleObjectAboutPoint(leaf_objIDs.at(petiole_index).at(leaf), scale_factor * make_vec3(1, 1, 1), leaf_bases.at(petiole_index).at(leaf));
    }

    // note: at time of phytomer creation, petiole curvature was based on the petiole length prior to this scaling. To stay consistent, we will scale the curvature appropriately.
    this->petiole_curvature.at(petiole_index) /= scale_factor;

    if (current_leaf_scale_factor.at(petiole_index) >= 1.f) {
        setLeafScaleFraction(petiole_index, 1.f);
        current_leaf_scale_factor.at(petiole_index) = 1.f;
    }
}

void Phytomer::setLeafPrototypeScale(float leaf_prototype_scale) {
    for (uint petiole_index = 0; petiole_index < leaf_objIDs.size(); petiole_index++) {
        setLeafPrototypeScale(petiole_index, leaf_prototype_scale);
    }
}

void Phytomer::scaleLeafPrototypeScale(uint petiole_index, float scale_factor) {
    if (leaf_objIDs.size() <= petiole_index) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Invalid petiole index for leaf prototype scale.");
    }
    if (scale_factor < 0.f) {
        scale_factor = 0;
    }

    current_leaf_scale_factor.at(petiole_index) /= scale_factor;

    for (int leaf = 0; leaf < leaf_objIDs.at(petiole_index).size(); leaf++) {
        leaf_size_max.at(petiole_index).at(leaf) *= scale_factor;
        context_ptr->scaleObjectAboutPoint(leaf_objIDs.at(petiole_index).at(leaf), scale_factor * make_vec3(1, 1, 1), leaf_bases.at(petiole_index).at(leaf));
    }

    // note: at time of phytomer creation, petiole curvature was based on the petiole length prior to this scaling. To stay consistent, we will scale the curvature appropriately.
    this->petiole_curvature.at(petiole_index) /= scale_factor;

    if (current_leaf_scale_factor.at(petiole_index) >= 1.f) {
        setLeafScaleFraction(petiole_index, 1.f);
        current_leaf_scale_factor.at(petiole_index) = 1.f;
    }
}

void Phytomer::scaleLeafPrototypeScale(float scale_factor) {
    for (uint petiole_index = 0; petiole_index < leaf_objIDs.size(); petiole_index++) {
        scaleLeafPrototypeScale(petiole_index, scale_factor);
    }
}

void Phytomer::setInflorescenceScaleFraction(FloralBud &fbud, float inflorescence_scale_factor_fraction) const {
    assert(inflorescence_scale_factor_fraction >= 0 && inflorescence_scale_factor_fraction <= 1);

    if (inflorescence_scale_factor_fraction == fbud.current_fruit_scale_factor) {
        return;
    }

    float delta_scale = inflorescence_scale_factor_fraction / fbud.current_fruit_scale_factor;

    fbud.current_fruit_scale_factor = inflorescence_scale_factor_fraction;

    // scale and translate flowers/fruit
    for (int inflorescence = 0; inflorescence < fbud.inflorescence_objIDs.size(); inflorescence++) {
        context_ptr->scaleObjectAboutPoint(fbud.inflorescence_objIDs.at(inflorescence), delta_scale * make_vec3(1, 1, 1), fbud.inflorescence_bases.at(inflorescence));
    }
}

void Phytomer::removeLeaf() {
    // parent_shoot_ptr->propagateDownstreamLeafArea( parent_shoot_ptr, this->shoot_index.x, -1.f*getLeafArea());

    this->petiole_radii.resize(0);
    //    this->petiole_vertices.resize(0);
    this->petiole_colors.resize(0);
    this->petiole_length.resize(0);
    this->leaf_size_max.resize(0);
    this->leaf_rotation.resize(0);
    this->leaf_bases.resize(0);

    context_ptr->deleteObject(flatten(leaf_objIDs));
    leaf_objIDs.clear();
    leaf_bases.clear();

    if (build_context_geometry_petiole) {
        context_ptr->deleteObject(flatten(petiole_objIDs));
        petiole_objIDs.resize(0);
    }
}

void Phytomer::deletePhytomer() {

    // prune the internode tube in the Context
    if (context_ptr->doesObjectExist(parent_shoot_ptr->internode_tube_objID)) {
        uint tube_nodes = context_ptr->getTubeObjectNodeCount(parent_shoot_ptr->internode_tube_objID);
        uint tube_segments = this->parent_shoot_ptr->shoot_parameters.phytomer_parameters.internode.length_segments;
        uint tube_prune_index;
        if ( this->shoot_index.x == 0 ) {
            tube_prune_index = 0;
        }else {
            tube_prune_index = this->shoot_index.x * tube_segments + 1; //note that first segment has an extra vertex
        }
        if ( tube_prune_index < tube_nodes ) {
            context_ptr->pruneTubeNodes(parent_shoot_ptr->internode_tube_objID, tube_prune_index);
        }
        parent_shoot_ptr->terminateApicalBud();
    }

    for (uint node = this->shoot_index.x; node < shoot_index.y; node++) {
        auto &phytomer = parent_shoot_ptr->phytomers.at(node);

        // leaves
        phytomer->removeLeaf();

        // inflorescence
        for (auto &petiole: phytomer->floral_buds) {
            for (auto &fbud: petiole) {
                for (int p = fbud.inflorescence_objIDs.size() - 1; p >= 0; p--) {
                    uint objID = fbud.inflorescence_objIDs.at(p);
                    context_ptr->deleteObject(objID);
                    fbud.inflorescence_objIDs.erase(fbud.inflorescence_objIDs.begin() + p);
                    fbud.inflorescence_bases.erase(fbud.inflorescence_bases.begin() + p);
                }
                for (int p = fbud.peduncle_objIDs.size() - 1; p >= 0; p--) {
                    context_ptr->deleteObject(fbud.peduncle_objIDs);
                    context_ptr->deleteObject(fbud.inflorescence_objIDs);
                    fbud.peduncle_objIDs.clear();
                    fbud.inflorescence_objIDs.clear();
                    fbud.inflorescence_bases.clear();
                    break;
                }
            }
        }

        // delete any child shoots
        if (parent_shoot_ptr->childIDs.find(node) != parent_shoot_ptr->childIDs.end()) {
            for (auto childID: parent_shoot_ptr->childIDs.at(node)) {
                auto child_shoot = plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(childID);
                if (!child_shoot->phytomers.empty()) {
                    child_shoot->phytomers.front()->deletePhytomer();
                }
            }
        }
    }

    // delete shoot arrays
    parent_shoot_ptr->shoot_internode_radii.resize(this->shoot_index.x);
    parent_shoot_ptr->shoot_internode_vertices.resize(this->shoot_index.x);
    parent_shoot_ptr->phytomers.resize(this->shoot_index.x);

    // set the correct node index for phytomers on this shoot
    for (const auto &phytomer: parent_shoot_ptr->phytomers) {
        phytomer->shoot_index.y = scast<int>(parent_shoot_ptr->phytomers.size());
    }
    parent_shoot_ptr->current_node_number = scast<int>(parent_shoot_ptr->phytomers.size());

}

bool Phytomer::hasLeaf() const { return (!leaf_bases.empty() && !leaf_bases.front().empty()); }

float Phytomer::calculateDownstreamLeafArea() const { return parent_shoot_ptr->sumShootLeafArea(shoot_index.x); }

Shoot::Shoot(uint plant_ID, int shoot_ID, int parent_shoot_ID, uint parent_node, uint parent_petiole_index, uint rank, const helios::vec3 &shoot_base_position, const AxisRotation &shoot_base_rotation, uint current_node_number,
             float internode_length_shoot_initial, ShootParameters &shoot_params, std::string shoot_type_label, PlantArchitecture *plant_architecture_ptr) :
    current_node_number(current_node_number), base_position(shoot_base_position), base_rotation(shoot_base_rotation), ID(shoot_ID), parent_shoot_ID(parent_shoot_ID), plantID(plant_ID), parent_node_index(parent_node), rank(rank),
    parent_petiole_index(parent_petiole_index), internode_length_max_shoot_initial(internode_length_shoot_initial), shoot_parameters(shoot_params), shoot_type_label(std::move(shoot_type_label)), plantarchitecture_ptr(plant_architecture_ptr) {
    carbohydrate_pool_molC = 0;
    phyllochron_counter = 0;
    isdormant = true;
    gravitropic_curvature = shoot_params.gravitropic_curvature.val();
    context_ptr = plant_architecture_ptr->context_ptr;
    phyllochron_instantaneous = shoot_parameters.phyllochron_min.val();
    elongation_rate_instantaneous = shoot_parameters.elongation_rate_max.val();

    if (parent_shoot_ID >= 0) {
        plant_architecture_ptr->plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID)->childIDs[(int) parent_node_index].push_back(shoot_ID);
    }
}

void Shoot::buildShootPhytomers(float internode_radius, float internode_length, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper) {
    for (int i = 0; i < current_node_number; i++) { // loop over phytomers to build up the shoot

        float taper = 1.f;
        if (current_node_number > 1) {
            taper = 1.f - radius_taper * float(i) / float(current_node_number - 1);
        }

        // Adding the phytomer(s) to the shoot
        appendPhytomer(internode_radius * taper, internode_length, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, shoot_parameters.phytomer_parameters);
    }
}

std::string Shoot::sampleChildShootType() const {
    auto shoot_ptr = this;

    assert(shoot_ptr->shoot_parameters.child_shoot_type_labels.size() == shoot_ptr->shoot_parameters.child_shoot_type_probabilities.size());

    std::string child_shoot_type_label;

    if (shoot_ptr->shoot_parameters.child_shoot_type_labels.empty()) { // if user doesn't specify child shoot types, generate the same type by default
        child_shoot_type_label = shoot_ptr->shoot_type_label;
    } else if (shoot_ptr->shoot_parameters.child_shoot_type_labels.size() == 1) { // if only one child shoot types was specified, use it
        child_shoot_type_label = shoot_ptr->shoot_parameters.child_shoot_type_labels.at(0);
    } else {
        float randf = context_ptr->randu();
        int shoot_type_index = -1;
        float cumulative_probability = 0;
        for (int s = 0; s < shoot_ptr->shoot_parameters.child_shoot_type_labels.size(); s++) {
            cumulative_probability += shoot_ptr->shoot_parameters.child_shoot_type_probabilities.at(s);
            if (randf < cumulative_probability) {
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

    return child_shoot_type_label;
}

bool Shoot::sampleVegetativeBudBreak(uint node_index) const {
    if (node_index >= phytomers.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::sampleVegetativeBudBreak): Invalid node index. Node index must be less than the number of phytomers on the shoot.");
    }

    float probability_min = plantarchitecture_ptr->shoot_types.at(this->shoot_type_label).vegetative_bud_break_probability_min.val();
    float probability_decay = plantarchitecture_ptr->shoot_types.at(this->shoot_type_label).vegetative_bud_break_probability_decay_rate.val();

    float bud_break_probability;
    if (!shoot_parameters.growth_requires_dormancy && probability_decay < 0) {
        bud_break_probability = probability_min;
    } else if (probability_decay > 0) { // probability maximum at apex
        bud_break_probability = std::fmax(probability_min, 1.f - probability_decay * float(this->current_node_number - node_index - 1));
    } else if (probability_decay < 0) { // probability maximum at base
        bud_break_probability = std::fmax(probability_min, 1.f - fabs(probability_decay) * float(node_index));
    } else {
        if (probability_decay == 0.f) {
            bud_break_probability = probability_min;
        } else {
            bud_break_probability = 1.f;
        }
    }

    bool bud_break = true;
    if (context_ptr->randu() > bud_break_probability) {
        bud_break = false;
    }

    return bud_break;
}

uint Shoot::sampleEpicormicShoot(float dt, std::vector<float> &epicormic_positions_fraction) const {
    std::string epicormic_shoot_label = plantarchitecture_ptr->plant_instances.at(this->plantID).epicormic_shoot_probability_perlength_per_day.first;

    if (epicormic_shoot_label.empty()) {
        return 0;
    }

    float epicormic_probability = plantarchitecture_ptr->plant_instances.at(this->plantID).epicormic_shoot_probability_perlength_per_day.second;

    if (epicormic_probability == 0) {
        return 0;
    }

    uint Nshoots = 0;

    epicormic_positions_fraction.clear();

    float shoot_length = this->calculateShootLength();

    float time = dt;
    while (time > 0) {
        float dta = std::min(time, 1.f);

        float shoot_fraction = context_ptr->randu();

        float elevation = fabs(getShootAxisVector(shoot_fraction).z);

        bool new_shoot = uint((epicormic_probability * shoot_length * dta * elevation > context_ptr->randu()));

        Nshoots += uint(new_shoot);

        if (new_shoot) {
            epicormic_positions_fraction.push_back(shoot_fraction);
        }

        time -= dta;
    }

    assert(epicormic_positions_fraction.size() == Nshoots);

    return Nshoots;
}

uint PlantArchitecture::addBaseStemShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction,
                                         float radius_taper, const std::string &shoot_type_label) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addBaseStemShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shoot_types.find(shoot_type_label) == shoot_types.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addBaseStemShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);
    validateShootTypes(shoot_parameters);

    if (current_node_number > shoot_parameters.max_nodes.val()) {
        helios_runtime_error("ERROR (PlantArchitecture::addBaseStemShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_parameters.max_nodes.val()) + ".");
    }

    uint shootID = shoot_tree_ptr->size();
    vec3 base_position = plant_instances.at(plantID).base_position;

    // Create the new shoot
    auto *shoot_new = (new Shoot(plantID, shootID, -1, 0, 0, 0, base_position, base_rotation, current_node_number, internode_length_max, shoot_parameters, shoot_type_label, this));
    shoot_tree_ptr->emplace_back(shoot_new);

    // Build phytomer geometry
    shoot_new->buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper);

    return shootID;
}

uint PlantArchitecture::appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction,
                                    float leaf_scale_factor_fraction, float radius_taper, const std::string &shoot_type_label) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shoot_types.find(shoot_type_label) == shoot_types.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = shoot_types.at(shoot_type_label);
    validateShootTypes(shoot_parameters);

    if (shoot_tree_ptr->empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot append shoot to empty shoot. You must call addBaseStemShoot() first for each plant.");
    } else if (parent_shoot_ID >= int(shoot_tree_ptr->size())) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    } else if (current_node_number > shoot_parameters.max_nodes.val()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_parameters.max_nodes.val()) + ".");
    } else if (shoot_tree_ptr->at(parent_shoot_ID)->phytomers.empty()) {
        std::cerr << "WARNING (PlantArchitecture::appendShoot): Shoot does not have any phytomers to append." << std::endl;
    }

    // stop parent shoot from producing new phytomers at the apex
    shoot_tree_ptr->at(parent_shoot_ID)->shoot_parameters.max_nodes = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number;
    shoot_tree_ptr->at(parent_shoot_ID)->terminateApicalBud(); // meristem should not keep growing after appending shoot

    // accumulate all the values that will be passed to Shoot constructor
    int appended_shootID = int(shoot_tree_ptr->size());
    uint parent_node = shoot_tree_ptr->at(parent_shoot_ID)->current_node_number - 1;
    uint rank = shoot_tree_ptr->at(parent_shoot_ID)->rank;
    vec3 base_position = interpolateTube(shoot_tree_ptr->at(parent_shoot_ID)->phytomers.back()->getInternodeNodePositions(), 0.9f);

    // Create the new shoot
    auto *shoot_new = (new Shoot(plantID, appended_shootID, parent_shoot_ID, parent_node, 0, rank, base_position, base_rotation, current_node_number, internode_length_max, shoot_parameters, shoot_type_label, this));
    shoot_tree_ptr->emplace_back(shoot_new);

    // Build phytomer geometry
    shoot_new->buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper);

    return appended_shootID;
}

uint PlantArchitecture::addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node_index, uint current_node_number, const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max,
                                      float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper, const std::string &shoot_type_label, uint petiole_index) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shoot_types.find(shoot_type_label) == shoot_types.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    if (parent_shoot_ID <= -1 || parent_shoot_ID >= shoot_tree_ptr->size()) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    } else if (shoot_tree_ptr->at(parent_shoot_ID)->phytomers.size() <= parent_node_index) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent shoot does not have a node " + std::to_string(parent_node_index) + ".");
    }

    // accumulate all the values that will be passed to Shoot constructor
    auto shoot_parameters = shoot_types.at(shoot_type_label);
    validateShootTypes(shoot_parameters);
    uint parent_rank = (int) shoot_tree_ptr->at(parent_shoot_ID)->rank;
    int childID = int(shoot_tree_ptr->size());

    // Calculate the position of the shoot base
    const auto parent_shoot_ptr = shoot_tree_ptr->at(parent_shoot_ID);

    vec3 shoot_base_position = parent_shoot_ptr->shoot_internode_vertices.at(parent_node_index).back();

    // Shift the shoot base position outward by the parent internode radius
    vec3 petiole_axis = parent_shoot_ptr->phytomers.at(parent_node_index)->getPetioleAxisVector(0, petiole_index);
    shoot_base_position += 0.9f * petiole_axis * parent_shoot_ptr->phytomers.at(parent_node_index)->getInternodeRadius(1.f);

    // Create the new shoot
    auto *shoot_new = (new Shoot(plantID, childID, parent_shoot_ID, parent_node_index, petiole_index, parent_rank + 1, shoot_base_position, shoot_base_rotation, current_node_number, internode_length_max, shoot_parameters, shoot_type_label, this));
    shoot_tree_ptr->emplace_back(shoot_new);

    // Build phytomer geometry
    shoot_new->buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper);

    return childID;
}

uint PlantArchitecture::addEpicormicShoot(uint plantID, int parent_shoot_ID, float parent_position_fraction, uint current_node_number, float zenith_perturbation_degrees, float internode_radius, float internode_length_max,
                                          float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper, const std::string &shoot_type_label) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addEpicormicShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shoot_types.find(shoot_type_label) == shoot_types.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addEpicormicShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto &parent_shoot = plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID);

    uint parent_node_index = 0;
    if (parent_position_fraction > 0) {
        parent_node_index = std::ceil(parent_position_fraction * float(parent_shoot->phytomers.size())) - 1;
    }

    vec3 petiole_axis = plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID)->phytomers.at(parent_node_index)->getPetioleAxisVector(0, 0);

    //\todo Figuring out how to set this correctly to make the shoot vertical, which avoids having to write a child shoot function.
    AxisRotation base_rotation = make_AxisRotation(0, acos_safe(petiole_axis.z), 0);

    return addChildShoot(plantID, parent_shoot_ID, parent_node_index, current_node_number, base_rotation, internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper, shoot_type_label, 0);
}

void PlantArchitecture::validateShootTypes(ShootParameters &shoot_parameters) const {
    assert(shoot_parameters.child_shoot_type_probabilities.size() == shoot_parameters.child_shoot_type_labels.size());

    for (int ind = shoot_parameters.child_shoot_type_labels.size() - 1; ind >= 0; ind--) {
        if (shoot_types.find(shoot_parameters.child_shoot_type_labels.at(ind)) == shoot_types.end()) {
            shoot_parameters.child_shoot_type_labels.erase(shoot_parameters.child_shoot_type_labels.begin() + ind);
            shoot_parameters.child_shoot_type_probabilities.erase(shoot_parameters.child_shoot_type_probabilities.begin() + ind);
        }
    }
}

int PlantArchitecture::appendPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_parameters, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction,
                                             float leaf_scale_factor_fraction) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendPhytomerToShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    if (shootID >= shoot_tree_ptr->size()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendPhytomerToShoot): Parent with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto current_shoot_ptr = plant_instances.at(plantID).shoot_tree.at(shootID);

    int pID = current_shoot_ptr->appendPhytomer(internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, phytomer_parameters);

    current_shoot_ptr->current_node_number++;
    current_shoot_ptr->nodes_this_season++;

    for (auto &phytomers: current_shoot_ptr->phytomers) {
        phytomers->shoot_index.y = current_shoot_ptr->current_node_number;
    }

    // If this shoot reached max nodes, add a terminal floral bud if max_terminal_floral_buds > 0
    if (current_shoot_ptr->current_node_number == current_shoot_ptr->shoot_parameters.max_nodes.val()) {
        if (!current_shoot_ptr->shoot_parameters.flowers_require_dormancy && current_shoot_ptr->shoot_parameters.max_terminal_floral_buds.val() > 0) {
            current_shoot_ptr->addTerminalFloralBud();
            BudState state;
            if (current_shoot_ptr->shoot_parameters.phytomer_parameters.inflorescence.flower_prototype_function != nullptr) {
                state = BUD_FLOWER_CLOSED;
            } else if (current_shoot_ptr->shoot_parameters.phytomer_parameters.inflorescence.fruit_prototype_function != nullptr) {
                state = BUD_FRUITING;
            } else {
                return pID;
            }
            for (auto &fbuds: current_shoot_ptr->phytomers.back()->floral_buds) {
                for (auto &fbud: fbuds) {
                    if (fbud.isterminal) {
                        fbud.state = state;
                        current_shoot_ptr->phytomers.back()->updateInflorescence(fbud);
                    }
                }
            }
        }
    }

    // If this shoot reached the max nodes for the season, add a dormant floral bud and make terminal vegetative bud dormant
    else if (current_shoot_ptr->nodes_this_season >= current_shoot_ptr->shoot_parameters.max_nodes_per_season.val()) {
        if (!current_shoot_ptr->shoot_parameters.flowers_require_dormancy && current_shoot_ptr->shoot_parameters.max_terminal_floral_buds.val() > 0) {
            current_shoot_ptr->addTerminalFloralBud();
            for (auto &fbuds: current_shoot_ptr->phytomers.back()->floral_buds) {
                for (auto &fbud: fbuds) {
                    if (fbud.isterminal) {
                        fbud.state = BUD_DORMANT;
                        current_shoot_ptr->phytomers.back()->updateInflorescence(fbud);
                    }
                }
            }
        }
        current_shoot_ptr->phytomers.at(pID)->isdormant = true;
    }

    return pID;
}

void PlantArchitecture::enableEpicormicChildShoots(uint plantID, const std::string &epicormic_shoot_type_label, float epicormic_probability_perlength_perday) {
    if (shoot_types.find(epicormic_shoot_type_label) == shoot_types.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::enableEpicormicChildShoots): Shoot type with label of " + epicormic_shoot_type_label + " does not exist.");
    } else if (epicormic_probability_perlength_perday < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::enableEpicormicChildShoots): Epicormic probability must be greater than or equal to zero.");
    } else if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::enableEpicormicChildShoots): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).epicormic_shoot_probability_perlength_per_day = std::make_pair(epicormic_shoot_type_label, epicormic_probability_perlength_perday);
}

void PlantArchitecture::disableInternodeContextBuild() { build_context_geometry_internode = false; }

void PlantArchitecture::disablePetioleContextBuild() { build_context_geometry_petiole = false; }

void PlantArchitecture::disablePeduncleContextBuild() { build_context_geometry_peduncle = false; }

void PlantArchitecture::enableGroundClipping(float ground_height) { ground_clipping_height = ground_height; }

void PlantArchitecture::incrementPhytomerInternodeGirth(uint plantID, uint shootID, uint node_number, float dt, bool update_context_geometry) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    } else if (node_number >= shoot->current_node_number) {
        helios_runtime_error("ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Cannot scale internode " + std::to_string(node_number) + " because there are only " + std::to_string(shoot->current_node_number) + " nodes in this shoot.");
    }

    auto phytomer = shoot->phytomers.at(node_number);

    // float leaf_area = phytomer->calculateDownstreamLeafArea();
    float leaf_area = phytomer->downstream_leaf_area;
    // std::cout << "leaf area: " << leaf_area_old << " " << leaf_area << std::endl;
    if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        context_ptr->setObjectData(shoot->internode_tube_objID, "leaf_area", leaf_area);
    }
    float phytomer_age = phytomer->age;
    float girth_area_factor = shoot->shoot_parameters.girth_area_factor.val();
    if (phytomer_age > 365) {
        girth_area_factor = shoot->shoot_parameters.girth_area_factor.val() * 365 / phytomer_age;
    }


    float internode_area = girth_area_factor * leaf_area * 1e-4;
    phytomer->parent_shoot_ptr->shoot_parameters.girth_area_factor.resample();

    float phytomer_radius = sqrtf(internode_area / PI_F);

    auto &segment = shoot->shoot_internode_radii.at(node_number);
    for (float &radius: segment) {
        if (phytomer_radius > radius) { // radius should only increase
            radius = radius + 0.5 * (phytomer_radius - radius);
        }
    }

    if (update_context_geometry && context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        context_ptr->setTubeRadii(shoot->internode_tube_objID, flatten(shoot->shoot_internode_radii));
    }
}

void PlantArchitecture::pruneGroundCollisions( uint plantID ) {

    if ( plant_instances.find(plantID) == plant_instances.end() ) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneGroundCollisions): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for (auto &shoot: plant_instances.at(plantID).shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {

            // internode
            if ((phytomer->shoot_index.x == 0 && phytomer->rank > 0) && context_ptr->doesObjectExist(shoot->internode_tube_objID) && detectGroundCollision(shoot->internode_tube_objID)) {
                context_ptr->deleteObject(shoot->internode_tube_objID);
                shoot->terminateApicalBud();
            }

            // leaves
            for (uint petiole = 0; petiole < phytomer->leaf_objIDs.size(); petiole++) {
                if (detectGroundCollision(phytomer->leaf_objIDs.at(petiole))) {
                    phytomer->removeLeaf();
                }
            }

            // inflorescence
            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    for (int p = fbud.inflorescence_objIDs.size() - 1; p >= 0; p--) {
                        uint objID = fbud.inflorescence_objIDs.at(p);
                        if (detectGroundCollision(objID)) {
                            context_ptr->deleteObject(objID);
                            fbud.inflorescence_objIDs.erase(fbud.inflorescence_objIDs.begin() + p);
                            fbud.inflorescence_bases.erase(fbud.inflorescence_bases.begin() + p);
                        }
                    }
                    for (int p = fbud.peduncle_objIDs.size() - 1; p >= 0; p--) {
                        uint objID = fbud.peduncle_objIDs.at(p);
                        if (detectGroundCollision(objID)) {
                            context_ptr->deleteObject(fbud.peduncle_objIDs);
                            context_ptr->deleteObject(fbud.inflorescence_objIDs);
                            fbud.peduncle_objIDs.clear();
                            fbud.inflorescence_objIDs.clear();
                            fbud.inflorescence_bases.clear();
                            break;
                        }
                    }
                }
            }
        }
    }

    // prune the shoots if all downstream leaves have been removed
    // for (auto &shoot: plant_instances.at(plantID).shoot_tree) {
    //     int node = -1;
    //     for ( node = shoot->phytomers.size() - 2; node >= 0; node--) {
    //         if ( shoot->phytomers.size() > node && shoot->phytomers.at(node)->hasLeaf() ) {
    //             break;
    //         }else {
    //         }
    //     }
    //     if ( node>=0 && node+1 < shoot-> phytomers.size()-1 ) {
    //         pruneBranch(plantID, shoot->ID, node+1);
    //     }
    // }

}

void PlantArchitecture::setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerLeafScale): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto parent_shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerLeafScale): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    } else if (node_number >= parent_shoot->current_node_number) {
        helios_runtime_error("ERROR (PlantArchitecture::setPhytomerLeafScale): Cannot scale leaf " + std::to_string(node_number) + " because there are only " + std::to_string(parent_shoot->current_node_number) + " nodes in this shoot.");
    }
    if (leaf_scale_factor_fraction < 0 || leaf_scale_factor_fraction > 1) {
        std::cerr << "WARNING (PlantArchitecture::setPhytomerLeafScale): Leaf scaling factor was outside the range of 0 to 1. No scaling was applied." << std::endl;
        return;
    }

    parent_shoot->phytomers.at(node_number)->setLeafScaleFraction(leaf_scale_factor_fraction);
}

void PlantArchitecture::setPlantBasePosition(uint plantID, const helios::vec3 &base_position) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).base_position = base_position;

    //\todo Does not work after shoots have been added to the plant.
    if (!plant_instances.at(plantID).shoot_tree.empty()) {
        std::cerr << "WARNING (PlantArchitecture::setPlantBasePosition): This function does not work after shoots have been added to the plant." << std::endl;
    }
}

void PlantArchitecture::setPlantLeafElevationAngleDistribution(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination) const {
    if (Beta_mu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafElevationAngleDistribution): Beta_mu_inclination must be greater than or equal to zero.");
    } else if (Beta_nu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafElevationAngleDistribution): Beta_nu_inclination must be greater than or equal to zero.");
    }

    setPlantLeafAngleDistribution_private({plantID}, Beta_mu_inclination, Beta_nu_inclination, 0.f, 0.f, true, false);
}

void PlantArchitecture::setPlantLeafElevationAngleDistribution(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination) const {
    if (Beta_mu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafElevationAngleDistribution): Beta_mu_inclination must be greater than or equal to zero.");
    } else if (Beta_nu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafElevationAngleDistribution): Beta_nu_inclination must be greater than or equal to zero.");
    }

    setPlantLeafAngleDistribution_private(plantIDs, Beta_mu_inclination, Beta_nu_inclination, 0.f, 0.f, true, false);
}

void PlantArchitecture::setPlantLeafAzimuthAngleDistribution(uint plantID, float eccentricity, float ellipse_rotation_degrees) const {
    if (eccentricity < 0.f || eccentricity > 1.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAzimuthAngleDistribution): Eccentricity must be between 0 and 1.");
    }

    setPlantLeafAngleDistribution_private({plantID}, 0.f, 0.f, eccentricity, ellipse_rotation_degrees, false, true);
}

void PlantArchitecture::setPlantLeafAzimuthAngleDistribution(const std::vector<uint> &plantIDs, float eccentricity, float ellipse_rotation_degrees) const {
    if (eccentricity < 0.f || eccentricity > 1.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAzimuthAngleDistribution): Eccentricity must be between 0 and 1.");
    }

    setPlantLeafAngleDistribution_private(plantIDs, 0.f, 0.f, eccentricity, ellipse_rotation_degrees, false, true);
}

void PlantArchitecture::setPlantLeafAngleDistribution(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees) const {
    if (Beta_mu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAngleDistribution): Beta_mu_inclination must be greater than or equal to zero.");
    } else if (Beta_nu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAngleDistribution): Beta_nu_inclination must be greater than or equal to zero.");
    } else if (eccentricity < 0.f || eccentricity > 1.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAngleDistribution): Eccentricity must be between 0 and 1.");
    }

    setPlantLeafAngleDistribution_private({plantID}, Beta_mu_inclination, Beta_nu_inclination, eccentricity, ellipse_rotation_degrees, true, true);
}

void PlantArchitecture::setPlantLeafAngleDistribution(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees) const {
    if (Beta_mu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAngleDistribution): Beta_mu_inclination must be greater than or equal to zero.");
    } else if (Beta_nu_inclination <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAngleDistribution): Beta_nu_inclination must be greater than or equal to zero.");
    } else if (eccentricity < 0.f || eccentricity > 1.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAngleDistribution): Eccentricity must be between 0 and 1.");
    }

    setPlantLeafAngleDistribution_private(plantIDs, Beta_mu_inclination, Beta_nu_inclination, eccentricity, ellipse_rotation_degrees, true, true);
}


helios::vec3 PlantArchitecture::getPlantBasePosition(const uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantBasePosition): Plant with ID of " + std::to_string(plantID) + " has no shoots, so could not get a base position.");
    }
    return plant_instances.at(plantID).base_position;
}

std::vector<helios::vec3> PlantArchitecture::getPlantBasePosition(const std::vector<uint> &plantIDs) const {
    std::vector<vec3> positions;
    positions.reserve(plantIDs.size());
    for (uint plantID: plantIDs) {
        positions.push_back(getPlantBasePosition(plantID));
    }
    return positions;
}

float PlantArchitecture::sumPlantLeafArea(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::sumPlantLeafArea): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> leaf_objIDs = getPlantLeafObjectIDs(plantID);

    float area = 0;
    for (uint objID: leaf_objIDs) {
        area += context_ptr->getObjectArea(objID);
    }

    return area;
}

float PlantArchitecture::getPlantStemHeight(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantStemHeight): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto base_shoot_ptr = plant_instances.at(plantID).shoot_tree.front();

    std::vector<uint> stem_objID{base_shoot_ptr->internode_tube_objID};

    if (!context_ptr->doesObjectExist(stem_objID.front())) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantStemHeight): The plant does not contain any geometry.");
    }

    // check if there was an appended shoot on this same shoot
    if (base_shoot_ptr->childIDs.find(base_shoot_ptr->current_node_number - 1) != base_shoot_ptr->childIDs.end()) {
        auto terminal_children = base_shoot_ptr->childIDs.at(base_shoot_ptr->current_node_number - 1);
        for (uint childID: terminal_children) {
            auto child_shoot_ptr = plant_instances.at(plantID).shoot_tree.at(childID);
            if (child_shoot_ptr->rank == base_shoot_ptr->rank) {
                if (context_ptr->doesObjectExist(child_shoot_ptr->internode_tube_objID)) {
                    stem_objID.push_back(child_shoot_ptr->internode_tube_objID);
                }
            }
        }
    }

    vec3 min_box;
    vec3 max_box;
    context_ptr->getObjectBoundingBox(stem_objID, min_box, max_box);

    return max_box.z - min_box.z;
}


float PlantArchitecture::getPlantHeight(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantHeight): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    vec3 min_box;
    vec3 max_box;
    context_ptr->getObjectBoundingBox(getAllPlantObjectIDs(plantID), min_box, max_box);

    return max_box.z - min_box.z;
}

std::vector<float> PlantArchitecture::getPlantLeafInclinationAngleDistribution(uint plantID, uint Nbins, bool normalize) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantLeafInclinationAngleDistribution): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    const std::vector<uint> leaf_objIDs = getPlantLeafObjectIDs(plantID);
    const std::vector<uint> leaf_UUIDs = context_ptr->getObjectPrimitiveUUIDs(leaf_objIDs);

    std::vector<float> leaf_inclination_angles(Nbins, 0.f);
    const float dtheta = 0.5f * PI_F / float(Nbins);
    for (const uint UUID: leaf_UUIDs) {
        const vec3 normal = context_ptr->getPrimitiveNormal(UUID);
        const float theta = acos_safe(fabs(normal.z));
        const float area = context_ptr->getPrimitiveArea(UUID);
        uint bin = static_cast<uint>(std::floor(theta / dtheta));
        if (bin >= Nbins) {
            bin = Nbins - 1; // Ensure bin index is within range
        }
        if (!std::isnan(area)) {
            leaf_inclination_angles.at(bin) += area;
        }
    }

    if (normalize) {
        const float sum = helios::sum(leaf_inclination_angles);
        if (sum > 0.f) {
            for (float &angle: leaf_inclination_angles) {
                angle /= sum;
            }
        }
    }

    return leaf_inclination_angles;
}

std::vector<float> PlantArchitecture::getPlantLeafInclinationAngleDistribution(const std::vector<uint> &plantIDs, uint Nbins, bool normalize) const {
    std::vector<float> leaf_inclination_angles(Nbins, 0.f);
    for (const uint plantID: plantIDs) {
        leaf_inclination_angles += getPlantLeafInclinationAngleDistribution(plantID, Nbins, false);
    }

    if (normalize) {
        const float sum = helios::sum(leaf_inclination_angles);
        if (sum > 0.f) {
            for (float &angle: leaf_inclination_angles) {
                angle /= sum;
            }
        }
    }

    return leaf_inclination_angles;
}

std::vector<float> PlantArchitecture::getPlantLeafAzimuthAngleDistribution(uint plantID, uint Nbins, bool normalize) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantLeafAzimuthAngleDistribution): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    const std::vector<uint> leaf_objIDs = getPlantLeafObjectIDs(plantID);
    const std::vector<uint> leaf_UUIDs = context_ptr->getObjectPrimitiveUUIDs(leaf_objIDs);

    std::vector<float> leaf_azimuth_angles(Nbins, 0.f);
    const float dtheta = 2.f * PI_F / static_cast<float>(Nbins);
    for (const uint UUID: leaf_UUIDs) {
        const vec3 normal = context_ptr->getPrimitiveNormal(UUID);
        const float phi = cart2sphere(normal).azimuth;
        const float area = context_ptr->getPrimitiveArea(UUID);
        uint bin = static_cast<uint>(std::floor(phi / dtheta));
        if (bin >= Nbins) {
            bin = Nbins - 1; // Ensure bin index is within range
        }
        if (!std::isnan(area)) {
            leaf_azimuth_angles.at(bin) += area;
        }
    }

    if (normalize) {
        const float sum = helios::sum(leaf_azimuth_angles);
        if (sum > 0.f) {
            for (float &angle: leaf_azimuth_angles) {
                angle /= sum;
            }
        }
    }

    return leaf_azimuth_angles;
}

std::vector<float> PlantArchitecture::getPlantLeafAzimuthAngleDistribution(const std::vector<uint> &plantIDs, uint Nbins, bool normalize) const {
    std::vector<float> leaf_azimuth_angles(Nbins, 0.f);
    for (const uint plantID: plantIDs) {
        leaf_azimuth_angles += getPlantLeafAzimuthAngleDistribution(plantID, Nbins, false);
    }

    if (normalize) {
        const float sum = helios::sum(leaf_azimuth_angles);
        if (sum > 0.f) {
            for (float &angle: leaf_azimuth_angles) {
                angle /= sum;
            }
        }
    }

    return leaf_azimuth_angles;
}


uint PlantArchitecture::getPlantLeafCount(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantLeafCount): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    return getPlantLeafObjectIDs(plantID).size();
}

std::vector<helios::vec3> PlantArchitecture::getPlantLeafBases(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantLeafBases): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<vec3> leaf_bases;

    // First calculate total size needed to avoid reallocations
    size_t total_size = 0;
    for (const auto &shoot: plant_instances.at(plantID).shoot_tree) {
        for (const auto &phytomer: shoot->phytomers) {
            total_size += phytomer->leaf_bases.size() * phytomer->leaf_bases.front().size();
        }
    }
    leaf_bases.reserve(total_size);

    // Now collect all leaf bases by appending at the end
    for (const auto &shoot: plant_instances.at(plantID).shoot_tree) {
        for (const auto &phytomer: shoot->phytomers) {
            std::vector<vec3> bases_flat = flatten(phytomer->leaf_bases);
            leaf_bases.insert(leaf_bases.end(), bases_flat.begin(), bases_flat.end());
        }
    }

    return leaf_bases;
}

std::vector<helios::vec3> PlantArchitecture::getPlantLeafBases(const std::vector<uint> &plantIDs) const {
    std::vector<helios::vec3> leaf_bases;
    for (const uint plantID: plantIDs) {
        auto bases = getPlantLeafBases(plantID);
        leaf_bases.insert(leaf_bases.end(), bases.begin(), bases.end());
    }
    return leaf_bases;
}

bool PlantArchitecture::isPlantDormant(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::isPlantDormant): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for (const auto &shoot: plant_instances.at(plantID).shoot_tree) {
        if (!shoot->isdormant) {
            return false;
        }
    }

    return true;
}

void PlantArchitecture::writePlantMeshVertices(uint plantID, const std::string &filename) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantMeshVertices): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> plant_UUIDs = getAllPlantUUIDs(plantID);

    std::ofstream file;
    file.open(filename);

    if (!file.is_open()) {
        helios_runtime_error("ERROR (PlantArchitecture::writePlantMeshVertices): Could not open file " + filename + " for writing.");
    }

    for (uint UUID: plant_UUIDs) {
        std::vector<vec3> vertex = context_ptr->getPrimitiveVertices(UUID);
        for (vec3 &v: vertex) {
            file << v.x << " " << v.y << " " << v.z << std::endl;
        }
    }

    file.close();
}

void PlantArchitecture::setPlantAge(uint plantID, float a_current_age) {
    //\todo
    //    this->current_age = current_age;
}

std::string PlantArchitecture::getPlantName(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantName): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    return plant_instances.at(plantID).plant_name;
}

float PlantArchitecture::getPlantAge(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAge): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAge): Plant with ID of " + std::to_string(plantID) + " has no shoots, so could not get a base position.");
    }
    return plant_instances.at(plantID).current_age;
}

void PlantArchitecture::harvestPlant(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::harvestPlant): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for (auto &shoot: plant_instances.at(plantID).shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {
            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    if (fbud.state != BUD_DORMANT) {
                        phytomer->setFloralBudState(BUD_DEAD, fbud);
                    }
                }
            }
        }
    }
}

void PlantArchitecture::removeShootLeaves(uint plantID, uint shootID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::removePlantLeaves): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::removeShootLeaves): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto &shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    for (auto &phytomer: shoot->phytomers) {
        phytomer->removeLeaf();
    }
}

void PlantArchitecture::removePlantLeaves(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::removePlantLeaves): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for (auto &shoot: plant_instances.at(plantID).shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {
            phytomer->removeLeaf();
        }
    }
}

void PlantArchitecture::makePlantDormant(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::makePlantDormant): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for (auto &shoot: plant_instances.at(plantID).shoot_tree) {
        shoot->makeDormant();
    }
    plant_instances.at(plantID).time_since_dormancy = 0;
}

void PlantArchitecture::breakPlantDormancy(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::breakPlantDormancy): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for (auto &shoot: plant_instances.at(plantID).shoot_tree) {
        shoot->breakDormancy();
    }
}

void PlantArchitecture::pruneBranch(uint plantID, uint shootID, uint node_index) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneBranch): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneBranch): Shoot with ID of " + std::to_string(shootID) + " does not exist on plant " + std::to_string(plantID) + ".");
    } else if (node_index >= plant_instances.at(plantID).shoot_tree.at(shootID)->current_node_number) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneBranch): Node index " + std::to_string(node_index) + " is out of range for shoot " + std::to_string(shootID) + ".");
    }

    auto &shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    shoot->phytomers.at(node_index)->deletePhytomer();

    if (plant_instances.at(plantID).shoot_tree.empty()) {
        std::cout << "WARNING (PlantArchitecture::pruneBranch): Plant " << plantID << " base shoot was pruned." << std::endl;
    }
}

// fallback axis if v×u is (near) zero:
static vec3 orthonormal_axis(const vec3 &v) {
    // try X axis
    vec3 ax = cross(v, vec3(1.f, 0.f, 0.f));
    if (ax.magnitude() < 1e-6f)
        ax = cross(v, vec3(0.f, 1.f, 0.f));
    return ax.normalize();
}

// Rodrigues formula: rotate v about unit‐axis k by angle α
static vec3 rodrigues(const vec3 &v, const vec3 &k, float a) {
    float c = std::cos(a);
    float s = std::sin(a);
    // dot = k·v
    float kv = k * v;
    return v * c + cross(k, v) * s + k * (kv * (1.f - c));
}

void PlantArchitecture::setPlantLeafAngleDistribution_private(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity_azimuth, float ellipse_rotation_azimuth_degrees, bool set_elevation,
                                                              bool set_azimuth) const {
    for (uint plantID: plantIDs) {
        if (plant_instances.find(plantID) == plant_instances.end()) {
            helios_runtime_error("ERROR (PlantArchitecture::setPlantLeafAngleDistribution): Plant with ID of " + std::to_string(plantID) + " does not exist.");
        }
    }

    // ── 2) Gather leaves ────────────────────────────────────────────────────
    std::vector<uint> objIDs = getPlantLeafObjectIDs(plantIDs);
    std::vector<vec3> bases = getPlantLeafBases(plantIDs);
    size_t N = objIDs.size();
    assert(bases.size() == N);
    if (N == 0 || (!set_elevation && !set_azimuth))
        return;

    // ── 3) Sample current & target (θ,φ) ───────────────────────────────────
    std::vector<float> theta(N), phi(N), theta_t(N), phi_t(N);
    for (size_t i = 0; i < N; ++i) {
        // current normal → (θ,φ)
        vec3 n0 = context_ptr->getObjectAverageNormal(objIDs[i]);
        if (!std::isfinite(n0.x) || !std::isfinite(n0.y) || !std::isfinite(n0.z) || n0.magnitude() < 1e-6f) {
            n0 = vec3(0.f, 0.f, 1.f);
        } else {
            n0 = n0.normalize();
        }
        n0.z = fabs(n0.z);
        SphericalCoord sc = cart2sphere(n0);
        theta[i] = sc.zenith;
        phi[i] = sc.azimuth;

        // target angles
        if (set_elevation && !set_azimuth) {
            theta_t[i] = sample_Beta_distribution(Beta_mu_inclination, Beta_nu_inclination, context_ptr->getRandomGenerator());
            phi_t[i] = phi[i];
        } else if (!set_elevation && set_azimuth) {
            theta_t[i] = theta[i];
            phi_t[i] = sample_ellipsoidal_azimuth(eccentricity_azimuth, ellipse_rotation_azimuth_degrees, context_ptr->getRandomGenerator());
        } else { // both elevation & azimuth
            theta_t[i] = sample_Beta_distribution(Beta_mu_inclination, Beta_nu_inclination, context_ptr->getRandomGenerator());
            phi_t[i] = sample_ellipsoidal_azimuth(eccentricity_azimuth, ellipse_rotation_azimuth_degrees, context_ptr->getRandomGenerator());
        }
    }

    // ── 4) Pure-1D shortcuts ─────────────────────────────────────────────────
    if (set_elevation && !set_azimuth) {
        // only θ changes
        for (size_t i = 0; i < N; ++i) {
            float elev = PI_F * 0.5f - theta_t[i];
            vec3 new_n = sphere2cart(SphericalCoord(1.f, elev, phi[i]));
            context_ptr->setObjectAverageNormal(objIDs[i], bases[i], new_n);
        }
        return;
    }
    if (!set_elevation && set_azimuth) {
        // only φ changes
        for (size_t i = 0; i < N; ++i) {
            float elev = PI_F * 0.5f - theta[i];
            vec3 new_n = sphere2cart(SphericalCoord(1.f, elev, phi_t[i]));
            context_ptr->setObjectAverageNormal(objIDs[i], bases[i], new_n);
        }
        return;
    }

    // ── 5) Full 2-D case: build V0/V1 ───────────────────────────────────────
    std::vector<vec3> V0(N), V1(N);
    for (size_t i = 0; i < N; ++i) {
        float e0 = PI_F * 0.5f - theta[i];
        float e1 = PI_F * 0.5f - theta_t[i];
        V0[i] = sphere2cart(SphericalCoord(1.f, e0, phi[i]));
        V1[i] = sphere2cart(SphericalCoord(1.f, e1, phi_t[i]));
    }

    // ── 6) Solve assignment ─────────────────────────────────────────────────
    std::vector<int> assignment(N);
    {
        HungarianAlgorithm hung;
        std::vector<std::vector<double>> C(N, std::vector<double>(N));
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double d = (V0[i] - V1[j]).magnitude();
                C[i][j] = std::isfinite(d) ? d : ((std::numeric_limits<double>::max)() * 0.5);
            }
        }
        hung.Solve(C, assignment);
    }

    // ── 7) Rotate & write back ───────────────────────────────────────────────
    for (size_t i = 0; i < N; ++i) {
        int j = assignment[i];
        // pick your target; if out-of-bounds, just keep the original V0[i]
        vec3 v = V0[i];
        vec3 u = (j >= 0 && j < (int) N ? V1[j] : V0[i]);

        // normalize
        v = (v.magnitude() < 1e-6f ? vec3(0, 0, 1) : v.normalize());
        u = (u.magnitude() < 1e-6f ? vec3(0, 0, 1) : u.normalize());

        // minimal‐angle between them
        float dot = std::clamp(v * u, -1.f, 1.f);
        float ang = acos_safe(dot);

        // choose axis
        vec3 axis = cross(v, u);
        if (!set_elevation && set_azimuth) {
            // if it's really just φ, rotate about Z
            axis = vec3(0.f, 0.f, 1.f);
        } else if (axis.magnitude() < 1e-6f) {
            // degenerate → pick any perpendicular
            axis = orthonormal_axis(v);
        } else {
            axis = axis.normalize();
        }

        // apply Rodrigues + final guard
        vec3 r = rodrigues(v, axis, ang);
        if (!std::isfinite(r.x) || !std::isfinite(r.y) || !std::isfinite(r.z) || r.magnitude() < 1e-6f) {
            r = u;
        } else {
            r = r.normalize();
        }

        // convert back & set
        SphericalCoord out = cart2sphere(r);
        float new_elev = PI_F * 0.5f - out.zenith;
        vec3 new_n = sphere2cart(SphericalCoord(1.f, new_elev, out.azimuth));
        context_ptr->setObjectAverageNormal(objIDs[i], bases[i], new_n);
    }
}

//     std::vector<uint> objIDs_leaf = getPlantLeafObjectIDs(plantIDs);
//     std::vector<vec3> leaf_bases = getPlantLeafBases(plantIDs);
//
//
//     assert( objIDs_leaf.size() == leaf_bases.size() );
//
//
//     const size_t Nleaves = objIDs_leaf.size();
//
//
//     std::vector<float> thetaL(Nleaves);
//     std::vector<float> phiL(Nleaves);
//     std::vector<float> thetaL_target(Nleaves);
//     std::vector<float> phiL_target(Nleaves);
//     for ( int i=0; i<Nleaves; i++ ) {
//         vec3 norm = context_ptr->getObjectAverageNormal(objIDs_leaf.at(i));
//         norm.z = fabs(norm.z);
//         SphericalCoord leaf_angle = cart2sphere(norm);
//         thetaL.at(i) = leaf_angle.zenith;
//         phiL.at(i) = leaf_angle.azimuth;
//         if ( set_elevation && !set_azimuth ) { //only set elevation
//             thetaL_target.at(i) = sample_Beta_distribution(Beta_mu_inclination, Beta_nu_inclination, context_ptr->getRandomGenerator());
//             phiL_target.at(i) = phiL.at(i);
//         }else if ( !set_elevation && set_azimuth ) {
//             thetaL_target.at(i) = thetaL.at(i);
//             phiL_target.at(i) = sample_ellipsoidal_azimuth( eccentricity_azimuth, ellipse_rotation_azimuth_degrees, context_ptr->getRandomGenerator() );
//         }else if ( set_elevation && set_azimuth ) {
//             thetaL_target.at(i) = sample_Beta_distribution(Beta_mu_inclination, Beta_nu_inclination, context_ptr->getRandomGenerator());
//             phiL_target.at(i) = sample_ellipsoidal_azimuth( eccentricity_azimuth, ellipse_rotation_azimuth_degrees, context_ptr->getRandomGenerator() );
//         }else {
//             return;
//         }
//     }
//
//
//     // ── Convert both sets to Cartesian using sphere2cart() ─────────────────
//     std::vector<vec3> V0, V1;
//     V0.reserve(Nleaves);  V1.reserve(Nleaves);
//     for (size_t i = 0; i < Nleaves; ++i) {
//         // Helios uses (radius, elevation, azimuth), where elevation = π/2 – zenith
//         float elev0 = PI_F*0.5f - thetaL[i];
//         SphericalCoord sc0(1.f, elev0, phiL[i]);
//         V0.push_back(sphere2cart(sc0));
//
//
//         float elev1 = PI_F*0.5f - thetaL_target[i];
//         SphericalCoord sc1(1.f, elev1, phiL_target[i]);
//         V1.push_back(sphere2cart(sc1));
//     }
//
//
//     // ── Build cost matrix of great‐circle angles ───────────────────────────
//     std::vector<std::vector<double>> cost(Nleaves, std::vector<double>(Nleaves));
//     for (size_t i = 0; i < Nleaves; ++i) {
//         for (size_t j = 0; j < Nleaves; ++j) {
//             float d = std::clamp(V0[i] * V1[j], -1.f, 1.f);  // dot product via operator*
//             cost[i][j] = std::acos(static_cast<double>(d));
//         }
//     }
//
//
//     // ── Global minimal‐sum assignment ──────────────────────────────────────
//     HungarianAlgorithm hungarian;
//     std::vector<int> assignment;
//     double totalCost = hungarian.Solve(cost, assignment);
//
//
//     // ── Rotate each V0[i] → V1[assignment[i]] by minimal axis–angle ────────
//     std::vector<vec3> V0_matched(Nleaves);
//     for (size_t i = 0; i < Nleaves; ++i) {
//         vec3 v = V0[i];
//         vec3 u = V1[assignment[i]];
//
//
//         float dot = std::clamp(v * u, -1.f, 1.f);
//         float a   = std::acos(dot);
//
//
//         vec3 axis = cross(v, u);
//         if (axis.magnitude() < 1e-6f)
//             axis = orthonormal_axis(v);
//         else
//             axis = axis.normalize();
//
//
//         V0_matched[i] = rodrigues(v, axis, a);
//     }
//
//
//     // ── Convert rotated vectors back to (θ,φ) via cart2sphere() ────
//     std::vector<float> theta_matched(Nleaves), phi_matched(Nleaves);
//     for (size_t i = 0; i < Nleaves; ++i) {
//         SphericalCoord out = cart2sphere(V0_matched[i]);
//         theta_matched[i] = out.zenith;      // your convention: zenith in [0,π]
//         phi_matched  [i] = out.azimuth;     // in [0,2π)
//
//
//         vec3 new_normal = sphere2cart(SphericalCoord(1.f, PI_F*0.5f - theta_matched[i], phi_matched[i]));
//         context_ptr->setObjectAverageNormal(objIDs_leaf.at(i), leaf_bases.at(i), new_normal);
//     }
//
//
// }


uint PlantArchitecture::getShootNodeCount(uint plantID, uint shootID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootNodeCount): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.size() <= shootID) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootNodeCount): Shoot ID is out of range.");
    }
    return plant_instances.at(plantID).shoot_tree.at(shootID)->current_node_number;
}

float PlantArchitecture::getShootTaper(uint plantID, uint shootID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootTaper): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.size() <= shootID) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootTaper): Shoot ID is out of range.");
    }

    float r0 = plant_instances.at(plantID).shoot_tree.at(shootID)->shoot_internode_radii.front().front();
    float r1 = plant_instances.at(plantID).shoot_tree.at(shootID)->shoot_internode_radii.back().back();

    float taper = (r0 - r1) / r0;
    if (taper < 0) {
        taper = 0;
    } else if (taper > 1) {
        taper = 1;
    }

    return taper;
}

std::vector<uint> PlantArchitecture::getAllPlantIDs() const {
    std::vector<uint> objIDs;
    objIDs.reserve(plant_instances.size());

    for (const auto &plant: plant_instances) {
        objIDs.push_back(plant.first);
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getAllPlantObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getAllPlantObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    for (const auto &shoot: plant_instances.at(plantID).shoot_tree) {
        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            objIDs.push_back(shoot->internode_tube_objID);
        }
        for (const auto &phytomer: shoot->phytomers) {
            std::vector<uint> petiole_objIDs_flat = flatten(phytomer->petiole_objIDs);
            objIDs.insert(objIDs.end(), petiole_objIDs_flat.begin(), petiole_objIDs_flat.end());
            std::vector<uint> leaf_objIDs_flat = flatten(phytomer->leaf_objIDs);
            objIDs.insert(objIDs.end(), leaf_objIDs_flat.begin(), leaf_objIDs_flat.end());
            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    std::vector<uint> inflorescence_objIDs_flat = fbud.inflorescence_objIDs;
                    objIDs.insert(objIDs.end(), inflorescence_objIDs_flat.begin(), inflorescence_objIDs_flat.end());
                    std::vector<uint> peduncle_objIDs_flat = fbud.peduncle_objIDs;
                    objIDs.insert(objIDs.end(), peduncle_objIDs_flat.begin(), peduncle_objIDs_flat.end());
                }
            }
        }
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getAllPlantUUIDs(uint plantID) const { return context_ptr->getObjectPrimitiveUUIDs(getAllPlantObjectIDs(plantID)); }

std::vector<uint> PlantArchitecture::getPlantInternodeObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInternodeObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (auto &shoot: shoot_tree) {
        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            objIDs.push_back(shoot->internode_tube_objID);
        }
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getPlantPetioleObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantPetioleObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (auto &shoot: shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {
            for (auto &petiole: phytomer->petiole_objIDs) {
                objIDs.insert(objIDs.end(), petiole.begin(), petiole.end());
            }
        }
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getPlantLeafObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantLeafObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (auto &shoot: shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {
            for (auto &leaf_objID: phytomer->leaf_objIDs) {
                objIDs.insert(objIDs.end(), leaf_objID.begin(), leaf_objID.end());
            }
        }
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getPlantLeafObjectIDs(const std::vector<uint> &plantIDs) const {
    std::vector<uint> objIDs;
    objIDs.reserve(50 * plantIDs.size()); // assume we have at least 50 leaves/plant
    for (const uint plantID: plantIDs) {
        std::vector<uint> leaf_objIDs = getPlantLeafObjectIDs(plantID);
        objIDs.insert(objIDs.end(), leaf_objIDs.begin(), leaf_objIDs.end());
    }
    return objIDs;
}

std::vector<uint> PlantArchitecture::getPlantPeduncleObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantPeduncleObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (auto &shoot: shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {
            for (auto &petiole: phytomer->floral_buds) {
                for (auto &fbud: petiole) {
                    objIDs.insert(objIDs.end(), fbud.peduncle_objIDs.begin(), fbud.peduncle_objIDs.end());
                }
            }
        }
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getPlantFlowerObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInflorescenceObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (auto &shoot: shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {
            for (int petiole = 0; petiole < phytomer->floral_buds.size(); petiole++) {
                for (int bud = 0; bud < phytomer->floral_buds.at(petiole).size(); bud++) {
                    if (phytomer->floral_buds.at(petiole).at(bud).state == BUD_FLOWER_OPEN || phytomer->floral_buds.at(petiole).at(bud).state == BUD_FLOWER_CLOSED) {
                        objIDs.insert(objIDs.end(), phytomer->floral_buds.at(petiole).at(bud).inflorescence_objIDs.begin(), phytomer->floral_buds.at(petiole).at(bud).inflorescence_objIDs.end());
                    }
                }
            }
        }
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getPlantFruitObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInflorescenceObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (auto &shoot: shoot_tree) {
        for (auto &phytomer: shoot->phytomers) {
            for (int petiole = 0; petiole < phytomer->floral_buds.size(); petiole++) {
                for (int bud = 0; bud < phytomer->floral_buds.at(petiole).size(); bud++) {
                    if (phytomer->floral_buds.at(petiole).at(bud).state == BUD_FRUITING) {
                        objIDs.insert(objIDs.end(), phytomer->floral_buds.at(petiole).at(bud).inflorescence_objIDs.begin(), phytomer->floral_buds.at(petiole).at(bud).inflorescence_objIDs.end());
                    }
                }
            }
        }
    }

    return objIDs;
}

std::vector<uint> PlantArchitecture::getAllUUIDs() const {
    std::vector<uint> UUIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> UUIDs = getAllPlantUUIDs(instance.first);
        UUIDs_all.insert(UUIDs_all.end(), UUIDs.begin(), UUIDs.end());
    }
    return UUIDs_all;
}

std::vector<uint> PlantArchitecture::getAllLeafUUIDs() const {
    std::vector<uint> UUIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> objIDs = getPlantLeafObjectIDs(instance.first);
        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objIDs);
        UUIDs_all.insert(UUIDs_all.end(), UUIDs.begin(), UUIDs.end());
    }
    return UUIDs_all;
}

std::vector<uint> PlantArchitecture::getAllInternodeUUIDs() const {
    std::vector<uint> UUIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> objIDs = getPlantInternodeObjectIDs(instance.first);
        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objIDs);
        UUIDs_all.insert(UUIDs_all.end(), UUIDs.begin(), UUIDs.end());
    }
    return UUIDs_all;
}

std::vector<uint> PlantArchitecture::getAllPetioleUUIDs() const {
    std::vector<uint> UUIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> objIDs = getPlantPetioleObjectIDs(instance.first);
        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objIDs);
        UUIDs_all.insert(UUIDs_all.end(), UUIDs.begin(), UUIDs.end());
    }
    return UUIDs_all;
}

std::vector<uint> PlantArchitecture::getAllPeduncleUUIDs() const {
    std::vector<uint> UUIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> objIDs = getPlantPeduncleObjectIDs(instance.first);
        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objIDs);
        UUIDs_all.insert(UUIDs_all.end(), UUIDs.begin(), UUIDs.end());
    }
    return UUIDs_all;
}

std::vector<uint> PlantArchitecture::getAllFlowerUUIDs() const {
    std::vector<uint> UUIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> objIDs = getPlantFlowerObjectIDs(instance.first);
        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objIDs);
        UUIDs_all.insert(UUIDs_all.end(), UUIDs.begin(), UUIDs.end());
    }
    return UUIDs_all;
}

std::vector<uint> PlantArchitecture::getAllFruitUUIDs() const {
    std::vector<uint> UUIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> objIDs = getPlantFruitObjectIDs(instance.first);
        std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objIDs);
        UUIDs_all.insert(UUIDs_all.end(), UUIDs.begin(), UUIDs.end());
    }
    return UUIDs_all;
}

std::vector<uint> PlantArchitecture::getAllObjectIDs() const {
    std::vector<uint> objIDs_all;
    for (const auto &instance: plant_instances) {
        std::vector<uint> objIDs = getAllPlantObjectIDs(instance.first);
        objIDs_all.insert(objIDs_all.end(), objIDs.begin(), objIDs.end());
    }
    return objIDs_all;
}

void PlantArchitecture::enableCarbohydrateModel() { carbon_model_enabled = true; }

void PlantArchitecture::disableCarbohydrateModel() { carbon_model_enabled = false; }

uint PlantArchitecture::addPlantInstance(const helios::vec3 &base_position, float current_age) {
    if (current_age < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::addPlantInstance): Current age must be greater than or equal to zero.");
    }

    PlantInstance instance(base_position, current_age, "custom", context_ptr);

    plant_instances.emplace(plant_count, instance);

    plant_count++;

    return plant_count - 1;
}

uint PlantArchitecture::duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, const AxisRotation &base_rotation, float current_age) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::duplicatePlantInstance): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto plant_shoot_tree = &plant_instances.at(plantID).shoot_tree;

    uint plantID_new = addPlantInstance(base_position, current_age);

    if (plant_shoot_tree->empty()) { // no shoots to add
        return plantID_new;
    }
    if (plant_shoot_tree->front()->phytomers.empty()) { // no phytomers to add
        return plantID_new;
    }

    for (const auto &shoot: *plant_shoot_tree) {
        uint shootID_new = 0; // ID of the new shoot; will be set once the shoot is created on the first loop iteration
        for (int node = 0; node < shoot->current_node_number; node++) {
            auto phytomer = shoot->phytomers.at(node);
            float internode_radius = phytomer->internode_radius_initial;
            float internode_length_max = phytomer->internode_length_max;
            float internode_scale_factor_fraction = phytomer->current_internode_scale_factor;
            float leaf_scale_factor_fraction = 1.f; // phytomer->current_leaf_scale_factor;

            if (node == 0) { // first phytomer on shoot
                AxisRotation original_base_rotation = shoot->base_rotation;
                if (shoot->parent_shoot_ID == -1) { // first shoot on plant
                    shootID_new = addBaseStemShoot(plantID_new, 1, original_base_rotation + base_rotation, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction, 0, shoot->shoot_type_label);
                } else { // child shoot
                    uint parent_node = plant_shoot_tree->at(shoot->parent_shoot_ID)->parent_node_index;
                    uint parent_petiole_index = 0;
                    for (auto &petiole: phytomer->axillary_vegetative_buds) {
                        shootID_new = addChildShoot(plantID_new, shoot->parent_shoot_ID, parent_node, 1, original_base_rotation, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction, 0,
                                                    shoot->shoot_type_label, parent_petiole_index);
                        parent_petiole_index++;
                    }
                }
            } else {
                // each phytomer needs to be added one-by-one to account for possible internodes/leaves that are not fully elongated
                appendPhytomerToShoot(plantID_new, shootID_new, shoot_types.at(shoot->shoot_type_label).phytomer_parameters, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction);
            }
            auto phytomer_new = plant_instances.at(plantID_new).shoot_tree.at(shootID_new)->phytomers.back();
            for (uint petiole_index = 0; petiole_index < phytomer->petiole_objIDs.size(); petiole_index++) {
                phytomer_new->setLeafScaleFraction(petiole_index, phytomer->current_leaf_scale_factor.at(petiole_index));
            }
        }
    }

    return plantID_new;
}

void PlantArchitecture::deletePlantInstance(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        return;
    }

    context_ptr->deleteObject(getAllPlantObjectIDs(plantID));

    plant_instances.erase(plantID);
}

void PlantArchitecture::deletePlantInstance(const std::vector<uint> &plantIDs) {
    for (uint ID: plantIDs) {
        deletePlantInstance(ID);
    }
}

void PlantArchitecture::setPlantPhenologicalThresholds(uint plantID, float time_to_dormancy_break, float time_to_flower_initiation, float time_to_flower_opening, float time_to_fruit_set, float time_to_fruit_maturity, float time_to_dormancy,
                                                       float max_leaf_lifespan, bool is_evergreen) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantPhenologicalThresholds): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).dd_to_dormancy_break = time_to_dormancy_break;
    plant_instances.at(plantID).dd_to_flower_initiation = time_to_flower_initiation;
    plant_instances.at(plantID).dd_to_flower_opening = time_to_flower_opening;
    plant_instances.at(plantID).dd_to_fruit_set = time_to_fruit_set;
    plant_instances.at(plantID).dd_to_fruit_maturity = time_to_fruit_maturity;
    plant_instances.at(plantID).dd_to_dormancy = time_to_dormancy;
    if (max_leaf_lifespan == 0) {
        plant_instances.at(plantID).max_leaf_lifespan = 1e6;
    } else {
        plant_instances.at(plantID).max_leaf_lifespan = max_leaf_lifespan;
    }
    plant_instances.at(plantID).is_evergreen = is_evergreen;
}

void PlantArchitecture::setPlantCarbohydrateModelParameters(uint plantID, const CarbohydrateParameters &carb_parameters) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantCarbohydrateModelParameters): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    plant_instances.at(plantID).carb_parameters = carb_parameters;
}

void PlantArchitecture::setPlantCarbohydrateModelParameters(const std::vector<uint> &plantIDs, const CarbohydrateParameters &carb_parameters) {
    for (uint plantID: plantIDs) {
        setPlantCarbohydrateModelParameters(plantID, carb_parameters);
    }
}

void PlantArchitecture::disablePlantPhenology(uint plantID) {
    plant_instances.at(plantID).dd_to_dormancy_break = 0;
    plant_instances.at(plantID).dd_to_flower_initiation = -1;
    plant_instances.at(plantID).dd_to_flower_opening = -1;
    plant_instances.at(plantID).dd_to_fruit_set = -1;
    plant_instances.at(plantID).dd_to_fruit_maturity = -1;
    plant_instances.at(plantID).dd_to_dormancy = 1e6;
}

void PlantArchitecture::advanceTime(float time_step_days) {
    for (auto &[plantID, plant_instance]: plant_instances) {
        advanceTime(plantID, time_step_days);
    }
}

void PlantArchitecture::advanceTime(int time_step_years, float time_step_days) {
    for (auto &[plantID, plant_instance]: plant_instances) {
        advanceTime(plantID, float(time_step_years) * 365.f + time_step_days);
    }
}

void PlantArchitecture::advanceTime(uint plantID, float time_step_days) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::advanceTime): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    PlantInstance &plant_instance = plant_instances.at(plantID);

    auto shoot_tree = &plant_instance.shoot_tree;

    if (shoot_tree->empty()) {
        return;
    }

    // accounting for case of time_step_days>phyllochron_min
    float phyllochron_min = shoot_tree->front()->shoot_parameters.phyllochron_min.val();
    for (int i = 1; i < shoot_tree->size(); i++) {
        if (shoot_tree->at(i)->shoot_parameters.phyllochron_min.val() < phyllochron_min) {
            phyllochron_min = shoot_tree->at(i)->shoot_parameters.phyllochron_min.val();
        }
    }

    // **** accumulate photosynthate **** //
    if (carbon_model_enabled) {
        accumulateShootPhotosynthesis();
    }

    float dt_max_days;
    int Nsteps;

    if (time_step_days <= phyllochron_min) {
        Nsteps = time_step_days;
        dt_max_days = 1;
    } else {
        Nsteps = std::floor(time_step_days / phyllochron_min);
        dt_max_days = phyllochron_min;
    }

    float remainder_time = time_step_days - dt_max_days * float(Nsteps);
    if (remainder_time > 0.f) {
        Nsteps++;
    }
    for (int timestep = 0; timestep < Nsteps; timestep++) {
        if (timestep == Nsteps - 1 && remainder_time != 0.f) {
            dt_max_days = remainder_time;
        }


        if (plant_instance.current_age <= plant_instance.max_age && plant_instance.current_age + dt_max_days > plant_instance.max_age) {
            std::cout << "PlantArchitecture::advanceTime: Plant has reached its maximum supported age. No further growth will occur." << std::endl;
        } else if (plant_instance.current_age >= plant_instance.max_age) {
            // update Context geometry
            shoot_tree->front()->updateShootNodes(true);
            return;
        }

        plant_instance.current_age += dt_max_days;
        plant_instance.time_since_dormancy += dt_max_days;

        if (plant_instance.time_since_dormancy > plant_instance.dd_to_dormancy_break + plant_instance.dd_to_dormancy) {
            // std::cout << "Going dormant " << plant_instance.current_age << " " << plant_instance.time_since_dormancy << std::endl;
            plant_instance.time_since_dormancy = 0;
            for (const auto &shoot: *shoot_tree) {
                shoot->makeDormant();
                shoot->phyllochron_counter = 0;
            }
            harvestPlant(plantID);
            continue;
        }

        size_t shoot_count = shoot_tree->size();
        for (int i = 0; i < shoot_count; i++) {
            auto shoot = shoot_tree->at(i);

            for (auto &phytomer: shoot->phytomers) {
                phytomer->age += dt_max_days;

                if (phytomer->phytomer_parameters.phytomer_callback_function != nullptr) {
                    phytomer->phytomer_parameters.phytomer_callback_function(phytomer);
                }
            }

            // ****** PHENOLOGICAL TRANSITIONS ****** //

            // breaking dormancy
            if (shoot->isdormant && plant_instance.time_since_dormancy >= plant_instance.dd_to_dormancy_break) {
                shoot->phyllochron_counter = 0;
                shoot->breakDormancy();
            }

            if (shoot->isdormant) {
                // dormant, don't do anything
                continue;
            }

            for (auto &phytomer: shoot->phytomers) {
                if (phytomer->age > plant_instance.max_leaf_lifespan) {
                    // delete old leaves that exceed maximum lifespan
                    phytomer->removeLeaf();
                }

                if (phytomer->floral_buds.empty()) {
                    // no floral buds - skip this phytomer
                    continue;
                }

                for (auto &petiole: phytomer->floral_buds) {
                    for (auto &fbud: petiole) {
                        if (fbud.state != BUD_DORMANT && fbud.state != BUD_DEAD) {
                            fbud.time_counter += dt_max_days;
                        }

                        // -- Flowering -- //
                        if (shoot->shoot_parameters.phytomer_parameters.inflorescence.flower_prototype_function != nullptr) { // user defined a flower prototype function
                            // -- Flower initiation (closed flowers) -- //
                            if (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation >= 0.f) { // bud is active and flower initiation is enabled
                                if ((!shoot->shoot_parameters.flowers_require_dormancy && fbud.time_counter >= plant_instance.dd_to_flower_initiation) ||
                                    (shoot->shoot_parameters.flowers_require_dormancy && fbud.time_counter >= plant_instance.dd_to_flower_initiation)) {
                                    fbud.time_counter = 0;
                                    if (context_ptr->randu() < shoot->shoot_parameters.flower_bud_break_probability.val()) {
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
                            } else if ((fbud.state == BUD_FLOWER_CLOSED && plant_instance.dd_to_flower_opening >= 0.f) || (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation < 0.f && plant_instance.dd_to_flower_opening >= 0.f)) {
                                if (fbud.time_counter >= plant_instance.dd_to_flower_opening) {
                                    fbud.time_counter = 0;
                                    if (fbud.state == BUD_FLOWER_CLOSED) {
                                        phytomer->setFloralBudState(BUD_FLOWER_OPEN, fbud);
                                    } else {
                                        if (context_ptr->randu() < shoot->shoot_parameters.flower_bud_break_probability.val()) {
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
                            if ((fbud.state == BUD_FLOWER_OPEN && plant_instance.dd_to_fruit_set >= 0.f) || // flower opened and fruit set is enabled
                                (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation < 0.f && plant_instance.dd_to_flower_opening < 0.f && plant_instance.dd_to_fruit_set >= 0.f) || // jumped straight to fruit set with no flowering
                                (fbud.state == BUD_FLOWER_CLOSED && plant_instance.dd_to_flower_opening < 0.f && plant_instance.dd_to_fruit_set >= 0.f)) { // jumped from closed flower to fruit set with no flower opening
                                if (fbud.time_counter >= plant_instance.dd_to_fruit_set) {
                                    fbud.time_counter = 0;
                                    if (context_ptr->randu() < shoot->shoot_parameters.fruit_set_probability.val()) {
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

            // ****** GROWTH/SCALING OF CURRENT PHYTOMERS/FRUIT ****** //

            int node_index = 0;
            for (auto &phytomer: shoot->phytomers) {
                // scale internode length
                if (phytomer->current_internode_scale_factor < 1) {
                    float dL_internode = dt_max_days * shoot->elongation_rate_instantaneous * phytomer->internode_length_max;
                    float length_scale = fmin(1.f, (phytomer->getInternodeLength() + dL_internode) / phytomer->internode_length_max);
                    phytomer->setInternodeLengthScaleFraction(length_scale, false);
                }

                // scale internode girth
                if (shoot->shoot_parameters.girth_area_factor.val() > 0.f) {
                    if (carbon_model_enabled) {
                        incrementPhytomerInternodeGirth_carb(plantID, shoot->ID, node_index, dt_max_days, false);
                    } else {
                        incrementPhytomerInternodeGirth(plantID, shoot->ID, node_index, dt_max_days, false);
                    }
                }

                node_index++;
            }

            node_index = 0;
            for (auto &phytomer: shoot->phytomers) {
                // scale petiole/leaves
                if (phytomer->hasLeaf()) {
                    for (uint petiole_index = 0; petiole_index < phytomer->current_leaf_scale_factor.size(); petiole_index++) {

                        if (phytomer->current_leaf_scale_factor.at(petiole_index) >= 1) {
                            continue;
                        }

                        float tip_ind = ceil(float(phytomer->leaf_size_max.at(petiole_index).size() - 1) / 2.f);
                        float leaf_length = phytomer->current_leaf_scale_factor.at(petiole_index) * phytomer->leaf_size_max.at(petiole_index).at(tip_ind);
                        float dL_leaf = dt_max_days * shoot->elongation_rate_instantaneous * phytomer->leaf_size_max.at(petiole_index).at(tip_ind);
                        float scale = fmin(1.f, (leaf_length + dL_leaf) / phytomer->phytomer_parameters.leaf.prototype_scale.val());
                        phytomer->phytomer_parameters.leaf.prototype_scale.resample();
                        phytomer->setLeafScaleFraction(petiole_index, scale);
                    }
                }

                // Fruit Growth
                for (auto &petiole: phytomer->floral_buds) {
                    for (auto &fbud: petiole) {
                        // If the floral bud it in a 'fruiting' state, the fruit grows with time
                        if (fbud.state == BUD_FRUITING && fbud.time_counter > 0) {
                            float scale = fmin(1, 0.25f + 0.75f * fbud.time_counter / plant_instance.dd_to_fruit_maturity);
                            phytomer->setInflorescenceScaleFraction(fbud, scale);
                        }
                    }
                }

                // ****** NEW CHILD SHOOTS FROM VEGETATIVE BUDS ****** //
                uint parent_petiole_index = 0;
                for (auto &petiole: phytomer->axillary_vegetative_buds) {
                    for (auto &vbud: petiole) {
                        if (vbud.state == BUD_ACTIVE && phytomer->age + dt_max_days > shoot->shoot_parameters.vegetative_bud_break_time.val()) {
                            ShootParameters *new_shoot_parameters = &shoot_types.at(vbud.shoot_type_label);
                            int parent_node_count = shoot->current_node_number;

                            //                            float insertion_angle_adjustment = fmin(new_shoot_parameters->insertion_angle_tip.val() + new_shoot_parameters->insertion_angle_decay_rate.val() * float(parent_node_count -
                            //                            phytomer->shoot_index.x - 1), 90.f); AxisRotation base_rotation = make_AxisRotation(deg2rad(insertion_angle_adjustment), deg2rad(new_shoot_parameters->base_yaw.val()),
                            //                            deg2rad(new_shoot_parameters->base_roll.val())); new_shoot_parameters->base_yaw.resample(); if( new_shoot_parameters->insertion_angle_decay_rate.val()==0 ){
                            //                                new_shoot_parameters->insertion_angle_tip.resample();
                            //                            }
                            float insertion_angle_adjustment = fmin(shoot->shoot_parameters.insertion_angle_tip.val() + shoot->shoot_parameters.insertion_angle_decay_rate.val() * float(parent_node_count - phytomer->shoot_index.x - 1), 90.f);
                            AxisRotation base_rotation = make_AxisRotation(deg2rad(insertion_angle_adjustment), deg2rad(new_shoot_parameters->base_yaw.val()), deg2rad(new_shoot_parameters->base_roll.val()));
                            new_shoot_parameters->base_yaw.resample();
                            if (shoot->shoot_parameters.insertion_angle_decay_rate.val() == 0) {
                                shoot->shoot_parameters.insertion_angle_tip.resample();
                            }

                            // scale the shoot internode length based on proximity from the tip
                            float internode_length_max;
                            if (new_shoot_parameters->growth_requires_dormancy) {
                                internode_length_max = fmax(new_shoot_parameters->internode_length_max.val() - new_shoot_parameters->internode_length_decay_rate.val() * float(parent_node_count - phytomer->shoot_index.x - 1),
                                                            new_shoot_parameters->internode_length_min.val());
                            } else {
                                internode_length_max = new_shoot_parameters->internode_length_max.val();
                            }

                            float internode_radius = phytomer->internode_radius_initial;

                            uint childID = addChildShoot(plantID, shoot->ID, node_index, 1, base_rotation, internode_radius, internode_length_max, 0.01, 0.01, 0, vbud.shoot_type_label, parent_petiole_index);

                            phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                            vbud.shoot_ID = childID;
                            shoot_tree->at(childID)->isdormant = false;
                        }
                    }
                    parent_petiole_index++;
                }

                if (output_object_data.at("age")) {
                    if (shoot->build_context_geometry_internode) {
                        //\todo This is redundant and only needs to be done once per shoot
                        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                            context_ptr->setObjectData(shoot->internode_tube_objID, "age", phytomer->age);
                        }
                    }
                    if (phytomer->build_context_geometry_petiole) {
                        context_ptr->setObjectData(phytomer->petiole_objIDs, "age", phytomer->age);
                    }
                    context_ptr->setObjectData(phytomer->leaf_objIDs, "age", phytomer->age);
                }

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
            shoot->phyllochron_counter += dt_max_days;
            if (shoot->phyllochron_counter >= shoot->phyllochron_instantaneous && !shoot->phytomers.back()->isdormant) {
                float internode_radius = shoot->shoot_parameters.phytomer_parameters.internode.radius_initial.val();
                shoot->shoot_parameters.phytomer_parameters.internode.radius_initial.resample();
                float internode_length_max = shoot->internode_length_max_shoot_initial;
                appendPhytomerToShoot(plantID, shoot->ID, shoot_types.at(shoot->shoot_type_label).phytomer_parameters, internode_radius, internode_length_max, 0.01, 0.01); //\todo These factors should be set to be consistent with the shoot
                shoot->phyllochron_counter = shoot->phyllochron_counter - shoot->phyllochron_instantaneous;
            }

            // ****** EPICORMIC SHOOTS ****** //
            std::string epicormic_shoot_label = plant_instance.epicormic_shoot_probability_perlength_per_day.first;
            if (!epicormic_shoot_label.empty()) {
                std::vector<float> epicormic_fraction;
                uint Nepicormic = shoot->sampleEpicormicShoot(time_step_days, epicormic_fraction);
                for (int s = 0; s < Nepicormic; s++) {
                    float internode_radius = shoot_types.at(epicormic_shoot_label).phytomer_parameters.internode.radius_initial.val();
                    shoot_types.at(epicormic_shoot_label).phytomer_parameters.internode.radius_initial.resample();
                    float internode_length_max = shoot_types.at(epicormic_shoot_label).internode_length_max.val();
                    shoot_types.at(epicormic_shoot_label).internode_length_max.resample();
                    std::cout << "Adding epicormic shoot" << std::endl;
                    addEpicormicShoot(plantID, shoot->ID, epicormic_fraction.at(s), 1, 0, internode_radius, internode_length_max, 0.01, 0.01, 0, epicormic_shoot_label);
                }
            }
            if (carbon_model_enabled) {
                if (output_object_data.find("carbohydrate_concentration") != output_object_data.end() && context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                    float shoot_volume = shoot->calculateShootInternodeVolume();
                    context_ptr->setObjectData(shoot->internode_tube_objID, "carbohydrate_concentration", shoot->carbohydrate_pool_molC / shoot_volume);
                }
            }
        }

        // **** subtract maintenance carbon costs **** //
        if (carbon_model_enabled) {
            subtractShootMaintenanceCarbon(dt_max_days);
            subtractShootGrowthCarbon();
            checkCarbonPool_transferCarbon(dt_max_days);
            checkCarbonPool_adjustPhyllochron(dt_max_days);
            checkCarbonPool_abortOrgans(dt_max_days);
        }
    }

    // update Context geometry
    shoot_tree->front()->updateShootNodes(true);

    // *** ground collision detection *** //
    if (ground_clipping_height != -99999) {

        pruneGroundCollisions( plantID );

    }

    // Assign current volume as old volume for your next timestep
    for (auto &shoot: *shoot_tree) {
        float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume(); // Find current volume for each shoot in the plant
        shoot->old_shoot_volume = shoot_volume; // Set old volume to the current volume for the next timestep
    }
}

std::vector<uint> makeTubeFromCones(uint radial_subdivisions, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr) {
    uint Nverts = vertices.size();

    if (radii.size() != Nverts || colors.size() != Nverts) {
        helios_runtime_error("ERROR (makeTubeFromCones): Length of vertex vectors is not consistent.");
    }

    std::vector<uint> objIDs;
    objIDs.reserve(Nverts - 1);

    for (uint v = 0; v < Nverts - 1; v++) {
        if ((vertices.at(v + 1) - vertices.at(v)).magnitude() < 1e-6f) {
            continue;
        }
        float r0 = std::max(radii.at(v), 1e-5f);
        float r1 = std::max(radii.at(v + 1), 1e-5f);
        objIDs.push_back(context_ptr->addConeObject(radial_subdivisions, vertices.at(v), vertices.at(v + 1), r0, r1, colors.at(v)));
    }

    return objIDs;
}

bool PlantArchitecture::detectGroundCollision(uint objID) {
    std::vector<uint> objIDs = {objID};
    return detectGroundCollision(objIDs);
}

bool PlantArchitecture::detectGroundCollision(const std::vector<uint> &objID) const {
    for (uint ID: objID) {
        if (context_ptr->doesObjectExist(ID)) {
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

void PlantArchitecture::optionalOutputObjectData(const std::string &object_data_label) {
    if (output_object_data.find(object_data_label) == output_object_data.end()) {
        std::cerr << "WARNING (PlantArchitecture::optionalOutputObjectData): Output object data of '" << object_data_label << "' is not a valid option." << std::endl;
        return;
    }
    output_object_data.at(object_data_label) = true;
}

void PlantArchitecture::optionalOutputObjectData(const std::vector<std::string> &object_data_labels) {
    for (auto &label: object_data_labels) {
        if (output_object_data.find(label) == output_object_data.end()) {
            std::cerr << "WARNING (PlantArchitecture::optionalOutputObjectData): Output object data of '" << label << "' is not a valid option." << std::endl;
            continue;
        }
        output_object_data.at(label) = true;
    }
}
