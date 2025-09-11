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
#include "CollisionDetection.h"

#include <unordered_set>
#include <utility>

using namespace helios;

static float clampOffset(int count_per_axis, float offset) {
    if (count_per_axis > 2) {
        float denom = 0.5f * float(count_per_axis) - 1.f;
        if (offset * denom > 1.f) {
            offset = 1.f / denom;
        }
    }
    return offset;
}

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

        if (frac >= f && (frac <= fplus || std::abs(frac - fplus) < 0.0001)) {
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


    // Initialize plant model registrations
    initializePlantModelRegistrations();

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

PlantArchitecture::~PlantArchitecture() {
    // Clean up owned CollisionDetection instance
    if (collision_detection_ptr != nullptr && owns_collision_detection) {
        delete collision_detection_ptr;
        collision_detection_ptr = nullptr;
        owns_collision_detection = false;
    }
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

PhytomerParameters::PhytomerParameters() : PhytomerParameters(nullptr) {
}

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

ShootParameters::ShootParameters() : ShootParameters(nullptr) {
}

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
                plantIDs.push_back(buildPlantInstanceFromLibrary(canopy_center_position + make_vec3(-0.5f * canopy_extent.x + float(i) * plant_spacing_xy.x, -0.5f * canopy_extent.y + float(j) * plant_spacing_xy.y, 0), 0));
            }
        }
    }

    if (age > 0) {
        advanceTime(plantIDs, age);
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
    if (this->shoot_types.find(shoot_type_label) != this->shoot_types.end()) {
        // shoot type already exists
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

helios::vec3 Phytomer::getInternodeAxisVector(const float stem_fraction) const {
    return getAxisVector(stem_fraction, getInternodeNodePositions());
}

helios::vec3 Phytomer::getPetioleAxisVector(const float stem_fraction, const uint petiole_index) const {
    if (petiole_index >= this->petiole_vertices.size()) {
        helios_runtime_error("ERROR (Phytomer::getPetioleAxisVector): Petiole index out of range.");
    }
    return getAxisVector(stem_fraction, this->petiole_vertices.at(petiole_index));
}

helios::vec3 Phytomer::getPeduncleAxisVector(const float stem_fraction, const uint petiole_index, const uint bud_index) const {
    if (petiole_index >= this->peduncle_vertices.size()) {
        helios_runtime_error("ERROR (Phytomer::getPeduncleAxisVector): Petiole index out of range.");
    }
    if (bud_index >= this->peduncle_vertices.at(petiole_index).size()) {
        helios_runtime_error("ERROR (Phytomer::getPeduncleAxisVector): Floral bud index out of range.");
    }
    return getAxisVector(stem_fraction, this->peduncle_vertices.at(petiole_index).at(bud_index));
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

float Phytomer::getInternodeRadius() const {
    return parent_shoot_ptr->shoot_internode_radii.at(shoot_index.x).front();
}

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

float Phytomer::getInternodeRadius(const float stem_fraction) const {
    return PlantArchitecture::interpolateTube(parent_shoot_ptr->shoot_internode_radii.at(shoot_index.x), stem_fraction);
}

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

void Phytomer::setVegetativeBudState(BudState state, VegetativeBud &vbud) {
    vbud.state = state;
}

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
        if (state == BUD_FLOWER_CLOSED || (fbud.state == BUD_ACTIVE && state == BUD_FLOWER_OPEN)) {
            // state went from active to closed flower or open flower
            float flower_cost = calculateFlowerConstructionCosts(fbud);
            plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(this->parent_shoot_ID)->carbohydrate_pool_molC -= flower_cost;
        } else if (state == BUD_FRUITING) {
            // adding a fruit
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

    if (state != BUD_DEAD) {
        // add new reproductive organs

        updateInflorescence(fbud);
        fbud.time_counter = 0;
        if (fbud.state == BUD_FRUITING) {
            setInflorescenceScaleFraction(fbud, 0.25);
        }
    }
}

helios::vec3 Phytomer::calculateCollisionAvoidanceDirection(const helios::vec3 &internode_base_origin, const helios::vec3 &internode_axis, bool &collision_detection_active) const {
    vec3 collision_optimal_direction;
    collision_detection_active = false;

    if (plantarchitecture_ptr->collision_detection_enabled && plantarchitecture_ptr->collision_detection_ptr != nullptr) {

        // BVH should already be built at timestep level - just use it
        if (!plantarchitecture_ptr->bvh_cached_for_current_growth) {
            if (plantarchitecture_ptr->printmessages) {
                std::cout << "WARNING: BVH not cached - this indicates rebuildBVHForTimestep() was not called" << std::endl;
            }
            return collision_optimal_direction; // Skip collision avoidance if BVH not ready
        }

        // Apply cone-aware culling based on actual collision detection geometry
        std::vector<uint> filtered_geometry;

        // Calculate spherical sector culling distance
        // The "cone" is actually a spherical sector with radius = look-ahead distance
        float look_ahead_distance = plantarchitecture_ptr->collision_cone_height;

        // Only obstacles within the look-ahead distance can be detected by collision rays
        // Add small buffer for obstacles at sector boundary
        float max_relevant_distance = look_ahead_distance * 1.1f; // 10% buffer


        // Always apply cone-aware culling for performance (no arbitrary thresholds)
        filtered_geometry = plantarchitecture_ptr->collision_detection_ptr->filterGeometryByDistance(internode_base_origin, max_relevant_distance, plantarchitecture_ptr->cached_target_geometry);


        // Update cached filtered geometry for this specific collision check
        plantarchitecture_ptr->cached_filtered_geometry = filtered_geometry;

        if (plantarchitecture_ptr->bvh_cached_for_current_growth && !plantarchitecture_ptr->cached_filtered_geometry.empty()) {
            // Set up cone parameters for optimal path finding
            vec3 apex = internode_base_origin;
            vec3 central_axis = internode_axis;
            central_axis.normalize();
            float height = plantarchitecture_ptr->collision_cone_height;
            float half_angle = plantarchitecture_ptr->collision_cone_half_angle_rad;
            int samples = plantarchitecture_ptr->collision_sample_count;

            // Find optimal cone path using gap detection (inertia blending handled later in PlantArchitecture)
            auto optimal_result = plantarchitecture_ptr->collision_detection_ptr->findOptimalConePath(apex, central_axis, half_angle, height, samples);

            // Store the optimal direction for later blending
            if (optimal_result.confidence > 0.0f) {
                collision_optimal_direction = optimal_result.direction;
                collision_optimal_direction.normalize();
                collision_detection_active = true;
            }
        }
    }
    return collision_optimal_direction;
}

helios::vec3 Phytomer::calculatePetioleCollisionAvoidanceDirection(const helios::vec3 &petiole_base_origin, const helios::vec3 &proposed_petiole_axis, bool &collision_detection_active) const {
    vec3 collision_optimal_direction;
    collision_detection_active = false;

    if (plantarchitecture_ptr->collision_detection_enabled && plantarchitecture_ptr->collision_detection_ptr != nullptr) {
        // Build restricted BVH with target geometry only
        std::vector<uint> target_geometry;
        if (!plantarchitecture_ptr->collision_target_UUIDs.empty()) {
            target_geometry = plantarchitecture_ptr->collision_target_UUIDs;
        } else if (!plantarchitecture_ptr->collision_target_object_IDs.empty()) {
            for (uint objID: plantarchitecture_ptr->collision_target_object_IDs) {
                std::vector<uint> obj_primitives = context_ptr->getObjectPrimitiveUUIDs(objID);
                target_geometry.insert(target_geometry.end(), obj_primitives.begin(), obj_primitives.end());
            }
        } else {
            // If no specific targets provided, use ALL geometry in Context for collision avoidance
            target_geometry = context_ptr->getAllUUIDs();
        }

        // Use cached BVH if available (same cache as internode collision avoidance)
        if (plantarchitecture_ptr->bvh_cached_for_current_growth && !plantarchitecture_ptr->cached_filtered_geometry.empty()) {
            // Set up cone parameters for optimal path finding using petiole-specific parameters
            vec3 apex = petiole_base_origin;
            vec3 central_axis = proposed_petiole_axis;
            central_axis.normalize();
            float height = plantarchitecture_ptr->collision_cone_height;
            float half_angle = plantarchitecture_ptr->collision_cone_half_angle_rad;
            int samples = plantarchitecture_ptr->collision_sample_count;

            // Find optimal cone path using gap detection for petiole direction
            auto optimal_result = plantarchitecture_ptr->collision_detection_ptr->findOptimalConePath(apex, central_axis, half_angle, height, samples);

            // Store the optimal direction for later blending
            if (optimal_result.confidence > 0.0f) {
                collision_optimal_direction = optimal_result.direction;
                collision_optimal_direction.normalize();
                collision_detection_active = true;
            }
        }
    }
    return collision_optimal_direction;
}

helios::vec3 Phytomer::calculateFruitCollisionAvoidanceDirection(const helios::vec3 &fruit_base_origin, const helios::vec3 &proposed_fruit_axis, bool &collision_detection_active) const {
    vec3 collision_optimal_direction;
    collision_detection_active = false;


    if (plantarchitecture_ptr->collision_detection_enabled && plantarchitecture_ptr->collision_detection_ptr != nullptr) {
        // Build restricted BVH with target geometry only
        std::vector<uint> target_geometry;
        if (!plantarchitecture_ptr->collision_target_UUIDs.empty()) {
            target_geometry = plantarchitecture_ptr->collision_target_UUIDs;
        } else if (!plantarchitecture_ptr->collision_target_object_IDs.empty()) {
            for (uint objID: plantarchitecture_ptr->collision_target_object_IDs) {
                std::vector<uint> obj_primitives = context_ptr->getObjectPrimitiveUUIDs(objID);
                target_geometry.insert(target_geometry.end(), obj_primitives.begin(), obj_primitives.end());
            }
        } else {
            // If no specific targets provided, use ALL geometry in Context for collision avoidance
            target_geometry = context_ptr->getAllUUIDs();
        }

        // Use cached BVH if available (same cache as internode collision avoidance)
        if (plantarchitecture_ptr->bvh_cached_for_current_growth && !plantarchitecture_ptr->cached_filtered_geometry.empty()) {
            // Set up cone parameters for optimal path finding using fruit-specific parameters
            vec3 apex = fruit_base_origin;
            vec3 central_axis = proposed_fruit_axis;
            central_axis.normalize();
            float height = plantarchitecture_ptr->collision_cone_height;
            float half_angle = plantarchitecture_ptr->collision_cone_half_angle_rad;
            int samples = plantarchitecture_ptr->collision_sample_count;

            // Find optimal cone path using gap detection for fruit direction
            auto optimal_result = plantarchitecture_ptr->collision_detection_ptr->findOptimalConePath(apex, central_axis, half_angle, height, samples);

            // Store the optimal direction for later blending
            if (optimal_result.confidence > 0.0f) {
                collision_optimal_direction = optimal_result.direction;
                collision_optimal_direction.normalize();
                collision_detection_active = true;
            }

            // Debug: track when collision detection doesn't find anything
            static int no_collision_count = 0;
            if (optimal_result.confidence <= 0.0f) {
                no_collision_count++;
            }
        } else {
            static int no_bvh_count = 0;
            no_bvh_count++;
        }
    }
    return collision_optimal_direction;
}

bool Phytomer::applySolidObstacleAvoidance(const helios::vec3 &current_position, helios::vec3 &internode_axis) const {
    if (!plantarchitecture_ptr->solid_obstacle_avoidance_enabled || plantarchitecture_ptr->solid_obstacle_UUIDs.empty()) {
        return false;
    }

    // Ignore solid obstacles for the first several nodes of the base stem to prevent U-turn growth
    // when plants start slightly below ground surface
    if (rank == 0 && (shoot_index.x < 3 || parent_shoot_ptr->calculateShootLength() < 0.05f)) {
        return false; // Skip solid obstacle avoidance for first 3 nodes OR if shoot length < 5cm
    }

    vec3 growth_direction = internode_axis;
    growth_direction.normalize();

    // Check for obstacles using cone-based detection
    float nearest_obstacle_distance;
    vec3 nearest_obstacle_direction;

    // Use smaller cone angle for hard obstacle detection (30 degrees vs 80 degrees for soft avoidance)
    float hard_detection_cone_angle = deg2rad(30.0f);
    float detection_distance = plantarchitecture_ptr->solid_obstacle_avoidance_distance;

    if (plantarchitecture_ptr->collision_detection_ptr != nullptr && plantarchitecture_ptr->collision_detection_ptr->findNearestSolidObstacleInCone(current_position, growth_direction, hard_detection_cone_angle, detection_distance,
                                                                                                                                                    plantarchitecture_ptr->solid_obstacle_UUIDs, nearest_obstacle_distance, nearest_obstacle_direction)) {

        // Define buffer distance as 5% of detection distance (cone length)
        float buffer_distance = detection_distance * 0.05f;

        // Normalize distance by detection distance for smooth calculations
        float normalized_distance = nearest_obstacle_distance / detection_distance;
        float buffer_threshold = buffer_distance / detection_distance; // Normalized buffer threshold

        vec3 avoidance_direction;
        float rotation_fraction;

        if (nearest_obstacle_distance <= buffer_distance) {
            // CRITICAL: Within buffer zone - use strong directional avoidance
            // Calculate direction that points directly away from the obstacle surface
            avoidance_direction = current_position - (current_position + nearest_obstacle_direction * nearest_obstacle_distance);
            if (avoidance_direction.magnitude() < 0.001f) {
                // Fallback if we can't determine clear avoidance direction
                avoidance_direction = cross(growth_direction, nearest_obstacle_direction);
                if (avoidance_direction.magnitude() < 0.001f) {
                    avoidance_direction = make_vec3(0, 0, 1); // Fallback to upward growth
                }
            }
            avoidance_direction.normalize();

            // Strong avoidance when in buffer zone
            rotation_fraction = 1.0f;

            // Blend growth direction away from obstacle to maintain buffer
            float buffer_blend_factor = 0.8f; // Strong influence to get out of buffer
            internode_axis = (1.0f - buffer_blend_factor) * growth_direction + buffer_blend_factor * avoidance_direction;
            internode_axis.normalize();

        } else {
            // NORMAL: Outside buffer zone - use smooth rotational avoidance

            // Calculate the angle between growth direction and obstacle direction
            float dot_with_obstacle = normalize(growth_direction) * normalize(nearest_obstacle_direction);
            float angle_deficit = asin_safe(fabs(dot_with_obstacle));

            // Calculate perpendicular direction to avoid obstacle
            vec3 rotation_axis = cross(growth_direction, -nearest_obstacle_direction);

            if (rotation_axis.magnitude() > 0.001f) {
                rotation_axis.normalize();
            } else {
                angle_deficit = 0.f;
            }

            if (rotation_axis.magnitude() > 0.001f) {

                // Use smooth, normalized distance-based approach
                // Use increasing function that reaches 1.0 at 20% of the surface distance
                float surface_threshold_fraction = 0.2f; // Function reaches max strength at 20% of detection distance

                if (normalized_distance <= surface_threshold_fraction) {
                    // Maximum avoidance strength (1.0) when very close to surface
                    rotation_fraction = 1.0f;
                } else {
                    // Smooth decay from 1.0 to minimum strength as distance increases
                    float remaining_distance = normalized_distance - surface_threshold_fraction;
                    float max_remaining_distance = 1.0f - surface_threshold_fraction;

                    // Exponential decay for smoother transitions
                    float distance_factor = remaining_distance / max_remaining_distance; // 0.0 to 1.0
                    float min_rotation_fraction = 0.05f; // Minimum background avoidance strength

                    // Exponential decay: strong avoidance close to threshold, gentle far away
                    rotation_fraction = min_rotation_fraction + (1.0f - min_rotation_fraction) * exp(-3.0f * distance_factor);
                }

                // Apply fraction of the total angle deficit
                float rotation_this_step = angle_deficit * rotation_fraction;

                // Apply the rotation
                internode_axis = rotatePointAboutLine(internode_axis, nullorigin, rotation_axis, rotation_this_step);
                internode_axis.normalize();
            }
        }

        return true; // Obstacle found and avoidance applied
    }

    return false; // No obstacle found
}

helios::vec3 Phytomer::calculateAttractionPointDirection(const helios::vec3 &internode_base_origin, const helios::vec3 &internode_axis, bool &attraction_active) const {
    vec3 attraction_direction;
    attraction_active = false;

    // First check if this plant has plant-specific attraction points enabled
    if (plantarchitecture_ptr->plant_instances.find(plantID) != plantarchitecture_ptr->plant_instances.end()) {
        const auto &plant = plantarchitecture_ptr->plant_instances.at(plantID);
        if (plant.attraction_points_enabled && !plant.attraction_points.empty()) {
            // Use plant-specific attraction points
            vec3 look_direction = internode_axis;
            look_direction.normalize();
            float half_angle_degrees = rad2deg(plant.attraction_cone_half_angle_rad);
            float look_ahead_distance = plant.attraction_cone_height;

            vec3 direction_to_closest;
            if (plantarchitecture_ptr->detectAttractionPointsInCone(plant.attraction_points, internode_base_origin, look_direction, look_ahead_distance, half_angle_degrees, direction_to_closest)) {
                attraction_direction = direction_to_closest;
                attraction_direction.normalize();
                attraction_active = true;
            }
            return attraction_direction;
        }
    }

    // Fall back to global attraction points for backward compatibility
    if (!plantarchitecture_ptr->attraction_points_enabled || plantarchitecture_ptr->attraction_points.empty()) {
        return attraction_direction;
    }

    // Use the native attraction points detection method from PlantArchitecture (no collision detection required)
    vec3 look_direction = internode_axis;
    look_direction.normalize();
    float half_angle_degrees = rad2deg(plantarchitecture_ptr->attraction_cone_half_angle_rad);
    float look_ahead_distance = plantarchitecture_ptr->attraction_cone_height;

    vec3 direction_to_closest;
    if (plantarchitecture_ptr->detectAttractionPointsInCone(plantarchitecture_ptr->attraction_points, internode_base_origin, look_direction, look_ahead_distance, half_angle_degrees, direction_to_closest)) {
        attraction_direction = direction_to_closest;
        attraction_direction.normalize();
        attraction_active = true;
    }

    return attraction_direction;
}

bool PlantArchitecture::detectAttractionPointsInCone(const helios::vec3 &vertex, const helios::vec3 &look_direction, float look_ahead_distance, float half_angle_degrees, helios::vec3 &direction_to_closest) const {

    // Validate input parameters
    if (attraction_points.empty()) {
        return false;
    }

    if (look_ahead_distance <= 0.0f) {
        if (printmessages) {
            std::cerr << "WARNING (PlantArchitecture::detectAttractionPointsInCone): Invalid look-ahead distance (<= 0)" << std::endl;
        }
        return false;
    }

    if (half_angle_degrees <= 0.0f || half_angle_degrees >= 180.0f) {
        if (printmessages) {
            std::cerr << "WARNING (PlantArchitecture::detectAttractionPointsInCone): Invalid half-angle (must be in range (0, 180) degrees)" << std::endl;
        }
        return false;
    }

    // Convert half-angle to radians
    float half_angle_rad = half_angle_degrees * M_PI / 180.0f;

    // Normalize look direction
    vec3 axis = look_direction;
    axis.normalize();

    // Variables to track the closest attraction point
    bool found_any = false;
    float min_angular_distance = std::numeric_limits<float>::max();
    vec3 closest_point;

    // Check each attraction point
    for (const vec3 &point: attraction_points) {
        // Calculate vector from vertex to attraction point
        vec3 to_point = point - vertex;
        float distance_to_point = to_point.magnitude();

        // Skip if point is at the vertex or beyond look-ahead distance
        if (distance_to_point < 1e-6f || distance_to_point > look_ahead_distance) {
            continue;
        }

        // Normalize the direction to the point
        vec3 direction_to_point = to_point;
        direction_to_point.normalize();

        // Calculate angle between look direction and direction to point
        float cos_angle = axis * direction_to_point;

        // Clamp to handle numerical precision issues
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));

        float angle = std::acos(cos_angle);

        // Check if point is within the perception cone
        if (angle <= half_angle_rad) {
            found_any = true;

            // Check if this is the closest to the centerline
            if (angle < min_angular_distance) {
                min_angular_distance = angle;
                closest_point = point;
            }
        }
    }

    // If we found any attraction points, calculate the direction to the closest one
    if (found_any) {
        direction_to_closest = closest_point - vertex;
        direction_to_closest.normalize();
        return true;
    }

    return false;
}

bool PlantArchitecture::detectAttractionPointsInCone(const std::vector<helios::vec3> &attraction_points_input, const helios::vec3 &vertex, const helios::vec3 &look_direction, float look_ahead_distance, float half_angle_degrees,
                                                     helios::vec3 &direction_to_closest) const {

    // Validate input parameters
    if (attraction_points_input.empty()) {
        return false;
    }

    if (look_ahead_distance <= 0.0f) {
        if (printmessages) {
            std::cerr << "WARNING (PlantArchitecture::detectAttractionPointsInCone): Invalid look-ahead distance (<= 0)" << std::endl;
        }
        return false;
    }

    if (half_angle_degrees <= 0.0f || half_angle_degrees >= 180.0f) {
        if (printmessages) {
            std::cerr << "WARNING (PlantArchitecture::detectAttractionPointsInCone): Invalid half-angle (must be in range (0, 180) degrees)" << std::endl;
        }
        return false;
    }

    // Convert half-angle to radians
    float half_angle_rad = half_angle_degrees * M_PI / 180.0f;

    // Normalize look direction
    vec3 axis = look_direction;
    axis.normalize();

    // Variables to track the closest attraction point
    bool found_any = false;
    float min_angular_distance = std::numeric_limits<float>::max();
    vec3 closest_point;

    // Check each attraction point
    for (const vec3 &point: attraction_points_input) {
        // Calculate vector from vertex to attraction point
        vec3 to_point = point - vertex;
        float distance_to_point = to_point.magnitude();

        // Skip if point is at the vertex or beyond look-ahead distance
        if (distance_to_point <= 1e-6 || distance_to_point > look_ahead_distance) {
            continue;
        }

        // Normalize the direction to the point
        vec3 direction_to_point = to_point;
        direction_to_point.normalize();

        // Calculate angle between look direction and direction to point
        float cos_angle = axis * direction_to_point;

        // Clamp to handle numerical precision issues
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));

        float angle = std::acos(cos_angle);

        // Check if point is within the perception cone
        if (angle <= half_angle_rad) {
            found_any = true;

            // Check if this is the closest to the centerline
            if (angle < min_angular_distance) {
                min_angular_distance = angle;
                closest_point = point;
            }
        }
    }

    // If we found any attraction points, calculate the direction to the closest one
    if (found_any) {
        direction_to_closest = closest_point - vertex;
        direction_to_closest.normalize();
        return true;
    }

    return false;
}

int Shoot::appendPhytomer(float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, const PhytomerParameters &phytomer_parameters) {
    auto shoot_tree_ptr = &plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree;

    // Determine the parent internode and petiole axes for rotation of the new phytomer
    vec3 parent_internode_axis;
    vec3 parent_petiole_axis;
    vec3 internode_base_position;
    if (phytomers.empty()) {
        // very first phytomer on shoot
        if (parent_shoot_ID == -1) {
            // very first shoot of the plant
            parent_internode_axis = make_vec3(0, 0, 1);
            parent_petiole_axis = make_vec3(0, -1, 0);
        } else {
            // first phytomer of a new shoot
            assert(parent_shoot_ID < shoot_tree_ptr->size() && parent_node_index < shoot_tree_ptr->at(parent_shoot_ID)->phytomers.size());
            parent_internode_axis = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->getInternodeAxisVector(1.f);
            parent_petiole_axis = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->getPetioleAxisVector(0.f, parent_petiole_index);
        }
        internode_base_position = base_position;
    } else {
        // additional phytomer being added to an existing shoot
        parent_internode_axis = phytomers.back()->getInternodeAxisVector(1.f);
        parent_petiole_axis = phytomers.back()->getPetioleAxisVector(0.f, 0);
        internode_base_position = shoot_internode_vertices.back().back();
    }

    std::shared_ptr<Phytomer> phytomer = std::make_shared<Phytomer>(phytomer_parameters, this, static_cast<uint>(phytomers.size()), parent_internode_axis, parent_petiole_axis, internode_base_position, this->base_rotation, internode_radius,
                                                                    internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, rank, plantarchitecture_ptr, context_ptr);
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
                    if (sampleVegetativeBudBreak_carb(phytomer->shoot_index.x)) {
                        // randomly sample bud
                        phytomer->setVegetativeBudState(BUD_ACTIVE, vbud);
                    } else {
                        phytomer->setVegetativeBudState(BUD_DEAD, vbud);
                    }
                } else {
                    if (sampleVegetativeBudBreak(phytomer->shoot_index.x)) {
                        // randomly sample bud
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
    if (parent_shoot_ID >= 0) {
        // only if not the base shoot

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
        if (parent_shoot_ID >= 0) {
            // shift petiole base outward by the parent internode radius
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
    this->phytomer_parameters = params;
    // note this needs to be an assignment operation not a copy in order to re-randomize all the parameters

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

    // initialize peduncle vertices storage (will be resized when floral buds are added)
    peduncle_vertices.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_pitch.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_curvature.resize(phytomer_parameters.petiole.petioles_per_internode);
    std::vector<float> dr_petiole(phytomer_parameters.petiole.petioles_per_internode);
    std::vector<float> dr_petiole_max(phytomer_parameters.petiole.petioles_per_internode);
    for (int p = 0; p < phytomer_parameters.petiole.petioles_per_internode; p++) {
        petiole_vertices.at(p).resize(Ndiv_petiole_length + 1);
        petiole_radii.at(p).resize(Ndiv_petiole_length + 1);

        petiole_length.at(p) = leaf_scale_factor_fraction * phytomer_parameters.petiole.length.val();
        if (petiole_length.at(p) <= 0.f) {
            petiole_length.at(p) = 1e-5f;
        }
        dr_petiole.at(p) = petiole_length.at(p) / float(phytomer_parameters.petiole.length_segments);
        dr_petiole_max.at(p) = phytomer_parameters.petiole.length.val() / float(phytomer_parameters.petiole.length_segments);

        petiole_radii.at(p).at(0) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val();
        if (petiole_radii.at(p).at(0) <= 0.f) {
            petiole_radii.at(p).at(0) = 1e-5f;
        }
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
    float leaflet_offset_val = clampOffset(leaves_per_petiole, phytomer_parameters.leaf.leaflet_offset.val());
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

        float roll_nudge = 0.f;
        //\todo Not clear if this is still needed. It causes problems when you want to plant base roll to be exactly 0.
        //        if( shoot_base_rotation.roll/180.f == floor(shoot_base_rotation.roll/180.f) ) {
        //            roll_nudge = 0.2;
        //        }
        if (shoot_base_rotation.roll != 0.f || roll_nudge != 0.f) {
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll + roll_nudge);
            // small additional rotation is to make sure the petiole is not exactly vertical
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

    // Store collision detection and attraction points parameters for later use (after all natural rotations)
    vec3 collision_optimal_direction;
    bool collision_detection_active = false;
    vec3 attraction_direction;
    bool attraction_active = false;
    bool obstacle_found = false;

    // Calculate collision avoidance direction if collision detection is enabled
    collision_optimal_direction = calculateCollisionAvoidanceDirection(internode_base_origin, internode_axis, collision_detection_active);

    // Calculate attraction point direction if attraction points are enabled
    attraction_direction = calculateAttractionPointDirection(internode_base_origin, internode_axis, attraction_active);

    // Solid obstacle avoidance is now handled inside the segment creation loop

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

        // Apply solid obstacle avoidance after natural rotations but before soft collision avoidance
        vec3 current_position = phytomer_internode_vertices.at(inode_segment - 1);
        obstacle_found = applySolidObstacleAvoidance(current_position, internode_axis);

        // Apply direction guidance after all natural rotations are complete
        // New approach: Blend hard obstacle avoidance with attraction to maintain surface attraction

        vec3 final_direction = internode_axis; // Start with current direction (includes hard obstacle avoidance if applied)

        if (attraction_active) {
            // Always apply attraction points if they're found
            float attraction_weight = plantarchitecture_ptr->attraction_weight;

            if (obstacle_found) {
                // When hard obstacles are present, reduce attraction influence to allow obstacle avoidance
                // but maintain some attraction to keep plant near surface
                attraction_weight *= plantarchitecture_ptr->attraction_obstacle_reduction_factor; // Reduce attraction when avoiding hard obstacles
            }

            // Blend current direction (which may include obstacle avoidance) with attraction direction
            final_direction = (1.0f - attraction_weight) * final_direction + attraction_weight * attraction_direction;
            final_direction.normalize();

            // Mark that attraction guidance was applied
            plantarchitecture_ptr->collision_avoidance_applied = true;

        } else if (collision_detection_active && !obstacle_found) {
            // No attraction points found and no hard obstacles - fall back to soft collision avoidance
            float inertia_weight = plantarchitecture_ptr->collision_inertia_weight;

            // Blend natural direction with optimal collision avoidance direction
            final_direction = inertia_weight * final_direction + (1.0f - inertia_weight) * collision_optimal_direction;
            final_direction.normalize();

            // Mark that collision avoidance was applied this timestep
            plantarchitecture_ptr->collision_avoidance_applied = true;
        }

        if (obstacle_found) {
            // Mark that hard obstacle avoidance was applied
            plantarchitecture_ptr->collision_avoidance_applied = true;
        }

        // Update the internode axis with the final blended direction
        internode_axis = final_direction;

        // vec3 displacement = dr_internode * internode_axis;
        // // Ensure minimum coordinate-wise displacement to avoid floating-point precision issues
        // if (fabs(displacement.x) < 1e-5f && fabs(displacement.y) < 1e-5f) {
        //     // If both x and y displacements are tiny, add small perturbation to avoid degenerate geometry
        //     if (fabs(internode_axis.z) > 0.9f) {
        //         // Nearly vertical - add horizontal perturbation
        //         displacement.x = (internode_axis.x >= 0) ? 1e-5f : -1e-5f;
        //     } else {
        //         // Not vertical - add z perturbation
        //         displacement.z = (internode_axis.z >= 0) ? 1e-5f : -1e-5f;
        //     }
        // }
        // phytomer_internode_vertices.at(inode_segment) = phytomer_internode_vertices.at(inode_segment - 1) + displacement;

        phytomer_internode_vertices.at(inode_segment) = phytomer_internode_vertices.at(inode_segment - 1) + dr_internode * internode_axis;

        phytomer_internode_radii.at(inode_segment) = internode_radius;
        internode_colors.at(inode_segment) = phytomer_parameters.internode.color;
    }

    if (shoot_index.x == 0) {
        // first phytomer on shoot
        parent_shoot_ptr->shoot_internode_vertices.push_back(phytomer_internode_vertices);
        parent_shoot_ptr->shoot_internode_radii.push_back(phytomer_internode_radii);
    } else {
        // if not the first phytomer on shoot, don't insert the first node because it's already defined on the previous phytomer
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

        if (!context_ptr->doesObjectExist(parent_shoot->internode_tube_objID)) {
            // first internode on shoot
            if (!phytomer_parameters.internode.image_texture.empty()) {
                parent_shoot->internode_tube_objID = context_ptr->addTubeObject(Ndiv_internode_radius, phytomer_internode_vertices, phytomer_internode_radii, phytomer_parameters.internode.image_texture.c_str(), uv_y);
            } else {
                parent_shoot->internode_tube_objID = context_ptr->addTubeObject(Ndiv_internode_radius, phytomer_internode_vertices, phytomer_internode_radii, internode_colors);
            }
            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(parent_shoot->internode_tube_objID), "object_label", "shoot");
        } else {
            // appending internode to shoot
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

    for (int petiole = 0; petiole < phytomer_parameters.petiole.petioles_per_internode; petiole++) {
        // looping over petioles

        vec3 petiole_axis = internode_axis;

        // petiole pitch rotation
        // Check if this is the last phytomer on a shoot that has reached max_nodes
        if (shoot_index.y + 1 == shoot_index.z) {
            // Last phytomer on shoot - set petiole pitch to 0
            petiole_pitch.at(petiole) = 0.0f;
        } else {
            // Normal phytomer - use standard pitch calculation
            petiole_pitch.at(petiole) = deg2rad(phytomer_parameters.petiole.pitch.val());
            phytomer_parameters.petiole.pitch.resample();
            if (fabs(petiole_pitch.at(petiole)) < deg2rad(5.f)) {
                petiole_pitch.at(petiole) = deg2rad(5.f);
            }
        }
        petiole_axis = rotatePointAboutLine(petiole_axis, nullorigin, petiole_rotation_axis, std::abs(petiole_pitch.at(petiole)));

        // petiole yaw rotation
        if (phytomer_index != 0 && internode_phyllotactic_angle != 0) {
            // not first phytomer along shoot
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

        // Apply collision avoidance for petiole direction (if enabled)
        vec3 collision_optimal_petiole_direction;
        bool petiole_collision_active = false;

        if (plantarchitecture_ptr->petiole_collision_detection_enabled) {
            collision_optimal_petiole_direction = calculatePetioleCollisionAvoidanceDirection(phytomer_internode_vertices.back(), // petiole base position
                                                                                              petiole_axis_actual, petiole_collision_active);
        }

        if (petiole_collision_active) {
            float inertia_weight = plantarchitecture_ptr->collision_inertia_weight;
            vec3 natural_petiole_direction = petiole_axis_actual;

            // Blend natural petiole direction with optimal direction
            // inertia = 1.0: use natural direction (no collision avoidance)
            // inertia = 0.0: use optimal direction (full collision avoidance)
            petiole_axis_actual = inertia_weight * natural_petiole_direction + (1.0f - inertia_weight) * collision_optimal_petiole_direction;
            petiole_axis_actual.normalize();

            // Adjust petiole curvature to bend toward optimal direction
            // Calculate desired bending direction perpendicular to natural petiole axis
            vec3 bending_direction = collision_optimal_petiole_direction - (collision_optimal_petiole_direction * natural_petiole_direction) * natural_petiole_direction;

            if (bending_direction.magnitude() > 1e-6f) {
                bending_direction.normalize();

                // Project bending direction onto petiole rotation plane to determine curvature adjustment
                // The rotation axis is perpendicular to both natural direction and bending direction
                vec3 curvature_axis = cross(natural_petiole_direction, bending_direction);

                if (curvature_axis.magnitude() > 1e-6f) {
                    curvature_axis.normalize();

                    // Calculate desired curvature angle based on angular deviation
                    float angular_deviation = acosf(std::max(-1.0f, std::min(1.0f, collision_optimal_petiole_direction * natural_petiole_direction)));

                    // Convert to degrees and scale by collision strength
                    float desired_curvature_deg = rad2deg(angular_deviation) * (1.0f - inertia_weight);

                    // Determine if curvature should be positive or negative based on rotation axis alignment
                    float curvature_sign = (curvature_axis * petiole_rotation_axis_actual > 0) ? 1.0f : -1.0f;

                    // Apply additional curvature for collision avoidance
                    petiole_curvature.at(petiole) += curvature_sign * desired_curvature_deg * 0.5f; // scale factor to prevent excessive bending
                }
            }
        }

        petiole_vertices.at(petiole).at(0) = phytomer_internode_vertices.back();

        for (int j = 1; j <= Ndiv_petiole_length; j++) {
            if (fabs(petiole_curvature.at(petiole)) > 0) {
                petiole_axis_actual = rotatePointAboutLine(petiole_axis_actual, nullorigin, petiole_rotation_axis_actual, -deg2rad(petiole_curvature.at(petiole) * dr_petiole_max.at(petiole)));
            }

            petiole_vertices.at(petiole).at(j) = petiole_vertices.at(petiole).at(j - 1) + dr_petiole.at(petiole) * petiole_axis_actual;

            petiole_radii.at(petiole).at(j) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val() * (1.f - phytomer_parameters.petiole.taper.val() / float(Ndiv_petiole_length) * float(j));
            petiole_colors.at(j) = phytomer_parameters.petiole.color;

            assert(!std::isnan(petiole_vertices.at(petiole).at(j).x) && std::isfinite(petiole_vertices.at(petiole).at(j).x));
            assert(!std::isnan(petiole_radii.at(petiole).at(j)) && std::isfinite(petiole_radii.at(petiole).at(j)));
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
            if (phytomer_parameters.leaf.prototype.unique_prototypes > 0) {
                // copy the existing prototype
                int prototype = context_ptr->randu(0, phytomer_parameters.leaf.prototype.unique_prototypes - 1);
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.find(phytomer_parameters.leaf.prototype.unique_prototype_identifier) != plantarchitecture_ptr->unique_leaf_prototype_objIDs.end());
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).size() > prototype);
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).size() > leaf);
                objID_leaf = context_ptr->copyObject(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).at(leaf));
            } else {
                // load a new prototype
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
                if (leaflet_offset_val == 0) {
                    float dphi = PI_F / (floor(0.5 * float(leaves_per_petiole - 1)) + 1);
                    compound_rotation = -float(PI_F) + dphi * (leaf + 0.5f);
                } else {
                    if (leaf == float(leaves_per_petiole - 1) / 2.f) {
                        // tip leaf
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
            if (leaves_per_petiole > 1 && leaflet_offset_val > 0) {
                if (ind_from_tip != 0) {
                    float offset = (fabs(ind_from_tip) - 0.5f) * leaflet_offset_val * phytomer_parameters.petiole.length.val();
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

    // Apply collision avoidance for peduncle direction (if enabled) - following petiole pattern
    vec3 collision_optimal_peduncle_direction;
    bool peduncle_collision_active = false;

    if (plantarchitecture_ptr->fruit_collision_detection_enabled) {
        collision_optimal_peduncle_direction = calculateFruitCollisionAvoidanceDirection(fbud.base_position, // peduncle base position
                                                                                         peduncle_axis, peduncle_collision_active);
    }

    if (peduncle_collision_active) {
        float inertia_weight = plantarchitecture_ptr->collision_inertia_weight;
        vec3 natural_peduncle_direction = peduncle_axis;

        // Blend natural peduncle direction with optimal direction
        // inertia = 1.0: use natural direction (no collision avoidance)
        // inertia = 0.0: use optimal direction (full collision avoidance)
        peduncle_axis = inertia_weight * natural_peduncle_direction + (1.0f - inertia_weight) * collision_optimal_peduncle_direction;
        peduncle_axis.normalize();
    }

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

    // Store peduncle vertices for later axis vector calculations
    // Use the parent_index to determine which petiole this floral bud belongs to
    uint petiole_idx = fbud.parent_index;

    // Ensure the peduncle_vertices storage has the right size for this floral bud
    if (petiole_idx < this->peduncle_vertices.size()) {
        if (this->peduncle_vertices.at(petiole_idx).size() <= fbud.bud_index) {
            this->peduncle_vertices.at(petiole_idx).resize(fbud.bud_index + 1);
        }
        this->peduncle_vertices.at(petiole_idx).at(fbud.bud_index) = peduncle_vertices;
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

    int flowers_per_peduncle = phytomer_parameters.inflorescence.flowers_per_peduncle.val();
    float flower_offset_val = clampOffset(flowers_per_peduncle, phytomer_parameters.inflorescence.flower_offset.val());
    for (int fruit = 0; fruit < flowers_per_peduncle; fruit++) {
        uint objID_fruit;
        helios::vec3 fruit_scale;

        if (fbud.state == BUD_FRUITING) {
            if (phytomer_parameters.inflorescence.unique_prototypes > 0) {
                // copy existing prototype
                int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
                objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_fruit_prototype_objIDs.at(phytomer_parameters.inflorescence.fruit_prototype_function).at(prototype));
            } else {
                // load new prototype
                objID_fruit = phytomer_parameters.inflorescence.fruit_prototype_function(context_ptr, 1);
            }
            fruit_scale = phytomer_parameters.inflorescence.fruit_prototype_scale.val() * make_vec3(1, 1, 1);
            phytomer_parameters.inflorescence.fruit_prototype_scale.resample();
        } else {
            bool flower_is_open;
            if (fbud.state == BUD_FLOWER_CLOSED) {
                flower_is_open = false;
                if (phytomer_parameters.inflorescence.unique_prototypes > 0) {
                    // copy existing prototype
                    int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
                    objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_closed_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype));
                } else {
                    // load new prototype
                    objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr, 1, flower_is_open);
                }
            } else {
                flower_is_open = true;
                if (phytomer_parameters.inflorescence.unique_prototypes > 0) {
                    // copy existing prototype
                    int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
                    objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_open_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype));
                } else {
                    // load new prototype
                    objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr, 1, flower_is_open);
                }
            }
            fruit_scale = phytomer_parameters.inflorescence.flower_prototype_scale.val() * make_vec3(1, 1, 1);
            phytomer_parameters.inflorescence.flower_prototype_scale.resample();
        }

        float ind_from_tip = fabs(fruit - float(flowers_per_peduncle - 1) / float(phytomer_parameters.petiole.petioles_per_internode));

        context_ptr->scaleObject(objID_fruit, fruit_scale);

        // if we have more than one flower/fruit, we need to adjust the base position of the fruit
        vec3 fruit_base = peduncle_vertices.back();
        float frac = 1;
        if (flowers_per_peduncle > 1 && flower_offset_val > 0) {
            if (ind_from_tip != 0) {
                float offset = (ind_from_tip - 0.5f) * flower_offset_val * phytomer_parameters.peduncle.length.val();
                if (phytomer_parameters.peduncle.length.val() > 0) {
                    frac = 1.f - offset / phytomer_parameters.peduncle.length.val();
                }
                fruit_base = PlantArchitecture::interpolateTube(peduncle_vertices, frac);
            }
        }

        // if we have more than one flower/fruit, we need to adjust the rotation about the peduncle
        float compound_rotation = 0;
        if (flowers_per_peduncle > 1) {
            if (flower_offset_val == 0) {
                // flowers/fruit are all at the tip, so just equally distribute them about the azimuth
                float dphi = PI_F / (floor(0.5 * float(flowers_per_peduncle - 1)) + 1);
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
        if (fbud.state == BUD_FRUITING) {
            // gravity effect for fruit
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
    if (leaves_per_petiole > 1 && leaf_index == float(leaves_per_petiole - 1) / 2.f) {
        // tip leaflet of compound leaf
        roll = 0;
        yaw = 0;
        compound_rotation = 0;
    } else if (leaves_per_petiole > 1 && leaf_index < float(leaves_per_petiole - 1) / 2.f) {
        // lateral leaflet of compound leaf
        yaw = -rotation.yaw;
        roll = -rotation.roll;
        compound_rotation = -0.5 * PI_F;
    } else {
        // not a compound leaf
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

    for (int s = s_start; s < parent_shoot_ptr->shoot_internode_vertices.at(p).size(); s++) {
        // looping over all segments within this phytomer internode

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

void Phytomer::setInternodeMaxRadius(float internode_radius_max_new) {
    this->internode_radius_max = internode_radius_max_new;
}

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
        for (uint objID: petiole_objIDs.at(petiole_index)) {
            // looping over cones/segments within petiole
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

        float leaflet_offset_val = clampOffset(int(leaf_objIDs.at(petiole_index).size()), phytomer_parameters.leaf.leaflet_offset.val());

        context_ptr->translateObject(leaf_objIDs.at(petiole_index).at(leaf), -1 * leaf_bases.at(petiole_index).at(leaf));
        context_ptr->scaleObject(leaf_objIDs.at(petiole_index).at(leaf), delta_scale * make_vec3(1, 1, 1));
        if (ind_from_tip == 0) {
            context_ptr->translateObject(leaf_objIDs.at(petiole_index).at(leaf), petiole_vertices.at(petiole_index).back());
            leaf_bases.at(petiole_index).at(leaf) = petiole_vertices.at(petiole_index).back();
        } else {
            float offset = (fabs(ind_from_tip) - 0.5f) * leaflet_offset_val * phytomer_parameters.petiole.length.val();
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
        if (this->shoot_index.x == 0) {
            tube_prune_index = 0;
        } else {
            tube_prune_index = this->shoot_index.x * tube_segments + 1; // note that first segment has an extra vertex
        }
        if (tube_prune_index < tube_nodes) {
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

bool Phytomer::hasLeaf() const {
    return (!leaf_bases.empty() && !leaf_bases.front().empty());
}

float Phytomer::calculateDownstreamLeafArea() const {
    return parent_shoot_ptr->sumShootLeafArea(shoot_index.x);
}

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
    for (int i = 0; i < current_node_number; i++) {
        // loop over phytomers to build up the shoot

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

    if (shoot_ptr->shoot_parameters.child_shoot_type_labels.empty()) {
        // if user doesn't specify child shoot types, generate the same type by default
        child_shoot_type_label = shoot_ptr->shoot_type_label;
    } else if (shoot_ptr->shoot_parameters.child_shoot_type_labels.size() == 1) {
        // if only one child shoot types was specified, use it
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
    } else if (probability_decay > 0) {
        // probability maximum at apex
        bud_break_probability = std::fmax(probability_min, 1.f - probability_decay * float(this->current_node_number - node_index - 1));
    } else if (probability_decay < 0) {
        // probability maximum at base
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

void PlantArchitecture::disableInternodeContextBuild() {
    build_context_geometry_internode = false;
}

void PlantArchitecture::disablePetioleContextBuild() {
    build_context_geometry_petiole = false;
}

void PlantArchitecture::disablePeduncleContextBuild() {
    build_context_geometry_peduncle = false;
}

void PlantArchitecture::enableGroundClipping(float ground_height) {
    ground_clipping_height = ground_height;
}

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
        if (phytomer_radius > radius) {
            // radius should only increase
            radius = radius + 0.5 * (phytomer_radius - radius);
        }
    }

    if (update_context_geometry && context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        context_ptr->setTubeRadii(shoot->internode_tube_objID, flatten(shoot->shoot_internode_radii));
    }
}

void PlantArchitecture::pruneGroundCollisions(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
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

void PlantArchitecture::removeShootVegetativeBuds(uint plantID, uint shootID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::removeShootVegetativeBuds): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::removeShootVegetativeBuds): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto &shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    for (auto &phytomer: shoot->phytomers) {
        phytomer->setVegetativeBudState(BUD_DEAD);
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
        } else {
            // both elevation & azimuth
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

std::vector<uint> PlantArchitecture::getAllPlantUUIDs(uint plantID) const {
    return context_ptr->getObjectPrimitiveUUIDs(getAllPlantObjectIDs(plantID));
}

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

std::vector<uint> PlantArchitecture::getPlantCollisionRelevantObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantCollisionRelevantObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> collision_relevant_objects;

    // Collect collision-relevant geometry for this plant based on current settings

    // Internodes - always include if enabled
    if (collision_include_internodes) {
        std::vector<uint> internodes = getPlantInternodeObjectIDs(plantID);
        collision_relevant_objects.insert(collision_relevant_objects.end(), internodes.begin(), internodes.end());
    }

    // Leaves - include if enabled
    if (collision_include_leaves) {
        std::vector<uint> leaves = getPlantLeafObjectIDs(plantID);
        collision_relevant_objects.insert(collision_relevant_objects.end(), leaves.begin(), leaves.end());
    }

    // Petioles - include if enabled (typically disabled for trees)
    if (collision_include_petioles) {
        std::vector<uint> petioles = getPlantPetioleObjectIDs(plantID);
        collision_relevant_objects.insert(collision_relevant_objects.end(), petioles.begin(), petioles.end());
    }

    // Flowers - include if enabled (typically disabled)
    if (collision_include_flowers) {
        std::vector<uint> flowers = getPlantFlowerObjectIDs(plantID);
        collision_relevant_objects.insert(collision_relevant_objects.end(), flowers.begin(), flowers.end());
    }

    // Fruit - include if enabled (typically disabled)
    if (collision_include_fruit) {
        std::vector<uint> fruit = getPlantFruitObjectIDs(plantID);
        collision_relevant_objects.insert(collision_relevant_objects.end(), fruit.begin(), fruit.end());
    }

    return collision_relevant_objects;
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

void PlantArchitecture::enableCarbohydrateModel() {
    carbon_model_enabled = true;
}

void PlantArchitecture::disableCarbohydrateModel() {
    carbon_model_enabled = false;
}

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

    if (plant_shoot_tree->empty()) {
        // no shoots to add
        return plantID_new;
    }
    if (plant_shoot_tree->front()->phytomers.empty()) {
        // no phytomers to add
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

            if (node == 0) {
                // first phytomer on shoot
                AxisRotation original_base_rotation = shoot->base_rotation;
                if (shoot->parent_shoot_ID == -1) {
                    // first shoot on plant
                    shootID_new = addBaseStemShoot(plantID_new, 1, original_base_rotation + base_rotation, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction, 0, shoot->shoot_type_label);
                } else {
                    // child shoot
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
    advanceTime(this->getAllPlantIDs(), time_step_days);
}

void PlantArchitecture::advanceTime(int time_step_years, float time_step_days) {
    advanceTime(this->getAllPlantIDs(), float(time_step_years) * 365.f + time_step_days);
}

void PlantArchitecture::advanceTime(uint plantID, float time_step_days) {
    std::vector<uint> plantIDs = {plantID};
    advanceTime(plantIDs, time_step_days);
}

void PlantArchitecture::advanceTime(const std::vector<uint> &plantIDs, float time_step_days) {
    for (uint plantID: plantIDs) {
        if (plant_instances.find(plantID) == plant_instances.end()) {
            helios_runtime_error("ERROR (PlantArchitecture::advanceTime): Plant with ID of " + std::to_string(plantID) + " does not exist.");
        }
    }

    // Clear BVH cache at start of plant growth operation
    clearBVHCache();

    // Rebuild BVH once at the start if collision detection is enabled
    if (collision_detection_enabled && collision_detection_ptr != nullptr) {
        rebuildBVHForTimestep();
    }

    // accounting for case of time_step_days>phyllochron_min
    float phyllochron_min = 9999;
    for (uint plantID: plantIDs) {
        PlantInstance &plant_instance = plant_instances.at(plantID);
        auto shoot_tree = &plant_instance.shoot_tree;
        if (shoot_tree->empty()) {
            continue;
        }
        float phyllochron_min_shoot = shoot_tree->front()->shoot_parameters.phyllochron_min.val();
        if (phyllochron_min_shoot < phyllochron_min) {
            phyllochron_min = phyllochron_min_shoot;
        }
        for (int i = 1; i < shoot_tree->size(); i++) {
            if (shoot_tree->at(i)->shoot_parameters.phyllochron_min.val() < phyllochron_min) {
                phyllochron_min_shoot = shoot_tree->at(i)->shoot_parameters.phyllochron_min.val();
                if (phyllochron_min_shoot < phyllochron_min) {
                    phyllochron_min = phyllochron_min_shoot;
                }
            }
        }
    }
    if (phyllochron_min == 9999) {
        std::cerr << "WARNING (PlantArchitecture::advanceTime): No shoots have been added ot the model. Returning.." << std::endl;
        return;
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

    // Initialize progress bar for timesteps
    helios::ProgressBar progress_bar(Nsteps, 50, Nsteps > 1 && printmessages, "Advancing time");

    for (int timestep = 0; timestep < Nsteps; timestep++) {

        // Rebuild BVH periodically - less frequent for per-tree BVH since trees are isolated
        bool should_rebuild_bvh = false;
        if (collision_detection_enabled && collision_detection_ptr != nullptr) {
            // For per-tree BVH, rebuild less frequently (every 25 timesteps) since spatial isolation reduces need
            // For unified BVH, keep original frequency (every 10 timesteps) for better accuracy
            if (collision_detection_ptr->isTreeBasedBVHEnabled()) {
                should_rebuild_bvh = (timestep % 25 == 0);
            } else {
                should_rebuild_bvh = (timestep % 10 == 0);
            }
        }

        if (should_rebuild_bvh) {
            rebuildBVHForTimestep();

            // Re-register plants with per-tree BVH to update primitive counts as plants grow
            if (collision_detection_ptr->isTreeBasedBVHEnabled()) {
                for (uint plantID: plantIDs) {
                    std::vector<uint> plant_primitives = getPlantCollisionRelevantObjectIDs(plantID);
                    if (!plant_primitives.empty()) {
                        collision_detection_ptr->registerTree(plantID, plant_primitives);
                    }
                }
            }
        }

        if (timestep == Nsteps - 1 && remainder_time != 0.f) {
            dt_max_days = remainder_time;
        }

        for (uint plantID: plantIDs) {
            PlantInstance &plant_instance = plant_instances.at(plantID);

            auto shoot_tree = &plant_instance.shoot_tree;

            if (shoot_tree->empty()) {
                continue;
            }

            if (plant_instance.current_age <= plant_instance.max_age && plant_instance.current_age + dt_max_days > plant_instance.max_age) {
            } else if (plant_instance.current_age >= plant_instance.max_age) {
                // update Context geometry
                shoot_tree->front()->updateShootNodes(true);
                return;
            }

            plant_instance.current_age += dt_max_days;
            plant_instance.time_since_dormancy += dt_max_days;

            if (plant_instance.time_since_dormancy > plant_instance.dd_to_dormancy_break + plant_instance.dd_to_dormancy) {
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
                            if (shoot->shoot_parameters.phytomer_parameters.inflorescence.flower_prototype_function != nullptr) {
                                // user defined a flower prototype function
                                // -- Flower initiation (closed flowers) -- //
                                if (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation >= 0.f) {
                                    // bud is active and flower initiation is enabled
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
                                if ((fbud.state == BUD_FLOWER_OPEN && plant_instance.dd_to_fruit_set >= 0.f) ||
                                    // flower opened and fruit set is enabled
                                    (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation < 0.f && plant_instance.dd_to_flower_opening < 0.f && plant_instance.dd_to_fruit_set >= 0.f) ||
                                    // jumped straight to fruit set with no flowering
                                    (fbud.state == BUD_FLOWER_CLOSED && plant_instance.dd_to_flower_opening < 0.f && plant_instance.dd_to_fruit_set >= 0.f)) {
                                    // jumped from closed flower to fruit set with no flower opening
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
                    appendPhytomerToShoot(plantID, shoot->ID, shoot_types.at(shoot->shoot_type_label).phytomer_parameters, internode_radius, internode_length_max, 0.01,
                                          0.01); //\todo These factors should be set to be consistent with the shoot
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


            // Update Context geometry based on scheduling configuration
            bool should_update_context = collision_detection_enabled && (geometry_update_counter >= geometry_update_frequency);

            // Force Context update if collision avoidance was applied and force_update_on_collision is enabled
            bool force_update = collision_avoidance_applied && force_update_on_collision;

            if (should_update_context || force_update) {
                shoot_tree->front()->updateShootNodes(true);
                // Note: geometry_update_counter reset moved outside plant loop
            } else {
                // Update plant structure but not Context geometry
                shoot_tree->front()->updateShootNodes(false);
            }

            // Reset collision avoidance flag for next timestep
            collision_avoidance_applied = false;

            // *** ground collision detection *** //
            if (ground_clipping_height != -99999) {
                pruneGroundCollisions(plantID);
            }

            // Assign current volume as old volume for your next timestep
            for (auto &shoot: *shoot_tree) {
                float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume();
                // Find current volume for each shoot in the plant
                shoot->old_shoot_volume = shoot_volume; // Set old volume to the current volume for the next timestep
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

        // Reset geometry counter if updates occurred this timestep
        if (geometry_update_counter >= geometry_update_frequency) {
            geometry_update_counter = 0;
        } else {
            geometry_update_counter++;
        }

        // Update progress bar
        progress_bar.update();
    }

    // Adjust fruit positions to avoid solid obstacle collisions
    adjustFruitForObstacleCollision();

    // Fallback collision detection: prune any objects that still intersect solid boundaries
    if (solid_obstacle_pruning_enabled) {
        pruneSolidBoundaryCollisions();
    }

    // When collision detection is disabled, update all plant geometry once at the end
    // This is more efficient than periodic updates and ensures correct visualization
    if (!collision_detection_enabled) {
        for (uint plantID: plantIDs) {
            if (plant_instances.find(plantID) != plant_instances.end()) {
                plant_instances.at(plantID).shoot_tree.front()->updateShootNodes(true);
            }
        }
    }

    // Ensure progress bar shows 100% completion
    progress_bar.finish();
}

void PlantArchitecture::adjustFruitForObstacleCollision() {
    if (!solid_obstacle_avoidance_enabled || solid_obstacle_UUIDs.empty() || !solid_obstacle_fruit_adjustment_enabled) {
        return; // No obstacles to check or fruit adjustment disabled
    }

    if (collision_detection_ptr == nullptr) {
        return; // No collision detection available
    }

    // Debug counter to limit output
    int debug_failures_shown = 0;
    const int max_debug_failures = 0; // Disable debugging for performance

    // Initialize progress bar for processing plants
    helios::ProgressBar progress_bar(plant_instances.size(), 50, plant_instances.size() > 1 && printmessages, "Adjusting fruit collisions");

    // Process each plant instance
    for (const auto &plant_instance: plant_instances) {
        uint plantID = plant_instance.first;

        // Get all fruit object IDs for this plant
        std::vector<uint> fruit_objIDs = getPlantFruitObjectIDs(plantID);

        if (fruit_objIDs.empty()) {
            continue; // No fruit to process
        }

        // Check each fruit for collision
        for (uint fruit_objID: fruit_objIDs) {
            // Get fruit primitives
            std::vector<uint> fruit_UUIDs = context_ptr->getObjectPrimitiveUUIDs(fruit_objID);

            if (fruit_UUIDs.empty()) {
                continue;
            }

            // Check if fruit collides with any solid obstacle
            std::vector<uint> collisions = collision_detection_ptr->findCollisions(fruit_UUIDs, {}, solid_obstacle_UUIDs, {}, false);

            if (!collisions.empty()) {
                // Fruit is colliding - need to rotate it up

                // Get fruit bounding box to estimate rotation needed
                vec3 bbox_min, bbox_max;
                context_ptr->getObjectBoundingBox(fruit_objID, bbox_min, bbox_max);

                // Find the fruit base position and peduncle info from the shoot tree
                vec3 fruit_base;
                vec3 peduncle_axis;
                const Phytomer *fruit_phytomer = nullptr;
                uint fruit_petiole_index = 0;
                uint fruit_bud_index = 0;
                bool found_base = false;

                // Search through shoot tree to find this fruit's base position
                for (const auto &shoot: plant_instance.second.shoot_tree) {
                    for (const auto &phytomer: shoot->phytomers) {
                        uint petiole_idx = 0;
                        for (const auto &petiole: phytomer->floral_buds) {
                            for (const auto &fbud: petiole) {
                                // Check if this floral bud contains our fruit
                                for (size_t idx = 0; idx < fbud.inflorescence_objIDs.size(); idx++) {
                                    if (fbud.inflorescence_objIDs[idx] == fruit_objID && idx < fbud.inflorescence_bases.size()) {
                                        // Found it! Use the correct index to get the base position
                                        fruit_base = fbud.inflorescence_bases[idx];
                                        fruit_phytomer = phytomer.get();
                                        fruit_petiole_index = petiole_idx;
                                        fruit_bud_index = fbud.bud_index;

                                        // Get actual peduncle axis using stored vertices
                                        try {
                                            peduncle_axis = phytomer->getPeduncleAxisVector(1.0f, petiole_idx, fbud.bud_index);
                                        } catch (const std::exception &e) {
                                            // Fallback if peduncle vertices not available
                                            peduncle_axis = make_vec3(0, 0, 1);
                                        }

                                        found_base = true;
                                        break;
                                    }
                                }
                                if (found_base)
                                    break;
                            }
                            if (found_base)
                                break;
                            petiole_idx++;
                        }
                        if (found_base)
                            break;
                    }
                    if (found_base)
                        break;
                }

                if (!found_base) {
                    continue; // Couldn't find fruit base position
                }

                // Calculate initial rotation estimate
                // Estimate fruit "radius" as distance from base to furthest point
                float fruit_radius = 0;
                fruit_radius = std::max(fruit_radius, (bbox_max - fruit_base).magnitude());
                fruit_radius = std::max(fruit_radius, (bbox_min - fruit_base).magnitude());
                fruit_radius = std::max(fruit_radius, (make_vec3(bbox_min.x, bbox_min.y, bbox_max.z) - fruit_base).magnitude());
                fruit_radius = std::max(fruit_radius, (make_vec3(bbox_min.x, bbox_max.y, bbox_min.z) - fruit_base).magnitude());
                fruit_radius = std::max(fruit_radius, (make_vec3(bbox_max.x, bbox_min.y, bbox_min.z) - fruit_base).magnitude());
                fruit_radius = std::max(fruit_radius, (make_vec3(bbox_min.x, bbox_max.y, bbox_max.z) - fruit_base).magnitude());
                fruit_radius = std::max(fruit_radius, (make_vec3(bbox_max.x, bbox_min.y, bbox_max.z) - fruit_base).magnitude());
                fruit_radius = std::max(fruit_radius, (make_vec3(bbox_max.x, bbox_max.y, bbox_min.z) - fruit_base).magnitude());

                // Calculate penetration depth more accurately
                // Use the lowest point of the fruit bounding box vs ground level (z=0)
                float penetration_depth = std::max(0.0f, -bbox_min.z);

                // Calculate initial rotation guess
                float initial_rotation = 0;
                if (fruit_radius > 0 && penetration_depth > 0) {
                    // Use arc sine to estimate rotation needed
                    float angle_estimate = std::asin(std::min(1.0f, penetration_depth / fruit_radius));
                    // Multiply by 1.5 to account for fruit shape complexity (less aggressive than before)
                    initial_rotation = std::min(deg2rad(35.0f), angle_estimate * 1.5f);
                } else {
                    // Default rotation for partially submerged cases
                    initial_rotation = deg2rad(10.0f);
                }

                // Ensure minimum rotation for any collision case
                initial_rotation = std::max(initial_rotation, deg2rad(8.0f)); // Slightly smaller minimum

                // Calculate the proper rotation axis based on peduncle orientation
                vec3 rotation_axis;

                // Ensure peduncle axis is normalized
                if (peduncle_axis.magnitude() < 1e-6f) {
                    // Fallback if peduncle axis is not available
                    peduncle_axis = make_vec3(0, 0, 1);
                } else {
                    peduncle_axis.normalize();
                }

                // Get vector from fruit base to fruit center
                vec3 bbox_center = 0.5f * (bbox_min + bbox_max);
                vec3 to_fruit_center = bbox_center - fruit_base;
                if (to_fruit_center.magnitude() > 1e-6f) {
                    to_fruit_center.normalize();
                } else {
                    // If fruit center is at base, use peduncle direction
                    to_fruit_center = peduncle_axis;
                }

                // Rotation axis is perpendicular to both peduncle axis and to_fruit_center
                // This gives us the pitch rotation axis used for the original fruit positioning
                rotation_axis = cross(peduncle_axis, to_fruit_center);
                if (rotation_axis.magnitude() < 1e-6f) {
                    // Peduncle and fruit are aligned, use perpendicular to peduncle
                    if (std::abs(peduncle_axis.z) < 0.9f) {
                        rotation_axis = cross(peduncle_axis, make_vec3(0, 0, 1));
                    } else {
                        rotation_axis = cross(peduncle_axis, make_vec3(1, 0, 0));
                    }
                }
                rotation_axis.normalize();

                // Iteratively rotate fruit until no collision
                float rotation_step = initial_rotation;
                float total_rotation = 0;
                const float max_rotation = deg2rad(120.0f); // Allow more rotation
                const int max_iterations = 25; // More iterations

                // Debug info for this fruit (only show first few)
                bool debug_this_fruit = (debug_failures_shown < max_debug_failures);
                if (debug_this_fruit && printmessages) {
                    std::cout << "\n=== DEBUG: Fruit " << fruit_objID << " collision adjustment ===" << std::endl;
                    std::cout << "Fruit base: " << fruit_base << std::endl;
                    std::cout << "Fruit bbox: " << bbox_min << " to " << bbox_max << std::endl;
                    std::cout << "Fruit radius: " << fruit_radius << std::endl;
                    std::cout << "Penetration depth: " << penetration_depth << std::endl;
                    std::cout << "Peduncle axis: " << peduncle_axis << std::endl;
                    std::cout << "Rotation axis: " << rotation_axis << std::endl;
                    std::cout << "Initial rotation: " << rad2deg(initial_rotation) << " degrees" << std::endl;
                    std::cout << "Initial collisions: " << collisions.size() << std::endl;
                }

                for (int iter = 0; iter < max_iterations && total_rotation < max_rotation; iter++) {
                    // Apply rotation about fruit base
                    // Negative rotation to lift fruit up (opposite of gravity)
                    context_ptr->rotateObject(fruit_objID, -rotation_step, fruit_base, rotation_axis);
                    total_rotation += rotation_step;

                    // Check if still colliding
                    fruit_UUIDs = context_ptr->getObjectPrimitiveUUIDs(fruit_objID);
                    collisions = collision_detection_ptr->findCollisions(fruit_UUIDs, {}, solid_obstacle_UUIDs, {}, false);

                    if (debug_this_fruit && printmessages) {
                        std::cout << "Iter " << iter << ": rotated " << rad2deg(rotation_step) << " deg (total " << rad2deg(total_rotation) << "), collisions: " << collisions.size() << std::endl;
                    }

                    if (collisions.empty()) {
                        // No longer colliding - now try to fine-tune by rotating back down slightly
                        // to get as close to the ground as possible
                        float fine_tune_step = deg2rad(3.0f); // Slightly larger steps for efficiency
                        float fine_tune_attempts = 5;
                        float original_total = total_rotation;

                        if (debug_this_fruit && printmessages) {
                            std::cout << "Fine-tuning: trying to rotate back down from " << rad2deg(total_rotation) << " degrees" << std::endl;
                        }

                        for (int fine_iter = 0; fine_iter < fine_tune_attempts; fine_iter++) {
                            // Try rotating back towards ground (positive rotation)
                            context_ptr->rotateObject(fruit_objID, fine_tune_step, fruit_base, rotation_axis);

                            // Check if still collision-free
                            fruit_UUIDs = context_ptr->getObjectPrimitiveUUIDs(fruit_objID);
                            std::vector<uint> test_collisions = collision_detection_ptr->findCollisions(fruit_UUIDs, {}, solid_obstacle_UUIDs, {}, false);

                            if (!test_collisions.empty()) {
                                // Collision detected - rotate back up and stop fine-tuning
                                context_ptr->rotateObject(fruit_objID, -fine_tune_step, fruit_base, rotation_axis);
                                break;
                            } else {
                                // Still collision-free, reduce total rotation count
                                total_rotation -= fine_tune_step;
                            }
                        }

                        break;
                    }

                    // Adaptive step size - reduce for fine tuning, but not too aggressively
                    if (iter > 8) {
                        rotation_step *= 0.7f; // Less aggressive reduction
                    }
                }

                if (!collisions.empty()) {
                    if (debug_this_fruit && printmessages) {
                        std::cout << "FAILED: Fruit " << fruit_objID << " still colliding after " << rad2deg(total_rotation) << " degrees rotation (" << max_iterations << " iterations)" << std::endl;

                        // Get final bounding box to see where it ended up
                        vec3 final_bbox_min, final_bbox_max;
                        context_ptr->getObjectBoundingBox(fruit_objID, final_bbox_min, final_bbox_max);
                        std::cout << "Final bbox: " << final_bbox_min << " to " << final_bbox_max << std::endl;
                        std::cout << "Lowest point: " << final_bbox_min.z << std::endl;

                        debug_failures_shown++;
                    }
                }
            }
        }

        // Update progress bar
        progress_bar.update();
    }

    // Ensure progress bar shows 100% completion
    progress_bar.finish();
}

void PlantArchitecture::pruneSolidBoundaryCollisions() {
    if (!solid_obstacle_avoidance_enabled || solid_obstacle_UUIDs.empty()) {
        return; // No solid boundaries defined
    }

    if (collision_detection_ptr == nullptr) {
        return; // No collision detection available
    }

    if (printmessages) {
        std::cout << "Performing solid boundary collision detection..." << std::endl;
    }

    // The BVH should already be current from advanceTime() - we're called at the very end
    // Collect all plant primitives and do one batch collision detection call for efficiency
    std::vector<uint> all_plant_primitives;

    all_plant_primitives = getAllUUIDs();

    std::vector<uint> intersecting_primitives = collision_detection_ptr->findCollisions(solid_obstacle_UUIDs, {}, all_plant_primitives, {}, false);

    std::vector<uint> intersecting_objIDs = context_ptr->getUniquePrimitiveParentObjectIDs(intersecting_primitives);


    if (intersecting_primitives.empty()) {
        if (printmessages) {
            std::cout << "No collisions detected - this is unexpected given visible fruit penetration" << std::endl;
        }
        return; // No collisions detected
    }

    if (printmessages) {
        std::cout << "Intersecting primitives found: " << intersecting_primitives.size() << std::endl;
    }

    // Create lookup set for O(1) collision checking
    std::unordered_set<uint> collision_set(intersecting_objIDs.begin(), intersecting_objIDs.end());

    // Traverse plant topology and prune intersected organs and all downstream organs
    for (auto &[plantID, plant]: plant_instances) {
        for (uint shootID = 0; shootID < plant.shoot_tree.size(); shootID++) {
            auto &shoot = plant.shoot_tree.at(shootID);
            bool shoot_was_deleted = false;

            // Check if entire shoot's internode tube is colliding
            if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                if (collision_set.count(shoot->internode_tube_objID)) {
                    // Protect the entire main stem (rank 0 shoots)
                    if (shoot->rank != 0) {
                        // Delete the entire branch shoot
                        pruneBranch(plantID, shootID, 0); // Prune from the beginning of the shoot
                        shoot_was_deleted = true;
                    }
                }
            }

            // If the shoot was deleted due to internode collision, skip checking individual organs
            if (shoot_was_deleted) {
                continue;
            }

            for (uint node = 0; node < shoot->current_node_number; node++) {
                auto &phytomer = shoot->phytomers.at(node);

                // Check leaves for collision
                for (uint petiole = 0; petiole < phytomer->leaf_objIDs.size(); petiole++) {
                    for (uint leaflet = 0; leaflet < phytomer->leaf_objIDs.at(petiole).size(); leaflet++) {
                        uint leaf_objID = phytomer->leaf_objIDs.at(petiole).at(leaflet);
                        if (collision_set.count(leaf_objID)) {
                            phytomer->removeLeaf();
                            break; // removeLeaf() removes all leaflets on this petiole
                        }
                    }
                }

                // Check petiole objects for collision
                for (uint petiole = 0; petiole < phytomer->petiole_objIDs.size(); petiole++) {
                    for (uint segment = 0; segment < phytomer->petiole_objIDs.at(petiole).size(); segment++) {
                        uint petiole_objID = phytomer->petiole_objIDs.at(petiole).at(segment);
                        if (collision_set.count(petiole_objID)) {
                            phytomer->removeLeaf();
                            break; // removeLeaf() removes petiole and all leaflets
                        }
                    }
                }

                // Check inflorescence for collision
                for (auto &petiole: phytomer->floral_buds) {
                    for (auto &fbud: petiole) {
                        // Check inflorescence objects
                        for (int p = fbud.inflorescence_objIDs.size() - 1; p >= 0; p--) {
                            uint objID = fbud.inflorescence_objIDs.at(p);
                            if (collision_set.count(objID)) {
                                context_ptr->deleteObject(objID);
                                fbud.inflorescence_objIDs.erase(fbud.inflorescence_objIDs.begin() + p);
                                fbud.inflorescence_bases.erase(fbud.inflorescence_bases.begin() + p);
                            }
                        }
                        // Check peduncle objects
                        for (int p = fbud.peduncle_objIDs.size() - 1; p >= 0; p--) {
                            uint objID = fbud.peduncle_objIDs.at(p);
                            if (collision_set.count(objID)) {
                                // Delete all peduncle and inflorescence objects for this floral bud
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

            if (shoot_was_deleted) {
                break; // This shoot was pruned, no need to check more nodes
            }
        }
    }

    if (printmessages) {
        std::cout << "Solid boundary collision pruning completed" << std::endl;
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

void PlantArchitecture::enableSoftCollisionAvoidance(const std::vector<uint> &target_object_UUIDs, const std::vector<uint> &target_object_IDs, bool enable_petiole_collision, bool enable_fruit_collision) {
    // Clean up any existing collision detection instance
    if (collision_detection_ptr != nullptr && owns_collision_detection) {
        delete collision_detection_ptr;
        collision_detection_ptr = nullptr;
        owns_collision_detection = false;
    }

    // Create new CollisionDetection instance
    try {
        collision_detection_ptr = new CollisionDetection(context_ptr);
        collision_detection_ptr->enableMessages(); // Enable debug output for debugging
        owns_collision_detection = true;
        collision_detection_enabled = true;
        collision_target_UUIDs = target_object_UUIDs;
        collision_target_object_IDs = target_object_IDs;

        // Set organ-specific collision detection flags
        petiole_collision_detection_enabled = enable_petiole_collision;
        fruit_collision_detection_enabled = enable_fruit_collision;

        // Disable automatic BVH rebuilds - PlantArchitecture will control rebuilds manually
        collision_detection_ptr->disableAutomaticBVHRebuilds();

        // Enable per-tree BVH for linear scaling with multiple trees
        collision_detection_ptr->enableTreeBasedBVH(collision_cone_height); // Use collision cone height as isolation distance

        // Set static obstacles (non-plant geometry that affects all trees)
        std::vector<uint> static_obstacles;
        static_obstacles.insert(static_obstacles.end(), target_object_UUIDs.begin(), target_object_UUIDs.end());
        static_obstacles.insert(static_obstacles.end(), target_object_IDs.begin(), target_object_IDs.end());

        // Build initial BVH cache to prevent warnings during early collision detection calls
        rebuildBVHForTimestep();

        // Also include solid obstacle avoidance primitives if enabled
        if (solid_obstacle_avoidance_enabled) {
            static_obstacles.insert(static_obstacles.end(), solid_obstacle_UUIDs.begin(), solid_obstacle_UUIDs.end());
        }

        collision_detection_ptr->setStaticObstacles(static_obstacles);

        // Register existing plants as separate trees for per-tree BVH
        // This allows each plant to have its own collision BVH for linear scaling
        std::vector<uint> plant_ids = getAllPlantIDs();
        for (uint plant_id: plant_ids) {
            std::vector<uint> plant_primitives = getPlantCollisionRelevantObjectIDs(plant_id);
            if (!plant_primitives.empty()) {
                collision_detection_ptr->registerTree(plant_id, plant_primitives);
            }
        }

        setGeometryUpdateScheduling(3, true); // Update every 3 timesteps, force on collision

    } catch (const std::exception &e) {
        helios_runtime_error("ERROR (PlantArchitecture::enableSoftCollisionAvoidance): Failed to create CollisionDetection instance: " + std::string(e.what()));
    }
}

void PlantArchitecture::disableCollisionDetection() {
    collision_detection_enabled = false;

    // Clean up owned CollisionDetection instance
    if (collision_detection_ptr != nullptr && owns_collision_detection) {
        delete collision_detection_ptr;
        owns_collision_detection = false;
    }

    collision_detection_ptr = nullptr;
    collision_target_UUIDs.clear();
    collision_target_object_IDs.clear();

    if (printmessages) {
        std::cout << "Collision detection disabled for plant growth and internal instance cleaned up" << std::endl;
    }
}

void PlantArchitecture::setSoftCollisionAvoidanceParameters(float view_half_angle_deg, float look_ahead_distance, int sample_count, float inertia_weight) {
    if (view_half_angle_deg <= 0.0f || view_half_angle_deg > 180.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setSoftCollisionAvoidanceParameters): cone_half_angle_deg must be between 0 and 180 degrees.");
    }
    if (look_ahead_distance <= 0.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setSoftCollisionAvoidanceParameters): sample_count must be positive.");
    }
    if (inertia_weight < 0.0f || inertia_weight > 1.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setSoftCollisionAvoidanceParameters): inertia_weight must be between 0.0 and 1.0.");
    }

    collision_cone_half_angle_rad = deg2rad(view_half_angle_deg);
    collision_cone_height = look_ahead_distance;
    collision_sample_count = sample_count;
    collision_inertia_weight = inertia_weight;
}

void PlantArchitecture::setStaticObstacles(const std::vector<uint> &target_UUIDs) {
    if (collision_detection_ptr == nullptr) {
        helios_runtime_error("ERROR (PlantArchitecture::setStaticObstacles): Collision detection must be enabled before setting static obstacles.");
    }

    collision_detection_ptr->setStaticGeometry(target_UUIDs);

    if (printmessages) {
        std::cout << "Marked " << target_UUIDs.size() << " primitives as static obstacles for collision detection" << std::endl;
    }
}

CollisionDetection *PlantArchitecture::getCollisionDetection() const {
    return collision_detection_ptr;
}

void PlantArchitecture::setCollisionRelevantOrgans(bool include_internodes, bool include_leaves, bool include_petioles, bool include_flowers, bool include_fruit) {
    collision_include_internodes = include_internodes;
    collision_include_leaves = include_leaves;
    collision_include_petioles = include_petioles;
    collision_include_flowers = include_flowers;
    collision_include_fruit = include_fruit;

    // Clear BVH cache to force rebuild with new organ filtering
    clearBVHCache();

    if (printmessages) {
        std::cout << "Set collision-relevant organs: internodes=" << (include_internodes ? "yes" : "no") << ", leaves=" << (include_leaves ? "yes" : "no") << ", petioles=" << (include_petioles ? "yes" : "no")
                  << ", flowers=" << (include_flowers ? "yes" : "no") << ", fruit=" << (include_fruit ? "yes" : "no") << std::endl;
    }
}


void PlantArchitecture::enableSolidObstacleAvoidance(const std::vector<uint> &obstacle_UUIDs, float avoidance_distance, bool enable_fruit_adjustment, bool enable_obstacle_pruning) {
    solid_obstacle_avoidance_enabled = true;
    solid_obstacle_UUIDs = obstacle_UUIDs;
    solid_obstacle_avoidance_distance = avoidance_distance;
    solid_obstacle_fruit_adjustment_enabled = enable_fruit_adjustment;
    solid_obstacle_pruning_enabled = enable_obstacle_pruning;

    // Create CollisionDetection instance if needed for solid obstacle avoidance
    if (collision_detection_ptr == nullptr) {
        try {
            collision_detection_ptr = new CollisionDetection(context_ptr);
            collision_detection_ptr->enableMessages(); // Enable debug output for debugging
            owns_collision_detection = true;
            collision_detection_enabled = true;

            // Disable automatic BVH rebuilds - PlantArchitecture will control rebuilds manually
            collision_detection_ptr->disableAutomaticBVHRebuilds();
            // Enable per-tree BVH for linear scaling with multiple trees
            collision_detection_ptr->enableTreeBasedBVH(collision_cone_height); // Use collision cone height as isolation distance

            // Build initial BVH cache to prevent warnings during early collision detection calls
            rebuildBVHForTimestep();
        } catch (std::exception &e) {
            helios_runtime_error("ERROR (PlantArchitecture::enableSolidObstacleAvoidance): Failed to create CollisionDetection instance: " + std::string(e.what()));
        }
    }

    // Update CollisionDetection static obstacles if per-tree BVH is enabled
    if (collision_detection_enabled && collision_detection_ptr != nullptr && collision_detection_ptr->isTreeBasedBVHEnabled()) {
        std::vector<uint> static_obstacles;
        static_obstacles.insert(static_obstacles.end(), collision_target_UUIDs.begin(), collision_target_UUIDs.end());
        static_obstacles.insert(static_obstacles.end(), collision_target_object_IDs.begin(), collision_target_object_IDs.end());
        static_obstacles.insert(static_obstacles.end(), solid_obstacle_UUIDs.begin(), solid_obstacle_UUIDs.end());

        collision_detection_ptr->setStaticObstacles(static_obstacles);
    }
}

void PlantArchitecture::clearBVHCache() const {
    bvh_cached_for_current_growth = false;
    cached_target_geometry.clear();
    cached_filtered_geometry.clear();
}


void PlantArchitecture::rebuildBVHForTimestep() {
    if (!collision_detection_enabled || collision_detection_ptr == nullptr) {
        return;
    }


    // Determine target geometry for BVH
    std::vector<uint> target_geometry;

    // Always include solid obstacles if enabled
    if (solid_obstacle_avoidance_enabled && !solid_obstacle_UUIDs.empty()) {
        target_geometry.insert(target_geometry.end(), solid_obstacle_UUIDs.begin(), solid_obstacle_UUIDs.end());
    }

    if (!collision_target_UUIDs.empty()) {
        // Validate that all target UUIDs still exist
        std::vector<uint> valid_targets;
        for (uint uuid: collision_target_UUIDs) {
            if (context_ptr->doesPrimitiveExist(uuid)) {
                valid_targets.push_back(uuid);
            }
        }
        // Add valid collision targets to existing target_geometry (which may include solid obstacles)
        target_geometry.insert(target_geometry.end(), valid_targets.begin(), valid_targets.end());
    } else if (!collision_target_object_IDs.empty()) {
        // Add object primitives to existing target_geometry (which may include solid obstacles)
        for (uint objID: collision_target_object_IDs) {
            if (context_ptr->doesObjectExist(objID)) {
                std::vector<uint> obj_primitives = context_ptr->getObjectPrimitiveUUIDs(objID);
                target_geometry.insert(target_geometry.end(), obj_primitives.begin(), obj_primitives.end());
            }
        }
    } else {
        // Use filtered plant geometry based on organ settings + external obstacles
        // Preserve solid obstacles that were already added
        std::vector<uint> preserved_solid_obstacles = target_geometry;
        target_geometry.clear();

        // Add collision-relevant plant organs based on filtering settings (with safety checks)
        try {
            if (collision_include_internodes) {
                std::vector<uint> internode_uuids = getAllInternodeUUIDs();
                target_geometry.insert(target_geometry.end(), internode_uuids.begin(), internode_uuids.end());
            }
            if (collision_include_leaves) {
                std::vector<uint> leaf_uuids = getAllLeafUUIDs();
                target_geometry.insert(target_geometry.end(), leaf_uuids.begin(), leaf_uuids.end());
            }
            if (collision_include_petioles) {
                std::vector<uint> petiole_uuids = getAllPetioleUUIDs();
                target_geometry.insert(target_geometry.end(), petiole_uuids.begin(), petiole_uuids.end());
            }
            if (collision_include_flowers) {
                std::vector<uint> flower_uuids = getAllFlowerUUIDs();
                target_geometry.insert(target_geometry.end(), flower_uuids.begin(), flower_uuids.end());
            }
            if (collision_include_fruit) {
                std::vector<uint> fruit_uuids = getAllFruitUUIDs();
                target_geometry.insert(target_geometry.end(), fruit_uuids.begin(), fruit_uuids.end());
            }
        } catch (const std::exception &e) {
            if (printmessages) {
                std::cout << "Warning: Exception in organ filtering, falling back to all geometry: " << e.what() << std::endl;
            }
            target_geometry = context_ptr->getAllUUIDs();
        }

        // Re-add the preserved solid obstacles
        target_geometry.insert(target_geometry.end(), preserved_solid_obstacles.begin(), preserved_solid_obstacles.end());

        // Add any external obstacles from Context (non-plant geometry)
        std::vector<uint> all_context_geometry = context_ptr->getAllUUIDs();
        std::set<uint> all_plant_geometry_set;
        try {
            std::vector<uint> all_plant = getAllUUIDs();
            all_plant_geometry_set.insert(all_plant.begin(), all_plant.end());
        } catch (const std::exception &e) {
            if (printmessages) {
                std::cout << "Warning: Could not get plant geometry for external obstacle filtering: " << e.what() << std::endl;
            }
        }

        for (uint uuid: all_context_geometry) {
            if (all_plant_geometry_set.find(uuid) == all_plant_geometry_set.end()) {
                target_geometry.push_back(uuid); // Add external obstacles
            }
        }
    }

    if (!target_geometry.empty()) {
        // Separate static obstacles from plant geometry for hierarchical BVH
        std::vector<uint> plant_geometry;
        try {
            plant_geometry = getAllUUIDs();
        } catch (const std::exception &e) {
            if (printmessages) {
                std::cout << "Warning: Could not get plant geometry for hierarchical BVH: " << e.what() << std::endl;
            }
            plant_geometry.clear();
        }
        std::set<uint> plant_set(plant_geometry.begin(), plant_geometry.end());

        std::vector<uint> static_obstacles;
        for (uint uuid: target_geometry) {
            if (plant_set.find(uuid) == plant_set.end()) {
                static_obstacles.push_back(uuid); // Not plant geometry = static obstacle
            }
        }

        collision_detection_ptr->setStaticGeometry(static_obstacles);

        // Build BVH once per timestep
        collision_detection_ptr->updateBVH(target_geometry, true); // Force rebuild


        // Cache the geometry for this growth cycle
        cached_target_geometry = target_geometry;
        cached_filtered_geometry = target_geometry; // No filtering at timestep level
        bvh_cached_for_current_growth = true;
    }
}

void PlantArchitecture::setGeometryUpdateScheduling(int update_frequency, bool force_update_on_collision) {
    if (update_frequency < 1) {
        helios_runtime_error("ERROR (PlantArchitecture::setGeometryUpdateScheduling): update_frequency must be at least 1.");
    }

    geometry_update_frequency = update_frequency;
    geometry_update_counter = 0; // Reset counter
}

// ----- Attraction Points Methods ----- //

void PlantArchitecture::enableAttractionPoints(const std::vector<helios::vec3> &attraction_points_input, float view_half_angle_deg, float look_ahead_distance, float attraction_weight_input) {
    if (view_half_angle_deg <= 0.0f || view_half_angle_deg > 180.f) {
        helios_runtime_error("ERROR (PlantArchitecture::enableAttractionPoints): view_half_angle_deg must be between 0 and 180 degrees.");
    }
    if (look_ahead_distance <= 0.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::enableAttractionPoints): look_ahead_distance must be positive.");
    }
    if (attraction_weight_input < 0.0f || attraction_weight_input > 1.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::enableAttractionPoints): attraction_weight must be between 0.0 and 1.0.");
    }

    // Set global attraction points for backward compatibility
    attraction_points_enabled = true;
    attraction_points = attraction_points_input;
    attraction_cone_half_angle_rad = deg2rad(view_half_angle_deg);
    attraction_cone_height = look_ahead_distance;
    attraction_weight = attraction_weight_input;

    // Also apply to all existing plants for backward compatibility
    for (auto &[plantID, plant]: plant_instances) {
        plant.attraction_points_enabled = true;
        plant.attraction_points = attraction_points_input;
        plant.attraction_cone_half_angle_rad = deg2rad(view_half_angle_deg);
        plant.attraction_cone_height = look_ahead_distance;
        plant.attraction_weight = attraction_weight_input;
    }
}

void PlantArchitecture::disableAttractionPoints() {
    // Disable global attraction points for backward compatibility
    attraction_points_enabled = false;
    attraction_points.clear();

    // Also disable for all existing plants for backward compatibility
    for (auto &[plantID, plant]: plant_instances) {
        plant.attraction_points_enabled = false;
        plant.attraction_points.clear();
    }
}

void PlantArchitecture::updateAttractionPoints(const std::vector<helios::vec3> &attraction_points_input) {
    if (!attraction_points_enabled) {
        helios_runtime_error("ERROR (PlantArchitecture::updateAttractionPoints): Attraction points must be enabled before updating positions.");
    }
    if (attraction_points_input.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::updateAttractionPoints): attraction_points cannot be empty.");
    }

    // Update global attraction points for backward compatibility
    attraction_points = attraction_points_input;

    // Also update for all existing plants for backward compatibility
    for (auto &[plantID, plant]: plant_instances) {
        if (plant.attraction_points_enabled) {
            plant.attraction_points = attraction_points_input;
        }
    }
}

void PlantArchitecture::appendAttractionPoints(const std::vector<helios::vec3> &attraction_points_input) {
    if (!attraction_points_enabled) {
        helios_runtime_error("ERROR (PlantArchitecture::appendAttractionPoints): Attraction points must be enabled before updating positions.");
    }
    if (attraction_points_input.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendAttractionPoints): attraction_points cannot be empty.");
    }

    // Append to global attraction points for backward compatibility
    attraction_points.insert(attraction_points.end(), attraction_points_input.begin(), attraction_points_input.end());

    // Also append for all existing plants for backward compatibility
    for (auto &[plantID, plant]: plant_instances) {
        if (plant.attraction_points_enabled) {
            plant.attraction_points.insert(plant.attraction_points.end(), attraction_points_input.begin(), attraction_points_input.end());
        }
    }
}

void PlantArchitecture::setAttractionParameters(float view_half_angle_deg, float look_ahead_distance, float attraction_weight_input, float obstacle_reduction_factor) {
    if (view_half_angle_deg <= 0.0f || view_half_angle_deg > 180.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): view_half_angle_deg must be between 0 and 180 degrees.");
    }
    if (look_ahead_distance <= 0.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): look_ahead_distance must be positive.");
    }
    if (attraction_weight_input < 0.0f || attraction_weight_input > 1.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): attraction_weight must be between 0.0 and 1.0.");
    }
    if (obstacle_reduction_factor < 0.0f || obstacle_reduction_factor > 1.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): obstacle_reduction_factor must be between 0.0 and 1.0.");
    }

    // Update global attraction parameters for backward compatibility
    attraction_cone_half_angle_rad = deg2rad(view_half_angle_deg);
    attraction_cone_height = look_ahead_distance;
    attraction_weight = attraction_weight_input;
    attraction_obstacle_reduction_factor = obstacle_reduction_factor;

    // Also update for all existing plants for backward compatibility
    for (auto &[plantID, plant]: plant_instances) {
        if (plant.attraction_points_enabled) {
            plant.attraction_cone_half_angle_rad = deg2rad(view_half_angle_deg);
            plant.attraction_cone_height = look_ahead_distance;
            plant.attraction_weight = attraction_weight_input;
            plant.attraction_obstacle_reduction_factor = obstacle_reduction_factor;
        }
    }

    if (printmessages) {
        std::cout << "Updated attraction parameters: cone_angle=" << view_half_angle_deg << "°, look_ahead=" << look_ahead_distance << "m, weight=" << attraction_weight_input << ", obstacle_reduction=" << obstacle_reduction_factor << std::endl;
        if (!plant_instances.empty()) {
            std::cout << "Applied to " << plant_instances.size() << " existing plants with attraction points enabled" << std::endl;
        }
    }
}

// Plant-specific attraction point methods

void PlantArchitecture::enableAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points_input, float view_half_angle_deg, float look_ahead_distance, float attraction_weight_input) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::enableAttractionPoints): Plant with ID " + std::to_string(plantID) + " does not exist.");
    }

    if (view_half_angle_deg <= 0.0f || view_half_angle_deg > 180.f) {
        helios_runtime_error("ERROR (PlantArchitecture::enableAttractionPoints): view_half_angle_deg must be between 0 and 180 degrees.");
    }
    if (look_ahead_distance <= 0.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::enableAttractionPoints): look_ahead_distance must be greater than 0.");
    }
    if (attraction_points_input.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::enableAttractionPoints): attraction_points cannot be empty.");
    }

    auto &plant = plant_instances.at(plantID);
    plant.attraction_points_enabled = true;
    plant.attraction_points = attraction_points_input;
    plant.attraction_cone_half_angle_rad = deg2rad(view_half_angle_deg);
    plant.attraction_cone_height = look_ahead_distance;
    plant.attraction_weight = attraction_weight_input;

    if (printmessages) {
        std::cout << "Enabled attraction points for plant " << plantID << " with " << attraction_points_input.size() << " target positions" << std::endl;
        std::cout << "Plant " << plantID << " attraction parameters: cone_angle=" << view_half_angle_deg << "°, look_ahead=" << look_ahead_distance << "m, weight=" << attraction_weight_input << std::endl;
    }
}

void PlantArchitecture::disableAttractionPoints(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::disableAttractionPoints): Plant with ID " + std::to_string(plantID) + " does not exist.");
    }

    auto &plant = plant_instances.at(plantID);
    plant.attraction_points_enabled = false;
    plant.attraction_points.clear();

    if (printmessages) {
        std::cout << "Disabled attraction points for plant " << plantID << " - will use natural growth patterns" << std::endl;
    }
}

void PlantArchitecture::updateAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points_input) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::updateAttractionPoints): Plant with ID " + std::to_string(plantID) + " does not exist.");
    }

    auto &plant = plant_instances.at(plantID);
    if (!plant.attraction_points_enabled) {
        helios_runtime_error("ERROR (PlantArchitecture::updateAttractionPoints): Attraction points must be enabled for plant " + std::to_string(plantID) + " before updating positions.");
    }
    if (attraction_points_input.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::updateAttractionPoints): attraction_points cannot be empty.");
    }

    plant.attraction_points = attraction_points_input;
}

void PlantArchitecture::appendAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points_input) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendAttractionPoints): Plant with ID " + std::to_string(plantID) + " does not exist.");
    }

    auto &plant = plant_instances.at(plantID);
    if (!plant.attraction_points_enabled) {
        helios_runtime_error("ERROR (PlantArchitecture::appendAttractionPoints): Attraction points must be enabled for plant " + std::to_string(plantID) + " before updating positions.");
    }
    if (attraction_points_input.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendAttractionPoints): attraction_points cannot be empty.");
    }

    plant.attraction_points.insert(plant.attraction_points.end(), attraction_points_input.begin(), attraction_points_input.end());
}

void PlantArchitecture::setAttractionParameters(uint plantID, float view_half_angle_deg, float look_ahead_distance, float attraction_weight_input, float obstacle_reduction_factor) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): Plant with ID " + std::to_string(plantID) + " does not exist.");
    }

    if (view_half_angle_deg <= 0.0f || view_half_angle_deg > 180.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): view_half_angle_deg must be between 0 and 180 degrees.");
    }
    if (look_ahead_distance <= 0.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): look_ahead_distance must be greater than 0.");
    }
    if (obstacle_reduction_factor < 0.0f || obstacle_reduction_factor > 1.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setAttractionParameters): obstacle_reduction_factor must be between 0 and 1.");
    }

    auto &plant = plant_instances.at(plantID);
    plant.attraction_cone_half_angle_rad = deg2rad(view_half_angle_deg);
    plant.attraction_cone_height = look_ahead_distance;
    plant.attraction_weight = attraction_weight_input;
    plant.attraction_obstacle_reduction_factor = obstacle_reduction_factor;

    if (printmessages) {
        std::cout << "Updated attraction parameters for plant " << plantID << ": cone_angle=" << view_half_angle_deg << "°, look_ahead=" << look_ahead_distance << "m, weight=" << attraction_weight_input
                  << ", obstacle_reduction=" << obstacle_reduction_factor << std::endl;
    }
}

void PlantArchitecture::setPlantAttractionPoints(uint plantID, const std::vector<helios::vec3> &attraction_points_input, float view_half_angle_deg, float look_ahead_distance, float attraction_weight_input, float obstacle_reduction_factor) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAttractionPoints): Plant with ID " + std::to_string(plantID) + " does not exist.");
    }

    if (view_half_angle_deg <= 0.0f || view_half_angle_deg > 180.f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAttractionPoints): view_half_angle_deg must be between 0 and 180 degrees.");
    }
    if (look_ahead_distance <= 0.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAttractionPoints): look_ahead_distance must be greater than 0.");
    }
    if (attraction_points_input.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAttractionPoints): attraction_points cannot be empty.");
    }
    if (obstacle_reduction_factor < 0.0f || obstacle_reduction_factor > 1.0f) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantAttractionPoints): obstacle_reduction_factor must be between 0 and 1.");
    }

    auto &plant = plant_instances.at(plantID);
    plant.attraction_points_enabled = true;
    plant.attraction_points = attraction_points_input;
    plant.attraction_cone_half_angle_rad = deg2rad(view_half_angle_deg);
    plant.attraction_cone_height = look_ahead_distance;
    plant.attraction_weight = attraction_weight_input;
    plant.attraction_obstacle_reduction_factor = obstacle_reduction_factor;

    if (printmessages) {
        std::cout << "Set attraction points for plant " << plantID << " with " << attraction_points_input.size() << " target positions (internal library call)" << std::endl;
    }
}

void PlantArchitecture::disableMessages() {
    printmessages = false;
    if (collision_detection_ptr != nullptr) {
        collision_detection_ptr->disableMessages();
    }
}

void PlantArchitecture::enableMessages() {
    printmessages = true;
    if (collision_detection_ptr != nullptr) {
        collision_detection_ptr->enableMessages();
    }
}
