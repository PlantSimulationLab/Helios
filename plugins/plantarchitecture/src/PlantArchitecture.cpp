/** \file "PlantArchitecture.cpp" Primary source file for plant architecture plug-in.

    Copyright (C) 2016-2026 Brian Bailey

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

// Minimum thresholds for creating tube geometry to avoid malformed triangles
static const float MIN_TUBE_RADIUS_FOR_GEOMETRY = 1e-5f;
static const float MIN_TUBE_LENGTH_FOR_GEOMETRY = 1e-4f;

static void renameAutoMaterial(helios::Context *context_ptr, uint objID, const std::string &desired_base_name) {
    std::vector<uint> UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID);
    if (UUIDs.empty()) return;

    std::string current_label = context_ptr->getPrimitiveMaterialLabel(UUIDs.front());
    if (current_label.substr(0, 7) != "__auto_") return;

    if (!context_ptr->doesMaterialExist(desired_base_name)) {
        context_ptr->renameMaterial(current_label, desired_base_name);
    } else {
        uint existing_id = context_ptr->getMaterialIDFromLabel(desired_base_name);
        uint current_id = context_ptr->getMaterialIDFromLabel(current_label);
        if (existing_id == current_id) return;

        int suffix = 1;
        std::string candidate;
        do {
            candidate = desired_base_name + "_" + std::to_string(suffix++);
        } while (context_ptr->doesMaterialExist(candidate));
        context_ptr->renameMaterial(current_label, candidate);
    }
}

static void renameAutoMaterial(helios::Context *context_ptr, const std::vector<uint> &objIDs, const std::string &desired_base_name) {
    for (uint objID : objIDs) {
        renameAutoMaterial(context_ptr, objID, desired_base_name);
    }
}

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
    output_object_data["plant_name"] = false;
    output_object_data["plant_height"] = false;
    output_object_data["plant_type"] = false;
    output_object_data["phenology_stage"] = false;
    output_object_data["leafID"] = false;
    output_object_data["peduncleID"] = false;
    output_object_data["closedflowerID"] = false;
    output_object_data["openflowerID"] = false;
    output_object_data["fruitID"] = false;
    output_object_data["carbohydrate_concentration"] = false;
}

void PlantArchitecture::setProgressCallback(std::function<void(float, const std::string&)> callback) {
    progress_callback = std::move(callback);
}

void PlantArchitecture::setCancelFlag(volatile int *flag) {
    cancel_flag = flag;
}

PlantArchitecture::~PlantArchitecture() {
    // Clean up owned CollisionDetection instance
    if (collision_detection_ptr != nullptr && owns_collision_detection) {
        delete collision_detection_ptr;
        collision_detection_ptr = nullptr;
        owns_collision_detection = false;
    }
}

std::string PlantArchitecture::resolveTextureFile(const std::string &texture_file) {
    // Empty path returns empty string
    if (texture_file.empty()) {
        return "";
    }

    std::filesystem::path filepath(texture_file);

    // Absolute paths that exist are returned as-is
    if (filepath.is_absolute() && std::filesystem::exists(filepath)) {
        return texture_file;
    }

    // Try resolving as a general file path (handles already-resolved paths and paths relative to cwd)
    std::filesystem::path resolved_path = helios::tryResolveFilePath(texture_file);
    if (!resolved_path.empty()) {
        return resolved_path.string();
    }

    // Try resolving as a plugin asset path
    resolved_path = helios::tryResolvePluginAsset("plantarchitecture", texture_file);
    if (!resolved_path.empty()) {
        return resolved_path.string();
    }

    // If path doesn't have "assets/" prefix, try appropriate asset subdirectory based on file extension
    if (texture_file.find("assets/") != 0) {
        std::string filename = std::filesystem::path(texture_file).filename().string();
        std::string extension = std::filesystem::path(texture_file).extension().string();

        // Determine correct asset subdirectory based on file extension
        std::string subdirectory = (extension == ".obj" || extension == ".mtl") ? "assets/obj/" : "assets/textures/";
        std::string assets_path = subdirectory + filename;

        resolved_path = helios::tryResolvePluginAsset("plantarchitecture", assets_path);
        if (!resolved_path.empty()) {
            return resolved_path.string();
        }
    }

    // None of the resolution strategies worked
    helios_runtime_error("ERROR (PlantArchitecture): Could not resolve asset file: " + texture_file + ". Tried: direct path, plugin asset path, and assets/ subdirectory prefix.");
    return ""; // Never reached
}

LeafPrototype::LeafPrototype(std::minstd_rand0 *generator) : generator(generator) {
    leaf_aspect_ratio.initialize(1.f, generator);
    midrib_fold_fraction.initialize(0.f, generator);
    longitudinal_curvature.initialize(0.f, generator);
    // Defaults to the quartic the leaf prototype has always used, so a species that does not set it keeps its existing blade shape exactly.
    longitudinal_curvature_exponent.initialize(4.f, generator);
    lateral_curvature.initialize(0.f, generator);
    petiole_roll.initialize(0.f, generator);
    wave_period.initialize(0.f, generator);
    wave_amplitude.initialize(0.f, generator);
    flexibility.initialize(0.f, generator);
    // Defaults to a blade of uniform stiffness, which is what the deflection assumed before this parameter existed.
    flexibility_taper.initialize(1.f, generator);
    // Defaults to no ageing at all: droop then follows from leaf size alone, as it did before these parameters existed.
    flexibility_aging.initialize(0.f, generator);
    flexibility_aging_max.initialize(4.f, generator);
    HELIOS_PUSH_IGNORE_DEPRECATED
    leaf_buckle_length.initialize(0.f, generator);
    leaf_buckle_angle.initialize(0.f, generator);
    HELIOS_POP_IGNORE_DEPRECATED
    subdivisions = 1;
    unique_prototypes = 1;
    leaf_offset = make_vec3(0, 0, 0);
    prototype_function = GenericLeafPrototype;
    build_petiolule = false;
    if (generator != nullptr) {
        sampleIdentifier();
    }
}

std::minstd_rand0 *LeafPrototype::setRandomGenerator(std::minstd_rand0 *rand_generator) {
    std::minstd_rand0 *previous_generator = generator;
    generator = rand_generator;
    leaf_aspect_ratio.setRandomGenerator(rand_generator);
    midrib_fold_fraction.setRandomGenerator(rand_generator);
    longitudinal_curvature.setRandomGenerator(rand_generator);
    longitudinal_curvature_exponent.setRandomGenerator(rand_generator);
    lateral_curvature.setRandomGenerator(rand_generator);
    petiole_roll.setRandomGenerator(rand_generator);
    wave_period.setRandomGenerator(rand_generator);
    wave_amplitude.setRandomGenerator(rand_generator);
    flexibility.setRandomGenerator(rand_generator);
    flexibility_taper.setRandomGenerator(rand_generator);
    flexibility_aging.setRandomGenerator(rand_generator);
    flexibility_aging_max.setRandomGenerator(rand_generator);
    HELIOS_PUSH_IGNORE_DEPRECATED
    leaf_buckle_length.setRandomGenerator(rand_generator);
    leaf_buckle_angle.setRandomGenerator(rand_generator);
    HELIOS_POP_IGNORE_DEPRECATED
    return previous_generator;
}

float LeafPrototype::resolveFlexibility() {

    const float flexibility_value = flexibility.val();
    flexibility.resample();

    HELIOS_PUSH_IGNORE_DEPRECATED
    const float buckle_angle_degrees = leaf_buckle_angle.val();
    const float buckle_length_fraction = leaf_buckle_length.val();
    leaf_buckle_angle.resample();
    leaf_buckle_length.resample();
    HELIOS_POP_IGNORE_DEPRECATED

    // An explicitly-set flexibility always wins, so code that has migrated is unaffected by a stale buckle value sitting beside it, and a caller that deliberately wants a rigid leaf is not overridden by one.
    if (flexibility_value > 0.f || buckle_angle_degrees <= 0.f) {
        return flexibility_value;
    }

    // Convert the retired kink into the flexibility that reproduces it. Bending the blade through an angle A at a fraction f along its length dropped the tip by (1-f)*sin(A) of the leaf length; a leaf of a
    // given flexibility droops its tip by aspect*flexibility/8 of its length once fully grown, which is the standard uniformly-loaded cantilever result with the length dependence divided out. Equating the
    // two gives the expression below. Checked against the three grass models this replaced, it lands within about ten percent of the values obtained by numerically matching their old shapes.
    const float aspect_ratio = std::max(leaf_aspect_ratio.val(), 1e-3f);
    const float hinge_tip_droop_fraction = (1.f - clamp(buckle_length_fraction, 0.f, 1.f)) * sinf(deg2rad(buckle_angle_degrees));

    return std::max(8.f * hinge_tip_droop_fraction / aspect_ratio, 0.f);
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
    vegetative_bud_break_probability_max.initialize(1.0, generator);
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

std::vector<std::string> ShootParameters::getChildShootTypeLabels() const {
    return child_shoot_type_labels;
}

std::vector<float> ShootParameters::getChildShootTypeProbabilities() const {
    return child_shoot_type_probabilities;
}

void ShootParameters::inheritCustomFunctionsFrom(const ShootParameters &source) {
    if (this == &source) {
        return;
    }

    // A null pointer is a valid value here -- some shoot types deliberately have no creation
    // function -- so these are unconditional assignments rather than a merge that skips nulls.
    this->phytomer_parameters.phytomer_creation_function = source.phytomer_parameters.phytomer_creation_function;
    this->phytomer_parameters.phytomer_callback_function = source.phytomer_parameters.phytomer_callback_function;
    this->phytomer_parameters.leaf.prototype.prototype_function = source.phytomer_parameters.leaf.prototype.prototype_function;
    this->phytomer_parameters.inflorescence.flower_prototype_function = source.phytomer_parameters.inflorescence.flower_prototype_function;
    this->phytomer_parameters.inflorescence.fruit_prototype_function = source.phytomer_parameters.inflorescence.fruit_prototype_function;
}

std::vector<uint> PlantArchitecture::buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &plant_spacing_xy, const helios::int2 &plant_count_xy, const float age, const float germination_rate,
                                                                 const std::map<std::string, float> &build_parameters) {
    if (plant_count_xy.x <= 0 || plant_count_xy.y <= 0) {
        helios_runtime_error("ERROR (PlantArchitecture::buildPlantCanopyFromLibrary): Plant count must be greater than zero.");
    }

    vec2 canopy_extent(plant_spacing_xy.x * float(plant_count_xy.x - 1), plant_spacing_xy.y * float(plant_count_xy.y - 1));

    std::vector<uint> plantIDs;
    plantIDs.reserve(plant_count_xy.x * plant_count_xy.y);
    for (int j = 0; j < plant_count_xy.y; j++) {
        // Cancellation checkpoint between plants: a cancelled canopy build stops
        // here (per-plant build is monolithic) and returns what was built so far.
        if (cancel_flag != nullptr && *cancel_flag != 0) {
            return plantIDs;
        }
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

std::vector<uint> PlantArchitecture::buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &canopy_extent_xy, const uint plant_count, const float age, const std::map<std::string, float> &build_parameters) {
    std::vector<uint> plantIDs;
    plantIDs.reserve(plant_count);
    for (int i = 0; i < plant_count; i++) {
        // Cancellation checkpoint between plants (see the spacing-based overload).
        if (cancel_flag != nullptr && *cancel_flag != 0) {
            return plantIDs;
        }
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
                float obj_area = context_ptr->getObjectArea(leaf_objID);
                float scale_factor = current_leaf_scale_factor.at(p);
                float scaled_area = obj_area / powi(scale_factor, 2);
                leaf_area += scaled_area;
            }
        }
        p++;
    }
    return leaf_area;
}

float Phytomer::getInflorescenceArea() const {
    float inflorescence_area = 0;
    for (const auto &petiole: floral_buds) {
        for (const auto &fbud: petiole) {
            for (uint objID: fbud.inflorescence_objIDs) {
                if (context_ptr->doesObjectExist(objID)) {
                    // Reported at full size, matching getLeafArea(): the area is divided out by the current scale factor so that a structure part-way through its growth still reports the area it is growing
                    // toward. The girth it demands of the stem should not shrink back as it expands.
                    const float scale_factor = fbud.current_fruit_scale_factor;
                    if (scale_factor > 0.f) {
                        inflorescence_area += context_ptr->getObjectArea(objID) / powi(scale_factor, 2);
                    }
                }
            }
        }
    }
    return inflorescence_area;
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
            plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(this->parent_shoot_ID)->sugar_pool_molC -= flower_cost;
        } else if (state == BUD_FRUITING) {
            // adding a fruit
            float fruit_cost = calculateFruitConstructionCosts(fbud);
            fbud.previous_fruit_scale_factor = fbud.current_fruit_scale_factor;
            if (plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(this->parent_shoot_ID)->sugar_pool_molC > fruit_cost) {
                plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_tree.at(this->parent_shoot_ID)->sugar_pool_molC -= fruit_cost;
            } else {
                setFloralBudState(BUD_DEAD, fbud);
            }
        }
    }

    // Delete geometry from previous reproductive state (if present)
    context_ptr->deleteObject(fbud.inflorescence_objIDs);
    fbud.inflorescence_objIDs.resize(0);
    fbud.inflorescence_bases.resize(0);
    fbud.inflorescence_rotation.resize(0);
    fbud.inflorescence_base_scales.resize(0);

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

    // Use smaller cone angle for hard obstacle detection
    float hard_detection_cone_angle = deg2rad(20.0f);
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
        }
        return false;
    }

    if (half_angle_degrees <= 0.0f || half_angle_degrees >= 180.0f) {
        if (printmessages) {
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
        }
        return false;
    }

    if (half_angle_degrees <= 0.0f || half_angle_degrees >= 180.0f) {
        if (printmessages) {
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
            // If the parent phytomer has no petioles, create a ghost petiole perpendicular to the internode
            if (shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->petiole_vertices.empty()) {
                parent_petiole_axis = cross(parent_internode_axis, make_vec3(0, 0, 1));
                if (parent_petiole_axis.magnitude() < 0.01f) {
                    // Internode is nearly vertical
                    parent_petiole_axis = make_vec3(0, 1, 0);
                }
                parent_petiole_axis.normalize();
                // Rotate ghost petiole by cumulative phyllotactic angle to match phyllotactic patterning
                float phyllotactic_angle = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->internode_phyllotactic_angle;
                float cumulative_rotation = float(parent_node_index) * phyllotactic_angle;
                parent_petiole_axis = rotatePointAboutLine(parent_petiole_axis, make_vec3(0, 0, 0), parent_internode_axis, cumulative_rotation);
            } else {
                parent_petiole_axis = shoot_tree_ptr->at(parent_shoot_ID)->phytomers.at(parent_node_index)->getPetioleAxisVector(0.f, parent_petiole_index);
            }
        }
        internode_base_position = base_position;
    } else {
        // additional phytomer being added to an existing shoot
        parent_internode_axis = phytomers.back()->getInternodeAxisVector(1.f);
        // If the parent phytomer has no petioles, create a ghost petiole perpendicular to the internode
        if (phytomers.back()->petiole_vertices.empty()) {
            parent_petiole_axis = cross(parent_internode_axis, make_vec3(0, 0, 1));
            if (parent_petiole_axis.magnitude() < 0.01f) {
                // Internode is nearly vertical
                parent_petiole_axis = make_vec3(0, 1, 0);
            }
            parent_petiole_axis.normalize();
            // Rotate ghost petiole by cumulative phyllotactic angle to match phyllotactic patterning
            uint prev_phytomer_index = phytomers.size() - 1;
            float phyllotactic_angle = phytomers.back()->internode_phyllotactic_angle;
            float cumulative_rotation = float(prev_phytomer_index) * phyllotactic_angle;
            parent_petiole_axis = rotatePointAboutLine(parent_petiole_axis, make_vec3(0, 0, 0), parent_internode_axis, cumulative_rotation);
        } else {
            parent_petiole_axis = phytomers.back()->getPetioleAxisVector(0.f, 0);
        }
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
        if (plantarchitecture_ptr->output_object_data.at("plant_name")) {
            context_ptr->setObjectData(internode_tube_objID, "plant_name", plantarchitecture_ptr->plant_instances.at(plantID).plant_name);
        }
    }
    if (plantarchitecture_ptr->build_context_geometry_petiole) {
        const std::vector<uint> petiole_objIDs_existing = phytomer->getExistingPetioleObjIDs();
        if (plantarchitecture_ptr->output_object_data.at("age")) {
            context_ptr->setObjectData(petiole_objIDs_existing, "age", phytomer->age);
        }
        if (plantarchitecture_ptr->output_object_data.at("rank")) {
            context_ptr->setObjectData(petiole_objIDs_existing, "rank", phytomer->rank);
        }
        if (plantarchitecture_ptr->output_object_data.at("plantID")) {
            context_ptr->setObjectData(petiole_objIDs_existing, "plantID", (int) plantID);
        }
        if (plantarchitecture_ptr->output_object_data.at("plant_name")) {
            context_ptr->setObjectData(petiole_objIDs_existing, "plant_name", plantarchitecture_ptr->plant_instances.at(plantID).plant_name);
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
    if (plantarchitecture_ptr->output_object_data.at("plant_name")) {
        context_ptr->setObjectData(phytomer->leaf_objIDs, "plant_name", plantarchitecture_ptr->plant_instances.at(plantID).plant_name);
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

    for (int p = 0; p < phytomers.size(); p++) {
        float phytomer_volume = phytomers.at(p)->calculatePhytomerVolume(p);
        shoot_volume += phytomer_volume;
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
    // A pruned shoot is left in the shoot_tree as an empty shell: its phytomers and internode vertices
    // were cleared (see Phytomer::deletePhytomer). It is still reachable here through the parent's childIDs
    // recursion below, so bail out before dereferencing the now-empty shoot_internode_vertices. All
    // descendants of a pruned shoot are likewise empty, so skipping the child recursion loses nothing.
    // Note: test phytomers rather than internode_tube_objID existence, since the tube object is legitimately
    // absent when internode context geometry is disabled (build_context_geometry_internode == false).
    if (phytomers.empty()) {
        return;
    }

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
        Shoot *parent_shoot = plantarchitecture_ptr->plant_instances.at(shoot->plantID).shoot_tree.at(shoot->parent_shoot_ID).get();
        propagateDownstreamLeafArea(parent_shoot, shoot->parent_node_index, leaf_area);
    }
}


float Shoot::sumShootLeafArea(uint start_node_index) const {
    // A pruned shoot has no phytomers and so no leaf area. Without this the range check below
    // rejects start_node_index 0 (0 >= 0) and reports it as a caller error.
    if (isPruned()) {
        return 0.f;
    }
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


float Shoot::refreshDownstreamInflorescenceArea() {
    // Inflorescence area cannot be accumulated the way downstream_leaf_area is, because a fruit or panicle keeps growing after it is created -- it starts at a quarter scale and expands -- so a value
    // banked once when the floral bud opened would describe a structure the plant no longer has. It is therefore recomputed, but once per shoot per time step rather than once per node.
    //
    // The sums are built by a single sweep from the shoot tip downward, so that the running total is carried from one node to the next: the value at node p is this phytomer's own inflorescence area plus
    // the value already computed at node p+1. Summing the suffix independently at every node instead -- which is what the original on-demand form did, once per node from the girth loop -- re-walks the
    // whole shoot and every subtree hanging off it for each node in turn, making the cost quadratic in phytomer count and re-visiting a deep subtree once per ancestor node. On a mature branched fruiting
    // plant that dominated the entire growth step.
    if (isPruned()) {
        return 0.f;
    }

    float running_total = 0;

    for (int p = static_cast<int>(phytomers.size()) - 1; p >= 0; p--) {
        running_total += phytomers.at(p)->getInflorescenceArea();

        // Inflorescences borne on child shoots load this shoot too, so they are counted the same way sumShootLeafArea() counts their leaves. Each child subtree is visited exactly once here, and its own
        // cached total is refreshed by the same recursion.
        if (childIDs.find(p) != childIDs.end()) {
            for (int child_shoot_ID: childIDs.at(p)) {
                running_total += plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(child_shoot_ID)->refreshDownstreamInflorescenceArea();
            }
        }

        phytomers.at(p)->downstream_inflorescence_area = running_total;
    }

    return running_total;
}

float Shoot::sumDownstreamInflorescenceArea(uint start_node_index) const {
    if (isPruned() || start_node_index >= phytomers.size()) {
        return 0.f;
    }

    return phytomers.at(start_node_index)->downstream_inflorescence_area;
}


float Shoot::sumChildVolume(uint start_node_index) const {
    // A pruned shoot has no phytomers and so no child shoots. See the note in sumShootLeafArea().
    if (isPruned()) {
        return 0.f;
    }
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

    // Note: build_context_geometry_petiole and build_context_geometry_peduncle are Phytomer members
    // (not locals), so that the decision made here at construction is the same one consulted later
    // when the geometry is deleted or transformed. Declaring locals of the same name here would
    // shadow the members and leave them permanently at their default of true.
    bool build_context_geometry_internode = plantarchitecture_ptr->build_context_geometry_internode;
    this->build_context_geometry_petiole = plantarchitecture_ptr->build_context_geometry_petiole;
    this->build_context_geometry_peduncle = plantarchitecture_ptr->build_context_geometry_peduncle;

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

    if (phytomer_parameters.petiole.petioles_per_internode == 0) {
        // Allow 0 petioles per internode, but ensure no leaves are created
        build_context_geometry_petiole = false;
        phytomer_parameters.leaf.leaves_per_petiole = 0;
    }

    if (phytomer_parameters.petiole.petioles_per_internode < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer): Number of petioles per internode cannot be negative.");
    }

    current_internode_scale_factor = internode_length_scale_factor_fraction;
    current_leaf_scale_factor.resize(phytomer_parameters.petiole.petioles_per_internode);
    std::fill(current_leaf_scale_factor.begin(), current_leaf_scale_factor.end(), leaf_scale_factor_fraction);

    if (internode_radius == 0.f) {
        internode_radius = MIN_TUBE_RADIUS_FOR_GEOMETRY;
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

    // initialize peduncle vertices and radii storage (will be resized when floral buds are added)
    peduncle_vertices.resize(phytomer_parameters.petiole.petioles_per_internode);
    peduncle_radii.resize(phytomer_parameters.petiole.petioles_per_internode);
    peduncle_length.resize(phytomer_parameters.petiole.petioles_per_internode);
    peduncle_radius.resize(phytomer_parameters.petiole.petioles_per_internode);
    peduncle_pitch.resize(phytomer_parameters.petiole.petioles_per_internode);
    peduncle_curvature.resize(phytomer_parameters.petiole.petioles_per_internode);
    peduncle_roll.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_pitch.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_curvature.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_taper.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_axis_initial.resize(phytomer_parameters.petiole.petioles_per_internode);
    petiole_rotation_axis.resize(phytomer_parameters.petiole.petioles_per_internode);
    std::vector<float> dr_petiole(phytomer_parameters.petiole.petioles_per_internode);
    std::vector<float> dr_petiole_max(phytomer_parameters.petiole.petioles_per_internode);
    // Per-petiole flag: if either radius or length is zero, suppress the petiole tube geometry but still compute vertices for leaf orientation
    std::vector<bool> suppress_petiole_geometry(phytomer_parameters.petiole.petioles_per_internode, false);
    for (int p = 0; p < phytomer_parameters.petiole.petioles_per_internode; p++) {
        petiole_vertices.at(p).resize(Ndiv_petiole_length + 1);
        petiole_radii.at(p).resize(Ndiv_petiole_length + 1);

        suppress_petiole_geometry.at(p) = (phytomer_parameters.petiole.radius.val() <= 0.f || phytomer_parameters.petiole.length.val() <= 0.f);

        petiole_length.at(p) = leaf_scale_factor_fraction * phytomer_parameters.petiole.length.val();
        if (petiole_length.at(p) <= 0.f) {
            petiole_length.at(p) = MIN_TUBE_LENGTH_FOR_GEOMETRY;
        }
        dr_petiole.at(p) = petiole_length.at(p) / float(phytomer_parameters.petiole.length_segments);
        dr_petiole_max.at(p) = phytomer_parameters.petiole.length.val() / float(phytomer_parameters.petiole.length_segments);

        petiole_radii.at(p).at(0) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val();
        if (petiole_radii.at(p).at(0) <= 0.f) {
            petiole_radii.at(p).at(0) = MIN_TUBE_RADIUS_FOR_GEOMETRY;
        }
    }
    phytomer_parameters.petiole.length.resample();
    // Always initialize petiole_objIDs vector for potential lazy creation later. Filled with the
    // sentinel rather than default-constructed, since 0 is a valid object ID.
    petiole_objIDs.assign(phytomer_parameters.petiole.petioles_per_internode, no_petiole_objID);

    // initialize leaf variables
    leaf_bases.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_objIDs.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_prototype_index.resize(phytomer_parameters.petiole.petioles_per_internode);
    leaf_last_deformed_scale.resize(phytomer_parameters.petiole.petioles_per_internode);

    // Resolved once here and held for the life of the phytomer. This also honours the retired leaf_buckle_* parameters when the flexibility itself was never set, so that code written against them still
    // produces a drooping leaf rather than a rigid one.
    leaf_flexibility = phytomer_parameters.leaf.prototype.resolveFlexibility();
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

    // Debug output for first phytomer construction
    if (phytomer_index == 0) {
    }

    if (phytomer_index == 0) { // if this is the first phytomer along a shoot, apply the origin rotation about the parent axis

        // internode pitch rotation for phytomer base
        if (internode_pitch != 0.f) {
            if (phytomer_index == 0) {
            }
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, petiole_rotation_axis, 0.5f * internode_pitch);
            if (phytomer_index == 0) {
            }
        }

        float roll_nudge = 0.f;
        //\todo Not clear if this is still needed. It causes problems when you want to plant base roll to be exactly 0.
        //        if( shoot_base_rotation.roll/180.f == floor(shoot_base_rotation.roll/180.f) ) {
        //            roll_nudge = 0.2;
        //        }

        if (phytomer_index == 0) {
        }

        if (shoot_base_rotation.roll != 0.f || roll_nudge != 0.f) {
            if (phytomer_index == 0) {
            }
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll + roll_nudge);
            // small additional rotation is to make sure the petiole is not exactly vertical
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll + roll_nudge);
            if (phytomer_index == 0) {
            }
        }

        vec3 base_pitch_axis = -1 * cross(parent_internode_axis, parent_petiole_axis);

        // internode pitch rotation for shoot base rotation
        if (shoot_base_rotation.pitch != 0.f) {
            if (phytomer_index == 0) {
            }
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, base_pitch_axis, -shoot_base_rotation.pitch);
            if (phytomer_index == 0) {
            }
        }

        // internode yaw rotation for shoot base rotation
        if (shoot_base_rotation.yaw != 0) {
            if (phytomer_index == 0) {
            }
            petiole_rotation_axis = rotatePointAboutLine(petiole_rotation_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw);
            if (phytomer_index == 0) {
            }
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

    // Resize perturbation vectors to capture stochastic state for XML reconstruction
    internode_curvature_perturbations.resize(Ndiv_internode_length);
    internode_yaw_perturbations.resize(Ndiv_internode_length);

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
            internode_curvature_perturbations[inode_segment - 1] = parent_shoot->curvature_perturbation;
            float curvature_angle = deg2rad((parent_shoot->gravitropic_curvature * current_curvature_fact * dr_internode_max + parent_shoot->curvature_perturbation));
            internode_axis = rotatePointAboutLine(internode_axis, nullorigin, shoot_bending_axis, curvature_angle);

            parent_shoot->yaw_perturbation += -0.5f * parent_shoot->yaw_perturbation * dt + parent_shoot_parameters.tortuosity.val() * context_ptr->randn() * sqrt(dt);
            internode_yaw_perturbations[inode_segment - 1] = parent_shoot->yaw_perturbation;
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

        // Resolve internode texture path (allows users to specify simple paths like "OliveBark.jpg")
        std::string resolved_internode_texture = PlantArchitecture::resolveTextureFile(phytomer_parameters.internode.image_texture);

        if (!context_ptr->doesObjectExist(parent_shoot->internode_tube_objID)) {
            // first internode on shoot
            if (!resolved_internode_texture.empty()) {
                parent_shoot->internode_tube_objID = context_ptr->addTubeObject(Ndiv_internode_radius, phytomer_internode_vertices, phytomer_internode_radii, resolved_internode_texture.c_str(), uv_y);
            } else {
                parent_shoot->internode_tube_objID = context_ptr->addTubeObject(Ndiv_internode_radius, phytomer_internode_vertices, phytomer_internode_radii, internode_colors);
            }
            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(parent_shoot->internode_tube_objID), "object_label", "shoot");
            std::string stem_material_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + "_" + parent_shoot->shoot_type_label + "_stem";
            renameAutoMaterial(context_ptr, parent_shoot->internode_tube_objID, stem_material_name);
        } else {
            // appending internode to shoot
            for (int inode_segment = 1; inode_segment <= Ndiv_internode_length; inode_segment++) {
                if (!resolved_internode_texture.empty()) {
                    context_ptr->appendTubeSegment(parent_shoot->internode_tube_objID, phytomer_internode_vertices.at(inode_segment), phytomer_internode_radii.at(inode_segment), resolved_internode_texture.c_str(),
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
        if (shoot_index.y + 1 == shoot_index.z) {
            // Last phytomer on shoot - apply a small near-zero pitch so the petiole
            // appears nearly parallel to the internode (e.g. the flag leaf in grasses)
            // without leaving the petiole axis fully degenerate with the internode.
            petiole_pitch.at(petiole) = deg2rad(5.f);
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

        // Store true initial vectors before collision avoidance and curvature application
        this->petiole_axis_initial.at(petiole) = petiole_axis_actual;
        this->petiole_rotation_axis.at(petiole) = petiole_rotation_axis_actual;

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

        // Sample taper once and store (avoids resampling each iteration which was a bug)
        petiole_taper.at(petiole) = phytomer_parameters.petiole.taper.val();
        phytomer_parameters.petiole.taper.resample();

        petiole_vertices.at(petiole).at(0) = phytomer_internode_vertices.back();

        for (int j = 1; j <= Ndiv_petiole_length; j++) {
            if (fabs(petiole_curvature.at(petiole)) > 0) {
                petiole_axis_actual = rotatePointAboutLine(petiole_axis_actual, nullorigin, petiole_rotation_axis_actual, -deg2rad(petiole_curvature.at(petiole) * dr_petiole_max.at(petiole)));
            }

            petiole_vertices.at(petiole).at(j) = petiole_vertices.at(petiole).at(j - 1) + dr_petiole.at(petiole) * petiole_axis_actual;

            petiole_radii.at(petiole).at(j) = leaf_scale_factor_fraction * phytomer_parameters.petiole.radius.val() * (1.f - petiole_taper.at(petiole) / float(Ndiv_petiole_length) * float(j));
            petiole_colors.at(j) = phytomer_parameters.petiole.color;

            assert(!std::isnan(petiole_vertices.at(petiole).at(j).x) && std::isfinite(petiole_vertices.at(petiole).at(j).x));
            assert(!std::isnan(petiole_radii.at(petiole).at(j)) && std::isfinite(petiole_radii.at(petiole).at(j)));
        }

        if (build_context_geometry_petiole && !suppress_petiole_geometry.at(petiole)) {
            petiole_objIDs.at(petiole) = makePetioleTube(Ndiv_petiole_radius, petiole_vertices.at(petiole), petiole_radii.at(petiole), petiole_colors, context_ptr);
            if (context_ptr->doesObjectExist(petiole_objIDs.at(petiole))) {
                context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(petiole_objIDs.at(petiole)), "object_label", "petiole");
                std::string petiole_material_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + "_" + parent_shoot->shoot_type_label + "_petiole";
                renameAutoMaterial(context_ptr, petiole_objIDs.at(petiole), petiole_material_name);
            }
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
            plantarchitecture_ptr->unique_leaf_prototype_rest_geometry[phytomer_parameters.leaf.prototype.unique_prototype_identifier].resize(phytomer_parameters.leaf.prototype.unique_prototypes);

            // The set of unique blade shapes is drawn from a private stream keyed on the prototype identifier
            // rather than from the Context's generator. The set is built lazily, at the first phytomer that needs
            // it, and PlantArchitecture::readPlantStructureXML() builds it at a different point in the plant's
            // sequence of random draws than growing the plant does -- so drawing from the shared stream gave a
            // reloaded plant a different set of blade shapes than the one that was saved. See the matching block
            // in readPlantStructureXML().
            std::minstd_rand0 prototype_generator(phytomer_parameters.leaf.prototype.unique_prototype_identifier);
            std::minstd_rand0 *context_generator = phytomer_parameters.leaf.prototype.setRandomGenerator(&prototype_generator);

            for (int prototype = 0; prototype < phytomer_parameters.leaf.prototype.unique_prototypes; prototype++) {
                for (int leaf = 0; leaf < leaves_per_petiole; leaf++) {
                    float ind_from_tip = float(leaf) - float(leaves_per_petiole - 1) / 2.f;
                    uint objID_leaf = phytomer_parameters.leaf.prototype.prototype_function(context_ptr, &phytomer_parameters.leaf.prototype, ind_from_tip);
                    if (phytomer_parameters.leaf.prototype.prototype_function == GenericLeafPrototype) {
                        // A petiolule loaded from an OBJ arrives already labelled by its own group in the file. Labelling the whole object "leaf" would overwrite that, and the filter below - which gives the
                        // petiolule the petiole's colour, and which downstream code uses to give it the petiole's optical properties rather than the blade's - would then match nothing.
                        const std::vector<uint> object_UUIDs = context_ptr->getObjectPrimitiveUUIDs(objID_leaf);
                        const std::vector<uint> labelled_UUIDs = context_ptr->filterPrimitivesByData(object_UUIDs, "object_label", "petiolule");
                        const std::set<uint> keep_label(labelled_UUIDs.begin(), labelled_UUIDs.end());
                        std::vector<uint> blade_UUIDs;
                        blade_UUIDs.reserve(object_UUIDs.size());
                        for (uint UUID: object_UUIDs) {
                            if (keep_label.find(UUID) == keep_label.end()) {
                                blade_UUIDs.push_back(UUID);
                            }
                        }
                        context_ptr->setPrimitiveData(blade_UUIDs, "object_label", "leaf");
                    }
                    plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).push_back(objID_leaf);

                    plantarchitecture_ptr->recordLeafPrototypeRestGeometry(phytomer_parameters.leaf.prototype, prototype, objID_leaf, leaf_flexibility);
                    std::string material_base_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + "_" + parent_shoot->shoot_type_label + "_leaf";
                    renameAutoMaterial(context_ptr, objID_leaf, material_base_name);
                    std::vector<uint> petiolule_UUIDs = context_ptr->filterPrimitivesByData(context_ptr->getObjectPrimitiveUUIDs(objID_leaf), "object_label", "petiolule");
                    context_ptr->setPrimitiveColor(petiolule_UUIDs, phytomer_parameters.petiole.color);
                    context_ptr->hideObject(objID_leaf);
                }
            }

            phytomer_parameters.leaf.prototype.setRandomGenerator(context_generator);
        }

        for (int leaf = 0; leaf < leaves_per_petiole; leaf++) {
            float ind_from_tip = float(leaf) - float(leaves_per_petiole - 1) / 2.f;

            uint objID_leaf;
            // Which cached prototype this leaf is a copy of, so its rest shape can be found again when it is deflected. A leaf built fresh rather than copied has no cached rest shape and stays rigid.
            int leaf_prototype_source = -1;
            if (phytomer_parameters.leaf.prototype.unique_prototypes > 0) {
                // copy the existing prototype
                int prototype = context_ptr->randu(0, phytomer_parameters.leaf.prototype.unique_prototypes - 1);
                leaf_prototype_source = prototype;
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.find(phytomer_parameters.leaf.prototype.unique_prototype_identifier) != plantarchitecture_ptr->unique_leaf_prototype_objIDs.end());
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).size() > prototype);
                assert(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).size() > leaf);
                objID_leaf = context_ptr->copyObject(plantarchitecture_ptr->unique_leaf_prototype_objIDs.at(phytomer_parameters.leaf.prototype.unique_prototype_identifier).at(prototype).at(leaf));
            } else {
                // load a new prototype
                objID_leaf = phytomer_parameters.leaf.prototype.prototype_function(context_ptr, &phytomer_parameters.leaf.prototype, ind_from_tip);
                std::string material_base_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + "_" + parent_shoot->shoot_type_label + "_leaf";
                renameAutoMaterial(context_ptr, objID_leaf, material_base_name);
            }

            // -- leaf scaling -- //

            if (leaves_per_petiole > 0 && phytomer_parameters.leaf.leaflet_scale.val() != 1.f && ind_from_tip != 0) {
                leaf_size_max.at(petiole).at(leaf) = powf(phytomer_parameters.leaf.leaflet_scale.val(), fabs(ind_from_tip)) * phytomer_parameters.leaf.prototype_scale.val();
            } else {
                leaf_size_max.at(petiole).at(leaf) = phytomer_parameters.leaf.prototype_scale.val();
            }
            vec3 leaf_scale = leaf_scale_factor_fraction * leaf_size_max.at(petiole).at(leaf) * make_vec3(1, 1, 1);

            context_ptr->scaleObject(objID_leaf, leaf_scale);

            float compound_rotation = compoundLeafRotation(leaves_per_petiole, leaf, leaflet_offset_val);

            // -- leaf rotations -- //

            // Sampled here rather than inside orientLeaf() so that the order of the random draws, and
            // therefore every subsequent draw for the plant, is unchanged. orientLeaf() samples nothing.
            const float leaf_roll_angle = deg2rad(phytomer_parameters.leaf.roll.val());
            phytomer_parameters.leaf.roll.resample();
            const float leaf_pitch_angle = deg2rad(phytomer_parameters.leaf.pitch.val());
            phytomer_parameters.leaf.pitch.resample();
            float leaf_yaw_angle = 0;
            if (ind_from_tip != 0) {
                const float yaw_sign = -compound_rotation / fabs(compound_rotation);
                leaf_yaw_angle = yaw_sign * deg2rad(phytomer_parameters.leaf.yaw.val());
                phytomer_parameters.leaf.yaw.resample();
            }

            orientLeaf(objID_leaf, petiole, leaf, leaves_per_petiole, ind_from_tip, compound_rotation, petiole_tip_axis, leaf_roll_angle, leaf_pitch_angle, leaf_yaw_angle);


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
            leaf_prototype_index.at(petiole).push_back(leaf_prototype_source);
            // Nothing has been deflected yet, so record a scale that no leaf can already be at; the first growth step then always deforms.
            leaf_last_deformed_scale.at(petiole).push_back(-1.f);
        }
        phytomer_parameters.leaf.prototype_scale.resample();

        inflorescence_bending_axis = cross(parent_internode_axis, petiole_axis_actual);
        if (inflorescence_bending_axis == make_vec3(0, 0, 0)) {
            inflorescence_bending_axis = make_vec3(1, 0, 0);
        }
    }

    // Special case: if there are no petioles, still create vegetative buds directly on the internode
    if (phytomer_parameters.petiole.petioles_per_internode == 0) {
        std::vector<VegetativeBud> vegetative_buds_new;
        vegetative_buds_new.resize(phytomer_parameters.internode.max_vegetative_buds_per_petiole.val());
        phytomer_parameters.internode.max_vegetative_buds_per_petiole.resample();
        axillary_vegetative_buds.push_back(vegetative_buds_new);

        std::vector<FloralBud> floral_buds_new;
        floral_buds_new.resize(phytomer_parameters.internode.max_floral_buds_per_petiole.val());
        phytomer_parameters.internode.max_floral_buds_per_petiole.resample();
        floral_buds.push_back(floral_buds_new);
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

void Phytomer::createInflorescenceGeometry(FloralBud &fbud, const helios::vec3 &fruit_base, const helios::vec3 &peduncle_axis, float pitch, float roll, float azimuth, float yaw_compound, float scale_factor, bool is_open_flower) {

    // Step 1: Create flower/fruit prototype based on current bud state
    uint objID_fruit;
    if (fbud.state == BUD_FRUITING) {
        if (phytomer_parameters.inflorescence.unique_prototypes > 0) {
            // Copy existing prototype
            int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
            objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_fruit_prototype_objIDs.at(phytomer_parameters.inflorescence.fruit_prototype_function).at(prototype));
        } else {
            // Load new prototype
            objID_fruit = phytomer_parameters.inflorescence.fruit_prototype_function(context_ptr, 1);
            std::string fruit_material_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + "_fruit";
            renameAutoMaterial(context_ptr, objID_fruit, fruit_material_name);
        }
    } else {
        // Flower (open or closed)
        if (phytomer_parameters.inflorescence.unique_prototypes > 0) {
            // Copy existing prototype
            int prototype = context_ptr->randu(0, int(phytomer_parameters.inflorescence.unique_prototypes - 1));
            if (is_open_flower) {
                objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_open_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype));
            } else {
                objID_fruit = context_ptr->copyObject(plantarchitecture_ptr->unique_closed_flower_prototype_objIDs.at(phytomer_parameters.inflorescence.flower_prototype_function).at(prototype));
            }
        } else {
            // Load new prototype
            objID_fruit = phytomer_parameters.inflorescence.flower_prototype_function(context_ptr, 1, is_open_flower);
            std::string flower_material_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + (is_open_flower ? "_flower_open" : "_flower_closed");
            renameAutoMaterial(context_ptr, objID_fruit, flower_material_name);
        }
    }

    // Step 2: Scale the flower/fruit
    vec3 fruit_scale = scale_factor * make_vec3(1, 1, 1);
    context_ptr->scaleObject(objID_fruit, fruit_scale);

    // Step 3: Apply rotations at origin in correct order (roll, pitch, azimuth)
    if (std::abs(roll) > 1e-6) {
        context_ptr->rotateObject(objID_fruit, roll, "x");
    }
    if (std::abs(pitch) > 1e-6) {
        context_ptr->rotateObject(objID_fruit, pitch, "y");
    }
    if (std::abs(azimuth) > 1e-6) {
        context_ptr->rotateObject(objID_fruit, azimuth, "z");
    }

    // Step 4: Translate to position on peduncle
    context_ptr->translateObject(objID_fruit, fruit_base);

    // Step 5: Apply compound rotation about peduncle axis (or vertical axis for fruit with gravity)
    if (std::abs(yaw_compound) > 1e-6) {
        context_ptr->rotateObject(objID_fruit, yaw_compound, fruit_base, peduncle_axis);
    }

    // Step 6: Store in floral bud data structures
    fbud.inflorescence_objIDs.push_back(objID_fruit);
    fbud.inflorescence_bases.push_back(fruit_base);

    AxisRotation flower_rotation;
    flower_rotation.pitch = pitch;
    flower_rotation.yaw = yaw_compound;
    flower_rotation.roll = roll;
    flower_rotation.azimuth = azimuth;
    flower_rotation.peduncle_axis = peduncle_axis;
    fbud.inflorescence_rotation.push_back(flower_rotation);

    fbud.inflorescence_base_scales.push_back(scale_factor);

    assert(fbud.inflorescence_objIDs.size() == fbud.inflorescence_bases.size());
    assert(fbud.inflorescence_bases.size() == fbud.inflorescence_rotation.size());
    assert(fbud.inflorescence_rotation.size() == fbud.inflorescence_base_scales.size());
}

void PlantArchitecture::ensureInflorescencePrototypesInitialized(const PhytomerParameters &params, const std::string &plant_name) {
    if (params.inflorescence.unique_prototypes > 0) {
        // Initialize closed flower prototypes
        if (params.inflorescence.flower_prototype_function != nullptr && unique_closed_flower_prototype_objIDs.find(params.inflorescence.flower_prototype_function) == unique_closed_flower_prototype_objIDs.end()) {
            unique_closed_flower_prototype_objIDs[params.inflorescence.flower_prototype_function].resize(params.inflorescence.unique_prototypes);
            for (int prototype = 0; prototype < params.inflorescence.unique_prototypes; prototype++) {
                uint objID_flower = params.inflorescence.flower_prototype_function(context_ptr, 1, false);
                unique_closed_flower_prototype_objIDs.at(params.inflorescence.flower_prototype_function).at(prototype) = objID_flower;
                renameAutoMaterial(context_ptr, objID_flower, plant_name + "_flower_closed");
                context_ptr->hideObject(objID_flower);
            }
        }
        // Initialize open flower prototypes
        if (params.inflorescence.flower_prototype_function != nullptr && unique_open_flower_prototype_objIDs.find(params.inflorescence.flower_prototype_function) == unique_open_flower_prototype_objIDs.end()) {
            unique_open_flower_prototype_objIDs[params.inflorescence.flower_prototype_function].resize(params.inflorescence.unique_prototypes);
            for (int prototype = 0; prototype < params.inflorescence.unique_prototypes; prototype++) {
                uint objID_flower = params.inflorescence.flower_prototype_function(context_ptr, 1, true);
                unique_open_flower_prototype_objIDs.at(params.inflorescence.flower_prototype_function).at(prototype) = objID_flower;
                renameAutoMaterial(context_ptr, objID_flower, plant_name + "_flower_open");
                context_ptr->hideObject(objID_flower);
            }
        }
        // Initialize fruit prototypes
        if (params.inflorescence.fruit_prototype_function != nullptr && unique_fruit_prototype_objIDs.find(params.inflorescence.fruit_prototype_function) == unique_fruit_prototype_objIDs.end()) {
            unique_fruit_prototype_objIDs[params.inflorescence.fruit_prototype_function].resize(params.inflorescence.unique_prototypes);
            for (int prototype = 0; prototype < params.inflorescence.unique_prototypes; prototype++) {
                uint objID_fruit = params.inflorescence.fruit_prototype_function(context_ptr, 1);
                unique_fruit_prototype_objIDs.at(params.inflorescence.fruit_prototype_function).at(prototype) = objID_fruit;
                renameAutoMaterial(context_ptr, objID_fruit, plant_name + "_fruit");
                context_ptr->hideObject(objID_fruit);
            }
        }
    }
}

float Phytomer::getReconciledPeduncleRadius() {
    // The peduncle attaches to the culm at a single point -- Shoot::addTerminalFloralBud puts its base on the last internode vertex -- so if the two tubes disagree in radius the join is a visible step
    // rather than a taper. They were free to disagree: the peduncle radius is an independent constant, while the internode radius comes from the pipe model, which sizes each internode from the leaf area
    // above it. The terminal internode carries the inflorescence rather than leaves, so the pipe model gives it almost nothing and it ends up far thinner than the structure it has to support.
    //
    // Reconciling against the culm rather than clamping in one direction matters because the error runs both ways: sorghum's peduncle was about 2.7x wider than its culm tip, maize's about 1.9x narrower.
    const float culm_tip_radius = getInternodeRadius(1.f);
    if (culm_tip_radius <= 0.f || !peduncleShouldMatchCulm()) {
        // Nothing to match against, or a lateral peduncle whose configured radius is meaningful in its own right.
        return phytomer_parameters.peduncle.radius.val();
    }

    // A peduncle is slightly narrower than the culm bearing it, continuing the stem's taper rather than stepping down. No measured peduncle diameter was found in the literature for either species, so this
    // ratio is a calibration.
    //
    // The reconciled radius is used directly rather than being bounded by the configured one, because the mismatch runs in both directions: sorghum's peduncle was about 2.7x too wide, maize's about 1.9x
    // too narrow, and a one-sided bound would correct only one of them.
    constexpr float peduncle_to_culm_radius_ratio = 0.9f;
    return peduncle_to_culm_radius_ratio * culm_tip_radius;
}

bool Phytomer::peduncleShouldMatchCulm() const {
    // Only a peduncle that continues the stem's own axis should be sized from it. That is the case for a terminal inflorescence on an unbranched herbaceous culm -- a grass panicle or tassel sits directly on
    // the stem apex, is the same organ continued, and looks wrong at any radius the stem does not lead into.
    //
    // A lateral peduncle is a different structure: the flower stalks of almond, apple, walnut and the other woody species in the library are deliberately slender (0.5-1 mm) and hang off branches that
    // thicken to centimeters, so matching them to their parent would replace a flower stalk with a stub. Those keep their configured radius.
    if (!parent_shoot_ptr->shoot_parameters.phytomer_parameters.internode.image_texture.empty()) {
        // A bark texture marks a woody stem.
        return false;
    }
    return shoot_index.x + 1 == shoot_index.z;
}

void Phytomer::updatePeduncleRadii() {
    // Most phytomers on most plants carry no peduncle geometry at all, and this runs for every phytomer on every time step from the girth loop. Establishing that there is nothing to update costs a walk
    // over the bud lists; computing the radius first does not, since getReconciledPeduncleRadius() measures the internode. Check first, so the common case exits before doing any of that work.
    bool has_peduncle_geometry = false;
    for (const auto &petiole: floral_buds) {
        for (const auto &fbud: petiole) {
            if (!fbud.peduncle_objIDs.empty()) {
                has_peduncle_geometry = true;
                break;
            }
        }
        if (has_peduncle_geometry) {
            break;
        }
    }
    if (!has_peduncle_geometry) {
        return;
    }

    const float radius = getReconciledPeduncleRadius();

    for (uint petiole_idx = 0; petiole_idx < floral_buds.size(); petiole_idx++) {
        for (auto &fbud: floral_buds.at(petiole_idx)) {
            for (uint objID: fbud.peduncle_objIDs) {
                if (!context_ptr->doesObjectExist(objID)) {
                    continue;
                }
                const std::vector<float> existing_radii = context_ptr->getTubeObjectNodeRadii(objID);
                if (existing_radii.empty()) {
                    continue;
                }
                // setTubeRadii() regenerates the tube's primitives, so it is worth confirming there is actually a change to apply. The radius converges as the culm stops thickening, after which every
                // remaining time step would otherwise rebuild the same geometry to the same dimensions.
                bool radii_differ = false;
                for (float existing_radius: existing_radii) {
                    if (std::abs(existing_radius - radius) > 1e-9f) {
                        radii_differ = true;
                        break;
                    }
                }
                if (radii_differ) {
                    context_ptr->setTubeRadii(objID, std::vector<float>(existing_radii.size(), radius));
                }

                // The peduncle tube is built once, at the height the culm had that day, and nothing carries it upward as the internode beneath it goes on elongating: Shoot::updateShootNodes() rebuilds the
                // internode tube and repositions petioles and leaves, but never peduncles. Re-anchoring it to the floral bud's base -- which does track the internode tip -- keeps the two joined; without
                // this a mature sorghum culm ends nearly half a metre below its own panicle, with the head left floating.
                //
                // The anchor is measured from the rendered geometry rather than from the tube's node list, because the two do not agree. Phytomer::setPetioleBase() and Phytomer::setLeafScaleFraction()
                // both translate peduncle objects by a shift derived from the PETIOLE base -- correct for an axillary bud, which sits there, and wrong for a terminal bud, which is anchored to the internode
                // tip -- and those translations move the primitives without rewriting the node list. The primitives are what is drawn, so they are what has to be placed correctly.
                vec3 lower_corner;
                vec3 upper_corner;
                context_ptr->getObjectBoundingBox(objID, lower_corner, upper_corner);

                // The peduncle grows upward from its base, so the bottom of its bounding box is where it attaches to the culm.
                const vec3 rendered_base = make_vec3(0.5f * (lower_corner.x + upper_corner.x), 0.5f * (lower_corner.y + upper_corner.y), lower_corner.z);
                const vec3 anchor_shift = fbud.base_position - rendered_base;
                if (anchor_shift.magnitude() > 1e-6f) {
                    // Rewritten through the node list rather than by translating the object, so that the tube's nodes and the primitives generated from them stay in agreement -- Context::setTubeNodes()
                    // moves the primitives to match, whereas translating the object moves only the primitives and leaves the node list behind.
                    std::vector<vec3> nodes = context_ptr->getTubeObjectNodes(objID);
                    if (!nodes.empty()) {
                        const vec3 node_base = nodes.front();
                        for (vec3 &node: nodes) {
                            node = fbud.base_position + (node - node_base);
                        }
                        context_ptr->setTubeNodes(objID, nodes);
                    }
                }
            }
            // Keep the stored radius in step with the geometry, since it is what gets written out when the plant structure is saved to XML.
            if (petiole_idx < peduncle_radius.size() && fbud.bud_index < peduncle_radius.at(petiole_idx).size()) {
                peduncle_radius.at(petiole_idx).at(fbud.bud_index) = radius;
            }
        }
    }
}

void Phytomer::updateInflorescence(FloralBud &fbud) {
    // Assign the member rather than a local of the same name, so that whether peduncle geometry was
    // actually built here is the same flag consulted later when that geometry is deleted.
    this->build_context_geometry_peduncle = plantarchitecture_ptr->build_context_geometry_peduncle;

    uint Ndiv_peduncle_length = std::max(uint(1), phytomer_parameters.peduncle.length_segments);
    uint Ndiv_peduncle_radius = std::max(uint(3), phytomer_parameters.peduncle.radial_subdivisions);
    if (phytomer_parameters.peduncle.length_segments == 0 || phytomer_parameters.peduncle.radial_subdivisions < 3) {
        this->build_context_geometry_peduncle = false;
    }

    // Sample length once before calculating dr (same fix as petioles - don't resample until after geometry is created)
    float peduncle_length = phytomer_parameters.peduncle.length.val();
    float dr_peduncle = peduncle_length / float(Ndiv_peduncle_length);

    std::vector<vec3> peduncle_vertices(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_vertices.at(0) = fbud.base_position;
    std::vector<float> peduncle_radii(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_radii.at(0) = getReconciledPeduncleRadius();
    std::vector<RGBcolor> peduncle_colors(phytomer_parameters.peduncle.length_segments + 1);
    peduncle_colors.at(0) = phytomer_parameters.peduncle.color;

    vec3 peduncle_axis = getAxisVector(1.f, getInternodeNodePositions());

    // Create local copy of inflorescence_bending_axis that will be rotated with the peduncle
    vec3 inflorescence_bending_axis_actual = inflorescence_bending_axis;

    // peduncle pitch rotation
    if (phytomer_parameters.peduncle.pitch.val() != 0.f || fbud.base_rotation.pitch != 0.f) {
        peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, inflorescence_bending_axis_actual, deg2rad(phytomer_parameters.peduncle.pitch.val()) + fbud.base_rotation.pitch);
    }

    // rotate peduncle to azimuth of petiole and apply peduncle base yaw rotation
    vec3 internode_axis = getAxisVector(1.f, getInternodeNodePositions());
    vec3 parent_petiole_base_axis;
    if (petiole_vertices.empty()) {
        // No petioles - use internode axis instead
        parent_petiole_base_axis = internode_axis;
    } else {
        parent_petiole_base_axis = getPetioleAxisVector(0.f, fbud.parent_index);
    }
    float parent_petiole_azimuth = -std::atan2(parent_petiole_base_axis.y, parent_petiole_base_axis.x);
    float current_peduncle_azimuth = -std::atan2(peduncle_axis.y, peduncle_axis.x);
    float azimuthal_rotation = current_peduncle_azimuth - parent_petiole_azimuth;
    peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, internode_axis, azimuthal_rotation);
    // Rotate the bending axis by the same azimuthal angle to keep it perpendicular to the peduncle
    inflorescence_bending_axis_actual = rotatePointAboutLine(inflorescence_bending_axis_actual, nullorigin, internode_axis, azimuthal_rotation);


    float theta_base = fabs(cart2sphere(peduncle_axis).zenith);

    // Apply collision avoidance for peduncle direction (if enabled) - following petiole pattern
    vec3 collision_optimal_peduncle_direction;
    bool peduncle_collision_active = false;

    if (plantarchitecture_ptr->fruit_collision_detection_enabled) {
        collision_optimal_peduncle_direction = calculateFruitCollisionAvoidanceDirection(fbud.base_position, peduncle_axis, peduncle_collision_active);
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

    // Sample curvature once and store (avoids resampling each iteration which was a bug - same fix as petioles)
    float peduncle_curvature = phytomer_parameters.peduncle.curvature.val();
    phytomer_parameters.peduncle.curvature.resample();

    // Read the roll here so that the value stored below is the same one the inflorescence yaw is
    // built from further down. It must NOT be resampled here: peduncle.roll is resampled once at
    // the end of this function, and drawing a second sample would both shift every subsequent
    // random draw for the plant and store a roll that was never applied to any geometry.
    float peduncle_roll = phytomer_parameters.peduncle.roll.val();

    // Store actual sampled peduncle parameters for XML reconstruction
    uint petiole_idx = fbud.parent_index;
    uint bud_idx = fbud.bud_index;
    if (petiole_idx < this->peduncle_length.size()) {
        if (this->peduncle_length.at(petiole_idx).size() <= bud_idx) {
            this->peduncle_length.at(petiole_idx).resize(bud_idx + 1);
            this->peduncle_radius.at(petiole_idx).resize(bud_idx + 1);
            this->peduncle_pitch.at(petiole_idx).resize(bud_idx + 1);
            this->peduncle_curvature.at(petiole_idx).resize(bud_idx + 1);
            this->peduncle_roll.at(petiole_idx).resize(bud_idx + 1);
        }
        this->peduncle_length.at(petiole_idx).at(bud_idx) = peduncle_length;
        this->peduncle_radius.at(petiole_idx).at(bud_idx) = phytomer_parameters.peduncle.radius.val();
        this->peduncle_pitch.at(petiole_idx).at(bud_idx) = phytomer_parameters.peduncle.pitch.val();
        this->peduncle_curvature.at(petiole_idx).at(bud_idx) = peduncle_curvature;
        this->peduncle_roll.at(petiole_idx).at(bud_idx) = peduncle_roll;
    }

    for (int i = 1; i <= phytomer_parameters.peduncle.length_segments; i++) {
        if (peduncle_curvature != 0.f) {
            float curvature_value = peduncle_curvature;

            // Calculate horizontal bending axis perpendicular to current peduncle direction
            // This ensures bending is purely upward or downward
            vec3 horizontal_bending_axis = cross(peduncle_axis, make_vec3(0, 0, 1));
            float axis_magnitude = horizontal_bending_axis.magnitude();

            // Check if peduncle is nearly vertical (axis magnitude near zero)
            if (axis_magnitude > 0.001f) {
                horizontal_bending_axis = horizontal_bending_axis / axis_magnitude; // normalize

                // Calculate current angle from target vertical direction
                float theta_curvature = deg2rad(curvature_value * dr_peduncle);
                float theta_from_target;

                if (curvature_value > 0) {
                    // Positive curvature: target is upward (0, 0, 1)
                    // Current angle from target = acos(peduncle_axis.z)
                    theta_from_target = std::acos(std::min(1.0f, std::max(-1.0f, peduncle_axis.z)));
                } else {
                    // Negative curvature: target is downward (0, 0, -1)
                    // Current angle from target = acos(-peduncle_axis.z)
                    theta_from_target = std::acos(std::min(1.0f, std::max(-1.0f, -peduncle_axis.z)));
                }

                // Clamp rotation to not overshoot vertical
                if (fabs(theta_curvature) >= theta_from_target) {
                    // Would overshoot - snap to exact vertical
                    if (curvature_value > 0) {
                        peduncle_axis = make_vec3(0, 0, 1);
                    } else {
                        peduncle_axis = make_vec3(0, 0, -1);
                    }
                } else {
                    // Won't overshoot - apply rotation
                    peduncle_axis = rotatePointAboutLine(peduncle_axis, nullorigin, horizontal_bending_axis, theta_curvature);
                    peduncle_axis.normalize();
                }
            } else {
                // Already vertical - snap to correct vertical direction based on curvature sign
                if (curvature_value > 0) {
                    peduncle_axis = make_vec3(0, 0, 1); // upward
                } else {
                    peduncle_axis = make_vec3(0, 0, -1); // downward
                }
            }
        }

        peduncle_vertices.at(i) = peduncle_vertices.at(i - 1) + dr_peduncle * peduncle_axis;

        peduncle_radii.at(i) = peduncle_radii.at(0);
        peduncle_colors.at(i) = phytomer_parameters.peduncle.color;
    }

    if (build_context_geometry_peduncle) {
        fbud.peduncle_objIDs.push_back(context_ptr->addTubeObject(Ndiv_peduncle_radius, peduncle_vertices, peduncle_radii, peduncle_colors));
        context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(fbud.peduncle_objIDs.back()), "object_label", "peduncle");
        std::string peduncle_material_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + "_" + parent_shoot_ptr->shoot_type_label + "_peduncle";
        renameAutoMaterial(context_ptr, fbud.peduncle_objIDs.back(), peduncle_material_name);
    }

    // Store peduncle vertices for later axis vector calculations
    // Use the parent_index to determine which petiole this floral bud belongs to (petiole_idx already defined above)

    // Ensure the peduncle_vertices storage has the right size for this floral bud
    if (petiole_idx < this->peduncle_vertices.size()) {
        if (this->peduncle_vertices.at(petiole_idx).size() <= fbud.bud_index) {
            this->peduncle_vertices.at(petiole_idx).resize(fbud.bud_index + 1);
        }
        this->peduncle_vertices.at(petiole_idx).at(fbud.bud_index) = peduncle_vertices;
    }

    // Store peduncle radii alongside vertices for exact geometry reconstruction
    if (petiole_idx < this->peduncle_radii.size()) {
        if (this->peduncle_radii.at(petiole_idx).size() <= fbud.bud_index) {
            this->peduncle_radii.at(petiole_idx).resize(fbud.bud_index + 1);
        }
        this->peduncle_radii.at(petiole_idx).at(fbud.bud_index) = peduncle_radii;
    }

    // Resample parameters after geometry is created (same pattern as petioles - avoids mismatch between saved values and geometry)
    phytomer_parameters.peduncle.length.resample();
    phytomer_parameters.peduncle.radius.resample();
    phytomer_parameters.peduncle.pitch.resample();

    // Create unique inflorescence prototypes for each shoot type so we can simply copy them for each leaf
    plantarchitecture_ptr->ensureInflorescencePrototypesInitialized(phytomer_parameters, plantarchitecture_ptr->plant_instances.at(plantID).plant_name);

    int flowers_per_peduncle = phytomer_parameters.inflorescence.flowers_per_peduncle.val();
    float flower_offset_val = clampOffset(flowers_per_peduncle, phytomer_parameters.inflorescence.flower_offset.val());
    for (int fruit = 0; fruit < flowers_per_peduncle; fruit++) {
        // Determine scale factor based on bud state
        float scale_factor;
        if (fbud.state == BUD_FRUITING) {
            scale_factor = phytomer_parameters.inflorescence.fruit_prototype_scale.val();
            phytomer_parameters.inflorescence.fruit_prototype_scale.resample();
        } else {
            scale_factor = phytomer_parameters.inflorescence.flower_prototype_scale.val();
            phytomer_parameters.inflorescence.flower_prototype_scale.resample();
        }

        float ind_from_tip = fabs(fruit - float(flowers_per_peduncle - 1) / float(phytomer_parameters.petiole.petioles_per_internode));

        // Calculate position on peduncle
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

        // Calculate compound rotation about the peduncle
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

        vec3 peduncle_axis = getAxisVector(frac, peduncle_vertices);

        // Calculate rotation parameters (sample BEFORE resampling for XML storage)
        float applied_roll = deg2rad(phytomer_parameters.inflorescence.roll.val());
        phytomer_parameters.inflorescence.roll.resample();

        float applied_pitch_param = deg2rad(phytomer_parameters.inflorescence.pitch.val());
        phytomer_parameters.inflorescence.pitch.resample();

        // Calculate pitch with peduncle alignment and gravity (for fruit)
        float pitch_inflorescence = -asin_safe(peduncle_axis.z) + applied_pitch_param;
        if (fbud.state == BUD_FRUITING) {
            // gravity effect for fruit
            pitch_inflorescence = pitch_inflorescence + phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.val() * (0.5f * PI_F - pitch_inflorescence);
        }
        phytomer_parameters.inflorescence.fruit_gravity_factor_fraction.resample();

        // Calculate azimuth to align with peduncle orientation
        float azimuth = -std::atan2(peduncle_axis.y, peduncle_axis.x);

        // Calculate compound yaw (peduncle roll + compound rotation)
        float yaw_compound = deg2rad(peduncle_roll) + compound_rotation;

        // Determine if flower is open
        bool is_open_flower = (fbud.state == BUD_FLOWER_OPEN);

        // Call unified creation function
        createInflorescenceGeometry(fbud, fruit_base, peduncle_axis, pitch_inflorescence, applied_roll, azimuth, yaw_compound, scale_factor, is_open_flower);
    }
    phytomer_parameters.inflorescence.flowers_per_peduncle.resample();
    phytomer_parameters.peduncle.roll.resample();

    if (plantarchitecture_ptr->output_object_data.at("age")) {
        context_ptr->setObjectData(fbud.inflorescence_objIDs, "age", fbud.age);
        context_ptr->setObjectData(fbud.peduncle_objIDs, "age", fbud.age);
    }

    if (plantarchitecture_ptr->output_object_data.at("rank")) {
        context_ptr->setObjectData(fbud.peduncle_objIDs, "rank", rank);
        context_ptr->setObjectData(fbud.inflorescence_objIDs, "rank", rank);
    }

    if (plantarchitecture_ptr->output_object_data.at("plant_name")) {
        context_ptr->setObjectData(fbud.peduncle_objIDs, "plant_name", plantarchitecture_ptr->plant_instances.at(plantID).plant_name);
        context_ptr->setObjectData(fbud.inflorescence_objIDs, "plant_name", plantarchitecture_ptr->plant_instances.at(plantID).plant_name);
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
    // If there are no petioles, nothing to update
    if (petiole_vertices.empty()) {
        return;
    }

    vec3 old_base = petiole_vertices.front().front();
    vec3 shift = base_position - old_base;

    for (auto &petiole_vertice: petiole_vertices) {
        for (auto &vertex: petiole_vertice) {
            vertex += shift;
        }
    }

    if (build_context_geometry_petiole) {
        context_ptr->translateObject(getExistingPetioleObjIDs(), shift);
    }
    context_ptr->translateObject(flatten(leaf_objIDs), shift);

    for (auto &petiole: leaf_bases) {
        for (auto &leaf_base: petiole) {
            leaf_base += shift;
        }
    }
    // Update peduncle vertices when the phytomer is translated
    for (auto &petiole_peduncles: peduncle_vertices) {
        for (auto &bud_peduncle_vertices: petiole_peduncles) {
            for (auto &vertex: bud_peduncle_vertices) {
                vertex += shift;
            }
        }
    }

    for (auto &floral_bud: floral_buds) {
        for (auto &fbud: floral_bud) {
            // Translated by the same shift as everything else above, rather than reassigned to the petiole base. An axillary bud does sit at the petiole base and so was unaffected, but a terminal bud is
            // anchored to the internode tip (see Shoot::addTerminalFloralBud), and snapping it to the petiole put it a whole internode too low. That was invisible while the last internode was the same
            // length as its neighbours, and became a visible break between the culm and its panicle once the flag-leaf internode was allowed to elongate.
            fbud.base_position += shift;
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

void Phytomer::rotatePetiole(uint petiole_index, const AxisRotation &rotation) {
    if (petiole_index >= petiole_vertices.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer::rotatePetiole): Invalid petiole index.");
    }
    if (petiole_vertices.at(petiole_index).empty()) {
        return;
    }
    if (rotation.pitch == 0.f && rotation.yaw == 0.f && rotation.roll == 0.f) {
        return;
    }

    const vec3 base = petiole_vertices.at(petiole_index).at(0);
    const vec3 internode_axis = getInternodeAxisVector(1.f);

    auto applyRotation = [this, petiole_index, &base](float angle, const vec3 &axis) {
        if (angle == 0.f) {
            return;
        }
        if (context_ptr->doesObjectExist(petiole_objIDs.at(petiole_index))) {
            context_ptr->rotateObject(petiole_objIDs.at(petiole_index), angle, base, axis);
        }
        if (petiole_index < leaf_objIDs.size() && !leaf_objIDs.at(petiole_index).empty()) {
            context_ptr->rotateObject(leaf_objIDs.at(petiole_index), angle, base, axis);
        }
        for (auto &vertex: petiole_vertices.at(petiole_index)) {
            vertex = rotatePointAboutLine(vertex, base, axis, angle);
        }
        if (petiole_index < leaf_bases.size()) {
            for (auto &leaf_base: leaf_bases.at(petiole_index)) {
                leaf_base = rotatePointAboutLine(leaf_base, base, axis, angle);
            }
        }
        if (petiole_index < petiole_axis_initial.size()) {
            petiole_axis_initial.at(petiole_index) = rotatePointAboutLine(petiole_axis_initial.at(petiole_index), nullorigin, axis, angle);
        }
    };

    // pitch — tilt away from the internode using the same convention as construction:
    // rotate about the stored petiole_rotation_axis by abs(pitch). This matches
    // PlantArchitecture.cpp:~1876 (`rotatePointAboutLine(..., petiole_rotation_axis,
    // std::abs(petiole_pitch))`), so a positive input pitch always tilts the petiole
    // further from the internode regardless of which side petiole_rotation_axis fell
    // on during phyllotactic accumulation.
    if (rotation.pitch != 0.f && petiole_index < petiole_rotation_axis.size()) {
        vec3 pitch_axis = petiole_rotation_axis.at(petiole_index);
        if (pitch_axis.magnitude() > 1e-6f) {
            pitch_axis.normalize();
            applyRotation(std::abs(rotation.pitch), pitch_axis);
            petiole_pitch.at(petiole_index) += std::abs(rotation.pitch);
        }
    }

    // yaw — rotate around the internode axis (azimuth around the stem)
    if (rotation.yaw != 0.f) {
        vec3 yaw_axis = internode_axis;
        if (yaw_axis.magnitude() > 1e-6f) {
            yaw_axis.normalize();
            applyRotation(rotation.yaw, yaw_axis);
        }
    }

    // roll — rotate around the petiole's current length axis
    if (rotation.roll != 0.f) {
        vec3 roll_axis = getPetioleAxisVector(1.f, petiole_index);
        if (roll_axis.magnitude() > 1e-6f) {
            roll_axis.normalize();
            applyRotation(rotation.roll, roll_axis);
        }
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
    if (leaf_scale_factor_fraction == current_leaf_scale_factor.at(petiole_index) || (leaf_objIDs.at(petiole_index).empty() && !context_ptr->doesObjectExist(petiole_objIDs.at(petiole_index)))) {
        // The leaf has not changed size, so none of the scaling below has anything to do. The deflection is still applied: a leaf reaches its full length before it reaches its final droop, and this is the
        // path every mature leaf takes on every timestep from then on, so returning outright here would freeze each leaf at whatever droop it happened to have while it was still expanding.
        for (uint leaf = 0; leaf < leaf_objIDs.at(petiole_index).size(); leaf++) {
            deformLeafUnderSelfWeight(petiole_index, leaf);
        }
        return;
    }

    float delta_scale = leaf_scale_factor_fraction / current_leaf_scale_factor.at(petiole_index);

    petiole_length.at(petiole_index) *= delta_scale;

    current_leaf_scale_factor.at(petiole_index) = leaf_scale_factor_fraction;

    assert(leaf_objIDs.size() == leaf_bases.size());

    // scale the petiole geometry if it exists, or create it if it doesn't but should now

    // Scale the stored centerline about the petiole base, and the radii uniformly. The geometry is
    // then driven from these arrays, so both the existing-geometry and create-geometry cases below
    // work from the same scaled state.
    const vec3 base = petiole_vertices.at(petiole_index).at(0);
    for (uint node = 0; node < petiole_radii.at(petiole_index).size(); node++) {
        petiole_radii.at(petiole_index).at(node) *= delta_scale;
    }
    for (uint node = 1; node < petiole_vertices.at(petiole_index).size(); node++) {
        vec3 offset = petiole_vertices.at(petiole_index).at(node) - base;
        petiole_vertices.at(petiole_index).at(node) = base + offset * delta_scale;
    }

    if (context_ptr->doesObjectExist(petiole_objIDs.at(petiole_index))) {
        context_ptr->setTubeNodes(petiole_objIDs.at(petiole_index), petiole_vertices.at(petiole_index));
        context_ptr->setTubeRadii(petiole_objIDs.at(petiole_index), petiole_radii.at(petiole_index));
    } else if (build_context_geometry_petiole) {
        // Petiole geometry doesn't exist - try to create it now that it has been scaled up
        uint Ndiv_petiole_radius = std::max(uint(3), phytomer_parameters.petiole.radial_subdivisions);
        petiole_objIDs.at(petiole_index) = makePetioleTube(Ndiv_petiole_radius, petiole_vertices.at(petiole_index), petiole_radii.at(petiole_index), petiole_colors, context_ptr);
        if (context_ptr->doesObjectExist(petiole_objIDs.at(petiole_index))) {
            context_ptr->setPrimitiveData(context_ptr->getObjectPrimitiveUUIDs(petiole_objIDs.at(petiole_index)), "object_label", "petiole");
            std::string petiole_material_name = plantarchitecture_ptr->plant_instances.at(plantID).plant_name + "_" + parent_shoot_ptr->shoot_type_label + "_petiole";
            renameAutoMaterial(context_ptr, petiole_objIDs.at(petiole_index), petiole_material_name);
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

        deformLeafUnderSelfWeight(petiole_index, leaf);
    }
}

void PlantArchitecture::recordLeafPrototypeRestGeometry(const LeafPrototype &prototype_params, int prototype_index, uint objID_leaf, float leaf_flexibility) {

    // Keep the prototype's undeformed blade so that a drooping leaf can be re-deflected from its rest shape as it grows, instead of being bent further from wherever it already is. Stored once
    // per prototype and shared by every leaf copied from it, so this costs a handful of meshes per plant type rather than one per leaf. Only leaves that actually droop are recorded.
    LeafRestGeometry rest_geometry;
    if (leaf_flexibility > 0.f) {
        // Recorded in the prototype's own local frame, not the world frame that getPolymeshObjectVertices() returns. A leaf is a copyObject() of this prototype, and copyObject preserves the
        // local vertices while giving the copy its own transform; the deflected lattice is later carried through that transform. Storing world-space vertices would bake the prototype's own
        // transform into them and then apply the leaf's on top, displacing every leaf from its petiole by a constant offset.
        //
        // The prototype is built at the origin and is only ever translated, never rotated or scaled, so its frame is removed by subtracting that translation rather than by inverting the
        // matrix. The translation is read from the matrix itself so that a prototype placed somewhere else still resolves correctly.
        rest_geometry.vertices = context_ptr->getPolymeshObjectVertices(objID_leaf);
        float prototype_transform[16];
        context_ptr->getObjectTransformationMatrix(objID_leaf, prototype_transform);
        const vec3 prototype_origin = make_vec3(prototype_transform[3], prototype_transform[7], prototype_transform[11]);
        for (vec3 &vertex: rest_geometry.vertices) {
            vertex = vertex - prototype_origin;
        }
        // The blade is a regular lattice, so its dimensions can be recovered from the mesh itself. Deriving them here rather than recomputing them from the prototype parameters is what keeps
        // this correct for a prototype whose lateral subdivision count came from a resampled random aspect ratio, which the caller cannot reproduce.
        rest_geometry.subdivisions_x = prototype_params.subdivisions;
        if (rest_geometry.subdivisions_x > 0 && rest_geometry.vertices.size() % (rest_geometry.subdivisions_x + 1) == 0) {
            rest_geometry.subdivisions_y = uint(rest_geometry.vertices.size() / (rest_geometry.subdivisions_x + 1)) - 1;
        } else {
            // The mesh is not the plain lattice this deformation understands - an OBJ-loaded leaf, or one carrying a petiolule appended to the blade - so it is left rigid rather than deformed
            // by an index mapping that does not describe it.
            rest_geometry.vertices.clear();
            rest_geometry.subdivisions_x = 0;
        }
    }
    unique_leaf_prototype_rest_geometry[prototype_params.unique_prototype_identifier].at(prototype_index).push_back(rest_geometry);
}

float Phytomer::compoundLeafRotation(int leaves_per_petiole, int leaf_index, float leaflet_offset_val) {
    float compound_rotation = 0;
    if (leaves_per_petiole > 1) {
        if (leaflet_offset_val == 0) {
            float dphi = PI_F / (floor(0.5 * float(leaves_per_petiole - 1)) + 1);
            compound_rotation = -float(PI_F) + dphi * (leaf_index + 0.5f);
        } else {
            if (leaf_index == float(leaves_per_petiole - 1) / 2.f) {
                // tip leaf
                compound_rotation = 0;
            } else if (leaf_index < float(leaves_per_petiole - 1) / 2.f) {
                compound_rotation = -0.5 * PI_F;
            } else {
                compound_rotation = 0.5 * PI_F;
            }
        }
    }
    return compound_rotation;
}

void Phytomer::orientLeaf(uint objID_leaf, uint petiole_index, uint leaf_index, int leaves_per_petiole, float ind_from_tip, float compound_rotation, const helios::vec3 &petiole_tip_axis, float leaf_roll_angle, float leaf_pitch_angle,
                          float leaf_yaw_angle) {

    // leaf roll rotation
    // Roll-X here only applies the user-configured `leaf.roll` parameter; the
    // curvature-driven blade-up correction is done after the pitch+yaw chain
    // (see "blade-up correction" block below) so that it can roll about the
    // leaf's actual length axis rather than about world X.
    float roll_rot = 0;
    if (leaves_per_petiole == 1) {
        int sign = (shoot_index.x % 2 == 0) ? 1 : -1;
        roll_rot = -leaf_roll_angle * sign;
    } else if (ind_from_tip != 0) {
        roll_rot = (asin_safe(petiole_tip_axis.z) + leaf_roll_angle) * compound_rotation / std::fabs(compound_rotation);
    }
    leaf_rotation.at(petiole_index).at(leaf_index).roll = leaf_roll_angle;
    context_ptr->rotateObject(objID_leaf, roll_rot, "x");

    // leaf pitch rotation
    leaf_rotation.at(petiole_index).at(leaf_index).pitch = leaf_pitch_angle;
    float pitch_rot = leaf_pitch_angle;
    if (ind_from_tip == 0) {
        pitch_rot += asin_safe(petiole_tip_axis.z);
    }
    context_ptr->rotateObject(objID_leaf, -pitch_rot, "y");

    // leaf yaw rotation
    if (ind_from_tip != 0) {
        leaf_rotation.at(petiole_index).at(leaf_index).yaw = leaf_yaw_angle;
        context_ptr->rotateObject(objID_leaf, leaf_yaw_angle, "z");
    } else {
        leaf_rotation.at(petiole_index).at(leaf_index).yaw = 0;
    }

    // rotate leaf to azimuth of petiole
    context_ptr->rotateObject(objID_leaf, std::atan2(petiole_tip_axis.y, petiole_tip_axis.x) + compound_rotation, "z");

    // Curvature-aware blade-up correction: after the pitch-Y / yaw-Z chain, the
    // leaf's blade normal lies in the vertical plane containing the petiole's
    // length, but tilted from world-up by the angle between the petiole and
    // world-up (= asin(petiole_tip.z)). Roll the leaf around its own length axis
    // (= petiole_tip_axis in world space) to bring the blade normal back toward
    // vertical. The amount of correction is scaled by petiole_length / leaf_size:
    //   - long petioles (leaf held far from stem) → full correction (the leaf
    //     can hang at its natural angle independent of stem curvature)
    //   - short petioles (leaf hugs the stem) → little to no correction (the
    //     leaf orientation is dictated by the stem)
    // The total correction is also clamped to <90° to avoid extreme rolls when
    // the stem is heavily curved.
    if (leaves_per_petiole == 1) {
        int sign = (shoot_index.x % 2 == 0) ? 1 : -1;
        const float r_h = sqrtf(petiole_tip_axis.x * petiole_tip_axis.x + petiole_tip_axis.y * petiole_tip_axis.y);
        if (r_h > 1e-4f) {
            float blade_correction = std::atan2(petiole_tip_axis.z * r_h, r_h * r_h);
            const float petiole_len = petiole_length.at(petiole_index);
            const float leaf_size_ref = std::max(leaf_size_max.at(petiole_index).at(leaf_index), 1e-6f);
            const float length_ratio = std::min(petiole_len / leaf_size_ref, 1.f);
            blade_correction *= length_ratio;
            const float max_correction = 0.5f * PI_F - deg2rad(1.f);
            if (blade_correction > max_correction)
                blade_correction = max_correction;
            if (blade_correction < -max_correction)
                blade_correction = -max_correction;
            context_ptr->rotateObject(objID_leaf, blade_correction * static_cast<float>(sign), petiole_tip_axis);
        }
    }
}

void Phytomer::deformLeafUnderSelfWeight(uint petiole_index, uint leaf_index) {

    if (leaf_flexibility <= 0.f) {
        // The species does not droop. This is the common case and must cost nothing beyond the check.
        return;
    }

    // The blade goes on bending after it has stopped growing: it loses the straightening that comes with growth while its tissue keeps stiffening and shrinking irreversibly. That is expressed here as the
    // leaf becoming steadily more compliant over its lifetime, so a leaf that has finished expanding continues to arch over instead of freezing at whatever shape its size alone dictates. Without this the
    // oldest leaves on a plant are also its straightest, which is backwards.
    float flexibility = leaf_flexibility;
    const float aging_doubling_days = phytomer_parameters.leaf.prototype.flexibility_aging.val();
    if (aging_doubling_days > 0.f) {
        const float aging_ceiling = std::max(1.f, phytomer_parameters.leaf.prototype.flexibility_aging_max.val());
        // Sigmoidal rather than exponential: a blade holds its shape while its tissue is still maturing, then softens over a relatively short window once maturation completes, and stops changing again.
        // A plain exponential instead starts softening the moment the leaf appears, so it drooped the still-erect leaves of the middle canopy long before their turn.
        // Normalised so the multiplier is exactly 1 on a leaf that has just appeared, rather than starting part-way up the curve: a newly emerged blade must be as stiff as the species' base flexibility says,
        // or the ageing term silently changes the behaviour of every leaf including the youngest.
        const float t = age / aging_doubling_days;
        const float sigmoid = 1.f / (1.f + expf(-2.f * (t - 2.f)));
        const float sigmoid_at_emergence = 1.f / (1.f + expf(4.f));
        const float progress = std::clamp((sigmoid - sigmoid_at_emergence) / (1.f - sigmoid_at_emergence), 0.f, 1.f);
        flexibility *= 1.f + (aging_ceiling - 1.f) * progress;
    }

    if (petiole_index >= leaf_prototype_index.size() || leaf_index >= leaf_prototype_index.at(petiole_index).size()) {
        return;
    }
    const int prototype = leaf_prototype_index.at(petiole_index).at(leaf_index);
    if (prototype < 0) {
        return;
    }

    const uint prototype_identifier = phytomer_parameters.leaf.prototype.unique_prototype_identifier;
    auto rest_cache = plantarchitecture_ptr->unique_leaf_prototype_rest_geometry.find(prototype_identifier);
    if (rest_cache == plantarchitecture_ptr->unique_leaf_prototype_rest_geometry.end() || rest_cache->second.size() <= uint(prototype) || rest_cache->second.at(prototype).size() <= leaf_index) {
        return;
    }
    const PlantArchitecture::LeafRestGeometry &rest = rest_cache->second.at(prototype).at(leaf_index);
    if (rest.vertices.empty() || rest.subdivisions_x == 0) {
        // No usable rest lattice was recorded for this prototype, so there is nothing to deflect from.
        return;
    }

    const float current_scale = current_leaf_scale_factor.at(petiole_index) * leaf_size_max.at(petiole_index).at(leaf_index);
    const float mature_scale = leaf_size_max.at(petiole_index).at(leaf_index);
    if (mature_scale <= 0.f) {
        return;
    }

    // A leaf that has not measurably grown since it was last deflected already has the right shape, so the vertex write is skipped. Once a leaf matures its scale stops changing and it costs nothing per
    // timestep from then on, which is what keeps this affordable on a canopy where most leaves are full-grown.
    float &last_scale = leaf_last_deformed_scale.at(petiole_index).at(leaf_index);
    // Keyed on the deflection the leaf would take, which is its size and its current compliance together: an ageing leaf that has stopped growing still needs redeforming, and a key based on size alone would
    // freeze it at the shape it had when it finished expanding. The tolerance is relative to the key itself rather than to a fixed scale, so it stays meaningful as ageing carries the compliance upward.
    const float deflection_state = current_scale * flexibility;
    if (last_scale >= 0.f && std::fabs(deflection_state - last_scale) < 1e-3f * std::max(deflection_state, last_scale)) {
        return;
    }

    const uint objID_leaf = leaf_objIDs.at(petiole_index).at(leaf_index);
    if (!context_ptr->doesObjectExist(objID_leaf)) {
        return;
    }

    // The rest lattice is the prototype's own geometry, which was recorded before the leaf was scaled, rotated into the canopy and translated onto its petiole. Rather than trying to reconstruct that chain
    // of placements from the phytomer's stored angles, it is read back from the leaf object's own transformation matrix, which already holds exactly the composition that was applied to it. The blade is
    // deflected in the prototype frame and then carried through that transform, so the leaf droops about its own base and keeps the orientation the rest of the model gave it.
    //
    // How far the blade bends depends on how long it physically is, so the mechanics are run at the leaf's true current and mature lengths. That also scales the returned lattice up to that physical size,
    // which must then be divided back out: setLeafScaleFraction() has already accumulated the leaf's growth into the object transform through scaleObject(), so leaving the size in here would apply it twice
    // and shrink every leaf by its own growth fraction - pulling the blade in toward its base and leaving it visibly detached from the stem.
    //
    // Dividing by the size the mechanics applied, rather than deflecting a unit lattice directly, keeps the amount of bend and the size of the leaf independent: the shape is the one belonging to a blade of
    // this physical length, expressed at unit length for the transform to scale.
    std::vector<vec3> deformed_prototype_frame = deformLeafLattice(rest.vertices, rest.subdivisions_x, rest.subdivisions_y, current_scale, mature_scale, flexibility, phytomer_parameters.leaf.prototype.flexibility_taper.val());
    for (vec3 &vertex: deformed_prototype_frame) {
        vertex = vertex / current_scale;
    }

    float transform[16];
    context_ptr->getObjectTransformationMatrix(objID_leaf, transform);

    std::vector<vec3> deformed_global(deformed_prototype_frame.size());
    for (size_t v = 0; v < deformed_prototype_frame.size(); v++) {
        const vec3 &p = deformed_prototype_frame.at(v);
        deformed_global.at(v) = make_vec3(transform[0] * p.x + transform[1] * p.y + transform[2] * p.z + transform[3], transform[4] * p.x + transform[5] * p.y + transform[6] * p.z + transform[7],
                                          transform[8] * p.x + transform[9] * p.y + transform[10] * p.z + transform[11]);
    }

    context_ptr->setPolymeshObjectVertices(objID_leaf, deformed_global);
    last_scale = deflection_state;
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

    // Only the mature size changes here. How far through its expansion the leaf is - current_leaf_scale_factor
    // - is a separate quantity and must not move, or the invariant that a leaf's rendered size is
    // leaf_size_max * current_leaf_scale_factor is broken. See scaleLeafPrototypeScale() for what that cost.
    for (int leaf = 0; leaf < leaf_objIDs.at(petiole_index).size(); leaf++) {
        leaf_size_max.at(petiole_index).at(leaf) *= scale_factor;
        context_ptr->scaleObjectAboutPoint(leaf_objIDs.at(petiole_index).at(leaf), scale_factor * make_vec3(1, 1, 1), leaf_bases.at(petiole_index).at(leaf));
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

    // Scale the mature leaf size and the geometry that follows from it, and leave the growth fraction alone.
    // current_leaf_scale_factor used to be divided by scale_factor here, which broke the invariant that a
    // leaf's rendered size is leaf_size_max * current_leaf_scale_factor: the geometry was scaled once but the
    // product was left unchanged, so the bookkeeping overstated every leaf by scale_factor. Everything that
    // reads that product was wrong by the same amount - writePlantStructureXML(), which then wrote a leaf
    // scale that no geometry ever had, and deformLeafUnderSelfWeight(), which bent the blade as though it
    // were longer than it is. Worse, dividing pushed the fraction toward one: a species calling this with 0.5
    // on a half-grown leaf drove the fraction to exactly one, and the clamp that used to sit at the end of
    // this function then marked the leaf fully grown at half its intended size, where it stayed for the rest
    // of the simulation.
    for (int leaf = 0; leaf < leaf_objIDs.at(petiole_index).size(); leaf++) {
        leaf_size_max.at(petiole_index).at(leaf) *= scale_factor;
        context_ptr->scaleObjectAboutPoint(leaf_objIDs.at(petiole_index).at(leaf), scale_factor * make_vec3(1, 1, 1), leaf_bases.at(petiole_index).at(leaf));
    }
}

void Phytomer::scaleLeafPrototypeScale(float scale_factor) {
    for (uint petiole_index = 0; petiole_index < leaf_objIDs.size(); petiole_index++) {
        scaleLeafPrototypeScale(petiole_index, scale_factor);
    }
}

void Phytomer::scalePetioleGeometry(uint petiole_index, float target_length, float target_base_radius) {
    if (petiole_index >= petiole_length.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer::scalePetioleGeometry): Invalid petiole index " + std::to_string(petiole_index) + ".");
    }
    if (target_length <= 0.f || target_base_radius <= 0.f) {
        helios_runtime_error("ERROR (PlantArchitecture::Phytomer::scalePetioleGeometry): Target length and radius must be positive.");
    }

    // Calculate scale factors from current to target dimensions
    float current_length = petiole_length.at(petiole_index);
    float current_base_radius = petiole_radii.at(petiole_index).at(0);

    if (current_length <= 0.f || current_base_radius <= 0.f) {
        // Petiole wasn't properly initialized, just update the stored values
        petiole_length.at(petiole_index) = target_length;
        if (!petiole_radii.at(petiole_index).empty()) {
            petiole_radii.at(petiole_index).at(0) = target_base_radius;
        }
        return;
    }

    float length_scale = target_length / current_length;
    float radius_scale = target_base_radius / current_base_radius;

    // Get the petiole base position for scaling reference
    vec3 petiole_base = petiole_vertices.at(petiole_index).at(0);

    // Update stored vertices: scale positions relative to base
    for (size_t j = 0; j < petiole_vertices.at(petiole_index).size(); j++) {
        vec3 offset = petiole_vertices.at(petiole_index).at(j) - petiole_base;
        petiole_vertices.at(petiole_index).at(j) = petiole_base + offset * length_scale;
    }

    // Update stored radii: scale by radius factor
    for (size_t j = 0; j < petiole_radii.at(petiole_index).size(); j++) {
        petiole_radii.at(petiole_index).at(j) *= radius_scale;
    }

    // Update scalar length
    petiole_length.at(petiole_index) = target_length;

    // Update the Context geometry in place if it exists. The tube keeps its object ID, its
    // primitive data and its material, so none of those need to be reapplied.
    if (context_ptr->doesObjectExist(petiole_objIDs.at(petiole_index))) {
        context_ptr->setTubeNodes(petiole_objIDs.at(petiole_index), petiole_vertices.at(petiole_index));
        context_ptr->setTubeRadii(petiole_objIDs.at(petiole_index), petiole_radii.at(petiole_index));
    }

    // Translate leaf bases to maintain their relative positions along the scaled petiole
    if (petiole_index < leaf_bases.size()) {
        for (size_t leaf = 0; leaf < leaf_bases.at(petiole_index).size(); leaf++) {
            vec3 offset = leaf_bases.at(petiole_index).at(leaf) - petiole_base;
            leaf_bases.at(petiole_index).at(leaf) = petiole_base + offset * length_scale;

            // Translate the actual leaf geometry in Context
            if (petiole_index < leaf_objIDs.size() && leaf < leaf_objIDs.at(petiole_index).size()) {
                vec3 translation = offset * length_scale - offset;
                context_ptr->translateObject(leaf_objIDs.at(petiole_index).at(leaf), translation);
            }
        }
    }

    // Translate floral buds to maintain their relative positions along the scaled petiole
    if (petiole_index < floral_buds.size()) {
        for (auto &fbud: floral_buds.at(petiole_index)) {
            vec3 offset = fbud.base_position - petiole_base;
            vec3 translation = offset * length_scale - offset;
            fbud.base_position = petiole_base + offset * length_scale;

            // Translate inflorescence bases
            for (size_t i = 0; i < fbud.inflorescence_bases.size(); i++) {
                fbud.inflorescence_bases.at(i) += translation;
            }

            // Translate the actual floral geometry in Context
            for (size_t i = 0; i < fbud.inflorescence_objIDs.size(); i++) {
                context_ptr->translateObject(fbud.inflorescence_objIDs.at(i), translation);
            }
            for (size_t i = 0; i < fbud.peduncle_objIDs.size(); i++) {
                context_ptr->translateObject(fbud.peduncle_objIDs.at(i), translation);
            }
        }
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
    // These are indexed in lockstep with leaf_objIDs, so they have to be cleared with it or a later leaf would be deflected against a stale prototype.
    leaf_prototype_index.clear();
    leaf_last_deformed_scale.clear();

    if (build_context_geometry_petiole) {
        context_ptr->deleteObject(getExistingPetioleObjIDs());
        petiole_objIDs.clear();
    }
}

void Phytomer::deletePhytomer() {
    // Everything needed after the resize below has to be read out of *this* first: resizing
    // parent_shoot_ptr->phytomers destroys the shared_ptr that owns this Phytomer, so from that
    // point on *this* is a dangling pointer and any member read through it is a use-after-free.
    Shoot *shoot = parent_shoot_ptr;
    const int first_deleted_node = this->shoot_index.x;
    const int node_count = this->shoot_index.y;

    // prune the internode tube in the Context
    if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        uint tube_nodes = context_ptr->getTubeObjectNodeCount(shoot->internode_tube_objID);
        uint tube_segments = shoot->shoot_parameters.phytomer_parameters.internode.length_segments;
        uint tube_prune_index;
        if (first_deleted_node == 0) {
            tube_prune_index = 0;
        } else {
            tube_prune_index = first_deleted_node * tube_segments + 1; // note that first segment has an extra vertex
        }
        if (tube_prune_index < tube_nodes) {
            context_ptr->pruneTubeNodes(shoot->internode_tube_objID, tube_prune_index);
        }
        shoot->terminateApicalBud();
    }

    for (int node = first_deleted_node; node < node_count; node++) {
        auto &phytomer = shoot->phytomers.at(node);

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
        if (shoot->childIDs.find(node) != shoot->childIDs.end()) {
            for (auto childID: shoot->childIDs.at(node)) {
                auto child_shoot = plantarchitecture_ptr->plant_instances.at(plantID).shoot_tree.at(childID);
                if (!child_shoot->isPruned()) {
                    child_shoot->phytomers.front()->deletePhytomer();
                }
            }
        }
    }

    // delete shoot arrays
    shoot->shoot_internode_radii.resize(first_deleted_node);
    shoot->shoot_internode_vertices.resize(first_deleted_node);
    shoot->phytomers.resize(first_deleted_node); // *this* is destroyed here; use 'shoot' from now on

    // The child shoots attached at the deleted nodes were just emptied by the loop above, so drop
    // them from this shoot's child list. Without this, every recursive traversal over childIDs
    // (leaf area, child volume, node updates, XML output, carbohydrate transfer) still descends
    // into shoots that have no phytomers left. The erase happens after the loop rather than inside
    // it so the map is not mutated while it is being iterated.
    for (auto it = shoot->childIDs.begin(); it != shoot->childIDs.end();) {
        if (it->first >= first_deleted_node) {
            it = shoot->childIDs.erase(it);
        } else {
            ++it;
        }
    }

    // Pruning at node 0 deletes the internode tube object outright (Tube::pruneTubeNodes() deletes
    // the object when fewer than two nodes remain). Reset the field to the sentinel rather than
    // leaving a freed object ID behind, matching pruneGroundCollisions().
    if (!shoot->context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        shoot->internode_tube_objID = Shoot::no_internode_tube_objID;
    }

    // A shoot with no phytomers left cannot grow, so its apical meristem must be dead. The
    // terminateApicalBud() call at the top of this function runs only when the internode tube
    // object exists, which it never does when internode context geometry is disabled; advanceTime()
    // would then treat the emptied shoot as growable and dereference phytomers.back() on it.
    if (shoot->phytomers.empty()) {
        shoot->terminateApicalBud();
    }

    // set the correct node index for phytomers on this shoot
    for (const auto &phytomer: shoot->phytomers) {
        phytomer->shoot_index.y = scast<int>(shoot->phytomers.size());
    }
    shoot->current_node_number = scast<int>(shoot->phytomers.size());
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
    sugar_pool_molC = 0;
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

    float probability_min = plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_types_snapshot.at(this->shoot_type_label).vegetative_bud_break_probability_min.val();
    float probability_max = plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_types_snapshot.at(this->shoot_type_label).vegetative_bud_break_probability_max.val();
    float probability_decay = plantarchitecture_ptr->plant_instances.at(this->plantID).shoot_types_snapshot.at(this->shoot_type_label).vegetative_bud_break_probability_decay_rate.val();

    float bud_break_probability;
    if (!shoot_parameters.growth_requires_dormancy && probability_decay < 0) {
        bud_break_probability = probability_min;
    } else if (probability_decay > 0) {
        // probability maximum at apex
        bud_break_probability = std::fmax(probability_min, probability_max - probability_decay * float(this->current_node_number - node_index - 1));
    } else if (probability_decay < 0) {
        // probability maximum at base
        bud_break_probability = std::fmax(probability_min, probability_max - fabs(probability_decay) * float(node_index));
    } else {
        if (probability_decay == 0.f) {
            bud_break_probability = probability_min;
        } else {
            bud_break_probability = probability_max;
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
    } else if (plant_instances.at(plantID).shoot_types_snapshot.find(shoot_type_label) == plant_instances.at(plantID).shoot_types_snapshot.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addBaseStemShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = plant_instances.at(plantID).shoot_types_snapshot.at(shoot_type_label);
    validateShootTypes(shoot_parameters, plant_instances.at(plantID).shoot_types_snapshot);

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
    } else if (plant_instances.at(plantID).shoot_types_snapshot.find(shoot_type_label) == plant_instances.at(plantID).shoot_types_snapshot.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    auto shoot_parameters = plant_instances.at(plantID).shoot_types_snapshot.at(shoot_type_label);
    validateShootTypes(shoot_parameters, plant_instances.at(plantID).shoot_types_snapshot);

    if (shoot_tree_ptr->empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot append shoot to empty shoot. You must call addBaseStemShoot() first for each plant.");
    } else if (parent_shoot_ID >= int(shoot_tree_ptr->size())) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    } else if (current_node_number > shoot_parameters.max_nodes.val()) {
        helios_runtime_error("ERROR (PlantArchitecture::appendShoot): Cannot add shoot with " + std::to_string(current_node_number) + " nodes since the specified max node number is " + std::to_string(shoot_parameters.max_nodes.val()) + ".");
    } else if (shoot_tree_ptr->at(parent_shoot_ID)->phytomers.empty()) {
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
    } else if (plant_instances.at(plantID).shoot_types_snapshot.find(shoot_type_label) == plant_instances.at(plantID).shoot_types_snapshot.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto shoot_tree_ptr = &plant_instances.at(plantID).shoot_tree;

    if (parent_shoot_ID <= -1 || parent_shoot_ID >= shoot_tree_ptr->size()) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent with ID of " + std::to_string(parent_shoot_ID) + " does not exist.");
    } else if (shoot_tree_ptr->at(parent_shoot_ID)->phytomers.size() <= parent_node_index) {
        helios_runtime_error("ERROR (PlantArchitecture::addChildShoot): Parent shoot does not have a node " + std::to_string(parent_node_index) + ".");
    }

    // accumulate all the values that will be passed to Shoot constructor
    auto shoot_parameters = plant_instances.at(plantID).shoot_types_snapshot.at(shoot_type_label);
    validateShootTypes(shoot_parameters, plant_instances.at(plantID).shoot_types_snapshot);
    uint parent_rank = (int) shoot_tree_ptr->at(parent_shoot_ID)->rank;
    int childID = int(shoot_tree_ptr->size());

    // Calculate the position of the shoot base
    const auto parent_shoot_ptr = shoot_tree_ptr->at(parent_shoot_ID);

    vec3 shoot_base_position = parent_shoot_ptr->shoot_internode_vertices.at(parent_node_index).back();

    // Shift the shoot base position outward by the parent internode radius
    vec3 axis_vector;
    if (parent_shoot_ptr->phytomers.at(parent_node_index)->petiole_vertices.empty()) {
        // No petioles - use internode axis instead
        axis_vector = parent_shoot_ptr->phytomers.at(parent_node_index)->getInternodeAxisVector(1.f);
    } else {
        axis_vector = parent_shoot_ptr->phytomers.at(parent_node_index)->getPetioleAxisVector(0, petiole_index);
    }
    shoot_base_position += 0.9f * axis_vector * parent_shoot_ptr->phytomers.at(parent_node_index)->getInternodeRadius(1.f);

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
    } else if (plant_instances.at(plantID).shoot_types_snapshot.find(shoot_type_label) == plant_instances.at(plantID).shoot_types_snapshot.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::addEpicormicShoot): Shoot type with label of " + shoot_type_label + " does not exist.");
    }

    auto &parent_shoot = plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID);

    uint parent_node_index = 0;
    if (parent_position_fraction > 0) {
        parent_node_index = std::ceil(parent_position_fraction * float(parent_shoot->phytomers.size())) - 1;
    }

    vec3 axis_vector;
    if (plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID)->phytomers.at(parent_node_index)->petiole_vertices.empty()) {
        // No petioles - use internode axis instead
        axis_vector = plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID)->phytomers.at(parent_node_index)->getInternodeAxisVector(1.f);
    } else {
        axis_vector = plant_instances.at(plantID).shoot_tree.at(parent_shoot_ID)->phytomers.at(parent_node_index)->getPetioleAxisVector(0, 0);
    }

    //\todo Figuring out how to set this correctly to make the shoot vertical, which avoids having to write a child shoot function.
    AxisRotation base_rotation = make_AxisRotation(0, acos_safe(axis_vector.z), 0);

    return addChildShoot(plantID, parent_shoot_ID, parent_node_index, current_node_number, base_rotation, internode_radius, internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper, shoot_type_label, 0);
}

void PlantArchitecture::validateShootTypes(ShootParameters &shoot_parameters, const std::map<std::string, ShootParameters> &shoot_types_ref) const {
    assert(shoot_parameters.child_shoot_type_probabilities.size() == shoot_parameters.child_shoot_type_labels.size());

    for (int ind = shoot_parameters.child_shoot_type_labels.size() - 1; ind >= 0; ind--) {
        if (shoot_types_ref.find(shoot_parameters.child_shoot_type_labels.at(ind)) == shoot_types_ref.end()) {
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
                        if (state == BUD_FRUITING) {
                            // Initialize the fruit at 25% scale so it grows in gradually, matching the behavior of
                            // setFloralBudState(). Without this, current_fruit_scale_factor remains at its default of 1.0
                            // and the fruit/panicle is created at full size, then abruptly snaps down to 25% on the first
                            // fruit-growth step before regrowing.
                            current_shoot_ptr->phytomers.back()->setInflorescenceScaleFraction(fbud, 0.25f);
                        }
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
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::enableEpicormicChildShoots): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_types_snapshot.find(epicormic_shoot_type_label) == plant_instances.at(plantID).shoot_types_snapshot.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::enableEpicormicChildShoots): Shoot type with label of " + epicormic_shoot_type_label + " does not exist.");
    } else if (epicormic_probability_perlength_perday < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::enableEpicormicChildShoots): Epicormic probability must be greater than or equal to zero.");
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

    float leaf_area = phytomer->downstream_leaf_area;

    if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
        context_ptr->setObjectData(shoot->internode_tube_objID, "leaf_area", leaf_area);
    }
    float phytomer_age = phytomer->age;
    float girth_area_factor = shoot->shoot_parameters.girth_area_factor.val();
    if (phytomer_age > 365) {
        girth_area_factor = shoot->shoot_parameters.girth_area_factor.val() * 365 / phytomer_age;
    }

    // The pipe model sizes an internode from the leaf area it supports, which leaves a terminal inflorescence out of the account entirely: a sorghum panicle is borne above every node on the culm but adds
    // nothing to the leaf area, so the upper culm tapered to a point far thinner than the head it carries. Counting the inflorescence alongside the leaves restores the taper the panicle's own load implies.
    const float supported_area = leaf_area + shoot->sumDownstreamInflorescenceArea(node_number);

    float internode_area = girth_area_factor * supported_area * 1e-4;
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

    // The peduncle is built once, when the terminal floral bud appears, which is the moment the culm is thinnest -- it then keeps thickening for the rest of the season while the peduncle stays as it was.
    // Re-matching it here keeps the junction continuous for the whole growth period rather than only on the day it was created.
    //
    // Deliberately not gated on update_context_geometry: the growth loop passes false there and flushes the internode tube separately through Shoot::updateShootNodes(), which does not touch peduncles. The
    // peduncle is its own Context object, so if this were skipped whenever that flag is false it would never be updated during ordinary growth at all.
    phytomer->updatePeduncleRadii();
}

void PlantArchitecture::pruneGroundCollisions(uint plantID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneGroundCollisions): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    for (auto &shoot: plant_instances.at(plantID).shoot_tree) {
        // A shoot pruned earlier in this loop, or by an earlier call, has nothing left to clip.
        if (shoot->isPruned()) {
            continue;
        }

        // internode
        if (shoot->rank > 0 && context_ptr->doesObjectExist(shoot->internode_tube_objID) && detectGroundCollision(shoot->internode_tube_objID)) {
            // Delete the whole shoot rather than just its internode tube. Deleting the tube on its own
            // left the shoot's phytomers in shoot_tree with no geometry behind them, so leaf area,
            // carbohydrate transfer and writePlantStructureXML() all still counted organs the Context no
            // longer had. deletePhytomer() at node 0 removes the tube, the phytomers and any child shoots
            // together, and terminates the apical bud, leaving the two views consistent.
            shoot->phytomers.front()->deletePhytomer();
            continue;
        }

        for (auto &phytomer: shoot->phytomers) {
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

    // First calculate total size needed to avoid reallocations. leaf_bases is emptied when a leaf is
    // removed (see Phytomer::removeLeaf), so front() must not be called without checking for that.
    size_t total_size = 0;
    for (const auto &shoot: plant_instances.at(plantID).shoot_tree) {
        for (const auto &phytomer: shoot->phytomers) {
            if (!phytomer->leaf_bases.empty()) {
                total_size += phytomer->leaf_bases.size() * phytomer->leaf_bases.front().size();
            }
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

void PlantArchitecture::getPlantLeafObjectIDsAndBases(const std::vector<uint> &plantIDs, std::vector<uint> &leaf_objIDs, std::vector<vec3> &leaf_bases) const {
    leaf_objIDs.clear();
    leaf_bases.clear();

    for (const uint plantID: plantIDs) {
        if (plant_instances.find(plantID) == plant_instances.end()) {
            helios_runtime_error("ERROR (PlantArchitecture::getPlantLeafObjectIDsAndBases): Plant with ID of " + std::to_string(plantID) + " does not exist.");
        }

        for (const auto &shoot: plant_instances.at(plantID).shoot_tree) {
            for (const auto &phytomer: shoot->phytomers) {
                // leaf_objIDs and leaf_bases are maintained in lockstep (see Phytomer::removeLeaf),
                // so walking them together keeps each object ID paired with its own base position.
                assert(phytomer->leaf_objIDs.size() == phytomer->leaf_bases.size());
                for (uint petiole_index = 0; petiole_index < phytomer->leaf_objIDs.size(); petiole_index++) {
                    const std::vector<uint> &petiole_leaf_objIDs = phytomer->leaf_objIDs.at(petiole_index);
                    const std::vector<vec3> &petiole_leaf_bases = phytomer->leaf_bases.at(petiole_index);
                    assert(petiole_leaf_objIDs.size() == petiole_leaf_bases.size());
                    for (uint leaf_index = 0; leaf_index < petiole_leaf_objIDs.size(); leaf_index++) {
                        leaf_objIDs.push_back(petiole_leaf_objIDs.at(leaf_index));
                        leaf_bases.push_back(petiole_leaf_bases.at(leaf_index));
                    }
                }
            }
        }
    }
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

std::string PlantArchitecture::determinePhenologyStage(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::determinePhenologyStage): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    // Check if plant is dormant
    if (isPlantDormant(plantID)) {
        return "dormant";
    }

    // Check if plant has flowers or fruits (reproductive stage)
    std::vector<uint> flowers = getPlantFlowerObjectIDs(plantID);
    std::vector<uint> fruits = getPlantFruitObjectIDs(plantID);
    if (!flowers.empty() || !fruits.empty()) {
        return "reproductive";
    }

    // Check if plant is approaching senescence
    const auto &plant_instance = plant_instances.at(plantID);
    if (plant_instance.dd_to_dormancy > 0) {
        float senescence_threshold = plant_instance.dd_to_dormancy_break + plant_instance.dd_to_dormancy * 0.9f;
        if (plant_instance.time_since_dormancy > senescence_threshold) {
            return "senescent";
        }
    }

    // Default to vegetative stage
    return "vegetative";
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

void PlantArchitecture::setPlantMaxAge(uint plantID, float max_age) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantMaxAge): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (max_age < 0) {
        helios_runtime_error("ERROR (PlantArchitecture::setPlantMaxAge): Maximum age must be greater than or equal to zero.");
    }

    plant_instances.at(plantID).max_age = max_age;

    // Clear the one-shot mature-geometry latch. It is set when a plant first reaches max_age and is
    // never otherwise reset, so a plant that already froze under the old cap would skip the geometry
    // sync on reaching the new one and be left stale in the Context.
    plant_instances.at(plantID).mature_geometry_synced = false;
}

std::string PlantArchitecture::getPlantName(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantName): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    return plant_instances.at(plantID).plant_name;
}

float PlantArchitecture::getPlantAge(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantAge): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantAge): Plant with ID of " + std::to_string(plantID) + " has no shoots.");
    }
    return plant_instances.at(plantID).current_age;
}

float PlantArchitecture::getPlantMaxAge(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantMaxAge): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }
    return plant_instances.at(plantID).max_age;
}

std::vector<std::string> PlantArchitecture::listShootTypeLabels(uint plantID) const {
    // Validate plant instance exists
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::listShootTypeLabels): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    // Get reference to shoot types snapshot
    const auto &shoot_types_snap = plant_instances.at(plantID).shoot_types_snapshot;

    // Extract shoot type labels
    std::vector<std::string> labels;
    labels.reserve(shoot_types_snap.size());
    for (const auto &pair: shoot_types_snap) {
        labels.push_back(pair.first);
    }

    return labels;
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

void PlantArchitecture::removeShootFloralBuds(uint plantID, uint shootID) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::removeShootFloralBuds): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::removeShootFloralBuds): Shoot with ID of " + std::to_string(shootID) + " does not exist.");
    }

    auto &shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    for (auto &phytomer: shoot->phytomers) {
        phytomer->setFloralBudState(BUD_DEAD);
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
        if (carbon_model_enabled)
        {
            shoot->mobilizeStarch();
        }
    }
}

void PlantArchitecture::pruneBranch(uint plantID, uint shootID, uint node_index) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneBranch): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (shootID >= plant_instances.at(plantID).shoot_tree.size()) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneBranch): Shoot with ID of " + std::to_string(shootID) + " does not exist on plant " + std::to_string(plantID) + ".");
    }

    const std::shared_ptr<Shoot> &shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    // Pruning a shoot that has already been pruned away is a no-op. This is reached routinely when
    // pruning a whole branch system: removing a shoot also empties all of its descendants, and a
    // loop over shoot IDs then arrives at those descendants a second time. Rejecting them aborted
    // the loop partway through and left the plant half-pruned.
    if (shoot->isPruned()) {
        return;
    }

    if (node_index >= shoot->current_node_number) {
        helios_runtime_error("ERROR (PlantArchitecture::pruneBranch): Node index " + std::to_string(node_index) + " is out of range for shoot " + std::to_string(shootID) + ".");
    }

    shoot->phytomers.at(node_index)->deletePhytomer();

    // A shoot pruned at its base is gone entirely, so unlink it from its parent's child list. It
    // keeps its slot in the shoot_tree (and therefore its ID stays a valid index, so IDs held by
    // the caller do not shift), but nothing must be able to reach it as part of the plant anymore.
    if (shoot->isPruned() && shoot->parent_shoot_ID >= 0) {
        std::map<int, std::vector<int>> &parent_childIDs = plant_instances.at(plantID).shoot_tree.at(shoot->parent_shoot_ID)->childIDs;
        for (auto it = parent_childIDs.begin(); it != parent_childIDs.end();) {
            std::vector<int> &siblings = it->second;
            siblings.erase(std::remove(siblings.begin(), siblings.end(), scast<int>(shootID)), siblings.end());
            if (siblings.empty()) {
                it = parent_childIDs.erase(it);
            } else {
                ++it;
            }
        }
    }

    if (printmessages && shoot->parent_shoot_ID < 0 && shoot->isPruned()) {
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
    // Object IDs and base positions are consumed index-for-index below (each leaf is rotated about
    // its own base), so they must come from a single traversal. Gathering them via two independent
    // getters would leave the correspondence resting on an assert that disappears in release builds.
    std::vector<uint> objIDs;
    std::vector<vec3> bases;
    getPlantLeafObjectIDsAndBases(plantIDs, objIDs, bases);
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

std::vector<uint> PlantArchitecture::getAllShootIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getAllShootIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> shootIDs;
    shootIDs.reserve(plant_instances.at(plantID).shoot_tree.size());
    for (uint shootID = 0; shootID < plant_instances.at(plantID).shoot_tree.size(); shootID++) {
        shootIDs.push_back(shootID);
    }
    return shootIDs;
}

//! Validates a plant/shoot ID pair for the shoot topology accessors
/**
 * \param[in] plantID Plant to validate.
 * \param[in] shootID Shoot to validate within that plant.
 * \param[in] function_name Name of the calling function, used in the error message.
 */
void PlantArchitecture::validateShootID(uint plantID, uint shootID, const std::string &function_name) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.size() <= shootID) {
        helios_runtime_error("ERROR (PlantArchitecture::" + function_name + "): Shoot with ID of " + std::to_string(shootID) + " does not exist on plant " + std::to_string(plantID) + ".");
    }
}

int PlantArchitecture::getParentShootID(uint plantID, uint shootID) const {
    validateShootID(plantID, shootID, "getParentShootID");
    return plant_instances.at(plantID).shoot_tree.at(shootID)->parent_shoot_ID;
}

uint PlantArchitecture::getShootRank(uint plantID, uint shootID) const {
    validateShootID(plantID, shootID, "getShootRank");
    return plant_instances.at(plantID).shoot_tree.at(shootID)->rank;
}

uint PlantArchitecture::getShootDepth(uint plantID, uint shootID) const {
    validateShootID(plantID, shootID, "getShootDepth");

    const std::vector<std::shared_ptr<Shoot>> &shoot_tree = plant_instances.at(plantID).shoot_tree;

    uint depth = 0;
    int current_shoot_ID = shoot_tree.at(shootID)->parent_shoot_ID;
    while (current_shoot_ID >= 0) {
        depth++;
        current_shoot_ID = shoot_tree.at(current_shoot_ID)->parent_shoot_ID;
    }
    return depth;
}

std::vector<uint> PlantArchitecture::getPathToRoot(uint plantID, uint shootID) const {
    validateShootID(plantID, shootID, "getPathToRoot");

    const std::vector<std::shared_ptr<Shoot>> &shoot_tree = plant_instances.at(plantID).shoot_tree;

    std::vector<uint> path;
    int current_shoot_ID = scast<int>(shootID);
    while (current_shoot_ID >= 0) {
        path.push_back(scast<uint>(current_shoot_ID));
        current_shoot_ID = shoot_tree.at(current_shoot_ID)->parent_shoot_ID;
    }
    return path;
}

std::vector<uint> PlantArchitecture::getChildShootIDs(uint plantID, uint shootID) const {
    validateShootID(plantID, shootID, "getChildShootIDs");

    const std::vector<std::shared_ptr<Shoot>> &shoot_tree = plant_instances.at(plantID).shoot_tree;

    std::vector<uint> childIDs;
    // childIDs is keyed by the node the child attaches to, so iterating the map orders the result by
    // attachment node. pruneBranch() unlinks a pruned shoot from its parent, so pruned shoots are
    // normally absent already; the check keeps this correct regardless of how the shell arose.
    for (const auto &[node_index, node_childIDs]: shoot_tree.at(shootID)->childIDs) {
        for (const int childID: node_childIDs) {
            if (!shoot_tree.at(childID)->isPruned()) {
                childIDs.push_back(scast<uint>(childID));
            }
        }
    }
    return childIDs;
}

std::vector<uint> PlantArchitecture::getAllDescendantShootIDs(uint plantID, uint shootID) const {
    validateShootID(plantID, shootID, "getAllDescendantShootIDs");

    std::vector<uint> descendantIDs;

    // Depth-first, so each shoot is listed before its own descendants. The shoot tree is a tree rather
    // than a general graph, so no visited-set is needed to terminate.
    std::vector<uint> pending = getChildShootIDs(plantID, shootID);
    while (!pending.empty()) {
        const uint current_shoot_ID = pending.back();
        pending.pop_back();
        descendantIDs.push_back(current_shoot_ID);

        const std::vector<uint> children = getChildShootIDs(plantID, current_shoot_ID);
        for (auto child = children.rbegin(); child != children.rend(); ++child) {
            pending.push_back(*child);
        }
    }
    return descendantIDs;
}

std::vector<std::vector<uint>> PlantArchitecture::getShootIDsByRank(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootIDsByRank): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<std::vector<uint>> shootIDs_by_rank;

    for (const std::shared_ptr<Shoot> &shoot: plant_instances.at(plantID).shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        // A rank with no live shoots is kept as an empty entry so that the index into the outer vector
        // is always the rank itself.
        if (shoot->rank >= shootIDs_by_rank.size()) {
            shootIDs_by_rank.resize(shoot->rank + 1);
        }
        shootIDs_by_rank.at(shoot->rank).push_back(scast<uint>(shoot->ID));
    }
    return shootIDs_by_rank;
}

std::map<uint, std::vector<uint>> PlantArchitecture::getShootHierarchyMap(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootHierarchyMap): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::map<uint, std::vector<uint>> hierarchy;

    for (const std::shared_ptr<Shoot> &shoot: plant_instances.at(plantID).shoot_tree) {
        if (shoot->isPruned()) {
            continue;
        }
        std::vector<uint> childIDs = getChildShootIDs(plantID, scast<uint>(shoot->ID));
        if (!childIDs.empty()) {
            hierarchy[scast<uint>(shoot->ID)] = std::move(childIDs);
        }
    }
    return hierarchy;
}

std::vector<uint> PlantArchitecture::getTerminalShootIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getTerminalShootIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> terminalIDs;

    for (const std::shared_ptr<Shoot> &shoot: plant_instances.at(plantID).shoot_tree) {
        // A pruned shell has no children but is not a tip of the plant -- it is not part of the plant at all.
        if (shoot->isPruned()) {
            continue;
        }
        if (getChildShootIDs(plantID, scast<uint>(shoot->ID)).empty()) {
            terminalIDs.push_back(scast<uint>(shoot->ID));
        }
    }
    return terminalIDs;
}

bool PlantArchitecture::isShootPruned(uint plantID, uint shootID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::isShootPruned): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.size() <= shootID) {
        helios_runtime_error("ERROR (PlantArchitecture::isShootPruned): Shoot ID is out of range.");
    }
    return plant_instances.at(plantID).shoot_tree.at(shootID)->isPruned();
}

const std::shared_ptr<Shoot> &PlantArchitecture::getPlantShoot(uint plantID, uint shootID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantShoot): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.size() <= shootID) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantShoot): Shoot ID is out of range.");
    }
    return plant_instances.at(plantID).shoot_tree.at(shootID);
}

float PlantArchitecture::getShootTaper(uint plantID, uint shootID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootTaper): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    } else if (plant_instances.at(plantID).shoot_tree.size() <= shootID) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootTaper): Shoot ID is out of range.");
    }

    const std::shared_ptr<Shoot> &shoot = plant_instances.at(plantID).shoot_tree.at(shootID);

    if (shoot->shoot_internode_radii.empty() || shoot->shoot_internode_radii.front().empty()) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootTaper): Shoot " + std::to_string(shootID) + " of plant " + std::to_string(plantID) +
                             " has no internodes, so its taper is undefined. This shoot was pruned away - check PlantArchitecture::isShootPruned() before querying shoot geometry.");
    }

    float r0 = shoot->shoot_internode_radii.front().front();
    float r1 = shoot->shoot_internode_radii.back().back();

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
            std::vector<uint> petiole_objIDs_existing = phytomer->getExistingPetioleObjIDs();
            objIDs.insert(objIDs.end(), petiole_objIDs_existing.begin(), petiole_objIDs_existing.end());
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

std::vector<uint> PlantArchitecture::getAllPrototypeObjectIDs() const {
    std::vector<uint> objIDs;
    for (const auto &[key, prototype_vec] : unique_leaf_prototype_objIDs) {
        for (const auto &leaflet_vec : prototype_vec) {
            for (uint objID : leaflet_vec) {
                if (context_ptr->doesObjectExist(objID)) {
                    objIDs.push_back(objID);
                }
            }
        }
    }
    for (const auto &[key, prototype_vec] : unique_closed_flower_prototype_objIDs) {
        for (uint objID : prototype_vec) {
            if (context_ptr->doesObjectExist(objID)) {
                objIDs.push_back(objID);
            }
        }
    }
    for (const auto &[key, prototype_vec] : unique_open_flower_prototype_objIDs) {
        for (uint objID : prototype_vec) {
            if (context_ptr->doesObjectExist(objID)) {
                objIDs.push_back(objID);
            }
        }
    }
    for (const auto &[key, prototype_vec] : unique_fruit_prototype_objIDs) {
        for (uint objID : prototype_vec) {
            if (context_ptr->doesObjectExist(objID)) {
                objIDs.push_back(objID);
            }
        }
    }
    return objIDs;
}

void PlantArchitecture::deleteAllPrototypes() {
    std::vector<uint> prototype_objIDs = getAllPrototypeObjectIDs();
    for (uint objID : prototype_objIDs) {
        context_ptr->deleteObject(objID);
    }
    unique_leaf_prototype_objIDs.clear();
    unique_open_flower_prototype_objIDs.clear();
    unique_closed_flower_prototype_objIDs.clear();
    unique_fruit_prototype_objIDs.clear();
}

std::vector<uint> PlantArchitecture::getAllPlantUUIDs(uint plantID, bool include_hidden) const {
    std::vector<uint> objIDs = getAllPlantObjectIDs(plantID);
    if (include_hidden) {
        std::vector<uint> prototype_objIDs = getAllPrototypeObjectIDs();
        objIDs.insert(objIDs.end(), prototype_objIDs.begin(), prototype_objIDs.end());
    }
    return context_ptr->getObjectPrimitiveUUIDs(objIDs);
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

std::vector<uint> PlantArchitecture::getPlantInternodeObjectIDs(uint plantID, const std::string &shoot_type_label) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInternodeObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    bool shoot_type_found = false;
    for (auto &shoot: shoot_tree) {
        if (shoot->shoot_type_label == shoot_type_label) {
            shoot_type_found = true;
            if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                objIDs.push_back(shoot->internode_tube_objID);
            }
        }
    }

    if (!shoot_type_found) {
        helios_runtime_error("ERROR (PlantArchitecture::getPlantInternodeObjectIDs): No shoots with shoot type label '" + shoot_type_label + "' exist for plant with ID " + std::to_string(plantID) + ".");
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
            std::vector<uint> petiole_objIDs_existing = phytomer->getExistingPetioleObjIDs();
            objIDs.insert(objIDs.end(), petiole_objIDs_existing.begin(), petiole_objIDs_existing.end());
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
        helios_runtime_error("ERROR (PlantArchitecture::getPlantFlowerObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
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
        helios_runtime_error("ERROR (PlantArchitecture::getPlantFruitObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
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


void PlantArchitecture::updateShootFruitCounts(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::updateShootFruitCounts): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (const auto& shoot : shoot_tree) {

        int fruit_count = 0;

        for (const auto& phytomer : shoot->phytomers) {
            for (int petiole = 0; petiole < phytomer->floral_buds.size(); petiole++) {
                for (int bud = 0; bud < phytomer->floral_buds.at(petiole).size(); bud++) {
                    if (phytomer->floral_buds.at(petiole).at(bud).state == BUD_FRUITING) {
                        fruit_count++;
                    }
                }
            }
        }

        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            context_ptr->setObjectData(shoot->internode_tube_objID, "fruit_count", fruit_count);
        }
    }
}



std::vector<uint> PlantArchitecture::getShootInternodeObjectIDs(uint plantID) const {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::getShootInternodeObjectIDs): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    std::vector<uint> objIDs;

    auto &shoot_tree = plant_instances.at(plantID).shoot_tree;

    for (auto &shoot: shoot_tree) {
        // Skip pruned shoots (whose internode tube object was deleted, leaving a dangling ID) and shoots
        // whose internode geometry was never built (sentinel ID). Returning those would hand the caller
        // object IDs that don't exist in the Context.
        if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
            objIDs.push_back(shoot->internode_tube_objID);
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

    // Capture current shoot parameters to prevent contamination between plant types
    plant_instances.at(plant_count).shoot_types_snapshot = shoot_types;

    plant_count++;

    return plant_count - 1;
}

uint PlantArchitecture::duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, const AxisRotation &base_rotation, float current_age) {
    if (plant_instances.find(plantID) == plant_instances.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::duplicatePlantInstance): Plant with ID of " + std::to_string(plantID) + " does not exist.");
    }

    auto plant_shoot_tree = &plant_instances.at(plantID).shoot_tree;

    uint plantID_new = addPlantInstance(base_position, current_age);

    // Carry over the per-plant configuration that is not recoverable from the shoot structure rebuilt
    // below. addPlantInstance() gives the copy the PlantInstance defaults, so without this the duplicate
    // of a library plant silently reverted to a 999-day maximum age, the "no phenology scheduled"
    // thresholds, a "custom" name, and default carbon/nitrogen parameters -- growing quite differently
    // from the plant it was copied from. Fields deliberately excluded are handled separately: the base
    // position and current age are function arguments, the shoot tree is rebuilt below, and the
    // attraction points need translating rather than copying.
    {
        const PlantInstance &source = plant_instances.at(plantID);
        PlantInstance &copy = plant_instances.at(plantID_new);

        copy.plant_name = source.plant_name;
        copy.shoot_types_snapshot = source.shoot_types_snapshot;
        copy.epicormic_shoot_probability_perlength_per_day = source.epicormic_shoot_probability_perlength_per_day;

        // Phenological thresholds.
        copy.dd_to_dormancy_break = source.dd_to_dormancy_break;
        copy.dd_to_flower_initiation = source.dd_to_flower_initiation;
        copy.dd_to_flower_opening = source.dd_to_flower_opening;
        copy.dd_to_fruit_set = source.dd_to_fruit_set;
        copy.dd_to_fruit_maturity = source.dd_to_fruit_maturity;
        copy.dd_to_dormancy = source.dd_to_dormancy;
        copy.max_leaf_lifespan = source.max_leaf_lifespan;
        copy.is_evergreen = source.is_evergreen;

        copy.max_age = source.max_age;

        // Carbohydrate and nitrogen model configuration. The pools themselves are state rather than
        // configuration and are left at their initial values, since the copy starts at current_age.
        copy.carb_parameters = source.carb_parameters;
        copy.stem_maintenance_respiration_rate = source.stem_maintenance_respiration_rate;
        copy.root_maintenance_respiration_rate = source.root_maintenance_respiration_rate;
        copy.nitrogen_parameters = source.nitrogen_parameters;

        // Attraction points are absolute world coordinates anchored to the plant's base (the library
        // builders add base_position to each one), so they are translated onto the new base rather than
        // copied verbatim -- otherwise the duplicate would be steered toward the original's trellis.
        copy.attraction_points_enabled = source.attraction_points_enabled;
        copy.attraction_points.clear();
        copy.attraction_points.reserve(source.attraction_points.size());
        const vec3 base_position_shift = base_position - source.base_position;
        for (const vec3 &point: source.attraction_points) {
            copy.attraction_points.push_back(point + base_position_shift);
        }
        copy.attraction_cone_half_angle_rad = source.attraction_cone_half_angle_rad;
        copy.attraction_cone_height = source.attraction_cone_height;
        copy.attraction_weight = source.attraction_weight;
        copy.attraction_obstacle_reduction_factor = source.attraction_obstacle_reduction_factor;
    }

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
                    // The node at which THIS shoot attaches to its parent -- not the parent's own
                    // attachment node on the grandparent, which is what this used to read. With the
                    // latter, every branch on a plant whose shoots leave the trunk at different heights
                    // was relocated onto a single wrong node, so the copy's architecture did not match
                    // the original's.
                    uint parent_node = shoot->parent_node_index;
                    uint parent_petiole_index = 0;
                    for (auto &petiole: phytomer->axillary_vegetative_buds) {
                        shootID_new = addChildShoot(plantID_new, shoot->parent_shoot_ID, parent_node, 1, original_base_rotation, internode_radius, internode_length_max, internode_scale_factor_fraction, leaf_scale_factor_fraction, 0,
                                                    shoot->shoot_type_label, parent_petiole_index);
                        parent_petiole_index++;
                    }
                }
            } else {
                // each phytomer needs to be added one-by-one to account for possible internodes/leaves that are not fully elongated
                appendPhytomerToShoot(plantID_new, shootID_new, plant_instances.at(plantID).shoot_types_snapshot.at(shoot->shoot_type_label).phytomer_parameters, internode_radius, internode_length_max, internode_scale_factor_fraction,
                                      leaf_scale_factor_fraction);
            }
            auto phytomer_new = plant_instances.at(plantID_new).shoot_tree.at(shootID_new)->phytomers.back();
            for (uint petiole_index = 0; petiole_index < phytomer->petiole_objIDs.size(); petiole_index++) {
                phytomer_new->setLeafScaleFraction(petiole_index, phytomer->current_leaf_scale_factor.at(petiole_index));
            }
        }
    }

    // Match the source's dormancy state. Shoot::Shoot() constructs every shoot dormant, so a duplicate of
    // an actively-growing plant came back dormant and stalled until its dormancy broke, while the plant it
    // was copied from kept growing. Done after the whole tree is built, since the shoots are created
    // incrementally above.
    plant_instances.at(plantID_new).time_since_dormancy = plant_instances.at(plantID).time_since_dormancy;
    const std::vector<std::shared_ptr<Shoot>> &source_shoot_tree = plant_instances.at(plantID).shoot_tree;
    std::vector<std::shared_ptr<Shoot>> &new_shoot_tree = plant_instances.at(plantID_new).shoot_tree;
    for (uint shootID = 0; shootID < new_shoot_tree.size() && shootID < source_shoot_tree.size(); shootID++) {
        if (!source_shoot_tree.at(shootID)->isdormant && new_shoot_tree.at(shootID)->isdormant) {
            new_shoot_tree.at(shootID)->breakDormancy();
        } else if (source_shoot_tree.at(shootID)->isdormant && !new_shoot_tree.at(shootID)->isdormant) {
            new_shoot_tree.at(shootID)->makeDormant();
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

    if (plant_instances.empty()) {
        deleteAllPrototypes();
    }
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
    // Not -1, despite the symmetry with the three stages above: dd_to_fruit_maturity is not gated on
    // ">= 0.f" anywhere, it is only used as a divisor during fruit growth. See the PlantInstance
    // declaration for the full rationale. 1e6 makes the fruit simply never mature.
    plant_instances.at(plantID).dd_to_fruit_maturity = 1e6;
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
    if (progress_callback) {
        progress_bar.setCallback(progress_callback);
    }

    for (int timestep = 0; timestep < Nsteps; timestep++) {

        // Cancellation checkpoint between timesteps: a cancelled build stops the
        // growth simulation here (each timestep is self-contained — the plants are
        // simply aged less far) and falls through to progress_bar.finish() below.
        if (cancel_flag != nullptr && *cancel_flag != 0) {
            break;
        }

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
                // The plant is static once it has reached max_age, so its geometry only needs to be
                // pushed to the Context once rather than rebuilt on every subsequent timestep. Rebuilding
                // it every step was an O(timesteps) waste that dominated runtime for plants advanced well
                // past max_age (e.g. a 5000-day almond whose max_age is 1825).
                if (!plant_instance.mature_geometry_synced) {
                    shoot_tree->front()->updateShootNodes(true);
                    plant_instance.mature_geometry_synced = true;
                }
                continue;
            }

            plant_instance.current_age += dt_max_days;
            plant_instance.time_since_dormancy += dt_max_days;

            // A non-positive dormancy period means no dormancy cycle is scheduled. Without this guard the
            // predicate is satisfied on the first timestep and -- because time_since_dormancy is reset to 0
            // just below -- on every step thereafter, repeatedly stripping leaves and killing buds via
            // makeDormant(). Every library plant supplies a positive dd_to_dormancy, so this is inert for them.
            const float dormancy_period = plant_instance.dd_to_dormancy_break + plant_instance.dd_to_dormancy;
            if (dormancy_period > 0.f && plant_instance.time_since_dormancy > dormancy_period) {
                plant_instance.time_since_dormancy = 0;
                for (const auto &shoot: *shoot_tree) {
                    shoot->makeDormant();
                    shoot->phyllochron_counter = 0;
                }
                harvestPlant(plantID);
                continue;
            }

            // Refresh the cached downstream inflorescence areas that the girth calculation reads below. This is done once per plant per time step, from the root shoots only: the refresh recurses into
            // child shoots, so running it on every shoot in the tree would re-walk each subtree once per ancestor and reintroduce the very cost the cache exists to remove.
            for (const auto &shoot: *shoot_tree) {
                if (shoot->parent_shoot_ID < 0) {
                    shoot->refreshDownstreamInflorescenceArea();
                }
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
                    if (carbon_model_enabled)
                    {
                        shoot->mobilizeStarch();
                    }
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
                                // Accumulate age for any bud that has broken (past dormant/active state)
                                if (fbud.state != BUD_ACTIVE) {
                                    fbud.age += dt_max_days;
                                }
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
                                    (fbud.state == BUD_ACTIVE && plant_instance.dd_to_flower_initiation < 0.f &&
                                     (plant_instance.dd_to_flower_opening < 0.f || shoot->shoot_parameters.phytomer_parameters.inflorescence.flower_prototype_function == nullptr) && plant_instance.dd_to_fruit_set >= 0.f) ||
                                    // jumped straight to fruit set with no flowering (either flower opening disabled OR no flower prototype defined)
                                    (fbud.state == BUD_FLOWER_CLOSED && plant_instance.dd_to_flower_opening < 0.f && plant_instance.dd_to_fruit_set >= 0.f)) {
                                    // jumped from closed flower to fruit set with no flower opening
                                    if (fbud.time_counter >= plant_instance.dd_to_fruit_set) {
                                        fbud.time_counter = 0;
                                        // When skipping flowering entirely (BUD_ACTIVE -> BUD_FRUITING), apply compound probability
                                        float fruit_set_prob = shoot->shoot_parameters.fruit_set_probability.val();
                                        if (fbud.state == BUD_ACTIVE) {
                                            // Apply compound probability: flower_bud_break_probability * fruit_set_probability
                                            fruit_set_prob *= shoot->shoot_parameters.flower_bud_break_probability.val();
                                        }
                                        if (context_ptr->randu() < fruit_set_prob) {
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
                                // The leaf has stopped growing, so there is no scaling left to do - but its shape is not necessarily finished. A blade goes on bending as it ages, so it is still handed to the
                                // deflection, which returns immediately for a species whose leaves are rigid or whose shape has settled. Skipping the phytomer outright froze every mature leaf at the shape it
                                // happened to have on the step it reached full size.
                                for (uint leaf = 0; leaf < phytomer->leaf_objIDs.at(petiole_index).size(); leaf++) {
                                    phytomer->deformLeafUnderSelfWeight(petiole_index, leaf);
                                }
                                continue;
                            }

                            // Calculate petiole growth based on target petiole length (similar to internode growth)
                            // float petiole_target_length = phytomer->phytomer_parameters.petiole.length.val();
                            // float current_petiole_length = phytomer->petiole_length.at(petiole_index);
                            // float dL_petiole = dt_max_days * shoot->elongation_rate_instantaneous * petiole_target_length;
                            // float petiole_scale = fmin(1.f, (current_petiole_length + dL_petiole) / petiole_target_length);

                            // Also calculate leaf growth for proper leaf scaling
                            float tip_ind = ceil(float(phytomer->leaf_size_max.at(petiole_index).size() - 1) / 2.f);
                            float leaf_length = phytomer->current_leaf_scale_factor.at(petiole_index) * phytomer->leaf_size_max.at(petiole_index).at(tip_ind);
                            float dL_leaf = dt_max_days * shoot->elongation_rate_instantaneous * phytomer->leaf_size_max.at(petiole_index).at(tip_ind);
                            float leaf_scale = fmin(1.f, (leaf_length + dL_leaf) / phytomer->phytomer_parameters.leaf.prototype_scale.val());

                            // Expressed as a fraction of THIS leaf's own full size rather than of the shoot type's prototype_scale. current_leaf_scale_factor means "how far this leaf has grown toward its
                            // own maximum", and leaf_size_max is that maximum -- which a phytomer creation function is free to set per rank. Dividing by prototype_scale instead capped a leaf whose target
                            // is smaller than the species maximum at target/prototype_scale, so it could never finish expanding and its size was multiplied by that shortfall a second time: a blade set to
                            // 62% of the maximum came out at 62% of 62%. Every species whose leaf size is uniform along the shoot is unaffected, since there the two divisors are equal.
                            const float leaf_size_full = phytomer->leaf_size_max.at(petiole_index).at(tip_ind);
                            float scale = (leaf_size_full > 0.f) ? fmin(1.f, (leaf_length + dL_leaf) / leaf_size_full) : 1.f;
                            phytomer->phytomer_parameters.leaf.prototype_scale.resample();
                            phytomer->setLeafScaleFraction(petiole_index, scale);
                        }
                    }

                    // Fruit Growth
                    for (auto &petiole: phytomer->floral_buds) {
                        for (auto &fbud: petiole) {
                            // If the floral bud it in a 'fruiting' state, the fruit grows with time.
                            // dd_to_fruit_maturity is a divisor here, so a non-positive value would give an
                            // infinite or negative scale factor; guard rather than trust the call sites, since
                            // setPlantPhenologicalThresholds() accepts any value and readPlantStructureXML()
                            // feeds it whatever the file contains.
                            if (fbud.state == BUD_FRUITING && fbud.time_counter > 0 && plant_instance.dd_to_fruit_maturity > 0) {
                                // Save current scale for nitrogen model growth tracking
                                fbud.previous_fruit_scale_factor = fbud.current_fruit_scale_factor;
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
                                ShootParameters *new_shoot_parameters = &plant_instance.shoot_types_snapshot.at(vbud.shoot_type_label);
                                int parent_node_count = shoot->current_node_number;

                                float insertion_angle_adjustment = fmin(new_shoot_parameters->insertion_angle_tip.val() + new_shoot_parameters->insertion_angle_decay_rate.val() * float(parent_node_count - phytomer->shoot_index.x - 1), 90.f);
                                // NOTE: No additional rotation offset needed here because the child shoot orientation
                                // is already determined by the parent_petiole_axis in appendPhytomer() (line 995),
                                // which correctly uses parent_petiole_index to get the specific petiole's axis vector
                                AxisRotation base_rotation = make_AxisRotation(deg2rad(insertion_angle_adjustment), deg2rad(new_shoot_parameters->base_yaw.val()), deg2rad(new_shoot_parameters->base_roll.val()));
                                new_shoot_parameters->base_yaw.resample();
                                if (new_shoot_parameters->insertion_angle_decay_rate.val() == 0) {
                                    new_shoot_parameters->insertion_angle_tip.resample();
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
                    appendPhytomerToShoot(plantID, shoot->ID, plant_instance.shoot_types_snapshot.at(shoot->shoot_type_label).phytomer_parameters, internode_radius, internode_length_max, 0.01,
                                          0.01); //\todo These factors should be set to be consistent with the shoot
                    shoot->phyllochron_counter = shoot->phyllochron_counter - shoot->phyllochron_instantaneous;
                }

                // ****** EPICORMIC SHOOTS ****** //
                std::string epicormic_shoot_label = plant_instance.epicormic_shoot_probability_perlength_per_day.first;
                if (!epicormic_shoot_label.empty()) {
                    std::vector<float> epicormic_fraction;
                    uint Nepicormic = shoot->sampleEpicormicShoot(time_step_days, epicormic_fraction);
                    for (int s = 0; s < Nepicormic; s++) {
                        float internode_radius = plant_instance.shoot_types_snapshot.at(epicormic_shoot_label).phytomer_parameters.internode.radius_initial.val();
                        plant_instance.shoot_types_snapshot.at(epicormic_shoot_label).phytomer_parameters.internode.radius_initial.resample();
                        float internode_length_max = plant_instance.shoot_types_snapshot.at(epicormic_shoot_label).internode_length_max.val();
                        plant_instance.shoot_types_snapshot.at(epicormic_shoot_label).internode_length_max.resample();
                        addEpicormicShoot(plantID, shoot->ID, epicormic_fraction.at(s), 1, 0, internode_radius, internode_length_max, 0.01, 0.01, 0, epicormic_shoot_label);
                    }
                }
                if (carbon_model_enabled) {
                    if (output_object_data.find("carbohydrate_concentration") != output_object_data.end() && context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                        float shoot_volume = shoot->calculateShootInternodeVolume();
                        context_ptr->setObjectData(shoot->internode_tube_objID, "carbohydrate_concentration", shoot->total_carbohydrate_pool_molC / shoot_volume);
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

            // **** subtract maintenance carbon costs **** //
            if (carbon_model_enabled) {
                subtractShootMaintenanceCarbon(dt_max_days);
                subtractShootGrowthCarbon();
                checkCarbonPool_transferCarbon(dt_max_days);
                checkCarbonPool_adjustPhyllochron(dt_max_days);
                checkCarbonPool_abortOrgans(dt_max_days);
            }

            // Assign current volume as old volume for your next timestep
            for (auto &shoot: *shoot_tree) {
                // Pruned shoots are left in the shoot_tree as empty shells with a deleted internode tube
                // object. Skip them: they have no geometry to update and their internode_tube_objID is dangling.
                if (!context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                    continue;
                }
                float shoot_volume = plant_instances.at(plantID).shoot_tree.at(shoot->ID)->calculateShootInternodeVolume();
                // Find current volume for each shoot in the plant
                float volume_ratio = shoot->old_shoot_volume/shoot_volume;
                context_ptr->setObjectData(shoot->internode_tube_objID, "volume_ratio", volume_ratio);
                shoot->old_shoot_volume = shoot_volume; // Set old volume to the current volume for the next timestep
                context_ptr->setObjectData(shoot->internode_tube_objID, "old_shoot_volume", shoot_volume);
            }

            // Update plant-level dynamic object data
            std::vector<uint> plant_primitives = getAllPlantObjectIDs(plantID);
            if (!plant_primitives.empty()) {
                if (output_object_data.at("plant_height")) {
                    context_ptr->setObjectData(plant_primitives, "plant_height", getPlantHeight(plantID));
                }
                if (output_object_data.at("phenology_stage")) {
                    context_ptr->setObjectData(plant_primitives, "phenology_stage", determinePhenologyStage(plantID));
                }
            }
        }

        // **** nitrogen model operations **** //
        if (nitrogen_model_enabled) {
            accumulateLeafNitrogen(dt_max_days); // Available pool → leaf pools (rate-limited)
            remobilizeNitrogen(dt_max_days); // Old leaves → young leaves (age-based)
            removeFruitNitrogen(); // Deduct N from available pool for fruit growth
            updateNitrogenStressFactor(); // Calculate and write stress factor to object data
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

    // Update age object data once at the end for performance
    // This avoids updating age data every timestep (which would be ~100x more calls)
    if (output_object_data.at("age")) {
        for (uint plantID: plantIDs) {
            if (plant_instances.find(plantID) == plant_instances.end()) {
                continue;
            }

            auto shoot_tree = &plant_instances.at(plantID).shoot_tree;
            for (auto &shoot: *shoot_tree) {
                // Update internode age once per shoot (fixes redundancy noted in previous TODO)
                // All phytomers in a shoot share the same internode tube object
                if (shoot->build_context_geometry_internode && !shoot->phytomers.empty()) {
                    if (context_ptr->doesObjectExist(shoot->internode_tube_objID)) {
                        // Use the age of the youngest (last) phytomer as the shoot age
                        float shoot_age = shoot->phytomers.back()->age;
                        context_ptr->setObjectData(shoot->internode_tube_objID, "age", shoot_age);
                    }
                }

                // Update each phytomer's petiole, leaf, and floral bud age
                for (auto &phytomer: shoot->phytomers) {
                    if (phytomer->build_context_geometry_petiole) {
                        context_ptr->setObjectData(phytomer->getExistingPetioleObjIDs(), "age", phytomer->age);
                    }
                    context_ptr->setObjectData(phytomer->leaf_objIDs, "age", phytomer->age);
                    for (auto &petiole: phytomer->floral_buds) {
                        for (auto &fbud: petiole) {
                            if (fbud.state != BUD_DORMANT && fbud.state != BUD_ACTIVE && fbud.state != BUD_DEAD) {
                                context_ptr->setObjectData(fbud.inflorescence_objIDs, "age", fbud.age);
                                context_ptr->setObjectData(fbud.peduncle_objIDs, "age", fbud.age);
                            }
                        }
                    }
                }
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
    if (progress_callback) {
        progress_bar.setCallback(progress_callback);
    }

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
                for (uint petiole_objID: phytomer->petiole_objIDs) {
                    if (collision_set.count(petiole_objID)) {
                        phytomer->removeLeaf();
                        break; // removeLeaf() removes petiole and all leaflets
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

std::vector<uint> Phytomer::getExistingPetioleObjIDs() const {
    std::vector<uint> objIDs;
    objIDs.reserve(petiole_objIDs.size());
    for (uint objID: petiole_objIDs) {
        if (context_ptr->doesObjectExist(objID)) {
            objIDs.push_back(objID);
        }
    }
    return objIDs;
}

uint makePetioleTube(uint radial_subdivisions, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr) {
    uint Nverts = vertices.size();

    if (radii.size() != Nverts || colors.size() != Nverts) {
        helios_runtime_error("ERROR (makePetioleTube): Length of vertex vectors is not consistent.");
    }

    // A tube needs at least two nodes to have any extent.
    if (Nverts < 2) {
        return Phytomer::no_petiole_objID;
    }

    // Check if tube is too small to create geometry - check both radii and total length
    bool all_radii_too_small = true;
    for (float radius: radii) {
        if (radius >= MIN_TUBE_RADIUS_FOR_GEOMETRY) {
            all_radii_too_small = false;
            break;
        }
    }

    // Calculate total tube length
    float total_length = 0.0f;
    for (uint v = 0; v < Nverts - 1; v++) {
        total_length += (vertices.at(v + 1) - vertices.at(v)).magnitude();
    }

    if (all_radii_too_small || total_length < MIN_TUBE_LENGTH_FOR_GEOMETRY) {
        return Phytomer::no_petiole_objID;
    }

    // Drop nodes that coincide with their predecessor. Zero-length segments would produce
    // degenerate triangles, and addTubeObject does not skip them the way the previous
    // cone-by-cone construction did.
    std::vector<helios::vec3> tube_vertices;
    std::vector<float> tube_radii;
    std::vector<helios::RGBcolor> tube_colors;
    tube_vertices.reserve(Nverts);
    tube_radii.reserve(Nverts);
    tube_colors.reserve(Nverts);

    tube_vertices.push_back(vertices.front());
    tube_radii.push_back(std::max(radii.front(), MIN_TUBE_RADIUS_FOR_GEOMETRY));
    tube_colors.push_back(colors.front());

    for (uint v = 1; v < Nverts; v++) {
        if ((vertices.at(v) - tube_vertices.back()).magnitude() < 1e-6f) {
            continue;
        }
        tube_vertices.push_back(vertices.at(v));
        tube_radii.push_back(std::max(radii.at(v), MIN_TUBE_RADIUS_FOR_GEOMETRY));
        tube_colors.push_back(colors.at(v));
    }

    if (tube_vertices.size() < 2) {
        return Phytomer::no_petiole_objID;
    }

    return context_ptr->addTubeObject(radial_subdivisions, tube_vertices, tube_radii, tube_colors);
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
    // Convert label to lowercase for case-insensitive comparison
    std::string label_lower = object_data_label;
    std::transform(label_lower.begin(), label_lower.end(), label_lower.begin(), ::tolower);

    // Check if "all" was requested
    if (label_lower == "all") {
        // Enable all optional output object data
        for (auto &item: output_object_data) {
            item.second = true;
        }
        return;
    }

    // Check if the label is valid
    if (output_object_data.find(object_data_label) == output_object_data.end()) {
        helios_runtime_error("ERROR (PlantArchitecture::optionalOutputObjectData): Output object data of '" + object_data_label + "' is not a valid option.");
    }

    output_object_data.at(object_data_label) = true;
}

void PlantArchitecture::optionalOutputObjectData(const std::vector<std::string> &object_data_labels) {
    for (const auto &label: object_data_labels) {
        // Call the single-string overload which handles "all" and error checking
        optionalOutputObjectData(label);
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
