/** \file "PlantArchitecture.h" Primary header file for plant architecture plug-in.

    Copyright (C) 2016-2023 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef PLANT_ARCHITECTURE
#define PLANT_ARCHITECTURE

#include "Context.h"
#include <random>
#include <utility>
#include "Assets.h"


struct RandomParameter_float {
public:

    explicit RandomParameter_float(){
        constval = 1.f;
        distribution = "constant";
        generator = nullptr;
        sampled = false;
    }

    void initialize( float a_val, std::minstd_rand0 *rand_generator){
        constval = a_val;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    void initialize( std::minstd_rand0 *rand_generator){
        constval = 1.f;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    RandomParameter_float& operator=(float a){
        this->distribution = "constant";
        this->constval = a;
        this->sampled = false;
        return *this;
    }

    void uniformDistribution( float minval, float maxval ){
        distribution = "uniform";
        distribution_parameters = {minval, maxval};
        sampled = false;
    }

    void normalDistribution( float mean, float std_dev ){
        distribution = "normal";
        distribution_parameters = {mean, std_dev};
        sampled = false;
    }

    void weibullDistribution( float shape, float scale ){
        distribution = "weibull";
        distribution_parameters = {shape, scale};
        sampled = false;
    }

    float val(){
        if( !sampled ){
            constval = resample();
        }
        return constval;
    }

    float resample(){
        sampled = true;
        if( distribution!="constant" ) {
            if (generator == nullptr) {
                throw (std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            if (distribution == "uniform") {
                std::uniform_real_distribution<float> unif_distribution;
                constval = distribution_parameters.at(0) + unif_distribution(*generator) * (distribution_parameters.at(1) - distribution_parameters.at(0));
            } else if (distribution == "normal") {
                std::normal_distribution<float> norm_distribution(distribution_parameters.at(0),distribution_parameters.at(1));
                constval = norm_distribution(*generator);
            } else if (distribution == "weibull") {
                std::weibull_distribution<float> wbull_distribution(distribution_parameters.at(0),distribution_parameters.at(1));
                constval = wbull_distribution(*generator);
            }
        }
        return constval;
    }

    std::string distribution;
    std::vector<float> distribution_parameters;

private:
    bool sampled;
    float constval;


    std::minstd_rand0 *generator;
};

struct RandomParameter_int {
public:

    explicit RandomParameter_int(){
        constval = 1;
        distribution = "constant";
        generator = nullptr;
        sampled = false;
    }

    void initialize(int a_val, std::minstd_rand0 *rand_generator){
        constval = a_val;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    void initialize( std::minstd_rand0 *rand_generator){
        constval = 1;
        distribution = "constant";
        generator = rand_generator;
        sampled = false;
    }

    RandomParameter_int& operator=(int a){
        this->distribution = "constant";
        this->constval = a;
        this->sampled = false;
        return *this;
    }

    void uniformDistribution( int minval, int maxval ){
        distribution = "uniform";
        distribution_parameters = {minval, maxval};
        sampled = false;
    }

    int val(){
        if( !sampled ){
            constval = resample();
        }
        return constval;
    }

    int resample(){
        sampled = true;
        if( distribution!="constant" ) {
            if (generator == nullptr) {
                throw (std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            if (distribution == "uniform") {
                std::uniform_int_distribution<> unif_distribution(distribution_parameters.at(0),distribution_parameters.at(1));
                constval = unif_distribution(*generator);
            }
        }
        return constval;
    }

private:
    bool sampled;
    int constval;
    std::string distribution;
    std::vector<int> distribution_parameters;
    std::minstd_rand0 *generator;
};

struct AxisRotation{
public:

    AxisRotation(){
        pitch = 0;
        yaw = 0;
        roll = 0;
    }

    AxisRotation( float a_pitch, float a_yaw, float a_roll ){
        pitch = a_pitch;
        yaw = a_yaw;
        roll = a_roll;
    }

    float pitch;
    float yaw;
    float roll;

    AxisRotation operator+(const AxisRotation& a) const;
    AxisRotation operator-(const AxisRotation& a) const;

};

inline AxisRotation make_AxisRotation( float a_pitch, float a_yaw, float a_roll ) {
    return {a_pitch,a_yaw,a_roll};
}

inline AxisRotation AxisRotation::operator+(const AxisRotation& a) const{
    return {a.pitch+pitch, a.yaw+yaw, a.roll+roll};
}

inline AxisRotation AxisRotation::operator-(const AxisRotation& a) const{
    return {a.pitch-pitch, a.yaw-yaw, a.roll-roll};
}

enum BudState{
    BUD_DORMANT = 0,
    BUD_ACTIVE = 1,
    BUD_FLOWERING = 2,
    BUD_FRUITING = 3,
    BUD_DEAD = 4
};

//! Add geometry to the Context consisting of a series of Cone objects to form a tube-like shape
/**
 * \param[in] radial_subdivisions Number of subdivisions around the circumference of each cone (must be be >= 3).
 * \param[in] vertices (x,y,z) Cartesian coordinates of vertices forming the centerline of the tube.
 * \param[in] radii Radius of the tube at each specified vertex.
 * \param[in] colors Color of the tube at each specified vertex.
 * \param[in] context_ptr Pointer to the Helios context.
 * \return Vector of Object IDs of the cones forming the tube.
 */
std::vector<uint> makeTubeFromCones(uint radial_subdivisions, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr);

struct PhytomerParameters{
private:

    struct InternodeParameters{
        RandomParameter_float pitch;
        float radius;
        RandomParameter_float length;
        uint petioles_per_internode;
        helios::RGBcolor color;
        uint length_segments;
        uint radial_subdivisions;
        InternodeParameters& operator=(const InternodeParameters &a){
            this->pitch = a.pitch;
            this->pitch.resample();
            this->radius = a.radius;
            this->length = a.length;
            this->length.resample();
            this->petioles_per_internode = a.petioles_per_internode;
            this->color = a.color;
            this->length_segments = a.length_segments;
            this->radial_subdivisions = a.radial_subdivisions;
            return *this;
        }
    };

    struct PetioleParameters{
        RandomParameter_float pitch;
        RandomParameter_float radius;
        RandomParameter_float length;
        RandomParameter_float curvature;
        RandomParameter_float taper;
        uint length_segments;
        uint radial_subdivisions;
        int leaves_per_petiole;
        PetioleParameters& operator=(const PetioleParameters &a){
            this->pitch = a.pitch;
            this->pitch.resample();
            this->radius = a.radius;
            this->radius.resample();
            this->length = a.length;
            this->length.resample();
            this->curvature = a.curvature;
            this->curvature.resample();
            this->taper = a.taper;
            this->taper.resample();
            this->length_segments = a.length_segments;
            this->radial_subdivisions = a.radial_subdivisions;
            this->leaves_per_petiole = a.leaves_per_petiole;
            return *this;
        }
    };

    struct LeafParameters{
        RandomParameter_float pitch;
        RandomParameter_float yaw;
        RandomParameter_float roll;
        RandomParameter_float leaflet_offset;
        RandomParameter_float leaflet_scale;
        RandomParameter_float prototype_scale;
        uint(*prototype_function)( helios::Context*, uint subdivisions, int flag ) = nullptr;
        LeafParameters& operator=(const LeafParameters &a){
            this->pitch = a.pitch;
            this->pitch.resample();
            this->yaw = a.yaw;
            this->yaw.resample();
            this->roll = a.roll;
            this->roll.resample();
            this->leaflet_offset = a.leaflet_offset;
            this->leaflet_offset.resample();
            this->leaflet_scale = a.leaflet_scale;
            this->leaflet_scale.resample();
            this->prototype_scale = a.prototype_scale;
            this->prototype_scale.resample();
            this->prototype_function = a.prototype_function;
            return *this;
        }
    };

    struct InflorescenceParameters{
        RandomParameter_float length;
        RandomParameter_int fruit_per_inflorescence;
        RandomParameter_float fruit_offset;
        RandomParameter_float rachis_radius;
        RandomParameter_float curvature;
        std::string fruit_arrangement_pattern;
        uint length_segments;
        uint radial_subdivisions;
        RandomParameter_float fruit_pitch;
        RandomParameter_float fruit_roll;
        RandomParameter_float fruit_prototype_scale;
        RandomParameter_float flower_prototype_scale;
        uint(*fruit_prototype_function)( helios::Context*, uint subdivisions, int flag ) = nullptr;
        uint(*flower_prototype_function)( helios::Context*, uint subdivisions, int flag ) = nullptr;
        InflorescenceParameters& operator=(const InflorescenceParameters &a){
            this->length = a.length;
            this->length.resample();
            this->fruit_per_inflorescence = a.fruit_per_inflorescence;
            this->fruit_per_inflorescence.resample();
            this->fruit_offset = a.fruit_offset;
            this->fruit_offset.resample();
            this->rachis_radius = a.rachis_radius;
            this->rachis_radius.resample();
            this->curvature = a.curvature;
            this->curvature.resample();
            this->fruit_arrangement_pattern = a.fruit_arrangement_pattern;
            this->length_segments = a.length_segments;
            this->radial_subdivisions = a.radial_subdivisions;
            this->fruit_pitch = a.fruit_pitch;
            this->fruit_pitch.resample();
            this->fruit_roll = a.fruit_roll;
            this->fruit_roll.resample();
            this->fruit_prototype_scale = a.fruit_prototype_scale;
            this->fruit_prototype_scale.resample();
            this->fruit_prototype_function = a.fruit_prototype_function;
            this->flower_prototype_scale = a.flower_prototype_scale;
            this->flower_prototype_scale.resample();
            this->flower_prototype_function = a.flower_prototype_function;
            return *this;
        }
    };

protected:

//    bool is_initialized = false;
//    helios::vec3 internode_base_position;
//    float internode_radius;
//    float internode_length;

public:

    InternodeParameters internode;

    PetioleParameters petiole;

    LeafParameters leaf;

    InflorescenceParameters inflorescence;

    //! Default constructor - does not set random number generator
    PhytomerParameters();

    //! Constructor - sets random number generator
    explicit PhytomerParameters( std::minstd_rand0 *generator );

    //! Copy constructor
    PhytomerParameters( const PhytomerParameters& parameters_copy );

    friend class PlantArchitecture;
    friend class Shoot;

};

struct ShootParameters{

    //! Default constructor - does not set random number generator
    ShootParameters();

    //! Constructor - sets random number generator
    explicit ShootParameters( std::minstd_rand0 *generator );

    PhytomerParameters phytomer_parameters;

    uint max_nodes;

    RandomParameter_float phyllotactic_angle;

    RandomParameter_float shoot_internode_taper;

    RandomParameter_float phyllochron; //phytomers/day
    RandomParameter_float elongation_rate; //length/day
    RandomParameter_float girth_growth_rate; //length/day

    RandomParameter_float gravitropic_curvature;  //degrees/length

    // Probability that bud with this shoot type will break and form a new shoot
    float bud_break_probability;

    // Probability that a phytomer will flower
    float flower_probability;

    // Probability that a flower will set fruit
    float fruit_set_probability;

    RandomParameter_float bud_time;  //days

    RandomParameter_float child_insertion_angle_tip;
    RandomParameter_float child_insertion_angle_decay_rate;

    RandomParameter_float child_internode_length_max;
    RandomParameter_float child_internode_length_min;
    RandomParameter_float child_internode_length_decay_rate;

    bool flowers_require_dormancy;
    bool growth_requires_dormancy;

    void defineChildShootTypes( const std::vector<std::string> &child_shoot_type_labels, const std::vector<float> &child_shoot_type_probabilities );

    //\todo These should be private
    std::vector<std::string> child_shoot_type_labels;
    std::vector<float> child_shoot_type_probabilities;

    ShootParameters& operator=(const ShootParameters &a) {
        this->phytomer_parameters = a.phytomer_parameters;
        this->max_nodes = a.max_nodes;
        this->phyllotactic_angle = a.phyllotactic_angle;
        this->phyllotactic_angle.resample();
        this->shoot_internode_taper = a.shoot_internode_taper;
        this->shoot_internode_taper.resample();
        this->phyllochron = a.phyllochron;
        this->phyllochron.resample();
        this->elongation_rate = a.elongation_rate;
        this->elongation_rate.resample();
        this->girth_growth_rate = a.girth_growth_rate;
        this->girth_growth_rate.resample();
        this->gravitropic_curvature = a.gravitropic_curvature;
        this->gravitropic_curvature.resample();
        this->bud_break_probability = a.bud_break_probability;
        this->flower_probability = a.flower_probability;
        this->fruit_set_probability = a.fruit_set_probability;
        this->bud_time = a.bud_time;
        this->bud_time.resample();
        this->child_insertion_angle_tip = a.child_insertion_angle_tip;
        this->child_insertion_angle_tip.resample();
        this->child_insertion_angle_decay_rate = a.child_insertion_angle_decay_rate;
        this->child_insertion_angle_decay_rate.resample();
        this->child_internode_length_max = a.child_internode_length_max;
        this->child_internode_length_max.resample();
        this->child_internode_length_min = a.child_internode_length_min;
        this->child_internode_length_min.resample();
        this->child_internode_length_decay_rate = a.child_internode_length_decay_rate;
        this->child_internode_length_decay_rate.resample();
        this->flowers_require_dormancy = a.flowers_require_dormancy;
        this->growth_requires_dormancy = a.growth_requires_dormancy;
        this->child_shoot_type_labels = a.child_shoot_type_labels;
        this->child_shoot_type_probabilities = a.child_shoot_type_probabilities;
        return *this;
    }

private:


};

struct Phytomer {
public:

    Phytomer(const PhytomerParameters &params, ShootParameters &parent_shoot_parameters, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, helios::vec3 internode_base_origin,
             const AxisRotation &shoot_base_rotation, float internode_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, helios::Context *context_ptr);

    helios::vec3 getInternodeAxisVector( float stem_fraction ) const;

    helios::vec3 getPetioleAxisVector( float stem_fraction ) const;

    helios::vec3 getAxisVector( float stem_fraction, const std::vector<helios::vec3> &axis_vertices ) const;

    float getInternodeLength() const;

    float getPetioleLength() const;

    float getInternodeRadius( float stem_fraction ) const;

    float getPetioleRadius( float stem_fraction ) const;

    void addInflorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation, const helios::vec3 &a_inflorescence_bending_axis);

    void setInternodeScale( float internode_scale_factor_fraction );

    void setLeafScale( float leaf_scale_factor_fraction );

    void setInflorescenceScale( float inflorescence_scale_factor_fraction );

    void setPetioleBase( const helios::vec3 &base_position );

    void setPhytomerBase( const helios::vec3 &base_position );

    void changeReproductiveState( BudState state );

    void removeLeaf();

    void removeInflorescence();

    bool hasLeaf() const;

    bool hasInflorescence() const;

    std::vector<helios::vec3> internode_vertices;
    std::vector<helios::vec3> petiole_vertices; //\todo this needs to be a multidimensional array for the case in which we have multiple buds per phytomer
    std::vector<helios::vec3> leaf_bases;
    std::vector<helios::vec3> inflorescence_bases;
    float internode_length;

    std::vector<float> internode_radii;
    std::vector<float> petiole_radii;
    float petiole_length;

    std::vector<helios::RGBcolor> internode_colors;
    std::vector<helios::RGBcolor> petiole_colors;

    std::vector<uint> internode_objIDs;
    std::vector<std::vector<uint> > petiole_objIDs;
    std::vector<uint> leaf_objIDs;
    std::vector<uint> inflorescence_objIDs;
    std::vector<uint> rachis_objIDs;

    PhytomerParameters phytomer_parameters;

    uint rank;

    float age = 0;
    float time_since_dormancy = 0;
    float time_since_flowering = 0;

    float current_internode_scale_factor = 1;
    float current_leaf_scale_factor = 1;
    float current_inflorescence_scale_factor = 1;

    BudState flower_bud_state = BUD_DORMANT;
    BudState vegetative_bud_state = BUD_DORMANT;

    float petiole_yaw = 0;

private:

    helios::vec3 inflorescence_bending_axis;

    helios::Context *context_ptr;

};

struct Shoot{

    Shoot(int ID, int parentID, uint parent_node, uint rank, const helios::vec3 &origin, const AxisRotation &shoot_base_rotation, uint current_node_number, ShootParameters shoot_params, const std::string &shoot_type_label,
          std::vector<std::shared_ptr<Shoot> > *shoot_tree_ptr, helios::Context *context_ptr);

    void buildShootPhytomers();

    int addPhytomer(const PhytomerParameters &params, const helios::vec3 internode_base_position, const AxisRotation &shoot_base_rotation, float internode_scale_factor_fraction, float leaf_scale_factor_fraction);

    uint current_node_number;

    helios::vec3 origin;

    AxisRotation base_rotation;

    int ID;
    int parentID;
    uint parentNode;
    uint rank;

    float assimilate_pool;  // mg SC/g DW

    void breakDormancy();
    void makeDormant();

    bool dormant;
    uint dormancy_cycles = 0;

    bool meristem_is_alive = true;

    float phyllochron_counter = 0;

    //map of node number to ID of shoot child
    std::map<int,int> childIDs;

    ShootParameters shoot_parameters;

    std::string shoot_type_label;

    std::vector<std::shared_ptr<Phytomer> > phytomers;

    std::vector<std::shared_ptr<Shoot> > *shoot_tree_ptr;

    helios::Context *context_ptr;

};

struct PlantInstance{

    PlantInstance(const helios::vec3 &a_base_position, float a_current_age) : base_position(a_base_position), current_age(a_current_age) {}
    std::vector<std::shared_ptr<Shoot> > shoot_tree;
    helios::vec3 base_position;
    float current_age;

    //Phenological thresholds
    float dd_to_dormancy_break = 0;
    float dd_to_flowering = 0;
    float dd_to_fruit_set = 0;
    float dd_to_fruit_maturity = 0;
    float dd_to_senescence = 0;

};

class PlantArchitecture{
public:

    //! Main architectural model class constructor
    /**
     * \param[in] context_ptr Pointer to the Helios context.
     */
    explicit PlantArchitecture( helios::Context* context_ptr );

    //! Create an instance of a plant
    /** This is the first step of the plant building process. It creates an empty plant instance that can be built up manually, or using a pre-defined plant type in the library.
     * \param[in] base_position Cartesian coordinates of the base of the plant.
     * \param[in] current_age Age of the plant in days.
     * \return ID of the plant instance.
     */
    uint addPlantInstance(const helios::vec3 &base_position, float current_age);

    //! Duplicate an existing plant instance and specify its base position and age
    /**
     * \param[in] plantID ID of the existing plant instance to be duplicated.
     * \param[in] base_position Cartesian coordinates of the base of the new plant copy.
     * \param[in] current_age Age of the new plant copy in days.
     * \return ID of the new plant instance.
     */
    uint duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, float current_age );

    PhytomerParameters getPhytomerParametersFromLibrary(const std::string &phytomer_label );

    //! Specify the threshold values for plant phenological stages
    /**
     * \param[in] plantID ID of the plant.
     * \param[in] dd_to_dormancy_break Minimum assimilate pool (mg SC/g DW) required to break dormancy.
     * \param[in] dd_to_flowering Degree-days from emergence/dormancy required to reach flowering.
     * \param[in] dd_to_fruit_set Degree-days from flowering date required to reach fruit set (i.e., flower dies and fruit is created).
     * \param[in] dd_to_fruit_maturity Degree-days from fruit set date required to reach fruit maturity.
     * \param[in] dd_to_senescence Degree-days from emergence/dormancy required to reach senescence.
     */
    void setPlantPhenologicalThresholds(uint plantID, float dd_to_dormancy_break, float dd_to_flowering, float dd_to_fruit_set, float dd_to_fruit_maturity, float dd_to_senescence);

    //! Advance plant growth by a specified time interval
    /**
     * \param[in] dt Time interval in days.
     */
    void advanceTime( float dt );

    void incrementAssimilatePool( uint plantID, uint shootID, float assimilate_increment_mg_g );

    void incrementAssimilatePool( uint plantID, float assimilate_increment_mg_g );

    // -- plant building methods -- //

    //! Define a new shoot type based on a set of ShootParameters
    /**
     * \param[in] shoot_type_label User-defined label for the new shoot type. This string is used later to reference this type of shoot.
     * \param[in] shoot_params Parameters structure for the new shoot type.
     */
    void defineShootType( const std::string &shoot_type_label, const ShootParameters &shoot_params );

    //! Define the stem/trunk shoot (base of plant) to start a new plant. This requires a plant instance has already been created using the addPlantInstance() method.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot.
     * \param[in] shoot_type_label Label of the shoot type to be used for the base stem shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint addBaseStemShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label);

    //! Manually append a new shoot at the end of an existing shoot. This is used when the characteristics of a shoot change along its length (e.g., from a unifolitate to trifoliate leaf).
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be appended.
     * \param[in] current_node_number Number of nodes of the newly appended shoot.
     * \param[in] base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot relative to the parent shoot.
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label);

    //! Manually add a child shoot at the axillary bud of a phytomer.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be added.
     * \param[in] parent_node Number of the node of the parent shoot at which the new shoot will be added.
     * \param[in] current_node_number Number of nodes of the newly added shoot.
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node, uint current_node_number, const std::string &shoot_type_label);

    bool sampleChildShootType( uint plantID, uint shootID, std::string &child_shoot_type_label ) const;

    //! Add a new phytomer at the terminal bud of a shoot.
    int addPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_params, float internode_scale_factor_fraction, float leaf_scale_factor_fraction);

    // -- methods for modifying the current plant state -- //

    void scalePhytomerInternodeLength(uint plantID, uint shootID, uint node_number, float length_scale_factor);

    void incrementPhytomerInternodeGirth(uint plantID, uint shootID, uint node_number, float girth_change);

    void setPhytomerInternodeScale(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction);

    void setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction);

    void setPhytomerScale(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction, float leaf_scale_factor_fraction);

    void setPhytomerVegetativeBudState(uint plantID, uint shootID, uint node_number, BudState state );

    void setPhytomerFlowerBudState(uint plantID, uint shootID, uint node_number, BudState state );

    void setPlantBasePosition(uint plantID, const helios::vec3 &base_position);

    helios::vec3 getPlantBasePosition(uint plantID) const;

    void setPlantAge(uint plantID, float current_age);

    float getPlantAge(uint plantID) const;

    void harvestPlant(uint plantID);

    void removeShootLeaves(uint plantID, uint shootID);

    void removePlantLeaves(uint plantID );

    void makePlantDormant( uint plantID );

    void breakPlantDormancy( uint plantID );

    uint getShootNodeCount( uint plantID, uint shootID ) const;

    std::vector<uint> getAllPlantObjectIDs(uint plantID) const;

    std::vector<uint> getAllPlantUUIDs(uint PlantID) const;

    std::string getLSystemsString(uint plantID) const;


    void addAlmondShoot();

    void addAlmondTree();

    void addWalnutShoot();

private:

    helios::Context* context_ptr;

    std::minstd_rand0 *generator = nullptr;

    uint plant_count = 0;

    std::map<uint,PlantInstance> plant_instances;

    std::string makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> & shoot_tree) const;

    std::map<std::string,ShootParameters> shoot_types;

    void validateShootTypes( ShootParameters &shoot_parameters ) const;

    void accumulateShootPhotosynthesis( float dt );

};


#endif //PLANT_ARCHITECTURE
