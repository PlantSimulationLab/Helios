/** \file "PlantArchitecture.h" Primary header file for plant architecture plug-in.

    Copyright (C) 2016-2024 Brian Bailey

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

//forward declarations of classes/structs
class PlantArchitecture;
struct Shoot;
struct Phytomer;

struct RandomParameter_float {
public:

    //! Constructor initializing to a constant default value of 0.
    /**
     * In order to make this a randomly varying parameter, the initialize() method must be called to set the random number generator.
     */
    explicit RandomParameter_float(){
        constval = 0.f;
        distribution = "constant";
        generator = nullptr;
        sampled = false;
    }

    //! Constructor initializing to a constant value.
    /**
     * In order to make this a randomly varying parameter, the initialize() method must be called to set the random number generator.
     */
    explicit RandomParameter_float( float val ){
        constval = val;
        distribution = "constant";
        generator = nullptr;
        sampled = false;
    }

    //! Constructor initializing the random number generator.
    /**
     * \param[in] rand_generator Pointer to a random number generator. Note: it is recommended to use the random number generator from the Context, which can be retrieved using the getContextRandomGenerator() method.
     */
    explicit RandomParameter_float( std::minstd_rand0 *rand_generator ){
        constval = 0.f;
        distribution = "constant";
        generator = rand_generator;
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
    BUD_FLOWER_CLOSED = 2,
    BUD_FLOWER_OPEN = 3,
    BUD_FRUITING = 4,
    BUD_DEAD = 5
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

struct VegetativeBud{

    //label of the shoot type that will be produced if the bud breaks into a shoot
    std::string shoot_type_label;
    //state of the bud
    BudState state = BUD_DORMANT;
    //ID of the shoot that the bud will produce if it breaks into a shoot
    uint shoot_ID = -1;

};

struct FloralBud{

    //state of the bud
    BudState state = BUD_DORMANT;
    //amount of time since the bud flowered (=0 if it has not yet flowered)
    float time_counter = 0;
    //Index of the petiole within the internode that this floral bud originates from
    uint parent_petiole_index = 0;
    //Index of the bud within the petiole that this floral bud originates from
    uint bud_index = 0;
    //Scaling factor fraction of the fruit (if present), ranging from 0 to 1
    float current_fruit_scale_factor = 1;

};

struct PhytomerParameters{
private:

    struct InternodeParameters{
        RandomParameter_float pitch;
        RandomParameter_float phyllotactic_angle;
        RandomParameter_int max_vegetative_buds_per_petiole;
        RandomParameter_int max_floral_buds_per_petiole;
        helios::RGBcolor color;
        uint length_segments;
        uint radial_subdivisions;
        InternodeParameters& operator=(const InternodeParameters &a){
            this->pitch = a.pitch;
            this->pitch.resample();
            this->phyllotactic_angle = a.phyllotactic_angle;
            this->phyllotactic_angle.resample();
            this->max_vegetative_buds_per_petiole = a.max_vegetative_buds_per_petiole;
            this->max_vegetative_buds_per_petiole.resample();
            this->max_floral_buds_per_petiole = a.max_floral_buds_per_petiole;
            this->max_floral_buds_per_petiole.resample();
            this->color = a.color;
            this->length_segments = a.length_segments;
            this->radial_subdivisions = a.radial_subdivisions;
            return *this;
        }
    };

    struct PetioleParameters{
        uint petioles_per_internode;
        RandomParameter_float pitch;
        RandomParameter_float radius;
        RandomParameter_float length;
        RandomParameter_float curvature;
        RandomParameter_float taper;
        helios::RGBcolor color;
        uint length_segments;
        uint radial_subdivisions;
        PetioleParameters& operator=(const PetioleParameters &a){
            this->petioles_per_internode = a.petioles_per_internode;
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
            this->color = a.color;
            this->length_segments = a.length_segments;
            this->radial_subdivisions = a.radial_subdivisions;
            return *this;
        }
    };

    struct LeafParameters{
        int leaves_per_petiole;
        RandomParameter_float pitch;
        RandomParameter_float yaw;
        RandomParameter_float roll;
        RandomParameter_float leaflet_offset;
        RandomParameter_float leaflet_scale;
        RandomParameter_float prototype_scale;
        uint(*prototype_function)( helios::Context*, uint subdivisions, int flag ) = nullptr;
        LeafParameters& operator=(const LeafParameters &a){
            this->leaves_per_petiole = a.leaves_per_petiole;
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
        RandomParameter_float peduncle_length;
        RandomParameter_float peduncle_pitch;
        RandomParameter_float peduncle_roll;
        RandomParameter_int flowers_per_rachis;
        RandomParameter_float flower_offset;
        RandomParameter_float peduncle_radius;
        RandomParameter_float curvature;
        std::string flower_arrangement_pattern;
        uint length_segments;
        uint radial_subdivisions;
        RandomParameter_float fruit_pitch;
        RandomParameter_float fruit_roll;
        RandomParameter_float fruit_prototype_scale;
        RandomParameter_float flower_prototype_scale;
        bool fruit_gravity_on;
        uint(*fruit_prototype_function)( helios::Context*, uint subdivisions, int flag ) = nullptr;
        uint(*flower_prototype_function)(helios::Context*, uint subdivisions, bool flower_is_open ) = nullptr;
        InflorescenceParameters& operator=(const InflorescenceParameters &a){
            this->peduncle_length = a.peduncle_length;
            this->peduncle_length.resample();
            this->peduncle_pitch = a.peduncle_pitch;
            this->peduncle_pitch.resample();
            this->peduncle_roll = a.peduncle_roll;
            this->peduncle_roll.resample();
            this->flowers_per_rachis = a.flowers_per_rachis;
            this->flowers_per_rachis.resample();
            this->flower_offset = a.flower_offset;
            this->flower_offset.resample();
            this->peduncle_radius = a.peduncle_radius;
            this->peduncle_radius.resample();
            this->curvature = a.curvature;
            this->curvature.resample();
            this->flower_arrangement_pattern = a.flower_arrangement_pattern;
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
            this->fruit_gravity_on = a.fruit_gravity_on;
            this->flower_prototype_scale.resample();
            this->flower_prototype_function = a.flower_prototype_function;
            return *this;
        }
    };

public:

    InternodeParameters internode;

    PetioleParameters petiole;

    LeafParameters leaf;

    InflorescenceParameters inflorescence;

    //Custom user-defined function that is called when a phytomer is created
    /**
     * \param[in] phytomer_ptr Pointer to the phytomer to which the function will be applied
     * \param[in] shoot_node_index Index of the phytomer within the shoot starting from 0 at the shoot base
     * \param[in] parent_shoot_node_index Node index of the current shoot along it's parent shoot
     * \param[in] shoot_max_nodes Maximum number of phytomers in the shoot
     * \param[in] plant_age Age of the plant in days
     */
    void (*phytomer_creation_function)(std::shared_ptr<Phytomer> phytomer_ptr, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) = nullptr;

    //! Default constructor - does not set random number generator
    PhytomerParameters();

    //! Constructor - sets random number generator
    explicit PhytomerParameters( std::minstd_rand0 *generator );

    friend class PlantArchitecture;
    friend class Phytomer;
    friend class Shoot;

};

struct ShootParameters{

    //! Default constructor - does not set random number generator
    ShootParameters();

    //! Constructor - sets random number generator
    explicit ShootParameters( std::minstd_rand0 *generator );

    PhytomerParameters phytomer_parameters;

    // ---- Geometric Parameters ---- //

    RandomParameter_int max_nodes;

    RandomParameter_float internode_radius_initial;

    RandomParameter_float child_insertion_angle_tip;
    RandomParameter_float child_insertion_angle_decay_rate;

    RandomParameter_float child_internode_length_max;
    RandomParameter_float child_internode_length_min;
    RandomParameter_float child_internode_length_decay_rate;

    RandomParameter_float base_roll;
    RandomParameter_float base_yaw;

    RandomParameter_float gravitropic_curvature;  //degrees/length

    RandomParameter_float tortuosity; //degrees/length (standard deviation of random curvature perturbation)

    // ---- Growth Parameters ---- //

    RandomParameter_float phyllochron; //phytomers/day
    uint leaf_flush_count;  //number of leaves in a 'flush' (=1 gives continuous leaf production)

    RandomParameter_float elongation_rate; //length/day
    RandomParameter_float girth_growth_rate; //1/day

    // Probability that bud with this shoot type will break and form a new shoot
    float bud_break_probability;

    // Probability that a phytomer will flower
    float flower_probability;

    // Probability that a flower will set fruit
    float fruit_set_probability;

    RandomParameter_float bud_time;  //days

    bool flowers_require_dormancy;
    bool growth_requires_dormancy;

    bool determinate_shoot_growth;  //true=determinate, false=indeterminate

    // ---- Custom Functions ---- //

    void defineChildShootTypes( const std::vector<std::string> &child_shoot_type_labels, const std::vector<float> &child_shoot_type_probabilities );

    ShootParameters& operator=(const ShootParameters &a) {
        this->phytomer_parameters = a.phytomer_parameters;
        this->max_nodes = a.max_nodes;
        max_nodes.resample();
        this->internode_radius_initial = a.internode_radius_initial;
        this->internode_radius_initial.resample();
        this->phyllochron = a.phyllochron;
        this->phyllochron.resample();
        this->leaf_flush_count = a.leaf_flush_count;
        this->elongation_rate = a.elongation_rate;
        this->elongation_rate.resample();
        this->girth_growth_rate = a.girth_growth_rate;
        this->girth_growth_rate.resample();
        this->gravitropic_curvature = a.gravitropic_curvature;
        this->gravitropic_curvature.resample();
        this->tortuosity = a.tortuosity;
        this->tortuosity.resample();
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
        this->base_roll = a.base_roll;
        this->base_roll.resample();
        this->base_yaw = a.base_yaw;
        this->base_yaw.resample();
        this->flowers_require_dormancy = a.flowers_require_dormancy;
        this->growth_requires_dormancy = a.growth_requires_dormancy;
        this->child_shoot_type_labels = a.child_shoot_type_labels;
        this->child_shoot_type_probabilities = a.child_shoot_type_probabilities;
        this->determinate_shoot_growth = a.determinate_shoot_growth;
        return *this;
    }

    friend class PlantArchitecture;
    friend class Shoot;

protected:

    std::vector<std::string> child_shoot_type_labels;
    std::vector<float> child_shoot_type_probabilities;

};

struct Phytomer {
public:

    // Constructor
    Phytomer(const PhytomerParameters &params, Shoot *parent_shoot, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, helios::vec3 internode_base_origin,
             const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, helios::Context *context_ptr);

    // ---- query info about the phytomer ---- //

    helios::vec3 getInternodeAxisVector( float stem_fraction ) const;

    helios::vec3 getPetioleAxisVector(float stem_fraction, uint petiole_index) const;

    helios::vec3 getAxisVector( float stem_fraction, const std::vector<helios::vec3> &axis_vertices ) const;

    float getInternodeLength() const;

    float getPetioleLength() const;

    float getInternodeRadius( float stem_fraction ) const;

    bool hasLeaf() const;

    bool hasInflorescence() const;

    // ---- modify the phytomer ---- //

    void addInflorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation, const helios::vec3 &a_inflorescence_bending_axis, FloralBud &fbud);

    void setInternodeScaleFraction(float internode_scale_factor_fraction );

    void setInternodeMaxLength( float internode_length_max );

    void scaleInternodeMaxLength( float scale_factor );

    void setLeafScaleFraction(float leaf_scale_factor_fraction );

    void setLeafPrototypeScale( float leaf_prototype_scale );

    void scaleLeafPrototypeScale( float scale_factor );

    void setInflorescenceScaleFraction(FloralBud &fbud, float inflorescence_scale_factor_fraction);

    void setPetioleBase( const helios::vec3 &base_position );

    void setPhytomerBase( const helios::vec3 &base_position );

    void changeReproductiveState(FloralBud &fbud, BudState state);

    void setVegetativeBudState( BudState state );

    void setVegetativeBudState(BudState state, uint petiole_index, uint bud_index);

    void setFloralBudState( BudState state );

    void setFloralBudState(BudState state, uint petiole_index, uint bud_index);

    void removeLeaf();

    void removeInflorescence();

    // ---- phytomer data ---- //

    std::vector<helios::vec3> internode_vertices; //index is tube segment within internode
    std::vector<std::vector<helios::vec3>> petiole_vertices; //first index is petiole within internode, second index is tube segment within petiole
    std::vector<std::vector<helios::vec3>> leaf_bases; //first index is petiole within internode, second index is leaf within petiole
    std::vector<std::vector<std::vector<helios::vec3>>> inflorescence_bases; //first index is the petiole within internode, second index is the floral bud, third index is flower/fruit within peduncle/rachis
    float internode_length;

    std::vector<float> internode_radii; //index is segment within internode
    std::vector<std::vector<float>> petiole_radii; //first index is petiole within internode, second index is segment within petiole
    std::vector<float> petiole_length; //index is petiole within internode

    std::vector<helios::RGBcolor> internode_colors;
    std::vector<helios::RGBcolor> petiole_colors;

    std::vector<uint> internode_objIDs; //index is segment within internode
    std::vector<std::vector<uint> > petiole_objIDs; //first index is petiole within internode, second index is segment within petiole
    std::vector<std::vector<uint>> leaf_objIDs; //first index is petiole within internode, second index is leaf within petiole
    std::vector<std::vector<std::vector<uint>>> inflorescence_objIDs; //first index is the petiole within internode, second index is the floral bud, third index is flower/fruit within peduncle/rachis
    std::vector<std::vector<std::vector<uint>>> peduncle_objIDs; //first index is the petiole within internode, second index is the floral bud, third index is flower/fruit within peduncle/rachis

    PhytomerParameters phytomer_parameters;

    uint rank;
    helios::int2 shoot_index; // .x = index of phytomer along shoot, .y = maximum number of phytomers on parent shoot

    float age = 0;
    float time_since_dormancy = 0;

    float current_internode_scale_factor = 1;
    float current_leaf_scale_factor = 1;

    std::vector<std::vector<VegetativeBud>> vegetative_buds; //first index is petiole within internode, second index is bud within petiole
    std::vector<std::vector<FloralBud>> floral_buds;

    float internode_radius_initial;
    float internode_length_max;

private:

    helios::vec3 inflorescence_bending_axis;

    helios::Context *context_ptr;

};

struct Shoot{

    Shoot(uint plant_ID, int shoot_ID, int parent_shoot_ID, uint parent_node, uint parent_petiole_index, uint rank, const helios::vec3 &origin, const AxisRotation &shoot_base_rotation, uint current_node_number,
          float internode_length_shoot_initial, ShootParameters shoot_params, const std::string &shoot_type_label, PlantArchitecture *plant_architecture_ptr);

    void buildShootPhytomers(float internode_radius, float internode_length, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction);

    int addPhytomer(const PhytomerParameters &params, const helios::vec3 internode_base_position, const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction,
                    float leaf_scale_factor_fraction);

    //! Randomly sample the type of a child shoot based on the probabilities defined in the shoot parameters
    /**
     * \param[out] child_shoot_type_label Label of the randomly selected child shoot type.
     * \return false if the bud dies, true if the bud survives and will produce a new shoot.
     */
    bool sampleChildShootType(std::string &child_shoot_type_label) const;

    void terminateApicalBud();

    void terminateAxillaryVegetativeBuds();

    uint current_node_number;

    helios::vec3 origin;

    AxisRotation base_rotation;

    const int ID;
    const int parent_shoot_ID;
    const uint plantID;
    const uint parent_node_index;
    const uint rank;
    const uint parent_petiole_index;

    float assimilate_pool;  // mg SC/g DW

    void breakDormancy();
    void makeDormant();

    bool dormant;
    uint dormancy_cycles = 0;

    bool meristem_is_alive = true;

    float phyllochron_counter = 0;

    float curvature_perturbation = 0;

    const float internode_length_max_shoot_initial;

    //map of node number to ID of shoot child
    std::map<int,int> childIDs;

    ShootParameters shoot_parameters;

    std::string shoot_type_label;

    std::vector<std::shared_ptr<Phytomer> > phytomers;

    PlantArchitecture* plant_architecture_ptr;

    helios::Context *context_ptr;

};

struct PlantInstance{

    PlantInstance(const helios::vec3 &a_base_position, float a_current_age) : base_position(a_base_position), current_age(a_current_age) {}
    std::vector<std::shared_ptr<Shoot> > shoot_tree;
    helios::vec3 base_position;
    float current_age;

    //Phenological thresholds
    float dd_to_dormancy_break = 0;
    float dd_to_flower_initiation = 0;
    float dd_to_flower_opening = 0;
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

    // ********* Methods for Building Plants from Existing Library ********* //

    void loadPlantModelFromLibrary( const std::string &plant_label );

    uint buildPlantInstanceFromLibrary( const helios::vec3 &base_position, float age );

    ShootParameters getCurrentShootParameters( const std::string &shoot_type_label );

    std::map<std::string, ShootParameters> getCurrentShootParameters( );

    //! Update the parameters of a single shoot type in the current plant model
    /**
     * \param[in] shoot_type_label User-defined label for the shoot type to be updated.
     * \param[in] params Updated parameters structure for the shoot type.
     * \note This will overwrite any existing shoot parameter definitions.
     */
    void updateCurrentShootParameters( const std::string &shoot_type_label, const ShootParameters &params );

    //! Update the parameters of all shoot types in the current plant model
    /**
     * \param[in] shoot_type_label User-defined label for the shoot type to be updated.
     * \param[in] params Updated parameters structure for the shoot type.
     * \note This will overwrite any existing shoot parameter definitions.
     */
    void updateCurrentShootParameters( const std::map<std::string, ShootParameters> &params );

    // ********* Methods for Building Custom Plant Geometry from Scratch ********* //

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
    uint duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, const AxisRotation &base_rotation, float current_age);

    //! Specify the threshold values for plant phenological stages
    /**
     * \param[in] plantID ID of the plant.
     * \param[in] dd_to_dormancy_break Minimum assimilate pool (mg SC/g DW) required to break dormancy.
     * \param[in] dd_to_flower_initiation Degree-days from emergence/dormancy required to reach flower creation (closed flower).
     * \param[in] dd_to_flower_opening Degree-days from flower initiation to flower opening.
     * \param[in] dd_to_fruit_set Degree-days from flower opening required to reach fruit set (i.e., flower dies and fruit is created).
     * \param[in] dd_to_fruit_maturity Degree-days from fruit set date required to reach fruit maturity.
     * \param[in] dd_to_senescence Degree-days from emergence/dormancy required to reach senescence.
     * \note Any phenological stage can be skipped by specifying a negative threshold value. In this case, the stage will be skipped and the threshold for the next stage will be relative to the previous stage.
     */
    void setPlantPhenologicalThresholds(uint plantID, float dd_to_dormancy_break, float dd_to_flower_initiation, float dd_to_flower_opening, float dd_to_fruit_set, float dd_to_fruit_maturity, float dd_to_senescence);

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
     * \param[in] current_node_number Number of nodes of the stem shoot.
     * \param[in] base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot.
     * \param[in] internode_radius Radius of the internodes along the shoot.
     * \param[in] internode_length_max Maximum length (i.e., fully elongated) of the internodes along the shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual internode length (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual leaf size (=1 applies no scaling).
     * \param[in] shoot_type_label Label of the shoot type to be used for the base stem shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint addBaseStemShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction,
                          const std::string &shoot_type_label);

    //! Manually append a new shoot at the end of an existing shoot. This is used when the characteristics of a shoot change along its length (e.g., from a unifoliate to trifoliate leaf).
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be appended.
     * \param[in] current_node_number Number of nodes of the newly appended shoot.
     * \param[in] internode_length_max Length of the internode of the newly appended shoot.
     * \param[in] base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot relative to the parent shoot.
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction,
                     float leaf_scale_factor_fraction, const std::string &shoot_type_label);

    //! Manually add a child shoot at the axillary bud of a phytomer.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be added.
     * \param[in] parent_node_index Number of the node of the parent shoot at which the new shoot will be added.
     * \param[in] current_node_number Number of nodes of the newly added shoot.
     * \param[in] shoot_base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot.
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node_index, uint current_node_number, const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max,
                       float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, const std::string &shoot_type_label, uint petiole_index);

    //! Add a new phytomer at the terminal bud of a shoot.
    int addPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_params, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction);

    // -- methods for modifying the current plant state -- //

    void incrementPhytomerInternodeGirth(uint plantID, uint shootID, uint node_number, float girth_change);

    void setPhytomerInternodeScale(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction);

    void setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction);

    void setShootOrigin(uint plantID, uint shootID, const helios::vec3 &origin);

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

    std::vector<uint> getPlantInternodeObjectIDs(uint plantID) const;

    std::vector<uint> getPlantPetioleObjectIDs(uint plantID) const;

    std::vector<uint> getPlantLeafObjectIDs(uint plantID) const;

    std::string getLSystemsString(uint plantID) const;

    uint generatePlantFromString(const std::string &generation_string, const PhytomerParameters &phytomer_parameters);

    uint generatePlantFromString(const std::string &generation_string, const std::map<std::string,PhytomerParameters> &phytomer_parameters);

    friend class Shoot;

protected:

    helios::Context* context_ptr;

    std::minstd_rand0 *generator = nullptr;

    uint plant_count = 0;

    std::string current_plant_model;

    std::map<uint,PlantInstance> plant_instances;

    std::string makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> & shoot_tree) const;

    std::map<std::string,ShootParameters> shoot_types;

    void validateShootTypes( ShootParameters &shoot_parameters ) const;

    void accumulateShootPhotosynthesis( float dt );

    void parseStringShoot(const std::string &LString_shoot, uint plantID, int parentID, uint parent_node, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters);

    void parseShootArgument(const std::string &shoot_argument, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters, AxisRotation &base_rotation, std::string &phytomer_label);

    void parseInternodeArgument(const std::string &internode_argument, float &internode_radius, float &internode_length, PhytomerParameters &phytomer_parameters);

    void parsePetioleArgument(const std::string& petiole_argument, PhytomerParameters &phytomer_parameters );

    void parseLeafArgument(const std::string& leaf_argument, PhytomerParameters &phytomer_parameters );

    void shiftDownstreamShoots(uint plantID, std::vector<std::shared_ptr<Shoot>> &shoot_tree, std::shared_ptr<Shoot> parent_shoot_ptr, const helios::vec3 &base_position );

    void initializeDefaultShoots( const std::string &plant_label );

    void initializeAlmondTreeShoots();

    uint buildAlmondTree( const helios::vec3 &base_position, float age );

    void initializeCowpeaShoots();

    uint buildCowpeaPlant( const helios::vec3 &base_position, float age );

    void initializeSoybeanShoots();

    uint buildSoybeanPlant( const helios::vec3 &base_position, float age );

    void initializeBeanShoots();

    uint buildBeanPlant( const helios::vec3 &base_position, float age );

    void initializeSorghumShoots();

    uint buildSorghumPlant( const helios::vec3 &base_position, float age );

    void initializeTomatoShoots();

    uint buildTomatoPlant( const helios::vec3 &base_position, float age );

};


#endif //PLANT_ARCHITECTURE
