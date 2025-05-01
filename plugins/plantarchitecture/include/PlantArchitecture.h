/** \file "PlantArchitecture.h" Primary header file for plant architecture plug-in.

    Copyright (C) 2016-2025 Brian Bailey

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
#include <utility>
#include "Hungarian.h"

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
        if( minval>maxval ){
            throw (std::runtime_error("ERROR (PlantArchitecture): RandomParameter_float::uniformDistribution() - minval must be less than or equal to maxval."));
        }
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
        if( minval>maxval ){
            throw (std::runtime_error("ERROR (PlantArchitecture): RandomParameter_int::uniformDistribution() - minval must be less than or equal to maxval."));
        }
        distribution = "uniform";
        distribution_parameters = {minval, maxval};
        sampled = false;
    }

    void discreteValues( const std::vector<int> &values ){
        distribution = "discretevalues";
        distribution_parameters = values;
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
            }else if (distribution == "discretevalues") {
                std::uniform_int_distribution<> unif_distribution(0,distribution_parameters.size()-1);
                constval = distribution_parameters.at(unif_distribution(*generator));
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

    friend std::ostream &operator<<(std::ostream &os, const AxisRotation &rot) {
        return os << "AxisRotation<" << rot.pitch << ", " << rot.yaw << ", " << rot.roll << ">";
    }

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

    //state of the bud
    BudState state = BUD_DORMANT;
    //label of the shoot type that will be produced if the bud breaks into a shoot
    std::string shoot_type_label;
    //ID of the shoot that the bud will produce if it breaks into a shoot
    uint shoot_ID = -1;

};

struct FloralBud{

    //state of the bud
    BudState state = BUD_DORMANT;
    //amount of time since the bud flowered (=0 if it has not yet flowered)
    float time_counter = 0;
    //=0 for axillary buds, =1 for terminal buds
    bool isterminal = false;
    //For axillary buds: index of the petiole within the internode that this floral bud originates from
    //For terminal buds: index of the phytomer within the shoot that this floral bud originates from
    uint parent_index = 0;
    //Index of the bud within the petiole that this floral bud originates from
    uint bud_index = 0;
    //Scaling factor fraction of the fruit (if present), ranging from 0 to 1
    float current_fruit_scale_factor = 1;
    float previous_fruit_scale_factor = 0;



    helios::vec3 base_position;
    AxisRotation base_rotation;
    helios::vec3 bending_axis;

    std::vector<helios::vec3> inflorescence_bases;
    std::vector<uint> peduncle_objIDs;
    std::vector<uint> inflorescence_objIDs;
};

struct LeafPrototype {
public:

    //! Constructor - sets random number generator
    explicit LeafPrototype( std::minstd_rand0 *generator );

    //! Constructor - does not set random number generator
    LeafPrototype( )= default;

    //! Custom prototype function for creating leaf prototypes
    uint (*prototype_function)(helios::Context *, LeafPrototype* prototype_parameters, int compound_leaf_index) = nullptr;

    //! OBJ model file to load for the leaf
    /**
     *\note If this is set, the leaf will be loaded from the OBJ file and the other leaf parameters will be ignored.
     */
    std::string OBJ_model_file;

    //! Image texture file for the leaf
    /**
     *\note Key is the index of the compound leaf (=0 is the tip leaf, <0 increases down left side of the leaflet, >0 increases down the right side of the leaflet), value is the texture file.
     */
    std::map<int,std::string> leaf_texture_file;

    // Ratio of leaf width to leaf length
    RandomParameter_float leaf_aspect_ratio;

    //! Fraction of folding along the leaf midrib. =0 means leaf is flat, =1 means leaf is completely folded in half along midrib.
    RandomParameter_float midrib_fold_fraction;

    // Parameters for leaf curvature
    //! Leaf curvature factor along the longitudinal/length (x-direction). (+curves upward, -curved downward)
    RandomParameter_float longitudinal_curvature;
    //! Leaf curvature factor along the lateral/width (y-direction). (+curves upward, -curved downward)
    RandomParameter_float lateral_curvature;

    //! Creates a rolling at the leaf where the petiole attaches to the leaf blade
    RandomParameter_float petiole_roll;

    // Parameters for leaf wave/wrinkles
    //! Period factor of leaf waves (sets how many waves there are along the leaf length)
    RandomParameter_float wave_period;
    //! Amplitude of leaf waves (sets the height of leaf waves)
    RandomParameter_float wave_amplitude;

    // Parameters for leaf buckling
    //! Fraction of the leaf length where the leaf buckles under its weight
    RandomParameter_float leaf_buckle_length;
    //! Angle of the leaf buckle (degrees)
    RandomParameter_float leaf_buckle_angle;

    //! Amount to shift the leaf
    helios::vec3 leaf_offset;

    //! Leaf subdivision count in each direction
    uint subdivisions;

    //! Number of unique prototypes to generate
    uint unique_prototypes;

    //! Add a petiolule to the base of the leaflet
    bool build_petiolule;

    uint unique_prototype_identifier = 0;

    void duplicate( const LeafPrototype &a){
        this->leaf_texture_file = a.leaf_texture_file;
        this->OBJ_model_file = a.OBJ_model_file;
        this->leaf_aspect_ratio = a.leaf_aspect_ratio;
        this->midrib_fold_fraction = a.midrib_fold_fraction;
        this->longitudinal_curvature = a.longitudinal_curvature;
        this->lateral_curvature = a.lateral_curvature;
        this->petiole_roll = a.petiole_roll;
        this->wave_period = a.wave_period;
        this->wave_amplitude = a.wave_amplitude;
        this->leaf_buckle_length = a.leaf_buckle_length;
        this->leaf_buckle_angle = a.leaf_buckle_angle;
        this->leaf_offset = a.leaf_offset;
        this->subdivisions = a.subdivisions;
        this->unique_prototypes = a.unique_prototypes;
        this->unique_prototype_identifier = a.unique_prototype_identifier;
        this->build_petiolule = a.build_petiolule;
        this->prototype_function = a.prototype_function;
        this->generator = a.generator;
    }

    //! Assignment operator
    LeafPrototype& operator=(const LeafPrototype &a){
        if (this != &a) {
            this->leaf_texture_file = a.leaf_texture_file;
            this->OBJ_model_file = a.OBJ_model_file;
            this->leaf_aspect_ratio = a.leaf_aspect_ratio;
            this->leaf_aspect_ratio.resample();
            this->midrib_fold_fraction = a.midrib_fold_fraction;
            this->midrib_fold_fraction.resample();
            this->longitudinal_curvature = a.longitudinal_curvature;
            this->longitudinal_curvature.resample();
            this->lateral_curvature = a.lateral_curvature;
            this->lateral_curvature.resample();
            this->petiole_roll = a.petiole_roll;
            this->petiole_roll.resample();
            this->wave_period = a.wave_period;
            this->wave_period.resample();
            this->wave_amplitude = a.wave_amplitude;
            this->wave_amplitude.resample();
            this->leaf_buckle_length = a.leaf_buckle_length;
            this->leaf_buckle_length.resample();
            this->leaf_buckle_angle = a.leaf_buckle_angle;
            this->leaf_buckle_angle.resample();
            this->leaf_offset = a.leaf_offset;
            this->subdivisions = a.subdivisions;
            this->unique_prototypes = a.unique_prototypes;
            this->unique_prototype_identifier = a.unique_prototype_identifier;
            this->build_petiolule = a.build_petiolule;
            this->prototype_function = a.prototype_function;
            this->generator = a.generator;
            if ( this->generator != nullptr ){
                this->sampleIdentifier();
            }
        }
        return *this;
    }

    void sampleIdentifier() {
        assert(generator!=nullptr);
        std::uniform_int_distribution<uint> unif_distribution;
        this->unique_prototype_identifier = unif_distribution(*generator);
    }

private:

    std::minstd_rand0 *generator;

};

struct PhytomerParameters{
private:

    struct InternodeParameters{
        RandomParameter_float pitch;
        RandomParameter_float phyllotactic_angle;
        RandomParameter_float radius_initial;
        RandomParameter_int max_vegetative_buds_per_petiole;
        RandomParameter_int max_floral_buds_per_petiole;
        helios::RGBcolor color;
        std::string image_texture;
        uint length_segments;
        uint radial_subdivisions;

        InternodeParameters& operator=(const InternodeParameters &a){
            if (this != &a) {
                this->pitch = a.pitch;
                this->pitch.resample();
                this->phyllotactic_angle = a.phyllotactic_angle;
                this->phyllotactic_angle.resample();
                this->radius_initial = a.radius_initial;
                this->radius_initial.resample();
                this->max_vegetative_buds_per_petiole = a.max_vegetative_buds_per_petiole;
                this->max_vegetative_buds_per_petiole.resample();
                this->max_floral_buds_per_petiole = a.max_floral_buds_per_petiole;
                this->max_floral_buds_per_petiole.resample();
                this->color = a.color;
                this->image_texture = a.image_texture;
                this->length_segments = a.length_segments;
                this->radial_subdivisions = a.radial_subdivisions;
            }
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
            if (this != &a) {
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
            }
            return *this;
        }
    };

    struct LeafParameters{
        RandomParameter_int leaves_per_petiole;
        RandomParameter_float pitch;
        RandomParameter_float yaw;
        RandomParameter_float roll;
        RandomParameter_float leaflet_offset;
        RandomParameter_float leaflet_scale;
        RandomParameter_float prototype_scale;
        LeafPrototype prototype;

        LeafParameters& operator=(const LeafParameters &a){
            if (this != &a) {
                this->leaves_per_petiole = a.leaves_per_petiole;
                this->leaves_per_petiole.resample();
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
                this->prototype.duplicate(a.prototype);
            }
            return *this;
        }
    };

    struct PeduncleParameters {
        RandomParameter_float length;
        RandomParameter_float radius;
        RandomParameter_float pitch;
        RandomParameter_float roll;
        RandomParameter_float curvature;
        helios::RGBcolor color;
        uint length_segments;
        uint radial_subdivisions;

        PeduncleParameters &operator=(const PeduncleParameters &a) {
            if (this != &a) {
                this->length = a.length;
                this->length.resample();
                this->radius = a.radius;
                this->radius.resample();
                this->pitch = a.pitch;
                this->pitch.resample();
                this->roll = a.roll;
                this->roll.resample();
                this->curvature = a.curvature;
                this->curvature.resample();
                this->color = a.color;
                this->length_segments = a.length_segments;
                this->radial_subdivisions = a.radial_subdivisions;
            }
            return *this;
        }
    };

    struct InflorescenceParameters {
        RandomParameter_int flowers_per_peduncle;
        RandomParameter_float flower_offset;
        RandomParameter_float pitch;
        RandomParameter_float roll;
        RandomParameter_float flower_prototype_scale;
        uint (*flower_prototype_function)(helios::Context *, uint subdivisions, bool flower_is_open) = nullptr;
        RandomParameter_float fruit_prototype_scale;
        uint (*fruit_prototype_function)(helios::Context *, uint subdivisions) = nullptr;
        RandomParameter_float fruit_gravity_factor_fraction;
        uint unique_prototypes;

        InflorescenceParameters &operator=(const InflorescenceParameters &a) {
            if (this != &a) {
                this->flowers_per_peduncle = a.flowers_per_peduncle;
                this->flowers_per_peduncle.resample();
                this->flower_offset = a.flower_offset;
                this->flower_offset.resample();
                this->pitch = a.pitch;
                this->pitch.resample();
                this->roll = a.roll;
                this->roll.resample();
                this->flower_prototype_scale = a.flower_prototype_scale;
                this->flower_prototype_scale.resample();
                this->flower_prototype_function = a.flower_prototype_function;
                this->fruit_prototype_scale = a.fruit_prototype_scale;
                this->fruit_prototype_scale.resample();
                this->fruit_prototype_function = a.fruit_prototype_function;
                this->fruit_gravity_factor_fraction = a.fruit_gravity_factor_fraction;
                this->fruit_gravity_factor_fraction.resample();
                this->unique_prototypes = a.unique_prototypes;
            }
            return *this;
        }
    };

public:

    InternodeParameters internode;

    PetioleParameters petiole;

    LeafParameters leaf;

    PeduncleParameters peduncle;

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

    //Custom user-defined function that is called for each phytomer on every time step
    /**
     * \param[in] phytomer_ptr Pointer to the phytomer to which the function will be applied
     */
    void (*phytomer_callback_function)(std::shared_ptr<Phytomer> phytomer_ptr) = nullptr;


    //! Default constructor - does not set random number generator
    PhytomerParameters();

    //! Constructor - sets random number generator
    explicit PhytomerParameters( std::minstd_rand0 *generator );

    friend class PlantArchitecture;
    friend struct Phytomer;
    friend struct Shoot;

};

struct ShootParameters{

    //! Default constructor - does not set random number generator
    ShootParameters();

    //! Constructor - sets random number generator
    explicit ShootParameters( std::minstd_rand0 *generator );

    PhytomerParameters phytomer_parameters;

    // ---- Geometric Parameters ---- //

    // Maximum number of nodes along the shoot before the terminal vegetative bud dies
    RandomParameter_int max_nodes;

    // Maximum number of nodes that can be produced by a shoot in a single season. By default, this is equal to max_nodes.
    RandomParameter_int max_nodes_per_season;

    RandomParameter_float girth_area_factor; //cm^2 branch area / m^2 downstream leaf area

    // Insertion angle of the most apical child shoot bud at the time it breaks
    RandomParameter_float insertion_angle_tip;
    RandomParameter_float insertion_angle_decay_rate;

    RandomParameter_float internode_length_max;
    RandomParameter_float internode_length_min;
    RandomParameter_float internode_length_decay_rate;

    RandomParameter_float base_roll;
    RandomParameter_float base_yaw;

    RandomParameter_float gravitropic_curvature;  //degrees/length

    RandomParameter_float tortuosity; //degrees/length (standard deviation of random curvature perturbation)

    // ---- Growth Parameters ---- //

    RandomParameter_float phyllochron_min; //days/phytomer

    RandomParameter_float elongation_rate_max; //length/day

    // Minimum probability that bud with this shoot type will break and form a new shoot
    RandomParameter_float vegetative_bud_break_probability_min;

    // Decay rate of the probability that a bud will break and form a new shoot
    /**
     * If this value is negative, the vegetative bud break probability starts at 1 at the shoot base and decreases at a rate of vegetative_bud_break_probability_decay_rate per phytomer until it reaches vegetative_bud_break_probability_min. If this value is positive, the reverse is true starting from the shoot tip.
     */
    RandomParameter_float vegetative_bud_break_probability_decay_rate;

    // Maximum number of floral buds at the shoot apex
    RandomParameter_int max_terminal_floral_buds;

    // Probability that a phytomer will flower
    RandomParameter_float flower_bud_break_probability;

    // Probability that a flower will set fruit
    RandomParameter_float fruit_set_probability;

    RandomParameter_float vegetative_bud_break_time;  //days

    bool flowers_require_dormancy;
    bool growth_requires_dormancy;

    bool determinate_shoot_growth;  //true=determinate, false=indeterminate

    // ---- Custom Functions ---- //

    void defineChildShootTypes( const std::vector<std::string> &child_shoot_type_labels, const std::vector<float> &child_shoot_type_probabilities );

    ShootParameters& operator=(const ShootParameters &a) {
        this->phytomer_parameters = a.phytomer_parameters;
        this->max_nodes = a.max_nodes;
        this->max_nodes.resample();
        this->max_nodes_per_season = a.max_nodes_per_season;
        this->max_nodes_per_season.resample();
        this->phyllochron_min = a.phyllochron_min;
        this->phyllochron_min.resample();
        this->girth_area_factor = a.girth_area_factor;
        this->girth_area_factor.resample();
        this->vegetative_bud_break_probability_min = a.vegetative_bud_break_probability_min;
        this->vegetative_bud_break_probability_min.resample();
        this->flower_bud_break_probability = a.flower_bud_break_probability;
        this->flower_bud_break_probability.resample();
        this->fruit_set_probability = a.fruit_set_probability;
        this->fruit_set_probability.resample();
        this->gravitropic_curvature = a.gravitropic_curvature;
        this->gravitropic_curvature.resample();
        this->tortuosity = a.tortuosity;
        this->tortuosity.resample();
        this->vegetative_bud_break_probability_min = a.vegetative_bud_break_probability_min;
        this->vegetative_bud_break_probability_min.resample();
        this->vegetative_bud_break_probability_decay_rate = a.vegetative_bud_break_probability_decay_rate;
        this->vegetative_bud_break_probability_decay_rate.resample();
        this->max_terminal_floral_buds = a.max_terminal_floral_buds;
        this->max_terminal_floral_buds.resample();
        this->flower_bud_break_probability = a.flower_bud_break_probability;
        this->flower_bud_break_probability.resample();
        this->fruit_set_probability = a.fruit_set_probability;
        this->fruit_set_probability.resample();
        this->vegetative_bud_break_time = a.vegetative_bud_break_time;
        this->vegetative_bud_break_time.resample();
        this->insertion_angle_tip = a.insertion_angle_tip;
        this->insertion_angle_tip.resample();
        this->insertion_angle_decay_rate = a.insertion_angle_decay_rate;
        this->insertion_angle_decay_rate.resample();
        this->internode_length_max = a.internode_length_max;
        this->internode_length_max.resample();
        this->internode_length_min = a.internode_length_min;
        this->internode_length_min.resample();
        this->internode_length_decay_rate = a.internode_length_decay_rate;
        this->internode_length_decay_rate.resample();
        this->base_roll = a.base_roll;
        this->base_roll.resample();
        this->base_yaw = a.base_yaw;
        this->base_yaw.resample();
        this->flowers_require_dormancy = a.flowers_require_dormancy;
        this->growth_requires_dormancy = a.growth_requires_dormancy;
        this->child_shoot_type_labels = a.child_shoot_type_labels;
        this->child_shoot_type_probabilities = a.child_shoot_type_probabilities;
        this->determinate_shoot_growth = a.determinate_shoot_growth;
        this->child_shoot_type_labels = a.child_shoot_type_labels;
        this->child_shoot_type_probabilities = a.child_shoot_type_probabilities;
        return *this;
    }

    friend class PlantArchitecture;
    friend struct Shoot;

protected:

    std::vector<std::string> child_shoot_type_labels;
    std::vector<float> child_shoot_type_probabilities;

};

struct Phytomer {
public:

    float phytomer_carbohydrate_cost_molC = 0;

    // Constructor
    Phytomer(const PhytomerParameters &params, Shoot *parent_shoot, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, helios::vec3 internode_base_origin,
             const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, PlantArchitecture *plantarchitecture_ptr,
             helios::Context *context_ptr);

    // ---- query info about the phytomer ---- //

    /**
     * \brief Retrieve the positions of nodes associated with an internode.
     *
     * \return A vector of vec3 objects representing the node positions for the current internode.
     */
    [[nodiscard]] std::vector<helios::vec3> getInternodeNodePositions() const;

    /**
     * \brief Retrieve the radii of nodes associated with an internode.
     *
     * \return A vector of floats representing the radii of the nodes for the current internode.
     */
    [[nodiscard]] std::vector<float> getInternodeNodeRadii() const;

    /**
     * \brief Retrieves the axis vector of the internode at a given fraction along the internode.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the axis vector is computed.
     * \return A vec3 representing the axis vector of the internode at the specified internode fraction.
     */
    [[nodiscard]] helios::vec3 getInternodeAxisVector(float stem_fraction) const;

    /**
     * \brief Retrieves the petiole axis vector for a given stem fraction and petiole index.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the axis vector is computed.
     * \param[in] petiole_index Index of the petiole for which the axis vector is retrieved.
     * \return The axis vector of the specified petiole as a helios::vec3.
     */
    [[nodiscard]] helios::vec3 getPetioleAxisVector(float stem_fraction, uint petiole_index) const;

    /**
     * \brief Computes the normalized vector direction along the axis at a given fraction of the stem.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the axis vector is computed.
     * \param[in] axis_vertices A list of 3D points defining the vertices of the axis.
     * \return A normalized vector direction at the given fraction along the axis.
     */
    [[nodiscard]] static helios::vec3 getAxisVector(float stem_fraction, const std::vector<helios::vec3> &axis_vertices) ;

    /**
     * \brief Computes the total length of the internode.
     *
     * \return The cumulative length of the internode based on its node positions.
     */
    [[nodiscard]] float getInternodeLength() const;

    /**
     * \brief Retrieves the radius of the internode.
     *
     * \return The radius of the internode.
     */
    [[nodiscard]] float getInternodeRadius() const;

    /**
     * \brief Retrieves the length of the petiole.
     *
     * \return Length of the petiole.
     */
    [[nodiscard]] float getPetioleLength() const;

    /**
     * \brief Retrieves the radius of the internode based on the stem fraction.
     *
     * \param[in] stem_fraction Fraction along the stem (value between 0.0 and 1.0) for which the radius is computed.
     * \return The radius of the internode at the specified stem fraction.
     */
    [[nodiscard]] float getInternodeRadius(float stem_fraction) const;

    /**
     * \brief Calculates the total leaf area for the current phytomer.
     *
     * \return Total leaf area as a float value.
     */
    [[nodiscard]] float getLeafArea() const;

    /**
     * \brief Retrieves the position of the base of a leaf within the phytomer.
     *
     * \param[in] petiole_index Index of the petiole within the phytomer structure.
     * \param[in] leaf_index Index of the leaf within the specified petiole.
     * \return A vec3 indicating the position of the leaf base.
     */
    [[nodiscard]] helios::vec3 getLeafBasePosition(uint petiole_index, uint leaf_index) const;

    /**
     * \brief Checks if the phytomer has a leaf.
     *
     * \return True if the phytomer has at least one leaf, false otherwise.
     */
    [[nodiscard]] bool hasLeaf() const;

    /**
     * \brief Calculates the total leaf area downstream of the current phytomer.
     *
     * \return The total downstream leaf area as a float.
     */
    [[nodiscard]] float calculateDownstreamLeafArea() const;

    // ---- modify the phytomer ---- //

    //! Set the current internode length as a fraction of its maximum length
    /**
     * \param[in] internode_scale_factor_fraction Fraction of the maximum internode length
     * \param[in] update_context_geometry If true, the context geometry will be immediately updated to reflect the new internode length. If false, the geometry will be updated the next time Context geometry information is needed.
     */
    void setInternodeLengthScaleFraction(float internode_scale_factor_fraction, bool update_context_geometry);

    //! Scale the fully-elongated (maximum) internode length as a fraction of its current fully-elongated length
    /**
     * \param[in] scale_factor Fraction by which to scale the fully-elongated internode length
     */
    void scaleInternodeMaxLength(float scale_factor);

    //! Set the fully-elongated (maximum) internode length
    /**
     * \param[in] internode_length_max_new Fully-elongated length of the internode
     */
    void setInternodeMaxLength(float internode_length_max_new);

    //! Set the maximum radius of the internode
    /**
     * \param[in] internode_radius_max_new Maximum radius of the internode
     */
    void setInternodeMaxRadius( float internode_radius_max_new );

    //! Set the leaf scale as a fraction of its total fully-elongated scale factor. Value is uniformly applied for all leaves/leaflets and petioles in the phytomer.
    /**
     * \param[in] leaf_scale_factor_fraction Fraction of the total fully-elongated leaf scale factor (i.e., =1 for fully-elongated leaf)
     */
    void setLeafScaleFraction(float leaf_scale_factor_fraction );

    //! Set the fully-elongated (maximum) leaf prototype scale. Value is uniformly applied for all leaves/leaflets in the phytomer.
    /**
     * \param[in] leaf_prototype_scale Leaf scale factor for fully-elongated leaf
     */
    void setLeafPrototypeScale( float leaf_prototype_scale );

    void scaleLeafPrototypeScale( float scale_factor );

    void setInflorescenceScaleFraction(FloralBud &fbud, float inflorescence_scale_factor_fraction) const;

    void setPetioleBase( const helios::vec3 &base_position );

    void rotateLeaf( uint petiole_index, uint leaf_index, const AxisRotation &rotation );

    void setVegetativeBudState( BudState state );

    void setVegetativeBudState(BudState state, uint petiole_index, uint bud_index);

    void setVegetativeBudState( BudState state, VegetativeBud &vbud ) const;

    void setFloralBudState(BudState state );

    void setFloralBudState(BudState state, uint petiole_index, uint bud_index);

    void setFloralBudState(BudState state, FloralBud &fbud);

    void removeLeaf();

    void deletePhytomer();

    // ---- phytomer data ---- //

    //! Coordinates of internode tube segments. Index is tube segment within internode
    std::vector<std::vector<helios::vec3>> petiole_vertices; //first index is petiole within internode, second index is tube segment within petiole
    std::vector<std::vector<helios::vec3>> leaf_bases; //first index is petiole within internode, second index is leaf within petiole
    float internode_pitch, internode_phyllotactic_angle;

    std::vector<std::vector<float>> petiole_radii; //first index is petiole within internode, second index is segment within petiole
    std::vector<float> petiole_length; //index is petiole within internode
    float petiole_pitch;
    float petiole_curvature;
    std::vector<std::vector<float>> leaf_size_max; //first index is petiole within internode, second index is leaf within petiole
    std::vector<std::vector<AxisRotation>> leaf_rotation; //first index is petiole within internode, second index is leaf within petiole

    std::vector<helios::RGBcolor> internode_colors;
    std::vector<helios::RGBcolor> petiole_colors;

    std::vector<std::vector<uint> > petiole_objIDs; //first index is petiole within internode, second index is segment within petiole
    std::vector<std::vector<uint>> leaf_objIDs; //first index is petiole within internode, second index is leaf within petiole

    PhytomerParameters phytomer_parameters;

    uint rank;
    //! .x = index of phytomer along shoot, .y = current number of phytomers on parent shoot, .z = maximum number of phytomers on parent shoot
    helios::int3 shoot_index;

    uint plantID;
    uint parent_shoot_ID;
    Shoot *parent_shoot_ptr;

    //! Time since the phytomer was created
    float age = 0;
    bool isdormant = false;

    float current_internode_scale_factor = 1;
    float current_leaf_scale_factor = 1;

    float downstream_leaf_area = 0;

    std::vector<std::vector<VegetativeBud>> axillary_vegetative_buds; //first index is petiole within internode, second index is bud within petiole
    std::vector<std::vector<FloralBud>> floral_buds; //first index is petiole within internode, second index is bud within petiole

    float internode_radius_initial;
    float internode_radius_max;
    float internode_length_max;

    bool build_context_geometry_petiole = true;
    bool build_context_geometry_peduncle = true;

protected:

    helios::vec3 inflorescence_bending_axis;

    helios::Context *context_ptr;

    PlantArchitecture *plantarchitecture_ptr;

    void updateInflorescence(FloralBud &fbud);

    [[nodiscard]] float calculatePhytomerConstructionCosts();
    [[nodiscard]] float calculateFlowerConstructionCosts(const FloralBud &fbud);
    [[nodiscard]] float calculateFruitConstructionCosts(const FloralBud &fbud);

    friend struct Shoot;
    friend class PlantArchitecture;

};

struct Shoot {

    Shoot(uint plant_ID, int shoot_ID, int parent_shoot_ID, uint parent_node, uint parent_petiole_index, uint rank, const helios::vec3 &shoot_base_position, const AxisRotation &shoot_base_rotation, uint current_node_number,
          float internode_length_shoot_initial, ShootParameters &shoot_params, std::string shoot_type_label, PlantArchitecture *plant_architecture_ptr);

    void buildShootPhytomers(float internode_radius, float internode_length, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper);

    //! Append a phytomer at the shoot apex
    /**
     * \param[in] internode_radius Initial radius of the internode
     * \param[in] internode_length_max Maximum length of the internode
     * \param[in] internode_length_scale_factor_fraction Fraction of the total fully-elongated internode length
     * \param[in] leaf_scale_factor_fraction Fraction of the total fully-elongated leaf scale factor (i.e., =1 for fully-elongated leaf)
     * \return Number of phytomers in the shoot after the new phytomer is appended
     */
    int appendPhytomer(float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, const PhytomerParameters &phytomer_parameters);

    //! Randomly sample the type of a child shoot based on the probabilities defined in the shoot parameters
    /**
     * \return Label of the randomly selected child shoot type.
     */
    [[nodiscard]] std::string sampleChildShootType() const;

    //! Randomly sample whether a vegetative bud should break into a new shoot
    /**
     * \param[in] node_index Index of the node along the shoot
     * \return True if the vegetative bud should break into a new shoot
     */
    [[nodiscard]] bool sampleVegetativeBudBreak( uint node_index ) const;

    //! Randomly sample whether the shoot should produce an epicormic shoot (water sprout) over timestep
    /**
     * \param[in] dt Time step in days
     * \param[out] epicormic_positions_fraction Vector of fractions of the shoot's length where epicormic shoots will be produced
     * \return Number of epicormic shoots to be produced; position of the epicormic shoot as a fraction of the shoot's length
     */
    uint sampleEpicormicShoot( float dt, std::vector<float> &epicormic_positions_fraction );

    /**
     * \brief Terminates the apical bud of the shoot.
     * This function marks the apical meristem of the shoot as no longer alive.
     */
    void terminateApicalBud();

    /**
     * \brief Terminates all axillary vegetative buds within the shoot.
     * This function iterates through all phytomers of the current shoot and marks the axillary vegetative buds as no longer alive.
     */
    void terminateAxillaryVegetativeBuds();

    /**
     * \brief Adds terminal floral buds to the shoot.
     * This function creates and initializes a specified number of terminal floral buds at the apex of the shoot.
     * Each bud is assigned its position, rotation, and bending axis, based on the shoot's parameters.
     * \note The number of floral buds is determined by the "max_terminal_floral_buds" parameter in the shoot's configuration,
     * and the parameter is resampled after creating the buds.
     */
    void addTerminalFloralBud();

    /**
     * \brief Calculates the total volume of the shoot's internode tubes.
     * This function iterates through all phytomers of the shoot and sums the volume
     * of the internode tube objects if they exist.
     * \return The total volume of the shoot's internode tubes.
     */
    [[nodiscard]] float calculateShootInternodeVolume() const;

    /**
     * \brief Calculates the total length of the shoot.
     * This function iterates through all phytomers in the shoot and sums their internode lengths.
     * \return Total length of the shoot.
     */
    [[nodiscard]] float calculateShootLength() const;

    /**
     * \brief Determines the shoot axis vector based on a specified fraction along the shoot's length.
     * \param[in] shoot_fraction A float value representing the fraction along the shoot's length (0 to 1).
     * \return A vec3 representing the direction vector of the shoot axis for the computed position.
     * \note The shoot_fraction parameter is clamped within the range of 0 to 1, with 0 corresponding to the base and 1 to the tip of the shoot. The function uses the closest phytomer's internode axis vector for calculation.
     */
    [[nodiscard]] helios::vec3 getShootAxisVector( float shoot_fraction ) const;

    /**
     * \brief Calculates the total leaf area of a shoot starting from a given node index.
     *
     * \param[in] start_node_index [optional] The index of the starting node in the shoot.
     * \return Total leaf area of the shoot and its child shoots starting from the given node.
     * \note Throws an error if the start_node_index is out of range.
     */
    [[nodiscard]] float sumShootLeafArea( uint start_node_index = 0 ) const;

    /**
     * \brief Calculates the total volume of all child shoots starting from a specified node index.
     * \param[in] start_node_index The starting index of the node from which to sum child volumes.
     * \return The total volume of all child shoots starting from the given node index.
     * \note Throws an error if start_node_index is out of range.
     */
    [[nodiscard]] float sumChildVolume( uint start_node_index = 0) const;

    void propagateDownstreamLeafArea(const Shoot* shoot, uint node_index, float leaf_area);

    /**
     * \brief Updates the shoot node positions and their associated geometries.
     * \param[in] update_context_geometry Indicates whether the geometry context should be updated for the shoot nodes.
     */
    void updateShootNodes(bool update_context_geometry = true );

    uint current_node_number = 0;
    uint nodes_this_season = 0;

    helios::vec3 base_position;
    AxisRotation base_rotation;
    helios::vec3 radial_outward_axis;

    const int ID;
    const int parent_shoot_ID;
    const uint plantID;
    const uint parent_node_index;
    const uint rank;
    const uint parent_petiole_index;

    float carbohydrate_pool_molC = 0;  // mol C
    float old_shoot_volume = 0;

    float phyllochron_increase = 2;
    float phyllochron_recovery = phyllochron_increase * 1.5;

    float elongation_decay = 0.5;
    float elongation_recovery = elongation_decay /1.5 ;

    uint days_with_negative_carbon_balance = 0;

    void breakDormancy();
    void makeDormant();

    bool isdormant;
    uint dormancy_cycles = 0;

    bool meristem_is_alive = true;

    float phyllochron_counter = 0;
    float phyllochron_min = 6;
    float elongation_max = .25;

    float curvature_perturbation = 0;
    float yaw_perturbation = 0;

    float gravitropic_curvature = 0;

    const float internode_length_max_shoot_initial;

    uint internode_tube_objID = 4294967294;

    std::vector<std::vector<helios::vec3>> shoot_internode_vertices; //first index is phytomer within shoot, second index is segment within phytomer internode tube
    std::vector<std::vector<float>> shoot_internode_radii; //first index is phytomer within shoot, second index is segment within phytomer internode tube

    bool build_context_geometry_internode = true;

    //map of node number (key) to IDs of shoot children (value)
    std::map<int,std::vector<int> > childIDs;

    ShootParameters shoot_parameters;

    std::string shoot_type_label;

    float phyllochron_instantaneous;
    float elongation_rate_instantaneous;

    std::vector<std::shared_ptr<Phytomer> > phytomers;

    PlantArchitecture* plantarchitecture_ptr;

    helios::Context *context_ptr;

};

struct PlantInstance{

    PlantInstance(const helios::vec3 &a_base_position, float a_current_age, const std::string &a_plant_name, helios::Context *a_context_ptr) : base_position(a_base_position), current_age(a_current_age), plant_name(a_plant_name), context_ptr(a_context_ptr) {}
    std::vector<std::shared_ptr<Shoot> > shoot_tree;
    helios::vec3 base_position;
    float current_age;
    float time_since_dormancy = 0;
    helios::Context *context_ptr;
    std::string plant_name;
    std::pair<std::string,float> epicormic_shoot_probability_perlength_per_day; //.first is the epicormic shoot label string, .second is the probability

    //Phenological thresholds
    float dd_to_dormancy_break = 0;
    float dd_to_flower_initiation = 0;
    float dd_to_flower_opening = 0;
    float dd_to_fruit_set = 0;
    float dd_to_fruit_maturity = 0;
    float dd_to_dormancy = 0;
    float max_leaf_lifespan = 1e6;
    bool is_evergreen = false;

    float max_age = 999;

};

class PlantArchitecture{
public:

    //! Main architectural model class constructor
    /**
     * \param[in] context_ptr Pointer to the Helios context.
     */
    explicit PlantArchitecture( helios::Context* context_ptr );

    //! Unit test routines
    static int selfTest();

    //! Add optional output object data values to the Context
    /**
     * \param[in] object_data_label Name of object data (e.g., "age", "rank")
    */
    void optionalOutputObjectData( const std::string &object_data_label );

    //! Add optional output object data values to the Context
    /**
     * \param[in] object_data_labels Vector of names of object data (e.g., {"age", "rank"})
    */
    void optionalOutputObjectData( const std::vector<std::string> &object_data_labels );

    // ********* Methods for Building Plants from Existing Library ********* //

    //! Load an existing plant model from the library
    /**
     * \param[in] plant_label User-defined label for the plant model to be loaded.
     */
    void loadPlantModelFromLibrary( const std::string &plant_label );

    //! Build a plant instance based on the model currently loaded from the library
    /**
     * \param[in] base_position Cartesian coordinates of the base of the plant.
     * \param[in] age Age of the plant.
     * \return ID of the plant instance.
     */
    uint buildPlantInstanceFromLibrary( const helios::vec3 &base_position, float age );

    //! Build a canopy of regularly spaced plants based on the model currently loaded from the library
    /**
     * \param[in] canopy_center_position Cartesian coordinates of the center of the canopy.
     * \param[in] plant_spacing_xy Spacing between plants in the canopy in the x- and y-directions.
     * \param[in] plant_count_xy Number of plants in the canopy in the x- and y-directions.
     * \param[in] age Age of the plants in the canopy.
     * \param[in] germination_rate [optional] Probability that a plant in the canopy germinates and a plant is created.
     * \return Vector of plant instance IDs.
     */
    std::vector<uint> buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &plant_spacing_xy, const helios::int2 &plant_count_xy, float age, float germination_rate = 1.f);

    //! Build a canopy of randomly scattered plants based on the model currently loaded from the library
    /**
     * \param[in] canopy_center_position Cartesian coordinates of the center of the canopy boundaries.
     * \param[in] canopy_extent_xy Size/extent of the canopy boundaries in the x- and y-directions.
     * \param[in] plant_count Number of plants to randomly generate inside canopy bounds.
     * \param[in] age Age of the plants in the canopy.
     * \return Vector of plant instance IDs.
     */
    std::vector<uint> buildPlantCanopyFromLibrary(const helios::vec3 &canopy_center_position, const helios::vec2 &canopy_extent_xy, uint plant_count, float age);
    
    //! Get the shoot parameters structure for a specific shoot type in the current plant model
    /**
     * \param[in] shoot_type_label User-defined label for the shoot type.
     * \return ShootParameters structure for the specified shoot type.
     */
    ShootParameters getCurrentShootParameters( const std::string &shoot_type_label );

    //! Get the shoot parameters structure for all shoot types in the current plant model
    /**
     * \return Map of shoot type labels to ShootParameters structures for all shoot types in the current plant model. The key is the user-defined label string for the shoot type, and the value is the corresponding ShootParameters structure.
     */
    std::map<std::string, ShootParameters> getCurrentShootParameters( );

    //! Get the phytomer parameters structure for all shoot types in the current plant model
    /**
     * \return Map of phytomer parameters for all type labels to ShootParameters structures for all shoot types in the current plant model. The key is the user-defined label string for the shoot type, and the value is the corresponding PhytomerParameters structure.
     */
    std::map<std::string, PhytomerParameters> getCurrentPhytomerParameters( );

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

    //! Delete an existing plant instance
    /**
     * \param[in] plantID ID of the plant instance to be deleted.
     */
    void deletePlantInstance(uint plantID);

    //! Delete multiple existing plant instances
    /**
     * \param[in] plantIDs IDs of the plant instances to be deleted.
     */
    void deletePlantInstance( const std::vector<uint> &plantIDs );

    //! Specify the threshold values for plant phenological stages
    /**
     * \param[in] plantID ID of the plant.
     * \param[in] time_to_dormancy_break Length of the dormancy period.
     * \param[in] time_to_flower_initiation Time from emergence/dormancy required to reach flower creation (closed flowers).
     * \param[in] time_to_flower_opening Time from flower initiation to flower opening.
     * \param[in] time_to_fruit_set Time from flower opening required to reach fruit set (i.e., flower dies and fruit is created).
     * \param[in] time_to_fruit_maturity Time from fruit set date required to reach fruit maturity.
     * \param[in] time_to_dormancy Time from emergence/dormancy break required to enter the next dormancy period.
     * \param[in] max_leaf_lifespan [optional] Maximum lifespan of a leaf in days.
     * \param[in] is_evergreen [optional] True if the plant is evergreen (i.e., does not lose all leaves during senescence).
     * \note Any phenological stage can be skipped by specifying a negative threshold value. In this case, the stage will be skipped and the threshold for the next stage will be relative to the previous stage.
     */
    void setPlantPhenologicalThresholds(uint plantID, float time_to_dormancy_break, float time_to_flower_initiation, float time_to_flower_opening, float time_to_fruit_set, float time_to_fruit_maturity, float time_to_dormancy, float max_leaf_lifespan = 1e6, bool is_evergreen= false);

    void disablePlantPhenology( uint plantID );

    //! Advance plant growth by a specified time interval for all plants
    /**
     * \param[in] time_step_days Time interval in days.
     */
    void advanceTime(float time_step_days);

    void accumulateHourlyLeafPhotosynthesis();

    //! Advance plant growth by a specified time interval for all plants
    /**
     * \param[in] time_step_years Number of years to advance.
     * \param[in] time_step_days Number of days to advance (added to number of years).
     */
    void advanceTime(int time_step_years, float time_step_days);

    //! Advance plant growth by a specified time interval for a single plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] time_step_days Time interval in days.
     */
    void advanceTime(uint plantID, float time_step_days);

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
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the base stem shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint addBaseStemShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction,
                          float radius_taper, const std::string &shoot_type_label);

    //! Manually append a new shoot at the end of an existing shoot. This is used when the characteristics of a shoot change along its length (e.g., from a unifoliate to trifoliate leaf).
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be appended.
     * \param[in] current_node_number Number of nodes/phytomers of the newly appended shoot.
     * \param[in] base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot relative to the parent shoot.
     * \param[in] internode_radius Initial radius of the internodes along the shoot.
     * \param[in] internode_length_max Length of the internode of the newly appended shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the new shoot to be used to reference it later.
     */
    uint appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction,
                     float radius_taper, const std::string &shoot_type_label);

    //! Manually add a child shoot at the axillary bud of a phytomer.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be added.
     * \param[in] parent_node_index Number of the node of the parent shoot at which the new shoot will be added.
     * \param[in] current_node_number Number of nodes of the newly added shoot.
     * \param[in] shoot_base_rotation AxisRotation object (pitch, yaw, roll) specifying the orientation of the base of the shoot.
     * \param[in] internode_radius Initial radius of the internodes along the shoot.
     * \param[in] internode_length_max Length of the internode of the newly appended shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \param[in] petiole_index [optional] Index of the petiole within the internode to which the new shoot will be attached (when there are multiple petioles per internode)
     * \return ID of the newly generated shoot.
     */
    uint addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node_index, uint current_node_number, const AxisRotation &shoot_base_rotation, float internode_radius, float internode_length_max,
                       float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper, const std::string &shoot_type_label, uint petiole_index = 0);

    //! Manually add a child epicormic shoot (water sprout) at an arbitrary position along the shoot
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] parent_shoot_ID ID of the shoot to which the new shoot will be added.
     * \param[in] parent_position_fraction Position along the parent shoot to add the epicormic shoot as a fraction of the parent shoot length.
     * \param[in] current_node_number Number of nodes of the newly added shoot.
     * \param[in] zenith_perturbation_degrees Pitch angle of epicormic shoot base away from parent shoot axis (degrees).
     * \param[in] internode_radius Initial radius of the internodes along the shoot.
     * \param[in] internode_length_max Length of the internode of the newly appended shoot.
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \param[in] radius_taper Tapering factor of the internode radius along the shoot (0=constant radius, 1=linear taper to zero radius).
     * \param[in] shoot_type_label Label of the shoot type to be used for the new shoot. This requires that the shoot type has already been defined using the defineShootType() method.
     * \return ID of the newly generated shoot.
     */
    uint addEpicormicShoot(uint plantID, int parent_shoot_ID, float parent_position_fraction, uint current_node_number, float zenith_perturbation_degrees, float internode_radius, float internode_length_max,
                           float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction, float radius_taper, const std::string &shoot_type_label);

    //! Add a new phytomer at the terminal bud of a shoot.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] shootID ID of the shoot to which the phytomer will be added
     * \param[in] phytomer_parameters Parameters of the phytomer to be added
     * \param[in] internode_radius Radius of the phytomer internode at the time of creation
     * \param[in] internode_length_max Maximum internode length at full elongation
     * \param[in] internode_length_scale_factor_fraction Scaling factor of the maximum internode length to determine the actual initial internode length at the time of creation (=1 applies no scaling).
     * \param[in] leaf_scale_factor_fraction Scaling factor of the leaf/petiole to determine the actual initial leaf size at the time of creation (=1 applies no scaling).
     * \return ID of generated phytomer
     */
    int appendPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_parameters, float internode_radius, float internode_length_max, float internode_length_scale_factor_fraction, float leaf_scale_factor_fraction);

    //! Enable shoot type to produce epicormic child shoots (water sprouts)
    /**
     * \note The probability that the shoot produces an epicormic shoot over period of dt is P = (epicormic_probability_perlength_perday * shoot_length * dt)*sin(shoot_inclinantion)
     * \param[in] plantID ID of the plant instance.
     * \param[in] epicormic_shoot_type_label Label of the shoot type corresponding to epicormic shoots.
     * \param[in] epicormic_probability_perlength_perday Probability of epicormic shoot formation per unit length of the parent shoot per day.
     */
    void enableEpicormicChildShoots(uint plantID, const std::string &epicormic_shoot_type_label, float epicormic_probability_perlength_perday);

    //! Do not build internode primitive geometry in the Context
    void disableInternodeContextBuild();

    //! Do not build petiole primitive geometry in the Context
    void disablePetioleContextBuild();

    //! Do not build peduncle primitive geometry in the Context
    void disablePeduncleContextBuild();

    //! Enable automatic removal of organs that are below the ground plane
    /**
     * \param[in] ground_height [optional] Height of the ground plane (default = 0).
     */
    void enableGroundClipping( float ground_height = 0.f );

    // -- methods for modifying the current plant state -- //

    /**
     * \brief Initializes the carbohydrate pool for all shoots of all plant instances
     * \param[in] carbohydrate_concentration_molC_m3 Concentration of carbohydrates in molC per cubic meter
     */
    void initializeCarbohydratePool(float carbohydrate_concentration_molC_m3);

    /**
     * \brief Initializes the carbohydrate pool for a specific plant.
     * \param[in] plantID Unique identifier of the plant.
     * \param[in] carbohydrate_concentration_molC_m3 Initial carbohydrate concentration in moles of carbon per cubic meter.
     * \note The plant with the specified ID must exist, and the carbohydrate concentration must be non-negative.
     */
    void initializePlantCarbohydratePool(uint plantID, float carbohydrate_concentration_molC_m3 );

    /**
     * \brief Initializes the carbohydrate pool for a specific shoot of a plant.
     *
     * \param[in] plantID Identifier for the plant whose shoot's carbohydrate pool is being initialized.
     * \param[in] shootID Identifier for the shoot of the plant.
     * \param[in] carbohydrate_concentration_molC_m3 Carbohydrate concentration in moles of carbon per cubic meter; must be non-negative.
     * \note Throws an error if the plant or shoot ID does not exist, or if the carbohydrate concentration is negative.
     */
    void initializeShootCarbohydratePool(uint plantID, uint shootID, float carbohydrate_concentration_molC_m3 );

    /**
     * \brief Adjusts the leaf scaling factor (length as a fraction of its maximal length) for a specific phytomer on a plant shoot.
     *
     * \param[in] plantID Identifier of the plant.
     * \param[in] shootID Identifier of the shoot on the specified plant.
     * \param[in] node_number Node number of the phytomer to adjust.
     * \param[in] leaf_scale_factor_fraction Fractional scaling factor for the leaf, must be in the range [0, 1].
     */
    void setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction);

    /**
     * \brief Sets the base position of a plant with the specified ID.
     * \param[in] plantID Unique identifier of the plant
     * \param[in] base_position Coordinates representing the new base position of the plant
     */
    void setPlantBasePosition(uint plantID, const helios::vec3 &base_position);

    /**
     * \brief Sets the leaf elevation angle distribution for a specific plant.
     *
     * This method modifies the elevation angles of leaves in the plant such that they follow a Beta distribution.
     * The methodology does not simply randomly sample angles from the Beta distribution, but it uses the Hungarian algorithm to minimize the total amount of rotation applied for the whole plant.
     * This makes the transformed plant look as similar as possible to the original plant while still following the specified distribution.
     * The more leaves in the plant the more accurate the distribution will be, and the more the plant will look like the original.
     *
     * \param[in] plantID Identifier for the plant.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     */
    void setPlantLeafElevationAngleDistribution(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination) const;

    /**
     * \brief Sets the leaf elevation angle distribution for a list of plants.
     *
     * This method modifies the elevation angles of leaves in the plants such that they follow a Beta distribution.
     * The methodology does not simply randomly sample angles from the Beta distribution, but it uses the Hungarian algorithm to minimize the total amount of rotation applied for the whole canopy.
     * This makes the transformed plants look as similar as possible to the original plants while still following the specified distribution.
     * The more leaves in the canopy the more accurate the distribution will be, and the more the plants will look like the originals.
     *
     * \param[in] plantIDs List of plant IDs for which to set the elevation angle distribution.
     * \param[in] Beta_mu_inclination Mean value parameter for the Beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the Beta distribution of inclination angles.
     */
    void setPlantLeafElevationAngleDistribution(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination) const;

    /**
     * \brief Sets the azimuth angle distribution for plant leaves.
     *
     * This method modifies the azimuth angles of leaves in the plant such that they follow an ellipsoidal distribution.
     * The methodology does not simply randomly sample angles from the ellipsoidal distribution, but it uses the Hungarian algorithm to minimize the total amount of rotation applied for the whole plant.
     * This makes the transformed plant look as similar as possible to the original plant while still following the specified distribution.
     * The more leaves in the plant the more accurate the distribution will be, and the more the plant will look like the original.
     *
     * \param[in] plantID Identifier of the plant whose leaf azimuth angle distribution is being set.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAzimuthAngleDistribution(uint plantID, float eccentricity, float ellipse_rotation_degrees) const;

    /**
     * \brief Sets the azimuth angle distribution of plant leaves.
     *
     * This method modifies the azimuth angles of leaves in the plants such that they follow a Beta distribution.
     * The methodology does not simply randomly sample angles from the Beta distribution, but it uses the Hungarian algorithm to minimize the total amount of rotation applied for the whole canopy.
     * This makes the transformed plants look as similar as possible to the original plants while still following the specified distribution.
     * The more leaves in the canopy the more accurate the distribution will be, and the more the plants will look like the originals.
     *
     * \param[in] plantIDs List of plant IDs to which the angle distribution will be applied.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAzimuthAngleDistribution(const std::vector<uint> &plantIDs, float eccentricity, float ellipse_rotation_degrees) const;

    /**
     * \brief Sets the leaf angle distribution (both elevation and azimuth) for a specific plant
     *
     * This method modifies the elevation angles of leaves in the plant such that they follow a Beta distribution, and the azimuth angles such that they follow an ellipsoidal distribution.
     * The methodology does not simply randomly sample angles from the distribution, but it uses the Hungarian algorithm to minimize the total amount of rotation applied for the whole plant.
     * This makes the transformed plant look as similar as possible to the original plant while still following the specified distribution.
     * The more leaves in the plant the more accurate the distribution will be, and the more the plant will look like the original.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] Beta_mu_inclination The mean inclination angle parameter (Beta distribution).
     * \param[in] Beta_nu_inclination The shape parameter nu of the distribution (Beta distribution).
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAngleDistribution(uint plantID, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees) const;

    /**
     * \brief Sets the leaf angle distribution (both elevation and azimuth) for a list of specified plants
     *
     * This method modifies the elevation angles of leaves in the plants such that they follow a Beta distribution, and the azimuth angles such that they follow an ellipsoidal distribution.
     * The methodology does not simply randomly sample angles from the distribution, but it uses the Hungarian algorithm to minimize the total amount of rotation applied for the whole canopy.
     * This makes the transformed plants look as similar as possible to the original plants while still following the specified distribution.
     * The more leaves in the canopy the more accurate the distribution will be, and the more the plants will look like the originals.
     *
     * \param[in] plantIDs Vector of plant IDs to which the leaf angle distribution is to be applied.
     * \param[in] Beta_mu_inclination Mean parameter for the beta distribution of inclination angles.
     * \param[in] Beta_nu_inclination Shape parameter for the beta distribution of inclination angles.
     * \param[in] eccentricity Eccentricity value for the ellipse defining the azimuth distribution.
     * \param[in] ellipse_rotation_degrees Rotation angle of the ellipse in degrees.
     */
    void setPlantLeafAngleDistribution(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity, float ellipse_rotation_degrees) const;

    void setPlantAge(uint plantID, float current_age);

    /**
     * \brief Harvests a plant by removing all leaves and fruit.
     *
     * \param[in] plantID The unique identifier of the plant to be harvested.
     */
    void harvestPlant(uint plantID);

    /**
     * \brief Removes all leaves from a specified shoot in a specified plant.
     *
     * \param[in] plantID ID of the plant from which leaves are to be removed.
     * \param[in] shootID ID of the shoot within the plant whose leaves are to be removed.
     */
    void removeShootLeaves(uint plantID, uint shootID);

    /**
     * \brief Removes all leaves from the plant with the specified ID.
     *
     * \param[in] plantID The unique identifier of the plant whose leaves are to be removed.
     */
    void removePlantLeaves(uint plantID );

    /**
     * \brief Makes the specified plant enter a dormant state.
     *
     * \param[in] plantID ID of the plant to be made dormant.
     */
    void makePlantDormant( uint plantID );

    /**
     * \brief Breaks the dormancy of all shoots in the specified plant.
     *
     * \param[in] plantID Identifier of the plant whose shoots' dormancy should be broken.
     */
    void breakPlantDormancy( uint plantID );

    /**
     * \brief Prunes a branch from a specific plant at a designated node index.
     *
     * \param[in] plantID Unique identifier of the plant to prune.
     * \param[in] shootID Identifier of the shoot to prune from within the plant.
     * \param[in] node_index Index of the node on the shoot where the branch will be pruned.
     */
    void pruneBranch(uint plantID, uint shootID, uint node_index);

    // -- methods for querying information about the plant -- //

    /**
     * \brief Retrieves the name of the plant associated with a given plant ID.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return The name of the plant corresponding to the provided plant ID.
     */
    [[nodiscard]] std::string getPlantName(uint plantID) const;

    /**
     * \brief Retrieves the age of a specific plant.
     *
     * \param[in] plantID Unique identifier of the plant.
     * \return The age of the plant in days associated with the given plantID.
     */
    [[nodiscard]] float getPlantAge(uint plantID) const;

    /**
     * \brief Retrieves the number of nodes in a specific shoot of a specific plant.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The index of the shoot within the plant.
     * \return The current number of nodes in the specified shoot.
     */
    [[nodiscard]] uint getShootNodeCount( uint plantID, uint shootID ) const;

    /**
     * \brief Retrieves the taper of a shoot based on its radius measurements.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \param[in] shootID The unique identifier of the shoot within the specified plant.
     * \return The taper of the shoot as a float value, constrained between 0 and 1.
     */
    [[nodiscard]] float getShootTaper( uint plantID, uint shootID ) const;

    /**
     * \brief Retrieves the base position of the specified plant.
     *
     * \param[in] plantID Identifier of the plant whose base position is being queried.
     * \return Base position of the plant as a helios::vec3 object.
     */
    [[nodiscard]] helios::vec3 getPlantBasePosition(uint plantID) const;

    /**
     * \brief Retrieves the base positions of multiple plants.
     *
     * \param[in] plantIDs A vector containing the IDs of the plants for which the base positions are required.
     * \return A vector of vec3 objects representing the base positions of the specified plants.
     */
    [[nodiscard]] std::vector<helios::vec3> getPlantBasePosition(const std::vector<uint> &plantIDs) const;

    //! Sum the one-sided leaf area of all leaves in the plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Total one-sided leaf area of all leaves in the plant.
     */
    [[nodiscard]] float sumPlantLeafArea(uint plantID) const;

    //! Calculate the height of the last internode on the base stem/shoot
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Height of the last internode on the base stem/shoot.
     */
    [[nodiscard]] float getPlantStemHeight(uint plantID) const;

    //! Calculate the height of the highest element in the plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Height of the highest element in the plant.
     */
    [[nodiscard]] float getPlantHeight(uint plantID) const;

    //! Calculate the leaf inclination angle distribution of all leaves in the plant.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf inclination angles. Bins are evenly spaced between 0 and 90 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafInclinationAngleDistribution(uint plantID, uint Nbins, bool normalize = true) const;

    //! Calculate the leaf inclination angle distribution of all leaves in multiple plants.
    /**
     * \param[in] plantIDs Vector of IDs of the plant instances.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf inclination angles. Bins are evenly spaced between 0 and 90 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafInclinationAngleDistribution(const std::vector<uint> &plantIDs, uint Nbins, bool normalize = true) const;

    //! Calculate the leaf azimuth angle distribution of all leaves in the plant.
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf azimuth angles. Bins are evenly spaced between 0 and 360 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafAzimuthAngleDistribution(uint plantID, uint Nbins, bool normalize = true) const;

    //! Calculate the leaf azimuth angle distribution of all leaves in multiple plants.
    /**
     * \param[in] plantIDs Vector of ID of the plant instances.
     * \param[in] Nbins Number of bins for the histogram.
     * \param[in] normalize [optional] Normalize the histogram (default = true).
     * \return Histogram of leaf azimuth angles. Bins are evenly spaced between 0 and 360 degrees.
     */
    [[nodiscard]] std::vector<float> getPlantLeafAzimuthAngleDistribution(const std::vector<uint> &plantIDs, uint Nbins, bool normalize = true) const;

    //! Get the total number of leaves on the plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Total number of leaves on the plant.
     */
    [[nodiscard]] uint getPlantLeafCount(uint plantID) const;

    //! Get the base positions of all leaves on the plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of base positions of all leaves on the plant.
     */
    [[nodiscard]] std::vector<helios::vec3> getPlantLeafBases(uint plantID) const;

    //! Get the base positions of all leaves for a list of plants
    /**
     * \param[in] plantIDs List of IDs of the plant instances.
     * \return Vector of base positions of all leaves on the plants.
     */
    [[nodiscard]] std::vector<helios::vec3> getPlantLeafBases(const std::vector<uint> &plantIDs) const;

    //! Write all vertices in the plant to a file for external processing (e.g., bounding volume, convex hull)
    /**
     * \param[in] plantID ID of the plant instance.
     * \param[in] filename Name/path of the output file.
     */
    void writePlantMeshVertices(uint plantID, const std::string &filename) const;

    //! Get IDs for all plant instances
    /**
     * \return Vector of plant IDs for all plant instances
     */
    [[nodiscard]] std::vector<uint> getAllPlantIDs() const;

    //! Get object IDs for all organs objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all organs in the plant.
     */
    [[nodiscard]] std::vector<uint> getAllPlantObjectIDs(uint plantID) const;

    //! Get primitive UUIDs for all primitives in a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of primitive UUIDs for all primitives in the plant.
     */
    [[nodiscard]] std::vector<uint> getAllPlantUUIDs(uint plantID) const;

    //! Get object IDs for all internode (Tube) objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all internodes in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantInternodeObjectIDs(uint plantID) const;

    //! Get object IDs for all petiole (Tube) objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all petioles in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantPetioleObjectIDs(uint plantID) const;

    //! Get object IDs for all leaf objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all leaves in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantLeafObjectIDs(uint plantID) const;

    //! Get object IDs for all leaf objects for a list of plants
    /**
     * \param[in] plantIDs List of IDs of the plant instances.
     * \return Vector of object IDs for all leaves in the plants.
     */
    [[nodiscard]] std::vector<uint> getPlantLeafObjectIDs(const std::vector<uint> &plantIDs) const;

    //! Get object IDs for all peduncle (Tube) objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all peduncles in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantPeduncleObjectIDs(uint plantID) const;

    //! Get object IDs for all inflorescence objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all inflorescences in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantFlowerObjectIDs(uint plantID) const;

    //! Get object IDs for all fruit objects for a given plant
    /**
     * \param[in] plantID ID of the plant instance.
     * \return Vector of object IDs for all fruits in the plant.
     */
    [[nodiscard]] std::vector<uint> getPlantFruitObjectIDs(uint plantID) const;

    //! Get UUIDs for all existing plant primitives
    /**
     * \return Vector of UUIDs for all plant primitives.
     */
    [[nodiscard]] std::vector<uint> getAllUUIDs() const;

    //! Get UUIDs for all existing leaf primitives
    /**
     * \return Vector of UUIDs for all leaf primitives.
     */
    [[nodiscard]] std::vector<uint> getAllLeafUUIDs() const;

    //! Get UUIDs for all existing internode primitives
    /**
     * \return Vector of UUIDs for all internode primitives.
     */
    [[nodiscard]] std::vector<uint> getAllInternodeUUIDs() const;

    //! Get UUIDs for all existing petiole primitives
    /**
     * \return Vector of UUIDs for all petiole primitives.
     */
    [[nodiscard]] std::vector<uint> getAllPetioleUUIDs() const;

    //! Get UUIDs for all existing peduncle primitives
    /**
     * \return Vector of UUIDs for all peduncle primitives.
     */
    [[nodiscard]] std::vector<uint> getAllPeduncleUUIDs() const;

    //! Get UUIDs for all existing flower primitives
    /**
     * \return Vector of UUIDs for all flower primitives.
     */
    [[nodiscard]] std::vector<uint> getAllFlowerUUIDs() const;

    //! Get UUIDs for all existing fruit primitives
    /**
     * \return Vector of UUIDs for all fruit primitives.
     */
    [[nodiscard]] std::vector<uint> getAllFruitUUIDs() const;

    //! Get object IDs for all existing plant compound objects
    /**
     * \return Vector of object IDs for all plant compound objects.
     */
    [[nodiscard]] std::vector<uint> getAllObjectIDs() const;

    // -- carbohydrate model -- //

    /**
     * \brief Enables the carbohydrate model in the plant architecture development.
     */
    void enableCarbohydrateModel();

    /**
     * \brief Disables the carbohydrate model
     */
    void disableCarbohydrateModel();

    // -- manual plant generation from input string -- //

    /**
     * \brief Retrieves a string representation of a plant based on its ID.
     *
     * \param[in] plantID The unique identifier of the plant.
     * \return A string encoding of the plant structure.
     */
    [[nodiscard]] std::string getPlantString(uint plantID) const;

    /**
     * \brief Generates a plant model based on the given generation string and phytomer parameters.
     *
     * \param[in] generation_string Input string describing the plant generation rules and structure.
     * \param[in] phytomer_parameters Parameters that define the characteristics of a single phytomer.
     * \return The total number of phytomers generated in the plant architecture.
     */
    uint generatePlantFromString(const std::string &generation_string, const PhytomerParameters &phytomer_parameters);

    /**
     * \brief Generates a plant based on the provided description string and phytomer parameters.
     *
     * \param[in] generation_string A string encoding of the plant structure.
     * \param[in] phytomer_parameters A map containing parameter configurations for each type of phytomer.
     * \return A unique identifier for the generated plant.
     * \note The input string must begin with '{', and valid phytomer parameters must be provided.
     */
    uint generatePlantFromString(const std::string &generation_string, const std::map<std::string,PhytomerParameters> &phytomer_parameters);

    /**
     * \brief Writes the structure of a plant instance to an XML file.
     *
     * \param[in] plantID The unique identifier for the plant instance.
     * \param[in] filename Path to the XML file where the plant structure will be saved.
     * \note The function checks if the plant instance exists and if the output file path
     * is valid and writable. Errors related to invalid plant ID or file issues will throw exceptions.
     */
    void writePlantStructureXML(uint plantID, const std::string &filename) const;

    /**
     * \brief Reads plant structure data from an XML file.
     *
     * Parses the specified XML file containing plant architecture information and extracts
     * relevant data.
     *
     * \param[in] filename The path to the XML file to load.
     * \param[in] quiet [optional] If true, suppresses console output of status messages.
     * \return A vector of unsigned integers representing the plant IDs parsed from the XML file.
     * \note Throws an exception if the file cannot be parsed or is missing required tags.
     */
    std::vector<uint> readPlantStructureXML( const std::string &filename, bool quiet = false);

    friend struct Phytomer;
    friend struct Shoot;

protected:

    helios::Context* context_ptr;

    std::minstd_rand0 *generator = nullptr;

    uint plant_count = 0;

    std::string current_plant_model;

    std::map<uint,PlantInstance> plant_instances;

    [[nodiscard]] std::string makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> & shoot_tree) const;

    std::map<std::string,ShootParameters> shoot_types;

    // Key is the prototype function pointer; value first index is the unique leaf prototype, second index is the leaflet along a compound leaf (if applicable)
    // std::map<uint(*)(helios::Context* context_ptr, LeafPrototype* prototype_parameters, int compound_leaf_index),std::vector<std::vector<uint>> > unique_leaf_prototype_objIDs;
    std::map<uint,std::vector<std::vector<uint>> > unique_leaf_prototype_objIDs;

    // Key is the prototype function pointer; value index is the unique flower prototype
    std::map<uint(*)(helios::Context* context_ptr, uint subdivisions, bool flower_is_open),std::vector<uint> > unique_open_flower_prototype_objIDs;
    // Key is the prototype function pointer; value index is the unique flower prototype
    std::map<uint(*)(helios::Context* context_ptr, uint subdivisions, bool flower_is_open),std::vector<uint> > unique_closed_flower_prototype_objIDs;
    // Key is the prototype function pointer; value index is the unique fruit prototype
    std::map<uint(*)(helios::Context* context_ptr, uint subdivisions),std::vector<uint> > unique_fruit_prototype_objIDs;

    bool build_context_geometry_internode = true;
    bool build_context_geometry_petiole = true;
    bool build_context_geometry_peduncle = true;

    float ground_clipping_height = -99999;

    void validateShootTypes( ShootParameters &shoot_parameters ) const;

    void parseStringShoot(const std::string &LString_shoot, uint plantID, int parentID, uint parent_node, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters);

    void parseShootArgument(const std::string &shoot_argument, const std::map<std::string, PhytomerParameters> &phytomer_parameters, ShootParameters &shoot_parameters, AxisRotation &base_rotation, std::string &phytomer_label);

    void parseInternodeArgument(const std::string &internode_argument, float &internode_radius, float &internode_length, PhytomerParameters &phytomer_parameters);

    void parsePetioleArgument(const std::string& petiole_argument, PhytomerParameters &phytomer_parameters );

    void parseLeafArgument(const std::string& leaf_argument, PhytomerParameters &phytomer_parameters );

//    void shiftDownstreamShoots(uint plantID, std::vector<std::shared_ptr<Shoot>> &shoot_tree, std::shared_ptr<Shoot> parent_shoot_ptr, const helios::vec3 &base_position );

    void initializeDefaultShoots( const std::string &plant_label );

    bool detectGroundCollision(uint objID);

    bool detectGroundCollision(const std::vector<uint> &objID);

    void setPlantLeafAngleDistribution_private(const std::vector<uint> &plantIDs, float Beta_mu_inclination, float Beta_nu_inclination, float eccentricity_azimuth, float
                                               ellipse_rotation_azimuth_degrees, bool set_elevation, bool set_azimuth) const;

    //! Names of additional object data to add to the Context
    std::map<std::string,bool> output_object_data;

    // --- Plant Growth --- //

    void incrementPhytomerInternodeGirth(uint plantID, uint shootID, uint node_number, bool update_context_geometry);

    // --- Carbohydrate Model --- //

    void accumulateShootPhotosynthesis();

    void subtractShootMaintenanceCarbon(float dt );
    void subtractShootGrowthCarbon();

    void checkCarbonPool_abortOrgans();
    void checkCarbonPool_adjustPhyllochron();
    void checkCarbonPool_transferCarbon();

    bool carbon_model_enabled = false;

    // --- Plant Library --- //

    void initializeAlmondTreeShoots();

    uint buildAlmondTree( const helios::vec3 &base_position );

    void initializeAppleTreeShoots();

    uint buildAppleTree( const helios::vec3 &base_position );

    void initializeAsparagusShoots();

    uint buildAsparagusPlant( const helios::vec3 &base_position );

    void initializeBindweedShoots();

    uint buildBindweedPlant( const helios::vec3 &base_position );

    void initializeBeanShoots();

    uint buildBeanPlant( const helios::vec3 &base_position );

    void initializeCapsicumShoots();

    uint buildCapsicumPlant( const helios::vec3 &base_position );

    void initializeCheeseweedShoots();

    uint buildCheeseweedPlant( const helios::vec3 &base_position );

    void initializeCowpeaShoots();

    uint buildCowpeaPlant( const helios::vec3 &base_position );

    void initializeGrapevineVSPShoots();

    uint buildGrapevineVSP( const helios::vec3 &base_position );

    void initializeGroundCherryWeedShoots();

    uint buildGroundCherryWeedPlant( const helios::vec3 &base_position );

    void initializeMaizeShoots();

    uint buildMaizePlant( const helios::vec3 &base_position );

    void initializeOliveTreeShoots();

    uint buildOliveTree( const helios::vec3 &base_position );

    void initializePistachioTreeShoots();

    uint buildPistachioTree( const helios::vec3 &base_position );

    void initializePuncturevineShoots();

    uint buildPuncturevinePlant( const helios::vec3 &base_position );

    void initializeEasternRedbudShoots();

    uint buildEasternRedbudPlant( const helios::vec3 &base_position );

    void initializeRiceShoots();

    uint buildRicePlant( const helios::vec3 &base_position );

    void initializeButterLettuceShoots();

    uint buildButterLettucePlant(const helios::vec3 &base_position );

    void initializeSoybeanShoots();

    uint buildSoybeanPlant( const helios::vec3 &base_position );

    void initializeSorghumShoots();

    uint buildSorghumPlant( const helios::vec3 &base_position );

    void initializeStrawberryShoots();

    uint buildStrawberryPlant( const helios::vec3 &base_position );

    void initializeSugarbeetShoots();

    uint buildSugarbeetPlant( const helios::vec3 &base_position );

    void initializeTomatoShoots();

    uint buildTomatoPlant( const helios::vec3 &base_position );

    void initializeWalnutTreeShoots();

    uint buildWalnutTree( const helios::vec3 &base_position );

    void initializeWheatShoots();

    uint buildWheatPlant( const helios::vec3 &base_position );


};

#include "Assets.h"

#endif //PLANT_ARCHITECTURE
