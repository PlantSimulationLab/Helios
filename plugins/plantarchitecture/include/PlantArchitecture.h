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

enum GrowthState{
    DORMANT = 0,
    VEGETATIVE = 1, //has active leaf, buds are viable
    FLOWERING = 2, //has active leaf, flowers are open
    FRUITING = 3, //has active leaf, fruit is growing
    GROWING = 4, //no leaves, buds are gone, but internode can increase girth
    DEAD = 5 //no leaves, no buds, no internode growth
};

std::vector<uint> makeTubeFromCones(uint Ndivs, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr);

struct PhytomerParameters{
private:

    struct InternodeParameters{
        helios::vec3 origin;
        RandomParameter_float pitch;
        float radius;
        RandomParameter_float length;
        RandomParameter_float curvature;
        uint petioles_per_internode;
        helios::RGBcolor color;
        uint tube_subdivisions;
        InternodeParameters& operator=(const InternodeParameters &a){
            this->origin = a.origin;
            this->pitch = a.pitch;
            this->pitch.resample();
            this->radius = a.radius;
            this->length = a.length;
            this->length.resample();
            this->curvature = a.curvature;
            this->curvature.resample();
            this->petioles_per_internode = a.petioles_per_internode;
            this->color = a.color;
            this->tube_subdivisions = a.tube_subdivisions;
            return *this;
        }
    };

    struct PetioleParameters{
        RandomParameter_float pitch;
        RandomParameter_float yaw;
        RandomParameter_float roll;
        RandomParameter_float radius;
        RandomParameter_float length;
        RandomParameter_float curvature;
        RandomParameter_float taper;
        uint tube_subdivisions;
        int leaves_per_petiole;
        PetioleParameters& operator=(const PetioleParameters &a){
            this->pitch = a.pitch;
            this->pitch.resample();
            this->yaw = a.yaw;
            this->yaw.resample();
            this->roll = a.roll;
            this->roll.resample();
            this->radius = a.radius;
            this->radius.resample();
            this->length = a.length;
            this->length.resample();
            this->curvature = a.curvature;
            this->curvature.resample();
            this->taper = a.taper;
            this->taper.resample();
            this->tube_subdivisions = a.tube_subdivisions;
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
        helios::vec3 prototype_scale;
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
        uint tube_subdivisions;
        RandomParameter_float fruit_pitch;
        RandomParameter_float fruit_roll;
        bool requires_dormancy;
        helios::vec3 fruit_prototype_scale;
        helios::vec3 flower_prototype_scale;
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
            this->tube_subdivisions = a.tube_subdivisions;
            this->fruit_pitch = a.fruit_pitch;
            this->fruit_pitch.resample();
            this->fruit_roll = a.fruit_roll;
            this->fruit_roll.resample();
            this->requires_dormancy = a.requires_dormancy;
            this->fruit_prototype_scale = a.fruit_prototype_scale;
            this->fruit_prototype_function = a.fruit_prototype_function;
            this->flower_prototype_scale = a.flower_prototype_scale;
            this->flower_prototype_function = a.flower_prototype_function;
            return *this;
        }
    };

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

};

struct ShootParameters{

    //! Default constructor - does not set random number generator
    ShootParameters();

    //! Constructor - sets random number generator
    explicit ShootParameters( std::minstd_rand0 *generator );

    PhytomerParameters phytomer_parameters;

    uint max_nodes;

    RandomParameter_float shoot_internode_taper;

    RandomParameter_float phyllochron; //phytomers/day
    RandomParameter_float growth_rate; //length/day

    // Probability that bud with this shoot type will break and form a new shoot
    float bud_break_probability;

    // Probability that a phytomer will flower
    float flower_probability;

    // Probability that a phytomer will fruit
    float fruit_probability;

    RandomParameter_float bud_time;  //days

    RandomParameter_float child_insertion_angle;

    std::vector<float> blind_nodes;

    void defineChildShootTypes( const std::vector<std::string> &child_shoot_type_labels, const std::vector<float> &child_shoot_type_probabilities );

    //\todo These should be private
    std::vector<std::string> child_shoot_type_labels;
    std::vector<float> child_shoot_type_probabilities;

private:


};

struct Phytomer {
public:

    Phytomer(const PhytomerParameters &params, const ShootParameters &parent_shoot_parameters, uint phytomer_index, const helios::vec3 &parent_internode_axis, const helios::vec3 &parent_petiole_axis, const AxisRotation &shoot_base_rotation,
             float internode_scale_factor_fraction, float leaf_scale_factor_fraction, uint rank, helios::Context *context_ptr);

    helios::vec3 getInternodeAxisVector( float stem_fraction ) const;

    helios::vec3 getPetioleAxisVector( float stem_fraction ) const;

    helios::vec3 getAxisVector( float stem_fraction, const std::vector<helios::vec3> &axis_vertices ) const;

    float getInternodeLength() const;

    float getPetioleLength() const;

    float getInternodeRadius( float stem_fraction ) const;

    float getPetioleRadius( float stem_fraction ) const;

    void addInfluorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation, const helios::vec3 &inflorescence_bending_axis);

    void scaleInternode( float girth_scale_factor, float length_scale_factor );

    void setInternodeScale( float internode_scale_factor_fraction );

    void setLeafScale( float leaf_scale_factor_fraction );

    void setPhytomerScale(float internode_scale_factor_fraction, float leaf_scale_factor_fraction);

    void setInflorescenceScale( float inflorescence_scale_factor_fraction );

    void setPetioleBase( const helios::vec3 &base_position );

    void setPhytomerBase( const helios::vec3 &base_position );

    void changeReproductiveState( GrowthState state );

    void removeLeaf();

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

    PhytomerParameters phytomer_parameters;

    uint rank;

    float age = 0;
    float inflorescence_age = 0;

    float current_internode_scale_factor = 1;
    float current_leaf_scale_factor = 1;
    float current_inflorescence_scale_factor = 1;

    GrowthState state = DORMANT;

    helios::Context *context_ptr;

};

struct Shoot{

    Shoot(int ID, int parentID, uint parent_node, uint rank, const helios::vec3 &origin, const AxisRotation &shoot_base_rotation, uint current_node_number, ShootParameters shoot_params, const std::string &shoot_type_label,
          std::vector<std::shared_ptr<Shoot> > *shoot_tree_ptr, helios::Context *context_ptr);

    void initializePhytomer();

    int addPhytomer(const PhytomerParameters &params, const AxisRotation &shoot_base_rotation, float internode_scale_factor_fraction, float leaf_scale_factor_fraction);

    void breakDormancy();

    uint current_node_number;

    helios::vec3 origin;

    AxisRotation base_rotation;

    int ID;
    int parentID;
    uint parentNode;
    uint rank;

    float assimilate_pool;  // mg SC/g DW

    float phyllochron_counter = 0;

    bool dormant = true;

    //map of node number to ID of shoot child
    std::map<int,int> childIDs;

    ShootParameters shoot_parameters;

    std::string shoot_type_label;

    std::vector<std::shared_ptr<Phytomer> > phytomers;

    std::vector<std::shared_ptr<Shoot> > *shoot_tree_ptr;

    helios::Context *context_ptr;

};

class PlantArchitecture{
public:

    explicit PlantArchitecture( helios::Context* context_ptr );

    uint addPlantInstance(const helios::vec3 &base_position, float current_age);

    uint duplicatePlantInstance(uint plantID, const helios::vec3 &base_position, float current_age );

    PhytomerParameters getPhytomerParametersFromLibrary(const std::string &phytomer_label );

    void setPlantPhenologicalThresholds(uint plantID, float assimilate_dormancy_threshold, float dd_to_flowering, float dd_to_fruit_set, float dd_to_fruit_maturity, float dd_to_senescence);

    void advanceTime( float dt );

    void incrementAssimilatePool( uint plantID, uint shootID, float assimilate_increment_mg_g );

    void incrementAssimilatePool( uint plantID, float assimilate_increment_mg_g );

    // -- plant building methods -- //

    void defineShootType( const std::string &shoot_type_label, const ShootParameters &shoot_params );

    uint addBaseShoot(uint plantID, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label);

    uint appendShoot(uint plantID, int parent_shoot_ID, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label);

    uint addChildShoot(uint plantID, int parent_shoot_ID, uint parent_node, uint current_node_number, const AxisRotation &base_rotation, const std::string &shoot_type_label);

    int addPhytomerToShoot(uint plantID, uint shootID, const PhytomerParameters &phytomer_params, float internode_scale_factor_fraction, float leaf_scale_factor_fraction);

    void scalePhytomerInternode(uint plantID, uint shootID, uint node_number, float girth_scale_factor, float length_scale_factor);

    void setPhytomerInternodeScale(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction);

    void setPhytomerLeafScale(uint plantID, uint shootID, uint node_number, float leaf_scale_factor_fraction);

    void setPhytomerScale(uint plantID, uint shootID, uint node_number, float internode_scale_factor_fraction, float leaf_scale_factor_fraction);

    void setPlantBasePosition(uint plantID, const helios::vec3 &base_position);

    helios::vec3 getPlantBasePosition(uint plantID) const;

    void setPlantAge(uint plantID, float current_age);

    float getPlantAge(uint plantID) const;

    uint getShootNodeCount( uint plantID, uint shootID ) const;

    std::vector<uint> getAllPlantObjectIDs(uint plantID) const;

    std::vector<uint> getAllPlantUUIDs(uint PlantID) const;

    std::string getLSystemsString(uint plantID) const;

    bool sampleChildShootType( uint plantID, uint shootID, std::string &child_shoot_type_label ) const;

    void addAlmondShoot();

private:

    helios::Context* context_ptr;

    std::minstd_rand0 *generator = nullptr;

    uint plant_count = 0;

    struct PlantInstance{
        PlantInstance(const helios::vec3 &a_base_position, float a_current_age) : base_position(a_base_position), current_age(a_current_age) {}
        std::vector<std::shared_ptr<Shoot> > shoot_tree;
        helios::vec3 base_position;
        float current_age;

        //Phenological thresholds
        float assimilate_dormancy_threshold = 0;
        float dd_to_flowering = 0;
        float dd_to_fruit_set = 0;
        float dd_to_fruit_maturity = 0;
        float dd_to_senescence = 0;
    };

    std::map<uint,PlantInstance> plant_instances;

    std::string makeShootString(const std::string &current_string, const std::shared_ptr<Shoot> &shoot, const std::vector<std::shared_ptr<Shoot>> & shoot_tree) const;

    std::map<std::string,ShootParameters> shoot_types;

    void accumulateShootPhotosynthesis( float dt );

};


#endif //PLANT_ARCHITECTURE
