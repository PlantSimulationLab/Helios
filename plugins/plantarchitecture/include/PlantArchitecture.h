/** \file "PlantArchitecture.h" Primary header file for plant architecture plug-in.
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
        generator = rand_generator;
        sampled = false;
    }

    void initialize( std::minstd_rand0 *rand_generator){
        constval = 1.f;
        generator = rand_generator;
        sampled = false;
    }

    RandomParameter_float& operator=(float a){
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
        distribution = "normal";
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

private:
    bool sampled;
    float constval;
    std::string distribution;
    std::vector<float> distribution_parameters;
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
        generator = rand_generator;
        sampled = false;
    }

    void initialize( std::minstd_rand0 *rand_generator){
        constval = 1;
        generator = rand_generator;
        sampled = false;
    }

    RandomParameter_int& operator=(int a){
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
                constval = distribution_parameters.at(0) + unif_distribution(*generator) * (distribution_parameters.at(1) - distribution_parameters.at(0));
            }
        }
        return constval;
    }

private:
    bool sampled;
    int constval;
    std::string distribution;
    std::vector<int> distribution_parameters;
    std::uniform_int_distribution<int> unif_distribution;
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

};

inline AxisRotation make_AxisRotation( float a_pitch, float a_yaw, float a_roll ) {
    return {a_pitch,a_yaw,a_roll};
}

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
        uint reproductive_state;
        float length;
        int fruit_per_inflorescence;
        float fruit_offset;
        float rachis_radius;
        float curvature;
        std::string fruit_arrangement_pattern;
        uint tube_subdivisions;
        AxisRotation fruit_rotation;
        helios::vec3 fruit_prototype_scale;
        uint(*fruit_prototype_function)( helios::Context*, uint subdivisions, int flag ) = nullptr;
        uint(*flower_prototype_function)( helios::Context*, uint subdivisions, int flag ) = nullptr;
        InflorescenceParameters& operator=(const InflorescenceParameters &a){
            this->reproductive_state = a.reproductive_state;
            this->length = a.length;
            this->fruit_per_inflorescence = a.fruit_per_inflorescence;
            this->fruit_offset = a.fruit_offset;
            this->rachis_radius = a.rachis_radius;
            this->curvature = a.curvature;
            this->fruit_arrangement_pattern = a.fruit_arrangement_pattern;
            this->tube_subdivisions = a.tube_subdivisions;
            this->fruit_rotation = a.fruit_rotation;
            this->fruit_prototype_scale = a.fruit_prototype_scale;
            this->fruit_prototype_function = a.fruit_prototype_function;
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

    PhytomerParameters& operator=(const PhytomerParameters& a){
        this->internode = a.internode;
        this->petiole = a.petiole;
        this->leaf = a.leaf;
        this->inflorescence = a.inflorescence;
        return *this;
    }

};

struct ShootParameters{

    ShootParameters();

    uint max_nodes;

    float shoot_internode_taper;

    float phyllochron; //phytomers/day
    float growth_rate; //length/day

    float bud_probability;
    float bud_time;  //days

};

struct Phytomer {
public:

    Phytomer(const PhytomerParameters &params, uint phytomer_index, const helios::vec3 &parent_internode_axis,
             const helios::vec3 &parent_petiole_axis, const AxisRotation &shoot_base_rotation, float scale, uint rank,
             helios::Context *context_ptr);

    helios::vec3 getInternodeAxisVector( float stem_fraction ) const;

    helios::vec3 getPetioleAxisVector( float stem_fraction ) const;

    helios::vec3 getAxisVector( float stem_fraction, const std::vector<helios::vec3> &axis_vertices ) const;

    float getInternodeLength() const;

    float getPetioleLength() const;

    void addInfluorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation, const helios::vec3 &inflorescence_bending_axis);

    void scaleInternode( float girth_scale_factor, float length_scale_factor );

    void setInternodeScale( float scale_factor_fraction );

    void setLeafScale( float scale_factor_fraction );

    void setPhytomerScale( float scale_factor_fraction );

    void setPetioleBase( const helios::vec3 &base_position );

    void setPhytomerBase( const helios::vec3 &base_position );

    std::vector<helios::vec3> internode_vertices;
    std::vector<helios::vec3> petiole_vertices; //\todo this needs to be a multidimensional array for the case in which we have multiple buds per phytomer
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

    float current_internode_scale_factor = 1;
    float current_leaf_scale_factor = 1;

    helios::Context *context_ptr;

};

struct Shoot{

    Shoot(int ID, int parentID, uint parent_node, uint rank, const helios::vec3 &origin,
          const AxisRotation &shoot_base_rotation, uint current_node_number,
          float phytomer_scale_factor_fraction, const PhytomerParameters &phytomer_params,
          const ShootParameters &shoot_params, std::vector<Shoot> *shoot_tree_ptr, helios::Context *context_ptr);

    int addPhytomer(const PhytomerParameters &params, const AxisRotation &shoot_base_rotation, float phytomer_scale_factor_fraction);

    uint current_node_number;

    helios::vec3 origin;

    AxisRotation base_rotation;

    int ID;
    int parentID;
    uint parentNode;
    uint rank;
    std::vector<int> childIDs;

    ShootParameters shoot_parameters;

    std::vector<Phytomer> phytomers;

    std::vector<Shoot> *shoot_tree_ptr;

    helios::Context *context_ptr;

};

class PlantArchitecture{
public:

    explicit PlantArchitecture( helios::Context* context_ptr );

    uint addShoot(int parentID, uint parent_node, uint rank, uint current_node_number, const helios::vec3 &base_position, const AxisRotation &base_rotation, float phytomer_scale_factor_fraction,
                  const PhytomerParameters &phytomer_parameters, const ShootParameters &shoot_params);

    uint addChildShoot(int parentID, uint parent_node, uint current_node_number, const AxisRotation &base_rotation, float phytomer_scale_factor_fraction, const PhytomerParameters &phytomer_parameters, const ShootParameters &shoot_params);

    int addPhytomerToShoot(uint shootID, const PhytomerParameters &phytomer_params, float scale_factor_fraction);

    PhytomerParameters getPhytomerParametersFromLibrary(const std::string &phytomer_label );

    void scalePhytomerInternode( uint shootID, uint node_number, float girth_scale_factor, float length_scale_factor );

    void setPhytomerInternodeScale( uint shootID, uint node_number, float scale_factor );

    void setPhytomerLeafScale( uint shootID, uint node_number, float scale_factor );

    void setPhytomerScale( uint shootID, uint node_number, float scale_factor );

    void advanceTime( float dt );

    static std::vector<uint> makeTubeFromCones(uint Ndivs, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr);

    //Primary data structure containing all phytomers for the plant
    // \todo should be private - keeping here for debugging
    std::vector<Shoot> shoot_tree;

private:

    helios::Context* context_ptr;

    std::minstd_rand0 *generator = nullptr;

};


#endif //PLANT_ARCHITECTURE
