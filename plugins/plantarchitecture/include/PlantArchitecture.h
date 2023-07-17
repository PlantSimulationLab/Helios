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
    }

    explicit RandomParameter_float(float a_val, std::minstd_rand0 *rand_generator) : constval(a_val){
        distribution = "constant";
        generator = rand_generator;
    }

    void initialize( float a_val, std::minstd_rand0 *rand_generator){
        constval = a_val;
        generator = rand_generator;
    }

    void initialize( std::minstd_rand0 *rand_generator){
        constval = 1.f;
        generator = rand_generator;
    }

    RandomParameter_float& operator=(float a){
        this->constval = a;
        return *this;
    }

    void uniformDistribution( float minval, float maxval ){
        distribution = "uniform";
        distribution_parameters = {minval, maxval};
    }

    void normalDistribution( float mean, float std_dev ){
        distribution = "normal";
        distribution_parameters = {mean, std_dev};
    }

    void weibullDistribution( float shape, float scale ){
        distribution = "normal";
        distribution_parameters = {shape, scale};
    }

    float val(){
        if( distribution=="constant" ) {
            return constval;
        }else if( distribution=="uniform" ) {
            if( generator== nullptr ){
                throw(std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            std::uniform_real_distribution<float> unif_distribution;
            return distribution_parameters.at(0)+unif_distribution(*generator)*(distribution_parameters.at(1)-distribution_parameters.at(0));
        }else if( distribution=="normal" ) {
            if( generator== nullptr ){
                throw(std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            std::normal_distribution<float> norm_distribution(distribution_parameters.at(0), distribution_parameters.at(1));
            return norm_distribution(*generator);
        }else if( distribution=="weibull" ){
            if( generator== nullptr ){
                throw(std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            std::weibull_distribution<float> wbull_distribution(distribution_parameters.at(0), distribution_parameters.at(1));
            return wbull_distribution(*generator);
        }else {
            assert(1);
        }
        return 0;
    }

    bool isnull() const{
        return distribution=="constant" && constval==0;
    }

private:
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
    }

    explicit RandomParameter_int(int a_val, std::minstd_rand0 *rand_generator) : constval(a_val){
        distribution = "constant";
        generator = rand_generator;
    }

    void initialize(int a_val, std::minstd_rand0 *rand_generator){
        constval = a_val;
        generator = rand_generator;
    }

    void initialize( std::minstd_rand0 *rand_generator){
        constval = 1;
        generator = rand_generator;
    }

    RandomParameter_int& operator=(int a){
        this->constval = a;
        return *this;
    }

    void uniformDistribution( int minval, int maxval ){
        distribution = "uniform";
        distribution_parameters = {minval, maxval};
    }

    int val(){
        if( distribution=="constant" ) {
            return constval;
        }else if( distribution=="uniform" ) {
            if (generator == nullptr) {
                throw (std::runtime_error("ERROR (PlantArchitecture): Random parameter was not properly initialized with random number generator."));
            }
            return distribution_parameters.at(0) + unif_distribution(*generator) * (distribution_parameters.at(1) - distribution_parameters.at(0));
        }else{
            assert(1);
        }
        return 0;
    }

    bool isnull() const{
        return distribution=="constant" && constval==0;
    }

private:
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

//    AxisRotation operator+(const AxisRotation& a) const;
//    AxisRotation operator-(const AxisRotation& a) const;
//    AxisRotation operator*( float a) const;
//    AxisRotation operator/( float a) const;
//    bool operator==(const AxisRotation& a) const;
//
//    friend std::ostream &operator<<(std::ostream &os, AxisRotation const &vec) {
//        return os << "AxisRotation<" << vec.pitch_static << ", " << vec.yaw_static << ", " << vec.roll_static << ">";
//    }

};

inline AxisRotation make_AxisRotation( float a_pitch, float a_yaw, float a_roll ) {
    return {a_pitch,a_yaw,a_roll};
}

//inline AxisRotation AxisRotation::operator+(const AxisRotation& a) const{
//    return  {a.pitch+pitch,a.yaw+yaw,a.roll+roll};
//}
//
//inline AxisRotation AxisRotation::operator-(const AxisRotation& a) const{
//    return  {pitch-a.pitch,yaw-a.yaw,roll-a.roll};
//}
//
//inline AxisRotation AxisRotation::operator*(const float a) const{
//    return  {pitch*a,yaw*a,roll*a};
//}
//
//inline AxisRotation operator*(const float a, const AxisRotation& v) {
//    return {a * v.pitch, a * v.yaw, a * v.roll};
//}
//
//inline AxisRotation AxisRotation::operator/(const float a) const{
//    return {pitch/a,yaw/a,roll/a};
//}
//
//inline bool AxisRotation::operator==(const AxisRotation& a) const{
//    return pitch == a.pitch && yaw == a.yaw && roll == a.roll;
//}

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
    };

    struct LeafParameters{
        RandomParameter_float pitch;
        RandomParameter_float yaw;
        RandomParameter_float roll;
        RandomParameter_float leaflet_offset;
        RandomParameter_float leaflet_scale;
        helios::vec3 prototype_scale;
        uint(*prototype_function)( helios::Context*, uint subdivisions, int flag );
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
        uint(*fruit_prototype_function)( helios::Context*, uint subdivisions, int flag );
        uint(*flower_prototype_function)( helios::Context*, uint subdivisions, int flag );
    };

    std::minstd_rand0 generator;

public:

    PhytomerParameters();

    InternodeParameters internode;

    PetioleParameters petiole;

    LeafParameters leaf;

    InflorescenceParameters inflorescence;

    void seedRandomGenerator( uint seed );

};

struct ShootParameters{

    ShootParameters();

    uint max_nodes;

    float shoot_internode_taper;

};

struct Phytomer {
public:

    Phytomer(const PhytomerParameters &params, uint phytomer_index, const helios::vec3 &parent_internode_axis,
             const helios::vec3 &parent_petiole_axis, const AxisRotation &shoot_base_rotation, float scale, uint rank,
             helios::Context *context_ptr);

    helios::vec3 getInternodeAxisVector( float stem_fraction ) const;

    helios::vec3 getPetioleAxisVector( float stem_fraction ) const;

    helios::vec3 getAxisVector( float stem_fraction, const std::vector<helios::vec3> &axis_vertices ) const;

    void addInfluorescence(const helios::vec3 &base_position, const AxisRotation &base_rotation, const helios::vec3 &inflorescence_bending_axis);

    void scaleInternode( float girth_scale_factor, float length_scale_factor );

    std::vector<helios::vec3> internode_vertices;
    std::vector<helios::vec3> petiole_vertices; //\todo this needs to be a multidimensional array for the case in which we have multiple buds per phytomer

    std::vector<float> internode_radii;
    std::vector<float> petiole_radii;

    std::vector<helios::RGBcolor> internode_colors;
    std::vector<helios::RGBcolor> petiole_colors;

    std::vector<uint> internode_UUIDs;
    std::vector<uint> petiole_UUIDs;
    std::vector<uint> leaf_UUIDs;
    std::vector<uint> inflorescence_UUIDs;

    PhytomerParameters phytomer_parameters;

    uint rank;

    float age = 0;

    helios::Context *context_ptr;

};

struct Shoot{

    Shoot(int ID, int parentID, uint parent_node, uint rank, const helios::vec3 &origin,
          const AxisRotation &shoot_base_rotation, uint current_node_number, const PhytomerParameters &phytomer_params,
          const ShootParameters &shoot_params, std::vector<Shoot> *shoot_tree_ptr, helios::Context *context_ptr);

    int addPhytomer(PhytomerParameters &params, const AxisRotation &shoot_base_rotation);

    uint current_node_number;

    helios::vec3 origin;

    AxisRotation base_rotation;

    int ID;
    int parentID;
    uint parentNode;
    uint rank;
    std::vector<int> childIDs;

    std::vector<Phytomer> phytomers;

    std::vector<Shoot> *shoot_tree_ptr;

    helios::Context *context_ptr;

};

class PlantArchitecture{
public:

    explicit PlantArchitecture( helios::Context* context_ptr );

    uint addShoot(int parentID, uint parent_node, uint rank, uint current_node_number, const helios::vec3 &base_position,
             const AxisRotation &base_rotation, const ShootParameters &shoot_params);

    uint addChildShoot(int parentID, uint parent_node, uint current_node_number, const AxisRotation &base_rotation,
                       const ShootParameters &shoot_params);

    int addPhytomerToShoot( uint shootID, PhytomerParameters phytomer_parameters );

    void setCurrentPhytomerParameters( const PhytomerParameters &phytomer_parameters_new );

    void setCurrentPhytomerParameters( const std::string &phytomer_label );

    void scalePhytomerInternode( uint shootID, uint node_number, float girth_scale_factor, float length_scale_factor );

    PhytomerParameters phytomer_parameters_current;

    static std::vector<uint> makeTubeFromCones(uint Ndivs, const std::vector<helios::vec3> &vertices, const std::vector<float> &radii, const std::vector<helios::RGBcolor> &colors, helios::Context *context_ptr);

private:

    //Primary data structure containing all phytomers for the plant
    std::vector<Shoot> shoot_tree;

    helios::Context* context_ptr;

};

extern PhytomerParameters BeanPhytomerParameters;


#endif //PLANT_ARCHITECTURE
