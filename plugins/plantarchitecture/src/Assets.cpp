/** \file "Assets.cpp" Function definitions for plant organ prototypes plant architecture plug-in.

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

using namespace helios;

uint buildGenericLeafPrototype(helios::Context *context_ptr, uint subdivisions, const std::string &leaf_texture, float leaf_aspect_ratio, float midrib_fold_fraction, float longitudinal_curvature, float lateral_curvature, float petiole_roll, float wave_period,
                               float wave_amplitude, bool build_petiolule) {

    // -- main leaf generation code -- //

    std::vector<uint> UUIDs;

    uint Nx = subdivisions; //number of leaf subdivisions in the x-direction (longitudinal)
    uint Ny = ceil(leaf_aspect_ratio*float(Nx)); //number of leaf subdivisions in the y-direction (lateral)

    if ( Ny % 2 != 0){ //Ny must be even
        Ny = Ny + 1;
    }

    float dx = 1.f/float(Nx); //length of leaf subdivision in the x-direction
    float dy = leaf_aspect_ratio/float(Ny); //length of leaf subdivision in the y-direction

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {

            float x = float(i) * dx; //x-coordinate of leaf subdivision
            float y = float(j) * dy - 0.5f * leaf_aspect_ratio; //y-coordinate of leaf subdivision

            //midrib leaf folding
            float y_fold = cosf(0.5f*midrib_fold_fraction*M_PI)*y;
            float y_fold_jplus = cosf(0.5f*midrib_fold_fraction*M_PI)*(y+dy);
            float z_fold = sinf(0.5f*midrib_fold_fraction*M_PI)*fabs(y);
            float z_fold_jplus = sinf(0.5f*midrib_fold_fraction*M_PI)*fabs(y+dy);

            //x-curvature
            float z_xcurve = longitudinal_curvature * powf(x, 4);
            float z_xcurve_iplus = longitudinal_curvature * powf(x + dx, 4);

            //y-curvature
            float z_ycurve = lateral_curvature * powf(y / leaf_aspect_ratio, 4);
            float z_ycurve_jplus = lateral_curvature * powf((y + dy) / leaf_aspect_ratio, 4);

            //petiole roll
            float z_petiole_0 = 0, z_petiole_1 = 0, z_petiole_2 = 0, z_petiole_3 = 0;
            if( petiole_roll != 0.0f ) {
                z_petiole_0 = fmin(0.1f, petiole_roll * powf(7.f * y / leaf_aspect_ratio, 4) * exp(-70.f * (x))) - 0.01 * petiole_roll / fabs(petiole_roll);
                z_petiole_1 = fmin(0.1f, petiole_roll * powf(7.f * y / leaf_aspect_ratio, 4) * exp(-70.f * (x + dx))) - 0.01 * petiole_roll / fabs(petiole_roll);
                z_petiole_2 = fmin(0.1f, petiole_roll * powf(7.f * (y + dy) / leaf_aspect_ratio, 4) * exp(-70.f * (x + dx))) - 0.01 * petiole_roll / fabs(petiole_roll);
                z_petiole_3 = fmin(0.1f, petiole_roll * powf(7.f * (y + dy) / leaf_aspect_ratio, 4) * exp(-70.f * (x))) - 0.01 * petiole_roll / fabs(petiole_roll);
            }

            //vertical displacement for leaf wave at each of the four subdivision vertices
            float z_wave_0 = 0, z_wave_1 = 0, z_wave_2 = 0, z_wave_3 = 0;
            if( wave_period > 0.0f && wave_amplitude > 0.0f ) {
                z_wave_0 = (2.f * fabs(y) * wave_amplitude * sinf((x + 0.5f * float(j >= 0.5 * Ny)) * M_PI / wave_period));
                z_wave_1 = (2.f * fabs(y) * wave_amplitude * sinf((x + 0.5f * float(j >= 0.5 * Ny) + dx) * M_PI / wave_period));
                z_wave_2 = (2.f * fabs(y + dy) * wave_amplitude * sinf((x + 0.5f * float(j >= 0.5 * Ny) + dx) * M_PI / wave_period));
                z_wave_3 = (2.f * fabs(y + dy) * wave_amplitude * sinf((x + 0.5f * float(j >= 0.5 * Ny)) * M_PI / wave_period));
            }

            //define the four vertices of the leaf subdivision
            vec3 v0(x, y_fold, z_fold+z_xcurve+z_ycurve+z_wave_0+z_petiole_0);

            vec3 v1(x + dx, y_fold, z_fold+z_xcurve_iplus+z_ycurve+z_wave_1+z_petiole_1);

            vec3 v2(x + dx, y_fold_jplus, z_fold_jplus+z_xcurve_iplus+z_ycurve_jplus+z_wave_2+z_petiole_2);

            vec3 v3(x, y_fold_jplus, z_fold_jplus+z_xcurve+z_ycurve_jplus+z_wave_3+z_petiole_3);

            vec2 uv0(x, (y+0.5f*leaf_aspect_ratio) / leaf_aspect_ratio);
            vec2 uv1(x + dx, (y+0.5f*leaf_aspect_ratio) / leaf_aspect_ratio);
            vec2 uv2(x + dx, (y + dy + 0.5f*leaf_aspect_ratio) / leaf_aspect_ratio);
            vec2 uv3(x, (y + dy + 0.5f*leaf_aspect_ratio) / leaf_aspect_ratio);

            UUIDs.push_back(context_ptr->addTriangle(v0, v1, v2, leaf_texture.c_str(), uv0, uv1, uv2));
            UUIDs.push_back(context_ptr->addTriangle(v0, v2, v3, leaf_texture.c_str(), uv0, uv2, uv3));

        }
    }

    if( build_petiolule ){
        std::vector<uint> UUIDs_petiolule = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/PetiolulePrototype.obj", make_vec3(0,0,0), 0, nullrotation, RGB::black, "ZUP", true);
        context_ptr->translatePrimitive(UUIDs, make_vec3(0.07,0,0.005) );
        UUIDs.insert(UUIDs.end(), UUIDs_petiolule.begin(), UUIDs_petiolule.end());
    }

    return context_ptr->addPolymeshObject( UUIDs );
}

uint AlmondLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
//    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
//    uint objID = context_ptr->addPolymeshObject( UUIDs );
//    return objID;

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/AlmondLeaf.png";

    float leaf_aspect = 0.4; //ratio of leaf width to leaf length

    float midrib_fold = 0.; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = 0.05; //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = 0.1; //curvature factor along y-direction. (+curves upward, -curved downward)

    uint objID = buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, 0, 0, false);

    return objID;

}

uint AlmondFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondHull_lowres.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void AlmondPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ) {

    if( phytomer->internode_length_max < 0.01 ){ //spurs
        phytomer->setInternodeMaxRadius( 0.005 );
        phytomer->setVegetativeBudState( BUD_DEAD );
        phytomer->scaleLeafPrototypeScale( 0.8 );
        phytomer->setFloralBudState( BUD_DEAD );
    }

    //blind nodes
//    if( shoot_node_index<3 ){
//        phytomer->setVegetativeBudState( BUD_DEAD );
//        phytomer->setFloralBudState( BUD_DEAD );
//    }

}

void AlmondPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer ){

    if( phytomer->isdormant ){
        if( phytomer->shoot_index.x >= phytomer->shoot_index.y-3 && phytomer->internode_length_max > 0.01 ){
            phytomer->setVegetativeBudState( BUD_DORMANT ); //first two vegetative buds always break
        }
    }

}

uint AsparagusLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){

    float curve_magnitude = context_ptr->randu(0.f,0.2f);

    std::vector<vec3> nodes;
    nodes.push_back( make_vec3(0,0,0) );
    nodes.push_back( make_vec3( context_ptr->randu(0.4f,0.7f),0,-0.25f*curve_magnitude) );
    nodes.push_back( make_vec3(0.95,0,-0.9f*curve_magnitude) );
    nodes.push_back( make_vec3(1,0,-curve_magnitude) );

    std::vector<float> radius;
    radius.push_back(0.015);
    radius.push_back(0.015);
    radius.push_back(0.015);
    radius.push_back(0.0);

    std::vector<RGBcolor> colors;
    colors.push_back( RGB::forestgreen );
    colors.push_back( RGB::forestgreen );
    colors.push_back( RGB::forestgreen );
    colors.push_back( RGB::forestgreen );

    uint objID = context_ptr->addTubeObject( 8, nodes, radius, colors);
    context_ptr->rotateObject( objID, context_ptr->randu(0,2.f*M_PI), "x" );
    return objID;
}

void AsparagusPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ) {

    //blind nodes
    if( shoot_node_index<=2 ){
        phytomer->scaleLeafPrototypeScale( 0.6 );
        phytomer->setVegetativeBudState( BUD_DEAD );
    }

}

uint BeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_unifoliate_centered.png";

    float leaf_aspect = 0.8; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = -0.2; //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -0.1; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, 0, 0, true);

}

uint BeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::string leaf_texture;
    if( compound_leaf_index==0 ){
        leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_tip.png";
    }else if( compound_leaf_index<0 ){
        leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_left_centered.png";
    }else{
        leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_right_centered.png";
    }

    float leaf_aspect = 1.0; //ratio of leaf width to leaf length

    float midrib_fold = 0.025; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = context_ptr->randu(-0.3f,-0.2f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -1.; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, 0, 0, true);

}

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_unifoliate.obj", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::vector<uint> UUIDs;
    if( compound_leaf_index==0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_tip.obj", true );
    }else if( compound_leaf_index<0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_left.obj", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_right.obj", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanPod.obj", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs;
    if( flower_is_open ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanFlower_open_white.obj", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanFlower_closed_white.obj", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void BeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    if( shoot_node_index>5 || phytomer->rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }else{
        phytomer->setFloralBudState(BUD_DEAD);
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.1 + 0.9 * plant_age / 15.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}

uint BindweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedLeaf.obj", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BindweedFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedFlower.obj", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CheeseweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CheeseweedLeaf.obj", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_unifoliate_centered.png";

    float leaf_aspect = 0.8; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = -0.2; //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -0.1; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, 0, 0, false);

}

uint CowpeaLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::string leaf_texture;
    if( compound_leaf_index==0 ){
        leaf_texture = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_tip_centered.png";
    }else if( compound_leaf_index<0 ){
        leaf_texture = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_left.png";
    }else{
        leaf_texture = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_right.png";
    }

    float leaf_aspect = 0.8; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = context_ptr->randu(-0.4f,-0.1f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, 0, 0, true);

}

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::vector<uint> UUIDs;
    if( compound_leaf_index<0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_left_highres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else if( compound_leaf_index==0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_tip_highres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_right_highres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaPod.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs;
    if( flower_is_open ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaFlower_open_yellow.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaFlower_closed_yellow.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void CowpeaPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    if( shoot_node_index>5 || phytomer->rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }else{
        phytomer->setFloralBudState(BUD_DEAD);
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.1 + 0.9 * plant_age / 15.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}

uint PuncturevineLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/PuncturevineLeaf.png";

    float leaf_aspect = 0.4; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = -0.1; //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = 0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, 0, 0, false);

}

uint PuncturevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/PuncturevineFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint RedbudLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/RedbudLeaf.png";

    float leaf_aspect = 1.0; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = -0.15; //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -0.1; //curvature factor along y-direction. (+curves upward, -curved downward)

    //parameters for leaf wave/wrinkles
    float wave_period = 0.3f; //period factor of leaf waves
    float wave_amplitude = 0.025f; // amplitude of leaf waves

    uint objID = buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, wave_period, wave_amplitude, false);
    context_ptr->translateObject( objID, make_vec3(-0.3,0,0) );

    return objID;
}

uint RedbudFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/RedbudFlower_open.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    return context_ptr->addPolymeshObject( UUIDs );
}

uint RedbudFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/RedbudPod.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    return context_ptr->addPolymeshObject( UUIDs );
}

void RedbudPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

}

void RedbudPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer ){

    int Nchild_shoots = randu(0,3);

    if( phytomer->isdormant ){
        if( phytomer->rank<=1 ){
            Nchild_shoots = std::max(2,Nchild_shoots);
        }
        if( phytomer->shoot_index.x < phytomer->shoot_index.y-Nchild_shoots ){
            phytomer->setVegetativeBudState( BUD_DEAD );
        }
        else{
            phytomer->setFloralBudState(BUD_DEAD);
        }
    }

}

uint RomaineLettuceLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/RomaineLettuceLeaf.png";

//    float leaf_aspect = 0.65; //ratio of leaf width to leaf length
    float leaf_aspect = 0.85; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

//    float longitudinal_curvature = context_ptr->randu(-0.05f,0.2f); //curvature factor along x-direction. (+curves upward, -curved downward)
    float longitudinal_curvature = context_ptr->randu(-0.2f,0.05f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    float petiole_roll = 0; //add a small radius roll at the based of the leaf to better mate with the petiole. Value is the magnitude of the roll (+rolls upward, - rolls downward)

    //parameters for leaf wave/wrinkles
//    float wave_period = context_ptr->randu( 0.1f, 0.2f); //period factor of leaf waves
    float wave_period = context_ptr->randu( 0.15f, 0.25f); //period factor of leaf waves
//    float wave_amplitude = context_ptr->randu(0.03f,0.075f); // amplitude of leaf waves
    float wave_amplitude = context_ptr->randu(0.05f,0.1f); // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, petiole_roll, wave_period, wave_amplitude, false);

}

void RomaineLettucePhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    float fact = float(shoot_max_nodes-shoot_node_index)/float(shoot_max_nodes);

    //set leaf scale based on position along the shoot
//    float scale = fmin(1.f, 1 + 0.1*fact);
//    phytomer->scaleLeafPrototypeScale(scale);

//    phytomer->rotateLeaf( 0, 0, make_AxisRotation(-deg2rad(15)*fact, 0, 0));
    phytomer->rotateLeaf( 0, 0, make_AxisRotation(-deg2rad(60)*fact, 0, 0));

}

uint SorghumLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";

    float leaf_aspect = 0.2; //ratio of leaf width to leaf length

    float midrib_fold_fraction = 0.3;

    float longitudinal_curvature = context_ptr->randu(-1.2f,-0.8f);
    float lateral_curvature = -0.3;

    float petiole_roll = 0.04f;

    //parameters for leaf wave/wrinkles
    float wave_period = 0.1f; //period factor of leaf waves
    float wave_amplitude = 0.1f; // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold_fraction, longitudinal_curvature, lateral_curvature, petiole_roll, wave_period, wave_amplitude, false);

}

uint SorghumPaniclePrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){

    if( subdivisions<=1 ){
        subdivisions = 3;
    }

    float panicle_height = 1;
    float panicle_width = 0.08;
    float width_seed = 0.08;
    float height_seed = 0.25;
    float seed_tilt = 50;
    subdivisions = 6;

    std::string seed_texture_file = "plugins/plantarchitecture/assets/textures/SorghumSeed.jpeg";
    RGBcolor stem_color(0.45,0.55,0.42);

    std::vector<uint> UUIDs;

    panicle_height -= 0.8*height_seed;

    std::vector<vec3> nodes_panicle;
    std::vector<float> radius_panicle;

    for (int n = 0; n < subdivisions ; n++) {
        float x = 0;
        float y = 0;
        float z;
        if( n==0 ){
            z = 0.5f*height_seed/float(subdivisions-1);
        }else if( n==subdivisions-1 ){
            z = (subdivisions-1.5f)*height_seed/float(subdivisions-1);
        }else{
            z = n*height_seed/float(subdivisions-1);
        }

        float angle = n * M_PI /float(subdivisions-1);
        float dr = 0.5f * width_seed * sin(angle);

        nodes_panicle.push_back(make_vec3(x, y, z));
        radius_panicle.push_back(dr);

    }

    std::vector<uint> UUIDs_seed_ptype = context_ptr->addTube( subdivisions, nodes_panicle, radius_panicle,seed_texture_file.c_str());

    int Ntheta = ceil( 6.f*panicle_height/height_seed );
    int Nphi = ceil( 2.f*M_PI*panicle_width/width_seed);

    for(int j=0; j < Nphi; j++ ) {
        for (int i = 0; i < Ntheta; i++) {

            if( i==0 && j==0 ){
                continue;
            }

            std::vector<uint> UUIDs_copy = context_ptr->copyPrimitive(UUIDs_seed_ptype);
            context_ptr->scalePrimitive( UUIDs_copy, make_vec3(1,1,1)*context_ptr->randu(0.9f,1.1f));

            float phi = 2.f * M_PI * float(j + 0.5f*float(i%2)) / float(Nphi);
            float theta = acos(1 - 2 * float(i + float(j)/float(Nphi)) / float(Ntheta));
            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = 0.5f + 0.5f*cos(theta);

            x *= 0.5f * panicle_width;
            y *= 0.5f * panicle_width;
            z *= panicle_height;

            float tilt = - deg2rad(seed_tilt) * sqrtf(1.f - z / panicle_height);

            context_ptr->rotatePrimitive(UUIDs_copy, tilt, "x");
            context_ptr->rotatePrimitive(UUIDs_copy, phi - 0.5f * M_PI, "z");

            context_ptr->translatePrimitive(UUIDs_copy, make_vec3(x, y, z));
            UUIDs.insert(UUIDs.end(), UUIDs_copy.begin(), UUIDs_copy.end());

        }
    }

    context_ptr->deletePrimitive(UUIDs_seed_ptype);

    std::vector<uint> UUIDs_sphere = context_ptr->addSphere( 10, make_vec3(0,0,0.5*panicle_height), 0.5f, seed_texture_file.c_str() );
    context_ptr->scalePrimitiveAboutPoint( UUIDs_sphere, make_vec3(1.9*panicle_width, 1.9*panicle_width, 0.8*panicle_height), make_vec3(0,0,0.5*panicle_height));
    UUIDs.insert(UUIDs.end(), UUIDs_sphere.begin(), UUIDs_sphere.end());

    context_ptr->rotatePrimitive(UUIDs, 0.5f*M_PI, "y");
    context_ptr->translatePrimitive( UUIDs, make_vec3(-0.2,0,0));

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;

}

void SorghumPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    //set leaf scale based on position along the shoot
    float scale = fmin(1.f, 0.7 + 0.3*float(shoot_node_index)/5.f);
    phytomer->scaleLeafPrototypeScale(scale);

    //set internode length based on position along the shoot
    phytomer->scaleInternodeMaxLength(scale);

}

uint SoybeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
//    std::vector<uint> UUIDs;
//    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/SoybeanLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
//    uint objID = context_ptr->addPolymeshObject( UUIDs );
//    return objID;

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SoybeanLeaf.png";

    float leaf_aspect = 1.0; //ratio of leaf width to leaf length

    float midrib_fold = 0.1; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = context_ptr->randu(-0.2f,0.f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -0.25; //curvature factor along y-direction. (+curves upward, -curved downward)

   return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, 0, 0, true);

}

uint SoybeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
    return SoybeanLeafPrototype_trifoliate(context_ptr, subdivisions, compound_leaf_index ); //\todo Add separate model for unifoliate leaves
}

uint SoybeanFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/SoybeanPod.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint SoybeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs;
    if( flower_is_open ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/SoybeanFlower_open_white.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanFlower_closed_white.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void SoybeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    if( shoot_node_index>5 || phytomer->rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }else{
        phytomer->setFloralBudState(BUD_DEAD);
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.1 + 0.9 * plant_age / 15.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}

uint StrawberryLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/StrawberryLeaf.png";

    float leaf_aspect = 1.0; //ratio of leaf width to leaf length

    float midrib_fold_fraction = 0.2;

    float longitudinal_curvature = -0.01;
    float lateral_curvature = -0.2;

    //parameters for leaf wave/wrinkles
    float wave_period = 0.3f; //period factor of leaf waves
    float wave_amplitude = 0.01f; // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold_fraction, longitudinal_curvature, lateral_curvature, 0, wave_period, wave_amplitude, true);

}

uint StrawberryFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/StrawberryFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint StrawberryFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/StrawberryFruit.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint SugarbeetLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SugarbeetLeaf.png";

    float leaf_aspect = 0.4; //ratio of leaf width to leaf length

    float midrib_fold = 0.1; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = -0.2; //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = 0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    float petiole_roll = 0.75; //add a small radius roll at the based of the leaf to better mate with the petiole. Value is the magnitude of the roll (+rolls upward, - rolls downward)

    //parameters for leaf wave/wrinkles
    float wave_period = context_ptr->randu( 0.08f, 0.15f); //period factor of leaf waves
    float wave_amplitude = context_ptr->randu(0.02f,0.04f); // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, petiole_roll, wave_period, wave_amplitude, false);

}

uint TomatoLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){
//    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
//    uint objID = context_ptr->addPolymeshObject( UUIDs );
//    return objID;

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/TomatoLeaf_centered.png";

    float leaf_aspect = 0.5; //ratio of leaf width to leaf length

    float midrib_fold = 0.1; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float longitudinal_curvature = context_ptr->randu(-0.45f,-0.2f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float lateral_curvature = -0.3; //curvature factor along y-direction. (+curves upward, -curved downward)

    //parameters for leaf wave/wrinkles
    float wave_period = 0.35f; //period factor of leaf waves
    float wave_amplitude = 0.08f; // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, longitudinal_curvature, lateral_curvature, 0, wave_period, wave_amplitude, false);

}

uint TomatoFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoFruit.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint TomatoFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoFlower.obj", make_vec3(0.0,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void TomatoPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    if( shoot_node_index>5 || phytomer->rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
    if( shoot_node_index<8 && phytomer->rank==0 ){
        phytomer->setFloralBudState(BUD_DEAD);
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.5 + 0.5 * plant_age / 10.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 10.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}

uint WheatLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";

    float leaf_aspect = 0.2; //ratio of leaf width to leaf length

    float midrib_fold_fraction = 0.3;

    float longitudinal_curvature = context_ptr->randu(-2.2f,-1.5f);
    float lateral_curvature = -0.3;

    float petiole_roll = 0.04f;

    //parameters for leaf wave/wrinkles
    float wave_period = 0.1f; //period factor of leaf waves
    float wave_amplitude = 0.05f; // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold_fraction, longitudinal_curvature, lateral_curvature, petiole_roll, wave_period, wave_amplitude, false);

}

uint WheatSpikePrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/WheatSpike.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void WheatPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    //set leaf scale based on position along the shoot
    float scale = fmin(1.f, 0.7 + 0.3*float(shoot_node_index)/5.f);
    phytomer->scaleLeafPrototypeScale(scale);

    //set internode length based on position along the shoot
    phytomer->scaleInternodeMaxLength(scale);

}