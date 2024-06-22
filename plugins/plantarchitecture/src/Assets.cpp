

#include "PlantArchitecture.h"

using namespace helios;

uint buildGenericLeafPrototype(helios::Context *context_ptr, uint subdivisions, const std::string &leaf_texture, float leaf_aspect_ratio, float midrib_fold_fraction, float x_curvature, float y_curvature, float petiole_roll, float wave_period, float wave_amplitude) {

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
            float z_xcurve = x_curvature*powf(x,4);
            float z_xcurve_iplus = x_curvature*powf(x+dx,4);

            //y-curvature
            float z_ycurve = y_curvature*powf(y/leaf_aspect_ratio,4);
            float z_ycurve_jplus = y_curvature*powf((y+dy)/leaf_aspect_ratio,4);

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

    return context_ptr->addPolymeshObject( UUIDs );
}

uint AlmondLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint AlmondFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondHull.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void AlmondPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, uint rank, float plant_age ) {

    if( phytomer->internode_length_max < 0.01 ){ //spurs
        phytomer->setInternodeMaxRadius( 0.005 );
        phytomer->setVegetativeBudState( BUD_DEAD );
        phytomer->scaleLeafPrototypeScale( 0.7 );
        //\todo Need to kill axillary buds here, but need to separate axillary and terminal buds
    }

}

uint BeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_unifoliate.png";

    float leaf_aspect = 0.8; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = -0.2; //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = -0.1; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, 0, 0);

}

uint BeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::string leaf_texture;
    if( compound_leaf_index==0 ){
        leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_tip.png";
    }else if( compound_leaf_index<0 ){
        leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_left.png";
    }else{
        leaf_texture = "plugins/plantarchitecture/assets/textures/BeanLeaf_right.png";
    }


    float leaf_aspect = 1.0; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = context_ptr->randu(-0.25f,-0.01f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = -0.2; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, 0, 0);

}

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_unifoliate.obj", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
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

void BeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, uint rank, float plant_age ){

    if( shoot_node_index>10 || rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
    if( shoot_node_index<=1 || shoot_node_index > 15){
        phytomer->setFloralBudState( BUD_DEAD );
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.3 + 0.7 * plant_age / 10.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 10.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}

uint BindweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedLeaf.obj", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BindweedFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedFlower.obj", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CheeseweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CheeseweedLeaf.obj", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/CowpeaLeaf_unifoliate.png";

    float leaf_aspect = 0.8; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = -0.2; //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = -0.1; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, 0, 0);

}

uint CowpeaLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
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

    float x_curvature = context_ptr->randu(-0.25f,-0.01f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = -0.2; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, 0, 0);

}

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
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

void CowpeaPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, uint rank, float plant_age ){

    if( shoot_node_index>5 || rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }else{
        phytomer->setFloralBudState( BUD_DEAD );
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 15.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}

uint PuncturevineLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/PuncturevineLeaf.png";

    float leaf_aspect = 0.4; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = -0.1; //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = 0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, 0, 0);

}

uint PuncturevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/PuncturevineFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint RedbudLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/RedbudLeaf.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint RedbudFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/RedbudFlower_open.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint RedbudFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/RedbudPod.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void RedbudPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, uint rank, float plant_age ){

//    //remove all vegetative buds
//    phytomer->setVegetativeBudState( BUD_DEAD );
//
//    //remove all floral buds except for the terminal one
//    if( shoot_node_index < shoot_max_nodes-1 ){
//        phytomer->setFloralBudState( BUD_DEAD );
//    }

}

uint SorghumLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";

    float leaf_aspect = 0.25; //ratio of leaf width to leaf length

    float midrib_fold_fraction = 0.3;

    float x_curvature = context_ptr->randu(-0.9f,-0.6f);
    float y_curvature = -0.3;

    float petiole_roll = 0.04f;

    //parameters for leaf wave/wrinkles
    float wave_period = 0.1f; //period factor of leaf waves
    float wave_amplitude = 0.035f; // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold_fraction, x_curvature, y_curvature, petiole_roll, wave_period, wave_amplitude);

}

uint SorghumPaniclePrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){

    if( subdivisions<=1 ){
        subdivisions = 3;
    }

    float panicle_height = 1;
    float panicle_width = 0.25;
    float width_seed = 0.1;
    float height_seed = 0.3;
    float seed_tilt = 50;

    std::string seed_texture_file = "plugins/plantarchitecture/assets/textures/SorghumSeed.png";
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

    int Ntheta = ceil( 3.f*panicle_height/height_seed );
    int Nphi = ceil( 2.f*M_PI*panicle_width/width_seed);

    for(int j=0; j < Nphi; j++ ) {
        for (int i = 0; i < Ntheta; i++) {

            if( i==0 && j==0 ){
                continue;
            }

            std::vector<uint> UUIDs_copy = context_ptr->copyPrimitive(UUIDs_seed_ptype);

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

    std::vector<vec3> stem_nodes;
    std::vector<float> stem_radius;
    std::vector<RGBcolor> stem_colors;
    for(int i=0; i<subdivisions; i++){
        stem_nodes.push_back(make_vec3(0,0, float(i)/float(subdivisions-1)*panicle_height*0.75f) );
        stem_radius.push_back(0.25f*panicle_width);
        stem_colors.push_back(stem_color);
    }

    std::vector<uint> UUIDS_stem = context_ptr->addTube( subdivisions, stem_nodes, stem_radius, stem_colors);
    UUIDs.insert(UUIDs.end(), UUIDS_stem.begin(), UUIDS_stem.end());

//    context_ptr->translatePrimitive(UUIDs, make_vec3(0,0,-0.3f*panicle_height) );
    context_ptr->rotatePrimitive(UUIDs, 0.5f*M_PI, "y");

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void SorghumPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, uint rank, float plant_age ){

    //set leaf scale based on position along the shoot
    float scale = fmin(1.f, 0.5 + 0.5*float(shoot_node_index)/5.f);
    phytomer->scaleLeafPrototypeScale(scale);

    //set internode length based on position along the shoot
    phytomer->scaleInternodeMaxLength(scale);

    //remove all vegetative buds
    phytomer->setVegetativeBudState( BUD_DEAD );

    //remove all floral buds except for the terminal one
    if( shoot_node_index < shoot_max_nodes-1 ){
        phytomer->setFloralBudState( BUD_DEAD );
    }

}

uint SoybeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
//    std::vector<uint> UUIDs;
//    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/SoybeanLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
//    uint objID = context_ptr->addPolymeshObject( UUIDs );
//    return objID;

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SoybeanLeaf.png";

    float leaf_aspect = 1.0; //ratio of leaf width to leaf length

    float midrib_fold = 0.1; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = context_ptr->randu(-0.2f,0.f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = -0.15; //curvature factor along y-direction. (+curves upward, -curved downward)

   return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, 0, 0);

}

uint SoybeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    return SoybeanLeafPrototype_trifoliate(context_ptr, subdivisions, compound_leaf_index, shoot_node_index, shoot_max_nodes); //\todo Add separate model for unifoliate leaves
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

void SoybeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, uint rank, float plant_age ){

    if( shoot_node_index>10 || rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
    if( shoot_node_index<=0 || shoot_node_index > 15){
        phytomer->setFloralBudState( BUD_DEAD );
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 5.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.75 + 0.25 * plant_age / 5.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}

uint StrawberryLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/StrawberryLeaf.png";

    float leaf_aspect = 1.0; //ratio of leaf width to leaf length

    float midrib_fold_fraction = 0.2;

    float x_curvature = -0.01;
    float y_curvature = -0.2;

    float petiole_roll = 0.05f;

    //parameters for leaf wave/wrinkles
    float wave_period = 0.f; //period factor of leaf waves
    float wave_amplitude = 0.0f; // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold_fraction, x_curvature, y_curvature, petiole_roll, wave_period, wave_amplitude);

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

uint SugarbeetLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SugarbeetLeaf.png";

    float leaf_aspect = 0.4; //ratio of leaf width to leaf length

    float midrib_fold = 0.1; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = -0.2; //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = 0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    float petiole_roll = 0.75; //add a small radius roll at the based of the leaf to better mate with the petiole. Value is the magnitude of the roll (+rolls upward, - rolls downward)

    //parameters for leaf wave/wrinkles
    float wave_period = context_ptr->randu( 0.08f, 0.15f); //period factor of leaf waves
    float wave_amplitude = context_ptr->randu(0.02f,0.04f); // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, petiole_roll, wave_period, wave_amplitude);

}

uint TomatoLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
//    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
//    uint objID = context_ptr->addPolymeshObject( UUIDs );
//    return objID;

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/TomatoLeaf_centered.png";

    float leaf_aspect = 0.5; //ratio of leaf width to leaf length

    float midrib_fold = 0.1; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = context_ptr->randu(-0.45f,-0.2f); //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = -0.3; //curvature factor along y-direction. (+curves upward, -curved downward)

    //parameters for leaf wave/wrinkles
    float wave_period = 0.2f; //period factor of leaf waves
    float wave_amplitude = 0.03f; // amplitude of leaf waves

    return buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, wave_period, wave_amplitude);

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

void TomatoPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, uint rank, float plant_age ){

    if( shoot_node_index>5 || rank>1 ) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    }

    //set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.5 + 0.5 * plant_age / 10.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    //set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 10.f);
    phytomer->scaleInternodeMaxLength(inode_scale);

}