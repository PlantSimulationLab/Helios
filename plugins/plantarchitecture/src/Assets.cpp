

#include "PlantArchitecture.h"

using namespace helios;

uint buildGenericLeafPrototype(helios::Context *context_ptr, uint subdivisions, const std::string &leaf_texture, float leaf_aspect_ratio, float midrib_fold_fraction, float x_curvature, float y_curvature, float petiole_roll, float wave_period, float wave_amplitude) {

    // -- main leaf generation code -- //

    std::vector<uint> UUIDs;

    int Nx = subdivisions; //number of leaf subdivisions in the x-direction (longitudinal)
    int Ny = ceil(leaf_aspect_ratio*float(Nx)); //number of leaf subdivisions in the y-direction (lateral)

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
                z_wave_0 = (2.f * fabs(y) * wave_amplitude * sinf((x + 0.5f * int(j >= 0.5 * Ny)) * M_PI / wave_period));
                z_wave_1 = (2.f * fabs(y) * wave_amplitude * sinf((x + 0.5f * int(j >= 0.5 * Ny) + dx) * M_PI / wave_period));
                z_wave_2 = (2.f * fabs(y + dy) * wave_amplitude * sinf((x + 0.5f * int(j >= 0.5 * Ny) + dx) * M_PI / wave_period));
                z_wave_3 = (2.f * fabs(y + dy) * wave_amplitude * sinf((x + 0.5f * int(j >= 0.5 * Ny)) * M_PI / wave_period));
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

uint BeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    if( compound_leaf_index==0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_tip.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else if( compound_leaf_index<0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_left.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_right.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanPod.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs;
    if( flower_is_open ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanFlower_open_white.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanFlower_closed_white.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void BeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

    if( shoot_node_index<=4 ) {

        //set leaf and internode scale based on position along the shoot
        float leaf_scale = fmin(1.f, 0.75 + 0.25 * float(shoot_node_index) / 4.f);
        phytomer->scaleLeafPrototypeScale(leaf_scale);

        //set internode length based on position along the shoot
        float inode_scale = fmin(1.f, 0.1 + 0.9 * float(shoot_node_index) / 4.f);
        phytomer->scaleInternodeMaxLength(inode_scale);

    }

}

uint BindweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BindweedFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedFlower.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CheeseweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CheeseweedLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    if( compound_leaf_index<0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_left_lowres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else if( compound_leaf_index==0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_tip_lowres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_right_lowres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
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

    if( shoot_node_index<=5 ) {

        //set leaf and internode scale based on position along the shoot
        float leaf_scale = fmin(1.f, 0.4 + 0.6 * float(shoot_node_index) / 5.f);
        phytomer->scaleLeafPrototypeScale(leaf_scale);

        //set internode length based on position along the shoot
        float inode_scale = fmin(1.f, 0.1 + 0.9 * float(shoot_node_index) / 5.f);
        phytomer->scaleInternodeMaxLength(inode_scale);

        phytomer->setFloralBudState( BUD_DEAD );

    }else{
        phytomer->setVegetativeBudState( BUD_DEAD );
    }

}

uint PuncturevineLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/PuncturevineLeaf.png";

    float leaf_aspect = 0.4; //ratio of leaf width to leaf length

    float midrib_fold = 0.2; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = -0.1; //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = 0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    uint objID = buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold, x_curvature, y_curvature, 0, 0, 0);

    return objID;
}

uint PuncturevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/PuncturevineFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint SorghumLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SorghumLeaf.png";

    float leaf_aspect = 0.25; //ratio of leaf width to leaf length

    float midrib_fold_fraction = 0.3;

    float x_curvature = -0.6;
    float y_curvature = -0.3;

    float petiole_roll = 0.05f;

    //parameters for leaf wave/wrinkles
    float wave_period = 0.1f; //period factor of leaf waves
    float wave_amplitude = 0.03f; // amplitude of leaf waves

    uint objID = buildGenericLeafPrototype(context_ptr, subdivisions, leaf_texture, leaf_aspect, midrib_fold_fraction, x_curvature, y_curvature, petiole_roll, wave_period, wave_amplitude);

    return objID;

}

uint SorghumPaniclePrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set ){

    float panicle_height = 1;
    float panicle_width = 0.25;
    int panicle_subdivisions = 5;

    std::string seed_texture_file = "plugins/plantarchitecture/assets/textures/SorghumSeed.png";

    std::vector<uint> UUIDs;

    std::vector<vec3> nodes_panicle;
    std::vector<float> radius_panicle;
    float adj = 20; // scale factor to match the length of meters
    float width_panicle = panicle_width * 10 * adj;

    for (float n = 0; n < width_panicle ; n++) {
        float x = 0;
        float y = 0;
        float dz = n * (0.01);
        float z = dz;
        float angle = n * M_PI / width_panicle;
        float dr = 0.01 * sin(angle);

        nodes_panicle.push_back(make_vec3(x, y, z));
        radius_panicle.push_back(dr);

    }

    std::vector<uint> UUIDs2 = context_ptr->addTube( panicle_subdivisions, nodes_panicle, radius_panicle,seed_texture_file.c_str());

    float z_value = 0;

    float di = 0;

    for (float i = (panicle_height*adj) + 2; i > -1; i--) {
        
        std::vector<uint> UUIDs_copy = context_ptr->copyPrimitive(UUIDs2);

        float rotation_angle;

        if (i > (((panicle_height*adj) + 2) /3.f )) {
            rotation_angle = 0.26;
        } else {
            rotation_angle = i * 0.0867;
        }
        float dz = 0.032;

        z_value = z_value + dz;
        vec3 tra1(0, 0, z_value - dz);
        vec3 base = nullorigin;//interpolateTube(nodes_panicle_stalk, 0.05);
        float rot1 = rotation_angle;

        context_ptr->rotatePrimitive(UUIDs_copy, rot1, "y");
        context_ptr->translatePrimitive(UUIDs_copy, base);
        context_ptr->translatePrimitive(UUIDs_copy, tra1);


        UUIDs.insert( UUIDs.end(), UUIDs_copy.begin(), UUIDs_copy.end() );

        float i_value_1, i_value_2;

        if (di == 0) {
            i_value_1 = 6;
            i_value_2 = 60;
            di = 1;
        } else {
            i_value_1 = 5;
            i_value_2 = 72;
            di = 0;
        }

        for (int ii = 0; ii < i_value_1; ii++) {
            UUIDs_copy = context_ptr->copyPrimitive(UUIDs_copy);
            UUIDs.insert( UUIDs.end(), UUIDs_copy.begin(), UUIDs_copy.end() );
//            s5_UUID_panicle_plant.front().push_back(context_ptr->copyPrimitive(UUIDs_copy));
            float rot2 = ii * i_value_2 * M_PI / float(180);
            context_ptr -> rotatePrimitive( UUIDs_copy, rot2, "z");

        }

    }


    context_ptr->deletePrimitive(UUIDs2);
    ;
//    vec3 V_1 = interpolateTube(nodes_stem, 1);
//    vec3 V_2 = interpolateTube(nodes_stem, 0.95);
//    vec3 V_3  = V_1 - V_2;
//
//    float Z = cart2sphere(V_3).zenith;
//
//    context_ptr->rotatePrimitive( UUIDs, Z, "y");

    context_ptr->translatePrimitive(UUIDs, make_vec3(0,0,panicle_height));

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

void SorghumPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age ){

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

uint SoybeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint SoybeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/SoybeanLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;

}

uint SugarbeetLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){

    // -- Adjustable parameters -- //

    std::string leaf_texture = "plugins/plantarchitecture/assets/textures/SugarbeetLeaf.png";

    float leaf_aspect = 0.4; //ratio of leaf width to leaf length

    float midrib_fold = 0.1; //fraction of folding along midrib (=0 leaf is flat, =1 leaf is completely folded in half)

    float x_curvature = -0.2; //curvature factor along x-direction. (+curves upward, -curved downward)

    float y_curvature = 0.4; //curvature factor along y-direction. (+curves upward, -curved downward)

    float petiole_roll = 0.75; //add a small radius roll at the based of the leaf to better mate with the petiole. Value is the magnitude of the roll (+rolls upward, - rolls downward)

    //parameters for leaf wave/wrinkles
    float wave_period = context_ptr->randu( 0.08f, 0.15f); //period factor of leaf waves
    float wave_amplitude = context_ptr->randu(0.02f,0.04f); // amplitude of leaf waves

    // -- main leaf generation code -- //

    std::vector<uint> UUIDs;

    int Nx = subdivisions; //number of leaf subdivisions in the x-direction (longitudinal)
    int Ny = ceil(leaf_aspect*float(Nx)); //number of leaf subdivisions in the y-direction (lateral)

    if ( Ny % 2 != 0){ //Ny must be even
        Ny = Ny + 1;
    }

    float dx = 1.f/float(Nx); //length of leaf subdivision in the x-direction
    float dy = leaf_aspect/float(Ny); //length of leaf subdivision in the y-direction

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {

            float x = float(i) * dx; //x-coordinate of leaf subdivision
            float y = float(j) * dy - 0.5f * leaf_aspect; //y-coordinate of leaf subdivision

            //midrib leaf folding
            float y_fold = cosf(0.5f*midrib_fold*M_PI)*y;
            float y_fold_jplus = cosf(0.5f*midrib_fold*M_PI)*(y+dy);
            float z_fold = sinf(0.5f*midrib_fold*M_PI)*fabs(y);
            float z_fold_jplus = sinf(0.5f*midrib_fold*M_PI)*fabs(y+dy);

            //x-curvature
            float z_xcurve = x_curvature*powf(x,4);
            float z_xcurve_iplus = x_curvature*powf(x+dx,4);

            //y-curvature
            float z_ycurve = y_curvature*powf(y/leaf_aspect,4);
            float z_ycurve_jplus = y_curvature*powf((y+dy)/leaf_aspect,4);

            //petiole roll
            float z_petiole_0 = fmin(0.1f,petiole_roll*powf(7.f*y/leaf_aspect,4)*exp(-70.f*(x))) - 0.01*petiole_roll/fabs(petiole_roll);
            float z_petiole_1 = fmin(0.1f,petiole_roll*powf(7.f*y/leaf_aspect,4)*exp(-70.f*(x+dx))) - 0.01*petiole_roll/fabs(petiole_roll);
            float z_petiole_2 = fmin(0.1f,petiole_roll*powf(7.f*(y+dy)/leaf_aspect,4)*exp(-70.f*(x+dx))) - 0.01*petiole_roll/fabs(petiole_roll);
            float z_petiole_3 = fmin(0.1f,petiole_roll*powf(7.f*(y+dy)/leaf_aspect,4)*exp(-70.f*(x))) - 0.01*petiole_roll/fabs(petiole_roll);

            //vertical displacement for leaf wave at each of the four subdivision vertices
            float z_wave_0 = (2.f*fabs(y)*wave_amplitude * sinf((x+0.5f*int(j>=0.5*Ny)) * M_PI / wave_period));
            float z_wave_1 = (2.f*fabs(y)*wave_amplitude * sinf((x +0.5f*int(j>=0.5*Ny) + dx) * M_PI / wave_period));
            float z_wave_2 = (2.f*fabs(y+dy)*wave_amplitude * sinf((x +0.5f*int(j>=0.5*Ny) + dx) * M_PI / wave_period));
            float z_wave_3 = (2.f*fabs(y+dy)*wave_amplitude * sinf((x+0.5f*int(j>=0.5*Ny)) * M_PI / wave_period));

            //define the four vertices of the leaf subdivision
            vec3 v0(x, y_fold, z_fold+z_xcurve+z_ycurve+z_wave_0+z_petiole_0);

            vec3 v1(x + dx, y_fold, z_fold+z_xcurve_iplus+z_ycurve+z_wave_1+z_petiole_1);

            vec3 v2(x + dx, y_fold_jplus, z_fold_jplus+z_xcurve_iplus+z_ycurve_jplus+z_wave_2+z_petiole_2);

            vec3 v3(x, y_fold_jplus, z_fold_jplus+z_xcurve+z_ycurve_jplus+z_wave_3+z_petiole_3);

            vec2 uv0(x, (y+0.5f*leaf_aspect) / leaf_aspect);
            vec2 uv1(x + dx, (y+0.5f*leaf_aspect) / leaf_aspect);
            vec2 uv2(x + dx, (y + dy + 0.5f*leaf_aspect) / leaf_aspect);
            vec2 uv3(x, (y + dy + 0.5f*leaf_aspect) / leaf_aspect);

            UUIDs.push_back(context_ptr->addTriangle(v0, v1, v2, leaf_texture.c_str(), uv0, uv1, uv2));
            UUIDs.push_back(context_ptr->addTriangle(v0, v2, v3, leaf_texture.c_str(), uv0, uv2, uv3));

        }
    }

    // --- Building from an OBJ file --- //

//     UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/SugarbeetLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    return context_ptr->addPolymeshObject( UUIDs );
}

uint TomatoLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index, uint shoot_node_index, uint shoot_max_nodes ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
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