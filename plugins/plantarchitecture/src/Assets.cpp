

#include "PlantArchitecture.h"

using namespace helios;

uint BeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs;
    if( flag==0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_tip.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else if( flag<0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_left.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_right.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanFruitPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
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

uint SoybeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint SoybeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/SoybeanLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;

}

uint CowpeaLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_unifoliate.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );

    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs;
    if( flag<0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_left_lowres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else if( flag==0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_tip_lowres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_right_lowres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaFruitPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaPod.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaFlower_open_yellow.obj", make_vec3(0.0,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint TomatoLeafPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint TomatoFruitPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoFruit.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint TomatoFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/TomatoFlower.obj", make_vec3(0.0,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint AlmondLeafPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint AlmondFruitPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondHull.obj", make_vec3(0.,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CheeseweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CheeseweedLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BindweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedLeaf.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BindweedFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BindweedFlower.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint SorghumLeafPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){

    std::vector<uint> UUIDs;

    float leaf_aspect = 0.1; //ratio of leaf width to leaf length

    //parameters for leaf wave/wrinkles
    float wave_period = 0.1f; //period factor of leaf waves
    float wave_amplitude = 0.015f; // amplitude of leaf waves

    //parameters for leaf curvature
    float leaf_amplitude = 0.25f; //amplitude of leaf curvature

    int Nx = 50; //number of leaf subdivisions in the x-direction (longitudinal)
    int Ny = ceil(leaf_aspect*float(Nx)); //number of leaf subdivisions in the y-direction (lateral)

    if ( Ny % 2 != 0){ //Ny must be even
        Ny = Ny + 1;
    }

    float dx = 1.f/float(Nx); //length of leaf subdivision in the x-direction
    float dy = leaf_aspect/float(Ny); //length of leaf subdivision in the y-direction

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {

            float y_frac, y_frac_plus;
            if (float(j) / float(Ny) >= 0.5) {
                y_frac = (1 - float(j) / float(Ny)) * 2;
                y_frac_plus = y_frac - 2.f / float(Ny);
            } else {
                y_frac = float(j) / float(Ny) * 2;
                y_frac_plus = y_frac + 2.f / float(Ny);
            }
            y_frac = 1.f - y_frac;
            y_frac_plus = 1.f - y_frac_plus;

            float x = float(i) * dx; //x-coordinate of leaf subdivision
            float y = float(j) * dy; //y-coordinate of leaf subdivision
            float z = 0;

            //vertical displacement for leaf wave at each of the four subdivision vertices
            float z_i = x * M_PI / wave_period;
            float z_iplus = (x + dx) * M_PI / wave_period;

            float leaf_wave_zoffset_0 = (wave_amplitude * sinf(z_i) + 0.01f) * y_frac;
            float leaf_wave_zoffset_1 = (wave_amplitude * sinf(z_iplus) + 0.01f) * y_frac;
            float leaf_wave_zoffset_2 = (wave_amplitude * sinf(z_iplus) + 0.01f) * y_frac_plus;
            float leaf_wave_zoffset_3 = (wave_amplitude * sinf(z_i) + 0.01f) * y_frac_plus;

            //vertical displacement due to leaf curvature at each of the four subdivision vertices
            float x_i = x * M_PI;
            float x_iplus = (x + dx) * M_PI;

            float leaf_curvature_zoffset_0 = leaf_amplitude * sin(x_i);
            float leaf_curvature_zoffset_1 = leaf_amplitude * sin(x_iplus);
            float leaf_curvature_zoffset_2 = leaf_amplitude * sin(x_iplus);
            float leaf_curvature_zoffset_3 = leaf_amplitude * sin(x_i);

            //define the four vertices of the leaf subdivision
            z = leaf_curvature_zoffset_0 + leaf_wave_zoffset_0;
            vec3 v0(x, y - 0.5f * leaf_aspect, z);

            z = leaf_curvature_zoffset_1 + leaf_wave_zoffset_1;
            vec3 v1(x + dx, y - 0.5f * leaf_aspect, z);

            z = leaf_curvature_zoffset_2 + leaf_wave_zoffset_2;
            vec3 v2(x + dx, y + dy - 0.5f * leaf_aspect, z);

            z = leaf_curvature_zoffset_3 + leaf_wave_zoffset_3;
            vec3 v3(x, y + dy - 0.5f * leaf_aspect, z);

            vec2 uv0(x, y / leaf_aspect);
            vec2 uv1(x + dx, y / leaf_aspect);
            vec2 uv2(x + dx, (y + dy) / leaf_aspect);
            vec2 uv3(x, (y + dy) / leaf_aspect);

            UUIDs.push_back(context_ptr->addTriangle(v0, v1, v2, "plugins/plantarchitecture/assets/textures/SorghumLeaf.png", uv0, uv1, uv2));
            UUIDs.push_back(context_ptr->addTriangle(v0, v2, v3, "plugins/plantarchitecture/assets/textures/SorghumLeaf.png", uv0, uv2, uv3));

        }
    }

    return context_ptr->addPolymeshObject( UUIDs );
}

uint SorghumPaniclePrototype( helios::Context* context_ptr, uint subdivisions, int flag ){

    uint objID;// = context_ptr->addPolymeshObject( UUIDs );
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