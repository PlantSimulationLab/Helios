

#include "PlantArchitecture.h"

using namespace helios;

uint BeanLeafPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
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
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanPod.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint BeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/BeanFlower.obj", make_vec3(0.0,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaLeafPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs;
    if( flag<0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_left_lowres.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else if( flag==0 ){
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_tip.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }else{
        UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaLeaf_right.obj", make_vec3(0.,0,0), 0, nullrotation, RGB::black, "ZUP", true );
    }
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaFruitPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/CowpeaPod.obj", make_vec3(0.,0,0), 0.75,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}

uint CowpeaFlowerPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
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

uint TomatoFlowerPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
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

uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions, int flag ){
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/AlmondFlower.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
    uint objID = context_ptr->addPolymeshObject( UUIDs );
    return objID;
}