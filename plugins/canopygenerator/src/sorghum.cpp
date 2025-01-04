#include "CanopyGenerator.h"

using namespace std;
using namespace helios;


// SORGHUM PLANT
uint CanopyGenerator::sorghum( const SorghumCanopyParameters &params, const vec3 &origin ) {
        // STAGE 1
        if (params.sorghum_stage ==1){

            std::vector<vec3> node_stem;
                vector<RGBcolor> Color_stem;

                std::vector<float> radius_stem;

                //map the route for the stem
                float stem_sects = params.s1_stem_length * 0.1; // split the stem length into 10 parts

                for( float n=0; n< params.s1_stem_length + stem_sects; n = n+stem_sects){
                    float y = 0 ;
                    float x = 0;
                    float z =  n;

                    node_stem.push_back(make_vec3(x,y,z) + origin);
                    radius_stem.push_back( params.s1_stem_radius);
                    Color_stem.push_back(make_RGBcolor(0.44,0.58,0.19));
                }

            std::vector<std::vector<uint> > s1_UUID_stem_plant;
            s1_UUID_stem_plant.resize(1);
            std::vector<uint> UUID_stem= context->addTube(params.s1_stem_subdivisions, node_stem, radius_stem, Color_stem);
            std::vector<uint> UUID_stem_top = context->addDisk(params.s1_stem_subdivisions,make_vec3(0,0,0),make_vec2(params.s1_stem_radius,params.s1_stem_radius),make_SphericalCoord( 0, 0, 0),RGBcolor(0.44,0.58,0.19));

            vec3 base_top = interpolateTube(node_stem, 1);
            context->translatePrimitive(UUID_stem_top, base_top);

            s1_UUID_stem_plant.push_back(context->copyPrimitive(UUID_stem));
            s1_UUID_stem_plant.push_back(context->copyPrimitive(UUID_stem_top));
            context->deletePrimitive(UUID_stem);
            context->deletePrimitive(UUID_stem_top);


                float angle = ((context->randu(0, 90) * PI_F) / float(180));
                for (float i = 1; i < 4; i++){
                    float leaf_length, leaf_width,rotation_value1, leaf_bend, rotation_value2,frac, leaf_curve, x_adj;

                    if (i == 1){
                        leaf_length =  params.s1_leaf_size1.x;
                        leaf_width =  params.s1_leaf_size1.y;
                        rotation_value1 = 0; // rotation in the y axis
                        leaf_bend = leaf_length * 0.2;
                        rotation_value2 = params.s1_leaf1_angle; // leaf rotation on the z axis
                        frac= 0.9;
                        leaf_curve = leaf_width * 0.25;
                        x_adj = 0;

                    }else if (i == 2){
                        leaf_length =  params.s1_leaf_size2.x;
                        leaf_width =  params.s1_leaf_size2.y;
                        rotation_value1 = PI_F;
                        leaf_bend = leaf_length * 0.2;
                        rotation_value2 = params.s1_leaf2_angle;
                        frac= 0.9;
                        leaf_curve = leaf_width * 0.25;
                        x_adj = 0;

                    }else{
                        leaf_length =  params.s1_leaf_size3.x;
                        leaf_width =  params.s1_leaf_size3.y;
                        rotation_value1 = PI_F*0.5;
                        leaf_bend = leaf_length * 0.2;
                        rotation_value2 = params.s1_leaf3_angle;
                        frac= 0.5;
                        leaf_curve = leaf_width * 0.25;
                        x_adj = 0;
                    }

                    std::vector<uint> UUIDs4;

                    float Nx = params.s1_leaf_subdivisions.x;
                    float Ny;

                    if ( params.s1_leaf_subdivisions.y % 2 == 0){
                        Ny = params.s1_leaf_subdivisions.y;
                    } else {
                        Ny = params.s1_leaf_subdivisions.y + 1;
                    }


                    float dx = leaf_length/Nx;
                    float dy = leaf_width/Ny;

                    float A_3 = leaf_length / float(0.08); // Half waves on the leaf

                    // leaf wave
                    float A_2 = leaf_length / float(80); // amplitude of each leaf wave

                    for (int i = 0; i < Nx; i++) {
                        for (float j = 0; j < Ny; j++) {

                            float frac = 2/Ny;
                            float y_frac = j/Ny;
                            float Ny_frac_1, Ny_frac_2;

                            if (y_frac > 0.5) {
                                Ny_frac_1 = (1 - y_frac) * 2;
                                Ny_frac_2 = Ny_frac_1 - frac;

                            }else if (y_frac < 0.5) {
                                Ny_frac_1 = y_frac * 2;
                                Ny_frac_2 = Ny_frac_1 + frac;
                            }else {
                                Ny_frac_1 = (1 - y_frac) * 2;
                                Ny_frac_2 = Ny_frac_1 - frac;
                            }

                            float x = i * dx;
                            float y = j * dy;
                            float z = 0;
                            float sx = dx;
                            float sy = dy;

                            float x_i = x * PI_F /(Nx * dx);
                            float sx_i = (x + sx) * PI_F /(Nx * dx);

                            float z_1 = (x * PI_F) /((Nx * dx) /(A_3));
                            float z_2 = ((x + sx) * PI_F) /((Nx * dx) /(A_3));

                            float leaf_wave_1;
                            float leaf_wave_2;
                            float leaf_wave_3;
                            float leaf_wave_4;

                            if (j == 0) {
                                leaf_wave_1 = A_2 * sin(z_1);
                                leaf_wave_2 = A_2 * sin(z_2);
                            } else {
                                leaf_wave_1 = -leaf_curve * Ny_frac_1;
                                leaf_wave_2 = -leaf_curve * Ny_frac_1;
                            }

                            z = leaf_bend * sin(x_i) + leaf_wave_1;
                            vec3 v0(x, y, z);

                            z = leaf_bend * sin(sx_i) + leaf_wave_2;
                            vec3 v1(x + sx, y, z);

                            if (j == Ny - 1) {
                                leaf_wave_3 = A_2 * sin(z_2);
                                leaf_wave_4 = A_2 * sin(z_1);
                            } else {
                                leaf_wave_3 = -leaf_curve * Ny_frac_2;
                                leaf_wave_4 = -leaf_curve * Ny_frac_2;

                            }

                            z = leaf_bend * sin(sx_i) + leaf_wave_3;
                            vec3 v2(x + sx, y + sy, z);

                            z = leaf_bend * sin(x_i) + leaf_wave_4;
                            vec3 v3(x, y + sy, z);

                            vec2 uv0(x / (Nx * dx), y /(leaf_width));
                            vec2 uv1((x + sx) /(Nx * dx), y /(leaf_width));
                            vec2 uv2((x + sx) /(Nx * dx), (y + sy) / (leaf_width));
                            vec2 uv3(x / (Nx * dx), (y + sy) /(leaf_width));

                            UUIDs4.push_back(context->addTriangle(v0, v1, v2, params.s1_leaf_texture_file.c_str(), uv0, uv1, uv2));
                            UUIDs4.push_back(context->addTriangle(v0, v2, v3, params.s1_leaf_texture_file.c_str(), uv0, uv2, uv3));
                        }
                    }

                    std::vector<std::vector<uint> > s1_UUID_leaf_plant;
                    s1_UUID_leaf_plant.push_back(context->copyPrimitive(UUIDs4));
                    context->deletePrimitive(UUIDs4);

                    vec3 translation(-x_adj, -leaf_width*0.5, 0);
                    float rotation_1 = -rotation_value2 * PI_F / float(180);

                    float rotation_2 = angle + rotation_value1;
                    vec3 base = interpolateTube(node_stem, frac);

                    context->translatePrimitive(s1_UUID_leaf_plant.back(), translation);
                    context->rotatePrimitive(s1_UUID_leaf_plant.back(), rotation_1, "y");
                    context->rotatePrimitive(s1_UUID_leaf_plant.back(), rotation_2, "z");
                    context->translatePrimitive(s1_UUID_leaf_plant.back(), base);


                    vector<vector<vector<uint>>> s1_UUID_panicle_plant; // empty vector
                    std::vector<uint> s1_UUID_branch_plant;    // empty vector

                    UUID_trunk.push_back(s1_UUID_stem_plant.front());
                    UUID_leaf.push_back(s1_UUID_leaf_plant);
                    UUID_fruit.push_back(s1_UUID_panicle_plant);
                    UUID_branch.push_back(s1_UUID_branch_plant);


                }

            // STAGE 2
            }else if (params.sorghum_stage == 2){

            std::vector<vec3> node_stem;
            vector<RGBcolor> Color_stem;

            std::vector<float> radius_stem;

            //map the route for the stem
            float stem_sects = params.s2_stem_length * 0.1;

            for( float n=0; n< params.s2_stem_length + stem_sects; n = n+stem_sects){
                float y = 0 ;
                float x = 0;
                float z =  n;

                node_stem.push_back(make_vec3(x,y,z) + origin);
                radius_stem.push_back( params.s2_stem_radius);
                Color_stem.push_back(make_RGBcolor(0.25,0.36,0.19));
            }

            std::vector<std::vector<uint> > s2_UUID_stem_plant;
            s2_UUID_stem_plant.resize(1);
            std::vector<uint> UUID_stem= context->addTube(50, node_stem, radius_stem, Color_stem);
            std::vector<uint> UUID_stem_top = context->addDisk(50,make_vec3(0,0,0),make_vec2(params.s2_stem_radius,params.s2_stem_radius),make_SphericalCoord( 0, 0, 0),RGBcolor(0.302,0.4314,0.2392));

            vec3 base_top = interpolateTube(node_stem, 1);
            context->translatePrimitive(UUID_stem_top, base_top);

            s2_UUID_stem_plant.push_back(context->copyPrimitive(UUID_stem));
            s2_UUID_stem_plant.push_back(context->copyPrimitive(UUID_stem_top));
            context->deletePrimitive(UUID_stem);
            context->deletePrimitive(UUID_stem_top);

            float angle = ((context->randu(0, 90) * PI_F) / float(180));
            for (float i = 1; i < 6; i++){
                float leaf_length, leaf_width,rotation_value1, leaf_bend,rotation_value2,frac, leaf_curve, leaf_wave_no, leaf_wave;

                if (i == 1){
                    leaf_length = params.s2_leaf_size1.x;
                    leaf_width = params.s2_leaf_size1.y;
                    rotation_value1 = 0;
                    leaf_bend = leaf_length * 0.2;
                    rotation_value2 = 25;
                    frac= 0.98;
                    leaf_curve = leaf_width * 0.25;
                    leaf_wave_no = 50;
                    leaf_wave = 0.0125;

                }else if (i == 2) {
                    leaf_length = params.s2_leaf_size2.x;
                    leaf_width = params.s2_leaf_size2.y;
                    rotation_value1 = PI_F;
                    leaf_bend = leaf_length * 0.2;
                    rotation_value2 = 50;
                    frac = 0.98;
                    leaf_curve = leaf_width * 0.5;
                    leaf_wave_no = 40;
                    leaf_wave = 0.02;

                }else if (i == 3) {
                    leaf_length = params.s2_leaf_size3.x;
                    leaf_width = params.s2_leaf_size3.y;
                    rotation_value1 = PI_F + ((context->randu(0, 45) * PI_F) / float(180));
                    leaf_bend = leaf_length * 0.3;
                    rotation_value2 = 15;
                    frac = 0.75;
                    leaf_curve = leaf_width * 0.5;
                    leaf_wave_no = 40;
                    leaf_wave = 0.015;

                }else if (i == 4){
                    leaf_length = params.s2_leaf_size4.x;
                    leaf_width = params.s2_leaf_size4.y;
                    rotation_value1 = ((context->randu(0, 45) * PI_F) / float(180));
                    leaf_bend = leaf_length * 0.3;
                    rotation_value2 = 25;
                    frac= 0.5;
                    leaf_curve = leaf_width * 0.5;
                    leaf_wave_no = 40;
                    leaf_wave = 0;

                }else{
                    leaf_length = params.s2_leaf_size5.x;
                    leaf_width = params.s2_leaf_size5.y;
                    rotation_value1 = PI_F + ((context->randu(0, 45) * PI_F) / float(180));
                    leaf_bend = leaf_length * 0.3;
                    rotation_value2 = 10;
                    frac= 0.25;
                    leaf_curve = leaf_width * 0.5;
                    leaf_wave_no = 40;
                    leaf_wave = 0;
                }

                std::vector<std::vector<uint> > s2_UUID_leaf_plant;

                std::vector<uint> UUIDs4;

                float Nx = params.s2_leaf_subdivisions.x;
                float Ny;

                if ( params.s2_leaf_subdivisions.y % 2 == 0){
                    Ny = params.s2_leaf_subdivisions.y;
                } else {
                    Ny = params.s2_leaf_subdivisions.y + 1;
                }

                float dx = leaf_length /Nx;
                float dy = leaf_width /Ny;

                float A_3 = leaf_length * leaf_wave_no; // Half waves on the leaf

                // leaf wave
                float A_2 = leaf_length * leaf_wave; // amplitude of each leaf wave

                for (int i = 0; i < Nx; i++) {
                    for (float j = 0; j < Ny; j++) {

                        float frac = 2/Ny;
                        float y_frac = j/Ny;
                        float Ny_frac_1, Ny_frac_2;

                        if (y_frac > 0.5) {
                            Ny_frac_1 = (1 - y_frac) * 2;
                            Ny_frac_2 = Ny_frac_1 - frac;

                        }else if (y_frac < 0.5) {
                            Ny_frac_1 = y_frac * 2;
                            Ny_frac_2 = Ny_frac_1 + frac;
                        }else {
                            Ny_frac_1 = (1 - y_frac) * 2;
                            Ny_frac_2 = Ny_frac_1 - frac;
                        }

                        float x = i * dx;
                        float y = j * dy;
                        float z = 0;
                        float sx = dx;
                        float sy = dy;

                        float x_i = x * PI_F /(Nx * dx);
                        float sx_i = (x + sx) * PI_F /(Nx * dx);

                        float z_1 = (x * PI_F) /((Nx * dx) /(A_3));
                        float z_2 = ((x + sx) * PI_F) /((Nx * dx) /(A_3));

                        float leaf_wave_1;
                        float leaf_wave_2;
                        float leaf_wave_3;
                        float leaf_wave_4;

                        if (j == 0) {
                            leaf_wave_1 = A_2 * sin(z_1);
                            leaf_wave_2 = A_2 * sin(z_2);
                        } else {
                            leaf_wave_1 = -leaf_curve * Ny_frac_1;
                            leaf_wave_2 = -leaf_curve * Ny_frac_1;
                        }

                        z = leaf_bend * sin(x_i) + leaf_wave_1;
                        vec3 v0(x, y, z);

                        z = leaf_bend * sin(sx_i) + leaf_wave_2;
                        vec3 v1(x + sx, y, z);

                        if (j == Ny - 1) {
                            leaf_wave_3 = A_2 * sin(z_2);
                            leaf_wave_4 = A_2 * sin(z_1);
                        } else {
                            leaf_wave_3 = -leaf_curve * Ny_frac_2;
                            leaf_wave_4 = -leaf_curve * Ny_frac_2;

                        }

                        z = leaf_bend * sin(sx_i) + leaf_wave_3;
                        vec3 v2(x + sx, y + sy, z);

                        z = leaf_bend* sin(x_i) + leaf_wave_4;
                        vec3 v3(x, y + sy, z);

                        vec2 uv0(x /(Nx * dx), y /(leaf_width));
                        vec2 uv1((x + sx) /(Nx * dx), y /(leaf_width));
                        vec2 uv2((x + sx) /(Nx * dx), (y + sy) /(leaf_width));
                        vec2 uv3(x /(Nx * dx), (y + sy) /(leaf_width));

                        UUIDs4.push_back(context->addTriangle(v0, v1, v2, params.s2_leaf_texture_file.c_str(), uv0, uv1, uv2));
                        UUIDs4.push_back(context->addTriangle(v0, v2, v3, params.s2_leaf_texture_file.c_str(), uv0, uv2, uv3));
                    }
                }

                s2_UUID_leaf_plant.push_back(context->copyPrimitive(UUIDs4));
                context->deletePrimitive(UUIDs4);

                vec3 translation(- 0.00015* rotation_value2, -leaf_width*0.5, 0);  // adjustment v0 amplitude,v1 radius v2 leaf_length
                float rotation_1 = -rotation_value2 * PI_F / float(180);

                float rotation_2 = angle + rotation_value1;
                vec3 base = interpolateTube(node_stem, frac);

                context->translatePrimitive(s2_UUID_leaf_plant.back(), translation);
                context->rotatePrimitive(s2_UUID_leaf_plant.back(), rotation_1, "y");
                context->rotatePrimitive(s2_UUID_leaf_plant.back(), rotation_2, "z");
                context->translatePrimitive(s2_UUID_leaf_plant.back(), base);

                vector<vector<vector<uint>>> s2_UUID_panicle_plant; // empty vector
                std::vector<uint> s2_UUID_branch_plant;    // empty vector

                UUID_trunk.push_back(s2_UUID_stem_plant.front());
                UUID_leaf.push_back(s2_UUID_leaf_plant);
                UUID_fruit.push_back(s2_UUID_panicle_plant);
                UUID_branch.push_back(s2_UUID_branch_plant);

            }

        }

            // STAGE 3
        else if (params.sorghum_stage == 3){
        // STEM
        std::vector<vec3> node_stem;
            vector<RGBcolor> Color_stem;
            std::vector<float> radius_stem;

            //map the route for the stem
            float stem_sects = params.s3_stem_length * 0.1;

            for( float n=0; n<params.s3_stem_length + stem_sects; n = n+stem_sects){
                float y = 0 ;
                float x = 0;
                float z =  n;

                node_stem.push_back(make_vec3(x,y,z) + origin );
                radius_stem.push_back(params.s3_stem_radius);
                Color_stem.push_back(make_RGBcolor(0.302,0.4314,0.2392));
            }

            std::vector<uint> s3_UUID_stem_plant = context->addTube(params.s3_stem_subdivisions, node_stem, radius_stem, Color_stem);

// THE LEAVES

            int nodes_no = params.s3_number_of_leaves;

            float rotation_x1 = (context->randu(0, 360) * PI_F) / float(180);// inclination rotation, different for each plant
            std::vector<std::vector<uint> > s3_UUID_leaf_plant;

            for (int i = 1; i < (nodes_no + 1); i++) {

                std::vector<uint> UUIDs4;

                float Nx = params.s3_leaf_subdivisions.x;
                float Ny;

                if ( params.s3_leaf_subdivisions.y % 2 == 0){
                    Ny = params.s3_leaf_subdivisions.y;
                } else {
                    Ny = params.s3_leaf_subdivisions.y + 1;
                }

                float x;

                if (i >= (nodes_no - 1) || i == 1) {
                    x =  0.75;
                } else {
                    x = 1;
                }

                float dx = params.s3_leaf_size.x * x/Nx;
                float dy = params.s3_leaf_size.y/Ny;

                float A_3 = params.s3_leaf_size.x / float(0.08); // Half waves on the leaf

                // leaf wave
                float A_2 = params.s3_leaf_size.x / float(80); // amplitude of each leaf wave
                float leaf_amplitude = params.s3_leaf_size.x / float(6);

                for (int i = 0; i < Nx; i++) {
                    for (float j = 0; j < Ny; j++) {

                        float frac = 2/Ny;
                        float y_frac = j/Ny;
                        float Ny_frac_1, Ny_frac_2;

                        if (y_frac > 0.5) {
                            Ny_frac_1 = (1 - y_frac) * 2;
                            Ny_frac_2 = Ny_frac_1 - frac;

                        }else if (y_frac < 0.5) {
                            Ny_frac_1 = y_frac * 2;
                            Ny_frac_2 = Ny_frac_1 + frac;
                        }else {
                            Ny_frac_1 = (1 - y_frac) * 2;
                            Ny_frac_2 = Ny_frac_1 - frac;
                        }

                        float x = i * dx;
                        float y = j * dy;
                        float z = 0;
                        float sx = dx;
                        float sy = dy;

                        float x_i = x * PI_F /(Nx * dx);
                        float sx_i = (x + sx) * PI_F /(Nx * dx);

                        float z_1 = (x * PI_F) /((Nx * dx) /(A_3));
                        float z_2 = ((x + sx) * PI_F) /((Nx * dx) /(A_3));

                        float leaf_wave_1;
                        float leaf_wave_2;
                        float leaf_wave_3;
                        float leaf_wave_4;

                        if (j == 0) {
                            leaf_wave_1 = A_2 * sin(z_1);
                            leaf_wave_2 = A_2 * sin(z_2);
                        } else {
                            leaf_wave_1 = -params.s3_leaf_size.y * 0.425 * Ny_frac_1;
                            leaf_wave_2 = -params.s3_leaf_size.y * 0.425 * Ny_frac_1;
                        }

                        z = leaf_amplitude * sin(x_i) + leaf_wave_1;
                        vec3 v0(x, y, z);

                        z = leaf_amplitude * sin(sx_i) + leaf_wave_2;
                        vec3 v1(x + sx, y, z);

                        if (j == Ny - 1) {
                            leaf_wave_3 = A_2 * sin(z_2);
                            leaf_wave_4 = A_2 * sin(z_1);
                        } else {
                            leaf_wave_3 = -params.s3_leaf_size.y * 0.425 * Ny_frac_2;
                            leaf_wave_4 = -params.s3_leaf_size.y * 0.425 * Ny_frac_2;

                        }

                        z = leaf_amplitude * sin(sx_i) + leaf_wave_3;
                        vec3 v2(x + sx, y + sy, z);

                        z = leaf_amplitude * sin(x_i) + leaf_wave_4;
                        vec3 v3(x, y + sy, z);

                        vec2 uv0(x /(Nx * dx), y /(params.s3_leaf_size.y));
                        vec2 uv1((x + sx) /(Nx * dx), y /(params.s3_leaf_size.y));
                        vec2 uv2((x + sx) /(Nx * dx), (y + sy) /(params.s3_leaf_size.y));
                        vec2 uv3(x /(Nx * dx), (y + sy) /(params.s3_leaf_size.y));

                        UUIDs4.push_back(context->addTriangle(v0, v1, v2, params.s3_leaf_texture_file.c_str(), uv0, uv1, uv2));
                        UUIDs4.push_back(context->addTriangle(v0, v2, v3, params.s3_leaf_texture_file.c_str(), uv0, uv2, uv3));
                    }
                }


                s3_UUID_leaf_plant.push_back(context->copyPrimitive(UUIDs4));
                context->deletePrimitive(UUIDs4);

                float frac;

                if (i == 1) {

                    frac = 1;
                } else {

                    frac = 1 - ((i - (context->randu() / float(2))) / float((nodes_no + 1)));
                }

                vec3 base = interpolateTube(node_stem, frac);
                float rotation_1 = - (params.s3_mean_leaf_angle * PI_F / float(180)) - (context->randu(0, 5) * PI_F) / float(180);
                float rotation_x2 = (context->randu(0, 45) * PI_F) / float(180);
                float rotation_2 = rotation_x1 + rotation_x2;
                float rotation_3 = rotation_2 + PI_F; //
                vec3 translation(-params.s3_stem_radius * 2.8, -0.9 / (2 / float(params.s3_leaf_size.y)), 0);  // adjustment

                if (i % 2 != 0) {
                    context->translatePrimitive(s3_UUID_leaf_plant.back(), translation);
                    context->rotatePrimitive(s3_UUID_leaf_plant.back(), rotation_1, "y");
                    context->rotatePrimitive(s3_UUID_leaf_plant.back(), rotation_2, "z");
                    context->translatePrimitive(s3_UUID_leaf_plant.back(), base);

                } else {
                    context->translatePrimitive(s3_UUID_leaf_plant.back(), translation);
                    context->rotatePrimitive(s3_UUID_leaf_plant.back(), rotation_1, "y");
                    context->rotatePrimitive(s3_UUID_leaf_plant.back(), rotation_3, "z");
                    context->translatePrimitive(s3_UUID_leaf_plant.back(), base);
                }

            }

        vector<vector<vector<uint>>> s3_UUID_panicle_plant; // empty vector
        std::vector<uint> s3_UUID_branch_plant;    // empty vector

        UUID_trunk.push_back(s3_UUID_stem_plant);
        UUID_leaf.push_back(s3_UUID_leaf_plant);
        UUID_fruit.push_back(s3_UUID_panicle_plant);
        UUID_branch.push_back(s3_UUID_branch_plant);

    }
            // STAGE 4
    else if (params.sorghum_stage == 4){
        // THE STEM
        std::vector<vec3> node_stem;

            vector<RGBcolor> Color_stem;

            std::vector<float> radius_stem;

            //map the route for the stem
            float stem_sects = params.s4_stem_length * 0.1;

            for( float n=0; n<params.s4_stem_length + stem_sects; n = n+stem_sects){
                float y = 0 ;
                float x = 0;
                float z =  n;

                node_stem.push_back(make_vec3(x,y,z) + origin );
                radius_stem.push_back(params.s4_stem_radius);
                Color_stem.push_back(make_RGBcolor(0.302,0.4314,0.2392));

            }

            std::vector<uint> s4_UUID_stem_plant = context->addTube(params.s4_stem_subdivisions, node_stem, radius_stem, Color_stem); // 50

        // THE PANICLE

            vector<vector<vector<uint>>> s4_UUID_panicle_plant;
            s4_UUID_panicle_plant.resize(1);
            vector<RGBcolor> Color_panicle_stalk;
            std::vector<vec3> nodes_panicle_stalk;
            std::vector<float> radius_panicle_stalk;

            //The panicle stalk (the stem of the panicle)
            float m = 0.125;
            for (float n = 0; n < m; n = n + 0.0125) {
                float y = 0;
                float x = 0;
                float z = n;

                nodes_panicle_stalk.push_back(make_vec3(x, y, z));
                radius_panicle_stalk.push_back(params.s4_stem_radius);

                Color_panicle_stalk.push_back(make_RGBcolor(0.302, 0.4314, 0.2392));

            }

            std::vector<uint> UUID1 = context->addTube(params.s4_stem_subdivisions, nodes_panicle_stalk, radius_panicle_stalk, Color_panicle_stalk);

            s4_UUID_panicle_plant.front().push_back(context->copyPrimitive(UUID1));
            context->deletePrimitive(UUID1);

            // now the panicle
            std::vector<vec3> nodes_panicle;
            std::vector<float> radius_panicle;
            float adj = 20; // scale factor to match the length of meters
            float width_panicle = params.s4_panicle_size.y * 10 * adj;

            for (float n = 0; n < (width_panicle); n++) { // Ball-shaped tubes which form the panicle
                float x = 0;
                float y = 0;
                float dz = n * (0.01);
                float z = dz;
                float angle = n * PI_F /width_panicle;
                float dr = 0.01 * sin(angle);

                nodes_panicle.push_back(make_vec3(x, y, z));
                radius_panicle.push_back(dr);

            }

            std::vector<uint> UUIDs2 = context->addTube(params.s4_panicle_subdivisions, nodes_panicle, radius_panicle, params.s4_seed_texture_file.c_str());

            float z_value = 0;
            float di = 0;

            for (int i = (params.s4_panicle_size.x * adj) + 2; i > -1; i--) {
                std::vector<uint> UUIDs2_copy = context->copyPrimitive(UUIDs2);

                float rotation_angle;

                if (i > (((params.s4_panicle_size.x* adj) + 2) / float(3))) {
                    rotation_angle = 0.26;
                } else {
                    rotation_angle = i * 0.0867;
                }
                float dz = 0.032;

                z_value = z_value + dz;
                vec3 tra1(0, 0, z_value - dz);
                vec3 base = interpolateTube(nodes_panicle_stalk, 0.05);
                float rot1 = rotation_angle;

                context->rotatePrimitive(UUIDs2_copy, rot1, "y");
                context->translatePrimitive(UUIDs2_copy, base);
                context->translatePrimitive(UUIDs2_copy, tra1);
                s4_UUID_panicle_plant.front().push_back(UUIDs2_copy);

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
                    s4_UUID_panicle_plant.front().push_back(context->copyPrimitive(UUIDs2_copy));
                    float rot2 = ii * i_value_2 * PI_F / float(180);
                    context -> rotatePrimitive(s4_UUID_panicle_plant.front().back(), rot2, "z");

                }

            };

            context->deletePrimitive(UUIDs2);
            vec3 V_1 = interpolateTube(node_stem, 1);
            context->translatePrimitive(flatten(s4_UUID_panicle_plant), V_1);

// THE LEAVES

            int nodes_no = params.s4_number_of_leaves;

            float rotation_x1 = (context->randu(0, 360) * PI_F) / float(180);// inclination rotation, different for each plant
            std::vector<std::vector<uint> > s4_UUID_leaf_plant;

            for (int i = 1; i < (nodes_no + 1); i++) {

                std::vector<uint> UUIDs4;

                float Nx = params.s4_leaf_subdivisions.x;
                float Ny;

                if ( params.s4_leaf_subdivisions.y % 2 == 0){
                    Ny = params.s4_leaf_subdivisions.y;
                } else {
                    Ny = params.s4_leaf_subdivisions.y + 1;
                }

                float x;

                if (i >= (nodes_no - 1) || i == 1) {

                    x =  0.75;
                } else {

                    x = 1;
                }

                float dx = params.s4_leaf_size.x * x / Nx;
                float dy = params.s4_leaf_size.y / Ny;

                float A_3 = params.s4_leaf_size.x / float(0.08); // Half waves on the leaf 10

                // leaf wave
                float A_2 = params.s4_leaf_size.x / float(80); // amplitude of each leaf wave

                float leaf_amplitude = params.s4_leaf_size.x / float(6);

                for (int i = 0; i < Nx; i++) {
                    for (float j = 0; j < Ny; j++) {

                        float frac = 2/Ny;
                        float y_frac = j/Ny;
                        float Ny_frac_1, Ny_frac_2;

                        if (y_frac > 0.5) {
                            Ny_frac_1 = (1 - y_frac) * 2;
                            Ny_frac_2 = Ny_frac_1 - frac;

                        }else if (y_frac < 0.5) {
                            Ny_frac_1 = y_frac * 2;
                            Ny_frac_2 = Ny_frac_1 + frac;
                        }else {
                            Ny_frac_1 = (1 - y_frac) * 2;
                            Ny_frac_2 = Ny_frac_1 - frac;
                        }

                        float x = i * dx;
                        float y = j * dy;
                        float z = 0;
                        float sx = dx;
                        float sy = dy;

                        float x_i = x * PI_F / (Nx * dx);
                        float sx_i = (x + sx) * PI_F /(Nx * dx);

                        float z_1 = (x * PI_F) /((Nx * dx) /A_3);
                        float z_2 = ((x + sx) * PI_F) /((Nx * dx) /A_3);

                        float leaf_wave_1;
                        float leaf_wave_2;
                        float leaf_wave_3;
                        float leaf_wave_4;

                        if (j == 0) {
                            leaf_wave_1 = A_2 * sin(z_1);
                            leaf_wave_2 = A_2 * sin(z_2);
                        } else {
                            leaf_wave_1 = -params.s4_leaf_size.y*0.425 * Ny_frac_1;
                            leaf_wave_2 = -params.s4_leaf_size.y*0.425 * Ny_frac_1;
                        }

                        z = leaf_amplitude * sin(x_i) + leaf_wave_1;
                        vec3 v0(x, y, z);

                        z = leaf_amplitude * sin(sx_i) + leaf_wave_2;
                        vec3 v1(x + sx, y, z);

                        if (j == Ny - 1) {
                            leaf_wave_3 = A_2 * sin(z_2);
                            leaf_wave_4 = A_2 * sin(z_1);
                        } else  {
                            leaf_wave_3 = -params.s4_leaf_size.y*0.425 * Ny_frac_2;
                            leaf_wave_4 = -params.s4_leaf_size.y*0.425 * Ny_frac_2;
                        }

                        z = leaf_amplitude * sin(sx_i) + leaf_wave_3;
                        vec3 v2(x + sx, y + sy, z);

                        z = leaf_amplitude * sin(x_i) + leaf_wave_4;
                        vec3 v3(x, y + sy, z);

                        vec2 uv0(x /(Nx * dx), y / float(params.s4_leaf_size.y));
                        vec2 uv1((x + sx) / (Nx * dx), y / float(params.s4_leaf_size.y));
                        vec2 uv2((x + sx) /(Nx * dx), (y + sy) / float(params.s4_leaf_size.y));
                        vec2 uv3(x /(Nx * dx), (y + sy) / float(params.s4_leaf_size.y));

                        UUIDs4.push_back(context->addTriangle(v0, v1, v2, params.s4_leaf_texture_file.c_str(), uv0, uv1, uv2));
                        UUIDs4.push_back(context->addTriangle(v0, v2, v3, params.s4_leaf_texture_file.c_str(), uv0, uv2, uv3));
                    }
                }


                s4_UUID_leaf_plant.push_back(context->copyPrimitive(UUIDs4));
                context->deletePrimitive(UUIDs4);

                float frac;

                if (i == 1) {
                    frac = 0.9;
                } else {
                    frac = (1 - ((i - (context->randu() / float(2))) / float((nodes_no + 1)*1.2))) - 0.15;
                }

                vec3 base = interpolateTube(node_stem, frac);
                float rotation_1 = - (params.s4_mean_leaf_angle * PI_F / float(180)) - (context->randu(0, 5) * PI_F) / float(180);
                float rotation_x2 = (context->randu(0, 45) * PI_F) / float(180);
                float rotation_2 = rotation_x1 + rotation_x2;
                float rotation_3 = rotation_2 + PI_F; //
                vec3 translation(-params.s4_stem_radius * 2, -0.9 / (2 / float(params.s4_leaf_size.y)), 0);  // adjustment

                if (i % 2 != 0) {
                    context->translatePrimitive(s4_UUID_leaf_plant.back(), translation);
                    context->rotatePrimitive(s4_UUID_leaf_plant.back(), rotation_1, "y");
                    context->rotatePrimitive(s4_UUID_leaf_plant.back(), rotation_2, "z");
                    context->translatePrimitive(s4_UUID_leaf_plant.back(), base);

                } else {
                    context->translatePrimitive(s4_UUID_leaf_plant.back(), translation);
                    context->rotatePrimitive(s4_UUID_leaf_plant.back(), rotation_1, "y");
                    context->rotatePrimitive(s4_UUID_leaf_plant.back(), rotation_3, "z");
                    context->translatePrimitive(s4_UUID_leaf_plant.back(), base);
                }

            }

       std::vector<uint> s4_UUID_branch_plant;    // empty vector

       UUID_trunk.push_back(s4_UUID_stem_plant);
       UUID_fruit.push_back(s4_UUID_panicle_plant);
       UUID_leaf.push_back(s4_UUID_leaf_plant);
       UUID_branch.push_back(s4_UUID_branch_plant);



    }else{// STAGE 5
        // THE STEM
        std::vector<vec3> nodes_stem;

        std::vector<float> radius_stem;
        vector<RGBcolor> Color_stem;

        float Nx_stem = params.s5_stem_subdivisions;
        float dx = 1.f / float(Nx_stem);

        for (float n = 0; n < params.s5_stem_length* Nx_stem; n++) {
            float mid_height = params.s5_stem_length * 0.5;
            float nx = n * dx;
            float i = nx * (180 / float(params.s5_stem_length)) * PI_F / float(180);
            float i_mid = (mid_height) * (180 / float(params.s5_stem_length)) * PI_F / float(180);
            float x_mid = -params.s5_stem_bend * sin(i_mid);
            float x1, x;
            if (nx <= mid_height) {
                x = 0;
            } else {
                x1 = -params.s5_stem_bend * sin(i);
                x = x1 - x_mid;
            }
            float y = 0;

            float z = nx;

            nodes_stem.push_back(make_vec3(x, y, z) + origin);

            radius_stem.push_back(params.s5_stem_radius);

            Color_stem.push_back(make_RGBcolor(0.302, 0.4314, 0.2392));

        }

        std::vector<uint> s5_UUID_stem_plant = context->addTube(params.s5_stem_subdivisions, nodes_stem, radius_stem,
                                                                       Color_stem); // 50


// THE PANICLE
//    a std::vector<std::vector<std::vector<uint> > > s5_UUID_panicle_plant;
//    a s5_UUID_panicle_plant.resize(1);

        vector<vector<vector<uint>>> s5_UUID_panicle_plant;
        s5_UUID_panicle_plant.resize(1);

        vector<RGBcolor> Color_panicle_stalk;
        std::vector<vec3> nodes_panicle_stalk;
        std::vector<float> radius_panicle_stalk;

        //The panicle stalk (the stem of the panicle)
        float m = 0.125;
        for (float n = 0; n < m; n = n + 0.0125) {
            float y = 0;
            float x = 0;
            float z = n;

            nodes_panicle_stalk.push_back(make_vec3(x, y, z));
            radius_panicle_stalk.push_back(params.s5_stem_radius);
            Color_panicle_stalk.push_back(make_RGBcolor(0.302, 0.4314, 0.2392));

        }

        std::vector<uint> UUID1 = context->addTube(params.s5_stem_subdivisions, nodes_panicle_stalk, radius_panicle_stalk, Color_panicle_stalk);

        s5_UUID_panicle_plant.front().push_back(context->copyPrimitive(UUID1));
        context->deletePrimitive(UUID1);

        // now the panicle
        std::vector<vec3> nodes_panicle;
        std::vector<float> radius_panicle;
        float adj = 20; // scale factor to match the length of meters
        float width_panicle = params.s5_panicle_size.y * 10 * adj;

        for (float n = 0; n < width_panicle ; n++) {
            float x = 0;
            float y = 0;
            float dz = n * (0.01);
            float z = dz;
            float angle = n * PI_F / width_panicle;
            float dr = 0.01 * sin(angle);

            nodes_panicle.push_back(make_vec3(x, y, z));
            radius_panicle.push_back(dr);

        }

        std::vector<uint> UUIDs2 = context->addTube(params.s5_panicle_subdivisions, nodes_panicle, radius_panicle,params.s5_seed_texture_file.c_str());

        float z_value = 0;

        float di = 0;

        for (int i = (params.s5_panicle_size.x*adj) + 2; i > -1; i--) {


            std::vector<uint> UUIDs_copy = context->copyPrimitive(UUIDs2);

            float rotation_angle;

            if (i > (((params.s5_panicle_size.x*adj) + 2) / float(3))) {
                rotation_angle = 0.26;
            } else {
                rotation_angle = i * 0.0867;
            }
            float dz = 0.032;

            z_value = z_value + dz;
            vec3 tra1(0, 0, z_value - dz);
            vec3 base = interpolateTube(nodes_panicle_stalk, 0.05);
            float rot1 = rotation_angle;

            context->rotatePrimitive(UUIDs_copy, rot1, "y");
            context->translatePrimitive(UUIDs_copy, base);
            context->translatePrimitive(UUIDs_copy, tra1);


            s5_UUID_panicle_plant.front().push_back(UUIDs_copy);

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
                s5_UUID_panicle_plant.front().push_back(context->copyPrimitive(UUIDs_copy));
                float rot2 = ii * i_value_2 * PI_F / float(180);
                context -> rotatePrimitive(s5_UUID_panicle_plant.front().back(), rot2, "z");

            }

        };


        context->deletePrimitive(UUIDs2);
        ;
        vec3 V_1 = interpolateTube(nodes_stem, 1);
        vec3 V_2 = interpolateTube(nodes_stem, 0.95);
        vec3 V_3  = V_1 - V_2;

        float Z = cart2sphere(V_3).zenith;

        context->rotatePrimitive(flatten(s5_UUID_panicle_plant), Z, "y");

        context->translatePrimitive(flatten(s5_UUID_panicle_plant), V_1);


// THE LEAVES

        int nodes_no = params.s5_number_of_leaves;


        float rotation_x1 = (context->randu(0, 360) * PI_F) / float(180);// inclination rotation, different for each plant
        std::vector<std::vector<uint> > s5_UUID_leaf_plant;

        for (int i = 1; i < (nodes_no + 1); i++) {

            std::vector<uint> UUIDs4;

            float Nx = params.s5_leaf_subdivisions.x;
            float Ny;

            if ( params.s5_leaf_subdivisions.y % 2 == 0){
                Ny = params.s5_leaf_subdivisions.y;
            } else {
                Ny = params.s5_leaf_subdivisions.y + 1;
            }

            float x;

            if (i >= (nodes_no - 1) || i == 1) {

                x = 0.75;
            } else {

                x = 1;
            }


            float dx = params.s5_leaf_size.x * x/ Nx;
            float dy = params.s5_leaf_size.y/Ny;

            float A_3 = params.s5_leaf_size.x / float(0.08); // Half waves on the leaf
            float A_2 = params.s5_leaf_size.x / float(90); //
            float leaf_amplitude = params.s5_leaf_size.x / float(5); //

            for (int i = 0; i < Nx; i++) {
                for (float j = 0; j < Ny; j++) {

                    float x = i * dx;
                    float y = j * dy;
                    float z = 0;
                    float sx = dx;
                    float sy = dy;

                    float x_i = x * PI_F / (Nx * dx);
                    float sx_i = (x + sx) * PI_F / (Nx * dx);

                    float z_1 = (x * PI_F) / ((Nx * dx) /A_3);
                    float z_2 = ((x + sx) * PI_F) /((Nx * dx) /A_3);

                    float leaf_wave_1;
                    float leaf_wave_2;
                    float leaf_wave_3;
                    float leaf_wave_4;

                    if (j == 0) {
                        leaf_wave_1 = A_2 * sin(z_1);
                        leaf_wave_2 = A_2 * sin(z_2);
                    } else {
                        leaf_wave_1 = 0;
                        leaf_wave_2 = 0;
                    }

                    z = leaf_amplitude * sin(x_i) + leaf_wave_1;
                    vec3 v0(x, y, z);

                    z = leaf_amplitude * sin(sx_i) + leaf_wave_2;
                    vec3 v1(x + sx, y, z);

                    if (j == Ny - 1) {
                        leaf_wave_3 = A_2 * sin(z_2);
                        leaf_wave_4 = A_2 * sin(z_1);
                    } else {
                        leaf_wave_3 = 0;
                        leaf_wave_4 = 0;

                    }

                    z = leaf_amplitude * sin(sx_i) + leaf_wave_3;
                    vec3 v2(x + sx, y + sy, z);

                    z = leaf_amplitude * sin(x_i) + leaf_wave_4;
                    vec3 v3(x, y + sy, z);

                    vec2 uv0(x / (Nx * dx), y / float(params.s5_leaf_size.y));
                    vec2 uv1((x + sx) / (Nx * dx), y / float(params.s5_leaf_size.y));
                    vec2 uv2((x + sx) / (Nx * dx), (y + sy) / float(params.s5_leaf_size.y));
                    vec2 uv3(x /(Nx * dx), (y + sy) / float(params.s5_leaf_size.y));

                    UUIDs4.push_back(context->addTriangle(v0, v1, v2, params.s5_leaf_texture_file.c_str(), uv0, uv1, uv2));
                    UUIDs4.push_back(context->addTriangle(v0, v2, v3, params.s5_leaf_texture_file.c_str(), uv0, uv2, uv3));
                }
            }

            s5_UUID_leaf_plant.push_back(context->copyPrimitive(UUIDs4));
            context->deletePrimitive(UUIDs4);

            float frac;

            if (i == 1) {

                frac = 0.9;
            } else {

                frac = (1 - ((i - (context->randu() / float(2))) / float((nodes_no + 1)*1.2))) - 0.15;
            }

            vec3 base = interpolateTube(nodes_stem, frac);
            float rotation_1 = - (params.s5_mean_leaf_angle * PI_F / float(180)) - (context->randu(0, 10) * PI_F) / float(180);
            float rotation_x2 = (context->randu(0, 45) * PI_F) / float(180);
            float rotation_2 = rotation_x1 + rotation_x2;
            float rotation_3 = rotation_2 + PI_F; //
            vec3 translation(0, -0.9 / (2 / float(params.s5_leaf_size.y)), 0);  // adjustment

            if (i % 2 != 0) {
                context->translatePrimitive(s5_UUID_leaf_plant.back(), translation);
                context->rotatePrimitive(s5_UUID_leaf_plant.back(), rotation_1, "y");
                context->rotatePrimitive(s5_UUID_leaf_plant.back(), rotation_2, "z");
                context->translatePrimitive(s5_UUID_leaf_plant.back(), base);

            } else {
                context->translatePrimitive(s5_UUID_leaf_plant.back(), translation);
                context->rotatePrimitive(s5_UUID_leaf_plant.back(), rotation_1, "y");
                context->rotatePrimitive(s5_UUID_leaf_plant.back(), rotation_3, "z");
                context->translatePrimitive(s5_UUID_leaf_plant.back(), base);
            }

        }

        float plant_rotation = (context->randu(0, 360) * PI_F) / float(180);

        context->rotatePrimitive( s5_UUID_stem_plant,plant_rotation,origin, make_vec3(0,0,1));
        context->rotatePrimitive( flatten(s5_UUID_panicle_plant),plant_rotation,origin, make_vec3(0,0,1));
        context->rotatePrimitive( flatten(s5_UUID_leaf_plant),plant_rotation,origin, make_vec3(0,0,1));

        std::vector<uint> s5_UUID_branch_plant;    // empty vector

        UUID_trunk.push_back(s5_UUID_stem_plant);
        UUID_fruit.push_back(s5_UUID_panicle_plant);
        UUID_leaf.push_back(s5_UUID_leaf_plant);
        UUID_branch.push_back(s5_UUID_branch_plant);

    }

    return UUID_leaf.size()-1;

};



