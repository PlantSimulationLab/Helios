#include "CanopyGenerator.h"
#include "math.h"

using namespace std;
using namespace helios;

// SORGHUM PLANT
void CanopyGenerator::sorghum( const SorghumCanopyParameters params, const vec3 origin ){

// THE STEM
    std::vector<vec3> nodes;
    std::vector<float> radius;
    vector<RGBcolor> Color;

    //Map the route for the stem
    for (float n = 0; n < params.stem_length; n++) {
        float mid_height = params.stem_length* 0.5;
        float i = n * (180/float(params.stem_length)) * M_PI/float(180);
        float i_mid = (mid_height)* (180/float(params.stem_length)) * M_PI/float(180);
        float x_mid = -params.stem_curvature*sin(i_mid);
        float x1,x;
        if (n <= mid_height){
            x = 0;
        }
        else {
            x1 = -params.stem_curvature*sin(i);
            x = x1 - x_mid;
        }
        float y = 0;

        float z = n;

        nodes.push_back(make_vec3(x, y, z) + origin);

        //Thin out the radius for the last 5% height of the stem
        if (n <= params.stem_length*0.95){
            radius.push_back(params.stem_radius);
        }else{
            radius.push_back(params.stem_radius*0.8);
        }

        Color.push_back(make_vec3(0,0.6,0)); // color of the stem (could be changed)

    }

    std::vector<uint> UUIDs1 = context->addTube(50, nodes, radius, Color);

// THE FRUIT
    std::vector<std::vector<uint> > UUIDs_all;
    vector<RGBcolor> Color2;
    std::vector<vec3> nods;
    std::vector<float> radis;

    //short stem for the fruit
    float mm = 1;
    for( float n=0; n<mm; n = n+0.5 ){
        float y = 0 ;
        float x = 0;
        float z =  n;

        nods.push_back(make_vec3(x,y,z) );
        radis.push_back(params.stem_radius*0.8);

        Color2.push_back(make_vec3(0,0.6,0));

    }

    std::vector<uint> UUID2 = context->addTube(50, nods, radis, Color2);
    UUIDs_all.push_back(context->copyPrimitive(UUID2));
    context->deletePrimitive(UUID2);

    std::vector<vec3> nodes1;
    std::vector<float> radius1;

    float fruit_width_1 = params.fruit_width * 10;

    for( float n=0; n< (fruit_width_1 + 1); n++ ){
        float x = 0 ;
        float y = 0;
        float dz = n * (0.15);
        float z = dz;
        float angle  = n* M_PI/float(fruit_width_1);
        float dr = 0.08 * sin(angle);

        nodes1.push_back(make_vec3(x,y,z) );
        radius1.push_back(dr);

    }

    std::vector<uint> UUIDs13 = context->addTube(50, nodes1, radius1, params.seed_texture_file.c_str());

    for (int i = params.fruit_height; i > -1; i --){
        std::vector<uint> UUIDs2 = context->copyPrimitive(UUIDs13);

        float rot_angle;

        if (i > 3){
            rot_angle = 0.26;
        }
        else {
            rot_angle = i*0.0867;
        }

        float z_value = z_value + 0.55;
        vec3 tra1(0,0,z_value-0.55);
        vec3 base = interpolateTube(nods,0.5);
        float rot1 = rot_angle;

        context->rotatePrimitive(UUIDs2,rot1,"y");
        context->translatePrimitive(UUIDs2,base);
        context->translatePrimitive(UUIDs2,tra1);
        UUIDs_all.push_back(UUIDs2);

        for (int ii = 0; ii < 8; ii ++){
            UUIDs_all.push_back(context->copyPrimitive(UUIDs2));
            float rot2 = ii * 45 * M_PI/float(180);
            context->rotatePrimitive(UUIDs_all.back(),rot2,"z");
        }

    };

    vec3 V_1 = interpolateTube(nodes,1);
    vec3 V_2 = nodes.at(params.stem_length - 1) - nodes.at(params.stem_length - 2);
    float Z = cart2sphere(V_2).zenith;

    context->rotatePrimitive(flatten(UUIDs_all) ,Z, "y");
    context->translatePrimitive(flatten(UUIDs_all), V_1 );

// THE LEAVES

    // Base leaf
    float x_size = params.leaf_length * 4;
    float width_split = params.leaf_width * 0.5;
    float y_size = params.leaf_width;

    // leaf pair
    int leaf_pair;
    if (params.number_of_leaves % 2 == 0){
        leaf_pair = params.number_of_leaves * 0.5;
    }else {
        leaf_pair = ((params.number_of_leaves - 1)* 0.5) + 1;
    }

    std::vector<uint> UUIDs;

    float rotation_x1 = (context->randu(0,360) * M_PI)/ float(180);// inclination rotation, different for each plant
    for (int i = 1; i < (leaf_pair + 1); i++) {

        float A_3 = 10; // Half waves on the leaf

        // leaf wave
        float A_2 = 0.2 + (context->randu(0,10)*0.01) ; // amplitude

        for (int i = 0; i < x_size; i++) {
            for (float j = 0; j < y_size; j = j + width_split) {

                float x = i * 0.25;
                float y = j;
                float z = 0;
                float sx = 0.25;
                float sy = width_split;

                float x_i = x * M_PI / float(params.leaf_length);
                float sx_i = (x + sx) * M_PI / float(params.leaf_length);

                float z_1 = (x * M_PI)/float(params.leaf_length/float(A_3));
                float z_2 = ((x + sx) * M_PI)/float(params.leaf_length/float(A_3));


                float leaf_wave_1;
                float leaf_wave_2;
                float leaf_wave_3;
                float leaf_wave_4;

                if (j == 0){
                    leaf_wave_1 = A_2*sin(z_1);
                    leaf_wave_2 = A_2*sin(z_2);
                }
                else {
                    leaf_wave_1 = 0;
                    leaf_wave_2 = 0;
                }

                z = params.leaf_amplitude * sin(x_i) + leaf_wave_1;
                vec3 v0(x, y, z);

                z = params.leaf_amplitude * sin(sx_i) + leaf_wave_2;
                vec3 v1(x + sx, y, z);

                if (j == y_size - width_split){
                    leaf_wave_3 = A_2*sin(z_2);
                    leaf_wave_4 = A_2*sin(z_1);
                }
                else {
                    leaf_wave_3 = 0;
                    leaf_wave_4 = 0;

                }

                z = params.leaf_amplitude * sin(sx_i) + leaf_wave_3;
                vec3 v2(x + sx, y + sy, z);

                z = params.leaf_amplitude * sin(x_i) + leaf_wave_4;
                vec3 v3(x, y + sy, z);

                vec2 uv0(x / float(params.leaf_length), y / float(params.leaf_width));
                vec2 uv1((x + sx) / float(params.leaf_length), y / float(params.leaf_width));
                vec2 uv2((x + sx) / float(params.leaf_length), (y + sy) / float(params.leaf_width));
                vec2 uv3(x / float(params.leaf_length), (y + sy) / float(params.leaf_width));

                UUIDs.push_back(context->addTriangle(v0, v1, v2, params.leaf_texture_file.c_str(), uv0, uv1, uv2));
                UUIDs.push_back(context->addTriangle(v0, v2, v3, params.leaf_texture_file.c_str(), uv0, uv2, uv3));
            }
        }

        // replicate and form more leaves
        std::vector<uint> UUIDs2 = context->copyPrimitive(UUIDs);
        std::vector<uint> UUIDs3 = context->copyPrimitive(UUIDs);

        float frac_1 = (i - (context->randu()/float(2))) / float(leaf_pair + 1);
        float frac_2 = (i - context->randu()) / float(leaf_pair + 1);
        vec3 base_1 = interpolateTube(nodes, frac_1);
        vec3 base_2 = interpolateTube(nodes, frac_2);
        float rotation_1 = -(i * 4 * M_PI) / float(180); // inclination angle
        float rotation_x2 = (i * 2 * M_PI) / float(180); // small rotations
        float rotation_3 = rotation_x1 + rotation_x2;
        float rotation_4 = rotation_3 + M_PI; //
        vec3 translation_1(0.08, -0.9/(2/float(params.leaf_width)), 0);  // adjustment

        if (i < leaf_pair){ // cater for even and odd number of leaves
            // leaf 1
            context->translatePrimitive(UUIDs2, translation_1);
            context->rotatePrimitive(UUIDs2, rotation_1, "y");
            context->rotatePrimitive(UUIDs2, rotation_3, "z");
            context->translatePrimitive(UUIDs2, base_1);

            // leaf 2
            context->translatePrimitive(UUIDs3, translation_1);
            context->rotatePrimitive(UUIDs3, rotation_1, "y");
            context->rotatePrimitive(UUIDs3, rotation_4, "z");
            context->translatePrimitive(UUIDs3, base_2);
        }else {
            if (params.number_of_leaves % 2 == 0){
                // leaf 1
                context->translatePrimitive(UUIDs2, translation_1);
                context->rotatePrimitive(UUIDs2, rotation_1, "y");
                context->rotatePrimitive(UUIDs2, rotation_3, "z");
                context->translatePrimitive(UUIDs2, base_1);

                // leaf 2
                context->translatePrimitive(UUIDs3, translation_1);
                context->rotatePrimitive(UUIDs3, rotation_1, "y");
                context->rotatePrimitive(UUIDs3, rotation_4, "z");
                context->translatePrimitive(UUIDs3, base_2);
            }else{

                context->translatePrimitive(UUIDs2, translation_1);
                context->rotatePrimitive(UUIDs2, rotation_1, "y");
                context->rotatePrimitive(UUIDs2, rotation_3, "z");
                context->translatePrimitive(UUIDs2, base_1);

                context->deletePrimitive(UUIDs3);
            }

        }

    }

    context->deletePrimitive(UUIDs);

}

