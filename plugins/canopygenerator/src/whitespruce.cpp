#include "CanopyGenerator.h"

using namespace helios;
using namespace std;

uint CanopyGenerator::whitespruce(const WhiteSpruceCanopyParameters &params, const vec3 &origin) {

    vector<uint> U;
    std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
    std::vector<std::vector<uint>> UUID_leaf_plant;

    std::uniform_real_distribution<float> unif_distribution;

    //------ trunk -------//

    std::vector<float> rad_main;
    std::vector<vec3> pos_main;

    float dz = params.trunk_height / 10.f;
    for (int i = 0; i < 10; i++) {
        rad_main.push_back(params.trunk_radius * (9 - i) / 9.f);
        pos_main.push_back(make_vec3(0., 0., i * dz));
    }

    for (uint i = 0; i < rad_main.size(); i++) {
        pos_main.at(i) = pos_main.at(i) + origin;
    }

    UUID_trunk_plant = context->addTube(params.wood_subdivisions, pos_main, rad_main, params.wood_texture_file.c_str());

    //---- first order branches -----//

    int Nlevels = floor((params.trunk_height - params.base_height) / params.level_spacing);

    for (int i = 0; i < Nlevels - 1; i++) {

        float vfrac = float(Nlevels - i - 1) / float(Nlevels - 1);

        float rcrown;
        if (vfrac > 0.3) {
            rcrown = params.crown_radius;
        } else {
            rcrown = fmax(1.f / 0.3 * vfrac * params.crown_radius, 0.2 * params.crown_radius);
        }
        // rcrown += getVariation(0.1*rcrown,generator);

        float z = fmin(params.trunk_height, params.base_height + i * params.level_spacing * (1 + getVariation(0.1f * params.level_spacing, generator)));

        int Nbranches = fmax(4, params.branches_per_level * vfrac);

        for (int j = 0; j < Nbranches; j++) {

            float phi = float(j) / float(Nbranches) * 2.f * PI_F * (1 + getVariation(0.1f, generator));

            float theta = -0.15 * PI_F;
            theta += getVariation(0.2f * fabs(theta), generator);

            std::vector<float> rad_branch;
            std::vector<vec3> pos_branch;

            z += getVariation(0.5f, generator);

            float r = rcrown + getVariation(0.1f * rcrown, generator);

            pos_branch.push_back(origin + make_vec3(0, 0, z));
            pos_branch.push_back(origin + make_vec3(r * sinf(phi) * cosf(theta), r * cosf(phi) * cosf(theta), z + r * sinf(theta)));

            // printf("pos_branch = (%f,%f,%f)\n",pos_branch.front().x,pos_branch.front().y,pos_branch.front().z);

            rad_branch.push_back(params.shoot_radius * vfrac);
            rad_branch.push_back(0.1 * params.shoot_radius * vfrac);

            // printf("rad_branch = %f\n",rad_branch.back());

            U = context->addTube(params.wood_subdivisions, pos_branch, rad_branch, params.wood_texture_file.c_str());
            UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

            for (int k = 0; k < 2 * Nbranches; k++) {

                float bfrac = float(k + 1) / float(2 * Nbranches);
                // bfrac = 4.f/0.8*bfrac/(4*bfrac+1.f);

                int side = k % 2;

                std::vector<float> rad_subbranch;
                std::vector<vec3> pos_subbranch;

                vec3 base = interpolateTube(pos_branch, bfrac);
                float rbase = interpolateTube(rad_branch, bfrac);

                pos_subbranch.push_back(base);

                float l = fmax(0.1, 0.6 * rcrown);
                l += getVariation(0.25f * l, generator);

                float bangle = 0.2 * PI_F;
                // vec3 bdir = l*make_vec3(sinf(phi)*cosf(theta+bangle),cosf(phi)*cosf(theta+0.5*PI_F),sinf(theta+bangle));
                // bdir = rotatePointAboutLine( bdir, base, interpolateTube( pos_branch, bfrac-0.01 ), getVariation(PI_F,generator) );

                vec3 bdir = l * make_vec3(sinf(phi + (0.4 + getVariation(0.1f, generator)) * PI_F * (-1 + 2 * side)), cosf(phi - 0.4 * PI_F + 0.8 * PI_F * side), 0);

                pos_subbranch.push_back(base + bdir);

                rad_subbranch.push_back(rbase);
                rad_subbranch.push_back(0.1 * rbase);

                // UUID_branch = context->addTube(params.wood_subdivisions,pos_subbranch,rad_subbranch, params.wood_texture_file.c_str() );
                // context->addTube(params.wood_subdivisions,pos_subbranch,rad_subbranch, params.wood_texture_file.c_str() );

                // needles
                int Nneedles = round(50 * l / (0.4 * params.crown_radius));
                for (int n = 0; n < Nneedles; n++) {

                    float nfrac = float(n + 1) / float(Nneedles);
                    // nfrac = 4.f/0.8*nfrac/(4*nfrac+1.f);

                    vec3 nbase = interpolateTube(pos_subbranch, nfrac);
                    vec3 nbase_p = interpolateTube(pos_subbranch, 0.99 * nfrac);

                    vec3 norm = nbase - nbase_p;
                    // n.normalize();

                    SphericalCoord rotation = cart2sphere(norm);

                    UUID_leaf_plant.push_back(context->addTile(make_vec3(0, 0, 0), make_vec2(0.1 * params.needle_length, params.needle_length), make_SphericalCoord(0, 0), params.needle_subdivisions, params.needle_color));

                    float downangle = 0.2 * PI_F;
                    context->rotatePrimitive(UUID_leaf_plant.back(), downangle - rotation.elevation, "x");
                    context->translatePrimitive(UUID_leaf_plant.back(), make_vec3(0, -0.55 * params.needle_length, 0));
                    context->rotatePrimitive(UUID_leaf_plant.back(), 0.5 * PI_F - rotation.azimuth, "z");
                    context->rotatePrimitive(UUID_leaf_plant.back(), getVariation(PI_F, generator), norm);
                    context->translatePrimitive(UUID_leaf_plant.back(), nbase);
                }
            }
        }
    }

    UUID_trunk.push_back(UUID_trunk_plant);
    UUID_branch.push_back(UUID_branch_plant);
    UUID_leaf.push_back(UUID_leaf_plant);
    std::vector<std::vector<std::vector<uint>>> UUID_fruit_plant;
    UUID_fruit.push_back(UUID_fruit_plant);

    return UUID_leaf.size() - 1;
}
