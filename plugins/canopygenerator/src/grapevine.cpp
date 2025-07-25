/** \file "grapevine.cpp" Definitions of functions for building grapevine plant geometries.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "CanopyGenerator.h"

using namespace helios;
using namespace std;

std::vector<std::vector<uint>> CanopyGenerator::addGrapeCluster(vec3 position, float grape_rad, float cluster_rad, helios::RGBcolor grape_color, uint grape_subdiv) {

    std::uniform_real_distribution<float> unif_distribution;

    float z = position.z;

    float theta = 0;
    float RB = cluster_rad;
    int N = 100;
    int count = 0;
    int max_grape_levels = 13.f - 4.f * unif_distribution(generator);

    vec3 xgrape;

    std::vector<std::vector<uint>> UUID;

    while ((N > 3 || count < 3) && count < max_grape_levels) {

        if (count < 3) {
            RB = cluster_rad * powf(0.5f, float(2 - count));
        } else if (count == 3) {
            RB = cluster_rad;
        }

        N = floor(PI_F * RB / grape_rad) + 1;

        float dtheta = 2.f * PI_F / float(N);
        for (int j = 0; j < N; j++) {

            xgrape.x = position.x + RB * sin(theta) + unif_distribution(generator) * 0.25f * grape_rad;
            xgrape.y = position.y + RB * cos(theta) + unif_distribution(generator) * 0.25f * grape_rad;
            xgrape.z = z + unif_distribution(generator) * 0.25f * grape_rad;

            uint objID = context->addSphereObject(fmax(2.f, float(grape_subdiv)), xgrape, grape_rad, grape_color);
            std::vector<uint> U = context->getObjectPrimitiveUUIDs(objID);
            UUID.push_back(U);

            theta = theta + dtheta;
        }
        z = z - 1.8f * grape_rad;
        theta = theta + 0.5f * dtheta;
        RB = RB * 0.85f + unif_distribution(generator) * RB * 0.15f;
        count = count + 1;
    }

    if (enable_element_labels) {
        context->setPrimitiveData(flatten(UUID), "element_label", "berry");
    }

    return UUID;
}

std::vector<uint> leafPrototype(const int2 leaf_subdivisions, const char *leaf_texture_file, Context *context) {

    int Nx = leaf_subdivisions.x;
    int Ny = ceil(leaf_subdivisions.y * 0.5);

    float dx = 1.f / float(Nx);
    float dy = 0.5f / float(Ny);

    std::vector<uint> UUIDs;

    float a0 = 0.15 * (1 + context->randu());
    float e0 = 0.1f * (1 + context->randu());

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {

            float x = i * dx;
            float y = j * dy;

            float mag;
            float z;

            mag = sqrt(x * x + 2 * y * y);
            // z = a0*mag/(e0+mag);
            z = a0 * y / (e0 + y) + a0 * x / (e0 + x);
            vec3 v0(x, y, z);

            mag = sqrt((x + dx) * (x + dx) + 2 * y * y);
            // z = a0*mag/(e0+mag);
            z = a0 * y / (e0 + y) + a0 * (x + dx) / (e0 + x + dx);
            vec3 v1(x + dx, y, z);

            mag = sqrt((x + dx) * (x + dx) + 2 * (y + dy) * (y + dy));
            // z = a0*mag/(e0+mag);
            z = a0 * (y + dy) / (e0 + y + dy) + a0 * (x + dx) / (e0 + x + dx);
            vec3 v2(x + dx, y + dy, z);

            mag = sqrt(x * x + 2 * (y + dy) * (y + dy));
            // z = a0*mag/(e0+mag);
            z = a0 * (y + dy) / (e0 + y + dy) + a0 * x / (e0 + x);
            vec3 v3(x, y + dy, z);

            float dj = float(j) * dy;
            float dj_plus = float(j + 1) * dy;

            vec2 uv0(1.f - x, 0.5f + dj);
            vec2 uv1(1.f - x - dx, 0.5f + dj);
            vec2 uv2(1.f - x - dx, 0.5f + dj_plus);
            vec2 uv3(1.f - x, 0.5f + dj_plus);

            UUIDs.push_back(context->addTriangle(v0, v1, v2, leaf_texture_file, uv0, uv1, uv2));
            UUIDs.push_back(context->addTriangle(v0, v2, v3, leaf_texture_file, uv0, uv2, uv3));

            v0.y = -v0.y;
            v1.y = -v1.y;
            v2.y = -v2.y;
            v3.y = -v3.y;

            uv0 = make_vec2(1.f - x, 0.5f - dj);
            uv1 = make_vec2(1.f - x - dx, 0.5f - dj);
            uv2 = make_vec2(1.f - x - dx, 0.5f - dj_plus);
            uv3 = make_vec2(1.f - x, 0.5f - dj_plus);

            UUIDs.push_back(context->addTriangle(v0, v2, v1, leaf_texture_file, uv0, uv2, uv1));
            UUIDs.push_back(context->addTriangle(v0, v3, v2, leaf_texture_file, uv0, uv3, uv2));
        }
    }

    context->translatePrimitive(UUIDs, make_vec3(-1, 0, 0));

    return UUIDs;
}

uint CanopyGenerator::grapevineVSP(const VSPGrapevineParameters &params, const vec3 &origin) {

    vector<uint> U;
    std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
    std::vector<std::vector<uint>> UUID_leaf_plant;
    std::vector<std::vector<std::vector<uint>>> UUID_fruit_plant;

    std::uniform_real_distribution<float> unif_distribution;

    float canopy_rotation = params.canopy_rotation + getVariation(params.canopy_rotation_spread, generator);

    bool is_dead = false;
    if (params.dead_probability > 0) {
        float random_draw = context->randu();
        is_dead = random_draw <= params.dead_probability;
    }

    //------ trunks -------//

    float trunk_radius = std::max(1e-5f, params.trunk_radius + getVariation(params.trunk_radius_spread, generator));
    float trunk_height = std::max(0.f, params.trunk_height + getVariation(params.trunk_height_spread, generator));

    std::vector<float> rad_main;
    rad_main.push_back(0.75f * trunk_radius);
    rad_main.push_back(0.8f * trunk_radius);
    rad_main.push_back(1.f * trunk_radius);
    rad_main.push_back(0.7f * trunk_radius);
    rad_main.push_back(0.95f * trunk_radius);
    rad_main.push_back(0.1f * trunk_radius);
    std::vector<vec3> pos_main;
    pos_main.push_back(make_vec3(0., 0., 0.0));
    pos_main.push_back(make_vec3(0, 0, 0.2f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.22f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.6f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.96f * trunk_height));
    pos_main.push_back(make_vec3(0., 0., trunk_height));

    for (uint i = 0; i < rad_main.size(); i++) {
        pos_main.at(i) = pos_main.at(i) + origin;
    }

    int wood_subdivisions = std::max(3, params.wood_subdivisions + getVariation(params.wood_subdivisions_spread, generator));

    uint objID = context->addTubeObject(wood_subdivisions, pos_main, rad_main, params.wood_texture_file.c_str());
    UUID_trunk_plant = context->getObjectPrimitiveUUIDs(objID);
    if (enable_element_labels) {
        context->setPrimitiveData(UUID_trunk_plant, "element_label", "trunk");
    }

    //---- Cordons -----//

    float cordon_height = std::max(0.f, params.cordon_height + getVariation(params.cordon_height_spread, generator));
    float cordon_radius = std::max(1e-5f, params.cordon_radius + getVariation(params.cordon_radius_spread, generator));
    float cordon_length = std::max(0.f, params.cordon_length + getVariation(params.cordon_length_spread, generator));

    float diff = cordon_height - trunk_height;

    float cost = cosf(canopy_rotation);
    float sint = sinf(canopy_rotation);

    int Ncord = 2 * wood_subdivisions;

    // West Cordon

    std::vector<float> rad_cordw;
    rad_cordw.push_back(cordon_radius);
    std::vector<vec3> pos_cordw;
    pos_cordw.push_back(make_vec3(0.01f * cordon_length * cost, 0.01f * cordon_length * sint, 0.95 * trunk_height));
    vec3 n_start = sphere2cart(make_SphericalCoord(cordon_height, 0.4f * PI_F * (1 + getVariation(0.2f, generator)), getVariation(0.2f * PI_F, generator)));
    vec3 n_end = make_vec3(0.5f * cordon_height, 0, 0);

    for (int i = 1; i < Ncord; i++) {
        float frac = float(i) / float(Ncord - 1);
        vec3 n = spline_interp3(frac, pos_cordw.front(), n_start, make_vec3(cordon_length * cost, cordon_length * sint, trunk_height + diff), n_end);
        pos_cordw.push_back(n);
        rad_cordw.push_back(cordon_radius * (1.f - 0.6f * frac));
    }

    std::vector<vec3> tmp;
    tmp.resize(pos_cordw.size());
    for (uint i = 0; i < pos_cordw.size(); i++) {
        tmp.at(i) = pos_cordw.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, tmp, rad_cordw, params.wood_texture_file.c_str());
    UUID_branch_plant = context->getObjectPrimitiveUUIDs(objID);

    // East Cordon
    std::vector<float> rad_corde;
    rad_corde.push_back(cordon_radius);
    std::vector<vec3> pos_corde;
    pos_corde.push_back(make_vec3(-0.01f * cordon_length * cost, -0.01f * cordon_length * sint, 0.95f * trunk_height));
    n_start = sphere2cart(make_SphericalCoord(cordon_height, 0.4f * PI_F * (1 + getVariation(0.2f, generator)), PI_F + getVariation(0.2f * PI_F, generator)));
    n_end = make_vec3(-0.5f * cordon_height, 0, 0);

    for (int i = 1; i < Ncord; i++) {
        float frac = float(i) / float(Ncord - 1);
        vec3 n = spline_interp3(frac, pos_corde.front(), n_start, make_vec3(-cordon_length * cost, -cordon_length * sint, trunk_height + diff), n_end);
        pos_corde.push_back(n);
        rad_corde.push_back(cordon_radius * (1.f - 0.6f * frac));
    }

    tmp.resize(pos_corde.size());
    for (uint i = 0; i < pos_corde.size(); i++) {
        tmp.at(i) = pos_corde.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, tmp, rad_corde, params.wood_texture_file.c_str());
    U = context->getObjectPrimitiveUUIDs(objID);
    UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

    if (enable_element_labels) { // note: only need to set label once because the UUID vector contains both cordons
        context->setPrimitiveData(UUID_branch_plant, "element_label", "cordon");
    }

    //------- primary shoots ---------//

    // uint ID0 = context->addTileObject( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() );

    std::vector<uint> leaf_ptype = leafPrototype(params.leaf_subdivisions, params.leaf_texture_file.c_str(), context);

    float shoot_length = std::max(0.f, params.shoot_length + getVariation(params.shoot_length_spread, generator));
    float shoot_radius = std::max(1e-5f, params.shoot_radius + getVariation(params.shoot_radius_spread, generator));
    uint shoots_per_cordon = std::max(uint(0), params.shoots_per_cordon + getVariation(params.shoots_per_cordon_spread, generator));

    float height = cordon_height + shoot_length;

    // Looping over each cordon
    for (uint c = 0; c < 2; c++) {

        std::vector<float> rad_cord;
        std::vector<vec3> pos_cord;
        float sign;
        if (c == 0) {
            pos_cord = pos_cordw;
            rad_cord = rad_cordw;
            sign = 1;
        } else {
            pos_cord = pos_corde;
            rad_cord = rad_corde;
            sign = -1;
        }

        float dx = fabs(pos_cord.back().y - pos_cord.at(0).y) / (float(shoots_per_cordon));

        // looping over each shoot in the cordon
        for (int j = 1; j < shoots_per_cordon + 1; j++) {

            // fraction of distance along cordon
            float frac_shoot = float(j) / float(shoots_per_cordon);

            vec3 cane_base = interpolateTube(pos_cord, frac_shoot);

            std::vector<float> rad_pshoot;
            std::vector<vec3> pos_pshoot;

            // shoot base
            rad_pshoot.push_back(shoot_radius);
            pos_pshoot.push_back(cane_base);

            // shoot nodes
            float phirot = (0.5f - unif_distribution(generator)) * 0.5f * PI_F;
            phirot = 0.5 * PI_F;
            if (unif_distribution(generator) < 0.5) {
                phirot += PI_F;
            }

            // tangent vector for start of shoot
            vec3 n_start = sphere2cart(make_SphericalCoord(0.3f * height, 0.5f * PI_F * (1 - unif_distribution(generator)), 2.f * PI_F * unif_distribution(generator)));
            // tangent vector for middle of shoot
            vec3 n_mid = sphere2cart(make_SphericalCoord(0.3f * height, 0.5f * PI_F * (1 - unif_distribution(generator) * 0.15), 2.f * PI_F * unif_distribution(generator)));
            // tangent vector for end of shoot
            vec3 n_end = sphere2cart(make_SphericalCoord(0.3f * height, 0.5f * PI_F * (1 - unif_distribution(generator) * 0.5), 2.f * PI_F * unif_distribution(generator)));

            uint Nz = 2 * wood_subdivisions;
            float dz = ((1 + getVariation(0.1f, generator)) * height - cordon_height) / float(Nz);

            float total_cordons_length = 2 * cordon_length;

            // position of middle of shoot
            vec3 cane_mid = cane_base + make_vec3(0.1f * getVariation(total_cordons_length, generator), 0.1f * getVariation(total_cordons_length, generator), 0.75f * Nz * dz);
            // position of end of shoot
            vec3 cane_end = cane_base + make_vec3(0.15f * getVariation(total_cordons_length, generator), 0.15f * getVariation(total_cordons_length, generator), Nz * dz);

            float zfrac_mid = (cane_mid.z - cane_base.z) / (cane_end.z - cane_base.z);

            // build nodes for shoot tube
            for (uint k = 1; k < Nz; k++) {

                float zfrac = float(k) / float(Nz - 1);

                vec3 n;
                if (zfrac < zfrac_mid) {
                    n = spline_interp3(zfrac / zfrac_mid, cane_base, n_start, cane_mid, n_mid);
                } else {
                    n = spline_interp3((zfrac - zfrac_mid) / (1 - zfrac_mid), cane_mid, n_mid, cane_end, n_end);
                }

                pos_pshoot.push_back(n);

                rad_pshoot.push_back(shoot_radius * (1.f - 0.5f * float(k) / float(Nz)));
            }

            rad_pshoot.back() = 0.0f;

            std::vector<vec3> tmp;
            tmp.resize(pos_pshoot.size());
            for (uint i = 0; i < pos_pshoot.size(); i++) {
                tmp.at(i) = pos_pshoot.at(i) + origin;
            }

            objID = context->addTubeObject(wood_subdivisions, tmp, rad_pshoot, params.wood_texture_file.c_str());
            U = context->getObjectPrimitiveUUIDs(objID);
            UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());
            if (enable_element_labels) {
                context->setPrimitiveData(U, "element_label", "cane");
            }

            if (is_dead) {
                // Don't add grapes and leaves
                continue;
            }

            // grape clusters
            float grape_radius = std::max(0.f, params.grape_radius + getVariation(params.grape_radius_spread, generator));
            float cluster_radius = std::max(0.f, params.cluster_radius + getVariation(params.cluster_radius_spread, generator));
            float cluster_height_max = std::max(0.f, params.cluster_height_max + getVariation(params.cluster_height_max_spread, generator));
            uint grape_subdivisions = std::max(3u, params.grape_subdivisions + getVariation(params.grape_subdivisions_spread, generator));
            std::vector<std::vector<uint>> UUID_grapes;
            if (grape_radius > 0 && cluster_radius > 0) {

                float fgrape = 0.035f + (cluster_height_max - 0.035) * unif_distribution(generator);
                vec3 p_grape = interpolateTube(tmp, fgrape);
                int sgn = 1;
                if (unif_distribution(generator) < 0.5) {
                    sgn = -1;
                }
                vec3 offset(sgn * (0.25f * cluster_radius + getVariation(0.1f, generator)) * sint, sgn * (0.1f * cluster_radius + getVariation(0.1f, generator)) * cost, 0.f);
                UUID_grapes = addGrapeCluster(p_grape + offset, grape_radius, cluster_radius, params.grape_color, grape_subdivisions);
            }
            UUID_fruit_plant.push_back(UUID_grapes);

            // leaves
            if (params.leaf_width == 0) {
                continue;
            }
            float leaf_width = std::max(0.f, params.leaf_width + getVariation(params.leaf_width_spread, generator));

            float flip = 0;
            if (unif_distribution(generator) < 0.5) {
                flip = 1;
            }
            float leaf_spacing_fraction = std::max(0.f, params.leaf_spacing_fraction + getVariation(params.leaf_spacing_fraction, generator));
            float lfrac = 1.f;
            int iter = 0;
            while (lfrac > 0.5 * leaf_width && iter < 100) {
                iter++;

                float lsize = fmaxf(leaf_width * (1.f - exp(-5.f * (1 - lfrac))), 0.1f * leaf_width);

                vec3 pos_leaf = interpolateTube(pos_pshoot, lfrac);

                vec3 parent_normal = interpolateTube(pos_pshoot, fmax(0, lfrac - 0.001)) - pos_leaf;
                parent_normal.normalize();
                vec3 leaf_offset = rotatePointAboutLine(make_vec3(getVariation(0.1f * lsize, generator), getVariation(0.75f * lsize, generator), 0), make_vec3(0, 0, 0), parent_normal, flip * PI_F + getVariation(0.25f * PI_F, generator));

                float s;
                if (int(flip) % 2 == 0) {
                    s = 1;
                } else {
                    s = -1;
                }

                float Rphi = -canopy_rotation - s * 0.5 * PI_F * (1.f + getVariation(0.4f, generator));
                float Rtheta = 0.25f * PI_F * (1.f + getVariation(0.2f, generator));

                vec3 position = origin + pos_leaf + leaf_offset;

                std::vector<uint> UUID_leaf = context->copyPrimitive(leaf_ptype);
                //                context->addPolymeshObject( UUID_leaf );
                context->scalePrimitive(UUID_leaf, make_vec3(lsize, lsize, lsize));
                context->rotatePrimitive(UUID_leaf, -Rtheta, "y");
                context->rotatePrimitive(UUID_leaf, Rphi, "z");
                context->translatePrimitive(UUID_leaf, position);

                UUID_leaf_plant.push_back(UUID_leaf);

                lfrac = lfrac - leaf_spacing_fraction * lsize * (1.f + getVariation(0.25f, generator));

                flip++;

                if (enable_element_labels) {
                    context->setPrimitiveData(UUID_leaf, "element_label", "leaf");
                }
            }
        }
    }

    // context->deleteObject(ID0);
    context->deletePrimitive(leaf_ptype);

    UUID_trunk.push_back(UUID_trunk_plant);
    UUID_branch.push_back(UUID_branch_plant);
    UUID_leaf.push_back(UUID_leaf_plant);
    UUID_fruit.push_back(UUID_fruit_plant);

    return UUID_leaf.size() - 1;
}

uint CanopyGenerator::grapevineSplit(const SplitGrapevineParameters &params, const vec3 &origin) {

    vector<uint> U;
    std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
    std::vector<std::vector<uint>> UUID_leaf_plant;
    std::vector<std::vector<std::vector<uint>>> UUID_fruit_plant;

    std::uniform_real_distribution<float> unif_distribution;

    float canopy_rotation = params.canopy_rotation + getVariation(params.canopy_rotation_spread, generator);

    bool is_dead = false;
    if (params.dead_probability > 0) {
        float random_draw = context->randu();
        is_dead = random_draw <= params.dead_probability;
    }

    //------ trunks -------//

    float trunk_radius = std::max(1e-5f, params.trunk_radius + getVariation(params.trunk_radius_spread, generator));
    float trunk_height = std::max(0.f, params.trunk_height + getVariation(params.trunk_height_spread, generator));

    std::vector<float> rad_main;
    rad_main.push_back(0.75f * trunk_radius);
    rad_main.push_back(0.8f * trunk_radius);
    rad_main.push_back(1.f * trunk_radius);
    rad_main.push_back(0.7f * trunk_radius);
    rad_main.push_back(0.95f * trunk_radius);
    rad_main.push_back(0.1f * trunk_radius);
    std::vector<vec3> pos_main;
    pos_main.push_back(make_vec3(0., 0., 0.0));
    pos_main.push_back(make_vec3(0, 0, 0.2f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.22f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.6f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.96f * trunk_height));
    pos_main.push_back(make_vec3(0., 0., trunk_height));

    for (uint i = 0; i < rad_main.size(); i++) {
        pos_main.at(i) = pos_main.at(i) + origin;
    }

    int wood_subdivisions = std::max(3, params.wood_subdivisions + getVariation(params.wood_subdivisions_spread, generator));

    uint objID = context->addTubeObject(wood_subdivisions, pos_main, rad_main, params.wood_texture_file.c_str());
    UUID_trunk_plant = context->getObjectPrimitiveUUIDs(objID);

    //------ crown -------//

    float cordon_height = std::max(0.f, params.cordon_height + getVariation(params.cordon_height_spread, generator));
    float cordon_radius = std::max(1e-5f, params.cordon_radius + getVariation(params.cordon_radius_spread, generator));
    float cordon_spacing = std::max(0.f, params.cordon_spacing + getVariation(params.cordon_spacing_spread, generator));

    float diff = cordon_height - trunk_height;

    float cost = cosf(canopy_rotation + 0.5f * PI_F);
    float sint = sinf(canopy_rotation + 0.5f * PI_F);

    std::vector<float> rad_crown;
    rad_crown.push_back(0.6f * trunk_radius);
    rad_crown.push_back(0.55f * trunk_radius);
    rad_crown.push_back(0.5f * trunk_radius);
    rad_crown.push_back(0.45f * trunk_radius);
    rad_crown.push_back(0.4f * trunk_radius);

    std::vector<vec3> pos_crownw;
    pos_crownw.push_back(make_vec3(0., 0., 0.95f * trunk_height));
    pos_crownw.push_back(make_vec3(0.05f * 0.5f * cordon_spacing * cost, 0.05f * 0.5f * cordon_spacing * sint, trunk_height));
    pos_crownw.push_back(make_vec3(0.25f * 0.5f * cordon_spacing * cost, 0.25f * 0.5f * cordon_spacing * sint, trunk_height + 0.1f * diff));
    pos_crownw.push_back(make_vec3(0.45f * 0.5f * cordon_spacing * cost, 0.45f * 0.5f * cordon_spacing * sint, trunk_height + 0.65f * diff));
    pos_crownw.push_back(make_vec3(0.75f * 0.5f * cordon_spacing * cost, 0.75f * 0.5f * cordon_spacing * sint, cordon_height));

    for (uint i = 0; i < rad_crown.size(); i++) {
        pos_crownw.at(i) = pos_crownw.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, pos_crownw, rad_crown, params.wood_texture_file.c_str());
    U = context->getObjectPrimitiveUUIDs(objID);
    UUID_trunk_plant.insert(UUID_trunk_plant.end(), U.begin(), U.end());

    std::vector<vec3> pos_crowne;
    pos_crowne.push_back(make_vec3(0., 0., 0.95f * trunk_height));
    pos_crowne.push_back(make_vec3(-0.05f * 0.5f * cordon_spacing * cost, -0.05f * 0.5f * cordon_spacing * sint, trunk_height));
    pos_crowne.push_back(make_vec3(-0.25f * 0.5f * cordon_spacing * cost, -0.25f * 0.5f * cordon_spacing * sint, trunk_height + 0.1f * diff));
    pos_crowne.push_back(make_vec3(-0.45f * 0.5f * cordon_spacing * cost, -0.45f * 0.5f * cordon_spacing * sint, trunk_height + 0.65f * diff));
    pos_crowne.push_back(make_vec3(-0.75f * 0.5f * cordon_spacing * cost, -0.75f * 0.5f * cordon_spacing * sint, cordon_height));

    for (uint i = 0; i < rad_crown.size(); i++) {
        pos_crowne.at(i) = pos_crowne.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, pos_crowne, rad_crown, params.wood_texture_file.c_str());
    U = context->getObjectPrimitiveUUIDs(objID);
    UUID_trunk_plant.insert(UUID_trunk_plant.end(), U.begin(), U.end());

    //---- Cordons -----//

    float cordon_length = std::max(0.f, params.cordon_length + getVariation(params.cordon_length_spread, generator));

    std::vector<float> rad_cord;
    rad_cord.push_back(cordon_radius);
    rad_cord.push_back(0.95f * cordon_radius);
    rad_cord.push_back(0.9f * cordon_radius);
    rad_cord.push_back(0.9f * cordon_radius);
    rad_cord.push_back(0.9f * cordon_radius);
    rad_cord.push_back(0.6f * cordon_radius);
    rad_cord.push_back(0.2f * cordon_radius);

    // West Cordon
    std::vector<vec3> pos_cordnw;
    pos_cordnw.push_back(make_vec3(0.7f * 0.5f * cordon_spacing * cost, 0.7f * 0.5f * cordon_spacing * sint, 0.99f * cordon_height));
    pos_cordnw.push_back(make_vec3(0.85f * 0.5f * cordon_spacing * cost + 0.025f * sint, 0.85f * 0.5f * cordon_spacing * sint + 0.025f * cost, cordon_height));
    pos_cordnw.push_back(make_vec3(0.95f * 0.5f * cordon_spacing * cost + 0.075f * sint, 0.95f * 0.5f * cordon_spacing * sint + 0.075f * cost, cordon_height));
    pos_cordnw.push_back(make_vec3(0.5f * cordon_spacing * cost + 0.12f * sint, 0.5f * cordon_spacing * sint + 0.12f * cost, cordon_height));
    pos_cordnw.push_back(make_vec3(0.5f * cordon_spacing * cost + 0.4f * cordon_length * sint, 0.5f * cordon_spacing * sint + 0.4f * cordon_length * cost, 0.94f * cordon_height));
    pos_cordnw.push_back(make_vec3(0.5f * cordon_spacing * cost + 0.8f * cordon_length * sint, 0.5f * cordon_spacing * sint + 0.8f * cordon_length * cost, 0.97f * cordon_height));
    pos_cordnw.push_back(make_vec3(0.5f * cordon_spacing * cost + cordon_length * sint, 0.5f * cordon_spacing * sint + cordon_length * cost, cordon_height));

    std::vector<vec3> tmp;
    tmp.resize(pos_cordnw.size());
    for (uint i = 0; i < pos_cordnw.size(); i++) {
        tmp.at(i) = pos_cordnw.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, tmp, rad_cord, params.wood_texture_file.c_str());
    UUID_branch_plant = context->getObjectPrimitiveUUIDs(objID);

    std::vector<vec3> pos_cordsw;
    pos_cordsw.push_back(make_vec3(0.7f * 0.5f * cordon_spacing * cost, 0.7f * 0.5f * cordon_spacing * sint, 0.99f * cordon_height));
    pos_cordsw.push_back(make_vec3(0.85f * 0.5f * cordon_spacing * cost - 0.025f * sint, 0.85f * 0.5f * cordon_spacing * sint - 0.025f * cost, cordon_height));
    pos_cordsw.push_back(make_vec3(0.95f * 0.5f * cordon_spacing * cost - 0.075f * sint, 0.95f * 0.5f * cordon_spacing * sint - 0.075f * cost, cordon_height));
    pos_cordsw.push_back(make_vec3(0.5f * cordon_spacing * cost - 0.12f * sint, 0.5f * cordon_spacing * sint - 0.12f * cost, cordon_height));
    pos_cordsw.push_back(make_vec3(0.5f * cordon_spacing * cost - 0.4f * cordon_length * sint, 0.5f * cordon_spacing * sint - 0.4f * cordon_length * cost, 0.94f * cordon_height));
    pos_cordsw.push_back(make_vec3(0.5f * cordon_spacing * cost - 0.8f * cordon_length * sint, 0.5f * cordon_spacing * sint - 0.8f * cordon_length * cost, 0.97f * cordon_height));
    pos_cordsw.push_back(make_vec3(0.5f * cordon_spacing * cost - cordon_length * sint, 0.5f * cordon_spacing * sint - cordon_length * cost, cordon_height));

    tmp.resize(pos_cordsw.size());
    for (uint i = 0; i < pos_cordsw.size(); i++) {
        tmp.at(i) = pos_cordsw.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, tmp, rad_cord, params.wood_texture_file.c_str());
    U = context->getObjectPrimitiveUUIDs(objID);
    UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

    // East Cordon
    std::vector<vec3> pos_cordne;
    pos_cordne.push_back(make_vec3(-0.7f * 0.5f * cordon_spacing * cost, -0.7f * 0.5f * cordon_spacing * sint, 0.99f * cordon_height));
    pos_cordne.push_back(make_vec3(-0.85f * 0.5f * cordon_spacing * cost + 0.025f * sint, -0.85f * 0.5f * cordon_spacing * sint + 0.025f * cost, cordon_height));
    pos_cordne.push_back(make_vec3(-0.95f * 0.5f * cordon_spacing * cost + 0.075f * sint, -0.95f * 0.5f * cordon_spacing * sint + 0.075f * cost, cordon_height));
    pos_cordne.push_back(make_vec3(-0.5f * cordon_spacing * cost + 0.12f * sint, -0.5f * cordon_spacing * sint + 0.12f * cost, cordon_height));
    pos_cordne.push_back(make_vec3(-0.5f * cordon_spacing * cost + 0.4f * cordon_length * sint, -0.5f * cordon_spacing * sint + 0.4f * cordon_length * cost, 0.94f * cordon_height));
    pos_cordne.push_back(make_vec3(-0.5f * cordon_spacing * cost + 0.8f * cordon_length * sint, -0.5f * cordon_spacing * sint + 0.8f * cordon_length * cost, 0.97f * cordon_height));
    pos_cordne.push_back(make_vec3(-0.5f * cordon_spacing * cost + cordon_length * sint, -0.5f * cordon_spacing * sint + cordon_length * cost, cordon_height));

    tmp.resize(pos_cordne.size());
    for (uint i = 0; i < pos_cordne.size(); i++) {
        tmp.at(i) = pos_cordne.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, tmp, rad_cord, params.wood_texture_file.c_str());
    U = context->getObjectPrimitiveUUIDs(objID);
    UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

    std::vector<vec3> pos_cordse;
    pos_cordse.push_back(make_vec3(-0.7f * 0.5f * cordon_spacing * cost, -0.7f * 0.5f * cordon_spacing * sint, 0.99f * cordon_height));
    pos_cordse.push_back(make_vec3(-0.85f * 0.5f * cordon_spacing * cost - 0.025f * sint, -0.85f * 0.5f * cordon_spacing * sint - 0.025f * cost, cordon_height));
    pos_cordse.push_back(make_vec3(-0.95f * 0.5f * cordon_spacing * cost - 0.075f * sint, -0.95f * 0.5f * cordon_spacing * sint - 0.075f * cost, cordon_height));
    pos_cordse.push_back(make_vec3(-0.5f * cordon_spacing * cost - 0.12f * sint, -0.5f * cordon_spacing * sint - 0.12f * cost, cordon_height));
    pos_cordse.push_back(make_vec3(-0.5f * cordon_spacing * cost - 0.4f * cordon_length * sint, -0.5f * cordon_spacing * sint - 0.4f * cordon_length * cost, 0.94f * cordon_height));
    pos_cordse.push_back(make_vec3(-0.5f * cordon_spacing * cost - 0.8f * cordon_length * sint, -0.5f * cordon_spacing * sint - 0.8f * cordon_length * cost, 0.97f * cordon_height));
    pos_cordse.push_back(make_vec3(-0.5f * cordon_spacing * cost - cordon_length * sint, -0.5f * cordon_spacing * sint - cordon_length * cost, cordon_height));

    tmp.resize(pos_cordse.size());
    for (uint i = 0; i < pos_cordse.size(); i++) {
        tmp.at(i) = pos_cordse.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, tmp, rad_cord, params.wood_texture_file.c_str());
    U = context->getObjectPrimitiveUUIDs(objID);
    UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

    //------- primary shoots ---------//

    uint ID0 = context->addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, PI_F), params.leaf_subdivisions, params.leaf_texture_file.c_str());

    float shoot_length = std::max(0.f, params.shoot_length + getVariation(params.shoot_length_spread, generator));
    float shoot_radius = std::max(1e-5f, params.shoot_radius + getVariation(params.shoot_radius_spread, generator));
    uint shoots_per_cordon = std::max(0u, params.shoots_per_cordon + getVariation(params.shoots_per_cordon_spread, generator));
    float shoot_angle_base = params.shoot_angle_base + getVariation(params.shoot_angle_base_spread, generator);
    float shoot_angle_tip = params.shoot_angle_tip + getVariation(params.shoot_angle_tip_spread, generator);

    float height = cordon_height + shoot_length;

    for (uint d = 0; d < 2; d++) { // cordons
        for (uint c = 0; c < 2; c++) { // shoot bend direction

            std::vector<vec3> pos_cord;
            float sign;
            if (c == 0) {
                if (d == 0) {
                    pos_cord = pos_cordsw;
                } else {
                    pos_cord = pos_cordnw;
                }
                sign = 1;
            } else {
                if (d == 0) {
                    pos_cord = pos_cordse;
                } else {
                    pos_cord = pos_cordne;
                }
                sign = -1;
            }

            float dx = fabs(pos_cord.back().y - pos_cord.at(0).y) / (float(shoots_per_cordon));

            for (int j = 1; j < shoots_per_cordon + 1; j++) {

                float frac_shoot = float(j) / float(shoots_per_cordon);

                vec3 cane_base = interpolateTube(pos_cord, frac_shoot);

                std::vector<float> rad_pshoot;
                std::vector<vec3> pos_pshoot;

                // cane base
                rad_pshoot.push_back(shoot_radius);
                pos_pshoot.push_back(cane_base);

                // cane nodes
                bool inside = false;
                float phirot = 0.5f * PI_F * (1 + (-0.5f + unif_distribution(generator)) * 1.0) + canopy_rotation;
                if (unif_distribution(generator) < 0.5) {
                    phirot += PI_F;
                    if (c == 0) {
                        inside = true;
                    }
                } else {
                    if (c == 1) {
                        inside = true;
                    }
                }

                float theta0;
                if (inside) {
                    theta0 = 0.5f * PI_F * unif_distribution(generator); //*(1.f+(-0.5+unif_distribution(generator))*0.6);
                } else {
                    theta0 = shoot_angle_base * (1.f + (-0.5 + unif_distribution(generator)) * 0.6);
                }
                float theta_end = shoot_angle_tip * (1.f + (-0.5 + unif_distribution(generator)) * 0.6);

                uint Nz = 2 * wood_subdivisions;
                float dz = ((1 + getVariation(0.1f, generator)) * height - cordon_height) / float(Nz);
                for (uint k = 1; k < Nz; k++) {

                    vec3 n = rotatePoint(make_vec3(0, 0, dz), (theta0 + (theta_end - theta0) * float(k) / float(Nz - 1)), phirot);

                    pos_pshoot.push_back(pos_pshoot.back() + n + make_vec3(getVariation(0.02f, generator), getVariation(0.01f, generator), 0));

                    rad_pshoot.push_back(shoot_radius);
                }

                rad_pshoot.back() = 0.0f;

                std::vector<vec3> tmp;
                tmp.resize(pos_pshoot.size());
                for (uint i = 0; i < pos_pshoot.size(); i++) {
                    tmp.at(i) = pos_pshoot.at(i) + origin;
                }

                objID = context->addTubeObject(wood_subdivisions, tmp, rad_pshoot, params.wood_texture_file.c_str());
                U = context->getObjectPrimitiveUUIDs(objID);
                UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

                if (is_dead) {
                    // Don't add grapes and leaves
                    continue;
                }

                // grape clusters
                float grape_radius = std::max(0.f, params.grape_radius + getVariation(params.grape_radius_spread, generator));
                float cluster_radius = std::max(0.f, params.cluster_radius + getVariation(params.cluster_radius_spread, generator));
                float cluster_height_max = std::max(0.f, params.cluster_height_max + getVariation(params.cluster_height_max_spread, generator));
                uint grape_subdivisions = std::max(0u, params.grape_subdivisions + getVariation(params.grape_subdivisions_spread, generator));
                std::vector<std::vector<uint>> UUID_grapes;
                if (grape_radius > 0 && cluster_radius > 0) {

                    float fgrape = 0.035 + (cluster_height_max - 0.035) * unif_distribution(generator);
                    vec3 p_grape = interpolateTube(tmp, fgrape);
                    int sgn = 1;
                    if (unif_distribution(generator) < 0.5) {
                        sgn = -1;
                    }
                    vec3 offset(sgn * (2.2 * cluster_radius + getVariation(0.1f, generator)) * sint, sgn * (2 * cluster_radius + getVariation(0.1f, generator)) * cost, 0.f);

                    UUID_grapes = addGrapeCluster(p_grape + offset, grape_radius, cluster_radius, params.grape_color, grape_subdivisions);
                }
                UUID_fruit_plant.push_back(UUID_grapes);

                // leaves
                if (params.leaf_width == 0) {
                    continue;
                }
                float leaf_width = std::max(0.f, params.leaf_width + getVariation(params.leaf_width_spread, generator));

                float flip = 0;
                if (unif_distribution(generator) < 0.5) {
                    flip = 1;
                }
                float leaf_spacing_fraction = std::max(0.f, params.leaf_spacing_fraction + getVariation(params.leaf_spacing_fraction_spread, generator));
                float lfrac = 1.f;
                int iter = 0;
                while (lfrac > 0.5 * leaf_width && iter < 100) {
                    iter++;

                    float lsize = fmaxf(leaf_width * (1.f - exp(-5.f * (1 - lfrac))), 0.1 * leaf_width);

                    vec3 pos_leaf = interpolateTube(pos_pshoot, lfrac);

                    vec3 parent_normal = interpolateTube(pos_pshoot, fmax(0, lfrac - 0.001)) - pos_leaf;
                    parent_normal.normalize();
                    vec3 leaf_offset = rotatePointAboutLine(make_vec3(0, lsize * (0.3 + getVariation(0.25f, generator)), 0), make_vec3(0, 0, 0), parent_normal, flip * PI_F + unif_distribution(generator) * 0.25 * PI_F);

                    float s;
                    if (int(flip) % 2 == 0) {
                        s = 1;
                    } else {
                        s = -1;
                    }

                    float Rphi = -canopy_rotation - s * 0.5f * PI_F * (1.f + getVariation(0.4f, generator));
                    float Rtheta = 0.4f * PI_F * (1.f + getVariation(0.1f, generator));

                    vec3 position = origin + pos_leaf + leaf_offset;

                    uint ID = context->copyObject(ID0);
                    context->getTileObjectPointer(ID)->scale(make_vec3(lsize, lsize, 1));
                    context->getObjectPointer(ID)->rotate(-Rtheta, "y");
                    context->getObjectPointer(ID)->rotate(Rphi, "z");
                    context->getObjectPointer(ID)->translate(position);

                    UUID_leaf_plant.push_back(context->getObjectPointer(ID)->getPrimitiveUUIDs());

                    lfrac = lfrac - leaf_spacing_fraction * lsize * (1.f + getVariation(0.25f, generator));

                    flip++;
                }
            }
        }
    }

    context->deleteObject(ID0);

    UUID_trunk.push_back(UUID_trunk_plant);
    UUID_branch.push_back(UUID_branch_plant);
    UUID_leaf.push_back(UUID_leaf_plant);
    UUID_fruit.push_back(UUID_fruit_plant);

    return UUID_leaf.size() - 1;
}

uint CanopyGenerator::grapevineUnilateral(const UnilateralGrapevineParameters &params, const vec3 &origin) {

    float mean_shoot_angle = 0.1 * PI_F;

    vector<uint> U;
    std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
    std::vector<std::vector<uint>> UUID_leaf_plant;
    std::vector<std::vector<std::vector<uint>>> UUID_fruit_plant;

    std::uniform_real_distribution<float> unif_distribution;

    float canopy_rotation = params.canopy_rotation + getVariation(params.canopy_rotation_spread, generator);

    float cost = cosf(canopy_rotation);
    float sint = sinf(canopy_rotation);

    bool is_dead = false;
    if (params.dead_probability > 0) {
        float random_draw = context->randu();
        is_dead = random_draw <= params.dead_probability;
    }

    float cordon_length = std::max(0.f, params.cordon_length + getVariation(params.cordon_length_spread, generator));

    //------ trunks -------//

    float trunk_radius = std::max(1e-5f, params.trunk_radius + getVariation(params.trunk_radius_spread, generator));
    float trunk_height = std::max(0.f, params.trunk_height + getVariation(params.trunk_height_spread, generator));

    std::vector<float> rad_main;
    rad_main.push_back(0.75 * trunk_radius);
    rad_main.push_back(0.8f * trunk_radius);
    rad_main.push_back(1.f * trunk_radius);
    rad_main.push_back(0.7f * trunk_radius);
    rad_main.push_back(0.95f * trunk_radius);
    rad_main.push_back(0.1 * trunk_radius);
    std::vector<vec3> pos_main;
    pos_main.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius) * cost, (-cordon_length + 0.5f * trunk_radius) * sint, 0.0f));
    pos_main.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius) * cost, (-cordon_length + 0.5f * trunk_radius) * sint, 0.2f * trunk_height));
    pos_main.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius) * cost, (-cordon_length + 0.5f * trunk_radius) * sint, 0.22f * trunk_height));
    pos_main.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius) * cost, (-cordon_length + 0.5f * trunk_radius) * sint, 0.6f * trunk_height));
    pos_main.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius) * cost, (-cordon_length + 0.5f * trunk_radius) * sint, 0.96f * trunk_height));
    pos_main.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius) * cost, (-cordon_length + 0.5f * trunk_radius) * sint, trunk_height));

    for (uint i = 0; i < rad_main.size(); i++) {
        pos_main.at(i) = pos_main.at(i) + origin;
    }

    int wood_subdivisions = std::max(0, params.wood_subdivisions + getVariation(params.wood_subdivisions_spread, generator));

    uint objID = context->addTubeObject(wood_subdivisions, pos_main, rad_main, params.wood_texture_file.c_str());
    UUID_trunk_plant = context->getObjectPrimitiveUUIDs(objID);

    //---- Cordons -----//

    float cordon_height = std::max(0.f, params.cordon_height + getVariation(params.cordon_height_spread, generator));
    float cordon_radius = std::max(1e-5f, params.cordon_radius + getVariation(params.cordon_radius_spread, generator));

    float total_cordons_length = 2 * cordon_length;

    float diff = cordon_height - trunk_height;

    // Cordon
    std::vector<float> rad_cord;
    rad_cord.push_back(cordon_radius);
    rad_cord.push_back(0.95 * cordon_radius);
    rad_cord.push_back(0.9f * cordon_radius);
    rad_cord.push_back(0.85f * cordon_radius);
    rad_cord.push_back(0.8 * cordon_radius);
    rad_cord.push_back(0.6 * cordon_radius);
    rad_cord.push_back(0.2f * cordon_radius);
    std::vector<vec3> pos_cord;
    pos_cord.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius + 0.01f * total_cordons_length) * cost, (-cordon_length + 0.5f * trunk_radius + 0.01f * total_cordons_length) * sint, 0.95f * trunk_height));
    pos_cord.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius + 0.05f * total_cordons_length) * cost, (-cordon_length + 0.5f * trunk_radius + 0.05f * total_cordons_length) * sint, trunk_height + 0.1f * diff));
    pos_cord.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius + 0.15f * total_cordons_length) * cost, (-cordon_length + 0.5f * trunk_radius + 0.15f * total_cordons_length) * sint, trunk_height + 0.65f * diff));
    pos_cord.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius + 0.45f * total_cordons_length) * cost, (-cordon_length + 0.5f * trunk_radius + 0.45f * total_cordons_length) * sint, trunk_height + 0.95f * diff));
    pos_cord.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius + 0.6f * total_cordons_length) * cost, (-cordon_length + 0.5f * trunk_radius + 0.6f * total_cordons_length) * sint, trunk_height + 1.05f * diff));
    pos_cord.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius + 0.85f * total_cordons_length) * cost, (-cordon_length + 0.5f * trunk_radius + 0.85f * total_cordons_length) * sint, trunk_height + diff));
    pos_cord.push_back(make_vec3((-cordon_length + 0.5f * trunk_radius + 1.0f * total_cordons_length) * cost, (-cordon_length + 0.5f * trunk_radius + 1.0f * total_cordons_length) * sint, trunk_height + diff));

    std::vector<vec3> tmp;
    tmp.resize(pos_cord.size());
    for (uint i = 0; i < pos_cord.size(); i++) {
        tmp.at(i) = pos_cord.at(i) + origin;
    }

    objID = context->addTubeObject(wood_subdivisions, tmp, rad_cord, params.wood_texture_file.c_str());
    UUID_branch_plant = context->getObjectPrimitiveUUIDs(objID);

    //------- primary shoots ---------//

    uint ID0 = context->addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, PI_F), params.leaf_subdivisions, params.leaf_texture_file.c_str());

    float shoot_length = std::max(0.f, params.shoot_length + getVariation(params.shoot_length_spread, generator));
    float shoot_radius = std::max(1e-5f, params.shoot_radius + getVariation(params.shoot_radius_spread, generator));
    uint shoots_per_cordon = std::max(0u, params.shoots_per_cordon + getVariation(params.shoots_per_cordon_spread, generator));

    float height = cordon_height + shoot_length;

    float dx = fabs(pos_cord.back().y - pos_cord.at(0).y) / (float(shoots_per_cordon));

    for (int j = 1; j < shoots_per_cordon + 1; j++) {

        float frac_shoot = float(j) / float(shoots_per_cordon);

        vec3 cane_base = interpolateTube(pos_cord, frac_shoot);

        std::vector<float> rad_pshoot;
        std::vector<vec3> pos_pshoot;

        // cane base
        rad_pshoot.push_back(shoot_radius);
        pos_pshoot.push_back(cane_base);

        // cane nodes
        float phirot = (0.5f - unif_distribution(generator)) * 0.5 * PI_F;
        phirot = 0.5 * PI_F;
        if (unif_distribution(generator) < 0.5) {
            phirot += PI_F;
        }

        float theta0 = (0.3f - unif_distribution(generator)) * 0.6;
        float theta_end = (0.1f - unif_distribution(generator)) * 0.2;

        uint Nz = 2 * wood_subdivisions;
        float dz = ((1 + getVariation(0.1f, generator)) * height - cordon_height) / float(Nz);
        for (uint k = 1; k < Nz; k++) {

            vec3 n = rotatePoint(make_vec3(0, 0, dz), mean_shoot_angle * PI_F / 180.f * (theta0 + 1.2 * float(k) / float(Nz - 1)), phirot);

            pos_pshoot.push_back(pos_pshoot.back() + n + make_vec3(getVariation(0.02f, generator), getVariation(0.01f, generator), 0));

            rad_pshoot.push_back(shoot_radius);
        }

        rad_pshoot.back() = 0.0f;

        std::vector<vec3> tmp;
        tmp.resize(pos_pshoot.size());
        for (uint i = 0; i < pos_pshoot.size(); i++) {
            tmp.at(i) = pos_pshoot.at(i) + origin;
        }

        objID = context->addTubeObject(wood_subdivisions, tmp, rad_pshoot, params.wood_texture_file.c_str());
        U = context->getObjectPrimitiveUUIDs(objID);
        UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

        if (is_dead) {
            // Don't add grapes and leaves
            continue;
        }

        // grape clusters
        float grape_radius = std::max(0.f, params.grape_radius + getVariation(params.grape_radius_spread, generator));
        float cluster_radius = std::max(0.f, params.cluster_radius + getVariation(params.cluster_radius_spread, generator));
        float cluster_height_max = std::max(0.f, params.cluster_height_max + getVariation(params.cluster_height_max_spread, generator));
        uint grape_subdivisions = std::max(0u, params.grape_subdivisions + getVariation(params.grape_subdivisions_spread, generator));
        std::vector<std::vector<uint>> UUID_grapes;
        if (grape_radius > 0 && cluster_radius > 0) {

            float fgrape = 0.035 + (cluster_height_max - 0.035) * unif_distribution(generator);
            vec3 p_grape = interpolateTube(tmp, fgrape);
            int sgn = 1;
            if (unif_distribution(generator) < 0.5) {
                sgn = -1;
            }
            vec3 offset(sgn * (2.2f * cluster_radius + getVariation(0.1f, generator)) * sint, sgn * (2 * cluster_radius + getVariation(0.1f, generator)) * cost, 0.f);
            UUID_grapes = addGrapeCluster(p_grape + offset, grape_radius, cluster_radius, params.grape_color, grape_subdivisions);
        }
        UUID_fruit_plant.push_back(UUID_grapes);

        // leaves
        if (params.leaf_width == 0) {
            continue;
        }
        float leaf_width = std::max(0.f, params.leaf_width + getVariation(params.leaf_width_spread, generator));

        float flip = 0;
        if (unif_distribution(generator) < 0.5) {
            flip = 1;
        }
        float leaf_spacing_fraction = std::max(0.f, params.leaf_spacing_fraction + getVariation(params.leaf_spacing_fraction_spread, generator));
        float lfrac = 1.f;
        int iter = 0;
        while (lfrac > 0.5 * leaf_width && iter < 100) {
            iter++;

            float lsize = fmaxf(leaf_width * (1.f - exp(-5.f * (1 - lfrac))), 0.1 * leaf_width);

            vec3 pos_leaf = interpolateTube(pos_pshoot, lfrac);

            vec3 parent_normal = interpolateTube(pos_pshoot, fmax(0, lfrac - 0.001)) - pos_leaf;
            parent_normal.normalize();
            vec3 leaf_offset = rotatePointAboutLine(make_vec3(0, lsize * (0.3 + getVariation(0.25f, generator)), 0), make_vec3(0, 0, 0), parent_normal, flip * PI_F + unif_distribution(generator) * 0.25f * PI_F);

            float s;
            if (int(flip) % 2 == 0) {
                s = 1;
            } else {
                s = -1;
            }

            float Rphi = -canopy_rotation - s * 0.5f * PI_F * (1.f + getVariation(0.4f, generator));
            float Rtheta = 0.4f * PI_F * (1.f + getVariation(0.1f, generator));

            vec3 position = origin + pos_leaf - leaf_offset;

            uint ID = context->copyObject(ID0);
            context->getTileObjectPointer(ID)->scale(make_vec3(lsize, lsize, 1));
            context->getObjectPointer(ID)->rotate(-Rtheta, "y");
            context->getObjectPointer(ID)->rotate(Rphi, "z");
            context->getObjectPointer(ID)->translate(position);

            UUID_leaf_plant.push_back(context->getObjectPointer(ID)->getPrimitiveUUIDs());

            lfrac = lfrac - leaf_spacing_fraction * lsize * (1.f + getVariation(0.25f, generator));

            flip++;
        }
    }

    context->deleteObject(ID0);

    UUID_trunk.push_back(UUID_trunk_plant);
    UUID_branch.push_back(UUID_branch_plant);
    UUID_leaf.push_back(UUID_leaf_plant);
    UUID_fruit.push_back(UUID_fruit_plant);

    return UUID_leaf.size() - 1;
}

uint CanopyGenerator::grapevineGoblet(const GobletGrapevineParameters &params, const vec3 &origin) {

    vector<uint> U;
    std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
    std::vector<std::vector<uint>> UUID_leaf_plant;
    std::vector<std::vector<std::vector<uint>>> UUID_fruit_plant;

    std::uniform_real_distribution<float> unif_distribution;

    float canopy_rotation = params.canopy_rotation + getVariation(params.canopy_rotation_spread, generator);

    float cost = cosf(canopy_rotation);
    float sint = sinf(canopy_rotation);

    bool is_dead = false;
    if (params.dead_probability > 0) {
        float random_draw = context->randu();
        is_dead = random_draw <= params.dead_probability;
    }

    //------ trunks -------//

    float trunk_radius = std::max(1e-5f, params.trunk_radius + getVariation(params.trunk_radius_spread, generator));
    float trunk_height = std::max(0.f, params.trunk_height + getVariation(params.trunk_height_spread, generator));

    std::vector<float> rad_main;
    rad_main.push_back(0.75 * trunk_radius);
    rad_main.push_back(0.8f * trunk_radius);
    rad_main.push_back(1.f * trunk_radius);
    rad_main.push_back(0.7f * trunk_radius);
    rad_main.push_back(0.95f * trunk_radius);
    rad_main.push_back(0.1 * trunk_radius);
    std::vector<vec3> pos_main;
    pos_main.push_back(make_vec3(0., 0., 0.0));
    pos_main.push_back(make_vec3(0, 0, 0.2f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.22f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.6f * trunk_height));
    pos_main.push_back(make_vec3(0, 0, 0.96f * trunk_height));
    pos_main.push_back(make_vec3(0., 0., trunk_height));

    for (uint i = 0; i < rad_main.size(); i++) {
        pos_main.at(i) = pos_main.at(i) + origin;
    }

    int wood_subdivisions = std::max(0, params.wood_subdivisions + getVariation(params.wood_subdivisions_spread, generator));

    uint objID = context->addTubeObject(wood_subdivisions, pos_main, rad_main, params.wood_texture_file.c_str());
    UUID_trunk_plant = context->getObjectPrimitiveUUIDs(objID);

    //------- primary shoots ---------//

    float cordon_height = std::max(0.f, params.cordon_height + getVariation(params.cordon_height_spread, generator));

    float shoot_length = std::max(0.f, params.shoot_length + getVariation(params.shoot_length_spread, generator));
    float shoot_radius = std::max(0.f, params.shoot_radius + getVariation(params.shoot_radius_spread, generator));
    uint shoots_per_cordon = std::max(0u, params.shoots_per_cordon + getVariation(params.shoots_per_cordon_spread, generator));

    uint ID0 = context->addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, PI_F), params.leaf_subdivisions, params.leaf_texture_file.c_str());

    for (uint c = 0; c < 2; c++) {

        for (int j = 1; j < shoots_per_cordon + 1; j++) {

            float frac_shoot = float(j) / float(shoots_per_cordon);

            float height = (cordon_height + shoot_length) * (1.f + 0.55 * (1 - frac_shoot));

            vec3 cane_base = make_vec3(0., 0., trunk_height);

            std::vector<float> rad_pshoot;
            std::vector<vec3> pos_pshoot;

            // cane base
            rad_pshoot.push_back(shoot_radius);
            pos_pshoot.push_back(cane_base);

            float theta0 = 0.5 * PI_F * (1.f - float(j - 1) / float(shoots_per_cordon));
            float theta_end = (0.1f - unif_distribution(generator)) * 0.2;

            uint Nz = 2 * wood_subdivisions;
            float dz = ((1.f + getVariation(0.1f, generator)) * height - cordon_height) / float(Nz);
            for (uint k = 1; k < Nz; k++) {

                vec3 n = rotatePoint(make_vec3(0, 0, dz), (theta0 + (theta_end - theta0) * float(k) / float(Nz - 1)), canopy_rotation + PI_F * float(c));

                pos_pshoot.push_back(pos_pshoot.back() + n + make_vec3(getVariation(0.02f, generator), getVariation(0.01f, generator), 0.f));

                rad_pshoot.push_back(shoot_radius);
            }

            rad_pshoot.back() = 0.0f;

            std::vector<vec3> tmp;
            tmp.resize(pos_pshoot.size());
            for (uint i = 0; i < pos_pshoot.size(); i++) {
                tmp.at(i) = pos_pshoot.at(i) + origin;
            }

            objID = context->addTubeObject(wood_subdivisions, tmp, rad_pshoot, params.wood_texture_file.c_str());
            U = context->getObjectPrimitiveUUIDs(objID);
            UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end());

            if (is_dead) {
                // Don't add grapes and leaves
                continue;
            }

            // grape clusters
            float grape_radius = std::max(0.f, params.grape_radius + getVariation(params.grape_radius_spread, generator));
            float cluster_radius = std::max(0.f, params.cluster_radius + getVariation(params.cluster_radius_spread, generator));
            float cluster_height_max = std::max(0.f, params.cluster_height_max + getVariation(params.cluster_height_max_spread, generator));
            uint grape_subdivisions = std::max(0u, params.grape_subdivisions + getVariation(params.grape_subdivisions_spread, generator));
            std::vector<std::vector<uint>> UUID_grapes;
            if (grape_radius > 0 && cluster_radius > 0) {

                float fgrape = 0.035 + (cluster_height_max - 0.035) * unif_distribution(generator);
                vec3 p_grape = interpolateTube(tmp, fgrape);
                int sgn = 1;
                if (unif_distribution(generator) < 0.5) {
                    sgn = -1;
                }
                vec3 offset(sgn * (2.2f * cluster_radius + getVariation(0.1f, generator)) * sint, sgn * (2.f * cluster_radius + getVariation(0.1f, generator)) * cost, 0.f);
                UUID_grapes = addGrapeCluster(p_grape + offset, grape_radius, cluster_radius, params.grape_color, grape_subdivisions);
            }
            UUID_fruit_plant.push_back(UUID_grapes);

            // leaves
            if (params.leaf_width == 0) {
                continue;
            }
            float leaf_width = std::max(0.f, params.leaf_width + getVariation(params.leaf_width_spread, generator));

            float flip = 0;
            if (unif_distribution(generator) < 0.5) {
                flip = 1;
            }
            float leaf_spacing_fraction = std::max(0.f, params.leaf_spacing_fraction + getVariation(params.leaf_spacing_fraction_spread, generator));
            float lfrac = 1.f;
            int iter = 0;
            while (lfrac > 0.5 * leaf_width && iter < 100) {
                iter++;

                float lsize = fmaxf(leaf_width * (1.f - exp(-5.f * (1 - lfrac))), 0.1 * leaf_width);

                vec3 pos_leaf = interpolateTube(pos_pshoot, lfrac);

                vec3 parent_normal = interpolateTube(pos_pshoot, fmax(0, lfrac - 0.001)) - pos_leaf;
                parent_normal.normalize();
                vec3 leaf_offset = rotatePointAboutLine(make_vec3(0, lsize * (0.3f + getVariation(0.25f, generator)), 0), make_vec3(0, 0, 0), parent_normal, flip * PI_F + unif_distribution(generator) * 0.25f * PI_F);

                float s;
                if (int(flip) % 2 == 0) {
                    s = 1;
                } else {
                    s = -1;
                }

                float Rphi = -canopy_rotation - s * 0.5 * PI_F * (1.f + getVariation(0.4f, generator));
                float Rtheta = 0.4f * PI_F * (1.f + getVariation(0.1f, generator));

                vec3 position = origin + pos_leaf - leaf_offset;

                uint ID = context->copyObject(ID0);
                context->getTileObjectPointer(ID)->scale(make_vec3(lsize, lsize, 1));
                context->getObjectPointer(ID)->rotate(-Rtheta, "y");
                context->getObjectPointer(ID)->rotate(Rphi, "z");
                context->getObjectPointer(ID)->translate(position);

                UUID_leaf_plant.push_back(context->getObjectPointer(ID)->getPrimitiveUUIDs());

                lfrac = lfrac - leaf_spacing_fraction * lsize * (1.f + getVariation(0.25f, generator));

                flip++;
            }
        }
    }

    context->deleteObject(ID0);

    UUID_trunk.push_back(UUID_trunk_plant);
    UUID_branch.push_back(UUID_branch_plant);
    UUID_leaf.push_back(UUID_leaf_plant);
    UUID_fruit.push_back(UUID_fruit_plant);

    return UUID_leaf.size() - 1;
}
