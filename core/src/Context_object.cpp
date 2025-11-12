/**
 * \file "Context_object.cpp" Context compound object definitions and methods.
 *
 * Copyright (C) 2016-2025 Brian Bailey
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "Context.h"

using namespace helios;

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius) {
    return addSphereObject(Ndivs, center, {radius, radius, radius}, {0.f, 0.75f, 0.f}); // Default color is green
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color) {
    return addSphereObject(Ndivs, center, {radius, radius, radius}, color);
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const char *texturefile) {
    return addSphereObject(Ndivs, center, {radius, radius, radius}, texturefile);
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius) {
    return addSphereObject(Ndivs, center, radius, {0.f, 0.75f, 0.f}); // Default color is green
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius, const RGBcolor &color) {
    if (radius.x <= 0.f || radius.y <= 0.f || radius.z <= 0.f) {
        helios_runtime_error("ERROR (Context::addSphereObject): Radius of sphere must be positive.");
    }

    std::vector<uint> UUID;
    UUID.reserve(Ndivs * (Ndivs - 2) * 2 + 2 * Ndivs);

    float dtheta = PI_F / float(Ndivs);
    float dphi = 2.0f * PI_F / float(Ndivs);

    vec3 cart;

    // bottom cap
    for (int j = 0; j < Ndivs; j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F, 0));
        vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + dtheta, float(j) * dphi));
        vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + dtheta, float(j + 1) * dphi));
        vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);

        UUID.push_back(addTriangle(v0, v1, v2, color));
    }

    // top cap
    for (int j = 0; j < Ndivs; j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F, 0));
        vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);

        UUID.push_back(addTriangle(v2, v1, v0, color));
    }

    // middle
    for (int j = 0; j < Ndivs; j++) {
        for (int i = 1; i < Ndivs - 1; i++) {
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i) * dtheta, float(j) * dphi));
            vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i + 1) * dtheta, float(j) * dphi));
            vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);

            UUID.push_back(addTriangle(v0, v1, v2, color));
            UUID.push_back(addTriangle(v0, v2, v3, color));
        }
    }

    auto *sphere_new = (new Sphere(currentObjectID, UUID, Ndivs, "", this));

    float T[16], transform[16];
    sphere_new->getTransformationMatrix(transform);

    makeScaleMatrix(radius, T);
    matmult(T, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    sphere_new->setTransformationMatrix(transform);

    sphere_new->setColor(color);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = sphere_new;
    currentObjectID++;
    return currentObjectID - 1;
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addSphereObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addSphereObject): Texture file " + std::string(texturefile) + " does not exist.");
    } else if (radius.x <= 0.f || radius.y <= 0.f || radius.z <= 0.f) {
        helios_runtime_error("ERROR (Context::addSphereObject): Radius of sphere must be positive.");
    }

    std::vector<uint> UUID;
    UUID.reserve(Ndivs * (Ndivs - 2) * 2 + 2 * Ndivs);

    float dtheta = PI_F / float(Ndivs);
    float dphi = 2.0f * PI_F / float(Ndivs);

    vec3 cart;

    // bottom cap
    for (int j = 0; j < Ndivs; j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F, 0));
        vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + dtheta, float(j) * dphi));
        vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + dtheta, float(j + 1) * dphi));
        vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);

        vec3 n0 = v0 - center;
        n0.normalize();
        vec3 n1 = v1 - center;
        n1.normalize();
        vec3 n2 = v2 - center;
        n2.normalize();

        vec2 uv0 = make_vec2(1.f - atan2f(sinf((float(j) + 0.5f) * dphi), -cosf((float(j) + 0.5f) * dphi)) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
        vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
        vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);

        if (j == Ndivs - 1) {
            uv2.x = 1;
        }

        uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
        if (getPrimitiveArea(triangle_uuid) > 0) {
            UUID.push_back(triangle_uuid);
        } else {
            deletePrimitive(triangle_uuid);
        }
    }

    // top cap
    for (int j = 0; j < Ndivs; j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F, 0));
        vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        ;

        vec3 n0 = v0 - center;
        n0.normalize();
        vec3 n1 = v1 - center;
        n1.normalize();
        vec3 n2 = v2 - center;
        n2.normalize();

        vec2 uv0 = make_vec2(1.f - atan2f(sinf((float(j) + 0.5f) * dphi), -cosf((float(j) + 0.5f) * dphi)) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
        vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
        vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);

        if (j == Ndivs - 1) {
            uv2.x = 1;
        }

        uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
        if (getPrimitiveArea(triangle_uuid) > 0) {
            UUID.push_back(triangle_uuid);
        } else {
            deletePrimitive(triangle_uuid);
        }
    }

    // middle
    for (int j = 0; j < Ndivs; j++) {
        for (int i = 1; i < Ndivs - 1; i++) {
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i) * dtheta, float(j) * dphi));
            vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i + 1) * dtheta, float(j) * dphi));
            vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
            cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + float(i) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);

            vec3 n0 = v0 - center;
            n0.normalize();
            vec3 n1 = v1 - center;
            n1.normalize();
            vec3 n2 = v2 - center;
            n2.normalize();
            vec3 n3 = v3 - center;
            n3.normalize();

            vec2 uv0 = make_vec2(1.f - atan2f(n0.x, -n0.y) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
            vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
            vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);
            vec2 uv3 = make_vec2(1.f - atan2f(n3.x, -n3.y) / (2.f * PI_F) - 0.5f, 1.f - n3.z * 0.5f - 0.5f);

            if (j == Ndivs - 1) {
                uv2.x = 1;
                uv3.x = 1;
            }

            uint triangle_uuid1 = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
            if (getPrimitiveArea(triangle_uuid1) > 0) {
                UUID.push_back(triangle_uuid1);
            } else {
                deletePrimitive(triangle_uuid1);
            }

            uint triangle_uuid2 = addTriangle(v0, v2, v3, texturefile, uv0, uv2, uv3);
            if (getPrimitiveArea(triangle_uuid2) > 0) {
                UUID.push_back(triangle_uuid2);
            } else {
                deletePrimitive(triangle_uuid2);
            }
        }
    }

    auto *sphere_new = (new Sphere(currentObjectID, UUID, Ndivs, texturefile, this));

    float T[16], transform[16];
    sphere_new->getTransformationMatrix(transform);

    makeScaleMatrix(radius, T);
    matmult(T, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    sphere_new->setTransformationMatrix(transform);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = sphere_new;
    currentObjectID++;

    return currentObjectID - 1;
}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv) {
    RGBcolor color(0.f, 0.75f, 0.f); // Default color is green

    return addTileObject(center, size, rotation, subdiv, color);
}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const RGBcolor &color) {
    if (size.x == 0 || size.y == 0) {
        helios_runtime_error("ERROR (Context::addTileObject): Size of tile must be greater than 0.");
    } else if (subdiv.x < 1 || subdiv.y < 1) {
        helios_runtime_error("ERROR (Context::addTileObject): Number of tile subdivisions must be greater than 0.");
    }

    std::vector<uint> UUID;
    UUID.reserve(subdiv.x * subdiv.y);

    vec2 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);

    for (uint j = 0; j < subdiv.y; j++) {
        for (uint i = 0; i < subdiv.x; i++) {
            vec3 subcenter = make_vec3(-0.5f * size.x + (float(i) + 0.5f) * subsize.x, -0.5f * size.y + (float(j) + 0.5f) * subsize.y, 0.f);

            UUID.push_back(addPatch(subcenter, subsize, make_SphericalCoord(0, 0), color));

            if (rotation.elevation != 0.f) {
                getPrimitivePointer_private(UUID.back())->rotate(-rotation.elevation, "x");
            }
            if (rotation.azimuth != 0.f) {
                getPrimitivePointer_private(UUID.back())->rotate(-rotation.azimuth, "z");
            }
            getPrimitivePointer_private(UUID.back())->translate(center);
        }
    }

    auto *tile_new = (new Tile(currentObjectID, UUID, subdiv, "", this));

    float T[16], S[16], R[16];

    float transform[16];
    tile_new->getTransformationMatrix(transform);

    makeScaleMatrix(make_vec3(size.x, size.y, 1.f), S);
    matmult(S, transform, transform);

    makeRotationMatrix(-rotation.elevation, "x", R);
    matmult(R, transform, transform);
    makeRotationMatrix(-rotation.azimuth, "z", R);
    matmult(R, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    tile_new->setTransformationMatrix(transform);

    tile_new->setColor(color);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    tile_new->object_origin = center;

    objects[currentObjectID] = tile_new;
    currentObjectID++;
    return currentObjectID - 1;
}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char *texturefile) {
    return addTileObject(center, size, rotation, subdiv, texturefile, make_int2(1, 1));
}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char *texturefile, const int2 &texture_repeat) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addTileObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addTileObject): Texture file " + std::string(texturefile) + " does not exist.");
    } else if (size.x == 0 || size.y == 0) {
        helios_runtime_error("ERROR (Context::addTileObject): Size of tile must be greater than 0.");
    } else if (subdiv.x < 1 || subdiv.y < 1) {
        helios_runtime_error("ERROR (Context::addTileObject): Number of tile subdivisions must be greater than 0.");
    } else if (texture_repeat.x < 1 || texture_repeat.y < 1) {
        helios_runtime_error("ERROR (Context::addTileObject): Number of texture repeats must be greater than 0.");
    }

    // Automatically resize the repeat count so that it evenly divides the subdivisions.
    int2 repeat = texture_repeat;
    repeat.x = std::min(subdiv.x, repeat.x);
    repeat.y = std::min(subdiv.y, repeat.y);
    while (subdiv.x % repeat.x != 0) {
        repeat.x--;
    }
    while (subdiv.y % repeat.y != 0) {
        repeat.y--;
    }

    std::vector<uint> UUID;
    UUID.reserve(subdiv.x * subdiv.y);

    vec2 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);

    std::vector<helios::vec2> uv(4);
    int2 sub_per_repeat;
    sub_per_repeat.x = subdiv.x / repeat.x;
    sub_per_repeat.y = subdiv.y / repeat.y;
    vec2 uv_sub;
    uv_sub.x = 1.f / float(sub_per_repeat.x);
    uv_sub.y = 1.f / float(sub_per_repeat.y);

    addTexture(texturefile);

    const int2 &sz = textures.at(texturefile).getImageResolution();
    if (subdiv.x >= repeat.x * sz.x || subdiv.y >= repeat.y * sz.y) {
        helios_runtime_error("ERROR (Context::addTileObject): The resolution of the texture image '" + std::string(texturefile) + "' is lower than the number of tile subdivisions. Increase resolution of the texture image.");
    }

    for (uint j = 0; j < subdiv.y; j++) {
        for (uint i = 0; i < subdiv.x; i++) {
            vec3 subcenter = make_vec3(-0.5f * size.x + (float(i) + 0.5f) * subsize.x, -0.5f * size.y + (float(j) + 0.5f) * subsize.y, 0.f);

            uint i_local = i % sub_per_repeat.x;
            uint j_local = j % sub_per_repeat.y;
            uv.at(0) = make_vec2(float(i_local) * uv_sub.x, float(j_local) * uv_sub.y);
            uv.at(1) = make_vec2(float(i_local + 1) * uv_sub.x, float(j_local) * uv_sub.y);
            uv.at(2) = make_vec2(float(i_local + 1) * uv_sub.x, float(j_local + 1) * uv_sub.y);
            uv.at(3) = make_vec2(float(i_local) * uv_sub.x, float(j_local + 1) * uv_sub.y);

            auto *patch_new = (new Patch(texturefile, uv, textures, 0, currentUUID));

            // \todo This is causing problems in the radiation intersection.
            // if( patch_new->getSolidFraction()==0 ){
            //   delete patch_new;
            //   continue;
            // }

            assert(size.x > 0.f && size.y > 0.f);
            patch_new->scale(make_vec3(subsize.x, subsize.y, 1));

            patch_new->translate(subcenter);

            if (rotation.elevation != 0) {
                patch_new->rotate(-rotation.elevation, "x");
            }
            if (rotation.azimuth != 0) {
                patch_new->rotate(-rotation.azimuth, "z");
            }

            patch_new->translate(center);

            primitives[currentUUID] = patch_new;
            currentUUID++;
            UUID.push_back(currentUUID - 1);
        }
    }

    auto *tile_new = (new Tile(currentObjectID, UUID, subdiv, texturefile, this));

    float T[16], S[16], R[16];

    float transform[16];
    tile_new->getTransformationMatrix(transform);

    makeScaleMatrix(make_vec3(size.x, size.y, 1.f), S);
    matmult(S, transform, transform);

    makeRotationMatrix(-rotation.elevation, "x", R);
    matmult(R, transform, transform);
    makeRotationMatrix(-rotation.azimuth, "z", R);
    matmult(R, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    tile_new->setTransformationMatrix(transform);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    tile_new->object_origin = center;

    objects[currentObjectID] = tile_new;
    currentObjectID++;
    return currentObjectID - 1;
}

uint Context::addTubeObject(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius) {
    uint node_count = nodes.size();

    std::vector<RGBcolor> color(node_count);

    for (uint i = 0; i < node_count; i++) {
        color.at(i) = make_RGBcolor(0.f, 0.75f, 0.f); // Default color is green
    }

    return addTubeObject(radial_subdivisions, nodes, radius, color);
}

uint Context::addTubeObject(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color) {
    const uint node_count = nodes.size();

    if (node_count == 0) {
        helios_runtime_error("ERROR (Context::addTubeObject): Node and radius arrays are empty.");
    } else if (node_count != radius.size()) {
        helios_runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `radius' arguments must agree.");
    } else if (node_count != color.size()) {
        helios_runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `color' arguments must agree.");
    }

    // Clamp very small radii to avoid creating degenerate triangles
    const float min_radius_threshold = 1e-5f;
    std::vector<float> radius_clamped = radius;
    for (int i = 0; i < node_count; i++) {
        if (radius_clamped[i] < min_radius_threshold && radius_clamped[i] >= 0) {
            radius_clamped[i] = min_radius_threshold;
        }
    }

    vec3 axial_vector;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3>> triangle_vertices;
    resize_vector(triangle_vertices, radial_subdivisions + 1, node_count);

    // Initialize trigonometric factors for circle points
    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    vec3 initial_radial(1.0f, 0.0f, 0.0f);
    vec3 previous_axial_vector;
    vec3 previous_radial_dir;

    for (int i = 0; i < node_count; i++) { // Looping over tube segments
        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTubeObject): Radius of tube must be positive.");
        }

        if (i == 0) {
            axial_vector = nodes[i + 1] - nodes[i];
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }
            if (fabs(axial_vector * initial_radial) > 0.95f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
            }
            // Also handle nearly vertical axes
            if (fabs(axial_vector.z) > 0.95f) {
                initial_radial = vec3(1.0f, 0.0f, 0.0f); // Use horizontal radial for vertical axes
            }
            previous_radial_dir = cross(axial_vector, initial_radial).normalize();
        } else {
            if (i == node_count - 1) {
                axial_vector = nodes[i] - nodes[i - 1];
            } else {
                axial_vector = 0.5f * ((nodes[i] - nodes[i - 1]) + (nodes[i + 1] - nodes[i]));
            }
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }

            // Calculate radial direction using parallel transport
            vec3 rotation_axis = cross(previous_axial_vector, axial_vector);
            if (rotation_axis.magnitude() > 1e-5) { // More conservative threshold
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
            } else {
                // Vectors are nearly parallel, use robust fallback
                vec3 fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                if (fabs(axial_vector * fallback_radial) > 0.95f) {
                    fallback_radial = vec3(0.0f, 1.0f, 0.0f);
                }
                if (fabs(axial_vector.z) > 0.95f) {
                    fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                }
                previous_radial_dir = cross(axial_vector, fallback_radial).normalize();
            }
        }

        previous_axial_vector = axial_vector;

        vec3 radial_dir = previous_radial_dir;
        vec3 orthogonal_dir = cross(radial_dir, axial_vector);
        orthogonal_dir.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal = cfact[j] * radius_clamped[i] * radial_dir + sfact[j] * radius_clamped[i] * orthogonal_dir;
            triangle_vertices[i][j] = nodes[i] + normal;
        }
    }


    std::vector<uint> UUIDs(2 * (node_count - 1) * radial_subdivisions);
    vec3 v0, v1, v2;

    int ii = 0;
    for (int j = 0; j < radial_subdivisions; j++) {
        for (int i = 0; i < node_count - 1; i++) {
            v0 = triangle_vertices[i][j];
            v1 = triangle_vertices[i + 1][j + 1];
            v2 = triangle_vertices[i][j + 1];

            UUIDs.at(ii) = addTriangle(v0, v1, v2, color.at(i));

            v0 = triangle_vertices[i][j];
            v1 = triangle_vertices[i + 1][j];
            v2 = triangle_vertices[i + 1][j + 1];

            UUIDs.at(ii + 1) = addTriangle(v0, v1, v2, color.at(i));

            ii += 2;
        }
    }

    auto *tube_new = (new Tube(currentObjectID, UUIDs, nodes, radius, color, triangle_vertices, radial_subdivisions, "", this));

    for (uint p: UUIDs) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = tube_new;
    currentObjectID++;

    uint objID = currentObjectID - 1;
    tube_new->object_origin = getObjectCenter(objID);

    return objID;
}

uint Context::addTubeObject(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char *texturefile) {
    size_t node_count = nodes.size();
    std::vector<float> textureuv_ufrac(node_count);
    for (int i = 0; i < node_count; i++) {
        textureuv_ufrac.at(i) = float(i) / float(node_count - 1);
    }

    return addTubeObject(radial_subdivisions, nodes, radius, texturefile, textureuv_ufrac);
}

uint Context::addTubeObject(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char *texturefile, const std::vector<float> &textureuv_ufrac) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addTubeObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addTubeObject): Texture file " + std::string(texturefile) + " does not exist.");
    }

    const uint node_count = nodes.size();

    if (node_count == 0) {
        helios_runtime_error("ERROR (Context::addTubeObject): Node and radius arrays are empty.");
    } else if (node_count != radius.size()) {
        helios_runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `radius' arguments must agree.");
    } else if (node_count != textureuv_ufrac.size()) {
        helios_runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `textureuv_ufrac' arguments must agree.");
    }

    // Clamp very small radii to avoid creating degenerate triangles
    const float min_radius_threshold = 1e-5f;
    std::vector<float> radius_clamped = radius;
    for (int i = 0; i < node_count; i++) {
        if (radius_clamped[i] < min_radius_threshold && radius_clamped[i] >= 0) {
            radius_clamped[i] = min_radius_threshold;
        }
    }

    vec3 axial_vector;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3>> triangle_vertices;
    resize_vector(triangle_vertices, radial_subdivisions + 1, node_count);
    std::vector<std::vector<vec2>> uv;
    resize_vector(uv, radial_subdivisions + 1, node_count);

    // Initialize trigonometric factors for circle points
    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    vec3 initial_radial(1.0f, 0.0f, 0.0f);
    vec3 previous_axial_vector;
    vec3 previous_radial_dir;

    for (int i = 0; i < node_count; i++) { // Looping over tube segments
        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTubeObject): Radius of tube must be positive.");
        }

        if (i == 0) {
            axial_vector = nodes[i + 1] - nodes[i];
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }
            if (fabs(axial_vector * initial_radial) > 0.95f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
            }
            // Also handle nearly vertical axes
            if (fabs(axial_vector.z) > 0.95f) {
                initial_radial = vec3(1.0f, 0.0f, 0.0f); // Use horizontal radial for vertical axes
            }
            previous_radial_dir = cross(axial_vector, initial_radial).normalize();
        } else {
            if (i == node_count - 1) {
                axial_vector = nodes[i] - nodes[i - 1];
            } else {
                axial_vector = 0.5f * ((nodes[i] - nodes[i - 1]) + (nodes[i + 1] - nodes[i]));
            }
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }

            // Calculate radial direction using parallel transport
            vec3 rotation_axis = cross(previous_axial_vector, axial_vector);
            if (rotation_axis.magnitude() > 1e-5) {
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
            } else {
                // Vectors are nearly parallel, use robust fallback
                vec3 fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                if (fabs(axial_vector * fallback_radial) > 0.95f) {
                    fallback_radial = vec3(0.0f, 1.0f, 0.0f);
                }
                if (fabs(axial_vector.z) > 0.95f) {
                    fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                }
                previous_radial_dir = cross(axial_vector, fallback_radial).normalize();
            }
        }

        previous_axial_vector = axial_vector;

        vec3 radial_dir = previous_radial_dir;
        vec3 orthogonal_dir = cross(radial_dir, axial_vector);
        orthogonal_dir.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal = cfact[j] * radius_clamped[i] * radial_dir + sfact[j] * radius_clamped[i] * orthogonal_dir;
            triangle_vertices[i][j] = nodes[i] + normal;

            uv[i][j].x = textureuv_ufrac[i];
            uv[i][j].y = float(j) / float(radial_subdivisions);
        }
    }

    std::vector<uint> UUIDs;
    UUIDs.reserve(2 * (node_count - 1) * radial_subdivisions); // Reserve expected capacity
    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    for (int j = 0; j < radial_subdivisions; j++) {
        for (int i = 0; i < node_count - 1; i++) {
            v0 = triangle_vertices[i][j];
            v1 = triangle_vertices[i + 1][j + 1];
            v2 = triangle_vertices[i][j + 1];

            uv0 = uv[i][j];
            uv1 = uv[i + 1][j + 1];
            uv2 = uv[i][j + 1];

            uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
            if (getPrimitiveArea(triangle_uuid) > 0) {
                UUIDs.push_back(triangle_uuid);
            } else {
                deletePrimitive(triangle_uuid);
            }

            v0 = triangle_vertices[i][j];
            v1 = triangle_vertices[i + 1][j];
            v2 = triangle_vertices[i + 1][j + 1];

            uv0 = uv[i][j];
            uv1 = uv[i + 1][j];
            uv2 = uv[i + 1][j + 1];

            uint triangle_uuid2 = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
            if (getPrimitiveArea(triangle_uuid2) > 0) {
                UUIDs.push_back(triangle_uuid2);
            } else {
                deletePrimitive(triangle_uuid2);
            }
        }
    }

    std::vector<RGBcolor> colors(nodes.size());

    auto *tube_new = (new Tube(currentObjectID, UUIDs, nodes, radius, colors, triangle_vertices, radial_subdivisions, texturefile, this));

    for (uint p: UUIDs) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = tube_new;
    currentObjectID++;

    uint objID = currentObjectID - 1;
    tube_new->object_origin = getObjectCenter(objID);

    return objID;
}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv) {
    RGBcolor color(0.f, 0.75f, 0.f); // Default color is green

    return addBoxObject(center, size, subdiv, color, false);
}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color) {
    return addBoxObject(center, size, subdiv, color, false);
}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const char *texturefile) {
    return addBoxObject(center, size, subdiv, texturefile, false);
}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals) {
    if (size.x <= 0 || size.y <= 0 || size.z <= 0) {
        helios_runtime_error("ERROR (Context::addBoxObject): Size of box must be positive.");
    } else if (subdiv.x < 1 || subdiv.y < 1 || subdiv.z < 1) {
        helios_runtime_error("ERROR (Context::addBoxObject): Number of box subdivisions must be positive.");
    }

    std::vector<uint> UUID;
    UUID.reserve(2 * (subdiv.z * (subdiv.x + subdiv.y) + subdiv.x * subdiv.y));

    vec3 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);
    subsize.z = size.z / float(subdiv.z);

    vec3 subcenter;
    std::vector<uint> U, U_copy;

    if (reverse_normals) { // normals point inward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 1.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 0.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { // normals point outward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 0.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 1.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());
    }

    auto *box_new = (new Box(currentObjectID, UUID, subdiv, "", this));

    float T[16], transform[16];
    box_new->getTransformationMatrix(transform);

    makeScaleMatrix(size, T);
    matmult(T, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    box_new->setTransformationMatrix(transform);

    box_new->setColor(color);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    box_new->object_origin = center;

    objects[currentObjectID] = box_new;
    currentObjectID++;
    return currentObjectID - 1;
}

uint Context::addBoxObject(vec3 center, const vec3 &size, const int3 &subdiv, const char *texturefile, bool reverse_normals) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addBoxObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addBoxObject): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;

    vec3 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);
    subsize.z = size.z / float(subdiv.z);

    vec3 subcenter;
    std::vector<uint> U, U_copy;

    if (reverse_normals) { // normals point inward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { // normals point outward

        // x-z faces (vertical)

        // right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        // front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        // top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());
    }

    auto *box_new = (new Box(currentObjectID, UUID, subdiv, texturefile, this));

    float T[16], transform[16];
    box_new->getTransformationMatrix(transform);

    makeScaleMatrix(size, T);
    matmult(T, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    box_new->setTransformationMatrix(transform);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    box_new->object_origin = center;

    objects[currentObjectID] = box_new;
    currentObjectID++;
    return currentObjectID - 1;
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size) {
    return addDiskObject(make_int2(Ndivs, 1), center, size, make_SphericalCoord(0, 0), make_RGBAcolor(1, 0, 0, 1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation) {
    return addDiskObject(make_int2(Ndivs, 1), center, size, rotation, make_RGBAcolor(1, 0, 0, 1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color) {
    return addDiskObject(make_int2(Ndivs, 1), center, size, rotation, make_RGBAcolor(color, 1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color) {
    return addDiskObject(make_int2(Ndivs, 1), center, size, rotation, color);
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file) {
    return addDiskObject(make_int2(Ndivs, 1), center, size, rotation, texture_file);
}

uint Context::addDiskObject(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color) {
    return addDiskObject(Ndivs, center, size, rotation, make_RGBAcolor(color, 1));
}

uint Context::addDiskObject(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color) {
    std::vector<uint> UUID(Ndivs.x + Ndivs.x * (Ndivs.y - 1) * 2);

    int i = 0;
    for (int r = 0; r < Ndivs.y; r++) {
        for (int t = 0; t < Ndivs.x; t++) {
            float dtheta = 2.f * PI_F / float(Ndivs.x);
            float theta = dtheta * float(t);
            float theta_plus = dtheta * float(t + 1);

            float rx = size.x / float(Ndivs.y) * float(r);
            float ry = size.y / float(Ndivs.y) * float(r);

            float rx_plus = size.x / float(Ndivs.y) * float(r + 1);
            float ry_plus = size.y / float(Ndivs.y) * float(r + 1);

            if (r == 0) {
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            } else {
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), color);
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
        }
    }

    auto *disk_new = (new Disk(currentObjectID, UUID, Ndivs, "", this));

    float T[16], transform[16];
    disk_new->getTransformationMatrix(transform);

    makeScaleMatrix(make_vec3(size.x, size.y, 1.f), T);
    matmult(T, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    disk_new->setTransformationMatrix(transform);

    disk_new->setColor(color);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    disk_new->object_origin = center;

    objects[currentObjectID] = disk_new;
    currentObjectID++;
    return currentObjectID - 1;
}

uint Context::addDiskObject(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addDiskObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addDiskObject): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;
    UUID.reserve(Ndivs.x + Ndivs.x * (Ndivs.y - 1) * 2); // Reserve expected capacity
    for (int r = 0; r < Ndivs.y; r++) {
        for (int t = 0; t < Ndivs.x; t++) {
            float dtheta = 2.f * PI_F / float(Ndivs.x);
            float theta = dtheta * float(t);
            float theta_plus = dtheta * float(t + 1);

            float rx = size.x / float(Ndivs.y) * float(r);
            float ry = size.y / float(Ndivs.y) * float(r);
            float rx_plus = size.x / float(Ndivs.y) * float(r + 1);
            float ry_plus = size.y / float(Ndivs.y) * float(r + 1);

            if (r == 0) {
                uint triangle_uuid = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile, make_vec2(0.5, 0.5),
                                                 make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                                 make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
                if (getPrimitiveArea(triangle_uuid) > 0) {
                    UUID.push_back(triangle_uuid);
                } else {
                    deletePrimitive(triangle_uuid);
                    continue;
                }
            } else {
                uint triangle_uuid1 = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), texturefile,
                                                  make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx / size.x), 0.5f * (1.f + sinf(theta) * ry / size.y)),
                                                  make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)));
                if (getPrimitiveArea(triangle_uuid1) > 0) {
                    UUID.push_back(triangle_uuid1);
                } else {
                    deletePrimitive(triangle_uuid1);
                }

                uint triangle_uuid2 =
                        addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile,
                                    make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                    make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
                if (getPrimitiveArea(triangle_uuid2) > 0) {
                    UUID.push_back(triangle_uuid2);
                } else {
                    deletePrimitive(triangle_uuid2);
                    continue;
                }
            }
            // Apply transformations to all valid triangles added in this iteration
            size_t start_idx = UUID.size() - (r == 0 ? 1 : 2);
            for (size_t uuid_idx = start_idx; uuid_idx < UUID.size(); uuid_idx++) {
                getPrimitivePointer_private(UUID.at(uuid_idx))->rotate(rotation.elevation, "y");
                getPrimitivePointer_private(UUID.at(uuid_idx))->rotate(rotation.azimuth, "z");
                getPrimitivePointer_private(UUID.at(uuid_idx))->translate(center);
            }
        }
    }

    auto *disk_new = (new Disk(currentObjectID, UUID, Ndivs, texturefile, this));

    float T[16], transform[16];
    disk_new->getTransformationMatrix(transform);

    makeScaleMatrix(make_vec3(size.x, size.y, 1.f), T);
    matmult(T, transform, transform);

    makeTranslationMatrix(center, T);
    matmult(T, transform, transform);
    disk_new->setTransformationMatrix(transform);

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    disk_new->object_origin = center;

    objects[currentObjectID] = disk_new;
    currentObjectID++;
    return currentObjectID - 1;
}

uint Context::addPolymeshObject(const std::vector<uint> &UUIDs) {
    if (UUIDs.empty()) {
        helios_runtime_error("ERROR (Context::addPolymeshObject): UUIDs array is empty. Cannot create polymesh object.");
    } else if (!doesPrimitiveExist(UUIDs)) {
        helios_runtime_error("ERROR (Context::addPolymeshObject): One or more of the provided UUIDs does not exist. Cannot create polymesh object.");
    }

    // Check whether primitives already belong to another object
    std::vector<uint> UUIDs_polymesh;
    UUIDs_polymesh.reserve(UUIDs.size());
    size_t skipped_UUIDs = 0;
    for (uint UUID: UUIDs) {
        if (getPrimitivePointer_private(UUID)->getParentObjectID() != 0) {
            skipped_UUIDs++;
        } else {
            UUIDs_polymesh.push_back(UUID);
        }
    }
    if (skipped_UUIDs > 0) {
        std::cerr << "WARNING (Context::addPolymeshObject): " << skipped_UUIDs << " primitives were not added to polymesh object because they already belong to another object." << std::endl;
    }

    auto *polymesh_new = (new Polymesh(currentObjectID, UUIDs_polymesh, "", this));

    float T[16], transform[16];
    polymesh_new->getTransformationMatrix(transform);

    makeTranslationMatrix(getPrimitivePointer_private(UUIDs_polymesh.front())->getVertices().front(), T);
    matmult(T, transform, transform);
    polymesh_new->setTransformationMatrix(transform);

    for (uint UUID: UUIDs_polymesh) {
        getPrimitivePointer_private(UUID)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = polymesh_new;
    currentObjectID++;

    uint objID = currentObjectID - 1;
    polymesh_new->object_origin = getObjectCenter(objID);

    return objID;
}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1) {
    RGBcolor color(0.f, 0.75f, 0.f); // Default color is green
    return addConeObject(Ndivs, node0, node1, radius0, radius1, color);
}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const RGBcolor &color) {
    const std::vector nodes{node0, node1};
    const std::vector radii{radius0, radius1};

    vec3 convec;
    std::vector<float> cfact(Ndivs + 1);
    std::vector<float> sfact(Ndivs + 1);
    std::vector<std::vector<vec3>> xyz(Ndivs + 1);
    std::vector<std::vector<vec3>> normal(Ndivs + 1);

    for (uint j = 0; j < Ndivs + 1; j++) {
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
    }
    vec3 nvec(0.1817f, 0.6198f, 0.7634f); // random vector to get things going

    for (int j = 0; j < Ndivs + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(Ndivs));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(Ndivs));
    }

    for (int i = 0; i < 2; i++) {
        vec3 vec;
        // looping over cone segments

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        }

        if (vec.magnitude() < 1e-6f) {
            vec = make_vec3(0, 0, 1);
        }
        float norm;
        convec = cross(nvec, vec);
        norm = convec.magnitude();
        if (norm < 1e-6f) {
            convec = cross(vec, fabs(vec.x) < 0.9f ? make_vec3(1, 0, 0) : make_vec3(0, 1, 0));
            norm = std::max(convec.magnitude(), 1e-6f);
        }
        convec = convec / norm;
        nvec = cross(vec, convec);
        norm = nvec.magnitude();
        if (norm < 1e-6f) {
            nvec = cross(convec, vec);
            norm = std::max(nvec.magnitude(), 1e-6f);
        }
        nvec = nvec / norm;

        for (int j = 0; j < Ndivs + 1; j++) {
            normal[j][i].x = cfact[j] * radii[i] * nvec.x + sfact[j] * radii[i] * convec.x;
            normal[j][i].y = cfact[j] * radii[i] * nvec.y + sfact[j] * radii[i] * convec.y;
            normal[j][i].z = cfact[j] * radii[i] * nvec.z + sfact[j] * radii[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal[j][i].x;
            xyz[j][i].y = nodes[i].y + normal[j][i].y;
            xyz[j][i].z = nodes[i].z + normal[j][i].z;

            normal[j][i] = normal[j][i] / radii[i];
        }
    }

    vec3 v0, v1, v2;
    std::vector<uint> UUID(2 * Ndivs);

    int i = 0;
    for (int j = 0; j < Ndivs; j++) {
        v0 = xyz[j][0];
        v1 = xyz[j + 1][1];
        v2 = xyz[j + 1][0];

        UUID.at(i) = addTriangle(v0, v1, v2, color);

        v0 = xyz[j][0];
        v1 = xyz[j][1];
        v2 = xyz[j + 1][1];

        UUID.at(i + 1) = addTriangle(v0, v1, v2, color);

        i += 2;
    }

    auto *cone_new = (new Cone(currentObjectID, UUID, node0, node1, radius0, radius1, Ndivs, "", this));

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    cone_new->setColor(color);

    objects[currentObjectID] = cone_new;
    currentObjectID++;

    uint objID = currentObjectID - 1;
    cone_new->object_origin = getObjectCenter(objID);

    return objID;
}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addConeObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addConeObject): Texture file " + std::string(texturefile) + " does not exist.");
    }

    const std::vector<helios::vec3> nodes{node0, node1};
    const std::vector<float> radii{radius0, radius1};

    vec3 convec;
    std::vector<float> cfact(Ndivs + 1);
    std::vector<float> sfact(Ndivs + 1);
    std::vector<std::vector<vec3>> xyz, normal;
    std::vector<std::vector<vec2>> uv;
    xyz.resize(Ndivs + 1);
    normal.resize(Ndivs + 1);
    uv.resize(Ndivs + 1);
    for (uint j = 0; j < Ndivs + 1; j++) {
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
        uv.at(j).resize(2);
    }
    vec3 nvec(0.f, 1.f, 0.f);

    for (int j = 0; j < Ndivs + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(Ndivs));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(Ndivs));
    }

    for (int i = 0; i < 2; i++) {
        vec3 vec;
        // looping over cone segments

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        }

        if (vec.magnitude() < 1e-6f) {
            vec = make_vec3(0, 0, 1);
        }
        float norm;
        convec = cross(nvec, vec);
        norm = convec.magnitude();
        if (norm < 1e-6f) {
            convec = cross(vec, fabs(vec.x) < 0.9f ? make_vec3(1, 0, 0) : make_vec3(0, 1, 0));
            norm = std::max(convec.magnitude(), 1e-6f);
        }
        convec = convec / norm;
        nvec = cross(vec, convec);
        norm = nvec.magnitude();
        if (norm < 1e-6f) {
            nvec = cross(convec, vec);
            norm = std::max(nvec.magnitude(), 1e-6f);
        }
        nvec = nvec / norm;

        for (int j = 0; j < Ndivs + 1; j++) {
            normal[j][i].x = cfact[j] * radii[i] * nvec.x + sfact[j] * radii[i] * convec.x;
            normal[j][i].y = cfact[j] * radii[i] * nvec.y + sfact[j] * radii[i] * convec.y;
            normal[j][i].z = cfact[j] * radii[i] * nvec.z + sfact[j] * radii[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal[j][i].x;
            xyz[j][i].y = nodes[i].y + normal[j][i].y;
            xyz[j][i].z = nodes[i].z + normal[j][i].z;

            uv[j][i].x = float(i) / float(2 - 1);
            uv[j][i].y = float(j) / float(Ndivs);

            normal[j][i] = normal[j][i] / radii[i];
        }
    }

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    std::vector<uint> UUID;

    for (int i = 0; i < 2 - 1; i++) {
        for (int j = 0; j < Ndivs; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            uv0 = uv[j][i];
            uv1 = uv[j + 1][i + 1];
            uv2 = uv[j + 1][i];

            if ((v1 - v0).magnitude() > 1e-6 && (v2 - v0).magnitude() > 1e-6 && (v2 - v1).magnitude() > 1e-6) {
                uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
                if (getPrimitiveArea(triangle_uuid) > 0) {
                    UUID.push_back(triangle_uuid);
                } else {
                    deletePrimitive(triangle_uuid);
                }
            }

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            uv0 = uv[j][i];
            uv1 = uv[j][i + 1];
            uv2 = uv[j + 1][i + 1];

            if ((v1 - v0).magnitude() > 1e-6 && (v2 - v0).magnitude() > 1e-6 && (v2 - v1).magnitude() > 1e-6) {
                uint triangle_uuid = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);
                if (getPrimitiveArea(triangle_uuid) > 0) {
                    UUID.push_back(triangle_uuid);
                } else {
                    deletePrimitive(triangle_uuid);
                }
            }
        }
    }

    auto *cone_new = (new Cone(currentObjectID, UUID, node0, node1, radius0, radius1, Ndivs, texturefile, this));

    for (uint p: UUID) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = cone_new;
    currentObjectID++;

    uint objID = currentObjectID - 1;
    cone_new->object_origin = getObjectCenter(objID);

    return objID;
}

// ============== COMPOUND OBJECT CLASS METHOD DEFINITIONS ==============

CompoundObject::~CompoundObject() = default;

uint CompoundObject::getObjectID() const {
    return OID;
}

helios::ObjectType CompoundObject::getObjectType() const {
    return type;
}

uint CompoundObject::getPrimitiveCount() const {
    return UUIDs.size();
}


std::vector<uint> CompoundObject::getPrimitiveUUIDs() const {
    return UUIDs;
}

bool CompoundObject::doesObjectContainPrimitive(uint UUID) {
    return find(UUIDs.begin(), UUIDs.end(), UUID) != UUIDs.end();
}

helios::vec3 CompoundObject::getObjectCenter() const {
    vec2 xbounds, ybounds, zbounds;

    const std::vector<uint> &U = getPrimitiveUUIDs();

    context->getDomainBoundingBox(U, xbounds, ybounds, zbounds);

    vec3 origin;

    origin.x = 0.5f * (xbounds.x + xbounds.y);
    origin.y = 0.5f * (ybounds.x + ybounds.y);
    origin.z = 0.5f * (zbounds.x + zbounds.y);

    return origin;
}

float CompoundObject::getArea() const {
    float area = 0.f;

    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            area += context->getPrimitiveArea(UUID);
        }
    }

    return area;
}

void CompoundObject::setColor(const helios::RGBcolor &a_color) {
    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->setPrimitiveColor(UUID, a_color);
        }
    }
}

void CompoundObject::setColor(const helios::RGBAcolor &a_color) {
    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->setPrimitiveColor(UUID, a_color);
        }
    }
}

void CompoundObject::overrideTextureColor() {
    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->overridePrimitiveTextureColor(UUID);
        }
    }
}

void CompoundObject::useTextureColor() {
    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->usePrimitiveTextureColor(UUID);
        }
    }
}

bool CompoundObject::hasTexture() const {
    if (getTextureFile().empty()) {
        return false;
    } else {
        return true;
    }
}

std::string CompoundObject::getTextureFile() const {
    return texturefile;
}

void CompoundObject::translate(const helios::vec3 &shift) {
    if (shift == nullorigin) {
        return;
    }

    float T[16], T_prim[16];
    makeTranslationMatrix(shift, T);

    matmult(T, transform, transform);

    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->getPrimitiveTransformationMatrix(UUID, T_prim);
            matmult(T, T_prim, T_prim);
            context->setPrimitiveTransformationMatrix(UUID, T_prim);
        }
    }
}

void CompoundObject::rotate(float rotation_radians, const char *rotation_axis_xyz_string) {
    if (rotation_radians == 0) {
        return;
    }

    if (strcmp(rotation_axis_xyz_string, "z") == 0) {
        float Rz[16], Rz_prim[16];
        makeRotationMatrix(-rotation_radians, "z", Rz);
        matmult(Rz, transform, transform);

        for (uint UUID: UUIDs) {
            if (context->doesPrimitiveExist(UUID)) {
                context->getPrimitiveTransformationMatrix(UUID, Rz_prim);
                matmult(Rz, Rz_prim, Rz_prim);
                context->setPrimitiveTransformationMatrix(UUID, Rz_prim);
            }
        }
    } else if (strcmp(rotation_axis_xyz_string, "y") == 0) {
        float Ry[16], Ry_prim[16];
        makeRotationMatrix(rotation_radians, "y", Ry);
        matmult(Ry, transform, transform);
        for (uint UUID: UUIDs) {
            if (context->doesPrimitiveExist(UUID)) {
                context->getPrimitiveTransformationMatrix(UUID, Ry_prim);
                matmult(Ry, Ry_prim, Ry_prim);
                context->setPrimitiveTransformationMatrix(UUID, Ry_prim);
            }
        }
    } else if (strcmp(rotation_axis_xyz_string, "x") == 0) {
        float Rx[16], Rx_prim[16];
        makeRotationMatrix(rotation_radians, "x", Rx);
        matmult(Rx, transform, transform);
        for (uint UUID: UUIDs) {
            if (context->doesPrimitiveExist(UUID)) {
                context->getPrimitiveTransformationMatrix(UUID, Rx_prim);
                matmult(Rx, Rx_prim, Rx_prim);
                context->setPrimitiveTransformationMatrix(UUID, Rx_prim);
            }
        }
    } else {
        helios_runtime_error("ERROR (CompoundObject::rotate): Rotation axis should be one of x, y, or z.");
    }
}

void CompoundObject::rotate(float rotation_radians, const helios::vec3 &rotation_axis_vector) {
    if (rotation_radians == 0) {
        return;
    }

    float R[16], R_prim[16];
    makeRotationMatrix(rotation_radians, rotation_axis_vector, R);
    matmult(R, transform, transform);

    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->getPrimitiveTransformationMatrix(UUID, R_prim);
            matmult(R, R_prim, R_prim);
            context->setPrimitiveTransformationMatrix(UUID, R_prim);
        }
    }
}

void CompoundObject::rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector) {
    if (rotation_radians == 0) {
        return;
    }

    float R[16], R_prim[16];
    makeRotationMatrix(rotation_radians, origin, rotation_axis_vector, R);
    matmult(R, transform, transform);

    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->getPrimitiveTransformationMatrix(UUID, R_prim);
            matmult(R, R_prim, R_prim);
            context->setPrimitiveTransformationMatrix(UUID, R_prim);
        }
    }
}

void CompoundObject::scale(const helios::vec3 &scale) {
    scaleAboutPoint(scale, nullorigin);
}

void CompoundObject::scaleAboutCenter(const helios::vec3 &scale) {
    scaleAboutPoint(scale, getObjectCenter());
}

void CompoundObject::scaleAboutPoint(const helios::vec3 &scale, const helios::vec3 &point) {
    if (scale.x == 1.f && scale.y == 1.f && scale.z == 1.f) {
        return;
    }

    float T[16], T_prim[16];
    makeScaleMatrix(scale, point, T);
    matmult(T, transform, transform);

    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->getPrimitiveTransformationMatrix(UUID, T_prim);
            matmult(T, T_prim, T_prim);
            context->setPrimitiveTransformationMatrix(UUID, T_prim);
        }
    }
}

void CompoundObject::getTransformationMatrix(float (&T)[16]) const {
    for (int i = 0; i < 16; i++) {
        T[i] = transform[i];
    }
}

void CompoundObject::setTransformationMatrix(float (&T)[16]) {
    for (int i = 0; i < 16; i++) {
        transform[i] = T[i];
    }
}

void CompoundObject::setPrimitiveUUIDs(const std::vector<uint> &a_UUIDs) {
    UUIDs = a_UUIDs;
}

void CompoundObject::deleteChildPrimitive(uint UUID) {
    auto it = find(UUIDs.begin(), UUIDs.end(), UUID);
    if (it != UUIDs.end()) {
        std::iter_swap(it, UUIDs.end() - 1);
        UUIDs.pop_back();
        primitivesarecomplete = false;
    }
}

void CompoundObject::deleteChildPrimitive(const std::vector<uint> &a_UUIDs) {
    for (uint UUID: a_UUIDs) {
        deleteChildPrimitive(UUID);
    }
}

bool CompoundObject::arePrimitivesComplete() const {
    return primitivesarecomplete;
}

// ============== TILE CLASS METHOD DEFINITIONS ==============

Tile::Tile(uint a_OID, const std::vector<uint> &a_UUIDs, const int2 &a_subdiv, const char *a_texturefile, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_TILE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
    context = a_context;
}

Tile *Context::getTileObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return dynamic_cast<Tile *>(objects.at(ObjID));
}

helios::vec2 Tile::getSize() const {
    const std::vector<vec3> &vertices = getVertices();
    float l = (vertices.at(1) - vertices.at(0)).magnitude();
    float w = (vertices.at(3) - vertices.at(0)).magnitude();
    return make_vec2(l, w);
}

vec3 Tile::getCenter() const {
    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;
}


helios::int2 Tile::getSubdivisionCount() const {
    return subdiv;
}

void Tile::setSubdivisionCount(const helios::int2 &a_subdiv) {
    subdiv = a_subdiv;
}


std::vector<helios::vec3> Tile::getVertices() const {
    std::vector<helios::vec3> vertices;
    vertices.resize(4);

    // subcenter = make_vec3(-0.5*size.x+(float(i)+0.5)*subsize.x,-0.5*size.y+(float(j)+0.5)*subsize.y,0);
    // Y[0] = make_vec3( -0.5f, -0.5f, 0.f);
    // Y[1] = make_vec3( 0.5f, -0.5f, 0.f);
    // Y[2] = make_vec3( 0.5f, 0.5f, 0.f);
    // Y[3] = make_vec3( -0.5f, 0.5f, 0.f);


    vec3 Y[4];
    Y[0] = make_vec3(-0.5f, -0.5f, 0.f);
    Y[1] = make_vec3(0.5f, -0.5f, 0.f);
    Y[2] = make_vec3(0.5f, 0.5f, 0.f);
    Y[3] = make_vec3(-0.5f, 0.5f, 0.f);

    for (int i = 0; i < 4; i++) {
        vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
        vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
        vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
    }

    return vertices;
}

vec3 Tile::getNormal() const {
    return context->getPrimitiveNormal(UUIDs.front());
}

std::vector<helios::vec2> Tile::getTextureUV() const {
    return {make_vec2(0, 0), make_vec2(1, 0), make_vec2(1, 1), make_vec2(0, 1)};
}

// ============== SPHERE CLASS METHOD DEFINITIONS ==============

Sphere::Sphere(uint a_OID, const std::vector<uint> &a_UUIDs, uint a_subdiv, const char *a_texturefile, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_SPHERE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
    context = a_context;
}

Sphere *Context::getSphereObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return dynamic_cast<Sphere *>(objects.at(ObjID));
}

helios::vec3 Sphere::getRadius() const {
    vec3 n0(0, 0, 0);
    vec3 nx(1, 0, 0);
    vec3 ny(0, 1, 0);
    vec3 nz(0, 0, 1);
    vec3 n0_T, nx_T, ny_T, nz_T;

    vecmult(transform, n0, n0_T);
    vecmult(transform, nx, nx_T);
    vecmult(transform, ny, ny_T);
    vecmult(transform, nz, nz_T);

    vec3 radii;
    radii.x = (nx_T - n0_T).magnitude();
    radii.y = (ny_T - n0_T).magnitude();
    radii.z = (nz_T - n0_T).magnitude();

    return radii;
}

vec3 Sphere::getCenter() const {
    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;
}

uint Sphere::getSubdivisionCount() const {
    return subdiv;
}

void Sphere::setSubdivisionCount(uint a_subdiv) {
    subdiv = a_subdiv;
}

float Sphere::getVolume() const {
    const vec3 &radii = getRadius();
    return 4.f / 3.f * PI_F * radii.x * radii.y * radii.z;
}

// ============== TUBE CLASS METHOD DEFINITIONS ==============

Tube::Tube(uint a_OID, const std::vector<uint> &a_UUIDs, const std::vector<vec3> &a_nodes, const std::vector<float> &a_radius, const std::vector<helios::RGBcolor> &a_colors, const std::vector<std::vector<helios::vec3>> &a_triangle_vertices,
           uint a_subdiv, const char *a_texturefile, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_TUBE;
    UUIDs = a_UUIDs;
    nodes = a_nodes;
    radius = a_radius;
    colors = a_colors;
    triangle_vertices = a_triangle_vertices;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
    context = a_context;
}

Tube *Context::getTubeObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return dynamic_cast<Tube *>(objects.at(ObjID));
}

std::vector<helios::vec3> Tube::getNodes() const {
    std::vector<vec3> nodes_T;
    nodes_T.resize(nodes.size());

    for (uint i = 0; i < nodes.size(); i++) {
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    return nodes_T;
}

uint Tube::getNodeCount() const {
    return scast<uint>(nodes.size());
}

std::vector<float> Tube::getNodeRadii() const {
    std::vector<float> radius_T;
    radius_T.resize(radius.size());
    for (int i = 0; i < radius.size(); i++) {
        vec3 n0(0, 0, 0), nx(radius.at(i), 0, 0);
        vec3 n0_T, nx_T;

        vecmult(transform, n0, n0_T);
        vecmult(transform, nx, nx_T);

        radius_T.at(i) = (nx_T - n0_T).magnitude();
    }
    return radius_T;
}

std::vector<helios::RGBcolor> Tube::getNodeColors() const {
    return colors;
}

std::vector<std::vector<helios::vec3>> Tube::getTriangleVertices() const {
    return triangle_vertices;
}

uint Tube::getSubdivisionCount() const {
    return subdiv;
}

float Tube::getLength() const {
    float length = 0.f;
    for (uint i = 0; i < nodes.size() - 1; i++) {
        length += (nodes.at(i + 1) - nodes.at(i)).magnitude();
    }
    return length;
}

float Tube::getVolume() const {
    const std::vector<float> &radii = getNodeRadii();
    float volume = 0.f;
    for (uint i = 0; i < radii.size() - 1; i++) {
        float segment_length = (nodes.at(i + 1) - nodes.at(i)).magnitude();
        float r0 = radii.at(i);
        float r1 = radii.at(i + 1);
        volume += PI_F * segment_length / 3.f * (r0 * r0 + r0 * r1 + r1 * r1);
    }

    return volume;
}

float Tube::getSegmentVolume(uint segment_index) const {
    if (segment_index >= nodes.size() - 1) {
        helios_runtime_error("ERROR (Tube::getSegmentVolume): Segment index out of bounds.");
    }

    float segment_length = (nodes.at(segment_index + 1) - nodes.at(segment_index)).magnitude();
    float r0 = radius.at(segment_index);
    float r1 = radius.at(segment_index + 1);
    float volume = PI_F * segment_length / 3.f * (r0 * r0 + r0 * r1 + r1 * r1);

    return volume;
}

void Tube::appendTubeSegment(const helios::vec3 &node_position, float node_radius, const helios::RGBcolor &node_color) {
    //\todo This is a computationally inefficient method for appending the tube, but it ensures that there is no twisting of the tube relative to the previous tube segments.

    if (node_radius < 0) {
        helios_runtime_error("ERROR (Tube::appendTubeSegment): Node radius must be positive.");
    }
    node_radius = std::max((float) 1e-5, node_radius);

    uint radial_subdivisions = subdiv;

    vec3 axial_vector;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);

    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    triangle_vertices.resize(triangle_vertices.size() + 1);
    triangle_vertices.back().resize(radial_subdivisions + 1);

    nodes.push_back(node_position);
    radius.push_back(node_radius);
    colors.push_back(node_color);

    int node_count = nodes.size();

    vec3 initial_radial(1.0f, 0.0f, 0.0f);
    vec3 previous_axial_vector;
    vec3 previous_radial_dir;

    for (int i = 0; i < node_count; i++) { // Looping over tube segments
        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTubeObject): Radius of tube must be positive.");
        }

        if (i == 0) {
            axial_vector = nodes[i + 1] - nodes[i];
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }
            if (fabs(axial_vector * initial_radial) > 0.95f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
            }
            // Also handle nearly vertical axes
            if (fabs(axial_vector.z) > 0.95f) {
                initial_radial = vec3(1.0f, 0.0f, 0.0f); // Use horizontal radial for vertical axes
            }
            previous_radial_dir = cross(axial_vector, initial_radial).normalize();
        } else {
            if (i == node_count - 1) {
                axial_vector = nodes[i] - nodes[i - 1];
            } else {
                axial_vector = 0.5f * ((nodes[i] - nodes[i - 1]) + (nodes[i + 1] - nodes[i]));
            }
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }

            // Calculate radial direction using parallel transport
            vec3 rotation_axis = cross(previous_axial_vector, axial_vector);
            if (rotation_axis.magnitude() > 1e-5) { // More conservative threshold
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
            } else {
                // Vectors are nearly parallel, use robust fallback
                vec3 fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                if (fabs(axial_vector * fallback_radial) > 0.95f) {
                    fallback_radial = vec3(0.0f, 1.0f, 0.0f);
                }
                if (fabs(axial_vector.z) > 0.95f) {
                    fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                }
                previous_radial_dir = cross(axial_vector, fallback_radial).normalize();
            }
            //            else {
            //                // Handle the case of nearly parallel vectors
            //                // Ensure previous_radial_dir remains orthogonal to axial_vector
            //                previous_radial_dir = cross(axial_vector, previous_radial_dir);
            //                if (previous_radial_dir.magnitude() < 1e-6) {
            //                    // If still degenerate, choose another orthogonal direction
            //                    previous_radial_dir = cross(axial_vector, vec3(1.0f, 0.0f, 0.0f));
            //                }
            //                previous_radial_dir.normalize();
            //            }
        }

        previous_axial_vector = axial_vector;

        vec3 radial_dir = previous_radial_dir;
        vec3 orthogonal_dir = cross(radial_dir, axial_vector);
        orthogonal_dir.normalize();

        if (i < node_count - 2) {
            continue;
        }

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal = cfact[j] * radius[i] * radial_dir + sfact[j] * radius[i] * orthogonal_dir;
            triangle_vertices[i][j] = nodes[i] + normal;
        }
    }

    // add triangles for new segment

    for (int j = 0; j < radial_subdivisions; j++) {
        vec3 v0 = triangle_vertices.at(1).at(j);
        vec3 v1 = triangle_vertices.at(1 + 1).at(j + 1);
        vec3 v2 = triangle_vertices.at(1).at(j + 1);

        UUIDs.push_back(context->addTriangle(v0, v1, v2, node_color));

        v0 = triangle_vertices.at(1).at(j);
        v1 = triangle_vertices.at(1 + 1).at(j);
        v2 = triangle_vertices.at(1 + 1).at(j + 1);

        UUIDs.push_back(context->addTriangle(v0, v1, v2, node_color));
    }

    for (uint p: UUIDs) {
        context->setPrimitiveParentObjectID(p, this->OID);
    }

    updateTriangleVertices();
}

void Tube::appendTubeSegment(const helios::vec3 &node_position, float node_radius, const char *texturefile, const helios::vec2 &textureuv_ufrac) {
    //\todo This is a computationally inefficient method for appending the tube, but it ensures that there is no twisting of the tube relative to the previous tube segments.

    if (node_radius < 0) {
        helios_runtime_error("ERROR (Tube::appendTubeSegment): Node radius must be positive.");
    } else if (textureuv_ufrac.x < 0 || textureuv_ufrac.y < 0 || textureuv_ufrac.x > 1 || textureuv_ufrac.y > 1) {
        helios_runtime_error("ERROR (Tube::appendTubeSegment): Texture U fraction must be between 0 and 1.");
    }
    node_radius = std::max((float) 1e-5, node_radius);

    uint radial_subdivisions = subdiv;

    vec3 axial_vector;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);

    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    triangle_vertices.resize(triangle_vertices.size() + 1);
    triangle_vertices.back().resize(radial_subdivisions + 1);
    std::vector<std::vector<vec2>> uv;
    resize_vector(uv, radial_subdivisions + 1, 2);

    nodes.push_back(node_position);
    radius.push_back(node_radius);
    colors.push_back(RGB::black);

    int node_count = nodes.size();

    vec3 initial_radial(1.0f, 0.0f, 0.0f);
    vec3 previous_axial_vector;
    vec3 previous_radial_dir;

    for (int i = 0; i < node_count; i++) { // Looping over tube segments
        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTubeObject): Radius of tube must be positive.");
        }

        if (i == 0) {
            axial_vector = nodes[i + 1] - nodes[i];
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }
            if (fabs(axial_vector * initial_radial) > 0.95f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
            }
            // Also handle nearly vertical axes
            if (fabs(axial_vector.z) > 0.95f) {
                initial_radial = vec3(1.0f, 0.0f, 0.0f); // Use horizontal radial for vertical axes
            }
            previous_radial_dir = cross(axial_vector, initial_radial).normalize();
        } else {
            if (i == node_count - 1) {
                axial_vector = nodes[i] - nodes[i - 1];
            } else {
                axial_vector = 0.5f * ((nodes[i] - nodes[i - 1]) + (nodes[i + 1] - nodes[i]));
            }
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }

            // Calculate radial direction using parallel transport
            vec3 rotation_axis = cross(previous_axial_vector, axial_vector);
            if (rotation_axis.magnitude() > 1e-5) { // More conservative threshold
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
            } else {
                // Vectors are nearly parallel, use robust fallback
                vec3 fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                if (fabs(axial_vector * fallback_radial) > 0.95f) {
                    fallback_radial = vec3(0.0f, 1.0f, 0.0f);
                }
                if (fabs(axial_vector.z) > 0.95f) {
                    fallback_radial = vec3(1.0f, 0.0f, 0.0f);
                }
                previous_radial_dir = cross(axial_vector, fallback_radial).normalize();
            }
        }

        previous_axial_vector = axial_vector;

        vec3 radial_dir = previous_radial_dir;
        vec3 orthogonal_dir = cross(radial_dir, axial_vector);
        orthogonal_dir.normalize();

        if (i < node_count - 2) {
            continue;
        }

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal = cfact[j] * radius[i] * radial_dir + sfact[j] * radius[i] * orthogonal_dir;
            triangle_vertices[i][j] = nodes[i] + normal;
        }
    }

    std::vector<float> ufrac{textureuv_ufrac.x, textureuv_ufrac.y};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < radial_subdivisions + 1; j++) {
            uv[i][j].x = ufrac[i];
            uv[i][j].y = float(j) / float(radial_subdivisions);
        }
    }

    int old_triangle_count = UUIDs.size();

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;

    // Add triangles for new segment
    for (int j = 0; j < radial_subdivisions; j++) {
        v0 = triangle_vertices[node_count - 2][j];
        v1 = triangle_vertices[node_count - 1][j + 1];
        v2 = triangle_vertices[node_count - 2][j + 1];

        uv0 = uv[0][j];
        uv1 = uv[1][j + 1];
        uv2 = uv[0][j + 1];

        UUIDs.push_back(context->addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));

        v0 = triangle_vertices[node_count - 2][j];
        v1 = triangle_vertices[node_count - 1][j];
        v2 = triangle_vertices[node_count - 1][j + 1];

        uv0 = uv[0][j];
        uv1 = uv[1][j];
        uv2 = uv[1][j + 1];

        UUIDs.push_back(context->addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
    }

    for (uint p: UUIDs) {
        context->setPrimitiveParentObjectID(p, this->OID);
    }

    updateTriangleVertices();
}

void Tube::scaleTubeGirth(float S) {
    for (int segment = 0; segment < triangle_vertices.size(); segment++) {
        for (vec3 &vertex: triangle_vertices.at(segment)) {
            vec3 axis = vertex - nodes.at(segment);

            float current_radius = axis.magnitude();
            axis = axis / current_radius;

            vertex = nodes.at(segment) + axis * current_radius * S;
        }
        radius.at(segment) *= S;
    }

    updateTriangleVertices();
}

void Tube::setTubeRadii(const std::vector<float> &node_radii) {
    if (node_radii.size() != nodes.size()) {
        helios_runtime_error("ERROR (Tube::setTubeRadii): Number of radii in input vector must match number of tube nodes.");
    }

    radius = node_radii;

    for (int segment = 0; segment < triangle_vertices.size(); segment++) {
        for (vec3 &vertex: triangle_vertices.at(segment)) {
            vec3 axis = vertex - nodes.at(segment);
            axis.normalize();

            vertex = nodes.at(segment) + axis * radius.at(segment);
        }
    }

    updateTriangleVertices();
}

void Tube::scaleTubeLength(float S) {
    for (int segment = 0; segment < triangle_vertices.size() - 1; segment++) {
        vec3 central_axis = (nodes.at(segment + 1) - nodes.at(segment));
        float current_length = central_axis.magnitude();
        central_axis = central_axis / current_length;
        vec3 dL = central_axis * current_length * (1.f - S);

        for (int downstream_segment = segment + 1; downstream_segment < triangle_vertices.size(); downstream_segment++) {
            nodes.at(downstream_segment) = nodes.at(downstream_segment) - dL;

            for (int v = 0; v < triangle_vertices.at(downstream_segment).size(); v++) {
                triangle_vertices.at(downstream_segment).at(v) = triangle_vertices.at(downstream_segment).at(v) - dL;
            }
        }
    }

    updateTriangleVertices();
}

void Tube::setTubeNodes(const std::vector<helios::vec3> &node_xyz) {
    if (node_xyz.size() != nodes.size()) {
        helios_runtime_error("ERROR (Tube::setTubeNodes): Number of nodes in input vector must match number of tube nodes.");
    }

    for (int segment = 0; segment < triangle_vertices.size(); segment++) {
        for (vec3 &vertex: triangle_vertices.at(segment)) {
            vertex = node_xyz.at(segment) + vertex - nodes.at(segment);
        }
    }

    nodes = node_xyz;

    updateTriangleVertices();
}

void Tube::pruneTubeNodes(uint node_index) {
    if (node_index >= nodes.size()) {
        helios_runtime_error("ERROR (Tube::pruneTubeNodes): Node index of " + std::to_string(node_index) + " is out of bounds.");
    }

    if (node_index == 0) {
        context->deleteObject(this->OID);
        return;
    }

    nodes.erase(nodes.begin() + node_index, nodes.end());
    triangle_vertices.erase(triangle_vertices.begin() + node_index, triangle_vertices.end());
    radius.erase(radius.begin() + node_index, radius.end());
    colors.erase(colors.begin() + node_index, colors.end());

    int ii = 0;
    for (int i = node_index; i < nodes.size() - 1; i++) {
        for (int j = 0; j < subdiv; j++) {
            context->deletePrimitive(UUIDs.at(ii));
            context->deletePrimitive(UUIDs.at(ii + 1));
            ii += 2;
        }
    }
}

void Tube::updateTriangleVertices() const {
    int ii = 0;
    for (int i = 0; i < nodes.size() - 1; i++) {
        for (int j = 0; j < subdiv; j++) {
            vec3 v0 = triangle_vertices.at(i).at(j);
            vec3 v1 = triangle_vertices.at(i + 1).at(j + 1);
            vec3 v2 = triangle_vertices.at(i).at(j + 1);
            context->setTriangleVertices(UUIDs.at(ii), v0, v1, v2);

            v0 = triangle_vertices.at(i).at(j);
            v1 = triangle_vertices.at(i + 1).at(j);
            v2 = triangle_vertices.at(i + 1).at(j + 1);

            context->setTriangleVertices(UUIDs.at(ii + 1), v0, v1, v2);

            ii += 2;
        }
    }
}

// ============== BOX CLASS METHOD DEFINITIONS ==============

Box::Box(uint a_OID, const std::vector<uint> &a_UUIDs, const int3 &a_subdiv, const char *a_texturefile, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_BOX;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
    context = a_context;
}

Box *Context::getBoxObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return dynamic_cast<Box *>(objects.at(ObjID));
}

vec3 Box::getSize() const {
    vec3 n0(0, 0, 0), nx(1, 0, 0), ny(0, 1, 0), nz(0, 0, 1);

    vec3 n0_T, nx_T, ny_T, nz_T;

    vecmult(transform, n0, n0_T);
    vecmult(transform, nx, nx_T);
    vecmult(transform, ny, ny_T);
    vecmult(transform, nz, nz_T);

    float x = (nx_T - n0_T).magnitude();
    float y = (ny_T - n0_T).magnitude();
    float z = (nz_T - n0_T).magnitude();

    return make_vec3(x, y, z);
}

vec3 Box::getCenter() const {
    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;
}

helios::int3 Box::getSubdivisionCount() const {
    return subdiv;
}

void Box::setSubdivisionCount(const helios::int3 &a_subdiv) {
    subdiv = a_subdiv;
}

float Box::getVolume() const {
    const vec3 &size = getSize();
    return size.x * size.y * size.z;
}

// ============== DISK CLASS METHOD DEFINITIONS ==============

Disk::Disk(uint a_OID, const std::vector<uint> &a_UUIDs, int2 a_subdiv, const char *a_texturefile, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_DISK;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
    context = a_context;
}

Disk *Context::getDiskObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return dynamic_cast<Disk *>(objects.at(ObjID));
}

vec2 Disk::getSize() const {
    vec3 n0(0, 0, 0), nx(1, 0, 0), ny(0, 1, 0);
    vec3 n0_T, nx_T, ny_T;

    vecmult(transform, n0, n0_T);
    vecmult(transform, nx, nx_T);
    vecmult(transform, ny, ny_T);

    float x = (nx_T - n0_T).magnitude();
    float y = (ny_T - n0_T).magnitude();

    return make_vec2(x, y);
}

vec3 Disk::getCenter() const {
    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;
}

int2 Disk::getSubdivisionCount() const {
    return subdiv;
}

void Disk::setSubdivisionCount(const helios::int2 &a_subdiv) {
    subdiv = a_subdiv;
}

// ============== POLYMESH CLASS METHOD DEFINITIONS ==============

Polymesh::Polymesh(uint a_OID, const std::vector<uint> &a_UUIDs, const char *a_texturefile, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_POLYMESH;
    UUIDs = a_UUIDs;
    texturefile = a_texturefile;
    context = a_context;
}

Polymesh *Context::getPolymeshObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return dynamic_cast<Polymesh *>(objects.at(ObjID));
}

float Polymesh::getVolume() const {
    float volume = 0.f;
    for (uint UUID: UUIDs) {
        if (context->getPrimitiveType(UUID) == PRIMITIVE_TYPE_TRIANGLE) {
            const vec3 &v0 = context->getTriangleVertex(UUID, 0);
            const vec3 &v1 = context->getTriangleVertex(UUID, 1);
            const vec3 &v2 = context->getTriangleVertex(UUID, 2);
            volume += (1.f / 6.f) * v0 * cross(v1, v2);
        } else if (context->getPrimitiveType(UUID) == PRIMITIVE_TYPE_PATCH) {
            const vec3 &v0 = context->getTriangleVertex(UUID, 0);
            const vec3 &v1 = context->getTriangleVertex(UUID, 1);
            const vec3 &v2 = context->getTriangleVertex(UUID, 2);
            const vec3 &v3 = context->getTriangleVertex(UUID, 3);
            volume += (1.f / 6.f) * v0 * cross(v1, v2) + (1.f / 6.f) * v0 * cross(v2, v3);
        }
    }
    return std::abs(volume);
}

// ============== CONE CLASS METHOD DEFINITIONS ==============

Cone::Cone(uint a_OID, const std::vector<uint> &a_UUIDs, const vec3 &a_node0, const vec3 &a_node1, float a_radius0, float a_radius1, uint a_subdiv, const char *a_texturefile, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_CONE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
    context = a_context;
    nodes = {a_node0, a_node1};
    radii = {a_radius0, a_radius1};
}

Cone *Context::getConeObjectPointer(const uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return dynamic_cast<Cone *>(objects.at(ObjID));
}

std::vector<helios::vec3> Cone::getNodeCoordinates() const {
    std::vector<vec3> nodes_T;
    nodes_T.resize(2);

    for (int i = 0; i < 2; i++) {
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    return nodes_T;
}

helios::vec3 Cone::getNodeCoordinate(int node_index) const {
    if (node_index < 0 || node_index > 1) {
        helios_runtime_error("ERROR (Cone::getNodeCoordinate): node number must be 0 or 1.");
    }

    vec3 node_T;

    node_T.x = transform[0] * nodes.at(node_index).x + transform[1] * nodes.at(node_index).y + transform[2] * nodes.at(node_index).z + transform[3];
    node_T.y = transform[4] * nodes.at(node_index).x + transform[5] * nodes.at(node_index).y + transform[6] * nodes.at(node_index).z + transform[7];
    node_T.z = transform[8] * nodes.at(node_index).x + transform[9] * nodes.at(node_index).y + transform[10] * nodes.at(node_index).z + transform[11];

    return node_T;
}

std::vector<float> Cone::getNodeRadii() const {
    return radii;
}

float Cone::getNodeRadius(int node_index) const {
    if (node_index < 0 || node_index > 1) {
        helios_runtime_error("ERROR (Cone::getNodeRadius): node number must be 0 or 1.");
    }

    return radii.at(node_index);
}

uint Cone::getSubdivisionCount() const {
    return subdiv;
}

void Cone::setSubdivisionCount(uint a_subdiv) {
    subdiv = a_subdiv;
}

helios::vec3 Cone::getAxisUnitVector() const {
    std::vector<vec3> nodes_T;
    nodes_T.resize(2);

    for (uint i = 0; i < 2; i++) {
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    helios::vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z);
    float length = powf(powf(axis_unit_vector.x, 2) + powf(axis_unit_vector.y, 2) + powf(axis_unit_vector.z, 2), 0.5);
    axis_unit_vector = axis_unit_vector / length;

    return axis_unit_vector;
}

float Cone::getLength() const {
    std::vector<vec3> nodes_T;
    nodes_T.resize(2);

    for (uint i = 0; i < 2; i++) {
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    float length = powf(powf(nodes_T.at(1).x - nodes_T.at(0).x, 2) + powf(nodes_T.at(1).y - nodes_T.at(0).y, 2) + powf(nodes_T.at(1).z - nodes_T.at(0).z, 2), 0.5);
    return length;
}

void Cone::scaleLength(float S) {
    // get the nodes and radii of the nodes with transformation matrix applied
    const std::vector<helios::vec3> &nodes_T = context->getConeObjectPointer(OID)->getNodeCoordinates();
    const std::vector<float> &radii_T = context->getConeObjectPointer(OID)->getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z);
    float length = powf(powf(axis_unit_vector.x, 2) + powf(axis_unit_vector.y, 2) + powf(axis_unit_vector.z, 2), 0.5);
    axis_unit_vector = axis_unit_vector / length;

    // translate node 0 back to origin
    context->getConeObjectPointer(OID)->translate(-1.0 * nodes_T.at(0));

    // rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0, 0, 1);
    // get the axis about which to rotate
    vec3 ra = cross(z_axis, axis_unit_vector);
    // get the angle to rotate
    float dot = axis_unit_vector.x * z_axis.x + axis_unit_vector.y * z_axis.y + axis_unit_vector.z * z_axis.z;
    float angle = acos_safe(dot);

    // only rotate if the cone is not alread aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
    if (angle != float(0.0)) {
        context->getConeObjectPointer(OID)->rotate(-1 * angle, ra);
    }

    // scale the cone in the z (length) dimension
    float T[16], T_prim[16];
    makeScaleMatrix(make_vec3(1, 1, S), T);
    matmult(T, transform, transform);
    for (uint UUID: UUIDs) {
        if (context->doesPrimitiveExist(UUID)) {
            context->getPrimitiveTransformationMatrix(UUID, T_prim);
            matmult(T, T_prim, T_prim);
            context->setPrimitiveTransformationMatrix(UUID, T_prim);
        }
    }

    // rotate back
    if (angle != 0.0) {
        context->getConeObjectPointer(OID)->rotate(angle, ra);
    }

    // translate back
    context->getConeObjectPointer(OID)->translate(nodes_T.at(0));
}

void Cone::scaleGirth(float S) {
    // get the nodes and radii of the nodes with transformation matrix applied
    const std::vector<helios::vec3> &nodes_T = context->getConeObjectPointer(OID)->getNodeCoordinates();
    const std::vector<float> &radii_T = context->getConeObjectPointer(OID)->getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z);
    axis_unit_vector.normalize();

    // translate node 0 back to origin
    context->getConeObjectPointer(OID)->translate(-1.0 * nodes_T.at(0));
    // rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0, 0, 1);
    // get the axis about which to rotate
    vec3 ra = cross(z_axis, axis_unit_vector);
    // get the angle to rotate
    float dot = axis_unit_vector * z_axis;
    float angle = acos_safe(dot);
    // only rotate if the cone is not already aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
    if (angle != float(0.0)) {
        context->getConeObjectPointer(OID)->rotate(-1 * angle, ra);
    }

    // scale the cone in the x and y dimensions
    context->scaleObject(OID, make_vec3(S, S, 1));


    // rotate back
    if (angle != 0.0) {
        context->getConeObjectPointer(OID)->rotate(angle, ra);
    }

    // translate back
    context->getConeObjectPointer(OID)->translate(nodes_T.at(0));

    radii.at(0) *= S;
    radii.at(1) *= S;
}

float Cone::getVolume() const {
    float r0 = getNodeRadius(0);
    float r1 = getNodeRadius(1);
    float h = getLength();

    return PI_F * h / 3.f * (r0 * r0 + r0 * r1 + r1 * r1);
}
