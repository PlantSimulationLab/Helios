/**
 * \file "Context_primitive.cpp" Context primitive type definitions and methods.
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

uint Context::addPatch() {
    return addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBAcolor(0, 0, 0, 1));
}

uint Context::addPatch(const vec3 &center, const vec2 &size) {
    return addPatch(center, size, make_SphericalCoord(0, 0), make_RGBAcolor(0, 0, 0, 1));
}

uint Context::addPatch(const vec3 &center, const vec2 &size, const SphericalCoord &rotation) {
    return addPatch(center, size, rotation, make_RGBAcolor(0, 0, 0, 1));
}

uint Context::addPatch(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color) {
    return addPatch(center, size, rotation, make_RGBAcolor(color, 1));
}

uint Context::addPatch(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color) {
    if (size.x < 1e-6f || size.y < 1e-6f) {
        helios_runtime_error("ERROR (Context::addPatch): Size of patch must be greater than 1e-6 to avoid numerical precision issues.");
    }

    auto *patch_new = (new Patch(color, 0, currentUUID));

    //    if( patch_new->getArea()==0 ){
    //        helios_runtime_error("ERROR (Context::addPatch): Patch has area of zero.");
    //    }

    patch_new->scale(make_vec3(size.x, size.y, 1));

    if (rotation.elevation != 0) {
        patch_new->rotate(-rotation.elevation, "x");
    }
    if (rotation.azimuth != 0) {
        patch_new->rotate(-rotation.azimuth, "z");
    }

    patch_new->translate(center);

    primitives[currentUUID] = patch_new;
    currentUUID++;
    invalidateAllUUIDsCache();
    return currentUUID - 1;
}

uint Context::addPatch(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file) {
    addTexture(texture_file);

    // Default (u, v) mapping
    const std::vector<helios::vec2> uv = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};

    auto *patch_new = (new Patch(texture_file, uv, textures, 0, currentUUID));

    //    if( patch_new->getArea()==0 ){
    //        helios_runtime_error("ERROR (Context::addPatch): Patch has area of zero.");
    //    }

    assert(size.x > 0.f && size.y > 0.f);
    patch_new->scale(make_vec3(size.x, size.y, 1));

    if (rotation.elevation != 0) {
        patch_new->rotate(-rotation.elevation, "x");
    }
    if (rotation.azimuth != 0) {
        patch_new->rotate(-rotation.azimuth, "z");
    }

    patch_new->translate(center);

    primitives[currentUUID] = patch_new;
    currentUUID++;
    invalidateAllUUIDsCache();
    return currentUUID - 1;
}

uint Context::addPatch(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file, const helios::vec2 &uv_center, const helios::vec2 &uv_size) {
    if (size.x < 1e-6f || size.y < 1e-6f) {
        helios_runtime_error("ERROR (Context::addPatch): Size of patch must be greater than 1e-6 to avoid numerical precision issues.");
    }

    if (uv_center.x - 0.5 * uv_size.x < -1e-3 || uv_center.y - 0.5 * uv_size.y < -1e-3 || uv_center.x + 0.5 * uv_size.x - 1.f > 1e-3 || uv_center.y + 0.5 * uv_size.y - 1.f > 1e-3) {
        helios_runtime_error("ERROR (Context::addPatch): Invalid texture coordinates. uv_center-0.5*uv_size should be >=0 and uv_center+0.5*uv_size should be <=1.");
    }

    addTexture(texture_file);

    const std::vector<helios::vec2> uv = {uv_center + make_vec2(-0.5f * uv_size.x, -0.5f * uv_size.y), uv_center + make_vec2(+0.5f * uv_size.x, -0.5f * uv_size.y), uv_center + make_vec2(+0.5f * uv_size.x, +0.5f * uv_size.y),
                                          uv_center + make_vec2(-0.5f * uv_size.x, +0.5f * uv_size.y)};

    auto *patch_new = (new Patch(texture_file, uv, textures, 0, currentUUID));

    //    if( patch_new->getArea()==0 ){
    //        helios_runtime_error("ERROR (Context::addPatch): Patch has area of zero.");
    //    }

    assert(size.x > 0.f && size.y > 0.f);
    patch_new->scale(make_vec3(size.x, size.y, 1));

    if (rotation.elevation != 0) {
        patch_new->rotate(-rotation.elevation, "x");
    }
    if (rotation.azimuth != 0) {
        patch_new->rotate(-rotation.azimuth, "z");
    }

    patch_new->translate(center);

    primitives[currentUUID] = patch_new;
    currentUUID++;
    invalidateAllUUIDsCache();
    return currentUUID - 1;
}

uint Context::addTriangle(const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2) {
    return addTriangle(vertex0, vertex1, vertex2, make_RGBAcolor(0, 0, 0, 1));
}

uint Context::addTriangle(const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const RGBcolor &color) {
    return addTriangle(vertex0, vertex1, vertex2, make_RGBAcolor(color, 1));
}

uint Context::addTriangle(const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const RGBAcolor &color) {
    auto *tri_new = (new Triangle(vertex0, vertex1, vertex2, color, 0, currentUUID));

#ifdef HELIOS_DEBUG
    if (calculateTriangleArea(vertex0, vertex1, vertex2) < 1e-10) {
        std::cerr << "WARNING (Context::addTriangle): Triangle is malformed and has near-zero surface area." << std::endl;
    }
#endif

    primitives[currentUUID] = tri_new;
    currentUUID++;
    invalidateAllUUIDsCache();
    return currentUUID - 1;
}

uint Context::addTriangle(const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2, const char *texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2) {
    addTexture(texture_file);

    const std::vector<helios::vec2> uv{uv0, uv1, uv2};

    auto *tri_new = (new Triangle(vertex0, vertex1, vertex2, texture_file, uv, textures, 0, currentUUID));

#ifdef HELIOS_DEBUG
    if (calculateTriangleArea(vertex0, vertex1, vertex2) < 1e-10) {
        std::cerr << "WARNING (Context::addTriangle): Triangle is malformed and has near-zero surface area." << std::endl;
    }
#endif

    primitives[currentUUID] = tri_new;
    currentUUID++;
    invalidateAllUUIDsCache();
    return currentUUID - 1;
}

uint Context::addVoxel(const vec3 &center, const vec3 &size) {
    return addVoxel(center, size, 0, make_RGBAcolor(0, 0, 0, 1));
}

uint Context::addVoxel(const vec3 &center, const vec3 &size, const float &rotation) {
    return addVoxel(center, size, rotation, make_RGBAcolor(0, 0, 0, 1));
}

uint Context::addVoxel(const vec3 &center, const vec3 &size, const float &rotation, const RGBcolor &color) {
    return addVoxel(center, size, rotation, make_RGBAcolor(color, 1));
}

uint Context::addVoxel(const vec3 &center, const vec3 &size, const float &rotation, const RGBAcolor &color) {
    auto *voxel_new = (new Voxel(color, 0, currentUUID));

    if (size.x * size.y * size.z == 0) {
        helios_runtime_error("ERROR (Context::addVoxel): Voxel has size of zero.");
    }

    voxel_new->scale(size);

    if (rotation != 0) {
        voxel_new->rotate(rotation, "z");
    }

    voxel_new->translate(center);

    primitives[currentUUID] = voxel_new;
    currentUUID++;
    invalidateAllUUIDsCache();
    return currentUUID - 1;
}

void Context::translatePrimitive(uint UUID, const vec3 &shift) {
    getPrimitivePointer_private(UUID)->translate(shift);
}

void Context::translatePrimitive(const std::vector<uint> &UUIDs, const vec3 &shift) {
    float T[16];
    makeTranslationMatrix(shift, T);

    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::rotatePrimitive(uint UUID, float rotation_rad, const char *axis) {
    getPrimitivePointer_private(UUID)->rotate(rotation_rad, axis);
}

void Context::rotatePrimitive(const std::vector<uint> &UUIDs, float rotation_rad, const char *axis) {
    if (rotation_rad == 0) {
        return;
    }

    float T[16];
    if (strcmp(axis, "z") == 0) {
        makeRotationMatrix(rotation_rad, "z", T);
    } else if (strcmp(axis, "y") == 0) {
        makeRotationMatrix(rotation_rad, "y", T);
    } else if (strcmp(axis, "x") == 0) {
        makeRotationMatrix(rotation_rad, "x", T);
    } else {
        helios_runtime_error("ERROR (Context::rotatePrimitive): Rotation axis should be one of x, y, or z.");
    }

    for (uint UUID: UUIDs) {
        if (strcmp(axis, "z") != 0 && getPrimitivePointer_private(UUID)->getType() == PRIMITIVE_TYPE_VOXEL) {
            std::cerr << "WARNING (Context::rotatePrimitive): Voxels can only be rotate about the z-axis. Ignoring this rotation." << std::endl;
        }
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::rotatePrimitive(uint UUID, float rotation_rad, const helios::vec3 &axis) {
    getPrimitivePointer_private(UUID)->rotate(rotation_rad, axis);
}

void Context::rotatePrimitive(const std::vector<uint> &UUIDs, float rotation_rad, const vec3 &axis) {
    if (rotation_rad == 0) {
        return;
    }

    float T[16];
    makeRotationMatrix(rotation_rad, axis, T);

    for (uint UUID: UUIDs) {
        if (getPrimitivePointer_private(UUID)->getType() == PRIMITIVE_TYPE_VOXEL) {
            std::cerr << "WARNING (Context::rotatePrimitive): Voxels can only be rotate about the z-axis. Ignoring this rotation." << std::endl;
        }
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::rotatePrimitive(uint UUID, float rotation_rad, const helios::vec3 &origin, const helios::vec3 &axis) {
    getPrimitivePointer_private(UUID)->rotate(rotation_rad, origin, axis);
}

void Context::rotatePrimitive(const std::vector<uint> &UUIDs, float rotation_rad, const helios::vec3 &origin, const vec3 &axis) {
    if (rotation_rad == 0) {
        return;
    }

    float T[16];
    makeRotationMatrix(rotation_rad, origin, axis, T);

    for (uint UUID: UUIDs) {
        if (getPrimitivePointer_private(UUID)->getType() == PRIMITIVE_TYPE_VOXEL) {
            std::cerr << "WARNING (Context::rotatePrimitive): Voxels can only be rotate about the z-axis. Ignoring this rotation." << std::endl;
        }
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::setPrimitiveNormal(uint UUID, const helios::vec3 &origin, const helios::vec3 &new_normal) {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::setPrimitiveNormal): UUID of " + std::to_string(UUID) + " not found in the context.");
    }
#endif

    auto *prim = getPrimitivePointer_private(UUID);

    // old and new normals, unitized
    helios::vec3 oldN = normalize(prim->getNormal());
    helios::vec3 newN = normalize(new_normal);

    // minimal rotation axis/angle
    float d = std::clamp(oldN * newN, -1.f, 1.f);
    float angle = acosf(d);
    helios::vec3 axis = cross(oldN, newN);
    if (axis.magnitude() < 1e-6f) {
        axis = (std::fabs(oldN.x) < std::fabs(oldN.z)) ? cross(oldN, {1, 0, 0}) : cross(oldN, {0, 0, 1});
    }
    axis = axis.normalize();

    // build M_delta about 'origin'
    float M_delta[16];
    makeRotationMatrix(angle, origin, axis, M_delta);

    // grab existing world‐space model matrix
    float M_old[16];
    prim->getTransformationMatrix(M_old);

    // preserve the rectangle's forward (local X) direction:
    //   - t0 is the world‐space image of (1,0,0) under M_old
    helios::vec3 t0{
            M_old[0], // row0·[1,0,0,0]
            M_old[4], // row1·[1,0,0,0]
            M_old[8] // row2·[1,0,0,0]
    };
    t0 = normalize(t0);

    //  apply M_delta to that direction (w=0)
    helios::vec3 t1{M_delta[0] * t0.x + M_delta[1] * t0.y + M_delta[2] * t0.z, M_delta[4] * t0.x + M_delta[5] * t0.y + M_delta[6] * t0.z, M_delta[8] * t0.x + M_delta[9] * t0.y + M_delta[10] * t0.z};
    t1 = normalize(t1);

    //  desired forward is world‐X projected onto the new plane
    helios::vec3 worldX{1.f, 0.f, 0.f};
    helios::vec3 targ = worldX - newN * (newN * worldX);
    targ = normalize(targ);

    // compute the twist about newN that carries t1 → targ
    //    using signed angle in that plane
    float twist = std::atan2(newN * cross(t1, targ), // dot(newN, t1×targ)
                             t1 * targ // dot(t1, targ)
    );

    // build that correction rotation
    float M_twist[16];
    makeRotationMatrix(twist, origin, newN, M_twist);

    // now combine: M_new = M_twist * (M_delta * M_old)
    float temp[16], M_new[16];
    matmult(M_delta, M_old, temp);
    matmult(M_twist, temp, M_new);

    // write it back
    prim->setTransformationMatrix(M_new);
}

void Context::setPrimitiveNormal(const std::vector<uint> &UUIDs, const helios::vec3 &origin, const vec3 &new_normal) {
    for (uint UUID: UUIDs) {
        setPrimitiveNormal(UUID, origin, new_normal);
    }
}

void Context::setPrimitiveElevation(uint UUID, const vec3 &origin, float elevation_rad) {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID))
        helios_runtime_error("setPrimitiveElevation: invalid UUID");
#endif

    // pull the existing normal
    auto *prim = getPrimitivePointer_private(UUID);
    vec3 oldN = prim->getNormal();

    // convert to spherical coords, extract azimuth
    SphericalCoord sc = cart2sphere(oldN);
    float az = sc.azimuth;

    // build the new unit‐normal with desired elevation, same azimuth
    SphericalCoord targetSC(1.0f, elevation_rad, az);
    vec3 targetN = sphere2cart(targetSC);

    // delegate to your normal‐setting routine
    setPrimitiveNormal(UUID, origin, targetN);
}

void Context::setPrimitiveAzimuth(uint UUID, const vec3 &origin, float azimuth_rad) {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID))
        helios_runtime_error("setPrimitiveAzimuth: invalid UUID");
#endif

    // pull the existing normal
    auto *prim = getPrimitivePointer_private(UUID);
    vec3 oldN = prim->getNormal();

    // convert to spherical coords, extract elevation
    SphericalCoord sc = cart2sphere(oldN);
    float elev = sc.elevation;

    // build the new unit‐normal with same elevation, desired azimuth
    SphericalCoord targetSC(1.0f, elev, azimuth_rad);
    vec3 targetN = sphere2cart(targetSC);

    // delegate to your normal‐setting routine
    setPrimitiveNormal(UUID, origin, targetN);
}

void Context::scalePrimitive(uint UUID, const helios::vec3 &S) {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::scalePrimitive): UUID of " + std::to_string(UUID) + " not found in the context.");
    }
#endif
    if (S.x == 1 && S.y == 1 && S.z == 1) {
        return;
    }

    float T[16];
    makeScaleMatrix(S, T);

    getPrimitivePointer_private(UUID)->applyTransform(T);
}

void Context::scalePrimitive(const std::vector<uint> &UUIDs, const helios::vec3 &S) {
    for (uint UUID: UUIDs) {
        scalePrimitive(UUID, S);
    }
}

void Context::scalePrimitiveAboutPoint(uint UUID, const helios::vec3 &S, const helios::vec3 &point) {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::scalePrimitiveAboutPoint): UUID of " + std::to_string(UUID) + " not found in the context.");
    }
#endif
    if (S.x == 1 && S.y == 1 && S.z == 1) {
        return;
    }

    getPrimitivePointer_private(UUID)->scale(S, point);
}

void Context::scalePrimitiveAboutPoint(const std::vector<uint> &UUIDs, const helios::vec3 &S, const helios::vec3 &point) {
    for (uint UUID: UUIDs) {
        scalePrimitiveAboutPoint(UUID, S, point);
    }
}

void Context::deletePrimitive(const std::vector<uint> &UUIDs) {
    for (uint UUID: UUIDs) {
        deletePrimitive(UUID);
    }
}

void Context::deletePrimitive(uint UUID) {
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::deletePrimitive): UUID of " + std::to_string(UUID) + " not found in the context.");
    }

    Primitive *prim = primitives.at(UUID);

    for (const auto &[label, type]: prim->primitive_data_types) {
        decrementPrimitiveDataLabelCounter(label);
    }

    if (prim->getParentObjectID() != 0) { // primitive belongs to an object

        uint ObjID = prim->getParentObjectID();
        if (doesObjectExist(ObjID)) {
            objects.at(ObjID)->deleteChildPrimitive(UUID);
            if (getObjectPointer_private(ObjID)->getPrimitiveUUIDs().empty()) {
                CompoundObject *obj = objects.at(ObjID);
                delete obj;
                objects.erase(ObjID);
            }
        }
    }

    delete prim;
    primitives.erase(UUID);
    dirty_deleted_primitives.push_back(UUID);
    invalidateAllUUIDsCache();
}

std::vector<uint> Context::copyPrimitive(const std::vector<uint> &UUIDs) {
    std::vector<uint> UUIDs_copy(UUIDs.size());
    size_t i = 0;
    for (uint UUID: UUIDs) {
        UUIDs_copy.at(i) = copyPrimitive(UUID);
        i++;
    }

    return UUIDs_copy;
}

uint Context::copyPrimitive(uint UUID) {
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::copyPrimitive): UUID of " + std::to_string(UUID) + " not found in the context.");
    }

    PrimitiveType type = primitives.at(UUID)->getType();
    uint parentID = primitives.at(UUID)->getParentObjectID();
    bool textureoverride = primitives.at(UUID)->isTextureColorOverridden();

    if (type == PRIMITIVE_TYPE_PATCH) {
        Patch *p = getPatchPointer_private(UUID);
        const std::vector<vec2> &uv = p->getTextureUV();
        const vec2 &size = p->getSize();
        float solid_fraction = p->getArea() / (size.x * size.y);
        Patch *patch_new;
        if (!p->hasTexture()) {
            patch_new = (new Patch(p->getColorRGBA(), parentID, currentUUID));
        } else {
            const std::string &texture_file = p->getTextureFile();
            if (uv.size() == 4) {
                patch_new = (new Patch(texture_file.c_str(), solid_fraction, parentID, currentUUID));
                patch_new->setTextureUV(uv);
            } else {
                patch_new = (new Patch(texture_file.c_str(), solid_fraction, parentID, currentUUID));
            }
        }
        float transform[16];
        p->getTransformationMatrix(transform);
        patch_new->setTransformationMatrix(transform);
        primitives[currentUUID] = patch_new;
    } else if (type == PRIMITIVE_TYPE_TRIANGLE) {
        Triangle *p = getTrianglePointer_private(UUID);
        const std::vector<vec3> &vertices = p->getVertices();
        const std::vector<vec2> &uv = p->getTextureUV();
        Triangle *tri_new;
        if (!p->hasTexture()) {
            tri_new = (new Triangle(vertices.at(0), vertices.at(1), vertices.at(2), p->getColorRGBA(), parentID, currentUUID));
        } else {
            const std::string &texture_file = p->getTextureFile();
            float solid_fraction = p->getArea() / calculateTriangleArea(vertices.at(0), vertices.at(1), vertices.at(2));
            tri_new = (new Triangle(vertices.at(0), vertices.at(1), vertices.at(2), texture_file.c_str(), uv, solid_fraction, parentID, currentUUID));
            tri_new->setSolidFraction(solid_fraction);
        }
        float transform[16];
        p->getTransformationMatrix(transform);
        tri_new->setTransformationMatrix(transform);
        primitives[currentUUID] = tri_new;
    } else if (type == PRIMITIVE_TYPE_VOXEL) {
        Voxel *p = getVoxelPointer_private(UUID);
        Voxel *voxel_new;
        // if( !p->hasTexture() ){
        voxel_new = (new Voxel(p->getColorRGBA(), parentID, currentUUID));
        //}else{
        //  voxel_new = (new Voxel( p->getColorRGBA(), currentUUID ));
        /* \todo Texture-mapped voxels constructor here */
        //}
        float transform[16];
        p->getTransformationMatrix(transform);
        voxel_new->setTransformationMatrix(transform);
        primitives[currentUUID] = voxel_new;
    }

    copyPrimitiveData(UUID, currentUUID);

    if (textureoverride) {
        getPrimitivePointer_private(currentUUID)->overrideTextureColor();
    }

    currentUUID++;
    invalidateAllUUIDsCache();
    return currentUUID - 1;
}

Primitive *Context::getPrimitivePointer_private(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getPrimitivePointer_private): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return primitives.at(UUID);
}

bool Context::doesPrimitiveExist(uint UUID) const {
    return primitives.find(UUID) != primitives.end();
}

bool Context::doesPrimitiveExist(const std::vector<uint> &UUIDs) const {
    if (UUIDs.empty()) {
        return false;
    }
    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            return false;
        }
    }
    return true;
}

Patch *Context::getPatchPointer_private(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getPatchPointer_private): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_PATCH) {
        helios_runtime_error("ERROR (Context::getPatchPointer_private): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
#endif
    return dynamic_cast<Patch *>(primitives.at(UUID));
}

helios::vec2 Context::getPatchSize(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getPatchSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_PATCH) {
        helios_runtime_error("ERROR (Context::getPatchSize): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
#endif
    return dynamic_cast<Patch *>(primitives.at(UUID))->getSize();
}

helios::vec3 Context::getPatchCenter(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getPatchCenter): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_PATCH) {
        helios_runtime_error("ERROR (Context::getPatchCenter): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
#endif
    return dynamic_cast<Patch *>(primitives.at(UUID))->getCenter();
}

Triangle *Context::getTrianglePointer_private(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getTrianglePointer_private): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_TRIANGLE) {
        helios_runtime_error("ERROR (Context::getTrianglePointer_private): UUID of " + std::to_string(UUID) + " is not a triangle.");
    }
#endif
    return dynamic_cast<Triangle *>(primitives.at(UUID));
}

helios::vec3 Context::getTriangleVertex(uint UUID, uint number) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getTriangleVertex): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_TRIANGLE) {
        helios_runtime_error("ERROR (Context::getTriangleVertex): UUID of " + std::to_string(UUID) + " is not a triangle.");
    } else if (number > 2) {
        helios_runtime_error("ERROR (Context::getTriangleVertex): Vertex index must be one of 0, 1, or 2.");
    }
#endif
    return dynamic_cast<Triangle *>(primitives.at(UUID))->getVertex(number);
}

void Context::setTriangleVertices(uint UUID, const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2) {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::setTriangleVertices): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Triangle *>(primitives.at(UUID))->setVertices(vertex0, vertex1, vertex2);
}

Voxel *Context::getVoxelPointer_private(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getVoxelPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_VOXEL) {
        helios_runtime_error("ERROR (Context::getVoxelPointer): UUID of " + std::to_string(UUID) + " is not a voxel.");
    }
#endif
    return dynamic_cast<Voxel *>(primitives.at(UUID));
}

helios::vec3 Context::getVoxelSize(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getVoxelSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_VOXEL) {
        helios_runtime_error("ERROR (Context::getVoxelSize): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
#endif
    return dynamic_cast<Voxel *>(primitives.at(UUID))->getSize();
}

helios::vec3 Context::getVoxelCenter(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getVoxelCenter): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (primitives.at(UUID)->getType() != PRIMITIVE_TYPE_VOXEL) {
        helios_runtime_error("ERROR (Context::getVoxelCenter): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
#endif
    return dynamic_cast<Voxel *>(primitives.at(UUID))->getCenter();
}

size_t Context::getPrimitiveCount(bool include_hidden_primitives) const {
    if (include_hidden_primitives) {
        return primitives.size();
    } else {
        size_t count = 0;
        for (const auto &[UUID, primitive]: primitives) {
            if (!primitive->ishidden) {
                count++;
            }
        }
        return count;
    }
}

size_t Context::getTriangleCount(bool include_hidden_primitives) const {
    size_t count = 0;
    for (const auto &[UUID, primitive]: primitives) {
        if (primitive->getType() == PRIMITIVE_TYPE_TRIANGLE && (include_hidden_primitives || !primitive->ishidden)) {
            count++;
        }
    }
    return count;
}

size_t Context::getPatchCount(bool include_hidden_primitives) const {
    size_t count = 0;
    for (const auto &[UUID, primitive]: primitives) {
        if (primitive->getType() == PRIMITIVE_TYPE_PATCH && (include_hidden_primitives || !primitive->ishidden)) {
            count++;
        }
    }
    return count;
}

// ===== PRIMITIVE, PATCH, TRIANGLE, AND VOXEL CLASS METHODS =====
// Moved from Context.cpp

Primitive::~Primitive() = default;

uint Primitive::getUUID() const {
    return UUID;
}

PrimitiveType Primitive::getType() const {
    return prim_type;
}

void Primitive::setParentObjectID(uint objID) {
    parent_object_ID = objID;
}

uint Primitive::getParentObjectID() const {
    return parent_object_ID;
}

void Primitive::getTransformationMatrix(float (&T)[16]) const {
    std::memcpy(T, transform, 16 * sizeof(float));
}

void Primitive::setTransformationMatrix(float (&T)[16]) {
    std::memcpy(transform, T, 16 * sizeof(float));
    dirty_flag = true;
}

float Patch::getArea() const {
    const vec2 &size = getSize();

    return size.x * size.y * solid_fraction;
}

float Triangle::getArea() const {
    const std::vector<vec3> &vertices = getVertices();

    float area = calculateTriangleArea(vertices[0], vertices[1], vertices[2]);

    return area * solid_fraction;
}

float Voxel::getArea() const {
    const vec3 size(transform[0], transform[5], transform[10]);

    return 2.f * size.x * size.y + 2.f * size.x * size.z + 2.f * size.y * size.z;
}

vec3 Patch::getNormal() const {
    return normalize(make_vec3(transform[2], transform[6], transform[10]));
}

vec3 Triangle::getNormal() const {
    const std::vector<vec3> &vertices = getVertices();
    return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[1]));
}

vec3 Voxel::getNormal() const {
    return nullorigin;
}

std::vector<vec3> Patch::getVertices() const {
    std::vector<vec3> vertices(4);

    const std::vector<vec3> Y = {{-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f}, {0.5f, 0.5f, 0.f}, {-0.5f, 0.5f, 0.f}};

    for (int i = 0; i < 4; i++) {
        vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
        vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
        vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
    }
    return vertices;
}

std::vector<vec3> Triangle::getVertices() const {
    std::vector<vec3> vertices(3);

    const std::vector<vec3> Y = {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}};

    for (int i = 0; i < 3; i++) {
        vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
        vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
        vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
    }
    return vertices;
}

std::vector<vec3> Voxel::getVertices() const {
    std::vector<vec3> vertices(8);

    const std::vector<vec3> Y = {{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}};


    for (int i = 0; i < 8; i++) {
        vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
        vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
        vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
    }
    return vertices;
}

RGBcolor Primitive::getColor() const {
    return {color.r, color.g, color.b};
}

RGBcolor Primitive::getColorRGB() const {
    return {color.r, color.g, color.b};
}

RGBAcolor Primitive::getColorRGBA() const {
    return color;
}

void Primitive::setColor(const helios::RGBcolor &newcolor) {
    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::setColor): Cannot set the color of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    color = make_RGBAcolor(newcolor, 1.f);
    dirty_flag = true;
}

void Primitive::setColor(const helios::RGBAcolor &newcolor) {
    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::setColor): Cannot set the color of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    color = newcolor;
    dirty_flag = true;
}

bool Primitive::hasTexture() const {
    if (texturefile.empty()) {
        return false;
    } else {
        return true;
    }
}

std::string Primitive::getTextureFile() const {
    return texturefile;
}

void Primitive::setTextureFile(const char *texture) {
    texturefile = texture;
    dirty_flag = true;
}

std::vector<vec2> Primitive::getTextureUV() {
    return uv;
}

void Primitive::setTextureUV(const std::vector<vec2> &a_uv) {
    uv = a_uv;
    dirty_flag = true;
}

void Primitive::overrideTextureColor() {
    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::overrideTextureColor): Cannot set the texture options of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    texturecoloroverridden = true;
    dirty_flag = true;
}

void Primitive::useTextureColor() {
    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::useTextureColor): Cannot set the texture options of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    texturecoloroverridden = false;
    dirty_flag = true;
}

bool Primitive::isTextureColorOverridden() const {
    return texturecoloroverridden;
}

float Primitive::getSolidFraction() const {
    return solid_fraction;
}

void Primitive::setSolidFraction(float solidFraction) {
    solid_fraction = solidFraction;
    dirty_flag = true;
}

bool Triangle::edgeFunction(const helios::vec2 &a, const helios::vec2 &b, const helios::vec2 &c) {
    return ((c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y) >= 0);
}

void Triangle::setVertices(const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2) {
    makeTransformationMatrix(vertex0, vertex1, vertex2);
    dirty_flag = true;
}

void Primitive::applyTransform(float (&T)[16]) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Primitive::applyTransform): Cannot transform individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    matmult(T, transform, transform);
    dirty_flag = true;
}

void Primitive::scale(const vec3 &S) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Primitive::scale): Cannot scale individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (S.x == 0 || S.y == 0 || S.z == 0) {
        helios_runtime_error("ERROR (Primitive::scale): Scaling factor cannot be zero.");
    } else if (S.x == 1 && S.y == 1 && S.z == 1) {
        return;
    }

    float T[16];
    makeScaleMatrix(S, T);
    matmult(T, transform, transform);
    dirty_flag = true;
}

void Primitive::scale(const vec3 &S, const vec3 &point) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Primitive::scale): Cannot scale individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (S.x == 0 || S.y == 0 || S.z == 0) {
        helios_runtime_error("ERROR (Primitive::scale): Scaling factor cannot be zero.");
    } else if (S.x == 1 && S.y == 1 && S.z == 1) {
        return;
    }

    float T[16];
    makeScaleMatrix(S, point, T);
    matmult(T, transform, transform);
    dirty_flag = true;
}

void Primitive::translate(const helios::vec3 &shift) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Primitive::translate): Cannot translate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    if (shift == nullorigin) {
        return;
    }

    float T[16];
    makeTranslationMatrix(shift, T);
    matmult(T, transform, transform);
    dirty_flag = true;
}

void Patch::rotate(float rotation_radians, const char *rotation_axis_xyz_string) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Patch::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (rotation_radians == 0) {
        return;
    }

    if (strcmp(rotation_axis_xyz_string, "z") == 0) {
        float Rz[16];
        makeRotationMatrix(rotation_radians, "z", Rz);
        matmult(Rz, transform, transform);
    } else if (strcmp(rotation_axis_xyz_string, "y") == 0) {
        float Ry[16];
        makeRotationMatrix(rotation_radians, "y", Ry);
        matmult(Ry, transform, transform);
    } else if (strcmp(rotation_axis_xyz_string, "x") == 0) {
        float Rx[16];
        makeRotationMatrix(rotation_radians, "x", Rx);
        matmult(Rx, transform, transform);
    } else {
        helios_runtime_error("ERROR (Patch::rotate): Rotation axis should be one of x, y, or z.");
    }
    dirty_flag = true;
}

void Patch::rotate(float rotation_radians, const helios::vec3 &rotation_axis_vector) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Patch::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (rotation_radians == 0) {
        return;
    }

    float R[16];
    makeRotationMatrix(rotation_radians, rotation_axis_vector, R);
    matmult(R, transform, transform);
    dirty_flag = true;
}

void Patch::rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Patch::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (rotation_radians == 0) {
        return;
    }

    float R[16];
    makeRotationMatrix(rotation_radians, origin, rotation_axis_vector, R);
    matmult(R, transform, transform);
    dirty_flag = true;
}

void Triangle::rotate(float rotation_radians, const char *rotation_axis_xyz_string) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Triangle::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (rotation_radians == 0) {
        return;
    }

    if (strcmp(rotation_axis_xyz_string, "z") == 0) {
        float Rz[16];
        makeRotationMatrix(rotation_radians, "z", Rz);
        matmult(Rz, transform, transform);
    } else if (strcmp(rotation_axis_xyz_string, "y") == 0) {
        float Ry[16];
        makeRotationMatrix(rotation_radians, "y", Ry);
        matmult(Ry, transform, transform);
    } else if (strcmp(rotation_axis_xyz_string, "x") == 0) {
        float Rx[16];
        makeRotationMatrix(rotation_radians, "x", Rx);
        matmult(Rx, transform, transform);
    } else {
        helios_runtime_error("ERROR (Triangle::rotate): Rotation axis should be one of x, y, or z.");
    }
    dirty_flag = true;
}

void Triangle::rotate(float rotation_radians, const helios::vec3 &rotation_axis_vector) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Triangle::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (rotation_radians == 0) {
        return;
    }

    float R[16];
    makeRotationMatrix(rotation_radians, rotation_axis_vector, R);
    matmult(R, transform, transform);
    dirty_flag = true;
}

void Triangle::rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Triangle::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (rotation_radians == 0) {
        return;
    }

    float R[16];
    makeRotationMatrix(rotation_radians, origin, rotation_axis_vector, R);
    matmult(R, transform, transform);
    dirty_flag = true;
}

void Voxel::rotate(float rotation_radians, const char *rotation_axis_xyz_string) {
    if (parent_object_ID != 0) {
        std::cerr << "WARNING (Voxel::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }
    if (rotation_radians == 0) {
        return;
    }

    float Rz[16];
    makeRotationMatrix(rotation_radians, "z", Rz);
    matmult(Rz, transform, transform);
    dirty_flag = true;
}

void Voxel::rotate(float rotation_radians, const helios::vec3 &rotation_axis_vector) {
    std::cerr << "WARNING (Voxel::rotate) - Voxels can only be rotated about the z-axis. Ignoring this call to rotate()." << std::endl;
}

void Voxel::rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector) {
    std::cerr << "WARNING (Voxel::rotate) - Voxels can only be rotated about the z-axis. Ignoring this call to rotate()." << std::endl;
}

void Triangle::makeTransformationMatrix(const helios::vec3 &vert0, const helios::vec3 &vert1, const helios::vec3 &vert2) {
    // We need to construct the Affine transformation matrix that transforms some generic triangle to a triangle with vertices at vertex0, vertex1, vertex2.

    // V1 is going to be our generic triangle.  This is the triangle that we'll intersect in the OptiX ray intersection program.  We just need to pass the transformation matrix to OptiX so that we'll end up with the right triangle.

    // We'll assume our generic triangle has vertices
    // v0 = (0,0,0)
    // v1 = (0,1,0)
    // v2 = (1,1,0)
    // this needs to match up with the triangle in triangle_intersect() and triangle_bounds() (see primitiveIntersection.cu).
    // Note that the matrix is padded with 1's to make it 4x4

    float V1[16];

    /* [0,0] */
    V1[0] = 0.f;
    /* [0,1] */
    V1[1] = 0.f;
    /* [0,2] */
    V1[2] = 1.f;

    /* [1,0] */
    V1[4] = 0.f;
    /* [1,1] */
    V1[5] = 1.f;
    /* [1,2] */
    V1[6] = 1.f;

    /* [2,0] */
    V1[8] = 0.f;
    /* [2,1] */
    V1[9] = 0.f;
    /* [2,2] */
    V1[10] = 0.f;

    /* [0,3] */
    V1[3] = 1.f;
    /* [1,3] */
    V1[7] = 1.f;
    /* [2,3] */
    V1[11] = 1.f;
    /* [3,0] */
    V1[12] = 1.f;
    /* [3,1] */
    V1[13] = 1.f;
    /* [3,2] */
    V1[14] = 1.f;
    /* [3,3] */
    V1[15] = 1.f;

    // V2 holds the vertex locations we want to transform to
    // Note that the matrix is padded with 1's to make it 4x4

    float V2[16];
    /* [0,0] */
    V2[0] = vert0.x;
    /* [0,1] */
    V2[1] = vert1.x;
    /* [0,2] */
    V2[2] = vert2.x;
    /* [0,3] */
    V2[3] = 1.f;
    /* [1,0] */
    V2[4] = vert0.y;
    /* [1,1] */
    V2[5] = vert1.y;
    /* [1,2] */
    V2[6] = vert2.y;
    /* [1,3] */
    V2[7] = 1.f;
    /* [2,0] */
    V2[8] = vert0.z;
    /* [2,1] */
    V2[9] = vert1.z;
    /* [2,2] */
    V2[10] = vert2.z;
    /* [2,3] */
    V2[11] = 1.f;
    /* [3,0] */
    V2[12] = 1.f;
    /* [3,1] */
    V2[13] = 1.f;
    /* [3,2] */
    V2[14] = 1.f;
    /* [3,3] */
    V2[15] = 1.f;

    // Now we just need to solve the linear system for our transform matrix T
    //  [T][V1] = [V2]  -->
    //  [T] = [V2]([V1]^-1)

    double inv[16], det, invV1[16];

    inv[0] = V1[5] * V1[10] * V1[15] - V1[5] * V1[11] * V1[14] - V1[9] * V1[6] * V1[15] + V1[9] * V1[7] * V1[14] + V1[13] * V1[6] * V1[11] - V1[13] * V1[7] * V1[10];

    inv[4] = -V1[4] * V1[10] * V1[15] + V1[4] * V1[11] * V1[14] + V1[8] * V1[6] * V1[15] - V1[8] * V1[7] * V1[14] - V1[12] * V1[6] * V1[11] + V1[12] * V1[7] * V1[10];

    inv[8] = V1[4] * V1[9] * V1[15] - V1[4] * V1[11] * V1[13] - V1[8] * V1[5] * V1[15] + V1[8] * V1[7] * V1[13] + V1[12] * V1[5] * V1[11] - V1[12] * V1[7] * V1[9];

    inv[12] = -V1[4] * V1[9] * V1[14] + V1[4] * V1[10] * V1[13] + V1[8] * V1[5] * V1[14] - V1[8] * V1[6] * V1[13] - V1[12] * V1[5] * V1[10] + V1[12] * V1[6] * V1[9];

    inv[1] = -V1[1] * V1[10] * V1[15] + V1[1] * V1[11] * V1[14] + V1[9] * V1[2] * V1[15] - V1[9] * V1[3] * V1[14] - V1[13] * V1[2] * V1[11] + V1[13] * V1[3] * V1[10];

    inv[5] = V1[0] * V1[10] * V1[15] - V1[0] * V1[11] * V1[14] - V1[8] * V1[2] * V1[15] + V1[8] * V1[3] * V1[14] + V1[12] * V1[2] * V1[11] - V1[12] * V1[3] * V1[10];

    inv[9] = -V1[0] * V1[9] * V1[15] + V1[0] * V1[11] * V1[13] + V1[8] * V1[1] * V1[15] - V1[8] * V1[3] * V1[13] - V1[12] * V1[1] * V1[11] + V1[12] * V1[3] * V1[9];

    inv[13] = V1[0] * V1[9] * V1[14] - V1[0] * V1[10] * V1[13] - V1[8] * V1[1] * V1[14] + V1[8] * V1[2] * V1[13] + V1[12] * V1[1] * V1[10] - V1[12] * V1[2] * V1[9];

    inv[2] = V1[1] * V1[6] * V1[15] - V1[1] * V1[7] * V1[14] - V1[5] * V1[2] * V1[15] + V1[5] * V1[3] * V1[14] + V1[13] * V1[2] * V1[7] - V1[13] * V1[3] * V1[6];

    inv[6] = -V1[0] * V1[6] * V1[15] + V1[0] * V1[7] * V1[14] + V1[4] * V1[2] * V1[15] - V1[4] * V1[3] * V1[14] - V1[12] * V1[2] * V1[7] + V1[12] * V1[3] * V1[6];

    inv[10] = V1[0] * V1[5] * V1[15] - V1[0] * V1[7] * V1[13] - V1[4] * V1[1] * V1[15] + V1[4] * V1[3] * V1[13] + V1[12] * V1[1] * V1[7] - V1[12] * V1[3] * V1[5];

    inv[14] = -V1[0] * V1[5] * V1[14] + V1[0] * V1[6] * V1[13] + V1[4] * V1[1] * V1[14] - V1[4] * V1[2] * V1[13] - V1[12] * V1[1] * V1[6] + V1[12] * V1[2] * V1[5];

    inv[3] = -V1[1] * V1[6] * V1[11] + V1[1] * V1[7] * V1[10] + V1[5] * V1[2] * V1[11] - V1[5] * V1[3] * V1[10] - V1[9] * V1[2] * V1[7] + V1[9] * V1[3] * V1[6];

    inv[7] = V1[0] * V1[6] * V1[11] - V1[0] * V1[7] * V1[10] - V1[4] * V1[2] * V1[11] + V1[4] * V1[3] * V1[10] + V1[8] * V1[2] * V1[7] - V1[8] * V1[3] * V1[6];

    inv[11] = -V1[0] * V1[5] * V1[11] + V1[0] * V1[7] * V1[9] + V1[4] * V1[1] * V1[11] - V1[4] * V1[3] * V1[9] - V1[8] * V1[1] * V1[7] + V1[8] * V1[3] * V1[5];

    inv[15] = V1[0] * V1[5] * V1[10] - V1[0] * V1[6] * V1[9] - V1[4] * V1[1] * V1[10] + V1[4] * V1[2] * V1[9] + V1[8] * V1[1] * V1[6] - V1[8] * V1[2] * V1[5];

    det = V1[0] * inv[0] + V1[1] * inv[4] + V1[2] * inv[8] + V1[3] * inv[12];

    // if (det == 0)
    // return false;

    det = 1.0 / det;

    for (int i = 0; i < 16; i++)
        invV1[i] = inv[i] * det;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transform[j + i * 4] = 0.f;
        }
    }

    // Multiply to get transformation matrix [T] = [V2]([V1]^-1)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                transform[j + i * 4] += V2[k + i * 4] * float(invV1[j + k * 4]);
            }
        }
    }
    dirty_flag = true;
}

Patch::Patch(const RGBAcolor &a_color, uint a_parent_objID, uint a_UUID) {
    makeIdentityMatrix(transform);

    color = a_color;
    assert(color.r >= 0 && color.r <= 1 && color.g >= 0 && color.g <= 1 && color.b >= 0 && color.b <= 1);
    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    solid_fraction = 1.f;
    texturefile = "";
    texturecoloroverridden = false;
    dirty_flag = true;
}

Patch::Patch(const char *a_texturefile, float a_solid_fraction, uint a_parent_objID, uint a_UUID) {
    makeIdentityMatrix(transform);

    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    texturefile = a_texturefile;
    solid_fraction = a_solid_fraction;
    texturecoloroverridden = false;
    dirty_flag = true;
}

Patch::Patch(const char *a_texturefile, const std::vector<helios::vec2> &a_uv, std::map<std::string, Texture> &textures, uint a_parent_objID, uint a_UUID) {
    makeIdentityMatrix(transform);

    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;

    texturefile = a_texturefile;
    uv = a_uv;
    for (auto &uv_vert: uv) {
        uv_vert.x = std::min(uv_vert.x, 1.f);
        uv_vert.y = std::min(uv_vert.y, 1.f);
    }
    texturecoloroverridden = false;

    solid_fraction = textures.at(texturefile).getSolidFraction(uv);
    dirty_flag = true;
}

helios::vec2 Patch::getSize() const {
    const std::vector<vec3> &vertices = getVertices();
    float l = (vertices.at(1) - vertices.at(0)).magnitude();
    float w = (vertices.at(3) - vertices.at(0)).magnitude();
    return {l, w};
}

helios::vec3 Patch::getCenter() const {
    return make_vec3(transform[3], transform[7], transform[11]);
}

Triangle::Triangle(const helios::vec3 &a_vertex0, const helios::vec3 &a_vertex1, const helios::vec3 &a_vertex2, const helios::RGBAcolor &a_color, uint a_parent_objID, uint a_UUID) {
    makeTransformationMatrix(a_vertex0, a_vertex1, a_vertex2);
    color = a_color;
    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;
    texturefile = "";
    solid_fraction = 1.f;
    texturecoloroverridden = false;
    dirty_flag = true;
}

Triangle::Triangle(const helios::vec3 &a_vertex0, const helios::vec3 &a_vertex1, const helios::vec3 &a_vertex2, const char *a_texturefile, const std::vector<helios::vec2> &a_uv, float solid_fraction, uint a_parent_objID, uint a_UUID) {
    makeTransformationMatrix(a_vertex0, a_vertex1, a_vertex2);
    color = make_RGBAcolor(RGB::red, 1);
    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;

    texturefile = a_texturefile;
    uv = a_uv;
    this->solid_fraction = solid_fraction;
    texturecoloroverridden = false;
    dirty_flag = true;
}

Triangle::Triangle(const helios::vec3 &a_vertex0, const helios::vec3 &a_vertex1, const helios::vec3 &a_vertex2, const char *a_texturefile, const std::vector<helios::vec2> &a_uv, std::map<std::string, Texture> &textures, uint a_parent_objID,
                   uint a_UUID) {
    makeTransformationMatrix(a_vertex0, a_vertex1, a_vertex2);
    color = make_RGBAcolor(RGB::red, 1);
    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;

    texturefile = a_texturefile;
    uv = a_uv;
    for (auto &uv_vert: uv) {
        uv_vert.x = std::min(uv_vert.x, 1.f);
        uv_vert.y = std::min(uv_vert.y, 1.f);
    }
    solid_fraction = 1.f;
    texturecoloroverridden = false;

    solid_fraction = textures.at(texturefile).getSolidFraction(uv);
    dirty_flag = true;
}

vec3 Triangle::getVertex(int vertex_index) const {
    if (vertex_index < 0 || vertex_index > 2) {
        helios_runtime_error("ERROR (Context::getVertex): vertex index must be 1, 2, or 3.");
    }

    const std::vector<vec3> Y = {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 1.f, 0.f}};

    vec3 vertex;

    vertex.x = transform[0] * Y[vertex_index].x + transform[1] * Y[vertex_index].y + transform[2] * Y[vertex_index].z + transform[3];
    vertex.y = transform[4] * Y[vertex_index].x + transform[5] * Y[vertex_index].y + transform[6] * Y[vertex_index].z + transform[7];
    vertex.z = transform[8] * Y[vertex_index].x + transform[9] * Y[vertex_index].y + transform[10] * Y[vertex_index].z + transform[11];

    return vertex;
}

vec3 Triangle::getCenter() const {
    //    Y[0] = make_vec3( 0.f, 0.f, 0.f);
    //    Y[1] = make_vec3( 0.f, 1.f, 0.f);
    //    Y[2] = make_vec3( 1.f/3.f, 1.f, 0.f);

    vec3 center0(1.f / 3.f, 2.f / 3.f, 0.f);
    vec3 center;

    center.x = transform[0] * center0.x + transform[1] * center0.y + transform[2] * center0.z + transform[3];
    center.y = transform[4] * center0.x + transform[5] * center0.y + transform[6] * center0.z + transform[7];
    center.z = transform[8] * center0.x + transform[9] * center0.y + transform[10] * center0.z + transform[11];

    return center;
}

Voxel::Voxel(const RGBAcolor &a_color, uint a_parent_objID, uint a_UUID) {
    makeIdentityMatrix(transform);

    color = a_color;
    assert(color.r >= 0 && color.r <= 1 && color.g >= 0 && color.g <= 1 && color.b >= 0 && color.b <= 1);
    solid_fraction = 1.f;
    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_VOXEL;
    texturefile = "";
    texturecoloroverridden = false;
    dirty_flag = true;
}

float Voxel::getVolume() {
    const vec3 &size = getSize();

    return size.x * size.y * size.z;
}

vec3 Voxel::getCenter() const {
    vec3 center;
    vec3 Y;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;
}

vec3 Voxel::getSize() const {
    vec3 n0(0, 0, 0), nx(1, 0, 0), ny(0, 1, 0), nz(0, 0, 1);
    vec3 n0_T, nx_T, ny_T, nz_T;

    vecmult(transform, n0, n0_T);
    vecmult(transform, nx, nx_T);
    vecmult(transform, ny, ny_T);
    vecmult(transform, nz, nz_T);

    float x = (nx_T - n0_T).magnitude();
    float y = (ny_T - n0_T).magnitude();
    float z = (nz_T - n0_T).magnitude();

    return {x, y, z};
}
