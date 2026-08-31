/**
 * \file "Context_object.cpp" Context compound object definitions and methods.
 *
 * Copyright (C) 2016-2026 Brian Bailey
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

namespace {

    //! Compute the inverse transpose of the upper-left 3x3 block of a Helios 4x4 affine transformation matrix
    /**
     * Normals must be transformed by the inverse transpose rather than by the matrix itself, otherwise they no longer remain perpendicular to the surface under non-uniform scaling. Returns false when the
     * linear part of the transformation is singular, in which case no meaningful normal direction exists.
     */
    bool makeNormalMatrix(const float (&T)[16], float (&N)[9]) {
        const float a = T[0], b = T[1], c = T[2];
        const float d = T[4], e = T[5], f = T[6];
        const float g = T[8], h = T[9], i = T[10];

        const float determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if (std::abs(determinant) < 1e-20f) {
            return false;
        }
        const float inv_det = 1.f / determinant;

        // Inverse of the 3x3 block (cofactor matrix / determinant), immediately transposed by the index assignment below.
        N[0] = (e * i - f * h) * inv_det;
        N[3] = -(b * i - c * h) * inv_det;
        N[6] = (b * f - c * e) * inv_det;
        N[1] = -(d * i - f * g) * inv_det;
        N[4] = (a * i - c * g) * inv_det;
        N[7] = -(a * f - c * d) * inv_det;
        N[2] = (d * h - e * g) * inv_det;
        N[5] = -(a * h - b * g) * inv_det;
        N[8] = (a * e - b * d) * inv_det;

        return true;
    }

    //! Apply a 3x3 normal matrix to a direction vector
    helios::vec3 transformNormalByMatrix(const float (&N)[9], const helios::vec3 &normal) {
        helios::vec3 result;
        result.x = N[0] * normal.x + N[1] * normal.y + N[2] * normal.z;
        result.y = N[3] * normal.x + N[4] * normal.y + N[5] * normal.z;
        result.z = N[6] * normal.x + N[7] * normal.y + N[8] * normal.z;
        return result;
    }

} // namespace

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

//! Reduce a requested texture repeat count so that it evenly divides the tile's subdivision count
/**
 * The (u,v) window assigned to each sub-patch is 1/(subdiv/repeat) of the texture image, so the repeat count must divide the subdivision count evenly for the sub-patches to tile the image. A requested count that does not
 * is silently resized downward to the nearest divisor. Both components of the result are guaranteed to lie in [1, subdiv].
 */
static helios::int2 correctTextureRepeat(const helios::int2 &subdiv, const helios::int2 &texture_repeat) {
    helios::int2 repeat = texture_repeat;
    repeat.x = std::min(subdiv.x, repeat.x);
    repeat.y = std::min(subdiv.y, repeat.y);
    // Terminates because both components are >= 1 at this point and subdiv % 1 == 0.
    while (subdiv.x % repeat.x != 0) {
        repeat.x--;
    }
    while (subdiv.y % repeat.y != 0) {
        repeat.y--;
    }
    return repeat;
}

//! Largest sub-patch count an adaptive tile object is permitted to generate
/**
 * The sub-patch count is extremely sensitive to the transition exponent: on a 50x50 m tile refined to 2 cm, an exponent of 0.25 yields roughly 9 thousand sub-patches while an exponent of 2 yields roughly 2 million.
 * Without a limit a plausible-looking set of parameters can exhaust memory, so the count is checked before any geometry is allocated.
 */
static constexpr size_t adaptive_tile_max_subpatch_count = 2000000;

//! Quadtree grid description used to generate the sub-patches of an adaptive tile object
struct AdaptiveTileLayout {
    //! Number of coarsest-level cells spanning the tile in the x- and y-directions
    helios::int2 base_subdiv;
    //! Maximum number of times a coarsest-level cell may be subdivided
    int max_level;
    //! Dimensions of a coarsest-level cell
    helios::vec2 base_cell_size;
    //! Refinement target, in tile-local coordinates relative to the tile center
    helios::vec2 target;
    //! Largest closest-point distance from the target to any coarsest-level cell
    float r_edge;
    //! Exponent controlling how rapidly sub-patch size grows with distance from the target
    float transition_exponent;
};

//! One leaf cell of an adaptive tile quadtree, described purely by integer indices
/**
 * Cell geometry is always recovered from these indices against a fixed base cell size, never by accumulating offsets from a parent cell. This is what guarantees that the leaves exactly partition the tile: a shared
 * edge between two cells is evaluated from the same expression on both sides, so it comes out bitwise identical and no gap or overlap can open up.
 */
struct AdaptiveTileCell {
    //! Index in the x-direction of the coarsest-level cell containing this cell
    int base_i;
    //! Index in the y-direction of the coarsest-level cell containing this cell
    int base_j;
    //! Index in the x-direction of this cell within its coarsest-level cell, in the range 0 to 2^level-1
    uint local_i;
    //! Index in the y-direction of this cell within its coarsest-level cell, in the range 0 to 2^level-1
    uint local_j;
    //! Number of times the coarsest-level cell has been subdivided to reach this cell
    int level;
};

//! Get the tile-local bounds of an adaptive tile quadtree cell
static void getAdaptiveTileCellBounds(const AdaptiveTileLayout &layout, const helios::vec2 &size, const AdaptiveTileCell &cell, float &x0, float &y0, float &dx, float &dy) {
    const float inv = 1.f / float(1u << cell.level);
    dx = layout.base_cell_size.x * inv;
    dy = layout.base_cell_size.y * inv;
    x0 = -0.5f * size.x + (float(cell.base_i) + float(cell.local_i) * inv) * layout.base_cell_size.x;
    y0 = -0.5f * size.y + (float(cell.base_j) + float(cell.local_j) * inv) * layout.base_cell_size.y;
}

//! Get the distance from a point to the closest point of an axis-aligned rectangle, which is zero when the point lies inside it
/**
 * Refinement is driven by the distance to the closest point of a cell rather than to its center. Using the center would bias refinement toward whichever side of the target the cell centers happen to fall on, making
 * the fine region asymmetric about the target; the closest-point distance makes it radially symmetric.
 */
static float adaptiveTileCellDistance(const helios::vec2 &target, float x0, float y0, float dx, float dy) {
    const float closest_x = std::min(std::max(target.x, x0), x0 + dx);
    const float closest_y = std::min(std::max(target.y, y0), y0 + dy);
    return std::hypot(target.x - closest_x, target.y - closest_y);
}

//! Determine the quadtree grid that best approximates a requested adaptive tile sub-patch size range
/**
 * The maximum refinement level is the base-2 logarithm of the requested size ratio, and the base cell size is the geometric mean of the requested maximum and of the requested minimum scaled back up by that many
 * levels. Taking the geometric mean balances the resulting relative error between the two ends of the range rather than making one end exact and the other arbitrarily wrong.
 */
static AdaptiveTileLayout computeAdaptiveTileLayout(const helios::vec2 &size, const helios::AdaptiveTileRefinement &refinement, const helios::int2 &texture_repeat) {
    AdaptiveTileLayout layout;

    layout.target = refinement.target;
    layout.transition_exponent = refinement.transition_exponent;
    layout.max_level = std::max(0, static_cast<int>(std::lround(std::log2(refinement.subpatch_size_max / refinement.subpatch_size_min))));

    const float base_size_target = std::sqrt(refinement.subpatch_size_max * refinement.subpatch_size_min * std::exp2(float(layout.max_level)));
    layout.base_subdiv.x = std::max(1, static_cast<int>(std::lround(size.x / base_size_target)));
    layout.base_subdiv.y = std::max(1, static_cast<int>(std::lround(size.y / base_size_target)));

    if (texture_repeat.x > layout.base_subdiv.x || texture_repeat.y > layout.base_subdiv.y) {
        // Honoring such a repeat count is possible, but only by inflating the coarsest-level grid well past what the
        // requested sub-patch size range asks for, since the grid must be snapped to a whole number of repeats below.
        // Rejecting it is better than silently abandoning the requested size range.
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): The requested texture repeat count of " + std::to_string(texture_repeat.x) + "x" + std::to_string(texture_repeat.y) +
                             " exceeds the number of coarsest-level cells spanning the tile (" + std::to_string(layout.base_subdiv.x) + "x" + std::to_string(layout.base_subdiv.y) +
                             "), so honoring it would require a coarsest-level grid much finer than the requested subpatch_size_max. Reduce the texture repeat count, increase the tile size, or decrease "
                             "subpatch_size_max.");
    }

    // Snap the base grid to a whole number of texture repeats. Every texture repeat boundary then coincides with a boundary between coarsest-level cells, and since quadtree children only ever subdivide within their
    // own coarsest-level cell, no sub-patch can straddle one. Snapping the grid rather than reducing the repeat count (as addTileObject does) honors the requested count exactly, at the cost of perturbing the achieved
    // sub-patch sizes by at most one repeat period. The grid is a derived quantity the user cannot see, so silently degrading their requested repeat count against it would be surprising.
    if (texture_repeat.x > 1) {
        layout.base_subdiv.x = texture_repeat.x * std::max(1, static_cast<int>(std::lround(float(layout.base_subdiv.x) / float(texture_repeat.x))));
    }
    if (texture_repeat.y > 1) {
        layout.base_subdiv.y = texture_repeat.y * std::max(1, static_cast<int>(std::lround(float(layout.base_subdiv.y) / float(texture_repeat.y))));
    }

    // Every coarsest-level cell yields at least one sub-patch, so the grid is a lower bound on the final sub-patch
    // count and can be rejected here. Leaving this to the traversal's own limit would apply it far too late: the grid
    // is materialized in full before any subdivision happens, so a domain large relative to subpatch_size_max would
    // first spend hundreds of megabytes — gigabytes, for a multi-kilometer domain — building a grid whose only purpose
    // is to be rejected, turning what should be a clean error into an out-of-memory crash.
    const size_t base_cell_count = size_t(layout.base_subdiv.x) * size_t(layout.base_subdiv.y);
    if (base_cell_count >= adaptive_tile_max_subpatch_count) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): The coarsest-level grid alone would contain " + std::to_string(base_cell_count) + " cells, which already exceeds the maximum of " +
                             std::to_string(adaptive_tile_max_subpatch_count) + " sub-patches. Increase subpatch_size_max or decrease the tile size.");
    }

    layout.base_cell_size = helios::make_vec2(size.x / float(layout.base_subdiv.x), size.y / float(layout.base_subdiv.y));

    // The transition is normalized by the largest closest-point distance attainable over the coarsest-level cells, NOT by the distance to the farthest tile corner. The cell owning that corner has its closest point at
    // its inner corner, which is nearer than the corner itself by roughly one cell diagonal, so normalizing by the corner distance would leave every cell short of the far end of the transition and force all of them to
    // subdivide at least once. That pins the achieved maximum sub-patch size at half of the requested one.
    layout.r_edge = 0.f;
    for (int j = 0; j < layout.base_subdiv.y; j++) {
        for (int i = 0; i < layout.base_subdiv.x; i++) {
            const float x0 = -0.5f * size.x + float(i) * layout.base_cell_size.x;
            const float y0 = -0.5f * size.y + float(j) * layout.base_cell_size.y;
            layout.r_edge = std::max(layout.r_edge, adaptiveTileCellDistance(layout.target, x0, y0, layout.base_cell_size.x, layout.base_cell_size.y));
        }
    }

    // Only possible when the base grid is a single cell containing the target, in which case there is no distance over which to transition.
    if (layout.r_edge <= 0.f) {
        layout.max_level = 0;
    }

    return layout;
}

//! Walk the adaptive tile quadtree and collect its leaf cells
/**
 * A cell is subdivided while its refinement level is below the level desired at its own distance from the target. The desired level is expressed directly in level space rather than as a desired sub-patch size, so that
 * the two ends of the transition land exactly on the coarsest and finest levels the grid can actually represent. Expressing it as a size instead would compare against a base cell size that, after rounding the grid to
 * a whole number of cells, may be slightly larger than the requested maximum, which would again force every cell to subdivide once.
 * \param[in] layout Quadtree grid description.
 * \param[in] size Size of the tile in the x- and y-directions.
 * \param[out] cells_out Leaf cells in traversal order, or null to count them without storing them.
 * \param[in] count_limit Traversal stops once this many leaf cells have been found.
 * \return Number of leaf cells, saturating at count_limit.
 */
static size_t traverseAdaptiveTileCells(const AdaptiveTileLayout &layout, const helios::vec2 &size, std::vector<AdaptiveTileCell> *cells_out, size_t count_limit) {
    std::vector<AdaptiveTileCell> stack;
    stack.reserve(size_t(layout.base_subdiv.x) * size_t(layout.base_subdiv.y) + 3 * size_t(layout.max_level) + 4);

    // Pushed in reverse so that the coarsest-level cells pop in row-major order.
    for (int j = layout.base_subdiv.y - 1; j >= 0; j--) {
        for (int i = layout.base_subdiv.x - 1; i >= 0; i--) {
            stack.push_back({i, j, 0, 0, 0});
        }
    }

    size_t cell_count = 0;
    while (!stack.empty()) {
        const AdaptiveTileCell cell = stack.back();
        stack.pop_back();

        float x0, y0, dx, dy;
        getAdaptiveTileCellBounds(layout, size, cell, x0, y0, dx, dy);

        const float distance = adaptiveTileCellDistance(layout.target, x0, y0, dx, dy);
        const float normalized_distance = (layout.r_edge > 0.f) ? std::min(distance / layout.r_edge, 1.f) : 1.f;
        const float level_desired = float(layout.max_level) * (1.f - std::pow(normalized_distance, layout.transition_exponent));

        if (cell.level < layout.max_level && float(cell.level) < level_desired) {
            // Pushed in reverse so that the children pop in the order (0,0), (1,0), (0,1), (1,1).
            for (int k = 3; k >= 0; k--) {
                stack.push_back({cell.base_i, cell.base_j, 2 * cell.local_i + uint(k & 1), 2 * cell.local_j + uint(k >> 1), cell.level + 1});
            }
            continue;
        }

        if (cells_out != nullptr) {
            cells_out->push_back(cell);
        }
        cell_count++;
        if (cell_count >= count_limit) {
            break;
        }
    }

    return cell_count;
}

//! Get the achieved minimum and maximum sub-patch edge length of a set of adaptive tile quadtree leaf cells
/**
 * Sub-patch size is taken as the longer of the two cell edges. The base grid rounds independently in each direction, so cells are square only when the tile dimensions happen to divide evenly; taking the longer edge
 * guarantees that both edges of every sub-patch are no larger than the reported size.
 */
static helios::vec2 getAdaptiveTileSubpatchSizeRange(const AdaptiveTileLayout &layout, const std::vector<AdaptiveTileCell> &cells) {
    int level_min = layout.max_level;
    int level_max = 0;
    for (const AdaptiveTileCell &cell: cells) {
        level_min = std::min(level_min, cell.level);
        level_max = std::max(level_max, cell.level);
    }

    const float base_size = std::max(layout.base_cell_size.x, layout.base_cell_size.y);
    return helios::make_vec2(base_size / float(1u << level_max), base_size / float(1u << level_min));
}

//! Check that the requested adaptive tile refinement parameters are usable, and raise an error explaining how to fix them if they are not
static void validateAdaptiveTileRefinement(const helios::vec2 &size, const helios::AdaptiveTileRefinement &refinement) {
    if (size.x <= 0.f || size.y <= 0.f) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Size of tile must be greater than 0.");
    } else if (!std::isfinite(refinement.target.x) || !std::isfinite(refinement.target.y)) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Refinement target must be finite.");
    } else if (refinement.subpatch_size_min <= 0.f || !std::isfinite(refinement.subpatch_size_min)) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Minimum sub-patch size must be greater than 0.");
    } else if (refinement.subpatch_size_max <= 0.f || !std::isfinite(refinement.subpatch_size_max)) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Maximum sub-patch size must be greater than 0.");
    } else if (refinement.subpatch_size_min > refinement.subpatch_size_max) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Minimum sub-patch size of " + std::to_string(refinement.subpatch_size_min) + " is greater than the maximum sub-patch size of " +
                             std::to_string(refinement.subpatch_size_max) + ".");
    } else if (refinement.transition_exponent <= 0.f || !std::isfinite(refinement.transition_exponent)) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Transition exponent must be greater than 0.");
    } else if (refinement.subpatch_size_max / refinement.subpatch_size_min > 16777216.f) {
        // Bounds the refinement level, and with it the shift used to index cells within a coarsest-level cell. The
        // sub-patch count limit does not cover this on its own: reaching a very deep level costs only a few cells per
        // level, so a chain deep enough to overflow the shift stays well under that limit.
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): The ratio of maximum to minimum sub-patch size is " + std::to_string(refinement.subpatch_size_max / refinement.subpatch_size_min) +
                             ", which exceeds the largest supported ratio of 16777216. Increase subpatch_size_min or decrease subpatch_size_max.");
    } else if (std::min(size.x, size.y) < 2.f * refinement.subpatch_size_max) {
        // Below this the base grid rounds to a single cell in one direction and the sizing heuristic breaks down: both ends of the requested range are then missed in the same direction, by as much as 60%.
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Maximum sub-patch size of " + std::to_string(refinement.subpatch_size_max) + " is too large for a tile of size " + std::to_string(size.x) + "x" +
                             std::to_string(size.y) + ". It must be no more than half of the smaller tile dimension.");
    }
}

//! Warn when an adaptive tile refinement target lies off the tile, since the finest requested sub-patch size will then not be reached anywhere
/**
 * This is a legal configuration rather than an error: refinement simply grades from whatever level the nearest corner of the tile reaches down to the coarsest, which is a reasonable thing to ask for when the object of
 * interest sits just outside the ground plane. It is worth warning about because the requested minimum sub-patch size is silently not achieved.
 */
static void warnIfAdaptiveTileTargetIsOutsideTile(const helios::vec2 &size, const helios::AdaptiveTileRefinement &refinement) {
    if (std::fabs(refinement.target.x) > 0.5f * size.x || std::fabs(refinement.target.y) > 0.5f * size.y) {
        std::cerr << "WARNING (Context::addAdaptiveTileObject): Refinement target (" << refinement.target.x << ", " << refinement.target.y << ") lies outside a tile of size " << size.x << "x" << size.y
                  << ". The requested minimum sub-patch size of " << refinement.subpatch_size_min << " will not be reached anywhere on the tile." << std::endl;
    }
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

    auto *tile_new = (new Tile(currentObjectID, UUID, subdiv, "", make_int2(1, 1), this));

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
    int2 repeat = correctTextureRepeat(subdiv, texture_repeat);

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

            // Set context pointer
            patch_new->context_ptr = this;

            // Create or reuse material with de-duplication
            std::string mat_label = generateMaterialLabel(make_RGBAcolor(0, 0, 0, 1), texturefile, false);
            if (!doesMaterialExist(mat_label)) {
                patch_new->materialID = addMaterial_internal(mat_label, make_RGBAcolor(0, 0, 0, 1), texturefile);
            } else {
                patch_new->materialID = getMaterialIDFromLabel(mat_label);
            }
            // Increment material reference count
            materials[patch_new->materialID].reference_count++;

            currentUUID++;
            UUID.push_back(currentUUID - 1);
        }
    }

    // Store the repeat that was requested, not the corrected local `repeat`. The correction is a function of the
    // subdivision count, so re-deriving it whenever the subdivisions change preserves the user's intent; storing
    // the corrected value instead would let the repeat decay monotonically across successive calls to
    // setTileObjectSubdivisionCount() and never recover.
    auto *tile_new = (new Tile(currentObjectID, UUID, subdiv, texturefile, texture_repeat, this));

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

uint Context::addAdaptiveTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const AdaptiveTileRefinement &refinement) {
    RGBcolor color(0.f, 0.75f, 0.f); // Default color is green

    return addAdaptiveTileObject(center, size, rotation, refinement, color);
}

uint Context::addAdaptiveTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const AdaptiveTileRefinement &refinement, const RGBcolor &color) {
    validateAdaptiveTileRefinement(size, refinement);

    const AdaptiveTileLayout layout = computeAdaptiveTileLayout(size, refinement, make_int2(1, 1));

    std::vector<AdaptiveTileCell> cells;
    if (traverseAdaptiveTileCells(layout, size, &cells, adaptive_tile_max_subpatch_count) >= adaptive_tile_max_subpatch_count) {
        cells.clear();
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): The requested refinement would generate at least " + std::to_string(adaptive_tile_max_subpatch_count) +
                             " sub-patches. Decrease transition_exponent (values below 1 reduce the count sharply), increase subpatch_size_min, or decrease the tile size.");
    }

    warnIfAdaptiveTileTargetIsOutsideTile(size, refinement);

    std::vector<uint> UUID;
    UUID.reserve(cells.size());

    for (const AdaptiveTileCell &cell: cells) {
        float x0, y0, dx, dy;
        getAdaptiveTileCellBounds(layout, size, cell, x0, y0, dx, dy);

        vec3 subcenter = make_vec3(x0 + 0.5f * dx, y0 + 0.5f * dy, 0.f);

        UUID.push_back(addPatch(subcenter, make_vec2(dx, dy), make_SphericalCoord(0, 0), color));

        if (rotation.elevation != 0.f) {
            getPrimitivePointer_private(UUID.back())->rotate(-rotation.elevation, "x");
        }
        if (rotation.azimuth != 0.f) {
            getPrimitivePointer_private(UUID.back())->rotate(-rotation.azimuth, "z");
        }
        getPrimitivePointer_private(UUID.back())->translate(center);
    }

    auto *tile_new = (new AdaptiveTile(currentObjectID, UUID, refinement, layout.base_subdiv, uint(layout.max_level), getAdaptiveTileSubpatchSizeRange(layout, cells), "", make_int2(1, 1), this));

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

uint Context::addAdaptiveTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const AdaptiveTileRefinement &refinement, const char *texturefile) {
    return addAdaptiveTileObject(center, size, rotation, refinement, texturefile, make_int2(1, 1));
}

uint Context::addAdaptiveTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const AdaptiveTileRefinement &refinement, const char *texturefile, const int2 &texture_repeat) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Texture file " + std::string(texturefile) + " does not exist.");
    } else if (texture_repeat.x < 1 || texture_repeat.y < 1) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Number of texture repeats must be greater than 0.");
    }

    validateAdaptiveTileRefinement(size, refinement);

    const AdaptiveTileLayout layout = computeAdaptiveTileLayout(size, refinement, texture_repeat);

    // Number of finest-level cells spanning a single repeat of the texture image. The (u,v) window of a sub-patch is a whole number of these, so this is also the denominator of every texture coordinate the tile
    // produces. Beyond 2^24 a float can no longer represent consecutive integers, at which point adjacent sub-patches would start sharing texture coordinates.
    // Deliberately long long rather than long: long is 32 bits on Windows, where the shift below would overflow before
    // the comparison could catch it, silently defeating the guard it exists to enforce.
    const long long finest_per_repeat_x = static_cast<long long>(layout.base_subdiv.x / texture_repeat.x) << layout.max_level;
    const long long finest_per_repeat_y = static_cast<long long>(layout.base_subdiv.y / texture_repeat.y) << layout.max_level;
    if (finest_per_repeat_x > 16777216LL || finest_per_repeat_y > 16777216LL) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): The requested refinement subdivides a single texture repeat into more sub-patches than can be assigned distinct texture coordinates. Increase "
                             "subpatch_size_min or decrease the texture repeat count.");
    }

    addTexture(texturefile);

    const int2 &sz = textures.at(texturefile).getImageResolution();
    if (finest_per_repeat_x >= sz.x || finest_per_repeat_y >= sz.y) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): The resolution of the texture image '" + std::string(texturefile) + "' (" + std::to_string(sz.x) + "x" + std::to_string(sz.y) +
                             ") is lower than the finest adaptive sub-patch resolution of " + std::to_string(finest_per_repeat_x) + "x" + std::to_string(finest_per_repeat_y) +
                             " sub-patches per texture repeat. Increase the resolution of the texture image, increase subpatch_size_min, or decrease the texture repeat count.");
    }

    std::vector<AdaptiveTileCell> cells;
    if (traverseAdaptiveTileCells(layout, size, &cells, adaptive_tile_max_subpatch_count) >= adaptive_tile_max_subpatch_count) {
        cells.clear();
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): The requested refinement would generate at least " + std::to_string(adaptive_tile_max_subpatch_count) +
                             " sub-patches. Decrease transition_exponent (values below 1 reduce the count sharply), increase subpatch_size_min, or decrease the tile size.");
    }

    warnIfAdaptiveTileTargetIsOutsideTile(size, refinement);

    // Each sub-patch is assigned its own (u,v) window of the texture image, so unlike a uniform tile there is no small set of windows to reuse. Every window therefore misses the solid-fraction cache and the texture
    // mask has to be evaluated from scratch. Opaque images short-circuit that evaluation, but an image with an alpha channel does not.
    if (cells.size() > 10000 && textures.at(texturefile).hasTransparencyChannel()) {
        std::cerr << "WARNING (Context::addAdaptiveTileObject): Texture file " << texturefile << " has a transparency channel, and this adaptive tile has " << cells.size()
                  << " sub-patches each requiring its own texture mask evaluation. This may take a long time. Consider using an opaque texture image for large adaptive tiles." << std::endl;
    }

    const int2 base_per_repeat = make_int2(layout.base_subdiv.x / texture_repeat.x, layout.base_subdiv.y / texture_repeat.y);

    std::vector<uint> UUID;
    UUID.reserve(cells.size());

    std::vector<helios::vec2> uv(4);

    for (const AdaptiveTileCell &cell: cells) {
        float x0, y0, dx, dy;
        getAdaptiveTileCellBounds(layout, size, cell, x0, y0, dx, dy);

        vec3 subcenter = make_vec3(x0 + 0.5f * dx, y0 + 0.5f * dy, 0.f);

        // Texture coordinates are built from exact integers and divided once, rather than multiplied by a precomputed reciprocal. A reciprocal multiply does not always reproduce the quotient exactly, which would leave
        // the shared edge between two neighboring sub-patches with two slightly different texture coordinates.
        const uint cells_per_repeat_x = uint(base_per_repeat.x) << cell.level;
        const uint cells_per_repeat_y = uint(base_per_repeat.y) << cell.level;
        const uint index_x = uint(cell.base_i % base_per_repeat.x) * (1u << cell.level) + cell.local_i;
        const uint index_y = uint(cell.base_j % base_per_repeat.y) * (1u << cell.level) + cell.local_j;

        const float u0 = float(index_x) / float(cells_per_repeat_x);
        const float u1 = float(index_x + 1) / float(cells_per_repeat_x);
        const float v0 = float(index_y) / float(cells_per_repeat_y);
        const float v1 = float(index_y + 1) / float(cells_per_repeat_y);

        uv.at(0) = make_vec2(u0, v0);
        uv.at(1) = make_vec2(u1, v0);
        uv.at(2) = make_vec2(u1, v1);
        uv.at(3) = make_vec2(u0, v1);

        auto *patch_new = (new Patch(texturefile, uv, textures, 0, currentUUID));

        patch_new->scale(make_vec3(dx, dy, 1));

        patch_new->translate(subcenter);

        if (rotation.elevation != 0) {
            patch_new->rotate(-rotation.elevation, "x");
        }
        if (rotation.azimuth != 0) {
            patch_new->rotate(-rotation.azimuth, "z");
        }

        patch_new->translate(center);

        primitives[currentUUID] = patch_new;

        // Set context pointer
        patch_new->context_ptr = this;

        // Create or reuse material with de-duplication
        std::string mat_label = generateMaterialLabel(make_RGBAcolor(0, 0, 0, 1), texturefile, false);
        if (!doesMaterialExist(mat_label)) {
            patch_new->materialID = addMaterial_internal(mat_label, make_RGBAcolor(0, 0, 0, 1), texturefile);
        } else {
            patch_new->materialID = getMaterialIDFromLabel(mat_label);
        }
        // Increment material reference count
        materials[patch_new->materialID].reference_count++;

        currentUUID++;
        UUID.push_back(currentUUID - 1);
    }

    auto *tile_new = (new AdaptiveTile(currentObjectID, UUID, refinement, layout.base_subdiv, uint(layout.max_level), getAdaptiveTileSubpatchSizeRange(layout, cells), texturefile, texture_repeat, this));

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

uint Context::addAdaptiveTileObject_fromPrimitives(const std::vector<uint> &UUIDs, const AdaptiveTileRefinement &refinement, const int2 &base_subdiv, uint max_level, const vec2 &subpatch_size_range,
                                                   const char *texturefile, const int2 &texture_repeat) {
    if (UUIDs.empty()) {
        helios_runtime_error("ERROR (Context::addAdaptiveTileObject): Cannot build an adaptive tile object from an empty set of primitives.");
    }

    auto *tile_new = (new AdaptiveTile(currentObjectID, UUIDs, refinement, base_subdiv, max_level, subpatch_size_range, texturefile, texture_repeat, this));

    for (uint p: UUIDs) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = tile_new;
    currentObjectID++;

    // Recovered from the adopted primitives rather than left at the origin, so that rotateObject() and
    // scaleObjectAboutPoint() act about the tile's own center after a reload as they do before one.
    uint objID = currentObjectID - 1;
    tile_new->object_origin = getObjectCenter(objID);

    return objID;
}

size_t Context::predictAdaptiveTileObjectSubpatchCount(const vec2 &size, const AdaptiveTileRefinement &refinement, const int2 &texture_repeat) const {
    validateAdaptiveTileRefinement(size, refinement);

    if (texture_repeat.x < 1 || texture_repeat.y < 1) {
        helios_runtime_error("ERROR (Context::predictAdaptiveTileObjectSubpatchCount): Number of texture repeats must be greater than 0.");
    }

    const AdaptiveTileLayout layout = computeAdaptiveTileLayout(size, refinement, texture_repeat);

    const size_t cell_count = traverseAdaptiveTileCells(layout, size, nullptr, adaptive_tile_max_subpatch_count);

    // Reporting the limit itself as though it were the true count would say that a refinement is viable when building
    // it is going to be rejected, which defeats the purpose of asking in advance. Raise the same error the builder
    // would instead, so that both answer the question the same way.
    if (cell_count >= adaptive_tile_max_subpatch_count) {
        helios_runtime_error("ERROR (Context::predictAdaptiveTileObjectSubpatchCount): The requested refinement would generate at least " + std::to_string(adaptive_tile_max_subpatch_count) +
                             " sub-patches. Decrease transition_exponent (values below 1 reduce the count sharply), increase subpatch_size_min, or decrease the tile size.");
    }

    return cell_count;
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
            // Apply transformations to all triangles added in this iteration (one for the center ring, two for outer rings)
            int start_idx = (r == 0) ? i : i - 1;
            for (int tri_idx = start_idx; tri_idx <= i; tri_idx++) {
                getPrimitivePointer_private(UUID.at(tri_idx))->rotate(rotation.elevation, "y");
                getPrimitivePointer_private(UUID.at(tri_idx))->rotate(rotation.azimuth, "z");
                getPrimitivePointer_private(UUID.at(tri_idx))->translate(center);
            }

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

    // The file loaders build a polymesh object themselves so that the mesh topology can be retained, and callers conventionally wrap the returned UUIDs in addPolymeshObject(). Recognize that case and
    // return the existing object rather than warning about primitives that already have a parent and then building an empty object from what is left.
    const uint first_parent_ObjID = getPrimitivePointer_private(UUIDs.front())->getParentObjectID();
    if (first_parent_ObjID != 0 && doesObjectExist(first_parent_ObjID) && objects.at(first_parent_ObjID)->getObjectType() == OBJECT_TYPE_POLYMESH) {
        bool all_belong_to_same_polymesh = true;
        for (uint UUID: UUIDs) {
            if (getPrimitivePointer_private(UUID)->getParentObjectID() != first_parent_ObjID) {
                all_belong_to_same_polymesh = false;
                break;
            }
        }
        if (all_belong_to_same_polymesh) {
            return first_parent_ObjID;
        }
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
    if (UUIDs_polymesh.empty()) {
        helios_runtime_error("ERROR (Context::addPolymeshObject): All of the provided primitives already belong to another object, so no polymesh object could be created. Detach the primitives from their "
                             "current object, or pass the object ID of the existing object instead.");
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
        makeRotationMatrix(rotation_radians, "z", Rz);
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
        // Order-preserving erase (rather than swap-and-pop) so the remaining UUIDs keep their relative
        // order. Because sub-primitives are created with ascending UUIDs, this keeps the vector sorted,
        // which writeXML() relies on to preserve sub-primitive ordering across a round-trip. The linear
        // find() above already makes this O(n), so erase() adds no asymptotic cost.
        UUIDs.erase(it);
        primitivesarecomplete = false;
        onChildPrimitiveDeleted(UUID);
    }
}

void CompoundObject::deleteChildPrimitive(const std::vector<uint> &a_UUIDs) {
    // Erase the members first and repair once for the whole batch, rather than repairing after each individual erase. For a derived type whose repair touches the entire object (such as the face table of a
    // Polymesh) the per-primitive form is quadratic in the number of primitives deleted.
    std::vector<uint> erased_UUIDs;
    erased_UUIDs.reserve(a_UUIDs.size());
    for (uint UUID: a_UUIDs) {
        auto it = find(UUIDs.begin(), UUIDs.end(), UUID);
        if (it != UUIDs.end()) {
            UUIDs.erase(it);
            primitivesarecomplete = false;
            erased_UUIDs.push_back(UUID);
        }
    }
    if (!erased_UUIDs.empty()) {
        onChildPrimitivesDeleted(erased_UUIDs);
    }
}

bool CompoundObject::arePrimitivesComplete() const {
    return primitivesarecomplete;
}

std::vector<std::vector<helios::vec3>> CompoundObject::getPrimitiveVertexNormals(const std::vector<uint> &UUIDs_query) const {
    // The base implementation simply repeats the single-primitive form. Types whose evaluation needs per-object preparation override this so that the preparation is done once for the whole batch.
    std::vector<std::vector<helios::vec3>> normals;
    normals.reserve(UUIDs_query.size());
    for (uint UUID: UUIDs_query) {
        normals.push_back(getPrimitiveVertexNormals(UUID));
    }
    return normals;
}

// ============== TILE CLASS METHOD DEFINITIONS ==============

Tile::Tile(uint a_OID, const std::vector<uint> &a_UUIDs, const int2 &a_subdiv, const char *a_texturefile, const int2 &a_texture_repeat, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_TILE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texture_repeat = a_texture_repeat;
    texturefile = a_texturefile;
    context = a_context;
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

helios::int2 Tile::getTextureRepeat() const {
    return texture_repeat;
}

helios::int2 Tile::getEffectiveTextureRepeat() const {
    return correctTextureRepeat(subdiv, texture_repeat);
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

// ============== ADAPTIVE TILE CLASS METHOD DEFINITIONS ==============

AdaptiveTile::AdaptiveTile(uint a_OID, const std::vector<uint> &a_UUIDs, const AdaptiveTileRefinement &a_refinement, const helios::int2 &a_base_subdiv, uint a_max_level, const helios::vec2 &a_subpatch_size_range,
                           const char *a_texturefile, const helios::int2 &a_texture_repeat, helios::Context *a_context) {
    makeIdentityMatrix(transform);

    OID = a_OID;
    type = helios::OBJECT_TYPE_ADAPTIVE_TILE;
    UUIDs = a_UUIDs;
    refinement = a_refinement;
    base_subdiv = a_base_subdiv;
    max_level = a_max_level;
    subpatch_size_range = a_subpatch_size_range;
    texture_repeat = a_texture_repeat;
    texturefile = a_texturefile;
    context = a_context;
}

helios::vec2 AdaptiveTile::getSize() const {
    const std::vector<vec3> &vertices = getVertices();
    float l = (vertices.at(1) - vertices.at(0)).magnitude();
    float w = (vertices.at(3) - vertices.at(0)).magnitude();
    return make_vec2(l, w);
}

vec3 AdaptiveTile::getCenter() const {
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

std::vector<helios::vec3> AdaptiveTile::getVertices() const {
    std::vector<helios::vec3> vertices;
    vertices.resize(4);

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

vec3 AdaptiveTile::getNormal() const {
    return context->getPrimitiveNormal(UUIDs.front());
}

std::vector<helios::vec2> AdaptiveTile::getTextureUV() const {
    return {make_vec2(0, 0), make_vec2(1, 0), make_vec2(1, 1), make_vec2(0, 1)};
}

AdaptiveTileRefinement AdaptiveTile::getRefinement() const {
    return refinement;
}

helios::int2 AdaptiveTile::getBaseSubdivisionCount() const {
    return base_subdiv;
}

uint AdaptiveTile::getMaxRefinementLevel() const {
    return max_level;
}

helios::vec2 AdaptiveTile::getSubpatchSizeRange() const {
    return subpatch_size_range;
}

helios::int2 AdaptiveTile::getTextureRepeat() const {
    return texture_repeat;
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

helios::vec3 Sphere::evaluateSurfaceNormal(const helios::vec3 &point, const float (&normal_matrix)[9]) const {
    // The sphere's transformation matrix is exactly the affine map that carries the unit sphere onto the object's geometry, so the point is first carried back to the unit sphere, where the outward normal at
    // a point is simply the point itself. That local normal is then carried to the global frame by the normal matrix. Working through the unit sphere this way keeps the result exact for an ellipsoid that has
    // also been rotated, which a componentwise division by the squared semi-axes in the global frame would not.
    const vec3 shifted = make_vec3(point.x - transform[3], point.y - transform[7], point.z - transform[11]);

    // The normal matrix is the inverse transpose of the linear part, so reading it transposed gives the inverse of the linear part itself.
    vec3 local;
    local.x = normal_matrix[0] * shifted.x + normal_matrix[3] * shifted.y + normal_matrix[6] * shifted.z;
    local.y = normal_matrix[1] * shifted.x + normal_matrix[4] * shifted.y + normal_matrix[7] * shifted.z;
    local.z = normal_matrix[2] * shifted.x + normal_matrix[5] * shifted.y + normal_matrix[8] * shifted.z;

    vec3 normal = transformNormalByMatrix(normal_matrix, local);
    const float magnitude = normal.magnitude();
    return (magnitude > 0.f) ? normal / magnitude : normal;
}

std::vector<helios::vec3> Sphere::getPrimitiveVertexNormals(uint UUID) const {
    float N[9];
    if (!makeNormalMatrix(transform, N)) {
        // A singular transformation means the sphere has been flattened along at least one axis and has no meaningful surface normal. Returning nothing lets the caller keep the face normal rather than
        // preventing the object from being drawn at all.
        return {};
    }

    const std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);

    std::vector<vec3> normals;
    normals.reserve(vertices.size());
    for (const vec3 &vertex: vertices) {
        normals.push_back(evaluateSurfaceNormal(vertex, N));
    }
    return normals;
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

    int second_last = node_count - 2;
    int last = node_count - 1;

    // The UUID list is ordered with the radial slot varying slowest and the segment index fastest, and updateTriangleVertices() relies on that when it assigns geometry to primitives by position. The new
    // segment's triangles therefore have to be inserted into each radial slot's block rather than appended to the end, otherwise every primitive is subsequently repainted with some other primitive's
    // geometry. The vertices would still describe a valid tube, but anything held per primitive rather than per vertex, the color in particular, would end up on the wrong part of the tube.
    const int new_segment_count = node_count - 1;
    std::vector<uint> reordered_UUIDs;
    reordered_UUIDs.reserve(2 * new_segment_count * radial_subdivisions);

    int existing_index = 0;
    for (int j = 0; j < radial_subdivisions; j++) {

        // The segments that were already present keep their existing primitives, in their existing order.
        for (int i = 0; i < new_segment_count - 1; i++) {
            reordered_UUIDs.push_back(UUIDs.at(existing_index));
            reordered_UUIDs.push_back(UUIDs.at(existing_index + 1));
            existing_index += 2;
        }

        vec3 v0 = triangle_vertices.at(second_last).at(j);
        vec3 v1 = triangle_vertices.at(last).at(j + 1);
        vec3 v2 = triangle_vertices.at(second_last).at(j + 1);

        reordered_UUIDs.push_back(context->addTriangle(v0, v1, v2, node_color));

        v0 = triangle_vertices.at(second_last).at(j);
        v1 = triangle_vertices.at(last).at(j);
        v2 = triangle_vertices.at(last).at(j + 1);

        reordered_UUIDs.push_back(context->addTriangle(v0, v1, v2, node_color));
    }

    UUIDs = reordered_UUIDs;

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

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;

    // The new segment's triangles are inserted into each radial slot's block rather than appended to the end, because updateTriangleVertices() assigns geometry to primitives by position and expects the
    // radial slot to vary slowest and the segment index fastest. See the color overload of this method for what goes wrong otherwise.
    const int new_segment_count = node_count - 1;
    std::vector<uint> reordered_UUIDs;
    reordered_UUIDs.reserve(2 * new_segment_count * radial_subdivisions);

    int existing_index = 0;
    for (int j = 0; j < radial_subdivisions; j++) {

        for (int i = 0; i < new_segment_count - 1; i++) {
            reordered_UUIDs.push_back(UUIDs.at(existing_index));
            reordered_UUIDs.push_back(UUIDs.at(existing_index + 1));
            existing_index += 2;
        }

        v0 = triangle_vertices[node_count - 2][j];
        v1 = triangle_vertices[node_count - 1][j + 1];
        v2 = triangle_vertices[node_count - 2][j + 1];

        uv0 = uv[0][j];
        uv1 = uv[1][j + 1];
        uv2 = uv[0][j + 1];

        reordered_UUIDs.push_back(context->addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));

        v0 = triangle_vertices[node_count - 2][j];
        v1 = triangle_vertices[node_count - 1][j];
        v2 = triangle_vertices[node_count - 1][j + 1];

        uv0 = uv[0][j];
        uv1 = uv[1][j];
        uv2 = uv[1][j + 1];

        reordered_UUIDs.push_back(context->addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
    }

    UUIDs = reordered_UUIDs;

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

    nodes = node_xyz;

    recomputeCrossSections();
    updateTriangleVertices();
}

void Tube::pruneTubeNodes(uint node_index) {
    if (node_index >= nodes.size()) {
        helios_runtime_error("ERROR (Tube::pruneTubeNodes): Node index of " + std::to_string(node_index) + " is out of bounds.");
    }

    if (node_index <= 1) {
        // 0 or 1 remaining nodes means 0 segments — delete the entire object.
        // NOTE: this deletes 'this', so we must return immediately after.
        context->deleteObject(this->OID);
        return;
    }

    int original_segment_count = (int)nodes.size() - 1;
    int segments_to_keep = (int)node_index - 1;

    // Partition UUIDs into kept vs. to-delete. UUIDs are in j-outer, i-inner order.
    std::vector<uint> kept_UUIDs;
    std::vector<uint> uuids_to_delete;
    int ii = 0;
    for (int j = 0; j < subdiv; j++) {
        for (int i = 0; i < original_segment_count; i++) {
            if (i < segments_to_keep) {
                kept_UUIDs.push_back(UUIDs.at(ii));
                kept_UUIDs.push_back(UUIDs.at(ii + 1));
            } else {
                uuids_to_delete.push_back(UUIDs.at(ii));
                uuids_to_delete.push_back(UUIDs.at(ii + 1));
            }
            ii += 2;
        }
    }

    // Update object data first
    nodes.erase(nodes.begin() + node_index, nodes.end());
    triangle_vertices.erase(triangle_vertices.begin() + node_index, triangle_vertices.end());
    radius.erase(radius.begin() + node_index, radius.end());
    colors.erase(colors.begin() + node_index, colors.end());
    UUIDs = kept_UUIDs;

    // Delete removed primitives. Since we already removed them from UUIDs,
    // deleteChildPrimitive won't find them and won't trigger object auto-deletion.
    for (uint uuid : uuids_to_delete) {
        context->deletePrimitive(uuid);
    }
}

void Tube::recomputeCrossSections() {
    int node_count = (int)nodes.size();
    uint radial_subdivisions = subdiv;

    // Clamp very small radii to avoid creating degenerate triangles
    const float min_radius_threshold = 1e-5f;
    std::vector<float> radius_clamped = radius;
    for (int i = 0; i < node_count; i++) {
        if (radius_clamped[i] < min_radius_threshold && radius_clamped[i] >= 0) {
            radius_clamped[i] = min_radius_threshold;
        }
    }

    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    vec3 axial_vector;
    vec3 initial_radial(1.0f, 0.0f, 0.0f);
    vec3 previous_axial_vector;
    vec3 previous_radial_dir;

    for (int i = 0; i < node_count; i++) {
        if (i == 0) {
            axial_vector = nodes[i + 1] - nodes[i];
            float mag = axial_vector.magnitude();
            if (mag < 1e-6f) {
                axial_vector = make_vec3(0, 0, 1);
            } else {
                axial_vector = axial_vector / mag;
            }
            if (fabs(axial_vector * initial_radial) > 0.95f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f);
            }
            if (fabs(axial_vector.z) > 0.95f) {
                initial_radial = vec3(1.0f, 0.0f, 0.0f);
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

            vec3 rotation_axis = cross(previous_axial_vector, axial_vector);
            if (rotation_axis.magnitude() > 1e-5) {
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
            } else {
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
            triangle_vertices[i][j] = nodes[i] + cfact[j] * radius_clamped[i] * radial_dir + sfact[j] * radius_clamped[i] * orthogonal_dir;
        }
    }
}

void Tube::updateTriangleVertices() const {
    int ii = 0;
    for (int j = 0; j < subdiv; j++) {
        for (int i = 0; i < nodes.size() - 1; i++) {
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

Tube::NormalFrame Tube::buildNormalFrame() const {
    NormalFrame frame;

    frame.nodes_global = getNodes();
    frame.radii_global = getNodeRadii();

    const size_t node_count = frame.nodes_global.size();
    frame.axis.resize(node_count);
    frame.taper_slope.resize(node_count);
    frame.cumulative_arclength.resize(node_count);

    // Arclength along the node polyline, used below to narrow the search for the ring a surface point belongs to.
    frame.cumulative_arclength.at(0) = 0.f;
    for (size_t i = 1; i < node_count; i++) {
        frame.cumulative_arclength.at(i) = frame.cumulative_arclength.at(i - 1) + (frame.nodes_global.at(i) - frame.nodes_global.at(i - 1)).magnitude();
    }

    for (size_t i = 0; i < node_count; i++) {

        // The axial direction uses the same one-sided ends and averaged interior as the cross-section construction in recomputeCrossSections(), so that the normal frame agrees with the geometry frame.
        vec3 axial_vector;
        if (i == 0) {
            axial_vector = frame.nodes_global.at(1) - frame.nodes_global.at(0);
        } else if (i == node_count - 1) {
            axial_vector = frame.nodes_global.at(i) - frame.nodes_global.at(i - 1);
        } else {
            axial_vector = 0.5f * ((frame.nodes_global.at(i) - frame.nodes_global.at(i - 1)) + (frame.nodes_global.at(i + 1) - frame.nodes_global.at(i)));
        }
        const float axial_magnitude = axial_vector.magnitude();
        // Matches the degenerate-segment substitution made when the cross-sections were built.
        frame.axis.at(i) = (axial_magnitude < 1e-6f) ? make_vec3(0, 0, 1) : axial_vector / axial_magnitude;

        // Rate of change of radius with respect to arclength. A tapering tube's surface is a cone rather than a cylinder, so its normal tilts toward the narrowing end by this slope; leaving it out would
        // shade a tapered stem as though it were a perfect cylinder. Differentiation is with respect to arclength rather than node index so that unevenly spaced nodes give the correct slope.
        float slope_sum = 0.f;
        int slope_count = 0;
        if (i > 0) {
            const float dl = frame.cumulative_arclength.at(i) - frame.cumulative_arclength.at(i - 1);
            if (dl > 1e-6f) {
                slope_sum += (frame.radii_global.at(i) - frame.radii_global.at(i - 1)) / dl;
                slope_count++;
            }
        }
        if (i + 1 < node_count) {
            const float dl = frame.cumulative_arclength.at(i + 1) - frame.cumulative_arclength.at(i);
            if (dl > 1e-6f) {
                slope_sum += (frame.radii_global.at(i + 1) - frame.radii_global.at(i)) / dl;
                slope_count++;
            }
        }
        frame.taper_slope.at(i) = (slope_count > 0) ? slope_sum / float(slope_count) : 0.f;
    }

    return frame;
}

helios::vec3 Tube::evaluateSurfaceNormal(const helios::vec3 &point, const NormalFrame &frame) const {
    const size_t node_count = frame.nodes_global.size();

    // Identify the ring the point belongs to. Every vertex of the tube lies exactly one node radius from its own node, so the residual between the measured distance and the stored radius is zero for the
    // correct ring and of the order of the node spacing for any other. That discriminates far more sharply than proximity alone, which can pick the wrong node on the inside of a tight bend.
    size_t ring = 0;
    float best_residual = -1.f;
    for (size_t i = 0; i < node_count; i++) {
        const float residual = std::abs((point - frame.nodes_global.at(i)).magnitude() - frame.radii_global.at(i));
        if (best_residual < 0.f || residual < best_residual) {
            best_residual = residual;
            ring = i;
        }
    }

    const vec3 &axis = frame.axis.at(ring);

    // Remove any axial component so that the radial direction is measured strictly perpendicular to the tube axis.
    vec3 radial = point - frame.nodes_global.at(ring);
    radial = radial - (radial * axis) * axis;

    const float radial_magnitude = radial.magnitude();
    if (radial_magnitude < 1e-9f) {
        // The point sits on the axis itself, where the surface has pinched to nothing and no radial direction exists.
        return axis;
    }

    vec3 normal = radial / radial_magnitude - frame.taper_slope.at(ring) * axis;
    const float magnitude = normal.magnitude();
    return (magnitude > 0.f) ? normal / magnitude : radial / radial_magnitude;
}

std::vector<helios::vec3> Tube::getPrimitiveVertexNormals(uint UUID) const {
    if (nodes.size() < 2) {
        return {};
    }
    const NormalFrame frame = buildNormalFrame();

    const std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);

    std::vector<vec3> normals;
    normals.reserve(vertices.size());
    for (const vec3 &vertex: vertices) {
        normals.push_back(evaluateSurfaceNormal(vertex, frame));
    }
    return normals;
}

std::vector<std::vector<helios::vec3>> Tube::getPrimitiveVertexNormals(const std::vector<uint> &UUIDs_query) const {
    if (nodes.size() < 2) {
        return std::vector<std::vector<helios::vec3>>(UUIDs_query.size());
    }

    // Building the frame costs one pass over the tube nodes, so it is done once here rather than once per primitive.
    const NormalFrame frame = buildNormalFrame();

    std::vector<std::vector<helios::vec3>> normals;
    normals.reserve(UUIDs_query.size());
    for (uint UUID: UUIDs_query) {
        const std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);
        std::vector<vec3> primitive_normals;
        primitive_normals.reserve(vertices.size());
        for (const vec3 &vertex: vertices) {
            primitive_normals.push_back(evaluateSurfaceNormal(vertex, frame));
        }
        normals.push_back(primitive_normals);
    }
    return normals;
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

namespace {

    //! Transform a point in object-local coordinates to global coordinates using a Helios 4x4 affine transformation matrix
    helios::vec3 transformPointByMatrix(const float (&T)[16], const helios::vec3 &point) {
        helios::vec3 result;
        result.x = T[0] * point.x + T[1] * point.y + T[2] * point.z + T[3];
        result.y = T[4] * point.x + T[5] * point.y + T[6] * point.z + T[7];
        result.z = T[8] * point.x + T[9] * point.y + T[10] * point.z + T[11];
        return result;
    }

    //! Build a canonically-ordered key identifying the undirected edge between two vertex indices
    /**
     * The two 32-bit indices are packed into a single 64-bit value so that edges can be counted in a hash map without needing a hash functor for std::pair.
     */
    int64_t makeEdgeKey(int vertex_a, int vertex_b) {
        const int64_t lower = (vertex_a < vertex_b) ? vertex_a : vertex_b;
        const int64_t upper = (vertex_a < vertex_b) ? vertex_b : vertex_a;
        return (lower << 32) | (upper & 0xffffffffLL);
    }

} // namespace

void Polymesh::setTopology(const std::vector<helios::vec3> &a_vertices, const std::vector<helios::int3> &a_faces, const std::vector<uint> &a_face_UUIDs, const std::vector<helios::vec3> &a_vertex_normals,
                           const std::vector<helios::vec2> &a_vertex_uv, helios::VertexNormalSource a_normal_source) {

    if (a_faces.size() != a_face_UUIDs.size()) {
        helios_runtime_error("ERROR (Polymesh::setTopology): The face table has " + std::to_string(a_faces.size()) + " entries but " + std::to_string(a_face_UUIDs.size()) +
                             " primitive UUIDs were given. Every face must correspond to exactly one primitive.");
    }
    if (!a_vertex_normals.empty() && a_vertex_normals.size() != a_vertices.size()) {
        helios_runtime_error("ERROR (Polymesh::setTopology): " + std::to_string(a_vertex_normals.size()) + " vertex normals were given for " + std::to_string(a_vertices.size()) +
                             " vertices. Vertex normals must either be empty or parallel to the vertex array.");
    }
    if (!a_vertex_uv.empty() && a_vertex_uv.size() != a_vertices.size()) {
        helios_runtime_error("ERROR (Polymesh::setTopology): " + std::to_string(a_vertex_uv.size()) + " texture coordinates were given for " + std::to_string(a_vertices.size()) +
                             " vertices. Texture coordinates must either be empty or parallel to the vertex array.");
    }
    if (a_vertex_normals.empty() && a_normal_source != helios::NORMAL_SOURCE_NONE) {
        helios_runtime_error("ERROR (Polymesh::setTopology): A vertex normal source was specified but no vertex normals were given. Pass NORMAL_SOURCE_NONE when the mesh has no vertex normals.");
    }
    for (const helios::int3 &face: a_faces) {
        if (face.x < 0 || face.y < 0 || face.z < 0 || face.x >= scast<int>(a_vertices.size()) || face.y >= scast<int>(a_vertices.size()) || face.z >= scast<int>(a_vertices.size())) {
            helios_runtime_error("ERROR (Polymesh::setTopology): Face references vertex indices (" + std::to_string(face.x) + ", " + std::to_string(face.y) + ", " + std::to_string(face.z) +
                                 ") but the mesh has only " + std::to_string(a_vertices.size()) + " vertices. Check that all face indices are within range.");
        }
    }

    // Incoming geometry is in global coordinates; store it in the object-local frame so that it stays correct as the object transformation matrix is subsequently updated by translate/rotate/scale.
    float T_inverse[9];
    if (!makeNormalMatrix(transform, T_inverse)) {
        helios_runtime_error("ERROR (Polymesh::setTopology): The object transformation matrix is singular and cannot be inverted. The mesh topology cannot be expressed in the object-local frame.");
    }

    vertices.resize(a_vertices.size());
    for (size_t v = 0; v < a_vertices.size(); v++) {
        // The linear part of a Helios object transform is inverted via its transpose-inverse, so invert the translation separately and apply the true inverse linear map.
        const helios::vec3 shifted = make_vec3(a_vertices.at(v).x - transform[3], a_vertices.at(v).y - transform[7], a_vertices.at(v).z - transform[11]);
        vertices.at(v).x = T_inverse[0] * shifted.x + T_inverse[3] * shifted.y + T_inverse[6] * shifted.z;
        vertices.at(v).y = T_inverse[1] * shifted.x + T_inverse[4] * shifted.y + T_inverse[7] * shifted.z;
        vertices.at(v).z = T_inverse[2] * shifted.x + T_inverse[5] * shifted.y + T_inverse[8] * shifted.z;
    }

    // Normals are transformed to the local frame by the inverse of the inverse transpose, which is the transpose of the linear part itself.
    vertex_normals.resize(a_vertex_normals.size());
    for (size_t v = 0; v < a_vertex_normals.size(); v++) {
        const helios::vec3 &n = a_vertex_normals.at(v);
        helios::vec3 local_normal;
        local_normal.x = transform[0] * n.x + transform[4] * n.y + transform[8] * n.z;
        local_normal.y = transform[1] * n.x + transform[5] * n.y + transform[9] * n.z;
        local_normal.z = transform[2] * n.x + transform[6] * n.y + transform[10] * n.z;
        const float magnitude = local_normal.magnitude();
        vertex_normals.at(v) = (magnitude > 0.f) ? local_normal / magnitude : local_normal;
    }

    faces = a_faces;
    face_UUIDs = a_face_UUIDs;
    vertex_uv = a_vertex_uv;
    normal_source = a_normal_source;

    UUID_to_face.clear();
    for (size_t f = 0; f < face_UUIDs.size(); f++) {
        UUID_to_face[face_UUIDs.at(f)] = f;
    }

    invalidateAdjacency();
}

std::vector<helios::vec3> Polymesh::getVertices() const {
    std::vector<vec3> vertices_global(vertices.size());
    for (size_t v = 0; v < vertices.size(); v++) {
        vertices_global.at(v) = transformPointByMatrix(transform, vertices.at(v));
    }
    return vertices_global;
}

std::vector<helios::int3> Polymesh::getFaces() const {
    return faces;
}

std::vector<helios::vec3> Polymesh::getVertexNormals() const {
    if (vertex_normals.empty()) {
        return {};
    }

    float N[9];
    if (!makeNormalMatrix(transform, N)) {
        helios_runtime_error("ERROR (Polymesh::getVertexNormals): The object transformation matrix is singular, so surface normals are undefined. This usually means the object was scaled by zero along one or "
                             "more axes.");
    }

    std::vector<vec3> normals_global(vertex_normals.size());
    for (size_t v = 0; v < vertex_normals.size(); v++) {
        vec3 n = transformNormalByMatrix(N, vertex_normals.at(v));
        const float magnitude = n.magnitude();
        if (magnitude > 0.f) {
            n = n / magnitude;
        }
        normals_global.at(v) = n;
    }
    return normals_global;
}

std::vector<helios::vec2> Polymesh::getVertexUV() const {
    return vertex_uv;
}

bool Polymesh::hasVertexNormals() const {
    return !vertex_normals.empty();
}

helios::VertexNormalSource Polymesh::getVertexNormalSource() const {
    return normal_source;
}

size_t Polymesh::getVertexCount() const {
    return vertices.size();
}

size_t Polymesh::getFaceCount() const {
    return faces.size();
}

size_t Polymesh::getFaceIndexForPrimitive(uint UUID) const {
    auto it = UUID_to_face.find(UUID);
    if (it == UUID_to_face.end()) {
        helios_runtime_error("ERROR (Polymesh::getFaceIndexForPrimitive): Primitive UUID " + std::to_string(UUID) + " does not correspond to a face of polymesh object " + std::to_string(OID) +
                             ". Check that the primitive belongs to this object and that the object carries mesh topology.");
    }
    return it->second;
}

uint Polymesh::getPrimitiveUUIDForFace(size_t face_index) const {
    if (face_index >= face_UUIDs.size()) {
        helios_runtime_error("ERROR (Polymesh::getPrimitiveUUIDForFace): Face index " + std::to_string(face_index) + " is out of range. Polymesh object " + std::to_string(OID) + " has " +
                             std::to_string(face_UUIDs.size()) + " faces.");
    }
    return face_UUIDs.at(face_index);
}

void Polymesh::invalidateAdjacency() const {
    adjacency_valid = false;
    vertex_to_faces.clear();
}

void Polymesh::buildAdjacency() const {
    if (adjacency_valid) {
        return;
    }
    vertex_to_faces.assign(vertices.size(), {});
    for (size_t f = 0; f < faces.size(); f++) {
        vertex_to_faces.at(faces.at(f).x).push_back(f);
        vertex_to_faces.at(faces.at(f).y).push_back(f);
        vertex_to_faces.at(faces.at(f).z).push_back(f);
    }
    adjacency_valid = true;
}

void Polymesh::syncVerticesFromPrimitive(uint UUID) {
    auto it = UUID_to_face.find(UUID);
    if (it == UUID_to_face.end()) {
        // The object carries no topology, so there is nothing to keep in sync.
        return;
    }

    const std::vector<vec3> primitive_vertices = context->getPrimitiveVertices(UUID);
    if (primitive_vertices.size() != 3) {
        return;
    }

    // The face table is stored in the object-local frame, so convert the primitive's global vertices back through the object transform.
    float T_inverse[9];
    if (!makeNormalMatrix(transform, T_inverse)) {
        helios_runtime_error("ERROR (Polymesh::syncVerticesFromPrimitive): The object transformation matrix of polymesh object " + std::to_string(OID) +
                             " is singular and cannot be inverted, so the deformed geometry cannot be expressed in the object-local frame. This usually means the object was scaled by zero along one or more "
                             "axes.");
    }

    const int3 &face = faces.at(it->second);
    const int face_corner[3] = {face.x, face.y, face.z};
    for (int i = 0; i < 3; i++) {
        const vec3 shifted = make_vec3(primitive_vertices.at(i).x - transform[3], primitive_vertices.at(i).y - transform[7], primitive_vertices.at(i).z - transform[11]);
        vec3 &mesh_vertex = vertices.at(face_corner[i]);
        mesh_vertex.x = T_inverse[0] * shifted.x + T_inverse[3] * shifted.y + T_inverse[6] * shifted.z;
        mesh_vertex.y = T_inverse[1] * shifted.x + T_inverse[4] * shifted.y + T_inverse[7] * shifted.z;
        mesh_vertex.z = T_inverse[2] * shifted.x + T_inverse[5] * shifted.y + T_inverse[8] * shifted.z;
    }

    // The stored vertices are shared, so this update is already visible to every face that references them. The sibling primitives are deliberately NOT rewritten here: doing so would make a bulk transform
    // of the whole mesh quadratic (each facet rewriting its ~6 neighbours, which are then rewritten again as the loop reaches them) and order-dependent, since each rewrite would re-derive a neighbour from
    // vertices that other facets in the same batch are still updating. A caller that moves only some facets of a welded mesh therefore leaves the untouched neighbouring primitives where they were, even
    // though the shared mesh vertex has moved; transform the neighbouring primitives too if the rendered geometry must follow.

    // Vertex normals were computed for the undeformed geometry and no longer describe the surface. Leave them in place rather than silently recomputing them; computeVertexNormals() should be called again if
    // exact normals are needed after a deformation.

    // The face table itself is unchanged - only vertex positions moved - so the cached vertex-to-face adjacency remains valid and is deliberately not invalidated here. Rebuilding it once per transformed
    // primitive would dominate the cost of a bulk transform.
}

void Polymesh::removeFacesAndCompact(const std::vector<bool> &remove_face) {

    std::vector<int3> faces_kept;
    std::vector<uint> face_UUIDs_kept;
    faces_kept.reserve(faces.size());
    face_UUIDs_kept.reserve(face_UUIDs.size());
    for (size_t f = 0; f < faces.size(); f++) {
        if (!remove_face.at(f)) {
            faces_kept.push_back(faces.at(f));
            face_UUIDs_kept.push_back(face_UUIDs.at(f));
        }
    }
    faces = faces_kept;
    face_UUIDs = face_UUIDs_kept;

    UUID_to_face.clear();
    for (size_t f = 0; f < face_UUIDs.size(); f++) {
        UUID_to_face[face_UUIDs.at(f)] = f;
    }

    // Drop any vertex left unreferenced by the surviving faces and reindex the remainder so the topology stays compact and valid.
    std::vector<bool> vertex_is_referenced(vertices.size(), false);
    for (const int3 &face: faces) {
        vertex_is_referenced.at(face.x) = true;
        vertex_is_referenced.at(face.y) = true;
        vertex_is_referenced.at(face.z) = true;
    }

    std::vector<int> remapped_index(vertices.size(), -1);
    std::vector<vec3> vertices_kept;
    std::vector<vec3> vertex_normals_kept;
    std::vector<vec2> vertex_uv_kept;
    vertices_kept.reserve(vertices.size());
    for (size_t v = 0; v < vertices.size(); v++) {
        if (!vertex_is_referenced.at(v)) {
            continue;
        }
        remapped_index.at(v) = scast<int>(vertices_kept.size());
        vertices_kept.push_back(vertices.at(v));
        if (!vertex_normals.empty()) {
            vertex_normals_kept.push_back(vertex_normals.at(v));
        }
        if (!vertex_uv.empty()) {
            vertex_uv_kept.push_back(vertex_uv.at(v));
        }
    }

    for (int3 &face: faces) {
        face.x = remapped_index.at(face.x);
        face.y = remapped_index.at(face.y);
        face.z = remapped_index.at(face.z);
    }

    vertices = vertices_kept;
    vertex_normals = vertex_normals_kept;
    vertex_uv = vertex_uv_kept;

    if (vertices.empty()) {
        normal_source = helios::NORMAL_SOURCE_NONE;
    }

    invalidateAdjacency();
}

void Polymesh::onChildPrimitiveDeleted(uint UUID) {
    auto it = UUID_to_face.find(UUID);
    if (it == UUID_to_face.end()) {
        // The object carries no topology (e.g. it was created by grouping loose primitives), so there is nothing to repair.
        return;
    }

    std::vector<bool> remove_face(faces.size(), false);
    remove_face.at(it->second) = true;
    removeFacesAndCompact(remove_face);
}

void Polymesh::onChildPrimitivesDeleted(const std::vector<uint> &a_UUIDs) {
    if (UUID_to_face.empty()) {
        return;
    }

    // Mark every deleted primitive's face first, then repair once. Repairing per primitive would rebuild the UUID map and recompact the vertex array once for each deletion, which is quadratic in the size
    // of the mesh when a large number of primitives is deleted together.
    std::vector<bool> remove_face(faces.size(), false);
    bool any_removed = false;
    for (uint UUID: a_UUIDs) {
        auto it = UUID_to_face.find(UUID);
        if (it != UUID_to_face.end()) {
            remove_face.at(it->second) = true;
            any_removed = true;
        }
    }

    if (any_removed) {
        removeFacesAndCompact(remove_face);
    }
}

void Polymesh::computeVertexNormals(float crease_angle_degrees) {
    if (faces.empty()) {
        helios_runtime_error("ERROR (Polymesh::computeVertexNormals): Polymesh object " + std::to_string(OID) +
                             " carries no mesh topology, so vertex normals cannot be computed. Vertex normals are only available for meshes loaded from a file with retained connectivity.");
    }
    if (crease_angle_degrees < 0.f || crease_angle_degrees > 180.f) {
        helios_runtime_error("ERROR (Polymesh::computeVertexNormals): Crease angle of " + std::to_string(crease_angle_degrees) + " degrees is out of range. The crease angle must be between 0 and 180 degrees.");
    }

    const float crease_angle_radians = deg2rad(crease_angle_degrees);

    // Face normals and areas are computed in the local frame; the area weighting and the crease comparison are both invariant to the object transform being applied afterwards.
    std::vector<vec3> face_normals(faces.size());
    std::vector<float> face_areas(faces.size());
    for (size_t f = 0; f < faces.size(); f++) {
        const vec3 &v0 = vertices.at(faces.at(f).x);
        const vec3 &v1 = vertices.at(faces.at(f).y);
        const vec3 &v2 = vertices.at(faces.at(f).z);
        const vec3 face_cross = cross(v1 - v0, v2 - v0);
        const float cross_magnitude = face_cross.magnitude();
        face_areas.at(f) = 0.5f * cross_magnitude;
        // A degenerate (zero-area) face has no meaningful normal; it contributes nothing to the averaging because its weight is zero.
        face_normals.at(f) = (cross_magnitude > 0.f) ? face_cross / cross_magnitude : make_vec3(0, 0, 0);
    }

    buildAdjacency();

    // A vertex whose incident faces do not all lie within the crease angle of one another carries more than one normal, which an indexed face set cannot represent at a single index. Such a vertex is
    // therefore split into one copy per smooth group, and the faces in each group are repointed at their own copy. This is what keeps a crease genuinely hard rather than averaging across it.
    std::vector<vec3> split_vertices;
    std::vector<vec3> split_normals;
    std::vector<vec2> split_uv;
    split_vertices.reserve(vertices.size());
    split_normals.reserve(vertices.size());

    // For each face, the index its corner should refer to after splitting.
    std::vector<int3> updated_faces = faces;

    for (size_t v = 0; v < vertices.size(); v++) {
        const std::vector<size_t> &incident_faces = vertex_to_faces.at(v);

        if (incident_faces.empty()) {
            // An unreferenced vertex has no adjacent face to derive a normal from; keep it in place with a zero normal so the vertex array stays parallel to any texture coordinates.
            split_vertices.push_back(vertices.at(v));
            split_normals.push_back(make_vec3(0, 0, 0));
            if (!vertex_uv.empty()) {
                split_uv.push_back(vertex_uv.at(v));
            }
            continue;
        }

        // Partition the incident faces into smooth groups, merging two faces whenever they lie within the crease angle of one another.
        std::vector<std::vector<size_t>> smooth_groups;
        for (size_t incident_face: incident_faces) {
            bool placed = false;
            for (std::vector<size_t> &group: smooth_groups) {
                for (size_t member_face: group) {
                    if (acos_safe(face_normals.at(incident_face) * face_normals.at(member_face)) <= crease_angle_radians) {
                        group.push_back(incident_face);
                        placed = true;
                        break;
                    }
                }
                if (placed) {
                    break;
                }
            }
            if (!placed) {
                smooth_groups.push_back({incident_face});
            }
        }

        for (const std::vector<size_t> &group: smooth_groups) {
            vec3 accumulated = make_vec3(0, 0, 0);
            for (size_t member_face: group) {
                accumulated = accumulated + face_normals.at(member_face) * face_areas.at(member_face);
            }
            const float magnitude = accumulated.magnitude();
            if (magnitude > 0.f) {
                accumulated = accumulated / magnitude;
            } else {
                // Every face in the group was degenerate, or their normals cancelled exactly; fall back to a member face normal rather than emitting a zero-length normal.
                accumulated = face_normals.at(group.front());
            }

            const int new_vertex_index = scast<int>(split_vertices.size());
            split_vertices.push_back(vertices.at(v));
            split_normals.push_back(accumulated);
            if (!vertex_uv.empty()) {
                split_uv.push_back(vertex_uv.at(v));
            }

            for (size_t member_face: group) {
                if (faces.at(member_face).x == scast<int>(v)) {
                    updated_faces.at(member_face).x = new_vertex_index;
                }
                if (faces.at(member_face).y == scast<int>(v)) {
                    updated_faces.at(member_face).y = new_vertex_index;
                }
                if (faces.at(member_face).z == scast<int>(v)) {
                    updated_faces.at(member_face).z = new_vertex_index;
                }
            }
        }
    }

    vertices = split_vertices;
    vertex_normals = split_normals;
    vertex_uv = split_uv;
    faces = updated_faces;

    normal_source = helios::NORMAL_SOURCE_COMPUTED;
    invalidateAdjacency();
}

bool Polymesh::isClosed() const {
    return getBoundaryEdges().empty();
}

std::vector<helios::int2> Polymesh::getBoundaryEdges() const {
    std::unordered_map<int64_t, int> edge_face_count;
    for (const int3 &face: faces) {
        edge_face_count[makeEdgeKey(face.x, face.y)]++;
        edge_face_count[makeEdgeKey(face.y, face.z)]++;
        edge_face_count[makeEdgeKey(face.z, face.x)]++;
    }

    std::vector<int2> boundary_edges;
    for (const auto &[edge, count]: edge_face_count) {
        if (count == 1) {
            boundary_edges.push_back(make_int2(scast<int>(edge >> 32), scast<int>(edge & 0xffffffffLL)));
        }
    }
    // The hash map does not preserve an ordering, so sort for a deterministic result across runs and platforms.
    std::sort(boundary_edges.begin(), boundary_edges.end(), [](const int2 &a, const int2 &b) { return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y); });
    return boundary_edges;
}

std::vector<std::vector<size_t>> Polymesh::getConnectedComponents() const {
    buildAdjacency();

    std::vector<std::vector<size_t>> components;
    std::vector<bool> face_visited(faces.size(), false);

    for (size_t seed_face = 0; seed_face < faces.size(); seed_face++) {
        if (face_visited.at(seed_face)) {
            continue;
        }

        std::vector<size_t> component;
        std::vector<size_t> pending = {seed_face};
        face_visited.at(seed_face) = true;

        while (!pending.empty()) {
            const size_t current_face = pending.back();
            pending.pop_back();
            component.push_back(current_face);

            const int3 &face = faces.at(current_face);
            for (int vertex_index: {face.x, face.y, face.z}) {
                for (size_t neighbor_face: vertex_to_faces.at(vertex_index)) {
                    if (!face_visited.at(neighbor_face)) {
                        face_visited.at(neighbor_face) = true;
                        pending.push_back(neighbor_face);
                    }
                }
            }
        }

        std::sort(component.begin(), component.end());
        components.push_back(component);
    }

    return components;
}

float Polymesh::getSurfaceArea() const {
    const std::vector<vec3> vertices_global = getVertices();
    float surface_area = 0.f;
    for (const int3 &face: faces) {
        surface_area += calculateTriangleArea(vertices_global.at(face.x), vertices_global.at(face.y), vertices_global.at(face.z));
    }
    return surface_area;
}

float Polymesh::getVolume() const {

    // When the object carries an indexed face set the surface can be verified before applying the divergence theorem, which is only valid for a closed surface.
    if (!faces.empty()) {
        const std::vector<int2> boundary_edges = getBoundaryEdges();
        if (!boundary_edges.empty()) {
            helios_runtime_error("ERROR (Polymesh::getVolume): Polymesh object " + std::to_string(OID) + " is not a closed surface (" + std::to_string(boundary_edges.size()) +
                                 " boundary edges were found), so its enclosed volume is undefined. Volume can only be computed for a watertight mesh; use getSurfaceArea() for an open mesh, or repair the "
                                 "holes in the source geometry.");
        }

        const std::vector<vec3> vertices_global = getVertices();
        float volume = 0.f;
        for (const int3 &face: faces) {
            const vec3 &v0 = vertices_global.at(face.x);
            const vec3 &v1 = vertices_global.at(face.y);
            const vec3 &v2 = vertices_global.at(face.z);
            volume += (1.f / 6.f) * v0 * cross(v1, v2);
        }
        return std::abs(volume);
    }

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
    std::vector<float> radii_T(radii.size());
    for (size_t i = 0; i < radii.size(); i++) {
        radii_T.at(i) = transformedRadius(radii.at(i));
    }
    return radii_T;
}

float Cone::getNodeRadius(int node_index) const {
    if (node_index < 0 || node_index > 1) {
        helios_runtime_error("ERROR (Cone::getNodeRadius): node number must be 0 or 1.");
    }

    return transformedRadius(radii.at(node_index));
}

float Cone::transformedRadius(float radius_local) const {
    // The stored radius is in the object's local frame, so it is scaled by the object transformation in the same way the node coordinates are. Measuring the transformed length of a radial vector keeps this
    // consistent with getNodeCoordinates() and getLength(), which also report the cone as it currently is rather than as it was constructed.
    vec3 origin_T, radial_T;
    vecmult(transform, make_vec3(0, 0, 0), origin_T);
    vecmult(transform, make_vec3(radius_local, 0, 0), radial_T);
    return (radial_T - origin_T).magnitude();
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
    const std::vector<helios::vec3> &nodes_T = getNodeCoordinates();
    const std::vector<float> &radii_T = getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z);
    float length = powf(powf(axis_unit_vector.x, 2) + powf(axis_unit_vector.y, 2) + powf(axis_unit_vector.z, 2), 0.5);
    axis_unit_vector = axis_unit_vector / length;

    // translate node 0 back to origin
    translate(-1.0 * nodes_T.at(0));

    // rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0, 0, 1);
    // get the axis about which to rotate
    vec3 ra = cross(z_axis, axis_unit_vector);
    // get the angle to rotate
    float dot = axis_unit_vector.x * z_axis.x + axis_unit_vector.y * z_axis.y + axis_unit_vector.z * z_axis.z;
    float angle = acos_safe(dot);

    // only rotate if the cone is not alread aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
    if (angle != 0.f) {
        rotate(-1 * angle, ra);
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
    if (angle != 0.f) {
        rotate(angle, ra);
    }

    // translate back
    translate(nodes_T.at(0));
}

void Cone::scaleGirth(float S) {
    // get the nodes and radii of the nodes with transformation matrix applied
    const std::vector<helios::vec3> &nodes_T = getNodeCoordinates();
    const std::vector<float> &radii_T = getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z);
    axis_unit_vector.normalize();

    // translate node 0 back to origin
    translate(-1.0 * nodes_T.at(0));
    // rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0, 0, 1);
    // get the axis about which to rotate
    vec3 ra = cross(z_axis, axis_unit_vector);
    // get the angle to rotate
    float dot = axis_unit_vector * z_axis;
    float angle = acos_safe(dot);
    // only rotate if the cone is not already aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
    if (angle != 0.f) {
        rotate(-1 * angle, ra);
    }

    // scale the cone in the x and y dimensions
    context->scaleObject(OID, make_vec3(S, S, 1));


    // rotate back
    if (angle != 0.f) {
        rotate(angle, ra);
    }

    // translate back
    translate(nodes_T.at(0));

    // The scaling has been applied to the object transformation, and getNodeRadii() reads the radii through that transformation, so the stored local radii must not be scaled a second time here.
}

float Cone::getVolume() const {
    float r0 = getNodeRadius(0);
    float r1 = getNodeRadius(1);
    float h = getLength();

    return PI_F * h / 3.f * (r0 * r0 + r0 * r1 + r1 * r1);
}

std::vector<helios::vec3> Cone::getPrimitiveVertexNormals(uint UUID) const {
    const std::vector<vec3> nodes_global = getNodeCoordinates();
    const vec3 axis = nodes_global.at(1) - nodes_global.at(0);
    const float length = axis.magnitude();
    if (length < 1e-6f) {
        return {};
    }
    const vec3 axis_unit = axis / length;

    // The taper slope is a ratio of a radius difference to a length, so both are taken in the object's local frame. Under a uniform scaling the numerator and denominator scale together and the ratio is
    // unchanged, which keeps the slope correct without having to reason about how the transformation is distributed between the two.
    const float local_length = (nodes.at(1) - nodes.at(0)).magnitude();
    const float taper_slope = (local_length > 1e-6f) ? (radii.at(1) - radii.at(0)) / local_length : 0.f;

    const std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);

    // The normal of a cone's lateral surface depends only on the azimuth of the point about the axis, so the axial component is removed and the remainder gives the radial direction. There is no need to
    // determine which end of the cone the point belongs to.
    std::vector<vec3> radial_directions(vertices.size());
    std::vector<bool> radial_defined(vertices.size(), false);
    for (size_t k = 0; k < vertices.size(); k++) {
        const vec3 offset = vertices.at(k) - nodes_global.at(0);
        const vec3 radial = offset - (offset * axis_unit) * axis_unit;
        const float radial_magnitude = radial.magnitude();
        if (radial_magnitude > 1e-9f) {
            radial_directions.at(k) = radial / radial_magnitude;
            radial_defined.at(k) = true;
        }
    }

    std::vector<vec3> normals;
    normals.reserve(vertices.size());
    for (size_t k = 0; k < vertices.size(); k++) {

        vec3 radial_direction;
        if (radial_defined.at(k)) {
            radial_direction = radial_directions.at(k);
        } else {
            // The vertex lies on an apex of zero radius, where the surface pinches to a point and has no single normal. Because the normal is constant along a slant line, borrowing the azimuth of another
            // vertex of the same triangle gives the normal of the surface arriving at the apex, which is what makes the shading continuous up to the tip.
            size_t donor = vertices.size();
            for (size_t m = 0; m < vertices.size(); m++) {
                if (radial_defined.at(m)) {
                    donor = m;
                    break;
                }
            }
            if (donor == vertices.size()) {
                // Every vertex is on the axis, so the primitive is degenerate and no normal can be recovered from it.
                return {};
            }
            radial_direction = radial_directions.at(donor);
        }

        // Tilt the radial direction toward the narrowing end by the taper slope, giving the true normal of the slanted surface rather than that of a cylinder.
        vec3 normal = radial_direction - taper_slope * axis_unit;
        const float magnitude = normal.magnitude();
        normals.push_back((magnitude > 0.f) ? normal / magnitude : radial_direction);
    }
    return normals;
}
