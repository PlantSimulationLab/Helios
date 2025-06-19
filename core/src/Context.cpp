/** \file "Context.cpp" Context declarations.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "Context.h"

using namespace helios;

Context::Context() {

    install_out_of_memory_handler();

    //---- ALL DEFAULT VALUES ARE SET HERE ----//

    sim_date = make_Date(1, 6, 2000);

    sim_time = make_Time(12, 0);

    sim_location = make_Location(38.55, 121.76, 8);

    // --- Initialize random number generator ---- //

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);

    // --- Set Geometry as `Clean' --- //

    currentUUID = 0;

    currentObjectID = 1; //object ID of 0 is reserved for default object
}

void Context::seedRandomGenerator(uint seed) {
    generator.seed(seed);
}

std::minstd_rand0 *Context::getRandomGenerator() {
    return &generator;
}

void Context::addTexture(const char *texture_file) {
    if (textures.find(texture_file) == textures.end()) { //texture has not already been added

        //texture must have type PNG or JPEG
        const std::string &fn = texture_file;
        const std::string &ext = getFileExtension(fn);
        if (ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG") {
            helios_runtime_error("ERROR (Context::addTexture): Texture file " + fn + " is not PNG or JPEG format.");
        } else if (!doesTextureFileExist(texture_file)) {
            helios_runtime_error("ERROR (Context::addTexture): Texture file " + std::string(texture_file) + " does not exist.");
        }

        textures.emplace(texture_file, Texture(texture_file));
    }
}

bool Context::doesTextureFileExist(const char *texture_file) const {
    return std::filesystem::exists(texture_file);
}

bool Context::validateTextureFileExtenstion(const char *texture_file) const {
    const std::string &fn = texture_file;
    const std::string &ext = getFileExtension(fn);
    if (ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG") {
        return false;
    } else {
        return true;
    }
}

Texture::Texture(const char *texture_file) {
    filename = texture_file;

    //------ determine if transparency channel exists ---------//

    //check if texture file has extension ".png"
    const std::string &ext = getFileExtension(filename);
    if (ext != ".png") {
        hastransparencychannel = false;
    } else {
        hastransparencychannel = PNGHasAlpha(filename.c_str());
    }

    //-------- load transparency channel (if exists) ------------//

    if (ext == ".png") {
        transparencydata = readPNGAlpha(filename);
        image_resolution = make_int2(int(transparencydata.front().size()), int(transparencydata.size()));
    } else {
        image_resolution = getImageResolutionJPEG(texture_file);
    }

    //-------- determine solid fraction --------------//

    if (hastransparencychannel) {
        size_t p = 0.f;
        for (auto &j: transparencydata) {
            for (bool transparency: j) {
                if (transparency) {
                    p += 1;
                }
            }
        }
        float sf = float(p) / float(transparencydata.size() * transparencydata.front().size());
        if (std::isnan(sf)) {
            sf = 0.f;
        }
        solidfraction = sf;
    } else {
        solidfraction = 1.f;
    }
}

std::string Texture::getTextureFile() const {
    return filename;
}

helios::int2 Texture::getImageResolution() const {
    return image_resolution;
}

bool Texture::hasTransparencyChannel() const {
    return hastransparencychannel;
}

const std::vector<std::vector<bool> > *Texture::getTransparencyData() const {
    return &transparencydata;
}

float Texture::getSolidFraction(const std::vector<helios::vec2> &uvs) {
    float solidfraction = 1;

    PixelUVKey key;
    key.coords.reserve(2 * uvs.size());
    for (auto &uvc: uvs) {
        key.coords.push_back(int(std::round(uvc.x * (image_resolution.x - 1))));
        key.coords.push_back(int(std::round(uvc.y * (image_resolution.y - 1))));
    }

    if (solidFracCache.find(key) != solidFracCache.end()) {
        return solidFracCache.at(key);
    }

    solidfraction = computeSolidFraction(uvs);
    solidFracCache.emplace(std::move(key), solidfraction);

    return solidfraction;
}

float Texture::computeSolidFraction(const std::vector<helios::vec2> &uvs) const {
    // Early out for opaque textures or degenerate UVs
    if (!hasTransparencyChannel() || uvs.size() < 3)
        return 1.0f;

    // Fetch alpha mask and dimensions
    const auto *alpha2D = getTransparencyData(); // vector<vector<bool>>
    int W = getImageResolution().x;
    int H = getImageResolution().y;

    // Flatten mask to contiguous array
    std::vector<uint8_t> mask(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            mask[y * W + x] = (*alpha2D)[H - 1 - y][x];

    // Compute pixel‐space bounding box from UVs
    float minU = uvs[0].x, maxU = uvs[0].x,
            minV = uvs[0].y, maxV = uvs[0].y;
    for (auto &p: uvs) {
        minU = std::min(minU, p.x);
        maxU = std::max(maxU, p.x);
        minV = std::min(minV, p.y);
        maxV = std::max(maxV, p.y);
    }
    int xmin = std::clamp(int(std::floor(minU * (W - 1))), 0, W - 1);
    int xmax = std::clamp(int(std::ceil(maxU * (W - 1))), 0, W - 1);
    int ymin = std::clamp(int(std::floor(minV * (H - 1))), 0, H - 1);
    int ymax = std::clamp(int(std::ceil(maxV * (H - 1))), 0, H - 1);

    if (xmin > xmax || ymin > ymax)
        return 0.0f;

    // Precompute half‐space coefficients for each edge i→i+1
    int N = int(uvs.size());
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        const auto &a = uvs[i], &b = uvs[j];
        // L(x,y) = (b.x - a.x)*y  - (b.y - a.y)*x  + (a.x*b.y - a.y*b.x)
        A[i] = b.x - a.x;
        B[i] = -(b.y - a.y);
        C[i] = a.x * b.y - a.y * b.x;
    }

    // Raster‐scan, test each pixel center
    int64_t countTotal = 0, countOpaque = 0;
    float invWm1 = 1.0f / float(W - 1);
    float invHm1 = 1.0f / float(H - 1);

    for (int j = ymin; j <= ymax; ++j) {
        float yuv = (j + 0.5f) * invHm1;
        for (int i = xmin; i <= xmax; ++i) {
            float xuv = (i + 0.5f) * invWm1;
            bool inside = true;

            // all edges must satisfy L(xuv,yuv) >= 0
            for (int k = 0; k < N; ++k) {
                float L = A[k] * yuv + B[k] * xuv + C[k];
                if (L < 0.0f) {
                    inside = false;
                    break;
                }
            }

            if (!inside)
                continue;

            ++countTotal;
            countOpaque += mask[j * W + i];
        }
    }

    return countTotal == 0 ? 0.0f : float(countOpaque) / float(countTotal);
}

void Context::markGeometryClean() {
    for (auto &[UUID, primitive]: primitives) {
        primitive->dirty_flag = false;
    }
    dirty_deleted_primitives.clear();
}

void Context::markGeometryDirty() {
    for (auto &[UUID, primitive]: primitives) {
        primitive->dirty_flag = true;
    }
}

bool Context::isGeometryDirty() const {
    if (!dirty_deleted_primitives.empty()) {
        return true;
    }
    for (auto &[UUID, primitive]: primitives) {
        if (primitive->dirty_flag) {
            return true;
        }
    }
    return false;
}

void Context::markPrimitiveDirty(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::markPrimitiveDirty): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
    }
#endif
    primitives.at(UUID)->dirty_flag = true;
}

void Context::markPrimitiveDirty(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        markPrimitiveDirty(UUID);
    }
}

void Context::markPrimitiveClean(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::markPrimitiveDirty): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
    }
#endif
    primitives.at(UUID)->dirty_flag = false;
}

void Context::markPrimitiveClean(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        markPrimitiveClean(UUID);
    }
}

[[nodiscard]] bool Context::isPrimitiveDirty(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::markPrimitiveDirty): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
    }
#endif
    return primitives.at(UUID)->dirty_flag;
}

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

    const std::vector<vec3> Y = {
        {-0.5f, -0.5f, 0.f},
        {0.5f, -0.5f, 0.f},
        {0.5f, 0.5f, 0.f},
        {-0.5f, 0.5f, 0.f}
    };

    for (int i = 0; i < 4; i++) {
        vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
        vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
        vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
    }
    return vertices;
}

std::vector<vec3> Triangle::getVertices() const {
    std::vector<vec3> vertices(3);

    const std::vector<vec3> Y = {
        {0.f, 0.f, 0.f},
        {0.f, 1.f, 0.f},
        {1.f, 1.f, 0.f}
    };

    for (int i = 0; i < 3; i++) {
        vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
        vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
        vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
    }
    return vertices;
}

std::vector<vec3> Voxel::getVertices() const {
    std::vector<vec3> vertices(8);

    const std::vector<vec3> Y = {
        {-0.5f, -0.5f, -0.5f},
        {0.5f, -0.5f, -0.5f},
        {0.5f, 0.5f, -0.5f},
        {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f},
        {-0.5f, 0.5f, 0.5f}
    };


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
    //We need to construct the Affine transformation matrix that transforms some generic triangle to a triangle with vertices at vertex0, vertex1, vertex2.

    //V1 is going to be our generic triangle.  This is the triangle that we'll intersect in the OptiX ray intersection program.  We just need to pass the transformation matrix to OptiX so that we'll end up with the right triangle.

    //We'll assume our generic triangle has vertices
    //v0 = (0,0,0)
    //v1 = (0,1,0)
    //v2 = (1,1,0)
    //this needs to match up with the triangle in triangle_intersect() and triangle_bounds() (see primitiveIntersection.cu).
    //Note that the matrix is padded with 1's to make it 4x4

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

    //V2 holds the vertex locations we want to transform to
    //Note that the matrix is padded with 1's to make it 4x4

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

    //Now we just need to solve the linear system for our transform matrix T
    // [T][V1] = [V2]  -->
    // [T] = [V2]([V1]^-1)

    double inv[16], det, invV1[16];

    inv[0] = V1[5] * V1[10] * V1[15] -
             V1[5] * V1[11] * V1[14] -
             V1[9] * V1[6] * V1[15] +
             V1[9] * V1[7] * V1[14] +
             V1[13] * V1[6] * V1[11] -
             V1[13] * V1[7] * V1[10];

    inv[4] = -V1[4] * V1[10] * V1[15] +
             V1[4] * V1[11] * V1[14] +
             V1[8] * V1[6] * V1[15] -
             V1[8] * V1[7] * V1[14] -
             V1[12] * V1[6] * V1[11] +
             V1[12] * V1[7] * V1[10];

    inv[8] = V1[4] * V1[9] * V1[15] -
             V1[4] * V1[11] * V1[13] -
             V1[8] * V1[5] * V1[15] +
             V1[8] * V1[7] * V1[13] +
             V1[12] * V1[5] * V1[11] -
             V1[12] * V1[7] * V1[9];

    inv[12] = -V1[4] * V1[9] * V1[14] +
              V1[4] * V1[10] * V1[13] +
              V1[8] * V1[5] * V1[14] -
              V1[8] * V1[6] * V1[13] -
              V1[12] * V1[5] * V1[10] +
              V1[12] * V1[6] * V1[9];

    inv[1] = -V1[1] * V1[10] * V1[15] +
             V1[1] * V1[11] * V1[14] +
             V1[9] * V1[2] * V1[15] -
             V1[9] * V1[3] * V1[14] -
             V1[13] * V1[2] * V1[11] +
             V1[13] * V1[3] * V1[10];

    inv[5] = V1[0] * V1[10] * V1[15] -
             V1[0] * V1[11] * V1[14] -
             V1[8] * V1[2] * V1[15] +
             V1[8] * V1[3] * V1[14] +
             V1[12] * V1[2] * V1[11] -
             V1[12] * V1[3] * V1[10];

    inv[9] = -V1[0] * V1[9] * V1[15] +
             V1[0] * V1[11] * V1[13] +
             V1[8] * V1[1] * V1[15] -
             V1[8] * V1[3] * V1[13] -
             V1[12] * V1[1] * V1[11] +
             V1[12] * V1[3] * V1[9];

    inv[13] = V1[0] * V1[9] * V1[14] -
              V1[0] * V1[10] * V1[13] -
              V1[8] * V1[1] * V1[14] +
              V1[8] * V1[2] * V1[13] +
              V1[12] * V1[1] * V1[10] -
              V1[12] * V1[2] * V1[9];

    inv[2] = V1[1] * V1[6] * V1[15] -
             V1[1] * V1[7] * V1[14] -
             V1[5] * V1[2] * V1[15] +
             V1[5] * V1[3] * V1[14] +
             V1[13] * V1[2] * V1[7] -
             V1[13] * V1[3] * V1[6];

    inv[6] = -V1[0] * V1[6] * V1[15] +
             V1[0] * V1[7] * V1[14] +
             V1[4] * V1[2] * V1[15] -
             V1[4] * V1[3] * V1[14] -
             V1[12] * V1[2] * V1[7] +
             V1[12] * V1[3] * V1[6];

    inv[10] = V1[0] * V1[5] * V1[15] -
              V1[0] * V1[7] * V1[13] -
              V1[4] * V1[1] * V1[15] +
              V1[4] * V1[3] * V1[13] +
              V1[12] * V1[1] * V1[7] -
              V1[12] * V1[3] * V1[5];

    inv[14] = -V1[0] * V1[5] * V1[14] +
              V1[0] * V1[6] * V1[13] +
              V1[4] * V1[1] * V1[14] -
              V1[4] * V1[2] * V1[13] -
              V1[12] * V1[1] * V1[6] +
              V1[12] * V1[2] * V1[5];

    inv[3] = -V1[1] * V1[6] * V1[11] +
             V1[1] * V1[7] * V1[10] +
             V1[5] * V1[2] * V1[11] -
             V1[5] * V1[3] * V1[10] -
             V1[9] * V1[2] * V1[7] +
             V1[9] * V1[3] * V1[6];

    inv[7] = V1[0] * V1[6] * V1[11] -
             V1[0] * V1[7] * V1[10] -
             V1[4] * V1[2] * V1[11] +
             V1[4] * V1[3] * V1[10] +
             V1[8] * V1[2] * V1[7] -
             V1[8] * V1[3] * V1[6];

    inv[11] = -V1[0] * V1[5] * V1[11] +
              V1[0] * V1[7] * V1[9] +
              V1[4] * V1[1] * V1[11] -
              V1[4] * V1[3] * V1[9] -
              V1[8] * V1[1] * V1[7] +
              V1[8] * V1[3] * V1[5];

    inv[15] = V1[0] * V1[5] * V1[10] -
              V1[0] * V1[6] * V1[9] -
              V1[4] * V1[1] * V1[10] +
              V1[4] * V1[2] * V1[9] +
              V1[8] * V1[1] * V1[6] -
              V1[8] * V1[2] * V1[5];

    det = V1[0] * inv[0] + V1[1] * inv[4] + V1[2] * inv[8] + V1[3] * inv[12];

    //if (det == 0)
    //return false;

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
    assert(color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1);
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
    for ( auto &uv_vert : uv ) {
        uv_vert.x = std::min(uv_vert.x, 1.f );
        uv_vert.y = std::min(uv_vert.y, 1.f );
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
    for ( auto &uv_vert : uv ) {
        uv_vert.x = std::min(uv_vert.x, 1.f );
        uv_vert.y = std::min(uv_vert.y, 1.f );
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

    const std::vector<vec3> Y = {
        {0.f, 0.f, 0.f},
        {0.f, 1.f, 0.f},
        {1.f, 1.f, 0.f}
    };

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
    assert(color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1);
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

void Context::setDate(int day, int month, int year) {
    if (day < 1 || day > 31) {
        helios_runtime_error("ERROR (Context::setDate): Day of month is out of range (day of " + std::to_string(day) + " was given).");
    } else if (month < 1 || month > 12) {
        helios_runtime_error("ERROR (Context::setDate): Month of year is out of range (month of " + std::to_string(month) + " was given).");
    } else if (year < 1000) {
        helios_runtime_error("ERROR (Context::setDate): Year should be specified in YYYY format.");
    }

    sim_date = make_Date(day, month, year);
}

void Context::setDate(const Date &date) {
    if (date.day < 1 || date.day > 31) {
        helios_runtime_error("ERROR (Context::setDate): Day of month is out of range (day of " + std::to_string(date.day) + " was given).");
    } else if (date.month < 1 || date.month > 12) {
        helios_runtime_error("ERROR (Context::setDate): Month of year is out of range (month of " + std::to_string(date.month) + " was given).");
    } else if (date.year < 1000) {
        helios_runtime_error("ERROR (Context::setDate): Year should be specified in YYYY format.");
    }

    sim_date = date;
}

void Context::setDate(int Julian_day, int year) {
    if (Julian_day < 1 || Julian_day > 366) {
        helios_runtime_error("ERROR (Context::setDate): Julian day out of range.");
    } else if (year < 1000) {
        helios_runtime_error("ERROR (Context::setDate): Year should be specified in YYYY format.");
    }

    sim_date = CalendarDay(Julian_day, year);
}

Date Context::getDate() const {
    return sim_date;
}

const char *Context::getMonthString() const {
    if (sim_date.month == 1) {
        return "JAN";
    } else if (sim_date.month == 2) {
        return "FEB";
    } else if (sim_date.month == 3) {
        return "MAR";
    } else if (sim_date.month == 4) {
        return "APR";
    } else if (sim_date.month == 5) {
        return "MAY";
    } else if (sim_date.month == 6) {
        return "JUN";
    } else if (sim_date.month == 7) {
        return "JUL";
    } else if (sim_date.month == 8) {
        return "AUG";
    } else if (sim_date.month == 9) {
        return "SEP";
    } else if (sim_date.month == 10) {
        return "OCT";
    } else if (sim_date.month == 11) {
        return "NOV";
    } else {
        return "DEC";
    }
}

int Context::getJulianDate() const {
    return JulianDay(sim_date.day, sim_date.month, sim_date.year);
}

void Context::setTime(int minute, int hour) {
    setTime(0, minute, hour);
}

void Context::setTime(int second, int minute, int hour) {
    if (second < 0 || second > 59) {
        helios_runtime_error("ERROR (Context::setTime): Second out of range (0-59).");
    } else if (minute < 0 || minute > 59) {
        helios_runtime_error("ERROR (Context::setTime): Minute out of range (0-59).");
    } else if (hour < 0 || hour > 23) {
        helios_runtime_error("ERROR (Context::setTime): Hour out of range (0-23).");
    }

    sim_time = make_Time(hour, minute, second);
}

void Context::setTime(const Time &time) {
    if (time.minute < 0 || time.minute > 59) {
        helios_runtime_error("ERROR (Context::setTime): Minute out of range (0-59).");
    } else if (time.hour < 0 || time.hour > 23) {
        helios_runtime_error("ERROR (Context::setTime): Hour out of range (0-23).");
    }

    sim_time = time;
}

Time Context::getTime() const {
    return sim_time;
}

void Context::setLocation(const helios::Location &location) {
    sim_location = location;
}

helios::Location Context::getLocation() const {
    return sim_location;
}

float Context::randu() {
    return unif_distribution(generator);
}

float Context::randu(float minrange, float maxrange) {
    if (maxrange < minrange) {
        helios_runtime_error("ERROR (Context::randu): Maximum value of range must be greater than minimum value of range.");
        return 0;
    } else if (maxrange == minrange) {
        return minrange;
    } else {
        return minrange + unif_distribution(generator) * (maxrange - minrange);
    }
}

int Context::randu(int minrange, int maxrange) {
    if (maxrange < minrange) {
        helios_runtime_error("ERROR (Context::randu): Maximum value of range must be greater than minimum value of range.");
        return 0;
    } else if (maxrange == minrange) {
        return minrange;
    } else {
        return minrange + (int) lroundf(unif_distribution(generator) * float(maxrange - minrange));
    }
}

float Context::randn() {
    return norm_distribution(generator);
}

float Context::randn(float mean, float stddev) {
    return mean + norm_distribution(generator) * fabs(stddev);
}

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
    if (size.x == 0 || size.y == 0) {
        helios_runtime_error("ERROR (Context::addPatch): Size of patch must be greater than 0.");
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

    assert(size.x>0.f && size.y>0.f);
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
    return currentUUID - 1;
}

uint Context::addPatch(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file, const helios::vec2 &uv_center, const helios::vec2 &uv_size) {
    if (size.x == 0 || size.y == 0) {
        helios_runtime_error("ERROR (Context::addPatch): Size of patch must be greater than 0.");
    }

    if (uv_center.x - 0.5 * uv_size.x < -1e-3 || uv_center.y - 0.5 * uv_size.y < -1e-3 || uv_center.x + 0.5 * uv_size.x - 1.f > 1e-3 || uv_center.y + 0.5 * uv_size.y - 1.f > 1e-3) {
        helios_runtime_error("ERROR (Context::addPatch): Invalid texture coordinates. uv_center-0.5*uv_size should be >=0 and uv_center+0.5*uv_size should be <=1.");
    }

    addTexture(texture_file);

    const std::vector<helios::vec2> uv = {
        uv_center + make_vec2(-0.5f * uv_size.x, -0.5f * uv_size.y),
        uv_center + make_vec2(+0.5f * uv_size.x, -0.5f * uv_size.y),
        uv_center + make_vec2(+0.5f * uv_size.x, +0.5f * uv_size.y),
        uv_center + make_vec2(-0.5f * uv_size.x, +0.5f * uv_size.y)
    };

    auto *patch_new = (new Patch(texture_file, uv, textures, 0, currentUUID));

    //    if( patch_new->getArea()==0 ){
    //        helios_runtime_error("ERROR (Context::addPatch): Patch has area of zero.");
    //    }

    assert(size.x>0.f && size.y>0.f);
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

    //    if( tri_new->getArea()==0 ){
    //        helios_runtime_error("ERROR (Context::addTriangle): Triangle has area of zero.");
    //    }

    primitives[currentUUID] = tri_new;
    currentUUID++;
    return currentUUID - 1;
}

uint Context::addTriangle(const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2, const char *texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2) {
    addTexture(texture_file);

    const std::vector<helios::vec2> uv{uv0, uv1, uv2};

    auto *tri_new = (new Triangle(vertex0, vertex1, vertex2, texture_file, uv, textures, 0, currentUUID));

    //    if( tri_new->getArea()==0 ){
    //        helios_runtime_error("ERROR (Context::addTriangle): Triangle has area of zero.");
    //    }

    primitives[currentUUID] = tri_new;
    currentUUID++;
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
        axis = (std::fabs(oldN.x) < std::fabs(oldN.z))
                   ? cross(oldN, {1, 0, 0})
                   : cross(oldN, {0, 0, 1});
    }
    axis = axis.normalize();

    // build M_delta about 'origin'
    float M_delta[16];
    makeRotationMatrix(angle, origin, axis, M_delta);

    // grab existing world‐space model matrix
    float M_old[16];
    prim->getTransformationMatrix(M_old);

    // preserve the rectangle’s forward (local X) direction:
    //   - t0 is the world‐space image of (1,0,0) under M_old
    helios::vec3 t0{
        M_old[0], // row0·[1,0,0,0]
        M_old[4], // row1·[1,0,0,0]
        M_old[8] // row2·[1,0,0,0]
    };
    t0 = normalize(t0);

    //  apply M_delta to that direction (w=0)
    helios::vec3 t1{
        M_delta[0] * t0.x + M_delta[1] * t0.y + M_delta[2] * t0.z,
        M_delta[4] * t0.x + M_delta[5] * t0.y + M_delta[6] * t0.z,
        M_delta[8] * t0.x + M_delta[9] * t0.y + M_delta[10] * t0.z
    };
    t1 = normalize(t1);

    //  desired forward is world‐X projected onto the new plane
    helios::vec3 worldX{1.f, 0.f, 0.f};
    helios::vec3 targ = worldX - newN * (newN * worldX);
    targ = normalize(targ);

    // compute the twist about newN that carries t1 → targ
    //    using signed angle in that plane
    float twist = std::atan2(
        newN * cross(t1, targ), // dot(newN, t1×targ)
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

    if (prim->getParentObjectID() != 0) { //primitive belongs to an object

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
        //if( !p->hasTexture() ){
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
        if (!include_hidden_primitives && !primitive->ishidden) {
            continue;
        }
        if (primitive->getType() == PRIMITIVE_TYPE_TRIANGLE) {
            count++;
        }
    }
    return count;
}

size_t Context::getPatchCount(bool include_hidden_primitives) const {
    size_t count = 0;
    for (const auto &[UUID, primitive]: primitives) {
        if (!include_hidden_primitives && !primitive->ishidden) {
            continue;
        }
        if (primitive->getType() == PRIMITIVE_TYPE_PATCH) {
            count++;
        }
    }
    return count;
}

std::vector<uint> Context::getAllUUIDs() const {
    std::vector<uint> UUIDs;
    UUIDs.reserve(primitives.size());
    for (const auto &[UUID, primitive]: primitives) {
        if (primitive->ishidden) {
            continue;
        }
        UUIDs.push_back(UUID);
    }
    return UUIDs;
}

std::vector<uint> Context::getDirtyUUIDs(bool include_deleted_UUIDs) const {

    size_t dirty_count = std::count_if(
        primitives.begin(), primitives.end(),
        [&](auto const &kv){ return isPrimitiveDirty(kv.first); }
    );

    std::vector<uint> dirty_UUIDs;
    dirty_UUIDs.reserve(dirty_count);
    for (const auto &[UUID, primitive]: primitives) {
        if (!primitive->dirty_flag || primitive->ishidden ) {
            continue;
        }
        dirty_UUIDs.push_back(UUID);
    }

    if ( include_deleted_UUIDs ) {
        dirty_UUIDs.insert( dirty_UUIDs.end(), dirty_deleted_primitives.begin(), dirty_deleted_primitives.end() );
    }

    return dirty_UUIDs;
}

std::vector<uint> Context::getDeletedUUIDs() const {
    return dirty_deleted_primitives;
}

void Context::hidePrimitive(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::hidePrimitive): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    primitives.at(UUID)->ishidden = true;
}

void Context::hidePrimitive(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        hidePrimitive(UUID);
    }
}

bool Context::isPrimitiveHidden(uint UUID) const {
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::isPrimitiveHidden): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
    return primitives.at(UUID)->ishidden;
}

void Context::cleanDeletedUUIDs(std::vector<uint> &UUIDs) const {
    for (auto it = UUIDs.begin(); it != UUIDs.end(); ) {
        if (!doesObjectExist(*it)) {
            it = UUIDs.erase(it);
        } else {
            ++it;
        }
    }
}

void Context::cleanDeletedUUIDs(std::vector<std::vector<uint> > &UUIDs) const {
    for (auto &vec : UUIDs) {
        for (auto it = vec.begin(); it != vec.end(); ) {
            if (!doesObjectExist(*it)) {
                it = vec.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void Context::cleanDeletedUUIDs(std::vector<std::vector<std::vector<uint> > > &UUIDs) const {
    for (auto &vec2D : UUIDs) {
        for (auto &vec : vec2D) {
            for (auto it = vec.begin(); it != vec.end(); ) {
                if (!doesObjectExist(*it)) {
                    it = vec.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

void Context::addTimeseriesData(const char *label, float value, const Date &date, const Time &time) {
    //floating point value corresponding to date and time
    double date_value = floor(date.year * 366.25) + date.JulianDay();
    date_value += double(time.hour) / 24. + double(time.minute) / 1440. + double(time.second) / 86400.;

    //Check if data label already exists
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        timeseries_data[label].push_back(value);
        timeseries_datevalue[label].push_back(date_value);
        return;
    } else { //exists

        uint N = getTimeseriesLength(label);

        auto it_data = timeseries_data[label].begin();
        auto it_datevalue = timeseries_datevalue[label].begin();

        if (N == 1) {
            if (date_value < timeseries_datevalue[label].front()) {
                timeseries_data[label].insert(it_data, value);
                timeseries_datevalue[label].insert(it_datevalue, date_value);
                return;
            } else {
                timeseries_data[label].insert(it_data + 1, value);
                timeseries_datevalue[label].insert(it_datevalue + 1, date_value);
                return;
            }
        } else {
            if (date_value < timeseries_datevalue[label].front()) { //check if data should be inserted at beginning of timeseries
                timeseries_data[label].insert(it_data, value);
                timeseries_datevalue[label].insert(it_datevalue, date_value);
                return;
            } else if (date_value > timeseries_datevalue[label].back()) { //check if data should be inserted at end of timeseries
                timeseries_data[label].push_back(value);
                timeseries_datevalue[label].push_back(date_value);
                return;
            }

            //data should be inserted somewhere in the middle of timeseries
            for (uint t = 0; t < N - 1; t++) {
                if (date_value == timeseries_datevalue[label].at(t)) {
                    std::cerr << "WARNING (Context::addTimeseriesData): Skipping duplicate timeseries date/time." << std::endl;
                    continue;
                }
                if (date_value > timeseries_datevalue[label].at(t) && date_value < timeseries_datevalue[label].at(t + 1)) {
                    timeseries_data[label].insert(it_data + t + 1, value);
                    timeseries_datevalue[label].insert(it_datevalue + t + 1, date_value);
                    return;
                }
            }
        }
    }

    helios_runtime_error("ERROR (Context::addTimeseriesData): Failed to insert timeseries data for unknown reason.");
}

void Context::setCurrentTimeseriesPoint(const char *label, uint index) {
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        helios_runtime_error("ERROR (setCurrentTimeseriesPoint): Timeseries variable `" + std::string(label) + "' does not exist.");
    }
    setDate(queryTimeseriesDate(label, index));
    setTime(queryTimeseriesTime(label, index));
}

float Context::queryTimeseriesData(const char *label, const Date &date, const Time &time) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        helios_runtime_error("ERROR (setCurrentTimeseriesData): Timeseries variable `" + std::string(label) + "' does not exist.");
    }

    double date_value = floor(date.year * 366.25) + date.JulianDay();
    date_value += double(time.hour) / 24. + double(time.minute) / 1440. + double(time.second) / 86400.;

    double tmin = timeseries_datevalue.at(label).front();
    double tmax = timeseries_datevalue.at(label).back();

    if (date_value < tmin) {
        std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the earliest data point in the timeseries." << std::endl;
        return timeseries_data.at(label).front();
    } else if (date_value > tmax) {
        std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the latest data point in the timeseries." << std::endl;
        return timeseries_data.at(label).back();
    }

    if (timeseries_datevalue.at(label).empty()) {
        std::cerr << "WARNING (queryTimeseriesData): timeseries " << label << " does not contain any data." << std::endl;
        return 0;
    } else if (timeseries_datevalue.at(label).size() == 1) {
        return timeseries_data.at(label).front();
    } else {
        int i;
        bool success = false;
        for (i = 0; i < timeseries_data.at(label).size() - 1; i++) {
            if (date_value >= timeseries_datevalue.at(label).at(i) && date_value <= timeseries_datevalue.at(label).at(i + 1)) {
                success = true;
                break;
            }
        }

        if (!success) {
            helios_runtime_error("ERROR (queryTimeseriesData): Failed to query timeseries data for unknown reason.");
        }

        double xminus = timeseries_data.at(label).at(i);
        double xplus = timeseries_data.at(label).at(i + 1);

        double tminus = timeseries_datevalue.at(label).at(i);
        double tplus = timeseries_datevalue.at(label).at(i + 1);

        return float(xminus + (xplus - xminus) * (date_value - tminus) / (tplus - tminus));
    }
}

float Context::queryTimeseriesData(const char *label) const {
    return queryTimeseriesData(label, sim_date, sim_time);
}

float Context::queryTimeseriesData(const char *label, const uint index) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        helios_runtime_error("ERROR( Context::getTimeseriesData): Timeseries variable " + std::string(label) + " does not exist.");
    }

    return timeseries_data.at(label).at(index);
}

Time Context::queryTimeseriesTime(const char *label, const uint index) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        helios_runtime_error("ERROR( Context::getTimeseriesTime): Timeseries variable " + std::string(label) + " does not exist.");
    }

    double dateval = timeseries_datevalue.at(label).at(index);

    int year = floor(floor(dateval) / 366.25);
    assert(year>1000 && year<10000);

    int JD = floor(dateval - floor(double(year) * 366.25));
    assert(JD>0 && JD<367);

    int hour = floor((dateval - floor(dateval)) * 24.);
    int minute = floor(((dateval - floor(dateval)) * 24. - double(hour)) * 60.);
    int second = (int) lround((((dateval - floor(dateval)) * 24. - double(hour)) * 60. - double(minute)) * 60.);

    if (second == 60) {
        second = 0;
        minute++;
    }

    if (minute == 60) {
        minute = 0;
        hour++;
    }

    assert(second>=0 && second<60);
    assert(minute>=0 && minute<60);
    assert(hour>=0 && hour<24);

    return make_Time(hour, minute, second);
}

Date Context::queryTimeseriesDate(const char *label, const uint index) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        helios_runtime_error("ERROR( Context::getTimeseriesDate): Timeseries variable " + std::string(label) + " does not exist.");
    }

    double dateval = timeseries_datevalue.at(label).at(index);

    int year = floor(floor(dateval) / 366.25);
    assert(year>1000 && year<10000);

    int JD = floor(dateval - floor(double(year) * 366.25));
    assert(JD>0 && JD<367);

    return Julian2Calendar(JD, year);
}

uint Context::getTimeseriesLength(const char *label) const {
    uint size = 0;
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        helios_runtime_error("ERROR (Context::getTimeseriesDate): Timeseries variable `" + std::string(label) + "' does not exist.");
    } else {
        size = timeseries_data.at(label).size();
    }

    return size;
}

bool Context::doesTimeseriesVariableExist(const char *label) const {
    if (timeseries_data.find(label) == timeseries_data.end()) { //does not exist
        return false;
    } else {
        return true;
    }
}

std::vector<std::string> Context::listTimeseriesVariables() const {
    std::vector<std::string> labels;
    labels.reserve(timeseries_data.size());
    for (const auto &[timeseries_label, timeseries_data]: timeseries_data) {
        labels.push_back(timeseries_label);
    }
    return labels;
}


void Context::getDomainBoundingBox(vec2 &xbounds, vec2 &ybounds, vec2 &zbounds) const {
    getDomainBoundingBox(getAllUUIDs(), xbounds, ybounds, zbounds);
}

void Context::getDomainBoundingBox(const std::vector<uint> &UUIDs, vec2 &xbounds, vec2 &ybounds, vec2 &zbounds) const {
    // Global bounding box initialization
    xbounds.x = 1e8; // global min x
    xbounds.y = -1e8; // global max x
    ybounds.x = 1e8; // global min y
    ybounds.y = -1e8; // global max y
    zbounds.x = 1e8; // global min z
    zbounds.y = -1e8; // global max z

    // Parallel region over the primitives (UUIDs)
#ifdef USE_OPENMP
    #pragma omp parallel
    {
        // Each thread creates its own local bounding box.
        float local_xmin = 1e8, local_xmax = -1e8;
        float local_ymin = 1e8, local_ymax = -1e8;
        float local_zmin = 1e8, local_zmax = -1e8;

        // Parallelize the outer loop over primitives. Use "for" inside the parallel region.
        #pragma omp for nowait
        for (size_t i = 0; i < UUIDs.size(); i++) {
            // For each primitive:
            const std::vector<vec3>& verts = getPrimitivePointer_private(UUIDs[i])->getVertices();
            // Update local bounding box for each vertex in this primitive.
            for (const auto &vert : verts) {
                local_xmin = std::min(local_xmin, vert.x);
                local_xmax = std::max(local_xmax, vert.x);
                local_ymin = std::min(local_ymin, vert.y);
                local_ymax = std::max(local_ymax, vert.y);
                local_zmin = std::min(local_zmin, vert.z);
                local_zmax = std::max(local_zmax, vert.z);
            }
        }

        // Merge the thread-local bounds into the global bounds.
        #pragma omp critical
        {
            xbounds.x = std::min(xbounds.x, local_xmin);
            xbounds.y = std::max(xbounds.y, local_xmax);
            ybounds.x = std::min(ybounds.x, local_ymin);
            ybounds.y = std::max(ybounds.y, local_ymax);
            zbounds.x = std::min(zbounds.x, local_zmin);
            zbounds.y = std::max(zbounds.y, local_zmax);
        }
    } // end parallel region

#else

    for (uint UUID: UUIDs) {
        const std::vector<vec3> &verts = getPrimitivePointer_private(UUID)->getVertices();

        for (auto &vert: verts) {
            if (vert.x < xbounds.x) {
                xbounds.x = vert.x;
            } else if (vert.x > xbounds.y) {
                xbounds.y = vert.x;
            }
            if (vert.y < ybounds.x) {
                ybounds.x = vert.y;
            } else if (vert.y > ybounds.y) {
                ybounds.y = vert.y;
            }
            if (vert.z < zbounds.x) {
                zbounds.x = vert.z;
            } else if (vert.z > zbounds.y) {
                zbounds.y = vert.z;
            }
        }
    }

#endif
}

void Context::getDomainBoundingSphere(vec3 &center, float &radius) const {
    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox(xbounds, ybounds, zbounds);

    center.x = xbounds.x + 0.5f * (xbounds.y - xbounds.x);
    center.y = ybounds.x + 0.5f * (ybounds.y - ybounds.x);
    center.z = zbounds.x + 0.5f * (zbounds.y - zbounds.x);

    radius = 0.5f * sqrtf(powf(xbounds.y - xbounds.x, 2) + powf(ybounds.y - ybounds.x, 2) + powf((zbounds.y - zbounds.x), 2));
}

void Context::getDomainBoundingSphere(const std::vector<uint> &UUIDs, vec3 &center, float &radius) const {
    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox(UUIDs, xbounds, ybounds, zbounds);

    center.x = xbounds.x + 0.5f * (xbounds.y - xbounds.x);
    center.y = ybounds.x + 0.5f * (ybounds.y - ybounds.x);
    center.z = zbounds.x + 0.5f * (zbounds.y - zbounds.x);

    radius = 0.5f * sqrtf(powf(xbounds.y - xbounds.x, 2) + powf(ybounds.y - ybounds.x, 2) + powf((zbounds.y - zbounds.x), 2));
}

void Context::cropDomainX(const vec2 &xbounds) {
    const std::vector<uint> &UUIDs_all = getAllUUIDs();

    for (uint p: UUIDs_all) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(p)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.x < xbounds.x || vertex.x > xbounds.y) {
                deletePrimitive(p);
                break;
            }
        }
    }

    if (getPrimitiveCount() == 0) {
        std::cerr << "WARNING (Context::cropDomainX): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }
}

void Context::cropDomainY(const vec2 &ybounds) {
    const std::vector<uint> &UUIDs_all = getAllUUIDs();

    for (uint p: UUIDs_all) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(p)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.y < ybounds.x || vertex.y > ybounds.y) {
                deletePrimitive(p);
                break;
            }
        }
    }

    if (getPrimitiveCount() == 0) {
        std::cerr << "WARNING (Context::cropDomainY): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }
}

void Context::cropDomainZ(const vec2 &zbounds) {
    const std::vector<uint> &UUIDs_all = getAllUUIDs();

    for (uint p: UUIDs_all) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(p)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.z < zbounds.x || vertex.z > zbounds.y) {
                deletePrimitive(p);
                break;
            }
        }
    }

    if (getPrimitiveCount() == 0) {
        std::cerr << "WARNING (Context::cropDomainZ): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }
}

void Context::cropDomain(std::vector<uint> &UUIDs, const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds) {
    size_t delete_count = 0;
    for (uint UUID: UUIDs) {
        const std::vector<vec3> &vertices = getPrimitivePointer_private(UUID)->getVertices();

        for (auto &vertex: vertices) {
            if (vertex.x < xbounds.x || vertex.x > xbounds.y || vertex.y < ybounds.x || vertex.y > ybounds.y || vertex.z < zbounds.x || vertex.z > zbounds.y) {
                deletePrimitive(UUID);
                delete_count++;
                break;
            }
        }
    }

    if (delete_count == UUIDs.size()) {
        std::cerr << "WARNING (Context::cropDomain): No specified primitives were entirely inside cropped area, and thus all specified primitives were deleted." << std::endl;
    }

    cleanDeletedUUIDs(UUIDs);
}

void Context::cropDomain(const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds) {
    std::vector<uint> UUIDs = getAllUUIDs();
    cropDomain(UUIDs, xbounds, ybounds, zbounds);
}

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

bool Context::areObjectPrimitivesComplete(uint objID) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(objID)) {
        helios_runtime_error("ERROR (Context::areObjectPrimitivesComplete): Object ID of " + std::to_string(objID) + " does not exist in the context.");
    }
#endif
    return getObjectPointer(objID)->arePrimitivesComplete();
}

void Context::cleanDeletedObjectIDs(std::vector<uint> &objIDs) const {
    for (auto it = objIDs.begin(); it != objIDs.end(); ) {
        if (!doesObjectExist(*it)) {
            it = objIDs.erase(it);
        } else {
            ++it;
        }
    }
}

void Context::cleanDeletedObjectIDs(std::vector<std::vector<uint> > &objIDs) const {
    for (auto &vec : objIDs) {
        for (auto it = vec.begin(); it != vec.end(); ) {
            if (!doesObjectExist(*it)) {
                it = vec.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void Context::cleanDeletedObjectIDs(std::vector<std::vector<std::vector<uint> > > &objIDs) const {
    for (auto &vec2D : objIDs) {
        for (auto &vec : vec2D) {
            for (auto it = vec.begin(); it != vec.end(); ) {
                if (!doesObjectExist(*it)) {
                    it = vec.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

CompoundObject *Context::getObjectPointer(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return objects.at(ObjID);
}

uint Context::getObjectCount() const {
    return objects.size();
}

bool Context::doesObjectExist(const uint ObjID) const {
    return objects.find(ObjID) != objects.end();
}

std::vector<uint> Context::getAllObjectIDs() const {
    std::vector<uint> objIDs;
    objIDs.reserve(objects.size());
    size_t i = 0;
    for (auto [objID, object]: objects) {
        if (object->ishidden) {
            continue;
        }
        objIDs.push_back(objID);
        i++;
    }
    return objIDs;
}

void Context::deleteObject(const std::vector<uint> &ObjIDs) {
    for (const uint ObjID: ObjIDs) {
        deleteObject(ObjID);
    }
}

void Context::deleteObject(uint ObjID) {
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::deleteObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }

    CompoundObject *obj = objects.at(ObjID);

    const std::vector<uint> &UUIDs = obj->getPrimitiveUUIDs();


    delete obj;
    objects.erase(ObjID);

    deletePrimitive(UUIDs);
}

std::vector<uint> Context::copyObject(const std::vector<uint> &ObjIDs) {
    std::vector<uint> ObjIDs_copy(ObjIDs.size());
    size_t i = 0;
    for (uint ObjID: ObjIDs) {
        ObjIDs_copy.at(i) = copyObject(ObjID);
        i++;
    }

    return ObjIDs_copy;
}

uint Context::copyObject(uint ObjID) {
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::copyObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }

    ObjectType type = objects.at(ObjID)->getObjectType();

    const std::vector<uint> &UUIDs = getObjectPointer(ObjID)->getPrimitiveUUIDs();

    const std::vector<uint> &UUIDs_copy = copyPrimitive(UUIDs);
    for (uint p: UUIDs_copy) {
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    const std::string &texturefile = objects.at(ObjID)->getTextureFile();

    if (type == OBJECT_TYPE_TILE) {
        Tile *o = getTileObjectPointer(ObjID);

        const int2 &subdiv = o->getSubdivisionCount();

        auto *tile_new = (new Tile(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = tile_new;
    } else if (type == OBJECT_TYPE_SPHERE) {
        Sphere *o = getSphereObjectPointer(ObjID);

        uint subdiv = o->getSubdivisionCount();

        auto *sphere_new = (new Sphere(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = sphere_new;
    } else if (type == OBJECT_TYPE_TUBE) {
        Tube *o = getTubeObjectPointer(ObjID);

        const std::vector<vec3> &nodes = o->getNodes();
        const std::vector<float> &radius = o->getNodeRadii();
        const std::vector<RGBcolor> &colors = o->getNodeColors();
        const std::vector<std::vector<vec3> > &triangle_vertices = o->getTriangleVertices();
        uint subdiv = o->getSubdivisionCount();

        auto *tube_new = (new Tube(currentObjectID, UUIDs_copy, nodes, radius, colors, triangle_vertices, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = tube_new;
    } else if (type == OBJECT_TYPE_BOX) {
        Box *o = getBoxObjectPointer(ObjID);

        const int3 &subdiv = o->getSubdivisionCount();

        auto *box_new = (new Box(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = box_new;
    } else if (type == OBJECT_TYPE_DISK) {
        Disk *o = getDiskObjectPointer(ObjID);

        const int2 &subdiv = o->getSubdivisionCount();

        auto *disk_new = (new Disk(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = disk_new;
    } else if (type == OBJECT_TYPE_POLYMESH) {
        auto *polymesh_new = (new Polymesh(currentObjectID, UUIDs_copy, texturefile.c_str(), this));

        objects[currentObjectID] = polymesh_new;
    } else if (type == OBJECT_TYPE_CONE) {
        Cone *o = getConeObjectPointer(ObjID);

        const std::vector<vec3> &nodes = o->getNodeCoordinates();
        const std::vector<float> &radius = o->getNodeRadii();
        uint subdiv = o->getSubdivisionCount();

        auto *cone_new = (new Cone(currentObjectID, UUIDs_copy, nodes.at(0), nodes.at(1), radius.at(0), radius.at(1), subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = cone_new;
    }

    copyObjectData(ObjID, currentObjectID);

    float T[16];
    getObjectPointer(ObjID)->getTransformationMatrix(T);

    getObjectPointer(currentObjectID)->setTransformationMatrix(T);

    currentObjectID++;
    return currentObjectID - 1;
}

std::vector<uint> Context::filterObjectsByData(const std::vector<uint> &IDs, const char *object_data, float threshold, const char *comparator) const {
    std::vector<uint> output_object_IDs;
    output_object_IDs.resize(IDs.size());
    uint passed_count = 0;

    for (uint i = 0; i < IDs.size(); i++) {
        if (doesObjectDataExist(IDs.at(i), object_data)) {
            HeliosDataType type = getObjectDataType(IDs.at(i), object_data);
            if (type == HELIOS_TYPE_UINT) {
                uint R;
                getObjectData(IDs.at(i), object_data, R);
                if (strcmp(comparator, "<") == 0) {
                    if (float(R) < threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, ">") == 0) {
                    if (float(R) > threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, "=") == 0) {
                    if (float(R) == threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }
            } else if (type == HELIOS_TYPE_FLOAT) {
                float R;
                getObjectData(IDs.at(i), object_data, R);

                if (strcmp(comparator, "<") == 0) {
                    if (R < threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, ">") == 0) {
                    if (R > threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, "=") == 0) {
                    if (R == threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }
            } else if (type == HELIOS_TYPE_INT) {
                int R;
                getObjectData(IDs.at(i), object_data, R);

                if (strcmp(comparator, "<") == 0) {
                    if (float(R) < threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, ">") == 0) {
                    if (float(R) > threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                } else if (strcmp(comparator, "=") == 0) {
                    if (float(R) == threshold) {
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }
            } else {
                std::cerr << "WARNING: Object data not of type UINT, INT, or FLOAT. Filtering for other types not yet supported." << std::endl;
            }
        }
    }

    output_object_IDs.resize(passed_count);

    return output_object_IDs;
}

void Context::translateObject(uint ObjID, const vec3 &shift) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::translateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->translate(shift);
}

void Context::translateObject(const std::vector<uint> &ObjIDs, const vec3 &shift) const {
    for (uint ID: ObjIDs) {
        translateObject(ID, shift);
    }
}

void Context::rotateObject(uint ObjID, float rotation_radians, const char *rotation_axis_xyz) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, rotation_axis_xyz);
}

void Context::rotateObject(const std::vector<uint> &ObjIDs, float rotation_radians, const char *rotation_axis_xyz) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, rotation_axis_xyz);
    }
}

void Context::rotateObject(uint ObjID, float rotation_radians, const vec3 &rotation_axis_vector) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, rotation_axis_vector);
}

void Context::rotateObject(const std::vector<uint> &ObjIDs, float rotation_radians, const vec3 &rotation_axis_vector) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, rotation_axis_vector);
    }
}

void Context::rotateObject(uint ObjID, float rotation_radians, const vec3 &rotation_origin, const vec3 &rotation_axis_vector) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, rotation_origin, rotation_axis_vector);
}

void Context::rotateObject(const std::vector<uint> &ObjIDs, float rotation_radians, const vec3 &rotation_origin, const vec3 &rotation_axis_vector) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, rotation_origin, rotation_axis_vector);
    }
}

void Context::rotateObjectAboutOrigin(uint ObjID, float rotation_radians, const vec3 &rotation_axis_vector) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::rotateObjectAboutOrigin): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->rotate(rotation_radians, objects.at(ObjID)->object_origin, rotation_axis_vector);
}

void Context::rotateObjectAboutOrigin(const std::vector<uint> &ObjIDs, float rotation_radians, const vec3 &rotation_axis_vector) const {
    for (uint ID: ObjIDs) {
        rotateObject(ID, rotation_radians, objects.at(ID)->object_origin, rotation_axis_vector);
    }
}

void Context::scaleObject(uint ObjID, const helios::vec3 &scalefact) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scale(scalefact);
}

void Context::scaleObject(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact) const {
    for (uint ID: ObjIDs) {
        scaleObject(ID, scalefact);
    }
}

void Context::scaleObjectAboutCenter(uint ObjID, const helios::vec3 &scalefact) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObjectAboutCenter): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scaleAboutCenter(scalefact);
}

void Context::scaleObjectAboutCenter(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact) const {
    for (uint ID: ObjIDs) {
        scaleObjectAboutCenter(ID, scalefact);
    }
}

void Context::scaleObjectAboutPoint(uint ObjID, const helios::vec3 &scalefact, const helios::vec3 &point) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObjectAboutPoint): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scaleAboutPoint(scalefact, point);
}

void Context::scaleObjectAboutPoint(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact, const helios::vec3 &point) const {
    for (uint ID: ObjIDs) {
        scaleObjectAboutPoint(ID, scalefact, point);
    }
}

void Context::scaleObjectAboutOrigin(uint ObjID, const helios::vec3 &scalefact) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::scaleObjectAboutOrigin): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    getObjectPointer(ObjID)->scaleAboutPoint(scalefact, objects.at(ObjID)->object_origin);
}

void Context::scaleObjectAboutOrigin(const std::vector<uint> &ObjIDs, const helios::vec3 &scalefact) const {
    for (uint ID: ObjIDs) {
        scaleObjectAboutPoint(ID, scalefact, objects.at(ID)->object_origin);
    }
}

std::vector<uint> Context::getObjectPrimitiveUUIDs(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID) && ObjID != 0) {
        helios_runtime_error("ERROR (Context::getObjectPrimitiveUUIDs): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif

    if (ObjID == 0) {
        // \todo This is inefficient and should be improved by storing the UUIDs for all objID = 0 primitives in the Context.
        std::vector<uint> UUIDs;
        UUIDs.reserve(getPrimitiveCount());
        for (uint UUID: getAllUUIDs()) {
            if (getPrimitiveParentObjectID(UUID) == 0) {
                UUIDs.push_back(UUID);
            }
        }
        return UUIDs;
    }

    return getObjectPointer(ObjID)->getPrimitiveUUIDs();
}

std::vector<uint> Context::getObjectPrimitiveUUIDs(const std::vector<uint> &ObjIDs) const {
    std::vector<uint> output_UUIDs;

    for (uint ObjID: ObjIDs) {
#ifdef HELIOS_DEBUG
        if (!doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (Context::getObjectPrimitiveUUIDs): Object ID of " + std::to_string(ObjID) + " not found in the context.");
        }
#endif
        const std::vector<uint> &current_UUIDs = getObjectPrimitiveUUIDs(ObjID);
        output_UUIDs.insert(output_UUIDs.end(), current_UUIDs.begin(), current_UUIDs.end());
    }
    return output_UUIDs;
}

std::vector<uint> Context::getObjectPrimitiveUUIDs(const std::vector<std::vector<uint> > &ObjIDs) const {
    std::vector<uint> output_UUIDs;

    for (uint j = 0; j < ObjIDs.size(); j++) {
        for (uint i = 0; i < ObjIDs.at(j).size(); i++) {
#ifdef HELIOS_DEBUG
            if (!doesObjectExist(ObjIDs.at(j).at(i))) {
                helios_runtime_error("ERROR (Context::getObjectPrimitiveUUIDs): Object ID of " + std::to_string(ObjIDs.at(j).at(i)) + " not found in the context.");
            }
#endif

            const std::vector<uint> &current_UUIDs = getObjectPointer(ObjIDs.at(j).at(i))->getPrimitiveUUIDs();
            output_UUIDs.insert(output_UUIDs.end(), current_UUIDs.begin(), current_UUIDs.end());
        }
    }
    return output_UUIDs;
}

helios::ObjectType Context::getObjectType(uint ObjID) const {
    if (ObjID == 0) {
        return OBJECT_TYPE_NONE;
    }
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::getObjectType): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    return getObjectPointer(ObjID)->getObjectType();
}

float Context::getTileObjectAreaRatio(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::getTileObjectAreaRatio): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
#endif
    if (getObjectPointer(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
        std::cerr << "WARNING (Context::getTileObjectAreaRatio): ObjectID " << ObjID << " is not a tile object. Skipping..." << std::endl;
        return 0.0;
    }

    if (!(getObjectPointer(ObjID)->arePrimitivesComplete())) {
        std::cerr << "WARNING (Context::getTileObjectAreaRatio): ObjectID " << ObjID << " is missing primitives. Area ratio calculated is area of non-missing subpatches divided by the area of an individual subpatch." << std::endl;
    }

    const int2 &subdiv = getTileObjectPointer(ObjID)->getSubdivisionCount();
    if (subdiv.x == 1 && subdiv.y == 1) {
        return 1.0;
    }

    float area = getTileObjectPointer(ObjID)->getArea();
    const vec2 size = getTileObjectPointer(ObjID)->getSize();

    float subpatch_area = size.x * size.y / scast<float>(subdiv.x * subdiv.y);
    return area / subpatch_area;
}

std::vector<float> Context::getTileObjectAreaRatio(const std::vector<uint> &ObjIDs) const {
    std::vector<float> AreaRatios(ObjIDs.size());
    for (uint i = 0; i < ObjIDs.size(); i++) {
        AreaRatios.at(i) = getTileObjectAreaRatio(ObjIDs.at(i));
    }

    return AreaRatios;
}

void Context::setTileObjectSubdivisionCount(const std::vector<uint> &ObjIDs, const int2 &new_subdiv) {
    //check that all objects are Tile Objects, and get vector of texture files
    std::vector<uint> tile_ObjectIDs;
    std::vector<uint> textured_tile_ObjectIDs;


    std::vector<std::string> tex;

    for (uint ObjID: ObjIDs) {
#ifdef HELIOS_DEBUG
        if (!doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (Context::setTileObjectSubdivisionCount): Object ID of " + std::to_string(ObjID) + " not found in the context.");
        }
#endif

        //check if the object ID is a tile object and if it is add it the tile_ObjectIDs vector
        if (getObjectPointer(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is not a tile object. Skipping..." << std::endl;
        } else if (!(getObjectPointer(ObjID)->arePrimitivesComplete())) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is missing primitives. Skipping..." << std::endl;
        } else {
            //test if the tile is textured and push into two different vectors
            Patch *p = getPatchPointer_private(getObjectPointer(ObjID)->getPrimitiveUUIDs().at(0));
            if (!p->hasTexture()) { //no texture
                tile_ObjectIDs.push_back(ObjID);
            } else { //texture
                textured_tile_ObjectIDs.push_back(ObjID);
                tex.push_back(p->getTextureFile());
            }
        }
    }

    //Here just call setSubdivisionCount directly for the non-textured tile objects
    for (unsigned int tile_ObjectID: tile_ObjectIDs) {
        Tile *current_object_pointer = getTileObjectPointer(tile_ObjectID);
        const std::vector<uint> &UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);
        RGBcolor color = getPrimitiveColor(UUIDs_old.front());

        std::vector<uint> UUIDs_new = addTile(center, size, rotation, new_subdiv, color);

        for (uint UUID: UUIDs_new) {
            getPrimitivePointer_private(UUID)->setParentObjectID(tile_ObjectID);
        }

        current_object_pointer->setPrimitiveUUIDs(UUIDs_new);
        current_object_pointer->setSubdivisionCount(new_subdiv);
        deletePrimitive(UUIDs_old);
    }

    // get a vector of unique texture files that are represented in the input tile objects
    sort(tex.begin(), tex.end());
    std::vector<std::string>::iterator it;
    it = std::unique(tex.begin(), tex.end());
    tex.resize(std::distance(tex.begin(), it));

    //create object templates for all the unique texture files
    std::vector<uint> object_templates;
    std::vector<std::vector<uint> > template_primitives;
    for (uint j = 0; j < tex.size(); j++) {
        //create a template object for the current texture
        uint object_template = addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, new_subdiv, tex.at(j).c_str());
        object_templates.emplace_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.emplace_back(object_primitives);
    }

    //keep loop over objects on the outside, otherwise need to update textured_tile_ObjectIDs vector all the time
    //for each textured tile object
    for (uint i = 0; i < textured_tile_ObjectIDs.size(); i++) {
        //get info from current object
        Tile *current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
        std::string current_texture_file = current_object_pointer->getTextureFile();

        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);

        //for unique textures
        for (uint j = 0; j < tex.size(); j++) {
            //if the current tile object has the same texture file as the current unique texture file
            if (current_texture_file == tex.at(j)) {
                //copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));

                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);

                //delete the original object primitives
                deletePrimitive(UUIDs_old);

                float IM[16];
                makeIdentityMatrix(IM);
                current_object_pointer->setTransformationMatrix(IM);

                current_object_pointer->scale(make_vec3(size.x, size.y, 1));

                //transform based on original object data
                if (rotation.elevation != 0) {
                    current_object_pointer->rotate(-rotation.elevation, "x");
                }
                if (rotation.azimuth != 0) {
                    current_object_pointer->rotate(rotation.azimuth, "z");
                }
                current_object_pointer->translate(center);
            }
        }
    }


    //delete the template (objects and primitives)
    deleteObject(object_templates);
}

void Context::setTileObjectSubdivisionCount(const std::vector<uint> &ObjIDs, float area_ratio) {
    //check that all objects are Tile Objects, and get vector of texture files
    std::vector<uint> tile_ObjectIDs;
    std::vector<uint> textured_tile_ObjectIDs;

    std::vector<std::string> tex;
    // for(uint i=1;i<ObjectIDs.size();i++)
    for (uint ObjID: ObjIDs) {
#ifdef HELIOS_DEBUG
        if (!doesObjectExist(ObjID)) {
            helios_runtime_error("ERROR (Context::setTileObjectSubdivisionCount): Object ID of " + std::to_string(ObjID) + " not found in the context.");
        }
#endif

        //check if the object ID is a tile object and if it is add it the tile_ObjectIDs vector
        if (getObjectPointer(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is not a tile object. Skipping..." << std::endl;
        } else if (!(getObjectPointer(ObjID)->arePrimitivesComplete())) {
            std::cerr << "WARNING (Context::setTileObjectSubdivisionCount): ObjectID " << ObjID << " is missing primitives. Skipping..." << std::endl;
        } else {
            //test if the tile is textured and push into two different vectors
            Patch *p = getPatchPointer_private(getObjectPointer(ObjID)->getPrimitiveUUIDs().at(0));
            if (!p->hasTexture()) { //no texture
                tile_ObjectIDs.push_back(ObjID);
            } else { //texture
                textured_tile_ObjectIDs.push_back(ObjID);
                tex.push_back(p->getTextureFile());
            }
        }
    }

    //Here just call setSubdivisionCount directly for the non-textured tile objects
    for (uint i = 0; i < tile_ObjectIDs.size(); i++) {
        Tile *current_object_pointer = getTileObjectPointer(tile_ObjectIDs.at(i));
        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);
        RGBcolor color = getPrimitiveColor(UUIDs_old.front());

        float tile_area = current_object_pointer->getArea();

        // subpatch dimensions needed to keep the correct ratio and have the solid fraction area = the input area
        float subpatch_dimension = sqrtf(tile_area / area_ratio);
        float subpatch_per_x = size.x / subpatch_dimension;
        float subpatch_per_y = size.y / subpatch_dimension;

        float option_1_AR = (tile_area / (size.x / ceil(subpatch_per_x) * size.y / floor(subpatch_per_y))) - area_ratio;
        float option_2_AR = (tile_area / (size.x / floor(subpatch_per_x) * size.y / ceil(subpatch_per_y))) - area_ratio;

        int2 new_subdiv;
        if ((int) area_ratio == 1) {
            new_subdiv = make_int2(1, 1);
        } else if (option_1_AR >= option_2_AR) {
            new_subdiv = make_int2(ceil(subpatch_per_x), floor(subpatch_per_y));
        } else {
            new_subdiv = make_int2(floor(subpatch_per_x), ceil(subpatch_per_y));
        }


        std::vector<uint> UUIDs_new = addTile(center, size, rotation, new_subdiv, color);

        for (uint UUID: UUIDs_new) {
            getPrimitivePointer_private(UUID)->setParentObjectID(tile_ObjectIDs.at(i));
        }

        current_object_pointer->setPrimitiveUUIDs(UUIDs_new);
        current_object_pointer->setSubdivisionCount(new_subdiv);
        deletePrimitive(UUIDs_old);
    }

    // get a vector of unique texture files that are represented in the input tile objects
    sort(tex.begin(), tex.end());
    std::vector<std::string>::iterator it;
    it = std::unique(tex.begin(), tex.end());
    tex.resize(std::distance(tex.begin(), it));

    //create object templates for all the unique texture files
    // the assumption here is that all tile objects with the same texture have the same aspect ratio
    //if this is not true then the copying method won't work well because a new template will need to be created for each texture/aspect ratio combination

    std::vector<uint> object_templates;
    std::vector<std::vector<uint> > template_primitives;
    for (uint j = 0; j < tex.size(); j++) {
        //here we just want to get one tile object with the matching texture
        uint ii;
        for (uint i = 0; i < textured_tile_ObjectIDs.size(); i++) {
            //get info from current object
            Tile *current_object_pointer_b = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
            std::string current_texture_file_b = current_object_pointer_b->getTextureFile();
            //if the current tile object has the same texture file as the current unique texture file
            if (current_texture_file_b == tex.at(j)) {
                ii = i;
                break;
            }
        }

        //get info from current object
        Tile *current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(ii));
        vec2 tile_size = current_object_pointer->getSize();
        float tile_area = current_object_pointer->getArea();

        // subpatch dimensions needed to keep the correct ratio and have the solid fraction area = the input area
        float subpatch_dimension = sqrtf(tile_area / area_ratio);
        float subpatch_per_x = tile_size.x / subpatch_dimension;
        float subpatch_per_y = tile_size.y / subpatch_dimension;

        float option_1_AR = (tile_area / (tile_size.x / ceil(subpatch_per_x) * tile_size.y / floor(subpatch_per_y))) - area_ratio;
        float option_2_AR = (tile_area / (tile_size.x / floor(subpatch_per_x) * tile_size.y / ceil(subpatch_per_y))) - area_ratio;

        int2 new_subdiv;
        if ((int) area_ratio == 1) {
            new_subdiv = make_int2(1, 1);
        } else if (option_1_AR >= option_2_AR) {
            new_subdiv = make_int2(ceil(subpatch_per_x), floor(subpatch_per_y));
        } else {
            new_subdiv = make_int2(floor(subpatch_per_x), ceil(subpatch_per_y));
        }

        //create a template object for the current texture
        uint object_template = addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, new_subdiv, tex.at(j).c_str());
        object_templates.emplace_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.emplace_back(object_primitives);
    }

    //keep loop over objects on the outside, otherwise need to update textured_tile_ObjectIDs vector all the time
    //for each textured tile object
    for (uint i = 0; i < textured_tile_ObjectIDs.size(); i++) {
        //get info from current object
        Tile *current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
        // std::string current_texture_file = getPrimitivePointer_private(current_object_pointer->getPrimitiveUUIDs().at(0))->getTextureFile();
        std::string current_texture_file = current_object_pointer->getTextureFile();
        // std::cout << "current_texture_file for ObjID " << textured_tile_ObjectIDs.at(i) << " = " << current_texture_file << std::endl;
        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);

        //for unique textures
        for (uint j = 0; j < tex.size(); j++) {
            //if the current tile object has the same texture file as the current unique texture file
            if (current_texture_file == tex.at(j)) {
                //copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));

                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));

                int2 new_subdiv = getTileObjectPointer(object_templates.at(j))->getSubdivisionCount();
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);

                //delete the original object primitives
                deletePrimitive(UUIDs_old);

                float IM[16];
                makeIdentityMatrix(IM);
                current_object_pointer->setTransformationMatrix(IM);

                current_object_pointer->scale(make_vec3(size.x, size.y, 1));

                if (rotation.elevation != 0) {
                    current_object_pointer->rotate(-rotation.elevation, "x");
                }
                if (rotation.azimuth != 0) {
                    current_object_pointer->rotate(rotation.azimuth, "z");
                }
                current_object_pointer->translate(center);
            }
        }
    }

    //delete the template (objects and primitives)
    deleteObject(object_templates);
}

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

    //subcenter = make_vec3(-0.5*size.x+(float(i)+0.5)*subsize.x,-0.5*size.y+(float(j)+0.5)*subsize.y,0);
    //Y[0] = make_vec3( -0.5f, -0.5f, 0.f);
    //Y[1] = make_vec3( 0.5f, -0.5f, 0.f);
    //Y[2] = make_vec3( 0.5f, 0.5f, 0.f);
    //Y[3] = make_vec3( -0.5f, 0.5f, 0.f);


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

Tube::Tube(uint a_OID, const std::vector<uint> &a_UUIDs, const std::vector<vec3> &a_nodes, const std::vector<float> &a_radius, const std::vector<helios::RGBcolor> &a_colors, const std::vector<std::vector<helios::vec3> > &a_triangle_vertices,
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

std::vector<std::vector<helios::vec3> > Tube::getTriangleVertices() const {
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
            if (fabs(axial_vector * initial_radial) > 0.99f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
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
            if (rotation_axis.magnitude() > 1e-6) {
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
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

    //add triangles for new segment

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
    std::vector<std::vector<vec2> > uv;
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
            if (fabs(axial_vector * initial_radial) > 0.99f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
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
            if (rotation_axis.magnitude() > 1e-6) {
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
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

Cone::Cone(uint a_OID, const std::vector<uint> &a_UUIDs, const vec3 &a_node0, const vec3 &a_node1, float a_radius0,
           float a_radius1, uint a_subdiv, const char *a_texturefile, helios::Context *a_context) {
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
    //get the nodes and radii of the nodes with transformation matrix applied
    const std::vector<helios::vec3> &nodes_T = context->getConeObjectPointer(OID)->getNodeCoordinates();
    const std::vector<float> &radii_T = context->getConeObjectPointer(OID)->getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z);
    float length = powf(powf(axis_unit_vector.x, 2) + powf(axis_unit_vector.y, 2) + powf(axis_unit_vector.z, 2), 0.5);
    axis_unit_vector = axis_unit_vector / length;

    //translate node 0 back to origin
    context->getConeObjectPointer(OID)->translate(-1.0 * nodes_T.at(0));

    //rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0, 0, 1);
    //get the axis about which to rotate
    vec3 ra = cross(z_axis, axis_unit_vector);
    //get the angle to rotate
    float dot = axis_unit_vector.x * z_axis.x + axis_unit_vector.y * z_axis.y + axis_unit_vector.z * z_axis.z;
    float angle = acos_safe(dot);

    //only rotate if the cone is not alread aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
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

    //rotate back
    if (angle != 0.0) {
        context->getConeObjectPointer(OID)->rotate(angle, ra);
    }

    // translate back
    context->getConeObjectPointer(OID)->translate(nodes_T.at(0));
}

void Cone::scaleGirth(float S) {
    //get the nodes and radii of the nodes with transformation matrix applied
    const std::vector<helios::vec3> &nodes_T = context->getConeObjectPointer(OID)->getNodeCoordinates();
    const std::vector<float> &radii_T = context->getConeObjectPointer(OID)->getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z);
    axis_unit_vector.normalize();

    //translate node 0 back to origin
    context->getConeObjectPointer(OID)->translate(-1.0 * nodes_T.at(0));
    //rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0, 0, 1);
    //get the axis about which to rotate
    vec3 ra = cross(z_axis, axis_unit_vector);
    //get the angle to rotate
    float dot = axis_unit_vector * z_axis;
    float angle = acos_safe(dot);
    //only rotate if the cone is not already aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
    if (angle != float(0.0)) {
        context->getConeObjectPointer(OID)->rotate(-1 * angle, ra);
    }

    // scale the cone in the x and y dimensions
    context->scaleObject(OID, make_vec3(S, S, 1));


    //rotate back
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

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius) {
    return addSphereObject(Ndivs, center, {radius, radius, radius}, {0.f, 0.75f, 0.f}); //Default color is green
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color) {
    return addSphereObject(Ndivs, center, {radius, radius, radius}, color);
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const char *texturefile) {
    return addSphereObject(Ndivs, center, {radius, radius, radius}, texturefile);
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius) {
    return addSphereObject(Ndivs, center, radius, {0.f, 0.75f, 0.f}); //Default color is green
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

    //bottom cap
    for (int j = 0; j < Ndivs; j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F, 0));
        vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + dtheta, float(j) * dphi));
        vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, -0.5f * PI_F + dtheta, float(j + 1) * dphi));
        vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);

        UUID.push_back(addTriangle(v0, v1, v2, color));
    }

    //top cap
    for (int j = 0; j < Ndivs; j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F, 0));
        vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);

        UUID.push_back(addTriangle(v2, v1, v0, color));
    }

    //middle
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

    //bottom cap
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

        UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
    }

    //top cap
    for (int j = 0; j < Ndivs; j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F, 0));
        vec3 v0 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v1 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v2 = center + make_vec3(cart.x * radius.x, cart.y * radius.y, cart.z * radius.z);;

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

        UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
    }

    //middle
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

            UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
            UUID.push_back(addTriangle(v0, v2, v3, texturefile, uv0, uv2, uv3));
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
    RGBcolor color(0.f, 0.75f, 0.f); //Default color is green

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
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addTileObject): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addTileObject): Texture file " + std::string(texturefile) + " does not exist.");
    } else if (size.x == 0 || size.y == 0) {
        helios_runtime_error("ERROR (Context::addTileObject): Size of tile must be greater than 0.");
    } else if (subdiv.x < 1 || subdiv.y < 1) {
        helios_runtime_error("ERROR (Context::addTileObject): Number of tile subdivisions must be greater than 0.");
    }

    std::vector<uint> UUID;
    UUID.reserve(subdiv.x * subdiv.y);

    vec2 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);

    std::vector<helios::vec2> uv(4);
    vec2 uv_sub;
    uv_sub.x = 1.f / float(subdiv.x);
    uv_sub.y = 1.f / float(subdiv.y);

    addTexture(texturefile);

    const int2 &sz = textures.at(texturefile).getImageResolution();
    if (subdiv.x >= sz.x || subdiv.y >= sz.y) {
        helios_runtime_error("ERROR (Context::addTileObject): The resolution of the texture image '" + std::string(texturefile) + "' is lower than the number of tile subdivisions. Increase resolution of the texture image.");
    }

    for (uint j = 0; j < subdiv.y; j++) {
        for (uint i = 0; i < subdiv.x; i++) {
            vec3 subcenter = make_vec3(-0.5f * size.x + (float(i) + 0.5f) * subsize.x, -0.5f * size.y + (float(j) + 0.5f) * subsize.y, 0.f);

            uv.at(0) = make_vec2(float(i) * uv_sub.x, float(j) * uv_sub.y);
            uv.at(1) = make_vec2(float(i + 1) * uv_sub.x, float(j) * uv_sub.y);
            uv.at(2) = make_vec2(float(i + 1) * uv_sub.x, float(j + 1) * uv_sub.y);
            uv.at(3) = make_vec2(float(i) * uv_sub.x, float(j + 1) * uv_sub.y);

            auto *patch_new = (new Patch(texturefile, uv, textures, 0, currentUUID));

            // \todo This is causing problems in the radiation intersection.
            // if( patch_new->getSolidFraction()==0 ){
            //   delete patch_new;
            //   continue;
            // }

            assert(size.x>0.f && size.y>0.f);
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
        color.at(i) = make_RGBcolor(0.f, 0.75f, 0.f); //Default color is green
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

    vec3 axial_vector;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3> > triangle_vertices;
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
            if (fabs(axial_vector * initial_radial) > 0.99f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
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
            if (rotation_axis.magnitude() > 1e-6) {
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
            }
        }

        previous_axial_vector = axial_vector;

        vec3 radial_dir = previous_radial_dir;
        vec3 orthogonal_dir = cross(radial_dir, axial_vector);
        orthogonal_dir.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal = cfact[j] * radius[i] * radial_dir + sfact[j] * radius[i] * orthogonal_dir;
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

    vec3 axial_vector;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3> > triangle_vertices;
    resize_vector(triangle_vertices, radial_subdivisions + 1, node_count);
    std::vector<std::vector<vec2> > uv;
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
            if (fabs(axial_vector * initial_radial) > 0.99f) {
                initial_radial = vec3(0.0f, 1.0f, 0.0f); // Avoid parallel vectors
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
            if (rotation_axis.magnitude() > 1e-6) {
                float angle = acos(std::clamp(previous_axial_vector * axial_vector, -1.0f, 1.0f));
                previous_radial_dir = rotatePointAboutLine(previous_radial_dir, nullorigin, rotation_axis, angle);
            }
        }

        previous_axial_vector = axial_vector;

        vec3 radial_dir = previous_radial_dir;
        vec3 orthogonal_dir = cross(radial_dir, axial_vector);
        orthogonal_dir.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal = cfact[j] * radius[i] * radial_dir + sfact[j] * radius[i] * orthogonal_dir;
            triangle_vertices[i][j] = nodes[i] + normal;

            uv[i][j].x = textureuv_ufrac[i];
            uv[i][j].y = float(j) / float(radial_subdivisions);
        }
    }

    std::vector<uint> UUIDs(2 * (node_count - 1) * radial_subdivisions);
    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;

    int ii = 0;
    for (int j = 0; j < radial_subdivisions; j++) {
        for (int i = 0; i < node_count - 1; i++) {
            v0 = triangle_vertices[i][j];
            v1 = triangle_vertices[i + 1][j + 1];
            v2 = triangle_vertices[i][j + 1];

            uv0 = uv[i][j];
            uv1 = uv[i + 1][j + 1];
            uv2 = uv[i][j + 1];

            UUIDs.at(ii) = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);

            v0 = triangle_vertices[i][j];
            v1 = triangle_vertices[i + 1][j];
            v2 = triangle_vertices[i + 1][j + 1];

            uv0 = uv[i][j];
            uv1 = uv[i + 1][j];
            uv2 = uv[i + 1][j + 1];

            UUIDs.at(ii + 1) = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);

            ii += 2;
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
    RGBcolor color(0.f, 0.75f, 0.f); //Default color is green

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

    if (reverse_normals) { //normals point inward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 1.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 0.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5f * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 0.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5f * PI_F, 1.5f * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
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

    if (reverse_normals) { //normals point inward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
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
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile, make_vec2(0.5, 0.5),
                                         make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
            } else {
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), texturefile,
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx / size.x), 0.5f * (1.f + sinf(theta) * ry / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)));
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile,
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
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

    auto *polymesh_new = (new Polymesh(currentObjectID, UUIDs, "", this));

    float T[16], transform[16];
    polymesh_new->getTransformationMatrix(transform);

    makeTranslationMatrix(getPrimitivePointer_private(UUIDs.front())->getVertices().front(), T);
    matmult(T, transform, transform);
    polymesh_new->setTransformationMatrix(transform);

    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = polymesh_new;
    currentObjectID++;

    uint objID = currentObjectID - 1;
    polymesh_new->object_origin = getObjectCenter(objID);

    return objID;
}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1) {
    RGBcolor color(0.f, 0.75f, 0.f); //Default color is green
    return addConeObject(Ndivs, node0, node1, radius0, radius1, color);
}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const RGBcolor &color) {
    const std::vector nodes{node0, node1};
    const std::vector radii{radius0, radius1};

    vec3 convec;
    std::vector<float> cfact(Ndivs + 1);
    std::vector<float> sfact(Ndivs + 1);
    std::vector<std::vector<vec3> > xyz(Ndivs + 1);
    std::vector<std::vector<vec3> > normal(Ndivs + 1);

    for (uint j = 0; j < Ndivs + 1; j++) {
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
    }
    vec3 nvec(0.1817f, 0.6198f, 0.7634f); //random vector to get things going

    for (int j = 0; j < Ndivs + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(Ndivs));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(Ndivs));
    }

    for (int i = 0; i < 2; i++) {
        vec3 vec;
        //looping over cone segments

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
    std::vector<std::vector<vec3> > xyz, normal;
    std::vector<std::vector<vec2> > uv;
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
        //looping over cone segments

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
                UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
            }

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            uv0 = uv[j][i];
            uv1 = uv[j][i + 1];
            uv2 = uv[j + 1][i + 1];

            if ((v1 - v0).magnitude() > 1e-6 && (v2 - v0).magnitude() > 1e-6 && (v2 - v1).magnitude() > 1e-6) {
                UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
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

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius) {
    RGBcolor color = make_RGBcolor(0.f, 0.75f, 0.f); //Default color is green

    return addSphere(Ndivs, center, radius, color);
}

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color) {
    std::vector<uint> UUID;

    float dtheta = PI_F / float(Ndivs);
    float dphi = 2.0f * PI_F / float(Ndivs);

    //bottom cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j + 1) * dphi));

        UUID.push_back(addTriangle(v0, v1, v2, color));
    }

    //top cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j + 1) * dphi));

        UUID.push_back(addTriangle(v2, v1, v0, color));
    }

    //middle
    for (int j = 0; j < Ndivs; j++) {
        for (int i = 1; i < Ndivs - 1; i++) {
            vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j) * dphi));
            vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j) * dphi));
            vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j + 1) * dphi));

            UUID.push_back(addTriangle(v0, v1, v2, color));
            UUID.push_back(addTriangle(v0, v2, v3, color));
        }
    }

    return UUID;
}

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addSphere): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addSphere): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;

    float dtheta = PI_F / float(Ndivs);
    float dphi = 2.0f * PI_F / float(Ndivs);

    //bottom cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, float(j + 1) * dphi));

        vec3 n0 = v0 - center;
        n0.normalize();
        vec3 n1 = v1 - center;
        n1.normalize();
        vec3 n2 = v2 - center;
        n2.normalize();

        vec2 uv0 = make_vec2(1.f - atan2f(sin((float(j) + 0.5f) * dphi), -cos((float(j) + 0.5f) * dphi)) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
        vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
        vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);

        if (j == Ndivs - 1) {
            uv2.x = 1;
        }

        UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
    }

    //top cap
    for (int j = 0; j < Ndivs; j++) {
        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, float(j) * dphi));

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

        UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
    }

    //middle
    for (int j = 0; j < Ndivs; j++) {
        for (int i = 1; i < Ndivs - 1; i++) {
            vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j) * dphi));
            vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j) * dphi));
            vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + float(i) * dtheta, float(j + 1) * dphi));

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

            UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
            UUID.push_back(addTriangle(v0, v2, v3, texturefile, uv0, uv2, uv3));
        }
    }

    return UUID;
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv) {
    RGBcolor color(0.f, 0.75f, 0.f); //Default color is green

    return addTile(center, size, rotation, subdiv, color);
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const RGBcolor &color) {
    vec2 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);

    std::vector<uint> UUID(subdiv.x * subdiv.y);

    size_t t = 0;
    for (uint j = 0; j < subdiv.y; j++) {
        for (uint i = 0; i < subdiv.x; i++) {
            vec3 subcenter = make_vec3(-0.5f * size.x + (float(i) + 0.5f) * subsize.x, -0.5f * size.y + (float(j) + 0.5f) * subsize.y, 0);

            UUID[t] = addPatch(subcenter, subsize, make_SphericalCoord(0, 0), color);

            if (rotation.elevation != 0.f) {
                getPrimitivePointer_private(UUID[t])->rotate(-rotation.elevation, "x");
            }
            if (rotation.azimuth != 0.f) {
                getPrimitivePointer_private(UUID[t])->rotate(-rotation.azimuth, "z");
            }
            getPrimitivePointer_private(UUID[t])->translate(center);

            t++;
        }
    }

    return UUID;
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addTile): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addTile): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;

    vec2 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);

    vec3 subcenter;

    std::vector<helios::vec2> uv(4);
    vec2 uv_sub;
    uv_sub.x = 1.f / float(subdiv.x);
    uv_sub.y = 1.f / float(subdiv.y);

    addTexture(texturefile);
    std::vector<std::vector<bool> > alpha;
    if (textures.at(texturefile).hasTransparencyChannel()) {
        alpha = *textures.at(texturefile).getTransparencyData();
    }

    int2 sz = textures.at(texturefile).getImageResolution();
    if (subdiv.x >= sz.x || subdiv.y >= sz.y) {
        helios_runtime_error("ERROR (Context::addTile): The resolution of the texture image '" + std::string(texturefile) + "' is lower than the number of tile subdivisions. Increase resolution of the texture image.");
    }

    for (uint j = 0; j < subdiv.y; j++) {
        for (uint i = 0; i < subdiv.x; i++) {
            subcenter = make_vec3(-0.5f * size.x + (float(i) + 0.5f) * subsize.x, -0.5f * size.y + (float(j) + 0.5f) * subsize.y, 0);

            uv[0] = make_vec2(float(i) * uv_sub.x, float(j) * uv_sub.y);
            uv[1] = make_vec2(float(i + 1) * uv_sub.x, float(j) * uv_sub.y);
            uv[2] = make_vec2(float(i + 1) * uv_sub.x, float(j + 1) * uv_sub.y);
            uv[3] = make_vec2(float(i) * uv_sub.x, float(j + 1) * uv_sub.y);

            auto *patch_new = (new Patch(texturefile, uv, textures, 0, currentUUID));

            if (patch_new->getSolidFraction() == 0) {
                delete patch_new;
                continue;
            }

            assert(size.x>0.f && size.y>0.f);
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

    return UUID;
}

std::vector<uint> Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius) {
    std::vector<RGBcolor> color(nodes.size(), make_RGBcolor(0.f, 0.75f, 0.f));

    return addTube(Ndivs, nodes, radius, color);
}

std::vector<uint> Context::addTube(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color) {
    const uint node_count = nodes.size();

    if (node_count == 0) {
        helios_runtime_error("ERROR (Context::addTube): Node and radius arrays are empty.");
    } else if (node_count != radius.size()) {
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree.");
    } else if (node_count != color.size()) {
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `color' arguments must agree.");
    }

    vec3 vec, convec;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3> > xyz;
    resize_vector(xyz, node_count, radial_subdivisions + 1);

    vec3 nvec(0.1817f, 0.6198f, 0.7634f); //random vector to get things going

    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    for (int i = 0; i < node_count; i++) { //looping over tube segments

        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTube): Radius of tube must be positive.");
        }

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == node_count - 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        } else {
            vec.x = 0.5f * ((nodes[i].x - nodes[i - 1].x) + (nodes[i + 1].x - nodes[i].x));
            vec.y = 0.5f * ((nodes[i].y - nodes[i - 1].y) + (nodes[i + 1].y - nodes[i].y));
            vec.z = 0.5f * ((nodes[i].z - nodes[i - 1].z) + (nodes[i + 1].z - nodes[i].z));
        }

        convec = cross(nvec, vec);
        convec.normalize();
        nvec = cross(vec, convec);
        nvec.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            vec3 normal;
            normal.x = cfact[j] * radius[i] * nvec.x + sfact[j] * radius[i] * convec.x;
            normal.y = cfact[j] * radius[i] * nvec.y + sfact[j] * radius[i] * convec.y;
            normal.z = cfact[j] * radius[i] * nvec.z + sfact[j] * radius[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal.x;
            xyz[j][i].y = nodes[i].y + normal.y;
            xyz[j][i].z = nodes[i].z + normal.z;
        }
    }

    vec3 v0, v1, v2;
    std::vector<uint> UUIDs(2 * (node_count - 1) * radial_subdivisions);

    int ii = 0;
    for (int i = 0; i < node_count - 1; i++) {
        for (int j = 0; j < radial_subdivisions; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            UUIDs.at(ii) = addTriangle(v0, v1, v2, color.at(i));

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            UUIDs.at(ii + 1) = addTriangle(v0, v1, v2, color.at(i));

            ii += 2;
        }
    }

    return UUIDs;
}

std::vector<uint> Context::addTube(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addTube): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addTube): Texture file " + std::string(texturefile) + " does not exist.");
    }

    const uint node_count = nodes.size();

    if (node_count == 0) {
        helios_runtime_error("ERROR (Context::addTube): Node and radius arrays are empty.");
    } else if (node_count != radius.size()) {
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree.");
    }

    vec3 vec, convec;
    std::vector<float> cfact(radial_subdivisions + 1);
    std::vector<float> sfact(radial_subdivisions + 1);
    std::vector<std::vector<vec3> > xyz, normal;
    std::vector<std::vector<vec2> > uv;
    resize_vector(xyz, node_count, radial_subdivisions + 1);
    resize_vector(normal, node_count, radial_subdivisions + 1);
    resize_vector(uv, node_count, radial_subdivisions + 1);

    vec3 nvec(0.1817f, 0.6198f, 0.7634f); //random vector to get things going

    for (int j = 0; j < radial_subdivisions + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(radial_subdivisions));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(radial_subdivisions));
    }

    for (int i = 0; i < node_count; i++) { //looping over tube segments

        if (radius.at(i) < 0) {
            helios_runtime_error("ERROR (Context::addTube): Radius of tube must be positive.");
        }

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == node_count - 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        } else {
            vec.x = 0.5f * ((nodes[i].x - nodes[i - 1].x) + (nodes[i + 1].x - nodes[i].x));
            vec.y = 0.5f * ((nodes[i].y - nodes[i - 1].y) + (nodes[i + 1].y - nodes[i].y));
            vec.z = 0.5f * ((nodes[i].z - nodes[i - 1].z) + (nodes[i + 1].z - nodes[i].z));
        }

        convec = cross(nvec, vec);
        convec.normalize();
        nvec = cross(vec, convec);
        nvec.normalize();

        for (int j = 0; j < radial_subdivisions + 1; j++) {
            normal[j][i].x = cfact[j] * radius[i] * nvec.x + sfact[j] * radius[i] * convec.x;
            normal[j][i].y = cfact[j] * radius[i] * nvec.y + sfact[j] * radius[i] * convec.y;
            normal[j][i].z = cfact[j] * radius[i] * nvec.z + sfact[j] * radius[i] * convec.z;

            xyz[j][i].x = nodes[i].x + normal[j][i].x;
            xyz[j][i].y = nodes[i].y + normal[j][i].y;
            xyz[j][i].z = nodes[i].z + normal[j][i].z;

            uv[j][i].x = float(i) / float(node_count - 1);
            uv[j][i].y = float(j) / float(radial_subdivisions);

            normal[j][i] = normal[j][i] / radius[i];
        }
    }

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    std::vector<uint> UUIDs(2 * (node_count - 1) * radial_subdivisions);

    int ii = 0;
    for (int i = 0; i < node_count - 1; i++) {
        for (int j = 0; j < radial_subdivisions; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            uv0 = uv[j][i];
            uv1 = uv[j + 1][i + 1];
            uv2 = uv[j + 1][i];

            UUIDs.at(ii) = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            uv0 = uv[j][i];
            uv1 = uv[j][i + 1];
            uv2 = uv[j + 1][i + 1];

            UUIDs.at(ii + 1) = addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2);

            ii += 2;
        }
    }

    return UUIDs;
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv) {
    RGBcolor color = make_RGBcolor(0.f, 0.75f, 0.f); //Default color is green

    return addBox(center, size, subdiv, color, false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color) {
    return addBox(center, size, subdiv, color, false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char *texturefile) {
    return addBox(center, size, subdiv, texturefile, false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals) {
    std::vector<uint> UUID;

    vec3 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);
    subsize.z = size.z / float(subdiv.z);

    vec3 subcenter;
    std::vector<uint> U;

    if (reverse_normals) { //normals point inward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), color);
        UUID.insert(UUID.end(), U.begin(), U.end());
    }

    return UUID;
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char *texturefile, bool reverse_normals) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addBox): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addBox): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<uint> UUID;

    vec3 subsize;
    subsize.x = size.x / float(subdiv.x);
    subsize.y = size.y / float(subdiv.y);
    subsize.z = size.z / float(subdiv.z);

    vec3 subcenter;
    std::vector<uint> U;

    if (reverse_normals) { //normals point inward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());
    } else { //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, 0), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //left
        subcenter = center - make_vec3(0, 0.5f * size.y, 0);
        U = addTile(subcenter, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, PI_F), make_int2(subdiv.x, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //back
        subcenter = center - make_vec3(0.5f * size.x, 0, 0);
        U = addTile(subcenter, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 1.5 * PI_F), make_int2(subdiv.y, subdiv.z), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(0, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());

        //bottom
        subcenter = center - make_vec3(0, 0, 0.5f * size.z);
        U = addTile(subcenter, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, 0), make_int2(subdiv.x, subdiv.y), texturefile);
        UUID.insert(UUID.end(), U.begin(), U.end());
    }

    return UUID;
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size) {
    return addDisk(make_int2(Ndivs, 1), center, size, make_SphericalCoord(0, 0), make_RGBAcolor(1, 0, 0, 1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, make_RGBAcolor(1, 0, 0, 1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, make_RGBAcolor(color, 1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, color);
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file) {
    return addDisk(make_int2(Ndivs, 1), center, size, rotation, texture_file);
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color) {
    return addDisk(Ndivs, center, size, rotation, make_RGBAcolor(color, 1));
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color) {
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

    return UUID;
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addDisk): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addDisk): Texture file " + std::string(texturefile) + " does not exist.");
    }

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
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile, make_vec2(0.5, 0.5),
                                         make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
            } else {
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), texturefile,
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx / size.x), 0.5f * (1.f + sinf(theta) * ry / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)));
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texturefile,
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx / size.x), 0.5f * (1.f + sinf(theta_plus) * ry / size.y)), make_vec2(0.5f * (1.f + cosf(theta) * rx_plus / size.x), 0.5f * (1.f + sinf(theta) * ry_plus / size.y)),
                                         make_vec2(0.5f * (1.f + cosf(theta_plus) * rx_plus / size.x), 0.5f * (1.f + sinf(theta_plus) * ry_plus / size.y)));
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
        }
    }

    return UUID;
}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1) {
    RGBcolor color;
    color = make_RGBcolor(0.f, 0.75f, 0.f); //Default color is green

    return addCone(Ndivs, node0, node1, radius0, radius1, color);
}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, RGBcolor &color) {
    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs + 1);
    std::vector<float> sfact(Ndivs + 1);
    std::vector<std::vector<vec3> > xyz, normal;
    xyz.resize(Ndivs + 1);
    normal.resize(Ndivs + 1);
    for (uint j = 0; j < Ndivs + 1; j++) {
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
    }
    vec3 nvec(0.1817f, 0.6198f, 0.7634f); //random vector to get things going

    for (int j = 0; j < Ndivs + 1; j++) {
        cfact[j] = cosf(2.f * PI_F * float(j) / float(Ndivs));
        sfact[j] = sinf(2.f * PI_F * float(j) / float(Ndivs));
    }

    for (int i = 0; i < 2; i++) { //looping over cone segments

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        }

        float norm;
        convec = cross(nvec, vec);
        norm = convec.magnitude();
        convec.x = convec.x / norm;
        convec.y = convec.y / norm;
        convec.z = convec.z / norm;
        nvec = cross(vec, convec);
        norm = nvec.magnitude();
        nvec.x = nvec.x / norm;
        nvec.y = nvec.y / norm;
        nvec.z = nvec.z / norm;


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
    std::vector<uint> UUID;

    for (int i = 0; i < 2 - 1; i++) {
        for (int j = 0; j < Ndivs; j++) {
            v0 = xyz[j][i];
            v1 = xyz[j + 1][i + 1];
            v2 = xyz[j + 1][i];

            UUID.push_back(addTriangle(v0, v1, v2, color));

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            UUID.push_back(addTriangle(v0, v1, v2, color));
        }
    }

    return UUID;
}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const char *texturefile) {
    if (!validateTextureFileExtenstion(texturefile)) {
        helios_runtime_error("ERROR (Context::addCone): Texture file " + std::string(texturefile) + " is not PNG or JPEG format.");
    } else if (!doesTextureFileExist(texturefile)) {
        helios_runtime_error("ERROR (Context::addCone): Texture file " + std::string(texturefile) + " does not exist.");
    }

    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs + 1);
    std::vector<float> sfact(Ndivs + 1);
    std::vector<std::vector<vec3> > xyz, normal;
    std::vector<std::vector<vec2> > uv;
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

    for (int i = 0; i < 2; i++) { //looping over cone segments

        if (i == 0) {
            vec.x = nodes[i + 1].x - nodes[i].x;
            vec.y = nodes[i + 1].y - nodes[i].y;
            vec.z = nodes[i + 1].z - nodes[i].z;
        } else if (i == 1) {
            vec.x = nodes[i].x - nodes[i - 1].x;
            vec.y = nodes[i].y - nodes[i - 1].y;
            vec.z = nodes[i].z - nodes[i - 1].z;
        }

        float norm;
        convec = cross(nvec, vec);
        norm = convec.magnitude();
        convec.x = convec.x / norm;
        convec.y = convec.y / norm;
        convec.z = convec.z / norm;
        nvec = cross(vec, convec);
        norm = nvec.magnitude();
        nvec.x = nvec.x / norm;
        nvec.y = nvec.y / norm;
        nvec.z = nvec.z / norm;

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
                UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
            }

            v0 = xyz[j][i];
            v1 = xyz[j][i + 1];
            v2 = xyz[j + 1][i + 1];

            uv0 = uv[j][i];
            uv1 = uv[j][i + 1];
            uv2 = uv[j + 1][i + 1];

            if ((v1 - v0).magnitude() > 1e-6 && (v2 - v0).magnitude() > 1e-6 && (v2 - v1).magnitude() > 1e-6) {
                UUID.push_back(addTriangle(v0, v1, v2, texturefile, uv0, uv1, uv2));
            }
        }
    }

    return UUID;
}

void Context::colorPrimitiveByDataPseudocolor(const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors) {
    colorPrimitiveByDataPseudocolor(UUIDs, primitive_data, colormap, Ncolors, 9999999, -9999999);
}

void Context::colorPrimitiveByDataPseudocolor(const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors, float data_min, float data_max) {
    std::map<uint, float> pcolor_data;

    float data_min_new = 9999999;
    float data_max_new = -9999999;
    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            std::cerr << "WARNING (Context::colorPrimitiveDataPseudocolor): primitive for UUID " << std::to_string(UUID) << " does not exist. Skipping this primitive." << std::endl;
            continue;
        }

        float dataf = 0;
        if (doesPrimitiveDataExist(UUID, primitive_data.c_str())) {
            if (getPrimitiveDataType(UUID, primitive_data.c_str()) != HELIOS_TYPE_FLOAT && getPrimitiveDataType(UUID, primitive_data.c_str()) != HELIOS_TYPE_INT && getPrimitiveDataType(UUID, primitive_data.c_str()) != HELIOS_TYPE_UINT &&
                getPrimitiveDataType(UUID, primitive_data.c_str()) != HELIOS_TYPE_DOUBLE) {
                std::cerr << "WARNING (Context::colorPrimitiveDataPseudocolor): Only primitive data types of int, uint, float, and double are supported for this function. Skipping this primitive." << std::endl;
                continue;
            }

            if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
                HELIOS_TYPE_FLOAT) {
                float data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = data;
            } else if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
                       HELIOS_TYPE_DOUBLE) {
                double data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = float(data);
            } else if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
                       HELIOS_TYPE_INT) {
                int data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = float(data);
            } else if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
                       HELIOS_TYPE_UINT) {
                uint data;
                getPrimitiveData(UUID, primitive_data.c_str(), data);
                dataf = float(data);
            }
        }

        if (data_min == 9999999 && data_max == -9999999) {
            if (dataf < data_min_new) {
                data_min_new = dataf;
            }
            if (dataf > data_max_new) {
                data_max_new = dataf;
            }
        }

        pcolor_data[UUID] = dataf;
    }

    if (data_min == 9999999 && data_max == -9999999) {
        data_min = data_min_new;
        data_max = data_max_new;
    }

    std::vector<RGBcolor> colormap_data = generateColormap(colormap, Ncolors);

    std::map<std::string, std::vector<std::string> > cmap_texture_filenames;

    for (auto &[UUID, pdata]: pcolor_data) {
        std::string texturefile = getPrimitiveTextureFile(UUID);

        int cmap_ind = std::round((pdata - data_min) / (data_max - data_min) * float(Ncolors - 1));

        if (cmap_ind < 0) {
            cmap_ind = 0;
        } else if (cmap_ind >= Ncolors) {
            cmap_ind = Ncolors - 1;
        }

        if (!texturefile.empty() && primitiveTextureHasTransparencyChannel(UUID)) { // primitive has texture with transparency channel

            overridePrimitiveTextureColor(UUID);
            setPrimitiveColor(UUID, colormap_data.at(cmap_ind));
        } else { // primitive does not have texture with transparency channel - assign constant color

            if (!getPrimitiveTextureFile(UUID).empty()) {
                overridePrimitiveTextureColor(UUID);
            }

            setPrimitiveColor(UUID, colormap_data.at(cmap_ind));
        }
    }
}

std::vector<RGBcolor> Context::generateColormap(const std::vector<helios::RGBcolor> &ctable, const std::vector<float> &cfrac, uint Ncolors) {
    if (Ncolors > 9999) {
        std::cerr << "WARNING (Context::generateColormap): Truncating number of color map textures to maximum value of 9999." << std::endl;
    }

    if (ctable.size() != cfrac.size()) {
        helios_runtime_error("ERROR (Context::generateColormap): The length of arguments 'ctable' and 'cfrac' must match.");
    }
    if (ctable.empty()) {
        helios_runtime_error("ERROR (Context::generateColormap): 'ctable' and 'cfrac' arguments contain empty vectors.");
    }

    std::vector<RGBcolor> color_table(Ncolors);

    for (int i = 0; i < Ncolors; i++) {
        float frac = float(i) / float(Ncolors - 1) * cfrac.back();

        int j;
        for (j = 0; j < cfrac.size() - 1; j++) {
            if (frac >= cfrac.at(j) && frac <= cfrac.at(j + 1)) {
                break;
            }
        }

        float cminus = std::fmaxf(0.f, cfrac.at(j));
        float cplus = std::fminf(1.f, cfrac.at(j + 1));

        float jfrac = (frac - cminus) / (cplus - cminus);

        RGBcolor color;
        color.r = ctable.at(j).r + jfrac * (ctable.at(j + 1).r - ctable.at(j).r);
        color.g = ctable.at(j).g + jfrac * (ctable.at(j + 1).g - ctable.at(j).g);
        color.b = ctable.at(j).b + jfrac * (ctable.at(j + 1).b - ctable.at(j).b);

        color_table.at(i) = color;
    }

    return color_table;
}

std::vector<RGBcolor> Context::generateColormap(const std::string &colormap, uint Ncolors) {
    std::vector<RGBcolor> ctable_c;
    std::vector<float> clocs_c;

    if (colormap == "hot") {
        ctable_c.resize(5);
        ctable_c.at(0) = make_RGBcolor(0.f, 0.f, 0.f);
        ctable_c.at(1) = make_RGBcolor(0.5f, 0.f, 0.5f);
        ctable_c.at(2) = make_RGBcolor(1.f, 0.f, 0.f);
        ctable_c.at(3) = make_RGBcolor(1.f, 0.5f, 0.f);
        ctable_c.at(4) = make_RGBcolor(1.f, 1.f, 0.f);

        clocs_c.resize(5);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.25f;
        clocs_c.at(2) = 0.5f;
        clocs_c.at(3) = 0.75f;
        clocs_c.at(4) = 1.f;
    } else if (colormap == "cool") {
        ctable_c.resize(2);
        ctable_c.at(1) = RGB::cyan;
        ctable_c.at(2) = RGB::magenta;

        clocs_c.resize(2);
        clocs_c.at(1) = 0.f;
        clocs_c.at(2) = 1.f;
    } else if (colormap == "lava") {
        ctable_c.resize(5);
        ctable_c.at(0) = make_RGBcolor(0.f, 0.05f, 0.05f);
        ctable_c.at(1) = make_RGBcolor(0.f, 0.6f, 0.6f);
        ctable_c.at(2) = make_RGBcolor(1.f, 1.f, 1.f);
        ctable_c.at(3) = make_RGBcolor(1.f, 0.f, 0.f);
        ctable_c.at(4) = make_RGBcolor(0.5f, 0.f, 0.f);

        clocs_c.resize(5);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.4f;
        clocs_c.at(2) = 0.5f;
        clocs_c.at(3) = 0.6f;
        clocs_c.at(4) = 1.f;
    } else if (colormap == "rainbow") {
        ctable_c.resize(4);
        ctable_c.at(0) = RGB::navy;
        ctable_c.at(1) = RGB::cyan;
        ctable_c.at(2) = RGB::yellow;
        ctable_c.at(3) = make_RGBcolor(0.75f, 0.f, 0.f);

        clocs_c.resize(4);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.3f;
        clocs_c.at(2) = 0.7f;
        clocs_c.at(3) = 1.f;
    } else if (colormap == "parula") {
        ctable_c.resize(4);
        ctable_c.at(0) = RGB::navy;
        ctable_c.at(1) = make_RGBcolor(0, 0.6, 0.6);
        ctable_c.at(2) = RGB::goldenrod;
        ctable_c.at(3) = RGB::yellow;

        clocs_c.resize(4);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 0.4f;
        clocs_c.at(2) = 0.7f;
        clocs_c.at(3) = 1.f;
    } else if (colormap == "gray") {
        ctable_c.resize(2);
        ctable_c.at(0) = RGB::black;
        ctable_c.at(1) = RGB::white;

        clocs_c.resize(2);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 1.f;
    } else if (colormap == "green") {
        ctable_c.resize(2);
        ctable_c.at(0) = RGB::black;
        ctable_c.at(1) = RGB::green;

        clocs_c.resize(2);
        clocs_c.at(0) = 0.f;
        clocs_c.at(1) = 1.f;
    } else {
        helios_runtime_error("ERROR (Context::generateColormapTextures): Unknown colormap " + colormap + ".");
    }

    return generateColormap(ctable_c, clocs_c, Ncolors);
}

std::vector<std::string> Context::generateTexturesFromColormap(const std::string &texturefile, const std::vector<RGBcolor> &colormap_data) {
    uint Ncolors = colormap_data.size();

    // check that texture file exists
    std::ifstream tfile(texturefile);
    if (!tfile) {
        helios_runtime_error("ERROR (Context::generateTexturesFromColormap): Texture file " + texturefile + " does not exist, or you do not have permission to read it.");
    }
    tfile.close();

    // get file extension
    std::string file_ext = getFileExtension(texturefile);

    // get file base/stem
    std::string file_base = getFileStem(texturefile);

    std::vector<RGBcolor> color_table(Ncolors);

    std::vector<std::string> texture_filenames(Ncolors);

    if (file_ext == "png" || file_ext == "PNG") {
        std::vector<RGBAcolor> pixel_data;
        uint width, height;
        readPNG(texturefile, width, height, pixel_data);

        for (int i = 0; i < Ncolors; i++) {
            std::ostringstream filename;
            filename << "lib/images/colormap_" << file_base << "_" << std::setw(4) << std::setfill('0') << std::to_string(i) << ".png";

            texture_filenames.at(i) = filename.str();

            RGBcolor color = colormap_data.at(i);

            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    pixel_data.at(row * width + col) = make_RGBAcolor(color, pixel_data.at(row * width + col).a);
                }
            }

            writePNG(filename.str(), width, height, pixel_data);
        }
    }

    return texture_filenames;
}

void Context::out_of_memory_handler() {
    helios_runtime_error("ERROR: Out of host memory. The program has run out of memory and cannot continue.");
}

void Context::install_out_of_memory_handler() { std::set_new_handler(out_of_memory_handler); }

Context::~Context() {
    for (auto &[UUID, primitive]: primitives) {
        delete getPrimitivePointer_private(UUID);
    }

    for (auto &[UUID, object]: objects) {
        delete getObjectPointer(UUID);
    }
}

PrimitiveType Context::getPrimitiveType(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::getPrimitiveType): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return getPrimitivePointer_private(UUID)->getType();
}

void Context::setPrimitiveParentObjectID(uint UUID, uint objID) {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::setPrimitiveParentObjectID): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif

    uint current_objID = getPrimitivePointer_private(UUID)->getParentObjectID();
    getPrimitivePointer_private(UUID)->setParentObjectID(objID);

    if (current_objID != 0u && current_objID != objID) {
        if (doesObjectExist(current_objID)) {
            objects.at(current_objID)->deleteChildPrimitive(UUID);

            if (getObjectPointer_private(current_objID)->getPrimitiveUUIDs().empty()) {
                CompoundObject *obj = objects.at(current_objID);
                delete obj;
                objects.erase(current_objID);
            }
        }
    }
}

void Context::setPrimitiveParentObjectID(const std::vector<uint> &UUIDs, uint objID) {
    for (uint UUID: UUIDs) {
        setPrimitiveParentObjectID(UUID, objID);
    }
}

uint Context::getPrimitiveParentObjectID(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::getPrimitiveParentObjectID): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return getPrimitivePointer_private(UUID)->getParentObjectID();
}

std::vector<uint> Context::getPrimitiveParentObjectID(const std::vector<uint> &UUIDs) const {
    std::vector<uint> objIDs(UUIDs.size());
    for (uint i = 0; i < UUIDs.size(); i++) {
#ifdef HELIOS_DEBUG
        if (!doesPrimitiveExist(UUIDs[i])) {
            helios_runtime_error("ERROR (Context::getPrimitiveParentObjectID): Primitive with UUID of " + std::to_string(UUIDs[i]) + " does not exist in the Context.");
        }
#endif
        objIDs[i] = getPrimitivePointer_private(UUIDs[i])->getParentObjectID();
    }
    return objIDs;
}


std::vector<uint> Context::getUniquePrimitiveParentObjectIDs(const std::vector<uint> &UUIDs) const {
    return getUniquePrimitiveParentObjectIDs(UUIDs, false);
}


std::vector<uint> Context::getUniquePrimitiveParentObjectIDs(const std::vector<uint> &UUIDs, bool include_ObjID_zero) const {
    std::vector<uint> primitiveObjIDs;
    if (UUIDs.empty()) {
        return primitiveObjIDs;
    }

    //vector of parent object ID for each primitive
    primitiveObjIDs.resize(UUIDs.size());
    for (uint i = 0; i < UUIDs.size(); i++) {
#ifdef HELIOS_DEBUG
        if (!doesPrimitiveExist(UUIDs.at(i))) {
            helios_runtime_error("ERROR (Context::getUniquePrimitiveParentObjectIDs): Primitive with UUID of " + std::to_string(UUIDs.at(i)) + " does not exist in the Context.");
        }
#endif
        primitiveObjIDs.at(i) = getPrimitivePointer_private(UUIDs.at(i))->getParentObjectID();
    }

    // sort
    std::sort(primitiveObjIDs.begin(), primitiveObjIDs.end());

    // unique
    auto it = unique(primitiveObjIDs.begin(), primitiveObjIDs.end());
    primitiveObjIDs.resize(distance(primitiveObjIDs.begin(), it));

    // remove object ID = 0 from the output if desired and it exists
    if (include_ObjID_zero == false & primitiveObjIDs.front() == uint(0)) {
        primitiveObjIDs.erase(primitiveObjIDs.begin());
    }

    return primitiveObjIDs;
}

float Context::getPrimitiveArea(uint UUID) const {
#ifdef HELIOS_DEBUG
    if (!doesPrimitiveExist(UUID)) {
        helios_runtime_error("ERROR (Context::getPrimitiveArea): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return getPrimitivePointer_private(UUID)->getArea();
}

void Context::getPrimitiveBoundingBox(uint UUID, vec3 &min_corner, vec3 &max_corner) const {
    const std::vector UUIDs = {UUID};
    getPrimitiveBoundingBox(UUIDs, min_corner, max_corner);
}

void Context::getPrimitiveBoundingBox(const std::vector<uint> &UUIDs, vec3 &min_corner, vec3 &max_corner) const {
    uint p = 0;
    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (Context::getPrimitiveBoundingBox): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
        }

        const std::vector<vec3> &vertices = getPrimitiveVertices(UUID);

        if (p == 0) {
            min_corner = vertices.front();
            max_corner = min_corner;
        }

        for (const vec3 &vert: vertices) {
            if (vert.x < min_corner.x) {
                min_corner.x = vert.x;
            }
            if (vert.y < min_corner.y) {
                min_corner.y = vert.y;
            }
            if (vert.z < min_corner.z) {
                min_corner.z = vert.z;
            }
            if (vert.x > max_corner.x) {
                max_corner.x = vert.x;
            }
            if (vert.y > max_corner.y) {
                max_corner.y = vert.y;
            }
            if (vert.z > max_corner.z) {
                max_corner.z = vert.z;
            }
        }

        p++;
    }
}

helios::vec3 Context::getPrimitiveNormal(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getNormal();
}

void Context::getPrimitiveTransformationMatrix(uint UUID, float (&T)[16]) const {
    getPrimitivePointer_private(UUID)->getTransformationMatrix(T);
}

void Context::setPrimitiveTransformationMatrix(uint UUID, float (&T)[16]) {
    getPrimitivePointer_private(UUID)->setTransformationMatrix(T);
}

void Context::setPrimitiveTransformationMatrix(const std::vector<uint> &UUIDs, float (&T)[16]) {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->setTransformationMatrix(T);
    }
}

std::vector<helios::vec3> Context::getPrimitiveVertices(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getVertices();
}


helios::RGBcolor Context::getPrimitiveColor(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColor();
}

helios::RGBcolor Context::getPrimitiveColorRGB(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColorRGB();
}

helios::RGBAcolor Context::getPrimitiveColorRGBA(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColorRGBA();
}

void Context::setPrimitiveColor(uint UUID, const RGBcolor &color) const {
    getPrimitivePointer_private(UUID)->setColor(color);
}

void Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBcolor &color) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->setColor(color);
    }
}

void Context::setPrimitiveColor(uint UUID, const RGBAcolor &color) const {
    getPrimitivePointer_private(UUID)->setColor(color);
}

void Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBAcolor &color) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->setColor(color);
    }
}

std::string Context::getPrimitiveTextureFile(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getTextureFile();
}

void Context::setPrimitiveTextureFile(uint UUID, const std::string &texturefile) const {
    getPrimitivePointer_private(UUID)->setTextureFile(texturefile.c_str());
}

helios::int2 Context::getPrimitiveTextureSize(uint UUID) const {
    std::string texturefile = getPrimitivePointer_private(UUID)->getTextureFile();
    if (!texturefile.empty() && textures.find(texturefile) != textures.end()) {
        return textures.at(texturefile).getImageResolution();
    }
    return {0, 0};
}

std::vector<helios::vec2> Context::getPrimitiveTextureUV(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getTextureUV();
}

bool Context::primitiveTextureHasTransparencyChannel(uint UUID) const {
    std::string texturefile = getPrimitivePointer_private(UUID)->getTextureFile();
    if (!texturefile.empty() && textures.find(texturefile) != textures.end()) {
        return textures.at(texturefile).hasTransparencyChannel();
    }
    return false;
}

const std::vector<std::vector<bool> > *Context::getPrimitiveTextureTransparencyData(uint UUID) const {
    if (primitiveTextureHasTransparencyChannel(UUID)) {
        const std::vector<std::vector<bool> > *data = textures.at(getPrimitivePointer_private(UUID)->getTextureFile()).getTransparencyData();
        return data;
    }

    helios_runtime_error("ERROR (Context::getPrimitiveTransparencyData): Texture transparency data does not exist for primitive " + std::to_string(UUID) + ".");
    return nullptr;
}

void Context::overridePrimitiveTextureColor(uint UUID) const {
    getPrimitivePointer_private(UUID)->overrideTextureColor();
}

void Context::overridePrimitiveTextureColor(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->overrideTextureColor();
    }
}

void Context::usePrimitiveTextureColor(uint UUID) const {
    getPrimitivePointer_private(UUID)->useTextureColor();
}

void Context::usePrimitiveTextureColor(const std::vector<uint> &UUIDs) const {
    for (uint UUID: UUIDs) {
        getPrimitivePointer_private(UUID)->useTextureColor();
    }
}

bool Context::isPrimitiveTextureColorOverridden(uint UUID) const {
    return getPrimitivePointer_private(UUID)->isTextureColorOverridden();
}

float Context::getPrimitiveSolidFraction(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getSolidFraction();
}

void Context::printPrimitiveInfo(uint UUID) const {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Info for UUID " << UUID << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    PrimitiveType type = getPrimitiveType(UUID);
    std::string stype;
    if (type == 0) {
        stype = "PRIMITIVE_TYPE_PATCH";
    } else if (type == 1) {
        stype = "PRIMITIVE_TYPE_TRIANGLE";
    } else if (type == 2) {
        stype = "PRIMITIVE_TYPE_VOXEL";
    }

    std::cout << "Type: " << stype << std::endl;
    std::cout << "Parent ObjID: " << getPrimitiveParentObjectID(UUID) << std::endl;
    std::cout << "Surface Area: " << getPrimitiveArea(UUID) << std::endl;
    std::cout << "Normal Vector: " << getPrimitiveNormal(UUID) << std::endl;

    if (type == PRIMITIVE_TYPE_PATCH) {
        std::cout << "Patch Center: " << getPatchCenter(UUID) << std::endl;
        std::cout << "Patch Size: " << getPatchSize(UUID) << std::endl;
    } else if (type == PRIMITIVE_TYPE_VOXEL) {
        std::cout << "Voxel Center: " << getVoxelCenter(UUID) << std::endl;
        std::cout << "Voxel Size: " << getVoxelSize(UUID) << std::endl;
    }

    std::vector<vec3> primitive_vertices = getPrimitiveVertices(UUID);
    std::cout << "Vertices: " << std::endl;
    for (uint i = 0; i < primitive_vertices.size(); i++) {
        std::cout << "   " << primitive_vertices.at(i) << std::endl;
    }

    float T[16];
    getPrimitiveTransformationMatrix(UUID, T);
    std::cout << "Transform: " << std::endl;
    std::cout << "   " << T[0] << "      " << T[1] << "      " << T[2] << "      " << T[3] << std::endl;
    std::cout << "   " << T[4] << "      " << T[5] << "      " << T[6] << "      " << T[7] << std::endl;
    std::cout << "   " << T[8] << "      " << T[9] << "      " << T[10] << "      " << T[11] << std::endl;
    std::cout << "   " << T[12] << "      " << T[13] << "      " << T[14] << "      " << T[15] << std::endl;

    std::cout << "Color: " << getPrimitiveColor(UUID) << std::endl;
    std::cout << "Texture File: " << getPrimitiveTextureFile(UUID) << std::endl;
    std::cout << "Texture Size: " << getPrimitiveTextureSize(UUID) << std::endl;
    std::cout << "Texture UV: " << std::endl;
    std::vector<vec2> uv = getPrimitiveTextureUV(UUID);
    for (uint i = 0; i < uv.size(); i++) {
        std::cout << "   " << uv.at(i) << std::endl;
    }

    std::cout << "Texture Transparency: " << primitiveTextureHasTransparencyChannel(UUID) << std::endl;
    std::cout << "Color Overridden: " << isPrimitiveTextureColorOverridden(UUID) << std::endl;
    std::cout << "Solid Fraction: " << getPrimitiveSolidFraction(UUID) << std::endl;


    std::cout << "Primitive Data: " << std::endl;
    // Primitive* pointer = getPrimitivePointer_private(UUID);
    std::vector<std::string> pd = listPrimitiveData(UUID);
    for (uint i = 0; i < pd.size(); i++) {
        uint dsize = getPrimitiveDataSize(UUID, pd.at(i).c_str());
        HeliosDataType dtype = getPrimitiveDataType(UUID, pd.at(i).c_str());
        std::string dstype;

        if (dtype == HELIOS_TYPE_INT) {
            dstype = "HELIOS_TYPE_INT";
        } else if (dtype == HELIOS_TYPE_UINT) {
            dstype = "HELIOS_TYPE_UINT";
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            dstype = "HELIOS_TYPE_FLOAT";
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            dstype = "HELIOS_TYPE_DOUBLE";
        } else if (dtype == HELIOS_TYPE_VEC2) {
            dstype = "HELIOS_TYPE_VEC2";
        } else if (dtype == HELIOS_TYPE_VEC3) {
            dstype = "HELIOS_TYPE_VEC3";
        } else if (dtype == HELIOS_TYPE_VEC4) {
            dstype = "HELIOS_TYPE_VEC4";
        } else if (dtype == HELIOS_TYPE_INT2) {
            dstype = "HELIOS_TYPE_INT2";
        } else if (dtype == HELIOS_TYPE_INT3) {
            dstype = "HELIOS_TYPE_INT3";
        } else if (dtype == HELIOS_TYPE_INT4) {
            dstype = "HELIOS_TYPE_INT4";
        } else if (dtype == HELIOS_TYPE_STRING) {
            dstype = "HELIOS_TYPE_STRING";
        } else {
            assert(false);
        }


        std::cout << "   " << "[name: " << pd.at(i) << ", type: " << dstype << ", size: " << dsize << "]:" << std::endl;


        if (dtype == HELIOS_TYPE_INT) {
            std::vector<int> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_UINT) {
            std::vector<uint> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            std::vector<float> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            std::vector<double> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC2) {
            std::vector<vec2> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC3) {
            std::vector<vec3> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC4) {
            std::vector<vec4> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT2) {
            std::vector<int2> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT3) {
            std::vector<int3> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT4) {
            std::vector<int4> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_STRING) {
            std::vector<std::string> pdata;
            getPrimitiveData(UUID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else {
            assert(false);
        }
    }
    std::cout << "-------------------------------------------" << std::endl;
}

void Context::printObjectInfo(uint ObjID) const {
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Info for ObjID " << ObjID << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    ObjectType otype = getObjectType(ObjID);
    std::string ostype;
    if (otype == 0) {
        ostype = "OBJECT_TYPE_TILE";
    } else if (otype == 1) {
        ostype = "OBJECT_TYPE_SPHERE";
    } else if (otype == 2) {
        ostype = "OBJECT_TYPE_TUBE";
    } else if (otype == 3) {
        ostype = "OBJECT_TYPE_BOX";
    } else if (otype == 4) {
        ostype = "OBJECT_TYPE_DISK";
    } else if (otype == 5) {
        ostype = "OBJECT_TYPE_POLYMESH";
    } else if (otype == 6) {
        ostype = "OBJECT_TYPE_CONE";
    }

    std::cout << "Type: " << ostype << std::endl;
    std::cout << "Object Bounding Box Center: " << getObjectCenter(ObjID) << std::endl;
    std::cout << "One-sided Surface Area: " << getObjectArea(ObjID) << std::endl;

    std::cout << "Primitive Count: " << getObjectPrimitiveCount(ObjID) << std::endl;

    if (areObjectPrimitivesComplete(ObjID)) {
        std::cout << "Object Primitives Complete" << std::endl;
    } else {
        std::cout << "Object Primitives Incomplete" << std::endl;
    }

    std::cout << "Primitive UUIDs: " << std::endl;
    std::vector<uint> primitive_UUIDs = getObjectPrimitiveUUIDs(ObjID);
    for (uint i = 0; i < primitive_UUIDs.size(); i++) {
        if (i < 5) {
            PrimitiveType ptype = getPrimitiveType(primitive_UUIDs.at(i));
            std::string pstype;
            if (ptype == 0) {
                pstype = "PRIMITIVE_TYPE_PATCH";
            } else if (ptype == 1) {
                pstype = "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(i) << " (" << pstype << ")" << std::endl;
        } else {
            std::cout << "   ..." << std::endl;
            PrimitiveType ptype = getPrimitiveType(primitive_UUIDs.at(primitive_UUIDs.size() - 2));
            std::string pstype;
            if (ptype == 0) {
                pstype = "PRIMITIVE_TYPE_PATCH";
            } else if (ptype == 1) {
                pstype = "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(primitive_UUIDs.size() - 2) << " (" << pstype << ")" << std::endl;
            ptype = getPrimitiveType(primitive_UUIDs.at(primitive_UUIDs.size() - 1));
            if (ptype == 0) {
                pstype = "PRIMITIVE_TYPE_PATCH";
            } else if (ptype == 1) {
                pstype = "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(primitive_UUIDs.size() - 1) << " (" << pstype << ")" << std::endl;
            break;
        }
    }

    if (otype == OBJECT_TYPE_TILE) {
        std::cout << "Tile Center: " << getTileObjectCenter(ObjID) << std::endl;
        std::cout << "Tile Size: " << getTileObjectSize(ObjID) << std::endl;
        std::cout << "Tile Subdivision Count: " << getTileObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Tile Normal: " << getTileObjectNormal(ObjID) << std::endl;

        std::cout << "Tile Texture UV: " << std::endl;
        std::vector<vec2> uv = getTileObjectTextureUV(ObjID);
        for (uint i = 0; i < uv.size(); i++) {
            std::cout << "   " << uv.at(i) << std::endl;
        }

        std::cout << "Tile Vertices: " << std::endl;
        std::vector<vec3> primitive_vertices = getTileObjectVertices(ObjID);
        for (uint i = 0; i < primitive_vertices.size(); i++) {
            std::cout << "   " << primitive_vertices.at(i) << std::endl;
        }
    } else if (otype == OBJECT_TYPE_SPHERE) {
        std::cout << "Sphere Center: " << getSphereObjectCenter(ObjID) << std::endl;
        std::cout << "Sphere Radius: " << getSphereObjectRadius(ObjID) << std::endl;
        std::cout << "Sphere Subdivision Count: " << getSphereObjectSubdivisionCount(ObjID) << std::endl;
    } else if (otype == OBJECT_TYPE_TUBE) {
        std::cout << "Tube Subdivision Count: " << getTubeObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Tube Nodes: " << std::endl;
        std::vector<vec3> nodes = getTubeObjectNodes(ObjID);
        for (uint i = 0; i < nodes.size(); i++) {
            if (i < 10) {
                std::cout << "   " << nodes.at(i) << std::endl;
            } else {
                std::cout << "      ..." << std::endl;
                std::cout << "   " << nodes.at(nodes.size() - 2) << std::endl;
                std::cout << "   " << nodes.at(nodes.size() - 1) << std::endl;
                break;
            }
        }
        std::cout << "Tube Node Radii: " << std::endl;
        std::vector<float> noderadii = getTubeObjectNodeRadii(ObjID);
        for (uint i = 0; i < noderadii.size(); i++) {
            if (i < 10) {
                std::cout << "   " << noderadii.at(i) << std::endl;
            } else {
                std::cout << "      ..." << std::endl;
                std::cout << "   " << noderadii.at(noderadii.size() - 2) << std::endl;
                std::cout << "   " << noderadii.at(noderadii.size() - 1) << std::endl;
                break;
            }
        }
        std::cout << "Tube Node Colors: " << std::endl;
        std::vector<helios::RGBcolor> nodecolors = getTubeObjectNodeColors(ObjID);
        for (uint i = 0; i < nodecolors.size(); i++) {
            if (i < 10) {
                std::cout << "   " << nodecolors.at(i) << std::endl;
            } else {
                std::cout << "      ..." << std::endl;
                std::cout << "      " << nodecolors.at(nodecolors.size() - 2) << std::endl;
                std::cout << "      " << nodecolors.at(nodecolors.size() - 1) << std::endl;
                break;
            }
        }
    } else if (otype == OBJECT_TYPE_BOX) {
        std::cout << "Box Center: " << getBoxObjectCenter(ObjID) << std::endl;
        std::cout << "Box Size: " << getBoxObjectSize(ObjID) << std::endl;
        std::cout << "Box Subdivision Count: " << getBoxObjectSubdivisionCount(ObjID) << std::endl;
    } else if (otype == OBJECT_TYPE_DISK) {
        std::cout << "Disk Center: " << getDiskObjectCenter(ObjID) << std::endl;
        std::cout << "Disk Size: " << getDiskObjectSize(ObjID) << std::endl;
        std::cout << "Disk Subdivision Count: " << getDiskObjectSubdivisionCount(ObjID) << std::endl;

        // }else if(type == OBJECT_TYPE_POLYMESH){
        // nothing for now
    } else if (otype == OBJECT_TYPE_CONE) {
        std::cout << "Cone Length: " << getConeObjectLength(ObjID) << std::endl;
        std::cout << "Cone Axis Unit Vector: " << getConeObjectAxisUnitVector(ObjID) << std::endl;
        std::cout << "Cone Subdivision Count: " << getConeObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Cone Nodes: " << std::endl;
        std::vector<vec3> nodes = getConeObjectNodes(ObjID);
        for (uint i = 0; i < nodes.size(); i++) {
            std::cout << "   " << nodes.at(i) << std::endl;
        }
        std::cout << "Cone Node Radii: " << std::endl;
        std::vector<float> noderadii = getConeObjectNodeRadii(ObjID);
        for (uint i = 0; i < noderadii.size(); i++) {
            std::cout << "   " << noderadii.at(i) << std::endl;
        }
    }


    float T[16];
    getObjectTransformationMatrix(ObjID, T);
    std::cout << "Transform: " << std::endl;
    std::cout << "   " << T[0] << "      " << T[1] << "      " << T[2] << "      " << T[3] << std::endl;
    std::cout << "   " << T[4] << "      " << T[5] << "      " << T[6] << "      " << T[7] << std::endl;
    std::cout << "   " << T[8] << "      " << T[9] << "      " << T[10] << "      " << T[11] << std::endl;
    std::cout << "   " << T[12] << "      " << T[13] << "      " << T[14] << "      " << T[15] << std::endl;

    std::cout << "Texture File: " << getObjectTextureFile(ObjID) << std::endl;

    std::cout << "Object Data: " << std::endl;
    // Primitive* pointer = getPrimitivePointer_private(ObjID);
    std::vector<std::string> pd = listObjectData(ObjID);
    for (uint i = 0; i < pd.size(); i++) {
        uint dsize = getObjectDataSize(ObjID, pd.at(i).c_str());
        HeliosDataType dtype = getObjectDataType(ObjID, pd.at(i).c_str());
        std::string dstype;

        if (dtype == HELIOS_TYPE_INT) {
            dstype = "HELIOS_TYPE_INT";
        } else if (dtype == HELIOS_TYPE_UINT) {
            dstype = "HELIOS_TYPE_UINT";
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            dstype = "HELIOS_TYPE_FLOAT";
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            dstype = "HELIOS_TYPE_DOUBLE";
        } else if (dtype == HELIOS_TYPE_VEC2) {
            dstype = "HELIOS_TYPE_VEC2";
        } else if (dtype == HELIOS_TYPE_VEC3) {
            dstype = "HELIOS_TYPE_VEC3";
        } else if (dtype == HELIOS_TYPE_VEC4) {
            dstype = "HELIOS_TYPE_VEC4";
        } else if (dtype == HELIOS_TYPE_INT2) {
            dstype = "HELIOS_TYPE_INT2";
        } else if (dtype == HELIOS_TYPE_INT3) {
            dstype = "HELIOS_TYPE_INT3";
        } else if (dtype == HELIOS_TYPE_INT4) {
            dstype = "HELIOS_TYPE_INT4";
        } else if (dtype == HELIOS_TYPE_STRING) {
            dstype = "HELIOS_TYPE_STRING";
        } else {
            assert(false);
        }


        std::cout << "   " << "[name: " << pd.at(i) << ", type: " << dstype << ", size: " << dsize << "]:" << std::endl;


        if (dtype == HELIOS_TYPE_INT) {
            std::vector<int> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_UINT) {
            std::vector<uint> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            std::vector<float> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            std::vector<double> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC2) {
            std::vector<vec2> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC3) {
            std::vector<vec3> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_VEC4) {
            std::vector<vec4> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT2) {
            std::vector<int2> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT3) {
            std::vector<int3> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize - 2) << std::endl;
                    std::cout << "      " << pdata.at(dsize - 1) << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_INT4) {
            std::vector<int4> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    break;
                }
            }
        } else if (dtype == HELIOS_TYPE_STRING) {
            std::vector<std::string> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata);
            for (uint j = 0; j < dsize; j++) {
                if (j < 10) {
                    std::cout << "      " << pdata.at(j) << std::endl;
                } else {
                    std::cout << "      ..." << std::endl;
                    break;
                }
            }
        } else {
            assert(false);
        }
    }
    std::cout << "-------------------------------------------" << std::endl;
}

CompoundObject *Context::getObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    return objects.at(ObjID);
}

void Context::hideObject(uint ObjID) {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::hideObject): Object ID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    objects.at(ObjID)->ishidden = true;
    for (uint UUID: objects.at(ObjID)->getPrimitiveUUIDs()) {
#ifdef HELIOS_DEBUG
        if (!doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (Context::hideObject): Primitive UUID of " + std::to_string(UUID) + " does not exist in the Context.");
        }
#endif
        primitives.at(UUID)->ishidden = true;
    }
}

void Context::hideObject(const std::vector<uint> &ObjIDs) {
    for (uint ObjID: ObjIDs) {
        hideObject(ObjID);
    }
}

bool Context::isObjectHidden(uint ObjID) const {
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::isObjectHidden): Object ID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
    return objects.at(ObjID)->ishidden;
}

float Context::getObjectArea(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getArea();
}

helios::vec3 Context::getObjectAverageNormal(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectAverageNormal): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif

    const std::vector<uint> &UUIDs = objects.at(ObjID)->getPrimitiveUUIDs();

    vec3 norm_avg;
    for (uint UUID: UUIDs) {
        norm_avg += getPrimitiveNormal(UUID);
    }
    norm_avg.normalize();

    return norm_avg;
}

uint Context::getObjectPrimitiveCount(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getPrimitiveCount();
}

helios::vec3 Context::getObjectCenter(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getObjectCenter();
}

std::string Context::getObjectTextureFile(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getTextureFile();
}

void Context::getObjectTransformationMatrix(uint ObjID, float (&T)[16]) const {
    getObjectPointer_private(ObjID)->getTransformationMatrix(T);
}

void Context::setObjectTransformationMatrix(uint ObjID, float (&T)[16]) const {
    getObjectPointer_private(ObjID)->setTransformationMatrix(T);
}

void Context::setObjectTransformationMatrix(const std::vector<uint> &ObjIDs, float (&T)[16]) const {
    for (uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->setTransformationMatrix(T);
    }
}

void Context::setObjectAverageNormal(uint ObjID, const vec3 &origin, const vec3 &new_normal) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("setObjectAverageNormal: invalid objectID");
    }
#endif

    // 1) Compute unit old & new normals
    vec3 oldN = normalize(getObjectAverageNormal(ObjID));
    vec3 newN = normalize(new_normal);

    // 2) Minimal‐angle axis & angle
    float d = std::clamp(oldN * newN, -1.f, 1.f);
    float angle = acosf(d);
    vec3 axis = cross(oldN, newN);
    if (axis.magnitude() < 1e-6f) {
        // pick any vector ⟂ oldN
        axis = (std::abs(oldN.x) < std::abs(oldN.z))
                   ? cross(oldN, {1, 0, 0})
                   : cross(oldN, {0, 0, 1});
    }
    axis = axis.normalize();

    // 3) Apply that minimal‐angle rotation to the compound (no pizza‐spin yet)
    //    NOTE: correct argument order is (objectID, angle, origin, axis)
    rotateObject(ObjID, angle, origin, axis);

    // 4) Fetch the updated transform and extract the world‐space “forward” (local +X)
    float M_mid[16];
    getObjectPointer_private(ObjID)->getTransformationMatrix(M_mid);

    vec3 localX{1, 0, 0};
    vec3 t1;
    // vecmult multiplies the 4×4 M_mid by v3 (w=0), writing into t1
    vecmult(M_mid, localX, t1);
    t1 = normalize(t1);

    // 5) Compute desired forward = world‐X projected into the new plane
    vec3 worldX{1, 0, 0};
    vec3 targ = worldX - newN * (newN * worldX);
    targ = normalize(targ);

    // 6) Compute signed twist about newN that carries t1→targ
    float twist = atan2f(
        newN * cross(t1, targ), // dot(newN, t1×targ)
        t1 * targ // dot(t1, targ)
    );

    // 7) Apply that compensating twist about the same origin
    rotateObject(ObjID, twist, origin, newN);
}

void Context::setObjectOrigin( uint ObjID, const vec3& origin ) const {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(ObjID)) {
        helios_runtime_error("ERROR (Context::setObjectOrigin): invalid objectID");
    }
#endif
    objects.at(ObjID)->object_origin = origin;
}

bool Context::objectHasTexture(uint ObjID) const {
    return getObjectPointer_private(ObjID)->hasTexture();
}

void Context::setObjectColor(uint ObjID, const RGBcolor &color) const {
    getObjectPointer_private(ObjID)->setColor(color);
}

void Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBcolor &color) const {
    for (const uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->setColor(color);
    }
}

void Context::setObjectColor(uint ObjID, const RGBAcolor &color) const {
    getObjectPointer_private(ObjID)->setColor(color);
}

void Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBAcolor &color) const {
    for (const uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->setColor(color);
    }
}

bool Context::doesObjectContainPrimitive(uint ObjID, uint UUID) const {
    return getObjectPointer_private(ObjID)->doesObjectContainPrimitive(UUID);
}

void Context::overrideObjectTextureColor(uint ObjID) const {
    getObjectPointer_private(ObjID)->overrideTextureColor();
}

void Context::overrideObjectTextureColor(const std::vector<uint> &ObjIDs) const {
    for (uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->overrideTextureColor();
    }
}

void Context::useObjectTextureColor(uint ObjID) const {
    getObjectPointer_private(ObjID)->useTextureColor();
}

void Context::useObjectTextureColor(const std::vector<uint> &ObjIDs) {
    for (uint ObjID: ObjIDs) {
        getObjectPointer_private(ObjID)->useTextureColor();
    }
}

void Context::getObjectBoundingBox(uint ObjID, vec3 &min_corner, vec3 &max_corner) const {
    const std::vector ObjIDs{ObjID};
    getObjectBoundingBox(ObjIDs, min_corner, max_corner);
}

void Context::getObjectBoundingBox(const std::vector<uint> &ObjIDs, vec3 &min_corner, vec3 &max_corner) const {
    uint o = 0;
    for (uint ObjID: ObjIDs) {
        if (objects.find(ObjID) == objects.end()) {
            helios_runtime_error("ERROR (Context::getObjectBoundingBox): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
        }

        const std::vector<uint> &UUIDs = objects.at(ObjID)->getPrimitiveUUIDs();

        uint p = 0;
        for (const uint UUID: UUIDs) {
            const std::vector<vec3> &vertices = getPrimitiveVertices(UUID);

            if (p == 0 && o == 0) {
                min_corner = vertices.front();
                max_corner = min_corner;
                p++;
                continue;
            }

            for (const vec3 &vert: vertices) {
                if (vert.x < min_corner.x) {
                    min_corner.x = vert.x;
                }
                if (vert.y < min_corner.y) {
                    min_corner.y = vert.y;
                }
                if (vert.z < min_corner.z) {
                    min_corner.z = vert.z;
                }
                if (vert.x > max_corner.x) {
                    max_corner.x = vert.x;
                }
                if (vert.y > max_corner.y) {
                    max_corner.y = vert.y;
                }
                if (vert.z > max_corner.z) {
                    max_corner.z = vert.z;
                }
            }
        }

        o++;
    }
}

Tile *Context::getTileObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_TILE) {
        helios_runtime_error("ERROR (Context::getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Tile Object.");
    }
#endif
    return dynamic_cast<Tile *>(objects.at(ObjID));
}

Sphere *Context::getSphereObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_SPHERE) {
        helios_runtime_error("ERROR (Context::getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Sphere Object.");
    }
#endif
    return dynamic_cast<Sphere *>(objects.at(ObjID));
}

Tube *Context::getTubeObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_TUBE) {
        helios_runtime_error("ERROR (Context::getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Tube Object.");
    }
#endif
    return dynamic_cast<Tube *>(objects.at(ObjID));
}

Box *Context::getBoxObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_BOX) {
        helios_runtime_error("ERROR (Context::getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Box Object.");
    }
#endif
    return dynamic_cast<Box *>(objects.at(ObjID));
}

Disk *Context::getDiskObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_DISK) {
        helios_runtime_error("ERROR (Context::getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Disk Object.");
    }
#endif
    return dynamic_cast<Disk *>(objects.at(ObjID));
}

Polymesh *Context::getPolymeshObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_POLYMESH) {
        helios_runtime_error("ERROR (Context::getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Polymesh Object.");
    }
#endif
    return dynamic_cast<Polymesh *>(objects.at(ObjID));
}

Cone *Context::getConeObjectPointer_private(uint ObjID) const {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    } else if (objects.at(ObjID)->getObjectType() != OBJECT_TYPE_CONE) {
        helios_runtime_error("ERROR (Context::getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Cone Object.");
    }
#endif
    return dynamic_cast<Cone *>(objects.at(ObjID));
}

helios::vec3 Context::getTileObjectCenter(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getCenter();
}

helios::vec2 Context::getTileObjectSize(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getSize();
}

helios::int2 Context::getTileObjectSubdivisionCount(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getSubdivisionCount();
}

helios::vec3 Context::getTileObjectNormal(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getNormal();
}

std::vector<helios::vec2> Context::getTileObjectTextureUV(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getTextureUV();
}

std::vector<helios::vec3> Context::getTileObjectVertices(uint ObjID) const {
    return getTileObjectPointer_private(ObjID)->getVertices();
}

helios::vec3 Context::getSphereObjectCenter(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getCenter();
}

helios::vec3 Context::getSphereObjectRadius(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getRadius();
}

uint Context::getSphereObjectSubdivisionCount(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getSubdivisionCount();
}

float Context::getSphereObjectVolume(uint ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getVolume();
}

uint Context::getTubeObjectSubdivisionCount(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getSubdivisionCount();
}

std::vector<helios::vec3> Context::getTubeObjectNodes(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodes();
}

uint Context::getTubeObjectNodeCount(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeCount();
}

std::vector<float> Context::getTubeObjectNodeRadii(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeRadii();
}

std::vector<RGBcolor> Context::getTubeObjectNodeColors(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeColors();
}

float Context::getTubeObjectVolume(uint ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getVolume();
}

float Context::getTubeObjectSegmentVolume(uint ObjID, uint segment_index) const {
    return getTubeObjectPointer_private(ObjID)->getSegmentVolume(segment_index);
}

void Context::appendTubeSegment(uint ObjID, const helios::vec3 &node_position, float node_radius, const RGBcolor &node_color) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::appendTubeSegment): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->appendTubeSegment(node_position, node_radius, node_color);
}

void Context::appendTubeSegment(uint ObjID, const helios::vec3 &node_position, float node_radius, const char *texturefile, const helios::vec2 &textureuv_ufrac) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::appendTubeSegment): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->appendTubeSegment(node_position, node_radius, texturefile, textureuv_ufrac);
}

void Context::scaleTubeGirth(uint ObjID, float scale_factor) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::scaleTubeGirth): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->scaleTubeGirth(scale_factor);
}

void Context::setTubeRadii(uint ObjID, const std::vector<float> &node_radii) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::setTubeRadii): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->setTubeRadii(node_radii);
}

void Context::scaleTubeLength(uint ObjID, float scale_factor) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::scaleTubeLength): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->scaleTubeLength(scale_factor);
}

void Context::pruneTubeNodes(uint ObjID, uint node_index) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::pruneTubeNodes): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->pruneTubeNodes(node_index);
}

void Context::setTubeNodes(uint ObjID, const std::vector<helios::vec3> &node_xyz) {
#ifdef HELIOS_DEBUG
    if (objects.find(ObjID) == objects.end()) {
        helios_runtime_error("ERROR (Context::setTubeNodes): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
#endif
    dynamic_cast<Tube *>(objects.at(ObjID))->setTubeNodes(node_xyz);
}

helios::vec3 Context::getBoxObjectCenter(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getCenter();
}

helios::vec3 Context::getBoxObjectSize(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getSize();
}

helios::int3 Context::getBoxObjectSubdivisionCount(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getSubdivisionCount();
}

float Context::getBoxObjectVolume(uint ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getVolume();
}

helios::vec3 Context::getDiskObjectCenter(uint ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getCenter();
}

helios::vec2 Context::getDiskObjectSize(uint ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getSize();
}

uint Context::getDiskObjectSubdivisionCount(uint ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getSubdivisionCount().x;
}

uint Context::getConeObjectSubdivisionCount(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getSubdivisionCount();
}

std::vector<helios::vec3> Context::getConeObjectNodes(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getNodeCoordinates();
}

std::vector<float> Context::getConeObjectNodeRadii(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getNodeRadii();
}

helios::vec3 Context::getConeObjectNode(uint ObjID, int number) const {
    return getConeObjectPointer_private(ObjID)->getNodeCoordinate(number);
}

float Context::getConeObjectNodeRadius(uint ObjID, int number) const {
    return getConeObjectPointer_private(ObjID)->getNodeRadius(number);
}

helios::vec3 Context::getConeObjectAxisUnitVector(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getAxisUnitVector();
}

float Context::getConeObjectLength(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getLength();
}

float Context::getConeObjectVolume(uint ObjID) const {
    return getConeObjectPointer_private(ObjID)->getVolume();
}

float Context::getPolymeshObjectVolume(uint ObjID) const {
    return getPolymeshObjectPointer_private(ObjID)->getVolume();
}
