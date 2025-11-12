/** \file "VisualizerGeometry.cpp" Visualizer geometry creation functions.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

// Freetype Libraries (rendering fonts)
extern "C" {
#include <ft2build.h>
#include FT_FREETYPE_H
}

#include "Visualizer.h"

using namespace helios;

void Visualizer::clearGeometry() {
    geometry_handler.clearAllGeometry();

    contextUUIDs_build.clear();
    colorPrimitives_UUIDs.clear();
    colorPrimitives_objIDs.clear();
    contextUUIDs_build.clear();
    depth_buffer_data.clear();
    colorbar_min = 0;
    colorbar_max = 0;
}

size_t Visualizer::addRectangleByCenter(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color, CoordinateSystem coordFlag) {
    return addRectangleByCenter(center, size, rotation, make_RGBAcolor(color.r, color.g, color.b, 1), coordFlag);
}

size_t Visualizer::addRectangleByCenter(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color, CoordinateSystem coordFlag) {
    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3(-0.5f * size.x, -0.5f * size.y, 0.f);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3(+0.5f * size.x, -0.5f * size.y, 0.f);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3(+0.5f * size.x, +0.5f * size.y, 0.f);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3(-0.5f * size.x, +0.5f * size.y, 0.f);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(3) = center + v3;

    return addRectangleByVertices(vertices, color, coordFlag);
}

size_t Visualizer::addRectangleByCenter(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file, CoordinateSystem coordFlag) {
    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3(-0.5f * size.x, -0.5f * size.y, 0.f);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3(+0.5f * size.x, -0.5f * size.y, 0.f);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3(+0.5f * size.x, +0.5f * size.y, 0.f);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3(-0.5f * size.x, +0.5f * size.y, 0.f);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(3) = center + v3;

    return addRectangleByVertices(vertices, texture_file, coordFlag);
}

size_t Visualizer::addRectangleByCenter(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color, const char *texture_file, CoordinateSystem coordFlag) {
    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3(-0.5f * size.x, -0.5f * size.y, 0.f);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3(+0.5f * size.x, -0.5f * size.y, 0.f);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3(+0.5f * size.x, +0.5f * size.y, 0.f);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3(-0.5f * size.x, +0.5f * size.y, 0.f);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(3) = center + v3;

    return addRectangleByVertices(vertices, color, texture_file, coordFlag);
}

size_t Visualizer::addRectangleByCenter(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color, const Glyph *glyph, CoordinateSystem coordFlag) {
    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3(-0.5f * size.x, -0.5f * size.y, 0.f);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3(+0.5f * size.x, -0.5f * size.y, 0.f);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3(+0.5f * size.x, +0.5f * size.y, 0.f);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3(-0.5f * size.x, +0.5f * size.y, 0.f);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(1, 0, 0), -rotation.elevation);
    v3 = rotatePointAboutLine(v3, make_vec3(0, 0, 0), make_vec3(0, 0, 1), -rotation.azimuth);
    vertices.at(3) = center + v3;

    return addRectangleByVertices(vertices, color, glyph, coordFlag);
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const RGBcolor &color, CoordinateSystem coordFlag) {
    return addRectangleByVertices(vertices, make_RGBAcolor(color.r, color.g, color.b, 1), coordFlag);
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const RGBAcolor &color, CoordinateSystem coordFlag) {
    if (coordFlag == COORDINATES_WINDOW_NORMALIZED) { // No vertex transformation (i.e., identity matrix)

        // Check that coordinates are inside drawable area
        for (auto vertex: vertices) {
            if (vertex.x < 0.f || vertex.x > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }
    }

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, vertices, color, {}, -1, false, false, coordFlag, true, false);
    return UUID;
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const char *texture_file, CoordinateSystem coordFlag) {
    const std::vector<vec2> uvs{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    return addRectangleByVertices(vertices, texture_file, uvs, coordFlag);
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const char *texture_file, const std::vector<vec2> &uvs, CoordinateSystem coordFlag) {
    if (coordFlag == COORDINATES_WINDOW_NORMALIZED) { // No vertex transformation (i.e., identity matrix)

        // Check that coordinates are inside drawable area
        for (auto vertex: vertices) {
            if (vertex.x < 0.f || vertex.x > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }
    }

    uint textureID = registerTextureImage(texture_file);

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, vertices, RGBA::black, uvs, textureID, false, false, coordFlag, true, false);
    return UUID;
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const RGBcolor &color, const char *texture_file, CoordinateSystem coordFlag) {
    const std::vector<vec2> uvs{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    return addRectangleByVertices(vertices, color, texture_file, uvs, coordFlag);
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const helios::RGBcolor &color, const char *texture_file, const std::vector<vec2> &uvs, CoordinateSystem coordFlag) {
    if (coordFlag == COORDINATES_WINDOW_NORMALIZED) { // No vertex transformation (i.e., identity matrix)

        // Check that coordinates are inside drawable area
        for (auto vertex: vertices) {
            if (vertex.x < 0.f || vertex.x > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }
    }

    uint textureID = registerTextureImage(texture_file);

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, vertices, make_RGBAcolor(color, 1.f), uvs, textureID, false, false, coordFlag, true, false);
    return UUID;
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const RGBcolor &color, const Glyph *glyph, CoordinateSystem coordFlag) {
    return addRectangleByVertices(vertices, make_RGBAcolor(color, 1), glyph, coordFlag);
}

size_t Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const RGBAcolor &color, const Glyph *glyph, CoordinateSystem coordFlag) {
    if (coordFlag == COORDINATES_WINDOW_NORMALIZED) { // No vertex transformation (i.e., identity matrix)

        // Check that coordinates are inside drawable area
        for (auto vertex: vertices) {
            if (vertex.x < 0.f || vertex.x > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }
    }

    uint textureID = registerTextureGlyph(glyph);

    const std::vector<vec2> uvs{{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    // Disable shadows for glyphs
    CoordinateSystem coordFlag2 = coordFlag;
    if (coordFlag == COORDINATES_CARTESIAN) {
        coordFlag2 = scast<CoordinateSystem>(2);
    }

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, vertices, color, uvs, textureID, true, true, coordFlag2, true, false);
    return UUID;
}

size_t Visualizer::addTriangle(const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const RGBcolor &color, CoordinateSystem coordFlag) {
    return addTriangle(vertex0, vertex1, vertex2, make_RGBAcolor(color.r, color.g, color.b, 1), coordFlag);
}

size_t Visualizer::addTriangle(const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const RGBAcolor &color, CoordinateSystem coordFlag) {
    const std::vector<vec3> vertices{vertex0, vertex1, vertex2};

    if (coordFlag == 0) { // No vertex transformation (i.e., identity matrix)

        // Check that coordinates are inside drawable area
        for (const auto &vertex: vertices) {
            if (vertex.x < 0.f || vertex.x > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }
    }

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, vertices, color, {}, -1, false, false, coordFlag, true, false);
    return UUID;
}

size_t Visualizer::addTriangle(const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const char *texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2, CoordinateSystem coordFlag) {
    const std::vector<vec3> vertices{vertex0, vertex1, vertex2};
    const std::vector<vec2> uvs{uv0, uv1, uv2};

    if (coordFlag == 0) { // No vertex transformation (i.e., identity matrix)

        // Check that coordinates are inside drawable area
        for (auto &vertex: vertices) {
            if (vertex.x < 0.f || vertex.x > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }
    }

    uint textureID = registerTextureImage(texture_file);

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, vertices, RGBA::black, uvs, textureID, false, false, coordFlag, true, false);
    return UUID;
}

size_t Visualizer::addTriangle(const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const char *texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2, const RGBAcolor &color, CoordinateSystem coordFlag) {
    const std::vector<vec3> vertices{vertex0, vertex1, vertex2};
    const std::vector<vec2> uvs{uv0, uv1, uv2};

    if (coordFlag == 0) { // No vertex transformation (i.e., identity matrix)

        // Check that coordinates are inside drawable area
        for (const auto &tri_vertex: vertices) {
            if (tri_vertex.x < 0.f || tri_vertex.x > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << tri_vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (tri_vertex.y < 0.f || tri_vertex.y > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << tri_vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (tri_vertex.z < -1.f || tri_vertex.z > 1.f) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << tri_vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }
    }

    uint textureID = registerTextureImage(texture_file);

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, vertices, color, uvs, textureID, true, false, coordFlag, true, false);
    return UUID;
}

std::vector<size_t> Visualizer::addVoxelByCenter(const vec3 &center, const vec3 &size, const SphericalCoord &rotation, const RGBcolor &color, CoordinateSystem coordFlag) {
    return addVoxelByCenter(center, size, rotation, make_RGBAcolor(color.r, color.g, color.b, 1), coordFlag);
}

std::vector<size_t> Visualizer::addVoxelByCenter(const vec3 &center, const vec3 &size, const SphericalCoord &rotation, const RGBAcolor &color, CoordinateSystem coordFlag) {
    float eps = 1e-4; // Avoid z-fighting

    float az = rotation.azimuth;

    std::vector<size_t> UUIDs(6);

    const vec3 c0 = center + rotatePoint(make_vec3(0, -0.5f * size.y, 0.f), 0, az) + eps;
    UUIDs.at(0) = addRectangleByCenter(c0, make_vec2(size.x, size.z), make_SphericalCoord(-0.5 * PI_F, az), color, coordFlag);

    const vec3 c1 = center + rotatePoint(make_vec3(0, 0.5f * size.y, 0.f), 0, az) + eps;
    UUIDs.at(1) = addRectangleByCenter(c1, make_vec2(size.x, size.z), make_SphericalCoord(0.5 * PI_F, az), color, coordFlag);

    const vec3 c2 = center + rotatePoint(make_vec3(0.5f * size.x, 0.f, 0.f), 0, az) + eps;
    UUIDs.at(2) = addRectangleByCenter(c2, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F + az), color, coordFlag);

    const vec3 c3 = center + rotatePoint(make_vec3(-0.5f * size.x, 0.f, 0.f), 0, az) + eps;
    UUIDs.at(3) = addRectangleByCenter(c3, make_vec2(size.y, size.z), make_SphericalCoord(0.5 * PI_F, 0.5 * PI_F + az), color, coordFlag);

    const vec3 c4 = center + make_vec3(0.f, 0.f, -0.5f * size.z) + eps;
    UUIDs.at(4) = addRectangleByCenter(c4, make_vec2(size.x, size.y), make_SphericalCoord(PI_F, az), color, coordFlag);

    const vec3 c5 = center + make_vec3(0.f, 0.f, 0.5f * size.z) + eps;
    UUIDs.at(5) = addRectangleByCenter(c5, make_vec2(size.x, size.y), make_SphericalCoord(0, az), color, coordFlag);

    return UUIDs;
}

size_t Visualizer::addLine(const vec3 &start, const vec3 &end, const RGBcolor &color, CoordinateSystem coordinate_system) {
    return addLine(start, end, make_RGBAcolor(color, 1), coordinate_system);
}

size_t Visualizer::addLine(const vec3 &start, const vec3 &end, const RGBAcolor &color, CoordinateSystem coordFlag) {
    const std::vector<vec3> vertices{start, end};

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_LINE, vertices, color, {}, -1, false, false, coordFlag, true, false);
    return UUID;
}

size_t Visualizer::addLine(const vec3 &start, const vec3 &end, const RGBcolor &color, float line_width, CoordinateSystem coordinate_system) {
    return addLine(start, end, make_RGBAcolor(color, 1), line_width, coordinate_system);
}

size_t Visualizer::addLine(const vec3 &start, const vec3 &end, const RGBAcolor &color, float line_width, CoordinateSystem coordFlag) {
    // Validate line width
    if (line_width <= 0.0f) {
        helios_runtime_error("ERROR (Visualizer::addLine): Line width must be positive (got " + std::to_string(line_width) + "). Please specify a positive line width value.");
    }

    // Reasonable maximum line width to prevent rendering issues
    // Lines are rendered using geometry shaders which expand line primitives into quads
    const float MAX_LINE_WIDTH = 100.0f;
    if (line_width > MAX_LINE_WIDTH) {
        helios_runtime_error("ERROR (Visualizer::addLine): Line width " + std::to_string(line_width) + " exceeds maximum supported width (" + std::to_string(MAX_LINE_WIDTH) + "). Please specify a smaller line width value.");
    }

    const std::vector<vec3> vertices{start, end};

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_LINE, vertices, color, {}, -1, false, false, coordFlag, true, false, false, line_width);
    return UUID;
}

size_t Visualizer::addPoint(const vec3 &position, const RGBcolor &color, float pointsize, CoordinateSystem coordinate_system) {
    return addPoint(position, make_RGBAcolor(color, 1), pointsize, coordinate_system);
}

size_t Visualizer::addPoint(const vec3 &position, const RGBAcolor &color, float pointsize, CoordinateSystem coordinate_system) {
    // Only perform OpenGL validation if we have a valid context (not in headless mode during initialization)
    if (!headless && window != nullptr) {
        // Use conservative OpenGL 3.3 Core Profile point size limits
        // Most implementations support at least 1.0 to 64.0 for point sizes
        const float MIN_POINT_SIZE = 1.0f;
        const float MAX_POINT_SIZE = 64.0f;

        if (pointsize < MIN_POINT_SIZE || pointsize > MAX_POINT_SIZE) {
            std::cerr << "WARNING (Visualizer::addPoint): Point size ( " << pointsize << " ) is outside of supported range ( " << MIN_POINT_SIZE << ", " << MAX_POINT_SIZE << " ). Clamping value.." << std::endl;
            if (pointsize < MIN_POINT_SIZE) {
                pointsize = MIN_POINT_SIZE;
            } else {
                pointsize = MAX_POINT_SIZE;
            }
        }
    }
    this->point_width = pointsize;

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_POINT, {position}, color, {}, -1, false, false, coordinate_system, true, false, pointsize);
    return UUID;
}

std::vector<size_t> Visualizer::addSphereByCenter(float radius, const vec3 &center, uint Ndivisions, const RGBcolor &color, CoordinateSystem coordinate_system) {
    return addSphereByCenter(radius, center, Ndivisions, make_RGBAcolor(color.r, color.g, color.b, 1), coordinate_system);
}

std::vector<size_t> Visualizer::addSphereByCenter(float radius, const vec3 &center, uint Ndivisions, const RGBAcolor &color, CoordinateSystem coordinate_system) {
    float dtheta = PI_F / scast<float>(Ndivisions);
    float dphi = 2.f * PI_F / scast<float>(Ndivisions);

    std::vector<size_t> UUIDs;
    UUIDs.reserve(2 * Ndivisions + 2 * (Ndivisions - 2) * (Ndivisions - 1));

    // bottom cap
    for (int j = 0; j < Ndivisions; j++) {
        float phi = scast<float>(j) * dphi;
        float phi_plus = scast<float>(j + 1) * dphi;

        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, phi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, -0.5f * PI_F + dtheta, phi_plus));

        UUIDs.push_back(addTriangle(v0, v1, v2, color, coordinate_system));
    }

    // top cap
    for (int j = 0; j < Ndivisions; j++) {
        float phi = scast<float>(j) * dphi;
        float phi_plus = scast<float>(j + 1) * dphi;

        vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F, 0));
        vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, phi));
        vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, 0.5f * PI_F - dtheta, phi_plus));

        UUIDs.push_back(addTriangle(v2, v1, v0, color, coordinate_system));
    }

    // middle
    for (int j = 0; j < Ndivisions; j++) {
        float phi = scast<float>(j) * dphi;
        float phi_plus = scast<float>(j + 1) * dphi;
        for (int i = 1; i < Ndivisions - 1; i++) {
            float theta = -0.5f * PI_F + scast<float>(i) * dtheta;
            float theta_plus = -0.5f * PI_F + scast<float>(i + 1) * dtheta;

            vec3 v0 = center + sphere2cart(make_SphericalCoord(radius, theta, phi));
            vec3 v1 = center + sphere2cart(make_SphericalCoord(radius, theta_plus, phi));
            vec3 v2 = center + sphere2cart(make_SphericalCoord(radius, theta_plus, phi_plus));
            vec3 v3 = center + sphere2cart(make_SphericalCoord(radius, theta, phi_plus));

            UUIDs.push_back(addTriangle(v0, v1, v2, color, coordinate_system));
            UUIDs.push_back(addTriangle(v0, v2, v3, color, coordinate_system));
        }
    }

    return UUIDs;
}

void Visualizer::addSkyDomeByCenter(float radius, const vec3 &center, uint Ndivisions, const char *texture_file, int layer) {
    // This deprecated overload with layer parameter simply ignores the layer argument
    // and calls the implementation directly to avoid cascading deprecation warnings
    std::cerr << "WARNING (Visualizer::addSkyDomeByCenter): This method is deprecated and will be removed in a future version. "
              << "Please use Visualizer::setBackgroundSkyTexture() instead, which provides a more robust sky rendering solution "
              << "that dynamically scales with camera movement and avoids the need to pre-specify a radius." << std::endl;

    float thetaStart = -0.1f * PI_F;

    float dtheta = (0.5f * PI_F - thetaStart) / float(Ndivisions - 1);
    float dphi = 2.f * PI_F / float(Ndivisions - 1);

    std::vector<size_t> UUIDs;
    UUIDs.reserve(2u * Ndivisions * Ndivisions);

    vec3 cart;

    // top cap
    for (int j = 0; j < scast<int>(Ndivisions - 1); j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F, 0));
        vec3 v0 = center + radius * cart;
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v1 = center + radius * cart;
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v2 = center + radius * cart;

        vec3 n0 = v0 - center;
        n0.normalize();
        vec3 n1 = v1 - center;
        n1.normalize();
        vec3 n2 = v2 - center;
        n2.normalize();

        vec2 uv0 = make_vec2(1.f - atan2f(sinf((float(j) + 0.5f) * dphi), -cosf((float(j) + 0.5f) * dphi)) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
        vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
        vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);

        if (j == scast<int>(Ndivisions - 2)) {
            uv2.x = 1;
        }

        UUIDs.push_back(addTriangle(v0, v1, v2, texture_file, uv0, uv1, uv2, scast<CoordinateSystem>(2)));
    }

    // middle
    for (int j = 0; j < scast<int>(Ndivisions - 1); j++) {
        for (int i = 0; i < scast<int>(Ndivisions - 1); i++) {
            cart = sphere2cart(make_SphericalCoord(1.f, float(i) * dtheta, float(j) * dphi));
            vec3 v0 = center + radius * cart;
            cart = sphere2cart(make_SphericalCoord(1.f, float(i + 1) * dtheta, float(j) * dphi));
            vec3 v1 = center + radius * cart;
            cart = sphere2cart(make_SphericalCoord(1.f, float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v2 = center + radius * cart;
            cart = sphere2cart(make_SphericalCoord(1.f, float(i) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + radius * cart;

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

            if (j == scast<int>(Ndivisions - 2)) {
                uv2.x = 1;
                uv3.x = 1;
            }

            UUIDs.push_back(addTriangle(v0, v1, v2, texture_file, uv0, uv1, uv2, scast<CoordinateSystem>(2)));
            UUIDs.push_back(addTriangle(v0, v2, v3, texture_file, uv0, uv2, uv3, scast<CoordinateSystem>(2)));
        }
    }
}

std::vector<size_t> Visualizer::addSkyDomeByCenter(float radius, const vec3 &center, uint Ndivisions, const char *texture_file) {
    std::cerr << "WARNING (Visualizer::addSkyDomeByCenter): This method is deprecated and will be removed in a future version. "
              << "Please use Visualizer::setBackgroundSkyTexture() instead, which provides a more robust sky rendering solution "
              << "that dynamically scales with camera movement and avoids the need to pre-specify a radius." << std::endl;

    float thetaStart = -0.1f * PI_F;

    float dtheta = (0.5f * PI_F - thetaStart) / float(Ndivisions - 1);
    float dphi = 2.f * PI_F / float(Ndivisions - 1);

    std::vector<size_t> UUIDs;
    UUIDs.reserve(2u * Ndivisions * Ndivisions);

    vec3 cart;

    // top cap
    for (int j = 0; j < scast<int>(Ndivisions - 1); j++) {
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F, 0));
        vec3 v0 = center + radius * cart;
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j + 1) * dphi));
        vec3 v1 = center + radius * cart;
        cart = sphere2cart(make_SphericalCoord(1.f, 0.5f * PI_F - dtheta, float(j) * dphi));
        vec3 v2 = center + radius * cart;

        vec3 n0 = v0 - center;
        n0.normalize();
        vec3 n1 = v1 - center;
        n1.normalize();
        vec3 n2 = v2 - center;
        n2.normalize();

        vec2 uv0 = make_vec2(1.f - atan2f(sinf((float(j) + 0.5f) * dphi), -cosf((float(j) + 0.5f) * dphi)) / (2.f * PI_F) - 0.5f, 1.f - n0.z * 0.5f - 0.5f);
        vec2 uv1 = make_vec2(1.f - atan2f(n1.x, -n1.y) / (2.f * PI_F) - 0.5f, 1.f - n1.z * 0.5f - 0.5f);
        vec2 uv2 = make_vec2(1.f - atan2f(n2.x, -n2.y) / (2.f * PI_F) - 0.5f, 1.f - n2.z * 0.5f - 0.5f);

        if (j == scast<int>(Ndivisions - 2)) {
            uv2.x = 1;
        }

        UUIDs.push_back(addTriangle(v0, v1, v2, texture_file, uv0, uv1, uv2, scast<CoordinateSystem>(2)));
    }

    // middle
    for (int j = 0; j < scast<int>(Ndivisions - 1); j++) {
        for (int i = 0; i < scast<int>(Ndivisions - 1); i++) {
            cart = sphere2cart(make_SphericalCoord(1.f, float(i) * dtheta, float(j) * dphi));
            vec3 v0 = center + radius * cart;
            cart = sphere2cart(make_SphericalCoord(1.f, float(i + 1) * dtheta, float(j) * dphi));
            vec3 v1 = center + radius * cart;
            cart = sphere2cart(make_SphericalCoord(1.f, float(i + 1) * dtheta, float(j + 1) * dphi));
            vec3 v2 = center + radius * cart;
            cart = sphere2cart(make_SphericalCoord(1.f, float(i) * dtheta, float(j + 1) * dphi));
            vec3 v3 = center + radius * cart;

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

            if (j == scast<int>(Ndivisions - 2)) {
                uv2.x = 1;
                uv3.x = 1;
            }

            UUIDs.push_back(addTriangle(v0, v1, v2, texture_file, uv0, uv1, uv2, scast<CoordinateSystem>(2)));
            UUIDs.push_back(addTriangle(v0, v2, v3, texture_file, uv0, uv2, uv3, scast<CoordinateSystem>(2)));
        }
    }

    return UUIDs;
}

std::vector<size_t> Visualizer::addTextboxByCenter(const char *textstring, const vec3 &center, const SphericalCoord &rotation, const RGBcolor &fontcolor, uint fontsize, const char *fontname, CoordinateSystem coordinate_system) {
    FT_Library ft; // FreeType objects
    FT_Face face;

    // initialize the freetype library
    if (FT_Init_FreeType(&ft) != 0) {
        helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Could not init freetype library");
    }

    std::vector<std::vector<unsigned char>> maskData; // This will hold the letter mask data

    // Load the font
    std::string font;
    // std::snprintf(font,100,"plugins/visualizer/fonts/%s.ttf",fontname);
    font = helios::resolvePluginAsset("visualizer", "fonts/" + (std::string) fontname + ".ttf").string();
    auto error = FT_New_Face(ft, font.c_str(), 0, &face);
    if (error != 0) {
        switch (error) {
            case FT_Err_Ok:; // do nothing
            case FT_Err_Cannot_Open_Resource:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Cannot open resource.");
            case FT_Err_Unknown_File_Format:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Unknown file format.");
            case FT_Err_Invalid_File_Format:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid file format.");
            case FT_Err_Invalid_Version:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid FreeType version.");
            case FT_Err_Lower_Module_Version:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Lower module version.");
            case FT_Err_Invalid_Argument:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid argument.");
            case FT_Err_Unimplemented_Feature:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Unimplemented feature.");
            case FT_Err_Invalid_Table:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid table.");
            case FT_Err_Invalid_Offset:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid offset.");
            case FT_Err_Array_Too_Large:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Array too large.");
            case FT_Err_Missing_Module:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Missing module.");
            case FT_Err_Out_Of_Memory:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Out of memory.");
            case FT_Err_Invalid_Face_Handle:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid face handle.");
            case FT_Err_Invalid_Size_Handle:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid size handle.");
            case FT_Err_Invalid_Slot_Handle:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid slot handle.");
            case FT_Err_Invalid_CharMap_Handle:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid charmap handle.");
            case FT_Err_Invalid_Glyph_Index:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid glyph index.");
            case FT_Err_Invalid_Character_Code:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid character code.");
            case FT_Err_Invalid_Glyph_Format:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid glyph format.");
            case FT_Err_Cannot_Render_Glyph:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Cannot render glyph.");
            case FT_Err_Invalid_Outline:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid outline.");
            case FT_Err_Invalid_Composite:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid composite glyph.");
            case FT_Err_Too_Many_Hints:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Too many hints.");
            case FT_Err_Invalid_Pixel_Size:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid pixel size.");
            case FT_Err_Invalid_Library_Handle:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid library handle.");
            case FT_Err_Invalid_Stream_Handle:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid stream handle.");
            case FT_Err_Invalid_Frame_Operation:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid frame operation.");
            case FT_Err_Nested_Frame_Access:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Nested frame access.");
            case FT_Err_Invalid_Frame_Read:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid frame read.");
            case FT_Err_Raster_Uninitialized:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Raster uninitialized.");
            case FT_Err_Raster_Corrupted:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Raster corrupted.");
            case FT_Err_Raster_Overflow:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Raster overflow.");
            case FT_Err_Raster_Negative_Height:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Raster negative height.");
            case FT_Err_Too_Many_Caches:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Too many caches.");
            case FT_Err_Invalid_Opcode:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Invalid opcode.");
            case FT_Err_Too_Few_Arguments:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Too few arguments.");
            case FT_Err_Stack_Overflow:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Stack overflow.");
            case FT_Err_Stack_Underflow:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Stack underflow.");
            case FT_Err_Ignore:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Ignore.");
            case FT_Err_No_Unicode_Glyph_Name:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): No Unicode glyph name.");
            case FT_Err_Missing_Property:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Missing property.");
            default:
                helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Unknown FreeType error.");
        }
    }
    if (error != 0) {
        helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Could not open font '" + std::string(fontname) + "'");
    }

    // Load the font size
    FT_Set_Pixel_Sizes(face, 0, fontsize);

    // x- and y- size of a pixel in [0,1] normalized coordinates
    float sx = 1.f / float(Wdisplay);
    float sy = 1.f / float(Hdisplay);

    FT_GlyphSlot gg = face->glyph; // FreeType glyph for font `fontname' and size `fontsize'

    // first, find out how wide the text is going to be
    // This is because we need to know the width beforehand if we want to center the text
    float wtext = 0;
    float htext = 0;
    const char *textt = textstring;
    for (const char *p = textt; *p; p++) { // looping over each letter in `textstring'
        if (FT_Load_Char(face, *p, FT_LOAD_RENDER)) // load the letter
            continue;
        float scale = 1;
        if (strncmp(p, "_", 1) == 0) { // subscript
            scale = 0.5;
            continue;
        } else if (strncmp(p, "^", 1) == 0) { // superscript
            scale = 0.5;
            continue;
        }
        wtext += gg->bitmap.width * sx * scale;
        htext = std::max(gg->bitmap.rows * sy, htext);
    }

    // location of the center of our textbox
    float xt = center.x - 0.5f * wtext;
    float yt = center.y - 0.5f * htext;

    if (message_flag) {
        if (coordinate_system == COORDINATES_WINDOW_NORMALIZED) {
            if (xt < 0 || xt > 1) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTextboxByCenter): text x-coordinate is outside of window area" << std::endl;
                }
            }
            if (yt < 0 || yt > 1) {
                if (message_flag) {
                    std::cerr << "WARNING (Visualizer::addTextboxByCenter): text y-coordinate is outside of window area" << std::endl;
                }
            }
        }
    }

    FT_GlyphSlot g = face->glyph; // Another FreeType glyph for font `fontname' and size `fontsize'

    std::vector<size_t> UUIDs;
    UUIDs.reserve(std::strlen(textstring));

    const char *text = textstring;

    float offset = 0; // baseline offset for subscript/superscript
    float scale = 1; // scaling factor for subscript/superscript
    for (const char *p = text; *p; p++) { // looping over each letter in `textstring'

        if (FT_Load_Char(face, *p, FT_LOAD_RENDER)) // load the letter
            continue;

        if (strncmp(p, "_", 1) == 0) { // subscript
            offset = -0.3f * sy;
            scale = 0.5f;
            continue;
        } else if (strncmp(p, "^", 1) == 0) { // superscript
            offset = 0.3f * sy;
            scale = 0.5f;
            continue;
        }

        // Copy the letter's mask into 2D `maskData' structure
        uint2 tsize(g->bitmap.width, g->bitmap.rows);
        maskData.resize(tsize.y);
        for (int j = 0; j < tsize.y; j++) {
            maskData.at(j).resize(tsize.x);
            for (int i = 0; i < tsize.x; i++) {
                maskData.at(j).at(i) = g->bitmap.buffer[i + j * tsize.x];
            }
        }

        // size of this letter (i.e., the size of the rectangle we're going to make
        vec2 lettersize = make_vec2(g->bitmap.width * scale * sx, g->bitmap.rows * scale * sy);

        // position of this letter (i.e., the center of the rectangle we're going to make
        vec3 letterposition = make_vec3(xt + g->bitmap_left * sx + 0.5 * lettersize.x, yt + g->bitmap_top * (sy + offset) - 0.5 * lettersize.y, center.z);

        // advance the x- and y- letter position
        xt += (g->advance.x >> 6) * sx * scale;
        yt += (g->advance.y >> 6) * sy * scale;

        // reset the offset and scale
        offset = 0;
        scale = 1;

        if (lettersize.x == 0 || lettersize.y == 0) { // if the size of the letter is 0, don't add a rectangle
            continue;
        }

        Glyph glyph(tsize, maskData);

        //\todo Currently, this adds a separate rectangle for each letter. Would be better to bake the whole string into a single rectangle/texture.
        UUIDs.push_back(addRectangleByCenter(letterposition, lettersize, rotation, make_RGBcolor(fontcolor.r, fontcolor.g, fontcolor.b), &glyph, coordinate_system));
    }

    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    return UUIDs;
}

void Visualizer::deleteGeometry(size_t geometry_id) {
    if (geometry_handler.doesGeometryExist(geometry_id)) {
        geometry_handler.deleteGeometry(geometry_id);
    }
}

std::vector<helios::vec3> Visualizer::getGeometryVertices(size_t geometry_id) const {
    return geometry_handler.getVertices(geometry_id);
}

void Visualizer::setGeometryVertices(size_t geometry_id, const std::vector<helios::vec3> &vertices) {
    geometry_handler.setVertices(geometry_id, vertices);
}

std::vector<size_t> Visualizer::addColorbarByCenter(const char *title, const helios::vec2 &size, const helios::vec3 &center, const helios::RGBcolor &font_color, const Colormap &colormap) {
    uint Ndivs = 50;

    float cmin = clamp(colormap.getLowerLimit(), -1e7f, 1e7f);
    float cmax = clamp(colormap.getUpperLimit(), -1e7f, 1e7f);

    // Generate tick values - either user-specified or auto-generated
    std::vector<float> tick_values;
    if (!colorbar_ticks.empty()) {
        tick_values = colorbar_ticks;
    } else {
        // Auto-generate nice tick values with adaptive count based on colorbar size
        // Estimate: need ~0.04 normalized units per tick label vertically (based on font size)
        // This accounts for font size, spacing, and readability
        float estimated_tick_spacing = 0.04f * (colorbar_fontsize / 12.0f); // Scale with font size
        int max_ticks = std::max(3, static_cast<int>(size.y / estimated_tick_spacing));
        max_ticks = std::min(max_ticks, 8); // Cap at 8 ticks maximum

        tick_values = generateNiceTicks(cmin, cmax, colorbar_integer_data, max_ticks);
    }

    uint Nticks = tick_values.size();

    std::vector<size_t> UUIDs;
    UUIDs.reserve(Ndivs + 2 * Nticks + 20);

    float dx = size.x / float(Ndivs);

    for (uint i = 0; i < Ndivs; i++) {
        float x = center.x - 0.5f * size.x + (float(i) + 0.5f) * dx;

        RGBcolor color = colormap.query(cmin + float(i) / float(Ndivs) * (cmax - cmin));

        UUIDs.push_back(addRectangleByCenter(make_vec3(x, center.y, center.z), make_vec2(dx, 0.5f * size.y), make_SphericalCoord(0, 0), color, COORDINATES_WINDOW_NORMALIZED));
    }

    std::vector<vec3> border;
    border.reserve(5);
    border.push_back(make_vec3(center.x - 0.5f * size.x, center.y + 0.25f * size.y, center.z - 0.001f));
    border.push_back(make_vec3(center.x + 0.5f * size.x, center.y + 0.25f * size.y, center.z - 0.001f));
    border.push_back(make_vec3(center.x + 0.5f * size.x, center.y - 0.25f * size.y, center.z - 0.001f));
    border.push_back(make_vec3(center.x - 0.5f * size.x, center.y - 0.25f * size.y, center.z - 0.001f));
    border.push_back(make_vec3(center.x - 0.5f * size.x, center.y + 0.25f * size.y, center.z - 0.001f));

    for (uint i = 0; i < border.size() - 1; i++) {
        UUIDs.push_back(addLine(border.at(i), border.at(i + 1), font_color, COORDINATES_WINDOW_NORMALIZED));
    }

    // Calculate tick spacing for formatting
    double tick_spacing = 1.0;
    if (Nticks > 1) {
        tick_spacing = std::fabs(tick_values[1] - tick_values[0]);
    }

    std::vector<vec3> ticks;
    ticks.resize(2);
    for (uint i = 0; i < Nticks; i++) {
        float value = tick_values.at(i);
        float x = center.x - 0.5f * size.x + (value - cmin) / (cmax - cmin) * size.x;

        // Format tick label using new formatting function
        std::string label = formatTickLabel(value, tick_spacing, colorbar_integer_data);

        // tick labels
        std::vector<size_t> UUIDs_text = addTextboxByCenter(label.c_str(), make_vec3(x, center.y - 0.4f * size.y, center.z), make_SphericalCoord(0, 0), font_color, colorbar_fontsize, "OpenSans-Regular", COORDINATES_WINDOW_NORMALIZED);
        UUIDs.insert(UUIDs.end(), UUIDs_text.begin(), UUIDs_text.end());

        if (i > 0 && i < Nticks - 1) {
            ticks[0] = make_vec3(x, center.y - 0.25f * size.y, center.z - 0.001f);
            ticks[1] = make_vec3(x, center.y - 0.25f * size.y + 0.05f * size.y, center.z - 0.001f);
            addLine(ticks[0], ticks[1], make_RGBcolor(0.25, 0.25, 0.25), COORDINATES_WINDOW_NORMALIZED);
            ticks[0] = make_vec3(x, center.y + 0.25f * size.y, center.z - 0.001f);
            ticks[1] = make_vec3(x, center.y + 0.25f * size.y - 0.05f * size.y, center.z - 0.001f);
            UUIDs.push_back(addLine(ticks[0], ticks[1], make_RGBcolor(0.25, 0.25, 0.25), COORDINATES_WINDOW_NORMALIZED));
        }
    }

    // title
    std::vector<size_t> UUIDs_text = addTextboxByCenter(title, make_vec3(center.x, center.y + 0.4f * size.y, center.z), make_SphericalCoord(0, 0), font_color, colorbar_fontsize, "CantoraOne-Regular", COORDINATES_WINDOW_NORMALIZED);
    UUIDs.insert(UUIDs.end(), UUIDs_text.begin(), UUIDs_text.end());

    return UUIDs;
}

void Visualizer::addCoordinateAxes() {
    addCoordinateAxes(helios::make_vec3(0, 0, 0), helios::make_vec3(1, 1, 1), "positive");
}

void Visualizer::addCoordinateAxes(const helios::vec3 &origin, const helios::vec3 &length, const std::string &sign) {
    float mult;
    if (sign == "both") {
        mult = 1.0;
    } else {
        mult = 0.0;
    }

    float Lmag = length.magnitude();

    std::vector<size_t> UUIDs, UUIDs_text;
    UUIDs.reserve(12);

    // x axis
    UUIDs.push_back(addLine(make_vec3(mult * -1.0f * length.x + origin.x, origin.y, origin.z), make_vec3(length.x + origin.x, origin.y, origin.z), RGB::black, Visualizer::COORDINATES_CARTESIAN));

    if (length.x > 0) {
        UUIDs_text = addTextboxByCenter("+ X", helios::make_vec3(1.2f * length.x + origin.x, origin.y, origin.z), nullrotation, helios::RGB::black, uint(200 * Lmag), "OpenSans-Regular", Visualizer::COORDINATES_CARTESIAN);
        UUIDs.insert(UUIDs.end(), UUIDs_text.begin(), UUIDs_text.end());
    }

    // y axis
    UUIDs.push_back(addLine(make_vec3(origin.x, mult * -1.0f * length.y + origin.y, origin.z), make_vec3(origin.x, length.y + origin.y, origin.z), RGB::black, Visualizer::COORDINATES_CARTESIAN));

    if (length.y > 0) {
        UUIDs_text = addTextboxByCenter("+ Y", helios::make_vec3(origin.x, 1.1f * length.y + origin.y, origin.z), nullrotation, RGB::black, uint(200 * Lmag), "OpenSans-Regular", Visualizer::COORDINATES_CARTESIAN);
        UUIDs.insert(UUIDs.end(), UUIDs_text.begin(), UUIDs_text.end());
    }

    // z axis
    UUIDs.push_back(addLine(make_vec3(origin.x, origin.y, mult * -1.f * length.z + origin.z), make_vec3(origin.x, origin.y, length.z + origin.z), RGB::black, Visualizer::COORDINATES_CARTESIAN));

    if (length.z > 0) {
        UUIDs_text = addTextboxByCenter("+ Z", helios::make_vec3(origin.x, origin.y, length.z + origin.z), nullrotation, RGB::black, uint(200 * Lmag), "OpenSans-Regular", Visualizer::COORDINATES_CARTESIAN);
        UUIDs.insert(UUIDs.end(), UUIDs_text.begin(), UUIDs_text.end());
    }

    this->coordinate_axes_IDs = UUIDs;
}

void Visualizer::disableCoordinateAxes() {
    if (!coordinate_axes_IDs.empty()) {
        geometry_handler.deleteGeometry(coordinate_axes_IDs);
    }
}

void Visualizer::addGridWireFrame(const helios::vec3 &center, const helios::vec3 &size, const helios::int3 &subdiv) {
    const helios::vec3 boxmin = make_vec3(center.x - 0.5f * size.x, center.y - 0.5f * size.y, center.z - 0.5f * size.z);
    const helios::vec3 boxmax = make_vec3(center.x + 0.5f * size.x, center.y + 0.5f * size.y, center.z + 0.5f * size.z);

    float spacing_x = size.x / scast<float>(subdiv.x);
    float spacing_y = size.y / scast<float>(subdiv.y);
    float spacing_z = size.z / scast<float>(subdiv.z);

    std::vector<size_t> UUIDs;
    UUIDs.reserve(subdiv.x * subdiv.y + subdiv.y * subdiv.z + subdiv.x * subdiv.z);

    for (int i = 0; i <= subdiv.x; i++) {
        for (int j = 0; j <= subdiv.y; j++) {
            UUIDs.push_back(addLine(make_vec3(boxmin.x + i * spacing_x, boxmin.y + j * spacing_y, boxmin.z), make_vec3(boxmin.x + i * spacing_x, boxmin.y + j * spacing_y, boxmax.z), RGB::black, Visualizer::COORDINATES_CARTESIAN));
        }
    }

    for (int i = 0; i <= subdiv.z; i++) {
        for (int j = 0; j <= subdiv.y; j++) {
            UUIDs.push_back(addLine(make_vec3(boxmin.x, boxmin.y + j * spacing_y, boxmin.z + i * spacing_z), make_vec3(boxmax.x, boxmin.y + j * spacing_y, boxmin.z + i * spacing_z), RGB::black, Visualizer::COORDINATES_CARTESIAN));
        }
    }

    for (int i = 0; i <= subdiv.x; i++) {
        for (int j = 0; j <= subdiv.z; j++) {
            UUIDs.push_back(addLine(make_vec3(boxmin.x + i * spacing_x, boxmin.y, boxmin.z + j * spacing_z), make_vec3(boxmin.x + i * spacing_x, boxmax.y, boxmin.z + j * spacing_z), RGB::black, Visualizer::COORDINATES_CARTESIAN));
        }
    }

    if (primitiveColorsNeedUpdate) {
        updateContextPrimitiveColors();
    }
}

void Visualizer::updateNavigationGizmo() {
    // If disabled, clean up and return
    if (!navigation_gizmo_enabled) {
        if (!navigation_gizmo_IDs.empty()) {
            geometry_handler.deleteGeometry(navigation_gizmo_IDs);
            navigation_gizmo_IDs.clear();
        }
        return;
    }

    // Delete existing gizmo geometry
    if (!navigation_gizmo_IDs.empty()) {
        geometry_handler.deleteGeometry(navigation_gizmo_IDs);
        navigation_gizmo_IDs.clear();
    }

    // Gizmo parameters
    const float axis_length = 0.03f; // Length of each axis line
    const float bubble_size = 0.025f; // Size of letter bubbles
    const float line_width = 4.f;

    // Calculate aspect ratio to maintain proper gizmo proportions in non-square windows
    float aspect_ratio = static_cast<float>(Wdisplay) / static_cast<float>(Hdisplay);

    // Gizmo center position - keep fixed regardless of aspect ratio
    // Use -0.9999 to ensure visibility even when scene geometry is extremely close to camera
    const vec3 gizmo_center = make_vec3(0.9f, 0.1f, -0.9999f); // Lower-right corner, as close as possible to near plane

    // Compute camera view matrix directly from current camera position
    // This avoids relying on potentially uninitialized cameraViewMatrix
    glm::mat4 view_matrix = glm::lookAt(glm_vec3(camera_eye_location), glm_vec3(camera_lookat_center), glm::vec3(0, 0, 1));

    // Extract camera right, up, and forward vectors from the view matrix
    // The view matrix transforms world coordinates to camera coordinates
    // We need the inverse to transform camera axes to world axes
    glm::mat3 rotation = glm::transpose(glm::mat3(view_matrix));

    vec3 camera_right = make_vec3(rotation[0][0], rotation[0][1], rotation[0][2]);
    vec3 camera_up = make_vec3(rotation[1][0], rotation[1][1], rotation[1][2]);

    // Define world axes
    vec3 world_x = make_vec3(1, 0, 0);
    vec3 world_y = make_vec3(0, 1, 0);
    vec3 world_z = make_vec3(0, 0, 1);

    // Project world axes onto camera's screen plane
    // For each world axis, we want to know how it appears on the 2D screen
    auto projectAxisToScreen = [&](const vec3 &world_axis) -> vec3 {
        // Project world axis onto camera's right and up vectors to get 2D screen coordinates
        float x_component = world_axis.x * camera_right.x + world_axis.y * camera_right.y + world_axis.z * camera_right.z;
        float y_component = world_axis.x * camera_up.x + world_axis.y * camera_up.y + world_axis.z * camera_up.z;

        // Calculate foreshortening based on uncorrected projection
        vec3 screen_dir_uncorrected = make_vec3(x_component, y_component, 0);
        float mag = screen_dir_uncorrected.magnitude();

        if (mag > 1e-6f) {
            // Apply foreshortening: the projected magnitude represents how much of the axis
            // is visible from the current viewing angle (1.0 = fully visible, 0.0 = end-on)
            float foreshortening_factor = mag;

            // Apply a minimum visibility threshold to prevent axes from becoming too small
            // This ensures axes remain somewhat visible even at extreme viewing angles
            const float min_visibility = 0.05f;
            foreshortening_factor = std::max(foreshortening_factor, min_visibility);

            // Normalize direction
            vec3 normalized_dir = screen_dir_uncorrected / mag;

            // Apply aspect ratio correction to maintain proper proportions
            // Key insight: keep Y-dimension constant (maintains height), adjust X-dimension
            if (aspect_ratio >= 1.0f) {
                // Wide window: compress X to compensate for wider pixels
                normalized_dir.x /= aspect_ratio;
            } else {
                // Tall window: expand Y to compensate for taller pixels
                // This maintains constant visual height while preventing skew
                normalized_dir.y *= aspect_ratio;
            }

            // Renormalize after aspect correction to maintain consistent visual size
            // Without this, the gizmo would shrink in wide windows and grow in tall windows
            float corrected_mag = normalized_dir.magnitude();
            if (corrected_mag > 1e-6f) {
                normalized_dir = normalized_dir / corrected_mag;
            }

            vec3 screen_dir = normalized_dir * (axis_length * foreshortening_factor);
            return screen_dir;
        } else {
            // If axis is perpendicular to screen (magnitude too small), make it invisible
            return make_vec3(0, 0, 0);
        }
    };

    vec3 x_screen_dir = projectAxisToScreen(world_x);
    vec3 y_screen_dir = projectAxisToScreen(world_y);
    vec3 z_screen_dir = projectAxisToScreen(world_z);

    // Calculate end points for each axis
    vec3 x_end = gizmo_center + x_screen_dir;
    vec3 y_end = gizmo_center + y_screen_dir;
    vec3 z_end = gizmo_center + z_screen_dir;

    // Add axis lines with appropriate colors
    RGBcolor x_color = make_RGBcolor(0.8, 0.26, 0.23); // Red
    RGBcolor y_color = make_RGBcolor(0.37, 0.59, 0.29); // Green
    RGBcolor z_color = make_RGBcolor(0.24, 0.39, 0.83); // Blue

    navigation_gizmo_IDs.push_back(addLine(gizmo_center, x_end, x_color, line_width, COORDINATES_WINDOW_NORMALIZED));
    navigation_gizmo_IDs.push_back(addLine(gizmo_center, y_end, y_color, line_width, COORDINATES_WINDOW_NORMALIZED));
    navigation_gizmo_IDs.push_back(addLine(gizmo_center, z_end, z_color, line_width, COORDINATES_WINDOW_NORMALIZED));

    // Calculate bubble positions - extend beyond axis endpoints and offset in z to render in front
    // Extension prevents axis lines from showing through transparent parts of the letter textures
    const float z_offset = 0.001f; // Z offset to ensure bubbles render in front of lines

    // Extension needs to account for aspect ratio to match the corrected bubble size
    // Use the average of the corrected bubble dimensions for a consistent extension
    float bubble_extension_base = bubble_size * 0.4f;

    auto extendBubblePosition = [&](const vec3 &end_pos, const vec3 &direction) -> vec3 {
        float dir_mag = direction.magnitude();
        vec3 extended_pos = end_pos;
        if (dir_mag > 1e-6f) {
            vec3 unit_dir = direction / dir_mag;

            // Calculate aspect-corrected extension distance based on direction
            // This ensures bubbles stay aligned with axis lines at all aspect ratios
            float extension_x = bubble_extension_base;
            float extension_y = bubble_extension_base;

            if (aspect_ratio >= 1.0f) {
                // Wide window: X is compressed, so use smaller X extension
                extension_x /= aspect_ratio;
            } else {
                // Tall window: Y is expanded, so use larger Y extension
                extension_y /= aspect_ratio;
            }

            // Use the component-wise corrected extension
            vec3 corrected_extension = make_vec3(unit_dir.x * extension_x, unit_dir.y * extension_y, 0.0f);

            extended_pos = end_pos + corrected_extension;
        }
        extended_pos.z += z_offset; // Move slightly toward camera for proper rendering order
        return extended_pos;
    };

    vec3 x_bubble_pos = extendBubblePosition(x_end, x_screen_dir);
    vec3 y_bubble_pos = extendBubblePosition(y_end, y_screen_dir);
    vec3 z_bubble_pos = extendBubblePosition(z_end, z_screen_dir);

    // Add textured bubbles at the extended positions, rendering in front of axis lines
    // Apply aspect ratio correction to maintain circular shape in non-square windows
    // Keep height constant, adjust width based on aspect ratio
    // Apply 25% size increase for hovered bubble
    const float hover_size_multiplier = 1.25f;

    auto calculateBubbleSize = [this, bubble_size, aspect_ratio, hover_size_multiplier](int bubble_index) -> vec2 {
        float effective_bubble_size = bubble_size;
        if (bubble_index == hovered_gizmo_bubble) {
            effective_bubble_size *= hover_size_multiplier;
        }

        if (aspect_ratio >= 1.0f) {
            // Wide window: reduce bubble width to compensate
            return make_vec2(effective_bubble_size / aspect_ratio, effective_bubble_size);
        } else {
            // Tall window: increase bubble height to maintain proportions
            return make_vec2(effective_bubble_size, effective_bubble_size / aspect_ratio);
        }
    };

    SphericalCoord no_rotation = make_SphericalCoord(0, 0);

    size_t x_bubble_id = addRectangleByCenter(x_bubble_pos, calculateBubbleSize(0), no_rotation, "plugins/visualizer/textures/nav_gizmo_x.png", COORDINATES_WINDOW_NORMALIZED);
    size_t y_bubble_id = addRectangleByCenter(y_bubble_pos, calculateBubbleSize(1), no_rotation, "plugins/visualizer/textures/nav_gizmo_y.png", COORDINATES_WINDOW_NORMALIZED);
    size_t z_bubble_id = addRectangleByCenter(z_bubble_pos, calculateBubbleSize(2), no_rotation, "plugins/visualizer/textures/nav_gizmo_z.png", COORDINATES_WINDOW_NORMALIZED);

    navigation_gizmo_IDs.push_back(x_bubble_id);
    navigation_gizmo_IDs.push_back(y_bubble_id);
    navigation_gizmo_IDs.push_back(z_bubble_id);
}
