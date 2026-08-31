/** \file "VisualizerGeometry.cpp" Visualizer geometry creation functions.

    Copyright (C) 2016-2026 Brian Bailey

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

//! Outline, fill and label colors for image annotation overlays
/**
 * The same seven colors as the visualizer's COLORMAP_LINES, written out as discrete entries rather than sampled from that Colormap. Colormap resamples its anchors into 100 interpolated entries, so all
 * but a few queries would return a blend of two adjacent anchors and lose the mutual distinctness that a categorical palette exists to provide. Bounding boxes index this palette by class ID, so that
 * every box of a class is drawn alike, while segmentation masks index it per annotation, so that two touching objects of the same class stay distinguishable.
 */
static const std::vector<helios::RGBcolor> annotation_palette_colors{{0.000f, 0.447f, 0.741f}, {0.850f, 0.325f, 0.098f}, {0.929f, 0.694f, 0.125f}, {0.494f, 0.184f, 0.556f},
                                                                     {0.466f, 0.674f, 0.188f}, {0.301f, 0.745f, 0.933f}, {0.635f, 0.078f, 0.184f}};

//! One horizontal run of filled pixels within a segmentation mask, in image pixel coordinates
struct MaskFillSpan {
    //! Row of pixels the span covers. The span spans the full height of this row, from y to y+1.
    float y;
    //! Leftmost and rightmost edge of the run
    float x_min, x_max;
};

//! Compute the horizontal runs covering the interior of a polygon, by even-odd scanline fill
/**
 * Used to fill segmentation mask contours. These are traced around a pixel mask and are routinely not simple polygons: a traced boundary crosses itself wherever the mask narrows to a one-pixel neck,
 * which is common for objects with thin or branching parts. A triangulating fill such as ear clipping fails outright on such a contour, so a scanline fill is used instead. The even-odd rule handles a
 * self-intersecting contour natively, needs no triangulation, and matches how the mask was defined in the first place, since these contours enclose whole pixels rather than describing a true vector shape.
 *
 * Scanlines are computed in image pixel space, one per pixel row, so the number of runs is bounded by the height of the mask rather than by its vertex count, and run edges land on the pixel boundaries the
 * mask actually has.
 *
 * \param[in] polygon Polygon vertices in absolute image pixel coordinates. The closing edge is implicit, so the last vertex should not repeat the first.
 * \return Runs covering the polygon interior, ordered by row. Empty if the polygon has fewer than three vertices.
 */
static std::vector<MaskFillSpan> computeMaskFillSpans(const std::vector<helios::vec2> &polygon) {

    std::vector<MaskFillSpan> spans;

    if (polygon.size() < 3) {
        return spans;
    }

    float y_min = polygon.front().y;
    float y_max = polygon.front().y;
    for (const helios::vec2 &vertex: polygon) {
        y_min = std::min(y_min, vertex.y);
        y_max = std::max(y_max, vertex.y);
    }

    const int first_row = static_cast<int>(std::floor(y_min));
    const int last_row = static_cast<int>(std::ceil(y_max));

    std::vector<float> crossings;

    for (int row = first_row; row < last_row; row++) {

        // Sampled at the center of the row rather than at its edge, so that a horizontal contour edge lying exactly on a pixel boundary does not register as a crossing.
        const float row_center = static_cast<float>(row) + 0.5f;

        crossings.clear();
        for (size_t i = 0; i < polygon.size(); i++) {
            const helios::vec2 &a = polygon.at(i);
            const helios::vec2 &b = polygon.at((i + 1) % polygon.size());

            // Half-open comparison, so a vertex touching the scanline is counted for exactly one of the two edges meeting there rather than for both or neither.
            if ((a.y <= row_center) != (b.y <= row_center)) {
                const float t = (row_center - a.y) / (b.y - a.y);
                crossings.push_back(a.x + t * (b.x - a.x));
            }
        }

        if (crossings.size() < 2) {
            continue;
        }

        std::sort(crossings.begin(), crossings.end());

        // Even-odd: the interior lies between the first and second crossing, the third and fourth, and so on. A self-intersecting contour simply yields more crossings on the affected rows, which is why
        // this fills correctly where a triangulating method fails.
        for (size_t i = 0; i + 1 < crossings.size(); i += 2) {
            if (crossings.at(i + 1) > crossings.at(i)) {
                spans.push_back({static_cast<float>(row), crossings.at(i), crossings.at(i + 1)});
            }
        }
    }

    return spans;
}

//! Open a FreeType face for a named visualizer font, sized in framebuffer pixels
/**
 * Shared by Visualizer::addTextboxByCenter() and Visualizer::getTextboxSize(). The caller name is used only to build the error messages, so that each reports failures under its own name. FreeType types are
 * confined to this file so that they do not leak into Visualizer.h.
 */
static void openVisualizerFontFace(FT_Library &ft, FT_Face &face, const char *fontname, uint fontsize_pixels, const char *caller_name) {
    // Load the font
    std::string font;
    // std::snprintf(font,100,"plugins/visualizer/fonts/%s.ttf",fontname);
    font = helios::resolvePluginAsset("visualizer", "fonts/" + (std::string) fontname + ".ttf").string();
    auto error = FT_New_Face(ft, font.c_str(), 0, &face);
    if (error != 0) {
        switch (error) {
            case FT_Err_Ok:; // do nothing
            case FT_Err_Cannot_Open_Resource:
                helios_runtime_error(std::string(caller_name) + ": Cannot open resource.");
            case FT_Err_Unknown_File_Format:
                helios_runtime_error(std::string(caller_name) + ": Unknown file format.");
            case FT_Err_Invalid_File_Format:
                helios_runtime_error(std::string(caller_name) + ": Invalid file format.");
            case FT_Err_Invalid_Version:
                helios_runtime_error(std::string(caller_name) + ": Invalid FreeType version.");
            case FT_Err_Lower_Module_Version:
                helios_runtime_error(std::string(caller_name) + ": Lower module version.");
            case FT_Err_Invalid_Argument:
                helios_runtime_error(std::string(caller_name) + ": Invalid argument.");
            case FT_Err_Unimplemented_Feature:
                helios_runtime_error(std::string(caller_name) + ": Unimplemented feature.");
            case FT_Err_Invalid_Table:
                helios_runtime_error(std::string(caller_name) + ": Invalid table.");
            case FT_Err_Invalid_Offset:
                helios_runtime_error(std::string(caller_name) + ": Invalid offset.");
            case FT_Err_Array_Too_Large:
                helios_runtime_error(std::string(caller_name) + ": Array too large.");
            case FT_Err_Missing_Module:
                helios_runtime_error(std::string(caller_name) + ": Missing module.");
            case FT_Err_Out_Of_Memory:
                helios_runtime_error(std::string(caller_name) + ": Out of memory.");
            case FT_Err_Invalid_Face_Handle:
                helios_runtime_error(std::string(caller_name) + ": Invalid face handle.");
            case FT_Err_Invalid_Size_Handle:
                helios_runtime_error(std::string(caller_name) + ": Invalid size handle.");
            case FT_Err_Invalid_Slot_Handle:
                helios_runtime_error(std::string(caller_name) + ": Invalid slot handle.");
            case FT_Err_Invalid_CharMap_Handle:
                helios_runtime_error(std::string(caller_name) + ": Invalid charmap handle.");
            case FT_Err_Invalid_Glyph_Index:
                helios_runtime_error(std::string(caller_name) + ": Invalid glyph index.");
            case FT_Err_Invalid_Character_Code:
                helios_runtime_error(std::string(caller_name) + ": Invalid character code.");
            case FT_Err_Invalid_Glyph_Format:
                helios_runtime_error(std::string(caller_name) + ": Invalid glyph format.");
            case FT_Err_Cannot_Render_Glyph:
                helios_runtime_error(std::string(caller_name) + ": Cannot render glyph.");
            case FT_Err_Invalid_Outline:
                helios_runtime_error(std::string(caller_name) + ": Invalid outline.");
            case FT_Err_Invalid_Composite:
                helios_runtime_error(std::string(caller_name) + ": Invalid composite glyph.");
            case FT_Err_Too_Many_Hints:
                helios_runtime_error(std::string(caller_name) + ": Too many hints.");
            case FT_Err_Invalid_Pixel_Size:
                helios_runtime_error(std::string(caller_name) + ": Invalid pixel size.");
            case FT_Err_Invalid_Library_Handle:
                helios_runtime_error(std::string(caller_name) + ": Invalid library handle.");
            case FT_Err_Invalid_Stream_Handle:
                helios_runtime_error(std::string(caller_name) + ": Invalid stream handle.");
            case FT_Err_Invalid_Frame_Operation:
                helios_runtime_error(std::string(caller_name) + ": Invalid frame operation.");
            case FT_Err_Nested_Frame_Access:
                helios_runtime_error(std::string(caller_name) + ": Nested frame access.");
            case FT_Err_Invalid_Frame_Read:
                helios_runtime_error(std::string(caller_name) + ": Invalid frame read.");
            case FT_Err_Raster_Uninitialized:
                helios_runtime_error(std::string(caller_name) + ": Raster uninitialized.");
            case FT_Err_Raster_Corrupted:
                helios_runtime_error(std::string(caller_name) + ": Raster corrupted.");
            case FT_Err_Raster_Overflow:
                helios_runtime_error(std::string(caller_name) + ": Raster overflow.");
            case FT_Err_Raster_Negative_Height:
                helios_runtime_error(std::string(caller_name) + ": Raster negative height.");
            case FT_Err_Too_Many_Caches:
                helios_runtime_error(std::string(caller_name) + ": Too many caches.");
            case FT_Err_Invalid_Opcode:
                helios_runtime_error(std::string(caller_name) + ": Invalid opcode.");
            case FT_Err_Too_Few_Arguments:
                helios_runtime_error(std::string(caller_name) + ": Too few arguments.");
            case FT_Err_Stack_Overflow:
                helios_runtime_error(std::string(caller_name) + ": Stack overflow.");
            case FT_Err_Stack_Underflow:
                helios_runtime_error(std::string(caller_name) + ": Stack underflow.");
            case FT_Err_Ignore:
                helios_runtime_error(std::string(caller_name) + ": Ignore.");
            case FT_Err_No_Unicode_Glyph_Name:
                helios_runtime_error(std::string(caller_name) + ": No Unicode glyph name.");
            case FT_Err_Missing_Property:
                helios_runtime_error(std::string(caller_name) + ": Missing property.");
            default:
                helios_runtime_error(std::string(caller_name) + ": Unknown FreeType error.");
        }
    }
    if (error != 0) {
        helios_runtime_error(std::string(caller_name) + ": Could not open font '" + std::string(fontname) + "'");
    }


    FT_Set_Pixel_Sizes(face, 0, fontsize_pixels);
}

//! Extent of a text string in framebuffer pixels, for a face already sized by openVisualizerFontFace()
/**
 * The advance is used rather than the bitmap width because it includes the side bearings. Summing bitmap widths underestimates the true extent and biases the text off-center. The '_' and '^'
 * subscript and superscript markers occupy no width of their own and halve the size of the character that follows.
 */
static helios::vec2 measureTextExtentPixels(FT_Face face, const char *textstring) {
    FT_GlyphSlot gg = face->glyph;

    float wtext = 0;
    float htext = 0;
    float measure_scale = 1; // scaling factor for subscript/superscript
    for (const char *p = textstring; *p; p++) { // looping over each letter in `textstring'
        if (*p == '_' || *p == '^') { // subscript/superscript marker: occupies no width of its own
            measure_scale = 0.5f;
            continue;
        }
        if (FT_Load_Char(face, *p, FT_LOAD_RENDER)) // load the letter
            continue;
        wtext += float(gg->advance.x >> 6) * measure_scale;
        htext = std::max(float(gg->bitmap.rows) * measure_scale, htext);
        measure_scale = 1;
    }

    return helios::make_vec2(wtext, htext);
}


void Visualizer::resetCachedGeometryIDs() {
    watermark_ID = 0;
    background_rectangle_ID = 0;
    background_sky_IDs.clear();
    coordinate_axes_IDs.clear();
    navigation_gizmo_IDs.clear();
    colorbar_IDs.clear();
}

void Visualizer::clearGeometry() {
    geometry_handler.clearAllGeometry();
    resetCachedGeometryIDs();

    contextUUIDs_build.clear();
    contextUUIDs_uploaded.clear();
    colorPrimitives_UUIDs.clear();
    colorPrimitives_objIDs.clear();
    depth_buffer_data.clear();
    colorbar_min = 0;
    colorbar_max = 0;
    colorbar_range_set = false;
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

size_t Visualizer::addAlphaBlendedRectangleByCenter(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char *texture_file, CoordinateSystem coordFlag) {
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

    const std::vector<vec2> uvs{{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    uint textureID = registerTextureImage(texture_file);

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, vertices, RGBA::black, uvs, textureID, false, false, coordFlag, true, false, false, 0, true);
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
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_POINT, {position}, color, {}, -1, false, false, coordinate_system, true, false, false, pointsize);
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

vec2 Visualizer::getTextboxSize(const char *textstring, uint fontsize, const char *fontname) const {
    FT_Library ft;
    FT_Face face;

    if (FT_Init_FreeType(&ft) != 0) {
        helios_runtime_error("ERROR (Visualizer::getTextboxSize): Could not init freetype library");
    }

    // Matches addTextboxByCenter(): glyphs are rasterized at framebuffer resolution, so the point size is scaled by the DPI ratio and the resulting metrics are in framebuffer pixels.
    const uint fontsize_pixels = std::max(1u, static_cast<uint>(std::lround(fontsize * getDPIScale())));

    openVisualizerFontFace(ft, face, fontname, fontsize_pixels, "ERROR (Visualizer::getTextboxSize)");

    const vec2 extent_pixels = measureTextExtentPixels(face, textstring);

    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    return make_vec2(extent_pixels.x / float(Wframebuffer), extent_pixels.y / float(Hframebuffer));
}

std::vector<size_t> Visualizer::addTextboxByCenter(const char *textstring, const vec3 &center, const SphericalCoord &rotation, const RGBcolor &fontcolor, uint fontsize, const char *fontname, CoordinateSystem coordinate_system) {
    FT_Library ft; // FreeType objects
    FT_Face face;

    // initialize the freetype library
    if (FT_Init_FreeType(&ft) != 0) {
        helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Could not init freetype library");
    }

    std::vector<std::vector<unsigned char>> maskData; // This will hold the letter mask data

    // Glyphs are rasterized at framebuffer resolution rather than at the window size in screen
    // coordinates. On a high-DPI display the two differ, and a bitmap generated at window
    // resolution is magnified when drawn into the larger framebuffer, which looks blocky. The quad
    // the glyph is drawn on is sized in window-normalized coordinates below and is unaffected;
    // only the resolution of the source bitmap changes.
    const float dpi_scale = getDPIScale();
    const uint fontsize_pixels = std::max(1u, static_cast<uint>(std::lround(fontsize * dpi_scale)));

    openVisualizerFontFace(ft, face, fontname, fontsize_pixels, "ERROR (Visualizer::addTextboxByCenter)");

    // x- and y- size of a framebuffer pixel in [0,1] normalized coordinates. The FreeType glyph
    // metrics used below are in framebuffer pixels, so these convert them directly.
    float sx = 1.f / float(Wframebuffer);
    float sy = 1.f / float(Hframebuffer);

    // first, find out how wide the text is going to be
    // This is because we need to know the width beforehand if we want to center the text
    const vec2 text_extent_pixels = measureTextExtentPixels(face, textstring);
    const float wtext = text_extent_pixels.x * sx;
    const float htext = text_extent_pixels.y * sy;

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

        // The offset is a fraction of the em size, so that the sub/superscript displacement is
        // independent of both the DPI scale and the window dimensions.
        if (*p == '_') { // subscript
            offset = -0.3f * float(fontsize_pixels) * sy;
            scale = 0.5f;
            continue;
        } else if (*p == '^') { // superscript
            offset = 0.3f * float(fontsize_pixels) * sy;
            scale = 0.5f;
            continue;
        }

        if (FT_Load_Char(face, *p, FT_LOAD_RENDER)) // load the letter
            continue;

        // Copy the letter's mask into 2D `maskData' structure
        uint2 tsize(g->bitmap.width, g->bitmap.rows);
        maskData.resize(tsize.y);
        for (int j = 0; j < tsize.y; j++) {
            maskData.at(j).resize(tsize.x);
            for (int i = 0; i < tsize.x; i++) {
                maskData.at(j).at(i) = g->bitmap.buffer[i + j * tsize.x];
            }
        }

        // size of this letter (i.e., the size of the rectangle we're going to make. The glyph
        // texture carries a one-texel transparent border on each side (see Visualizer::Texture),
        // so the rectangle is grown to match, leaving the glyph itself at its true size.
        constexpr float glyph_texture_border = 1.f;
        vec2 lettersize = make_vec2((float(g->bitmap.width) + 2.f * glyph_texture_border) * scale * sx, (float(g->bitmap.rows) + 2.f * glyph_texture_border) * scale * sy);

        // position of this letter (i.e., the center of the rectangle we're going to make. The
        // bearings are scaled alongside the glyph so that sub/superscripts are not positioned as
        // though they were full size, and `offset' translates the baseline rather than scaling it.
        // The bearings locate the glyph itself, so they are offset by the border to keep the glyph
        // in the same place now that the rectangle extends beyond it.
        vec3 letterposition = make_vec3(xt + (float(g->bitmap_left) - glyph_texture_border) * scale * sx + 0.5f * lettersize.x, yt + (float(g->bitmap_top) + glyph_texture_border) * scale * sy + offset - 0.5f * lettersize.y, center.z);

        // advance the x- and y- letter position
        xt += float(g->advance.x >> 6) * sx * scale;
        yt += float(g->advance.y >> 6) * sy * scale;

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

std::vector<size_t> Visualizer::addBoundingBoxOverlay(const std::vector<BoundingBox> &bounding_boxes, const std::map<uint, std::string> &class_names, const vec4 &image_extent, float line_width, uint fontsize) {

    // Window-normalized z is passed straight through to normalized device coordinates and the depth test is GL_LEQUAL, so smaller z is nearer the viewer. The image quad sits at z=0, so the overlay needs a
    // negative z. The label text must be strictly nearer than its own chip: glyphs render in the depth-sorted transparent pass with depth writes disabled, but they still depth-test, so a glyph coplanar
    // with its chip would be discarded by it.
    constexpr float z_box_outline = -0.01f;
    constexpr float z_label_chip = -0.02f;
    constexpr float z_label_text = -0.03f;

    constexpr char label_font[] = "OpenSans-Regular";

    const float image_width = image_extent.z - image_extent.x;
    const float image_height = image_extent.w - image_extent.y;

    // One em in window-normalized units. The chip height is derived from this rather than from the measured text height so that every chip in an overlay is the same height. The measured height is that of
    // the tallest glyph in that particular string, so "corn" would otherwise get a visibly shorter chip than "bunny".
    const uint fontsize_pixels = std::max(1u, static_cast<uint>(std::lround(fontsize * getDPIScale())));
    const float em = float(fontsize_pixels) / float(Hframebuffer);

    std::vector<size_t> UUIDs;

    for (const BoundingBox &box: bounding_boxes) {

        std::string label;
        if (class_names.empty()) {
            // No class name file was available, so the numeric class ID is all there is to show.
            label = std::to_string(box.class_ID);
        } else if (class_names.find(box.class_ID) == class_names.end()) {
            helios_runtime_error("ERROR (Visualizer::addBoundingBoxOverlay): The bounding boxes contain class ID " + std::to_string(box.class_ID) +
                                 ", which is not defined in the class name file. The bounding box annotations and the class names do not correspond to each other.");
        } else {
            label = class_names.at(box.class_ID);
        }

        // Map from normalized image coordinates, whose origin is the top-left corner of the image, into window-normalized coordinates, whose origin is the bottom-left corner of the window. Measuring down
        // from the top edge of the image extent performs the vertical flip.
        float x_left = image_extent.x + (box.center.x - 0.5f * box.size.x) * image_width;
        float x_right = image_extent.x + (box.center.x + 0.5f * box.size.x) * image_width;
        float y_top = image_extent.w - (box.center.y - 0.5f * box.size.y) * image_height;
        float y_bottom = image_extent.w - (box.center.y + 0.5f * box.size.y) * image_height;

        // A box touching the image border can extend slightly outside it once its center and size are recombined. Clamping to the image rather than to the window also keeps the label off the letterboxed
        // margin, and keeps every vertex inside the drawable area so that no "outside of drawable area" warnings are emitted.
        x_left = clamp(x_left, image_extent.x, image_extent.z);
        x_right = clamp(x_right, image_extent.x, image_extent.z);
        y_bottom = clamp(y_bottom, image_extent.y, image_extent.w);
        y_top = clamp(y_top, image_extent.y, image_extent.w);

        const RGBcolor box_color = annotation_palette_colors.at(box.class_ID % annotation_palette_colors.size());

        // The outline is drawn as four lines because there is no unfilled-rectangle primitive, following addColorbarByCenter(), which draws its border the same way. The line width is scaled by the DPI
        // ratio because the line geometry shader expresses width against the framebuffer, so an unscaled width would draw at half size on a high-DPI display.
        const std::vector<vec3> outline{make_vec3(x_left, y_bottom, z_box_outline), make_vec3(x_right, y_bottom, z_box_outline), make_vec3(x_right, y_top, z_box_outline), make_vec3(x_left, y_top, z_box_outline),
                                        make_vec3(x_left, y_bottom, z_box_outline)};
        for (size_t i = 0; i + 1 < outline.size(); i++) {
            UUIDs.push_back(addLine(outline.at(i), outline.at(i + 1), box_color, line_width * getDPIScale(), COORDINATES_WINDOW_NORMALIZED));
        }

        // The label chip is anchored inside the box at its top-left corner. Placing it inside rather than above the box means that a box at the top edge of the image still has somewhere to put its label,
        // and that a vertical stack of boxes does not draw each label over the box above it.
        const vec2 text_size = getTextboxSize(label.c_str(), fontsize, label_font);
        const vec2 chip_size = make_vec2(text_size.x + em, 1.6f * em);

        float chip_x_min = x_left;
        if (chip_x_min + chip_size.x > image_extent.z) {
            chip_x_min = image_extent.z - chip_size.x;
        }
        chip_x_min = std::max(chip_x_min, image_extent.x);
        const float chip_x_max = std::min(chip_x_min + chip_size.x, image_extent.z);
        const float chip_y_max = y_top;
        const float chip_y_min = std::max(chip_y_max - chip_size.y, image_extent.y);

        // Built from explicit corners rather than from a center and a size, so that the clamped edges survive exactly. Recovering the corners from a center and a half-size can land a rounding step outside
        // the drawable area, which would emit a warning for a box that is legitimately flush with the image border.
        const std::vector<vec3> chip_vertices{make_vec3(chip_x_min, chip_y_min, z_label_chip), make_vec3(chip_x_max, chip_y_min, z_label_chip), make_vec3(chip_x_max, chip_y_max, z_label_chip),
                                              make_vec3(chip_x_min, chip_y_max, z_label_chip)};
        UUIDs.push_back(addRectangleByVertices(chip_vertices, make_RGBAcolor(box_color, 1.f), COORDINATES_WINDOW_NORMALIZED));

        const vec3 chip_center = make_vec3(0.5f * (chip_x_min + chip_x_max), 0.5f * (chip_y_min + chip_y_max), z_label_chip);

        // Black or white text, whichever contrasts with the chip, so that the label stays readable across the whole palette.
        const float chip_luminance = 0.299f * box_color.r + 0.587f * box_color.g + 0.114f * box_color.b;
        const RGBcolor font_color = (chip_luminance > 0.5f) ? RGB::black : RGB::white;

        const std::vector<size_t> label_UUIDs = addTextboxByCenter(label.c_str(), make_vec3(chip_center.x, chip_center.y, z_label_text), make_SphericalCoord(0, 0), font_color, fontsize, label_font,
                                                                   COORDINATES_WINDOW_NORMALIZED);
        UUIDs.insert(UUIDs.end(), label_UUIDs.begin(), label_UUIDs.end());
    }

    return UUIDs;
}

std::vector<size_t> Visualizer::addSegmentationMaskOverlay(const std::vector<SegmentationMask> &masks, const vec4 &image_extent, float fill_opacity, float line_width, uint fontsize, bool show_labels) {

    // Same depth convention as addBoundingBoxOverlay(): the image quad sits at z=0 and smaller z is nearer the viewer, so the overlay needs a negative z. The fill sits behind the outline so that an
    // outline is never dimmed by the translucent fill drawn over it, and the label text stays strictly nearer than its own chip because glyphs still depth-test against it.
    constexpr float z_mask_fill = -0.005f;
    constexpr float z_mask_outline = -0.01f;
    constexpr float z_label_chip = -0.02f;
    constexpr float z_label_text = -0.03f;

    constexpr char label_font[] = "OpenSans-Regular";

    const float image_width = image_extent.z - image_extent.x;
    const float image_height = image_extent.w - image_extent.y;

    // One em in window-normalized units, giving every chip in an overlay the same height regardless of which glyphs its label happens to contain.
    const uint fontsize_pixels = std::max(1u, static_cast<uint>(std::lround(fontsize * getDPIScale())));
    const float em = float(fontsize_pixels) / float(Hframebuffer);

    std::vector<size_t> UUIDs;

    for (size_t mask_index = 0; mask_index < masks.size(); mask_index++) {

        const SegmentationMask &mask = masks.at(mask_index);

        // Colored by position in the file rather than by class ID, so that two touching objects of the same class do not merge into one indistinguishable blob.
        const RGBcolor mask_color = annotation_palette_colors.at(mask_index % annotation_palette_colors.size());

        const std::string label = mask.class_name.empty() ? std::to_string(mask.class_ID) : mask.class_name;

        // Map from absolute image pixels, whose origin is the top-left corner of the image, into window-normalized coordinates, whose origin is the bottom-left corner of the window. Measuring down from
        // the top edge of the image extent performs the vertical flip. Clamping to the image keeps every vertex inside the drawable area, so no "outside of drawable area" warnings are emitted.
        auto toWindowCoordinates = [&](const vec2 &pixel) {
            const float x = image_extent.x + (pixel.x / mask.image_size.x) * image_width;
            const float y = image_extent.w - (pixel.y / mask.image_size.y) * image_height;
            return make_vec2(clamp(x, image_extent.x, image_extent.z), clamp(y, image_extent.y, image_extent.w));
        };

        // Tracks the top-left-most vertex of the whole mask, where the label chip is anchored.
        vec2 label_anchor = make_vec2(image_extent.z, image_extent.y);
        bool label_anchor_set = false;

        for (const std::vector<vec2> &polygon: mask.polygons) {

            std::vector<vec2> vertices;
            vertices.reserve(polygon.size());
            for (const vec2 &pixel: polygon) {
                vertices.push_back(toWindowCoordinates(pixel));
            }

            for (const vec2 &vertex: vertices) {
                // Highest vertex wins, and the leftmost of those if several share that height, matching where addBoundingBoxOverlay() puts its chip.
                if (!label_anchor_set || vertex.y > label_anchor.y || (vertex.y == label_anchor.y && vertex.x < label_anchor.x)) {
                    label_anchor = vertex;
                    label_anchor_set = true;
                }
            }

            if (fill_opacity > 0.f) {
                // The fill is computed in image pixel space and mapped afterwards, so that its runs align with the pixel rows the mask is defined on rather than with the window.
                for (const MaskFillSpan &span: computeMaskFillSpans(polygon)) {
                    const vec2 top_left = toWindowCoordinates(make_vec2(span.x_min, span.y));
                    const vec2 bottom_right = toWindowCoordinates(make_vec2(span.x_max, span.y + 1.f));

                    // A run clipped away to nothing by the image extent would otherwise emit a degenerate rectangle.
                    if (bottom_right.x <= top_left.x || top_left.y <= bottom_right.y) {
                        continue;
                    }

                    const std::vector<vec3> span_vertices{make_vec3(top_left.x, bottom_right.y, z_mask_fill), make_vec3(bottom_right.x, bottom_right.y, z_mask_fill),
                                                          make_vec3(bottom_right.x, top_left.y, z_mask_fill), make_vec3(top_left.x, top_left.y, z_mask_fill)};
                    UUIDs.push_back(addRectangleByVertices(span_vertices, make_RGBAcolor(mask_color, fill_opacity), COORDINATES_WINDOW_NORMALIZED));
                }
            }

            // The outline closes back onto the first vertex, since the contour is a loop whose closing edge is implicit.
            for (size_t i = 0; i < vertices.size(); i++) {
                const vec2 &start = vertices.at(i);
                const vec2 &end = vertices.at((i + 1) % vertices.size());
                UUIDs.push_back(addLine(make_vec3(start.x, start.y, z_mask_outline), make_vec3(end.x, end.y, z_mask_outline), mask_color, line_width * getDPIScale(), COORDINATES_WINDOW_NORMALIZED));
            }
        }

        // The chip and its text are the only geometry the label contributes, so suppressing it is simply a matter of not adding them. The fill and outline above are unaffected.
        if (!show_labels || !label_anchor_set) {
            continue;
        }

        const vec2 text_size = getTextboxSize(label.c_str(), fontsize, label_font);
        const vec2 chip_size = make_vec2(text_size.x + em, 1.6f * em);

        float chip_x_min = label_anchor.x;
        if (chip_x_min + chip_size.x > image_extent.z) {
            chip_x_min = image_extent.z - chip_size.x;
        }
        chip_x_min = std::max(chip_x_min, image_extent.x);
        const float chip_x_max = std::min(chip_x_min + chip_size.x, image_extent.z);
        const float chip_y_max = label_anchor.y;
        const float chip_y_min = std::max(chip_y_max - chip_size.y, image_extent.y);

        const std::vector<vec3> chip_vertices{make_vec3(chip_x_min, chip_y_min, z_label_chip), make_vec3(chip_x_max, chip_y_min, z_label_chip), make_vec3(chip_x_max, chip_y_max, z_label_chip),
                                              make_vec3(chip_x_min, chip_y_max, z_label_chip)};
        UUIDs.push_back(addRectangleByVertices(chip_vertices, make_RGBAcolor(mask_color, 1.f), COORDINATES_WINDOW_NORMALIZED));

        const vec3 chip_center = make_vec3(0.5f * (chip_x_min + chip_x_max), 0.5f * (chip_y_min + chip_y_max), z_label_chip);

        // Black or white text, whichever contrasts with the chip, so that the label stays readable across the whole palette.
        const float chip_luminance = 0.299f * mask_color.r + 0.587f * mask_color.g + 0.114f * mask_color.b;
        const RGBcolor font_color = (chip_luminance > 0.5f) ? RGB::black : RGB::white;

        const std::vector<size_t> label_UUIDs = addTextboxByCenter(label.c_str(), make_vec3(chip_center.x, chip_center.y, z_label_text), make_SphericalCoord(0, 0), font_color, fontsize, label_font,
                                                                   COORDINATES_WINDOW_NORMALIZED);
        UUIDs.insert(UUIDs.end(), label_UUIDs.begin(), label_UUIDs.end());
    }

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
    double tick_spacing = 1.0;
    if (!colorbar_ticks.empty()) {
        tick_values = colorbar_ticks;
        // User-supplied ticks have no generating spacing, so derive it from the values themselves.
        if (tick_values.size() > 1) {
            tick_spacing = std::fabs(tick_values[1] - tick_values[0]);
        }
    } else {
        // Auto-generate nice tick values with adaptive count based on colorbar size.
        // Tick labels are laid out along the length of the bar, so the space available for them is
        // set by size.x - size.y is the bar's thickness and does not constrain the label count.
        // Estimate ~0.06 normalized units of width per label, scaled with the font size.
        float estimated_tick_spacing = 0.06f * (colorbar_fontsize / 12.0f);
        int max_ticks = std::max(3, static_cast<int>(size.x / estimated_tick_spacing));
        max_ticks = std::min(max_ticks, 8); // Cap at 8 ticks maximum

        // Ticks are confined to [cmin, cmax] as they are generated. Filtering them afterwards -
        // as this code used to - could delete every tick that generateNiceTicks() placed outside
        // the range, leaving the colorbar with a single label and no sense of scale.
        tick_values = generateColorbarTicks(cmin, cmax, colorbar_integer_data, max_ticks, &tick_spacing);
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

    // tick_spacing was set alongside tick_values above. It is the spacing used to generate them,
    // not one recovered from the final vector, so it stays correct when there is only one tick or
    // when the values are not uniformly spaced.

    // Guard against a degenerate colormap range, which would otherwise divide by zero below and
    // produce non-finite vertex coordinates.
    const float colorbar_span = cmax - cmin;
    const bool has_finite_span = std::isfinite(colorbar_span) && std::fabs(colorbar_span) > 0.f;

    std::vector<vec3> ticks;
    ticks.resize(2);
    for (uint i = 0; i < Nticks; i++) {
        float value = tick_values.at(i);
        float x = has_finite_span ? center.x - 0.5f * size.x + (value - cmin) / colorbar_span * size.x : center.x;

        // Format tick label using new formatting function
        std::string label = formatTickLabel(value, tick_spacing, colorbar_integer_data);

        // tick labels
        std::vector<size_t> UUIDs_text = addTextboxByCenter(label.c_str(), make_vec3(x, center.y - 0.4f * size.y, center.z), make_SphericalCoord(0, 0), font_color, colorbar_fontsize, "OpenSans-Regular", COORDINATES_WINDOW_NORMALIZED);
        UUIDs.insert(UUIDs.end(), UUIDs_text.begin(), UUIDs_text.end());

        if (i > 0 && i < Nticks - 1) {
            ticks[0] = make_vec3(x, center.y - 0.25f * size.y, center.z - 0.001f);
            ticks[1] = make_vec3(x, center.y - 0.25f * size.y + 0.05f * size.y, center.z - 0.001f);
            // The returned UUID must be recorded like every other piece of colorbar geometry:
            // updateColorbar() deletes the old colorbar by iterating colorbar_IDs, so a tick mark
            // left out of this list is never deleted and accumulates on every colorbar refresh.
            UUIDs.push_back(addLine(ticks[0], ticks[1], make_RGBcolor(0.25, 0.25, 0.25), COORDINATES_WINDOW_NORMALIZED));
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
    const float line_width = 3.f;

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

    // Resolved once rather than on every call: this function runs on each frame the gizmo is
    // visible, and resolvePluginAsset() stats the filesystem. If resolution throws because an asset
    // is missing, the static is left uninitialized and the next call retries, so a transient failure
    // is not cached.
    static const std::string x_bubble_texture = helios::resolvePluginAsset("visualizer", "textures/nav_gizmo_x.png").string();
    static const std::string y_bubble_texture = helios::resolvePluginAsset("visualizer", "textures/nav_gizmo_y.png").string();
    static const std::string z_bubble_texture = helios::resolvePluginAsset("visualizer", "textures/nav_gizmo_z.png").string();

    size_t x_bubble_id = addRectangleByCenter(x_bubble_pos, calculateBubbleSize(0), no_rotation, x_bubble_texture.c_str(), COORDINATES_WINDOW_NORMALIZED);
    size_t y_bubble_id = addRectangleByCenter(y_bubble_pos, calculateBubbleSize(1), no_rotation, y_bubble_texture.c_str(), COORDINATES_WINDOW_NORMALIZED);
    size_t z_bubble_id = addRectangleByCenter(z_bubble_pos, calculateBubbleSize(2), no_rotation, z_bubble_texture.c_str(), COORDINATES_WINDOW_NORMALIZED);

    navigation_gizmo_IDs.push_back(x_bubble_id);
    navigation_gizmo_IDs.push_back(y_bubble_id);
    navigation_gizmo_IDs.push_back(z_bubble_id);
}
