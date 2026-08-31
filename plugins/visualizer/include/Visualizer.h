/** \file "Visualizer.h" Visualizer header.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_VISUALIZER
#define HELIOS_VISUALIZER

#include "Context.h"

// GLM Libraries (math-related functions for graphics)
#define GLM_FORCE_RADIANS
#ifndef APIENTRY
#define APIENTRY
#endif
#include <GLFW/glfw3.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"

#include "GeometryHandler.h"

class Visualizer;


/**
 * \brief Validates the given texture file.
 *
 * This function checks whether the provided texture file can be used for loading a texture.
 * If the pngonly flag is set to true, the function specifically validates for PNG file formats.
 *
 * \param[in] texture_file The path to the texture file to validate.
 * \param[in] pngonly [optional] If true, only validates the file for PNG format. Defaults to false.
 * \return True if the texture file is valid, false otherwise.
 */
bool validateTextureFile(const std::string &texture_file, bool pngonly = false);

//! Callback function for mouse button presses
void mouseCallback(GLFWwindow *window, int button, int action, int mods);

//! Callback function for mouse cursor movements
void cursorCallback(GLFWwindow *window, double x, double y);

//! Callback function for mouse scroll
void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);

//! Glyph object - 2D matrix shape
class Glyph {
public:
    Glyph() = default;
    Glyph(const helios::uint2 &size, const std::vector<std::vector<unsigned char>> &data) : size(size), data(data) {
    }
    helios::uint2 size;
    std::vector<std::vector<unsigned char>> data;
};

//! OpenGL Shader data structure
struct Shader {

    //! Disable texture maps and color fragments by interpolating vertex colors
    void disableTextures() const;

    //! Enable texture maps and color fragments using an RGB texture map
    void enableTextureMaps() const;

    //! Enable texture masks and color fragments by interpolating vertex colors
    void enableTextureMasks() const;

    //! Set the shader transformation matrix, i.e., the Affine transformation applied to all vertices
    void setTransformationMatrix(const glm::mat4 &matrix) const;

    //! Set the view matrix for camera transformations
    void setViewMatrix(const glm::mat4 &matrix) const;

    //! Set the projection matrix for camera perspective
    void setProjectionMatrix(const glm::mat4 &matrix) const;

    //! Set the depth bias matrix for shadows
    void setDepthBiasMatrix(const glm::mat4 &matrix) const;

    //! Set the direction of the light (sun)
    void setLightDirection(const helios::vec3 &direction) const;

    //! Set the lighting model
    void setLightingModel(uint lightingmodel) const;

    //! Set the intensity of the light source
    void setLightIntensity(float lightintensity) const;

    //! Set the multiplier applied to vertex-interpolated primitive colors
    /**
     * \param[in] colorboost Multiplier applied to vertex colors in the fragment shader.
     */
    void setColorBoost(float colorboost) const;

    //! Enable or disable the linear-light rendering pipeline
    /**
     * \param[in] enabled True to decode albedo to linear light, tone-map, and re-encode to sRGB; false to write shaded values directly.
     */
    void setLinearPipeline(bool enabled) const;

    //! Set the exposure applied in linear light before tone mapping
    /**
     * \param[in] exposure Linear exposure multiplier. Has no effect when the linear pipeline is disabled.
     */
    void setExposure(float exposure) const;

    //! Set the camera position in world space, used to form the view vector for the specular term
    void setCameraPositionUniform(const helios::vec3 &position) const;

    //! Set the Phong material reflectance parameters
    /**
     * \param[in] ambient Ambient reflectance weight.
     * \param[in] diffuse Diffuse reflectance weight.
     * \param[in] specular Specular reflectance weight.
     * \param[in] shininess Specular exponent.
     */
    void setPhongMaterial(float ambient, float diffuse, float specular, float shininess) const;

    //! Set the hemispheric ambient sky and ground colors
    void setAmbientColors(const helios::RGBcolor &sky_color, const helios::RGBcolor &ground_color) const;

    //! Enable or disable smooth (interpolated per-vertex) normals
    void setSmoothShading(bool enabled) const;

    //! Bind the packed per-material Phong parameter table
    void setPhongMaterialTable(GLint table_size) const;

    //! Set shader as current
    void useShader() const;

    //! Initialize the shader
    /**
     * \param[in] vertex_shader_file Name of vertex shader file to be used by OpenGL in rendering graphics
     * \param[in] fragment_shader_file Name of fragment shader file to be used by OpenGL in rendering graphics
     * \param[in] visualizer_ptr Pointer to the Visualizer class
     * \param[in] geometry_shader_file [optional] Name of geometry shader file to be used by OpenGL in rendering graphics (default is nullptr for no geometry shader)
     */
    void initialize(const char *vertex_shader_file, const char *fragment_shader_file, Visualizer *visualizer_ptr, const char *geometry_shader_file = nullptr);

    ~Shader();

    // Primary Shader
    uint shaderID;
    GLint textureUniform;
    GLint shadowmapUniform;
    GLint transformMatrixUniform;
    GLint viewMatrixUniform;
    GLint projectionMatrixUniform;
    GLint depthBiasUniform;
    GLint lightDirectionUniform;
    GLint lightingModelUniform;
    GLint RboundUniform;
    GLint lightIntensityUniform;
    GLint colorBoostUniform;
    GLint linearPipelineUniform;
    GLint exposureUniform;
    GLint cameraPositionUniform;
    GLint materialAmbientUniform;
    GLint materialDiffuseUniform;
    GLint materialSpecularUniform;
    GLint materialShininessUniform;
    GLint ambientSkyColorUniform;
    GLint ambientGroundColorUniform;
    GLint smoothShadingUniform;
    GLint phongMaterialTableUniform;
    GLint phongMaterialTableSizeUniform;
    GLint materialIndexTextureObjectUniform;
    std::vector<GLuint> vertex_array_IDs;
    GLint uvRescaleUniform;

    // Texture buffer uniform locations (cached to avoid glGetUniformLocation during rendering)
    GLint colorTextureObjectUniform;
    GLint normalTextureObjectUniform;
    GLint textureFlagTextureObjectUniform;
    GLint textureIDTextureObjectUniform;
    GLint coordinateFlagTextureObjectUniform;
    GLint skyGeometryFlagTextureObjectUniform;
    GLint hiddenFlagTextureObjectUniform;

    //! Indicates whether initialize() has been successfully called
    bool initialized = false;
};

//! RGB color map
struct Colormap {

    Colormap() : cmapsize(0), minval(0.0f), maxval(1.0f) {};

    Colormap(const std::vector<helios::RGBcolor> &ctable, const std::vector<float> &clocs, int size, float minval_, float maxval_) : cmapsize(size), minval(minval_), maxval(maxval_) {
        set(ctable, clocs, size, minval_, maxval_);
    }

    void set(const std::vector<helios::RGBcolor> &ctable, const std::vector<float> &clocs, int size, float a_minval, float a_maxval) {
        cmapsize = size;
        minval = a_minval;
        maxval = a_maxval;

        size_t Ncolors = ctable.size();

        assert(clocs.size() == Ncolors && minval < maxval);

        cmap.resize(Ncolors);

        std::vector<float> cinds;
        cinds.resize(Ncolors);

        for (uint i = 0; i < Ncolors; i++) {
            cinds.at(i) = clocs.at(i) * static_cast<float>(cmapsize - 1);
        }

        cmap.resize(cmapsize);
        for (uint c = 0; c < Ncolors - 1; c++) {
            float cmin = cinds.at(c);
            float cmax = cinds.at(c + 1);

            for (uint i = 0; i < cmapsize; i++) {
                auto i_f = static_cast<float>(i);

                if (i_f >= cmin && i_f <= cmax) {
                    cmap.at(i).r = ctable.at(c).r + (i_f - cmin) / (cmax - cmin) * (ctable.at(c + 1).r - ctable.at(c).r);
                    cmap.at(i).g = ctable.at(c).g + (i_f - cmin) / (cmax - cmin) * (ctable.at(c + 1).g - ctable.at(c).g);
                    cmap.at(i).b = ctable.at(c).b + (i_f - cmin) / (cmax - cmin) * (ctable.at(c + 1).b - ctable.at(c).b);
                }
            }
        }
    }

    [[nodiscard]] helios::RGBcolor query(float x) const {
        assert(cmapsize > 0 && !cmap.empty());

        helios::RGBcolor color;

        uint color_ind;
        if (minval == maxval) {
            color_ind = 0;
        } else {
            float normalized_pos = (x - minval) / (maxval - minval) * float(cmapsize - 1);

            // Handle values below minimum range
            if (normalized_pos < 0) {
                color_ind = 0;
            }
            // Handle values above maximum range
            else if (normalized_pos > float(cmapsize - 1)) {
                color_ind = cmapsize - 1;
            }
            // Handle values within range
            else {
                color_ind = std::round(normalized_pos);
            }
        }

        color.r = cmap.at(color_ind).r;
        color.g = cmap.at(color_ind).g;
        color.b = cmap.at(color_ind).b;

        return color;
    }

    void setRange(float min, float max) {
        minval = min;
        maxval = max;
    }

    [[nodiscard]] helios::vec2 getRange() const {
        return {minval, maxval};
    }

    [[nodiscard]] float getLowerLimit() const {
        return minval;
    }

    [[nodiscard]] float getUpperLimit() const {
        return maxval;
    }

private:
    std::vector<helios::RGBcolor> cmap;
    unsigned int cmapsize;
    float minval, maxval;
};

//! Reads a JPEG file and extracts its pixel data.
/**
 * This function reads a JPEG file from the specified path, decodes it into RGB pixel data,
 * and populates the provided texture vector with RGBA (Red, Green, Blue, Alpha) values.
 * Each pixel in the texture is represented as four unsigned bytes, with the alpha channel
 * always set to 255 (opaque). The function also outputs the height and width of the image.
 *
 * \param[in] filename The path to the JPEG file to read.
 * \param[out] texture Vector that will be populated with the decoded RGBA pixel data.
 * \param[out] height Reference to store the height of the read image.
 * \param[out] width Reference to store the width of the read image.
 * \return Always returns 0 upon completion.
 */
int read_JPEG_file(const char *filename, std::vector<unsigned char> &texture, uint &height, uint &width);

//! Writes an image to a JPEG file.
/**
 * This function captures the current framebuffer content, converts it into a JPEG-compatible
 * data structure, and writes it to the specified file.
 *
 * \param[in] filename The path to the output JPEG file.
 * \param[in] width The width of the image to be written.
 * \param[in] height The height of the image to be written.
 * \param[in] print_messages [optional] If true, outputs status messages to the console. Defaults to false.
 * \return An integer indicating success (1) or failure (0) of the writing operation.
 */
int write_JPEG_file(const char *filename, uint width, uint height, bool buffers_swapped_since_render, bool print_messages);

//! Writes image data to a JPEG file.
/**
 * This function saves the given image data as a JPEG file to the specified filename,
 * with the provided width and height. Optionally, it can print status messages
 * to the console during the process.
 *
 * \param[in] filename The name of the file where the image will be saved.
 * \param[in] width The width of the image in pixels.
 * \param[in] height The height of the image in pixels.
 * \param[in] data A vector containing the RGB color data for the image.
 * \param[in] print_messages [optional] Whether to print status messages to the console. Defaults to false.
 * \return Returns 1 if the file was successfully written.
 */
int write_JPEG_file(const char *filename, uint width, uint height, const std::vector<helios::RGBcolor> &data, bool print_messages);

//! Writes an image to a PNG file.
/**
 * This function captures the current framebuffer content, converts it into a PNG-compatible
 * data structure, and writes it to the specified file.
 *
 * \param[in] filename The path to the output PNG file.
 * \param[in] width The width of the image to be written.
 * \param[in] height The height of the image to be written.
 * \param[in] buffers_swapped_since_render Flag indicating whether buffers have been swapped since last render.
 * \param[in] transparent_background If true, writes with alpha transparency; if false, writes opaque RGB data.
 * \param[in] print_messages [optional] If true, outputs status messages to the console. Defaults to false.
 * \return An integer indicating success (1) or failure (0) of the writing operation.
 */
int write_PNG_file(const char *filename, uint width, uint height, bool buffers_swapped_since_render, bool transparent_background, bool print_messages);

//! Writes image data to a PNG file.
/**
 * This function saves the given image data as a PNG file to the specified filename,
 * with the provided width and height. Optionally, it can print status messages
 * to the console during the process.
 *
 * \param[in] filename The name of the file where the image will be saved.
 * \param[in] width The width of the image in pixels.
 * \param[in] height The height of the image in pixels.
 * \param[in] data A vector containing the RGBA color data for the image.
 * \param[in] print_messages [optional] Whether to print status messages to the console. Defaults to false.
 * \return Returns 1 if the file was successfully written.
 */
int write_PNG_file(const char *filename, uint width, uint height, const std::vector<helios::RGBAcolor> &data, bool print_messages);

//! Reads a PNG file and extracts its pixel data.
/**
 * This function loads a PNG file and processes its pixel data into a texture format.
 * It also retrieves the dimensions of the image.
 *
 * \param[in] filename Path to the PNG file to be read.
 * \param[out] texture Vector to store the extracted RGBA pixel data as unsigned char values.
 * \param[out] height Variable to store the height (in pixels) of the loaded image.
 * \param[out] width Variable to store the width (in pixels) of the loaded image.
 */
void read_png_file(const char *filename, std::vector<unsigned char> &texture, uint &height, uint &width);

//! Class for visualization of simulation results
class Visualizer {
public:
    //! forbid the default constructor
    Visualizer() = delete;

    //! Visualizer constructor
    /**
     * \param[in] Wdisplay Width of the display window in pixels, and assumes default window aspect ratio of 1.25
     */
    explicit Visualizer(uint Wdisplay);

    //! Visualizer constructor
    /**
     * \param[in] Wdisplay Width of the display window in pixels
     * \param[in] Hdisplay Height of the display window in pixels
     */
    Visualizer(uint Wdisplay, uint Hdisplay);

    //! Constructs a Visualizer object with the specified display dimensions and anti-aliasing settings.
    /**
     * \param[in] Wdisplay Width of the display in pixels.
     * \param[in] Hdisplay Height of the display in pixels.
     * \param[in] aliasing_samples Number of anti-aliasing samples to use.
     */
    Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples);

    //! Visualizer constructor with option to remove window decorations (e.g., header bar, trim). This is a workaround for an error that occurs on Linux systems when printing the window to a JPEG image (printWindow). Once a fix is found, this
    //! function will likely be removed
    /**
     * \param[in] Wdisplay Width of the display in pixels.
     * \param[in] Hdisplay Height of the display in pixels.
     * \param[in] aliasing_samples Number of anti-aliasing samples to use.
     * \param[in] window_decorations Flag to remove window decorations.
     * \param[in] headless If true, initializes the visualizer without opening a window.
     */
    Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples, bool window_decorations, bool headless);

    //! Visualizer destructor
    ~Visualizer();

    //! Visualizer self-test routine
    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Enable standard output from this plug-in (default)
    void enableMessages();

    //! Disable standard output from this plug-in
    void disableMessages();

    //! Coordinate system to be used when specifying spatial coordinates
    enum CoordinateSystem {
        //! Coordinates are normalized to unity and are window-aligned.  The point (x,y)=(0,0) is in the bottom left corner of the window, and (x,y)=(1,1) is in the upper right corner of the window.  The z-coordinate specifies the depth in the
        //! screen-normal direction, with values ranging from -1 to 1.  Smaller z is nearer to the viewer, so an object at z=0.5 would be behind an object at z=0.
        COORDINATES_WINDOW_NORMALIZED = 0,

        //! Coordinates are specified in a 3D Cartesian system (right-handed), where +z is vertical.
        COORDINATES_CARTESIAN = 1
    };

    //! A single bounding box read from a bounding box annotation file
    /**
     * Coordinates follow the YOLO convention: normalized to [0,1] by the image width and height, with the origin at the TOP-LEFT corner of the image and y increasing downward. Note that this is not the
     * convention of Visualizer::COORDINATES_WINDOW_NORMALIZED, whose origin is the bottom-left corner. This is the convention written by RadiationModel::writeImageBoundingBoxes().
     */
    struct BoundingBox {
        //! Integer class identifier of the object enclosed by the box
        uint class_ID = 0;
        //! Center of the box, normalized by the image dimensions, with the origin at the top-left corner of the image
        helios::vec2 center;
        //! Width and height of the box, normalized by the image width and height respectively
        helios::vec2 size;
    };

    //! Read a bounding box annotation file in Ultralytics YOLO format
    /**
     * Every non-blank line must hold exactly five whitespace-separated fields, `class_ID x_center y_center width height`, where the four geometry fields are normalized to [0,1] and are measured from the
     * top-left corner of the image. Blank lines are ignored. Both plain and scientific float notation are accepted. This is the format written by RadiationModel::writeImageBoundingBoxes().
     * \param[in] bbox_file Path to the bounding box annotation file.
     * \return Boxes in the order they appear in the file. Empty if the file contains no boxes.
     */
    [[nodiscard]] static std::vector<BoundingBox> readBoundingBoxFile(const std::string &bbox_file);

    //! Read a file mapping bounding box class IDs to class names
    /**
     * Two line formats are accepted, detected line by line. If the first whitespace-separated token is a non-negative integer and at least one more token follows, the line is read as `class_ID class_name`,
     * which is the format written by RadiationModel::writeImageBoundingBoxes(); the name is the remainder of the line, so names may contain spaces. Otherwise the whole trimmed line is the class name and its
     * class ID is the index of the line among the non-blank lines, counting from zero, which is the standard Ultralytics convention.
     * \param[in] classes_file Path to the class name file.
     * \return Map from class ID to class name.
     * \note A line in the implicit format whose class name begins with a number is indistinguishable from the explicit format and will be read as the explicit format.
     */
    [[nodiscard]] static std::map<uint, std::string> readBoundingBoxClassNames(const std::string &classes_file);

    //! A single segmentation mask read from a COCO JSON annotation file
    /**
     * Each mask holds one or more polygon contours whose vertices are in absolute pixel coordinates measured from the TOP-LEFT corner of the image, with y increasing downward. Note that this differs both
     * from Visualizer::BoundingBox, whose coordinates are normalized to [0,1], and from Visualizer::COORDINATES_WINDOW_NORMALIZED, whose origin is the bottom-left corner. This is the convention written by
     * RadiationModel::writeImageSegmentationMasks().
     */
    struct SegmentationMask {
        //! Integer class identifier of the masked object, taken from the annotation's COCO "category_id"
        uint class_ID = 0;
        //! Name of the class, resolved from the "categories" array of the same file. Empty if the file declares no name for this class ID.
        std::string class_name;
        //! Polygon contours bounding the mask, in absolute pixel coordinates measured from the top-left corner of the image
        std::vector<std::vector<helios::vec2>> polygons;
        //! Width and height in pixels of the image the polygons were authored against, taken from the "images" entry the annotation refers to
        helios::vec2 image_size;
    };

    //! Read a segmentation mask annotation file in COCO JSON format
    /**
     * The file must contain the "images", "annotations" and "categories" arrays of the COCO format. Each annotation contributes one mask, whose polygons come from its "segmentation" field: an array of
     * contours, each a flat list of alternating x and y pixel coordinates. This is the format written by RadiationModel::writeImageSegmentationMasks().
     * \param[in] mask_file Path to the COCO JSON annotation file.
     * \param[in] image_file [optional] Path to the image whose annotations should be read. Only the file name is compared, so a path from a different directory still matches. If this is empty, which is the default, the file must describe exactly one image.
     * \return Masks in the order their annotations appear in the file. Empty if no annotation refers to the selected image.
     * \note Run-length encoded ("counts") segmentations are not supported, because RadiationModel::writeImageSegmentationMasks() never writes them.
     */
    [[nodiscard]] static std::vector<SegmentationMask> readSegmentationMaskFile(const std::string &mask_file, const std::string &image_file = "");

    //! Pseudocolor map tables
    enum Ctable {
        //! "Hot" colormap
        COLORMAP_HOT = 0,
        //! "Cool" colormap
        COLORMAP_COOL = 1,
        //! "Rainbow" colormap
        COLORMAP_RAINBOW = 2,
        //! "Lava" colormap
        COLORMAP_LAVA = 3,
        //! "Parula" colormap
        COLORMAP_PARULA = 4,
        //! "Gray" colormap
        COLORMAP_GRAY = 5,
        //! Custom colormap
        COLORMAP_CUSTOM = 6,
        //! "Lines" colormap with distinct colors
        COLORMAP_LINES = 7
    };

    //! Set camera position
    /**
     * \param[in] cameraPosition (x,y,z) position of the camera, i.e., this is where the actual camera or `eye' is positioned.
     * \param[in] lookAt (x,y,z) position of where the camera is looking at.
     */
    void setCameraPosition(const helios::vec3 &cameraPosition, const helios::vec3 &lookAt);

    //! Set camera position
    /**
     * \param[in] cameraAngle (elevation,azimuth) angle to the camera with respect to the `lookAt' position.
     * \param[in] lookAt (x,y,z) position of where the camera is looking at.
     */
    void setCameraPosition(const helios::SphericalCoord &cameraAngle, const helios::vec3 &lookAt);

    //! Set the camera field of view (angle width) in degrees. Default value is 45 degrees.
    /**
     * \param[in] angle_FOV Angle of camera field of view in degrees.
     */
    void setCameraFieldOfView(float angle_FOV);

    //! Set the direction of the light source
    /**
     * \param[in] direction Vector pointing in the direction of the light source (vector starts at light source and points toward scene.)
     */
    void setLightDirection(const helios::vec3 &direction);

    //! Lighting model to use for shading primitives
    enum LightingModel {
        //! No shading, primitive is colored by its diffuse color
        LIGHTING_NONE = 0,

        //! Phong lighting model is applied to add shading effects to the diffuse color
        LIGHTING_PHONG = 1,

        //! Phong lighting model plus shadowing is applied to add shading effects to the diffuse color
        LIGHTING_PHONG_SHADOWED = 2
    };

    //! Set the lighting model for shading of all primitives
    /**
     * \param[in] lightingmodel Lighting model to be used
     * \sa LightingModel
     */
    void setLightingModel(LightingModel lightingmodel);

    //! Set the light intensity scaling factor
    /**
     * \param[in] lightintensityfactor Scaling factor for light intensity. Default is 1.0
     */
    void setLightIntensityFactor(float lightintensityfactor);

    //! Render primitive colors exactly as they are set in the Context
    /**
     * By default the fragment shader multiplies vertex-interpolated primitive colors by 1.5, which
     * brightens ordinary renders but means the color read back out of the framebuffer is not the
     * color that was set on the primitive. This mode disables that multiplier so that a color
     * written with \ref helios::Context::setPrimitiveColor() is reproduced exactly in the rendered
     * image, which is required when the framebuffer is used to carry data rather than to be looked
     * at -- for example when object ID codes are encoded as RGB values and decoded from the
     * rendered pixels, as the synthetic annotation plug-in does.
     *
     * Note that exact color reproduction additionally requires that no lighting be applied (see
     * \ref setLightingModel() with \ref LIGHTING_NONE, which is the default) and that the
     * Visualizer be constructed with anti-aliasing disabled, since anti-aliasing blends colors at
     * primitive edges and produces pixel values that decode to meaningless IDs.
     *
     * This mode also disables the linear-light rendering pipeline (see \ref enableLinearPipeline()),
     * because tone mapping and sRGB encoding are non-linear transformations of the color channels
     * and would corrupt any data carried in them.
     *
     * \sa disableExactColorMode()
     */
    void enableExactColorMode();

    //! Restore the default brightening of primitive colors
    /**
     * Also restores the linear-light pipeline, which \ref enableExactColorMode() turns off.
     *
     * \sa enableExactColorMode()
     */
    void disableExactColorMode();

    //! Render using a physically-based linear-light pipeline
    /**
     * This is the default. Albedo is decoded from sRGB to linear light before shading, the shaded
     * radiance is multiplied by the exposure (see \ref setExposure()), passed through an ACES
     * filmic tone curve, and re-encoded to sRGB. Highlights roll off smoothly instead of clipping,
     * and mid-tone gradients are correct.
     *
     * The alternative, selected with \ref disableLinearPipeline(), performs the lighting arithmetic
     * directly on sRGB-encoded color values and writes the result straight to the framebuffer. That
     * is not physically meaningful: sRGB values are non-linear in radiance, so summing an ambient
     * and a diffuse term in that space produces mid-tones that are too dark and shadow terminators
     * that are unnaturally abrupt, and bright surfaces clip hard against the 8-bit framebuffer.
     *
     * This method is therefore only needed to restore the default after \ref
     * disableLinearPipeline() or \ref enableExactColorMode() has turned it off.
     *
     * Text and blended image overlays are authored in display space and are deliberately excluded,
     * as is all 2D screen-space geometry.
     *
     * \note This mode is mutually exclusive with \ref enableExactColorMode(), which needs the
     * framebuffer to read back bit-unchanged. Enabling exact color mode turns this off.
     *
     * \sa disableLinearPipeline(), setExposure()
     */
    void enableLinearPipeline();

    //! Disable the linear-light rendering pipeline
    /**
     * Reverts to performing the lighting arithmetic directly on sRGB-encoded values and writing the
     * result straight to the framebuffer, which is how the Visualizer rendered prior to v1.3.83.
     * Use this to reproduce images generated by earlier versions.
     *
     * \sa enableLinearPipeline()
     */
    void disableLinearPipeline();

    //! Set the exposure applied in linear light before tone mapping
    /**
     * Values greater than 1 brighten the image, values less than 1 darken it. Because the tone
     * curve rolls off smoothly, raising exposure recovers highlight detail rather than clipping it.
     *
     * \param[in] exposure Linear exposure multiplier. Must be positive. Default is 1.0.
     * \note Has no effect if the linear-light pipeline has been turned off (see \ref disableLinearPipeline()).
     */
    void setExposure(float exposure);

    //! Get the exposure applied in linear light before tone mapping
    [[nodiscard]] float getExposure() const;

    //! Query whether the linear-light rendering pipeline is enabled
    [[nodiscard]] bool isLinearPipelineEnabled() const;

    //! Phong material reflectance parameters
    /**
     * The shaded radiance is `ambient*A + diffuse*max(0,N.L) + specular*max(0,N.H)^shininess`,
     * where `A` is the hemispheric ambient term (see \ref Visualizer::setAmbientColors()).
     *
     * \sa Visualizer::setPhongMaterial()
     */
    struct PhongMaterial {
        //! Ambient reflectance weight, scaling the hemispheric ambient contribution
        float ambient = 1.0f;
        //! Diffuse (Lambertian) reflectance weight
        float diffuse = 0.8f;
        //! Specular reflectance weight. Zero disables the highlight entirely.
        float specular = 0.2f;
        //! Specular exponent. Larger values give a tighter, glossier highlight; typical range is 4 (matte sheen) to 128 (near-mirror).
        float shininess = 32.f;
    };

    //! Set the Phong material parameters used to shade Context primitives
    /**
     * Controls the relative weights of the ambient, diffuse and specular lighting terms, and the
     * tightness of the specular highlight.
     *
     * The defaults (ambient 1.0, diffuse 0.8, specular 0.2, shininess 32) approximate a matte
     * surface with a slight sheen, which suits foliage. Raising `specular` and `shininess` gives a
     * glossier, wetter-looking leaf; setting `specular` to zero removes the highlight and recovers
     * a purely Lambertian appearance.
     *
     * \param[in] material Phong material parameters.
     * \note Has no effect unless a lighting model other than \ref LIGHTING_NONE is active.
     * \note The specular highlight is most useful in combination with \ref enableLinearPipeline(),
     * which lets bright highlights roll off smoothly rather than clipping.
     * \sa getPhongMaterial(), setAmbientColors()
     */
    void setPhongMaterial(const PhongMaterial &material);

    //! Get the Phong material parameters used to shade Context primitives
    [[nodiscard]] PhongMaterial getPhongMaterial() const;

    //! Set the sky and ground colors used by the hemispheric ambient term
    /**
     * Ambient light is approximated as a sky color arriving from above and a ground-bounce color
     * arriving from below, blended according to the vertical component of the surface normal. This
     * gives surfaces facing away from the light soft directional grounding, rather than the flat
     * uniform fill produced by a single constant ambient color.
     *
     * Defaults are a cool sky (0.5, 0.6, 0.75) and a warm ground bounce (0.35, 0.3, 0.22).
     * Setting both to the same value recovers a constant, non-directional ambient term.
     *
     * \param[in] sky_color Ambient color arriving from above.
     * \param[in] ground_color Ambient color arriving from below.
     * \sa getAmbientSkyColor(), getAmbientGroundColor(), setPhongMaterial()
     */
    void setAmbientColors(const helios::RGBcolor &sky_color, const helios::RGBcolor &ground_color);

    //! Get the sky color used by the hemispheric ambient term
    [[nodiscard]] helios::RGBcolor getAmbientSkyColor() const;

    //! Get the ground-bounce color used by the hemispheric ambient term
    [[nodiscard]] helios::RGBcolor getAmbientGroundColor() const;

    //! Shade surfaces using smooth, interpolated per-vertex normals
    /**
     * Each fragment is shaded using a normal interpolated across the primitive from its vertices,
     * rather than the single geometric normal of the face. On a tessellated curved surface this
     * removes the faceted appearance, so that a stem, fruit or trunk reads as smoothly curved
     * rather than as a series of flat panels.
     *
     * This only changes the appearance of geometry that actually supplies distinct vertex normals.
     * Primitives added without them carry the face normal replicated across every vertex, so they
     * shade identically whether smooth shading is enabled or not. Vertex normals are supplied by
     * \ref helios::Context "Context" `Sphere`, `Tube` and `Cone` objects, whose normals are
     * evaluated exactly from the shape's own definition, and by `Polymesh` objects that retain
     * them -- for example a mesh loaded from an OBJ or PLY file carrying vertex normals.
     *
     * Smooth shading is the default. Note that on alpha-masked cutouts such as leaf textures it can
     * look worse than flat shading, because the interpolated normal no longer agrees with the
     * visible silhouette; \ref disableSmoothShading() is the remedy.
     *
     * \sa disableSmoothShading(), isSmoothShadingEnabled()
     */
    void enableSmoothShading();

    //! Shade surfaces using the flat geometric normal of each face
    /**
     * \sa enableSmoothShading()
     */
    void disableSmoothShading();

    //! Query whether smooth shading is enabled
    [[nodiscard]] bool isSmoothShadingEnabled() const;

    //! Set the background color for the visualizer window
    /**
     * \param[in] color Background color
     */
    void setBackgroundColor(const helios::RGBcolor &color);

    //! Set the background to be transparent
    /**
     * This method enables transparent background mode. When rendering to the screen (via plotUpdate() or plotInteractive()),
     * a checkerboard pattern is displayed to indicate transparency. When saving to a PNG file (via printWindow() with
     * image_format="png"), the background will have true alpha channel transparency.
     * \note This only affects PNG output. JPEG output will still have an opaque background.
     * \sa setBackgroundColor()
     */
    void setBackgroundTransparent();

    //! Set a custom background image
    /**
     * This method sets a custom texture image as the background. The image will be stretched to fill the window
     * while maintaining the aspect ratio of the window to avoid distortion.
     * \param[in] texture_file Path to the texture image file to use as background
     * \note Supported formats include JPEG and PNG files
     * \sa setBackgroundColor(), setBackgroundTransparent()
     */
    void setBackgroundImage(const char *texture_file);

    //! Set a sky sphere texture as background
    /**
     * \brief Creates a dynamically scaling full sky sphere that keeps the camera always inside using shader-based transformations
     * \param[in] texture_file Path to the spherical/equirectangular texture image file. If nullptr, uses default sky texture (plugins/visualizer/textures/SkyDome_clouds.jpg)
     * \param[in] Ndivisions Number of divisions for sphere tessellation (default: 50)
     * \note This method creates a complete sphere (not just a hemisphere) that surrounds the camera in all directions
     * \note Uses shader transformations to make the sky appear infinitely distant and always contain the camera
     * \note Can be toggled with other background methods (setBackgroundColor, setBackgroundImage, setBackgroundTransparent)
     * \sa setBackgroundColor(), setBackgroundImage(), setBackgroundTransparent()
     */
    void setBackgroundSkyTexture(const char *texture_file = nullptr, uint Ndivisions = 50);

    //! Add a rectangle by giving the coordinates of its center
    /**
     * \param[in] center (x,y,z) location of the rectangle center
     * \param[in] size Size in the x- and y-directions
     * \param[in] rotation spherical rotation angle (elevation,azimuth)
     * \param[in] color R-G-B color of the rectangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByCenter(const helios::vec3 &center, const helios::vec2 &size, const helios::SphericalCoord &rotation, const helios::RGBcolor &color, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its center
    /**
     * \param[in] center (x,y,z) location of the rectangle center
     * \param[in] size Size in the x- and y-directions
     * \param[in] rotation spherical rotation angle (elevation,azimuth)
     * \param[in] color R-G-B-A color of the rectangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByCenter(const helios::vec3 &center, const helios::vec2 &size, const helios::SphericalCoord &rotation, const helios::RGBAcolor &color, CoordinateSystem coordFlag);

    //! Add a texture mapped rectangle by giving the coordinates of its center
    /**
     * \param[in] center (x,y,z) location of the rectangle center
     * \param[in] size Size in the x- and y-directions
     * \param[in] rotation spherical rotation angle (elevation,azimuth)
     * \param[in] texture_file File corresponding to the JPEG image to be used as a texture map
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByCenter(const helios::vec3 &center, const helios::vec2 &size, const helios::SphericalCoord &rotation, const char *texture_file, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its center - rectangle is colored by and RGB color value but is masked by the alpha channel of a PNG image file
    /**
     * \param[in] center (x,y,z) location of the rectangle center
     * \param[in] size Size in the x- and y-directions
     * \param[in] rotation spherical rotation angle (elevation,azimuth)
     * \param[in] color R-G-B color of the rectangle
     * \param[in] texture_file File corresponding to the JPEG image to be used as a texture map
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByCenter(const helios::vec3 &center, const helios::vec2 &size, const helios::SphericalCoord &rotation, const helios::RGBcolor &color, const char *texture_file, CoordinateSystem coordFlag);

    //! Add a texture masked rectangle by giving the coordinates of its center
    /**
     * \param[in] center (x,y,z) location of the rectangle center
     * \param[in] size Size in the x- and y-directions
     * \param[in] rotation spherical rotation angle (elevation,azimuth)
     * \param[in] color R-G-B color of the rectangle
     * \param[in] glyph Pixel map of true/false values for a transparency mask
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByCenter(const helios::vec3 &center, const helios::vec2 &size, const helios::SphericalCoord &rotation, const helios::RGBcolor &color, const Glyph *glyph, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] color R-G-B color of the rectangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const helios::RGBcolor &color, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] color R-G-B-A color of the rectangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const helios::RGBAcolor &color, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] texture_file File corresponding to the JPEG image to be used as a texture map
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const char *texture_file, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices and color by texture map
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] texture_file File corresponding to the JPEG image to be used as a texture map
     * \param[in] uvs u-v coordinates for rectangle vertices
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const char *texture_file, const std::vector<helios::vec2> &uvs, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices and mask by texture map transparency channel, but color by R-G-B value
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] texture_file File corresponding to the JPEG image to be used as a texture map
     * \param[in] uvs u-v coordinates for rectangle vertices
     * \param[in] color R-G-B color of the rectangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const helios::RGBcolor &color, const char *texture_file, const std::vector<helios::vec2> &uvs, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices - rectangle is colored by an RGB color value but is masked by the alpha channel of a PNG image file
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] color R-G-B color of the rectangle
     * \param[in] texture_file File corresponding to the JPEG image to be used as a texture map
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const helios::RGBcolor &color, const char *texture_file, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] color R-G-B color of the glyph
     * \param[in] glyph Glyph object used to render rectangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const helios::RGBcolor &color, const Glyph *glyph, CoordinateSystem coordFlag);

    //! Add a rectangle by giving the coordinates of its four vertices
    /**
     * \param[in] vertices (x,y,z) coordinates of four vertices
     * \param[in] color R-G-B-A color of the glyph
     * \param[in] glyph Glyph object used to render rectangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addRectangleByVertices(const std::vector<helios::vec3> &vertices, const helios::RGBAcolor &color, const Glyph *glyph, CoordinateSystem coordFlag);

    //! Add a triangle by giving the coordinates of its three vertices
    /**
     * \param[in] vertex0 (x,y,z) location of first vertex
     * \param[in] vertex1 (x,y,z) location of first vertex
     * \param[in] vertex2 (x,y,z) location of first vertex
     * \param[in] color R-G-B color of the triangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addTriangle(const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2, const helios::RGBcolor &color, CoordinateSystem coordFlag);

    //! Add a triangle by giving the coordinates of its three vertices
    /**
     * \param[in] vertex0 (x,y,z) location of first vertex
     * \param[in] vertex1 (x,y,z) location of first vertex
     * \param[in] vertex2 (x,y,z) location of first vertex
     * \param[in] color R-G-B-A color of the triangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addTriangle(const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2, const helios::RGBAcolor &color, CoordinateSystem coordFlag);

    //! Add a triangle by giving the coordinates of its three vertices and color by texture map
    /**
     * \param[in] vertex0 (x,y,z) location of first vertex
     * \param[in] vertex1 (x,y,z) location of first vertex
     * \param[in] vertex2 (x,y,z) location of first vertex
     * \param[in] texture_file File corresponding to the image to be used as a texture map
     * \param[in] uv0 u-v texture coordinates of vertex0
     * \param[in] uv1 u-v texture coordinates of vertex1
     * \param[in] uv2 u-v texture coordinates of vertex2
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addTriangle(const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2, const char *texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2, CoordinateSystem coordFlag);

    //! Add a triangle by giving the coordinates of its three vertices and color by a constant color, but mask using transparency channel of texture map
    /**
     * \param[in] vertex0 (x,y,z) location of first vertex
     * \param[in] vertex1 (x,y,z) location of first vertex
     * \param[in] vertex2 (x,y,z) location of first vertex
     * \param[in] texture_file File corresponding to the image to be used as a texture map
     * \param[in] uv0 u-v texture coordinates of vertex0
     * \param[in] uv1 u-v texture coordinates of vertex1
     * \param[in] uv2 u-v texture coordinates of vertex2
     * \param[in] color R-G-B-A color of the triangle
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addTriangle(const helios::vec3 &vertex0, const helios::vec3 &vertex1, const helios::vec3 &vertex2, const char *texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2, const helios::RGBAcolor &color,
                       CoordinateSystem coordFlag);

    //! Add a voxel by giving the coordinates of its center
    /**
     * \param[in] center (x,y,z) location of the voxel center
     * \param[in] size Size in the x-, y- and z-directions
     * \param[in] rotation Spherical rotation angle (elevation,azimuth)
     * \param[in] color R-G-B color of the voxel
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    std::vector<size_t> addVoxelByCenter(const helios::vec3 &center, const helios::vec3 &size, const helios::SphericalCoord &rotation, const helios::RGBcolor &color, CoordinateSystem coordFlag);

    //! Add a voxel by giving the coordinates of its center
    /**
     * \param[in] center (x,y,z) location of the voxel center
     * \param[in] size Size in the x-, y- and z-directions
     * \param[in] rotation Spherical rotation angle (elevation,azimuth)
     * \param[in] color R-G-B-A color of the voxel
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    std::vector<size_t> addVoxelByCenter(const helios::vec3 &center, const helios::vec3 &size, const helios::SphericalCoord &rotation, const helios::RGBAcolor &color, CoordinateSystem coordFlag);

    //! Add Lines by giving the coordinates of points along the Lines
    /**
     * \param[in] start (x,y,z) coordinates of line starting position
     * \param[in] end (x,y,z) coordinates of line ending position
     * \param[in] color R-G-B color of the line
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addLine(const helios::vec3 &start, const helios::vec3 &end, const helios::RGBcolor &color, CoordinateSystem coordinate_system);

    //! Add Lines by giving the coordinates of points along the Lines
    /**
     * \param[in] start (x,y,z) coordinates of line starting position
     * \param[in] end (x,y,z) coordinates of line ending position
     * \param[in] color R-G-B-A color of the line
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addLine(const helios::vec3 &start, const helios::vec3 &end, const helios::RGBAcolor &color, CoordinateSystem coordFlag);

    //! Add Lines by giving the coordinates of points along the Lines with custom line width
    /**
     * \param[in] start (x,y,z) coordinates of line starting position
     * \param[in] end (x,y,z) coordinates of line ending position
     * \param[in] color R-G-B color of the line
     * \param[in] line_width Width of the line in pixels (rendered using geometry shaders for cross-platform wide line support)
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     * \note Line widths are rendered using geometry shaders which expand line primitives into screen-aligned quads, providing consistent wide line rendering across all platforms including macOS.
     */
    size_t addLine(const helios::vec3 &start, const helios::vec3 &end, const helios::RGBcolor &color, float line_width, CoordinateSystem coordinate_system);

    //! Add Lines by giving the coordinates of points along the Lines with custom line width
    /**
     * \param[in] start (x,y,z) coordinates of line starting position
     * \param[in] end (x,y,z) coordinates of line ending position
     * \param[in] color R-G-B-A color of the line
     * \param[in] line_width Width of the line in pixels (rendered using geometry shaders for cross-platform wide line support)
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     * \note Line widths are rendered using geometry shaders which expand line primitives into screen-aligned quads, providing consistent wide line rendering across all platforms including macOS.
     */
    size_t addLine(const helios::vec3 &start, const helios::vec3 &end, const helios::RGBAcolor &color, float line_width, CoordinateSystem coordFlag);

    //! Add a point by giving its coordinates and size
    /**
     * \param[in] position (x,y,z) coordinates of Point
     * \param[in] color R-G-B color of the Point
     * \param[in] pointsize Size of the point in font points
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addPoint(const helios::vec3 &position, const helios::RGBcolor &color, float pointsize, CoordinateSystem coordinate_system);

    //! Add a point by giving its coordinates and size
    /**
     * \param[in] position (x,y,z) coordinates of Point
     * \param[in] color R-G-B-A color of the Point
     * \param[in] pointsize Size of the point in font points
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addPoint(const helios::vec3 &position, const helios::RGBAcolor &color, float pointsize, CoordinateSystem coordinate_system);

    //! Add a sphere by giving the radius and center
    /**
     * \param[in] radius Radius of the sphere
     * \param[in] center (x,y,z) location of sphere center
     * \param[in] Ndivisions Number of discrete divisions in making sphere
     * \param[in] color R-G-B color of the sphere
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    std::vector<size_t> addSphereByCenter(float radius, const helios::vec3 &center, uint Ndivisions, const helios::RGBcolor &color, CoordinateSystem coordinate_system);

    //! Add a sphere by giving the radius and center
    /**
     * \param[in] radius Radius of the sphere
     * \param[in] center (x,y,z) location of sphere center
     * \param[in] Ndivisions Number of discrete divisions in making sphere
     * \param[in] color R-G-B-A color of the sphere
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    std::vector<size_t> addSphereByCenter(float radius, const helios::vec3 &center, uint Ndivisions, const helios::RGBAcolor &color, CoordinateSystem coordinate_system);

    //! Add a Sky Dome, which is a hemispherical dome colored by a sky texture map
    /**
     * \deprecated This method is deprecated and will be removed in a future version. Use setBackgroundSkyTexture() instead, which provides a more robust sky rendering solution that dynamically scales with camera movement.
     * \param[in] radius Radius of the dome
     * \param[in] center (x,y,z) location of dome center
     * \param[in] Ndivisions Number of discrete divisions in making hemisphere
     * \param[in] texture_file Name of the texture map file
     * \sa setBackgroundSkyTexture()
     */
    [[deprecated]]
    std::vector<size_t> addSkyDomeByCenter(float radius, const helios::vec3 &center, uint Ndivisions, const char *texture_file);

    //! Add a Sky Dome, which is a hemispherical dome colored by a sky texture map
    /** \note This function has been deprecated, as layers are no longer supported. */
    [[deprecated]]
    void addSkyDomeByCenter(float radius, const helios::vec3 &center, uint Ndivisions, const char *texture_file, int layer);

    //! Add a text box by giving the coordinates of its center
    /**
     * \param[in] textstring String of text to display
     * \param[in] center (x,y,z) location of the text box center
     * \param[in] rotation Spherical rotation angle in radians (elevation,azimuth)
     * \param[in] fontcolor Color of the font
     * \param[in] fontsize Size of the text font in points
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    std::vector<size_t> addTextboxByCenter(const char *textstring, const helios::vec3 &center, const helios::SphericalCoord &rotation, const helios::RGBcolor &fontcolor, uint fontsize, const char *fontname, CoordinateSystem coordinate_system);

    //! Measure the rendered size of a text string without adding it to the visualizer
    /**
     * Returns the extent that \ref addTextboxByCenter() would occupy for the same string, font and font size, in window-normalized units. The width is the sum of the glyph advances, so it includes the side
     * bearings; the height is that of the tallest glyph in the string, so it depends on which characters the string contains. The '_' and '^' subscript and superscript markers are handled exactly as
     * \ref addTextboxByCenter() handles them: they occupy no width themselves and halve the size of the character that follows.
     * \param[in] textstring Text to be measured.
     * \param[in] fontsize Size of the text font in points.
     * \param[in] fontname Name of a font in the plugins/visualizer/fonts directory, for example "OpenSans-Regular".
     * \return Width and height of the text in window-normalized units.
     * \note The result depends on the current framebuffer dimensions and DPI scale, and therefore changes when the window is resized.
     */
    [[nodiscard]] helios::vec2 getTextboxSize(const char *textstring, uint fontsize, const char *fontname) const;

    //! Removes the geometry with the specified ID from the visualizer.
    /**
     * \param[in] geometry_id The unique identifier of the geometry to delete.
     */
    void deleteGeometry(size_t geometry_id);

    //! Get the vertices of a geometry primitive
    /**
     * \param[in] geometry_id The unique identifier of the geometry
     * \return Vector of vertices in the same coordinate system they were added with
     * \note For COORDINATES_WINDOW_NORMALIZED, returns vertices in [0,1] range. For COORDINATES_CARTESIAN, returns vertices in world coordinates.
     */
    [[nodiscard]] std::vector<helios::vec3> getGeometryVertices(size_t geometry_id) const;

    //! Set the vertices of a geometry primitive
    /**
     * \param[in] geometry_id The unique identifier of the geometry
     * \param[in] vertices New vertex positions in the same coordinate system the geometry was created with
     * \note For COORDINATES_WINDOW_NORMALIZED, provide vertices in [0,1] range. For COORDINATES_CARTESIAN, provide vertices in world coordinates.
     */
    void setGeometryVertices(size_t geometry_id, const std::vector<helios::vec3> &vertices);

    //! Add a coordinate axis with at the origin with unit length
    void addCoordinateAxes();

    //! Add a coordinate axis
    /**
     * \param[in] origin (x,y,z) location of the coordinate axes orign
     * \param[in] length length of coordinate axis lines from origin in each direction
     * \param[in] sign either "both" or "positive" should the axes be drawn in both positive and negative directions or just positive
     */
    void addCoordinateAxes(const helios::vec3 &origin, const helios::vec3 &length, const std::string &sign);

    //! Remove coordinate axes (if created with Visualizer::addCoordinateAxes)
    void disableCoordinateAxes();

    //! Add a coordinate axis
    /**
     * \param[in] center (x,y,z) location of the center of the grid
     * \param[in] size size of the grid in each direction
     * \param[in] subdiv number of grid subdivisions in each direction
     */
    void addGridWireFrame(const helios::vec3 &center, const helios::vec3 &size, const helios::int3 &subdiv);

    //! Enable the colorbar
    void enableColorbar();

    //! Disable the colorbar
    void disableColorbar();

    //! Set the position of the colorbar in normalized window coordinates (0-1)
    /**
     * \param[in] position Position of the colorbar in normalized window coordinates
     */
    void setColorbarPosition(helios::vec3 position);

    //! Set the size of the colorbar in normalized window units (0-1)
    /**
     * \param[in] size Size of the colorbar in normalized window units (0-1)
     */
    void setColorbarSize(helios::vec2 size);

    //! Set the range of the Colorbar
    /**
     * \param[in] cmin Minimum value
     * \param[in] cmax Maximum value
     * \note The command is ignored if cmin is greater than cmax.
     */
    void setColorbarRange(float cmin, float cmax);

    //! Set the values in the colorbar where ticks and labels should be placed
    /**
     * \param[in] ticks Vector of values corresponding to ticks
        \note If tick values are outside of the colorbar range (see setColorbarRange()), the colorbar range will be automatically expanded to fit the tick values, and a warning is issued if messages are enabled. Because the colormap limits follow the colorbar range, this changes the colors shown as well as the labels. To keep an explicit range authoritative, call setColorbarRange() after setColorbarTicks().
    */
    void setColorbarTicks(const std::vector<float> &ticks);

    //! Set the title of the Colorbar
    /**
     * \param[in] title Colorbar title
     */
    void setColorbarTitle(const char *title);

    //! Set the RGB color of the colorbar text
    /**
     * \param[in] color Font color
     */
    void setColorbarFontColor(helios::RGBcolor color);

    //! Set the font size of the colorbar text
    /**
     * \param[in] font_size Font size
     */
    void setColorbarFontSize(uint font_size);

    //! Set the colormap used in Colorbar/visualization based on pre-defined colormaps
    /**
     * \param[in] colormap_name Name of a colormap.
     * \note Valid colormaps are "COLORMAP_HOT", "COLORMAP_COOL", "COLORMAP_LAVA", "COLORMAP_RAINBOW", "COLORMAP_PARULA", "COLORMAP_GRAY", "COLORMAP_LINES".
     */
    void setColormap(Ctable colormap_name);

    //! Set the colormap used in Colorbar/visualization based on a custom colormap
    /**
     * \param[in] colors Vector of colors defining control points on the colormap.
     * \param[in] divisions Vector of values defining the normalized coordinates of each color control point on the colormap.
     */
    void setColormap(const std::vector<helios::RGBcolor> &colors, const std::vector<float> &divisions);

    //! Get the current colormap used in Colorbar/visualization
    [[nodiscard]] Colormap getCurrentColormap() const;

    //! Add all geometry from the Context to the visualizer
    /**
     * \param[in] context_ptr Pointer to the simulation context
     */
    void buildContextGeometry(helios::Context *context_ptr);

    //! Add select geometry from the Context to the visualizer by their UUIDs
    /**
     * \param[in] context_ptr Pointer to the simulation context
     * \param[in] UUIDs UUIDs of Context primitives to be added to the visualizer
     */
    void buildContextGeometry(helios::Context *context_ptr, const std::vector<uint> &UUIDs);

    //! Updates the colors of context primitives based on current visualization settings.
    /**
     * This method processes all primitive geometries within the context, applies appropriate color mapping
     * based on configured data or object data, updates their color values, and handles internal logic for
     * colormap range adjustments and primitive existence checks.
     */
    void updateContextPrimitiveColors();

    //! Color primitives from Context by color mapping their `Primitive Data'
    /**
     * \param[in] data_name Name of `Primitive Data'
     * \note If the data value does not exist for a certain primitive, a value of 0 is assumed.
     */
    void colorContextPrimitivesByData(const char *data_name);

    //! Color primitives from Context by color mapping their `Primitive Data'
    /**
     * \param[in] data_name Name of `Primitive Data'
     * \param[in] UUIDs UUID's of primitives to be colored by data
     * \note If the data value does not exist for a certain primitive, a value of 0 is assumed.
     */
    void colorContextPrimitivesByData(const char *data_name, const std::vector<uint> &UUIDs);

    //! Color primitives from Context by color mapping their `Object Data'
    /**
     * \param[in] data_name Name of `Object Data'
     * \note If the data value does not exist for a certain primitive, a value of 0 is assumed.
     */
    void colorContextPrimitivesByObjectData(const char *data_name);

    //! Color primitives from Context by color mapping their `Object Data'
    /**
     * \param[in] data_name Name of `Object Data'
     * \param[in] ObjIDs Object ID's of primitives to be colored by object data
     * \note If the data value does not exist for a certain primitive, a value of 0 is assumed.
     */
    void colorContextPrimitivesByObjectData(const char *data_name, const std::vector<uint> &ObjIDs);

    //! Color primitives from Context with a random color
    /**
     * \param[in] UUIDs Primitive UUIDs to color randomly
     * \note Useful for visualizing individual primitives that are part of compound objects
     */
    void colorContextPrimitivesRandomly(const std::vector<uint> &UUIDs);

    //! Color primitives from Context with a random color
    /**
     * \note Useful for visualizing individual primitives that are part of compound objects
     */
    void colorContextPrimitivesRandomly();

    //! Color objects from Context with a random color
    /**
     * \note Useful for visualizing individual objects
     */
    void colorContextObjectsRandomly(const std::vector<uint> &ObjIDs);

    //! Color objects from Context with a random color
    /**
     * \note Useful for visualizing individual objects
     */
    void colorContextObjectsRandomly();

    //! Make Helios logo watermark invisible
    void hideWatermark();

    //! Make Helios logo watermark visible
    void showWatermark();

    //! Update watermark geometry to match current window size
    void updateWatermark();

    //! Make navigation gizmo (coordinate axes indicator) invisible
    void hideNavigationGizmo();

    //! Make navigation gizmo (coordinate axes indicator) visible
    void showNavigationGizmo();

    //! Handle mouse click for navigation gizmo interaction
    /**
     * \param[in] screen_x Mouse x-coordinate in screen space
     * \param[in] screen_y Mouse y-coordinate in screen space
     */
    void handleGizmoClick(double screen_x, double screen_y);

    //! Handle mouse hover for navigation gizmo interaction
    /**
     * \param[in] screen_x Mouse x-coordinate in screen space
     * \param[in] screen_y Mouse y-coordinate in screen space
     */
    void handleGizmoHover(double screen_x, double screen_y);

    //! Plot current geometry into an interactive graphics window
    std::vector<helios::vec3> plotInteractive();

    //! Run one rendering loop from plotInteractive()
    /**
     * This is the body of plotInteractive()'s render loop, exposed so that an external loop can
     * drive rendering itself. Any geometry pending upload is transferred to the GPU before
     * rendering, but unlike plotUpdate() the Context geometry is not rebuilt: call
     * buildContextGeometry() or plotUpdate() if primitives have been added to or changed in the
     * Context since the last render.
     * \param[in] getKeystrokes If false, do not update visualization with input keystrokes.
     */
    void plotOnce(bool getKeystrokes);

    //! Plot the depth map (distance from camera to nearest object)
    /**
     * The resulting image is normalized depth, where white = closest and black = farthest.
     */
    void plotDepthMap();

    //! Update the graphics window based on current geometry, then continue the program
    void plotUpdate();

    //! Update the graphics window based on current geometry, then continue the program, with the option not to display the graphic window
    /** If running a large number of renderings, or running remotely, it can be desirable to not open the graphic window.
     * \param[in] hide_window If false, do not display the graphic window.
     */
    void plotUpdate(bool hide_window);

    //! Print the current graphics window to a JPEG image file. File will be given a default filename and saved to the current directory from which the executable was run.
    void printWindow();

    //! Print the current graphics window to an image file
    /**
     * \param[in] outfile Path to file where image should be saved.
     * \param[in] image_format Format of the output image: "jpeg" (default) or "png".
     * \note If using PNG format with transparent background mode (setBackgroundTransparent()), the output will have alpha channel transparency.
     * \note If outfile does not have an appropriate extension, it will be appended based on image_format.
     */
    void printWindow(const char *outfile, const std::string &image_format = "jpeg");

    /**
     * \brief Displays an image using the provided pixel data and dimensions.
     *
     * Note that this function clears any existing geometry in the visualizer.
     *
     * \param[in] pixel_data The pixel data of the image. Each pixel requires 4 components (RGBA), and the vector size should be 4 * width_pixels * height_pixels.
     * \param[in] width_pixels The width of the image in pixels.
     * \param[in] height_pixels The height of the image in pixels.
     *
     * \note The function assumes the pixel data has a length consistent with the resolution specified by width_pixels and height_pixels.
     */
    void displayImage(const std::vector<unsigned char> &pixel_data, uint width_pixels, uint height_pixels);

    /**
     * \brief Displays an image file in the visualizer.
     *
     * Note that this function clears any existing geometry in the visualizer.
     *
     * \param[in] file_name Path to the image file to display.
     */
    void displayImage(const std::string &file_name);

    //! Display an image file in the visualizer with bounding boxes overlaid
    /**
     * Loads and displays the image exactly as \ref displayImage( const std::string & ) does, then overlays every box in `bbox_file` as a colored outline with the class name drawn on a filled chip inside the
     * box's top-left corner. Boxes are colored by class ID from a fixed palette of seven colors, so classes whose IDs differ by a multiple of seven share a color.
     *
     * \param[in] image_file Path to the image file (JPEG or PNG) to display.
     * \param[in] bbox_file Path to the bounding box annotation file for this image. See \ref readBoundingBoxFile().
     * \param[in] classes_file [optional] Path to the class name file. See \ref readBoundingBoxClassNames(). If this is empty, which is the default, a file named "classes.txt" in the same directory as `bbox_file` is used when one exists; when none exists, boxes are labeled with their numeric class ID.
     * \param[in] line_width [optional] Width of the box outlines in screen pixels. Default is 2.
     * \param[in] fontsize [optional] Size of the class label font in points. Default is 12.
     * \note As with \ref displayImage(), this function clears any existing geometry and does not return until the window is closed.
     */
    void displayImageWithBoundingBoxes(const std::string &image_file, const std::string &bbox_file, const std::string &classes_file = "", float line_width = 2.f, uint fontsize = 12);

    //! Display an image file in the visualizer with segmentation masks overlaid
    /**
     * Loads and displays the image exactly as \ref displayImage( const std::string & ) does, then overlays every mask in `mask_file` as a translucent filled polygon with a solid outline and the class name
     * drawn on a filled chip. Masks are colored by their position in the file from a fixed palette of seven colors, so each mask is colored independently of its class and two touching objects of the same
     * class remain distinguishable.
     *
     * \param[in] image_file Path to the image file (JPEG or PNG) to display.
     * \param[in] mask_file Path to the COCO JSON segmentation mask file for this image. See \ref readSegmentationMaskFile().
     * \param[in] fill_opacity [optional] Opacity of the translucent polygon fill, between 0 and 1. Default is 0.4. A value of 0 draws the outline without a fill.
     * \param[in] line_width [optional] Width of the mask outlines in screen pixels. Default is 2.
     * \param[in] fontsize [optional] Size of the class label font in points. Default is 12.
     * \param[in] show_labels [optional] Whether to draw the class name chip on each mask. Default is true. Pass false to see the masks alone, which is useful when many masks overlap and their chips would cover the image.
     * \note As with \ref displayImage(), this function clears any existing geometry and does not return until the window is closed.
     * \note The fill is computed by an even-odd scanline fill in image pixel space, so it is correct even for a contour that crosses itself. The contours written by RadiationModel::writeImageSegmentationMasks() are traced around a pixel mask and are routinely not simple polygons, because a traced boundary crosses itself wherever the mask narrows to a one-pixel neck.
     */
    void displayImageWithSegmentationMasks(const std::string &image_file, const std::string &mask_file, float fill_opacity = 0.4f, float line_width = 2.f, uint fontsize = 12, bool show_labels = true);

    //! Get R-G-B pixel data in the current display window
    /**
     * \param[out] buffer Pixel data. The data is stored as r-g-b * column * row. So indices (0,1,2) would be the RGB values for row 0 and column 0, indices (3,4,5) would be RGB values for row 0 and column 1, and so on.
     * \note The buffer must hold `3*width*height` elements, where width and height come from \ref getFramebufferSize() — **not** from \ref getWindowSize() and not from the dimensions passed to the constructor. On a high-DPI (Retina) display the framebuffer is larger than the window, typically by a factor of two per axis, so sizing the buffer from the window dimensions overflows it. Prefer the std::vector overload, which allocates correctly on the caller's behalf.
     */
    void getWindowPixelsRGB(uint *buffer) const;

    //! Get R-G-B pixel data in the current display window
    /**
     * Resizes the vector to the current framebuffer dimensions, so it cannot be undersized by the caller.
     * \param[out] pixel_data Pixel data, stored as r-g-b * column * row.
     * \param[out] width_pixels Width of the returned image in pixels
     * \param[out] height_pixels Height of the returned image in pixels
     */
    void getWindowPixelsRGB(std::vector<uint> &pixel_data, uint &width_pixels, uint &height_pixels) const;

    //! Get depth buffer data for the current display window
    /**
     * \param[out] buffer Distance to nearest object from the camera location. The buffer must hold `width*height` elements as reported by \ref getWindowSize(). Note this differs from \ref getWindowPixelsRGB(), which is sized from the framebuffer rather than the window.
     */
    [[deprecated]]
    void getDepthMap(float *buffer);

    void getDepthMap(std::vector<float> &depth_pixels, uint &width_pixels, uint &height_pixels);

    //! Get the size of the display window in pixels
    /**
     * \param[out] width Width of the display window in pixels
     * \param[out] height Height of the display window in pixels
     */
    void getWindowSize(uint &width, uint &height) const;

    //! Get the size of the framebuffer in pixels
    /**
     * \param[out] width Width of the framebuffer in pixels
     * \param[out] height Height of the framebuffer in pixels
     */
    void getFramebufferSize(uint &width, uint &height) const;

    //! Clear all geometry previously added to the visualizer
    void clearGeometry();

    //! Clear all Context geometry previously added to the visualizer
    void clearContextGeometry();

    //! Close the graphics window
    void closeWindow() const;

    /**
     * \brief Retrieves the background color of the visualizer.
     *
     * \return The current background color as an RGBcolor object.
     */
    [[nodiscard]] helios::RGBcolor getBackgroundColor() const;

    /**
     * \brief Retrieves the current camera position.
     *
     * \return A vector containing the camera look-at center and the camera eye location as two elements of type helios::vec3.
     */
    [[nodiscard]] std::vector<helios::vec3> getCameraPosition() const;

    /**
     * \brief Clears the primitive colors based on primitive data from a previous call to colorContextPrimitivesByData() or colorContextPrimitivesByObjectData().
     */
    void clearColor();

    /**
     * \brief Retrieves the window associated with the Visualizer.
     *
     * \return Pointer to the window object.
     */
    [[nodiscard]] void *getWindow() const;

    /**
     * \brief Calculates the perspective transformation matrix for mapping between two quadrilaterals.
     *
     * \return 4x4 perspective transformation matrix.
     */
    [[nodiscard]] glm::mat4 getPerspectiveTransformationMatrix() const;

    //! Point cloud culling configuration methods
    /**
     * \brief Enable or disable point cloud culling optimization
     * \param[in] enabled True to enable culling, false to disable
     */
    void setPointCullingEnabled(bool enabled);

    /**
     * \brief Set the minimum number of points required to trigger culling
     * \param[in] threshold Point count threshold for enabling culling
     */
    void setPointCullingThreshold(size_t threshold);

    /**
     * \brief Set the maximum rendering distance for points
     * \param[in] distance Maximum distance in world units (0 = auto-calculate)
     */
    void setPointMaxRenderDistance(float distance);

    /**
     * \brief Set the level-of-detail factor for distance-based culling
     * \param[in] factor LOD factor (higher values = more aggressive culling)
     */
    void setPointLODFactor(float factor);

    /**
     * \brief Get point cloud rendering performance metrics
     * \param[out] total_points Total number of points in the scene
     * \param[out] rendered_points Number of points actually rendered after culling
     * \param[out] culling_time_ms Time spent on culling in milliseconds
     */
    void getPointRenderingMetrics(size_t &total_points, size_t &rendered_points, float &culling_time_ms) const;

private:
    //! Helper function to round a value to a "nice" number (1, 2, or 5 times a power of 10)
    /**
     * \param[in] value The value to round
     * \param[in] round If true, round to nearest nice number; if false, round up
     * \return The rounded "nice" number
     */
    static double niceNumber(double value, bool round);

    //! Helper function to format a tick label with appropriate precision
    /**
     * \param[in] value The tick value to format
     * \param[in] spacing The spacing between ticks
     * \param[in] isIntegerData Whether the data represents integer values
     * \return The formatted label string
     */
    static std::string formatTickLabel(double value, double spacing, bool isIntegerData);

    //! Generate optimal tick values using nice numbers algorithm
    /**
     * Tick bounds are extended outward to the next "nice" number past the data, which is the
     * correct behavior for general axis labeling. For a colorbar, whose ends are fixed at the
     * colormap limits, use generateColorbarTicks() instead.
     * \param[in] dataMin Minimum data value
     * \param[in] dataMax Maximum data value
     * \param[in] isIntegerData Whether the data represents integer values
     * \param[in] targetTicks Target number of ticks (default 5)
     * \return Vector of tick values
     */
    static std::vector<float> generateNiceTicks(float dataMin, float dataMax, bool isIntegerData, int targetTicks = 5);

    //! Generate "nice" tick values confined to the colorbar range [cmin, cmax]
    /**
     * Unlike generateNiceTicks(), which extends outward past the data for true axis semantics,
     * every returned tick lies within [cmin, cmax] because a colorbar has hard ends. At least two
     * ticks are returned whenever cmax > cmin and both are finite, falling back to progressively
     * finer "nice" spacings and finally to the range endpoints if no nice grid fits.
     *
     * This is called from the rendering path, so degenerate input (non-finite limits, or a range
     * that is empty or narrower than 1e-10) returns a single-tick vector rather than throwing.
     *
     * \param[in] cmin Lower colorbar limit
     * \param[in] cmax Upper colorbar limit
     * \param[in] isIntegerData Whether the data represents integer values
     * \param[in] targetTicks Target number of ticks
     * \param[out] tick_spacing_out If non-null, receives the spacing used to generate the ticks. This is the generating spacing, not one derived from the returned values, so it remains correct when the endpoint fallback produces non-uniform ticks.
     * \return Vector of tick values, all within [cmin, cmax]
     */
    static std::vector<float> generateColorbarTicks(float cmin, float cmax, bool isIntegerData, int targetTicks, double *tick_spacing_out = nullptr);

    /**
     * \brief Retrieves the size of the framebuffer.
     *
     * \return A vector containing the width and height of the framebuffer.
     */
    [[nodiscard]] std::vector<uint> getFrameBufferSize() const;

    //! Read pixels from offscreen framebuffer for internal use by printWindow
    std::vector<helios::RGBcolor> readOffscreenPixels() const;

    //! Read RGBA pixels from offscreen framebuffer for internal use by printWindow (with optional transparency)
    /**
     * \param[in] read_alpha If true, reads RGBA pixels; if false, reads RGB pixels with opaque alpha
     * \return Vector of RGBA color values
     */
    std::vector<helios::RGBAcolor> readOffscreenPixelsRGBA(bool read_alpha) const;

    /**
     * \brief Sets the size of the frame buffer.
     *
     * \param[in] width Width of the frame buffer
     * \param[in] height Height of the frame buffer
     */
    void setFrameBufferSize(int width, int height);

    /**
     * \brief Retrieves the primary shader used by the visualizer.
     *
     * \return The primary shader.
     */
    [[nodiscard]] Shader getPrimaryShader() const;

    /**
     * \brief Calculates and returns the view matrix for the camera.
     *
     * \return The view matrix representing the camera's position and orientation in the scene.
     */
    [[nodiscard]] glm::mat4 getViewMatrix() const;

    /**
     * \brief Retrieves the primary lighting model of the visualizer.
     *
     * \return The primary lighting model.
     */
    [[nodiscard]] LightingModel getPrimaryLightingModel();

    /**
     * \brief Retrieves the depth texture identifier.
     *
     * \return Identifier of the depth texture as an unsigned integer.
     */
    [[nodiscard]] uint getDepthTexture() const;

    void openWindow();

    void createOffscreenContext();

    //! Create the shadow-map framebuffer and depth texture
    /**
     * Called lazily the first time shadowed lighting is actually rendered, in both windowed
     * and headless modes. The shadow map is large (see shadow_buffer_size), so deferring it
     * avoids allocating it for the many Visualizer instances that never enable shadows.
     * Does nothing if the framebuffer has already been created.
     */
    void createShadowFramebuffer();

    //! Setup offscreen framebuffer for headless rendering
    void setupOffscreenFramebuffer();

    //! Clean up offscreen framebuffer resources
    void cleanupOffscreenFramebuffer();

    //! Switch rendering target to offscreen buffer
    void renderToOffscreenBuffer();

    //! Remove background rectangle (helper for background mode switching)
    void removeBackgroundRectangle();

    //! Set background to use gradient texture (default background mode)
    void setBackgroundGradient();

    //! Callback when the window framebuffer is resized
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    /**
     * \brief Callback function to handle window resizing.
     *
     * \param[in] window Pointer to the GLFW window being resized.
     * \param[in] width The new width of the window.
     * \param[in] height The new height of the window.
     */
    static void windowResizeCallback(GLFWwindow *window, int width, int height);

    /**
     * \brief Initializes the visualizer with specified configuration.
     *
     * \param[in] window_width_pixels Width of the window in pixels.
     * \param[in] window_height_pixels Height of the window in pixels.
     * \param[in] aliasing_samples Number of aliasing samples for rendering.
     * \param[in] window_decorations Indicates whether window decorations (e.g., borders, title bar) should be enabled.
     * \param[in] headless_mode [optional] If true, skips creation of the OpenGL window.
     */
    void initialize(uint window_width_pixels, uint window_height_pixels, int aliasing_samples, bool window_decorations, bool headless_mode);

    /**
     * \brief Renders the geometry using the current shader program.
     *
     * \param[in] shadow Indicates whether shadows should be included in the rendering process.
     */
    void render(bool shadow) const;

    /**
     * \brief Transfers buffer data to the GPU and sets up related textures.
     *
     * This function handles the transfer of updated geometry and texture data to GPU memory, ensuring
     * that changes in the application's data structures are properly reflected in rendering.
     */
    void transferBufferData();

    //! Uploads all textures to the texture array and updates UV rescaling.
    void transferTextureData();

    /**
     * \brief Registers a texture file and obtains its unique texture ID.
     *
     * \param[in] texture_file Path to the texture file to be registered.
     * \return A unique texture ID associated with the registered texture file.
     */
    [[nodiscard]] uint registerTextureImage(const std::string &texture_file);

    /**
     * \brief Registers a texture image with the visualizer and returns its unique texture ID.
     *
     * \param[in] texture_data The raw texture image data, expected in a flattened format with 4 components per pixel (RGBA).
     * \param[in] image_resolution The resolution of the image as a 2D integer vector (width and height).
     * \return A unique texture ID for the registered texture image.
     * \note This will always create a new texture for this data, even if the data is the same as a previously registered texture.
     */
    [[nodiscard]] uint registerTextureImage(const std::vector<unsigned char> &texture_data, const helios::uint2 &image_resolution);

    /**
     * \brief Registers a transparency mask for a given texture file.
     *
     * \param[in] texture_file The file path of the texture to register the transparency mask for.
     * \return The unique identifier (texture ID) for the registered texture.
     */
    [[nodiscard]] uint registerTextureTransparencyMask(const std::string &texture_file);

    /**
     * \brief Registers a texture glyph with the visualizer and assigns it a unique texture ID.
     *
     * \param[in] glyph Pointer to the glyph to be registered as a texture
     * \return Unique texture ID assigned to the registered glyph
     */
    [[nodiscard]] uint registerTextureGlyph(const Glyph *glyph);

    /**
     * \brief Retrieves the resolution of a texture.
     *
     * \param[in] textureID Identifier of the texture whose resolution is requested.
     * \return The resolution of the texture as an int2 structure, where the first element is the width and the second element is the height.
     */
    [[nodiscard]] helios::uint2 getTextureResolution(uint textureID) const;

    //~~~~~~~~~~~~~~~~ Primitives ~~~~~~~~~~~~~~~~~~~~//

    std::string colorPrimitivesByObjectData, colorPrimitivesByData;
    std::map<uint, uint> colorPrimitives_UUIDs, colorPrimitives_objIDs;

    std::vector<uint> contextUUIDs_build;

    std::vector<float> depth_buffer_data;

    void getViewKeystrokes(helios::vec3 &eye, helios::vec3 &center);

    /**
     * \brief Adds a colorbar to the visualization by specifying its center position.
     *
     * \param[in] title The text to be displayed as the title of the colorbar.
     * \param[in] size The size of the colorbar, where x represents the width, and y represents the height.
     * \param[in] center The position in 3D space representing the center of the colorbar.
     * \param[in] font_color The color of the font to be used for the title and ticks.
     * \param[in] colormap The colormap defining the gradient and range of the colorbar.
     * \return A vector of unique identifiers for the graphical elements created for the colorbar.
     */
    std::vector<size_t> addColorbarByCenter(const char *title, const helios::vec2 &size, const helios::vec3 &center, const helios::RGBcolor &font_color, const Colormap &colormap);

    //! Forget every cached identifier of geometry the visualizer manages internally
    /**
     * Call immediately after clearing all geometry. The watermark, background rectangle, background sky, coordinate axes, navigation gizmo and colorbar each cache the identifiers of the geometry they
     * created, and clearing the geometry destroys those identifiers without resetting the caches. Deleting by a stale identifier afterwards indexes GeometryHandler's UUID_map with at() on a key that is
     * no longer there, which throws.
     */
    void resetCachedGeometryIDs();

    //! Clear existing geometry and add the quad that displays an image, without entering the render loop
    /**
     * This is the shared body of \ref displayImage() and \ref displayImageWithBoundingBoxes(): everything those two do apart from plotting. The extent is returned so that overlay geometry can be positioned
     * against the displayed image rather than against the window, which differ whenever the image and window aspect ratios do not match.
     *
     * \param[in] pixel_data Pixel data of the image, of length 4*width_pixels*height_pixels.
     * \param[in] width_pixels Width of the image in pixels.
     * \param[in] height_pixels Height of the image in pixels.
     * \return Extent of the image quad in window-normalized coordinates, ordered as x_min, y_min, x_max, y_max.
     */
    helios::vec4 buildImageDisplayGeometry(const std::vector<unsigned char> &pixel_data, uint width_pixels, uint height_pixels);

    //! Add outline, label chip and label text geometry for a set of bounding boxes drawn over a displayed image
    /**
     * \param[in] bounding_boxes Boxes in normalized image coordinates, measured from the top-left corner of the image.
     * \param[in] class_names Map from class ID to the name displayed on the label. If this is empty, boxes are labeled with their numeric class ID. If it is not empty, a box whose class ID it does not contain is an error, because that means the annotation file and the class name file do not correspond.
     * \param[in] image_extent Extent of the displayed image quad in window-normalized coordinates, as returned by \ref buildImageDisplayGeometry().
     * \param[in] line_width Width of the box outlines in screen pixels.
     * \param[in] fontsize Size of the class label font in points.
     * \return Identifiers of every geometry element added, per box in input order: the four outline lines, then the label chip, then one rectangle per glyph of the label.
     */
    std::vector<size_t> addBoundingBoxOverlay(const std::vector<BoundingBox> &bounding_boxes, const std::map<uint, std::string> &class_names, const helios::vec4 &image_extent, float line_width, uint fontsize);

    //! Add fill, outline, label chip and label text geometry for a set of segmentation masks drawn over a displayed image
    /**
     * \param[in] masks Masks whose polygon vertices are in absolute pixel coordinates measured from the top-left corner of the image.
     * \param[in] image_extent Extent of the displayed image quad in window-normalized coordinates, as returned by \ref buildImageDisplayGeometry().
     * \param[in] fill_opacity Opacity of the translucent polygon fill, between 0 and 1. No fill geometry is added when this is 0.
     * \param[in] line_width Width of the mask outlines in screen pixels.
     * \param[in] fontsize Size of the class label font in points.
     * \param[in] show_labels Whether to add the label chip and its text. When false, neither is added and a mask contributes only its fill and outline.
     * \return Identifiers of every geometry element added, per mask in input order: the fill runs, then the outline lines, then the label chip, then one rectangle per glyph of the label.
     * \note The fill is emitted as one rectangle per horizontal run of covered pixels, so its element count scales with the height of the mask rather than with its vertex count.
     */
    std::vector<size_t> addSegmentationMaskOverlay(const std::vector<SegmentationMask> &masks, const helios::vec4 &image_extent, float fill_opacity, float line_width, uint fontsize, bool show_labels);

    void updateDepthBuffer();

    //! Width of the display window in screen coordinates
    uint Wdisplay;
    //! Height of the display window in screen coordinates
    uint Hdisplay;

    //! Width of the display window in pixels
    uint Wframebuffer;
    //! Height of the display window in pixels
    uint Hframebuffer;

    //! Ratio of framebuffer pixels to window screen coordinates
    /**
     * On a high-DPI (Retina) display the framebuffer is larger than the window in screen
     * coordinates, typically by a factor of two per axis. Content rasterized on the CPU at
     * screen-coordinate resolution and then drawn into the framebuffer is magnified by this
     * factor, so text glyphs must be rasterized at framebuffer resolution to appear sharp.
     * \return Scale factor, or 1 when the framebuffer matches the window (including headless mode).
     */
    [[nodiscard]] float getDPIScale() const;

    helios::uint2 shadow_buffer_size;

    uint frame_counter;

    //! Track whether buffers have been swapped since last render
    bool buffers_swapped_since_render;

    //! Handle to the GUI window
    /** \note This will be recast to have type GLFWwindow*.  This has to be done in order to keep library-dependent variables out of the header. */
    void *window;

    //! (x,y,z) coordinates of location where the camera is looking
    helios::vec3 camera_lookat_center;

    //! (x,y,z) coordinates of the camera (a.k.a. the `eye' location)
    helios::vec3 camera_eye_location;

    //! Minimum allowable distance from the camera eye location to the lookat location
    float minimum_view_radius;

    //! Handle to the OpenGL shader (primary)
    Shader primaryShader;

    //! Handle to the OpenGL shader (depth buffer for shadows)
    Shader depthShader;

    //! Handle to the OpenGL shader for wide lines (uses geometry shader)
    Shader lineShader;

    Shader *currentShader;

    uint framebufferID = 0;
    uint depthTexture = 0;

    // Separate framebuffer/texture for updateDepthBuffer(), which renders a camera-space
    // depth map at the window resolution rather than the shadow map's own resolution.
    uint depthbufferFramebufferID = 0;
    uint depthbufferTexture = 0;

    // Offscreen rendering support for CI testing
    uint offscreenFramebufferID = 0;
    uint offscreenColorTexture = 0;
    uint offscreenDepthTexture = 0;

    //! Multisampled framebuffer that headless rendering draws into, resolved into offscreenFramebufferID
    /**
     * Windowed rendering gets anti-aliasing from the GLFW window's own multisampled default
     * framebuffer, but the offscreen framebuffer used in headless mode is single-sampled, so every
     * headless image -- which is what saved figures are made from -- came out fully aliased. These
     * attachments give headless rendering the same anti-aliasing: geometry is drawn into the
     * multisampled framebuffer and blit-resolved into the single-sampled color texture before
     * readback. Zero when anti-aliasing is disabled, in which case rendering targets
     * offscreenFramebufferID directly.
     */
    uint offscreenMultisampleFramebufferID = 0;
    uint offscreenMultisampleColorBuffer = 0;
    uint offscreenMultisampleDepthBuffer = 0;

    //! Number of anti-aliasing samples requested at construction
    int antialiasing_sample_count = 0;

    //! Resolve the multisampled headless framebuffer into the single-sampled color texture
    /**
     * Does nothing when headless multisampling is not active. Must be called after all rendering for
     * a frame is complete and before the color texture is read back or sampled.
     */
    void resolveOffscreenMultisampleFramebuffer() const;

public:
    //! Query whether headless rendering is using a multisampled framebuffer
    /**
     * True when anti-aliasing was requested, the Visualizer is headless, and the driver provided the
     * multisampled attachments. False when anti-aliasing is off or the driver refused them, in which
     * case headless images are rendered without anti-aliasing.
     */
    [[nodiscard]] bool isHeadlessMultisamplingActive() const;

private:

    //! Lighting model for Context object primitives (default is LIGHTING_NONE)
    LightingModel primaryLightingModel;

    float lightintensity = 1.f;

    //! Multiplier applied to vertex-interpolated primitive colors in the fragment shader.
    //! 1.5 by default to brighten ordinary renders; set to 1 by enableExactColorMode() so that
    //! colors survive a Context -> framebuffer round trip unchanged.
    float colorboost = 1.5f;

    //! Whether the linear-light pipeline (sRGB decode, tone map, sRGB encode) is active
    bool linear_pipeline_enabled = true;

    //! Exposure multiplier applied in linear light before tone mapping
    float exposure = 1.f;

    //! Phong material parameters used to shade Context primitives
    PhongMaterial phong_material;

    //! Sky color for the hemispheric ambient term
    helios::RGBcolor ambient_sky_color = helios::make_RGBcolor(0.5f, 0.6f, 0.75f);

    //! Ground-bounce color for the hemispheric ambient term
    helios::RGBcolor ambient_ground_color = helios::make_RGBcolor(0.35f, 0.3f, 0.22f);

    //! Whether fragments are shaded with interpolated per-vertex normals rather than face normals
    bool smooth_shading_enabled = true;

    //! Resolve per-material Phong parameters into the packed GPU table
    /**
     * Walks only the distinct material IDs actually referenced by the geometry, reads each one's
     * Phong material data once, and packs the result into \ref phong_material_table_buffer. The
     * returned map assigns each material ID a dense index into that table; a material that
     * specifies no Phong data is absent from the map and its primitives keep an index of -1,
     * meaning they fall back to the global Phong material.
     *
     * This runs once per geometry build over the handful of materials in the scene, rather than
     * once per primitive, so the string-keyed material data lookups never enter the hot loop.
     *
     * \param[in] context Pointer to the Context whose materials are to be resolved.
     * \param[in] referenced_material_IDs Distinct material IDs used by the geometry being built.
     * \return Map from material ID to its dense index in the packed table.
     */
    std::unordered_map<uint, int> buildPhongMaterialTable(const helios::Context *context, const std::set<uint> &referenced_material_IDs);

    //! Resolve per-material Phong parameters and stamp the resulting table index onto each primitive
    /**
     * Called at every exit of \ref buildContextGeometry_private(), including the early return taken
     * when no primitive is dirty: material data can change without dirtying any primitive, because
     * \ref helios::Context::setMaterialData() does not touch the primitives that reference the
     * material.
     *
     * The per-primitive loop is skipped entirely when the resolved Phong data is unchanged from the
     * previous build, which is the common case in an interactive session.
     */
    void updatePhongMaterialIndices();

    //! Fingerprint of the Phong material data resolved into the packed table on the last build
    /**
     * Used to skip the whole per-material resolve and per-primitive index assignment when no Phong
     * material data has changed since the previous build, which is the overwhelmingly common case.
     * Rebuilding the table is cheap, but reassigning indices walks every displayed primitive, so it
     * must not run on every frame of an interactive session.
     */
    std::vector<float> phong_material_fingerprint;

    //! Whether the per-primitive Phong material indices have ever been assigned
    bool phong_material_indices_assigned = false;

    bool isWatermarkVisible;

    //! UUID associated with the watermark rectangle
    //! Add a textured rectangle whose texture alpha is blended rather than thresholded
    /**
     * Used for image overlays such as the watermark. The usual textured-rectangle path tests the
     * texture's alpha against a fixed threshold, which is what produces the cutout of an
     * alpha-masked leaf or bark texture but discards the partially-transparent edge pixels of an
     * image that is genuinely antialiased, leaving a jagged boundary.
     * \param[in] center (x,y,z) location of the rectangle center
     * \param[in] size Width and height of the rectangle
     * \param[in] rotation Spherical rotation angle
     * \param[in] texture_file Path to the image file
     * \param[in] coordFlag Coordinate system used for the spatial coordinates
     * \return Unique identifier for the rectangle geometry
     */
    size_t addAlphaBlendedRectangleByCenter(const helios::vec3 &center, const helios::vec2 &size, const helios::SphericalCoord &rotation, const char *texture_file, CoordinateSystem coordFlag);

    size_t watermark_ID;


    //! Color of the window background
    helios::RGBcolor backgroundColor;

    //! Flag indicating whether background is transparent (uses checkerboard texture for display, alpha transparency for PNG output)
    bool background_is_transparent;

    //! Track whether watermark was visible before transparent background was enabled (to restore it when switching back to solid color)
    bool watermark_was_visible_before_transparent;

    //! Track whether navigation gizmo was enabled before displaying an image (to restore it when building geometry)
    bool navigation_gizmo_was_enabled_before_image_display;

    //! UUID associated with the background rectangle (used for gradient, transparent checkerboard, or custom image backgrounds)
    size_t background_rectangle_ID;
    std::vector<size_t> background_sky_IDs;

    //! Vector pointing from the light source to the scene
    helios::vec3 light_direction;

    //! Vector containing UUIDs of the coordinate axes
    std::vector<size_t> coordinate_axes_IDs;

    //! Flag indicating whether navigation gizmo is enabled
    bool navigation_gizmo_enabled;

    //! Previous camera eye location for change detection
    helios::vec3 previous_camera_eye_location;

    //! Previous camera lookat center for change detection
    helios::vec3 previous_camera_lookat_center;

    //! Vector containing UUIDs of the navigation gizmo geometry
    std::vector<size_t> navigation_gizmo_IDs;

    //! Index of currently hovered navigation gizmo bubble (-1 = no hover, 0 = X, 1 = Y, 2 = Z)
    int hovered_gizmo_bubble;

    //! Flag indicating whether colorbar is enabled
    /** colorbar_flag=0 means the colorbar is off and no enable/disable functions have been called, colorbar_flag=1 means the colorbar is off and disableColorbar() was explicitly called and thus the colorbar should remain off, colorbar_flag=2 means
     * the colorbar is on. */
    uint colorbar_flag;

    //! Title of the colorbar
    std::string colorbar_title;

    //! Fontsize of colorbar text
    uint colorbar_fontsize;

    //! Width of points (if applicable) in pixels
    float point_width;

    //! Point cloud culling settings
    bool point_culling_enabled;
    size_t point_culling_threshold;
    float point_max_render_distance;
    float point_lod_factor;

    //! Point cloud performance metrics
    mutable size_t points_total_count;
    mutable size_t points_rendered_count;
    mutable float last_culling_time_ms;

    //! Color of colorbar text
    helios::RGBcolor colorbar_fontcolor;

    //! Position of colorbar center in normalized window coordinates
    helios::vec3 colorbar_position;

    //! x- and y- dimensions of colorbar in normalized window coordinates
    helios::vec2 colorbar_size;

    //! Intended aspect ratio (width/height) of the colorbar for maintaining proportions across window sizes
    float colorbar_intended_aspect_ratio;

    //! UUIDs associated with the current colorbar geometry
    std::vector<size_t> colorbar_IDs;

    //! Buffer objects to hold per-vertex data
    std::vector<GLuint> face_index_buffer, vertex_buffer, uv_buffer, vertex_normal_buffer;
    //! Buffer objects to hold per-primitive data. We will use textures to hold this data.
    std::vector<GLuint> color_buffer, normal_buffer, texture_flag_buffer, texture_ID_buffer, coordinate_flag_buffer, sky_geometry_flag_buffer, hidden_flag_buffer, material_index_buffer;
    //! Texture objects to hold per-primitive data.
    std::vector<GLuint> color_texture_object, normal_texture_object, texture_flag_texture_object, texture_ID_texture_object, coordinate_flag_texture_object, sky_geometry_flag_texture_object, hidden_flag_texture_object, material_index_texture_object;

    //! Buffer and texture holding the packed per-material Phong parameter table
    /** One RGBA32F texel per material: (ambient, diffuse, specular, shininess). Primitives index
        into this by the integer in material_index_buffer, so the four Phong floats are stored once
        per material rather than once per primitive. */
    GLuint phong_material_table_buffer = 0;
    GLuint phong_material_table_texture = 0;

    //! Number of entries currently in the packed Phong material table
    GLint phong_material_table_size = 0;

    //! Rescaling factor for texture (u,v)'s for when the texture size is smaller than the maximum texture size
    GLuint uv_rescale_buffer;
    GLuint uv_rescale_texture_object;

    //! These are index values for drawing of rectangles as a TRIANGLE_FAN. This needs to be stored so that it does not need to be re-computed for each render loop.
    std::vector<GLint> rectangle_vertex_group_firsts;
    std::vector<GLint> rectangle_vertex_group_counts;

    /**
     * \brief Computes the Model-View-Projection matrix for shadow depth rendering.
     *
     * \return The computed Model-View-Projection (MVP) matrix for shadow depth rendering.
     */
    [[nodiscard]] glm::mat4 computeShadowDepthMVP() const;

    void updatePerspectiveTransformation(bool shadow);

    //! Point cloud culling methods for performance optimization
    void cullPointsByFrustum();
    void cullPointsByDistance(float maxDistance, float lodFactor);
    void updatePointCulling();
    std::vector<glm::vec4> extractFrustumPlanes() const;

    glm::mat4 perspectiveTransformationMatrix;

    glm::mat4 cameraViewMatrix;
    glm::mat4 cameraProjectionMatrix;

    void updateCustomTransformation(const glm::mat4 &matrix);

    glm::mat4 customTransformationMatrix;

    //! Field of view of the camera in degrees
    float camera_FOV;

    bool build_all_context_geometry = false;

    //! UUIDs that have already been built into the geometry handler, and the Context dirty state they were built from
    /**
     * Context dirty flags are sticky: Context::markGeometryClean() is the only thing that clears them, and it is the
     * user's call to make once every plug-in has processed the change. A plug-in must therefore not clear them itself,
     * or the other consumers of getDirtyUUIDs() (e.g. CollisionDetection) would silently miss updates.
     *
     * Without plug-in-local tracking, though, getDirtyUUIDs() keeps reporting every primitive in the scene on every
     * frame, and each one is re-uploaded to the GPU one primitive at a time. Recording what has already been built lets
     * buildContextGeometry_private() re-upload only genuinely new or changed primitives, which is what makes the
     * per-frame cost proportional to what actually changed rather than to the size of the scene.
     *
     * \sa buildContextGeometry_private()
     */
    std::unordered_set<uint> contextUUIDs_uploaded;

    bool primitiveColorsNeedUpdate;

    helios::Context *context;

    //! Function to actually update Context geometry (if needed), which is called by the visualizer before plotting
    void buildContextGeometry_private();

    //! Update navigation gizmo geometry to match current camera orientation
    void updateNavigationGizmo();

    //! Update colorbar geometry to match current window aspect ratio
    void updateColorbar();

    //! Test if a normalized window coordinate hits a navigation gizmo bubble
    /**
     * \param[in] normalized_pos Position in normalized window coordinates (0-1)
     * \param[in] bubble_index Index of the bubble to test (0=X, 1=Y, 2=Z)
     * \return True if the position intersects with the bubble bounds
     */
    [[nodiscard]] bool testGizmoBubbleHit(const helios::vec2 &normalized_pos, int bubble_index) const;

    //! Reorient the camera to face along a specific axis
    /**
     * \param[in] axis_index Index of the axis (0=X, 1=Y, 2=Z)
     */
    void reorientCameraToAxis(int axis_index);

    //! Check if the camera position or orientation has changed since last update
    [[nodiscard]] bool cameraHasChanged() const;

    float colorbar_min;
    float colorbar_max;

    //! Whether the colorbar range was set explicitly by the user
    /**
     * Distinguishes an explicit range from an unset one. Inferring this from `colorbar_min == 0 && colorbar_max == 0` made a legitimate setColorbarRange(0, 0) indistinguishable from "never set", so it was silently replaced by the auto-detected data range.
     */
    bool colorbar_range_set = false;
    std::vector<float> colorbar_ticks;
    bool colorbar_integer_data;

    //! Current colormap used in visualization
    Colormap colormap_current;

    //! "hot" colormap used in visualization
    Colormap colormap_hot;

    //! "cool" colormap used in visualization
    Colormap colormap_cool;

    //! "lava" colormap used in visualization
    Colormap colormap_lava;

    //! "rainbow" colormap used in visualization
    Colormap colormap_rainbow;

    //! "parula" colormap used in visualization
    Colormap colormap_parula;

    //! "gray" colormap used in visualization
    Colormap colormap_gray;

    //! "lines" colormap used in visualization
    Colormap colormap_lines;

    bool message_flag;

    //! Flag indicating whether the visualizer is running without an OpenGL window
    bool headless;

    GeometryHandler geometry_handler;

    GLuint texArray = 0;
    size_t texture_array_layers = 0;
    //! Dimensions each layer of the texture array is currently allocated at, in texels
    /**
     * Every layer of a texture array is the same size, so this is the size of the largest texture
     * in the manager rather than a fixed maximum. Allocating every layer at
     * \ref maximum_texture_size instead would give a 7x9-texel glyph a 2048x2048 RGBA8 layer, or
     * 16 MB to store roughly 250 bytes. It is tracked here because the array is immutable once
     * created, so a change in the required size forces the same reallocation a change in the layer
     * count does, and because the shader's UV rescale factors are relative to it.
     */
    helios::uint2 texture_array_layer_size = helios::make_uint2(0, 0);
    bool textures_dirty;

    //! Sampler applying linear filtering, bound only while drawing text glyphs
    /**
     * The texture array is shared by glyphs and image textures. Glyphs carry an antialiased
     * coverage mask that must be interpolated to look smooth, whereas image textures are
     * alpha-masked against a fixed threshold and are filtered with GL_NEAREST so that the
     * resulting cutout is not shifted. A sampler object overrides the texture object's own filter
     * state for the duration of a draw, which keeps the two cases separate.
     */
    GLuint glyph_sampler = 0;

    //! Sampler applying linear filtering, bound only while drawing the watermark
    /**
     * Shares its filter settings with \ref glyph_sampler but is bound from the opaque draw path
     * rather than the transparent one. Kept as a separate object so that the two uses can diverge
     * without disturbing each other.
     */
    GLuint image_sampler = 0;

    helios::uint2 maximum_texture_size;

    const glm::mat4 biasMatrix = {0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0};

    struct Texture {

        /**
         * \brief Constructs a Texture object and loads a texture from a given file.
         *
         * This constructor initializes the Texture object by loading texture data
         * from the specified file, setting up OpenGL texture properties, and ensuring
         * compatibility with rendering requirements. It supports loading .jpg, .jpeg,
         * and .png file formats and handles texture padding to align dimensions to
         * the next power of two.
         *
         * \param[in] texture_file Path to the texture file to be loaded.
         * \param[in] textureID Unique identifier for the texture in OpenGL.
         * \param[in] maximum_texture_size Maximum texture size supported by the system.
         * \param[in] loadalphaonly Indicates whether to load only the alpha channel
         *                          of the texture. If true, only the alpha channel
         *                          will be considered, otherwise all channels are loaded.
         */
        explicit Texture(const std::string &texture_file, uint textureID, const helios::uint2 &maximum_texture_size, bool loadalphaonly = false);

        /**
         * \brief Constructs a Texture object for a specific glyph and texture ID.
         *
         * This constructor initializes a texture object using the given glyph data
         * and assigns it a unique texture ID for OpenGL rendering. The glyph data
         * is used to define the texture resolution and fill the texture buffer.
         * Additional OpenGL texture properties are configured for rendering.
         *
         * \param[in] glyph_ptr Pointer to a Glyph object containing glyph data.
         * \param[in] textureID Unique identifier for the texture in OpenGL.
         * \param[in] maximum_texture_size Maximum texture size supported by the system.
         */
        explicit Texture(const Glyph *glyph_ptr, uint textureID, const helios::uint2 &maximum_texture_size);

        /**
         * \brief Constructs a Texture object using pixel data and additional parameters.
         *
         * This constructor initializes the texture object with the provided pixel data, texture ID, image resolution,
         * and maximum texture size. If the texture image exceeds the maximum allowable resolution, it is resized accordingly.
         *
         * \param[in] pixel_data The raw pixel data for the texture, represented as a vector of unsigned chars.
         * \param[in] textureID Unique identifier for the texture in OpenGL.
         * \param[in] image_resolution The resolution of the texture, specified as a 2D integer vector.
         * \param[in] maximum_texture_size The maximum allowable texture size, specified as a 2D integer vector.
         */
        explicit Texture(const std::vector<unsigned char> &pixel_data, uint textureID, const helios::uint2 &image_resolution, const helios::uint2 &maximum_texture_size);

        //! Path to the texture file to be loaded.
        std::string texture_file;
        //! Data structure representing a glyph object.
        Glyph glyph;
        //! Represents the resolution of a texture in 2D space.
        helios::uint2 texture_resolution;
        //! Unique identifier for the texture in OpenGL.
        uint textureID;
        //! Stores the raw texture data.
        std::vector<unsigned char> texture_data;
        //! Number of channels in the texture data.
        unsigned char num_channels;

        /**
         * \brief Resizes the texture to a new resolution.
         *
         * This function updates the texture resolution to the specified new resolution,
         * resampling the existing texture data to fit the new dimensions using bilinear interpolation.
         *
         * \param[in] new_image_resolution New resolution for the texture, specified as a 2D vector (width, height).
         */
        void resizeTexture(const helios::uint2 &new_image_resolution);
    };

    /**
     * \brief A mapping of texture IDs to Texture objects.
     *
     * This container is used to manage and access textures efficiently via their unique IDs.
     */
    std::unordered_map<uint, Texture> texture_manager;

    friend struct Shader;
    friend struct Texture;

    //! Test-only accessor for private tick-generation helpers and colorbar state
    /**
     * Defined solely in plugins/visualizer/tests/selfTest.cpp. The tick helpers are private
     * because they are internal implementation details rather than user-facing API, but they are
     * pure functions that are worth unit testing directly.
     */
    friend class VisualizerTestHelper;
};

inline glm::vec3 glm_vec3(const helios::vec3 &v) {
    return {v.x, v.y, v.z};
}


int checkerrors();

//! Safe error checking that throws exceptions instead of using assert
void check_opengl_errors_safe(const std::string &context);


#endif
