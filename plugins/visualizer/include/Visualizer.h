/** \file "Visualizer.h" Visualizer header.

    Copyright (C) 2016-2025 Brian Bailey

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
#include "Visualizer.h"

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

    //! Set the depth bias matrix for shadows
    void setDepthBiasMatrix(const glm::mat4 &matrix) const;

    //! Set the direction of the light (sun)
    void setLightDirection(const helios::vec3 &direction) const;

    //! Set the lighting model
    void setLightingModel(uint lightingmodel) const;

    //! Set the intensity of the light source
    void setLightIntensity(float lightintensity) const;

    //! Set shader as current
    void useShader() const;

    //! Initialize the shader
    /**
     * \param[in] vertex_shader_file Name of vertex shader file to be used by OpenGL in rendering graphics
     * \param[in] fragment_shader_file Name of fragment shader file to be used by OpenGL in rendering graphics
     * \param[in] visualizer_ptr Pointer to the Visualizer class
     */
    void initialize(const char *vertex_shader_file, const char *fragment_shader_file, Visualizer *visualizer_ptr);

    ~Shader();

    // Primary Shader
    uint shaderID;
    GLint textureUniform;
    GLint shadowmapUniform;
    GLint transformMatrixUniform;
    GLint depthBiasUniform;
    GLint lightDirectionUniform;
    GLint lightingModelUniform;
    GLint RboundUniform;
    GLint lightIntensityUniform;
    std::vector<GLuint> vertex_array_IDs;
    GLint uvRescaleUniform;

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
int write_JPEG_file(const char *filename, uint width, uint height, bool print_messages);

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
    static int selfTest(int argc = 0, char** argv = nullptr);

    //! Enable standard output from this plug-in (default)
    void enableMessages();

    //! Disable standard output from this plug-in
    void disableMessages();

    //! Coordinate system to be used when specifying spatial coordinates
    enum CoordinateSystem {
        //! Coordinates are normalized to unity and are window-aligned.  The point (x,y)=(0,0) is in the bottom left corner of the window, and (x,y)=(1,1) is in the upper right corner of the window.  The z-coordinate specifies the depth in the
        //! screen-normal direction, with values ranging from -1 to 1.  For example, an object at z=0.5 would be in front of an object at z=0.
        COORDINATES_WINDOW_NORMALIZED = 0,

        //! Coordinates are specified in a 3D Cartesian system (right-handed), where +z is vertical.
        COORDINATES_CARTESIAN = 1
    };

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
        COLORMAP_CUSTOM = 6
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

    //! Set the background color for the visualizer window
    /**
     * \param[in] color Background color
     */
    void setBackgroundColor(const helios::RGBcolor &color);

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
     * \param[in] line_width Width of the line in pixels
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     */
    size_t addLine(const helios::vec3 &start, const helios::vec3 &end, const helios::RGBcolor &color, float line_width, CoordinateSystem coordinate_system);

    //! Add Lines by giving the coordinates of points along the Lines with custom line width
    /**
     * \param[in] start (x,y,z) coordinates of line starting position
     * \param[in] end (x,y,z) coordinates of line ending position
     * \param[in] color R-G-B-A color of the line
     * \param[in] line_width Width of the line in pixels
     * \param[in] coordFlag Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
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
     * \param[in] radius Radius of the dome
     * \param[in] center (x,y,z) location of dome center
     * \param[in] Ndivisions Number of discrete divisions in making hemisphere
     * \param[in] texture_file Name of the texture map file
     */
    std::vector<size_t> addSkyDomeByCenter(float radius, const helios::vec3 &center, uint Ndivisions, const char *texture_file);

    //! Add a Sky Dome, which is a hemispherical dome colored by a sky texture map
    /** \note This function has been deprecated, as layers are no longer supported. */
    DEPRECATED(void addSkyDomeByCenter(float radius, const helios::vec3 &center, uint Ndivisions, const char *texture_file, int layer));

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

    //! Removes the geometry with the specified ID from the visualizer.
    /**
     * \param[in] geometry_id The unique identifier of the geometry to delete.
     */
    void deleteGeometry(size_t geometry_id);

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
     * \param[out] cmax Maximum value
     */
    void setColorbarRange(float cmin, float cmax);

    //! Set the values in the colorbar where ticks and labels should be placed
    /**
     * \param[in] ticks Vector of values corresponding to ticks
        \note If tick values are outside of the colorbar range (see setColorBarRange()), the colorbar will be automatically expanded to fit the tick values.
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
     * \note Valid colormaps are "COLORMAP_HOT", "COLORMAP_COOL", "COLORMAP_LAVA", "COLORMAP_RAINBOW", "COLORMAP_PARULA", "COLORMAP_GRAY".
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

    //! Plot current geometry into an interactive graphics window
    std::vector<helios::vec3> plotInteractive();

    //! Run one rendering loop from plotInteractive()
    /**
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

    //! Print the current graphics window to a JPEG image file
    /**
     * \param[in] outfile Path to file where image should be saved.
     * \note If outfile does not have extension `.jpg', it will be appended to the file name.
     */
    void printWindow(const char *outfile) const;

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

    //! Get R-G-B pixel data in the current display window
    /**
     * \param[out] buffer Pixel data. The data is stored as r-g-b * column * row. So indices (0,1,2) would be the RGB values for row 0 and column 0, indices (3,4,5) would be RGB values for row 0 and column 1, and so on. Thus, buffer is of size
     * 3*width*height.
     */
    void getWindowPixelsRGB(uint *buffer) const;

    //! Get depth buffer data for the current display window
    /**
     * \param[out] buffer Distance to nearest object from the camera location.
     */
    DEPRECATED(void getDepthMap(float *buffer));

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
    /**
     * \brief Retrieves the size of the framebuffer.
     *
     * \return A vector containing the width and height of the framebuffer.
     */
    [[nodiscard]] std::vector<uint> getFrameBufferSize() const;

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
     * \return A vector containing the primary lighting model(s).
     */
    [[nodiscard]] std::vector<LightingModel> getPrimaryLightingModel();

    /**
     * \brief Retrieves the depth texture identifier.
     *
     * \return Identifier of the depth texture as an unsigned integer.
     */
    [[nodiscard]] uint getDepthTexture() const;

    void openWindow();

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

    void updateDepthBuffer();

    //! Width of the display window in screen coordinates
    uint Wdisplay;
    //! Height of the display window in screen coordinates
    uint Hdisplay;

    //! Width of the display window in pixels
    uint Wframebuffer;
    //! Height of the display window in pixels
    uint Hframebuffer;

    helios::uint2 shadow_buffer_size;

    uint frame_counter;

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

    Shader *currentShader;

    uint framebufferID;
    uint depthTexture;

    //! Lighting model for Context object primitives (default is LIGHTING_NONE)
    std::vector<LightingModel> primaryLightingModel;

    float lightintensity = 1.f;

    bool isWatermarkVisible;

    //! UUID associated with the watermark rectangle
    size_t watermark_ID;

    //! Color of the window background
    helios::RGBcolor backgroundColor;

    //! Vector pointing from the light source to the scene
    helios::vec3 light_direction;

    //! Vector containing UUIDs of the coordinate axes
    std::vector<size_t> coordinate_axes_IDs;

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

    //! UUIDs associated with the current colorbar geometry
    std::vector<size_t> colorbar_IDs;

    //! Buffer objects to hold per-vertex data
    std::vector<GLuint> face_index_buffer, vertex_buffer, uv_buffer;
    //! Buffer objects to hold per-primitive data. We will use textures to hold this data.
    std::vector<GLuint> color_buffer, normal_buffer, texture_flag_buffer, texture_ID_buffer, coordinate_flag_buffer, hidden_flag_buffer;
    //! Texture objects to hold per-primitive data.
    std::vector<GLuint> color_texture_object, normal_texture_object, texture_flag_texture_object, texture_ID_texture_object, coordinate_flag_texture_object, hidden_flag_texture_object;

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

    bool primitiveColorsNeedUpdate;

    helios::Context *context;

    //! Function to actually update Context geometry (if needed), which is called by the visualizer before plotting
    void buildContextGeometry_private();

    float colorbar_min;
    float colorbar_max;
    std::vector<float> colorbar_ticks;

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

    bool message_flag;

    //! Flag indicating whether the visualizer is running without an OpenGL window
    bool headless;

    GeometryHandler geometry_handler;

    GLuint texArray;
    size_t texture_array_layers;
    bool textures_dirty;

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
};

inline glm::vec3 glm_vec3(const helios::vec3 &v) {
    return {v.x, v.y, v.z};
}


int checkerrors();


#endif
