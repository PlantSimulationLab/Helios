/** \file "VisualizerCore.cpp" Visualizer core functions including constructors, initialization, and utility functions.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

// OpenGL Includes
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Freetype Libraries (rendering fonts)
extern "C" {
#include <ft2build.h>
#include FT_FREETYPE_H
}

#include "Visualizer.h"

using namespace helios;

int read_JPEG_file(const char *filename, std::vector<unsigned char> &texture, uint &height, uint &width) {
    std::vector<helios::RGBcolor> rgb_data;
    helios::readJPEG(filename, width, height, rgb_data);

    texture.clear();
    texture.reserve(width * height * 4);

    for (const auto &pixel: rgb_data) {
        texture.push_back(static_cast<unsigned char>(pixel.r * 255.0f));
        texture.push_back(static_cast<unsigned char>(pixel.g * 255.0f));
        texture.push_back(static_cast<unsigned char>(pixel.b * 255.0f));
        texture.push_back(255); // alpha channel - opaque
    }

    return 0;
}

int write_JPEG_file(const char *filename, uint width, uint height, bool print_messages) {
    if (print_messages) {
        std::cout << "writing JPEG image: " << filename << std::endl;
    }

    const size_t bsize = 3 * width * height;
    std::vector<GLubyte> screen_shot_trans;
    screen_shot_trans.resize(bsize);

#if defined(__APPLE__)
    constexpr GLenum read_buf = GL_FRONT;
#else
    constexpr GLenum read_buf = GL_BACK;
#endif
    glReadBuffer(read_buf);
    glReadPixels(0, 0, scast<GLsizei>(width), scast<GLsizei>(height), GL_RGB, GL_UNSIGNED_BYTE, &screen_shot_trans[0]);
    glFinish();

    // Convert to RGBcolor vector and use Context's writeJPEG
    std::vector<helios::RGBcolor> rgb_data;
    rgb_data.reserve(width * height);

    for (size_t i = 0; i < width * height; i++) {
        size_t byte_idx = i * 3;
        rgb_data.emplace_back(screen_shot_trans[byte_idx] / 255.0f, screen_shot_trans[byte_idx + 1] / 255.0f, screen_shot_trans[byte_idx + 2] / 255.0f);
    }

    helios::writeJPEG(filename, width, height, rgb_data);
    return 1;
}

int write_JPEG_file(const char *filename, uint width, uint height, const std::vector<helios::RGBcolor> &data, bool print_messages) {
    if (print_messages) {
        std::cout << "writing JPEG image: " << filename << std::endl;
    }

    helios::writeJPEG(filename, width, height, data);
    return 1;
}

void read_png_file(const char *filename, std::vector<unsigned char> &texture, uint &height, uint &width) {
    std::vector<helios::RGBAcolor> rgba_data;
    helios::readPNG(filename, width, height, rgba_data);

    texture.clear();
    texture.reserve(width * height * 4);

    for (const auto &pixel: rgba_data) {
        texture.push_back(static_cast<unsigned char>(pixel.r * 255.0f));
        texture.push_back(static_cast<unsigned char>(pixel.g * 255.0f));
        texture.push_back(static_cast<unsigned char>(pixel.b * 255.0f));
        texture.push_back(static_cast<unsigned char>(pixel.a * 255.0f));
    }
}

Visualizer::Visualizer(uint Wdisplay) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, uint(std::round(Wdisplay * 0.8)), 16, true, false);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, Hdisplay, 16, true, false);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, Hdisplay, aliasing_samples, true, false);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples, bool window_decorations, bool headless) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, Hdisplay, aliasing_samples, window_decorations, headless);
}

void Visualizer::openWindow() {
    // Open a window and create its OpenGL context
    GLFWwindow *_window = glfwCreateWindow(Wdisplay, Hdisplay, "Helios 3D Simulation", nullptr, nullptr);
    if (_window == nullptr) {
        std::string errorsrtring;
        errorsrtring.append("ERROR(Visualizer): Failed to initialize graphics.\n");
        errorsrtring.append("Common causes for this error:\n");
        errorsrtring.append("-- OSX\n  - Is XQuartz installed (xquartz.org) and configured as the default X11 window handler?  When running the visualizer, XQuartz should automatically open and appear in the dock, indicating it is working.\n");
        errorsrtring.append("-- Linux\n  - Are you running this program remotely via SSH? Remote X11 graphics along with OpenGL are not natively supported.  Installing and using VirtualGL is a good solution for this (virtualgl.org).\n");
        helios_runtime_error(errorsrtring);
    }
    glfwMakeContextCurrent(_window);

    // Associate this Visualizer instance with the GLFW window so that
    // callbacks have access to it.
    glfwSetWindowUserPointer(_window, this);

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(_window, GLFW_STICKY_KEYS, GL_TRUE);

    window = (void *) _window;

    int window_width, window_height;
    glfwGetWindowSize(_window, &window_width, &window_height);

    int framebuffer_width, framebuffer_height;
    glfwGetFramebufferSize(_window, &framebuffer_width, &framebuffer_height);

    Wframebuffer = uint(framebuffer_width);
    Hframebuffer = uint(framebuffer_height);

    if (window_width < Wdisplay || window_height < Hdisplay) {
        std::cerr << "WARNING (Visualizer): requested size of window is larger than the screen area." << std::endl;
        Wdisplay = uint(window_width);
        Hdisplay = uint(window_height);
    }

    glfwSetWindowSize(_window, window_width, window_height);

    // Allow the window to freely resize so that entering full-screen
    // results in the framebuffer matching the display resolution.
    // This prevents the operating system from simply scaling the
    // window contents, which can skew geometry.
    glfwSetWindowAspectRatio(_window, GLFW_DONT_CARE, GLFW_DONT_CARE);

    // Register callbacks so that window and framebuffer size changes
    // properly update the internal dimensions used for rendering.
    glfwSetWindowSizeCallback(_window, Visualizer::windowResizeCallback);
    glfwSetFramebufferSizeCallback(_window, Visualizer::framebufferResizeCallback);

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        helios_runtime_error("ERROR (Visualizer): Failed to initialize GLEW.");
    }

    // NOTE: for some reason calling glewInit throws an error.  Need to clear it to move on.
    glGetError();
}

void Visualizer::initialize(uint window_width_pixels, uint window_height_pixels, int aliasing_samples, bool window_decorations, bool headless_mode) {
    Wdisplay = window_width_pixels;
    Hdisplay = window_height_pixels;

    headless = headless_mode;

    shadow_buffer_size = make_uint2(8192, 8192);

    maximum_texture_size = make_uint2(2048, 2048);

    texArray = 0;
    texture_array_layers = 0;
    textures_dirty = false;

    message_flag = true;

    frame_counter = 0;

    camera_FOV = 45;

    minimum_view_radius = 0.05f;

    context = nullptr;
    primitiveColorsNeedUpdate = false;

    isWatermarkVisible = true;
    watermark_ID = 0;

    colorbar_flag = 0;

    colorbar_min = 0.f;
    colorbar_max = 0.f;

    colorbar_title = "";
    colorbar_fontsize = 12;
    colorbar_fontcolor = RGB::black;

    colorbar_position = make_vec3(0.65, 0.1, 0.1);
    colorbar_size = make_vec2(0.15, 0.1);
    colorbar_IDs.clear();

    point_width = 1;

    // Initialize point cloud culling settings
    point_culling_enabled = true;
    point_culling_threshold = 10000; // Enable culling for point clouds with 10K+ points
    point_max_render_distance = 0; // Auto-calculated based on scene size
    point_lod_factor = 10.0f; // Cull every 10th point in far regions

    // Initialize performance metrics
    points_total_count = 0;
    points_rendered_count = 0;
    last_culling_time_ms = 0;

    // Initialize OpenGL context for both regular and headless modes
    // Headless mode needs an offscreen context for geometry operations

    // Initialise GLFW
    if (!glfwInit()) {
        helios_runtime_error("ERROR (Visualizer::initialize): Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_SAMPLES, std::max(0, aliasing_samples)); // antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL
#endif

    if (headless) {
        // Create offscreen context for headless mode
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Ensure window is not visible
    } else {
        // Regular windowed mode
        glfwWindowHint(GLFW_VISIBLE, 0); // Initially hidden, will show later if needed

        if (!window_decorations) {
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        }
    }

    openWindow();

    // Initialize GLEW - required for both headless and windowed modes
    glewExperimental = GL_TRUE; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        helios_runtime_error("ERROR (Visualizer::initialize): Failed to initialize GLEW");
    }

    // NOTE: for some reason calling glewInit throws an error.  Need to clear it to move on.
    glGetError();

    // Check for OpenGL errors after GLEW initialization
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::initialize): OpenGL context initialization failed after GLEW setup. "
                             "This often occurs in headless CI environments without proper GPU drivers or display servers. "
                             "For headless operation, ensure proper virtual display or software rendering is configured.");
    }

    // Enable relevant parameters for both regular and headless modes

    glEnable(GL_DEPTH_TEST); // Enable depth test
    glDepthFunc(GL_LESS); // Accept fragment if it closer to the camera than the former one
    // glEnable(GL_DEPTH_CLAMP);

    if (aliasing_samples <= 0) {
        glDisable(GL_MULTISAMPLE);
        glDisable(GL_MULTISAMPLE_ARB);
    }

    if (aliasing_samples <= 1) {
        glDisable(GL_POLYGON_SMOOTH);
    } else {
        glEnable(GL_POLYGON_SMOOTH);
    }

    // glEnable(GL_TEXTURE0);
    //  glEnable(GL_TEXTURE_2D_ARRAY);
    //  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    //  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Check for OpenGL errors after basic setup
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::initialize): OpenGL context setup failed during basic parameter configuration. "
                             "This typically indicates graphics driver incompatibility or missing OpenGL support in the execution environment.");
    }

    // glEnable(GL_TEXTURE1);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glDisable(GL_CULL_FACE);

    // Check for OpenGL errors after advanced setup
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::initialize): OpenGL context setup failed during advanced parameter configuration. "
                             "Verify that the graphics environment supports the required OpenGL version and features.");
    }

    // Initialize VBO's and texture buffers
    constexpr size_t Ntypes = GeometryHandler::all_geometry_types.size();
    // per-vertex data
    face_index_buffer.resize(Ntypes);
    vertex_buffer.resize(Ntypes);
    uv_buffer.resize(Ntypes);
    glGenBuffers((GLsizei) face_index_buffer.size(), face_index_buffer.data());
    glGenBuffers((GLsizei) vertex_buffer.size(), vertex_buffer.data());
    glGenBuffers((GLsizei) uv_buffer.size(), uv_buffer.data());

    // per-primitive data
    color_buffer.resize(Ntypes);
    color_texture_object.resize(Ntypes);
    normal_buffer.resize(Ntypes);
    normal_texture_object.resize(Ntypes);
    texture_flag_buffer.resize(Ntypes);
    texture_flag_texture_object.resize(Ntypes);
    texture_ID_buffer.resize(Ntypes);
    texture_ID_texture_object.resize(Ntypes);
    coordinate_flag_buffer.resize(Ntypes);
    coordinate_flag_texture_object.resize(Ntypes);
    hidden_flag_buffer.resize(Ntypes);
    hidden_flag_texture_object.resize(Ntypes);
    glGenBuffers((GLsizei) color_buffer.size(), color_buffer.data());
    glGenTextures((GLsizei) color_texture_object.size(), color_texture_object.data());
    glGenBuffers((GLsizei) normal_buffer.size(), normal_buffer.data());
    glGenTextures((GLsizei) normal_texture_object.size(), normal_texture_object.data());
    glGenBuffers((GLsizei) texture_flag_buffer.size(), texture_flag_buffer.data());
    glGenTextures((GLsizei) texture_flag_texture_object.size(), texture_flag_texture_object.data());
    glGenBuffers((GLsizei) texture_ID_buffer.size(), texture_ID_buffer.data());
    glGenTextures((GLsizei) texture_ID_texture_object.size(), texture_ID_texture_object.data());
    glGenBuffers((GLsizei) coordinate_flag_buffer.size(), coordinate_flag_buffer.data());
    glGenTextures((GLsizei) coordinate_flag_texture_object.size(), coordinate_flag_texture_object.data());
    glGenBuffers((GLsizei) hidden_flag_buffer.size(), hidden_flag_buffer.data());
    glGenTextures((GLsizei) hidden_flag_texture_object.size(), hidden_flag_texture_object.data());

    glGenBuffers(1, &uv_rescale_buffer);
    glGenTextures(1, &uv_rescale_texture_object);

    // Check for OpenGL errors after buffer creation
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::initialize): OpenGL buffer creation failed. "
                             "This indicates insufficient graphics memory or unsupported buffer operations in the current OpenGL context.");
    }

    //~~~~~~~~~~~~~ Load the Shaders ~~~~~~~~~~~~~~~~~~~//

    primaryShader.initialize("plugins/visualizer/shaders/primaryShader.vert", "plugins/visualizer/shaders/primaryShader.frag", this);
    depthShader.initialize("plugins/visualizer/shaders/shadow.vert", "plugins/visualizer/shaders/shadow.frag", this);

    // Check for OpenGL errors after shader initialization
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::initialize): Shader initialization failed. "
                             "Verify that shader files are accessible and the OpenGL context supports the required shading language version.");
    }

    primaryShader.useShader();

    // Initialize frame buffer only for windowed mode
    if (!headless) {
        // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
        glGenFramebuffers(1, &framebufferID);
        glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);

        // Depth texture. Slower than a depth buffer, but you can sample it later in your shader
        glActiveTexture(GL_TEXTURE1);
        glGenTextures(1, &depthTexture);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, shadow_buffer_size.x, shadow_buffer_size.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // clamp to border so any lookup outside [0,1] returns 1.0 (no shadow)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        GLfloat borderColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        // enable hardware depth comparison
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

        if (!checkerrors()) {
            helios_runtime_error("ERROR (Visualizer::initialize): OpenGL setup failed during texture configuration. "
                                 "This may indicate graphics driver issues or insufficient OpenGL support.");
        }

        // restore default active texture for subsequent texture setup
        glActiveTexture(GL_TEXTURE0);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);

        glDrawBuffer(GL_NONE); // No color buffer is drawn to.

        // Always check that our framebuffer is ok
        int max_checks = 10000;
        int checks = 0;
        while (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE && checks < max_checks) {
            checks++;
        }
        // Check framebuffer completeness instead of using assert
        GLenum framebuffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (framebuffer_status != GL_FRAMEBUFFER_COMPLETE) {
            std::string error_message = "ERROR (Visualizer::initialize): Framebuffer is incomplete. Status: ";
            switch (framebuffer_status) {
                case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                    error_message += "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT - Framebuffer attachment is incomplete";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                    error_message += "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT - No attachments";
                    break;
                case GL_FRAMEBUFFER_UNSUPPORTED:
                    error_message += "GL_FRAMEBUFFER_UNSUPPORTED - Unsupported framebuffer format";
                    break;
                default:
                    error_message += "Unknown framebuffer error code: " + std::to_string(framebuffer_status);
                    break;
            }
            error_message += ". This typically occurs in CI environments with limited graphics support or missing GPU drivers.";
            helios_runtime_error(error_message);
        }

        // Finished OpenGL setup
        // Check for OpenGL errors after framebuffer setup
        if (!checkerrors()) {
            helios_runtime_error("ERROR (Visualizer::initialize): Framebuffer setup failed. "
                                 "This indicates issues with OpenGL framebuffer operations, often related to graphics driver limitations or insufficient resources.");
        }
    } else {
        // Set framebuffer dimensions for headless mode (no framebuffer created)
        Wframebuffer = Wdisplay;
        Hframebuffer = Hdisplay;
    }

    // Initialize transformation matrices

    perspectiveTransformationMatrix = glm::mat4(1.f);

    customTransformationMatrix = glm::mat4(1.f);

    // Default values

    light_direction = make_vec3(1, 1, 1);
    light_direction.normalize();
    if (!headless) {
        primaryShader.setLightDirection(light_direction);
    }

    primaryLightingModel.push_back(Visualizer::LIGHTING_NONE);

    camera_lookat_center = make_vec3(0, 0, 0);
    camera_eye_location = camera_lookat_center + sphere2cart(make_SphericalCoord(2.f, 90.f * PI_F / 180.f, 0));

    backgroundColor = make_RGBcolor(0.8, 0.8, 0.8);

    // colormaps

    // HOT
    std::vector<RGBcolor> ctable_c{{0.f, 0.f, 0.f}, {0.5f, 0.f, 0.5f}, {1.f, 0.f, 0.f}, {1.f, 0.5f, 0.f}, {1.f, 1.f, 0.f}};

    std::vector<float> clocs_c{0.f, 0.25f, 0.5f, 0.75f, 1.f};

    colormap_hot.set(ctable_c, clocs_c, 100, 0, 1);

    // COOL
    ctable_c = {RGB::cyan, RGB::magenta};

    clocs_c = {0.f, 1.f};

    colormap_cool.set(ctable_c, clocs_c, 100, 0, 1);

    // LAVA
    ctable_c = {{0.f, 0.05f, 0.05f}, {0.f, 0.6f, 0.6f}, {1.f, 1.f, 1.f}, {1.f, 0.f, 0.f}, {0.5f, 0.f, 0.f}};

    clocs_c = {0.f, 0.4f, 0.5f, 0.6f, 1.f};

    colormap_lava.set(ctable_c, clocs_c, 100, 0, 1);

    // RAINBOW
    ctable_c = {RGB::navy, RGB::cyan, RGB::yellow, make_RGBcolor(0.75f, 0.f, 0.f)};

    clocs_c = {0, 0.3f, 0.7f, 1.f};

    colormap_rainbow.set(ctable_c, clocs_c, 100, 0, 1);

    // PARULA
    ctable_c = {RGB::navy, make_RGBcolor(0, 0.6, 0.6), RGB::goldenrod, RGB::yellow};

    clocs_c = {0, 0.4f, 0.7f, 1.f};

    colormap_parula.set(ctable_c, clocs_c, 100, 0, 1);

    // GRAY
    ctable_c = {RGB::black, RGB::white};

    clocs_c = {0.f, 1.f};

    colormap_gray.set(ctable_c, clocs_c, 100, 0, 1);

    colormap_current = colormap_hot;

    if (!headless) {
        glfwSetMouseButtonCallback((GLFWwindow *) window, mouseCallback);
        glfwSetCursorPosCallback((GLFWwindow *) window, cursorCallback);
        glfwSetScrollCallback((GLFWwindow *) window, scrollCallback);

        // Check for OpenGL errors after callback setup
        if (!checkerrors()) {
            helios_runtime_error("ERROR (Visualizer::initialize): Final OpenGL setup failed during callback configuration. "
                                 "The OpenGL context may be in an invalid state or missing required extensions.");
        }
    }
}

Visualizer::~Visualizer() {
    if (!headless) {
        glDeleteFramebuffers(1, &framebufferID);
        glDeleteTextures(1, &depthTexture);

        glDeleteBuffers((GLsizei) face_index_buffer.size(), face_index_buffer.data());
        glDeleteBuffers((GLsizei) vertex_buffer.size(), vertex_buffer.data());
        glDeleteBuffers((GLsizei) uv_buffer.size(), uv_buffer.data());

        glDeleteBuffers((GLsizei) color_buffer.size(), color_buffer.data());
        glDeleteTextures((GLsizei) color_texture_object.size(), color_texture_object.data());
        glDeleteBuffers((GLsizei) normal_buffer.size(), normal_buffer.data());
        glDeleteTextures((GLsizei) normal_texture_object.size(), normal_texture_object.data());
        glDeleteBuffers((GLsizei) texture_flag_buffer.size(), texture_flag_buffer.data());
        glDeleteTextures((GLsizei) texture_flag_texture_object.size(), texture_flag_texture_object.data());
        glDeleteBuffers((GLsizei) texture_ID_buffer.size(), texture_ID_buffer.data());
        glDeleteTextures((GLsizei) texture_ID_texture_object.size(), texture_ID_texture_object.data());
        glDeleteBuffers((GLsizei) coordinate_flag_buffer.size(), coordinate_flag_buffer.data());
        glDeleteTextures((GLsizei) coordinate_flag_texture_object.size(), coordinate_flag_texture_object.data());
        glDeleteBuffers((GLsizei) hidden_flag_buffer.size(), hidden_flag_buffer.data());
        glDeleteTextures((GLsizei) hidden_flag_texture_object.size(), hidden_flag_texture_object.data());

        // Clean up texture array and UV rescaling resources
        if (texArray != 0) {
            glDeleteTextures(1, &texArray);
        }
        glDeleteBuffers(1, &uv_rescale_buffer);
        glDeleteTextures(1, &uv_rescale_texture_object);

        glfwDestroyWindow(scast<GLFWwindow *>(window));
        glfwTerminate();
    }
}

void Visualizer::enableMessages() {
    message_flag = true;
}

void Visualizer::disableMessages() {
    message_flag = false;
}

void Visualizer::setCameraPosition(const helios::vec3 &cameraPosition, const helios::vec3 &lookAt) {
    camera_eye_location = cameraPosition;
    camera_lookat_center = lookAt;
}

void Visualizer::setCameraPosition(const helios::SphericalCoord &cameraAngle, const helios::vec3 &lookAt) {
    camera_lookat_center = lookAt;
    camera_eye_location = camera_lookat_center + sphere2cart(cameraAngle);
}

void Visualizer::setCameraFieldOfView(float angle_FOV) {
    camera_FOV = angle_FOV;
}

void Visualizer::setLightDirection(const helios::vec3 &direction) {
    light_direction = direction / direction.magnitude();
    primaryShader.setLightDirection(direction);
}

void Visualizer::setLightingModel(LightingModel lightingmodel) {
    for (auto &i: primaryLightingModel) {
        i = lightingmodel;
    }
}

void Visualizer::setLightIntensityFactor(float lightintensityfactor) {
    lightintensity = lightintensityfactor;
}

void Visualizer::setBackgroundColor(const helios::RGBcolor &color) {
    backgroundColor = color;
}

void Visualizer::hideWatermark() {
    isWatermarkVisible = false;
    if (watermark_ID != 0) {
        geometry_handler.deleteGeometry(watermark_ID);
        watermark_ID = 0;
    }
}

void Visualizer::showWatermark() {
    isWatermarkVisible = true;
    updateWatermark();
}

void Visualizer::updatePerspectiveTransformation(bool shadow) {
    float dist = glm::distance(glm_vec3(camera_lookat_center), glm_vec3(camera_eye_location));
    float nearPlane = std::max(0.1f, 0.05f * dist); // avoid 0
    if (shadow) {
        float farPlane = std::max(5.f * camera_eye_location.z, 2.0f * dist);
        cameraProjectionMatrix = glm::perspective(glm::radians(camera_FOV), float(Wframebuffer) / float(Hframebuffer), nearPlane, farPlane);
    } else {
        cameraProjectionMatrix = glm::infinitePerspective(glm::radians(camera_FOV), float(Wframebuffer) / float(Hframebuffer), nearPlane);
    }
    cameraViewMatrix = glm::lookAt(glm_vec3(camera_eye_location), glm_vec3(camera_lookat_center), glm::vec3(0, 0, 1));

    perspectiveTransformationMatrix = cameraProjectionMatrix * cameraViewMatrix;
}

void Visualizer::updateCustomTransformation(const glm::mat4 &matrix) {
    customTransformationMatrix = matrix;
}

void Visualizer::enableColorbar() {
    colorbar_flag = 2;
}

void Visualizer::disableColorbar() {
    if (!colorbar_IDs.empty()) {
        geometry_handler.deleteGeometry(colorbar_IDs);
        colorbar_IDs.clear();
    }
    colorbar_flag = 1;
}

void Visualizer::setColorbarPosition(vec3 position) {
    if (position.x < 0 || position.x > 1 || position.y < 0 || position.y > 1 || position.z < -1 || position.z > 1) {
        helios_runtime_error("ERROR (Visualizer::setColorbarPosition): position is out of range.  Coordinates must be: 0<x<1, 0<y<1, -1<z<1.");
    }
    colorbar_position = position;
}

void Visualizer::setColorbarSize(vec2 size) {
    if (size.x < 0 || size.x > 1 || size.y < 0 || size.y > 1) {
        helios_runtime_error("ERROR (Visualizer::setColorbarSize): Size must be greater than 0 and less than the window size (i.e., 1).");
    }
    colorbar_size = size;
}

void Visualizer::setColorbarRange(float cmin, float cmax) {
    if (message_flag && cmin > cmax) {
        std::cerr << "WARNING (Visualizer::setColorbarRange): Maximum colorbar value must be greater than minimum value...Ignoring command." << std::endl;
        return;
    }
    colorbar_min = cmin;
    colorbar_max = cmax;
}

void Visualizer::setColorbarTicks(const std::vector<float> &ticks) {
    // check that vector is not empty
    if (ticks.empty()) {
        helios_runtime_error("ERROR (Visualizer::setColorbarTicks): Colorbar ticks vector is empty.");
    }

    // Check that ticks are monotonically increasing
    for (int i = 1; i < ticks.size(); i++) {
        if (ticks.at(i) <= ticks.at(i - 1)) {
            helios_runtime_error("ERROR (Visualizer::setColorbarTicks): Colorbar ticks must be monotonically increasing.");
        }
    }

    // Check that ticks are within the range of colorbar values
    for (int i = ticks.size() - 1; i >= 0; i--) {
        if (ticks.at(i) < colorbar_min) {
            colorbar_min = ticks.at(i);
        }
    }
    for (float tick: ticks) {
        if (tick > colorbar_max) {
            colorbar_max = tick;
        }
    }

    colorbar_ticks = ticks;
}

void Visualizer::setColorbarTitle(const char *title) {
    colorbar_title = title;
}

void Visualizer::setColorbarFontSize(uint font_size) {
    if (font_size <= 0) {
        helios_runtime_error("ERROR (Visualizer::setColorbarFontSize): Font size must be greater than zero.");
    }
    colorbar_fontsize = font_size;
}

void Visualizer::setColorbarFontColor(RGBcolor fontcolor) {
    colorbar_fontcolor = fontcolor;
}

void Visualizer::setColormap(Ctable colormap_name) {
    if (colormap_name == COLORMAP_HOT) {
        colormap_current = colormap_hot;
    } else if (colormap_name == COLORMAP_COOL) {
        colormap_current = colormap_cool;
    } else if (colormap_name == COLORMAP_LAVA) {
        colormap_current = colormap_lava;
    } else if (colormap_name == COLORMAP_RAINBOW) {
        colormap_current = colormap_rainbow;
    } else if (colormap_name == COLORMAP_PARULA) {
        colormap_current = colormap_parula;
    } else if (colormap_name == COLORMAP_GRAY) {
        colormap_current = colormap_gray;
    } else if (colormap_name == COLORMAP_CUSTOM) {
        helios_runtime_error("ERROR (Visualizer::setColormap): Setting a custom colormap requires calling setColormap with additional arguments defining the colormap.");
    } else {
        helios_runtime_error("ERROR (Visualizer::setColormap): Invalid colormap.");
    }
}

void Visualizer::setColormap(const std::vector<RGBcolor> &colors, const std::vector<float> &divisions) {
    if (colors.size() != divisions.size()) {
        helios_runtime_error("ERROR (Visualizer::setColormap): The number of colors must be equal to the number of divisions.");
    }

    Colormap colormap_custom(colors, divisions, 100, 0, 1);

    colormap_current = colormap_custom;
}

Colormap Visualizer::getCurrentColormap() const {
    return colormap_current;
}

glm::mat4 Visualizer::computeShadowDepthMVP() const {
    glm::vec3 lightDir = -glm::normalize(glm_vec3(light_direction));

    const float margin = 0.01;

    // Get the eight corners of the camera frustum in world space (NDC cube corners → clip → view → world)

    // NDC cube
    static const std::array<glm::vec4, 8> ndcCorners = {glm::vec4(-1, -1, -1, 1), glm::vec4(+1, -1, -1, 1), glm::vec4(+1, +1, -1, 1), glm::vec4(-1, +1, -1, 1),
                                                        glm::vec4(-1, -1, +1, 1), glm::vec4(+1, -1, +1, 1), glm::vec4(+1, +1, +1, 1), glm::vec4(-1, +1, +1, 1)};

    glm::mat4 invCam = glm::inverse(this->perspectiveTransformationMatrix);

    std::array<glm::vec3, 8> frustumWs;
    for (std::size_t i = 0; i < 8; i++) {
        glm::vec4 ws = invCam * ndcCorners[i];
        frustumWs[i] = glm::vec3(ws) / ws.w;
    }

    // Build a light-view matrix (orthographic, directional light) We choose an arbitrary but stable "up" vector.
    glm::vec3 lightUp(0.0f, 1.0f, 0.0f);
    if (glm::abs(glm::dot(lightUp, lightDir)) > 0.9f) // almost collinear
        lightUp = glm::vec3(1, 0, 0);

    // Position the "camera" that generates the shadow map so that every
    // frustum corner is in front of it.  We place it on the negative light
    // direction, centered on the frustum’s centroid.
    glm::vec3 centroid(0);
    for (auto &c: frustumWs)
        centroid += c;
    centroid /= 8.0f;

    glm::vec3 lightPos = centroid - lightDir * 100.0f; // 100 is arbitrary,
    // we will tighten z
    glm::mat4 lightView = glm::lookAt(lightPos, centroid, lightUp);

    // Transform frustum corners to light space and find min/max extents
    glm::vec3 minL(std::numeric_limits<float>::infinity());
    glm::vec3 maxL(-std::numeric_limits<float>::infinity());

    for (auto &c: frustumWs) {
        glm::vec3 p = glm::vec3(lightView * glm::vec4(c, 1));
        minL = glm::min(minL, p);
        maxL = glm::max(maxL, p);
    }

    // Build orthographic projection that exactly fits the camera frustum and enlarge slightly to avoid clipping due to kernel offsets.
    glm::vec3 extent = maxL - minL;
    minL -= extent * margin;
    maxL += extent * margin;

    float zNear = -maxL.z; // light space points toward -z
    float zFar = -minL.z;

    glm::mat4 lightProj = glm::ortho(minL.x, maxL.x, minL.y, maxL.y, zNear, zFar);

    // Transform into [0,1] texture space (bias matrix)
    const glm::mat4 bias(0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f);

    return bias * lightProj * lightView;
}

Visualizer::Texture::Texture(const std::string &texture_file, uint textureID, const helios::uint2 &maximum_texture_size, bool loadalphaonly) : texture_file(texture_file), glyph(), textureID(textureID) {
#ifdef HELIOS_DEBUG
    if (loadalphaonly) {
        assert(validateTextureFile(texture_file, true));
    } else {
        assert(validateTextureFile(texture_file));
    }
#endif

    //--- Load the Texture ----//

    if (loadalphaonly) {
        num_channels = 1;
    } else {
        num_channels = 4;
    }

    std::vector<unsigned char> image_data;

    if (texture_file.substr(texture_file.find_last_of('.') + 1) == "png") {
        read_png_file(texture_file.c_str(), image_data, texture_resolution.y, texture_resolution.x);
    } else { // JPEG
        read_JPEG_file(texture_file.c_str(), image_data, texture_resolution.y, texture_resolution.x);
    }

    texture_data = std::move(image_data);

    // If the texture image is too large, resize it
    if (texture_resolution.x > maximum_texture_size.x || texture_resolution.y > maximum_texture_size.y) {
        const uint2 new_texture_resolution(std::min(texture_resolution.x, maximum_texture_size.x), std::min(texture_resolution.y, maximum_texture_size.y));
        resizeTexture(new_texture_resolution);
    }
}

Visualizer::Texture::Texture(const Glyph *glyph_ptr, uint textureID, const helios::uint2 &maximum_texture_size) : textureID(textureID) {
    assert(glyph_ptr != nullptr);

    glyph = *glyph_ptr;

    texture_resolution = glyph_ptr->size;

    // Texture only has 1 channel, and contains transparency data
    texture_data.resize(texture_resolution.x * texture_resolution.y);
    for (int j = 0; j < texture_resolution.y; j++) {
        for (int i = 0; i < texture_resolution.x; i++) {
            texture_data[i + j * texture_resolution.x] = glyph_ptr->data.at(j).at(i);
        }
    }

    // If the texture image is too large, resize it
    if (texture_resolution.x > maximum_texture_size.x || texture_resolution.y > maximum_texture_size.y) {
        const uint2 new_texture_resolution(std::min(texture_resolution.x, maximum_texture_size.x), std::min(texture_resolution.y, maximum_texture_size.y));
        resizeTexture(new_texture_resolution);
    }

    num_channels = 1;
}

Visualizer::Texture::Texture(const std::vector<unsigned char> &pixel_data, uint textureID, const helios::uint2 &image_resolution, const helios::uint2 &maximum_texture_size) : textureID(textureID) {
#ifdef HELIOS_DEBUG
    assert(pixel_data.size() == 4u * image_resolution.x * image_resolution.y);
#endif

    texture_data = pixel_data;
    texture_resolution = image_resolution;
    num_channels = 4;

    // If the texture image is too large, resize it
    if (texture_resolution.x > maximum_texture_size.x || texture_resolution.y > maximum_texture_size.y) {
        const uint2 new_texture_resolution(std::min(texture_resolution.x, maximum_texture_size.x), std::min(texture_resolution.y, maximum_texture_size.y));
        resizeTexture(new_texture_resolution);
    }
}

void Visualizer::Texture::resizeTexture(const helios::uint2 &new_image_resolution) {
    int old_width = texture_resolution.x;
    int old_height = texture_resolution.y;
    int new_width = new_image_resolution.x;
    int new_height = new_image_resolution.y;

    std::vector<unsigned char> new_data(new_width * new_height * num_channels);

    // map each new pixel to a floating-point src coordinate
    float x_ratio = scast<float>(old_width) / scast<float>(new_width);
    float y_ratio = scast<float>(old_height) / scast<float>(new_height);

    for (int y = 0; y < new_height; ++y) {
        float srcY = y * y_ratio;
        int y0 = std::min(scast<int>(std::floor(srcY)), old_height - 1);
        int y1 = std::min(y0 + 1, old_height - 1);
        float dy = srcY - y0;

        for (int x = 0; x < new_width; ++x) {
            float srcX = x * x_ratio;
            int x0 = std::min(scast<int>(std::floor(srcX)), old_width - 1);
            int x1 = std::min(x0 + 1, old_width - 1);
            float dx = srcX - x0;

            // for each channel, fetch 4 neighbors and lerp
            for (int c = 0; c < num_channels; ++c) {
                float p00 = texture_data[(y0 * old_width + x0) * num_channels + c];
                float p10 = texture_data[(y0 * old_width + x1) * num_channels + c];
                float p01 = texture_data[(y1 * old_width + x0) * num_channels + c];
                float p11 = texture_data[(y1 * old_width + x1) * num_channels + c];

                float top = p00 * (1.f - dx) + p10 * dx;
                float bottom = p01 * (1.f - dx) + p11 * dx;
                float value = top * (1.f - dy) + bottom * dy;

                new_data[(y * new_width + x) * num_channels + c] = scast<unsigned char>(clamp(std::round(value + 0.5f), 0.f, 255.f));
            }
        }
    }

    texture_data = std::move(new_data);
    texture_resolution = new_image_resolution;
}


uint Visualizer::registerTextureImage(const std::string &texture_file) {
#ifdef HELIOS_DEBUG
    // assert( validateTextureFile(texture_file) );
#endif

    for (const auto &[textureID, texture]: texture_manager) {
        if (texture.texture_file == texture_file) {
            // if it does, return its texture ID
            return textureID;
        }
    }

    const uint textureID = texture_manager.size();

    texture_manager.try_emplace(textureID, texture_file, textureID, this->maximum_texture_size, false);
    textures_dirty = true;

    return textureID;
}

uint Visualizer::registerTextureImage(const std::vector<unsigned char> &texture_data, const helios::uint2 &image_resolution) {
#ifdef HELIOS_DEBUG
    assert(!texture_data.empty() && texture_data.size() == 4 * image_resolution.x * image_resolution.y);
#endif

    const uint textureID = texture_manager.size();

    texture_manager.try_emplace(textureID, texture_data, textureID, image_resolution, this->maximum_texture_size);
    textures_dirty = true;

    return textureID;
}

uint Visualizer::registerTextureTransparencyMask(const std::string &texture_file) {
#ifdef HELIOS_DEBUG
    assert(validateTextureFile(texture_file));
#endif

    for (const auto &[textureID, texture]: texture_manager) {
        if (texture.texture_file == texture_file) {
            // if it does, return its texture ID
            return textureID;
        }
    }

    const uint textureID = texture_manager.size();

    texture_manager.try_emplace(textureID, texture_file, textureID, this->maximum_texture_size, true);
    textures_dirty = true;

    return textureID;
}

uint Visualizer::registerTextureGlyph(const Glyph *glyph) {

    const uint textureID = texture_manager.size();

    texture_manager.try_emplace(textureID, glyph, textureID, this->maximum_texture_size);
    textures_dirty = true;

    return textureID;
}

helios::uint2 Visualizer::getTextureResolution(uint textureID) const {
    if (texture_manager.find(textureID) == texture_manager.end()) {
    }
    return texture_manager.at(textureID).texture_resolution;
}

bool validateTextureFile(const std::string &texture_file, bool pngonly) {
    const std::filesystem::path p(texture_file);

    // Check that the file exists and is a regular file
    if (!std::filesystem::exists(p) || !std::filesystem::is_regular_file(p)) {
        return false;
    }

    // Extract and lowercase the extension
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char c) { return scast<char>(std::tolower(c)); });

    // Verify it's .png, .jpg or .jpeg
    if (pngonly) {
        if (ext != ".png") {
            return false;
        }
    } else {
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") {
            return false;
        }
    }

    return true;
}

void *Visualizer::getWindow() const {
    return window;
}

std::vector<uint> Visualizer::getFrameBufferSize() const {
    return {Wframebuffer, Hframebuffer};
}

void Visualizer::setFrameBufferSize(int width, int height) {
    Wframebuffer = width;
    Hframebuffer = height;
}

helios::RGBcolor Visualizer::getBackgroundColor() const {
    return backgroundColor;
}

Shader Visualizer::getPrimaryShader() const {
    return primaryShader;
}

std::vector<helios::vec3> Visualizer::getCameraPosition() const {
    return {camera_lookat_center, camera_eye_location};
}

glm::mat4 Visualizer::getPerspectiveTransformationMatrix() const {
    return perspectiveTransformationMatrix;
}

glm::mat4 Visualizer::getViewMatrix() const {
    vec3 forward = camera_lookat_center - camera_eye_location;
    forward = forward.normalize();

    vec3 right = cross(vec3(0, 0, 1), forward);
    right = right.normalize();

    vec3 up = cross(forward, right);
    up = up.normalize();

    glm::vec3 camera_pos{camera_eye_location.x, camera_eye_location.y, camera_eye_location.z};
    glm::vec3 lookat_pos{camera_lookat_center.x, camera_lookat_center.y, camera_lookat_center.z};
    glm::vec3 up_vec{up.x, up.y, up.z};

    return glm::lookAt(camera_pos, lookat_pos, up_vec);
}

std::vector<Visualizer::LightingModel> Visualizer::getPrimaryLightingModel() {
    return primaryLightingModel;
}

uint Visualizer::getDepthTexture() const {
    return depthTexture;
}
