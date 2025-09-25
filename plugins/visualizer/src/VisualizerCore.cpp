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

int write_JPEG_file(const char *filename, uint width, uint height, bool buffers_swapped_since_render, bool print_messages) {
    if (print_messages) {
        std::cout << "writing JPEG image: " << filename << std::endl;
    }

    // Validate framebuffer completeness
    GLenum framebuffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (framebuffer_status != GL_FRAMEBUFFER_COMPLETE) {
        helios_runtime_error("ERROR (write_JPEG_file): Framebuffer is not complete (status: " + std::to_string(framebuffer_status) + ")");
    }

    // Clear any existing OpenGL errors
    while (glGetError() != GL_NO_ERROR) {}

    const size_t bsize = 3 * width * height;
    std::vector<GLubyte> screen_shot_trans;
    screen_shot_trans.resize(bsize);

    // Set proper pixel alignment for reliable reading
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // Deterministic buffer selection based on swap state tracking
    // If buffers have been swapped since last render, newest content is in GL_FRONT
    // If buffers have NOT been swapped since last render, newest content is in GL_BACK (current render target)
    GLenum error;
    if (buffers_swapped_since_render) {
        glReadBuffer(GL_FRONT);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Fallback to back buffer if front buffer fails
            glReadBuffer(GL_BACK);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                helios_runtime_error("ERROR (write_JPEG_file): Cannot set read buffer (error: " + std::to_string(error) + ")");
            }
        }
    } else {
        glReadBuffer(GL_BACK);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            // Fallback to front buffer if back buffer fails
            glReadBuffer(GL_FRONT);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                helios_runtime_error("ERROR (write_JPEG_file): Cannot set read buffer (error: " + std::to_string(error) + ")");
            }
        }
    }

    // Ensure all rendering commands complete before reading
    glFinish();

    // Read pixels with error checking
    glReadPixels(0, 0, scast<GLsizei>(width), scast<GLsizei>(height), GL_RGB, GL_UNSIGNED_BYTE, &screen_shot_trans[0]);
    error = glGetError();
    if (error != GL_NO_ERROR) {
        helios_runtime_error("ERROR (write_JPEG_file): glReadPixels failed (error: " + std::to_string(error) + ")");
    }

    // Check if we got all black pixels (common failure mode)
    bool all_black = true;
    for (size_t i = 0; i < bsize && all_black; i++) {
        if (screen_shot_trans[i] != 0) {
            all_black = false;
        }
    }

    if (all_black) {
        std::cout << "WARNING (write_JPEG_file): All pixels are black - this may indicate a timing or buffer issue" << std::endl;
    }

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
    GLenum glew_result = glewInit();
    if (glew_result != GLEW_OK) {
        std::string error_msg = "ERROR (Visualizer): Failed to initialize GLEW. ";
        error_msg += "GLEW error: " + std::string((const char*)glewGetErrorString(glew_result));
        helios_runtime_error(error_msg);
    }

    // Check for and handle the expected GL_INVALID_ENUM error from glewExperimental
    // This is a known issue with glewExperimental on some OpenGL implementations
    GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR && gl_error != GL_INVALID_ENUM) {
        std::string error_msg = "ERROR (Visualizer): Unexpected OpenGL error after GLEW initialization: ";
        error_msg += std::to_string(gl_error);
        helios_runtime_error(error_msg);
    }
}

void Visualizer::createOffscreenContext() {
    // Create an offscreen context for headless rendering
    // This avoids the need for a display server on CI systems
    //
    // NOTE: On macOS, you may see OpenGL warnings like:
    // "UNSUPPORTED (log once): POSSIBLE ISSUE: unit 6 GLD_TEXTURE_INDEX_2D is unloadable..."
    // This is a known macOS OpenGL driver warning in headless mode and is harmless.

    // Check for environment variables that indicate CI/headless operation
    const char* ci_env = std::getenv("CI");
    const char* display_env = std::getenv("DISPLAY");
    const char* force_offscreen = std::getenv("HELIOS_FORCE_OFFSCREEN");

    bool is_ci = (ci_env != nullptr && std::string(ci_env) == "true");
    bool has_display = (display_env != nullptr && std::strlen(display_env) > 0);
    bool force_software = (force_offscreen != nullptr && std::string(force_offscreen) == "1");

    // Configure GLFW window hints for optimal CI compatibility
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);

#if __APPLE__
    // On macOS, configure for better CI compatibility
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);  // Disable double buffering for offscreen
    if (is_ci || force_software) {
        // In CI environments, prefer compatibility profile for better software rendering support
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    }
#elif __linux__
    // On Linux, detect and configure for software rendering if needed
    if (is_ci && !has_display) {
        // Likely a headless CI environment - use software rendering hints
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
        glfwWindowHint(GLFW_SAMPLES, 0);  // Disable multisampling for software rendering
    }
#elif _WIN32
    // On Windows, configure for CI environments
    if (is_ci || force_software) {
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    }
#endif

    // Create a minimal 1x1 window for the OpenGL context
    // This window will be invisible but provides the necessary OpenGL context
    GLFWwindow* _window = glfwCreateWindow(1, 1, "Helios Offscreen", nullptr, nullptr);

    if (_window == nullptr) {
        // Failed to create even an offscreen context - provide platform-specific guidance
        std::string error_msg = "ERROR (Visualizer::createOffscreenContext): Unable to create OpenGL context for headless rendering.\n";
        error_msg += "This typically occurs in CI environments without GPU drivers or display servers.\n";
        error_msg += "Platform-specific solutions:\n";

#if __APPLE__
        error_msg += "-- macOS CI: Install Xcode command line tools and ensure graphics frameworks are available\n";
        error_msg += "-- Try setting HELIOS_FORCE_OFFSCREEN=1 environment variable\n";
        error_msg += "-- Consider using a macOS runner with graphics support\n";
#elif __linux__
        error_msg += "-- Linux CI: Install virtual display server: apt-get install xvfb\n";
        error_msg += "-- Start virtual display: Xvfb :99 -screen 0 1024x768x24 &\n";
        error_msg += "-- Set display variable: export DISPLAY=:99\n";
        error_msg += "-- Install Mesa software rendering: apt-get install mesa-utils libgl1-mesa-dev\n";
        error_msg += "-- Force software rendering: export LIBGL_ALWAYS_SOFTWARE=1\n";
#elif _WIN32
        error_msg += "-- Windows CI: Ensure OpenGL drivers are available\n";
        error_msg += "-- Install Mesa3D for software rendering\n";
        error_msg += "-- Try using a Windows runner with graphics support\n";
#endif
        error_msg += "-- Alternative: Skip visualizer tests in CI with conditional test execution\n";
        error_msg += "-- Set HELIOS_FORCE_OFFSCREEN=1 to attempt software rendering";

        helios_runtime_error(error_msg);
    }

    glfwMakeContextCurrent(_window);
    window = (void*)_window;

    // Verify the context is current and functional
    const char* gl_version = (const char*)glGetString(GL_VERSION);
    if (gl_version == nullptr) {
        glfwDestroyWindow(_window);
        helios_runtime_error("ERROR (Visualizer::createOffscreenContext): Failed to obtain OpenGL version. Context creation failed.");
    }

    // Set framebuffer dimensions to match requested display size for offscreen rendering
    Wframebuffer = Wdisplay;
    Hframebuffer = Hdisplay;

    // Initialize offscreen framebuffer for true headless rendering
    // Delay this until after GLEW initialization
    // setupOffscreenFramebuffer();

    // Note: In headless mode, we won't set up window callbacks since there's no user interaction
}

void Visualizer::setupOffscreenFramebuffer() {
    // Create a complete framebuffer for offscreen rendering with both color and depth attachments
    // This enables full OpenGL testing in CI environments

    // Validate OpenGL context and required extensions are available
    const char* gl_version = (const char*)glGetString(GL_VERSION);
    if (gl_version == nullptr) {
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): OpenGL context is not valid - unable to retrieve version string.");
    }

    // Check for framebuffer object support (OpenGL 3.0+ or ARB_framebuffer_object extension)
    if (!GLEW_VERSION_3_0 && !GLEW_ARB_framebuffer_object) {
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): OpenGL context does not support framebuffer objects (requires OpenGL 3.0+ or ARB_framebuffer_object extension).");
    }

    // Validate framebuffer dimensions (must be positive and within reasonable limits)
    if (Wframebuffer == 0 || Hframebuffer == 0) {
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Invalid framebuffer dimensions (" + std::to_string(Wframebuffer) + "x" + std::to_string(Hframebuffer) + "). Dimensions must be positive.");
    }

    // Get maximum texture size to validate our request
    GLint max_texture_size;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    if (static_cast<GLint>(Wframebuffer) > max_texture_size || static_cast<GLint>(Hframebuffer) > max_texture_size) {
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Requested framebuffer size (" + std::to_string(Wframebuffer) + "x" + std::to_string(Hframebuffer) + ") exceeds maximum texture size (" + std::to_string(max_texture_size) + ").");
    }

    // Ensure we start with a clean OpenGL state
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): OpenGL errors detected before framebuffer setup.");
    }

    // Generate the framebuffer with error checking
    glGenFramebuffers(1, &offscreenFramebufferID);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR || offscreenFramebufferID == 0) {
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Failed to generate framebuffer object. OpenGL error: " + std::to_string(error));
    }

    glBindFramebuffer(GL_FRAMEBUFFER, offscreenFramebufferID);
    error = glGetError();
    if (error != GL_NO_ERROR) {
        glDeleteFramebuffers(1, &offscreenFramebufferID);
        offscreenFramebufferID = 0;
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Failed to bind framebuffer object. OpenGL error: " + std::to_string(error));
    }

    // Create color texture attachment with comprehensive error checking
    glGenTextures(1, &offscreenColorTexture);
    error = glGetError();
    if (error != GL_NO_ERROR || offscreenColorTexture == 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &offscreenFramebufferID);
        offscreenFramebufferID = 0;
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Failed to generate color texture. OpenGL error: " + std::to_string(error));
    }

    glBindTexture(GL_TEXTURE_2D, offscreenColorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, static_cast<GLsizei>(Wframebuffer), static_cast<GLsizei>(Hframebuffer), 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    error = glGetError();
    if (error != GL_NO_ERROR) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteTextures(1, &offscreenColorTexture);
        glDeleteFramebuffers(1, &offscreenFramebufferID);
        offscreenColorTexture = 0;
        offscreenFramebufferID = 0;
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Failed to create color texture storage. OpenGL error: " + std::to_string(error) + ". This may indicate insufficient GPU memory or unsupported texture format.");
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, offscreenColorTexture, 0);

    // Create depth texture attachment with error checking
    glGenTextures(1, &offscreenDepthTexture);
    error = glGetError();
    if (error != GL_NO_ERROR || offscreenDepthTexture == 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteTextures(1, &offscreenColorTexture);
        glDeleteFramebuffers(1, &offscreenFramebufferID);
        offscreenColorTexture = 0;
        offscreenFramebufferID = 0;
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Failed to generate depth texture. OpenGL error: " + std::to_string(error));
    }

    glBindTexture(GL_TEXTURE_2D, offscreenDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, static_cast<GLsizei>(Wframebuffer), static_cast<GLsizei>(Hframebuffer), 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    error = glGetError();
    if (error != GL_NO_ERROR) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteTextures(1, &offscreenDepthTexture);
        glDeleteTextures(1, &offscreenColorTexture);
        glDeleteFramebuffers(1, &offscreenFramebufferID);
        offscreenDepthTexture = 0;
        offscreenColorTexture = 0;
        offscreenFramebufferID = 0;
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): Failed to create depth texture storage. OpenGL error: " + std::to_string(error) + ". This may indicate insufficient GPU memory or unsupported depth format.");
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, offscreenDepthTexture, 0);

    // Check framebuffer completeness with detailed error reporting
    GLenum framebuffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (framebuffer_status != GL_FRAMEBUFFER_COMPLETE) {
        // Clean up before throwing error
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        cleanupOffscreenFramebuffer();

        std::string error_message = "ERROR (Visualizer::setupOffscreenFramebuffer): Offscreen framebuffer is not complete. Status: ";
        switch (framebuffer_status) {
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                error_message += "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT - One or more attachment points are not framebuffer attachment complete";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                error_message += "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT - No images are attached to the framebuffer";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                error_message += "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER - Draw buffer configuration error";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
                error_message += "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER - Read buffer configuration error";
                break;
            case GL_FRAMEBUFFER_UNSUPPORTED:
                error_message += "GL_FRAMEBUFFER_UNSUPPORTED - Combination of internal formats is not supported";
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
                error_message += "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE - Multisample configuration error";
                break;
            default:
                error_message += "Unknown framebuffer error (code: " + std::to_string(framebuffer_status) + ")";
                break;
        }
        error_message += ". This typically occurs in virtualized graphics environments or with limited OpenGL driver support.";
        helios_runtime_error(error_message);
    }

    // Restore default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Final error check
    if (!checkerrors()) {
        cleanupOffscreenFramebuffer();
        helios_runtime_error("ERROR (Visualizer::setupOffscreenFramebuffer): OpenGL errors occurred during offscreen framebuffer setup completion.");
    }
}

void Visualizer::cleanupOffscreenFramebuffer() {
    // Clean up offscreen rendering resources safely
    // Only attempt cleanup if we have a valid OpenGL context
    if (window != nullptr) {
        if (offscreenFramebufferID != 0) {
            glDeleteFramebuffers(1, &offscreenFramebufferID);
            offscreenFramebufferID = 0;
        }
        if (offscreenColorTexture != 0) {
            glDeleteTextures(1, &offscreenColorTexture);
            offscreenColorTexture = 0;
        }
        if (offscreenDepthTexture != 0) {
            glDeleteTextures(1, &offscreenDepthTexture);
            offscreenDepthTexture = 0;
        }
    }
}

std::vector<helios::RGBcolor> Visualizer::readOffscreenPixels() const {
    // Read pixels from the offscreen framebuffer for printWindow functionality

    if (offscreenFramebufferID == 0) {
        helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): No offscreen framebuffer available. "
                           "Ensure setupOffscreenFramebuffer() was called successfully in headless mode.");
    }

    // Validate framebuffer dimensions
    if (Wframebuffer == 0 || Hframebuffer == 0) {
        helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): Invalid framebuffer dimensions (" +
                           std::to_string(Wframebuffer) + "x" + std::to_string(Hframebuffer) +
                           "). This indicates the offscreen framebuffer was not properly initialized.");
    }

    // Check that we have a valid OpenGL context
    const char* gl_version = (const char*)glGetString(GL_VERSION);
    if (gl_version == nullptr) {
        helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): Invalid OpenGL context. "
                           "This indicates OpenGL initialization failed or the context was lost.");
    }

    // Clear any existing OpenGL errors
    while (glGetError() != GL_NO_ERROR) {}

    // Bind the offscreen framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, offscreenFramebufferID);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): Failed to bind offscreen framebuffer (OpenGL error: " +
                           std::to_string(error) + "). This indicates graphics driver issues or corrupted framebuffer.");
    }

    // Verify framebuffer is complete
    GLenum framebuffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (framebuffer_status != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): Framebuffer is not complete (status: " +
                           std::to_string(framebuffer_status) +
                           "). This indicates missing attachments or graphics driver incompatibility.");
    }

    // Calculate pixel data size with overflow protection
    const size_t pixel_count = static_cast<size_t>(Wframebuffer) * static_cast<size_t>(Hframebuffer);
    const size_t data_size = pixel_count * 3;

    // Check for potential overflow
    if (pixel_count > SIZE_MAX / 3 || data_size > SIZE_MAX / sizeof(unsigned char)) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): Framebuffer dimensions too large (" +
                           std::to_string(Wframebuffer) + "x" + std::to_string(Hframebuffer) +
                           "). This would cause memory allocation overflow.");
    }

    std::vector<helios::RGBcolor> pixels;
    try {
        // Read pixels from the color attachment
        std::vector<unsigned char> pixel_data(data_size);
        glReadPixels(0, 0, static_cast<GLsizei>(Wframebuffer), static_cast<GLsizei>(Hframebuffer), GL_RGB, GL_UNSIGNED_BYTE, pixel_data.data());

        error = glGetError();
        if (error != GL_NO_ERROR) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): Failed to read pixels from framebuffer (OpenGL error: " +
                               std::to_string(error) + "). This indicates graphics driver issues or framebuffer format problems.");
        }

        // Convert to RGBcolor format with bounds checking
        pixels.reserve(pixel_count);
        for (size_t i = 0; i + 2 < pixel_data.size(); i += 3) {
            float r = pixel_data[i] / 255.0f;
            float g = pixel_data[i + 1] / 255.0f;
            float b = pixel_data[i + 2] / 255.0f;
            pixels.emplace_back(make_RGBcolor(r, g, b));
        }
    } catch (const std::exception& e) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        helios_runtime_error("ERROR (Visualizer::readOffscreenPixels): Memory allocation or conversion failed: " +
                           std::string(e.what()) + ". This may indicate insufficient memory or data corruption.");
    }

    // Restore default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return pixels;
}

void Visualizer::renderToOffscreenBuffer() {
    // Switch rendering target to offscreen framebuffer
    if (offscreenFramebufferID != 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, offscreenFramebufferID);
        glViewport(0, 0, Wframebuffer, Hframebuffer);
    }
}

void Visualizer::initialize(uint window_width_pixels, uint window_height_pixels, int aliasing_samples, bool window_decorations, bool headless_mode) {
    Wdisplay = window_width_pixels;
    Hdisplay = window_height_pixels;

    // Check environment variables for automatic headless mode detection
    const char* force_offscreen = std::getenv("HELIOS_FORCE_OFFSCREEN");
    const char* ci_env = std::getenv("CI");
    const char* display_env = std::getenv("DISPLAY");

    bool should_force_headless = false;
    if (force_offscreen != nullptr && std::string(force_offscreen) == "1") {
        should_force_headless = true;
    } else if (ci_env != nullptr && std::string(ci_env) == "true") {
        // In CI environment, check if we have a display
#if __linux__
        if (display_env == nullptr || std::strlen(display_env) == 0) {
            should_force_headless = true;  // Linux CI without DISPLAY
        }
#elif __APPLE__ || _WIN32
        // On macOS and Windows CI, graphics might not be available
        should_force_headless = true;  // Can be overridden by explicit headless=false
#endif
    }

    // Final headless determination: explicit parameter OR environment-forced
    headless = headless_mode || should_force_headless;

    shadow_buffer_size = make_uint2(8192, 8192);

    maximum_texture_size = make_uint2(2048, 2048);

    texArray = 0;
    texture_array_layers = 0;
    textures_dirty = false;

    // Initialize offscreen rendering variables
    offscreenFramebufferID = 0;
    offscreenColorTexture = 0;
    offscreenDepthTexture = 0;

    message_flag = true;

    frame_counter = 0;

    buffers_swapped_since_render = false;

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
        createOffscreenContext();
    } else {
        // Regular windowed mode
        glfwWindowHint(GLFW_VISIBLE, 0); // Initially hidden, will show later if needed

        if (!window_decorations) {
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        }

        openWindow();
    }

    // Initialize GLEW - required for both headless and windowed modes
    glewExperimental = GL_TRUE; // Needed in core profile
    GLenum glew_result = glewInit();
    if (glew_result != GLEW_OK) {
        std::string error_msg = "ERROR (Visualizer::initialize): Failed to initialize GLEW. ";
        error_msg += "GLEW error: " + std::string((const char*)glewGetErrorString(glew_result));

        if (headless) {
            error_msg += "\nIn headless mode, this usually indicates:";
            error_msg += "\n- Missing or incompatible OpenGL drivers";
            error_msg += "\n- Virtual display server not properly configured";
            error_msg += "\n- VirtualGL or Mesa software rendering issues";
            error_msg += "\nConsider setting LIBGL_ALWAYS_SOFTWARE=1 for software rendering";
        }

        helios_runtime_error(error_msg);
    }

    // Check for and handle the expected GL_INVALID_ENUM error from glewExperimental
    // This is a known issue with glewExperimental on some OpenGL implementations
    GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR && gl_error != GL_INVALID_ENUM) {
        std::string error_msg = "ERROR (Visualizer): Unexpected OpenGL error after GLEW initialization: ";
        error_msg += std::to_string(gl_error);
        if (headless) {
            error_msg += "\nThis indicates a serious OpenGL context or driver issue in headless mode.";
        }
        helios_runtime_error(error_msg);
    }

    // Validate basic OpenGL functionality after GLEW initialization
    const char* gl_version = (const char*)glGetString(GL_VERSION);
    const char* gl_vendor = (const char*)glGetString(GL_VENDOR);
    const char* gl_renderer = (const char*)glGetString(GL_RENDERER);

    if (gl_version == nullptr || gl_vendor == nullptr || gl_renderer == nullptr) {
        helios_runtime_error("ERROR (Visualizer::initialize): OpenGL context is not functional - unable to query basic GL information. "
                             "This indicates a fundamental issue with OpenGL context creation or driver compatibility.");
    }

    // In debug mode or verbose CI, log the OpenGL information
    if (headless && (std::getenv("CI") != nullptr || std::getenv("HELIOS_DEBUG") != nullptr)) {
        std::cout << "OpenGL Version: " << gl_version << std::endl;
        std::cout << "OpenGL Vendor: " << gl_vendor << std::endl;
        std::cout << "OpenGL Renderer: " << gl_renderer << std::endl;
    }

    // Test basic OpenGL operations that are required for visualizer functionality
    GLint max_texture_size;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    GLenum validation_error = glGetError();
    if (validation_error != GL_NO_ERROR) {
        helios_runtime_error("ERROR (Visualizer::initialize): Basic OpenGL query operations failed (error: " + std::to_string(validation_error) + "). "
                             "This indicates the OpenGL context is not properly initialized or lacks required functionality.");
    }

    // Warn if texture size is unusually small (indicates software rendering or limited drivers)
    if (max_texture_size < 1024 && headless) {
        std::cerr << "WARNING (Visualizer::initialize): Maximum texture size is very small (" << max_texture_size
                  << "x" << max_texture_size << "). This may indicate software rendering or limited driver support." << std::endl;
    }

    // Final verification that we don't have accumulated OpenGL errors
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::initialize): OpenGL context initialization failed after GLEW setup and validation. "
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

    // Initialize offscreen framebuffer for headless mode after OpenGL context is fully set up
    if (headless) {
        setupOffscreenFramebuffer();
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

    // Initialize VBO's and texture buffers with comprehensive error checking
    constexpr size_t Ntypes = GeometryHandler::all_geometry_types.size();

    // Validate that we have a reasonable number of geometry types to prevent memory issues
    if (Ntypes == 0 || Ntypes > 1000) {
        helios_runtime_error("ERROR (Visualizer::initialize): Invalid number of geometry types (" + std::to_string(Ntypes) + "). "
                             "This indicates a configuration issue with GeometryHandler.");
    }

    try {
        // per-vertex data with error checking after each allocation
        face_index_buffer.resize(Ntypes);
        vertex_buffer.resize(Ntypes);
        uv_buffer.resize(Ntypes);

        // Generate per-vertex buffers with immediate error checking
        glGenBuffers((GLsizei) face_index_buffer.size(), face_index_buffer.data());
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate face index buffers. OpenGL error: " + std::to_string(error));
        }

        glGenBuffers((GLsizei) vertex_buffer.size(), vertex_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate vertex buffers. OpenGL error: " + std::to_string(error));
        }

        glGenBuffers((GLsizei) uv_buffer.size(), uv_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate UV buffers. OpenGL error: " + std::to_string(error));
        }

        // per-primitive data with error checking after each allocation
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

        // Generate per-primitive buffers and textures with comprehensive error checking
        glGenBuffers((GLsizei) color_buffer.size(), color_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate color buffers. OpenGL error: " + std::to_string(error));
        }

        glGenTextures((GLsizei) color_texture_object.size(), color_texture_object.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate color texture objects. OpenGL error: " + std::to_string(error));
        }

        glGenBuffers((GLsizei) normal_buffer.size(), normal_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate normal buffers. OpenGL error: " + std::to_string(error));
        }

        glGenTextures((GLsizei) normal_texture_object.size(), normal_texture_object.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate normal texture objects. OpenGL error: " + std::to_string(error));
        }

        glGenBuffers((GLsizei) texture_flag_buffer.size(), texture_flag_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate texture flag buffers. OpenGL error: " + std::to_string(error));
        }

        glGenTextures((GLsizei) texture_flag_texture_object.size(), texture_flag_texture_object.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate texture flag texture objects. OpenGL error: " + std::to_string(error));
        }

        glGenBuffers((GLsizei) texture_ID_buffer.size(), texture_ID_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate texture ID buffers. OpenGL error: " + std::to_string(error));
        }

        glGenTextures((GLsizei) texture_ID_texture_object.size(), texture_ID_texture_object.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate texture ID texture objects. OpenGL error: " + std::to_string(error));
        }

        glGenBuffers((GLsizei) coordinate_flag_buffer.size(), coordinate_flag_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate coordinate flag buffers. OpenGL error: " + std::to_string(error));
        }

        glGenTextures((GLsizei) coordinate_flag_texture_object.size(), coordinate_flag_texture_object.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate coordinate flag texture objects. OpenGL error: " + std::to_string(error));
        }

        glGenBuffers((GLsizei) hidden_flag_buffer.size(), hidden_flag_buffer.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate hidden flag buffers. OpenGL error: " + std::to_string(error));
        }

        glGenTextures((GLsizei) hidden_flag_texture_object.size(), hidden_flag_texture_object.data());
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate hidden flag texture objects. OpenGL error: " + std::to_string(error));
        }

        // Generate UV rescaling buffers
        glGenBuffers(1, &uv_rescale_buffer);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate UV rescale buffer. OpenGL error: " + std::to_string(error));
        }

        glGenTextures(1, &uv_rescale_texture_object);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (Visualizer::initialize): Failed to generate UV rescale texture object. OpenGL error: " + std::to_string(error));
        }

    } catch (const std::exception& e) {
        helios_runtime_error("ERROR (Visualizer::initialize): Exception during buffer allocation: " + std::string(e.what()) +
                             ". This may indicate insufficient memory or OpenGL driver issues.");
    }

    // Final verification that all buffer operations completed successfully
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer::initialize): OpenGL buffer creation failed with accumulated errors. "
                             "This indicates insufficient graphics memory or unsupported buffer operations in the current OpenGL context.");
    }

    //~~~~~~~~~~~~~ Load the Shaders ~~~~~~~~~~~~~~~~~~~//

    std::string primaryVertShader = helios::resolvePluginAsset("visualizer", "shaders/primaryShader.vert").string();
    std::string primaryFragShader = helios::resolvePluginAsset("visualizer", "shaders/primaryShader.frag").string();
    std::string shadowVertShader = helios::resolvePluginAsset("visualizer", "shaders/shadow.vert").string();
    std::string shadowFragShader = helios::resolvePluginAsset("visualizer", "shaders/shadow.frag").string();

    primaryShader.initialize(primaryVertShader.c_str(), primaryFragShader.c_str(), this);
    depthShader.initialize(shadowVertShader.c_str(), shadowFragShader.c_str(), this);

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
    // Clean up resources for both headless and windowed modes
    if (headless) {
        cleanupOffscreenFramebuffer();
    } else {
        if (framebufferID != 0) {
            glDeleteFramebuffers(1, &framebufferID);
        }
        if (depthTexture != 0) {
            glDeleteTextures(1, &depthTexture);
        }
    }

    // Clean up common OpenGL resources regardless of mode
    if (window != nullptr) {

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
