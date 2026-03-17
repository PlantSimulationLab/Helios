/** \file "VisualizerRendering.cpp" Visualizer rendering and display functions.

    Copyright (C) 2016-2026 Brian Bailey

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

// #include <chrono>

#include "Visualizer.h"

using namespace helios;

float dphi = 0.0;
float dtheta = 0.0;
float dx = 0.0;
float dy = 0.0;
float dz = 0.0;
float dx_m = 0.0;
float dy_m = 0.0;
float dscroll = 0.0;

void Visualizer::printWindow() {
    char outfile[100];
    if (context != nullptr) { // context has been given to visualizer via buildContextGeometry()
        Date date = context->getDate();
        Time time = context->getTime();
        std::snprintf(outfile, 100, "%02d-%02d-%4d_%02d-%02d-%02d_frame%d.jpg", date.day, date.month, date.year, time.hour, time.minute, time.second, frame_counter);
    } else {
        std::snprintf(outfile, 100, "frame%d.jpg", frame_counter);
    }
    frame_counter++;

    printWindow(outfile);
}

void Visualizer::printWindow(const char *outfile, const std::string &image_format) {

    // Save current navigation gizmo state and temporarily hide it for screenshot
    bool gizmo_was_visible = navigation_gizmo_enabled;
    if (gizmo_was_visible) {
        hideNavigationGizmo();
    }

    // Update the plot window to ensure latest rendering
    this->plotUpdate(true);

    std::string outfile_str = outfile;

    // Validate image format
    std::string format_lower = image_format;
    std::transform(format_lower.begin(), format_lower.end(), format_lower.begin(), ::tolower);

    bool is_png = (format_lower == "png");
    bool is_jpeg = (format_lower == "jpeg" || format_lower == "jpg");

    if (!is_png && !is_jpeg) {
        helios_runtime_error("ERROR (Visualizer::printWindow): Invalid image_format '" + image_format + "'. Must be 'jpeg', 'jpg', or 'png'.");
    }

    // Add or verify file extension
    std::string ext = getFileExtension(outfile_str);
    if (ext.empty()) {
        outfile_str += is_png ? ".png" : ".jpeg";
    } else {
        // Validate extension matches format (getFileExtension returns extension with leading dot, e.g., ".jpg")
        std::string ext_lower = ext;
        std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
        if (is_png && ext_lower != ".png") {
            helios_runtime_error("ERROR (Visualizer::printWindow): File extension '" + ext + "' does not match image_format 'png'.");
        } else if (is_jpeg && ext_lower != ".jpg" && ext_lower != ".jpeg") {
            helios_runtime_error("ERROR (Visualizer::printWindow): File extension '" + ext + "' does not match image_format 'jpeg'.");
        }
    }

    // Warn if transparent background requested with JPEG format
    if (background_is_transparent && is_jpeg && message_flag) {
        std::cerr << "WARNING (Visualizer::printWindow): Transparent background requested but JPEG format does not support transparency. Output will have opaque background." << std::endl;
    }

    // Ensure window is visible and rendering is complete
    if (window != nullptr && !headless) {
        // Check if window is minimized or occluded
        int window_attrib = glfwGetWindowAttrib((GLFWwindow *) window, GLFW_ICONIFIED);
        if (window_attrib == GLFW_TRUE) {
            std::cerr << "WARNING (printWindow): Window is minimized - screenshot may be unreliable" << std::endl;
        }

        glfwPollEvents();
    }

    // Temporarily delete the transparent background checkerboard rectangle for screenshot
    // (we want the actual scene with transparent background, not the checkerboard)
    bool had_transparent_background = background_is_transparent && background_rectangle_ID != 0;
    if (had_transparent_background) {
        // Delete the checkerboard completely
        geometry_handler.deleteGeometry(background_rectangle_ID);
        background_rectangle_ID = 0;

        // Update buffers to reflect the defragmentation
        transferBufferData();

        // Ensure GPU has finished processing before rendering
        glFinish();

        // Bind the appropriate framebuffer
        if (headless && offscreenFramebufferID != 0) {
            glBindFramebuffer(GL_FRAMEBUFFER, offscreenFramebufferID);
            glViewport(0, 0, Wframebuffer, Hframebuffer);
        } else {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, Wframebuffer, Hframebuffer);
            // In windowed mode, explicitly set draw buffer to back buffer
            glDrawBuffer(GL_BACK);
        }

        // Clear with transparent background
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set up shader and render
        primaryShader.useShader();
        updatePerspectiveTransformation(false);
        primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);
        primaryShader.setViewMatrix(cameraViewMatrix);
        primaryShader.setProjectionMatrix(cameraProjectionMatrix);
        primaryShader.enableTextureMaps();
        primaryShader.enableTextureMasks();
        primaryShader.setLightingModel(primaryLightingModel);
        primaryShader.setLightIntensity(lightintensity);

        render(false);

        // After re-rendering, the content is in the BACK buffer (not swapped yet)
        // Force read buffer to BACK for the screenshot
        if (!headless) {
            glReadBuffer(GL_BACK);
        }
        buffers_swapped_since_render = false;
    }

    // Additional safety: ensure all OpenGL commands have completed
    glFinish();

    // Handle screenshot based on format and headless mode
    if (is_png) {
        // PNG output
        if (headless && offscreenFramebufferID != 0) {
            // In headless mode, read pixels from the offscreen framebuffer
            std::vector<helios::RGBAcolor> pixels = readOffscreenPixelsRGBA(background_is_transparent);
            if (pixels.empty()) {
                helios_runtime_error("ERROR (Visualizer::printWindow): Failed to read pixels from offscreen framebuffer.");
            }

            int result = write_PNG_file(outfile_str.c_str(), Wframebuffer, Hframebuffer, pixels, message_flag);
            if (result == 0) {
                helios_runtime_error("ERROR (Visualizer::printWindow): Failed to save screenshot to " + outfile_str);
            }
        } else {
            // In windowed mode, use the traditional framebuffer reading
            int result = write_PNG_file(outfile_str.c_str(), Wframebuffer, Hframebuffer, buffers_swapped_since_render, background_is_transparent, message_flag);
            if (result == 0) {
                helios_runtime_error("ERROR (Visualizer::printWindow): Failed to save screenshot to " + outfile_str);
            }
        }
    } else {
        // JPEG output
        if (headless && offscreenFramebufferID != 0) {
            // In headless mode, read pixels from the offscreen framebuffer
            std::vector<helios::RGBcolor> pixels = readOffscreenPixels();
            if (pixels.empty()) {
                helios_runtime_error("ERROR (Visualizer::printWindow): Failed to read pixels from offscreen framebuffer.");
            }

            int result = write_JPEG_file(outfile_str.c_str(), Wframebuffer, Hframebuffer, pixels, message_flag);
            if (result == 0) {
                helios_runtime_error("ERROR (Visualizer::printWindow): Failed to save screenshot to " + outfile_str);
            }
        } else {
            // In windowed mode, use the traditional framebuffer reading
            int result = write_JPEG_file(outfile_str.c_str(), Wframebuffer, Hframebuffer, buffers_swapped_since_render, message_flag);
            if (result == 0) {
                helios_runtime_error("ERROR (Visualizer::printWindow): Failed to save screenshot to " + outfile_str);
            }
        }
    }

    // Restore the transparent background checkerboard rectangle if it was active
    if (had_transparent_background) {
        // Define rectangle vertices in normalized window coordinates
        std::vector<helios::vec3> vertices = {
                helios::make_vec3(0.f, 0.f, 0.99f), // Bottom-left
                helios::make_vec3(1.f, 0.f, 0.99f), // Bottom-right
                helios::make_vec3(1.f, 1.f, 0.99f), // Top-right
                helios::make_vec3(0.f, 1.f, 0.99f) // Top-left
        };

        // Calculate aspect ratio and adjust UV coordinates to maintain square checkerboard pattern
        float aspect_ratio = static_cast<float>(Wframebuffer) / static_cast<float>(Hframebuffer);
        std::vector<helios::vec2> uvs;
        if (aspect_ratio > 1.f) {
            // Window is wider than tall - stretch UV in x-direction
            uvs = {helios::make_vec2(0.f, 0.f), helios::make_vec2(aspect_ratio, 0.f), helios::make_vec2(aspect_ratio, 1.f), helios::make_vec2(0.f, 1.f)};
        } else {
            // Window is taller than wide - stretch UV in y-direction
            uvs = {helios::make_vec2(0.f, 0.f), helios::make_vec2(1.f, 0.f), helios::make_vec2(1.f, 1.f / aspect_ratio), helios::make_vec2(0.f, 1.f / aspect_ratio)};
        }

        background_rectangle_ID = addRectangleByVertices(vertices, "plugins/visualizer/textures/transparent.jpg", uvs, COORDINATES_WINDOW_NORMALIZED);
    }

    // Restore navigation gizmo state
    if (gizmo_was_visible) {
        showNavigationGizmo();
    }
}

void Visualizer::displayImage(const std::vector<unsigned char> &pixel_data, uint width_pixels, uint height_pixels) {
    if (pixel_data.empty()) {
        helios_runtime_error("ERROR (Visualizer::displayImage): Pixel data was empty.");
    }
    if (pixel_data.size() != 4 * width_pixels * height_pixels) {
        helios_runtime_error("ERROR (Visualizer::displayImage): Pixel data size does not match the given width and height. Argument 'pixel_data' must have length of 4*width_pixels*height_pixels.");
    }

    // Clear out any existing geometry
    geometry_handler.clearAllGeometry();

    // Register the data as a texture
    uint textureID = registerTextureImage(pixel_data, helios::make_uint2(width_pixels, height_pixels));

    // Figure out size to render
    vec2 image_size;
    float data_aspect = float(width_pixels) / float(height_pixels);
    float window_aspect = float(Wdisplay) / float(Hdisplay);
    if (data_aspect > window_aspect) {
        // fill width, shrink height
        image_size = make_vec2(1.0f, window_aspect / data_aspect);
    } else {
        // fill height, shrink width
        image_size = make_vec2(data_aspect / window_aspect, 1.0f);
    }

    constexpr vec3 center(0.5, 0.5, 0);
    const std::vector vertices{center + make_vec3(-0.5f * image_size.x, -0.5f * image_size.y, 0.f), center + make_vec3(+0.5f * image_size.x, -0.5f * image_size.y, 0.f), center + make_vec3(+0.5f * image_size.x, +0.5f * image_size.y, 0.f),
                               center + make_vec3(-0.5f * image_size.x, +0.5f * image_size.y, 0.f)};
    const std::vector<vec2> uvs{{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    size_t UUID = geometry_handler.sampleUUID();
    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, vertices, RGBA::black, uvs, textureID, false, false, COORDINATES_WINDOW_NORMALIZED, true, false);

    hideWatermark();

    // Save navigation gizmo state before hiding it (so we can restore it when building geometry)
    navigation_gizmo_was_enabled_before_image_display = navigation_gizmo_enabled;
    hideNavigationGizmo();

    plotInteractive();
}


void Visualizer::displayImage(const std::string &file_name) {
    if (!validateTextureFile(file_name)) {
        helios_runtime_error("ERROR (Visualizer::displayImage): File " + file_name + " does not exist or is not a valid image file.");
    }

    std::vector<unsigned char> image_data;
    uint image_width, image_height;

    if (file_name.substr(file_name.find_last_of('.') + 1) == "png") {
        read_png_file(file_name.c_str(), image_data, image_height, image_width);
    } else { // JPEG
        read_JPEG_file(file_name.c_str(), image_data, image_height, image_width);
    }

    displayImage(image_data, image_width, image_height);
}

void Visualizer::getWindowPixelsRGB(uint *buffer) const {
    // Validate framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        helios_runtime_error("ERROR (getWindowPixelsRGB): Framebuffer is not complete");
    }

    std::vector<GLubyte> buff;
    buff.resize(3 * Wframebuffer * Hframebuffer);

    // Set proper pixel alignment
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // Handle different pixel reading approaches based on headless mode
    if (headless && offscreenFramebufferID != 0) {
        // In headless mode with offscreen framebuffer, ensure we're reading from the correct framebuffer
        // Bind the offscreen framebuffer to ensure we read from the rendered content
        GLint current_framebuffer;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_framebuffer);

        glBindFramebuffer(GL_FRAMEBUFFER, offscreenFramebufferID);

        // Framebuffer objects don't use front/back buffer concepts, read directly
        glReadPixels(0, 0, GLsizei(Wframebuffer), GLsizei(Hframebuffer), GL_RGB, GL_UNSIGNED_BYTE, &buff[0]);
        GLenum error = glGetError();

        // Restore the previous framebuffer binding
        glBindFramebuffer(GL_FRAMEBUFFER, current_framebuffer);

        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (getWindowPixelsRGB): glReadPixels failed in headless mode (error: " + std::to_string(error) + ")");
        }
    } else {
        // In windowed mode, robustly determine which buffer contains rendered content
        // Sample multiple pixels to avoid false negatives from legitimately black pixels
        const int num_samples = 9;
        const uint sample_positions[][2] = {{Wframebuffer / 4, Hframebuffer / 4},     {Wframebuffer / 2, Hframebuffer / 4},     {3 * Wframebuffer / 4, Hframebuffer / 4},
                                            {Wframebuffer / 4, Hframebuffer / 2},     {Wframebuffer / 2, Hframebuffer / 2},     {3 * Wframebuffer / 4, Hframebuffer / 2},
                                            {Wframebuffer / 4, 3 * Hframebuffer / 4}, {Wframebuffer / 2, 3 * Hframebuffer / 4}, {3 * Wframebuffer / 4, 3 * Hframebuffer / 4}};
        GLubyte test_pixels[num_samples * 3];

        auto count_non_black_pixels = [](const GLubyte *pixels, int count) -> int {
            int non_black = 0;
            for (int i = 0; i < count * 3; i += 3) {
                if (pixels[i] > 5 || pixels[i + 1] > 5 || pixels[i + 2] > 5) { // Use small threshold to account for compression artifacts
                    non_black++;
                }
            }
            return non_black;
        };

        int back_buffer_content_score = 0;
        int front_buffer_content_score = 0;

        // Test back buffer with multiple samples
        glReadBuffer(GL_BACK);
        GLenum error = glGetError();
        if (error == GL_NO_ERROR) {
            glFinish();
            for (int i = 0; i < num_samples; i++) {
                glReadPixels(sample_positions[i][0], sample_positions[i][1], 1, 1, GL_RGB, GL_UNSIGNED_BYTE, &test_pixels[i * 3]);
            }
            if (glGetError() == GL_NO_ERROR) {
                back_buffer_content_score = count_non_black_pixels(test_pixels, num_samples);
            }
        }

        // Test front buffer with multiple samples
        glReadBuffer(GL_FRONT);
        error = glGetError();
        if (error == GL_NO_ERROR) {
            glFinish();
            for (int i = 0; i < num_samples; i++) {
                glReadPixels(sample_positions[i][0], sample_positions[i][1], 1, 1, GL_RGB, GL_UNSIGNED_BYTE, &test_pixels[i * 3]);
            }
            if (glGetError() == GL_NO_ERROR) {
                front_buffer_content_score = count_non_black_pixels(test_pixels, num_samples);
            }
        }

        // Choose the buffer with higher content score, prefer back buffer if scores are equal
        if (back_buffer_content_score >= front_buffer_content_score && back_buffer_content_score > 0) {
            glReadBuffer(GL_BACK);
        } else if (front_buffer_content_score > 0) {
            glReadBuffer(GL_FRONT);
        } else {
            // Neither buffer has detectable content, default to back buffer
            glReadBuffer(GL_BACK);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                glReadBuffer(GL_FRONT);
                error = glGetError();
                if (error != GL_NO_ERROR) {
                    helios_runtime_error("ERROR (getWindowPixelsRGB): Cannot set read buffer");
                }
            }
        }

        glReadPixels(0, 0, GLsizei(Wframebuffer), GLsizei(Hframebuffer), GL_RGB, GL_UNSIGNED_BYTE, &buff[0]);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            helios_runtime_error("ERROR (getWindowPixelsRGB): glReadPixels failed");
        }
    }

    glFinish();

    for (int i = 0; i < 3 * Wframebuffer * Hframebuffer; i++) {
        buffer[i] = (unsigned int) buff[i];
    }
}

void Visualizer::getDepthMap(float *buffer) {
    // if (depth_buffer_data.empty()) {
    //     helios_runtime_error("ERROR (Visualizer::getDepthMap): No depth map data available. You must run 'plotDepthMap' before depth map can be retrieved.");
    // }
    //
    // updatePerspectiveTransformation(camera_lookat_center, camera_eye_location, true);
    //
    // for (int i = 0; i < depth_buffer_data.size(); i++) {
    //     buffer[i] = -perspectiveTransformationMatrix[3].z / (depth_buffer_data.at(i) * -2.0f + 1.0f - perspectiveTransformationMatrix[2].z);
    // }

    std::vector<float> depth_pixels;
    uint width, height;
    getDepthMap(depth_pixels, width, height);
    for (size_t i = 0; i < depth_pixels.size(); ++i) {
        buffer[i] = depth_pixels[i];
    }
}

void Visualizer::getDepthMap(std::vector<float> &depth_pixels, uint &width_pixels, uint &height_pixels) {
    width_pixels = Wdisplay;
    height_pixels = Hdisplay;

    depth_pixels.resize(width_pixels * height_pixels);

    updateDepthBuffer();
    // updatePerspectiveTransformation( false );

    // un-project depth values to give physical depth

    // build a viewport vector for unProject
    // const glm::vec4 viewport(0, 0, width_pixels, height_pixels);
    //
    // for (size_t i = 0; i < width_pixels*height_pixels; ++i) {
    //     // compute pixel coords from linear index
    //     int x = int(i % width_pixels);
    //     int y = int(i / height_pixels);
    //
    //     // center of the pixel
    //     float winx = float(x) + 0.5f;
    //     float winy = float(y) + 0.5f;
    //     float depth = depth_buffer_data.at(i);  // in [0..1]
    //
    //     // build the window‐space coordinate
    //     glm::vec3 winCoord(winx, winy, depth);
    //
    //     // unProject to get world‐space position
    //     glm::vec3 worldPos = glm::unProject( winCoord, cameraViewMatrix, cameraProjectionMatrix, viewport );
    //
    //     // transform into camera‐space
    //     const glm::vec4 camPos = cameraViewMatrix * glm::vec4(worldPos, 1.0f);
    //
    //     // camPos.z is negative in front of the eye; flip sign for a positive distance
    //     // depth_pixels[i] = -camPos.z;
    //     depth_pixels[i] = depth*255.f;
    // }

    // normalize data and invert the color space so white=closest, black = furthest
    float depth_min = (std::numeric_limits<float>::max)();
    float depth_max = (std::numeric_limits<float>::min)();
    for (auto depth: depth_buffer_data) {
        if (depth < depth_min) {
            depth_min = depth;
        }
        if (depth > depth_max) {
            depth_max = depth;
        }
    }
    for (size_t i = 0; i < depth_pixels.size(); i++) {
        float value = std::round((depth_buffer_data.at(i) - depth_min) / (depth_max - depth_min) * 255);
        value = clamp(value, 0.f, 255.f);
        depth_pixels.at(i) = 255.f - value;
    }

    //\todo This is not working. Basically the same code works in the plotDepthMap() method, but for some reason doesn't seem to yield the correct float values.
}

void Visualizer::getWindowSize(uint &width, uint &height) const {
    width = Wdisplay;
    height = Hdisplay;
}

void Visualizer::getFramebufferSize(uint &width, uint &height) const {
    width = Wframebuffer;
    height = Hframebuffer;
}

void Visualizer::closeWindow() const {
    glfwHideWindow((GLFWwindow *) window);
    glfwPollEvents();
}

std::vector<helios::vec3> Visualizer::plotInteractive() {
    if (message_flag) {
        std::cout << "Generating interactive plot..." << std::flush;
    }


    // Update the Context geometry
    buildContextGeometry_private();

    // Set the view to fit window
    if (camera_lookat_center.x == 0 && camera_lookat_center.y == 0 && camera_lookat_center.z == 0) { // default center
        if (camera_eye_location.x < 1e-4 && camera_eye_location.y < 1e-4 && camera_eye_location.z == 2.f) { // default eye position

            vec3 center_sph;
            vec3 radius;
            geometry_handler.getDomainBoundingSphere(center_sph, radius);
            float domain_bounding_radius = radius.magnitude();

            vec2 xbounds, ybounds, zbounds;
            geometry_handler.getDomainBoundingBox(xbounds, ybounds, zbounds);
            camera_lookat_center = make_vec3(0.5f * (xbounds.x + xbounds.y), 0.5f * (ybounds.x + ybounds.y), zbounds.x);
            camera_eye_location = camera_lookat_center + sphere2cart(make_SphericalCoord(2.f * domain_bounding_radius, 20.f * PI_F / 180.f, 0));
        }
    }

    // Colorbar - update with aspect ratio correction
    updateColorbar();

    // Watermark
    updateWatermark();

    // Navigation gizmo - initialize before entering render loop
    if (navigation_gizmo_enabled) {
        updateNavigationGizmo();
        previous_camera_eye_location = camera_eye_location;
        previous_camera_lookat_center = camera_lookat_center;
    }

    transferBufferData();

    assert(checkerrors());

    bool shadow_flag = (primaryLightingModel == Visualizer::LIGHTING_PHONG_SHADOWED);

    glm::mat4 depthMVP;

    assert(checkerrors());

    std::vector<vec3> camera_output;

    glfwShowWindow((GLFWwindow *) window);

    do {
        // Update navigation gizmo if camera has changed
        if (navigation_gizmo_enabled && cameraHasChanged()) {
            updateNavigationGizmo();
            previous_camera_eye_location = camera_eye_location;
            previous_camera_lookat_center = camera_lookat_center;
            // Transfer updated geometry to GPU immediately after updating gizmo
            transferBufferData();
        }

        if (shadow_flag) {
            assert(checkerrors());
            // Depth buffer for shadows
            glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);
            glViewport(0, 0, shadow_buffer_size.x, shadow_buffer_size.y); // Render on the whole framebuffer, complete from the lower left corner to the upper right

            // Clear the screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            depthShader.useShader();

            updatePerspectiveTransformation(true);

            // Compute the MVP matrix from the light's point of view
            depthMVP = computeShadowDepthMVP();
            depthShader.setTransformationMatrix(depthMVP);

            // bind depth texture
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, depthTexture);
            glActiveTexture(GL_TEXTURE0);

            depthShader.enableTextureMaps();
            depthShader.enableTextureMasks();

            assert(checkerrors());

            render(true);
        } else {
            depthMVP = glm::mat4(1.0);
        }

        // Render to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, Wframebuffer, Hframebuffer);

        if (background_is_transparent) {
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 1.0f);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        primaryShader.useShader();

        glm::mat4 DepthBiasMVP = biasMatrix * depthMVP;

        primaryShader.setDepthBiasMatrix(DepthBiasMVP);

        updatePerspectiveTransformation(false);

        primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);
        primaryShader.setViewMatrix(cameraViewMatrix);
        primaryShader.setProjectionMatrix(cameraProjectionMatrix);

        primaryShader.enableTextureMaps();
        primaryShader.enableTextureMasks();

        primaryShader.setLightingModel(primaryLightingModel);
        primaryShader.setLightIntensity(lightintensity);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glUniform1i(primaryShader.shadowmapUniform, 1);
        glActiveTexture(GL_TEXTURE0);

        buffers_swapped_since_render = false;
        render(false);

        assert(checkerrors());

        glfwPollEvents();
        getViewKeystrokes(camera_eye_location, camera_lookat_center);

        glfwSwapBuffers((GLFWwindow *) window);
        buffers_swapped_since_render = true;

        glfwWaitEventsTimeout(1.0 / 30.0);

        int width, height;
        glfwGetFramebufferSize((GLFWwindow *) window, &width, &height);
        Wframebuffer = width;
        Hframebuffer = height;
    } while (glfwGetKey((GLFWwindow *) window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose((GLFWwindow *) window) == 0);

    glfwPollEvents();

    assert(checkerrors());

    camera_output.push_back(camera_eye_location);
    camera_output.push_back(camera_lookat_center);

    if (message_flag) {
        std::cout << "done." << std::endl;
    }

    return camera_output;
}

void Visualizer::plotOnce(bool getKeystrokes) {

    bool shadow_flag = (primaryLightingModel == Visualizer::LIGHTING_PHONG_SHADOWED);

    glm::mat4 depthMVP;

    if (shadow_flag) {
        // Depth buffer for shadows
        glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);
        glViewport(0, 0, shadow_buffer_size.x, shadow_buffer_size.y); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        depthShader.useShader();

        updatePerspectiveTransformation(true);

        // Compute the MVP matrix from the light's point of view
        depthMVP = computeShadowDepthMVP();
        depthShader.setTransformationMatrix(depthMVP);

        // bind depth texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glActiveTexture(GL_TEXTURE0);

        depthShader.enableTextureMaps();
        depthShader.enableTextureMasks();

        render(true);
    } else {
        depthMVP = glm::mat4(1.0);
    }

    assert(checkerrors());

    // Render to the screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, Wframebuffer, Hframebuffer);

    if (background_is_transparent) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 1.0f);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    primaryShader.useShader();

    glm::mat4 DepthBiasMVP = biasMatrix * depthMVP;

    primaryShader.setDepthBiasMatrix(DepthBiasMVP);

    updatePerspectiveTransformation(false);

    primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);
    primaryShader.setViewMatrix(cameraViewMatrix);
    primaryShader.setProjectionMatrix(cameraProjectionMatrix);

    primaryShader.enableTextureMaps();
    primaryShader.enableTextureMasks();

    primaryShader.setLightingModel(primaryLightingModel);
    primaryShader.setLightIntensity(lightintensity);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glUniform1i(primaryShader.shadowmapUniform, 1);
    glActiveTexture(GL_TEXTURE0);

    // Set buffer state before rendering (plotOnce doesn't call glfwSwapBuffers)
    buffers_swapped_since_render = false;
    render(false);

    // glfwPollEvents();
    if (getKeystrokes) {
        getViewKeystrokes(camera_eye_location, camera_lookat_center);

        // Update navigation gizmo if camera has changed
        if (navigation_gizmo_enabled && cameraHasChanged()) {
            updateNavigationGizmo();
            previous_camera_eye_location = camera_eye_location;
            previous_camera_lookat_center = camera_lookat_center;
            // Transfer updated geometry to GPU
            transferBufferData();
        }
    }

    int width, height;
    glfwGetFramebufferSize((GLFWwindow *) window, &width, &height);
    Wframebuffer = width;
    Hframebuffer = height;
}

void Visualizer::transferBufferData() {
    assert(checkerrors());

    const auto &dirty = geometry_handler.getDirtyUUIDs();
    if (dirty.empty()) {
        return;
    }

    auto ensureArrayBuffer = [](GLuint buf, GLenum target, GLsizeiptr size, const void *data) {
        glBindBuffer(target, buf);
        GLint current_size = 0;
        glGetBufferParameteriv(target, GL_BUFFER_SIZE, &current_size);
        if (current_size != size) {
            glBufferData(target, size, data, GL_STATIC_DRAW);
        } else {
            // Buffer size matches, but data may have changed
            // Update the buffer contents
            glBufferSubData(target, 0, size, data);
        }
    };

    auto ensureTextureBuffer = [](GLuint buf, GLuint tex, GLenum format, GLsizeiptr size, const void *data) {
        glBindBuffer(GL_TEXTURE_BUFFER, buf);
        GLint current_size = 0;
        glGetBufferParameteriv(GL_TEXTURE_BUFFER, GL_BUFFER_SIZE, &current_size);
        if (current_size != size) {
            glBufferData(GL_TEXTURE_BUFFER, size, data, GL_STATIC_DRAW);
        } else {
            // Buffer size matches, but data may have changed (e.g., visibility flags)
            // Update the buffer contents
            glBufferSubData(GL_TEXTURE_BUFFER, 0, size, data);
        }
        glBindTexture(GL_TEXTURE_BUFFER, tex);
        glTexBuffer(GL_TEXTURE_BUFFER, format, buf);
    };

    // Ensure buffers are allocated to the correct size
    for (size_t gi = 0; gi < GeometryHandler::all_geometry_types.size(); ++gi) {
        const auto geometry_type = GeometryHandler::all_geometry_types[gi];
        const auto *vertex_data = geometry_handler.getVertexData_ptr(geometry_type);
        const auto *uv_data = geometry_handler.getUVData_ptr(geometry_type);
        const auto *face_index_data = geometry_handler.getFaceIndexData_ptr(geometry_type);
        const auto *color_data = geometry_handler.getColorData_ptr(geometry_type);
        const auto *normal_data = geometry_handler.getNormalData_ptr(geometry_type);
        const auto *texture_flag_data = geometry_handler.getTextureFlagData_ptr(geometry_type);
        const auto *texture_ID_data = geometry_handler.getTextureIDData_ptr(geometry_type);
        const auto *coordinate_flag_data = geometry_handler.getCoordinateFlagData_ptr(geometry_type);
        const auto *sky_geometry_flag_data = geometry_handler.getSkyGeometryFlagData_ptr(geometry_type);
        const auto *visible_flag_data = geometry_handler.getVisibilityFlagData_ptr(geometry_type);

        ensureArrayBuffer(vertex_buffer.at(gi), GL_ARRAY_BUFFER, vertex_data->size() * sizeof(GLfloat), vertex_data->data());
        ensureArrayBuffer(uv_buffer.at(gi), GL_ARRAY_BUFFER, uv_data->size() * sizeof(GLfloat), uv_data->data());
        ensureArrayBuffer(face_index_buffer.at(gi), GL_ARRAY_BUFFER, face_index_data->size() * sizeof(GLint), face_index_data->data());
        ensureTextureBuffer(color_buffer.at(gi), color_texture_object.at(gi), GL_RGBA32F, color_data->size() * sizeof(GLfloat), color_data->data());
        ensureTextureBuffer(normal_buffer.at(gi), normal_texture_object.at(gi), GL_RGB32F, normal_data->size() * sizeof(GLfloat), normal_data->data());
        ensureTextureBuffer(texture_flag_buffer.at(gi), texture_flag_texture_object.at(gi), GL_R32I, texture_flag_data->size() * sizeof(GLint), texture_flag_data->data());
        ensureTextureBuffer(texture_ID_buffer.at(gi), texture_ID_texture_object.at(gi), GL_R32I, texture_ID_data->size() * sizeof(GLint), texture_ID_data->data());
        ensureTextureBuffer(coordinate_flag_buffer.at(gi), coordinate_flag_texture_object.at(gi), GL_R32I, coordinate_flag_data->size() * sizeof(GLint), coordinate_flag_data->data());
        ensureTextureBuffer(sky_geometry_flag_buffer.at(gi), sky_geometry_flag_texture_object.at(gi), GL_R8I, sky_geometry_flag_data->size() * sizeof(GLbyte), sky_geometry_flag_data->data());
        ensureTextureBuffer(hidden_flag_buffer.at(gi), hidden_flag_texture_object.at(gi), GL_R8I, visible_flag_data->size() * sizeof(GLbyte), visible_flag_data->data());

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }

    bool rect_dirty = false;
    for (size_t UUID: dirty) {
        if (!geometry_handler.doesGeometryExist(UUID)) {
            continue;
        }

        const auto &index_map = geometry_handler.getIndexMap(UUID);
        auto geometry_type = index_map.geometry_type;
        size_t i = std::find(GeometryHandler::all_geometry_types.begin(), GeometryHandler::all_geometry_types.end(), geometry_type) - GeometryHandler::all_geometry_types.begin();

        const char vcount = GeometryHandler::getVertexCount(geometry_type);

        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.vertex_index * sizeof(GLfloat), vcount * 3 * sizeof(GLfloat), geometry_handler.getVertexData_ptr(geometry_type)->data() + index_map.vertex_index);

        glBindBuffer(GL_ARRAY_BUFFER, uv_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.uv_index * sizeof(GLfloat), vcount * 2 * sizeof(GLfloat), geometry_handler.getUVData_ptr(geometry_type)->data() + index_map.uv_index);

        glBindBuffer(GL_ARRAY_BUFFER, face_index_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.face_index_index * sizeof(GLint), vcount * sizeof(GLint), geometry_handler.getFaceIndexData_ptr(geometry_type)->data() + index_map.face_index_index);

        glBindBuffer(GL_TEXTURE_BUFFER, color_buffer.at(i));
        glBufferSubData(GL_TEXTURE_BUFFER, index_map.color_index * sizeof(GLfloat), 4 * sizeof(GLfloat), geometry_handler.getColorData_ptr(geometry_type)->data() + index_map.color_index);
        glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, color_buffer.at(i));

        glBindBuffer(GL_ARRAY_BUFFER, normal_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.normal_index * sizeof(GLfloat), 3 * sizeof(GLfloat), geometry_handler.getNormalData_ptr(geometry_type)->data() + index_map.normal_index);
        glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, normal_buffer.at(i));

        glBindBuffer(GL_ARRAY_BUFFER, texture_flag_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.texture_flag_index * sizeof(GLint), sizeof(GLint), geometry_handler.getTextureFlagData_ptr(geometry_type)->data() + index_map.texture_flag_index);
        glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, texture_flag_buffer.at(i));

        glBindBuffer(GL_ARRAY_BUFFER, texture_ID_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.texture_ID_index * sizeof(GLint), sizeof(GLint), geometry_handler.getTextureIDData_ptr(geometry_type)->data() + index_map.texture_ID_index);
        glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, texture_ID_buffer.at(i));

        glBindBuffer(GL_ARRAY_BUFFER, coordinate_flag_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.coordinate_flag_index * sizeof(GLint), sizeof(GLint), geometry_handler.getCoordinateFlagData_ptr(geometry_type)->data() + index_map.coordinate_flag_index);
        glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, coordinate_flag_buffer.at(i));

        glBindBuffer(GL_ARRAY_BUFFER, sky_geometry_flag_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.sky_geometry_flag_index * sizeof(GLbyte), sizeof(GLbyte), geometry_handler.getSkyGeometryFlagData_ptr(geometry_type)->data() + index_map.sky_geometry_flag_index);
        glBindTexture(GL_TEXTURE_BUFFER, sky_geometry_flag_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R8I, sky_geometry_flag_buffer.at(i));

        glBindBuffer(GL_ARRAY_BUFFER, hidden_flag_buffer.at(i));
        glBufferSubData(GL_ARRAY_BUFFER, index_map.visible_index * sizeof(GLbyte), sizeof(GLbyte), geometry_handler.getVisibilityFlagData_ptr(geometry_type)->data() + index_map.visible_index);
        glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R8I, hidden_flag_buffer.at(i));

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindTexture(GL_TEXTURE_BUFFER, 0);

        if (geometry_type == GeometryHandler::GEOMETRY_TYPE_RECTANGLE) {
            rect_dirty = true;
        }
    }

    if (rect_dirty) {
        size_t rectangle_count = geometry_handler.getRectangleCount();

        rectangle_vertex_group_firsts.resize(rectangle_count);
        rectangle_vertex_group_counts.resize(rectangle_count, 4);
        for (int j = 0; j < rectangle_count; ++j) {
            rectangle_vertex_group_firsts[j] = j * 4;
        }
    }

    if (textures_dirty || texArray == 0) {
        transferTextureData();
        textures_dirty = false;
    }

    geometry_handler.clearDirtyUUIDs();

    assert(checkerrors());
}

void Visualizer::transferTextureData() {

    const size_t layers = std::max<size_t>(1, texture_manager.size());

    // Check if texture needs recreation (Windows requires deleting and recreating texture
    // when layer count changes due to glTexStorage3D immutable format restrictions)
    if (texArray == 0 || layers != texture_array_layers) {
        // Delete existing texture if it exists
        if (texArray != 0) {
            glDeleteTextures(1, &texArray);
        }

        // Create new texture
        glGenTextures(1, &texArray);
        glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, maximum_texture_size.x, maximum_texture_size.y, layers);
        texture_array_layers = layers;
    } else {
        glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);
    }

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    std::vector<GLfloat> uv_rescale;
    uv_rescale.resize(texture_manager.size() * 2);

    for (const auto &[textureID, texture]: texture_manager) {
        GLenum externalFormat = 0;
        switch (texture.num_channels) {
            case 1:
                externalFormat = GL_RED;
                break;
            case 3:
                externalFormat = GL_RGB;
                break;
            case 4:
                externalFormat = GL_RGBA;
                break;
            default:
                throw std::runtime_error("unsupported channel count");
        }

        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, textureID, texture.texture_resolution.x, texture.texture_resolution.y, 1, externalFormat, GL_UNSIGNED_BYTE, texture.texture_data.data());

        uv_rescale.at(textureID * 2 + 0) = float(texture.texture_resolution.x) / float(maximum_texture_size.x);
        uv_rescale.at(textureID * 2 + 1) = float(texture.texture_resolution.y) / float(maximum_texture_size.y);
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    glUniform1i(glGetUniformLocation(primaryShader.shaderID, "textureSampler"), 0);

    glBindBuffer(GL_TEXTURE_BUFFER, uv_rescale_buffer);
    glBufferData(GL_TEXTURE_BUFFER, uv_rescale.size() * sizeof(GLfloat), uv_rescale.data(), GL_STATIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, uv_rescale_texture_object);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, uv_rescale_buffer);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}


void Visualizer::render(bool shadow) const {
    size_t rectangle_ind = std::find(GeometryHandler::all_geometry_types.begin(), GeometryHandler::all_geometry_types.end(), GeometryHandler::GEOMETRY_TYPE_RECTANGLE) - GeometryHandler::all_geometry_types.begin();
    size_t triangle_ind = std::find(GeometryHandler::all_geometry_types.begin(), GeometryHandler::all_geometry_types.end(), GeometryHandler::GEOMETRY_TYPE_TRIANGLE) - GeometryHandler::all_geometry_types.begin();
    size_t point_ind = std::find(GeometryHandler::all_geometry_types.begin(), GeometryHandler::all_geometry_types.end(), GeometryHandler::GEOMETRY_TYPE_POINT) - GeometryHandler::all_geometry_types.begin();
    size_t line_ind = std::find(GeometryHandler::all_geometry_types.begin(), GeometryHandler::all_geometry_types.end(), GeometryHandler::GEOMETRY_TYPE_LINE) - GeometryHandler::all_geometry_types.begin();

    size_t triangle_count = geometry_handler.getTriangleCount();
    size_t rectangle_count = geometry_handler.getRectangleCount();
    size_t line_count = geometry_handler.getLineCount();
    size_t point_count = geometry_handler.getPointCount();

    // Look up the currently loaded shader
    GLint current_shader_program = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &current_shader_program);

    // Bind our texture array
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);

    assert(checkerrors());

    glActiveTexture(GL_TEXTURE9);
    assert(checkerrors());
    glBindTexture(GL_TEXTURE_BUFFER, uv_rescale_texture_object);
    assert(checkerrors());
    glUniform1i(glGetUniformLocation(current_shader_program, "uv_rescale"), 9);

    //--- Triangles---//

    assert(checkerrors());

    if (triangle_count > 0) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(triangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(triangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(triangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(triangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(triangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER, sky_geometry_flag_texture_object.at(triangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(triangle_ind));
        // Note: Uniform location already set during shader initialization

        glBindVertexArray(primaryShader.vertex_array_IDs.at(triangle_ind));
        assert(checkerrors());
        glDrawArrays(GL_TRIANGLES, 0, triangle_count * 3);
    }

    assert(checkerrors());

    //--- Rectangles---//

    if (rectangle_count > 0) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(rectangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(rectangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(rectangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(rectangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(rectangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER, sky_geometry_flag_texture_object.at(rectangle_ind));
        // Note: Uniform location already set during shader initialization

        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(rectangle_ind));
        // Note: Uniform location already set during shader initialization

        glBindVertexArray(primaryShader.vertex_array_IDs.at(rectangle_ind));

        std::vector<GLint> opaque_firsts;
        std::vector<GLint> opaque_counts;
        struct TransparentRect {
            size_t index;
            float depth;
        };
        std::vector<TransparentRect> transparent_rects;

        const auto &texFlags = *geometry_handler.getTextureFlagData_ptr(GeometryHandler::GEOMETRY_TYPE_RECTANGLE);
        const auto &colors = *geometry_handler.getColorData_ptr(GeometryHandler::GEOMETRY_TYPE_RECTANGLE);
        const auto &verts = *geometry_handler.getVertexData_ptr(GeometryHandler::GEOMETRY_TYPE_RECTANGLE);
        const auto &visibilityFlags = *geometry_handler.getVisibilityFlagData_ptr(GeometryHandler::GEOMETRY_TYPE_RECTANGLE);

        opaque_firsts.reserve(rectangle_count);
        opaque_counts.reserve(rectangle_count);
        transparent_rects.reserve(rectangle_count);

        for (size_t i = 0; i < rectangle_count; ++i) {
            // Skip invisible rectangles
            if (!visibilityFlags.at(i)) {
                continue;
            }

            bool isGlyph = texFlags.at(i) == 3;
            float alpha = colors.at(i * 4 + 3);
            if (!isGlyph && alpha >= 1.f) {
                opaque_firsts.push_back(static_cast<GLint>(i * 4));
                opaque_counts.push_back(4);
            } else {
                glm::vec3 center(0.f);
                for (int j = 0; j < 4; ++j) {
                    center.x += verts.at(i * 12 + j * 3 + 0);
                    center.y += verts.at(i * 12 + j * 3 + 1);
                    center.z += verts.at(i * 12 + j * 3 + 2);
                }
                center /= 4.f;
                glm::vec4 viewPos = cameraViewMatrix * glm::vec4(center, 1.f);
                transparent_rects.push_back({i, viewPos.z});
            }
        }

        if (!opaque_firsts.empty()) {
            glMultiDrawArrays(GL_TRIANGLE_FAN, opaque_firsts.data(), opaque_counts.data(), static_cast<GLsizei>(opaque_firsts.size()));
        }

        if (!transparent_rects.empty()) {
            std::sort(transparent_rects.begin(), transparent_rects.end(), [](const TransparentRect &a, const TransparentRect &b) {
                return a.depth > b.depth; // farthest first
            });

            glDepthMask(GL_FALSE);
            for (const auto &tr: transparent_rects) {
                glDrawArrays(GL_TRIANGLE_FAN, static_cast<GLint>(tr.index * 4), 4);
            }
            glDepthMask(GL_TRUE);
        }
    }

    assert(checkerrors());

    if (!shadow) {
        //--- Lines ---//

        if (line_count > 0) {
            // Switch to line shader which uses geometry shader for wide lines
            lineShader.useShader();
            lineShader.setTransformationMatrix(perspectiveTransformationMatrix);
            lineShader.setDepthBiasMatrix(computeShadowDepthMVP());
            lineShader.setLightDirection(light_direction);
            lineShader.setLightingModel(primaryLightingModel);
            lineShader.setLightIntensity(lightintensity);

            // Set viewport size for geometry shader
            GLint viewportSizeLoc = glGetUniformLocation(lineShader.shaderID, "viewportSize");
            glUniform2f(viewportSizeLoc, static_cast<float>(Wframebuffer), static_cast<float>(Hframebuffer));

            // Rebind texture array and uv_rescale for line shader
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);
            glActiveTexture(GL_TEXTURE9);
            glBindTexture(GL_TEXTURE_BUFFER, uv_rescale_texture_object);
            glUniform1i(glGetUniformLocation(lineShader.shaderID, "uv_rescale"), 9);

            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(line_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(line_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(line_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(line_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE7);
            glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(line_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_BUFFER, sky_geometry_flag_texture_object.at(line_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE8);
            glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(line_ind));
            // Note: Uniform location already set during shader initialization

            glBindVertexArray(lineShader.vertex_array_IDs.at(line_ind));

            // Group lines by width and render each group separately
            const std::vector<float> *size_data = geometry_handler.getSizeData_ptr(GeometryHandler::GEOMETRY_TYPE_LINE);
            if (size_data && !size_data->empty()) {
                // Create map of line width -> line indices for grouped rendering
                std::map<float, std::vector<size_t>> width_groups;
                for (size_t i = 0; i < size_data->size(); ++i) {
                    float width = size_data->at(i);
                    if (width <= 0)
                        width = 1.0f; // Default width for invalid values
                    width_groups[width].push_back(i);
                }

                // Get the uniform location for line width
                GLint lineWidthLoc = glGetUniformLocation(lineShader.shaderID, "lineWidth");

                // Render each width group separately
                for (const auto &group: width_groups) {
                    float width = group.first;
                    const std::vector<size_t> &line_indices = group.second;

                    // Set line width uniform for geometry shader
                    glUniform1f(lineWidthLoc, width);

                    // Render each line individually
                    // The geometry shader will expand each line into a quad
                    for (size_t line_idx: line_indices) {
                        glDrawArrays(GL_LINES, static_cast<GLint>(line_idx * 2), 2);
                    }
                }
            } else {
                // Fallback to default width of 1.0 if no size data available
                GLint lineWidthLoc = glGetUniformLocation(lineShader.shaderID, "lineWidth");
                glUniform1f(lineWidthLoc, 1.0f);
                glDrawArrays(GL_LINES, 0, line_count * 2);
            }

            // Switch back to primary shader for subsequent rendering
            primaryShader.useShader();
            primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);
            primaryShader.setDepthBiasMatrix(computeShadowDepthMVP());
            primaryShader.setLightDirection(light_direction);
            primaryShader.setLightIntensity(lightintensity);

            // Rebind texture array and uv_rescale for primary shader
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);
            glActiveTexture(GL_TEXTURE9);
            glBindTexture(GL_TEXTURE_BUFFER, uv_rescale_texture_object);
            glUniform1i(glGetUniformLocation(primaryShader.shaderID, "uv_rescale"), 9);
        }

        assert(checkerrors());

        //--- Points ---//

        if (point_count > 0) {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(point_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(point_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(point_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(point_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE7);
            glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(point_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_BUFFER, sky_geometry_flag_texture_object.at(point_ind));
            // Note: Uniform location already set during shader initialization

            glActiveTexture(GL_TEXTURE8);
            glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(point_ind));
            // Note: Uniform location already set during shader initialization

            glBindVertexArray(primaryShader.vertex_array_IDs.at(point_ind));

            // Group points by size and render each group separately
            const std::vector<float> *size_data = geometry_handler.getSizeData_ptr(GeometryHandler::GEOMETRY_TYPE_POINT);
            if (size_data && !size_data->empty()) {
                // Create map of point size -> point indices for grouped rendering
                std::map<float, std::vector<size_t>> size_groups;
                for (size_t i = 0; i < size_data->size(); ++i) {
                    float size = size_data->at(i);
                    if (size <= 0)
                        size = 1.0f; // Default size for invalid values
                    size_groups[size].push_back(i);
                }

                // Render each size group separately
                for (const auto &group: size_groups) {
                    float size = group.first;
                    const std::vector<size_t> &point_indices = group.second;

                    glPointSize(size);

                    // For simplicity, render each point individually
                    // In a more optimized implementation, we would batch points with same size
                    for (size_t point_idx: point_indices) {
                        glDrawArrays(GL_POINTS, point_idx, 1);
                    }
                }
            } else {
                // Fallback to default behavior if no size data available
                glPointSize(point_width);
                glDrawArrays(GL_POINTS, 0, point_count);
            }
        }

        // if( !positionData["sky"].empty() ){
        //     primaryShader.setLightingModel( LIGHTING_NONE );
        //     glBindTexture(GL_TEXTURE_RECTANGLE,textureIDData["sky"].at(0));
        //     glDrawArrays(GL_TRIANGLES, triangle_size+line_size+point_size, positionData["sky"].size()/3 );
        // }
    }

    // Clean up texture units to prevent macOS OpenGL warnings
    // This helps prevent texture unit binding issues in headless mode
    glActiveTexture(GL_TEXTURE0);

    assert(checkerrors());
}

void Visualizer::plotUpdate() {
    plotUpdate(false);
}

void Visualizer::plotUpdate(bool hide_window) {
    // Check if window is marked for closure to prevent hanging on glfwSwapBuffers()
    if (!headless && window != nullptr && glfwWindowShouldClose(scast<GLFWwindow *>(window))) {
        return; // Don't render to a window that should be closed
    }

    if (message_flag) {
        std::cout << "Updating the plot..." << std::flush;
    }

    // Warn user if they request visible window in headless mode
    if (!hide_window && headless) {
        if (message_flag) {
            std::cout << "\nWARNING: plotUpdate(false) called in headless mode - window cannot be displayed. Use plotUpdate(true) for headless rendering." << std::endl;
        }
    }

    if (!hide_window && !headless && window != nullptr) {
        glfwShowWindow(scast<GLFWwindow *>(window));
    }

    // Update the Context geometry
    buildContextGeometry_private();

    // Apply point cloud culling for performance optimization
    updatePointCulling();

    // Set the view to fit window
    if (camera_lookat_center.x == 0 && camera_lookat_center.y == 0 && camera_lookat_center.z == 0) { // default center
        if (camera_eye_location.x < 1e-4 && camera_eye_location.y < 1e-4 && camera_eye_location.z == 2.f) { // default eye position

            vec3 center_sph;
            vec3 radius;
            geometry_handler.getDomainBoundingSphere(center_sph, radius);
            float domain_bounding_radius = radius.magnitude();

            vec2 xbounds, ybounds, zbounds;
            geometry_handler.getDomainBoundingBox(xbounds, ybounds, zbounds);
            camera_lookat_center = make_vec3(0.5f * (xbounds.x + xbounds.y), 0.5f * (ybounds.x + ybounds.y), 0.5f * (zbounds.x + zbounds.y));
            camera_eye_location = camera_lookat_center + sphere2cart(make_SphericalCoord(2.f * domain_bounding_radius, 20.f * PI_F / 180.f, 0));
        }
    }

    // Colorbar - update with aspect ratio correction
    updateColorbar();

    // Watermark
    updateWatermark();

    // Navigation gizmo - update if camera has changed or create initial geometry if needed
    if (navigation_gizmo_enabled) {
        if (navigation_gizmo_IDs.empty() || cameraHasChanged()) {
            updateNavigationGizmo();
            previous_camera_eye_location = camera_eye_location;
            previous_camera_lookat_center = camera_lookat_center;
        }
    }

    transferBufferData();

    bool shadow_flag = (primaryLightingModel == Visualizer::LIGHTING_PHONG_SHADOWED);

    glm::mat4 depthMVP;

    if (shadow_flag) {
        // Depth buffer for shadows
        glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);
        glViewport(0, 0, shadow_buffer_size.x, shadow_buffer_size.y); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        depthShader.useShader();

        updatePerspectiveTransformation(true);

        // Compute the MVP matrix from the light's point of view
        depthMVP = computeShadowDepthMVP();
        depthShader.setTransformationMatrix(depthMVP);

        // bind depth texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glActiveTexture(GL_TEXTURE0);

        depthShader.enableTextureMaps();
        depthShader.enableTextureMasks();

        render(true);
    } else {
        depthMVP = glm::mat4(1.0);
    }

    assert(checkerrors());

    // Render to the screen or offscreen framebuffer depending on headless mode
    if (headless && offscreenFramebufferID != 0) {
        // Render to offscreen framebuffer for headless mode
        glBindFramebuffer(GL_FRAMEBUFFER, offscreenFramebufferID);
        glViewport(0, 0, Wframebuffer, Hframebuffer);
    } else {
        // Render to the default framebuffer for windowed mode
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, Wframebuffer, Hframebuffer);
    }

    if (background_is_transparent) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 1.0f);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    primaryShader.useShader();

    glm::mat4 DepthBiasMVP = biasMatrix * depthMVP;

    primaryShader.setDepthBiasMatrix(DepthBiasMVP);

    updatePerspectiveTransformation(false);

    primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);
    primaryShader.setViewMatrix(cameraViewMatrix);
    primaryShader.setProjectionMatrix(cameraProjectionMatrix);

    primaryShader.enableTextureMaps();
    primaryShader.enableTextureMasks();

    primaryShader.setLightingModel(primaryLightingModel);
    primaryShader.setLightIntensity(lightintensity);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glUniform1i(primaryShader.shadowmapUniform, 1);
    glActiveTexture(GL_TEXTURE0);

    buffers_swapped_since_render = false;
    render(false);

    // Skip window-specific operations in headless mode
    if (!headless && window != nullptr) {
        glfwPollEvents();
        getViewKeystrokes(camera_eye_location, camera_lookat_center);

        // Update navigation gizmo if camera has changed
        if (navigation_gizmo_enabled && cameraHasChanged()) {
            updateNavigationGizmo();
            previous_camera_eye_location = camera_eye_location;
            previous_camera_lookat_center = camera_lookat_center;
            // Transfer updated geometry to GPU
            transferBufferData();
        }

        int width, height;
        glfwGetFramebufferSize((GLFWwindow *) window, &width, &height);
        Wframebuffer = width;
        Hframebuffer = height;

        glfwSwapBuffers((GLFWwindow *) window);
        buffers_swapped_since_render = true;
    }

    if (message_flag) {
        std::cout << "done." << std::endl;
    }
}

void Visualizer::updateDepthBuffer() {
    // Update the Context geometry (if needed)
    if (true) {
        buildContextGeometry_private();
    }

    transferBufferData();

    // Depth buffer for shadows
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);
    glViewport(0, 0, Wframebuffer, Hframebuffer);

    // bind depth texture
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, Wframebuffer, Hframebuffer, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glActiveTexture(GL_TEXTURE0);

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    depthShader.useShader();

    updatePerspectiveTransformation(false);
    depthShader.setTransformationMatrix(perspectiveTransformationMatrix);

    depthShader.enableTextureMaps();
    depthShader.enableTextureMasks();

    render(true);

    assert(checkerrors());

    depth_buffer_data.resize(Wframebuffer * Hframebuffer);

#if defined(__APPLE__)
    constexpr GLenum read_buf = GL_FRONT;
#else
    constexpr GLenum read_buf = GL_BACK;
#endif
    glReadBuffer(read_buf);
    glReadPixels(0, 0, Wframebuffer, Hframebuffer, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buffer_data.data());
    glFinish();

    assert(checkerrors());

    // Updates this->depth_buffer_data()
}

void Visualizer::plotDepthMap() {
    if (message_flag) {
        std::cout << "Rendering depth map..." << std::flush;
    }

    updateDepthBuffer();

    // normalize data, flip in y direction, and invert the color space so white=closest, black = furthest
    float depth_min = (std::numeric_limits<float>::max)();
    float depth_max = (std::numeric_limits<float>::min)();
    for (auto depth: depth_buffer_data) {
        if (depth < depth_min) {
            depth_min = depth;
        }
        if (depth > depth_max) {
            depth_max = depth;
        }
    }
    std::vector<unsigned char> depth_uchar(depth_buffer_data.size() * 4);
    for (size_t i = 0; i < depth_buffer_data.size(); i++) {
        auto value = scast<unsigned char>(std::round((depth_buffer_data.at(i) - depth_min) / (depth_max - depth_min) * 255));
        value = clamp(value, scast<unsigned char>(0), scast<unsigned char>(255));
        size_t row = i / Wframebuffer;
        size_t col = i % Wframebuffer;
        size_t flipped_i = (Hframebuffer - 1 - row) * Wframebuffer + col; // flipping
        depth_uchar.at(flipped_i * 4) = 255 - value; // R
        depth_uchar.at(flipped_i * 4 + 1) = 255 - value; // G
        depth_uchar.at(flipped_i * 4 + 2) = 255 - value; // B
        depth_uchar.at(flipped_i * 4 + 3) = 255; // A
    }

    displayImage(depth_uchar, Wframebuffer, Hframebuffer);

    if (message_flag) {
        std::cout << "done." << std::endl;
    }
}

void Shader::initialize(const char *vertex_shader_file, const char *fragment_shader_file, Visualizer *visualizer_ptr, const char *geometry_shader_file) {
    // ~~~~~~~~~~~~~~~ COMPILE SHADERS ~~~~~~~~~~~~~~~~~~~~~~~~~//

    assert(checkerrors());

    // Create the shaders
    unsigned int VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    unsigned int FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
    unsigned int GeometryShaderID = 0;

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_shader_file, std::ios::in);
    assert(VertexShaderStream.is_open());
    std::string Line;
    while (getline(VertexShaderStream, Line))
        VertexShaderCode += "\n" + Line;
    VertexShaderStream.close();

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_shader_file, std::ios::in);
    assert(FragmentShaderStream.is_open());
    Line = "";
    while (getline(FragmentShaderStream, Line))
        FragmentShaderCode += "\n" + Line;
    FragmentShaderStream.close();

    // Read the Geometry Shader code from the file (if provided)
    std::string GeometryShaderCode;
    bool hasGeometryShader = (geometry_shader_file != nullptr);
    if (hasGeometryShader) {
        GeometryShaderID = glCreateShader(GL_GEOMETRY_SHADER);
        std::ifstream GeometryShaderStream(geometry_shader_file, std::ios::in);
        assert(GeometryShaderStream.is_open());
        Line = "";
        while (getline(GeometryShaderStream, Line))
            GeometryShaderCode += "\n" + Line;
        GeometryShaderStream.close();
    }

    // Compile Vertex Shader
    char const *VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer, nullptr);
    glCompileShader(VertexShaderID);

    assert(checkerrors());

    // check vertex‐shader compile status
    GLint compileOK = GL_FALSE;
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &compileOK);
    if (compileOK != GL_TRUE) {
        GLint logLen = 0;
        glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<char> log(logLen);
        glGetShaderInfoLog(VertexShaderID, logLen, nullptr, log.data());
        fprintf(stderr, "Vertex shader compilation failed:\n%s\n", log.data());
        throw std::runtime_error("vertex shader compile error");
    }

    // Compile Fragment Shader
    char const *FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, nullptr);
    glCompileShader(FragmentShaderID);

    assert(checkerrors());

    // check fragment‐shader compile status
    compileOK = GL_FALSE;
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &compileOK);
    if (compileOK != GL_TRUE) {
        GLint logLen = 0;
        glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<char> log(logLen);
        glGetShaderInfoLog(FragmentShaderID, logLen, nullptr, log.data());
        fprintf(stderr, "Fragment shader compilation failed:\n%s\n", log.data());
        throw std::runtime_error("fragment shader compile error");
    }

    // Compile Geometry Shader (if provided)
    if (hasGeometryShader) {
        char const *GeometrySourcePointer = GeometryShaderCode.c_str();
        glShaderSource(GeometryShaderID, 1, &GeometrySourcePointer, nullptr);
        glCompileShader(GeometryShaderID);

        assert(checkerrors());

        // check geometry‐shader compile status
        compileOK = GL_FALSE;
        glGetShaderiv(GeometryShaderID, GL_COMPILE_STATUS, &compileOK);
        if (compileOK != GL_TRUE) {
            GLint logLen = 0;
            glGetShaderiv(GeometryShaderID, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetShaderInfoLog(GeometryShaderID, logLen, nullptr, log.data());
            fprintf(stderr, "Geometry shader compilation failed:\n%s\n", log.data());
            throw std::runtime_error("geometry shader compile error");
        }
    }

    // Link the program
    shaderID = glCreateProgram();
    glAttachShader(shaderID, VertexShaderID);
    glAttachShader(shaderID, FragmentShaderID);
    if (hasGeometryShader) {
        glAttachShader(shaderID, GeometryShaderID);
    }
    glLinkProgram(shaderID);

    assert(checkerrors());

    GLint linkOK = GL_FALSE;
    glGetProgramiv(shaderID, GL_LINK_STATUS, &linkOK);
    if (linkOK != GL_TRUE) {
        GLint logLen = 0;
        glGetProgramiv(shaderID, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<char> log(logLen);
        glGetProgramInfoLog(shaderID, logLen, nullptr, log.data());
        fprintf(stderr, "Shader program link failed:\n%s\n", log.data());
        throw std::runtime_error("program link error");
    }

    assert(checkerrors());

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);
    if (hasGeometryShader) {
        glDeleteShader(GeometryShaderID);
    }

    assert(checkerrors());

    // ~~~~~~~~~~~ Create a Vertex Array Object (VAO) ~~~~~~~~~~//
    vertex_array_IDs.resize(GeometryHandler::all_geometry_types.size());
    glGenVertexArrays(GeometryHandler::all_geometry_types.size(), vertex_array_IDs.data());

    assert(checkerrors());

    // set up vertex buffers

    int i = 0;
    for (const auto &geometry_type: GeometryHandler::all_geometry_types) {
        glBindVertexArray(vertex_array_IDs.at(i));

        // 1st attribute buffer : vertex positions
        glBindBuffer(GL_ARRAY_BUFFER, visualizer_ptr->vertex_buffer.at(i));
        glEnableVertexAttribArray(0); // vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        // 2nd attribute buffer : vertex uv
        glBindBuffer(GL_ARRAY_BUFFER, visualizer_ptr->uv_buffer.at(i));
        glEnableVertexAttribArray(1); // uv
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        // 3rd attribute buffer : face index
        glBindBuffer(GL_ARRAY_BUFFER, visualizer_ptr->face_index_buffer.at(i));
        glEnableVertexAttribArray(2); // face index
        glVertexAttribIPointer(2, 1, GL_INT, 0, nullptr);

        i++;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    assert(checkerrors());

    glUseProgram(shaderID);

    assert(checkerrors());

    // ~~~~~~~~~~~ Primary Shader Uniforms ~~~~~~~~~~//

    // Transformation Matrix
    transformMatrixUniform = glGetUniformLocation(shaderID, "MVP");

    // View and Projection Matrices (for sky geometry transformations)
    viewMatrixUniform = glGetUniformLocation(shaderID, "view");
    projectionMatrixUniform = glGetUniformLocation(shaderID, "projection");

    // Depth Bias Matrix (for shadows)
    depthBiasUniform = glGetUniformLocation(shaderID, "DepthBiasMVP");

    // Texture Sampler
    textureUniform = glGetUniformLocation(shaderID, "textureSampler");

    // Shadow Map Sampler
    shadowmapUniform = glGetUniformLocation(shaderID, "shadowMap");
    glUniform1i(shadowmapUniform, 1);

    // Unit vector in the direction of the light (sun)
    lightDirectionUniform = glGetUniformLocation(shaderID, "lightDirection");
    glUniform3f(lightDirectionUniform, 0, 0, 1); // Default is directly above

    // Lighting model used for shading primitives
    lightingModelUniform = glGetUniformLocation(shaderID, "lightingModel");
    glUniform1i(lightingModelUniform, 0); // Default is none

    RboundUniform = glGetUniformLocation(shaderID, "Rbound");
    glUniform1i(RboundUniform, 0);

    // Lighting intensity factor
    lightIntensityUniform = glGetUniformLocation(shaderID, "lightIntensity");
    glUniform1f(lightIntensityUniform, 1.f);

    // Texture (u,v) rescaling factor
    uvRescaleUniform = glGetUniformLocation(shaderID, "uv_rescale");

    // Cache texture buffer uniform locations to avoid glGetUniformLocation during rendering
    // This prevents macOS OpenGL warnings about texture unit binding issues
    colorTextureObjectUniform = glGetUniformLocation(shaderID, "color_texture_object");
    normalTextureObjectUniform = glGetUniformLocation(shaderID, "normal_texture_object");
    textureFlagTextureObjectUniform = glGetUniformLocation(shaderID, "texture_flag_texture_object");
    textureIDTextureObjectUniform = glGetUniformLocation(shaderID, "texture_ID_texture_object");
    coordinateFlagTextureObjectUniform = glGetUniformLocation(shaderID, "coordinate_flag_texture_object");
    skyGeometryFlagTextureObjectUniform = glGetUniformLocation(shaderID, "sky_geometry_flag_texture_object");
    hiddenFlagTextureObjectUniform = glGetUniformLocation(shaderID, "hidden_flag_texture_object");

    // Set the texture unit assignments for texture buffers
    if (colorTextureObjectUniform >= 0)
        glUniform1i(colorTextureObjectUniform, 3);
    if (normalTextureObjectUniform >= 0)
        glUniform1i(normalTextureObjectUniform, 4);
    if (textureFlagTextureObjectUniform >= 0)
        glUniform1i(textureFlagTextureObjectUniform, 5);
    if (textureIDTextureObjectUniform >= 0)
        glUniform1i(textureIDTextureObjectUniform, 6);
    if (coordinateFlagTextureObjectUniform >= 0)
        glUniform1i(coordinateFlagTextureObjectUniform, 7);
    if (skyGeometryFlagTextureObjectUniform >= 0)
        glUniform1i(skyGeometryFlagTextureObjectUniform, 2);
    if (hiddenFlagTextureObjectUniform >= 0)
        glUniform1i(hiddenFlagTextureObjectUniform, 8);

    assert(checkerrors());

    initialized = true;
}

Shader::~Shader() {
    if (!initialized) {
        return;
    }
    glDeleteVertexArrays(vertex_array_IDs.size(), vertex_array_IDs.data());
    glDeleteProgram(shaderID);
}

void Shader::disableTextures() const {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Shader::enableTextureMaps() const {
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(textureUniform, 0);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Shader::enableTextureMasks() const {
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(textureUniform, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Shader::setTransformationMatrix(const glm::mat4 &matrix) const {
    glUniformMatrix4fv(transformMatrixUniform, 1, GL_FALSE, &matrix[0][0]);
}

void Shader::setViewMatrix(const glm::mat4 &matrix) const {
    glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, &matrix[0][0]);
}

void Shader::setProjectionMatrix(const glm::mat4 &matrix) const {
    glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, &matrix[0][0]);
}

void Shader::setDepthBiasMatrix(const glm::mat4 &matrix) const {
    glUniformMatrix4fv(depthBiasUniform, 1, GL_FALSE, &matrix[0][0]);
}

void Shader::setLightDirection(const helios::vec3 &direction) const {
    glUniform3f(lightDirectionUniform, direction.x, direction.y, direction.z);
}

void Shader::setLightingModel(uint lightingmodel) const {
    glUniform1i(lightingModelUniform, lightingmodel);
}

void Shader::setLightIntensity(float lightintensity) const {
    glUniform1f(lightIntensityUniform, lightintensity);
}

void Shader::useShader() const {
    glUseProgram(shaderID);
}

void Visualizer::framebufferResizeCallback(GLFWwindow *window, int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }
    auto *viz = static_cast<Visualizer *>(glfwGetWindowUserPointer(window));
    if (viz != nullptr) {
        viz->Wframebuffer = static_cast<uint>(width);
        viz->Hframebuffer = static_cast<uint>(height);
    }
}

void Visualizer::windowResizeCallback(GLFWwindow *window, int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }
    auto *viz = static_cast<Visualizer *>(glfwGetWindowUserPointer(window));
    if (viz != nullptr) {
        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        if (fbw != width || fbh != height) {
            glfwSetWindowSize(window, width, height);
            fbw = width;
            fbh = height;
        }
        viz->Wdisplay = static_cast<uint>(width);
        viz->Hdisplay = static_cast<uint>(height);
        viz->Wframebuffer = static_cast<uint>(fbw);
        viz->Hframebuffer = static_cast<uint>(fbh);
        viz->updateWatermark();
        viz->updateNavigationGizmo();
        viz->transferBufferData();
    }
}

void Visualizer::updateWatermark() {
    if (!isWatermarkVisible) {
        if (watermark_ID != 0) {
            geometry_handler.deleteGeometry(watermark_ID);
            watermark_ID = 0;
        }
        return;
    }

    constexpr float texture_aspect = 675.f / 195.f; // image width / height

    float window_aspect = float(Wframebuffer) / float(Hframebuffer);
    float width = 0.07f * texture_aspect / window_aspect;
    if (watermark_ID != 0) {
        geometry_handler.deleteGeometry(watermark_ID);
    }
    std::string watermarkPath = helios::resolvePluginAsset("visualizer", "textures/Helios_watermark.png").string();
    watermark_ID = addRectangleByCenter(make_vec3(0.75f * width, 0.95f, 0), make_vec2(width, 0.07), make_SphericalCoord(0, 0), watermarkPath.c_str(), COORDINATES_WINDOW_NORMALIZED);
}

void Visualizer::updateColorbar() {
    // Check if colorbar should be displayed
    if (colorbar_flag != 2) {
        // If disabled, delete geometry if it exists
        if (!colorbar_IDs.empty()) {
            geometry_handler.deleteGeometry(colorbar_IDs);
            colorbar_IDs.clear();
        }
        return;
    }

    // Calculate window aspect ratio
    float window_aspect = float(Wframebuffer) / float(Hframebuffer);

    // Apply aspect ratio correction to width to maintain intended aspect ratio
    // Formula: corrected_width = height * intended_aspect / window_aspect
    // This keeps the colorbar height constant in normalized coordinates while adjusting width
    float corrected_width = colorbar_size.y * colorbar_intended_aspect_ratio / window_aspect;

    // Delete old colorbar geometry
    if (!colorbar_IDs.empty()) {
        geometry_handler.deleteGeometry(colorbar_IDs);
        colorbar_IDs.clear();
    }

    // Create new colorbar with aspect-corrected size
    colorbar_IDs = addColorbarByCenter(colorbar_title.c_str(), make_vec2(corrected_width, colorbar_size.y), colorbar_position, colorbar_fontcolor, colormap_current);
}


bool lbutton_down = false;
bool rbutton_down = false;
bool mbutton_down = false;
double startX, startY;
double scrollX, scrollY;
bool scroll = false;

// Click detection state for navigation gizmo
double click_startX = 0.0, click_startY = 0.0;
bool potential_click = false;
const double click_threshold = 5.0; // Maximum mouse movement in pixels to register as click


void mouseCallback(GLFWwindow *window, int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        glfwGetCursorPos(window, &startX, &startY);
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (GLFW_PRESS == action) {
            lbutton_down = true;
            // Track potential click for gizmo interaction
            glfwGetCursorPos(window, &click_startX, &click_startY);
            potential_click = true;
        } else if (GLFW_RELEASE == action) {
            lbutton_down = false;
            // Check if this was a click (minimal movement) rather than a drag
            if (potential_click) {
                double releaseX, releaseY;
                glfwGetCursorPos(window, &releaseX, &releaseY);
                double dx_click = releaseX - click_startX;
                double dy_click = releaseY - click_startY;
                double movement = std::sqrt(dx_click * dx_click + dy_click * dy_click);

                if (movement < click_threshold) {
                    // This is a click - check if it hit the navigation gizmo
                    auto *visualizer = static_cast<Visualizer *>(glfwGetWindowUserPointer(window));
                    if (visualizer != nullptr) {
                        visualizer->handleGizmoClick(releaseX, releaseY);
                    }
                }
                potential_click = false;
            }
        }
    } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (GLFW_PRESS == action) {
            mbutton_down = true;
        } else if (GLFW_RELEASE == action) {
            mbutton_down = false;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (GLFW_PRESS == action) {
            rbutton_down = true;
        } else if (GLFW_RELEASE == action) {
            rbutton_down = false;
        }
    }
}


void cursorCallback(GLFWwindow *window, double xpos, double ypos) {
    if (rbutton_down) {
        dx = xpos - startX;
        dy = ypos - startY;
    } else if (lbutton_down || mbutton_down) {
        dphi = scast<float>(xpos - startX);
        dtheta = scast<float>(ypos - startY);

        // If mouse moved too much during left button press, invalidate potential click
        if (potential_click && lbutton_down) {
            double dx_click = xpos - click_startX;
            double dy_click = ypos - click_startY;
            double movement = std::sqrt(dx_click * dx_click + dy_click * dy_click);
            if (movement >= click_threshold) {
                potential_click = false;
            }
        }
    } else {
        dphi = dtheta = 0.f;

        // Check for hover over navigation gizmo bubbles when no buttons are pressed
        auto *visualizer = static_cast<Visualizer *>(glfwGetWindowUserPointer(window));
        if (visualizer != nullptr) {
            visualizer->handleGizmoHover(xpos, ypos);
        }
    }
    startX = xpos;
    startY = ypos;
}


void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    dscroll = scast<float>(yoffset);
    scrollY = yoffset;
    if (yoffset > 0.0 || yoffset < 0.0) {
        scroll = true;
    } else {
        scroll = false;
    }
}


void Visualizer::getViewKeystrokes(vec3 &eye, vec3 &center) {
    vec3 forward = center - eye;
    forward = forward.normalize();

    vec3 right = cross(forward, vec3(0, 0, 1));
    right = right.normalize();

    vec3 up = cross(right, forward);
    up = up.normalize();

    SphericalCoord Spherical = cart2sphere(eye - center);
    float radius = Spherical.radius;
    float theta = Spherical.elevation;
    float phi = Spherical.azimuth;

    phi += PI_F * (dphi / 160.f);
    if (dtheta > 0 && theta + PI_F / 80.f < 0.49f * PI_F) {
        theta += PI_F * (dtheta / 120.f);
    } else if (dtheta < 0 && theta > -0.25 * PI_F) {
        theta += PI_F * (dtheta / 120.f);
    }
    dtheta = dphi = 0;
    if (dx != 0.f) {
        center -= 0.025f * dx * right;
    }
    if (dy != 0.f) {
        center += 0.025f * dy * up;
    }
    dx = dy = 0.f;
    if (scroll) {
        if (dscroll > 0.0f) {
            radius = (radius * 0.9f > minimum_view_radius) ? radius * 0.9f : minimum_view_radius;
        } else {
            radius *= 1.1f;
        }
    }
    scroll = false;

    auto *_window = scast<GLFWwindow *>(window);

    //----- Holding SPACEBAR -----//
    if (glfwGetKey(_window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        // Move center to the left - SPACE + LEFT KEY
        if (glfwGetKey(_window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            center.x += 0.1f * sin(phi);
            center.y += 0.1f * cos(phi);
        }
        // Move center to the right - SPACE + RIGHT KEY
        else if (glfwGetKey(_window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            center.x -= 0.1f * sin(phi);
            center.y -= 0.1f * cos(phi);
        }
        // Move center upward - SPACE + UP KEY
        else if (glfwGetKey(_window, GLFW_KEY_UP) == GLFW_PRESS) {
            center.z += 0.2f;
        }
        // Move center downward - SPACE + DOWN KEY
        else if (glfwGetKey(_window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            center.z -= 0.2f;
        }

        //----- Not Holding SPACEBAR -----//
    } else {
        //   Orbit left - LEFT ARROW KEY
        if (glfwGetKey(_window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            phi += PI_F / 40.f;
        }
        // Orbit right - RIGHT ARROW KEY
        else if (glfwGetKey(_window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            phi -= PI_F / 40.f;
        }

        // Increase Elevation - UP ARROW KEY
        else if (glfwGetKey(_window, GLFW_KEY_UP) == GLFW_PRESS) {
            if (theta + PI_F / 80.f < 0.49f * PI_F) {
                theta += PI_F / 80.f;
            }
        }
        // Decrease Elevation - DOWN ARROW KEY
        else if (glfwGetKey(_window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            if (theta > -0.25 * PI_F) {
                theta -= PI_F / 80.f;
            }
        }

        //   Zoom in - "+" KEY
        if (glfwGetKey(_window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
            radius = (radius * 0.9f > minimum_view_radius) ? radius * 0.9f : minimum_view_radius;
        }
        // Zoom out - "-" KEY
        else if (glfwGetKey(_window, GLFW_KEY_MINUS) == GLFW_PRESS) {
            radius *= 1.1;
        }
    }

    if (glfwGetKey(_window, GLFW_KEY_P) == GLFW_PRESS) {
        std::cout << "View is angle: (R,theta,phi)=(" << radius << "," << theta << "," << phi << ") at from position (" << camera_eye_location.x << "," << camera_eye_location.y << "," << camera_eye_location.z << ") looking at (" << center.x << ","
                  << center.y << "," << center.z << ")" << std::endl;
    }

    camera_eye_location = sphere2cart(make_SphericalCoord(radius, theta, phi)) + center;
}

void Visualizer::cullPointsByFrustum() {
    const std::vector<float> *vertex_data = geometry_handler.getVertexData_ptr(GeometryHandler::GEOMETRY_TYPE_POINT);
    if (!vertex_data || vertex_data->empty()) {
        return;
    }

    std::vector<glm::vec4> frustum_planes = extractFrustumPlanes();

    // Check each point against all 6 frustum planes
    size_t point_count = vertex_data->size() / 3;
    for (size_t i = 0; i < point_count; ++i) {
        glm::vec3 point(vertex_data->at(i * 3), vertex_data->at(i * 3 + 1), vertex_data->at(i * 3 + 2));

        bool inside_frustum = true;
        for (const auto &plane: frustum_planes) {
            // Plane equation: ax + by + cz + d = 0
            // Point is outside if dot(plane.xyz, point) + plane.w < 0
            if (glm::dot(glm::vec3(plane), point) + plane.w < 0) {
                inside_frustum = false;
                break;
            }
        }

        // Find the UUID for this point index and update visibility
        std::vector<size_t> all_UUIDs = geometry_handler.getAllGeometryIDs();
        size_t point_index = 0;
        for (size_t UUID: all_UUIDs) {
            if (geometry_handler.getIndexMap(UUID).geometry_type == GeometryHandler::GEOMETRY_TYPE_POINT) {
                if (point_index == i) {
                    geometry_handler.setVisibility(UUID, inside_frustum);
                    break;
                }
                point_index++;
            }
        }
    }
}

void Visualizer::cullPointsByDistance(float maxDistance, float lodFactor) {
    const std::vector<float> *vertex_data = geometry_handler.getVertexData_ptr(GeometryHandler::GEOMETRY_TYPE_POINT);
    if (!vertex_data || vertex_data->empty()) {
        return;
    }

    glm::vec3 camera_pos(camera_eye_location.x, camera_eye_location.y, camera_eye_location.z);

    // Apply distance-based culling with level-of-detail and adaptive sizing
    size_t point_count = vertex_data->size() / 3;
    for (size_t i = 0; i < point_count; ++i) {
        glm::vec3 point(vertex_data->at(i * 3), vertex_data->at(i * 3 + 1), vertex_data->at(i * 3 + 2));

        float distance = glm::length(point - camera_pos);
        bool should_render = true;
        float adaptive_size = 1.0f; // Default point size

        // Cull points beyond max distance
        if (distance > maxDistance) {
            should_render = false;
        }
        // Apply level-of-detail culling and adaptive sizing
        else if (distance > maxDistance * 0.3f) { // Start LOD at 30% of max distance
            float distance_ratio = distance / maxDistance;
            float lod_threshold = distance_ratio * lodFactor;

            // Cull every Nth point based on distance
            if ((i % static_cast<size_t>(std::max(1.0f, lod_threshold))) != 0) {
                should_render = false;
            } else {
                // For distant points that we keep, increase their size to maintain visual coverage
                adaptive_size = 1.0f + (distance_ratio * 3.0f); // Scale up to 4x size for far points
            }
        }

        // Find the UUID for this point index and update visibility and size
        std::vector<size_t> all_UUIDs = geometry_handler.getAllGeometryIDs();
        size_t point_index = 0;
        for (size_t UUID: all_UUIDs) {
            if (geometry_handler.getIndexMap(UUID).geometry_type == GeometryHandler::GEOMETRY_TYPE_POINT) {
                if (point_index == i) {
                    geometry_handler.setVisibility(UUID, should_render);
                    if (should_render) {
                        // Apply adaptive sizing to maintain visual quality
                        float original_size = geometry_handler.getSize(UUID);
                        if (original_size <= 0)
                            original_size = 1.0f;
                        geometry_handler.setSize(UUID, original_size * adaptive_size);
                    }
                    break;
                }
                point_index++;
            }
        }
    }
}

void Visualizer::updatePointCulling() {
    // Update total point count
    points_total_count = geometry_handler.getPointCount();

    // Only perform culling if enabled and we have enough points
    if (!point_culling_enabled || points_total_count < point_culling_threshold) {
        points_rendered_count = points_total_count;
        last_culling_time_ms = 0;
        return;
    }

    // Measure culling performance
    auto start_time = std::chrono::high_resolution_clock::now();

    // Apply frustum culling first
    cullPointsByFrustum();

    // Calculate max distance if not set
    float max_distance = point_max_render_distance;
    if (max_distance <= 0) {
        helios::vec2 xbounds, ybounds, zbounds;
        geometry_handler.getDomainBoundingBox(xbounds, ybounds, zbounds);
        float scene_size = std::max({xbounds.y - xbounds.x, ybounds.y - ybounds.x, zbounds.y - zbounds.x});
        max_distance = scene_size * 5.0f; // Render points up to 5x scene size
    }

    // Apply distance-based culling with configurable parameters
    cullPointsByDistance(max_distance, point_lod_factor);

    // Update metrics
    points_rendered_count = geometry_handler.getPointCount(false); // Count only visible points

    auto end_time = std::chrono::high_resolution_clock::now();
    last_culling_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
}

std::vector<glm::vec4> Visualizer::extractFrustumPlanes() const {
    std::vector<glm::vec4> planes(6);

    // Extract frustum planes from the MVP matrix
    glm::mat4 mvp = cameraProjectionMatrix * cameraViewMatrix;

    // Left plane
    planes[0] = glm::vec4(mvp[0][3] + mvp[0][0], mvp[1][3] + mvp[1][0], mvp[2][3] + mvp[2][0], mvp[3][3] + mvp[3][0]);
    // Right plane
    planes[1] = glm::vec4(mvp[0][3] - mvp[0][0], mvp[1][3] - mvp[1][0], mvp[2][3] - mvp[2][0], mvp[3][3] - mvp[3][0]);
    // Bottom plane
    planes[2] = glm::vec4(mvp[0][3] + mvp[0][1], mvp[1][3] + mvp[1][1], mvp[2][3] + mvp[2][1], mvp[3][3] + mvp[3][1]);
    // Top plane
    planes[3] = glm::vec4(mvp[0][3] - mvp[0][1], mvp[1][3] - mvp[1][1], mvp[2][3] - mvp[2][1], mvp[3][3] - mvp[3][1]);
    // Near plane
    planes[4] = glm::vec4(mvp[0][3] + mvp[0][2], mvp[1][3] + mvp[1][2], mvp[2][3] + mvp[2][2], mvp[3][3] + mvp[3][2]);
    // Far plane
    planes[5] = glm::vec4(mvp[0][3] - mvp[0][2], mvp[1][3] - mvp[1][2], mvp[2][3] - mvp[2][2], mvp[3][3] - mvp[3][2]);

    // Normalize the planes
    for (auto &plane: planes) {
        float length = glm::length(glm::vec3(plane));
        if (length > 0) {
            plane /= length;
        }
    }

    return planes;
}

void Visualizer::setPointCullingEnabled(bool enabled) {
    point_culling_enabled = enabled;
}

void Visualizer::setPointCullingThreshold(size_t threshold) {
    point_culling_threshold = threshold;
}

void Visualizer::setPointMaxRenderDistance(float distance) {
    point_max_render_distance = distance;
}

void Visualizer::setPointLODFactor(float factor) {
    point_lod_factor = factor;
}

void Visualizer::getPointRenderingMetrics(size_t &total_points, size_t &rendered_points, float &culling_time_ms) const {
    total_points = points_total_count;
    rendered_points = points_rendered_count;
    culling_time_ms = last_culling_time_ms;
}

std::string errorString(GLenum err) {
    std::string message;
    message.assign("");

    if (err == GL_INVALID_ENUM) {
        message.assign("GL_INVALID_ENUM - An unacceptable value is specified for an enumerated argument.");
    } else if (err == GL_INVALID_VALUE) {
        message.assign("GL_INVALID_VALUE - A numeric argument is out of range.");
    } else if (err == GL_INVALID_OPERATION) {
        message.assign("GL_INVALID_OPERATION - The specified operation is not allowed in the current state.");
    } else if (err == GL_STACK_OVERFLOW) {
        message.assign("GL_STACK_OVERFLOW - This command would cause a stack overflow.");
    } else if (err == GL_STACK_UNDERFLOW) {
        message.assign("GL_STACK_UNDERFLOW - This command would cause a stack underflow.");
    } else if (err == GL_OUT_OF_MEMORY) {
        message.assign("GL_OUT_OF_MEMORY - There is not enough memory left to execute the command.");
    } else if (err == GL_TABLE_TOO_LARGE) {
        message.assign("GL_TABLE_TOO_LARGE - The specified table exceeds the implementation's maximum supported table size.");
    }

    return message;
}

int checkerrors() {
    GLenum err;
    int err_count = 0;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "glError #" << err_count << ": " << errorString(err) << std::endl;
        err_count++;
    }
    if (err_count > 0) {
        return 0;
    } else {
        return 1;
    }
}

// Safe error checking that throws exceptions instead of using assert
void check_opengl_errors_safe(const std::string &context) {
    if (!checkerrors()) {
        helios_runtime_error("ERROR (Visualizer): OpenGL errors detected in " + context + ". Check console output for specific error details.");
    }
}
