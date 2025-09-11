/** \file "VisualizerRendering.cpp" Visualizer rendering and display functions.

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

void Visualizer::printWindow(const char *outfile) const {
    std::string outfile_str = outfile;

    std::string ext = getFileExtension(outfile_str);
    if (ext.empty()) {
        outfile_str += ".jpeg";
    }

    if (!validateOutputPath(outfile_str, {".jpeg", ".jpg", ".JPEG", ".JPG"})) {
        helios_runtime_error("ERROR (Visualizer::printWindow): Output path is not valid or does not have a valid image extension (.jpeg, .jpg).");
    }

    // Don't swap buffers again - content is already displayed by plotUpdate()
    // Just ensure rendering is complete and read from the front buffer
    if (window != nullptr && !headless) {
        glfwPollEvents();
    }

    // Ensure rendering is complete
    glFinish();

    // Read pixels from front buffer (where the displayed content is)
    write_JPEG_file(outfile_str.c_str(), Wframebuffer, Hframebuffer, message_flag);
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
    std::vector<GLubyte> buff;
    buff.resize(3 * Wframebuffer * Hframebuffer);

#if defined(__APPLE__)
    constexpr GLenum read_buf = GL_FRONT;
#else
    constexpr GLenum read_buf = GL_BACK;
#endif
    glReadBuffer(read_buf);
    glReadPixels(0, 0, GLsizei(Wframebuffer), GLsizei(Hframebuffer), GL_RGB, GL_UNSIGNED_BYTE, &buff[0]);
    glFinish();

    // assert( checkerrors() );

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

    // Update
    if (colorbar_flag == 2) {
        if (!colorbar_IDs.empty()) {
            geometry_handler.deleteGeometry(colorbar_IDs);
            colorbar_IDs.clear();
        }
        colorbar_IDs = addColorbarByCenter(colorbar_title.c_str(), colorbar_size, colorbar_position, colorbar_fontcolor, colormap_current);
    }


    // Watermark
    updateWatermark();

    transferBufferData();

    assert(checkerrors());

    bool shadow_flag = false;
    for (const auto &model: primaryLightingModel) {
        if (model == Visualizer::LIGHTING_PHONG_SHADOWED) {
            shadow_flag = true;
            break;
        }
    }

    glm::mat4 depthMVP;

    assert(checkerrors());

    std::vector<vec3> camera_output;

    glfwShowWindow((GLFWwindow *) window);

    do {
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

        glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 0.0f);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        primaryShader.useShader();

        glm::mat4 DepthBiasMVP = biasMatrix * depthMVP;

        primaryShader.setDepthBiasMatrix(DepthBiasMVP);

        updatePerspectiveTransformation(false);

        primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);

        primaryShader.enableTextureMaps();
        primaryShader.enableTextureMasks();

        primaryShader.setLightingModel(primaryLightingModel.at(0));
        primaryShader.setLightIntensity(lightintensity);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glUniform1i(primaryShader.shadowmapUniform, 1);
        glActiveTexture(GL_TEXTURE0);

        render(false);

        glfwPollEvents();
        getViewKeystrokes(camera_eye_location, camera_lookat_center);

        glfwSwapBuffers((GLFWwindow *) window);

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

    bool shadow_flag = false;
    for (const auto &model: primaryLightingModel) {
        if (model == Visualizer::LIGHTING_PHONG_SHADOWED) {
            shadow_flag = true;
            break;
        }
    }

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

    glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 0.0f);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    primaryShader.useShader();

    glm::mat4 DepthBiasMVP = biasMatrix * depthMVP;

    primaryShader.setDepthBiasMatrix(DepthBiasMVP);

    updatePerspectiveTransformation(false);

    primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);

    primaryShader.enableTextureMaps();
    primaryShader.enableTextureMasks();

    primaryShader.setLightingModel(primaryLightingModel.at(0));
    primaryShader.setLightIntensity(lightintensity);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glUniform1i(primaryShader.shadowmapUniform, 1);
    glActiveTexture(GL_TEXTURE0);

    render(false);

    // glfwPollEvents();
    if (getKeystrokes) {
        getViewKeystrokes(camera_eye_location, camera_lookat_center);
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
        }
    };

    auto ensureTextureBuffer = [](GLuint buf, GLuint tex, GLenum format, GLsizeiptr size, const void *data) {
        glBindBuffer(GL_TEXTURE_BUFFER, buf);
        GLint current_size = 0;
        glGetBufferParameteriv(GL_TEXTURE_BUFFER, GL_BUFFER_SIZE, &current_size);
        if (current_size != size) {
            glBufferData(GL_TEXTURE_BUFFER, size, data, GL_STATIC_DRAW);
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
        const auto *visible_flag_data = geometry_handler.getVisibilityFlagData_ptr(geometry_type);

        ensureArrayBuffer(vertex_buffer.at(gi), GL_ARRAY_BUFFER, vertex_data->size() * sizeof(GLfloat), vertex_data->data());
        ensureArrayBuffer(uv_buffer.at(gi), GL_ARRAY_BUFFER, uv_data->size() * sizeof(GLfloat), uv_data->data());
        ensureArrayBuffer(face_index_buffer.at(gi), GL_ARRAY_BUFFER, face_index_data->size() * sizeof(GLint), face_index_data->data());
        ensureTextureBuffer(color_buffer.at(gi), color_texture_object.at(gi), GL_RGBA32F, color_data->size() * sizeof(GLfloat), color_data->data());
        ensureTextureBuffer(normal_buffer.at(gi), normal_texture_object.at(gi), GL_RGB32F, normal_data->size() * sizeof(GLfloat), normal_data->data());
        ensureTextureBuffer(texture_flag_buffer.at(gi), texture_flag_texture_object.at(gi), GL_R32I, texture_flag_data->size() * sizeof(GLint), texture_flag_data->data());
        ensureTextureBuffer(texture_ID_buffer.at(gi), texture_ID_texture_object.at(gi), GL_R32I, texture_ID_data->size() * sizeof(GLint), texture_ID_data->data());
        ensureTextureBuffer(coordinate_flag_buffer.at(gi), coordinate_flag_texture_object.at(gi), GL_R32I, coordinate_flag_data->size() * sizeof(GLint), coordinate_flag_data->data());
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
    if (texArray == 0) {
        glGenTextures(1, &texArray);
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);

    const size_t layers = std::max<size_t>(1, texture_manager.size());
    if (layers != texture_array_layers) {
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, maximum_texture_size.x, maximum_texture_size.y, layers);
        texture_array_layers = layers;
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
        glUniform1i(glGetUniformLocation(current_shader_program, "color_texture_object"), 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(triangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "normal_texture_object"), 4);

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(triangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "texture_flag_texture_object"), 5);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(triangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "texture_ID_texture_object"), 6);

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(triangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "coordinate_flag_texture_object"), 7);

        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(triangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "hidden_flag_texture_object"), 8);

        glBindVertexArray(primaryShader.vertex_array_IDs.at(triangle_ind));
        assert(checkerrors());
        glDrawArrays(GL_TRIANGLES, 0, triangle_count * 3);
    }

    assert(checkerrors());

    //--- Rectangles---//

    if (rectangle_count > 0) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(rectangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "color_texture_object"), 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(rectangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "normal_texture_object"), 4);

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(rectangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "texture_flag_texture_object"), 5);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(rectangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "texture_ID_texture_object"), 6);

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(rectangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "coordinate_flag_texture_object"), 7);

        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(rectangle_ind));
        glUniform1i(glGetUniformLocation(current_shader_program, "hidden_flag_texture_object"), 8);

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

        opaque_firsts.reserve(rectangle_count);
        opaque_counts.reserve(rectangle_count);
        transparent_rects.reserve(rectangle_count);

        for (size_t i = 0; i < rectangle_count; ++i) {
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
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(line_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "color_texture_object"), 3);

            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(line_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "normal_texture_object"), 4);

            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(line_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "texture_flag_texture_object"), 5);

            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(line_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "texture_ID_texture_object"), 6);

            glActiveTexture(GL_TEXTURE7);
            glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(line_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "coordinate_flag_texture_object"), 7);

            glActiveTexture(GL_TEXTURE8);
            glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(line_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "hidden_flag_texture_object"), 8);

            glBindVertexArray(primaryShader.vertex_array_IDs.at(line_ind));

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

                // Render each width group separately
                for (const auto &group: width_groups) {
                    float width = group.first;
                    const std::vector<size_t> &line_indices = group.second;

                    glLineWidth(width);

                    // For simplicity, render each line individually
                    // In a more optimized implementation, we would batch lines with same width
                    for (size_t line_idx: line_indices) {
                        glDrawArrays(GL_LINES, line_idx * 2, 2);
                    }
                }
            } else {
                // Fallback to default behavior if no size data available
                glLineWidth(1);
                glDrawArrays(GL_LINES, 0, line_count * 2);
            }
        }

        //--- Points ---//

        if (point_count > 0) {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(point_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "color_texture_object"), 3);

            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(point_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "normal_texture_object"), 4);

            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(point_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "texture_flag_texture_object"), 5);

            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(point_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "texture_ID_texture_object"), 6);

            glActiveTexture(GL_TEXTURE7);
            glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(point_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "coordinate_flag_texture_object"), 7);

            glActiveTexture(GL_TEXTURE8);
            glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(point_ind));
            glUniform1i(glGetUniformLocation(current_shader_program, "hidden_flag_texture_object"), 8);

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

    // Update
    if (colorbar_flag == 2) {
        if (!colorbar_IDs.empty()) {
            geometry_handler.deleteGeometry(colorbar_IDs);
            colorbar_IDs.clear();
        }
        colorbar_IDs = addColorbarByCenter(colorbar_title.c_str(), colorbar_size, colorbar_position, colorbar_fontcolor, colormap_current);
    }

    // Watermark
    updateWatermark();

    transferBufferData();

    bool shadow_flag = false;
    for (const auto &model: primaryLightingModel) {
        if (model == Visualizer::LIGHTING_PHONG_SHADOWED) {
            shadow_flag = true;
            break;
        }
    }

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

    glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 0.0f);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    primaryShader.useShader();

    glm::mat4 DepthBiasMVP = biasMatrix * depthMVP;

    primaryShader.setDepthBiasMatrix(DepthBiasMVP);

    updatePerspectiveTransformation(false);

    primaryShader.setTransformationMatrix(perspectiveTransformationMatrix);

    primaryShader.enableTextureMaps();
    primaryShader.enableTextureMasks();

    primaryShader.setLightingModel(primaryLightingModel.at(0));
    primaryShader.setLightIntensity(lightintensity);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glUniform1i(primaryShader.shadowmapUniform, 1);
    glActiveTexture(GL_TEXTURE0);

    render(false);

    glfwPollEvents();
    getViewKeystrokes(camera_eye_location, camera_lookat_center);

    int width, height;
    glfwGetFramebufferSize((GLFWwindow *) window, &width, &height);
    Wframebuffer = width;
    Hframebuffer = height;

    glfwSwapBuffers((GLFWwindow *) window);

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

void Shader::initialize(const char *vertex_shader_file, const char *fragment_shader_file, Visualizer *visualizer_ptr) {
    // ~~~~~~~~~~~~~~~ COMPILE SHADERS ~~~~~~~~~~~~~~~~~~~~~~~~~//

    assert(checkerrors());

    // Create the shaders
    unsigned int VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    unsigned int FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

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

    // Link the program
    shaderID = glCreateProgram();
    glAttachShader(shaderID, VertexShaderID);
    glAttachShader(shaderID, FragmentShaderID);
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
    watermark_ID = addRectangleByCenter(make_vec3(0.75f * width, 0.95f, 0), make_vec2(width, 0.07), make_SphericalCoord(0, 0), "plugins/visualizer/textures/Helios_watermark.png", COORDINATES_WINDOW_NORMALIZED);
}


bool lbutton_down = false;
bool rbutton_down = false;
bool mbutton_down = false;
double startX, startY;
double scrollX, scrollY;
bool scroll = false;


void mouseCallback(GLFWwindow *window, int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        glfwGetCursorPos(window, &startX, &startY);
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (GLFW_PRESS == action) {
            lbutton_down = true;
        } else if (GLFW_RELEASE == action) {
            lbutton_down = false;
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
    } else {
        dphi = dtheta = 0.f;
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
