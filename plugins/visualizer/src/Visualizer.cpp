/** \file "Visualizer.cpp" Visualizer plugin declarations.

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

// JPEG Libraries (reading and writing JPEG images)
#include <cstdio> //<-- note libjpeg requires this header be included before its headers.
#include <jpeglib.h>
// #include <setjmp.h>

// PNG Libraries (reading and writing PNG images)
#ifndef _WIN32
#include <unistd.h>
#endif
#define PNG_DEBUG 3
#define PNG_SKIP_SETJMP_CHECK 1
#include <png.h>

#include "Visualizer.h"

using namespace helios;

struct my_error_mgr {
    jpeg_error_mgr pub; /* "public" fields */
};

using my_error_ptr = my_error_mgr *;

METHODDEF(void) my_error_exit(j_common_ptr cinfo) {
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    throw std::runtime_error(buffer);
}

int read_JPEG_file(const char *filename, std::vector<unsigned char> &texture, uint &height, uint &width) {
    std::string fn = filename;
    if (fn.substr(fn.find_last_of('.') + 1) != "jpg" && fn.substr(fn.find_last_of('.') + 1) != "jpeg" && fn.substr(fn.find_last_of('.') + 1) != "JPG" && fn.substr(fn.find_last_of('.') + 1) != "JPEG") {
        helios_runtime_error("ERROR (read_JPEG_file): File " + fn + " is not JPEG format.");
    }

    jpeg_decompress_struct cinfo{};
    my_error_mgr jerr{};
    JSAMPARRAY buffer;
    uint row_stride;

    std::unique_ptr<FILE, decltype(&fclose)> infile(fopen(filename, "rb"), fclose);
    if (!infile) {
        fprintf(stderr, "can't open %s\n", filename);
        return 0;
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    try {
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, infile.get());
        (void) jpeg_read_header(&cinfo, TRUE);

        (void) jpeg_start_decompress(&cinfo);

        row_stride = cinfo.output_width * cinfo.output_components;
        buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

        width = cinfo.output_width;
        height = cinfo.output_height;

        assert(cinfo.output_components == 3);
        texture.reserve(width * height * 4);

        JSAMPLE *ba;
        while (cinfo.output_scanline < cinfo.output_height) {
            (void) jpeg_read_scanlines(&cinfo, buffer, 1);

            ba = buffer[0];

            for (int i = 0; i < row_stride; i += 3) {
                texture.push_back(ba[i]);
                texture.push_back(ba[i + 1]);
                texture.push_back(ba[i + 2]);
                texture.push_back(255.f); // alpha channel -- opaque
            }
        }

        (void) jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
    } catch (...) {
        jpeg_destroy_decompress(&cinfo);
        throw;
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

    jpeg_compress_struct cinfo{};
    jpeg_error_mgr jerr{};
    JSAMPROW row_pointer;
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = my_error_exit;

    std::unique_ptr<FILE, decltype(&fclose)> outfile(fopen(filename, "wb"), fclose);
    if (!outfile) {
        helios_runtime_error("ERROR (write_JPEG_file): Can't open file " + std::string(filename));
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile.get());

    cinfo.image_width = width; /* image width and height, in pixels */
    try {
        cinfo.image_height = height;
        cinfo.input_components = 3; /* # of color components per pixel */
        cinfo.in_color_space = JCS_RGB; /* colorspace of input image */

        jpeg_set_defaults(&cinfo);

        jpeg_set_quality(&cinfo, 100, TRUE /* limit to baseline-JPEG values */);

        jpeg_start_compress(&cinfo, TRUE);

        row_stride = width * 3; /* JSAMPLEs per row in image_buffer */

        while (cinfo.next_scanline < cinfo.image_height) {
            row_pointer = &screen_shot_trans[(cinfo.image_height - cinfo.next_scanline - 1) * row_stride];
            (void) jpeg_write_scanlines(&cinfo, &row_pointer, 1);
        }

        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);
    } catch (...) {
        jpeg_destroy_compress(&cinfo);
        throw;
    }

    return 1;
}

int write_JPEG_file(const char *filename, uint width, uint height, const std::vector<helios::RGBcolor> &data, bool print_messages) {
    assert(data.size() == width * height);

    if (print_messages) {
        std::cout << "writing JPEG image: " << filename << std::endl;
    }

    const uint bsize = 3 * width * height;
    std::vector<GLubyte> screen_shot_trans;
    screen_shot_trans.resize(bsize);

    size_t ii = 0;
    for (size_t i = 0; i < width * height; i++) {
        screen_shot_trans.at(ii) = (unsigned char) data.at(i).r * 255;
        screen_shot_trans.at(ii + 1) = (unsigned char) data.at(i).g * 255;
        screen_shot_trans.at(ii + 2) = (unsigned char) data.at(i).b * 255;
        ii += 3;
    }

    jpeg_compress_struct cinfo{};

    jpeg_error_mgr jerr{};
    JSAMPROW row_pointer;
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = my_error_exit;

    std::unique_ptr<FILE, decltype(&fclose)> outfile(fopen(filename, "wb"), fclose);
    if (!outfile) {
        helios_runtime_error("ERROR (write_JPEG_file): Can't open file " + std::string(filename));
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile.get());

    cinfo.image_width = width; /* image width and height, in pixels */
    try {
        cinfo.image_height = height;
        cinfo.input_components = 3; /* # of color components per pixel */
        cinfo.in_color_space = JCS_RGB; /* colorspace of input image */

        jpeg_set_defaults(&cinfo);

        jpeg_set_quality(&cinfo, 100, TRUE /* limit to baseline-JPEG values */);

        jpeg_start_compress(&cinfo, TRUE);

        row_stride = width * 3; /* JSAMPLEs per row in image_buffer */

        while (cinfo.next_scanline < cinfo.image_height) {
            row_pointer = &screen_shot_trans[(cinfo.image_height - cinfo.next_scanline - 1) * row_stride];
            (void) jpeg_write_scanlines(&cinfo, &row_pointer, 1);
        }

        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);
    } catch (...) {
        jpeg_destroy_compress(&cinfo);
        throw;
    }

    return 1;
}

void read_png_file(const char *filename, std::vector<unsigned char> &texture, uint &height, uint &width) {
    std::string fn = filename;
    if (fn.substr(fn.find_last_of('.') + 1) != "png" && fn.substr(fn.find_last_of('.') + 1) != "PNG") {
        helios_runtime_error("ERROR (read_PNG_file): File " + fn + " is not PNG format.");
    }

    int y;

    png_structp png_ptr;
    png_infop info_ptr;

    char header[8]; // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        helios_runtime_error("ERROR (read_png_file): File " + std::string(filename) + " could not be opened for reading");
    }
    fread(header, 1, 8, fp);

    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!png_ptr) {
        helios_runtime_error("ERROR (read_png_file): png_create_read_struct failed.");
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        helios_runtime_error("ERROR (read_png_file): png_create_info_struct failed.");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        helios_runtime_error("ERROR (read_png_file): init_io failed.");
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0 || png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0;

    if (bit_depth == 16) {
        png_set_strip_16(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
    }
    if (!has_alpha) {
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }

    png_read_update_info(png_ptr, info_ptr);

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr))) {
        helios_runtime_error("ERROR (read_png_file): read_image failed.");
    }

    auto *row_pointers = static_cast<png_bytep *>(malloc(sizeof(png_bytep) * height));
    for (y = 0; y < height; y++)
        row_pointers[y] = static_cast<png_byte *>(malloc(png_get_rowbytes(png_ptr, info_ptr)));

    png_read_image(png_ptr, row_pointers);

    fclose(fp);

    for (uint j = 0; j < height; j++) {
        png_byte *row = row_pointers[j];
        for (int i = 0; i < width; i++) {
            png_byte *ba = &(row[i * 4]);
            texture.push_back(ba[0]);
            texture.push_back(ba[1]);
            texture.push_back(ba[2]);
            texture.push_back(ba[3]);
        }
    }

    for (y = 0; y < height; y++)
        png_free(png_ptr, row_pointers[y]);
    png_free(png_ptr, row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
}

Visualizer::Visualizer(uint Wdisplay) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, uint(std::round(Wdisplay * 0.8)), 16, true);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, Hdisplay, 16, true);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, Hdisplay, aliasing_samples, true);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples, bool window_decorations) : colormap_current(), colormap_hot(), colormap_cool(), colormap_lava(), colormap_rainbow(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, Hdisplay, aliasing_samples, window_decorations);
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
        printf("WARNING (Visualizer): requested size of window is larger than the screen area.\n");
        // printf("Changing width from %d to %d and height from %d to %d\n",Wdisplay,window_width,Hdisplay,window_height);
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

void Visualizer::initialize(uint window_width_pixels, uint window_height_pixels, int aliasing_samples, bool window_decorations) {
    Wdisplay = window_width_pixels;
    Hdisplay = window_height_pixels;

    shadow_buffer_size = make_uint2(8192, 8192);

    maximum_texture_size = make_uint2(2048, 2048);

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

    // Initialize OpenGL context and open graphic window

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
    glfwWindowHint(GLFW_VISIBLE, 0);

    if (!window_decorations) {
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    }

    openWindow();

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        helios_runtime_error("ERROR (Visualizer::initialize): Failed to initialize GLEW");
    }

    // NOTE: for some reason calling glewInit throws an error.  Need to clear it to move on.
    glGetError();

    assert(checkerrors());

    // Enable relevant parameters

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

    assert(checkerrors());

    // glEnable(GL_TEXTURE1);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glDisable(GL_CULL_FACE);

    assert(checkerrors());

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

    assert(checkerrors());

    //~~~~~~~~~~~~~ Load the Shaders ~~~~~~~~~~~~~~~~~~~//

    primaryShader.initialize("plugins/visualizer/shaders/primaryShader.vert", "plugins/visualizer/shaders/primaryShader.frag", this);
    depthShader.initialize("plugins/visualizer/shaders/shadow.vert", "plugins/visualizer/shaders/shadow.frag", this);

    assert(checkerrors());

    primaryShader.useShader();

    // Initialize frame buffer

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

    assert(checkerrors());

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
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);


    // Initialize transformation matrices

    perspectiveTransformationMatrix = glm::mat4(1.f);

    customTransformationMatrix = glm::mat4(1.f);

    assert(checkerrors());

    // Default values

    light_direction = make_vec3(1, 1, 1);
    light_direction.normalize();
    primaryShader.setLightDirection(light_direction);

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

    glfwSetMouseButtonCallback((GLFWwindow *) window, mouseCallback);
    glfwSetCursorPosCallback((GLFWwindow *) window, cursorCallback);
    glfwSetScrollCallback((GLFWwindow *) window, scrollCallback);

    assert(checkerrors());
}

Visualizer::~Visualizer() {
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

    // for( auto iter=textureIDData.begin(); iter!=textureIDData.end(); ++iter ){
    //     std::vector<int> ID = textureIDData.at(iter->first);
    //     for(int i : ID){
    //         uint IDu = uint(i);
    //         glDeleteTextures(1, &IDu );
    //     }
    // }

    glfwDestroyWindow(scast<GLFWwindow *>(window));
    glfwTerminate();
}

int Visualizer::selfTest() const {
    if (message_flag) {
        std::cout << "Running visualizer self-test..." << std::flush;
    }

    Visualizer visualizer(1000);

    visualizer.setCameraPosition(make_SphericalCoord(10, 0.49 * PI_F, 0), make_vec3(0, 0, 0));

    visualizer.setLightingModel(Visualizer::LIGHTING_NONE);

    //---- rectangles ----//

    visualizer.addRectangleByCenter(make_vec3(-1.5, 0, 0), make_vec2(1, 2), make_SphericalCoord(0.f, 0.f), make_RGBAcolor(RGB::yellow, 0.5), Visualizer::COORDINATES_CARTESIAN);
    visualizer.addRectangleByCenter(make_vec3(-0.5, -0.5, 0), make_vec2(1, 1), make_SphericalCoord(0.f, 0.f), RGB::blue, Visualizer::COORDINATES_CARTESIAN);
    visualizer.addRectangleByCenter(make_vec3(-0.5, 0.5, 0), make_vec2(1, 1), make_SphericalCoord(0.f, 0.f), RGB::red, Visualizer::COORDINATES_CARTESIAN);
    visualizer.addRectangleByCenter(make_vec3(1.5, 0.5, 0), make_vec2(3.41, 1), make_SphericalCoord(0, 0), "plugins/visualizer/textures/Helios_logo.png", Visualizer::COORDINATES_CARTESIAN);
    visualizer.addRectangleByCenter(make_vec3(1.5, -0.5, 0), make_vec2(3.41, 1), make_SphericalCoord(0, 0), "plugins/visualizer/textures/Helios_logo.jpeg", Visualizer::COORDINATES_CARTESIAN);

    std::vector<vec3> vertices;
    vertices.resize(4);

    vertices.at(0) = make_vec3(-2, -1, 0);
    vertices.at(1) = make_vec3(-2, 1, 0);
    vertices.at(2) = make_vec3(-3, 0.5, 0);
    vertices.at(3) = make_vec3(-3, -0.5, 0);
    visualizer.addRectangleByVertices(vertices, RGB::green, Visualizer::COORDINATES_CARTESIAN);

    vertices.at(0) = make_vec3(-3, -0.5, 0);
    vertices.at(1) = make_vec3(-3, 0.5, 0);
    vertices.at(2) = make_vec3(-4, 1, 0);
    vertices.at(3) = make_vec3(-4, -1, 0);
    visualizer.addRectangleByVertices(vertices, make_RGBAcolor(RGB::violet, 0.5), Visualizer::COORDINATES_CARTESIAN);

    vertices.at(0) = make_vec3(-4, -1, 0);
    vertices.at(1) = make_vec3(-4, 1, 0);
    vertices.at(2) = make_vec3(-5, 0.5, 0);
    vertices.at(3) = make_vec3(-5, -0.5, 0);
    visualizer.addRectangleByVertices(vertices, "plugins/visualizer/textures/Helios_logo.png", Visualizer::COORDINATES_CARTESIAN);

    //---- triangles ----//

    vec3 v0, v1, v2;

    v0 = make_vec3(-1, -3, 0);
    v1 = make_vec3(1, -3, 0);
    v2 = make_vec3(1, -4, 0);
    visualizer.addTriangle(v0, v1, v2, make_RGBAcolor(RGB::red, 0.5), Visualizer::COORDINATES_CARTESIAN);

    v0 = make_vec3(-1, -3, 0);
    v1 = make_vec3(-1, -4, 0);
    v2 = make_vec3(1, -4, 0);
    visualizer.addTriangle(v0, v1, v2, RGB::blue, Visualizer::COORDINATES_CARTESIAN);

    //---- lines ----//

    visualizer.addLine(make_vec3(-1, 3, 0), make_vec3(0, 4, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN);
    visualizer.addLine(make_vec3(0, 4, 0), make_vec3(1, 3, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN);
    visualizer.addLine(make_vec3(1, 3, 0), make_vec3(0, 2, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN);
    visualizer.addLine(make_vec3(0, 2, 0), make_vec3(-1, 3, 0), RGB::red, Visualizer::COORDINATES_CARTESIAN);

    if (message_flag) {
        std::cout << "done." << std::endl;
    }

    return 0;
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

void Visualizer::clearContextGeometry() {
    geometry_handler.clearContextGeometry();

    contextUUIDs_build.clear();
    colorPrimitives_UUIDs.clear();
    colorPrimitives_objIDs.clear();
    depth_buffer_data.clear();
    colorbar_min = 0;
    colorbar_max = 0;
}

void Visualizer::closeWindow() const {
    glfwHideWindow((GLFWwindow *) window);
    glfwPollEvents();
}

void Visualizer::hideWatermark() {
    isWatermarkVisible = false;
    if (watermark_ID != 0) {
        geometry_handler.setVisibility(watermark_ID, false);
        uploadPrimitiveVisibility(watermark_ID);
    }
}

void Visualizer::showWatermark() {
    bool need_transfer = (watermark_ID == 0);
    isWatermarkVisible = true;
    updateWatermark();
    if (need_transfer && watermark_ID != 0) {
        transferBufferData();
    }
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
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
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
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
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
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
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
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
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
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
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
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.y < 0.f || vertex.y > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (vertex.z < -1.f || vertex.z > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
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
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << tri_vertex.x << " ) is outside of drawable area." << std::endl;
                }
            } else if (tri_vertex.y < 0.f || tri_vertex.y > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << tri_vertex.y << " ) is outside of drawable area." << std::endl;
                }
            } else if (tri_vertex.z < -1.f || tri_vertex.z > 1.f) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << tri_vertex.z << " ) is outside of drawable area." << std::endl;
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

size_t Visualizer::addPoint(const vec3 &position, const RGBcolor &color, float pointsize, CoordinateSystem coordinate_system) {
    return addPoint(position, make_RGBAcolor(color, 1), pointsize, coordinate_system);
}

size_t Visualizer::addPoint(const vec3 &position, const RGBAcolor &color, float pointsize, CoordinateSystem coordinate_system) {
    GLfloat range[2] = {0, 0};
    glGetFloatv(GL_POINT_SIZE_RANGE, range);
    if (pointsize < range[0] || pointsize > range[1]) {
        std::cerr << "WARNING (Visualizer::addPoint): Point size ( " << pointsize << " ) is outside of supported range ( " << range[0] << ", " << range[1] << " ). Clamping value.." << std::endl;
        if (pointsize < range[0]) {
            pointsize = range[0];
        } else {
            pointsize = range[1];
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
    addSkyDomeByCenter(radius, center, Ndivisions, texture_file);
}

std::vector<size_t> Visualizer::addSkyDomeByCenter(float radius, const vec3 &center, uint Ndivisions, const char *texture_file) {
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
    font = "plugins/visualizer/fonts/" + (std::string) fontname + ".ttf";
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
                    std::cout << "WARNING (Visualizer::addTextboxByCenter): text x-coordinate is outside of window area" << std::endl;
                }
            }
            if (yt < 0 || yt > 1) {
                if (message_flag) {
                    std::cout << "WARNING (Visualizer::addTextboxByCenter): text y-coordinate is outside of window area" << std::endl;
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

std::vector<size_t> Visualizer::addColorbarByCenter(const char *title, const helios::vec2 &size, const helios::vec3 &center, const helios::RGBcolor &font_color, const Colormap &colormap) {
    uint Ndivs = 50;

    uint Nticks = 4;

    std::vector<size_t> UUIDs;
    UUIDs.reserve(Ndivs + 2 * Nticks + 20);

    if (!colorbar_ticks.empty()) {
        Nticks = colorbar_ticks.size();
    }

    float dx = size.x / float(Ndivs);

    float cmin = clamp(colormap.getLowerLimit(), -1e7f, 1e7f);
    float cmax = clamp(colormap.getUpperLimit(), -1e7f, 1e7f);

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

    dx = size.x / float(Nticks - 1);

    std::vector<vec3> ticks;
    ticks.resize(2);
    for (uint i = 0; i < Nticks; i++) {
        /** \todo Need to use the more sophisticated formatting of tick strings */
        char textstr[10], precision[10];

        float x;
        float value;
        if (colorbar_ticks.empty()) {
            x = center.x - 0.5f * size.x + float(i) * dx;
            value = cmin + float(i) / float(Nticks - 1) * (cmax - cmin);
        } else {
            value = colorbar_ticks.at(i);
            x = center.x - 0.5f * size.x + (value - cmin) / (cmax - cmin) * size.x;
        }

        if (std::fabs(floor(value) - value) < 1e-4) { // value is an integer
            std::snprintf(precision, 10, "%%d");
            std::snprintf(textstr, 10, precision, int(floor(value)));
        } else if (value != 0.f) {
            // value needs decimal formatting
            int d1 = floor(log10(std::fabs(value)));
            int d2 = -d1 + 1;
            if (d2 < 1) {
                d2 = 1;
            }
            std::snprintf(precision, 10, "%%%u.%uf", (char) abs(d1) + 1, (char) d2);
            std::snprintf(textstr, 10, precision, value);
        }

        // tick labels
        std::vector<size_t> UUIDs_text = addTextboxByCenter(textstr, make_vec3(x, center.y - 0.4f * size.y, center.z), make_SphericalCoord(0, 0), font_color, colorbar_fontsize, "OpenSans-Regular", COORDINATES_WINDOW_NORMALIZED);
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
        std::cout << "WARNING (Visualizer::setColorbarRange): Maximum colorbar value must be greater than minimum value...Ignoring command." << std::endl;
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

void Visualizer::buildContextGeometry(helios::Context *context_ptr) {
    context = context_ptr;

    build_all_context_geometry = true;
}

void Visualizer::buildContextGeometry(helios::Context *context_ptr, const std::vector<uint> &UUIDs) {
    if (UUIDs.empty()) {
        std::cerr << "WARNING (Visualizer::buildContextGeometry): There is no Context geometry to build...exiting." << std::endl;
        return;
    }

    context = context_ptr;

    build_all_context_geometry = false;
    contextUUIDs_build = UUIDs;
}

void Visualizer::buildContextGeometry_private() {

    // If building all context geometry, get all dirty UUIDs from the Context
    if (build_all_context_geometry) {
        bool include_deleted_UUIDs = true;
        if (contextUUIDs_build.empty()) {
            include_deleted_UUIDs = false;
        }
        if ( primitiveColorsNeedUpdate ) {
            //\todo This is a temporary fix to ensure that the colors are update if the visualization mode changes. This is inefficient because it would be better to just update the colors since the geometry has not changed.
            contextUUIDs_build = context->getAllUUIDs();
        }else {
            contextUUIDs_build = context->getDirtyUUIDs(include_deleted_UUIDs);
        }
    }

    // Populate contextUUIDs_needupdate based on dirty primitives in the Context
    std::vector<uint> contextUUIDs_needupdate;
    contextUUIDs_needupdate.reserve(contextUUIDs_build.size());

    for (uint UUID: contextUUIDs_build) {

        // Check if primitives in contextUUIDs_build have since been deleted from the Context. If so, remove them from contextUUIDs_build and from the geometry handler
        if (!context->doesPrimitiveExist(UUID)) {
            auto it = std::find(contextUUIDs_build.begin(), contextUUIDs_build.end(), UUID);
            if (it != contextUUIDs_build.end()) {
                // swap-and-pop delete from contextUUIDs_build
                *it = contextUUIDs_build.back();
                contextUUIDs_build.pop_back();
                // delete from the geometry handler
                if (geometry_handler.doesGeometryExist(UUID)) {
                    geometry_handler.deleteGeometry(UUID);
                }
            }
        }
        // check if the primitive is dirty, if so, add it to contextUUIDs_needupdate
        else {
            contextUUIDs_needupdate.push_back(UUID);
        }
    }

    if (contextUUIDs_needupdate.empty()) {
        return;
    }

    if (!colorPrimitivesByData.empty()) {
        if (colorPrimitives_UUIDs.empty()) { // load all primitives
            for (uint UUID: contextUUIDs_build) {
                if (context->doesPrimitiveExist(UUID)) {
                    colorPrimitives_UUIDs[UUID] = UUID;
                }
            }
        } else { // double check that primitives exist
            for (uint UUID: contextUUIDs_build) {
                if (!context->doesPrimitiveExist(UUID)) {
                    auto it = colorPrimitives_UUIDs.find(UUID);
                    colorPrimitives_UUIDs.erase(it);
                }
            }
        }
    } else if (!colorPrimitivesByObjectData.empty()) {
        if (colorPrimitives_objIDs.empty()) { // load all primitives
            std::vector<uint> ObjIDs = context->getAllObjectIDs();
            for (uint objID: ObjIDs) {
                if (context->doesObjectExist(objID)) {
                    std::vector<uint> UUIDs = context->getObjectPointer(objID)->getPrimitiveUUIDs();
                    for (uint UUID: UUIDs) {
                        if (context->doesPrimitiveExist(UUID)) {
                            colorPrimitives_UUIDs[UUID] = UUID;
                        }
                    }
                }
            }
        } else { // load primitives specified by user
            for (const auto &objID: colorPrimitives_objIDs) {
                if (context->doesObjectExist(objID.first)) {
                    std::vector<uint> UUIDs = context->getObjectPointer(objID.first)->getPrimitiveUUIDs();
                    for (uint UUID: UUIDs) {
                        if (context->doesPrimitiveExist(UUID)) {
                            colorPrimitives_UUIDs[UUID] = UUID;
                        }
                    }
                }
            }
        }
    }

    if (!colorPrimitives_UUIDs.empty() && colorbar_flag == 0) {
        enableColorbar();
    }

    //------ Colormap ------//

    uint psize = contextUUIDs_needupdate.size();
    if (message_flag) {
        if (psize > 0) {
            if (psize >= 1e3 && psize < 1e6) {
                std::cout << "updating " << psize / 1e3 << "K Context primitives to visualizer...." << std::flush;
            } else if (psize >= 1e6) {
                std::cout << "updating " << psize / 1e6 << "M Context primitives to visualizer...." << std::flush;
            } else {
                std::cout << "updating " << psize << " Context primitives to visualizer...." << std::flush;
            }
        } else {
            std::cout << "WARNING (Visualizer::buildContextGeometry): No primitives were found in the Context..." << std::endl;
        }
    }

    // figure out colorbar range
    //  \todo Figure out how to avoid doing this when not necessary

    colormap_current.setRange(colorbar_min, colorbar_max);
    if ((!colorPrimitivesByData.empty() || !colorPrimitivesByObjectData.empty()) && colorbar_min == 0 && colorbar_max == 0) { // range was not set by user, use full range of values

        colorbar_min = (std::numeric_limits<float>::max)();
        colorbar_max = (std::numeric_limits<float>::lowest)();

        for (uint UUID: contextUUIDs_build) {
            float colorValue = -9999;
            if (!colorPrimitivesByData.empty()) {
                if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                    if (context->doesPrimitiveDataExist(UUID, colorPrimitivesByData.c_str())) {
                        HeliosDataType type = context->getPrimitiveDataType(UUID, colorPrimitivesByData.c_str());
                        if (type == HELIOS_TYPE_FLOAT) {
                            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), colorValue);
                        } else if (type == HELIOS_TYPE_INT) {
                            int cv;
                            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                            colorValue = float(cv);
                        } else if (type == HELIOS_TYPE_UINT) {
                            uint cv;
                            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                            colorValue = float(cv);
                        } else if (type == HELIOS_TYPE_DOUBLE) {
                            double cv;
                            context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                            colorValue = float(cv);
                        } else {
                            colorValue = 0;
                        }
                    } else {
                        colorValue = 0;
                    }
                }
            } else if (!colorPrimitivesByObjectData.empty()) {
                if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                    uint ObjID = context->getPrimitiveParentObjectID(UUID);
                    if (ObjID != 0 && context->doesObjectDataExist(ObjID, colorPrimitivesByObjectData.c_str())) {
                        HeliosDataType type = context->getObjectDataType(ObjID, colorPrimitivesByObjectData.c_str());
                        if (type == HELIOS_TYPE_FLOAT) {
                            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), colorValue);
                        } else if (type == HELIOS_TYPE_INT) {
                            int cv;
                            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                            colorValue = float(cv);
                        } else if (type == HELIOS_TYPE_UINT) {
                            uint cv;
                            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                            colorValue = float(cv);
                        } else if (type == HELIOS_TYPE_DOUBLE) {
                            double cv;
                            context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                            colorValue = float(cv);
                        } else {
                            colorValue = 0;
                        }
                    } else {
                        colorValue = 0;
                    }
                }
            }

            if (std::isnan(colorValue) || std::isinf(colorValue)) { // check for NaN or infinity
                colorValue = 0;
            }

            if (colorValue != -9999) {
                if (colorValue < colorbar_min) {
                    colorbar_min = colorValue;
                    ;
                }
                if (colorValue > colorbar_max) {
                    colorbar_max = colorValue;
                    ;
                }
            }
        }

        if (!std::isinf(colorbar_min) && !std::isinf(colorbar_max)) {
            colormap_current.setRange(colorbar_min, colorbar_max);
        }
    }

    if (!colorPrimitivesByData.empty()) {
        assert(colorbar_min <= colorbar_max);
    }

    //------- Simulation Geometry -------//

    // add primitives

    size_t patch_count = context->getPatchCount();
    geometry_handler.allocateBufferSize(patch_count, GeometryHandler::GEOMETRY_TYPE_RECTANGLE);
    size_t triangle_count = context->getTriangleCount();
    geometry_handler.allocateBufferSize(triangle_count, GeometryHandler::GEOMETRY_TYPE_TRIANGLE);

    for (unsigned int UUID: contextUUIDs_needupdate) {

        if (!context->doesPrimitiveExist(UUID)) {
            std::cerr << "WARNING (Visualizer::buildContextGeometry): UUID vector contains ID(s) that do not exist in the Context...they will be ignored." << std::endl;
            continue;
        }

        helios::PrimitiveType ptype = context->getPrimitiveType(UUID);

        const std::vector<vec3> verts = context->getPrimitiveVertices(UUID);
        const std::string texture_file = context->getPrimitiveTextureFile(UUID);

        RGBAcolor color;
        float colorValue;
        if (!colorPrimitivesByData.empty()) {
            if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                if (context->doesPrimitiveDataExist(UUID, colorPrimitivesByData.c_str())) {
                    HeliosDataType type = context->getPrimitiveDataType(UUID, colorPrimitivesByData.c_str());
                    if (type == HELIOS_TYPE_FLOAT) {
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), colorValue);
                    } else if (type == HELIOS_TYPE_INT) {
                        int cv;
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_UINT) {
                        uint cv;
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_DOUBLE) {
                        double cv;
                        context->getPrimitiveData(UUID, colorPrimitivesByData.c_str(), cv);
                        colorValue = float(cv);
                    } else {
                        colorValue = 0;
                    }
                } else {
                    colorValue = 0;
                }

                if (std::isnan(colorValue) || std::isinf(colorValue)) { // check for NaN or infinity
                    colorValue = 0;
                }

                color = make_RGBAcolor(colormap_current.query(colorValue), 1);
            } else {
                color = context->getPrimitiveColorRGBA(UUID);
            }
        } else if (!colorPrimitivesByObjectData.empty()) {
            if (colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()) {
                uint ObjID = context->getPrimitiveParentObjectID(UUID);
                if (ObjID != 0 && context->doesObjectDataExist(ObjID, colorPrimitivesByObjectData.c_str())) {
                    HeliosDataType type = context->getObjectDataType(ObjID, colorPrimitivesByObjectData.c_str());
                    if (type == HELIOS_TYPE_FLOAT) {
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), colorValue);
                    } else if (type == HELIOS_TYPE_INT) {
                        int cv;
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_UINT) {
                        uint cv;
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                        colorValue = float(cv);
                    } else if (type == HELIOS_TYPE_DOUBLE) {
                        double cv;
                        context->getObjectData(ObjID, colorPrimitivesByObjectData.c_str(), cv);
                        colorValue = float(cv);
                    } else {
                        colorValue = 0;
                    }
                } else {
                    colorValue = 0;
                }

                if (std::isnan(colorValue) || std::isinf(colorValue)) { // check for NaN or infinity
                    colorValue = 0;
                }

                color = make_RGBAcolor(colormap_current.query(colorValue), 1);
            } else {
                color = context->getPrimitiveColorRGBA(UUID);
            }
        } else {
            color = context->getPrimitiveColorRGBA(UUID);
        }

        int textureID = -1;
        if (!texture_file.empty()) {
            textureID = registerTextureImage(texture_file);
        }

        // ---- PATCHES ---- //
        if (ptype == helios::PRIMITIVE_TYPE_PATCH) {
            // - Patch does not have an associated texture or we are ignoring texture
            if (texture_file.empty()) {
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, verts, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
            }
            // - Patch has a texture
            else {
                std::vector<vec2> uvs = context->getPrimitiveTextureUV(UUID);

                // - coloring primitive based on texture
                if ((colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.empty()) && !context->isPrimitiveTextureColorOverridden(UUID)) {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, verts, color, uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                }
                // - coloring primitive based on primitive data
                else {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, verts, color, uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                }
            }
        }
        // ---- TRIANGLES ---- //
        else if (ptype == helios::PRIMITIVE_TYPE_TRIANGLE) {
            // - Triangle does not have an associated texture or we are ignoring texture
            if (texture_file.empty()) {
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, verts, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
            }
            // - Triangle has a texture
            else {
                std::vector<vec2> uvs = context->getPrimitiveTextureUV(UUID);

                // - coloring primitive based on texture
                if ((colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.empty()) && !context->isPrimitiveTextureColorOverridden(UUID)) {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, verts, color, uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                }
                // - coloring primitive based on RGB color but mask using texture
                else {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_TRIANGLE, verts, color, uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                }
            }
        }
        // ---- VOXELS ---- //
        else if (ptype == helios::PRIMITIVE_TYPE_VOXEL) {
            std::vector<vec3> v_vertices = context->getPrimitiveVertices(UUID);

            // bottom
            const std::vector<vec3> bottom_vertices{v_vertices.at(0), v_vertices.at(1), v_vertices.at(2), v_vertices.at(3)};

            // top
            const std::vector<vec3> top_vertices{v_vertices.at(4), v_vertices.at(5), v_vertices.at(6), v_vertices.at(7)};

            //-x
            const std::vector<vec3> mx_vertices{v_vertices.at(0), v_vertices.at(3), v_vertices.at(7), v_vertices.at(4)};

            //+x
            const std::vector<vec3> px_vertices{v_vertices.at(1), v_vertices.at(2), v_vertices.at(6), v_vertices.at(5)};

            //-y
            const std::vector<vec3> my_vertices{v_vertices.at(0), v_vertices.at(1), v_vertices.at(5), v_vertices.at(4)};

            //+y
            const std::vector<vec3> py_vertices{v_vertices.at(2), v_vertices.at(3), v_vertices.at(7), v_vertices.at(6)};

            // Voxel does not have an associated texture or we are ignoring texture
            if (texture_file.empty()) {
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, bottom_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, top_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, mx_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, px_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, my_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
                geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, py_vertices, color, {}, -1, false, false, COORDINATES_CARTESIAN, true, true);
            }
            // Voxel has a texture
            else {
                const std::vector<helios::vec2> voxel_uvs = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};

                // coloring primitive based on texture
                if ((colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.empty()) && context->isPrimitiveTextureColorOverridden(UUID)) {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, bottom_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, top_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, mx_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, px_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, my_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, py_vertices, color, voxel_uvs, textureID, false, false, COORDINATES_CARTESIAN, true, true);
                }
                // coloring primitive based on primitive data
                else {
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, bottom_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, top_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, mx_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, px_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, my_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                    geometry_handler.addGeometry(UUID, GeometryHandler::GEOMETRY_TYPE_RECTANGLE, py_vertices, color, voxel_uvs, textureID, true, false, COORDINATES_CARTESIAN, true, true);
                }
            }
        }
    }
}

void Visualizer::colorContextPrimitivesByData(const char *data_name) {
    colorPrimitivesByData = data_name;
    colorPrimitivesByObjectData = "";
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
}

void Visualizer::colorContextPrimitivesByData(const char *data_name, const std::vector<uint> &UUIDs) {
    colorPrimitivesByData = data_name;
    colorPrimitivesByObjectData = "";
    for (uint UUID: UUIDs) {
        colorPrimitives_UUIDs[UUID] = UUID;
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
}

void Visualizer::colorContextPrimitivesByObjectData(const char *data_name) {
    colorPrimitivesByObjectData = data_name;
    colorPrimitivesByData = "";
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
}

void Visualizer::colorContextPrimitivesByObjectData(const char *data_name, const std::vector<uint> &ObjIDs) {
    colorPrimitivesByObjectData = data_name;
    colorPrimitivesByData = "";
    for (uint objID: ObjIDs) {
        colorPrimitives_objIDs[objID] = objID;
    }
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    primitiveColorsNeedUpdate = true;
}

void Visualizer::colorContextPrimitivesRandomly(const std::vector<uint> &UUIDs) {
    disableColorbar();
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    for (uint UUID: UUIDs) {
        float rc = randu();
        context->setPrimitiveData(UUID, "random_color", rc);
    }

    colorPrimitivesByData = "random_color";
    colorPrimitivesByObjectData = "";
    for (uint UUID: UUIDs) {
        colorPrimitives_UUIDs[UUID] = UUID;
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
}

void Visualizer::colorContextPrimitivesRandomly() {
    disableColorbar();

    std::vector<uint> all_UUIDs = context->getAllUUIDs();
    for (uint UUID: all_UUIDs) {
        float rc = randu();
        context->setPrimitiveData(UUID, "random_color", rc);
    }

    colorPrimitivesByData = "random_color";
    colorPrimitivesByObjectData = "";
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
}


void Visualizer::colorContextObjectsRandomly(const std::vector<uint> &ObjIDs) {
    disableColorbar();
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    for (uint ObjID: ObjIDs) {
        float rc = randu();
        context->setObjectData(ObjID, "random_color", rc);
    }

    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "random_color";
}

void Visualizer::colorContextObjectsRandomly() {
    std::vector<uint> all_ObjIDs = context->getAllObjectIDs();
    disableColorbar();
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    for (uint ObjID: all_ObjIDs) {
        float rc = randu();
        context->setObjectData(ObjID, "random_color", rc);
    }

    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "random_color";
}


float dphi = 0.0;
float dtheta = 0.0;
float dx = 0.0;
float dy = 0.0;
float dz = 0.0;
float dx_m = 0.0;
float dy_m = 0.0;
float dscroll = 0.0;

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

        glfwWaitEvents();

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
    //\todo Re-work this so that it only transfers data that has changed.

    assert(checkerrors());

    int i = 0;
    for (const auto &geometry_type: GeometryHandler::all_geometry_types) {
        // 1st attribute buffer : vertex positions
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getVertexData_ptr(geometry_type)->size() * sizeof(GLfloat), geometry_handler.getVertexData_ptr(geometry_type)->data(), GL_STATIC_DRAW);

        // 2nd attribute buffer : vertex uv
        glBindBuffer(GL_ARRAY_BUFFER, uv_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getUVData_ptr(geometry_type)->size() * sizeof(GLfloat), geometry_handler.getUVData_ptr(geometry_type)->data(), GL_STATIC_DRAW);

        // 3rd attribute buffer : face index
        glBindBuffer(GL_ARRAY_BUFFER, face_index_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getFaceIndexData_ptr(geometry_type)->size() * sizeof(GLint), geometry_handler.getFaceIndexData_ptr(geometry_type)->data(), GL_STATIC_DRAW);

        // 1st texture buffer : vertex colors
        glBindBuffer(GL_TEXTURE_BUFFER, color_buffer.at(i));
        glBufferData(GL_TEXTURE_BUFFER, geometry_handler.getColorData_ptr(geometry_type)->size() * sizeof(GLfloat), geometry_handler.getColorData_ptr(geometry_type)->data(), GL_STATIC_DRAW);
        glBindTexture(GL_TEXTURE_BUFFER, color_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, color_buffer.at(i));

        // 2nd texture buffer : face normals
        glBindBuffer(GL_ARRAY_BUFFER, normal_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getNormalData_ptr(geometry_type)->size() * sizeof(GLfloat), geometry_handler.getNormalData_ptr(geometry_type)->data(), GL_STATIC_DRAW);
        glBindTexture(GL_TEXTURE_BUFFER, normal_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, normal_buffer.at(i));

        // 3rd texture buffer : face texture flag
        glBindBuffer(GL_ARRAY_BUFFER, texture_flag_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getTextureFlagData_ptr(geometry_type)->size() * sizeof(GLint), geometry_handler.getTextureFlagData_ptr(geometry_type)->data(), GL_STATIC_DRAW);
        glBindTexture(GL_TEXTURE_BUFFER, texture_flag_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, texture_flag_buffer.at(i));

        // 4th texture buffer : image texture ID
        glBindBuffer(GL_ARRAY_BUFFER, texture_ID_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getTextureIDData_ptr(geometry_type)->size() * sizeof(GLint), geometry_handler.getTextureIDData_ptr(geometry_type)->data(), GL_STATIC_DRAW);
        glBindTexture(GL_TEXTURE_BUFFER, texture_ID_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, texture_ID_buffer.at(i));

        // 5th attribute buffer : face coordinate flag
        glBindBuffer(GL_ARRAY_BUFFER, coordinate_flag_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getCoordinateFlagData_ptr(geometry_type)->size() * sizeof(GLint), geometry_handler.getCoordinateFlagData_ptr(geometry_type)->data(), GL_STATIC_DRAW);
        glBindTexture(GL_TEXTURE_BUFFER, coordinate_flag_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, coordinate_flag_buffer.at(i));

        // 6th attribute buffer : hidden flag
        glBindBuffer(GL_ARRAY_BUFFER, hidden_flag_buffer.at(i));
        glBufferData(GL_ARRAY_BUFFER, geometry_handler.getVisibilityFlagData_ptr(geometry_type)->size() * sizeof(GLbyte), geometry_handler.getVisibilityFlagData_ptr(geometry_type)->data(), GL_STATIC_DRAW);
        glBindTexture(GL_TEXTURE_BUFFER, hidden_flag_texture_object.at(i));
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R8I, hidden_flag_buffer.at(i));

        i++;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // Pre-compute indexing to draw rectangles as TRIANGLE_FAN
    size_t rectangle_count = geometry_handler.getRectangleCount();

    rectangle_vertex_group_firsts.resize(rectangle_count);
    rectangle_vertex_group_counts.resize(rectangle_count, 4);
    for (int i = 0; i < rectangle_count; ++i) {
        rectangle_vertex_group_firsts[i] = i * 4; // quad 0 starts at v[0], quad 1 at v[4], …
    }

    // Set up textures

    // if ( !texture_manager.empty() ) {
    glGenTextures(1, &texArray);
    glBindTexture(GL_TEXTURE_2D_ARRAY, texArray);

    // Allocate L layers of size W×H, 1 mip level
    glTexStorage3D(GL_TEXTURE_2D_ARRAY,
                   /*levels=*/1,
                   GL_RGBA8, // 8 bit RGBA per texel
                   /*width=*/maximum_texture_size.x,
                   /*height=*/maximum_texture_size.y,
                   /*layers=*/std::max(1, (int) texture_manager.size()));

    // Set filtering & wrap
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    const size_t Ntextures = texture_manager.size();
    std::vector<GLfloat> uv_rescale;
    uv_rescale.resize(Ntextures * 2);

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
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                        /*level=*/0,
                        /*xoffset=*/0, /*yoffset=*/0, /*zoffset=*/textureID,
                        /*width=*/texture.texture_resolution.x, /*height=*/texture.texture_resolution.y, /*depth=*/1, externalFormat, GL_UNSIGNED_BYTE,
                        texture.texture_data.data() // pointer to pixel data
        );

        uv_rescale.at(textureID * 2 + 0) = float(texture.texture_resolution.x) / float(maximum_texture_size.x);
        uv_rescale.at(textureID * 2 + 1) = float(texture.texture_resolution.y) / float(maximum_texture_size.y);
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    glUniform1i(glGetUniformLocation(primaryShader.shaderID, "textureSampler"), 0);

    // Set up (u,v) rescaling
    glBindBuffer(GL_TEXTURE_BUFFER, uv_rescale_buffer);
    glBufferData(GL_TEXTURE_BUFFER, uv_rescale.size() * sizeof(GLfloat), uv_rescale.data(), GL_STATIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, uv_rescale_texture_object);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, uv_rescale_buffer);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    assert(checkerrors());
}

void Visualizer::uploadPrimitiveVertices(size_t UUID) {
    const auto &idx = geometry_handler.getIndexMap(UUID);
    size_t type_ind = std::find(GeometryHandler::all_geometry_types.begin(),
                                GeometryHandler::all_geometry_types.end(),
                                idx.geometry_type) -
                       GeometryHandler::all_geometry_types.begin();

    char vcount = GeometryHandler::getVertexCount(idx.geometry_type);

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer.at(type_ind));
    glBufferSubData(GL_ARRAY_BUFFER,
                    static_cast<GLintptr>(idx.vertex_index * sizeof(GLfloat)),
                    static_cast<GLsizeiptr>(vcount * 3 * sizeof(GLfloat)),
                    geometry_handler.getVertexData_ptr(idx.geometry_type)->data() +
                        idx.vertex_index);

    glBindBuffer(GL_ARRAY_BUFFER, normal_buffer.at(type_ind));
    glBufferSubData(GL_ARRAY_BUFFER,
                    static_cast<GLintptr>(idx.normal_index * sizeof(GLfloat)),
                    static_cast<GLsizeiptr>(3 * sizeof(GLfloat)),
                    geometry_handler.getNormalData_ptr(idx.geometry_type)->data() +
                        idx.normal_index);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Visualizer::uploadPrimitiveVisibility(size_t UUID) {
    const auto &idx = geometry_handler.getIndexMap(UUID);
    size_t type_ind = std::find(GeometryHandler::all_geometry_types.begin(),
                                GeometryHandler::all_geometry_types.end(),
                                idx.geometry_type) -
                       GeometryHandler::all_geometry_types.begin();

    glBindBuffer(GL_ARRAY_BUFFER, hidden_flag_buffer.at(type_ind));
    glBufferSubData(GL_ARRAY_BUFFER,
                    static_cast<GLintptr>(idx.visible_index * sizeof(GLbyte)),
                    static_cast<GLsizeiptr>(sizeof(GLbyte)),
                    geometry_handler.getVisibilityFlagData_ptr(idx.geometry_type)->data() +
                        idx.visible_index);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
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

        glMultiDrawArrays(GL_TRIANGLE_FAN, rectangle_vertex_group_firsts.data(), rectangle_vertex_group_counts.data(), rectangle_count);
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
            glLineWidth(1);
            glDrawArrays(GL_LINES, 0, line_count * 2);
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
            glPointSize(point_width);
            glDrawArrays(GL_POINTS, 0, point_count);
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
    if (message_flag) {
        std::cout << "Updating the plot..." << std::flush;
    }

    if (!hide_window) {
        glfwShowWindow(scast<GLFWwindow *>(window));
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

    return textureID;
}

uint Visualizer::registerTextureImage(const std::vector<unsigned char> &texture_data, const helios::uint2 &image_resolution) {
#ifdef HELIOS_DEBUG
    assert(!texture_data.empty() && texture_data.size() == 4 * image_resolution.x * image_resolution.y);
#endif

    const uint textureID = texture_manager.size();

    texture_manager.try_emplace(textureID, texture_data, textureID, image_resolution, this->maximum_texture_size);

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

    return textureID;
}

uint Visualizer::registerTextureGlyph(const Glyph *glyph) {

    const uint textureID = texture_manager.size();

    texture_manager.try_emplace(textureID, glyph, textureID, this->maximum_texture_size);

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
        std::cout << "Does not exist" << std::endl;
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

    // initialize default texture in case none are added to the scene
    //  glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    //  glTexImage2D(GL_TEXTURE_2D_ARRAY, 0,GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    assert(checkerrors());
}

Shader::~Shader() {
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

void Visualizer::handleWindowResize(int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }

    Wdisplay = static_cast<uint>(width);
    Hdisplay = static_cast<uint>(height);

    int fbw = width;
    int fbh = height;
    glfwGetFramebufferSize(static_cast<GLFWwindow *>(window), &fbw, &fbh);
    Wframebuffer = static_cast<uint>(fbw);
    Hframebuffer = static_cast<uint>(fbh);

    updateWatermark();
}

void Visualizer::framebufferResizeCallback(GLFWwindow *glfw_window, int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }
    auto *viz = static_cast<Visualizer *>(glfwGetWindowUserPointer(glfw_window));
    if (viz != nullptr) {
        int w = 0;
        int h = 0;
        glfwGetWindowSize(glfw_window, &w, &h);
        viz->handleWindowResize(w, h);
    }
}

void Visualizer::windowResizeCallback(GLFWwindow *glfw_window, int width, int height) {
    auto *viz = static_cast<Visualizer *>(glfwGetWindowUserPointer(glfw_window));
    if (viz != nullptr) {
        viz->handleWindowResize(width, height);
    }
}

void Visualizer::updateWatermark() {
    constexpr float texture_aspect = 675.f / 195.f; // image width / height

    float window_aspect = float(Wframebuffer) / float(Hframebuffer);
    float width = 0.07f * texture_aspect / window_aspect;

    helios::vec3 center = make_vec3(0.75f * width, 0.95f, 0.f);
    helios::vec2 size = make_vec2(width, 0.07f);

    if (!isWatermarkVisible) {
        if (watermark_ID != 0) {
            geometry_handler.setVisibility(watermark_ID, false);
            uploadPrimitiveVisibility(watermark_ID);
        }
        return;
    }

    if (watermark_ID == 0) {
        watermark_ID = addRectangleByCenter(center, size, make_SphericalCoord(0, 0),
                                            "plugins/visualizer/textures/Helios_watermark.png",
                                            COORDINATES_WINDOW_NORMALIZED);
        transferBufferData();
        return;
    }

    std::vector<helios::vec3> verts{
        make_vec3(center.x - 0.5f * size.x, center.y - 0.5f * size.y, 0.f),
        make_vec3(center.x + 0.5f * size.x, center.y - 0.5f * size.y, 0.f),
        make_vec3(center.x + 0.5f * size.x, center.y + 0.5f * size.y, 0.f),
        make_vec3(center.x - 0.5f * size.x, center.y + 0.5f * size.y, 0.f)};

    geometry_handler.setVertices(watermark_ID, verts);
    geometry_handler.setVisibility(watermark_ID, true);
    uploadPrimitiveVertices(watermark_ID);
    uploadPrimitiveVisibility(watermark_ID);
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

void Visualizer::clearColor() {
    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "";
    if (!colorPrimitives_UUIDs.empty()) {
        colorPrimitives_UUIDs.clear();
    }
    if (!colorPrimitives_objIDs.empty()) {
        colorPrimitives_objIDs.clear();
    }
    disableColorbar();
    colorbar_min = 0;
    colorbar_max = 0;
    colorbar_flag = 0;
    primitiveColorsNeedUpdate = true;
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
