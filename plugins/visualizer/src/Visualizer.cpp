/** \file "Visualizer.cpp" Visualizer plugin declarations.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

//OpenGL Includes
#include <GL/glew.h>
#include <GLFW/glfw3.h>

//Freetype Libraries (rendering fonts)
#include <ft2build.h>
#include FT_FREETYPE_H

//JPEG Libraries (reading and writing JPEG images)
#include <cstdio> //<-- note libjpeg requires this header be included before its headers.
#include <jpeglib.h>
//#include <setjmp.h>

//PNG Libraries (reading and writing PNG images)
#ifndef _WIN32
    #include <unistd.h>
#endif
#define PNG_DEBUG 3
#define PNG_SKIP_SETJMP_CHECK 1
#include <png.h>

#include "Visualizer.h"

using namespace helios;

/** \todo This is crap associated with the JPEG reading library need to figure out if some of it can be removed. */
struct my_error_mgr {

    struct jpeg_error_mgr pub;	/* "public" fields */

    jmp_buf setjmp_buffer;	/* for return to caller */
};
typedef struct my_error_mgr * my_error_ptr;
METHODDEF(void) my_error_exit (j_common_ptr cinfo){
    /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
    auto myerr = (my_error_ptr) cinfo->err;

    /* Always display the message. */
    /* We could postpone this until after returning, if we chose. */
    (*cinfo->err->output_message) (cinfo);

    /* Return control to the setjmp point */
    longjmp(myerr->setjmp_buffer, 1);
}

int read_JPEG_file (const char * filename, std::vector<unsigned char> &texture, uint & height, uint & width){

    std::string fn = filename;
    if( fn.substr(fn.find_last_of('.') + 1) != "jpg" && fn.substr(fn.find_last_of('.') + 1) != "jpeg" && fn.substr(fn.find_last_of('.') + 1) != "JPG" && fn.substr(fn.find_last_of('.') + 1) != "JPEG" ) {
        helios_runtime_error("ERROR (read_JPEG_file): File " + fn + " is not JPEG format.");
    }

    struct jpeg_decompress_struct cinfo;

    struct my_error_mgr jerr;
    FILE * infile;		/* source file */
    JSAMPARRAY buffer;		/*output row buffer */
    uint row_stride;

    if ((infile = fopen(filename, "rb")) == nullptr ) {
        fprintf(stderr, "can't open %s\n", filename);
        return 0;
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return 0;
    }

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);

    (void) jpeg_read_header(&cinfo, TRUE);

    (void) jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)
            ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    width=cinfo.output_width;
    height=cinfo.output_height;

    assert(cinfo.output_components==3);

    JSAMPLE* ba;
    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);

        ba=buffer[0];

        for (int i=0; i < row_stride; i=i+3){
            texture.push_back(ba[i]);
            texture.push_back(ba[i+1]);
            texture.push_back(ba[i+2]);
            texture.push_back(255.f);//alpha channel -- opaque
        }

    }

    (void) jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);

    fclose(infile);

    return 0;
}

int write_JPEG_file ( const char* filename, uint width, uint height, void* _window ){

    std::cout << "writing JPEG image: " << filename << std::endl;

    const uint bsize = 3 * width * height;
    std::vector<GLubyte> screen_shot_trans;
    screen_shot_trans.resize(bsize);

    glfwSwapBuffers((GLFWwindow*)_window);
    glReadPixels(0, 0, GLsizei(width), GLsizei(height), GL_RGB, GL_UNSIGNED_BYTE, &screen_shot_trans[0]);

    //depending on the active frame buffer, we may get all zero data and need to swap it again.
    bool zeros = true;
    for( int i=0; i<bsize; i++ ){
        if( screen_shot_trans[i]!=0 ){
            zeros = false;
        }
    }
    if( zeros ){

        glfwSwapBuffers((GLFWwindow*)_window);

        glReadPixels(0, 0, GLsizei(width), GLsizei(height), GL_RGB, GL_UNSIGNED_BYTE, &screen_shot_trans[0]);

    }

    struct jpeg_compress_struct cinfo;

    struct jpeg_error_mgr jerr;
    /* More stuff */
    FILE * outfile;		/* target file */
    JSAMPROW row_pointer;	/* pointer to JSAMPLE row[s] */
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr);
    /* Now we can initialize the JPEG compression object. */
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename, "wb")) == nullptr ) {
        helios_runtime_error("ERROR (write_JPEG_file): Can't open file " + std::string(filename));
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width; 	/* image width and height, in pixels */
    cinfo.image_height = height;
    cinfo.input_components = 3;		/* # of color components per pixel */
    cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */

    jpeg_set_defaults(&cinfo);

    jpeg_set_quality(&cinfo, 100, TRUE /* limit to baseline-JPEG values */);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width * 3;	/* JSAMPLEs per row in image_buffer */

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW) &screen_shot_trans[ (cinfo.image_height-cinfo.next_scanline-1) * row_stride ];
        (void) jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    /* After finish_compress, we can close the output file. */
    fclose(outfile);

    jpeg_destroy_compress(&cinfo);

    return 1;

}

int write_JPEG_file ( const char* filename, const uint width, const uint height, const std::vector<helios::RGBcolor>& data ){

    assert( data.size()==width*height );

    std::cout << "writing JPEG image: " << filename << std::endl;

    const uint bsize = 3 * width * height;
    std::vector<GLubyte> screen_shot_trans;
    screen_shot_trans.resize(bsize);

    size_t ii = 0;
    for( size_t i=0; i<width*height; i++ ){
        screen_shot_trans.at(ii) = (unsigned char)data.at(i).r*255;
        screen_shot_trans.at(ii+1) = (unsigned char)data.at(i).g*255;
        screen_shot_trans.at(ii+2) = (unsigned char)data.at(i).b*255;
        ii+=3;
    }

    struct jpeg_compress_struct cinfo;

    struct jpeg_error_mgr jerr;
    /* More stuff */
    FILE * outfile;		/* target file */
    JSAMPROW row_pointer;	/* pointer to JSAMPLE row[s] */
    int row_stride;

    cinfo.err = jpeg_std_error(&jerr);
    /* Now we can initialize the JPEG compression object. */
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename, "wb")) == nullptr ) {
        helios_runtime_error("ERROR (write_JPEG_file): Can't open file " + std::string(filename));
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width; 	/* image width and height, in pixels */
    cinfo.image_height = height;
    cinfo.input_components = 3;		/* # of color components per pixel */
    cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */

    jpeg_set_defaults(&cinfo);

    jpeg_set_quality(&cinfo, 100, TRUE /* limit to baseline-JPEG values */);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width * 3;	/* JSAMPLEs per row in image_buffer */

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW) &screen_shot_trans[ (cinfo.image_height-cinfo.next_scanline-1) * row_stride ];
        (void) jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    /* After finish_compress, we can close the output file. */
    fclose(outfile);

    jpeg_destroy_compress(&cinfo);

    return 1;

}

void read_png_file( const char* filename, std::vector<unsigned char> &texture, uint & height, uint & width){

    std::string fn = filename;
    if( fn.substr(fn.find_last_of('.') + 1) != "png" && fn.substr(fn.find_last_of('.') + 1) != "PNG" ){
        helios_runtime_error("ERROR (read_PNG_file): File " + fn + " is not PNG format.");
    }

    int x, y;

    png_byte color_type;
    png_byte bit_depth;

    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes;
    png_bytep * row_pointers;

    char header[8];    // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(filename, "rb");
    if (!fp){
        helios_runtime_error("ERROR (read_png_file): File " + std::string(filename) + " could not be opened for reading");
    }
    size_t sz = fread(header, 1, 8, fp);
    // if (png_sig_cmp(header, 0, 8)){
    //   std::cerr << "ERROR (read_png_file): File " << filename << " is not recognized as a PNG file." << std::endl;
    //   exit(EXIT_FAILURE);
    // }

    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!png_ptr){
        helios_runtime_error("ERROR (read_png_file): png_create_read_struct failed.");
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr){
        helios_runtime_error("ERROR (read_png_file): png_create_info_struct failed.");
    }

    if (setjmp(png_jmpbuf(png_ptr))){
        helios_runtime_error("ERROR (read_png_file): init_io failed.");
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr))){
        helios_runtime_error("ERROR (read_png_file): read_image failed.");
    }

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
        row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

    png_read_image(png_ptr, row_pointers);

    fclose(fp);

    for (uint j=0; j<height; j++){
        png_byte* row=row_pointers[j];
        for (int i=0; i < width; i++ ){
            png_byte* ba=&(row[i*4]);
            texture.push_back(ba[0]);
            texture.push_back(ba[1]);
            texture.push_back(ba[2]);
            texture.push_back(ba[3]);
        }
    }

    for(y = 0;y<height;y++)
      png_free (png_ptr,row_pointers[y]);
    png_free (png_ptr, row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

}

Visualizer::Visualizer( uint Wdisplay ) : colormap_current(), colormap_lava(), colormap_rainbow(), colormap_gray(), colormap_cool(), colormap_parula(), colormap_hot() {
    initialize(Wdisplay, uint(std::round(Wdisplay * 0.8)), 16, true);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay ) : colormap_lava(), colormap_current(), colormap_cool(), colormap_rainbow(), colormap_parula(), colormap_gray(), colormap_hot() {
    initialize(Wdisplay, Hdisplay, 16, true);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples ) : colormap_lava(), colormap_cool(), colormap_rainbow(), colormap_current(), colormap_hot(), colormap_parula(), colormap_gray() {
    initialize(Wdisplay, Hdisplay, aliasing_samples, true);
}

Visualizer::Visualizer(uint Wdisplay, uint Hdisplay, int aliasing_samples, bool window_decorations ) : colormap_current(), colormap_cool(), colormap_gray(), colormap_hot(), colormap_parula(), colormap_rainbow(), colormap_lava() {
    initialize(Wdisplay, Hdisplay, aliasing_samples, window_decorations);
}

void Visualizer::openWindow(){

    // Open a window and create its OpenGL context
    GLFWwindow* _window;
    _window = glfwCreateWindow( Wdisplay, Hdisplay, "Helios 3D Simulation", nullptr, nullptr);
    if( _window == nullptr ){
        std::string errorsrtring;
        errorsrtring.append("ERROR(Visualizer): Failed to initialize graphics.\n");
        errorsrtring.append("Common causes for this error:\n");
        errorsrtring.append("-- OSX\n  - Is XQuartz installed (xquartz.org) and configured as the default X11 window handler?  When running the visualizer, XQuartz should automatically open and appear in the dock, indicating it is working.\n" );
        errorsrtring.append("-- Linux\n  - Are you running this program remotely via SSH? Remote X11 graphics along with OpenGL are not natively supported.  Installing and using VirtualGL is a good solution for this (virtualgl.org).\n" );
        helios_runtime_error(errorsrtring);
    }
    glfwMakeContextCurrent(_window);

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(_window, GLFW_STICKY_KEYS, GL_TRUE);

    window = (void*) _window;

    int window_width, window_height;
    glfwGetWindowSize(_window, &window_width, &window_height);

    int framebuffer_width, framebuffer_height;
    glfwGetFramebufferSize(_window, &framebuffer_width, &framebuffer_height);

    Wframebuffer = uint(framebuffer_width);
    Hframebuffer = uint(framebuffer_height);

    if( window_width<Wdisplay || window_height<Hdisplay ){
        printf("WARNING (Visualizer): requested size of window is larger than the screen area.\n");
        //printf("Changing width from %d to %d and height from %d to %d\n",Wdisplay,window_width,Hdisplay,window_height);
        Wdisplay = uint(window_width);
        Hdisplay = uint(window_height);
    }

    glfwSetWindowSize( _window, window_width, window_height );

    glfwSetWindowAspectRatio( _window, Wdisplay, Hdisplay );

    // Initialize GLEW
    glewExperimental=GL_TRUE; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        helios_runtime_error("ERROR (Visualizer): Failed to initialize GLEW.");
    }

    //NOTE: for some reason calling glewInit throws an error.  Need to clear it to move on.
    glGetError();

}

void Visualizer::initialize(uint window_width_pixels, uint window_height_pixels, int aliasing_samples, bool window_decorations ){

    Wdisplay = window_width_pixels;
    Hdisplay = window_height_pixels;

    message_flag = true;

    frame_counter = 0;

    camera_FOV = 45;

    context = nullptr;
    contextGeomNeedsUpdate = false;
    primitiveColorsNeedUpdate = false;

    isWatermarkVisible = true;

    colorbar_flag = 0;

    colorbar_min = 0.f;
    colorbar_max = 0.f;

    colorbar_title = "";
    colorbar_fontsize = 12;
    colorbar_fontcolor = RGB::black;

    colorbar_position = make_vec3(0.65,0.1,0.1);
    colorbar_size = make_vec2( 0.15, 0.1 );

    point_width = 1;

    //Initialize OpenGL context and open graphic window

    if( message_flag ){
        std::cout << "Initializing graphics..." << std::flush;
    }

    // Initialise GLFW
    if( !glfwInit() ){
        helios_runtime_error("ERROR (Visualizer::initialize): Failed to initialize GLFW" );
    }

    glfwWindowHint(GLFW_SAMPLES, std::max(0,aliasing_samples) ); // antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
#endif
    glfwWindowHint( GLFW_VISIBLE, 0 );

    if( !window_decorations ){
        glfwWindowHint( GLFW_DECORATED, GLFW_FALSE );
    }

    openWindow();

    // Initialize GLEW
    glewExperimental=GL_TRUE; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        helios_runtime_error("ERROR (Visualizer::initialize): Failed to initialize GLEW");
    }

    //NOTE: for some reason calling glewInit throws an error.  Need to clear it to move on.
    glGetError();

    assert(checkerrors());

    // Enable relevant parameters

    glEnable(GL_DEPTH_TEST); // Enable depth test
    glDepthFunc(GL_LESS); // Accept fragment if it closer to the camera than the former one
    //glEnable(GL_DEPTH_CLAMP);

    if( aliasing_samples<=0 ){
        glDisable( GL_MULTISAMPLE );
        glDisable( GL_MULTISAMPLE_ARB );
    }

    if( aliasing_samples<=1 ){
        glDisable( GL_POLYGON_SMOOTH );
    }else{
        glEnable( GL_POLYGON_SMOOTH );
    }

    //glEnable(GL_TEXTURE0);
    //glEnable(GL_TEXTURE_RECTANGLE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    assert(checkerrors());

    //glEnable(GL_TEXTURE1);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glDisable(GL_CULL_FACE);

    assert(checkerrors());

    //~~~~~~~~~~~~~ Load the Shaders ~~~~~~~~~~~~~~~~~~~//

    primaryShader.initialize( "plugins/visualizer/shaders/primaryShader.vert", "plugins/visualizer/shaders/primaryShader.frag" );
    depthShader.initialize( "plugins/visualizer/shaders/shadow.vert", "plugins/visualizer/shaders/shadow.frag" );

    assert(checkerrors());

    primaryShader.useShader();
    //currentShader = &primaryShader;

    // Initialize frame buffer

    //The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
    glGenFramebuffers(1, &framebufferID);
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);

    //Depth texture. Slower than a depth buffer, but you can sample it later in your shader
    //glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT16, 8192, 8192, 0,GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    //glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT16, 16384, 16384, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);

    assert(checkerrors());

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);

    glDrawBuffer(GL_NONE); // No color buffer is drawn to.

    // Always check that our framebuffer is ok
    int max_checks=10000;
    int checks = 0;
    while( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE && checks<max_checks ){
        checks++;
    }
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);


    // Initialize transformation matrices

    perspectiveTransformationMatrix = glm::mat4(1.f);

    customTransformationMatrix = glm::mat4(1.f);

    // Initialize VBO's
    glGenBuffers(1, &positionBuffer);
    glGenBuffers(1, &colorBuffer);
    glGenBuffers(1, &normalBuffer);
    glGenBuffers(1, &uvBuffer);
    glGenBuffers(1, &textureFlagBuffer);
    glGenBuffers(1, &coordinateFlagBuffer);

    assert(checkerrors());

    // Default values

    light_direction = make_vec3(1,1,1);
    light_direction.normalize();
    primaryShader.setLightDirection(light_direction);

    primaryLightingModel.push_back( Visualizer::LIGHTING_NONE );

    camera_lookat_center = make_vec3(0,0,0);
    camera_eye_location = camera_lookat_center + sphere2cart( make_SphericalCoord(2.f,90.f*M_PI/180.f,0) );

    backgroundColor = make_RGBcolor( 0.8, 0.8, 0.8 );

    //colormaps

    //HOT
    std::vector<RGBcolor> ctable_c{
            make_RGBcolor(0.f, 0.f, 0.f),
            make_RGBcolor(0.5f, 0.f, 0.5f),
            make_RGBcolor( 1.f, 0.f, 0.f ),
            make_RGBcolor( 1.f, 0.5f, 0.f ),
            make_RGBcolor( 1.f, 1.f, 0.f )
    };

    std::vector<float> clocs_c{0.f, 0.25f, 0.5f, 0.75f, 1.f};

    colormap_hot.set( ctable_c, clocs_c, 100, 0, 1 );

    //COOL
    ctable_c = {RGB::cyan, RGB::magenta};

    clocs_c = {0.f, 1.f};

    colormap_cool.set( ctable_c, clocs_c, 100, 0, 1 );

    //LAVA
    ctable_c = {
            make_RGBcolor(0.f, 0.05f, 0.05f),
            make_RGBcolor( 0.f, 0.6f, 0.6f ),
            make_RGBcolor( 1.f, 1.f, 1.f ),
            make_RGBcolor( 1.f, 0.f, 0.f ),
            make_RGBcolor( 0.5f, 0.f, 0.f )
    };

    clocs_c = { 0.f, 0.4f, 0.5f, 0.6f, 1.f };

    colormap_lava.set( ctable_c, clocs_c, 100, 0, 1 );

    //RAINBOW
    ctable_c = {
            RGB::navy,
            RGB::cyan,
            RGB::yellow,
            make_RGBcolor(0.75f, 0.f, 0.f)
    };

    clocs_c = {0, 0.3f, 0.7f, 1.f};

    colormap_rainbow.set( ctable_c, clocs_c, 100, 0, 1 );

    //PARULA
    ctable_c = {
            RGB::navy,
            make_RGBcolor(0,0.6,0.6),
            RGB::goldenrod,
            RGB::yellow
    };

    clocs_c = {0, 0.4f, 0.7f, 1.f};

    colormap_parula.set( ctable_c, clocs_c, 100, 0, 1 );

    //GRAY
    ctable_c = { RGB::black, RGB::white };

    clocs_c = {0.f, 1.f };

    colormap_gray.set( ctable_c, clocs_c, 100, 0, 1 );

    colormap_current = colormap_hot;

    assert(checkerrors());

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}

Visualizer::~Visualizer(){
    glDeleteFramebuffers(1, &framebufferID);
    glDeleteTextures( 1, &depthTexture);
    glDeleteBuffers( 1, &positionBuffer);
    glDeleteBuffers(1, &colorBuffer);
    glDeleteBuffers(1, &normalBuffer);
    glDeleteBuffers(1, &uvBuffer);
    glDeleteBuffers(1, &textureFlagBuffer);
    glDeleteBuffers(1, &coordinateFlagBuffer);

    for( auto iter=textureIDData.begin(); iter!=textureIDData.end(); ++iter ){
        std::vector<int> ID = textureIDData.at(iter->first);
        for(int i : ID){
            uint IDu = uint(i);
            glDeleteTextures(1, &IDu );
        }
    }

    glfwDestroyWindow((GLFWwindow*)window);
    glfwTerminate();
}

int Visualizer::selfTest() {

    if( message_flag ){
        std::cout << "Running visualizer self-test..." << std::flush;
    }

    Visualizer visualizer( 1000 );

    visualizer.setCameraPosition( make_SphericalCoord(10,0.49*M_PI,0), make_vec3(0,0,0) );

    visualizer.setLightingModel( Visualizer::LIGHTING_NONE );

    //---- rectangles ----//

    visualizer.addRectangleByCenter( make_vec3(-1.5,0,0), make_vec2(1,2), make_SphericalCoord(0.f,0.f), make_RGBAcolor(RGB::yellow,0.5), Visualizer::COORDINATES_CARTESIAN );
    visualizer.addRectangleByCenter( make_vec3(-0.5,-0.5,0), make_vec2(1,1), make_SphericalCoord(0.f,0.f), RGB::blue, Visualizer::COORDINATES_CARTESIAN );
    visualizer.addRectangleByCenter( make_vec3(-0.5,0.5,0), make_vec2(1,1), make_SphericalCoord(0.f,0.f), RGB::red, Visualizer::COORDINATES_CARTESIAN );
    visualizer.addRectangleByCenter( make_vec3(1.5,0.5,0), make_vec2(3.41,1), make_SphericalCoord(0,0), "plugins/visualizer/textures/Helios_logo.png", Visualizer::COORDINATES_CARTESIAN );
    visualizer.addRectangleByCenter( make_vec3(1.5,-0.5,0), make_vec2(3.41,1), make_SphericalCoord(0,0), "plugins/visualizer/textures/Helios_logo.jpeg", Visualizer::COORDINATES_CARTESIAN );

    std::vector<vec3> vertices;
    vertices.resize(4);

    vertices.at(0) = make_vec3(-2,-1,0);
    vertices.at(1) = make_vec3(-2,1,0);
    vertices.at(2) = make_vec3(-3,0.5,0);
    vertices.at(3) = make_vec3(-3,-0.5,0);
    visualizer.addRectangleByVertices( vertices, RGB::green, Visualizer::COORDINATES_CARTESIAN );

    vertices.at(0) = make_vec3(-3,-0.5,0);
    vertices.at(1) = make_vec3(-3,0.5,0);
    vertices.at(2) = make_vec3(-4,1,0);
    vertices.at(3) = make_vec3(-4,-1,0);
    visualizer.addRectangleByVertices( vertices, make_RGBAcolor(RGB::violet,0.5), Visualizer::COORDINATES_CARTESIAN );

    vertices.at(0) = make_vec3(-4,-1,0);
    vertices.at(1) = make_vec3(-4,1,0);
    vertices.at(2) = make_vec3(-5,0.5,0);
    vertices.at(3) = make_vec3(-5,-0.5,0);
    visualizer.addRectangleByVertices( vertices, "plugins/visualizer/textures/Helios_logo.png", Visualizer::COORDINATES_CARTESIAN );

    //---- triangles ----//

    vec3 v0, v1, v2;

    v0 = make_vec3(-1,-3,0);
    v1 = make_vec3(1,-3,0);
    v2 = make_vec3(1,-4,0);
    visualizer.addTriangle(v0,v1,v2,make_RGBAcolor(RGB::red,0.5), Visualizer::COORDINATES_CARTESIAN );

    v0 = make_vec3(-1,-3,0);
    v1 = make_vec3(-1,-4,0);
    v2 = make_vec3(1,-4,0);
    visualizer.addTriangle(v0,v1,v2,RGB::blue, Visualizer::COORDINATES_CARTESIAN );

    //---- disks ----//

    visualizer.addDiskByCenter( make_vec3(0,3,0), make_vec2(sqrtf(2)/2.f,sqrtf(2)/2.f), make_SphericalCoord(0,0), 50, RGB::blue, Visualizer::COORDINATES_CARTESIAN );

    visualizer.addDiskByCenter( make_vec3(-3,3,0), make_vec2(sqrtf(2)/2.f,sqrtf(2)/2.f), make_SphericalCoord(0,0), 50, "plugins/visualizer/textures/compass.jpg", Visualizer::COORDINATES_CARTESIAN );

    //---- lines ----//

    visualizer.addLine( make_vec3(-1,3,0), make_vec3(0,4,0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN );
    visualizer.addLine( make_vec3(0,4,0), make_vec3(1,3,0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN );
    visualizer.addLine( make_vec3(1,3,0), make_vec3(0,2,0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN );
    visualizer.addLine( make_vec3(0,2,0), make_vec3(-1,3,0), RGB::red, 1, Visualizer::COORDINATES_CARTESIAN );

    if( message_flag ){
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

void Visualizer::setCameraPosition(const helios::vec3 &cameraPosition, const helios::vec3 &lookAt ){
    camera_eye_location = cameraPosition;
    camera_lookat_center = lookAt;
}

void Visualizer::setCameraPosition(const helios::SphericalCoord &cameraAngle, const helios::vec3 &lookAt ){
    camera_lookat_center = lookAt;
    camera_eye_location = camera_lookat_center + sphere2cart(cameraAngle);vec3 camvec = camera_eye_location-camera_lookat_center;
}

void Visualizer::setCameraFieldOfView( float angle_FOV ){
    camera_FOV = angle_FOV;
}

void Visualizer::setLightDirection(const helios::vec3 &direction ){

    light_direction = direction/direction.magnitude();
    primaryShader.setLightDirection(direction);

}

void Visualizer::getDomainBoundingBox( vec2& xbounds, vec2& ybounds, vec2& zbounds ) const{

    xbounds.x = 1e8;
    xbounds.y = -1e8;
    ybounds.x = 1e8;
    ybounds.y = -1e8;
    zbounds.x = 1e8;
    zbounds.y = -1e8;

    for( std::map<std::string,std::vector<float> >::const_iterator iter = positionData.begin(); iter != positionData.end(); ++iter ){

        std::string ptype = iter->first;

        std::vector<float> positions = iter->second;

        for( size_t p=0; p<positions.size()/3; p++ ){

            if( coordinateFlagData.at(ptype).at(p)==COORDINATES_WINDOW_NORMALIZED ){
                continue;
            }

            vec3 verts;
            verts.x = positions.at(p*3);
            verts.y = positions.at(p*3+1);
            verts.z = positions.at(p*3+2);

            if( verts.x<xbounds.x ){
                xbounds.x = verts.x;
            }
            if( verts.x>xbounds.y ){
                xbounds.y = verts.x;
            }
            if( verts.y<ybounds.x ){
                ybounds.x = verts.y;
            }
            if( verts.y>ybounds.y ){
                ybounds.y = verts.y;
            }
            if( verts.z<zbounds.x ){
                zbounds.x = verts.z;
            }
            if( verts.z>zbounds.y ){
                zbounds.y = verts.z;
            }
        }

    }

    return;

}

float Visualizer::getDomainBoundingRadius() const{

    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox( xbounds, ybounds, zbounds );

    vec3 R;
    R.x = fmax( xbounds.x, xbounds.y );
    R.y = fmax( ybounds.x, ybounds.y );
    R.z = fmax( zbounds.x, zbounds.y );

    return R.magnitude();

}

void Visualizer::setLightingModel(LightingModel lightingmodel ){
    for( uint i=0; i<primaryLightingModel.size(); i++ ){
        primaryLightingModel.at(i) = lightingmodel;
    }
}

void Visualizer::setBackgroundColor(const helios::RGBcolor &color ){
    backgroundColor = color;
}

void Visualizer::printWindow() {

    char outfile[100];
    if( context!=nullptr ){//context has been given to visualizer via buildContextGeometry()
        Date date = context->getDate();
        Time time = context->getTime();
        std::snprintf(outfile,100,"%02d-%02d-%4d_%02d-%02d-%02d_frame%d.jpg",date.day,date.month,date.year,time.hour,time.minute,time.second,frame_counter);
    }else{
        std::snprintf(outfile,100,"frame%d.jpg",frame_counter);
    }
    frame_counter++;

    printWindow( outfile );
}

void Visualizer::printWindow( const char* outfile ){

    write_JPEG_file( outfile, Wframebuffer, Hframebuffer, window );

}

void Visualizer::getWindowPixelsRGB( uint * buffer ){

    std::vector<GLubyte> buff;
    buff.resize( 3*Wframebuffer*Hframebuffer );

    glfwSwapBuffers((GLFWwindow*)window);
    glReadPixels(0, 0, GLsizei(Wframebuffer), GLsizei(Hframebuffer), GL_RGB, GL_UNSIGNED_BYTE, &buff[0]);

    //depending on the active frame buffer, we may get all zero data and need to swap it again.
    bool zeros = true;
    for( int i=0; i<3*Wframebuffer*Hframebuffer; i++ ){
        if( buff[i]!=0 ){
            zeros = false;
        }
    }
    if( zeros ){

        glfwSwapBuffers((GLFWwindow*)window);

        glReadPixels(0, 0, GLsizei(Wframebuffer), GLsizei(Hframebuffer), GL_RGB, GL_UNSIGNED_BYTE, &buff[0]);

    }

    //assert( checkerrors() );

    for( int i=0; i<3*Wframebuffer*Hframebuffer; i++ ){
        buffer[i] = (unsigned int)buff[i];
    }

}

void Visualizer::getDepthMap( float * buffer ){

    if( depth_buffer_data.empty() ){
        helios_runtime_error("ERROR (Visualizer::getDepthMap): No depth map data available. You must run 'plotDepthMap' before depth map can be retrieved.");
    }

    updatePerspectiveTransformation( camera_lookat_center, camera_eye_location );

    for( int i=0; i<depth_buffer_data.size(); i++ ){
        buffer[i] = -perspectiveTransformationMatrix[3].z/(depth_buffer_data.at(i) * -2.0f + 1.0f - perspectiveTransformationMatrix[2].z);
        //buffer[i] = -(depth_buffer_data.at(i) * 2.0 - 1.0 - perspectiveTransformationMatrix[3].z)/perspectiveTransformationMatrix[2].z;
    }

}

void Visualizer::getWindowSize( uint &width, uint &height ) const{
    width = Wdisplay;
    height = Hdisplay;
}

void Visualizer::getFramebufferSize( uint &width, uint &height ) const{
    width = Wframebuffer;
    height = Hframebuffer;
}

void Visualizer::clearGeometry() {
    positionData.clear();
    colorData.clear();
    normalData.clear();
    uvData.clear();
    coordinateFlagData.clear();
    textureFlagData.clear();
    textureIDData.clear();
    contextPrimitiveIDs.clear();
    colorPrimitives_UUIDs.clear();
    colorPrimitives_objIDs.clear();
    contextPrimitiveIDs.clear();
    depth_buffer_data.clear();
    group_start.clear();
}

void Visualizer::closeWindow() {
    glfwHideWindow( (GLFWwindow*) window);
    glfwPollEvents();
}

void Visualizer::hideWatermark() {
    isWatermarkVisible = false;
}

void Visualizer::showWatermark() {
    isWatermarkVisible = true;
}

void Visualizer::updatePerspectiveTransformation(const helios::vec3 &center, const helios::vec3 &eye ){

    float m = fmax( fabs(center.x-eye.x), fmax( fabs(center.y-eye.y), fabs(center.z-eye.z) ) );
    glm::mat4 Projection = glm::perspective( glm::radians(camera_FOV), float(Wdisplay)/float(Hdisplay), 0.01f*m, 100.f*m );
    glm::mat4 View       = glm::lookAt( glm::vec3(eye.x,eye.y,eye.z),glm::vec3(center.x,center.y,center.z),glm::vec3(0,0,1) );

    perspectiveTransformationMatrix = Projection * View ;
}

void Visualizer::updateCustomTransformation(const glm::mat4 &matrix ){
    customTransformationMatrix = matrix;
}

void Visualizer::addRectangleByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color, CoordinateSystem coordFlag ){
    addRectangleByCenter( center, size, rotation, make_RGBAcolor(color.r,color.g,color.b,1), coordFlag );
}

void Visualizer::addRectangleByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color, CoordinateSystem coordFlag ){

    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3( -0.5f*size.x, - 0.5f*size.y, 0.f );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3( +0.5f*size.x, - 0.5f*size.y, 0.f );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3( +0.5f*size.x, +0.5f*size.y, 0.f );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3( -0.5f*size.x, +0.5f*size.y, 0.f );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(3) = center + v3;

    addRectangleByVertices( vertices, color, coordFlag );

}

void Visualizer::addRectangleByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file, CoordinateSystem coordFlag ){

    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3( -0.5f*size.x, - 0.5f*size.y, 0.f );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3( +0.5f*size.x, - 0.5f*size.y, 0.f );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3( +0.5f*size.x, +0.5f*size.y, 0.f );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3( -0.5f*size.x, +0.5f*size.y, 0.f );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(3) = center + v3;

    addRectangleByVertices( vertices, texture_file, coordFlag );

}

void Visualizer::addRectangleByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color, const char* texture_file, CoordinateSystem coordFlag ){

    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3( -0.5f*size.x, - 0.5f*size.y, 0.f );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3( +0.5f*size.x, - 0.5f*size.y, 0.f );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3( +0.5f*size.x, +0.5f*size.y, 0.f );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3( -0.5f*size.x, +0.5f*size.y, 0.f );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(3) = center + v3;

    addRectangleByVertices( vertices, color, texture_file, coordFlag );

}

void Visualizer::addRectangleByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color, const Glyph* glyph, CoordinateSystem coordFlag ){

    std::vector<vec3> vertices;
    vertices.resize(4);

    vec3 v0 = make_vec3( -0.5f*size.x, - 0.5f*size.y, 0.f );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v0 = rotatePointAboutLine( v0, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(0) = center + v0;

    vec3 v1 = make_vec3( +0.5f*size.x, - 0.5f*size.y, 0.f );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v1 = rotatePointAboutLine( v1, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(1) = center + v1;

    vec3 v2 = make_vec3( +0.5f*size.x, +0.5f*size.y, 0.f );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v2 = rotatePointAboutLine( v2, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(2) = center + v2;

    vec3 v3 = make_vec3( -0.5f*size.x, +0.5f*size.y, 0.f );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(1,0,0), -rotation.elevation );
    v3 = rotatePointAboutLine( v3, make_vec3(0,0,0), make_vec3(0,0,1), -rotation.azimuth );
    vertices.at(3) = center + v3;

    addRectangleByVertices( vertices, color, glyph, coordFlag );

}

void Visualizer::addRectangleByVertices( const std::vector<vec3>& vertices, const RGBcolor &color, CoordinateSystem coordFlag ){
    addRectangleByVertices( vertices, make_RGBAcolor(color.r,color.g,color.b,1),  coordFlag );
}

void Visualizer::addRectangleByVertices( const std::vector<vec3>& vertices, const RGBAcolor &color, CoordinateSystem coordFlag ){

    std::vector<vec3> v = vertices; //make a copy so we can modify

    if( coordFlag == COORDINATES_WINDOW_NORMALIZED ){ //No vertex transformation (i.e., identity matrix)

        //Check that coordinates are inside drawable area
        for(auto vertex : vertices){
            if(vertex.x < 0.f || vertex.x > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.y < 0.f || vertex.y > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.z < -1.f || vertex.z > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for( uint i=0; i<vertices.size(); i++ ){
            v.at(i).x = 2.f*v.at(i).x - 1.f;
            v.at(i).y = 2.f*v.at(i).y - 1.f;
        }

    }

    std::vector<float> position_data, color_data, normal_data, uv_data;
    position_data.resize(18,0);
    color_data.resize(24,0);
    normal_data.resize(18,0);
    uv_data.resize(12,0);

    vec3 normal = cross( v.at(1)-v.at(0), v.at(2)-v.at(1) );
    normal.normalize();

    for( int i=0; i<6; i++ ){
        color_data.at(i*4) = color.r;
        color_data.at(i*4+1) = color.g;
        color_data.at(i*4+2) = color.b;
        color_data.at(i*4+3) = color.a;

        normal_data.at(i*3) = normal.x;
        normal_data.at(i*3+1) = normal.y;
        normal_data.at(i*3+2) = normal.z;
    }

    //Lower left vertex
    position_data.at(0) = v.at(0).x;
    position_data.at(1) = v.at(0).y;
    position_data.at(2) = v.at(0).z;

    //Lower right vertex
    position_data.at(3) = v.at(1).x;
    position_data.at(4) = v.at(1).y;
    position_data.at(5) = v.at(1).z;

    //Upper right vertex
    position_data.at(6) = v.at(2).x;
    position_data.at(7) = v.at(2).y;
    position_data.at(8) = v.at(2).z;

    //Lower left vertex
    position_data.at(9) = v.at(0).x;
    position_data.at(10) = v.at(0).y;
    position_data.at(11) = v.at(0).z;

    //Upper right vertex
    position_data.at(12) = v.at(2).x;
    position_data.at(13) = v.at(2).y;
    position_data.at(14) = v.at(2).z;

    //Upper left vertex
    position_data.at(15) = v.at(3).x;
    position_data.at(16) = v.at(3).y;
    position_data.at(17) = v.at(3).z;

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data;
    std::vector<int> coord_data;
    texture_data.resize(6,0);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );
    coord_data.resize(6,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addRectangleByVertices( const std::vector<vec3>& vertices, const char* texture_file, CoordinateSystem coordFlag ){
    std::vector<vec2> uvs;
    uvs.resize(4);
    uvs.at(0) = make_vec2(0,0);
    uvs.at(1) = make_vec2(1,0);
    uvs.at(2) = make_vec2(1,1);
    uvs.at(3) = make_vec2(0,1);
    addRectangleByVertices(vertices,texture_file,uvs,coordFlag);
}

void Visualizer::addRectangleByVertices(const std::vector<vec3> &vertices, const char* texture_file, const std::vector<vec2> &uvs, CoordinateSystem coordFlag ){

    std::vector<vec3> v = vertices; //make a copy so we can modify

    if( coordFlag == COORDINATES_WINDOW_NORMALIZED ){ //No vertex transformation (i.e., identity matrix)

        //Check that coordinates are inside drawable area
        for(auto vertex : vertices){
            if(vertex.x < 0.f || vertex.x > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.y < 0.f || vertex.y > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.z < -1.f || vertex.z > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for( uint i=0; i<vertices.size(); i++ ){
            v.at(i).x = 2.f*v.at(i).x - 1.f;
            v.at(i).y = 2.f*v.at(i).y - 1.f;
        }

    }

    uint textureID;
    int2 texture_size;
    primaryShader.setTextureMap(texture_file,textureID,texture_size);

    std::vector<float> position_data, color_data, normal_data, uv_data;
    position_data.resize(18,0);
    color_data.resize(24,0);
    normal_data.resize(18,0);
    uv_data.resize(12,0);

    vec3 normal = cross( v.at(1)-v.at(0), v.at(2)-v.at(1) );
    normal.normalize();

    for( int i=0; i<6; i++ ){
        normal_data.at(i*3) = normal.x;
        normal_data.at(i*3+1) = normal.y;
        normal_data.at(i*3+2) = normal.z;
    }

    //Lower left vertex
    position_data.at(0) = v.at(0).x;
    position_data.at(1) = v.at(0).y;
    position_data.at(2) = v.at(0).z;
    uv_data.at(0) = uvs.at(0).x*float(texture_size.x-1);
    uv_data.at(1) = (1.f-uvs.at(0).y)*float(texture_size.y-1);

    //Lower right vertex
    position_data.at(3) = v.at(1).x;
    position_data.at(4) = v.at(1).y;
    position_data.at(5) = v.at(1).z;
    uv_data.at(2) = uvs.at(1).x*float(texture_size.x-1);
    uv_data.at(3) = (1.f-uvs.at(1).y)*float(texture_size.y-1);

    //Upper right vertex
    position_data.at(6) = v.at(2).x;
    position_data.at(7) = v.at(2).y;
    position_data.at(8) = v.at(2).z;
    uv_data.at(4) = uvs.at(2).x*float(texture_size.x-1);
    uv_data.at(5) = (1.f-uvs.at(2).y)*float(texture_size.y-1);

    //Lower left vertex
    position_data.at(9) = v.at(0).x;
    position_data.at(10) = v.at(0).y;
    position_data.at(11) = v.at(0).z;
    uv_data.at(6) = uvs.at(0).x*float(texture_size.x-1);
    uv_data.at(7) = (1.f-uvs.at(0).y)*float(texture_size.y-1);

    //Upper right vertex
    position_data.at(12) = v.at(2).x;
    position_data.at(13) = v.at(2).y;
    position_data.at(14) = v.at(2).z;
    uv_data.at(8) = uvs.at(2).x*float(texture_size.x-1);
    uv_data.at(9) = (1.f-uvs.at(2).y)*float(texture_size.y-1);

    //Upper left vertex
    position_data.at(15) = v.at(3).x;
    position_data.at(16) = v.at(3).y;
    position_data.at(17) = v.at(3).z;
    uv_data.at(10) = uvs.at(3).x*float(texture_size.x-1);
    uv_data.at(11) = (1.f-uvs.at(3).y)*float(texture_size.y-1);

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.resize(6,1);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    texture_data.assign(6, textureID);
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );
    coord_data.resize(6,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addRectangleByVertices( const std::vector<vec3>& vertices, const RGBcolor &color, const char* texture_file, CoordinateSystem coordFlag ){
    std::vector<vec2> uvs;
    uvs.resize(4);
    uvs.at(0) = make_vec2(0,0);
    uvs.at(1) = make_vec2(1,0);
    uvs.at(2) = make_vec2(1,1);
    uvs.at(3) = make_vec2(0,1);
    addRectangleByVertices( vertices, color, texture_file, uvs, coordFlag );

}

void Visualizer::addRectangleByVertices( const std::vector<vec3>& vertices, const helios::RGBcolor &color, const char* texture_file, const std::vector<vec2> &uvs, CoordinateSystem coordFlag ){

    std::vector<vec3> v = vertices; //make a copy so we can modify

    if( coordFlag == COORDINATES_WINDOW_NORMALIZED ){ //No vertex transformation (i.e., identity matrix)

        //Check that coordinates are inside drawable area
        for(auto vertex : vertices){
            if(vertex.x < 0.f || vertex.x > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.y < 0.f || vertex.y > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.z < -1.f || vertex.z > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for( uint i=0; i<vertices.size(); i++ ){
            v.at(i).x = 2.f*v.at(i).x - 1.f;
            v.at(i).y = 2.f*v.at(i).y - 1.f;
        }

    }

    uint textureID;
    int2 texture_size;
    primaryShader.setTextureMap(texture_file,textureID,texture_size);

    std::vector<float> position_data, color_data, normal_data, uv_data;
    position_data.resize(18,0);
    color_data.resize(24,0);
    normal_data.resize(18,0);
    uv_data.resize(12,0);

    vec3 normal = cross( v.at(1)-v.at(0), v.at(2)-v.at(1) );
    normal.normalize();

    for( int i=0; i<6; i++ ){
        color_data.at(i*4) = color.r;
        color_data.at(i*4+1) = color.g;
        color_data.at(i*4+2) = color.b;
        color_data.at(i*4+3) = 1;

        normal_data.at(i*3) = normal.x;
        normal_data.at(i*3+1) = normal.y;
        normal_data.at(i*3+2) = normal.z;
    }

    //Lower left vertex
    position_data.at(0) = v.at(0).x;
    position_data.at(1) = v.at(0).y;
    position_data.at(2) = v.at(0).z;
    uv_data.at(0) = uvs.at(0).x*float(texture_size.x-1);
    uv_data.at(1) = (1.f-uvs.at(0).y)*float(texture_size.y-1);

    //Lower right vertex
    position_data.at(3) = v.at(1).x;
    position_data.at(4) = v.at(1).y;
    position_data.at(5) = v.at(1).z;
    uv_data.at(2) = uvs.at(1).x*float(texture_size.x-1);
    uv_data.at(3) = (1.f-uvs.at(1).y)*float(texture_size.y-1);

    //Upper right vertex
    position_data.at(6) = v.at(2).x;
    position_data.at(7) = v.at(2).y;
    position_data.at(8) = v.at(2).z;
    uv_data.at(4) = uvs.at(2).x*float(texture_size.x-1);
    uv_data.at(5) = (1.f-uvs.at(2).y)*float(texture_size.y-1);

    //Lower left vertex
    position_data.at(9) = v.at(0).x;
    position_data.at(10) = v.at(0).y;
    position_data.at(11) = v.at(0).z;
    uv_data.at(6) = uvs.at(0).x*float(texture_size.x-1);
    uv_data.at(7) = (1.f-uvs.at(0).y)*float(texture_size.y-1);

    //Upper right vertex
    position_data.at(12) = v.at(2).x;
    position_data.at(13) = v.at(2).y;
    position_data.at(14) = v.at(2).z;
    uv_data.at(8) = uvs.at(2).x*float(texture_size.x-1);
    uv_data.at(9) = (1.f-uvs.at(2).y)*float(texture_size.y-1);

    //Upper left vertex
    position_data.at(15) = v.at(3).x;
    position_data.at(16) = v.at(3).y;
    position_data.at(17) = v.at(3).z;
    uv_data.at(10) = uvs.at(3).x*float(texture_size.x-1);
    uv_data.at(11) = (1.f-uvs.at(3).y)*float(texture_size.y-1);

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.resize(6,2);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    texture_data.assign(6,textureID);
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );
    coord_data.resize(6,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addRectangleByVertices( const std::vector<vec3>& vertices, const RGBcolor &color, const Glyph* glyph, CoordinateSystem coordFlag ){
    addRectangleByVertices( vertices, make_RGBAcolor(color,1), glyph,  coordFlag );
}

void Visualizer::addRectangleByVertices( const std::vector<vec3>& vertices, const RGBAcolor &color, const Glyph* glyph, CoordinateSystem coordFlag ){

    std::vector<vec3> v = vertices; //make a copy so we can modify

    if( coordFlag == COORDINATES_WINDOW_NORMALIZED ){ //No vertex transformation (i.e., identity matrix)

        //Check that coordinates are inside drawable area
        for(auto vertex : vertices){
            if(vertex.x < 0.f || vertex.x > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.y < 0.f || vertex.y > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.z < -1.f || vertex.z > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addRectangleByVertices): Rectangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for( uint i=0; i<vertices.size(); i++ ){
            v.at(i).x = 2.f*v.at(i).x - 1.f;
            v.at(i).y = 2.f*v.at(i).y - 1.f;
        }

    }

    uint textureID;
    int2 texture_size = glyph->size;
    primaryShader.setTextureMask(glyph,textureID);

    std::vector<float> position_data, color_data, normal_data, uv_data;
    position_data.resize(18,0);
    color_data.resize(24,0);
    normal_data.resize(18,0);
    uv_data.resize(12,0);

    vec3 normal = cross( v.at(1)-v.at(0), v.at(2)-v.at(1) );
    normal.normalize();

    for( int i=0; i<6; i++ ){
        color_data.at(i*4) = color.r;
        color_data.at(i*4+1) = color.g;
        color_data.at(i*4+2) = color.b;
        color_data.at(i*4+3) = color.a;

        normal_data.at(i*3) = normal.x;
        normal_data.at(i*3+1) = normal.y;
        normal_data.at(i*3+2) = normal.z;
    }

    //Lower left vertex
    position_data.at(0) = v.at(0).x;
    position_data.at(1) = v.at(0).y;
    position_data.at(2) = v.at(0).z;
    uv_data.at(0) = 0;
    uv_data.at(1) = texture_size.y;

    //Lower right vertex
    position_data.at(3) = v.at(1).x;
    position_data.at(4) = v.at(1).y;
    position_data.at(5) = v.at(1).z;
    uv_data.at(2) = texture_size.x;
    uv_data.at(3) = texture_size.y;

    //Upper right vertex
    position_data.at(6) = v.at(2).x;
    position_data.at(7) = v.at(2).y;
    position_data.at(8) = v.at(2).z;
    uv_data.at(4) = texture_size.x;
    uv_data.at(5) = 0;

    //Lower left vertex
    position_data.at(9) = v.at(0).x;
    position_data.at(10) = v.at(0).y;
    position_data.at(11) = v.at(0).z;
    uv_data.at(6) = 0;
    uv_data.at(7) = texture_size.y;

    //Upper right vertex
    position_data.at(12) = v.at(2).x;
    position_data.at(13) = v.at(2).y;
    position_data.at(14) = v.at(2).z;
    uv_data.at(8) = texture_size.x;
    uv_data.at(9) = 0;

    //Upper left vertex
    position_data.at(15) = v.at(3).x;
    position_data.at(16) = v.at(3).y;
    position_data.at(17) = v.at(3).z;
    uv_data.at(10) = 0;
    uv_data.at(11) = 0;

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.resize(6,3);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    texture_data.assign(6,textureID);
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );
    coord_data.resize(6,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addTriangle( const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const RGBcolor &color, CoordinateSystem coordFlag ){
    addTriangle( vertex0, vertex1,vertex2, make_RGBAcolor(color.r,color.g,color.b,1), coordFlag);
}

void Visualizer::addTriangle( const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const RGBAcolor &color, CoordinateSystem coordFlag ){

    std::vector<vec3> v{vertex0, vertex1, vertex2};

    if( coordFlag == 0 ){ //No vertex transformation (i.e., identity matrix)

        //Check that coordinates are inside drawable area
        for(auto & vertex : v){
            if(vertex.x < 0.f || vertex.x > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.y < 0.f || vertex.y > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.z < -1.f || vertex.z > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for(auto & vertex : v){
            vertex.x = 2.f * vertex.x - 1.f;
            vertex.y = 2.f * vertex.y - 1.f;
        }

    }

    std::vector<GLfloat> position_data(9), color_data(12), normal_data(9), uv_data(6);

    vec3 normal = cross( vertex1-vertex0, vertex2-vertex0 );
    normal.normalize();

    //Vertex 0
    position_data[0] = v.at(0).x;
    position_data[1] = v.at(0).y;
    position_data[2] = v.at(0).z;
    color_data[0] = color.r;
    color_data[1] = color.g;
    color_data[2] = color.b;
    color_data[3] = color.a;
    normal_data[0] = normal.x;
    normal_data[1] = normal.y;
    normal_data[2] = normal.z;

    //Vertex 1
    position_data[3] = v.at(1).x;
    position_data[4] = v.at(1).y;
    position_data[5] = v.at(1).z;
    color_data[4] = color.r;
    color_data[5] = color.g;
    color_data[6] = color.b;
    color_data[7] = color.a;
    normal_data[3] = normal.x;
    normal_data[4] = normal.y;
    normal_data[5] = normal.z;

    //Vertex 2
    position_data[6] = v.at(2).x;
    position_data[7] = v.at(2).y;
    position_data[8] = v.at(2).z;
    color_data[8] = color.r;
    color_data[9] = color.g;
    color_data[10] = color.b;
    color_data[11] = color.a;
    normal_data[6] = normal.x;
    normal_data[7] = normal.y;
    normal_data[8] = normal.z;

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.resize(3,0);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );

    coord_data.resize(3,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addTriangle( const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const char* texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2, CoordinateSystem coordFlag){

    std::vector<vec3> v{vertex0, vertex1, vertex2};

    if( coordFlag == 0 ){ //No vertex transformation (i.e., identity matrix)

        //Check that coordinates are inside drawable area
        for(auto & vertex : v){
            if(vertex.x < 0.f || vertex.x > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << vertex.x << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.y < 0.f || vertex.y > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << vertex.y << " ) is outside of drawable area." << std::endl;
                }
            }else if(vertex.z < -1.f || vertex.z > 1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << vertex.z << " ) is outside of drawable area." << std::endl;
                }
            }
        }

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for(auto & vertex : v){
            vertex.x = 2.f * vertex.x - 1.f;
            vertex.y = 2.f * vertex.y - 1.f;
        }

    }

    uint textureID;
    int2 texture_size;
    primaryShader.setTextureMap(texture_file,textureID,texture_size);

    std::vector<GLfloat> position_data(9), color_data(12), normal_data(9), uv_data(6);

    vec3 normal = cross( vertex1-vertex0, vertex2-vertex0 );
    normal.normalize();

    //Vertex 0
    position_data[0] = v.at(0).x;
    position_data[1] = v.at(0).y;
    position_data[2] = v.at(0).z;
    uv_data[0] = uv0.x*texture_size.x;
    uv_data[1] = (1.f-uv0.y)*texture_size.y;
    normal_data[0] = normal.x;
    normal_data[1] = normal.y;
    normal_data[2] = normal.z;

    //Vertex 1
    position_data[3] = v.at(1).x;
    position_data[4] = v.at(1).y;
    position_data[5] = v.at(1).z;
    uv_data[2] = uv1.x*texture_size.x;
    uv_data[3] = (1.f-uv1.y)*texture_size.y;
    normal_data[3] = normal.x;
    normal_data[4] = normal.y;
    normal_data[5] = normal.z;

    //Vertex 2
    position_data[6] = v.at(2).x;
    position_data[7] = v.at(2).y;
    position_data[8] = v.at(2).z;
    uv_data[4] = uv2.x*texture_size.x;
    uv_data[5] = (1.f-uv2.y)*texture_size.y;
    normal_data[6] = normal.x;
    normal_data[7] = normal.y;
    normal_data[8] = normal.z;

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.assign(3,1);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    texture_data.assign(3,textureID);
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );

    coord_data.assign(3,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addTriangle( const vec3 &vertex0, const vec3 &vertex1, const vec3 &vertex2, const char* texture_file, const helios::vec2 &uv0, const helios::vec2 &uv1, const helios::vec2 &uv2, const RGBAcolor &color, CoordinateSystem coordFlag){

    std::vector<vec3> v{vertex0, vertex1, vertex2};

    if( coordFlag == 0 ){ //No vertex transformation (i.e., identity matrix)

        //Check that coordinates are inside drawable area
        for( uint i=0; i<v.size(); i++ ){
            if( v.at(i).x<0.f || v.at(i).x>1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `x' position ( " << v.at(i).x << " ) is outside of drawable area." << std::endl;
                }
            }else if( v.at(i).y<0.f || v.at(i).y>1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `y' position ( " << v.at(i).y << " ) is outside of drawable area." << std::endl;
                }
            }else if( v.at(i).z<-1.f || v.at(i).z>1.f ){
                if( message_flag ){
                    std::cout << "WARNING (Visualizer::addTriangle): Triangle `z' position ( " << v.at(i).z << " ) is outside of drawable area." << std::endl;
                }
            }
        }

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for( uint i=0; i<v.size(); i++ ){
            v.at(i).x = 2.f*v.at(i).x - 1.f;
            v.at(i).y = 2.f*v.at(i).y - 1.f;
        }

    }

    uint textureID;
    int2 texture_size;
    primaryShader.setTextureMap(texture_file,textureID,texture_size);

    std::vector<GLfloat> position_data(9), color_data(12), normal_data(9), uv_data(6);

    vec3 normal = cross( vertex1-vertex0, vertex2-vertex0 );
    normal.normalize();

    //Vertex 0
    position_data[0] = v.at(0).x;
    position_data[1] = v.at(0).y;
    position_data[2] = v.at(0).z;
    color_data[0] = color.r;
    color_data[1] = color.g;
    color_data[2] = color.b;
    color_data[3] = color.a;
    uv_data[0] = uv0.x*texture_size.x;
    uv_data[1] = (1.f-uv0.y)*texture_size.y;
    normal_data[0] = normal.x;
    normal_data[1] = normal.y;
    normal_data[2] = normal.z;

    //Vertex 1
    position_data[3] = v.at(1).x;
    position_data[4] = v.at(1).y;
    position_data[5] = v.at(1).z;
    color_data[4] = color.r;
    color_data[5] = color.g;
    color_data[6] = color.b;
    color_data[7] = color.a;
    uv_data[2] = uv1.x*texture_size.x;
    uv_data[3] = (1.f-uv1.y)*texture_size.y;
    normal_data[3] = normal.x;
    normal_data[4] = normal.y;
    normal_data[5] = normal.z;

    //Vertex 2
    position_data[6] = v.at(2).x;
    position_data[7] = v.at(2).y;
    position_data[8] = v.at(2).z;
    color_data[8] = color.r;
    color_data[9] = color.g;
    color_data[10] = color.b;
    color_data[11] = color.a;
    uv_data[4] = uv2.x*texture_size.x;
    uv_data[5] = (1.f-uv2.y)*texture_size.y;
    normal_data[6] = normal.x;
    normal_data[7] = normal.y;
    normal_data[8] = normal.z;

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.resize(3,2);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    texture_data.resize(0);
    texture_data.resize(3,textureID);
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );

    coord_data.resize(3,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addVoxelByCenter( const vec3 &center, const vec3 &size, const SphericalCoord &rotation, const RGBcolor &color, CoordinateSystem coordFlag ){
    addVoxelByCenter( center, size, rotation, make_RGBAcolor(color.r,color.g,color.b,1), coordFlag );
}

void Visualizer::addVoxelByCenter( const vec3 &center, const vec3 &size, const SphericalCoord &rotation, const RGBAcolor &color, CoordinateSystem coordFlag ){

    float eps = 1e-4;  //Avoid z-fighting

    float az = rotation.azimuth;

    vec3 c0 = center + rotatePoint(make_vec3(0,-0.5f*size.y, 0.f),0,az) + eps;
    addRectangleByCenter( c0, make_vec2(size.x,size.z), make_SphericalCoord(-0.5*M_PI,az), color, coordFlag );

    vec3 c1 = center + rotatePoint(make_vec3(0, 0.5f*size.y, 0.f),0,az) + eps;
    addRectangleByCenter( c1, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,az), color, coordFlag );

    vec3 c2 = center + rotatePoint(make_vec3(0.5f*size.x, 0.f, 0.f),0,az) + eps;
    addRectangleByCenter( c2, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI+az), color, coordFlag );

    vec3 c3 = center + rotatePoint(make_vec3(-0.5f*size.x, 0.f, 0.f),0,az) + eps;
    addRectangleByCenter( c3, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI+az), color, coordFlag );

    vec3 c4 = center + make_vec3(0.f, 0.f, -0.5f*size.z) + eps;
    addRectangleByCenter( c4, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,az), color, coordFlag );

    vec3 c5 = center + make_vec3(0.f, 0.f, 0.5f*size.z) + eps;
    addRectangleByCenter( c5, make_vec2(size.x,size.y), make_SphericalCoord(0,az), color, coordFlag );

}

void Visualizer::addDiskByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, uint Ndivisions, const RGBcolor &color, CoordinateSystem coordFlag ){
    addDiskByCenter( center, size, rotation, Ndivisions, make_RGBAcolor(color.r,color.g,color.b,1), coordFlag );
}

void Visualizer::addDiskByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, uint Ndivisions, const RGBAcolor &color, CoordinateSystem coordFlag ){

    vec3 disk_center = center;
    vec2 disk_size = size;

    if( coordFlag == 0 ){ //No vertex transformation (i.e., identity matrix)

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our disk coordinates are from 0 to 1 ---- need to convert
        disk_center.x = 2.f*disk_center.x - 1.f;
        disk_center.y = 2.f*disk_center.y - 1.f;
        disk_size.x = 2.f*disk_size.x;
        disk_size.y = 2.f*disk_size.y;

    }

    std::vector<float> position_data, color_data, normal_data, uv_data;

    vec3 normal = rotatePoint(make_vec3(0,0,1),rotation);

    for( int i=0; i<Ndivisions-1; i++ ){

        //Center
        position_data.push_back( disk_center.x );
        position_data.push_back( disk_center.y );
        position_data.push_back( disk_center.z );
        color_data.push_back( color.r );
        color_data.push_back( color.g );
        color_data.push_back( color.b );
        color_data.push_back( color.a );
        normal_data.push_back( normal.x );
        normal_data.push_back( normal.y);
        normal_data.push_back( normal.z);

        float theta1 = 2.f*M_PI*float(i)/float(Ndivisions-1);
        vec3 v1 = make_vec3( disk_size.x*cos(theta1), disk_size.y*sin(theta1), 0 );
        v1 = rotatePoint(v1,rotation);

        position_data.push_back( disk_center.x + v1.x );
        position_data.push_back( disk_center.y + v1.y );
        position_data.push_back( disk_center.z + v1.z );
        color_data.push_back( color.r );
        color_data.push_back( color.g );
        color_data.push_back( color.b );
        color_data.push_back( color.a );
        normal_data.push_back( normal.x );
        normal_data.push_back( normal.y );
        normal_data.push_back( normal.z );

        float theta2 = 2.f*M_PI*float(i)/float(Ndivisions-1);

        vec3 v2 = make_vec3( disk_size.x*cos(theta2), disk_size.y*sin(theta2), 0 );
        v2 = rotatePoint(v2,rotation);
        position_data.push_back( disk_center.x + v2.x );
        position_data.push_back( disk_center.y + v2.y );
        position_data.push_back( disk_center.z + v2.z );
        color_data.push_back( color.r );
        color_data.push_back( color.g );
        color_data.push_back( color.b );
        color_data.push_back( color.a );
        normal_data.push_back( normal.x );
        normal_data.push_back( normal.y );
        normal_data.push_back( normal.z );

    }

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.resize(position_data.size()/3,0);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );

    coord_data.assign(position_data.size()/3,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addDiskByCenter( const vec3 &center, const vec2 &size, const SphericalCoord &rotation, uint Ndivisions, const char* texture_file, CoordinateSystem coordFlag ){

    vec3 disk_center = center;
    vec2 disk_size = size;

    if( coordFlag == 0 ){ //No vertex transformation (i.e., identity matrix)

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        disk_center.x = 2.f*disk_center.x - 1.f;
        disk_center.y = 2.f*disk_center.y - 1.f;
        disk_size.x = 2.f*disk_size.x;
        disk_size.y = 2.f*disk_size.y;

    }

    uint textureID;
    int2 texture_size;
    primaryShader.setTextureMap(texture_file,textureID,texture_size);

    std::vector<float> position_data, color_data, normal_data, uv_data;

    vec3 normal = rotatePoint(make_vec3(0,0,1),rotation);

    for( int i=0; i<Ndivisions-1; i++ ){

        //Center
        position_data.push_back( disk_center.x );
        position_data.push_back( disk_center.y );
        position_data.push_back( disk_center.z );
        normal_data.push_back( normal.x );
        normal_data.push_back( normal.y);
        normal_data.push_back( normal.z);
        uv_data.push_back( 0.5*(texture_size.x-1) );
        uv_data.push_back( 0.5*(texture_size.y-1) );

        float theta1 = 2.f*M_PI*float(i)/float(Ndivisions-1);
        vec3 v1 = make_vec3( disk_size.x*cos(theta1), disk_size.y*sin(theta1), 0 );
        v1 = rotatePoint(v1,rotation);

        position_data.push_back( disk_center.x + v1.x );
        position_data.push_back( disk_center.y + v1.y );
        position_data.push_back( disk_center.z + v1.z );
        normal_data.push_back( normal.x );
        normal_data.push_back( normal.y );
        normal_data.push_back( normal.z );
        uv_data.push_back( (texture_size.x-1)*(0.5f+0.5f*cos(theta1)) );
        uv_data.push_back( (texture_size.y-1)*(0.5f+0.5f*sin(theta1+M_PI)) );

        float theta2 = 2.f*M_PI*float(i)/float(Ndivisions-1);

        vec3 v2 = make_vec3( disk_size.x*cos(theta2), disk_size.y*sin(theta2), 0 );
        v2 = rotatePoint(v2,rotation);
        position_data.push_back( disk_center.x + v2.x );
        position_data.push_back( disk_center.y + v2.y );
        position_data.push_back( disk_center.z + v2.z );
        normal_data.push_back( normal.x );
        normal_data.push_back( normal.y );
        normal_data.push_back( normal.z );
        uv_data.push_back( (texture_size.x-1)*(0.5f+0.5f*cos(theta2)) );
        uv_data.push_back( (texture_size.y-1)*(0.5f+0.5f*sin(theta2+M_PI)) );

    }

    positionData["triangle"].insert( positionData["triangle"].end(), position_data.begin(), position_data.end() );
    color_data.resize( position_data.size()/3*4 );
    colorData["triangle"].insert( colorData["triangle"].end(), color_data.begin(), color_data.end() );
    normalData["triangle"].insert( normalData["triangle"].end(), normal_data.begin(), normal_data.end() );
    uvData["triangle"].insert( uvData["triangle"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.assign(position_data.size()/3,1);
    textureFlagData["triangle"].insert( textureFlagData["triangle"].end(), texture_data.begin(), texture_data.end() );
    texture_data.assign(position_data.size()/3,textureID);
    textureIDData["triangle"].insert( textureIDData["triangle"].end(), texture_data.begin(), texture_data.end() );

    coord_data.assign(position_data.size()/3,coordFlag);
    coordinateFlagData["triangle"].insert( coordinateFlagData["triangle"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addLine( const vec3 &start, const vec3 &end, const RGBcolor &color, uint linewidth, CoordinateSystem coordFlag ){
    addLine( start, end, make_RGBAcolor(color,1), linewidth, coordFlag );
}

void Visualizer::addLine( const vec3 &start, const vec3 &end, const RGBAcolor &color, uint linewidth, CoordinateSystem coordFlag ){

    line_width = linewidth;

    vec3 s = start;  //copy so that can be modified
    vec3 e = end;

    if( coordFlag == 0 ){ //No vertex transformation (i.e., identity matrix)

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        s = 2.f*s - 1.f;
        e = 2.f*e - 1.f;

    }

    std::vector<float> position_data, color_data, normal_data, uv_data;

    position_data.assign( 6, 0 );
    color_data.assign( 8, 0 );
    normal_data.assign( 6, 0 );
    uv_data.assign( 4, 0 );

    //start
    position_data.at(0) = s.x;
    position_data.at(1) = s.y;
    position_data.at(2) = s.z;
    color_data.at(0) = color.r;
    color_data.at(1) = color.g;
    color_data.at(2) = color.b;
    color_data.at(3) = color.a;

    //end
    position_data.at(3) = e.x;
    position_data.at(4) = e.y;
    position_data.at(5) = e.z;
    color_data.at(4) = color.r;
    color_data.at(5) = color.g;
    color_data.at(6) = color.b;
    color_data.at(7) = color.a;

    positionData["line"].insert( positionData["line"].end(), position_data.begin(), position_data.end() );
    colorData["line"].insert( colorData["line"].end(), color_data.begin(), color_data.end() );
    normalData["line"].insert( normalData["line"].end(), normal_data.begin(), normal_data.end() );
    uvData["line"].insert( uvData["line"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.assign(position_data.size()/3,0);
    textureFlagData["line"].insert( textureFlagData["line"].end(), texture_data.begin(), texture_data.end() );
    textureIDData["line"].insert( textureIDData["line"].end(), texture_data.begin(), texture_data.end() );

    coord_data.assign(position_data.size()/3,coordFlag);
    coordinateFlagData["line"].insert( coordinateFlagData["line"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addPoint( const vec3 &position, const RGBcolor &color, uint pointsize,  CoordinateSystem coordFlag ){
    addPoint( position, make_RGBAcolor(color,1), pointsize,  coordFlag );
}

void Visualizer::addPoint( const vec3 &position, const RGBAcolor &color, uint pointsize,  CoordinateSystem coordFlag ){

    point_width = pointsize;

    vec3 p = position;  //copy so that can be modified

    if( coordFlag == 0 ){ //No vertex transformation (i.e., identity matrix)

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        p = 2.f*p - 1.f;

    }

    std::vector<float> position_data, color_data,normal_data, uv_data;

    position_data.assign( 3, 0 );
    color_data.assign( 4, 0 );
    normal_data.assign( 3, 0 );
    uv_data.assign( 2, 0 );

    position_data.at(0) = p.x;
    position_data.at(1) = p.y;
    position_data.at(2) = p.z;
    color_data.at(0) = color.r;
    color_data.at(1) = color.g;
    color_data.at(2) = color.b;
    color_data.at(3) = color.a;

    positionData["point"].insert( positionData["point"].end(), position_data.begin(), position_data.end() );
    colorData["point"].insert( colorData["point"].end(), color_data.begin(), color_data.end() );
    normalData["point"].insert( normalData["point"].end(), normal_data.begin(), normal_data.end() );
    uvData["point"].insert( uvData["point"].end(), uv_data.begin(), uv_data.end() );

    std::vector<int> texture_data, coord_data;
    texture_data.assign(position_data.size()/3,0);
    textureFlagData["point"].insert( textureFlagData["point"].end(), texture_data.begin(), texture_data.end() );
    textureIDData["point"].insert( textureIDData["point"].end(), texture_data.begin(), texture_data.end() );

    coord_data.assign(position_data.size()/3,coordFlag);
    coordinateFlagData["point"].insert( coordinateFlagData["point"].end(), coord_data.begin(), coord_data.end() );

}

void Visualizer::addSphereByCenter( float radius, const vec3 &center, uint Ndivisions, const RGBcolor &color, CoordinateSystem coordFlag ){
    addSphereByCenter( radius, center, Ndivisions, make_RGBAcolor(color.r,color.g,color.b,1), coordFlag );
}

void Visualizer::addSphereByCenter( float radius, const vec3 &center, uint Ndivisions, const RGBAcolor &color, CoordinateSystem coordFlag ){

    float theta;
    float dtheta=M_PI/float(Ndivisions);
    float dphi=2.0*M_PI/float(Ndivisions);

    //bottom cap
    for( int j=0; j<Ndivisions; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI, 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+dtheta, float(j)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+dtheta, float(j+1)*dphi ) );

        addTriangle(v0,v1,v2,color,coordFlag);

    }

    //top cap
    for( int j=0; j<Ndivisions; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, 0.5*M_PI, 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, 0.5*M_PI-dtheta, float(j)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, 0.5*M_PI-dtheta, float(j+1)*dphi ) );

        addTriangle(v2,v1,v0,color,coordFlag);

    }

    //middle
    for( int j=0; j<Ndivisions; j++ ){
        for( int i=1; i<Ndivisions-1; i++ ){

            vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i)*dtheta, float(j)*dphi ) );
            vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i+1)*dtheta, float(j)*dphi ) );
            vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i+1)*dtheta, float(j+1)*dphi ) );
            vec3 v3 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i)*dtheta, float(j+1)*dphi ) );

            addTriangle(v0,v1,v2,color,coordFlag);
            addTriangle(v0,v2,v3,color,coordFlag);

        }
    }

}

void Visualizer::addSkyDomeByCenter( float radius, const vec3 &center, uint Ndivisions, const char* texture_file, int layer ){
    addSkyDomeByCenter( radius, center, Ndivisions, texture_file );
}

void Visualizer::addSkyDomeByCenter( float radius, const vec3 &center, uint Ndivisions, const char* texture_file ){


    int Ntheta=Ndivisions;
    int Nphi=2*Ndivisions;

    float thetaStart = -0.1*M_PI;

    float dtheta=(0.5*M_PI-thetaStart)/float(Ntheta-1);
    float dphi=2.f*M_PI/float(Nphi-1);

    uint textureID;
    int2 texture_size;
    primaryShader.setTextureMap(texture_file,textureID,texture_size);

    std::vector<float> color_data, normal_data;
    std::vector<int> texture_data, coord_data;

    color_data.assign(24,0);
    normal_data.assign(18,0);

    for( uint j=1; j<Nphi; j++ ){
        for( uint i=1; i<Ntheta; i++ ){

            float theta = thetaStart+i*dtheta;
            float phi=j*dphi;

            vec3 v0( center.x+radius*cos(theta)*cos(phi),center.y+radius*cos(theta)*sin(phi), center.z+radius*(sin(theta)) );
            vec2 uv0( float(Nphi-j)/float(Nphi-1)*(texture_size.x-1), float(Ntheta-i)/float(Ntheta-1)*(texture_size.y-1) );

            vec3 v1( center.x+radius*cos(theta)*cos(phi+dphi),center.y+radius*cos(theta)*sin(phi+dphi), center.z+radius*(sin(theta)) );
            vec2 uv1( float(Nphi-j-1)/float(Nphi-1)*(texture_size.x-1), float(Ntheta-i)/float(Ntheta-1)*(texture_size.y-1) );

            vec3 v2( center.x+radius*cos(theta+dtheta)*cos(phi+dphi),center.y+radius*cos(theta+dtheta)*sin(phi+dphi), center.z+radius*(sin(theta+dtheta)) );
            vec2 uv2( float(Nphi-j-1)/float(Nphi-1)*(texture_size.x-1), float(Ntheta-i-1)/float(Ntheta-1)*(texture_size.y-1) );

            vec3 v3( center.x+radius*cos(theta+dtheta)*cos(phi),center.y+radius*cos(theta+dtheta)*sin(phi), center.z+radius*(sin(theta+dtheta)) );
            vec2 uv3( float(Nphi-j)/float(Nphi-1)*(texture_size.x-1), float(Ntheta-i-1)/float(Ntheta-1)*(texture_size.y-1) );

            positionData["sky"].push_back(v0.x);
            positionData["sky"].push_back(v0.y);
            positionData["sky"].push_back(v0.z);
            positionData["sky"].push_back(v1.x);
            positionData["sky"].push_back(v1.y);
            positionData["sky"].push_back(v1.z);
            positionData["sky"].push_back(v2.x);
            positionData["sky"].push_back(v2.y);
            positionData["sky"].push_back(v2.z);
            uvData["sky"].push_back(uv0.x);
            uvData["sky"].push_back(uv0.y);
            uvData["sky"].push_back(uv1.x);
            uvData["sky"].push_back(uv1.y);
            uvData["sky"].push_back(uv2.x);
            uvData["sky"].push_back(uv2.y);

            positionData["sky"].push_back(v0.x);
            positionData["sky"].push_back(v0.y);
            positionData["sky"].push_back(v0.z);
            positionData["sky"].push_back(v2.x);
            positionData["sky"].push_back(v2.y);
            positionData["sky"].push_back(v2.z);
            positionData["sky"].push_back(v3.x);
            positionData["sky"].push_back(v3.y);
            positionData["sky"].push_back(v3.z);
            uvData["sky"].push_back(uv0.x);
            uvData["sky"].push_back(uv0.y);
            uvData["sky"].push_back(uv2.x);
            uvData["sky"].push_back(uv2.y);
            uvData["sky"].push_back(uv3.x);
            uvData["sky"].push_back(uv3.y);

            texture_data.assign(6,1);
            textureFlagData["sky"].insert( textureFlagData["sky"].end(), texture_data.begin(), texture_data.end() );
            texture_data.assign(6,textureID);
            textureIDData["sky"].insert( textureIDData["sky"].end(), texture_data.begin(), texture_data.end() );

            coord_data.assign(6,COORDINATES_CARTESIAN);
            coordinateFlagData["sky"].insert( coordinateFlagData["sky"].end(), coord_data.begin(), coord_data.end() );

            colorData["sky"].insert( colorData["sky"].end(), color_data.begin(), color_data.end() );
            normalData["sky"].insert( normalData["sky"].end(), normal_data.begin(), normal_data.end() );

        }
    }



}

void Visualizer::addTextboxByCenter( const char* textstring, const vec3 &center, const SphericalCoord &rotation, const RGBcolor &fontcolor, uint fontsize, const char* fontname, CoordinateSystem coordFlag ){

    FT_Library ft; //FreeType objects
    FT_Face face;

    //initialize the freetype library
    if(FT_Init_FreeType(&ft)!=0) {
        helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Could not init freetype library");
    }

    std::vector<std::vector<uint> > maskData; //This will hold the letter mask data

    //Load the font
    char font[100];
    std::snprintf(font,100,"plugins/visualizer/fonts/%s.ttf",fontname);
    if(FT_New_Face(ft, font, 0, &face)!=0) {
        helios_runtime_error("ERROR (Visualizer::addTextboxByCenter): Could not open font '" + std::string(fontname) + "'");
    }

    //Load the font size
    FT_Set_Pixel_Sizes(face, 0, fontsize);

    //x- and y- size of a pixel in [0,1] normalized coordinates
    float sx=1.f/float(Wdisplay);
    float sy=1.f/float(Hdisplay);

    FT_GlyphSlot gg = face->glyph;  //FreeType glyph for font `fontname' and size `fontsize'

    //first, find out how wide the text is going to be
    //This is because we need to know the width beforehand if we want to center the text
    float wtext=0;
    float htext=0;
    const char* textt = textstring;
    for( const char* p = textt; *p; p++) { //looping over each letter in `textstring'
        if(FT_Load_Char(face, *p, FT_LOAD_RENDER)) //load the letter
            continue;
        float offset=0, scale=1;
        if(strncmp(p,"_",1)==0){ //subscript
            offset=-0.3f*sy;
            scale=0.5;
            continue;
        }else if(strncmp(p,"^",1)==0){ //superscript
            offset=0.3f*sy;
            scale=0.5;
            continue;
        }
        wtext += gg->bitmap.width*sx*scale;
        htext = std::max( gg->bitmap.rows*sy, htext );
    }

    //location of the center of our textbox
    float xt=center.x-0.5f*wtext;
    float yt=center.y-0.5f*htext;

    if( message_flag ){
        if( coordFlag==COORDINATES_WINDOW_NORMALIZED ){
            if(xt<0 || xt>1){
                std::cout << "WARNING (Visualizer::addTextboxByCenter): text x-coordinate is outside of window area" << std::endl;
            }
            if(yt<0 || yt>1){
                std::cout << "WARNING (Visualizer::addTextboxByCenter): text y-coordinate is outside of window area" << std::endl;
            }
        }
    }

    FT_GlyphSlot g = face->glyph; //Another FreeType glyph for font `fontname' and size `fontsize'

    const char* text = textstring;

    float offset=0; //baseline offset for subscript/superscript
    float scale=1; //scaling factor for subscript/superscript
    for( const char* p = text; *p; p++) { //looping over each letter in `textstring'

        if(FT_Load_Char(face, *p, FT_LOAD_RENDER)) //load the letter
            continue;

        if(strncmp(p,"_",1)==0){ //subscript
            offset=-0.3*sy;
            scale=0.5;
            continue;
        }else if(strncmp(p,"^",1)==0){ //superscript
            offset=0.3*sy;
            scale=0.5;
            continue;
        }

        //Copy the letter's mask into 2D `maskData' structure
        int2 tsize = make_int2( g->bitmap.width, g->bitmap.rows );
        maskData.resize(tsize.y);
        for(int j=0;j<tsize.y;j++){
            maskData.at(j).resize(tsize.x);
            for(int i=0;i<tsize.x;i++){
                maskData.at(j).at(i) = (uint)g->bitmap.buffer[i+j*tsize.x];
            }
        }

        //size of this letter (i.e., the size of the rectangle we're going to make
        vec2 lettersize = make_vec2( g->bitmap.width*scale*sx, g->bitmap.rows*scale*sy );

        //position of this letter (i.e., the center of the rectangle we're going to make
        vec3 letterposition = make_vec3(xt + g->bitmap_left*sx + 0.5*lettersize.x , yt + g->bitmap_top*( sy+offset ) - 0.5*lettersize.y, center.z );

        //advance the x- and y- letter position
        xt += (g->advance.x >> 6) * sx * scale;
        yt += (g->advance.y >> 6) * sy * scale;

        //reset the offset and scale
        offset=0;
        scale=1;

        if( lettersize.x==0 || lettersize.y==0 ){ //if the size of the letter is 0, don't add a rectangle
            continue;
        }

        Glyph glyph;
        glyph.data = maskData;
        glyph.size = tsize;
        glyph.filename = (char*)""; //Note: we are setting an empty filename to make sure the shader loads a new texture

        addRectangleByCenter(letterposition,lettersize,rotation,make_RGBcolor(fontcolor.r,fontcolor.g,fontcolor.b),&glyph,coordFlag);

    }

    FT_Done_FreeType(ft);

}

void Visualizer::addColorbarByCenter(const char* title, const helios::vec2 &size, const helios::vec3 &center, const helios::RGBcolor &font_color, const Colormap &colormap ){

    uint Ndivs = 50;

    uint Nticks = 4;

    if( !colorbar_ticks.empty() ){
        Nticks = colorbar_ticks.size();
    }

    float dx = size.x/float(Ndivs);

    float cmin = clamp(colormap.getLowerLimit(), -1e7f, 1e7f);
    float cmax = clamp(colormap.getUpperLimit(), -1e7f, 1e7f);

    for( uint i=0; i<Ndivs; i++ ){

        float x = center.x -0.5f*size.x + (float(i)+0.5)*dx;

        RGBcolor color = colormap.query( cmin+float(i)/float(Ndivs)*(cmax-cmin) );

        addRectangleByCenter( make_vec3(x,center.y,center.z), make_vec2(dx,0.5f*size.y), make_SphericalCoord(0,0), color, COORDINATES_WINDOW_NORMALIZED );

    }

    std::vector<vec3> border;
    border.push_back( make_vec3( center.x-0.5*size.x, center.y+0.25*size.y, center.z-0.001 ) );
    border.push_back( make_vec3( center.x+0.5*size.x, center.y+0.25*size.y, center.z-0.001 ) );
    border.push_back( make_vec3( center.x+0.5*size.x, center.y-0.25*size.y, center.z-0.001 ) );
    border.push_back( make_vec3( center.x-0.5*size.x, center.y-0.25*size.y, center.z-0.001 ) );
    border.push_back( make_vec3( center.x-0.5*size.x, center.y+0.25*size.y, center.z-0.001 ) );

    for( uint i=0; i<border.size()-1; i++ ){
        addLine( border.at(i), border.at(i+1), font_color, 1, COORDINATES_WINDOW_NORMALIZED );
    }

    dx = size.x/float(Nticks-1);

    std::vector<vec3> ticks;
    ticks.resize(2);
    for( uint i=0; i<Nticks; i++ ){

        /** \todo Need to use the more sophisticated formatting of tick strings */
        char textstr[10],precision[10];

        float x;
        float value;
        if( colorbar_ticks.size()==0 ){
            x = center.x -0.5f*size.x + float(i)*dx;
            value = cmin+float(i)/float(Nticks-1)*(cmax-cmin);
        }else{
            value = colorbar_ticks.at(i);
            x = center.x -0.5f*size.x + (value-cmin)/(cmax-cmin)*size.x;
        }

        int d1, d2;
        if( std::fabs(floor(value)-value)<1e-4 ){ //value is an integer
            std::snprintf(precision,10,"%%d");
            std::snprintf(textstr,10,precision,int(floor(value)));
        }else if( value!=0.f ){ //value needs decimal formatting
            d1 = floor(log10(std::fabs(value)));
            d2= -d1+1;
            if(d2<1){
                d2=1;
            }
            std::snprintf(precision,10,"%%%u.%uf",(char)abs(d1)+1,(char)d2);
            std::snprintf(textstr,10,precision,value);
        }

        // tick labels
        addTextboxByCenter( textstr, make_vec3(x,center.y-0.4f*size.y,center.z), make_SphericalCoord(0,0), font_color, colorbar_fontsize, "OpenSans-Regular", COORDINATES_WINDOW_NORMALIZED );

        if(i>0 && i<Nticks-1){
            ticks[0] = make_vec3( x, center.y-0.25f*size.y, center.z-0.001f );
            ticks[1] = make_vec3( x, center.y-0.25f*size.y+0.05f*size.y, center.z-0.001f );
            addLine(ticks[0],ticks[1],make_RGBcolor(0.25,0.25,0.25),1,COORDINATES_WINDOW_NORMALIZED);
            ticks[0] = make_vec3( x, center.y+0.25f*size.y, center.z-0.001f );
            ticks[1] = make_vec3( x, center.y+0.25f*size.y-0.05f*size.y, center.z-0.001f );
            addLine(ticks[0],ticks[1], make_RGBcolor(0.25,0.25,0.25),1,COORDINATES_WINDOW_NORMALIZED);
        }

    }

    //title
    addTextboxByCenter( title, make_vec3( center.x, center.y+0.4*size.y, center.z), make_SphericalCoord(0,0), font_color, colorbar_fontsize, "CantoraOne-Regular", COORDINATES_WINDOW_NORMALIZED );

}

void Visualizer::addCoordinateAxes(){
    addCoordinateAxes(helios::make_vec3(0,0,0), helios::make_vec3(1,1,1), "positive");
}

void Visualizer::addCoordinateAxes(const helios::vec3 &origin, const helios::vec3 &length, const std::string &sign){


    float mult;
    if(sign == "both"){
        mult = 1.0;
    }else{
        mult = 0.0;
    }

    float Lmag = length.magnitude();

    // x axis
    addLine(make_vec3(mult*-1.0*length.x + origin.x, origin.y, origin.z), make_vec3(length.x + origin.x, origin.y, origin.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);

    if(length.x > 0){
        addTextboxByCenter("+ X", helios::make_vec3(1.2*length.x + origin.x, origin.y, origin.z), helios::make_SphericalCoord(0,0), helios::make_RGBcolor( 0.f, 0.f, 0.f ), uint(200*Lmag), "OpenSans-Regular", Visualizer::COORDINATES_CARTESIAN);
    }

    // y axis
    addLine(make_vec3(origin.x, mult*-1.0*length.y+origin.y, origin.z), make_vec3(origin.x, length.y+origin.y, origin.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);

    if(length.y > 0){
        addTextboxByCenter("+ Y", helios::make_vec3( origin.x, 1.1*length.y+origin.y, origin.z), helios::make_SphericalCoord(0,0), helios::make_RGBcolor( 0.f, 0.f, 0.f ), uint(200*Lmag), "OpenSans-Regular", Visualizer::COORDINATES_CARTESIAN);
    }

    // z axis
    addLine(make_vec3(origin.x, origin.y, mult*-1*length.z+origin.z), make_vec3(origin.x, origin.y, length.z+origin.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);

    if(length.z > 0){
        addTextboxByCenter("+ Z", helios::make_vec3( origin.x, origin.y, length.z+origin.z), helios::make_SphericalCoord(0,0), helios::make_RGBcolor( 0.f, 0.f, 0.f ), uint(200*Lmag), "OpenSans-Regular", Visualizer::COORDINATES_CARTESIAN);

    }


}


void Visualizer::addGridWireFrame(const helios::vec3 &center, const helios::vec3 &size, const helios::int3 &subdiv){
    
    helios::vec3 boxmin, boxmax;
    boxmin = make_vec3(center.x - 0.5*size.x, center.y - 0.5*size.y, center.z - 0.5*size.z);
    boxmax = make_vec3(center.x + 0.5*size.x, center.y + 0.5*size.y, center.z + 0.5*size.z);
    
    float spacing_x = size.x/subdiv.x;
    float spacing_y = size.y/subdiv.y;
    float spacing_z = size.z/subdiv.z;
    
    for(int i=0; i <= subdiv.x; i++)
    {
        for(int j=0; j <= subdiv.y; j++)
        {
            addLine(make_vec3(boxmin.x + i*spacing_x, boxmin.y + j*spacing_y, boxmin.z), make_vec3(boxmin.x + i*spacing_x, boxmin.y + j*spacing_y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        }
    }
    
    for(int i=0; i <= subdiv.z ; i++)
    {
        for(int j=0; j <= subdiv.y; j++)
        {
            addLine(make_vec3(boxmin.x , boxmin.y + j*spacing_y, boxmin.z + i*spacing_z), make_vec3(boxmax.x, boxmin.y + j*spacing_y, boxmin.z + i*spacing_z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        }
    }
    
    for(int i=0; i <= subdiv.x ; i++)
    {
        for(int j=0; j <= subdiv.z ; j++)
        {
            addLine(make_vec3(boxmin.x + i*spacing_x , boxmin.y, boxmin.z + j*spacing_z), make_vec3(boxmin.x + i*spacing_x, boxmax.y, boxmin.z + j*spacing_z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        }
    }
    
}

void Visualizer::enableColorbar() {
    colorbar_flag = 2;
}

void Visualizer::disableColorbar() {
    colorbar_flag = 1;
}

void Visualizer::setColorbarPosition( vec3 position ){
    if( position.x < 0 || position.x>1 || position.y<0 || position.y>1 || position.z<-1 || position.z>1 ){
        helios_runtime_error("ERROR (Visualizer::setColorbarPosition): position is out of range.  Coordinates must be: 0<x<1, 0<y<1, -1<z<1.");
    }
    colorbar_position = position;
}

void Visualizer::setColorbarSize( vec2 size ){
    if( size.x < 0 || size.x>1 || size.y<0 || size.y>1 ){
        helios_runtime_error("ERROR (Visualizer::setColorbarSize): Size must be greater than 0 and less than the window size (i.e., 1).");
    }
    colorbar_size = size;
}

void Visualizer::setColorbarRange( float cmin, float cmax ){
    if( message_flag && cmin>cmax ){
        std::cout << "WARNING (Visualizer::setColorbarRange): Maximum colorbar value must be greater than minimum value...Ignoring command." << std::endl;
        return;
    }
    colorbar_min = cmin;
    colorbar_max = cmax;
}

void Visualizer::setColorbarTicks(const std::vector<float> &ticks ){

    //check that vector is not empty
    if( ticks.empty() ){
        helios_runtime_error("ERROR (Visualizer::setColorbarTicks): Colorbar ticks vector is empty.");
    }

    //Check that ticks are monotonically increasing
    for( int i=1; i<ticks.size(); i++ ){
        if( ticks.at(i)<=ticks.at(i-1) ){
            helios_runtime_error("ERROR (Visualizer::setColorbarTicks): Colorbar ticks must be monotonically increasing.");
        }
    }

    //Check that ticks are within the range of colorbar values
    for( int i=ticks.size()-1; i>=0; i-- ){
        if( ticks.at(i)<colorbar_min ){
            colorbar_min = ticks.at(i);
        }
    }
    for( float tick : ticks){
        if( tick>colorbar_max ){
            colorbar_max = tick;
        }
    }

    colorbar_ticks = ticks;
}

void Visualizer::setColorbarTitle( const char* title ){
    colorbar_title = title;
}

void Visualizer::setColorbarFontSize( uint font_size ){
    if( font_size<=0 ){
        helios_runtime_error("ERROR (Visualizer::setColorbarFontSize): Font size must be greater than zero.");
    }
    colorbar_fontsize = font_size;
}

void Visualizer::setColorbarFontColor( RGBcolor fontcolor ){
    colorbar_fontcolor = fontcolor;
}

void Visualizer::setColormap( Ctable colormap_name ){
    if( colormap_name==COLORMAP_HOT ){
        colormap_current = colormap_hot;
    }else if( colormap_name==COLORMAP_COOL ){
        colormap_current = colormap_cool;
    }else if( colormap_name==COLORMAP_LAVA ){
        colormap_current = colormap_lava;
    }else if( colormap_name==COLORMAP_RAINBOW ){
        colormap_current = colormap_rainbow;
    }else if( colormap_name==COLORMAP_PARULA ){
        colormap_current = colormap_parula;
    }else if( colormap_name==COLORMAP_GRAY ){
        colormap_current = colormap_gray;
    }else if( colormap_name==COLORMAP_CUSTOM ){
        helios_runtime_error("ERROR (Visualizer::setColormap): Setting a custom colormap requires calling setColormap with additional arguments defining the colormap.");
    }else{
        helios_runtime_error("ERROR (Visualizer::setColormap): Invalid colormap.");
    }
}

void Visualizer::setColormap(const std::vector<RGBcolor> &colors, const std::vector<float> &divisions ){

    if( colors.size() != divisions.size() ){
        helios_runtime_error("ERROR (Visualizer::setColormap): The number of colors must be equal to the number of divisions.");
    }

    Colormap colormap_custom( colors, divisions, 100, 0, 1 );

    colormap_current = colormap_custom;

}

Colormap Visualizer::getCurrentColormap() const{
    return colormap_current;
}

void Visualizer::buildContextGeometry( helios::Context* context_ptr ){
    std::vector<uint> UUIDs = context_ptr->getAllUUIDs();
    buildContextGeometry(context_ptr, UUIDs);
}

void Visualizer::buildContextGeometry(helios::Context* context_ptr, const std::vector<uint>& UUIDs ){

    if( UUIDs.empty() ){
        std::cerr << "WARNING (Visualizer::buildContextGeometry): There is no Context geometry to build...exiting." << std::endl;
        return;
    }


    context = context_ptr;
    contextGeomNeedsUpdate = true;

    contextPrimitiveIDs.insert(contextPrimitiveIDs.begin(), UUIDs.begin(), UUIDs.end() );

    //Set the view to fit window
    if( camera_lookat_center.x==0 && camera_lookat_center.y==0 && camera_lookat_center.z==0 ){ //default center
        if( camera_eye_location.x<1e-4 && camera_eye_location.y<1e-4 && camera_eye_location.z==2.f ){ //default eye position

            vec3 center_sph;
            vec2 xbounds, ybounds, zbounds;
            float radius;
            context->getDomainBoundingSphere( UUIDs, center_sph, radius );
            context->getDomainBoundingBox( UUIDs, xbounds, ybounds, zbounds );
            camera_lookat_center = make_vec3( 0.5*(xbounds.x+xbounds.y), 0.5*(ybounds.x+ybounds.y), 0.5*(zbounds.x+zbounds.y) );
            camera_eye_location = camera_lookat_center + sphere2cart( make_SphericalCoord(2.f*radius,20.f*M_PI/180.f,0) );

        }
    }

}

void Visualizer::buildContextGeometry_private() {

    contextGeomNeedsUpdate = false;

    if( !colorPrimitivesByData.empty() ){

        if( colorPrimitives_UUIDs.empty() ) { //load all primitives
            for (uint UUID: contextPrimitiveIDs) {
                if (context->doesPrimitiveExist(UUID)) {
                    colorPrimitives_UUIDs[UUID] = UUID;
                }
            }
        }else { //double check that primitives exist
            for (uint UUID: contextPrimitiveIDs) {
                if ( !context->doesPrimitiveExist(UUID)) {
                    auto it = colorPrimitives_UUIDs.find(UUID);
                    colorPrimitives_UUIDs.erase( it );
                }
            }
        }

    }else if( !colorPrimitivesByObjectData.empty()  ) {

        if( colorPrimitives_objIDs.empty() ) { //load all primitives
            std::vector<uint> ObjIDs = context->getAllObjectIDs();
            for( uint objID: ObjIDs ) {
                if( context->doesObjectExist(objID) ) {
                    std::vector<uint> UUIDs = context->getObjectPointer(objID)->getPrimitiveUUIDs();
                    for( uint UUID : UUIDs ) {
                        if (context->doesPrimitiveExist(UUID)) {
                            colorPrimitives_UUIDs[UUID] = UUID;
                        }
                    }
                }
            }
        }else { //load primitives specified by user
            for ( const auto &objID: colorPrimitives_objIDs ) {
                if( context->doesObjectExist( objID.first )) {
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

    if( !colorPrimitives_UUIDs.empty() && colorbar_flag == 0) {
        enableColorbar();
    }

    //------ Colormap ------//

    uint psize = contextPrimitiveIDs.size();
    if( message_flag ){
        if( psize>0 ){
            if( psize>=1e3 && psize<1e6 ){
                std::cout << "Adding " << psize/1e3 << "K Context primitives to visualizer...." << std::flush;
            }else if( psize>=1e6 ){
                std::cout << "Adding " << psize/1e6 << "M Context primitives to visualizer...." << std::flush;
            }else{
                std::cout << "Adding " << psize << " Context primitives to visualizer...." << std::flush;
            }
        }else{
            std::cout << "WARNING (Visualizer::buildContextGeometry): No primitives were found in the Context..." << std::endl;
        }
    }

    //do a pre-sort of primitive UUIDs by texture
    std::map<std::string,std::vector<uint> > UUID_texture;
    for( uint p=0; p<psize; p++ ){

        uint UUID = contextPrimitiveIDs.at(p);

        if( context->doesPrimitiveExist(UUID) ) {

            std::string texture_file = context->getPrimitiveTextureFile(UUID);

            UUID_texture[texture_file].push_back(UUID);

        }

    }

    //figure out colorbar range

    colormap_current.setRange( colorbar_min, colorbar_max );
    if( colorbar_min==0 && colorbar_max==0 ){//range was not set by user, use full range of values

        colorbar_min = INFINITY;
        colorbar_max = -INFINITY;

        for(std::map<std::string,std::vector<uint> >::iterator iter = UUID_texture.begin(); iter != UUID_texture.end(); ++iter){

            std::vector<uint> UUIDs = iter->second;

            for( size_t u=0; u<UUIDs.size(); u++ ){

                uint UUID = UUIDs.at(u);

                float colorValue=-9999;
                if( colorPrimitivesByData.size()!=0 ){
                    if( colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end() ){
                        if( context->doesPrimitiveDataExist( UUID, colorPrimitivesByData.c_str() ) ){
                            HeliosDataType type = context->getPrimitiveDataType( UUID, colorPrimitivesByData.c_str() );
                            if( type==HELIOS_TYPE_FLOAT ){
                                context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), colorValue );
                            }else if( type==HELIOS_TYPE_INT ){
                                int cv;
                                context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), cv );
                                colorValue = float(cv);
                            }else if( type==HELIOS_TYPE_UINT ){
                                uint cv;
                                context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), cv );
                                colorValue = float(cv);
                            }else if( type==HELIOS_TYPE_DOUBLE ){
                                double cv;
                                context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), cv );
                                colorValue = float(cv);
                            }else{
                                colorValue = 0;
                            }
                        }else{
                            colorValue = 0;
                        }
                    }
                }else if( colorPrimitivesByObjectData.size()!=0 ){
                    if( colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end() ){
                        uint ObjID = context->getPrimitiveParentObjectID(UUID);
                        if( ObjID==0 ){
                            colorValue = 0;
                        }else if( context->doesObjectDataExist( ObjID, colorPrimitivesByObjectData.c_str() ) ){
                            HeliosDataType type = context->getObjectDataType( ObjID, colorPrimitivesByObjectData.c_str() );
                            if( type==HELIOS_TYPE_FLOAT ){
                                context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), colorValue );
                            }else if( type==HELIOS_TYPE_INT ){
                                int cv;
                                context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), cv );
                                colorValue = float(cv);
                            }else if( type==HELIOS_TYPE_UINT ){
                                uint cv;
                                context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), cv );
                                colorValue = float(cv);
                            }else if( type==HELIOS_TYPE_DOUBLE ){
                                double cv;
                                context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), cv );
                                colorValue = float(cv);
                            }else{
                                colorValue = 0;
                            }
                        }else{
                            colorValue = 0;
                        }
                    }
                }

                if( colorValue!=colorValue || colorValue==INFINITY ) {//check for NaN or infinity
                    colorValue = 0;
                }

                if( colorValue!=-9999 ){
                    if( colorValue<colorbar_min ){
                        colorbar_min = colorValue;;
                    }
                    if( colorValue>colorbar_max ){
                        colorbar_max = colorValue;;
                    }
                }

            }
        }

        if( colorbar_min!=INFINITY && colorbar_max!=-INFINITY ){
            colormap_current.setRange( colorbar_min, colorbar_max );
        }
    }

    if( !colorPrimitivesByData.empty() ) {
        assert(colorbar_min <= colorbar_max);
    }

    //------- Simulation Geometry -------//

    //add primiitves

    for(std::map<std::string,std::vector<uint> >::iterator iter = UUID_texture.begin(); iter != UUID_texture.end(); ++iter){

        std::vector<uint> UUIDs = iter->second;

        for( size_t u=0; u<UUIDs.size(); u++ ){

            uint UUID = UUIDs.at(u);

            if( !context->doesPrimitiveExist(UUID) ){
                std::cerr << "WARNING (Visualizer::buildContextGeometry): UUID vector contains ID(s) that do not exist in the Context...they will be ignored." << std::endl;
                continue;
            }

            helios::PrimitiveType ptype = context->getPrimitiveType(UUID);

            std::vector<vec3> verts = context->getPrimitiveVertices(UUID);
            std::string texture_file = context->getPrimitiveTextureFile(UUID);

            RGBAcolor color;
            float colorValue;
            if( colorPrimitivesByData.size()!=0 ){
                if( colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end()  ){
                    if( context->doesPrimitiveDataExist( UUID, colorPrimitivesByData.c_str() ) ){
                        HeliosDataType type = context->getPrimitiveDataType( UUID, colorPrimitivesByData.c_str() );
                        if( type==HELIOS_TYPE_FLOAT ){
                            context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), colorValue );
                        }else if( type==HELIOS_TYPE_INT ){
                            int cv;
                            context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), cv );
                            colorValue = float(cv);
                        }else if( type==HELIOS_TYPE_UINT ){
                            uint cv;
                            context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), cv );
                            colorValue = float(cv);
                        }else if( type==HELIOS_TYPE_DOUBLE ){
                            double cv;
                            context->getPrimitiveData( UUID, colorPrimitivesByData.c_str(), cv );
                            colorValue = float(cv);
                        }else{
                            colorValue = 0;
                        }

                    }else{
                        colorValue = 0;
                    }

                    if( colorValue!=colorValue || colorValue==INFINITY ) {//check for NaN or infinity
                        colorValue = 0;
                    }

                    color = make_RGBAcolor(colormap_current.query( colorValue ),1);
                }else{
                    color = context->getPrimitiveColorRGBA(UUID);
                }
            }else if( colorPrimitivesByObjectData.size()!=0 ){
                if( colorPrimitives_UUIDs.find(UUID) != colorPrimitives_UUIDs.end() ){
                    uint ObjID = context->getPrimitiveParentObjectID(UUID);
                    if( ObjID==0 ){
                        colorValue = 0;
                    }else if( context->doesObjectDataExist( ObjID, colorPrimitivesByObjectData.c_str() ) ){
                        HeliosDataType type = context->getObjectDataType( ObjID, colorPrimitivesByObjectData.c_str() );
                        if( type==HELIOS_TYPE_FLOAT ){
                            context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), colorValue );
                        }else if( type==HELIOS_TYPE_INT ){
                            int cv;
                            context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), cv );
                            colorValue = float(cv);
                        }else if( type==HELIOS_TYPE_UINT ){
                            uint cv;
                            context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), cv );
                            colorValue = float(cv);
                        }else if( type==HELIOS_TYPE_DOUBLE ){
                            double cv;
                            context->getObjectData( ObjID, colorPrimitivesByObjectData.c_str(), cv );
                            colorValue = float(cv);
                        }else{
                            colorValue = 0;
                        }
                    }else{
                        colorValue = 0;
                    }

                    if( colorValue!=colorValue || colorValue==INFINITY ) {//check for NaN or infinity
                        colorValue = 0;
                    }

                    color = make_RGBAcolor(colormap_current.query( colorValue ),1);
                }else{
                    color = context->getPrimitiveColorRGBA(UUID);
                }
            }else{
                color = context->getPrimitiveColorRGBA(UUID);
            }

            if( ptype == helios::PRIMITIVE_TYPE_PATCH  ){

                if( texture_file.size()==0 ){//Patch does not have an associated texture or we are ignoring texture
                    addRectangleByVertices( verts, color, COORDINATES_CARTESIAN );
                }else{ //Patch has a texture

                    std::vector<vec2> uvs = context->getPrimitiveTextureUV(UUID);

                    if( colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.size()==0 ){//coloring primitive based on texture
                        if( uvs.size()==4 ){//custom (u,v) coordinates
                            if( context->isPrimitiveTextureColorOverridden(UUID) ){
                                addRectangleByVertices( verts, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), uvs, COORDINATES_CARTESIAN );
                            }else{
                                addRectangleByVertices( verts, texture_file.c_str(), uvs, COORDINATES_CARTESIAN );
                            }
                        }else{//default (u,v) coordinates
                            if( context->isPrimitiveTextureColorOverridden(UUID) ){
                                addRectangleByVertices( verts, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                            }else{
                                addRectangleByVertices( verts, texture_file.c_str(), COORDINATES_CARTESIAN );
                            }
                        }
                    }else{//coloring primitive based on primitive data
                        if( uvs.size()==4 ){//custom (u,v) coordinates
                            addRectangleByVertices( verts, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), uvs, COORDINATES_CARTESIAN );
                        }else{//default (u,v) coordinates
                            addRectangleByVertices( verts, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                        }
                    }
                }

            }else if( ptype == helios::PRIMITIVE_TYPE_TRIANGLE ){

                if( texture_file.size()==0 ){//Triangle does not have an associated texture or we are ignoring texture
                    addTriangle( verts.at(0), verts.at(1), verts.at(2), color, COORDINATES_CARTESIAN );
                }else{ //Triangle has a texture

                    std::vector<vec2> uvs = context->getPrimitiveTextureUV(UUID);

                    if( colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.size()==0  ){//coloring primitive based on texture
                        if( context->isPrimitiveTextureColorOverridden(UUID) ){
                            addTriangle( verts.at(0), verts.at(1), verts.at(2), texture_file.c_str(), uvs.at(0), uvs.at(1), uvs.at(2), make_RGBAcolor(color.r,color.g,color.b,1), COORDINATES_CARTESIAN );
                        }else{
                            addTriangle( verts.at(0), verts.at(1), verts.at(2), texture_file.c_str(), uvs.at(0), uvs.at(1), uvs.at(2), COORDINATES_CARTESIAN );
                        }
                    }else{//coloring primitive based on primitive data
                        addTriangle( verts.at(0), verts.at(1), verts.at(2), texture_file.c_str(), uvs.at(0), uvs.at(1), uvs.at(2), make_RGBAcolor(color.r,color.g,color.b,1), COORDINATES_CARTESIAN );
                    }

                }

            }else if( ptype == helios::PRIMITIVE_TYPE_VOXEL ){

                std::vector<vec3> v_vertices = context->getPrimitiveVertices(UUID);

                std::vector<vec3> bottom_vertices, top_vertices, mx_vertices, px_vertices, my_vertices, py_vertices;
                bottom_vertices.resize(4);
                top_vertices.resize(4);
                mx_vertices.resize(4);
                px_vertices.resize(4);
                my_vertices.resize(4);
                py_vertices.resize(4);

                //bottom
                bottom_vertices.at(0) = v_vertices.at(0);
                bottom_vertices.at(1) = v_vertices.at(1);
                bottom_vertices.at(2) = v_vertices.at(2);
                bottom_vertices.at(3) = v_vertices.at(3);

                //top
                top_vertices.at(0) = v_vertices.at(4);
                top_vertices.at(1) = v_vertices.at(5);
                top_vertices.at(2) = v_vertices.at(6);
                top_vertices.at(3) = v_vertices.at(7);

                //-x
                mx_vertices.at(0) = v_vertices.at(0);
                mx_vertices.at(1) = v_vertices.at(3);
                mx_vertices.at(2) = v_vertices.at(7);
                mx_vertices.at(3) = v_vertices.at(4);

                //+x
                px_vertices.at(0) = v_vertices.at(1);
                px_vertices.at(1) = v_vertices.at(2);
                px_vertices.at(2) = v_vertices.at(6);
                px_vertices.at(3) = v_vertices.at(5);

                //-y
                my_vertices.at(0) = v_vertices.at(0);
                my_vertices.at(1) = v_vertices.at(1);
                my_vertices.at(2) = v_vertices.at(5);
                my_vertices.at(3) = v_vertices.at(4);

                //+y
                py_vertices.at(0) = v_vertices.at(2);
                py_vertices.at(1) = v_vertices.at(3);
                py_vertices.at(2) = v_vertices.at(7);
                py_vertices.at(3) = v_vertices.at(6);

                if( texture_file.size()==0 ){//Voxel does not have an associated texture or we are ignoring texture

                    addRectangleByVertices( bottom_vertices, color, COORDINATES_CARTESIAN );
                    addRectangleByVertices( top_vertices, color, COORDINATES_CARTESIAN );
                    addRectangleByVertices( mx_vertices, color, COORDINATES_CARTESIAN );
                    addRectangleByVertices( px_vertices, color, COORDINATES_CARTESIAN );
                    addRectangleByVertices( my_vertices, color, COORDINATES_CARTESIAN );
                    addRectangleByVertices( py_vertices, color, COORDINATES_CARTESIAN );

                }else{ //Voxel has a texture

                    if( colorPrimitives_UUIDs.find(UUID) == colorPrimitives_UUIDs.end() || colorPrimitives_UUIDs.size()==0  ){//coloring primitive based on texture
                        if( context->isPrimitiveTextureColorOverridden(UUID) ){
                            addRectangleByVertices( bottom_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( top_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( mx_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( px_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( my_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( py_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                        }else{
                            addRectangleByVertices( bottom_vertices, texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( top_vertices, texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( mx_vertices, texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( px_vertices, texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( my_vertices, texture_file.c_str(), COORDINATES_CARTESIAN );
                            addRectangleByVertices( py_vertices, texture_file.c_str(), COORDINATES_CARTESIAN );
                        }
                    }else{//coloring primitive based on primitive data
                        addRectangleByVertices( bottom_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                        addRectangleByVertices( top_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                        addRectangleByVertices( mx_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                        addRectangleByVertices( px_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                        addRectangleByVertices( my_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                        addRectangleByVertices( py_vertices, make_RGBcolor(color.r,color.g,color.b), texture_file.c_str(), COORDINATES_CARTESIAN );
                    }

                }


            }

        }
    }

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}

void Visualizer::colorContextPrimitivesByData( const char* data_name ){
    colorPrimitivesByData = data_name;
    colorPrimitivesByObjectData = "";
    if( !colorPrimitives_UUIDs.empty() ){
        colorPrimitives_UUIDs.clear();
    }
    if( !colorPrimitives_objIDs.empty() ){
        colorPrimitives_objIDs.clear();
    }
}

void Visualizer::colorContextPrimitivesByData( const char* data_name, const std::vector<uint>& UUIDs ){
    colorPrimitivesByData = data_name;
    colorPrimitivesByObjectData = "";
    for( uint UUID : UUIDs ){
        colorPrimitives_UUIDs[UUID] = UUID;
    }
    if( !colorPrimitives_objIDs.empty() ){
        colorPrimitives_objIDs.clear();
    }
}

void Visualizer::colorContextPrimitivesByObjectData( const char* data_name ){
    colorPrimitivesByObjectData = data_name;
    colorPrimitivesByData = "";
    if( !colorPrimitives_UUIDs.empty() ){
        colorPrimitives_UUIDs.clear();
    }
    if( !colorPrimitives_objIDs.empty() ){
        colorPrimitives_objIDs.clear();
    }
}

void Visualizer::colorContextPrimitivesByObjectData( const char* data_name, const std::vector<uint>& ObjIDs ){
    colorPrimitivesByObjectData = data_name;
    colorPrimitivesByData = "";
    for( uint objID : ObjIDs ){
        colorPrimitives_objIDs[objID] = objID;
    }
    if( !colorPrimitives_UUIDs.empty() ){
        colorPrimitives_UUIDs.clear();
    }
}

void Visualizer::colorContextPrimitivesRandomly(const std::vector<uint>& UUIDs ){

    disableColorbar();
    if( !colorPrimitives_UUIDs.empty() ){
        colorPrimitives_UUIDs.clear();
    }
    for( uint UUID : UUIDs ){
        float rc = randu();
        context->setPrimitiveData(UUID, "random_color", rc);
    }
    
    colorPrimitivesByData = "random_color";
    colorPrimitivesByObjectData = "";
    for( uint UUID : UUIDs ){
        colorPrimitives_UUIDs[UUID] = UUID;
    }
    if( !colorPrimitives_objIDs.empty() ){
        colorPrimitives_objIDs.clear();
    }
}

void Visualizer::colorContextPrimitivesRandomly(){

    disableColorbar();
    
    std::vector<uint> all_UUIDs = context->getAllUUIDs(); 
    for( uint UUID : all_UUIDs ){
        float rc = randu();
        context->setPrimitiveData(UUID, "random_color", rc);
    }
    
    colorPrimitivesByData = "random_color";
    colorPrimitivesByObjectData = "";
    if( !colorPrimitives_UUIDs.empty() ){
        colorPrimitives_UUIDs.clear();
    }
    if( !colorPrimitives_objIDs.empty() ){
        colorPrimitives_objIDs.clear();
    }
    
} 


void Visualizer::colorContextObjectsRandomly(const std::vector<uint>& ObjIDs ){

    disableColorbar();
    if( !colorPrimitives_UUIDs.empty() ){
        colorPrimitives_UUIDs.clear();
    }
    for( uint ObjID : ObjIDs ){
        float rc = randu();
        context->setObjectData(ObjID, "random_color", rc);
    }
    
    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "random_color";

}

void Visualizer::colorContextObjectsRandomly(){
    std::vector<uint> all_ObjIDs = context->getAllObjectIDs();
    disableColorbar();
    if( !colorPrimitives_UUIDs.empty() ){
        colorPrimitives_UUIDs.clear();
    }
    for( uint ObjID : all_ObjIDs ){
        float rc = randu();
        context->setObjectData(ObjID, "random_color", rc);
    }
    
    colorPrimitivesByData = "";
    colorPrimitivesByObjectData = "random_color";

}








std::vector<helios::vec3> Visualizer::plotInteractive() {

    if( message_flag ){
        std::cout << "Generating interactive plot..." << std::endl;
    }

//    glfwShowWindow( (GLFWwindow*) window);

//    openWindow();

    //Update the Context geometry (if needed)
    if( contextGeomNeedsUpdate ){
        buildContextGeometry_private();
    }else{
        colormap_current.setRange( colorbar_min, colorbar_max );
    }

    //Update
    if( colorbar_flag==2 ){
        addColorbarByCenter( colorbar_title.c_str(), colorbar_size, colorbar_position, colorbar_fontcolor, colormap_current );
    }

    //Watermark
    if( isWatermarkVisible ){
        float hratio = float(Wdisplay)/float(Hdisplay);
        float width = 0.2389f/0.8f/hratio;
        addRectangleByCenter( make_vec3(0.75f*width,0.95f,0), make_vec2(width,0.07), make_SphericalCoord(0,0), "plugins/visualizer/textures/Helios_watermark.png", COORDINATES_WINDOW_NORMALIZED );
    }

    setupPlot();

    //domain bounding box
    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox( xbounds, ybounds, zbounds );

    glm::vec3 view_center = glm::vec3( xbounds.x+0.5*(xbounds.y-xbounds.x), ybounds.x+0.5*(ybounds.y-ybounds.x), zbounds.x+0.5*(zbounds.y-zbounds.x) );
    //float bound_R = 2.f*fmax(xbounds.y-xbounds.x,fmax(ybounds.y-ybounds.x,zbounds.y-zbounds.x));
    float bound_R = 0.75*sqrtf( pow(xbounds.y-xbounds.x,2) + pow(ybounds.y-ybounds.x,2) + pow(zbounds.y-zbounds.x,2) );

    glm::vec3 lightInvDir = view_center + glm::vec3(light_direction.x,light_direction.y,light_direction.z);

    bool shadow_flag = false;
    for( uint m=0; m<primaryLightingModel.size(); m++ ){
        if( primaryLightingModel.at(m) == Visualizer::LIGHTING_PHONG_SHADOWED ){
            shadow_flag = true;
            break;
        }
    }

    glm::mat4 depthMVP;

    if( shadow_flag ){

        // Depth buffer for shadows
        glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);
        glViewport(0,0,8192,8192); // Render on the whole framebuffer, complete from the lower left corner to the upper right
        //glViewport(0,0,16384,16384); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        // Clear the screen
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        depthShader.useShader();

        // Compute the MVP matrix from the light's point of view
        glm::mat4 depthProjectionMatrix = glm::ortho<float>(-bound_R,bound_R,-bound_R,bound_R,-bound_R,bound_R);
        glm::mat4 depthViewMatrix = glm::lookAt(lightInvDir, view_center, glm::vec3(0,0,1));
        depthMVP = depthProjectionMatrix * depthViewMatrix;

        depthShader.setTransformationMatrix( depthMVP );

        //bind depth texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTexture);

        depthShader.enableTextureMaps();
        depthShader.enableTextureMasks();

        render( 1 );

    }else{

        depthMVP = glm::mat4(1.0);

    }

    assert(checkerrors());

    std::vector<vec3> camera_output;

    glfwShowWindow( (GLFWwindow*) window);

    do{

        // Render to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        //glViewport(0,0,Wdisplay,Hdisplay); // Render on the whole framebuffer, complete from the lower left corner to the upper right
        glViewport(0,0,Wframebuffer,Hframebuffer);

        glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 0.0f);

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        primaryShader.useShader();

        updatePerspectiveTransformation( camera_lookat_center, camera_eye_location );

        glm::mat4 biasMatrix(
                0.5, 0.0, 0.0, 0.0,
                0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.5, 0.5, 0.5, 1.0
        );

        glm::mat4 DepthBiasMVP = biasMatrix*depthMVP;

        primaryShader.setDepthBiasMatrix( DepthBiasMVP );

        primaryShader.setTransformationMatrix( perspectiveTransformationMatrix );

        primaryShader.enableTextureMaps();
        primaryShader.enableTextureMasks();

        primaryShader.setLightingModel( primaryLightingModel.at(0) );

        glBindTexture(GL_TEXTURE_2D, depthTexture);
        glUniform1i(primaryShader.shadowmapUniform,1);

        render( 0 );

        glfwPollEvents();
        getViewKeystrokes( camera_eye_location, camera_lookat_center );

        glfwSwapBuffers((GLFWwindow*)window);

        glfwWaitEvents();

        int width, height;
        //glfwGetWindowSize((GLFWwindow*)window, &width, &height );
        //Wdisplay = width;
        //Hdisplay = height;
        glfwGetFramebufferSize((GLFWwindow*)window, &width, &height );
        Wframebuffer = width;
        Hframebuffer = height;

    }while( glfwGetKey((GLFWwindow*)window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose((GLFWwindow*)window) == 0 );

    glfwPollEvents();

    assert(checkerrors());

    camera_output.push_back(camera_eye_location);
    camera_output.push_back(camera_lookat_center);

    return camera_output;

}

void Visualizer::setupPlot(){

    glEnableVertexAttribArray(0); //position
    glEnableVertexAttribArray(1); //color
    glEnableVertexAttribArray(2); //normal
    glEnableVertexAttribArray(3); //uv
    glEnableVertexAttribArray(4); //texture flag
    glEnableVertexAttribArray(5); //coordinate flag

    std::vector<float> position_data, color_data, normal_data, uv_data;
    std::vector<int> coordinate_data, texture_data, textureID_data;

    std::vector<std::string> keys{"triangle", "line", "point", "sky" };

    for( int i=0; i<keys.size(); i++ ){
        position_data.insert( position_data.end(), positionData[keys.at(i)].begin(), positionData[keys.at(i)].end() );
        color_data.insert( color_data.end(), colorData[keys.at(i)].begin(), colorData[keys.at(i)].end() );
        normal_data.insert( normal_data.end(), normalData[keys.at(i)].begin(), normalData[keys.at(i)].end() );
        uv_data.insert( uv_data.end(), uvData[keys.at(i)].begin(), uvData[keys.at(i)].end() );
        coordinate_data.insert( coordinate_data.end(), coordinateFlagData[keys.at(i)].begin(), coordinateFlagData[keys.at(i)].end() );
        texture_data.insert( texture_data.end(), textureFlagData[keys.at(i)].begin(), textureFlagData[keys.at(i)].end() );
    }

    // 1st attribute buffer : vertex positions
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer );
    glBufferData(GL_ARRAY_BUFFER, position_data.size()*sizeof(GLfloat), &position_data[0], GL_STATIC_DRAW);
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    // 2nd attribute buffer : vertex colors
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer );
    glBufferData(GL_ARRAY_BUFFER, color_data.size()*sizeof(GLfloat), &color_data[0], GL_STATIC_DRAW);
    glVertexAttribPointer( 1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    // 3rd attribute buffer : vertex normals
    glBindBuffer(GL_ARRAY_BUFFER, normalBuffer );
    glBufferData(GL_ARRAY_BUFFER, normal_data.size()*sizeof(GLfloat), &normal_data[0], GL_STATIC_DRAW);
    glVertexAttribPointer( 2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    // 4th attribute buffer : vertex uv
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer );
    glBufferData(GL_ARRAY_BUFFER, uv_data.size()*sizeof(GLfloat), &uv_data[0], GL_STATIC_DRAW);
    glVertexAttribPointer( 3, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    // 5th attribute buffer : vertex texture flag
    glBindBuffer(GL_ARRAY_BUFFER, textureFlagBuffer );
    glBufferData(GL_ARRAY_BUFFER, texture_data.size()*sizeof(GLint), &texture_data[0], GL_STATIC_DRAW);
    glVertexAttribIPointer( 4, 1, GL_INT, 0, (void*)0 );

    // 6th attribute buffer : vertex coordinate flag
    glBindBuffer(GL_ARRAY_BUFFER, coordinateFlagBuffer );
    glBufferData(GL_ARRAY_BUFFER, coordinate_data.size()*sizeof(GLint), &coordinate_data[0], GL_STATIC_DRAW);
    glVertexAttribIPointer( 5, 1, GL_INT, 0, (void*)0 );

    //figure out texture switches

    uint textureID_current;
    if( textureIDData["triangle"].size()==0 ){
        textureID_current = 0;
    }else{
        textureID_current = textureIDData["triangle"].at(0);
    }

    size_t start_current = 0;
    for( size_t p=1; p<textureIDData["triangle"].size(); p++ ){

        if( textureID_current!=textureIDData["triangle"].at(p) || p == textureIDData["triangle"].size()-1 ){
            group_start[ start_current ] = make_int2(textureID_current,p-start_current+1);
            textureID_current = textureIDData["triangle"].at(p);
            start_current = p;
        }else{
            continue;
        }

    }

}

void Visualizer::render( bool shadow ){

    //--- Triangles------

    uint textureID_current;
    if( textureIDData["triangle"].size()==0 ){
        textureID_current = 0;
    }else{
        textureID_current = textureIDData["triangle"].at(0);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE,0);

    for( std::map<uint,int2>::iterator iter=group_start.begin(); iter!=group_start.end(); ++iter ){

        if( iter->second.x!=0 ){
            glBindTexture(GL_TEXTURE_RECTANGLE,iter->second.x);
        }

        glDrawArrays(GL_TRIANGLES, iter->first, iter->second.y );

    }

    if( !shadow ){

        size_t triangle_size = positionData["triangle"].size()/3;
        size_t line_size = positionData["line"].size()/3;
        size_t point_size = positionData["point"].size()/3;

        if( line_size>0 ){
            glDrawArrays(GL_LINES, triangle_size, line_size );
        }

        if( point_size>0 ){
            glPointSize( point_width );
            glDrawArrays(GL_POINTS, triangle_size+line_size, point_size );
        }

        if( positionData["sky"].size()>0 ){
            primaryShader.setLightingModel( LIGHTING_NONE );
            glBindTexture(GL_TEXTURE_RECTANGLE,textureIDData["sky"].at(0));
            glDrawArrays(GL_TRIANGLES, triangle_size+line_size+point_size, positionData["sky"].size()/3 );
        }

    }

}

void Visualizer::plotUpdate() {
    plotUpdate( false );
}

void Visualizer::plotUpdate( bool hide_window ){

    if( message_flag ){
        std::cout << "Updating the plot..." << std::flush;
    }

    if( !hide_window ) {
        glfwShowWindow((GLFWwindow *) window);
    }

    //Update the Context geometry (if needed)
    if( contextGeomNeedsUpdate ){
        buildContextGeometry_private();
    }else{
        colormap_current.setRange( colorbar_min, colorbar_max );
    }

    //Update
    if( colorbar_flag==2 ){
        addColorbarByCenter( colorbar_title.c_str(), colorbar_size, colorbar_position, colorbar_fontcolor, colormap_current );
    }

    //Watermark
    if( isWatermarkVisible ){
        float hratio = float(Wdisplay)/float(Hdisplay);
        float width = 0.2389f/0.8f/hratio;
        addRectangleByCenter( make_vec3(0.75f*width,0.95f,0), make_vec2(width,0.07), make_SphericalCoord(0,0), "plugins/visualizer/textures/Helios_watermark.png", COORDINATES_WINDOW_NORMALIZED );
    }

    setupPlot();

    //domain bounding box
    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox( xbounds, ybounds, zbounds );

    glm::vec3 view_center = glm::vec3( xbounds.x+0.5*(xbounds.y-xbounds.x), ybounds.x+0.5*(ybounds.y-ybounds.x), zbounds.x+0.5*(zbounds.y-zbounds.x) );
    //float bound_R = 2.f*fmax(xbounds.y-xbounds.x,fmax(ybounds.y-ybounds.x,zbounds.y-zbounds.x));
    float bound_R = 0.75*sqrtf( pow(xbounds.y-xbounds.x,2) + pow(ybounds.y-ybounds.x,2) + pow(zbounds.y-zbounds.x,2) );

    glm::vec3 lightInvDir = view_center + glm::vec3(light_direction.x,light_direction.y,light_direction.z);

    bool shadow_flag = false;
    for( uint m=0; m<primaryLightingModel.size(); m++ ){
        if( primaryLightingModel.at(m) == Visualizer::LIGHTING_PHONG_SHADOWED ){
            shadow_flag = true;
            break;
        }
    }

    glm::mat4 depthMVP;

    if( shadow_flag ){

        // Depth buffer for shadows
        glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);
        glViewport(0,0,8192,8192); // Render on the whole framebuffer, complete from the lower left corner to the upper right
        //glViewport(0,0,16384,16384); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        // Clear the screen
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        depthShader.useShader();

        // Compute the MVP matrix from the light's point of view
        glm::mat4 depthProjectionMatrix = glm::ortho<float>(-bound_R,bound_R,-bound_R,bound_R,-bound_R,bound_R);
        glm::mat4 depthViewMatrix = glm::lookAt(lightInvDir, view_center, glm::vec3(0,0,1));
        depthMVP = depthProjectionMatrix * depthViewMatrix;

        depthShader.setTransformationMatrix( depthMVP );

        //bind depth texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTexture);

        depthShader.enableTextureMaps();
        depthShader.enableTextureMasks();

        render( 1 );

    }else{

        depthMVP = glm::mat4(1.0);

    }

    assert(checkerrors());

    // Render to the screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //glViewport(0,0,Wdisplay,Hdisplay); // Render on the whole framebuffer, complete from the lower left corner to the upper right
    glViewport(0,0,Wframebuffer,Hframebuffer);

    glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 0.0f);

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    primaryShader.useShader();

    updatePerspectiveTransformation( camera_lookat_center, camera_eye_location );

    glm::mat4 biasMatrix(
            0.5, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.5, 0.5, 0.5, 1.0
    );

    glm::mat4 DepthBiasMVP = biasMatrix*depthMVP;

    primaryShader.setDepthBiasMatrix( DepthBiasMVP );

    primaryShader.setTransformationMatrix( perspectiveTransformationMatrix );

    primaryShader.enableTextureMaps();
    primaryShader.enableTextureMasks();

    primaryShader.setLightingModel( primaryLightingModel.at(0) );

    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glUniform1i(primaryShader.shadowmapUniform,1);

    render( 0 );

    glfwPollEvents();
    getViewKeystrokes( camera_eye_location, camera_lookat_center );

    int width, height;
    glfwGetFramebufferSize((GLFWwindow *) window, &width, &height);
    Wframebuffer = width;
    Hframebuffer = height;

    glfwSwapBuffers((GLFWwindow*)window);

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}

void Visualizer::plotDepthMap() {

    if( message_flag ){
        std::cout << "Rendering depth map..." << std::flush;
    }

    //Update the Context geometry (if needed)
    if( contextGeomNeedsUpdate ){
        buildContextGeometry_private();
    }

    setupPlot();

    //domain bounding box
    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox( xbounds, ybounds, zbounds );

    glm::vec3 view_center = glm::vec3( xbounds.x+0.5*(xbounds.y-xbounds.x), ybounds.x+0.5*(ybounds.y-ybounds.x), zbounds.x+0.5*(zbounds.y-zbounds.x) );
    float bound_R = 1.4f*0.5f*fmax(xbounds.y-xbounds.x,fmax(ybounds.y-ybounds.x,zbounds.y-zbounds.x));

    glm::mat4 depthMVP;

    // Depth buffer for shadows
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferID);
    //glViewport(0,0, 8192, 8192); // Render on the whole framebuffer, complete from the lower left corner to the upper right
    //glViewport(0,0, Wdisplay, Hdisplay); // Render on the whole framebuffer, complete from the lower left corner to the upper right
    glViewport(0,0,Wframebuffer,Hframebuffer);

    //bind depth texture
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    //glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT16, Wdisplay, Hdisplay, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT16, Wframebuffer, Hframebuffer, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    // Clear the screen
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    depthShader.useShader();

    updatePerspectiveTransformation( camera_lookat_center, camera_eye_location );
    depthShader.setTransformationMatrix( perspectiveTransformationMatrix );

    depthShader.enableTextureMaps();
    depthShader.enableTextureMasks();

    render( 1 );

    assert(checkerrors());

    depth_buffer_data.resize( Wframebuffer*Hframebuffer );

    glfwSwapBuffers((GLFWwindow*)window);
    glReadPixels(0, 0, Wframebuffer, Hframebuffer, GL_DEPTH_COMPONENT, GL_FLOAT, &depth_buffer_data[0] );

    //depending on the ative frame buffer, we may get all zero data and need to swap it again.
    bool zeros = true;
    for( int i=0; i<3*Wframebuffer*Hframebuffer; i++){
        if( depth_buffer_data[i]!=0 ){
            zeros = false;
        }
    }
    if( zeros ){

        glfwSwapBuffers((GLFWwindow*)window);

        glReadPixels(0, 0, Wframebuffer, Hframebuffer, GL_DEPTH_COMPONENT, GL_FLOAT, &depth_buffer_data[0] );

    }

    assert(checkerrors());

    glEnableVertexAttribArray(0); //position
    glEnableVertexAttribArray(1); //color
    glDisableVertexAttribArray(2); //normal
    glEnableVertexAttribArray(3); //uv
    glEnableVertexAttribArray(4); //texture flag
    glEnableVertexAttribArray(5); //coordinate flag

    std::vector<float> position_data, color_data, normal_data, uv_data;
    std::vector<int> coordinate_data, texture_data;

    position_data.assign(12,0);
    color_data.assign(16,0);
    normal_data.assign(12,0);
    uv_data.assign(8,0);
    coordinate_data.assign(4,0);
    texture_data.assign(4,0);

    position_data[0] = -1;
    position_data[1] = 1;
    position_data[2] = 0;
    uv_data[0] = 0;
    uv_data[1] = 1;
    color_data[0] = 1;
    color_data[1] = 0;
    color_data[2] = 0;
    color_data[3] = 1;
    texture_data[0] = 4;
    coordinate_data[0] = 0;

    position_data[3] = 1;
    position_data[4] = 1;
    position_data[5] = 0;
    uv_data[2] = 1;
    uv_data[3] = 1;
    color_data[4] = 1;
    color_data[5] = 0;
    color_data[6] = 0;
    color_data[7] = 1;
    texture_data[1] = 4;
    coordinate_data[1] = 0;

    position_data[6] = 1;
    position_data[7] = -1;
    position_data[8] = 0;
    uv_data[4] = 1;
    uv_data[5] = 0;
    color_data[8] = 1;
    color_data[9] = 0;
    color_data[10] = 0;
    color_data[11] = 1;
    texture_data[2] = 4;
    coordinate_data[2] = 0;

    position_data[9] = -1;
    position_data[10] = -1;
    position_data[11] = 0;
    uv_data[6] = 0;
    uv_data[7] = 0;
    color_data[12] = 1;
    color_data[13] = 0;
    color_data[14] = 0;
    color_data[15] = 1;
    texture_data[3] = 4;
    coordinate_data[3] = 0;

    // 1st attribute buffer : vertex positions
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer );
    glBufferData(GL_ARRAY_BUFFER, position_data.size()*sizeof(GLfloat), &position_data[0], GL_STATIC_DRAW);
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    // 2nd attribute buffer : vertex colors
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer );
    glBufferData(GL_ARRAY_BUFFER, color_data.size()*sizeof(GLfloat), &color_data[0], GL_STATIC_DRAW);
    glVertexAttribPointer( 1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    // 4th attribute buffer : vertex uv
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer );
    glBufferData(GL_ARRAY_BUFFER, uv_data.size()*sizeof(GLfloat), &uv_data[0], GL_STATIC_DRAW);
    glVertexAttribPointer( 3, 2, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    // 5th attribute buffer : vertex texture flag
    glBindBuffer(GL_ARRAY_BUFFER, textureFlagBuffer );
    glBufferData(GL_ARRAY_BUFFER, texture_data.size()*sizeof(GLint), &texture_data[0], GL_STATIC_DRAW);
    glVertexAttribIPointer( 4, 1, GL_INT, 0, (void*)0 );

    // 6th attribute buffer : vertex coordinate flag
    glBindBuffer(GL_ARRAY_BUFFER, coordinateFlagBuffer );
    glBufferData(GL_ARRAY_BUFFER, coordinate_data.size()*sizeof(GLint), &coordinate_data[0], GL_STATIC_DRAW);
    glVertexAttribIPointer( 5, 1, GL_INT, 0, (void*)0 );


    //do{

    // Render to the screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //glViewport(0,0,Wdisplay,Hdisplay); // Render on the whole framebuffer, complete from the lower left corner to the upper right
    glViewport(0,0,Wframebuffer,Hframebuffer);

    glClearColor(0,0,0, 0.0f);

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    primaryShader.useShader();

    primaryShader.enableTextureMaps();
    primaryShader.enableTextureMasks();

    primaryShader.setLightingModel( LIGHTING_NONE );

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);

    glUniform1i(primaryShader.RboundUniform,bound_R);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4 );

    glfwSwapBuffers((GLFWwindow*)window);

    //glfwWaitEvents();
    //}while( glfwGetKey((GLFWwindow*)window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose((GLFWwindow*)window) == 0 );

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}

void Shader::initialize( const char* vertex_shader_file, const char* fragment_shader_file ){

    // ~~~~~~~~~~~~~~~ COMPILE SHADERS ~~~~~~~~~~~~~~~~~~~~~~~~~//

    assert(checkerrors());

    // Create the shaders
    unsigned int VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    unsigned int FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_shader_file, std::ios::in);
    assert(VertexShaderStream.is_open());
    std::string Line = "";
    while(getline(VertexShaderStream, Line))
        VertexShaderCode += "\n" + Line;
    VertexShaderStream.close();

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_shader_file, std::ios::in);
    assert(FragmentShaderStream.is_open());
    Line = "";
    while(getline(FragmentShaderStream, Line))
        FragmentShaderCode += "\n" + Line;
    FragmentShaderStream.close();

    int Result = GL_FALSE;
    int InfoLogLength;

    // Compile Vertex Shader
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);

    assert(checkerrors());

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> VertexShaderErrorMessage(InfoLogLength);
    if( !VertexShaderErrorMessage.empty() ) {
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
    }
    int success = 0;
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &success);
    if( success==GL_FALSE ){
        fprintf(stderr, "%s\n", &VertexShaderErrorMessage[0]);
        throw(1);
    }

    // Compile Fragment Shader
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);

    assert(checkerrors());

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength);
    if( !VertexShaderErrorMessage.empty() ) {
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
    }
    success = 0;
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &success);
    if( success==GL_FALSE ){
        fprintf(stderr, "%s\n", &FragmentShaderErrorMessage[0]);
        throw(1);
    }

    // Link the program
    shaderID = glCreateProgram();
    glAttachShader(shaderID, VertexShaderID);
    glAttachShader(shaderID, FragmentShaderID);
    glLinkProgram(shaderID);

    assert(checkerrors());

    // Check the program
    glGetProgramiv(shaderID, GL_LINK_STATUS, &Result);
    glGetProgramiv(shaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> ProgramErrorMessage( std::max(InfoLogLength, int(1)) );
    glGetProgramInfoLog(shaderID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
    if( isspace( ProgramErrorMessage[0] ) ){
        fprintf(stderr, "%s\n", &ProgramErrorMessage[0]);
        throw(1);
    }

    assert(checkerrors());

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    assert(checkerrors());

    glUseProgram(shaderID);

    assert(checkerrors());

    // ~~~~~~~~~~~ Create a Vertex Array Object (VAO) ~~~~~~~~~~//
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    assert(checkerrors());

    // ~~~~~~~~~~~ Primary Shader Uniforms ~~~~~~~~~~//

    //Transformation Matrix
    transformMatrixUniform = glGetUniformLocation(shaderID, "MVP");

    //Depth Bias Matrix (for shadows)
    depthBiasUniform = glGetUniformLocation(shaderID, "DepthBiasMVP");

    //Texture Sampler
    textureUniform = glGetUniformLocation(shaderID, "textureSampler");
    glUniform1i(textureUniform,0); //tell shader we are using GL_TEXTURE0

    //Shadow Map Sampler
    shadowmapUniform = glGetUniformLocation(shaderID, "shadowMap");
    glUniform1i(shadowmapUniform, 1);

    //Flag to tell how to shade fragments. See also: setTextureMap, setMaskTexture, disableTextures
    //textureFlagUniform = glGetUniformLocation(shaderID, "textureFlag");
    //glUniform1i(textureFlagUniform,0); //Default is zero, which is to disable textures

    //Unit vector in the direction of the light (sun)
    lightDirectionUniform = glGetUniformLocation(shaderID, "lightDirection" );
    glUniform3f( lightDirectionUniform, 0, 0, 1 ); //Default is directly above

    //Lighting model used for shading primitives
    lightingModelUniform = glGetUniformLocation(shaderID, "lightingModel" );
    glUniform1i( lightingModelUniform, 0 ); //Default is none

    RboundUniform = glGetUniformLocation(shaderID, "Rbound");
    glUniform1i(RboundUniform,0);

    //initialize default texture in case none are added to the scene
    glBindTexture(GL_TEXTURE_RECTANGLE,0);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0,GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    assert(checkerrors());

}

Shader::~Shader( void ){
    glDeleteVertexArrays( 1, &VertexArrayID);
    glDeleteProgram(shaderID);
}

void Shader::disableTextures() const{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Shader::setTextureMap( const char* texture_file, uint& textureID, int2& texture_size ){

    //Check if the file exists
    std::ifstream f(texture_file);
    assert( f.good() );
    f.close();

    //Check if this texture has already been added to the shader
    //Note: if the texture_file is empy, it will automatically load a new texture
    for( uint i=0; i<textureMapFiles.size(); i++ ){
        if( textureMapFiles.at(i)==texture_file && strlen(texture_file)>0 ){
            textureID = textureMaps.at(i); //copy the handle
            texture_size = textureSizes.at(i); //copy the size
            return;
        }
    }

    //--- Load the Texture ----//

    int Nchannels=3;
    std::vector<unsigned char> texture;
    uint texture_height, texture_width;
    std::string file = texture_file;
    if(file.substr(file.find_last_of('.') + 1) == "jpg" || file.substr(file.find_last_of('.') + 1) == "jpeg" ){
        read_JPEG_file (texture_file,texture,texture_height,texture_width);
    } else if(file.substr(file.find_last_of('.') + 1) == "png") {
        read_png_file (texture_file,texture,texture_height,texture_width);
    }else {
        assert(false);
    }

    texture_size = make_int2( texture_width, texture_height );

    //Find next power of two up from size
    int POT = 0;
    while( pow(2,POT)<texture_size.x ){
        POT++;
    }
    int2 texture_size_POT = make_int2( int(pow(2,POT)), texture_size.y );

    //OpenGL is a pain, so we need to pad the first dimension of the texture so that it size is a power of two
    std::vector<unsigned char> texture_;
    texture_.resize(texture_size_POT.y*texture_size_POT.x*4);
    for( int i=0; i<texture_size_POT.y*texture_size_POT.x*4; i++ ){
        texture_.at(i) = 0;
    }
    for(int j=0;j<texture_size.y;j++){
        for(int i=0;i<texture_size.x;i++){
            for( int c=0; c<4; c++ ){
                texture_[c+i*4+j*texture_size_POT.x*4] = texture[ c + i*4 + j*texture_size.x*4 ];
            }
        }
    }
    for(int j=texture_size.y-1; j<texture_size_POT.y; j++){
        for(int i=texture_size.x-1; i<texture_size_POT.x; i++){
            for( int c=0; c<4; c++ ){
                texture_[c+i*4+j*texture_size_POT.x*4] = 255.f;
            }
        }
    }

    unsigned char* texture_ptr = &texture_[0];

    glGenTextures(1, &textureID);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE,textureID);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0,GL_RGBA, texture_size_POT.x, texture_size_POT.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_ptr);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    textureMaps.push_back(textureID);
    textureSizes.push_back(texture_size);
    textureMapFiles.push_back(texture_file);

}

void Shader::enableTextureMaps() const{
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(textureUniform,0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}


void Shader::setTextureMask( const Glyph* glyph, uint& textureID ){

    int2 texture_size = glyph->size;

    //Check if this texture has already been added to the shader
    for( uint i=0; i<textureMaskFiles.size(); i++ ){
        if( textureMaskFiles.at(i).compare(glyph->filename)==0 ){
            textureID = textureMasks.at(i); //copy the handle
            return;
        }
    }

    //OpenGL is a pain, so we need to pad the first dimension of the texture so that it size is a power of two
    int POT = 0;
    while( pow(2,POT)<texture_size.x ){
        POT++;
    }
    int2 texture_size_POT = make_int2( pow(2,POT), texture_size.y );

    std::vector<unsigned char> texture(texture_size_POT.x*texture_size_POT.y);
    for(int j=0;j<texture_size.y;j++){
        for(int i=0;i<texture_size.x;i++){
            texture[i+j*texture_size_POT.x]=glyph->data.at(j).at(i);
        }
    }
    unsigned char* texture_ptr = &texture[0];

    glGenTextures(1, &textureID);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE,textureID);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0,GL_RED, texture_size_POT.x, texture_size_POT.y, 0, GL_RED, GL_UNSIGNED_BYTE, texture_ptr);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if( strlen(glyph->filename)>0 ){
        textureMasks.push_back(textureID);
        textureMaskFiles.push_back(glyph->filename);
    }

}

void Shader::setTextureMask( const char* texture_file, uint& textureID, int2& texture_size ){

    //Check if this texture has already been added to the shader
    //Note: if the texture_file is empy, it will automatically load a new texture
    for( uint i=0; i<textureMapFiles.size(); i++ ){
        if( textureMapFiles.at(i).compare(texture_file)==0 && strlen(texture_file)>0 ){
            textureID = textureMaps.at(i); //copy the handle
            texture_size = textureSizes.at(i); //copy the size
            return;
        }
    }

    //--- Load the Texture ----//

    int Nchannels=3;
    std::vector<unsigned char> texture;
    uint texture_height, texture_width;
    std::string file = texture_file;
    if(file.substr(file.find_last_of(".") + 1) == "png") {
        read_png_file (texture_file,texture,texture_height,texture_width);
    }else {
        std::cerr << "ERROR: texture mask file " << texture_file << " must be a PNG image." << std::endl;
    }

    texture_size = make_int2( texture_width, texture_height );

    //Find next power of two up from size
    int POT = 0;
    while( pow(2,POT)<texture_size.x ){
        POT++;
    }
    int2 texture_size_POT = make_int2( int(pow(2,POT)), texture_size.y );

    //OpenGL is a pain, so we need to pad the first dimension of the texture so that it size is a power of two
    std::vector<unsigned char> texture_;
    texture_.assign(texture_size_POT.y*texture_size_POT.x,0);
    for(int j=0;j<texture_size.y;j++){
        for(int i=0;i<texture_size.x;i++){
            texture_[j*texture_size_POT.x+i] = texture[ 3 + i*4 + j*texture_size.x*4 ];
        }
    }

    unsigned char* texture_ptr = &texture_[0];

    glGenTextures(1, &textureID);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE,textureID);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0,GL_RED, texture_size_POT.x, texture_size_POT.y, 0, GL_RED, GL_UNSIGNED_BYTE, texture_ptr);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    textureMaps.push_back(textureID);
    textureSizes.push_back(texture_size);
    textureMapFiles.push_back(texture_file);

}

void Shader::enableTextureMasks() const{
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(textureUniform,0);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Shader::setTransformationMatrix(const glm::mat4 &matrix ){
    glUniformMatrix4fv(transformMatrixUniform, 1, GL_FALSE, &matrix[0][0]);
}

void Shader::setDepthBiasMatrix(const glm::mat4 &matrix ){
    glUniformMatrix4fv(depthBiasUniform, 1, GL_FALSE, &matrix[0][0]);
}

void Shader::setLightDirection(const helios::vec3 &direction ){
    glUniform3f( lightDirectionUniform, direction.x, direction.y, direction.z );
}

void Shader::setLightingModel( uint lightingmodel ){
    glUniform1i( lightingModelUniform, lightingmodel );
}

void Shader::useShader() {
    glUseProgram(shaderID);
}

void Visualizer::getViewKeystrokes( vec3& eye, vec3& center ){


    SphericalCoord Spherical = cart2sphere( eye-center );
    float radius = Spherical.radius;
    float theta = Spherical.elevation;
    float phi = Spherical.azimuth;

    GLFWwindow* _window = (GLFWwindow*) window;

    //----- Holding SPACEBAR -----//
    if (glfwGetKey( _window, GLFW_KEY_SPACE ) == GLFW_PRESS){

        // Move center to the left - SPACE + LEFT KEY
        if (glfwGetKey( _window, GLFW_KEY_LEFT ) == GLFW_PRESS){
            center.x +=0.1*sin(phi);
            center.y +=0.1*cos(phi);
        }
            // Move center to the right - SPACE + RIGHT KEY
        else if (glfwGetKey( _window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
            center.x -=0.1*sin(phi);
            center.y -=0.1*cos(phi);
        }
            // Move center upward - SPACE + UP KEY
        else if (glfwGetKey( _window, GLFW_KEY_UP ) == GLFW_PRESS){
            center.z +=0.2;
        }
            // Move center downward - SPACE + DOWN KEY
        else if (glfwGetKey( _window, GLFW_KEY_DOWN ) == GLFW_PRESS){
            center.z -=0.2;
        }

        //----- Not Holding SPACEBAR -----//
    }else{

        //   Orbit left - LEFT ARROW KEY
        if (glfwGetKey( _window, GLFW_KEY_LEFT ) == GLFW_PRESS){
            phi+=M_PI/40.f;
        }
            // Orbit right - RIGHT ARROW KEY
        else if (glfwGetKey( _window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
            phi-=M_PI/40.f;
        }

            // Increase Elevation - UP ARROW KEY
        else if (glfwGetKey( _window, GLFW_KEY_UP ) == GLFW_PRESS){
            if( theta + M_PI/80.f < 0.49f*M_PI ){
                theta+=M_PI/80.f;
            }
        }
            // Decrease Elevation - DOWN ARROW KEY
        else if (glfwGetKey( _window, GLFW_KEY_DOWN ) == GLFW_PRESS){
            if( theta>-0.25*M_PI ){
                theta-=M_PI/80.f;
            }
        }

        //   Zoom in - "+" KEY
        if (glfwGetKey( _window, GLFW_KEY_EQUAL  ) == GLFW_PRESS){
            radius *= 0.9;
        }
            // Zoom out - "-" KEY
        else if (glfwGetKey( _window, GLFW_KEY_MINUS ) == GLFW_PRESS){
            radius *= 1.1;
        }
    }

    if (glfwGetKey( _window, GLFW_KEY_P  ) == GLFW_PRESS){
        std::cout << "View is angle: (R,theta,phi)=(" << radius << "," << theta << "," << phi << ") at from position (" << camera_eye_location.x << "," << camera_eye_location.y << "," << camera_eye_location.z << ") looking at (" << center.x << "," << center.y << "," << center.z << ")" << std::endl;
    }


    camera_eye_location = sphere2cart( make_SphericalCoord(radius,theta,phi) ) + center;


}

std::string errorString( GLenum err ){

    std::string message;
    message.assign("");

    if( err==GL_INVALID_ENUM ){
        message.assign("GL_INVALID_ENUM - An unacceptable value is specified for an enumerated argument.");
    }else if( err==GL_INVALID_VALUE ){
        message.assign("GL_INVALID_VALUE - A numeric argument is out of range.");
    }else if( err==GL_INVALID_OPERATION ){
        message.assign("GL_INVALID_OPERATION - The specified operation is not allowed in the current state.");
    }else if( err==GL_STACK_OVERFLOW ){
        message.assign("GL_STACK_OVERFLOW - This command would cause a stack overflow.");
    }else if( err==GL_STACK_UNDERFLOW ){
        message.assign("GL_STACK_UNDERFLOW - This command would cause a stack underflow.");
    }else if( err==GL_OUT_OF_MEMORY ){
        message.assign("GL_OUT_OF_MEMORY - There is not enough memory left to execute the command.");
    }else if( err==GL_TABLE_TOO_LARGE ){
        message.assign("GL_TABLE_TOO_LARGE - The specified table exceeds the implementation's maximum supported table size.");
    }

    return message;

}

int checkerrors() {

    GLenum err;
    int err_count=0;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "glError #" << err_count << ": " << errorString(err) << std::endl;
        err_count++;
    }
    if(err_count>0){
        return 0;
    }else{
        return 1;
    }

}
