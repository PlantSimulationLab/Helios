/** \file "Visualizer.h" Visualizer header. 
    \author Brian Bailey

    Copyright (C) 2018  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef __VISUALIZER__
#define __VISUALIZER__

#include "Context.h"

//GLM Libraries (math-related functions for graphics)
#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"

//! Function to create a texture map from a JPEG image
/** \param[in] "filename" Name of the JPEG image file
    \param[out] "texture" Texture map created from JPEG image
    \param[out] "height" Height of the image/texture in pixels
    \param[out] "width" Width of the image/texture in pixels
*/
int read_JPEG_file (const char * filename, std::vector<unsigned char> &texture, uint & height, uint & width);

//! Write current graphics window to a JPEG image file
/** \param[in] "filename" Name of the JPEG image file
    \param[in] "width" Width of the graphics window in pixels
    \param[in] "height" Height of the graphics window in pixels
    \param[in] "window" Pointer to the window object
*/
int write_JPEG_file ( const char* filename, uint width, uint height, void* _window );

//! Function to create a texture map from a PNG image
/** \param[in] "filename" Name of the PNG image file
    \param[out] "texture" Texture map created from PNG image
    \param[out] "height" Height of the image/texture in pixels
    \param[out] "width" Width of the image/texture in pixels
*/
void read_png_file( const char* filename, std::vector<unsigned char> &texture, uint & height, uint & width);

//! Glyph object - 2D matrix shape
class Glyph{
 public:
    
  helios::int2 size;
  std::vector<std::vector<uint> > data;
  float solidFraction;
  char* filename;
    
  void readFile( const char* glyphfile );
  
};

//! OpenGL Shader data structure
struct Shader{
public:
    
  //! Disable texture maps and color fragments by interpolating vertex colors
  void disableTextures( void ) const;
  
  //! Color fragments using an RGB texture map
  /** \param[in] "texture" Handle to a texture map */
  void setTextureMap( const char* texture_file, uint& textureID, helios::int2& texture_size );

  //! Enable texture maps and color fragments using an RGB texture map
  void enableTextureMaps( void ) const;
  
  //! Set fragment opacity using a Glyph (red channel)
  /** \param[in] "texture" Handle to a texture map */
  void setTextureMask( const Glyph* glyph, uint& textureID );

  //! Set fragment opacity using a texture map file (red channel)
  /** \param[in] "texture" Handle to a texture map */
  void setTextureMask( const char* texture_file, uint& textureID, helios::int2& texture_size );
  
  //! Enable texture masks and color fragments by interpolating vertex colors
  void enableTextureMasks( void ) const;

  //! Set the shader transformation matrix, i.e., the Affine transformation applied to all vertices
  void setTransformationMatrix( const glm::mat4 matrix );

  //! Set the depth bias matrix for shadows
  void setDepthBiasMatrix( const glm::mat4 matrix );

  //! Set the direction of the light (sun)
  void setLightDirection( const helios::vec3 direction );

  //! Set the lighting model
  void setLightingModel( const uint lightingmodel );

  //! Set shader as current
  void useShader(void);
  
  //! Initialize the shader
  /** \param[in] "vertex_shader_file" Name of vertex shader file to be used by OpenGL in rendering graphics
      \param[in] "fragment_shader_file" Name of fragment shader file to be used by OpenGL in rendering graphics
  */
  void initialize( const char* vertex_shader_file, const char* fragment_shader_file );

  ~Shader();
  
  //Primary Shader
  uint shaderID;  
  uint textureUniform;
  uint shadowmapUniform;
  uint transformMatrixUniform;
  uint depthBiasUniform;
  uint lightDirectionUniform;
  uint lightingModelUniform;
  uint RboundUniform;
  uint VertexArrayID;
  
private:

  std::vector<uint> textureMaps;
  std::vector<helios::int2> textureSizes;
  std::vector<std::string> textureMapFiles;
  std::vector<uint> textureMasks;
  std::vector<std::string> textureMaskFiles;
  
};

//! RGB color map
struct Colormap{
public:

  Colormap( void ){};

  Colormap( std::vector<helios::RGBcolor> ctable, std::vector<float> clocs, int size, float minval_, float maxval_ ){

    set( ctable, clocs, size, minval_, maxval_ );

  }

  void set( std::vector<helios::RGBcolor> ctable, std::vector<float> clocs, int size, float minval_, float maxval_ ){

    cmapsize = size;
    minval=minval_;
    maxval=maxval_;
    
    int Ncolors=ctable.size();
    
    if( clocs.size() != Ncolors ){
      std::cerr << "ERROR: colormap - sizes of ctable and clocs must match." << std::endl;
      exit(EXIT_FAILURE);
    }
    
    cmap.resize(Ncolors);
    
    float cmin,cmax;
    std::vector<float> cinds;
    cinds.resize(Ncolors);
    
    for( int i=0; i<Ncolors; i++ ){
      cinds.at(i)=clocs.at(i)*(cmapsize-1);
    }
    
    cmap.resize(cmapsize);
    for( int c=0; c<Ncolors-1; c++ ){
      
      cmin=cinds.at(c);
      cmax=cinds.at(c+1);
      
      for( int i=0; i<cmapsize; i++){
	
	if( i>=cmin && i<=cmax ){
	  
	  cmap.at(i).r=ctable.at(c).r+(float(i)-cmin)/(cmax-cmin)*(ctable.at(c+1).r-ctable.at(c).r);
	  cmap.at(i).g=ctable.at(c).g+(float(i)-cmin)/(cmax-cmin)*(ctable.at(c+1).g-ctable.at(c).g);
	  cmap.at(i).b=ctable.at(c).b+(float(i)-cmin)/(cmax-cmin)*(ctable.at(c+1).b-ctable.at(c).b);
	  
	}
      }
    }

  }

  helios::RGBcolor query( float x ) const{

    if( cmapsize==0 || cmap.size()==0 ){
      std::cerr << "ERROR: colormap was not initialized properly." << std::endl;
      std::cout << cmapsize << " " << cmap.size() << std::endl;
      exit(EXIT_FAILURE);
    }
    
    helios::RGBcolor color;

    int color_ind;
    if( minval==maxval ){
      color_ind=0;
    }else{
      color_ind=round( (x-minval)/(maxval-minval) * (cmapsize-1) );
    }

    if(color_ind<0){color_ind=0;}
    if(color_ind>cmapsize-1){color_ind=cmapsize-1;}
    color.r=cmap.at(color_ind).r;
    color.g=cmap.at(color_ind).g;
    color.b=cmap.at(color_ind).b;
    
    return color;
    
  }

  void setRange( float min, float max ){
    minval = min;
    maxval = max;
  }

  helios::vec2 getRange( void ) const{
    return helios::make_vec2( minval, maxval );
  }

  float getLowerLimit( void ) const{
    return minval;
  }

  float getUpperLimit( void ) const{
    return maxval;
  }

private:

  std::vector<helios::RGBcolor> cmap;
  unsigned int cmapsize;
  float minval, maxval;

};

//! Class for visualization of simulation results
class Visualizer{
public:

  //! Visualizer constructor
  /** 
      \param[in] "Wdisplay" Width of the display window in pixels, and assumes default window aspect ratio of 1.25
  */
  Visualizer( uint Wdisplay );

  //! Visualizer constructor
  /** 
      \param[in] "Wdisplay" Width of the display window in pixels
      \param[in] "Hdisplay" Height of the display window in pixels
  */
  Visualizer( uint Wdisplay, uint Hdisplay );

  Visualizer( uint Wdisplay, uint Hdisplay, int aliasing_samples );

  // !Visualizer destructor
  ~Visualizer(void);

  //! Visualizer self-test routine
  int selfTest( void );

  /* //! Type of transformation applied to a geometric object */
  /* enum TransformationMethod {  */
  /*   //! Do not apply any coordinate transformation to positions. In this case, all positions must be between -1 and 1. */
  /*   NO_TRANSFORM=0,  */

  /*   //! Apply a perspective transformation to positions according to the current view. To set the view, use \ref setCameraPosition(). */
  /*   PERSPECTIVE_TRANSFORM=1,  */

  /*   //! Apply a custom transformation to all positions. */
  /*   CUSTOM_TRANSFORM=2  */
  /* }; */
  //! Coordinate system to be used when specifying spatial coordinates
  enum CoordinateSystem{

    //! Coordinates are normalized to unity and are window-aligned.  The point (x,y)=(0,0) is in the bottom left corner of the window, and (x,y)=(1,1) is in the upper right corner of the window.  The z-coordinate specifies the depth in the screen-normal direction, with values ranging from -1 to 1.  For example, an object at z=0.5 would be in front of an object at z=0.
    COORDINATES_WINDOW_NORMALIZED = 0,

    //! Coordinates are specified in a 3D Cartesian system (right-handed), where +z is vertical.
    COORDINATES_CARTESIAN = 1

  };

  //! Pseudocolor map tables
  enum Ctable{
    //! ``Hot" colormap
    COLORMAP_HOT = 0,
    //! ``Cool" colormap
    COLORMAP_COOL = 1,
    //! ``Rainbow" colormap
    COLORMAP_RAINBOW = 2,
    //! ``Lava" colormap
    COLORMAP_LAVA = 3,
    //! ``Parula" colormap
    COLORMAP_PARULA = 4,
    //! ``Gray" colormap
    COLORMAP_GRAY = 5,
    //! Custom colormap
    COLORMAP_CUSTOM = 6
  };

  //! Set camera position
  /** \param[in] "cameraPosition" (x,y,z) position of the camera, i.e., this is where the actual camera or `eye' is positioned.
      \param[in] "lookAt" (x,y,z) position of where the camera is looking at.
  */
  void setCameraPosition( helios::vec3 cameraPosition, helios::vec3 lookAt );

  //! Set camera position
  /** \param[in] "cameraAngle" (elevation,azimuth) angle to the camera with respect to the `lookAt' position.
      \param[in] "cameraRadius" Distance from the camera to the `lookAt' position.
      \param[in] "lookAt" (x,y,z) position of where the camera is looking at.
  */
  void setCameraPosition( helios::SphericalCoord cameraAngle, helios::vec3 lookAt );

  //! Set the camera field of view (angle width) in degrees. Default value is 45 degrees.
  /** \param[in] "angle_FOV" Angle of camera field of view in degrees.
   */
  void setCameraFieldOfView( const float angle_FOV );

  //! Set the direction of the light source
  /** \param[in] "direction" Vector pointing in the direction of the light source (vector starts at light source and points toward scene.) */
  void setLightDirection( helios::vec3 direction );

  //! Get a box that bounds all primitives in the domain
  void getDomainBoundingBox( helios::vec2& xbounds, helios::vec2& ybounds, helios::vec2& zbounds ) const;

  //! Get the radius of a sphere that bounds all primitives in the domain
  float getDomainBoundingRadius( void ) const;

  //! Lighting model to use for shading primitives
  enum LightingModel { 
    //! No shading, primitive is colored by its diffuse color
    LIGHTING_NONE=0, 

    //! Phong lighting model is applied to add shading effects to the diffuse color
    LIGHTING_PHONG=1, 
 
    //! Phong lighting model plus shadowing is applied to add shading effects to the diffuse color
    LIGHTING_PHONG_SHADOWED=2 
  };

  //! Set the lighting model for shading of all primitives
  /** \param[in] "lightingmodel" Lighting model to be used
      \sa \ref LightingModel 
  */
  void setLightingModel( LightingModel lightingmodel );

  //! Set the background color for the visualizer window
  /** \param[in] "color" Background color
   */
  void setBackgroundColor( helios::RGBcolor color );

  //! Add a rectangle by giving the coordinates of its center
  /** \param[in] "center" (x,y,z) location of the rectangle center
      \param[in] "size" Size in the x- and y-directions
      \param[in] "rotation" spherical rotation angle (elevation,azimuth)
      \param[in] "color" R-G-B color of the rectangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByCenter( const helios::vec3 center, const helios::vec2 size, const helios::SphericalCoord rotation, const helios::RGBcolor color, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its center
  /** \param[in] "center" (x,y,z) location of the rectangle center
      \param[in] "size" Size in the x- and y-directions
      \param[in] "rotation" spherical rotation angle (elevation,azimuth)
      \param[in] "color" R-G-B-A color of the rectangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByCenter( const helios::vec3 center, const helios::vec2 size, const helios::SphericalCoord rotation, const helios::RGBAcolor color, const CoordinateSystem coordFlag );

  //! Add a texture mapped rectangle by giving the coordinates of its center
  /** \param[in] "center" (x,y,z) location of the rectangle center
      \param[in] "size" Size in the x- and y-directions
      \param[in] "rotation" spherical rotation angle (elevation,azimuth)
      \param[in] "texture_file" File corresponding to the JPEG image to be used as a texture map
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByCenter( const helios::vec3 center, const helios::vec2 size, const helios::SphericalCoord rotation, const char* texture_file, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its center - rectangle is colored by and RGB color value but is masked by the alpha channel of a PNG image file
  /** \param[in] "center" (x,y,z) location of the rectangle center
      \param[in] "size" Size in the x- and y-directions
      \param[in] "rotation" spherical rotation angle (elevation,azimuth)
      \param[in] "color" R-G-B color of the rectangle
      \param[in] "texture_file" File corresponding to the JPEG image to be used as a texture map
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByCenter( const helios::vec3 center, const helios::vec2 size, const helios::SphericalCoord rotation, const helios::RGBcolor color, const char* texture_file, const CoordinateSystem coordFlag );

  //! Add a texture masked rectangle by giving the coordinates of its center
  /** \param[in] "center" (x,y,z) location of the rectangle center
      \param[in] "size" Size in the x- and y-directions
      \param[in] "rotation" spherical rotation angle (elevation,azimuth)
      \param[in] "color" R-G-B color of the rectangle
      \param[in] "glyph" Pixel map of true/false values for a transparency mask
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByCenter( const helios::vec3 center, const helios::vec2 size, const helios::SphericalCoord rotation, const helios::RGBcolor color, Glyph* glyph, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its four vertices
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "color" R-G-B color of the rectangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const helios::RGBcolor color, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its four vertices
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "color" R-G-B-A color of the rectangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const helios::RGBAcolor color, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its four vertices
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "texture_file" File corresponding to the JPEG image to be used as a texture map
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const char* texture_file, const CoordinateSystem coordFlag );
  
  //! Add a rectangle by giving the coordinates of its four vertices and color by texture map
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "texture_file" File corresponding to the JPEG image to be used as a texture map
      \param[in] "uvs" u-v coordinates for rectangle vertices
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const char* texture_file, const std::vector<helios::vec2> uvs, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its four vertices and mask by texture map transparency channel, but color by R-G-B value
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "texture_file" File corresponding to the JPEG image to be used as a texture map
      \param[in] "uvs" u-v coordinates for rectangle vertices
      \param[in] "color" R-G-B color of the rectangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const helios::RGBcolor color, const char* texture_file, const std::vector<helios::vec2> uvs, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its four vertices - rectangle is colored by an RGB color value but is masked by the alpha channel of a PNG image file
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "color" R-G-B color of the rectangle
      \param[in] "texture_file" File corresponding to the JPEG image to be used as a texture map
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const helios::RGBcolor color, const char* texture_file, const CoordinateSystem coordFlag);
  
  //! Add a rectangle by giving the coordinates of its four vertices
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "color" R-G-B color of the glyph
      \param[in] "glyph" Glyph object used to render rectangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const helios::RGBcolor color, const Glyph* glyph, const CoordinateSystem coordFlag );

  //! Add a rectangle by giving the coordinates of its four vertices
  /** \param[in] "vertices" (x,y,z) coordinates of four vertices
      \param[in] "color" R-G-B-A color of the glyph
      \param[in] "glyph" Glyph object used to render rectangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `vertices' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addRectangleByVertices( const std::vector<helios::vec3>& vertices, const helios::RGBAcolor color, const Glyph* glyph, const CoordinateSystem coordFlag );
  
  //! Add a triangle by giving the coordinates of its three vertices
  /** \param[in] "vertex0" (x,y,z) location of first vertex
      \param[in] "vertex1" (x,y,z) location of first vertex
      \param[in] "vertex2" (x,y,z) location of first vertex
      \param[in] "color" R-G-B color of the triangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the triangle, and physical dimensions are used.
  */
  void addTriangle( helios::vec3 vertex0, helios::vec3 vertex1, helios::vec3 vertex2, helios::RGBcolor color, CoordinateSystem coordFlag );

  //! Add a triangle by giving the coordinates of its three vertices
  /** \param[in] "vertex0" (x,y,z) location of first vertex
      \param[in] "vertex1" (x,y,z) location of first vertex
      \param[in] "vertex2" (x,y,z) location of first vertex
      \param[in] "color" R-G-B-A color of the triangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the triangle, and physical dimensions are used.
  */
  void addTriangle( helios::vec3 vertex0, helios::vec3 vertex1, helios::vec3 vertex2, helios::RGBAcolor color, CoordinateSystem coordFlag );

  //! Add a triangle by giving the coordinates of its three vertices and color by texture map
  /** \param[in] "vertex0" (x,y,z) location of first vertex
      \param[in] "vertex1" (x,y,z) location of first vertex
      \param[in] "vertex2" (x,y,z) location of first vertex
      \param[in] "texture_file" File corresponding to the image to be used as a texture map
      \param[in] "uv0" u-v texture coordinates of vertex0
      \param[in] "uv1" u-v texture coordinates of vertex1
      \param[in] "uv2" u-v texture coordinates of vertex2
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the triangle, and physical dimensions are used.
  */
  void addTriangle( helios::vec3 vertex0, helios::vec3 vertex1, helios::vec3 vertex2, const char* texture_file, helios::vec2 uv0, helios::vec2 uv1, helios::vec2 uv2, CoordinateSystem coordFlag );

  //! Add a triangle by giving the coordinates of its three vertices and color by a constant color, but mask using transparency channel of texture map
  /** \param[in] "vertex0" (x,y,z) location of first vertex
      \param[in] "vertex1" (x,y,z) location of first vertex
      \param[in] "vertex2" (x,y,z) location of first vertex
      \param[in] "texture_file" File corresponding to the image to be used as a texture map
      \param[in] "uv0" u-v texture coordinates of vertex0
      \param[in] "uv1" u-v texture coordinates of vertex1
      \param[in] "uv2" u-v texture coordinates of vertex2
      \param[in] "color" R-G-B-A color of the triangle
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the triangle, and physical dimensions are used.
  */
  void addTriangle( helios::vec3 vertex0, helios::vec3 vertex1, helios::vec3 vertex2, const char* texture_file, helios::vec2 uv0, helios::vec2 uv1, helios::vec2 uv2, helios::RGBAcolor color, CoordinateSystem coordFlag );

  //! Add a voxel by giving the coordinates of its center
  /** \param[in] "size" Size in the x-, y- and z-directions
      \param[in] "center" (x,y,z) location of the voxel center
      \param[in] "color" R-G-B color of the voxel
  */
  void addVoxelByCenter( const helios::vec3 center, const helios::vec3 size, const helios::SphericalCoord rotation, const helios::RGBcolor color, const CoordinateSystem coordFlag );

  //! Add a voxel by giving the coordinates of its center
  /** \param[in] "size" Size in the x-, y- and z-directions
      \param[in] "center" (x,y,z) location of the voxel center
      \param[in] "color" R-G-B-A color of the voxel
      \param[in] "coordFlag" If flag value is false, no transform is applied to the voxel.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the voxel, and physical dimensions are used.
  */
  void addVoxelByCenter( const helios::vec3 center, const helios::vec3 size, const helios::SphericalCoord rotation, const helios::RGBAcolor color, const CoordinateSystem coordFlag );

  //! Add a disk by giving the coordinates of its center
  /** \param[in] "center" (x,y,z) location of the disk center
      \param[in] "size" length of disk semi-major and semi-minor axes
      \param[in] "Ndivisions" Number of discrete divisions in making disk. (e.g., Ndivisions=4 makes a square, Ndivisions=5 makes a pentagon, etc.)
      \param[in] "color" R-G-B color of the disk
      \param[in] "coordFlag" If flag value is false, no transform is applied to the disk.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the disk, and physical dimensions are used.
  */
  void addDiskByCenter( helios::vec3 center, helios::vec2 size, helios::SphericalCoord rotation, uint Ndivisions, helios::RGBcolor color, CoordinateSystem coordFlag );

  //! Add a disk by giving the coordinates of its center
  /** \param[in] "center" (x,y,z) location of the disk center
      \param[in] "size" length of disk semi-major and semi-minor axes
      \param[in] "Ndivisions" Number of discrete divisions in making disk. (e.g., Ndivisions=4 makes a square, Ndivisions=5 makes a pentagon, etc.)
      \param[in] "color" R-G-B-A color of the disk
      \param[in] "coordFlag" If flag value is false, no transform is applied to the disk.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the disk, and physical dimensions are used.
  */
  void addDiskByCenter( helios::vec3 center, helios::vec2 size, helios::SphericalCoord rotation, uint Ndivisions, helios::RGBAcolor color, CoordinateSystem coordFlag );

  //! Add a texture mapped disk by giving the coordinates of its center
  /** \param[in] "center" (x,y,z) location of the disk center
      \param[in] "size" length of disk semi-major and semi-minor axes
      \param[in] "Ndivisions" Number of discrete divisions in making disk. (e.g., Ndivisions=4 makes a square, Ndivisions=5 makes a pentagon, etc.)
      \param[in] "texture_file" File corresponding to the JPEG image to be used as a texture map
      \param[in] "coordFlag" If flag value is false, no transform is applied to the disk.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the disk, and physical dimensions are used.
  */
  void addDiskByCenter( helios::vec3 center, helios::vec2 size, helios::SphericalCoord rotation, uint Ndivisions, const char* texture_file, CoordinateSystem coordFlag );

  //! Add Lines by giving the coordinates of points along the Lines
  /** 
      \param[in] "start" (x,y,z) coordinates of line starting position
      \param[in] "end" (x,y,z) coordinates of line ending position
      \param[in] "color" R-G-B color of the line
      \param[in] "linewidth" Width of the line in points
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the lines, and physical dimensions are used.
  */
  void addLine( const helios::vec3 start, const helios::vec3 end, const helios::RGBcolor color, const uint linewidth, const CoordinateSystem coordFlag );

  //! Add Lines by giving the coordinates of points along the Lines
  /** 
      \param[in] "start" (x,y,z) coordinates of line starting position
      \param[in] "end" (x,y,z) coordinates of line ending position
      \param[in] "color" R-G-B-A color of the line
      \param[in] "linewidth" Width of the line in points
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the lines, and physical dimensions are used.
  */
  void addLine( const helios::vec3 start, const helios::vec3 end, const helios::RGBAcolor color, const uint linewidth, const CoordinateSystem coordFlag );

  //! Add a point by giving its coordinates and size
  /** 
      \param[in] "position" (x,y,z) coordinates of Point
      \param[in] "color" R-G-B color of the Point
      \param[in] "size" Size of the point in font points
      \param[in]  "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the lines, and physical dimensions are used.
  */
  void addPoint( const helios::vec3 position, const helios::RGBcolor color, const uint pointsize,  const CoordinateSystem coordFlag);

  //! Add a point by giving its coordinates and size
  /** 
      \param[in] "position" (x,y,z) coordinates of Point
      \param[in] "color" R-G-B-A color of the Point
      \param[in] "size" Size of the point in font points
      \param[in]  "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the lines, and physical dimensions are used.
  */
  void addPoint( const helios::vec3 position, const helios::RGBAcolor color, const uint pointsize,  const CoordinateSystem coordFlag );
  
  //! Add a sphere by giving the radius and center
  /** 
      \param[in] "radius" Radius of the sphere
      \param[in] "center" (x,y,z) location of sphere center
      \param[in] "Ndivisions" Number of discrete divisions in making sphere
      \param[in] "color" R-G-B color of the sphere
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the sphere, and physical dimensions are used.
  */
  void addSphereByCenter( const float radius, const helios::vec3 center, const uint Ndivisions, const helios::RGBcolor color, const CoordinateSystem coordFlag );

  //! Add a sphere by giving the radius and center
  /** 
      \param[in] "radius" Radius of the sphere
      \param[in] "center" (x,y,z) location of sphere center
      \param[in] "Ndivisions" Number of discrete divisions in making sphere
      \param[in] "color" R-G-B-A color of the sphere
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the sphere, and physical dimensions are used.
  */
  void addSphereByCenter( const float radius, const helios::vec3 center, const uint Ndivisions, const helios::RGBAcolor color, const CoordinateSystem coordFlag );

  //! Add a Sky Dome, which is a hemispherical dome colored by a sky texture map
  /** 
      \param[in] "radius" Radius of the dome
      \param[in] "center" (x,y,z) location of dome center
      \param[in] "Ndivisions" Number of discrete divisions in making hemisphere
      \param[in] "texture_file" Name of the texture map file
  */
  void addSkyDomeByCenter( const float radius, const helios::vec3 center, const uint Ndivisions, const char* texture_file );

  //! Add a Sky Dome, which is a hemispherical dome colored by a sky texture map
  /** \note This function has been deprecated, as layers are no longer supported. */
  DEPRECATED( void addSkyDomeByCenter( const float radius, const helios::vec3 center, const uint Ndivisions, const char* texture_file, int layer ) );

  //! Add a text box by giving the coordinates of its center
  /** \param[in] "textstring" String of text to display
      \param[in] "center" (x,y,z) location of the text box center
      \param[in] "rotation" Spherical rotation angle in radians (elevation,azimuth)
      \param[in] "fontcolor" Color of the font
      \param[in] "fontsize" Size of the text font in points
      \param[in] "coordFlag" If flag value is false, no transform is applied to the rectangle.  In this case `size' and `center' values are normalized window coordinates, where the size of the window is 1x1 and (x=0,y=0) is the bottom left of the window. If flag value is true, a perspective transform is applied to the rectangle, and physical dimensions are used.
  */
  void addTextboxByCenter( const char* textstring, const helios::vec3 center, helios::SphericalCoord rotation, const helios::RGBcolor fontcolor, const uint fontsize, const char* fontname, CoordinateSystem coordFlag );

  //! Enable the colorbar
  void enableColorbar( void );

  //! Disable the colorbar
  void disableColorbar( void );

  //! Set the position of the colorbar in normalized window coordinates (0-1)
  /** \param[in] "position" Position of the colorbar in normalized window coordinates
  */
  void setColorbarPosition( helios::vec3 position );

  //! Set the size of the colorbar in normalized window units (0-1)
  /** \param[in] "size" Size of the colorbar in normalized window units (0-1)
  */
  void setColorbarSize( helios::vec2 size );

  //! Set the range of the Colorbar
  /** \param[in] "cmin" Minimum value
      \param[out] "cmax" Maximum value
  */
  void setColorbarRange( float cmin, float cmax );

  //! Set the values in the colorbar where ticks and labels should be placed
  /** \param[in] "ticks" Vector of values corresponding to ticks
      \note If tick values are outside of the colorbar range (see \ref setColorBarRange()), the colorbar will be automatically expanded to fit the tick values.
  */
  void setColorbarTicks( std::vector<float> ticks );

  //! Set the title of the Colorbar
  /** \param[in] "title" Colorbar title 
  */
  void setColorbarTitle( const char* title );

  //! Set the RGB color of the colorbar text
  /** \param[in] "color" Font color
  */
  void setColorbarFontColor( helios::RGBcolor color );

  //! Set the font size of the colorbar text
  /** \param[in] "font_size" Font size
  */
  void setColorbarFontSize( uint font_size );

  //! Set the colormap used in Colorbar/visualization
  /** \param[in] "colormap_name" Name of a colormap. Valid colormaps are "hot" and "lava". 
  */
  void setColormap( Ctable colormap_name );

  //! Set the colormap used in Colorbar/visualization
  /** \param[in] "colormap_name" Name of a colormap. Valid colormaps are "hot" and "lava". 
  */
  void setColormap( Ctable colormap_name, std::vector<helios::RGBcolor> colors, std::vector<float> divisions );

  //! Get the current colormap used in Colorbar/visualization
  Colormap getCurrentColormap( void ) const;
  
  //! Add all geometry from the \ref Context to the visualizer
  /** \param[in] "context" Pointer to the simulation context */
  void buildContextGeometry( helios::Context* context );

  //! Add select geometry from the \ref Context to the visualizer by their UUIDs
  /** \param[in] "context" Pointer to the simulation context 
      \param[in] "UUIDs" UUIDs of Context primitives to be added to the visualizer
  */
  void buildContextGeometry( helios::Context* context, const std::vector<uint>& UUIDs );

  //! Color primitives from Context by color mapping their `Primitive Data'
  /** \param[in] "data_name" Name of `Primitive Data'
      \note If the data value does not exist for a certain primitive, a value of 0 is assumed.
  */
  void colorContextPrimitivesByData( const char* data_name );

  //! Color primitives from Context by color mapping their `Primitive Data'
  /** \param[in] "data_name" Name of `Primitive Data'
      \param[in] "UUIDs" UUID's of primitives to be colored by data
      \note If the data value does not exist for a certain primitive, a value of 0 is assumed.
  */
  void colorContextPrimitivesByData( const char* data_name, const std::vector<uint>& UUIDs );

  //! Make Helios logo watermark invisible
  void hideWatermark( void );

  //! Make Helios logo watermark visible
  void showWatermark( void );

  //! Set whether the dashboard should be visible in the graphics window
  /** \param[in] "visible" `true' if visible, `false' if not visible
  */
  void setDashboardVisibility( const bool visible );

  //! Plot current geometry into an interactive graphics window
  void plotInteractive(void);

  //! Plot the depth map (distance from camera to nearest object)
  void plotDepthMap(void);

  //! Update the graphics window based on current geometry, then continue the program
  void plotUpdate(void);

  //! Print the current graphics window to a JPEG image file. File will be given a default filename and saved to the current directory from which the executable was run.
  void printWindow( void );
  
  //! Print the current graphics window to a JPEG image file
  /** \param[in] "outfile" Path to file where image should be saved.
      \note If outfile does not have extension `.jpg', it will be appended to the file name.
  */
  void printWindow( const char* outfile );

  //! Get R-G-B pixel data in the current display window
  /** \param[out] "buffer" Pixel data. The data is stored as r-g-b * column * row. So indices (0,1,2) would be the RGB values for row 0 and column 0, indices (3,4,5) would be RGB values for row 0 and column 1, and so on. Thus, buffer is of size 3*width*height.
   */
  void getWindowPixelsRGB( unsigned int * buffer );

  //! Get depth buffer data for the current display window
  /** \param[out] "buffer" Distance to nearest object from the camera location.
      \note The function \ref plotDepthMap() must be called prior to \ref getDepthMap().
  */
  void getDepthMap( float * buffer );

  //! Get the size of the display window in pixels
  /** \param[out] "width" Width of the display window in pixels
      \param[out] "height" Height of the display window in pixels
  */
  void getWindowSize( uint &width, uint &height );

  //! Clear all geometry previously added to the visualizer
  void clearGeometry( void );

  //! Close the graphics window
  void closeWindow( void );

private:

  void initialize( uint Wdisplay, uint Hdisplay, int aliasing_samples );

  void render( bool shadow );

  void setupPlot( void );
  
  //~~~~~~~~~~~~~~~~ Primitives ~~~~~~~~~~~~~~~~~~~~//

  std::string colorPrimitivesByVariable,colorPrimitivesByData,colorPrimitivesByValue;
  std::map<uint,uint> colorPrimitives_UUIDs;

  std::vector<int> contextPrimitiveIDs;

  std::vector<float> depth_buffer_data;

  std::map<uint,helios::int2> group_start;

  void getViewKeystrokes( helios::vec3& eye, helios::vec3& center );

  //! Add a Colorbar given its center position
  void addColorbarByCenter( const char* title, const helios::vec2 size, const helios::vec3 center, const helios::RGBcolor font_color, const Colormap colormap );

  //! Width of the display window in pixels
  uint Wdisplay;
  //! Height of the display window in pixels
  uint Hdisplay;

  uint frame_counter;

  //! Handle to the GUI window
  /** \note This will be recast to have type GLFWwindow*.  This has to be done in order to keep library-dependent variables out of the header. */
  void* window;

  //! (x,y,z) coordinates of location where the camera is looking
  helios::vec3 center;

  //! (x,y,z) coordinates of the camera (a.k.a. the `eye' location)
  helios::vec3 eye;

  //! Handle to the OpenGL shader (primary)
  Shader primaryShader;

  //! Handle to the OpenGL shader (depth buffer for shadows)
  Shader depthShader;

  Shader* currentShader;

  uint framebufferID;
  uint depthTexture;

  //! Lighting model for Context object primitives (default is LIGHTING_NONE)
  std::vector<LightingModel> primaryLightingModel;

  bool isWatermarkVisible;

  //! Color of the window background
  helios::RGBcolor backgroundColor;

  //! Vector pointing from the light source to the scene
  helios::vec3 light_direction;

  //! Flag indicating whether colorbar is enabled
  bool colorbar_flag;

  //! Title of the colorbar
  std::string colorbar_title;

  //! Fontsize of colorbar text
  uint colorbar_fontsize;

  //! Width of points (if applicable) in pixels
  uint point_width;

  //! Color of colorbar text
  helios::RGBcolor colorbar_fontcolor;

  //! Position of colorbar center in normalized window coordinates
  helios::vec3 colorbar_position;

  //! x- and y- dimensions of colorbar in normalized window coordinates
  helios::vec2 colorbar_size;

  uint positionBuffer, colorBuffer, normalBuffer, uvBuffer, textureFlagBuffer, coordinateFlagBuffer;
  std::map<std::string,std::vector<float> > positionData, colorData, normalData, uvData;
  std::map<std::string,std::vector<int> > coordinateFlagData, textureFlagData, textureIDData;

  void updatePerspectiveTransformation( const helios::vec3 center, const helios::vec3 eye );

  glm::mat4 perspectiveTransformationMatrix;

  void updateCustomTransformation( const glm::mat4 matrix );

  glm::mat4 customTransformationMatrix;

  //!Field of view of the camera in degrees
  float camera_FOV;

  bool contextGeomNeedsUpdate;

  bool primitiveColorsNeedUpdate;

  helios::Context* context;

  //! Function to actually update Context geometry (if needed), which is called by the visualizer before plotting
  void buildContextGeometry_private(void);

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


};

int checkerrors( void );


#endif
