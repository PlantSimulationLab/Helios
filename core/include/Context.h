/** \file "Context.h" Context header file. 
    \author Brian Bailey

    Copyright (C) 2018  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef __CONTEXT__
#define __CONTEXT__

//pugi XML parser
#include "pugixml.hpp"

#include "global.h"

//! Timeseries-related functions
/** \defgroup timeseries Timeseries */

namespace helios {

  /// Type of primitive element
  enum PrimitiveType{
    /// < Rectangular primitive
    PRIMITIVE_TYPE_PATCH = 0, 
    /// < Triangular primitive
    /** \ref Triangle */
    PRIMITIVE_TYPE_TRIANGLE = 1,
    //! Rectangular prism primitive
    /** \ref Voxel */
    PRIMITIVE_TYPE_VOXEL = 2
  };

  //! Data types
  enum HeliosDataType{
    //! integer data type
    HELIOS_TYPE_INT = 0,
    //! unsigned integer data type
    HELIOS_TYPE_UINT = 1,
    //! floating point data taype
    HELIOS_TYPE_FLOAT = 2,
    //! double data type
    HELIOS_TYPE_DOUBLE = 3,
    //! \ref helios::vec2 data type
    HELIOS_TYPE_VEC2 = 4,
    //! \ref helios::vec3 data type
    HELIOS_TYPE_VEC3 = 5,
    //! \ref helios::vec4 data type
    HELIOS_TYPE_VEC4 = 6,
    //! \ref helios::int2 data type
    HELIOS_TYPE_INT2 = 7,
    //! \ref helios::int3 data type
    HELIOS_TYPE_INT3 = 8,
    //! \ref helios::int4 data type
    HELIOS_TYPE_INT4 = 9,
    //! string data type
    HELIOS_TYPE_STRING = 10,
  };

  class Texture{
   public:

    Texture( void ){};

    //! Constructor - initialize with given texture file
    /** \param[in] "texture_file" Path to texture file. Note that file paths can be absolute, or relative to the directory in which the program is being run.
    */
    Texture( const char* texture_file );

    std::string getTextureFile( void ) const;

    helios::int2 getSize( void ) const;

    bool hasTransparencyChannel( void ) const;

    std::vector<std::vector<bool> >* getTransparencyData( void );

    float getSolidFraction( void ) const;
    
  private:
    std::string filename;
    bool hastransparencychannel;
    std::vector<std::vector<bool> > transparencydata;
    float solidfraction;
  };

  //! Structure for Global Data Entities
  struct GlobalData{
    
    std::vector<int> global_data_int;
    std::vector<uint> global_data_uint;
    std::vector<float> global_data_float;
    std::vector<double> global_data_double;
    std::vector<vec2> global_data_vec2;
    std::vector<vec3> global_data_vec3;
    std::vector<vec4> global_data_vec4;
    std::vector<int2> global_data_int2;
    std::vector<int3> global_data_int3;
    std::vector<int4> global_data_int4;
    std::vector<std::string> global_data_string;
    std::vector<bool> global_data_bool;

    size_t size;
    HeliosDataType type;

  };

  //! Primitive class
  /** All primitive objects inherit this class, and it provides functionality universal to all primitives.  There may be additional functionality associated with specific primitive elements. 
      \ingroup primitives 
  */
  class Primitive{
  public:

    //! Virtual destructor
    virtual ~Primitive(){};

    //! Function to get the Primitive universal unique identifier (UUID)
    /** \return UUID of primitive */
    uint getUUID(void) const;
  
    //! Function to get the Primitive type
    /** \sa \ref PrimitiveType */
    PrimitiveType getType(void) const;
    
    //! Function to return the surface area of a Primitive
    virtual float getArea() const = 0;
    
    //! Function to return the normal vector of a Primitive
    virtual helios::vec3 getNormal() const = 0;
    
    //! Function to return the Affine transformation matrix of a Primitive
    /** \param[out] "T" 1D vector corresponding to Primitive transformation matrix */
    void getTransformationMatrix( float (&T)[16] ) const;

    void setTransformationMatrix( float (&T)[16] );
    
    //! Function to return the (x,y,z) coordinates of the vertices of a Primitve
    virtual std::vector<helios::vec3> getVertices( void ) const = 0;
    
    //! Function to return the diffuse color of a Primitive
    helios::RGBcolor getColor() const;

    //! Function to return the diffuse color of a Primitive
    helios::RGBcolor getColorRGB() const;

    //! Function to return the diffuse color of a Primitive with transparency
    helios::RGBAcolor getColorRGBA() const;

    //! Function to set the diffuse color of a Primitive
    /** /param[in] "color" New color of primitive
     */
    void setColor( const helios::RGBcolor color );

    //! Function to set the diffuse color of a Primitive with transparency
    /** /param[in] "color" New color of primitive
     */
    void setColor( const helios::RGBAcolor color );

    //! Function to return a pointer to the \ref Texture data associated with this primitive
    /** \sa \ref Texture */
    Texture* getTexture( void ) const;

    //! Function to check whether this primitive has texture data
    bool hasTexture( void ) const;
    
    //! Function to return the texture map file of a Primitive
    std::string getTextureFile( void ) const;

    //! Get u-v texture coordinates at primitive vertices
    /** 
	\return 2D vector of u-v texture coordinates
    */
    std::vector<vec2> getTextureUV( void );

    //! Function to translate/shift a Primitive
    /** \param[in] "shift" Distance to translate in (x,y,z) directions.
    */
    void translate( const helios::vec3 shift );

    //! Function to rotate a Primitive about the x-, y-, or z-axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Axis about which to rotate (must be one of x, y, z )
    */
    virtual void rotate( const float rot, const char* axis ) = 0;

    //! Function to rotate a Primitive about an arbitrary axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Vector describing axis about which to rotate.
    */
    virtual void rotate( const float rot, const helios::vec3 axis ) = 0;

    //! Function to scale the dimensions of a Primitive
    /** \param[in] "S" Scaling factor
    */
    void scale( const helios::vec3 S );

    //-------- Primitive Data Functions ---------- //

    //! Add data value (int) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const int& data );
    
    //! Add data value (uint) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const uint& data );

    //! Add data value (float) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const float& data );

    //! Add data value (double) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const double& data );

    //! Add data value (vec2) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const helios::vec2& data );

    //! Add data value (vec3) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const helios::vec3& data );

    //! Add data value (vec4) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const helios::vec4& data );

    //! Add data value (int2) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const helios::int2& data );

    //! Add data value (int3) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const helios::int3& data );

    //! Add data value (int4) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const helios::int4& data );

    //! Add data value (string) associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "data" Primitive data value (scalar)
    */
    void setPrimitiveData( const char* label, const std::string& data );

    //! Add (array) data associated with a primitive element
    /** 
	\param[in] "label" Name/label associated with data
	\param[in] "type" Helios data type of primitive data (see \ref HeliosDataType)
	\param[in] "size" Number of data elements
	\param[in] "data" Pointer to primitive data
	\note While this function still works for scalar data, it is typically prefereable to use the scalar versions of this function.
    */
    void setPrimitiveData( const char* label, HeliosDataType type, uint size, void* data );

    //! Get data associated with a primitive element (integer scalar)
    /** 
	\param[in] "label" Name/label associated with data
	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, int& data ) const;
    //! Get data associated with a primitive element (vector of integers)
    /** 
	\param[in] "label" Name/label associated with data
	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<int>& data ) const;
    //! Get data associated with a primitive element (unsigned integer scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, uint& data ) const;
    //! Get data associated with a primitive element (vector of unsigned integers)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<uint>& data ) const;
    //! Get data associated with a primitive element (float scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, float& data ) const;
    //! Get data associated with a primitive element (vector of floats)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<float>& data ) const;
    //! Get data associated with a primitive element (double scalar)
    /*
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, double& data ) const;
    //! Get data associated with a primitive element (vector of doubles)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<double>& data ) const;
    //! Get data associated with a primitive element (vec2 scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, vec2& data ) const;
    //! Get data associated with a primitive element (vector of vec2's)
    /*
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<vec2>& data ) const;
    //! Get data associated with a primitive element (vec3 scalar)
    /*
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, vec3& data ) const;
    //! Get data associated with a primitive element (vector of vec3's)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<vec3>& data ) const;
    //! Get data associated with a primitive element (vec4 scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, vec4& data ) const;
    //! Get data associated with a primitive element (vector of vec4's)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<vec4>& data ) const;
    //! Get data associated with a primitive element (int2 scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, int2& data ) const;
    //! Get data associated with a primitive element (vector of int2's)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<int2>& data ) const;
    //! Get data associated with a primitive element (int3 scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, int3& data ) const;
    //! Get data associated with a primitive element (vector of int3's)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<int3>& data ) const;
    //! Get data associated with a primitive element (int4 scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, int4& data ) const;
    //! Get data associated with a primitive element (vector of int4's)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<int4>& data ) const;
    //! Get data associated with a primitive element (string scalar)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::string& data ) const;
    //! Get data associated with a primitive element (vector of strings)
    /**
    	\param[in] "label" Name/label associated with data
    	\param[out] "data" Primitive data structure
    */
    void getPrimitiveData( const char* label, std::vector<std::string>& data ) const;
   
    //! Get the Helios data type of primitive data
    /** 
	\param[in] "label" Name/label associated with data
	\return Helios data type of primitive data
	\sa HeliosDataType
    */
    HeliosDataType getPrimitiveDataType( const char* label ) const;
    
    //! Get the size/length of primitive data
    /** 
	\param[in] "label" Name/label associated with data
	\return Size/length of primitive data array
    */
    uint getPrimitiveDataSize( const char* label ) const;

    //! Check if primitive data 'label' exists
    /** 
	\param[in] "label" Name/label associated with data
      \return True/false
    */
    bool doesPrimitiveDataExist( const char* label ) const;

    //! Return labels for all primitive data for this particular primitive
    std::vector<std::string> listPrimitiveData( void ) const;
    
  protected:

    //! Unique universal identifier
    uint UUID;
    
    //! Type of primitive element (e.g., patch, triangle, etc.)
    PrimitiveType prim_type;

    //! Diffuse RGB color
    helios::RGBAcolor color;
    
    //! Texture data (pointer)
    Texture* texture;

    //! Affine transformation matrix
    float transform[16];
    
    //! (u,v) texture coordinates
    std::vector<vec2> uv;
  
    std::map<std::string,HeliosDataType > primitive_data_types;
    std::map<std::string, std::vector<int> > primitive_data_int;
    std::map<std::string, std::vector<uint> > primitive_data_uint;
    std::map<std::string, std::vector<float> > primitive_data_float;
    std::map<std::string, std::vector<double> > primitive_data_double;
    std::map<std::string, std::vector<vec2> > primitive_data_vec2;
    std::map<std::string, std::vector<vec3> > primitive_data_vec3;
    std::map<std::string, std::vector<vec4> > primitive_data_vec4;
    std::map<std::string, std::vector<int2> > primitive_data_int2;
    std::map<std::string, std::vector<int3> > primitive_data_int3;
    std::map<std::string, std::vector<int4> > primitive_data_int4;
    std::map<std::string, std::vector<std::string> > primitive_data_string;
    std::map<std::string, std::vector<bool> > primitive_data_bool;
    
  };

  //--------------------- GEOMETRIC PRIMITIVES -----------------------------------//
  
  //! Rectangular geometric object
  /** Position is given with respect to its center. Patches can only be added through the Context member function \ref Context::addPatch(). 

      \image html doc/images/Patch.png "Sample image of a Patch." width=3cm
      \ingroup primitives
  */
  class Patch : public Primitive{
  public:
    
    //! Patch constructor - colored by RGBcolor
    Patch( const helios::vec3 center, const helios::vec2 size, const helios::vec3 rotation, const helios::RGBAcolor color, const uint UUID );

    //! Patch constructor - colored by texture map
    Patch( const helios::vec3 center, const helios::vec2 size, const helios::vec3 rotation, Texture* texture, const uint UUID );

    //! Patch constructor - colored by texture map with custom (u,v) coordinates
    Patch( const helios::vec3 center, const helios::vec2 size, const helios::vec3 rotation, Texture* texture, const helios::vec2 _uv_center_, const helios::vec2 _uv_size_, const uint UUID );

    //! Patch destructor
    ~Patch(){};

    //! Get the primitive surface area
    /** \return Surface area of the Patch. */
    float getArea(void) const;

    //! Get a unit vector normal to the primitive surface
    /** \return Unit vector normal to the surface of the Patch. */
    helios::vec3 getNormal(void) const;

    //! Function to return the (x,y,z) coordinates of the vertices of a Primitve
    /** \return Vector containing four sets of the (x,y,z) coordinates of each vertex.*/
    std::vector<helios::vec3> getVertices( void ) const;

    //! Get the size of the Patch in x- and y-directions
    /** \return vec2 describing the length and width of the Patch.*/
    helios::vec2 getSize(void) const;

    //! Get the (x,y,z) coordinates of the Patch center point
    /** \return vec3 describing (x,y,z) coordinate of Patch center.*/
    helios::vec3 getCenter(void) const;

    //! Function to rotate a Primitive about the x-, y-, or z-axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Axis about which to rotate (must be one of x, y, z )
    */
    void rotate( const float rot, const char* axis );

    //! Function to rotate a Primitive about an arbitrary axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Vector describing axis about which to rotate.
    */
    void rotate( const float rot, const helios::vec3 axis );
    
  protected:

    //! surface area
    float area;

  };

  //! Triangular geometric primitive object
  /** A Triangle is specified by the positions of its three vertices. Triangles can only be added through the Context member function \ref Context::addTriangle().
      
      \image html doc/images/Triangle.png "Sample image of a Triangle." width=3cm
      \ingroup primitives 
  */
  class Triangle : public Primitive{
  public:
    
    //! Triangle constructor 
    Triangle( const helios::vec3 vertex0, const helios::vec3 vertex1, const helios::vec3 vertex2, const helios::RGBAcolor color, const uint UUID );

    //! Triangle constructor 
    Triangle( const helios::vec3 vertex0, const helios::vec3 vertex1, const helios::vec3 vertex2, Texture* texture, const helios::vec2 uv0, const helios::vec2 uv1, const helios::vec2 uv2, const uint UUID );
    
    //! Triangle destructor
    ~Triangle(){};

    //! Get the primitive surface area
    /** \return Surface area of the Triangle. */
    float getArea(void) const;

    //! Get a unit vector normal to the primitive surface
    /** \return Unit vector normal to the surface of the Triangle. */
    helios::vec3 getNormal(void) const;

    //! Function to return the (x,y,z) coordinates of the vertices of a Primitve
    /** \return Vector containing three sets of the (x,y,z) coordinates of each vertex.*/
    std::vector<helios::vec3> getVertices( void ) const;

    //! Function to return the (x,y,z) coordinates of a given Triangle vertex
    /**
       \param[in] "number" Triangle vertex (0, 1, or 2)
       \return (x,y,z) coordinates of triangle vertex
    */
    helios::vec3 getVertex( int number );

    //! Function to rotate a Primitive about the x-, y-, or z-axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Axis about which to rotate (must be one of x, y, z )
    */
    void rotate( const float rot, const char* axis );

    //! Function to rotate a Primitive about an arbitrary axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Vector describing axis about which to rotate.
    */
    void rotate( const float rot, const helios::vec3 axis );

  private:

    //!(x,y,z) coordinates of triangle vertex #0
    helios::vec3 vertex0;

    //!(x,y,z) coordinates of triangle vertex #1
    helios::vec3 vertex1;
    
    //!(x,y,z) coordinates of triangle vertex #2
    helios::vec3 vertex2;

    //!surface area
    float area;

    void makeTransformationMatrix( const helios::vec3 vertex0, const helios::vec3 vertex1, const helios::vec3 vertex2 );
    
  };

  //! Parallelpiped geometric object filled with a participating medium
  /** Position is given with respect to its center. Voxels can only be added through the Context member function \ref Context::addVoxel(). 
      \ingroup primitives
  */
  class Voxel : public Primitive{
  public:
  
    //! Voxel constructors 
    Voxel( const helios::vec3 center, const helios::vec3 size, const float rotation, const helios::RGBAcolor color, const uint UUID );

    //! Voxel destructor
    ~Voxel(){};

    //! Get the primitive surface area
    /** \return Surface area of the Voxel. */
    float getArea(void) const;

    //! This function is not used for a Voxel
    helios::vec3 getNormal(void) const;

    //! Function to return the (x,y,z) coordinates of the vertices of a Primitve
    /** \return Vector containing eight sets of the (x,y,z) coordinates of each vertex.*/
    std::vector<helios::vec3> getVertices( void ) const;
    
    //! Function to return the Volume of a Voxel
    /** \return Volume of the Voxel. */
    float getVolume(void);

    //! Get the (x,y,z) coordinates of the Voxel center point
    /** \return vec3 describing (x,y,z) coordinate of Patch center.*/
    helios::vec3 getCenter(void);

    //! Get the size of the Voxel in x-, y-, and z-directions
    /** \return vec3 describing the length, width and depth of the Voxel.*/
    helios::vec3 getSize(void);

    //! Function to rotate a Primitive about the x-, y-, or z-axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Axis about which to rotate (must be one of x, y, z )
    */
    void rotate( const float rot, const char* axis );

    //! Function to rotate a Primitive about an arbitrary axis
    /** \param[in] "rot" Rotation angle in radians.
	\param[in] "axis" Vector describing axis about which to rotate.
    */
    void rotate( const float rot, const helios::vec3 axis );
    
  private:
    
    float area;

    float volume;

  };

//! Stores the state associated with simulation
/** The Context provides an interface to global information about the application environment.   It allows access to application-level operations such as adding geometry, running models, and visualization. After creation, the Context must first be initialized via a call to initializeContext(), after which geometry and models can be added and simulated.  Memory associated with the Context is freed by calling finalize().
 */ 
class Context{
private:
  
  //---------- PRIMITIVE/OBJECT HELIOS::VECTORS ----------------//

  //! Helios::Vector containing a pointer to each primitive
  /** \note A Primitive's index in this vector is its \ref UUID */
  std::map<uint,Primitive*> primitives;

  //! Map containing variable data for each primitive
  std::map<std::string,std::vector<float> > variables;

  //! Map containing data values for timeseries
  std::map<std::string, std::vector<float> > timeseries_data;

  //! Map containging floating point values corresponding to date/time in timeseries
  /** \note The date value is calculated as Julian_day + hour/24.f + minute/24.f/60.f */
  std::map<std::string, std::vector<double> > timeseries_datevalue;

  //------------ TEXTURES ----------------//

  std::map<std::string,Texture> textures;

  Texture* addTexture( const char* texture_file );
  
  //----------- GLOBAL DATA -------------//

  std::map<std::string, GlobalData> globaldata;

  //---------- CONTEXT PRIVATE MEMBER VARIABLES ---------//
  //NOTE: variables are initialized and documented in initializeContext() member function

  //! Simulation date (\ref Date vector)
  /** 
      \sa \ref setDate(), \ref getDate()
  */
  helios::Date sim_date;

  //! Simutation time (\ref Time vector)
  /** 
      \sa \ref setTime(), \ref getTime()
  */
  helios::Time sim_time;

  //! Radius of a sphere centered at (0,0) that bounds all objects in domain. 
  float scene_radius;

  //! Random number generation engine
  std::minstd_rand0 generator;

  //! Random uniform distribution
  std::uniform_real_distribution<float> unif_distribution;

  //! Random normal distribution (mean = zero, std dev = 1)
  std::normal_distribution<float> norm_distribution;

  //---------- CONTEXT I/O ---------//

  std::vector<std::string> XMLfiles;

  std::map<std::string,std::string> loadMTL( std::string filebase, std::string filename );

  void loadPData( pugi::xml_node p, uint UUID );
  
  //---------- CONTEXT INITIALIZATION FLAGS ---------//

  //! Flag indicating whether Context has been initialized
  /**
     \sa \ref initializeContext()
  */
  bool iscontextinitialized;

  //! Flag indicating whether Context geometry has been modified
  bool isgeometrydirty;

  uint currentUUID;

public:

  //! Context default constructor
  Context(void);

  //! Context destructor
  ~Context(void);

  //! Run a self-test of the Context
  /** The Context self-test runs through validation checks of Context-related functions to ensure they are working properly.
      \return 0 if test was successful, 1 if test failed.
  */
  int selfTest( void );

  //! Mark the Context geometry as ``clean", meaning that the geometry has not been modified since last set as clean
  /** \sa \ref markGeometryDirty(), \ref isGeometryDirty() */
  void markGeometryClean(void);

  //! Mark the Context geometry as ``dirty", meaning that the geometry has been modified since last set as clean
  /** \sa \ref markGeometryClean(), \ref isGeometryDirty() */
  void markGeometryDirty(void);

  //! Query whether the Context geometry is ``dirty", meaning has the geometry been modified since last set as clean
  /** \sa \ref markGeometryClean(), \ref markGeometryDirty() */
  bool isGeometryDirty(void);

  //! Add new Patch geometric primitive
  /** Function to add a new Patch to the Context given its center, and size.
      \param[in] "center" 3D coordinates of Patch center
      \param[in] "size" width and length of Patch
      \return UUID of Patch
      \note Assumes that patch is horizontal.
      \ingroup primitives
  */
  uint addPatch( const helios::vec3& center, const helios::vec2& size );

  //! Add new Patch geometric primitive
  /** Function to add a new Patch to the Context given its center, size, and spherical rotation.
      \param[in] "center" 3D coordinates of Patch center
      \param[in] "size" width and length of Patch
      \param[in] "rotation" Spherical rotation
      \return UUID of Patch
      \note Assumes that patch is horizontal.
      \ingroup primitives
  */
  uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation );

  //! Add new Patch geometric primitive
  /** Function to add a new Patch to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
      \param[in] "center" 3D coordinates of Patch center
      \param[in] "size" width and length of Patch
      \param[in] "rotation" Spherical rotation
      \param[in] "color" diffuse R-G-B color of Patch
      \return UUID of Patch
  */
  uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBcolor& color );

  //! Add new Patch geometric primitive
  /** Function to add a new Patch to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
      \param[in] "center" 3D coordinates of Patch center
      \param[in] "size" width and length of Patch
      \param[in] "rotation" Spherical rotation
      \param[in] "color" diffuse R-G-B-A color of Patch
      \return UUID of Patch
  */
  uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBAcolor& color );

  //! Add new Patch geometric primitive
  /** Function to add a new Patch to the Context given its center, size, spherical rotation, and a texture map handle.
      \param[in] "center" 3D coordinates of Patch center
      \param[in] "size" width and length of Patch
      \param[in] "rotation" Spherical rotation
      \param[in] "texture_file" path to image file (JPEG or PNG) to be used as texture
      \return UUID of Patch
      \ingroup primitives
  */
  uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file );

  //! Add new Patch geometric primitive
  /** Function to add a new Patch to the Context given its center, size, spherical rotation, and a texture map handle.
      \param[in] "center" 3D coordinates of Patch center
      \param[in] "size" width and length of Patch
      \param[in] "rotation" Spherical rotation
      \param[in] "texture_file" path to image file (JPEG or PNG) to be used as texture
      \param[in] "uv_center" u-v coordinates of the center of texture map
      \param[in] "uv_size" size of the texture in u-v coordinates 
      \return UUID of Patch
      \ingroup primitives
  */
  uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file, const helios::vec2& uv_center, const helios::vec2& uv_size );

  //! Add new Triangle geometric primitive
  /** Function to add a new Triangle to the Context given the (x,y,z) coordinates of its vertices.
      \param[in] "vertex0" 3D coordinate of Triangle vertex #0
      \param[in] "vertex1" 3D coordinate of Triangle vertex #1
      \param[in] "vertex2" 3D coordinate of Triangle vertex #2
      \return UUID of Triangle
      \ingroup primitives
  */
  uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2 );

  //! Add new Triangle geometric primitive
  /** Function to add a new Triangle to the Context given its the (x,y,z) coordinates of its vertices and diffuse RGBcolor.
      \param[in] "vertex0" 3D coordinate of Triangle vertex #0
      \param[in] "vertex1" 3D coordinate of Triangle vertex #1
      \param[in] "vertex2" 3D coordinate of Triangle vertex #2
      \param[in] "color" diffuse R-G-B color of Triangle
      \return UUID of Triangle
      \ingroup primitives
  */
  uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const helios::RGBcolor& color );

  //! Add new Triangle geometric primitive
  /** Function to add a new Triangle to the Context given its the (x,y,z) coordinates of its vertices and diffuse RGBAcolor.
      \param[in] "vertex0" 3D coordinate of Triangle vertex #0
      \param[in] "vertex1" 3D coordinate of Triangle vertex #1
      \param[in] "vertex2" 3D coordinate of Triangle vertex #2
      \param[in] "color" diffuse R-G-B-A color of Triangle
      \return UUID of Triangle
      \ingroup primitives
  */
  uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const helios::RGBAcolor& color );

  //! Add new Triangle geometric primitive
  /** Function to add a new Triangle to the Context given its the (x,y,z) coordinates of its vertices and diffuse RGBcolor.
      \param[in] "vertex0" 3D coordinate of Triangle vertex #0
      \param[in] "vertex1" 3D coordinate of Triangle vertex #1
      \param[in] "vertex2" 3D coordinate of Triangle vertex #2
      \param[in] "texture_file" path to image file (JPEG or PNG) to be used as texture
      \param[in] "uv0" u-v texture coordinates for vertex0
      \param[in] "uv1" u-v texture coordinates for vertex1
      \param[in] "uv2" u-v texture coordinates for vertex2
      \return UUID of Triangle
      \note Assumes a default color of black.
      \ingroup primitives
  */
  uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const char* texture_file, const helios::vec2& uv0, const helios::vec2& uv1, const helios::vec2& uv2 );
  
  //! Add new Voxel geometric primitive
  /** Function to add a new Voxel to the Context given its center, and size.
      \param[in] "center" 3D coordinates of Voxel center
      \param[in] "size" width, length, and height of Voxel
      \return UUID of Voxel
      \note Assumes that voxel is horizontal.
      \ingroup primitives
  */
  uint addVoxel( const helios::vec3& center, const helios::vec3& size );

  //! Add new Voxel geometric primitive
  /** Function to add a new Voxel to the Context given its center, size, and spherical rotation.
      \param[in] "center" 3D coordinates of Voxel center
      \param[in] "size" width, length, and height of Voxel
      \param[in] "rotation" spherical rotation angle (elevation,azimuth) in radians of Voxel
      \return UUID of Voxel
      \note Assumes a default color of black.
      \ingroup primitives
  */
  uint addVoxel( const helios::vec3& center, const helios::vec3& size, const float& rotation );

  //! Add new Voxel geometric primitive
  /** Function to add a new Voxel to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
      \param[in] "center" 3D coordinates of Voxel center
      \param[in] "size" width, length, and height of Voxel
      \param[in] "rotation" spherical rotation angle (elevation,azimuth) in radians of Voxel
      \param[in] "color" diffuse R-G-B color of Voxel
      \return UUID of Voxel
  */
  uint addVoxel( const helios::vec3& center, const helios::vec3& size, const float& rotation, const helios::RGBcolor& color );

  //! Add new Voxel geometric primitive
  /** Function to add a new Voxel to the Context given its center, size, spherical rotation, and diffuse RGBAcolor.
      \param[in] "center" 3D coordinates of Voxel center
      \param[in] "size" width, length, and height of Voxel
      \param[in] "rotation" spherical rotation angle (elevation,azimuth) in radians of Voxel
      \param[in] "color" diffuse R-G-B-A color of Voxel
      \return UUID of Voxel
  */
  uint addVoxel( const helios::vec3& center, const helios::vec3& size, const float& rotation, const helios::RGBAcolor& color );

  //! Delete a single primitive from the context
  /** \param[in] "UUID" Unique universal identifier (UUID) of primitive to be deleted
  */
  void deletePrimitive( const uint UUID );
  
  //! Delete a group of primitives from the context
  /** \param[in] "UUIDs" Vector of unique universal identifiers (UUIDs) of primitives to be deleted
  */
  void deletePrimitive( const std::vector<uint> UUIDs );

  //! Make a copy of a primitive from the context
  /** \param[in] "UUID" Unique universal identifier (UUID) of primitive to be copied
      \return UUID for copied primitive
  */
  uint copyPrimitive( const uint UUID );

  //! Make a copy of a group of primitives from the context
  /** \param[in] "UUIDs" Vector of unique universal identifiers (UUIDs) of primitive to be copied
      \return UUIDs for copied primitives
  */
  std::vector<uint> copyPrimitive( const std::vector<uint> UUIDs );
  
  //!Get a pointer to a Primitive element from the Context
  /** \param[in] "UUID" Unique universal identifier (UUID) of primitive element
      \ingroup primitives
  */
  Primitive* getPrimitivePointer( const uint UUID ) const;

  //! Check if primitive exists for a given UUID
  /** \param[in] "UUID" Unique universal identifier of primitive element
   */
  bool doesPrimitiveExist( const uint UUID ) const;

  //!Get a pointer to a Patch from the Context
  /** \param[in] "UUID" Unique universal identifier (UUID) of Patch
      \ingroup primitives
  */
  Patch* getPatchPointer( const uint UUID ) const;

  //!Get a pointer to a Triangle from the Context
  /** \param[in] "UUID" Unique universal identifier (UUID) of Triangle
      \ingroup primitives
  */
  Triangle* getTrianglePointer( const uint UUID ) const;

  //!Get a pointer to a Voxel from the Context
  /** \param[in] "UUID" Unique universal identifier (UUID) of Voxel
      \ingroup primitives
  */
  Voxel* getVoxelPointer( const uint UUID ) const;

  //!Get the total number of Primitives in the Context
  /** \ingroup primitives
  */
  uint getPrimitiveCount(void) const;

  //!Get all primitive UUIDs currently in the Context
  std::vector<uint> getAllUUIDs(void) const;

  //-------- Primitive Data Functions ---------- //

  //! Add data value (int) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const int& data );
  
  //! Add data value (uint) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const uint& data );
  
  //! Add data value (float) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const float& data );
  
  //! Add data value (double) associated with a primitive element
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const double& data );
  
  //! Add data value (vec2) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const helios::vec2& data );
  
  //! Add data value (vec3) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const helios::vec3& data );
  
  //! Add data value (vec4) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const helios::vec4& data );
  
  //! Add data value (int2) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const helios::int2& data );
  
  //! Add data value (int3) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const helios::int3& data );
  
  //! Add data value (int4) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const helios::int4& data );
  
  //! Add data value (string) associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const uint UUID, const char* label, const std::string& data );

  //! Add data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element
      \param[in] "label" Name/label associated with data
      \param[in] "type" Helios data type of primitive data (see \ref HeliosDataType)
      \param[in] "size" Number of data elements
      \param[in] "data" Pointer to primitive data
  */
  void setPrimitiveData( const uint UUIDs, const char* label, HeliosDataType type, uint size, void* data );

  //! Add data value (int) associated with a primitive element
  /** 
      \param[in] "UUIDs" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const int& data );
  
  //! Add data value (uint) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const uint& data );
  
  //! Add data value (float) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const float& data );
  
  //! Add data value (double) associated with a primitive element
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const double& data );
  
  //! Add data value (vec2) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::vec2& data );
  
  //! Add data value (vec3) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::vec3& data );
  
  //! Add data value (vec4) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::vec4& data );
  
  //! Add data value (int2) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::int2& data );
  
  //! Add data value (int3) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::int3& data );
  
  //! Add data value (int4) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::int4& data );
  
  //! Add data value (string) associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Primitive data value (scalar)
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const std::string& data );
  
  //! Add data associated with a primitive element
  /** 
      \param[in] "UUID" Vector of unique universal identifiers of Primitive elements 
      \param[in] "label" Name/label associated with data
      \param[in] "type" Helios data type of primitive data (see \ref HeliosDataType)
      \param[in] "size" Number of data elements
      \param[in] "data" Pointer to primitive data
  */
  void setPrimitiveData( const std::vector<uint> UUIDs, const char* label, HeliosDataType type, uint size, void* data );

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar integer)
  */
  void getPrimitiveData( const uint UUID, const char* label, int& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of integers)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<int>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar unsigned integer)
  */
  void getPrimitiveData( const uint UUID, const char* label, uint& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of unsigned integers)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<uint>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar float)
  */
  void getPrimitiveData( const uint UUID, const char* label, float& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of floats)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<float>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar double)
  */
  void getPrimitiveData( const uint UUID, const char* label, double& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of doubles)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<double>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar vec2)
  */
  void getPrimitiveData( const uint UUID, const char* label, vec2& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of vec2's)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<vec2>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar vec3)
  */
  void getPrimitiveData( const uint UUID, const char* label, vec3& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of vec3's)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<vec3>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar vec4)
  */
  void getPrimitiveData( const uint UUID, const char* label, vec4& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of vec4's)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<vec4>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar int2)
  */
  void getPrimitiveData( const uint UUID, const char* label, int2& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of int2's)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<int2>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar int3)
  */
  void getPrimitiveData( const uint UUID, const char* label, int3& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of int3's)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<int3>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar int4)
  */
  void getPrimitiveData( const uint UUID, const char* label, int4& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of int4's)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<int4>& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (scalar string)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::string& data ) const;

  //! Get data associated with a primitive element
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \param[out] "data" Primitive data structure (vector of strings)
  */
  void getPrimitiveData( const uint UUID, const char* label, std::vector<std::string>& data ) const;  

  //! Get the Helios data type of primitive data
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \return Helios data type of primitive data
      \sa HeliosDataType
  */
  HeliosDataType getPrimitiveDataType( const uint UUID, const char* label ) const;

  //! Get the size/length of primitive data
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \return Size/length of primitive data array
  */
  uint getPrimitiveDataSize( const uint UUID, const char* label ) const;

  //! Check if primitive data 'label' exists
  /** 
      \param[in] "UUID" Unique universal identifier of Primitive element 
      \param[in] "label" Name/label associated with data
      \return True/false
  */
  bool doesPrimitiveDataExist( const uint UUID, const char* label ) const;

  //-------- Global Data Functions ---------- //

  //! Add global data value (int)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const int& data );

  //! Add global data value (uint)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const uint& data );

  //! Add global data value (float)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const float& data );

  //! Add global data value (double)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const double& data );

  //! Add global data value (vec2)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const helios::vec2& data );

  //! Add global data value (vec3)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const helios::vec3& data );

  //! Add global data value (vec4)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const helios::vec4& data );

  //! Add global data value (int2)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const helios::int2& data );

  //! Add global data value (int3)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const helios::int3& data );

  //! Add global data value (int4)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */

  void setGlobalData( const char* label, const helios::int4& data );

  //! Add global data value (string)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, const std::string& data );

  //! Add global data value (any type)
  /** 
      \param[in] "label" Name/label associated with data
      \param[in] "size" Number of elements in global data
      \param[in] "data" Global data value (scalar)
  */
  void setGlobalData( const char* label, HeliosDataType type, size_t size, void* data );

  //! Get global data value (scalar integer)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar integer)
  */
  void getGlobalData( const char* label, int& data ) const;

  //! Get global data (array of integers)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of integers)
  */
  void getGlobalData( const char* label, std::vector<int>& data ) const;

  //! Get global data value (scalar uint)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar uint)
  */
  void getGlobalData( const char* label, uint& data ) const;

  //! Get global data (array of uint's)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of uint's)
  */
  void getGlobalData( const char* label, std::vector<uint>& data ) const;

  //! Get global data value (scalar float)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar float)
  */
  void getGlobalData( const char* label, float& data ) const;

  //! Get global data (array of floats)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of floats)
  */
  void getGlobalData( const char* label, std::vector<float>& data ) const;

  //! Get global data value (scalar double)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar double)
  */
  void getGlobalData( const char* label, double& data ) const;

  //! Get global data (array of doubles)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of doubles)
  */
  void getGlobalData( const char* label, std::vector<double>& data ) const;

  //! Get global data value (scalar vec2)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar vec2)
  */
  void getGlobalData( const char* label, helios::vec2& data ) const;

  //! Get global data (array of vec2's)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of vec2's)
  */
  void getGlobalData( const char* label, std::vector<helios::vec2>& data ) const;

  //! Get global data value (scalar vec3)
  /**  
      \param[in] "label" Name/label associated with data
      \param[in] "data" Global data value (scalar vec3)
  */
  void getGlobalData( const char* label, helios::vec3& data ) const;

  //! Get global data (array of vec3's)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of vec3's)
  */
  void getGlobalData( const char* label, std::vector<helios::vec3>& data ) const;

  //! Get global data value (scalar vec4)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data' Global data value (scalar vec4)
  */
  void getGlobalData( const char* label, helios::vec4& data ) const;

  //! Get global data (array of vec4's)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of vec4's)
  */
  void getGlobalData( const char* label, std::vector<helios::vec4>& data ) const;

  //! Get global data value (scalar int2)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar int2)
  */
  void getGlobalData( const char* label, helios::int2& data ) const;

  //! Get global data (array of int2's)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of int2's)
  */
  void getGlobalData( const char* label, std::vector<helios::int2>& data ) const;

  //! Get global data value (scalar int3)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar int3)
  */
  void getGlobalData( const char* label, helios::int3& data ) const;

  //! Get global data (array of int3's)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of int3's)
  */
  void getGlobalData( const char* label, std::vector<helios::int3>& data ) const;

  //! Get global data value (scalar int4)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar int4)
  */
  void getGlobalData( const char* label, helios::int4& data ) const;

  //! Get global data (array of int4's)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of int4's)
  */
  void getGlobalData( const char* label, std::vector<helios::int4>& data ) const;

  //! Get global data value (scalar string)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Global data value (scalar string)
  */
  void getGlobalData( const char* label, std::string& data ) const;

  //! Get global data (array of strings)
  /**  
      \param[in] "label" Name/label associated with data
      \param[out] "data" Pointer to global data (array of strings)
  */
  void getGlobalData( const char* label, std::vector<std::string>& data ) const;

  //! Get the Helios data type of global data
  /**  
      \param[in] "label" Name/label associated with data
      \return Helios data type of global data
      \sa HeliosDataType
  */
  HeliosDataType getGlobalDataType( const char* label ) const;
  
  //! Get the size/length of global data
  /** 
      \param[in] "label" Name/label associated with data
      \return Size/length of global data array
  */
  size_t getGlobalDataSize( const char* label ) const;
  
  //! Check if global data 'label' exists
  /** 
      \param[in] "label" Name/label associated with data
      \return True/false
  */
  bool doesGlobalDataExist( const char* label ) const;
  
  //--------- Compound Objects Functions -------------//
  
  //! Add a spherical compound object to the Context
  /** \param[in] "Ndivs" Number of tesselations in zenithal and azimuthal directions
      \param[in] "center" (x,y,z) coordinate of sphere center
      \param[in] "radius" Radius of sphere
      \note Assumes a default color of green
      \ingroup compoundobjects
  */
  std::vector<uint> addSphere( const uint Ndivs, const helios::vec3 center, const float radius );

  //! Add a spherical compound object to the Context
  /** \param[in] "Ndivs" Number of tesselations in zenithal and azimuthal directions
      \param[in] "center" (x,y,z) coordinate of sphere center
      \param[in] "radius" Radius of sphere
      \param[in] "color" r-g-b color of sphere
      \ingroup compoundobjects
  */
  std::vector<uint> addSphere( const uint Ndivs, const helios::vec3 center, const float radius, const RGBcolor color );

  //! Add a spherical compound object to the Context
  /** \param[in] "Ndivs" Number of tesselations in zenithal and azimuthal directions
      \param[in] "center" (x,y,z) coordinate of sphere center
      \param[in] "radius" Radius of sphere
      \param[in] "color" r-g-b-a color of sphere
      \ingroup compoundobjects
  */
  //std::vector<uint> addSphere( const uint Ndivs, const helios::vec3 center, const float radius, const RGBAcolor color );
  
  //! Add a 3D tube compound object to the Context
  /** A `tube' or `snake' compound object comprised of Triangle primitives
      \image html doc/images/Tube.png "Sample image of a Tube compound object." width=0.1cm
      \param[in] "Ndivs" Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.  
      \param[in] "nodes" Vector of (x,y,z) positions defining Tube segments.
      \param[in] "radius" Radius of the tube at each node position.
      \note Ndivs must be greater than 2.
      \ingroup compoundobjects
  */
  std::vector<uint> addTube( const uint Ndivs, const std::vector<helios::vec3> nodes, const std::vector<float> radius );

  //! Add a 3D tube compound object to the Context and specify its diffuse color
  /** A `tube' or `snake' compound object comprised of Triangle primitives
      \param[in] "Ndivs" Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.  
      \param[in] "nodes" Vector of (x,y,z) positions defining Tube segments.
      \param[in] "radius" Radius of the tube at each node position.
      \param[in] "color" Diffuse color of each tube segment.
      \note Ndivs must be greater than 2.
      \ingroup compoundobjects
  */
  std::vector<uint> addTube( const uint Ndivs, const std::vector<helios::vec3> nodes, const std::vector<float> radius, const std::vector<helios::RGBcolor> color );

  //! Add a 3D tube compound object to the Context that is texture-mapped
  /** A `tube' or `snake' compound object comprised of Triangle primitives
      \param[in] "Ndivs" Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.  
      \param[in] "nodes" Vector of (x,y,z) positions defining Tube segments.
      \param[in] "radius" Radius of the tube at each node position.
      \param[in] "texturefile" Name of image file for texture map
      \note Ndivs must be greater than 2.
      \ingroup compoundobjects
  */
  std::vector<uint> addTube( const uint Ndivs, const std::vector<helios::vec3> nodes, const std::vector<float> radius, const char* texturefile );

  //! Add a rectangular prism tesselated with Patch primitives
  /** 
      \param[in] "center" 3D coordinates of box center
      \param[in] "size" Size of the box in the x-, y-, and z-directions
      \param[in] "subdiv" Number of subdivisions in x-, y-, and z-directions 
      \return Vector of UUIDs for each sub-patch
      \note Assumes default color of green
      \note This version of addBox assumes that all surface normal vectors point away from the box
      \ingroup compoundobjects
  */
  std::vector<uint> addBox( const vec3 center, const vec3 size, const int3 subdiv );

  //! Add a rectangular prism tesselated with Patch primitives
  /** 
      \param[in] "center" 3D coordinates of box center
      \param[in] "size" Size of the box in the x-, y-, and z-directions
      \param[in] "subdiv" Number of subdivisions in x-, y-, and z-directions 
      \param[in] "color" r-g-b color of box
      \return Vector of UUIDs for each sub-patch
      \note This version of addBox assumes that all surface normal vectors point away from the box
      \ingroup compoundobjects
  */
  std::vector<uint> addBox( const vec3 center, const vec3 size, const int3 subdiv, const RGBcolor color );

  //! Add a rectangular prism tesselated with Patch primitives
  /** 
      \param[in] "center" 3D coordinates of box center
      \param[in] "size" Size of the box in the x-, y-, and z-directions
      \param[in] "subdiv" Number of subdivisions in x-, y-, and z-directions 
      \param[in] "color" r-g-b color of box
      \param[in] "reverse_normals" Flip all surface normals so that patch normals point inside the box
      \return Vector of UUIDs for each sub-patch
      \note This version of addBox assumes that all surface normal vectors point away from the box
      \ingroup compoundobjects
  */
  std::vector<uint> addBox( const vec3 center, const vec3 size, const int3 subdiv, const helios::RGBcolor color, const bool reverse_normals );

  //! Add a patch that is subdivided into a regular grid of sub-patches (tiled) 
  /** 
      \param[in] "center" 3D coordinates of box center
      \param[in] "size" Size of the box in the x- and y-directions
      \param[in] "rotation" Spherical rotation of tiled surface
      \param[in] "subdiv" Number of subdivisions in x- and y-directions 
      \return Vector of UUIDs for each sub-patch
      \note Assumes default color of green
      \ingroup compoundobjects
  */
  std::vector<uint> addTile( const vec3 center, const vec2 size, const SphericalCoord rotation, const int2 subdiv );

  //! Add a patch that is subdivided into a regular grid of sub-patches (tiled) 
  /** 
      \param[in] "center" 3D coordinates of box center
      \param[in] "size" Size of the box in the x- and y-directions
      \param[in] "rotation" Spherical rotation of tiled surface
      \param[in] "subdiv" Number of subdivisions in x- and y-directions 
      \param[in] "color" r-g-b color of tiled surface
      \return Vector of UUIDs for each sub-patch
      \ingroup compoundobjects
  */
  std::vector<uint> addTile( const vec3 center, const vec2 size, const SphericalCoord rotation, const int2 subdiv, const RGBcolor color );

  //! Add a patch that is subdivided into a regular grid of sub-patches (tiled) 
  /** 
      \param[in] "center" 3D coordinates of box center
      \param[in] "size" Size of the box in the x- and y-directions
      \param[in] "rotation" Spherical rotation of tiled surface
      \param[in] "subdiv" Number of subdivisions in x- and y-directions 
      \param[in] "texturefile" Name of image file for texture map 
      \return Vector of UUIDs for each sub-patch
      \ingroup compoundobjects
  */
  std::vector<uint> addTile( const vec3 center, const vec2 size, const SphericalCoord rotation, const int2 subdiv, const char* texturefile );

  //! Add new Disk geometric primitive to the Context given its center, and size.
  /** 
      \param[in] "Ndiv" Number to triangles used to form disk
      \param[in] "center" 3D coordinates of Disk center
      \param[in] "size" length of Disk semi-major and semi-minor radii
      \return Vector of UUIDs for each sub-triangle
      \note Assumes that disk is horizontal.
      \note Assumes a default color of black.
      \ingroup compoundobjects
  */
  std::vector<uint> addDisk( const uint Ndiv, const helios::vec3& center, const helios::vec2& size );

  //! Add new Disk geometric primitive
  /** Function to add a new Disk to the Context given its center, size, and spherical rotation.
      \param[in] "Ndiv" Number to triangles used to form disk
      \param[in] "center" 3D coordinates of Disk center
      \param[in] "size" length of Disk semi-major and semi-minor radii
      \param[in] "rotation" spherical rotation angle (elevation,azimuth) in radians of Disk
      \return Vector of UUIDs for each sub-triangle
      \note Assumes a default color of black.
      \ingroup compoundobjects
  */
  std::vector<uint> addDisk( const uint Ndiv, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation );

  //! Add new Disk geometric primitive
  /** Function to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
      \param[in] "Ndiv" Number to triangles used to form disk
      \param[in] "center" 3D coordinates of Disk center
      \param[in] "size" length of Disk semi-major and semi-minor radii
      \param[in] "rotation" spherical rotation angle (elevation,azimuth) in radians of Disk
      \param[in] "color" diffuse R-G-B color of Disk
      \return Vector of UUIDs for each sub-triangle
      \ingroup compoundobjects
  */
  std::vector<uint> addDisk( const uint Ndiv, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBcolor& color );

  //! Add new Disk geometric primitive
  /** Function to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBAcolor.
      \param[in] "Ndiv" Number to triangles used to form disk
      \param[in] "center" 3D coordinates of Disk center
      \param[in] "size" length of Disk semi-major and semi-minor radii
      \param[in] "rotation" spherical rotation angle (elevation,azimuth) in radians of Disk
      \param[in] "color" diffuse R-G-B-A color of Disk
      \return Vector of UUIDs for each sub-triangle
      \ingroup compoundobjects
  */
  std::vector<uint> addDisk( const uint Ndiv, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBAcolor& color );

  //! Add new Disk geometric primitive
  /** Function to add a new Disk to the Context given its center, size, spherical rotation, and a texture map handle.
      \param[in] "Ndiv" Number to triangles used to form disk
      \param[in] "center" 3D coordinates of Disk center
      \param[in] "size" length of Disk semi-major and semi-minor radii
      \param[in] "rotation" spherical rotation angle (elevation,azimuth) in radians of Disk
      \param[in] "texture_file" path to JPEG file to be used as texture
      \return Vector of UUIDs for each sub-triangle
      \note Assumes a default color of black.
      \ingroup compoundobjects
  */
  std::vector<uint> addDisk( const uint Ndiv, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file );

  //! Add a data point to timeseries of data
  /**\param[in] "label" Name of timeseries variable (e.g., temperature)
     \param[in] "value" Value of timeseries data point
     \param[in] "date" Date vector corresponding to the time of `value' (see \ref helios::Date "Date")
     \param[in] "time" Time vector corresponding to the time of `value' (see \ref helios::time "Time")
     \ingroup timeseries
  */
  void addTimeseriesData( const char* label, const float value, const Date date, const Time time );

  //! Set the Context date and time by providing the index of a timeseries data point
  /**\param[in] "label" Name of timeseries variable (e.g., temperature)
     \param[in] "index" Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     \ingroup timeseries
  */
  void setCurrentTimeseriesPoint( const char* label, const uint index );

  //! Get a timeseries data point by specifying a date and time vector.
  /**This function interpolates the timeseries data to provide a value at exactly `date' and `time'.  Thus, `date' and `time' must be between the first and last timeseries values.
     \param[in] "label" Name of timeseries variable (e.g., temperature)
     \param[in] "date" Date vector corresponding to the time of `value' (see \ref helios::Date "Date")
     \param[in] "time" Time vector corresponding to the time of `value' (see \ref helios::time "Time")
     \return Value of timeseries data point
     \ingroup timeseries
   */
  float queryTimeseriesData( const char* label, const Date date, const Time time ) const;

  //! Get a timeseries data point by index in the timeseries
  /**This function returns timeseries data by index, and is typically used when looping over all data in the timeseries.  See \ref getTimeseriesLength() to get the total length of the timeseries data.
     \param[in] "label" Name of timeseries variable (e.g., temperature)
     \param[in] "index" Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     \return Value of timeseries data point
     \ingroup timeseries
   */
  float queryTimeseriesData( const char* label, const uint index ) const;

  //! Get the time associated with a timeseries data point
  /**\param[in] "label" Name of timeseries variable (e.g., temperature)
     \param[in] "index" Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     \return Time of timeseries data point
     \ingroup timeseries
   */
  Time queryTimeseriesTime( const char* label, const uint index ) const;

  //! Get the date associated with a timeseries data point
  /**\param[in] "label" Name of timeseries variable (e.g., temperature)
     \param[in] "index" Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     \return Date of timeseries data point
     \ingroup timeseries
   */
  Date queryTimeseriesDate( const char* label, const uint index ) const;

  //! Get the length of timeseries data
  /**
     \param[in] "label" Name of timeseries variable (e.g., temperature) 
     \ingroup timeseries
  */
  uint getTimeseriesLength( const char* label ) const;

  //! Get a box that bounds all primitives in the domain
  void getDomainBoundingBox( helios::vec2& xbounds, helios::vec2& ybounds, helios::vec2& zbounds ) const;

  //! Get the center and radius of a sphere that bounds all primitives in the domain
  void getDomainBoundingSphere( helios::vec3& center, float& radius ) const;

  //! Load inputs specified in an XML file. 
  /** \param[in] "filename" name of XML file.
      \note This function is based on the pugi xml parser.  See <a href="www.pugixml.org">pugixml.org</a>
      \ingroup context
   */
  std::vector<uint> loadXML( const char* filename );

  //! Get names of XML files that are currently loaded
  std::vector<std::string> getLoadedXMLFiles( void );

  //! Write Context geometry and data to XML file
  /** \param[in] "filename" name of XML file.
      \ingroup context
   */
  void writeXML( const char* filename ) const;

  //! Load geometry contained in a Stanford polygon file (.ply)
  /** \param[in] "filename" name of ply file.
      \param[in] "origin" (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
      \param[in] "scale" Scaling factor to be applied to PLY vertex coordinates
      \note Assumes default color of blue if no colors are specified in the .ply file
      \ingroup context
   */
  std::vector<uint> loadPLY( const char* filename, helios::vec3 origin, float scale );

  //! Load geometry contained in a Stanford polygon file (.ply)
  /** \param[in] "filename" name of ply file.
      \param[in] "origin" (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
      \param[in] "scale" Scaling factor to be applied to PLY vertex coordinates
      \param[in] "rotation" Spherical rotation of PLY object about origin
      \note Assumes default color of blue if no colors are specified in the .ply file
      \ingroup context
   */
  std::vector<uint> loadPLY( const char* filename, const helios::vec3 origin, const float scale, const SphericalCoord rotation );

  //! Load geometry contained in a Stanford polygon file (.ply)
  /** \param[in] "filename" name of ply file.
      \param[in] "origin" (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
      \param[in] "scale" Scaling factor to be applied to PLY vertex coordinates
      \param[in] "default_color" Color to be used if no r-g-b color values are given for PLY nodes
      \ingroup context
   */
  std::vector<uint> loadPLY( const char* filename, const helios::vec3 origin, const float scale, const RGBcolor default_color );

  //! Load geometry contained in a Stanford polygon file (.ply)
  /** \param[in] "filename" name of ply file.
      \param[in] "origin" (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
      \param[in] "scale" Scaling factor to be applied to PLY vertex coordinates
      \param[in] "rotation" Spherical rotation of PLY object about origin
      \param[in] "default_color" Color to be used if no r-g-b color values are given for PLY nodes
      \ingroup context
   */
  std::vector<uint> loadPLY( const char* filename, const helios::vec3 origin, const float scale, const SphericalCoord rotation, const RGBcolor default_color );

  //! Write geometry in the Context to a Stanford polygon file (.ply)
  /**
     \param[in] "filename" name of ply file
   */
  void writePLY( const char* filename ) const;

  //! Load geometry contained in a Wavefront OBJ file (.obj)
  /** \param[in] "filename" name of OBJ file.
      \param[in] "origin" (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
      \param[in] "scale" Scaling factor to be applied to PLY vertex coordinates
      \param[in] "rotation" Spherical rotation of PLY object about origin
      \param[in] "default_color" Color to be used if no r-g-b color values are given for PLY nodes
      \ingroup context
   */
  std::vector<uint> loadOBJ( const char* filename, const helios::vec3 origin, const float scale, const SphericalCoord rotation, const RGBcolor default_color );

  //! Load geometry contained in a Wavefront OBJ file (.obj)
  /** \param[in] "filename" name of OBJ file.
      \param[in] "origin" (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
      \param[in] "scale" Scaling factor to be applied to PLY vertex coordinates
      \param[in] "rotation" Spherical rotation of PLY object about origin
      \param[in] "default_color" Color to be used if no r-g-b color values are given for PLY nodes
      \param[in] "upaxis" Direction of "up" vector used when creating OBJ file
      \ingroup context
   */
  std::vector<uint> loadOBJ( const char* filename, const helios::vec3 origin, const float scale, const SphericalCoord rotation, const RGBcolor default_color, const char* upaxis );

  //! Write geometry in the Context to a Wavefront file (.obj)
  /**
     \param[in] "filename" Base filename of .obj and .mtl file
  */
  void writeOBJ( const char* filename ) const;

  //! Set simulation date by day, month, year
  /**  
       \param[in] "day" Day of the month (1-31)
       \param[in] "month" Month of year (1-12)
       \param[in] "year" Year in YYYY format
       \sa \ref getDate()
   */
  void setDate( int day, int month, int year );

  //! Set simulation date by \ref Date vector
  /**  
       \param[in] "date" \ref Date vector
       \sa \ref getDate()
   */
  void setDate( helios::Date date );

  //! Set simulation date by Julian day
  /**  
       \param[in] "Julian_day" Julian day of year (1-366)
       \param[in] "year" Year in YYYY format. Note: this is used to determine leap years.
       \sa \ref getDate()
   */
  void setDate( int Julian_day, int year );

  //! Get simulation date
  /** 
      \return \ref Date vector
      \sa \ref setDate(), \ref getJulianDate()
  */
  helios::Date getDate( void ) const;

  //! Get a string corresponding to the month of the simulation date
  /** 
      \return \ref Month string (e.g., Jan, Feb, Mar, etc)
      \sa \ref setDate(), \ref getJulianDate()
  */
  const char* getMonthString( void ) const;

  //! Get simulation date by Julian day
  /**
     \return Julian day of year (1-366)
     \sa \ref setDate(), \ref getDate()
  */
  int getJulianDate( void ) const;

  //! Set simulation time
  /**  
       \param[in] "minute" Minute of hour (0-59)
       \param[in] "hour" Hour of day (0-23)
       \sa \ref getTime(), \ref setSunDirection()
   */
  void setTime( int minute, int hour );

  //! Set simulation time
  /**  
       \param[in] "second" Second of minute (0-59)
       \param[in] "minute" Minute of hour (0-59)
       \param[in] "hour" Hour of day (0-23)
       \sa \ref getTime(), \ref setSunDirection()
   */
  void setTime( int second, int minute, int hour );

  //! Set simulation time using \ref Time vector
  /**  
       \param[in] "time" Time vector
       \sa \ref getTime(), \ref setSunDirection()
   */
  void setTime( helios::Time time );

  //! Get the simulation time
  /** 
      \return \ref \ref Time vector
      \sa \ref setTime()
  */
  helios::Time getTime( void ) const;

  //! Draw a random number from a uniform distribution between 0 and 1
  float randu(void);

  //! Draw a random number from a uniform distribution with specified range
  /** \param[in] "min" Minimum value of random uniform distribution (float)
      \param[in] "max" Maximum value of random uniform distribution (float)
  */
  float randu( float min, float max );

  //! Draw a random number from a uniform distribution with specified range
  /** \param[in] "min" Minimum value of random uniform distribution (integer)
      \param[in] "max" Maximum value of random uniform distribution (integer)
  */
  int randu( int min, int max );

  //! Draw a random number from a normal distribution with mean = 0, stddev = 1
  float randn(void);

  //! Draw a random number from a normal distribution with specified mean and standard deviation
  /** \param[in] "mean" Mean value of random distribution
      \param[in] "stddev" Standard deviation of random normal distribution
      \note If standard deviation is specified as negative, the absolute value is used.
  */
  float randn( float mean, float stddev );

};

}


#endif
