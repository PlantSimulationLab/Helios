/** \file "Context.h" Context header file.
 
 Copyright (C) 2016-2024 Brian Bailey
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, version 2
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 */

#ifndef HELIOS_CONTEXT
#define HELIOS_CONTEXT

#include <utility>

#include "global.h"

//! Timeseries-related functions
/** \defgroup timeseries Timeseries */

namespace helios {

class Context; //forward declaration of Context class

//! Type of primitive element
enum PrimitiveType{
    //! < Rectangular primitive
    PRIMITIVE_TYPE_PATCH = 0,
    //! < Triangular primitive
    PRIMITIVE_TYPE_TRIANGLE = 1,
    //! Rectangular prism primitive
    PRIMITIVE_TYPE_VOXEL = 2
};

//! Data types
enum HeliosDataType{
    //! integer data type
    HELIOS_TYPE_INT = 0,
    //! unsigned integer data type
    HELIOS_TYPE_UINT = 1,
    //! floating point data type
    HELIOS_TYPE_FLOAT = 2,
    //! double data type
    HELIOS_TYPE_DOUBLE = 3,
    //! helios::vec2 data type
    HELIOS_TYPE_VEC2 = 4,
    //! helios::vec3 data type
    HELIOS_TYPE_VEC3 = 5,
    //! helios::vec4 data type
    HELIOS_TYPE_VEC4 = 6,
    //! helios::int2 data type
    HELIOS_TYPE_INT2 = 7,
    //! helios::int3 data type
    HELIOS_TYPE_INT3 = 8,
    //! helios::int4 data type
    HELIOS_TYPE_INT4 = 9,
    //! std::string data type
    HELIOS_TYPE_STRING = 10,
};

//! Texture map data structure
class Texture{
public:
    
    Texture()= default;
    
    //! Constructor - initialize with given texture file
    /**
     * \param[in] "texture_file" Path to texture file. Note that file paths can be absolute, or relative to the directory in which the program is being run.
     */
    explicit Texture( const char* texture_file );
    
    //! Get the name/path of the texture map file
    std::string getTextureFile() const;
    
    //! Get the size of the texture in pixels (horizontal x vertical)
    helios::int2 getImageResolution() const;
    
    //! Check whether the texture has a transparency channel
    bool hasTransparencyChannel() const;
    
    //! Get the data in the texture transparency channel (if it exists)
    const std::vector<std::vector<bool> >* getTransparencyData() const;
    
    //! Get the solid fraction of the texture transparency channel (if it exists)
    float getSolidFraction() const;
    
private:
    std::string filename;
    bool hastransparencychannel{};
    std::vector<std::vector<bool> > transparencydata;
    helios::int2 image_resolution;
    float solidfraction{};
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

//--------------------- GEOMETRIC PRIMITIVES -----------------------------------//


//---------- COMPOUND OBJECTS ----------------//

/// Type of compound object
enum ObjectType{
    /// < Tile
    OBJECT_TYPE_TILE = 0,
    /// < Sphere
    OBJECT_TYPE_SPHERE = 1,
    //! Tube
    OBJECT_TYPE_TUBE = 2,
    //! Box
    OBJECT_TYPE_BOX = 3,
    /// < Disk
    OBJECT_TYPE_DISK = 4,
    /// < Triangular Mesh
    OBJECT_TYPE_POLYMESH = 5,
    /// < Cone/tapered cylinder
    OBJECT_TYPE_CONE = 6,
};

class CompoundObject{
public:
    
    virtual ~CompoundObject()= default;
    
    //! Get the unique identifier for the object
    uint getObjectID() const;
    
    //! Get an enumeration specifying the type of the object
    helios::ObjectType getObjectType() const;
    
    //! Return the number of primitives contained in the object
    uint getPrimitiveCount() const;
    
    //! Get the UUIDs for all primitives contained in the object
    std::vector<uint> getPrimitiveUUIDs() const;
    
    //! Check whether a primitive is a member of the object based on its UUID
    bool doesObjectContainPrimitive( uint UUID );
    
    //! Calculate the Cartesian (x,y,z) point of the center of a bounding box for the Compound Object
    helios::vec3 getObjectCenter() const;
    
    //! Calculate the total one-sided surface area of the Compound Object
    float getArea() const;
    
    //! Method to set the diffuse color for all primitives in the Compound Object
    /** /param[in] "color" New color of primitives
     */
    void setColor( const helios::RGBcolor& color );
    
    //! Method to set the diffuse color (with transparency) for all primitives in the Compound Object
    /**
     * /param[in] "color" New color of primitives
     */
    void setColor( const helios::RGBAcolor& color );
    
    //! Override the color in the texture map for all primitives in the Compound Object, in which case the primitives will be colored by the constant RGB color, but will apply the transparency channel in the texture to determine its shape
    void overrideTextureColor();
    
    //! For all primitives in the Compound Object, use the texture map to color the primitives rather than the constant RGB color. This is method reverses a previous call to overrideTextureColor(). Note that using the texture color is the default behavior.
    void useTextureColor();
    
    //! Method to check whether this object has texture data
    bool hasTexture( ) const;
    
    //! Method to return the texture map file of an Object
    std::string getTextureFile( ) const;
    
    //! Method to translate/shift a Compound Object
    /**
     * \param[in] shift Distance to translate in (x,y,z) directions.
     */
    void translate( const helios::vec3& shift );
    
    //! Method to rotate a Compound Object about the x-, y-, or z-axis
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_xyz_string Axis about which to rotate (must be one of x, y, z )
     */
    void rotate(float rotation_radians, const char* rotation_axis_xyz_string );
    
    //! Method to rotate a Compound Object about an arbitrary axis passing through the origin
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_vector Vector describing axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3& rotation_axis_vector );
    
    //! Method to rotate a Compound Object about an arbitrary line (not necessarily passing through the origin)
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis.
     * \param[in] rotation_axis_vector Vector describing axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3& origin, const helios::vec3& rotation_axis_vector );

    //! Method to scale a compound object in the x-, y- and z-directions
    /**
     * \param[in] scale Scaling factor to apply in the x-, y- and z-directions
     */
    void scale( const helios::vec3 &scale );
    
    //! Method to return the Affine transformation matrix of a Compound Object
    /**
     * \param[out] T 1D vector corresponding to Compound Object transformation matrix
     */
    void getTransformationMatrix( float (&T)[16] ) const;
    
    //! Method to set the Affine transformation matrix of a Compound Object
    /**
     * \param[in] T 1D vector corresponding to Compound Object transformation matrix
     */
    void setTransformationMatrix( float (&T)[16] );
    
    //! Method to set the UUIDs of object child primitives
    /**
     * \param[in] UUIDs Set of UUIDs corresponding to child primitives of object
     */
    void setPrimitiveUUIDs( const std::vector<uint> &UUIDs );
    
    //! Delete a single child member of the object
    /**
     * \param[in] UUID Universally unique identifier of primitive member
     */
    void deleteChildPrimitive( uint UUID );
    
    //! Delete multiple child member of the object based on a vector of UUIDs
    /**
     * \param[in] UUIDs Vector of universally unique identifiers for primitive members
     */
    void deleteChildPrimitive( const std::vector<uint> &UUIDs );
    
    //! Method to query whether all object primitives are in tact
    /**
     * \return False if any primitives have been deleted from the object since creation; True otherwise.
     */
    bool arePrimitivesComplete() const;
    
    //-------- Object Data Methods ---------- //
    
    //! Add data value (int) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const int& data );
    
    //! Add data value (uint) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const uint& data );
    
    //! Add data value (float) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const float& data );
    
    //! Add data value (double) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const double& data );
    
    //! Add data value (vec2) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const char* label, const std::string& data );
    
    //! Add (array) data associated with a object element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] type Helios data type of object data (see \ref HeliosDataType)
     * \param[in] size Number of data elements
     * \param[in] data Pointer to object data
     * \note While this method still works for scalar data, it is typically preferable to use the scalar versions of this method.
     */
    void setObjectData( const char* label, HeliosDataType type, uint size, void* data );
    
    //! Get data associated with a object element (integer scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, int& data ) const;
    
    //! Get data associated with a object element (vector of integers)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<int>& data ) const;
    
    //! Get data associated with a object element (unsigned integer scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, uint& data ) const;
    
    //! Get data associated with a object element (vector of unsigned integers)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<uint>& data ) const;
    
    //! Get data associated with a object element (float scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, float& data ) const;
    
    //! Get data associated with a object element (vector of floats)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<float>& data ) const;
    
    //! Get data associated with a object element (double scalar)
    /**
     *  \param[in] label Name/label associated with data
     *  \param[out] data Object data structure
     */
    void getObjectData( const char* label, double& data ) const;
    
    //! Get data associated with a object element (vector of doubles)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<double>& data ) const;
    
    //! Get data associated with a object element (vec2 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, vec2& data ) const;
    
    //! Get data associated with a object element (vector of vec2's)
    /**
     *  \param[in] label Name/label associated with data
     *  \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<vec2>& data ) const;
    
    //! Get data associated with a object element (vec3 scalar)
    void getObjectData( const char* label, vec3& data ) const;
    
    //! Get data associated with a object element (vector of vec3's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<vec3>& data ) const;
    //! Get data associated with a object element (vec4 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, vec4& data ) const;
    //! Get data associated with a object element (vector of vec4's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<vec4>& data ) const;
    //! Get data associated with a object element (int2 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, int2& data ) const;
    //! Get data associated with a object element (vector of int2's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<int2>& data ) const;
    //! Get data associated with a object element (int3 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, int3& data ) const;
    //! Get data associated with a object element (vector of int3's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<int3>& data ) const;
    //! Get data associated with a object element (int4 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, int4& data ) const;
    //! Get data associated with a object element (vector of int4's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<int4>& data ) const;
    //! Get data associated with a object element (string scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::string& data ) const;
    //! Get data associated with a object element (vector of strings)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure
     */
    void getObjectData( const char* label, std::vector<std::string>& data ) const;
    
    //! Get the Helios data type of object data
    /**
     * \param[in] label Name/label associated with data
     * \return Helios data type of object data
     * \sa HeliosDataType
     */
    HeliosDataType getObjectDataType( const char* label ) const;
    
    //! Get the size/length of object data
    /**
     * \param[in] label Name/label associated with data
     * \return Size/length of object data array
     */
    uint getObjectDataSize( const char* label ) const;
    
    //! Check if object data 'label' exists
    /**
     * \param[in] label Name/label associated with data
     * \return True/false
     */
    bool doesObjectDataExist( const char* label ) const;
    
    //! Clear the object data for this object
    /**
     * \param[in] label Name/label associated with data
     */
    void clearObjectData( const char* label );
    
    //! Return labels for all object data for this particular object
    std::vector<std::string> listObjectData() const;
    
protected:
    
    //! Object ID
    uint OID;
    
    //! Type of object
    helios::ObjectType type;
    
    //! UUIDs for all primitives contained in object
    std::vector<uint> UUIDs;
    
    //! Pointer to the Helios context object was added to
    helios::Context* context;
    
    //! Path to texture map file
    std::string texturefile;
    
    //! Affine transformation matrix
    float transform[16];
    
    //! Flag to indicate whether all object primitives are in tact. If any primitives have been deleted, this flag will be set to false.
    bool primitivesarecomplete = true;
    
    std::map<std::string,HeliosDataType > object_data_types;
    std::map<std::string, std::vector<int> > object_data_int;
    std::map<std::string, std::vector<uint> > object_data_uint;
    std::map<std::string, std::vector<float> > object_data_float;
    std::map<std::string, std::vector<double> > object_data_double;
    std::map<std::string, std::vector<vec2> > object_data_vec2;
    std::map<std::string, std::vector<vec3> > object_data_vec3;
    std::map<std::string, std::vector<vec4> > object_data_vec4;
    std::map<std::string, std::vector<int2> > object_data_int2;
    std::map<std::string, std::vector<int3> > object_data_int3;
    std::map<std::string, std::vector<int4> > object_data_int4;
    std::map<std::string, std::vector<std::string> > object_data_string;
    std::map<std::string, std::vector<bool> > object_data_bool;
    
};

//! Tile compound object class
class Tile : public CompoundObject {
public:
    
    //! Default constructor
    Tile(uint a_OID, const std::vector<uint> &a_UUIDs, const int2 &a_subdiv, const char *a_texturefile, helios::Context *a_context);
    
    //! Tile destructor
    ~Tile() override = default;
    
    //! Get the dimensions of the entire tile object
    helios::vec2 getSize() const;
    
    //! Get the Cartesian coordinates of the center of the tile object
    vec3 getCenter() const;
    
    //! Get the number of sub-patch divisions of the tile
    helios::int2 getSubdivisionCount() const;
    
    //! Set the number of tile sub-patch divisions
    /**
     * \param[in] subdiv Number of subdivisions in x- and y-directions.
     * \note This will delete all prior child primitives and add new UUIDs
     */
    void setSubdivisionCount( const helios::int2 &subdiv );
    
    //! Get the Cartesian coordinates of each of the four corners of the tile object
    std::vector<helios::vec3> getVertices() const;
    
    //! Get a unit vector normal to the tile object surface
    vec3 getNormal() const;
    
    //! Get the normalized (u,v) coordinates of the texture at each of the four corners of the tile object
    std::vector<helios::vec2> getTextureUV() const;
    
protected:
    
    helios::int2 subdiv;
    
};

//! Sphere compound object class
class Sphere : public CompoundObject {
public:
    
    //! Default constructor
    Sphere(uint a_OID, const std::vector<uint> &a_UUIDs, uint a_subdiv, const char *a_texturefile, helios::Context *a_context);
    
    //! Sphere destructor
    ~Sphere() override = default;
    
    //! Get the radius of the sphere
    helios::vec3 getRadius() const;
    
    //! Get the Cartesian coordinates of the center of the sphere object
    vec3 getCenter() const;
    
    //! Get the number of sub-patch divisions of the sphere object
    uint getSubdivisionCount() const;
    
    //! Set the number of sphere tesselation divisions
    /**
     * \param[in] subdiv Number of subdivisions in zenithal and azimuthal directions.
     */
    void setSubdivisionCount(uint subdiv );
    
protected:
    
    uint subdiv;
    
};

//! Tube compound object class
class Tube: public CompoundObject {
public:
    
    //! Default constructor
    Tube(uint a_OID, const std::vector<uint> &a_UUIDs, const std::vector<vec3> &a_nodes, const std::vector<float> &a_radius, const std::vector<helios::RGBcolor> &a_colors, uint a_subdiv, const char *a_texturefile, helios::Context *a_context);
    
    //! Tube destructor
    ~Tube() override = default;
    
    //! Get the Cartesian coordinates of each of the tube object nodes
    std::vector<helios::vec3> getNodes() const;
    
    //! Get the radius at each of the tube object nodes
    std::vector<float> getNodeRadii() const;
    
    //! Get the colors at each of the tube object nodes
    std::vector<helios::RGBcolor> getNodeColors() const;
    
    //! Get the number of sub-triangle divisions of the tube object
    uint getSubdivisionCount() const;
    
    //! Set the number of sphere tesselation divisions
    /**
     * \param[in] subdiv Number of subdivisions in zenithal and azimuthal directions.
     */
    void setSubdivisionCount( uint subdiv );
    
protected:
    
    std::vector<helios::vec3> nodes;
    
    std::vector<float> radius;
    
    std::vector<helios::RGBcolor> colors;
    
    uint subdiv;
    
};

//! Box compound object class
class Box : public CompoundObject {
public:
    
    //! Default constructor
    Box(uint a_OID, const std::vector<uint> &a_UUIDs, const int3 &a_subdiv, const char *a_texturefile, helios::Context *a_context);
    
    //! Box destructor
    ~Box() override = default;
    
    //! Get the dimensions of the box object in each Cartesian direction
    vec3 getSize() const;
    
    //! Get the Cartesian coordinates of the center of the box object
    vec3 getCenter() const;
    
    //! Get the number of sub-patch divisions of the box object in each Cartesian direction
    helios::int3 getSubdivisionCount() const;
    
    //! Set the number of box sub-patch divisions
    /**
     * \param[in] subdiv Number of patch subdivisions in each direction.
     */
    void setSubdivisionCount( const helios::int3 &subdiv );
    
protected:
    
    helios::int3 subdiv;
    
};

//! Disk compound object class
class Disk : public CompoundObject {
public:
    
    //! Default constructor
    Disk(uint a_OID, const std::vector<uint> &a_UUIDs, int2 a_subdiv, const char *a_texturefile, helios::Context *a_context);
    
    //! Disk destructor
    ~Disk() override = default;
    
    //! Get the lateral dimensions of the disk object
    vec2 getSize() const;
    
    //! Get the Cartesian coordinates of the center of the disk object
    vec3 getCenter() const;
    
    //! Get the number of sub-triangle divisions of the disk object
    int2 getSubdivisionCount() const;
    
    //! Set the number of disk sub-triangle divisions
    /**
     * \param[in] subdiv Number of triangle subdivisions.
     */
    void setSubdivisionCount(int2 subdiv );
    
protected:
    
    int2 subdiv;
    
};

//! Polymesh compound object class
class Polymesh : public CompoundObject {
public:
    
    //! Default constructor
    Polymesh(uint a_OID, const std::vector<uint> &a_UUIDs, const char *a_texturefile, helios::Context *a_context);
    
    //! Polymesh destructor
    ~Polymesh() override = default;
    
protected:
    
    
};

//! Cone compound object class
class Cone: public CompoundObject {
public:
    
    //! Default constructor
    Cone(uint a_OID, const std::vector<uint> &a_UUIDs, const vec3 &a_node0, const vec3 &a_node1, float a_radius0, float a_radius1, uint a_subdiv, const char *a_texturefile, helios::Context *a_context);
    
    //! Cone destructor
    ~Cone() override = default;
    
    //! Get the Cartesian coordinates of each of the cone object nodes
    /**
     * \return Vector of Cartesian coordinates of cone object nodes (length = 2).
     */
    std::vector<helios::vec3> getNodeCoordinates() const;
    
    //! Get the Cartesian coordinates of a cone object node
    /**
     * \param[in] node_index Index of the cone node. Either 0 for the cone base, or 1 for the cone apex.
     * \return Cartesian coordinate of the cone node center.
     */
    helios::vec3 getNodeCoordinate(int node_index) const;
    
    //! Get the radius at each of the cone object nodes
    /**
     * \return Vector of radii at cone object nodes (length = 2).
     */
    std::vector<float> getNodeRadii() const;
    
    //! Get the radius of a cone object node
    /**
     * \param[in] node_index Index of the cone node. Either 0 for the cone base, or 1 for the cone apex.
     * \return Radius of the cone node.
     */
    float getNodeRadius(int node_index) const;
    
    //! Get the number of sub-triangle divisions of the cone object
    uint getSubdivisionCount() const;
    
    //! Set the number of radial sub-triangle divisions
    /**
     * \param[in] subdiv Number of radial sub-triangle divisions.
     */
    void setSubdivisionCount( uint subdiv );
    
    //! Get a unit vector pointing in the direction of the cone central axis
    helios::vec3 getAxisUnitVector() const;
    
    //! Get the length of the cone along the axial direction
    float getLength() const;
    
    //! Method to scale the length of the cone
    /**
     * \param[in] S Scaling factor
     */
    void scaleLength( float S );
    
    //! Method to scale the girth of the cone
    /**
     * \param[in] S Scaling factor
     */
    void scaleGirth( float S );
    
protected:
    
    std::vector<helios::vec3> nodes;
    std::vector<float> radii;
    
    uint subdiv;
    
};


//! Primitive class
/**
 * All primitive objects inherit this class, and it provides functionality universal to all primitives.  There may be additional functionality associated with specific primitive elements.
 * @private
 */
class Primitive{
public:
    
    //! Virtual destructor
    virtual ~Primitive()= default;
    
    //! Method to get the Primitive universal unique identifier (UUID)
    /**
     * \return UUID of primitive
     */
    uint getUUID() const;
    
    //! Method to get the Primitive type
    /**
     * \sa PrimitiveType
     */
    PrimitiveType getType() const;
    
    //! Method to set the ID of the parent object the primitive belongs to (default is object 0)
    /**
     * \param[in] objID Identifier of primitive's parent object.
     */
    void setParentObjectID( uint objID );
    
    //! Method to return the ID of the parent object the primitive belongs to (default is object 0)
    uint getParentObjectID()const;
    
    //! Method to return the surface area of a Primitive
    virtual float getArea() const = 0;
    
    //! Method to return the normal vector of a Primitive
    virtual helios::vec3 getNormal() const = 0;
    
    //! Method to return the Affine transformation matrix of a Primitive
    /**
     * \param[out] T 1D vector corresponding to Primitive transformation matrix
     */
    void getTransformationMatrix( float (&T)[16] ) const;
    
    //! Method to set the Affine transformation matrix of a Primitive
    /**
     * \param[in] T 1D vector corresponding to Primitive transformation matrix
     */
    void setTransformationMatrix( float (&T)[16] );
    
    //! Method to return the (x,y,z) coordinates of the vertices of a Primitve
    virtual std::vector<helios::vec3> getVertices( ) const = 0;
    
    //! Method to return the (x,y,z) coordinates of the Primitive centroid
    virtual helios::vec3 getCenter() const = 0;
    
    //! Method to return the diffuse color of a Primitive
    helios::RGBcolor getColor() const;
    
    //! Method to return the diffuse color of a Primitive
    helios::RGBcolor getColorRGB() const;
    
    //! Method to return the diffuse color of a Primitive with transparency
    helios::RGBAcolor getColorRGBA() const;
    
    //! Method to set the diffuse color of a Primitive
    /**
     * /param[in] color New color of primitive
     */
    void setColor( const helios::RGBcolor& color );
    
    //! Method to set the diffuse color of a Primitive with transparency
    /**
     * /param[in] color New color of primitive
     */
    void setColor( const helios::RGBAcolor& color );
    
    //! Method to check whether this primitive has texture data
    bool hasTexture( ) const;
    
    //! Method to return the texture map file of a Primitive
    std::string getTextureFile( ) const;

    //! Method to set the texture map file of a Primitive
    /**
    * \param[in] texture Path to texture file
    */
    void setTextureFile( const char* texture );

  //! Get u-v texture coordinates at primitive vertices
    /**
     *\return 2D vector of u-v texture coordinates
     */
    std::vector<vec2> getTextureUV( );


    //! Set u-v texture coordinates at primitive vertices
    /**
     *\param[in] vector of u-v texture coordinates
     */
    void setTextureUV( const std::vector<vec2> &uv );
    
    //! Override the color in the texture map, in which case the primitive will be colored by the constant RGB color, but will apply the transparency channel in the texture to determine its shape
    void overrideTextureColor( );
    
    //! Use the texture map to color the primitive rather than the constant RGB color. This is method reverses a previous call to overrideTextureColor(). Note that using the texture color is the default behavior.
    void useTextureColor( );
    
    //! Check if color of texture map is overridden by the diffuse R-G-B color of the primitive
    bool isTextureColorOverridden( ) const;
    
    //! Get fraction of primitive surface area that is non-transparent
    /**
     * \return Fraction of non-transparent area (=1 if primitive does not have a transparent texture).
     */
    float getSolidFraction() const;

    //! Get fraction of primitive surface area that is non-transparent
    /**
     * \param[in] solidFraction Fraction of primitive surface area that is non-transparent
    */
    void setSolidFraction( float solidFraction );

    //! Method to transform a Primitive based on a transformation matrix
    /**
     * \param[in] T 4x4 Affine transformation matrix.
     */
    void applyTransform( float (&T)[16] );

    //! Method to translate/shift a Primitive
    /**
    * \param[in] shift Distance to translate in (x,y,z) directions.
    */
    void translate( const helios::vec3& shift );
    
    //! Method to rotate a Primitive about the x-, y-, or z-axis
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_xyz_string Axis about which to rotate (must be one of x, y, z )
     */
    virtual void rotate(float rotation_radians, const char* rotation_axis_xyz_string ) = 0;
    
    //! Method to rotate a Primitive about an arbitrary axis passing through the origin.
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_vector Vector describing axis about which to rotate.
     */
    virtual void rotate(float rotation_radians, const helios::vec3& rotation_axis_vector ) = 0;
    
    //! Method to rotate a Primitive about an arbitrary line (not necessarily passing through the origin)
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis.
     * \param[in] rotation_axis_vector Vector describing the direction of the axis about which to rotate.
     */
    virtual void rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector ) = 0;
    
    //! Method to scale the dimensions of a Primitive
    /**
     * \param[in] S Scaling factor
     */
    void scale( const helios::vec3& S );
    
    //-------- Primitive Data Methods ---------- //
    
    //! Add data value (int) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const int& data );
    
    //! Add data value (uint) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const uint& data );
    
    //! Add data value (float) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const float& data );
    
    //! Add data value (double) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const double& data );
    
    //! Add data value (vec2) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const char* label, const std::string& data );
    //! Add (array) data associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] type Helios data type of primitive data (see \ref HeliosDataType)
     * \param[in] size Number of data elements
     * \param[in] data Pointer to primitive data
     * \note While this method still works for scalar data, it is typically preferable to use the scalar versions of this method.
     */
    void setPrimitiveData( const char* label, HeliosDataType type, uint size, void* data );
    //! Get data associated with a primitive element (integer scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, int& data ) const;
    //! Get data associated with a primitive element (vector of integers)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<int>& data ) const;
    //! Get data associated with a primitive element (unsigned integer scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, uint& data ) const;
    //! Get data associated with a primitive element (vector of unsigned integers)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<uint>& data ) const;
    //! Get data associated with a primitive element (float scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, float& data ) const;
    //! Get data associated with a primitive element (vector of floats)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<float>& data ) const;
    //! Get data associated with a primitive element (double scalar)
    /**
     *  \param[in] label Name/label associated with data
     *  \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, double& data ) const;
    //! Get data associated with a primitive element (vector of doubles)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<double>& data ) const;
    //! Get data associated with a primitive element (vec2 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, vec2& data ) const;
    //! Get data associated with a primitive element (vector of vec2's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<vec2>& data ) const;
    //! Get data associated with a primitive element (vec3 scalar)
    /**
     *  \param[in] label Name/label associated with data
     *  \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, vec3& data ) const;
    //! Get data associated with a primitive element (vector of vec3's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<vec3>& data ) const;
    //! Get data associated with a primitive element (vec4 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, vec4& data ) const;
    //! Get data associated with a primitive element (vector of vec4's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<vec4>& data ) const;
    //! Get data associated with a primitive element (int2 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, int2& data ) const;
    //! Get data associated with a primitive element (vector of int2's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<int2>& data ) const;
    //! Get data associated with a primitive element (int3 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, int3& data ) const;
    //! Get data associated with a primitive element (vector of int3's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<int3>& data ) const;
    //! Get data associated with a primitive element (int4 scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, int4& data ) const;
    //! Get data associated with a primitive element (vector of int4's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<int4>& data ) const;
    //! Get data associated with a primitive element (string scalar)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::string& data ) const;
    //! Get data associated with a primitive element (vector of strings)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure
     */
    void getPrimitiveData( const char* label, std::vector<std::string>& data ) const;
    
    //! Get the Helios data type of primitive data
    /**
     * \param[in] label Name/label associated with data
     * \return Helios data type of primitive data
     * \sa HeliosDataType
     */
    HeliosDataType getPrimitiveDataType( const char* label ) const;
    
    //! Get the size/length of primitive data
    /**
     * \param[in] label Name/label associated with data
     * \return Size/length of primitive data array
     */
    uint getPrimitiveDataSize( const char* label ) const;
    
    //! Check if primitive data 'label' exists
    /**
     * \param[in] label Name/label associated with data
     * \return True/false
     */
    bool doesPrimitiveDataExist( const char* label ) const;
    
    //! Clear the primitive data for this primitive
    /**
     * \param[in] label Name/label associated with data
     */
    void clearPrimitiveData( const char* label );
    
    //! Return labels for all primitive data for this particular primitive
    std::vector<std::string> listPrimitiveData() const;

    virtual void calculateSolidFraction( const std::map<std::string,Texture> &textures ) = 0;

    friend class Context;
    
protected:
    
    //! Unique universal identifier
    uint UUID;
    
    //! Type of primitive element (e.g., patch, triangle, etc.)
    PrimitiveType prim_type;
    
    //! Identifier of parent object (default is object 0)
    uint parent_object_ID;
    
    //! Diffuse RGB color
    helios::RGBAcolor color;
    
    //! Path to texture image
    std::string texturefile;
    //! Affine transformation matrix
    float transform[16];
    
    //! (u,v) texture coordinates
    std::vector<vec2> uv;
    
    //! fraction of surface area that is solid material (i.e., non-transparent)
    float solid_fraction;
    
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
    
    bool texturecoloroverridden;
    
};


//! Rectangular geometric object
/**
 * Position is given with respect to its center. Patches can only be added through the Context member method \ref Context::addPatch().
 * \image html doc/images/Patch.png "Sample image of a Patch." width=3cm
 * @private
 */
class Patch : public Primitive{
public:
    
    //! Patch constructor - colored by RGBcolor
    /**
     * \param[in] color RGB color of the Patch.
     * \param[in] parent_objID Identifier of the parent compound object.
     * \param[in] UUID Unique universal identifier.
     */
    Patch( const helios::RGBAcolor& color, uint parent_objID, uint UUID );
    
    //! Patch constructor - colored by texture map
    /**
     * \param[in] texturefile Path to texture image.
     * \param[in] parent_objID Identifier of the parent compound object.
     * \param[in] UUID Unique universal identifier.
     */
    Patch( const char* texturefile, float solid_fraction, uint parent_objID, uint UUID );
    
    //! Patch constructor - colored by texture map with custom (u,v) coordinates
    /**
     * \param[in] texturefile Path to texture image.
     * \param[in] uv Vector of (u,v) texture coordinates.
     * \param[in] parent_objID Identifier of the parent compound object.
     * \param[in] UUID Unique universal identifier.
     */
    Patch( const char* texturefile, const std::vector<helios::vec2>& uv, const std::map<std::string,Texture> &textures, uint parent_objID, uint UUID );
    
    //! Patch destructor
    ~Patch() override= default;
    
    //! Get the primitive surface area
    /**
     * \return Surface area of the Patch.
     */
    float getArea() const override;
    
    //! Get a unit vector normal to the primitive surface
    /**
     * \return Unit vector normal to the surface of the Patch.
     */
    helios::vec3 getNormal() const override;
    
    //! Method to return the (x,y,z) coordinates of the vertices of a Primitive
    /**
     * \return Vector containing four sets of the (x,y,z) coordinates of each vertex.
     */
    std::vector<helios::vec3> getVertices() const override;
    
    //! Get the size of the Patch in x- and y-directions
    /**
     * \return vec2 describing the length and width of the Patch.
     */
    helios::vec2 getSize() const;
    
    //! Get the (x,y,z) coordinates of the Patch center point
    /**
     * \return vec3 describing (x,y,z) coordinate of Patch center.
     */
    helios::vec3 getCenter() const override;
    
    //! Method to rotate a Primitive about the x-, y-, or z-axis
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_xyz_string Axis about which to rotate (must be one of x, y, z )
     */
    void rotate(float rotation_radians, const char* rotation_axis_xyz_string ) override;
    
    //! Method to rotate a Primitive about an arbitrary axis passing through the origin
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_vector Vector describing axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3& rotation_axis_vector ) override;
    
    //! Method to rotate a Primitive about an arbitrary line (not necessarily passing through the origin)
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis.
     * \param[in] rotation_axis_vector Vector describing the direction of the axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector ) override;

    //! Calculate the fraction of the Patch surface area that is solid (non-transparent)
    void calculateSolidFraction( const std::map<std::string,Texture> &textures ) override;
    
};

//! Triangular geometric primitive object
/**
 * A Triangle is specified by the positions of its three vertices. Triangles can only be added through the Context member method \ref Context::addTriangle().
 * \image html doc/images/Triangle.png "Sample image of a Triangle." width=3cm
 * @private
 */
class Triangle : public Primitive{
public:
    
    //! Triangle constructor - constant color
    /**
     * \param[in] vertex0 Cartesian coordinate of triangle vertex 0.
     * \param[in] vertex1 Cartesian coordinate of triangle vertex 1.
     * \param[in] vertex2 Cartesian coordinate of triangle vertex 2.
     * \param[in] color RGB color of the Triangle.
     * \param[in] parent_objID Identifier of the parent compound object.
     * \param[in] UUID Unique universal identifier.
     */
    Triangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const helios::RGBAcolor& color, uint parent_objID, uint UUID );

    //! Triangle constructor - texture map with known solid fraction
    /**
     * \param[in] vertex0 Cartesian coordinate of triangle vertex 0.
     * \param[in] vertex1 Cartesian coordinate of triangle vertex 1.
     * \param[in] vertex2 Cartesian coordinate of triangle vertex 2.
     * \param[in] texturefile Path to texture image.
     * \param[in] solid_fraction Fraction of the Triangle surface area that is solid material.
     * \param[in] parent_objID Identifier of the parent compound object.
     * \param[in] UUID Unique universal identifier.
     */
    Triangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const char* texturefile, const std::vector<helios::vec2>& uv, float solid_fraction, uint parent_objID, uint UUID );

    //! Triangle constructor - texture map with solid fraction to be computed
    /**
     * \param[in] vertex0 Cartesian coordinate of triangle vertex 0.
     * \param[in] vertex1 Cartesian coordinate of triangle vertex 1.
     * \param[in] vertex2 Cartesian coordinate of triangle vertex 2.
     * \param[in] texturefile Path to texture image.
     * \param[in] uv Vector of (u,v) texture coordinates.
     * \param[in] textures Map of texture objects.
     * \param[in] parent_objID Identifier of the parent compound object.
     * \param[in] UUID Unique universal identifier.
     */
    Triangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const char* texturefile, const std::vector<helios::vec2>& uv, const std::map<std::string,Texture> &textures, uint parent_objID, uint UUID );
    
    //! Triangle destructor
    ~Triangle() override= default;
    
    //! Get the primitive surface area
    /**
     * \return Surface area of the Triangle.
     */
    float getArea() const override;
    
    //! Get a unit vector normal to the primitive surface
    /**
     * \return Unit vector normal to the surface of the Triangle.
     */
    helios::vec3 getNormal() const override;
    
    //! Method to return the (x,y,z) coordinates of the vertices of a Primitve
    /**
     * \return Vector containing three sets of the (x,y,z) coordinates of each vertex.
     */
    std::vector<helios::vec3> getVertices() const override;
    
    //! Method to return the (x,y,z) coordinates of a given Triangle vertex
    /**
     * \param[in] vertex_index Triangle vertex (0, 1, or 2)
     * \return (x,y,z) coordinates of triangle vertex
     */
    helios::vec3 getVertex( int vertex_index );
    
    //! Method to return the (x,y,z) coordinates of a given Triangle's center (centroid)
    /**
     * \return (x,y,z) coordinates of triangle centroid
     */
    helios::vec3 getCenter() const override;
    
    //! Method to rotate a Primitive about the x-, y-, or z-axis
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_xyz_string Axis about which to rotate (must be one of x, y, z )
     */
    void rotate(float rotation_radians, const char* rotation_axis_xyz_string ) override;
    
    //! Method to rotate a Primitive about an arbitrary axis passing through the origin
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_vector Vector describing axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3& rotation_axis_vector ) override;
    
    //! Method to rotate a Primitive about an arbitrary line (not necessarily passing through the origin)
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis.
     * \param[in] rotation_axis_vector Vector describing the direction of the axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector ) override;

    //! Calculate the fraction of the Triangle surface area that is solid (non-transparent)
    void calculateSolidFraction( const std::map<std::string,Texture> &textures ) override;
    
private:
    
//    //!(x,y,z) coordinates of triangle vertex #0
//    helios::vec3 vertex0;
//
//    //!(x,y,z) coordinates of triangle vertex #1
//    helios::vec3 vertex1;
//
//    //!(x,y,z) coordinates of triangle vertex #2
//    helios::vec3 vertex2;
    
    void makeTransformationMatrix( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2 );

    bool edgeFunction(const helios::vec2 &a, const helios::vec2 &b, const helios::vec2 &c);
    
};

//! Parallelpiped geometric object filled with a participating medium
/** Position is given with respect to its center. Voxels can only be added through the Context member method \ref Context::addVoxel().
 @private
 */
class Voxel : public Primitive{
public:
    
    //! Voxel constructors
    /**
     * \param[in] color RGB color of the Voxel.
     * \param[in] parent_objID Identifier of the parent compound object.
     * \param[in] UUID Unique universal identifier.
     */
    Voxel( const helios::RGBAcolor& color, uint parent_objID, uint UUID );
    
    //! Voxel destructor
    ~Voxel() override= default;
    
    //! Get the primitive surface area
    /**
     * \return Surface area of the Voxel.
     */
    float getArea() const override;
    
    //! This method is not used for a Voxel
    helios::vec3 getNormal() const override;
    
    //! Method to return the (x,y,z) coordinates of the vertices of a Primitve
    /**
     * \return Vector containing eight sets of the (x,y,z) coordinates of each vertex.
     */
    std::vector<helios::vec3> getVertices() const override;
    
    //! Method to return the Volume of a Voxel
    /**
     * \return Volume of the Voxel.
     */
    float getVolume();
    
    //! Get the (x,y,z) coordinates of the Voxel center point
    /**
     * \return vec3 describing (x,y,z) coordinate of Voxel center.
     */
    helios::vec3 getCenter() const override;
    
    //! Get the size of the Voxel in x-, y-, and z-directions
    /**
     * \return vec3 describing the length, width and depth of the Voxel.
     */
    helios::vec3 getSize();
    
    //! Method to rotate a Primitive about the x-, y-, or z-axis
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_xyz_string Axis about which to rotate (must be one of x, y, z )
     */
    void rotate(float rotation_radians, const char* rotation_axis_xyz_string ) override;
    
    //! Method to rotate a Primitive about an arbitrary axis passing through the origin
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] rotation_axis_vector Vector describing axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3& rotation_axis_vector ) override;
    
    
    //! Method to rotate a Primitive about an arbitrary line (not necessarily passing through the origin)
    /**
     * \param[in] rotation_radians Rotation angle in radians.
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis.
     * \param[in] rotation_axis_vector Vector describing the direction of the axis about which to rotate.
     */
    void rotate(float rotation_radians, const helios::vec3 &origin, const helios::vec3 &rotation_axis_vector ) override;

    //! Calculate the fraction of the Voxel surface area that is solid (non-transparent)
    void calculateSolidFraction( const std::map<std::string,Texture> &textures ) override;
    
};

//! Class for parsing XML input files
class XMLparser{
public:

  //! Parse vector data of type 'float' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_float( const pugi::xml_node &node_data, std::vector<float> &data );

  //! Parse vector data of type 'double' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_double( const pugi::xml_node &node_data, std::vector<double> &data );

  //! Parse vector data of type 'int' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_int( const pugi::xml_node &node_data, std::vector<int> &data );

  //! Parse vector data of type 'uint' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_uint( const pugi::xml_node &node_data, std::vector<uint> &data );

  //! Parse vector data of type 'string' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_string( const pugi::xml_node &node_data, std::vector<std::string> &data );

  //! Parse vector data of type 'vec2' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_vec2( const pugi::xml_node &node_data, std::vector<vec2> &data );

  //! Parse vector data of type 'vec3' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_vec3( const pugi::xml_node &node_data, std::vector<vec3> &data );

  //! Parse vector data of type 'vac4' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_vec4( const pugi::xml_node &node_data, std::vector<vec4> &data );

  //! Parse vector data of type 'int2' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_int2( const pugi::xml_node &node_data, std::vector<int2> &data );

  //! Parse vector data of type 'int3' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_int3( const pugi::xml_node &node_data, std::vector<int3> &data );

  //! Parse vector data of type 'int4' within an XML tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] data Vector data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_data_int4( const pugi::xml_node &node_data, std::vector<int4> &data );

  //! Parse the value within an \<objID> (object ID) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] objID Object ID value parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_objID( const pugi::xml_node &node_data, uint &objID );

  //! Parse the value within a \<transform> (transformation matrix) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] transform Transformation matrix data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly, 3 if data was parsed successfully but tag did not contain 16 values.
   */
  static int parse_transform( const pugi::xml_node &node_data, float (&transform)[16] );

  //! Parse the value within an \<texture> (path to image texture) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] texture Path to image texture file parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty.
   */
  static int parse_texture( const pugi::xml_node &node_data, std::string &texture );

  //! Parse the value within a \<textureUV> (texture coordinates) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] uvs (u,v) texture coordinate data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_textureUV( const pugi::xml_node &node_data, std::vector<vec2> &uvs );

  //! Parse the value within a \<solid_fraction> (primitive solid fraction) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] solid_fraction Primitive solid fraction value parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_solid_fraction( const pugi::xml_node &node_data, float &solid_fraction );

  //! Parse the value within a \<vertices> (primitive vertex coordinates) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] vertices (x,y,z) primitive vertex data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_vertices( const pugi::xml_node &node_data, std::vector<float> &vertices );

  //! Parse the value within a \<subdivisions> (object sub-primitive resolution) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] subdivisions Scalar (uint) subdivisions value parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_subdivisions( const pugi::xml_node &node_data, uint &subdivisions );

  //! Parse the value within a \<subdivisions> (object sub-primitive resolution) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] subdivisions 2D (vec2) subdivisions parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_subdivisions( const pugi::xml_node &node_data, int2 &subdivisions );

  //! Parse the value within a \<subdivisions> (object sub-primitive resolution) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] subdivisions 3D (vec3) subdivisions parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_subdivisions( const pugi::xml_node &node_data, int3 &subdivisions );

  //! Parse the value within a \<nodes> (coordinates defining nodes of a tube or cone) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] nodes Vector of node coordinate data parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_nodes( const pugi::xml_node &node_data, std::vector<vec3> &nodes );

  //! Parse the value within a \<radius> (radius at nodes of a tube or cone) tag
  /**
   * \param[in] node_data PUGI XML node for tag containing vector data
   * \param[out] radius Vector of radii at nodes parsed from XML node
   * \return 0 if parsing was successful, 1 if tag/node did not exist or was empty, 2 if tag/node contained invalid data that could not be converted properly.
   */
  static int parse_radius( const pugi::xml_node &node_data, std::vector<float> &radius );

};

//! Stores the state associated with simulation
/** The Context provides an interface to global information about the application environment.   It allows access to application-level operations such as adding geometry, running models, and visualization. After creation, the Context must first be initialized via a call to initializeContext(), after which geometry and models can be added and simulated.  
 */
class Context{
private:
    
    //---------- PRIMITIVE/OBJECT HELIOS::VECTORS ----------------//
    
    //!Get a pointer to a Primitive element from the Context
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive element
     * @private
     */
    Primitive* getPrimitivePointer_private( uint UUID ) const;
    
    //!Get a pointer to a Patch from the Context
    /**
     * \param[in] UUID Unique universal identifier (UUID) of Patch
     * @private
     */
    Patch* getPatchPointer_private( uint UUID ) const;
    
    
    //!Get a pointer to a Triangle from the Context
    /**
     * \param[in] UUID Unique universal identifier (UUID) of Triangle
     * @private
     */
    Triangle* getTrianglePointer_private( uint UUID ) const;
    
    
    //!Get a pointer to a Voxel from the Context
    /**
     * \param[in] UUID Unique universal identifier (UUID) of Voxel
     * @private
     */
    Voxel* getVoxelPointer_private( uint UUID ) const;
    
    //!Get a pointer to an Object from the Context
    /**
     * \param[in] ObjID ID of Object
     * @private
     */
    CompoundObject* getObjectPointer_private( uint ObjID ) const;
    
    //!Get a pointer to a Tile Object from the Context
    /**
     * \param[in] ObjID ID of Tile Object
     * @private
     */
    Tile* getTileObjectPointer_private( uint ObjID ) const;
    
    //!Get a pointer to a Sphere Object from the Context
    /**
     * \param[in] ObjID ID of Sphere Object
     * @private
     */
    Sphere* getSphereObjectPointer_private( uint ObjID ) const;
    
    //!Get a pointer to a Tube Object from the Context
    /**
     * \param[in] ObjID ID of Tube Object
     * @private
     */
    Tube* getTubeObjectPointer_private( uint ObjID ) const;
    
    //!Get a pointer to a Box Object from the Context
    /**
     * \param[in] ObjID ID of Box Object
     * @private
     */
    Box* getBoxObjectPointer_private( uint ObjID ) const;
    
    //!Get a pointer to a Disk Object from the Context
    /**
     * \param[in] ObjID ID of Disk Object
     * @private
     */
    Disk* getDiskObjectPointer_private( uint ObjID ) const;
    
    //!Get a pointer to a Polymesh Object from the Context
    /**
     * \param[in] ObjID ID of Polymesh Object
     * @private
     */
    Polymesh* getPolymeshObjectPointer_private( uint ObjID ) const;
    
    //!Get a pointer to a Cone Object from the Context
    /**
     * \param[in] ObjID ID of Cone Object
     * @private
     */
    Cone* getConeObjectPointer_private( uint ObjID ) const;
    
    
    //! Map containing a pointer to each primitive
    /** \note A Primitive's index in this map is its \ref UUID */
    std::map<uint,Primitive*> primitives;
    
    //! Map containing a pointer to each compound object
    std::map<uint,CompoundObject*> objects;
    
    //! Map containing data values for timeseries
    std::map<std::string, std::vector<float> > timeseries_data;
    
    //! Map containing floating point values corresponding to date/time in timeseries
    /** \note The date value is calculated as Julian_day + hour/24.f + minute/24.f/60.f */
    std::map<std::string, std::vector<double> > timeseries_datevalue;
    
    //------------ TEXTURES ----------------//
    
    std::map<std::string,Texture> textures;
    
    void addTexture( const char* texture_file );

    bool doesTextureFileExist(const char* texture_file ) const;
    
    //----------- GLOBAL DATA -------------//
    
    std::map<std::string, GlobalData> globaldata;
    
    //---------- CONTEXT PRIVATE MEMBER VARIABLES ---------//
    //NOTE: variables are initialized and documented in initializeContext() member method
    
    //! Simulation date (Date vector)
    /**
     * sa setDate(), getDate()
     */
    helios::Date sim_date;
    
    //! Simulation time (Time vector)
    /**
     \sa setTime(), getTime()
     */
    helios::Time sim_time;
    
    //! Random number generation engine
    std::minstd_rand0 generator;
    
    //! Random uniform distribution
    std::uniform_real_distribution<float> unif_distribution;
    
    //! Random normal distribution (mean = zero, std dev = 1)
    std::normal_distribution<float> norm_distribution;
    
    //---------- CONTEXT I/O ---------//
    
    std::vector<std::string> XMLfiles;

    struct OBJmaterial{

      RGBcolor color;
      std::string texture;
      uint materialID;
      bool textureHasTransparency = false;
      bool textureColorIsOverridden = false;

      OBJmaterial( const RGBcolor &a_color, std::string a_texture, uint a_materialID ) : color{a_color}, texture{std::move(a_texture)}, materialID{a_materialID} {};

    };
    
    static std::map<std::string,OBJmaterial> loadMTL(const std::string &filebase, const std::string &material_file );
    
    void loadPData( pugi::xml_node p, uint UUID );
    
    void loadOData( pugi::xml_node p, uint ID );
    
    void loadOsubPData( pugi::xml_node p, uint ID );
    
    void writeDataToXMLstream( const char* data_group, const std::vector<std::string> &data_labels, void* ptr, std::ofstream &outfile ) const;

    std::vector<std::string> generateTexturesFromColormap( const std::string &texturefile, const std::vector<RGBcolor> &colormap_data );

    std::vector<RGBcolor> generateColormap( const std::string &colormap, uint Ncolors );

    std::vector<RGBcolor> generateColormap( const std::vector<helios::RGBcolor> &ctable, const std::vector<float> &cfrac, uint Ncolors );


    //---------- CONTEXT INITIALIZATION FLAGS ---------//
    
    //! Flag indicating whether Context geometry has been modified
    bool isgeometrydirty;
    
    uint currentUUID;
    
    uint currentObjectID;
    
public:
    
  //! Context default constructor
    Context();
    
    //! Context destructor
    ~Context();

    //! Deleted copy constructor to prevent copying of Context
    Context( const Context& ) = delete;

    //! Deleted assignment operator to prevent copying of Context
    void operator=( const Context& ) = delete;
    
    //! Run a self-test of the Context. The Context self-test runs through validation checks of Context-related methods to ensure they are working properly.
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest();

    //! Set seed for random generator
    /**
     * \param[in] seed uint used to seed the generator
     */
    void seedRandomGenerator(uint seed);

    //! Get the random number generator engine
    /**
     * \return std::minstd_rand0 random number generator engine
     */
    std::minstd_rand0* getRandomGenerator();

    //! Mark the Context geometry as "clean", meaning that the geometry has not been modified since last set as clean
    /**
     * \sa markGeometryDirty()
     * \sa isGeometryDirty()
     */
    void markGeometryClean();
    
    //! Mark the Context geometry as "dirty", meaning that the geometry has been modified since last set as clean
    /**
     * \sa markGeometryClean()
     * \sa isGeometryDirty()
     */
    void markGeometryDirty();
    
    //! Query whether the Context geometry is "dirty", meaning has the geometry been modified since last set as clean
    /**
     * \sa markGeometryClean()
     * \sa markGeometryDirty()
     */
    bool isGeometryDirty() const;
    
    //! Add new default Patch geometric primitive, which is centered at the origin (0,0,0), has unit length and width, horizontal orientation, and black color
    /** Method to add a new default Patch to the Context
     * \ingroup primitives
     */
    uint addPatch();
    
    //! Add new Patch geometric primitive
    /** Method to add a new Patch to the Context given its center, and size.
     * \param[in] center 3D coordinates of Patch center
     * \param[in] size width and length of Patch
     * \return UUID of Patch
     * \note Assumes that patch is horizontal.
     * \ingroup primitives
     */
    uint addPatch( const helios::vec3& center, const helios::vec2& size );
    
    //! Add new Patch geometric primitive
    /** Method to add a new Patch to the Context given its center, size, and spherical rotation.
     * \param[in] center 3D coordinates of Patch center
     * \param[in] size width and length of Patch
     * \param[in] rotation Spherical rotation
     * \return UUID of Patch
     * \note Assumes that patch is horizontal.
     * \ingroup primitives
     */
    uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation );
    
    //! Add new Patch geometric primitive
    /** Method to add a new Patch to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
     * \param[in] center 3D coordinates of Patch center
     * \param[in] size width and length of Patch
     * \param[in] rotation Spherical rotation
     * \param[in] color diffuse R-G-B color of Patch
     * \return UUID of Patch
     */
    uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBcolor& color );
    
    //! Add new Patch geometric primitive
    /** Method to add a new Patch to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
     * \param[in] center 3D coordinates of Patch center
     * \param[in] size width and length of Patch
     * \param[in] rotation Spherical rotation
     * \param[in] color diffuse R-G-B-A color of Patch\return UUID of Patch
     */
    uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBAcolor& color );
    
    //! Add new Patch geometric primitive
    /** Method to add a new Patch to the Context given its center, size, spherical rotation, and a texture map handle.
     * \param[in] center 3D coordinates of Patch center
     * \param[in] size width and length of Patch
     * \param[in] rotation Spherical rotation
     * \param[in] texture_file path to image file (JPEG or PNG) to be used as texture
     * \return UUID of Patch
     * \ingroup primitives
     */
    uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file );
    
    //! Add new Patch geometric primitive
    /** Method to add a new Patch to the Context given its center, size, spherical rotation, and a texture map handle.
     * \param[in] center 3D coordinates of Patch center
     * \param[in] size width and length of Patch
     * \param[in] rotation Spherical rotation
     * \param[in] texture_file path to image file (JPEG or PNG) to be used as texture
     * \param[in] uv_center u-v coordinates of the center of texture map
     * \param[in] uv_size size of the texture in u-v coordinates
     * \return UUID of Patch
     * \ingroup primitives
     */
    uint addPatch( const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file, const helios::vec2& uv_center, const helios::vec2& uv_size );
    
    //! Add new Triangle geometric primitive
    /** Method to add a new Triangle to the Context given the (x,y,z) coordinates of its vertices.
     * \param[in] vertex0 3D coordinate of Triangle vertex #0
     * \param[in] vertex1 3D coordinate of Triangle vertex #1
     * \param[in] vertex2 3D coordinate of Triangle vertex #2
     * \return UUID of Triangle
     * \ingroup primitives
     */
    uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2 );
    
    //! Add new Triangle geometric primitive
    /** Method to add a new Triangle to the Context given its the (x,y,z) coordinates of its vertices and diffuse RGBcolor.
     * \param[in] vertex0 3D coordinate of Triangle vertex #0
     * \param[in] vertex1 3D coordinate of Triangle vertex #1
     * \param[in] vertex2 3D coordinate of Triangle vertex #2
     * \param[in] color diffuse R-G-B color of Triangle
     * \return UUID of Triangle
     * \ingroup primitives
     */
    uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const helios::RGBcolor& color );
    
    //! Add new Triangle geometric primitive
    /** Method to add a new Triangle to the Context given its the (x,y,z) coordinates of its vertices and diffuse RGBAcolor.
     * \param[in] vertex0 3D coordinate of Triangle vertex #0
     * \param[in] vertex1 3D coordinate of Triangle vertex #1
     * \param[in] vertex2 3D coordinate of Triangle vertex #2
     * \param[in] color diffuse R-G-B-A color of Triangle
     * \return UUID of Triangle
     * \ingroup primitives
     */
    uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const helios::RGBAcolor& color );
    
    //! Add new Triangle geometric primitive
    /** Method to add a new Triangle to the Context given its the (x,y,z) coordinates of its vertices and diffuse RGBcolor.
     * \param[in] vertex0 3D coordinate of Triangle vertex #0
     * \param[in] vertex1 3D coordinate of Triangle vertex #1
     * \param[in] vertex2 3D coordinate of Triangle vertex #2
     * \param[in] texture_file path to image file (JPEG or PNG) to be used as texture
     * \param[in] uv0 u-v texture coordinates for vertex0
     * \param[in] uv1 u-v texture coordinates for vertex1
     * \param[in] uv2 u-v texture coordinates for vertex2
     * \return UUID of Triangle
     * \note Assumes a default color of black.
     * \ingroup primitives
     */
    uint addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const char* texture_file, const helios::vec2& uv0, const helios::vec2& uv1, const helios::vec2& uv2 );
    
    //! Add new Voxel geometric primitive
    /** Method to add a new Voxel to the Context given its center, and size.
     * \param[in] center 3D coordinates of Voxel center
     * \param[in] size width, length, and height of Voxel
     * \return UUID of Voxel
     * \note Assumes that voxel is horizontal.
     * \ingroup primitives
     */
    uint addVoxel( const helios::vec3& center, const helios::vec3& size );
    
    //! Add new Voxel geometric primitive
    /** Method to add a new Voxel to the Context given its center, size, and spherical rotation.
     * \param[in] center 3D coordinates of Voxel center
     * \param[in] size width, length, and height of Voxel
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Voxel
     * \return UUID of Voxel
     * \note Assumes a default color of black.
     * \ingroup primitives
     */
    uint addVoxel( const helios::vec3& center, const helios::vec3& size, const float& rotation );
    
    //! Add new Voxel geometric primitive
    /** Method to add a new Voxel to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
     * \param[in] center 3D coordinates of Voxel center
     * \param[in] size width, length, and height of Voxel
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Voxel
     * \param[in] color diffuse R-G-B color of Voxel
     * \return UUID of Voxel
     */
    uint addVoxel( const helios::vec3& center, const helios::vec3& size, const float& rotation, const helios::RGBcolor& color );
    
    //! Add new Voxel geometric primitive
    /** Method to add a new Voxel to the Context given its center, size, spherical rotation, and diffuse RGBAcolor.
     * \param[in] center 3D coordinates of Voxel center
     * \param[in] size width, length, and height of Voxel
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Voxel
     * \param[in] color diffuse R-G-B-A color of Voxel
     * \return UUID of Voxel
     */
    uint addVoxel( const helios::vec3& center, const helios::vec3& size, const float& rotation, const helios::RGBAcolor& color );
    
    //! Translate a primitive using its UUID
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be translated
     * \param[in] shift Distance to translate in (x,y,z) directions
     */
    void translatePrimitive( uint UUID, const vec3& shift );
    
    //! Translate a group of primitives using a vector of UUIDs
    /**
     * \param[in] UUID Vector of unique universal identifiers (UUIDs) of primitives to be translated
     * \param[in] shift Distance to translate in (x,y,z) directions
     */
    void translatePrimitive( const std::vector<uint>& UUIDs, const vec3& shift );
    
    //! Rotate a primitive about the x, y, or z axis using its UUID
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be translated
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Axis about which to rotate (must be one of x, y, z )
     */
    void rotatePrimitive( uint UUID, float rot, const char* axis );
    
    //! Rotate a group of primitives about the x, y, or z axis using a vector of UUIDs
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be translated
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Axis about which to rotate (must be one of x, y, z )
     */
    void rotatePrimitive( const std::vector<uint>& UUIDs, float rot, const char* axis );
    
    //! Rotate a primitive about an arbitrary axis passing through the origin using its UUID
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be translated
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Vector describing axis about which to rotate
     */
    void rotatePrimitive( uint UUID, float rot, const helios::vec3& axis );
    
    //! Rotate a group of primitives about an arbitrary axis passing through the origin using a vector of UUIDs
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be translated
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Vector describing axis about which to rotate
     */
    void rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const vec3 &axis );
    
    //! Rotate a primitive about an arbitrary line (not necessarily passing through the origin) using its UUID
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be translated
     * \param[in] rot Rotation angle in radians
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis
     * \param[in] axis Vector describing axis about which to rotate
     */
    void rotatePrimitive( uint UUID, float rot, const helios::vec3& origin, const helios::vec3& axis );
    
    //! Rotate a group of primitives about an arbitrary line (not necessarily passing through the origin) using a vector of UUIDs
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be translated
     * \param[in] rot Rotation angle in radians
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis
     * \param[in] axis Vector describing axis about which to (rotate)
     */
    void rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const helios::vec3& origin, const vec3 &axis );
    
    //! Scale a primitive using its UUID
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be scaled
     * \param[in] S Scaling factor
     */
    void scalePrimitive( uint UUID, const helios::vec3& S );
    
    //! Scale a group of primitives using a vector of UUIDs
    /**
     * \param[in] UUID Vector of unique universal identifiers (UUIDs) of primitives to be scaled
     * \param[in] S Scaling factor
     */
    void scalePrimitive( const std::vector<uint>& UUIDs, const helios::vec3& S );
    
    //! Delete a single primitive from the context
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be deleted
     */
    void deletePrimitive( uint UUID );
    
    //! Delete a group of primitives from the context
    /**
     * \param[in] UUIDs Vector of unique universal identifiers (UUIDs) of primitives to be deleted
     */
    void deletePrimitive( const std::vector<uint>& UUIDs );
    
    //! Make a copy of a primitive from the context
    /**
     * \param[in] UUID Unique universal identifier (UUID) of primitive to be copied
     * \return UUID for copied primitive
     */
    uint copyPrimitive(uint UUID );
    
    //! Make a copy of a group of primitives from the context
    /**
     * \param[in] UUIDs Vector of unique universal identifiers (UUIDs) of primitive to be copied
     * \return UUIDs for copied primitives
     */
    std::vector<uint> copyPrimitive(const std::vector<uint> &UUIDs );
    
    //! copy all primitive data from one primitive to another
    /**
     * \param[in] UUID unique universal identifier (UUID) of primitive that is the source of data for copying
     * \param[in] currentUUID unique universal identifier (UUID) of primitive that is the destination for data copying
     */
    void copyPrimitiveData( uint UUID, uint currentUUID);

    //! Rename primitive data for a primitive
    /**
     * \param[in] UUID unique universal identifier (UUID) of primitive that is the source of data for copying
     * \param[in] old_label old label of data to be renamed
     * \param[in] new_label new label of data to be renamed
     */
    void renamePrimitiveData( uint UUID, const char* old_label, const char* new_label );

    //! Duplicate/copy primitive data
    /**
     * \param[in] UUID unique universal identifier (UUID) of primitive that is the source of data for copying
     * \param[in] old_label old label of data to be copied
     * \param[in] new_label new label of data to be copied
     */
    void duplicatePrimitiveData( uint UUID, const char* old_label, const char* new_label );
    
    //! Check if primitive exists for a given UUID
    /**
     * \param[in] UUID Unique universal identifier of primitive element
     */
    bool doesPrimitiveExist( uint UUID ) const;

    //! Check if ALL primitives exists for a vector UUIDs
    /**
     * \param[in] UUIDs Vector of unique universal identifiers of primitive elements
     * \return true if all primitives exist, false if any do not exist
     */
    bool doesPrimitiveExist( const std::vector<uint> &UUIDs ) const;
    
    //! Get the size of a patch element
    /**
     * \param[in] UUID Unique universal identifier for patch.
     * \return Length x width of Patch element.
     * \note If the UUID passed to this method does not correspond to a Patch, an error will be thrown.
     */
    helios::vec2 getPatchSize( uint UUID ) const;
    
    
    //! Get the Cartesian (x,y,z) center position of a patch element
    /**
     * \param[in] UUID Unique universal identifier for patch.
     * \return Center position of Patch element.
     * \note If the UUID passed to this method does not correspond to a Patch, an error will be thrown.
     */
    helios::vec3 getPatchCenter( uint UUID ) const;
    
    //! Get a single vertex of a Triangle based on an index
    /**
     * \param[in] UUID Universal unique identifier of Triangle element.
     * \param[in] number Index of vertex (0, 1, or 2)
     * \return Cartesian (x,y,z) coordinate of triangle vertices indexed at "vertex"
     * \note If the UUID passed to this method does not correspond to a Triangle, an error will be thrown.
     */
    helios::vec3 getTriangleVertex( uint UUID, uint number ) const;
    
    //! Get the Cartesian (x,y,z) center position of a voxel element
    /**
     * \param[in] UUID Unique universal identifier for voxel.
     * \return Center position of voxel element.
     * \note If the UUID passed to this method does not correspond to a voxel, an error will be thrown.
     */
    helios::vec3 getVoxelCenter( uint UUID ) const;
    
    //! Get the size of a voxel element
    /**
     * \param[in] UUID Unique universal identifier for voxel.
     * \return Length x width x height of voxel element.
     * \note If the UUID passed to this method does not correspond to a voxel, an error will be thrown.
     */
    helios::vec3 getVoxelSize( uint UUID ) const;
    
    //!Get the total number of Primitives in the Context
    /**
     * \ingroup primitives
     */
    uint getPrimitiveCount() const;
    
    //!Get all primitive UUIDs currently in the Context
    std::vector<uint> getAllUUIDs() const;

    //! Delete UUIDs from vector if primitives no longer exist (1D vector)
    /**
     * \param[inout] UUIDs Vector of primitive UUIDs. UUIDs for primitives that do not exist will be deleted from the vector.
     */
    void cleanDeletedUUIDs( std::vector<uint> &UUIDs ) const;

    //! Delete UUIDs from vector if primitives no longer exist (2D vector)
    /**
     * \param[inout] UUIDs Vector of primitive UUIDs. UUIDs for primitives that do not exist will be deleted from the vector.
     */
    void cleanDeletedUUIDs( std::vector<std::vector<uint>> &UUIDs ) const;

    //! Delete UUIDs from vector if primitives no longer exist (3D vector)
    /**
     * \param[inout] UUIDs Vector of primitive UUIDs. UUIDs for primitives that do not exist will be deleted from the vector.
     */
    void cleanDeletedUUIDs( std::vector<std::vector<std::vector<uint>>> &UUIDs ) const;
    
    //-------- Primitive Data Methods ---------- //
    
    //! Add data value (int) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const int& data );
    
    //! Add data value (uint) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const uint& data );
    
    //! Add data value (float) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const float& data );
    
    //! Add data value (double) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const uint& UUID, const char* label, const std::string& data );
    
    //! Add data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[in] type Helios data type of primitive data (see \ref HeliosDataType)
     * \param[in] size Number of data elements
     * \param[in] data Pointer to primitive data
     */
    void setPrimitiveData( const uint& UUIDs, const char* label, HeliosDataType type, uint size, void* data );
    
    //! Add data value (int) associated with a primitive element
    /**
     * \param[in] UUIDs Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const int& data );
    
    //! Add data value (uint) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const uint& data );
    
    //! Add data value (float) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const float& data );
    
    //! Add data value (double) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const std::string& data );
    
    //! Add data value (int) associated with a primitive element
    /**
     * \param[in] UUIDs Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const int& data );
    
    //! Add data value (uint) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const uint& data );
    
    //! Add data value (float) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const float& data );
    
    //! Add data value (double) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const std::string& data );
    
    //! Add data value (int) associated with a primitive element
    /**
     * \param[in] UUIDs Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const int& data );
    
    //! Add data value (uint) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const uint& data );
    
    //! Add data value (float) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const float& data );
    
    //! Add data value (double) associated with a primitive element
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a primitive element
    /**
     * \param[in] UUID Vector of unique universal identifiers of Primitive elements
     * \param[in] label Name/label associated with data
     * \param[in] data Primitive data value (scalar)
     */
    void setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const std::string& data );
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar integer)
     */
    void getPrimitiveData( uint UUID, const char* label, int& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of integers)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<int>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar unsigned integer)
     */
    void getPrimitiveData( uint UUID, const char* label, uint& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of unsigned integers)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<uint>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar float)
     */
    void getPrimitiveData( uint UUID, const char* label, float& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of floats)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<float>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar double)
     */
    void getPrimitiveData( uint UUID, const char* label, double& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of doubles)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<double>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar vec2)
     */
    void getPrimitiveData( uint UUID, const char* label, vec2& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of vec2's)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<vec2>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar vec3)
     */
    void getPrimitiveData( uint UUID, const char* label, vec3& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of vec3's)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<vec3>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar vec4)
     */
    void getPrimitiveData( uint UUID, const char* label, vec4& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of vec4's)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<vec4>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar int2)
     */
    void getPrimitiveData( uint UUID, const char* label, int2& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of int2's)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<int2>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar int3)
     */
    void getPrimitiveData( uint UUID, const char* label, int3& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of int3's)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<int3>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar int4)
     */
    void getPrimitiveData( uint UUID, const char* label, int4& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of int4's)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<int4>& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (scalar string)
     */
    void getPrimitiveData( uint UUID, const char* label, std::string& data ) const;
    
    //! Get data associated with a primitive element
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     * \param[out] data Primitive data structure (vector of strings)
     */
    void getPrimitiveData( uint UUID, const char* label, std::vector<std::string>& data ) const;
    
    //! Get the Helios data type of primitive data
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     \return Helios data type of primitive data
     \sa HeliosDataType
     */
    HeliosDataType getPrimitiveDataType( uint UUID, const char* label ) const;
    
    //! Get the size/length of primitive data
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     \return Size/length of primitive data array
     */
    uint getPrimitiveDataSize( uint UUID, const char* label ) const;
    
    //! Check if primitive data 'label' exists
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     \return True/false
     */
    bool doesPrimitiveDataExist( uint UUID, const char* label ) const;
    
    //! Clear primitive data for a single primitive based on its UUID
    /**
     * \param[in] UUID Unique universal identifier of Primitive element
     * \param[in] label Name/label associated with data
     */
    void clearPrimitiveData( uint UUID, const char* label );
    
    //! Clear primitive data for multiple primitives based on a vector of UUIDs
    /**
     * \param[in] UUIDs Vector of unique universal identifiers for Primitive elements
     * \param[in] label Name/label associated with data
     */
    void clearPrimitiveData( const std::vector<uint>& UUIDs, const char* label );
    
    //! Method to get the Primitive type
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     * sa \ref PrimitiveType
     */
    PrimitiveType getPrimitiveType( uint UUID ) const;
    
    //! Method to set the ID of the parent object the primitive belongs to (default is object 0)
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     * \param[in] objID Identifier of primitive's parent object.
     */
    void setPrimitiveParentObjectID( uint UUID, uint objID );
    
    //! Method to set the ID of the parent object the primitive belongs to (default is object 0) for a vector of UUIDs
    /**
     * \param[in] UUIDs Vector of universal unique identifiers of primitives.
     * \param[in] objID Identifier of primitive's parent object.
     */
    void setPrimitiveParentObjectID( const std::vector<uint> &UUIDs, uint objID );
    
    //! Method to return the ID of the parent object the primitive belongs to (default is object 0)
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    uint getPrimitiveParentObjectID( uint UUID  )const;
    
    
    //! Method to return unique parent object IDs for a vector of primitive UUIDs
    /**
     * \param[in] UUIDs Vector of universal unique identifiers of primitives.
     */
    std::vector<uint> getUniquePrimitiveParentObjectIDs(std::vector<uint> UUIDs) const;
    
    
    //! Method to return unique parent object IDs for a vector of primitive UUIDs
    /**
     * \param[in] UUIDs Vector of universal unique identifiers of primitives.
     */
    std::vector<uint> getUniquePrimitiveParentObjectIDs(std::vector<uint> UUIDs, bool include_ObjID_zero) const;
    
    
    //! Method to return the surface area of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    float getPrimitiveArea( uint UUID ) const;

    //! Get the axis-aligned bounding box for a single primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive
     * \param[out] min_corner (x,y,z) coordinate of the bounding box corner in the -x, -y and -z direction.
     * \param[out] max_corner (x,y,z) coordinate of the bounding box corner in the +x, +y and +z direction.
     */
    void getPrimitiveBoundingBox( uint UUID, vec3 &min_corner, vec3 &max_corner ) const;

    //! Get the axis-aligned bounding box for a group of primitives
    /**
     * \param[in] UUIDs Vector of universal unique identifiers of primitives
     * \param[out] min_corner (x,y,z) coordinate of the bounding box corner in the -x, -y and -z direction.
     * \param[out] max_corner (x,y,z) coordinate of the bounding box corner in the +x, +y and +z direction.
     */
    void getPrimitiveBoundingBox( const std::vector<uint> &UUID, vec3 &min_corner, vec3 &max_corner ) const;
    
    //! Method to return the one-sided surface area of an object
    /**
     * \param[in] ObjID Identifier of the object.
     */
    float getObjectArea( uint ObjID ) const;
    
    //! Method to return the number of primitives contained in the object
    /**
     * \param[in] ObjID Identifier of the object.
     */
    uint getObjectPrimitiveCount( uint ObjID ) const;
    
    //! Method to return the Cartesian (x,y,z) point of the center of a bounding box for the object
    /**
     * \param[in] ObjID Identifier of the object.
     */
    helios::vec3 getObjectCenter( uint ObjID ) const;

    //! Method to set the diffuse color of an Object
    /**
     * \param[in] ObjID Universal unique identifier of object.
     * \param[in] color New color of object
     */
    void setObjectColor( uint ObjID, const helios::RGBcolor& color );

    //! Method to set the diffuse color of an Object for a vector of ObjIDs
    /**
     * \param[in] ObjIDs Vector of identifiers of object.
     * \param[in] color New color of object
     */
    void setObjectColor( const std::vector<uint> &ObjIDs, const helios::RGBcolor& color );

    //! Method to set the diffuse color of an Object with transparency
    /**
     * \param[in] ObjID Identifier of object.
     * \param[in] color New color of object
     */
    void setObjectColor( uint ObjID, const helios::RGBAcolor& color );

    //! Method to set the diffuse color of an Object with transparency for a vector of ObjIDs
    /**
     * \param[in] ObjIDs Vector of identifiers of objects.
     * \param[in] color New color of object
     */
    void setObjectColor( const std::vector<uint> &ObjIDs, const helios::RGBAcolor& color );
    
    //! Method to return the texture map file of an Object
    /**
     * \param[in] ObjID Identifier of the object.
     */
    std::string getObjectTextureFile( uint ObjID ) const;
    
    //! Method to return the Affine transformation matrix of an Object
    /**
     * \param[in] ObjID Identifier of the object.
     * \param[out] T 1D vector corresponding to Primitive transformation matrix
     */
    void getObjectTransformationMatrix( uint ObjID, float (&T)[16] ) const;
    
    //! Method to set the Affine transformation matrix of an Object
    /**
     * \param[in] ObjID Identifier of the object.
     * \param[in] T 1D vector corresponding to Primitive transformation matrix
     */
    void setObjectTransformationMatrix( uint ObjID, float (&T)[16] );
    
    //! Method to set the Affine transformation matrix of an Object for a vector Object IDs
    /**
     * \param[in] ObjIDs Vector of identifiers of the objects.
     * \param[in] T 1D vector corresponding to Primitive transformation matrix
     */
    void setObjectTransformationMatrix( const std::vector<uint> &ObjIDs, float (&T)[16] );
    
    //! Method to check whether an Object has texture data
    /**
     * \param[in] ObjID Identifier of the object.
     */
    bool objectHasTexture( uint ObjID ) const;
    
    //! Method to check if an Object contains a Primitive
    /**
     * \param[in] ObjID Identifier of the object.
     * \param[in] UUID Universal unique identifier of primitive.
     */
    bool doesObjectContainPrimitive(uint ObjID, uint UUID );
    
    //! Override the color in the texture map for all primitives in the Compound Object, in which case the primitives will be colored by the constant RGB color, but will apply the transparency channel in the texture to determine its shape
    /**
     * \param[in] ObjID Identifier of the object.
     */
    void overrideObjectTextureColor( uint ObjID );

    //! Override the color in the texture map for all primitives in the Compound Object, in which case the primitives will be colored by the constant RGB color, but will apply the transparency channel in the texture to determine its shape
    /**
     * \param[in] ObjIDs Vector of object identifiers.
     */
    void overrideObjectTextureColor( const std::vector<uint> &ObjIDs );

    //! For all primitives in the Compound Object, use the texture map to color the primitives rather than the constant RGB color. This is function reverses a previous call to overrideTextureColor(). Note that using the texture color is the default behavior.
    /**
     * \param[in] ObjID Identifier of the object.
     */
    void useObjectTextureColor( uint ObjID );

    //! For all primitives in the Compound Object, use the texture map to color the primitives rather than the constant RGB color. This is function reverses a previous call to overrideTextureColor(). Note that using the texture color is the default behavior.
    /**
     * \param[in] ObjIDs Vector of object identifiers.
     */
    void useObjectTextureColor( const std::vector<uint> &ObjIDs );

    //! Get the axis-aligned bounding box for a single object
    /**
     * \param[in] ObjID Identifier of the object.
     * \param[out] min_corner (x,y,z) coordinate of the bounding box corner in the -x, -y and -z direction.
     * \param[out] max_corner (x,y,z) coordinate of the bounding box corner in the +x, +y and +z direction.
     */
    void getObjectBoundingBox( uint ObjID, vec3 &min_corner, vec3 &max_corner ) const;

    //! Get the axis-aligned bounding box for a group of objects
    /**
     * \param[in] ObjIDs Vector of object identifiers.
     * \param[out] min_corner (x,y,z) coordinate of the bounding box corner in the -x, -y and -z direction.
     * \param[out] max_corner (x,y,z) coordinate of the bounding box corner in the +x, +y and +z direction.
     */
    void getObjectBoundingBox( const std::vector<uint> &ObjID, vec3 &min_corner, vec3 &max_corner ) const;
    
    //! Prints object properties to console (useful for debugging purposes)
    /**
     * \param[in] ObjID Object ID of the object that's information will be printed'.
     */
    void printObjectInfo(uint ObjID) const;
    
    //! Return labels for all object data for this particular object
    std::vector<std::string> listObjectData(uint ObjID) const;
    
    //! Return labels for all primitive data for this particular primitive
    std::vector<std::string> listPrimitiveData(uint UUID) const;
    
    //! Get fraction of primitive surface area that is non-transparent
    /**
     * \param[in] UUID Universal unique identifier for primitive.
     * \return Fraction of non-transparent area (=1 if primitive does not have a semi-transparent texture).
     */
    float getPrimitiveSolidFraction( uint UUID ) const;
    
    //! Method to return the normal vector of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    helios::vec3 getPrimitiveNormal( uint UUID ) const;
    
    //! Method to return the Affine transformation matrix of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     * \param[out] T 1D vector corresponding to Primitive transformation matrix
     */
    void getPrimitiveTransformationMatrix( uint UUID, float (&T)[16] ) const;
    
    //! Method to set the Affine transformation matrix of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     * \param[in] T 1D vector corresponding to Primitive transformation matrix
     */
    void setPrimitiveTransformationMatrix( uint UUID, float (&T)[16] );
    
    //! Method to set the Affine transformation matrix of a Primitive for a vector UUIDs
    /**
     * \param[in] UUIDs Vector of universal unique identifiers of primitives.
     * \param[in] T 1D vector corresponding to Primitive transformation matrix
     */
    void setPrimitiveTransformationMatrix( const std::vector<uint> &UUIDs, float (&T)[16] );
    
    //! Method to return the (x,y,z) coordinates of the vertices of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    std::vector<helios::vec3> getPrimitiveVertices( uint UUID ) const;
    
    //! Method to return the diffuse color of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    helios::RGBcolor getPrimitiveColor( uint UUID ) const;
    
    //! Method to return the diffuse color of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    helios::RGBcolor getPrimitiveColorRGB( uint UUID ) const;
    
    //! Method to return the diffuse color of a Primitive with transparency
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    helios::RGBAcolor getPrimitiveColorRGBA( uint UUID ) const;
    
    //! Method to set the diffuse color of a Primitive
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     * \param[in] color New color of primitive
     */
    void setPrimitiveColor( uint UUID, const helios::RGBcolor& color );
    
    //! Method to set the diffuse color of a Primitive for a vector of UUIDs
    /**
     * \param[in] UUIDs Vector of universal unique identifiers of primitives.
     * \param[in] color New color of primitive
     */
    void setPrimitiveColor( const std::vector<uint> &UUIDs, const helios::RGBcolor& color );
    
    //! Method to set the diffuse color of a Primitive with transparency
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     * \param[in] color New color of primitive
     */
    void setPrimitiveColor( uint UUID, const helios::RGBAcolor& color );
    
    //! Method to set the diffuse color of a Primitive with transparency for a vector of UUIDs
    /**
     * \param[in] UUIDs Vector of universal unique identifiers of primitives.
     * \param[in] color New color of primitive
     */
    void setPrimitiveColor( const std::vector<uint> &UUIDs, const helios::RGBAcolor& color );
    
    //! Get the path to texture map file for primitive. If primitive does not have a texture map, the result will be an empty string.
    /**
     * \param[in] UUID Unique universal identifier of primitive to be queried
     * @return Path to texture map file.
     */
    std::string getPrimitiveTextureFile( uint UUID ) const;


    //! Set the texture map file for a primitive
    /**
     * \param[in] UUID Unique universal identifier of primitive to be queried.
     * \param[in] texturefile Path to texture image file.
     */
    void setPrimitiveTextureFile( uint UUID, const std::string &texturefile );
    
    //! Get the size (number of pixels) of primitive texture map image.
    /**
     * \param[in] UUID Unique universal identifier of primitive to be queried
     * @return Texture image resolution (columns x rows).
     */
    helios::int2 getPrimitiveTextureSize( uint UUID ) const;
    
    //! Get u-v texture coordinates at primitive vertices
    /**
     * \param[in] UUID Unique universal identifier of primitive to be queried
     */
    std::vector<vec2> getPrimitiveTextureUV( uint UUID ) const;
    
    //! Check if primitive texture map has a transparency channel
    /**
     * \param[in] UUID Unique universal identifier of primitive to be queried
     * \return True if transparency channel data exists, false otherwise
     */
    bool primitiveTextureHasTransparencyChannel(uint UUID ) const;
    
    //! Get the transparency channel pixel data from primitive texture map. If transparency channel does not exist, an error will be thrown.
    /**
     * \param[in] UUID Unique universal identifier of primitive to be queried
     * \return Transparency value (0 or 1) for each pixel in primitive texture map.
     */
    const std::vector<std::vector<bool>> * getPrimitiveTextureTransparencyData(uint UUID) const;
    
    //! Override the color in the texture map for all primitives in the Compound Object, in which case the primitives will be colored by the constant RGB color, but will apply the transparency channel in the texture to determine its shape
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    void overridePrimitiveTextureColor( uint UUID );

    //! Override the color in the texture map for multiple primitives, in which case the primitives will be colored by the constant RGB color, but will apply the transparency channel in the texture to determine its shape
    /**
     * \param[in] UUIDs Vector of universal unique identifier of primitives.
     */
    void overridePrimitiveTextureColor( const std::vector<uint> &UUIDs );
    
    //! Use the texture map to color the primitive rather than the constant RGB color. This is function reverses a previous call to overrideTextureColor(). Note that using the texture color is the default behavior.
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    void usePrimitiveTextureColor( uint UUID );

    //! Use the texture map to color the primitives rather than the constant RGB color. This is function reverses a previous call to overrideTextureColor(). Note that using the texture color is the default behavior.
    /**
     * \param[in] UUIDs Vector of universal unique identifier of primitives.
     */
    void usePrimitiveTextureColor( const std::vector<uint> &UUIDs );

    //! Check if color of texture map is overridden by the diffuse R-G-B color of the primitive
    /**
     * \param[in] UUID Unique universal identifier of primitive to be queried
     */
    bool isPrimitiveTextureColorOverridden( uint UUID ) const;
    
    //! Prints primitive properties to console (useful for debugging purposes)
    /**
     * \param[in] UUID Universal unique identifier of primitive.
     */
    void printPrimitiveInfo(uint UUID) const;
    
    //-------- Compound Object Data Methods ---------- //
    
    //! Add data value (int) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const int& data );
    
    //! Add data value (uint) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const uint& data );
    
    //! Add data value (float) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const float& data );
    
    //! Add data value (double) associated with a compound object
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( uint objID, const char* label, const std::string& data );
    
    //! Add data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[in] type Helios data type of primitive data (see \ref HeliosDataType)
     * \param[in] size Number of data elements
     * \param[in] data Pointer to primitive data
     */
    void setObjectData( uint objIDs, const char* label, HeliosDataType type, uint size, void* data );
    
    //! Add data value (int) associated with a compound object
    /**
     * \param[in] objIDs Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const int& data );
    
    //! Add data value (uint) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const uint& data );
    
    //! Add data value (float) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const float& data );
    
    //! Add data value (double) associated with a compound object
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<uint>& objIDs, const char* label, const std::string& data );
    
    //! Add data value (int) associated with a compound object
    /**
     * \param[in] objIDs Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const int& data );
    
    //! Add data value (uint) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const uint& data );
    
    //! Add data value (float) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const float& data );
    
    //! Add data value (double) associated with a compound object
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const std::string& data );
    //! Add data value (int) associated with a compound object
    /**
     * \param[in] objIDs Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const int& data );
    
    //! Add data value (uint) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const uint& data );
    
    //! Add data value (float) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const float& data );
    
    //! Add data value (double) associated with a compound object
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const double& data );
    
    //! Add data value (vec2) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec2& data );
    
    //! Add data value (vec3) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec3& data );
    
    //! Add data value (vec4) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec4& data );
    
    //! Add data value (int2) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int2& data );
    
    //! Add data value (int3) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int3& data );
    
    //! Add data value (int4) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int4& data );
    
    //! Add data value (string) associated with a compound object
    /**
     * \param[in] objID Vector of unique universal identifiers of compound objects
     * \param[in] label Name/label associated with data
     * \param[in] data Object data value (scalar)
     */
    void setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const std::string& data );
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar integer)
     */
    void getObjectData( uint objID, const char* label, int& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of integers)
     */
    void getObjectData( uint objID, const char* label, std::vector<int>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar unsigned integer)
     */
    void getObjectData( uint objID, const char* label, uint& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of unsigned integers)
     */
    void getObjectData( uint objID, const char* label, std::vector<uint>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar float)
     */
    void getObjectData( uint objID, const char* label, float& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of floats)
     */
    void getObjectData( uint objID, const char* label, std::vector<float>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar double)
     */
    void getObjectData( uint objID, const char* label, double& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of doubles)
     */
    void getObjectData( uint objID, const char* label, std::vector<double>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar vec2)
     */
    void getObjectData( uint objID, const char* label, vec2& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of vec2's)
     */
    void getObjectData( uint objID, const char* label, std::vector<vec2>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar vec3)
     */
    void getObjectData( uint objID, const char* label, vec3& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of vec3's)
     */
    void getObjectData( uint objID, const char* label, std::vector<vec3>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar vec4)
     */
    void getObjectData( uint objID, const char* label, vec4& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of vec4's)
     */
    void getObjectData( uint objID, const char* label, std::vector<vec4>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar int2)
     */
    void getObjectData( uint objID, const char* label, int2& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of int2's)
     */
    void getObjectData( uint objID, const char* label, std::vector<int2>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar int3)
     */
    void getObjectData( uint objID, const char* label, int3& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of int3's)
     */
    void getObjectData( uint objID, const char* label, std::vector<int3>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar int4)
     */
    void getObjectData( uint objID, const char* label, int4& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of int4's)
     */
    void getObjectData( uint objID, const char* label, std::vector<int4>& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (scalar string)
     */
    void getObjectData( uint objID, const char* label, std::string& data ) const;
    
    //! Get data associated with a compound object
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     * \param[out] data Object data structure (vector of strings)
     */
    void getObjectData( uint objID, const char* label, std::vector<std::string>& data ) const;
    
    //! Get the Helios data type of primitive data
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     \return Helios data type of primitive data
     \sa HeliosDataType
     */
    HeliosDataType getObjectDataType( uint objID, const char* label ) const;
    
    //! Get the size/length of primitive data
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     \return Size/length of primitive data array
     */
    uint getObjectDataSize( uint objID, const char* label ) const;
    
    //! Check if primitive data 'label' exists
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     \return True/false
     */
    bool doesObjectDataExist( uint objID, const char* label ) const;
    
    //! Clear primitive data for a single primitive based on its objID
    /**
     * \param[in] objID Unique universal identifier of compound object
     * \param[in] label Name/label associated with data
     */
    void clearObjectData( uint objID, const char* label );
    
    //! Clear primitive data for multiple primitives based on a vector of objIDs
    /**
     * \param[in] objIDs Vector of unique universal identifiers for compound objects
     * \param[in] label Name/label associated with data
     */
    void clearObjectData( const std::vector<uint>& objIDs, const char* label );
    
    //! Method to query whether all object primitives are in tact
    /**
     * * \param[in] objID Object ID for object to be queried.
     * \return False if any primitives have been deleted from the object since creation; True otherwise.
     */
    bool areObjectPrimitivesComplete( uint objID ) const;

    //! Delete Object IDs from vector if objects no longer exist (1D vector)
    /**
     * \param[inout] objIDs Vector of object IDs. Object IDs for objects that do not exist will be deleted from the vector.
     */
    void cleanDeletedObjectIDs( std::vector<uint> &objIDs ) const;

    //! Delete Object IDs from vector if objects no longer exist (2D vector)
    /**
     * \param[inout] objIDs Vector of object IDs. Object IDs for objects that do not exist will be deleted from the vector.
     */
    void cleanDeletedObjectIDs( std::vector<std::vector<uint>> &objIDs ) const;

    //! Delete Object IDs from vector if objects no longer exist (3D vector)
    /**
     * \param[inout] objIDs Vector of object IDs. Object IDs for objects that do not exist will be deleted from the vector.
     */
    void cleanDeletedObjectIDs( std::vector<std::vector<std::vector<uint>>> &objIDs ) const;
    
    //-------- Global Data Methods ---------- //
    
    //! Add global data value (int)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const int& data );
    
    //! Add global data value (uint)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const uint& data );
    
    //! Add global data value (float)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const float& data );
    
    //! Add global data value (double)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const double& data );
    
    //! Add global data value (vec2)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const helios::vec2& data );
    
    //! Add global data value (vec3)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const helios::vec3& data );
    
    //! Add global data value (vec4)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const helios::vec4& data );
    
    //! Add global data value (int2)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const helios::int2& data );
    
    //! Add global data value (int3)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const helios::int3& data );
    
    //! Add global data value (int4)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    
    void setGlobalData( const char* label, const helios::int4& data );
    
    //! Add global data value (string)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, const std::string& data );
    
    //! Add global data value (any type)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] size Number of elements in global data
     * \param[in] data Global data value (scalar)
     */
    void setGlobalData( const char* label, HeliosDataType type, size_t size, void* data );

    //! Rename global data
    /**
     * \param[in] old_label Old name/label associated with data
     * \param[in] new_label New name/label associated with data
     */
    void renameGlobalData( const char* old_label, const char* new_label );

    //! Make a copy of global data
    /**
     * \param[in] old_label Old name/label associated with data
     * \param[in] new_label New name/label associated with data
     */
    void duplicateGlobalData(const char* old_label, const char* new_label );

    //! Delete/clear global data
    /**
     * \param[in] label Name/label associated with data
     */
    void clearGlobalData( const char* label );
    
    //! Get global data value (scalar integer)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar integer)
     */
    void getGlobalData( const char* label, int& data ) const;
    
    //! Get global data (array of integers)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of integers)
     */
    void getGlobalData( const char* label, std::vector<int>& data ) const;
    
    //! Get global data value (scalar uint)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar uint)
     */
    void getGlobalData( const char* label, uint& data ) const;
    
    //! Get global data (array of uint's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of uint's)
     */
    void getGlobalData( const char* label, std::vector<uint>& data ) const;
    
    //! Get global data value (scalar float)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar float)
     */
    void getGlobalData( const char* label, float& data ) const;
    
    //! Get global data (array of floats)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of floats)
     */
    void getGlobalData( const char* label, std::vector<float>& data ) const;
    
    //! Get global data value (scalar double)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar double)
     */
    void getGlobalData( const char* label, double& data ) const;
    
    //! Get global data (array of doubles)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of doubles)
     */
    void getGlobalData( const char* label, std::vector<double>& data ) const;
    
    //! Get global data value (scalar vec2)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar vec2)
     */
    void getGlobalData( const char* label, helios::vec2& data ) const;
    
    //! Get global data (array of vec2's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of vec2's)
     */
    void getGlobalData( const char* label, std::vector<helios::vec2>& data ) const;
    
    //! Get global data value (scalar vec3)
    /**
     * \param[in] label Name/label associated with data
     * \param[in] data Global data value (scalar vec3)
     */
    void getGlobalData( const char* label, helios::vec3& data ) const;
    
    //! Get global data (array of vec3's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of vec3's)
     */
    void getGlobalData( const char* label, std::vector<helios::vec3>& data ) const;
    
    //! Get global data value (scalar vec4)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data' Global data value (scalar vec4)
     */
    void getGlobalData( const char* label, helios::vec4& data ) const;
    
    //! Get global data (array of vec4's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of vec4's)
     */
    void getGlobalData( const char* label, std::vector<helios::vec4>& data ) const;
    
    //! Get global data value (scalar int2)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar int2)
     */
    void getGlobalData( const char* label, helios::int2& data ) const;
    
    //! Get global data (array of int2's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of int2's)
     */
    void getGlobalData( const char* label, std::vector<helios::int2>& data ) const;
    
    //! Get global data value (scalar int3)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar int3)
     */
    void getGlobalData( const char* label, helios::int3& data ) const;
    
    //! Get global data (array of int3's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of int3's)
     */
    void getGlobalData( const char* label, std::vector<helios::int3>& data ) const;
    
    //! Get global data value (scalar int4)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar int4)
     */
    void getGlobalData( const char* label, helios::int4& data ) const;
    
    //! Get global data (array of int4's)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of int4's)
     */
    void getGlobalData( const char* label, std::vector<helios::int4>& data ) const;
    
    //! Get global data value (scalar string)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Global data value (scalar string)
     */
    void getGlobalData( const char* label, std::string& data ) const;
    
    //! Get global data (array of strings)
    /**
     * \param[in] label Name/label associated with data
     * \param[out] data Pointer to global data (array of strings)
     */
    void getGlobalData( const char* label, std::vector<std::string>& data ) const;
    
    //! Get the Helios data type of global data
    /**
     * \param[in] label Name/label associated with data
     * \return Helios data type of global data
     */
    HeliosDataType getGlobalDataType( const char* label ) const;
    
    //! Get the size/length of global data
    /**
     * \param[in] label Name/label associated with data
     * \return Size/length of global data array
     */
    size_t getGlobalDataSize( const char* label ) const;
    
    //! Check if global data 'label' exists
    /**
     * \param[in] label Name/label associated with data
     * \return True/false
     */
    bool doesGlobalDataExist( const char* label ) const;

    //! Increase value of global data (int) by some value
    /**
     * \param[in] label Global data label string
     * \param[in] increment Value to increment global data by
     * \note If global data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementGlobalData( const char* label, int increment );

    //! Increase value of global data (uint) by some value
    /**
     * \param[in] label Global data label string
     * \param[in] increment Value to increment global data by
     * \note If global data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementGlobalData( const char* label, uint increment );

    //! Increase value of global data (float) by some value
    /**
     * \param[in] label Global data label string
     * \param[in] increment Value to increment global data by
     * \note If global data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementGlobalData( const char* label, float increment );

    //! Increase value of global data (double) by some value
    /**
     * \param[in] label Global data label string
     * \param[in] increment Value to increment global data by
     \note If global data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementGlobalData( const char* label, double increment );
    
    //--------- Compound Objects Methods -------------//
    
    //! Get a pointer to a Compound Object
    /**
     * \param[in] ObjID Identifier for Compound Object.
     */
    CompoundObject* getObjectPointer( uint ObjID ) const;
    
    //! Get the total number of objects that have been created in the Context
    /**
     * \return Total number of objects that have been created in the Context
     */
    uint getObjectCount() const;
    
    //! Check whether Compound Object exists in the Context
    /**
     * \param[in] ObjID Identifier for Compound Object.
     */
    bool doesObjectExist( uint ObjID ) const;
    
    //! Get the IDs for all Compound Objects in the Context
    /**
     * \return Vector of IDs for all objects.
     */
    std::vector<uint> getAllObjectIDs() const;
    
    //! Delete a single Compound Object from the context
    /**
     * \param[in] ObjID Identifier for Compound Object.
     */
    void deleteObject(uint ObjID );
    
    //! Delete a group of Compound Objects from the context
    /**
     * \param[in] ObjID Identifier for Compound Object.
     */
    void deleteObject(const std::vector<uint> &ObjIDs );
    
    //! Make a copy of a Compound Objects from the context
    /**
     * \param[in] ObjID Identifier for Compound Object.
     * \return ID for copied object.
     */
    uint copyObject(uint ObjID );
    
    //! Make a copy of a group of Compound Objects from the context
    /**
     * \param[in] ObjID Identifier for Compound Object.
     * \return ID for copied object.
     */
    std::vector<uint> copyObject(const std::vector<uint> &ObjIDs );

    //! copy all object data from one compound object to another
    /**
     * \param[in] objID uint Object identifier for compound object that is the source of data for copying
     * \param[in] currentObjID uint Object identifier for compound object that is the destination for data copying
     */
    void copyObjectData( uint objID, uint currentObjID);

    //! Duplicate/copy existing object data
    /**
     * \param[in] old_label Name/label associated with data
     * \param[in] new_label Name/label associated with data
     */
    void duplicateObjectData( uint objID, const char* old_label, const char* new_label );

    //! Rename existing object data
    /**
     * \param[in] old_label Name/label associated with data
     * \param[in] new_label Name/label associated with data
     */
    void renameObjectData( uint objID, const char* old_label, const char* new_label );
    
    //! Get a vector of object IDs that meet filtering criteria based on object data
    /**
     * \param[in] ObjIDs Vector of object IDs to filter
     * \param[in] object_data object data field to use when filtering
     * \param[in] threshold Value for filter threshold
     * \param[in] comparator Points will be filtered if "object_data (comparator) threshold", where (comparator) is one of ">", "<", or "="
     */
    std::vector<uint> filterObjectsByData( const std::vector<uint> &ObjIDs, const char* object_data, float threshold, const char* comparator) const;
    
    //! Translate a single compound object
    /**
     * \param[in] ObjID Object ID to translate
     * \param[in] shift Distance to translate in the (x,y,z) directions
     */
    void translateObject(uint ObjID, const vec3& shift );
    
    //! Translate multiple compound objects based on a vector of UUIDs
    /**
     * \param[in] ObjIDs Vector of object IDs to translate
     * \param[in] shift Distance to translate in the (x,y,z) directions
     */
    void translateObject(const std::vector<uint>& ObjIDs, const vec3& shift );
    
    //! Rotate a single compound object about the x, y, or z axis
    /**
     * \param[in] ObjID Object ID to rotate
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Axis about which to rotate (must be one of x, y, z)
     */
    void rotateObject(uint ObjID, float rot, const char* axis );
    
    //! Rotate multiple compound objects about the x, y, or z axis based on a vector of UUIDs
    /**
     * \param[in] ObjIDs Vector of object IDs to translate
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Axis about which to rotate (must be one of x, y, z)
     */
    void rotateObject(const std::vector<uint>& ObjIDs, float rot, const char* axis );
    
    //! Rotate a single compound object about an arbitrary axis passing through the origin
    /**
     * \param[in] ObjID Object ID to rotate
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Vector describing axis about which to rotate
     */
    void rotateObject(uint ObjID, float rot, const vec3& axis );
    
    //! Rotate multiple compound objects about an arbitrary axis passing through the origin based on a vector of UUIDs
    /**
     * \param[in] ObjIDs Vector of object IDs to translate
     * \param[in] rot Rotation angle in radians
     * \param[in] axis Vector describing axis about which to rotate
     */
    void rotateObject(const std::vector<uint>& ObjIDs, float rot, const vec3& axis );
    
    //! Rotate a single compound object about an arbitrary line (not necessarily passing through the origin)
    /**
     * \param[in] ObjID Object ID to rotate
     * \param[in] rot Rotation angle in radians
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis
     * \param[in] axis Vector describing axis about which to rotate
     */
    void rotateObject( uint ObjID, float rot, const vec3& origin, const vec3& axis );
    
    //! Rotate multiple compound objects about an arbitrary line (not necessarily passing through the origin) based on a vector of UUIDs
    /**
     * \param[in] ObjIDs Vector of object IDs to translate
     * \param[in] rot Rotation angle in radians
     * \param[in] origin Cartesian coordinate of the base/origin of rotation axis
     * \param[in] axis Vector describing axis about which to rotate
     */
    void rotateObject( const std::vector<uint>& ObjIDs, float rot, const vec3& origin, const vec3& axis );

    //! Method to scale a compound object in the x-, y- and z-directions
    /**
     * \param[in] ObjID Object ID to scale
     * \param[in] scalefact Scaling factor to apply in the x-, y- and z-directions
     */
    void scaleObject( uint ObjID, const helios::vec3 &scalefact );

    //! Method to scale a compound object in the x-, y- and z-directions
    /**
     * \param[in] ObjID Vector of object IDs to scale
     * \param[in] scalefact Scaling factor to apply in the x-, y- and z-directions
     */
    void scaleObject( const std::vector<uint>& ObjIDs, const helios::vec3 &scalefact );
    
    //! Get primitive UUIDs associated with compound object (single object ID input)
    /**
     * \param[in] ObjID object ID to retrieve primitive UUIDs for
     */
    std::vector<uint> getObjectPrimitiveUUIDs( uint ObjID ) const;

    //! Get primitive UUIDs associated with compound objects (1D vector of object IDs input)
    /**
     * \param[in] ObjIDs vector of object IDs to retrieve primitive UUIDs for
     */
    std::vector<uint> getObjectPrimitiveUUIDs( const std::vector<uint> &ObjIDs) const;

    //! Get primitive UUIDs associated with compound objects (2D vector of object IDs input)
    /**
     * \param[in] ObjIDs vector of object IDs to retrieve primitive UUIDs for
     */
    std::vector<uint> getObjectPrimitiveUUIDs( const std::vector<std::vector<uint> > &ObjIDs) const;
    
    //! Get an enumeration specifying the type of the object
    /**
     * \param[in] ObjID Object ID for which object type will be retrieved
     */
    helios::ObjectType getObjectType( uint ObjID ) const;
    
    //! Get a pointer to a Tile Compound Object
    /**
     * \param[in] ObjID Identifier for Tile Compound Object.
     */
    Tile* getTileObjectPointer(uint ObjID ) const;
    
    //! Get the area ratio of a tile object (total object area / sub-patch area)
    /**
     * \param[in] ObjID Identifier for Tile Compound Object.
     */
    float getTileObjectAreaRatio(const uint &ObjectID) const;
    
    //! Get the area ratio of a multiple tile objects (total object area / sub-patch area)
    /**
     * \param[in] ObjID Vector of identifiers for Tile Compound Object.
     */
    std::vector<float> getTileObjectAreaRatio(const std::vector<uint> &ObjectID) const;
    
    //! Change the subdivision count of a tile object
    /**
     * \param[in] ObjectIDs object IDs of the tile objects to change
     * \param[in] new_subdiv the new subdivisions desired
     */
    void setTileObjectSubdivisionCount(const std::vector<uint> &ObjectIDs, int2 new_subdiv);
    
    //! change the subdivisions of a tile object
    /**
     * \param[in] ObjectIDs object IDs of the tile objects to change
     * \param[in] area_ratio the approximate ratio between individual tile object area and individual subpatch area desired
     */
    void setTileObjectSubdivisionCount(const std::vector<uint> &ObjectIDs, float area_ratio);
    
    
    //! Get the Cartesian (x,y,z) center position of a tile object
    /**
     * \param[in] ObjectID object ID of the tile object
     * \return Center position of a Tile Object.
     * \note If the ObjID passed to this method does not correspond to a Tile Object, an error will be thrown.
     */
    helios::vec3 getTileObjectCenter(uint &ObjectID) const;
    
    
    //! get the size of a tile object from the context
    /**
     * \param[in] ObjectID object ID of the tile object
     */
    helios::vec2 getTileObjectSize(uint &ObjectID) const;
    
    //! get the subdivision count of a tile object from the context
    /**
     * \param[in] ObjectID object ID of the tile object
     */
    helios::int2 getTileObjectSubdivisionCount(uint &ObjectID) const;
    
    //! get the normal of a tile object from the context
    /**
     * \param[in] ObjectID object ID of the tile object
     */
    helios::vec3 getTileObjectNormal(uint &ObjectID) const;
    
    //! get the texture UV coordinates of a tile object from the context
    /**
     * \param[in] ObjectID object ID of the tile object
     */
    std::vector<helios::vec2> getTileObjectTextureUV(uint &ObjectID) const;
    
    //! get the vertices of a tile object from the context
    /**
     * \param[in] ObjectID object ID of the tile object
     */
    std::vector<helios::vec3> getTileObjectVertices(uint &ObjectID) const;
    
    //! Get a pointer to a Sphere Compound Object
    /**
     * \param[in] ObjID Identifier for Sphere Compound Object.
     */
    Sphere* getSphereObjectPointer(uint ObjID ) const;
    
    //! get the center of a Sphere object from the context
    /**
     * \param[in] ObjectID object ID of the Sphere object
     */
    helios::vec3 getSphereObjectCenter(uint &ObjectID) const;
    
    //! get the radius of a Sphere object from the context
    /**
     * \param[in] ObjectID object ID of the Sphere object
     */
    helios::vec3 getSphereObjectRadius(uint &ObjID) const;
    
    //! get the subdivision count of a Sphere object from the context
    /**
     * \param[in] ObjectID object ID of the Sphere object
     */
    uint getSphereObjectSubdivisionCount(uint &ObjectID) const;
    
    //! Get a pointer to a Tube Compound Object
    /** \param[in] ObjID Identifier for Tube Compound Object.
     */
    Tube* getTubeObjectPointer(uint ObjID ) const;
    
    //! get the subdivision count of a Tube object from the context
    /**
     * \param[in] ObjectID object ID of the Tube object
     */
    uint getTubeObjectSubdivisionCount(uint &ObjectID) const;
    
    //! get the nodes of a Tube object from the context
    /**
     * \param[in] ObjectID object ID of the Tube object
     */
    std::vector<helios::vec3> getTubeObjectNodes(uint &ObjectID) const;
    
    //! get the node radii of a Tube object from the context
    /**
     * \param[in] ObjectID object ID of the Tube object
     */
    std::vector<float> getTubeObjectNodeRadii(uint &ObjectID) const;
    
    //! get the node colors of a Tube object from the context
    /**
     * \param[in] ObjectID object ID of the Tube object
     */
    std::vector<RGBcolor> getTubeObjectNodeColors(uint &ObjectID) const;
    
    
    //! Get a pointer to a Box Compound Object
    /**
     * \param[in] ObjID Identifier for Box Compound Object.
     */
    Box* getBoxObjectPointer(uint ObjID ) const;
    
    //! get the center of a Box object from the context
    /**
     * \param[in] ObjectID object ID of the Box object
     */
    helios::vec3 getBoxObjectCenter(uint &ObjectID) const;
    
    //! get the size of a Box object from the context
    /**
     * \param[in] ObjectID object ID of the Box object
     */
    helios::vec3 getBoxObjectSize(uint &ObjectID) const;
    
    //! get the subdivision count of a Box object from the context
    /**
     * \param[in] ObjectID object ID of the Box object
     */
    helios::int3 getBoxObjectSubdivisionCount(uint &ObjectID) const;
    
    //! Get a pointer to a Disk Compound Object
    /**
     * \param[in] ObjID Identifier for Disk Compound Object.
     */
    Disk* getDiskObjectPointer(uint ObjID ) const;
    
    //! get the center of a Disk object from the context
    /**
     * \param[in] ObjectID object ID of the Disk object
     */
    helios::vec3 getDiskObjectCenter(uint &ObjectID) const;
    
    //! get the size of a Disk object from the context
    /**
     * \param[in] ObjectID object ID of the Disk object
     */
    helios::vec2 getDiskObjectSize(uint &ObjectID) const;
    
    //! get the subdivision count of a Disk object from the context
    /**
     * \param[in] ObjectID object ID of the Disk object
     */
    uint getDiskObjectSubdivisionCount(uint &ObjectID) const;
    
    //! Get a pointer to a Polygon Mesh Compound Object
    /** \param[in] ObjID Identifier for Polygon Mesh Compound Object.
     */
    Polymesh* getPolymeshObjectPointer(uint ObjID ) const;
    
    //! Get a pointer to a Cone Compound Object
    /**
     * \param[in] ObjID Identifier for Cone Compound Object.
     */
    Cone* getConeObjectPointer( uint ObjID ) const;
    
    //! get the subdivision count of a Cone object from the context
    /**
     * \param[in] ObjectID object ID of the Cone object
     */
    uint getConeObjectSubdivisionCount(uint &ObjectID) const;
    
    //! get the nodes of a Cone object from the context
    /**
     * \param[in] ObjectID object ID of the Cone object
     */
    std::vector<helios::vec3> getConeObjectNodes(uint &ObjectID) const;
    
    //! get the node radii of a Cone object from the context
    /**
     * \param[in] ObjectID object ID of the Cone object
     */
    std::vector<float> getConeObjectNodeRadii(uint &ObjectID) const;
    
    //! get a node of a Cone object from the context
    /**
     * \param[in] ObjectID object ID of the Cone object
     */
    helios::vec3 getConeObjectNode(uint &ObjectID, int number) const;
    
    //! get a node radius of a Cone object from the context
    /**
     * \param[in] ObjectID object ID of the Cone object
     */
    float getConeObjectNodeRadius(uint &ObjectID, int number) const;
    
    //! get the axis unit vector of a Cone object from the context
    /**
     * \param[in] ObjectID object ID of the Cone object
     */
    helios::vec3 getConeObjectAxisUnitVector(uint &ObjectID) const;
    
    //! get the length of a Cone object from the context
    /**
     * \param[in] ObjectID object ID of the Cone object
     * \return Dimension of cone object in the axial direction
     */
    float getConeObjectLength(uint &ObjectID) const;
    
    //! Add a patch that is subdivided into a regular grid of sub-patches (tiled)
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x- and y-directions
     * \param[in] rotation Spherical rotation of tiled surface
     * \param[in] subdiv Number of subdivisions in x- and y-directions
     * \return Object ID of new tile object.
     * \note Assumes default color of green
     * \ingroup compoundobjects
     */
    uint addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv);
    
    //! Add a patch that is subdivided into a regular grid of sub-patches (tiled)
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x- and y-directions
     * \param[in] rotation Spherical rotation of tiled surface
     * \param[in] subdiv Number of subdivisions in x- and y-directions
     * \param[in] color r-g-b color of tiled surface
     * \return Object ID of new tile object.
     * \ingroup compoundobjects
     */
    uint addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const RGBcolor &color );
    
    //! Add a patch that is subdivided into a regular grid of sub-patches (tiled)
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x- and y-directions
     * \param[in] rotation Spherical rotation of tiled surface
     * \param[in] subdiv Number of subdivisions in x- and y-directions
     * \param[in] texturefile Name of image file for texture map
     * \return Object ID of new tile object.
     * \ingroup compoundobjects
     */
    uint addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char* texturefile );

    //! Add a spherical compound object to the Context
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere
     * \return Object ID of new sphere object.
     * \note Assumes a default color of green
     * \ingroup compoundobjects
     */
    uint addSphereObject(uint Ndivs, const vec3 &center, float radius );

    //! Add a spherical compound object to the Context
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere
     * \param[in] color r-g-b color of sphere
     * \return Object ID of new sphere object.
     * \ingroup compoundobjects
     */
    uint addSphereObject(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color );

    //! Add a spherical compound object to the Context colored by texture map
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere
     * \param[in] texturefile Name of image file for texture map
     * \return Object ID of new sphere object.
     * \ingroup compoundobjects
     */
    uint addSphereObject(uint Ndivs, const vec3 &center, float radius, const char* texturefile );

    //! Add a spherical/ellipsoidal compound object to the Context
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere in x-, y-, and z-directions
     * \return Object ID of new sphere object.
     * \note Assumes a default color of green
     * \ingroup compoundobjects
     */
    uint addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius );

    //! Add a spherical/ellipsoidal compound object to the Context
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere in x-, y-, and z-directions
     * \param[in] color r-g-b color of sphere
     * \return Object ID of new sphere object.
     * \ingroup compoundobjects
     */
    uint addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius, const RGBcolor &color );

    //! Add a spherical/ellipsoidal compound object to the Context colored by texture map
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere in x-, y-, and z-directions
     * \param[in] texturefile Name of image file for texture map
     * \return Object ID of new sphere object.
     * \ingroup compoundobjects
     */
    uint addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius, const char* texturefile );
    
    //! Add a 3D tube compound object to the Context
    /** A `tube' or `snake' compound object comprised of Triangle primitives
     * \image html doc/images/Tube.png "Sample image of a Tube compound object." width=0.1cm
     * \param[in] Ndivs Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] nodes Vector of (x,y,z) positions defining Tube segments.
     * \param[in] radius Radius of the tube at each node position.
     * \return Object ID of new tube object.
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    uint addTubeObject(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius );
    
    //! Add a 3D tube compound object to the Context and specify its diffuse color
    /** A `tube' or `snake' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] nodes Vector of (x,y,z) positions defining Tube segments.
     * \param[in] radius Radius of the tube at each node position.
     * \param[in] color Diffuse color of each tube segment.
     * \return Object ID of new tube object.
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    uint addTubeObject( uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color );
    
    //! Add a 3D tube compound object to the Context that is texture-mapped
    /** A `tube' or `snake' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] nodes Vector of (x,y,z) positions defining Tube segments.
     * \param[in] radius Radius of the tube at each node position.
     * \param[in] texturefile Name of image file for texture map
     * \return Object ID of new tube object.
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    uint addTubeObject( uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char* texturefile );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \return Object ID of new box object
     * \note Assumes default color of green
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    uint addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] color r-g-b color of box
     * \return Object ID of new box object
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    uint addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] texturefile Name of image file for texture map
     * \return Object ID of new box object
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    uint addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] color r-g-b color of box
     * \param[in] reverse_normals Flip all surface normals so that patch normals point inside the box
     * \return Object ID of new box object
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    uint addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] texturefile Name of image file for texture map
     * \return Object ID of new box object
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    uint addBoxObject(vec3 center, const vec3 &size, const int3 &subdiv, const char* texturefile, bool reverse_normals );
    
    //! Add new Disk geometric primitive to the Context given its center, and size.
    /**
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \return Object ID of new disk object
     * \note Assumes that disk is horizontal.
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    uint addDiskObject(uint Ndivs, const helios::vec3& center, const helios::vec2& size );
    
    //! Add new Disk Compound Object
    /** Method to add a new Disk to the Context given its center, size, and spherical rotation.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \return Object ID of new disk object
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    uint addDiskObject(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation );
    
    //! Add new Disk Compound Object
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B color of Disk
     * \return Object ID of new disk object
     * \ingroup compoundobjects
     */
    uint addDiskObject(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBcolor& color );
    
    //! Add new Disk Compound Object
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBAcolor.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B-A color of Disk
     * \return Object ID of new disk object
     * \ingroup compoundobjects
     */
    uint addDiskObject(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBAcolor& color );
    
    //! Add new Disk Compound Object
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and a texture map handle.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] texture_file path to JPEG file to be used as texture
     * \return Object ID of new disk object
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    uint addDiskObject(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file );

    //! Add new Disk Compound Object
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
     * \param[in] Ndiv Number of subdivisions for triangulation in the polar (.x) and radial (.y) directions
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B color of Disk
     * \return Object ID of new disk object
     * \ingroup compoundobjects
     */
    uint addDiskObject(const int2 &Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBcolor& color );

    //! Add new Disk Compound Object
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBAcolor.
     * \param[in] Ndiv Number of subdivisions for triangulation in the polar (.x) and radial (.y) directions
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B-A color of Disk
     * \return Object ID of new disk object
     * \ingroup compoundobjects
     */
    uint addDiskObject(const int2 &Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBAcolor& color );

    //! Add new Disk Compound Object
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and a texture map handle.
     * \param[in] Ndiv Number of subdivisions for triangulation in the polar (.x) and radial (.y) directions
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] texture_file path to JPEG file to be used as texture
     * \return Object ID of new disk object
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    uint addDiskObject(const int2 &Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file );
    
    //! Add new Polymesh Compound Object
    /** Method to add a new Polymesh to the Context given a vector of UUIDs
     * \param[in] UUIDs Unique universal identifiers of primitives to be added to polymesh object
     * \return Object ID of new Polymesh
     * \ingroup compoundobjects
     */
    uint addPolymeshObject(const std::vector<uint> &UUIDs );
    
    //! Add a 3D cone compound object to the Context
    /** A `cone' or `cone frustum' or 'cylinder' compound object comprised of Triangle primitives
     * \image html doc/images/Tube.png "Sample image of a Tube compound object." width=0.1cm
     * \param[in] Ndivs Number of radial divisions of the Cone. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] node0 (x,y,z) position defining the base of the cone
     * \param[in] node1 (x,y,z) position defining the end of the cone
     * \param[in] radius0 Radius of the cone at the base node.
     * \param[in] radius1 Radius of the cone at the base node.
     * \return Vector of UUIDs for each sub-triangle
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    uint addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1 );
    
    //! Add a 3D cone compound object to the Context and specify its diffuse color
    /** A `cone' or `cone frustum' or 'cylinder' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Cone. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] node0 (x,y,z) position defining the base of the cone
     * \param[in] node1 (x,y,z) position defining the end of the cone
     * \param[in] radius0 Radius of the cone at the base node.
     * \param[in] radius1 Radius of the cone at the base node.
     * \param[in] color Diffuse color of each tube segment.
     * \return Vector of UUIDs for each sub-triangle
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    uint addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const RGBcolor &color );
    
    //! Add a 3D cone compound object to the Context that is texture-mapped
    /** A `cone' or `cone frustum' or 'cylinder' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Cone. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] node0 (x,y,z) position defining the base of the cone
     * \param[in] node1 (x,y,z) position defining the end of the cone
     * \param[in] radius0 Radius of the cone at the base node.
     * \param[in] radius1 Radius of the cone at the base node.
     * \param[in] texturefile Name of image file for texture map
     * \return Vector of UUIDs for each sub-triangle
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    uint addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const char* texturefile );
    
    //! Add a spherical compound object to the Context
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere
     * \return Vector of UUIDs for each sub-triangle
     * \note Assumes a default color of green
     * \ingroup compoundobjects
     */
    std::vector<uint> addSphere(uint Ndivs, const vec3 &center, float radius );
    
    //! Add a spherical compound object to the Context
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere
     * \param[in] color r-g-b color of sphere
     * \return Vector of UUIDs for each sub-triangle
     * \ingroup compoundobjects
     */
    std::vector<uint> addSphere(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color );
    
    //! Add a spherical compound object to the Context colored by texture map
    /**
     * \param[in] Ndivs Number of tessellations in zenithal and azimuthal directions
     * \param[in] center (x,y,z) coordinate of sphere center
     * \param[in] radius Radius of sphere
     * \param[in] texturefile Name of image file for texture map
     * \return Vector of UUIDs for each sub-triangle
     * \ingroup compoundobjects
     */
    std::vector<uint> addSphere(uint Ndivs, const vec3 &center, float radius, const char* texturefile );
    
    //! Add a patch that is subdivided into a regular grid of sub-patches (tiled)
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x- and y-directions
     * \param[in] rotation Spherical rotation of tiled surface
     * \param[in] subdiv Number of subdivisions in x- and y-directions
     * \return Vector of UUIDs for each sub-patch
     * \note Assumes default color of green
     * \ingroup compoundobjects
     */
    std::vector<uint> addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv );
    
    //! Add a patch that is subdivided into a regular grid of sub-patches (tiled)
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x- and y-directions
     * \param[in] rotation Spherical rotation of tiled surface
     * \param[in] subdiv Number of subdivisions in x- and y-directions
     * \param[in] color r-g-b color of tiled surface
     * \return Vector of UUIDs for each sub-patch
     * \ingroup compoundobjects
     */
    std::vector<uint> addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const RGBcolor &color );
    
    //! Add a patch that is subdivided into a regular grid of sub-patches (tiled)
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x- and y-directions
     * \param[in] rotation Spherical rotation of tiled surface
     * \param[in] subdiv Number of subdivisions in x- and y-directions
     * \param[in] texturefile Name of image file for texture map
     * \return Vector of UUIDs for each sub-patch
     * \ingroup compoundobjects
     */
    std::vector<uint> addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char* texturefile );
    
    //! Add a 3D tube compound object to the Context
    /** A `tube' or `snake' compound object comprised of Triangle primitives
     * \image html doc/images/Tube.png "Sample image of a Tube compound object." width=0.1cm
     * \param[in] Ndivs Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] nodes Vector of (x,y,z) positions defining Tube segments.
     * \param[in] radius Radius of the tube at each node position.
     * \return Vector of UUIDs for each sub-triangle
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    std::vector<uint> addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius );
    
    //! Add a 3D tube compound object to the Context and specify its diffuse color
    /** A `tube' or `snake' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] nodes Vector of (x,y,z) positions defining Tube segments.
     * \param[in] radius Radius of the tube at each node position.
     * \param[in] color Diffuse color of each tube segment.
     * \return Vector of UUIDs for each sub-triangle
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    std::vector<uint> addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color );
    
    //! Add a 3D tube compound object to the Context that is texture-mapped
    /** A `tube' or `snake' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Tube. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] nodes Vector of (x,y,z) positions defining Tube segments.
     * \param[in] radius Radius of the tube at each node position.
     * \param[in] texturefile Name of image file for texture map
     * \return Vector of UUIDs for each sub-triangle
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    std::vector<uint> addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char* texturefile );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \return Vector of UUIDs for each sub-patch
     * \note Assumes default color of green
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    std::vector<uint> addBox(const vec3 &center, const vec3 &size, const int3 &subdiv );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] color r-g-b color of box
     * \return Vector of UUIDs for each sub-patch
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    std::vector<uint> addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] texturefile Name of image file for texture map
     * \return Vector of UUIDs for each sub-patch
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    std::vector<uint> addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] color r-g-b color of box
     * \param[in] reverse_normals Flip all surface normals so that patch normals point inside the box
     * \return Vector of UUIDs for each sub-patch
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    std::vector<uint> addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals );
    
    //! Add a rectangular prism tessellated with Patch primitives
    /**
     * \param[in] center 3D coordinates of box center
     * \param[in] size Size of the box in the x-, y-, and z-directions
     * \param[in] subdiv Number of subdivisions in x-, y-, and z-directions
     * \param[in] texturefile Name of image file for texture map
     * \return Vector of UUIDs for each sub-patch
     * \note This version of addBox assumes that all surface normal vectors point away from the box
     * \ingroup compoundobjects
     */
    std::vector<uint> addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile, bool reverse_normals );
    
    //! Add new Disk geometric primitive to the Context given its center, and size.
    /**
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \return Vector of UUIDs for each sub-triangle
     * \note Assumes that disk is horizontal.
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(uint Ndivs, const helios::vec3& center, const helios::vec2& size );
    
    //! Add new Disk geometric primitive
    /** Method to add a new Disk to the Context given its center, size, and spherical rotation.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \return Vector of UUIDs for each sub-triangle
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation );
    
    //! Add new Disk geometric primitive
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B color of Disk
     * \return Vector of UUIDs for each sub-triangle
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBcolor& color );
    
    //! Add new Disk geometric primitive
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBAcolor.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B-A color of Disk
     * \return Vector of UUIDs for each sub-triangle
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBAcolor& color );
    
    //! Add new Disk geometric primitive
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and a texture map handle.
     * \param[in] Ndiv Number to triangles used to form disk
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] texture_file path to JPEG file to be used as texture
     * \return Vector of UUIDs for each sub-triangle
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(uint Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file );

    //! Add new Disk geometric primitive
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBcolor.
     * \param[in] Ndiv Number of subdivisions for triangulation in the polar (.x) and radial (.y) directions
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B color of Disk
     * \return Vector of UUIDs for each sub-triangle
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(const int2 &Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBcolor& color );

    //! Add new Disk geometric primitive
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and diffuse RGBAcolor.
     * \param[in] Ndiv Number of subdivisions for triangulation in the polar (.x) and radial (.y) directions
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] color diffuse R-G-B-A color of Disk
     * \return Vector of UUIDs for each sub-triangle
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(const int2 &Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const helios::RGBAcolor& color );

    //! Add new Disk geometric primitive
    /** Method to add a new Disk to the Context given its center, size, spherical rotation, and a texture map handle.
     * \param[in] Ndiv Number of subdivisions for triangulation in the polar (.x) and radial (.y) directions
     * \param[in] center 3D coordinates of Disk center
     * \param[in] size length of Disk semi-major and semi-minor radii
     * \param[in] rotation spherical rotation angle (elevation,azimuth) in radians of Disk
     * \param[in] texture_file path to JPEG file to be used as texture
     * \return Vector of UUIDs for each sub-triangle
     * \note Assumes a default color of black.
     * \ingroup compoundobjects
     */
    std::vector<uint> addDisk(const int2 &Ndivs, const helios::vec3& center, const helios::vec2& size, const helios::SphericalCoord& rotation, const char* texture_file );
    
    //! Add a 3D cone to the Context
    /** A `cone' or `cone frustum' or 'cylinder' compound object comprised of Triangle primitives
     * \image html doc/images/Tube.png "Sample image of a Tube compound object." width=0.1cm
     * \param[in] Ndivs Number of radial divisions of the Cone. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] node0 (x,y,z) position defining the base of the cone
     * \param[in] node1 (x,y,z) position defining the end of the cone
     * \param[in] radius0 Radius of the cone at the base node.
     * \param[in] radius1 Radius of the cone at the base node.
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    std::vector<uint> addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1 );
    
    //! Add a 3D cone to the Context and specify its diffuse color
    /** A `cone' or `cone frustum' or 'cylinder' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Cone. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] node0 (x,y,z) position defining the base of the cone
     * \param[in] node1 (x,y,z) position defining the end of the cone
     * \param[in] radius0 Radius of the cone at the base node.
     * \param[in] radius1 Radius of the cone at the base node.
     * \param[in] color Diffuse color of the cone.
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    std::vector<uint> addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, RGBcolor &color );
    
    //! Add a 3D cone to the Context that is texture-mapped
    /** A `cone' or `cone frustum' or 'cylinder' compound object comprised of Triangle primitives
     * \param[in] Ndivs Number of radial divisions of the Cone. E.g., Ndivs = 3 would be a triangular prism, Ndivs = 4 would be a rectangular prism, etc.
     * \param[in] node0 (x,y,z) position defining the base of the cone
     * \param[in] node1 (x,y,z) position defining the end of the cone
     * \param[in] radius0 Radius of the cone at the base node.
     * \param[in] radius1 Radius of the cone at the base node.
     * \param[in] texturefile Name of image file for texture map
     * \note Ndivs must be greater than 2.
     * \ingroup compoundobjects
     */
    std::vector<uint> addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const char* texturefile );
    
    //! Add a data point to timeseries of data
    /**
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \param[in] value Value of timeseries data point
     * \param[in] date Date vector corresponding to the time of `value' (see \ref helios::Date "Date")
     * \param[in] time Time vector corresponding to the time of `value' (see \ref helios::time "Time")
     * \ingroup timeseries
     */
    void addTimeseriesData(const char* label, float value, const Date &date, const Time &time );
    
    //! Set the Context date and time by providing the index of a timeseries data point
    /**
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \param[in] index Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     * \ingroup timeseries
     */
    void setCurrentTimeseriesPoint(const char* label, uint index );
    
    //! Get a timeseries data point by specifying a date and time vector.
    /**This method interpolates the timeseries data to provide a value at exactly `date' and `time'.  Thus, `date' and `time' must be between the first and last timeseries values.
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \param[in] date Date vector corresponding to the time of `value' (see \ref helios::Date "Date")
     * \param[in] time Time vector corresponding to the time of `value' (see \ref helios::time "Time")
     * \return Value of timeseries data point
     * \ingroup timeseries
     */
    float queryTimeseriesData(const char* label, const Date &date, const Time &time ) const;

    //! Get a timeseries data point at the time currently set in the Context
    /**
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \return Value of timeseries data point
     * \ingroup timeseries
     */
    float queryTimeseriesData( const char* label ) const;
    
    //! Get a timeseries data point by index in the timeseries
    /**This method returns timeseries data by index, and is typically used when looping over all data in the timeseries.  See \ref getTimeseriesLength() to get the total length of the timeseries data.
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \param[in] index Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     * \return Value of timeseries data point
     * \ingroup timeseries
     */
    float queryTimeseriesData( const char* label, uint index ) const;
    
    //! Get the time associated with a timeseries data point
    /**
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \param[in] index Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     * \return Time of timeseries data point
     * \ingroup timeseries
     */
    Time queryTimeseriesTime( const char* label, uint index ) const;
    
    //! Get the date associated with a timeseries data point
    /**
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \param[in] index Index of timeseries data point. The index starts at 0 for the earliest data point, and moves chronologically in time.
     * \return Date of timeseries data point
     * \ingroup timeseries
     */
    Date queryTimeseriesDate( const char* label, uint index ) const;
    
    //! Get the length of timeseries data
    /**
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \ingroup timeseries
     */
    uint getTimeseriesLength( const char* label ) const;

    //! Query whether a timeseries variable exists
    /**
     * \param[in] label Name of timeseries variable (e.g., temperature)
     * \ingroup timeseries
     */
    bool doesTimeseriesVariableExist( const char* label ) const;

    //! Load tabular weather data from text file into timeseries
    void loadTabularTimeseriesData( const std::string &data_file, const std::vector<std::string> &column_labels, const std::string &delimiter, const std::string &date_string_format="YYYYMMDD", uint headerlines=0 );
    
    //! Get a box that bounds all primitives in the domain
    /**
     * \param[out] xbounds Domain bounds in x-direction (xbounds.x=min bound, xbounds.y=max bound)
     * \param[out] ybounds Domain bounds in x-direction (ybounds.x=min bound, ybounds.y=max bound)
     * \param[out] zbounds Domain bounds in x-direction (zbounds.x=min bound, zbounds.y=max bound)
     */
    void getDomainBoundingBox( helios::vec2& xbounds, helios::vec2& ybounds, helios::vec2& zbounds ) const;
    
    //! Get a box that bounds a subset of primitives
    /** *
     * \param[in] UUIDs Subset of primitive UUIDs for bounding box calculation.
     * \param[out] xbounds Domain bounds in x-direction (xbounds.x=min bound, xbounds.y=max bound)
     * \param[out] ybounds Domain bounds in x-direction (ybounds.x=min bound, ybounds.y=max bound)
     * \param[out] zbounds Domain bounds in x-direction (zbounds.x=min bound, zbounds.y=max bound)
     */
    void getDomainBoundingBox( const std::vector<uint>& UUIDs, helios::vec2& xbounds, helios::vec2& ybounds, helios::vec2& zbounds ) const;
    
    //! Get the center and radius of a sphere that bounds all primitives in the domain
    /**
     * \param[out] center Center of domain bounding sphere.
     * \param[out] radius Radius of domain bounding sphere.
     */
    void getDomainBoundingSphere( helios::vec3& center, float& radius ) const;
    
    //! Get the center and radius of a sphere that bounds a subset of primitives
    /**
     * \param[in] UUIDs Subset of primitive UUIDs for bounding sphere calculation.
     * \param[out] center Center of primitive bounding sphere.
     * \param[out] radius Radius of primitive bounding sphere.
     */
    void getDomainBoundingSphere( const std::vector<uint>& UUIDs, helios::vec3& center, float& radius ) const;
    
    //! Crop the domain in the x-direction such that all primitives lie within some specified x interval.
    /**
     * \param[in] xbounds Minimum (xbounds.x) and maximum (xbounds.y) extent of cropped domain in x-direction.
     */
    void cropDomainX(const vec2 &xbounds );
    
    //! Crop the domain in the y-direction such that all primitives lie within some specified y interval.
    /**
     * \param[in] ybounds Minimum (ybounds.x) and maximum (ybounds.y) extent of cropped domain in y-direction.
     */
    void cropDomainY(const vec2 &ybounds );
    
    //! Crop the domain in the z-direction such that all primitives lie within some specified z interval.
    /**
     * \param[in] zbounds Minimum (zbounds.x) and maximum (zbounds.y) extent of cropped domain in z-direction.
     */
    void cropDomainZ(const vec2 &zbounds );
    
    //! Crop specified UUIDs such that they lie within some specified axis-aligned box
    /**
     * \param[inout] UUIDs vector of UUIDs to crop
     * \param[in] xbounds Minimum (xbounds.x) and maximum (xbounds.y) extent of cropped domain in x-direction.
     * \param[in] ybounds Minimum (ybounds.x) and maximum (ybounds.y) extent of cropped domain in y-direction.
     * \param[in] zbounds Minimum (zbounds.x) and maximum (zbounds.y) extent of cropped domain in z-direction.
     */
    void cropDomain( std::vector<uint> &UUIDs, const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds );
    
    //! Crop the domain such that all primitives lie within some specified axis-aligned box
    /**
     * \param[in] xbounds Minimum (xbounds.x) and maximum (xbounds.y) extent of cropped domain in x-direction.
     * \param[in] ybounds Minimum (ybounds.x) and maximum (ybounds.y) extent of cropped domain in y-direction.
     * \param[in] zbounds Minimum (zbounds.x) and maximum (zbounds.y) extent of cropped domain in z-direction.
     */
    void cropDomain(const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds );

    //! Overwrite primitive color based on a pseudocolor mapping of primitive data values
    /**
     * \param[in] UUIDs Primitives to apply psudocolor mapping
     * \param[in] primitive_data Label of primitive data for mapping
     * \param[in] colormap Label of a Helios colormap (e.g., "hot", "cool", "rainbow", "lava")
     * \param[in] Ncolors Number of discrete colors in color mapping
     */
    void colorPrimitiveByDataPseudocolor( const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors );

    //! Overwrite primitive color based on a pseudocolor mapping of primitive data values. Clamp to specified data range.
    /**
     * \param[in] UUIDs Primitives to apply psudocolor mapping
     * \param[in] primitive_data Label of primitive data for mapping
     * \param[in] colormap Label of a Helios colormap (e.g., "hot", "cool", "rainbow", "lava")
     * \param[in] Ncolors Number of discrete colors in color mapping
     * \param[in] data_min Minimum data value to clip colormap
     * \param[in] data_max Maximum data value to clip colormap
     */
    void colorPrimitiveByDataPseudocolor( const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors, float data_min, float data_max );

    //! Load inputs specified in an XML file.
    /**
     * \param[in] filename name of XML file.
     * \param[in] quiet If set to true, command line output will be disabled. Optional argument - default value is false.
     * \return Vector of UUIDs for each primitive loaded from the XML file
     * \note This method is based on the pugi xml parser.  See <a href="www.pugixml.org">pugixml.org</a>
     */
    std::vector<uint> loadXML( const char* filename, bool quiet = false );
    
    //! Get names of XML files that are currently loaded
    /**
     * \return Vector of XML files.
     */
    std::vector<std::string> getLoadedXMLFiles();
    
    //! Write Context geometry and data to XML file for all UUIDs in the context
    /**
     * \param[in] filename name of XML file.
     * \param[in] quiet output messages are disabled if quiet is set to 'true' (default is quiet='false')
     */
    void writeXML( const char* filename, bool quiet = false ) const;

    //! Write Context geometry and data to XML file for a subset of UUIDs in the context
    /**
     * \param[in] filename name of XML file.
     * \param[in] UUIDs Universal unique identifiers of primitives that should be written to XML file.
     * \param[in] quiet output messages are disabled if quiet is set to 'true' (default is quiet='false').
     */
    void writeXML( const char* filename, const std::vector<uint> &UUIDs, bool quiet = false ) const;

    //! Write Context geometry and data to XML file for a subset of compound object IDs in the context
    /**
     * \param[in] filename name of XML file.
     * \param[in] objIDs Identifiers for compound objects that should be written to XML file.
     * \param[in] quiet output messages are disabled if quiet is set to 'true' (default is quiet='false').
     */
    void writeXML_byobject( const char* filename, const std::vector<uint> &UUIDs, bool quiet = false ) const;
    
    //! Write primitive data to an ASCII text file for all primitives in the Context
    /**
     * \param[in] filename Path to file that will be written.
     * \param[in] column_format Vector of strings with primitive data labels - the order of the text file columns will be determined by the order of the labels in the vector. If primitive data does not exist, an error will be thrown.
     * \param[in] print_header Flag specifying whether to print the name of the primitive data in the column header.
     */
    void writePrimitiveData( std::string filename, const std::vector<std::string> &column_format, bool print_header = false ) const;
    
    //! Write primitive data to an ASCII text file for selected primitives in the Context
    /**
     * \param[in] filename Path to file that will be written.
     * \param[in] column_format Vector of strings with primitive data labels - the order of the text file columns will be determined by the order of the labels in the vector. If primitive data does not exist, an error will be thrown.
     * \param[in] UUIDs Unique universal identifiers for primitives to include when writing data to file.
     * \param[in] print_header Flag specifying whether to print the name of the primitive data in the column header.
     */
    void writePrimitiveData( std::string filename, const std::vector<std::string> &column_format, const std::vector<uint> &UUIDs, bool print_header = false ) const;

    //! Load geometry contained in a Stanford polygon file (.ply). Model will be placed at the origin with no scaling or rotation applied.
    /**
     * \param[in] filename name of ply file.
     * \param[in] silent (optional) If set to true, output messaged will be disabled.
     * \return Vector of UUIDs for each primitive generated from the PLY model
     * \note Assumes default color of blue if no colors are specified in the .ply file
     */
    std::vector<uint> loadPLY(const char* filename, bool silent=false );

    //! Load geometry contained in a Stanford polygon file (.ply)
    /**
     * \param[in] filename name of ply file.
     * \param[in] origin (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
     * \param[in] height Scaling factor to be applied to give model an overall height of "height" (setting height=0 applies no scaling)
     * \param[in] upaxis (optional) Axis defining upward direction used in the PLY file ("XUP", "YUP", or "ZUP"). Default is "YUP".
     * \param[in] silent (optional) If set to true, output messaged will be disabled.
     * \return Vector of UUIDs for each primitive generated from the PLY model
     * \note Assumes default color of blue if no colors are specified in the .ply file
     */
    std::vector<uint> loadPLY(const char* filename, const vec3 &origin, float height, const std::string &upaxis="YUP", bool silent=false );
    
    //! Load geometry contained in a Stanford polygon file (.ply)
    /**
     * \param[in] filename name of ply file.
     * \param[in] origin (x,y,z) coordinate of PLY object origin (i.e., coordinate shift).
     * \param[in] height Scaling factor to be applied to give model an overall height of "height" (setting height=0 applies no scaling)
     * \param[in] rotation Spherical rotation of PLY object about origin
     * \param[in] upaxis (optional) Axis defining upward direction used in the PLY file ("XUP", "YUP", or "ZUP"). Default is "YUP".
     * \param[in] silent (optional) If set to true, output messaged will be disabled
     * \return Vector of UUIDs for each primitive generated from the PLY model
     * \note Assumes default color of blue if no colors are specified in the .ply file
     */
    std::vector<uint> loadPLY(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation, const std::string &upaxis="YUP", bool silent=false );
    
    //! Load geometry contained in a Stanford polygon file (.ply)
    /**
     * \param[in] filename name of ply file.
     * \param[in] origin (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
     * \param[in] height Scaling factor to be applied to give model an overall height of "height" (setting height=0 applies no scaling)
     * \param[in] default_color Color to be used if no r-g-b color values are given for PLY nodes
     * \param[in] upaxis (optional) Axis defining upward direction used in the PLY file ("XUP", "YUP", or "ZUP"). Default is "YUP".
     * \param[in] silent (optional) If set to true, output messaged will be disabled
     * \return Vector of UUIDs for each primitive generated from the PLY model
     */
    std::vector<uint> loadPLY(const char* filename, const vec3 &origin, float height, const RGBcolor &default_color, const std::string &upaxis="YUP", bool silent=false );
    
    //! Load geometry contained in a Stanford polygon file (.ply)
    /**
     * \param[in] filename name of ply file.
     * \param[in] origin (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
     * \param[in] height Scaling factor to be applied to give model an overall height of "height" (setting height=0 applies no scaling)
     * \param[in] rotation Spherical rotation of PLY object about origin
     * \param[in] default_color Color to be used if no r-g-b color values are given for PLY nodes
     * \param[in] upaxis (optional) Axis defining upward direction used in the PLY file ("XUP", "YUP", or "ZUP"). Default is "YUP".
     * \param[in] silent (optional) If set to true, output messaged will be disabled
     * \return Vector of UUIDs for each primitive generated from the PLY model
     */
    std::vector<uint> loadPLY(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color, const std::string &upaxis="YUP", bool silent=false );
    
    //! Write geometry in the Context to a Stanford polygon file (.ply)
    /**
     * \param[in] filename name of ply file
    */
    void writePLY( const char* filename ) const;

    //! Load geometry contained in a Wavefront OBJ file (.obj). Model will be placed at the origin without any scaling or rotation applied.
    /**
     * \param[in] filename name of OBJ file
     * \param[in] silent (optional) If set to true, output messaged will be disabled
     */
    std::vector<uint> loadOBJ(const char* filename, bool silent=false );
    
    //! Load geometry contained in a Wavefront OBJ file (.obj)
    /**
     * \param[in] filename name of OBJ file
     * \param[in] origin (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
     * \param[in] height A z-scaling factor is applied to make the model 'height' tall. If height=0, no scaling is applied
     * \param[in] rotation Spherical rotation of PLY object about origin
     * \param[in] default_color Color to be used if no r-g-b color values are given for PLY nodes
     * \param[in] silent (optional) If set to true, output messaged will be disabled
     * \return Vector of UUIDs for each primitive generated from the OBJ model
     */
    std::vector<uint> loadOBJ(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color, bool silent=false );
    
    //! Load geometry contained in a Wavefront OBJ file (.obj)
    /**
     * \param[in] filename name of OBJ file
     * \param[in] origin (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
     * \param[in] height A z-scaling factor is applied to make the model 'height' tall. If height=0, no scaling is applied
     * \param[in] rotation Spherical rotation of PLY object about origin
     * \param[in] default_color Color to be used if no r-g-b color values are given for PLY nodes
     * \param[in] upaxis Direction of "up" vector used when creating OBJ file (one of "XUP", "YUP", or "ZUP" - "ZUP" is default).
     * \param[in] silent (optional) If set to true, output messaged will be disabled
     * \return Vector of UUIDs for each primitive generated from the PLY model
     */
    std::vector<uint> loadOBJ(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color, const char* upaxis, bool silent=false );


    //! Load geometry contained in a Wavefront OBJ file (.obj)
    /**
     * \param[in] filename name of OBJ file
     * \param[in] origin (x,y,z) coordinate of PLY object origin (i.e., coordinate shift)
     * \param[in] scale (x,y,z) scaling factor to be applied to OBJ vertex coordinates (if scale.x=scale.y=scale.z=0, no scaling is applied).
     * \param[in] rotation Spherical rotation of PLY object about origin
     * \param[in] default_color Color to be used if no r-g-b color values are given for PLY nodes
     * \param[in] upaxis Direction of "up" vector used when creating OBJ file (one of "XUP", "YUP", or "ZUP" - "ZUP" is default).
     * \param[in] silent (optional) If set to true, output messaged will be disabled
     * \return Vector of UUIDs for each primitive generated from the PLY model
     */
    std::vector<uint> loadOBJ(const char* filename, const vec3 &origin, const helios::vec3 &scale, const SphericalCoord &rotation, const RGBcolor &default_color, const char* upaxis, bool silent=false );

    //! Write geometry in the Context to a Wavefront file (.obj)
    /**
     * \param[in] filename Base filename of .obj and .mtl file
     */
    void writeOBJ( const std::string &filename ) const;

    //! Write geometry in the Context to a Wavefront file (.obj) for a subset of UUIDs
    /**
     * \param[in] filename Base filename of .obj and .mtl file
     * \param[in] UUIDs Vector of UUIDs for which geometry should be written
     */
    void writeOBJ( const std::string &filename, const std::vector<uint> &UUIDs ) const;

    //! Write geometry in the Context to a Wavefront file (.obj)
    /**
     * \param[in] filename Base filename of .obj and .mtl file
     * \param[in] UUIDs Vector of UUIDs for which geometry should be written
     * \param[in] primitive_dat_fields A .dat file will be written containing primitive data given in this vector (for Unity visualization)
     */
    void writeOBJ( const std::string &filename, const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_dat_fields ) const;
    
    //! Set simulation date by day, month, year
    /**
     * \param[in] day Day of the month (1-31)
     * \param[in] month Month of year (1-12)
     * \param[in] year Year in YYYY format
     * \sa \ref getDate()
     */
    void setDate( int day, int month, int year );
    
    //! Set simulation date by Date vector
    /**
     * \param[in] date Date vector
     * \sa getDate()
     */
    void setDate( helios::Date date );
    
    //! Set simulation date by Julian day
    /**
     * \param[in] Julian_day Julian day of year (1-366)
     * \param[in] year Year in YYYY format. Note: this is used to determine leap years.
     * \sa \ref getDate()
     */
    void setDate( int Julian_day, int year );
    
    //! Get simulation date
    /**
     * \return Date vector
     * \sa setDate(), getJulianDate()
     */
    helios::Date getDate() const;
    
    //! Get a string corresponding to the month of the simulation date
    /**
     * \return Month string (e.g., Jan, Feb, Mar, etc)
     * \sa setDate(), \ref getJulianDate()
     */
    const char* getMonthString() const;
    
    //! Get simulation date by Julian day
    /**
     * \return Julian day of year (1-366)
     * \sa setDate(), getDate()
     */
    int getJulianDate() const;
    
    //! Set simulation time
    /**
     * \param[in] minute Minute of hour (0-59)
     * \param[in] hour Hour of day (0-23)
     * \sa getTime()
     * \sa setSunDirection()
     */
    void setTime( int minute, int hour );
    
    //! Set simulation time
    /**
     * \param[in] second Second of minute (0-59)
     * \param[in] minute Minute of hour (0-59)
     * \param[in] hour Hour of day (0-23)
     * \sa getTime()
     * \sa setSunDirection()
     */
    void setTime( int second, int minute, int hour );
    
    //! Set simulation time using Time vector
    /**
     * \param[in] time Time vector
     * \sa getTime()
     * \sa setSunDirection()
     */
    void setTime( helios::Time time );
    
    //! Get the simulation time
    /**
     * \return Time vector
     * \sa setTime()
     */
    helios::Time getTime() const;
    
    //! Draw a random number from a uniform distribution between 0 and 1
    /**
     * \return Random float between 0 and 1
     */
    float randu();
    
    //! Draw a random number from a uniform distribution with specified range
    /**
     * \param[in] min Minimum value of random uniform distribution (float)
     * \param[in] max Maximum value of random uniform distribution (float)
     * \return Random float between 'min' and 'max'
     */
    float randu( float min, float max );
    
    //! Draw a random number from a uniform distribution with specified range
    /**
     * \param[in] min Minimum value of random uniform distribution (integer)
     * \param[in] max Maximum value of random uniform distribution (integer)
     * \return Random integer between 'min' and 'max'
     */
    int randu( int min, int max );
    
    //! Draw a random number from a normal distribution with mean = 0, stddev = 1
    /**
     * \return Random float from normal distribution
     */
    float randn();
    
    //! Draw a random number from a normal distribution with specified mean and standard deviation
    /**
     * \param[in] mean Mean value of random distribution
     * \param[in] stddev Standard deviation of random normal distribution
     * \note If standard deviation is specified as negative, the absolute value is used.
     */
    float randn( float mean, float stddev );

    //! Duplicate primitive data to create a copy of the primitive data with a new name/label
    /**
     * \param[in] existing_data_label Name of existing primitive data to be duplicated
     * \param[in] copy_data_label Name of new primitive data copy
     */
    void duplicatePrimitiveData( const char* existing_data_label, const char* copy_data_label );

    //! Calculate mean of primitive data values (float) for a subset of primitives
    /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] mean Mean of primitive data
     */
     void calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, float &mean ) const;

     //! Calculate mean of primitive data values (double) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] mean Mean of primitive data
      */
     void calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, double &mean ) const;

     //! Calculate mean of primitive data values (vec2) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] mean Mean of primitive data
      */
     void calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &mean ) const;

     //! Calculate mean of primitive data values (vec3) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] mean Mean of primitive data
      */
     void calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &mean ) const;

     //! Calculate mean of primitive data values (vec4) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] mean Mean of primitive data
      */
     void calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &mean ) const;

     //! Calculate mean of primitive data values (float) for a subset of primitives, where each value in the mean calculation is weighted by the primitive's one-sided surface area
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] awt_mean Area-weighted mean of primitive data
      */
     void calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, float &awt_mean ) const;

     //! Calculate mean of primitive data values (double) for a subset of primitives, where each value in the mean calculation is weighted by the primitive's one-sided surface area
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] awt_mean Area-weighted mean of primitive data
      */
     void calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, double &awt_mean ) const;

     //! Calculate mean of primitive data values (vec2) for a subset of primitives, where each value in the mean calculation is weighted by the primitive's one-sided surface area
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] awt_mean Area-weighted mean of primitive data
      */
     void calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &awt_mean ) const;

     //! Calculate mean of primitive data values (vec2) for a subset of primitives, where each value in the mean calculation is weighted by the primitive's one-sided surface area
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] awt_mean Area-weighted mean of primitive data
      */
     void calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &awt_mean ) const;

     //! Calculate mean of primitive data values (vec2) for a subset of primitives, where each value in the mean calculation is weighted by the primitive's one-sided surface area
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] awt_mean Area-weighted mean of primitive data
      */
     void calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &awt_mean ) const;

     //! Calculate sum of primitive data values (float) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] sum Sum of primitive data
      */
     void calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, float &sum ) const;

     //! Calculate sum of primitive data values (double) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] sum Sum of primitive data
      */
     void calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, double &sum ) const;

     //! Calculate sum of primitive data values (vec2) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] sum Sum of primitive data
      */
     void calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &sum ) const;

     //! Calculate sum of primitive data values (vec3) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] sum Sum of primitive data
      */
     void calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &sum ) const;

     //! Calculate sum of primitive data values (vec4) for a subset of primitives
     /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label Primitive data label
     * \param[out] sum Sum of primitive data
      */
     void calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &sum ) const;

  //! Calculate sum of primitive data values (float) for a subset of primitives, where each value in the sum calculation is weighted by the primitive's one-sided surface area
  /**
  * \param[in] UUIDs Universal unique identifiers of primitives
  * \param[in] label Primitive data label
  * \param[out] awt_sum Sum of primitive data
   */
  void calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, float &awt_sum ) const;

  //! Calculate sum of primitive data values (double) for a subset of primitives, where each value in the sum calculation is weighted by the primitive's one-sided surface area
  /**
  * \param[in] UUIDs Universal unique identifiers of primitives
  * \param[in] label Primitive data label
  * \param[out] awt_sum Sum of primitive data
   */
  void calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, double &sum ) const;

  //! Calculate sum of primitive data values (vec2) for a subset of primitives, where each value in the sum calculation is weighted by the primitive's one-sided surface area
  /**
  * \param[in] UUIDs Universal unique identifiers of primitives
  * \param[in] label Primitive data label
  * \param[out] awt_sum Sum of primitive data
   */
  void calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &sum ) const;

  //! Calculate sum of primitive data values (vec3) for a subset of primitives, where each value in the sum calculation is weighted by the primitive's one-sided surface area
  /**
  * \param[in] UUIDs Universal unique identifiers of primitives
  * \param[in] label Primitive data label
  * \param[out] awt_sum Sum of primitive data
   */
  void calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &sum ) const;

  //! Calculate sum of primitive data values (vec4) for a subset of primitives, where each value in the sum calculation is weighted by the primitive's one-sided surface area
  /**
  * \param[in] UUIDs Universal unique identifiers of primitives
  * \param[in] label Primitive data label
  * \param[out] awt_sum Sum of primitive data
   */
  void calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &sum ) const;

  //! Multiply primitive data values by a constant scaling factor for a subset of primitives.
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives
   * \param[in] label Primitive data label
   * \param[in] scaling_factor Factor to scale primitive data
   * \note An error will be thrown for primitive data types of uint, int, int2, int3, int4, and string.
   */
  void scalePrimitiveData( const std::vector<uint> &UUIDs, const std::string &label, float scaling_factor );

    //! Multiply primitive data values by a constant scaling factor for all primitives.
    /**
     * \param[in] label Primitive data label
     * \param[in] scaling_factor Factor to scale primitive data
     * \note An error will be thrown for primitive data types of uint, int, int2, int3, int4, and string.
     */
    void scalePrimitiveData( const std::string &label, float scaling_factor );

    //! Increase value of primitive data (int) by some value
    /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label primitive data label string
     * \param[in] increment Value to increment primitive data by
     * \note If primitive data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, int increment );

    //! Increase value of primitive data (uint) by some value
    /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label primitive data label string
     * \param[in] increment Value to increment primitive data by
     * \note If primitive data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, uint increment );

    //! Increase value of primitive data (float) by some value
    /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label primitive data label string
     * \param[in] increment Value to increment primitive data by
     * \note If primitive data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, float increment );

    //! Increase value of primitive data (double) by some value
    /**
     * \param[in] UUIDs Universal unique identifiers of primitives
     * \param[in] label primitive data label string
     * \param[in] increment Value to increment primitive data by
     * \note If primitive data is a vector, each value in the vector will be incremented by the same amount
    */
    void incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, double increment );

  //! Sum multiple primitive data values for each primitive together and store result in new primitive data
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives
   * \param[in] primitive_data_labels Vector of primitive data labels, whose values for each primitive will be summed
   * \param[in] result_primitive_data_label New primitive data label where result will be stored
   * \note An error will be thrown for primitive data types of string.
   */
  void aggregatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label  );

  //! Multiply primitive data values for each primitive together and store result in new primitive data
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives
   * \param[in] primitive_data_labels Vector of primitive data labels, whose values for each primitive will be multiplied
   * \param[in] result_primitive_data_label New primitive data label where result will be stored
   * \note An error will be thrown for primitive data types of string.
   */
  void aggregatePrimitiveDataProduct( const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label  );

  //! Sum the one-sided surface area of a group of primitives
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives
   * \return Sum of primitive area
   */
  float sumPrimitiveSurfaceArea( const std::vector<uint> &UUIDs ) const;

  //! Filter a set of primitives based on their primitive data and a condition and float value
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives to be considered.
   * \param[in] primitive_data_label Name of primitive data.
   * \param[in] filter_value Value to use as filter.
   * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
   * \return Set of UUIDs for primitives with primitive data meeting the condition of 'primitive_data' 'comparator' 'filter_value'.
   * \note If primitive data does not exist, or primitive data does not have type of 'float', the primitive is excluded.
   */
  std::vector<uint> filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, float filter_value, const std::string &comparator );

  //! Filter a set of primitives based on their primitive data and a condition and double value
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives to be considered.
   * \param[in] primitive_data_label Name of primitive data.
   * \param[in] filter_value Value to use as filter.
   * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
   * \return Set of UUIDs for primitives with primitive data meeting the condition of 'primitive_data' 'comparator' 'filter_value'.
   * \note If primitive data does not exist, or primitive data does not have type of 'double', the primitive is excluded.
   */
  std::vector<uint> filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, double filter_value, const std::string &comparator );

  //! Filter a set of primitives based on their primitive data and a condition and int value
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives to be considered.
   * \param[in] primitive_data_label Name of primitive data.
   * \param[in] filter_value Value to use as filter.
   * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
   * \return Set of UUIDs for primitives with primitive data meeting the condition of 'primitive_data' 'comparator' 'filter_value'.
   * \note If primitive data does not exist, or primitive data does not have type of 'int', the primitive is excluded.
   */
  std::vector<uint> filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, int filter_value, const std::string &comparator );

  //! Filter a set of primitives based on their primitive data and a condition and uint value
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives to be considered.
   * \param[in] primitive_data_label Name of primitive data.
   * \param[in] filter_value Value to use as filter.
   * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
   * \return Set of UUIDs for primitives with primitive data meeting the condition of 'primitive_data' 'comparator' 'filter_value'.
   * \note If primitive data does not exist, or primitive data does not have type of 'uint', the primitive is excluded.
   */
  std::vector<uint> filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, uint filter_value, const std::string &comparator );

  //! Get set of primitives whose primitive data matches a given string
  /**
   * \param[in] UUIDs Universal unique identifiers of primitives to be considered.
   * \param[in] primitive_data_label Name of primitive data.
   * \param[in] filter_value String to use as filter.
   * \return Set of UUIDs for primitives with primitive data matching the string 'filter_value'.
   * \note If primitive data does not exist, or primitive data does not have type of 'std::string', the primitive is excluded.
   */
  std::vector<uint> filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, const std::string &filter_value );

    //! Filter a set of compound objects based on their object data and a condition and float value
    /**
     * \param[in] objIDs Universal unique identifiers of objects to be considered.
     * \param[in] object_data_label Name of object data.
     * \param[in] filter_value Value to use as filter.
     * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
     * \return Set of objIDs for objects with object data meeting the condition of 'object_data' 'comparator' 'filter_value'.
     * \note If object data does not exist, or object data does not have type of 'float', the object is excluded.
     */
    std::vector<uint> filterObjectsByData( const std::vector<uint> &objIDs, const std::string &object_data_label, float filter_value, const std::string &comparator );

    //! Filter a set of compound objects based on their object data and a condition and double value
    /**
     * \param[in] objIDs Universal unique identifiers of objects to be considered.
     * \param[in] object_data_label Name of object data.
     * \param[in] filter_value Value to use as filter.
     * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
     * \return Set of objIDs for objects with object data meeting the condition of 'object_data' 'comparator' 'filter_value'.
     * \note If object data does not exist, or object data does not have type of 'double', the object is excluded.
     */
    std::vector<uint> filterObjectsByData( const std::vector<uint> &objIDs, const std::string &object_data_label, double filter_value, const std::string &comparator );

    //! Filter a set of compound objects based on their object data and a condition and int value
    /**
     * \param[in] objIDs Universal unique identifiers of objects to be considered.
     * \param[in] object_data_label Name of object data.
     * \param[in] filter_value Value to use as filter.
     * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
     * \return Set of objIDs for objects with object data meeting the condition of 'object_data' 'comparator' 'filter_value'.
     * \note If object data does not exist, or object data does not have type of 'int', the object is excluded.
     */
    std::vector<uint> filterObjectsByData( const std::vector<uint> &objIDs, const std::string &object_data_label, int filter_value, const std::string &comparator );

    //! Filter a set of compound objects based on their object data and a condition and uint value
    /**
     * \param[in] objIDs Universal unique identifiers of objects to be considered.
     * \param[in] object_data_label Name of object data.
     * \param[in] filter_value Value to use as filter.
     * \param[in] comparator Operator to use for filter comparison. Valid comparators are '==', '>', '\<', '>=','\<='.
     * \return Set of objIDs for objects with object data meeting the condition of 'object_data' 'comparator' 'filter_value'.
     * \note If object data does not exist, or object data does not have type of 'uint', the object is excluded.
     */
    std::vector<uint> filterObjectsByData( const std::vector<uint> &objIDs, const std::string &object_data_label, uint filter_value, const std::string &comparator );

    //! Get set of compound objects whose object data matches a given string
    /**
     * \param[in] objIDs Universal unique identifiers of objects to be considered.
     * \param[in] object_data_label Name of object data.
     * \param[in] filter_value String to use as filter.
     * \return Set of objIDs for objects with object data matching the string 'filter_value'.
     * \note If object data does not exist, or object data does not have type of 'std::string', the object is excluded.
     */
    std::vector<uint> filterObjectsByData( const std::vector<uint> &objIDs, const std::string &object_data_label, const std::string &filter_value );
    
};

}



#endif
