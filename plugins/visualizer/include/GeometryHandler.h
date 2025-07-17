/** \file "GeometryHandler.h"

Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef VISUALIZER_GEOMETRY_HANDLER_H
#define VISUALIZER_GEOMETRY_HANDLER_H

#include "global.h"
#include <unordered_set>

class GeometryHandler {
private:
    struct PrimitiveIndexMap;

public:

    //! Enum representing different types of visualizer geometry.
    /**
     * This enumeration is used to define the geometry types
     * that can be handled by the visualizer.
    */
    enum VisualizerGeometryType {
        GEOMETRY_TYPE_RECTANGLE = 1,
        GEOMETRY_TYPE_TRIANGLE = 2,
        GEOMETRY_TYPE_POINT = 3,
        GEOMETRY_TYPE_LINE = 4
    };

    //! Constructor for the GeometryHandler class.
    GeometryHandler() : random_generator(std::random_device{}()) {
        for ( const auto &geometry_type : all_geometry_types ) {
            face_index_data[geometry_type] = {};
            vertex_data[geometry_type] = {};
            normal_data[geometry_type] = {};
            color_data[geometry_type] = {};
            uv_data[geometry_type] = {};
            texture_flag_data[geometry_type] = {};
            texture_ID_data[geometry_type] = {};
            coordinate_flag_data[geometry_type] = {};
            visible_flag_data[geometry_type] = {};
            context_geometry_flag_data[geometry_type] = {};
            delete_flag_data[geometry_type] = {};
        }

    }

    //! Pre-allocate space in geometry buffers for known number of patches and triangles
    /**
     *
     * This calls the std::vector::reserve() method to pre-allocate space in the geometry buffers for the specified number of patches and triangles.
     *
     * \param[in] primitive_count Number of patches to allocate space for
     * \param geometry_type
     */

    void allocateBufferSize(size_t primitive_count, VisualizerGeometryType geometry_type);

    /**
     * \brief Adds a geometric element to the geometry handler using provided vertices, color, UV mapping, and texture details.
     *
     * There are three different ways to call this method based on how the geometry should be colored:
     * 1. Using a flat RGB color: Pass and empty 'uvs' vector, which overrides any texture behavior (in this case the values of "texture", "textureID", and "override_texture_color" are ignored).
     * 2. Using an image texture: Pass a non-empty 'uvs' vector and set "override_texture_color" to false, which will be used for texture mapping (the value of 'color' will be ignored). Note that a valid texture ID must also be given.
     * 3. Using a flat RGB color with a texture alpha mask: Pass a non-empty 'uvs' vector and set "override_texture_color" to true, which will use the RGB color for the triangle but mask the shape using the alpha channel of the texture.
     *
     * There are four different geometric types that can be added:
     * 1. GEOMETRY_TYPE_PATCH: Defined by four vertex coordinates. Can be textured or colored. Size argument is ignored.
     * 2. GEOMETRY_TYPE_TRIANGLE: Defined by three vertex coordinates. Can be textured or colored. Size argument is ignored.
     * 3. GEOMETRY_TYPE_POINT: Defined by a single vertex coordinate. Texture-related arguments are ignored.
     * 4. GEOMETRY_TYPE_LINE: Defined by two vertex coordinates. Texture-related arguments are ignored.
     *
     * \param[in] UUID Unique identifier for the triangle.
     * \param[in] geometry_type
     * \param[in] vertices 3D vertex positions of the geometry, provided as a vector of vec3's.
     * \param[in] color RGBA color to apply to the geometry.
     * \param[in] uvs UV texture coordinates for the geometry, provided as a vector of vec2's.
     * \param[in] textureID Identifier for the texture to use.
     * \param[in] override_texture_color Boolean flag indicating whether to override the texture with the color provided.
     * \param has_glyph_texture
     * \param[in] coordinate_system Coordinate system to be used when specifying spatial coordinates. Should be one of "Visualizer::COORDINATES_WINDOW_NORMALIZED" or "Visualizer::COORDINATES_CARTESIAN".
     * \param[in] visible_flag Boolean flag to determine if the geometry is initially visible.
     * \param[in] iscontextgeometry True if geometry is from the Context, false if geometry is added manually through the visualizer
     * \param[in] size [optional] Size of the point or line in pixel units. This is ignored for GEOMETRY_TYPE_PATCH and GEOMETRY_TYPE_TRIANGLE.
     */
    void addGeometry(size_t UUID, const VisualizerGeometryType &geometry_type, const std::vector<helios::vec3> &vertices, const helios::RGBAcolor &color,
                     const std::vector<helios::vec2> &uvs, int textureID, bool override_texture_color, bool has_glyph_texture, uint coordinate_system, bool visible_flag, bool iscontextgeometry, int size = 0);

    //! Mark a geometry primitive as modified
    void markDirty(size_t UUID);

    //! Retrieve the set of modified primitives
    [[nodiscard]] const std::unordered_set<size_t> &getDirtyUUIDs() const;

    //! Clear the list of modified primitives
    void clearDirtyUUIDs();

    [[nodiscard]] bool doesGeometryExist(size_t UUID) const;

    [[nodiscard]] std::vector<size_t> getAllGeometryIDs() const;

    [[nodiscard]] size_t getPrimitiveCount( bool include_deleted = true ) const;

    /**
     * \brief Retrieves the count of rectangles in the geometry handler.
     *
     * \param[in] include_deleted [optional] If true, includes rectangles marked as deleted in the count.
     * \return The number of rectangles, optionally including those marked as deleted.
     */
    [[nodiscard]] size_t getRectangleCount( bool include_deleted = true ) const;

    /**
     * \brief Retrieves the number of triangles in the geometry.
     *
     * \param[in] include_deleted [optional] If true, includes triangles marked as deleted in the count.
     * \return The total number of triangles, optionally including those marked as deleted.
     */
    [[nodiscard]] size_t getTriangleCount( bool include_deleted = true ) const;

    /**
     * \brief Retrieves the count of points in the geometry.
     *
     * \param[in] include_deleted [optional] If true, includes points marked as deleted in the count. Otherwise, only non-deleted points are counted.
     * \return Total number of points, optionally including deleted points.
     */
    [[nodiscard]] size_t getPointCount( bool include_deleted = true ) const;

    /**
     * \brief Retrieves the count of line geometries.
     *
     * \param[in] include_deleted [optional] If true, includes geometries marked as deleted in the count. Defaults to false.
     * \return The total number of line geometries, including or excluding deleted ones based on the parameter.
     */
    [[nodiscard]] size_t getLineCount( bool include_deleted = true ) const;

    /**
     * \brief Retrieves a pointer to the face index data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the face index data is requested.
     * \return Constant pointer to the vector containing face index data.
     */
    [[nodiscard]] const std::vector<int>* getFaceIndexData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Sets the vertices for a geometry element identified by a unique UUID.
     *
     * Updates the geometry buffer with the specified 3D vertex positions.
     * The number of vertices in the provided vector must correspond to the expected vertex count
     * based on the geometry type associated with the given UUID.
     *
     * \param[in] UUID Unique identifier for the geometry element to update.
     * \param[in] vertices 3D vertex positions to set, provided as a vector of helios::vec3.
     */
    void setVertices(size_t UUID, const std::vector<helios::vec3> &vertices);

    /**
     * \brief Retrieves the vertices associated with the geometry of a given UUID.
     *
     * \param[in] UUID Unique identifier representing the geometry whose vertices need to be fetched.
     * \return A vector containing the vertices as helios::vec3 objects.
     */
    [[nodiscard]] std::vector<helios::vec3> getVertices( size_t UUID ) const;

    /**
     * \brief Returns the vertex count for a given geometry type.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the vertex count is required.
     * \return The vertex count corresponding to the specified geometry type.
     * \note Returns 0 for unsupported or invalid geometry types.
    */
    static char getVertexCount(const VisualizerGeometryType &geometry_type);

    /**
     * \brief Retrieves a pointer to the vertex data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the vertex data is requested.
     * \return Constant pointer to the vector containing vertex data.
     */
    [[nodiscard]] const std::vector<float>* getVertexData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Retrieves the normal vector associated with the geometry of a given UUID.
     *
     * \param[in] UUID Unique identifier representing the geometry whose normal needs to be fetched.
     * \return Normal vector as helios::vec3 object.
     */
    [[nodiscard]] helios::vec3 getNormal( size_t UUID ) const;

    /**
     * \brief Retrieves a pointer to the normal data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the normal data is requested.
     * \return Constant pointer to the vector containing normal data.
     */
    [[nodiscard]] const std::vector<float>* getNormalData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Sets the color for the geometry identified by a specific UUID.
     *
     * \param[in] UUID The unique identifier of the geometry to modify.
     * \param[in] color The RGBA color to set for the specified geometry.
     */
    void setColor(size_t UUID, const helios::RGBAcolor &color);

    /**
     * \brief Retrieves the color associated with a specific UUID.
     *
     * \param[in] UUID Unique identifier of the geometry whose color is to be retrieved.
     * \return The RGBAcolor corresponding to the specified UUID.
     */
    [[nodiscard]] helios::RGBAcolor getColor( size_t UUID ) const;

    /**
     * \brief Retrieves a pointer to the color data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the color data is requested.
     * \return Pointer to the vector containing color data.
     */
    [[nodiscard]] const std::vector<float>* getColorData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Assigns UV coordinates to the specified geometry by UUID.
     *
     * \param[in] UUID A unique identifier for the geometry to which UV coordinates are being assigned.
     * \param[in] uvs A collection of UV coordinates to set for the specified geometry.
     *                The size of this vector must match the vertex count of the geometry.
     * \note This function only applies to certain geometry types where UV mapping is applicable.
     */
    void setUVs(size_t UUID, const std::vector<helios::vec2> &uvs);

    /**
     * \brief Retrieves the UV coordinates associated with a specific geometry UUID.
     *
     * \param[in] UUID A unique identifier representing the geometry whose UV coordinates are to be fetched.
     * \return A vector containing the UV coordinates as vec2 objects.
     */
    [[nodiscard]] std::vector<helios::vec2> getUVs( size_t UUID ) const;

    /**
     * \brief Retrieves a pointer to the UV data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the UV data is requested.
     * \return Pointer to a vector containing the UV data.
     */
    [[nodiscard]] const std::vector<float>* getUVData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Sets the texture ID for the given geometry identified by its UUID.
     *
     * \param[in] UUID Unique identifier of the geometry.
     * \param[in] textureID Texture identifier to associate with the specified geometry.
     */
    void setTextureID(size_t UUID, int textureID);

    /**
     * \brief Retrieves the texture ID associated with the given UUID.
     *
     * \param[in] UUID Unique identifier for the geometry object.
     * \return The texture ID linked to the specified UUID.
     */
    [[nodiscard]] int getTextureID( size_t UUID ) const;

    /**
     * \brief Retrieves a pointer to the texture ID data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the texture ID data is requested.
     * \return Pointer to the vector containing texture ID data.
     */
    [[nodiscard]] const std::vector<int>* getTextureIDData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Overrides the texture color attribute for a given geometry primitive.
     *
     * \param[in] UUID The unique identifier for the geometry primitive.
     */
    void overrideTextureColor(size_t UUID);

    /**
     * \brief Enables texture color usage for the specified geometry.
     *
     * \param[in] UUID Unique identifier associated with the geometry for which texture color will be enabled.
     */
    void useTextureColor(size_t UUID);

    /**
     * \brief Retrieves a pointer to the texture flag data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the texture flag data is requested.
     * \return Pointer to the vector containing texture flag data.
     */
    [[nodiscard]] const std::vector<int>* getTextureFlagData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Sets the visibility state for a given geometry component.
     *
     * \param[in] UUID The unique identifier of the geometry component.
     * \param[in] isvisible The visibility state to set (true for visible, false for hidden).
     */
    void setVisibility(size_t UUID, bool isvisible);

    /**
     * \brief Checks if a primitive with the given UUID is visible.
     *
     * \param[in] UUID The unique identifier of the primitive to check.
     * \return True if the primitive is visible, otherwise false.
     */
    [[nodiscard]] bool isPrimitiveVisible( size_t UUID ) const;

    /**
     * \brief Retrieves a pointer to the vector containing visibility flag data.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the visibility flag data is requested.
     * \return Pointer to the vector of boolean values representing visibility flags.
     */
    [[nodiscard]] const std::vector<char> *getVisibilityFlagData_ptr(VisualizerGeometryType geometry_type) const;

    /**
     * \brief Retrieves a pointer to the coordinate flag data associated with the specified geometry type.
     *
     * \param[in] geometry_type The type of visualizer geometry for which the coordinate flag data is requested.
     * \return A constant pointer to a vector of integers representing the coordinate flag data for the given geometry type.
     */
    [[nodiscard]] const std::vector<int>* getCoordinateFlagData_ptr(VisualizerGeometryType geometry_type) const;

    [[nodiscard]] bool getDeleteFlag( size_t UUID ) const;

    /**
     * \brief Marks a geometry resource identified by its unique UUID for deletion.
     *
     * \param[in] UUID Identifier of the geometry resource to delete.
     */
    void deleteGeometry(size_t UUID);

    /**
     * \brief Marks a geometry resource identified by its unique UUID for deletion.
     *
     * \param[in] UUIDs Vector of identifiers of the geometry resource to delete.
     */
    void deleteGeometry(const std::vector<size_t> &UUIDs);

    /**
     * \brief Clears all geometry data managed by the GeometryHandler.
     *
     * Resets all associated data structures for geometry types and
     * resets counters for deleted primitives.
     */
    void clearAllGeometry();

    /**
     * \brief Clears context-specific geometry data.
     *
     * Marks context-associated geometry data for deletion
     * based on predefined flags and defragments the buffers.
     */
    void clearContextGeometry();

    /**
     * \brief Computes the axis-aligned bounding box for the geometry domain.
     *
     * \param[out] xbounds Outputs the minimum and maximum x-coordinates in the domain.
     * \param[out] ybounds Outputs the minimum and maximum y-coordinates in the domain.
     * \param[out] zbounds Outputs the minimum and maximum z-coordinates in the domain.
     */
    void getDomainBoundingBox( helios::vec2& xbounds, helios::vec2& ybounds, helios::vec2& zbounds ) const;

    /**
     * \brief Computes the bounding sphere of the domain.
     *
     * \param[out] center The computed center of the bounding sphere.
     * \param[out] radius The computed radius of the bounding sphere.
     */
    void getDomainBoundingSphere(helios::vec3& center, helios::vec3 &radius) const;

    /**
     * \brief Generates and returns a random unique identifier value as a size_t.
     *
     * \return A randomly generated UUID value.
     */
    size_t sampleUUID();

    //! Retrieve internal buffer indices for a primitive
    [[nodiscard]] const PrimitiveIndexMap &getIndexMap(size_t UUID) const;

    constexpr static std::array<VisualizerGeometryType,4> all_geometry_types = {
        GEOMETRY_TYPE_RECTANGLE,
        GEOMETRY_TYPE_TRIANGLE,
        GEOMETRY_TYPE_POINT,
        GEOMETRY_TYPE_LINE
    };

private:

    struct PrimitiveIndexMap {
        VisualizerGeometryType geometry_type;
        size_t face_index_index;
        size_t vertex_index;
        size_t normal_index;
        size_t uv_index;
        size_t color_index;
        size_t texture_flag_index;
        size_t texture_ID_index;
        size_t coordinate_flag_index;
        size_t visible_index;
        size_t context_geometry_flag_index;
        size_t delete_flag_index;
    };

    std::mt19937_64 random_generator;

    std::unordered_map<size_t, PrimitiveIndexMap> UUID_map;

    std::unordered_map< VisualizerGeometryType, std::vector<int> > face_index_data;
    std::unordered_map< VisualizerGeometryType, std::vector<float> > vertex_data;
    std::unordered_map< VisualizerGeometryType, std::vector<float> > normal_data;
    std::unordered_map< VisualizerGeometryType, std::vector<float> > uv_data;
    std::unordered_map< VisualizerGeometryType, std::vector<float> > color_data;
    std::unordered_map< VisualizerGeometryType, std::vector<int> > texture_flag_data;
    std::unordered_map< VisualizerGeometryType, std::vector<int> > texture_ID_data;
    std::unordered_map< VisualizerGeometryType, std::vector<int> > coordinate_flag_data;
    std::unordered_map< VisualizerGeometryType, std::vector<char> > visible_flag_data;
    std::unordered_map< VisualizerGeometryType, std::vector<bool> > context_geometry_flag_data;
    std::unordered_map< VisualizerGeometryType, std::vector<bool> > delete_flag_data;

    std::unordered_set<size_t> dirty_UUIDs;

    size_t deleted_primitive_count = 0;

    /**
     * \brief Defragments the internal geometry buffers by removing unused or deleted primitives.
     *
     * This function reorganizes internal data structures to consolidate active primitives and
     * removes data associated with deleted primitives, optimizing memory usage. It operates across
     * all geometry types managed by the system.
     */
    void defragmentBuffers();

    /**
     * \brief Registers the UUID with the specified visualizer geometry type.
     *
     * \param[in] UUID Unique identifier to register.
     * \param[in] geometry_type The type of visualizer geometry associated with the UUID.
     */
    void registerUUID(size_t UUID, const VisualizerGeometryType &geometry_type);

};

#endif