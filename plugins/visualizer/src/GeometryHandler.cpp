/** \file "GeometryHandler.cpp"

Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "GeometryHandler.h"

void GeometryHandler::allocateBufferSize(size_t primitive_count, VisualizerGeometryType geometry_type) {

    char vertex_count = getVertexCount( geometry_type );

    face_index_data[geometry_type].reserve( face_index_data[geometry_type].size() + primitive_count );
    vertex_data[geometry_type].reserve(vertex_data[geometry_type].size() + primitive_count * vertex_count * 3);
    normal_data[geometry_type].reserve(normal_data[geometry_type].size() + primitive_count * vertex_count * 3);
    color_data[geometry_type].reserve(color_data[geometry_type].size() + primitive_count * 4);
    uv_data[geometry_type].reserve(uv_data[geometry_type].size() + primitive_count * vertex_count * 2);
    texture_flag_data[geometry_type].reserve( texture_flag_data[geometry_type].size() + primitive_count);
    texture_ID_data[geometry_type].reserve(texture_ID_data[geometry_type].size() + primitive_count);
    coordinate_flag_data[geometry_type].reserve(coordinate_flag_data[geometry_type].size() + primitive_count);
    delete_flag_data[geometry_type].reserve(delete_flag_data[geometry_type].size() + primitive_count);
    context_geometry_flag_data[geometry_type].reserve(context_geometry_flag_data[geometry_type].size() + primitive_count);

}

void GeometryHandler::addGeometry(size_t UUID, const VisualizerGeometryType& geometry_type, const std::vector<helios::vec3>& vertices,
                                  const helios::RGBAcolor &color, const std::vector<helios::vec2>& uvs, int textureID,
                                  bool override_texture_color, bool has_glyph_texture, uint coordinate_system, bool visible_flag, bool iscontextgeometry, int size) {

    char vertex_count = getVertexCount( geometry_type );

#ifdef HELIOS_DEBUG
    assert( vertices.size() == vertex_count );
    assert( uvs.empty() || uvs.size() == vertex_count );
#endif

    std::vector<helios::vec3> vertices_copy = vertices; // make a copy so it can be modified

    bool geometry_is_new = false;
    if ( !doesGeometryExist(UUID) ) {
        registerUUID( UUID, geometry_type );
        geometry_is_new = true;
    }

    if( coordinate_system == 0 ){ //No vertex transformation (i.e., identity matrix)

        //NOTE for vertex positions: OpenGL window coordinates range from -1 to 1, but our rectangle coordinates are from 0 to 1 ---- need to convert
        for ( auto & vertex : vertices_copy ) {
            vertex.x = 2.f * vertex.x - 1.f;
            vertex.y = 2.f * vertex.y - 1.f;
        }

    }

    size_t vertex_index = UUID_map.at(UUID).vertex_index;
    size_t normal_index = UUID_map.at(UUID).normal_index;
    size_t uv_index = UUID_map.at(UUID).uv_index;
    size_t color_index = UUID_map.at(UUID).color_index;
    size_t texture_flag_index = UUID_map.at(UUID).texture_flag_index;
    size_t texture_ID_index = UUID_map.at(UUID).texture_ID_index;

    for ( char i=0; i<vertex_count; i++ ) {

        if ( geometry_is_new ) {
            face_index_data[geometry_type].push_back( static_cast<int>(visible_flag_data[geometry_type].size()) );

            vertex_data[geometry_type].push_back(vertices_copy.at(i).x);
            vertex_data[geometry_type].push_back(vertices_copy.at(i).y);
            vertex_data[geometry_type].push_back(vertices_copy.at(i).z);
        }else {
            //face index doesn't need to be updated

            vertex_data[geometry_type].at( vertex_index) = vertices_copy.at(i).x;
            vertex_data[geometry_type].at( vertex_index+1) = vertices_copy.at(i).y;
            vertex_data[geometry_type].at( vertex_index+2) = vertices_copy.at(i).z;
            vertex_index += 3;
        }

        if ( ( geometry_type == GEOMETRY_TYPE_TRIANGLE || geometry_type == GEOMETRY_TYPE_RECTANGLE ) && !uvs.empty() ) { //if (u, v)'s are provided, color triangle based on texture map image

            if ( geometry_is_new ) {
                uv_data[geometry_type].push_back(uvs.at(i).x);
                uv_data[geometry_type].push_back(1.f - uvs.at(i).y);

                if (i == 0) { //only need to do this one time
                    if (has_glyph_texture) {
                        texture_flag_data[geometry_type].push_back(3); // 3 means use RGB for color and red channel of texture for alpha
                    } else if (override_texture_color) { //if texture color is overridden, color primitive based on RGB color but use texture transparence for alpha mask
                        texture_flag_data[geometry_type].push_back(2); // 2 means use RGB for color and texture transparency for alpha
                    } else {
                        texture_flag_data[geometry_type].push_back(1); // 1 means use texture for color and transparency for alpha
                    }
                    texture_ID_data[geometry_type].push_back(textureID);
                }
            }else {
                uv_data[geometry_type].at(uv_index) = uvs.at(i).x;
                uv_data[geometry_type].at(uv_index+1) = 1.f - uvs.at(i).y;
                uv_index += 2;

                if ( i==0 ) { //only need to do this one time
                    if ( has_glyph_texture ) {
                        texture_flag_data[geometry_type].at(texture_flag_index) = 3; // 3 means use RGB for color and red channel of texture for alpha
                    }else if ( override_texture_color ) { //if texture color is overridden, color primitive based on RGB color but use texture transparence for alpha mask
                        texture_flag_data[geometry_type].at(texture_flag_index) = 2; // 2 means use RGB for color and texture transparency for alpha
                    }else {
                        texture_flag_data[geometry_type].at(texture_flag_index) = 1; // 1 means use texture for color and transparency for alpha
                    }
                    texture_flag_index++;
                    texture_ID_data[geometry_type].at(texture_ID_index) = textureID;
                    texture_ID_index++;

                }
            }
        } else { //if (u,v)'s are not provided, color primitive based on RGB color

            if ( geometry_is_new ) {
                uv_data[geometry_type].push_back( 0.f );
                uv_data[geometry_type].push_back( 0.f );

                if ( i==0 ) { //only need to do this one time
                    if ( has_glyph_texture ) {
                        texture_flag_data[geometry_type].push_back( 3 ); // 3 means use RGB for color and red channel of texture for alpha
                        texture_ID_data[geometry_type].push_back( textureID );
                    }else {
                        texture_flag_data[geometry_type].push_back( 0 ); // 0 means use RGB color
                        texture_ID_data[geometry_type].push_back( 0 );
                    }
                }
            }else{
                uv_data[geometry_type].at(uv_index) = 0.f;
                uv_data[geometry_type].at(uv_index+1) = 0.f;
                uv_index += 2;

                if ( i==0 ) { //only need to do this one time
                    if ( has_glyph_texture ) {
                        texture_flag_data[geometry_type].at(texture_flag_index) = 3; // 3 means use RGB for color and red channel of texture for alpha
                        texture_ID_data[geometry_type].at(texture_ID_index) = textureID;
                    }else {
                        texture_flag_data[geometry_type].at(texture_flag_index) = 0; // 0 means use RGB color
                        texture_ID_data[geometry_type].at(texture_ID_index) = 0 ;
                    }
                    texture_flag_index++;
                    texture_ID_index++;
                }
            }
        }
    }

    helios::vec3 normal;
    if (geometry_type == GEOMETRY_TYPE_TRIANGLE || geometry_type == GEOMETRY_TYPE_RECTANGLE) {
        normal = normalize(cross(vertices_copy.at(1) - vertices_copy.at(0), vertices_copy.at(2) - vertices_copy.at(0)));
    }
    if ( geometry_is_new ) {
        normal_data[geometry_type].push_back(normal.x);
        normal_data[geometry_type].push_back(normal.y);
        normal_data[geometry_type].push_back(normal.z);

        color_data[geometry_type].push_back(color.r);
        color_data[geometry_type].push_back(color.g);
        color_data[geometry_type].push_back(color.b);
        color_data[geometry_type].push_back(color.a);

        coordinate_flag_data[geometry_type].push_back(coordinate_system);

        visible_flag_data[geometry_type].push_back(visible_flag);

        delete_flag_data[geometry_type].push_back(false);

        context_geometry_flag_data[geometry_type].push_back(iscontextgeometry);
    }else {
        normal_data[geometry_type].at(normal_index) = normal.x;
        normal_data[geometry_type].at(normal_index+1) = normal.y;
        normal_data[geometry_type].at(normal_index+2) = normal.z;

        color_data[geometry_type].at(color_index) = color.r;
        color_data[geometry_type].at(color_index+1) = color.g;
        color_data[geometry_type].at(color_index+2) = color.b;
        color_data[geometry_type].at(color_index+3) = color.a;
    }

    markDirty(UUID);

}

bool GeometryHandler::doesGeometryExist(size_t UUID) const {
    return (UUID_map.find(UUID) != UUID_map.end());
}

std::vector<size_t> GeometryHandler::getAllGeometryIDs() const {
    std::vector<size_t> result;
    result.reserve(UUID_map.size());
    for ( const auto&[UUID, primitivemap] : UUID_map ) {
        if ( getDeleteFlag(UUID) ) {
            // Only include non-deleted geometries
            continue;
        }
        result.push_back(UUID);
    }
    return result;
}

size_t GeometryHandler::getPrimitiveCount( bool include_deleted ) const {
    size_t count = 0;
    if ( include_deleted ) {
        for ( auto &type : all_geometry_types ) {
            count += delete_flag_data.at(type).size();
        }
        return count;
    }

    for ( auto &type : all_geometry_types ) {
        count += std::count(delete_flag_data.at(type).begin(), delete_flag_data.at(type).end(), false);
    }
    return count;
}

[[nodiscard]] size_t GeometryHandler::getRectangleCount( bool include_deleted ) const {
    VisualizerGeometryType type = GEOMETRY_TYPE_RECTANGLE;
    if ( include_deleted ) {
        return delete_flag_data.at(type).size();
    }

    size_t count = std::count(delete_flag_data.at(type).begin(), delete_flag_data.at(type).end(), false);
    return count;
}

[[nodiscard]] size_t GeometryHandler::getTriangleCount( bool include_deleted ) const {
    VisualizerGeometryType type = GEOMETRY_TYPE_TRIANGLE;
    if ( include_deleted ) {
        return delete_flag_data.at(type).size();
    }

    size_t count = std::count(delete_flag_data.at(type).begin(), delete_flag_data.at(type).end(), false);
    return count;
}

[[nodiscard]] size_t GeometryHandler::getPointCount( bool include_deleted ) const {
    VisualizerGeometryType type = GEOMETRY_TYPE_POINT;
    if ( include_deleted ) {
        return delete_flag_data.at(type).size();
    }

    size_t count = std::count(delete_flag_data.at(type).begin(), delete_flag_data.at(type).end(), false);
    return count;
}

[[nodiscard]] size_t GeometryHandler::getLineCount( bool include_deleted ) const {
    VisualizerGeometryType type = GEOMETRY_TYPE_LINE;
    if ( include_deleted ) {
        return delete_flag_data.at(type).size();
    }

    size_t count = std::count(delete_flag_data.at(type).begin(), delete_flag_data.at(type).end(), false);
    return count;
}

const std::vector<int>* GeometryHandler::getFaceIndexData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( face_index_data.find(geometry_type) != face_index_data.end() );
#endif
    return &face_index_data.at(geometry_type);
}

void GeometryHandler::setVertices( size_t UUID, const std::vector<helios::vec3>& vertices ) {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const char vertex_count = getVertexCount( index_map.geometry_type );

#ifdef HELIOS_DEBUG
    assert( vertices.size() == vertex_count );
#endif

    const size_t vertex_ind = index_map.vertex_index;

    int ii=0;
    for ( int i=0; i<vertex_count; i++ ) {
        vertex_data[index_map.geometry_type].at( vertex_ind + ii+0 ) = vertices.at( i ).x;
        vertex_data[index_map.geometry_type].at( vertex_ind + ii+1 ) = vertices.at( i ).y;
        vertex_data[index_map.geometry_type].at( vertex_ind + ii+2 ) = vertices.at( i ).z;
        ii+=3;
    }

    const size_t normal_ind = index_map.normal_index;

    const helios::vec3 normal = normalize( cross( vertices.at(1) - vertices.at(0), vertices.at(2) - vertices.at(0) ) );
    normal_data[index_map.geometry_type].at( normal_ind + ii+0 ) = normal.x;
    normal_data[index_map.geometry_type].at( normal_ind + ii+1 ) = normal.y;
    normal_data[index_map.geometry_type].at( normal_ind + ii+2 ) = normal.z;

    markDirty(UUID);

}

std::vector<helios::vec3> GeometryHandler::getVertices( size_t UUID ) const {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t vertex_ind = index_map.vertex_index;

    const char vertex_count = getVertexCount( index_map.geometry_type );

    std::vector<helios::vec3> vertices(vertex_count);

    for ( int i=0; i<vertex_count; i++ ) {
        vertices.at(i).x = vertex_data.at(index_map.geometry_type).at( vertex_ind + i*3+0 );
        vertices.at(i).y = vertex_data.at(index_map.geometry_type).at( vertex_ind + i*3+1 );
        vertices.at(i).z = vertex_data.at(index_map.geometry_type).at( vertex_ind + i*3+2 );
    }

    return vertices;

}

const std::vector<float>* GeometryHandler::getVertexData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( vertex_data.find(geometry_type) != vertex_data.end() );
#endif
    return &vertex_data.at(geometry_type);
}

helios::vec3 GeometryHandler::getNormal( size_t UUID ) const {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t normal_ind = index_map.normal_index;

    helios::vec3 normal;

    normal.x = normal_data.at(index_map.geometry_type).at(normal_ind + 0);
    normal.y = normal_data.at(index_map.geometry_type).at(normal_ind + 1);
    normal.z = normal_data.at(index_map.geometry_type).at( normal_ind + 2);

    return normal;

}

const std::vector<float>* GeometryHandler::getNormalData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( normal_data.find(geometry_type) != normal_data.end() );
#endif
    return &normal_data.at(geometry_type);
}

void GeometryHandler::setColor( size_t UUID, const helios::RGBAcolor &color ) {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t color_ind = index_map.color_index;

    color_data[index_map.geometry_type].at( color_ind + 0 ) = color.r;
    color_data[index_map.geometry_type].at( color_ind + 1 ) = color.g;
    color_data[index_map.geometry_type].at( color_ind + 2 ) = color.b;
    color_data[index_map.geometry_type].at( color_ind + 3 ) = color.a;

    markDirty(UUID);

}

helios::RGBAcolor GeometryHandler::getColor( size_t UUID ) const {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t color_ind = index_map.color_index;

    const helios::RGBAcolor color{
        color_data.at(index_map.geometry_type).at( color_ind ),
        color_data.at(index_map.geometry_type).at( color_ind + 1 ),
        color_data.at(index_map.geometry_type).at( color_ind + 2 ),
        color_data.at(index_map.geometry_type).at( color_ind + 3 )
    };

    return color;

}

const std::vector<float>* GeometryHandler::getColorData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( color_data.find(geometry_type) != color_data.end() );
#endif
    return &color_data.at(geometry_type);
}

void GeometryHandler::setUVs( size_t UUID, const std::vector<helios::vec2>& uvs ) {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    if ( index_map.geometry_type == GEOMETRY_TYPE_LINE || index_map.geometry_type == GEOMETRY_TYPE_POINT ) {
        // These types do not have texture mapping
        return;
    }

    const char vertex_count = getVertexCount( index_map.geometry_type );

#ifdef HELIOS_DEBUG
    assert( uvs.size() == vertex_count );
#endif

    const size_t uv_ind = index_map.uv_index;

    int ii=0;
    for ( int i=0; i<vertex_count; i++ ) {
        uv_data.at(index_map.geometry_type).at( uv_ind + ii ) = uvs.at( i ).x;
        uv_data.at(index_map.geometry_type).at( uv_ind + ii+1 ) = uvs.at( i ).y;
        ii+=2;
    }

    markDirty(UUID);

}

std::vector<helios::vec2> GeometryHandler::getUVs( size_t UUID ) const {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t uv_ind = index_map.uv_index;

    const char vertex_count = getVertexCount( index_map.geometry_type );

    std::vector<helios::vec2> uvs(vertex_count);

    for ( int i=0; i<vertex_count; i++ ) {
        uvs.at(i).x = uv_data.at(index_map.geometry_type).at( uv_ind + i*2 );
        uvs.at(i).x = uv_data.at(index_map.geometry_type).at( uv_ind + i*2+1 );
    }

    return uvs;

}

const std::vector<float>* GeometryHandler::getUVData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( uv_data.find(geometry_type) != uv_data.end() );
#endif
    return &uv_data.at(geometry_type);
}

void GeometryHandler::setTextureID( size_t UUID, int textureID ) {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    if ( index_map.geometry_type == GEOMETRY_TYPE_LINE || index_map.geometry_type == GEOMETRY_TYPE_POINT ) {
        // These types do not have texture mapping
        return;
    }

    const size_t texture_ind = index_map.texture_ID_index;

    texture_ID_data.at(index_map.geometry_type).at( texture_ind ) = textureID;

    markDirty(UUID);

}

int GeometryHandler::getTextureID( size_t UUID ) const {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t texture_ID_ind = index_map.texture_ID_index;

    return texture_ID_data.at(index_map.geometry_type).at( texture_ID_ind );

}

const std::vector<int>* GeometryHandler::getTextureIDData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( texture_ID_data.find(geometry_type) != texture_ID_data.end() );
#endif
    return &texture_ID_data.at(geometry_type);
}

void GeometryHandler::overrideTextureColor( size_t UUID ) {
        
#ifdef HELIOS_DEBUG
        assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    if ( index_map.geometry_type == GEOMETRY_TYPE_LINE || index_map.geometry_type == GEOMETRY_TYPE_POINT ) {
        // These types do not have texture mapping
        return;
    }

    const size_t texture_flag_ind = index_map.texture_flag_index;

    const int current_flag = texture_flag_data.at(index_map.geometry_type).at(texture_flag_ind);
    if (current_flag == 1) {
        // \todo This might be a problem in the case that the primitive does not have a texture with a transparency. In that case we would want to set to 0
        texture_flag_data.at(index_map.geometry_type).at(texture_flag_ind) = 2;
    }

    markDirty(UUID);

}

void GeometryHandler::useTextureColor( size_t UUID ) {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    if ( index_map.geometry_type == GEOMETRY_TYPE_LINE || index_map.geometry_type == GEOMETRY_TYPE_POINT ) {
        // These types do not have texture mapping
        return;
    }

    const size_t texture_flag_ind = index_map.texture_flag_index;

    // \todo This might be a problem in the case that the primitive does not have a texture image. In this case, we would want to set to 0.
    texture_flag_data.at(index_map.geometry_type).at(texture_flag_ind) = 1;

    markDirty(UUID);

}

const std::vector<int>* GeometryHandler::getTextureFlagData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( texture_flag_data.find(geometry_type) != texture_flag_data.end() );
#endif
    return &texture_flag_data.at(geometry_type);
}

void GeometryHandler::setVisibility( size_t UUID, bool isvisible ) {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t visibile_ind = index_map.visible_index;

    visible_flag_data.at(index_map.geometry_type).at(visibile_ind) = static_cast<char>(isvisible);

    markDirty(UUID);

}

bool GeometryHandler::isPrimitiveVisible( size_t UUID ) const {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t visible_ind = index_map.visible_index;

    return static_cast<bool>(visible_flag_data.at(index_map.geometry_type).at(visible_ind));

}

const std::vector<char> *GeometryHandler::getVisibilityFlagData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( visible_flag_data.find(geometry_type) != visible_flag_data.end() );
#endif
    return &visible_flag_data.at(geometry_type);
}

const std::vector<int>* GeometryHandler::getCoordinateFlagData_ptr(VisualizerGeometryType geometry_type) const {
#ifdef HELIOS_DEBUG
    assert( coordinate_flag_data.find(geometry_type) != coordinate_flag_data.end() );
#endif
    return &coordinate_flag_data.at(geometry_type);
}

bool GeometryHandler::getDeleteFlag( size_t UUID ) const {
#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    const size_t delete_flag_ind = index_map.delete_flag_index;

    return delete_flag_data.at(index_map.geometry_type).at( delete_flag_ind );
}

void GeometryHandler::deleteGeometry( size_t UUID ) {

#ifdef HELIOS_DEBUG
    assert( UUID_map.find(UUID) != UUID_map.end() );
#endif

    const PrimitiveIndexMap &index_map = UUID_map.at(UUID);

    delete_flag_data.at(index_map.geometry_type).at(index_map.delete_flag_index) = true;
    visible_flag_data.at(index_map.geometry_type).at(index_map.visible_index) = false;

    markDirty(UUID);

    deleted_primitive_count ++;

    if ( deleted_primitive_count>250000 ) {
        defragmentBuffers();
    }

}

void GeometryHandler::deleteGeometry(const std::vector<size_t> &UUIDs ) {
    for ( const auto &UUID : UUIDs ) {
        deleteGeometry( UUID );
    }
}

void GeometryHandler::clearAllGeometry() {

    for ( const auto &geometry_type : all_geometry_types ) {
        if ( vertex_data.find(geometry_type) == vertex_data.end() ) {
            continue;
        }

        face_index_data.at(geometry_type).clear();
        vertex_data.at(geometry_type).clear();
        normal_data.at(geometry_type).clear();
        uv_data.at(geometry_type).clear();
        color_data.at(geometry_type).clear();
        texture_flag_data.at(geometry_type).clear();
        texture_ID_data.at(geometry_type).clear();
        coordinate_flag_data.at(geometry_type).clear();
        visible_flag_data.at(geometry_type).clear();
        context_geometry_flag_data.at(geometry_type).clear();
        delete_flag_data.at(geometry_type).clear();
    }

    UUID_map.clear();

    deleted_primitive_count = 0;

}

void GeometryHandler::clearContextGeometry() {

    for ( const auto &geometry_type : all_geometry_types ) {
        if ( vertex_data.find(geometry_type) == vertex_data.end() ) {
            continue;
        }

        for ( size_t i=0; i<context_geometry_flag_data.at(geometry_type).size(); i++ ) {
            assert (context_geometry_flag_data.at(geometry_type).size() == delete_flag_data.at(geometry_type).size());
            if ( context_geometry_flag_data.at(geometry_type).at(i) ) {
                delete_flag_data.at(geometry_type).at(i) = true;
                // visible_flag_data.at(geometry_type).at(i) = false;
                deleted_primitive_count++;
            }
        }
    }

    defragmentBuffers();

}

void GeometryHandler::getDomainBoundingBox(helios::vec2 &xbounds, helios::vec2 &ybounds, helios::vec2 &zbounds) const {

    xbounds.x = 1e8;
    xbounds.y = -1e8;
    ybounds.x = 1e8;
    ybounds.y = -1e8;
    zbounds.x = 1e8;
    zbounds.y = -1e8;

    for (const auto &[UUID, index_map]: UUID_map) {
        // Only primitives with Cartesian coordinates should be included in bounding box
        if (coordinate_flag_data.at(index_map.geometry_type).at(index_map.coordinate_flag_index) == 0) {
            continue;
        }

        const auto &vertices = getVertices(UUID);

        for (const auto &vertex: vertices) {
            if (vertex.x < xbounds.x) {
                xbounds.x = vertex.x;
            }
            if (vertex.x > xbounds.y) {
                xbounds.y = vertex.x;
            }
            if (vertex.y < ybounds.x) {
                ybounds.x = vertex.y;
            }
            if (vertex.y > ybounds.y) {
                ybounds.y = vertex.y;
            }
            if (vertex.z < zbounds.x) {
                zbounds.x = vertex.z;
            }
            if (vertex.z > zbounds.y) {
                zbounds.y = vertex.z;
            }
        }
    }

}

void GeometryHandler::getDomainBoundingSphere(helios::vec3& center, helios::vec3 &radius) const {

    helios::vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox( xbounds, ybounds, zbounds );

    radius = {
        0.5f*(xbounds.y - xbounds.x),
        0.5f*(ybounds.y - ybounds.x),
        0.5f*(zbounds.y - zbounds.x)
    };

    center = {
        xbounds.x + radius.x,
        ybounds.x + radius.y,
        zbounds.x + radius.z
    };

}

size_t GeometryHandler::sampleUUID(){
    std::uniform_int_distribution<size_t> dist(1, (std::numeric_limits<size_t>::max)());
    size_t UUID = 0;
    while ( UUID==0 || UUID_map.find(UUID) != UUID_map.end() ) {
        UUID = dist(random_generator);
    }
    return UUID;
}

const GeometryHandler::PrimitiveIndexMap &GeometryHandler::getIndexMap(size_t UUID) const {
#ifdef HELIOS_DEBUG
    assert(UUID_map.find(UUID) != UUID_map.end());
#endif
    return UUID_map.at(UUID);
}

void GeometryHandler::defragmentBuffers() {

    // If no primitives were deleted, nothing to do
    if (deleted_primitive_count == 0) {
        return;
    }

    // Iterate each geometry type

    for ( const auto &geometry_type : all_geometry_types ) {
        if ( vertex_data.find(geometry_type) == vertex_data.end() ) {
            continue;
        }

        auto &oldFace = face_index_data.at(geometry_type);
        auto &oldVertex = vertex_data.at(geometry_type);
        auto &oldNormal = normal_data.at(geometry_type);
        auto &oldUV = uv_data.at(geometry_type);
        auto &oldColor = color_data.at(geometry_type);
        auto &oldTexFlag = texture_flag_data.at(geometry_type);
        auto &oldTexID = texture_ID_data.at(geometry_type);
        auto &oldCoordFlag = coordinate_flag_data.at(geometry_type);
        auto &oldVisible = visible_flag_data.at(geometry_type);
        auto &oldContextFlag = context_geometry_flag_data.at(geometry_type);
        auto &oldDeleteFlag = delete_flag_data.at(geometry_type);

        // New buffers
        std::vector<VisualizerGeometryType> newType;
        std::vector<float> newVertex, newNormal, newUV, newColor;
        std::vector<int> newFace, newTexFlag, newTexID, newCoordFlag;
        std::vector<bool> newDeleteFlag, newContextFlag;
        std::vector<char> newVisible;

        // Collect UUIDs to drop
        std::vector<size_t> toErase;

        // Walk the UUID_map and rebuild entries
        for (auto &[UUID, prim]: UUID_map) {
            if (prim.geometry_type != geometry_type) {
                continue; // this UUID is another geometry type
            }

            // If marked deleted => drop
            if (oldDeleteFlag[prim.delete_flag_index]) {
                toErase.push_back(UUID);
                continue;
            }

            // Otherwise copy its slices into the new buffers
            const char vcount = getVertexCount(prim.geometry_type);
            const size_t v3 = static_cast<size_t>(vcount) * 3;
            const size_t v2 = static_cast<size_t>(vcount) * 2;

            // Record new base indices
            const size_t fi = newFace.size();
            const size_t vi = newVertex.size();
            const size_t ni = newNormal.size();
            const size_t ui = newUV.size();
            const size_t ci = newColor.size();
            const size_t tfi = newTexFlag.size();
            const size_t tidi = newTexID.size();
            const size_t cfi = newCoordFlag.size();
            const size_t vi2 = newVisible.size();
            const size_t cfi2 = newContextFlag.size();
            const size_t dfi = newDeleteFlag.size();

            // Copy raw data
            newFace.insert( newFace.end(), vcount, scast<int>(newVisible.size()) ); //the new face index should be the new index not just copying the previous value. Note that we take the size of newVisible, but could be any per-face array size.

            newVertex.insert(newVertex.end(),
                             oldVertex.begin() + prim.vertex_index,
                             oldVertex.begin() + prim.vertex_index + v3);

            newNormal.insert(newNormal.end(),
                             oldNormal.begin() + prim.normal_index,
                             oldNormal.begin() + prim.normal_index + 3);

            newUV.insert(newUV.end(),
                         oldUV.begin() + prim.uv_index,
                         oldUV.begin() + prim.uv_index + v2);

            newColor.insert(newColor.end(),
                            oldColor.begin() + prim.color_index,
                            oldColor.begin() + prim.color_index + 4);

            newTexFlag.push_back(oldTexFlag[prim.texture_flag_index]);
            newTexID.push_back(oldTexID[prim.texture_ID_index]);
            newCoordFlag.push_back(oldCoordFlag[prim.coordinate_flag_index]);
            newVisible.push_back(oldVisible[prim.visible_index]);
            newContextFlag.push_back(oldContextFlag[prim.context_geometry_flag_index]);
            newDeleteFlag.push_back(oldDeleteFlag[prim.delete_flag_index]);

            // Update the map entry to point at the new positions
            prim.face_index_index = fi;
            prim.vertex_index = vi;
            prim.normal_index = ni;
            prim.uv_index = ui;
            prim.color_index = ci;
            prim.texture_flag_index = tfi;
            prim.texture_ID_index = tidi;
            prim.coordinate_flag_index = cfi;
            prim.visible_index = vi2;
            prim.context_geometry_flag_index = cfi2;
            prim.delete_flag_index = dfi;
        }

        // Erase deleted UUIDs
        for (auto UUID: toErase) {
            UUID_map.erase(UUID);
        }

        // Swap into place
        oldFace.swap(newFace);
        oldVertex.swap(newVertex);
        oldNormal.swap(newNormal);
        oldUV.swap(newUV);
        oldColor.swap(newColor);
        oldTexFlag.swap(newTexFlag);
        oldTexID.swap(newTexID);
        oldCoordFlag.swap(newCoordFlag);
        oldVisible.swap(newVisible);
        oldContextFlag.swap(newContextFlag);
        oldDeleteFlag.swap(newDeleteFlag);
    }

    // Reset deleted count
    deleted_primitive_count = 0;
}

void GeometryHandler::registerUUID(size_t UUID, const VisualizerGeometryType &geometry_type) {

    UUID_map[UUID] = {
        geometry_type,
        face_index_data.at(geometry_type).size(),
        vertex_data.at(geometry_type).size(),
        normal_data.at(geometry_type).size(),
        uv_data.at(geometry_type).size(),
        color_data.at(geometry_type).size(),
        texture_flag_data.at(geometry_type).size(),
        texture_ID_data.at(geometry_type).size(),
        coordinate_flag_data.at(geometry_type).size(),
        visible_flag_data.at(geometry_type).size(),
        context_geometry_flag_data.at(geometry_type).size(),
        delete_flag_data.at(geometry_type).size()
    };

}

char GeometryHandler::getVertexCount(const VisualizerGeometryType &geometry_type) {

    switch (geometry_type) {
        case GEOMETRY_TYPE_RECTANGLE: return 4;
        case GEOMETRY_TYPE_TRIANGLE:  return 3;
        case GEOMETRY_TYPE_POINT:     return 1;
        case GEOMETRY_TYPE_LINE:      return 2;
        default:
            assert(true);
            return 0;
    }

}

void GeometryHandler::markDirty(size_t UUID) {
    dirty_UUIDs.insert(UUID);
}

const std::unordered_set<size_t> &GeometryHandler::getDirtyUUIDs() const {
    return dirty_UUIDs;
}

void GeometryHandler::clearDirtyUUIDs() {
    dirty_UUIDs.clear();
}