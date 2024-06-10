/** \file "Context_data.cpp" Context primitive data, object data, and global data declarations.

Copyright (C) 2016-2024 Brian Bailey

                    This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "Context.h"

using namespace helios;

// ------ Primitive Data -------- //

void Primitive::setPrimitiveData( const char* label, const int& data ){
  std::vector<int> vec{data};
  primitive_data_int[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT;
}

void Primitive::setPrimitiveData( const char* label, const uint& data ){
  std::vector<uint> vec{data};
  primitive_data_uint[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_UINT;
}

void Primitive::setPrimitiveData( const char* label, const float& data ){
  std::vector<float> vec{data};
  primitive_data_float[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_FLOAT;
}

void Primitive::setPrimitiveData( const char* label, const double& data ){
  std::vector<double> vec{data};
  primitive_data_double[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_DOUBLE;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec2& data ){
  std::vector<vec2> vec{data};
  primitive_data_vec2[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_VEC2;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec3& data ){
  std::vector<vec3> vec{data};
  primitive_data_vec3[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_VEC3;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec4& data ){
  std::vector<vec4> vec{data};
  primitive_data_vec4[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_VEC4;
}

void Primitive::setPrimitiveData( const char* label, const helios::int2& data ){
  std::vector<int2> vec{data};
  primitive_data_int2[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT2;
}

void Primitive::setPrimitiveData( const char* label, const helios::int3& data ){
  std::vector<int3> vec{data};
  primitive_data_int3[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT3;
}

void Primitive::setPrimitiveData( const char* label, const helios::int4& data ){
  std::vector<int4> vec{data};
  primitive_data_int4[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT4;
}

void Primitive::setPrimitiveData( const char* label, const std::string& data ){
  std::vector<std::string> vec{data};
  primitive_data_string[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_STRING;
}

void Primitive::setPrimitiveData( const char* label, HeliosDataType type, uint size, void* data ){

  primitive_data_types[label] = type;

  if( type==HELIOS_TYPE_INT ){

    int* data_ptr = (int*)data;

    std::vector<int> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int[label] = vec;

  }else if( type==HELIOS_TYPE_UINT ){

    uint* data_ptr = (uint*)data;

    std::vector<uint> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_uint[label] = vec;

  }else if( type==HELIOS_TYPE_FLOAT ){

    auto* data_ptr = (float*)data;

    std::vector<float> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_float[label] = vec;

  }else if( type==HELIOS_TYPE_DOUBLE ){

    auto* data_ptr = (double*)data;

    std::vector<double> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_double[label] = vec;

  }else if( type==HELIOS_TYPE_VEC2 ){

    auto* data_ptr = (vec2*)data;

    std::vector<vec2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_vec2[label] = vec;

  }else if( type==HELIOS_TYPE_VEC3 ){

    auto* data_ptr = (vec3*)data;

    std::vector<vec3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_vec3[label] = vec;

  }else if( type==HELIOS_TYPE_VEC4 ){

    auto* data_ptr = (vec4*)data;

    std::vector<vec4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_vec4[label] = vec;

  }else if( type==HELIOS_TYPE_INT2 ){

    auto* data_ptr = (int2*)data;

    std::vector<int2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int2[label] = vec;

  }else if( type==HELIOS_TYPE_INT3 ){

    auto* data_ptr = (int3*)data;

    std::vector<int3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int3[label] = vec;

  }else if( type==HELIOS_TYPE_INT4 ){

    auto* data_ptr = (int4*)data;

    std::vector<int4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int4[label] = vec;

  }else if( type==HELIOS_TYPE_STRING ){

    auto* data_ptr = (std::string*)data;

    std::vector<std::string> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_string[label] = vec;

  }

}

void Primitive::getPrimitiveData( const char* label, int& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT ){
    std::vector<int> d = primitive_data_int.at(label);
    data = d.at(0);
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT ){
    std::vector<int> d = primitive_data_int.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int." );
  }

}

void Primitive::getPrimitiveData( const char* label, uint& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = primitive_data_uint.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type uint, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type uint." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<uint>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = primitive_data_uint.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type uint, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type uint." );
  }

}

void Primitive::getPrimitiveData( const char* label, float& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = primitive_data_float.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type float, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type float." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<float>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = primitive_data_float.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type float, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type float." );
  }

}

void Primitive::getPrimitiveData( const char* label, double& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = primitive_data_double.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type double, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type double." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<double>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = primitive_data_double.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type double, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type double." );
  }

}

void Primitive::getPrimitiveData( const char* label, vec2& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = primitive_data_vec2.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type vec2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec2." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<vec2>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = primitive_data_vec2.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type vec2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec2." );
  }

}

void Primitive::getPrimitiveData( const char* label, vec3& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = primitive_data_vec3.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type vec3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec3." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<vec3>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = primitive_data_vec3.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type vec3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec3." );
  }

}

void Primitive::getPrimitiveData( const char* label, vec4& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = primitive_data_vec4.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type vec4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec4." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<vec4>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = primitive_data_vec4.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type vec4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec4." );
  }

}

void Primitive::getPrimitiveData( const char* label, int2& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = primitive_data_int2.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int2." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int2>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = primitive_data_int2.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int2." );
  }

}

void Primitive::getPrimitiveData( const char* label, int3& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = primitive_data_int3.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int3." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int3>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = primitive_data_int3.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int3." );
  }

}

void Primitive::getPrimitiveData( const char* label, int4& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = primitive_data_int4.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int4." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int4>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = primitive_data_int4.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type int4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int4." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::string& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = primitive_data_string.at(label);
    data = d.front();
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type string, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type string." );
  }

}

void Primitive::getPrimitiveData( const char* label, std::vector<std::string>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = primitive_data_string.at(label);
    data = d;
  }else{
    helios_runtime_error( "ERROR (Primitive::getPrimitiveData): Attempted to get data for type string, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type string." );
  }

}

HeliosDataType Primitive::getPrimitiveDataType( const char* label ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveDataType): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  return primitive_data_types.at(label);

}

uint Primitive::getPrimitiveDataSize( const char* label ) const{

  if( !doesPrimitiveDataExist( label ) ){
    helios_runtime_error( "ERROR (Primitive::getPrimitiveDataSize): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) );
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT ){
    return primitive_data_int.at(label).size();
  }else if( type==HELIOS_TYPE_UINT ){
    return primitive_data_uint.at(label).size();
  }else if( type==HELIOS_TYPE_FLOAT ){
    return primitive_data_float.at(label).size();
  }else if( type==HELIOS_TYPE_DOUBLE ){
    return primitive_data_double.at(label).size();
  }else if( type==HELIOS_TYPE_VEC2 ){
    return primitive_data_vec2.at(label).size();
  }else if( type==HELIOS_TYPE_VEC3 ){
    return primitive_data_vec3.at(label).size();
  }else if( type==HELIOS_TYPE_VEC4 ){
    return primitive_data_vec4.at(label).size();
  }else if( type==HELIOS_TYPE_INT2 ){
    return primitive_data_int2.at(label).size();
  }else if( type==HELIOS_TYPE_INT3 ){
    return primitive_data_int3.at(label).size();
  }else if( type==HELIOS_TYPE_INT4 ){
    return primitive_data_int4.at(label).size();
  }else if( type==HELIOS_TYPE_STRING ){
    return primitive_data_string.at(label).size();
  }else{
    assert( false );
  }

  return 0;

}

void Primitive::clearPrimitiveData( const char* label ){

  if( !doesPrimitiveDataExist( label ) ){
    return;
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT ){
    primitive_data_int.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_UINT ){
    primitive_data_uint.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_FLOAT ){
    primitive_data_float.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_DOUBLE ){
    primitive_data_double.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_VEC2 ){
    primitive_data_vec2.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_VEC3 ){
    primitive_data_vec3.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_VEC4 ){
    primitive_data_vec4.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_INT2 ){
    primitive_data_int2.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_INT3 ){
    primitive_data_int3.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_INT4 ){
    primitive_data_int4.erase(label);
    primitive_data_types.erase(label);
  }else if( type==HELIOS_TYPE_STRING ){
    primitive_data_string.erase(label);
    primitive_data_types.erase(label);
  }else{
    assert(false);
  }

}

bool Primitive::doesPrimitiveDataExist( const char* label ) const{

  if( primitive_data_types.find(label) == primitive_data_types.end() ){
    return false;
  }else{
    return true;
  }

}

std::vector<std::string> Primitive::listPrimitiveData() const{

  std::vector<std::string> labels(primitive_data_types.size());

  size_t i=0;
  for(const auto & primitive_data_type : primitive_data_types){
    labels.at(i) = primitive_data_type.first;
    i++;
  }

  return labels;

}

void Context::setPrimitiveData( const uint& UUID, const char* label, const int& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const uint& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const float& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const double& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::vec2& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::vec3& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::vec4& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::int2& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::int3& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::int4& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const std::string& data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, HeliosDataType type, uint size, void* data ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error( "ERROR (Context::setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->setPrimitiveData(label,type,size,data);
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const int& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const uint& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const float& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const double& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec2& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec3& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec4& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int2& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int3& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int4& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const std::string& data ){
  for( uint UUID : UUIDs){
    setPrimitiveData( UUID, label, data );
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const int& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const uint& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const float& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const double& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec2& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec3& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec4& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int2& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int3& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int4& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const std::string& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const int& data ){
  for(const auto & j : UUIDs){
    for( const auto& UUID : j ){
      setPrimitiveData( UUID, label, data );
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const uint& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const float& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const double& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec2& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec3& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec4& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int2& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int3& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int4& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const std::string& data ){
  for(const auto & j : UUIDs){
    for( const auto& i : j ) {
      for (const auto &UUID: i) {
        setPrimitiveData(UUID, label, data);
      }
    }
  }
}

void Context::getPrimitiveData(uint UUID, const char* label, int& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, uint& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<uint>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, float& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<float>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, double& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<double>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec2& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec2>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec3& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec3>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec4& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec4>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int2& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int2>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int3& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int3>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int4& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int4>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::string& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<std::string>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

HeliosDataType Context::getPrimitiveDataType( const uint UUID, const char* label )const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveDataType): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  return primitives.at(UUID)->getPrimitiveDataType(label);
}

uint Context::getPrimitiveDataSize( const uint UUID, const char* label )const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveDataSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  return primitives.at(UUID)->getPrimitiveDataSize(label);
}

bool Context::doesPrimitiveDataExist( const uint UUID, const char* label ) const{
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::doesPrimitiveDataExist): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  return primitives.at(UUID)->doesPrimitiveDataExist(label);
}

void Context::clearPrimitiveData( const uint UUID, const char* label ){
  if( primitives.find(UUID) == primitives.end() ){
    helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
  }
  primitives.at(UUID)->clearPrimitiveData(label);
}

void Context::clearPrimitiveData( const std::vector<uint>& UUIDs, const char* label ){
  for( unsigned int UUID : UUIDs){
    if( primitives.find(UUID) == primitives.end() ){
      helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
    primitives.at(UUID)->clearPrimitiveData(label);
  }
}

void Context::copyPrimitiveData( uint UUID, uint oldUUID){
  //copy the primitive data
  std::vector<std::string> plabel = getPrimitivePointer_private(UUID)->listPrimitiveData();
  for(auto & p : plabel){

    HeliosDataType type = getPrimitiveDataType( UUID, p.c_str() );

    if( type==HELIOS_TYPE_INT ){
      std::vector<int> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_UINT ){
      std::vector<uint> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_UINT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_FLOAT ){
      std::vector<float> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_FLOAT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_DOUBLE ){
      std::vector<double> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_DOUBLE, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC2 ){
      std::vector<vec2> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_VEC2, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC3 ){
      std::vector<vec3> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_VEC3, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC4 ){
      std::vector<vec4> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_VEC4, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT2 ){
      std::vector<int2> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT2, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT3 ){
      std::vector<int3> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT3, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT4 ){
      std::vector<int4> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT4, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_STRING ){
      std::vector<std::string> pdata;
      getPrimitiveData( UUID, p.c_str(), pdata );
      setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_STRING, pdata.size(), &pdata.at(0) );
    }else{
      assert(false);
    }

  }
}

void Context::renamePrimitiveData( uint UUID, const char* old_label, const char* new_label ){

    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (Context::renamePrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( !primitives.at(UUID)->doesPrimitiveDataExist(old_label) ){
        helios_runtime_error("ERROR (Context::renamePrimitiveData): Primitive data of " + std::string(old_label) + " does not exist for primitive " + std::to_string(UUID) + ".");
    }

    duplicatePrimitiveData( UUID, old_label, new_label );
    clearPrimitiveData( UUID, old_label );

}

void Context::duplicatePrimitiveData( uint UUID, const char* old_label, const char* new_label ){

    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (Context::duplicatePrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( !primitives.at(UUID)->doesPrimitiveDataExist(old_label) ){
        helios_runtime_error("ERROR (Context::duplicatePrimitiveData): Primitive data of " + std::string(old_label) + " does not exist for primitive " + std::to_string(UUID) + ".");
    }

    HeliosDataType type = getPrimitiveDataType( UUID, old_label );

    if( type==HELIOS_TYPE_INT ){
        std::vector<int> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_INT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_UINT ){
        std::vector<uint> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_UINT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_FLOAT ){
        std::vector<float> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_FLOAT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_DOUBLE ){
        std::vector<double> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_DOUBLE, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_VEC2, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_VEC3, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_VEC4, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT2 ){
        std::vector<int2> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_INT2, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT3 ){
        std::vector<int3> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_INT3, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT4 ){
        std::vector<int4> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_INT4, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_STRING ){
        std::vector<std::string> pdata;
        getPrimitiveData( UUID, old_label, pdata );
        setPrimitiveData( UUID, new_label, HELIOS_TYPE_STRING, pdata.size(), &pdata.at(0) );
    }else{
        assert(false);
    }

}

std::vector<std::string> Context::listPrimitiveData(uint UUID) const{
  return getPrimitivePointer_private(UUID)->listPrimitiveData();
}

void Context::duplicatePrimitiveData( const char* existing_data_label, const char* copy_data_label ){

    for( auto primitive : primitives){
        if( primitive.second->doesPrimitiveDataExist(existing_data_label) ){
            HeliosDataType type = primitive.second->getPrimitiveDataType(existing_data_label);
            if( type==HELIOS_TYPE_FLOAT ){
                std::vector<float> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_DOUBLE ) {
                std::vector<double> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_INT ) {
                std::vector<int> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_UINT ) {
                std::vector<uint> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_VEC2 ) {
                std::vector<vec2> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_VEC3 ) {
                std::vector<vec3> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_VEC4 ) {
                std::vector<vec4> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_INT2 ) {
                std::vector<int2> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_INT3 ) {
                std::vector<int3> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_STRING ){
                std::vector<std::string> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }
        }
    }


}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, float &mean ) const{
    float value;
    float sum = 0.f;
    size_t count = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
            getPrimitiveData(UUID,label.c_str(),value);
            sum += value;
            count++;
        }

    }

    if( count==0 ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = 0;
    }else{
        mean = sum/float(count);
    }

}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, double &mean ) const{
    double value;
    double sum = 0.f;
    size_t count = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
            getPrimitiveData(UUID,label.c_str(),value);
            sum += value;
            count++;
        }

    }

    if( count==0 ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = 0;
    }else{
        mean = sum/float(count);
    }

}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &mean ) const{
    vec2 value;
    vec2 sum(0.f,0.f);
    size_t count = 0;
    for (uint UUID : UUIDs) {

        if ( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(UUID, label.c_str()) == HELIOS_TYPE_VEC2 ) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            count++;
        }
    }

    if (count == 0) {
        std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = make_vec2(0,0);
    } else {
        mean = sum / float(count);
    }
}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &mean ) const{
    vec3 value;
    vec3 sum(0.f,0.f,0.f);
    size_t count = 0;
    for (uint UUID : UUIDs) {

        if ( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(UUID, label.c_str()) == HELIOS_TYPE_VEC3 ) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            count++;
        }
    }

    if (count == 0) {
        std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = make_vec3(0,0,0);
    } else {
        mean = sum / float(count);
    }
}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &mean ) const{
    vec4 value;
    vec4 sum(0.f,0.f,0.f,0.f);
    size_t count = 0;
    for (uint UUID : UUIDs) {

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(UUID, label.c_str()) == HELIOS_TYPE_VEC4 ) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            count++;
        }
    }

    if (count == 0) {
        std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = make_vec4(0,0,0,0);
    } else {
        mean = sum / float(count);
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, float &awt_mean ) const{
    float value, A;
    float sum = 0.f;
    float area = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
            getPrimitiveData(UUID,label.c_str(),value);
            A = getPrimitiveArea(UUID);
            sum += value*A;
            area += A;
        }

    }

    if( area==0 ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = 0;
    }else{
        awt_mean = sum/area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, double &awt_mean ) const{
    double value;
    float A;
    double sum = 0.f;
    double area = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
            getPrimitiveData(UUID,label.c_str(),value);
            A = getPrimitiveArea(UUID);
            sum += value*double(A);
            area += A;
        }

    }

    if( area==0 ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = 0;
    }else{
        awt_mean = sum/area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &awt_mean ) const{
    vec2 value;
    float A;
    vec2 sum(0.f,0.f);
    float area = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC2 ){
            getPrimitiveData(UUID,label.c_str(),value);
            A = getPrimitiveArea(UUID);
            sum = sum + (value*A);
            area += A;
        }

    }

    if( area==0 ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = make_vec2(0,0);
    }else{
        awt_mean = sum/area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &awt_mean ) const{
    vec3 value;
    float A;
    vec3 sum(0.f,0.f,0.f);
    float area = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC3 ){
            getPrimitiveData(UUID,label.c_str(),value);
            A = getPrimitiveArea(UUID);
            sum = sum + (value*A);
            area += A;
        }

    }

    if( area==0 ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = make_vec3(0,0,0);
    }else{
        awt_mean = sum/area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &awt_mean ) const{
    vec4 value;
    float A;
    vec4 sum(0.f,0.f,0.f,0.f);
    float area = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC4 ){
            getPrimitiveData(UUID,label.c_str(),value);
            A = getPrimitiveArea(UUID);
            sum = sum + (value*A);
            area += A;
        }

    }

    if( area==0 ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = make_vec4(0,0,0,0);
    }else{
        awt_mean = sum/area;
    }
}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, float &sum ) const{

    float value;
    sum = 0.f;
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
            getPrimitiveData(UUID,label.c_str(),value);
            sum += value;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, double &sum ) const{

    double value;
    sum = 0.f;
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
            getPrimitiveData(UUID,label.c_str(),value);
            sum += value;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &sum ) const{

    vec2 value;
    sum = make_vec2(0.f,0.f);
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC2 ){
            getPrimitiveData(UUID,label.c_str(),value);
            sum = sum + value;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &sum ) const{

    vec3 value;
    sum = make_vec3(0.f,0.f,0.f);
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC3 ){
            getPrimitiveData(UUID,label.c_str(),value);
            sum = sum + value;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &sum ) const{

    vec4 value;
    sum = make_vec4(0.f,0.f,0.f,0.f);
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID)  && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC4 ){
            getPrimitiveData(UUID,label.c_str(),value);
            sum = sum + value;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, float &awt_sum ) const{

    float value;
    awt_sum = 0.f;
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
            float area = getPrimitiveArea(UUID);
            getPrimitiveData(UUID,label.c_str(),value);
            awt_sum += value*area;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, double &awt_sum ) const{

    double value;
    awt_sum = 0.f;
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
            float area = getPrimitiveArea(UUID);
            getPrimitiveData(UUID,label.c_str(),value);
            awt_sum += value*area;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &awt_sum ) const{

    vec2 value;
    awt_sum = make_vec2(0.f,0.f);
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC2 ){
            float area = getPrimitiveArea(UUID);
            getPrimitiveData(UUID,label.c_str(),value);
            awt_sum = awt_sum + value*area;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &awt_sum ) const{

    vec3 value;
    awt_sum = make_vec3(0.f,0.f,0.f);
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC3 ){
            float area = getPrimitiveArea(UUID);
            getPrimitiveData(UUID,label.c_str(),value);
            awt_sum = awt_sum + value*area;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &awt_sum ) const{

    vec4 value;
    awt_sum = make_vec4(0.f,0.f,0.f,0.F);
    bool added_to_sum = false;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID)  && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC4 ){
            float area = getPrimitiveArea(UUID);
            getPrimitiveData(UUID,label.c_str(),value);
            awt_sum = awt_sum + value*area;
            added_to_sum = true;
        }

    }

    if( !added_to_sum ) {
        std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }

}

void Context::scalePrimitiveData( const std::vector<uint> &UUIDs, const std::string &label, float scaling_factor ){

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;
    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            primitives_not_exist++;
            continue;
        }
        if( !doesPrimitiveDataExist(UUID, label.c_str()) ){
            primitive_data_not_exist++;
            continue;
        }
        HeliosDataType data_type = getPrimitiveDataType(UUID,label.c_str());
        if( data_type==HELIOS_TYPE_FLOAT ){
            float data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_DOUBLE ){
            double data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_VEC2 ){
            vec2 data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_VEC3 ){
            vec3 data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_VEC4 ){
            vec4 data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else{
            helios_runtime_error("ERROR (Context::scalePrimitiveData): This operation only supports primitive data of type float, double, vec2, vec3, and vec4.");
        }
    }

    if( primitives_not_exist>0 ){
        std::cout << "WARNING (Context::scalePrimitiveData): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if( primitive_data_not_exist>0 ){
        std::cout << "WARNING (Context::scalePrimitiveData): Primitive data did not exist for " << primitive_data_not_exist << " primitives, and thus no scaling was applied." << std::endl;
    }

}

void Context::scalePrimitiveData( const std::string &label, float scaling_factor ) {
    scalePrimitiveData( getAllUUIDs(), label, scaling_factor );
}

void Context::incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, int increment ){

    for( uint UUID : UUIDs ) {

        if (!doesPrimitiveDataExist(UUID, label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID, label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_INT) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_int.at(label).at(i) += increment;
            }
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment primitive data for type int, but data '" << label << "' does not have type int." << std::endl;
        }

    }

}

void Context::incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, uint increment ){

    for( uint UUID : UUIDs ) {

        if (!doesPrimitiveDataExist(UUID,label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID,label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_UINT) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_uint.at(label).at(i) += increment;
            }
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment Primitive data for type uint, but data '" << label << "' does not have type uint." << std::endl;
        }

    }

}

void Context::incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, float increment ){

    for( uint UUID : UUIDs ) {

        if (!doesPrimitiveDataExist(UUID, label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID, label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_FLOAT) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_float.at(label).at(i) += increment;
            }
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment Primitive data for type float, but data '" << label << "' does not have type float." << std::endl;
        }

    }

}

void Context::incrementPrimitiveData( const std::vector<uint> &UUIDs, const char* label, double increment ){

    for( uint UUID : UUIDs ) {

        if (!doesPrimitiveDataExist(UUID, label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID, label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_DOUBLE) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_double.at(label).at(i) += increment;
            }
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment Primitive data for type double, but data '" << label << "' does not have type double." << std::endl;
        }

    }

}

void Context::aggregatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label  ){

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;

    float data_float = 0;
    double data_double = 0;
    uint data_uint = 0;
    int data_int = 0;
    int2 data_int2;
    int3 data_int3;
    int4 data_int4;
    vec2 data_vec2;
    vec3 data_vec3;
    vec4 data_vec4;

    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            primitives_not_exist++;
            continue;
        }

        HeliosDataType data_type;

        bool init_type = false;
        for( const auto &label : primitive_data_labels ) {

            if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                continue;
            }

            HeliosDataType data_type_current = getPrimitiveDataType(UUID, label.c_str());
            if( !init_type ) {
                data_type = data_type_current;
                init_type = true;
            }else{
                if( data_type!=data_type_current ){
                    helios_runtime_error("ERROR (Context::aggregatePrimitiveDataSum): Primitive data types are not consistent for UUID " + std::to_string(UUID));
                }
            }

            if ( data_type_current == HELIOS_TYPE_FLOAT) {
                float data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_float += data;
            } else if ( data_type_current == HELIOS_TYPE_DOUBLE) {
                double data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_double += data;
            } else if ( data_type_current == HELIOS_TYPE_VEC2) {
                vec2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec2 = data_vec2 + data;
            } else if ( data_type_current == HELIOS_TYPE_VEC3) {
                vec3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec3 = data_vec3 + data;
            } else if ( data_type_current == HELIOS_TYPE_VEC4) {
                vec4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec4 = data_vec4 + data;
            } else if ( data_type_current == HELIOS_TYPE_INT) {
                int data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int = data_int + data;
            } else if ( data_type_current == HELIOS_TYPE_UINT) {
                uint data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_uint = data_uint + data;
            } else if ( data_type_current == HELIOS_TYPE_INT2) {
                int2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int2 = data_int2 + data;
            } else if ( data_type_current == HELIOS_TYPE_INT3) {
                int3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int3 = data_int3 + data;
            } else if ( data_type_current == HELIOS_TYPE_INT4) {
                int4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int4 = data_int4 + data;
            } else {
                helios_runtime_error("ERROR (Context::aggregatePrimitiveDataSum): This operation is not supported for string primitive data types.");
            }
        }

        if( !init_type ){
            primitive_data_not_exist++;
            continue;
        }else if ( data_type == HELIOS_TYPE_FLOAT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_float );
            data_float = 0;
        } else if ( data_type == HELIOS_TYPE_DOUBLE) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_double );
            data_double = 0;
        } else if ( data_type == HELIOS_TYPE_VEC2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec2 );
            data_vec2 = make_vec2(0,0);
        } else if ( data_type == HELIOS_TYPE_VEC3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec3 );
            data_vec3 = make_vec3(0,0,0);
        } else if ( data_type == HELIOS_TYPE_VEC4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec4 );
            data_vec4 = make_vec4(0,0,0,0);
        } else if ( data_type == HELIOS_TYPE_INT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int );
            data_int = 0;
        } else if ( data_type == HELIOS_TYPE_UINT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_uint );
            data_uint = 0;
        } else if ( data_type == HELIOS_TYPE_INT2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int2 );
            data_int2 = make_int2(0,0);
        } else if ( data_type == HELIOS_TYPE_INT3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int3 );
            data_int3 = make_int3(0,0,0);
        } else if ( data_type == HELIOS_TYPE_INT4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int4 );
            data_int4 = make_int4(0,0,0,0);
        }

    }

    if( primitives_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataSum): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if( primitive_data_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataSum): Primitive data did not exist for " << primitive_data_not_exist << " primitives, and thus no scaling summation was performed and new primitive data was not created for this primitive." << std::endl;
    }

}

void Context::aggregatePrimitiveDataProduct( const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label  ){

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;

    float data_float = 0;
    double data_double = 0;
    uint data_uint = 0;
    int data_int = 0;
    int2 data_int2;
    int3 data_int3;
    int4 data_int4;
    vec2 data_vec2;
    vec3 data_vec3;
    vec4 data_vec4;

    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            primitives_not_exist++;
            continue;
        }

        HeliosDataType data_type;

        bool init_type = false;
        int i=0;
        for( const auto &label : primitive_data_labels ) {

            if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                continue;
            }

            HeliosDataType data_type_current = getPrimitiveDataType(UUID, label.c_str());
            if( !init_type ) {
                data_type = data_type_current;
                init_type = true;
            }else{
                if( data_type!=data_type_current ){
                    helios_runtime_error("ERROR (Context::aggregatePrimitiveDataProduct): Primitive data types are not consistent for UUID " + std::to_string(UUID));
                }
            }

            if ( data_type_current == HELIOS_TYPE_FLOAT) {
                float data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_float = data;
                }else {
                    data_float *= data;
                }
            } else if ( data_type_current == HELIOS_TYPE_DOUBLE) {
                double data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ) {
                    data_double *= data;
                }else{
                    data_double = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_VEC2) {
                vec2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_vec2.x *= data.x;
                    data_vec2.y *= data.y;
                }else{
                    data_vec2 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_VEC3) {
                vec3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_vec3.x *= data.x;
                    data_vec3.y *= data.y;
                    data_vec3.z *= data.z;
                }else{
                    data_vec3 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_VEC4) {
                vec4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_vec4.x *= data.x;
                    data_vec4.y *= data.y;
                    data_vec4.z *= data.z;
                    data_vec4.w *= data.w;
                }else{
                    data_vec4 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT) {
                int data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int = data_int * data;
                }else{
                    data_int = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_UINT) {
                uint data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_uint = data_uint * data;
                }else{
                    data_uint = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT2) {
                int2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int2.x *= data.x;
                    data_int2.y *= data.y;
                }else{
                    data_int2 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT3) {
                int3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int3.x *= data.x;
                    data_int3.y *= data.y;
                    data_int3.z *= data.z;
                }else{
                    data_int3 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT4) {
                int4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int4.x *= data.x;
                    data_int4.y *= data.y;
                    data_int4.z *= data.z;
                    data_int4.w *= data.w;
                }else{
                    data_int4 = data;
                }
            } else {
                helios_runtime_error("ERROR (Context::aggregatePrimitiveDataProduct): This operation is not supported for string primitive data types.");
            }
            i++;
        }

        if( !init_type ){
            primitive_data_not_exist++;
            continue;
        }else if ( data_type == HELIOS_TYPE_FLOAT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_float );
        } else if ( data_type == HELIOS_TYPE_DOUBLE) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_double );
        } else if ( data_type == HELIOS_TYPE_VEC2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec2 );
        } else if ( data_type == HELIOS_TYPE_VEC3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec3 );
        } else if ( data_type == HELIOS_TYPE_VEC4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec4 );
        } else if ( data_type == HELIOS_TYPE_INT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int );
        } else if ( data_type == HELIOS_TYPE_UINT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_uint );
        } else if ( data_type == HELIOS_TYPE_INT2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int2 );
        } else if ( data_type == HELIOS_TYPE_INT3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int3 );
        } else if ( data_type == HELIOS_TYPE_INT4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int4 );
        }

    }

    if( primitives_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataProduct): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if( primitive_data_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataProduct): Primitive data did not exist for " << primitive_data_not_exist << " primitives, and thus no multiplication was performed and new primitive data was not created for this primitive." << std::endl;
    }

}


float Context::sumPrimitiveSurfaceArea( const std::vector<uint> &UUIDs ) const{

    bool primitive_warning = false;
    float area = 0;
    for( uint UUID : UUIDs ){

        if( doesPrimitiveExist(UUID) ){
            area += getPrimitiveArea(UUID);
        }else{
            primitive_warning = true;
        }

    }

    if( primitive_warning ){
        std::cout << "WARNING (Context::sumPrimitiveSurfaceArea): One or more primitives reference in the UUID vector did not exist.";
    }

    return area;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, float filter_value, const std::string &comparator ){

    if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for( int p=UUIDs.size()-1; p>=0; p-- ){
        uint UUID = UUIDs_out.at(p);
        if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_FLOAT ){
            float data;
            getPrimitiveData(UUID,primitive_data_label.c_str(),data);
            if( comparator=="==" && data==filter_value ){
                continue;
            }else if ( comparator==">" && data>filter_value ) {
                continue;
            }else if ( comparator=="<" && data<filter_value ){
                continue;
            }else if ( comparator==">=" && data>=filter_value ){
                continue;
            }else if ( comparator=="<=" && data<=filter_value ){
                continue;
            }

            std::swap( UUIDs_out.at(p),UUIDs_out.back() );
            UUIDs_out.pop_back();
        }else{
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, double filter_value, const std::string &comparator ){

    if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for( int p=UUIDs.size()-1; p>=0; p-- ){
        uint UUID = UUIDs_out.at(p);
        if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_DOUBLE ){
            double data;
            getPrimitiveData(UUID,primitive_data_label.c_str(),data);
            if( comparator=="==" && data==filter_value ){
                continue;
            }else if ( comparator==">" && data>filter_value ) {
                continue;
            }else if ( comparator=="<" && data<filter_value ){
                continue;
            }else if ( comparator==">=" && data>=filter_value ){
                continue;
            }else if ( comparator=="<=" && data<=filter_value ){
                continue;
            }

            std::swap( UUIDs_out.at(p),UUIDs_out.back() );
            UUIDs_out.pop_back();
        }else{
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, int filter_value, const std::string &comparator ){

    if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for( int p=UUIDs.size()-1; p>=0; p-- ){
        uint UUID = UUIDs_out.at(p);
        if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_INT ){
            int data;
            getPrimitiveData(UUID,primitive_data_label.c_str(),data);
            if( comparator=="==" && data==filter_value ){
                continue;
            }else if ( comparator==">" && data>filter_value ) {
                continue;
            }else if ( comparator=="<" && data<filter_value ){
                continue;
            }else if ( comparator==">=" && data>=filter_value ){
                continue;
            }else if ( comparator=="<=" && data<=filter_value ){
                continue;
            }

            std::swap( UUIDs_out.at(p),UUIDs_out.back() );
            UUIDs_out.pop_back();
        }else{
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, uint filter_value, const std::string &comparator ){

    if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for( int p=UUIDs.size()-1; p>=0; p-- ){
        uint UUID = UUIDs_out.at(p);
        if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_UINT ){
            uint data;
            getPrimitiveData(UUID,primitive_data_label.c_str(),data);
            if( comparator=="==" && data==filter_value ){
                continue;
            }else if ( comparator==">" && data>filter_value ) {
                continue;
            }else if ( comparator=="<" && data<filter_value ){
                continue;
            }else if ( comparator==">=" && data>=filter_value ){
                continue;
            }else if ( comparator=="<=" && data<=filter_value ){
                continue;
            }

            std::swap( UUIDs_out.at(p),UUIDs_out.back() );
            UUIDs_out.pop_back();
        }else{
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, const std::string &filter_value ){

    std::vector<uint> UUIDs_out = UUIDs;
    for( int p=UUIDs.size()-1; p>=0; p-- ){
        uint UUID = UUIDs_out.at(p);
        if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_STRING ){
            std::string data;
            getPrimitiveData(UUID,primitive_data_label.c_str(),data);
            if( data!=filter_value ) {
                std::swap(UUIDs_out.at(p), UUIDs_out.back());
                UUIDs_out.pop_back();
            }
        }else{
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;

}

//------ Object Data ------- //

void Context::setObjectData( const uint objID, const char* label, const int& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const uint& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const float& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const double& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::vec2& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::vec3& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::vec4& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::int2& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::int3& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::int4& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const std::string& data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, HeliosDataType type, uint size, void* data ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->setObjectData(label,type,size,data);
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const int& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const uint& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const float& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const double& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec2& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec3& data ){
  for(unsigned int objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec4& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int2& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int3& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int4& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const std::string& data ){
  for( uint objID : objIDs){
    setObjectData( objID, label, data );
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const int& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const uint& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const float& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const double& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec2& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec3& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec4& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int2& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int3& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int4& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const std::string& data ){
  for( const auto & j : objIDs){
    for( const auto & objID : j ){
      setObjectData( objID, label, data );
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const int& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const uint& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const float& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const double& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec2& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec3& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec4& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int2& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int3& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int4& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const std::string& data ){
  for( const auto & j : objIDs){
    for( const auto & i : j ){
      for( const auto & objID : i ) {
        setObjectData(objID, label, data);
      }
    }
  }
}

void Context::getObjectData( const uint objID, const char* label, int& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, uint& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<uint>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, float& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<float>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, double& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<double>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, vec2& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<vec2>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, vec3& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<vec3>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, vec4& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<vec4>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, int2& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int2>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, int3& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int3>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, int4& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int4>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::string& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<std::string>& data ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->getObjectData(label,data);
}

HeliosDataType Context::getObjectDataType( const uint objID, const char* label )const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (getObjectDataType): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  return objects.at(objID)->getObjectDataType(label);
}

uint Context::getObjectDataSize( const uint objID, const char* label )const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (getObjectDataSize): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  return objects.at(objID)->getObjectDataSize(label);
}

bool Context::doesObjectDataExist( const uint objID, const char* label ) const{
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (doesObjectDataExist): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  return objects.at(objID)->doesObjectDataExist(label);
}

void Context::copyObjectData( uint objID, uint oldObjID){

    //copy the object data
    std::vector<std::string> plabel = getObjectPointer_private(objID)->listObjectData();
    for(auto & p : plabel){

        HeliosDataType type = getObjectDataType( objID, p.c_str() );

        if( type==HELIOS_TYPE_INT ){
            std::vector<int> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_INT, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_UINT ){
            std::vector<uint> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_UINT, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_FLOAT ){
            std::vector<float> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_FLOAT, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_DOUBLE ){
            std::vector<double> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_DOUBLE, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_VEC2 ){
            std::vector<vec2> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_VEC2, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_VEC3 ){
            std::vector<vec3> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_VEC3, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_VEC4 ){
            std::vector<vec4> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_VEC4, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_INT2 ){
            std::vector<int2> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_INT2, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_INT3 ){
            std::vector<int3> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_INT3, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_INT4 ){
            std::vector<int4> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_INT4, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_STRING ){
            std::vector<std::string> pdata;
            getObjectData( objID, p.c_str(), pdata );
            setObjectData( oldObjID, p.c_str(), HELIOS_TYPE_STRING, pdata.size(), &pdata.at(0) );
        }else{
            assert(false);
        }

    }
}

void Context::duplicateObjectData( uint objID, const char* old_label, const char* new_label ){

    if( objects.find(objID) == objects.end() ){
        helios_runtime_error("ERROR (Context::duplicateObjectData): Object ID of " + std::to_string(objID) + " does not exist in the Context.");
    }else if( !doesObjectDataExist(objID, old_label) ){
        helios_runtime_error("ERROR (Context::duplicateObjectData): Object ID of " + std::to_string(objID) + " does not have data with label " + std::string(old_label) + ".");
    }

    HeliosDataType type = getObjectDataType( objID, old_label );

    if( type==HELIOS_TYPE_INT ){
        std::vector<int> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_INT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_UINT ){
        std::vector<uint> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_UINT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_FLOAT ){
        std::vector<float> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_FLOAT, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_DOUBLE ){
        std::vector<double> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_DOUBLE, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_VEC2, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_VEC3, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_VEC4, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT2 ){
        std::vector<int2> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_INT2, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT3 ){
        std::vector<int3> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_INT3, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_INT4 ){
        std::vector<int4> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_INT4, pdata.size(), &pdata.at(0) );
    }else if( type==HELIOS_TYPE_STRING ){
        std::vector<std::string> pdata;
        getObjectData( objID, old_label, pdata );
        setObjectData( objID, new_label, HELIOS_TYPE_STRING, pdata.size(), &pdata.at(0) );
    }else{
        assert(false);
    }

}


void Context::renameObjectData( uint objID, const char* old_label, const char* new_label ){

    if( objects.find(objID) == objects.end() ){
        helios_runtime_error("ERROR (Context::renameObjectData): Object ID of " + std::to_string(objID) + " does not exist in the Context.");
    }else if( !doesObjectDataExist(objID, old_label) ){
        helios_runtime_error("ERROR (Context::renameObjectData): Object ID of " + std::to_string(objID) + " does not have data with label " + std::string(old_label) + ".");
    }

    duplicateObjectData( objID, old_label, new_label );
    clearObjectData( objID, old_label );

}

void Context::clearObjectData( const uint objID, const char* label ){
  if( objects.find(objID) == objects.end() ){
    helios_runtime_error("ERROR (Context::clearObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
  }
  objects.at(objID)->clearObjectData(label);
}

void Context::clearObjectData( const std::vector<uint>& objIDs, const char* label ){
  for( uint objID : objIDs ){
    if( objects.find(objID) == objects.end() ){
      helios_runtime_error("ERROR (Context::clearObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
    }
    objects.at(objID)->clearObjectData(label);
  }
}

std::vector<std::string> Context::listObjectData(uint ObjID) const{
  return getObjectPointer_private(ObjID)->listObjectData();
}

void CompoundObject::setObjectData( const char* label, const int& data ){
  std::vector<int> vec{data};
  object_data_int[label] = vec;
  object_data_types[label] = HELIOS_TYPE_INT;
}

void CompoundObject::setObjectData( const char* label, const uint& data ){
  std::vector<uint> vec{data};
  object_data_uint[label] = vec;
  object_data_types[label] = HELIOS_TYPE_UINT;
}

void CompoundObject::setObjectData( const char* label, const float& data ){
  std::vector<float> vec{data};
  object_data_float[label] = vec;
  object_data_types[label] = HELIOS_TYPE_FLOAT;
}

void CompoundObject::setObjectData( const char* label, const double& data ){
  std::vector<double> vec{data};
  object_data_double[label] = vec;
  object_data_types[label] = HELIOS_TYPE_DOUBLE;
}

void CompoundObject::setObjectData( const char* label, const helios::vec2& data ){
  std::vector<vec2> vec{data};
  object_data_vec2[label] = vec;
  object_data_types[label] = HELIOS_TYPE_VEC2;
}

void CompoundObject::setObjectData( const char* label, const helios::vec3& data ){
  std::vector<vec3> vec{data};
  object_data_vec3[label] = vec;
  object_data_types[label] = HELIOS_TYPE_VEC3;
}

void CompoundObject::setObjectData( const char* label, const helios::vec4& data ){
  std::vector<vec4> vec{data};
  object_data_vec4[label] = vec;
  object_data_types[label] = HELIOS_TYPE_VEC4;
}

void CompoundObject::setObjectData( const char* label, const helios::int2& data ){
  std::vector<int2> vec{data};
  object_data_int2[label] = vec;
  object_data_types[label] = HELIOS_TYPE_INT2;
}

void CompoundObject::setObjectData( const char* label, const helios::int3& data ){
  std::vector<int3> vec{data};
  object_data_int3[label] = vec;
  object_data_types[label] = HELIOS_TYPE_INT3;
}

void CompoundObject::setObjectData( const char* label, const helios::int4& data ){
  std::vector<int4> vec{data};
  object_data_int4[label] = vec;
  object_data_types[label] = HELIOS_TYPE_INT4;
}

void CompoundObject::setObjectData( const char* label, const std::string& data ){
  std::vector<std::string> vec{data};
  object_data_string[label] = vec;
  object_data_types[label] = HELIOS_TYPE_STRING;
}

void CompoundObject::setObjectData( const char* label, HeliosDataType a_type, uint size, void* data ){

  object_data_types[label] = a_type;

  if( a_type==HELIOS_TYPE_INT ){

    int* data_ptr = (int*)data;

    std::vector<int> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_int[label] = vec;

  }else if( a_type==HELIOS_TYPE_UINT ){

    uint* data_ptr = (uint*)data;

    std::vector<uint> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_uint[label] = vec;

  }else if( a_type==HELIOS_TYPE_FLOAT ){

    auto* data_ptr = (float*)data;

    std::vector<float> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_float[label] = vec;

  }else if( a_type==HELIOS_TYPE_DOUBLE ){

    auto* data_ptr = (double*)data;

    std::vector<double> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_double[label] = vec;

  }else if( a_type==HELIOS_TYPE_VEC2 ){

    auto* data_ptr = (vec2*)data;

    std::vector<vec2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_vec2[label] = vec;

  }else if( a_type==HELIOS_TYPE_VEC3 ){

    auto* data_ptr = (vec3*)data;

    std::vector<vec3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_vec3[label] = vec;

  }else if( a_type==HELIOS_TYPE_VEC4 ){

    auto* data_ptr = (vec4*)data;

    std::vector<vec4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_vec4[label] = vec;

  }else if( a_type==HELIOS_TYPE_INT2 ){

    auto* data_ptr = (int2*)data;

    std::vector<int2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_int2[label] = vec;

  }else if( a_type==HELIOS_TYPE_INT3 ){

    auto* data_ptr = (int3*)data;

    std::vector<int3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_int3[label] = vec;

  }else if( a_type==HELIOS_TYPE_INT4 ){

    auto* data_ptr = (int4*)data;

    std::vector<int4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_int4[label] = vec;

  }else if( a_type==HELIOS_TYPE_STRING ){

    auto* data_ptr = (std::string*)data;

    std::vector<std::string> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    object_data_string[label] = vec;

  }

}

void CompoundObject::getObjectData( const char* label, int& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT ){
    std::vector<int> d = object_data_int.at(label);
    data = d.at(0);
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<int>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT ){
    std::vector<int> d = object_data_int.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int.");
  }

}

void CompoundObject::getObjectData( const char* label, uint& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_UINT ){
    std::vector<uint> d = object_data_uint.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type uint, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type uint.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<uint>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_UINT ){
    std::vector<uint> d = object_data_uint.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type uint, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type uint.");
  }

}

void CompoundObject::getObjectData( const char* label, float& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = object_data_float.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type float, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type float.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<float>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = object_data_float.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type float, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type float.");
  }

}

void CompoundObject::getObjectData( const char* label, double& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = object_data_double.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type double, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type double.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<double>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = object_data_double.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type double, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type double.");
  }

}

void CompoundObject::getObjectData( const char* label, vec2& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = object_data_vec2.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type vec2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec2.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<vec2>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = object_data_vec2.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type vec2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec2.");
  }

}

void CompoundObject::getObjectData( const char* label, vec3& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = object_data_vec3.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type vec3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec3.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<vec3>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = object_data_vec3.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type vec3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec3.");
  }

}

void CompoundObject::getObjectData( const char* label, vec4& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = object_data_vec4.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type vec4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec4.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<vec4>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = object_data_vec4.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type vec4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec4.");
  }

}

void CompoundObject::getObjectData( const char* label, int2& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = object_data_int2.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int2.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<int2>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = object_data_int2.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int2.");
  }

}

void CompoundObject::getObjectData( const char* label, int3& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = object_data_int3.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int3.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<int3>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID) );
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = object_data_int3.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int3.");
  }

}

void CompoundObject::getObjectData( const char* label, int4& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = object_data_int4.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int4.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<int4>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = object_data_int4.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type int4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int4.");
  }

}

void CompoundObject::getObjectData( const char* label, std::string& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = object_data_string.at(label);
    data = d.front();
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type string, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type string.");
  }

}

void CompoundObject::getObjectData( const char* label, std::vector<std::string>& data ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  if( object_data_types.at(label)==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = object_data_string.at(label);
    data = d;
  }else{
    helios_runtime_error("ERROR (CompoundObject::getObjectData): Attempted to get data for type string, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type string.");
  }

}

HeliosDataType CompoundObject::getObjectDataType( const char* label ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectDataType): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  return object_data_types.at(label);

}

uint CompoundObject::getObjectDataSize( const char* label ) const{

  if( !doesObjectDataExist( label ) ){
    helios_runtime_error("ERROR (CompoundObject::getObjectDataSize): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
  }

  HeliosDataType qtype = object_data_types.at(label);

  if( qtype==HELIOS_TYPE_INT ){
    return object_data_int.at(label).size();
  }else if( qtype==HELIOS_TYPE_UINT ){
    return object_data_uint.at(label).size();
  }else if( qtype==HELIOS_TYPE_FLOAT ){
    return object_data_float.at(label).size();
  }else if( qtype==HELIOS_TYPE_DOUBLE ){
    return object_data_double.at(label).size();
  }else if( qtype==HELIOS_TYPE_VEC2 ){
    return object_data_vec2.at(label).size();
  }else if( qtype==HELIOS_TYPE_VEC3 ){
    return object_data_vec3.at(label).size();
  }else if( qtype==HELIOS_TYPE_VEC4 ){
    return object_data_vec4.at(label).size();
  }else if( qtype==HELIOS_TYPE_INT2 ){
    return object_data_int2.at(label).size();
  }else if( qtype==HELIOS_TYPE_INT3 ){
    return object_data_int3.at(label).size();
  }else if( qtype==HELIOS_TYPE_INT4 ){
    return object_data_int4.at(label).size();
  }else if( qtype==HELIOS_TYPE_STRING ){
    return object_data_string.at(label).size();
  }else{
    assert( false );
  }

  return 0;

}

void CompoundObject::clearObjectData( const char* label ){

  if( !doesObjectDataExist( label ) ){
    return;
  }

  HeliosDataType qtype = object_data_types.at(label);

  if( qtype==HELIOS_TYPE_INT ){
    object_data_int.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_UINT ){
    object_data_uint.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_FLOAT ){
    object_data_float.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_DOUBLE ){
    object_data_double.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_VEC2 ){
    object_data_vec2.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_VEC3 ){
    object_data_vec3.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_VEC4 ){
    object_data_vec4.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_INT2 ){
    object_data_int2.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_INT3 ){
    object_data_int3.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_INT4 ){
    object_data_int4.erase(label);
    object_data_types.erase(label);
  }else if( qtype==HELIOS_TYPE_STRING ){
    object_data_string.erase(label);
    object_data_types.erase(label);
  }else{
    assert(false);
  }

}

bool CompoundObject::doesObjectDataExist( const char* label ) const{

  if( object_data_types.find(label) == object_data_types.end() ){
    return false;
  }else{
    return true;
  }

}

std::vector<std::string> CompoundObject::listObjectData() const{

  std::vector<std::string> labels(object_data_types.size());

  size_t i=0;
  for(const auto & object_data_type : object_data_types){
    labels.at(i) = object_data_type.first;
    i++;
  }

  return labels;

}

// -------- Global Data ---------- //

void Context::setGlobalData( const char* label, const int& data ){
  std::vector<int> vec{data};
  globaldata[label].type = HELIOS_TYPE_INT;
  globaldata[label].size = 1;
  globaldata[label].global_data_int = vec;
}

void Context::setGlobalData( const char* label, const uint& data ){
  std::vector<uint> vec{data};
  globaldata[label].type = HELIOS_TYPE_UINT;
  globaldata[label].size = 1;
  globaldata[label].global_data_uint = vec;
}

void Context::setGlobalData( const char* label, const float& data ){
  std::vector<float> vec{data};
  globaldata[label].type = HELIOS_TYPE_FLOAT;
  globaldata[label].size = 1;
  globaldata[label].global_data_float = vec;
}

void Context::setGlobalData( const char* label, const double& data ){
  std::vector<double> vec{data};
  globaldata[label].type = HELIOS_TYPE_DOUBLE;
  globaldata[label].size = 1;
  globaldata[label].global_data_double = vec;
}

void Context::setGlobalData( const char* label, const helios::vec2& data ){
  std::vector<vec2> vec{data};
  globaldata[label].type = HELIOS_TYPE_VEC2;
  globaldata[label].size = 1;
  globaldata[label].global_data_vec2 = vec;
}

void Context::setGlobalData( const char* label, const helios::vec3& data ){
  std::vector<vec3> vec{data};
  globaldata[label].type = HELIOS_TYPE_VEC3;
  globaldata[label].size = 1;
  globaldata[label].global_data_vec3 = vec;
}

void Context::setGlobalData( const char* label, const helios::vec4& data ){
  std::vector<vec4> vec{data};
  globaldata[label].type = HELIOS_TYPE_VEC4;
  globaldata[label].size = 1;
  globaldata[label].global_data_vec4 = vec;
}

void Context::setGlobalData( const char* label, const helios::int2& data ){
  std::vector<int2> vec{data};
  globaldata[label].type = HELIOS_TYPE_INT2;
  globaldata[label].size = 1;
  globaldata[label].global_data_int2 = vec;
}

void Context::setGlobalData( const char* label, const helios::int3& data ){
  std::vector<int3> vec{data};
  globaldata[label].type = HELIOS_TYPE_INT3;
  globaldata[label].size = 1;
  globaldata[label].global_data_int3 = vec;
}

void Context::setGlobalData( const char* label, const helios::int4& data ){
  std::vector<int4> vec{data};
  globaldata[label].type = HELIOS_TYPE_INT4;
  globaldata[label].size = 1;
  globaldata[label].global_data_int4 = vec;
}

void Context::setGlobalData( const char* label, const std::string& data ){
  std::vector<std::string> vec{data};
  globaldata[label].type = HELIOS_TYPE_STRING;
  globaldata[label].size = 1;
  globaldata[label].global_data_string = vec;
}

void Context::setGlobalData( const char* label, HeliosDataType type, size_t size, void* data ){

  globaldata[label].type = type;
  globaldata[label].size = size;

  if( type==HELIOS_TYPE_INT ){

    auto* data_ptr = (int*)data;

    std::vector<int> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int = vec;

  }else if( type==HELIOS_TYPE_UINT ){

    auto* data_ptr = (uint*)data;

    std::vector<uint> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_uint = vec;

  }else if( type==HELIOS_TYPE_FLOAT ){

    auto* data_ptr = (float*)data;

    std::vector<float> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_float = vec;

  }else if( type==HELIOS_TYPE_DOUBLE ){

    auto* data_ptr = (double*)data;

    std::vector<double> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_double = vec;

  }else if( type==HELIOS_TYPE_VEC2 ){

    auto* data_ptr = (vec2*)data;

    std::vector<vec2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_vec2 = vec;

  }else if( type==HELIOS_TYPE_VEC3 ){

    auto* data_ptr = (vec3*)data;

    std::vector<vec3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_vec3= vec;

  }else if( type==HELIOS_TYPE_VEC4 ){

    auto* data_ptr = (vec4*)data;

    std::vector<vec4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_vec4 = vec;

  }else if( type==HELIOS_TYPE_INT2 ){

    auto* data_ptr = (int2*)data;

    std::vector<int2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int2 = vec;

  }else if( type==HELIOS_TYPE_INT3 ){

    auto* data_ptr = (int3*)data;

    std::vector<int3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int3 = vec;

  }else if( type==HELIOS_TYPE_INT4 ){

    auto* data_ptr = (int4*)data;

    std::vector<int4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int4 = vec;

  }else if( type==HELIOS_TYPE_STRING ){

    auto* data_ptr = (std::string*)data;

    std::vector<std::string> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_string = vec;

  }

}

void Context::renameGlobalData( const char* old_label, const char* new_label ){

    if( !doesGlobalDataExist( old_label ) ){
        helios_runtime_error("ERROR (Context::duplicateGlobalData): Global data " + std::string(old_label) + " does not exist in the Context.");
    }

    duplicateGlobalData( old_label, new_label );
    clearGlobalData( old_label );

}

void Context::duplicateGlobalData(const char* old_label, const char* new_label ){

    if( !doesGlobalDataExist( old_label ) ){
        helios_runtime_error("ERROR (Context::duplicateGlobalData): Global data " + std::string(old_label) + " does not exist in the Context.");
    }

    HeliosDataType type = getGlobalDataType( old_label );

    if( type==HELIOS_TYPE_INT ){
        std::vector<int> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_INT, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_UINT ){
        std::vector<uint> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_UINT, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_FLOAT ){
        std::vector<float> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_FLOAT, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_DOUBLE ){
        std::vector<double> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_DOUBLE, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_VEC2, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_VEC3, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_VEC4, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_INT2 ){
        std::vector<int2> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_INT2, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_INT3 ){
        std::vector<int3> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_INT3, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_INT4 ){
        std::vector<int4> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_INT4, gdata.size(), &gdata.at(0) );
    }else if( type==HELIOS_TYPE_STRING ){
        std::vector<std::string> gdata;
        getGlobalData( old_label, gdata );
        setGlobalData( new_label, HELIOS_TYPE_STRING, gdata.size(), &gdata.at(0) );
    }else{
        assert(false);
    }

}

void Context::clearGlobalData( const char* label ){

    if(doesGlobalDataExist(label)){
        globaldata.erase(label);
    }

}

void Context::getGlobalData( const char* label, int& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT ){
    std::vector<int> d = gdata.global_data_int;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int, but data '" + std::string(label) + "' does not have type int.");
  }

}

void Context::getGlobalData( const char* label, std::vector<int>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT ){
    std::vector<int> d = gdata.global_data_int;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int, but data '" + std::string(label) + "' does not have type int.");
  }

}

void Context::getGlobalData( const char* label, uint& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = gdata.global_data_uint;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type uint, but data '" + std::string(label) + "' does not have type uint.");
  }

}

void Context::getGlobalData( const char* label, std::vector<uint>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = gdata.global_data_uint;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type uint, but data '" + std::string(label) + "' does not have type uint.");
  }

}

void Context::getGlobalData( const char* label, float& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = gdata.global_data_float;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type float, but data '" + std::string(label) + "' does not have type float.");
  }

}

void Context::getGlobalData( const char* label, std::vector<float>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = gdata.global_data_float;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type float, but data '" + std::string(label) + "' does not have type float.");
  }

}

void Context::getGlobalData( const char* label, double& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = gdata.global_data_double;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type double, but data '" + std::string(label) + "' does not have type double.");
  }

}

void Context::getGlobalData( const char* label, std::vector<double>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = gdata.global_data_double;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type double, but data '" + std::string(label) + "' does not have type double.");
  }

}

void Context::getGlobalData( const char* label, helios::vec2& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = gdata.global_data_vec2;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type vec2, but data '" + std::string(label) + "' does not have type vec2.");
  }

}

void Context::getGlobalData( const char* label, std::vector<helios::vec2>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = gdata.global_data_vec2;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type vec2, but data '" + std::string(label) + "' does not have type vec2.");
  }

}

void Context::getGlobalData( const char* label, helios::vec3& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = gdata.global_data_vec3;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type vec3, but data '" + std::string(label) + "' does not have type vec3.");
  }

}

void Context::getGlobalData( const char* label, std::vector<helios::vec3>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = gdata.global_data_vec3;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type vec3, but data '" + std::string(label) + "' does not have type vec3.");
  }

}

void Context::getGlobalData( const char* label, helios::vec4& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = gdata.global_data_vec4;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type vec4, but data '" + std::string(label) + "' does not have type vec4.");
  }

}

void Context::getGlobalData( const char* label, std::vector<helios::vec4>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = gdata.global_data_vec4;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type vec4, but data '" + std::string(label) + "' does not have type vec4.");
  }

}

void Context::getGlobalData( const char* label, helios::int2& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = gdata.global_data_int2;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int2, but data '" + std::string(label) + "' does not have type int2.");
  }

}

void Context::getGlobalData( const char* label, std::vector<helios::int2>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = gdata.global_data_int2;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int2, but data '" + std::string(label) + "' does not have type int2.");
  }

}

void Context::getGlobalData( const char* label, helios::int3& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = gdata.global_data_int3;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int3, but data '" + std::string(label) + "' does not have type int3.");
  }

}

void Context::getGlobalData( const char* label, std::vector<helios::int3>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = gdata.global_data_int3;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int3, but data '" + std::string(label) + "' does not have type int3.");
  }

}

void Context::getGlobalData( const char* label, helios::int4& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = gdata.global_data_int4;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int4, but data '" + std::string(label) + "' does not have type int4.");
  }

}

void Context::getGlobalData( const char* label, std::vector<helios::int4>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = gdata.global_data_int4;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type int4, but data '" + std::string(label) + "' does not have type int4.");
  }

}

void Context::getGlobalData( const char* label, std::string& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = gdata.global_data_string;
    data = d.front();
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type string, but data '" + std::string(label) + "' does not have type string.");
  }

}

void Context::getGlobalData( const char* label, std::vector<std::string>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = gdata.global_data_string;
    data = d;
  }else{
    helios_runtime_error("ERROR (Context::getGlobalData): Attempted to get global data for type string, but data '" + std::string(label) + "' does not have type string.");
  }

}

HeliosDataType Context::getGlobalDataType( const char* label ) const{

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalDataType): Global data " + std::string(label) + " does not exist in the Context.");
  }

  return globaldata.at(label).type;

}

size_t Context::getGlobalDataSize(const char *label) const {

  if( !doesGlobalDataExist( label ) ){
    helios_runtime_error("ERROR (Context::getGlobalDataSize): Global data " + std::string(label) + " does not exist in the Context.");
  }

  return globaldata.at(label).size;

}

bool Context::doesGlobalDataExist( const char* label ) const{

  if( globaldata.find(label) == globaldata.end() ){
    return false;
  }else{
    return true;
  }

}

void Context::incrementGlobalData( const char* label, int increment ){

    if( !doesGlobalDataExist( label ) ){
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if( globaldata.at(label).type==HELIOS_TYPE_INT ) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_int.at(i) += increment;
        }
    }else{
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type int, but data '" << label << "' does not have type int." << std::endl;
    }

}

void Context::incrementGlobalData( const char* label, uint increment ){

    if( !doesGlobalDataExist( label ) ){
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if( globaldata.at(label).type==HELIOS_TYPE_UINT ) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_uint.at(i) += increment;
        }
    }else{
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type uint, but data '" << label << "' does not have type uint." << std::endl;
    }

}

void Context::incrementGlobalData( const char* label, float increment ){

    if( !doesGlobalDataExist( label ) ){
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if( globaldata.at(label).type==HELIOS_TYPE_FLOAT ) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_float.at(i) += increment;
        }
    }else{
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type float, but data '" << label << "' does not have type float." << std::endl;
    }

}

void Context::incrementGlobalData( const char* label, double increment ){

    if( !doesGlobalDataExist( label ) ){
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if( globaldata.at(label).type==HELIOS_TYPE_DOUBLE ) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_double.at(i) += increment;
        }
    }else{
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type double, but data '" << label << "' does not have type double." << std::endl;
    }

}