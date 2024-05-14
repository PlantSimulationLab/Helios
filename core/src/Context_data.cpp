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