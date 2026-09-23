#version 330 core

/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

// Ouput data
layout(location = 0) out float fragmentdepth;

in vec2 texcoord;

uniform isamplerBuffer texture_flag_texture_object;
uniform isamplerBuffer texture_ID_texture_object;
uniform isamplerBuffer coordinate_flag_texture_object;
uniform isamplerBuffer sky_geometry_flag_texture_object;
uniform isamplerBuffer hidden_flag_texture_object;

uniform sampler2DArray textureSampler;

out vec3 color;

flat in int faceID;

void main(){

  // Delete hidden/deleted primitives
  if( texelFetch(hidden_flag_texture_object, faceID).r == 0 ){
    discard;
  }

  int textureFlag = texelFetch(texture_flag_texture_object, faceID).r;
  int textureID = texelFetch(texture_ID_texture_object, faceID).r;
  int coordinateFlag = texelFetch(coordinate_flag_texture_object, faceID).r;
  int skyGeometryFlag = texelFetch(sky_geometry_flag_texture_object, faceID).r;

  vec3 texcoord3 = vec3(texcoord, textureID);

  // Exclude sky geometry from shadow rendering
  if( skyGeometryFlag == 1 ){
    discard;
  }

  if( coordinateFlag==0 || coordinateFlag==2 ){
    // If this is a 2D projected primitive, don't consider for shadows
    discard;
  }else if( ( textureFlag==1 || textureFlag==2 ) && texture(textureSampler, texcoord3).a<0.05f ){
    // Ignore transparent texture pixels (alpha channel is .a channel)
    discard;
  }else if( textureFlag==3 && texture(textureSampler, texcoord3).r<0.9f ){
    // Ignore transparent texture pixels (alpha channel is .r channel)
    discard;
  }
}
