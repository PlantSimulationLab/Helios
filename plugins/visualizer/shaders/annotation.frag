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

//Annotation pass: records which primitive is visible in each pixel and how far it is from the
//camera, instead of a color. Paired with shadow.vert, which applies the camera transformation.

//Identity of the visible primitive: (geometry type index + 1, face index within that type). The
//face index alone is not unique because each geometry type numbers its faces from zero. The
//buffer is cleared to (0,0), so a zero in the first component means no primitive is visible.
layout(location = 0) out ivec2 primitiveIndex;

//Distance from the camera to the visible surface, measured along the camera viewing direction.
layout(location = 1) out float viewDepth;

in vec2 texcoord;

uniform isamplerBuffer texture_flag_texture_object;
uniform isamplerBuffer texture_ID_texture_object;
uniform isamplerBuffer coordinate_flag_texture_object;
uniform isamplerBuffer sky_geometry_flag_texture_object;
uniform isamplerBuffer hidden_flag_texture_object;

uniform sampler2DArray textureSampler;

//Index of the geometry type currently being drawn, set before each draw call
uniform int geometryType;

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

  // The sky is background, not an object
  if( skyGeometryFlag == 1 ){
    discard;
  }

  if( coordinateFlag==0 || coordinateFlag==2 ){
    // 2D overlays (colorbar, watermark, text) are not part of the scene
    discard;
  }else if( ( textureFlag==1 || textureFlag==2 ) && texture(textureSampler, texcoord3).a<0.5f ){
    // Transparent texture pixels, using the same alpha threshold as primaryShader.frag so that the
    // annotations agree with the rendered image about where a masked primitive is visible
    discard;
  }else if( textureFlag==3 && texture(textureSampler, texcoord3).r<0.5f ){
    // Glyph textures carry their coverage in the red channel
    discard;
  }

  primitiveIndex = ivec2(geometryType + 1, faceID);

  // For a perspective projection the clip-space w coordinate is the eye-space distance along the
  // viewing direction, and gl_FragCoord.w is its reciprocal.
  viewDepth = 1.0 / gl_FragCoord.w;
}
