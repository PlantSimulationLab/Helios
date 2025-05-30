#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;

layout(location = 1) in vec2 uv;

layout(location = 2) in int face_index;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

uniform isamplerBuffer texture_ID_texture_object;
uniform samplerBuffer uv_rescale;

// Output data
out vec2 texcoord;

flat out int faceID;
 
void main(){
  faceID = face_index;
  int textureID = texelFetch(texture_ID_texture_object, faceID).r;
  vec4 scale = texelFetch(uv_rescale, textureID);
  texcoord = vec2(uv.x*scale.r, uv.y*scale.g);
  gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
}
