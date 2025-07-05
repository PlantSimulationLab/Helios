#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;

layout(location = 1) in vec2 uv;

layout(location = 2) in int face_index;

uniform mat4 MVP;

uniform mat4 DepthBiasMVP;
out vec4 ShadowCoord;

out vec2 texcoord;
out vec2 uv_scale;

flat out int faceID;

uniform isamplerBuffer texture_ID_texture_object;
uniform isamplerBuffer coordinate_flag_texture_object;

uniform samplerBuffer uv_rescale;

void main(){

  faceID = face_index;

  int textureID = texelFetch(texture_ID_texture_object, faceID).r;
  int coordinateFlag = texelFetch(coordinate_flag_texture_object, faceID).r;

  vec4 v = vec4(vertexPosition_modelspace,1); // Transform a homogeneous 4D vector
  if( coordinateFlag==0 ){
    gl_Position = v;
  }else{
    gl_Position = MVP * v;
  }

  vec4 scale = texelFetch(uv_rescale, textureID);
  texcoord = uv;            // pass original texture coordinates
  uv_scale = scale.rg;      // scale factors for this texture layer

  ShadowCoord = DepthBiasMVP*v;

}
