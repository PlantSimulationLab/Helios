#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;

layout(location = 1) in vec4 vertexColor;

layout(location = 2) in vec3 vertexNormal;

layout(location = 3) in vec2 uv;

layout(location = 4) in int vertexTextureFlag;

layout(location = 5) in int vertexCoordinateFlag;

uniform mat4 MVP;

uniform mat4 DepthBiasMVP;
out vec4 ShadowCoord;

// Output data ; will be interpolated for each fragment.
out vec4 fragmentColor;

out vec2 texcoord;

out vec3 normal;

flat out int textureFlag;

flat out int coordinateFlag;

void main(){

  fragmentColor = vertexColor;

  vec4 v = vec4(vertexPosition_modelspace,1); // Transform a homogeneous 4D vector
  if( vertexCoordinateFlag==0 ){
    gl_Position = v;
  }else{
    gl_Position = MVP * v;
  }

  texcoord = uv;

  normal = vertexNormal;

  textureFlag = vertexTextureFlag;

  coordinateFlag = vertexCoordinateFlag;

  ShadowCoord = DepthBiasMVP*v;

}
