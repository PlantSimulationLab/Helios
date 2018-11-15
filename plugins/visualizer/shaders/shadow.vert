#version 330 core
 
// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;

layout(location = 1) in vec4 vertexColor; //NOTE: these are not used, but need to be consistent with primaryShader.vert
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec2 uv;

layout(location = 4) in int vertexTextureFlag;

layout(location = 5) in int vertexCoordinateFlag;

flat out int textureFlag;

flat out int coordinateFlag;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

out vec2 texcoord;
 
void main(){
  texcoord = uv;
  gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
  textureFlag = vertexTextureFlag;
  coordinateFlag = vertexCoordinateFlag;
}
