#version 330 core
 
// Ouput data
layout(location = 0) out float fragmentdepth;

in vec2 texcoord;

flat in int textureFlag;
flat in int coordinateFlag;

uniform sampler2DRect textureSampler;

out vec3 color;
 
void main(){
  if( coordinateFlag==0 ){
    discard;
  }else if( ( textureFlag==1 || textureFlag==2 )&& texture(textureSampler, texcoord).a<0.05f ){
    discard;
  }else if( textureFlag==3 && texture(textureSampler, texcoord).r<0.9f ){
    discard;
  }
}
