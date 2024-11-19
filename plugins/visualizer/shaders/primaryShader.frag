#version 330 core

out vec4 color;

// Interpolated values from the vertex shaders
in vec4 fragmentColor;

in vec2 texcoord;
in vec3 normal;

uniform sampler2DRect textureSampler;
uniform sampler2D shadowMap;
in vec4 ShadowCoord;

uniform vec3 lightDirection;

uniform int lightingModel;

uniform float lightIntensity;

uniform int Rbound;

flat in int textureFlag;

flat in int coordinateFlag;
 
void main(){

  //There are several confusing flags that determine how this shader colors fragments
  // textureFlag:
  //    = 0 -> color fragments based on their RGB color value
  //    = 1 -> color fragments based on texture map color, set transparency according to texture alpha-value
  //    = 2 -> color fragments based on their RGB color value, set transparency according to texture alpha-value
  //    = 3 -> color fragments based on their RGB color value, set transparency according to texture r-value
  // lightingModel: 
  //    = 0 -> color fragments based only on their RGBA color value (no shading, etc.)
  //    = 1 -> use a simplified Phong lighting model to shade fragments
  //    = 2 -> use a simplified Phong lighting model plus the shadow map to shade/shadow fragments

  vec3 ld = lightDirection;

  float visibility = 1.0;
  if(lightingModel==2 && dot(normal,ld)>0 ){ //Simplified Phong lighting model + shadows

    vec2 poissonDisk[4] = vec2[](
				 vec2( -0.94201624, -0.39906216 ),
				 vec2( 0.94558609, -0.76890725 ),
				 vec2( -0.094184101, -0.92938870 ),
				 vec2( 0.34495938, 0.29387760 )
				 );
    
    for (int i=0;i<4;i++){
      if ( texture( shadowMap, (ShadowCoord.xy/ShadowCoord.w+poissonDisk[i]/140000.0) ).x  <  (ShadowCoord.z/ShadowCoord.w) ){
	    visibility-=0.35;
      }
    } 
  }

  //vec4 color;
  if( textureFlag==0 ){//Color by interpolating the colors at vertices
    color = fragmentColor;
  }else if( textureFlag==1 ){//Color by texture map
    color = texture(textureSampler, texcoord);
    if(color.a<0.05 ){
      discard;
    }
  }else if( textureFlag==2 ){//Color by interpolating the colors at vertices, and set the transparency according to the red channel of the texture map given by textureSampler
    color = fragmentColor;
    color.a = texture(textureSampler, texcoord).a;
    if( color.a<0.05 ){
      discard;
    }
  }else if( textureFlag==3 ){//Color by interpolating the colors at vertices, and set the transparency according to the red channel of the texture map given by textureSampler
    color = fragmentColor;
    color.a = clamp(texture(textureSampler, texcoord).r,0,1);
    if( color.a<0.05 ){
      discard;
    }
  }else if( textureFlag==4 ){
    float zNear = 0.01;
    float zFar = Rbound*2;
    float depth = texture( shadowMap, texcoord ).x;
    color = vec4( 1, 1, 1, 1 )*(1.f-(2.0 * zNear) / (zFar + zNear - depth * (zFar - zNear)));
  }

  if( lightingModel>0 && coordinateFlag==1 ){ //Simplified Phong lighting model
    color = vec4( lightIntensity*( 0.75*color.rgb + visibility*(0.75*dot(normal,ld)*color.rgb) ) , color.a );
  }

}

