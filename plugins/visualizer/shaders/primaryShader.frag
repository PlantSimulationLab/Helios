#version 330 core

out vec4 color;

in vec2 texcoord;
in vec2 uv_scale;

uniform sampler2DArray textureSampler;

uniform sampler2D shadowMap;
in vec4 ShadowCoord;

uniform vec3 lightDirection;

uniform int lightingModel;

uniform float lightIntensity;

uniform int Rbound;

uniform samplerBuffer color_texture_object;
uniform samplerBuffer normal_texture_object;
uniform isamplerBuffer texture_flag_texture_object;
uniform isamplerBuffer texture_ID_texture_object;
uniform isamplerBuffer coordinate_flag_texture_object;
uniform isamplerBuffer sky_geometry_flag_texture_object;
uniform isamplerBuffer hidden_flag_texture_object;

flat in int faceID;

void main(){

    // Delete hidden/deleted primitives
    if( texelFetch(hidden_flag_texture_object, faceID).r == 0 ){
        discard;
    }

    vec4 fragmentColor = texelFetch(color_texture_object, faceID);
    vec3 normal = texelFetch(normal_texture_object, faceID).rgb;
    int textureFlag = texelFetch(texture_flag_texture_object, faceID).r;
    int textureID = texelFetch(texture_ID_texture_object, faceID).r;
    int coordinateFlag = texelFetch(coordinate_flag_texture_object, faceID).r;
    int skyGeometryFlag = texelFetch(sky_geometry_flag_texture_object, faceID).r;

    //There are several confusing flags that determine how this shader colors fragments
    // textureFlag:
    //    = 0 -> color fragments based on their RGB color value
    //    = 1 -> color fragments based on texture map color, set transparency according to texture alpha-value
    //    = 2 -> color fragments based on their RGB color value, set transparency according to texture alpha-value
    //    = 3 -> color fragments based on their RGB color value, set transparency according to texture r-value
    // coordinateFlag:
    //    = 0 -> 2D projection in window-normalized coordinates
    //    = 1 -> 3D Cartesian coordinates
    //    = 2 -> special case using 3D coordinates but with no shading applied (used for sky dome)
    // lightingModel:
    //    = 0 -> color fragments based only on their RGBA color value (no shading, etc.)
    //    = 1 -> use a simplified Phong lighting model to shade fragments
    //    = 2 -> use a simplified Phong lighting model plus the shadow map to shade/shadow fragments

    vec3 ld = lightDirection;

    float visibility = 1.0;
    if(lightingModel==2 && coordinateFlag==1 && dot(normal,ld)>0 ){ //Simplified Phong lighting model + shadows

        vec3 proj = ShadowCoord.xyz / ShadowCoord.w;   // 0…1 in valid area

        bool inside =
        proj.x >= 0.0 && proj.x <= 1.0 &&
        proj.y >= 0.0 && proj.y <= 1.0 &&
        proj.z >= 0.0 && proj.z <= 1.0;

        if( inside ){

            vec2 poissonDisk[4] = vec2[](
            vec2(-0.94201624, -0.39906216),
            vec2(0.94558609, -0.76890725),
            vec2(-0.094184101, -0.92938870),
            vec2(0.34495938, 0.29387760)
            );

            for (int i=0;i<4;i++){
                if (texture(shadowMap, (ShadowCoord.xy/ShadowCoord.w+poissonDisk[i]/140000.0)).x  <  (ShadowCoord.z/ShadowCoord.w)){
                    visibility-=0.35;
                }
            }

        }
    }

    vec2 wrapped = fract(texcoord) * uv_scale;
    vec3 texcoord3 = vec3(wrapped, float(textureID));

    if( textureFlag==0 ){//Color by interpolating the colors at vertices
        color = fragmentColor;
        color.rgb = color.rgb*1.5;
    }else if( textureFlag==1 ){//Color by texture map
        color = texture(textureSampler, texcoord3);
        if(color.a<0.5 ){
            discard;
        }
    }else if( textureFlag==2 ){//Color by interpolating the colors at vertices, and set the transparency according to the red channel of the texture map given by textureSampler
        color = fragmentColor*1.5;
        color.a = texture(textureSampler, texcoord3).a;
        if( color.a<0.5 ){
            discard;
        }
    }else if( textureFlag==3 ){//Color by interpolating the colors at vertices, and set the transparency according to the red channel of the texture map given by textureSampler
        float rawAlpha = texture(textureSampler, texcoord3).r;
        float edge = fwidth(rawAlpha);
        float alpha = smoothstep(0.5 - edge, 0.5 + edge, rawAlpha);
        color = vec4(fragmentColor.rgb * alpha, alpha);
        color.a = clamp(alpha,0,1);
        if( color.a < 0.01 ){
            discard;
        }
    }

    // Apply lighting only to non-sky geometry
    if( lightingModel>0 && coordinateFlag==1 && skyGeometryFlag==0 ){ //Simplified Phong lighting model
        vec3 intensity = lightIntensity*vec3(1.0,0.9,0.8);
        color = vec4( intensity*( 0.75*color.rgb + visibility*max(0,dot(normal,ld))*color.rgb ) , color.a );
    }

}