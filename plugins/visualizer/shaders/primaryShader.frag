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

//Multiplier applied to vertex-interpolated colors. Normally 1.5, which brightens ordinary
//renders. Visualizer::enableExactColorMode() sets it to 1.0 so that a color written into the
//Context is read back out of the framebuffer unchanged, which is required when the framebuffer
//is used to carry data (e.g. object ID codes) rather than to be looked at.
uniform float colorBoost;

uniform int Rbound;

//Enables the linear-light rendering pipeline: albedo is decoded from sRGB to linear before
//shading, and the shaded result is tone-mapped and re-encoded to sRGB at the end of this shader.
//Disabled by Visualizer::enableExactColorMode(), where the framebuffer carries data (e.g. object
//ID codes) rather than an image and must read back bit-unchanged.
uniform int linearPipeline;

//Exposure applied in linear light before tone mapping.
uniform float exposure;

in vec3 worldPosition;

//Per-vertex normal interpolated by the rasterizer, used when smooth shading is enabled.
in vec3 interpolatedNormal;

//Selects between smooth (interpolated per-vertex) and flat (per-face) normals. Geometry with no
//authored vertex normals carries the face normal replicated across its vertices, so it shades the
//same either way; the setting only changes the appearance of meshes that supply true vertex
//normals, such as a Polymesh loaded from a file that provides them.
uniform int smoothShading;

//Camera position in world space, used to form the view vector for the specular term.
uniform vec3 cameraPosition;

//Phong material parameters. ambient/diffuse/specular are reflectance weights applied to the
//corresponding lighting terms; shininess is the specular exponent, where larger values give a
//tighter, glossier highlight.
uniform float materialAmbient;
uniform float materialDiffuse;
uniform float materialSpecular;
uniform float materialShininess;

//Packed per-material Phong parameter table, one RGBA texel per material holding
//(ambient, diffuse, specular, shininess), together with the per-primitive index into it. A
//primitive whose index is negative uses the global uniforms above. The parameters are stored once
//per material rather than once per primitive, so a scene with millions of primitives and a handful
//of materials pays for the handful.
uniform samplerBuffer phongMaterialTable;
uniform int phongMaterialTableSize;
uniform isamplerBuffer material_index_texture_object;

//Hemispheric ambient: incoming ambient light is approximated as a sky color from above and a
//ground-bounce color from below, blended by the surface normal's vertical component. This gives
//unlit surfaces soft directional grounding instead of the flat, uniform fill a constant ambient
//term produces.
uniform vec3 ambientSkyColor;
uniform vec3 ambientGroundColor;

//sRGB transfer functions (IEC 61966-2-1). Applied per-component to color only, never to alpha,
//which is a coverage value and is already linear.
vec3 srgbToLinear(vec3 c){
    bvec3 cutoff = lessThanEqual(c, vec3(0.04045));
    vec3 low  = c/12.92;
    vec3 high = pow((c+vec3(0.055))/1.055, vec3(2.4));
    return mix(high, low, vec3(cutoff));
}

vec3 linearToSrgb(vec3 c){
    c = max(c, vec3(0.0));
    bvec3 cutoff = lessThanEqual(c, vec3(0.0031308));
    vec3 low  = c*12.92;
    vec3 high = 1.055*pow(c, vec3(1.0/2.4)) - vec3(0.055);
    return mix(high, low, vec3(cutoff));
}

//ACES filmic tone curve (Narkowicz 2015 fit). Maps unbounded linear radiance into [0,1] with a
//soft highlight rolloff, so bright sunlit surfaces desaturate gracefully instead of clipping flat
//to white the way a hard clamp does.
vec3 tonemapACES(vec3 x){
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

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
    //    = 4 -> color fragments based on texture map color, blending the texture alpha rather than
    //           thresholding it (image overlays whose edges are genuinely partially transparent)
    // coordinateFlag:
    //    = 0 -> 2D projection in window-normalized coordinates
    //    = 1 -> 3D Cartesian coordinates
    //    = 2 -> special case using 3D coordinates but with no shading applied (used for sky dome)
    // lightingModel:
    //    = 0 -> color fragments based only on their RGBA color value (no shading, etc.)
    //    = 1 -> use a simplified Phong lighting model to shade fragments
    //    = 2 -> use a simplified Phong lighting model plus the shadow map to shade/shadow fragments

    vec3 ld = lightDirection;

    vec3 viewDirection = normalize(cameraPosition - worldPosition);

    //Two normals are needed, and conflating them is a bug: how much light a surface receives is a
    //property of the geometry and the light source alone, while whether the front or the back of a
    //single-sided surface is being looked at is a property of the camera.
    //
    //geometricNormal is the true surface orientation and drives everything sun-dependent: the
    //diffuse and specular terms and the shadow-map lookup. It is never flipped, so a leaf turned
    //away from the sun receives no direct light no matter where it is viewed from.
    vec3 geometricNormal = normalize(smoothShading==1 ? interpolatedNormal : normal);

    //viewFacingNormal points toward the camera and drives only the hemispheric ambient term, so
    //that the unlit back of a single-sided primitive picks up ambient light from the side it is
    //actually being viewed from rather than rendering black.
    vec3 viewFacingNormal = geometricNormal;
    if( lightingModel>0 && coordinateFlag==1 && dot(viewFacingNormal, viewDirection) < 0.0 ){
        viewFacingNormal = -viewFacingNormal;
    }

    float visibility = 1.0;
    if(lightingModel==2 && coordinateFlag==1 && dot(geometricNormal,ld)>0 ){ //Blinn-Phong lighting model + shadows

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
        color.rgb = color.rgb*colorBoost;
        if( linearPipeline==1 ){
            color.rgb = srgbToLinear(clamp(color.rgb,0.0,1.0));
        }
    }else if( textureFlag==1 ){//Color by texture map
        color = texture(textureSampler, texcoord3);
        if(color.a<0.5 ){
            discard;
        }
        if( linearPipeline==1 ){
            color.rgb = srgbToLinear(color.rgb);
        }
    }else if( textureFlag==2 ){//Color by interpolating the colors at vertices, and set the transparency according to the red channel of the texture map given by textureSampler
        color = fragmentColor*colorBoost;
        color.a = texture(textureSampler, texcoord3).a;
        if( color.a<0.5 ){
            discard;
        }
        if( linearPipeline==1 ){
            color.rgb = srgbToLinear(clamp(color.rgb,0.0,1.0));
        }
    }else if( textureFlag==4 ){//Color by texture map, blending the texture's alpha rather than testing it
        //Used for image overlays such as the watermark. textureFlag==1 discards fragments below a
        //fixed alpha threshold, which is correct for the alpha-masked cutouts of leaf and bark
        //textures but destroys the antialiasing of an image whose edges are genuinely partially
        //transparent: those edge pixels are exactly the ones the test throws away, leaving a
        //jagged boundary. Here the alpha is used as-is and left to the blend function.
        color = texture(textureSampler, texcoord3);
        if( color.a < 0.004 ){
            discard;
        }
    }else if( textureFlag==3 ){//Color by interpolating the colors at vertices, and set the transparency according to the red channel of the texture map given by textureSampler
        //The red channel holds the per-pixel antialiased coverage computed by FreeType, not a
        //signed distance field, so it is used directly as the alpha value. Thresholding it would
        //discard the antialiasing and leave hard, aliased glyph edges. The color is not
        //premultiplied by alpha here because the blend function is GL_SRC_ALPHA/
        //GL_ONE_MINUS_SRC_ALPHA, which applies that multiply itself.
        float alpha = texture(textureSampler, texcoord3).r;
        color = vec4(fragmentColor.rgb, clamp(alpha,0,1));
        if( color.a < 0.01 ){
            discard;
        }
    }

    // Apply lighting only to non-sky geometry
    if( lightingModel>0 && coordinateFlag==1 && skyGeometryFlag==0 ){ //Blinn-Phong lighting model
        vec3 intensity = lightIntensity*vec3(1.0,0.9,0.8);
        vec3 albedo = color.rgb;

        //Per-material Phong parameters, falling back to the global ones when this primitive's
        //material specifies none. The bounds check keeps a stale index from a previous build from
        //sampling outside the table.
        float phongAmbient = materialAmbient;
        float phongDiffuse = materialDiffuse;
        float phongSpecular = materialSpecular;
        float phongShininess = materialShininess;
        int materialIndex = texelFetch(material_index_texture_object, faceID).r;
        if( materialIndex >= 0 && materialIndex < phongMaterialTableSize ){
            vec4 phongParams = texelFetch(phongMaterialTable, materialIndex);
            phongAmbient = phongParams.r;
            phongDiffuse = phongParams.g;
            phongSpecular = phongParams.b;
            phongShininess = phongParams.a;
        }

        //Hemispheric ambient uses the view-facing normal: it describes light arriving at the side
        //of the surface being looked at, so the shaded back of a leaf picks up ground bounce rather
        //than rendering black. normal.z is the vertical component in Helios' z-up convention;
        //remapping it from [-1,1] to [0,1] blends ground bounce into sky light.
        float hemisphereBlend = 0.5*viewFacingNormal.z + 0.5;
        vec3 ambient = phongAmbient*mix(ambientGroundColor, ambientSkyColor, hemisphereBlend)*albedo;

        //Direct illumination uses the true geometric normal, so a surface turned away from the
        //light receives none of it regardless of which side the camera is on.
        vec3 N = geometricNormal;

        float NdotL = max(0.0, dot(N, ld));
        vec3 diffuse = phongDiffuse*NdotL*albedo;

        //Blinn-Phong specular, using the half-vector between the light and view directions. The
        //highlight is white rather than albedo-tinted, which is correct for the dielectric surfaces
        //(leaves, bark, soil) this renders. It is gated on NdotL so that a highlight cannot appear
        //on a surface turned away from the light, and is attenuated by the same shadow visibility
        //as the diffuse term.
        vec3 specular = vec3(0.0);
        if( NdotL > 0.0 ){
            vec3 halfVector = normalize(ld + viewDirection);
            specular = vec3(phongSpecular*pow(max(0.0, dot(N, halfVector)), phongShininess));
        }

        color = vec4( intensity*( ambient + visibility*(diffuse + specular) ), color.a );
    }

    // Linear-light output stage: expose, tone-map, and re-encode to sRGB. Only fragments that were
    // decoded to linear light above may pass through here; glyph coverage (textureFlag==3) and
    // blended image overlays (textureFlag==4) are authored in display space and are left untouched,
    // as is all 2D screen-space geometry. Alpha is never transformed.
    if( linearPipeline==1 && textureFlag!=3 && textureFlag!=4 && coordinateFlag==1 ){
        color.rgb = linearToSrgb(tonemapACES(color.rgb*exposure));
    }

}