#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 uv;
layout(location = 2) in int face_index;

uniform mat4 MVP;
uniform mat4 DepthBiasMVP;

// Outputs to geometry shader (prefixed with vs_)
out vec2 vs_texcoord;
out vec2 vs_uv_scale;
out vec4 vs_ShadowCoord;
flat out int vs_faceID;

uniform isamplerBuffer texture_ID_texture_object;
uniform isamplerBuffer coordinate_flag_texture_object;
uniform samplerBuffer uv_rescale;

void main() {
    vs_faceID = face_index;

    int textureID = texelFetch(texture_ID_texture_object, vs_faceID).r;
    int coordinateFlag = texelFetch(coordinate_flag_texture_object, vs_faceID).r;

    vec4 v = vec4(vertexPosition_modelspace, 1);

    // Transform to clip space
    if (coordinateFlag == 0) {
        gl_Position = v;
    } else {
        gl_Position = MVP * v;
    }

    vec4 scale = texelFetch(uv_rescale, textureID);
    vs_texcoord = uv;
    vs_uv_scale = scale.rg;

    vs_ShadowCoord = DepthBiasMVP * v;
}
