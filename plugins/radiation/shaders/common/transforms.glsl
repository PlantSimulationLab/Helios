/** transforms.glsl - Matrix and vector transformations
 */

#ifndef TRANSFORMS_GLSL
#define TRANSFORMS_GLSL

// Transform point by 4x4 row-major matrix
vec3 transform_point(mat4 transform, vec3 point) {
    vec4 result = transform * vec4(point, 1.0);
    return result.xyz;
}

// Transform direction by 3x3 rotation part of 4x4 matrix (no translation)
vec3 transform_direction(mat4 transform, vec3 dir) {
    return mat3(transform) * dir;
}

// Inverse transform direction (for normals)
vec3 inverse_transform_direction(mat4 transform, vec3 dir) {
    // For orthonormal transforms, inverse = transpose
    return transpose(mat3(transform)) * dir;
}

// Convert row-major float array to mat4
mat4 floats_to_mat4(float m[16]) {
    return mat4(
        m[0], m[4], m[8], m[12],
        m[1], m[5], m[9], m[13],
        m[2], m[6], m[10], m[14],
        m[3], m[7], m[11], m[15]
    );
}

#endif // TRANSFORMS_GLSL
