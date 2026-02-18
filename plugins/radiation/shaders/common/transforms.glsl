/** transforms.glsl - Matrix and vector transformations
 *
 * IMPORTANT: Matrix Layout Convention
 * ====================================
 * CPU-side (C++): Matrices are stored in ROW-MAJOR order (16 floats)
 *   Layout: [m0 m1 m2 m3] (row 0), [m4 m5 m6 m7] (row 1), etc.
 *
 * GPU-side (GLSL): Matrices are COLUMN-MAJOR (GLSL standard)
 *   The floats_to_mat4() function transposes during conversion
 *
 * This ensures correct transformation on both CPU (BVHBuilder) and GPU (shaders).
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

// Convert row-major float array to mat4 (transposes to column-major)
// Input: m[16] in row-major order from C++ side
// Output: mat4 in column-major order (GLSL standard)
mat4 floats_to_mat4(float m[16]) {
    return mat4(
        m[0], m[4], m[8], m[12],   // Column 0 (from row-major rows)
        m[1], m[5], m[9], m[13],   // Column 1
        m[2], m[6], m[10], m[14],  // Column 2
        m[3], m[7], m[11], m[15]   // Column 3
    );
}

// Build ZYX Euler angle rotation matrix (matches OptiX d_makeTransformMatrix)
// Input: rotation angles (rx, ry, rz) in radians
// Output: 4x4 rotation matrix (column-major)
mat4 make_rotation_matrix(vec3 rotation) {
    float cx = cos(rotation.x);
    float sx = sin(rotation.x);
    float cy = cos(rotation.y);
    float sy = sin(rotation.y);
    float cz = cos(rotation.z);
    float sz = sin(rotation.z);

    // ZYX Euler order: Rz * Ry * Rx (matches OptiX RayTracing.cuh:289-307)
    return mat4(
        cz * cy,                    sz * cy,                    -sy,      0.0,
        cz * sy * sx - sz * cx,     sz * sy * sx + cz * cx,     cy * sx,  0.0,
        cz * sy * cx + sz * sx,     sz * sy * cx - cz * sx,     cy * cx,  0.0,
        0.0,                        0.0,                        0.0,      1.0
    );
}

#endif // TRANSFORMS_GLSL
