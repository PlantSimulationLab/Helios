#version 330 core

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

// Inputs from vertex shader (per-vertex)
in vec2 vs_texcoord[];
in vec2 vs_uv_scale[];
in vec4 vs_ShadowCoord[];
flat in int vs_faceID[];

// Outputs to fragment shader (per-vertex of generated quad)
out vec2 texcoord;
out vec2 uv_scale;
out vec4 ShadowCoord;
flat out int faceID;

// Uniform for line width (passed from the rendering code)
uniform float lineWidth;

// Uniform for viewport dimensions (needed for aspect ratio correction)
uniform vec2 viewportSize;

void main() {
    // Get the two endpoints of the line in clip space
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    // Convert to normalized device coordinates (NDC)
    vec2 ndc0 = p0.xy / p0.w;
    vec2 ndc1 = p1.xy / p1.w;

    // Convert to screen space (pixels)
    vec2 screen0 = (ndc0 + 1.0) * 0.5 * viewportSize;
    vec2 screen1 = (ndc1 + 1.0) * 0.5 * viewportSize;

    // Calculate line direction in screen space
    vec2 lineDir = normalize(screen1 - screen0);

    // Calculate perpendicular direction (90 degrees rotation)
    vec2 perpDir = vec2(-lineDir.y, lineDir.x);

    // Calculate offset in screen space (half the line width on each side)
    vec2 offset = perpDir * (lineWidth * 0.5);

    // Convert offset back to NDC space
    vec2 ndcOffset = offset / (viewportSize * 0.5);

    // Generate 4 vertices for the quad (triangle strip)
    // We generate the quad in the order: bottom-left, bottom-right, top-left, top-right
    // This creates a triangle strip that forms a rectangle

    // Vertex 0: p0 - offset (bottom-left)
    gl_Position = vec4((ndc0 - ndcOffset) * p0.w, p0.z, p0.w);
    texcoord = vs_texcoord[0];
    uv_scale = vs_uv_scale[0];
    ShadowCoord = vs_ShadowCoord[0];
    faceID = vs_faceID[0];
    EmitVertex();

    // Vertex 1: p0 + offset (top-left)
    gl_Position = vec4((ndc0 + ndcOffset) * p0.w, p0.z, p0.w);
    texcoord = vs_texcoord[0];
    uv_scale = vs_uv_scale[0];
    ShadowCoord = vs_ShadowCoord[0];
    faceID = vs_faceID[0];
    EmitVertex();

    // Vertex 2: p1 - offset (bottom-right)
    gl_Position = vec4((ndc1 - ndcOffset) * p1.w, p1.z, p1.w);
    texcoord = vs_texcoord[1];
    uv_scale = vs_uv_scale[1];
    ShadowCoord = vs_ShadowCoord[1];
    faceID = vs_faceID[1];
    EmitVertex();

    // Vertex 3: p1 + offset (top-right)
    gl_Position = vec4((ndc1 + ndcOffset) * p1.w, p1.z, p1.w);
    texcoord = vs_texcoord[1];
    uv_scale = vs_uv_scale[1];
    ShadowCoord = vs_ShadowCoord[1];
    faceID = vs_faceID[1];
    EmitVertex();

    EndPrimitive();
}
