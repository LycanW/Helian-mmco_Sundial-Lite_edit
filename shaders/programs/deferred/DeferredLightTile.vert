// Viewport-clipped vertex shader for screen-space light tile reduction passes.
// Expects SS_TARGET_W and SS_TARGET_H to be defined by the wrapper .vsh.

#include "/settings/GlobalSettings.glsl"
#include "/libs/Uniform.glsl"

void main() {
    vec2 offset = vec2(-float(gl_VertexID & 2), float(min(gl_VertexID, 1) << 1)) + vec2(1.0, -1.0);
    vec2 tileScale = min(vec2(float(SS_TARGET_W) + 1.0, float(SS_TARGET_H) + 1.0) * texelSize, vec2(1.0));
    offset *= tileScale;
    gl_Position = vec4(offset * 2.0 - 1.0, 0.0, 1.0);
}
