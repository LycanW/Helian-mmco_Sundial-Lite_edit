// Screen-space block light extraction Stage 2 (32 x 18 tiles).
// Reduces 5x5 Stage-1 tiles (160x90 -> 32x18) preserving peak emissive color and exact block-center coordinates.

layout(location = 0) out vec4 texBuffer8;
layout(location = 1) out vec4 texBuffer9;

#include "/settings/GlobalSettings.glsl"
#include "/libs/Uniform.glsl"
#include "/libs/Common.glsl"
#include "/libs/ScreenLight.glsl"

void main() {
    ivec2 tile = ivec2(gl_FragCoord.st);
    if (tile.x >= SS_TILE2_W || tile.y >= SS_TILE2_H) {
        discard;
    }

    ivec2 base = tile * 5;

    float bestScore = 0.0;
    vec3 bestPos = vec3(0.0);

    vec3 colorSum = vec3(0.0);
    float weightSum = 0.0;
    float maxEmissive = 0.0;

    for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            ivec2 sTexel = base + ivec2(dx, dy);
            vec4 p = texelFetch(colortex8, sTexel, 0);
            if (p.w <= 0.0) {
                continue;
            }
            vec4 c = texelFetch(colortex9, sTexel, 0);
            float viewZ = max(-p.z, 0.15);
            float lum = max(luminanceLiner(c.rgb), 0.05);
            float score = (p.w * p.w) * (0.4 + lum) / (1.0 + viewZ * 0.08);
            float peakW = pow2(pow2(p.w)) + 1e-6;

            if (bestScore <= 0.0) {
                bestScore = score;
                bestPos = p.xyz;
                colorSum = c.rgb * peakW;
                weightSum = peakW;
                maxEmissive = p.w;
            } else {
                vec3 d = p.xyz - bestPos;
                if (dot(d, d) < 0.25) {
                    colorSum += c.rgb * peakW;
                    weightSum += peakW;
                    maxEmissive = max(maxEmissive, p.w);
                    if (score > bestScore) {
                        bestScore = score;
                    }
                } else if (score > bestScore * 1.1) {
                    bestScore = score;
                    bestPos = p.xyz;
                    colorSum = c.rgb * peakW;
                    weightSum = peakW;
                    maxEmissive = p.w;
                }
            }
        }
    }

    if (weightSum > 0.0) {
        texBuffer8 = vec4(bestPos, maxEmissive);
        texBuffer9 = vec4(colorSum / weightSum, weightSum);
    } else {
        texBuffer8 = vec4(0.0);
        texBuffer9 = vec4(0.0);
    }
}

/* RENDERTARGETS: 8,9 */
