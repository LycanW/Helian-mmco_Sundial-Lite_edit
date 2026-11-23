// Screen-space block light extraction Stage 1 (160 x 90 tiles).
// Scans every single screen pixel inside each tile (zero gaps at <= 1440p!), skips particles,
// locks color onto peak emissive pixels (so dim torch sticks never dilute the flame brightness),
// and snaps every emitter to its exact 3D integer world block center.

layout(location = 0) out vec4 texBuffer8;
layout(location = 1) out vec4 texBuffer9;

#include "/settings/GlobalSettings.glsl"
#include "/libs/Uniform.glsl"
#include "/libs/Common.glsl"
#include "/libs/GbufferData.glsl"
#include "/libs/ScreenLight.glsl"

void main() {
    ivec2 tile = ivec2(gl_FragCoord.st);
    if (tile.x >= SS_TILE1_W || tile.y >= SS_TILE1_H) {
        discard;
    }

    vec2 renderPixels = floor(screenSize * screenEdge);
    ivec2 maxTexel = ivec2(renderPixels) - ivec2(1);
    vec2 tileGrid = vec2(float(SS_TILE1_W), float(SS_TILE1_H));

    ivec2 pMin = clamp(ivec2(floor(vec2(tile) * renderPixels / tileGrid)), ivec2(0), maxTexel);
    ivec2 pMax = clamp(ivec2(ceil(vec2(tile + 1) * renderPixels / tileGrid)), ivec2(0), maxTexel + ivec2(1));
    ivec2 span = max(pMax - pMin, ivec2(1));
    ivec2 stepPx = max(ivec2(1), (span + ivec2(15)) / 16);

    float bestScore = 0.0;
    vec3 bestBlockViewPos = vec3(0.0);

    vec3 colorSum = vec3(0.0);
    float colorWeightSum = 0.0;
    float maxEmissive = 0.0;

    for (int py = pMin.y; py < pMax.y; py += stepPx.y) {
        for (int px = pMin.x; px < pMax.x; px += stepPx.x) {
            ivec2 sTexel = ivec2(px, py);

            // Ultra-fast early reject: single fetch checks emissive before touching depth/albedo/normal
            // Threshold 0.30 ignores dim brown torch sticks (~0.22) while keeping all true glowing flames/blocks
            float emissive = unpack2x8Bit(texelFetch(colortex1, sTexel, 0).w).y;
            if (emissive <= 0.30) {
                continue;
            }

            float rawDepth = uintBitsToFloat(texelFetch(colortex6, sTexel, 0).r);
            // Ignore sky, hand (> 1.0), and LOD (< 0.0)
            if (rawDepth >= 0.999999 || rawDepth > 1.0 || rawDepth < 0.0) {
                continue;
            }

            vec4 tex2 = texelFetch(colortex2, sTexel, 0);
            float matID = round(unpackM3P13(tex2.w).x * 7.0);
            // Ignore particles (flame/spark particles) so they never spawn flickering ghost lights
            if (matID == MAT_PARTICLE) {
                continue;
            }

            vec2 exactUV = (vec2(sTexel) + 0.5) * texelSize;
            vec3 sViewPos = screenToViewPos(exactUV, rawDepth);
            float viewZ = -sViewPos.z;
            if (viewZ < 0.15 || viewZ > 72.0) {
                continue;
            }

            vec3 sAlbedo = pow(texelFetch(colortex0, sTexel, 0).rgb, vec3(2.2));
            vec3 sGeoNormalView = decodeNormal(tex2.rg);
            vec3 sWorldGeoNormal = normalize(mat3(gbufferModelViewInverse) * sGeoNormalView);
            vec3 sWorldPos = viewToWorldPos(sViewPos);
            vec3 sMcPos = sWorldPos + cameraPosition;

            // Snap to exact 3D integer block center (0.04 inward bias works for both full cubes and wall torches)
            vec3 blockCenterMc = floor(sMcPos - sWorldGeoNormal * 0.04) + 0.5;
            vec3 blockCenterView = worldToViewPos(blockCenterMc - cameraPosition);

            vec3 emitColor = sAlbedo * emissive;
            float lum = max(luminanceLiner(emitColor), 1e-3);
            float pMaxC = max(emitColor.r, max(emitColor.g, emitColor.b));
            float pMinC = min(emitColor.r, min(emitColor.g, emitColor.b));
            float pSat = (pMaxC > 1e-4) ? (pMaxC - pMinC) / pMaxC : 0.0;
            // Favor saturated flame/core pixels over white hot-spots so torch/lantern hue stays rock-solid
            float colorW = (emissive * emissive) * (0.25 + 1.5 * pSat);
            float score = (emissive * emissive) * (0.35 + lum) / (1.0 + viewZ * 0.08);

            if (bestScore <= 0.0) {
                bestScore = score;
                bestBlockViewPos = blockCenterView;
                colorSum = (emitColor / lum) * colorW;
                colorWeightSum = colorW;
                maxEmissive = emissive;
            } else {
                vec3 d = blockCenterView - bestBlockViewPos;
                if (dot(d, d) < 0.25) {
                    // Exact same 1x1x1 glowing block: saturation-weighted chromaticity average
                    colorSum += (emitColor / lum) * colorW;
                    colorWeightSum += colorW;
                    maxEmissive = max(maxEmissive, emissive);
                    if (score > bestScore) {
                        bestScore = score;
                    }
                } else if (score > bestScore * 1.1) {
                    bestScore = score;
                    bestBlockViewPos = blockCenterView;
                    colorSum = (emitColor / lum) * colorW;
                    colorWeightSum = colorW;
                    maxEmissive = emissive;
                }
            }
        }
    }

    if (colorWeightSum > 0.0) {
        vec3 avgColor = colorSum / colorWeightSum;
        float maxC = max(avgColor.r, max(avgColor.g, avgColor.b));
        float minC = min(avgColor.r, min(avgColor.g, avgColor.b));
        float sat = (maxC > 1e-4) ? (maxC - minC) / maxC : 0.0;
        vec3 gray = vec3(dot(avgColor, vec3(0.3333)));
        avgColor = max(vec3(0.0), mix(gray, avgColor, 1.0 + 0.25 * sat));

        // Normalize luminance to a steady intensity based only on peak emissive level,
        // preventing brightness spikes when partial occluders hide dimmer parts of a torch/block.
        float cLum = max(luminanceLiner(avgColor), 1e-4);
        float steadyIntensity = smoothstep(0.28, 0.65, maxEmissive) * 1.0;
        vec3 steadyColor = (avgColor / cLum) * steadyIntensity;

        texBuffer8 = vec4(bestBlockViewPos, maxEmissive);
        texBuffer9 = vec4(steadyColor, colorWeightSum);
    } else {
        texBuffer8 = vec4(0.0);
        texBuffer9 = vec4(0.0);
    }
}

/* RENDERTARGETS: 8,9 */
