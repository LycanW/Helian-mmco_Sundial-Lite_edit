// Screen-space block light extraction Stage 3 (128 x 1 compacted dense light list).
// Performs exact global per-block deduplication AND maintains short temporal grace memory
// in colortex10/11 so smoke particles drifting over torches or 1-frame TAA misses never cause blinking.

layout(location = 0) out vec4 texBuffer10;
layout(location = 1) out vec4 texBuffer11;

#include "/settings/GlobalSettings.glsl"
#include "/libs/Uniform.glsl"
#include "/libs/Common.glsl"
#include "/libs/GbufferData.glsl"
#include "/libs/ScreenLight.glsl"

void main() {
    ivec2 coord = ivec2(gl_FragCoord.st);
    if (coord.x >= SS_LIGHT_LIST_MAX || coord.y >= 1) {
        discard;
    }

    int targetIdx = coord.x;
    int found = 0;

    // Pass 1: Emit all unique lights detected directly on screen THIS frame (grace = 1.0)
    for (int y = 0; y < SS_TILE2_H; y++) {
        for (int x = 0; x < SS_TILE2_W; x++) {
            vec4 p = texelFetch(colortex8, ivec2(x, y), 0);
            if (p.w <= 0.0) {
                continue;
            }

            // Global highest-emissive-wins per-block deduplication across the entire 32x18 grid:
            // prevents a lower tile containing a dim block edge from overriding an upper tile's bright flame tip.
            bool isDominated = false;
            for (int py = 0; py < SS_TILE2_H && !isDominated; py++) {
                for (int px = 0; px < SS_TILE2_W; px++) {
                    if (px == x && py == y) {
                        continue;
                    }
                    vec4 otherP = texelFetch(colortex8, ivec2(px, py), 0);
                    if (otherP.w > 0.0) {
                        vec3 diff = p.xyz - otherP.xyz;
                        if (dot(diff, diff) < 0.25) {
                            bool otherEarlier = (py < y) || (py == y && px < x);
                            if (otherP.w > p.w + 1e-4 || (abs(otherP.w - p.w) <= 1e-4 && otherEarlier)) {
                                isDominated = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (isDominated) {
                continue;
            }

            if (found == targetIdx) {
                vec4 c = texelFetch(colortex9, ivec2(x, y), 0);
                // Strong temporal inertia on light color + freeze updates near screen edges
                // so partially clipped torches/lanterns never shift color temperature as you move.
                if (frameCounter > 1) {
                    float colorAlpha = clamp(1.0 - exp(-frameTime * 2.0), 0.01, 0.08);
                    if (p.z < -0.15) {
                        vec2 pUV = viewToProjectionPos(p.xyz).xy * 0.5 + 0.5;
                        vec2 edgeD = min(pUV, screenEdge - pUV) / max(screenEdge, vec2(1e-4));
                        colorAlpha *= smoothstep(0.04, 0.14, min(edgeD.x, edgeD.y));
                    }
                    for (int j = 0; j < SS_LIGHT_LIST_MAX; j++) {
                        vec4 oldP = texelFetch(colortex10, ivec2(j, 0), 0);
                        if (oldP.w <= 0.0) {
                            break;
                        }
                        vec3 oldViewNow = worldToViewPos(prevViewToWorldPos(oldP.xyz) - cameraMovement);
                        vec3 d = p.xyz - oldViewNow;
                        if (dot(d, d) < 0.25) {
                            vec4 oldC = texelFetch(colortex11, ivec2(j, 0), 0);
                            c.rgb = mix(oldC.rgb, c.rgb, colorAlpha);
                            break;
                        }
                    }
                }
                texBuffer10 = vec4(p.xyz, p.w);
                texBuffer11 = vec4(c.rgb, 1.0);
                return;
            }
            found++;
        }
    }

    // Pass 2: Carry over briefly-occluded lights (e.g. smoke/flame particles passing in front of a torch)
    // with particle-aware grace retention so torches peeking over a wall never blink on/off!
    if (frameCounter > 1) {
        float dt = clamp(frameTime, 0.004, 0.05);
        ivec2 maxTexel = ivec2(floor(screenSize * screenEdge)) - ivec2(1);
        for (int j = 0; j < SS_LIGHT_LIST_MAX; j++) {
            vec4 oldP = texelFetch(colortex10, ivec2(j, 0), 0);
            if (oldP.w <= 0.0) {
                break;
            }
            vec4 oldC = texelFetch(colortex11, ivec2(j, 0), 0);
            vec3 currViewPos = worldToViewPos(prevViewToWorldPos(oldP.xyz) - cameraMovement);

            bool onScreen = false;
            bool coveredByParticle = false;
            if (currViewPos.z < -0.15) {
                vec3 projPos = viewToProjectionPos(currViewPos);
                vec2 uv = projPos.xy * 0.5 + 0.5;
                onScreen = all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, screenEdge));

                // Also check slightly above block center (torch flame tip sits at y + 0.2 relative to center)
                vec3 tipViewPos = currViewPos + mat3(gbufferModelView) * vec3(0.0, 0.22, 0.0);
                vec2 tipUV = (tipViewPos.z < -0.15) ? (viewToProjectionPos(tipViewPos).xy * 0.5 + 0.5) : uv;
                bool tipOnScreen = all(greaterThanEqual(tipUV, vec2(0.0))) && all(lessThan(tipUV, screenEdge));
                onScreen = onScreen || tipOnScreen;

                if (tipOnScreen) {
                    ivec2 centerPx = clamp(ivec2(tipUV * screenSize), ivec2(0), maxTexel);
                    for (int dy = -2; dy <= 2 && !coveredByParticle; dy += 2) {
                        for (int dx = -2; dx <= 2; dx += 2) {
                            ivec2 sp = clamp(centerPx + ivec2(dx, dy), ivec2(0), maxTexel);
                            float matID = round(unpackM3P13(texelFetch(colortex2, sp, 0).w).x * 7.0);
                            if (matID == MAT_PARTICLE) {
                                coveredByParticle = true;
                                break;
                            }
                        }
                    }
                }
            }

            // If a smoke/flame particle is covering the light tip, hold grace very slowly (~2.5s)
            // so standing still with a peeking torch NEVER blinks off; otherwise fade smoothly.
            float decayRate = coveredByParticle ? 0.35 : (onScreen ? 1.8 : 4.5);
            float newGrace = oldC.w - dt * decayRate;
            if (newGrace <= 0.01) {
                continue;
            }

            // Skip if this exact block was already detected in Pass 1 (current frame's 32x18 grid)
            bool seenThisFrame = false;
            for (int y = 0; y < SS_TILE2_H && !seenThisFrame; y++) {
                for (int x = 0; x < SS_TILE2_W; x++) {
                    vec4 p = texelFetch(colortex8, ivec2(x, y), 0);
                    if (p.w > 0.0) {
                        vec3 diff = currViewPos - p.xyz;
                        if (dot(diff, diff) < 0.25) {
                            seenThisFrame = true;
                            break;
                        }
                    }
                }
            }
            if (seenThisFrame) {
                continue;
            }

            if (found == targetIdx) {
                texBuffer10 = vec4(currViewPos, oldP.w);
                texBuffer11 = vec4(oldC.rgb, newGrace);
                return;
            }
            found++;
        }
    }

    // Null terminator
    texBuffer10 = vec4(0.0);
    texBuffer11 = vec4(0.0);
}

/* RENDERTARGETS: 10,11 */
