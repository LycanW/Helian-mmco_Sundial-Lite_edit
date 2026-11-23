// Shared constants for 3-stage hierarchical screen-space emissive block light extraction.
// Stage 1 (deferred1): 160 x 90 exact full-pixel emissive scan (zero missed 1px torches)
// Stage 2 (deferred2):  32 x 18 tile reduction (5x5 merge)
// Stage 3 (deferred3): 128 x  1 compacted dense light list with temporal grace memory (colortex10/11)

const int SS_TILE1_W = 160;
const int SS_TILE1_H = 90;

const int SS_TILE2_W = 32;
const int SS_TILE2_H = 18;

const int SS_LIGHT_LIST_MAX = 128;
