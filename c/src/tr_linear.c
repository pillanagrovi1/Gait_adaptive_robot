/*
 * tr_linear.c - linear Q-function on hashed tile-coding features.
 *
 * T tilings, each a grid of `tiles` cells per dimension, shifted by an
 * asymmetric offset (Sutton & Barto 2018, sec. 9.5.4: displacement vector
 * (1, 3, 5, ...) in units of tile_width / T). Each tiling activates exactly
 * one tile; its grid coordinates are hashed into one of M = `features`
 * weights per action. So
 *
 *     Q(s, a) = sum_{t=1..T} w[h_t(s), a]
 *     w[h_t(s), a] += (alpha / T) * delta     for every t
 *
 * Memory: M * n_actions * 4 bytes, independent of obs_dim. That is the
 * point of hashing: the feature count is a free memory knob.
 */
#include "tr_internal.h"

size_t tr_lin_mem(const tr_cfg_t *c)
{
    return TR_ALIGN4((size_t)c->features * c->n_actions * sizeof(q16_t))
         + TR_ALIGN4((size_t)c->tilings * sizeof(uint32_t));
}

int tr_lin_init(tr_agent_t *ag, uint8_t **cursor)
{
    const tr_cfg_t *c = &ag->cfg;
    ag->u.lin.w   = (q16_t *)tr_carve(cursor, (size_t)c->features * c->n_actions * sizeof(q16_t));
    ag->u.lin.idx = (uint32_t *)tr_carve(cursor, (size_t)c->tilings * sizeof(uint32_t));
    ag->u.lin.alpha_t = c->alpha / c->tilings;     /* integer division: exact enough */
    if (ag->u.lin.alpha_t == 0) return TR_ERR_CFG;
    return TR_OK;
}

/* Fill idx[0..T-1] with the active (hashed) feature of each tiling. */
static void active_features(tr_agent_t *ag, const q16_t *obs)
{
    const tr_cfg_t *c = &ag->cfg;
    const int T = c->tilings;
    q16_t *u = ag->xn;
    tr_normalise01(ag, obs, u);
    for (int t = 0; t < T; t++) {
        uint32_t h = 2166136261u ^ ((uint32_t)t * 0x9E3779B1u);
        for (int i = 0; i < c->obs_dim; i++) {
            /* offset in [0, 1) tile widths, as Q16.16 */
            int32_t off = (int32_t)((((int64_t)t * (2 * i + 1)) % T) * Q16_ONE / T);
            int64_t scaled = (int64_t)u[i] * c->tiles + off;   /* Q16.16, >= 0 */
            uint32_t coord = (uint32_t)(scaled >> 16);         /* 0 .. tiles */
            h ^= coord;
            h *= 16777619u;                                    /* FNV-1a prime */
        }
        ag->u.lin.idx[t] = h % (uint32_t)c->features;
    }
}

void tr_lin_q(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target)
{
    (void)use_target;
    const tr_cfg_t *c = &ag->cfg;
    const int A = c->n_actions;
    active_features(ag, obs);
    for (int k = 0; k < A; k++) {
        int64_t acc = 0;
        for (int t = 0; t < c->tilings; t++) acc += ag->u.lin.w[ag->u.lin.idx[t] * A + k];
        out[k] = q16_sat(acc);
    }
}

void tr_lin_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y)
{
    const tr_cfg_t *c = &ag->cfg;
    const int A = c->n_actions;
    active_features(ag, obs);
    int64_t acc = 0;
    for (int t = 0; t < c->tilings; t++) acc += ag->u.lin.w[ag->u.lin.idx[t] * A + a];
    q16_t delta = tr_clip_delta(c, q16_sub(y, q16_sat(acc)));
    q16_t step = q16_mul(ag->u.lin.alpha_t, delta);
    for (int t = 0; t < c->tilings; t++) {
        q16_t *w = &ag->u.lin.w[ag->u.lin.idx[t] * A + a];
        *w = q16_add(*w, step);
    }
}
