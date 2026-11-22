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
 *
 * tile_dense: 0 = hashed: FNV-1a + murmur3 fmix32 finaliser (DEFAULT since decision D37:
 *                 at load 0.63 shared slots drop from 64% to 51%, the random-hash level)
 *             1 = collision-free dense indices
 *             2 = legacy plain FNV-1a (kept only for the tile_collisions comparison)
 *
 * Speed (17 Nov 2026 revision, same results bit for bit):
 *   - tiling offsets precomputed once (no 64-bit modulo/division per step)
 *   - 32-bit arithmetic for tile coordinates (needs tiles < 32768)
 *   - power-of-two M uses a mask instead of a modulo
 *   - a 2-slot cache of active features: one environment step queries
 *     act(s), target(s'), update(s) and next act(s'), but only hashes twice
 */
#include <string.h>
#include "tr_internal.h"

static uint32_t trace_cap(const tr_cfg_t *c)
{
    return c->trace_cap > 0 ? (uint32_t)c->trace_cap : 8u * (uint32_t)c->tilings;
}

/* Dense (unhashed) layout: tiling t, coordinates c_i in 0..tiles  ->
 *   index = t * (tiles+1)^d + sum_i c_i * (tiles+1)^i     (collision-free)
 * Needs features >= tilings * (tiles+1)^d. Returns 0 if it does not fit. */
static uint32_t dense_cells(const tr_cfg_t *c)
{
    uint64_t n = (uint64_t)c->tilings;
    for (int i = 0; i < c->obs_dim; i++) {
        n *= (uint64_t)(c->tiles + 1);
        if (n > (uint64_t)c->features) return 0;
    }
    return (uint32_t)n;
}

size_t tr_lin_mem(const tr_cfg_t *c)
{
    if (c->tiles >= 32768) return 0;
    if (c->tile_dense == 1 && dense_cells(c) == 0) return 0;
    size_t m = TR_ALIGN4((size_t)c->features * c->n_actions * sizeof(q16_t))
             + TR_ALIGN4((size_t)c->tilings * c->obs_dim * sizeof(int32_t))          /* offsets */
             + 2 * (TR_ALIGN4((size_t)c->obs_dim * sizeof(q16_t))                    /* cache */
                  + TR_ALIGN4((size_t)c->tilings * sizeof(uint32_t)));
    if (tr_stream_on(c)) m += tr_strace_mem(trace_cap(c));
    return m;
}

int tr_lin_init(tr_agent_t *ag, uint8_t **cursor)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_lin_t *L = &ag->u.lin;
    const int T = c->tilings, d = c->obs_dim;
    L->w   = (q16_t *)tr_carve(cursor, (size_t)c->features * c->n_actions * sizeof(q16_t));
    L->off = (int32_t *)tr_carve(cursor, (size_t)T * d * sizeof(int32_t));
    for (int s = 0; s < 2; s++) {
        L->cache.obs[s] = (q16_t *)tr_carve(cursor, (size_t)d * sizeof(q16_t));
        L->cache.idx[s] = (uint32_t *)tr_carve(cursor, (size_t)T * sizeof(uint32_t));
        L->cache.valid[s] = 0;
    }
    L->cache.next = 0;
    L->idx = L->cache.idx[0];
    if (tr_stream_on(c)) tr_strace_init(&L->tr, cursor, trace_cap(c));
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)   /* offset in [0, 1) tile widths, Q16.16 */
            L->off[t * d + i] = (int32_t)((((int64_t)t * (2 * i + 1)) % T) * Q16_ONE / T);
    L->mask = ((c->features & (c->features - 1)) == 0) ? (uint32_t)c->features - 1u : 0u;
    L->alpha_t = c->alpha / c->tilings;     /* integer division: exact enough */
    if (L->alpha_t == 0) return TR_ERR_CFG;
    return TR_OK;
}

/* Point L->idx at the T active (hashed) features of obs, using the cache. */
static const uint32_t *active_features(tr_agent_t *ag, const q16_t *obs)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_lin_t *L = &ag->u.lin;
    const int T = c->tilings, d = c->obs_dim;
    for (int s = 0; s < 2; s++) {
        if (L->cache.valid[s] && memcmp(L->cache.obs[s], obs, (size_t)d * sizeof(q16_t)) == 0) {
            L->idx = L->cache.idx[s];
            L->cache.next = (uint8_t)(1 - s);   /* least recently used goes next; see tr_lin_touch (D62) */
            return L->idx;
        }
    }
    int s = L->cache.next;
    L->cache.next ^= 1u;
    memcpy(L->cache.obs[s], obs, (size_t)d * sizeof(q16_t));
    L->cache.valid[s] = 1;
    uint32_t *idx = L->cache.idx[s];
    q16_t *u = ag->xn;
    tr_normalise01(ag, obs, u);
    const uint32_t side = (uint32_t)c->tiles + 1u;
    uint32_t per_tiling = 1u;
    if (c->tile_dense == 1) for (int i = 0; i < d; i++) per_tiling *= side;
    for (int t = 0; t < T; t++) {
        uint32_t h = 2166136261u ^ ((uint32_t)t * 0x9E3779B1u);
        uint32_t dense = (uint32_t)t * per_tiling, stride = 1u;
        const int32_t *off = &L->off[t * d];
        for (int i = 0; i < d; i++) {
            uint32_t scaled = (uint32_t)u[i] * (uint32_t)c->tiles + (uint32_t)off[i];  /* Q16.16 */
            uint32_t coord = scaled >> 16;                                           /* 0 .. tiles */
            h ^= coord;
            h *= 16777619u;                                                          /* FNV-1a prime */
            dense += coord * stride;
            stride *= side;
        }
        if (c->tile_dense == 1) { idx[t] = dense; continue; }
        if (c->tile_dense == 0) {            /* murmur3 fmix32: FNV alone mixes small integers poorly */
            h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
        }                                    /* tile_dense == 2: legacy plain FNV-1a (comparison only) */
        idx[t] = L->mask ? (h & L->mask) : (h % (uint32_t)c->features);
    }
    L->idx = idx;
    return idx;
}

/* Weights changed: cached indices are still valid (they depend on obs only). */

void tr_lin_q(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target)
{
    (void)use_target;
    const tr_cfg_t *c = &ag->cfg;
    const int A = c->n_actions;
    const uint32_t *idx = active_features(ag, obs);
    for (int k = 0; k < A; k++) {
        int64_t acc = 0;
        for (int t = 0; t < c->tilings; t++) acc += ag->u.lin.w[idx[t] * A + k];
        out[k] = q16_sat(acc);
    }
}

/* Called with s before the target looks up s', so that s' never evicts s: a random action did not
 * look at s, so least-recently-used eviction alone would (D62). Cache only; no value changes. */
void tr_lin_touch(tr_agent_t *ag, const q16_t *obs) { (void)active_features(ag, obs); }

void tr_lin_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y)
{
    const tr_cfg_t *c = &ag->cfg;
    const int A = c->n_actions;
    const uint32_t *idx = active_features(ag, obs);
    int64_t acc = 0;
    for (int t = 0; t < c->tilings; t++) acc += ag->u.lin.w[idx[t] * A + a];
    q16_t delta = tr_clip_delta(c, q16_sub(y, q16_sat(acc)));
    q16_t step = tr_mul_upd(ag, ag->u.lin.alpha_t, delta);
    for (int t = 0; t < c->tilings; t++) {
        q16_t *w = &ag->u.lin.w[idx[t] * A + a];
        *w = tr_add_upd(ag, *w, step);
    }
}

/* ---- streaming path ---- */

q16_t tr_lin_trace(tr_agent_t *ag, const q16_t *obs, int a, q16_t decay, int64_t *norm1)
{
    const tr_cfg_t *c = &ag->cfg;
    const int A = c->n_actions;
    tr_lin_q(ag, obs, ag->q, 0);
    const uint32_t *idx = ag->u.lin.idx;
    tr_strace_t *t = &ag->u.lin.tr;
    if (decay == 0) tr_strace_clear(t); else tr_strace_decay(t, decay);
    for (int k = 0; k < c->tilings; k++) tr_strace_add(t, idx[k] * A + (uint32_t)a, Q16_ONE);
    *norm1 = tr_strace_norm1(t);
    return ag->q[a];
}

void tr_lin_apply(tr_agent_t *ag, q16_t step, int sh)
{
    /* Without ObGD the step must be divided by T, as in the one-step update. */
    if (ag->cfg.obgd_kappa == 0) step /= ag->cfg.tilings;
    tr_strace_apply(ag, &ag->u.lin.tr, ag->u.lin.w, step, sh);
}

void tr_lin_clear(tr_agent_t *ag) { tr_strace_clear(&ag->u.lin.tr); }
