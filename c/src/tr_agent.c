/*
 * tr_agent.c - the part every learner shares: config checks, arena layout,
 * normalisation, epsilon-greedy, the Q-learning target and the replay buffer.
 */
#include <string.h>
#include "tr_internal.h"

static size_t replay_mem(const tr_cfg_t *c)
{
    size_t n = (size_t)c->replay_cap;
    if (n == 0) return 0;
    if (c->replay_compact)
        return TR_ALIGN4(n * c->obs_dim * sizeof(uint16_t))    /* s as uint16 */
             + TR_ALIGN4(n)                                    /* action | flags */
             + TR_ALIGN4(n * sizeof(q16_t))                    /* r */
             + 3 * TR_ALIGN4(c->obs_dim * sizeof(q16_t));      /* last_s2, decode scratch x2 */
    return 2 * TR_ALIGN4(n * c->obs_dim * sizeof(q16_t))   /* s, s2 */
         + TR_ALIGN4(n * sizeof(q16_t))                    /* r */
         + 2 * TR_ALIGN4(n);                               /* a, done */
}

static int cfg_valid(const tr_cfg_t *c)
{
    if (!c) return 0;
    if (c->obs_dim < 1 || c->obs_dim > TR_MAX_OBS) return 0;
    if (c->n_actions < 2 || c->n_actions > TR_MAX_ACTIONS) return 0;
    for (int i = 0; i < c->obs_dim; i++)
        if (c->obs_hi[i] <= c->obs_lo[i]) return 0;
    if (c->gamma < 0 || c->gamma > Q16_ONE) return 0;
    if (c->alpha <= 0) return 0;
    if (c->replay_cap < 0 || c->replay_cap > 65535) return 0;
    if (c->replay_cap > 0 && (c->batch < 1 || c->batch > c->replay_cap)) return 0;
    if (c->lambda < 0 || c->lambda > Q16_ONE || c->obgd_kappa < 0) return 0;
    if (c->round_mode < 0 || c->round_mode > 1 || c->norm < 0 || c->norm > 1 || c->trace_cap < 0) return 0;
    if (c->norm != 0 && c->kind != TR_MLP) return 0;
    if (tr_stream_on(c) && c->replay_cap > 0) return 0;   /* traces need on-policy order */
    if (c->replay_compact < 0 || c->replay_compact > 1 || c->tile_dense < 0 || c->tile_dense > 1) return 0;
    if (c->replay_compact && c->n_actions > 63) return 0;
    if (c->cbp_period < 0 || c->cbp_maturity < 0) return 0;
    if (c->watkins < 0 || c->watkins > 1) return 0;
    if (c->cbp_period > 0 && c->kind != TR_MLP) return 0;
    if (c->tile_dense && c->kind != TR_LINEAR) return 0;
    switch (c->kind) {
    case TR_TABULAR:
        if (c->bins < 1) return 0;
        break;
    case TR_LINEAR:
        if (c->tilings < 1 || c->tilings > TR_MAX_TILINGS) return 0;
        if (c->tiles < 1 || c->features < 1) return 0;
        break;
    case TR_MLP:
        if (c->hidden < 1 || c->target_period < 0) return 0;
        break;
    default:
        return 0;
    }
    return 1;
}

static size_t learner_mem(const tr_cfg_t *c)
{
    switch (c->kind) {
    case TR_TABULAR: return tr_tab_mem(c);
    case TR_LINEAR:  return tr_lin_mem(c);
    case TR_MLP:     return tr_mlp_mem(c);
    }
    return 0;
}

size_t tr_mem_bytes(const tr_cfg_t *c)
{
    if (!cfg_valid(c)) return 0;
    size_t m = TR_ALIGN4(c->obs_dim * sizeof(q16_t))       /* xn */
             + TR_ALIGN4(c->n_actions * sizeof(q16_t))     /* q  */
             + replay_mem(c);
    size_t lm = learner_mem(c);
    if (lm == 0) return 0;                                 /* e.g. table too large */
    return m + lm;
}

size_t tr_agent_sizeof(void) { return sizeof(tr_agent_t); }
uint32_t tr_underflow_count(const tr_agent_t *ag) { return ag->n_underflow; }
uint32_t tr_update_terms(const tr_agent_t *ag) { return ag->n_upd_terms; }

uint32_t tr_saturation_count(const tr_agent_t *ag)
{
#ifndef TR_NO_DIAG
    return ag->n_saturated;
#else
    (void)ag;
    return 0;
#endif
}

uint32_t tr_param_count(const tr_cfg_t *c)
{
    if (!cfg_valid(c)) return 0;
    switch (c->kind) {
    case TR_TABULAR: return (uint32_t)(tr_tab_mem(c) / sizeof(q16_t));
    case TR_LINEAR:  return (uint32_t)c->features * (uint32_t)c->n_actions;
    case TR_MLP:     return tr_mlp_nparams(c);
    }
    return 0;
}

int tr_init(tr_agent_t *ag, const tr_cfg_t *cfg, void *arena, size_t arena_bytes, uint32_t seed)
{
    if (!ag || !cfg_valid(cfg)) return TR_ERR_CFG;
    size_t need = tr_mem_bytes(cfg);
    if (need == 0) return TR_ERR_CFG;
    if (!arena || arena_bytes < need) return TR_ERR_ARENA;

    memset(ag, 0, sizeof(*ag));
    ag->cfg = *cfg;
    ag->rng = tr_rng_seed(seed);
    for (int i = 0; i < cfg->obs_dim; i++) {
        int64_t range = (int64_t)cfg->obs_hi[i] - cfg->obs_lo[i];
        /* rounded, not truncated (decision D34): truncation biased every normalised value
         * down by up to 1 LSB, so points exactly on a tile edge fell into the tile below */
        ag->inv_range[i] = q16_sat(((1LL << 32) + range / 2) / range);
    }

    memset(arena, 0, need);
    uint8_t *cur = (uint8_t *)arena;
    ag->xn = (q16_t *)tr_carve(&cur, cfg->obs_dim * sizeof(q16_t));
    ag->q  = (q16_t *)tr_carve(&cur, cfg->n_actions * sizeof(q16_t));
    if (cfg->replay_cap > 0 && cfg->replay_compact) {
        size_t n = (size_t)cfg->replay_cap;
        ag->rb.su      = (uint16_t *)tr_carve(&cur, n * cfg->obs_dim * sizeof(uint16_t));
        ag->rb.af      = (uint8_t *)tr_carve(&cur, n);
        ag->rb.r       = (q16_t *)tr_carve(&cur, n * sizeof(q16_t));
        ag->rb.last_s2 = (q16_t *)tr_carve(&cur, cfg->obs_dim * sizeof(q16_t));
        ag->rb.ds      = (q16_t *)tr_carve(&cur, cfg->obs_dim * sizeof(q16_t));
        ag->rb.ds2     = (q16_t *)tr_carve(&cur, cfg->obs_dim * sizeof(q16_t));
    } else if (cfg->replay_cap > 0) {
        size_t n = (size_t)cfg->replay_cap;
        ag->rb.s    = (q16_t *)tr_carve(&cur, n * cfg->obs_dim * sizeof(q16_t));
        ag->rb.s2   = (q16_t *)tr_carve(&cur, n * cfg->obs_dim * sizeof(q16_t));
        ag->rb.r    = (q16_t *)tr_carve(&cur, n * sizeof(q16_t));
        ag->rb.a    = (uint8_t *)tr_carve(&cur, n);
        ag->rb.done = (uint8_t *)tr_carve(&cur, n);
    }
    int rc = TR_ERR_CFG;
    switch (cfg->kind) {
    case TR_TABULAR: rc = tr_tab_init(ag, &cur); break;
    case TR_LINEAR:  rc = tr_lin_init(ag, &cur); break;
    case TR_MLP:     rc = tr_mlp_init(ag, &cur); break;
    }
    ag->arena_used = (uint32_t)(cur - (uint8_t *)arena);
    return rc;
}

void tr_normalise01(tr_agent_t *ag, const q16_t *obs, q16_t *u)
{
    const tr_cfg_t *c = &ag->cfg;
    for (int i = 0; i < c->obs_dim; i++) {
        q16_t x = q16_clip(obs[i], c->obs_lo[i], c->obs_hi[i]);
        u[i] = q16_clip(q16_mul(q16_sub(x, c->obs_lo[i]), ag->inv_range[i]), 0, Q16_ONE);
    }
}

static void q_all(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target)
{
    switch (ag->cfg.kind) {
    case TR_TABULAR: tr_tab_q(ag, obs, out, use_target); break;
    case TR_LINEAR:  tr_lin_q(ag, obs, out, use_target); break;
    case TR_MLP:     tr_mlp_q(ag, obs, out, use_target); break;
    }
}

void tr_qvalues(tr_agent_t *ag, const q16_t *obs, q16_t *out)
{
    q_all(ag, obs, out, 0);
}

int tr_greedy(tr_agent_t *ag, const q16_t *obs)
{
    const int A = ag->cfg.n_actions;
    q_all(ag, obs, ag->q, 0);
    q16_t best = ag->q[0];
    int n_best = 1;
    for (int k = 1; k < A; k++) {
        if (ag->q[k] > best) { best = ag->q[k]; n_best = 1; }
        else if (ag->q[k] == best) n_best++;
    }
    if (n_best == 1) {
        for (int k = 0; k < A; k++) if (ag->q[k] == best) return k;
    }
    uint32_t pick = tr_rng_below(&ag->rng, (uint32_t)n_best);
    for (int k = 0; k < A; k++) {
        if (ag->q[k] == best) {
            if (pick == 0) return k;
            pick--;
        }
    }
    return 0; /* unreachable */
}

int tr_act(tr_agent_t *ag, const q16_t *obs, q16_t eps)
{
    if (tr_rng_unit_q16(&ag->rng) < eps)
        return (int)tr_rng_below(&ag->rng, (uint32_t)ag->cfg.n_actions);
    return tr_greedy(ag, obs);
}

/* The table's one-step write is checked from outside, so that tr_tabular.c holds only the
 * algorithm: a value left on the int32 limit counts as clipped (D55). Reads only; no RNG. */
static inline void count_table_saturation(tr_agent_t *ag, const q16_t *s, int a)
{
#ifndef TR_NO_DIAG
    tr_tab_q(ag, s, ag->q, 0);
    if (ag->q[a] == Q16_MAX || ag->q[a] == Q16_MIN) tr_mark_saturated(ag);
#else
    (void)ag; (void)s; (void)a;
#endif
}

/* One Q-learning update on one transition. Identical for every learner. */
static void learn_one(tr_agent_t *ag, const q16_t *s, int a, q16_t r, const q16_t *s2, int done)
{
    const tr_cfg_t *c = &ag->cfg;
    q16_t y = r;
    if (!done) {
        q_all(ag, s2, ag->q, 1);
        q16_t m = ag->q[0];
        for (int k = 1; k < c->n_actions; k++) if (ag->q[k] > m) m = ag->q[k];
        y = q16_add(r, q16_mul(c->gamma, m));
    }
    switch (c->kind) {
    case TR_TABULAR: tr_tab_update(ag, s, a, y); count_table_saturation(ag, s, a); break;
    case TR_LINEAR:  tr_lin_update(ag, s, a, y); break;
    case TR_MLP:     tr_mlp_update(ag, s, a, y); break;
    }
    ag->n_updates++;
    if (c->kind == TR_MLP && c->target_period > 0 &&
        (ag->n_updates % (uint32_t)c->target_period) == 0)
        tr_mlp_sync_target(ag);
}

int tr_is_stream(const tr_cfg_t *cfg) { return tr_stream_on(cfg); }

/*
 * Compressed replay (replay_compact = 1), after docs/research_review.md:
 *   - s stored as a uint16 fraction of [lo, hi]   (2 B per dimension instead of 4)
 *   - s' NOT stored: it is the next entry's s, when that entry continues the same
 *     episode (flag bit 7, set when the next transition's s equals this one's s')
 *   - action and flags packed in one byte
 * Per transition: 2*d + 1 + 4 bytes, e.g. 11 B for d = 3 instead of 30 B.
 * The decoded s differs from the original by at most range/65536 (lossy).
 */
#define AF_DONE 0x40u
#define AF_NEXT 0x80u
#define AF_ACT  0x3Fu

static void encode_obs(tr_agent_t *ag, const q16_t *obs, uint16_t *out)
{
    const tr_cfg_t *c = &ag->cfg;
    for (int i = 0; i < c->obs_dim; i++) {
        q16_t x = q16_clip(obs[i], c->obs_lo[i], c->obs_hi[i]);
        uint64_t num = (uint64_t)((int64_t)x - c->obs_lo[i]) << 16;
        uint64_t den = (uint64_t)((int64_t)c->obs_hi[i] - c->obs_lo[i]);
        uint64_t u = (num + den / 2) / den;
        out[i] = (uint16_t)(u > 65535u ? 65535u : u);
    }
}

static void decode_obs(tr_agent_t *ag, const uint16_t *in, q16_t *out)
{
    const tr_cfg_t *c = &ag->cfg;
    for (int i = 0; i < c->obs_dim; i++) {
        uint64_t range = (uint64_t)((int64_t)c->obs_hi[i] - c->obs_lo[i]);
        out[i] = (q16_t)((int64_t)c->obs_lo[i] + (int64_t)((in[i] * range + 32768u) >> 16));
    }
}

static int observe_compact(tr_agent_t *ag, const q16_t *s, int a, q16_t r, const q16_t *s2, int done)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_replay_t *rb = &ag->rb;
    const int d = c->obs_dim;
    const uint32_t cap = (uint32_t)c->replay_cap;
    uint32_t h = rb->head;
    if (rb->size > 0) {
        uint32_t prev = (h + cap - 1u) % cap;
        if (!(rb->af[prev] & AF_DONE) && memcmp(rb->last_s2, s, (size_t)d * sizeof(q16_t)) == 0)
            rb->af[prev] |= AF_NEXT;
    }
    encode_obs(ag, s, &rb->su[h * d]);
    rb->af[h] = (uint8_t)((uint32_t)a & AF_ACT) | (done ? AF_DONE : 0u);
    rb->r[h] = r;
    memcpy(rb->last_s2, s2, (size_t)d * sizeof(q16_t));
    rb->head = (h + 1u) % cap;
    if (rb->size < cap) rb->size++;
    if (rb->size < (uint32_t)c->batch) return 0;   /* warm-up */
    int n = 0;
    for (int k = 0; k < c->batch; k++) {
        /* rejection-sample an entry whose next state is known (or that is terminal) */
        for (int tries = 0; tries < 16; tries++) {
            uint32_t j = tr_rng_below(&ag->rng, rb->size);
            uint8_t f = rb->af[j];
            if (!(f & (AF_DONE | AF_NEXT))) continue;
            decode_obs(ag, &rb->su[j * d], rb->ds);
            if (f & AF_NEXT) decode_obs(ag, &rb->su[((j + 1u) % cap) * d], rb->ds2);
            else memcpy(rb->ds2, rb->ds, (size_t)d * sizeof(q16_t));   /* terminal: s' unused */
            learn_one(ag, rb->ds, (int)(f & AF_ACT), rb->r[j], rb->ds2, (f & AF_DONE) != 0);
            n++;
            break;
        }
    }
    return n;
}

void tr_episode_start(tr_agent_t *ag)
{
    if (!tr_stream_on(&ag->cfg)) return;
    switch (ag->cfg.kind) {
    case TR_TABULAR: tr_tab_clear(ag); break;
    case TR_LINEAR:  tr_lin_clear(ag); break;
    case TR_MLP:     tr_mlp_clear(ag); break;
    }
}

/*
 * Streaming update: Q(lambda) WITHOUT the Watkins cut, with an ObGD step-size cap
 * (Elsayed, Vasan & Mahmood 2024, "Streaming Deep RL Finally Works").
 *
 *   y      = r + gamma * max_a' Q_target(s', a')      (r if terminal): the SAME target
 *   e      <- gamma*lambda * e + grad Q(s, a)          watkins = 1: e <- grad Q(s, a) if Q(s, a) < max Q(s, .)
 *                                                     watkins = 0 (every archived run): never cut (D68)
 *   delta  = y - Q(s, a)
 *   ObGD:  dbar = max(|delta|, 1);  alpha_eff = alpha * min(1, 1 / (kappa * dbar * ||e||_1))
 *   w      <- w + alpha_eff * delta * e
 *   e      <- 0 at terminal states and at episode start (tr_episode_start)
 *
 * With lambda = 0 and kappa = 0 this is exactly the one-step update.
 */
static void learn_stream(tr_agent_t *ag, const q16_t *s, int a, q16_t r, const q16_t *s2, int done)
{
    const tr_cfg_t *c = &ag->cfg;
    q16_t y = r;
    if (!done) {
        q_all(ag, s2, ag->q, 1);
        q16_t m = ag->q[0];
        for (int k = 1; k < c->n_actions; k++) if (ag->q[k] > m) m = ag->q[k];
        y = q16_add(r, q16_mul(c->gamma, m));
    }
    q16_t gl = q16_mul(c->gamma, c->lambda);
    if (c->watkins && gl != 0) {
        /* Watkins cut: a non-greedy a (strictly below the max; ties count as greedy, no RNG) starts a new trace.
         * Same as the authors' reset after the update when the next action is non-greedy (D68). */
        q_all(ag, s, ag->q, 0);
        q16_t m = ag->q[0];
        for (int k = 1; k < c->n_actions; k++) if (ag->q[k] > m) m = ag->q[k];
        if (ag->q[a] < m) gl = 0;
    }
    int64_t n1 = 0;
    q16_t qsa = 0;
    switch (c->kind) {
    case TR_TABULAR: qsa = tr_tab_trace(ag, s, a, gl, &n1); break;
    case TR_LINEAR:  qsa = tr_lin_trace(ag, s, a, gl, &n1); break;
    case TR_MLP:     qsa = tr_mlp_trace(ag, s, a, gl, &n1); break;
    }
    q16_t delta = tr_clip_delta(c, q16_sub(y, qsa));
    q16_t step = q16_mul(c->alpha, delta);
#ifndef TR_NO_DIAG
    int64_t unclipped = ((int64_t)c->alpha * delta + (1LL << 15)) >> 16;
    ag->step_saturated = (step != unclipped);       /* then every term made from it is clipped */
#endif
    int sh = 16;                      /* step is Q16.16 ... */
    if (c->obgd_kappa > 0) {
        q16_t ad = delta < 0 ? -delta : delta;
        q16_t dbar = ad > Q16_ONE ? ad : Q16_ONE;
        int64_t denom = ((int64_t)q16_mul(c->obgd_kappa, dbar) * n1) >> 16;   /* Q16.16 */
        if (denom > Q16_ONE) {
            /* ... unless ObGD shrinks it: then keep 14 extra fraction bits (Q2.30), so a
             * step of e.g. 1e-4 is not rounded to 6 LSBs (or 0) before it multiplies e.
             * One division per update, never per weight. */
            int64_t step30 = ((int64_t)q16_mul(c->alpha, delta) * (1LL << 30)) / denom;
            if (step30 > -(1LL << 31) && step30 < (1LL << 31)) { step = (q16_t)step30; sh = 30; }
            else step = q16_sat(((int64_t)step * 65536) / denom);
        }
    }
    if (step != 0) {
        switch (c->kind) {
        case TR_TABULAR: tr_tab_apply(ag, step, sh); break;
        case TR_LINEAR:  tr_lin_apply(ag, step, sh); break;
        case TR_MLP:     tr_mlp_apply(ag, step, sh); break;
        }
    }
    if (done) tr_episode_start(ag);
    ag->n_updates++;
    if (c->kind == TR_MLP && c->target_period > 0 &&
        (ag->n_updates % (uint32_t)c->target_period) == 0)
        tr_mlp_sync_target(ag);
}

int tr_observe(tr_agent_t *ag, const q16_t *s, int a, q16_t r, const q16_t *s2, int done)
{
    const tr_cfg_t *c = &ag->cfg;
    if (tr_stream_on(c)) {
        learn_stream(ag, s, a, r, s2, done);
        return 1;
    }
    if (c->replay_cap == 0) {
        learn_one(ag, s, a, r, s2, done);
        return 1;
    }
    if (c->replay_compact) return observe_compact(ag, s, a, r, s2, done);
    tr_replay_t *rb = &ag->rb;
    const int d = c->obs_dim;
    uint32_t h = rb->head;
    memcpy(&rb->s[h * d], s, d * sizeof(q16_t));
    memcpy(&rb->s2[h * d], s2, d * sizeof(q16_t));
    rb->r[h] = r;
    rb->a[h] = (uint8_t)a;
    rb->done[h] = (uint8_t)(done != 0);
    rb->head = (h + 1) % (uint32_t)c->replay_cap;
    if (rb->size < (uint32_t)c->replay_cap) rb->size++;
    if (rb->size < (uint32_t)c->batch) return 0;   /* warm-up */
    for (int k = 0; k < c->batch; k++) {
        uint32_t j = tr_rng_below(&ag->rng, rb->size);
        learn_one(ag, &rb->s[j * d], rb->a[j], rb->r[j], &rb->s2[j * d], rb->done[j]);
    }
    return c->batch;
}
