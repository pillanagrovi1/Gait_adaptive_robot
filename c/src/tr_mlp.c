/*
 * tr_mlp.c - tiny Q-network: obs_dim -> hidden (ReLU) -> n_actions (linear).
 *
 * Forward:   x = 2u - 1 in [-1, 1]
 *            z_j = sum_i W1[j,i] x_i + b1_j
 *            (norm = 1: z <- (z - mean z) / sqrt(var z + eps), parameter-free LayerNorm)
 *            h_j = max(0, z_j)
 *            Q_k = sum_j W2[k,j] h_j + b2_k
 * Update (semi-gradient Q-learning on the taken action a only):
 *            delta = clip(y - Q_a)
 *            g_j   = dQ_a/dz_j  (= W2[a,j] * [z_j > 0] without LayerNorm; uses W2 BEFORE its update)
 *            W2[a,j] += alpha * delta * h_j,  b2_a += alpha * delta
 *            W1[j,i] += alpha * delta * g_j * x_i,    b1_j += alpha * delta * g_j
 * Streaming path (lambda/ObGD): the same gradient goes into a dense eligibility
 * trace e (one value per parameter) instead of straight into the weights.
 *
 * Parameter layout in p[]: W1 (H*d, row j = unit j), b1 (H), W2 (A*H), b2 (A).
 * Memory: params*4 (+ target params*4) (+ trace params*4) + 3*H*4 scratch.
 *
 * Continual backprop (Dohare et al., Nature 2024), cbp_period > 0: every hidden unit
 * keeps a running utility u_j = 0.99 u_j + 0.01 |h_j| sum_k |W2[k,j]|; every cbp_period
 * updates the least useful unit older than cbp_maturity is re-initialised (fresh input
 * weights, zero output weights, so Q does not jump). Restores plasticity after a
 * dynamics change. Memory: 8 bytes per hidden unit.
 *
 * LayerNorm as a target-network replacement: Gallici et al. 2025 (PQN) and
 * Elsayed et al. 2024 (Stream Q(lambda)). Their nets are >= 128 wide; whether it
 * helps at 8-64 units is an open question this project can answer.
 */
#include <string.h>
#include "tr_internal.h"

#define LN_EPS 66   /* ~1e-3 in Q16.16 */

uint32_t tr_mlp_nparams(const tr_cfg_t *c)
{
    uint32_t d = (uint32_t)c->obs_dim, H = (uint32_t)c->hidden, A = (uint32_t)c->n_actions;
    return H * d + H + A * H + A;
}

size_t tr_mlp_mem(const tr_cfg_t *c)
{
    size_t np = tr_mlp_nparams(c);
    size_t m = TR_ALIGN4(np * sizeof(q16_t));
    if (c->target_period > 0) m += TR_ALIGN4(np * sizeof(q16_t));
    if (tr_stream_on(c)) m += TR_ALIGN4(np * sizeof(q16_t));
    m += 3 * TR_ALIGN4((size_t)c->hidden * sizeof(q16_t));
    if (c->cbp_period > 0) m += 2 * TR_ALIGN4((size_t)c->hidden * sizeof(uint32_t));
    return m;
}

static q16_t init_scale(int fan_in)
{
    return (q16_t)(((int64_t)Q16_ONE << 16) / q16_sqrt(fan_in * Q16_ONE));
}

/* uniform in [-s, s) */
static q16_t uniform_sym(uint32_t *rng, q16_t s)
{
    int32_t v = (int32_t)(tr_rng_next(rng) >> 16) - 32768;   /* [-0.5, 0.5) in Q16.16 */
    return q16_mul(s, v * 2);
}

int tr_mlp_init(tr_agent_t *ag, uint8_t **cursor)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_mlp_t *m = &ag->u.mlp;
    const int d = c->obs_dim, H = c->hidden, A = c->n_actions;
    m->n_params = tr_mlp_nparams(c);
    m->p  = (q16_t *)tr_carve(cursor, m->n_params * sizeof(q16_t));
    m->tp = c->target_period > 0 ? (q16_t *)tr_carve(cursor, m->n_params * sizeof(q16_t)) : NULL;
    m->e  = tr_stream_on(c) ? (q16_t *)tr_carve(cursor, m->n_params * sizeof(q16_t)) : NULL;
    m->z  = (q16_t *)tr_carve(cursor, H * sizeof(q16_t));
    m->h  = (q16_t *)tr_carve(cursor, H * sizeof(q16_t));
    m->g  = (q16_t *)tr_carve(cursor, H * sizeof(q16_t));
    m->util = c->cbp_period > 0 ? (q16_t *)tr_carve(cursor, H * sizeof(q16_t)) : NULL;
    m->age  = c->cbp_period > 0 ? (uint32_t *)tr_carve(cursor, H * sizeof(uint32_t)) : NULL;
    m->inv_sigma = Q16_ONE;

    /* Uniform fan-in init: W ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)), biases 0. */
    q16_t s1 = init_scale(d), s2 = init_scale(H);
    q16_t *W1 = m->p, *W2 = m->p + H * d + H;
    for (int k = 0; k < H * d; k++) W1[k] = uniform_sym(&ag->rng, s1);
    for (int k = 0; k < A * H; k++) W2[k] = uniform_sym(&ag->rng, s2);
    if (m->tp) memcpy(m->tp, m->p, m->n_params * sizeof(q16_t));
    return TR_OK;
}

void tr_mlp_sync_target(tr_agent_t *ag)
{
    tr_mlp_t *m = &ag->u.mlp;
    if (m->tp) memcpy(m->tp, m->p, m->n_params * sizeof(q16_t));
}

/* Parameter-free LayerNorm over the H pre-activations in z (in place). */
static void layer_norm(tr_mlp_t *m, int H)
{
    int64_t sum = 0;
    for (int j = 0; j < H; j++) sum += m->z[j];
    q16_t mu = (q16_t)(sum / H);
    int64_t ss = 0;
    for (int j = 0; j < H; j++) {
        int64_t dz = (int64_t)m->z[j] - mu;
        ss += dz * dz;                                  /* Q32.32 */
    }
    q16_t var = q16_sat((ss / H) >> 16);
    q16_t sigma = q16_sqrt(q16_add(var, LN_EPS));
    m->inv_sigma = q16_sat(((int64_t)1 << 32) / (sigma > 0 ? sigma : 1));
    for (int j = 0; j < H; j++) m->z[j] = q16_mul(q16_sub(m->z[j], mu), m->inv_sigma);
}

/* Forward pass with parameter set P. Leaves z, h (scratch) and out filled. */
static void forward(tr_agent_t *ag, const q16_t *P, const q16_t *obs, q16_t *out)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_mlp_t *m = &ag->u.mlp;
    const int d = c->obs_dim, H = c->hidden, A = c->n_actions;
    const q16_t *W1 = P, *b1 = P + H * d, *W2 = b1 + H, *b2 = W2 + A * H;
    q16_t *x = ag->xn;

    tr_normalise01(ag, obs, x);
    for (int i = 0; i < d; i++) x[i] = 2 * x[i] - Q16_ONE;      /* [0,1] -> [-1,1] */

    for (int j = 0; j < H; j++) {
        int64_t acc = 0;
        const q16_t *row = &W1[j * d];
        for (int i = 0; i < d; i++) acc += (int64_t)row[i] * x[i];
        m->z[j] = q16_add(q16_from_acc(acc), b1[j]);
    }
    if (c->norm == 1) layer_norm(m, H);
    for (int j = 0; j < H; j++) m->h[j] = m->z[j] > 0 ? m->z[j] : 0;
    for (int k = 0; k < A; k++) {
        int64_t acc = 0;
        const q16_t *row = &W2[k * H];
        for (int j = 0; j < H; j++) acc += (int64_t)row[j] * m->h[j];
        out[k] = q16_add(q16_from_acc(acc), b2[k]);
    }
}

void tr_mlp_q(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target)
{
    const q16_t *P = (use_target && ag->u.mlp.tp) ? ag->u.mlp.tp : ag->u.mlp.p;
    forward(ag, P, obs, out);
}

/* g_j = dQ_a / d(pre-norm z_j), after a forward pass on the online params. */
static void hidden_grad(tr_agent_t *ag, int a)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_mlp_t *m = &ag->u.mlp;
    const int d = c->obs_dim, H = c->hidden;
    const q16_t *w2a = m->p + H * d + H + a * H;
    for (int j = 0; j < H; j++) m->g[j] = m->z[j] > 0 ? w2a[j] : 0;   /* dQ/d(normalised z) */
    if (c->norm != 1) return;
    /* LayerNorm backward: dz = (1/sigma) * (g - mean(g) - zn * mean(g * zn)) */
    int64_t sg = 0, sgz = 0;
    for (int j = 0; j < H; j++) { sg += m->g[j]; sgz += (int64_t)m->g[j] * m->z[j]; }
    q16_t mg = (q16_t)(sg / H);
    q16_t mgz = q16_sat(((sgz / H) + (1LL << 15)) >> 16);
    for (int j = 0; j < H; j++) {
        q16_t t = q16_sub(q16_sub(m->g[j], mg), q16_mul(m->z[j], mgz));
        m->g[j] = q16_mul(t, m->inv_sigma);
    }
}

/* Continual backprop bookkeeping; call after each update, with h from the forward on s. */
static void cbp_step(tr_agent_t *ag)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_mlp_t *m = &ag->u.mlp;
    if (!m->util) return;
    const int d = c->obs_dim, H = c->hidden, A = c->n_actions;
    q16_t *W1 = m->p, *b1 = m->p + H * d, *W2 = b1 + H;
    const q16_t keep = 64881, gain = Q16_ONE - 64881;          /* 0.99 / 0.01 */
    for (int j = 0; j < H; j++) {
        int64_t out = 0;
        for (int k = 0; k < A; k++) out += W2[k * H + j] < 0 ? -(int64_t)W2[k * H + j] : W2[k * H + j];
        q16_t contrib = q16_mul(m->h[j], q16_sat(out));
        m->util[j] = q16_add(q16_mul(m->util[j], keep), q16_mul(contrib, gain));
        if (m->age[j] < UINT32_MAX) m->age[j]++;
    }
    if ((ag->n_updates + 1u) % (uint32_t)c->cbp_period != 0) return;
    int worst = -1;
    for (int j = 0; j < H; j++)
        if (m->age[j] >= (uint32_t)c->cbp_maturity && (worst < 0 || m->util[j] < m->util[worst])) worst = j;
    if (worst < 0) return;
    q16_t s1 = init_scale(d);
    for (int i = 0; i < d; i++) W1[worst * d + i] = uniform_sym(&ag->rng, s1);
    b1[worst] = 0;
    for (int k = 0; k < A; k++) W2[k * H + worst] = 0;
    m->util[worst] = 0;
    m->age[worst] = 0;
    if (m->e) {                                                  /* forget its trace too */
        q16_t *eW1 = m->e, *eb1 = m->e + H * d, *eW2 = eb1 + H;
        for (int i = 0; i < d; i++) eW1[worst * d + i] = 0;
        eb1[worst] = 0;
        for (int k = 0; k < A; k++) eW2[k * H + worst] = 0;
    }
}

void tr_mlp_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_mlp_t *m = &ag->u.mlp;
    const int d = c->obs_dim, H = c->hidden, A = c->n_actions;
    q16_t *W1 = m->p, *b1 = m->p + H * d, *W2 = b1 + H, *b2 = W2 + A * H;

    forward(ag, m->p, obs, ag->q);
    q16_t delta = tr_clip_delta(c, q16_sub(y, ag->q[a]));
    if (delta == 0) { cbp_step(ag); return; }
    const q16_t lr = c->alpha;
    q16_t *w2a = &W2[a * H];
    const q16_t *x = ag->xn;

    if (c->norm == 0) {
        /* original path, kept bit for bit: hidden error signals stored in z */
        for (int j = 0; j < H; j++)
            m->z[j] = (m->z[j] > 0) ? q16_mul(delta, w2a[j]) : 0;
    } else {
        hidden_grad(ag, a);
        for (int j = 0; j < H; j++) m->z[j] = q16_mul(delta, m->g[j]);
    }

    for (int j = 0; j < H; j++) w2a[j] = tr_add_upd(ag, w2a[j], tr_mul3_upd(ag, lr, delta, m->h[j]));
    b2[a] = tr_add_upd(ag, b2[a], tr_mul_upd(ag, lr, delta));

    for (int j = 0; j < H; j++) {
        q16_t g = m->z[j];
        if (g == 0) continue;
        q16_t *row = &W1[j * d];
        for (int i = 0; i < d; i++) row[i] = tr_add_upd(ag, row[i], tr_mul3_upd(ag, lr, g, x[i]));
        b1[j] = tr_add_upd(ag, b1[j], tr_mul_upd(ag, lr, g));
    }
    /* hidden error signals overwrote z; h (activations of s) is intact for CBP */
    cbp_step(ag);
}

/* ---- streaming path ---- */

q16_t tr_mlp_trace(tr_agent_t *ag, const q16_t *obs, int a, q16_t decay, int64_t *norm1)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_mlp_t *m = &ag->u.mlp;
    const int d = c->obs_dim, H = c->hidden, A = c->n_actions;
    forward(ag, m->p, obs, ag->q);
    hidden_grad(ag, a);
    q16_t *e = m->e;
    q16_t *eW1 = e, *eb1 = e + H * d, *eW2 = eb1 + H, *eb2 = eW2 + A * H;
    const uint32_t np = m->n_params;
    if (decay == 0) memset(e, 0, np * sizeof(q16_t));
    else for (uint32_t k = 0; k < np; k++) e[k] = q16_mul(e[k], decay);
    const q16_t *x = ag->xn;
    for (int j = 0; j < H; j++) {
        q16_t g = m->g[j];
        if (g != 0) {
            for (int i = 0; i < d; i++) eW1[j * d + i] = q16_add(eW1[j * d + i], q16_mul(g, x[i]));
            eb1[j] = q16_add(eb1[j], g);
        }
        eW2[a * H + j] = q16_add(eW2[a * H + j], m->h[j]);
    }
    eb2[a] = q16_add(eb2[a], Q16_ONE);
    int64_t n1 = 0;
    for (uint32_t k = 0; k < np; k++) n1 += e[k] < 0 ? -(int64_t)e[k] : e[k];
    *norm1 = n1;
    return ag->q[a];
}

void tr_mlp_apply(tr_agent_t *ag, q16_t step, int sh)
{
    tr_mlp_t *m = &ag->u.mlp;
    for (uint32_t k = 0; k < m->n_params; k++)
        if (m->e[k] != 0) m->p[k] = tr_add_upd(ag, m->p[k], tr_mul_upd_sh(ag, step, m->e[k], sh));
    cbp_step(ag);
}

void tr_mlp_clear(tr_agent_t *ag)
{
    tr_mlp_t *m = &ag->u.mlp;
    if (m->e) memset(m->e, 0, m->n_params * sizeof(q16_t));
}
