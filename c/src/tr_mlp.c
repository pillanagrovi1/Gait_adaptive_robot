/*
 * tr_mlp.c - tiny Q-network: obs_dim -> hidden (ReLU) -> n_actions (linear).
 *
 * Forward:   x = 2u - 1 in [-1, 1]
 *            z_j = sum_i W1[j,i] x_i + b1_j,   h_j = max(0, z_j)
 *            Q_k = sum_j W2[k,j] h_j + b2_k
 * Update (semi-gradient Q-learning on the taken action a only):
 *            delta = clip(y - Q_a)
 *            g_j   = delta * W2[a,j] * [z_j > 0]        (uses W2 BEFORE its update)
 *            W2[a,j] += alpha * delta * h_j,  b2_a += alpha * delta
 *            W1[j,i] += alpha * g_j * x_i,    b1_j += alpha * g_j
 *
 * Parameter layout in p[]: W1 (H*d, row j = unit j), b1 (H), W2 (A*H), b2 (A).
 * Memory: n_params*4 (online) + n_params*4 (target, optional) + 2*H*4 scratch.
 */
#include <string.h>
#include "tr_internal.h"

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
    m += 2 * TR_ALIGN4((size_t)c->hidden * sizeof(q16_t));
    return m;
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
    m->z  = (q16_t *)tr_carve(cursor, H * sizeof(q16_t));
    m->h  = (q16_t *)tr_carve(cursor, H * sizeof(q16_t));

    /* Uniform fan-in init: W ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)), biases 0. */
    q16_t s1 = (q16_t)(((int64_t)Q16_ONE << 16) / q16_sqrt(d * Q16_ONE));
    q16_t s2 = (q16_t)(((int64_t)Q16_ONE << 16) / q16_sqrt(H * Q16_ONE));
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
        q16_t z = q16_add(q16_from_acc(acc), b1[j]);
        m->z[j] = z;
        m->h[j] = z > 0 ? z : 0;
    }
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

void tr_mlp_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y)
{
    const tr_cfg_t *c = &ag->cfg;
    tr_mlp_t *m = &ag->u.mlp;
    const int d = c->obs_dim, H = c->hidden, A = c->n_actions;
    q16_t *W1 = m->p, *b1 = m->p + H * d, *W2 = b1 + H, *b2 = W2 + A * H;

    forward(ag, m->p, obs, ag->q);
    q16_t delta = tr_clip_delta(c, q16_sub(y, ag->q[a]));
    if (delta == 0) return;
    const q16_t lr = c->alpha;
    q16_t *w2a = &W2[a * H];

    /* hidden error signals, stored in z (z is not needed after this) */
    for (int j = 0; j < H; j++)
        m->z[j] = (m->z[j] > 0) ? q16_mul(delta, w2a[j]) : 0;

    for (int j = 0; j < H; j++) w2a[j] = q16_add(w2a[j], q16_mul3(lr, delta, m->h[j]));
    b2[a] = q16_add(b2[a], q16_mul(lr, delta));

    const q16_t *x = ag->xn;
    for (int j = 0; j < H; j++) {
        q16_t g = m->z[j];
        if (g == 0) continue;
        q16_t *row = &W1[j * d];
        for (int i = 0; i < d; i++) row[i] = q16_add(row[i], q16_mul3(lr, g, x[i]));
        b1[j] = q16_add(b1[j], q16_mul(lr, g));
    }
    (void)A;
}
