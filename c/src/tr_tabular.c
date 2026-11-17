/*
 * tr_tabular.c - Q-table over a uniform grid.
 *
 * cell(s) = sum_i bin_i(s_i) * bins^i,   bin_i = floor((s_i - lo_i) * bins / (hi_i - lo_i))
 * Memory: bins^obs_dim * n_actions * 4 bytes. Grows exponentially with obs_dim:
 * this is the curse of dimensionality the comparison is meant to expose.
 */
#include "tr_internal.h"

#define TAB_MAX_CELLS (1u << 22)

static uint32_t n_cells(const tr_cfg_t *c)
{
    uint64_t n = 1;
    for (int i = 0; i < c->obs_dim; i++) {
        n *= (uint64_t)c->bins;
        if (n > TAB_MAX_CELLS) return 0;
    }
    return (uint32_t)n;
}

size_t tr_tab_mem(const tr_cfg_t *c)
{
    uint32_t n = n_cells(c);
    if (n == 0) return 0;
    return TR_ALIGN4((size_t)n * c->n_actions * sizeof(q16_t));
}

int tr_tab_init(tr_agent_t *ag, uint8_t **cursor)
{
    const tr_cfg_t *c = &ag->cfg;
    ag->u.tab.n_cells = n_cells(c);
    ag->u.tab.table = (q16_t *)tr_carve(cursor, (size_t)ag->u.tab.n_cells * c->n_actions * sizeof(q16_t));
    /* table starts at 0 (arena was zeroed) */
    return TR_OK;
}

static uint32_t cell_of(const tr_agent_t *ag, const q16_t *obs)
{
    const tr_cfg_t *c = &ag->cfg;
    uint32_t cell = 0, stride = 1;
    for (int i = 0; i < c->obs_dim; i++) {
        q16_t x = q16_clip(obs[i], c->obs_lo[i], c->obs_hi[i]);
        int64_t num = ((int64_t)x - c->obs_lo[i]) * c->bins;
        int64_t den = (int64_t)c->obs_hi[i] - c->obs_lo[i];
        int32_t b = (int32_t)(num / den);          /* num >= 0, so this is floor */
        if (b >= c->bins) b = c->bins - 1;         /* x == hi lands in the last bin */
        cell += (uint32_t)b * stride;
        stride *= (uint32_t)c->bins;
    }
    return cell;
}

void tr_tab_q(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target)
{
    (void)use_target;                              /* no target table */
    const int A = ag->cfg.n_actions;
    const q16_t *row = &ag->u.tab.table[cell_of(ag, obs) * A];
    for (int k = 0; k < A; k++) out[k] = row[k];
}

void tr_tab_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y)
{
    const tr_cfg_t *c = &ag->cfg;
    q16_t *qa = &ag->u.tab.table[cell_of(ag, obs) * c->n_actions + a];
    q16_t delta = tr_clip_delta(c, q16_sub(y, *qa));
    *qa = q16_add(*qa, q16_mul(c->alpha, delta));
}
