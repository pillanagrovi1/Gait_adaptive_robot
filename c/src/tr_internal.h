/* tr_internal.h - shared helpers between the agent core and the learners. */
#ifndef TR_INTERNAL_H
#define TR_INTERNAL_H

#include "tinyrl.h"
#include "tr_rng.h"

#define TR_ALIGN4(n) (((size_t)(n) + 3u) & ~(size_t)3u)

/* Map obs into u in [0, 1] (Q16.16), clipping to the configured bounds. */
void tr_normalise01(tr_agent_t *ag, const q16_t *obs, q16_t *u);

/* Per-learner hooks. Each learner implements these four. */
size_t   tr_tab_mem(const tr_cfg_t *c);
int      tr_tab_init(tr_agent_t *ag, uint8_t **cursor);
void     tr_tab_q(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target);
void     tr_tab_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y);

size_t   tr_lin_mem(const tr_cfg_t *c);
int      tr_lin_init(tr_agent_t *ag, uint8_t **cursor);
void     tr_lin_q(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target);
void     tr_lin_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y);

size_t   tr_mlp_mem(const tr_cfg_t *c);
int      tr_mlp_init(tr_agent_t *ag, uint8_t **cursor);
void     tr_mlp_q(tr_agent_t *ag, const q16_t *obs, q16_t *out, int use_target);
void     tr_mlp_update(tr_agent_t *ag, const q16_t *obs, int a, q16_t y);
void     tr_mlp_sync_target(tr_agent_t *ag);
uint32_t tr_mlp_nparams(const tr_cfg_t *c);

/* Clip the TD error if the config asks for it (any learner). */
static inline q16_t tr_clip_delta(const tr_cfg_t *c, q16_t d)
{
    if (c->td_clip > 0) return q16_clip(d, -c->td_clip, c->td_clip);
    return d;
}

static inline void *tr_carve(uint8_t **cursor, size_t bytes)
{
    void *p = *cursor;
    *cursor += TR_ALIGN4(bytes);
    return p;
}

#endif
