/*
 * tinyrl.h - online Q-learning agents that fit in a microcontroller's SRAM.
 *
 * Three value-function representations share ONE learning target:
 *
 *     y = r + gamma * max_a' Q_target(s', a')      (y = r if s' is terminal)
 *     delta = y - Q(s, a)
 *
 * and differ only in how Q(s, a) is stored and how delta moves it:
 *
 *   TR_TABULAR  uniform grid, one value per (cell, action)
 *   TR_LINEAR   hashed tile coding, Q = sum of T weights
 *   TR_MLP      one hidden ReLU layer, semi-gradient SGD
 *
 * Optional stabilisers (all counted in memory): replay buffer (any learner),
 * target network and TD-error clipping (MLP).
 *
 * Memory model: NO malloc. The caller owns two blocks:
 *   - a tr_agent_t (fixed size, sizeof differs between host and M4)
 *   - an arena of tr_mem_bytes(cfg) bytes, carved up by tr_init()
 * On the microcontroller both are static arrays, so the linker sees them.
 *
 * All numbers crossing this API are Q16.16 (see tr_fixed.h).
 */
#ifndef TINYRL_H
#define TINYRL_H

#include <stddef.h>
#include <stdint.h>
#include "tr_fixed.h"

#define TR_MAX_OBS      12   /* raised from 8 (17 Nov): Go1 with contacts + gait parameters */
#define TR_MAX_ACTIONS  32   /* raised from 16: Go1 factored actions (18) */
#define TR_MAX_TILINGS  32

enum { TR_TABULAR = 0, TR_LINEAR = 1, TR_MLP = 2 };

enum {
    TR_OK          =  0,
    TR_ERR_CFG     = -1,   /* invalid configuration */
    TR_ERR_ARENA   = -2,   /* arena too small */
};

/*
 * Configuration. Every field is 32-bit so the Python ctypes mirror
 * (python/tinyrl/cbind.py) is trivial. Keep the field order in sync.
 */
typedef struct {
    int32_t kind;                     /* TR_TABULAR / TR_LINEAR / TR_MLP */
    int32_t obs_dim;                  /* 1..TR_MAX_OBS */
    int32_t n_actions;                /* 2..TR_MAX_ACTIONS */
    q16_t   obs_lo[TR_MAX_OBS];       /* observation bounds; values outside */
    q16_t   obs_hi[TR_MAX_OBS];       /* are clipped before use */
    q16_t   alpha;                    /* step size */
    q16_t   gamma;                    /* discount */
    /* tabular */
    int32_t bins;                     /* cells per dimension */
    /* linear: hashed tile coding */
    int32_t tilings;                  /* T, number of offset grids */
    int32_t tiles;                    /* tiles per dimension per tiling */
    int32_t features;                 /* M, hashed weights per action */
    /* mlp */
    int32_t hidden;                   /* hidden units */
    q16_t   td_clip;                  /* |delta| clip, 0 = off */
    int32_t target_period;            /* copy to target net every N updates, 0 = no target net */
    /* replay (any learner) */
    int32_t replay_cap;               /* transitions stored, 0 = pure streaming */
    int32_t batch;                    /* updates per env step when replay is on */
    /* ---- added 17 Nov 2026 after the literature review (docs/research_review.md) ----
     * All default to 0 = the original behaviour, bit for bit. */
    q16_t   lambda;                   /* Q(lambda) trace decay, no Watkins cut (D68); >0 enables traces */
    q16_t   obgd_kappa;               /* ObGD step-size cap (Elsayed et al. 2024), 0 = off; typical 2.0 */
    int32_t round_mode;               /* 0 = round to nearest, 1 = stochastic rounding of updates */
    int32_t norm;                     /* MLP only: 0 = none, 1 = parameter-free LayerNorm on hidden layer */
    int32_t trace_cap;                /* table/linear: sparse-trace entries kept, 0 = automatic */
    /* ---- added 17 Nov 2026, second round ---- */
    int32_t replay_compact;           /* 1 = compressed replay: uint16 states, next state implied */
    int32_t tile_dense;               /* linear: 1 = collision-free (unhashed) tile indices */
    int32_t cbp_period;               /* MLP: continual backprop, replace 1 unit every N updates, 0 = off */
    int32_t cbp_maturity;             /* MLP: units younger than this many updates are never replaced */
} tr_cfg_t;

typedef struct {
    /* plain layout (replay_compact = 0): full Q16.16 copies of s and s' */
    q16_t   *s;        /* replay_cap * obs_dim */
    q16_t   *s2;       /* replay_cap * obs_dim */
    uint8_t *a;        /* replay_cap */
    uint8_t *done;     /* replay_cap */
    /* compressed layout (replay_compact = 1): s as uint16 fraction of [lo, hi],
     * s' = the next entry's s (flag bit 7 says the successor continues this episode) */
    uint16_t *su;      /* replay_cap * obs_dim */
    uint8_t  *af;      /* replay_cap: action (bits 0-5) | done (bit 6) | has_next (bit 7) */
    q16_t    *last_s2; /* obs_dim: s' of the newest entry, to detect continuity */
    q16_t    *ds, *ds2;/* obs_dim each: decode scratch */
    q16_t   *r;        /* replay_cap (both layouts) */
    uint32_t head, size;
} tr_replay_t;

/* Sparse eligibility trace: (index, value) pairs, for one-hot / few-hot features. */
typedef struct {
    uint32_t *idx;
    q16_t    *val;
    uint32_t  n, cap;
} tr_strace_t;

typedef struct {
    q16_t *table;                     /* n_cells * n_actions */
    uint32_t n_cells;
    tr_strace_t tr;                   /* used only when lambda/ObGD is on */
} tr_tab_t;

/* Two-slot cache of active features: act(s), target(s'), update(s) and the
 * next act(s') reuse each other's hashing instead of recomputing it. */
typedef struct {
    q16_t    *obs[2];                 /* obs_dim each */
    uint32_t *idx[2];                 /* tilings each */
    uint8_t   valid[2];
    uint8_t   next;                   /* slot to overwrite next */
} tr_lin_cache_t;

typedef struct {
    q16_t    *w;                      /* features * n_actions */
    uint32_t *idx;                    /* points into the cache: active features of the last query */
    int32_t  *off;                    /* tilings * obs_dim precomputed tiling offsets (Q16.16) */
    q16_t     alpha_t;                /* alpha / tilings */
    uint32_t  mask;                   /* features-1 if features is a power of two, else 0 */
    tr_lin_cache_t cache;
    tr_strace_t tr;
} tr_lin_t;

typedef struct {
    q16_t *p;                         /* online params: W1 b1 W2 b2 */
    q16_t *tp;                        /* target params (NULL if no target net) */
    q16_t *z;                         /* hidden pre-activations (normalised if norm=1), scratch */
    q16_t *h;                         /* hidden activations, scratch */
    q16_t *g;                         /* dQ/dz scratch (norm or trace paths) */
    q16_t *e;                         /* eligibility trace, n_params (NULL if no traces) */
    q16_t *util;                      /* continual backprop: unit utility, hidden (NULL if off) */
    uint32_t *age;                    /* continual backprop: updates since (re)initialisation */
    q16_t  inv_sigma;                 /* LayerNorm 1/sigma of the last forward pass */
    uint32_t n_params;
} tr_mlp_t;

typedef struct {
    tr_cfg_t cfg;
    uint32_t rng;
    q16_t    inv_range[TR_MAX_OBS];   /* 1/(hi-lo), precomputed at init */
    q16_t   *xn;                      /* normalised obs scratch, obs_dim */
    q16_t   *q;                       /* action values scratch, n_actions */
    tr_replay_t rb;
    uint32_t n_updates;
    uint32_t arena_used;
    uint32_t n_upd_terms;             /* weight-update products computed */
    uint32_t n_underflow;             /* ... of which were non-zero before rounding, zero after */
    union { tr_tab_t tab; tr_lin_t lin; tr_mlp_t mlp; } u;
#ifndef TR_NO_DIAG
    /* Last, and absent from the M4 build, so its memory and instruction counts are untouched (D55). */
    uint32_t n_saturated;             /* update terms whose product or weight write was clipped */
    uint32_t term_saturated;          /* the current term is already counted */
    uint32_t step_saturated;          /* streaming: the step shared by this update's terms was clipped */
#endif
} tr_agent_t;

/* Bytes of arena this configuration needs (0 if the config is invalid). */
size_t   tr_mem_bytes(const tr_cfg_t *cfg);
/* sizeof(tr_agent_t) on this build (Python needs it to allocate). */
size_t   tr_agent_sizeof(void);
/* Number of learnable parameters (for logs). */
uint32_t tr_param_count(const tr_cfg_t *cfg);

int  tr_init(tr_agent_t *ag, const tr_cfg_t *cfg, void *arena, size_t arena_bytes, uint32_t seed);

/* Q(s, .) into out[n_actions]. */
void tr_qvalues(tr_agent_t *ag, const q16_t *obs, q16_t *out);
/* Greedy action, ties broken uniformly at random. */
int  tr_greedy(tr_agent_t *ag, const q16_t *obs);
/* Epsilon-greedy action; eps is Q16.16 in [0, 1]. */
int  tr_act(tr_agent_t *ag, const q16_t *obs, q16_t eps);
/* Store/learn from one transition. Returns number of updates performed. */
int  tr_observe(tr_agent_t *ag, const q16_t *s, int a, q16_t r, const q16_t *s2, int done);
/* Call at the start of every episode: clears eligibility traces (no-op without them).
 * Needed because a time-limit truncation is not passed to tr_observe as done. */
void tr_episode_start(tr_agent_t *ag);
/* Diagnostics: how many update products rounded to exactly zero (Q16.16 underflow). */
uint32_t tr_underflow_count(const tr_agent_t *ag);
uint32_t tr_update_terms(const tr_agent_t *ag);
/* ... and how many were clipped to the int32 range (always 0 with TR_NO_DIAG). */
uint32_t tr_saturation_count(const tr_agent_t *ag);
/* 1 if this config uses the streaming path (lambda > 0 or ObGD on). */
int  tr_is_stream(const tr_cfg_t *cfg);

#endif
