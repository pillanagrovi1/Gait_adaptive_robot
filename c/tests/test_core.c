/*
 * test_core.c - unit tests for fixed-point maths and the three learners.
 * Plain asserts, no framework. Run: make test
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "tinyrl.h"
#include "tr_rng.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double f(q16_t x) { return x / 65536.0; }

static void test_fixed(void)
{
    CHECK(q16_mul(Q16(1.5), Q16(2.0)) == Q16(3.0), "1.5*2");
    CHECK(q16_mul(Q16(-1.5), Q16(2.0)) == Q16(-3.0), "-1.5*2");
    CHECK(q16_add(Q16_MAX, Q16(1)) == Q16_MAX, "saturating add");
    CHECK(q16_sub(Q16_MIN, Q16(1)) == Q16_MIN, "saturating sub");
    CHECK(fabs(f(q16_sqrt(Q16(2.0))) - sqrt(2.0)) < 1e-4, "sqrt 2 = %f", f(q16_sqrt(Q16(2.0))));
    CHECK(q16_mul3(Q16(0.001), Q16(0.01), Q16(1.0)) != 0, "mul3 keeps small products");
    for (int k = -50; k <= 50; k++) {
        double a = k * 0.37, b = k * -0.11;
        double got = f(q16_mul(Q16(a), Q16(b)));
        /* inputs are themselves rounded to 1/65536, so error ~ (|a|+|b|)/65536 */
        double tol = (fabs(a) + fabs(b) + 1.0) / 65536.0;
        CHECK(fabs(got - a * b) < tol, "mul %f*%f=%f got %f", a, b, a * b, got);
    }
}

static void base_cfg(tr_cfg_t *c, int kind, int d, int A)
{
    memset(c, 0, sizeof(*c));
    c->kind = kind; c->obs_dim = d; c->n_actions = A;
    for (int i = 0; i < d; i++) { c->obs_lo[i] = Q16(-1); c->obs_hi[i] = Q16(1); }
    c->alpha = Q16(0.1); c->gamma = Q16(0.9);
    c->bins = 8; c->tilings = 8; c->tiles = 4; c->features = 256;
    c->hidden = 16; c->td_clip = Q16(1.0); c->target_period = 0;
}

/*
 * Regression test: a 1-step task where Q(s, a) should converge to a known
 * function f(s, a) = s0 * (a ? 1 : -1) * 0.5. Every learner must get close.
 */
static void test_regression(int kind, double tol)
{
    tr_cfg_t c; base_cfg(&c, kind, 2, 2);
    if (kind == TR_MLP) { c.alpha = Q16(0.02); c.hidden = 16; }
    if (kind == TR_LINEAR) { c.alpha = Q16(0.2); }
    static uint8_t arena[1 << 16];
    static tr_agent_t ag;
    size_t need = tr_mem_bytes(&c);
    CHECK(need > 0 && need <= sizeof(arena), "mem %zu", need);
    CHECK(tr_init(&ag, &c, arena, sizeof(arena), 7) == TR_OK, "init kind %d", kind);
    uint32_t rng = 12345;
    for (int it = 0; it < 40000; it++) {
        q16_t s[2] = { (q16_t)((int32_t)(tr_rng_next(&rng) >> 15) - Q16_ONE),
                       (q16_t)((int32_t)(tr_rng_next(&rng) >> 15) - Q16_ONE) };
        int a = (int)(tr_rng_next(&rng) & 1);
        q16_t r = q16_mul(s[0], a ? Q16(0.5) : Q16(-0.5));
        tr_observe(&ag, s, a, r, s, 1);            /* terminal: y = r */
    }
    double err = 0; int n = 0;
    for (double x = -0.9; x <= 0.9; x += 0.3) {
        q16_t s[2] = { Q16(x), Q16(0.2) }, q[2];
        tr_qvalues(&ag, s, q);
        err += fabs(f(q[1]) - 0.5 * x) + fabs(f(q[0]) + 0.5 * x);
        n += 2;
    }
    err /= n;
    printf("  kind %d regression mean abs err %.4f (arena %u B)\n", kind, err, ag.arena_used);
    CHECK(err < tol, "kind %d err %.4f >= %.4f", kind, err, tol);
    CHECK(ag.arena_used == need, "arena_used %u != mem %zu", ag.arena_used, need);
}

static void test_replay_and_target(void)
{
    tr_cfg_t c; base_cfg(&c, TR_MLP, 3, 3);
    c.replay_cap = 64; c.batch = 4; c.target_period = 50;
    static uint8_t arena[1 << 16];
    static tr_agent_t ag;
    CHECK(tr_init(&ag, &c, arena, sizeof(arena), 3) == TR_OK, "init replay");
    q16_t s[3] = {0}, s2[3] = {Q16(0.1), 0, 0};
    int n0 = tr_observe(&ag, s, 0, Q16(1), s2, 0);
    CHECK(n0 == 0, "warm-up should not update (got %d)", n0);
    for (int k = 0; k < 10; k++) tr_observe(&ag, s, 1, Q16(1), s2, 0);
    CHECK(ag.n_updates > 0, "replay updates happen");
    CHECK(ag.rb.size == 11, "replay size %u", ag.rb.size);
}

static void test_bad_cfg(void)
{
    tr_cfg_t c; base_cfg(&c, TR_TABULAR, 8, 2);
    c.bins = 100;                                  /* 100^8 cells: must be rejected */
    CHECK(tr_mem_bytes(&c) == 0, "huge table rejected");
    static uint8_t small[16];
    static tr_agent_t ag;
    base_cfg(&c, TR_TABULAR, 2, 2);
    CHECK(tr_init(&ag, &c, small, sizeof(small), 1) == TR_ERR_ARENA, "small arena rejected");
}

/* Streaming path (ObGD, lambda = 0): every learner must still fit the 1-step function. */
static void test_stream_regression(int kind, int norm, int round_mode, double tol)
{
    tr_cfg_t c; base_cfg(&c, kind, 2, 2);
    c.alpha = Q16(1.0); c.obgd_kappa = Q16(2.0); c.lambda = 0; c.td_clip = 0;
    c.norm = norm; c.round_mode = round_mode;
    if (kind == TR_MLP) c.alpha = Q16(0.5);
    static uint8_t arena[1 << 16];
    static tr_agent_t ag;
    size_t need = tr_mem_bytes(&c);
    CHECK(tr_init(&ag, &c, arena, sizeof(arena), 11) == TR_OK, "stream init kind %d", kind);
    uint32_t rng = 999;
    for (int it = 0; it < 40000; it++) {
        q16_t s[2] = { (q16_t)((int32_t)(tr_rng_next(&rng) >> 15) - Q16_ONE),
                       (q16_t)((int32_t)(tr_rng_next(&rng) >> 15) - Q16_ONE) };
        int a = (int)(tr_rng_next(&rng) & 1);
        q16_t r = q16_mul(s[0], a ? Q16(0.5) : Q16(-0.5));
        tr_episode_start(&ag);
        tr_observe(&ag, s, a, r, s, 1);
    }
    double err = 0; int n = 0;
    for (double x = -0.9; x <= 0.9; x += 0.3) {
        q16_t s[2] = { Q16(x), Q16(0.2) }, q[2];
        tr_qvalues(&ag, s, q);
        err += fabs(f(q[1]) - 0.5 * x) + fabs(f(q[0]) + 0.5 * x);
        n += 2;
    }
    err /= n;
    printf("  stream kind %d norm %d round %d: err %.4f, underflow %u/%u (arena %u B)\n",
           kind, norm, round_mode, err, tr_underflow_count(&ag), tr_update_terms(&ag), ag.arena_used);
    CHECK(err < tol, "stream kind %d err %.4f >= %.4f", kind, err, tol);
    CHECK(ag.arena_used == need, "stream arena_used %u != mem %zu", ag.arena_used, need);
}

/* Watkins Q(lambda) on the corridor: traces must reach the optimum too. */
static void test_qlambda_corridor(int kind)
{
    tr_cfg_t c; memset(&c, 0, sizeof(c));
    c.kind = kind; c.obs_dim = 1; c.n_actions = 2;
    c.obs_lo[0] = 0; c.obs_hi[0] = 6 * Q16_ONE;
    c.bins = 6; c.tilings = 1; c.tiles = 6; c.features = 64; c.hidden = 16;
    c.alpha = Q16(0.5); c.gamma = Q16(0.9); c.lambda = Q16(0.8);
    static uint8_t arena[1 << 14];
    static tr_agent_t ag;
    CHECK(tr_init(&ag, &c, arena, sizeof(arena), 5) == TR_OK, "qlambda init");
    for (int ep = 0; ep < 300; ep++) {
        int pos = 0, done = 0, steps = 0;
        tr_episode_start(&ag);
        while (!done && steps++ < 100) {
            /* cell centres: an observation exactly on a tile edge can round into the
             * tile below (inv_range is truncated), which is not what this test is about */
            q16_t s = pos * Q16_ONE + Q16_HALF;
            int a = tr_act(&ag, &s, Q16(0.2));
            pos += a ? 1 : -1; if (pos < 0) pos = 0;
            done = pos == 5;
            q16_t s2 = pos * Q16_ONE + Q16_HALF;
            tr_observe(&ag, &s, a, done ? Q16_ONE : 0, &s2, done);
        }
    }
    double worst = 0;
    for (int cell = 0; cell < 5; cell++) {
        q16_t s = cell * Q16_ONE + Q16_HALF, q[2];
        tr_qvalues(&ag, &s, q);
        double e = fabs(f(q[1]) - pow(0.9, 4 - cell));
        if (e > worst) worst = e;
    }
    printf("  Q(lambda=0.8) corridor kind %d: worst |Q - Q*| = %.4f\n", kind, worst);
    CHECK(worst < 0.02, "Q(lambda) corridor kind %d worst err %.4f", kind, worst);
}

static void test_stream_cfg_rules(void)
{
    tr_cfg_t c; base_cfg(&c, TR_LINEAR, 2, 2);
    c.lambda = Q16(0.8); c.replay_cap = 100; c.batch = 4;
    CHECK(tr_mem_bytes(&c) == 0, "traces + replay must be rejected");
    base_cfg(&c, TR_TABULAR, 2, 2);
    c.norm = 1;
    CHECK(tr_mem_bytes(&c) == 0, "LayerNorm on a table must be rejected");
}

/* Stochastic rounding keeps sub-LSB updates alive on average. */
static void test_stochastic_rounding(void)
{
    for (int mode = 0; mode <= 1; mode++) {
        tr_cfg_t c; base_cfg(&c, TR_TABULAR, 1, 2);
        c.bins = 1; c.alpha = 2;                        /* alpha = 2/65536: tiny */
        c.round_mode = mode;
        static uint8_t arena[256];
        static tr_agent_t ag;
        tr_init(&ag, &c, arena, sizeof(arena), 3);
        q16_t s = 0;
        for (int k = 0; k < 20000; k++) tr_observe(&ag, &s, 0, Q16(0.1), &s, 1);
        q16_t q[2];
        tr_qvalues(&ag, &s, q);
        /* exact: Q -> 0.1 * (1 - (1-alpha)^20000) = 0.1 * (1 - e^-0.61) = 0.0457 */
        printf("  rounding mode %d: Q after 20000 tiny updates = %.4f (exact ~0.0457), underflow %u/%u\n",
               mode, f(q[0]), tr_underflow_count(&ag), tr_update_terms(&ag));
        if (mode == 0) CHECK(f(q[0]) == 0.0, "nearest rounding should lose every sub-LSB update");
        else CHECK(fabs(f(q[0]) - 0.0457) < 0.01, "stochastic rounding mean %.4f", f(q[0]));
    }
}

/* Saturation counter (D55): one count per update term that was clipped. */
static void test_saturation(void)
{
    static uint8_t arena[8192];
    static tr_agent_t ag;
    for (int kind = TR_TABULAR; kind <= TR_LINEAR; kind++) {
        for (int stream = 0; stream <= 1; stream++) {
            tr_cfg_t c; base_cfg(&c, kind, 1, 2);
            c.bins = 1; c.tilings = 1; c.td_clip = 0;
            c.alpha = Q16(1.5);
            if (stream) c.lambda = Q16(0.5);
            q16_t s = 0;
            CHECK(tr_init(&ag, &c, arena, sizeof(arena), 1) == TR_OK, "init");
            tr_observe(&ag, &s, 0, Q16(20000), &s, 1);     /* weight = 30000: in range */
            CHECK(tr_saturation_count(&ag) == 0, "kind %d stream %d: in range", kind, stream);
            tr_observe(&ag, &s, 0, Q16(32000), &s, 1);     /* 30000 + 3000: the write clips */
            CHECK(tr_saturation_count(&ag) == 1, "kind %d stream %d: clipped write, got %u",
                  kind, stream, tr_saturation_count(&ag));
            c.alpha = Q16(2.0);
            tr_init(&ag, &c, arena, sizeof(arena), 1);
            tr_observe(&ag, &s, 0, Q16(20000), &s, 1);     /* 2 * 20000: the product clips */
            CHECK(tr_saturation_count(&ag) == 1, "kind %d stream %d: clipped product or step counts once, got %u",
                  kind, stream, tr_saturation_count(&ag));
        }
    }
    for (int stream = 0; stream <= 1; stream++) {
        tr_cfg_t c; base_cfg(&c, TR_MLP, 1, 2);
        c.td_clip = 0;
        c.alpha = Q16(2.0);
        if (stream) c.lambda = Q16(0.5);
        q16_t s = 0;
        CHECK(tr_init(&ag, &c, arena, sizeof(arena), 1) == TR_OK, "init");
        for (int k = 0; k < 4; k++) tr_observe(&ag, &s, 0, k % 2 ? Q16(-30000) : Q16(30000), &s, 1);
        uint32_t n = tr_saturation_count(&ag);
        CHECK(n > 0 && n <= tr_update_terms(&ag), "mlp stream %d: %u", stream, n);
        c.alpha = Q16(0.01);
        tr_init(&ag, &c, arena, sizeof(arena), 1);
        tr_observe(&ag, &s, 0, Q16(0.5), &s, 1);
        CHECK(tr_saturation_count(&ag) == 0, "mlp stream %d: small update", stream);
    }
}

/* D62: the feature cache evicts the least recently used state, so act(s) / target(s') / update(s)
 * never evicts the state that is needed next. */
static void test_linear_cache_lru(void)
{
    static uint8_t arena[8192];
    static tr_agent_t ag;
    tr_cfg_t c; base_cfg(&c, TR_LINEAR, 1, 2);
    CHECK(tr_init(&ag, &c, arena, sizeof(arena), 1) == TR_OK, "init");
    q16_t s0 = Q16(-0.5), s1 = Q16(0.0), s2 = Q16(0.5), q[2];
    tr_qvalues(&ag, &s0, q);
    tr_qvalues(&ag, &s1, q);
    tr_qvalues(&ag, &s0, q);        /* a hit: s0 is now the most recent */
    tr_qvalues(&ag, &s2, q);        /* a miss: must replace s1 */
    const tr_lin_cache_t *k = &ag.u.lin.cache;
    CHECK(k->obs[0][0] == s0 || k->obs[1][0] == s0, "recently used state was evicted");
    CHECK(k->obs[0][0] == s2 || k->obs[1][0] == s2, "new state was not cached");
}

/* Corridor with any config tweak; returns worst |Q(c,right) - Q*|. Observations at cell centres. */
static double corridor_worst(tr_cfg_t *c, int episodes)
{
    static uint8_t arena[1 << 16];
    static tr_agent_t ag;
    if (tr_init(&ag, c, arena, sizeof(arena), 9) != TR_OK) return 99.0;
    for (int ep = 0; ep < episodes; ep++) {
        int pos = 0, done = 0, steps = 0;
        tr_episode_start(&ag);
        while (!done && steps++ < 100) {
            q16_t s = pos * Q16_ONE + Q16_HALF;
            int a = tr_act(&ag, &s, Q16(0.2));
            pos += a ? 1 : -1; if (pos < 0) pos = 0;
            done = pos == 5;
            q16_t s2 = pos * Q16_ONE + Q16_HALF;
            tr_observe(&ag, &s, a, done ? Q16_ONE : 0, &s2, done);
        }
    }
    double worst = 0;
    for (int cell = 0; cell < 5; cell++) {
        q16_t s = cell * Q16_ONE + Q16_HALF, q[2];
        tr_qvalues(&ag, &s, q);
        double e = fabs(f(q[1]) - pow(0.9, 4 - cell));
        if (e > worst) worst = e;
    }
    return worst;
}

static void corridor_cfg(tr_cfg_t *c, int kind)
{
    memset(c, 0, sizeof(*c));
    c->kind = kind; c->obs_dim = 1; c->n_actions = 2;
    c->obs_lo[0] = 0; c->obs_hi[0] = 6 * Q16_ONE;
    c->bins = 6; c->tilings = 1; c->tiles = 6; c->features = 64; c->hidden = 16;
    c->alpha = Q16(0.5); c->gamma = Q16(0.9);
}

static void test_compact_replay(void)
{
    tr_cfg_t c; corridor_cfg(&c, TR_TABULAR);
    c.replay_cap = 200; c.batch = 4; c.replay_compact = 1;
    double w = corridor_worst(&c, 300);
    printf("  compact replay corridor (table): worst |Q - Q*| = %.4f\n", w);
    CHECK(w < 0.02, "compact replay corridor err %.4f", w);

    tr_cfg_t p; base_cfg(&p, TR_MLP, 3, 3);
    p.replay_cap = 2000; p.batch = 8;
    size_t plain = tr_mem_bytes(&p);
    p.replay_compact = 1;
    size_t compact = tr_mem_bytes(&p);
    printf("  replay 2000 x 3-D: plain %zu B, compact %zu B (%.2fx smaller)\n", plain, compact, (double)plain / compact);
    CHECK(compact < plain, "compact replay must be smaller");
}

static void test_dense_tiles(void)
{
    tr_cfg_t c; corridor_cfg(&c, TR_LINEAR);
    c.tile_dense = 1;                                 /* 1 tiling x 7 cells <= 64 features */
    double w = corridor_worst(&c, 300);
    printf("  dense tiles corridor: worst |Q - Q*| = %.4f\n", w);
    CHECK(w < 0.02, "dense tiles corridor err %.4f", w);
    tr_cfg_t b; base_cfg(&b, TR_LINEAR, 4, 2);
    b.tile_dense = 1; b.tiles = 8; b.tilings = 8; b.features = 1024;   /* needs 8 * 9^4 = 52488 */
    CHECK(tr_mem_bytes(&b) == 0, "dense tiles that do not fit must be rejected");
}

static void test_cbp(void)
{
    tr_cfg_t c; base_cfg(&c, TR_MLP, 2, 2);
    c.alpha = Q16(0.02); c.cbp_period = 200; c.cbp_maturity = 1000;
    static uint8_t arena[1 << 16];
    static tr_agent_t ag;
    CHECK(tr_init(&ag, &c, arena, sizeof(arena), 7) == TR_OK, "cbp init");
    uint32_t rng = 12345;
    for (int it = 0; it < 40000; it++) {
        q16_t s[2] = { (q16_t)((int32_t)(tr_rng_next(&rng) >> 15) - Q16_ONE),
                       (q16_t)((int32_t)(tr_rng_next(&rng) >> 15) - Q16_ONE) };
        int a = (int)(tr_rng_next(&rng) & 1);
        tr_observe(&ag, s, a, q16_mul(s[0], a ? Q16(0.5) : Q16(-0.5)), s, 1);
    }
    double err = 0; int n = 0;
    for (double x = -0.9; x <= 0.9; x += 0.3) {
        q16_t s[2] = { Q16(x), Q16(0.2) }, q[2];
        tr_qvalues(&ag, s, q);
        err += fabs(f(q[1]) - 0.5 * x) + fabs(f(q[0]) + 0.5 * x);
        n += 2;
    }
    err /= n;
    uint32_t young = 0;
    for (int j = 0; j < c.hidden; j++) young += ag.u.mlp.age[j] < 40000u;
    printf("  continual backprop: err %.4f, %u of %d units were replaced at least once\n", err, young, c.hidden);
    CHECK(err < 0.08, "cbp err %.4f", err);
    CHECK(young > 0, "cbp should have replaced some units");
}

int main(void)
{
    test_fixed();
    test_regression(TR_TABULAR, 0.08);   /* bins=8: discretisation error ~0.06 */
    test_regression(TR_LINEAR, 0.05);
    test_regression(TR_MLP, 0.05);
    test_replay_and_target();
    test_bad_cfg();
    test_stream_regression(TR_TABULAR, 0, 0, 0.08);
    test_stream_regression(TR_LINEAR, 0, 0, 0.06);
    test_stream_regression(TR_MLP, 0, 0, 0.06);
    test_stream_regression(TR_MLP, 1, 0, 0.08);
    test_stream_regression(TR_MLP, 1, 1, 0.08);
    test_qlambda_corridor(TR_TABULAR);
    test_qlambda_corridor(TR_LINEAR);
    test_stream_cfg_rules();
    test_stochastic_rounding();
    test_saturation();
    test_linear_cache_lru();
    test_compact_replay();
    test_dense_tiles();
    test_cbp();
    printf("%s (%d failures)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
