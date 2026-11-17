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

int main(void)
{
    test_fixed();
    test_regression(TR_TABULAR, 0.08);   /* bins=8: discretisation error ~0.06 */
    test_regression(TR_LINEAR, 0.05);
    test_regression(TR_MLP, 0.05);
    test_replay_and_target();
    test_bad_cfg();
    printf("%s (%d failures)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
