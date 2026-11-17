/*
 * kernels_main.c - cost of candidate learning updates on a Cortex-M4F (in QEMU).
 *
 * NOT the project's learners: these are minimal update kernels, one per
 * candidate from docs/research_review_2.md, so their memory and instruction
 * counts can be put on one trade-off map next to the real learners.
 * Data efficiency is NOT measured here; it comes from the literature or from
 * full learning experiments.
 *
 * One kernel is compiled per build (KERNEL), with sizes from -D flags:
 *   1 table_q16      Q-table update, Q16.16
 *   2 table_f32      same, float32 (hardware FPU)
 *   3 table_q15      same, int16 storage (Q4.11), 32-bit maths
 *   4 tile_q16       tile-coded linear update, T active features (indices precomputed)
 *   5 tile_f32
 *   6 tile_q15
 *   7 mlp_q16        1 hidden layer, forward(s') + forward(s) + backward + SGD
 *   8 mlp_f32
 *   9 lstd_f32       recursive LSTD-Q (Sherman-Morrison on a k x k inverse), dense k features
 *  10 dyna_q16       Q-table + model table, NPLAN planning updates per real step
 * Sizes: CELLS, ACTIONS, T (tilings), M (features per action), D (inputs), H (hidden), K, NPLAN.
 *
 * Output (semihosting): bytes=<static learner memory> stack=<peak> checksum=<..>
 * Instructions per step: scripts/bench_kernels.py runs N1 and N2 steps and differences.
 */
#include <stdint.h>
#include <string.h>
#include "semihost.h"

#ifndef STEPS
#define STEPS 100
#endif
#ifndef ACTIONS
#define ACTIONS 3
#endif

#define PAINT 0xA5A5A5A5u
extern uint32_t _ebss, _estack;

static uint32_t rng = 0x12345u;
static inline uint32_t xs32(void) { uint32_t x = rng; x ^= x << 13; x ^= x >> 17; x ^= x << 5; rng = x; return x; }

typedef int32_t q16;
static inline q16 qmul(q16 a, q16 b) { return (q16)(((int64_t)a * b + 32768) >> 16); }
#define QA 19661   /* alpha 0.3 */
#define QG 62259   /* gamma 0.95 */

/* ------------------------------------------------------------------ table */
#if KERNEL == 1 || KERNEL == 2 || KERNEL == 3 || KERNEL == 10
#if KERNEL == 2
typedef float val_t;
#define VA 0.3f
#define VG 0.95f
static inline val_t vr(void) { return (float)(int32_t)(xs32() & 0xFFFF) / 65536.0f - 0.5f; }
#define UPD(q, y) ((q) + VA * ((y) - (q)))
#define TGT(r, m) ((r) + VG * (m))
#elif KERNEL == 3
typedef int16_t val_t;      /* Q4.11: range +-16, step 1/2048 */
static inline val_t vr(void) { return (val_t)((int32_t)(xs32() & 0x7FF) - 1024); }
#define UPD(q, y) (val_t)((q) + (((int32_t)19661 * ((int32_t)(y) - (q)) + 32768) >> 16))
#define TGT(r, m) (val_t)((r) + (((int32_t)62259 * (m) + 32768) >> 16))
#else
typedef q16 val_t;
static inline val_t vr(void) { return (q16)(xs32() & 0xFFFF) - 32768; }
#define UPD(q, y) ((q) + qmul(QA, (y) - (q)))
#define TGT(r, m) ((r) + qmul(QG, (m)))
#endif
static val_t Q[CELLS * ACTIONS];
#if KERNEL == 10
static uint16_t model_next[CELLS * ACTIONS];     /* next cell + 1 (0 = unseen) */
static val_t model_r[CELLS * ACTIONS];
#endif
static uint32_t learn_step(uint32_t s, uint32_t a, val_t r, uint32_t s2)
{
    const val_t *row = &Q[s2 * ACTIONS];
    val_t m = row[0];
    for (int k = 1; k < ACTIONS; k++) if (row[k] > m) m = row[k];
    val_t *q = &Q[s * ACTIONS + a];
    *q = UPD(*q, TGT(r, m));
#if KERNEL == 10
    model_next[s * ACTIONS + a] = (uint16_t)(s2 + 1);
    model_r[s * ACTIONS + a] = r;
    for (int p = 0; p < NPLAN; p++) {          /* Dyna: replay the model */
        for (int tries = 0; tries < 8; tries++) {
            uint32_t j = xs32() % (CELLS * ACTIONS);
            if (!model_next[j]) continue;
            uint32_t pn = model_next[j] - 1u;
            const val_t *rw = &Q[pn * ACTIONS];
            val_t mm = rw[0];
            for (int k = 1; k < ACTIONS; k++) if (rw[k] > mm) mm = rw[k];
            Q[j] = UPD(Q[j], TGT(model_r[j], mm));
            break;
        }
    }
#endif
    return (uint32_t)*q;
}
static uint32_t run(uint32_t steps)
{
    uint32_t chk = 0, s = 0;
#if KERNEL == 10
    /* steady state: every (cell, action) already seen once, so each planning
     * attempt succeeds (otherwise early steps under-count the planning cost) */
    for (uint32_t j = 0; j < CELLS * ACTIONS; j++) { model_next[j] = (uint16_t)(xs32() % CELLS + 1u); model_r[j] = vr(); }
#endif
    for (uint32_t k = 0; k < steps; k++) {
        uint32_t s2 = xs32() % CELLS, a = xs32() % ACTIONS;
        chk ^= learn_step(s, a, vr(), s2);
        s = s2;
    }
    return chk;
}
static uint32_t mem_bytes(void)
{
    uint32_t b = sizeof(Q);
#if KERNEL == 10
    b += sizeof(model_next) + sizeof(model_r);
#endif
    return b;
}
#endif

/* ------------------------------------------------------------------ tile coding */
#if KERNEL == 4 || KERNEL == 5 || KERNEL == 6
#if KERNEL == 5
typedef float w_t;
#elif KERNEL == 6
typedef int16_t w_t;        /* Q4.11 */
#else
typedef q16 w_t;
#endif
static w_t W[M * ACTIONS];
static uint32_t idx_s[T], idx_s2[T];
static uint32_t run(uint32_t steps)
{
    uint32_t chk = 0;
    for (int t = 0; t < T; t++) idx_s[t] = xs32() % M;
    for (uint32_t k = 0; k < steps; k++) {
        for (int t = 0; t < T; t++) idx_s2[t] = xs32() % M;       /* stand-in for hashing (cached) */
        uint32_t a = xs32() % ACTIONS;
#if KERNEL == 5
        float best = -1e30f, qsa = 0.0f, r = (float)(int32_t)(xs32() & 0xFFFF) / 65536.0f - 0.5f;
        for (int b = 0; b < ACTIONS; b++) {
            float acc = 0.0f;
            for (int t = 0; t < T; t++) acc += W[idx_s2[t] * ACTIONS + b];
            if (acc > best) best = acc;
        }
        for (int t = 0; t < T; t++) qsa += W[idx_s[t] * ACTIONS + a];
        float step = (0.3f / T) * (r + 0.95f * best - qsa);
        for (int t = 0; t < T; t++) W[idx_s[t] * ACTIONS + a] += step;
        chk ^= (uint32_t)(int32_t)(step * 65536.0f);
#else
        int32_t best = INT32_MIN, qsa = 0, r = (int32_t)(xs32() & 0xFFFF) - 32768;
#if KERNEL == 6
        r >>= 5;                                                   /* Q16.16 -> Q4.11 */
#endif
        for (int b = 0; b < ACTIONS; b++) {
            int32_t acc = 0;
            for (int t = 0; t < T; t++) acc += W[idx_s2[t] * ACTIONS + b];
            if (acc > best) best = acc;
        }
        for (int t = 0; t < T; t++) qsa += W[idx_s[t] * ACTIONS + a];
        int32_t delta = r + (int32_t)(((int64_t)62259 * best + 32768) >> 16) - qsa;
        int32_t step = (int32_t)(((int64_t)(19661 / T) * delta + 32768) >> 16);
        for (int t = 0; t < T; t++) W[idx_s[t] * ACTIONS + a] = (w_t)(W[idx_s[t] * ACTIONS + a] + step);
        chk ^= (uint32_t)step;
#endif
        for (int t = 0; t < T; t++) idx_s[t] = idx_s2[t];
    }
    return chk;
}
static uint32_t mem_bytes(void) { return sizeof(W) + sizeof(idx_s) + sizeof(idx_s2); }
#endif

/* ------------------------------------------------------------------ tiny network */
#if KERNEL == 7 || KERNEL == 8
#define NP (H * D + H + ACTIONS * H + ACTIONS)
#if KERNEL == 8
typedef float n_t;
#else
typedef q16 n_t;
#endif
static n_t P[NP], Z[H], Hh[H], X[D], Qo[ACTIONS];
static void fwd(void)
{
    const n_t *W1 = P, *b1 = P + H * D, *W2 = b1 + H, *b2 = W2 + ACTIONS * H;
    for (int j = 0; j < H; j++) {
#if KERNEL == 8
        float acc = b1[j];
        for (int i = 0; i < D; i++) acc += W1[j * D + i] * X[i];
        Z[j] = acc; Hh[j] = acc > 0.0f ? acc : 0.0f;
#else
        int64_t acc = 0;
        for (int i = 0; i < D; i++) acc += (int64_t)W1[j * D + i] * X[i];
        Z[j] = (q16)((acc + 32768) >> 16) + b1[j]; Hh[j] = Z[j] > 0 ? Z[j] : 0;
#endif
    }
    for (int k = 0; k < ACTIONS; k++) {
#if KERNEL == 8
        float acc = b2[k];
        for (int j = 0; j < H; j++) acc += W2[k * H + j] * Hh[j];
        Qo[k] = acc;
#else
        int64_t acc = 0;
        for (int j = 0; j < H; j++) acc += (int64_t)W2[k * H + j] * Hh[j];
        Qo[k] = (q16)((acc + 32768) >> 16) + b2[k];
#endif
    }
}
static void rand_x(void)
{
    for (int i = 0; i < D; i++)
#if KERNEL == 8
        X[i] = (float)(int32_t)(xs32() & 0xFFFF) / 32768.0f - 1.0f;
#else
        X[i] = (q16)(xs32() & 0x1FFFF) - 65536;
#endif
}
static uint32_t run(uint32_t steps)
{
    uint32_t chk = 0;
    for (int k = 0; k < NP; k++)
#if KERNEL == 8
        P[k] = (float)(int32_t)(xs32() & 0xFFFF) / 65536.0f - 0.5f;
#else
        P[k] = (q16)(xs32() & 0xFFFF) - 32768;
#endif
    n_t *W1 = P, *b1 = P + H * D, *W2 = b1 + H, *b2 = W2 + ACTIONS * H;
    for (uint32_t s = 0; s < steps; s++) {
        rand_x(); fwd();                               /* target: forward on s' */
        n_t m = Qo[0];
        for (int k = 1; k < ACTIONS; k++) if (Qo[k] > m) m = Qo[k];
        rand_x(); fwd();                               /* forward on s */
        int a = (int)(xs32() % ACTIONS);
#if KERNEL == 8
        float delta = 0.1f + 0.95f * m - Qo[a], lr = 0.01f;
        for (int j = 0; j < H; j++) {
            float g = Z[j] > 0.0f ? delta * W2[a * H + j] : 0.0f;
            W2[a * H + j] += lr * delta * Hh[j];
            if (g != 0.0f) { for (int i = 0; i < D; i++) W1[j * D + i] += lr * g * X[i]; b1[j] += lr * g; }
        }
        b2[a] += lr * delta;
        chk ^= (uint32_t)(int32_t)(delta * 65536.0f);
#else
        q16 delta = 6554 + qmul(QG, m) - Qo[a], lr = 655;
        for (int j = 0; j < H; j++) {
            q16 g = Z[j] > 0 ? qmul(delta, W2[a * H + j]) : 0;
            W2[a * H + j] += qmul(qmul(lr, delta), Hh[j]);
            if (g != 0) { for (int i = 0; i < D; i++) W1[j * D + i] += qmul(qmul(lr, g), X[i]); b1[j] += qmul(lr, g); }
        }
        b2[a] += qmul(lr, delta);
        chk ^= (uint32_t)delta;
#endif
    }
    return chk;
}
static uint32_t mem_bytes(void) { return sizeof(P) + sizeof(Z) + sizeof(Hh) + sizeof(X) + sizeof(Qo); }
#endif

/* ------------------------------------------------------------------ recursive LSTD-Q */
#if KERNEL == 9
/* Ainv (k x k), b, theta; phi, phi2 dense features of (s,a) and (s', pi(s')).
 * Sherman-Morrison:  u = Ainv phi;  v = Ainv^T (phi - g phi2);  Ainv -= u v^T / (1 + v.phi)
 * theta = Ainv b is recomputed every step here (k^2), the expensive but simplest choice. */
static float Ainv[K * K], bv[K], th[K], phi[K], phi2[K], u[K], v[K];
static uint32_t run(uint32_t steps)
{
    uint32_t chk = 0;
    for (int i = 0; i < K; i++) Ainv[i * K + i] = 100.0f;      /* 1/epsilon initial inverse */
    for (uint32_t s = 0; s < steps; s++) {
        for (int i = 0; i < K; i++) {
            phi[i] = (float)(xs32() & 0xFF) / 256.0f;
            phi2[i] = (float)(xs32() & 0xFF) / 256.0f;
        }
        float r = (float)(int32_t)(xs32() & 0xFFFF) / 65536.0f - 0.5f;
        for (int i = 0; i < K; i++) {
            float acc = 0.0f;
            for (int j = 0; j < K; j++) acc += Ainv[i * K + j] * phi[j];
            u[i] = acc;
        }
        for (int j = 0; j < K; j++) {
            float acc = 0.0f;
            for (int i = 0; i < K; i++) acc += (phi[i] - 0.95f * phi2[i]) * Ainv[i * K + j];
            v[j] = acc;
        }
        float den = 1.0f;
        for (int i = 0; i < K; i++) den += v[i] * phi[i];
        float inv = 1.0f / den;
        for (int i = 0; i < K; i++) {
            float ui = u[i] * inv;
            for (int j = 0; j < K; j++) Ainv[i * K + j] -= ui * v[j];
            bv[i] += r * phi[i];
        }
        for (int i = 0; i < K; i++) {
            float acc = 0.0f;
            for (int j = 0; j < K; j++) acc += Ainv[i * K + j] * bv[j];
            th[i] = acc;
        }
        chk ^= (uint32_t)(int32_t)(th[0] * 1000.0f);
    }
    return chk;
}
static uint32_t mem_bytes(void) { return sizeof(Ainv) + sizeof(bv) + sizeof(th) + sizeof(phi) + sizeof(phi2) + sizeof(u) + sizeof(v); }
#endif

__attribute__((noinline)) static uint32_t bench(void) { return run(STEPS); }

int main(void)
{
    uint32_t sp;
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    for (uint32_t *p = &_ebss; p < (uint32_t *)(sp - 64u); p++) *p = PAINT;
    uint32_t chk = bench();
    uint32_t *low = &_ebss;
    while (low < (uint32_t *)sp && *low == PAINT) low++;
    sh_kv("bytes", mem_bytes());
    sh_kv("stack", sp - (uint32_t)low);
    sh_kv("checksum", chk);
    return 0;
}
