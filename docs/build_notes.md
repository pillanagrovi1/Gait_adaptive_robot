# Build notes (17 Nov 2026): NOT results

Observations made by Claude while building and debugging the pipeline. Mostly
single seeds and short runs. They show the code runs and give you leads. They
are **not** evidence for any claim. Re-run anything here properly before
believing it.

## Verified checks (deterministic, re-run any time)

- C unit tests pass (`build/test_core`): fixed-point maths, each learner fits a
  known 1-step function, `arena_used == tr_mem_bytes`, replay warm-up, oversized tables rejected.
- Stage 1 gate passes: C corridor trace equals the Python Q16.16 mirror for seeds
  {1, 2, 3, 42, 1234} × ε {0, 0.1, 0.5}.
- Host and M4 (QEMU) builds take identical actions for every smoke and Go1-smoke
  config (`measure_m4.py smoke --check`).

## Single-seed observations (leads only)

Pendulum (seed 0, greedy eval return, higher is better, typical "solved" ≈ −150):
- Table, 21 bins, γ 0.98, 1500 episodes: about −130 from episode ~1200; noisy before that.
- Linear, 8×8 tiles, 1024 features, γ 0.98, 1000 episodes: −125 to −160 from episode ~400, one dip to −255.
- Network (32 hidden, replay 2000, batch 4, γ 0.98, θ input): did not learn in 600 episodes,
  and the float twin did not learn either → algorithm/hyperparameters, not fixed point.
- Network, γ 0.95, target sync 100, batch 8: unstable with θ input; with (cos θ, sin θ, θ̇)
  it reached about −130 by episode ~240 and stayed there. This is why D01 exists.
- Table and linear with γ 0.95 also reached about −130 to −155.

Cart-pole (seed 0, max 500):
- Table 8 bins: noisy, peak 415. Linear 2048 features: reached 500 several times.
- Network: unstable in float and C across several settings. With a 4000 replay buffer
  it needs about 154 KB, more than the F446's 128 KB.

Go1 (seed 0, roughness 0.06, 200 episodes):
- Open-loop trot: walks about 4 m in 10 s on flat ground; at roughness 0.06 no falls
  (6 episodes); at 0.09, 5 of 6 fall; at 0.12 few falls but little progress.
- Table (5 bins) improved from 18 to 25 eval return; linear from 24 to 29. Earlier, the
  open-loop nominal trot averaged 37 over 6 different terrain seeds. So it is
  unclear whether the learners beat doing nothing. Use the `nominal-trot` baseline to find out.

M4 (QEMU, smoke configs): instructions per act+learn step were about 830 (table),
8.4k (linear, 8 tilings), and 12.5k–21k (network, 8–16 hidden, batch 4). The
linear cost is dominated by hashing and a 64-bit modulo per tiling, which is
an easy optimisation target.

## Bugs found and fixed during the build

- `m4/bench_cfg.h` next to `bench_main.c` shadowed every per-config header, because
  `#include "…"` searches the source file's folder first. All M4 builds silently used
  the default config. It was caught because instructions/step came out as exactly 0.
  The default config now lives in `m4/default_cfg/`.
- A test tolerance in `test_fixed` ignored input quantisation error, which grows with operand size.


## Second round (17 Nov 2026, after the literature review): NOT results either

Deterministic checks (re-run any time):
- Defaults unchanged: C regression errors identical before/after (0.0273 / 0.0236 / 0.0041); Stage 1 gate still bit-exact.
- Streaming path: every learner fits the 1-step test function; Q(λ = 0.8) reaches Q* on the corridor (table and linear).
- Stochastic rounding test (α = 2/65536, 20,000 updates towards 0.1): round-to-nearest stays at exactly 0.0000;
  stochastic rounding reaches 0.0456 (exact 0.0457).
- In the synthetic 1-step test, 46% of the streaming network's update products rounded to zero under
  round-to-nearest (555,296 of 1,208,833). Real tasks will differ; the `_underflow.png` figure measures it per run.
- Host == M4 bit-exact for all smoke configs including the streaming table and network.

M4 (QEMU) instruction counts per act+learn step, smoke configs:

| learner | before | after |
|---|---|---|
| table | ~830 | ~530 (pendulum 2-D) / ~630 (3-D) |
| linear, 8 tilings | ~8,415 | ~1,604 (1,754–1,887 with 3-D input) |
| network 8 / 16 hidden, replay batch 4 | 12,519 / 20,831 | 13,424 / 22,450 (+7%: per-weight rounding-mode branch) |
| streaming network, 16 hidden | — | 19,010, peak SRAM 1,812 B |
| streaming table, 9 bins, 3-D | — | 2,792 (sparse-trace search), peak 9,800 B |

Go1 open-loop difficulty (`go1_difficulty.py --phase-lock`, 10 episodes per level; success = no fall and ≥ 2 m):
0.05 → 1.00, 0.07 → 0.90, 0.08 → 0.50, 0.09 → 0.40, 0.11 → 0.00 (stuck rather than fallen: 0.6 m mean).

Bugs found and fixed in this round:
- The M4 benchmark header generator wrote the Python field name `.lam` instead of C's `.lambda`, so every
  build failed and was reported as "does not fit in SRAM". Error messages now distinguish the two.
- With phase lock, 0.25 s was 12 CPG ticks of 0.02 s (not 12.5), so the decision phase drifted by 4% of a
  half cycle per decision. Phase lock now runs the CPG at 100 Hz (25 ticks).
