# Changelog

What was added when, and what it measured at the time. The README describes the current state.

## 17 Nov 2026 — reference implementation

All stages built: Q16.16 C learners (table, hashed tile coding, tiny network) with static arenas, the Stage 1
bit-exact gate, MuJoCo pendulum / cart-pole / Go1, harness, sweeps, plots, Cortex-M4 measurement in QEMU.

## 17 Nov 2026 — first literature review (`docs/research_review.md`)

[`docs/research_review.md`](docs/research_review.md) surveys recent work. These are in the code now, all opt-in, with defaults reproducing the original behaviour:

| Area | Change | Measured so far |
|---|---|---|
| Tiny network | Stream Q(λ): ObGD step cap, eligibility traces, LayerNorm, no replay, no target net (`lam`, `obgd_kappa`, `norm`) | 16 hidden: **1.8 KB** peak SRAM vs 16.8 KB with a 500-transition buffer (QEMU M4) |
| All learners | λ and ObGD available to table and linear too, so the comparison stays about representation | sparse traces: 64 (table) / 8·T (linear) entries |
| Fixed point | stochastic rounding (`round_mode: 1`) and an underflow counter in every run | a sub-LSB update test: nearest rounding keeps 0%, stochastic 99.8% of the exact value |
| Tile coding | cached features, precomputed offsets, 32-bit maths, mask for 2^k features: same outputs | **8,415 → 1,604** instructions/step on the M4 |
| Table | 32-bit division when it fits: same outputs | ~830 → ~530 instructions/step |
| Evaluation | step budget, 10 paired seeds, equal-budget tuning (`scripts/tune.py`), seed dots, paired differences, new adaptation metric, from-scratch reference | — |
| Go1 | phase-locked decisions + pair bit + half-cycle IMU averaging (`go1_gait_v2`); terrain difficulty sweep (`scripts/go1_difficulty.py`) | open-loop success: 1.0 at roughness 0.05, 0.4–0.5 at 0.08–0.09 (10 episodes each: a lead) |

## 17 Nov 2026 — second review (`docs/research_review_2.md`): algorithm trade-offs and self-adapting gaits

[`docs/research_review_2.md`](docs/research_review_2.md). Main message: on this chip compute is nearly free at 4–50 Hz;
SRAM and real samples are scarce, so spend cycles to save samples without spending SRAM.

| Added | Measured so far |
|---|---|
| Compressed replay (`replay_compact`), replay-ratio experiment (Stage 4 redefined) | buffer 2.67× smaller; M4 network 16.4 → 6.9 KB |
| Dense tiles (`tile_dense: 1`), hash finaliser (now the default, `tile_dense: 0`; legacy FNV is `tile_dense: 2`), `scripts/tile_collisions.py` | default pendulum hashing shares 64% of slots; with finaliser 51% |
| Continual backprop for the network, Q2.30 ObGD step | — |
| Kernel trade-off map: Q16.16 vs float32 vs q15, LSTD-Q, Dyna-Q (`scripts/bench_kernels.py`) | float32 ≈ Q16.16 for table and tiles; network 14–20% fewer instructions in float (QEMU understates float cost); LSTD-Q k=64: 135k instr., 18 KB |
| Go1 self-adapting gait: Tegotae phase feedback, IMU posture reflex, stumble reflex (B1–B4), ES over gains (B5), contacts, factored and persistent actions, `go1_gait_v3` | stumble reflex: success 0.1 → 0.7 at roughness 0.10 (10 episodes: a lead) |

## 17 Nov 2026 — processor-in-the-loop, decision review, first results

The learner can run **on the microcontroller** while the laptop only simulates the robot:

```
 laptop: MuJoCo task + harness  ──UART frames: obs, reward──▶  Cortex-M4: tr_act / tr_observe in SRAM
                                ◀──── action, cycle count ────
```

Add `"pil": "tcp:HOST:PORT"` (QEMU) or `"pil": "serial:/dev/ttyACM0"` (Nucleo-F446RE) to any learner config.
`python3 scripts/pil_check.py` starts the firmware in QEMU and checks that on-chip learning reproduces the host
learning curve **bit for bit** (passes for table, linear, compressed-replay network and streaming network; also in CI).
On a Nucleo it records cycles per `act()` / `observe()`: the first timing numbers the project may claim.
The Nucleo build (`make -C m4/pil nucleo`) has **not been run on hardware yet**.


## 22 Nov 2026 — pre-registration, the remaining experiments, write-up

| Area | Change |
|---|---|
| Protocol | D49–D60 registered before the runs: tuning at the full budget, fixed seeds, primary comparisons, t intervals on differences |
| Bugs fixed | evaluation at the shift step ran on the shifted task (D58); resume skipped interrupted runs (D59); analysis read unfinished seeds and miscounted pairs; stale configs and host ≠ M4 went unnoticed |
| Measurement | saturation counter (D55), firmware flash size (D56), γ = 0 baselines (D54), A→B→A shift for forgetting (D57) |
| Tools | `scripts/results_table.py` refuses tables without valid provenance; `scripts/compare.py` prints the registered differences |
| Experiments | replay ratio (gate not met, stopped), cart-pole, adaptation, stream ablation: see `docs/results.md` |
| Documents | `docs/report.md`, `docs/study_guide.md` (replaces the rewrite exercise) |
| Go1 | `go1_gait_v3` run: bar not met (no learner beat the stumble reflex), D65 |
| Linear learner | feature cache hashes once per step (D62): same learned values, every M4 count re-measured |
| Audit | evaluation limits documented (D66); online return and memory estimates corrected (D67); `scripts/check_docs.py` in CI |
| Known, not changed | ObGD differs from the paper for α ≠ 1 (D61) |
