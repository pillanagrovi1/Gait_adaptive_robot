# Results log

Rule: an entry exists only if it was measured, with ≥ 5 seeds where learning
is involved, and it records **date, git commit, config file, exact command, n**.
Single-seed observations from the build are in `build_notes.md`, not here.

| date | stage | claim | numbers (mean ± 95% CI, n) | commit | command / config | figure |
|------|-------|-------|----------------------------|--------|------------------|--------|
| 17 Nov 2026 | 1 | C tabular learner reproduces the Python Q16.16 reference bit for bit on the corridor | 15/15 seed×ε cases identical | 0e0a247 | `make test` | — |
| 17 Nov 2026 | 2 | Pendulum, table learner: final greedy return improves monotonically with table resolution, and memory grows with bins² | 5 bins (1,008 B): −948 ± 47 · 7 (1,296 B): −933 ± 82 · 11 (2,160 B): −770 ± 41 · 15 (3,408 B): −582 ± 70 · 21 (6,000 B): −345 ± 63 · 31 (12,240 B): −245 ± 70 · random −1,147 ± 20 · zero torque −1,068; n = 10 seeds, t 95%, 300k steps, α tuned (0.45, K = 20) | ff8d592 (learner code; runs by Claude in the cloud workspace) | `bash scripts/run_experiment.sh stage2_pendulum_table` | `results/figures/stage2_pendulum_table_tuned_*.png` |
| 17 Nov 2026 | 2 | Table learner cost on the Cortex-M4 (QEMU) is flat in table size | 523–535 instructions per act + learn step; peak SRAM as above; host == M4 bit-exact for all 6 configs | ff8d592 | same | `_compute.png` |
| 17 Nov 2026 | M4 | Learning on the emulated chip (processor-in-the-loop) reproduces host learning bit for bit | 4/4 learners (table, linear, compressed-replay net, stream net), 2,000 pendulum steps each | 1e15897 | `python3 scripts/pil_check.py` | `results/m4/pil_check.json` |

| 17 Nov 2026 | 3 (first pass) | Pendulum, all learners, equal 300k-step budget, equal tuning (K = 20 each), both encodings. **Per byte, the streaming network is the most memory-efficient learner; per instruction it is the most expensive** | streaming net (θ input): 1.8 KB −281 ± 106 · 2.8 KB −179 ± 41, at 17.6k–32.1k instructions/step · table (θ): 6.0 KB −342 ± 191 at 534 instr. · linear (θ): 4.0 KB −221 ± 58 at 1.8k instr. | c3888b4 | `bash scripts/run_experiment.sh pendulum_main --seeds 0 1 2 3 4` | `results/figures/pendulum_main_tuned_headline.png` |
| 17 Nov 2026 | 3 (first pass) | Best final performance at any memory: linear tile coding with cos/sin input | 1,024 features (13.3 KB): −142 ± 13 · 4,096 features (50.1 KB): −125 ± 9 · replay network (2,000-transition compressed buffer, H = 64, 27.2 KB): −137 ± 32 · table (cos/sin, 21 bins, 111.8 KB): −236 ± 26 | c3888b4 | same | same |
| 17 Nov 2026 | 3 (first pass) | Q16.16 update underflow differs by learner | share of update products rounded to zero: table ≤ 0.3% · linear ≤ 2.4% · replay network 0.9–12.6% (grows with width) · streaming network 25–51% (stochastic rounding on, so unbiased but noisy) | c3888b4 | same | `_underflow.png` |
| 17 Nov 2026 | 3 (first pass) | Input encoding (D01) changes the ranking | cos/sin costs the table bins³ instead of bins² (21 bins: 111.8 KB vs 6.0 KB) and hurts small tile codings (256 features: −722 ± 94 vs −221 ± 58), but helps linear from 1,024 features | c3888b4 | same | same |

All "first pass" rows: **n = 5 seeds** (the protocol asks for 10), 95% t-intervals, all memory and instruction numbers
measured on the M4 in QEMU (host == M4 bit-exact for all 34 learner configs), runs by Claude in the cloud workspace.
Rerun with `bash scripts/run_experiment.sh pendulum_main` (10 seeds) before citing. The pendulum_main log file was
truncated by a git operation during the run; the data (per-seed CSVs, metadata with the commit) is complete.

Notes on the Stage 2 entry: no table configuration reached the usual "solved" level (≈ −150) within
300k steps; the largest table was still improving (see the curves). The tuning budget (100k steps) is
shorter than the table needs (open decision D29).

## Stage 4: replay ratio on the pendulum, table arm (22 Nov 2026)

5 replay updates per step did not measurably speed up learning (AUC difference +25, 95% CI [-22, +72], 10 paired seeds) at 2.7x the instructions per step.

**Provenance.** Commit `c32e1fc` (same tree as `d440a56`) · config `configs/replay_ratio.json` · command
`python3 scripts/run_sweep.py replay_ratio --jobs 6 --only table` · seeds 0–9 · mujoco 4.0.1, numpy 2.5.5, pandas 3.1.1 ·
run 22 Nov 2026, 17:14–17:15 CET · instructions and memory from `results/m4/replay_ratio.csv` (QEMU, synthetic
workload; instruction counts, not timing) · runs `results/raw_archive/replay_ratio.tar.gz` · log
`results/logs/replay_ratio_gate.log` · versions `results/logs/replay_ratio.versions`.

**What is compared.** n = 5 against n = 1 replay updates per environment step, **both learning from the same
1,000-transition compressed buffer**. This is not replay against the plain table. n = 1 is not online Q-learning
either: its one update uses a transition drawn uniformly from the stored ones whose successor is known, so never
the transition just observed. The table (11 bins per dimension, cos/sin input) and α = 0.300 are the same in every
arm. α was tuned without replay, on the angle input (D51). All three arms peak at 27,720 B of SRAM, buffer included.

**Criterion.** Pre-registered in D53 (`0fd1c9f`, 22 Nov 16:17 CET, before the run): the paired 95% t interval of
n = 5 − n = 1 on AUC must lie above 0. It does not, so Stage 4 stops and the linear and network arms were not run.
D53 replaced the criterion of 17 Nov (steps-to-threshold), which is reported below as a secondary check.
The original metric gives the same decision (inconclusive).

**Metrics.** AUC = mean greedy evaluation return over training (30 evaluations in 300k steps; higher = learned
sooner). Final = mean of the last 3 evaluations. Steps-to-threshold = first evaluation at or above −200.

| difference (paired, 10 seeds) | status | mean | 95% CI | verdict |
|---|---|---|---|---|
| AUC, n = 5 − n = 1 | pre-registered gate | +24.6 | [-22.4, +71.5] | inconclusive; gate not met |
| AUC, n = 20 − n = 1 | exploratory | -32.6 | [-83.3, +18.0] | inconclusive |
| final, n = 5 − n = 1 | exploratory, not pre-registered | -41.3 | [-137.5, +54.9] | inconclusive |
| final, n = 20 − n = 1 | exploratory, not pre-registered | -86.6 | [-183.4, +10.3] | inconclusive |

| updates per step | instructions per step | seeds that reached −200 |
|---|---|---|
| 1 | 1206.6 | 9/10 |
| 5 | 3306.3 | 10/10 |
| 20 | 11210.3 | 7/10 |

**Original criterion (secondary).** Steps-to-threshold, n = 1 vs n = 5:
9/10 vs 10/10 seeds reached -200; the direction depends on how the missing seed is counted, so inconclusive.

**Reproducibility.** The table arm was run once and has not been repeated. Every run's metadata records the commit
and the full configuration.

**Limits.** One task, one table size, one buffer size, untuned per replay ratio. The interval reaches +72, so a
useful speed-up from 5 updates is not ruled out.
