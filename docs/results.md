# Results log

Rule: an entry exists only if it was measured, with ≥ 5 seeds where learning
is involved, and it records **date, git commit, config file, exact command, n**.
Single-seed observations from the build are in `build_notes.md`, not here.

| date | stage | claim | numbers (mean ± 95% CI, n) | commit | command / config | figure |
|------|-------|-------|----------------------------|--------|------------------|--------|
| 17 Nov 2026 | 1 | C tabular learner reproduces the Python Q16.16 reference bit for bit on the corridor | 15/15 seed×ε cases identical | ec4ce4e | `make test` | — |
| 17 Nov 2026 | 2 | Pendulum, table learner: final greedy return improves monotonically with table resolution, and memory grows with bins² | 5 bins (1,008 B): −948 ± 47 · 7 (1,296 B): −933 ± 82 · 11 (2,160 B): −770 ± 41 · 15 (3,408 B): −582 ± 70 · 21 (6,000 B): −345 ± 63 · 31 (12,240 B): −245 ± 70 · random −1,147 ± 20 · zero torque −1,068; n = 10 seeds, t 95%, 300k steps, α tuned (0.45, K = 20) | af48728 (learner code; runs by Claude in the cloud workspace) | `bash scripts/run_experiment.sh stage2_pendulum_table` | `results/figures/stage2_pendulum_table_tuned_*.png` |
| 17 Nov 2026 | 2 | Table learner cost on the Cortex-M4 (QEMU) is flat in table size | 523–535 instructions per act + learn step; peak SRAM as above; host == M4 bit-exact for all 6 configs | af48728 | same | `_compute.png` |
| 17 Nov 2026 | M4 | Learning on the emulated chip (processor-in-the-loop) reproduces host learning bit for bit | 4/4 learners (table, linear, compressed-replay net, stream net), 2,000 pendulum steps each | 7ac6fb9 | `python3 scripts/pil_check.py` | `results/m4/pil_check.json` |

| 17 Nov 2026 | 3 (first pass) | Pendulum, all learners, equal 300k-step budget, equal tuning (K = 20 each), both encodings. **Per byte, the streaming network is the most memory-efficient learner; per instruction it is the most expensive** | streaming net (θ input): 1.8 KB −281 ± 106 · 2.8 KB −179 ± 41, at 17.6k–32.1k instructions/step · table (θ): 6.0 KB −342 ± 191 at 534 instr. · linear (θ): 4.0 KB −221 ± 58 at 1.8k instr. | 42a5157 | `bash scripts/run_experiment.sh pendulum_main --seeds 0 1 2 3 4` | `results/figures/pendulum_main_tuned_headline.png` |
| 17 Nov 2026 | 3 (first pass) | Best final performance at any memory: linear tile coding with cos/sin input | 1,024 features (13.3 KB): −142 ± 13 · 4,096 features (50.1 KB): −125 ± 9 · replay network (2,000-transition compressed buffer, H = 64, 27.2 KB): −137 ± 32 · table (cos/sin, 21 bins, 111.8 KB): −236 ± 26 | 42a5157 | same | same |
| 17 Nov 2026 | 3 (first pass) | Q16.16 update underflow differs by learner | share of update products rounded to zero: table ≤ 0.3% · linear ≤ 2.4% · replay network 0.9–12.6% (grows with width) · streaming network 25–51% (stochastic rounding on, so unbiased but noisy) | 42a5157 | same | `_underflow.png` |
| 17 Nov 2026 | 3 (first pass) | Input encoding (D01) changes the ranking | cos/sin costs the table bins³ instead of bins² (21 bins: 111.8 KB vs 6.0 KB) and hurts small tile codings (256 features: −722 ± 94 vs −221 ± 58), but helps linear from 1,024 features | 42a5157 | same | same |

All "first pass" rows: **n = 5 seeds** (the protocol asks for 10), 95% t-intervals, all memory and instruction numbers
measured on the M4 in QEMU (host == M4 bit-exact for all 34 learner configs), runs by Claude in the cloud workspace.
Rerun with `bash scripts/run_experiment.sh pendulum_main` (10 seeds) before citing. The pendulum_main log file was
truncated by a git operation during the run; the data (per-seed CSVs, metadata with the commit) is complete.

Notes on the Stage 2 entry: no table configuration reached the usual "solved" level (≈ −150) within
300k steps; the largest table was still improving (see the curves). The tuning budget (100k steps) is
shorter than the table needs (open decision D29).
