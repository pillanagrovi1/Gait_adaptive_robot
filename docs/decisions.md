# Decisions log

## Decision review, 17 Nov 2026 (made by Claude at Rahul's request: "make some obvious decisions yourself")

Every entry below was "to review". The clear-cut ones are now settled; the reason is given in one line.
**Open** means the decision is genuinely Rahul's (it changes the research question, the scope or the claims).
Re-open anything by adding a new entry; do not edit history.

| ID | Topic | Status | Reason |
|---|---|---|---|
| D01 | pendulum encoding differs between learners | **changed → D25** | encoding is now a factor for every learner |
| D02 | equal episode budget | **changed → D26** | equal step budget (Patterson et al. 2024) |
| D03 | plain SGD, no Adam | **accepted** | Adam triples memory; the stream path (D23) gives the network a stable option |
| D04 | γ per task, same for all learners | **accepted** | γ is part of the task definition; never tuned per learner |
| D05 | Go1 task design v1 | **superseded → D44** | `go1_gait_v3` is the default Go1 experiment; v1/v2 kept for comparison |
| D06 | Q16.16 everywhere | **accepted for the core** | float32 is measured as a factor at kernel level (D46); a full float build waits for Nucleo cycle counts |
| D07 | round-half-up rounding | **accepted** | deterministic and host == M4; stochastic rounding is the opt-in remedy |
| D08 | random tie-breaking | **accepted** | removes action-0 bias at zero initialisation |
| D09 | hashed tile coding, T = 8 | **changed** | finaliser by default (D37); T = 8 for 2–3-D states, 16 for ≥ 4-D (Sutton & Barto's 4k rule), as in the configs |
| D10 | 1 hidden ReLU layer, fan-in init | **accepted** | the research question is about tiny networks |
| D11 | TD clip ±1 (replay network) | **accepted** | the stream path uses ObGD instead |
| D12 | replay stored in Q16.16 | **changed → D35** | compressed replay is used in every config except one CI smoke learner |
| D13 | time-outs bootstrap | **accepted** | standard (Pardo et al. 2018) |
| D14 | fixed reward scales | **accepted** | keeps Q-values inside Q16.16; plots show unscaled returns |
| D15 | peak SRAM = agent + arena + learner stack | **accepted** | — |
| D16 | QEMU netduinoplus2 + F446 memory map | **accepted** | plus the real-board path (D47) |
| D17 | `-O2`, soft-float ABI | **accepted** | soft-float is right for integer code; `-Os` can be a later factor |
| D18 | instruction counts by differencing | **accepted** | — |
| D19 | 5 seeds, t-intervals | **superseded → D27** | — |
| D20 | ε linear 1 → 0.05 over the first 60% | **accepted** | in steps since D26 |
| D21 | 90 % recovery metric | **superseded → D28** | — |
| D22 | baselines in every experiment | **accepted** | extended by the Go1 reflex ladder (D43) |
| D23–D33 | first-review features | **accepted** | all opt-in, tested, host == M4 |
| D34 | truncated normalisation reciprocal | **changed (applied)** | rounded now; no results existed yet, so nothing is invalidated |
| D35–D36 | compressed replay, dense tiles | **accepted** | — |
| D37 | hash finaliser optional | **changed (applied)** | now the default (`tile_dense: 0`); plain FNV kept as `tile_dense: 2` for the comparison |
| D38 | continual backprop | **accepted as an option** | used by `net-stream-cbp` in the adaptation experiments |
| D39 | Q2.30 ObGD step | **accepted** | harmless; did not reduce underflow (documented) |
| D40 | larger limits | **accepted** | +~150 B per agent, equal for all learners |
| D41–D44 | Go1 contacts, actions, reflex baselines, v3 | **accepted** | — |
| D45 | (1+1)-ES over reflex gains | **accepted as a baseline only** | the brief retired ARS-type methods as core learners |
| D46 | Dyna-Q | **decided: not in the learners** | the brief lists world models as out of scope; Dyna stays a measured point on the trade-off map |
| D29 | tune once at the middle of each sweep, 100k-step tuning budget | **decided 22 Nov → D49** | Stage 2 showed the table is still learning at 100k steps, so tuning rewards fast starters. Rahul: keep (cheap, equal) or tune at the full budget (3× cost)? |
| D47 | processor-in-the-loop | **accepted** | see below |
| — | float32 learner library | **deferred 22 Nov → D60** | worth it only if Nucleo cycle counts show float is not slower (QEMU cannot tell) |


Format: **ID — decision.** Why. Alternatives. Status.

Every entry below was made by Claude while building the reference
implementation (17 Nov 2026), so each starts as **to review**. Change the
status to *accepted*, *changed (→ new ID)* or *rejected*, with the reason
and the date. The ones most likely to change a conclusion come first.

## Most likely to change the conclusions

**D01 — The pendulum's state representation differs between learners in `pendulum_main`.** → *changed 17 Nov (D25): the encoding is now an explicit factor for every learner.*
The table and linear learners see (θ, θ̇). The network sees (cos θ, sin θ, θ̇),
because with raw θ it oscillated in a one-seed probe, probably due to the jump
at ±π. This is a confound: part of any gap between learners could come from the
input encoding, not the representation. `state_dim_pendulum` runs every learner on
both encodings to measure it. *Alternative:* give every learner the same encoding
(trig costs the table 21³ cells). **Status: to review — decide before Stage 3 results.**

**D02 — Equal episode budget, not equal compute or update budget.** → *changed 17 Nov (D26): equal environment-step budget.*
Within a task every learner gets the same number of episodes. With replay, the
network does `batch` updates per step, so it gets 8× more gradient steps.
*Alternatives:* equal updates; equal instruction budget (measurable from `results/m4`).
**Status: to review.**

**D03 — Network optimiser is plain SGD, no Adam.** → *extended 17 Nov (D23): a streaming variant with ObGD is added; still no Adam.*
Adam keeps two extra numbers per parameter, which would triple memory, and it
is the fair-to-memory baseline. Plain-SGD semi-gradient Q-learning with small
networks is known to be unstable. That instability is part of the finding, not
a bug to hide. *Alternatives:* Adam or RMSprop with their memory counted;
streaming methods à la Elsayed et al. 2024 (ObGD, sparse init, layer norm).
**Status: to review.**

**D04 — Discount γ: pendulum 0.95, cart-pole 0.99, Go1 0.9, the same for all learners within a task.**
For the pendulum, 0.98 was tried first and changed to 0.95 after the network
failed with 0.98 in a one-seed probe. All three learners then learned with
0.95. Choosing γ to suit one learner biases the comparison slightly.
**Status: to review.**

**D05 — Go1 task design: discrete gait adjustments on a fixed CPG, IMU state, 4 Hz decisions.**
This keeps state and action spaces table-sized. The CPG constants
(2 Hz, thigh amplitude 0.25 rad, calf lift 0.35 rad) came from one manual check
that the trot walks forward (about 0.4 m/s) on flat ground. Nominal roughness 0.06
(no falls for the open-loop trot); shift +50% → 0.09 (5 of 6 falls).
*Open problem:* in single-seed probes the learners did not clearly beat the
open-loop nominal trot. The `nominal-trot` baseline exists for this reason.
*Alternatives:* continuous residuals on CPG parameters; foot-contact
observations; reward shaping. **Status: to review — this is the main open design question for Stage 7.**

## Representation and algorithm

**D06 — Q16.16 everywhere (values, weights, observations).**
One format keeps the C/Python mirror simple and makes "4 bytes per parameter"
equal across learners. *Alternatives:* int16 weights (halves memory, needs scaling);
Q8.24 for network weights (finer steps). **Status: to review.**

**D07 — Rounding: round-half-up then arithmetic shift; saturate on every add/multiply.**
Deterministic, and identical on x86 and ARM GCC (verified by the corridor gate and
the host==M4 checksum). *Alternatives:* stochastic rounding (helps tiny updates).
**Status: to review.**

**D08 — Greedy ties broken uniformly at random (consumes RNG).**
With zero-initialised tables, lowest-index tie-breaking biases early behaviour
towards action 0. **Status: to review.**

**D09 — Linear features = hashed tile coding, T = 8 tilings, asymmetric offsets (1, 3, 5, …).**
Following Sutton & Barto 2018, §9.5.4. Hashing makes memory a free knob
(`features`) independent of state dimension. Collisions are the price.
*Alternatives:* unhashed tiles; RBF; Fourier basis. **Status: to review.**

**D10 — Network: one hidden ReLU layer, uniform fan-in init, biases 0.**
**Status: to review.**

**D11 — TD-error clipping at ±1 (scaled reward units) for the network.**
Equivalent to the Huber-loss gradient. Available to every learner (`td_clip`), but
only the network uses it in the configs. **Status: to review.**

**D12 — Replay: uniform sampling, stored in Q16.16, warm-up = `batch` transitions.**
Memory = cap·(8·d + 6) bytes. 2000 transitions of a 3-D state cost about 60 KB,
nearly half the F446's SRAM. *Alternatives:* int16 storage; smaller buffers;
prioritised replay. Stage 4 sweeps the size. **Status: to review.**

**D13 — Truncation (time limit) is not terminal; only real failures stop bootstrapping.**
Standard (Pardo et al. 2018). **Status: to review.**

**D14 — Reward scales: pendulum ÷16, cart-pole ×0.1, Go1 ×0.1.**
Chosen so that Q-values stay well inside the Q16.16 range and TD errors are of
order 1. Plots and logs always show unscaled returns. **Status: to review.**

## Measurement

**D15 — Peak working memory = sizeof(agent) + arena + learner stack, measured on the M4 build.**
Excludes code/flash, `main`, startup and the environment. **Status: to review.**

**D16 — Platform: QEMU `netduinoplus2` (STM32F405, Cortex-M4) with an F446-sized linker script (512 KB flash, 128 KB SRAM).**
Not the F446 itself. Fine for memory and instruction counts, useless for timing. **Status: to review.**

**D17 — M4 build flags: `-O2 -mcpu=cortex-m4 -mthumb -mfloat-abi=soft`.**
-O2 to match the host build; no FPU needed (integer only). Stack and
instruction counts depend on this. *Alternative:* `-Os` (common on MCUs).
**Status: to review.**

**D18 — Instruction count per step by differencing two runs (100 and 300 steps), synthetic workload.**
**Status: to review.**

## Evaluation protocol

**D19 — (superseded by D27 for seeds, episodes per evaluation and intervals.)** Learning performance = greedy return over 5 fixed-seed evaluation episodes, every 50 training episodes; final = mean of the last 3 evaluation points; 95% t-interval over 5 seeds.**
**Status: to review.**

**D20 — Exploration: ε linear 1.0 → 0.05 over the first 60% of episodes.**
**Status: to review.**

**D21 — (superseded by D28.)** Adaptation metric: episodes after the shift until the evaluation return gets 90% of the way from its post-shift minimum to its post-shift final level.**
"Not recovered" is reported as missing, never as a number. **Status: to review.**

**D22 — Baselines in every experiment: random policy, plus zero-torque (pendulum) or the open-loop nominal trot (Go1).**
**Status: to review.**


## Added 17 Nov 2026 after the literature review (`docs/research_review.md`)

Same rule: made by Claude, **to review**. All new C features are opt-in config fields that default to the
original behaviour (the C unit tests confirm identical regression errors; the Stage 1 gate still passes bit for bit).

**D23 — Streaming path: Watkins Q(λ) + ObGD step-size cap (Elsayed et al. 2024), available to ALL three learners.**
*Corrected 23 Nov by D68:* the code never made the Watkins cut described below; it is Q(λ) without the cut.
Config: `lam` (trace decay), `obgd_kappa` (cap; 0 = off). The learning target stays the shared max-target
y = r + γ max Q(s′,·); λ only spreads δ over recent states, with the trace cut after exploratory actions
(Watkins). Making λ available to every learner keeps the comparison about representation (λ is a shared
factor, not a network-only advantage). Traces: dense for the network (one value per weight), sparse
(index, value) lists for table (64 entries) and linear (8·T entries). Not included from the paper:
running observation normalisation (bounds-based normalisation already exists), reward scaling by the
return's running std (fixed scales, D14), and 90% sparse init (with 2–6 inputs it leaves ~1 live input per
unit). Traces cannot be combined with replay (on-policy order needed). **Status: to review.**

**D24 — Optional parameter-free LayerNorm on the hidden layer (`norm: 1`), the target-network replacement of PQN (Gallici et al. 2025) and Stream Q(λ).**
Its theory is for wide networks; at 8–64 units it is an open question this project can answer.
**Status: to review.**

**D25 — Pendulum input encoding (θ vs cos/sin) is a task factor for every learner in `pendulum_main`.** Resolves D01. `state_dim_pendulum` was folded in. **Status: to review.**

**D26 — Step budget (`steps`, `decay_steps`, `eval_every_steps`, `shift_at_step`), evaluation exactly at step checkpoints.** Patterson et al. 2024; with an episode budget, good cart-pole balancers get ~10× more data. Episode mode still works for old configs. **Status: to review.**

**D27 — 10 paired seeds by default (20 for adaptation), 10 evaluation episodes per checkpoint; t-interval below 20 seeds, percentile bootstrap from 20; plots show every seed; `analysis.paired_diff` puts the interval on per-seed differences.** Seed k gives every learner the same episode start states (common random numbers). **Status: to review.**

**D28 — Adaptation metric: target = P − 0.1|P| (anchored to the pre-shift level P, not to the noisy minimum), 3-point moving average, non-recoveries counted (`recovered_frac`), post-shift AUC as the main number, plus a from-scratch reference experiment for forward transfer.** **Status: to review.**

**D29 — Equal-budget tuning (`scripts/tune.py`): K = 20 sampled configurations per learner, tuning seeds {1000, 1001, 1002} disjoint from reporting seeds, score = mean evaluation return (AUC), tuned once at the middle of each memory sweep.** **Status: to review.**

**D30 — Stochastic rounding of learning updates (`round_mode: 1`), plus an underflow counter for every learner (`underflow_frac` in logs and the `_underflow.png` figure).** The counter is compiled out of the M4 build (`-DTR_NO_DIAG`) so instruction counts measure the learner. **Status: to review.**

**D31 — Tile coding made cheaper without changing its output: precomputed offsets, 32-bit coordinates, mask for power-of-two M, 2-slot feature cache.** Measured on the M4 (QEMU, smoke configs, 8 tilings): 8,415 → 1,604 instructions per step. Not done: dense (unhashed) indexing where the grid fits, and the hash load-factor sweep the review suggests (a small new result). **Status: to review.**

**D32 — Table bin index uses a 32-bit division when operands fit (same result).** M4: ~830 → ~530 instructions per step. **Status: to review.**

**D33 — Go1 v2 task (`go1_gait_v2`): decisions locked to the gait phase every half cycle (CPG at 100 Hz so the half cycle is exactly 25 ticks), a "which pair swings next" bit, IMU averaged over the half cycle, roughness chosen where the open-loop trot succeeds 40–80% (`scripts/go1_difficulty.py`; success = no fall AND ≥ 2 m, because at high roughness the trot gets stuck rather than falls).** v1 is kept unchanged for comparison. Options (default off): friction randomisation, rate penalty, switch penalty, fall-penalty size. **Status: to review.**

**D34 — Normalisation reciprocal `inv_range` is truncated, so an observation exactly on a tile edge can land in the tile below (1 LSB).** Found while testing; left unchanged to keep earlier behaviour; rounding it would be a one-line change. **Status: to review.**

### Review recommendations from round 1: status

| Recommendation (research_review.md) | Status |
|---|---|
| Compressed FIFO replay | built (D35) |
| Dense (unhashed) tile indexing + hash load-factor sweep | built (D36, D37) |
| Continual backprop for the network after a shift | built (D38) |
| Scalar α_eff in a finer format (Q2.30) | built (D39) |
| Foot-contact observations, persistent / factored Go1 actions | built (D41, D42) |
| rliable IQM / performance profiles across tasks | not yet: only meaningful once all three tasks have results |
| A stream + small-buffer hybrid | not built: traces need on-policy order (D23) |


## Added 17 Nov 2026, round 2 (`docs/research_review_2.md`: algorithm trade-offs and self-adapting gaits)

All opt-in, defaults unchanged (C regression errors identical, Stage 1 gate bit-exact, Go1 v1 reproduces its
earlier distances exactly). **Status of every entry: to review.**

**D35 — Compressed replay (`replay_compact: 1`).** s stored as a uint16 fraction of [lo, hi]; s′ not stored but taken
from the next entry when it continues the same episode; action and flags in one byte. Lossy by ≤ range/65536.
Measured: 2,000 × 3-D transitions 60,676 → 22,712 B; M4 smoke network 16,400 → 6,936 B peak.

**D36 — Dense tile indexing (`tile_dense: 1`)**, collision-free when features ≥ T·(tiles+1)^d.

**D37 — Hash finaliser option (`tile_dense: 2`, murmur3 fmix32).** `scripts/tile_collisions.py`: with the pendulum
defaults (8 × 8 tiles, M = 1024, load 0.63) plain FNV puts 64% of tiles in shared slots; fmix32 51%; an ideal random
hash 47%. Default left as plain FNV (bit-exact with earlier runs); `tile_collisions.json` compares all three.

**D38 — Continual backprop for the network (`cbp_period`, `cbp_maturity`; Dohare et al. 2024)**: utility
0.99/0.01 running average of |h|·Σ|W2|; replaced units get fresh input weights and zero output weights.

**D39 — ObGD step kept in Q2.30 when it is smaller than Q16.16 can hold.** Measured effect on underflow: none for the
network, because the zeros come from the final weight increment (step × trace below 1/65536 of a weight), not from the
step. Remaining remedies: stochastic rounding (built) or wider weights (not built).

**D40 — TR_MAX_OBS 8 → 12, TR_MAX_ACTIONS 16 → 32**, for Go1 contacts + parameters and factored actions. Cost: the
agent struct grows by ~150 B on the M4 for every learner (e.g. table 9,312 → 9,460 B peak); equal for all learners.

**D41 — Go1 contact features (`obs_mode: imu4pc`)**: contact fraction of the pair that should be in stance and of the
pair that should be in swing over the last half cycle (CPG-RL: contacts were necessary to learn to walk).

**D42 — Go1 action modes:** `factored` (18: height × pitch × roll, simultaneously) and `incremental` (7: nudge
persistent clearance / posture-reflex gain / body height; parameters appended to the state).

**D43 — Self-adapting gait mechanisms as non-learning baselines (B1 Tegotae load feedback, B2 IMU posture reflex,
B3 stumble reflex, B4 all three), from Owaki et al., Kimura/Fukuoka/Cohen 2007.** Gains untuned (σ = 1, k = 0.5).
10-episode probe (`go1_difficulty.py --ladder`): success at roughness 0.08 / 0.10: B0 0.5 / 0.1, B1 0.6 / 0.2,
B2 1.0 / 0.1, B3 1.0 / 0.7, B4 0.6 / 0.1 (combining untuned reflexes hurt). A lead, not a result.

**D44 — Go1 v3 (`go1_gait_v3`): the learner tunes a self-adapting gait** (stumble reflex on, incremental actions,
10-D state with contacts and parameters) at roughness 0.10 against B0/B3/B4. A 10-D table only fits with 2 bins.

**D45 — B5: (1+1)-ES over reflex gains (`scripts/go1_es.py`), host-side, one update per episode.** Two parameter
vectors of memory. Kept as an outer-loop baseline, not a core learner (the brief retired ARS as a core comparison).

**D46 — Candidate update kernels measured on the M4 (`scripts/bench_kernels.py`, `results/m4/kernels.csv`,
`results/figures/tradeoff_map.png`): table, tile coding and network in Q16.16 / float32 / q15; recursive LSTD-Q in
float32 (k = 32 / 64 / 128); tabular Dyna-Q with n = 0 / 5 / 20 / 50 planning steps.** Kernels exclude feature
computation and action selection. QEMU counts float MACs (3 cycles) like integer ones (1 cycle), so float looks
cheaper than it is. Dyna-Q's model table is a (tabular) world model, which the brief lists as out of scope:
**scope decision needed before building Dyna into the learners.** Replay ratio (`replay_ratio.json`) uses stored real
transitions and stays in scope.


## Added 17 Nov 2026, round 3

**D47 — Processor-in-the-loop learning (`m4/pil`, `python/tinyrl/pil.py`, `scripts/pil_check.py`).** The learner runs
on the microcontroller; the laptop runs MuJoCo and relays observations, rewards and actions over a UART (QEMU: a TCP
socket with `nodelay=on`, ~2 ms per step; Nucleo: ST-LINK virtual COM port, 115200 baud). Any learner config runs
on the chip by adding `"pil": "tcp:HOST:PORT"` or `"pil": "serial:/dev/ttyACM0"`. Check: on-chip learning in QEMU
reproduces the host learning curve bit for bit for table, linear, compressed-replay network and streaming network.
The Nucleo build (USART2, DWT cycle counter, HSI 16 MHz or PLL 180 MHz) is written from the reference manual and
**has not run on hardware**. Peak stack is reported with a 256-byte painting margin (a value of 256 means ≤ 256 B).
**Status: accepted.**

**D48 — First experiments run by Claude in the cloud workspace (2 cores), recorded in `docs/results.md` with commit
and command.** Stage 2 with 10 seeds; pendulum_main as a first pass with 5 seeds (the protocol asks for 10: rerun with
`bash scripts/run_experiment.sh pendulum_main` before citing). **Status: accepted.**


## Pre-registration, 22 Nov 2026 (delegated by Rahul, written by Claude)

Written before any run of `replay_ratio`, `cartpole_main`, `adaptation_pendulum` (+ `_scratch`), `stream_ablation` or
`go1_gait_v3` (no results of these are in the repo or its local results folders). Written **after** the 10-seed
`pendulum_main` results (c3f4a55) were seen, so nothing below is a pre-registration for that experiment.
A later change to this section is a new entry that says whether the affected results had been seen.

| experiment | seeds | hyperparameters | primary comparisons (A − B, paired) |
|---|---|---|---|
| `replay_ratio` | 0–9 | from `pendulum_main_tuned` (D51) | gate: table n = 5 − n = 1, AUC. Then the same for linear and net |
| `cartpole_main` | 0–9 | tuned at the full budget (D49) | linear 1024 − table 8 bins; net 32 − linear 1024; net-stream 32 − linear 1024; final and AUC |
| `adaptation_pendulum` + `_scratch` | 0–19 | from `pendulum_main_tuned` (D51) | post-shift AUC: net-stream − linear, net-stream-cbp − net-stream; per learner: forward transfer, forgetting (D57) |
| `stream_ablation` | 0–9 | untuned, as committed (D51) | final and AUC: with − without stochastic rounding, per learner; sgd − bandit for table and linear (D54) |
| `go1_gait_v3` (only if the others are written up) | 0–9 | untuned, as committed | pre-shift final: each learner − B3 and − B4; a learner beats the reflex gait only if both intervals are above 0 |

Any other comparison is exploratory and labelled so. Each experiment is run once; seeds are never added.

**D49 — Tuning budget (resolves D29): tune at the full environment-step budget for every tuning run from now on.**
Tuning on a shorter budget favours fast starters, and the table is still learning at 100k steps. The cost is small:
the 380 runs of 300k steps in `pendulum_main` took 1,257 s at 6 jobs (`results/logs/pendulum_main.log`). So
`cartpole_main` tunes at 200k steps instead of 80k. K = 20 draws, tuning seeds 1000–1002 and one tuning point
(middle of the sweep) stay, equal for all learners. `pendulum_main` is not retuned, because its results have been
seen. Its limitations are reported: 100k-step tuning; tuned on the angle encoding only; the streaming network's 20
draws covered only 9 distinct settings, and its winner (κ = 3.0, λ = 0.6) is on the edge of that grid
(`results/tuning/pendulum_main.csv`). **Status: accepted.**

**D50 — The pendulum frontier comparisons stay at 10 seeds; what is inconclusive stays "inconclusive".**
Streaming net vs small linear and replay net vs big linear were singled out after seeing that some of their
intervals include 0. Adding seeds 10–19 for that reason would be optional stopping. No pendulum rerun.
**Status: accepted.**

**D51 — Where hyperparameters come from in experiments without a tuning step.** `replay_ratio` and both adaptation
experiments compare learners, so table, linear, net and net-stream take their values from
`configs/pendulum_main_tuned.json` (equal tuning budget), replacing the hand-set ones; net-stream-cbp takes
net-stream's. They are not retuned per replay ratio or encoding. `table-stream` has no tuned values: it is a
reference and is not ranked. `stream_ablation` stays untuned and supports only within-learner statements, no ranking
of learners. **Status: accepted.**

**D52 — Intervals on differences are Student-t 95% at every n.** `analysis.paired_diff` currently switches to a
percentile bootstrap at n ≥ 20 (through `ci95`), which would apply to the adaptation experiments. It will use t.
Per-configuration intervals keep D27. A difference counts only if its interval excludes 0; otherwise it is
"inconclusive". Forward transfer pairs the same seed across the two adaptation experiments. **Status: accepted.**

**D53 — Stage 4 kill criterion is measured on AUC.** The table with n = 5 replay updates per
step must beat n = 1 (both with the 1000-transition buffer) on AUC, with the paired interval above 0. The table arm
runs first; if it fails, Stage 4 stops and the null result is reported. This replaces "steps-to-threshold" in
`replay_ratio.json` and `stages.md`, which is undefined for runs that never reach the threshold.
Steps-to-threshold, where reported, always comes with the number of seeds that reached it. **Status: accepted.**
*Provenance (corrected 22 Nov, after the run; the entry first said "set by Rahul"):* Set via Rahul's master prompt
(Phase 3: 'paired AUC difference excludes 0'), which replaced steps-to-threshold without flagging it; Rahul confirmed
on 22 Nov after the run. Committed 57 min before any replay-ratio result. The original metric
gives the same decision (inconclusive).
Steps-to-threshold, n = 1 vs n = 5: 9/10 vs 10/10 seeds reached -200; the direction depends on how the missing seed is counted, so inconclusive.

**D54 — γ = 0 bandit baseline.** Same representation and α, no bootstrapping: `table-bandit` and `linear-bandit` in
`stream_ablation` (against `table-sgd`, `linear-sgd`), `linear-bandit` in `go1_gait_v3`. Not on cart-pole, where the
reward is +1 on every step and γ = 0 has nothing to learn. **Status: accepted, to implement.**

**D55 — Saturation counter next to the underflow counter.** Counts learning-update results clipped to the int32
range, reported as `saturation_frac`. Diagnostic only: compiled out with `TR_NO_DIAG`, no change to any learned
value, and M4 instruction counts must not change. **Status: accepted, to implement.**

**D56 — Flash size.** `measure_m4.py` already records `flash_text`. Report flash = text + data and carry it into the
summary and table, labelled as the size of the benchmark firmware (learner, harness and startup), not of the
learner alone. **Status: accepted, to implement.**

**D57 — A→B→A shift for forgetting.** `adaptation_pendulum`: A for 0–300k steps, B (+50% mass) for 300k–450k, A
again for 450k–600k. The B window and the scratch reference are unchanged. With P = final performance before the
first shift: forgetting = return on A immediately after the switch back, before any further learning, − P
(negative = forgot). Secondary: mean return over the second A window − P. `post_auc` and recovery use the B window
only. **Status: accepted, to implement.**

**D58 — Bug: the evaluation at the shift step runs on the shifted task.** In step mode `harness.run` applies the
shift and then evaluates, while `analysis.py` counts that point as pre-shift, so one of the three evaluations that
define P would come from the wrong task. Reproduced on 22 Nov with the random baseline (shift at step 2000: the
evaluation at step 2000 already differs from an unshifted run). Fix: evaluate, then shift, with a test. No reported
result is affected, because none has a shift. Must be merged before any adaptation or Go1 run.
**Status: accepted, to implement.**

**D59 — Bug: an interrupted run can be skipped on resume.** `run_sweep.py` treats an existing `seed<k>.csv` as a
finished run, but that file is written before the evaluation and meta files. Resume will key on `seed<k>_meta.json`,
which is written last. **Status: accepted, to implement before the runs.**

**D60 — Deferred: float32 learner build and Q15 weight storage**, unless the write-up leaves time. QEMU cannot show
whether float is slower (D46), and Q15 needs its own scaling and underflow study. **Status: deferred.**

## Added 22 Nov 2026, during the write-up (results had been seen; no update rule and no result changes)

**D61 — ObGD here against the paper.** The December 2024 version of Elsayed et al. uses
α / max(1, α κ δ̄ ‖e‖₁); this code uses α · min(1, 1 / (κ δ̄ ‖e‖₁)). The two are equal at α = 1, and every ObGD
configuration in `configs/` has α = 1, so no result differs. They would differ if α were tuned. The September 2026
version replaced ObGD by a per-coordinate optimiser, which is not studied here. **Status: documented, rule unchanged.**
*Added 23 Nov (D68):* a second deviation. The trace is not cut after a non-greedy action, as the authors' code does.

**D62 — The linear learner's instructions per step are path dependent.** The two-slot feature cache replaces
round-robin, so after some action sequences it evicts the state it needs next and hashes twice per step instead of
once. Measured in the benchmark window on cart-pole: 1.72 hashes per step at 256 features (4,155 instructions) and
1.00 at 1,024 (3,046). Learned values are unaffected. Removing it needs two parts, and every M4 count re-measured.
*Corrected 22 Nov:* this entry first said a least-recently-used replacement would remove it. Alone it does not: after
a random action it evicts the state needed next (1.05 hashes per step at ε = 0.05, 2.00 at ε = 1). The second part is
to look up s before the target. That pair (4fa0de6) worked but sat in the shared tr_agent.c and cost the table and
replay-network learners instructions and 8 B of stack, so it was replaced the same evening (e98a5da): the learner
gives up the slot of the state it has just learned from, inside tr_linear.c only. One hash per step plus one per
episode start on every tested path. Condition set by Rahul, met on e98a5da: seeds 0-2 of all 22 linear configs replay
byte for byte (66/66, plus 4 controls of other learners). Every reported M4 count was re-measured at 296e101.
**Status: fixed.**

**D63 — Memory claims need equal-memory comparisons (22 Nov, after the results).** Rahul set the rule and the 3-10x
band. Claude chose "about equal = within 1.5x" afterwards, having seen all 41 pairs. The cutoff matters: for final
return on the pendulum, 1.5x gives 6 higher of 8 pairs, 1.25x gives 3 of 4, 1.1x gives 1 of 1. Exploratory, not
pre-registered: every streaming-net / linear pair on the same task variant in either band, as counts, by
`scripts/compare.py <exp> --by-sram`. That is 19 pairs reported out of 41 examined, plus a check against linear's
better encoding on final return (added after a review): 86 uncorrected 95% intervals, so if no true difference existed
about two "detectable" differences per metric, about four in total, would be expected by chance.
**Status: accepted.**

**D64 — Novelty claim narrowed (22 Nov).** Elsayed, Lupu, Vasan and Mahmood (arXiv:2410.14606v3, 21 Sep 2026) learn
Pong on an ESP32-S3, and is citable from now on. Learning a gait on a microcontroller has also been done (Hackaday,
11 Dec 2016, a four-servo walker on a Cortex-M4), and this project's Go1 learner runs on the host, so that is not
claimed. The claim is the controlled memory-vs-representation comparison only. **Status: accepted.**
*Sources opened 22 Nov:* https://hackaday.com/2016/12/11/train-your-robot-to-walk-with-a-neural-network/ and the
project's own repository at commit 707c9cd, https://github.com/Counterfeiter/Q-LearningRobot/blob/707c9cd7d04364997a36479439ca966edae6762a/Src/ann.c (README: Q-learning, four servos, STM32F4
Cortex-M4; `Src/ann.c`: the network is trained on the chip).

**D65 — Go1: learner against random nudges (22 Nov, added after the results).** Exploratory. The registered bar
compares with B3 and B4 only; the random baseline turned out to score like the learners, so "beat B4" cannot be read
as learning without this comparison. The bar is unchanged and not met; no further Go1 run follows from it.
**Status: accepted.**

**D66 — Two properties of the evaluation, found in the whole-project audit (22 Nov, after all results).**
(1) Greedy evaluation breaks ties with the learner's own random-number generator, so evaluating changes the training
that follows: the same table learner and seed trained differently from episode 5 on with evaluation every 1,000 or
every 2,000 steps. Comparisons inside one experiment are unaffected (one schedule for all learners); comparisons
across experiments with different schedules are not like for like. (2) Evaluation always starts from the same 10
states (`base_seed` 10000), in tuning and in reporting, so tuning seeds separate training, not test states.
Neither is changed in the code: changing either would make the archived runs unreproducible. A future protocol should
save and restore the generator around evaluation and give tuning its own evaluation states. **Status: documented.**

**D67 — Two analysis corrections from the audit (22 Nov, after all results).** (1) Online return left in the last
episode even when the step budget had cut it short (cart-pole seed 0: a 21-step episode). It is now left out; this
moves cart-pole online means by at most 0.35 and pendulum, adaptation, ablation and Go1 not at all. (2) Configurations
that could not be measured on the M4 were estimated as arena + 220 B; the measured agent struct and stack are 688 B,
so the estimate now uses the measured overhead (cart-pole 12 bins: 166,600 B, still far above 128 KB). Baselines now
get 0 B, as they have no learner. No registered comparison changes. **Status: fixed.**

## Added 23 Nov 2026, after a review from claude.ai (all results had been seen)

**D68 — The streaming path has no Watkins cut.** D23 says the trace is cut after an exploratory action; `learn_stream`
in `tr_agent.c` never does this. The trace is reset only at episode start and at terminal states. So every
configuration with `lam` > 0 ran Q(λ) *without the Watkins cut*: the streaming network in `pendulum_main_tuned`,
`cartpole_main_tuned`, `adaptation_pendulum` (+ `_scratch`) and `go1_gait_v3`, and the `*-lam*` arms of
`stream_ablation`. The authors' code cuts it: in `stream_q.py` (github.com/mohmdelsayed/streaming-drl, branch 2026,
commit 9326fc3, opened 23 Nov) the update sets `reset = terminated or truncated or is_nongreedy`, where
`is_nongreedy` says the next action is a random one that differs from the greedy one, and `optimizer.py` then zeroes
the trace after the weight step. Check (23 Nov, two states, table and linear, α 0.5, γ 0.9, λ 0.9): after the
transition (s0, a0) → s1, a non-greedy a0 in s1 ending with reward 1 moves Q(s0, a0) from 0.225 to 0.63; with the cut
it would stay at 0.225. Without the cut, credit flows back through exploratory actions, so the traces mix the
behaviour policy's returns into the Q-learning target. How much this changed any result was not measured.
The code is **not** changed, so that every archive still replays byte for byte. Every description now says
"Q(λ) without the Watkins cut", and D61 lists it as a deviation from the paper. An opt-in cut, with a unit test that
fails without it, is part of the proposed `*_v2` experiments. **Status: documented, code unchanged.**

**D69 — The main comparisons are confounded with the learning rule.** In `pendulum_main_tuned`,
`cartpole_main_tuned`, `adaptation_pendulum` and `go1_gait_v3` only the streaming network uses traces and ObGD. The
table and linear learners are one-step Q-learners, and `tune.py` searched only their α (the streaming net: κ and λ).
So every streaming-net-versus-linear or -table difference, including all D63 tables, is a difference of
representation *and* update rule. D23's own fairness argument ("λ is a shared factor") was not applied in those
experiments. `stream_ablation` shows the size of the effect (exploratory, added after the results, untuned, paired,
10 seeds, generated by `scripts/compare.py stream_ablation`): adding λ 0.8 on top of ObGD changed AUC by table
+284.9 [+251.4, +318.5], linear +152.1 [+132.6, +171.5] and network −154.5 [−218.5, −90.6]; the one-step network
(`net-sgd`) beat the full recipe (`net-obgd-lam-ln-sr`) on final return by +149.0 [+70.7, +227.4]. The tuner chose
λ 0.6, the smallest value it offered, on both tasks, so smaller λ, including 0, was never tried for the network.
No result is changed; the report states the confound wherever these comparisons appear. A fair design is proposed as
`pendulum_main_v2` / `cartpole_main_v2` (every learner tunes λ and κ). **Status: documented.**

## Proposed 23 Nov 2026: pre-registration of `pendulum_main_v2` and `cartpole_main_v2` (not run; needs Rahul's approval)

**D70 — A like-for-like rerun of the two main comparisons.** Written after every v1 result had been seen (D68, D69),
before any v2 run. Configs `configs/pendulum_main_v2.json` and `configs/cartpole_main_v2.json`; the design is pinned by
`python/tests/test_v2_tools.py`.

*Learners.* table, linear (hashed tile coding), replay network (`net`), streaming network (`net-stream`, LayerNorm and
stochastic rounding as in v1) and a new one-step network (`net-sgd`: no replay, no target, no traces, no ObGD).
Baselines: random, and zero torque on the pendulum. On the pendulum the encoding is part of the label (`-angle`,
`-trig`), so every learner is tuned on each encoding separately (v1 tuned on the angle encoding only).

*Tuning.* K = 40 draws for every learner, 3 tuning seeds (1000-1002), score AUC, at the full step budget, evaluated on
their own start states (`eval_base_seed` 50000; reporting keeps 10000, D66). Table, linear and streaming network draw
λ from {0, 0.6, 0.8, 0.9} and κ from {0, 1.5, 2, 3}; α is drawn (log-uniform: table and linear [0.05, 1], networks
[0.002, 0.05]) only where κ = 0, and is 1 where ObGD is on (D61). The replay network cannot use traces (D23): α and
target period as in v1. `net-sgd`: α only. Every learner with traces uses the Watkins cut (`watkins: 1`, D68). Tuning
happens at the middle of each sweep, as in v1.

*Memory sweeps.* table bins and network widths as in v1. Linear pairs `tiles` with `features` (`sweep_zip`) so the
hash load, tilings × (tiles + 1)^d / features, stays near 1. It is reported for every linear configuration:

| task, encoding | tilings | features: tiles (load) |
|---|---|---|
| pendulum, angle (d = 2) | 8 | 64: 2 (1.12) · 256: 5 (1.12) · 1,024: 10 (0.95) · 4,096: 22 (1.03) |
| pendulum, trig (d = 3) | 8 | 64: 1 (1.00) · 256: 2 (0.84) · 1,024: 4 (0.98) · 4,096: 7 (1.00) |
| cart-pole (d = 4) | 16 | 256: 1 (1.00) · 1,024: 2 (1.27) · 4,096: 3 (1.00) |

*Comparisons* (paired by seed, Student-t 95%, final and AUC, `inconclusive` unless the interval excludes 0; 10 seeds,
none added):

1. Kept from v1, now pre-registered on both tasks, at the same sweep positions: cart-pole linear 1,024 − table 8 bins,
   net 32 − linear 1,024, net-stream 32 − linear 1,024; pendulum net-trig 64 − linear-trig 4,096 and
   net-stream-angle 32 − linear-angle 256.
2. New, primary: **best network against best linear at matched SRAM.** Budgets of peak SRAM measured on the M4:
   pendulum 4, 16 and 64 KB; cart-pole 6, 16 and 64 KB. In each budget, each family's entry is the configuration with
   the highest AUC on the tuning seeds and tuning evaluation states among those that fit (network family: `net`,
   `net-stream`, `net-sgd`, any encoding; linear family: any encoding). The choice uses no reporting seed. The pair
   is then compared on the reporting seeds. 6 budgets × 2 metrics = 12 intervals, uncorrected: about 0.6 would
   exclude 0 by chance if no difference existed.
3. Exploratory, reported for every learner: the λ, κ and α its tuning chose, and whether that value is on the edge of
   its grid.

*To build after approval, before any run:* the selection run (every sweep point on the tuning seeds, with the tuned
values) and comparison 2 in `scripts/compare.py`; a load-factor column for linear rows in `scripts/results_table.py`.

*Code added with this proposal* (opt-in, off by default, so every archive replays: 6 archived runs re-run byte for
byte, `cfg_id` unchanged): `watkins` in `tr_cfg_t`, with `test_watkins_cut` (fails without the cut);
`eval_base_seed` in the harness; conditional draws (`when`/`otherwise`) in `tune.py`; `sweep_zip` in `sweep.py`. The
new field makes the M4 agent struct 372 B instead of 368 B in future measurements; archived M4 tables stay valid for
their commits.

*Cost estimate* from v1 wall times (pendulum ~21 s, cart-pole ~11 s per run on this laptop, 2 jobs): tuning about
3.5 h + 1 h, selection about 0.5 h, reporting runs about 1.5 h, M4 measurement extra. Under the 12-hour rule, one
heavy run at a time, in a separate worktree. **Status: proposed, not approved, nothing run.**
## Proposed 23 Nov 2026: pre-registration of Go1 v4, headroom first (not run; needs Rahul's approval)

**D71 — Go1 v4: first show that there is something to learn.** Written after every v3 result had been seen. Why: in
v3 no learner beat B3, and random nudges scored like the learners (D65). A review from claude.ai reported its own
check (a check by the reviewer, not a result of this project, not reproduced here): at roughness 0.10 a fixed setting
chosen on the 5 evaluation terrains beat B3 there but not on 20 other terrains, and at 0.12 no setting walked. If no
reachable fixed setting beats B3 on terrain it was not chosen on, a learner that only moves these parameters cannot
either, so v3's null result may say nothing about the learners.

*Stage A, headroom* (`scripts/go1_headroom.py`, no learning). The learner's action space reaches 180 fixed settings
(clearance 0.8-1.8 step 0.2 × posture gain 0-1 step 0.2 × height −0.1-0.1 step 0.05); B3 is the default setting
(1.0, 0, 0) with the stumble reflex. Task as `go1_gait_v3`, no shift, action 0 = hold the setting all episode. At
each roughness in {0.06, 0.08, 0.10}:
1. *select*: every setting and B3 on terrains 30000-30019; take the setting with the highest mean return;
2. *confirm*: that one setting and B3 on terrains 40000-40029, paired by terrain, Student-t 95%.
Headroom exists at a roughness only if the confirmation interval lies above 0. Rule: v4 runs at the **highest**
roughness with headroom. If there is none, v4 stops there and the null is reported: no learner is run. About 2 h on
this laptop (0.65 s per episode, 2 jobs). The confirmed gap, best − B3, is reported as the most a parameter-nudging
learner could gain.

*Stage B, learners* (only if stage A finds headroom; config `go1_gait_v4` written then, from v3 with these changes):
roughness = the stage A level, no shift; learners table (2 bins), linear 1,024 and 4,096, streaming net 16 and 32,
**all with the same update rule**: λ 0.8, ObGD κ 2, α 1, Watkins cut (`watkins: 1`, D68, D69), γ 0.9. Untuned:
tuning at 390 s per Go1 run would cost about 80 h, which is over the 12-hour rule. Baselines: B3, B4, random nudges,
and *B-best* = the stage A setting, held fixed. 10 seeds. Learning curves keep 5 evaluation terrains (10000-10004).
After training each seed is scored once on **30 held-out terrains** (`heldout_episodes` 30, `heldout_base_seed`
20000: terrains 20000-20029), metric `heldout`.

*Comparisons* (paired by seed, Student-t 95%, metric `heldout`): primary, each learner − B3 (5 intervals; the bar:
a learner beats B3 if its interval lies above 0); secondary, each learner − random and each learner − B-best. With
5 uncorrected primary intervals, about 0.25 would exclude 0 by chance if no difference existed.

*Code added with this proposal* (opt-in, defaults unchanged): `init_params` in the Go1 environment (start every
episode from a given setting), `heldout_episodes` / `heldout_base_seed` in the harness with metric `heldout` in
`analysis.py`, `scripts/go1_headroom.py`; tests in `python/tests/test_go1_headroom.py`. Check: the script's B3 score
on terrains 10000-10004 is 19.2025, the archived v3 B3 evaluation. **Status: proposed, not approved, nothing run.**

## Approved 23 Nov 2026

**D72 — Rahul approved D70 and D71 as written.** Order: Go1 v4 stage A first, then the D70 runs, one heavy run at a
time. D70's still-to-build parts (selection run, comparison 2, load column) are added before any D70 run, without
changing its design. **Status: approved; D71 stage A running.**

**D73 — D70's missing parts, built before any D70 run.** `scripts/tune.py` also writes
`configs/<name>_tuned_select.json` (the tuned sweep on the tuning seeds 1000-1002 and evaluation states from 50000);
`scripts/compare.py <name>_tuned --matched` picks, per budget and family, the run with the highest AUC on those
selection runs among the runs whose M4 peak SRAM fits, and compares the pair on the reporting seeds. 1 KB = 1024 B.
It refuses to run without M4 measurements. `scripts/results_table.py` adds the hash-load column when a linear learner
uses `sweep_zip` (v2 only, so the v1 tables are unchanged). The v1 pairs of comparison 1 are registered in
`COMPARISONS`. Design unchanged. **Status: built; nothing run.**

## Added 23 Nov 2026, after Go1 v4 stage A (its results had been seen)

**D74 — Go1 v4 stage B runs at roughness 0.08.** Stage A (D71) found headroom at 0.06 and 0.08, not at 0.10
(docs/results.md, "Go1 v4, stage A"). D71's rule picks the highest such level, 0.08. `configs/go1_gait_v4.json`
follows D71 exactly: v3's task at 0.08 without the shift; table (2 bins), linear 1,024 and 4,096, streaming net 16
and 32, all with λ 0.8, κ 2, α 1, Watkins cut, γ 0.9, untuned; baselines B3, B4, random and B-best (clearance 1.0,
posture gain 0.4, height −0.05, the stage A setting at 0.08); 10 seeds; 30 held-out terrains 20000-20029.
v3's linear-bandit and B0 arms are dropped, as D71 lists neither. The comparisons are registered in
`scripts/compare.py` (`go1_gait_v4`): 5 primary (learner − B3, `heldout`), 10 secondary (learner − random,
learner − B-best). Cost: 90 runs, about 390 s each for the learners; about 2-3 h with 6 jobs. **Status: about to run.**

## Added 23 Nov 2026, after Go1 v4 stage B (its results had been seen)

**D75 — Go1 v4 result: the bar is not met, now with headroom shown.** `go1_gait_v4` ran as registered (commit
`88b4153`, 90 runs; results.md, "Go1 v4, stage B"). No learner beat B3 on the 30 held-out terrains (5 inconclusive
intervals). Secondary: every learner beat random nudges and stayed below B-best. The M4 step of
`run_experiment.sh` failed in the worktree only because the QEMU plugin was not yet built there; it was built and
`measure_m4.py` and `make_plots.py` were run on the same results, nothing re-run. `results_table.py` gains a held-out
column, shown only for experiments that have the metric. **Status: reported.**

## Added 23 Nov 2026, after the v2 runs (their results had been seen)

**D76 — v2 result and how the write-up changes.** `pendulum_main_v2` and `cartpole_main_v2` ran as registered
(commit `21b3edb`; results.md, "v2"). The report's Summary now leads with v2: tile coding beat the table on
cart-pole; the streaming network's v1 cart-pole AUC advantage did not reappear; at matched SRAM the network won only
at the pendulum's 4 KB budget. The v1 tables stay, labelled v1. Not planned in D70 but needed to check the docs:
summary CSVs for the two `_select` archives (`make_plots.py`). The tuned-values table marks a value as *edge* if it is
the first or last grid choice, or an α within 25% of a log-uniform bound. **Status: reported.**

## Proposed 23 Nov 2026: adaptation with the same choice of update rule (v2 results had been seen)

**D77 — `adaptation_pendulum_v2`.** Why: the adaptation result (streaming net − linear after the heavier rod,
+86.8) is the last headline number from a design where only the network had traces (D69). Design: v1's task,
schedule (shift at 300k, back at 450k, 600k steps), 20 seeds and from-scratch reference, unchanged; every learner's
hyperparameters copied from `pendulum_main_v2_tuned` (as v1 copied `pendulum_main_tuned`, D51), at v1's sizes:
table 11 bins and linear 1,024 features (tiles 10, load 0.95) on the angle encoding; replay, streaming and one-step
networks with 32 hidden units on trig; `net-stream-cbp` = the streaming net plus continual backprop as in v1. Note
that v2's tuning chose λ 0 and κ 0 for the streaming net on trig, so here it is a one-step network with LayerNorm
and stochastic rounding; that is the tuned learner, and the comparison keeps it. Comparisons: v1's set, with
`net-sgd` added to the per-learner rows (`scripts/compare.py`, `adaptation_pendulum_v2`): primary streaming net −
linear and cbp − streaming net on post-shift AUC, forward transfer, forgetting; secondary, the second A window.
Cost: about 1 h on this laptop with 8 jobs. **Status: proposed; runs after merge, per Rahul's "do it" of 23 Nov.**

**D78 — Exploratory: a wider λ/κ grid for tile coding on the pendulum.** Designed after the v2 results, so every
result from it is exploratory. v2's linear tuning chose the largest λ (0.9) and κ (3) on both encodings, so the grid
may have held linear back. `configs/pendulum_linear_wide.json`: v2's two linear arms only, tuned again (K = 40, same
seeds and evaluation states) with λ {0.8, 0.9, 0.95, 0.98} and κ {0, 3, 4, 6}. Reported: the chosen values, linear
1,024 (angle) wide − v2, and wide-grid linear 1,024 − v2's streaming net 32 (the 16 KB pair). Runs after D77.
About 30 min. **Status: built; runs after the D77 run.**
## Added 23 Nov 2026, after the adaptation v2 run (its results had been seen)

**D79 — Adaptation v2 result.** `adaptation_pendulum_v2` and its scratch reference ran as registered in D77 (commit
`9cd3469`). Primary: streaming net − linear, post-shift AUC +9.85 [−17.25, +36.95], inconclusive; v1's +86.8 is
therefore not supported once every learner has its own tuned update rule. The Summary, §4.4 and the README now say
so; the v1 table stays, labelled v1. **Status: reported.**
