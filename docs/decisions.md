# Decisions log

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

### Review recommendations not implemented yet

| Recommendation (research_review.md) | Why not now |
|---|---|
| Compressed FIFO replay (int16 states, next state implied) for a stream + small-buffer hybrid | Needs a buffer format change; Stage 4 work |
| Dense (unhashed) tile indexing + hash load-factor sweep | Small new result; good Stage 3 exercise |
| Continual backprop for the network after a shift | Adaptation work, Stage 5 |
| Scalar α_eff in a finer format (Q2.30) | Wait for the underflow numbers on real runs |
| Foot-contact observations, persistent / factored Go1 actions | Breaks "IMU only" / larger design change; Stage 7 |
| rliable IQM / performance profiles across tasks | Only meaningful once all three tasks have results |
