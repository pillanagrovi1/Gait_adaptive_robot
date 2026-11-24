# Stages: what to run and what to read

Every stage is already built. The work in each stage is now:

1. **Understand.** Read the files listed, in order. Do the paper exercise.
2. **Study.** The stage's section in `study_guide.md` (it replaces the rewrite, 22 Nov).
3. **Run.** Full experiment, with the seeds registered in `decisions.md`.
4. **Decide.** Go through the stage's entries in `decisions.md`.
5. **Log.** Add the numbers to `results.md` with commit and command.

Run all commands from the repo root with the venv active (`source .venv/bin/activate`).

**Protocol (revised 17 Nov after `docs/research_review.md`):** budgets are in environment steps,
10 paired seeds (20 for adaptation), and hyperparameters come from `scripts/tune.py`, which gives
every learner the same number of tries on separate tuning seeds. The standard loop for any experiment `E`:

```bash
python3 scripts/tune.py E               # -> configs/E_tuned.json (equal budget per learner)
python3 scripts/run_sweep.py E_tuned --jobs 6
python3 scripts/measure_m4.py E_tuned
python3 scripts/make_plots.py E_tuned
```

To compare two learners, put the interval on their per-seed differences:
`python3 -c "import sys; sys.path.insert(0,'python'); from tinyrl import analysis; print(analysis.paired_diff('E_tuned', 'RUN_A', 'RUN_B'))"`

---

## Stage 1 — Corridor (18–24 Nov)

**Goal:** know Q-learning in your fingers, and see that C and Python agree to the bit.

```
 cell:   0     1     2     3     4     5
         S ──▶ · ──▶ · ──▶ · ──▶ · ──▶ G      reward +1 on reaching G
 Q*(c, right) = 0.9^(4−c):  0.656  0.729  0.81  0.9  1.0
```

**Paper exercise** (α = 0.5, γ = 0.9, start with Q = 0, always go right):
fill in Q(c, right) after episodes 1, 2 and 3. Then check with:

```bash
cd python && python3 -m tinyrl.ref.corridor_ref    # prints every update of 3 greedy episodes
```

**Read, in order:**
1. `python/tinyrl/ref/corridor_ref.py`: `run_float` is the textbook algorithm in about 15 lines.
2. `c/include/tr_fixed.h`: Q16.16. Work out `q16_mul(Q16(1.5), Q16(-2.0))` by hand.
3. `c/include/tr_rng.h`: xorshift32.
4. `python/tinyrl/ref/fixed.py` and `TabularQ16` in `corridor_ref.py`: the same things in Python.
5. `c/src/tr_agent.c`: `tr_act`, `tr_greedy`, `learn_one`.
6. `c/src/tr_tabular.c`.

**Study (replaces the rewrite, 22 Nov):** `docs/study_guide.md`. Do the corridor exercise at the top on paper
first, then the Stage 1 section.

**Gate:** `make test` → `test_c_matches_python_bit_exact` passes (15 seed × ε cases).
If not, one more week.

**Git practice:** one branch per stage (`git switch -c stage1`). Commit small.

---

## Stage 2 — Pendulum with the table (25 Nov – 8 Dec)

**Read:** `python/tinyrl/envs/pendulum.py`, `harness.py` (training loop), `cbind.py` (ctypes boundary),
`analysis.py` (metric definitions), `m4/bench_main.c`, `m4/link.ld`, `scripts/measure_m4.py`.

```bash
python3 scripts/tune.py stage2_pendulum_table                       # alpha for the table, K = 20
python3 scripts/run_sweep.py stage2_pendulum_table_tuned --jobs 6   # 6 resolutions × 10 seeds (+ baselines)
python3 scripts/measure_m4.py stage2_pendulum_table_tuned
python3 scripts/make_plots.py stage2_pendulum_table_tuned
```

New figure: `*_underflow.png`, the share of update products that rounded to exactly zero.
Record the table's number in `results.md`: it decides whether stochastic rounding (`round_mode: 1`) is needed.

Outputs: `results/figures/stage2_pendulum_table_{curves,headline,compute}.png`: your first
learning curve and first RAM figure.

**Exercise:** predict the arena bytes for `bins = 21` before you look: 21² cells · 3 actions · 4 B
plus scratch. Check against `results/m4/stage2_pendulum_table.csv`.

**Gate:** if the emulator path costs you more than 6 h, take RAM from
`arm-none-eabi-size` and `tr_mem_bytes` and move on.

---

## Stage 3 — Linear and network learners (9–22 Dec)

**Read:** `c/src/tr_linear.c` (draw 2 tilings over a 2-D grid on paper first),
`c/src/tr_mlp.c` (derive the backward pass on paper and compare),
`python/tinyrl/ref/float_agents.py`.

**Study:** `study_guide.md`, Stage 3: `active_features` and `tr_lin_update`, then `forward` and
`tr_mlp_update`. Tests: `c/tests/test_core.c` (`test_regression`) checks each
learner can fit a known function.

**Read also:** `learn_stream` in `tr_agent.c`, `tr_trace.c`, and the "Stream Q(λ)" section of
`docs/research_review.md`. Draw the trace for 3 steps of the corridor on paper first.

```bash
python3 scripts/tune.py pendulum_main && python3 scripts/run_sweep.py pendulum_main_tuned --jobs 6
python3 scripts/measure_m4.py pendulum_main_tuned && python3 scripts/make_plots.py pendulum_main_tuned
python3 scripts/run_sweep.py stream_ablation --jobs 6     # one stream ingredient at a time, all learners
```

`pendulum_main` now runs every learner on both encodings (θ and cos/sin), which resolves D01, and
has two networks: `net` (SGD + target net + 2000-transition replay) and `net-stream` (Stream Q(λ): ObGD,
traces, LayerNorm, stochastic rounding, no replay, no target). The question for Stage 3: does
`net-stream` learn stably at 8–64 hidden units? A controlled comparison was not found in a limited search.

**Debugging tool:** if the C network misbehaves, run the same config with
`"kind": "mlp_float"`. Float fails too → algorithm or hyperparameters. Only C
fails → fixed point.

**Gate:** if the network will not learn stably, report it and continue with two.

---

## Stage 4 — Replay buffer, redefined as "replay ratio" (23 Dec – 5 Jan)

`docs/research_review_2.md`: compute is nearly free at these rates; samples and SRAM are not. So the sharper
Stage 4 question is how much a few extra replay updates per real step (from a compressed buffer) buy:

```bash
# stopped at the gate, 22 Nov: only the table arm was run
python3 scripts/run_sweep.py replay_ratio --jobs 6 --only table && python3 scripts/measure_m4.py replay_ratio --only table && python3 scripts/make_plots.py replay_ratio
```

Kill criterion (D53): run the table arm first (`python3 scripts/run_sweep.py replay_ratio --jobs 6 --only table`).
Unless n = 5 beats n = 1 on AUC (paired difference, interval above 0), stop and report the null result.
The buffer-size sweep that would have followed was removed unrun (D81).

**Read:** `tr_observe` in `tr_agent.c`; D12 in the decisions log.

---

## Stage 5 — Cart-pole and sweeps (6–19 Jan)

```bash
python3 scripts/tune.py cartpole_main
for e in cartpole_main_tuned adaptation_pendulum adaptation_pendulum_scratch; do
  python3 scripts/run_sweep.py $e --jobs 6 && python3 scripts/measure_m4.py $e && python3 scripts/make_plots.py $e
done
```

Adaptation numbers: `post_auc` (main), `recovery` (steps to get back within 10% of the pre-shift
level, missing if never), `recovered_frac`. Forward transfer = `post_auc` in `adaptation_pendulum`
minus `auc` in `adaptation_pendulum_scratch` for the same learner.

Cart-pole is 4-D: watch the table's memory explode (`bins^4`). Some table
configurations will not fit in 128 KB. `measure_m4.py` reports this as a link
failure, and the plot draws those points hollow.

**Gate:** no plot means cut the sweep until one exists.

---

## Stage 6 — Write-up (20–26 Jan)

The write-up is `report.md`. Every number in it comes from `results.md`; tables come from
`scripts/results_table.py` and `scripts/compare.py`. Related work: cite only sources that were opened.

---

## Stage 7 — Gait adaptation on the Go1 (after exams)

**Read:** `python/tinyrl/envs/go1.py`, `assets/unitree_go1/scene_rough.xml`, D05.

```bash
python3 scripts/view_go1.py                   # nominal trot
python3 scripts/go1_difficulty.py --phase-lock --episodes 30   # where does the open-loop trot fail?
# v2 was not run: go1_gait_v3 replaced it (results in results.md)
python3 scripts/run_sweep.py go1_gait_v2 --jobs 6               # slow: about 0.8 s per episode
python3 scripts/measure_m4.py go1_gait_v2 && python3 scripts/make_plots.py go1_gait_v2
```

First question: **does any learner beat the `nominal-trot` baseline?** The literature says a learner
can only win where the open-loop gait fails, so set `roughness` in `go1_gait_v2` from the difficulty
sweep (success rate 0.4–0.8) before running it. `go1_gait_v2` also locks decisions to the gait phase
and tells the learner which leg pair swings next (see D33). `go1_gait` (v1) is kept for comparison.
A negative answer ("the learner only helps once open loop fails more than X% of the time") is still a result.

**Self-adapting gaits first (no learning):** the reflex ladder shows how much the learner has to beat.

```bash
python3 scripts/go1_difficulty.py --phase-lock --ladder --episodes 30    # B0..B4 at several roughness levels
python3 scripts/go1_es.py --evals 60 --roughness 0.10                    # B5: (1+1)-ES over reflex gains
python3 scripts/run_sweep.py go1_gait_v3 --jobs 6                         # learner tunes the reflexive gait
```

In a 10-episode probe the stumble reflex alone took success at roughness 0.10 from 0.1 to 0.7: beat B3, not B0.

**Algorithm trade-off map (any stage):** `python3 scripts/bench_kernels.py` measures candidate update kernels
(Q16.16 / float32 / q15, LSTD-Q, Dyna-Q) on the M4 and draws `results/figures/tradeoff_map.png`.

**Hardware (optional, Nucleo-F446RE): processor-in-the-loop.**

```bash
python3 scripts/pil_check.py                       # in QEMU: on-chip learning == host, bit for bit
make -C m4/pil nucleo                              # USART2 (ST-LINK VCP), DWT cycles, 16 MHz HSI
st-flash write build/pil/nucleo/pil.bin 0x8000000  # or copy pil.bin to the NUCLEO USB drive
pip install pyserial
python3 scripts/pil_check.py --no-qemu --link serial:/dev/ttyACM0
```

The Nucleo build has never run on hardware: expect to debug the UART first (a logic analyser or the ST-LINK
VCP at 115200 baud). Only after this produces cycle counts can you make a timing claim, for that board and clock.
