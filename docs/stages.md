# Stages: what to run, what to read, what to rewrite

Every stage is already built. The work in each stage is now:

1. **Understand.** Read the files listed, in order. Do the paper exercise.
2. **Rewrite.** Delete the learner's body and rewrite it yourself. The tests
   tell you when it is right.
3. **Run.** Full experiment, 5 seeds.
4. **Decide.** Go through the stage's entries in `decisions.md`.
5. **Log.** Add the numbers to `results.md` with commit and command.

Run all commands from the repo root with the venv active (`source .venv/bin/activate`).

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

**Rewrite:** empty the bodies of `cell_of`, `tr_tab_q` and `tr_tab_update` in
`tr_tabular.c` and write them again.

**Gate:** `make test` → `test_c_matches_python_bit_exact` passes (15 seed × ε cases).
If not, one more week.

**Git practice:** one branch per stage (`git switch -c stage1`). Commit small.

---

## Stage 2 — Pendulum with the table (25 Nov – 8 Dec)

**Read:** `python/tinyrl/envs/pendulum.py`, `harness.py` (training loop), `cbind.py` (ctypes boundary),
`analysis.py` (metric definitions), `m4/bench_main.c`, `m4/link.ld`, `scripts/measure_m4.py`.

```bash
python3 scripts/run_sweep.py stage2_pendulum_table --jobs 6   # 6 resolutions × 5 seeds (+ baselines)
python3 scripts/measure_m4.py stage2_pendulum_table
python3 scripts/make_plots.py stage2_pendulum_table
```

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

**Rewrite:** `active_features` and `tr_lin_update`, then `forward` and
`tr_mlp_update`. Tests: `c/tests/test_core.c` (`test_regression`) checks each
learner can fit a known function.

```bash
python3 scripts/run_sweep.py pendulum_main --jobs 6
python3 scripts/measure_m4.py pendulum_main
python3 scripts/make_plots.py pendulum_main
python3 scripts/run_sweep.py state_dim_pendulum --jobs 6   # resolves decision D01
```

**Debugging tool:** if the C network misbehaves, run the same config with
`"kind": "mlp_float"`. Float fails too → algorithm or hyperparameters. Only C
fails → fixed point.

**Gate:** if the network will not learn stably, report it and continue with two.

---

## Stage 4 — Replay buffer (23 Dec – 5 Jan)

**Read:** `tr_observe` in `tr_agent.c`; D12 in the decisions log.

```bash
python3 scripts/run_sweep.py stage4_buffer --jobs 6
python3 scripts/measure_m4.py stage4_buffer
python3 scripts/make_plots.py stage4_buffer
```

Question to answer with the plot: *how much performance does each extra kilobyte of buffer buy?*

---

## Stage 5 — Cart-pole and sweeps (6–19 Jan)

```bash
for e in cartpole_main adaptation_pendulum; do
  python3 scripts/run_sweep.py $e --jobs 6 && python3 scripts/measure_m4.py $e && python3 scripts/make_plots.py $e
done
```

Cart-pole is 4-D: watch the table's memory explode (`bins^4`). Some table
configurations will not fit in 128 KB. `measure_m4.py` reports this as a link
failure, and the plot draws those points hollow.

**Gate:** no plot means cut the sweep until one exists.

---

## Stage 6 — Write-up (20–26 Jan)

Use the `_summary.csv` tables, `results.md` and `decisions.md`. Related work: the list
in the project brief (verify every citation before using it).

---

## Stage 7 — Gait adaptation on the Go1 (after exams)

**Read:** `python/tinyrl/envs/go1.py`, `assets/unitree_go1/scene_rough.xml`, D05.

```bash
python3 scripts/view_go1.py                   # nominal trot
python3 scripts/view_go1.py --roughness 0.09  # what the shift does to it
python3 scripts/run_sweep.py go1_gait --jobs 6  # slow: about 0.8 s per episode
python3 scripts/measure_m4.py go1_gait && python3 scripts/make_plots.py go1_gait
```

First question: **does any learner beat the `nominal-trot` baseline?** In the
single-seed probes during the build, they did not clearly. Treat the task
design (actions, observation, reward) as open.

**Hardware (optional, Nucleo-F446RE):** `m4/` builds for the F446 memory map
already. To run on the board you need the STM32 startup and clock code, swap
semihosting for UART output, and read the DWT cycle counter for timing. Only
after that measurement exists can you make a timing claim.
