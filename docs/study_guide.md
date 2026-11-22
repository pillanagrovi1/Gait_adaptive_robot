# Study guide

## Exercise first: the 6-cell corridor, on paper

```
 cell     0      1      2      3      4      5
       +------+------+------+------+------+------+
       |  S   |      |      |      |      |  G   |
       +------+------+------+------+------+------+
          -->    -->    -->    -->    -->
          every step: move right

 reward: 1 for entering G (the episode ends), 0 for every other step
 start:  every Q = 0        alpha = 0.5        gamma = 0.9
```

At every step, in the order the steps happen:

```
 target:   y = r                          if the step entered G
           y = r + 0.9 * max Q(next cell) otherwise
 update:   Q(cell, right) = Q(cell, right) + 0.5 * (y - Q(cell, right))
```

Fill this in by hand. Each update uses the values as they are at that moment.

| Q(cell, right) | cell 0 | cell 1 | cell 2 | cell 3 | cell 4 |
|---|---|---|---|---|---|
| after episode 1 | | | | | |
| after episode 2 | | | | | |
| after episode 3 | | | | | |

The answers are deliberately not in this file. When you have finished, this prints every update:

```bash
cd ~/Gait_adaptive_robot/python && python3 -m tinyrl.ref.corridor_ref
```

---

## Stage 1: Q-learning, from paper to C

### The picture

```
          +------------------ environment: the corridor ------------------+
          |                                                                |
          | state s                            reward r, next state s', done
          v                                                                v
   +--------------+    action a     +--------------+    +----------------------------+
   |    tr_act    | --------------> |  env moves   | -> |         tr_observe         |
   | mostly best, |                 +--------------+    |  y     = r + gamma*max Q   |
   | at times any |                                     |  delta = y - Q(s, a)       |
   +--------------+                                     |  Q    += alpha * delta     |
          ^                                             +----------------------------+
          |                  the table: cells x actions                    |
          +----------------------------------------------------------------+
```

### The idea in plain words

The table holds one number for each (cell, action): a guess of the reward still to come if you take that action
there and act well afterwards. After each step you get a slightly better guess: the reward you just saw, plus the
discounted best guess from the cell you landed in. You move the old guess part of the way towards it.

### The maths

| symbol | meaning | here |
|---|---|---|
| Q(s, a) | current guess for action a in cell s | starts at 0 |
| gamma | how much one step later is worth | 0.9 |
| alpha | how far to move towards the new guess | 0.5 |
| y | the new guess (target) | r, or r + gamma * max Q(s', .) |
| delta | how wrong the old guess was | y - Q(s, a) |
| epsilon | chance of a random action | not used on paper |

### Where it lives in the code

| what | file:function |
|---|---|
| the algorithm in floats, about 15 lines | `python/tinyrl/ref/corridor_ref.py:run_float` |
| the corridor | `c/envs/corridor.h:corridor_step` |
| choose an action | `c/src/tr_agent.c:tr_act`, `tr_greedy` |
| compute the target y | `c/src/tr_agent.c:learn_one` |
| observation to cell number | `c/src/tr_tabular.c:cell_of` |
| read one row of the table | `c/src/tr_tabular.c:tr_tab_q` |
| move the guess | `c/src/tr_tabular.c:tr_tab_update` |
| multiply without floats | `c/include/tr_fixed.h:q16_mul` |
| proof that C and Python agree | `python/tests/test_corridor_gate.py:test_c_matches_python_bit_exact` |

### Worked example: `q16_mul(Q16(1.5), Q16(-2.0))`

Q16.16 stores a number as an integer that counts steps of 1/65536.

```
 1.5   -->    1.5 * 65536  =        98304
-2.0   -->   -2.0 * 65536  =      -131072

 multiply in 64 bits        98304 * -131072  =  -12884901888     (now in steps of 1/65536^2)
 add half a step                     + 32768 =  -12884869120     (so the shift rounds)
 shift right by 16          floor(-196607.5) =       -196608     (back to steps of 1/65536)
 fits in int32? yes, so nothing is clipped

 read it back               -196608 / 65536  =          -3.0
```

Shifting a negative number right is implementation-defined in C. GCC on both x86 and ARM rounds it down, which
this project assumes and documents.

### Five interview questions

1. Why is Q-learning called off-policy?
2. Why must a time-out not be treated like reaching the goal?
3. Why are ties between equal Q-values broken at random?
4. What can go wrong when alpha * delta is smaller than 1/65536?
5. What does "C matches Python bit for bit" prove, and what does it not prove?

<br><br><br>

### Answers

1. The target uses the best next action (the max), whatever action the behaviour policy really takes next. So it
   learns about the greedy policy while following another one.
2. At the goal there is no future, so y = r. At a time-out the future exists and we only stopped watching, so the
   target still adds gamma * max Q (decision D13). Otherwise the learner would think late states are worthless.
3. All values start at 0. "Pick the first maximum" would always choose action 0 (left here) and bias early
   behaviour (decision D08).
4. The product can round to 0, so that value silently stops learning. The underflow counter measures how often;
   stochastic rounding (D30) keeps such updates alive on average.
5. It proves the C implements the same integer algorithm as the reference, on the tested seeds and settings. It
   does not prove that the algorithm is a good one, that it learns on harder tasks, or anything about speed on
   real hardware.

---

## Stage 2: the pendulum and the table's memory

### The picture

```
 state (theta, thetadot)          grid, bins = 5                 the table in SRAM
                                  thetadot
   o  <- rod tip                  4 |  .  .  .  .  .             cell 0 : Q left  none  right
    \                             3 |  .  .  .  .  .             cell 1 : Q left  none  right
     \  theta = 0 is upright      2 |  .  .  .  .  .   ----->    ...
      +                           1 |  .  .  X  .  .             cell 7 : Q left  none  right   <- X
   torque -2, 0 or +2 N m         0 |  .  .  .  .  .             ...
                                    +---------------- theta      cell 24
                                       0  1  2  3  4
```

### The idea in plain words

A table cannot store a continuous state, so it cuts each dimension into bins and treats every state in a box as
the same cell. Small boxes give a finer policy, but there are many more of them to store and to visit. Stage 2
measures that trade: return against bytes.

### The maths

```
 bin_i = floor((s_i - lo_i) * bins / (hi_i - lo_i))      the top edge goes into the last bin
 cell  = bin_0 + bins * bin_1 + bins^2 * bin_2 + ...
 table = bins^d * actions * 4 bytes
 peak  = sizeof(agent) + arena + learner stack
```

### Where it lives in the code

| what | file:function |
|---|---|
| physics and reward | `python/tinyrl/envs/pendulum.py:Pendulum.step` |
| training and evaluation loop | `python/tinyrl/harness.py:run`, `evaluate` |
| Python calls C | `python/tinyrl/cbind.py:Agent` |
| state to cell | `c/src/tr_tabular.c:cell_of` |
| bytes a config needs | `c/src/tr_agent.c:tr_mem_bytes`, `c/src/tr_tabular.c:tr_tab_mem` |
| memory and instructions on the M4 | `scripts/measure_m4.py:measure`, `m4/bench_main.c:main` |
| final, AUC, online | `python/tinyrl/analysis.py:_per_seed_metrics` |

### Worked example: which number is read, and how many bytes

bins = 5, theta = 0.5, thetadot = -3, action 2. Bounds: theta in [-pi, pi], thetadot in [-8, 8].

```
 bin_theta    = floor((0.5 + 3.1416) * 5 / 6.2832) = floor(2.898)  = 2
 bin_thetadot = floor((-3 + 8) * 5 / 16)           = floor(1.5625) = 1
 cell         = 2 + 5 * 1 = 7
 table index  = 7 * 3 + 2 = 23

 table        = 25 cells * 3 actions * 4 B = 300 B
 scratch      = 2 * 4 B (xn) + 3 * 4 B (q) =  20 B
 arena        =                              320 B
 peak         = 368 (agent) + 320 (arena) + 320 (stack) = 1008 B     as measured in QEMU
```

### Five interview questions

1. Why does the table suffer from the curse of dimensionality?
2. What is the difference between final, AUC and online return?
3. Why are peak SRAM and flash counted separately?
4. Why is an instruction count from QEMU not a time?
5. Why does every experiment include a random and, where it makes sense, a do-nothing baseline?

<br><br><br>

### Answers

1. The number of cells is bins^d. Each extra dimension multiplies both the memory and the states to visit.
2. Final: how good it ends up. AUC: how good it was on average while learning, so early learning counts. Online:
   what it earned while still exploring, which is what a robot would really experience.
3. Code and constants live in flash (512 KB here), values that change live in SRAM (128 KB). Learning needs SRAM.
4. Instructions take different numbers of cycles, and QEMU models no flash wait states, pipeline or cache.
5. A return means nothing alone. A learner has to beat what you get without learning.

---

## Stage 3: tile coding, the tiny network, and learning from a stream

### The picture

```
 TILE CODING (linear)                      TINY NETWORK
 tiling 0  |----|----|-X--|----|           x1 --+--> [ReLU] --+--> Q(left)
 tiling 1    |----|---X|----|----|         x2 --+--> [ReLU] --+--> Q(none)
 Q = sum of the active weights             x3 --+--> [ReLU] --+--> Q(right)
 nearby states share some tiles            every weight affects every state

 HOW THE UPDATE GETS STABLE
 replay net:    (s,a,r,s') -> buffer -> random batch -> update, target from a slow copy of the net
 streaming net: (s,a,r,s') -> update the trace e -> limit the step (ObGD) -> update -> forget the sample
```

### The idea in plain words

Tile coding lays several coarse grids over each other, slightly shifted. One update then also helps nearby
states. Hashing folds all tiles into a fixed number of weights, so you choose the memory. A network generalises
over the whole state space, which saves memory but makes Q-learning unstable. Replay fixes that with a buffer
(kilobytes). The streaming recipe fixes it with a step limit and a trace (one extra number per weight).

### The maths

```
 linear:    Q(s,a) = sum over T active tiles of w[tile, a]        each active w += (alpha / T) * delta
 network:   z = W1 x + b1,  h = max(0, z),  Q = W2 h + b2         each weight += alpha * delta * dQ/dweight
 trace:     e = gamma * lambda * e + dQ/dw                        (NOT cleared after an exploratory action: D68)
 ObGD:      alpha_eff = alpha * min(1, 1 / (kappa * max(|delta|, 1) * sum|e|))
            w += alpha_eff * delta * e
```

### Where it lives in the code

| what | file:function |
|---|---|
| which tiles are active | `c/src/tr_linear.c:active_features` |
| linear update | `c/src/tr_linear.c:tr_lin_update` |
| forward pass | `c/src/tr_mlp.c:forward` |
| backward pass | `c/src/tr_mlp.c:tr_mlp_update`, `hidden_grad` |
| streaming update | `c/src/tr_agent.c:learn_stream` |
| traces | `c/src/tr_trace.c:tr_strace_add`, `c/src/tr_mlp.c:tr_mlp_trace` |

### Worked example: one update of each

Tile coding: 1 input, 2 tilings of 4 tiles with offsets 0 and 0.5, weights 0, alpha = 0.5, target y = 1.

```
 u = 0.3:  tiles floor(1.2), floor(1.2 + 0.5) = 1, 1
 u = 0.4:  tiles floor(1.6), floor(1.6 + 0.5) = 1, 2        they share one tile of two

 update at u = 0.3:  delta = 1,  each of its 2 weights += 0.5/2 * 1 = 0.25
 afterwards:         Q(0.3) = 0.50    Q(0.4) = 0.25          it generalised, half as strongly
```

Network: 1 input x = 0.5, 1 hidden unit, W1 = 1, b1 = 0, W2 = 2, b2 = 0, alpha = 0.1, y = 1.5.

```
 forward:    z = 0.5,  h = 0.5,  Q = 2 * 0.5 = 1,  delta = 0.5
 gradients:  dQ/dW2 = h = 0.5   dQ/db2 = 1   dQ/dW1 = W2 * x = 1   dQ/db1 = W2 = 2
 update:     W2 = 2.025         b2 = 0.05    W1 = 1.05             b1 = 0.1
 again:      Q = 2.025 * (1.05 * 0.5 + 0.1) + 0.05 = 1.316       closer to 1.5

 with ObGD (alpha = 1, kappa = 3):  sum|e| = 4.5,  alpha_eff = 1 / (3 * 1 * 4.5) = 0.074
```

### Five interview questions

1. What do you gain and lose by hashing tiles?
2. Why does Q-learning with a network tend to be unstable?
3. What does a target network do, and what does it cost?
4. What does ObGD protect against?
5. Why is the comparison only fair with the same tuning budget and the same environment steps?

<br><br><br>

### Answers

1. Memory becomes a free choice that does not grow with the dimension. You pay with collisions: unrelated states
   can share a weight.
2. It bootstraps (learns from its own guess), off-policy (the max), with function approximation. Together these
   three can diverge: one update moves many values, including its own target.
3. It freezes the target for a number of updates. It costs a second copy of all the weights.
4. Overshooting: a step so large that the error after the update is worse than before. It shrinks the step when
   the error or the trace is large.
5. More tries or more data make any learner look better. Equal budgets keep the difference about representation.

---

## Stage 4: replay ratio, and reporting a null result

### The picture

```
                    real step
 environment -----------------------> [ buffer: last 1000 transitions, 11 KB ]
                                               |
                                               | draw n at random
                                               v
                                       n Q-learning updates        n = 1, 5 or 20

 the question fixed before the run:  is AUC(n=5) - AUC(n=1) above 0, with its whole 95% interval?

            -22.4            +24.6            +71.5
              |------|---------o----------------|        the answer: not shown, so Stage 4 stopped
                     0
```

### The idea in plain words

Compute is cheap and real steps are expensive, so it is tempting to learn several times from stored steps. The
test was fixed in advance: if 5 updates do not measurably beat 1, stop. They did not. "Not measurably" is weaker
than "replay is useless": the interval still allows a useful gain, and also a small loss.

### The maths

```
 buffer bytes (compressed) = cap * 2d + cap * 1 + cap * 4 + 3 * 4d
 paired differences        d_k = AUC(n=5, seed k) - AUC(n=1, seed k)
 95% interval              mean(d) +/- t * sd(d) / sqrt(number of seeds)       t = 2.262 for 10 seeds
```

### Where it lives in the code

| what | file:function |
|---|---|
| store, compress, sample | `c/src/tr_agent.c:observe_compact`, `encode_obs`, `decode_obs` |
| the paired interval | `python/tinyrl/analysis.py:paired_diff`, `t95` |
| the printed comparison | `scripts/compare.py:main` |
| the rule | `docs/decisions.md`, D53 |

### Worked example: the gate by hand

```
 buffer:  1000*2*3 + 1000 + 1000*4 + 3*4*3 = 11036 B

 AUC differences, seeds 0-9 (rounded):   -72  117  -11  -32  126   70   54  -24   11    7
 sum = 246        mean = 24.6
 sd  = 65.3       standard error = 65.3 / 3.162 = 20.7
 interval = 24.6 +/- 2.262 * 20.7 = [-22, +71]      0 is inside: inconclusive
 (with unrounded differences: [-22.4, +71.5], the numbers in results.md)
```

### Five interview questions

1. Why pair the runs by seed?
2. Why not add ten more seeds when the interval just includes 0?
3. Why is n = 1 in this experiment not the plain table learner?
4. Does this result show that replay does not help?
5. What do you do when some seeds never reach a threshold?

<br><br><br>

### Answers

1. Both arms get the same start states per seed, so shared luck cancels in the difference.
2. Looking first and then deciding to collect more data (optional stopping) raises the false-positive rate.
3. Its one update comes from the buffer, never from the transition just observed, and it pays for the buffer.
4. No. It did not show a gain for one table size on one task. The interval reaches +72.
5. Count them and say so. Dropping them can flip the direction: here it was 9/10 against 10/10 seeds.

---

## Stage 5: cart-pole and adapting to a change

### The picture

```
 return
   ^          A (normal rod)        |   B (+50% mass)    |   A again
   |                    ___P________|                    |x______________
   |              _____/            |\        _______    |
   |        _____/                  | \______/           |
   |  _____/                        |                    |
   +--------------------------------+--------------------+----------------> environment steps
   0                              300k                 450k            600k

   post-shift AUC = mean over B        x = return right after the switch back, before relearning
   forgetting     = x - P              forward transfer = post-shift AUC - AUC of a fresh learner on B
```

### The idea in plain words

Cart-pole has four state numbers, so it shows the table's problem: 12 bins already need more than the chip's
SRAM. Adaptation asks three separate things. How well do you cope after a change? Did old knowledge help,
compared with starting fresh? And what did you lose of the old task while learning the new one?

### The maths

```
 P                = mean of the last 3 evaluations before the first shift
 target           = P - 0.1 * |P|
 recovery         = steps after the shift until the 3-point moving average reaches the target
 post-shift AUC   = mean evaluation return in the B window
 forgetting       = (return on A right after the switch back) - P       negative = forgot
```

### Where it lives in the code

| what | file:function |
|---|---|
| cart-pole physics | `python/tinyrl/envs/cartpole.py:CartPole` |
| making the rod heavier | `python/tinyrl/envs/pendulum.py:Pendulum.set_shift` |
| when to shift and switch back | `python/tinyrl/harness.py:run` |
| the metrics | `python/tinyrl/analysis.py:_per_seed_metrics` |

### Worked example: numbers made up for practice

```
 P = -200, so the target is -220.    Evaluations in B, every 5000 steps: -600 -300 -240 -210 -180

 3-point averages:  -380  -250  -210          the third reaches -220
 it ends at the 5th evaluation:               recovery = 25000 steps
 post-shift AUC = (-600-300-240-210-180)/5  = -306
 right after switching back it scores -260:   forgetting = -260 - (-200) = -60

 cart-pole table, 12 bins: 12^4 cells * 2 actions * 4 B = 165888 B  >  131072 B of SRAM
```

### Five interview questions

1. Why must the evaluation at the shift step run before the shift?
2. Why anchor recovery to P and not to the worst point after the shift?
3. Why is a from-scratch run needed to talk about transfer?
4. Why can forgetting only be seen with A, B, A?
5. Why does the 500-step limit make good learners hard to tell apart?

<br><br><br>

### Answers

1. Otherwise P contains a score from the new task. That was a real bug here (D58), found before the runs.
2. The worst point is noisy, and a learner that crashes deeper would get an easier target.
3. Doing well on B could just mean B is easy. Transfer is the advantage over starting fresh on B.
4. With A then B you never test A again, so you cannot know what was lost.
5. All of them hit 500, so their differences disappear at the ceiling. AUC still separates them.

---

## Stage 6: turning measurements into claims

### The picture

```
 return
   ^                                   o C        a point is on the FRONTIER if nothing
   |              o A                             with the same or less memory is better
   |
   |                        o B                   A and C are on it. B is not: A is better and smaller.
   +-------------------------------------------> peak SRAM

 raw files --> summary CSV --> results_table.py / compare.py --> results.md --> report.md
   archived      committed        refuse bad provenance           numbers         claims
```

### The idea in plain words

A claim is only as good as the path back to its file. Every number in the report comes from a table in
`results.md`, every table from a script, every script input from an archive with a commit and a config. A
difference counts only if its interval excludes 0. Everything else is called inconclusive.

### The maths

```
 A higher       mean - half width > 0
 A lower        mean + half width < 0
 inconclusive   anything else
```

### Where it lives in the code

| what | file:function |
|---|---|
| one row per config | `python/tinyrl/analysis.py:summarize` |
| the tables | `scripts/results_table.py:main` |
| the differences and the verdict | `scripts/compare.py:verdict` |
| the headline plot | `python/tinyrl/plots.py:headline` |

### Worked example: three verdicts

```
 cart-pole, linear 1024 - table 8 bins, final:    +141.3 +/- 44.4  -> [+96.9, +185.7]   A higher
 cart-pole, replay net 32 - linear 1024, AUC:     -114.4 +/- 46.5  -> [-160.9, -67.9]   A lower
 cart-pole, stream net 32 - linear 1024, final:    +13.5 +/- 50.8  -> [-37.2, +64.3]    inconclusive
```

### Five interview questions

1. What is the difference between pre-registered and exploratory?
2. Why can two overlapping intervals still hide a real paired difference?
3. What does byte-identical reproduction prove?
4. Why does "learns faster" always come with an instruction count here?
5. What can this project not claim?

<br><br><br>

### Answers

1. Pre-registered: written down before seeing the data. Exploratory: anything else. It can suggest, not confirm.
2. Each interval contains the luck the arms share. The difference removes it.
3. Determinism on that platform. A wrong program can be perfectly reproducible.
4. Faster in samples can cost 12-75x the compute per step. Without both numbers the trade is hidden.
5. Timing, power, behaviour on real hardware or a real robot, and being the first to do this.

---

## Stage 7: a gait that tunes itself (Unitree Go1, in simulation)

### The picture

```
 learner, every half stride (0.25 s)
    |  one nudge: clearance +/-, posture gain +/-, body height +/-, or keep
    v
 gait parameters --> CPG (rhythm, 100 Hz) + stumble reflex --> joint targets --> PD control --> robot on rough ground
    ^                                                                                               |
    +---------- IMU average + which pair swings next + foot-contact mismatch + the parameters ------+

 baselines:  B0 plain trot      B3 trot + stumble reflex      B4 trot + all reflexes
 to count as a win:  learner - B3 above 0  AND  learner - B4 above 0
```

### The idea in plain words

The learner does not learn to walk. A rhythm generator already trots and a reflex lifts a foot that stumbles.
The learner only nudges three settings of that gait, slowly enough for a microcontroller. So beating the plain
trot would be too easy a bar. It has to beat the gait that already adapts by itself.

### The maths

```
 each leg:   phase += 2 pi * f * dt          f = 2 Hz, the diagonal pairs half a cycle apart
 thigh     = thigh0 + A * cos(phase) - 0.5 * extension
 calf      = calf0  - lift * max(0, sin(phase)) + extension
 reward    = 10 * metres forward - 2 * (roll^2 + pitch^2) - 5 on a fall
```

### Where it lives in the code

| what | file:function |
|---|---|
| rhythm, reflexes, actions | `python/tinyrl/envs/go1.py:_advance_phase`, `_targets`, `_action_offsets` |
| what the learner sees | `python/tinyrl/envs/go1.py:_obs` |
| the experiment and baselines | `configs/go1_gait_v3.json` |
| how hard the terrain is | `scripts/go1_difficulty.py` |

### Worked example: timing and one reward

```
 half stride = 1 / (2 * 2 Hz) = 0.25 s = 25 ticks of 0.01 s
 episode     = 10 s / 0.25 s  = 40 decisions

 one decision: 0.10 m forward, roll 0.1 rad, pitch 0.2 rad, no fall
 reward = 10 * 0.10 - 2 * (0.01 + 0.04) = 0.9        the learner sees it times 0.1 = 0.09

 clearance: starts at 1.0, "up" twice -> 1.4         it cannot go above 1.8
 table on this 10-number state, 2 bins: 2^10 cells * 7 actions * 4 B = 28672 B
```

### Five interview questions

1. Why lock decisions to the gait phase?
2. Why put the gait parameters into the state?
3. Why is beating the open-loop trot not enough?
4. Why can the table only have 2 bins here?
5. What would still be needed before any claim about a real robot?

<br><br><br>

### Answers

1. The same tilt needs a different reaction depending on which legs are about to swing.
2. A nudge has an effect that depends on the current setting. Without it the state would not be Markov.
3. A fixed reflex beats it without any learning or SRAM for learning.
4. The state has 10 numbers, so 3 bins would already be 59049 cells, about 1.6 MB.
5. Cycle counts on the real board, the sensor noise and delays of the real robot, and the experiment itself.
