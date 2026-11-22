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
