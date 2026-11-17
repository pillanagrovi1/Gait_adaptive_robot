<!-- Second literature review, 17 Nov 2026: on-board algorithm trade-offs, number formats, memory tricks, self-adapting gaits, on-robot learning. Measured follow-ups: results/m4/kernels.csv, scripts/bench_kernels.py, scripts/go1_difficulty.py --ladder, scripts/go1_es.py. Implementation status: docs/decisions.md D35-D46. -->

# Spend idle cycles to save samples

On a Cortex-M4F that makes one gait decision every 0.25 s, **compute is almost free and SRAM and real samples are scarce**. The algorithm choices that help most therefore spend spare cycles to save samples while keeping memory bounded. The three learners stay the project's core: the Q-table, the hashed tile-coded linear model, and the tiny network with optional Stream Q(λ)/ObGD, all in Q16.16 with static allocation. Everything in this report is a complement to them. The strongest complement is **replay or Dyna-style planning**, which runs n extra updates per real step from stored or modelled transitions. In Sutton and Barto's maze, 5 planning steps per real step cut learning from about 25 episodes to about 5, and 50 steps cut it to about 3 ([White course notes](https://marthawhite.github.io/rlcourse/docs/2019/w9.pdf)). It can be applied to all three learners equally, and it fits the project's existing "Buffer" stage. Second is a **float32 recursive LSTD-Q reference arm** for pendulum and cart-pole. Least-squares methods beat TD per sample, and the gap grows with problem size ([Geramifard et al. 2006](https://www.ece.uvic.ca/~bctill/papers/learning/Geramifard_etal_2006.pdf)), but they need O(k²) memory and floating-point range. On number formats, the M4 Technical Reference Manual shows that **Q16.16 is not slower than float32**: the integer 64-bit multiply-accumulate takes 1 cycle and the float one takes 3 ([Cortex-M4 TRM](https://documentation-service.arm.com/static/5f19da2a20b7cf4bc524d99a)). Fixed point's real weakness is range: small updates round to zero. For the Go1, the literature says the learner must beat **hand-tuned reflexes, not just the open-loop trot**. Tekken walked outdoors over 30 mm pebbles and 14° slopes with constant reflex gains ([Kimura, Fukuoka & Cohen 2007](https://www.cvl.iis.u-tokyo.ac.jp/class2017/2017w/papers/4.Robotics/AdaptiveDynamicWalking.pdf)). Finally, every verified real-robot gait learner, from the 2004 AIBO to the 2022 Unitree A1, ran its learning on a host PC or GPU. What is realistic on an MCU is adjusting a few reflex gains on top of a good prior, measured on chip. Learning a gait from scratch is not.

## How to read this report, and the budget that shapes it

Each main section has two parts. **"What the sources show"** contains only claims that the research notes checked against a fetched source, cited inline. **"What it means here"** is interpretation: byte counts, cycle guesses, rankings and design advice. Treat all of it as hypotheses to measure. In line with the project rule, no cycle, timing or power figure for the STM32F446RE is stated here as fact. Cycle numbers below are either from ARM's manual (per instruction) or are labelled estimates.

The budget arithmetic explains most of what follows. It is simple division, not a citation.

```
One gait decision every half cycle of a 2 Hz trot = 4 decisions/s

 CPU cycles between decisions   180 MHz / 4 Hz = 45,000,000
 a learning step (estimate)     ~20,000-40,000 cycles        -> < 0.1 % of the budget
                                |▏                                          | 45 M
 SRAM                           128 KB total (112 KB SRAM1 + 16 KB SRAM2), no CCM
                                |██████████████████████████████████████████| 128 KB  <- binding
 Real samples per 20 s episode  80 transitions; ~14,400 per hour of walking  <- binding

=> the right trade: spend cycles (abundant) to save samples (scarce), without spending SRAM.
```

| Term | Plain meaning |
|---|---|
| Planning / Dyna | Extra value updates per real step, using transitions drawn from a stored list or a learned model ("imagined" experience) |
| Replay ratio (UTD) | Updates per real environment step. 1 = pure streaming; 20 = Walk in the Park |
| Prioritized sweeping | Dyna that spends planning updates on states whose values changed most, using a priority queue |
| LSTD / LSTD-Q / LSPI | Least-squares TD: solve for the best linear weights directly from all samples instead of nudging them. Needs a k×k matrix for k features |
| RLS | Recursive least squares: updates the inverse of that matrix one sample at a time (Sherman-Morrison) |
| ES / (1+1)-ES / CMA-ES / ARS | Gradient-free search: perturb a parameter vector, run whole episodes, keep what scored better |
| BO | Bayesian optimisation: fit a Gaussian-process model of "score vs parameters" and choose the next trial cleverly |
| Q16.16 | 32-bit fixed point: 16 integer bits and 16 fraction bits. Range ±32,768; smallest step 2⁻¹⁶ ≈ 1.5×10⁻⁵ |
| q15 / int8 | 16-bit / 8-bit fixed point. The M4 can do two 16-bit multiply-accumulates in one instruction (SMLAD) |
| CPG | Central pattern generator: the oscillator that produces the rhythmic trot |
| Tegotae | Load-based phase feedback: a leg's oscillator slows while that foot is carrying weight |
| Reflex | A fixed, fast sensor-to-action rule (for example, "if the body rolls right, extend the right legs") |
| QEMU / DWT CYCCNT | The emulator (counts instructions, not cycles) / the hardware cycle counter on the Nucleo |

The project's structure, and how the ideas in this report attach to it, looks like this:

```
              ┌──────────────────────── CORE (unchanged) ────────────────────────┐
              │  Q-table   |  hashed tile-coded linear Q  |  tiny net (+Stream Q(λ)) │
              │  Q16.16, static arenas, same max-based Q-learning target          │
              └──────────────────────────────────────────────────────────────────┘
   wraps the core ▲               bounds the comparison ▲          sets the bar ▲
 ┌────────────────┴───┐  ┌──────────────────────────────┴─┐  ┌──────────────┴─────────┐
 │ replay / Dyna:     │  │ reference arms (not core):     │  │ Go1 non-learning ladder │
 │ n updates per step │  │ LSTD-Q f32; episodic ES or BO  │  │ B0 open loop ... B4     │
 └────────────────────┘  └────────────────────────────────┘  └─────────────────────────┘
          implementation knobs under all of it: number format, memory placement
```

## Planning and least squares convert idle cycles into samples

### What the sources show

**Model-based planning** is the best-documented way to trade compute for samples. In the Dyna maze, "the nonplanning agent took about 25 episodes to reach (ε-)optimal performance, whereas the n = 5 agent took about five episodes, and the n = 50 agent took only three" ([White course notes](https://marthawhite.github.io/rlcourse/docs/2019/w9.pdf)). The step from n = 5 to n = 50 bought far less than the step from 0 to 5, so returns diminish quickly. Dyna-Q+ adds an exploration bonus for state-action pairs not tried recently, aimed at environments that change ([White course notes](https://marthawhite.github.io/rlcourse/docs/2019/w9.pdf)). Prioritized sweeping's title promises "less data and less real time" ([Moore & Atkeson 1993](https://link.springer.com/article/10.1007/BF00993104)), but the notes could not retrieve its numbers. Linear Dyna stores an n×n forward matrix per action. On mountain car it showed a "clear early learning-rate advantage over Sarsa". The authors note that with linear models and one-step TD, model-based and model-free methods reach the same asymptote, and on the Boyan chain Dyna started *slower* because its early models were noisy ([Sutton et al. 2008](https://arxiv.org/pdf/1206.3285)). PILCO, the data-efficiency ceiling, learned a real cart-pole swing-up from about **17.5 s of interaction**. It used Gaussian-process dynamics with a per-step cost of O(n²E²D) in the number of stored points n ([Deisenroth et al.](https://arxiv.org/pdf/1502.02860)), far beyond an MCU.

**Least-squares TD** buys samples with quadratic memory. LSTD costs O(n²) per step even with incremental updates, while TD(0) costs O(n). On Boyan chains both LSTD and iLSTD clearly beat TD(0), and **the gap widened as the problem grew**. TD was about 2× cheaper per step but did worse even with twice the data ([Geramifard et al. 2006](https://www.ece.uvic.ca/~bctill/papers/learning/Geramifard_etal_2006.pdf)). LSPI learned inverted-pendulum and bicycle control from "a relatively small number of trials where actions are selected randomly". On the bicycle, Q-learning with the same features, with or without experience replay, "rarely balanced for more than a small fraction of the time needed" ([Lagoudakis & Parr 2003](https://jmlr.org/papers/v4/lagoudakis03a.html)). Kalman TD has the same O(p²) memory. It also tracked a reward change that LSTD failed to follow ([Geist & Pietquin](https://arxiv.org/pdf/1406.3270)), which bears on the project's adaptation axis.

**Gradient-free episodic search** has the smallest memory and the best real-robot gait record, but it learns once per episode, not once per step. Kohl and Stone tuned 12 AIBO gait parameters by finite-difference policy gradient. They ran 15 perturbed policies per iteration, each evaluated 3 times, for **just over 1,000 traversals in about 3 hours** on 3 robots. The result, 291 ± 3 mm/s, beat their own hand-tuned 245 mm/s and the best hand-tuned gait of the time, 254 mm/s. The robots timed themselves, but a host PC computed the updates ([Kohl & Stone 2004](https://www.cs.utexas.edu/users/AustinVilla/papers/icra04.pdf)). Bayesian optimisation found "a fast and robust gait in less than 30 experiments" for 4 parameters of the Fox biped. It reached 0.337 m/s, against 0.148 m/s for grid search, and an 8-parameter run found a stable gait after about 80 experiments ([Calandra et al.](https://www.ias.tu-darmstadt.de/uploads/Site/EditPublication/Calandra2015a.pdf)). ARS with linear policies matched state-of-the-art sample efficiency on some MuJoCo tasks but needed 2-3× more rollouts on Walker2d and Ant, and it "exhibits a large variance" across 100 seeds ([Mania et al. 2018](https://ar5iv.arxiv.org/html/1803.07055)). Separable CMA-ES has "linear time and space complexity" in the number of parameters ([Ros & Hansen](https://core.ac.uk/download/50252948.pdf)).

**Recent MCU work is almost entirely inference-only.** Micro neural policies ran on a Cortex-M33 in **40-304 B of RAM and 10-259 µs**, after training on an HPC cluster ([arXiv 2610.08541](https://arxiv.org/html/2610.08541v1)). A microrobot study ran an int8 MLP of 51.1 kB on a Cortex-M0 at 47.62 Hz, after PPO training in Isaac Gym ([arXiv 2512.24740](https://arxiv.org/pdf/2512.24740)). The only true on-device RL training found is tabular, for example a 420 B Q-table on an ESP32 from a low-evidence venue ([ESP32 PID Q-learning](https://joscex.shmpublisher.com/index.php/journal/article/download/8/29/656)).

### What it means here

The trade-off map below combines these facts with the notes' byte estimates. Bytes assume 4 B per value unless marked. "Compute per real step" is asymptotic. Every cycle figure is an estimate to be replaced by the benchmarks in a later section.

| Family | SRAM (example at project sizes) | Compute per real step | Learns | Data-efficiency evidence | Number format | Role here |
|---|---|---|---|---|---|---|
| **Q-table** (core) | S×A; 729×6 = 17.5 KB | O(A) | every step | baseline | Q16.16 excellent | **core** |
| **Hashed tile-coded linear Q(λ)** (core) | M×A + ~1.1 KB sparse trace | O(T·A) | every step | generalises to neighbours | Q16.16 good; α/T updates underflow first | **core** |
| **Tiny net + Stream Q(λ)/ObGD** (core) | 6-64-6: 3.35 KB weights + 3.35 KB trace | O(P) | every step | matched DQN on MinAtar at ≥128 units ([Elsayed et al.](https://arxiv.org/pdf/2410.14606)); untested at 8-64 | Q16.16 workable; norms need √ and ÷ | **core** |
| Replay (compressed FIFO) | ~11 B/transition; 1,000 ≈ 11 KB | O(n × learner step) | every step + n replays | UTD = 20 key on real A1 | same as learner | **complement, all 3 learners** |
| Tabular Dyna-Q / Dyna-Q+ | ~3× table; pendulum 21²×3 ≈ 15.9 KB; 729×6 ≈ 52 KB; 729×18 ≈ 157 KB (does not fit) | O(n·A) | every step + n plans | maze 25 → 5 → 3 episodes | Q16.16 excellent | **complement, table** |
| Prioritized sweeping | Dyna + 10-35 KB queue and predecessors | O(n·(A + preds)) | every step | title claim only | good | optional |
| Linear Dyna (feature space) | A·f²: f = 32, A = 3 → 12.7 KB; f = 64, A = 18 → ~300 KB | O(f²) | every step | early-learning gain only | model iteration risky | skip for hashed features |
| Learned linear dynamics + Dyna | A·d²: d = 10, A = 18 → ~8 KB | O(n·(A·d² + Q eval)) | every step + n plans | no direct source | f32 for the RLS fit | idea only; scope check |
| **Recursive LSTD-Q / LSPI** | k² f32: k = 64 → 16 KB; 128 → 64 KB; 179 → all 128 KB | O(k²) (RLS) | every step | beats TD, gap grows; LSPI beat Q-learning + replay | **f32 required** | **reference arm** |
| iLSTD / KTD / GPTD | k² or m²; m = 100 → ~80 KB | O(k²) | every step | KTD tracks changes | f32 | skip |
| (1+1)-ES, sep-CMA, finite-difference PG, ARS | O(n): < 2 KB even at n = 180 | O(n) to act | **per episode** | AIBO 1,000 runs; ARS high variance | excellent | **Go1 baseline B5** |
| Full CMA-ES | n²: n = 12 → ~1.2 KB; n = 180 → ~130 KB | O(n²) per sample | per episode | — | f32 | small n only |
| BO (GP) | N²: N = 100 → ~45 KB f32 | O(N³) per episode | per episode | < 30 trials (4 params) | **f32 required** | host-side ceiling |
| Per-step linear actor-critic | d·A, same as linear Q | O(d·A) + exp per action | every step | none verified | exp needs a LUT | skip |
| PILCO / GP models | n² GP + optimiser | very high | per episode | 17.5 s real cart-pole | float only | citation only |

Plotted roughly, the families fall into three groups. The positions are judgement, not measurement.

```
fewer real samples needed
      ▲
 high │                                BO(f32, ~45KB)        PILCO (>>128 KB, infeasible)
      │                    LSTD-Q (f32, 16-90 KB)
      │      Dyna-Q / replay (+11-35 KB)
      │                 learned-model Dyna (~8 KB + Q)
      │   tile-coded Q(λ)     tiny net + Stream Q(λ)
      │   Q-table
      │ (1+1)-ES / ARS / finite-diff PG  <- tiny memory, but one update per EPISODE
  low └──────────────────────────────────────────────────────────────────────────────► SRAM
          < 2 KB         ~5-20 KB           ~20-60 KB           ~60-128 KB      > 128 KB
```

Three conclusions follow. First, **replay is the cheapest compute-for-samples lever, and it is fair to all three learners.** A ring buffer of past transitions, replayed n times between decisions, is Dyna with the buffer acting as the model. It costs the same SRAM whichever learner sits on top. It also turns the project's Stage 4 ("Buffer") from "how big a buffer" into "how many replays per real step", the same axis as Walk in the Park's UTD = 20. One interaction needs care. Replayed transitions are not in time order, so they should be one-step (λ = 0) updates, while the live stream keeps its Watkins trace. Mixing the two has no source behind it and needs a test. A second caveat: a deterministic tabular model (store the last s′ and r for each (s, a)) is wrong for rough terrain and for coarsely binned cart-pole states, because the same cell can lead to different next cells. Replaying real transitions avoids that bias; a last-seen model does not.

Second, **least-squares methods are worth one reference arm, not a fourth learner.** LSTD-Q fits only where k stays near 64-128, which means dense features (RBFs or a small grid) on pendulum and cart-pole. With 18 Go1 actions, k = 128 leaves about 7 features per action, too few. Its recursive inverse starts from (1/δ)·I with small δ. That produces huge entries first and tiny ones later, a dynamic range that Q16.16 (±32,768 and 1.5×10⁻⁵) cannot hold, so it must run in float32. Its value to the project is as a ceiling: "how many samples would a linear learner need if it used them optimally?"

Third, **episodic search is the honest competitor for Go1, not for pendulum.** It needs under 2 KB, so on the headline memory axis it would beat every core learner. But it cannot adapt within an episode, and at 20 s per episode, 1,000 episodes is about 5.5 hours of simulated walking. Its natural role is the Go1 baseline B5 described below.

Two scope warnings come from the project brief. Tabular Dyna-Q with a model table, and a learned linear dynamics model, are small learned models of the world, and "world models" are listed as out of scope. Replay of real transitions is not a model and stays in scope; the other two need an explicit entry in the decisions log before any work starts. Likewise, the brief retired the ARS comparison "by stored size". Using an episodic hill-climber as a *sample-count* baseline for Go1 is a different question, but it reopens that thread. It belongs in optional Stage 7 only, and it also needs a decisions-log entry.

## Float32 is nearly free on the M4F; Q16.16 fails on range, not speed

### What the sources show

The Cortex-M4 TRM gives these cycle counts. Float VADD/VSUB/VMUL take **1 cycle**, but float multiply-accumulate (VMLA/VFMA) takes **3**, and VDIV/VSQRT take **14**. On the integer side, SMULL/SMLAL (32×32→64) take **1 cycle**, the dual 16-bit MACs SMLAD/SMLALD take **1 cycle**, and SDIV takes 2-12 cycles ([Cortex-M4 TRM](https://documentation-service.arm.com/static/5f19da2a20b7cf4bc524d99a)). SDIV divides only 32-bit values, and a 64-bit division compiles to a library call ([compiler-rt](https://codebrowser.dev/llvm/compiler-rt/lib/builtins/arm/aeabi_ldivmod.S.html)). The single measured comparison found was a forum micro-benchmark, not peer-reviewed: a Q31 MAC with `__SMMLA` took **9 cycles** and a float `acc += a*b` took **10**. An Arm engineer in the same thread showed that float code can pick up extra register moves, or become a library call if the FPU is not enabled ([ARM forum](https://community.arm.com/support-forums/f/armds-forum/47684/on-cortex-m4f-microcontrollers-is-fixed-point-math-faster-or-floating-point)). Using the FPU inside an interrupt adds 17 words of stack frame, and that state is saved lazily, only if the handler actually touches the FPU ([ARM AN298](https://documentation-service.arm.com/static/5f0352c0cafe527e86f5bc66)).

Practitioners who train on MCUs mostly choose float. RLtools reported "the first-ever training of a deep RL agent for continuous control on a microcontroller", on a Teensy 4.1 (Cortex-M7). Its inference benchmark on an STM32F405 (Cortex-M4F, 168 MHz) ran a 13-64-64-4 network in **743 µs with generic code and 293 µs optimised** ([Eschmann et al., JMLR](https://jmlr.org/papers/volume25/24-0248/24-0248.pdf)). The PULP-TrainLib authors argue that float training avoids "accuracy degradations due to limited integer ranges" ([Nadalini et al. 2023](https://arxiv.org/pdf/2305.19167)). Their float32 training kernels on an STM32L476 (M4F) reached about **0.10-0.12 MACs per clock** ([PULP-TrainLib](https://cris.unibo.it/bitstream/11585/900686/6/PULP-TrainLib%20-%20Springer%20Version.pdf)). PocketNN names integer overflow in backpropagation as the key obstacle to integer-only training. It works around it with direct feedback alignment and lands 1-2 points below float on MNIST and Fashion-MNIST ([PocketNN](https://arxiv.org/abs/2201.02863v5)). Int8 training on the device works only with gradient rescaling. In MCUNetV3, plain int8 SGD scored 64.9%, Quantisation-Aware Scaling 73.5%, and fp32 73.3% ([Lin et al. 2022](https://arxiv.org/pdf/2206.15472)). In 16-bit fixed point, stochastic rounding decided whether training worked at all ([Gupta et al. 2015](https://proceedings.mlr.press/v37/gupta15.html)). CMSIS-NN's SIMD kernels delivered **4.6× throughput and 4.9× energy** gains, but that was measured on a Cortex-M7, and the library covers inference only. Its docs describe int8/int16 kernels with a DSP-extension path for the M4 and say nothing about backward passes ([CMSIS-NN paper](https://arxiv.org/pdf/1801.06601); [CMSIS-NN docs](https://arm-software.github.io/CMSIS-NN/latest/)).

### What it means here

Instruction costs predict a well-written Q16.16 dot product at about 1 cycle per MAC plus loads, and a float one at about 3 plus loads. Loads dominate both, so the real gap is probably small. That matches the forum's 9 versus 10 cycles. The figures from optimised versus generic code (about 9.5 versus about 24 cycles per MAC, derived from the RLtools timings) suggest that **writing the loop well matters more than the number format**. The real difference between formats is range, as the table shows.

| Format | Bytes | Multiply-accumulate (TRM) | Range / smallest step | ÷ and √ | Failure mode in RL | Best use here |
|---|---|---|---|---|---|---|
| **Q16.16** (core) | 4 | SMLAL 1 cycle into 64-bit | ±32,768 / 1.5×10⁻⁵ | software; avoid 64-bit ÷ | small α·δ·z rounds to 0; traces hit the floor after ~105 steps at γλ = 0.9 | the three core learners; portable to FPU-less chips |
| **float32** (M4F FPU) | 4 | VMLA/VFMA 3 cycles | ~10⁻³⁸ to 10³⁸ / 24-bit relative | VDIV/VSQRT 14 cycles | updates below \|w\|·2⁻²⁴ are lost; FPU use in ISRs costs stack | LSTD/RLS, BO, normalisers; an ablation build of the core |
| **q15** (int16) | 2 | SMLAD: 2 MACs in 1 cycle | ±1 / 3×10⁻⁵ of full scale | software | underflow worse than Q16.16 | storage for tile weights or Q-tables, with 32-bit accumulators and stochastic rounding |
| **int8** | 1 | SXTB16 + SMLAD | 256 levels | — | needs per-tensor scales (QAS) to train | frozen priors in flash; inference |

Three practical rules follow. Keep **Q16.16 as the core format**: its justification is portability to FPU-less Cortex-M parts, not speed on the M4F. Add **one float32 build of the same learners** as a factor, since it removes the underflow failure at small cost. Use **q15 only as a storage format** where memory actually binds: the Q-table and the tile-weight table, not the network. The 838-weight network is 3.35 KB in Q16.16, and halving it saves only 1.7 KB. One trap affects the measurements themselves. QEMU counts VMLA (3 cycles) and SMLAL (1 cycle) as one instruction each, so **QEMU will make float look cheaper than it is**. Any float-versus-fixed speed statement must wait for DWT cycle counts on the Nucleo. Also keep the learner out of interrupt handlers, so the FPU's lazy context save never triggers.

### Memory tricks, ranked by what they save at this project's sizes

#### What the sources show

The F446RE has **128 KB of SRAM (112 + 16 KB), 4 KB of backup SRAM and 512 KB of flash**. Flash runs with 5 wait states at 180 MHz behind the ART accelerator, which gives "equivalent to 0 wait state program execution". Flash endurance is **10,000 erase cycles minimum**, and erasing a 16 KB sector takes about 250-400 ms typically ([ST datasheet DS10693](https://st.com/resource/en/datasheet/stm32f446re.pdf)). An ST community expert states that the F446 has **no CCM RAM**, unlike the F405/F407 ([ST community](https://community.st.com/stm32-mcus-products-25/ccm-on-stm32f405-and-stm32f446-97562)). The main published memory trick for on-device training is the MCUNetV3 sparse update. Frozen weights stay in read-only flash, and only the updated subset gets an SRAM copy. Sparse update alone cut peak training memory 7-9×, and 20-21× with operator reordering ([Lin et al. 2022](https://arxiv.org/pdf/2206.15472)). TinyOL similarly trains only the last layer in RAM and keeps the frozen network in flash ([TinyOL](https://arxiv.org/pdf/2103.08295)).

#### What it means here

| Trick | What it saves here (estimate) | What it costs | Verdict |
|---|---|---|---|
| Priors in flash (sim-trained Q-table, CPG/reflex gain library, LUTs) | SRAM equal to the prior's size; only a delta table stays in SRAM | flash reads after ART misses; unverified for data reads | **Use** for any "sim prior + online correction" variant |
| Compressed FIFO replay (int16 state, s′ implied by the next entry) | 41 → ~11 B per cart-pole transition | a few shifts per write and read | **Use** with the replay sweep |
| Sparse replacing-trace list | full per-weight trace → ~1.1 KB (linear) or ~140 B (table) | list bookkeeping | already in the reference code (opt-in) |
| q15 storage for Q-table and tile weights | 50% of the largest arrays | worse underflow; needs stochastic rounding | **Benchmark** |
| PRNG-seeded perturbations (ES) | no stored perturbation vectors | regenerate them from the seed | use if B5 is built |
| Last-layer-only network update (W1 frozen in flash) | ~1.8 KB of trace at 6-64-6 (W1 is 53% of parameters) | less plasticity | low value at this size |
| Recompute activations / ReLU bitmask | ~0.5 KB / 248 B at H = 64 | one extra forward pass / bit operations | not worth it |
| Backup SRAM checkpoint (4 KB) | no flash wear | fits only small states | good for processor-in-the-loop runs |
| Flash checkpoint | SRAM-to-flash persistence | erase stalls of 0.25-2.6 s; 10,000 cycles | episode boundaries only, never per step |

The pattern is the same as in the algorithm map. **At these sizes, the arrays that grow with resolution, the Q-table and the tile-weight table, dominate SRAM.** Network tricks that matter for MobileNet-sized models save less than 2 KB here. Recomputation, the classic compute-for-memory trick, has almost nothing to recompute in a one-hidden-layer network. The useful compute-for-memory trades in this project are **hashing** (memory against collisions, already a core sweep axis) and **regenerating data instead of storing it**: PRNG seeds, implied s′, and analytic CPG trajectories instead of stored tables.

## Benchmark replay first, number formats second, LSTD-Q third

### What the sources show

The notes found **no published Cortex-M4 cycle counts for any RL update** (tabular, linear, LSTD or Dyna). The nearest numbers are float training throughput on an M4F (PULP-TrainLib, above) and an AIfES autoencoder training step on an STM32F407 that took about **12 M cycles (71.9 ms)** ([PULP-TrainLib](https://cris.unibo.it/bitstream/11585/900686/6/PULP-TrainLib%20-%20Springer%20Version.pdf)). Walk in the Park shows why compute per sample matters. Its learner ran 20 critic updates per environment step, and the authors found layer normalisation and that high update ratio to be the components that mattered most ([Smith et al. 2022](https://ar5iv.labs.arxiv.org/html/2208.07860)).

### What it means here

The measurement pipeline stays as the project brief defines it. Only the list of things to put through it is new.

```
 MuJoCo task (laptop) ──► same C learner ──► QEMU Cortex-M4 ──► Nucleo-F446RE (optional, later)
                          host build          instructions/step     DWT->CYCCNT cycles/step
                                              peak SRAM             flash wait-state effects
                                              (agent+arena+stack)   float vs fixed REAL cost
 learning curves come from the host; QEMU and the Nucleo measure cost; claims wait for the right column
```

The shortlist is ranked by information gained per hour of work, given the gates and dates in the brief.

| Rank | Benchmark | QEMU: measure | Nucleo: measure later | Why this one | Stage |
|---|---|---|---|---|---|
| 1 | **Replay ratio n ∈ {0, 1, 5, 20, 50} for all three learners** on pendulum and cart-pole, with a compressed FIFO buffer of a few sizes | instructions per replay update; peak SRAM with buffer | cycles per replay; largest n that fits at 50 Hz | Directly measures the compute-for-samples axis, treats all three learners equally, and replaces "bigger buffer" with a sharper question | 4 |
| 2 | **Number format: Q16.16 vs float32 vs q15 storage** for the table and linear learners (network optional) | instructions/step, code size, peak SRAM, underflow counter | cycles/step (QEMU understates float) | Separates "fixed point is slow" (probably false) from "fixed point loses updates" (measurable) | 2-3 |
| 3 | **Recursive LSTD-Q in float32**, k = 64 and 128 dense features, pendulum and cart-pole | instructions per RLS update, SRAM for the k² matrix | cycles; numerical drift (re-symmetrise every N steps) | Upper reference for linear-learner sample efficiency; tests whether k² memory buys enough to be worth it | 5 (if the gate allows) |
| 4 | **Tabular Dyna-Q+ with a model table** vs plain replay at equal SRAM | as rank 1 | as rank 1 | Model vs replay under stochastic transitions; Dyna-Q+ targets terrain change. Needs a scope decision | 5 or 7 |
| 5 | **(1+1)-ES or finite-difference search over 3-6 reflex gains** (Go1 B5) | SRAM (< 1 KB); instructions per episode update | — | The cheapest serious competitor to a per-step learner on Go1; if it ties, that is a finding | 7 |
| 6 | **Last-layer-only network update** with W1 initialised in simulation | SRAM saved; instructions/step | — | Tests "sim prior + small online correction" on the network | 5 or 7 |
| 7 | **BO over the same reflex gains, on the host** | none (laptop only) | — | Sample-efficiency ceiling for B5; port to the M4 only if it wins by a lot | 7 |

The list deliberately leaves things out. PILCO, GPTD/KTD, full CMA-ES above about 20 parameters, linear Dyna on hashed features, iLSTD and per-step actor-critic all cost memory, need unbounded memory, or save compute that is not scarce.

Each benchmark should report the same four numbers, so the results line up on the headline plot: **episodes (or steps) to a fixed threshold**, **peak SRAM**, **instructions per real step including all replay or planning work**, and **replay ratio**. The fourth is what makes a Dyna-style learner comparable with a pure streaming one. Without it, a learner that uses 50× the compute per sample looks like a free win. One kill criterion protects the schedule. If replay at n = 5 does not improve episodes-to-threshold for the Q-table on pendulum, where Dyna's benefit is best documented, stop the sweep and report the null result instead of tuning it.

## Reflexes, not open loop, set the Go1 bar

### What the sources show

The CPG-RL family uses an **amplitude-controlled phase oscillator** with interpretable, bounded knobs. Amplitude μ ∈ [1, 2] scales the step, ω ∈ [0, 4.5] Hz sets the frequency, and the foot mapping has a step length d_step, a body height h, a swing clearance g_c and a stance penetration g_p. The policy runs at 100 Hz and the oscillators and PD loop at 1 kHz ([Visual CPG-RL](https://ar5iv.labs.arxiv.org/html/2212.14400)). Explicit coupling between legs helped under sensing delay: at 0.09 s delay, MLP success was 0.55 without coupling and 0.71 with it ([Visual CPG-RL](https://ar5iv.labs.arxiv.org/html/2212.14400)). CPG-RL transferred to an A1 with "only contact booleans" as feedback and no domain randomisation ([Bellegarda & Ijspeert](https://arxiv.org/abs/2211.00458)). Its ablation found that CPG states without contacts could not learn to walk ([CPG-RL v1](https://arxiv.org/pdf/2211.00458v1)).

**Tegotae** is load-based phase feedback, φ̇ᵢ = 2π(ω − σ·Fᵢ·(cos φᵢ + ξ)), with Fᵢ the foot's ground-reaction force divided by robot mass and clamped to [0, 1]. On a Mini Cheetah model, the leg phases converged in about 10 s with the feedback on, nearly 4× faster than a policy trained without it, and feedback-trained policies failed less under velocity impulses. It worked even though simulator force estimates were "highly inaccurate" ([Zhang et al. 2024](https://arxiv.org/pdf/2402.08662)). On Owaki and Ishiguro's robot, changing only a speed parameter produced walk→trot→gallop transitions using "only local load sensing" ([Owaki & Ishiguro 2017](https://doaj.org/article/90a21442c4e747f19a9ebc2593b560f2)).

**Tekken's reflex set** is the most complete non-learning precedent. Hip-angle feedback into the CPG (k_tsr = 3.0/rad) delays a leg's oscillator when the leg itself is held back. Body-roll feedback (k_tlrr = 2.25/rad) shifts the gait toward a walk. A stumble (flexor) reflex raises the knee target by 0.7 rad when the ankle deflects more than 5° during swing. Corrective stepping fires when no contact is found at stance onset, and sideways stepping scales with roll. With all gains held constant, Tekken2 walked outdoors over 14° pitch slopes, 6° roll slopes and 30 mm pebbles. Without corrective stepping it fell on a 7 cm step-down; with it, it did not ([Kimura, Fukuoka & Cohen 2007](https://www.cvl.iis.u-tokyo.ac.jp/class2017/2017w/papers/4.Robotics/AdaptiveDynamicWalking.pdf)). Open loop is not hopeless either. EPFL's Cheetah-cub ran an open-loop CPG at 1.42 m/s and passed step-downs of up to 20% of leg length ([EPFL BioRob](https://www.epfl.ch/labs/biorob/cheetah-2/)). The headroom lesson from earlier research still holds. In D2-GMBC the open-loop gait fell in all 1,000 simulated trials, which left room for a learned policy ([Rahme et al. 2020](https://arxiv.org/pdf/2010.12070)). In ETG-RL, on easy terrain, pure SAC matched the structured method ([Shi et al. 2022](https://arxiv.org/pdf/2109.06409)).

### What it means here

The recommended baseline ladder adds one feedback mechanism at a time. Every rung costs a few multiplies per leg in the 50-100 Hz loop. That is negligible on the M4 by the notes' estimate (about 100-150 FLOPs per step for a four-leg oscillator, under 200 B of state), but it is unmeasured.

| ID | Mechanism | Law (per leg i) | Sensors | Parameters | Purpose |
|---|---|---|---|---|---|
| **B0** | Open-loop CPG (current) | fixed ω = 2 Hz, no feedback | none | ω, g_c, h, step length | the floor |
| **B1** | Contact phase feedback: Tegotae, or the simpler reset/hold | Tegotae: θ̇ᵢ = 2π(ω − σ·Nᵢ·cos θᵢ); reset: on touchdown jump to stance start, on late touchdown hold end-of-swing | foot load or contact | σ (≈1-4 in Zhang et al.), offset ξ | fixes diagonal-pair desynchronisation; **required whenever the learner sees contacts** |
| **B2** | IMU posture reflex | Δhᵢ = k_r·roll·sᵢ + k_p·pitch·fᵢ (sᵢ, fᵢ = ±1 for side, front/back); optional PD form with roll and pitch rates | IMU only | k_r, k_p (Tekken's 2.25/rad needs rescaling) | **same sensors as the learner: the fairest no-learning baseline** |
| **B3** | Stumble reflex | if swing-phase contact or a large tracking error: raise clearance for that swing | contact or joint error | Δg_c (Tekken: +0.7 rad knee) | handles toe-catching on bumps |
| **B4** | B1 + B2 + B3, gains hand-tuned in simulation | — | IMU + contact | all of the above | **the bar a learner must beat** |
| B5 | Episodic search over B4 gains (optional) | (1+1)-ES or finite-difference, one update per episode | as B4 | 3-6 gains | the cheapest learning competitor |

The layering keeps the core learner's role clean. Reflexes react within a cycle. The learner adapts the reflexes and gait setpoints over several cycles.

```
 4 Hz   Q-learner (table / tile-coded / tiny net)  ── picks Δ gains and Δ setpoints at touchdown
          state: half-cycle-averaged roll, pitch (+rates), pair bit, (+contacts)
                │ persistent setpoints
                ▼
 50-100 Hz  reflex layer:  posture Δh(roll,pitch) · stumble lift · Tegotae / phase reset
                │ foot targets
                ▼
 50-100 Hz  CPG phase oscillator (ω, μ, g_c, h) ──► inverse kinematics ──► joint PD
```

Learnable parameters suitable for discrete, persistent "±Δ" actions at each half-cycle decision:

| Parameter | Physical meaning | Suggested discrete options | Needs | Evidence for the knob |
|---|---|---|---|---|
| k_att (= k_r, k_p) | posture-reflex strength | {0.5k, k, 1.5k} | IMU | Tekken roll feedback |
| g_c | swing clearance | {low, nominal, high} | — | CPG-RL mapping; Tekken flexor reflex |
| h offset per diagonal pair | stance height / body attitude | {−Δ, 0, +Δ} | IMU | D2-GMBC outputs (earlier research) |
| μ / step length | stride amplitude | {short, nominal} | — | CPG-RL μ ∈ [1, 2] |
| ω offset | cadence | {−0.25, 0, +0.25} Hz around 2 Hz | — | CPG-RL ω range; Tekken: roll pushes toward walk |
| σ (Tegotae) | strength of load feedback | {low, nominal, high} | contacts | Zhang et al. schedule 1-4 |

Keep the action set to about 9-12 joint actions, or factor it into separate small heads, so the Q-table stays small. The project currently uses 6. Three comparisons make the Go1 result interpretable. With contacts off, **compare learners against B2**, because it uses the same sensors. With contacts on, **compare against B1 and B4**, because CPG-RL suggests contacts carry most of the useful signal. And run the **learner on top of B4** (choosing reflex gains) as well as on top of B0. A learner that helps only on B0 has rediscovered a reflex. A learner that helps on top of B4 has found something reflexes miss. Because reflexes will probably raise the current 40-50% open-loop success rate (a build-note lead, not a result), the roughness sweep already in the reference code becomes essential. Plot success against roughness for B0-B4 and each learner, and the plot shows where each layer stops helping, instead of producing one number at a single roughness. If a hand-tuned linear reflex matches the learner everywhere, report it: that would be a clean result about how much adaptation this task needs.

## Every published on-robot gait learner ran off-board

### What the sources show

| Robot / study | Method | Real data needed | Where learning computed |
|---|---|---|---|
| Unitree A1, Walk in the Park ([Smith et al.](https://ar5iv.labs.arxiv.org/html/2208.07860)) | SAC-family, UTD = 20, layer norm | < 20,000 samples, ~20 min | laptop RTX 2070, 2,400 critic updates/s |
| Minitaur ([Haarnoja et al.](https://ar5iv.labs.arxiv.org/html/1812.11103)) | SAC | ~400 rollouts, ~2 h | separate workstation |
| Minitaur ([Yang et al.](https://ar5iv.labs.arxiv.org/html/1907.03613)) | learned model + MPC over trajectory generators | **4.5 min**; PPO ~1000× more, SAC ~10× more | GTX 1080 GPU |
| Unitree A1, DayDreamer ([Wu et al.](https://ar5iv.labs.arxiv.org/html/2206.14176)) | world model + actor-critic | walks at ~1 h; adapts to pushes in 10 min | GPU |
| Sony AIBO ([Kohl & Stone](https://www.cs.utexas.edu/users/AustinVilla/papers/icra04.pdf)) | finite-difference PG, 12 parameters | ~1,000 traversals, ~3 h | host PC over wireless |
| Fox biped ([Calandra et al.](https://www.ias.tu-darmstadt.de/uploads/Site/EditPublication/Calandra2015a.pdf)) | BO, 4-8 parameters | < 30 to ~80 evaluations | off-robot |
| Millirobot, GrBAL ([Nagabandi et al.](https://ar5iv.arxiv.org/html/1803.11347)) | meta-learned model, one gradient step | ~30 min per terrain to meta-train | external computer; the robot's MCU ran only a 10 Hz PID |
| Unitree A1, RMA ([Kumar et al.](https://ar5iv.labs.arxiv.org/html/2107.04034)) | sim-trained policy + latent adaptation | **no real-world learning**; adapts in < 1 s | ~1.2 B simulated steps offline |
| Hexapod, Intelligent Trial & Error ([press release](https://members.loria.fr/JBMouret/ite/press_release_eng.pdf)) | simulated behaviour map + BO | adapts to damage in < 2 min | map built offline |
| Hexapod, spiking CPG ([Lele et al.](https://arxiv.org/pdf/2003.10026)) | reward-modulated CPG coupling | tripod at cycle 66 (~4 min); ~70% of runs converge | **on board, Raspberry Pi 3B+** |
| "Scratchy" 4-servo walker ([Arm blog](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/teaching-scratchy-to-walk-with-neural-networks)) | Q-learning with a small FANN net | no numbers reported | **on an STM32F407 (Cortex-M4)** |

RMA's simulation ablation is informative. Its learned implicit adaptation reached 73.5% success, a domain-randomised robust policy 62.4%, and explicit system identification 56.5% ([Kumar et al.](https://ar5iv.labs.arxiv.org/html/2107.04034)). Streaming deep RL can adapt a pretrained simulated quadruped to a broken leg or a slippery floor. ObGD-style optimisers clearly beat Adam there, but the adaptation phase lasted **1.5 M steps** with 256-unit networks, and no hardware figures were reported ([Vitchutripop et al.](https://arxiv.org/pdf/2609.28807)).

### What it means here

The table shows three ways to buy down real-robot data: **more compute per sample** (UTD, model-based planning), **offline compute stored as a prior** (RMA, IT&E maps), and **a small structured parameter space** (AIBO, Fox, trajectory-generator residuals). A Cortex-M4 can use the first only in miniature: replay at n ≤ 50, not 2,400 updates per second. The second and third cost flash and design effort rather than SRAM, and they suit the M4 well. That is why the replay sweep, flash-resident priors and reflex-gain actions keep recurring in this report.

The data arithmetic sets expectations. At 4 decisions per second, the 1.5 M adaptation steps of the streaming study would take about **104 hours** of walking, so streaming deep RL from scratch is not a realistic on-board gait learner here. It is realistic only as a small correction to a pretrained policy. The closest analogues to this project, the spiking hexapod at 66-207 cycles and IT&E at under 2 minutes, adapt a few parameters on top of a strong structure. Both are the same order as the project's 80-160 decisions per episode. **A realistic MCU claim is "adapts a handful of reflex and gait gains within N episodes, using X KB of SRAM measured on chip".** "Learns to walk" is not a realistic claim.

| Safe to claim (once measured) | Do not claim |
|---|---|
| Peak SRAM and instructions per step for each learner, from QEMU | "First on-device RL" or "first on-MCU gait learning". Use the brief's wording: "a controlled comparison was not found in a limited search" |
| Cycles per step, only after DWT CYCCNT on the Nucleo | Any cycle, latency, energy or power figure derived from QEMU or the datasheet |
| Learning and adaptation curves in simulation, with seeds, step budget and replay ratio stated | Real-robot gait adaptation (there is no physical robot) |
| "Learner L beats/ties reflex baseline B4 at roughness r" in simulation | Sample efficiency comparable to Walk in the Park, RMA or IT&E: different robots, compute and granularity |
| "Float32 vs Q16.16 cost X vs Y cycles on the F446RE", after measurement | "Fixed point is faster on the M4F" (the TRM says the gap is small and load-dominated) |
| The underflow count per learner and the effect of stochastic rounding | CMSIS-NN speedups for the M4 (measured on an M7, inference only) |
| Episode counts of B5 or BO against the per-step learners | Prioritized-sweeping gains, Owaki's exact equation, or IT&E map sizes (not verified in these notes) |

One citation hazard is worth fixing now. The earlier notes and the project brief describe the RLtools microcontroller run as TD3 on Pendulum, but this round's notes, read from the JMLR version, report SAC. Until the paper is checked directly, cite only "deep RL training on a Teensy 4.1 (Cortex-M7)" without naming the algorithm ([Eschmann et al., JMLR](https://jmlr.org/papers/volume25/24-0248/24-0248.pdf)).

## Conclusion

The research reframes the project's resource question. The original axes were learning performance against peak SRAM. On a 4 Hz decision clock, compute per step has enough slack to become a deliberate third axis, and **replay ratio is the dial that moves it** without touching the three core learners. That gives the headline plot a natural second panel: episodes-to-threshold against peak SRAM, with one curve per replay ratio. It also gives Stage 4 a question the literature cannot answer for this setting, namely how much of Walk in the Park's UTD effect survives at n ≤ 50, in Q16.16, in under 128 KB. The float32 build and the LSTD-Q arm serve the same purpose from the other side. They show whether a loss comes from the representation, from the number format, or from using samples inefficiently, which the core comparison alone cannot tell apart.

For the Go1, the reflex literature changes what a positive result looks like. Tekken and Tegotae show that a few fixed gains already make a CPG walk on rough ground. The defensible contribution is therefore not "a learner beats open loop" but **"a learner small enough for 128 KB adapts reflex gains faster or better than a hand-tuned reflex or an episodic hill-climber, and here is where on the roughness axis it stops mattering"**. That result holds whichever way it comes out, and every component of it can be measured with the tools the project already has.
