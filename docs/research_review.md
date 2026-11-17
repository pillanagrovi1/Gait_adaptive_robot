<!-- Literature review, 17 Nov 2026. Five research threads (streaming Q-learning, MCU on-device training, linear features and traces, CPG gait RL, evaluation methodology) synthesised into one report. Implementation status of each recommendation: see the table at the end of docs/decisions.md. -->

# Stream the network, cache the tiles

The biggest improvement available without touching the core is to make the tiny network learn from a stream instead of a replay buffer. Stream Q(λ) (Elsayed, Vasan and Mahmood, 2024) stabilises single-sample Q-learning without a replay buffer or a target network. It does this with five cheap parts: a scalar step-size bound called ObGD, LayerNorm without learnable parameters, running observation normalisation, reward scaling, and sparse initialisation. Its only per-weight memory cost is one eligibility trace value. For this project's largest network (6-64-6, 838 weights), that would take the network learner from roughly **67-157 KB to roughly 7.5 KB** of learner state. The top of today's 60-150 KB buffer range does not even fit in the STM32F446RE's 128 KB. There is a catch: no published result tests the recipe below 128 hidden units or on cart-pole or pendulum, so for 8-64 units it is a hypothesis this project would be the one to test. The second-largest gain is in the linear learner. Most of its ~8,400 instructions per step come from hashing every state twice and from a 64-bit modulo, which the Cortex-M4 can only do through a library call. Caching indices and using a power-of-two hash should remove most of that cost. Third, the Q16.16 underflow problem is the "swamping" failure that stochastic rounding was shown to fix in 16-bit fixed-point training. It applies to all three learners and leaves the number format unchanged. Fourth, the Go1 learners probably fail to beat the open-loop trot for two reasons. The terrain leaves them little to fix, and the 4 Hz state cannot tell which leg pair is about to swing. Published residual-over-gait-generator successes come from settings where the open-loop gait fails badly: in D2-GMBC it fell in all 1,000 simulated trials. Finally, the evaluation is the weakest part of the current design. Five seeds, hand-tuned hyperparameters and an equal *episode* budget cannot support claims that one learner beats another, and the learners are cheap enough to run 10-30 seeds. The recommended order: fix the evaluation protocol and add underflow diagnostics now (Stages 1-2); do the stream recipe and the tile-coding fixes in Stage 3; turn Stage 4 into a compressed-buffer comparison; leave the Go1 redesign for the optional Stage 7.

## How to read this report: verified facts versus interpretation

Every area section below is split into two parts. **"What the sources show"** holds claims that the research notes checked against a fetched source, each cited inline. A few are marked *(abstract only)* or *(not re-checked this round)* where the notes say so. **"What it means here"** is interpretation: arithmetic, design judgement and predictions for this project. Treat that part as hypotheses to measure, never as results. In line with the project rule, no cycle, timing or power figure for the STM32F446RE appears here as a fact. QEMU instruction counts are not cycles, and any cycle figure needs a measurement on the Nucleo with the DWT cycle counter.

The diagram below shows the shared learning loop and where each recommended change attaches. All three learners keep the same Q-learning target. Only the box marked "features → Q" differs between them.

```
 state s_t ─► [Welford obs-normaliser]* ─► features φ(s) ─► Q(s,·) ─► ε-greedy ─► action a_t
                                           (cache φ(s') for t+1)                     │
 reward r ─► [reward scaler]* ─► TD error δ = r + γ·max_a' Q(s',a') − Q(s,a) ◄───────┘
                                        │
                                        ▼
               trace z ← γλ·z + ∇Q(s,a)      (λ shared by all three learners; Watkins cut
                                        │      resets z after an exploratory action)
                                        ▼
               scalar step α_eff = ObGD bound(α, κ, δ, ‖z‖₁)
                                        │
                                        ▼
               w ← w + α_eff·δ·z   → stochastic rounding back to Q16.16
 (* = mainly for the network; the table and linear learners can skip or share them)
```

| Term | Plain meaning |
|---|---|
| TD error δ | How wrong the current estimate was: r + γ·max Q(s′,·) − Q(s,a) |
| Replay buffer / target network | Stored past transitions that get re-sampled / a delayed weight copy used to compute the target. Both are classic DQN stabilisers |
| Streaming (online) learning | Learn from each transition once, then throw it away. Batch size 1 |
| Eligibility trace z, λ | A decaying record of which weights mattered recently. λ ∈ [0,1] sets how fast it fades, so one δ can credit several past steps |
| Watkins cut | Reset z to zero after an exploratory (non-greedy) action, which keeps Q(λ) a correct off-policy Q-learning method |
| ObGD | "Overshooting-bounded gradient descent": shrinks the step whenever δ or ‖z‖₁ is large, so one update cannot overshoot |
| LayerNorm (no affine) | Rescale a layer's pre-activations to mean 0 and variance 1 at every step, with no learned scale or bias |
| Welford | Running mean and variance computed one sample at a time |
| LSB of Q16.16 | Smallest representable step, 2⁻¹⁶ ≈ 1.5×10⁻⁵ |
| Stochastic rounding | Round up with probability equal to the discarded fraction, so tiny updates survive on average |
| Tile coding / hashing | T offset grids, each lighting one cell; Q = sum of the lit weights. Hashing folds a huge grid into M weights, so cells can collide |
| CPG | Central pattern generator: the oscillator that produces the rhythmic trot |
| IQM / bootstrap CI | Mean of the middle 50% of runs / a confidence interval obtained by resampling runs |

## Stream Q(λ) trades a 150 KB buffer for a 3 KB trace

### What the sources show

**Stream Q(λ)** is Watkins-style Q(λ) run with batch size 1, no replay buffer and no target network. Five components make it stable: ObGD, 90% sparse initialisation, parameter-free LayerNorm before every activation, Welford observation normalisation, and dividing the reward by the running standard deviation of the discounted return, without subtracting the mean ([Elsayed et al. 2024](https://arxiv.org/pdf/2410.14606)). The trace is reset at the start of each episode and after every non-greedy action. Each step it updates as z ← γλz + ∇q̂(S,A) and then calls ObGD. The defaults are **α = 1, κ = 2, γ = 0.99, and λ = 0.8 in the experiments** ([Elsayed et al. 2024](https://arxiv.org/pdf/2410.14606)). ObGD computes δ̄ = max(|δ|, 1), M = α·κ·δ̄·‖z‖₁ and α_eff = min(α/M, α), then sets w ← w + α_eff·δ·z. In words, the step size falls automatically when the error or the trace is large. In the ablations, **removing ObGD, or replacing it with tuned Adam, hurt performance more than removing any other component** ([Elsayed et al. 2024](https://arxiv.org/pdf/2410.14606)). Classic Q(λ), SARSA(λ) and DQN with batch size 1 "perform poorly" or fail. Stream Q(0.8) was comparable to full DQN on MinAtar, and on Atari it beat DQN in 9 games and trailed in 6 (200M frames, 10 runs) ([Elsayed et al. 2024](https://arxiv.org/pdf/2410.14606)). The scale limit matters for this project: **every network in the paper is at least 128 units wide**, and the notes found no classic-control results. The official code is at github.com/mohmdelsayed/streaming-drl, but the notes could not open it. A third-party port implements only the actor-critic variant ([streaming-deep-rl on PyPI](https://pypi.org/project/streaming-deep-rl/0.1.6/)).

PQN (Gallici et al., ICLR 2025) adds independent theoretical support for the LayerNorm component. It shows that LayerNorm, plus a small ℓ2 term needed only in pathological cases, makes off-policy nonlinear TD satisfy a stability criterion **without a target network**. The bound holds "for all k > k′", i.e. for networks wide enough ([Gallici et al. 2025](https://arxiv.org/html/2407.04811v2); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c23f3852601f6dd7f0b39223d031806f-Abstract-Conference.html)). PQN removes the replay buffer differently, by "obtaining mini-batches directly from interactions with parallel environments" ([Gallici et al. 2025](https://arxiv.org/html/2407.04811v2)). AVG (NeurIPS 2024) is continuous-action only. It contributes two cheap stabilisers: L2-normalising the penultimate features, which beat LayerNorm and RMSNorm in its ablation, and dividing δ by a running TD-error scale ([Vasan et al. 2024](https://arxiv.org/pdf/2411.15370)). It also learned on a Jetson Nano-driven robot where SAC with buffer size 1 learned nothing ([Vasan et al. 2024](https://arxiv.org/pdf/2411.15370)). Work from 2026 refines the step-size rule. "Intentional TD" chooses the step so that each update achieves a fixed fractional reduction of the TD error, an extension of normalised LMS, and reports state-of-the-art streaming results *(abstract only)* ([Sharifnassab et al. 2026](https://arxiv.org/abs/2604.19033)).

Evidence on adaptation, the project's second axis, is newer. A 2026 preprint warm-started streaming actor-critic agents (256-unit MLPs) on a simulated Anymal C and then changed the dynamics. On "Broken Leg", Adaptive ObGD reached **0.968 max success versus 0.412 for batch PPO and 0.188 for Adam**, over 5 seeds. The authors conclude Adam is "not well-suited for stream learning". **Stream Q was not evaluated** ([Vitchutripop et al. 2026](https://arxiv.org/pdf/2609.28807)). Continual backprop (CBP) keeps one utility value and one age counter per hidden unit, and occasionally reinitialises the least useful mature unit, zeroing its outgoing weights so the network's output does not change. It used a replacement rate of ρ = 10⁻⁴ and a maturity threshold of 10⁴ steps for RL ([Dohare et al., Nature 2024](https://doi.org/10.1038/s41586-024-07711-7); [PDF copy](https://rivista.ai/wp-content/uploads/2024/08/Loss_of_plasticity_in_deep_continual_learning.pdf)). In the preprint above, adding CBP raised mean success on the Goal Shift and Slippery Floor tasks ([Vitchutripop et al. 2026](https://arxiv.org/pdf/2609.28807)).

### What it means here

The recipe fits the core unusually well. It keeps the max-based Q-learning target, does exactly one update per environment step, and needs only statically sized arrays. The table below shows the memory cost for the largest configuration, based on the notes' accounting with P = 838 parameters, H = 64 hidden units and D = 6 inputs. **Every square root and division occurs O(layers + inputs) times per step, never once per weight.** The exception is Adaptive ObGD and Adam, whose per-weight √v this project should avoid.

| Component | Extra memory (6-64-6 net, Q16.16) | Extra work per step | Division/√ per step |
|---|---|---|---|
| Eligibility trace | +3.35 KB (one value per weight) | P multiply-adds | 0 |
| ObGD | one scalar ‖z‖₁ | P abs-adds | 1 division |
| LayerNorm, no affine (hidden layer) | ~0.25 KB workspace | ~3H | 1 √ and 1 division |
| Welford obs norm + reward scaling | ~0.03 KB | ~5D + 5 | D+1 √ (can refresh every N steps) |
| CBP | 0.5 KB (utility + age, 64 units) | ~2H | 0 |
| Removed: target network | −3.35 KB | — | — |
| Removed: replay buffer | −60 to −150 KB | — | — |

```
Network learner state, largest config, against the F446RE's 128 KB SRAM
                                      0        32        64        96       128 KB
 today: weights+target+60 KB buffer  |███████████████▋                    |   ~67 KB
 today: weights+target+150 KB buffer |██████████████████████████████████████████ ~157 KB  (does not fit)
 Stream Q(λ) + LN + CBP              |██▏                                 |   ~7.5 KB
```

Three risks are specific to this scale, and no source tests any of them. First, **90% sparse init is meaningless with 2-6 inputs**: it leaves about one non-zero input weight per hidden unit, so use a much lower sparsity or skip it. Second, LayerNorm computed over 8 hidden units is a noisy normaliser, and PQN's guarantee only covers wide networks. AVG's penultimate L2 normalisation, which needs no mean, is cheaper in fixed point and worth testing as an alternative at H = 8-16. Third, ObGD with α = 1 makes the effective step about 1/(2·δ̄·‖z‖₁). With an 838-weight trace, ‖z‖₁ can reach the hundreds, which puts individual weight increments near or below the Q16.16 LSB. Stream Q(λ) therefore depends on the precision fix in the next section. The Watkins cut also means that while ε is high early in training, traces are reset every few steps and λ adds little until ε decays.

One fairness issue matters more than any of these risks. Stream Q(λ) introduces λ. If only the network gets λ = 0.8 while the table and linear learners stay at one-step Q-learning, any win for the network could come from multi-step credit assignment rather than from its representation. The project brief allows "the stabilisers it needs" as long as their memory is counted. A trace is more than a stabiliser, though, because it changes how credit is assigned. The clean design treats λ as a shared factor applied to all three learners (see the linear-learner section). A housekeeping correction also belongs here: the project's related-work list says Stream-X learned Pong on an ESP32-S3, but **the notes found no microcontroller experiment in the paper text they read**. Remove that line until it is confirmed from the official repository or another source.

| Recommendation card | |
|---|---|
| Technique | Stream Q(λ): ObGD + LayerNorm without affine (or L2 norm) + Welford obs norm + return-std reward scaling + low or no sparse init; no buffer, no target network |
| Evidence | Strong at 128-1024 units on MinAtar, Atari and MuJoCo; none at 8-64 units or on classic control |
| Cost | ~2 values per weight; a handful of √ and divisions per step |
| Fit with core | Same Q-learning target, one update per step, static allocation. Keep the current SGD + target + buffer learner as the baseline arm |
| Do this | Implement it as a second network-learner variant in Stage 3, then ablate one component at a time, ObGD first. Add CBP only for the shift and adaptation runs |

## Sub-LSB updates, not multiplies, are the Q16.16 bottleneck

### What the sources show

Low-precision training has a known failure mode with a known fix. Gupta et al. showed that deep networks "can be trained using only 16-bit wide fixed-point number representation when using **stochastic rounding**" with little loss, and that the rounding scheme is what decides whether training works *(abstract only)* ([Gupta et al. 2015](https://proceedings.mlr.press/v37/gupta15.html)). MCUNetV3 trained on a 256 KB-SRAM microcontroller budget and quantified the cost of ignoring scale effects. Plain int8 SGD scored **64.9% against 73.3% for fp32, and Quantisation-Aware Scaling (QAS) recovered it to 73.5%**. Keeping only the last layer in higher precision added 2.3 points, and batch size 1 matched batch size 128 (72.3% vs 72.4%) ([Lin et al. 2022](https://arxiv.org/html/2206.15472v4)). Tin-Tin (2025) is integer-only training for FPU-less M0/M4 cores. It uses int32 accumulators, tracks a per-tensor scale, and has an optional rule that rounds any non-zero sub-LSB update up to ±1 LSB ([Tin-Tin 2025](https://arxiv.org/html/2504.09405v1)).

The Cortex-M4 instruction costs explain why multiplies are not the problem. The TRM lists **SMULL/SMLAL (32×32→64) and the dual 16-bit MACs SMLAD/SMLALD at 1 cycle**, while hardware SDIV/UDIV is 32-bit only at 2-12 cycles ([Cortex-M4 TRM r0p1, mirror](https://www.cse.scu.edu/~dlewis/book3/docs/Cortex-M4%20Proc%20Tech%20Ref%20Manual.pdf)). A 64-bit division compiles to `__aeabi_ldivmod`, which branches to `__divmoddi4`: a function call, not an instruction ([compiler-rt source](https://codebrowser.dev/llvm/compiler-rt/lib/builtins/arm/aeabi_ldivmod.S.html)). CMSIS-DSP's q15 dot product sums SMLALD products into a 64-bit accumulator with 33 guard bits, so it cannot overflow ([CMSIS-DSP arm_dot_prod_q15](https://os.mbed.com/users/mbed_official/code/mbed-dsp/file/fdd22bb7aa52/cmsis_dsp/BasicMathFunctions/arm_dot_prod_q15.c)).

On RL specifically, RLtools reported **the first deep-RL training on a microcontroller**: TD3 on Pendulum-v1 on a Teensy 4.1 (Cortex-M7, 1 MB RAM). The authors chose that chip "because of the memory required to store the replay buffer", and noted that Pendulum trains with buffers "much smaller than 10,000" ([Eschmann et al., arXiv v2](https://arxiv.org/pdf/2306.03530v2); [v4](https://arxiv.org/html/2306.03530v4)). Its inference benchmark on a Cortex-M4 at 168 MHz ran a 13-64-64-4 MLP forward pass in 293 µs once optimised, against 743 µs for generic code ([Eschmann et al.](https://arxiv.org/html/2306.03530v4)). The only replay-compression technique found in a primary source is to store each state once and refer to it by index, which R³ estimates saves about 75% for stacked frames ([R³, arXiv 2308.15039](https://arxiv.org/pdf/2308.15039)). A May 2026 survey of about 70 on-device-learning works reports a "persistent gap between methodological benchmarks and real-world deployment" ([Pavan et al. 2026](https://arxiv.org/abs/2605.31226)). That supports the project's careful novelty wording.

### What it means here

Round-half-up silently drops every update smaller than half an LSB. As a result, a learner's effective step size depends on the number format as well as on α. The fix that leaves Q16.16 untouched is to change only the rounding function inside the shared update. The sketch below is pseudocode, not finished code.

```c
/* w += a_d * z   (a_d = alpha_eff*delta, z = trace or gradient; all Q16.16)  */
int64_t p = (int64_t)a_d * z;              /* SMULL: product in Q32.32           */
uint32_t r = xorshift32(&rng) & 0xFFFFu;   /* 16 random bits = U[0,1) LSB        */
w += (int32_t)((p + r) >> 16);             /* floor(p + U): unbiased on average  */
/* ARM GCC shifts signed values arithmetically; check this once, since C leaves it implementation-defined. */
```

There is a subtle trap here. α_eff·δ can itself underflow before it ever multiplies z: α_eff ≈ 10⁻³ times δ ≈ 10⁻² is below the LSB. Because α_eff is a single scalar, store it in a finer format such as Q2.30. That costs nothing per weight. Measure before choosing among the options below. Add a counter for "update was non-zero before rounding but zero after", per learner, so the size of the problem becomes a number in the results log.

| Rounding option | Memory | Work per updated weight | Bias | Use it for |
|---|---|---|---|---|
| Round-half-up (current) | 0 | 0 | Drops updates < ½ LSB | Baseline |
| Stochastic rounding | RNG state only | 1 RNG draw + add + shift | Unbiased | Default for all three |
| Minimum 1-LSB step (Tin-Tin) | 0 | 1 compare | Inflates small updates | Ablation |
| Wider master weights (int64, or Q16.16 + int16 residual) | +2-4 B per weight | ~2 ops | None | Network only, if SR proves too noisy |

The table and linear learners update only 1 or 8 weights per step, so stochastic rounding is essentially free for them. The network updates all 838. One 32-bit RNG draw can supply 16 random bits for two weights, which halves the overhead, but the real cost should be measured as an instruction count in QEMU. For the 64-bit division, every `int64 / int64` or `%` in a hot path (visit-count normalisers, `(a<<16)/b`, hashing) should become a shift, a precomputed reciprocal multiplied with SMULL, or a 32-bit SDIV. The actual cycle cost of `__aeabi_ldivmod` on the F446RE is still unmeasured and should be measured with DWT->CYCCNT before it is quoted.

Stage 4 ("Buffer") should be redefined. The stream recipe may make a 60-150 KB buffer unnecessary, so the useful comparison is a **compressed small buffer** combined with the stream recipe. The table compresses one transition for cart-pole (d = 4). It assumes a layout that stores s and s′ in Q16.16, so check it against the actual struct.

| Field | Current (assumed) | Compressed | How |
|---|---|---|---|
| s | 16 B | 8 B | int16 per dimension with a fixed known scale |
| s′ | 16 B | 0 B | implied by the next ring-buffer entry; skip episode boundaries |
| action + done | 5 B | 1 B | uint8, with done folded into the top bit |
| reward | 4 B | 2 B | int16 with a fixed scale |
| **Total / transitions in 60 KB** | **41 B / ~1,500** | **11 B / ~5,600** | ~3.7× |

Use a FIFO ring rather than reservoir sampling, because old terrain data goes stale after a shift. That is a design judgement, not a sourced result. Moving parameter storage to int16 with SMLAD (half the memory, two MACs per instruction) and int8 training with QAS **change the Q16.16 core**, so they belong under future work.

| Recommendation card | |
|---|---|
| Technique | Stochastic rounding in the shared update; scalar α_eff in a finer format; no 64-bit division in hot paths; compressed FIFO replay for the hybrid arm |
| Evidence | Stochastic rounding: 16-bit fixed-point deep training (Gupta). M4 costs: TRM. Buffer de-duplication: R³. No source tests stochastic rounding on tabular or linear TD learning on an MCU |
| Cost | ~0 memory; a few instructions per updated weight |
| Fit with core | Keeps Q16.16 and static allocation. Only the rounding and the buffer layout change |
| Do this | Add the underflow counter in Stage 2; switch to stochastic rounding for all three learners if the counter is non-trivial; build the compressed buffer in Stage 4 |

## Most of the 8,400 tile-coding instructions are self-inflicted

### What the sources show

Sutton's original tile-coding C code hashes coordinates through a 2048-entry random table, reduces the result with `sum % memory_size`, and states "We assume that hashing colisions are to be ignored". It also recommends power-of-two sizes ([tiles.C mirror](https://waxworksmath.com/Authors/N_Z/Sutton/RLAI_1st_Edition/Code/Chapter_8/tiles.C)). The later tiles3 design quantises each variable once (q = ⌊x·T⌋), builds each tiling's coordinates using only integer adds and divides with asymmetric offsets, and hands out indices from a table, so there are **no collisions until the table fills** ([tilecoding crate, faithful port](https://docs.rs/tilecoding/0.3.0/src/tilecoding/lib.rs.html)). The evidence on collisions is mostly qualitative. A CMAC review calls collision reduction "still considered an art", reports an argument that hashing does not improve approximation capability, and describes a hardware CMAC that hashes by selecting index bits, which amounts to a mask ([CMAC review](https://ar5iv.labs.arxiv.org/html/1702.02277)). One RL study attributed poor results from large tile codings to hashing without running an ablation ([Le et al. 2017](https://ar5iv.labs.arxiv.org/html/1707.08316)). Ghiassian et al. used 8 tilings for Mountain Car and hashed Acrobot to 512 features. They also swept **power-of-two step sizes α = 2⁻ᶜ**, which map directly to shifts ([Ghiassian et al. 2020](https://arxiv.org/pdf/2003.07417)). Sutton and Barto recommend at least 4k tilings for k dimensions and the step-size rule α/T for T active binary features *(not re-checked this round)* ([Sutton & Barto 2018](http://incompleteideas.net/book/RLbook2020.pdf)).

On traces, true online TD(λ) adds "two small changes" at "negligible" cost and learns "often better, but never worse" than regular TD(λ). Accumulating traces "can easily cause divergence" at large α and λ ([van Seijen et al. 2016](https://ar5iv.labs.arxiv.org/html/1512.04087)). True online Sarsa(λ) is on-policy, however. For off-policy Q-learning, the Stream-X work uses Watkins cuts, warns that traces "can be arbitrarily large, potentially causing divergence", and makes them safe with ObGD ([Elsayed et al. 2024](https://arxiv.org/pdf/2410.14606)). With adaptive step sizes, replacing traces were the most robust choice at every λ ([Kearney et al., TIDBD](https://ar5iv.arxiv.org/html/1804.03334)). Dabney and Barto derive a meta-parameter-free upper bound on α for online linear TD that prevents divergence *(abstract only)* ([Dabney & Barto 2012](https://ojs.aaai.org/index.php/AAAI/article/view/8313)). TIDBD, by contrast, keeps two extra vectors per weight ([Kearney et al.](https://ar5iv.arxiv.org/html/1804.03334)). Among linear alternatives, the Fourier basis beat RBF and polynomial bases on Acrobot, but it was never compared with tile coding, and its feature count grows as (n+1)^d ([Konidaris et al. 2011](https://people.cs.umass.edu/%7Epthomas/papers/Konidaris2011a.pdf)). Learned sparse features such as FTA beat a tile-coded-input network, but they require a network ([Pan et al. 2021](https://arxiv.org/pdf/1911.08068)).

### What it means here

The cost problem comes down to three avoidable choices. The table lists the fixes. The expected result, a few hundred instructions per step, is an estimate that needs measuring in QEMU.

| Step in act + learn | Now (per brief) | Proposed | Why it saves |
|---|---|---|---|
| Index computations | twice per step (s and s′) | once: s′'s T indices become the next step's s (8 × uint16 = 16 B cache) | Halves hashing |
| Reduction to M | 64-bit `%` | M = 2ᵐ; `h = mix(coords) * 0x9E3779B1u; idx = h >> (32-m)` | Removes a library call per tiling |
| Per-tiling offsets | floating-point-style arithmetic and `%` | pre-scale once into integer units of 1/T tile; per tiling: add, then `>> log2(T)` | Add and shift only |
| Weight layout | (check) | `w[idx][a]`, contiguous over actions | One hash serves every action's Q |
| Pendulum | hashed | dense, collision-free grid: 8 × 11² = 968 weights per action | No hash at all |
| Step size | α | α₀/T = a right-shift by 3 (this is NLMS for T-hot features) | No division |

Cart-pole without hashing needs about 19,200 weights per action (8 × 7⁴), so hashing is only necessary from 4 dimensions up. That makes a small result available that the literature lacks: **a dense-versus-hashed comparison at equal M, and a sweep of performance against load factor** (unique cells visited ÷ M). The notes found no published curve of this kind. If T stays at 8 for cart-pole, it falls below the textbook 4k = 16 recommendation. Doubling T doubles the per-step cost, so T is a natural sweep axis on the compute side.

Traces raise the fairness question from the network section. The cleanest design runs **Watkins Q(λ) for all three learners with λ ∈ {0, 0.8} as an explicit factor**, so the headline comparison changes only the representation. On-policy Sarsa(λ) would change the shared target and is therefore out. Full traces double the table's and the linear model's memory, which hurts them on the memory axis. A sparse trace list does not. Entries decay by γλ = 0.99 × 0.8 = 0.792 per step, so dropping entries below 0.01 keeps about ln(0.01)/ln(0.792) ≈ 20 steps of history. For the linear model that is 8 × 20 = 160 entries of (index, action, value), about **1.1 KB whatever M is**. For the table, with one active cell per step, it is about 140 B. Under ε = 0.1 the Watkins cut fires about every 10 steps, which keeps the list short in practice. For binary T-hot features, ObGD reduces to a scalar bound on α, because ‖z‖₁ ≤ T/(1−γλ). It adds no per-weight memory and gives the linear learner the same safety net as the network.

| Step-size method | Extra memory per weight | Verdict here |
|---|---|---|
| α₀/T (NLMS) | 0 | Default; already a shift |
| ObGD scalar bound / Dabney-Barto bound | 0 | Use one of them with traces |
| IDBD / TIDBD | +2 vectors (12-16 B per weight in total) | Skip: shrinks M 3-4× for the same SRAM |
| Adam / Adaptive ObGD | +1-2 vectors and a √ per weight | Skip |

| Recommendation card | |
|---|---|
| Technique | Cached indices, power-of-two multiplicative hash, integer offsets, dense grid where it fits; Watkins Q(λ) with sparse replacing traces and a scalar ObGD bound |
| Evidence | Code-level design from tiles.C and tiles3; trace stability from true online TD and Stream-X; collision effects are unquantified in the literature |
| Cost | Fewer instructions per step (estimated); ~1.1 KB for sparse traces |
| Fit with core | Same hashed tile-coding linear Q-learner; only the implementation and λ change |
| Do this | Rewrite the feature code in Stage 3 and record instructions per step before and after; add the collision sweep; run λ as a shared factor in Stage 5. Skip Fourier, FTA and SCoPE because they replace the core representation |

## The open-loop trot leaves the Go1 learner little to fix

### What the sources show

The closest precedent for the gait task is residual learning over a trajectory generator (TG). PMTG used a **77-parameter linear policy** whose inputs were exactly roll, pitch, roll rate and pitch rate plus the desired velocity and the TG phase as sin/cos. Its outputs were 3 TG parameters and 8 per-leg residuals ([Iscen et al. 2018](https://arxiv.org/pdf/1910.02812v1.pdf)). D2-GMBC used a 168-parameter linear policy with IMU and per-foot phase as inputs, and clearance height, ground penetration and per-foot residuals as outputs. It randomised mass, friction and terrain up to about 40% of robot height. **In simulation, the open-loop gait fell in all 1,000 trials**, while the learned policy often survived. On the real robot over rocks it succeeded 0.60 of the time versus 0.14 for open loop ([Rahme et al. 2020](https://arxiv.org/pdf/2010.12070)). The paper also contains an internal inconsistency: 146/305 is reported as 45%. ETG-RL found the converse. On easy uneven "Terrain", where the robot rarely falls, **pure SAC (2024.3) matched ETG-RL (2003.9)**, because there was little failure to correct ([Shi et al. 2022](https://arxiv.org/pdf/2109.06409)).

Every successful method observes the gait phase: PMTG uses sin/cos, D2-GMBC per-foot phase, Lee et al. sin/cos plus frequency, and CPG-RL the full CPG state ([Lee et al. 2020](https://arxiv.org/pdf/2010.11251); [Bellegarda & Ijspeert 2022](https://arxiv.org/pdf/2211.00458v1)). CPG-RL's ablation shows that feedback is needed too: foot contacts plus CPG states transferred to hardware, but **CPG states without contacts could not learn to locomote** ([Bellegarda & Ijspeert 2022](https://arxiv.org/pdf/2211.00458v1)). The CPG-RL family modulates persistent, low-pass-filtered oscillator setpoints at 100 Hz ([Bellegarda & Ijspeert 2022](https://arxiv.org/pdf/2211.00458v1)). Every verified policy rate falls between 50 and 400 Hz, and **no paper tests rates near 4 Hz** ([Rudin et al.](https://ar5iv.labs.arxiv.org/html/2109.11978); [Lee et al. 2020](https://arxiv.org/pdf/2010.11251)). Rudin et al. penalise action rate, terminate on base contact, and bootstrap on time-outs rather than treating them as terminal, which improved total reward by about 10-20%. They also promote or demote each robot's terrain level according to its progress ([Rudin et al.](https://ar5iv.labs.arxiv.org/html/2109.11978)). Lee et al. keep terrains whose traversability lies in **[0.5, 0.9]**: neither trivial nor impossible ([Lee et al. 2020](https://arxiv.org/pdf/2010.11251)).

Two related-work entries need correcting. Salih, Mazinan and Modaresi (Sci. Rep. 2026) describe a **navigation system** (Tiny-YOLOv3 plus a "Q-iteration" whose Bellman update, in the paper's own words, "is not detailed"), not on-board gait learning ([Nature](https://www.nature.com/articles/s41598-025-30610-4)). The microrobot study that runs a policy on a Cortex-M0 trained it offline on a GPU ([arXiv 2512.24740](https://arxiv.org/html/2512.24740v1)). The notes found no verified precedent for a quadruped learning a gait on-device with tabular or linear Q-learning over a CPG.

### What it means here

The most likely reason the learners do not beat open loop is **missing headroom**, not weak learners. If the open-loop trot rarely falls on the current height field, the best achievable policy is "do nothing". ETG-RL's easy-terrain result shows exactly that. Calibrate before redesigning anything: sweep terrain roughness, and optionally friction and mass, until the open-loop trot fails roughly 20-60% of the time. That is Lee's traversability band applied to the baseline. Then report the learner's gain across the sweep instead of at one level. A Rudin-style promote/demote rule on a single roughness value is simple enough to implement for an on-board learner.

The second problem is **phase aliasing**. At a 2 Hz trot, a decision every 0.25 s lands half a cycle after the previous one, so consecutive decisions alternate between "pair A about to swing" and "pair B about to swing". The 4-value IMU state does not say which. And if any action changes the CPG frequency, or the decision clock is offset from touchdown, the phase also drifts.

```
time (s)     0.00       0.25       0.50       0.75       1.00
pair A       |--swing---|--stance--|--swing---|--stance--|
pair B       |--stance--|--swing---|--stance--|--swing---|
decide at    ^ A next   ^ B next   ^ A next   ^ B next      <- same IMU reading, different meaning
fix: lock decisions to touchdown + add 1 bit "which pair swings next" (doubles the table)
```

Locking decisions to touchdown, adding that one bit, and averaging the IMU over the last half cycle all keep the learner IMU-driven. Averaging also removes the trot's own built-in roll and pitch oscillation from the state. Foot-contact flags would help, according to CPG-RL, but they extend the observation beyond the IMU. Treat them as an optional variant that is clearly labelled. Discrete actions that apply a one-shot posture change are weaker than the persistent setpoints the published methods modulate. Two alternatives stay discrete: incremental actions such as ±Δclearance or ±Δfront/rear height, whose effect persists and accumulates, and factored actions (3 step heights × 3 pitch corrections × 3 roll corrections = 27 actions, or separate Q-heads). The reward has the right shape, close to D2-GMBC's, but its scale needs checking. At about 0.3 m/s, progress earns about 10 × 0.075 = 0.75 per decision, so a fall (−5) costs only about 7 decisions' worth. Add a small action-switching penalty, consider penalising roll and pitch *rates*, and bootstrap on time-outs. Data volume also matters before any conclusion is drawn: a 20 s episode at 4 Hz produces only **80 transitions**, so short probes say little. Keep 4 Hz. With phase locking it is a reasonable rate for choosing a gait mode, even though nobody has published results at that rate. Do not add an ARS comparator: the project already retired it because D2-GMBC covers linear ARS.

| Recommendation card | |
|---|---|
| Technique | Calibrate terrain difficulty against open loop; phase-locked decisions with a phase bit; half-cycle IMU averaging; persistent or factored discrete actions; reward-scale fixes |
| Evidence | D2-GMBC and ETG on headroom; universal phase observation in PMTG, D2-GMBC, Lee and CPG-RL; Rudin and Lee on curricula and time-outs |
| Cost | Table ×2 for the phase bit; no change to the learners |
| Fit with core | Same discrete adjustments on top of a CPG, chosen from IMU state; contacts are the only optional break from that |
| Do this | In Stage 7 only: run the roughness sweep first and accept the result even if the learners still do not win, since that is a reportable finding about headroom |

## Five seeds and hand-tuning cannot separate three learners

### What the sources show

Henderson et al. split 10 identical TRPO runs into two groups of 5 and obtained **significantly different curves (p = 0.0016)** ([Henderson et al. 2018](https://ar5iv.labs.arxiv.org/html/1709.06560)). Patterson et al. state that "5 runs is insufficient to make strong claims". They also recommend putting an interval on the **paired per-run difference** rather than comparing overlapping intervals, giving every algorithm the same number of tested hyperparameter settings, selecting on one set of seeds and reporting on fresh ones, budgeting by **environment steps rather than episodes**, and keeping IQM for comparisons across many environments, because within one environment it can hide failures ([Patterson et al. 2024](https://arxiv.org/pdf/2304.01315)). Colas et al. relate seed count to effect size: about 5-10 seeds for very large effects, about 20 for a relative effect of 1, and about 100 for 0.5. They recommend Welch's t-test, at a strict α for small N, and advise against bootstrap tests below N = 50 for a single task ([Colas et al. 2019](https://arxiv.org/pdf/1904.06979v1.pdf); [Colas et al. 2018](https://ar5iv.labs.arxiv.org/html/1806.08295)). rliable's percentile bootstrap intervals work at N = 10 when pooled over 26 games, but **under-cover with 3 runs** ([Agarwal et al. 2021](https://ar5iv.labs.arxiv.org/html/2108.13264); [rliable](https://pypi.org/project/rliable)). Adkins et al. define hyperparameter sensitivity as the gap between per-environment-tuned and cross-environment-tuned scores ([Adkins et al. 2024](https://arxiv.org/html/2412.07165v1)). Jordan et al. instead sample hyperparameters from a fixed distribution per algorithm ([Jordan et al. 2020](https://ar5iv.labs.arxiv.org/html/2006.16958)). The continual-RL literature has **no standard recovery-time metric**. It uses post-shift AUC, forward transfer relative to a from-scratch learner, forgetting, and resource proxies reported alongside performance ([CRL survey 2025](https://arxiv.org/pdf/2506.21872)). Replay ratio, the number of updates per environment step, is the standard way to state compute per sample ([D'Oro et al. 2023](https://mlanthology.org/iclr/2023/doro2023iclr-sampleefficient)). The best-known defence of 5 seeds is compute: Abbas et al. say "Ideally, more runs would be used", because their experiments took a month ([Abbas et al. 2023](https://ar5iv.labs.arxiv.org/html/2303.07507)).

### What it means here

That compute excuse does not apply to Q-learners on pendulum and cart-pole written in C. Of everything in this report, the evaluation changes give the most credibility for the least engineering. An equal *episode* budget also builds in a bias. A cart-pole learner that balances for 500 steps per episode receives about 10× the data of one that falls at step 50. This is exactly decision D02 in the project log. The project's D01 confound, where the network gets a different pendulum input encoding from the table and linear learners, needs either fixing or a dedicated ablation. Ghiassian et al.'s finding that tile-coded inputs help networks ([Ghiassian et al. 2020](https://arxiv.org/pdf/2003.07417)) shows that input encoding alone can move a network's results.

| Item | Current | Recommended |
|---|---|---|
| Seeds | 5 | ≥ 10 everywhere; 20-30 at headline budgets and for adaptation; Go1 may use fewer if stated |
| Budget and x-axis | equal episodes | equal environment steps, plotted against steps; state replay ratio = 1 for streaming learners |
| Hyperparameters | hand-chosen | same number K (e.g. 20-30) of configurations per learner, tuning seeds ≠ reporting seeds; report two-stage tuned and fixed cross-task defaults |
| Pairing | fixed eval seeds | common random numbers for training too, so CIs can be put on paired differences A − B |
| Intervals | Student-t over 5 seeds | plot every seed; use t only if the per-seed scores look roughly Gaussian (Q-learners are often bimodal); bootstrap from N ≥ 20 |
| Final performance | 3 evals × 5 episodes | 10-20 episodes per eval and a longer tail window |
| Online return | not reported | report exploratory training return; a robot falls while learning |
| Compute | instructions/step, peak SRAM | add replay ratio; cycles only after a Nucleo DWT measurement |
| Cross-task summary | none | normalised IQM and probability of improvement as secondary plots only |

The 90%-recovery metric has four weaknesses. The post-shift minimum of a noisy curve is biased low. Evaluating every 50 episodes quantises recovery time to multiples of 50. When a learner barely drops, the denominator approaches zero. And a learner that recovers to a low plateau looks fast. The fix keeps the idea but anchors it to a fixed reference.

```
return
  ▲   pre-shift level P ─────────┐
  │   target = 0.9·P (fixed) ┄┄┄┄│┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ ← cross this, smoothed over k evals
  │                              │       .·''''''''   learner's own plateau (don't anchor to it)
  │                              ▼    .·'
  │                               \.·'            denser evals here, right after the shift
  │                                ↑ noisy minimum (don't anchor to it)
  └──────────────────────────────┼─────────────────► environment steps
                               shift
  Report: steps-to-target (runs that never reach it are censored, never dropped), fraction recovered,
          post-shift AUC over a fixed window, and forward transfer vs a learner trained from scratch.
```

| Recommendation card | |
|---|---|
| Technique | Step budget, ≥10-30 paired seeds, equal tuning budget with separate tuning seeds, paired-difference CIs, a reworked adaptation metric, online return |
| Evidence | Henderson, Colas, Patterson, Agarwal, Adkins, and the CRL survey: methodological consensus rather than one experiment |
| Cost | Laptop compute only |
| Fit with core | Same three learners and tasks; only the protocol changes |
| Do this | Fix the protocol before the first pendulum curve in Stage 2, because later results inherit it |

## Do the cheap measurements first and the Go1 redesign last

The order follows two rules. Changes that every later result inherits come first. And no change should cost a stage gate. The table maps each recommendation onto the project's existing stages, which start on 18 Nov 2026.

| Priority | Change | Stage | Effort | Why now |
|---|---|---|---|---|
| 1 | Step-based budget, ≥10 paired seeds, tuning/reporting seed split, equal K per learner | 1-2 | Low | Every curve depends on it; resolves D02 |
| 2 | Underflow counter in the shared update; stochastic rounding switch | 2 | Low | Turns a suspected problem into a number; affects all three learners |
| 3 | Remove 64-bit `/` and `%` from hot paths | 2-3 | Low | The cost is structural, not a matter of tuning |
| 4 | Tile-coding rewrite (cache, power-of-two hash, integer offsets, dense pendulum), measured before and after | 3 | Medium | Largest instruction-count gain; part of the planned rewrite anyway |
| 5 | Stream Q(λ) network variant + one-at-a-time ablation; old SGD + target + buffer kept as baseline | 3 | Medium-high | The only route to a stable network that fits in 128 KB; the Stage 3 gate allows reporting failure |
| 6 | Resolve D01 (same input encoding or a dedicated ablation) | 3 | Low | Removes a confound in the headline plot |
| 7 | Compressed FIFO buffer as the hybrid arm (stream + small buffer) | 4 | Medium | Gives Stage 4 a sharper question than "bigger buffer" |
| 8 | λ ∈ {0, 0.8} for all three learners, sparse traces for table/linear | 5 | Medium | Keeps the comparison about representation |
| 9 | Reworked adaptation metric + from-scratch reference; CBP for the network | 5 | Low-medium | Adaptation is a stated research axis |
| 10 | Go1: roughness sweep against open loop, then phase lock + phase bit, then action/reward changes | 7 | Medium | Optional stage; diagnose headroom before redesigning |

Several attractive ideas should be skipped. Adam and Adaptive ObGD cost one or two extra vectors per weight plus a square root per weight, and streaming evidence calls Adam unsuitable. IDBD, TIDBD and Autostep triple or quadruple the linear learner's memory. Fourier bases, FTA and learned sparse coding replace the core representations. PQN's parallel environments do not exist on a single-environment on-device learner. True online Sarsa(λ) changes the shared target. ARS was already retired. Int8 training with QAS, and int16 storage, are good future work but abandon Q16.16. Rates of 50-400 Hz conflict with a 4 Hz learner and are unnecessary once decisions are phase-locked. The two related-work corrections, the unconfirmed ESP32 claim and the Salih paper being a navigation paper, take five minutes and protect the novelty statement.

## Conclusion

These recommendations turn the project from three different learning systems into **one shared streaming core with a swappable representation**: normalise, compute features, take the max-target TD error, update a Watkins trace, bound the step, then round stochastically. Today the network differs from the other two in its stabilisers (target network, buffer), its memory profile (dominated by the buffer) and possibly its input encoding, so a win or loss on the SRAM axis cannot be attributed to the representation. After the changes, the network's peak SRAM drops into the same few-kilobyte range as the table and linear models. The headline plot then measures what the research question asks about, and the network loses the automatic memory penalty it carries today.

The changes also open three small, honest results that the reviewed literature lacks: whether stream Q(λ) works at all at 8-64 hidden units on classic control; how often Q16.16 updates vanish, and whether stochastic rounding brings them back, for tabular, linear and network TD learners; and how tile-coding performance degrades with hash load factor. None of these needs hardware, and each stands even if the answer is negative. The same holds for the Go1 task. "The learner helps only once the open-loop trot fails more than X% of the time" is a more useful finding than a forced win on easy terrain.
