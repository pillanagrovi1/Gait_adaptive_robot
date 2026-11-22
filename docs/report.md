# Learning on a microcontroller's memory budget: table, tile coding or tiny network?

Every number below comes from a table or line in [`results.md`](results.md), which names its commit, config and
archive. The rules that were fixed before the runs are in [`decisions.md`](decisions.md) (D49–D60).

## Summary

```
                     peak SRAM for a      final return        instructions per    after the rod
                     good result          (best mean)         learning step       got 50% heavier
 table               large                weakest             525-715             lowest post-shift AUC
 linear tile coding  medium (9-50 KB)     among the best      1,814-4,155         forgot on the way back
 streaming network   smallest (2-5 KB)    good, more varied   10,530-71,974       beat linear, also forgot
 replay network      19-31 KB (buffer)    among the best      24,426-159,046      (no registered comparison)
```

*The picture is descriptive. The tested statements are the ones below.*

- On cart-pole, linear tile coding beat the table on final return (+141.3 [+96.9, +185.7]) and AUC, at 4.3x the
  instructions per step.
- On cart-pole, the streaming network learned faster than linear tile coding (AUC +63.6 [+26.0, +101.1]) in a third of the
  memory, at 12x the instructions. The difference in final return is inconclusive.
- After a change of dynamics on the pendulum, the streaming network returned more than linear tile coding (+87
  [+67, +107]) in under a quarter of the memory, at 20x the instructions.
- 5 replay updates per step did not measurably speed up the table (AUC difference +25, 95% CI [-22, +72], 10 paired
  seeds) at 2.7x the instructions per step. Stage 4 stopped there, as registered.
- Stochastic rounding made no measurable difference, although a fifth to two thirds of the update products underflow.

## 1. Question

How do different reinforcement-learning representations (tabular, linear tile coding, tiny neural network) trade
learning performance, adaptation speed and compute cost against peak working memory, when learning online on a
Cortex-M4-class platform? A controlled comparison was not found in a limited search.

## 2. Setup

```
 laptop (x86-64)                                        Cortex-M4 emulated in QEMU
 MuJoCo task <--> libtinyrl.so  --> learning curves     the same C files --> peak SRAM, instructions per step
                       ^                                        ^
                       +------ checksum: must be identical -----+
```

**Learners.** Three representations share one Q-learning target, y = r + γ max Q(s′, ·):

- a table over a uniform grid;
- linear function approximation on hashed tile coding (Sutton & Barto 2018, §9.5.4);
- a network with one hidden ReLU layer, in two variants. The *replay* network uses SGD, a 2,000-transition
  compressed buffer and a target network. The *streaming* network keeps no samples: it uses eligibility traces, a
  step-size cap (ObGD) and LayerNorm, after the December 2024 version of Elsayed et al. (2024), plus stochastic
  rounding (Gupta et al. 2015). The September 2026 version of that paper replaced ObGD by another optimiser, which
  is not studied here. Observation normalisation, reward scaling by a running statistic and sparse initialisation
  from that recipe are not used (D23, D61).

**Fixed point.** All learner arithmetic is Q16.16 C with static allocation. Counters record update products that
round to zero (underflow) or clip (saturation). No measured saturation was above 0% in any experiment that recorded
it (replay ratio, cart-pole, adaptation, ablation).

**Memory.** Peak working memory = agent struct + arena + learner stack, measured on the M4 build by stack painting.
Code, environment and startup are excluded.

**What QEMU can and cannot say.** It gives memory and instruction counts. It gives no timing: no cycles, wait states,
pipeline or power. The emulated machine is an STM32F405 with the memory map of an STM32F446RE. The
processor-in-the-loop build for a Nucleo-F446RE has **never run on hardware**; processor-in-the-loop learning was
checked in QEMU only.

**Tasks.** Pendulum swing-up (2 or 3 state dimensions), cart-pole (4 dimensions, at most 500 steps) and, as an
extension, gait tuning on a simulated Unitree Go1.

## 3. Method

- **Budgets.** Equal environment steps and equal tuning effort: 20 sampled configurations per learner, 3 tuning
  seeds separate from the reporting seeds, one tuning point in the middle of each memory sweep (Patterson et al.
  2023). The pendulum was tuned at a third of its step budget, cart-pole at the full budget (D49).
- **Seeds.** 10 paired seeds, 20 for adaptation, fixed before the runs. None were added after seeing a result (D50),
  because few-run comparisons are easy to over-read (Henderson et al. 2017; Colas et al. 2018; Agarwal et al. 2021).
- **Differences.** Student-t 95% interval on per-seed differences. A difference counts only if the interval excludes
  0. Otherwise it is "inconclusive".
- **Registration.** Replay ratio, cart-pole, adaptation, ablation and Go1 comparisons were written down before their
  runs. The pendulum main comparison had already been seen, so its comparisons are labelled not pre-registered. The
  Stage 4 metric changed from steps-to-threshold to AUC before the run and after the no-replay pendulum results were
  seen. Both metrics give the same decision (D53).
- **Metrics.** Final = mean of the last 3 greedy evaluations. AUC = mean evaluation over training. Online = mean
  training return over the last 20% of episodes.

## 4. Results

### 4.1 Pendulum: return against SRAM

![pendulum](../results/figures/pendulum_main_tuned_headline.png)

- Replay network (27.2 KB) against linear tile coding (50.1 KB): final return inconclusive (−8.73 [−25.28, +7.82]). The
  network learned faster (AUC +40.9 [+20.6, +61.2]) at 68.5x the instructions per step. *Not pre-registered.*
- Streaming network (2.8 KB) against linear tile coding (4.0 KB): inconclusive on final return (+51.4 [−2.9, +105.8])
  and on AUC (+45.4 [−7.9, +98.8]), at 17.7x the instructions per step. *Not pre-registered.*
- Descriptively, the streaming networks hold the mean final-return frontier between 1.3 and 5.2 KB. Linear coding
  and the replay network hold it from 13.3 KB. The table is on it only at 1,008 B and 1,296 B.
- The table's mean return rises with its size (−948 ± 47 at 1.0 KB to −245 ± 70 at 12.2 KB in the table-only
  experiment), and it did not reach the usual solved level within 300k steps.
- The input encoding matters: with cos/sin the 21-bin table costs 111.8 KB instead of 6.0 KB.

### 4.2 Cart-pole: return against SRAM

![cart-pole](../results/figures/cartpole_main_tuned_headline.png)

| A − B | final return | AUC | instructions, A / B |
|---|---|---|---|
| linear 1,024 (9.4 KB) − table 8 bins (33.5 KB) | +141.3 [+96.9, +185.7], higher | +80.4 [+53.0, +107.8], higher | 4.3x |
| replay net 32 (29.1 KB) − linear 1,024 | −21.3 [−128.6, +86.0], inconclusive | −114.4 [−160.9, −67.9], lower | 28x |
| streaming net 32 (3.0 KB) − linear 1,024 | +13.5 [−37.2, +64.3], inconclusive | +63.6 [+26.0, +101.1], higher | 12x |

The table gains nothing visible from more memory here. At 12 bins it no longer fits in 128 KB.

### 4.3 Compute

| learner | pendulum, instructions per step | cart-pole, instructions per step |
|---|---|---|
| table | 525–644 | 708–715 |
| linear tile coding | 1,814–2,280 | 3,046–4,155 |
| streaming network, 8/16–64 hidden | 10,530–68,885 | 20,500–71,974 |
| replay network, 8/16–64 hidden | 24,426–149,541 | 47,571–159,046 |

These are QEMU instruction counts on a synthetic workload, not times. The linear learner's count depends on the path
the benchmark takes (one or two hashes per step, D62), so ratios against it are uncertain by that spread.

### 4.4 Adaptation: A → B → A

![adaptation](../results/figures/adaptation_pendulum_adaptation.png)

![both switches](../results/figures/adaptation_pendulum_curves.png)

| learner | peak SRAM | post-shift AUC | forward transfer | forgetting |
|---|---|---|---|---|
| table | 2,160 B | −840 ± 12 | +101.7 [+84.5, +118.9] | −106.8 [−223.4, +9.8], inconclusive |
| linear | 13,220 B | −484 ± 13 | +108.4 [+84.3, +132.5] | −164.6 [−226.6, −102.5], forgot |
| replay net | 25,068 B | −330 ± 20 | +182.7 [+157.6, +207.8] | −104.0 [−216.4, +8.4], inconclusive |
| streaming net | 3,048 B | −397 ± 16 | +341.6 [+310.7, +372.6] | −152.0 [−257.6, −46.5], forgot |
| streaming net + continual backprop | 3,304 B | −704 ± 62 | +233.0 [+161.4, +304.5] | +61.0 [−126.3, +248.2], inconclusive |

- Streaming network − linear, post-shift AUC: +86.8 [+66.9, +106.8], at 20x the instructions.
- Continual backprop (Dohare et al. 2024) at one untuned setting made the streaming network worse after the shift
  (−306.8 [−369.7, −244.0]). It was already degrading before the shift, so this is not a verdict on the method.
- Over the whole second A window no learner differed measurably from its pre-shift level.
- Forward transfer includes the fresh learner's exploration cost.

### 4.5 Replay ratio: a null result

5 replay updates per step did not measurably speed up learning (AUC difference +25, 95% CI [-22, +72], 10 paired seeds) at 2.7x the instructions per step.

Both arms learned from the same 1,000-transition buffer, so this compares n = 5 with n = 1, not replay with the plain
table. It does not show that replay is useless: the interval reaches +72. The linear and network arms were not run,
as the stop rule required.

### 4.6 Underflow and stochastic rounding

In the pendulum comparison the streaming networks round 24–51% of their update products to zero, the table at most
0.4%. The ablation switched stochastic rounding on and off within each learner:

| learner with traces | underflow without it | final, with − without | AUC, with − without |
|---|---|---|---|
| table | 64.5% | +12.3 [−36.0, +60.5] | −15.1 [−45.1, +15.0] |
| linear | 22.8% | −1.38 [−10.00, +7.25] | +2.23 [−8.71, +13.16] |
| network | 33.8% | −32.6 [−115.1, +49.8] | +16.0 [−33.2, +65.2] |

All six are inconclusive. The same experiment shows that bootstrapping is needed on this task: with γ = 0 the table
and linear learners end 820 and 1,006 below their Q-learning twins.

### 4.7 Reproducibility

190 runs repeated on a second x86-64 Linux machine were byte-identical. For every configuration measured on the M4
the host and M4 checksums are equal. This shows determinism, not correctness, and says nothing about ARM or macOS
hosts.

## 5. Limitations

- **No hardware.** No timing, cycles, energy or power. The Nucleo build never ran.
- **QEMU is not timing.** Instructions are not cycles, and the workload behind the counts is synthetic.
- **One machine type.** x86-64 Linux for learning, one emulated Cortex-M4 for measurement.
- **Tuning budget.** 20 draws, one tuning point per sweep. The pendulum was tuned on a third of the budget and on
  one encoding. Several winners sit on the edge of their search space, and the streaming network's 20 draws covered
  only 9 distinct settings. Replay ratio and adaptation reuse the pendulum's values; the ablation is untuned.
- **Small scope.** Two control tasks for the core question, one shift size, one replay table and buffer size.
- **Statistical power.** With 10 seeds many intervals are tens of return units wide. Inconclusive does not mean
  equal. No correction for multiple comparisons was made.
- **Ceiling.** Cart-pole's 500-step limit compresses the differences between good learners.
- **Measurement detail.** The linear learner's instruction count is path dependent (D62). The flash figures are for
  a firmware that contains all three learners.

## 6. Related work

Elsayed et al. (2024) showed that deep RL can learn from a stream, without a buffer, by combining normalisation,
representation stabilisation and controlled updates. Their continuous-control benchmark networks are larger (for
example two layers of 128 units). This project tested a reduced form of their earlier recipe at 8–64 units in fixed
point. Gallici et al. (2024) give theory and evidence
that LayerNorm can replace target networks in temporal-difference learning. Sutton & Barto (2018) describe hashing
as a way to free tile coding's memory from the state dimension, which is the memory knob used here. Gupta et al.
(2015) trained networks in 16-bit fixed point with stochastic rounding; here, at 32 bits, rounding made no measurable
difference. Dohare et al. (2024) introduced continual backpropagation against loss of plasticity. RLtools (Eschmann
et al. 2023) reports training deep RL directly on a microcontroller. For legged robots, CPG-RL (Bellegarda & Ijspeert
2022) lets a policy modulate oscillators and not joints, which is the structure of the Go1 extension here.

A controlled comparison of these three representations on return against peak SRAM was not found in a limited
search.

## 7. Conclusion

On these two tasks the choice is a trade between bytes and instructions, and no learner is best at both.

- If **SRAM** binds, the streaming network is the candidate: 2–5 KB, good returns, better than linear tile coding
  after a change of dynamics. It pays 12x to 20x linear's instructions and varies more between seeds.
- If **compute** binds, linear tile coding is the candidate: a few thousand instructions per step and among the best
  returns, for roughly 9–50 KB.
- The **table** is the cheapest per step, and was beaten by linear coding on cart-pole even with more memory.
- **More replay updates** and **stochastic rounding** did not show a measurable benefit where they were tested.

Whether the instruction counts fit a real control period is a question for the board, which this project has not
answered.

## References

Only sources that were opened for this report are listed. For those marked (abstract) only the abstract was read.

- Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* arXiv:2108.13264. (abstract)
- Bellegarda, G. & Ijspeert, A. (2022). *CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion.* arXiv:2211.00458. (abstract)
- Colas, C., Sigaud, O. & Oudeyer, P.-Y. (2018). *How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.* arXiv:1806.08295. (abstract)
- Dohare, S. et al. (2024). *Loss of plasticity in deep continual learning.* Nature, doi:10.1038/s41586-024-07711-7. (abstract)
- Elsayed, M. et al. (2024). *Streaming Deep Reinforcement Learning Finally Works.* arXiv:2410.14606 (v3 read in part: method and Appendix C, which describes the earlier version).
- Eschmann, J., Albani, D. & Loianno, G. (2023). *RLtools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control.* arXiv:2306.03530. (abstract)
- Gallici, M. et al. (2024). *Simplifying Deep Temporal Difference Learning.* arXiv:2407.04811. (abstract)
- Gupta, S. et al. (2015). *Deep Learning with Limited Numerical Precision.* ICML, PMLR 37. (abstract)
- Henderson, P. et al. (2017). *Deep Reinforcement Learning that Matters.* arXiv:1709.06560. (abstract)
- Patterson, A. et al. (2023). *Empirical Design in Reinforcement Learning.* arXiv:2304.01315. (abstract)
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed., §9.5.4.
