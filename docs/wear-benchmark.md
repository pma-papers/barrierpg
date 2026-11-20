# Paired pilot on the wear-allowance model

An asset earns utility from its operating rate and fails if accumulated wear exceeds its allowance before independent maintenance access. We compare fixed failure prices and expected-cost constraints using this same geometric breakdown event. We also vary the interior loss in the reward and compare a conditional value-at-risk (CVaR) constraint on the size of the overshoot. Every run uses a categorical policy over three operating rates.

```bash
python examples/benchmark_wear.py --output /tmp/wear-comparison.json
```

## Model, objectives and updates

An action sets an operating rate $a_t$ from the three rates in the example. The actor uses the price index $s_t$ and normalized reserve as inputs to categorical logits. The index follows $s_{t+1}=0.8s_t+0.2W_t$, with $W_t$ uniform on $[-1,1]$. The asset earns $u_t=(1+0.5s_t)a_t/1.5$ before breakdown and consumes $0.1a_t X_t$, with a gamma shock $X_t$ of shape $0.5$ and scale $2$. The reserve starts at one and failure occurs at its first strictly negative update. Maintenance access occurs independently with probability $0.05$ after a nonfailing action.

Let $U(\pi)$ be mean utility earned before maintenance or breakdown, $D$ the flag for breakdown before maintenance, and $Z$ the loss equal to overshoot on breakdown and zero otherwise. Let $\mathcal I_{\rm clip}(\pi)$ be the mean interior deduction computed at the shared clipping level $m_{\rm clip}=0.30$. The fixed-price runs maximize $U(\pi)-\mathcal I_{\rm clip}(\pi)-m\mathbb E_\pi[D]/(1-\beta)$, with $\beta=0.95$. The CPO runs maximize $U(\pi)-\mathcal I_{\rm clip}(\pi)$ subject to $\mathbb E_\pi[D]\le d$. When the interior barrier is disabled, the objective uses $\mathcal I_{\rm clip}=0$. The CVaR run maximizes $U(\pi)$ subject to a Gaussian approximation of $\mathrm{CVaR}_{0.20}(Z)\le0.16$ during training. Evaluation computes CVaR directly from observed overshoots. These objectives differ in which loss or event they penalize or constrain.

At each update, the pilot estimates utility from remaining episode rewards. The cost statistic is measured once per episode. In the native actor update its cost correction is averaged per decision and scaled by the ratio of decisions to episodes. CPO proposes an update with a local cost limit and a bound on mean KL from the current actor. For CVaR, the cost gradient is replaced by the gradient of the Gaussian tail surrogate. The fixed-price configurations use a reward advantage with the cost penalty included and the reward-only trust-region step. [Actor surrogates and the quadratic subproblem](learning-algorithms.md#cpo-with-separate-reward-and-cost)

## Design

Each setting uses six seeds, the same initial actor and 18 updates with 768 complete episodes per update. The initial policy's measured breakdown probability is 6.7–7.1%. Within each seed, methods share the sequences of wear shocks, price shocks, maintenance times and action uniforms. Ten thousand fresh episodes per seed are used for evaluation, with common random numbers within each evaluation pair. The fixed-price runs use the unconstrained step of the same trust-region solver. Matching the episode count leaves different numbers of decisions, which are reported.

The cost in the expected-cost formulation is the indicator of a breakdown before independent maintenance access. Its mean is the geometric risk used in the fixed-price objective. The log-barrier runs share the interior loss at strength 0.10 and clipping price 0.30. In the CPO variant the objective is stopped utility minus this fixed interior loss. In the priced variants there is an additional fixed failure charge. At price 0.45 the clip is held at 0.30 so the interior loss stays the same. The CVaR formulation limits the average of the worst 20% of overshoots through a Gaussian approximation, where overshoot is the amount by which wear exceeds the allowance on the failing decision and is zero on all other episodes.

The CPO cap of 0.08 and the failure price of 0.45 were added after the nominal-cap pilot. The reported evaluation stream was generated separately after these sensitivity settings were chosen.

## Results

The table reports means across six fitted policies per configuration, each evaluated separately. The range in the failure column is the range of seed estimates. The low-reserve frequency includes breakdowns and measures whether the remaining allowance ever fell to 5% or less. Overshoot CVaR is computed directly from the evaluation episodes for every candidate.

| Training configuration | Stopped utility | Utility net of the common interior loss | Failure probability (range) | Low-reserve frequency | Overshoot CVaR | Training decisions |
|---|---:|---:|---:|---:|---:|---:|
| Fixed price 0.10 | 4.508 | 2.533 | 0.306 (0.273–0.336) | 0.346 | 0.096 | 228,208 |
| Fixed price 0.30 | 3.864 | 1.417 | 0.097 (0.070–0.116) | 0.125 | 0.016 | 254,530 |
| Fixed price 1.00 | 2.826 | 0.694 | 0.030 (0.015–0.049) | 0.040 | 0.003 | 269,169 |
| CPO, cap 0.10 | 2.949 | 1.009 | 0.122 (0.116–0.132) | 0.140 | 0.033 | 248,943 |
| Log barrier, price 0.30 | 3.509 | 1.307 | 0.115 (0.065–0.163) | 0.140 | 0.025 | 249,715 |
| CPO and log barrier, cap 0.10 | 2.728 | 0.879 | 0.120 (0.110–0.127) | 0.135 | 0.038 | 247,857 |
| Gaussian overshoot CVaR, cap 0.16 | 4.630 | 3.030 | 0.396 (0.377–0.408) | 0.423 | 0.156 | 218,022 |
| CPO, cap 0.08 | 2.670 | 0.757 | 0.097 (0.083–0.106) | 0.112 | 0.026 | 253,140 |
| CPO and log barrier, cap 0.08 | 2.576 | 0.713 | 0.099 (0.087–0.113) | 0.112 | 0.029 | 251,538 |
| Log barrier, price 0.45 | 2.993 | 0.819 | 0.047 (0.028–0.065) | 0.061 | 0.007 | 263,128 |

For both CPO variants trained at cap 0.10, the measured breakdown frequency exceeded 0.10 on each of the six evaluation seeds. Lowering the training cap reduced the observed breakdown probabilities. For each CPO variant at the lower cap, two seeds had individual one-sided 95% binomial upper bounds below 0.10. The log-barrier run at price 0.45 had higher stopped utility and lower failure risk than the CPO log-barrier runs in this pilot. The overshoot-CVaR configuration had a mean tail cost close to its cap and a mean breakdown frequency of about 40%. An overshoot cap and a breakdown-probability cap impose different deployment criteria. These are comparisons for this particular actor, optimizer and sample budget.

The [per-seed results](../examples/wear_comparison.json) include the fitted parameters, individual binomial upper bounds, number of accepted updates and training and evaluation decision counts. The individual bounds are suitable for examining one fixed candidate. Simultaneous certification across candidates requires a separate error allocation.
