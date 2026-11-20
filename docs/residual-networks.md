# Residual networks for vector observations

`ResidualMLP` uses a linear input projection and residual feed-forward blocks without an attention layer.
In each block, the branch applies LayerNorm, a linear map, SiLU and a second linear map. The skip connection
adds the branch output to the unnormalized input. This placement gives gradients a direct path through the
block. Pre-normalization is used in Transformers, and residual networks with LayerNorm have also been used
to scale critics in continuous-control RL. These studies motivate testing the design with a given learner and
task. [Xiong et al. (2020)](https://proceedings.mlr.press/v119/xiong20b.html) ·
[Nauman et al. (2024)](https://proceedings.nips.cc/paper_files/paper/2024/hash/cd3b5d2ed967e906af24b33d6a356cac-Abstract-Conference.html)

## Using PPO

```python
from stable_baselines3 import PPO
from barrierpg import Stage
from barrierpg.examples.wear import make_wear_env
from barrierpg.sb3_policies import ResidualActorCriticPolicy

env = make_wear_env(Stage(m=0.3), geometric_stop=True)
model = PPO(
    ResidualActorCriticPolicy,
    env,
    gamma=1.0,  # The stopping clock implements the task discount.
    policy_kwargs=dict(width=32, depth=2),
)
model.learn(32_768)
```

The actor and critic have separate backbones. Each takes the current observation, including the reserve
reported by the wrapper. With the default flattening of observations, the toolbox's actor parameter
regularizers include the projection, all feed-forward layers and LayerNorm parameters, and the action head.
Set `residual=False` for a matched-parameter comparison that removes the skips. Set `layer_norm=False`
to test the contribution of normalization separately.

## Using CPO or Gaussian CVaR

Use `barrierpg.networks.ResidualMLP` as the body of a PyTorch actor. Its head should return a distribution
when called on a batch of observations, as in the [native actor-update example](../examples/train_risk_constrained.py).
The toolbox's CPO and Gaussian CVaR steps use a dense KL Hessian, so keep the actor small when using these
implementations. For larger actors, use a solver designed around Hessian-vector products.

The paper's direct risk checks apply to the deployed policy under their stated sampling assumptions. Applying
the population learning theorem to a new actor family requires verifying its representation and derivative assumptions.

## Comparing learning curves

Keep the budget formulation, learner and number of environment interactions fixed. Compare stopped utility,
failure frequency and the training objective on separate evaluation rollouts at matched checkpoints. The
`residual=False` comparison isolates the skip connection when LayerNorm is held fixed. If runtime matters
to the application, also compare wall-clock time on the same hardware.

```bash
python examples/benchmark_residual_wear.py --output /tmp/residual-wear.json
```

The pilot uses the wear allowance with failure price `m=0.3`, four paired training seeds, and
600 evaluation episodes per policy at each checkpoint. Both architectures have 4,514 actor
parameters and 4,513 critic parameters. Both use LayerNorm, and the policy and value heads start
at zero. The experiment toggles the skip connections in both networks together. The table gives
averages across the four seeds. Breakdown risk is the probability of exhausting the allowance
before maintenance, and priced value is stopped utility minus six times that risk.

| Training steps | Utility, no skips | Utility, skips | Risk, no skips | Risk, skips | Value, no skips | Value, skips |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 3.205 | 3.260 | 10.63% | 10.71% | 2.568 | 2.618 |
| 16,384 | 3.531 | 3.656 | 7.54% | 6.83% | 3.078 | 3.246 |
| 32,768 | 4.136 | 4.134 | 6.58% | 6.00% | 3.741 | 3.774 |

The estimated value was higher with skips for all four seeds at 16,384 steps and three of four
at 32,768 steps. The largest mean difference was at the middle checkpoint. The pilot measures
progress per environment interaction for this PPO setup. LayerNorm was held fixed, and runtime
was outside the scope of the comparison. [Run records](../examples/residual_wear_comparison.json)
