# Under the hood: the guarantees on problems with exact solutions

These notebooks are for readers who want to see the results behind the toolbox at work. They use the test problems
of the paper *The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization*, which are small
enough to be solved exactly, so that every claim can be checked against the exact answer. The notebooks of the four
RL-level problems, one folder up, are the place to start for applying the method.

| Notebook | What it shows | Needs | Time |
|---|---|---|---|
| [T1. The price on a problem with an exact solution](T1_price_and_exact_frontier.ipynb) | the clipped reward and the identity behind the results; candidates against the exact trade-off of utility and risk; validation; the report | numpy | 20 s |
| [T2. Training a learner against an exact benchmark](T2_training_against_an_exact_benchmark.ipynb) | PPO through the wrapper; the optimization error measured; training at several prices; an adaptive multiplier and its diagnosis; the interior barrier under a misjudged allowance | RL extras | 11 min |
| [T3. What the regularizers do](T3_roles_of_the_regularizers.ipynb) | action and parameter regularization; noise, pull and spread on a solvable learning problem; the prior is a policy; parameter noise on PPO | RL extras | 12 min |
| [T4. Validation and the report in detail](T4_validation_in_detail.ipynb) | resolution floors; allocating the error before selecting; the geometric clock against a fixed horizon; when certificates from the value are informative; a lump sanction with your own evaluator | numpy | 1 min |

The [decisions and the theory behind them](../../docs/under-the-hood/decisions-and-theory.md) go through the
practitioner guide's steps with the paper's results, and the [results map](../../docs/under-the-hood/results-map.md)
lists each result by its title with the function that implements it.
