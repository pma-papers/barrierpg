# Under the hood: benchmark problems and validation

These notebooks are for readers who want to see the results behind the toolbox at work. They use the test problems
of the paper *The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization*. The wear
examples use a grid-based benchmark. The regularizer notebook also uses an analytically tractable policy model.
The four applied notebooks one folder up introduce the workflow.

| Notebook | What it shows | Needs | Time |
|---|---|---|---|
| [T1. The price on a problem with a grid benchmark](T1_price_and_exact_frontier.ipynb) | the clipped-return identity, candidate policies against the benchmark, and validation | NumPy, SciPy, Matplotlib | 20 s |
| [T2. Training a learner against a grid benchmark](T2_training_against_an_exact_benchmark.ipynb) | PPO at several prices, descriptive benchmark gaps, multiplier diagnostics, and a smaller-allowance stress test | RL extras | 11 min |
| [T3. What the regularizers do](T3_roles_of_the_regularizers.ipynb) | action and parameter regularization; noise, drift and spread on a solvable learning problem; the prior is a policy; parameter noise on PPO | RL extras | 12 min |
| [T4. Validation and the report in detail](T4_validation_in_detail.ipynb) | what zero failures certify, allocation of validation error, geometric and fixed horizons, and a lump sanction with a custom evaluator | NumPy, SciPy, Matplotlib | 1 min |

The [decisions and the theory behind them](../../docs/under-the-hood/decisions-and-theory.md) go through the
practitioner guide's steps with the paper's results, and the [results map](../../docs/under-the-hood/results-map.md)
lists each result by its title with the function that implements it.
