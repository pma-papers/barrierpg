# Technical tutorials

These notebooks show how the toolbox implements the objectives, optimizers and validation calculations. The live PPO comparisons use tutorial budgets and illustrate the workflow. See the [worked examples from the manuscript](../manuscript/) for the reported campaigns and their figures, and the [additional workflow examples](../../README.md#additional-workflow-examples) for fuel, cash, battery and navigation tasks.

| Notebook | Calculation or comparison | Dependencies |
|---|---|---|
| [T1. Fixed failure prices and the wear benchmark](T1_price_and_exact_frontier.ipynb) | Clipped-value decomposition, a grid-based policy comparison, and validation under geometric maintenance access | Core and Matplotlib |
| [T2. PPO and adaptive pricing on the wear problem](T2_training_against_an_exact_benchmark.ipynb) | Four-million-interaction PPO runs at several prices, adaptive failure pricing, and a sensitivity test at a reduced wear allowance | RL and notebook extras |
| [T3. Action and parameter regularization](T3_roles_of_the_regularizers.ipynb) | Noise and drift in the population model, prior-policy evaluation, and three-million-interaction PPO comparisons | RL and notebook extras |
| [T4. Statistical validation at fixed and geometric horizons](T4_validation_in_detail.ipynb) | Exact binomial endpoints, error allocation, and a value certificate for a fixed physical horizon | Core and Matplotlib |

Each notebook states the objective or constraint it uses and explains the calculation beside the corresponding cells. The algorithm guide develops the PPO, CPO and CVaR updates from their inputs. [Algorithm guide](../../docs/learning-algorithms.md)

The theory guide provides the conditions behind these calculations. For an exact solution of a known finite transition model, see the dynamic programming guide. [Theory and implementation](../../docs/under-the-hood/decisions-and-theory.md) · [Results and functions](../../docs/under-the-hood/results-map.md) · [Exact dynamic programming](../../docs/dynamic-programming.md)
